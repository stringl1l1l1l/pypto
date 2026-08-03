/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file backend_cce_ops.cpp
 * \brief Backend op registration for BackendCCE
 *
 * This file registers all block operations for the CCE backend.
 * Each registration specifies the pipe type and CCE codegen function.
 */

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "backend/backend_cce.h"
#include "backend/common/backend.h"
#include "backend/common/backend_utils.h"
#include "ir/op_attr_types.h"
#include "codegen/cce/cce_codegen.h"
#include "codegen/codegen_base.h"
#include "core/error.h"
#include "core/logging.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/pipe.h"
#include "ir/type.h"
#include "tilefwk/error.h"

namespace pypto {
namespace backend {
using ir::DataType;

// ============================================================================
// Helper Functions for CCE Code Generation
// ============================================================================

static std::string BuildStaticNZShapeType(const DataType& dtype, int64_t rows, int64_t cols)
{
    const int64_t c0 = cce::GetNZInnerCols(dtype);
    CHECK(c0 > 0) << "NZ C0 size must be positive";
    return "pto::Shape<1, " + std::to_string(cols / c0) + ", " + std::to_string(rows / 16) + ", 16, " +
           std::to_string(c0) + ">";
}

static std::string BuildStaticNZStrideType(const DataType& dtype, int64_t full_rows, int64_t full_cols)
{
    const int64_t c0 = cce::GetNZInnerCols(dtype);
    return "pto::Stride<" + std::to_string(full_rows * full_cols) + ", " + std::to_string(full_rows * c0) + ", " +
           std::to_string(16 * c0) + ", " + std::to_string(c0) + ", 1>";
}

static int64_t ComputeStaticNZPhysicalOffset(const DataType& dtype, int64_t full_rows, int64_t row_offset,
                                             int64_t col_offset)
{
    const int64_t c0 = cce::GetNZInnerCols(dtype);
    return col_offset * full_rows + row_offset * c0;
}

static void ValidateDebugDumpNZWindowStructure(const ir::TensorTypePtr& tensor_type, const ir::MakeTuplePtr& offsets,
                                               const ir::MakeTuplePtr& shapes)
{
    CHECK(tensor_type != nullptr) << "debug.dump_tensor NZ lowering requires TensorType";
    CHECK(offsets != nullptr) << "debug.dump_tensor NZ lowering requires offsets tuple";
    CHECK(shapes != nullptr) << "debug.dump_tensor NZ lowering requires shapes tuple";

    if (tensor_type->shape_.size() != 2 || offsets->elements_.size() != 2 || shapes->elements_.size() != 2) {
        throw pypto::ir::ValueError(
            "debug.dump_tensor: CCE NZ dump currently requires a 2D tensor and 2D offsets/shapes");
    }
}

static bool IsStaticDebugDumpNZWindow(const ir::TensorTypePtr& tensor_type, const ir::MakeTuplePtr& offsets,
                                      const ir::MakeTuplePtr& shapes)
{
    return tensor_type && offsets && shapes && ir::As<ir::ConstInt>(tensor_type->shape_[0]) &&
           ir::As<ir::ConstInt>(tensor_type->shape_[1]) && ir::As<ir::ConstInt>(offsets->elements_[0]) &&
           ir::As<ir::ConstInt>(offsets->elements_[1]) && ir::As<ir::ConstInt>(shapes->elements_[0]) &&
           ir::As<ir::ConstInt>(shapes->elements_[1]);
}

static bool CanUseStaticDebugDumpNZFastPath(const ir::TensorTypePtr& tensor_type, const ir::MakeTuplePtr& offsets,
                                            const ir::MakeTuplePtr& shapes)
{
    if (!IsStaticDebugDumpNZWindow(tensor_type, offsets, shapes)) {
        return false;
    }
    const int64_t full_rows = ir::As<ir::ConstInt>(tensor_type->shape_[0])->value_;
    const int64_t full_cols = ir::As<ir::ConstInt>(tensor_type->shape_[1])->value_;
    const int64_t row_offset = ir::As<ir::ConstInt>(offsets->elements_[0])->value_;
    const int64_t col_offset = ir::As<ir::ConstInt>(offsets->elements_[1])->value_;
    const int64_t rows = ir::As<ir::ConstInt>(shapes->elements_[0])->value_;
    const int64_t cols = ir::As<ir::ConstInt>(shapes->elements_[1])->value_;
    const int64_t c0 = cce::GetNZInnerCols(tensor_type->dtype_);
    CHECK(c0 > 0) << "NZ C0 size must be positive";
    if (full_rows % 16 != 0 || full_cols % c0 != 0) {
        return false;
    }
    if (row_offset < 0 || col_offset < 0 || rows <= 0 || cols <= 0) {
        return false;
    }
    if (row_offset % 16 != 0 || rows % 16 != 0 || col_offset % c0 != 0 || cols % c0 != 0) {
        return false;
    }
    return row_offset + rows <= full_rows && col_offset + cols <= full_cols;
}

static std::string AddStartOffsetIfNeeded(const std::string& start_offset, const std::string& physical_offset)
{
    if (start_offset.empty()) {
        return physical_offset;
    }
    return "(" + start_offset + " + " + physical_offset + ")";
}

static int NextDebugDumpId()
{
    static int next_debug_dump_id = 0;
    return next_debug_dump_id++;
}

static std::string JoinExpressions(const std::vector<std::string>& expressions, const std::string& delimiter)
{
    std::ostringstream oss;
    for (size_t i = 0; i < expressions.size(); ++i) {
        if (i > 0)
            oss << delimiter;
        oss << expressions[i];
    }
    return oss.str();
}

static void EmitDebugLocationHeaderCCE(codegen::CCECodegen& codegen, const ir::Span& span, const std::string& op_name)
{
    std::string header = debug_printf::FormatDebugLocationHeader(span, op_name);
    if (!header.empty()) {
        codegen.Emit("cce::printf(\"" + debug_printf::EscapeStringLiteral(header + "\n") + "\");");
    }
}

static bool NeedsCcePrintfSignedLongLong(const DataType& dtype, char conversion)
{
    return (conversion == 'd' || conversion == 'i') && (dtype == DataType::INT64 || dtype == DataType::INDEX);
}

static bool NeedsCcePrintfUnsignedU64Helper(const DataType& dtype, char conversion)
{
    return (conversion == 'u' || conversion == 'x') && (dtype == DataType::UINT64 || dtype == DataType::INDEX);
}

static std::string RewriteCcePrintfFormatForScalarType(const std::string& format_segment, char conversion,
                                                       const DataType& dtype)
{
    if (!NeedsCcePrintfSignedLongLong(dtype, conversion) && conversion != 'p') {
        return format_segment;
    }

    size_t conv_idx = debug_printf::FindPrintfConversionIndex(format_segment);
    std::string rewritten = format_segment;
    rewritten.replace(conv_idx, 1, (conversion == 'd' || conversion == 'p') ? "lld" : "lli");
    return rewritten;
}

static std::string CastCcePrintfArgIfNeeded(const std::string& arg, const DataType& dtype, char conversion)
{
    if (conversion == 'f') {
        return arg;
    }
    if (conversion == 'p') {
        return "static_cast<long long>((uint64_t)" + arg + ")";
    }
    if (NeedsCcePrintfSignedLongLong(dtype, conversion)) {
        return "static_cast<long long>(" + arg + ")";
    }
    if (dtype == DataType::BOOL) {
        return conversion == 'u' ? "static_cast<unsigned int>(" + arg + ")" : "static_cast<int>(" + arg + ")";
    }
    if (dtype == DataType::INT8 || dtype == DataType::INT16) {
        return "static_cast<int>(" + arg + ")";
    }
    if (dtype == DataType::UINT8 || dtype == DataType::UINT16) {
        return "static_cast<unsigned int>(" + arg + ")";
    }
    return arg;
}

static void AppendCcePrintfCall(std::vector<std::string>* statements, const std::string& format,
                                const std::vector<std::string>& args = {})
{
    if (format.empty()) {
        return;
    }
    std::string statement = "cce::printf(\"" + debug_printf::EscapeStringLiteral(format) + "\"";
    for (const auto& arg : args) {
        statement += ", " + arg;
    }
    statement += ");";
    statements->push_back(statement);
}

static bool CcePrintfSpecHasFlag(const std::string& conversion_spec, char flag)
{
    return conversion_spec.find(flag) != std::string::npos;
}

static void AppendCcePrintfUnsignedDecimalU64(std::vector<std::string>* statements, const std::string& arg,
                                              int* temp_id)
{
    const std::string suffix = std::to_string((*temp_id)++);
    const std::string value = "__pypto_printf_u64_value_" + suffix;
    const std::string low = "__pypto_printf_u64_low_" + suffix;
    const std::string mid = "__pypto_printf_u64_mid_" + suffix;
    const std::string high = "__pypto_printf_u64_high_" + suffix;
    const std::string rest = "__pypto_printf_u64_rest_" + suffix;

    statements->push_back("{");
    statements->push_back("  uint64_t " + value + " = static_cast<uint64_t>(" + arg + ");");
    statements->push_back("  unsigned int " + low + " = static_cast<unsigned int>(" + value + " % 1000000000ULL);");
    statements->push_back("  uint64_t " + rest + " = " + value + " / 1000000000ULL;");
    statements->push_back("  unsigned int " + mid + " = static_cast<unsigned int>(" + rest + " % 1000000000ULL);");
    statements->push_back("  unsigned int " + high + " = static_cast<unsigned int>(" + rest + " / 1000000000ULL);");
    statements->push_back("  if (" + high + " != 0U) {");
    statements->push_back("    cce::printf(\"%u%09u%09u\", " + high + ", " + mid + ", " + low + ");");
    statements->push_back("  } else if (" + mid + " != 0U) {");
    statements->push_back("    cce::printf(\"%u%09u\", " + mid + ", " + low + ");");
    statements->push_back("  } else {");
    statements->push_back("    cce::printf(\"%u\", " + low + ");");
    statements->push_back("  }");
    statements->push_back("}");
}

static void AppendCcePrintfUnsignedHexU64(std::vector<std::string>* statements, const std::string& arg,
                                          const std::string& conversion_spec, int* temp_id)
{
    const std::string suffix = std::to_string((*temp_id)++);
    const std::string value = "__pypto_printf_u64_value_" + suffix;
    const std::string high = "__pypto_printf_u64_high_" + suffix;
    const std::string low = "__pypto_printf_u64_low_" + suffix;
    const bool alternate = CcePrintfSpecHasFlag(conversion_spec, '#');

    statements->push_back("{");
    statements->push_back("  uint64_t " + value + " = static_cast<uint64_t>(" + arg + ");");
    statements->push_back("  unsigned int " + high + " = static_cast<unsigned int>(" + value + " >> 32);");
    statements->push_back("  unsigned int " + low + " = static_cast<unsigned int>(" + value + " & 0xffffffffULL);");
    statements->push_back("  if (" + high + " != 0U) {");
    if (alternate) {
        statements->push_back("    cce::printf(\"0x%x%08x\", " + high + ", " + low + ");");
    } else {
        statements->push_back("    cce::printf(\"%x%08x\", " + high + ", " + low + ");");
    }
    statements->push_back("  } else {");
    statements->push_back("    cce::printf(\"" + debug_printf::EscapeStringLiteral(conversion_spec) + "\", " + low +
                          ");");
    statements->push_back("  }");
    statements->push_back("}");
}

static std::vector<std::string> MakeCcePrintfStatements(const std::string& format, const std::vector<std::string>& args,
                                                        const std::vector<DataType>& arg_dtypes)
{
    CHECK(args.size() == arg_dtypes.size()) << "debug.printf CCE argument/type count mismatch";

    auto segments = debug_printf::ParsePrintfSegments(format);
    CHECK(segments.size() == args.size())
        << "debug.printf format expects " << segments.size() << " scalar arguments, but got " << args.size();

    std::vector<std::string> statements;
    int temp_id = 0;
    if (segments.empty()) {
        AppendCcePrintfCall(&statements, format);
        return statements;
    }

    // Merge consecutive non-U64 segments into a single printf call.
    // Flush the batch when hitting a U64 segment (which needs multi-statement expansion).
    std::string batch_format;
    std::vector<std::string> batch_args;

    auto flush_batch = [&]() {
        if (!batch_format.empty()) {
            AppendCcePrintfCall(&statements, batch_format, batch_args);
            batch_format.clear();
            batch_args.clear();
        }
    };

    for (size_t i = 0; i < segments.size(); ++i) {
        if (NeedsCcePrintfUnsignedU64Helper(arg_dtypes[i], segments[i].conversion)) {
            flush_batch();
            debug_printf::PrintfFormatParts parts = debug_printf::SplitPrintfSegment(segments[i].format_segment);
            AppendCcePrintfCall(&statements, parts.prefix);
            if (segments[i].conversion == 'u') {
                AppendCcePrintfUnsignedDecimalU64(&statements, args[i], &temp_id);
            } else {
                AppendCcePrintfUnsignedHexU64(&statements, args[i], parts.conversion_spec, &temp_id);
            }
            AppendCcePrintfCall(&statements, parts.suffix);
        } else {
            std::string rewritten_format = RewriteCcePrintfFormatForScalarType(segments[i].format_segment,
                                                                               segments[i].conversion, arg_dtypes[i]);
            std::string rewritten_arg = CastCcePrintfArgIfNeeded(args[i], arg_dtypes[i], segments[i].conversion);
            batch_format += rewritten_format;
            batch_args.push_back(rewritten_arg);
        }
    }
    flush_batch();

    return statements;
}

static bool HasDynamicTensorShape(const ir::TensorTypePtr& tensor_type)
{
    for (const auto& dim : tensor_type->shape_) {
        if (!ir::As<ir::ConstInt>(dim)) {
            return true;
        }
    }
    return false;
}

static bool IsFullTensorWindow(const ir::TensorTypePtr& tensor_type, const ir::MakeTuplePtr& offsets,
                               const ir::MakeTuplePtr& shapes)
{
    if (!tensor_type || !offsets || !shapes) {
        return false;
    }
    const size_t rank = tensor_type->shape_.size();
    if (offsets->elements_.size() != rank || shapes->elements_.size() != rank) {
        return false;
    }

    for (size_t i = 0; i < rank; ++i) {
        auto offset_const = ir::As<ir::ConstInt>(offsets->elements_[i]);
        if (!offset_const || offset_const->value_ != 0) {
            return false;
        }

        auto shape_const = ir::As<ir::ConstInt>(shapes->elements_[i]);
        auto tensor_dim_const = ir::As<ir::ConstInt>(tensor_type->shape_[i]);
        if (shape_const && tensor_dim_const) {
            if (shape_const->value_ != tensor_dim_const->value_) {
                return false;
            }
            continue;
        }
        if (shapes->elements_[i].get() != tensor_type->shape_[i].get()) {
            return false;
        }
    }

    return true;
}

static std::string GetRuntimeTensorShapeExpr(const std::string& tensor_name, size_t rank, size_t axis)
{
    const size_t gt_dim = 5 - rank + axis;
    return tensor_name + ".GetShape(GlobalTensorDim::DIM_" + std::to_string(gt_dim) + ")";
}

static std::string GetRuntimeTensorStrideExpr(const std::string& tensor_name, size_t rank, size_t axis)
{
    const size_t gt_dim = 5 - rank + axis;
    return tensor_name + ".GetStride(GlobalTensorDim::DIM_" + std::to_string(gt_dim) + ")";
}

static std::string BuildShapeTypeForDump(codegen::CCECodegen& codegen, const std::string& tensor_name,
                                         const ir::TensorTypePtr& tensor_type,
                                         const std::vector<ir::ExprPtr>& shape_exprs, bool use_runtime_full_shape,
                                         std::vector<std::string>* ctor_args)
{
    CHECK(shape_exprs.size() >= 1 && shape_exprs.size() <= 5)
        << "debug.dump_tensor currently supports tensor rank 1..5, but got " << shape_exprs.size();

    const size_t pad_dims = 5 - shape_exprs.size();
    std::vector<std::string> template_dims(5, "1");
    ctor_args->clear();
    for (size_t i = 0; i < shape_exprs.size(); ++i) {
        if (auto dim = ir::As<ir::ConstInt>(shape_exprs[i])) {
            template_dims[pad_dims + i] = std::to_string(dim->value_);
        } else {
            template_dims[pad_dims + i] = "-1";
            if (use_runtime_full_shape) {
                ctor_args->push_back(GetRuntimeTensorShapeExpr(tensor_name, tensor_type->shape_.size(), i));
            } else {
                ctor_args->push_back(codegen.GetExprAsCode(shape_exprs[i]));
            }
        }
    }
    return "pto::Shape<" + JoinExpressions(template_dims, ", ") + ">";
}

static void ComputeStridesFromShape(codegen::CCECodegen& codegen, const ir::TensorTypePtr& tensor_type, size_t rank,
                                    size_t pad_dims, std::vector<std::string>& stride_template_dims,
                                    std::vector<std::string>* ctor_args)
{
    for (size_t i = 0; i < rank; ++i) {
        bool all_const = true;
        int64_t const_stride = 1;
        std::vector<std::string> factors;
        for (size_t j = i + 1; j < rank; ++j) {
            if (auto dim = ir::As<ir::ConstInt>(tensor_type->shape_[j])) {
                const_stride *= dim->value_;
            } else {
                all_const = false;
                factors.push_back(codegen.GetExprAsCode(tensor_type->shape_[j]));
            }
        }
        if (all_const) {
            stride_template_dims[pad_dims + i] = std::to_string(const_stride);
        } else {
            std::string expr = std::to_string(const_stride);
            if (!factors.empty()) {
                expr += " * " + JoinExpressions(factors, " * ");
            }
            stride_template_dims[pad_dims + i] = "-1";
            ctor_args->push_back("(" + expr + ")");
        }
    }
}

static std::string BuildStrideTypeForDump(codegen::CCECodegen& codegen, const std::string& tensor_name,
                                          const ir::TensorTypePtr& tensor_type, bool use_runtime_tensor_view,
                                          std::vector<std::string>* ctor_args)
{
    CHECK(tensor_type) << "debug.dump_tensor requires TensorType for stride generation";
    const size_t rank = tensor_type->shape_.size();
    CHECK(rank >= 1 && rank <= 5) << "debug.dump_tensor currently supports tensor rank 1..5, but got " << rank;

    std::vector<std::string> stride_template_dims(5, "1");
    ctor_args->clear();
    const size_t pad_dims = 5 - rank;

    auto append_dynamic_stride = [&](size_t axis, const std::string& expr) {
        stride_template_dims[pad_dims + axis] = "-1";
        ctor_args->push_back(expr);
    };

    if (use_runtime_tensor_view) {
        for (size_t i = 0; i < rank; ++i) {
            append_dynamic_stride(i, GetRuntimeTensorStrideExpr(tensor_name, rank, i));
        }
        return "pto::Stride<" + JoinExpressions(stride_template_dims, ", ") + ">";
    }

    if (tensor_type->tensor_view_.has_value() && !tensor_type->tensor_view_->stride.empty()) {
        const auto& strides = tensor_type->tensor_view_->stride;
        CHECK(strides.size() == rank) << "debug.dump_tensor tensor_view stride rank (" << strides.size()
                                      << ") must match tensor rank (" << rank << ")";
        for (size_t i = 0; i < rank; ++i) {
            if (auto stride = ir::As<ir::ConstInt>(strides[i])) {
                stride_template_dims[pad_dims + i] = std::to_string(stride->value_);
            } else {
                append_dynamic_stride(i, codegen.GetExprAsCode(strides[i]));
            }
        }
        return "pto::Stride<" + JoinExpressions(stride_template_dims, ", ") + ">";
    }

    ComputeStridesFromShape(codegen, tensor_type, rank, pad_dims, stride_template_dims, ctor_args);
    return "pto::Stride<" + JoinExpressions(stride_template_dims, ", ") + ">";
}

static std::string ComputeRuntimeStrideBasedOffset(codegen::CCECodegen& codegen, const std::string& tensor_name,
                                                   const ir::TensorTypePtr& tensor_type,
                                                   const ir::MakeTuplePtr& offsets, const std::string& start_offset)
{
    CHECK(tensor_type) << "debug.dump_tensor requires TensorType for runtime offset generation";
    const size_t rank = tensor_type->shape_.size();
    CHECK(offsets) << "debug.dump_tensor requires offsets tuple for runtime offset generation";
    CHECK(offsets->elements_.size() == rank) << "debug.dump_tensor offset rank (" << offsets->elements_.size()
                                             << ") must match tensor rank (" << rank << ")";

    std::ostringstream offset_computation;
    offset_computation << "(";
    bool has_term = false;
    if (!start_offset.empty()) {
        offset_computation << start_offset;
        has_term = true;
    }

    for (size_t i = 0; i < rank; ++i) {
        if (has_term) {
            offset_computation << " + ";
        }
        offset_computation << codegen.GetExprAsCode(offsets->elements_[i]) << " * ";
        offset_computation << GetRuntimeTensorStrideExpr(tensor_name, rank, i);
        has_term = true;
    }

    if (!has_term) {
        offset_computation << "0";
    }
    offset_computation << ")";
    return offset_computation.str();
}

static std::string GetTensorLogicalDimForNZDump(codegen::CCECodegen& codegen, const ir::TensorTypePtr& tensor_type,
                                                size_t axis)
{
    CHECK(tensor_type) << "debug.dump_tensor NZ lowering requires TensorType";
    CHECK(axis < tensor_type->shape_.size()) << "debug.dump_tensor NZ axis out of range";
    if (auto dim = ir::As<ir::ConstInt>(tensor_type->shape_[axis])) {
        return std::to_string(dim->value_);
    }
    return codegen.GetExprAsCode(tensor_type->shape_[axis]);
}

static std::string GetExprForNZDump(codegen::CCECodegen& codegen, const ir::ExprPtr& expr)
{
    if (auto value = ir::As<ir::ConstInt>(expr)) {
        return std::to_string(value->value_);
    }
    return codegen.GetExprAsCode(expr);
}

static std::string ComputeNZDumpPhysicalOffsetExpr(codegen::CCECodegen& codegen, const ir::TensorTypePtr& tensor_type,
                                                   const ir::MakeTuplePtr& offsets, const std::string& full_rows)
{
    CHECK(tensor_type) << "debug.dump_tensor NZ lowering requires TensorType";
    CHECK(offsets && offsets->elements_.size() == 2) << "debug.dump_tensor NZ lowering requires 2D offsets";
    const int64_t c0 = cce::GetNZInnerCols(tensor_type->dtype_);
    const std::string row_offset = GetExprForNZDump(codegen, offsets->elements_[0]);
    const std::string col_offset = GetExprForNZDump(codegen, offsets->elements_[1]);
    return "(" + col_offset + " * " + full_rows + " + " + row_offset + " * " + std::to_string(c0) + ")";
}

struct DebugDumpTensorNZContext {
    int debug_id = 0;
    std::string tensor_name;
    std::string base_ptr;
    std::string start_offset;
};

static DebugDumpTensorNZContext PrepareDebugDumpTensorNZContext(codegen::CCECodegen& codegen,
                                                                const ir::VarPtr& tensor_var,
                                                                const ir::TensorTypePtr& tensor_type,
                                                                const ir::MakeTuplePtr& offsets_tuple,
                                                                const ir::MakeTuplePtr& shapes_tuple)
{
    ValidateDebugDumpNZWindowStructure(tensor_type, offsets_tuple, shapes_tuple);

    DebugDumpTensorNZContext context;
    context.debug_id = NextDebugDumpId();
    context.tensor_name = codegen.GetVarName(tensor_var);
    context.base_ptr = codegen.GetPointer(context.tensor_name);
    if (context.base_ptr.empty()) {
        context.base_ptr = context.tensor_name + ".data()";
    }
    return context;
}

static std::string MakeDebugDumpTensorNZStaticCodegenCCE(codegen::CCECodegen& codegen, const ir::VarPtr& tensor_var,
                                                         const ir::TensorTypePtr& tensor_type,
                                                         const ir::MakeTuplePtr& offsets_tuple,
                                                         const ir::MakeTuplePtr& shapes_tuple)
{
    DebugDumpTensorNZContext context = PrepareDebugDumpTensorNZContext(codegen, tensor_var, tensor_type, offsets_tuple,
                                                                       shapes_tuple);

    auto get_static_const_int_or_throw = [](const ir::ExprPtr& expr, const std::string& message) -> int64_t {
        auto value = ir::As<ir::ConstInt>(expr);
        if (!value) {
            throw pypto::ir::ValueError(message);
        }
        return value->value_;
    };

    const int64_t full_rows = get_static_const_int_or_throw(
        tensor_type->shape_[0], "debug.dump_tensor: CCE NZ dump static path requires a static destination row shape");
    const int64_t full_cols = get_static_const_int_or_throw(
        tensor_type->shape_[1],
        "debug.dump_tensor: CCE NZ dump static path requires a static destination column shape");
    const int64_t row_offset = get_static_const_int_or_throw(
        offsets_tuple->elements_[0], "debug.dump_tensor: CCE NZ dump static path requires a static row offset");
    const int64_t col_offset = get_static_const_int_or_throw(
        offsets_tuple->elements_[1], "debug.dump_tensor: CCE NZ dump static path requires a static column offset");
    const int64_t rows = get_static_const_int_or_throw(
        shapes_tuple->elements_[0], "debug.dump_tensor: CCE NZ dump static path requires a static row shape");
    const int64_t cols = get_static_const_int_or_throw(
        shapes_tuple->elements_[1], "debug.dump_tensor: CCE NZ dump static path requires a static column shape");

    const int64_t physical_offset = ComputeStaticNZPhysicalOffset(tensor_type->dtype_, full_rows, row_offset,
                                                                  col_offset);

    const std::string effective_offset = AddStartOffsetIfNeeded(context.start_offset, std::to_string(physical_offset));
    const std::string shape_alias = "__debug_dump_tensor_shape_" + std::to_string(context.debug_id);
    const std::string stride_alias = "__debug_dump_tensor_stride_" + std::to_string(context.debug_id);
    const std::string global_alias = "__debug_dump_tensor_type_" + std::to_string(context.debug_id);
    const std::string view_name = "__debug_dump_tensor_view_" + std::to_string(context.debug_id);

    codegen.Emit("using " + shape_alias + " = " + BuildStaticNZShapeType(tensor_type->dtype_, rows, cols) + ";");
    codegen.Emit("using " + stride_alias + " = " + BuildStaticNZStrideType(tensor_type->dtype_, full_rows, full_cols) +
                 ";");
    codegen.Emit("using " + global_alias + " = GlobalTensor<" + codegen.GetTypeString(tensor_type->dtype_) + ", " +
                 shape_alias + ", " + stride_alias + ", Layout::NZ>;");
    codegen.Emit(global_alias + " " + view_name + "(" + context.base_ptr + " + " + effective_offset + ");");
    codegen.Emit("TPRINT(" + view_name + ");");
    return "";
}

static std::string MakeDebugDumpTensorNZDynamicCodegenCCE(codegen::CCECodegen& codegen, const ir::VarPtr& tensor_var,
                                                          const ir::TensorTypePtr& tensor_type,
                                                          const ir::MakeTuplePtr& offsets_tuple,
                                                          const ir::MakeTuplePtr& shapes_tuple)
{
    DebugDumpTensorNZContext context = PrepareDebugDumpTensorNZContext(codegen, tensor_var, tensor_type, offsets_tuple,
                                                                       shapes_tuple);

    const std::string full_rows = GetTensorLogicalDimForNZDump(codegen, tensor_type, 0);
    const std::string full_cols = GetTensorLogicalDimForNZDump(codegen, tensor_type, 1);
    const bool is_full_tensor_window = IsFullTensorWindow(tensor_type, offsets_tuple, shapes_tuple);
    const std::string rows = is_full_tensor_window ? full_rows : GetExprForNZDump(codegen, shapes_tuple->elements_[0]);
    const std::string cols = is_full_tensor_window ? full_cols : GetExprForNZDump(codegen, shapes_tuple->elements_[1]);
    const std::string physical_offset = AddStartOffsetIfNeeded(
        context.start_offset, ComputeNZDumpPhysicalOffsetExpr(codegen, tensor_type, offsets_tuple, full_rows));

    const std::string dtype = codegen.GetTypeString(tensor_type->dtype_);
    const std::string shape_alias = "__debug_dump_tensor_shape_" + std::to_string(context.debug_id);
    const std::string stride_alias = "__debug_dump_tensor_stride_" + std::to_string(context.debug_id);
    const std::string global_alias = "__debug_dump_tensor_type_" + std::to_string(context.debug_id);
    const std::string view_name = "__debug_dump_tensor_view_" + std::to_string(context.debug_id);

    codegen.Emit("using " + shape_alias + " = pto::TileShape2D<" + dtype +
                 ", pto::DYNAMIC, pto::DYNAMIC, Layout::NZ>;");
    codegen.Emit("using " + stride_alias + " = pto::BaseShape2D<" + dtype +
                 ", pto::DYNAMIC, pto::DYNAMIC, Layout::NZ>;");
    codegen.Emit("using " + global_alias + " = GlobalTensor<" + dtype + ", " + shape_alias + ", " + stride_alias +
                 ", Layout::NZ>;");
    codegen.Emit(global_alias + " " + view_name + "(" + context.base_ptr + " + " + physical_offset + ", " +
                 shape_alias + "(" + rows + ", " + cols + "), " + stride_alias + "(" + full_rows + ", " + full_cols +
                 "));");
    codegen.Emit("TPRINT(" + view_name + ");");
    return "";
}

static void EmitDumpFlagHeaderCCE(codegen::CCECodegen& codegen, const ir::CallPtr& op)
{
    std::string dump_flag = op->HasKwarg("dump_flag") ? op->GetKwarg<std::string>("dump_flag") : "";
    if (dump_flag.empty()) {
        return;
    }
    codegen.Emit("cce::printf(\"=== [flag] %s ===\\n\", \"" + debug_printf::EscapeStringLiteral(dump_flag) + "\");");
}

static std::string MakeDebugDumpTensorCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 3) << "debug.dump_tensor requires 3 arguments, but got " << op->args_.size();
    if (op->GetKwarg<bool>("show_location", false)) {
        EmitDebugLocationHeaderCCE(codegen, op->span_, "dump_tensor");
    }
    EmitDumpFlagHeaderCCE(codegen, op);

    auto tensor_var = ir::As<ir::Var>(op->args_[0]);
    CHECK(tensor_var) << "debug.dump_tensor first argument must be a Var";
    auto tensor_type = ir::As<ir::TensorType>(tensor_var->GetType());
    CHECK(tensor_type) << "debug.dump_tensor first argument must be TensorType";
    auto offsets_tuple = ir::As<ir::MakeTuple>(op->args_[1]);
    CHECK(offsets_tuple) << "debug.dump_tensor second argument must be a tuple (offsets)";
    auto shapes_tuple = ir::As<ir::MakeTuple>(op->args_[2]);
    CHECK(shapes_tuple) << "debug.dump_tensor third argument must be a tuple (shapes)";

    if (cce::IsNZTensorType(tensor_type)) {
        ValidateDebugDumpNZWindowStructure(tensor_type, offsets_tuple, shapes_tuple);
        if (CanUseStaticDebugDumpNZFastPath(tensor_type, offsets_tuple, shapes_tuple)) {
            return MakeDebugDumpTensorNZStaticCodegenCCE(codegen, tensor_var, tensor_type, offsets_tuple, shapes_tuple);
        }
        return MakeDebugDumpTensorNZDynamicCodegenCCE(codegen, tensor_var, tensor_type, offsets_tuple, shapes_tuple);
    }

    const int debug_id = NextDebugDumpId();
    const std::string tensor_name = codegen.GetVarName(tensor_var);
    std::string base_ptr = codegen.GetPointer(tensor_name);
    if (base_ptr.empty()) {
        base_ptr = tensor_name + ".data()";
    }
    const std::string shape_alias = "__debug_dump_tensor_shape_" + std::to_string(debug_id);
    const std::string stride_alias = "__debug_dump_tensor_stride_" + std::to_string(debug_id);
    const std::string global_alias = "__debug_dump_tensor_type_" + std::to_string(debug_id);
    const std::string view_name = "__debug_dump_tensor_view_" + std::to_string(debug_id);
    const bool has_dynamic_tensor_shape = HasDynamicTensorShape(tensor_type);
    const bool is_full_tensor_window = IsFullTensorWindow(tensor_type, offsets_tuple, shapes_tuple);
    const bool use_runtime_tensor_view = has_dynamic_tensor_shape;

    std::string start_offset;
    const std::string offset_expr = use_runtime_tensor_view ?
                                        ComputeRuntimeStrideBasedOffset(codegen, tensor_name, tensor_type,
                                                                        offsets_tuple, start_offset) :
                                        cce::ComputeStrideBasedOffset(codegen, offsets_tuple, tensor_type);

    std::vector<std::string> shape_ctor_args;
    std::vector<std::string> stride_ctor_args;
    const std::string shape_type = BuildShapeTypeForDump(codegen, tensor_name, tensor_type, shapes_tuple->elements_,
                                                         use_runtime_tensor_view && is_full_tensor_window,
                                                         &shape_ctor_args);
    const std::string stride_type = BuildStrideTypeForDump(codegen, tensor_name, tensor_type, use_runtime_tensor_view,
                                                           &stride_ctor_args);

    std::string layout_suffix = ", Layout::ND";
    if (tensor_type->shape_.size() == 2) {
        if (auto last_dim = ir::As<ir::ConstInt>(tensor_type->shape_.back())) {
            if (last_dim->value_ == 1) {
                layout_suffix = ", Layout::DN";
            }
        }
    }

    codegen.Emit("using " + shape_alias + " = " + shape_type + ";");
    codegen.Emit("using " + stride_alias + " = " + stride_type + ";");
    codegen.Emit("using " + global_alias + " = GlobalTensor<" + codegen.GetTypeString(tensor_type->dtype_) + ", " +
                 shape_alias + ", " + stride_alias + layout_suffix + ">;");

    std::string shape_ctor = shape_alias + "(" + JoinExpressions(shape_ctor_args, ", ") + ")";
    if (shape_ctor_args.empty()) {
        shape_ctor = shape_alias + "()";
    }
    std::string stride_ctor = stride_alias + "(" + JoinExpressions(stride_ctor_args, ", ") + ")";
    if (stride_ctor_args.empty()) {
        stride_ctor = stride_alias + "()";
    }

    codegen.Emit(global_alias + " " + view_name + "(" + base_ptr + " + " + offset_expr + ", " + shape_ctor + ", " +
                 stride_ctor + ");");
    codegen.Emit("TPRINT(" + view_name + ");");
    return "";
}

static std::string MakeDebugPrintfCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);

    std::string format = op->GetKwarg<std::string>("format");
    if (op->GetKwarg<bool>("show_location", false)) {
        std::string location = debug_printf::FormatDebugLocation(op->span_);
        if (!location.empty()) {
            format = location + " " + format;
        }
    }

    std::vector<std::string> args;
    std::vector<DataType> arg_dtypes;
    args.reserve(op->args_.size());
    arg_dtypes.reserve(op->args_.size());

    // Parse format to know which args are %p (pointer) vs scalar
    auto segments = debug_printf::ParsePrintfSegments(format);
    for (size_t i = 0; i < op->args_.size(); ++i) {
        args.emplace_back(codegen.GetExprAsCode(op->args_[i]));
        if (i < segments.size() && segments[i].conversion == 'p') {
            // Pointer argument: use INDEX as a dummy dtype; CastCcePrintfArgIfNeeded handles %p specially
            arg_dtypes.emplace_back(DataType::INDEX);
        } else {
            auto scalar_type = ir::As<ir::ScalarType>(op->args_[i]->GetType());
            CHECK(scalar_type) << "debug.printf argument must be ScalarType in CCE lowering";
            arg_dtypes.emplace_back(scalar_type->dtype_);
        }
    }

    for (const auto& statement : MakeCcePrintfStatements(format, args, arg_dtypes)) {
        codegen.Emit(statement);
    }
    return "";
}

static std::string MakeDebugDumpTileCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 1 || op->args_.size() == 3 || op->args_.size() == 4)
        << "debug.dump_tile requires 1 argument (tile), 3 arguments (tile, offsets, shapes), "
        << "or 4 arguments (tile, offsets, shapes, workspace), but got " << op->args_.size();
    if (op->GetKwarg<bool>("show_location", false)) {
        EmitDebugLocationHeaderCCE(codegen, op->span_, "dump_tile");
    }
    EmitDumpFlagHeaderCCE(codegen, op);

    std::string src = codegen.GetExprAsCode(op->args_[0]);

    if (op->args_.size() == 4) {
        auto tile_type = ir::As<ir::TileType>(op->args_[0]->GetType());
        CHECK(tile_type) << "debug.dump_tile first argument must be TileType";
        CHECK(tile_type->shape_.size() == 2) << "debug.dump_tile Acc window dump only supports 2D tiles";
        auto workspace_var = ir::As<ir::Var>(op->args_[3]);
        CHECK(workspace_var) << "debug.dump_tile workspace (4th argument) must be a Var";
        std::string workspace_name = codegen.GetVarName(workspace_var);
        std::string workspace_ptr = codegen.GetPointer(workspace_name);

        auto tile_rows = ir::As<ir::ConstInt>(tile_type->shape_[0]);
        auto tile_cols = ir::As<ir::ConstInt>(tile_type->shape_[1]);
        CHECK(tile_rows && tile_cols) << "debug.dump_tile Acc dump requires static physical tile shape";

        auto offsets_tuple = ir::As<ir::MakeTuple>(op->args_[1]);
        CHECK(offsets_tuple) << "debug.dump_tile second argument must be a tuple (offsets)";
        auto shapes_tuple = ir::As<ir::MakeTuple>(op->args_[2]);
        CHECK(shapes_tuple) << "debug.dump_tile third argument must be a tuple (shapes)";

        const int debug_id = NextDebugDumpId();
        const std::string row_off = codegen.GetExprAsCode(offsets_tuple->elements_[0]);
        const std::string col_off = codegen.GetExprAsCode(offsets_tuple->elements_[1]);
        const std::string row_shape = codegen.GetExprAsCode(shapes_tuple->elements_[0]);
        const std::string col_shape = codegen.GetExprAsCode(shapes_tuple->elements_[1]);
        const std::string dtype_str = codegen.GetTypeString(tile_type->dtype_);
        const std::string requested_row = "__debug_dump_acc_rrow_" + std::to_string(debug_id);
        const std::string requested_col = "__debug_dump_acc_rcol_" + std::to_string(debug_id);
        const std::string valid_row = "__debug_dump_acc_vrow_" + std::to_string(debug_id);
        const std::string valid_col = "__debug_dump_acc_vcol_" + std::to_string(debug_id);
        const std::string row_idx = "__debug_dump_acc_r_" + std::to_string(debug_id);
        const std::string col_idx = "__debug_dump_acc_c_" + std::to_string(debug_id);
        const std::string debug_val = "__debug_dump_acc_val_" + std::to_string(debug_id);
        const std::string gm_buf = "__debug_dump_acc_gm_" + std::to_string(debug_id);
        const std::string cc_src = "__debug_dump_acc_cc_" + std::to_string(debug_id);

        codegen.Emit("pipe_barrier(PIPE_ALL);");
        codegen.Emit("{");
        codegen.Emit("  __gm__ " + dtype_str + "* " + gm_buf + " = reinterpret_cast<__gm__ " + dtype_str + "*>(" +
                     workspace_ptr + ");");
        codegen.Emit("  auto " + cc_src + " = " + src + ".data();");
        codegen.Emit("  constexpr uint16_t __m = " + std::to_string(tile_rows->value_) + ";");
        codegen.Emit("  constexpr uint16_t __n = " + std::to_string(tile_cols->value_) + ";");
        codegen.Emit("  constexpr uint16_t __src_stride = (__m + 15u) / 16u * 16u;");
        codegen.Emit("  constexpr uint16_t __c0 = 16;");
        codegen.Emit("  constexpr uint16_t __nd_num = 1;");
        codegen.Emit("  constexpr uint16_t __src_nd_stride = static_cast<uint16_t>(__src_stride * __n * __c0);");
        codegen.Emit("  constexpr uint16_t __dst_nd_stride = static_cast<uint16_t>(__m * __n);");
        codegen.Emit("  uint64_t __xm = ((uint64_t)(__n & 0xfff) << 4) | ((uint64_t)(__m & 0xffff) << 16) | "
                     "((uint64_t)(__n) << 32);");
        codegen.Emit("  uint64_t __xt = (uint64_t)__src_stride | ((uint64_t)1 << 43);");
        codegen.Emit("  uint64_t __cfg = (uint64_t)__nd_num | ((uint64_t)(__src_nd_stride & 0xffff) << 16) | "
                     "((uint64_t)(__dst_nd_stride & 0xffff) << 32);");
        codegen.Emit("  set_nd_para(__cfg);");
        codegen.Emit("  copy_matrix_cc_to_gm(" + gm_buf + ", " + cc_src + ", __xm, __xt);");
        codegen.Emit("}");
        codegen.Emit("pipe_barrier(PIPE_ALL);");

        codegen.Emit("int " + requested_row + " = " + row_shape + ";");
        codegen.Emit("if (" + requested_row + " < 0) " + requested_row + " = 0;");
        codegen.Emit("int " + requested_col + " = " + col_shape + ";");
        codegen.Emit("if (" + requested_col + " < 0) " + requested_col + " = 0;");
        codegen.Emit("int " + valid_row + " = " + requested_row + ";");
        codegen.Emit("if (" + valid_row + " > " + std::to_string(tile_rows->value_) + " - (" + row_off + ")) " +
                     valid_row + " = " + std::to_string(tile_rows->value_) + " - (" + row_off + ");");
        codegen.Emit("if (" + valid_row + " < 0) " + valid_row + " = 0;");
        codegen.Emit("int " + valid_col + " = " + requested_col + ";");
        codegen.Emit("if (" + valid_col + " > " + std::to_string(tile_cols->value_) + " - (" + col_off + ")) " +
                     valid_col + " = " + std::to_string(tile_cols->value_) + " - (" + col_off + ");");
        codegen.Emit("if (" + valid_col + " < 0) " + valid_col + " = 0;");

        codegen.Emit("cce::printf(\"=== [TPRINT Acc Tile Window] Data Type: %s, Layout: NZ, TileType: Acc ===\\n\", "
                     "__pypto_dtype_name<" +
                     dtype_str + ">());");
        codegen.Emit("cce::printf(\"  Source Shape: [%d, %d], Window Offsets: [%d, %d], Requested Shape: [%d, %d], "
                     "Valid Shape: [%d, %d]\\n\", " +
                     std::to_string(tile_rows->value_) + ", " + std::to_string(tile_cols->value_) +
                     ", static_cast<int>(" + row_off + "), static_cast<int>(" + col_off + "), " + requested_row + ", " +
                     requested_col + ", " + valid_row + ", " + valid_col + ");");

        codegen.Emit("{");
        codegen.Emit("  __gm__ " + dtype_str + "* __ws = reinterpret_cast<__gm__ " + dtype_str + "*>(" + workspace_ptr +
                     ");");
        codegen.Emit("  for (int " + row_idx + " = 0; " + row_idx + " < " + valid_row + "; ++" + row_idx + ") {");
        codegen.Emit("    for (int " + col_idx + " = 0; " + col_idx + " < " + valid_col + "; ++" + col_idx + ") {");
        codegen.Emit("      " + dtype_str + " " + debug_val + " = *(__ws + (" + row_idx + " + (" + row_off + ")) * " +
                     std::to_string(tile_cols->value_) + " + (" + col_idx + " + (" + col_off + ")));");
        codegen.Emit("      __pypto_print_val(" + debug_val + ");");
        codegen.Emit("    }");
        codegen.Emit("    cce::printf(\"\\n\");");
        codegen.Emit("  }");
        codegen.Emit("}");
        return "";
    }

    if (op->args_.size() == 1) {
        codegen.Emit("TPRINT(" + src + ");");
        return "";
    }

    auto tile_type = ir::As<ir::TileType>(op->args_[0]->GetType());
    CHECK(tile_type) << "debug.dump_tile first argument must be TileType";
    CHECK(tile_type->shape_.size() == 2) << "debug.dump_tile CCE lowering currently only supports 2D tiles";
    auto offsets_tuple = ir::As<ir::MakeTuple>(op->args_[1]);
    CHECK(offsets_tuple) << "debug.dump_tile second argument must be a tuple (offsets)";
    auto shapes_tuple = ir::As<ir::MakeTuple>(op->args_[2]);
    CHECK(shapes_tuple) << "debug.dump_tile third argument must be a tuple (shapes)";

    auto tile_rows = ir::As<ir::ConstInt>(tile_type->shape_[0]);
    auto tile_cols = ir::As<ir::ConstInt>(tile_type->shape_[1]);
    CHECK(tile_rows && tile_cols) << "debug.dump_tile CCE lowering requires static physical tile shape";

    const int debug_id = NextDebugDumpId();
    const std::string requested_row = "__debug_dump_tile_requested_row_" + std::to_string(debug_id);
    const std::string requested_col = "__debug_dump_tile_requested_col_" + std::to_string(debug_id);
    const std::string src_valid_row = "__debug_dump_tile_src_valid_row_" + std::to_string(debug_id);
    const std::string src_valid_col = "__debug_dump_tile_src_valid_col_" + std::to_string(debug_id);
    const std::string valid_row = "__debug_dump_tile_valid_row_" + std::to_string(debug_id);
    const std::string valid_col = "__debug_dump_tile_valid_col_" + std::to_string(debug_id);
    const std::string row_idx = "__debug_dump_tile_r_" + std::to_string(debug_id);
    const std::string col_idx = "__debug_dump_tile_c_" + std::to_string(debug_id);
    const std::string row_off = codegen.GetExprAsCode(offsets_tuple->elements_[0]);
    const std::string col_off = codegen.GetExprAsCode(offsets_tuple->elements_[1]);
    const std::string row_shape = codegen.GetExprAsCode(shapes_tuple->elements_[0]);
    const std::string col_shape = codegen.GetExprAsCode(shapes_tuple->elements_[1]);
    const std::string debug_val = "__debug_dump_tile_val_" + std::to_string(debug_id);

    codegen.Emit("pipe_barrier(PIPE_ALL);");
    codegen.Emit("int " + requested_row + " = " + row_shape + ";");
    codegen.Emit("if (" + requested_row + " < 0) " + requested_row + " = 0;");
    codegen.Emit("int " + requested_col + " = " + col_shape + ";");
    codegen.Emit("if (" + requested_col + " < 0) " + requested_col + " = 0;");
    codegen.Emit("int " + src_valid_row + " = " + src + ".GetValidRow() - (" + row_off + ");");
    codegen.Emit("if (" + src_valid_row + " < 0) " + src_valid_row + " = 0;");
    codegen.Emit("int " + src_valid_col + " = " + src + ".GetValidCol() - (" + col_off + ");");
    codegen.Emit("if (" + src_valid_col + " < 0) " + src_valid_col + " = 0;");
    codegen.Emit("int " + valid_row + " = " + requested_row + ";");
    codegen.Emit("if (" + valid_row + " > " + src_valid_row + ") " + valid_row + " = " + src_valid_row + ";");
    codegen.Emit("if (" + valid_row + " < 0) " + valid_row + " = 0;");
    codegen.Emit("if (" + valid_row + " > " + std::to_string(tile_rows->value_) + ") " + valid_row + " = " +
                 std::to_string(tile_rows->value_) + ";");
    codegen.Emit("int " + valid_col + " = " + requested_col + ";");
    codegen.Emit("if (" + valid_col + " > " + src_valid_col + ") " + valid_col + " = " + src_valid_col + ";");
    codegen.Emit("if (" + valid_col + " < 0) " + valid_col + " = 0;");
    codegen.Emit("if (" + valid_col + " > " + std::to_string(tile_cols->value_) + ") " + valid_col + " = " +
                 std::to_string(tile_cols->value_) + ";");
    codegen.Emit("cce::printf(\"=== [TPRINT Tile Window] Data Type: %s, Layout: %s, TileType: %s ===\\n\", "
                 "__pypto_dtype_name<" +
                 codegen.GetTypeString(tile_type->dtype_) +
                 ">(), pto::GetLayoutName(std::remove_reference_t<decltype(" + src +
                 ")>::BFractal, std::remove_reference_t<decltype(" + src + ")>::SFractal), \"Vec\");");
    codegen.Emit("cce::printf(\"  Source Shape: [%d, %d], Window Offsets: [%d, %d], Requested Shape: [%d, %d], "
                 "Valid Shape: [%d, %d]\\n\", " +
                 std::to_string(tile_rows->value_) + ", " + std::to_string(tile_cols->value_) + ", static_cast<int>(" +
                 row_off + "), static_cast<int>(" + col_off + "), " + requested_row + ", " + requested_col + ", " +
                 valid_row + ", " + valid_col + ");");
    codegen.Emit("for (int " + row_idx + " = 0; " + row_idx + " < " + valid_row + "; ++" + row_idx + ") {");
    codegen.Emit("  for (int " + col_idx + " = 0; " + col_idx + " < " + valid_col + "; ++" + col_idx + ") {");
    codegen.Emit("    auto __debug_src_offset = pto::GetTileOffset<std::remove_reference_t<decltype(" + src + ")>>(" +
                 row_idx + " + (" + row_off + "), " + col_idx + " + (" + col_off + "));");
    codegen.Emit("    auto " + debug_val + " = " + src + ".data()[__debug_src_offset];");
    codegen.Emit("    __pypto_print_val(" + debug_val + ");");
    codegen.Emit("  }");
    codegen.Emit("  cce::printf(\"\\n\");");
    codegen.Emit("}");
    return "";
}

// Helper function for get_block_idx (returns value expression).
// Matches AscendC GetBlockIdx(): AIV returns global AIV index, AIC returns AIC index.
static std::string MakeBlockGetBlockIdxCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    CHECK(op->args_.size() == 0) << "get_block_idx requires no arguments";
    auto& cg = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const auto target = cg.GetTarget();
    if (target == ir::SectionKind::Vector) {
        return "(int32_t)(get_block_idx() * get_subblockdim() + get_subblockid())";
    }
    return "(int32_t)(get_block_idx())";
}

// Helper function for block.make_tile (no-op: allocation handled elsewhere)
static std::string MakeBlockCreateTileCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    (void)op;
    (void)codegen_base;
    return ""; // No C++ emission - Tile declaration handled in prologue
}

// Helper for ptr.make_tensor (tensor view). Emits the GlobalTensor declaration in place
// at the make_tensor op: the source pointer (op->args_[0]) is already in C++ scope here
// (a function parameter or an earlier ptr.addptr local), so we resolve it directly via
// GetExprAsCode instead of relying on PtrType base/offset annotations (those are ptoas-only).
// The view's access_shape/is_transpose/tile_dims come from the prescanned TensorDefs, looked up
// by the assignment target name -- one per layout its accesses need. Returns "" (no inline value).
static std::string MakeBlockMakeTensorCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& cg = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string name = cg.GetCurrentResultTarget();
    // The source of the view is either a raw pointer (PtrType) or an existing tensor (TensorType,
    // re-viewed with a new shape/stride). Resolve the base pointer accordingly and record the
    // source element dtype.
    std::string ptr_code;
    ir::DataType source_dtype;
    if (auto ptr_type = ir::As<ir::PtrType>(op->args_[0]->GetType())) {
        ptr_code = cg.GetExprAsCode(op->args_[0]);
        source_dtype = ptr_type->dtype_;
    } else if (auto src_tensor_type = ir::As<ir::TensorType>(op->args_[0]->GetType())) {
        // Re-view of an existing tensor: reuse its already-registered base pointer (a function
        // parameter's "<name>_ptr" or an earlier make_tensor view's pointer).
        auto src_var = std::dynamic_pointer_cast<const ir::Var>(op->args_[0]);
        CHECK(src_var != nullptr) << "ptr.make_tensor from a tensor requires the source to be a tensor variable";
        const std::string src_name = cg.GetVarName(src_var);
        ptr_code = cg.HasPointer(src_name) ? cg.GetPointer(src_name) : (src_name + ".data()");
        source_dtype = src_tensor_type->dtype_;
    } else {
        CHECK(false) << "ptr.make_tensor source must be a PtrType or TensorType";
    }
    // The view's element dtype may differ from the source's element dtype (e.g. a raw uint8
    // pointer reinterpreted as an fp16 view via ptr.make_tensor(..., dtype=FP16)). The
    // GlobalTensor<element_type> instance is constructed from this pointer, so reinterpret-cast
    // the base pointer to the view element type when the dtypes differ (a no-op when they match).
    auto tensor_type = ir::As<ir::TensorType>(op->GetType());
    if (tensor_type && !(source_dtype == tensor_type->dtype_)) {
        ptr_code = "(__gm__ " + tensor_type->dtype_.ToCTypeString() + "*)(" + ptr_code + ")";
    }
    cg.RegisterPointer(name, ptr_code);
    for (const codegen::TensorDef* def : cg.GetTensorDefs(name)) {
        cg.GenerateGlobalTensorTypeDeclaration(*def);
    }
    return "";
}

// Helper for ptr.addptr / advancing a raw pointer by an element offset. Emits no
// statement; returns the pointer-arithmetic expression so the result var maps to
// it (used as a base address by a subsequent ptr.make_tensor).
static std::string MakePtrAddPtrCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 2) << "ptr.addptr requires 2 arguments: ptr, offset";
    std::string ptr = codegen.GetExprAsCode(op->args_[0]);
    std::string offset = codegen.GetExprAsCode(op->args_[1]);
    return "(" + ptr + " + " + offset + ")";
}

// Helper for ptr.make_ptr / reinterpreting a raw pointer (or extracting a pointer from a
// tensor) as a different element type. Emits no statement; returns the expression so the
// result var maps to it (used as a base address by a subsequent ptr.addptr / ptr.make_tensor).
static std::string MakePtrMakePtrCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 1) << "ptr.make_ptr requires 1 argument: ptr";
    std::string ptr;
    if (auto ptr_type = ir::As<ir::PtrType>(op->args_[0]->GetType())) {
        ptr = codegen.GetExprAsCode(op->args_[0]);
    } else if (ir::As<ir::TensorType>(op->args_[0]->GetType())) {
        auto src_var = std::dynamic_pointer_cast<const ir::Var>(op->args_[0]);
        CHECK(src_var != nullptr) << "ptr.make_ptr from a tensor requires the source to be a tensor variable";
        const std::string src_name = codegen.GetVarName(src_var);
        ptr = codegen.HasPointer(src_name) ? codegen.GetPointer(src_name) : (src_name + ".data()");
    } else {
        CHECK(false) << "ptr.make_ptr source must be a PtrType or TensorType";
    }
    auto result_ptr_type = ir::As<ir::PtrType>(op->GetType());
    CHECK(result_ptr_type != nullptr) << "ptr.make_ptr result must be a PtrType";
    return "((__gm__ " + result_ptr_type->dtype_.ToCTypeString() + "*)(" + ptr + "))";
}

// ============================================================================
// Matmul Operations
// ============================================================================

// ============================================================================
// Elementwise Operations
// ============================================================================

// ============================================================================
// Unary Operations
// ============================================================================

// ============================================================================
// Memory Operations
// ============================================================================

REGISTER_BACKEND_OP(BackendCCE, "block.make_tile")
    .set_pipe(ir::PipeType::MTE2)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeBlockCreateTileCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "ptr.make_tensor")
    .set_pipe(ir::PipeType::MTE2)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeBlockMakeTensorCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "ptr.addptr")
    .set_pipe(ir::PipeType::MTE2)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakePtrAddPtrCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "ptr.make_ptr")
    .set_pipe(ir::PipeType::MTE2)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakePtrMakePtrCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "get_block_idx")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeBlockGetBlockIdxCodegenCCE(op, codegen);
    });

// Helper function for get_spr (reads AR special purpose register via get_ar())
static std::string MakeGetSprCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    (void)codegen_base;
    CHECK(op->args_.size() == 0) << "get_spr requires no arguments";
    return "get_ar()";
}

REGISTER_BACKEND_OP(BackendCCE, "get_spr")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return MakeGetSprCodegenCCE(op, codegen); });

// ============================================================================
// Reduction Operations
// ============================================================================

// ============================================================================
// Broadcast Operations
// ============================================================================

// ============================================================================
// Transform Operations (view/reshape/transpose: same buffer, reinterpret)
// ============================================================================

[[maybe_unused]] static std::string MakeTileTransposeCodegenCCE(const ir::CallPtr& op,
                                                                codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    std::string target_var = codegen.GetCurrentResultTarget();
    std::string input_var = codegen.GetExprAsCode(op->args_[0]);
    auto axis1 = codegen.GetConstIntValue(op->args_[1]);
    auto axis2 = codegen.GetConstIntValue(op->args_[2]);
    int64_t ndim = static_cast<int64_t>(ir::As<ir::TileType>(op->args_[0]->GetType())->shape_.size());

    INTERNAL_CHECK(ndim == 2) << "Codegen only supports 2D tiles, but got " << ndim << "D tile";
    INTERNAL_CHECK(axis1 != axis2) << "tile.transpose: axis1 and axis2 must be different, but got axis1=axis2="
                                   << axis1;
    INTERNAL_CHECK(axis1 >= 0 && axis1 < ndim && axis2 >= 0 && axis2 < ndim)
        << "tile.transpose: axis1 and axis2 must be in range [0, " << ndim << "), but got axis1=" << axis1
        << ", axis2=" << axis2;

    codegen.Emit("TTRANS(" + target_var + ", " + input_var + ");");
    return "";
}

// ============================================================================
// Sync / Barrier Operations (inserted by insert_sync_pass)
// ============================================================================

static std::string PipeTypeToCCEString(ir::PipeType pipe)
{
    switch (pipe) {
        case ir::PipeType::MTE1:
            return "PIPE_MTE1";
        case ir::PipeType::MTE2:
            return "PIPE_MTE2";
        case ir::PipeType::MTE3:
            return "PIPE_MTE3";
        case ir::PipeType::M:
            return "PIPE_M";
        case ir::PipeType::V:
            return "PIPE_V";
        case ir::PipeType::S:
            return "PIPE_S";
        case ir::PipeType::FIX:
            return "PIPE_FIX";
        case ir::PipeType::ALL:
            return "PIPE_ALL";
        default:
            return "PIPE_V";
    }
}

static std::string MakeSyncCodegenCCE(const std::string& isa_name, const ir::CallPtr& op,
                                      codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    auto set_pipe = static_cast<ir::PipeType>(op->GetKwarg<int>("set_pipe"));
    auto wait_pipe = static_cast<ir::PipeType>(op->GetKwarg<int>("wait_pipe"));
    int event_id = op->GetKwarg<int>("event_id");
    std::string set_pipe_str = PipeTypeToCCEString(set_pipe);
    std::string wait_pipe_str = PipeTypeToCCEString(wait_pipe);
    std::string event_id_str = "EVENT_ID" + std::to_string(event_id);
    codegen.Emit(isa_name + "(" + set_pipe_str + ", " + wait_pipe_str + ", " + event_id_str + ");");
    return "";
}

static std::string EnumValueName(const char* full_name)
{
    const char* sep = std::strrchr(full_name, ':');
    return sep ? std::string(sep + 1) : std::string(full_name);
}

static std::string NormalizeDcciCacheLine(int cache_line)
{
    auto cl = static_cast<ir::CacheLine>(cache_line);
    return EnumValueName(ir::EnumToString(cl));
}

static std::string NormalizeDcciDst(int dst, bool is_tile)
{
    auto d = static_cast<ir::DcciDst>(dst);
    if (d == ir::DcciDst::AUTO) {
        return is_tile ? "CACHELINE_UB" : "CACHELINE_OUT";
    }
    return EnumValueName(ir::EnumToString(d));
}

static std::string MakeDcciCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 1 || op->args_.size() == 2)
        << "system.dcci requires 1 or 2 arguments, got " << op->args_.size();

    int cache_line_int = op->HasKwarg("cache_line") ? op->GetKwarg<int>("cache_line") : 1; // ENTIRE_DATA_CACHE
    int dst_int = op->HasKwarg("dst") ? op->GetKwarg<int>("dst") : 0;                      // AUTO

    const std::string cache_line = NormalizeDcciCacheLine(cache_line_int);

    auto tensor_type = ir::As<ir::TensorType>(op->args_[0]->GetType());
    if (tensor_type != nullptr) {
        auto tensor_var_ptr = std::dynamic_pointer_cast<const ir::Var>(op->args_[0]);
        CHECK(tensor_var_ptr != nullptr) << "system.dcci: tensor target must be a Var";
        std::string tensor_var = codegen.GetVarName(tensor_var_ptr);
        std::string offset = "0";
        if (op->args_.size() == 2) {
            auto offsets_tuple = std::dynamic_pointer_cast<const ir::MakeTuple>(op->args_[1]);
            if (offsets_tuple != nullptr) {
                offset = cce::ComputeStrideBasedOffset(codegen, offsets_tuple, tensor_type);
            } else {
                CHECK(ir::As<ir::ScalarType>(op->args_[1]->GetType()) != nullptr)
                    << "system.dcci: tensor target offset must be a tuple or scalar expression";
                offset = codegen.GetExprAsCode(op->args_[1]);
            }
        }
        std::string tensor_ptr = codegen.GetPointer(tensor_var);
        if (tensor_ptr.empty()) {
            tensor_ptr = tensor_var + ".data()";
        }
        const std::string dst_attr = NormalizeDcciDst(dst_int, false);
        codegen.Emit("dcci(reinterpret_cast<__gm__ void*>(" + tensor_ptr + " + " + offset + "), " + cache_line + ", " +
                     dst_attr + ");");
        return "";
    }

    auto tile_type = ir::As<ir::TileType>(op->args_[0]->GetType());
    CHECK(tile_type != nullptr) << "system.dcci: target must be TensorType or TileType";
    CHECK(tile_type->memref_.has_value()) << "system.dcci: tile target must have an allocated memory space";
    if (tile_type->memref_.value()->memorySpace_ != ir::MemorySpace::Vec) {
        throw pypto::ir::ValueError("system.dcci: tile target must be allocated in Vec memory");
    }

    std::string tile = codegen.GetExprAsCode(op->args_[0]);
    std::string offset = "0";
    if (op->args_.size() == 2) {
        offset = codegen.GetExprAsCode(op->args_[1]);
    }
    const std::string dst_attr = NormalizeDcciDst(dst_int, true);
    codegen.Emit("dcci(reinterpret_cast<__ubuf__ void*>(" + tile + ".data() + " + offset + "), " + cache_line + ", " +
                 dst_attr + ");");
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "system.sync_src")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeSyncCodegenCCE("set_flag", op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "system.sync_dst")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeSyncCodegenCCE("wait_flag", op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "system.bar_m")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
        (void)op;
        dynamic_cast<codegen::CCECodegen&>(codegen_base).Emit("pipe_barrier(PIPE_M);");
        return "";
    });

REGISTER_BACKEND_OP(BackendCCE, "system.bar_mte1")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
        (void)op;
        dynamic_cast<codegen::CCECodegen&>(codegen_base).Emit("pipe_barrier(PIPE_MTE1);");
        return "";
    });

REGISTER_BACKEND_OP(BackendCCE, "system.bar_mte2")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
        (void)op;
        dynamic_cast<codegen::CCECodegen&>(codegen_base).Emit("pipe_barrier(PIPE_MTE2);");
        return "";
    });

REGISTER_BACKEND_OP(BackendCCE, "system.bar_mte3")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
        (void)op;
        dynamic_cast<codegen::CCECodegen&>(codegen_base).Emit("pipe_barrier(PIPE_MTE3);");
        return "";
    });

REGISTER_BACKEND_OP(BackendCCE, "system.bar_fix")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
        (void)op;
        dynamic_cast<codegen::CCECodegen&>(codegen_base).Emit("pipe_barrier(PIPE_FIX);");
        return "";
    });

REGISTER_BACKEND_OP(BackendCCE, "system.bar_all")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
        (void)op;
        dynamic_cast<codegen::CCECodegen&>(codegen_base).Emit("pipe_barrier(PIPE_ALL);");
        return "";
    });

REGISTER_BACKEND_OP(BackendCCE, "system.set_mask_count")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
        (void)op;
        dynamic_cast<codegen::CCECodegen&>(codegen_base).Emit("set_mask_count();");
        return "";
    });

REGISTER_BACKEND_OP(BackendCCE, "system.set_mask_norm")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
        (void)op;
        dynamic_cast<codegen::CCECodegen&>(codegen_base).Emit("set_mask_norm();");
        return "";
    });

REGISTER_BACKEND_OP(BackendCCE, "system.set_vec_mask")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
        auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
        CHECK(op->args_.size() == 2) << "system.set_vec_mask requires 2 arguments, but got " << op->args_.size();
        std::string mask_high = codegen.GetExprAsCode(op->args_[0]);
        std::string mask_low = codegen.GetExprAsCode(op->args_[1]);
        codegen.Emit("set_vector_mask(" + mask_high + ", " + mask_low + ");");
        return "";
    });

REGISTER_BACKEND_OP(BackendCCE, "system.reset_mask")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
        (void)op;
        dynamic_cast<codegen::CCECodegen&>(codegen_base)
            .Emit("set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));");
        return "";
    });

REGISTER_BACKEND_OP(BackendCCE, "system.dcci")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
        return MakeDcciCodegenCCE(op, codegen_base);
    });

// ============================================================================
// Cross-core Sync Operations
// ============================================================================

// Cross-core SET: ffts_cross_core_sync(PIPE_xxx, getFFTSMsg(mode, event_id))
//   - INTER_BLOCK(0): inter-core sync
//   - INTER_SUBBLOCK(1): intra-core AIV-to-AIV sync
//   - INTRA_BLOCK(2): intra-core AIC↔AIV both subcores (A5 uses set_intra_block)
//   - UNICAST_BLOCK(3): intra-core AIC↔AIV one subcore (A5 uses set_intra_block)
// Cross-core WAIT: wait_flag_dev / wait_intra_block
// SET signals completion from a pipe; WAIT blocks until the other core signals.

static std::string MakeCrossCoreSetCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base,
                                              bool is_dynamic)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    auto pipe = op->GetKwarg<int>("pipe");
    std::string pipe_str = PipeTypeToCCEString(static_cast<ir::PipeType>(pipe));
    bool is_a5 = (codegen.GetArch() == "a5");
    auto sync_mode = static_cast<ir::CrossCoreSyncMode>(op->GetKwarg<int>("sync_mode"));
    bool is_intra_unicast = sync_mode == ir::CrossCoreSyncMode::INTRA_BLOCK ||
                            sync_mode == ir::CrossCoreSyncMode::UNICAST_BLOCK;
    if (is_a5 && is_intra_unicast) {
        // A5 + INTRA_BLOCK or UNICAST_BLOCK: set_intra_block
        //     CUBE→VEC: INTRA_BLOCK expands to two calls (v0: id, v1: id+16); UNICAST_BLOCK single call
        //     VEC→CUBE: single set
        if (codegen.GetTarget() == ir::SectionKind::Cube && sync_mode == ir::CrossCoreSyncMode::INTRA_BLOCK) {
            if (is_dynamic) {
                std::string event_id = codegen.GetExprAsCode(op->args_[0]);
                codegen.Emit("set_intra_block(" + pipe_str + ", " + event_id + ");");
                codegen.Emit("set_intra_block(" + pipe_str + ", " + event_id + " + 16);");
            } else {
                int event_id = op->GetKwarg<int>("event_id");
                codegen.Emit("set_intra_block(" + pipe_str + ", " + std::to_string(event_id) + ");");
                codegen.Emit("set_intra_block(" + pipe_str + ", " + std::to_string(event_id + 16) + ");");
            }
        } else {
            if (is_dynamic) {
                std::string event_id = codegen.GetExprAsCode(op->args_[0]);
                codegen.Emit("set_intra_block(" + pipe_str + ", " + event_id + ");");
            } else {
                int event_id = op->GetKwarg<int>("event_id");
                codegen.Emit("set_intra_block(" + pipe_str + ", " + std::to_string(event_id) + ");");
            }
        }
    } else {
        // non-A5, or A5 + INTER_BLOCK / INTER_SUBBLOCK: ffts_cross_core_sync
        std::string mode_str = std::to_string(static_cast<int>(sync_mode));
        if (is_dynamic) {
            std::string event_id = codegen.GetExprAsCode(op->args_[0]);
            codegen.Emit("ffts_cross_core_sync(" + pipe_str + ", getFFTSMsg(" + mode_str + ", " + event_id + "));");
        } else {
            int event_id = op->GetKwarg<int>("event_id");
            codegen.Emit("ffts_cross_core_sync(" + pipe_str + ", getFFTSMsg(" + mode_str + ", " +
                         std::to_string(event_id) + "));");
        }
    }
    return "";
}

static void EmitWaitIntraBlockCCE(codegen::CCECodegen& codegen, const ir::CallPtr& op, const std::string& pipe_str,
                                  bool is_dynamic, int event_id_offset = 0)
{
    if (is_dynamic) {
        std::string event_id = codegen.GetExprAsCode(op->args_[0]);
        if (event_id_offset != 0) {
            event_id += " + " + std::to_string(event_id_offset);
        }
        codegen.Emit("wait_intra_block(" + pipe_str + ", " + event_id + ");");
        return;
    }

    int event_id = op->GetKwarg<int>("event_id") + event_id_offset;
    codegen.Emit("wait_intra_block(" + pipe_str + ", " + std::to_string(event_id) + ");");
}

static std::string MakeCrossCoreWaitCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base,
                                               bool is_dynamic)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    auto pipe = op->GetKwarg<int>("pipe");
    std::string pipe_str = PipeTypeToCCEString(static_cast<ir::PipeType>(pipe));
    auto sync_mode = static_cast<ir::CrossCoreSyncMode>(op->GetKwarg<int>("sync_mode"));
    bool is_intra_unicast = (sync_mode == ir::CrossCoreSyncMode::INTRA_BLOCK ||
                             sync_mode == ir::CrossCoreSyncMode::UNICAST_BLOCK);
    bool wait_two_vec_subcores = (sync_mode == ir::CrossCoreSyncMode::INTRA_BLOCK);
    bool is_a5 = (codegen.GetArch() == "a5");
    if (is_a5 && is_intra_unicast) {
        // A5 + INTRA_BLOCK(2) or UNICAST_BLOCK(3): wait_intra_block
        //     CUBE waiting for VEC: INTRA_BLOCK expands to two calls (v0: id, v1: id+16); UNICAST_BLOCK single call
        //     VEC waiting for CUBE: single wait
        if (codegen.GetTarget() == ir::SectionKind::Cube && wait_two_vec_subcores) {
            EmitWaitIntraBlockCCE(codegen, op, pipe_str, is_dynamic);
            EmitWaitIntraBlockCCE(codegen, op, pipe_str, is_dynamic, 16);
        } else {
            EmitWaitIntraBlockCCE(codegen, op, pipe_str, is_dynamic);
        }
    } else {
        // non-A5, or A5 + INTER_BLOCK(0) / INTER_SUBBLOCK(1): wait_flag_dev
        std::string event_id = is_dynamic ? codegen.GetExprAsCode(op->args_[0]) :
                                            std::to_string(op->GetKwarg<int>("event_id"));
        if (is_a5) {
            codegen.Emit("wait_flag_dev(" + pipe_str + ", " + event_id + ");");
        } else {
            codegen.Emit("wait_flag_dev(" + event_id + ");");
        }
    }
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "system.set_cross_core")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeCrossCoreSetCodegenCCE(op, codegen, false);
    });

REGISTER_BACKEND_OP(BackendCCE, "system.wait_cross_core")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeCrossCoreWaitCodegenCCE(op, codegen, false);
    });

REGISTER_BACKEND_OP(BackendCCE, "system.set_cross_core_dyn")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeCrossCoreSetCodegenCCE(op, codegen, true);
    });

REGISTER_BACKEND_OP(BackendCCE, "system.wait_cross_core_dyn")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeCrossCoreWaitCodegenCCE(op, codegen, true);
    });

// ============================================================================
// Dynamic event_id sync operations
// ============================================================================

static std::string MakeSyncSrcDynCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    auto set_pipe = static_cast<ir::PipeType>(op->GetKwarg<int>("set_pipe"));
    auto wait_pipe = static_cast<ir::PipeType>(op->GetKwarg<int>("wait_pipe"));
    std::string event_id = codegen.GetExprAsCode(op->args_[0]);
    std::string set_pipe_str = PipeTypeToCCEString(set_pipe);
    std::string wait_pipe_str = PipeTypeToCCEString(wait_pipe);
    // EventId array access (contains '[') already returns event_t —no cast needed
    if (event_id.find('[') != std::string::npos) {
        codegen.Emit("set_flag(" + set_pipe_str + ", " + wait_pipe_str + ", " + event_id + ");");
    } else {
        codegen.Emit("set_flag(" + set_pipe_str + ", " + wait_pipe_str + ", (event_t)" + event_id + ");");
    }
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "system.sync_src_dyn")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeSyncSrcDynCodegenCCE(op, codegen);
    });

static std::string MakeSyncDstDynCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    auto set_pipe = static_cast<ir::PipeType>(op->GetKwarg<int>("set_pipe"));
    auto wait_pipe = static_cast<ir::PipeType>(op->GetKwarg<int>("wait_pipe"));
    std::string event_id = codegen.GetExprAsCode(op->args_[0]);
    std::string set_pipe_str = PipeTypeToCCEString(set_pipe);
    std::string wait_pipe_str = PipeTypeToCCEString(wait_pipe);
    // EventId array access (contains '[') already returns event_t —no cast needed
    if (event_id.find('[') != std::string::npos) {
        codegen.Emit("wait_flag(" + set_pipe_str + ", " + wait_pipe_str + ", " + event_id + ");");
    } else {
        codegen.Emit("wait_flag(" + set_pipe_str + ", " + wait_pipe_str + ", (event_t)" + event_id + ");");
    }
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "system.sync_dst_dyn")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeSyncDstDynCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "debug.dump_tensor")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeDebugDumpTensorCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "debug.dump_tile")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeDebugDumpTileCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "debug.printf")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeDebugPrintfCodegenCCE(op, codegen);
    });

// ============================================================================
// Debug operations: assert and trap
// ============================================================================

static std::string MakeDebugAssertCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);

    CHECK(op->args_.size() >= 1) << "debug.assert requires at least 1 argument (condition)";

    std::string condition = codegen.GetExprAsCode(op->args_[0]);
    std::string condition_text = op->GetKwarg<std::string>("condition_text");
    std::string format = op->GetKwarg<std::string>("format");

    codegen.Emit("if (!(" + condition + ")) {");

    if (op->GetKwarg<bool>("show_location", false)) {
        std::string location = debug_printf::FormatDebugLocation(op->span_);
        if (!location.empty()) {
            codegen.Emit("  cce::printf(\"" +
                         debug_printf::EscapeStringLiteral(location + " Assertion failed: " + condition_text + "\n") +
                         "\");");
        } else {
            codegen.Emit("  cce::printf(\"" +
                         debug_printf::EscapeStringLiteral("Assertion failed: " + condition_text + "\n") + "\");");
        }
    } else {
        codegen.Emit("  cce::printf(\"" +
                     debug_printf::EscapeStringLiteral("Assertion failed: " + condition_text + "\n") + "\");");
    }

    if (!format.empty() && op->args_.size() > 1) {
        std::vector<std::string> args;
        std::vector<DataType> arg_dtypes;
        for (size_t i = 1; i < op->args_.size(); ++i) {
            args.emplace_back(codegen.GetExprAsCode(op->args_[i]));
            auto scalar_type = ir::As<ir::ScalarType>(op->args_[i]->GetType());
            CHECK(scalar_type) << "debug.assert argument must be ScalarType";
            arg_dtypes.emplace_back(scalar_type->dtype_);
        }
        for (const auto& statement : MakeCcePrintfStatements(format, args, arg_dtypes)) {
            codegen.Emit("  " + statement);
        }
    }

    codegen.Emit("}");
    return "";
}

static std::string MakeDebugTrapCodegenCCE(const ir::CallPtr& /*op*/, codegen::CodegenBase& /*codegen_base*/)
{
    return "return";
}

REGISTER_BACKEND_OP(BackendCCE, "debug.assert")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeDebugAssertCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "debug.trap")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeDebugTrapCodegenCCE(op, codegen);
    });

// ============================================================================
// Language operations: get_block_num, get_subblock_idx
// ============================================================================

REGISTER_BACKEND_OP(BackendCCE, "get_block_num")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& /*codegen_base*/) {
        CHECK(op->args_.size() == 0) << "get_block_num requires no arguments";
        return std::string("(int32_t)(get_block_num())");
    });

REGISTER_BACKEND_OP(BackendCCE, "get_subblock_idx")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& /*codegen_base*/) {
        CHECK(op->args_.size() == 0) << "get_subblock_idx requires no arguments";
        return std::string("(int32_t)(get_subblockid())");
    });

REGISTER_BACKEND_OP(BackendCCE, "get_subblock_num")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
        CHECK(op->args_.size() == 0) << "get_subblock_num requires no arguments";
        // Matches AscendC GetTaskRation(): AIC returns 1, AIV returns get_subblockdim().
        auto& cg = dynamic_cast<codegen::CCECodegen&>(codegen_base);
        const auto target = cg.GetTarget();
        if (target == ir::SectionKind::Vector) {
            return std::string("(int32_t)(get_subblockdim())");
        }
        return std::string("(int32_t)(1)");
    });

// ============================================================================
// GetVal/SetVal Operations (unified: tile and tensor)
// ============================================================================

static std::string MakeGetValCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 2) << "getval requires 2 arguments, but got " << op->args_.size();

    auto first_type = op->args_[0]->GetType();
    if (ir::As<ir::TileType>(first_type)) {
        std::string tile = codegen.GetExprAsCode(op->args_[0]);
        std::string offset = codegen.GetExprAsCode(op->args_[1]);
        if (codegen.IsInSimtContext()) {
            return tile + "[" + offset + "]";
        }
        return tile + ".GetValue(" + offset + ")";
    }

    auto tensor_var = ir::As<ir::Var>(op->args_[0]);
    INTERNAL_CHECK(tensor_var) << "getval requires tensor to be a Var";
    auto tensor_type = ir::As<ir::TensorType>(tensor_var->GetType());
    INTERNAL_CHECK(tensor_type) << "getval requires TensorType";
    std::string tensor_name = codegen.GetVarName(tensor_var);
    std::string offset = codegen.GetExprAsCode(op->args_[1]);
    std::string dtype_str = codegen.GetTypeString(tensor_type->dtype_);

    std::string tensor_ptr = codegen.GetPointer(tensor_name);

    return "*((__gm__ " + dtype_str + "*)" + tensor_ptr + " + " + offset + ")";
}

static std::string MakeSetValCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 3) << "setval requires 3 arguments, but got " << op->args_.size();

    auto first_type = op->args_[0]->GetType();
    if (ir::As<ir::TileType>(first_type)) {
        std::string tile = codegen.GetExprAsCode(op->args_[0]);
        std::string offset = codegen.GetExprAsCode(op->args_[1]);
        std::string value = codegen.GetExprAsCode(op->args_[2]);
        if (codegen.IsInSimtContext()) {
            codegen.Emit(tile + "[" + offset + "] = " + value + ";");
            return "";
        }
        codegen.Emit(tile + ".SetValue(" + offset + ", " + value + ");");
        return "";
    }

    auto tensor_var = ir::As<ir::Var>(op->args_[0]);
    INTERNAL_CHECK(tensor_var) << "setval requires tensor to be a Var";
    auto tensor_type = ir::As<ir::TensorType>(tensor_var->GetType());
    INTERNAL_CHECK(tensor_type) << "setval requires TensorType";
    std::string tensor_name = codegen.GetVarName(tensor_var);
    std::string offset = codegen.GetExprAsCode(op->args_[1]);
    std::string value = codegen.GetExprAsCode(op->args_[2]);
    std::string dtype_str = codegen.GetTypeString(tensor_type->dtype_);

    std::string tensor_ptr = codegen.GetPointer(tensor_name);

    codegen.Emit("*((__gm__ " + dtype_str + "*)" + tensor_ptr + " + " + offset + ") = " + value + ";");
    return "";
}

static std::string MakeTileValidShapeCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    CHECK(op->args_.size() == 1 && ir::As<ir::TileType>(op->args_[0]->GetType()))
        << "block.tile_valid_shape requires one Tile argument";
    int axis = op->GetKwarg<int>("axis");
    CHECK(axis >= 0 && axis <= 1) << "block.tile_valid_shape axis must be in [0, 1]";

    if (codegen.IsInSimtContext()) {
        return codegen.GetExprAsCode(op->args_[0]) + (axis == 0 ? "__valid_row" : "__valid_col");
    }
    return codegen.GetExprAsCode(op->args_[0]) + (axis == 0 ? ".GetValidRow()" : ".GetValidCol()");
}

// getval/setval use the "block." IR namespace like every other explicit-output
// block op (block.add, block.matmul, ...). This keeps codegen dispatch (keyed on
// op->name_) and the parser's auto_mutex pipe lookup (get_op_pipe -> "block.<name>")
// consistent, so getval/setval resolve to PIPE_S and participate in auto_mutex.
REGISTER_BACKEND_OP(BackendCCE, "block.getval")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return MakeGetValCodegenCCE(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "block.setval")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return MakeSetValCodegenCCE(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "block.tile_valid_shape")
    .set_pipe(ir::PipeType::S)
    .f_codegen(MakeTileValidShapeCodegenCCE);

// ============================================================================
// block.subview - tile/tensor sub-view with offset and new shape.
//
// Tile in VF section: returns pointer arithmetic expression.
// Tile in non-VF section: emits TASSIGN with new shape, returns the new tile
// variable name.
// ============================================================================
static std::string MakeBlockSubviewCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);

    auto tile_type = ir::As<ir::TileType>(op->args_[0]->GetType());
    auto offset_tuple = ir::As<ir::MakeTuple>(op->args_[1]);
    bool is_tuple = (offset_tuple != nullptr);

    int64_t rows = tile_type->shape_.size() >= 1 ? codegen.GetConstIntValue(tile_type->shape_[0]) : 1;
    int64_t cols = tile_type->shape_.size() >= 2 ? codegen.GetConstIntValue(tile_type->shape_[1]) : 1;
    const auto& hw = tile_type->hardwareInfo_.value();

    // Element offset: DN (col-major) col*rows+row; ND (row-major) row*cols+col.
    std::string elem_off;
    if (is_tuple) {
        auto row_expr = codegen.GetExprAsCode(offset_tuple->elements_[0]);
        auto col_expr = codegen.GetExprAsCode(offset_tuple->elements_[1]);
        if (hw.blayout == ir::TileLayout::col_major) { // DN
            elem_off = "(" + col_expr + ")*" + std::to_string(rows) + "+(" + row_expr + ")";
        } else { // ND
            elem_off = "(" + row_expr + ")*" + std::to_string(cols) + "+(" + col_expr + ")";
        }
    }

    // VF section: pointer arithmetic (element offset), no tile descriptor.
    if (codegen.IsInVFSection()) {
        std::string base_ptr = codegen.GetOrCreateVFTilePtr(op->args_[0], /*is_post_update=*/false);
        if (!is_tuple) {
            return "(" + base_ptr + " + (" + codegen.GetExprAsCode(op->args_[1]) + "))";
        }
        return "(" + base_ptr + " + " + elem_off + ")";
    }

    std::string base_tile = codegen.GetExprAsCode(op->args_[0]);
    int elem_bytes = std::max(1, static_cast<int>(tile_type->dtype_.GetBit() / 8));
    std::string byte_offset_expr = "(" + elem_off + ")*" + std::to_string(elem_bytes);

    // Resolve base address: tile_addresses_ → .data() → memref addr
    std::string base_addr;
    if (codegen.HasTileAddress(base_tile)) {
        base_addr = codegen.GetTileAddress(base_tile);
    } else if (base_tile.find('[') != std::string::npos) {
        base_addr = "(uint64_t)" + base_tile + ".data()";
    } else {
        INTERNAL_CHECK(tile_type->memref_.has_value())
            << "block.subview: base tile '" << base_tile << "' has no address info";
        int64_t addr_val = codegen.GetConstIntValue((*tile_type->memref_)->addr_);
        std::ostringstream oss;
        oss << "0x" << std::hex << addr_val;
        base_addr = oss.str();
    }

    // Sub-window valid_shape from args[2] (computed by parser as the intersection
    // of slice size and original valid_shape - start).
    auto shape_tuple = ir::As<ir::MakeTuple>(op->args_[2]);
    std::string vs_row = codegen.GetExprAsCode(shape_tuple->elements_[0]);
    std::string vs_col = codegen.GetExprAsCode(shape_tuple->elements_[1]);

    // Build type with original shape (preserves row_stride) but WITHOUT tileView_,
    // so valid_shape template params are -1 (dynamic).  This is required because
    // subview calls SetValidShape at runtime, which only works when the template
    // valid_shape params are DYNAMIC (-1).
    std::vector<int64_t> dims;
    for (const auto& expr : tile_type->shape_) {
        dims.push_back(codegen.GetConstIntValue(expr));
    }
    int64_t tile_rows = dims.size() >= 1 ? dims[0] : 1;
    int64_t tile_cols = dims.size() >= 2 ? dims[1] : 1;
    auto subview_type = tile_type;
    std::string type_str = codegen.GetTypeConverter().ConvertTileType(subview_type, tile_rows, tile_cols);

    // Sanitize base_tile into a valid C++ identifier prefix
    std::string base_tile_id = base_tile;
    for (char& c : base_tile_id)
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_')
            c = '_';
    std::string temp_name = codegen.GetCurrentResultTarget().empty() ?
                                (base_tile_id + "_view_" + std::to_string(codegen.GetTileOffsetCounter())) :
                                codegen.GetCurrentResultTarget();

    // Emit: declare tile, TASSIGN offset address, then SetValidShape (must be
    // after TASSIGN — TASSIGN overwrites the constructor's valid_shape).
    std::string temp_addr = base_addr + " + " + byte_offset_expr;
    codegen.Emit(type_str + " " + temp_name + "(" + vs_row + ", " + vs_col + ");");
    codegen.Emit("TASSIGN(" + temp_name + ", " + temp_addr + ");");
    codegen.Emit(temp_name + ".SetValidShape(" + vs_row + ", " + vs_col + ");");
    codegen.SetTileAddress(temp_name, temp_addr);

    return temp_name;
}

REGISTER_BACKEND_OP(BackendCCE, "block.subview")
    .set_pipe(ir::PipeType::MTE2)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
        return MakeBlockSubviewCodegenCCE(op, codegen_base);
    });

// ============================================================================
// Mutex (Buffer-ID Token) - A5 CCE Codegen
// ----------------------------------------------------------------------------
// Lowers system.mutex_lock/unlock to CCE intrinsics get_buf/rls_buf.
// API: get_buf(PIPE_MTE2, mutexId, 0);  rls_buf(PIPE_MTE2, mutexId, 0);
// ============================================================================

static int GetMutexModeCCE(const ir::CallPtr& op)
{
    int mode = 0;
    for (const auto& [key, value] : op->kwargs_) {
        if (key == "mode")
            mode = std::any_cast<int>(value);
    }
    return mode;
}

static std::string MakeMutexBufCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base,
                                          const std::string& intrinsic, bool is_dynamic)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    auto pipe = static_cast<ir::PipeType>(op->GetKwarg<int>("pipe"));

    std::string static_mutex_id_expr;
    if (!is_dynamic) {
        auto mutex_id = op->GetKwarg<int>("mutex_id");
        static_mutex_id_expr = std::to_string(mutex_id);
    }

    bool auto_mutex = op->GetKwarg<bool>("auto_mutex", false);
    if (auto_mutex) {
        auto mutex_ids = mutex_id::GetMutexIdsFromKwargs(op);
        if (!mutex_ids.empty() && codegen.ShouldSkipVPipeMutex(pipe, mutex_ids))
            return "";
    }

    int mode = GetMutexModeCCE(op);
    std::string pipe_str = PipeTypeToCCEString(pipe);

    // N-way cross-Tile dedup: mutex IDs owned by one Tile are already known distinct.
    // Compare only across Tiles so each unique mutex_id is acquired and released once.
    // Without cross-Tile dedup, two get_buf(pipe, same_id) on the same pipe hang the hardware.
    if (is_dynamic && op->args_.size() >= 2) {
        std::vector<std::string> id_exprs;
        id_exprs.reserve(op->args_.size());
        for (const auto& arg : op->args_) {
            id_exprs.push_back(codegen.GetExprAsCode(arg));
        }
        // Backward-compatible default: without owner metadata every expression
        // has a separate owner, preserving the original all-pairs dedup.
        std::vector<int> mutex_id_owner_indices(id_exprs.size());
        for (size_t i = 0; i < mutex_id_owner_indices.size(); ++i) {
            mutex_id_owner_indices[i] = static_cast<int>(i);
        }
        for (const auto& [key, value] : op->kwargs_) {
            if (key == "mutex_id_owner_indices") {
                mutex_id_owner_indices = std::any_cast<std::vector<int>>(value);
                break;
            }
        }
        CHECK(mutex_id_owner_indices.size() == id_exprs.size())
            << "mutex_id_owner_indices size must match dynamic mutex args size";

        auto emit_id = [&](size_t i) {
            std::string condition;
            for (size_t j = 0; j < i; ++j) {
                // IDs owned by one Tile are guaranteed distinct by the frontend.
                if (mutex_id_owner_indices[i] == mutex_id_owner_indices[j])
                    continue;
                if (!condition.empty())
                    condition += " && ";
                condition += "(" + id_exprs[i] + " != " + id_exprs[j] + ")";
            }
            if (condition.empty()) {
                codegen.Emit(intrinsic + "(" + pipe_str + ", " + id_exprs[i] + ", " + std::to_string(mode) + ");");
                return;
            }
            codegen.Emit("if (" + condition + ") {");
            codegen.Emit("  " + intrinsic + "(" + pipe_str + ", " + id_exprs[i] + ", " + std::to_string(mode) + ");");
            codegen.Emit("}");
        };
        for (size_t i = 0; i < id_exprs.size(); ++i)
            emit_id(i);
        return "";
    }

    std::string mutex_id_expr = is_dynamic ? codegen.GetExprAsCode(op->args_[0]) : static_mutex_id_expr;
    codegen.Emit(intrinsic + "(" + pipe_str + ", " + mutex_id_expr + ", " + std::to_string(mode) + ");");
    return "";
}

static std::string MakeMutexLockCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return MakeMutexBufCodegenCCE(op, codegen_base, "get_buf", false);
}

static std::string MakeMutexUnlockCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return MakeMutexBufCodegenCCE(op, codegen_base, "rls_buf", false);
}

REGISTER_BACKEND_OP(BackendCCE, "system.mutex_lock")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeMutexLockCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "system.mutex_unlock")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeMutexUnlockCodegenCCE(op, codegen);
    });

static std::string MakeMutexLockDynCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return MakeMutexBufCodegenCCE(op, codegen_base, "get_buf", true);
}

static std::string MakeMutexUnlockDynCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return MakeMutexBufCodegenCCE(op, codegen_base, "rls_buf", true);
}

REGISTER_BACKEND_OP(BackendCCE, "system.mutex_lock_dyn")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeMutexLockDynCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "system.mutex_unlock_dyn")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeMutexUnlockDynCodegenCCE(op, codegen);
    });

// ============================================================================
// Global Core Synchronization (sync_all)
// ============================================================================
// Delegates to pto-isa SYNCALL<SyncAllMode, SyncCoreType>(...) / SYNCALL<SyncCoreType>().

static std::string MakeSystemSyncAllCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);

    auto mode = static_cast<ir::SyncAllMode>(op->HasKwarg("mode") ? op->GetKwarg<int>("mode") : 0);
    auto core_type = static_cast<ir::SyncCoreType>(op->HasKwarg("core_type") ? op->GetKwarg<int>("core_type") : 2);

    std::string core_type_tok;
    if (core_type == ir::SyncCoreType::AIV_ONLY)
        core_type_tok = "SyncCoreType::AIVOnly";
    else if (core_type == ir::SyncCoreType::AIC_ONLY)
        core_type_tok = "SyncCoreType::AICOnly";
    else
        core_type_tok = "SyncCoreType::Mix";

    if (mode == ir::SyncAllMode::HARD) {
        // args[0] is an empty MakeTuple for hard mode
        codegen.Emit("SYNCALL<" + core_type_tok + ">();");
        return "";
    }

    // Soft mode: args[0] is a MakeTuple with workspace elements, dispatch by type
    CHECK(op->args_.size() == 1) << "system.sync_all expects 1 arg (workspaces tuple)";
    auto tuple = ir::As<ir::MakeTuple>(op->args_[0]);
    CHECK(tuple) << "system.sync_all: workspaces must be a tuple";

    std::string gm, ub, l1, used_cores = "0";
    for (const auto& elem : tuple->elements_) {
        auto elem_type = elem->GetType();
        if (ir::As<ir::TensorType>(elem_type)) {
            gm = codegen.GetExprAsCode(elem);
        } else if (auto tile_type = ir::As<ir::TileType>(elem_type)) {
            auto space = tile_type->memref_.value()->memorySpace_;
            if (space == ir::MemorySpace::Vec) {
                ub = codegen.GetExprAsCode(elem);
            } else if (space == ir::MemorySpace::Mat) {
                l1 = codegen.GetExprAsCode(elem);
            }
        } else if (ir::As<ir::ScalarType>(elem_type)) {
            used_cores = codegen.GetExprAsCode(elem);
        }
    }

    std::ostringstream oss;
    oss << "SYNCALL<SyncAllMode::Soft, " << core_type_tok << ">(" << gm;
    if (core_type == ir::SyncCoreType::AIV_ONLY) {
        oss << ", " << ub;
    } else if (core_type == ir::SyncCoreType::AIC_ONLY) {
        oss << ", " << l1;
    } else { // MIX
        oss << ", " << ub << ", " << l1;
    }
    oss << ", " << used_cores << ");";
    codegen.Emit(oss.str());
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "system.sync_all")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeSystemSyncAllCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "system.set_mm_layout_transform")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
        auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
        auto enabled = op->GetKwarg<int>("enabled");
        if (codegen.GetArch() == "a5") {
            // Direct register manipulation: MM_LAYOUT_MODE_BIT = 51
            if (enabled) {
                codegen.Emit("set_ctrl(sbitset1(get_ctrl(), 51));");
            } else {
                codegen.Emit("set_ctrl(sbitset0(get_ctrl(), 51));");
            }
        }
        return "";
    });

} // namespace backend
} // namespace pypto
