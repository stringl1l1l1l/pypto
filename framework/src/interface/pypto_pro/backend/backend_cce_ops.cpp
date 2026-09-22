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
#include "ir/type_inference.h"
#include "pypto_pro/error.h"
#include "tilefwk/error.h"

namespace pypto {
namespace backend {
using ir::DataType;
using npu::tile_fwk::ExternalError;

// ============================================================================
// Helper Functions for CCE Code Generation
// ============================================================================

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
        codegen.Emit("pypto_printf(\"" + debug_printf::EscapeStringLiteral(header + "\n") + "\");");
    }
}

static bool NeedsAscPrintfSignedLongLong(const DataType& dtype, char conversion)
{
    return (conversion == 'd' || conversion == 'i') && (dtype == DataType::INT64 || dtype == DataType::INDEX);
}

static bool NeedsAscPrintfUnsignedLongLong(const DataType& dtype, char conversion)
{
    return (conversion == 'u' || conversion == 'x') && (dtype == DataType::UINT64 || dtype == DataType::INDEX);
}

static std::string RewriteAscPrintfFormatForScalarType(const std::string& format_segment, char conversion,
                                                       const DataType& dtype)
{
    const bool signed_long_long = NeedsAscPrintfSignedLongLong(dtype, conversion);
    const bool unsigned_long_long = NeedsAscPrintfUnsignedLongLong(dtype, conversion);
    if (!signed_long_long && !unsigned_long_long) {
        return format_segment;
    }

    size_t conv_idx = debug_printf::FindPrintfConversionIndex(format_segment);
    std::string rewritten = format_segment;
    if (conversion == 'd') {
        rewritten.replace(conv_idx, 1, "lld");
    } else if (conversion == 'i') {
        rewritten.replace(conv_idx, 1, "lli");
    } else if (conversion == 'u') {
        rewritten.replace(conv_idx, 1, "llu");
    } else {
        rewritten.replace(conv_idx, 1, "llx");
    }
    return rewritten;
}

static std::string CastAscPrintfArgIfNeeded(const std::string& arg, const DataType& dtype, char conversion)
{
    if (conversion == 'f') {
        return arg;
    }
    if (NeedsAscPrintfSignedLongLong(dtype, conversion)) {
        return "static_cast<long long>(" + arg + ")";
    }
    if (NeedsAscPrintfUnsignedLongLong(dtype, conversion)) {
        return "static_cast<unsigned long long>(" + arg + ")";
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

static std::string BuildAscPrintfCall(const std::string& format, const std::vector<std::string>& args,
                                      const std::vector<DataType>& arg_dtypes)
{
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == arg_dtypes.size())
        << "debug.printf ASC argument/type count mismatch";

    auto segments = debug_printf::ParsePrintfSegments(format);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, segments.size() == args.size())
        << "debug.printf format expects " << segments.size() << " scalar arguments, but got " << args.size();

    if (format.empty()) {
        return "";
    }

    std::string rewritten_format = segments.empty() ? format : "";
    std::vector<std::string> rewritten_args;
    for (size_t i = 0; i < segments.size(); ++i) {
        rewritten_format += RewriteAscPrintfFormatForScalarType(segments[i].format_segment, segments[i].conversion,
                                                                arg_dtypes[i]);
        rewritten_args.push_back(CastAscPrintfArgIfNeeded(args[i], arg_dtypes[i], segments[i].conversion));
    }

    std::string call = "pypto_printf(\"" + debug_printf::EscapeStringLiteral(rewritten_format) + "\"";
    for (const auto& arg : rewritten_args) {
        call += ", " + arg;
    }
    return call + ");";
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
    PRO_CODEGEN_CHECK(ExternalError::INVALID_SHAPE, shape_exprs.size() >= 1 && shape_exprs.size() <= 5)
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
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, tensor_type)
        << "debug.dump_tensor requires TensorType for stride generation";
    const size_t rank = tensor_type->shape_.size();
    PRO_CODEGEN_CHECK(ExternalError::INVALID_SHAPE, rank >= 1 && rank <= 5)
        << "debug.dump_tensor currently supports tensor rank 1..5, but got " << rank;

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
        PRO_CODEGEN_CHECK(ExternalError::INVALID_SHAPE, strides.size() == rank)
            << "debug.dump_tensor tensor_view stride rank (" << strides.size() << ") must match tensor rank (" << rank
            << ")";
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
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, tensor_type)
        << "debug.dump_tensor requires TensorType for runtime offset generation";
    const size_t rank = tensor_type->shape_.size();
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, offsets)
        << "debug.dump_tensor requires offsets tuple for runtime offset generation";
    PRO_CODEGEN_CHECK(ExternalError::INVALID_SHAPE, offsets->elements_.size() == rank)
        << "debug.dump_tensor offset rank (" << offsets->elements_.size() << ") must match tensor rank (" << rank
        << ")";

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

static std::string MakeDebugDumpTensorNZCodegenCCE(codegen::CCECodegen& codegen, const ir::VarPtr& tensor_var,
                                                   const ir::TensorTypePtr& tensor_type,
                                                   const ir::MakeTuplePtr& offsets_tuple,
                                                   const ir::MakeTuplePtr& shapes_tuple)
{
    const size_t ndim = tensor_type->shape_.size();
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, ndim >= 2)
        << "debug.dump_tensor NZ lowering requires a tensor rank of at least 2";
    const size_t row_axis = ndim - 2;
    const size_t col_axis = ndim - 1;

    const int64_t c0 = cce::GetNZInnerCols(tensor_type->dtype_);
    const auto known_col_offset = ir::GetConstantDimension(offsets_tuple->elements_[col_axis]);
    const auto known_window_rows = ir::GetConstantDimension(shapes_tuple->elements_[row_axis]);
    const auto known_window_cols = ir::GetConstantDimension(shapes_tuple->elements_[col_axis]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_FORMAT,
                      !known_window_rows.has_value() || known_window_rows.value() % 16 == 0)
        << "debug.dump_tensor: NZ window rows must be divisible by 16";
    PRO_CODEGEN_CHECK(ExternalError::INVALID_FORMAT,
                      !known_window_cols.has_value() || known_window_cols.value() % c0 == 0)
        << "debug.dump_tensor: NZ window columns must be divisible by C0";
    PRO_CODEGEN_CHECK(ExternalError::INVALID_FORMAT,
                      !known_col_offset.has_value() || known_col_offset.value() % c0 == 0)
        << "debug.dump_tensor: NZ column offset must be divisible by C0";

    const int debug_id = NextDebugDumpId();
    const std::string tensor_name = codegen.GetVarName(tensor_var);
    std::string base_ptr = codegen.GetPointer(tensor_name);
    if (base_ptr.empty()) {
        base_ptr = tensor_name + ".data()";
    }

    const std::string padded_rows = codegen.ComputeAlignedShapeDimension(tensor_type->shape_[row_axis], 16);
    const std::string padded_cols = codegen.ComputeAlignedShapeDimension(tensor_type->shape_[col_axis], c0);
    const std::string rows = codegen.GetExprAsCode(shapes_tuple->elements_[row_axis]);
    const std::string cols = codegen.GetExprAsCode(shapes_tuple->elements_[col_axis]);
    const std::string offset = codegen.ComputeTensorOffset(tensor_type, offsets_tuple);

    const std::string dtype = codegen.GetTypeString(tensor_type->dtype_);
    const std::string shape_alias = "__debug_dump_tensor_shape_" + std::to_string(debug_id);
    const std::string stride_alias = "__debug_dump_tensor_stride_" + std::to_string(debug_id);
    const std::string global_alias = "__debug_dump_tensor_type_" + std::to_string(debug_id);
    const std::string view_name = "__debug_dump_tensor_view_" + std::to_string(debug_id);

    codegen.Emit("using " + shape_alias + " = pto::TileShape2D<" + dtype +
                 ", pto::DYNAMIC, pto::DYNAMIC, Layout::NZ>;");
    codegen.Emit("using " + stride_alias + " = pto::BaseShape2D<" + dtype +
                 ", pto::DYNAMIC, pto::DYNAMIC, Layout::NZ>;");
    codegen.Emit("using " + global_alias + " = GlobalTensor<" + dtype + ", " + shape_alias + ", " + stride_alias +
                 ", Layout::NZ>;");
    std::vector<std::string> batch_indices;
    for (size_t axis = 0; axis < row_axis; ++axis) {
        const std::string index = "__debug_dump_tensor_batch_" + std::to_string(debug_id) + "_" + std::to_string(axis);
        batch_indices.push_back(index);
        codegen.Emit("for (int " + index + " = 0; " + index + " < " +
                     codegen.GetExprAsCode(shapes_tuple->elements_[axis]) + "; ++" + index + ") {");
    }

    std::string view_offset = offset;
    if (!batch_indices.empty()) {
        std::ostringstream batch_delta;
        for (size_t axis = 0; axis < row_axis; ++axis) {
            if (axis != 0) {
                batch_delta << " + ";
            }
            batch_delta << batch_indices[axis];
            for (size_t inner = axis + 1; inner < row_axis; ++inner) {
                batch_delta << " * " << codegen.GetExprAsCode(tensor_type->shape_[inner]);
            }
            batch_delta << " * " << padded_rows << " * " << padded_cols;
        }
        const std::string delta = "(" + batch_delta.str() + ")";
        view_offset = "(" + offset + " + " + delta + ")";

        std::ostringstream batch_header;
        batch_header << "pypto_printf(\"=== [dump_tensor] Batch [";
        for (size_t axis = 0; axis < row_axis; ++axis) {
            if (axis != 0) {
                batch_header << ", ";
            }
            batch_header << "%d";
        }
        batch_header << "] ===\\n\"";
        for (size_t axis = 0; axis < row_axis; ++axis) {
            batch_header << ", static_cast<int>(" << codegen.GetExprAsCode(offsets_tuple->elements_[axis]) << " + "
                         << batch_indices[axis] << ")";
        }
        batch_header << ");";
        codegen.Emit(batch_header.str());
    }

    codegen.Emit(global_alias + " " + view_name + "(" + base_ptr + " + " + view_offset + ", " + shape_alias + "(" +
                 rows + ", " + cols + "), " + stride_alias + "(" + padded_rows + ", " + padded_cols + "));");
    codegen.Emit("TPRINT(" + view_name + ");");
    for (size_t axis = 0; axis < row_axis; ++axis) {
        codegen.Emit("}");
    }
    return "";
}

static void EmitDumpFlagHeaderCCE(codegen::CCECodegen& codegen, const ir::CallPtr& op)
{
    std::string dump_flag = op->HasKwarg("dump_flag") ? op->GetKwarg<std::string>("dump_flag") : "";
    if (dump_flag.empty()) {
        return;
    }
    codegen.Emit("pypto_printf(\"=== [flag] %s ===\\n\", \"" + debug_printf::EscapeStringLiteral(dump_flag) + "\");");
}

static std::string MakeDebugDumpTensorCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "debug.dump_tensor requires 3 arguments, but got " << op->args_.size();
    if (op->GetKwarg<bool>("show_location", false)) {
        EmitDebugLocationHeaderCCE(codegen, op->span_, "dump_tensor");
    }
    EmitDumpFlagHeaderCCE(codegen, op);

    auto tensor_var = ir::As<ir::Var>(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, tensor_var) << "debug.dump_tensor first argument must be a Var";
    auto tensor_type = ir::As<ir::TensorType>(tensor_var->GetType());
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, tensor_type)
        << "debug.dump_tensor first argument must be TensorType";
    auto offsets_tuple = ir::As<ir::MakeTuple>(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, offsets_tuple)
        << "debug.dump_tensor second argument must be a tuple (offsets)";
    auto shapes_tuple = ir::As<ir::MakeTuple>(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, shapes_tuple)
        << "debug.dump_tensor third argument must be a tuple (shapes)";

    if (cce::IsNZTensorType(tensor_type)) {
        return MakeDebugDumpTensorNZCodegenCCE(codegen, tensor_var, tensor_type, offsets_tuple, shapes_tuple);
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
                                        codegen.ComputeTensorOffset(tensor_type, offsets_tuple);

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

    for (size_t i = 0; i < op->args_.size(); ++i) {
        args.emplace_back(codegen.GetExprAsCode(op->args_[i]));
        if (ir::As<ir::PtrType>(op->args_[i]->GetType())) {
            // Pointer arguments are passed directly to ASC printf.
            arg_dtypes.emplace_back(DataType::INT64);
        } else {
            auto scalar_type = ir::As<ir::ScalarType>(op->args_[i]->GetType());
            PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, scalar_type)
                << "debug.printf argument must be ScalarType in CCE lowering";
            arg_dtypes.emplace_back(scalar_type->dtype_);
        }
    }

    std::string call = BuildAscPrintfCall(format, args, arg_dtypes);
    if (!call.empty()) {
        codegen.Emit(call);
    }
    return "";
}

static std::string MakeDebugDumpTileCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT,
                      op->args_.size() == 1 || op->args_.size() == 3 || op->args_.size() == 4)
        << "debug.dump_tile requires 1 argument (tile), 3 arguments (tile, offsets, shapes), "
        << "or 4 arguments (tile, offsets, shapes, workspace), but got " << op->args_.size();
    if (op->GetKwarg<bool>("show_location", false)) {
        EmitDebugLocationHeaderCCE(codegen, op->span_, "dump_tile");
    }
    EmitDumpFlagHeaderCCE(codegen, op);

    std::string src = codegen.GetExprAsCode(op->args_[0]);

    if (op->args_.size() == 4) {
        auto tile_type = ir::As<ir::TileType>(op->args_[0]->GetType());
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, tile_type) << "debug.dump_tile first argument must be TileType";
        PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, tile_type->shape_.size() == 2)
            << "debug.dump_tile Acc window dump only supports 2D tiles";
        auto workspace_var = ir::As<ir::Var>(op->args_[3]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, workspace_var)
            << "debug.dump_tile workspace (4th argument) must be a Var";
        std::string workspace_name = codegen.GetVarName(workspace_var);
        std::string workspace_ptr = codegen.GetPointer(workspace_name);

        auto tile_rows = ir::As<ir::ConstInt>(tile_type->shape_[0]);
        auto tile_cols = ir::As<ir::ConstInt>(tile_type->shape_[1]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_SHAPE, tile_rows && tile_cols)
            << "debug.dump_tile Acc dump requires static physical tile shape";

        auto offsets_tuple = ir::As<ir::MakeTuple>(op->args_[1]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, offsets_tuple)
            << "debug.dump_tile second argument must be a tuple (offsets)";
        auto shapes_tuple = ir::As<ir::MakeTuple>(op->args_[2]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, shapes_tuple)
            << "debug.dump_tile third argument must be a tuple (shapes)";

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

        codegen.Emit("pypto_printf(\"=== [TPRINT Acc Tile Window] Data Type: %s, Layout: NZ, TileType: Acc "
                     "===\\n\", "
                     "__pypto_dtype_name<" +
                     dtype_str + ">());");
        codegen.Emit("pypto_printf(\"  Source Shape: [%d, %d], Window Offsets: [%d, %d], Requested Shape: "
                     "[%d, %d], "
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
        codegen.Emit("    pypto_printf(\"\\n\");");
        codegen.Emit("  }");
        codegen.Emit("}");
        return "";
    }

    if (op->args_.size() == 1) {
        codegen.Emit("TPRINT(" + src + ");");
        return "";
    }

    auto tile_type = ir::As<ir::TileType>(op->args_[0]->GetType());
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, tile_type) << "debug.dump_tile first argument must be TileType";
    PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, tile_type->shape_.size() == 2)
        << "debug.dump_tile CCE lowering currently only supports 2D tiles";
    auto offsets_tuple = ir::As<ir::MakeTuple>(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, offsets_tuple)
        << "debug.dump_tile second argument must be a tuple (offsets)";
    auto shapes_tuple = ir::As<ir::MakeTuple>(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, shapes_tuple)
        << "debug.dump_tile third argument must be a tuple (shapes)";

    auto tile_rows = ir::As<ir::ConstInt>(tile_type->shape_[0]);
    auto tile_cols = ir::As<ir::ConstInt>(tile_type->shape_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_SHAPE, tile_rows && tile_cols)
        << "debug.dump_tile CCE lowering requires static physical tile shape";

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
    codegen.Emit("pypto_printf(\"=== [TPRINT Tile Window] Data Type: %s, Layout: %s, TileType: %s ===\\n\", "
                 "__pypto_dtype_name<" +
                 codegen.GetTypeString(tile_type->dtype_) +
                 ">(), pto::GetLayoutName(std::remove_reference_t<decltype(" + src +
                 ")>::BFractal, std::remove_reference_t<decltype(" + src + ")>::SFractal), \"Vec\");");
    codegen.Emit("pypto_printf(\"  Source Shape: [%d, %d], Window Offsets: [%d, %d], Requested Shape: "
                 "[%d, %d], "
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
    codegen.Emit("  pypto_printf(\"\\n\");");
    codegen.Emit("}");
    return "";
}

// Helper function for get_block_idx (returns value expression).
// Matches AscendC GetBlockIdx(): AIV returns global AIV index, AIC returns AIC index.
static std::string MakeBlockGetBlockIdxCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 0) << "get_block_idx requires no arguments";
    auto& cg = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const auto target = cg.GetTarget();
    if (target == ir::SectionKind::Vector) {
        return "(int64_t)(get_block_idx() * get_subblockdim() + get_subblockid())";
    }
    return "(int64_t)(get_block_idx())";
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
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_var != nullptr)
            << "ptr.make_tensor from a tensor requires the source to be a tensor variable";
        const std::string src_name = cg.GetVarName(src_var);
        ptr_code = cg.HasPointer(src_name) ? cg.GetPointer(src_name) : (src_name + ".data()");
        source_dtype = src_tensor_type->dtype_;
    } else {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false)
            << "ptr.make_tensor source must be a PtrType or TensorType";
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
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2)
        << "ptr.addptr requires 2 arguments: ptr, offset";
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
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 1)
        << "ptr.make_ptr requires 1 argument: ptr";
    std::string ptr;
    if (auto ptr_type = ir::As<ir::PtrType>(op->args_[0]->GetType())) {
        ptr = codegen.GetExprAsCode(op->args_[0]);
    } else if (ir::As<ir::TensorType>(op->args_[0]->GetType())) {
        auto src_var = std::dynamic_pointer_cast<const ir::Var>(op->args_[0]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_var != nullptr)
            << "ptr.make_ptr from a tensor requires the source to be a tensor variable";
        const std::string src_name = codegen.GetVarName(src_var);
        ptr = codegen.HasPointer(src_name) ? codegen.GetPointer(src_name) : (src_name + ".data()");
    } else {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false) << "ptr.make_ptr source must be a PtrType or TensorType";
    }
    auto result_ptr_type = ir::As<ir::PtrType>(op->GetType());
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, result_ptr_type != nullptr)
        << "ptr.make_ptr result must be a PtrType";
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
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 0) << "get_spr requires no arguments";
    return "get_ar()";
}

REGISTER_BACKEND_OP(BackendCCE, "get_spr")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return MakeGetSprCodegenCCE(op, codegen); });

// ============================================================================
// Saturation flag (CTRL register bit manipulation)
// ============================================================================

static int8_t GetSaturationModeBit(ir::SaturationFlagMode mode, const std::string& op_name)
{
    switch (mode) {
        case ir::SaturationFlagMode::FLOAT:
            return 48;
        case ir::SaturationFlagMode::FLOAT8:
            return 50;
        case ir::SaturationFlagMode::INT:
            return 53;
        case ir::SaturationFlagMode::CAST:
            return 59;
        default:
            PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, false) << op_name << ": unsupported mode";
            return -1;
    }
}

// FLOAT/FLOAT8/CAST: polarity inverted (bit=0 means saturation ON, bit=1 means OFF)
// INT: polarity normal (bit=1 means saturation ON, bit=0 means OFF)
static bool IsInvertedPolarity(ir::SaturationFlagMode mode)
{
    return mode == ir::SaturationFlagMode::FLOAT || mode == ir::SaturationFlagMode::FLOAT8 ||
           mode == ir::SaturationFlagMode::CAST;
}

static std::string MakeSetSaturationFlagCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->HasKwarg("mode"))
        << "set_saturation_flag requires 'mode' kwarg";
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->HasKwarg("enable"))
        << "set_saturation_flag requires 'enable' kwarg";
    auto mode = static_cast<ir::SaturationFlagMode>(op->GetKwarg<int>("mode"));
    int8_t bit = GetSaturationModeBit(mode, "set_saturation_flag");
    bool inverted = IsInvertedPolarity(mode);

    bool enable = op->GetKwarg<bool>("enable");

    // For inverted polarity: enable=true → set bit to 0 (sbitset0), enable=false → set bit to 1 (sbitset1)
    // For normal polarity (INT): enable=true → set bit to 1 (sbitset1), enable=false → set bit to 0 (sbitset0)
    std::string set_fn;
    if (inverted) {
        set_fn = enable ? "sbitset0" : "sbitset1";
    } else {
        set_fn = enable ? "sbitset1" : "sbitset0";
    }
    codegen.Emit("set_ctrl(" + set_fn + "(get_ctrl(), " + std::to_string(bit) + "));");
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "set_saturation_flag")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeSetSaturationFlagCodegenCCE(op, codegen);
    });

static std::string MakeGetSaturationFlagCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    (void)codegen_base;
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->HasKwarg("mode"))
        << "get_saturation_flag requires 'mode' kwarg";
    auto mode = static_cast<ir::SaturationFlagMode>(op->GetKwarg<int>("mode"));
    int8_t bit = GetSaturationModeBit(mode, "get_saturation_flag");
    bool inverted = IsInvertedPolarity(mode);

    // Read CTRL bit, invert for FLOAT/FLOAT8/CAST modes
    if (inverted) {
        return "((get_ctrl() >> " + std::to_string(bit) + ") & 1) == 0";
    } else {
        return "((get_ctrl() >> " + std::to_string(bit) + ") & 1) != 0";
    }
}

REGISTER_BACKEND_OP(BackendCCE, "get_saturation_flag")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeGetSaturationFlagCodegenCCE(op, codegen);
    });

// ============================================================================
// CTRL SPR direct access (SetCtrlSpr / GetCtrlSpr / ResetCtrlSpr)
// ============================================================================

// A5 writable CTRL bits: 6-10 (range), 45, 48, 50, 53, 59, 60 (single bits)
static void CheckCtrlBitRange(int8_t startBit, int8_t endBit, const std::string& op_name)
{
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, startBit >= 0 && endBit < 64 && startBit <= endBit)
        << op_name << ": invalid bit range [" << static_cast<int>(startBit) << ", " << static_cast<int>(endBit)
        << "], must be 0 <= startBit <= endBit < 64";
    bool valid = (6 <= startBit && startBit <= 10 && 6 <= endBit && endBit <= 10) ||
                 (startBit == endBit && (startBit == 45 || startBit == 48 || startBit == 50 || startBit == 53 ||
                                         startBit == 59 || startBit == 60));
    PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, valid)
        << op_name << ": bits [" << static_cast<int>(startBit) << ", " << static_cast<int>(endBit)
        << "] are not writable on current device. Writable: bits 6-10, 45, 48, 50, 53, 59, 60";
}

static std::string MakeSetCtrlSprCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "set_ctrl_spr requires 3 args (start_bit, end_bit, value)";
    auto start_val = ir::As<ir::ConstInt>(op->args_[0]);
    auto end_val = ir::As<ir::ConstInt>(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_VAL, start_val != nullptr && end_val != nullptr)
        << "set_ctrl_spr: start_bit and end_bit must be compile-time constants";
    int8_t startBit = static_cast<int8_t>(start_val->value_);
    int8_t endBit = static_cast<int8_t>(end_val->value_);
    CheckCtrlBitRange(startBit, endBit, "set_ctrl_spr");
    auto value_const = ir::As<ir::ConstInt>(op->args_[2]);
    if (value_const != nullptr && startBit >= 6 && endBit <= 10) {
        // CTRL[8:6]=3'b111 (atomic operand dtype) and CTRL[10:9]=2'b11 (atomic op
        // type) are undefined encodings that fault the device — reject up front.
        int64_t v = value_const->value_ & ((int64_t(1) << (endBit - startBit + 1)) - 1);
        if (startBit <= 6 && endBit >= 8 && ((v >> (6 - startBit)) & 0x7) == 0x7) {
            PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false)
                << "set_ctrl_spr: value " << value_const->value_
                << " sets CTRL[8:6]=3'b111, an undefined atomic operand dtype "
                << "(supported: 0-6 = none/float/half/int16/int32/int8/bfloat16)";
        }
        if (startBit <= 9 && endBit >= 10 && ((v >> (9 - startBit)) & 0x3) == 0x3) {
            PRO_CODEGEN_CHECK(ExternalError::NAME_ERROR, false)
                << "set_ctrl_spr: value " << value_const->value_
                << " sets CTRL[10:9]=2'b11, an undefined atomic op type " << "(supported: 0-2 = ADD/MAX/MIN)";
        }
    }
    std::string value = codegen.GetExprAsCode(op->args_[2]);
    if (endBit - startBit == 63) {
        codegen.Emit("set_ctrl(" + value + ");");
    } else {
        codegen.Emit("set_ctrl((get_ctrl() & ~(((uint64_t(1) << " + std::to_string(endBit - startBit + 1) +
                     ") - 1) << " + std::to_string(startBit) + ")) | ((" + value + " & ((uint64_t(1) << " +
                     std::to_string(endBit - startBit + 1) + ") - 1)) << " + std::to_string(startBit) + "));");
    }
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "set_ctrl_spr")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeSetCtrlSprCodegenCCE(op, codegen);
    });

static std::string MakeGetCtrlSprCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    (void)codegen_base;
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2)
        << "get_ctrl_spr requires 2 args (start_bit, end_bit)";
    auto start_val = ir::As<ir::ConstInt>(op->args_[0]);
    auto end_val = ir::As<ir::ConstInt>(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_VAL, start_val != nullptr && end_val != nullptr)
        << "get_ctrl_spr: start_bit and end_bit must be compile-time constants";
    int8_t startBit = static_cast<int8_t>(start_val->value_);
    int8_t endBit = static_cast<int8_t>(end_val->value_);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, startBit >= 0 && endBit < 64 && startBit <= endBit)
        << "get_ctrl_spr: invalid bit range [" << static_cast<int>(startBit) << ", " << static_cast<int>(endBit) << "]";
    if (endBit - startBit == 63) {
        return "get_ctrl()";
    }
    return "(get_ctrl() >> " + std::to_string(startBit) + ") & ((uint64_t(1) << " +
           std::to_string(endBit - startBit + 1) + ") - 1)";
}

REGISTER_BACKEND_OP(BackendCCE, "get_ctrl_spr")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeGetCtrlSprCodegenCCE(op, codegen);
    });

static std::string MakeResetCtrlSprCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2)
        << "reset_ctrl_spr requires 2 args (start_bit, end_bit)";
    auto start_val = ir::As<ir::ConstInt>(op->args_[0]);
    auto end_val = ir::As<ir::ConstInt>(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_VAL, start_val != nullptr && end_val != nullptr)
        << "reset_ctrl_spr: start_bit and end_bit must be compile-time constants";
    int8_t startBit = static_cast<int8_t>(start_val->value_);
    int8_t endBit = static_cast<int8_t>(end_val->value_);
    CheckCtrlBitRange(startBit, endBit, "reset_ctrl_spr");
    constexpr int64_t defaultCtrl = 0x1000000000000008LL;
    if (endBit - startBit == 63) {
        codegen.Emit("set_ctrl(" + std::to_string(defaultCtrl) + "LL);");
    } else {
        uint64_t mask = ((uint64_t(1) << (endBit - startBit + 1)) - 1) << startBit;
        int64_t defaultBits = defaultCtrl & static_cast<int64_t>(mask);
        codegen.Emit("set_ctrl((get_ctrl() & ~(((uint64_t(1) << " + std::to_string(endBit - startBit + 1) +
                     ") - 1) << " + std::to_string(startBit) + ")) | " + std::to_string(defaultBits) + "LL);");
    }
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "reset_ctrl_spr")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeResetCtrlSprCodegenCCE(op, codegen);
    });

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

    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, ndim == 2)
        << "Codegen only supports 2D tiles, but got " << ndim << "D tile";
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, axis1 != axis2)
        << "tile.transpose: axis1 and axis2 must be different, but got axis1=axis2=" << axis1;
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR,
                               axis1 >= 0 && axis1 < ndim && axis2 >= 0 && axis2 < ndim)
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
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 1 || op->args_.size() == 2)
        << "system.dcci requires 1 or 2 arguments, got " << op->args_.size();

    int cache_line_int = op->HasKwarg("cache_line") ? op->GetKwarg<int>("cache_line") : 1; // ENTIRE_DATA_CACHE
    int dst_int = op->HasKwarg("dst") ? op->GetKwarg<int>("dst") : 0;                      // AUTO

    const std::string cache_line = NormalizeDcciCacheLine(cache_line_int);

    auto tensor_type = ir::As<ir::TensorType>(op->args_[0]->GetType());
    if (tensor_type != nullptr) {
        auto tensor_var_ptr = std::dynamic_pointer_cast<const ir::Var>(op->args_[0]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, tensor_var_ptr != nullptr)
            << "system.dcci: tensor target must be a Var";
        std::string tensor_var = codegen.GetVarName(tensor_var_ptr);
        std::string offset = "0";
        if (op->args_.size() == 2) {
            auto offsets_tuple = std::dynamic_pointer_cast<const ir::MakeTuple>(op->args_[1]);
            if (offsets_tuple != nullptr) {
                offset = codegen.ComputeTensorOffset(tensor_type, offsets_tuple);
            } else {
                PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                                  ir::As<ir::ScalarType>(op->args_[1]->GetType()) != nullptr)
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
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, tile_type != nullptr)
        << "system.dcci: target must be TensorType or TileType";
    PRO_CODEGEN_CHECK(ExternalError::INVALID_VAL, tile_type->memref_.has_value())
        << "system.dcci: tile target must have an allocated memory space";
    if (tile_type->memref_.value()->memorySpace_ != ir::MemorySpace::Vec) {
        PRO_CODEGEN_THROW(::pypto::ir::ValueError, ExternalError::INVALID_OPERATION)
            << "system.dcci: tile target must be allocated in Vec memory";
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
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2)
            << "system.set_vec_mask requires 2 arguments, but got " << op->args_.size();
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
// Flag synchronization operations
// ============================================================================

static std::string MakeSyncCodegenCCE(const std::string& intrinsic, const ir::CallPtr& op,
                                      codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    auto set_pipe = static_cast<ir::PipeType>(op->GetKwarg<int>("set_pipe"));
    auto wait_pipe = static_cast<ir::PipeType>(op->GetKwarg<int>("wait_pipe"));
    std::string set_pipe_str = PipeTypeToCCEString(set_pipe);
    std::string wait_pipe_str = PipeTypeToCCEString(wait_pipe);
    std::string event_id = codegen.GetExprAsCode(op->args_[0]);
    codegen.Emit(intrinsic + "(" + set_pipe_str + ", " + wait_pipe_str + ", (event_t)" + event_id + ");");
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "system.sync_src_dyn")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeSyncCodegenCCE("set_flag", op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "system.sync_dst_dyn")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeSyncCodegenCCE("wait_flag", op, codegen);
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

    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() >= 1)
        << "debug.assert requires at least 1 argument (condition)";

    std::string condition = codegen.GetExprAsCode(op->args_[0]);
    std::string condition_text = op->GetKwarg<std::string>("condition_text");
    std::string format = op->GetKwarg<std::string>("format");

    codegen.Emit("if (!(" + condition + ")) {");

    if (op->GetKwarg<bool>("show_location", false)) {
        std::string location = debug_printf::FormatDebugLocation(op->span_);
        if (!location.empty()) {
            codegen.Emit("  pypto_printf(\"" +
                         debug_printf::EscapeStringLiteral(location + " Assertion failed: " + condition_text + "\n") +
                         "\");");
        } else {
            codegen.Emit("  pypto_printf(\"" +
                         debug_printf::EscapeStringLiteral("Assertion failed: " + condition_text + "\n") + "\");");
        }
    } else {
        codegen.Emit("  pypto_printf(\"" +
                     debug_printf::EscapeStringLiteral("Assertion failed: " + condition_text + "\n") + "\");");
    }

    if (!format.empty()) {
        std::vector<std::string> args;
        std::vector<DataType> arg_dtypes;
        for (size_t i = 1; i < op->args_.size(); ++i) {
            args.emplace_back(codegen.GetExprAsCode(op->args_[i]));
            auto scalar_type = ir::As<ir::ScalarType>(op->args_[i]->GetType());
            PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, scalar_type) << "debug.assert argument must be ScalarType";
            arg_dtypes.emplace_back(scalar_type->dtype_);
        }
        codegen.Emit("  " + BuildAscPrintfCall(format, args, arg_dtypes));
    }

    codegen.Emit("}");
    return "";
}

static std::string MakeDebugTrapCodegenCCE(const ir::CallPtr& /*op*/, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    codegen.Emit("trap();");
    return "";
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
// Sanitizer log record - device-side write of one per-detection variable-size
// record into the GM log buffer passed as the trailing `sanitizer_log`
// parameter. One op covers every detection type: the leading det_id argument
// selects the record layout on the host side (pypto_pro.runtime.
// sanitizer_replay decodes by det_id); the trailing three arguments are the
// hidden log-buffer pointer / capacity and the span id.
//   args = [det_id, field..., log_ptr, capacity, span_id]
//   record = [det_id, field...] (rec_u32 = args.size() - 3)
// The buffer region per sub-block is [ctr u32, records...], where ctr is the
// region's u32-word offset past itself and the capacity
// (`sanitizer_log_capacity` scalar) is the per-region u32 word budget for
// records.
// ============================================================================

namespace {

void EmitSanitizerLogPrologue(codegen::CCECodegen& codegen, const std::string& log, const std::string& capacity)
{
    // Region addressing. A mixed kernel runs the Cube part on the AICs and
    // the Vector part on the AIVs of the SAME launch, so both parts share
    // one log buffer and their region indices must never overlap (the
    // append counter is a plain read-modify-write; two writers on one
    // region silently drop records). The Vector part is therefore offset by
    // the block count: Cube covers [0, block_num), Vector covers
    // [block_num, block_num + block_num * subblockdim) = at most [N, 3N),
    // which the host allocation of 4 * block_dim regions holds. A
    // vector-only kernel wastes the first block_num regions, which is safe.
    // Sub-block addressing follows the established get_block_idx() backend-op
    // contract: the Vector target computes the global AIV index, so each
    // sub-block gets its own region even when several AIVs run under one
    // AIC block.
    std::string block_idx;
    if (codegen.GetTarget() == ir::SectionKind::Vector) {
        block_idx = "(int32_t)(get_block_num() + get_block_idx() * get_subblockdim() + get_subblockid())";
    } else {
        block_idx = "(int32_t)(get_block_idx())";
    }

    // Append-mode logging, no atomics: each executing sub-block targets its own
    // region (independently addressable via the global block index above), so
    // the leading counter is only written by this single writer. This keeps
    // every loop iteration's record (no overwrite) without relying on a GM
    // atomic instruction. Region layout: [ctr u32, records...], ctr is the
    // u32-word offset of the first free record (read, incremented, written
    // back); buffer size = sub_block_count * (capacity + 1) * 4 bytes.
    codegen.Emit("if (" + log + " != 0) {");
    codegen.Emit("  int32_t __log_block = " + block_idx + ";");
    codegen.Emit("  __gm__ uint32_t* __log_base = reinterpret_cast<__gm__ uint32_t*>(" + log +
                 ") + "
                 "static_cast<uint64_t>(__log_block) * (static_cast<uint64_t>(" +
                 capacity + ") + 1u);");
    codegen.Emit("  uint32_t __log_off = __log_base[0];");
}

} // namespace

static std::string MakeSanitizerLogCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // args = [det_id, field..., log_ptr, capacity, span_id] -> at least the
    // det_id, the two hidden buffer parameters and the span id.
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, op->args_.size() >= 4)
        << "block.sanitizer_log requires (det_id, field..., log_ptr, capacity, span_id)";

    const size_t n = op->args_.size();
    std::string det_id = codegen.GetExprAsCode(op->args_[0]);
    // Hidden log-buffer parameters: resolve their final (SSA-suffixed) names.
    std::string log = codegen.GetExprAsCode(op->args_[n - 3]);
    std::string capacity = codegen.GetExprAsCode(op->args_[n - 2]);
    std::string span_id = codegen.GetExprAsCode(op->args_[n - 1]);
    // Record size in u32 words: det_id + fields + span_id (fields exclude the
    // hidden log/capacity parameters).
    const uint32_t rec_u32 = static_cast<uint32_t>(n) - 2u;

    EmitSanitizerLogPrologue(codegen, log, capacity);
    codegen.Emit("  if (__log_off + " + std::to_string(rec_u32) + "u <= static_cast<uint32_t>(" + capacity + ")) {");
    codegen.Emit("    __gm__ uint32_t* __log_rec = __log_base + 1u + __log_off;");
    codegen.Emit("    __log_rec[0] = static_cast<uint32_t>(" + det_id + ");  // det_id");
    for (size_t i = 1; i + 3 < n; ++i) {
        codegen.Emit("    __log_rec[" + std::to_string(i) + "] = static_cast<int32_t>(" +
                     codegen.GetExprAsCode(op->args_[i]) + ");");
    }
    codegen.Emit("    __log_rec[" + std::to_string(rec_u32 - 1u) + "] = static_cast<uint32_t>(" + span_id + ");");
    codegen.Emit("    __log_base[0] = __log_off + " + std::to_string(rec_u32) + "u;");
    codegen.Emit("  }");
    codegen.Emit("}");
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "block.sanitizer_log").set_pipe(ir::PipeType::S).f_codegen(MakeSanitizerLogCodegenCCE);

// ============================================================================
// Language operations: get_block_num, get_subblock_idx
// ============================================================================

REGISTER_BACKEND_OP(BackendCCE, "get_block_num")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& /*codegen_base*/) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 0)
            << "get_block_num requires no arguments";
        return std::string("(int64_t)(get_block_num())");
    });

REGISTER_BACKEND_OP(BackendCCE, "get_subblock_idx")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& /*codegen_base*/) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 0)
            << "get_subblock_idx requires no arguments";
        return std::string("(int64_t)(get_subblockid())");
    });

REGISTER_BACKEND_OP(BackendCCE, "get_subblock_num")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 0)
            << "get_subblock_num requires no arguments";
        // Matches AscendC GetTaskRation(): AIC returns 1, AIV returns get_subblockdim().
        auto& cg = dynamic_cast<codegen::CCECodegen&>(codegen_base);
        const auto target = cg.GetTarget();
        if (target == ir::SectionKind::Vector) {
            return std::string("(int64_t)(get_subblockdim())");
        }
        return std::string("(int64_t)(1)");
    });

// ============================================================================
// GetVal/SetVal Operations (unified: tile and tensor)
// ============================================================================

static std::string MakeGetValCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2)
        << "getval requires 2 arguments, but got " << op->args_.size();

    auto result_type = ir::As<ir::ScalarType>(op->GetType());
    INTERNAL_CHECK(result_type) << "getval result must be ScalarType";
    auto cast_result_if_dtype_changed = [](std::string result, const ir::DataType& result_dtype,
                                           const ir::DataType& input_dtype) {
        if (result_dtype != input_dtype) {
            return "(" + result_dtype.ToCTypeString() + ")(" + result + ")";
        }
        return result;
    };

    auto first_type = op->args_[0]->GetType();
    if (auto tile_type = ir::As<ir::TileType>(first_type)) {
        std::string tile = codegen.GetExprAsCode(op->args_[0]);
        std::string offset = codegen.GetExprAsCode(op->args_[1]);
        if (codegen.IsInSimtContext()) {
            return cast_result_if_dtype_changed(tile + "[" + offset + "]", result_type->dtype_, tile_type->dtype_);
        }
        return cast_result_if_dtype_changed(tile + ".GetValue(" + offset + ")", result_type->dtype_, tile_type->dtype_);
    }

    auto tensor_var = ir::As<ir::Var>(op->args_[0]);
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, tensor_var)
        << "getval requires tensor to be a Var";
    auto tensor_type = ir::As<ir::TensorType>(tensor_var->GetType());
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, tensor_type)
        << "getval requires TensorType";
    std::string tensor_name = codegen.GetVarName(tensor_var);
    std::string offset = codegen.GetExprAsCode(op->args_[1]);
    std::string dtype_str = codegen.GetTypeString(tensor_type->dtype_);

    std::string tensor_ptr = codegen.GetPointer(tensor_name);

    return cast_result_if_dtype_changed("*((__gm__ " + dtype_str + "*)" + tensor_ptr + " + " + offset + ")",
                                        result_type->dtype_, tensor_type->dtype_);
}

static std::string MakeSetValCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "setval requires 3 arguments, but got " << op->args_.size();

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
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, tensor_var)
        << "setval requires tensor to be a Var";
    auto tensor_type = ir::As<ir::TensorType>(tensor_var->GetType());
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, tensor_type)
        << "setval requires TensorType";
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
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT,
                      op->args_.size() == 1 && ir::As<ir::TileType>(op->args_[0]->GetType()))
        << "block.tile_valid_shape requires one Tile argument";
    int axis = op->GetKwarg<int>("axis");
    PRO_CODEGEN_CHECK(ExternalError::INVALID_SHAPE, axis >= 0 && axis <= 1)
        << "block.tile_valid_shape axis must be in [0, 1]";

    if (codegen.IsInSimtContext()) {
        return "(int64_t)(" + codegen.GetExprAsCode(op->args_[0]) + (axis == 0 ? "__valid_row)" : "__valid_col)");
    }
    return "(int64_t)(" + codegen.GetExprAsCode(op->args_[0]) + (axis == 0 ? ".GetValidRow())" : ".GetValidCol())");
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
        PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, tile_type->memref_.has_value())
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

static std::string MakeMutexBufCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base,
                                          const std::string& intrinsic)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    auto pipe = static_cast<ir::PipeType>(op->GetKwarg<int>("pipe"));

    std::vector<int> mutex_ids = op->GetKwarg<std::vector<int>>("mutex_ids");
    // Candidate IDs are attached only by the auto-mutex path. A manual mutex op
    // has no candidate set and must always be emitted, including on PIPE_V.
    if (!mutex_ids.empty() && codegen.ShouldSkipVPipeMutex(pipe, mutex_ids))
        return "";

    std::string pipe_str = PipeTypeToCCEString(pipe);

    // N-way cross-Tile dedup: mutex IDs owned by one Tile are already known distinct.
    // Compare only across Tiles so each unique mutex_id is acquired and released once.
    // Without cross-Tile dedup, two get_buf(pipe, same_id) on the same pipe hang the hardware.
    if (op->args_.size() >= 2) {
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
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, mutex_id_owner_indices.size() == id_exprs.size())
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
                codegen.Emit(intrinsic + "(" + pipe_str + ", " + id_exprs[i] + ", 0);");
                return;
            }
            codegen.Emit("if (" + condition + ") {");
            codegen.Emit("  " + intrinsic + "(" + pipe_str + ", " + id_exprs[i] + ", 0);");
            codegen.Emit("}");
        };
        for (size_t i = 0; i < id_exprs.size(); ++i)
            emit_id(i);
        return "";
    }

    std::string mutex_id_expr = codegen.GetExprAsCode(op->args_[0]);
    codegen.Emit(intrinsic + "(" + pipe_str + ", " + mutex_id_expr + ", 0);");
    return "";
}

static std::string MakeMutexLockDynCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return MakeMutexBufCodegenCCE(op, codegen_base, "get_buf");
}

static std::string MakeMutexUnlockDynCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return MakeMutexBufCodegenCCE(op, codegen_base, "rls_buf");
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
// Delegates to pto-isa SYNCALL<SyncCoreType>().

static std::string MakeSystemSyncAllCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);

    auto core_type = static_cast<ir::SyncCoreType>(op->HasKwarg("core_type") ? op->GetKwarg<int>("core_type") : 2);

    std::string core_type_tok;
    if (core_type == ir::SyncCoreType::AIV_ONLY)
        core_type_tok = "SyncCoreType::AIVOnly";
    else if (core_type == ir::SyncCoreType::AIC_ONLY)
        core_type_tok = "SyncCoreType::AICOnly";
    else
        core_type_tok = "SyncCoreType::Mix";

    codegen.Emit("SYNCALL<" + core_type_tok + ">();");
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
