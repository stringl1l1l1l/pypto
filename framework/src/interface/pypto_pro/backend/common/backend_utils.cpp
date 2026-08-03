/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "backend/common/backend_utils.h"

#include <any>
#include <cctype>
#include <sstream>

#include "codegen/cce/cce_codegen.h"
#include "core/error.h"
#include "core/logging.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "tilefwk/error.h"

namespace pypto {
namespace backend {
namespace round_mode {

int FindIndex(const std::string& mode, const std::string& op_name)
{
    static const char* const kRoundModeNames[] = {"none", "rint",  "round", "floor",
                                                  "ceil", "trunc", "odd",   "cast_rint"};
    for (size_t i = 0; i < sizeof(kRoundModeNames) / sizeof(kRoundModeNames[0]); ++i) {
        if (mode == kRoundModeNames[i]) {
            return static_cast<int>(i);
        }
    }
    CHECK(false) << op_name << ": unknown round mode '" << mode << "'";
    return -1;
}

} // namespace round_mode

namespace mutex_id {

std::vector<int> GetMutexIdsFromKwargs(const ir::CallPtr& op)
{
    std::vector<int> values;
    for (const auto& [key, value] : op->kwargs_) {
        if (key == "mutex_ids") {
            values = std::any_cast<std::vector<int>>(value);
            return values;
        }
    }
    return values;
}

} // namespace mutex_id

namespace gather {

CompareAttrs GetCompareAttrs(const ir::CallPtr& op)
{
    CompareAttrs attrs;
    for (const auto& [key, val] : op->kwargs_) {
        if (key == "cmp_mode") {
            attrs.has_cmp_mode = true;
            attrs.cmp_mode = std::any_cast<int>(val);
        } else if (key == "offset") {
            attrs.offset = std::any_cast<int>(val);
        }
    }
    return attrs;
}

} // namespace gather

namespace cce {

std::string ComputeStrideBasedOffset(codegen::CCECodegen& codegen, const ir::MakeTuplePtr& offsets,
                                     const ir::TensorTypePtr& tensor_type)
{
    return codegen.ComputeIRBasedOffset(tensor_type, offsets);
}

bool IsNZTensorType(const ir::TensorTypePtr& tensor_type)
{
    return tensor_type && tensor_type->tensor_view_.has_value() &&
           tensor_type->tensor_view_->layout == ir::TensorLayout::NZ;
}

int64_t GetNZInnerCols(const ir::DataType& dtype, const std::string& error_prefix)
{
    if (dtype == ir::DataType::BOOL || dtype == ir::DataType::INT8 || dtype == ir::DataType::UINT8) {
        return 32;
    }
    if (dtype == ir::DataType::FP16 || dtype == ir::DataType::BF16 || dtype == ir::DataType::INT16 ||
        dtype == ir::DataType::UINT16) {
        return 16;
    }
    if (dtype == ir::DataType::FP32 || dtype == ir::DataType::INT32 || dtype == ir::DataType::UINT32) {
        return 8;
    }
    if (dtype == ir::DataType::INT64 || dtype == ir::DataType::UINT64) {
        return 4;
    }
    throw pypto::ir::ValueError(error_prefix + dtype.ToString());
}

static int64_t GetStaticConstIntOrThrow(const ir::ExprPtr& expr, const std::string& message)
{
    auto value = ir::As<ir::ConstInt>(expr);
    if (!value) {
        throw pypto::ir::ValueError(message);
    }
    return value->value_;
}

static bool IsConstZero(const ir::ExprPtr& expr)
{
    auto value = ir::As<ir::ConstInt>(expr);
    return value && value->value_ == 0;
}

void ValidateStoreNZPreconditions(const std::string& op_name, const ir::ExprPtr& src_expr,
                                  const ir::MakeTuplePtr& offsets, const ir::TensorTypePtr& dst_tensor_type)
{
    if (!IsNZTensorType(dst_tensor_type)) {
        return;
    }

    auto src_tile_type = ir::As<ir::TileType>(src_expr->GetType());
    CHECK(src_tile_type != nullptr) << op_name << ": source must be TileType";
    CHECK(src_tile_type->memref_.has_value()) << op_name << ": source tile must have an allocated memory space";
    if (src_tile_type->memref_.value()->memorySpace_ != ir::MemorySpace::Acc) {
        throw pypto::ir::ValueError(op_name + ": CCE NZ output currently only supports Acc source tiles");
    }

    if (dst_tensor_type->shape_.size() != 2) {
        throw pypto::ir::ValueError(op_name + ": CCE NZ output currently requires a 2D destination tensor");
    }
    if (offsets->elements_.size() != 2 || !IsConstZero(offsets->elements_[0]) || !IsConstZero(offsets->elements_[1])) {
        throw pypto::ir::ValueError(op_name + ": CCE NZ output currently requires offsets=[0, 0]");
    }

    const int64_t rows = GetStaticConstIntOrThrow(
        dst_tensor_type->shape_[0], op_name + ": CCE NZ output currently requires a static destination row shape");
    const int64_t cols = GetStaticConstIntOrThrow(
        dst_tensor_type->shape_[1], op_name + ": CCE NZ output currently requires a static destination column shape");
    const int64_t c0 = GetNZInnerCols(dst_tensor_type->dtype_, "CCE NZ store does not support destination dtype ");
    CHECK(c0 > 0) << op_name << ": NZ C0 size must be positive";
    if (rows % 16 != 0 || cols % c0 != 0) {
        throw pypto::ir::ValueError(
            op_name + ": CCE NZ output requires rows divisible by 16 and cols divisible by the destination C0 size");
    }
}

} // namespace cce

namespace debug_printf {

std::string EscapeStringLiteral(const std::string& text)
{
    std::ostringstream oss;
    for (char c : text) {
        switch (c) {
            case '\\':
                oss << "\\\\";
                break;
            case '"':
                oss << "\\\"";
                break;
            case '\n':
                oss << "\\n";
                break;
            case '\t':
                oss << "\\t";
                break;
            case '\r':
                oss << "\\r";
                break;
            default:
                oss << c;
                break;
        }
    }
    return oss.str();
}

std::string QuoteMlirStringLiteral(const std::string& text) { return "\"" + EscapeStringLiteral(text) + "\""; }

std::string FormatDebugLocation(const ir::Span& span)
{
    if (!span.IsValid() || span.Filename().empty() || span.BeginLine() <= 0) {
        return "";
    }

    size_t last_sep = span.Filename().find_last_of("/\\");
    std::string basename = last_sep == std::string::npos ? span.Filename() : span.Filename().substr(last_sep + 1);
    if (basename.empty()) {
        return "";
    }
    return "[" + basename + ":" + std::to_string(span.BeginLine()) + "]";
}

std::string FormatDebugLocationHeader(const ir::Span& span, const std::string& op_name)
{
    std::string location = FormatDebugLocation(span);
    if (location.empty()) {
        return "";
    }
    return location + " " + op_name;
}

bool IsSupportedPrintfConversion(char conversion)
{
    return conversion == 'd' || conversion == 'i' || conversion == 'u' || conversion == 'x' || conversion == 'f' ||
           conversion == 'p';
}

size_t FindPrintfConversionIndex(const std::string& format_segment)
{
    size_t i = 0;
    while (i < format_segment.size()) {
        if (format_segment[i] != '%') {
            ++i;
            continue;
        }
        CHECK(!(i + 1 < format_segment.size() && format_segment[i + 1] == '%'))
            << "debug.printf does not support literal '%%'";

        size_t j = i + 1;
        while (j < format_segment.size()) {
            char c = format_segment[j];
            if (c == '-' || c == '+' || c == ' ' || c == '#' || c == '0') {
                ++j;
            } else {
                break;
            }
        }
        while (j < format_segment.size() && std::isdigit(static_cast<unsigned char>(format_segment[j]))) {
            ++j;
        }
        if (j < format_segment.size() && format_segment[j] == '.') {
            ++j;
            CHECK(j < format_segment.size() && std::isdigit(static_cast<unsigned char>(format_segment[j])))
                << "debug.printf precision must be followed by digits";
            while (j < format_segment.size() && std::isdigit(static_cast<unsigned char>(format_segment[j]))) {
                ++j;
            }
        }

        CHECK(j < format_segment.size()) << "debug.printf format ends with an incomplete conversion";
        CHECK(IsSupportedPrintfConversion(format_segment[j]))
            << "debug.printf does not support conversion '%" << format_segment[j] << "'";
        return j;
    }
    CHECK(false) << "debug.printf format segment must contain a supported conversion";
    return std::string::npos;
}

PrintfFormatParts SplitPrintfSegment(const std::string& format_segment)
{
    size_t conv_idx = FindPrintfConversionIndex(format_segment);
    size_t percent_idx = format_segment.rfind('%', conv_idx);
    INTERNAL_CHECK(percent_idx != std::string::npos)
        << "debug.printf failed to locate '%' while splitting format segment";
    return {format_segment.substr(0, percent_idx), format_segment.substr(percent_idx, conv_idx - percent_idx + 1),
            format_segment.substr(conv_idx + 1)};
}

std::vector<PrintfSegment> ParsePrintfSegments(const std::string& format)
{
    std::vector<PrintfSegment> segments;
    std::string pending_text;
    size_t i = 0;
    while (i < format.size()) {
        if (format[i] != '%') {
            pending_text.push_back(format[i]);
            ++i;
            continue;
        }
        if (i + 1 < format.size() && format[i + 1] == '%') {
            CHECK(false) << "debug.printf does not support literal '%%'";
        }

        size_t j = FindPrintfConversionIndex(format.substr(i)) + i;
        char conversion = format[j];
        segments.push_back({pending_text + format.substr(i, j - i + 1), conversion});
        pending_text.clear();
        i = j + 1;
    }

    if (!pending_text.empty()) {
        if (segments.empty()) {
            return segments;
        }
        segments.back().format_segment += pending_text;
    }
    return segments;
}

} // namespace debug_printf
} // namespace backend
} // namespace pypto
