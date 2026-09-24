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
 * @file backend_cce_simt_ops.cpp
 * \brief Direct CCE lowering for the A5 SIMT context and launch operations.
 */

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>

#include "backend/backend_cce.h"
#include "backend/common/backend.h"
#include "codegen/cce/cce_codegen.h"
#include "codegen/codegen_base.h"
#include "core/dtype.h"
#include "core/logging.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/op_attr_types.h"
#include "ir/pipe.h"
#include "ir/scalar_expr.h"
#include "pypto_pro/error.h"

namespace pypto {
namespace backend {
using npu::tile_fwk::ExternalError;

namespace {

const char* GetSimtAxisName(int axis)
{
    constexpr const char* axis_names[] = {"x", "y", "z"};
    return axis_names[axis];
}

std::string MakeSimtContextComponentCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base,
                                               const char* op_name, const char* context_name)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, codegen.IsInSimtContext())
        << op_name << " reached CCE codegen outside a SIMT function";
    return std::string(context_name) + "." + GetSimtAxisName(op->GetKwarg<int>("axis"));
}

std::string MakeSimtThreadIdxCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return MakeSimtContextComponentCodegenCCE(op, codegen_base, "simt.thread_idx", "threadIdx");
}

std::string MakeSimtBlockDimCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return MakeSimtContextComponentCodegenCCE(op, codegen_base, "simt.block_dim", "blockDim");
}

std::string MakeSimtBlockIdxCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return MakeSimtContextComponentCodegenCCE(op, codegen_base, "simt.block_idx", "blockIdx");
}

std::string MakeSimtGridDimCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return MakeSimtContextComponentCodegenCCE(op, codegen_base, "simt.grid_dim", "gridDim");
}

std::string MakeSimtLinearThreadIdxCodegenCCE([[maybe_unused]] const ir::CallPtr& op,
                                              codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, codegen.IsInSimtContext())
        << "simt.linear_thread_idx reached CCE codegen outside a SIMT function";
    return "(threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)";
}

std::string MakeSimtWarpSizeCodegenCCE([[maybe_unused]] const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, codegen.IsInSimtContext())
        << "simt.warp_size reached CCE codegen outside a SIMT function";
    return "warpSize";
}

std::string MakeSimtSyncthreadsCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, codegen.IsInSimtContext())
        << "simt.syncthreads reached CCE codegen outside a SIMT function";
    PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, op->args_.empty() && op->kwargs_.empty())
        << "simt.syncthreads does not accept arguments";
    codegen.Emit("__sync_workitems();");
    return "";
}

std::string MakeSimtThreadfenceBlockCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, codegen.IsInSimtContext())
        << "simt.threadfence_block reached CCE codegen outside a SIMT function";
    PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, op->args_.empty() && op->kwargs_.empty())
        << "simt.threadfence_block does not accept arguments";
    codegen.Emit("__threadfence_block();");
    return "";
}

std::string MakeSimtThreadfenceCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, codegen.IsInSimtContext())
        << "simt.threadfence reached CCE codegen outside a SIMT function";
    PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, op->args_.empty() && op->kwargs_.empty())
        << "simt.threadfence does not accept arguments";
    codegen.Emit("__threadfence();");
    return "";
}

const char* GetSimtCastRoundModeCCE(ir::RoundMode mode)
{
    switch (mode) {
        case ir::RoundMode::CAST_NONE:
        case ir::RoundMode::CAST_RINT:
            return "ROUND::R";
        case ir::RoundMode::CAST_ROUND:
            return "ROUND::A";
        case ir::RoundMode::CAST_FLOOR:
            return "ROUND::F";
        case ir::RoundMode::CAST_CEIL:
            return "ROUND::C";
        case ir::RoundMode::CAST_TRUNC:
            return "ROUND::Z";
        case ir::RoundMode::CAST_ODD:
            return "ROUND::O";
        default:
            PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, false) << "Unsupported simt.cast round mode";
            return "";
    }
}

const char* GetSimtCastIntrinsicCCE(ir::DataType dtype)
{
    if (dtype == ir::DataType::FP16) {
        return "__cvt_half";
    }
    if (dtype == ir::DataType::BF16) {
        return "__cvt_bfloat16_t";
    }
    if (dtype == ir::DataType::FP32) {
        return "__cvt_float";
    }
    if (dtype == ir::DataType::INT32) {
        return "__cvt_int32_t";
    }
    if (dtype == ir::DataType::UINT32) {
        return "__cvt_uint32_t";
    }
    if (dtype == ir::DataType::INT64) {
        return "__cvt_int64_t";
    }
    if (dtype == ir::DataType::UINT64) {
        return "__cvt_uint64_t";
    }
    return nullptr;
}

std::string MakeSimtCastIntrinsicCallCCE(ir::DataType target_dtype, ir::RoundMode mode, const char* saturation,
                                         const std::string& operand)
{
    const char* intrinsic = GetSimtCastIntrinsicCCE(target_dtype);
    PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, intrinsic != nullptr)
        << "No A5 scalar conversion intrinsic for " << target_dtype.ToString();
    return std::string(intrinsic) + "<" + GetSimtCastRoundModeCCE(mode) + ", " + saturation + ">(" + operand + ")";
}

std::string MakeSimtCastNarrowIntegerCCE(ir::DataType target_dtype, ir::RoundMode mode, const std::string& operand)
{
    // A5 has no direct low-float-to-8/16-bit scalar conversion. Match the reference semantics by converting to
    // a 32-bit carrier first, then clamp before narrowing. This is fixed per-conversion behavior, not public SatMode.
    const bool is_signed = target_dtype == ir::DataType::INT8 || target_dtype == ir::DataType::INT16;
    const ir::DataType carrier_dtype = is_signed ? ir::DataType::INT32 : ir::DataType::UINT32;
    const std::string converted = MakeSimtCastIntrinsicCallCCE(carrier_dtype, mode,
                                                               "RoundingSaturation::RS_ENABLE_VALUE", operand);

    const char* upper_bound = nullptr;
    const char* lower_bound = nullptr;
    if (target_dtype == ir::DataType::INT8) {
        upper_bound = "127";
        lower_bound = "-128";
    } else if (target_dtype == ir::DataType::UINT8) {
        upper_bound = "255U";
    } else if (target_dtype == ir::DataType::INT16) {
        upper_bound = "32767";
        lower_bound = "-32768";
    } else if (target_dtype == ir::DataType::UINT16) {
        upper_bound = "65535U";
    } else {
        PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, false)
            << "Unsupported narrow simt.cast target " << target_dtype.ToString();
    }

    const std::string target_type = target_dtype.ToCTypeString();
    const std::string carrier_type = carrier_dtype.ToCTypeString();
    std::ostringstream s;
    s << "({" << carrier_type << " __simt_cast_value = " << converted << ";"
      << "__simt_cast_value > " << upper_bound << " ? (" << target_type << ")" << upper_bound << " : ";
    if (is_signed) {
        s << "(__simt_cast_value < " << lower_bound << " ? (" << target_type << ")" << lower_bound << " : ("
          << target_type << ")__simt_cast_value);";
    } else {
        s << "(" << target_type << ")__simt_cast_value;";
    }
    s << "})";
    return s.str();
}

std::string MakeSimtCastPlainCCE(ir::DataType target_dtype, ir::RoundMode mode, const std::string& operand)
{
    PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, mode == ir::RoundMode::CAST_NONE)
        << "Explicit simt.cast rounding requires an A5 scalar conversion intrinsic";
    return "((" + target_dtype.ToCTypeString() + ")" + operand + ")";
}

std::string MakeSimtCastFromFp16OrBf16CCE(ir::DataType target_dtype, ir::RoundMode mode, const std::string& operand)
{
    if (target_dtype == ir::DataType::FP16 || target_dtype == ir::DataType::BF16 ||
        target_dtype == ir::DataType::FP32) {
        return MakeSimtCastIntrinsicCallCCE(target_dtype, mode, "RoundingSaturation::RS_DISABLE_VALUE", operand);
    }
    if (mode != ir::RoundMode::CAST_NONE &&
        (target_dtype == ir::DataType::INT32 || target_dtype == ir::DataType::UINT32)) {
        return MakeSimtCastIntrinsicCallCCE(target_dtype, mode, "RoundingSaturation::RS_ENABLE_VALUE", operand);
    }
    if (mode != ir::RoundMode::CAST_NONE &&
        (target_dtype == ir::DataType::INT64 || target_dtype == ir::DataType::UINT64)) {
        const std::string fp32 = MakeSimtCastIntrinsicCallCCE(ir::DataType::FP32, mode,
                                                              "RoundingSaturation::RS_ENABLE_VALUE", operand);
        return MakeSimtCastIntrinsicCallCCE(target_dtype, mode, "RoundingSaturation::RS_ENABLE_VALUE", fp32);
    }
    if (mode != ir::RoundMode::CAST_NONE &&
        (target_dtype == ir::DataType::INT8 || target_dtype == ir::DataType::UINT8 ||
         target_dtype == ir::DataType::INT16 || target_dtype == ir::DataType::UINT16)) {
        return MakeSimtCastNarrowIntegerCCE(target_dtype, mode, operand);
    }
    return MakeSimtCastPlainCCE(target_dtype, mode, operand);
}

std::string MakeSimtCastFromFp32CCE(ir::DataType target_dtype, ir::RoundMode mode, const std::string& operand)
{
    if (target_dtype == ir::DataType::FP16 || target_dtype == ir::DataType::BF16) {
        return MakeSimtCastIntrinsicCallCCE(target_dtype, mode, "RoundingSaturation::RS_DISABLE_VALUE", operand);
    }
    if (mode != ir::RoundMode::CAST_NONE &&
        (target_dtype == ir::DataType::INT32 || target_dtype == ir::DataType::UINT32 ||
         target_dtype == ir::DataType::INT64 || target_dtype == ir::DataType::UINT64)) {
        return MakeSimtCastIntrinsicCallCCE(target_dtype, mode, "RoundingSaturation::RS_ENABLE_VALUE", operand);
    }
    return MakeSimtCastPlainCCE(target_dtype, mode, operand);
}

std::string MakeSimtCastFromInt16OrUint16CCE(ir::DataType source_dtype, ir::DataType target_dtype, ir::RoundMode mode,
                                             const std::string& operand)
{
    if (mode != ir::RoundMode::CAST_NONE &&
        (target_dtype == ir::DataType::FP16 || target_dtype == ir::DataType::BF16)) {
        const ir::DataType carrier_dtype = source_dtype == ir::DataType::INT16 ? ir::DataType::INT32 :
                                                                                 ir::DataType::UINT32;
        const std::string widened = "((" + carrier_dtype.ToCTypeString() + ")" + operand + ")";
        return MakeSimtCastIntrinsicCallCCE(target_dtype, mode, "RoundingSaturation::RS_DISABLE_VALUE", widened);
    }
    return MakeSimtCastPlainCCE(target_dtype, mode, operand);
}

std::string MakeSimtCastFromInt32OrUint32CCE(ir::DataType target_dtype, ir::RoundMode mode, const std::string& operand)
{
    if (mode != ir::RoundMode::CAST_NONE && (target_dtype == ir::DataType::FP16 || target_dtype == ir::DataType::BF16 ||
                                             target_dtype == ir::DataType::FP32)) {
        return MakeSimtCastIntrinsicCallCCE(target_dtype, mode, "RoundingSaturation::RS_DISABLE_VALUE", operand);
    }
    return MakeSimtCastPlainCCE(target_dtype, mode, operand);
}

std::string MakeSimtCastFromInt64OrUint64CCE(ir::DataType source_dtype, ir::DataType target_dtype, ir::RoundMode mode,
                                             const std::string& operand)
{
    if (mode != ir::RoundMode::CAST_NONE &&
        (target_dtype == ir::DataType::FP16 || target_dtype == ir::DataType::BF16)) {
        // The A5 uint64-to-BF16 reference path always uses nearest-even for the uint64-to-FP32 carrier conversion.
        const auto fp32_mode = source_dtype == ir::DataType::UINT64 && target_dtype == ir::DataType::BF16 ?
                                   ir::RoundMode::CAST_RINT :
                                   mode;
        const std::string fp32 = MakeSimtCastIntrinsicCallCCE(ir::DataType::FP32, fp32_mode,
                                                              "RoundingSaturation::RS_DISABLE_VALUE", operand);
        return MakeSimtCastIntrinsicCallCCE(target_dtype, mode, "RoundingSaturation::RS_DISABLE_VALUE", fp32);
    }
    if (mode != ir::RoundMode::CAST_NONE && target_dtype == ir::DataType::FP32) {
        return MakeSimtCastIntrinsicCallCCE(target_dtype, mode, "RoundingSaturation::RS_DISABLE_VALUE", operand);
    }
    return MakeSimtCastPlainCCE(target_dtype, mode, operand);
}

std::string MakeSimtCastCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, codegen.IsInSimtContext())
        << "simt.cast reached CCE codegen outside a SIMT function";
    PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, codegen.GetArch() == npu::tile_fwk::NPUArch::DAV_3510)
        << "simt.cast currently requires arch='3510'";
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 1)
        << "simt.cast requires one scalar argument";

    auto source_type = ir::As<ir::ScalarType>(op->args_[0]->GetType());
    auto target_type = ir::As<ir::ScalarType>(op->GetType());
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, source_type) << "simt.cast source must be ScalarType";
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, target_type) << "simt.cast target must be ScalarType";

    const auto source_dtype = source_type->dtype_;
    const auto target_dtype = target_type->dtype_;
    const auto mode = static_cast<ir::RoundMode>(op->GetKwarg<int>("mode"));
    const std::string operand = codegen.GetExprAsCode(op->args_[0]);
    if (source_dtype == target_dtype) {
        return operand;
    }

    if (source_dtype == ir::DataType::INT16 || source_dtype == ir::DataType::UINT16) {
        return MakeSimtCastFromInt16OrUint16CCE(source_dtype, target_dtype, mode, operand);
    }
    if (source_dtype == ir::DataType::INT32 || source_dtype == ir::DataType::UINT32) {
        return MakeSimtCastFromInt32OrUint32CCE(target_dtype, mode, operand);
    }
    if (source_dtype == ir::DataType::INT64 || source_dtype == ir::DataType::UINT64) {
        return MakeSimtCastFromInt64OrUint64CCE(source_dtype, target_dtype, mode, operand);
    }
    if (source_dtype == ir::DataType::FP16 || source_dtype == ir::DataType::BF16) {
        return MakeSimtCastFromFp16OrBf16CCE(target_dtype, mode, operand);
    }
    if (source_dtype == ir::DataType::FP32) {
        return MakeSimtCastFromFp32CCE(target_dtype, mode, operand);
    }
    return MakeSimtCastPlainCCE(target_dtype, mode, operand);
}

std::string MakeSimtBitcastCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, codegen.IsInSimtContext())
        << "simt.bitcast reached CCE codegen outside a SIMT function";
    PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, codegen.GetArch() == npu::tile_fwk::NPUArch::DAV_3510)
        << "simt.bitcast currently requires arch='3510'";
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 1)
        << "simt.bitcast requires one scalar argument";

    auto source_type = ir::As<ir::ScalarType>(op->args_[0]->GetType());
    auto target_type = ir::As<ir::ScalarType>(op->GetType());
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, source_type) << "simt.bitcast source must be ScalarType";
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, target_type) << "simt.bitcast target must be ScalarType";

    const std::string operand = codegen.GetExprAsCode(op->args_[0]);
    std::ostringstream s;
    s << "({union {" << source_type->dtype_.ToCTypeString() << " source;" << target_type->dtype_.ToCTypeString()
      << " target;} __simt_bitcast_data;"
      << "__simt_bitcast_data.source = (" << operand << ");"
      << "__simt_bitcast_data.target;})";
    return s.str();
}

std::string MakeSimtTrigFP32Codegen(const std::string& operand, bool is_sin)
{
    std::ostringstream s;
    s << "({"
      << "float __t = (" << operand << ");"
      << "__t = __fma(__t, 0.0f, __t);"
      << "int __q;"
      << "float __y;"
      << "if (__fabsf(__t) > 71476.0625f) {"
      << "uint32_t __bits = reinterpret_cast<uint32_t&>(__t);"
      << "int32_t __exp = ((__bits & 0x7F800000) >> 23) - 127;"
      << "uint32_t __ei = (uint32_t)__exp >> 5;"
      << "const uint32_t __tbl[] = {0x517cc1b7, 0x27220a94, 0xfe13abe8, 0xfa9a6ee0, 0x6db14acc, 0x9e21c820};"
      << "uint32_t __hi = __ei ? __tbl[__ei - 1] : 0;"
      << "uint32_t __mid = __tbl[__ei];"
      << "uint32_t __lo = __tbl[__ei + 1];"
      << "uint32_t __last = __tbl[__ei + 2];"
      << "int32_t __er = (uint32_t)__exp & 0x1F;"
      << "if (__er) {"
      << "__hi = (__hi << __er) | (__mid >> (32 - __er));"
      << "__mid = (__mid << __er) | (__lo >> (32 - __er));"
      << "__lo = (__lo << __er) | (__last >> (32 - __er));"
      << "}"
      << "uint32_t __mant = (__bits & 0x007FFFFF) | 0x4F000000;"
      << "uint32_t __nmant = (uint32_t)reinterpret_cast<float&>(__mant);"
      << "uint64_t __prod = (uint64_t)__nmant * __lo;"
      << "__prod = (uint64_t)__nmant * __mid + (__prod >> 32);"
      << "__prod = ((uint64_t)(__nmant * __hi) << 32) + __prod;"
      << "int32_t __quot = (int32_t)(__prod >> 62);"
      << "__prod &= 0x3FFFFFFFFFFFFFFFULL;"
      << "if (__prod & 0x2000000000000000ULL) { __prod -= 0x4000000000000000ULL; __quot++; }"
      << "int64_t __pi = (int64_t)__prod;"
      << "float __hf = (float)__pi;"
      << "__pi -= (int64_t)__hf;"
      << "float __lf = (float)__pi;"
      << "__y = (__hf + __lf) * 3.4061215800865545e-19f;"
      << "if (__t < 0.0f) { __y = -__y; __quot = -__quot; }"
      << "__q = __quot;"
      << "} else {"
      << "float __r = __fma(__t, 0.636619747f, 12582912.0f);"
      << "__q = reinterpret_cast<int&>(__r);"
      << "__r -= 12582912.0f;"
      << "__t = __fma(__r, -1.57079601e+00f, __t);"
      << "__t = __fma(__r, -3.13916473e-07f, __t);"
      << "__y = __fma(__r, -5.39030253e-15f, __t);"
      << "}"
      << "float __yy = __y * __y;"
      << "float __m = __fma(__y, __yy, 0.0f);"
      << "float __z = __fma(__yy, 2.86567956e-6f, -1.98559923e-4f);"
      << "__z = __fma(__yy, __z, 8.33338592e-3f);"
      << "__z = __fma(__yy, __z, -1.66666672e-1f);"
      << "float __s = __fma(__z, __m, __y);"
      << "float __c = __fma(__yy, 2.44677067e-5f, -1.38877297e-3f);"
      << "__c = __fma(__yy, __c, 4.16666567e-2f);"
      << "__c = __fma(__yy, __c, -5.00000000e-1f);"
      << "__c = __fma(__yy, __c, 1.00000000e+0f);"
      << "if (__q & 2) { __s = -__s; __c = -__c; }";
    if (is_sin) {
        s << "if (__q & 1) { __s = __c; }"
          << "__s;";
    } else {
        s << "if (__q & 1) { __c = -__s; }"
          << "__c;";
    }
    s << "})";
    return s.str();
}

struct SimtUnaryCodegenInput {
    ir::DataType dtype;
    std::string operand;
};

SimtUnaryCodegenInput GetSimtUnaryCodegenInput(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, codegen.IsInSimtContext())
        << op->name_ << " reached CCE codegen outside a SIMT function";
    PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, codegen.GetArch() == npu::tile_fwk::NPUArch::DAV_3510)
        << op->name_ << " currently requires arch='3510'";
    auto scalar_type = ir::As<ir::ScalarType>(op->args_[0]->GetType());
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, scalar_type != nullptr) << op->name_ << " operand must be a scalar";
    return {scalar_type->dtype_, codegen.GetExprAsCode(op->args_[0])};
}

std::string MakeSimtAbsCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const auto& dtype = input.dtype;
    const auto& a = input.operand;
    if (dtype == ir::DataType::INT64)
        return "abs(" + a + ")";
    if (dtype == ir::DataType::FP32)
        return "__fabsf(" + a + ")";
    if (dtype == ir::DataType::FP16) {
        const std::string cvt_in = "__cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(" + a + ")";
        return "__cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(__fabsf(" + cvt_in + "))";
    }
    if (dtype == ir::DataType::BF16) {
        const std::string cvt_in = "__cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(" + a + ")";
        return "__cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(__fabsf(" + cvt_in + "))";
    }
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false) << "Unsupported simt.abs dtype " << dtype.ToString();
    return "";
}

std::string MakeSimtSqrtCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const auto& dtype = input.dtype;
    const auto& a = input.operand;
    if (dtype == ir::DataType::FP32)
        return "__sqrtf(" + a + ")";
    if (dtype == ir::DataType::FP16)
        return "__sqrtf(" + a + ")";
    if (dtype == ir::DataType::BF16) {
        const std::string cvt_in = "__cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(" + a + ")";
        return "__cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(__sqrtf(" + cvt_in + "))";
    }
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false) << "Unsupported simt.sqrt dtype " << dtype.ToString();
    return "";
}

std::string MakeSimtRsqrtCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const auto& dtype = input.dtype;
    const auto& a = input.operand;
    if (dtype == ir::DataType::FP32)
        return "(1.0f / __sqrtf(" + a + "))";
    if (dtype == ir::DataType::FP16)
        return "((half)1.0 / __sqrtf(" + a + "))";
    if (dtype == ir::DataType::BF16) {
        const std::string cvt_in = "__cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(" + a + ")";
        return "__cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(1.0f / __sqrtf(" + cvt_in + "))";
    }
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false) << "Unsupported simt.rsqrt dtype " << dtype.ToString();
    return "";
}

std::string MakeSimtExpCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const auto& dtype = input.dtype;
    const auto& a = input.operand;
    if (dtype == ir::DataType::FP32)
        return "__expf(" + a + ")";
    if (dtype == ir::DataType::FP16)
        return "__expf(" + a + ")";
    if (dtype == ir::DataType::BF16) {
        const std::string cvt_in = "__cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(" + a + ")";
        return "__cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(__expf(" + cvt_in + "))";
    }
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false) << "Unsupported simt.exp dtype " << dtype.ToString();
    return "";
}

std::string MakeSimtExp2CodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const auto& dtype = input.dtype;
    const auto& a = input.operand;
    if (dtype == ir::DataType::FP32)
        return "__expf(" + a + " * 0.6931471805599453f)";
    if (dtype == ir::DataType::FP16) {
        const std::string cvt_in = "__cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(" + a + ")";
        return "__cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(__expf(" + cvt_in +
               " * 0.6931471805599453f))";
    }
    if (dtype == ir::DataType::BF16) {
        const std::string cvt_in = "__cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(" + a + ")";
        return "__cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(__expf(" + cvt_in +
               " * 0.6931471805599453f))";
    }
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false) << "Unsupported simt.exp2 dtype " << dtype.ToString();
    return "";
}

std::string MakeSimtLogCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const auto& dtype = input.dtype;
    const auto& a = input.operand;
    if (dtype == ir::DataType::FP32) {
        return "(((" + a + " > 0.0f && " + a +
               " < 1.17549435e-38f) ? "
               "(__logf(__expf(23.0f) * " +
               a + ") - 23.0f) : __logf(" + a + ")))";
    }
    if (dtype == ir::DataType::FP16)
        return "__logf(" + a + ")";
    if (dtype == ir::DataType::BF16) {
        const std::string cvt_in = "__cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(" + a + ")";
        return "__cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(__logf(" + cvt_in + "))";
    }
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false) << "Unsupported simt.log dtype " << dtype.ToString();
    return "";
}

std::string MakeSimtLog2CodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const auto& dtype = input.dtype;
    const auto& a = input.operand;
    if (dtype == ir::DataType::FP32) {
        return "(((" + a + " > 0.0f && " + a +
               " < 1.17549435e-38f) ? "
               "(__logf(__expf(23.0f) * " +
               a + ") - 23.0f) : __logf(" + a + ")) / __logf(2.0f))";
    }
    if (dtype == ir::DataType::FP16) {
        const std::string cvt_in = "__cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(" + a + ")";
        return "__cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(__logf(" + cvt_in + ") / __logf(2.0f))";
    }
    if (dtype == ir::DataType::BF16) {
        const std::string cvt_in = "__cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(" + a + ")";
        return "__cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(__logf(" + cvt_in +
               ") / __logf(2.0f))";
    }
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false) << "Unsupported simt.log2 dtype " << dtype.ToString();
    return "";
}

std::string MakeSimtLog1pCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const auto& dtype = input.dtype;
    const auto& a = input.operand;
    if (dtype == ir::DataType::FP32)
        return "__logf(1.0f + " + a + ")";
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false) << "Unsupported simt.log1p dtype " << dtype.ToString();
    return "";
}

std::string MakeSimtTanhCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const auto& dtype = input.dtype;
    const auto& a = input.operand;
    if (dtype == ir::DataType::FP32)
        return "(1.0f - (2.0f / (__expf(2.0f * " + a + ") + 1.0f)))";
    if (dtype == ir::DataType::FP16) {
        const std::string cvt_in = "__cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(" + a + ")";
        return "__cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(1.0f - (2.0f / (__expf(2.0f * " + cvt_in +
               ") + 1.0f)))";
    }
    if (dtype == ir::DataType::BF16) {
        const std::string cvt_in = "__cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(" + a + ")";
        return "__cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(1.0f - (2.0f / (__expf(2.0f * " +
               cvt_in + ") + 1.0f)))";
    }
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false) << "Unsupported simt.tanh dtype " << dtype.ToString();
    return "";
}

std::string MakeSimtRintCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const auto& dtype = input.dtype;
    const auto& a = input.operand;
    if (dtype == ir::DataType::FP32 || dtype == ir::DataType::FP16 || dtype == ir::DataType::BF16) {
        return "__rintf(" + a + ")";
    }
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false) << "Unsupported simt.rint dtype " << dtype.ToString();
    return "";
}

std::string MakeSimtRoundCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const auto& dtype = input.dtype;
    const auto& a = input.operand;
    if (dtype == ir::DataType::FP32)
        return "__roundf(" + a + ")";
    if (dtype == ir::DataType::FP16) {
        return "__cvt_half<ROUND::A, RoundingSaturation::RS_DISABLE_VALUE>(" + a + ")";
    }
    if (dtype == ir::DataType::BF16) {
        return "__cvt_bfloat16_t<ROUND::A, RoundingSaturation::RS_DISABLE_VALUE>(" + a + ")";
    }
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false) << "Unsupported simt.round dtype " << dtype.ToString();
    return "";
}

std::string MakeSimtFloorCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const auto& dtype = input.dtype;
    const auto& a = input.operand;
    if (dtype == ir::DataType::FP32 || dtype == ir::DataType::FP16 || dtype == ir::DataType::BF16) {
        return "__floorf(" + a + ")";
    }
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false) << "Unsupported simt.floor dtype " << dtype.ToString();
    return "";
}

std::string MakeSimtCeilCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const auto& dtype = input.dtype;
    const auto& a = input.operand;
    if (dtype == ir::DataType::FP32 || dtype == ir::DataType::FP16 || dtype == ir::DataType::BF16) {
        return "__ceilf(" + a + ")";
    }
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false) << "Unsupported simt.ceil dtype " << dtype.ToString();
    return "";
}

std::string MakeSimtTruncCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const auto& dtype = input.dtype;
    const auto& a = input.operand;
    if (dtype == ir::DataType::FP32)
        return "((" + a + " > 0.0f) ? __floorf(" + a + ") : __ceilf(" + a + "))";
    if (dtype == ir::DataType::FP16)
        return "((" + a + " > (half)0) ? __floorf(" + a + ") : __ceilf(" + a + "))";
    if (dtype == ir::DataType::BF16) {
        return "((" + a + " > (bfloat16_t)0) ? __floorf(" + a + ") : __ceilf(" + a + "))";
    }
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false) << "Unsupported simt.trunc dtype " << dtype.ToString();
    return "";
}

std::string MakeSimtIsnanCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const auto& dtype = input.dtype;
    const auto& a = input.operand;
    if (dtype == ir::DataType::FP32 || dtype == ir::DataType::FP16 || dtype == ir::DataType::BF16) {
        return "__isnan(" + a + ")";
    }
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false) << "Unsupported simt.isnan dtype " << dtype.ToString();
    return "";
}

std::string MakeSimtIsinfCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const auto& dtype = input.dtype;
    const auto& a = input.operand;
    if (dtype == ir::DataType::FP32 || dtype == ir::DataType::FP16 || dtype == ir::DataType::BF16) {
        return "__isinf(" + a + ")";
    }
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false) << "Unsupported simt.isinf dtype " << dtype.ToString();
    return "";
}

std::string MakeSimtIsfiniteCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      input.dtype == ir::DataType::FP16 || input.dtype == ir::DataType::FP32)
        << "simt.isfinite requires FP16 or FP32";
    return "__isfinite(" + input.operand + ")";
}

std::string MakeSimtPopcountCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    if (input.dtype == ir::DataType::UINT32) {
        return "__popc((unsigned int)(" + input.operand + "))";
    }
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, input.dtype == ir::DataType::UINT64)
        << "simt.popcount requires UINT32 or UINT64";
    return "__popc((unsigned long long)(" + input.operand + "))";
}

std::string MakeSimtMulHiCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string rhs = codegen.GetExprAsCode(op->args_[1]);
    const char* intrinsic = nullptr;
    const char* operand_type = nullptr;
    if (input.dtype == ir::DataType::INT32) {
        intrinsic = "__mulhi";
        operand_type = "int";
    } else if (input.dtype == ir::DataType::UINT32) {
        intrinsic = "__umulhi";
        operand_type = "unsigned int";
    } else if (input.dtype == ir::DataType::INT64) {
        intrinsic = "__mul64hi";
        operand_type = "long long";
    } else if (input.dtype == ir::DataType::UINT64) {
        intrinsic = "__umul64hi";
        operand_type = "unsigned long long";
    } else {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false)
            << "Unsupported simt.mul_hi dtype " << input.dtype.ToString();
        return "";
    }
    return "((" + input.dtype.ToCTypeString() + ")" + intrinsic + "((" + operand_type + ")(" + input.operand + "), (" +
           operand_type + ")(" + rhs + ")))";
}

std::string MakeSimtFmodCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, input.dtype == ir::DataType::FP32) << "simt.fmod requires FP32";
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string rhs = codegen.GetExprAsCode(op->args_[1]);
    return pypto::codegen::BuildFP32FmodExpression(input.operand, rhs);
}

std::string MakeSimtSinCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const auto& dtype = input.dtype;
    const auto& a = input.operand;
    if (dtype == ir::DataType::FP32)
        return MakeSimtTrigFP32Codegen(a, true);
    if (dtype == ir::DataType::FP16) {
        const std::string cvt_in = "__cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(" + a + ")";
        return "__cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(" + MakeSimtTrigFP32Codegen(cvt_in, true) +
               ")";
    }
    if (dtype == ir::DataType::BF16) {
        const std::string cvt_in = "__cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(" + a + ")";
        return "__cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(" +
               MakeSimtTrigFP32Codegen(cvt_in, true) + ")";
    }
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false) << "Unsupported simt.sin dtype " << dtype.ToString();
    return "";
}

std::string MakeSimtCosCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const auto& dtype = input.dtype;
    const auto& a = input.operand;
    if (dtype == ir::DataType::FP32)
        return MakeSimtTrigFP32Codegen(a, false);
    if (dtype == ir::DataType::FP16) {
        const std::string cvt_in = "__cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(" + a + ")";
        return "__cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(" + MakeSimtTrigFP32Codegen(cvt_in, false) +
               ")";
    }
    if (dtype == ir::DataType::BF16) {
        const std::string cvt_in = "__cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(" + a + ")";
        return "__cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(" +
               MakeSimtTrigFP32Codegen(cvt_in, false) + ")";
    }
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false) << "Unsupported simt.cos dtype " << dtype.ToString();
    return "";
}

std::string MakeSimtMinCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, codegen.IsInSimtContext())
        << "simt.min reached CCE codegen outside a SIMT function";
    PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, codegen.GetArch() == npu::tile_fwk::NPUArch::DAV_3510)
        << "simt.min currently requires arch='3510'";
    auto scalar_type = ir::As<ir::ScalarType>(op->args_[0]->GetType());
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, scalar_type != nullptr) << "simt.min operand must be a scalar";
    const auto& dtype = scalar_type->dtype_;
    const std::string a = codegen.GetExprAsCode(op->args_[0]);
    const std::string b = codegen.GetExprAsCode(op->args_[1]);
    if (dtype.IsInt()) {
        std::string cpp_type = dtype.ToCTypeString();
        return "min((" + cpp_type + ")(" + a + "), (" + cpp_type + ")(" + b + "))";
    }
    if (dtype == ir::DataType::FP32) {
        return "(__isnan(" + a + ") ? " + b + " : (__isnan(" + b + ") ? " + a + " : __fminf(" + a + ", " + b + ")))";
    }
    if (dtype == ir::DataType::FP16) {
        return "(__isnan(" + a + ") ? " + b + " : (__isnan(" + b + ") ? " + a + " : __hmin_nan(" + a + ", " + b + ")))";
    }
    if (dtype == ir::DataType::BF16) {
        return "(__isnan(" + a + ") ? " + b + " : (__isnan(" + b + ") ? " + a + " : __min(" + a + ", " + b + ")))";
    }
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false) << "Unsupported simt.min dtype " << dtype.ToString();
    return "";
}

std::string MakeSimtMaxCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, codegen.IsInSimtContext())
        << "simt.max reached CCE codegen outside a SIMT function";
    PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, codegen.GetArch() == npu::tile_fwk::NPUArch::DAV_3510)
        << "simt.max currently requires arch='3510'";
    auto scalar_type = ir::As<ir::ScalarType>(op->args_[0]->GetType());
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, scalar_type != nullptr) << "simt.max operand must be a scalar";
    const auto& dtype = scalar_type->dtype_;
    const std::string a = codegen.GetExprAsCode(op->args_[0]);
    const std::string b = codegen.GetExprAsCode(op->args_[1]);
    if (dtype.IsInt()) {
        std::string cpp_type = dtype.ToCTypeString();
        return "max((" + cpp_type + ")(" + a + "), (" + cpp_type + ")(" + b + "))";
    }
    if (dtype == ir::DataType::FP32) {
        return "(__isnan(" + a + ") ? " + b + " : (__isnan(" + b + ") ? " + a + " : __fmaxf(" + a + ", " + b + ")))";
    }
    if (dtype == ir::DataType::FP16) {
        return "(__isnan(" + a + ") ? " + b + " : (__isnan(" + b + ") ? " + a + " : __hmax_nan(" + a + ", " + b + ")))";
    }
    if (dtype == ir::DataType::BF16) {
        return "(__isnan(" + a + ") ? " + b + " : (__isnan(" + b + ") ? " + a + " : __max(" + a + ", " + b + ")))";
    }
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false) << "Unsupported simt.max dtype " << dtype.ToString();
    return "";
}

std::string MakeSimtFmaCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, codegen.IsInSimtContext())
        << "simt.fma reached CCE codegen outside a SIMT function";
    PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, codegen.GetArch() == npu::tile_fwk::NPUArch::DAV_3510)
        << "simt.fma currently requires arch='3510'";
    return "__fma(" + codegen.GetExprAsCode(op->args_[0]) + ", " + codegen.GetExprAsCode(op->args_[1]) + ", " +
           codegen.GetExprAsCode(op->args_[2]) + ")";
}

struct SimtAtomicSpec {
    const char* intrinsic;
    size_t operand_count;
};

SimtAtomicSpec GetSimtAtomicSpec(const std::string& op_name)
{
    if (op_name == "simt.atomic_add") {
        return {"atomicAdd", 1};
    }
    if (op_name == "simt.atomic_sub") {
        return {"atomicSub", 1};
    }
    if (op_name == "simt.atomic_exch") {
        return {"atomicExch", 1};
    }
    if (op_name == "simt.atomic_max") {
        return {"atomicMax", 1};
    }
    if (op_name == "simt.atomic_min") {
        return {"atomicMin", 1};
    }
    if (op_name == "simt.atomic_inc") {
        return {"atomicInc", 1};
    }
    if (op_name == "simt.atomic_dec") {
        return {"atomicDec", 1};
    }
    if (op_name == "simt.atomic_cas") {
        return {"atomicCAS", 2};
    }
    if (op_name == "simt.atomic_and") {
        return {"atomicAnd", 1};
    }
    if (op_name == "simt.atomic_or") {
        return {"atomicOr", 1};
    }
    if (op_name == "simt.atomic_xor") {
        return {"atomicXOr", 1};
    }
    PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, false) << "Unsupported SIMT atomic operation " << op_name;
    return {"", 0};
}

std::string MakeSimtAtomicCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    SimtAtomicSpec spec = GetSimtAtomicSpec(op->name_);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, codegen.IsInSimtContext())
        << op->name_ << " reached CCE codegen outside a SIMT function";
    PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, codegen.GetArch() == npu::tile_fwk::NPUArch::DAV_3510)
        << op->name_ << " currently requires arch='3510'";
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == spec.operand_count + 2)
        << op->name_ << " requires container, offset, and " << spec.operand_count << " scalar operand(s)";

    auto tile_type = ir::As<ir::TileType>(op->args_[0]->GetType());
    auto tensor_type = ir::As<ir::TensorType>(op->args_[0]->GetType());
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, tile_type || tensor_type)
        << op->name_ << " container must be a Tile or Tensor";

    std::string base;
    if (tile_type) {
        auto tile_var = ir::As<ir::Var>(op->args_[0]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, tile_var != nullptr)
            << op->name_ << " Tile container must be a Var";
        base = codegen.GetExprAsCode(op->args_[0]);
    } else {
        auto tensor_var = ir::As<ir::Var>(op->args_[0]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, tensor_var != nullptr)
            << op->name_ << " Tensor container must be a Var";
        base = codegen.GetPointer(codegen.GetVarName(tensor_var));
    }
    std::string offset = codegen.GetExprAsCode(op->args_[1]);
    std::stringstream call;
    call << spec.intrinsic << "(" << base << " + (" << offset << ")";
    for (size_t i = 0; i < spec.operand_count; ++i) {
        call << ", " << codegen.GetExprAsCode(op->args_[i + 2]);
    }
    call << ")";
    if (ir::As<ir::NoneType>(op->GetType())) {
        codegen.Emit(call.str() + ";");
        return "";
    }
    std::string result = "__simt_atomic_result_" + std::to_string(codegen.GetTileOffsetCounter());
    codegen.Emit("auto " + result + " = " + call.str() + ";");
    return result;
}

std::string MakeSimtLaunchCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, !codegen.IsInSimtContext())
        << "Nested simt.launch is not supported";
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, codegen.GetTarget() == ir::SectionKind::Vector)
        << "simt.launch requires the Vector target";
    PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, codegen.GetArch() == npu::tile_fwk::NPUArch::DAV_3510)
        << "simt.launch currently requires arch='3510'";
    int64_t thread_dims[3] = {};
    for (size_t i = 0; i < 3; ++i) {
        auto dim = ir::As<ir::ConstInt>(op->args_[i]);
        thread_dims[i] = dim->value_;
    }

    std::ostringstream call;
    call << "cce::async_invoke<" << op->GetKwarg<std::string>("callee") << ">(cce::dim3{" << thread_dims[0] << ", "
         << thread_dims[1] << ", " << thread_dims[2] << "}";
    for (size_t i = 3; i < op->args_.size(); ++i) {
        auto arg_type = op->args_[i]->GetType();
        auto tile_type = ir::As<ir::TileType>(arg_type);
        if (tile_type != nullptr) {
            std::string tile = codegen.GetExprAsCode(op->args_[i]);
            call << ", (__ubuf__ " << tile_type->dtype_.ToCTypeString() << "*)" << tile << ".data()";
            call << ", (uint32_t)" << tile << ".GetValidRow(), (uint32_t)" << tile << ".GetValidCol()";
        } else if (auto tensor_type = ir::As<ir::TensorType>(arg_type)) {
            auto tensor_var = ir::As<ir::Var>(op->args_[i]);
            std::string tensor_name = codegen.GetVarName(tensor_var);
            call << ", (__gm__ " << tensor_type->dtype_.ToCTypeString() << "*)" << codegen.GetPointer(tensor_name);
        } else {
            call << ", " << codegen.GetExprAsCode(op->args_[i]);
        }
    }
    call << ");";
    codegen.Emit(call.str());
    return "";
}

} // namespace

REGISTER_BACKEND_OP(BackendCCE, "simt.thread_idx").set_pipe(ir::PipeType::S).f_codegen(MakeSimtThreadIdxCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.block_dim").set_pipe(ir::PipeType::S).f_codegen(MakeSimtBlockDimCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.block_idx").set_pipe(ir::PipeType::S).f_codegen(MakeSimtBlockIdxCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.grid_dim").set_pipe(ir::PipeType::S).f_codegen(MakeSimtGridDimCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.linear_thread_idx")
    .set_pipe(ir::PipeType::S)
    .f_codegen(MakeSimtLinearThreadIdxCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.warp_size").set_pipe(ir::PipeType::S).f_codegen(MakeSimtWarpSizeCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.syncthreads").set_pipe(ir::PipeType::S).f_codegen(MakeSimtSyncthreadsCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.threadfence_block")
    .set_pipe(ir::PipeType::S)
    .f_codegen(MakeSimtThreadfenceBlockCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.threadfence").set_pipe(ir::PipeType::S).f_codegen(MakeSimtThreadfenceCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.cast").set_pipe(ir::PipeType::S).f_codegen(MakeSimtCastCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.bitcast").set_pipe(ir::PipeType::S).f_codegen(MakeSimtBitcastCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.abs").set_pipe(ir::PipeType::S).f_codegen(MakeSimtAbsCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.sqrt").set_pipe(ir::PipeType::S).f_codegen(MakeSimtSqrtCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.rsqrt").set_pipe(ir::PipeType::S).f_codegen(MakeSimtRsqrtCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.exp").set_pipe(ir::PipeType::S).f_codegen(MakeSimtExpCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.exp2").set_pipe(ir::PipeType::S).f_codegen(MakeSimtExp2CodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.log").set_pipe(ir::PipeType::S).f_codegen(MakeSimtLogCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.log2").set_pipe(ir::PipeType::S).f_codegen(MakeSimtLog2CodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.log1p").set_pipe(ir::PipeType::S).f_codegen(MakeSimtLog1pCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.tanh").set_pipe(ir::PipeType::S).f_codegen(MakeSimtTanhCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.rint").set_pipe(ir::PipeType::S).f_codegen(MakeSimtRintCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.round").set_pipe(ir::PipeType::S).f_codegen(MakeSimtRoundCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.floor").set_pipe(ir::PipeType::S).f_codegen(MakeSimtFloorCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.ceil").set_pipe(ir::PipeType::S).f_codegen(MakeSimtCeilCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.trunc").set_pipe(ir::PipeType::S).f_codegen(MakeSimtTruncCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.isnan").set_pipe(ir::PipeType::S).f_codegen(MakeSimtIsnanCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.isinf").set_pipe(ir::PipeType::S).f_codegen(MakeSimtIsinfCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.isfinite").set_pipe(ir::PipeType::S).f_codegen(MakeSimtIsfiniteCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.popcount").set_pipe(ir::PipeType::S).f_codegen(MakeSimtPopcountCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.mul_hi").set_pipe(ir::PipeType::S).f_codegen(MakeSimtMulHiCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.fmod").set_pipe(ir::PipeType::S).f_codegen(MakeSimtFmodCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.sin").set_pipe(ir::PipeType::S).f_codegen(MakeSimtSinCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.cos").set_pipe(ir::PipeType::S).f_codegen(MakeSimtCosCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.min").set_pipe(ir::PipeType::S).f_codegen(MakeSimtMinCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.max").set_pipe(ir::PipeType::S).f_codegen(MakeSimtMaxCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.fma").set_pipe(ir::PipeType::S).f_codegen(MakeSimtFmaCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.atomic_add").set_pipe(ir::PipeType::S).f_codegen(MakeSimtAtomicCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.atomic_sub").set_pipe(ir::PipeType::S).f_codegen(MakeSimtAtomicCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.atomic_exch").set_pipe(ir::PipeType::S).f_codegen(MakeSimtAtomicCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.atomic_max").set_pipe(ir::PipeType::S).f_codegen(MakeSimtAtomicCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.atomic_min").set_pipe(ir::PipeType::S).f_codegen(MakeSimtAtomicCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.atomic_inc").set_pipe(ir::PipeType::S).f_codegen(MakeSimtAtomicCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.atomic_dec").set_pipe(ir::PipeType::S).f_codegen(MakeSimtAtomicCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.atomic_cas").set_pipe(ir::PipeType::S).f_codegen(MakeSimtAtomicCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.atomic_and").set_pipe(ir::PipeType::S).f_codegen(MakeSimtAtomicCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.atomic_or").set_pipe(ir::PipeType::S).f_codegen(MakeSimtAtomicCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.atomic_xor").set_pipe(ir::PipeType::S).f_codegen(MakeSimtAtomicCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.launch").set_pipe(ir::PipeType::V).f_codegen(MakeSimtLaunchCodegenCCE);

} // namespace backend
} // namespace pypto
