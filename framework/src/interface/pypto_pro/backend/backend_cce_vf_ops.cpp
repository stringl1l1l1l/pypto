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
 * @file backend_950_cce_vf_ops.cpp
 * \brief CCE backend op registration for VF API operations (A5 target).
 *
 * VF ops directly emit CCE intrinsics (vlds, vmax, vdup, etc.) without
 * going through the PTO-ISA intermediate layer.
 */

#include <string>
#include <unordered_map>

#include "backend/backend_cce.h"
#include "backend/common/backend.h"
#include "codegen/cce/cce_codegen.h"
#include "codegen/codegen_base.h"
#include "core/logging.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/pipe.h"
#include "ir/op_attr_types.h"
#include "ir/type.h"
#include "pypto_pro/error.h"
#include "tilefwk/error.h"

namespace pypto {
namespace backend {
using ir::DataType;
using npu::tile_fwk::ExternalError;

static std::string VFEnumValueName(const char* full_name)
{
    const char* sep = std::strrchr(full_name, ':');
    return sep ? std::string(sep + 1) : std::string(full_name);
}

// The dist kwarg is type-erased to a raw int at the Python/C++ boundary
// (ConvertKwargsDict extracts the pybind enum's .value), so the backend cannot
// tell which dist enum type the value came from. It can still reject ints that
// are not a valid enumerator of the op's own dist enum — EnumToString yields
// "UNKNOWN" for those — e.g. the load-only DINTLV/E2B_B16+ values passed to
// store_align. In-range cross-enum values are undetectable here.
static std::string VFCheckedDistName(int dist_val, bool is_load, const std::string& op_name)
{
    const char* full_name = is_load ? ir::EnumToString(static_cast<ir::LoadDist>(dist_val)) :
                                      ir::EnumToString(static_cast<ir::StoreDist>(dist_val));
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, std::strcmp(full_name, "UNKNOWN") != 0)
        << op_name << " dist=" << dist_val << " is not a valid " << (is_load ? "ir::LoadDist" : "ir::StoreDist")
        << " enumerator";
    return VFEnumValueName(full_name);
}

// Format a DataType for log messages in the frontend DT_XXX style.
static std::string DTypeStr(const DataType& dt) { return "DT_" + ir::DTypeToString(dt); }

// Returns true when the dst argument (args_[0], or args_[0]/args_[1] for 2-dst
// ops) is a MaskReg variable. Used by unified emitters to dispatch between
// v* (RegTensor) and p* (MaskReg) CCE intrinsics.
static bool IsDstMaskReg(const ir::CallPtr& op, codegen::CCECodegen& codegen, size_t idx = 0)
{
    if (idx >= op->args_.size())
        return false;
    auto dst_var = ir::As<ir::Var>(op->args_[idx]);
    if (dst_var) {
        return codegen.IsMaskRegVar(codegen.GetVarName(dst_var));
    }
    return false;
}

// ============================================================================
// RegTensor declaration
// ============================================================================

static std::string EmitVFRegTensor(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    auto dtype = op->GetKwarg<DataType>("dtype");
    std::string reg_name = codegen.GetCurrentResultTarget();
    std::string decl;
    if (dtype == DataType::INT4) {
        decl = "vector_s4x2 " + reg_name + ";";
    } else if (dtype == DataType::UINT4) {
        decl = "vector_u4x2 " + reg_name + ";";
    } else {
        decl = "RegTensor<" + dtype.ToCTypeString() + "> " + reg_name + ";";
    }
    codegen.HoistRegTensorDecl(decl);
    codegen.RegisterRegTensorVar(reg_name);
    return "";
}

// ============================================================================
// MaskReg declaration (no initialization — unlike create_mask which emits pset)
// ============================================================================

static std::string EmitVFMaskReg(const ir::CallPtr& /*op*/, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    std::string reg_name = codegen.GetCurrentResultTarget();
    codegen.HoistRegTensorDecl("MaskReg " + reg_name + ";");
    codegen.RegisterMaskRegVar(reg_name);
    return "";
}

// Best-effort dtype extraction from an Expr's deduced type. Handles both
// ScalarType (RegTensor outputs) and ShapedType (Tile / Tensor expressions).
static DataType GetExprDtype(const ir::ExprPtr& expr, DataType fallback = DataType::UINT32)
{
    auto type = expr->GetType();
    if (auto st = ir::As<ir::ScalarType>(type))
        return st->dtype_;
    if (auto sh = ir::As<ir::ShapedType>(type))
        return sh->dtype_;
    return fallback;
}

// Coerce a float-constant scalar to an integer literal matching the src type,
// so that the emitted intrinsic (vmuls/vadds/vmins/vmaxs/vcmps_*/vaxpy)
// receives a correctly-typed integer literal instead of a float literal
// that would trigger -Wliteral-conversion. The value is truncated to the width
// of src_dt, matching AscendC's implicit float→int conversion. The front end now
// rejects a scalar the src type cannot represent, so what reaches here is only
// the fractional part being dropped — not a value silently wrapping.
static std::string CoerceScalarToInt(const ir::ExprPtr& scalar_expr, const DataType& src_dt,
                                     const std::string& original_str)
{
    if (!src_dt.IsInt()) {
        return original_str;
    }
    auto cf = ir::As<ir::ConstFloat>(scalar_expr);
    if (cf == nullptr) {
        return original_str;
    }
    const size_t bits = src_dt.GetBit();
    if (bits == 0 || bits > 64) {
        return original_str;
    }
    const auto raw = static_cast<uint64_t>(static_cast<int64_t>(cf->value_));
    const uint64_t truncated = (bits == 64) ? raw : (raw & ((static_cast<uint64_t>(1) << bits) - 1));
    if (src_dt.IsUnsignedInt()) {
        return std::to_string(truncated) + "u";
    }
    // Sign-extend back out of the truncated width.
    const uint64_t sign_bit = static_cast<uint64_t>(1) << (bits - 1);
    return std::to_string(static_cast<int64_t>((truncated ^ sign_bit) - sign_bit));
}

// For ops that only support ZEROING: return "MODE_ZEROING", reject MERGING.
static std::string VFZeroingOnly(const ir::CallPtr& op, const std::string& op_name)
{
    if (op->HasKwarg("mode")) {
        auto mode = static_cast<ir::MergeMode>(op->GetKwarg<int>("mode"));
        PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, mode == ir::MergeMode::ZEROING)
            << op_name << " only supports ZEROING mode on current device";
    }
    return "MODE_ZEROING";
}

// Check if a DataType is a b8-width type (8-bit storage).
// Includes INT8, UINT8, BOOL, and all FP8 types (FP8E4M3FN, FP8E5M2, HF8).
// FP4 types (FP4E2M1, FP4E1M2, FP4) are b4 but stored as b8 (packed 2-per-byte),
// so they are also treated as b8 for load/store mode selection.
static bool IsB8Type(DataType dt) { return dt.GetBit() <= 8; }

// Integer types with a native VF arithmetic instruction form (vadd/vsub/vmax/
// vcmp ... have no 4-bit form, so INT4/UINT4 must be rejected up front).
static bool IsArithIntType(DataType dt) { return dt.IsInt() && dt.GetBit() >= 8; }
static bool IsArithSignedIntType(DataType dt) { return dt.IsSignedInt() && dt.GetBit() >= 8; }

// Check if a DataType is a b16-width type (16-bit storage).
static bool IsB16Type(DataType dt) { return dt.GetBit() == 16; }

// ============================================================================
// B64 helper: deinterleave a b64 RegTensor into b32 low/high halves.
// Emits temp RegTensor declarations and a vdintlv instruction.
// After call, {prefix}_lo_ holds the low 32 bits and {prefix}_hi_ holds the
// high 32 bits of each b64 element (32 meaningful elements in positions 0-31).
// ============================================================================
static void EmitB64Deinterleave(codegen::CCECodegen& codegen, const std::string& prefix, const std::string& b64_reg)
{
    std::string lo = prefix + "_lo_";
    std::string hi = prefix + "_hi_";
    std::string zero = prefix + "_zero_";
    std::string dump = prefix + "_ddump_";
    std::string m = prefix + "_dm_";
    codegen.Emit("RegTensor<uint32_t> " + lo + ";");
    codegen.Emit("RegTensor<uint32_t> " + hi + ";");
    codegen.Emit("RegTensor<uint32_t> " + zero + ";");
    codegen.Emit("RegTensor<uint32_t> " + dump + ";");
    codegen.Emit("MaskReg " + m + " = pset_b32(PAT_ALL);");
    codegen.Emit("vdup(" + zero + ", 0, " + m + ", MODE_ZEROING);");
    codegen.Emit("vdintlv(" + lo + ", " + hi + ", (RegTensor<uint32_t>&)" + b64_reg + ", " + zero + ");");
}

// ============================================================================
// B64 helper: interleave b32 low/high halves into a b64 RegTensor.
// Emits a vintlv instruction that combines lo and hi into b64_dst.
// ============================================================================
static void EmitB64Interleave(codegen::CCECodegen& codegen, const std::string& b64_dst, const std::string& lo,
                              const std::string& hi, const std::string& prefix)
{
    std::string dump = prefix + "_idump_";
    codegen.Emit("RegTensor<uint32_t> " + dump + ";");
    codegen.Emit("vintlv((RegTensor<uint32_t>&)" + b64_dst + ", " + dump + ", (RegTensor<uint32_t>&)" + lo +
                 ", (RegTensor<uint32_t>&)" + hi + ");");
}

// ============================================================================
// B64 helper: emit a b64 ZEROING mask application.
// Emits vsel(dst, src, zero_b64, mask) to zero out inactive b64 elements.
// The b64 mask is packed to b32 via ppack, then expanded via pintlv_b32
// (mirrors AscendC CopyMerging: ppack + pintlv_b32 to produce a b32 mask
// where each b64 element's 2 b32 halves share the same mask bit).
// ============================================================================
static void EmitB64Zeroing(codegen::CCECodegen& codegen, const std::string& dst, const std::string& src,
                           const std::string& mask, const std::string& prefix)
{
    std::string zero = prefix + "_zero64_";
    std::string m = prefix + "_zm_";
    std::string packed_m = prefix + "_pm_";
    std::string expanded_m = prefix + "_em_";
    std::string dump_m = prefix + "_dm_";
    codegen.Emit("RegTensor<uint32_t> " + zero + ";");
    codegen.Emit("MaskReg " + m + " = pset_b32(PAT_ALL);");
    codegen.Emit("vdup(" + zero + ", 0, " + m + ", MODE_ZEROING);");
    // Pack b64 mask to b32, then expand back to b32 width (mirrors AscendC
    // CopyMerging: ppack + pintlv_b32)
    codegen.Emit("MaskReg " + packed_m + ";");
    codegen.Emit("ppack(" + packed_m + ", " + mask + ", LOWER);");
    codegen.Emit("MaskReg " + expanded_m + ", " + dump_m + ";");
    codegen.Emit("pintlv_b32(" + expanded_m + ", " + dump_m + ", " + packed_m + ", " + packed_m + ");");
    codegen.Emit("vsel((RegTensor<uint32_t>&)" + dst + ", (RegTensor<uint32_t>&)" + src + ", " + zero + ", " +
                 expanded_m + ");");
}

// ============================================================================
// B64 helper: emit a b32 bitwise instruction over b64 registers with a b64
// mask (mirrors AscendC AndImpl/XorImpl/OrImpl/NotImpl b64 path): pack the b64
// mask (2-bit-per-element) to a b32 mask via ppack, deinterleave each operand
// into low/high u32 halves, apply the b32 bitwise instruction on both halves
// with the packed mask, then interleave the halves back into the b64 dst.
// MODE_ZEROING zeroes both halves of an inactive element, matching b64 zeroing
// semantics. Bitwise ops are sign-agnostic, so u32 halves are used for both
// INT64 and UINT64.
// ============================================================================
static void EmitB64Bitwise(codegen::CCECodegen& codegen, const std::string& instruction,
                           const std::vector<std::string>& srcs, const std::string& dst, const std::string& mask,
                           const std::string& mode, const std::string& prefix)
{
    std::string packed_m = prefix + "_pm_";
    codegen.Emit("MaskReg " + packed_m + ";");
    codegen.Emit("ppack(" + packed_m + ", " + mask + ", LOWER);");
    std::string lo_args;
    std::string hi_args;
    for (size_t i = 0; i < srcs.size(); ++i) {
        EmitB64Deinterleave(codegen, prefix + "s" + std::to_string(i), srcs[i]);
        lo_args += prefix + "s" + std::to_string(i) + "_lo_, ";
        hi_args += prefix + "s" + std::to_string(i) + "_hi_, ";
    }
    std::string lo_d = prefix + "_lod_";
    std::string hi_d = prefix + "_hid_";
    codegen.Emit("RegTensor<uint32_t> " + lo_d + ";");
    codegen.Emit("RegTensor<uint32_t> " + hi_d + ";");
    codegen.Emit(instruction + "(" + lo_d + ", " + lo_args + packed_m + ", " + mode + ");");
    codegen.Emit(instruction + "(" + hi_d + ", " + hi_args + packed_m + ", " + mode + ");");
    EmitB64Interleave(codegen, dst, lo_d, hi_d, prefix + "_ilv");
}

// ============================================================================
// B64 helper: reduce SUM via the hierarchical 16-bit carry decomposition
// (mirrors AscendC ReduceSumB64Impl). The b64 src is deinterleaved into
// low/high u32 halves; each 64-bit element is split into 16-bit lanes whose
// vcadd partial sums are combined with carries.
// ============================================================================
static void EmitB64ReduceSum(codegen::CCECodegen& codegen, const std::string& dst, const std::string& src,
                             const std::string& mask)
{
    const std::string p = dst + "_b64rs_";
    const std::string pm = p + "pm";
    codegen.Emit("MaskReg " + pm + ";");
    codegen.Emit("ppack(" + pm + ", " + mask + ", LOWER);");
    EmitB64Deinterleave(codegen, p + "s", src);
    const std::string lo_s = p + "s_lo_";
    const std::string hi_s = p + "s_hi_";
    const std::string low_f = p + "lowf_";
    const std::string low_r = p + "lowr_";
    const std::string mid_r = p + "midr_";
    const std::string hi_r = p + "hir_";
    const std::string tmp_r = p + "tmpr_";
    const std::string lo32 = p + "lo32_";
    const std::string dump = p + "dump_";
    codegen.Emit("RegTensor<uint32_t> " + low_f + ";");
    codegen.Emit("RegTensor<uint32_t> " + low_r + ";");
    codegen.Emit("RegTensor<uint32_t> " + mid_r + ";");
    codegen.Emit("RegTensor<uint32_t> " + hi_r + ";");
    codegen.Emit("RegTensor<uint32_t> " + tmp_r + ";");
    codegen.Emit("RegTensor<uint32_t> " + lo32 + ";");
    codegen.Emit("RegTensor<uint32_t> " + dump + ";");
    codegen.Emit("vdup(" + low_f + ", (int32_t)0xFFFF, " + pm + ", MODE_ZEROING);");
    codegen.Emit("vand(" + low_r + ", " + low_f + ", " + lo_s + ", " + pm + ", MODE_ZEROING);");
    codegen.Emit("vcadd(" + low_r + ", " + low_r + ", " + pm + ", MODE_ZEROING);");
    codegen.Emit("vshrs(" + mid_r + ", " + lo_s + ", (int16_t)16, " + pm + ", MODE_ZEROING);");
    codegen.Emit("vcadd(" + mid_r + ", " + mid_r + ", " + pm + ", MODE_ZEROING);");
    codegen.Emit("vcadd(" + hi_r + ", " + hi_s + ", " + pm + ", MODE_ZEROING);");
    codegen.Emit("vshrs(" + tmp_r + ", " + low_r + ", (int16_t)16, " + pm + ", MODE_ZEROING);");
    codegen.Emit("vadd(" + mid_r + ", " + mid_r + ", " + tmp_r + ", " + pm + ", MODE_ZEROING);");
    codegen.Emit("vshrs(" + tmp_r + ", " + mid_r + ", (int16_t)16, " + pm + ", MODE_ZEROING);");
    codegen.Emit("vadd(" + hi_r + ", " + hi_r + ", " + tmp_r + ", " + pm + ", MODE_ZEROING);");
    codegen.Emit("vand(" + low_r + ", " + low_r + ", " + low_f + ", " + pm + ", MODE_ZEROING);");
    codegen.Emit("vand(" + mid_r + ", " + mid_r + ", " + low_f + ", " + pm + ", MODE_ZEROING);");
    codegen.Emit("vintlv((RegTensor<uint16_t>&)" + lo32 + ", (RegTensor<uint16_t>&)" + tmp_r +
                 ", (RegTensor<uint16_t>&)" + low_r + ", (RegTensor<uint16_t>&)" + mid_r + ");");
    codegen.Emit("vintlv((RegTensor<uint32_t>&)" + dst + ", (RegTensor<uint32_t>&)" + dump +
                 ", (RegTensor<uint32_t>&)" + lo32 + ", (RegTensor<uint32_t>&)" + hi_r + ");");
}

static void EmitB64ReduceMaxMin(codegen::CCECodegen& codegen, const std::string& dst, const std::string& src,
                                const std::string& mask, bool is_max, bool is_signed)
{
    const std::string p = dst + "_b64rm_";
    const std::string pm = p + "pm";
    codegen.Emit("MaskReg " + pm + ";");
    codegen.Emit("ppack(" + pm + ", " + mask + ", LOWER);");
    EmitB64Deinterleave(codegen, p + "s", src);
    const std::string lo_s = p + "s_lo_";
    const std::string hi_s = p + "s_hi_";
    const std::string red = is_max ? "vcmax" : "vcmin";
    const std::string hi_r = p + "hir_";
    const std::string bcast = p + "bc_";
    const std::string eq_m = p + "eqm_";
    const std::string lo_r = p + "lor_";
    const std::string dump = p + "dump_";
    codegen.Emit("MaskReg " + eq_m + ";");
    codegen.Emit("RegTensor<uint32_t> " + hi_r + ";");
    codegen.Emit("RegTensor<uint32_t> " + bcast + ";");
    codegen.Emit("RegTensor<uint32_t> " + lo_r + ";");
    codegen.Emit("RegTensor<uint32_t> " + dump + ";");
    // The hi half holds the SIGNED b32 view of an int64 element: the hi-half
    // reduce must compare signed for INT64 (a negative int64 has a hi half
    // that is huge when viewed as u32; mirrors Int64RowMinMax's s32/u32
    // split). The lo half is an unsigned magnitude in both cases.
    const std::string hi_sc = is_signed ? "(RegTensor<int32_t>&)" : "(RegTensor<uint32_t>&)";
    codegen.Emit(red + "(" + hi_sc + hi_r + ", " + hi_sc + hi_s + ", " + pm + ", MODE_ZEROING);");
    // Register broadcast uses the 5-arg vdup (mask + POS + MODE); the 4-arg
    // form only accepts a scalar (mirrors Int64RowMinMax's highDup broadcast).
    codegen.Emit("vdup(" + bcast + ", " + hi_r + ", " + pm + ", POS_LOWEST, MODE_ZEROING);");
    codegen.Emit("vcmp_eq(" + eq_m + ", " + bcast + ", " + hi_s + ", " + pm + ");");
    codegen.Emit(red + "(" + lo_r + ", " + lo_s + ", " + eq_m + ", MODE_ZEROING);");
    codegen.Emit("vintlv((RegTensor<uint32_t>&)" + dst + ", (RegTensor<uint32_t>&)" + dump +
                 ", (RegTensor<uint32_t>&)" + lo_r + ", (RegTensor<uint32_t>&)" + hi_r + ");");
}

static void EmitB64Div(codegen::CCECodegen& codegen, const std::string& dst, const std::string& src0,
                       const std::string& src1, const std::string& mask, bool is_signed)
{
    const std::string p = dst + "_div_";
    const std::string all_m = p + "_allm_";
    const std::string pm = p + "_pm_";
    const std::string cy = p + "_cy_";
    const std::string cy1 = p + "_cy1_";
    codegen.Emit("MaskReg " + all_m + " = pset_b32(PAT_ALL);");
    codegen.Emit("MaskReg " + pm + ";");
    codegen.Emit("MaskReg " + cy + ";");
    codegen.Emit("MaskReg " + cy1 + ";");
    codegen.Emit("ppack(" + pm + ", " + mask + ", LOWER);");
    // Deinterleave the b64 sources into lo/hi u32 half pairs (each half has 64
    // lanes: 32 meaningful + 32 zeros from the deinterleave zero operand).
    const std::string s0 = p + "s0", s0_lo = s0 + "_lo", s0_hi = s0 + "_hi";
    const std::string s1 = p + "s1", s1_lo = s1 + "_lo", s1_hi = s1 + "_hi";
    const std::string z = p + "z", z_lo = z + "_lo", z_hi = z + "_hi";
    const std::string dz = p + "_dz";
    codegen.Emit("RegTensor<uint32_t> " + s0_lo + ";");
    codegen.Emit("RegTensor<uint32_t> " + s0_hi + ";");
    codegen.Emit("RegTensor<uint32_t> " + s1_lo + ";");
    codegen.Emit("RegTensor<uint32_t> " + s1_hi + ";");
    codegen.Emit("RegTensor<uint32_t> " + z_lo + ";");
    codegen.Emit("RegTensor<uint32_t> " + z_hi + ";");
    codegen.Emit("RegTensor<uint32_t> " + dz + ";");
    codegen.Emit("vdup(" + z_lo + ", 0, " + all_m + ", MODE_ZEROING);");
    codegen.Emit("vdup(" + z_hi + ", 0, " + all_m + ", MODE_ZEROING);");
    codegen.Emit("vdintlv((RegTensor<uint32_t>&)" + s0_lo + ", (RegTensor<uint32_t>&)" + s0_hi +
                 ", (RegTensor<uint32_t>&)" + src0 + ", (RegTensor<uint32_t>&)" + z_lo + ");");
    codegen.Emit("vdintlv((RegTensor<uint32_t>&)" + s1_lo + ", (RegTensor<uint32_t>&)" + s1_hi +
                 ", (RegTensor<uint32_t>&)" + src1 + ", (RegTensor<uint32_t>&)" + z_hi + ");");
    // Composite b64 helper emissions over {x_lo, x_hi} u32 half pairs.
    auto VmulUsingU32 = [&](const std::string& d, const std::string& a, const std::string& b, const std::string& m) {
        codegen.Emit("vmull((RegTensor<uint32_t>&)" + d + "_lo, (RegTensor<uint32_t>&)" + d +
                     "_hi, (RegTensor<uint32_t>&)" + a + "_lo, (RegTensor<uint32_t>&)" + b + "_lo, " + m + ");");
        codegen.Emit("vmula((RegTensor<uint32_t>&)" + d + "_hi, (RegTensor<uint32_t>&)" + a +
                     "_lo, (RegTensor<uint32_t>&)" + b + "_hi, " + m + ", MODE_ZEROING);");
        codegen.Emit("vmula((RegTensor<uint32_t>&)" + d + "_hi, (RegTensor<uint32_t>&)" + a +
                     "_hi, (RegTensor<uint32_t>&)" + b + "_lo, " + m + ", MODE_ZEROING);");
    };
    auto VnotInPlace = [&](const std::string& x, const std::string& m) {
        codegen.Emit("vnot((RegTensor<uint32_t>&)" + x + "_lo, (RegTensor<uint32_t>&)" + x + "_lo, " + m +
                     ", MODE_ZEROING);");
        codegen.Emit("vnot((RegTensor<uint32_t>&)" + x + "_hi, (RegTensor<uint32_t>&)" + x + "_hi, " + m +
                     ", MODE_ZEROING);");
    };
    auto AddB64 = [&](const std::string& d, const std::string& a, const std::string& b, const std::string& m) {
        codegen.Emit("vaddc(" + cy + ", " + d + "_lo, " + a + "_lo, " + b + "_lo, " + m + ");");
        codegen.Emit("vaddcs(" + cy + ", " + d + "_hi, " + a + "_hi, " + b + "_hi, " + cy + ", " + m + ");");
    };
    auto VsubUsingU32 = [&](const std::string& d, const std::string& a, const std::string& b, const std::string& m) {
        codegen.Emit("vsubc(" + cy + ", " + d + "_lo, " + a + "_lo, " + b + "_lo, " + m + ");");
        codegen.Emit("vsubcs(" + cy + ", " + d + "_hi, " + a + "_hi, " + b + "_hi, " + cy + ", " + m + ");");
    };
    auto VaddUsingU32 = [&](const std::string& d, const std::string& a, const std::string& b, const std::string& m) {
        codegen.Emit("vaddc(" + cy + ", " + d + "_lo, " + a + "_lo, " + b + "_lo, " + m + ");");
        codegen.Emit("vaddcs(" + cy + ", " + d + "_hi, " + a + "_hi, " + b + "_hi, " + cy + ", " + m + ");");
    };
    auto VselUsingU32 = [&](const std::string& d, const std::string& a, const std::string& b, const std::string& m) {
        codegen.Emit("vsel((RegTensor<uint32_t>&)" + d + "_lo, (RegTensor<uint32_t>&)" + a +
                     "_lo, (RegTensor<uint32_t>&)" + b + "_lo, " + m + ");");
        codegen.Emit("vsel((RegTensor<uint32_t>&)" + d + "_hi, (RegTensor<uint32_t>&)" + a +
                     "_hi, (RegTensor<uint32_t>&)" + b + "_hi, " + m + ");");
    };
    auto VcmpEqUsingU32 = [&](const std::string& d, const std::string& a, const std::string& b, const std::string& m) {
        codegen.Emit("vcmp_eq(" + d + "_l, " + a + "_lo, " + b + "_lo, " + m + ");");
        codegen.Emit("vcmp_eq(" + d + "_h, " + a + "_hi, " + b + "_hi, " + m + ");");
        codegen.Emit("pand(" + d + ", " + d + "_l, " + d + "_h, " + m + ");");
    };
    auto VcmpGeUsingU32 = [&](const std::string& d, const std::string& a, const std::string& b, const std::string& m) {
        codegen.Emit("vcmp_eq(" + d + "_heq, " + a + "_hi, " + b + "_hi, " + m + ");");
        codegen.Emit("vcmp_ge(" + d + "_lge, " + a + "_lo, " + b + "_lo, " + m + ");");
        codegen.Emit("vcmp_ge(" + d + "_hge, " + a + "_hi, " + b + "_hi, " + m + ");");
        codegen.Emit("psel(" + d + ", " + d + "_lge, " + d + "_hge, " + d + "_heq);");
    };
    auto VcmpLtUsingU32 = [&](const std::string& d, const std::string& a, const std::string& b, const std::string& m) {
        codegen.Emit("vcmp_eq(" + d + "_heq, " + a + "_hi, " + b + "_hi, " + m + ");");
        codegen.Emit("vcmp_lt(" + d + "_llt, " + a + "_lo, " + b + "_lo, " + m + ");");
        codegen.Emit("vcmp_lt(" + d + "_hlt, " + a + "_hi, " + b + "_hi, " + m + ");");
        codegen.Emit("psel(" + d + ", " + d + "_llt, " + d + "_hlt, " + d + "_heq);");
    };
    auto VbrU32 = [&](const std::string& d, const std::string& hi_val, const std::string& lo_val) {
        codegen.Emit("vdup(" + d + "_hi, (int32_t)(" + hi_val + "), " + all_m + ", MODE_ZEROING);");
        codegen.Emit("vdup(" + d + "_lo, (int32_t)(" + lo_val + "), " + all_m + ", MODE_ZEROING);");
    };
    auto DeclPair = [&](const std::string& base) {
        codegen.Emit("RegTensor<uint32_t> " + base + "_lo;");
        codegen.Emit("RegTensor<uint32_t> " + base + "_hi;");
    };
    auto DeclU32 = [&](const std::string& name) { codegen.Emit("RegTensor<uint32_t> " + name + ";"); };
    // B128Calc: vb = high64(va * vb + s) (mirrors AscendC B128Calc; vb is both
    // read and written, va and s are read-only). Uses two carry masks (mirrors
    // the carry0/carry1 pair in the reference: the second vaddc's carry-out
    // must not clobber the first, which is still consumed by the next vaddcs).
    auto B128Calc = [&](const std::string& tag, const std::string& va, const std::string& vb, const std::string& s,
                        const std::string& m) {
        const std::string m0l = tag + "m0l", m0h = tag + "m0h", m1l = tag + "m1l", m1h = tag + "m1h";
        const std::string m2l = tag + "m2l", m2h = tag + "m2h", m3l = tag + "m3l", m3h = tag + "m3h";
        const std::string dd0 = tag + "dd0", dd1 = tag + "dd1";
        DeclU32(m0l);
        DeclU32(m0h);
        DeclU32(m1l);
        DeclU32(m1h);
        DeclU32(m2l);
        DeclU32(m2h);
        DeclU32(m3l);
        DeclU32(m3h);
        DeclU32(dd0);
        DeclU32(dd1);
        codegen.Emit("vmull((RegTensor<uint32_t>&)" + m0l + ", (RegTensor<uint32_t>&)" + m0h +
                     ", (RegTensor<uint32_t>&)" + va + "_lo, (RegTensor<uint32_t>&)" + vb + "_lo, " + m + ");");
        codegen.Emit("vmull((RegTensor<uint32_t>&)" + m1l + ", (RegTensor<uint32_t>&)" + m1h +
                     ", (RegTensor<uint32_t>&)" + va + "_lo, (RegTensor<uint32_t>&)" + vb + "_hi, " + m + ");");
        codegen.Emit("vmull((RegTensor<uint32_t>&)" + m2l + ", (RegTensor<uint32_t>&)" + m2h +
                     ", (RegTensor<uint32_t>&)" + va + "_hi, (RegTensor<uint32_t>&)" + vb + "_lo, " + m + ");");
        codegen.Emit("vmull((RegTensor<uint32_t>&)" + m3l + ", (RegTensor<uint32_t>&)" + m3h +
                     ", (RegTensor<uint32_t>&)" + va + "_hi, (RegTensor<uint32_t>&)" + vb + "_hi, " + m + ");");
        codegen.Emit("vaddc(" + cy + ", " + dd0 + ", " + m0h + ", " + m1l + ", " + m + ");");
        codegen.Emit("vaddc(" + cy1 + ", " + dd1 + ", " + dd0 + ", " + m2l + ", " + m + ");");
        codegen.Emit("vaddcs(" + cy + ", " + dd0 + ", " + m3l + ", " + m1h + ", " + cy + ", " + m + ");");
        codegen.Emit("vaddcs(" + cy1 + ", " + vb + "_lo, " + dd0 + ", " + m2h + ", " + cy1 + ", " + m + ");");
        codegen.Emit("vaddcs(" + cy + ", " + dd0 + ", " + s + ", " + m3h + ", " + cy + ", " + m + ");");
        codegen.Emit("vaddcs(" + cy + ", " + vb + "_hi, " + s + ", " + dd0 + ", " + cy1 + ", " + m + ");");
    };
    // Abs of src0/src1 via hi-half sign compare + borrow-chain neg + vsel
    // (mirrors AbsUsingS32). Unsigned passes through unchanged.
    const std::string abs0 = p + "abs0", abs1 = p + "abs1";
    const std::string abs0_lo = abs0 + "_lo", abs0_hi = abs0 + "_hi";
    const std::string abs1_lo = abs1 + "_lo", abs1_hi = abs1 + "_hi";
    DeclPair(abs0);
    DeclPair(abs1);
    const std::string ac0 = p + "ac0", ac1 = p + "ac1", ac2 = p + "ac2";
    const std::string at20 = p + "at20", at30 = p + "at30";
    const std::string bc0 = p + "bc0", bc1 = p + "bc1", bc2 = p + "bc2";
    const std::string bt20 = p + "bt20", bt30 = p + "bt30";
    codegen.Emit("MaskReg " + ac0 + ";");
    codegen.Emit("MaskReg " + ac1 + ";");
    codegen.Emit("MaskReg " + ac2 + ";");
    codegen.Emit("RegTensor<int32_t> " + at20 + ";");
    codegen.Emit("RegTensor<int32_t> " + at30 + ";");
    codegen.Emit("MaskReg " + bc0 + ";");
    codegen.Emit("MaskReg " + bc1 + ";");
    codegen.Emit("MaskReg " + bc2 + ";");
    codegen.Emit("RegTensor<int32_t> " + bt20 + ";");
    codegen.Emit("RegTensor<int32_t> " + bt30 + ";");
    if (is_signed) {
        codegen.Emit("vcmp_lt(" + p + "ac0, (RegTensor<int32_t>&)" + s0_hi + ", (RegTensor<int32_t>&)" + z_hi + ", " +
                     pm + ");");
        codegen.Emit("vsubc(" + p + "ac1, (RegTensor<int32_t>&)" + p + "at20, (RegTensor<int32_t>&)" + z_lo +
                     ", (RegTensor<int32_t>&)" + s0_lo + ", " + p + "ac0);");
        codegen.Emit("vsubcs(" + p + "ac2, (RegTensor<int32_t>&)" + p + "at30, (RegTensor<int32_t>&)" + z_hi +
                     ", (RegTensor<int32_t>&)" + s0_hi + ", " + p + "ac1, " + pm + ");");
        codegen.Emit("vsel((RegTensor<int32_t>&)" + abs0_lo + ", (RegTensor<int32_t>&)" + p +
                     "at20, (RegTensor<int32_t>&)" + s0_lo + ", " + p + "ac0);");
        codegen.Emit("vsel((RegTensor<int32_t>&)" + abs0_hi + ", (RegTensor<int32_t>&)" + p +
                     "at30, (RegTensor<int32_t>&)" + s0_hi + ", " + p + "ac0);");
        codegen.Emit("vcmp_lt(" + p + "bc0, (RegTensor<int32_t>&)" + s1_hi + ", (RegTensor<int32_t>&)" + z_hi + ", " +
                     pm + ");");
        codegen.Emit("vsubc(" + p + "bc1, (RegTensor<int32_t>&)" + p + "bt20, (RegTensor<int32_t>&)" + z_lo +
                     ", (RegTensor<int32_t>&)" + s1_lo + ", " + p + "bc0);");
        codegen.Emit("vsubcs(" + p + "bc2, (RegTensor<int32_t>&)" + p + "bt30, (RegTensor<int32_t>&)" + z_hi +
                     ", (RegTensor<int32_t>&)" + s1_hi + ", " + p + "bc1, " + pm + ");");
        codegen.Emit("vsel((RegTensor<int32_t>&)" + abs1_lo + ", (RegTensor<int32_t>&)" + p +
                     "bt20, (RegTensor<int32_t>&)" + s1_lo + ", " + p + "bc0);");
        codegen.Emit("vsel((RegTensor<int32_t>&)" + abs1_hi + ", (RegTensor<int32_t>&)" + p +
                     "bt30, (RegTensor<int32_t>&)" + s1_hi + ", " + p + "bc0);");
    } else {
        codegen.Emit("vmov((RegTensor<int32_t>&)" + abs0_lo + ", (RegTensor<int32_t>&)" + s0_lo + ");");
        codegen.Emit("vmov((RegTensor<int32_t>&)" + abs0_hi + ", (RegTensor<int32_t>&)" + s0_hi + ");");
        codegen.Emit("vmov((RegTensor<int32_t>&)" + abs1_lo + ", (RegTensor<int32_t>&)" + s1_lo + ");");
        codegen.Emit("vmov((RegTensor<int32_t>&)" + abs1_hi + ", (RegTensor<int32_t>&)" + s1_hi + ");");
    }
    // 64-bit constants: qZero = all-ones (div-by-zero result); one = 1.
    const std::string qz = p + "qz", c1 = p + "c1";
    DeclU32(qz + "_lo");
    DeclU32(qz + "_hi");
    DeclU32(c1 + "_lo");
    DeclU32(c1 + "_hi");
    VbrU32(qz, "-1", "-1");
    VbrU32(c1, "0", "1");
    // zeroMask: divisor == 0 (VcmpEqUsingU32 on both halves).
    const std::string zero_m = p + "zm", nonzero_m = p + "nzm";
    const std::string one_m = p + "om", nonone_m = p + "nom";
    codegen.Emit("MaskReg " + zero_m + ", " + zero_m + "_l, " + zero_m + "_h;");
    codegen.Emit("MaskReg " + nonzero_m + ";");
    codegen.Emit("MaskReg " + one_m + ", " + one_m + "_l, " + one_m + "_h;");
    codegen.Emit("MaskReg " + nonone_m + ";");
    VcmpEqUsingU32(zero_m, s1, z, pm);
    codegen.Emit("pnot(" + nonzero_m + ", " + zero_m + ", " + pm + ");");
    // oneMask: abs(divisor) == 1.
    VcmpEqUsingU32(one_m, abs1, c1, pm);
    codegen.Emit("pnot(" + nonone_m + ", " + one_m + ", " + pm + ");");
    // Newton work mask + (unsigned-only) pre-checks (mirrors DivU64Impl):
    // divisor > INT64_MAX (hi half negative as int32) invalidates the f32
    // reciprocal path; the quotient is 1 (src0 >= src1) or 0 (src0 < src1).
    // usrc1_m keeps the "large divisor" meaning until the final bypass select,
    // so the work-mask construction uses a separate notlarge_m mask.
    const std::string work_m = p + "wm", usrc1_m = p + "u1m", srccmp_m = p + "scm", cmpdiv_m = p + "cdm";
    const std::string ge0 = p + "ge0";
    const std::string notlarge_m = p + "nlm";
    const std::string q = p + "q", q_lo = q + "_lo", q_hi = q + "_hi";
    codegen.Emit("MaskReg " + work_m + ";");
    codegen.Emit("MaskReg " + usrc1_m + ";");
    codegen.Emit("MaskReg " + srccmp_m + ";");
    codegen.Emit("MaskReg " + cmpdiv_m + ";");
    codegen.Emit("MaskReg " + ge0 + ";");
    codegen.Emit("MaskReg " + notlarge_m + ";");
    // Temp masks written by VcmpGeUsingU32/VcmpLtUsingU32 (shared across call
    // sites, declared once here).
    codegen.Emit("MaskReg " + srccmp_m + "_heq, " + srccmp_m + "_lge, " + srccmp_m + "_hge;");
    codegen.Emit("MaskReg " + srccmp_m + "_llt, " + srccmp_m + "_hlt;");
    codegen.Emit("MaskReg " + ge0 + "_heq, " + ge0 + "_lge, " + ge0 + "_hge;");
    DeclU32(q_lo);
    DeclU32(q_hi);
    if (!is_signed) {
        codegen.Emit("vcmp_lt(" + usrc1_m + ", (RegTensor<int32_t>&)" + s1_hi + ", (RegTensor<int32_t>&)" + z_hi +
                     ", " + pm + ");");
        VcmpGeUsingU32(srccmp_m, s0, s1, pm);
        codegen.Emit("pand(" + cmpdiv_m + ", " + srccmp_m + ", " + usrc1_m + ", " + pm + ");");
        VselUsingU32(q, c1, z, cmpdiv_m);
        VcmpLtUsingU32(srccmp_m, s0, s1, pm);
        codegen.Emit("pand(" + cmpdiv_m + ", " + srccmp_m + ", " + usrc1_m + ", " + pm + ");");
        VselUsingU32(q, z, c1, cmpdiv_m);
        codegen.Emit("vcmp_ge(" + notlarge_m + ", (RegTensor<int32_t>&)" + s1_hi + ", (RegTensor<int32_t>&)" + z_hi +
                     ", " + pm + ");");
        codegen.Emit("pand(" + work_m + ", " + nonzero_m + ", " + notlarge_m + ", " + pm + ");");
        codegen.Emit("pand(" + work_m + ", " + work_m + ", " + nonone_m + ", " + pm + ");");
    } else {
        codegen.Emit("pand(" + work_m + ", " + nonone_m + ", " + nonzero_m + ", " + pm + ");");
    }
    // Newton-Raphson reciprocal refinement.
    const std::string t2 = p + "t2f", t3 = p + "t3f", t4 = p + "t4u";
    const std::string t2o = p + "t2o", t2c = p + "t2c", t2d = p + "t2d";
    const std::string prl = p + "prl", prh = p + "prh", cf0 = p + "cf0", cf1 = p + "cf1";
    const std::string ci0 = p + "ci0", ci1 = p + "ci1";
    const std::string t5 = p + "t5", t6 = p + "t6", t7 = p + "t7", t8 = p + "t8", t9 = p + "t9";
    const std::string d0 = p + "d0", d1 = p + "d1";
    const std::string t5_lo = t5 + "_lo", t5_hi = t5 + "_hi";
    const std::string t6_lo = t6 + "_lo", t6_hi = t6 + "_hi";
    codegen.Emit("RegTensor<float> " + t2 + ";");
    codegen.Emit("RegTensor<float> " + t3 + ";");
    codegen.Emit("RegTensor<float> " + t2o + ";");
    codegen.Emit("RegTensor<float> " + t2c + ";");
    codegen.Emit("RegTensor<float> " + t2d + ";");
    DeclU32(t4);
    // prl/prh receive the interleaved b64 divisor (s64 view), cf0/cf1 the
    // split f32 halves, ci0/ci1 the converted s64 reciprocal (mirrors
    // Int64ToFloat / FloatToInt64 in the reference).
    codegen.Emit("RegTensor<int64_t> " + prl + ";");
    codegen.Emit("RegTensor<int64_t> " + prh + ";");
    codegen.Emit("RegTensor<float> " + cf0 + ";");
    codegen.Emit("RegTensor<float> " + cf1 + ";");
    codegen.Emit("RegTensor<int64_t> " + ci0 + ";");
    codegen.Emit("RegTensor<int64_t> " + ci1 + ";");
    DeclPair(t5);
    DeclPair(t6);
    DeclPair(t7);
    DeclPair(t8);
    DeclPair(t9);
    DeclU32(d0);
    DeclU32(d1);
    // VcvtS642F32(t2c, abs1): re-interleave the halves to b64, convert to f32,
    // then compact. The s64->f32 vcvt writes its 32 results to the EVEN f32
    // lanes only, so the converted values must be gathered to consecutive
    // lanes 0-31 with a vdintlv before the divide (mirrors Int64ToFloat's
    // vcvt + vdintlv tail).
    // vintlv must use a single element view for all four arguments: the s32
    // v64 overload takes vector_s32 refs for the dst pair and vector_s32
    // values for the lo/hi sources (mirrors Int64ToFloat's (vector_s32&)
    // casts; mixing s32 dsts with u32 srcs matches no overload).
    codegen.Emit("vintlv((RegTensor<int32_t>&)" + prl + ", (RegTensor<int32_t>&)" + prh + ", (RegTensor<int32_t>&)" +
                 abs1_lo + ", (RegTensor<int32_t>&)" + abs1_hi + ");");
    codegen.Emit("vcvt(" + t2 + ", " + prl + ", " + all_m + ", ROUND_R, PART_EVEN);");
    codegen.Emit("vcvt(" + t2o + ", " + prh + ", " + all_m + ", ROUND_R, PART_EVEN);");
    codegen.Emit("vdintlv((RegTensor<float>&)" + t2c + ", (RegTensor<float>&)" + t2d + ", (RegTensor<float>&)" + t2 +
                 ", (RegTensor<float>&)" + t2o + ");");
    // F32PreProcess(t4, t3, t2c): 1.0 / divisor_f32, then the f32 exponent bias.
    codegen.Emit("vdup(" + t3 + ", 1.0f, " + all_m + ", MODE_ZEROING);");
    codegen.Emit("vdiv(" + t3 + ", " + t3 + ", " + t2c + ", " + work_m + ", MODE_ZEROING);");
    codegen.Emit("vadds(" + t4 + ", (RegTensor<uint32_t>&)" + t3 + ", (int32_t)0x1FFFFFFE, " + work_m + ");");
    // VcvtF322S64(t5, t4): f32 -> s64 (the 6-arg overload: ROUND + RS + PART).
    codegen.Emit("vintlv(" + cf0 + ", " + cf1 + ", (RegTensor<float>&)" + t4 + ", (RegTensor<float>&)" + t4 + ");");
    codegen.Emit("vcvt(" + ci0 + ", " + cf0 + ", " + all_m + ", ROUND_Z, RS_DISABLE, PART_EVEN);");
    codegen.Emit("vcvt(" + ci1 + ", " + cf1 + ", " + all_m + ", ROUND_Z, RS_DISABLE, PART_EVEN);");
    codegen.Emit("vdintlv((RegTensor<int32_t>&)" + t5_lo + ", (RegTensor<int32_t>&)" + t5_hi +
                 ", (RegTensor<int32_t>&)" + ci0 + ", (RegTensor<int32_t>&)" + ci1 + ");");
    // Newton refinement: t6 = -(abs1 * t5) + 1.
    VmulUsingU32(t6, abs1, t5, work_m);
    VnotInPlace(t6, work_m);
    AddB64(t6, t6, c1, work_m);
    codegen.Emit("vdup(" + t4 + ", 0, " + all_m + ", MODE_ZEROING);");
    // Refinement rounds via B128Calc (128-bit square/product high parts).
    B128Calc(p + "r1", t5, t6, t4, work_m);
    VaddUsingU32(t7, t5, t6, work_m);
    VmulUsingU32(t6, abs1, t7, work_m);
    VnotInPlace(t6, work_m);
    AddB64(t6, t6, c1, work_m);
    B128Calc(p + "r2", t7, t6, t4, work_m);
    VaddUsingU32(t6, t7, t6, work_m);
    B128Calc(p + "r3", abs0, t6, t4, work_m);
    // Final correction: two conditional subtract + increment rounds.
    VmulUsingU32(t7, t6, abs1, work_m);
    VsubUsingU32(t7, abs0, t7, work_m);
    VcmpGeUsingU32(ge0, t7, abs1, work_m);
    VsubUsingU32(t8, t7, abs1, ge0);
    AddB64(t9, t6, c1, ge0);
    VselUsingU32(t7, t8, t7, ge0);
    VselUsingU32(t6, t9, t6, ge0);
    VcmpGeUsingU32(ge0, t7, abs1, work_m);
    AddB64(t9, t6, c1, ge0);
    VselUsingU32(t6, t9, t6, ge0);
    VselUsingU32(t6, abs0, t6, one_m);
    if (!is_signed) {
        VselUsingU32(t6, q, t6, usrc1_m);
    }
    // DivSignCal + negate the quotient when the operand signs differ (signed).
    // The negation (0 - q) runs under the full mask (mirrors
    // Int64DivSignedRestoreSign: the negate happens on every lane and the
    // following vsel picks the original quotient on same-sign lanes).
    if (is_signed) {
        const std::string s0ge_m = p + "s0ge", s1ge_m = p + "s1ge", sign_m = p + "sign";
        codegen.Emit("MaskReg " + s0ge_m + ", " + s1ge_m + ", " + sign_m + ";");
        codegen.Emit("vcmp_ge(" + s0ge_m + ", (RegTensor<int32_t>&)" + s0_hi + ", (RegTensor<int32_t>&)" + z_lo + ", " +
                     pm + ");");
        codegen.Emit("vcmp_ge(" + s1ge_m + ", (RegTensor<int32_t>&)" + s1_hi + ", (RegTensor<int32_t>&)" + z_lo + ", " +
                     pm + ");");
        codegen.Emit("pxor(" + sign_m + ", " + s0ge_m + ", " + s1ge_m + ", " + pm + ");");
        codegen.Emit("pnot(" + sign_m + ", " + sign_m + ", " + pm + ");");
        VsubUsingU32(t8, z, t6, all_m);
        VselUsingU32(t6, t6, t8, sign_m);
    }
    // Divisor == 0 -> all-ones quotient; assemble the interleaved dst.
    VselUsingU32(t6, qz, t6, zero_m);
    codegen.Emit("vintlv((RegTensor<uint32_t>&)" + dst + ", (RegTensor<uint32_t>&)" + d1 + ", (RegTensor<uint32_t>&)" +
                 t6_lo + ", (RegTensor<uint32_t>&)" + t6_hi + ");");
}

// Check if a DataType lacks a direct vdup/vlds/vsts intrinsic overload and
// must be reinterpreted as uint8_t. The bisheng __VF_VDUP/__VF_VLDS/__VF_VSTS
// macros only instantiate overloads for u8/s8/u16/s16/f16/u32/s32/f32/bf16/
// f8e4m3/f8e5m2/f8e8m0/f4e2m1x2/f4e1m2x2 and b64 types. FP4/HF4 (b4 packed as
// b8) and HF8 (b8) have no overload, but share the same physical b8 register
// layout as u8.
static bool NeedsB8Reinterpret(DataType dt)
{
    return dt == DataType::FP4 || dt == DataType::FP4E2M1 || dt == DataType::FP4E1M2 || dt == DataType::HF4 ||
           dt == DataType::HF8 || dt == DataType::INT4 || dt == DataType::UINT4;
}

// Map a DataType to the correct C pointer type for __ubuf__ load/store.
// VF load/store intrinsics (vlds/vsts/vld/vst/vsldb/vsstb) accept both
// signed and unsigned pointer types, so we use the native C type to match
// the RegTensor element type and avoid type-mismatch errors.
// B64 types (INT64/UINT64) are stored as pairs of 32-bit halves, so the
// pointer type is the 32-bit half type for non-B64-overload paths.
// (B64 vlds/vsts overloads override the pointer type via GetB64PtrType.)
// Types lacking a direct bisheng overload (FP4/HF4/HF8/INT4/UINT4) are
// reinterpreted as uint8_t.
static std::string DtypeToPtrType(DataType dt)
{
    if (NeedsB8Reinterpret(dt))
        return "uint8_t";
    if (dt == DataType::INT64)
        return "int32_t";
    if (dt == DataType::UINT64)
        return "uint32_t";
    return dt.ToCTypeString();
}

// Return the (RegTensor<uint8_t>&) cast prefix for types that need B8 reinterpret.
static std::string GetB8Cast(DataType dt) { return NeedsB8Reinterpret(dt) ? "(RegTensor<uint8_t>&)" : ""; }

// Resolve an offset argument to a C++ code string.
// If the argument is a 2-element MakeTuple [row, col], compute the linear
// offset `row * cols + col` using the tile's shape[1] (number of columns).
// Otherwise, emit the expression directly (integer offset, AddrReg, etc.).
// Returns empty string if the argument is not an offset (caller should skip).
static std::string ResolveOffsetArg(codegen::CCECodegen& codegen, const ir::ExprPtr& offset_expr,
                                    const ir::ExprPtr& tile_expr)
{
    if (auto tuple = ir::As<ir::MakeTuple>(offset_expr)) {
        if (tuple->elements_.size() == 2) {
            std::string row_str = codegen.GetExprAsCode(tuple->elements_[0]);
            std::string col_str = codegen.GetExprAsCode(tuple->elements_[1]);
            // Get cols from tile shape[1]
            std::string cols_str = "1";
            if (auto tile_type = ir::As<ir::TileType>(tile_expr->GetType())) {
                if (tile_type->shape_.size() >= 2) {
                    cols_str = codegen.GetExprAsCode(tile_type->shape_[1]);
                }
            }
            return "((" + row_str + ") * (" + cols_str + ") + (" + col_str + "))";
        }
    }
    return codegen.GetExprAsCode(offset_expr);
}

// ============================================================================
// CreateMask — declares MaskReg + emits VF init instruction
// ============================================================================

// For b64 element width (INT64/UINT64) on a single-register trait, the hardware
// interprets mask bits at 2-bit-per-element granularity. Since pset_b32 produces
// 1-bit-per-b32-element masks, we must:
//   1. Remap H/Q patterns to VL16/VL8 (punpack doubles the bit count, so the
//      source pattern must be halved to compensate).
//   2. Emit punpack(reg, reg, LOWER) after pset_b32 to expand each bit into a
//      pair, matching the 2-bit-per-b64-element mask width.
// This mirrors AscendC CreateMaskImpl<T, mode, RegTraitNumOne> for sizeof(T)==8.
static ir::MaskPattern RemapB64MaskPattern(ir::MaskPattern pattern)
{
    switch (pattern) {
        case ir::MaskPattern::H:
            return ir::MaskPattern::VL16;
        case ir::MaskPattern::Q:
            return ir::MaskPattern::VL8;
        default:
            return pattern;
    }
}

static std::string EmitVFCreateMask(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // pattern defaults to ALL, dtype defaults to FP32 — either kwarg may be omitted.
    auto pattern = op->HasKwarg("pattern") ? static_cast<ir::MaskPattern>(op->GetKwarg<int>("pattern")) :
                                             ir::MaskPattern::ALL;
    auto dtype = op->HasKwarg("dtype") ? op->GetKwarg<DataType>("dtype") : DataType::FP32;
    // MaskReg dtype must be b8/b16/b32/b64 — determines mask granularity
    // FP4 types (GetBit()==4) are b8 storage (packed 2-per-byte), treated as b8
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      IsB8Type(dtype) || dtype.GetBit() == 16 || dtype.GetBit() == 32 || dtype.GetBit() == 64)
        << "vf.create_mask dtype must be b8/b16/b32/b64, got " << DTypeStr(dtype);
    bool is_b64 = (dtype.GetBit() == 64);
    if (is_b64) {
        pattern = RemapB64MaskPattern(pattern);
    }
    std::string reg_name = codegen.GetCurrentResultTarget();
    codegen.Emit("MaskReg " + reg_name + ";");
    codegen.RegisterMaskRegVar(reg_name);
    // Map pypto pattern enum to CCE PAT_* constant
    std::string pat;
    switch (pattern) {
        case ir::MaskPattern::ALL:
            pat = "PAT_ALL";
            break;
        case ir::MaskPattern::ALLF:
            pat = "PAT_ALLF";
            break;
        case ir::MaskPattern::VL1:
            pat = "PAT_VL1";
            break;
        case ir::MaskPattern::VL2:
            pat = "PAT_VL2";
            break;
        case ir::MaskPattern::VL3:
            pat = "PAT_VL3";
            break;
        case ir::MaskPattern::VL4:
            pat = "PAT_VL4";
            break;
        case ir::MaskPattern::VL8:
            pat = "PAT_VL8";
            break;
        case ir::MaskPattern::VL16:
            pat = "PAT_VL16";
            break;
        case ir::MaskPattern::VL32:
            pat = "PAT_VL32";
            break;
        case ir::MaskPattern::VL64:
            pat = "PAT_VL64";
            break;
        case ir::MaskPattern::VL128:
            pat = "PAT_VL128";
            break;
        case ir::MaskPattern::M3:
            pat = "PAT_M3";
            break;
        case ir::MaskPattern::M4:
            pat = "PAT_M4";
            break;
        case ir::MaskPattern::H:
            pat = "PAT_H";
            break;
        case ir::MaskPattern::Q:
            pat = "PAT_Q";
            break;
        default:
            // The pattern kwarg is type-erased to int at the Python/C++ boundary;
            // an out-of-enum value must not silently fall back to PAT_ALL.
            // Unreachable for values produced by the front end (0..14 all have
            // a case above) — this only guards direct IR construction paths.
            PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, false)
                << "vf.create_mask pattern=" << static_cast<int>(pattern)
                << " is not a valid ir::MaskPattern enumerator";
            return "";
    }
    // Select pset instruction based on data element size (not mask type)
    // float/int32 (4 bytes) → pset_b32, half/bf16 (2 bytes) → pset_b16, int8 (1 byte) → pset_b8
    // FP8/FP4 types are b8 storage → pset_b8
    // b64 (INT64/UINT64): pset_b32 + punpack(LOWER) — punpack expands each bit
    // into a pair to match the 2-bit-per-element mask width for 8-byte elements.
    if (IsB8Type(dtype)) {
        codegen.Emit(reg_name + " = pset_b8(" + pat + ");");
    } else if (dtype.GetBit() == 32 || is_b64) {
        codegen.Emit(reg_name + " = pset_b32(" + pat + ");");
        if (is_b64) {
            codegen.Emit("punpack(" + reg_name + ", " + reg_name + ", LOWER);");
        }
    } else {
        // FP16, BF16, UINT16, INT16 etc. (2 bytes)
        codegen.Emit(reg_name + " = pset_b16(" + pat + ");");
    }
    return "";
}

// ============================================================================
// Duplicate — scalar broadcast
// ============================================================================

static std::string EmitVFDuplicate(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // args: [dst, src, (optional) mask]
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() >= 2 && op->args_.size() <= 3)
        << "vf.full requires 2-3 args (dst, src[, mask])";
    // vdup supports b8/b16/b32/b64 element widths (bool, int, float, FP8/FP4/HF8 types)
    DataType src_dt = GetExprDtype(op->args_[1], DataType::FP32);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (IsB8Type(src_dt) || src_dt.GetBit() == 16 || src_dt.GetBit() == 32 || src_dt.GetBit() == 64))
        << "vf.full src only supports b8/b16/b32/b64 types, got " << DTypeStr(src_dt);
    // MERGING is not supported by the underlying vdup/vbr instructions on current device.
    VFZeroingOnly(op, "vf.full");
    // The fill width follows the DST tile. The Scalar-mode ``dtype`` kwarg is
    // what the frontend uses to declare dst (a bare int literal is INT64, so it
    // must not be compared against dst nor route a b16/b32 fill into the b64
    // split path).
    DataType dst_dt = GetExprDtype(op->args_[0]);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src_str = codegen.GetExprAsCode(op->args_[1]);
    // Detect vector-source broadcast: either explicit pos kwarg, or src is a RegTensor variable.
    std::string pos = "";
    if (op->HasKwarg("pos")) {
        pos = VFEnumValueName(ir::EnumToString(static_cast<ir::DuplicatePos>(op->GetKwarg<int>("pos"))));
    }
    bool is_vector_src = !pos.empty();
    if (!is_vector_src) {
        auto src_var = ir::As<ir::Var>(op->args_[1]);
        if (src_var) {
            std::string src_name = codegen.GetVarName(src_var);
            is_vector_src = codegen.IsRegTensorVar(src_name);
        }
    }
    if (is_vector_src) {
        // Vector-source broadcast (Tensor mode): vdup(dst, src_vec, mask, POS_xxx, MODE)
        // pos kwarg: "LOWEST" -> POS_LOWEST, "HIGHEST" -> POS_HIGHEST
        if (pos.empty() || pos == "LOWEST")
            pos = "POS_LOWEST";
        else if (pos == "HIGHEST")
            pos = "POS_HIGHEST";
        std::string mode = VFZeroingOnly(op, "vf.full");
        // FP4/HF8/HF4/INT4/UINT4 lack a vdup overload — reinterpret as uint8_t
        std::string cast = GetB8Cast(src_dt);
        // Tensor mode always requires a mask (vdup(dstReg, srcReg, mask))
        if (op->args_.size() >= 3) {
            std::string mask = codegen.GetExprAsCode(op->args_[2]);
            codegen.Emit("vdup(" + cast + dst + ", " + cast + src_str + ", " + mask + ", " + pos + ", " + mode + ");");
        } else {
            // No mask provided — create an ALL mask inline for Tensor mode
            static int dup_mask_counter = 0;
            std::string mask_var = "__dup_mask_" + std::to_string(dup_mask_counter++);
            std::string pat = "PAT_ALL";
            std::string pset_fn = "pset_b32";
            if (IsB8Type(src_dt))
                pset_fn = "pset_b8";
            else if (src_dt.GetBit() == 16)
                pset_fn = "pset_b16";
            codegen.Emit("MaskReg " + mask_var + " = " + pset_fn + "(" + pat + ");");
            codegen.RegisterMaskRegVar(mask_var);
            codegen.Emit("vdup(" + cast + dst + ", " + cast + src_str + ", " + mask_var + ", " + pos + ", " + mode +
                         ");");
        }
    } else if (op->args_.size() >= 3) {
        // Scalar broadcast with mask: vdup(dst, scalar, preg, MODE_ZEROING/MERGING)
        std::string mask = codegen.GetExprAsCode(op->args_[2]);
        std::string mode = VFZeroingOnly(op, "vf.full");
        // The fill width follows DST: an int literal is INT64 and must not route
        // a b16/b32 fill into the b64 split path.
        if (dst_dt.GetBit() == 64) {
            // B64 has no vdup single-register overload; create two b32 halves,
            // vdup each with packed b32 mask, then vintlv into b64 dst
            // (mirrors AscendC DuplicateB64Impl + MaskPack).
            std::string p = dst + "_dup_";
            std::string packed_m = p + "_pm_";
            codegen.Emit("MaskReg " + packed_m + ";");
            codegen.Emit("ppack(" + packed_m + ", " + mask + ", LOWER);");
            std::string lo_d = p + "_lo_";
            std::string hi_d = p + "_hi_";
            std::string cast_type = (dst_dt == DataType::INT64) ? "(int64_t)" : "(uint64_t)";
            codegen.Emit("RegTensor<uint32_t> " + lo_d + ";");
            codegen.Emit("RegTensor<uint32_t> " + hi_d + ";");
            codegen.Emit("vdup(" + lo_d + ", (uint32_t)(" + cast_type + "(" + src_str + ")), " + packed_m + ", " +
                         mode + ");");
            codegen.Emit("vdup(" + hi_d + ", (uint32_t)((" + cast_type + "(" + src_str + ")) >> 32), " + packed_m +
                         ", " + mode + ");");
            EmitB64Interleave(codegen, dst, lo_d, hi_d, p + "_ilv");
        } else {
            codegen.Emit("vdup(" + dst + ", " + src_str + ", " + mask + ", " + mode + ");");
        }
    } else {
        // Scalar broadcast without mask: vbr(dst, scalar)
        if (dst_dt.GetBit() == 64) {
            // B64 has no vbr single-register overload; create two b32 halves,
            // vbr each, then vintlv into b64 dst (mirrors AscendC DuplicateB64Impl).
            std::string p = dst + "_br_";
            std::string lo_d = p + "_lo_";
            std::string hi_d = p + "_hi_";
            std::string cast_type = (dst_dt == DataType::INT64) ? "(int64_t)" : "(uint64_t)";
            codegen.Emit("RegTensor<uint32_t> " + lo_d + ";");
            codegen.Emit("RegTensor<uint32_t> " + hi_d + ";");
            codegen.Emit("vbr(" + lo_d + ", (uint32_t)(" + cast_type + "(" + src_str + ")));");
            codegen.Emit("vbr(" + hi_d + ", (uint32_t)((" + cast_type + "(" + src_str + ")) >> 32));");
            EmitB64Interleave(codegen, dst, lo_d, hi_d, p + "_ilv");
        } else {
            codegen.Emit("vbr(" + dst + ", " + src_str + ");");
        }
    }
    return "";
}

// ============================================================================
// Helper: get __ubuf__ pointer from tile or tile-flavored GetItemExpr
// ============================================================================

static std::string GetUBufPtr(codegen::CCECodegen& codegen, const ir::ExprPtr& expr,
                              const std::string& cast_type = "float", bool is_post_update = false)
{
    std::string ptr = codegen.GetOrCreateVFTilePtr(expr, is_post_update);
    std::string tile_ctype = GetExprDtype(expr, DataType::FP32).ToCTypeString();
    if (cast_type == tile_ctype)
        return ptr;
    return "(__ubuf__ " + cast_type + " *)" + ptr;
}

// ============================================================================
// LoadAlign (unified) — vlds / plds with dist & post_update kwargs
// Replaces: LoadAlign, LoadAlignMode, LoadAlignPostUpdate, LoadAlignPostupdate, LoadAlignUnpackV2
// ============================================================================

static std::string EmitVFLoadAlign(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // 4-arg form: deinterleave load_align(dst0, dst1, ptr, offset, dist="DINTLV_Bxx")
    if (op->args_.size() == 4) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("data_copy_mode"))
            << "vf.load_align 4-arg (de-interleave) form does not support data_copy_mode";
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("block_stride"))
            << "vf.load_align 4-arg (de-interleave) form does not support block_stride";
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("repeat_stride"))
            << "vf.load_align 4-arg (de-interleave) form does not support repeat_stride";
        // 4-arg de-interleave form requires both dsts to be RegTensor (not MaskReg)
        for (int i = 0; i < 2; i++) {
            auto dst_v = ir::As<ir::Var>(op->args_[i]);
            if (dst_v != nullptr) {
                PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, !codegen.IsMaskRegVar(codegen.GetVarName(dst_v)))
                    << "vf.load_align 4-arg (de-interleave) form requires RegTensor dst, " << "but args[" << i
                    << "] is a MaskReg";
            }
        }
        std::string dst0 = codegen.GetExprAsCode(op->args_[0]);
        std::string dst1 = codegen.GetExprAsCode(op->args_[1]);
        std::string offset_str = ResolveOffsetArg(codegen, op->args_[3], op->args_[2]);
        DataType dst_dt = GetExprDtype(op->args_[0]);
        // The dual-load instruction loads both dsts at dst0's element width:
        // dst1 must have the same bit width (equal-width reinterpreting views
        // stay legal, mirroring the de_interleave validation).
        DataType dst1_dt = GetExprDtype(op->args_[1]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, dst_dt.GetBit() == dst1_dt.GetBit())
            << "vf.load_align 4-arg (de-interleave) requires dst0 and dst1 to have the same bit width, got dst0="
            << DTypeStr(dst_dt) << " dst1=" << DTypeStr(dst1_dt);
        // Pointer type follows the register's declared dtype. When it differs
        // from the tile's dtype, adjust the offset by the element-width ratio so
        // the byte address stays correct (mirrors b64→b32 stride*2).
        std::string ptr_type = DtypeToPtrType(dst_dt);
        std::string ub_ptr = GetUBufPtr(codegen, op->args_[2], ptr_type);
        DataType tile_dt = GetExprDtype(op->args_[2], dst_dt);
        std::string cast = GetB8Cast(dst_dt);
        std::string effective_offset = offset_str;
        if (tile_dt.GetBit() != dst_dt.GetBit() && !offset_str.empty()) {
            effective_offset = "(" + offset_str + ") * " + std::to_string(tile_dt.GetBit()) + " / " +
                               std::to_string(dst_dt.GetBit());
        }
        std::string dintlv_mode;
        if (op->HasKwarg("dist")) {
            dintlv_mode = VFCheckedDistName(op->GetKwarg<int>("dist"), /*is_load=*/true, "vf.load_align");
            PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT,
                              dintlv_mode == "DINTLV_B8" || dintlv_mode == "DINTLV_B16" || dintlv_mode == "DINTLV_B32")
                << "vf.load_align 4-arg (de-interleave) form requires dist=DINTLV_B8/B16/B32, got " << dintlv_mode;
        } else {
            if (IsB8Type(dst_dt))
                dintlv_mode = "DINTLV_B8";
            else if (IsB16Type(dst_dt))
                dintlv_mode = "DINTLV_B16";
            else
                dintlv_mode = "DINTLV_B32";
        }
        bool post_update = false;
        if (op->HasKwarg("post_update")) {
            post_update = op->GetKwarg<bool>("post_update");
        }
        if (post_update) {
            codegen.Emit("vlds(" + cast + dst0 + ", " + cast + dst1 + ", " + ub_ptr + ", " + effective_offset + ", " +
                         dintlv_mode + ", POST_UPDATE);");
        } else {
            codegen.Emit("vlds(" + cast + dst0 + ", " + cast + dst1 + ", " + ub_ptr + ", " + effective_offset + ", " +
                         dintlv_mode + ");");
        }
        return "";
    }
    // 2-arg form: load_align(dst, ptr) — MaskReg dst → plds, RegTensor dst → vlds
    if (op->args_.size() == 2) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("data_copy_mode"))
            << "vf.load_align 2-arg form does not support data_copy_mode";
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("block_stride"))
            << "vf.load_align 2-arg form does not support block_stride";
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("repeat_stride"))
            << "vf.load_align 2-arg form does not support repeat_stride";
        std::string dst = codegen.GetExprAsCode(op->args_[0]);
        DataType dst_dt = GetExprDtype(op->args_[0]);
        bool dst_is_mask = false;
        if (auto dst_v = ir::As<ir::Var>(op->args_[0])) {
            dst_is_mask = codegen.IsMaskRegVar(codegen.GetVarName(dst_v));
        }
        if (dst_is_mask) {
            std::string mode = "NORM";
            if (op->HasKwarg("dist")) {
                mode = VFCheckedDistName(op->GetKwarg<int>("dist"), /*is_load=*/true, "vf.load_align");
                PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, mode == "NORM" || mode == "US" || mode == "DS")
                    << "vf.load_align 2-arg (MaskReg) only supports NORM/US/DS dist, got " << mode;
            }
            std::string plds_ptr = GetUBufPtr(codegen, op->args_[1], "uint32_t");
            codegen.Emit("plds(" + dst + ", " + plds_ptr + ", 0, " + mode + ");");
        } else {
            // RegTensor 2-arg form: vlds with hardcoded NORM, dist kwarg not supported
            if (op->HasKwarg("dist")) {
                auto dist_val = VFCheckedDistName(op->GetKwarg<int>("dist"), /*is_load=*/true, "vf.load_align");
                PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, dist_val == "NORM")
                    << "vf.load_align 2-arg (RegTensor) only supports NORM dist, got " << dist_val;
            }
            std::string ptr_type = DtypeToPtrType(dst_dt);
            std::string ub_ptr = GetUBufPtr(codegen, op->args_[1], ptr_type);
            std::string cast = GetB8Cast(dst_dt);
            // B64 types (INT64/UINT64): load as b32 (mirrors AscendC DataCopyImpl
            // b64 RegTraitNumOne: vlds((RegTensor<uint32_t>&)reg, uint32_t* ptr, 0, NORM))
            if (dst_dt.GetBit() == 64) {
                std::string b32_ptr = GetUBufPtr(codegen, op->args_[1], "uint32_t");
                codegen.Emit("vlds((RegTensor<uint32_t>&)" + dst + ", " + b32_ptr + ", 0, NORM);");
            } else {
                codegen.Emit("vlds(" + cast + dst + ", " + ub_ptr + ", 0, NORM);");
            }
        }
        return "";
    }
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "vf.load_align requires 2, 3, or 4 args (dst, src_ptr[, offset]) or (dst0, dst1, src_ptr, offset)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string offset_str = ResolveOffsetArg(codegen, op->args_[2], op->args_[1]);
    DataType dst_dt = GetExprDtype(op->args_[0]);
    // Pointer type follows the register's declared dtype. When it differs from
    // the tile's dtype, adjust the offset by the element-width ratio so the byte
    // address stays correct (mirrors b64→b32 stride*2).
    std::string ptr_type = DtypeToPtrType(dst_dt);
    DataType tile_dt = GetExprDtype(op->args_[1], dst_dt);
    // Is the destination a MaskReg? (routes to pld/plds instead of vld/vlds)
    bool dst_is_mask = false;
    if (auto dst_v = ir::As<ir::Var>(op->args_[0])) {
        dst_is_mask = codegen.IsMaskRegVar(codegen.GetVarName(dst_v));
    }
    // Offset adjustment only for RegTensor path (MaskReg always uses uint32_t ptr)
    std::string effective_offset = offset_str;
    if (!dst_is_mask && tile_dt.GetBit() != dst_dt.GetBit()) {
        effective_offset = "(" + offset_str + ") * " + std::to_string(tile_dt.GetBit()) + " / " +
                           std::to_string(dst_dt.GetBit());
    }
    // Determine mode from kwargs (dist kwarg is legacy alias for mode).
    // MaskReg (plds) and RegTensor (vlds) paths share the same LoadDist enum:
    // LoadDist includes NORM/US/DS/BRC/... and EnumToString yields the bare
    // name (e.g. "DS"), which plds accepts directly and vlds maps to Bxx suffix.
    std::string mode = "NORM";
    if (op->HasKwarg("dist"))
        mode = VFCheckedDistName(op->GetKwarg<int>("dist"), /*is_load=*/true, "vf.load_align");
    // 3-arg form (single dst) cannot use DINTLV modes (de-interleave requires 2 dsts)
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT,
                      mode != "DINTLV_B8" && mode != "DINTLV_B16" && mode != "DINTLV_B32")
        << "vf.load_align with 3 args (single dst) does not support DINTLV dist, "
        << "use 4-arg form: dst0, dst1 = vf.load_align(src, offset, dist=...)";
    // AddrReg offset path: MaskReg dst -> pld, RegTensor dst -> vld
    if (codegen.IsAddrRegVar(offset_str)) {
        std::string ub_ptr = GetUBufPtr(codegen, op->args_[1], dst_is_mask ? "uint32_t" : ptr_type);
        if (dst_is_mask) {
            codegen.Emit("pld(" + dst + ", " + ub_ptr + ", " + offset_str + ", " + mode + ");");
        } else {
            codegen.Emit("vld(" + dst + ", " + ub_ptr + ", " + effective_offset + ", " + mode + ");");
        }
        return "";
    }
    // Check for DataBlock load path (vsldb)
    std::string data_copy_mode = "NORM";
    if (op->HasKwarg("data_copy_mode")) {
        data_copy_mode = VFEnumValueName(
            ir::EnumToString(static_cast<ir::DataCopyMode>(op->GetKwarg<int>("data_copy_mode"))));
    }
    bool post_update = false;
    if (op->HasKwarg("post_update")) {
        post_update = op->GetKwarg<bool>("post_update");
    }
    // vsldb path (non-contiguous datablock load). Accept both DATA_BLOCK_LOAD
    // (pypto legacy name) and DATA_BLOCK_COPY so
    // code written against either naming does not silently fall back to vlds.
    if (data_copy_mode == "DATA_BLOCK_LOAD" || data_copy_mode == "DATA_BLOCK_COPY") {
        // In DataBlock mode, args[2] is a mask register, not an offset.
        // Verify the user didn't pass an integer/AddrReg offset by mistake.
        auto mask_var = ir::As<ir::Var>(op->args_[2]);
        PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, mask_var != nullptr)
            << "vf.load_align with data_copy_mode=DATA_BLOCK_COPY requires args[2] to be a "
            << "mask register, but got an offset value (offset is not supported in DataBlock mode)";
        // Verify args[2] is actually a MaskReg, not a Tile or other Var type
        std::string mask_name = codegen.GetVarName(mask_var);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, codegen.IsMaskRegVar(mask_name))
            << "vf.load_align with data_copy_mode=DATA_BLOCK_COPY requires args[2] to be a "
            << "mask register, but got a non-mask variable '" << mask_name << "'";
        // DataBlock mode requires RegTensor dst (not MaskReg)
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, !dst_is_mask)
            << "vf.load_align with data_copy_mode=DATA_BLOCK_COPY does not support MaskReg dst";
        // vsldb only supports b8/b16/b32 (not b64)
        // vsldb path: load_align(dst, ptr, mask, data_copy_mode=..., block_stride=N, ...)
        std::string mask_reg = codegen.GetExprAsCode(op->args_[2]);
        std::string block_stride = "0";
        std::string repeat_stride = "0";
        if (op->HasKwarg("block_stride")) {
            block_stride = std::to_string(op->GetKwarg<int>("block_stride"));
        }
        if (op->HasKwarg("repeat_stride")) {
            repeat_stride = std::to_string(op->GetKwarg<int>("repeat_stride"));
        }
        if (post_update) {
            std::string ub_ptr = codegen.GetOrCreateVFTilePtr(op->args_[1], /*is_post_update=*/true);
            codegen.Emit("vsldb(" + dst + ", " + ub_ptr + ", (" + block_stride + " << 16u) | (" + repeat_stride +
                         " & 0xFFFFU), " + mask_reg + ", POST_UPDATE);");
        } else {
            std::string ub_ptr = GetUBufPtr(codegen, op->args_[1], ptr_type);
            codegen.Emit("vsldb(" + dst + ", " + ub_ptr + ", (" + block_stride + " << 16u), " + mask_reg + ");");
        }
        return "";
    }
    // Route by dst variable type: MaskReg → plds, RegTensor → vlds
    if (dst_is_mask) {
        // plds path: dst is MaskReg, pointer is always uint32_t*
        // pld/plds: only supports NORM/US/DS dist
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, mode == "NORM" || mode == "US" || mode == "DS")
            << "vf.load_align (MaskReg) only supports NORM/US/DS dist, got " << mode;
        std::string plds_mode = mode; // pass through NORM/US/DS directly
        std::string plds_ptr;
        if (post_update) {
            plds_ptr = codegen.GetOrCreateVFTilePtr(op->args_[1], /*is_post_update=*/true);
            codegen.Emit("plds(" + dst + ", " + plds_ptr + ", " + offset_str + ", " + plds_mode + ", POST_UPDATE);");
        } else {
            plds_ptr = GetUBufPtr(codegen, op->args_[1], "uint32_t");
            codegen.Emit("plds(" + dst + ", " + plds_ptr + ", " + offset_str + ", " + plds_mode + ");");
        }
        return "";
    }
    // vlds path: dst is RegTensor
    // Get UB pointer (post_update uses reference-based ptr)
    std::string ub_ptr;
    if (post_update) {
        ub_ptr = codegen.GetOrCreateVFTilePtr(op->args_[1], /*is_post_update=*/true);
    } else {
        ub_ptr = GetUBufPtr(codegen, op->args_[1], ptr_type);
    }
    // Determine vlds mode string
    std::string vlds_mode;
    if (mode == "NORM") {
        vlds_mode = "NORM";
    } else if (mode == "BRC") {
        if (IsB8Type(dst_dt))
            vlds_mode = "BRC_B8";
        else if (IsB16Type(dst_dt))
            vlds_mode = "BRC_B16";
        else
            vlds_mode = "BRC_B32";
    } else if (mode == "US") {
        if (IsB8Type(dst_dt))
            vlds_mode = "US_B8";
        else
            vlds_mode = "US_B16";
    } else if (mode == "DS") {
        if (IsB8Type(dst_dt))
            vlds_mode = "DS_B8";
        else
            vlds_mode = "DS_B16";
    } else if (mode == "UNPK") {
        if (IsB8Type(dst_dt))
            vlds_mode = "UNPK_B8";
        else if (IsB16Type(dst_dt))
            vlds_mode = "UNPK_B16";
        else
            vlds_mode = "UNPK_B32";
    } else if (mode == "UNPK4") {
        vlds_mode = "UNPK4_B8";
    } else if (mode == "BLK") {
        vlds_mode = "BLK";
    } else if (mode == "E2B") {
        if (IsB16Type(dst_dt))
            vlds_mode = "E2B_B16";
        else
            vlds_mode = "E2B_B32";
    } else {
        // Fallback: pass through directly (e.g. BRC_B32, DS_B16, E2B_B32, etc.)
        vlds_mode = mode;
    }
    std::string cast = GetB8Cast(dst_dt);
    // B64 types (INT64/UINT64): load as b32 (mirrors AscendC DataCopyImpl b64
    // path, which passes the resolved dist through unchanged)
    if (dst_dt.GetBit() == 64) {
        std::string b32_ptr;
        if (post_update) {
            b32_ptr = codegen.GetOrCreateVFTilePtr(op->args_[1], /*is_post_update=*/true);
            std::string post_offset = "(" + effective_offset + ") * 2";
            codegen.Emit("vlds((RegTensor<uint32_t>&)" + dst + ", (__ubuf__ uint32_t*&)" + b32_ptr + ", " +
                         post_offset + ", " + vlds_mode + ", POST_UPDATE);");
        } else {
            b32_ptr = GetUBufPtr(codegen, op->args_[1], "uint32_t");
            codegen.Emit("vlds((RegTensor<uint32_t>&)" + dst + ", " + b32_ptr + ", " + effective_offset + ", " +
                         vlds_mode + ");");
        }
    } else if (post_update) {
        // B64 register types need stride doubled (postUpdateStride * 2 for 8-byte elements)
        std::string post_offset = effective_offset;
        if (dst_dt.GetBit() == 64) {
            post_offset = "(" + effective_offset + ") * 2";
        }
        codegen.Emit("vlds(" + cast + dst + ", " + ub_ptr + ", " + post_offset + ", " + vlds_mode + ", POST_UPDATE);");
    } else {
        codegen.Emit("vlds(" + cast + dst + ", " + ub_ptr + ", " + effective_offset + ", " + vlds_mode + ");");
    }
    return "";
}

// ============================================================================
// StoreAlign — vsts
// ============================================================================

static std::string EmitVFStoreAlign(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // args: [dst_ptr, src_reg, mask, (optional) block_stride, (optional) repeat_stride]
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() >= 2)
        << "vf.store_align requires at least 2 args (dst_ptr, src_reg)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    // vsts supports b8/b16/b32/b64 element widths
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      IsB8Type(src_dt) || src_dt.GetBit() == 16 || src_dt.GetBit() == 32 || src_dt.GetBit() == 64)
        << "vf.store_align only supports b8/b16/b32/b64 types, got " << DTypeStr(src_dt);
    std::string src_reg = codegen.GetExprAsCode(op->args_[1]);
    std::string cast = GetB8Cast(src_dt);
    // MaskReg src path: when args[1] is a MaskReg, dispatch to psts/pst (mask store)
    bool src_is_mask = false;
    if (auto src_v = ir::As<ir::Var>(op->args_[1])) {
        src_is_mask = codegen.IsMaskRegVar(codegen.GetVarName(src_v));
    }
    // vsts stores raw register bits; GetUBufPtr already casts the dst pointer
    // to the src register type (equivalent to AscendC's (RegTensor<T>&) cast),
    // so no runtime dtype compatibility check is needed here.
    if (src_is_mask) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("data_copy_mode"))
            << "vf.store_align (MaskReg src) does not support data_copy_mode";
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("block_stride"))
            << "vf.store_align (MaskReg src) does not support block_stride";
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("repeat_stride"))
            << "vf.store_align (MaskReg src) does not support repeat_stride";
        std::string dist = "NORM";
        if (op->HasKwarg("dist")) {
            dist = VFCheckedDistName(op->GetKwarg<int>("dist"), /*is_load=*/false, "vf.store_align");
            // pst/psts: only supports NORM and PACK dist
            PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, dist == "NORM" || dist == "PACK" || dist == "PK")
                << "vf.store_align (MaskReg) only supports NORM/PACK dist, got " << dist;
            // psts/pst use PK (not PACK) for packed mode
            if (dist == "PACK")
                dist = "PK";
        }
        // AddrReg offset path: 3rd arg is AddrReg -> pst(mask, ptr, areg, dist)
        if (op->args_.size() >= 3) {
            std::string third_arg = codegen.GetExprAsCode(op->args_[2]);
            if (codegen.IsAddrRegVar(third_arg)) {
                std::string ub_ptr = GetUBufPtr(codegen, op->args_[0], "uint32_t");
                codegen.Emit("pst(" + src_reg + ", " + ub_ptr + ", " + third_arg + ", " + dist + ");");
                return "";
            }
            // Post-update path: int offset with post_update kwarg
            bool post_update = op->HasKwarg("post_update") && op->GetKwarg<bool>("post_update");
            if (post_update) {
                std::string ptr_var = codegen.GetOrCreateVFTilePtr(op->args_[0], /*is_post_update=*/true);
                codegen.Emit("psts(" + src_reg + ", " + ptr_var + ", " + third_arg + ", " + dist + ", POST_UPDATE);");
                return "";
            }
        }
        // Default: psts with offset=0
        std::string ub_ptr = GetUBufPtr(codegen, op->args_[0], "uint32_t");
        codegen.Emit("psts(" + src_reg + ", " + ub_ptr + ", 0, " + dist + ");");
        return "";
    }
    // Resolve the dist kwarg for the single-source paths (vst/vsts). The MaskReg
    // path above handles its own NORM/PACK dist. Coarse names (NORM,
    // FIRST_ELEMENT, PACK, PACK4, INTLV) auto-expand to the element-width
    // variant based on the src dtype; width-qualified names map to the CCE
    // DistVST constants (mirrors AscendC Reg::StoreDist / GetStoreDist).
    std::string dist = "";
    if (op->HasKwarg("dist")) {
        dist = VFCheckedDistName(op->GetKwarg<int>("dist"), /*is_load=*/false, "vf.store_align");
    }
    if (dist.empty()) {
        // Auto-select default dist based on src dtype (uses NORM_B8/B16/B32 by element width)
        DataType src_dtype = DataType::FP32;
        if (auto scalar_type = ir::As<ir::ScalarType>(op->args_[1]->GetType())) {
            src_dtype = scalar_type->dtype_;
        } else if (auto tile_type_tmp = ir::As<ir::TileType>(op->args_[0]->GetType())) {
            src_dtype = tile_type_tmp->dtype_;
        }
        if (IsB8Type(src_dtype))
            dist = "NORM_B8";
        else if (IsB16Type(src_dtype))
            dist = "NORM_B16";
        else
            dist = "NORM_B32";
    }
    // Auto-expand shorthand dist names to element-width-qualified CCE constants
    if (dist == "NORM") {
        DataType sd = GetExprDtype(op->args_[1]);
        if (IsB8Type(sd))
            dist = "NORM_B8";
        else if (IsB16Type(sd))
            dist = "NORM_B16";
        else
            dist = "NORM_B32";
    } else if (dist == "FIRST_ELEMENT" || dist == "FIRST_ELE") {
        DataType sd = GetExprDtype(op->args_[1]);
        if (IsB8Type(sd))
            dist = "ONEPT_B8";
        else if (IsB16Type(sd))
            dist = "ONEPT_B16";
        else
            dist = "ONEPT_B32";
    } else if (dist == "PACK") {
        DataType sd = GetExprDtype(op->args_[1]);
        if (IsB8Type(sd) || IsB16Type(sd))
            dist = "PK_B16";
        else if (sd.GetBit() == 32)
            dist = "PK_B32";
        else
            dist = "PK_B64";
    } else if (dist == "PACK4") {
        dist = "PK4_B32";
    } else if (dist == "INTLV") {
        DataType sd = GetExprDtype(op->args_[1]);
        if (IsB8Type(sd))
            dist = "INTLV_B8";
        else if (IsB16Type(sd))
            dist = "INTLV_B16";
        else
            dist = "INTLV_B32";
    } else if (dist == "FIRST_ELEMENT_B8" || dist == "FIRST_ELEMENT_B16" || dist == "FIRST_ELEMENT_B32") {
        // Explicit-width variants (mirrors AscendC Reg::StoreDist): map to the CCE
        // DistVST constants; NORM_Bx/INTLV_Bx already match and pass through as-is.
        dist = "ONEPT" + dist.substr(std::strlen("FIRST_ELEMENT"));
    } else if (dist == "PACK_B16" || dist == "PACK_B32" || dist == "PACK_B64") {
        dist = "PK" + dist.substr(std::strlen("PACK"));
    } else if (dist == "PACK4_B32") {
        dist = "PK4_B32";
    }
    // AddrReg offset path: when 4th arg is an AddrReg variable,
    // emit vst(src, ptr, areg, dist, mask) — 5 args (note: vst, not vsts).
    // The resolved dist is honored here as well (previously the dist kwarg was
    // silently ignored in this path and NORM_Bx was always used).
    if (op->args_.size() >= 4) {
        std::string addr_reg = codegen.GetExprAsCode(op->args_[3]);
        if (codegen.IsAddrRegVar(addr_reg)) {
            // Verify args[2] is a MaskReg
            auto mask_var_4 = ir::As<ir::Var>(op->args_[2]);
            PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, mask_var_4 != nullptr)
                << "vf.store_align (AddrReg) requires args[2] to be a mask register, but got non-Var type";
            PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, codegen.IsMaskRegVar(codegen.GetVarName(mask_var_4)))
                << "vf.store_align (AddrReg) requires args[2] to be a mask register";
            // vst has no post mode — the AddrReg offset must be recreated each
            // iteration via vf.create_addr_reg
            PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT,
                              !(op->HasKwarg("post_update") && op->GetKwarg<bool>("post_update")))
                << "vf.store_align (AddrReg) does not support post_update; recreate the AddrReg offset each "
                   "iteration via vf.create_addr_reg";
            // vsstb (DataBlock copy) has no AddrReg form
            PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("data_copy_mode"))
                << "vf.store_align (AddrReg) does not support data_copy_mode";
            // vst (AddrReg) is a single-source intrinsic: INTLV dists need two
            // source registers (vsts dual form)
            PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT,
                              dist != "INTLV_B8" && dist != "INTLV_B16" && dist != "INTLV_B32")
                << "vf.store_align (AddrReg) does not support INTLV dist (requires two source registers), got " << dist;
            std::string mask_reg = codegen.GetExprAsCode(op->args_[2]);
            std::string ptr_type = "float";
            if (auto scalar_type = ir::As<ir::ScalarType>(op->args_[1]->GetType())) {
                ptr_type = DtypeToPtrType(scalar_type->dtype_);
            }
            std::string dst_ptr = GetUBufPtr(codegen, op->args_[0], ptr_type);
            std::string vst_src = cast + src_reg;
            std::string vst_mask = mask_reg;
            if (src_dt.GetBit() == 64) {
                // b64: simulate with b32 vst — reinterpret the register/pointer as
                // int32 and expand the b64-element mask to per-b32-lane bits
                // (ppack + pintlv_b32, mirrors the AscendC DataCopyImpl vst-areg
                // b64 path); dist passes through unchanged
                std::string packed_mask = mask_reg + "_b32lo_";
                std::string dump_mask = mask_reg + "_b32hi_";
                codegen.Emit("MaskReg " + packed_mask + ";");
                codegen.Emit("MaskReg " + dump_mask + ";");
                codegen.Emit("ppack(" + packed_mask + ", " + mask_reg + ", LOWER);");
                codegen.Emit("pintlv_b32(" + packed_mask + ", " + dump_mask + ", " + packed_mask + ", " + packed_mask +
                             ");");
                vst_src = "(RegTensor<int32_t>&)" + src_reg;
                vst_mask = packed_mask;
                dst_ptr = GetUBufPtr(codegen, op->args_[0], "int32_t");
            }
            codegen.Emit("vst(" + vst_src + ", " + dst_ptr + ", " + addr_reg + ", " + dist + ", " + vst_mask + ");");
            return "";
        }
    }
    bool post_update = false;
    if (op->HasKwarg("post_update")) {
        post_update = op->GetKwarg<bool>("post_update");
    }
    std::string data_copy_mode = "NORM";
    if (op->HasKwarg("data_copy_mode")) {
        data_copy_mode = VFEnumValueName(
            ir::EnumToString(static_cast<ir::DataCopyMode>(op->GetKwarg<int>("data_copy_mode"))));
    }
    // Pointer type follows the register's declared dtype. When it differs from
    // the tile's dtype, the offset/stride must be adjusted by the element-width
    // ratio so the byte address stays correct (mirrors b64→b32
    // stride*2). No register cast needed — the register is already src_dt.
    std::string ptr_type = DtypeToPtrType(src_dt);
    DataType tile_dt = DataType::FP32;
    auto tile_type = ir::As<ir::TileType>(op->args_[0]->GetType());
    if (tile_type) {
        tile_dt = tile_type->dtype_;
    } else if (auto scalar_type = ir::As<ir::ScalarType>(op->args_[1]->GetType())) {
        tile_dt = scalar_type->dtype_;
    } else {
        tile_dt = src_dt;
    }
    std::string width_op;
    if (tile_dt.GetBit() != src_dt.GetBit()) {
        width_op = " * " + std::to_string(tile_dt.GetBit()) + " / " + std::to_string(src_dt.GetBit());
    }
    // INTLV modes need two src registers: args = [dst_ptr, src_reg, src1, mask]
    bool is_intlv = (dist == "INTLV_B8" || dist == "INTLV_B16" || dist == "INTLV_B32");
    // INTLV requires 4 args; 4-arg with args[2] as register (not mask) requires INTLV
    // Skip this check for post_update path (4 args = dst, src, mask, stride is valid)
    if (is_intlv) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
            << "vf.store_align INTLV requires 4 args (dst_ptr, src_reg, src1, mask)";
    } else if (op->args_.size() == 4 && !src_is_mask && !post_update) {
        auto third_arg_var = ir::As<ir::Var>(op->args_[2]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT,
                          third_arg_var == nullptr || codegen.IsMaskRegVar(codegen.GetVarName(third_arg_var)))
            << "vf.store_align with 4 args where args[2] is a register (not mask) requires INTLV dist, " << "got "
            << dist;
    }
    if (is_intlv) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
            << "vf.store_align INTLV requires 4 args (dst_ptr, src_reg, src1, mask)";
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("data_copy_mode"))
            << "vf.store_align INTLV mode is incompatible with data_copy_mode";
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("post_update"))
            << "vf.store_align INTLV mode does not support post_update";
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("block_stride"))
            << "vf.store_align INTLV mode does not support block_stride";
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("repeat_stride"))
            << "vf.store_align INTLV mode does not support repeat_stride";
        // INTLV mode: args[2] is src1 (second source register), args[3] is mask
        auto intlv_mask_var = ir::As<ir::Var>(op->args_[3]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, intlv_mask_var != nullptr)
            << "vf.store_align INTLV requires args[3] to be a mask register, but got non-Var type";
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, codegen.IsMaskRegVar(codegen.GetVarName(intlv_mask_var)))
            << "vf.store_align INTLV requires args[3] to be a mask register";
        std::string src1 = codegen.GetExprAsCode(op->args_[2]);
        std::string mask_reg = codegen.GetExprAsCode(op->args_[3]);
        // vsts 2-source overload (__VF_VSTSX2) does not exist for FP32;
        // cast to UINT32 (same 32-bit width) to use the UINT32 overload.
        std::string intlv_ptr_type = ptr_type;
        if (src_dt == DataType::FP32) {
            src_reg = "(RegTensor<uint32_t> &)" + src_reg;
            src1 = "(RegTensor<uint32_t> &)" + src1;
            intlv_ptr_type = "uint32_t";
        }
        std::string dst_ptr = GetUBufPtr(codegen, op->args_[0], intlv_ptr_type);
        codegen.Emit("vsts(" + cast + src_reg + ", " + cast + src1 + ", " + dst_ptr + ", 0, " + dist + ", " + mask_reg +
                     ");");
    } else if (data_copy_mode == "DATA_BLOCK_LOAD" || data_copy_mode == "DATA_BLOCK_COPY") {
        // Accept both DATA_BLOCK_LOAD (pypto legacy name) and DATA_BLOCK_COPY
        //, consistent with load_align behavior.
        // vsstb only supports b8/b16/b32 (not b64)
        DataType dc_src_dt = GetExprDtype(op->args_[1]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                          dc_src_dt.GetBit() == 8 || dc_src_dt.GetBit() == 16 || dc_src_dt.GetBit() == 32)
            << "vf.store_align (DATA_BLOCK_COPY) only supports b8/b16/b32, got " << DTypeStr(dc_src_dt);
        // In DataBlock mode, args[2] is a mask register
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() >= 3)
            << "vf.store_align (DATA_BLOCK_COPY) requires at least 3 args (dst_ptr, src_reg, mask)";
        auto db_mask_var = ir::As<ir::Var>(op->args_[2]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, db_mask_var != nullptr)
            << "vf.store_align (DATA_BLOCK_COPY) requires args[2] to be a mask register, but got non-Var type";
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, codegen.IsMaskRegVar(codegen.GetVarName(db_mask_var)))
            << "vf.store_align (DATA_BLOCK_COPY) requires args[2] to be a mask register";
        std::string mask_reg = codegen.GetExprAsCode(op->args_[2]);
        std::string block_stride = "0";
        std::string repeat_stride = "0";
        if (op->args_.size() >= 4) {
            block_stride = codegen.GetExprAsCode(op->args_[3]);
        } else if (op->HasKwarg("block_stride")) {
            block_stride = std::to_string(op->GetKwarg<int>("block_stride"));
        }
        if (op->args_.size() >= 5) {
            repeat_stride = codegen.GetExprAsCode(op->args_[4]);
        } else if (op->HasKwarg("repeat_stride")) {
            repeat_stride = std::to_string(op->GetKwarg<int>("repeat_stride"));
        }
        if (post_update) {
            std::string ptr_var = codegen.GetOrCreateVFTilePtr(op->args_[0], /*is_post_update=*/true);
            codegen.Emit("vsstb(" + cast + src_reg + ", " + ptr_var + ", " + "(" + block_stride + " << 16u) | (" +
                         repeat_stride + " & 0xFFFFU), " + mask_reg + ", POST_UPDATE);");
        } else {
            std::string dst_ptr = GetUBufPtr(codegen, op->args_[0], ptr_type);
            codegen.Emit("vsstb(" + cast + src_reg + ", " + dst_ptr + ", " + "(" + block_stride + " << 16u) | (" +
                         repeat_stride + " & 0xFFFFU), " + mask_reg + ");");
        }
    } else if (post_update) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() >= 3)
            << "vf.store_align (post_update) requires at least 3 args (dst_ptr, src_reg, mask)";
        auto pu_mask_var = ir::As<ir::Var>(op->args_[2]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, pu_mask_var != nullptr)
            << "vf.store_align requires args[2] to be a mask register, but got non-Var type";
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, codegen.IsMaskRegVar(codegen.GetVarName(pu_mask_var)))
            << "vf.store_align requires args[2] to be a mask register";
        std::string mask_reg = codegen.GetExprAsCode(op->args_[2]);
        std::string stride = (op->args_.size() >= 4) ? codegen.GetExprAsCode(op->args_[3]) : "0";
        // B64 register types need stride doubled (postUpdateStride * 2)
        std::string effective_stride = stride;
        if (src_dt.GetBit() == 64) {
            effective_stride = "(" + stride + ") * 2";
        } else if (!width_op.empty()) {
            effective_stride = "(" + stride + ")" + width_op;
        }
        std::string ptr_var = codegen.GetOrCreateVFTilePtr(op->args_[0], /*is_post_update=*/true);
        // B64 types: use b32 vsts with ppack+pintlv_b32 mask splitting (mirrors AscendC
        // DataCopyImpl post_update b64 path). The b64 register is reinterpreted as
        // RegTensor<uint32_t>& and stored via b32 vsts with the split mask.
        if (src_dt.GetBit() == 64) {
            std::string p = src_reg + "_b64pu_";
            std::string tm = p + "tm_";
            std::string m0 = p + "m0_";
            std::string m1 = p + "m1_";
            codegen.Emit("MaskReg " + tm + ";");
            codegen.Emit("ppack(" + tm + ", " + mask_reg + ", LOWER);");
            codegen.Emit("MaskReg " + m0 + ", " + m1 + ";");
            codegen.Emit("pintlv_b32(" + m0 + ", " + m1 + ", " + tm + ", " + tm + ");");
            codegen.Emit("vsts((RegTensor<uint32_t>&)" + src_reg + ", (__ubuf__ uint32_t*&)" + ptr_var + ", " +
                         effective_stride + ", " + dist + ", " + m0 + ", POST_UPDATE);");
        } else {
            codegen.Emit("vsts(" + cast + src_reg + ", " + ptr_var + ", " + effective_stride + ", " + dist + ", " +
                         mask_reg + ", POST_UPDATE);");
        }
    } else {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() >= 3)
            << "vf.store_align requires at least 3 args (dst_ptr, src_reg, mask)";
        auto def_mask_var = ir::As<ir::Var>(op->args_[2]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, def_mask_var != nullptr)
            << "vf.store_align requires args[2] to be a mask register, but got non-Var type";
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, codegen.IsMaskRegVar(codegen.GetVarName(def_mask_var)))
            << "vf.store_align requires args[2] to be a mask register";
        std::string mask_reg = codegen.GetExprAsCode(op->args_[2]);
        std::string dst_ptr = GetUBufPtr(codegen, op->args_[0], ptr_type);
        std::string offset_str = "0";
        if (op->args_.size() >= 4) {
            offset_str = ResolveOffsetArg(codegen, op->args_[3], op->args_[0]);
        }
        std::string effective_offset = offset_str;
        if (!width_op.empty()) {
            effective_offset = "(" + offset_str + ")" + width_op;
        }
        // B64 types (INT64/UINT64): use b32 vsts with ppack+pintlv_b32 mask splitting
        // (mirrors AscendC DataCopyImpl b64 path). The b64 register is reinterpreted
        // as RegTensor<uint32_t>& and stored via b32 vsts with the split mask.
        if (src_dt.GetBit() == 64) {
            std::string p = src_reg + "_b64st_";
            std::string tm = p + "tm_";
            std::string m0 = p + "m0_";
            std::string m1 = p + "m1_";
            codegen.Emit("MaskReg " + tm + ";");
            codegen.Emit("ppack(" + tm + ", " + mask_reg + ", LOWER);");
            codegen.Emit("MaskReg " + m0 + ", " + m1 + ";");
            codegen.Emit("pintlv_b32(" + m0 + ", " + m1 + ", " + tm + ", " + tm + ");");
            std::string b32_ptr = GetUBufPtr(codegen, op->args_[0], "uint32_t");
            codegen.Emit("vsts((RegTensor<uint32_t>&)" + src_reg + ", " + b32_ptr + ", " + effective_offset + ", " +
                         dist + ", " + m0 + ");");
        } else {
            codegen.Emit("vsts(" + cast + src_reg + ", " + dst_ptr + ", " + effective_offset + ", " + dist + ", " +
                         mask_reg + ");");
        }
    }
    return "";
}

// ============================================================================
// MemBar — mem_bar
// ============================================================================

static std::string EmitVFMemBar(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    std::string mode = "VST_VLD";
    if (op->HasKwarg("mode")) {
        mode = VFEnumValueName(ir::EnumToString(static_cast<ir::MemBarMode>(op->GetKwarg<int>("mode"))));
    }
    codegen.Emit("mem_bar(" + mode + ");");
    return "";
}

// ============================================================================
// Max — vmax
// ============================================================================

static std::string EmitVFMax(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // Parser args order: [dst, src0, src1, mask]
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.max requires 4 args (dst, src0, src1, mask)";
    DataType s0_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (IsArithIntType(s0_dt) || s0_dt == DataType::FP16 ||
                                                    s0_dt == DataType::FP32 || s0_dt == DataType::BF16))
        << "vf.max src only supports supported types, got " << DTypeStr(s0_dt);
    DataType s1_dt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (IsArithIntType(s1_dt) || s1_dt == DataType::FP16 ||
                                                    s1_dt == DataType::FP32 || s1_dt == DataType::BF16))
        << "vf.max src only supports supported types, got " << DTypeStr(s1_dt);
    DataType vf_max_dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, s0_dt == vf_max_dst_dt && s1_dt == vf_max_dst_dt)
        << "vf.max requires dst, src0, src1 to have the same type, got dst=" << DTypeStr(vf_max_dst_dt)
        << " src0=" << DTypeStr(s0_dt) << " src1=" << DTypeStr(s1_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    // MERGING is not supported by the underlying vmax instruction on current device.
    std::string mode = VFZeroingOnly(op, "vf.max");
    if (s0_dt.GetBit() == 64) {
        // B64 max via deinterleave + b32 compare + vsel + interleave.
        // For INT64: compare high halves (signed), then low halves (unsigned) on tie.
        // For UINT64: compare high halves (unsigned), then low halves (unsigned) on tie.
        std::string p = dst + "_max_";
        EmitB64Deinterleave(codegen, p + "s0", src0);
        EmitB64Deinterleave(codegen, p + "s1", src1);
        std::string lo0 = p + "s0_lo_", hi0 = p + "s0_hi_";
        std::string lo1 = p + "s1_lo_", hi1 = p + "s1_hi_";
        std::string all_m = p + "_allm_";
        codegen.Emit("MaskReg " + all_m + " = pset_b32(PAT_ALL);");
        // Compare high halves
        std::string hi_gt = p + "_higt_";
        std::string hi_eq = p + "_hieq_";
        std::string lo_gt = p + "_logt_";
        codegen.Emit("MaskReg " + hi_gt + ";");
        codegen.Emit("MaskReg " + hi_eq + ";");
        codegen.Emit("MaskReg " + lo_gt + ";");
        if (s0_dt == DataType::INT64) {
            codegen.Emit("vcmp_gt(" + hi_gt + ", (RegTensor<int32_t>&)" + hi0 + ", (RegTensor<int32_t>&)" + hi1 + ", " +
                         all_m + ");");
            codegen.Emit("vcmp_eq(" + hi_eq + ", (RegTensor<int32_t>&)" + hi0 + ", (RegTensor<int32_t>&)" + hi1 + ", " +
                         all_m + ");");
        } else {
            codegen.Emit("vcmp_gt(" + hi_gt + ", " + hi0 + ", " + hi1 + ", " + all_m + ");");
            codegen.Emit("vcmp_eq(" + hi_eq + ", " + hi0 + ", " + hi1 + ", " + all_m + ");");
        }
        codegen.Emit("vcmp_gt(" + lo_gt + ", " + lo0 + ", " + lo1 + ", " + all_m + ");");
        // Combine: src0 > src1 iff hi_gt OR (hi_eq AND lo_gt)
        std::string eq_lo = p + "_eqlo_";
        std::string s0_gt = p + "_s0gt_";
        codegen.Emit("MaskReg " + eq_lo + ";");
        codegen.Emit("MaskReg " + s0_gt + ";");
        codegen.Emit("pand(" + eq_lo + ", " + hi_eq + ", " + lo_gt + ", " + all_m + ");");
        codegen.Emit("por(" + s0_gt + ", " + hi_gt + ", " + eq_lo + ", " + all_m + ");");
        // Select: dst = src0 > src1 ? src0 : src1 (per b32 half)
        std::string lo_dst = p + "_lod_";
        std::string hi_dst = p + "_hid_";
        codegen.Emit("RegTensor<uint32_t> " + lo_dst + ";");
        codegen.Emit("RegTensor<uint32_t> " + hi_dst + ";");
        codegen.Emit("vsel(" + lo_dst + ", " + lo0 + ", " + lo1 + ", " + s0_gt + ");");
        codegen.Emit("vsel(" + hi_dst + ", " + hi0 + ", " + hi1 + ", " + s0_gt + ");");
        // Interleave back to b64
        EmitB64Interleave(codegen, dst, lo_dst, hi_dst, p + "_ilv");
        // Apply ZEROING
        EmitB64Zeroing(codegen, dst, dst, mask, p + "_zero");
    } else {
        codegen.Emit("vmax(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// Add — vadd
// ============================================================================

static std::string EmitVFAdd(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.add requires 4 args (dst, src0, src1, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    DataType s0_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (IsArithIntType(s0_dt) || s0_dt == DataType::FP16 ||
                                                    s0_dt == DataType::FP32 || s0_dt == DataType::BF16))
        << "vf.add src0 only supports INT/UINT/FP16/FP32/BF16, got " << DTypeStr(s0_dt);
    DataType dst_dt = GetExprDtype(op->args_[0]);
    DataType s1_dt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, dst_dt == s0_dt && dst_dt == s1_dt)
        << "vf.add requires dst, src0, src1 to have the same type, got dst=" << DTypeStr(dst_dt)
        << " src0=" << DTypeStr(s0_dt) << " src1=" << DTypeStr(s1_dt);
    // MERGING is not supported by the underlying vadd instruction on current device.
    std::string mode = VFZeroingOnly(op, "vf.add");
    if (s0_dt.GetBit() == 64) {
        // B64 add: Mirrors AscendC AddB64Impl (vaddc + vaddcs carry-chain).
        // Uses CalTraitOneByTransToTraitTwo pattern: ppack mask, deinterleave both sources,
        // AddB64 (vaddc + vaddcs), interleave back.
        std::string p = dst + "_add_";
        std::string b32_cast = (s0_dt == DataType::INT64) ? "(RegTensor<int32_t>&)" : "(RegTensor<uint32_t>&)";
        std::string packed_m = p + "_pm_";
        codegen.Emit("MaskReg " + packed_m + ";");
        codegen.Emit("ppack(" + packed_m + ", " + mask + ", LOWER);");
        EmitB64Deinterleave(codegen, p + "s0", src0);
        EmitB64Deinterleave(codegen, p + "s1", src1);
        std::string lo0 = p + "s0_lo_", hi0 = p + "s0_hi_";
        std::string lo1 = p + "s1_lo_", hi1 = p + "s1_hi_";
        std::string carry = p + "_carry_";
        std::string lo_d = p + "_lod_";
        std::string hi_d = p + "_hid_";
        codegen.Emit("MaskReg " + carry + ";");
        codegen.Emit("RegTensor<uint32_t> " + lo_d + ";");
        codegen.Emit("RegTensor<uint32_t> " + hi_d + ";");
        codegen.Emit("vaddc(" + carry + ", " + b32_cast + lo_d + ", " + b32_cast + lo0 + ", " + b32_cast + lo1 + ", " +
                     packed_m + ");");
        codegen.Emit("vaddcs(" + carry + ", " + b32_cast + hi_d + ", " + b32_cast + hi0 + ", " + b32_cast + hi1 + ", " +
                     carry + ", " + packed_m + ");");
        EmitB64Interleave(codegen, dst, lo_d, hi_d, p + "_ilv");
    } else {
        codegen.Emit("vadd(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// Sub — vsub
// ============================================================================

static std::string EmitVFSub(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.sub requires 4 args (dst, src0, src1, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    DataType s0_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (IsArithIntType(s0_dt) || s0_dt == DataType::FP16 ||
                                                    s0_dt == DataType::FP32 || s0_dt == DataType::BF16))
        << "vf.sub src0 only supports INT/UINT/FP16/FP32/BF16, got " << DTypeStr(s0_dt);
    DataType dst_dt = GetExprDtype(op->args_[0]);
    DataType s1_dt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, dst_dt == s0_dt && dst_dt == s1_dt)
        << "vf.sub requires dst, src0, src1 to have the same type, got dst=" << DTypeStr(dst_dt)
        << " src0=" << DTypeStr(s0_dt) << " src1=" << DTypeStr(s1_dt);
    std::string mode = VFZeroingOnly(op, "vf.sub");
    if (s0_dt.GetBit() == 64) {
        // B64 sub: Mirrors AscendC SubB64Impl (vsubc + vsubcs borrow-chain).
        // Uses CalTraitOneByTransToTraitTwo pattern: ppack mask, deinterleave both sources,
        // SubB64 (vsubc + vsubcs), interleave back.
        std::string p = dst + "_sub_";
        std::string b32_cast = (s0_dt == DataType::INT64) ? "(RegTensor<int32_t>&)" : "(RegTensor<uint32_t>&)";
        std::string packed_m = p + "_pm_";
        codegen.Emit("MaskReg " + packed_m + ";");
        codegen.Emit("ppack(" + packed_m + ", " + mask + ", LOWER);");
        EmitB64Deinterleave(codegen, p + "s0", src0);
        EmitB64Deinterleave(codegen, p + "s1", src1);
        std::string lo0 = p + "s0_lo_", hi0 = p + "s0_hi_";
        std::string lo1 = p + "s1_lo_", hi1 = p + "s1_hi_";
        std::string borrow = p + "_borrow_";
        std::string lo_d = p + "_lod_";
        std::string hi_d = p + "_hid_";
        codegen.Emit("MaskReg " + borrow + ";");
        codegen.Emit("RegTensor<uint32_t> " + lo_d + ";");
        codegen.Emit("RegTensor<uint32_t> " + hi_d + ";");
        codegen.Emit("vsubc(" + borrow + ", " + b32_cast + lo_d + ", " + b32_cast + lo0 + ", " + b32_cast + lo1 + ", " +
                     packed_m + ");");
        codegen.Emit("vsubcs(" + borrow + ", " + b32_cast + hi_d + ", " + b32_cast + hi0 + ", " + b32_cast + hi1 +
                     ", " + borrow + ", " + packed_m + ");");
        EmitB64Interleave(codegen, dst, lo_d, hi_d, p + "_ilv");
    } else {
        codegen.Emit("vsub(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// And — vand
// ============================================================================

// ============================================================================
// And — vand, Xor — vxor, Or — vor
// Bitwise operations: type-agnostic, only requires same bit width across
// dst/src0/src1.  Any b8/b16/b32/b64 type (including FP16/BF16/FP32) is valid
// because vand/vxor/vor operate on raw bits.
// ============================================================================

static std::string EmitVFAnd(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.and_ requires 4 args (dst, src0, src1, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    if (IsDstMaskReg(op, codegen)) {
        codegen.Emit("pand(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ");");
        return "";
    }
    DataType dst_dt = GetExprDtype(op->args_[0]);
    DataType s0_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      s0_dt == DataType::INT8 || s0_dt == DataType::UINT8 || s0_dt == DataType::BOOL ||
                          s0_dt == DataType::FP8E4M3FN || s0_dt == DataType::FP8E5M2 || s0_dt == DataType::FP8E8M0 ||
                          s0_dt == DataType::HF8 || s0_dt == DataType::INT16 || s0_dt == DataType::UINT16 ||
                          s0_dt == DataType::FP16 || s0_dt == DataType::BF16 || s0_dt == DataType::INT32 ||
                          s0_dt == DataType::UINT32 || s0_dt == DataType::FP32 || s0_dt == DataType::INT64 ||
                          s0_dt == DataType::UINT64)
        << "vf.and_ src0 only supports "
           "INT8/UINT8/INT16/UINT16/FP16/BF16/INT32/UINT32/FP32/FP8E4M3FN/FP8E5M2/FP8E8M0/HF8/INT64/UINT64, got "
        << DTypeStr(s0_dt);
    DataType s1_dt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      s1_dt == DataType::INT8 || s1_dt == DataType::UINT8 || s1_dt == DataType::BOOL ||
                          s1_dt == DataType::FP8E4M3FN || s1_dt == DataType::FP8E5M2 || s1_dt == DataType::FP8E8M0 ||
                          s1_dt == DataType::HF8 || s1_dt == DataType::INT16 || s1_dt == DataType::UINT16 ||
                          s1_dt == DataType::FP16 || s1_dt == DataType::BF16 || s1_dt == DataType::INT32 ||
                          s1_dt == DataType::UINT32 || s1_dt == DataType::FP32 || s1_dt == DataType::INT64 ||
                          s1_dt == DataType::UINT64)
        << "vf.and_ src1 only supports "
           "INT8/UINT8/INT16/UINT16/FP16/BF16/INT32/UINT32/FP32/FP8E4M3FN/FP8E5M2/FP8E8M0/HF8/INT64/UINT64, got "
        << DTypeStr(s1_dt);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, dst_dt == s0_dt && dst_dt == s1_dt)
        << "vf.and_ requires dst, src0, src1 to have the same type, got dst=" << DTypeStr(dst_dt)
        << " src0=" << DTypeStr(s0_dt) << " src1=" << DTypeStr(s1_dt);
    std::string cast_prefix = "(RegTensor<" + dst_dt.ToCTypeString() + "> &)";
    std::string s0_expr = (s0_dt == dst_dt) ? src0 : (cast_prefix + src0);
    std::string s1_expr = (s1_dt == dst_dt) ? src1 : (cast_prefix + src1);
    std::string mode = VFZeroingOnly(op, "vf.and_");
    if (dst_dt.GetBit() == 64) {
        // B64 bitwise via b32 vand (mirrors AscendC AndImpl b64 path, see EmitB64Bitwise).
        EmitB64Bitwise(codegen, "vand", {s0_expr, s1_expr}, dst, mask, mode, dst + "_and_");
    } else {
        codegen.Emit("vand(" + dst + ", " + s0_expr + ", " + s1_expr + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// Xor — vxor, Or — vor
// Bitwise operations: type-agnostic, only requires same bit width across
// dst/src0/src1.  Any b8/b16/b32/b64 type (including FP16/BF16/FP32) is valid
// because vxor/vor operate on raw bits.
// ============================================================================

static std::string EmitVFBinaryBitwise(const ir::CallPtr& op, codegen::CodegenBase& codegen_base,
                                       const std::string& op_name, const std::string& instruction)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << op_name << " requires 4 args (dst, src0, src1, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    if (IsDstMaskReg(op, codegen)) {
        std::string p_instr = "p" + instruction.substr(1);
        codegen.Emit(p_instr + "(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ");");
        return "";
    }
    DataType dst_dt = GetExprDtype(op->args_[0]);
    DataType s0_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      s0_dt == DataType::INT8 || s0_dt == DataType::UINT8 || s0_dt == DataType::BOOL ||
                          s0_dt == DataType::FP8E4M3FN || s0_dt == DataType::FP8E5M2 || s0_dt == DataType::FP8E8M0 ||
                          s0_dt == DataType::HF8 || s0_dt == DataType::INT16 || s0_dt == DataType::UINT16 ||
                          s0_dt == DataType::FP16 || s0_dt == DataType::BF16 || s0_dt == DataType::INT32 ||
                          s0_dt == DataType::UINT32 || s0_dt == DataType::FP32 || s0_dt == DataType::INT64 ||
                          s0_dt == DataType::UINT64)
        << op_name
        << " src0 only supports "
           "INT8/UINT8/INT16/UINT16/FP16/BF16/INT32/UINT32/FP32/FP8E4M3FN/FP8E5M2/FP8E8M0/HF8/INT64/UINT64, got "
        << DTypeStr(s0_dt);
    DataType s1_dt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      s1_dt == DataType::INT8 || s1_dt == DataType::UINT8 || s1_dt == DataType::BOOL ||
                          s1_dt == DataType::FP8E4M3FN || s1_dt == DataType::FP8E5M2 || s1_dt == DataType::FP8E8M0 ||
                          s1_dt == DataType::HF8 || s1_dt == DataType::INT16 || s1_dt == DataType::UINT16 ||
                          s1_dt == DataType::FP16 || s1_dt == DataType::BF16 || s1_dt == DataType::INT32 ||
                          s1_dt == DataType::UINT32 || s1_dt == DataType::FP32 || s1_dt == DataType::INT64 ||
                          s1_dt == DataType::UINT64)
        << op_name
        << " src1 only supports "
           "INT8/UINT8/INT16/UINT16/FP16/BF16/INT32/UINT32/FP32/FP8E4M3FN/FP8E5M2/FP8E8M0/HF8/INT64/UINT64, got "
        << DTypeStr(s1_dt);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, dst_dt == s0_dt && dst_dt == s1_dt)
        << op_name << " requires dst, src0, src1 to have the same type, got dst=" << DTypeStr(dst_dt)
        << " src0=" << DTypeStr(s0_dt) << " src1=" << DTypeStr(s1_dt);
    std::string cast_prefix = "(RegTensor<" + dst_dt.ToCTypeString() + "> &)";
    std::string s0_expr = (s0_dt == dst_dt) ? src0 : (cast_prefix + src0);
    std::string s1_expr = (s1_dt == dst_dt) ? src1 : (cast_prefix + src1);
    std::string mode = VFZeroingOnly(op, op_name);
    if (dst_dt.GetBit() == 64) {
        // B64 bitwise via b32 instruction (mirrors AscendC XorImpl/OrImpl b64 path, see EmitB64Bitwise).
        std::string p = dst + "_" + instruction.substr(1) + "_";
        EmitB64Bitwise(codegen, instruction, {s0_expr, s1_expr}, dst, mask, mode, p);
    } else {
        codegen.Emit(instruction + "(" + dst + ", " + s0_expr + ", " + s1_expr + ", " + mask + ", " + mode + ");");
    }
    return "";
}

static std::string EmitVFXor(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return EmitVFBinaryBitwise(op, codegen_base, "vf.xor", "vxor");
}

// ============================================================================
// Or — vor
// ============================================================================

static std::string EmitVFOr(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return EmitVFBinaryBitwise(op, codegen_base, "vf.or_", "vor");
}

// ============================================================================
// Reduce — vcadd/vcmax/vcmin + vcgadd/vcgmax/vcgmin (unified)
// Supports both new-style (mode=SUM/MAX/MIN, datablock) and legacy (reduce_type=ADD/MAX, merge_mode)
// ============================================================================

// Shared reduction emitter. `reduce_mode` is one of "SUM"/"MAX"/"MIN".
static std::string EmitVFReduceImpl(const ir::CallPtr& op, codegen::CodegenBase& codegen_base,
                                    const std::string& reduce_mode)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << op->name_ << " requires 3 args (dst, src, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    bool datablock = false;
    if (op->HasKwarg("datablock")) {
        datablock = op->GetKwarg<bool>("datablock");
    }
    if (datablock) {
        // Datablock reduce only supports b16/b32 (no b64, no BF16)
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                          (src_dt.GetBit() == 16 || src_dt.GetBit() == 32) &&
                              (IsArithIntType(src_dt) || src_dt == DataType::FP16 || src_dt == DataType::FP32))
            << op->name_ << " (datablock) src only supports b16/b32 INT/UINT/FP16/FP32, got " << DTypeStr(src_dt);
        DataType reduce_dst_dt = GetExprDtype(op->args_[0]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == reduce_dst_dt)
            << op->name_ << " requires src and dst to have the same type, got dst=" << DTypeStr(reduce_dst_dt)
            << " src=" << DTypeStr(src_dt);
    } else {
        // Non-datablock reduce supports b16/b32/b64 (no BF16)
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                          (src_dt.GetBit() == 16 || src_dt.GetBit() == 32 || src_dt.GetBit() == 64) &&
                              (IsArithIntType(src_dt) || src_dt == DataType::FP16 || src_dt == DataType::FP32))
            << op->name_ << " src only supports b16/b32/b64 INT/UINT/FP16/FP32, got " << DTypeStr(src_dt);
        DataType reduce_dst_dt = GetExprDtype(op->args_[0]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == reduce_dst_dt)
            << op->name_ << " requires src and dst to have the same type, got dst=" << DTypeStr(reduce_dst_dt)
            << " src=" << DTypeStr(src_dt);
    }
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    // vcadd for 16-bit int requires 32-bit accumulator: s16→s32, u16→u32
    if (!datablock && reduce_mode == "SUM") {
        if (src_dt == DataType::INT16) {
            dst = "(RegTensor<int32_t>&)" + dst;
        } else if (src_dt == DataType::UINT16) {
            dst = "(RegTensor<uint32_t>&)" + dst;
        }
    }
    std::string intrinsic;
    if (reduce_mode == "SUM")
        intrinsic = datablock ? "vcgadd" : "vcadd";
    else if (reduce_mode == "MAX")
        intrinsic = datablock ? "vcgmax" : "vcmax";
    else
        intrinsic = datablock ? "vcgmin" : "vcmin";
    // reduce ops use "merge_mode" kwarg (not "mode")
    std::string mode = "MODE_ZEROING";
    if (op->HasKwarg("merge_mode")) {
        auto merge_mode = static_cast<ir::MergeMode>(op->GetKwarg<int>("merge_mode"));
        PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, merge_mode == ir::MergeMode::ZEROING)
            << op->name_ << " only supports ZEROING mode on current device, but got MERGING";
        mode = merge_mode == ir::MergeMode::MERGING ? "MODE_MERGING" : "MODE_ZEROING";
    }
    if (!datablock && src_dt.GetBit() == 64) {
        // B64 reduce (INT64/UINT64): no native b64 vcadd/vcmax/vcmin. Defer to
        // the b64 emulation helpers (mirrors AscendC ReduceSumB64Impl /
        // ReduceMaxB64Impl / ReduceMinB64Impl). The result lands in element 0
        // of dst; the remaining dst lanes are don't-care.
        if (reduce_mode == "SUM") {
            EmitB64ReduceSum(codegen, dst, src, mask);
        } else {
            EmitB64ReduceMaxMin(codegen, dst, src, mask, reduce_mode == "MAX", src_dt == DataType::INT64);
        }
        return "";
    }
    codegen.Emit(intrinsic + "(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    return "";
}

static std::string EmitVFReduceSum(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return EmitVFReduceImpl(op, codegen_base, "SUM");
}

static std::string EmitVFReduceMax(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return EmitVFReduceImpl(op, codegen_base, "MAX");
}

static std::string EmitVFReduceMin(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return EmitVFReduceImpl(op, codegen_base, "MIN");
}

// ============================================================================
// Mul — vmul
// ============================================================================

static std::string EmitVFMul(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.mul requires 4 args (dst, src0, src1, mask)";
    DataType s0_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (s0_dt.GetBit() == 16 || s0_dt.GetBit() == 32 || s0_dt.GetBit() == 64))
        << "vf.mul src only supports supported types, got " << DTypeStr(s0_dt);
    DataType s1_dt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (s1_dt.GetBit() == 16 || s1_dt.GetBit() == 32 || s1_dt.GetBit() == 64))
        << "vf.mul src only supports supported types, got " << DTypeStr(s1_dt);
    DataType vf_mul_dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, s0_dt == vf_mul_dst_dt && s1_dt == vf_mul_dst_dt)
        << "vf.mul requires dst, src0, src1 to have the same type, got dst=" << DTypeStr(vf_mul_dst_dt)
        << " src0=" << DTypeStr(s0_dt) << " src1=" << DTypeStr(s1_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.mul");
    if (s0_dt.GetBit() == 64) {
        // B64 mul: Mirrors AscendC MulB64Impl (vmull + vmula).
        // MulB64Impl uses B64TraitOneToTraitTwo to split both sources into b32 halves,
        // then vmull (32x32→64 long multiply) + vmula (cross-term accumulate).
        std::string p = dst + "_mul_";
        // AscendC MulB64Impl: vmull always uses uint32_t (unsigned 32x32→64);
        // vmula uses int32_t for int64_t (signed cross-term), uint32_t for uint64_t.
        std::string vmull_cast = "(RegTensor<uint32_t>&)";
        std::string vmula_cast = (s0_dt == DataType::INT64) ? "(RegTensor<int32_t>&)" : "(RegTensor<uint32_t>&)";
        // Pack b64 mask to b32 (mirrors MaskPack in CalTraitOneByTransToTraitTwo)
        std::string packed_m = p + "_pm_";
        codegen.Emit("MaskReg " + packed_m + ";");
        codegen.Emit("ppack(" + packed_m + ", " + mask + ", LOWER);");
        // Deinterleave src0 and src1 to b32 halves
        EmitB64Deinterleave(codegen, p + "s0", src0);
        EmitB64Deinterleave(codegen, p + "s1", src1);
        std::string lo0 = p + "s0_lo_", hi0 = p + "s0_hi_";
        std::string lo1 = p + "s1_lo_", hi1 = p + "s1_hi_";
        // MulB64: vmull (lo0 × lo1 → dst_lo, dst_hi) + vmula (lo0 × hi1 → dst_hi) + vmula (hi0 × lo1 → dst_hi)
        std::string lo_d = p + "_lod_";
        std::string hi_d = p + "_hid_";
        codegen.Emit("RegTensor<uint32_t> " + lo_d + ";");
        codegen.Emit("RegTensor<uint32_t> " + hi_d + ";");
        codegen.Emit("vmull(" + vmull_cast + lo_d + ", " + vmull_cast + hi_d + ", " + vmull_cast + lo0 + ", " +
                     vmull_cast + lo1 + ", " + packed_m + ");");
        codegen.Emit("vmula(" + vmula_cast + hi_d + ", " + vmula_cast + lo0 + ", " + vmula_cast + hi1 + ", " +
                     packed_m + ", MODE_ZEROING);");
        codegen.Emit("vmula(" + vmula_cast + hi_d + ", " + vmula_cast + hi0 + ", " + vmula_cast + lo1 + ", " +
                     packed_m + ", MODE_ZEROING);");
        // Interleave back (mirrors B64TraitTwoToTraitOne)
        EmitB64Interleave(codegen, dst, lo_d, hi_d, p + "_ilv");
    } else {
        codegen.Emit("vmul(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// MulAddDst — vmula (hardware FMA: dst = src0 * src1 + dst)
// ============================================================================

static std::string EmitVFMulAddDst(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.mul_add_dst requires 4 args (dst, src0, src1, mask)";
    DataType s0_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (s0_dt.GetBit() == 16 || s0_dt.GetBit() == 32 || s0_dt.GetBit() == 64))
        << "vf.mul_add_dst src only supports supported types, got " << DTypeStr(s0_dt);
    DataType s1_dt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (s1_dt.GetBit() == 16 || s1_dt.GetBit() == 32 || s1_dt.GetBit() == 64))
        << "vf.mul_add_dst src only supports supported types, got " << DTypeStr(s1_dt);
    DataType vf_mul_add_dst_dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, s0_dt == vf_mul_add_dst_dst_dt && s1_dt == vf_mul_add_dst_dst_dt)
        << "vf.mul_add_dst requires dst, src0, src1 to have the same type, got dst=" << DTypeStr(vf_mul_add_dst_dst_dt)
        << " src0=" << DTypeStr(s0_dt) << " src1=" << DTypeStr(s1_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.mul_add_dst");
    codegen.Emit("vmula(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Div — vdiv
// ============================================================================

static std::string EmitVFDiv(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.div requires 4 args (dst, src0, src1, mask)";
    DataType s0_dt = GetExprDtype(op->args_[1]);
    // No vdiv overloads for 8-bit ints (mirrors AscendC DivImpl: u16..i64 + half/float).
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (IsArithIntType(s0_dt) || s0_dt == DataType::FP16 || s0_dt == DataType::FP32) &&
                          s0_dt != DataType::INT8 && s0_dt != DataType::UINT8)
        << "vf.div src0 only supports INT16/UINT16/INT32/UINT32/INT64/UINT64/FP16/FP32, got " << DTypeStr(s0_dt);
    DataType s1_dt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (IsArithIntType(s1_dt) || s1_dt == DataType::FP16 || s1_dt == DataType::FP32) &&
                          s1_dt != DataType::INT8 && s1_dt != DataType::UINT8)
        << "vf.div src1 only supports INT16/UINT16/INT32/UINT32/INT64/UINT64/FP16/FP32, got " << DTypeStr(s1_dt);
    DataType vf_div_dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, s0_dt == vf_div_dst_dt && s1_dt == vf_div_dst_dt)
        << "vf.div requires dst, src0, src1 to have the same type, got dst=" << DTypeStr(vf_div_dst_dt)
        << " src0=" << DTypeStr(s0_dt) << " src1=" << DTypeStr(s1_dt);
    const bool use_precision = op->HasKwarg("precision") && op->GetKwarg<bool>("precision");
    // High-precision mode supports FP16/FP32 only; validated up front before
    // any emission (B64 division and integer types take the standard path).
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, !use_precision || s0_dt == DataType::FP16 || s0_dt == DataType::FP32)
        << "vf.div high-precision mode only supports FP16/FP32, got " << DTypeStr(s0_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.div");
    if (s0_dt.GetBit() == 64) {
        // B64 division (INT64/UINT64): Newton-Raphson reciprocal refinement
        // mirroring AscendC DivS64Impl / DivU64Impl. No native b64 vdiv.
        EmitB64Div(codegen, dst, src0, src1, mask, s0_dt == DataType::INT64);
        return "";
    }
    if (use_precision) {
        // Mirrors AscendC high-precision division: error-complementation
        // quotient correction with inf/nan/zero bypass and subnormal-input
        // scaling. FP32 follows DivPrecisionImpl (vec_binary_impl.h:868-976);
        // FP16 follows DivIEEE754HalfImpl (vec_binary_impl.h:1199-1413).
        if (s0_dt == DataType::FP16) {
            // IEEE754 manual half division: normalize subnormal inputs by 2^10,
            // standardize exponents, divide, then compensate, clamp overflow/
            // underflow and bypass inf/zero/nan lanes.
            const std::string h = dst + "_h_";
            const std::string t0 = h + "t0", t1 = h + "t1", t2 = h + "t2", z1 = h + "z1", z2 = h + "z2";
            const std::string a0abs = h + "a0abs", a0sub = h + "a0sub", a0nrm = h + "a0nrm";
            const std::string a0all = h + "a0all", a0abn = h + "a0abn", a0exp = h + "a0exp";
            const std::string b1abs = h + "b1abs", b1sub = h + "b1sub", b1nrm = h + "b1nrm";
            const std::string b1all = h + "b1all", b1abn = h + "b1abn", b1exp = h + "b1exp";
            const std::string dsgn = h + "dsgn", scl = h + "scl";
            const std::string m0 = h + "m0", ms0n = h + "ms0n", ms0s = h + "ms0s", ms1n = h + "ms1n";
            const std::string ms1s = h + "ms1s", mt = h + "mt", mnan = h + "mnan", minf = h + "minf";
            const std::string mz0 = h + "mz0", mz1 = h + "mz1", mv = h + "mv", mn = h + "mn";
            codegen.Emit("union { uint16_t i; half f; } " + h + "thr_ = {0x03FF};");
            codegen.Emit("union { uint16_t i; half f; } " + h + "enl_ = {0x6400};");
            codegen.Emit("union { uint16_t i; half f; } " + h + "erd_ = {0x1400};");
            codegen.Emit("RegTensor<half> " + a0abs + ", " + a0sub + ", " + a0nrm + ", " + a0all + ", " + a0abn + ";");
            codegen.Emit("RegTensor<half> " + b1abs + ", " + b1sub + ", " + b1nrm + ", " + b1all + ", " + b1abn + ";");
            codegen.Emit("RegTensor<half> " + z1 + ", " + z2 + ";");
            codegen.Emit("RegTensor<uint16_t> " + t0 + ", " + t2 + ", " + a0exp + ", " + b1exp + ", " + dsgn + ";");
            codegen.Emit("RegTensor<int16_t> " + t1 + ", " + scl + ";");
            codegen.Emit("MaskReg " + m0 + ", " + ms0n + ", " + ms0s + ", " + ms1n + ", " + ms1s + ", " + mt + ";");
            codegen.Emit("MaskReg " + mnan + ", " + minf + ", " + mz0 + ", " + mz1 + ", " + mv + ", " + mn + ";");
            // Acquire valid numbers (no inf, no 0): DivIEEE754HalfImpl:1259-1281
            codegen.Emit("vabs(" + a0abs + ", " + src0 + ", " + mask + ", " + mode + ");");
            codegen.Emit("vabs(" + b1abs + ", " + src1 + ", " + mask + ", " + mode + ");");
            codegen.Emit("vdup(" + t0 + ", (uint16_t)0x7C00, " + mask + ", " + mode + ");");
            codegen.Emit("vcmp_eq(" + minf + ", (RegTensor<uint16_t>&)" + a0abs + ", " + t0 + ", " + mask + ");");
            codegen.Emit("vcmp_eq(" + mt + ", (RegTensor<uint16_t>&)" + b1abs + ", " + t0 + ", " + mask + ");");
            codegen.Emit("por(" + mv + ", " + minf + ", " + mt + ", " + mask + ");");
            codegen.Emit("vdup(" + t0 + ", (uint16_t)0x0, " + mask + ", " + mode + ");");
            codegen.Emit("vcmp_eq(" + mz0 + ", (RegTensor<uint16_t>&)" + a0abs + ", " + t0 + ", " + mask + ");");
            codegen.Emit("por(" + mv + ", " + mv + ", " + mz0 + ", " + mask + ");");
            codegen.Emit("vcmp_eq(" + mz1 + ", (RegTensor<uint16_t>&)" + b1abs + ", " + t0 + ", " + mask + ");");
            codegen.Emit("por(" + mv + ", " + mv + ", " + mz1 + ", " + mask + ");");
            codegen.Emit("pnot(" + mv + ", " + mv + ", " + mask + ");");
            // Normalize subnormal elements of src0/src1 (1284-1298)
            codegen.Emit("vcmps_lt(" + ms0s + ", " + a0abs + ", " + h + "thr_.f, " + mask + ");");
            codegen.Emit("pnot(" + ms0n + ", " + ms0s + ", " + mask + ");");
            codegen.Emit("vmuls(" + a0sub + ", " + src0 + ", " + h + "enl_.f, " + ms0s + ", " + mode + ");");
            codegen.Emit("vcmps_lt(" + ms1s + ", " + b1abs + ", " + h + "thr_.f, " + mask + ");");
            codegen.Emit("pnot(" + ms1n + ", " + ms1s + ", " + mask + ");");
            codegen.Emit("vmuls(" + b1sub + ", " + src1 + ", " + h + "enl_.f, " + ms1s + ", " + mode + ");");
            codegen.Emit("vsel(" + a0all + ", " + src0 + ", " + a0sub + ", " + ms0n + ");");
            codegen.Emit("vsel(" + b1all + ", " + src1 + ", " + b1sub + ", " + ms1n + ");");
            // Standardize the exponent bits of src0 and src1 (1301-1317).
            // Both steps are u16 bit-pattern ops (mask the exponent field away,
            // then integer-add the 1.0h exponent bits back in), so dst and the
            // recursive src0 are taken through the u16 register view as well:
            // vand/vadd require all three data operands to share one type.
            codegen.Emit("vdup(" + t0 + ", (uint16_t)0x83FF, " + mask + ", " + mode + ");");
            codegen.Emit("vand((RegTensor<uint16_t>&)" + a0nrm + ", (RegTensor<uint16_t>&)" + a0all + ", " + t0 + ", " +
                         mv + ", " + mode + ");");
            codegen.Emit("vand((RegTensor<uint16_t>&)" + b1nrm + ", (RegTensor<uint16_t>&)" + b1all + ", " + t0 + ", " +
                         mv + ", " + mode + ");");
            codegen.Emit("vdup(" + t0 + ", (uint16_t)0x3C00, " + mask + ", " + mode + ");");
            codegen.Emit("vadd((RegTensor<uint16_t>&)" + a0nrm + ", (RegTensor<uint16_t>&)" + a0nrm + ", " + t0 + ", " +
                         mv + ", " + mode + ");");
            codegen.Emit("vadd((RegTensor<uint16_t>&)" + b1nrm + ", (RegTensor<uint16_t>&)" + b1nrm + ", " + t0 + ", " +
                         mv + ", " + mode + ");");
            codegen.Emit("vsel(" + a0nrm + ", " + a0nrm + ", " + a0all + ", " + mv + ");");
            codegen.Emit("vsel(" + b1nrm + ", " + b1nrm + ", " + b1all + ", " + mv + ");");
            codegen.Emit("vabs(" + a0abn + ", " + a0nrm + ", " + mv + ", " + mode + ");");
            codegen.Emit("vabs(" + b1abn + ", " + b1nrm + ", " + mv + ", " + mode + ");");
            codegen.Emit("vcmp_le(" + mn + ", " + a0abn + ", " + b1abn + ", " + mv + ");");
            codegen.Emit("vdiv(" + dst + ", " + a0nrm + ", " + b1nrm + ", " + mask + ", " + mode + ");");
            // Normalization compensation for subnormal operands (1324-1334)
            codegen.Emit("pand(" + m0 + ", " + ms0s + ", " + ms1n + ", " + mask + ");");
            codegen.Emit("vmuls(" + z1 + ", " + dst + ", " + h + "erd_.f, " + m0 + ", " + mode + ");");
            codegen.Emit("vsel(" + dst + ", " + z1 + ", " + dst + ", " + m0 + ");");
            codegen.Emit("pand(" + m0 + ", " + ms0n + ", " + ms1s + ", " + mask + ");");
            codegen.Emit("vmuls(" + z1 + ", " + dst + ", " + h + "enl_.f, " + m0 + ", " + mode + ");");
            codegen.Emit("vsel(" + dst + ", " + z1 + ", " + dst + ", " + m0 + ");");
            // Preserve sign for the exception handling below (1337-1339)
            codegen.Emit("vdup(" + t0 + ", (uint16_t)0x8000, " + mask + ", " + mode + ");");
            codegen.Emit("vand(" + dsgn + ", (RegTensor<uint16_t>&)" + dst + ", " + t0 + ", " + mask + ", " + mode +
                         ");");
            // Exponent subtraction (effectively fp number division) (1342-1356)
            codegen.Emit("vdup(" + t0 + ", (uint16_t)0x7C00, " + mask + ", " + mode + ");");
            codegen.Emit("vand(" + a0exp + ", (RegTensor<uint16_t>&)" + a0all + ", " + t0 + ", " + mask + ", " + mode +
                         ");");
            codegen.Emit("vand(" + b1exp + ", (RegTensor<uint16_t>&)" + b1all + ", " + t0 + ", " + mask + ", " + mode +
                         ");");
            codegen.Emit("vshrs(" + a0exp + ", " + a0exp + ", (int16_t)10, " + mask + ", " + mode + ");");
            codegen.Emit("vshrs(" + b1exp + ", " + b1exp + ", (int16_t)10, " + mask + ", " + mode + ");");
            codegen.Emit("vsub(" + scl + ", (RegTensor<int16_t>&)" + a0exp + ", (RegTensor<int16_t>&)" + b1exp + ", " +
                         mask + ", " + mode + ");");
            codegen.Emit("vadds(" + scl + ", " + scl + ", (int16_t)15, " + mask + ", " + mode + ");");
            // scale == -9: clamp to signed min denormal (1360-1368)
            codegen.Emit("vdup(" + t1 + ", (int16_t)-9, " + mask + ", " + mode + ");");
            codegen.Emit("vcmp_eq(" + m0 + ", " + scl + ", " + t1 + ", " + mask + ");");
            codegen.Emit("pand(" + m0 + ", " + m0 + ", " + mv + ", " + mask + ");");
            codegen.Emit("vdup(" + t0 + ", (uint16_t)0x1, " + m0 + ", " + mode + ");");
            codegen.Emit("vadd((RegTensor<uint16_t>&)" + z1 + ", " + dsgn + ", " + t0 + ", " + m0 + ", " + mode + ");");
            codegen.Emit("vdup(" + t2 + ", (uint16_t)0x0, " + m0 + ", " + mode + ");");
            codegen.Emit("vadd((RegTensor<uint16_t>&)" + z2 + ", " + dsgn + ", " + t2 + ", " + m0 + ", " + mode + ");");
            codegen.Emit("vsel(" + z1 + ", " + z2 + ", " + z1 + ", " + mn + ");");
            codegen.Emit("vsel(" + dst + ", " + z1 + ", " + dst + ", " + m0 + ");");
            codegen.Emit("pnot(" + m0 + ", " + m0 + ", " + mask + ");");
            codegen.Emit("pand(" + mv + ", " + m0 + ", " + mv + ", " + mask + ");");
            // scale < -9: underflow to signed zero (1370-1375)
            codegen.Emit("vcmp_lt(" + m0 + ", " + scl + ", " + t1 + ", " + mask + ");");
            codegen.Emit("pand(" + m0 + ", " + m0 + ", " + mv + ", " + mask + ");");
            codegen.Emit("vdup(" + t0 + ", (uint16_t)0x0, " + mask + ", " + mode + ");");
            codegen.Emit("vadd((RegTensor<uint16_t>&)" + z1 + ", " + dsgn + ", " + t0 + ", " + m0 + ", " + mode + ");");
            codegen.Emit("vsel(" + dst + ", " + z1 + ", " + dst + ", " + m0 + ");");
            codegen.Emit("pnot(" + m0 + ", " + m0 + ", " + mask + ");");
            codegen.Emit("pand(" + mv + ", " + m0 + ", " + mv + ", " + mask + ");");
            // scale == 31: double the result and fix the exponent (1377-1384)
            codegen.Emit("vdup(" + t0 + ", (uint16_t)0x1F, " + mask + ", " + mode + ");");
            codegen.Emit("vcmp_eq(" + m0 + ", " + scl + ", (RegTensor<int16_t>&)" + t0 + ", " + mask + ");");
            codegen.Emit("pand(" + m0 + ", " + m0 + ", " + mv + ", " + mask + ");");
            codegen.Emit("vdup(" + t1 + ", (int16_t)0x1, " + m0 + ", " + mode + ");");
            codegen.Emit("vsub(" + t1 + ", " + scl + ", " + t1 + ", " + m0 + ", " + mode + ");");
            codegen.Emit("vsel(" + scl + ", " + t1 + ", " + scl + ", " + m0 + ");");
            codegen.Emit("vmuls(" + z1 + ", " + dst + ", 2.0f, " + m0 + ", " + mode + ");");
            codegen.Emit("vsel(" + dst + ", " + z1 + ", " + dst + ", " + m0 + ");");
            // scale > 31: overflow to signed infinity (1386-1394)
            codegen.Emit("vcmp_gt(" + m0 + ", " + scl + ", (RegTensor<int16_t>&)" + t0 + ", " + mask + ");");
            codegen.Emit("pand(" + m0 + ", " + m0 + ", " + mv + ", " + mask + ");");
            codegen.Emit("vdup(" + t0 + ", (uint16_t)0x7C00, " + mask + ", " + mode + ");");
            codegen.Emit("vadd((RegTensor<uint16_t>&)" + z1 + ", " + dsgn + ", " + t0 + ", " + m0 + ", " + mode + ");");
            codegen.Emit("vsel(" + dst + ", " + z1 + ", " + dst + ", " + m0 + ");");
            codegen.Emit("pnot(" + m0 + ", " + m0 + ", " + mask + ");");
            codegen.Emit("pand(" + mv + ", " + m0 + ", " + mv + ", " + mask + ");");
            // scale > 0: rescale by 2^(10*scale) (1396-1403)
            codegen.Emit("vdup(" + t0 + ", (uint16_t)0x0, " + mv + ", " + mode + ");");
            codegen.Emit("vcmp_gt(" + m0 + ", " + scl + ", (RegTensor<int16_t>&)" + t0 + ", " + mv + ");");
            codegen.Emit("vshls(" + t1 + ", " + scl + ", (int16_t)10, " + m0 + ", " + mode + ");");
            codegen.Emit("vmul(" + z1 + ", " + dst + ", (RegTensor<half>&)" + t1 + ", " + m0 + ", " + mode + ");");
            codegen.Emit("vsel(" + dst + ", " + z1 + ", " + dst + ", " + m0 + ");");
            // scale <= 0: subnormal result, scale by half(512 >> |scale|) (1405-1416)
            codegen.Emit("pnot(" + m0 + ", " + m0 + ", " + mv + ");");
            codegen.Emit("vdup(" + t0 + ", (uint16_t)0x200, " + m0 + ", " + mode + ");");
            codegen.Emit("vabs(" + scl + ", " + scl + ", " + m0 + ", " + mode + ");");
            codegen.Emit("vshr(" + scl + ", (RegTensor<int16_t>&)" + t0 + ", " + scl + ", " + m0 + ", " + mode + ");");
            codegen.Emit("vmul(" + z1 + ", " + dst + ", (RegTensor<half>&)" + scl + ", " + m0 + ", " + mode + ");");
            codegen.Emit("vsel(" + dst + ", " + z1 + ", " + dst + ", " + m0 + ");");
            // Set output with nan input to nan (1418-1424)
            codegen.Emit("vdup(" + t0 + ", (uint16_t)0x7E00, " + mask + ", " + mode + ");");
            codegen.Emit("vcmp_ne(" + mnan + ", " + a0abs + ", " + a0abs + ", " + mask + ");");
            codegen.Emit("vcmp_ne(" + mt + ", " + b1abs + ", " + b1abs + ", " + mask + ");");
            codegen.Emit("por(" + mnan + ", " + mnan + ", " + mt + ", " + mask + ");");
            codegen.Emit("vsel(" + dst + ", (RegTensor<half>&)" + t0 + ", " + dst + ", " + mnan + ");");
            return "";
        }
        const std::string p = dst + "_p_";
        const std::string pall = p + "all", nz = p + "nz", infnan = p + "infn", z = p + "z", q0 = p + "q0";
        const std::string m_inf = p + "minf", m_zero = p + "mzero", m_scale = p + "mscale";
        const std::string m_cmp = p + "mcmp", expbits = p + "expb", scratch = p + "scr", expo = p + "expo";
        const std::string kreg = p + "k", thrvec = p + "thr", zerovec = p + "zv", newexp = p + "newexp";
        const std::string scalebits = p + "sb", scale = p + "sc", one = p + "one";
        const std::string asc = p + "asc", bsc = p + "bsc", y = p + "y";
        const std::string r = p + "r", rpre = p + "rpre", rnext = p + "rnext", zpre = p + "zpre", znext = p + "znext";
        codegen.Emit("MaskReg " + pall + " = pset_b8(PAT_ALL);");
        codegen.Emit("RegTensor<uint32_t> " + nz + ";");
        codegen.Emit("RegTensor<uint32_t> " + infnan + ";");
        codegen.Emit("RegTensor<float> " + z + ";");
        codegen.Emit("RegTensor<float> " + q0 + ";");
        codegen.Emit("MaskReg " + m_inf + ";");
        codegen.Emit("MaskReg " + m_zero + ";");
        // Inf/nan/zero bypass (DivPrecisionImpl:908-914): set the sign bit of the
        // quotient, then >= 0xFF800000 (unsigned) catches +/-Inf, +/-0 and NaNs.
        codegen.Emit("vdup(" + nz + ", (int32_t)0x80000000, " + pall + ", " + mode + ");");
        codegen.Emit("vdiv(" + z + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
        codegen.Emit("vor(" + infnan + ", (RegTensor<uint32_t>&)" + z + ", " + nz + ", " + mask + ", " + mode + ");");
        codegen.Emit("vmov(" + q0 + ", " + z + ");");
        codegen.Emit("vcmps_eq(" + m_zero + ", " + z + ", 0.0f, " + mask + ");");
        codegen.Emit("vcmps_ge(" + m_inf + ", " + infnan + ", (uint32_t)0xFF800000, " + mask + ");");
        codegen.Emit("por(" + m_inf + ", " + m_inf + ", " + m_zero + ", " + mask + ");");
        // Subnormal-input scaling (DivPrecisionImpl:916-950): scale a/b by 2^k with
        // k = max(-64 - exp(a), 0) so the residual stays in the normal range.
        codegen.Emit("RegTensor<uint32_t> " + expbits + ";");
        codegen.Emit("RegTensor<uint32_t> " + scratch + ";");
        codegen.Emit("RegTensor<int32_t> " + expo + ";");
        codegen.Emit("RegTensor<int32_t> " + kreg + ";");
        codegen.Emit("RegTensor<int32_t> " + thrvec + ";");
        codegen.Emit("RegTensor<int32_t> " + zerovec + ";");
        codegen.Emit("RegTensor<int32_t> " + newexp + ";");
        codegen.Emit("RegTensor<uint32_t> " + scalebits + ";");
        codegen.Emit("RegTensor<float> " + scale + ";");
        codegen.Emit("RegTensor<float> " + one + ";");
        codegen.Emit("RegTensor<float> " + asc + ";");
        codegen.Emit("RegTensor<float> " + bsc + ";");
        codegen.Emit("RegTensor<float> " + y + ";");
        codegen.Emit("MaskReg " + m_scale + ";");
        codegen.Emit("vdup(" + scratch + ", (int32_t)0x7F800000, " + mask + ", " + mode + ");");
        codegen.Emit("vand(" + expbits + ", (RegTensor<uint32_t>&)" + src0 + ", " + scratch + ", " + mask + ", " +
                     mode + ");");
        codegen.Emit("vshrs(" + expbits + ", " + expbits + ", (int16_t)23, " + mask + ", " + mode + ");");
        codegen.Emit("vdup(" + scratch + ", (int32_t)127, " + mask + ", " + mode + ");");
        codegen.Emit("vsub(" + expo + ", (RegTensor<int32_t>&)" + expbits + ", (RegTensor<int32_t>&)" + scratch + ", " +
                     mask + ", " + mode + ");");
        codegen.Emit("vcmps_lt(" + m_scale + ", " + expo + ", (int32_t)-64, " + mask + ");");
        codegen.Emit("vdup(" + thrvec + ", (int32_t)-64, " + mask + ", " + mode + ");");
        codegen.Emit("vdup(" + zerovec + ", (int32_t)0, " + mask + ", " + mode + ");");
        codegen.Emit("vsub(" + kreg + ", " + thrvec + ", " + expo + ", " + mask + ", " + mode + ");");
        codegen.Emit("vmax(" + kreg + ", " + kreg + ", " + zerovec + ", " + mask + ", " + mode + ");");
        codegen.Emit("vadds(" + newexp + ", " + kreg + ", (int32_t)127, " + m_scale + ", " + mode + ");");
        codegen.Emit("vshls(" + scalebits + ", (RegTensor<uint32_t>&)" + newexp + ", (int16_t)23, " + m_scale + ", " +
                     mode + ");");
        codegen.Emit("vdup(" + one + ", 1.0f, " + mask + ", " + mode + ");");
        codegen.Emit("vsel(" + scale + ", (RegTensor<float>&)" + scalebits + ", " + one + ", " + m_scale + ");");
        codegen.Emit("vmul(" + asc + ", " + src0 + ", " + scale + ", " + mask + ", " + mode + ");");
        codegen.Emit("vmul(" + bsc + ", " + src1 + ", " + scale + ", " + mask + ", " + mode + ");");
        // Corrected quotient (DivPrecisionImpl:952-975): r = a' - x1*b', pick the
        // bit-pattern neighbor of x1 (-1/+1 ulp) with the smaller |residual|, then
        // bypass invalid lanes back to the raw quotient.
        codegen.Emit("vmuls(" + y + ", " + bsc + ", -1.0f, " + mask + ", " + mode + ");");
        codegen.Emit("RegTensor<float> " + r + ";");
        codegen.Emit("RegTensor<float> " + rpre + ";");
        codegen.Emit("RegTensor<float> " + rnext + ";");
        codegen.Emit("RegTensor<float> " + zpre + ";");
        codegen.Emit("RegTensor<float> " + znext + ";");
        codegen.Emit("MaskReg " + m_cmp + ";");
        codegen.Emit("vmov(" + r + ", " + asc + ");");
        codegen.Emit("vmula(" + r + ", " + z + ", " + y + ", " + mask + ", " + mode + ");");
        codegen.Emit("vadds((RegTensor<int32_t>&)" + zpre + ", (RegTensor<int32_t>&)" + z + ", (int32_t)-1, " + mask +
                     ", " + mode + ");");
        codegen.Emit("vadds((RegTensor<int32_t>&)" + znext + ", (RegTensor<int32_t>&)" + z + ", (int32_t)1, " + mask +
                     ", " + mode + ");");
        codegen.Emit("vmov(" + rpre + ", " + asc + ");");
        codegen.Emit("vmov(" + rnext + ", " + asc + ");");
        codegen.Emit("vmula(" + rpre + ", " + zpre + ", " + y + ", " + mask + ", " + mode + ");");
        codegen.Emit("vmula(" + rnext + ", " + znext + ", " + y + ", " + mask + ", " + mode + ");");
        codegen.Emit("vabs(" + r + ", " + r + ", " + mask + ", " + mode + ");");
        codegen.Emit("vabs(" + rpre + ", " + rpre + ", " + mask + ", " + mode + ");");
        codegen.Emit("vabs(" + rnext + ", " + rnext + ", " + mask + ", " + mode + ");");
        codegen.Emit("vcmp_lt(" + m_cmp + ", " + r + ", " + rpre + ", " + mask + ");");
        codegen.Emit("vsel(" + r + ", " + r + ", " + rpre + ", " + m_cmp + ");");
        codegen.Emit("vsel(" + z + ", " + z + ", " + zpre + ", " + m_cmp + ");");
        codegen.Emit("vcmp_lt(" + m_cmp + ", " + rnext + ", " + r + ", " + mask + ");");
        codegen.Emit("vsel(" + z + ", " + znext + ", " + z + ", " + m_cmp + ");");
        codegen.Emit("vsel(" + dst + ", " + q0 + ", " + z + ", " + m_inf + ");");
    } else {
        codegen.Emit("vdiv(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// Muls — vmuls
// ============================================================================

static std::string EmitVFMuls(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // Parser args order: [dst, src, scalar, mask]
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.muls requires 4 args (dst, src, scalar, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (src_dt == DataType::INT16 || src_dt == DataType::UINT16 || src_dt == DataType::INT32 ||
                       src_dt == DataType::UINT32 || src_dt == DataType::INT64 || src_dt == DataType::UINT64 ||
                       src_dt == DataType::FP16 || src_dt == DataType::FP32))
        << "vf.muls src only supports INT16/UINT16/INT32/UINT32/INT64/UINT64/FP16/FP32, got " << DTypeStr(src_dt);
    DataType scalar_dt = GetExprDtype(op->args_[2]);
    if (scalar_dt == DataType::INDEX || scalar_dt == DataType::INT64) {
        scalar_dt = src_dt;
    }
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (scalar_dt == DataType::INT16 || scalar_dt == DataType::UINT16 || scalar_dt == DataType::INT32 ||
                       scalar_dt == DataType::UINT32 || scalar_dt == DataType::INT64 || scalar_dt == DataType::UINT64 ||
                       scalar_dt == DataType::FP16 || scalar_dt == DataType::FP32))
        << "vf.muls scalar only supports INT16/UINT16/INT32/UINT32/INT64/UINT64/FP16/FP32, got " << DTypeStr(scalar_dt);
    DataType vf_muls_dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == vf_muls_dst_dt)
        << "vf.muls requires src and dst to have the same type, got dst=" << DTypeStr(vf_muls_dst_dt)
        << " src=" << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string scalar_str = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.muls");
    scalar_str = CoerceScalarToInt(op->args_[2], src_dt, scalar_str);
    if (src_dt.GetBit() == 64) {
        // B64 Muls: dst = src * scalar. Mirrors AscendC MulsImpl: Duplicate(scalar) + Mul(src, scalar_reg).
        std::string p = dst + "_muls_";
        std::string cast_type = (src_dt == DataType::INT64) ? "(int64_t)" : "(uint64_t)";
        // AscendC MulB64Impl: vmull always uses uint32_t (unsigned 32x32→64);
        // vmula uses int32_t for int64_t (signed cross-term), uint32_t for uint64_t.
        std::string vmull_cast = "(RegTensor<uint32_t>&)";
        std::string vmula_cast = (src_dt == DataType::INT64) ? "(RegTensor<int32_t>&)" : "(RegTensor<uint32_t>&)";
        // Pack b64 mask to b32 (mirrors MaskPack in CalTraitOneByTransToTraitTwo)
        std::string packed_m = p + "_pm_";
        codegen.Emit("MaskReg " + packed_m + ";");
        codegen.Emit("ppack(" + packed_m + ", " + mask + ", LOWER);");
        // Broadcast scalar to b32 halves (mirrors DuplicateB64Impl)
        std::string all_m = p + "_allm_";
        codegen.Emit("MaskReg " + all_m + " = pset_b32(PAT_ALL);");
        std::string lo_sc = p + "_losc_";
        std::string hi_sc = p + "_hisc_";
        codegen.Emit("RegTensor<uint32_t> " + lo_sc + ";");
        codegen.Emit("RegTensor<uint32_t> " + hi_sc + ";");
        codegen.Emit("vdup(" + lo_sc + ", (int32_t)(" + cast_type + "(" + scalar_str + ")), " + all_m +
                     ", MODE_ZEROING);");
        codegen.Emit("vdup(" + hi_sc + ", (int32_t)((" + cast_type + "(" + scalar_str + ")) >> 32), " + all_m +
                     ", MODE_ZEROING);");
        // Deinterleave src to b32 halves (mirrors B64TraitOneToTraitTwo)
        EmitB64Deinterleave(codegen, p + "s", src);
        std::string lo_s = p + "s_lo_", hi_s = p + "s_hi_";
        // MulB64: vmull (lo_s × lo_sc → dst_lo, dst_hi) + vmula (lo_s × hi_sc → dst_hi) + vmula (hi_s × lo_sc → dst_hi)
        std::string lo_d = p + "_lod_";
        std::string hi_d = p + "_hid_";
        codegen.Emit("RegTensor<uint32_t> " + lo_d + ";");
        codegen.Emit("RegTensor<uint32_t> " + hi_d + ";");
        codegen.Emit("vmull(" + vmull_cast + lo_d + ", " + vmull_cast + hi_d + ", " + vmull_cast + lo_s + ", " +
                     vmull_cast + lo_sc + ", " + packed_m + ");");
        codegen.Emit("vmula(" + vmula_cast + hi_d + ", " + vmula_cast + lo_s + ", " + vmula_cast + hi_sc + ", " +
                     packed_m + ", MODE_ZEROING);");
        codegen.Emit("vmula(" + vmula_cast + hi_d + ", " + vmula_cast + hi_s + ", " + vmula_cast + lo_sc + ", " +
                     packed_m + ", MODE_ZEROING);");
        // Interleave back (mirrors B64TraitTwoToTraitOne)
        EmitB64Interleave(codegen, dst, lo_d, hi_d, p + "_ilv");
    } else {
        codegen.Emit("vmuls(" + dst + ", " + src + ", " + scalar_str + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// Ln — vln (natural logarithm, basic precision)
// ============================================================================

static std::string EmitVFUnary(const ir::CallPtr& op, codegen::CodegenBase& codegen_base, const std::string& op_name,
                               const std::string& instruction)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << op_name << " requires 3 args (dst, src, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string mode = VFZeroingOnly(op, op_name);
    codegen.Emit(instruction + "(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    return "";
}

static std::string EmitVFLn(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    DataType src_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (src_dt == DataType::FP16 || src_dt == DataType::FP32))
        << "vf.ln src only supports FP16/FP32, got " << DTypeStr(src_dt);
    DataType vf_ln_dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == vf_ln_dst_dt)
        << "vf.ln requires src and dst to have the same type, got dst=" << DTypeStr(vf_ln_dst_dt)
        << " src=" << DTypeStr(src_dt);
    if (op->HasKwarg("precision") && op->GetKwarg<bool>("precision")) {
        auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
        std::string dst = codegen.GetExprAsCode(op->args_[0]);
        std::string src = codegen.GetExprAsCode(op->args_[1]);
        std::string mask = codegen.GetExprAsCode(op->args_[2]);
        std::string mode = VFZeroingOnly(op, "vf.ln");
        std::string mc = dst + "_mcmp_";
        std::string tp = dst + "_tmp_";
        std::string sc = dst + "_srcc_";
        std::string dc = dst + "_dstc_";
        std::string ctype = src_dt.ToCTypeString();
        codegen.Emit("MaskReg " + mc + ";");
        codegen.Emit("RegTensor<" + ctype + "> " + tp + ";");
        codegen.Emit("RegTensor<" + ctype + "> " + sc + " = (RegTensor<" + ctype + ">&)" + src + ";");
        codegen.Emit("RegTensor<" + ctype + "> " + dc + ";");
        if (src_dt == DataType::FP16) {
            codegen.Emit("union { uint16_t i; half f; } " + dst + "_thr_ = {0x03FF};");
            codegen.Emit("vcmps_lt(" + mc + ", " + sc + ", " + dst + "_thr_.f, " + mask + ");");
            codegen.Emit("vmuls(" + tp + ", " + sc + ", 1024.0f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + sc + ", " + tp + ", " + sc + ", " + mc + ");");
            codegen.Emit("vln(" + dc + ", " + sc + ", " + mask + ", " + mode + ");");
            codegen.Emit("vadds(" + tp + ", " + dc + ", -6.931471805599453094172f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + dst + ", " + tp + ", " + dc + ", " + mc + ");");
        } else {
            codegen.Emit("union { uint32_t i; float f; } " + dst + "_thr_ = {0x007FFFFF};");
            codegen.Emit("vcmps_lt(" + mc + ", " + sc + ", " + dst + "_thr_.f, " + mask + ");");
            codegen.Emit("vmuls(" + tp + ", " + sc + ", 8388608.0f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + sc + ", " + tp + ", " + sc + ", " + mc + ");");
            codegen.Emit("vln(" + dc + ", " + sc + ", " + mask + ", " + mode + ");");
            codegen.Emit("vadds(" + tp + ", " + dc + ", -15.9423851528787421f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + dst + ", " + tp + ", " + dc + ", " + mc + ");");
        }
        return "";
    }
    return EmitVFUnary(op, codegen_base, "vf.ln", "vln");
}

// ============================================================================
// Log — vln (natural logarithm, same as Ln on A5/dav_3510)
// ============================================================================

static std::string EmitVFLog(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    DataType src_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (src_dt == DataType::FP16 || src_dt == DataType::FP32))
        << "vf.log src only supports FP16/FP32, got " << DTypeStr(src_dt);
    DataType vf_log_dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == vf_log_dst_dt)
        << "vf.log requires src and dst to have the same type, got dst=" << DTypeStr(vf_log_dst_dt)
        << " src=" << DTypeStr(src_dt);
    if (op->HasKwarg("precision") && op->GetKwarg<bool>("precision")) {
        auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
        std::string dst = codegen.GetExprAsCode(op->args_[0]);
        std::string src = codegen.GetExprAsCode(op->args_[1]);
        std::string mask = codegen.GetExprAsCode(op->args_[2]);
        std::string mode = VFZeroingOnly(op, "vf.log");
        std::string mc = dst + "_mcmp_";
        std::string tp = dst + "_tmp_";
        std::string sc = dst + "_srcc_";
        std::string dc = dst + "_dstc_";
        std::string ctype = src_dt.ToCTypeString();
        codegen.Emit("MaskReg " + mc + ";");
        codegen.Emit("RegTensor<" + ctype + "> " + tp + ";");
        codegen.Emit("RegTensor<" + ctype + "> " + sc + " = (RegTensor<" + ctype + ">&)" + src + ";");
        codegen.Emit("RegTensor<" + ctype + "> " + dc + ";");
        if (src_dt == DataType::FP16) {
            codegen.Emit("union { uint16_t i; half f; } " + dst + "_thr_ = {0x03FF};");
            codegen.Emit("vcmps_lt(" + mc + ", " + sc + ", " + dst + "_thr_.f, " + mask + ");");
            codegen.Emit("vmuls(" + tp + ", " + sc + ", 1024.0f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + sc + ", " + tp + ", " + sc + ", " + mc + ");");
            codegen.Emit("vln(" + dc + ", " + sc + ", " + mask + ", " + mode + ");");
            codegen.Emit("vadds(" + tp + ", " + dc + ", -6.931471805599453094172f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + dst + ", " + tp + ", " + dc + ", " + mc + ");");
        } else {
            codegen.Emit("union { uint32_t i; float f; } " + dst + "_thr_ = {0x007FFFFF};");
            codegen.Emit("vcmps_lt(" + mc + ", " + sc + ", " + dst + "_thr_.f, " + mask + ");");
            codegen.Emit("vmuls(" + tp + ", " + sc + ", 8388608.0f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + sc + ", " + tp + ", " + sc + ", " + mc + ");");
            codegen.Emit("vln(" + dc + ", " + sc + ", " + mask + ", " + mode + ");");
            codegen.Emit("vadds(" + tp + ", " + dc + ", -15.9423851528787421f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + dst + ", " + tp + ", " + dc + ", " + mc + ");");
        }
        return "";
    }
    return EmitVFUnary(op, codegen_base, "vf.log", "vln");
}

// ============================================================================
// Min — vmin
// ============================================================================

static std::string EmitVFMin(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.min requires 4 args (dst, src0, src1, mask)";
    DataType s0_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (IsArithIntType(s0_dt) || s0_dt == DataType::FP16 ||
                                                    s0_dt == DataType::FP32 || s0_dt == DataType::BF16))
        << "vf.min src only supports supported types, got " << DTypeStr(s0_dt);
    DataType s1_dt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (IsArithIntType(s1_dt) || s1_dt == DataType::FP16 ||
                                                    s1_dt == DataType::FP32 || s1_dt == DataType::BF16))
        << "vf.min src only supports supported types, got " << DTypeStr(s1_dt);
    DataType vf_min_dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, s0_dt == vf_min_dst_dt && s1_dt == vf_min_dst_dt)
        << "vf.min requires dst, src0, src1 to have the same type, got dst=" << DTypeStr(vf_min_dst_dt)
        << " src0=" << DTypeStr(s0_dt) << " src1=" << DTypeStr(s1_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.min");
    if (s0_dt.GetBit() == 64) {
        // B64 min via deinterleave + b32 compare + vsel + interleave.
        // min = src1 when src0 > src1, else src0 (i.e., pick the smaller).
        std::string p = dst + "_min_";
        EmitB64Deinterleave(codegen, p + "s0", src0);
        EmitB64Deinterleave(codegen, p + "s1", src1);
        std::string lo0 = p + "s0_lo_", hi0 = p + "s0_hi_";
        std::string lo1 = p + "s1_lo_", hi1 = p + "s1_hi_";
        std::string all_m = p + "_allm_";
        codegen.Emit("MaskReg " + all_m + " = pset_b32(PAT_ALL);");
        std::string hi_gt = p + "_higt_";
        std::string hi_eq = p + "_hieq_";
        std::string lo_gt = p + "_logt_";
        codegen.Emit("MaskReg " + hi_gt + ";");
        codegen.Emit("MaskReg " + hi_eq + ";");
        codegen.Emit("MaskReg " + lo_gt + ";");
        if (s0_dt == DataType::INT64) {
            codegen.Emit("vcmp_gt(" + hi_gt + ", (RegTensor<int32_t>&)" + hi0 + ", (RegTensor<int32_t>&)" + hi1 + ", " +
                         all_m + ");");
            codegen.Emit("vcmp_eq(" + hi_eq + ", (RegTensor<int32_t>&)" + hi0 + ", (RegTensor<int32_t>&)" + hi1 + ", " +
                         all_m + ");");
        } else {
            codegen.Emit("vcmp_gt(" + hi_gt + ", " + hi0 + ", " + hi1 + ", " + all_m + ");");
            codegen.Emit("vcmp_eq(" + hi_eq + ", " + hi0 + ", " + hi1 + ", " + all_m + ");");
        }
        codegen.Emit("vcmp_gt(" + lo_gt + ", " + lo0 + ", " + lo1 + ", " + all_m + ");");
        std::string eq_lo = p + "_eqlo_";
        std::string s0_gt = p + "_s0gt_";
        codegen.Emit("MaskReg " + eq_lo + ";");
        codegen.Emit("MaskReg " + s0_gt + ";");
        codegen.Emit("pand(" + eq_lo + ", " + hi_eq + ", " + lo_gt + ", " + all_m + ");");
        codegen.Emit("por(" + s0_gt + ", " + hi_gt + ", " + eq_lo + ", " + all_m + ");");
        // Select: min = src0 > src1 ? src1 : src0 (pick smaller)
        std::string lo_dst = p + "_lod_";
        std::string hi_dst = p + "_hid_";
        codegen.Emit("RegTensor<uint32_t> " + lo_dst + ";");
        codegen.Emit("RegTensor<uint32_t> " + hi_dst + ";");
        codegen.Emit("vsel(" + lo_dst + ", " + lo1 + ", " + lo0 + ", " + s0_gt + ");");
        codegen.Emit("vsel(" + hi_dst + ", " + hi1 + ", " + hi0 + ", " + s0_gt + ");");
        EmitB64Interleave(codegen, dst, lo_dst, hi_dst, p + "_ilv");
        EmitB64Zeroing(codegen, dst, dst, mask, p + "_zero");
    } else {
        codegen.Emit("vmin(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// Exp — vexp
// ============================================================================

static std::string EmitVFExp(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "vf.exp requires 3 args (dst, src, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (src_dt == DataType::FP16 || src_dt == DataType::FP32))
        << "vf.exp src only supports FP16/FP32, got " << DTypeStr(src_dt);
    DataType vf_exp_dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == vf_exp_dst_dt)
        << "vf.exp requires src and dst to have the same type, got dst=" << DTypeStr(vf_exp_dst_dt)
        << " src=" << DTypeStr(src_dt);
    DataType exp_dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == exp_dst_dt)
        << "vf.exp requires src and dst to have the same type, got dst=" << DTypeStr(exp_dst_dt)
        << " src=" << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string mode = VFZeroingOnly(op, "vf.exp");
    if (op->HasKwarg("precision") && op->GetKwarg<bool>("precision")) {
        std::string ms = dst + "_sub_";
        std::string tw = dst + "_two_";
        std::string t0 = dst + "_t0_";
        std::string t1 = dst + "_t1_";
        std::string ctype = src_dt.ToCTypeString();
        codegen.Emit("MaskReg " + ms + ";");
        codegen.Emit("RegTensor<" + ctype + "> " + tw + ";");
        codegen.Emit("RegTensor<" + ctype + "> " + t0 + ";");
        codegen.Emit("RegTensor<" + ctype + "> " + t1 + ";");
        if (src_dt == DataType::FP16) {
            codegen.Emit("union { uint16_t i; half f; } " + dst + "_thr_ = {0x03FF};");
        } else {
            codegen.Emit("union { uint32_t i; float f; } " + dst + "_thr_ = {0x007FFFFF};");
        }
        codegen.Emit("vexp(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
        codegen.Emit("vcmps_le(" + ms + ", " + dst + ", " + dst + "_thr_.f, " + mask + ");");
        codegen.Emit("vdup(" + tw + ", 2.0f, " + ms + ", " + mode + ");");
        codegen.Emit("vdiv(" + t0 + ", " + src + ", " + tw + ", " + ms + ", " + mode + ");");
        codegen.Emit("vexp(" + t0 + ", " + t0 + ", " + ms + ", " + mode + ");");
        codegen.Emit("vmul(" + t1 + ", " + t0 + ", " + t0 + ", " + ms + ", " + mode + ");");
        codegen.Emit("vsel(" + dst + ", " + t1 + ", " + dst + ", " + ms + ");");
    } else {
        codegen.Emit("vexp(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// Abs — vabs
// ============================================================================

static std::string EmitVFAbs(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "vf.abs requires 3 args (dst, src, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (IsArithSignedIntType(src_dt) || src_dt == DataType::FP16 || src_dt == DataType::FP32))
        << "vf.abs src only supports INT8/INT16/INT32/INT64/FP16/FP32, got " << DTypeStr(src_dt);
    DataType vf_abs_dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == vf_abs_dst_dt)
        << "vf.abs requires src and dst to have the same type, got dst=" << DTypeStr(vf_abs_dst_dt)
        << " src=" << DTypeStr(src_dt);
    DataType abs_dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == abs_dst_dt)
        << "vf.abs requires src and dst to have the same type, got dst=" << DTypeStr(abs_dst_dt)
        << " src=" << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string mode = VFZeroingOnly(op, "vf.abs");
    if (src_dt == DataType::INT64) {
        // B64 abs: abs(x) = x if x >= 0, -x if x < 0.
        // Mirrors AscendC AbsB64Impl: vbr(0) + vcmp_lt + Sub(vsubc) + SubC(vsubcs) + vsel.
        std::string p = dst + "_abs_";
        // Deinterleave src to get b32 low/high halves
        EmitB64Deinterleave(codegen, p + "s", src);
        std::string lo_s = p + "s_lo_", hi_s = p + "s_hi_";
        std::string all_m = p + "_allm_";
        codegen.Emit("MaskReg " + all_m + " = pset_b32(PAT_ALL);");
        // Zero register (vbr equivalent)
        std::string zero_reg = p + "_zero_";
        codegen.Emit("RegTensor<int32_t> " + zero_reg + ";");
        codegen.Emit("vdup(" + zero_reg + ", 0, " + all_m + ", MODE_ZEROING);");
        // Sign mask: hi_s < 0 (signed)
        std::string sign_m = p + "_sign_";
        codegen.Emit("MaskReg " + sign_m + ";");
        codegen.Emit("vcmp_lt(" + sign_m + ", (RegTensor<int32_t>&)" + hi_s + ", (RegTensor<int32_t>&)" + zero_reg +
                     ", " + all_m + ");");
        // Negate via borrow-chain sub: neg = 0 - src
        // vsubc: borrow = (0 < lo_s), neg_lo = 0 - lo_s
        // vsubcs: neg_hi = 0 - hi_s - borrow
        std::string borrow = p + "_borrow_";
        std::string lo_n = p + "_nlo_";
        std::string hi_n = p + "_nhi_";
        codegen.Emit("MaskReg " + borrow + ";");
        codegen.Emit("RegTensor<int32_t> " + lo_n + ";");
        codegen.Emit("RegTensor<int32_t> " + hi_n + ";");
        codegen.Emit("vsubc(" + borrow + ", " + lo_n + ", " + zero_reg + ", (RegTensor<int32_t>&)" + lo_s + ", " +
                     sign_m + ");");
        codegen.Emit("vsubcs(" + borrow + ", " + hi_n + ", " + zero_reg + ", (RegTensor<int32_t>&)" + hi_s + ", " +
                     borrow + ", " + sign_m + ");");
        // Select: if negative -> neg, else -> src (per b32 half)
        // All vsel operands must be the same type (int32_t).
        std::string lo_dst = p + "_lod_";
        std::string hi_dst = p + "_hid_";
        codegen.Emit("RegTensor<int32_t> " + lo_dst + ";");
        codegen.Emit("RegTensor<int32_t> " + hi_dst + ";");
        codegen.Emit("vsel(" + lo_dst + ", " + lo_n + ", (RegTensor<int32_t>&)" + lo_s + ", " + sign_m + ");");
        codegen.Emit("vsel(" + hi_dst + ", " + hi_n + ", (RegTensor<int32_t>&)" + hi_s + ", " + sign_m + ");");
        EmitB64Interleave(codegen, dst, lo_dst, hi_dst, p + "_ilv");
        EmitB64Zeroing(codegen, dst, dst, mask, p + "_zero");
    } else {
        codegen.Emit("vabs(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// Not — vnot
// ============================================================================

static std::string EmitVFNot(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "vf.not_ requires 3 args (dst, src, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      src_dt == DataType::INT8 || src_dt == DataType::UINT8 || src_dt == DataType::BOOL ||
                          src_dt == DataType::INT16 || src_dt == DataType::UINT16 || src_dt == DataType::INT32 ||
                          src_dt == DataType::UINT32 || src_dt == DataType::FP16 || src_dt == DataType::FP32 ||
                          src_dt == DataType::INT64 || src_dt == DataType::UINT64)
        << "vf.not_ src only supports INT8/UINT8/INT16/UINT16/INT32/UINT32/FP16/FP32/INT64/UINT64, got "
        << DTypeStr(src_dt);
    DataType not_dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == not_dst_dt)
        << "vf.not_ requires src and dst to have the same type, got dst=" << DTypeStr(not_dst_dt)
        << " src=" << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    if (IsDstMaskReg(op, codegen)) {
        codegen.Emit("pnot(" + dst + ", " + src + ", " + mask + ");");
        return "";
    }
    std::string mode = VFZeroingOnly(op, "vf.not_");
    if (src_dt.GetBit() == 64) {
        // B64 bitwise via b32 vnot (mirrors AscendC NotImpl b64 path, see EmitB64Bitwise).
        EmitB64Bitwise(codegen, "vnot", {src}, dst, mask, mode, dst + "_not_");
    } else {
        codegen.Emit("vnot(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// Sqrt — vsqrt
// ============================================================================

static std::string EmitVFSqrt(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "vf.sqrt requires 3 args (dst, src, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (src_dt == DataType::FP16 || src_dt == DataType::FP32))
        << "vf.sqrt src only supports FP16/FP32, got " << DTypeStr(src_dt);
    DataType vf_sqrt_dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == vf_sqrt_dst_dt)
        << "vf.sqrt requires src and dst to have the same type, got dst=" << DTypeStr(vf_sqrt_dst_dt)
        << " src=" << DTypeStr(src_dt);
    DataType sqrt_dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == sqrt_dst_dt)
        << "vf.sqrt requires src and dst to have the same type, got dst=" << DTypeStr(sqrt_dst_dt)
        << " src=" << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string mode = VFZeroingOnly(op, "vf.sqrt");
    if (op->HasKwarg("precision") && op->GetKwarg<bool>("precision")) {
        std::string mc = dst + "_mcmp_";
        std::string tp = dst + "_tmp_";
        std::string sc = dst + "_srcc_";
        std::string dc = dst + "_dstc_";
        std::string ctype = src_dt.ToCTypeString();
        codegen.Emit("MaskReg " + mc + ";");
        codegen.Emit("RegTensor<" + ctype + "> " + tp + ";");
        codegen.Emit("RegTensor<" + ctype + "> " + sc + " = (RegTensor<" + ctype + ">&)" + src + ";");
        codegen.Emit("RegTensor<" + ctype + "> " + dc + ";");
        if (src_dt == DataType::FP16) {
            codegen.Emit("union { uint16_t i; half f; } " + dst + "_thr_ = {0x03FF};");
            codegen.Emit("vcmps_lt(" + mc + ", " + sc + ", " + dst + "_thr_.f, " + mask + ");");
            codegen.Emit("vmuls(" + tp + ", " + sc + ", 4096.0f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + sc + ", " + tp + ", " + sc + ", " + mc + ");");
            codegen.Emit("vsqrt(" + dc + ", " + sc + ", " + mask + ", " + mode + ");");
            // AscendC SqrtImpl PRECISION_1ULP_FTZ_FALSE (half): multiplyFactor1 = 0x2400 = 2^-6.
            // sqrt(2^12) = 2^6, so the scale-down compensation is 2^-6 — NOT the FP32
            // branch's 2^-12, which made subnormal results come out 1/64 too small.
            codegen.Emit("vmuls(" + tp + ", " + dc + ", 0.015625f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + dst + ", " + tp + ", " + dc + ", " + mc + ");");
        } else {
            codegen.Emit("union { uint32_t i; float f; } " + dst + "_thr_ = {0x007FFFFF};");
            codegen.Emit("vcmps_lt(" + mc + ", " + sc + ", " + dst + "_thr_.f, " + mask + ");");
            codegen.Emit("vmuls(" + tp + ", " + sc + ", 16777216.0f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + sc + ", " + tp + ", " + sc + ", " + mc + ");");
            codegen.Emit("vsqrt(" + dc + ", " + sc + ", " + mask + ", " + mode + ");");
            codegen.Emit("vmuls(" + tp + ", " + dc + ", 0.000244140625f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + dst + ", " + tp + ", " + dc + ", " + mc + ");");
        }
    } else {
        codegen.Emit("vsqrt(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// Relu — vrelu
// ============================================================================

static std::string EmitVFRelu(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "vf.relu requires 3 args (dst, src, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (src_dt == DataType::INT32 || src_dt == DataType::INT64 ||
                                                    src_dt == DataType::FP16 || src_dt == DataType::FP32))
        << "vf.relu src only supports INT32/INT64/FP16/FP32, got " << DTypeStr(src_dt);
    DataType vf_relu_dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == vf_relu_dst_dt)
        << "vf.relu requires src and dst to have the same type, got dst=" << DTypeStr(vf_relu_dst_dt)
        << " src=" << DTypeStr(src_dt);
    DataType relu_dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == relu_dst_dt)
        << "vf.relu requires src and dst to have the same type, got dst=" << DTypeStr(relu_dst_dt)
        << " src=" << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string mode = VFZeroingOnly(op, "vf.relu");
    if (src_dt == DataType::INT64) {
        // B64 relu (mirrors AscendC ReluB64Impl: Maxs(src, 0)): deinterleave +
        // scalar(0) two-stage compare + vsel + interleave. No native b64 vrelu.
        std::string p = dst + "_relu_";
        std::string packed_m = p + "_pm_";
        codegen.Emit("MaskReg " + packed_m + ";");
        codegen.Emit("ppack(" + packed_m + ", " + mask + ", LOWER);");
        EmitB64Deinterleave(codegen, p + "s", src);
        std::string lo_s = p + "s_lo_", hi_s = p + "s_hi_";
        std::string all_m = p + "_allm_";
        codegen.Emit("MaskReg " + all_m + " = pset_b32(PAT_ALL);");
        std::string lo_sc = p + "_losc_", hi_sc = p + "_hisc_";
        codegen.Emit("RegTensor<uint32_t> " + lo_sc + ";");
        codegen.Emit("RegTensor<uint32_t> " + hi_sc + ";");
        codegen.Emit("vdup(" + lo_sc + ", (int32_t)0, " + all_m + ", MODE_ZEROING);");
        codegen.Emit("vdup(" + hi_sc + ", (int32_t)0, " + all_m + ", MODE_ZEROING);");
        // src > 0 iff hi > 0 (signed) OR (hi == 0 AND lo > 0, unsigned).
        std::string hi_gt = p + "_higt_", hi_eq = p + "_hieq_", lo_gt = p + "_logt_";
        std::string eq_lo = p + "_eqlo_", gt = p + "_gt_";
        codegen.Emit("MaskReg " + hi_gt + ";");
        codegen.Emit("MaskReg " + hi_eq + ";");
        codegen.Emit("MaskReg " + lo_gt + ";");
        codegen.Emit("MaskReg " + eq_lo + ";");
        codegen.Emit("MaskReg " + gt + ";");
        codegen.Emit("vcmp_gt(" + hi_gt + ", (RegTensor<int32_t>&)" + hi_s + ", (RegTensor<int32_t>&)" + hi_sc + ", " +
                     packed_m + ");");
        codegen.Emit("vcmp_eq(" + hi_eq + ", (RegTensor<int32_t>&)" + hi_s + ", (RegTensor<int32_t>&)" + hi_sc + ", " +
                     packed_m + ");");
        codegen.Emit("vcmp_gt(" + lo_gt + ", " + lo_s + ", " + lo_sc + ", " + packed_m + ");");
        codegen.Emit("pand(" + eq_lo + ", " + hi_eq + ", " + lo_gt + ", " + packed_m + ");");
        codegen.Emit("por(" + gt + ", " + hi_gt + ", " + eq_lo + ", " + packed_m + ");");
        std::string lo_d = p + "_lod_", hi_d = p + "_hid_";
        codegen.Emit("RegTensor<uint32_t> " + lo_d + ";");
        codegen.Emit("RegTensor<uint32_t> " + hi_d + ";");
        codegen.Emit("vsel(" + lo_d + ", " + lo_s + ", " + lo_sc + ", " + gt + ");");
        codegen.Emit("vsel(" + hi_d + ", " + hi_s + ", " + hi_sc + ", " + gt + ");");
        EmitB64Interleave(codegen, dst, lo_d, hi_d, p + "_ilv");
        EmitB64Zeroing(codegen, dst, dst, mask, p + "_zero");
        return "";
    }
    codegen.Emit("vrelu(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Neg — vneg
// ============================================================================

static std::string EmitVFNeg(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "vf.neg requires 3 args (dst, src, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (IsArithSignedIntType(src_dt) || src_dt == DataType::FP16 || src_dt == DataType::FP32))
        << "vf.neg src only supports INT8/INT16/INT32/INT64/FP16/FP32, got " << DTypeStr(src_dt);
    DataType vf_neg_dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == vf_neg_dst_dt)
        << "vf.neg requires src and dst to have the same type, got dst=" << DTypeStr(vf_neg_dst_dt)
        << " src=" << DTypeStr(src_dt);
    DataType neg_dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == neg_dst_dt)
        << "vf.neg requires src and dst to have the same type, got dst=" << DTypeStr(neg_dst_dt)
        << " src=" << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string mode = VFZeroingOnly(op, "vf.neg");
    if (src_dt == DataType::INT64 || src_dt == DataType::UINT64) {
        // B64 neg: dst = 0 - src. Mirrors AscendC NegB64Impl: Duplicate(0) + Sub(0, src).
        // Sub for b64 uses vsubc + vsubcs borrow-chain (mirrors SubB64Impl).
        std::string p = dst + "_neg_";
        std::string b32_cast = (src_dt == DataType::INT64) ? "(RegTensor<int32_t>&)" : "(RegTensor<uint32_t>&)";
        // Pack b64 mask to b32 (mirrors MaskPack in CalTraitOneByTransToTraitTwo)
        std::string packed_m = p + "_pm_";
        codegen.Emit("MaskReg " + packed_m + ";");
        codegen.Emit("ppack(" + packed_m + ", " + mask + ", LOWER);");
        // Zero register (vbr equivalent)
        std::string all_m = p + "_allm_";
        codegen.Emit("MaskReg " + all_m + " = pset_b32(PAT_ALL);");
        std::string zero_reg = p + "_zero_";
        codegen.Emit("RegTensor<int32_t> " + zero_reg + ";");
        codegen.Emit("vdup(" + zero_reg + ", 0, " + all_m + ", MODE_ZEROING);");
        // Deinterleave src to b32 halves (mirrors B64TraitOneToTraitTwo)
        EmitB64Deinterleave(codegen, p + "s", src);
        std::string lo_s = p + "s_lo_", hi_s = p + "s_hi_";
        // SubB64: vsubc + vsubcs (borrow-chain sub: 0 - src)
        std::string borrow = p + "_borrow_";
        std::string lo_d = p + "_lod_";
        std::string hi_d = p + "_hid_";
        codegen.Emit("MaskReg " + borrow + ";");
        codegen.Emit("RegTensor<uint32_t> " + lo_d + ";");
        codegen.Emit("RegTensor<uint32_t> " + hi_d + ";");
        codegen.Emit("vsubc(" + borrow + ", " + b32_cast + lo_d + ", " + b32_cast + zero_reg + ", " + b32_cast + lo_s +
                     ", " + packed_m + ");");
        codegen.Emit("vsubcs(" + borrow + ", " + b32_cast + hi_d + ", " + b32_cast + zero_reg + ", " + b32_cast + hi_s +
                     ", " + borrow + ", " + packed_m + ");");
        // Interleave back (mirrors B64TraitTwoToTraitOne)
        EmitB64Interleave(codegen, dst, lo_d, hi_d, p + "_ilv");
    } else {
        codegen.Emit("vneg(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// Adds — vadds (scalar addition)
// ============================================================================

static std::string EmitVFAdds(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.adds requires 4 args (dst, src, scalar, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (IsArithIntType(src_dt) || src_dt == DataType::FP16 ||
                                                    src_dt == DataType::FP32 || src_dt == DataType::BF16))
        << "vf.adds src only supports supported types, got " << DTypeStr(src_dt);
    DataType scalar_dt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (scalar_dt.IsInt() || scalar_dt == DataType::FP16 ||
                                                    scalar_dt == DataType::FP32 || scalar_dt == DataType::BF16))
        << "vf.adds scalar only supports INT8/UINT8/INT16/UINT16/INT32/UINT32/INT64/UINT64/FP16/FP32/BF16, got "
        << DTypeStr(scalar_dt);
    DataType vf_adds_dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == vf_adds_dst_dt)
        << "vf.adds requires src and dst to have the same type, got dst=" << DTypeStr(vf_adds_dst_dt)
        << " src=" << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string scalar_str = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.adds");
    scalar_str = CoerceScalarToInt(op->args_[2], src_dt, scalar_str);
    if (src_dt.GetBit() == 64) {
        // B64 Adds: dst = src + scalar.
        // Mirrors AscendC AddsImpl: Duplicate(scalar) + Add(src, scalar_reg).
        // Duplicate → DuplicateB64Impl (vdup b32 halves → interleave to b64).
        // Add → CalTraitOneByTransToTraitTwo (ppack mask → deinterleave both
        // → AddB64Impl: vaddc + vaddcs → interleave back).
        std::string p = dst + "_adds_";
        std::string cast_type = (src_dt == DataType::INT64) ? "(int64_t)" : "(uint64_t)";
        std::string b32_cast = (src_dt == DataType::INT64) ? "(RegTensor<int32_t>&)" : "(RegTensor<uint32_t>&)";
        // 1. Pack b64 mask to b32 (mirrors MaskPack in CalTraitOneByTransToTraitTwo)
        std::string packed_m = p + "_pm_";
        codegen.Emit("MaskReg " + packed_m + ";");
        codegen.Emit("ppack(" + packed_m + ", " + mask + ", LOWER);");
        // 2. Broadcast scalar to b32 halves (mirrors DuplicateB64Impl)
        std::string all_m = p + "_allm_";
        codegen.Emit("MaskReg " + all_m + " = pset_b32(PAT_ALL);");
        std::string lo_sc = p + "_losc_";
        std::string hi_sc = p + "_hisc_";
        codegen.Emit("RegTensor<uint32_t> " + lo_sc + ";");
        codegen.Emit("RegTensor<uint32_t> " + hi_sc + ";");
        codegen.Emit("vdup(" + lo_sc + ", (int32_t)(" + cast_type + "(" + scalar_str + ")), " + all_m +
                     ", MODE_ZEROING);");
        codegen.Emit("vdup(" + hi_sc + ", (int32_t)((" + cast_type + "(" + scalar_str + ")) >> 32), " + all_m +
                     ", MODE_ZEROING);");
        // 3. Deinterleave src to b32 halves (mirrors B64TraitOneToTraitTwo)
        EmitB64Deinterleave(codegen, p + "s", src);
        std::string lo_s = p + "s_lo_", hi_s = p + "s_hi_";
        // 4. AddB64: vaddc + vaddcs (mirrors AddB64Impl, using packed b32 mask)
        std::string carry = p + "_carry_";
        std::string lo_d = p + "_lod_";
        std::string hi_d = p + "_hid_";
        codegen.Emit("MaskReg " + carry + ";");
        codegen.Emit("RegTensor<uint32_t> " + lo_d + ";");
        codegen.Emit("RegTensor<uint32_t> " + hi_d + ";");
        codegen.Emit("vaddc(" + carry + ", " + b32_cast + lo_d + ", " + b32_cast + lo_s + ", " + b32_cast + lo_sc +
                     ", " + packed_m + ");");
        codegen.Emit("vaddcs(" + carry + ", " + b32_cast + hi_d + ", " + b32_cast + hi_s + ", " + b32_cast + hi_sc +
                     ", " + carry + ", " + packed_m + ");");
        // 5. Interleave back (mirrors B64TraitTwoToTraitOne)
        EmitB64Interleave(codegen, dst, lo_d, hi_d, p + "_ilv");
        // No separate EmitB64Zeroing — AscendC ZEROING mode relies on vaddc/vaddcs
        // producing 0 for inactive elements (mask-controlled). The packed b32 mask
        // ensures correct element selection.
    } else {
        codegen.Emit("vadds(" + dst + ", " + src + ", " + scalar_str + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// Mins — vmins (scalar minimum)
// ============================================================================

static std::string EmitVFMins(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.mins requires 4 args (dst, src, scalar, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (IsArithIntType(src_dt) || src_dt == DataType::FP16 ||
                                                    src_dt == DataType::FP32 || src_dt == DataType::BF16))
        << "vf.mins src only supports supported types, got " << DTypeStr(src_dt);
    DataType scalar_dt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (scalar_dt.IsInt() || scalar_dt == DataType::FP16 ||
                                                    scalar_dt == DataType::FP32 || scalar_dt == DataType::BF16))
        << "vf.mins scalar only supports INT8/UINT8/INT16/UINT16/INT32/UINT32/INT64/UINT64/FP16/FP32/BF16, got "
        << DTypeStr(scalar_dt);
    DataType vf_mins_dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == vf_mins_dst_dt)
        << "vf.mins requires src and dst to have the same type, got dst=" << DTypeStr(vf_mins_dst_dt)
        << " src=" << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string scalar_str = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.mins");
    scalar_str = CoerceScalarToInt(op->args_[2], src_dt, scalar_str);
    if (src_dt.GetBit() == 64) {
        // B64 mins: min(src, scalar) via deinterleave + scalar compare + vsel + interleave.
        std::string p = dst + "_mins_";
        EmitB64Deinterleave(codegen, p + "s", src);
        std::string lo_s = p + "s_lo_", hi_s = p + "s_hi_";
        std::string all_m = p + "_allm_";
        codegen.Emit("MaskReg " + all_m + " = pset_b32(PAT_ALL);");
        // Broadcast scalar to b32 halves
        std::string cast_type = (src_dt == DataType::INT64) ? "(int64_t)" : "(uint64_t)";
        std::string lo_scalar = p + "_losc_";
        std::string hi_scalar = p + "_hisc_";
        codegen.Emit("RegTensor<uint32_t> " + lo_scalar + ";");
        codegen.Emit("RegTensor<uint32_t> " + hi_scalar + ";");
        codegen.Emit("vdup(" + lo_scalar + ", (int32_t)(" + cast_type + "(" + scalar_str + ")), " + all_m +
                     ", MODE_ZEROING);");
        codegen.Emit("vdup(" + hi_scalar + ", (int32_t)((" + cast_type + "(" + scalar_str + ")) >> 32), " + all_m +
                     ", MODE_ZEROING);");
        // Compare: src < scalar → pick src, else pick scalar
        std::string hi_lt = p + "_hilt_";
        std::string hi_eq = p + "_hieq_";
        std::string lo_lt = p + "_lolt_";
        codegen.Emit("MaskReg " + hi_lt + ";");
        codegen.Emit("MaskReg " + hi_eq + ";");
        codegen.Emit("MaskReg " + lo_lt + ";");
        if (src_dt == DataType::INT64) {
            codegen.Emit("vcmp_lt(" + hi_lt + ", (RegTensor<int32_t>&)" + hi_s + ", (RegTensor<int32_t>&)" + hi_scalar +
                         ", " + all_m + ");");
            codegen.Emit("vcmp_eq(" + hi_eq + ", (RegTensor<int32_t>&)" + hi_s + ", (RegTensor<int32_t>&)" + hi_scalar +
                         ", " + all_m + ");");
        } else {
            codegen.Emit("vcmp_lt(" + hi_lt + ", " + hi_s + ", " + hi_scalar + ", " + all_m + ");");
            codegen.Emit("vcmp_eq(" + hi_eq + ", " + hi_s + ", " + hi_scalar + ", " + all_m + ");");
        }
        codegen.Emit("vcmp_lt(" + lo_lt + ", " + lo_s + ", " + lo_scalar + ", " + all_m + ");");
        std::string eq_lo = p + "_eqlo_";
        std::string s_lt = p + "_slt_";
        codegen.Emit("MaskReg " + eq_lo + ";");
        codegen.Emit("MaskReg " + s_lt + ";");
        codegen.Emit("pand(" + eq_lo + ", " + hi_eq + ", " + lo_lt + ", " + all_m + ");");
        codegen.Emit("por(" + s_lt + ", " + hi_lt + ", " + eq_lo + ", " + all_m + ");");
        // Select: min = src < scalar ? src : scalar
        std::string lo_dst = p + "_lod_";
        std::string hi_dst = p + "_hid_";
        codegen.Emit("RegTensor<uint32_t> " + lo_dst + ";");
        codegen.Emit("RegTensor<uint32_t> " + hi_dst + ";");
        codegen.Emit("vsel(" + lo_dst + ", " + lo_s + ", " + lo_scalar + ", " + s_lt + ");");
        codegen.Emit("vsel(" + hi_dst + ", " + hi_s + ", " + hi_scalar + ", " + s_lt + ");");
        EmitB64Interleave(codegen, dst, lo_dst, hi_dst, p + "_ilv");
        EmitB64Zeroing(codegen, dst, dst, mask, p + "_zero");
    } else {
        codegen.Emit("vmins(" + dst + ", " + src + ", " + scalar_str + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// Maxs — vmaxs (scalar maximum)
// ============================================================================

static std::string EmitVFMaxs(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.maxs requires 4 args (dst, src, scalar, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (IsArithIntType(src_dt) || src_dt == DataType::FP16 ||
                                                    src_dt == DataType::FP32 || src_dt == DataType::BF16))
        << "vf.maxs src only supports supported types, got " << DTypeStr(src_dt);
    DataType scalar_dt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (scalar_dt.IsInt() || scalar_dt == DataType::FP16 ||
                                                    scalar_dt == DataType::FP32 || scalar_dt == DataType::BF16))
        << "vf.maxs scalar only supports INT8/UINT8/INT16/UINT16/INT32/UINT32/INT64/UINT64/FP16/FP32/BF16, got "
        << DTypeStr(scalar_dt);
    DataType vf_maxs_dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == vf_maxs_dst_dt)
        << "vf.maxs requires src and dst to have the same type, got dst=" << DTypeStr(vf_maxs_dst_dt)
        << " src=" << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string scalar_str = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.maxs");
    scalar_str = CoerceScalarToInt(op->args_[2], src_dt, scalar_str);
    if (src_dt.GetBit() == 64) {
        // B64 maxs: max(src, scalar) via deinterleave + scalar compare + vsel + interleave.
        std::string p = dst + "_maxs_";
        EmitB64Deinterleave(codegen, p + "s", src);
        std::string lo_s = p + "s_lo_", hi_s = p + "s_hi_";
        std::string all_m = p + "_allm_";
        codegen.Emit("MaskReg " + all_m + " = pset_b32(PAT_ALL);");
        std::string cast_type = (src_dt == DataType::INT64) ? "(int64_t)" : "(uint64_t)";
        std::string lo_scalar = p + "_losc_";
        std::string hi_scalar = p + "_hisc_";
        codegen.Emit("RegTensor<uint32_t> " + lo_scalar + ";");
        codegen.Emit("RegTensor<uint32_t> " + hi_scalar + ";");
        codegen.Emit("vdup(" + lo_scalar + ", (int32_t)(" + cast_type + "(" + scalar_str + ")), " + all_m +
                     ", MODE_ZEROING);");
        codegen.Emit("vdup(" + hi_scalar + ", (int32_t)((" + cast_type + "(" + scalar_str + ")) >> 32), " + all_m +
                     ", MODE_ZEROING);");
        // Compare: src > scalar → pick src, else pick scalar
        std::string hi_gt = p + "_higt_";
        std::string hi_eq = p + "_hieq_";
        std::string lo_gt = p + "_logt_";
        codegen.Emit("MaskReg " + hi_gt + ";");
        codegen.Emit("MaskReg " + hi_eq + ";");
        codegen.Emit("MaskReg " + lo_gt + ";");
        if (src_dt == DataType::INT64) {
            codegen.Emit("vcmp_gt(" + hi_gt + ", (RegTensor<int32_t>&)" + hi_s + ", (RegTensor<int32_t>&)" + hi_scalar +
                         ", " + all_m + ");");
            codegen.Emit("vcmp_eq(" + hi_eq + ", (RegTensor<int32_t>&)" + hi_s + ", (RegTensor<int32_t>&)" + hi_scalar +
                         ", " + all_m + ");");
        } else {
            codegen.Emit("vcmp_gt(" + hi_gt + ", " + hi_s + ", " + hi_scalar + ", " + all_m + ");");
            codegen.Emit("vcmp_eq(" + hi_eq + ", " + hi_s + ", " + hi_scalar + ", " + all_m + ");");
        }
        codegen.Emit("vcmp_gt(" + lo_gt + ", " + lo_s + ", " + lo_scalar + ", " + all_m + ");");
        std::string eq_lo = p + "_eqlo_";
        std::string s_gt = p + "_sgt_";
        codegen.Emit("MaskReg " + eq_lo + ";");
        codegen.Emit("MaskReg " + s_gt + ";");
        codegen.Emit("pand(" + eq_lo + ", " + hi_eq + ", " + lo_gt + ", " + all_m + ");");
        codegen.Emit("por(" + s_gt + ", " + hi_gt + ", " + eq_lo + ", " + all_m + ");");
        // Select: max = src > scalar ? src : scalar
        std::string lo_dst = p + "_lod_";
        std::string hi_dst = p + "_hid_";
        codegen.Emit("RegTensor<uint32_t> " + lo_dst + ";");
        codegen.Emit("RegTensor<uint32_t> " + hi_dst + ";");
        codegen.Emit("vsel(" + lo_dst + ", " + lo_s + ", " + lo_scalar + ", " + s_gt + ");");
        codegen.Emit("vsel(" + hi_dst + ", " + hi_s + ", " + hi_scalar + ", " + s_gt + ");");
        EmitB64Interleave(codegen, dst, lo_dst, hi_dst, p + "_ilv");
        EmitB64Zeroing(codegen, dst, dst, mask, p + "_zero");
    } else {
        codegen.Emit("vmaxs(" + dst + ", " + src + ", " + scalar_str + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// LeakyRelu — vlrelu
// ============================================================================

static std::string EmitVFLeakyRelu(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.leaky_relu requires 4 args (dst, src, alpha, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (src_dt == DataType::FP16 || src_dt == DataType::FP32))
        << "vf.leaky_relu src only supports supported types, got " << DTypeStr(src_dt);
    DataType alpha_dt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (alpha_dt == DataType::FP16 || alpha_dt == DataType::FP32))
        << "vf.leaky_relu scalar only supports FP16/FP32, got " << DTypeStr(alpha_dt);
    DataType vf_leaky_relu_dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == vf_leaky_relu_dst_dt)
        << "vf.leaky_relu requires src and dst to have the same type, got dst=" << DTypeStr(vf_leaky_relu_dst_dt)
        << " src=" << DTypeStr(src_dt);
    DataType lrelu_dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == lrelu_dst_dt)
        << "vf.leaky_relu requires src and dst to have the same type, got dst=" << DTypeStr(lrelu_dst_dt)
        << " src=" << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string alpha = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.leaky_relu");
    codegen.Emit("vlrelu(" + dst + ", " + src + ", " + alpha + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Interleave — vintlv
// ============================================================================

static std::string EmitVFInterleave(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.interleave requires 4 args (dst0, dst1, src0, src1)";
    std::string dst0 = codegen.GetExprAsCode(op->args_[0]);
    std::string dst1 = codegen.GetExprAsCode(op->args_[1]);
    std::string src0 = codegen.GetExprAsCode(op->args_[2]);
    std::string src1 = codegen.GetExprAsCode(op->args_[3]);
    if (IsDstMaskReg(op, codegen)) {
        DataType dtype = DataType::FP32;
        if (op->HasKwarg("dtype")) {
            dtype = op->GetKwarg<DataType>("dtype");
        }
        // MaskInterleave: only supports b8/b16/b32 (not b64)
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                          dtype.GetBit() == 8 || dtype.GetBit() == 16 || dtype.GetBit() == 32)
            << "vf.interleave (MaskReg) only supports b8/b16/b32, got " << DTypeStr(dtype);
        std::string pintlv_op;
        if (dtype == DataType::UINT8 || dtype == DataType::INT8) {
            pintlv_op = "pintlv_b8";
        } else if (dtype.GetBit() == 16) {
            pintlv_op = "pintlv_b16";
        } else {
            pintlv_op = "pintlv_b32";
        }
        codegen.Emit(pintlv_op + "(" + dst0 + ", " + dst1 + ", " + src0 + ", " + src1 + ");");
        return "";
    }
    // vintlv requires src0/src1 to be b8/b16/b32/b64
    DataType src0_dt = GetExprDtype(op->args_[2]);
    DataType src1_dt = GetExprDtype(op->args_[3]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (IsB8Type(src0_dt) || src0_dt.GetBit() == 16 || src0_dt.GetBit() == 32 || src0_dt.GetBit() == 64))
        << "vf.interleave only supports b8/b16/b32/b64 types, got " << DTypeStr(src0_dt);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (src0_dt == src1_dt))
        << "vf.interleave requires src0 and src1 to have the same type, got src0=" << DTypeStr(src0_dt)
        << " src1=" << DTypeStr(src1_dt);
    DataType dst0_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src0_dt == dst0_dt)
        << "vf.interleave requires src and dst to have the same type, got dst=" << DTypeStr(dst0_dt)
        << " src=" << DTypeStr(src0_dt);
    if (src0_dt.GetBit() == 64) {
        codegen.Emit("vintlv((RegTensor<uint32_t>&)" + dst0 + ", (RegTensor<uint32_t>&)" + dst1 +
                     ", (RegTensor<uint32_t>&)" + src0 + ", (RegTensor<uint32_t>&)" + src1 + ");");
    } else {
        codegen.Emit("vintlv(" + dst0 + ", " + dst1 + ", " + src0 + ", " + src1 + ");");
    }
    return "";
}

// ============================================================================
// PairReduceSum — vcpadd
// ============================================================================

static std::string EmitVFPairReduceSum(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "vf.pair_reduce_sum requires 3 args (dst, src, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (src_dt == DataType::FP16 || src_dt == DataType::FP32))
        << "vf.pair_reduce_sum src only supports FP16/FP32, got " << DTypeStr(src_dt);
    DataType prs_dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == prs_dst_dt)
        << "vf.pair_reduce_sum requires src and dst to have the same type, got dst=" << DTypeStr(prs_dst_dt)
        << " src=" << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string mode = VFZeroingOnly(op, "vf.pair_reduce_sum");
    codegen.Emit("vcpadd(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// AbsSub — vabsdif (absolute difference: |src0 - src1|)
// ============================================================================

static std::string EmitVFAbsSub(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.abs_sub requires 4 args (dst, src0, src1, mask)";
    DataType s0_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (s0_dt == DataType::FP16 || s0_dt == DataType::FP32 || s0_dt == DataType::INT64))
        << "vf.abs_sub src0 only supports FP16/FP32/INT64, got " << DTypeStr(s0_dt);
    DataType s1_dt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (s1_dt == DataType::FP16 || s1_dt == DataType::FP32 || s1_dt == DataType::INT64))
        << "vf.abs_sub src1 only supports FP16/FP32/INT64, got " << DTypeStr(s1_dt);
    DataType vf_abs_sub_dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, s0_dt == vf_abs_sub_dst_dt && s1_dt == vf_abs_sub_dst_dt)
        << "vf.abs_sub requires dst, src0, src1 to have the same type, got dst=" << DTypeStr(vf_abs_sub_dst_dt)
        << " src0=" << DTypeStr(s0_dt) << " src1=" << DTypeStr(s1_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.abs_sub");
    if (s0_dt == DataType::INT64) {
        // B64 abs_sub: |src0 - src1| = abs(vsub(src0, src1)).
        // vsub has no b64 single-register overload; use deinterleave + vsub/vsubc
        // borrow-chain (mirrors AscendC SubB64Impl for int64_t).
        std::string p = dst + "_abssub_";
        std::string diff_m = p + "_diffm_";
        codegen.Emit("MaskReg " + diff_m + " = pset_b32(PAT_ALL);");
        // Deinterleave both sources to b32 low/high halves
        EmitB64Deinterleave(codegen, p + "s0", src0);
        EmitB64Deinterleave(codegen, p + "s1", src1);
        std::string lo0 = p + "s0_lo_", hi0 = p + "s0_hi_";
        std::string lo1 = p + "s1_lo_", hi1 = p + "s1_hi_";
        // Sub with borrow: diff_lo = src0_lo - src1_lo, borrow = (src0_lo < src1_lo)
        std::string borrow = p + "_borrow_";
        std::string lo_d = p + "_dl_";
        std::string hi_d = p + "_dh_";
        codegen.Emit("MaskReg " + borrow + ";");
        codegen.Emit("RegTensor<int32_t> " + lo_d + ";");
        codegen.Emit("RegTensor<int32_t> " + hi_d + ";");
        codegen.Emit("vsubc(" + borrow + ", " + lo_d + ", (RegTensor<int32_t>&)" + lo0 + ", (RegTensor<int32_t>&)" +
                     lo1 + ", " + diff_m + ");");
        codegen.Emit("vsubcs(" + borrow + ", " + hi_d + ", (RegTensor<int32_t>&)" + hi0 + ", (RegTensor<int32_t>&)" +
                     hi1 + ", " + borrow + ", " + diff_m + ");");
        // Now diff = {lo_d, hi_d}. Compute abs via sign check + negate.
        std::string all_m = p + "_allm_";
        codegen.Emit("MaskReg " + all_m + " = pset_b32(PAT_ALL);");
        // Zero register (vbr equivalent)
        std::string zero_reg = p + "_zero_";
        codegen.Emit("RegTensor<int32_t> " + zero_reg + ";");
        codegen.Emit("vdup(" + zero_reg + ", 0, " + all_m + ", MODE_ZEROING);");
        // Sign mask: hi_d < 0 (signed)
        std::string sign_m = p + "_sign_";
        codegen.Emit("MaskReg " + sign_m + ";");
        codegen.Emit("vcmp_lt(" + sign_m + ", (RegTensor<int32_t>&)" + hi_d + ", (RegTensor<int32_t>&)" + zero_reg +
                     ", " + all_m + ");");
        // Negate via borrow-chain sub: neg = 0 - diff
        // vsubc: borrow = (0 < lo_d), neg_lo = 0 - lo_d
        // vsubcs: neg_hi = 0 - hi_d - borrow
        std::string borrow2 = p + "_borrow2_";
        std::string lo_n = p + "_nlo_";
        std::string hi_n = p + "_nhi_";
        codegen.Emit("MaskReg " + borrow2 + ";");
        codegen.Emit("RegTensor<int32_t> " + lo_n + ";");
        codegen.Emit("RegTensor<int32_t> " + hi_n + ";");
        codegen.Emit("vsubc(" + borrow2 + ", " + lo_n + ", " + zero_reg + ", (RegTensor<int32_t>&)" + lo_d + ", " +
                     sign_m + ");");
        codegen.Emit("vsubcs(" + borrow2 + ", " + hi_n + ", " + zero_reg + ", (RegTensor<int32_t>&)" + hi_d + ", " +
                     borrow2 + ", " + sign_m + ");");
        // Select: if negative -> neg, else -> diff (per b32 half)
        // All vsel operands must be the same type (int32_t).
        std::string lo_dst = p + "_lod_";
        std::string hi_dst = p + "_hid_";
        codegen.Emit("RegTensor<int32_t> " + lo_dst + ";");
        codegen.Emit("RegTensor<int32_t> " + hi_dst + ";");
        codegen.Emit("vsel(" + lo_dst + ", " + lo_n + ", (RegTensor<int32_t>&)" + lo_d + ", " + sign_m + ");");
        codegen.Emit("vsel(" + hi_dst + ", " + hi_n + ", (RegTensor<int32_t>&)" + hi_d + ", " + sign_m + ");");
        EmitB64Interleave(codegen, dst, lo_dst, hi_dst, p + "_ilv");
        EmitB64Zeroing(codegen, dst, dst, mask, p + "_zero");
    } else {
        codegen.Emit("vabsdif(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// Axpy — vaxpy (accumulate: dst = src * scalar + dst)
// ============================================================================

static std::string EmitVFAxpy(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.axpy requires 4 args (dst, src, scalar, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (src_dt == DataType::FP16 || src_dt == DataType::FP32 ||
                                                    src_dt == DataType::INT64 || src_dt == DataType::UINT64))
        << "vf.axpy src only supports FP16/FP32/INT64/UINT64, got " << DTypeStr(src_dt);
    // vaxpy supports half/float/uint64_t/int64_t
    DataType scalar_dt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, scalar_dt == DataType::FP16 || scalar_dt == DataType::FP32 ||
                                                       scalar_dt == DataType::UINT64 || scalar_dt == DataType::INT64 ||
                                                       scalar_dt == DataType::INDEX)
        << "vf.axpy scalar only supports FP16/FP32/UINT64/INT64, got " << DTypeStr(scalar_dt);
    DataType vf_axpy_dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == vf_axpy_dst_dt)
        << "vf.axpy requires src and dst to have the same type, got dst=" << DTypeStr(vf_axpy_dst_dt)
        << " src=" << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string scalar_str = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.axpy");
    scalar_str = CoerceScalarToInt(op->args_[2], src_dt, scalar_str);
    if (src_dt == DataType::INT64 || src_dt == DataType::UINT64) {
        // B64 Axpy: dst = src * scalar + dst → Muls (Duplicate + MulB64Impl) + Add (AddB64Impl).
        // Mirrors AscendC AxpyImpl for b64.
        std::string p = dst + "_axpy_";
        std::string cast_type = (src_dt == DataType::INT64) ? "(int64_t)" : "(uint64_t)";
        // AscendC MulB64Impl: vmull always uses uint32_t (unsigned 32x32→64);
        // vmula uses int32_t for int64_t (signed cross-term), uint32_t for uint64_t.
        std::string vmull_cast = "(RegTensor<uint32_t>&)";
        std::string vmula_cast = (src_dt == DataType::INT64) ? "(RegTensor<int32_t>&)" : "(RegTensor<uint32_t>&)";
        std::string b32_cast = (src_dt == DataType::INT64) ? "(RegTensor<int32_t>&)" : "(RegTensor<uint32_t>&)";
        // 1. Pack b64 mask to b32 (mirrors MaskPack in CalTraitOneByTransToTraitTwo)
        std::string packed_m = p + "_pm_";
        codegen.Emit("MaskReg " + packed_m + ";");
        codegen.Emit("ppack(" + packed_m + ", " + mask + ", LOWER);");
        // 2. Broadcast scalar as b32 halves (mirrors DuplicateB64Impl)
        std::string all_m = p + "_allm_";
        codegen.Emit("MaskReg " + all_m + " = pset_b32(PAT_ALL);");
        std::string lo_sc = p + "_losc_";
        std::string hi_sc = p + "_hisc_";
        codegen.Emit("RegTensor<uint32_t> " + lo_sc + ";");
        codegen.Emit("RegTensor<uint32_t> " + hi_sc + ";");
        codegen.Emit("vdup(" + lo_sc + ", (int32_t)(" + cast_type + "(" + scalar_str + ")), " + all_m +
                     ", MODE_ZEROING);");
        codegen.Emit("vdup(" + hi_sc + ", (int32_t)((" + cast_type + "(" + scalar_str + ")) >> 32), " + all_m +
                     ", MODE_ZEROING);");
        // 3. Deinterleave src
        EmitB64Deinterleave(codegen, p + "s", src);
        std::string lo_s = p + "s_lo_", hi_s = p + "s_hi_";
        // 4. MulB64: vmull + vmula (32×32→64 long multiply)
        std::string mul_lo = p + "_mul_lo_";
        std::string mul_hi = p + "_mul_hi_";
        codegen.Emit("RegTensor<uint32_t> " + mul_lo + ";");
        codegen.Emit("RegTensor<uint32_t> " + mul_hi + ";");
        codegen.Emit("vmull(" + vmull_cast + mul_lo + ", " + vmull_cast + mul_hi + ", " + vmull_cast + lo_s + ", " +
                     vmull_cast + lo_sc + ", " + packed_m + ");");
        codegen.Emit("vmula(" + vmula_cast + mul_hi + ", " + vmula_cast + lo_s + ", " + vmula_cast + hi_sc + ", " +
                     packed_m + ", MODE_ZEROING);");
        codegen.Emit("vmula(" + vmula_cast + mul_hi + ", " + vmula_cast + hi_s + ", " + vmula_cast + lo_sc + ", " +
                     packed_m + ", MODE_ZEROING);");
        // 5. Deinterleave dst (for AddB64: dst = mul_result + dst)
        EmitB64Deinterleave(codegen, p + "d", dst);
        std::string lo_d = p + "d_lo_", hi_d = p + "d_hi_";
        // 6. AddB64: vaddc + vaddcs (carry-chain add, using packed b32 mask)
        std::string carry = p + "_carry_";
        std::string add_lo = p + "_add_lo_";
        std::string add_hi = p + "_add_hi_";
        codegen.Emit("MaskReg " + carry + ";");
        codegen.Emit("RegTensor<uint32_t> " + add_lo + ";");
        codegen.Emit("RegTensor<uint32_t> " + add_hi + ";");
        codegen.Emit("vaddc(" + carry + ", " + b32_cast + add_lo + ", " + b32_cast + mul_lo + ", " + b32_cast + lo_d +
                     ", " + packed_m + ");");
        codegen.Emit("vaddcs(" + carry + ", " + b32_cast + add_hi + ", " + b32_cast + mul_hi + ", " + b32_cast + hi_d +
                     ", " + carry + ", " + packed_m + ");");
        EmitB64Interleave(codegen, dst, add_lo, add_hi, p + "_ilv");
    } else {
        codegen.Emit("vaxpy(" + dst + ", " + src + ", " + scalar_str + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// Madd — vmadd (multiply-accumulate: dst = src0 * src1 + dst)
// ============================================================================

static std::string EmitVFMulDstAdd(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.mul_dst_add requires 4 args (dst, src0, src1, mask)";
    DataType s0_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (s0_dt == DataType::FP16 || s0_dt == DataType::FP32 || s0_dt == DataType::BF16))
        << "vf.mul_dst_add src0 only supports FP16/FP32/BF16, got " << DTypeStr(s0_dt);
    DataType s1_dt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (s1_dt == DataType::FP16 || s1_dt == DataType::FP32 || s1_dt == DataType::BF16))
        << "vf.mul_dst_add src1 only supports FP16/FP32/BF16, got " << DTypeStr(s1_dt);
    DataType vf_mul_dst_add_dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, s0_dt == vf_mul_dst_add_dst_dt && s1_dt == vf_mul_dst_add_dst_dt)
        << "vf.mul_dst_add requires dst, src0, src1 to have the same type, got dst=" << DTypeStr(vf_mul_dst_add_dst_dt)
        << " src0=" << DTypeStr(s0_dt) << " src1=" << DTypeStr(s1_dt);
    // vmadd supports u16/i16/u32/i32/half/float/bf/i64/u64
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.mul_dst_add");
    codegen.Emit("vmadd(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Pack — vpack (narrow data type)
// ============================================================================

static std::string EmitVFPack(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2) << "vf.pack requires 2 args (dst, src)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string part = "LOWER";
    if (op->HasKwarg("part")) {
        int part_val = op->GetKwarg<int>("part");
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, part_val == static_cast<int>(ir::PackPart::LOWER) ||
                                                               part_val == static_cast<int>(ir::PackPart::UPPER))
            << "vf.pack part must be PackPart::LOWER or PackPart::UPPER, got " << part_val;
        part = VFEnumValueName(ir::EnumToString(static_cast<ir::PackPart>(part_val)));
    }
    if (IsDstMaskReg(op, codegen)) {
        std::string cce_half = (part == "LOWER" || part == "LOWEST") ? "LOWER" : "HIGHER";
        codegen.Emit("ppack(" + dst + ", " + src + ", " + cce_half + ");");
        return "";
    }

    DataType src_dt = GetExprDtype(op->args_[1]);
    DataType dst_dt = GetExprDtype(op->args_[0]);
    // Validate (dst, src) as a pair (mirrors AscendC PackImpl SupportType
    // tuples): dst is always unsigned and half the src width; src may be
    // signed or unsigned.
    //   (UINT8, INT16|UINT16), (UINT16, INT32|UINT32), (UINT32, INT64|UINT64)
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (dst_dt == DataType::UINT8 && (src_dt == DataType::INT16 || src_dt == DataType::UINT16)) ||
                          (dst_dt == DataType::UINT16 && (src_dt == DataType::INT32 || src_dt == DataType::UINT32)) ||
                          (dst_dt == DataType::UINT32 && (src_dt == DataType::INT64 || src_dt == DataType::UINT64)))
        << "vf.pack supports UINT8<-INT16/UINT16, UINT16<-INT32/UINT32, UINT32<-INT64/UINT64 pairs, got dst="
        << DTypeStr(dst_dt) << " src=" << DTypeStr(src_dt);

    if (src_dt.GetBit() == 64) {
        // 64-bit source → 32-bit dst: use DeInterleave with a zero register,
        // mirroring PackImpl<..., part>(dst, src) for 8-byte src.
        std::string part_check = (part == "LOWER" || part == "LOWEST") ? "LOWEST" : "HIGHEST";
        std::string zero_var = dst + "_pack_zero_";
        std::string dump_var = dst + "_pack_dump_";
        std::string mask_var = dst + "_pack_mask_";
        codegen.Emit("RegTensor<uint32_t> " + zero_var + ";");
        codegen.Emit("RegTensor<uint32_t> " + dump_var + ";");
        codegen.Emit("MaskReg " + mask_var + " = pset_b32(PAT_ALL);");
        codegen.Emit("vdup(" + zero_var + ", 0, " + mask_var + ", MODE_ZEROING);");
        if (part_check == "LOWEST") {
            codegen.Emit("vdintlv((RegTensor<uint32_t>&)" + dst + ", " + dump_var + ", " + "(RegTensor<uint32_t>&)" +
                         src + ", " + zero_var + ");");
        } else {
            codegen.Emit("vdintlv((RegTensor<uint32_t>&)" + dst + ", " + dump_var + ", " + zero_var +
                         ", (RegTensor<uint32_t>&)" + src + ");");
        }
    } else {
        // MODE_UNKNOWN mirrors AscendC: the C API asc_pack_impl (npu_arch_3510)
        // passes MODE_UNKNOWN and Reg::PackImpl omits the mode (intrinsic
        // default), leaving the half not selected by part unspecified.
        std::string cce_part = (part == "LOWER" || part == "LOWEST") ? "LOWER" : "HIGHER";
        codegen.Emit("vpack(" + dst + ", " + src + ", " + cce_part + ", MODE_UNKNOWN);");
    }
    return "";
}

// ============================================================================
// Unpack — vunpack (widen data type)
// ============================================================================

static std::string EmitVFUnpack(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2) << "vf.unpack requires 2 args (dst, src)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string part = "LOWER";
    if (op->HasKwarg("part")) {
        int part_val = op->GetKwarg<int>("part");
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, part_val == static_cast<int>(ir::PackPart::LOWER) ||
                                                               part_val == static_cast<int>(ir::PackPart::UPPER))
            << "vf.unpack part must be PackPart::LOWER or PackPart::UPPER, got " << part_val;
        part = VFEnumValueName(ir::EnumToString(static_cast<ir::PackPart>(part_val)));
    }

    if (IsDstMaskReg(op, codegen)) {
        std::string cce_half = (part == "LOWER" || part == "LOWEST") ? "LOWER" : "HIGHER";
        codegen.Emit("punpack(" + dst + ", " + src + ", " + cce_half + ");");
        return "";
    }

    DataType dst_dt = GetExprDtype(op->args_[0]);
    DataType src_dt = GetExprDtype(op->args_[1]);
    // Validate (dst, src) as a pair (mirrors AscendC UnPackImpl SupportType
    // tuples): dst is double the src width with matching signedness. FP and
    // other types are rejected here instead of failing late in bisheng
    // compilation.
    //   (INT16, INT8), (UINT16, UINT8), (INT32, INT16),
    //   (UINT32, UINT16), (INT64, INT32), (UINT64, UINT32)
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (dst_dt == DataType::INT16 && src_dt == DataType::INT8) ||
                                                       (dst_dt == DataType::UINT16 && src_dt == DataType::UINT8) ||
                                                       (dst_dt == DataType::INT32 && src_dt == DataType::INT16) ||
                                                       (dst_dt == DataType::UINT32 && src_dt == DataType::UINT16) ||
                                                       (dst_dt == DataType::INT64 && src_dt == DataType::INT32) ||
                                                       (dst_dt == DataType::UINT64 && src_dt == DataType::UINT32))
        << "vf.unpack supports INT16<-INT8, UINT16<-UINT8, INT32<-INT16, UINT32<-UINT16, INT64<-INT32, "
           "UINT64<-UINT32 pairs, got dst="
        << DTypeStr(dst_dt) << " src=" << DTypeStr(src_dt);

    if (dst_dt.GetBit() == 64) {
        std::string src_ctype = src_dt.ToCTypeString();
        std::string part_check = (part == "LOWER" || part == "LOWEST") ? "LOWEST" : "HIGHEST";
        std::string pad_var = dst + "_unpack_pad_";
        std::string dump_var = dst + "_unpack_dump_";
        std::string mask_var = dst + "_unpack_mask_";
        codegen.Emit("RegTensor<" + src_ctype + "> " + pad_var + ";");
        codegen.Emit("RegTensor<" + src_ctype + "> " + dump_var + ";");
        codegen.Emit("MaskReg " + mask_var + " = pset_b32(PAT_ALL);");
        if (src_dt == DataType::INT32) {
            codegen.Emit("vshrs(" + pad_var + ", " + src + ", 31, " + mask_var + ", MODE_ZEROING);");
        } else {
            codegen.Emit("vdup(" + pad_var + ", 0, " + mask_var + ", MODE_ZEROING);");
        }
        if (part_check == "LOWEST") {
            codegen.Emit("vintlv((RegTensor<" + src_ctype + ">&)" + dst + ", " + dump_var + ", " + "(RegTensor<" +
                         src_ctype + ">&)" + src + ", " + pad_var + ");");
        } else {
            codegen.Emit("vintlv(" + dump_var + ", (RegTensor<" + src_ctype + ">&)" + dst + ", " + "(RegTensor<" +
                         src_ctype + ">&)" + src + ", " + pad_var + ");");
        }
    } else {
        std::string cce_part = (part == "LOWER" || part == "LOWEST") ? "LOWER" : "HIGHER";
        codegen.Emit("vunpack(" + dst + ", " + src + ", " + cce_part + ");");
    }
    return "";
}

// ============================================================================
// PRelu — vprelu (parametric ReLU with per-element slope)
// ============================================================================

static std::string EmitVFPRelu(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.prelu requires 4 args (dst, src, slope, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (src_dt == DataType::FP16 || src_dt == DataType::FP32))
        << "vf.prelu src only supports supported types, got " << DTypeStr(src_dt);
    DataType slope_dt = GetExprDtype(op->args_[2]);
    DataType prelu_dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == prelu_dst_dt && slope_dt == prelu_dst_dt)
        << "vf.prelu requires dst, src, slope to have the same type, got dst=" << DTypeStr(prelu_dst_dt)
        << " src=" << DTypeStr(src_dt) << " slope=" << DTypeStr(slope_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string slope = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.prelu");
    codegen.Emit("vprelu(" + dst + ", " + src + ", " + slope + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// ShiftLeft — unified left shift: vshl (per-lane, shift is a RegTensor) or
// vshls (uniform, shift is a scalar). The former standalone vf.shift_lefts
// (scalar) op is merged in here; the register-vs-scalar decision is made from
// the codegen RegTensor registry (mirrors EmitVFDuplicate's IsRegTensorVar
// dispatch), not from the op name.
// ============================================================================

// Returns true when the shift-amount arg (op->args_[2]) is a per-lane vector
// register; false for a uniform scalar shift (integer literal or plain scalar).
static bool ShiftAmountIsRegister(const ir::CallPtr& op, codegen::CCECodegen& codegen)
{
    // Integer literals are always scalar shifts — no need to consult the registry.
    if (ir::As<ir::ConstInt>(op->args_[2]))
        return false;
    auto shift_var = ir::As<ir::Var>(op->args_[2]);
    if (!shift_var)
        return false;
    return codegen.IsRegTensorVar(codegen.GetVarName(shift_var));
}

static std::string EmitVFShift(const ir::CallPtr& op, codegen::CodegenBase& codegen_base, const std::string& op_name,
                               const std::string& vector_instruction, const std::string& scalar_instruction)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << op_name << " requires 4 args (dst, src, shift, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == DataType::INT8 || src_dt == DataType::UINT8 ||
                                                       src_dt == DataType::INT16 || src_dt == DataType::UINT16 ||
                                                       src_dt == DataType::INT32 || src_dt == DataType::UINT32 ||
                                                       src_dt == DataType::INT64 || src_dt == DataType::UINT64)
        << op_name << " src only supports integer types, got " << DTypeStr(src_dt);
    DataType shift_dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == shift_dst_dt)
        << op_name << " requires src and dst to have the same type, got dst=" << DTypeStr(shift_dst_dt)
        << " src=" << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string shift = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, op_name);
    if (src_dt.GetBit() == 64) {
        // B64 shift via deinterleave + b32 shift + cross-half carry + interleave.
        // Scalar shift only (shift amount is a compile-time or runtime scalar).
        std::string p = dst + "_shift_";
        EmitB64Deinterleave(codegen, p + "s", src);
        std::string lo_s = p + "s_lo_", hi_s = p + "s_hi_";
        std::string all_m = p + "_allm_";
        codegen.Emit("MaskReg " + all_m + " = pset_b32(PAT_ALL);");
        std::string lo_dst = p + "_lod_";
        std::string hi_dst = p + "_hid_";
        codegen.Emit("RegTensor<uint32_t> " + lo_dst + ";");
        codegen.Emit("RegTensor<uint32_t> " + hi_dst + ";");
        if (op_name == "vf.shift_left") {
            // Left shift: lo_dst = lo << N, carry = lo >> (32-N), hi_dst = (hi << N) | carry
            std::string carry = p + "_carry_";
            codegen.Emit("RegTensor<uint32_t> " + carry + ";");
            codegen.Emit("vshls(" + lo_dst + ", " + lo_s + ", (int16_t)(" + shift + "), " + all_m + ", MODE_ZEROING);");
            codegen.Emit("vshrs(" + carry + ", " + lo_s + ", (int16_t)(32 - (" + shift + ")), " + all_m +
                         ", MODE_ZEROING);");
            codegen.Emit("vshls(" + hi_dst + ", " + hi_s + ", (int16_t)(" + shift + "), " + all_m + ", MODE_ZEROING);");
            codegen.Emit("vor(" + hi_dst + ", " + hi_dst + ", " + carry + ", " + all_m + ", MODE_ZEROING);");
        } else {
            // Right shift: for INT64 (arithmetic), for UINT64 (logical)
            std::string carry = p + "_carry_";
            codegen.Emit("RegTensor<uint32_t> " + carry + ";");
            if (src_dt == DataType::INT64) {
                // Arithmetic right shift: sign-extend hi. The vshrs overload
                // requires dst and src to have the same signedness, so the hi
                // half is shifted in the int32 domain on both sides (mirrors
                // AscendC ShiftR<int32_t>).
                codegen.Emit("vshrs((RegTensor<int32_t>&)" + hi_dst + ", (RegTensor<int32_t>&)" + hi_s +
                             ", (int16_t)(" + shift + "), " + all_m + ", MODE_ZEROING);");
            } else {
                codegen.Emit("vshrs(" + hi_dst + ", " + hi_s + ", (int16_t)(" + shift + "), " + all_m +
                             ", MODE_ZEROING);");
            }
            codegen.Emit("vshls(" + carry + ", " + hi_s + ", (int16_t)(32 - (" + shift + ")), " + all_m +
                         ", MODE_ZEROING);");
            codegen.Emit("vshrs(" + lo_dst + ", " + lo_s + ", (int16_t)(" + shift + "), " + all_m + ", MODE_ZEROING);");
            codegen.Emit("vor(" + lo_dst + ", " + lo_dst + ", " + carry + ", " + all_m + ", MODE_ZEROING);");
        }
        EmitB64Interleave(codegen, dst, lo_dst, hi_dst, p + "_ilv");
        EmitB64Zeroing(codegen, dst, dst, mask, p + "_zero");
    } else if (ShiftAmountIsRegister(op, codegen)) {
        DataType shift_dt = GetExprDtype(op->args_[2]);
        DataType shift_need = DataType::INT8;
        if (src_dt.GetBit() == 16) {
            shift_need = DataType::INT16;
        } else if (src_dt.GetBit() == 32) {
            shift_need = DataType::INT32;
        } else if (src_dt.GetBit() == 64) {
            shift_need = DataType::INT64;
        }
        PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, shift_dt == shift_need)
            << op_name << " (vector) shift register must be the signed type matching the " << DTypeStr(src_dt)
            << " data width, expected " << DTypeStr(shift_need) << ", got " << DTypeStr(shift_dt);
        codegen.Emit(vector_instruction + "(" + dst + ", " + src + ", " + shift + ", " + mask + ", " + mode + ");");
    } else {
        codegen.Emit(scalar_instruction + "(" + dst + ", " + src + ", (int16_t)(" + shift + "), " + mask + ", " + mode +
                     ");");
    }
    return "";
}

static std::string EmitVFShiftLeft(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return EmitVFShift(op, codegen_base, "vf.shift_left", "vshl", "vshls");
}

// ============================================================================
// ShiftRight — unified right shift: vshr (per-lane, shift is a RegTensor) or
// vshrs (uniform, shift is a scalar). The former standalone vf.shift_rights
// (scalar) op is merged in here; dispatch uses ShiftAmountIsRegister (codegen
// RegTensor registry), not the op name.
// ============================================================================

static std::string EmitVFShiftRight(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return EmitVFShift(op, codegen_base, "vf.shift_right", "vshr", "vshrs");
}

// ============================================================================
// Mull — vmull (long multiply: 32x32->64, lo/hi split)
// ============================================================================

static std::string EmitVFMull(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 5)
        << "vf.mull requires 5 args (dst_lo, dst_hi, src0, src1, mask)";
    DataType s0_dt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (s0_dt == DataType::INT32 || s0_dt == DataType::UINT32))
        << "vf.mull src only supports supported types, got " << DTypeStr(s0_dt);
    DataType s1_dt = GetExprDtype(op->args_[3]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (s1_dt == DataType::INT32 || s1_dt == DataType::UINT32))
        << "vf.mull src only supports supported types, got " << DTypeStr(s1_dt);
    DataType dst_lo_dt = GetExprDtype(op->args_[0]);
    DataType dst_hi_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, dst_lo_dt == s0_dt && dst_hi_dt == s0_dt && s0_dt == s1_dt)
        << "vf.mull requires dst_lo, dst_hi, src0, src1 to have the same type, got dst_lo=" << DTypeStr(dst_lo_dt)
        << " dst_hi=" << DTypeStr(dst_hi_dt) << " src0=" << DTypeStr(s0_dt) << " src1=" << DTypeStr(s1_dt);
    std::string dst_lo = codegen.GetExprAsCode(op->args_[0]);
    std::string dst_hi = codegen.GetExprAsCode(op->args_[1]);
    std::string src0 = codegen.GetExprAsCode(op->args_[2]);
    std::string src1 = codegen.GetExprAsCode(op->args_[3]);
    std::string mask = codegen.GetExprAsCode(op->args_[4]);
    codegen.Emit("vmull(" + dst_lo + ", " + dst_hi + ", " + src0 + ", " + src1 + ", " + mask + ");");
    return "";
}

// ============================================================================
// Addc — vaddcs (add with carry)
// ============================================================================

static std::string EmitVFAddc(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 6)
        << "vf.addc requires 6 args (carry_out, dst, src0, src1, carry_in, mask)";
    DataType s0_dt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, s0_dt == DataType::INT32 || s0_dt == DataType::UINT32)
        << "vf.addc src0 only supports INT32/UINT32, got " << DTypeStr(s0_dt);
    DataType dst_dt = GetExprDtype(op->args_[1]);
    DataType s1_dt = GetExprDtype(op->args_[3]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, dst_dt == s0_dt && dst_dt == s1_dt)
        << "vf.addc requires dst, src0, src1 to have the same type, got dst=" << DTypeStr(dst_dt)
        << " src0=" << DTypeStr(s0_dt) << " src1=" << DTypeStr(s1_dt);
    std::string carry_out = codegen.GetExprAsCode(op->args_[0]);
    std::string dst = codegen.GetExprAsCode(op->args_[1]);
    std::string src0 = codegen.GetExprAsCode(op->args_[2]);
    std::string src1 = codegen.GetExprAsCode(op->args_[3]);
    std::string carry_in = codegen.GetExprAsCode(op->args_[4]);
    std::string mask = codegen.GetExprAsCode(op->args_[5]);
    codegen.Emit("vaddcs(" + carry_out + ", " + dst + ", " + src0 + ", " + src1 + ", " + carry_in + ", " + mask + ");");
    return "";
}

// ============================================================================
// Subc — vsubcs (subtract with borrow)
// ============================================================================

static std::string EmitVFSubc(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 6)
        << "vf.subc requires 6 args (borrow_out, dst, src0, src1, borrow_in, mask)";
    DataType s0_dt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, s0_dt == DataType::INT32 || s0_dt == DataType::UINT32)
        << "vf.subc src0 only supports INT32/UINT32, got " << DTypeStr(s0_dt);
    DataType dst_dt = GetExprDtype(op->args_[1]);
    DataType s1_dt = GetExprDtype(op->args_[3]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, dst_dt == s0_dt && dst_dt == s1_dt)
        << "vf.subc requires dst, src0, src1 to have the same type, got dst=" << DTypeStr(dst_dt)
        << " src0=" << DTypeStr(s0_dt) << " src1=" << DTypeStr(s1_dt);
    std::string borrow_out = codegen.GetExprAsCode(op->args_[0]);
    std::string dst = codegen.GetExprAsCode(op->args_[1]);
    std::string src0 = codegen.GetExprAsCode(op->args_[2]);
    std::string src1 = codegen.GetExprAsCode(op->args_[3]);
    std::string borrow_in = codegen.GetExprAsCode(op->args_[4]);
    std::string mask = codegen.GetExprAsCode(op->args_[5]);
    codegen.Emit("vsubcs(" + borrow_out + ", " + dst + ", " + src0 + ", " + src1 + ", " + borrow_in + ", " + mask +
                 ");");
    return "";
}

// ============================================================================
// ExpSub — vexpdif
// ============================================================================

// Helpers for multi-step int→int conversion via float intermediate.
// Available int→float vcvt overloads (bisheng intrinsics):
//   b8→f16 (widening), s16→f16 (same-width), s16→f32 (widening), s32→f32 (same-width)
// NOT available: u16→f16, u16→f32, u32→f32, s32→f16

// Emit int→float vcvt step. Returns the float dtype used (FP16 or FP32).
// For unavailable src→f16 paths (u16, u32, s32), routes through f32 instead.
static DataType EmitIntToFloatVcvt(codegen::CCECodegen& codegen, const std::string& dst_reg, const std::string& src_reg,
                                   const std::string& mask, const std::string& part, const std::string& round,
                                   const std::string& mode, DataType src_dt)
{
    if (src_dt.GetBit() == 8) {
        // b8→f16: widening → vcvt(dst, src, mask, PART, MODE) — 5 args
        codegen.Emit("vcvt(" + dst_reg + ", " + src_reg + ", " + mask + ", " + part + ", " + mode + ");");
        return DataType::FP16;
    }
    if (src_dt == DataType::INT16) {
        // s16→f16: same-width → vcvt(dst, src, mask, ROUND, MODE) — 5 args
        codegen.Emit("vcvt(" + dst_reg + ", " + src_reg + ", " + mask + ", " + round + ", " + mode + ");");
        return DataType::FP16;
    }
    // b32 (s32/u32): s32→f32 same-width. u32 needs reinterpret as s32 first.
    std::string effective_src = src_reg;
    if (src_dt == DataType::UINT32)
        effective_src = "(RegTensor<int32_t> &)" + src_reg;
    // s32→f32: same-width → vcvt(dst, src, mask, ROUND, MODE) — 5 args
    codegen.Emit("vcvt(" + dst_reg + ", " + effective_src + ", " + mask + ", " + round + ", " + mode + ");");
    return DataType::FP32;
}

// Emit float→int vcvt step. float_dt is always FP16 (from EmitIntToFloatVcvt).
static void EmitFloatToIntVcvt(codegen::CCECodegen& codegen, const std::string& dst_reg, const std::string& src_reg,
                               const std::string& mask, const std::string& part, const std::string& round,
                               const std::string& sat, const std::string& mode, DataType dst_dt)
{
    if (dst_dt.GetBit() == 8) {
        // float→b8: narrower-int → vcvt(dst, src, mask, ROUND, SAT, PART, MODE) — 7 args
        codegen.Emit("vcvt(" + dst_reg + ", " + src_reg + ", " + mask + ", " + round + ", " + sat + ", " + part + ", " +
                     mode + ");");
    } else if (dst_dt.GetBit() == 16) {
        // f16→b16: same-width → vcvt(dst, src, mask, ROUND, RS, MODE) — 5 args
        codegen.Emit("vcvt(" + dst_reg + ", " + src_reg + ", " + mask + ", " + round + ", " + sat + ", " + mode + ");");
    } else {
        // f16→b32: wider-int → vcvt(dst, src, mask, ROUND, PART, MODE) — 6 args
        codegen.Emit("vcvt(" + dst_reg + ", " + src_reg + ", " + mask + ", " + round + ", " + part + ", " + mode +
                     ");");
    }
}

static std::string EmitVFExpSub(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // Parser args order: [dst, src, max, mask]
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.exp_sub requires 4 args (dst, src, max, mask)";
    // vexpdiff: dst must be FP32, src can be FP32 or FP16
    DataType src_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == DataType::FP32 || src_dt == DataType::FP16)
        << "vf.exp_sub only supports FP32/FP16 src, got " << DTypeStr(src_dt);
    DataType dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, dst_dt == DataType::FP32)
        << "vf.exp_sub destination only supports FP32, got " << DTypeStr(dst_dt);
    DataType max_dt = GetExprDtype(op->args_[2]);
    // vexpdiff: src0 and src1 must be the same type (both float or both half)
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == max_dt)
        << "vf.exp_sub requires src and max to have the same type, got src=" << DTypeStr(src_dt)
        << " max=" << DTypeStr(max_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string max_reg = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    // layout kwarg selects the result half: ZERO -> PART_EVEN (default), ONE -> PART_ODD.
    // vexpdiff: only supports RegLayout ZERO/ONE
    std::string part = "PART_EVEN";
    if (op->HasKwarg("layout")) {
        auto layout = VFEnumValueName(ir::EnumToString(static_cast<ir::CastLayout>(op->GetKwarg<int>("layout"))));
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, layout == "ZERO" || layout == "ONE")
            << "vf.exp_sub only supports layout ZERO/ONE, got " << layout;
        if (layout == "ONE")
            part = "PART_ODD";
    }
    codegen.Emit("vexpdif(" + dst + ", " + src + ", " + max_reg + ", " + mask + ", " + part + ");");
    return "";
}

// ============================================================================
// Cast — vcvt
// ============================================================================

static std::string EmitVFCast(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // Parser args order: [dst, src, mask]
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "vf.astype requires 3 args (dst, src, mask)";
    // vcvt: src and dst must have different types
    DataType src_dtype = GetExprDtype(op->args_[1], DataType::FP32);
    DataType dst_dtype = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dtype != dst_dtype)
        << "vf.astype: src and dst must have different types (both are " << DTypeStr(src_dtype) << ")";
    // vcvt has no uint64 overloads on this device, and the AscendC Cast micro
    // instruction it mirrors doesn't support uint64 either — reject early
    // instead of failing late in bisheng.
    PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR,
                      src_dtype != DataType::UINT64 && dst_dtype != DataType::UINT64)
        << "vf.astype does not support DT_UINT64 (no vcvt overload), got src=" << DTypeStr(src_dtype)
        << " dst=" << DTypeStr(dst_dtype);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    // Get layout and round_mode with defaults
    std::string layout = "ZERO";
    if (op->HasKwarg("layout")) {
        layout = VFEnumValueName(ir::EnumToString(static_cast<ir::CastLayout>(op->GetKwarg<int>("layout"))));
        // vcvt: layout must be ZERO/ONE/TWO/THREE
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT,
                          layout == "ZERO" || layout == "ONE" || layout == "TWO" || layout == "THREE")
            << "vf.astype only supports layout ZERO/ONE/TWO/THREE, got " << layout;
    }
    std::string round_mode = "CAST_RINT";
    if (op->HasKwarg("round_mode")) {
        round_mode = VFEnumValueName(ir::EnumToString(static_cast<ir::VFRoundMode>(op->GetKwarg<int>("round_mode"))));
    }
    // A5 vcvt only supports MODE_ZEROING at the instruction level.
    std::string mode_value = VFZeroingOnly(op, "vf.astype");
    std::string part;
    if (layout == "ZERO")
        part = "PART_EVEN";
    else if (layout == "ONE")
        part = "PART_ODD";
    else if (layout == "TWO")
        part = "PART_TWO";
    else if (layout == "THREE")
        part = "PART_THREE";
    else
        part = "PART_EVEN";
    std::string part_pp;
    if (layout == "ZERO")
        part_pp = "PART_P0";
    else if (layout == "ONE")
        part_pp = "PART_P1";
    else if (layout == "TWO")
        part_pp = "PART_P2";
    else if (layout == "THREE")
        part_pp = "PART_P3";
    else
        part_pp = "PART_P0";
    // Map round_mode to ::ROUND constants (dav_3510 GetRound mapping):
    //   CAST_RINT  → ROUND_R (round to nearest even)
    //   CAST_ROUND → ROUND_A (round to nearest, away from zero)
    //   CAST_FLOOR → ROUND_F
    //   CAST_CEIL  → ROUND_C
    //   CAST_TRUNC → ROUND_Z
    //   CAST_ODD   → ROUND_O (Von Neumann rounding)
    //   CAST_HYBRID→ ROUND_H (3510 only, not supported by all vcvt overloads)
    //   unknown    → ROUND_R (safe fallback, universally supported)
    std::string round;
    if (round_mode == "CAST_RINT")
        round = "ROUND_R";
    else if (round_mode == "CAST_ROUND")
        round = "ROUND_A";
    else if (round_mode == "CAST_FLOOR")
        round = "ROUND_F";
    else if (round_mode == "CAST_CEIL")
        round = "ROUND_C";
    else if (round_mode == "CAST_TRUNC")
        round = "ROUND_Z";
    else if (round_mode == "CAST_ODD")
        round = "ROUND_O"; // only supported by widening float→float vcvt overloads
    else if (round_mode == "CAST_HYBRID")
        round = "ROUND_H"; // only supported by widening float→float vcvt overloads
    else
        round = "ROUND_R"; // unknown → ROUND_R (safe fallback)
    // Get saturation kwarg (default: disabled)
    std::string sat = "RS_DISABLE";
    if (op->HasKwarg("saturate")) {
        std::string sat_val = VFEnumValueName(
            ir::EnumToString(static_cast<ir::SaturateMode>(op->GetKwarg<int>("saturate"))));
        if (sat_val == "ON" || sat_val == "ENABLE")
            sat = "RS_ENABLE";
    }
    // Determine if narrowing or widening conversion based on src/dst dtype.
    // Widening: dst wider than src — no ROUND/RS needed, just PART + MODE
    // Narrowing: dst narrower — needs ROUND + RS + PART + MODE
    // Float→same-width-int (FP32→S32): ROUND + RS + MODE (no PART)
    // INT→FLOAT same-width (S32→FP32): ROUND + MODE (no RS, no PART)
    // FLOAT→FLOAT same-width (FP16→BF16): ROUND + MODE (no RS, no PART)
    // INT→INT narrowing: RS + PART + MODE (no ROUND)
    // Cross-width INT↔FLOAT (S16→FP32, S32→FP16, FP32→S64, S64→FP32): ROUND + PART + MODE
    bool is_widening = false;
    bool is_int_to_float = false;
    bool is_float_to_same_int = false;     // FP32→S32, FP16→S16 (same-width float→int)
    bool is_float_to_narrower_int = false; // FP16→INT8/UINT8, BF16→INT8/UINT8 (2x narrowing float→int)
    bool is_float_to_wider_int = false;    // FP16→S32, BF16→S32 (widening float→int)
    bool is_int_2x_narrowing = false;      // 2x narrowing with RS: b32→b16, b16→u8, b64→b32
    bool is_int_4x_narrowing = false;      // 4x narrowing with RS: b32→u8
    bool is_int_4x_widening = false;       // 4x int widening: b8→b32 (PART_P0/P1/P2/P3 only)
    bool is_cross_width = false;           // cross-width INT↔FLOAT with ROUND + PART + MODE
    bool is_s4_widening = false;           // INT4→FP16/BF16/INT16 (vcvt_s42f16/bf16/s16)
    bool is_s4_narrowing = false;          // FP16→INT4 (vcvt_f162s4)
    bool is_s16_to_s4 = false;             // INT16→INT4 (two-step: s16→f16→s4, mirroring AscendC Cast)
    bool is_int_int_two_step = false;      // int→int via f16 intermediate (cross-sign widening, int→INT8 narrowing)
    // FP8/FP4 low-precision conversions
    bool is_fp_widen_pp = false;          // 4x widening PP: FP8→FP32, FP4→BF16 → vcvt(dst,src,mask,PART_PP,MODE)
    bool is_fp_narrow_rnd_sat_pp = false; // 4x narrowing RND_SAT_PP: FP32→FP8 → vcvt(dst,src,mask,ROUND,SAT)
    bool is_fp_narrow_rnd_pp = false;     // 4x narrowing RND_PP: BF16→FP4 → vcvt(dst,src,mask,ROUND,PART_PP,MODE)
    // S4 (INT4) special instructions: vcvt_s42f16, vcvt_s42bf16, vcvt_s42s16, vcvt_f162s4
    if (src_dtype == DataType::INT4 &&
        (dst_dtype == DataType::FP16 || dst_dtype == DataType::BF16 || dst_dtype == DataType::INT16)) {
        is_s4_widening = true;
    } else if (dst_dtype == DataType::INT4 && src_dtype == DataType::FP16) {
        is_s4_narrowing = true;
    } else if (dst_dtype == DataType::INT4 && src_dtype == DataType::INT16) {
        is_s16_to_s4 = true;
    }
    // Widening cases: vcvt(dst, src, mask, PART, MODE) — no ROUND/RS
    // FP16→FP32, BF16→FP32, INT16→INT32, UINT16→UINT32, INT8→INT16, UINT8→UINT16,
    // INT8/UINT8→FP16, S32→S64, INT16/UINT16→FP32 (__VF_VCVTIF_PART)
    // Cross-sign s→u 2x widening: INT16→UINT32 (AscendC layoutMerge partCondition supports this)
    // NOTE: Cross-sign u→s widening (u8→s16, u16→s32) and s8→u16 are NOT supported by hardware.
    // NOTE: u32→s64 is NOT supported (only s32→s64 is).
    else if ((src_dtype == DataType::FP16 && dst_dtype == DataType::FP32) ||
             (src_dtype == DataType::BF16 && dst_dtype == DataType::FP32) ||
             (src_dtype == DataType::UINT16 && dst_dtype == DataType::UINT32) ||
             (src_dtype == DataType::INT16 && dst_dtype == DataType::INT32) ||
             (src_dtype == DataType::INT16 && dst_dtype == DataType::UINT32) ||
             (src_dtype == DataType::INT8 && dst_dtype == DataType::INT16) ||
             (src_dtype == DataType::UINT8 && dst_dtype == DataType::UINT16) ||
             (src_dtype == DataType::INT8 && dst_dtype == DataType::FP16) ||
             (src_dtype == DataType::UINT8 && dst_dtype == DataType::FP16) ||
             (src_dtype == DataType::INT32 && dst_dtype == DataType::INT64) ||
             (src_dtype == DataType::INT16 && dst_dtype == DataType::FP32) ||
             (src_dtype == DataType::UINT16 && dst_dtype == DataType::FP32) ||
             (src_dtype == DataType::HF8 && dst_dtype == DataType::FP16)) {
        is_widening = true;
    }
    // INT→FLOAT same-width or FLOAT→FLOAT same-width: vcvt(dst, src, mask, ROUND, MODE_ZEROING)
    // S32/U32→FP32, S16/U16→FP16, FP16→BF16, BF16→FP16
    else if (((src_dtype == DataType::INT32 || src_dtype == DataType::UINT32) && dst_dtype == DataType::FP32) ||
             ((src_dtype == DataType::INT16 || src_dtype == DataType::UINT16) && dst_dtype == DataType::FP16) ||
             (src_dtype == DataType::FP16 && dst_dtype == DataType::BF16) ||
             (src_dtype == DataType::BF16 && dst_dtype == DataType::FP16)) {
        is_int_to_float = true;
    }
    // FLOAT→same-width-INT: vcvt(dst, src, mask, ROUND, RS, MODE_ZEROING) — no PART
    // FP32→S32/U32, FP16→S16/U16 (same element width)
    else if ((src_dtype == DataType::FP32 && (dst_dtype == DataType::INT32 || dst_dtype == DataType::UINT32)) ||
             (src_dtype == DataType::FP16 && (dst_dtype == DataType::INT16 || dst_dtype == DataType::UINT16))) {
        is_float_to_same_int = true;
    }
    // FLOAT→narrower-INT (2x narrowing): vcvt(dst, src, mask, ROUND, RS, PART, MODE_ZEROING)
    // FP16→INT8/UINT8, BF16→INT8/UINT8 (b16→b8, uses __VF_VCVTFI_SAT_PART with PART_EVEN/ODD)
    else if ((src_dtype == DataType::FP16 || src_dtype == DataType::BF16) &&
             (dst_dtype == DataType::INT8 || dst_dtype == DataType::UINT8)) {
        is_float_to_narrower_int = true;
    }
    // FLOAT→wider-INT: vcvt(dst, src, mask, ROUND, PART, MODE_ZEROING) — no RS
    // FP16→S32/U32, BF16→S32/U32 (half-width float to full-width int)
    else if ((src_dtype == DataType::FP16 && (dst_dtype == DataType::INT32 || dst_dtype == DataType::UINT32)) ||
             (src_dtype == DataType::BF16 && (dst_dtype == DataType::INT32 || dst_dtype == DataType::UINT32))) {
        is_float_to_wider_int = true;
    }
    // Cross-width INT↔FLOAT: vcvt(dst, src, mask, ROUND, PART, MODE_ZEROING)
    // S32/U32→FP16, FP32→S64, S64→FP32
    // (S16/U16→FP32 moved to is_widening — uses __VF_VCVTIF_PART with PART+MODE only)
    else if (((src_dtype == DataType::INT32 || src_dtype == DataType::UINT32) && dst_dtype == DataType::FP16) ||
             (src_dtype == DataType::FP32 && dst_dtype == DataType::INT64) ||
             (src_dtype == DataType::INT64 && dst_dtype == DataType::FP32)) {
        is_cross_width = true;
    }
    // INT→INT 4x narrowing to UINT8 (b32→u8): vcvt(dst, src, mask, RS, PART_PP, MODE) — 6 args
    // __VF_VCVTII_SAT_PP: uses PART_P0/P1/P2/P3, supports RS parameter
    else if ((src_dtype == DataType::INT32 || src_dtype == DataType::UINT32) && dst_dtype == DataType::UINT8) {
        is_int_4x_narrowing = true;
    }
    // INT→INT 2x narrowing to UINT8/INT16/UINT16 (b16→u8, b32→b16, b64→b32): vcvt(dst, src, mask, RS, PART, MODE) — 6
    // args
    // __VF_VCVTII_SAT_PART: uses PART_EVEN/ODD, supports RS parameter
    else if (((src_dtype == DataType::INT32 || src_dtype == DataType::UINT32) &&
              (dst_dtype == DataType::INT16 || dst_dtype == DataType::UINT16)) ||
             ((src_dtype == DataType::INT16 || src_dtype == DataType::UINT16) && dst_dtype == DataType::UINT8) ||
             (src_dtype == DataType::INT64 && (dst_dtype == DataType::INT32 || dst_dtype == DataType::UINT32))) {
        is_int_2x_narrowing = true;
    }
    // INT→INT 4x widening (b8→b32): vcvt(dst, src, mask, PART_PP, MODE_ZEROING)
    // __VF_VCVTII_PP: uses PART_P0/P1/P2/P3
    // NOTE: Cross-sign 4x widening (s8→u32, u8→s32) is NOT supported by hardware.
    else if ((src_dtype == DataType::INT8 && dst_dtype == DataType::INT32) ||
             (src_dtype == DataType::UINT8 && dst_dtype == DataType::UINT32)) {
        is_int_4x_widening = true;
    }
    // FP8/FP4 4x widening (PP mode): FP8→FP32, FP4→BF16
    // __VF_VCVTFF_PP: vcvt(dst, src, mask, PART_PP, MODE_ZEROING) — 5 args
    else if (((src_dtype == DataType::FP8E4M3FN || src_dtype == DataType::FP8E5M2 || src_dtype == DataType::HF8) &&
              dst_dtype == DataType::FP32) ||
             ((src_dtype == DataType::FP4E2M1 || src_dtype == DataType::FP4E1M2) && dst_dtype == DataType::BF16)) {
        is_fp_widen_pp = true;
    }
    // FP8/FP4 4x narrowing with SAT (RND_SAT_PP): FP32→FP8
    // __VF_VCVTFF_RND_SAT_PP: vcvt(dst, src, mask, ROUND, SAT) — 5 args
    else if (src_dtype == DataType::FP32 &&
             (dst_dtype == DataType::FP8E4M3FN || dst_dtype == DataType::FP8E5M2 || dst_dtype == DataType::HF8)) {
        is_fp_narrow_rnd_sat_pp = true;
    }
    // FP4 4x narrowing without SAT (RND_PP): BF16→FP4
    // __VF_VCVTFF_RND_PP: vcvt(dst, src, mask, ROUND, PART_PP, MODE_ZEROING) — 6 args
    else if (src_dtype == DataType::BF16 && (dst_dtype == DataType::FP4E2M1 || dst_dtype == DataType::FP4E1M2)) {
        is_fp_narrow_rnd_pp = true;
    }
    // Remaining int→int paths not directly supported by hardware vcvt overloads:
    // cross-sign widening (u→s, s→u) and int→INT8 narrowing.
    // Route through f16 intermediate (two-step), mirroring AscendC's s16→f16→s4 approach.
    if (src_dtype.IsInt() && dst_dtype.IsInt() && !is_s4_widening && !is_s4_narrowing && !is_s16_to_s4 &&
        !is_widening && !is_int_4x_widening && !is_int_4x_narrowing && !is_int_2x_narrowing) {
        is_int_int_two_step = true;
    }

    // Validate round_mode against the specific vcvt overload constraints:
    // - Widening (no precision loss): round_mode is ignored (UNKNOWN), warn if set
    // - is_int_to_float / is_float_to_same_int: R/A/F/C/Z only (no O/H)
    // - is_fp_narrow_rnd_sat_pp (FP32→FP8E4M3FN/FP8E5M2): only ROUND_R
    // - is_fp_narrow_rnd_sat_pp (FP32→HF8): only ROUND_A/ROUND_H
    // - is_fp_narrow_rnd_pp (BF16→FP4): R/A/F/C/Z only (no O/H)
    // - FP16→HF8 (else/fallback path): only ROUND_A/ROUND_H
    // - is_fp_widen_pp (FP8→FP32, FP4→BF16): no round_mode (UNKNOWN)
    // - is_s4_widening (INT4→FP16/BF16/INT16): no round_mode (UNKNOWN)
    // - is_int_2x_narrowing / is_int_4x_narrowing / is_int_4x_widening / is_s16_to_s4: no round_mode
    if (is_widening || is_fp_widen_pp || is_s4_widening || is_s16_to_s4 || is_int_2x_narrowing || is_int_4x_narrowing ||
        is_int_4x_widening) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("round_mode") || round_mode == "CAST_RINT")
            << "vf.astype: round_mode is not applicable for this widening/no-precision-loss "
            << "conversion path (src=" << DTypeStr(src_dtype) << ", dst=" << DTypeStr(dst_dtype)
            << "), only default CAST_RINT is accepted";
    }
    if (is_int_to_float || is_float_to_same_int || is_float_to_narrower_int || is_fp_narrow_rnd_pp ||
        is_float_to_wider_int || is_cross_width || is_int_int_two_step) {
        PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, round != "ROUND_O" && round != "ROUND_H")
            << "vf.astype: round_mode CAST_ODD/CAST_HYBRID is not supported for this conversion path "
            << "(src=" << DTypeStr(src_dtype) << ", dst=" << DTypeStr(dst_dtype) << "), "
            << "supported values are CAST_RINT/CAST_ROUND/CAST_FLOOR/CAST_CEIL/CAST_TRUNC";
    }
    if (is_fp_narrow_rnd_sat_pp) {
        if (dst_dtype == DataType::HF8) {
            // FP32→HF8: only CAST_ROUND/CAST_HYBRID. Default CAST_RINT is invalid,
            // so if user didn't specify, silently use CAST_ROUND (the hardware default).
            if (!op->HasKwarg("round_mode")) {
                round = "ROUND_A";
            } else {
                PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, round == "ROUND_A" || round == "ROUND_H")
                    << "vf.astype: FP32→HF8 only supports round_mode CAST_ROUND/CAST_HYBRID, got " << round_mode;
            }
        } else {
            // FP32→FP8E4M3FN/FP8E5M2: only CAST_RINT. Default is CAST_RINT, so
            // not specifying round_mode is fine.
            if (op->HasKwarg("round_mode")) {
                PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, round == "ROUND_R")
                    << "vf.astype: FP32→FP8E4M3FN/FP8E5M2 only supports round_mode CAST_RINT, got " << round_mode;
            }
        }
    }
    if (src_dtype == DataType::FP16 && dst_dtype == DataType::HF8) {
        // FP16→HF8: only CAST_ROUND/CAST_HYBRID. Default CAST_RINT is invalid,
        // so if user didn't specify, silently use CAST_ROUND.
        if (!op->HasKwarg("round_mode")) {
            round = "ROUND_A";
        } else {
            PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, round == "ROUND_A" || round == "ROUND_H")
                << "vf.astype: FP16→HF8 only supports round_mode CAST_ROUND/CAST_HYBRID, got " << round_mode;
        }
    }
    // FP16→INT4 (is_s4_narrowing): R/A/F/C/Z only (no O/H)
    if (is_s4_narrowing) {
        PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, round != "ROUND_O" && round != "ROUND_H")
            << "vf.astype: round_mode CAST_ODD/CAST_HYBRID is not supported for FP16→INT4, "
            << "supported values are CAST_RINT/CAST_ROUND/CAST_FLOOR/CAST_CEIL/CAST_TRUNC";
    }
    // Validate saturate against conversion path:
    // - Widening / is_fp_widen_pp / is_s4_widening / is_int_4x_widening: saturate not applicable
    // - is_int_to_float (INT→FLOAT): saturate not applicable (default saturated, no choice)
    // - is_int_4x_narrowing (b32→u8): saturate is mandatory (always RS_ENABLE)
    // - is_int_2x_narrowing (b32→b16, b16→u8, b64→b32): saturate OFF/ON both supported
    // - is_s16_to_s4 (INT16→INT4): saturate OFF/ON both supported (like is_s4_narrowing)
    // - is_float_to_narrower_int (FP16→INT8/UINT8): saturate OFF/ON both supported
    // - is_float_to_same_int (FP32→S32): saturate OFF/ON both supported
    // - BF16→FP16 (float→float same-width): saturate OFF/ON both supported
    // - is_fp_narrow_rnd_sat_pp: saturate is mandatory (always RS_ENABLE)
    // - is_fp_narrow_rnd_pp (BF16→FP4): saturate not applicable (UNKNOWN)
    // - FP→FP32 (widening float): only OFF (non-saturated)
    if (is_widening || is_fp_widen_pp || is_s4_widening || is_int_4x_widening || is_fp_narrow_rnd_pp ||
        (is_int_to_float && !(src_dtype == DataType::BF16 && dst_dtype == DataType::FP16))) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("saturate") || sat == "RS_DISABLE")
            << "vf.astype: saturate is not applicable for this conversion path " << "(src=" << DTypeStr(src_dtype)
            << ", dst=" << DTypeStr(dst_dtype) << ")";
    }
    if (is_int_4x_narrowing) {
        // b32→b8 (4x narrowing): saturate is always enabled (RS_ENABLE). Default is RS_DISABLE,
        // so if user didn't specify, silently enable it.
        if (!op->HasKwarg("saturate")) {
            sat = "RS_ENABLE";
        } else {
            PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, sat == "RS_ENABLE")
                << "vf.astype: 4x int narrowing (b32→b8) requires saturate=ON (RS_ENABLE), "
                << "OFF is not supported for this path";
        }
    }
    if (is_fp_narrow_rnd_sat_pp) {
        // FP32→FP8: saturate is always enabled (RS_ENABLE). Default is RS_DISABLE,
        // so if user didn't specify, silently enable it.
        if (!op->HasKwarg("saturate")) {
            sat = "RS_ENABLE";
        } else {
            PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, sat == "RS_ENABLE")
                << "vf.astype: FP32→FP8 conversion requires saturate=ON (RS_ENABLE), "
                << "OFF is not supported for this path";
        }
    }
    if (dst_dtype == DataType::FP32 && src_dtype.GetBit() < 32 && (src_dtype.IsFloat() || src_dtype == DataType::HF8)) {
        // FP→FP32 widening: only non-saturated mode
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, sat == "RS_DISABLE")
            << "vf.astype: conversion to FP32 only supports saturate=OFF (non-saturated mode)";
    }
    // Validate layout against conversion path:
    // - Same-width conversions (is_int_to_float, is_float_to_same_int): layout not applicable
    // - FP16→BF16 and BF16→FP16 (same-width float→float): layout not applicable
    // - FP32→INT64 and INT64→FP32: same 64-bit width, layout not applicable
    if (is_int_to_float || is_float_to_same_int || (src_dtype == DataType::FP16 && dst_dtype == DataType::BF16) ||
        (src_dtype == DataType::BF16 && dst_dtype == DataType::FP16) ||
        (src_dtype == DataType::FP32 && dst_dtype == DataType::INT64) ||
        (src_dtype == DataType::INT64 && dst_dtype == DataType::FP32)) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("layout") || layout == "ZERO")
            << "vf.astype: layout is not applicable for this same-width conversion path "
            << "(src=" << DTypeStr(src_dtype) << ", dst=" << DTypeStr(dst_dtype) << ")";
    }
    // FP8/FP4 widening: layout supports ZERO/ONE/TWO/THREE (4x expansion)
    // FP8/FP4 2x widening (HF8→FP16): layout supports ZERO/ONE only
    if (is_fp_widen_pp) {
        if (src_dtype == DataType::HF8 && dst_dtype == DataType::FP16) {
            PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, layout == "ZERO" || layout == "ONE")
                << "vf.astype: HF8→FP16 only supports layout ZERO/ONE, got " << layout;
        } else if (src_dtype == DataType::FP8E8M0 && dst_dtype == DataType::BF16) {
            PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, layout == "ZERO" || layout == "ONE")
                << "vf.astype: FP8E8M0→BF16 only supports layout ZERO/ONE, got " << layout;
        }
        // FP8E4M3FN/FP8E5M2→FP32 and FP4→BF16 support all four layouts
    }
    // 2x widening (is_widening): layout supports ZERO/ONE only
    if (is_widening) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, layout == "ZERO" || layout == "ONE")
            << "vf.astype: 2x widening conversion only supports layout ZERO/ONE, got " << layout
            << " (src=" << DTypeStr(src_dtype) << ", dst=" << DTypeStr(dst_dtype) << ")";
    }
    // 4x int widening (b8→b32): layout supports ZERO/ONE/TWO/THREE
    if (is_int_4x_widening) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT,
                          layout == "ZERO" || layout == "ONE" || layout == "TWO" || layout == "THREE")
            << "vf.astype: 4x int widening conversion only supports layout ZERO/ONE/TWO/THREE, got " << layout
            << " (src=" << DTypeStr(src_dtype) << ", dst=" << DTypeStr(dst_dtype) << ")";
    }
    // 4x int narrowing (b32→u8) and INT16→INT4: layout supports ZERO/ONE/TWO/THREE
    if (is_int_4x_narrowing || is_s16_to_s4) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT,
                          layout == "ZERO" || layout == "ONE" || layout == "TWO" || layout == "THREE")
            << "vf.astype: 4x int narrowing conversion only supports layout ZERO/ONE/TWO/THREE, got " << layout
            << " (src=" << DTypeStr(src_dtype) << ", dst=" << DTypeStr(dst_dtype) << ")";
    }
    // Float narrowing with PART (partCondition in AscendC): layout supports ZERO/ONE only.
    // This covers FP32→FP16, FP32→BF16, FP16→HF8, FP16→UINT8, FP16→INT8, BF16→INT32, FP32→INT16,
    // FP32→INT64 (all use vcvt(dst,src,mask,ROUND,SAT,PART,MODE) — 7 args).
    // AscendC: static_assert(SupportEnum<layoutMode, RegLayout::ZERO, RegLayout::ONE>());
    if (!is_s4_widening && !is_s4_narrowing && !is_s16_to_s4 && !is_widening && !is_fp_widen_pp && !is_int_to_float &&
        !is_float_to_same_int && !is_float_to_narrower_int && !is_float_to_wider_int && !is_cross_width &&
        !is_int_2x_narrowing && !is_int_int_two_step && !is_int_4x_narrowing && !is_int_4x_widening &&
        !is_fp_narrow_rnd_sat_pp && !is_fp_narrow_rnd_pp &&
        !(src_dtype == DataType::FP16 && dst_dtype == DataType::BF16) &&
        !(src_dtype == DataType::BF16 && dst_dtype == DataType::FP16) &&
        !(src_dtype == DataType::FP32 && dst_dtype == DataType::INT64) &&
        !(src_dtype == DataType::INT64 && dst_dtype == DataType::FP32)) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, layout == "ZERO" || layout == "ONE")
            << "vf.astype: layout only supports ZERO/ONE for this conversion path " << "(src=" << DTypeStr(src_dtype)
            << ", dst=" << DTypeStr(dst_dtype) << "), got " << layout;
        // Round mode validation for fallback (partCondition) paths:
        // FP32→FP16: supports CAST_ODD but NOT CAST_HYBRID
        if (src_dtype == DataType::FP32 && dst_dtype == DataType::FP16) {
            PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, round != "ROUND_H")
                << "vf.astype: FP32→FP16 does not support round_mode CAST_HYBRID, "
                << "supported values are CAST_RINT/CAST_ROUND/CAST_FLOOR/CAST_CEIL/CAST_TRUNC/CAST_ODD";
        } else {
            // All other fallback paths (FP32→BF16, FP16→INT8, FP16→UINT8, BF16→FP16, FP32→INT16, etc.):
            // support CAST_RINT/CAST_ROUND/CAST_FLOOR/CAST_CEIL/CAST_TRUNC only (no CAST_ODD/CAST_HYBRID)
            PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, round != "ROUND_O" && round != "ROUND_H")
                << "vf.astype: round_mode CAST_ODD/CAST_HYBRID is not supported for this conversion path "
                << "(src=" << DTypeStr(src_dtype) << ", dst=" << DTypeStr(dst_dtype) << "), "
                << "supported values are CAST_RINT/CAST_ROUND/CAST_FLOOR/CAST_CEIL/CAST_TRUNC";
        }
    }
    if (is_s4_widening) {
        // INT4→FP16/BF16/INT16: specialized vcvt_s42*
        if (dst_dtype == DataType::FP16) {
            codegen.Emit("vcvt_s42f16(" + dst + ", " + src + ", " + mask + ", " + part_pp + ", " + mode_value + ");");
        } else if (dst_dtype == DataType::BF16) {
            codegen.Emit("vcvt_s42bf16(" + dst + ", " + src + ", " + mask + ", " + part_pp + ", " + mode_value + ");");
        } else {
            codegen.Emit("vcvt_s42s16(" + dst + ", " + src + ", " + mask + ", " + part_pp + ", " + mode_value + ");");
        }
    } else if (is_s4_narrowing) {
        // FP16→INT4: vcvt_f162s4
        codegen.Emit("vcvt_f162s4(" + dst + ", " + src + ", " + mask + ", " + round + ", " + sat + ", " + part_pp +
                     ", " + mode_value + ");");
    } else if (is_widening) {
        // vcvt(dst, src, mask, PART, MODE_ZEROING)
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + part + ", " + mode_value + ");");
    } else if (is_int_to_float) {
        // vcvt(dst, src, mask, ROUND, MODE_ZEROING)
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + round + ", " + mode_value + ");");
    } else if (is_float_to_same_int) {
        // vcvt(dst, src, mask, ROUND, RS, MODE_ZEROING) — no PART
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + round + ", " + sat + ", " + mode_value + ");");
    } else if (src_dtype == DataType::FP32 && dst_dtype == DataType::INT64) {
        // FP32→INT64: __VF_VCVTFI_SAT_PART(f32, s64) — the overload takes
        // (dst, src, mask, ROUND, RS, PART, MODE): the 5th arg is the
        // rounding-saturation flag and the 6th is PART_EVEN/PART_ODD (mirrors
        // AscendC CastImpl: vcvt(..., round, sat, part, mode), 7 args).
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, layout == "ZERO" || layout == "ONE")
            << "vf.astype FP32->INT64 only supports layout ZERO/ONE, got " << layout;
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + round + ", " + sat + ", " + part + ", " +
                     mode_value + ");");
    } else if (is_float_to_wider_int || is_cross_width) {
        // vcvt(dst, src, mask, ROUND, PART, MODE_ZEROING) — no RS
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + round + ", " + part + ", " + mode_value + ");");
    } else if (is_float_to_narrower_int) {
        // FLOAT→narrower-INT (2x narrowing): FP16→INT8/UINT8, BF16→INT8/UINT8
        // AscendC CastOperator<T,U,RoundMode,SatMode,RegLayout,MaskMergeMode> emits:
        //   vcvt(dstReg, srcReg, mask, roundModeValue, satModeValue, partModeValue, modeValue)
        // satMode is passed through directly (NO_SAT → RS_DISABLE, SAT → RS_ENABLE).
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + round + ", " + sat + ", " + part + ", " +
                     mode_value + ");");
    } else if (is_int_4x_widening) {
        // 4x int widening (b8→b32): vcvt(dst, src, mask, PART_PP, MODE_ZEROING) — 5 args
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + part_pp + ", " + mode_value + ");");
    } else if (is_int_4x_narrowing) {
        // 4x int narrowing to UINT8 (b32→u8): vcvt(dst, src, mask, RS, PART_PP, MODE) — 6 args
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + sat + ", " + part_pp + ", " + mode_value + ");");
    } else if (is_int_2x_narrowing) {
        // 2x int narrowing with RS (b32→b16, b16→u8, b64→b32): vcvt(dst, src, mask, RS, PART, MODE) — 6 args
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + sat + ", " + part + ", " + mode_value + ");");
    } else if (is_s16_to_s4) {
        // INT16→INT4: two-step conversion mirroring AscendC CastOperator (s16→f16→s4).
        // AscendC hardcodes RoundMode::CAST_RINT for both steps (round_mode not in trait).
        // Step 1: vcvt(tmp_f16, src_s16, mask, ROUND_R, MODE) — s16→f16 (5 args, rndMergeCast)
        // Step 2: vcvt_f162s4(dst_s4, tmp_f16, mask, ROUND_R, SAT, PART_PP, MODE) — f16→s4 (7 args)
        std::string tmp = dst + "_f16_tmp";
        codegen.HoistRegTensorDecl("RegTensor<half> " + tmp + ";");
        codegen.Emit("vcvt(" + tmp + ", " + src + ", " + mask + ", ROUND_R, " + mode_value + ");");
        codegen.Emit("vcvt_f162s4(" + dst + ", " + tmp + ", " + mask + ", ROUND_R, " + sat + ", " + part_pp + ", " +
                     mode_value + ");");
    } else if (is_int_int_two_step) {
        // Multi-step int→int for paths not directly supported by hardware vcvt overloads.
        // Available int→float overloads: b8→f16, s16→f16, s16→f32, s32→f32
        // NOT available: u16→f16, u16→f32, u32→f32, s32→f16
        //
        // Strategy by src/dst type combination:
        // 1. src→f16 available (b8, s16): two-step src→f16→dst
        // 2. s32/u32 src + b8 dst: three-step src→f32→f16→dst (f32→f16 narrowing needed)
        // 3. u16 src + b8 dst: four-step u16→u32→f32→f16→dst (u16→f16 not available)
        // 4. u16 src + b32 dst: u16→u32 widening + dst cast (zero-extend, semantically correct)
        // 5. same-width cross-sign (u16→s16, s32→u32, u32→s32): reinterpret cast, no instruction
        std::string dst_c_type = dst_dtype.ToCTypeString();
        bool src_to_f16_ok = (src_dtype.GetBit() == 8) || (src_dtype == DataType::INT16);
        bool src_is_b32 = (src_dtype == DataType::INT32 || src_dtype == DataType::UINT32);
        bool dst_is_b8 = (dst_dtype.GetBit() == 8);
        bool dst_is_b32 = (dst_dtype.GetBit() == 32);
        bool same_width = (src_dtype.GetBit() == dst_dtype.GetBit());

        if (src_to_f16_ok) {
            // Two-step: src→f16→dst
            std::string tmp = dst + "_f16_tmp";
            codegen.HoistRegTensorDecl("RegTensor<half> " + tmp + ";");
            EmitIntToFloatVcvt(codegen, tmp, src, mask, part, round, mode_value, src_dtype);
            EmitFloatToIntVcvt(codegen, dst, tmp, mask, part, round, sat, mode_value, dst_dtype);
        } else if (src_is_b32 && dst_is_b8) {
            // Three-step: src→f32→f16→dst (s32→f32→f16→s8, u32 reinterpreted as s32 first)
            std::string tmp_f32 = dst + "_f32_tmp";
            std::string tmp_f16 = dst + "_f16_tmp";
            codegen.HoistRegTensorDecl("RegTensor<float> " + tmp_f32 + ";");
            codegen.HoistRegTensorDecl("RegTensor<half> " + tmp_f16 + ";");
            EmitIntToFloatVcvt(codegen, tmp_f32, src, mask, part, round, mode_value, src_dtype);
            // f32→f16: float narrowing → vcvt(tmp_f16, tmp_f32, mask, ROUND, SAT, PART, MODE) — 7 args
            codegen.Emit("vcvt(" + tmp_f16 + ", " + tmp_f32 + ", " + mask + ", " + round + ", " + sat + ", " + part +
                         ", " + mode_value + ");");
            EmitFloatToIntVcvt(codegen, dst, tmp_f16, mask, part, round, sat, mode_value, dst_dtype);
        } else if (src_dtype == DataType::UINT16 && dst_is_b8) {
            // Four-step: u16→u32→f32→f16→dst (u16→f16 not available, so widen to u32 first)
            std::string tmp_u32 = dst + "_u32_tmp";
            std::string tmp_f32 = dst + "_f32_tmp";
            std::string tmp_f16 = dst + "_f16_tmp";
            codegen.HoistRegTensorDecl("RegTensor<uint32_t> " + tmp_u32 + ";");
            codegen.HoistRegTensorDecl("RegTensor<float> " + tmp_f32 + ";");
            codegen.HoistRegTensorDecl("RegTensor<half> " + tmp_f16 + ";");
            // u16→u32: widening → vcvt(tmp_u32, src, mask, PART, MODE) — 5 args
            codegen.Emit("vcvt(" + tmp_u32 + ", " + src + ", " + mask + ", " + part + ", " + mode_value + ");");
            EmitIntToFloatVcvt(codegen, tmp_f32, tmp_u32, mask, part, round, mode_value, DataType::UINT32);
            // f32→f16: float narrowing → vcvt(tmp_f16, tmp_f32, mask, ROUND, SAT, PART, MODE) — 7 args
            codegen.Emit("vcvt(" + tmp_f16 + ", " + tmp_f32 + ", " + mask + ", " + round + ", " + sat + ", " + part +
                         ", " + mode_value + ");");
            EmitFloatToIntVcvt(codegen, dst, tmp_f16, mask, part, round, sat, mode_value, dst_dtype);
        } else if (src_dtype == DataType::UINT16 && dst_is_b32) {
            // u16→s32: u16→u32 widening with dst cast (zero-extend preserves value)
            std::string u32_c_type = DataType::UINT32.ToCTypeString();
            codegen.Emit("vcvt((RegTensor<" + u32_c_type + "> &)" + dst + ", " + src + ", " + mask + ", " + part +
                         ", " + mode_value + ");");
        } else if (same_width && src_dtype != dst_dtype) {
            // Same-width cross-sign (u16→s16, s32→u32, u32→s32): reinterpret cast
            codegen.Emit(dst + " = (RegTensor<" + dst_c_type + "> &)" + src + ";");
        }
    } else if (is_fp_widen_pp) {
        // FP8/FP4 4x widening PP: vcvt(dst, src, mask, PART_PP, MODE_ZEROING)
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + part_pp + ", " + mode_value + ");");
    } else if (is_fp_narrow_rnd_sat_pp) {
        // FP32→FP8 4x narrowing RND_SAT_PP: vcvt(dst, src, mask, ROUND, SAT, PART_PP, MODE_ZEROING)
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + round + ", " + sat + ", " + part_pp + ", " +
                     mode_value + ");");
    } else if (is_fp_narrow_rnd_pp) {
        // BF16→FP4 4x narrowing RND_PP: vcvt(dst, src, mask, ROUND, PART_PP, MODE_ZEROING)
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + round + ", " + part_pp + ", " + mode_value +
                     ");");
    } else {
        // Float narrowing (FP32→FP16, FP32→BF16, FP16→INT8, etc.)
        // vcvt(dst, src, mask, ROUND, RS, PART, MODE_ZEROING)
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + round + ", " + sat + ", " + part + ", " +
                     mode_value + ");");
    }
    return "";
}

// ============================================================================
// DeInterleave — vdintlv
// ============================================================================

static std::string EmitVFDeInterleave(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // Parser args order: [dst0, dst1, src0, src1]
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.de_interleave requires 4 args (dst0, dst1, src0, src1)";
    std::string dst0 = codegen.GetExprAsCode(op->args_[0]);
    std::string dst1 = codegen.GetExprAsCode(op->args_[1]);
    std::string src0 = codegen.GetExprAsCode(op->args_[2]);
    std::string src1 = codegen.GetExprAsCode(op->args_[3]);
    if (IsDstMaskReg(op, codegen)) {
        DataType dtype = DataType::FP32;
        if (op->HasKwarg("dtype")) {
            dtype = op->GetKwarg<DataType>("dtype");
        }
        // MaskDeInterleave: only supports b8/b16/b32 (not b64)
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                          dtype.GetBit() == 8 || dtype.GetBit() == 16 || dtype.GetBit() == 32)
            << "vf.de_interleave (MaskReg) only supports b8/b16/b32, got " << DTypeStr(dtype);
        std::string pdintlv_op;
        if (dtype == DataType::UINT8 || dtype == DataType::INT8) {
            pdintlv_op = "pdintlv_b8";
        } else if (dtype.GetBit() == 16) {
            pdintlv_op = "pdintlv_b16";
        } else {
            pdintlv_op = "pdintlv_b32";
        }
        codegen.Emit(pdintlv_op + "(" + dst0 + ", " + dst1 + ", " + src0 + ", " + src1 + ");");
        return "";
    }
    // DeInterleave requires all four operands to have the same dtype (mirrors
    // AscendC DeInterleaveImpl: dst0/dst1/src0/src1 are one register type U and
    // vdintlv is keyed on that element type). Cross-dtype reinterpretation is
    // the caller's job — cast explicitly first, e.g. the
    // `(RegTensor<uint8_t>&)vreg0U16` pattern in vf_topk.h:60.
    DataType dst_dt = GetExprDtype(op->args_[0]);
    DataType dst1_dt = GetExprDtype(op->args_[1]);
    DataType s0_dt = GetExprDtype(op->args_[2]);
    DataType s1_dt = GetExprDtype(op->args_[3]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (IsB8Type(s0_dt) || s0_dt.GetBit() == 16 || s0_dt.GetBit() == 32 || s0_dt.GetBit() == 64))
        << "vf.de_interleave only supports b8/b16/b32/b64 types, got " << DTypeStr(s0_dt);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, s0_dt == s1_dt)
        << "vf.de_interleave requires src0 and src1 to have the same type, got src0=" << DTypeStr(s0_dt)
        << " src1=" << DTypeStr(s1_dt);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, s0_dt == dst_dt)
        << "vf.de_interleave requires dst0 and src0 to have the same type, got dst0=" << DTypeStr(dst_dt)
        << " src0=" << DTypeStr(s0_dt);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, dst1_dt == dst_dt)
        << "vf.de_interleave requires dst0 and dst1 to have the same type, got dst0=" << DTypeStr(dst_dt)
        << " dst1=" << DTypeStr(dst1_dt);
    if (dst_dt.GetBit() == 64) {
        codegen.Emit("vdintlv((RegTensor<uint32_t>&)" + dst0 + ", (RegTensor<uint32_t>&)" + dst1 +
                     ", (RegTensor<uint32_t>&)" + src0 + ", (RegTensor<uint32_t>&)" + src1 + ");");
    } else {
        codegen.Emit("vdintlv(" + dst0 + ", " + dst1 + ", " + src0 + ", " + src1 + ");");
    }
    return "";
}

// ============================================================================
// Select — vsel
// ============================================================================

static std::string EmitVFSelect(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // Parser args order: [dst, src_true, src_false, mask]
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.select requires 4 args (dst, src_true, src_false, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src_true = codegen.GetExprAsCode(op->args_[1]);
    std::string src_false = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    if (IsDstMaskReg(op, codegen)) {
        codegen.Emit("psel(" + dst + ", " + src_true + ", " + src_false + ", " + mask + ");");
        return "";
    }
    // Doc: select supports BOOL/INT8/UINT8/INT16/UINT16/FP16/BF16/INT32/UINT32/FP32/INT64/UINT64
    DataType dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      IsB8Type(dst_dt) || dst_dt.GetBit() == 16 || dst_dt.GetBit() == 32 || dst_dt.GetBit() == 64)
        << "vf.select only supports b8/b16/b32/b64 types, got " << DTypeStr(dst_dt);
    DataType st_dt = GetExprDtype(op->args_[1]);
    DataType sf_dt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      dst_dt.GetBit() == st_dt.GetBit() && dst_dt.GetBit() == sf_dt.GetBit())
        << "vf.select requires dst, src_true, src_false to have the same bit width, got dst=" << dst_dt.GetBit()
        << "-bit src_true=" << st_dt.GetBit() << "-bit src_false=" << sf_dt.GetBit() << "-bit";
    if (dst_dt.GetBit() == 64) {
        // B64 select: deinterleave both sources, vsel on both b32 halves, interleave back.
        // Mirrors AscendC SelectImpl: B64TraitOneToTraitTwo + vsel(reg[0], reg[1]) + B64TraitTwoToTraitOne.
        std::string p = dst + "_sel_";
        EmitB64Deinterleave(codegen, p + "t", src_true);
        EmitB64Deinterleave(codegen, p + "f", src_false);
        std::string lot = p + "t_lo_", hit = p + "t_hi_";
        std::string lof = p + "f_lo_", hif = p + "f_hi_";
        // Pack b64 mask to b32 for vsel (mirrors MaskPack in AscendC)
        std::string packed_m = p + "_pm_";
        codegen.Emit("MaskReg " + packed_m + ";");
        codegen.Emit("ppack(" + packed_m + ", " + mask + ", LOWER);");
        std::string lo_dst = p + "_lod_";
        std::string hi_dst = p + "_hid_";
        codegen.Emit("RegTensor<uint32_t> " + lo_dst + ";");
        codegen.Emit("RegTensor<uint32_t> " + hi_dst + ";");
        codegen.Emit("vsel(" + lo_dst + ", " + lot + ", " + lof + ", " + packed_m + ");");
        codegen.Emit("vsel(" + hi_dst + ", " + hit + ", " + hif + ", " + packed_m + ");");
        EmitB64Interleave(codegen, dst, lo_dst, hi_dst, p + "_ilv");
        return "";
    }
    std::string cast_prefix = "(RegTensor<" + dst_dt.ToCTypeString() + "> &)";
    std::string st_expr = (st_dt == dst_dt) ? src_true : (cast_prefix + src_true);
    std::string sf_expr = (sf_dt == dst_dt) ? src_false : (cast_prefix + src_false);
    codegen.Emit("vsel(" + dst + ", " + st_expr + ", " + sf_expr + ", " + mask + ");");
    return "";
}

// ============================================================================
// UpdateMask — plt_b32/plt_b16
// ============================================================================

static std::string EmitVFUpdateMask(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 1)
        << "vf.update_mask requires 1 arg (scalar)";
    std::string scalar = codegen.GetExprAsCode(op->args_[0]);
    std::string reg_name = codegen.GetCurrentResultTarget();
    // Default to b32 (float), use dtype kwarg to select b16 or b8
    bool use_b8 = false;
    bool use_b16 = false;
    bool use_b64 = false;
    if (op->HasKwarg("dtype")) {
        auto dtype = op->GetKwarg<DataType>("dtype");
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                          IsB8Type(dtype) || dtype.GetBit() == 16 || dtype.GetBit() == 32 || dtype.GetBit() == 64)
            << "vf.update_mask dtype must be b8/b16/b32/b64, got " << DTypeStr(dtype);
        use_b8 = (dtype == DataType::UINT8 || dtype == DataType::INT8);
        use_b16 = (dtype.GetBit() == 16);
        use_b64 = (dtype.GetBit() == 64);
    }
    // plt_b32/plt_b16 requires uint32_t& reference, so declare a variable first
    std::string scalar_var = "_vf_mask_scalar_" + std::to_string(codegen.GetTileOffsetCounter());
    codegen.Emit("uint32_t " + scalar_var + " = (uint32_t)" + scalar + ";");
    codegen.Emit("MaskReg " + reg_name + ";");
    codegen.RegisterMaskRegVar(reg_name);
    if (use_b8) {
        codegen.Emit(reg_name + " = plt_b8(" + scalar_var + ", POST_UPDATE);");
    } else if (use_b16) {
        codegen.Emit(reg_name + " = plt_b16(" + scalar_var + ", POST_UPDATE);");
    } else {
        // b32 and b64 both use plt_b32; b64 additionally needs punpack to expand
        // each bit into a pair, matching the 2-bit-per-element mask width.
        codegen.Emit(reg_name + " = plt_b32(" + scalar_var + ", POST_UPDATE);");
        if (use_b64) {
            codegen.Emit("punpack(" + reg_name + ", " + reg_name + ", LOWER);");
        }
    }
    return "";
}

static std::string EmitVFHistograms(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // args: [dst, src, mask]
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "vf.histograms requires 3 args (dst, src, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    // vhist: src must be uint8_t, dst is derived from src and cast to uint16_t
    DataType src_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == DataType::UINT8)
        << "vf.histograms source only supports UINT8, got " << DTypeStr(src_dt);
    // Reinterpret src as RegTensor<uint8_t>& if its dtype isn't u8
    std::string bin_type = VFEnumValueName(ir::EnumToString(static_cast<ir::BinType>(op->GetKwarg<int>("bin_type"))));
    std::string bin_const = (bin_type == "BIN1") ? "Bin_N1" : "Bin_N0";
    std::string src_expr = (src_dt == DataType::UINT8) ? src : ("(RegTensor<uint8_t> &)" + src);
    // hist_type: ACCUMULATE (chistv2, default) or FREQUENCY (dhistv2)
    std::string hist_type = "ACCUMULATE";
    if (op->HasKwarg("hist_type")) {
        hist_type = VFEnumValueName(ir::EnumToString(static_cast<ir::HistType>(op->GetKwarg<int>("hist_type"))));
    }
    std::string dst_cast = "(RegTensor<uint16_t> &)";
    if (hist_type == "FREQUENCY") {
        codegen.Emit("dhistv2(" + dst_cast + dst + ", " + src_expr + ", " + mask + ", " + bin_const + ");");
    } else {
        codegen.Emit("chistv2(" + dst_cast + dst + ", " + src_expr + ", " + mask + ", " + bin_const + ");");
    }
    return "";
}

static std::string EmitVFCompareImpl(const ir::CallPtr& op, codegen::CodegenBase& codegen_base,
                                     const std::string& cmp_mode)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // args: [dst, src0, src1, mask_src]
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << op->name_ << " requires 4 args (dst, src0, src1, mask)";
    std::string mask_dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask_src = codegen.GetExprAsCode(op->args_[3]);
    // vcmp supports: u8,s8,u16,s16,u32,s32,half,float,bf16,u64,s64 (no bool, no FP8)
    DataType s0_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (IsArithIntType(s0_dt) || s0_dt == DataType::FP16 ||
                                                    s0_dt == DataType::FP32 || s0_dt == DataType::BF16))
        << op->name_ << " source only supports INT/UINT/FP16/FP32/BF16, got " << DTypeStr(s0_dt);
    DataType s1_dt = GetExprDtype(op->args_[2]);
    // FP8/FP4 types have no vcmp overloads (mirrors AscendC CompareImpl):
    // reject them for either compared operand.
    auto is_fp8_fp4 = [](DataType dt) {
        return dt == DataType::FP8E4M3FN || dt == DataType::FP8E5M2 || dt == DataType::FP8E8M0 || dt == DataType::HF8 ||
               dt == DataType::FP4 || dt == DataType::FP4E2M1 || dt == DataType::FP4E1M2 || dt == DataType::HF4;
    };
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, !is_fp8_fp4(s0_dt) && !is_fp8_fp4(s1_dt))
        << op->name_ << " does not support FP8/FP4 compare types, got src0=" << DTypeStr(s0_dt)
        << " src1=" << DTypeStr(s1_dt);
    // Bit width check: in scalar path, uses is_convertible (allows int64 scalar -> int32 reg)
    // so we defer the strict check to the vector-vector path only.
    // Here we just ensure s1 is not a wider type that can't convert (e.g. float vs int).
    bool is_scalar_src = true;
    auto src1_var = ir::As<ir::Var>(op->args_[2]);
    if (src1_var) {
        std::string src1_name = codegen.GetVarName(src1_var);
        is_scalar_src = !codegen.IsRegTensorVar(src1_name);
    }
    if (!is_scalar_src) {
        // AscendC CompareImpl takes both sources as the same register type U —
        // mixed dtypes (even equal bit width) are rejected, no reinterpret casts.
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, s0_dt == s1_dt)
            << op->name_ << " requires src0 and src1 to have the same type, got src0=" << DTypeStr(s0_dt)
            << " src1=" << DTypeStr(s1_dt);
    } else {
        // Scalar compare: is_convertible allows scalar to be wider than reg
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, s1_dt.GetBit() >= s0_dt.GetBit())
            << op->name_ << " scalar must be convertible to reg type, got src0=" << DTypeStr(s0_dt)
            << " scalar=" << DTypeStr(s1_dt);
    }
    std::string suffix = "eq";
    if (cmp_mode == "NE")
        suffix = "ne";
    else if (cmp_mode == "LT")
        suffix = "lt";
    else if (cmp_mode == "GT")
        suffix = "gt";
    else if (cmp_mode == "GE")
        suffix = "ge";
    else if (cmp_mode == "LE")
        suffix = "le";
    if (is_scalar_src) {
        src1 = CoerceScalarToInt(op->args_[2], s0_dt, src1);
        if (s0_dt.GetBit() == 64) {
            // B64 scalar compare: deinterleave src0, broadcast scalar halves, compare b32.
            // Must pack b64 mask to b32 (MaskPack) and unpack result (MaskUnPack),
            // matching AscendC CompareImpl for b64 RegTraitNumOne.
            std::string p = mask_dst + "_cmps_";
            std::string packed_m = p + "_pm_";
            codegen.Emit("MaskReg " + packed_m + ";");
            codegen.Emit("ppack(" + packed_m + ", " + mask_src + ", LOWER);");
            EmitB64Deinterleave(codegen, p + "s", src0);
            std::string lo_s = p + "s_lo_", hi_s = p + "s_hi_";
            std::string cast_type = (s0_dt == DataType::INT64) ? "(int64_t)" : "(uint64_t)";
            std::string lo_sc = p + "_losc_";
            std::string hi_sc = p + "_hisc_";
            codegen.Emit("RegTensor<uint32_t> " + lo_sc + ";");
            codegen.Emit("RegTensor<uint32_t> " + hi_sc + ";");
            codegen.Emit("vdup(" + lo_sc + ", (int32_t)(" + cast_type + "(" + src1 + ")), " + packed_m +
                         ", MODE_ZEROING);");
            codegen.Emit("vdup(" + hi_sc + ", (int32_t)((" + cast_type + "(" + src1 + ")) >> 32), " + packed_m +
                         ", MODE_ZEROING);");
            // Compare high halves
            std::string hi_r = p + "_hir_";
            std::string hi_eq = p + "_hieq_";
            codegen.Emit("MaskReg " + hi_r + ";");
            codegen.Emit("MaskReg " + hi_eq + ";");
            if (s0_dt == DataType::INT64) {
                codegen.Emit("vcmp_" + suffix + "(" + hi_r + ", (RegTensor<int32_t>&)" + hi_s +
                             ", (RegTensor<int32_t>&)" + hi_sc + ", " + packed_m + ");");
                codegen.Emit("vcmp_eq(" + hi_eq + ", (RegTensor<int32_t>&)" + hi_s + ", (RegTensor<int32_t>&)" + hi_sc +
                             ", " + packed_m + ");");
            } else {
                codegen.Emit("vcmp_" + suffix + "(" + hi_r + ", " + hi_s + ", " + hi_sc + ", " + packed_m + ");");
                codegen.Emit("vcmp_eq(" + hi_eq + ", " + hi_s + ", " + hi_sc + ", " + packed_m + ");");
            }
            // Compare low halves (always unsigned) for equality tie-break
            std::string lo_r = p + "_lor_";
            codegen.Emit("MaskReg " + lo_r + ";");
            codegen.Emit("vcmp_" + suffix + "(" + lo_r + ", " + lo_s + ", " + lo_sc + ", " + packed_m + ");");
            // For EQ: result = hi_eq AND lo_eq (lo_r is eq result)
            // For NE: result = NOT(hi_eq AND lo_eq)
            // For GT/GE/LT/LE: psel(lowCmp, highCmp, highEq) — if highEq pick lowCmp, else pick highCmp.
            // Matches AscendC CompareScalarImpl b64 path.
            if (cmp_mode == "EQ") {
                codegen.Emit("pand(" + mask_dst + ", " + hi_eq + ", " + lo_r + ", " + packed_m + ");");
            } else if (cmp_mode == "NE") {
                // NE: low_ne OR high_ne. lo_r is vcmp_ne(low), hi_r is vcmp_ne(high).
                codegen.Emit("por(" + mask_dst + ", " + lo_r + ", " + hi_r + ", " + packed_m + ");");
            } else {
                codegen.Emit("psel(" + mask_dst + ", " + lo_r + ", " + hi_r + ", " + hi_eq + ");");
            }
            codegen.Emit("punpack(" + mask_dst + ", " + mask_dst + ", LOWER);");
        } else {
            codegen.Emit("vcmps_" + suffix + "(" + mask_dst + ", " + src0 + ", " + src1 + ", " + mask_src + ");");
        }
    } else {
        if (s0_dt.GetBit() == 64) {
            // B64 vector compare: deinterleave both sources, compare b32 halves.
            // Must pack b64 mask to b32 (MaskPack) and unpack result (MaskUnPack),
            // matching AscendC CompareImpl for b64 RegTraitNumOne.
            std::string p = mask_dst + "_cmpv_";
            std::string packed_m = p + "_pm_";
            codegen.Emit("MaskReg " + packed_m + ";");
            codegen.Emit("ppack(" + packed_m + ", " + mask_src + ", LOWER);");
            EmitB64Deinterleave(codegen, p + "s0", src0);
            EmitB64Deinterleave(codegen, p + "s1", src1);
            std::string lo0 = p + "s0_lo_", hi0 = p + "s0_hi_";
            std::string lo1 = p + "s1_lo_", hi1 = p + "s1_hi_";
            std::string hi_r = p + "_hir_";
            std::string hi_eq = p + "_hieq_";
            codegen.Emit("MaskReg " + hi_r + ";");
            codegen.Emit("MaskReg " + hi_eq + ";");
            if (s0_dt == DataType::INT64) {
                codegen.Emit("vcmp_" + suffix + "(" + hi_r + ", (RegTensor<int32_t>&)" + hi0 +
                             ", (RegTensor<int32_t>&)" + hi1 + ", " + packed_m + ");");
                codegen.Emit("vcmp_eq(" + hi_eq + ", (RegTensor<int32_t>&)" + hi0 + ", (RegTensor<int32_t>&)" + hi1 +
                             ", " + packed_m + ");");
            } else {
                codegen.Emit("vcmp_" + suffix + "(" + hi_r + ", " + hi0 + ", " + hi1 + ", " + packed_m + ");");
                codegen.Emit("vcmp_eq(" + hi_eq + ", " + hi0 + ", " + hi1 + ", " + packed_m + ");");
            }
            std::string lo_r = p + "_lor_";
            codegen.Emit("MaskReg " + lo_r + ";");
            codegen.Emit("vcmp_" + suffix + "(" + lo_r + ", " + lo0 + ", " + lo1 + ", " + packed_m + ");");
            if (cmp_mode == "EQ") {
                codegen.Emit("pand(" + mask_dst + ", " + hi_eq + ", " + lo_r + ", " + packed_m + ");");
            } else if (cmp_mode == "NE") {
                // NE: low_ne OR high_ne. lo_r is vcmp_ne(low), hi_r is vcmp_ne(high).
                codegen.Emit("por(" + mask_dst + ", " + lo_r + ", " + hi_r + ", " + packed_m + ");");
            } else {
                // GT/GE/LT/LE: psel(lowCmp, highCmp, highEq) — if highEq pick lowCmp, else pick highCmp.
                // Matches AscendC CompareInt64Impl/CompareUint64Impl.
                codegen.Emit("psel(" + mask_dst + ", " + lo_r + ", " + hi_r + ", " + hi_eq + ");");
            }
            codegen.Emit("punpack(" + mask_dst + ", " + mask_dst + ", LOWER);");
        } else {
            // Same-type operands (enforced above): no reinterpret casts needed.
            codegen.Emit("vcmp_" + suffix + "(" + mask_dst + ", " + src0 + ", " + src1 + ", " + mask_src + ");");
        }
    }
    return "";
}

static std::string EmitVFSqueeze(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // Parser args order: [dst, src, mask]
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "vf.squeeze requires 3 args (dst, src, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    // vsqz requires dst & src to share the same vector element type. Reinterpret
    // src as RegTensor<dst-dtype>& if necessary (mirrors vf_topk.h pattern of
    // `(RegTensor<u32>&)idxC` before passing to Squeeze).
    DataType dst_dt = GetExprDtype(op->args_[0]);
    DataType src_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (src_dt == DataType::INT8 || src_dt == DataType::UINT8 || src_dt == DataType::INT16 ||
                       src_dt == DataType::UINT16 || src_dt == DataType::INT32 || src_dt == DataType::UINT32 ||
                       src_dt == DataType::FP16 || src_dt == DataType::FP32))
        << "vf.squeeze src only supports INT8/UINT8/INT16/UINT16/INT32/UINT32/FP16/FP32, got " << DTypeStr(src_dt);
    // Doc: dst supports the same type list as src
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (dst_dt == DataType::INT8 || dst_dt == DataType::UINT8 || dst_dt == DataType::INT16 ||
                       dst_dt == DataType::UINT16 || dst_dt == DataType::INT32 || dst_dt == DataType::UINT32 ||
                       dst_dt == DataType::FP16 || dst_dt == DataType::FP32))
        << "vf.squeeze dst only supports INT8/UINT8/INT16/UINT16/INT32/UINT32/FP16/FP32, got " << DTypeStr(dst_dt);
    std::string dst_expr = dst;
    std::string src_expr = src;
    if (dst_dt != src_dt) {
        src_expr = "(RegTensor<" + dst_dt.ToCTypeString() + "> &)" + src;
    }
    // mode kwarg: "STORED" (default) or "NO_STORED"
    // backward compat: gather_mode="STORE_REG" / "NO_STORE_REG"
    std::string mode = "MODE_STORED";
    if (op->HasKwarg("gather_mode")) {
        auto gm = static_cast<ir::SqueezeMode>(op->GetKwarg<int>("gather_mode"));
        if (gm == ir::SqueezeMode::NO_STORE_REG)
            mode = "MODE_NO_STORED";
        else
            mode = "MODE_STORED";
    }
    codegen.Emit("vsqz(" + dst_expr + ", " + src_expr + ", " + mask + ", " + mode + ");");
    return "";
}

static std::string EmitVFArange(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // args: [dst, start]
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2)
        << "vf.arange requires 2 args (dst, start)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string start = codegen.GetExprAsCode(op->args_[1]);
    DataType dst_dt = GetExprDtype(op->args_[0]);
    // vci: supports INT8/UINT8/INT16/UINT16/INT32/UINT32/INT64/FP16/FP32 (no UINT64)
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      dst_dt == DataType::INT8 || dst_dt == DataType::UINT8 || dst_dt == DataType::INT16 ||
                          dst_dt == DataType::UINT16 || dst_dt == DataType::INT32 || dst_dt == DataType::UINT32 ||
                          dst_dt == DataType::INT64 || dst_dt == DataType::FP16 || dst_dt == DataType::FP32)
        << "vf.arange only supports INT8/UINT8/INT16/UINT16/INT32/UINT32/INT64/FP16/FP32, got " << DTypeStr(dst_dt);
    // vci: scalarValue must be convertible to RegTensor data type
    // is_convertible<U, ActualT>() — start must be convertible to dst type.
    // This allows e.g. int64(index) start with int32 dst (narrowing conversion).
    // Only reject if start is smaller than dst (would lose data).
    // index_order kwarg selects the direction: INCREASE_ORDER (default) ->
    // dst[i] = start + i; DECREASE_ORDER -> dst[i] = start - i.
    bool is_decrease = false;
    if (op->HasKwarg("index_order")) {
        auto o = static_cast<ir::IndexOrder>(op->GetKwarg<int>("index_order"));
        if (o == ir::IndexOrder::DECREASE_ORDER)
            is_decrease = true;
    }
    // vci accepts signed integer types (int8/int16/int32) and float types (half/float).
    // Unsigned types (uint8/uint16/uint32) have no vci overload — cast to signed.
    // Signed and float types are passed directly (no cast needed).
    std::string elem_type = dst_dt.ToCTypeString();
    if (dst_dt == DataType::UINT8)
        elem_type = "int8_t";
    else if (dst_dt == DataType::UINT16)
        elem_type = "int16_t";
    else if (dst_dt == DataType::UINT32)
        elem_type = "int32_t";
    // b64 (INT64/UINT64): single vci does not support 8-byte elements.
    // Build the index in u32 halves, fold in the 64-bit start scalar with a
    // carry chain (mirrors EmitVFAdds' b64 scalar add: vdup the scalar halves,
    // vaddc/vaddcs), then vintlv the halves into dst:
    //   lo = i + (u32)start;  hi = (u32)(start >> 32) + carry
    // (b64 vadds on a single 256B RegTensor<int64_t> is not callable — the
    // bisheng overload takes vector_2xvl_s64 512B register pairs.)
    // NOTE: DECREASE_ORDER assumes non-negative results (no borrow from the
    // low half into the sign-extended high half).
    if (dst_dt == DataType::INT64 || dst_dt == DataType::UINT64) {
        std::string lo = dst + "_b64_lo_";
        std::string hi = dst + "_b64_hi_";
        std::string dump = dst + "_b64_dump_";
        std::string m = dst + "_b64_m_";
        std::string cast_type = (dst_dt == DataType::INT64) ? "(int64_t)" : "(uint64_t)";
        std::string b32_cast = (dst_dt == DataType::INT64) ? "(RegTensor<int32_t>&)" : "(RegTensor<uint32_t>&)";
        codegen.Emit("RegTensor<int32_t> " + lo + ";");
        codegen.Emit("RegTensor<int32_t> " + hi + ";");
        codegen.Emit("RegTensor<int32_t> " + dump + ";");
        codegen.Emit("MaskReg " + m + " = pset_b32(PAT_ALL);");
        codegen.Emit("vci(" + lo + ", 0, INC_ORDER);");
        if (is_decrease) {
            codegen.Emit("vneg(" + lo + ", " + lo + ", " + m + ", MODE_ZEROING);");
            // Sign-extend the negated low half: lane 0 keeps hi=0, lanes i>=1
            // extend to 0xFFFFFFFF so the pair holds the b64 value -i.
            codegen.Emit("vshrs(" + hi + ", " + lo + ", (int16_t)31, " + m + ", MODE_ZEROING);");
        } else {
            codegen.Emit("vdup(" + hi + ", 0, " + m + ", MODE_ZEROING);");
        }
        std::string sc_lo = dst + "_b64_sclo_";
        std::string sc_hi = dst + "_b64_schi_";
        std::string carry = dst + "_b64_carry_";
        codegen.Emit("RegTensor<uint32_t> " + sc_lo + ";");
        codegen.Emit("RegTensor<uint32_t> " + sc_hi + ";");
        codegen.Emit("MaskReg " + carry + ";");
        codegen.Emit("vdup(" + sc_lo + ", (int32_t)(" + cast_type + "(" + start + ")), " + m + ", MODE_ZEROING);");
        codegen.Emit("vdup(" + sc_hi + ", (int32_t)((" + cast_type + "(" + start + ")) >> 32), " + m +
                     ", MODE_ZEROING);");
        codegen.Emit("vaddc(" + carry + ", " + b32_cast + lo + ", " + b32_cast + lo + ", " + b32_cast + sc_lo + ", " +
                     m + ");");
        codegen.Emit("vaddcs(" + carry + ", " + b32_cast + hi + ", " + b32_cast + hi + ", " + b32_cast + sc_hi + ", " +
                     carry + ", " + m + ");");
        codegen.Emit("vintlv((RegTensor<uint32_t> &)" + dst + ", (RegTensor<uint32_t> &)" + dump +
                     ", (RegTensor<uint32_t> &)" + lo + ", (RegTensor<uint32_t> &)" + hi + ");");
        return "";
    }
    // Non-b64: vci INC_ORDER generates value, value+1, ..., value+VL-1.
    // vci DEC_ORDER generates value+VL-1, value+VL-2, ..., value.
    std::string order_str = is_decrease ? "DEC_ORDER" : "INC_ORDER";
    if (elem_type != dst_dt.ToCTypeString()) {
        codegen.Emit("vci((RegTensor<" + elem_type + "> &)" + dst + ", " + start + ", " + order_str + ");");
    } else {
        codegen.Emit("vci(" + dst + ", " + start + ", " + order_str + ");");
    }
    return "";
}

static std::string EmitVFGather(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // Two forms, dispatched by src argument type:
    //   Reg-to-Reg: args = [dst, src_reg, indices]  — no mask, src/dst same type
    //   UB-to-Reg:  args = [dst, src_ub, indices, mask] — with mask, b8 zero-extends to b16
    auto src_tile_type = ir::As<ir::TileType>(op->args_[1]->GetType());
    if (!src_tile_type) {
        // Reg-to-Reg form
        PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, op->args_.size() == 3)
            << "vf.gather (reg→reg) requires exactly 3 args (dst, src, indices), "
            << "mask is not supported in reg→reg form; got " << op->args_.size() << " args";
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("data_copy_mode"))
            << "vf.gather (reg→reg) does not support data_copy_mode " << "(only Tile→Reg form supports it)";
        // Supports b8/b16/b32; src and dst must have the same type (not just the
        // same bit width — e.g. FP16 src with BF16 dst must be rejected); b64 is
        // not supported (vselr limitation, mirrors AscendC GatherImpl
        // SupportBytes<1,2,4> and the gather.md reg→reg type table).
        DataType dst_dt = GetExprDtype(op->args_[0]);
        DataType src_dt = GetExprDtype(op->args_[1]);
        DataType idx_dt = GetExprDtype(op->args_[2]);
        PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, src_dt.GetBit() != 64)
            << "vf.gather (reg→reg) does not support b64 types (vselr limitation), got " << DTypeStr(src_dt);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (src_dt.GetBit() == idx_dt.GetBit()))
            << "vf.gather (reg→reg) requires index bit width to match src, got src=" << DTypeStr(src_dt)
            << " index=" << DTypeStr(idx_dt);
        DataType gather_dst_dt = GetExprDtype(op->args_[0]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == gather_dst_dt)
            << "vf.gather (reg→reg) requires src and dst to have the same type, got dst=" << DTypeStr(gather_dst_dt)
            << " src=" << DTypeStr(src_dt);
        std::string dst = codegen.GetExprAsCode(op->args_[0]);
        std::string src = codegen.GetExprAsCode(op->args_[1]);
        std::string indices = codegen.GetExprAsCode(op->args_[2]);
        std::string cast_type = "uint32_t";
        if (dst_dt.GetBit() <= 8) {
            cast_type = "uint8_t";
        } else if (dst_dt.GetBit() == 16) {
            cast_type = "uint16_t";
        }
        codegen.Emit("vselr((RegTensor<" + cast_type + ">&)" + dst + ", (RegTensor<" + cast_type + ">&)" + src +
                     ", (RegTensor<" + cast_type + ">&)" + indices + ");");
        return "";
    }

    // UB-to-Reg form
    // args: [dst, src_ub, indices, mask]
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.gather requires 4 args (dst, src, indices, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    DataType dst_dt = GetExprDtype(op->args_[0]);
    DataType src_dt = GetExprDtype(op->args_[1]);
    DataType idx_dt = GetExprDtype(op->args_[2]);
    std::string indices = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);

    // Check mode: DATA_BLOCK_LOAD -> block gather, otherwise -> per-element gather
    bool is_datablock = false;
    if (op->HasKwarg("data_copy_mode")) {
        auto mode = static_cast<ir::DataCopyMode>(op->GetKwarg<int>("data_copy_mode"));
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT,
                          mode == ir::DataCopyMode::NORM || mode == ir::DataCopyMode::DATA_BLOCK_LOAD)
            << "vf.gather only supports data_copy_mode=NORM or DATA_BLOCK_LOAD, got "
            << VFEnumValueName(ir::EnumToString(mode));
        is_datablock = (mode == ir::DataCopyMode::DATA_BLOCK_LOAD);
        if (!is_datablock) {
            PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("block_stride"))
                << "vf.gather (NORM mode) does not support block_stride";
        }
    }

    if (is_datablock) {
        // DataCopyGatherB: dst b8/b16/b32/b64, index must be uint32_t
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, idx_dt == DataType::UINT32)
            << "vf.gather (DATA_BLOCK_LOAD) index must be UINT32, got " << DTypeStr(idx_dt);
        // Block gather: always casts to signed types (s8/s16/s32/s64)
        // and uses int8_t*/int16_t*/int32_t*/int64_t* for the UB pointer.
        std::string signed_c_type;
        if (dst_dt.GetBit() <= 8) {
            signed_c_type = "int8_t";
        } else if (dst_dt.GetBit() == 16) {
            signed_c_type = "int16_t";
        } else if (dst_dt.GetBit() == 32) {
            signed_c_type = "int32_t";
        } else {
            signed_c_type = "int64_t";
        }
        std::string gb_ub_ptr = GetUBufPtr(codegen, op->args_[1], signed_c_type);
        codegen.Emit("vgatherb((RegTensor<" + signed_c_type + ">&)" + dst + ", " + gb_ub_ptr +
                     ", (RegTensor<uint32_t> &)" + indices + ", " + mask + ");");
    } else {
        // DataCopyGather: specific src-dst-index type combinations
        // src b8 -> dst b16 + idx u16; src b16 -> dst b16 + idx u16/u32;
        // src b32 -> dst b32 + idx u32; src b64 -> dst b64 + idx u32/u64
        bool is_b16_src = (dst_dt == DataType::INT16 || dst_dt == DataType::UINT16 || dst_dt == DataType::FP16 ||
                           dst_dt == DataType::BF16);
        bool use_vgather2_bc = is_b16_src && (idx_dt.GetBit() >= 32);
        if (dst_dt.GetBit() == 16) {
            PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt.GetBit() == 8 || src_dt.GetBit() == 16)
                << "vf.gather (NORM) b16 dst requires b8/b16 src, got src=" << DTypeStr(src_dt);
            if (!use_vgather2_bc) {
                PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, idx_dt == DataType::UINT16)
                    << "vf.gather (NORM) b16 dst requires UINT16 index (or UINT32 for vgather2_bc), got "
                    << DTypeStr(idx_dt);
            } else {
                PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, idx_dt == DataType::UINT32)
                    << "vf.gather (NORM) b16 dst with vgather2_bc requires UINT32 index, got " << DTypeStr(idx_dt);
            }
        } else if (dst_dt.GetBit() == 32) {
            PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt.GetBit() == 32)
                << "vf.gather (NORM) b32 requires b32 src, got src=" << DTypeStr(src_dt);
            PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, idx_dt == DataType::UINT32)
                << "vf.gather (NORM) b32 dst requires UINT32 index, got " << DTypeStr(idx_dt);
        } else if (dst_dt.GetBit() == 64) {
            PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt.GetBit() == 64)
                << "vf.gather (NORM) b64 requires b64 src, got src=" << DTypeStr(src_dt);
            PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, idx_dt == DataType::UINT32)
                << "vf.gather (NORM) b64 dst requires UINT32 index, got " << DTypeStr(idx_dt);
        }
        if (dst_dt.GetBit() == 64) {
            // B64 gather: vgather2 has no b64 single-register overload.
            // Mirrors AscendC DataCopyGatherB64Impl: pack mask, compute odd/even
            // b32 indices (index*2, index*2+1), vgather2 each b32 half, interleave.
            std::string p = dst + "_gather_";
            std::string packed_m = p + "_pm_";
            std::string vl32_m = p + "_vl32_";
            std::string all_m = p + "_allm_";
            std::string and_m = p + "_andm_";
            std::string lo_idx = p + "_loi_";
            std::string hi_idx = p + "_hii_";
            std::string lo_reg = p + "_lor_";
            std::string hi_reg = p + "_hir_";
            std::string tmp_reg = p + "_tmp_";
            codegen.Emit("MaskReg " + packed_m + ";");
            codegen.Emit("ppack(" + packed_m + ", " + mask + ", LOWER);");
            // MaskAnd with VL32: b64 has 32 elements → 64 b32 slots after ppack,
            // but only first 32 b32 slots are valid. Mirrors AscendC
            // DataCopyGatherB64Impl: MaskAnd(dstMask, dstMask, lowerMask, preg).
            codegen.Emit("MaskReg " + vl32_m + " = pset_b32(PAT_VL32);");
            codegen.Emit("MaskReg " + all_m + " = pset_b32(PAT_ALL);");
            codegen.Emit("MaskReg " + and_m + ";");
            codegen.Emit("pand(" + and_m + ", " + packed_m + ", " + vl32_m + ", " + all_m + ");");
            codegen.Emit("RegTensor<uint32_t> " + lo_idx + ";");
            codegen.Emit("RegTensor<uint32_t> " + hi_idx + ";");
            codegen.Emit("vmuls(" + lo_idx + ", (RegTensor<uint32_t> &)" + indices + ", (uint32_t)2, " + and_m +
                         ", MODE_ZEROING);");
            codegen.Emit("vadds(" + hi_idx + ", " + lo_idx + ", (uint32_t)1, " + and_m + ", MODE_ZEROING);");
            codegen.Emit("RegTensor<uint32_t> " + lo_reg + ";");
            codegen.Emit("RegTensor<uint32_t> " + hi_reg + ";");
            std::string ub_ptr = GetUBufPtr(codegen, op->args_[1], "uint32_t");
            codegen.Emit("vgather2(" + lo_reg + ", " + ub_ptr + ", " + lo_idx + ", " + and_m + ");");
            codegen.Emit("vgather2(" + hi_reg + ", " + ub_ptr + ", " + hi_idx + ", " + and_m + ");");
            EmitB64Interleave(codegen, dst, lo_reg, hi_reg, p + "_ilv");
            return "";
        }
        std::string idx_c_type = use_vgather2_bc ? "uint32_t" : ((dst_dt.GetBit() >= 32) ? "uint32_t" : "uint16_t");
        std::string dst_expr = dst;
        std::string ub_ptr = GetUBufPtr(codegen, op->args_[1], src_dt.ToCTypeString());
        if (dst_dt == DataType::INT8 || dst_dt == DataType::UINT8) {
            dst_expr = "(RegTensor<int16_t>&)" + dst;
            ub_ptr = GetUBufPtr(codegen, op->args_[1], "int8_t");
        } else if (is_b16_src) {
            // vgather2_bc b16 overload expects vector_s16& (RegTensor<int16_t>&).
            // The b16 element occupies the lower 16 bits of each 32-bit slot;
            // the upper 16 bits are zero. Both vgather2 and vgather2_bc use the
            // same dst cast for b16 source types.
            dst_expr = "(RegTensor<int16_t>&)" + dst;
            ub_ptr = GetUBufPtr(codegen, op->args_[1], "int16_t");
        } else if (dst_dt.GetBit() == 32) {
            dst_expr = "(RegTensor<int32_t>&)" + dst;
            ub_ptr = GetUBufPtr(codegen, op->args_[1], "int32_t");
        }
        std::string instr = use_vgather2_bc ? "vgather2_bc" : "vgather2";
        codegen.Emit(instr + "(" + dst_expr + ", " + ub_ptr + ", (RegTensor<" + idx_c_type + "> &)" + indices + ", " +
                     mask + ");");
    }
    return "";
}

static std::string EmitVFStoreUnAlign(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // MaskReg src path: when args[1] is a MaskReg, dispatch to pstu
    bool src_is_mask = false;
    if (auto src_v = ir::As<ir::Var>(op->args_[1])) {
        src_is_mask = codegen.IsMaskRegVar(codegen.GetVarName(src_v));
    }
    if (src_is_mask) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
            << "vf.store_unalign mask path requires 3 args (ptr, mask, ureg)";
        // DataCopyUnAlign: only supports b16/b32 (SupportBytes<T, 2, 4>)
        DataType tile_dt = GetExprDtype(op->args_[0]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, tile_dt.GetBit() == 16 || tile_dt.GetBit() == 32)
            << "vf.store_unalign (mask path) only supports b16/b32 tile types, got " << DTypeStr(tile_dt);
        std::string vreg = codegen.GetExprAsCode(op->args_[1]);
        DataType mask_dt = GetExprDtype(op->args_[1], DataType::UINT16);
        int elem_bytes = static_cast<int>(mask_dt.GetBit() / 8);
        if (elem_bytes <= 0)
            elem_bytes = 4;
        // pstu only accepts uint16_t* or uint32_t* (DataCopyUnAlignImpl
        // casts to unsigned regardless of template T). b16→uint16_t, b32→uint32_t.
        std::string ptr_type = (elem_bytes <= 2) ? "uint16_t" : "uint32_t";
        // pstu modifies the pointer in-place (*&), so use post-update ref.
        // AscendC signature: pstu(ureg, mask, (__ubuf__ uint32_t*&)dstAddr)
        // The & in the cast is required so pstu advances the cursor, otherwise
        // vstar in store_unalign_post would overwrite pstu's output at the same address.
        std::string tile_ptr = codegen.GetOrCreateVFTilePtr(op->args_[0], /*is_post_update=*/true);
        std::string ureg = codegen.GetExprAsCode(op->args_[2]);
        codegen.Emit("pstu(" + ureg + ", " + vreg + ", (__ubuf__ " + ptr_type + " *&)" + tile_ptr + ");");
        return "";
    }
    // 4-arg form: vstus(ureg, stride, vreg, dst, POST_UPDATE|NORM) or
    //              vstu(ureg, areg, vreg, dst, POST_UPDATE) when args[3] is an AddrReg.
    // Strideless vstur mode is in vf.squeeze_store_unalign.
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.store_unalign requires 4 args (dst, vreg, ureg, stride|areg); "
        << "use vf.squeeze_store_unalign for strideless (vstur) mode";
    DataType src_dt = GetExprDtype(op->args_[1]);
    // vstus/vstu support b8/b16/b32/b64 element widths. b8 covers the 4-bit
    // FP4 types (packed 2-per-byte), mirroring the load_unalign side and the
    // DataCopyUnAlignImpl b8->uint8_t cast rule.
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      IsB8Type(src_dt) || src_dt.GetBit() == 16 || src_dt.GetBit() == 32 || src_dt.GetBit() == 64)
        << "vf.store_unalign source only supports b8/b16/b32/b64 types, got " << DTypeStr(src_dt);
    auto ureg_var = ir::As<ir::Var>(op->args_[2]);
    if (ureg_var) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, codegen.IsUnalignRegVar(codegen.GetVarName(ureg_var)))
            << "vf.store_unalign requires the ureg argument to be an UnalignReg (from vf.unalign_reg_for_store)";
    }
    // AscendC DataCopyUnAlignImpl cast rules (dav_3510 store_impl.h):
    //   b8  → uint8_t
    //   b16 → no cast (native type)
    //   b32 → int32_t
    //   b64 → int32_t (RegTraitNumOne: simulated as b32 with stride×2 for vstus;
    //                  cast to int32_t for vstu — no stride scaling)
    bool is_b64 = (src_dt.GetBit() == 64);
    DataType cast_dt = src_dt;
    if (IsB8Type(src_dt)) {
        cast_dt = DataType::UINT8;
    } else if (src_dt.GetBit() == 32 || is_b64) {
        cast_dt = DataType::INT32;
    }
    std::string base_c_type = cast_dt.ToCTypeString();
    std::string tile_ptr_var = codegen.GetOrCreateVFTilePtr(op->args_[0], /*is_post_update=*/true);
    std::string vreg = codegen.GetExprAsCode(op->args_[1]);
    std::string ureg = codegen.GetExprAsCode(op->args_[2]);
    std::string vreg_expr = (cast_dt == src_dt) ? vreg : ("(RegTensor<" + base_c_type + "> &)" + vreg);
    std::string fourth_arg = codegen.GetExprAsCode(op->args_[3]);
    bool post_update = op->HasKwarg("post_update") && op->GetKwarg<bool>("post_update");
    std::string pu = post_update ? "POST_UPDATE" : "NORM";
    std::string ptr_cast = "(__ubuf__ " + base_c_type + " *&)";
    if (codegen.IsAddrRegVar(fourth_arg)) {
        // vstu(ureg, areg, vreg, dst, POST_UPDATE|NORM) — AddrReg-based unaligned store.
        // vstu accepts the same post mode as vstus (AscendC DataCopyUnAlignImpl
        // passes postValue to vstu's 5th arg, same as vstus).
        codegen.Emit("vstu(" + ureg + ", " + fourth_arg + ", " + vreg_expr + ", " + ptr_cast + tile_ptr_var + ", " +
                     pu + ");");
    } else {
        // vstus(ureg, stride, vreg, dst, POST_UPDATE|NORM)
        // b64: stride must be doubled to account for b32 simulation of b64 elements.
        std::string stride = is_b64 ? ("(" + fourth_arg + ") * 2") : fourth_arg;
        codegen.Emit("vstus(" + ureg + ", " + stride + ", " + vreg_expr + ", " + ptr_cast + tile_ptr_var + ", " + pu +
                     ");");
    }
    return "";
}

static std::string EmitVFStoreUnAlignPost(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() >= 3)
        << "vf.store_unalign_post requires 3 args (dst, ureg, stride|areg); "
        << "use vf.squeeze_store_unalign_post for strideless (vstar) mode";
    DataType tile_dt = GetExprDtype(op->args_[0]);
    auto ureg_var = ir::As<ir::Var>(op->args_[1]);
    if (ureg_var) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, codegen.IsUnalignRegVar(codegen.GetVarName(ureg_var)))
            << "vf.store_unalign_post requires the ureg argument to be an UnalignReg (from vf.unalign_reg_for_store)";
    }
    // AscendC DataCopyUnAlignPostImpl cast rules (dav_3510 store_impl.h):
    //   b8  → uint8_t
    //   b16 → no cast (native type)
    //   b32 → int32_t
    //   b64 → int32_t (simulated as b32 with stride×2 for vstas; cast to
    //                  int32_t for vsta — no stride scaling)
    bool is_b64 = (tile_dt.GetBit() == 64);
    DataType cast_dt = tile_dt;
    if (IsB8Type(tile_dt)) {
        cast_dt = DataType::UINT8;
    } else if (tile_dt.GetBit() == 32 || is_b64) {
        cast_dt = DataType::INT32;
    }
    std::string base_c_type = cast_dt.ToCTypeString();
    std::string tile_ptr_var = codegen.GetOrCreateVFTilePtr(op->args_[0], /*is_post_update=*/true);
    std::string ureg = codegen.GetExprAsCode(op->args_[1]);
    std::string third_arg = codegen.GetExprAsCode(op->args_[2]);
    bool post_update = op->HasKwarg("post_update") && op->GetKwarg<bool>("post_update");
    std::string ptr_cast = "(__ubuf__ " + base_c_type + " *&)";
    if (codegen.IsAddrRegVar(third_arg)) {
        // vsta(ureg, dst, areg) — AddrReg-based unaligned store post.
        // vsta has no post mode; post_update kwarg is not applicable in this mode.
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("post_update"))
            << "vf.store_unalign_post (AddrReg mode) does not support post_update; "
            << "vsta has no post mode parameter";
        codegen.Emit("vsta(" + ureg + ", " + ptr_cast + tile_ptr_var + ", " + third_arg + ");");
        return "";
    }
    // vstas always requires __ubuf__ T*& (lvalue ref).
    // POST_UPDATE: 4 args (ureg, ptr, stride, POST_UPDATE)
    // NORM: 3 args (ureg, ptr, stride) — no Post argument
    // b64: stride must be doubled to account for b32 simulation of b64 elements.
    std::string stride = is_b64 ? ("(" + third_arg + ") * 2") : third_arg;
    if (post_update) {
        codegen.Emit("vstas(" + ureg + ", " + ptr_cast + tile_ptr_var + ", " + stride + ", POST_UPDATE);");
    } else {
        codegen.Emit("vstas(" + ureg + ", " + ptr_cast + tile_ptr_var + ", " + stride + ");");
    }
    return "";
}

// ============================================================================
// SqueezeStoreUnAlign — vstur (strideless unaligned store, reads AR register
// for byte count). Must be paired with vf.squeeze(STORE_REG) and
// vf.squeeze_store_unalign_post. The AR register (written by squeeze) provides
// the valid byte count; vstur uses it as the implicit stride.
// ============================================================================
static std::string EmitVFSqueezeStoreUnAlign(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "vf.squeeze_store_unalign requires 3 args (dst, src, align_reg)";
    // Squeeze path does not support mask_reg src — use vf.store_unalign (pstu) for mask store.
    if (auto src_v = ir::As<ir::Var>(op->args_[1])) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, !codegen.IsMaskRegVar(codegen.GetVarName(src_v)))
            << "vf.squeeze_store_unalign does not support MaskReg src; use vf.store_unalign instead";
    }
    auto ureg_var = ir::As<ir::Var>(op->args_[2]);
    if (ureg_var) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, codegen.IsUnalignRegVar(codegen.GetVarName(ureg_var)))
            << "vf.squeeze_store_unalign requires the align_reg argument to be an UnalignReg"
            << " (from vf.unalign_reg_for_store)";
    }
    DataType src_dt = GetExprDtype(op->args_[1]);
    // vstur supports b8/b16/b32/b64 element widths
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      src_dt.GetBit() == 8 || src_dt.GetBit() == 16 || src_dt.GetBit() == 32 || src_dt.GetBit() == 64)
        << "vf.squeeze_store_unalign source only supports b8/b16/b32/b64 types, got " << DTypeStr(src_dt);
    // AscendC DataCopyUnAlignImpl cast rules (dav_3510 store_impl.h:496-518):
    //   b8  → uint8_t (line 506)
    //   b16 → no cast  (line 515, else branch uses original type)
    //   b32 → int32_t  (line 511)
    //   b64 → int64_t  (line 513)
    DataType cast_dt = src_dt;
    if (src_dt.GetBit() == 32) {
        cast_dt = DataType::INT32;
    } else if (src_dt.GetBit() == 64) {
        cast_dt = DataType::INT64;
    } else if (IsB8Type(src_dt)) {
        cast_dt = DataType::UINT8;
    }
    std::string base_c_type = cast_dt.ToCTypeString();
    std::string tile_ptr_var = codegen.GetOrCreateVFTilePtr(op->args_[0], /*is_post_update=*/true);
    std::string vreg = codegen.GetExprAsCode(op->args_[1]);
    std::string ureg = codegen.GetExprAsCode(op->args_[2]);
    std::string vreg_expr = (cast_dt == src_dt) ? vreg : ("(RegTensor<" + base_c_type + "> &)" + vreg);
    // vstur takes __ubuf__ T* (not *&)
    std::string ptr_cast = "(__ubuf__ " + base_c_type + " *)";
    codegen.Emit("vstur(" + ureg + ", " + vreg_expr + ", " + ptr_cast + tile_ptr_var + ", POST_UPDATE);");
    return "";
}

// ============================================================================
// SqueezeStoreUnAlignPost — vstar (strideless unaligned store post, reads AR
// register for remaining byte count). Must be paired with
// vf.squeeze_store_unalign.
// ============================================================================
static std::string EmitVFSqueezeStoreUnAlignPost(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2)
        << "vf.squeeze_store_unalign_post requires 2 args (dst, align_reg)";
    DataType tile_dt = GetExprDtype(op->args_[0]);
    // vstar supports b8/b16/b32/b64 element widths
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      IsB8Type(tile_dt) || tile_dt.GetBit() == 16 || tile_dt.GetBit() == 32 || tile_dt.GetBit() == 64)
        << "vf.squeeze_store_unalign_post only supports b8/b16/b32/b64 types, got " << DTypeStr(tile_dt);
    auto ureg_var = ir::As<ir::Var>(op->args_[1]);
    if (ureg_var) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, codegen.IsUnalignRegVar(codegen.GetVarName(ureg_var)))
            << "vf.squeeze_store_unalign_post requires the align_reg argument to be an UnalignReg"
            << " (from vf.unalign_reg_for_store)";
    }
    // AscendC DataCopyUnAlignPostImpl cast rules (dav_3510 store_impl.h:520-537):
    //   b8  → uint8_t (line 525)
    //   b16 → no cast  (line 534, else branch uses original type)
    //   b32 → int32_t  (line 530)
    //   b64 → int64_t  (line 532)
    DataType cast_dt = tile_dt;
    if (tile_dt.GetBit() == 32) {
        cast_dt = DataType::INT32;
    } else if (tile_dt.GetBit() == 64) {
        cast_dt = DataType::INT64;
    } else if (IsB8Type(tile_dt)) {
        cast_dt = DataType::UINT8;
    }
    std::string base_c_type = cast_dt.ToCTypeString();
    std::string tile_ptr_var = codegen.GetOrCreateVFTilePtr(op->args_[0], /*is_post_update=*/true);
    std::string ureg = codegen.GetExprAsCode(op->args_[1]);
    // vstar takes __ubuf__ T* (not *&)
    std::string ptr_cast = "(__ubuf__ " + base_c_type + " *)";
    codegen.Emit("vstar(" + ureg + ", " + ptr_cast + tile_ptr_var + ");");
    return "";
}

static std::string EmitVFUnalignRegForStore(const ir::CallPtr& /*op*/, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    std::string reg_name = codegen.GetCurrentResultTarget();
    codegen.Emit("UnalignReg " + reg_name + ";");
    codegen.RegisterUnalignRegVar(reg_name);
    return "";
}

static std::string EmitVFClearSpr(const ir::CallPtr& /*op*/, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    codegen.Emit("sprclr(SPR_AR);");
    return "";
}

// ============================================================================
// UnalignRegForLoad — declare unaligned load register
// ============================================================================

static std::string EmitVFUnalignRegForLoad(const ir::CallPtr& /*op*/, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    std::string reg_name = codegen.GetCurrentResultTarget();
    codegen.Emit("UnalignReg " + reg_name + ";");
    codegen.RegisterUnalignRegVar(reg_name);
    return "";
}

// ============================================================================
// LoadUnalignPre — vldas (setup unaligned load)
// ============================================================================

static std::string EmitVFLoadUnalignPre(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2)
        << "vf.load_unalign_pre requires 2 args (ureg, src_ptr)";
    std::string ureg = codegen.GetExprAsCode(op->args_[0]);
    auto ureg_var = ir::As<ir::Var>(op->args_[0]);
    if (ureg_var) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, codegen.IsUnalignRegVar(codegen.GetVarName(ureg_var)))
            << "vf.load_unalign_pre requires the first argument to be an UnalignReg (from vf.load_unalign_init), got "
            << ureg;
    }
    DataType dt = GetExprDtype(op->args_[1], DataType::FP32);
    // vldas supports b8/b16/b32/b64 element widths
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      IsB8Type(dt) || dt.GetBit() == 16 || dt.GetBit() == 32 || dt.GetBit() == 64)
        << "vf.load_unalign_pre only supports b8/b16/b32/b64 types, got " << DTypeStr(dt);
    int elem_bytes = static_cast<int>(dt.GetBit() / 8);
    if (elem_bytes <= 0)
        elem_bytes = 4;
    std::string ptr_type;
    if (elem_bytes == 1) {
        ptr_type = "uint8_t";
    } else if (elem_bytes == 8) {
        ptr_type = "uint32_t";
    } else if (elem_bytes == 4) {
        ptr_type = "int32_t";
    } else {
        if (dt == DataType::FP16 || dt == DataType::BF16)
            ptr_type = "half";
        else
            ptr_type = "uint16_t";
    }
    std::string src_ptr = GetUBufPtr(codegen, op->args_[1], ptr_type);
    codegen.Emit("vldas(" + ureg + ", " + src_ptr + ");");
    return "";
}

// ============================================================================
// LoadUnalign — vldus (unaligned load body, supports 3-arg and 4-arg strided)
// ============================================================================

static std::string EmitVFLoadUnalign(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() >= 3)
        << "vf.load_unalign requires 3-4 args (dst, ureg, src_ptr [, stride])";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string ureg = codegen.GetExprAsCode(op->args_[1]);
    auto ureg_var = ir::As<ir::Var>(op->args_[1]);
    if (ureg_var) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, codegen.IsUnalignRegVar(codegen.GetVarName(ureg_var)))
            << "vf.load_unalign requires the ureg argument to be an UnalignReg (from vf.load_unalign_init), got "
            << ureg;
    }
    DataType dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (IsB8Type(dst_dt) || dst_dt.GetBit() == 16 || dst_dt.GetBit() == 32 || dst_dt.GetBit() == 64))
        << "vf.load_unalign only supports b8/b16/b32/b64 types, got " << DTypeStr(dst_dt);
    // vldus supports b8/b16/b32/b64 element widths
    std::string ptr_type = dst_dt.ToCTypeString();
    if (op->args_.size() >= 4) {
        std::string stride = codegen.GetExprAsCode(op->args_[3]);
        std::string src_ptr = codegen.GetOrCreateVFTilePtr(op->args_[2], /*is_post_update=*/true);
        int elem_bytes = static_cast<int>(dst_dt.GetBit() / 8);
        if (elem_bytes <= 0)
            elem_bytes = 4;
        bool is_b64 = (elem_bytes == 8);
        // AscendC cast rules (dav_m510 load_impl.h DataCopyUnAlignImpl):
        //   vldus (stride): b64 is simulated as uint32_t with stride*2, else native
        std::string eff_ptr_type = is_b64 ? "uint32_t" : ptr_type;
        std::string dst_expr = is_b64 ? ("(RegTensor<uint32_t>&)" + dst) : dst;
        if (eff_ptr_type != "float") {
            src_ptr = "(__ubuf__ " + eff_ptr_type + " *&)" + src_ptr;
        }
        std::string effective_stride = is_b64 ? ("(" + stride + ") * 2") : stride;
        codegen.Emit("vldus(" + dst_expr + ", " + ureg + ", " + src_ptr + ", " + effective_stride + ", POST_UPDATE);");
    } else {
        std::string src_ptr = GetUBufPtr(codegen, op->args_[2], ptr_type);
        codegen.Emit("vldus(" + dst + ", " + ureg + ", " + src_ptr + ");");
    }
    return "";
}

// ============================================================================
// Scatter — vscatter (scatter store by indices)
// ============================================================================

static std::string EmitVFScatter(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.scatter requires 4 args (base_ptr, src, index, mask)";
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string index = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    DataType src_dt = GetExprDtype(op->args_[1]);
    DataType idx_dt = GetExprDtype(op->args_[2]);
    // Doc: src supports DT_INT8,DT_UINT8,DT_INT16,DT_UINT16,DT_FP16,DT_BF16,
    // DT_INT32,DT_UINT32,DT_FP32,DT_INT64,DT_UINT64
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      src_dt == DataType::INT8 || src_dt == DataType::UINT8 || src_dt == DataType::BOOL ||
                          src_dt == DataType::INT16 || src_dt == DataType::UINT16 || src_dt == DataType::FP16 ||
                          src_dt == DataType::BF16 || src_dt == DataType::INT32 || src_dt == DataType::UINT32 ||
                          src_dt == DataType::FP32 || src_dt == DataType::INT64 || src_dt == DataType::UINT64)
        << "vf.scatter only supports INT8/UINT8/INT16/UINT16/FP16/BF16/INT32/UINT32/FP32/INT64/UINT64, got "
        << DTypeStr(src_dt);
    if (src_dt.GetBit() == 8 || src_dt.GetBit() == 16) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, idx_dt == DataType::UINT16)
            << "vf.scatter b8/b16 src requires UINT16 index, got " << DTypeStr(idx_dt);
    } else if (src_dt.GetBit() == 32) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, idx_dt == DataType::UINT32)
            << "vf.scatter b32 src requires UINT32 index, got " << DTypeStr(idx_dt);
    } else {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, idx_dt == DataType::UINT32 || idx_dt == DataType::UINT64)
            << "vf.scatter b64 src requires UINT32/UINT64 index, got " << DTypeStr(idx_dt);
    }
    std::string base_c_type = src_dt.ToCTypeString();
    std::string base_ptr = GetUBufPtr(codegen, op->args_[0], base_c_type);
    if (src_dt.GetBit() == 64) {
        // B64 scatter: vscatter has no b64 single-register overload.
        // Mirrors AscendC DataCopyScatterB64Impl: pack mask, compute odd/even
        // b32 indices (index*2, index*2+1), deinterleave src into b32 halves,
        // vscatter each half.
        std::string p = src + "_scatter_";
        std::string packed_m = p + "_pm_";
        std::string vl32_m = p + "_vl32_";
        std::string all_m = p + "_allm_";
        std::string and_m = p + "_andm_";
        std::string odd_idx = p + "_oi_";
        std::string even_idx = p + "_ei_";
        std::string lo_reg = p + "_lor_";
        std::string hi_reg = p + "_hir_";
        std::string dump = p + "_dump_";
        codegen.Emit("MaskReg " + packed_m + ";");
        codegen.Emit("ppack(" + packed_m + ", " + mask + ", LOWER);");
        codegen.Emit("MaskReg " + vl32_m + " = pset_b32(PAT_VL32);");
        codegen.Emit("MaskReg " + all_m + " = pset_b32(PAT_ALL);");
        codegen.Emit("MaskReg " + and_m + ";");
        codegen.Emit("pand(" + and_m + ", " + packed_m + ", " + vl32_m + ", " + all_m + ");");
        codegen.Emit("RegTensor<uint32_t> " + odd_idx + ";");
        codegen.Emit("RegTensor<uint32_t> " + even_idx + ";");
        codegen.Emit("vmuls(" + odd_idx + ", (RegTensor<uint32_t> &)" + index + ", (uint32_t)2, " + and_m +
                     ", MODE_ZEROING);");
        codegen.Emit("vadds(" + even_idx + ", " + odd_idx + ", (uint32_t)1, " + and_m + ", MODE_ZEROING);");
        codegen.Emit("RegTensor<uint32_t> " + lo_reg + ";");
        codegen.Emit("RegTensor<uint32_t> " + hi_reg + ";");
        codegen.Emit("RegTensor<uint32_t> " + dump + ";");
        codegen.Emit("vdintlv(" + lo_reg + ", " + hi_reg + ", (RegTensor<uint32_t>&)" + src +
                     ", (RegTensor<uint32_t>&)" + src + ");");
        std::string ub_ptr = GetUBufPtr(codegen, op->args_[0], "uint32_t");
        codegen.Emit("vscatter(" + lo_reg + ", " + ub_ptr + ", " + odd_idx + ", " + and_m + ");");
        codegen.Emit("vscatter(" + hi_reg + ", " + ub_ptr + ", " + even_idx + ", " + and_m + ");");
        return "";
    }
    std::string idx_c_type = (src_dt.GetBit() >= 32) ? "uint32_t" : "uint16_t";
    codegen.Emit("vscatter(" + src + ", " + base_ptr + ", (RegTensor<" + idx_c_type + "> &)" + index + ", " + mask +
                 ");");
    return "";
}

// ============================================================================
// Unsqueeze — vusqz (expand mask bits into register)
// ============================================================================

static std::string EmitVFUnsqueeze(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2)
        << "vf.unsqueeze requires 2 args (dst, mask)";
    // PrefixSum (vusqz): int8/uint8/int16/uint16/int32/uint32 only
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string mask = codegen.GetExprAsCode(op->args_[1]);
    DataType dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (dst_dt == DataType::INT8 || dst_dt == DataType::UINT8 || dst_dt == DataType::INT16 ||
                       dst_dt == DataType::UINT16 || dst_dt == DataType::INT32 || dst_dt == DataType::UINT32))
        << "vf.unsqueeze dst only supports INT8/UINT8/INT16/UINT16/INT32/UINT32, got " << DTypeStr(dst_dt);
    codegen.Emit("vusqz(" + dst + ", " + mask + ");");
    return "";
}

// ============================================================================
// Truncate — vtrc with ROUND_Z (alias for Round with round_mode=TRUNC)
// ============================================================================

static std::string EmitVFTruncate(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "vf.truncate requires 3 args (dst, src, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      src_dt == DataType::FP16 || src_dt == DataType::BF16 || src_dt == DataType::FP32)
        << "vf.truncate src only supports FP16/BF16/FP32, got " << DTypeStr(src_dt);
    DataType vf_truncate_dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == vf_truncate_dst_dt)
        << "vf.truncate requires src and dst to have the same type, got dst=" << DTypeStr(vf_truncate_dst_dt)
        << " src=" << DTypeStr(src_dt);
    DataType trc_dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == trc_dst_dt)
        << "vf.truncate requires src and dst to have the same type, got dst=" << DTypeStr(trc_dst_dt)
        << " src=" << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string round_const = "ROUND_Z";
    if (op->HasKwarg("round_mode")) {
        auto rm = static_cast<ir::VFRoundMode>(op->GetKwarg<int>("round_mode"));
        if (rm == ir::VFRoundMode::CAST_RINT)
            round_const = "ROUND_R";
        else if (rm == ir::VFRoundMode::CAST_CEIL)
            round_const = "ROUND_C";
        else if (rm == ir::VFRoundMode::CAST_FLOOR)
            round_const = "ROUND_F";
        else if (rm == ir::VFRoundMode::CAST_TRUNC)
            round_const = "ROUND_Z";
        else
            PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false)
                << "vf.truncate only supports round_mode CAST_RINT/CAST_CEIL/CAST_FLOOR/CAST_TRUNC, got "
                << VFEnumValueName(ir::EnumToString(rm));
    }
    std::string mode = VFZeroingOnly(op, "vf.truncate");
    codegen.Emit("vtrc(" + dst + ", " + src + ", " + round_const + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// MaskGenWithRegTensor — movvp (generate MaskReg from RegTensor bit offset)
// ============================================================================

static std::string EmitVFMaskGenWithRegTensor(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 1)
        << "vf.mask_gen_with_reg_tensor requires 1 arg (src)";
    std::string src = codegen.GetExprAsCode(op->args_[0]);
    std::string mask_dst = codegen.GetCurrentResultTarget();
    codegen.Emit("MaskReg " + mask_dst + ";");
    codegen.RegisterMaskRegVar(mask_dst);
    std::string offset = "0";
    if (op->HasKwarg("offset")) {
        offset = std::to_string(op->GetKwarg<int>("offset"));
    }
    DataType src_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt.GetBit() == 16 || src_dt.GetBit() == 32)
        << "vf.mask_gen_with_reg_tensor source only supports b16/b32 types, got " << DTypeStr(src_dt);
    // offset must be 0~15 for b16, 0~31 for b32
    int offset_val = 0;
    if (op->HasKwarg("offset")) {
        offset_val = op->GetKwarg<int>("offset");
    }
    if (src_dt.GetBit() == 16) {
        PRO_CODEGEN_CHECK(ExternalError::OUT_OF_RANGE, offset_val >= 0 && offset_val <= 15)
            << "vf.mask_gen_with_reg_tensor offset must be 0~15 for b16, got " << offset_val;
    } else {
        PRO_CODEGEN_CHECK(ExternalError::OUT_OF_RANGE, offset_val >= 0 && offset_val <= 31)
            << "vf.mask_gen_with_reg_tensor offset must be 0~31 for b32, got " << offset_val;
    }
    if (src_dt.GetBit() == 16) {
        codegen.Emit("movvp(" + mask_dst + ", (RegTensor<uint16_t> &)" + src + ", " + offset + ");");
    } else {
        codegen.Emit("movvp(" + mask_dst + ", (RegTensor<uint32_t> &)" + src + ", " + offset + ");");
    }
    return "";
}

// ============================================================================
// GetMaskSpr (unified) — movp_b32/movp_b16 with width kwarg
// Replaces: GetMaskSprB32, GetMaskSprB16
// ============================================================================

static std::string EmitVFGetMaskSpr(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    std::string reg_name = codegen.GetCurrentResultTarget();
    std::string width = "B32";
    if (op->HasKwarg("width"))
        width = VFEnumValueName(ir::EnumToString(static_cast<ir::MaskWidth>(op->GetKwarg<int>("width"))));
    // MoveMask: only supports b16/b32 (SupportBytes<T, 2, 4>)
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, width == "B16" || width == "B32")
        << "vf.get_mask_spr only supports B16/B32 width, got " << width;
    if (width == "B16")
        codegen.Emit("MaskReg " + reg_name + " = movp_b16();");
    else
        codegen.Emit("MaskReg " + reg_name + " = movp_b32();");
    codegen.RegisterMaskRegVar(reg_name);
    return "";
}

// ============================================================================
// Registration
// ============================================================================

REGISTER_BACKEND_OP(BackendCCE, "vf.reg_tensor")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFRegTensor(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.mask_reg")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMaskReg(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.create_mask")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFCreateMask(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.full")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFDuplicate(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.load_align")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFLoadAlign(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.store_align")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFStoreAlign(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.max")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMax(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.add")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFAdd(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFSub(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.and_")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFAnd(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.xor")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFXor(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.or_")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFOr(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.reduce_sum")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFReduceSum(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.reduce_max")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFReduceMax(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.reduce_min")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFReduceMin(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.mul")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMul(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.mul_add_dst")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMulAddDst(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.div")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFDiv(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.muls")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMuls(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.ln")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFLn(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.log")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFLog(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.min")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMin(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.exp")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFExp(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.abs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFAbs(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.not_")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFNot(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.sqrt")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFSqrt(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.relu")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFRelu(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.neg")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFNeg(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.adds")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFAdds(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.mins")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMins(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.maxs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMaxs(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.leaky_relu")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFLeakyRelu(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.interleave")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFInterleave(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.pair_reduce_sum")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFPairReduceSum(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.abs_sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFAbsSub(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.axpy")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFAxpy(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.mul_dst_add")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMulDstAdd(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.pack")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFPack(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.unpack")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFUnpack(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.prelu")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFPRelu(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.shift_left")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFShiftLeft(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.shift_right")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFShiftRight(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.mull")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMull(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.addc")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFAddc(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.subc")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFSubc(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.exp_sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFExpSub(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.astype")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFCast(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.de_interleave")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFDeInterleave(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.select")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFSelect(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.update_mask")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFUpdateMask(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.mem_bar")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMemBar(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.histograms")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFHistograms(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.eq")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFCompareImpl(op, codegen, "EQ");
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.ne")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFCompareImpl(op, codegen, "NE");
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.lt")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFCompareImpl(op, codegen, "LT");
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.gt")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFCompareImpl(op, codegen, "GT");
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.le")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFCompareImpl(op, codegen, "LE");
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.ge")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFCompareImpl(op, codegen, "GE");
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.squeeze")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFSqueeze(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.arange")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFArange(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.gather")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFGather(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.store_unalign")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFStoreUnAlign(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.store_unalign_post")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFStoreUnAlignPost(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.squeeze_store_unalign")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFSqueezeStoreUnAlign(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.squeeze_store_unalign_post")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFSqueezeStoreUnAlignPost(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.unalign_reg_for_store")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFUnalignRegForStore(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.clear_spr")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFClearSpr(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.load_unalign_init")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFUnalignRegForLoad(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.load_unalign_pre")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFLoadUnalignPre(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.load_unalign")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFLoadUnalign(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.scatter")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFScatter(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.unsqueeze")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFUnsqueeze(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.truncate")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFTruncate(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.mask_gen_with_reg_tensor")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFMaskGenWithRegTensor(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.get_mask_spr")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFGetMaskSpr(op, codegen); });

// ============================================================================
// Log2 — composite: vln + vmuls(1/ln2) = ln(x) * 1.4426950408889634
// ============================================================================

static std::string EmitVFLog2(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "vf.log2 requires 3 args (dst, src, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (src_dt == DataType::FP16 || src_dt == DataType::FP32))
        << "vf.log2 src only supports FP16/FP32, got " << DTypeStr(src_dt);
    DataType vf_log2_dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == vf_log2_dst_dt)
        << "vf.log2 requires src and dst to have the same type, got dst=" << DTypeStr(vf_log2_dst_dt)
        << " src=" << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string mode = VFZeroingOnly(op, "vf.log2");
    if (op->HasKwarg("precision") && op->GetKwarg<bool>("precision")) {
        std::string mc = dst + "_mcmp_";
        std::string tp = dst + "_tmp_";
        std::string sc = dst + "_srcc_";
        std::string dc = dst + "_dstc_";
        std::string ctype = src_dt.ToCTypeString();
        codegen.Emit("MaskReg " + mc + ";");
        codegen.Emit("RegTensor<" + ctype + "> " + tp + ";");
        codegen.Emit("RegTensor<" + ctype + "> " + sc + " = (RegTensor<" + ctype + ">&)" + src + ";");
        codegen.Emit("RegTensor<" + ctype + "> " + dc + ";");
        if (src_dt == DataType::FP16) {
            codegen.Emit("union { uint16_t i; half f; } " + dst + "_thr_ = {0x03FF};");
            codegen.Emit("vcmps_lt(" + mc + ", " + sc + ", " + dst + "_thr_.f, " + mask + ");");
            codegen.Emit("vmuls(" + tp + ", " + sc + ", 1024.0f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + sc + ", " + tp + ", " + sc + ", " + mc + ");");
            codegen.Emit("vln(" + dc + ", " + sc + ", " + mask + ", " + mode + ");");
            codegen.Emit("vmuls(" + dc + ", " + dc + ", 1.4426950408889634f, " + mask + ", " + mode + ");");
            codegen.Emit("vadds(" + tp + ", " + dc + ", -10.0f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + dst + ", " + tp + ", " + dc + ", " + mc + ");");
        } else {
            codegen.Emit("union { uint32_t i; float f; } " + dst + "_thr_ = {0x007FFFFF};");
            codegen.Emit("vcmps_lt(" + mc + ", " + sc + ", " + dst + "_thr_.f, " + mask + ");");
            codegen.Emit("vmuls(" + tp + ", " + sc + ", 8388608.0f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + sc + ", " + tp + ", " + sc + ", " + mc + ");");
            codegen.Emit("vln(" + dc + ", " + sc + ", " + mask + ", " + mode + ");");
            codegen.Emit("vmuls(" + dc + ", " + dc + ", 1.4426950408889634f, " + mask + ", " + mode + ");");
            codegen.Emit("vadds(" + tp + ", " + dc + ", -23.0f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + dst + ", " + tp + ", " + dc + ", " + mc + ");");
        }
    } else {
        codegen.Emit("vln(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
        codegen.Emit("vmuls(" + dst + ", " + dst + ", 1.4426950408889634f, " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// Log10 — composite: vln + vmuls(1/ln10) = ln(x) * 0.4342944819032518
// ============================================================================

static std::string EmitVFLog10(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "vf.log10 requires 3 args (dst, src, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (src_dt == DataType::FP16 || src_dt == DataType::FP32))
        << "vf.log10 src only supports FP16/FP32, got " << DTypeStr(src_dt);
    DataType vf_log10_dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == vf_log10_dst_dt)
        << "vf.log10 requires src and dst to have the same type, got dst=" << DTypeStr(vf_log10_dst_dt)
        << " src=" << DTypeStr(src_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string mode = VFZeroingOnly(op, "vf.log10");
    if (op->HasKwarg("precision") && op->GetKwarg<bool>("precision")) {
        std::string mc = dst + "_mcmp_";
        std::string tp = dst + "_tmp_";
        std::string sc = dst + "_srcc_";
        std::string dc = dst + "_dstc_";
        std::string ctype = src_dt.ToCTypeString();
        codegen.Emit("MaskReg " + mc + ";");
        codegen.Emit("RegTensor<" + ctype + "> " + tp + ";");
        codegen.Emit("RegTensor<" + ctype + "> " + sc + " = (RegTensor<" + ctype + ">&)" + src + ";");
        codegen.Emit("RegTensor<" + ctype + "> " + dc + ";");
        if (src_dt == DataType::FP16) {
            codegen.Emit("union { uint16_t i; half f; } " + dst + "_thr_ = {0x03FF};");
            codegen.Emit("vcmps_lt(" + mc + ", " + sc + ", " + dst + "_thr_.f, " + mask + ");");
            codegen.Emit("vmuls(" + tp + ", " + sc + ", 1024.0f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + sc + ", " + tp + ", " + sc + ", " + mc + ");");
            codegen.Emit("vln(" + dc + ", " + sc + ", " + mask + ", " + mode + ");");
            codegen.Emit("vmuls(" + dc + ", " + dc + ", 0.43429448190325176f, " + mask + ", " + mode + ");");
            codegen.Emit("vadds(" + tp + ", " + dc + ", -3.01029995663981f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + dst + ", " + tp + ", " + dc + ", " + mc + ");");
        } else {
            codegen.Emit("union { uint32_t i; float f; } " + dst + "_thr_ = {0x007FFFFF};");
            codegen.Emit("vcmps_lt(" + mc + ", " + sc + ", " + dst + "_thr_.f, " + mask + ");");
            codegen.Emit("vmuls(" + tp + ", " + sc + ", 8388608.0f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + sc + ", " + tp + ", " + sc + ", " + mc + ");");
            codegen.Emit("vln(" + dc + ", " + sc + ", " + mask + ", " + mode + ");");
            codegen.Emit("vmuls(" + dc + ", " + dc + ", 0.43429448190325176f, " + mask + ", " + mode + ");");
            codegen.Emit("vadds(" + tp + ", " + dc + ", -6.923689900271567f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + dst + ", " + tp + ", " + dc + ", " + mc + ");");
        }
    } else {
        codegen.Emit("vln(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
        codegen.Emit("vmuls(" + dst + ", " + dst + ", 0.4342944819032518f, " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// MulsCast — vmulscvt (fused multiply-scalar-cast: dst(fp16) = cast(src(fp32) * scalar))
// ============================================================================

static std::string EmitVFMulsCast(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.muls_cast requires 4 args (dst, src, scalar, mask)";
    DataType src_dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == DataType::FP32)
        << "vf.muls_cast source only supports FP32, got " << DTypeStr(src_dt);
    DataType dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, dst_dt == DataType::FP16)
        << "vf.muls_cast destination only supports FP16, got " << DTypeStr(dst_dt);
    // FusedMulsCast: scalar must be float (Tuple<half, float, float>)
    DataType scalar_dt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, scalar_dt == DataType::FP32)
        << "vf.muls_cast scalar only supports FP32, got " << DTypeStr(scalar_dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string scalar_str = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    // layout kwarg selects the result half: ZERO -> PART_EVEN (default), ONE -> PART_ODD.
    // FusedMulsCast: only supports RegLayout ZERO/ONE
    std::string part = "PART_EVEN";
    if (op->HasKwarg("layout")) {
        auto layout = VFEnumValueName(ir::EnumToString(static_cast<ir::CastLayout>(op->GetKwarg<int>("layout"))));
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, layout == "ZERO" || layout == "ONE")
            << "vf.muls_cast only supports layout ZERO/ONE, got " << layout;
        if (layout == "ONE")
            part = "PART_ODD";
    }
    codegen.Emit("vmulscvt(" + dst + ", " + src + ", " + scalar_str + ", " + mask + ", " + part + ");");
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "vf.log2")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFLog2(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.log10")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFLog10(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.muls_cast")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMulsCast(op, codegen); });

// ============================================================================
// Load (unified) — vldas + vldus all-in-one unaligned load
// ============================================================================

static std::string EmitVFLoad(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() >= 2 && op->args_.size() <= 3)
        << "vf.load requires 2-3 args (dst, src_ptr[, stride])";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    DataType dst_dt = GetExprDtype(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (IsB8Type(dst_dt) || dst_dt.GetBit() == 16 || dst_dt.GetBit() == 32 || dst_dt.GetBit() == 64))
        << "vf.load only supports b8/b16/b32/b64 types, got " << DTypeStr(dst_dt);
    // vldas/vldus support b8/b16/b32/b64 element widths
    int elem_bytes = static_cast<int>(dst_dt.GetBit() / 8);
    if (elem_bytes <= 0)
        elem_bytes = 4;
    // vldas and vldus both use the native dtype pointer (matches asc_load C-API).
    // For b8 types that lack a direct vldas/vldus overload, reinterpret as uint8_t.
    std::string ptr_type = NeedsB8Reinterpret(dst_dt) ? "uint8_t" : dst_dt.ToCTypeString();
    static int load_counter = 0;
    std::string ureg_name = "__ureg_ld_" + std::to_string(load_counter++);
    codegen.Emit("UnalignReg " + ureg_name + ";");
    if (op->args_.size() == 3) {
        std::string src_ptr = GetUBufPtr(codegen, op->args_[1], ptr_type, /*is_post_update=*/true);
        std::string stride = codegen.GetExprAsCode(op->args_[2]);
        codegen.Emit("vldas(" + ureg_name + ", " + src_ptr + ");");
        codegen.Emit("vldus(" + dst + ", " + ureg_name + ", " + src_ptr + ", " + stride + ", POST_UPDATE);");
    } else {
        std::string src_ptr = GetUBufPtr(codegen, op->args_[1], ptr_type);
        codegen.Emit("vldas(" + ureg_name + ", " + src_ptr + ");");
        codegen.Emit("vldus(" + dst + ", " + ureg_name + ", " + src_ptr + ");");
    }
    return "";
}

// ============================================================================
// Store (unified) — vstus + vstas all-in-one unaligned store
// ============================================================================

static std::string EmitVFStore(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() >= 2 && op->args_.size() <= 3)
        << "vf.store requires 2-3 args (dst_ptr, src[, count])";
    DataType src_dt = GetExprDtype(op->args_[1]);
    // vstus/vstas support b8/b16/b32/b64 element widths
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      IsB8Type(src_dt) || src_dt.GetBit() == 16 || src_dt.GetBit() == 32 || src_dt.GetBit() == 64)
        << "vf.store only supports b8/b16/b32/b64 types, got " << DTypeStr(src_dt);
    // Check src/dst dtype consistency (doc: src and dst must have the same dtype)
    DataType tile_dt = GetExprDtype(op->args_[0], src_dt);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, tile_dt == src_dt)
        << "vf.store requires src and dst to have the same dtype, got src=" << DTypeStr(src_dt)
        << " dst=" << DTypeStr(tile_dt);
    int elem_bytes = static_cast<int>(src_dt.GetBit() / 8);
    if (elem_bytes <= 0)
        elem_bytes = 4;
    // Determine count (positional arg, default 256B/elem_bytes)
    std::string count;
    int max_count = 256 / elem_bytes;
    if (op->args_.size() == 3) {
        count = codegen.GetExprAsCode(op->args_[2]);
        auto const_int = std::dynamic_pointer_cast<const ir::ConstInt>(op->args_[2]);
        if (const_int != nullptr) {
            PRO_CODEGEN_CHECK(ExternalError::OUT_OF_RANGE, const_int->value_ <= max_count)
                << "vf.store count must not exceed 256B/sizeof(dtype) = " << max_count << ", got " << const_int->value_;
        }
    } else {
        count = std::to_string(max_count);
    }
    std::string ptr_type = NeedsB8Reinterpret(src_dt) ? "uint8_t" : src_dt.ToCTypeString();
    std::string src_expr = codegen.GetExprAsCode(op->args_[1]);
    // vstus/vstas use POST_UPDATE so the pointer is advanced internally by vstus
    // and vstas flushes the tail at the advanced position. This matches AscendC
    // StoreImpl which always uses POST_MODE_UPDATE via DataCopyUnAlignImpl.
    std::string dst_ptr = GetUBufPtr(codegen, op->args_[0], ptr_type, /*is_post_update=*/true);
    static int store_counter = 0;
    std::string ureg_name = "__ureg_st_" + std::to_string(store_counter++);
    codegen.Emit("UnalignReg " + ureg_name + ";");
    // B64 single-register: vstus/vstas have no 8-byte-element overload; reinterpret
    // as uint32_t pairs and double the count (mirrors AscendC DataCopyUnAlignImpl
    // b64 path; bitwise-identical for both INT64 and UINT64).
    if (src_dt.GetBit() == 64) {
        std::string b32_count = "(" + count + ") * 2";
        std::string ptr_var = codegen.GetOrCreateVFTilePtr(op->args_[0], /*is_post_update=*/true);
        std::string b32_ptr = "(__ubuf__ uint32_t*&)" + ptr_var;
        codegen.Emit("vstus(" + ureg_name + ", " + b32_count + ", (RegTensor<uint32_t>&)" +
                     codegen.GetExprAsCode(op->args_[1]) + ", " + b32_ptr + ", POST_UPDATE);");
        codegen.Emit("vstas(" + ureg_name + ", " + b32_ptr + ", 0, POST_UPDATE);");
    } else {
        codegen.Emit("vstus(" + ureg_name + ", " + count + ", " + src_expr + ", " + dst_ptr + ", POST_UPDATE);");
        codegen.Emit("vstas(" + ureg_name + ", " + dst_ptr + ", 0, POST_UPDATE);");
    }
    return "";
}

// EmitVFMaskLoad/Store/StoreUnalign have been removed — their logic is now
// unified into EmitVFLoadAlign/EmitVFStoreAlign/EmitVFStoreUnAlign via
// IsMaskRegVar dispatch, matching function-overloading model.

REGISTER_BACKEND_OP(BackendCCE, "vf.load")

    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFLoad(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.store")

    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFStore(op, codegen); });

// mask_load/mask_store/mask_store_unalign backend registrations removed —
// the parser redirects these to vf.load_align/vf.store_align/vf.store_unalign
// which dispatch via IsMaskRegVar.

// ============================================================================
// CreateAddrReg — AddrReg declaration + vag_b8/b16/b32 intrinsic
// ============================================================================

static std::string EmitVFCreateAddrReg(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() >= 1 && op->args_.size() <= 4)
        << "vf.create_addr_reg requires 1-4 strides";
    std::string reg_name = codegen.GetCurrentResultTarget();
    // Determine element width from dtype kwarg (default b32)
    DataType dt = DataType::FP32;
    if (op->HasKwarg("dtype")) {
        dt = op->GetKwarg<DataType>("dtype");
    }
    // vag_b8/b16/b32 support b8/b16/b32/b64 element widths (b64 uses vag_b32 with doubled stride)
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      IsB8Type(dt) || dt.GetBit() == 16 || dt.GetBit() == 32 || dt.GetBit() == 64)
        << "vf.create_addr_reg dtype must be b8/b16/b32/b64, got " << DTypeStr(dt);
    std::string vag_fn;
    if (dt == DataType::UINT8 || dt == DataType::INT8)
        vag_fn = "vag_b8";
    else if (dt.GetBit() == 16)
        vag_fn = "vag_b16";
    else
        vag_fn = "vag_b32";
    // Collect stride args. For b64, each stride is doubled.
    std::string stride_args;
    for (size_t i = 0; i < op->args_.size(); ++i) {
        std::string stride = codegen.GetExprAsCode(op->args_[i]);
        if (dt == DataType::UINT64 || dt == DataType::INT64) {
            stride = "(" + stride + ") * 2";
        }
        if (!stride_args.empty())
            stride_args += ", ";
        stride_args += stride;
    }
    // AddrReg (vector_address) must be declared and initialized in a single
    // statement (bisheng rejects a separate declaration + assignment). Emit the
    // declaration and vag_* initializer together, matching the
    // `AddrReg x = CreateAddrReg<T>(...)` usage. The vag_* must sit inside the
    // physical loop it is bound to (HardwareLoop).
    codegen.Emit("AddrReg " + reg_name + " = " + vag_fn + "(" + stride_args + ");");
    codegen.RegisterAddrRegVar(reg_name);
    return "";
}

// ============================================================================
// Move — vmov (RegTensor) / pmov (MaskReg), with or without mask
// ============================================================================

static std::string EmitVFMove(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2 || op->args_.size() == 3)
        << "vf.move requires 2 args (dst, src) or 3 args (dst, src, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    // Detect dst type: MaskReg vs RegTensor
    bool is_mask_dst = false;
    auto dst_var = ir::As<ir::Var>(op->args_[0]);
    if (dst_var) {
        is_mask_dst = codegen.IsMaskRegVar(codegen.GetVarName(dst_var));
    }
    if (!is_mask_dst) {
        DataType src_dt = GetExprDtype(op->args_[1]);
        // AscendC Move supports bool + the standard int/float types; the FP8/FP4
        // family and 4-bit ints are not part of the Move contract.
        PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR,
                          src_dt == DataType::BOOL || IsArithIntType(src_dt) || src_dt == DataType::FP16 ||
                              src_dt == DataType::BF16 || src_dt == DataType::FP32)
            << "vf.move src only supports BOOL/INT8/UINT8/INT16/UINT16/INT32/UINT32/INT64/UINT64/FP16/BF16/FP32, got "
            << DTypeStr(src_dt);
        DataType vf_move_dst_dt = GetExprDtype(op->args_[0]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_dt == vf_move_dst_dt)
            << "vf.move requires src and dst to have the same type, got dst=" << DTypeStr(vf_move_dst_dt)
            << " src=" << DTypeStr(src_dt);
    }
    if (op->args_.size() == 3) {
        std::string mask = codegen.GetExprAsCode(op->args_[2]);
        if (is_mask_dst) {
            codegen.Emit("pmov(" + dst + ", " + src + ", " + mask + ");");
        } else {
            // vf.move only supports MERGING mode (AscendC Copy/Move default).
            if (op->HasKwarg("mode")) {
                auto mode_val = static_cast<ir::MergeMode>(op->GetKwarg<int>("mode"));
                PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, mode_val == ir::MergeMode::MERGING)
                    << "vf.move only supports MERGING mode on current device";
            }
            std::string mode = "MODE_MERGING";
            DataType src_dt = GetExprDtype(op->args_[1]);
            if (src_dt == DataType::BOOL) {
                codegen.Emit("vmov((RegTensor<int8_t>&)" + dst + ", (RegTensor<int8_t>&)" + src + ", " + mask + ", " +
                             mode + ");");
            } else if (src_dt.GetBit() == 64) {
                std::string p = dst + "_mov_";
                std::string tm = p + "tm_";
                std::string m0 = p + "m0_";
                std::string m1 = p + "m1_";
                codegen.Emit("MaskReg " + tm + ";");
                codegen.Emit("ppack(" + tm + ", " + mask + ", LOWER);");
                codegen.Emit("MaskReg " + m0 + ", " + m1 + ";");
                codegen.Emit("pintlv_b32(" + m0 + ", " + m1 + ", " + tm + ", " + tm + ");");
                codegen.Emit("vmov((RegTensor<uint32_t>&)" + dst + ", (RegTensor<uint32_t>&)" + src + ", " + m0 + ", " +
                             mode + ");");
            } else {
                codegen.Emit("vmov(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
            }
        }
    } else {
        if (is_mask_dst) {
            codegen.Emit("pmov(" + dst + ", " + src + ");");
        } else {
            codegen.Emit("vmov(" + dst + ", " + src + ");");
        }
    }
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "vf.create_addr_reg")

    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFCreateAddrReg(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "vf.move")

    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFMove(op, codegen); });

// ============================================================================
// BitCast — type reinterpretation (no instruction, just C++ reference cast)
// ============================================================================

static std::string EmitVFBitCast(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 1 || op->args_.size() == 2)
        << "vf.bit_cast requires 1 arg (src) or 2 args (dst, src)";
    DataType target_dt = op->GetKwarg<DataType>("dtype");
    std::string src = codegen.GetExprAsCode(op->args_.back());
    if (op->args_.size() == 2) {
        // Assignment form: dst = vf.bit_cast(src, dtype=xxx)
        std::string dst = codegen.GetExprAsCode(op->args_[0]);
        codegen.Emit(dst + " = (RegTensor<" + target_dt.ToCTypeString() + ">&)" + src + ";");
    } else {
        // Nested form: vf.bit_cast(src, dtype=xxx) used as expression argument.
        // Return the cast expression directly so it inlines into the parent op's
        // instruction call, matching the documented behaviour:
        //   vor(reg_c, (RegTensor<uint8_t>&)reg_a, (RegTensor<uint8_t>&)reg_b, preg, MODE_ZEROING);
        // The parser materializes this form into its own temp variable, whose
        // `auto` declaration takes the cast's C++ type (RegTensor<T>) — register
        // it here so register-taking ops (compare, select, ...) dispatch the
        // temp to the vector form instead of the scalar one.
        if (!codegen.GetCurrentResultTarget().empty()) {
            codegen.RegisterRegTensorVar(codegen.GetCurrentResultTarget());
        }
        return "(RegTensor<" + target_dt.ToCTypeString() + "> &)" + src;
    }
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "vf.bit_cast")

    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitVFBitCast(op, codegen); });

} // namespace backend
} // namespace pypto
