/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include <any>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "backend/backend_cce.h"
#include "codegen/cce/cce_codegen.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/op_attr_types.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"

namespace pypto {
namespace backend {
namespace {

using Kwargs = std::vector<std::pair<std::string, std::any>>;

class CapturingCCECodegen final : public codegen::CCECodegen {
public:
    using codegen::CCECodegen::CCECodegen;
    void SetTarget(std::string target) { target_ = std::move(target); }

    [[nodiscard]] std::string GetCurrentResultTarget() const override { return target_; }

    void Emit(const std::string& line) override
    {
        emitted_ += line;
        emitted_ += '\n';
    }

    std::string GetExprAsCode(const ir::ExprPtr& expr) override
    {
        if (auto var = ir::As<ir::Var>(expr)) {
            return var->name_;
        }
        if (auto value = ir::As<ir::ConstInt>(expr)) {
            return std::to_string(value->value_);
        }
        if (auto value = ir::As<ir::ConstFloat>(expr)) {
            return std::to_string(value->value_);
        }
        return codegen::CCECodegen::GetExprAsCode(expr);
    }

    std::string GetVarName(const ir::VarPtr& var) override { return var->name_; }

    [[nodiscard]] const std::string& Emitted() const { return emitted_; }

private:
    std::string target_{"result"};
    std::string emitted_;
};

ir::TypePtr ScalarType(ir::DataType dtype) { return std::make_shared<const ir::ScalarType>(dtype); }

ir::VarPtr MakeVar(const std::string& name, ir::DataType dtype = ir::DataType::FP32)
{
    return std::make_shared<const ir::Var>(name, ScalarType(dtype), ir::Span::Unknown());
}

ir::VarPtr MakeTile(const std::string& name, ir::DataType dtype = ir::DataType::FP32)
{
    auto type = std::make_shared<const ir::TileType>(std::vector<int64_t>{16, 16}, dtype,
                                                     std::optional<ir::MemRefPtr>(std::nullopt),
                                                     std::optional<ir::TileView>(std::nullopt));
    return std::make_shared<const ir::Var>(name, type, ir::Span::Unknown());
}

ir::ExprPtr Int(int64_t value)
{
    return std::make_shared<const ir::ConstInt>(value, ir::DataType::INT64, ir::Span::Unknown());
}

ir::ExprPtr Float(double value)
{
    return std::make_shared<const ir::ConstFloat>(value, ir::DataType::FP32, ir::Span::Unknown());
}

ir::ExprPtr IndexVal(int64_t value)
{
    return std::make_shared<const ir::ConstInt>(value, ir::DataType::INDEX, ir::Span::Unknown());
}

int EnumValue(ir::MergeMode value) { return static_cast<int>(value); }

template <typename Enum>
int EnumValue(Enum value)
{
    return static_cast<int>(value);
}

ir::CallPtr MakeCall(const std::string& name, std::vector<ir::ExprPtr> args = {}, Kwargs kwargs = {})
{
    return std::make_shared<const ir::Call>(name, std::move(args), std::move(kwargs), ir::Span::Unknown());
}

std::string Invoke(CapturingCCECodegen& codegen, const std::string& name, std::vector<ir::ExprPtr> args = {},
                   Kwargs kwargs = {}, const std::string& target = "result")
{
    const std::size_t emitted_size = codegen.Emitted().size();
    const auto* info = BackendCCE::Instance().GetOpInfo(name);
    EXPECT_NE(info, nullptr) << name;
    if (info == nullptr) {
        return "";
    }
    EXPECT_EQ(info->pipe, ir::PipeType::V) << name;
    codegen.SetTarget(target);
    EXPECT_TRUE(info->codegen_func(MakeCall(name, std::move(args), std::move(kwargs)), codegen).empty());
    return codegen.Emitted().substr(emitted_size);
}

void ExpectContains(const std::string& generated, const std::vector<std::string>& fragments)
{
    for (const auto& fragment : fragments) {
        EXPECT_NE(generated.find(fragment), std::string::npos) << fragment << "\n" << generated;
    }
}

void ExpectInvoke(CapturingCCECodegen& codegen, const std::string& name, const std::vector<std::string>& expected,
                  std::vector<ir::ExprPtr> args = {}, Kwargs kwargs = {}, const std::string& target = "result")
{
    SCOPED_TRACE(name);
    ExpectContains(Invoke(codegen, name, std::move(args), std::move(kwargs), target), expected);
}

TEST(BackendCCEVFOpsTest, RegistersExpectedVectorFunctionOperations)
{
    const std::vector<std::string> names = {"vf.reg_tensor",
                                            "vf.create_mask",
                                            "vf.full",
                                            "vf.load_align",
                                            "vf.store_align",
                                            "vf.max",
                                            "vf.add",
                                            "vf.sub",
                                            "vf.and_",
                                            "vf.xor",
                                            "vf.or_",
                                            "vf.reduce_sum",
                                            "vf.reduce_max",
                                            "vf.reduce_min",
                                            "vf.mul",
                                            "vf.mul_add_dst",
                                            "vf.div",
                                            "vf.muls",
                                            "vf.ln",
                                            "vf.log",
                                            "vf.min",
                                            "vf.exp",
                                            "vf.abs",
                                            "vf.not_",
                                            "vf.sqrt",
                                            "vf.relu",
                                            "vf.neg",
                                            "vf.adds",
                                            "vf.mins",
                                            "vf.maxs",
                                            "vf.leaky_relu",
                                            "vf.interleave",
                                            "vf.pair_reduce_sum",
                                            "vf.abs_sub",
                                            "vf.axpy",
                                            "vf.mul_dst_add",
                                            "vf.pack",
                                            "vf.unpack",
                                            "vf.prelu",
                                            "vf.shift_left",
                                            "vf.shift_right",
                                            "vf.mull",
                                            "vf.addc",
                                            "vf.subc",
                                            "vf.exp_sub",
                                            "vf.astype",
                                            "vf.de_interleave",
                                            "vf.select",
                                            "vf.update_mask",
                                            "vf.mem_bar",
                                            "vf.histograms",
                                            "vf.eq",
                                            "vf.ne",
                                            "vf.lt",
                                            "vf.gt",
                                            "vf.le",
                                            "vf.ge",
                                            "vf.squeeze",
                                            "vf.arange",
                                            "vf.gather",
                                            "vf.store_unalign",
                                            "vf.store_unalign_post",
                                            "vf.unalign_reg_for_store",
                                            "vf.clear_spr",
                                            "vf.load_unalign_init",
                                            "vf.load_unalign_pre",
                                            "vf.load_unalign",
                                            "vf.scatter",
                                            "vf.unsqueeze",
                                            "vf.truncate",
                                            "vf.mask_gen_with_reg_tensor",
                                            "vf.get_mask_spr",
                                            "vf.log2",
                                            "vf.log10",
                                            "vf.muls_cast",
                                            "vf.load",
                                            "vf.store",
                                            "vf.create_addr_reg",
                                            "vf.move"};

    for (const auto& name : names) {
        const auto* info = BackendCCE::Instance().GetOpInfo(name);
        ASSERT_NE(info, nullptr) << name;
        EXPECT_EQ(info->pipe, ir::PipeType::V) << name;
        EXPECT_TRUE(static_cast<bool>(info->codegen_func)) << name;
    }
}

TEST(BackendCCEVFOpsTest, EmitsDeclarationsMasksBroadcastsAndMoves)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto fp = MakeVar("fp");
    auto s4 = MakeVar("s4", ir::DataType::INT4);
    auto u4 = MakeVar("u4", ir::DataType::UINT4);
    auto i32_reg = MakeVar("i32_reg", ir::DataType::INT32);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    auto mask2 = MakeVar("mask2", ir::DataType::UINT32);
    auto addr = MakeVar("addr", ir::DataType::INT64);

    Invoke(codegen, "vf.reg_tensor", {}, {{"dtype", ir::DataType::FP32}}, "fp");
    Invoke(codegen, "vf.reg_tensor", {}, {{"dtype", ir::DataType::INT4}}, "s4");
    Invoke(codegen, "vf.reg_tensor", {}, {{"dtype", ir::DataType::UINT4}}, "u4");
    Invoke(codegen, "vf.reg_tensor", {}, {{"dtype", ir::DataType::INT32}}, "i32_reg");
    EXPECT_TRUE(codegen.IsRegTensorVar("fp"));
    EXPECT_TRUE(codegen.IsRegTensorVar("s4"));
    EXPECT_TRUE(codegen.IsRegTensorVar("u4"));

    ExpectInvoke(codegen, "vf.create_mask", {"MaskReg mask;", "mask = pset_b8(PAT_VL16);"}, {},
                 {{"pattern", EnumValue(ir::MaskPattern::VL16)}, {"dtype", ir::DataType::INT8}}, "mask");
    ExpectInvoke(codegen, "vf.create_mask", {"MaskReg mask2;", "mask2 = pset_b16(PAT_ALL);"}, {},
                 {{"dtype", ir::DataType::FP16}}, "mask2");
    ExpectInvoke(codegen, "vf.create_mask", {"MaskReg mask_default;", "mask_default = pset_b32(PAT_ALL);"}, {}, {},
                 "mask_default");
    ExpectInvoke(codegen, "vf.create_mask", {"pset_b8("}, {},
                 {{"pattern", EnumValue(ir::MaskPattern::ALL)}, {"dtype", ir::DataType::FP8E4M3FN}}, "mask_fp8");
    ExpectInvoke(codegen, "vf.create_mask", {"pset_b8("}, {},
                 {{"pattern", EnumValue(ir::MaskPattern::ALL)}, {"dtype", ir::DataType::FP4E2M1}}, "mask_fp4");
    // b64 (INT64/UINT64): pset_b32 + punpack(LOWER)
    ExpectInvoke(codegen, "vf.create_mask", {"pset_b32(", "punpack(", "LOWER"}, {},
                 {{"pattern", EnumValue(ir::MaskPattern::ALL)}, {"dtype", ir::DataType::INT64}}, "mask_i64");
    // b64 H pattern remapped to VL16 before pset_b32 + punpack
    ExpectInvoke(codegen, "vf.create_mask", {"pset_b32(PAT_VL16)", "punpack("}, {},
                 {{"pattern", EnumValue(ir::MaskPattern::H)}, {"dtype", ir::DataType::INT64}}, "mask_i64_h");
    // b64 Q pattern remapped to VL8 before pset_b32 + punpack
    ExpectInvoke(codegen, "vf.create_mask", {"pset_b32(PAT_VL8)", "punpack("}, {},
                 {{"pattern", EnumValue(ir::MaskPattern::Q)}, {"dtype", ir::DataType::UINT64}}, "mask_u64_q");
    EXPECT_TRUE(codegen.IsMaskRegVar("mask"));

    ExpectInvoke(codegen, "vf.full", {"vbr(fp, 2.500000);"}, {fp, Float(2.5)});
    // full rejects MERGING on current device (vdup/vbr have no merging form)
    EXPECT_ANY_THROW(Invoke(codegen, "vf.full", {fp, Float(1.0), mask}, {{"mode", EnumValue(ir::MergeMode::MERGING)}}));
    codegen.RegisterRegTensorVar("s4");
    ExpectInvoke(codegen, "vf.full", {"POS_HIGHEST", "MODE_ZEROING"}, {s4, s4},
                 {{"pos", EnumValue(ir::DuplicatePos::HIGHEST)}});

    ExpectInvoke(codegen, "vf.create_addr_reg", {"AddrReg addr = vag_b32((4) * 2, (8) * 2);"}, {Int(4), Int(8)},
                 {{"dtype", ir::DataType::INT64}}, "addr");
    EXPECT_TRUE(codegen.IsAddrRegVar("addr"));
    ExpectInvoke(codegen, "vf.move", {"vmov(fp, fp);"}, {fp, fp});
    ExpectInvoke(codegen, "vf.move", {"vmov(fp, fp, mask, MODE_MERGING);"}, {fp, fp, mask});
    ExpectInvoke(codegen, "vf.move", {"pmov(mask2, mask);"}, {mask2, mask});
    ExpectInvoke(codegen, "vf.move", {"pmov(mask2, mask, mask);"}, {mask2, mask, mask});
}

TEST(BackendCCEVFOpsTest, EmitsArithmeticIntrinsics)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto dst = MakeVar("dst");
    auto src0 = MakeVar("src0");
    auto src1 = MakeVar("src1");
    auto int_dst = MakeVar("int_dst", ir::DataType::INT32);
    auto int_src0 = MakeVar("int_src0", ir::DataType::INT32);
    auto int_src1 = MakeVar("int_src1", ir::DataType::INT32);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    auto carry = MakeVar("carry", ir::DataType::UINT32);

    const Kwargs zeroing = {{"mode", EnumValue(ir::MergeMode::ZEROING)}};
    const auto expect_binary = [&](const std::string& name, const std::string& intrinsic) {
        ExpectInvoke(codegen, name, {intrinsic}, {dst, src0, src1, mask}, zeroing);
    };
    expect_binary("vf.max", "vmax(");
    expect_binary("vf.add", "vadd(");
    expect_binary("vf.sub", "vsub(");
    expect_binary("vf.and_", "vand(");
    expect_binary("vf.xor", "vxor(");
    expect_binary("vf.or_", "vor(");
    expect_binary("vf.mul", "vmul(");
    expect_binary("vf.mul_add_dst", "vmula(");
    expect_binary("vf.div", "vdiv(");
    expect_binary("vf.min", "vmin(");
    expect_binary("vf.abs_sub", "vabsdif(");
    expect_binary("vf.mul_dst_add", "vmadd(");
    expect_binary("vf.prelu", "vprelu(");

    // B64 bitwise ops (mirrors AscendC b64 path): b64 mask packed via ppack,
    // operands deinterleaved into u32 lo/hi halves, b32 vand/vxor/vor/vnot per
    // half with the packed mask, halves re-interleaved into the b64 dst.
    auto dst64 = MakeVar("dst64", ir::DataType::INT64);
    auto src64a = MakeVar("src64a", ir::DataType::INT64);
    auto src64b = MakeVar("src64b", ir::DataType::INT64);
    auto mask64 = MakeVar("mask64", ir::DataType::UINT64);
    ExpectInvoke(codegen, "vf.and_",
                 {"ppack(dst64_and__pm_, mask64, LOWER)", "vdintlv(dst64_and_s0_lo_", "vand(dst64_and__lod_",
                  "vand(dst64_and__hid_", "vintlv((RegTensor<uint32_t>&)dst64"},
                 {dst64, src64a, src64b, mask64}, zeroing);
    ExpectInvoke(codegen, "vf.xor",
                 {"ppack(dst64_xor__pm_, mask64, LOWER)", "vxor(dst64_xor__lod_", "vxor(dst64_xor__hid_",
                  "vintlv((RegTensor<uint32_t>&)dst64"},
                 {dst64, src64a, src64b, mask64}, zeroing);
    ExpectInvoke(codegen, "vf.or_", {"ppack(dst64_or__pm_", "vor(dst64_or__lod_", "vor(dst64_or__hid_"},
                 {dst64, src64a, src64b, mask64}, zeroing);
    ExpectInvoke(codegen, "vf.not_",
                 {"ppack(dst64_not__pm_", "vdintlv(dst64_not_s0_lo_", "vnot(dst64_not__lod_", "vnot(dst64_not__hid_",
                  "vintlv((RegTensor<uint32_t>&)dst64"},
                 {dst64, src64a, mask64}, zeroing);

    const auto expect_unary = [&](const std::string& name, const std::vector<std::string>& expected) {
        ExpectInvoke(codegen, name, expected, {dst, src0, mask}, zeroing);
    };
    expect_unary("vf.ln", {"vln("});
    expect_unary("vf.log", {"vln("});
    expect_unary("vf.exp", {"vexp("});
    expect_unary("vf.abs", {"vabs("});
    expect_unary("vf.not_", {"vnot("});
    expect_unary("vf.sqrt", {"vsqrt("});
    ExpectInvoke(codegen, "vf.sqrt", {"vcmps_lt(", "16777216.0f", "0.000244140625f", "vsel("}, {dst, src0, mask},
                 {{"precision", true}});
    ExpectInvoke(codegen, "vf.exp", {"vexp(", "vcmps_le(", "vdup(", "vdiv(", "vmul(", "vsel("}, {dst, src0, mask},
                 {{"precision", true}});
    ExpectInvoke(codegen, "vf.ln", {"vcmps_lt(", "8388608.0f", "-15.9423851528787421f", "vsel("}, {dst, src0, mask},
                 {{"precision", true}});
    ExpectInvoke(codegen, "vf.log", {"vcmps_lt(", "8388608.0f", "-15.9423851528787421f", "vsel("}, {dst, src0, mask},
                 {{"precision", true}});
    ExpectInvoke(codegen, "vf.log2", {"vcmps_lt(", "8388608.0f", "1.4426950408889634f", "-23.0f", "vsel("},
                 {dst, src0, mask}, {{"precision", true}});
    ExpectInvoke(codegen, "vf.log10",
                 {"vcmps_lt(", "8388608.0f", "0.43429448190325176f", "-6.923689900271567f", "vsel("}, {dst, src0, mask},
                 {{"precision", true}});
    ExpectInvoke(codegen, "vf.div", {"vdiv(", "vmuls(", "vmula(", "vcmps_ge(", "vabs("}, {dst, src0, src1, mask},
                 {{"precision", true}});
    expect_unary("vf.relu", {"vrelu("});
    expect_unary("vf.neg", {"vneg("});
    expect_unary("vf.log2", {"vln(", "1.4426950408889634f"});
    expect_unary("vf.log10", {"vln(", "0.4342944819032518f"});

    const auto expect_scalar = [&](const std::string& name, const std::string& intrinsic) {
        ExpectInvoke(codegen, name, {intrinsic}, {dst, src0, Float(0.5), mask}, zeroing);
    };
    expect_scalar("vf.muls", "vmuls(");
    expect_scalar("vf.adds", "vadds(");
    expect_scalar("vf.mins", "vmins(");
    expect_scalar("vf.maxs", "vmaxs(");
    expect_scalar("vf.leaky_relu", "vlrelu(");
    auto fp32_src = MakeVar("fp32_src", ir::DataType::FP32);
    auto fp16_dst = MakeVar("fp16_dst", ir::DataType::FP16);
    codegen.RegisterRegTensorVar("fp32_src");
    codegen.RegisterRegTensorVar("fp16_dst");
    ExpectInvoke(codegen, "vf.muls_cast", {"vmulscvt("}, {fp16_dst, fp32_src, Float(0.5), mask},
                 {{"mode", EnumValue(ir::MergeMode::ZEROING)}, {"dtype", ir::DataType::FP16}});
    auto f8_src = MakeVar("f8_src", ir::DataType::FP8E4M3FN);
    codegen.RegisterRegTensorVar("f8_src");
    ExpectInvoke(codegen, "vf.full", {"vdup("}, {f8_src, f8_src, mask},
                 {{"mode", EnumValue(ir::MergeMode::ZEROING)}, {"pos", EnumValue(ir::DuplicatePos::LOWEST)}});
}

TEST(BackendCCEVFOpsTest, EmitsReductionAndPermutationIntrinsics)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto dst = MakeVar("dst");
    auto dst2 = MakeVar("dst2");
    auto src0 = MakeVar("src0");
    auto src1 = MakeVar("src1");
    auto int_dst = MakeVar("int_dst", ir::DataType::INT32);
    auto int_src0 = MakeVar("int_src0", ir::DataType::INT32);
    auto int_src1 = MakeVar("int_src1", ir::DataType::INT32);
    auto int_src = MakeVar("int_src", ir::DataType::INT32);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    auto carry = MakeVar("carry", ir::DataType::UINT32);
    codegen.RegisterRegTensorVar("int_src");
    const Kwargs zeroing = {{"mode", EnumValue(ir::MergeMode::ZEROING)}};

    ExpectInvoke(codegen, "vf.reduce_sum", {"vcgadd("}, {dst, src0, mask},
                 {{"datablock", true}, {"merge_mode", EnumValue(ir::MergeMode::ZEROING)}});
    ExpectInvoke(codegen, "vf.reduce_max", {"vcmax("}, {dst, src0, mask});
    ExpectInvoke(codegen, "vf.reduce_min", {"vcgmin("}, {dst, src0, mask}, {{"datablock", true}});
    ExpectInvoke(codegen, "vf.interleave", {"vintlv("}, {dst, dst2, src0, src1});
    ExpectInvoke(codegen, "vf.de_interleave", {"vdintlv("}, {dst, dst2, src0, src1});
    ExpectInvoke(codegen, "vf.pair_reduce_sum", {"vcpadd("}, {dst, src0, mask}, zeroing);
    ExpectInvoke(codegen, "vf.axpy", {"vaxpy("}, {dst, src0, Float(0.25), mask}, zeroing);
    ExpectInvoke(codegen, "vf.shift_left", {"vshls("}, {int_dst, int_src0, Int(2), mask}, zeroing);
    ExpectInvoke(codegen, "vf.shift_right", {"vshr("}, {int_dst, int_src0, int_src, mask}, zeroing);
    ExpectInvoke(codegen, "vf.mull", {"vmull("}, {int_dst, int_dst, int_src0, int_src1, mask});
    ExpectInvoke(codegen, "vf.addc", {"vaddcs("}, {carry, int_dst, int_src0, int_src1, mask, mask});
    ExpectInvoke(codegen, "vf.subc", {"vsubcs("}, {carry, int_dst, int_src0, int_src1, mask, mask});
    ExpectInvoke(codegen, "vf.exp_sub", {"vexpdif(", "PART_ODD"}, {dst, src0, src1, mask},
                 {{"layout", EnumValue(ir::CastLayout::ONE)}, {"dtype", ir::DataType::FP32}});
    ExpectInvoke(codegen, "vf.select", {"vsel("}, {dst, src0, int_src, mask});
    ExpectInvoke(codegen, "vf.mem_bar", {"mem_bar(VV_ALL)"}, {}, {{"mode", EnumValue(ir::MemBarMode::VV_ALL)}});
}

TEST(BackendCCEVFOpsTest, EmitsPackAndCastIntrinsics)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto fp32 = MakeVar("fp32", ir::DataType::FP32);
    auto fp16 = MakeVar("fp16", ir::DataType::FP16);
    auto bf16 = MakeVar("bf16", ir::DataType::BF16);
    auto i64 = MakeVar("i64", ir::DataType::INT64);
    auto i32 = MakeVar("i32", ir::DataType::INT32);
    auto i16 = MakeVar("i16", ir::DataType::INT16);
    auto i8 = MakeVar("i8", ir::DataType::INT8);
    auto u16 = MakeVar("u16", ir::DataType::UINT16);
    auto u32 = MakeVar("u32", ir::DataType::UINT32);
    auto u8 = MakeVar("u8", ir::DataType::UINT8);
    auto s4 = MakeVar("s4", ir::DataType::INT4);
    auto f8e4m3 = MakeVar("f8e4m3", ir::DataType::FP8E4M3FN);
    auto f8e5m2 = MakeVar("f8e5m2", ir::DataType::FP8E5M2);
    auto hf8 = MakeVar("hf8", ir::DataType::HF8);
    auto f4e2m1 = MakeVar("f4e2m1", ir::DataType::FP4E2M1);
    auto f4e1m2 = MakeVar("f4e1m2", ir::DataType::FP4E1M2);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    for (const auto& var :
         {fp32, fp16, bf16, i64, i32, i16, i8, u16, u32, u8, s4, f8e4m3, f8e5m2, hf8, f4e2m1, f4e1m2}) {
        codegen.RegisterRegTensorVar(var->name_);
    }

    ExpectInvoke(codegen, "vf.pack", {"vpack(", "HIGHER"}, {u16, u32},
                 {{"part", EnumValue(ir::PackPart::UPPER)}, {"dtype", ir::DataType::UINT16}});
    ExpectInvoke(codegen, "vf.pack", {"vdintlv("}, {u32, i64}, {{"dtype", ir::DataType::UINT32}});
    ExpectInvoke(codegen, "vf.unpack", {"vunpack(", "HIGHER"}, {u32, u16},
                 {{"part", EnumValue(ir::PackPart::UPPER)}, {"dtype", ir::DataType::UINT32}});
    ExpectInvoke(codegen, "vf.unpack", {"vintlv("}, {i64, i32}, {{"dtype", ir::DataType::INT64}});

    const Kwargs cast_options = {{"layout", EnumValue(ir::CastLayout::ONE)},
                                 {"round_mode", EnumValue(ir::VFRoundMode::CAST_FLOOR)},
                                 {"saturate", EnumValue(ir::SaturateMode::ON)}};
    auto with_dtype = [](const Kwargs& base, ir::DataType dt) {
        Kwargs result = base;
        result.emplace_back("dtype", dt);
        return result;
    };
    // FP16→FP32 (2x widening): no round_mode/saturate
    ExpectInvoke(codegen, "vf.astype", {"vcvt(fp32, fp16, mask, PART_ODD, MODE_ZEROING);"}, {fp32, fp16, mask},
                 {{"layout", EnumValue(ir::CastLayout::ONE)}, {"dtype", ir::DataType::FP32}});
    // INT32→FP32 (same-width int→float): layout=ZERO, no saturate
    ExpectInvoke(codegen, "vf.astype", {"vcvt(fp32, i32, mask, ROUND_F, MODE_ZEROING);"}, {fp32, i32, mask},
                 {{"layout", EnumValue(ir::CastLayout::ZERO)},
                  {"round_mode", EnumValue(ir::VFRoundMode::CAST_FLOOR)},
                  {"dtype", ir::DataType::FP32}});
    // FP32→INT32 (same-width float→int): layout=ZERO
    ExpectInvoke(codegen, "vf.astype", {"vcvt(i32, fp32, mask, ROUND_F, RS_ENABLE, MODE_ZEROING);"}, {i32, fp32, mask},
                 {{"layout", EnumValue(ir::CastLayout::ZERO)},
                  {"round_mode", EnumValue(ir::VFRoundMode::CAST_FLOOR)},
                  {"saturate", EnumValue(ir::SaturateMode::ON)},
                  {"dtype", ir::DataType::INT32}});
    // FP16→INT32 (wider int): uses ROUND + PART
    ExpectInvoke(codegen, "vf.astype", {"vcvt(i32, fp16, mask, ROUND_F, PART_ODD, MODE_ZEROING);"}, {i32, fp16, mask},
                 with_dtype(cast_options, ir::DataType::INT32));
    // INT32→FP16 (cross-width): uses ROUND + PART
    ExpectInvoke(codegen, "vf.astype", {"vcvt(fp16, i32, mask, ROUND_F, PART_ODD, MODE_ZEROING);"}, {fp16, i32, mask},
                 with_dtype(cast_options, ir::DataType::FP16));
    // INT32→INT16 (int narrowing): no round_mode, uses RS + PART
    ExpectInvoke(codegen, "vf.astype", {"vcvt(i16, i32, mask, RS_ENABLE, PART_ODD, MODE_ZEROING);"}, {i16, i32, mask},
                 {{"layout", EnumValue(ir::CastLayout::ONE)},
                  {"saturate", EnumValue(ir::SaturateMode::ON)},
                  {"dtype", ir::DataType::INT16}});
    // INT4→FP16 (s4 widening): no round_mode/saturate
    ExpectInvoke(codegen, "vf.astype", {"vcvt_s42f16(fp16, s4, mask, PART_P1, MODE_ZEROING);"}, {fp16, s4, mask},
                 {{"layout", EnumValue(ir::CastLayout::ONE)}, {"dtype", ir::DataType::FP16}});
    ExpectInvoke(codegen, "vf.astype", {"vcvt_f162s4(s4, fp16, mask, ROUND_F, RS_ENABLE, PART_P1, MODE_ZEROING);"},
                 {s4, fp16, mask}, with_dtype(cast_options, ir::DataType::INT4));
    // INT16→INT4 (two-step: s16→f16→s4, mirroring AscendC Cast)
    // Default layout=ZERO → PART_P0, default saturate=OFF → RS_DISABLE
    ExpectInvoke(codegen, "vf.astype",
                 {"vcvt(s4_f16_tmp, i16, mask, ROUND_R, MODE_ZEROING);",
                  "vcvt_f162s4(s4, s4_f16_tmp, mask, ROUND_R, RS_DISABLE, PART_P0, MODE_ZEROING);"},
                 {s4, i16, mask}, {{"dtype", ir::DataType::INT4}});
    // int→int two-step through f16 (cross-sign widening, int→INT8 narrowing)
    // UINT8→INT16: u8→f16 (widening, 5 args) → f16→s16 (same-int, 5 args)
    ExpectInvoke(codegen, "vf.astype",
                 {"vcvt(i16_f16_tmp, u8, mask, PART_EVEN, MODE_ZEROING);",
                  "vcvt(i16, i16_f16_tmp, mask, ROUND_R, RS_DISABLE, MODE_ZEROING);"},
                 {i16, u8, mask}, {{"dtype", ir::DataType::INT16}});
    // INT32→INT8: three-step s32→f32→f16→s8
    ExpectInvoke(codegen, "vf.astype",
                 {"vcvt(i8_f32_tmp, i32, mask, ROUND_R, MODE_ZEROING);",
                  "vcvt(i8_f16_tmp, i8_f32_tmp, mask, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);",
                  "vcvt(i8, i8_f16_tmp, mask, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);"},
                 {i8, i32, mask}, {{"dtype", ir::DataType::INT8}});
    // UINT16→INT32: u16→u32 widening with dst cast (zero-extend, semantically correct)
    ExpectInvoke(codegen, "vf.astype", {"vcvt((RegTensor<uint32_t> &)i32, u16, mask, PART_EVEN, MODE_ZEROING);"},
                 {i32, u16, mask}, {{"dtype", ir::DataType::INT32}});
    // FP8/FP4 low-precision conversions
    // 4x widening PP (5-arg): FP8→FP32, FP4→BF16 → vcvt(dst,src,mask,PART_PP,MODE)
    // Widening paths: round_mode and saturate are not applicable
    const Kwargs widen_options = {{"layout", EnumValue(ir::CastLayout::ONE)}};
    ExpectInvoke(codegen, "vf.astype", {"vcvt(fp32, f8e4m3, mask, PART_P1, MODE_ZEROING);"}, {fp32, f8e4m3, mask},
                 widen_options);
    ExpectInvoke(codegen, "vf.astype", {"vcvt(fp32, f8e5m2, mask, PART_P1, MODE_ZEROING);"}, {fp32, f8e5m2, mask},
                 widen_options);
    ExpectInvoke(codegen, "vf.astype", {"vcvt(bf16, f4e2m1, mask, PART_P1, MODE_ZEROING);"}, {bf16, f4e2m1, mask},
                 widen_options);
    // 2x widening PART (5-arg): HF8→FP16 → vcvt(dst,src,mask,PART,MODE)
    ExpectInvoke(codegen, "vf.astype", {"vcvt(fp16, hf8, mask, PART_ODD, MODE_ZEROING);"}, {fp16, hf8, mask},
                 widen_options);
    // 4x narrowing RND_SAT_PP (7-arg): FP32→FP8 → vcvt(dst,src,mask,ROUND,SAT,PART_PP,MODE)
    // FP32→FP8E4M3FN: round_mode must be CAST_RINT
    ExpectInvoke(codegen, "vf.astype", {"vcvt(f8e4m3, fp32, mask, ROUND_R, RS_ENABLE, PART_P1, MODE_ZEROING);"},
                 {f8e4m3, fp32, mask},
                 {{"layout", EnumValue(ir::CastLayout::ONE)},
                  {"round_mode", EnumValue(ir::VFRoundMode::CAST_RINT)},
                  {"saturate", EnumValue(ir::SaturateMode::ON)}});
    // 4x narrowing RND_PP (6-arg): BF16→FP4 → vcvt(dst,src,mask,ROUND,PART_PP,MODE)
    // BF16→FP4: saturate not applicable
    ExpectInvoke(codegen, "vf.astype", {"vcvt(f4e2m1, bf16, mask, ROUND_F, PART_P1, MODE_ZEROING);"},
                 {f4e2m1, bf16, mask},
                 {{"layout", EnumValue(ir::CastLayout::ONE)}, {"round_mode", EnumValue(ir::VFRoundMode::CAST_FLOOR)}});
    // 2x narrowing RND_SAT_PART (7-arg): FP16→HF8 → vcvt(dst,src,mask,ROUND,SAT,PART,MODE)
    // FP16→HF8: round_mode must be CAST_ROUND/CAST_HYBRID
    ExpectInvoke(codegen, "vf.astype", {"vcvt(hf8, fp16, mask, ROUND_A, RS_ENABLE, PART_ODD, MODE_ZEROING);"},
                 {hf8, fp16, mask},
                 {{"layout", EnumValue(ir::CastLayout::ONE)},
                  {"round_mode", EnumValue(ir::VFRoundMode::CAST_ROUND)},
                  {"saturate", EnumValue(ir::SaturateMode::ON)}});
    // FP8E5M2 / FP4E1M2 narrowing casts with dtype
    // FP32→FP8E5M2: round_mode must be CAST_RINT
    ExpectInvoke(codegen, "vf.astype", {"vcvt(f8e5m2, fp32, mask, ROUND_R, RS_ENABLE, PART_P1, MODE_ZEROING);"},
                 {f8e5m2, fp32, mask},
                 {{"layout", EnumValue(ir::CastLayout::ONE)},
                  {"round_mode", EnumValue(ir::VFRoundMode::CAST_RINT)},
                  {"saturate", EnumValue(ir::SaturateMode::ON)},
                  {"dtype", ir::DataType::FP8E5M2}});
    // BF16→FP4E1M2: saturate not applicable
    ExpectInvoke(codegen, "vf.astype", {"vcvt(f4e1m2, bf16, mask, ROUND_F, PART_P1, MODE_ZEROING);"},
                 {f4e1m2, bf16, mask},
                 {{"layout", EnumValue(ir::CastLayout::ONE)},
                  {"round_mode", EnumValue(ir::VFRoundMode::CAST_FLOOR)},
                  {"dtype", ir::DataType::FP4E1M2}});
    // layout TWO/THREE → PART_P2/PART_P3 (only for 4x narrowing: FP32→FP8)
    const Kwargs layout_two = {{"layout", EnumValue(ir::CastLayout::TWO)},
                               {"round_mode", EnumValue(ir::VFRoundMode::CAST_RINT)},
                               {"saturate", EnumValue(ir::SaturateMode::ON)},
                               {"dtype", ir::DataType::FP8E4M3FN}};
    ExpectInvoke(codegen, "vf.astype", {"PART_P2"}, {f8e4m3, fp32, mask}, layout_two);
    const Kwargs layout_three = {{"layout", EnumValue(ir::CastLayout::THREE)},
                                 {"round_mode", EnumValue(ir::VFRoundMode::CAST_RINT)},
                                 {"saturate", EnumValue(ir::SaturateMode::ON)},
                                 {"dtype", ir::DataType::FP8E4M3FN}};
    ExpectInvoke(codegen, "vf.astype", {"PART_P3"}, {f8e4m3, fp32, mask}, layout_three);
}

// ============================================================================
// Tests for new validation CHECKs added for data type consistency and
// parameter combination constraints.
// ============================================================================

TEST(BackendCCEVFOpsTest, RejectsMismatchedSrcDstTypes)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto fp32_dst = MakeVar("fp32_dst", ir::DataType::FP32);
    auto fp16_src = MakeVar("fp16_src", ir::DataType::FP16);
    auto mask = MakeVar("mask", ir::DataType::UINT32);

    // vf.add: dst=FP32, src0=FP16, src1=FP16 → should reject
    EXPECT_ANY_THROW(Invoke(codegen, "vf.add", {fp32_dst, fp16_src, fp16_src, mask}));

    // vf.sub: dst=FP32, src0=FP16, src1=FP16 → should reject
    EXPECT_ANY_THROW(Invoke(codegen, "vf.sub", {fp32_dst, fp16_src, fp16_src, mask}));

    // vf.move: dst=FP32, src=FP16 → should reject
    EXPECT_ANY_THROW(Invoke(codegen, "vf.move", {fp32_dst, fp16_src, mask}));

    // vf.abs: dst=FP32, src=FP16 → should reject
    EXPECT_ANY_THROW(Invoke(codegen, "vf.abs", {fp32_dst, fp16_src, mask}));

    // vf.truncate: dst=INT32, src=FP32 → should reject
    auto i32_dst = MakeVar("i32_dst", ir::DataType::INT32);
    auto fp32_src = MakeVar("fp32_src", ir::DataType::FP32);
    EXPECT_ANY_THROW(Invoke(codegen, "vf.truncate", {i32_dst, fp32_src, mask}));

    // vf.xor: dst=32-bit, src0=16-bit → should reject (bit width mismatch)
    auto fp16_dst = MakeVar("fp16_dst", ir::DataType::FP16);
    EXPECT_ANY_THROW(Invoke(codegen, "vf.xor", {fp32_dst, fp16_src, fp16_dst, mask}));
}

TEST(BackendCCEVFOpsTest, RejectsMergingForZeroingOnlyOps)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto dst = MakeVar("dst", ir::DataType::FP32);
    auto src0 = MakeVar("src0", ir::DataType::FP32);
    auto src1 = MakeVar("src1", ir::DataType::FP32);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    const Kwargs merging = {{"mode", EnumValue(ir::MergeMode::MERGING)}};

    // Ops that only support ZEROING should reject MERGING
    EXPECT_ANY_THROW(Invoke(codegen, "vf.sub", {dst, src0, src1, mask}, merging));
    EXPECT_ANY_THROW(Invoke(codegen, "vf.mul", {dst, src0, src1, mask}, merging));
    EXPECT_ANY_THROW(Invoke(codegen, "vf.div", {dst, src0, src1, mask}, merging));
    EXPECT_ANY_THROW(Invoke(codegen, "vf.add", {dst, src0, src1, mask}, merging));
    EXPECT_ANY_THROW(Invoke(codegen, "vf.max", {dst, src0, src1, mask}, merging));
    EXPECT_ANY_THROW(Invoke(codegen, "vf.min", {dst, src0, src1, mask}, merging));
    EXPECT_ANY_THROW(Invoke(codegen, "vf.full", {dst, Float(1.0), mask}, merging));
    EXPECT_ANY_THROW(Invoke(codegen, "vf.and_", {dst, src0, src1, mask}, merging));
    EXPECT_ANY_THROW(Invoke(codegen, "vf.xor", {dst, src0, src1, mask}, merging));
    EXPECT_ANY_THROW(Invoke(codegen, "vf.adds", {dst, src0, mask}, merging));
    EXPECT_ANY_THROW(Invoke(codegen, "vf.abs", {dst, src0, mask}, merging));
    EXPECT_ANY_THROW(Invoke(codegen, "vf.sqrt", {dst, src0, mask}, merging));
    EXPECT_ANY_THROW(Invoke(codegen, "vf.truncate", {dst, src0, mask}, merging));
    EXPECT_ANY_THROW(Invoke(codegen, "vf.astype", {dst, src0, mask},
                            {{"mode", EnumValue(ir::MergeMode::MERGING)}, {"dtype", ir::DataType::FP16}}));

    // vf.move only supports MERGING, should reject ZEROING
    const Kwargs zeroing = {{"mode", EnumValue(ir::MergeMode::ZEROING)}};
    EXPECT_ANY_THROW(Invoke(codegen, "vf.move", {dst, src0, mask}, zeroing));
}

TEST(BackendCCEVFOpsTest, RejectsInvalidParameterCombinations)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::FP16);
    auto tile8 = MakeTile("tile8", ir::DataType::UINT8);
    auto fp16 = MakeVar("fp16", ir::DataType::FP16);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    auto addr = MakeVar("addr", ir::DataType::INT64);
    codegen.RegisterAddrRegVar("addr");

    // load_align 2-arg with data_copy_mode → should reject
    EXPECT_ANY_THROW(Invoke(codegen, "vf.load_align", {fp16, tile},
                            {{"data_copy_mode", EnumValue(ir::DataCopyMode::DATA_BLOCK_COPY)}}));

    // load_align 4-arg (de-interleave) with data_copy_mode → should reject
    auto fp16b = MakeVar("fp16b", ir::DataType::FP16);
    EXPECT_ANY_THROW(Invoke(codegen, "vf.load_align", {fp16, fp16b, tile, Int(0)},
                            {{"data_copy_mode", EnumValue(ir::DataCopyMode::DATA_BLOCK_COPY)}}));

    // store_align MaskReg with data_copy_mode → should reject
    codegen.RegisterMaskRegVar("mask");
    EXPECT_ANY_THROW(Invoke(codegen, "vf.store_align", {tile, mask},
                            {{"data_copy_mode", EnumValue(ir::DataCopyMode::DATA_BLOCK_COPY)}}));

    // store_align INTLV with data_copy_mode → should reject
    auto fp16c = MakeVar("fp16c", ir::DataType::FP16);
    EXPECT_ANY_THROW(Invoke(
        codegen, "vf.store_align", {tile, fp16, fp16c, mask},
        {{"dist", EnumValue(ir::StoreDist::INTLV)}, {"data_copy_mode", EnumValue(ir::DataCopyMode::DATA_BLOCK_COPY)}}));

    // gather reg→reg with 4 args (mask) → should reject
    auto u16_idx = MakeVar("u16_idx", ir::DataType::UINT16);
    EXPECT_ANY_THROW(Invoke(codegen, "vf.gather", {fp16, fp16b, u16_idx, mask}));

    // gather reg→reg with data_copy_mode → should reject
    EXPECT_ANY_THROW(Invoke(codegen, "vf.gather", {fp16, fp16b, u16_idx},
                            {{"data_copy_mode", EnumValue(ir::DataCopyMode::DATA_BLOCK_LOAD)}}));
}

TEST(BackendCCEVFOpsTest, RejectsInvalidAstypeRoundMode)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto fp16 = MakeVar("fp16", ir::DataType::FP16);
    auto fp32 = MakeVar("fp32", ir::DataType::FP32);
    auto mask = MakeVar("mask", ir::DataType::UINT32);

    // FP16→FP32 widening with round_mode=CAST_ROUND → should reject
    EXPECT_ANY_THROW(Invoke(codegen, "vf.astype", {fp32, fp16, mask},
                            {{"layout", EnumValue(ir::CastLayout::ZERO)},
                             {"round_mode", EnumValue(ir::VFRoundMode::CAST_ROUND)},
                             {"saturate", EnumValue(ir::SaturateMode::ON)},
                             {"dtype", ir::DataType::FP32}}));

    // FP32→FP8E4M3FN with round_mode=CAST_ROUND (only CAST_RINT allowed) → should reject
    auto f8_dst = MakeVar("f8_dst", ir::DataType::FP8E4M3FN);
    EXPECT_ANY_THROW(Invoke(codegen, "vf.astype", {f8_dst, fp32, mask},
                            {{"layout", EnumValue(ir::CastLayout::ZERO)},
                             {"round_mode", EnumValue(ir::VFRoundMode::CAST_ROUND)},
                             {"saturate", EnumValue(ir::SaturateMode::ON)},
                             {"dtype", ir::DataType::FP8E4M3FN}}));

    // INT32→FP32 (is_int_to_float) with round_mode=CAST_ODD → should reject
    auto i32 = MakeVar("i32", ir::DataType::INT32);
    EXPECT_ANY_THROW(Invoke(codegen, "vf.astype", {fp32, i32, mask},
                            {{"layout", EnumValue(ir::CastLayout::ZERO)},
                             {"round_mode", EnumValue(ir::VFRoundMode::CAST_ODD)},
                             {"dtype", ir::DataType::FP32}}));
}

TEST(BackendCCEVFOpsTest, RejectsInvalidAstypeSaturateAndLayout)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto fp16 = MakeVar("fp16", ir::DataType::FP16);
    auto fp32 = MakeVar("fp32", ir::DataType::FP32);
    auto bf16 = MakeVar("bf16", ir::DataType::BF16);
    auto mask = MakeVar("mask", ir::DataType::UINT32);

    // FP32→FP8 widening path with saturate=ON → should reject (not applicable for narrowing)
    // Actually FP32→FP8 is narrowing RND_SAT_PP, saturate must be ON.
    // Test the reverse: FP8→FP32 widening with saturate=ON → should reject
    auto f8_src = MakeVar("f8_src", ir::DataType::FP8E4M3FN);
    EXPECT_ANY_THROW(Invoke(codegen, "vf.astype", {fp32, f8_src, mask},
                            {{"layout", EnumValue(ir::CastLayout::ZERO)},
                             {"saturate", EnumValue(ir::SaturateMode::ON)},
                             {"dtype", ir::DataType::FP32}}));

    // FP32→INT32 (is_float_to_same_int) with layout=ONE → should reject (layout not applicable)
    auto i32_dst = MakeVar("i32_dst", ir::DataType::INT32);
    EXPECT_ANY_THROW(Invoke(codegen, "vf.astype", {i32_dst, fp32, mask},
                            {{"layout", EnumValue(ir::CastLayout::ONE)},
                             {"round_mode", EnumValue(ir::VFRoundMode::CAST_RINT)},
                             {"saturate", EnumValue(ir::SaturateMode::ON)},
                             {"dtype", ir::DataType::INT32}}));

    // HF8→FP16 widening with layout=TWO → should reject (only ZERO/ONE for 2x widening)
    auto hf8 = MakeVar("hf8", ir::DataType::HF8);
    EXPECT_ANY_THROW(Invoke(codegen, "vf.astype", {fp16, hf8, mask},
                            {{"layout", EnumValue(ir::CastLayout::TWO)}, {"dtype", ir::DataType::FP16}}));
}

TEST(BackendCCEVFOpsTest, EmitsCompareHistogramAndMaskConversions)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto fp32 = MakeVar("fp32", ir::DataType::FP32);
    auto fp16 = MakeVar("fp16", ir::DataType::FP16);
    auto i64 = MakeVar("i64", ir::DataType::INT64);
    auto i32 = MakeVar("i32", ir::DataType::INT32);
    auto u16 = MakeVar("u16", ir::DataType::UINT16);
    auto u8 = MakeVar("u8", ir::DataType::UINT8);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    for (const auto& var : {fp32, fp16, i64, i32, u16, u8}) {
        codegen.RegisterRegTensorVar(var->name_);
    }

    ExpectInvoke(codegen, "vf.update_mask", {"plt_b8("}, {Int(17)}, {{"dtype", ir::DataType::UINT8}}, "mask8");
    ExpectInvoke(codegen, "vf.update_mask", {"plt_b16("}, {Int(17)}, {{"dtype", ir::DataType::FP16}}, "mask16");
    ExpectInvoke(codegen, "vf.update_mask", {"plt_b32("}, {Int(17)}, {}, "mask32");
    // b64 update_mask: plt_b32 + punpack(LOWER)
    ExpectInvoke(codegen, "vf.update_mask", {"plt_b32(", "punpack(", "LOWER"}, {Int(17)},
                 {{"dtype", ir::DataType::INT64}}, "mask64");
    ExpectInvoke(codegen, "vf.histograms", {"dhistv2("}, {u16, u8, mask},
                 {{"bin_type", EnumValue(ir::BinType::BIN1)}, {"hist_type", EnumValue(ir::HistType::FREQUENCY)}});
    ExpectInvoke(codegen, "vf.histograms", {"chistv2("}, {u16, u8, mask}, {{"bin_type", EnumValue(ir::BinType::BIN0)}});

    ExpectInvoke(codegen, "vf.eq", {"vcmps_eq("}, {mask, fp32, Float(1.0), mask});
    // Vector-vector compare requires src0/src1 to have the same type (AscendC
    // CompareImpl takes both sources as one register type U).
    ExpectInvoke(codegen, "vf.ne", {"vcmp_ne("}, {mask, fp32, fp32, mask});
    ExpectInvoke(codegen, "vf.lt", {"vcmp_lt("}, {mask, fp32, fp32, mask});
    ExpectInvoke(codegen, "vf.gt", {"vcmp_gt("}, {mask, fp32, fp32, mask});
    ExpectInvoke(codegen, "vf.le", {"vcmp_le("}, {mask, fp32, fp32, mask});
    ExpectInvoke(codegen, "vf.ge", {"vcmp_ge("}, {mask, fp32, fp32, mask});
    ExpectInvoke(codegen, "vf.squeeze", {"vsqz(", "MODE_NO_STORED"}, {i32, fp16, mask},
                 {{"gather_mode", EnumValue(ir::SqueezeMode::NO_STORE_REG)}});
    ExpectInvoke(codegen, "vf.arange", {"vci(i32, 3, DEC_ORDER)"}, {i32, Int(3)},
                 {{"index_order", EnumValue(ir::IndexOrder::DECREASE_ORDER)}, {"dtype", ir::DataType::INT32}});
    ExpectInvoke(codegen, "vf.arange", {"vci(i64_b64_lo_"}, {i64, Int(5)}, {{"dtype", ir::DataType::INT64}});
    ExpectInvoke(codegen, "vf.unsqueeze", {"vusqz("}, {i32, mask});
    ExpectInvoke(codegen, "vf.truncate", {"vtrc(fp32, fp32, ROUND_C, mask, MODE_ZEROING)"}, {fp32, fp32, mask},
                 {{"round_mode", EnumValue(ir::VFRoundMode::CAST_CEIL)}, {"mode", EnumValue(ir::MergeMode::ZEROING)}});
}

TEST(BackendCCEVFOpsTest, EmitsAlignedDataMovement)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::FP16);
    auto tile8 = MakeTile("tile8", ir::DataType::UINT8);
    auto tile64 = MakeTile("tile64", ir::DataType::INT64);
    auto fp16 = MakeVar("fp16", ir::DataType::FP16);
    auto fp16b = MakeVar("fp16b", ir::DataType::FP16);
    auto u8 = MakeVar("u8", ir::DataType::UINT8);
    auto i64 = MakeVar("i64", ir::DataType::INT64);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    auto addr = MakeVar("addr", ir::DataType::INT64);
    auto f8 = MakeVar("f8", ir::DataType::FP8E4M3FN);
    codegen.RegisterAddrRegVar("addr");
    codegen.RegisterMaskRegVar("mask");
    codegen.RegisterRegTensorVar("f8");

    ExpectInvoke(codegen, "vf.load_align", {"vlds(fp16"}, {fp16, tile, Int(0)});
    ExpectInvoke(codegen, "vf.load_align", {"BRC_B8"}, {u8, tile8, Int(1)}, {{"dist", EnumValue(ir::LoadDist::BRC)}});
    ExpectInvoke(codegen, "vf.load_align", {"(2) * 2", "POST_UPDATE"}, {i64, tile64, Int(2)}, {{"post_update", true}});
    ExpectInvoke(codegen, "vf.load_align", {"vld(fp16"}, {fp16, tile, addr});
    ExpectInvoke(codegen, "vf.load_align", {"plds(mask"}, {mask, tile, Int(0)},
                 {{"dist", EnumValue(ir::LoadDist::US)}});
    ExpectInvoke(codegen, "vf.load_align", {"DINTLV_B16", "POST_UPDATE"}, {fp16, fp16b, tile, Int(3)},
                 {{"dist", EnumValue(ir::LoadDist::DINTLV_B16)}, {"post_update", true}});
    ExpectInvoke(codegen, "vf.load_align", {"vsldb(", "POST_UPDATE"}, {u8, tile8, mask},
                 {{"data_copy_mode", EnumValue(ir::DataCopyMode::DATA_BLOCK_COPY)},
                  {"block_stride", 2},
                  {"repeat_stride", 3},
                  {"post_update", true}});
    ExpectInvoke(codegen, "vf.load_align", {"BRC_B8"}, {f8, tile8, Int(1)}, {{"dist", EnumValue(ir::LoadDist::BRC)}});

    ExpectInvoke(codegen, "vf.store_align", {"vsts("}, {tile, fp16, mask});
    ExpectInvoke(codegen, "vf.store_align", {"ONEPT_B8"}, {tile8, u8, mask},
                 {{"dist", EnumValue(ir::StoreDist::FIRST_ELEMENT)}});
    ExpectInvoke(codegen, "vf.store_align", {"INTLV_B16"}, {tile, fp16, fp16b, mask},
                 {{"dist", EnumValue(ir::StoreDist::INTLV)}});
    ExpectInvoke(codegen, "vf.store_align", {"vsstb(", "POST_UPDATE"}, {tile, fp16, mask, Int(2), Int(3)},
                 {{"data_copy_mode", EnumValue(ir::DataCopyMode::DATA_BLOCK_COPY)}, {"post_update", true}});
    ExpectInvoke(codegen, "vf.store_align", {"vst(fp16"}, {tile, fp16, mask, addr});
    ExpectInvoke(codegen, "vf.store_align", {"NORM_B8"}, {tile8, f8, mask});
}

TEST(BackendCCEVFOpsTest, EmitsGatherAndUnalignedDataMovement)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::FP16);
    auto tile8 = MakeTile("tile8", ir::DataType::UINT8);
    auto tile64 = MakeTile("tile64", ir::DataType::INT64);
    auto fp16 = MakeVar("fp16", ir::DataType::FP16);
    auto u8 = MakeVar("u8", ir::DataType::UINT8);
    auto i64 = MakeVar("i64", ir::DataType::INT64);
    auto index = MakeVar("index", ir::DataType::UINT32);
    auto index_u16 = MakeVar("index_u16", ir::DataType::UINT16);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    auto ureg = MakeVar("ureg", ir::DataType::INT64);
    codegen.RegisterUnalignRegVar("ureg");

    // b16 src + uint16 index -> vgather2
    ExpectInvoke(codegen, "vf.gather", {"vgather2("}, {fp16, tile, index_u16, mask});
    // b16 src + uint32 index -> vgather2_bc
    ExpectInvoke(codegen, "vf.gather", {"vgather2_bc("}, {fp16, tile, index, mask});
    ExpectInvoke(codegen, "vf.gather", {"vgatherb("}, {fp16, tile, index, mask},
                 {{"data_copy_mode", EnumValue(ir::DataCopyMode::DATA_BLOCK_LOAD)}});
    ExpectInvoke(codegen, "vf.gather", {"vselr("}, {fp16, fp16, index_u16});
    ExpectInvoke(codegen, "vf.scatter", {"vscatter("}, {tile, fp16, index_u16, mask});
    ExpectInvoke(codegen, "vf.load", {"UnalignReg __ureg_ld_", "vldas(", "vldus("}, {fp16, tile});
    ExpectInvoke(codegen, "vf.load", {"UnalignReg __ureg_ld_", "vldas(", "vldus(", "POST_UPDATE"},
                 {i64, tile64, Int(4)});
    ExpectInvoke(codegen, "vf.store", {"UnalignReg __ureg_st_", "vstus(", "vstas("}, {tile, fp16});
    ExpectInvoke(codegen, "vf.store", {"UnalignReg __ureg_st_", "vstus(", "vstas("}, {tile64, i64, Int(7)});

    ExpectInvoke(codegen, "vf.unalign_reg_for_store", {"UnalignReg store_ureg;"}, {}, {}, "store_ureg");
    ExpectInvoke(codegen, "vf.load_unalign_init", {"UnalignReg load_ureg;"}, {}, {}, "load_ureg");
    ExpectInvoke(codegen, "vf.load_unalign_pre", {"vldas("}, {ureg, tile});
    ExpectInvoke(codegen, "vf.load_unalign", {"vldus("}, {fp16, ureg, tile});
    ExpectInvoke(codegen, "vf.load_unalign", {"vldus(", "(4) * 2", "POST_UPDATE"}, {i64, ureg, tile64, Int(4)});
    ExpectInvoke(codegen, "vf.squeeze_store_unalign", {"vstur("}, {tile8, u8, ureg});
    ExpectInvoke(codegen, "vf.store_unalign", {"vstus(", "POST_UPDATE"}, {tile8, u8, ureg, Int(2)},
                 {{"post_update", true}});
    ExpectInvoke(codegen, "vf.squeeze_store_unalign_post", {"vstar("}, {tile8, ureg});
    ExpectInvoke(codegen, "vf.store_unalign_post", {"vstas(", "POST_UPDATE"}, {tile8, ureg, Int(2)},
                 {{"post_update", true}});
    ExpectInvoke(codegen, "vf.clear_spr", {"sprclr(SPR_AR)"});
}

TEST(BackendCCEVFOpsTest, EmitsScatterWithVariousDtypes)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    auto fp16_reg = MakeVar("fp16_reg", ir::DataType::FP16);
    auto int8_reg = MakeVar("int8_reg", ir::DataType::INT8);
    auto uint8_reg = MakeVar("uint8_reg", ir::DataType::UINT8);
    auto int16_reg = MakeVar("int16_reg", ir::DataType::INT16);
    auto int32_reg = MakeVar("int32_reg", ir::DataType::INT32);
    auto uint32_reg = MakeVar("uint32_reg", ir::DataType::UINT32);
    auto fp32_reg = MakeVar("fp32_reg", ir::DataType::FP32);
    auto int64_reg = MakeVar("int64_reg", ir::DataType::INT64);
    auto uint64_reg = MakeVar("uint64_reg", ir::DataType::UINT64);
    auto idx_u16 = MakeVar("idx_u16", ir::DataType::UINT16);
    auto idx_u32 = MakeVar("idx_u32", ir::DataType::UINT32);
    auto idx_u64 = MakeVar("idx_u64", ir::DataType::UINT64);

    // b8 src + u16 index → vscatter with uint16_t index cast
    ExpectInvoke(codegen, "vf.scatter", {"vscatter(", "RegTensor<uint16_t>"},
                 {MakeTile("t8"), int8_reg, idx_u16, mask});
    ExpectInvoke(codegen, "vf.scatter", {"vscatter(", "RegTensor<uint16_t>"},
                 {MakeTile("tu8"), uint8_reg, idx_u16, mask});
    // b16 src + u16 index → vscatter with uint16_t index cast
    ExpectInvoke(codegen, "vf.scatter", {"vscatter(", "RegTensor<uint16_t>"},
                 {MakeTile("t16"), int16_reg, idx_u16, mask});
    ExpectInvoke(codegen, "vf.scatter", {"vscatter(", "RegTensor<uint16_t>"},
                 {MakeTile("th16"), fp16_reg, idx_u16, mask});
    // b32 src + u32 index → vscatter with uint32_t index cast
    ExpectInvoke(codegen, "vf.scatter", {"vscatter(", "RegTensor<uint32_t>"},
                 {MakeTile("ti32"), int32_reg, idx_u32, mask});
    ExpectInvoke(codegen, "vf.scatter", {"vscatter(", "RegTensor<uint32_t>"},
                 {MakeTile("tu32"), uint32_reg, idx_u32, mask});
    ExpectInvoke(codegen, "vf.scatter", {"vscatter(", "RegTensor<uint32_t>"},
                 {MakeTile("tf32"), fp32_reg, idx_u32, mask});
    // b64 src + u32 index → vscatter with uint32_t index cast
    ExpectInvoke(codegen, "vf.scatter", {"vscatter(", "RegTensor<uint32_t>"},
                 {MakeTile("ti64"), int64_reg, idx_u32, mask});
    ExpectInvoke(codegen, "vf.scatter", {"vscatter(", "RegTensor<uint32_t>"},
                 {MakeTile("tu64"), uint64_reg, idx_u32, mask});
    // b64 src + u64 index → vscatter with uint32_t index cast
    ExpectInvoke(codegen, "vf.scatter", {"vscatter(", "RegTensor<uint32_t>"},
                 {MakeTile("ti64b"), int64_reg, idx_u64, mask});
    ExpectInvoke(codegen, "vf.scatter", {"vscatter(", "RegTensor<uint32_t>"},
                 {MakeTile("tu64b"), uint64_reg, idx_u64, mask});
}

TEST(BackendCCEVFOpsTest, EmitsStoreWithUint64Reinterpret)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto u64_tile = MakeTile("tile_u64", ir::DataType::UINT64);
    auto u64_reg = MakeVar("reg_u64", ir::DataType::UINT64);
    auto i64_tile = MakeTile("tile_i64", ir::DataType::INT64);
    auto i64_reg = MakeVar("reg_i64", ir::DataType::INT64);

    // UINT64: vstus/vstas must reinterpret src as uint32_t pairs and double count
    ExpectInvoke(
        codegen, "vf.store",
        {"UnalignReg __ureg_st_", "vstus(", "vstas(", "POST_UPDATE", "(RegTensor<uint32_t>&)", "uint32_t", "* 2"},
        {u64_tile, u64_reg, Int(7)});
    // INT64: same b64 path as UINT64 (u32 reinterpret + count doubling).
    ExpectInvoke(codegen, "vf.store",
                 {"UnalignReg __ureg_st_", "vstus(", "vstas(", "POST_UPDATE", "(RegTensor<uint32_t>&)", "* 2"},
                 {i64_tile, i64_reg, Int(7)});
    // Verify UINT64 uses uint32_t ptr cast (tile is uint64_t, cast to uint32_t)
    auto u64_out = Invoke(codegen, "vf.store", {u64_tile, u64_reg, Int(7)});
    EXPECT_NE(u64_out.find("__ubuf__ uint32_t"), std::string::npos);
}

TEST(BackendCCEVFOpsTest, EmitsStoreWithB8NativeType)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile_u8 = MakeTile("tile_u8", ir::DataType::UINT8);
    auto reg_u8 = MakeVar("reg_u8", ir::DataType::UINT8);
    auto tile_i8 = MakeTile("tile_i8", ir::DataType::INT8);
    auto reg_i8 = MakeVar("reg_i8", ir::DataType::INT8);

    // INT8/UINT8 have direct vstus/vstas overloads, no reinterpret needed.
    // Pointer type matches the tile's native C type (int8_t / uint8_t).
    auto u8_out = Invoke(codegen, "vf.store", {tile_u8, reg_u8, Int(64)});
    ExpectContains(u8_out, {"UnalignReg __ureg_st_", "vstus(", "vstas(", "POST_UPDATE", "reg_u8"});
    auto i8_out = Invoke(codegen, "vf.store", {tile_i8, reg_i8, Int(64)});
    ExpectContains(i8_out, {"UnalignReg __ureg_st_", "vstus(", "vstas(", "POST_UPDATE", "reg_i8"});
}

TEST(BackendCCEVFOpsTest, EmitsStoreWithDefaultCount)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::FP32);
    auto reg = MakeVar("reg", ir::DataType::FP32);

    // Default count = 256/4 = 64 for FP32
    ExpectInvoke(codegen, "vf.store", {"UnalignReg __ureg_st_", "vstus(", "vstas(", "POST_UPDATE", "64"}, {tile, reg});
}

TEST(BackendCCEVFOpsTest, EmitsStoreRejectsExceedCount)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::FP32);
    auto reg = MakeVar("reg", ir::DataType::FP32);

    // FP32 max count = 256/4 = 64; 1000 exceeds the limit
    EXPECT_THROW(Invoke(codegen, "vf.store", {tile, reg, Int(1000)}), std::exception);
    // FP32 max count = 64; 65 just exceeds
    EXPECT_THROW(Invoke(codegen, "vf.store", {tile, reg, Int(65)}), std::exception);
    // FP32 max count = 64; 64 is valid
    ExpectInvoke(codegen, "vf.store", {"vstus(", "64"}, {tile, reg, Int(64)});
}

TEST(BackendCCEVFOpsTest, EmitsLoadWithUnifiedPointerType)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile_fp16 = MakeTile("tile_fp16", ir::DataType::FP16);
    auto fp16_reg = MakeVar("fp16_reg", ir::DataType::FP16);
    auto tile_i64 = MakeTile("tile_i64", ir::DataType::INT64);
    auto i64_reg = MakeVar("i64_reg", ir::DataType::INT64);
    auto tile_u8 = MakeTile("tile_u8", ir::DataType::UINT8);
    auto u8_reg = MakeVar("u8_reg", ir::DataType::UINT8);

    // FP16: vldas and vldus both use the same pointer (native half, no separate int ptr)
    auto fp16_out = Invoke(codegen, "vf.load", {fp16_reg, tile_fp16});
    ExpectContains(fp16_out, {"UnalignReg __ureg_ld_", "vldas(", "vldus(", "fp16_reg"});

    // INT64: vldas and vldus both use native int64_t pointer, no "* 2" multiplier
    ExpectInvoke(codegen, "vf.load", {"UnalignReg __ureg_ld_", "vldas(", "vldus("}, {i64_reg, tile_i64});
    auto i64_out = Invoke(codegen, "vf.load", {i64_reg, tile_i64, Int(4)});
    EXPECT_EQ(i64_out.find("* 2"), std::string::npos);

    // UINT8: b8 uses uint8_t pointer (tile is UINT8, matches natively)
    auto u8_out = Invoke(codegen, "vf.load", {u8_reg, tile_u8});
    ExpectContains(u8_out, {"UnalignReg __ureg_ld_", "vldas(", "vldus(", "u8_reg"});
}

TEST(BackendCCEVFOpsTest, EmitsMaskLogicOperations)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto mask0 = MakeVar("mask0", ir::DataType::UINT32);
    auto mask1 = MakeVar("mask1", ir::DataType::UINT32);
    auto mask2 = MakeVar("mask2", ir::DataType::UINT32);
    codegen.RegisterMaskRegVar("mask0");
    codegen.RegisterMaskRegVar("mask1");
    codegen.RegisterMaskRegVar("mask2");

    ExpectInvoke(codegen, "vf.and_", {"pand("}, {mask0, mask1, mask2, mask0});
    ExpectInvoke(codegen, "vf.or_", {"por("}, {mask0, mask1, mask2, mask0});
    ExpectInvoke(codegen, "vf.xor", {"pxor("}, {mask0, mask1, mask2, mask0});
    ExpectInvoke(codegen, "vf.not_", {"pnot("}, {mask0, mask1, mask2});
    ExpectInvoke(codegen, "vf.move", {"pmov("}, {mask0, mask1, mask2});
    ExpectInvoke(codegen, "vf.select", {"psel("}, {mask0, mask1, mask2, mask0});
    ExpectInvoke(codegen, "vf.pack", {"ppack(mask0, mask1, HIGHER)"}, {mask0, mask1},
                 {{"part", EnumValue(ir::PackPart::UPPER)}});
    ExpectInvoke(codegen, "vf.unpack", {"punpack(mask0, mask1, LOWER)"}, {mask0, mask1});
    ExpectInvoke(codegen, "vf.interleave", {"pintlv_b8("}, {mask0, mask1, mask1, mask2},
                 {{"dtype", ir::DataType::UINT8}});
    ExpectInvoke(codegen, "vf.de_interleave", {"pdintlv_b16("}, {mask0, mask1, mask1, mask2},
                 {{"dtype", ir::DataType::FP16}});
}

TEST(BackendCCEVFOpsTest, EmitsMaskMemoryAndSpecialRegisterOperations)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::UINT32);
    auto reg = MakeVar("reg", ir::DataType::UINT16);
    auto mask0 = MakeVar("mask0", ir::DataType::UINT32);
    auto mask1 = MakeVar("mask1", ir::DataType::UINT32);
    auto addr = MakeVar("addr", ir::DataType::INT64);
    auto ureg = MakeVar("ureg", ir::DataType::UINT32);
    codegen.RegisterMaskRegVar("mask0");
    codegen.RegisterMaskRegVar("mask1");
    codegen.RegisterAddrRegVar("addr");
    codegen.RegisterUnalignRegVar("ureg");

    ExpectInvoke(codegen, "vf.load_align", {"plds(mask0"}, {mask0, tile, Int(0)},
                 {{"dist", EnumValue(ir::LoadDist::DS)}});
    ExpectInvoke(codegen, "vf.load_align", {"pld(mask1"}, {mask1, tile, addr});
    ExpectInvoke(codegen, "vf.store_align", {"psts(mask0"}, {tile, mask0, mask1},
                 {{"dist", EnumValue(ir::StoreDist::NORM)}});
    ExpectInvoke(codegen, "vf.store_align", {"pst(mask0"}, {tile, mask0, addr});
    ExpectInvoke(codegen, "vf.store_unalign", {"pstu("}, {tile, mask0, ureg});
    ExpectInvoke(codegen, "vf.mask_gen_with_reg_tensor", {"movvp(generated_mask, (RegTensor<uint16_t> &)reg, 4)"},
                 {reg}, {{"offset", 4}}, "generated_mask");
    ExpectInvoke(codegen, "vf.get_mask_spr", {"movp_b16()"}, {}, {{"width", EnumValue(ir::MaskWidth::B16)}},
                 "spr_mask16");
    ExpectInvoke(codegen, "vf.get_mask_spr", {"movp_b32()"}, {}, {}, "spr_mask32");
}

TEST(BackendCCEVFOpsTest, BitCastEmitInlineReferenceCast)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto src = MakeVar("src", ir::DataType::FP8E4M3FN);

    // Create a vf.bit_cast Call with proper type so GetExprDtype returns FP32
    auto fp32_type = std::make_shared<const ir::ScalarType>(ir::DataType::FP32);
    auto bit_cast_call = std::make_shared<const ir::Call>(
        "vf.bit_cast", std::vector<ir::ExprPtr>{src},
        std::vector<std::pair<std::string, std::any>>{{"dtype", ir::DataType::FP32}}, fp32_type, ir::Span::Unknown());

    // GetExprAsCode on bit_cast should return the cast expression directly
    EXPECT_EQ(codegen.GetExprAsCode(bit_cast_call), "(RegTensor<float> &)src");

    // vf.xor with both args as bit_cast: vxor(dst, (RegTensor<float>&)src, ...)
    auto dst = MakeVar("dst", ir::DataType::FP32);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    ExpectInvoke(codegen, "vf.xor", {"vxor(", "(RegTensor<float> &)src", "(RegTensor<float> &)src"},
                 {dst, bit_cast_call, bit_cast_call, mask});

    // vf.xor with one bit_cast arg and one plain FP32 arg (src2 must match bit_cast's 32-bit width)
    auto src2 = MakeVar("src2", ir::DataType::FP32);
    ExpectInvoke(codegen, "vf.xor", {"vxor(", "(RegTensor<float> &)src", "src2"}, {dst, bit_cast_call, src2, mask});
}

TEST(BackendCCEVFOpsTest, EmitsB64LoadStoreAndNewCastPaths)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile64 = MakeTile("tile64", ir::DataType::INT64);
    auto tile64u = MakeTile("tile64u", ir::DataType::UINT64);
    auto i64 = MakeVar("i64", ir::DataType::INT64);
    auto u64 = MakeVar("u64", ir::DataType::UINT64);
    auto i32 = MakeVar("i32", ir::DataType::INT32);
    auto u32 = MakeVar("u32", ir::DataType::UINT32);
    auto u16 = MakeVar("u16", ir::DataType::UINT16);
    auto u8 = MakeVar("u8", ir::DataType::UINT8);
    auto i16 = MakeVar("i16", ir::DataType::INT16);
    auto i8 = MakeVar("i8", ir::DataType::INT8);
    auto fp16 = MakeVar("fp16", ir::DataType::FP16);
    auto fp32 = MakeVar("fp32", ir::DataType::FP32);
    auto bf16 = MakeVar("bf16", ir::DataType::BF16);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    codegen.RegisterMaskRegVar("mask");
    for (const auto& var : {i64, u64, i32, u32, u16, u8, i16, i8, fp16, fp32, bf16}) {
        codegen.RegisterRegTensorVar(var->name_);
    }

    // B64 load_align with post_update: b32 vlds simulation (RegTensor<uint32_t>&)
    ExpectInvoke(codegen, "vf.load_align", {"RegTensor<uint32_t>&", "POST_UPDATE"}, {i64, tile64, Int(2)},
                 {{"post_update", true}});

    // B64 store_align with post_update: b32 vsts simulation (ppack + pintlv_b32 + vsts)
    ExpectInvoke(codegen, "vf.store_align", {"RegTensor<uint32_t>&", "POST_UPDATE"}, {tile64, i64, mask, Int(2)},
                 {{"post_update", true}});

    // INT32→UINT8 (4x int narrowing): vcvt(dst, src, mask, RS, PART_PP, MODE) — 6 args
    ExpectInvoke(codegen, "vf.astype", {"vcvt(u8, i32, mask, RS_ENABLE, PART_P0, MODE_ZEROING);"}, {u8, i32, mask},
                 {{"layout", EnumValue(ir::CastLayout::ZERO)}, {"dtype", ir::DataType::UINT8}});

    // UINT32→UINT8 (4x int narrowing)
    ExpectInvoke(codegen, "vf.astype", {"vcvt(u8, u32, mask, RS_ENABLE, PART_P0, MODE_ZEROING);"}, {u8, u32, mask},
                 {{"layout", EnumValue(ir::CastLayout::ZERO)}, {"dtype", ir::DataType::UINT8}});

    // INT8→INT32 (4x int widening): vcvt(dst, src, mask, PART_PP, MODE_ZEROING) — 5 args
    ExpectInvoke(codegen, "vf.astype", {"vcvt(i32, i8, mask, PART_P0, MODE_ZEROING);"}, {i32, i8, mask},
                 {{"layout", EnumValue(ir::CastLayout::ZERO)}, {"dtype", ir::DataType::INT32}});

    // FP16→INT8 (float_to_narrower_int): vcvt(dst, src, mask, ROUND, SAT, PART, MODE) — 7 args
    ExpectInvoke(codegen, "vf.astype", {"vcvt(i8, fp16, mask, ROUND_F, RS_ENABLE, PART_EVEN, MODE_ZEROING);"},
                 {i8, fp16, mask},
                 {{"layout", EnumValue(ir::CastLayout::ZERO)},
                  {"round_mode", EnumValue(ir::VFRoundMode::CAST_FLOOR)},
                  {"saturate", EnumValue(ir::SaturateMode::ON)},
                  {"dtype", ir::DataType::INT8}});

    // BF16→INT8 (float_to_narrower_int)
    ExpectInvoke(codegen, "vf.astype", {"vcvt(i8, bf16, mask, ROUND_F, RS_ENABLE, PART_EVEN, MODE_ZEROING);"},
                 {i8, bf16, mask},
                 {{"layout", EnumValue(ir::CastLayout::ZERO)},
                  {"round_mode", EnumValue(ir::VFRoundMode::CAST_FLOOR)},
                  {"saturate", EnumValue(ir::SaturateMode::ON)},
                  {"dtype", ir::DataType::INT8}});

    // INT16→UINT8 (2x int narrowing): vcvt(dst, src, mask, RS, PART, MODE) — 6 args
    ExpectInvoke(codegen, "vf.astype", {"vcvt(u8, i16, mask, RS_DISABLE, PART_EVEN, MODE_ZEROING);"}, {u8, i16, mask},
                 {{"layout", EnumValue(ir::CastLayout::ZERO)}, {"dtype", ir::DataType::UINT8}});

    // INT32→INT16 (2x int narrowing)
    ExpectInvoke(codegen, "vf.astype", {"vcvt(i16, i32, mask, RS_DISABLE, PART_EVEN, MODE_ZEROING);"}, {i16, i32, mask},
                 {{"layout", EnumValue(ir::CastLayout::ZERO)}, {"dtype", ir::DataType::INT16}});

    // INT64→INT32 (2x int narrowing)
    ExpectInvoke(codegen, "vf.astype", {"vcvt(i32, i64, mask, RS_DISABLE, PART_EVEN, MODE_ZEROING);"}, {i32, i64, mask},
                 {{"layout", EnumValue(ir::CastLayout::ZERO)}, {"dtype", ir::DataType::INT32}});

    // INT16→INT4 (s16_to_s4 two-step)
    auto s4 = MakeVar("s4", ir::DataType::INT4);
    codegen.RegisterRegTensorVar("s4");
    ExpectInvoke(codegen, "vf.astype",
                 {"vcvt(s4_f16_tmp, i16, mask, ROUND_R, MODE_ZEROING);",
                  "vcvt_f162s4(s4, s4_f16_tmp, mask, ROUND_R, RS_DISABLE, PART_P0, MODE_ZEROING);"},
                 {s4, i16, mask}, {{"layout", EnumValue(ir::CastLayout::ZERO)}, {"dtype", ir::DataType::INT4}});

    // UINT32→INT8 (int_int_two_step: u32→s32 reinterpret → s32→f32→f16→s8)
    ExpectInvoke(codegen, "vf.astype",
                 {"vcvt(i8_f32_tmp, (RegTensor<int32_t> &)u32, mask, ROUND_R, MODE_ZEROING);",
                  "vcvt(i8_f16_tmp, i8_f32_tmp, mask, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);",
                  "vcvt(i8, i8_f16_tmp, mask, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);"},
                 {i8, u32, mask}, {{"dtype", ir::DataType::INT8}});

    // UINT16→INT8 (int_int_two_step: u16→u32→f32→f16→s8, 4-step)
    ExpectInvoke(codegen, "vf.astype",
                 {"vcvt(i8_u32_tmp, u16, mask, PART_EVEN, MODE_ZEROING);",
                  "vcvt(i8_f32_tmp, (RegTensor<int32_t> &)i8_u32_tmp, mask, ROUND_R, MODE_ZEROING);",
                  "vcvt(i8_f16_tmp, i8_f32_tmp, mask, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);",
                  "vcvt(i8, i8_f16_tmp, mask, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);"},
                 {i8, u16, mask}, {{"dtype", ir::DataType::INT8}});

    // UINT16→INT16 (int_int_two_step: same-width cross-sign reinterpret cast)
    ExpectInvoke(codegen, "vf.astype", {"i16 = (RegTensor<int16_t> &)u16;"}, {i16, u16, mask},
                 {{"dtype", ir::DataType::INT16}});

    // UINT32→INT32 (int_int_two_step: same-width cross-sign reinterpret cast)
    ExpectInvoke(codegen, "vf.astype", {"i32 = (RegTensor<int32_t> &)u32;"}, {i32, u32, mask},
                 {{"dtype", ir::DataType::INT32}});

    // FP16→INT8 with saturate=OFF (float_to_narrower_int, no saturation)
    ExpectInvoke(codegen, "vf.astype", {"vcvt(i8, fp16, mask, ROUND_F, RS_DISABLE, PART_EVEN, MODE_ZEROING);"},
                 {i8, fp16, mask},
                 {{"layout", EnumValue(ir::CastLayout::ZERO)},
                  {"round_mode", EnumValue(ir::VFRoundMode::CAST_FLOOR)},
                  {"saturate", EnumValue(ir::SaturateMode::OFF)},
                  {"dtype", ir::DataType::INT8}});

    // INT32→UINT8 (4x int narrowing) without explicit saturate → implicit RS_ENABLE
    ExpectInvoke(codegen, "vf.astype", {"vcvt(u8, i32, mask, RS_ENABLE, PART_P1, MODE_ZEROING);"}, {u8, i32, mask},
                 {{"layout", EnumValue(ir::CastLayout::ONE)}, {"dtype", ir::DataType::UINT8}});

    // INT8→INT32 (4x int widening) with layout TWO
    ExpectInvoke(codegen, "vf.astype", {"vcvt(i32, i8, mask, PART_P2, MODE_ZEROING);"}, {i32, i8, mask},
                 {{"layout", EnumValue(ir::CastLayout::TWO)}, {"dtype", ir::DataType::INT32}});

    // INT8→INT32 (4x int widening) with layout THREE
    ExpectInvoke(codegen, "vf.astype", {"vcvt(i32, i8, mask, PART_P3, MODE_ZEROING);"}, {i32, i8, mask},
                 {{"layout", EnumValue(ir::CastLayout::THREE)}, {"dtype", ir::DataType::INT32}});

    // FP32→INT16 (float narrowing): vcvt(dst, src, mask, ROUND, SAT, PART, MODE) — 7 args
    ExpectInvoke(codegen, "vf.astype", {"vcvt(i16, fp32, mask, ROUND_F, RS_ENABLE, PART_EVEN, MODE_ZEROING);"},
                 {i16, fp32, mask},
                 {{"layout", EnumValue(ir::CastLayout::ZERO)},
                  {"round_mode", EnumValue(ir::VFRoundMode::CAST_FLOOR)},
                  {"saturate", EnumValue(ir::SaturateMode::ON)},
                  {"dtype", ir::DataType::INT16}});

    // FP32→INT32 (float_to_same_int with saturate=OFF): vcvt(dst, src, mask, ROUND, RS, MODE) — 5 args
    ExpectInvoke(codegen, "vf.astype", {"vcvt(i32, fp32, mask, ROUND_F, RS_DISABLE, MODE_ZEROING);"}, {i32, fp32, mask},
                 {{"layout", EnumValue(ir::CastLayout::ZERO)},
                  {"round_mode", EnumValue(ir::VFRoundMode::CAST_FLOOR)},
                  {"saturate", EnumValue(ir::SaturateMode::OFF)},
                  {"dtype", ir::DataType::INT32}});

    // INT32→UINT8 (4x int narrowing) with explicit saturate=OFF → should reject
    EXPECT_ANY_THROW(Invoke(codegen, "vf.astype", {u8, i32, mask},
                            {{"layout", EnumValue(ir::CastLayout::ZERO)},
                             {"saturate", EnumValue(ir::SaturateMode::OFF)},
                             {"dtype", ir::DataType::UINT8}}));

    // INT16→INT8 (int_int_two_step src_to_f16_ok: s16→f16→s8)
    ExpectInvoke(codegen, "vf.astype",
                 {"vcvt(i8_f16_tmp, i16, mask, ROUND_R, MODE_ZEROING);",
                  "vcvt(i8, i8_f16_tmp, mask, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);"},
                 {i8, i16, mask}, {{"dtype", ir::DataType::INT8}});

    // INT32→INT8 with saturate=ON (int_int_two_step src_is_b32 && dst_is_b8: s32→f32→f16→s8)
    ExpectInvoke(codegen, "vf.astype",
                 {"vcvt(i8_f32_tmp, i32, mask, ROUND_R, MODE_ZEROING);",
                  "vcvt(i8_f16_tmp, i8_f32_tmp, mask, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);",
                  "vcvt(i8, i8_f16_tmp, mask, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);"},
                 {i8, i32, mask},
                 {{"layout", EnumValue(ir::CastLayout::ZERO)},
                  {"round_mode", EnumValue(ir::VFRoundMode::CAST_RINT)},
                  {"saturate", EnumValue(ir::SaturateMode::ON)},
                  {"dtype", ir::DataType::INT8}});
}

// ============================================================================
// ============================================================================
// B64 arithmetic/shift/arange emulation sequences (INT64/UINT64 lowering).
// ============================================================================

TEST(BackendCCEVFOpsTest, EmitsB64ArithmeticShiftAndArangeSequences)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto dst64 = MakeVar("dst64", ir::DataType::INT64);
    auto src64a = MakeVar("src64a", ir::DataType::INT64);
    auto src64b = MakeVar("src64b", ir::DataType::INT64);
    auto dstu64 = MakeVar("dstu64", ir::DataType::UINT64);
    auto srcu64 = MakeVar("srcu64", ir::DataType::UINT64);
    auto i64 = MakeVar("i64", ir::DataType::INT64);
    auto fp32 = MakeVar("fp32", ir::DataType::FP32);
    auto mask64 = MakeVar("mask64", ir::DataType::UINT64);

    const Kwargs zeroing = {{"mode", EnumValue(ir::MergeMode::ZEROING)}};

    // Add: ppack mask + deinterleave both sources + vaddc/vaddcs carry chain.
    ExpectInvoke(codegen, "vf.add",
                 {"ppack(dst64_add__pm_, mask64, LOWER)", "vdintlv(dst64_add_s0_lo_", "vaddc(dst64_add__carry_",
                  "vaddcs(dst64_add__carry_", "vintlv((RegTensor<uint32_t>&)dst64"},
                 {dst64, src64a, src64b, mask64}, zeroing);
    // Sub: borrow chain via vsubc/vsubcs.
    ExpectInvoke(codegen, "vf.sub", {"vsubc(dst64_sub__", "vsubcs(dst64_sub__", "vintlv((RegTensor<uint32_t>&)dst64"},
                 {dst64, src64a, src64b, mask64}, zeroing);
    // Mul: vmull (32x32->64) + two vmula cross-terms (int32 view for INT64).
    ExpectInvoke(codegen, "vf.mul",
                 {"vmull((RegTensor<uint32_t>&)dst64_mul__lod_", "vmula((RegTensor<int32_t>&)dst64_mul__hid_",
                  "vintlv((RegTensor<uint32_t>&)dst64"},
                 {dst64, src64a, src64b, mask64}, zeroing);
    // Max/Min: per-half compare + vsel + interleave.
    ExpectInvoke(codegen, "vf.max",
                 {"vdintlv(dst64_max_s0_lo_", "vcmp_gt(", "vsel(", "vintlv((RegTensor<uint32_t>&)dst64"},
                 {dst64, src64a, src64b, mask64}, zeroing);
    ExpectInvoke(codegen, "vf.min", {"vdintlv(dst64_min_s0_lo_", "vcmp", "vsel(", "vintlv((RegTensor<uint32_t>&)dst64"},
                 {dst64, src64a, src64b, mask64}, zeroing);
    // Neg: ppack mask + borrow chain (0 - src) + interleave.
    ExpectInvoke(codegen, "vf.neg",
                 {"ppack(dst64_neg__pm_, mask64, LOWER)", "vsubc(", "vsubcs(", "vintlv((RegTensor<uint32_t>&)dst64"},
                 {dst64, src64a, mask64}, zeroing);
    // Abs (INT64): sign compare on the hi half + neg borrow chain + vsel.
    ExpectInvoke(codegen, "vf.abs", {"vdintlv(dst64_abs_s_lo_", "vcmp_lt(", "vsubc(", "vsubcs(", "vsel("},
                 {dst64, src64a, mask64}, zeroing);
    // Adds: 64-bit scalar broadcast to b32 halves + carry chain.
    ExpectInvoke(
        codegen, "vf.adds",
        {"vdup(dst64_adds__losc_, (int32_t)((int64_t)(5))", "vaddc(", "vaddcs(", "vintlv((RegTensor<uint32_t>&)dst64"},
        {dst64, src64a, Int(5), mask64}, zeroing);
    // Muls: scalar broadcast + vmull/vmula cross-terms.
    ExpectInvoke(codegen, "vf.muls", {"ppack(dst64_muls__pm_, mask64, LOWER)", "vmull(", "vmula("},
                 {dst64, src64a, Int(5), mask64}, zeroing);
    // Maxs: scalar broadcast + per-half compare + vsel.
    ExpectInvoke(codegen, "vf.maxs", {"vdup(dst64_maxs__losc_", "vcmp", "vsel(", "vintlv((RegTensor<uint32_t>&)dst64"},
                 {dst64, src64a, Int(5), mask64}, zeroing);

    // Shift right (INT64): the arithmetic hi-half shift must use int32 views on
    // BOTH dst and src — the vshrs overloads require matching signedness.
    ExpectInvoke(codegen, "vf.shift_right",
                 {"vdintlv(dst64_shift_s_lo_", "vshrs((RegTensor<int32_t>&)dst64_shift__hid_", "vshls(",
                  "vintlv((RegTensor<uint32_t>&)dst64"},
                 {dst64, src64a, Int(2), mask64}, zeroing);
    // Shift left (INT64): logical u32 halves + vor cross-half carry.
    ExpectInvoke(codegen, "vf.shift_left", {"vshls(", "vshrs(", "vor(", "vintlv((RegTensor<uint32_t>&)dst64"},
                 {dst64, src64a, Int(2), mask64}, zeroing);
    // Shift right (UINT64): logical u32 hi-half shift (no int32 cast).
    ExpectInvoke(codegen, "vf.shift_right", {"vshrs(dstu64_shift__hid_", "vintlv((RegTensor<uint32_t>&)dstu64"},
                 {dstu64, srcu64, Int(2), mask64}, zeroing);

    // Cast: FP32->INT64 cross-width uses the 7-arg vcvt (ROUND, RS, PART, MODE).
    ExpectInvoke(codegen, "vf.astype", {"vcvt(i64, fp32, mask64, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);"},
                 {i64, fp32, mask64}, {{"dtype", ir::DataType::INT64}});
    // Cast: INT64->FP32 keeps the 6-arg (ROUND, PART, MODE) form.
    ExpectInvoke(codegen, "vf.astype", {"vcvt(fp32, i64, mask64, ROUND_R, PART_EVEN, MODE_ZEROING);"},
                 {fp32, i64, mask64}, {{"dtype", ir::DataType::FP32}});

    // Arange (INT64): vci index + vdup hi + 64-bit start fold via vaddc/vaddcs
    // carry chain + vintlv.
    ExpectInvoke(codegen, "vf.arange",
                 {"vci(dst64_b64_lo_", "vdup(dst64_b64_hi_, 0", "vdup(dst64_b64_sclo_, (int32_t)((int64_t)(5))",
                  "vaddc(dst64_b64_carry_", "vaddcs(dst64_b64_carry_", "vintlv((RegTensor<uint32_t> &)dst64"},
                 {dst64, Int(5)}, {{"dtype", ir::DataType::INT64}});
    // Arange (INT64, DECREASE_ORDER): vneg the index, same carry chain.
    ExpectInvoke(codegen, "vf.arange", {"vneg(dst64_b64_lo_", "vaddcs(dst64_b64_carry_"}, {dst64, Int(5)},
                 {{"index_order", EnumValue(ir::IndexOrder::DECREASE_ORDER)}, {"dtype", ir::DataType::INT64}});
}

// ============================================================================
// B64 div / reduce max/min / relu emulation sequences (INT64/UINT64 lowering).
// ============================================================================

TEST(BackendCCEVFOpsTest, EmitsB64DivReduceAndReluSequences)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto dst64 = MakeVar("dst64", ir::DataType::INT64);
    auto src64a = MakeVar("src64a", ir::DataType::INT64);
    auto src64b = MakeVar("src64b", ir::DataType::INT64);
    auto dstu64 = MakeVar("dstu64", ir::DataType::UINT64);
    auto srcu64a = MakeVar("srcu64a", ir::DataType::UINT64);
    auto srcu64b = MakeVar("srcu64b", ir::DataType::UINT64);
    auto mask64 = MakeVar("mask64", ir::DataType::UINT64);
    auto mask = MakeVar("mask", ir::DataType::UINT32);

    const Kwargs zeroing = {{"mode", EnumValue(ir::MergeMode::ZEROING)}};

    // Div (INT64): ppack mask + deinterleave + signed abs (vsubc/vsubcs borrow
    // chain) + f32 reciprocal + B128Calc refinement rounds + sign restore.
    // The reciprocal vintlv/vdintlv must use the s32 element view on ALL four
    // arguments (mixing s32 dsts with u32 srcs matches no bisheng overload).
    // The s64->f32 vcvt lands results on even f32 lanes only — the compacting
    // vdintlv before the vdiv is mandatory (mirrors Int64ToFloat).
    ExpectInvoke(
        codegen, "vf.div",
        {"ppack(dst64_div__pm_, mask64, LOWER)", "vdintlv((RegTensor<uint32_t>&)dst64_div_s0_lo", "vsubc(dst64_div_ac1",
         "vmull((RegTensor<uint32_t>&)dst64_div_r1m0l", "vintlv((RegTensor<int32_t>&)dst64_div_prl",
         "vcvt(dst64_div_t2f, dst64_div_prl", "vcvt(dst64_div_t2o, dst64_div_prh",
         "vdintlv((RegTensor<float>&)dst64_div_t2c", "vdiv(dst64_div_t3f, dst64_div_t3f, dst64_div_t2c",
         "vdintlv((RegTensor<int32_t>&)dst64_div_t5_lo", "vintlv((RegTensor<uint32_t>&)dst64"},
        {dst64, src64a, src64b, mask64}, zeroing);
    // Div (UINT64): abs is a vmov copy, plus the large-divisor (u1m) bypass.
    ExpectInvoke(codegen, "vf.div",
                 {"vmov((RegTensor<int32_t>&)dstu64_div_abs0_lo", "vcmp_lt(dstu64_div_u1m,", "vcmp_ge(dstu64_div_nlm,",
                  "vintlv((RegTensor<uint32_t>&)dstu64"},
                 {dstu64, srcu64a, srcu64b, mask64}, zeroing);

    // Reduce max/min (b64): per-half vcmax/vcmin + 5-arg register broadcast
    // (mask + POS + MODE; the 4-arg form only accepts a scalar) + eq-mask
    // second pass (no native b64 reduce). The hi-half reduce compares SIGNED
    // for INT64 (negative int64 hi halves are huge when viewed as u32).
    ExpectInvoke(codegen, "vf.reduce_max",
                 {"ppack(dst64_b64rm_pm, mask, LOWER)", "vcmax((RegTensor<int32_t>&)dst64_b64rm_hir_",
                  "vdup(dst64_b64rm_bc_, dst64_b64rm_hir_, dst64_b64rm_pm, POS_LOWEST, MODE_ZEROING);",
                  "vcmp_eq(dst64_b64rm_eqm_", "vcmax(dst64_b64rm_lor_", "vintlv((RegTensor<uint32_t>&)dst64"},
                 {dst64, src64a, mask});
    ExpectInvoke(codegen, "vf.reduce_min",
                 {"vcmin((RegTensor<int32_t>&)dst64_b64rm_hir_", "vcmp_eq(dst64_b64rm_eqm_", "vcmin(dst64_b64rm_lor_"},
                 {dst64, src64a, mask});

    // Relu (INT64): ppack mask + hi/lo two-stage compare vs scalar(0) + vsel +
    // interleave + b64 zeroing (no native b64 vrelu).
    ExpectInvoke(codegen, "vf.relu",
                 {"ppack(dst64_relu__pm_, mask64, LOWER)", "vdup(dst64_relu__losc_, (int32_t)0",
                  "vcmp_gt(dst64_relu__higt_", "vcmp_eq(dst64_relu__hieq_", "vsel(dst64_relu__lod_",
                  "ppack(dst64_relu__zero_pm_", "vintlv((RegTensor<uint32_t>&)dst64"},
                 {dst64, src64a, mask64}, zeroing);
}

// ============================================================================
// B64 select / compare / interleave / abs_sub / full / move emulation
// sequences (INT64 lowering).
// ============================================================================

TEST(BackendCCEVFOpsTest, EmitsB64SelectCompareAndMiscSequences)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto dst64 = MakeVar("dst64", ir::DataType::INT64);
    auto dst64b = MakeVar("dst64b", ir::DataType::INT64);
    auto src64a = MakeVar("src64a", ir::DataType::INT64);
    auto src64b = MakeVar("src64b", ir::DataType::INT64);
    auto cmp_dst = MakeVar("cmp_dst", ir::DataType::INT64);
    auto mask64 = MakeVar("mask64", ir::DataType::UINT64);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    codegen.RegisterRegTensorVar("dst64");
    codegen.RegisterRegTensorVar("dst64b");
    codegen.RegisterRegTensorVar("src64a");
    codegen.RegisterRegTensorVar("src64b");
    codegen.RegisterMaskRegVar("cmp_dst");

    // Select (b64): deinterleave both sources + ppack mask + vsel per half +
    // interleave back (no native b64 vsel over b64 pairs).
    ExpectInvoke(codegen, "vf.select",
                 {"ppack(dst64_sel__pm_, mask64, LOWER)", "vdintlv(dst64_sel_t_lo_", "vdintlv(dst64_sel_f_lo_",
                  "vsel(dst64_sel__lod_", "vintlv((RegTensor<uint32_t>&)dst64"},
                 {dst64, src64a, src64b, mask64});

    // Compare (b64, vector EQ): ppack mask + per-half vcmp_eq (signed int32
    // views on the SOURCES for INT64) + pand + punpack (the b64 result mask is
    // expanded back to 2-bit-per-element granularity).
    ExpectInvoke(codegen, "vf.eq",
                 {"ppack(cmp_dst_cmpv__pm_, mask, LOWER)", "vdintlv(cmp_dst_cmpv_s0_lo_",
                  "vcmp_eq(cmp_dst_cmpv__hir_, (RegTensor<int32_t>&)cmp_dst_cmpv_s0_hi_",
                  "pand(cmp_dst, cmp_dst_cmpv__hieq_, cmp_dst_cmpv__lor_", "punpack(cmp_dst, cmp_dst, LOWER)"},
                 {cmp_dst, src64a, src64b, mask});

    // Compare (b64, scalar GT): scalar broadcast to b32 halves + SIGNED hi-half
    // compare (int32 views on the SOURCES for INT64) + psel tie-break via hi_eq
    // (mirrors CompareScalarImpl b64 path).
    ExpectInvoke(codegen, "vf.gt",
                 {"ppack(cmp_dst_cmps__pm_, mask, LOWER)", "vdup(cmp_dst_cmps__losc_, (int32_t)((int64_t)(5))",
                  "vcmp_gt(cmp_dst_cmps__hir_, (RegTensor<int32_t>&)cmp_dst_cmps_s_hi_",
                  "psel(cmp_dst, cmp_dst_cmps__lor_", "punpack(cmp_dst, cmp_dst, LOWER)"},
                 {cmp_dst, src64a, Int(5), mask});

    // Interleave/de_interleave (b64): all operands reinterpreted as u32.
    ExpectInvoke(codegen, "vf.interleave", {"vintlv((RegTensor<uint32_t>&)dst64, (RegTensor<uint32_t>&)dst64b"},
                 {dst64, dst64b, src64a, src64b});
    ExpectInvoke(codegen, "vf.de_interleave", {"vdintlv((RegTensor<uint32_t>&)dst64, (RegTensor<uint32_t>&)dst64b"},
                 {dst64, dst64b, src64a, src64b});

    // AbsSub (b64): borrow-chain sub + signed sign-check + negate + vsel +
    // interleave + b64 zeroing (no native b64 vabsdif).
    ExpectInvoke(
        codegen, "vf.abs_sub",
        {"vdintlv(dst64_abssub_s0_lo_", "vsubc(dst64_abssub__borrow_, dst64_abssub__dl_", "vcmp_lt(dst64_abssub__sign_",
         "vsel(dst64_abssub__lod_", "ppack(dst64_abssub__zero_pm_", "vintlv((RegTensor<uint32_t>&)dst64"},
        {dst64, src64a, src64b, mask64});

    // Full (b64): scalar broadcast duplicates the value into two b32 halves
    // then interleaves (masked: vdup halves; unmasked: vbr halves).
    ExpectInvoke(codegen, "vf.full",
                 {"ppack(dst64_dup__pm_, mask, LOWER)", "vdup(dst64_dup__lo_, (uint32_t)((int64_t)(5))",
                  "vintlv((RegTensor<uint32_t>&)dst64"},
                 {dst64, Int(5), mask});
    ExpectInvoke(codegen, "vf.full", {"vbr(dst64_br__lo_, (uint32_t)((int64_t)(5))"}, {dst64, Int(5)});

    // Move (b64): the b64 mask is expanded via pintlv_b32 so both b32 halves
    // share the element's mask bit, then vmov runs on the u32 view.
    ExpectInvoke(
        codegen, "vf.move",
        {"ppack(dst64_mov_tm_, mask64, LOWER)", "pintlv_b32(dst64_mov_m0_", "vmov((RegTensor<uint32_t>&)dst64"},
        {dst64, src64a, mask64});
}

// Tests for load_align/store_align validation CHECKs
// ============================================================================

TEST(BackendCCEVFOpsTest, LoadAlignDataBlockRequiresMaskRegNotOffset)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::FP16);
    auto fp16 = MakeVar("fp16", ir::DataType::FP16);
    codegen.RegisterRegTensorVar("fp16");

    // DataBlock mode with integer offset instead of mask → should reject
    EXPECT_ANY_THROW(Invoke(codegen, "vf.load_align", {fp16, tile, Int(0)},
                            {{"data_copy_mode", EnumValue(ir::DataCopyMode::DATA_BLOCK_COPY)}}));
}

TEST(BackendCCEVFOpsTest, LoadAlignDataBlockRejectsNonMaskVar)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::FP16);
    auto fp16 = MakeVar("fp16", ir::DataType::FP16);
    codegen.RegisterRegTensorVar("fp16");

    // DataBlock mode with a RegTensor (not MaskReg) as args[2] → should reject
    EXPECT_ANY_THROW(Invoke(codegen, "vf.load_align", {fp16, tile, fp16},
                            {{"data_copy_mode", EnumValue(ir::DataCopyMode::DATA_BLOCK_COPY)}}));
}

TEST(BackendCCEVFOpsTest, LoadAlignDataBlockRejectsMaskRegDst)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::UINT32);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    codegen.RegisterMaskRegVar("mask");

    // DataBlock mode with MaskReg dst → should reject
    EXPECT_ANY_THROW(Invoke(codegen, "vf.load_align", {mask, tile, mask},
                            {{"data_copy_mode", EnumValue(ir::DataCopyMode::DATA_BLOCK_COPY)}}));
}

TEST(BackendCCEVFOpsTest, LoadAlign3ArgRejectsDintlvDist)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::FP16);
    auto fp16 = MakeVar("fp16", ir::DataType::FP16);
    codegen.RegisterRegTensorVar("fp16");

    // 3-arg form with DINTLV_B16 dist → should reject
    EXPECT_ANY_THROW(
        Invoke(codegen, "vf.load_align", {fp16, tile, Int(0)}, {{"dist", EnumValue(ir::LoadDist::DINTLV_B16)}}));
}

TEST(BackendCCEVFOpsTest, LoadAlign4ArgRejectsNonDintlvDist)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::FP16);
    auto fp16 = MakeVar("fp16", ir::DataType::FP16);
    auto fp16b = MakeVar("fp16b", ir::DataType::FP16);

    // 4-arg form with NORM dist → should reject
    EXPECT_ANY_THROW(
        Invoke(codegen, "vf.load_align", {fp16, fp16b, tile, Int(0)}, {{"dist", EnumValue(ir::LoadDist::NORM)}}));
}

TEST(BackendCCEVFOpsTest, LoadAlign4ArgRejectsMaskRegDst)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::UINT32);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    auto mask2 = MakeVar("mask2", ir::DataType::UINT32);
    codegen.RegisterMaskRegVar("mask");
    codegen.RegisterMaskRegVar("mask2");

    // 4-arg form with MaskReg dst → should reject
    EXPECT_ANY_THROW(
        Invoke(codegen, "vf.load_align", {mask, mask2, tile, Int(0)}, {{"dist", EnumValue(ir::LoadDist::DINTLV_B16)}}));
}

TEST(BackendCCEVFOpsTest, StoreAlignDataBlockRejectsNonMaskArg)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::FP16);
    auto fp16 = MakeVar("fp16", ir::DataType::FP16);
    codegen.RegisterRegTensorVar("fp16");

    // DataBlock mode with RegTensor (not MaskReg) as args[2] → should reject
    EXPECT_ANY_THROW(Invoke(codegen, "vf.store_align", {tile, fp16, fp16},
                            {{"data_copy_mode", EnumValue(ir::DataCopyMode::DATA_BLOCK_COPY)}}));
}

TEST(BackendCCEVFOpsTest, StoreAlignIntlvRejectsNonMaskArg3)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::FP16);
    auto fp16 = MakeVar("fp16", ir::DataType::FP16);
    auto fp16b = MakeVar("fp16b", ir::DataType::FP16);
    codegen.RegisterRegTensorVar("fp16");
    codegen.RegisterRegTensorVar("fp16b");

    // INTLV mode with RegTensor (not MaskReg) as args[3] → should reject
    EXPECT_ANY_THROW(
        Invoke(codegen, "vf.store_align", {tile, fp16, fp16b, fp16}, {{"dist", EnumValue(ir::StoreDist::INTLV)}}));
}

TEST(BackendCCEVFOpsTest, StoreAlign4ArgNonIntlvRejectsRegArg)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::FP16);
    auto fp16 = MakeVar("fp16", ir::DataType::FP16);
    auto fp16b = MakeVar("fp16b", ir::DataType::FP16);
    codegen.RegisterRegTensorVar("fp16");
    codegen.RegisterRegTensorVar("fp16b");

    // 4-arg non-INTLV, non-post_update with RegTensor as args[2] → should reject
    EXPECT_ANY_THROW(Invoke(codegen, "vf.store_align", {tile, fp16, fp16b, fp16}));
}

TEST(BackendCCEVFOpsTest, StoreAlignPostUpdateAccepts4Args)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::INT64);
    auto i64 = MakeVar("i64", ir::DataType::INT64);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    codegen.RegisterRegTensorVar("i64");
    codegen.RegisterMaskRegVar("mask");

    // 4-arg post_update path (dst, src, mask, stride) → should succeed
    ExpectInvoke(codegen, "vf.store_align", {"POST_UPDATE"}, {tile, i64, mask, Int(2)}, {{"post_update", true}});
}

TEST(BackendCCEVFOpsTest, StoreAlignExplicitNormDistExpandsToWidthQualified)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::FP32);
    auto fp32 = MakeVar("fp32", ir::DataType::FP32);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    codegen.RegisterRegTensorVar("fp32");
    codegen.RegisterMaskRegVar("mask");

    auto emitted = Invoke(codegen, "vf.store_align", {tile, fp32, mask}, {{"dist", EnumValue(ir::StoreDist::NORM)}});
    ExpectContains(emitted, {"vsts(", "NORM_B32"});
    EXPECT_EQ(emitted.find("NORM,"), std::string::npos)
        << "Should expand NORM to NORM_B32, not emit bare NORM: " << emitted;
}

TEST(BackendCCEVFOpsTest, StoreAlignExplicitWidthQualifiedDist)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::FP16);
    auto fp16 = MakeVar("fp16", ir::DataType::FP16);
    auto fp16b = MakeVar("fp16b", ir::DataType::FP16);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    codegen.RegisterRegTensorVar("fp16");
    codegen.RegisterRegTensorVar("fp16b");
    codegen.RegisterMaskRegVar("mask");

    // Width-qualified dists select the granularity explicitly and are not
    // overridden by the src dtype (fp16 src + explicit NORM_B8 stays NORM_B8).
    auto emitted = Invoke(codegen, "vf.store_align", {tile, fp16, mask}, {{"dist", EnumValue(ir::StoreDist::NORM_B8)}});
    ExpectContains(emitted, {"vsts(", "NORM_B8"});

    // FIRST_ELEMENT_Bx maps to the ONEPT_Bx CCE DistVST constant
    emitted = Invoke(codegen, "vf.store_align", {tile, fp16, mask},
                     {{"dist", EnumValue(ir::StoreDist::FIRST_ELEMENT_B16)}});
    ExpectContains(emitted, {"ONEPT_B16"});

    // PACK_Bx maps to the PK_Bx CCE DistVST constant
    emitted = Invoke(codegen, "vf.store_align", {tile, fp16, mask}, {{"dist", EnumValue(ir::StoreDist::PACK_B32)}});
    ExpectContains(emitted, {"PK_B32"});

    // PACK4_B32 maps to the PK4_B32 CCE DistVST constant
    emitted = Invoke(codegen, "vf.store_align", {tile, fp16, mask}, {{"dist", EnumValue(ir::StoreDist::PACK4_B32)}});
    ExpectContains(emitted, {"PK4_B32"});

    // INTLV_Bx dual-source with explicit granularity
    emitted = Invoke(codegen, "vf.store_align", {tile, fp16, fp16b, mask},
                     {{"dist", EnumValue(ir::StoreDist::INTLV_B8)}});
    ExpectContains(emitted, {"vsts(", "INTLV_B8"});
}

TEST(BackendCCEVFOpsTest, StoreAlignAddrRegHonorsDist)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::FP16);
    auto tile64 = MakeTile("tile64", ir::DataType::INT64);
    auto fp16 = MakeVar("fp16", ir::DataType::FP16);
    auto i64 = MakeVar("i64", ir::DataType::INT64);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    auto addr = MakeVar("addr", ir::DataType::INT64);
    codegen.RegisterAddrRegVar("addr");
    codegen.RegisterRegTensorVar("fp16");
    codegen.RegisterRegTensorVar("i64");
    codegen.RegisterMaskRegVar("mask");

    // Default (no dist): auto-select NORM_B16 by src dtype
    ExpectInvoke(codegen, "vf.store_align", {"vst(fp16", "NORM_B16"}, {tile, fp16, mask, addr});
    // Explicit coarse dist expands by dtype (previously ignored in this path)
    auto emitted = Invoke(codegen, "vf.store_align", {tile, fp16, mask, addr},
                          {{"dist", EnumValue(ir::StoreDist::FIRST_ELEMENT)}});
    ExpectContains(emitted, {"vst(", "ONEPT_B16"});
    // Explicit width-qualified dist passes through (previously ignored)
    emitted = Invoke(codegen, "vf.store_align", {tile, fp16, mask, addr},
                     {{"dist", EnumValue(ir::StoreDist::PACK_B32)}});
    ExpectContains(emitted, {"vst(", "PK_B32"});
    // INTLV dist requires two source registers → reject with single-source AddrReg
    EXPECT_ANY_THROW(
        Invoke(codegen, "vf.store_align", {tile, fp16, mask, addr}, {{"dist", EnumValue(ir::StoreDist::INTLV)}}));
    // vst has no post mode → reject post_update with AddrReg
    EXPECT_ANY_THROW(Invoke(codegen, "vf.store_align", {tile, fp16, mask, addr}, {{"post_update", true}}));
    // b64: vst simulated via int32 reinterpret + ppack/pintlv_b32 mask expansion
    emitted = Invoke(codegen, "vf.store_align", {tile64, i64, mask, addr});
    ExpectContains(emitted, {"ppack(", "pintlv_b32(", "(RegTensor<int32_t>&)i64", "NORM_B32"});
}

TEST(BackendCCEVFOpsTest, LoadUnalignB64SimAndSqueezeDstWhitelist)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile64 = MakeTile("tile64", ir::DataType::INT64);
    auto fp16 = MakeVar("fp16", ir::DataType::FP16);
    auto bf16 = MakeVar("bf16", ir::DataType::BF16);
    auto i64 = MakeVar("i64", ir::DataType::INT64);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    auto ureg = MakeVar("ureg", ir::DataType::INT64);
    codegen.RegisterUnalignRegVar("ureg");
    codegen.RegisterRegTensorVar("fp16");
    codegen.RegisterRegTensorVar("bf16");
    codegen.RegisterRegTensorVar("i64");
    codegen.RegisterMaskRegVar("mask");

    // load_unalign b64 with stride: simulated as uint32_t with stride*2
    // (mirrors AscendC DataCopyUnAlignImpl)
    auto emitted = Invoke(codegen, "vf.load_unalign", {i64, ureg, tile64, Int(4)});
    ExpectContains(emitted, {"(RegTensor<uint32_t>&)i64", "(__ubuf__ uint32_t", "(4) * 2"});
    // squeeze: dst must be in the doc type list (same as src)
    EXPECT_ANY_THROW(Invoke(codegen, "vf.squeeze", {bf16, fp16, mask}));
}

TEST(BackendCCEVFOpsTest, LoadAlignDualDstWidthAndB64Dist)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::FP16);
    auto tile64 = MakeTile("tile64", ir::DataType::INT64);
    auto fp16 = MakeVar("fp16", ir::DataType::FP16);
    auto bf16 = MakeVar("bf16", ir::DataType::BF16);
    auto i64 = MakeVar("i64", ir::DataType::INT64);
    auto u32 = MakeVar("u32", ir::DataType::UINT32);
    codegen.RegisterRegTensorVar("fp16");
    codegen.RegisterRegTensorVar("bf16");
    codegen.RegisterRegTensorVar("i64");
    codegen.RegisterRegTensorVar("u32");

    // 4-arg (de-interleave) form: dst1 is loaded at dst0's element width, so a
    // dst0/dst1 bit width mismatch is rejected
    EXPECT_ANY_THROW(
        Invoke(codegen, "vf.load_align", {fp16, u32, tile, Int(0)}, {{"dist", EnumValue(ir::LoadDist::DINTLV_B16)}}));
    // equal-width reinterpreting views stay legal
    ExpectInvoke(codegen, "vf.load_align", {"vlds(", "DINTLV_B16"}, {fp16, bf16, tile, Int(0)},
                 {{"dist", EnumValue(ir::LoadDist::DINTLV_B16)}});
    // load_align b64: the resolved dist is passed through (not forced to NORM),
    // mirroring the AscendC DataCopyImpl b64 path
    ExpectInvoke(codegen, "vf.load_align", {"(RegTensor<uint32_t>&)i64", "BRC_B32"}, {i64, tile64, Int(0)},
                 {{"dist", EnumValue(ir::LoadDist::BRC)}});
}

TEST(BackendCCEVFOpsTest, RejectsDistOutsideOpEnum)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::FP16);
    auto fp16 = MakeVar("fp16", ir::DataType::FP16);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    codegen.RegisterRegTensorVar("fp16");
    codegen.RegisterMaskRegVar("mask");

    // The dist kwarg is type-erased to int at the Python/C++ boundary, so the
    // backend validates that the value is a valid enumerator of the op's own
    // dist enum: load-only values beyond the StoreDist range (DINTLV_B8 = 20)
    // and out-of-range garbage are rejected. In-range cross-enum values are
    // undetectable here (the enum type is lost at the int boundary).
    EXPECT_ANY_THROW(Invoke(codegen, "vf.store_align", {tile, fp16, mask}, {{"dist", 20}}));
    EXPECT_ANY_THROW(Invoke(codegen, "vf.load_align", {fp16, tile, Int(0)}, {{"dist", 30}}));
}

TEST(BackendCCEVFOpsTest, RejectsCreateMaskPatternOutsideEnum)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);

    // The pattern kwarg is type-erased to int: an out-of-enum value must not
    // silently fall back to PAT_ALL. (In-range values that alias another
    // enum's numeric value, e.g. MaskWidth.B16 == 1 == MaskPattern.ALLF, are
    // indistinguishable after type erasure and are treated as legal
    // MaskPattern members.)
    EXPECT_ANY_THROW(Invoke(codegen, "vf.create_mask", {}, {{"pattern", 99}}, "mask"));
}

TEST(BackendCCEVFOpsTest, RejectsAstypeUint64)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::FP32);
    auto u64 = MakeVar("u64", ir::DataType::UINT64);
    auto f32 = MakeVar("f32", ir::DataType::FP32);
    auto i64 = MakeVar("i64", ir::DataType::INT64);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    codegen.RegisterRegTensorVar("u64");
    codegen.RegisterRegTensorVar("f32");
    codegen.RegisterRegTensorVar("i64");
    codegen.RegisterMaskRegVar("mask");

    // vcvt has no uint64 overloads, and the AscendC Cast micro instruction
    // doesn't support uint64 either — reject early instead of failing late
    // in bisheng.
    EXPECT_ANY_THROW(Invoke(codegen, "vf.astype", {f32, u64, mask}, {{"dtype", ir::DataType::FP32}}));
    EXPECT_ANY_THROW(Invoke(codegen, "vf.astype", {u64, f32, mask}, {{"dtype", ir::DataType::UINT64}}));
    EXPECT_ANY_THROW(Invoke(codegen, "vf.astype", {i64, u64, mask}, {{"dtype", ir::DataType::INT64}}));
}

TEST(BackendCCEVFOpsTest, StoreAlignInt64NormalStoreEmitsB32Simulation)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::INT64);
    auto i64 = MakeVar("i64", ir::DataType::INT64);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    codegen.RegisterRegTensorVar("i64");
    codegen.RegisterMaskRegVar("mask");

    auto emitted = Invoke(codegen, "vf.store_align", {tile, i64, mask});
    ExpectContains(emitted, {"ppack(", "pintlv_b32(", "RegTensor<uint32_t>&", "NORM_B32", "vsts("});
    EXPECT_EQ(emitted.find("vector_2xvl_s64"), std::string::npos) << "Should not use __VF_VSTS_B64: " << emitted;
}

TEST(BackendCCEVFOpsTest, StoreAlignInt64PostUpdateEmitsB32Simulation)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::INT64);
    auto i64 = MakeVar("i64", ir::DataType::INT64);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    codegen.RegisterRegTensorVar("i64");
    codegen.RegisterMaskRegVar("mask");

    auto emitted = Invoke(codegen, "vf.store_align", {tile, i64, mask, Int(2)}, {{"post_update", true}});
    ExpectContains(emitted, {"ppack(", "pintlv_b32(", "RegTensor<uint32_t>&", "POST_UPDATE", "vsts("});
    EXPECT_EQ(emitted.find("vector_2xvl_s64"), std::string::npos) << "Should not use __VF_VSTS_B64: " << emitted;
}

TEST(BackendCCEVFOpsTest, StoreAlignRejectsTooFewArgs)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::FP16);
    auto fp16 = MakeVar("fp16", ir::DataType::FP16);

    // Only 2 args (dst_ptr, src_reg) without MaskReg src → should reject
    EXPECT_ANY_THROW(Invoke(codegen, "vf.store_align", {tile, fp16}));
}

// ============================================================================
// CoerceScalarToInt: float-constant scalar coerced to int literal for int src
// ============================================================================

TEST(BackendCCEVFOpsTest, CoercesFloatScalarToIntForInt32Src)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto i32_dst = MakeVar("i32_dst", ir::DataType::INT32);
    auto i32_src = MakeVar("i32_src", ir::DataType::INT32);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    codegen.RegisterRegTensorVar("i32_dst");
    codegen.RegisterRegTensorVar("i32_src");
    const Kwargs zeroing = {{"mode", EnumValue(ir::MergeMode::ZEROING)}};

    // Float 3.5 should be truncated to 3 in the emitted code, not "3.500000"
    ExpectInvoke(codegen, "vf.muls", {"vmuls(", ", 3, "}, {i32_dst, i32_src, Float(3.5), mask}, zeroing);
    ExpectInvoke(codegen, "vf.adds", {"vadds(", ", 3, "}, {i32_dst, i32_src, Float(3.5), mask}, zeroing);
    ExpectInvoke(codegen, "vf.mins", {"vmins(", ", 3, "}, {i32_dst, i32_src, Float(3.5), mask}, zeroing);
    ExpectInvoke(codegen, "vf.maxs", {"vmaxs(", ", 3, "}, {i32_dst, i32_src, Float(3.5), mask}, zeroing);
    // 0.9 should be truncated to 0
    ExpectInvoke(codegen, "vf.muls", {"vmuls(", ", 0, "}, {i32_dst, i32_src, Float(0.9), mask}, zeroing);
    // 1e10 should wrap to int32: static_cast<int32_t>(10000000000) = 1410065408
    ExpectInvoke(codegen, "vf.muls", {"vmuls(", ", 1410065408, "}, {i32_dst, i32_src, Float(1e10), mask}, zeroing);
}

TEST(BackendCCEVFOpsTest, CoercesFloatScalarForUintAndInt16)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto u16_dst = MakeVar("u16_dst", ir::DataType::UINT16);
    auto u16_src = MakeVar("u16_src", ir::DataType::UINT16);
    auto u32_dst = MakeVar("u32_dst", ir::DataType::UINT32);
    auto u32_src = MakeVar("u32_src", ir::DataType::UINT32);
    auto i16_dst = MakeVar("i16_dst", ir::DataType::INT16);
    auto i16_src = MakeVar("i16_src", ir::DataType::INT16);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    codegen.RegisterRegTensorVar("u16_dst");
    codegen.RegisterRegTensorVar("u16_src");
    codegen.RegisterRegTensorVar("u32_dst");
    codegen.RegisterRegTensorVar("u32_src");
    codegen.RegisterRegTensorVar("i16_dst");
    codegen.RegisterRegTensorVar("i16_src");
    const Kwargs zeroing = {{"mode", EnumValue(ir::MergeMode::ZEROING)}};

    // UINT16: 3.5→3, emitted with "u" suffix
    ExpectInvoke(codegen, "vf.muls", {"vmuls(", ", 3u, "}, {u16_dst, u16_src, Float(3.5), mask}, zeroing);
    // UINT32: 3.5→3u
    ExpectInvoke(codegen, "vf.muls", {"vmuls(", ", 3u, "}, {u32_dst, u32_src, Float(3.5), mask}, zeroing);
    // INT16: 3.5→3 (no suffix)
    ExpectInvoke(codegen, "vf.muls", {"vmuls(", ", 3, "}, {i16_dst, i16_src, Float(3.5), mask}, zeroing);
    // INT16: 700.0 wraps to static_cast<int16_t>(700) = 700
    ExpectInvoke(codegen, "vf.muls", {"vmuls(", ", 700, "}, {i16_dst, i16_src, Float(700.0), mask}, zeroing);
}

TEST(BackendCCEVFOpsTest, DoesNotCoerceForFloatSrc)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto fp32_dst = MakeVar("fp32_dst", ir::DataType::FP32);
    auto fp32_src = MakeVar("fp32_src", ir::DataType::FP32);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    codegen.RegisterRegTensorVar("fp32_dst");
    codegen.RegisterRegTensorVar("fp32_src");
    const Kwargs zeroing = {{"mode", EnumValue(ir::MergeMode::ZEROING)}};

    // FP32 src: float scalar should NOT be coerced — stays as float literal
    auto emitted = Invoke(codegen, "vf.muls", {fp32_dst, fp32_src, Float(3.5), mask}, zeroing);
    ExpectContains(emitted, {"vmuls("});
    // Should contain the float value, not truncated integer
    EXPECT_NE(emitted.find("3.5"), std::string::npos) << emitted;
}

TEST(BackendCCEVFOpsTest, MulsAcceptsIndexOrInt64ScalarConvertedToSrcType)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto i32_dst = MakeVar("i32_dst", ir::DataType::INT32);
    auto i32_src = MakeVar("i32_src", ir::DataType::INT32);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    codegen.RegisterRegTensorVar("i32_dst");
    codegen.RegisterRegTensorVar("i32_src");
    const Kwargs zeroing = {{"mode", EnumValue(ir::MergeMode::ZEROING)}};

    // Keep accepting the legacy INDEX scalar and the new INT64 default integer.
    ExpectInvoke(codegen, "vf.muls", {"vmuls("}, {i32_dst, i32_src, IndexVal(3), mask}, zeroing);
    ExpectInvoke(codegen, "vf.muls", {"vmuls("}, {i32_dst, i32_src, Int(3), mask}, zeroing);
}

TEST(BackendCCEVFOpsTest, MulsRejectsUnlistedIntTypes)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto i8_src = MakeVar("i8_src", ir::DataType::INT8);
    auto u8_src = MakeVar("u8_src", ir::DataType::UINT8);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    codegen.RegisterRegTensorVar("i8_src");
    codegen.RegisterRegTensorVar("u8_src");
    const Kwargs zeroing = {{"mode", EnumValue(ir::MergeMode::ZEROING)}};

    // INT8/UINT8 src not in the supported type list (b16/b32/b64 int + FP16/FP32)
    // → should reject. INT64/UINT64 are supported via the b64 emulation path.
    EXPECT_ANY_THROW(Invoke(codegen, "vf.muls", {i8_src, i8_src, Float(2.0), mask}, zeroing));
    EXPECT_ANY_THROW(Invoke(codegen, "vf.muls", {u8_src, u8_src, Float(2.0), mask}, zeroing));
}

TEST(BackendCCEVFOpsTest, CompareScalarCoercesFloatToInt)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto i32_dst = MakeVar("i32_dst", ir::DataType::UINT32);
    auto i32_src = MakeVar("i32_src", ir::DataType::INT32);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    codegen.RegisterMaskRegVar("i32_dst");
    codegen.RegisterRegTensorVar("i32_src");
    codegen.RegisterMaskRegVar("mask");

    // vcmps_eq with INT32 src + float 3.5 scalar → should emit "3" not "3.500000"
    ExpectInvoke(codegen, "vf.eq", {"vcmps_eq("}, {i32_dst, i32_src, Float(3.5), mask});
    auto emitted = Invoke(codegen, "vf.eq", {i32_dst, i32_src, Float(3.5), mask});
    EXPECT_NE(emitted.find(", 3,"), std::string::npos) << "Expected int 3 in: " << emitted;
    EXPECT_EQ(emitted.find("3.5"), std::string::npos) << "Should not have float literal: " << emitted;
}

TEST(BackendCCEVFOpsTest, StoreAlignAcceptsCompatibleIntDtypes)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile_i32 = MakeTile("tile_i32", ir::DataType::INT32);
    auto u32_reg = MakeVar("u32_reg", ir::DataType::UINT32);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    codegen.RegisterRegTensorVar("u32_reg");
    codegen.RegisterMaskRegVar("mask");

    // INT32 dst + UINT32 src: GetUBufPtr casts dst pointer to src type
    ExpectInvoke(codegen, "vf.store_align", {"vsts("}, {tile_i32, u32_reg, mask});
}

TEST(BackendCCEVFOpsTest, AxpyAcceptsInt64Scalar)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto i64_dst = MakeVar("i64_dst", ir::DataType::INT64);
    auto i64_src = MakeVar("i64_src", ir::DataType::INT64);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    codegen.RegisterRegTensorVar("i64_dst");
    codegen.RegisterRegTensorVar("i64_src");
    const Kwargs zeroing = {{"mode", EnumValue(ir::MergeMode::ZEROING)}};

    // Python int 2 → ConstInt(INT64) → allowed for axpy. B64 axpy lowers to
    // the Muls+Add emulation (no native b64 vaxpy): scalar broadcast +
    // MulB64 + AddB64 carry chain + interleave.
    ExpectInvoke(codegen, "vf.axpy",
                 {"ppack(i64_dst_axpy__pm_", "vdup(i64_dst_axpy__losc_, (int32_t)((int64_t)(2))",
                  "vmull((RegTensor<uint32_t>&)i64_dst_axpy__mul_lo_", "vaddc(i64_dst_axpy__carry_",
                  "vintlv((RegTensor<uint32_t>&)i64_dst"},
                 {i64_dst, i64_src, Int(2), mask}, zeroing);
}

TEST(BackendCCEVFOpsTest, AxpyCoercesFloatScalarToInt64)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto i64_dst = MakeVar("i64_dst", ir::DataType::INT64);
    auto i64_src = MakeVar("i64_src", ir::DataType::INT64);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    codegen.RegisterRegTensorVar("i64_dst");
    codegen.RegisterRegTensorVar("i64_src");
    const Kwargs zeroing = {{"mode", EnumValue(ir::MergeMode::ZEROING)}};

    // INT64 src + float 3.5 → coerced to int 3 in the b64 scalar broadcast
    auto emitted = Invoke(codegen, "vf.axpy", {i64_dst, i64_src, Float(3.5), mask}, zeroing);
    ExpectContains(emitted, {"vdup(i64_dst_axpy__losc_, (int32_t)((int64_t)(3))", "vaddc(i64_dst_axpy__carry_"});
    EXPECT_EQ(emitted.find("3.5"), std::string::npos) << "Scalar should be coerced to int: " << emitted;
}

TEST(BackendCCEVFOpsTest, StoreAlignAddrRegRejectsNonMaskArg2)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::FP16);
    auto fp16 = MakeVar("fp16", ir::DataType::FP16);
    auto fp16b = MakeVar("fp16b", ir::DataType::FP16);
    codegen.RegisterRegTensorVar("fp16");
    codegen.RegisterRegTensorVar("fp16b");

    // AddrReg path: 4 args with RegTensor (not MaskReg) as args[2] → reject
    EXPECT_ANY_THROW(Invoke(codegen, "vf.store_align", {tile, fp16, fp16b, fp16b}));
}

// ============================================================================
// ReduceSum: INT16/UINT16 32-bit accumulator cast, INT64/UINT64 rejection
// ============================================================================

TEST(BackendCCEVFOpsTest, ReduceSumInt16EmitsInt32AccumulatorCast)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto i16_dst = MakeVar("i16_dst", ir::DataType::INT16);
    auto i16_src = MakeVar("i16_src", ir::DataType::INT16);
    auto mask = MakeVar("mask", ir::DataType::UINT32);

    // INT16 src: vcadd requires 32-bit accumulator, dst cast to RegTensor<int32_t>&
    ExpectInvoke(codegen, "vf.reduce_sum", {"vcadd(", "RegTensor<int32_t>&"}, {i16_dst, i16_src, mask});
}

TEST(BackendCCEVFOpsTest, ReduceSumUint16EmitsUint32AccumulatorCast)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto u16_dst = MakeVar("u16_dst", ir::DataType::UINT16);
    auto u16_src = MakeVar("u16_src", ir::DataType::UINT16);
    auto mask = MakeVar("mask", ir::DataType::UINT32);

    // UINT16 src: dst cast to RegTensor<uint32_t>&
    ExpectInvoke(codegen, "vf.reduce_sum", {"vcadd(", "RegTensor<uint32_t>&"}, {u16_dst, u16_src, mask});
}

TEST(BackendCCEVFOpsTest, ReduceSumInt32NoCast)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto i32_dst = MakeVar("i32_dst", ir::DataType::INT32);
    auto i32_src = MakeVar("i32_src", ir::DataType::INT32);
    auto mask = MakeVar("mask", ir::DataType::UINT32);

    // INT32 src: same-type, no cast needed
    auto emitted = Invoke(codegen, "vf.reduce_sum", {i32_dst, i32_src, mask});
    ExpectContains(emitted, {"vcadd("});
    EXPECT_EQ(emitted.find("RegTensor<int32_t>&"), std::string::npos) << "INT32 should not need cast: " << emitted;
}

TEST(BackendCCEVFOpsTest, ReduceSumFp32NoCast)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto fp32_dst = MakeVar("fp32_dst", ir::DataType::FP32);
    auto fp32_src = MakeVar("fp32_src", ir::DataType::FP32);
    auto mask = MakeVar("mask", ir::DataType::UINT32);

    auto emitted = Invoke(codegen, "vf.reduce_sum", {fp32_dst, fp32_src, mask});
    ExpectContains(emitted, {"vcadd("});
    EXPECT_EQ(emitted.find("RegTensor<"), std::string::npos) << "FP32 should not need cast: " << emitted;
}

TEST(BackendCCEVFOpsTest, ReduceSumDatablockInt16NoCast)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto i16_dst = MakeVar("i16_dst", ir::DataType::INT16);
    auto i16_src = MakeVar("i16_src", ir::DataType::INT16);
    auto mask = MakeVar("mask", ir::DataType::UINT32);

    // Datablock mode uses vcgadd which supports INT16 same-type, no cast needed
    auto emitted = Invoke(codegen, "vf.reduce_sum", {i16_dst, i16_src, mask}, {{"datablock", true}});
    ExpectContains(emitted, {"vcgadd("});
    EXPECT_EQ(emitted.find("RegTensor<int32_t>&"), std::string::npos)
        << "Datablock INT16 should not need cast: " << emitted;
}

TEST(BackendCCEVFOpsTest, ReduceSumAcceptsInt64)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto i64_dst = MakeVar("i64_dst", ir::DataType::INT64);
    auto i64_src = MakeVar("i64_src", ir::DataType::INT64);
    auto mask = MakeVar("mask", ir::DataType::UINT32);

    // B64 sum lowers to the 16-bit carry decomposition (no native b64 vcadd).
    ExpectInvoke(codegen, "vf.reduce_sum",
                 {"ppack(i64_dst_b64rs_pm, mask, LOWER)", "vdup(i64_dst_b64rs_lowf_, (int32_t)0xFFFF",
                  "vcadd(i64_dst_b64rs_lowr_", "vintlv((RegTensor<uint32_t>&)i64_dst"},
                 {i64_dst, i64_src, mask});
}

TEST(BackendCCEVFOpsTest, ReduceSumAcceptsUint64)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto u64_dst = MakeVar("u64_dst", ir::DataType::UINT64);
    auto u64_src = MakeVar("u64_src", ir::DataType::UINT64);
    auto mask = MakeVar("mask", ir::DataType::UINT32);

    ExpectInvoke(
        codegen, "vf.reduce_sum",
        {"ppack(u64_dst_b64rs_pm, mask, LOWER)", "vcadd(u64_dst_b64rs_lowr_", "vintlv((RegTensor<uint32_t>&)u64_dst"},
        {u64_dst, u64_src, mask});
}

TEST(BackendCCEVFOpsTest, ReduceSumRejectsBf16)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto bf16_dst = MakeVar("bf16_dst", ir::DataType::BF16);
    auto bf16_src = MakeVar("bf16_src", ir::DataType::BF16);
    auto mask = MakeVar("mask", ir::DataType::UINT32);

    EXPECT_ANY_THROW(Invoke(codegen, "vf.reduce_sum", {bf16_dst, bf16_src, mask}));
}

TEST(BackendCCEVFOpsTest, ReduceSumRejectsInt8)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto i8_dst = MakeVar("i8_dst", ir::DataType::INT8);
    auto i8_src = MakeVar("i8_src", ir::DataType::INT8);
    auto mask = MakeVar("mask", ir::DataType::UINT32);

    EXPECT_ANY_THROW(Invoke(codegen, "vf.reduce_sum", {i8_dst, i8_src, mask}));
}

// ============================================================================
// SqueezeStoreUnAlign / SqueezeStoreUnAlignPost — vstur/vstar squeeze path
// ============================================================================

TEST(BackendCCEVFOpsTest, SqueezeStoreUnAlignEmitsVstur)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::FP32);
    auto fp32 = MakeVar("fp32", ir::DataType::FP32);
    auto ureg = MakeVar("ureg", ir::DataType::UINT32);
    codegen.RegisterRegTensorVar("fp32");
    codegen.RegisterUnalignRegVar("ureg");

    auto emitted = Invoke(codegen, "vf.squeeze_store_unalign", {tile, fp32, ureg});
    ExpectContains(emitted, {"vstur(", "POST_UPDATE", "RegTensor<int32_t>", "(__ubuf__ int32_t *)"});
}

TEST(BackendCCEVFOpsTest, SqueezeStoreUnAlignRejectsWrongArgCount)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::FP32);
    auto fp32 = MakeVar("fp32", ir::DataType::FP32);

    // 2 args (missing align_reg) → should reject
    EXPECT_ANY_THROW(Invoke(codegen, "vf.squeeze_store_unalign", {tile, fp32}));
}

TEST(BackendCCEVFOpsTest, SqueezeStoreUnAlignPostEmitsVstar)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::FP32);
    auto ureg = MakeVar("ureg", ir::DataType::UINT32);
    codegen.RegisterUnalignRegVar("ureg");

    auto emitted = Invoke(codegen, "vf.squeeze_store_unalign_post", {tile, ureg});
    ExpectContains(emitted, {"vstar(", "(__ubuf__ int32_t *)"});
}

TEST(BackendCCEVFOpsTest, SqueezeStoreUnAlignPostRejectsWrongArgCount)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::FP32);
    auto ureg = MakeVar("ureg", ir::DataType::UINT32);
    auto extra = MakeVar("extra", ir::DataType::UINT32);

    // 3 args (extra) → should reject
    EXPECT_ANY_THROW(Invoke(codegen, "vf.squeeze_store_unalign_post", {tile, ureg, extra}));
}

TEST(BackendCCEVFOpsTest, StoreUnAlignNormEmitsVstusWithLvalueRef)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::FP32);
    auto fp32 = MakeVar("fp32", ir::DataType::FP32);
    auto ureg = MakeVar("ureg", ir::DataType::UINT32);
    codegen.RegisterRegTensorVar("fp32");
    codegen.RegisterUnalignRegVar("ureg");

    // post_update=False (NORM): vstus still needs __ubuf__ T*& (lvalue ref)
    auto emitted = Invoke(codegen, "vf.store_unalign", {tile, fp32, ureg, Int(2)});
    ExpectContains(emitted, {"vstus(", "NORM", "(__ubuf__ int32_t *&)"});
}

TEST(BackendCCEVFOpsTest, StoreUnAlignPostNormEmitsVstasThreeArgs)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::FP32);
    auto ureg = MakeVar("ureg", ir::DataType::UINT32);
    codegen.RegisterUnalignRegVar("ureg");

    // post_update=False (NORM): vstas takes 3 args (no POST_UPDATE suffix)
    auto emitted = Invoke(codegen, "vf.store_unalign_post", {tile, ureg, Int(2)});
    ExpectContains(emitted, {"vstas(", "(__ubuf__ int32_t *&)"});
    // NORM mode must NOT emit POST_UPDATE as 4th arg
    EXPECT_EQ(emitted.find("POST_UPDATE"), std::string::npos);
}

TEST(BackendCCEVFOpsTest, StoreUnAlignAddrRegEmitsVstu)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::FP32);
    auto fp32 = MakeVar("fp32", ir::DataType::FP32);
    auto ureg = MakeVar("ureg", ir::DataType::UINT32);
    auto addr = MakeVar("addr", ir::DataType::INT64);
    codegen.RegisterRegTensorVar("fp32");
    codegen.RegisterAddrRegVar("addr");
    codegen.RegisterUnalignRegVar("ureg");

    // AddrReg as 4th arg → vstu(ureg, areg, vreg, dst, POST_UPDATE)
    auto emitted = Invoke(codegen, "vf.store_unalign", {tile, fp32, ureg, addr}, {{"post_update", true}});
    ExpectContains(emitted, {"vstu(", "POST_UPDATE", "(__ubuf__ int32_t *&)"});
}

TEST(BackendCCEVFOpsTest, StoreUnAlignAddrRegNormEmitsVstuWithNorm)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::FP32);
    auto fp32 = MakeVar("fp32", ir::DataType::FP32);
    auto ureg = MakeVar("ureg", ir::DataType::UINT32);
    auto addr = MakeVar("addr", ir::DataType::INT64);
    codegen.RegisterRegTensorVar("fp32");
    codegen.RegisterAddrRegVar("addr");
    codegen.RegisterUnalignRegVar("ureg");

    // vstu also supports NORM mode (same post mode handling as vstus)
    auto emitted = Invoke(codegen, "vf.store_unalign", {tile, fp32, ureg, addr}, {{"post_update", false}});
    ExpectContains(emitted, {"vstu(", "NORM", "(__ubuf__ int32_t *&)"});
}

TEST(BackendCCEVFOpsTest, StoreUnAlignPostAddrRegEmitsVsta)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::FP32);
    auto ureg = MakeVar("ureg", ir::DataType::UINT32);
    auto addr = MakeVar("addr", ir::DataType::INT64);
    codegen.RegisterAddrRegVar("addr");
    codegen.RegisterUnalignRegVar("ureg");

    // AddrReg as 3rd arg → vsta(ureg, dst, areg)
    auto emitted = Invoke(codegen, "vf.store_unalign_post", {tile, ureg, addr});
    ExpectContains(emitted, {"vsta(", "(__ubuf__ int32_t *&)"});
    // vsta has no POST_UPDATE suffix
    EXPECT_EQ(emitted.find("POST_UPDATE"), std::string::npos);
}

TEST(BackendCCEVFOpsTest, StoreUnAlignPostAddrRegRejectsPostUpdate)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::FP32);
    auto ureg = MakeVar("ureg", ir::DataType::UINT32);
    auto addr = MakeVar("addr", ir::DataType::INT64);
    codegen.RegisterAddrRegVar("addr");
    codegen.RegisterUnalignRegVar("ureg");

    // vsta has no post mode; post_update kwarg should be rejected
    EXPECT_ANY_THROW(Invoke(codegen, "vf.store_unalign_post", {tile, ureg, addr}, {{"post_update", true}}));
}

TEST(BackendCCEVFOpsTest, StoreUnAlignB64EmitsVstusWithInt32CastAndDoubledStride)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::INT64);
    auto i64 = MakeVar("i64", ir::DataType::INT64);
    auto ureg = MakeVar("ureg", ir::DataType::UINT32);
    codegen.RegisterRegTensorVar("i64");
    codegen.RegisterUnalignRegVar("ureg");

    // b64 strided: cast to int32_t, stride doubled (mirrors AscendC b64 vstus path)
    auto emitted = Invoke(codegen, "vf.store_unalign", {tile, i64, ureg, Int(4)}, {{"post_update", true}});
    ExpectContains(emitted, {"vstus(", "(4) * 2", "RegTensor<int32_t>", "(__ubuf__ int32_t *&)"});
}

TEST(BackendCCEVFOpsTest, StoreUnAlignPostB64EmitsVstasWithInt32CastAndDoubledStride)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto tile = MakeTile("tile", ir::DataType::INT64);
    auto ureg = MakeVar("ureg", ir::DataType::UINT32);
    codegen.RegisterUnalignRegVar("ureg");

    // b64 strided post: cast to int32_t, stride doubled
    auto emitted = Invoke(codegen, "vf.store_unalign_post", {tile, ureg, Int(4)}, {{"post_update", true}});
    ExpectContains(emitted, {"vstas(", "(4) * 2", "(__ubuf__ int32_t *&)"});
}

// ============================================================================
// High-precision mode: subnormal threshold, union declaration, temp regs
// ============================================================================

TEST(BackendCCEVFOpsTest, HighPrecisionEmitsSubnormalThreshold)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto dst = MakeVar("dst");
    auto src = MakeVar("src");
    auto mask = MakeVar("mask", ir::DataType::UINT32);

    // All unary high-precision ops must emit the subnormal threshold 0x007FFFFF
    ExpectInvoke(codegen, "vf.exp", {"0x007FFFFF"}, {dst, src, mask}, {{"precision", true}});
    ExpectInvoke(codegen, "vf.ln", {"0x007FFFFF"}, {dst, src, mask}, {{"precision", true}});
    ExpectInvoke(codegen, "vf.log", {"0x007FFFFF"}, {dst, src, mask}, {{"precision", true}});
    ExpectInvoke(codegen, "vf.sqrt", {"0x007FFFFF"}, {dst, src, mask}, {{"precision", true}});
    ExpectInvoke(codegen, "vf.log2", {"0x007FFFFF"}, {dst, src, mask}, {{"precision", true}});
    ExpectInvoke(codegen, "vf.log10", {"0x007FFFFF"}, {dst, src, mask}, {{"precision", true}});
}

TEST(BackendCCEVFOpsTest, HighPrecisionEmitsUnionAndTempRegs)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto dst = MakeVar("dst");
    auto src = MakeVar("src");
    auto mask = MakeVar("mask", ir::DataType::UINT32);

    // exp: union + MaskReg + RegTensor<float> temps
    ExpectInvoke(codegen, "vf.exp", {"union { uint32_t i; float f; }", "MaskReg ", "RegTensor<float> "},
                 {dst, src, mask}, {{"precision", true}});
    // sqrt: same pattern
    ExpectInvoke(codegen, "vf.sqrt", {"union { uint32_t i; float f; }", "RegTensor<float> "}, {dst, src, mask},
                 {{"precision", true}});
    // ln: same pattern
    ExpectInvoke(codegen, "vf.ln", {"union { uint32_t i; float f; }", "MaskReg ", "RegTensor<float> "},
                 {dst, src, mask}, {{"precision", true}});
}

TEST(BackendCCEVFOpsTest, HighPrecisionDivEmitsTempRegsAndInstructions)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto dst = MakeVar("dst");
    auto src0 = MakeVar("src0");
    auto src1 = MakeVar("src1");
    auto mask = MakeVar("mask", ir::DataType::UINT32);

    // div high-precision: emits RegTensor<float> temps + full DivPrecisionImpl
    // sequence (bypass mask, subnormal scaling, +/-1 ulp rounding correction)
    auto emitted = Invoke(codegen, "vf.div", {dst, src0, src1, mask}, {{"precision", true}});
    ExpectContains(emitted, {"RegTensor<float>", "vmuls(", "vmula(", "vdiv(", "vcmps_ge(", "vabs("});
    // Should NOT emit a subnormal threshold constant (div scales via exponent
    // extraction, not a fixed threshold)
    EXPECT_EQ(emitted.find("0x007FFFFF"), std::string::npos) << emitted;
}

TEST(BackendCCEVFOpsTest, HighPrecisionDivFp16EmitsIeee754HalfImpl)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto dst = MakeVar("dst", ir::DataType::FP16);
    auto src0 = MakeVar("src0", ir::DataType::FP16);
    auto src1 = MakeVar("src1", ir::DataType::FP16);
    auto mask = MakeVar("mask", ir::DataType::UINT32);

    // FP16 high-precision mirrors AscendC DivIEEE754HalfImpl: subnormal
    // normalization (threshold 0x03FF, scale 2^10), exponent standardization
    // (0x83FF mask, 0x3C00 bias), raw vdiv on normalized operands, compensation
    // and overflow/underflow clamping on the exponent difference.
    auto emitted = Invoke(codegen, "vf.div", {dst, src0, src1, mask}, {{"precision", true}});
    ExpectContains(emitted, {"union { uint16_t i; half f; }", "0x03FF", "0x83FF", "0x3C00", "RegTensor<half>",
                             "RegTensor<int16_t>", "vcmps_lt(", "vcmp_le(", "vdiv(", "vmuls(", "vshr("});
    EXPECT_EQ(emitted.find("0x007FFFFF"), std::string::npos) << "FP16 should use 0x03FF, not 0x007FFFFF: " << emitted;
}

TEST(BackendCCEVFOpsTest, HighPrecisionDivRejectsNonFloat)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto i32_dst = MakeVar("i32_dst", ir::DataType::INT32);
    auto i32_src0 = MakeVar("i32_src0", ir::DataType::INT32);
    auto i32_src1 = MakeVar("i32_src1", ir::DataType::INT32);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    codegen.RegisterRegTensorVar("i32_dst");
    codegen.RegisterRegTensorVar("i32_src0");
    codegen.RegisterRegTensorVar("i32_src1");

    // INT32 src + precision=True → should reject (only FP16/FP32 supported)
    EXPECT_ANY_THROW(Invoke(codegen, "vf.div", {i32_dst, i32_src0, i32_src1, mask}, {{"precision", true}}));
}

TEST(BackendCCEVFOpsTest, PrecisionFalseBehavesAsDefault)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto dst = MakeVar("dst");
    auto src = MakeVar("src");
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    const Kwargs zeroing = {{"mode", EnumValue(ir::MergeMode::ZEROING)}, {"precision", false}};

    // precision=False should emit standard single-instruction, not multi-instruction
    ExpectInvoke(codegen, "vf.exp", {"vexp("}, {dst, src, mask}, zeroing);
    ExpectInvoke(codegen, "vf.ln", {"vln("}, {dst, src, mask}, zeroing);
    ExpectInvoke(codegen, "vf.sqrt", {"vsqrt("}, {dst, src, mask}, zeroing);
    auto emitted = Invoke(codegen, "vf.sqrt", {dst, src, mask}, zeroing);
    EXPECT_EQ(emitted.find("vcmps_lt("), std::string::npos) << "precision=False should not emit high-precision";
}

TEST(BackendCCEVFOpsTest, HighPrecisionLog2EmitsCorrectCompensation)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto dst = MakeVar("dst");
    auto src = MakeVar("src");
    auto mask = MakeVar("mask", ir::DataType::UINT32);

    // log2 high-precision: compensation is -23 (integer), scale factor 1/ln2
    auto emitted = Invoke(codegen, "vf.log2", {dst, src, mask}, {{"precision", true}});
    ExpectContains(emitted, {"vcmps_lt(", "8388608.0f", "1.4426950408889634f", "-23.0f", "vsel("});
    // Should NOT have the default-mode 0.434294 constant (that's log10)
    EXPECT_EQ(emitted.find("0.434294"), std::string::npos) << emitted;
}

TEST(BackendCCEVFOpsTest, HighPrecisionLog10EmitsCorrectCompensation)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto dst = MakeVar("dst");
    auto src = MakeVar("src");
    auto mask = MakeVar("mask", ir::DataType::UINT32);

    // log10 high-precision: compensation is -6.923689900271567f, scale 1/ln10
    auto emitted = Invoke(codegen, "vf.log10", {dst, src, mask}, {{"precision", true}});
    ExpectContains(emitted, {"vcmps_lt(", "8388608.0f", "0.43429448190325176f", "-6.923689900271567f", "vsel("});
    // Should NOT have the log2 constant 1/ln2
    EXPECT_EQ(emitted.find("1.442695"), std::string::npos) << emitted;
}

TEST(BackendCCEVFOpsTest, HighPrecisionHalfEmitsHalfSubnormalThreshold)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto dst = MakeVar("dst", ir::DataType::FP16);
    auto src = MakeVar("src", ir::DataType::FP16);
    auto mask = MakeVar("mask", ir::DataType::UINT32);

    ExpectInvoke(codegen, "vf.exp", {"0x03FF"}, {dst, src, mask}, {{"precision", true}});
    ExpectInvoke(codegen, "vf.ln", {"0x03FF"}, {dst, src, mask}, {{"precision", true}});
    ExpectInvoke(codegen, "vf.log", {"0x03FF"}, {dst, src, mask}, {{"precision", true}});
    ExpectInvoke(codegen, "vf.sqrt", {"0x03FF"}, {dst, src, mask}, {{"precision", true}});
    ExpectInvoke(codegen, "vf.log2", {"0x03FF"}, {dst, src, mask}, {{"precision", true}});
    ExpectInvoke(codegen, "vf.log10", {"0x03FF"}, {dst, src, mask}, {{"precision", true}});
}

TEST(BackendCCEVFOpsTest, HighPrecisionHalfEmitsHalfTypeRegsAndConstants)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto dst = MakeVar("dst", ir::DataType::FP16);
    auto src = MakeVar("src", ir::DataType::FP16);
    auto mask = MakeVar("mask", ir::DataType::UINT32);

    auto emitted = Invoke(codegen, "vf.ln", {dst, src, mask}, {{"precision", true}});
    ExpectContains(emitted, {"RegTensor<half>", "1024.0f", "-6.931471805599453094172f"});

    emitted = Invoke(codegen, "vf.log2", {dst, src, mask}, {{"precision", true}});
    ExpectContains(emitted, {"RegTensor<half>", "1024.0f", "-10.0f", "1.4426950408889634f"});

    emitted = Invoke(codegen, "vf.log10", {dst, src, mask}, {{"precision", true}});
    ExpectContains(emitted, {"RegTensor<half>", "1024.0f", "-3.01029995663981f", "0.43429448190325176f"});

    emitted = Invoke(codegen, "vf.sqrt", {dst, src, mask}, {{"precision", true}});
    // Scale-down compensation is 2^-6 (sqrt(2^12) = 2^6), mirroring AscendC 0x2400.
    ExpectContains(emitted, {"RegTensor<half>", "4096.0f", "0.015625f"});

    emitted = Invoke(codegen, "vf.exp", {dst, src, mask}, {{"precision", true}});
    ExpectContains(emitted, {"RegTensor<half>"});
    EXPECT_EQ(emitted.find("0x007FFFFF"), std::string::npos) << "FP16 should use 0x03FF, not 0x007FFFFF: " << emitted;
}

TEST(BackendCCEVFOpsTest, CompareRejectsMixedDtypeSources)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto mask = MakeVar("mask", ir::DataType::UINT32);
    auto i32_src = MakeVar("i32_src", ir::DataType::INT32);
    auto u32_src = MakeVar("u32_src", ir::DataType::UINT32);
    auto fp32_src = MakeVar("fp32_src", ir::DataType::FP32);
    codegen.RegisterRegTensorVar("i32_src");
    codegen.RegisterRegTensorVar("u32_src");
    codegen.RegisterRegTensorVar("fp32_src");
    codegen.RegisterMaskRegVar("mask");

    // AscendC CompareImpl takes both sources as one register type U — mixed
    // dtypes (even equal bit width) are rejected, no reinterpret casts.
    EXPECT_ANY_THROW(Invoke(codegen, "vf.eq", {mask, i32_src, u32_src, mask}));
    EXPECT_ANY_THROW(Invoke(codegen, "vf.lt", {mask, i32_src, u32_src, mask}));
    EXPECT_ANY_THROW(Invoke(codegen, "vf.eq", {mask, fp32_src, i32_src, mask}));
    // Same-type positive control still takes the vector form.
    ExpectInvoke(codegen, "vf.eq", {"vcmp_eq("}, {mask, i32_src, i32_src, mask});
}

TEST(BackendCCEVFOpsTest, BitCastEmitsReferenceCast)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto bf16_reg = MakeVar("bf16_reg", ir::DataType::BF16);
    auto u32_reg = MakeVar("u32_reg", ir::DataType::UINT32);
    auto view = MakeVar("view", ir::DataType::UINT16);
    codegen.RegisterRegTensorVar("bf16_reg");
    codegen.RegisterRegTensorVar("u32_reg");
    const auto* info = BackendCCE::Instance().GetOpInfo("vf.bit_cast");
    ASSERT_NE(info, nullptr);

    // Nested (1-arg) form: the inline reference cast is RETURNED for the parent
    // op to inline, not emitted.
    auto ret = info->codegen_func(MakeCall("vf.bit_cast", {bf16_reg}, {{"dtype", ir::DataType::UINT16}}), codegen);
    EXPECT_EQ(ret, "(RegTensor<uint16_t> &)bf16_reg");
    ret = info->codegen_func(MakeCall("vf.bit_cast", {u32_reg}, {{"dtype", ir::DataType::UINT32}}), codegen);
    EXPECT_EQ(ret, "(RegTensor<uint32_t> &)u32_reg");

    // Assignment (2-arg) form: view register bound to the reference cast.
    ExpectInvoke(codegen, "vf.bit_cast", {"view = (RegTensor<uint16_t>&)bf16_reg;"}, {view, bf16_reg},
                 {{"dtype", ir::DataType::UINT16}});
}

TEST(BackendCCEVFOpsTest, BitCastViewFeedsVectorCompare)
{
    CapturingCCECodegen codegen(ir::SectionKind::Vector);
    auto preg_b8 = MakeVar("preg_b8", ir::DataType::UINT8);
    auto vreg_high = MakeVar("vreg_high", ir::DataType::UINT8);
    auto idx_high = MakeVar("idx_high", ir::DataType::UINT32);
    auto idx_view = MakeVar("idx_high_view", ir::DataType::UINT8);
    codegen.RegisterRegTensorVar("vreg_high");
    codegen.RegisterRegTensorVar("idx_high");
    codegen.RegisterRegTensorVar("idx_high_view");
    codegen.RegisterMaskRegVar("preg_b8");

    // The parser materializes a nested bit_cast into a temp register; the compare
    // must dispatch the temp to the vector form (vcmp_eq), not vcmps_eq.
    Invoke(codegen, "vf.bit_cast", {idx_view, idx_high}, {{"dtype", ir::DataType::UINT8}});
    ExpectInvoke(codegen, "vf.eq", {"vcmp_eq("}, {preg_b8, vreg_high, idx_view, preg_b8});
    EXPECT_EQ(Invoke(codegen, "vf.eq", {preg_b8, vreg_high, idx_view, preg_b8}).find("vcmps_"), std::string::npos)
        << "u8 register source must not take the scalar compare path";
}

} // namespace
} // namespace backend
} // namespace pypto
