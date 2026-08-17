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
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "backend/backend_cce.h"
#include "backend/common/backend.h"
#include "backend/common/backend_utils.h"
#include "codegen/cce/cce_codegen.h"
#include "core/error.h"
#include "ir/debug_info.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/memref.h"
#include "ir/memory_space.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/type.h"

namespace pypto {
namespace backend {
namespace {

class TestableCCECodegen : public codegen::CCECodegen {
public:
    using codegen::CCECodegen::CCECodegen;
    std::string GetEmittedCode() { return emitter_.GetCode(); }
    void SetCurrentTargetVar(const std::string& var) { current_target_var_ = var; }
};

ir::ExprPtr MakeConstInt(int64_t value)
{
    return std::make_shared<const ir::ConstInt>(value, ir::DataType::INT64, ir::Span::Unknown());
}

ir::VarPtr MakeVar(const std::string& name, const ir::TypePtr& type)
{
    return std::make_shared<const ir::Var>(name, type, ir::Span::Unknown());
}

ir::TypePtr MakeTileType(std::vector<int64_t> shape = {16, 16}, ir::DataType dtype = ir::DataType::FP16,
                         std::optional<ir::MemRefPtr> memref = std::nullopt,
                         std::optional<ir::HardwareInfo> hw = std::nullopt)
{
    return std::make_shared<const ir::TileType>(shape, dtype, memref, std::nullopt, hw);
}

ir::TypePtr MakeTensorType(std::vector<int64_t> shape = {32, 32}, ir::DataType dtype = ir::DataType::FP16)
{
    return std::make_shared<const ir::TensorType>(shape, dtype, std::nullopt);
}

ir::VarPtr MakeTensorVar(const std::string& name, const std::vector<int64_t>& shape, ir::DataType dtype)
{
    auto ptr = MakeVar(name + "_base", std::make_shared<const ir::PtrType>(dtype));
    ir::TensorView view({}, ir::TensorLayout::ND, ptr);
    auto tensor_type = std::make_shared<const ir::TensorType>(shape, dtype, std::optional<ir::MemRefPtr>(std::nullopt),
                                                              std::optional<ir::TensorView>(view));
    return MakeVar(name, tensor_type);
}

ir::TypePtr MakeScalarType(ir::DataType dtype = ir::DataType::INT32)
{
    return std::make_shared<const ir::ScalarType>(dtype);
}

ir::ExprPtr MakeOffsets(int64_t m, int64_t k)
{
    return std::make_shared<const ir::MakeTuple>(std::vector<ir::ExprPtr>{MakeConstInt(m), MakeConstInt(k)},
                                                 ir::Span::Unknown());
}

ir::CallPtr MakeCall(const std::string& name, std::vector<ir::ExprPtr> args)
{
    return std::make_shared<const ir::Call>(name, args, ir::Span::Unknown());
}

ir::CallPtr MakeCallWithKwargs(const std::string& name, std::vector<ir::ExprPtr> args,
                               std::vector<std::pair<std::string, std::any>> kwargs)
{
    return std::make_shared<const ir::Call>(name, args, kwargs, ir::Span::Unknown());
}

ir::MemRefPtr MakeMemRef(ir::MemorySpace space)
{
    return std::make_shared<const ir::MemRef>(space, MakeConstInt(0), 1024, ir::Span::Unknown());
}

ir::ProgramPtr MakeProgram(const ir::StmtPtr& body, const std::vector<ir::VarPtr>& params = {},
                           ir::IRDebugInfoPtr debug_info = nullptr)
{
    auto function = std::make_shared<const ir::Function>("kernel", params, std::vector<ir::TypePtr>{}, body,
                                                         ir::Span::Unknown(), ir::FunctionType::IN_CORE, true);
    return std::make_shared<const ir::Program>(std::vector<ir::FunctionPtr>{function}, "test_program",
                                               ir::Span::Unknown(), std::move(debug_info));
}

std::string RunCodegen(const std::string& op_name, const ir::CallPtr& call)
{
    TestableCCECodegen codegen(ir::SectionKind::Vector);
    auto* info = BackendCCE::Instance().GetOpInfo(op_name);
    EXPECT_NE(info, nullptr);
    if (info != nullptr) {
        info->codegen_func(call, codegen);
    }
    return codegen.GetEmittedCode();
}

std::string RunSsbufCodegen(const std::string& op_name)
{
    auto scalar = MakeScalarType();
    auto tuple_type = std::make_shared<const ir::TupleType>(std::vector<ir::TypePtr>{scalar});
    // What makes this tuple a struct is the parser's side table, which codegen reads to resolve
    // the C++ type name. A hand-built program has to supply it the same way the parser does.
    auto debug_info = std::make_shared<ir::IRDebugInfo>();
    debug_info->RegisterTupleName(tuple_type, "SsbufStruct");
    debug_info->RegisterTupleFields(tuple_type, {"value"});
    auto struct_var = MakeVar("struct_var", tuple_type);
    auto create = std::make_shared<const ir::Call>("struct.create", std::vector<ir::ExprPtr>{MakeVar("value", scalar)},
                                                   std::vector<std::pair<std::string, std::any>>{
                                                       {"name", std::string("SsbufStruct")},
                                                       {"fields", std::vector<std::string>{"value"}},
                                                   },
                                                   tuple_type, ir::Span::Unknown());
    auto create_stmt = std::make_shared<const ir::AssignStmt>(struct_var, create, ir::Span::Unknown());
    auto ssbuf = MakeCall(op_name, {struct_var, MakeConstInt(0)});
    auto body = std::make_shared<const ir::SeqStmts>(
        std::vector<ir::StmtPtr>{create_stmt, std::make_shared<const ir::EvalStmt>(ssbuf, ir::Span::Unknown())},
        ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    return codegen.GenerateSingle(MakeProgram(body, {}, debug_info), "a5");
}

#define EXPECT_CONTAINS(haystack, needle)                                                                          \
    do {                                                                                                           \
        const auto& haystack_value = (haystack);                                                                   \
        const auto needle_value = (needle);                                                                        \
        EXPECT_NE(haystack_value.find(needle_value), std::string::npos) << needle_value << "\n" << haystack_value; \
    } while (false)

#define EXPECT_NOT_CONTAINS(haystack, needle)                                                                      \
    do {                                                                                                           \
        const auto& haystack_value = (haystack);                                                                   \
        const auto needle_value = (needle);                                                                        \
        EXPECT_EQ(haystack_value.find(needle_value), std::string::npos) << needle_value << "\n" << haystack_value; \
    } while (false)

struct SimpleOpParam {
    std::string op_name;
    std::string cce_op;
};

class BinaryOpTest : public ::testing::TestWithParam<SimpleOpParam> {};
TEST_P(BinaryOpTest, EmitsCorrectCode)
{
    auto& p = GetParam();
    auto tile = MakeTileType();
    auto call = MakeCall(p.op_name, {MakeVar("dst", tile), MakeVar("lhs", tile), MakeVar("rhs", tile)});
    auto code = RunCodegen(p.op_name, call);
    EXPECT_CONTAINS(code, p.cce_op + "(dst, lhs, rhs);");
}
INSTANTIATE_TEST_SUITE_P(
    BlockOutBinaryOps, BinaryOpTest,
    ::testing::Values(SimpleOpParam{"block.add", "TADD"}, SimpleOpParam{"block.sub", "TSUB"},
                      SimpleOpParam{"block.mul", "TMUL"}, SimpleOpParam{"block.div", "TDIV"},
                      SimpleOpParam{"block.rem", "TREM"}, SimpleOpParam{"block.maximum", "TMAX"},
                      SimpleOpParam{"block.minimum", "TMIN"}, SimpleOpParam{"block.and", "TAND"},
                      SimpleOpParam{"block.or", "TOR"}, SimpleOpParam{"block.shl", "TSHL"},
                      SimpleOpParam{"block.shr", "TSHR"}, SimpleOpParam{"block.gatherb", "TGATHERB"},
                      SimpleOpParam{"block.scatter", "TSCATTER"}, SimpleOpParam{"block.gemv", "TGEMV"},
                      SimpleOpParam{"block.partadd", "TPARTADD"}, SimpleOpParam{"block.partmax", "TPARTMAX"},
                      SimpleOpParam{"block.partmin", "TPARTMIN"}, SimpleOpParam{"block.partmul", "TPARTMUL"}));

class UnaryOpTest : public ::testing::TestWithParam<SimpleOpParam> {};
TEST_P(UnaryOpTest, EmitsCorrectCode)
{
    auto& p = GetParam();
    auto tile = MakeTileType();
    auto call = MakeCall(p.op_name, {MakeVar("dst", tile), MakeVar("src", tile)});
    auto code = RunCodegen(p.op_name, call);
    EXPECT_CONTAINS(code, p.cce_op + "(dst, src);");
}
INSTANTIATE_TEST_SUITE_P(BlockOutUnaryOps, UnaryOpTest,
                         ::testing::Values(SimpleOpParam{"block.neg", "TNEG"}, SimpleOpParam{"block.exp", "TEXP"},
                                           SimpleOpParam{"block.sqrt", "TSQRT"}, SimpleOpParam{"block.rsqrt", "TRSQRT"},
                                           SimpleOpParam{"block.recip", "TRECIP"}, SimpleOpParam{"block.log", "TLOG"},
                                           SimpleOpParam{"block.abs", "TABS"}, SimpleOpParam{"block.relu", "TRELU"},
                                           SimpleOpParam{"block.not", "TNOT"}));

class ScalarOpTest : public ::testing::TestWithParam<SimpleOpParam> {};
TEST_P(ScalarOpTest, EmitsCorrectCode)
{
    auto& p = GetParam();
    auto tile = MakeTileType();
    auto scal = MakeScalarType();
    auto call = MakeCall(p.op_name, {MakeVar("dst", tile), MakeVar("tile", tile), MakeVar("scalar", scal)});
    auto code = RunCodegen(p.op_name, call);
    EXPECT_CONTAINS(code, p.cce_op + "(dst, tile, scalar);");
}
INSTANTIATE_TEST_SUITE_P(BlockOutScalarOps, ScalarOpTest,
                         ::testing::Values(SimpleOpParam{"block.adds", "TADDS"}, SimpleOpParam{"block.subs", "TSUBS"},
                                           SimpleOpParam{"block.muls", "TMULS"}, SimpleOpParam{"block.divs", "TDIVS"},
                                           SimpleOpParam{"block.rems", "TREMS"}, SimpleOpParam{"block.ands", "TANDS"},
                                           SimpleOpParam{"block.ors", "TORS"}, SimpleOpParam{"block.shls", "TSHLS"},
                                           SimpleOpParam{"block.shrs", "TSHRS"}, SimpleOpParam{"block.maxs", "TMAXS"},
                                           SimpleOpParam{"block.mins", "TMINS"}, SimpleOpParam{"block.axpy", "TAXPY"}));

class RowReductionOpTest : public ::testing::TestWithParam<SimpleOpParam> {};
TEST_P(RowReductionOpTest, EmitsCorrectCode)
{
    auto& p = GetParam();
    auto tile = MakeTileType();
    auto call = MakeCall(p.op_name, {MakeVar("dst", tile), MakeVar("tile", tile), MakeVar("tmp", tile)});
    auto code = RunCodegen(p.op_name, call);
    EXPECT_CONTAINS(code, p.cce_op + "(dst, tile, tmp);");
}
INSTANTIATE_TEST_SUITE_P(
    BlockOutRowReductionOps, RowReductionOpTest,
    ::testing::Values(SimpleOpParam{"block.row_sum", "TROWSUM"}, SimpleOpParam{"block.row_max", "TROWMAX"},
                      SimpleOpParam{"block.row_min", "TROWMIN"}, SimpleOpParam{"block.row_prod", "TROWPROD"},
                      SimpleOpParam{"block.row_argmax", "TROWARGMAX"}, SimpleOpParam{"block.row_argmin", "TROWARGMIN"},
                      SimpleOpParam{"block.col_argmax", "TCOLARGMAX"},
                      SimpleOpParam{"block.col_argmin", "TCOLARGMIN"}));

class RowColExpandOpTest : public ::testing::TestWithParam<SimpleOpParam> {};
TEST_P(RowColExpandOpTest, EmitsCorrectCode)
{
    auto& p = GetParam();
    auto tile = MakeTileType();
    auto call = MakeCall(p.op_name, {MakeVar("dst", tile), MakeVar("tile", tile), MakeVar("red", tile)});
    auto code = RunCodegen(p.op_name, call);
    EXPECT_CONTAINS(code, p.cce_op + "(dst, tile, red);");
}
INSTANTIATE_TEST_SUITE_P(
    BlockOutExpandOps, RowColExpandOpTest,
    ::testing::Values(
        SimpleOpParam{"block.row_expand_add", "TROWEXPANDADD"}, SimpleOpParam{"block.row_expand_sub", "TROWEXPANDSUB"},
        SimpleOpParam{"block.row_expand_mul", "TROWEXPANDMUL"}, SimpleOpParam{"block.row_expand_div", "TROWEXPANDDIV"},
        SimpleOpParam{"block.row_expand_max", "TROWEXPANDMAX"}, SimpleOpParam{"block.row_expand_min", "TROWEXPANDMIN"},
        SimpleOpParam{"block.row_expand_expdif", "TROWEXPANDEXPDIF"},
        SimpleOpParam{"block.col_expand_add", "TCOLEXPANDADD"}, SimpleOpParam{"block.col_expand_sub", "TCOLEXPANDSUB"},
        SimpleOpParam{"block.col_expand_mul", "TCOLEXPANDMUL"}, SimpleOpParam{"block.col_expand_div", "TCOLEXPANDDIV"},
        SimpleOpParam{"block.col_expand_max", "TCOLEXPANDMAX"}, SimpleOpParam{"block.col_expand_min", "TCOLEXPANDMIN"},
        SimpleOpParam{"block.col_expand_expdif", "TCOLEXPANDEXPDIF"}));

class TernaryOpTest : public ::testing::TestWithParam<SimpleOpParam> {};
TEST_P(TernaryOpTest, EmitsCorrectCode)
{
    auto& p = GetParam();
    auto tile = MakeTileType();
    auto call = MakeCall(p.op_name, {MakeVar("dst", tile), MakeVar("a", tile), MakeVar("b", tile), MakeVar("c", tile)});
    auto code = RunCodegen(p.op_name, call);
    EXPECT_CONTAINS(code, p.cce_op + "(dst, a, b, c);");
}
INSTANTIATE_TEST_SUITE_P(BlockOutTernaryOps, TernaryOpTest,
                         ::testing::Values(SimpleOpParam{"block.xor", "TXOR"}, SimpleOpParam{"block.xors", "TXORS"},
                                           SimpleOpParam{"block.addc", "TADDC"}, SimpleOpParam{"block.subc", "TSUBC"},
                                           SimpleOpParam{"block.addsc", "TADDSC"},
                                           SimpleOpParam{"block.subsc", "TSUBSC"},
                                           SimpleOpParam{"block.gemv_acc", "TGEMV_ACC"},
                                           SimpleOpParam{"block.gemv_bias", "TGEMV_BIAS"}));

class QuaternaryOpTest : public ::testing::TestWithParam<SimpleOpParam> {};
TEST_P(QuaternaryOpTest, EmitsCorrectCode)
{
    auto& p = GetParam();
    auto tile = MakeTileType();
    auto call = MakeCall(p.op_name, {MakeVar("dst", tile), MakeVar("a", tile), MakeVar("b", tile), MakeVar("c", tile),
                                     MakeVar("d", tile)});
    auto code = RunCodegen(p.op_name, call);
    EXPECT_CONTAINS(code, p.cce_op + "(dst, a, b, c, d);");
}
INSTANTIATE_TEST_SUITE_P(BlockOutQuaternaryOps, QuaternaryOpTest,
                         ::testing::Values(SimpleOpParam{"block.sel", "TSEL"}, SimpleOpParam{"block.sels", "TSELS"}));

TEST(BackendCCEBlockOutOps, ColMax)
{
    auto tile = MakeTileType();
    auto call = MakeCall("block.col_max", {MakeVar("dst", tile), MakeVar("tile", tile), MakeVar("tmp", tile)});
    EXPECT_CONTAINS(RunCodegen("block.col_max", call), "TCOLMAX(dst, tile);");
}

TEST(BackendCCEBlockOutOps, ColMin)
{
    auto tile = MakeTileType();
    auto call = MakeCall("block.col_min", {MakeVar("dst", tile), MakeVar("tile", tile), MakeVar("tmp", tile)});
    EXPECT_CONTAINS(RunCodegen("block.col_min", call), "TCOLMIN(dst, tile);");
}

TEST(BackendCCEBlockOutOps, ColProd)
{
    auto tile = MakeTileType();
    auto call = MakeCall("block.col_prod", {MakeVar("dst", tile), MakeVar("tile", tile), MakeVar("tmp", tile)});
    EXPECT_CONTAINS(RunCodegen("block.col_prod", call), "TCOLPROD(dst, tile);");
}

TEST(BackendCCEBlockOutOps, ColSum)
{
    auto tile = MakeTileType();
    auto call = MakeCall("block.col_sum", {MakeVar("dst", tile), MakeVar("tile", tile), MakeVar("tmp", tile)});
    EXPECT_CONTAINS(RunCodegen("block.col_sum", call), "TCOLSUM(dst, tile, tmp, false);");
}

TEST(BackendCCEBlockOutOps, RowExpandUnary)
{
    auto tile = MakeTileType();
    auto call = MakeCall("block.row_expand", {MakeVar("dst", tile), MakeVar("src", tile)});
    EXPECT_CONTAINS(RunCodegen("block.row_expand", call), "TROWEXPAND(dst, src);");
}

TEST(BackendCCEBlockOutOps, ColExpandUnary)
{
    auto tile = MakeTileType();
    auto call = MakeCall("block.col_expand", {MakeVar("dst", tile), MakeVar("src", tile)});
    EXPECT_CONTAINS(RunCodegen("block.col_expand", call), "TCOLEXPAND(dst, src);");
}

TEST(BackendCCEBlockOutOps, ExpandBinop)
{
    auto tile = MakeTileType();
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"block.row_expand_binop", "TROWEXPANDADD"}, {"block.row_expand_binop", "TROWEXPANDSUB"},
        {"block.row_expand_binop", "TROWEXPANDMUL"}, {"block.row_expand_binop", "TROWEXPANDDIV"},
        {"block.row_expand_binop", "TROWEXPANDMAX"}, {"block.row_expand_binop", "TROWEXPANDMIN"},
        {"block.col_expand_binop", "TCOLEXPANDADD"}, {"block.col_expand_binop", "TCOLEXPANDSUB"},
        {"block.col_expand_binop", "TCOLEXPANDMUL"}, {"block.col_expand_binop", "TCOLEXPANDDIV"},
        {"block.col_expand_binop", "TCOLEXPANDMAX"}, {"block.col_expand_binop", "TCOLEXPANDMIN"},
    };
    for (size_t i = 0; i < cases.size(); ++i) {
        auto call = MakeCallWithKwargs(cases[i].first,
                                       {MakeVar("dst", tile), MakeVar("tile", tile), MakeVar("red", tile)},
                                       {{"op_type", static_cast<int>(i % 6)}});
        EXPECT_CONTAINS(RunCodegen(cases[i].first, call), cases[i].second + "(dst, tile, red);");
    }
}

TEST(BackendCCEBlockOutOps, BinaryRelu)
{
    auto tile = MakeTileType();
    auto call = MakeCall("block.add_relu", {MakeVar("dst", tile), MakeVar("lhs", tile), MakeVar("rhs", tile)});
    auto code = RunCodegen("block.add_relu", call);
    EXPECT_CONTAINS(code, "TADD(lhs, lhs, rhs);");
    EXPECT_CONTAINS(code, "TRELU(dst, lhs);");

    auto call2 = MakeCall("block.sub_relu", {MakeVar("dst", tile), MakeVar("lhs", tile), MakeVar("rhs", tile)});
    auto code2 = RunCodegen("block.sub_relu", call2);
    EXPECT_CONTAINS(code2, "TSUB(lhs, lhs, rhs);");
    EXPECT_CONTAINS(code2, "TRELU(dst, lhs);");
}

TEST(BackendCCEBlockOutOps, BinaryReluCast)
{
    auto tile = MakeTileType();
    auto call = MakeCallWithKwargs("block.add_relu_cast",
                                   {MakeVar("dst", tile), MakeVar("lhs", tile), MakeVar("rhs", tile)}, {{"mode", 1}});
    auto code = RunCodegen("block.add_relu_cast", call);
    EXPECT_CONTAINS(code, "TADD(lhs, lhs, rhs);");
    EXPECT_CONTAINS(code, "TRELU(lhs, lhs);");
    EXPECT_CONTAINS(code, "TCVT(dst, lhs, RoundMode::CAST_RINT);");

    auto call2 = MakeCallWithKwargs("block.sub_relu_cast",
                                    {MakeVar("dst", tile), MakeVar("lhs", tile), MakeVar("rhs", tile)}, {{"mode", 2}});
    auto code2 = RunCodegen("block.sub_relu_cast", call2);
    EXPECT_CONTAINS(code2, "TSUB(lhs, lhs, rhs);");
    EXPECT_CONTAINS(code2, "RoundMode::CAST_ROUND");
}

TEST(BackendCCEBlockOutOps, BinaryCast)
{
    auto tile = MakeTileType();
    auto call = MakeCallWithKwargs("block.mul_cast", {MakeVar("dst", tile), MakeVar("lhs", tile), MakeVar("rhs", tile)},
                                   {{"mode", 0}});
    auto code = RunCodegen("block.mul_cast", call);
    EXPECT_CONTAINS(code, "TMUL(lhs, lhs, rhs);");
    EXPECT_CONTAINS(code, "TCVT(dst, lhs, RoundMode::CAST_NONE);");
}

TEST(BackendCCEBlockOutOps, FusedOps)
{
    auto tile = MakeTileType();
    auto call = MakeCall("block.mul_add_dst", {MakeVar("out", tile), MakeVar("lhs", tile), MakeVar("rhs", tile)});
    auto code = RunCodegen("block.mul_add_dst", call);
    EXPECT_CONTAINS(code, "TMUL(lhs, lhs, rhs);");
    EXPECT_CONTAINS(code, "TADD(out, lhs, out);");

    auto call2 = MakeCall("block.fused_mul_add", {MakeVar("out", tile), MakeVar("lhs", tile), MakeVar("rhs", tile)});
    auto code2 = RunCodegen("block.fused_mul_add", call2);
    EXPECT_CONTAINS(code2, "TMUL(lhs, lhs, out);");
    EXPECT_CONTAINS(code2, "TADD(out, lhs, rhs);");

    auto call3 = MakeCall("block.fused_mul_add_relu",
                          {MakeVar("out", tile), MakeVar("lhs", tile), MakeVar("rhs", tile)});
    auto code3 = RunCodegen("block.fused_mul_add_relu", call3);
    EXPECT_CONTAINS(code3, "TMUL(lhs, lhs, out);");
    EXPECT_CONTAINS(code3, "TADD(lhs, lhs, rhs);");
    EXPECT_CONTAINS(code3, "TRELU(out, lhs);");
}

TEST(BackendCCEBlockOutOps, Matmul)
{
    auto tile = MakeTileType();
    auto call = MakeCall("block.matmul", {MakeVar("dst", tile), MakeVar("left", tile), MakeVar("right", tile)});
    EXPECT_CONTAINS(RunCodegen("block.matmul", call), "TMATMUL(dst, left, right);");

    auto callPhase = MakeCallWithKwargs(
        "block.matmul", {MakeVar("dst", tile), MakeVar("left", tile), MakeVar("right", tile)}, {{"phase", 1}});
    EXPECT_CONTAINS(RunCodegen("block.matmul", callPhase), "TMATMUL<AccPhase::Partial>(dst, left, right);");
}

TEST(BackendCCEBlockOutOps, MatmulAcc)
{
    auto tile = MakeTileType();
    auto call = MakeCall("block.matmul_acc",
                         {MakeVar("dst", tile), MakeVar("acc", tile), MakeVar("left", tile), MakeVar("right", tile)});
    EXPECT_CONTAINS(RunCodegen("block.matmul_acc", call), "TMATMUL_ACC(dst, acc, left, right);");

    auto callPhase = MakeCallWithKwargs(
        "block.matmul_acc", {MakeVar("dst", tile), MakeVar("acc", tile), MakeVar("left", tile), MakeVar("right", tile)},
        {{"phase", 2}});
    EXPECT_CONTAINS(RunCodegen("block.matmul_acc", callPhase), "TMATMUL_ACC<AccPhase::Final>(dst, acc, left, right);");
}

TEST(BackendCCEBlockOutOps, MatmulBias)
{
    auto tile = MakeTileType();
    auto call = MakeCall("block.matmul_bias",
                         {MakeVar("dst", tile), MakeVar("left", tile), MakeVar("right", tile), MakeVar("bias", tile)});
    EXPECT_CONTAINS(RunCodegen("block.matmul_bias", call), "TMATMUL_BIAS(dst, left, right, bias);");

    auto callPartial = MakeCallWithKwargs(
        "block.matmul_bias",
        {MakeVar("dst", tile), MakeVar("left", tile), MakeVar("right", tile), MakeVar("bias", tile)}, {{"phase", 1}});
    EXPECT_CONTAINS(RunCodegen("block.matmul_bias", callPartial),
                    "TMATMUL_BIAS<AccPhase::Partial>(dst, left, right, bias);");

    auto callFinal = MakeCallWithKwargs(
        "block.matmul_bias",
        {MakeVar("dst", tile), MakeVar("left", tile), MakeVar("right", tile), MakeVar("bias", tile)}, {{"phase", 2}});
    EXPECT_CONTAINS(RunCodegen("block.matmul_bias", callFinal),
                    "TMATMUL_BIAS<AccPhase::Final>(dst, left, right, bias);");
}

TEST(BackendCCEBlockOutOps, Cast)
{
    auto tile = MakeTileType();
    auto call = MakeCallWithKwargs("block.cast", {MakeVar("dst", tile), MakeVar("src", tile)}, {{"mode", 3}});
    auto code = RunCodegen("block.cast", call);
    EXPECT_CONTAINS(code, "TCVT(dst, src, RoundMode::CAST_FLOOR);");
}

TEST(BackendCCEBlockOutOps, Cmp)
{
    auto tile = MakeTileType();
    for (int i = 0; i <= 5; ++i) {
        auto call = MakeCallWithKwargs("block.cmp", {MakeVar("dst", tile), MakeVar("lhs", tile), MakeVar("rhs", tile)},
                                       {{"cmp_mode", i}});
        auto code = RunCodegen("block.cmp", call);
        EXPECT_CONTAINS(code, "TCMP(dst, lhs, rhs, ");
    }
    auto call0 = MakeCallWithKwargs("block.cmp", {MakeVar("dst", tile), MakeVar("lhs", tile), MakeVar("rhs", tile)},
                                    {{"cmp_mode", 0}});
    EXPECT_CONTAINS(RunCodegen("block.cmp", call0), "TCMP(dst, lhs, rhs, CmpMode::EQ);");
}

TEST(BackendCCEBlockOutOps, Cmps)
{
    auto tile = MakeTileType();
    auto scal = MakeScalarType();
    auto call = MakeCallWithKwargs("block.cmps", {MakeVar("dst", tile), MakeVar("tile", tile), MakeVar("scalar", scal)},
                                   {{"cmp_mode", 4}});
    EXPECT_CONTAINS(RunCodegen("block.cmps", call), "TCMPS(dst, tile, scalar, CmpMode::GT);");
}

TEST(BackendCCEBlockOutOps, Expands)
{
    auto fp_tile = MakeTileType({16, 16}, ir::DataType::FP16);
    auto int_scal = MakeScalarType(ir::DataType::INT32);
    auto call = MakeCall("block.expands", {MakeVar("dst", fp_tile), MakeVar("scalar", int_scal)});
    auto code = RunCodegen("block.expands", call);
    EXPECT_CONTAINS(code, "TEXPANDS(dst, (float)((int64_t)(scalar)));");

    auto fp_scal = MakeScalarType(ir::DataType::FP16);
    auto call2 = MakeCall("block.expands", {MakeVar("dst", fp_tile), MakeVar("scalar", fp_scal)});
    auto code2 = RunCodegen("block.expands", call2);
    EXPECT_CONTAINS(code2, "TEXPANDS(dst, scalar);");
    EXPECT_NOT_CONTAINS(code2, "(float)");
}

TEST(BackendCCEBlockOutOps, Full)
{
    auto fp_tile = MakeTileType({16, 16}, ir::DataType::FP16);
    auto fp_scal = MakeScalarType(ir::DataType::FP32);
    auto call = MakeCall("block.full", {MakeVar("dst", fp_tile), MakeVar("scalar", fp_scal)});
    EXPECT_CONTAINS(RunCodegen("block.full", call), "TEXPANDS(dst, scalar);");
}

TEST(BackendCCEBlockOutOps, FillIndex)
{
    auto tile = MakeTileType({16, 16}, ir::DataType::INT32);
    auto call = MakeCall("block.fill_index", {MakeVar("dst", tile), MakeConstInt(0)});
    EXPECT_CONTAINS(RunCodegen("block.fill_index", call), "TCI<");

    auto tile16 = MakeTileType({16, 16}, ir::DataType::INT16);
    auto call16 = MakeCall("block.fill_index", {MakeVar("dst", tile16), MakeConstInt(0)});
    EXPECT_CONTAINS(RunCodegen("block.fill_index", call16), "int16_t");

    auto tile32 = MakeTileType({16, 16}, ir::DataType::UINT32);
    auto call32 = MakeCall("block.fill_index", {MakeVar("dst", tile32), MakeConstInt(0)});
    EXPECT_CONTAINS(RunCodegen("block.fill_index", call32), "uint32_t");

    auto tile16u = MakeTileType({16, 16}, ir::DataType::UINT16);
    auto call16u = MakeCall("block.fill_index", {MakeVar("dst", tile16u), MakeConstInt(0)});
    EXPECT_CONTAINS(RunCodegen("block.fill_index", call16u), "uint16_t");
}

TEST(BackendCCEBlockOutOps, Reshape)
{
    auto tile = MakeTileType();
    auto call = MakeCall("block.reshape", {MakeVar("dst", tile), MakeVar("src", tile), MakeConstInt(16)});
    EXPECT_CONTAINS(RunCodegen("block.reshape", call), "TRESHAPE(dst, src);");
}

TEST(BackendCCEBlockOutOps, Transpose)
{
    auto tile = MakeTileType();
    auto call = MakeCall("block.transpose", {MakeVar("dst", tile), MakeVar("src", tile)});
    EXPECT_CONTAINS(RunCodegen("block.transpose", call), "TTRANS(dst, src, src);");
}

TEST(BackendCCEBlockOutOps, Insert)
{
    auto tile = MakeTileType();
    auto call = MakeCall("block.insert",
                         {MakeVar("dst", tile), MakeVar("src", tile), MakeConstInt(0), MakeConstInt(0)});
    EXPECT_CONTAINS(RunCodegen("block.insert", call), "TINSERT(dst, src, 0, 0);");
}

TEST(BackendCCEBlockOutOps, SetValidShape)
{
    auto tile = MakeTileType();
    auto call = MakeCall("block.set_validshape", {MakeVar("tile", tile), MakeConstInt(16), MakeConstInt(8)});
    auto code = RunCodegen("block.set_validshape", call);
    EXPECT_CONTAINS(code, "tile.SetValidShape(16, 8);");
}

TEST(BackendCCEBlockOutOps, SetStride)
{
    auto tensor_type = MakeTensorType();
    auto call = MakeCall("block.set_stride", {MakeVar("tensor", tensor_type), MakeConstInt(128), MakeConstInt(64)});
    auto code = RunCodegen("block.set_stride", call);
    EXPECT_CONTAINS(code, "tensor.SetStride<pto::GlobalTensorDim::DIM_3, pto::GlobalTensorDim::DIM_4>");
    EXPECT_CONTAINS(code, "static_cast<int64_t>(128)");
    EXPECT_CONTAINS(code, "static_cast<int64_t>(64)");
}

TEST(BackendCCEBlockOutOps, Load)
{
    TestableCCECodegen codegen(ir::SectionKind::Vector);
    auto tensor_type = MakeTensorType();
    auto tile_type = MakeTileType();
    auto tensor_var = MakeVar("tensor", tensor_type);
    auto out_var = MakeVar("out", tile_type);
    codegen.RegisterPointer("tensor", "raw_ptr");
    auto call = MakeCall("block.load", {out_var, tensor_var, MakeOffsets(0, 0)});
    auto* info = BackendCCE::Instance().GetOpInfo("block.load");
    ASSERT_NE(info, nullptr);
    info->codegen_func(call, codegen);
    auto code = codegen.GetEmittedCode();
    EXPECT_CONTAINS(code, "TASSIGN(tensor, raw_ptr + ");
    EXPECT_CONTAINS(code, "TLOAD(out, tensor);");
    EXPECT_CONTAINS(code, "SetShape");
}

TEST(BackendCCEBlockOutOps, Store)
{
    TestableCCECodegen codegen(ir::SectionKind::Vector);
    auto tensor_type = MakeTensorType();
    auto tile_type = MakeTileType();
    auto tensor_var = MakeVar("tensor", tensor_type);
    auto tile_var = MakeVar("tile", tile_type);
    codegen.RegisterPointer("tensor", "raw_ptr");
    auto call = MakeCall("block.store", {tensor_var, tile_var, MakeOffsets(0, 0)});
    auto* info = BackendCCE::Instance().GetOpInfo("block.store");
    ASSERT_NE(info, nullptr);
    info->codegen_func(call, codegen);
    auto code = codegen.GetEmittedCode();
    EXPECT_CONTAINS(code, "TASSIGN(tensor, raw_ptr + ");
    EXPECT_CONTAINS(code, "TSTORE(tensor, tile);");
}

TEST(BackendCCEBlockOutOps, GeneratesLoadAndStoreThroughFullCodegen)
{
    auto tensor = MakeTensorVar("tensor", {32, 32}, ir::DataType::FP16);
    auto tile_type = MakeTileType({16, 16}, ir::DataType::FP16);
    auto tile = MakeVar("tile", tile_type);
    auto make_tile = std::make_shared<const ir::Call>("block.make_tile", std::vector<ir::ExprPtr>{}, tile_type,
                                                      ir::Span::Unknown());
    auto tile_assign = std::make_shared<const ir::AssignStmt>(tile, make_tile, ir::Span::Unknown());
    auto offsets = MakeOffsets(0, 0);
    auto load = std::make_shared<const ir::Call>("block.load", std::vector<ir::ExprPtr>{tile, tensor, offsets},
                                                 ir::Span::Unknown());
    auto store = std::make_shared<const ir::Call>("block.store", std::vector<ir::ExprPtr>{tensor, tile, offsets},
                                                  ir::Span::Unknown());
    auto body = std::make_shared<const ir::SeqStmts>(
        std::vector<ir::StmtPtr>{tile_assign, std::make_shared<const ir::EvalStmt>(load, ir::Span::Unknown()),
                                 std::make_shared<const ir::EvalStmt>(store, ir::Span::Unknown())},
        ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body, {tensor}), "a5");

    EXPECT_CONTAINS(generated, "TASSIGN(tensor_0, tensor_0_ptr + ");
    EXPECT_CONTAINS(generated, "TLOAD(tile");
    EXPECT_CONTAINS(generated, "TSTORE(tensor_0, tile");
}

TEST(BackendCCEBlockOutOps, StoreWithPhase)
{
    TestableCCECodegen codegen(ir::SectionKind::Vector);
    auto tensor_type = MakeTensorType();
    auto tile_type = MakeTileType();
    codegen.RegisterPointer("tensor", "raw_ptr");
    auto call = MakeCallWithKwargs(
        "block.store", {MakeVar("tensor", tensor_type), MakeVar("tile", tile_type), MakeOffsets(0, 0)}, {{"phase", 1}});
    auto* info = BackendCCE::Instance().GetOpInfo("block.store");
    ASSERT_NE(info, nullptr);
    info->codegen_func(call, codegen);
    auto code = codegen.GetEmittedCode();
    EXPECT_CONTAINS(code, "STPhase::Partial");
    EXPECT_CONTAINS(code, "TSTORE<");
}

TEST(BackendCCEBlockOutOps, StoreWithPreQuant)
{
    TestableCCECodegen codegen(ir::SectionKind::Vector);
    auto tensor_type = MakeTensorType({32, 32}, ir::DataType::FP16);
    auto tile_type = MakeTileType();
    codegen.RegisterPointer("tensor", "raw_ptr");
    auto call = MakeCall("block.store", {MakeVar("tensor", tensor_type), MakeVar("tile", tile_type), MakeOffsets(0, 0),
                                         MakeConstInt(42)});
    auto* info = BackendCCE::Instance().GetOpInfo("block.store");
    ASSERT_NE(info, nullptr);
    info->codegen_func(call, codegen);
    auto code = codegen.GetEmittedCode();
    EXPECT_CONTAINS(code, "static_cast<uint64_t>(42)");
    EXPECT_CONTAINS(code, "TSTORE<");
}

TEST(BackendCCEBlockOutOps, StoreWithPreQuantInt8)
{
    TestableCCECodegen codegen(ir::SectionKind::Vector);
    auto tensor_type = MakeTensorType({32, 32}, ir::DataType::INT8);
    auto tile_type = MakeTileType();
    codegen.RegisterPointer("tensor", "raw_ptr");
    auto call = MakeCall("block.store", {MakeVar("tensor", tensor_type), MakeVar("tile", tile_type), MakeOffsets(0, 0),
                                         MakeConstInt(42)});
    auto* info = BackendCCE::Instance().GetOpInfo("block.store");
    ASSERT_NE(info, nullptr);
    info->codegen_func(call, codegen);
    auto code = codegen.GetEmittedCode();
    EXPECT_CONTAINS(code, "(1ULL << 46)");
    EXPECT_CONTAINS(code, "TSTORE<");
}

TEST(BackendCCEBlockOutOps, StoreWithReluPreMode)
{
    TestableCCECodegen codegen(ir::SectionKind::Vector);
    auto tensor_type = MakeTensorType();
    auto tile_type = MakeTileType();
    codegen.RegisterPointer("tensor", "raw_ptr");
    auto call = MakeCallWithKwargs("block.store",
                                   {MakeVar("tensor", tensor_type), MakeVar("tile", tile_type), MakeOffsets(0, 0)},
                                   {{"relu_pre_mode", 0}});
    auto* info = BackendCCE::Instance().GetOpInfo("block.store");
    ASSERT_NE(info, nullptr);
    info->codegen_func(call, codegen);
    auto code = codegen.GetEmittedCode();
    EXPECT_CONTAINS(code, "ReluPreMode::NormalRelu");
    EXPECT_CONTAINS(code, "AtomicType::AtomicNone");
}

TEST(BackendCCEBlockOutOps, StoreWithAtomic)
{
    for (auto dtype : {ir::DataType::FP32, ir::DataType::FP16, ir::DataType::BF16, ir::DataType::INT32,
                       ir::DataType::INT16, ir::DataType::INT8}) {
        TestableCCECodegen codegen(ir::SectionKind::Vector);
        auto tensor_type = MakeTensorType({32, 32}, dtype);
        auto tile_type = MakeTileType({16, 16}, dtype);
        codegen.RegisterPointer("tensor", "raw_ptr");
        auto call = MakeCallWithKwargs("block.store",
                                       {MakeVar("tensor", tensor_type), MakeVar("tile", tile_type), MakeOffsets(0, 0)},
                                       {{"atomic", 1}});
        auto* info = BackendCCE::Instance().GetOpInfo("block.store");
        ASSERT_NE(info, nullptr);
        info->codegen_func(call, codegen);
        auto code = codegen.GetEmittedCode();
        EXPECT_CONTAINS(code, "set_atomic_add();");
        EXPECT_CONTAINS(code, "set_atomic_none();");
    }
}

TEST(BackendCCEBlockOutOps, StoreFp)
{
    TestableCCECodegen codegen(ir::SectionKind::Vector);
    auto acc_memref = MakeMemRef(ir::MemorySpace::Acc);
    auto scaling_memref = MakeMemRef(ir::MemorySpace::Scaling);
    auto tensor_type = MakeTensorType();
    auto acc_tile = MakeTileType({16, 16}, ir::DataType::FP16, acc_memref);
    auto scaling_tile = MakeTileType({16, 16}, ir::DataType::FP16, scaling_memref);
    codegen.RegisterPointer("tensor", "raw_ptr");
    auto call = MakeCall("block.store_fp", {MakeVar("tensor", tensor_type), MakeVar("tile", acc_tile),
                                            MakeVar("fp", scaling_tile), MakeOffsets(0, 0)});
    auto* info = BackendCCE::Instance().GetOpInfo("block.store_fp");
    ASSERT_NE(info, nullptr);
    info->codegen_func(call, codegen);
    auto code = codegen.GetEmittedCode();
    EXPECT_CONTAINS(code, "TASSIGN(tensor, raw_ptr + ");
    EXPECT_CONTAINS(code, "TSTORE_FP(tensor, tile, fp);");
}

TEST(BackendCCEBlockOutOps, StoreFpWrongSpace)
{
    TestableCCECodegen codegen(ir::SectionKind::Vector);
    auto vec_memref = MakeMemRef(ir::MemorySpace::Vec);
    auto scaling_memref = MakeMemRef(ir::MemorySpace::Scaling);
    auto tensor_type = MakeTensorType();
    auto vec_tile = MakeTileType({16, 16}, ir::DataType::FP16, vec_memref);
    auto scaling_tile = MakeTileType({16, 16}, ir::DataType::FP16, scaling_memref);
    codegen.RegisterPointer("tensor", "raw_ptr");
    auto call = MakeCall("block.store_fp", {MakeVar("tensor", tensor_type), MakeVar("tile", vec_tile),
                                            MakeVar("fp", scaling_tile), MakeOffsets(0, 0)});
    auto* info = BackendCCE::Instance().GetOpInfo("block.store_fp");
    ASSERT_NE(info, nullptr);
    EXPECT_THROW(info->codegen_func(call, codegen), ir::ValueError);
}

TEST(BackendCCEBlockOutOps, Move)
{
    auto tile = MakeTileType();
    auto call = MakeCall("block.move", {MakeVar("dst", tile), MakeVar("src", tile)});
    auto code = RunCodegen("block.move", call);
    EXPECT_CONTAINS(code, "TMOV<");
    EXPECT_CONTAINS(code, "decltype(dst)");
    EXPECT_CONTAINS(code, "decltype(src)");
    EXPECT_CONTAINS(code, "(dst, src);");
}

TEST(BackendCCEBlockOutOps, MoveWithOffset)
{
    auto tile = MakeTileType();
    auto call = MakeCall("block.move", {MakeVar("dst", tile), MakeVar("src", tile), MakeOffsets(2, 4)});
    auto code = RunCodegen("block.move", call);
    EXPECT_CONTAINS(code, "TEXTRACT<");
    EXPECT_CONTAINS(code, "(dst, src, 2, 4);");
}

TEST(BackendCCEBlockOutOps, MoveWithAccToVecMode)
{
    auto tile = MakeTileType();
    auto call = MakeCallWithKwargs("block.move", {MakeVar("dst", tile), MakeVar("src", tile)},
                                   {{"acc_to_vec_mode", 0}});
    auto code = RunCodegen("block.move", call);
    EXPECT_CONTAINS(code, "AccToVecMode::SingleModeVec0");
}

TEST(BackendCCEBlockOutOps, MoveWithDualModeSplitMAlignsValidShape)
{
    auto tile = MakeTileType();
    auto call = MakeCallWithKwargs("block.move", {MakeVar("dst", tile), MakeVar("src", tile)},
                                   {{"acc_to_vec_mode", static_cast<int>(ir::AccToVecMode::DualModeSplitM)}});
    auto code = RunCodegen("block.move", call);
    EXPECT_CONTAINS(code, "AccToVecMode::DualModeSplitM");
    EXPECT_CONTAINS(code, "src.SetValidShape((src.GetValidRow() + 1) / 2 * 2, src.GetValidCol());");
}

TEST(BackendCCEBlockOutOps, MoveWithDualModeSplitNAlignsValidShape)
{
    auto tile = MakeTileType();
    auto call = MakeCallWithKwargs("block.move", {MakeVar("dst", tile), MakeVar("src", tile)},
                                   {{"acc_to_vec_mode", static_cast<int>(ir::AccToVecMode::DualModeSplitN)}});
    auto code = RunCodegen("block.move", call);
    EXPECT_CONTAINS(code, "AccToVecMode::DualModeSplitN");
    EXPECT_CONTAINS(code, "src.SetValidShape(src.GetValidRow(), (src.GetValidCol() + 31) / 32 * 32);");
}

TEST(BackendCCEBlockOutOps, MoveWithSingleModeVec1NoAlignment)
{
    auto tile = MakeTileType();
    auto call = MakeCallWithKwargs("block.move", {MakeVar("dst", tile), MakeVar("src", tile)},
                                   {{"acc_to_vec_mode", static_cast<int>(ir::AccToVecMode::SingleModeVec1)}});
    auto code = RunCodegen("block.move", call);
    EXPECT_CONTAINS(code, "AccToVecMode::SingleModeVec1");
    EXPECT_NOT_CONTAINS(code, "SetValidShape");
}

TEST(BackendCCEBlockOutOps, MoveWithReluPreMode)
{
    auto tile = MakeTileType();
    auto call = MakeCallWithKwargs("block.move", {MakeVar("dst", tile), MakeVar("src", tile)}, {{"relu_pre_mode", 0}});
    auto code = RunCodegen("block.move", call);
    EXPECT_CONTAINS(code, "ReluPreMode::NormalRelu");
}

TEST(BackendCCEBlockOutOps, MoveWithPreQuant)
{
    auto tile = MakeTileType();
    auto call = MakeCall("block.move", {MakeVar("dst", tile), MakeVar("src", tile), MakeConstInt(99)});
    auto code = RunCodegen("block.move", call);
    EXPECT_CONTAINS(code, "static_cast<uint64_t>(99)");
}

TEST(BackendCCEBlockOutOps, MoveWithPreQuantInt8)
{
    auto int8_tile = MakeTileType({16, 16}, ir::DataType::INT8);
    auto tile = MakeTileType();
    auto call = MakeCall("block.move", {MakeVar("dst", int8_tile), MakeVar("src", tile), MakeConstInt(99)});
    auto code = RunCodegen("block.move", call);
    EXPECT_CONTAINS(code, "(1ULL << 46)");
}

TEST(BackendCCEBlockOutOps, MoveFp)
{
    auto vec_memref = MakeMemRef(ir::MemorySpace::Vec);
    auto acc_memref = MakeMemRef(ir::MemorySpace::Acc);
    auto scaling_memref = MakeMemRef(ir::MemorySpace::Scaling);
    auto vec_tile = MakeTileType({16, 16}, ir::DataType::FP16, vec_memref);
    auto acc_tile = MakeTileType({16, 16}, ir::DataType::FP16, acc_memref);
    auto scaling_tile = MakeTileType({16, 16}, ir::DataType::FP16, scaling_memref);
    auto call = MakeCall("block.move_fp",
                         {MakeVar("dst", vec_tile), MakeVar("src", acc_tile), MakeVar("fp", scaling_tile)});
    auto code = RunCodegen("block.move_fp", call);
    EXPECT_CONTAINS(code, "TMOV_FP(dst, src, fp);");
}

TEST(BackendCCEBlockOutOps, MoveFpWithAccToVecMode)
{
    auto vec_memref = MakeMemRef(ir::MemorySpace::Vec);
    auto acc_memref = MakeMemRef(ir::MemorySpace::Acc);
    auto scaling_memref = MakeMemRef(ir::MemorySpace::Scaling);
    auto vec_tile = MakeTileType({16, 16}, ir::DataType::FP16, vec_memref);
    auto acc_tile = MakeTileType({16, 16}, ir::DataType::FP16, acc_memref);
    auto scaling_tile = MakeTileType({16, 16}, ir::DataType::FP16, scaling_memref);
    auto call = MakeCallWithKwargs("block.move_fp",
                                   {MakeVar("dst", vec_tile), MakeVar("src", acc_tile), MakeVar("fp", scaling_tile)},
                                   {{"acc_to_vec_mode", 0}});
    auto code = RunCodegen("block.move_fp", call);
    EXPECT_CONTAINS(code, "TMOV<");
    EXPECT_CONTAINS(code, "AccToVecMode::SingleModeVec0");
}

TEST(BackendCCEBlockOutOps, MoveFpWithReluPreMode)
{
    auto vec_memref = MakeMemRef(ir::MemorySpace::Vec);
    auto acc_memref = MakeMemRef(ir::MemorySpace::Acc);
    auto scaling_memref = MakeMemRef(ir::MemorySpace::Scaling);
    auto vec_tile = MakeTileType({16, 16}, ir::DataType::FP16, vec_memref);
    auto acc_tile = MakeTileType({16, 16}, ir::DataType::FP16, acc_memref);
    auto scaling_tile = MakeTileType({16, 16}, ir::DataType::FP16, scaling_memref);
    auto call = MakeCallWithKwargs("block.move_fp",
                                   {MakeVar("dst", vec_tile), MakeVar("src", acc_tile), MakeVar("fp", scaling_tile)},
                                   {{"relu_pre_mode", 0}});
    auto code = RunCodegen("block.move_fp", call);
    EXPECT_CONTAINS(code, "TMOV_FP<");
    EXPECT_CONTAINS(code, "ReluPreMode::NormalRelu");
}

TEST(BackendCCEBlockOutOps, MoveFpDualModeThrows)
{
    auto vec_memref = MakeMemRef(ir::MemorySpace::Vec);
    auto acc_memref = MakeMemRef(ir::MemorySpace::Acc);
    auto scaling_memref = MakeMemRef(ir::MemorySpace::Scaling);
    auto vec_tile = MakeTileType({16, 16}, ir::DataType::FP16, vec_memref);
    auto acc_tile = MakeTileType({16, 16}, ir::DataType::FP16, acc_memref);
    auto scaling_tile = MakeTileType({16, 16}, ir::DataType::FP16, scaling_memref);
    auto call = MakeCallWithKwargs("block.move_fp",
                                   {MakeVar("dst", vec_tile), MakeVar("src", acc_tile), MakeVar("fp", scaling_tile)},
                                   {{"acc_to_vec_mode", static_cast<int>(ir::AccToVecMode::DualModeSplitM)}});
    EXPECT_THROW(RunCodegen("block.move_fp", call), ir::ValueError);
}

TEST(BackendCCEBlockOutOps, MoveFpWithSingleModeVec1)
{
    auto vec_memref = MakeMemRef(ir::MemorySpace::Vec);
    auto acc_memref = MakeMemRef(ir::MemorySpace::Acc);
    auto scaling_memref = MakeMemRef(ir::MemorySpace::Scaling);
    auto vec_tile = MakeTileType({16, 16}, ir::DataType::FP16, vec_memref);
    auto acc_tile = MakeTileType({16, 16}, ir::DataType::FP16, acc_memref);
    auto scaling_tile = MakeTileType({16, 16}, ir::DataType::FP16, scaling_memref);
    auto call = MakeCallWithKwargs("block.move_fp",
                                   {MakeVar("dst", vec_tile), MakeVar("src", acc_tile), MakeVar("fp", scaling_tile)},
                                   {{"acc_to_vec_mode", static_cast<int>(ir::AccToVecMode::SingleModeVec1)}});
    auto code = RunCodegen("block.move_fp", call);
    EXPECT_CONTAINS(code, "TMOV<");
    EXPECT_CONTAINS(code, "AccToVecMode::SingleModeVec1");
}

TEST(BackendCCEBlockOutOps, MoveFpWrongSpace)
{
    auto vec_memref = MakeMemRef(ir::MemorySpace::Vec);
    auto vec_tile = MakeTileType({16, 16}, ir::DataType::FP16, vec_memref);
    auto scaling_memref = MakeMemRef(ir::MemorySpace::Scaling);
    auto scaling_tile = MakeTileType({16, 16}, ir::DataType::FP16, scaling_memref);
    auto call = MakeCall("block.move_fp",
                         {MakeVar("dst", vec_tile), MakeVar("src", vec_tile), MakeVar("fp", scaling_tile)});
    EXPECT_THROW(RunCodegen("block.move_fp", call), ir::ValueError);
}

TEST(BackendCCEBlockOutOps, MoveWithPhase)
{
    auto tile = MakeTileType();
    auto call = MakeCallWithKwargs("block.move", {MakeVar("dst", tile), MakeVar("src", tile)}, {{"phase", 1}});
    auto code = RunCodegen("block.move", call);
    EXPECT_CONTAINS(code, "STPhase::Partial");
    EXPECT_CONTAINS(code, "TMOV<");
}

TEST(BackendCCEBlockOutOps, MoveFpWithPhase)
{
    auto vec_memref = MakeMemRef(ir::MemorySpace::Vec);
    auto acc_memref = MakeMemRef(ir::MemorySpace::Acc);
    auto scaling_memref = MakeMemRef(ir::MemorySpace::Scaling);
    auto vec_tile = MakeTileType({16, 16}, ir::DataType::FP16, vec_memref);
    auto acc_tile = MakeTileType({16, 16}, ir::DataType::FP16, acc_memref);
    auto scaling_tile = MakeTileType({16, 16}, ir::DataType::FP16, scaling_memref);
    auto call = MakeCallWithKwargs("block.move_fp",
                                   {MakeVar("dst", vec_tile), MakeVar("src", acc_tile), MakeVar("fp", scaling_tile)},
                                   {{"phase", 2}});
    auto code = RunCodegen("block.move_fp", call);
    EXPECT_CONTAINS(code, "STPhase::Final");
}

TEST(BackendCCEBlockOutOps, MoveWithPhaseAndAccToVecMode)
{
    auto tile = MakeTileType();
    auto call = MakeCallWithKwargs("block.move", {MakeVar("dst", tile), MakeVar("src", tile)},
                                   {{"phase", 1}, {"acc_to_vec_mode", 0}});
    auto code = RunCodegen("block.move", call);
    EXPECT_CONTAINS(code, "STPhase::Partial");
    EXPECT_CONTAINS(code, "AccToVecMode::SingleModeVec0");
}

TEST(BackendCCEBlockOutOps, UbCopy)
{
    auto tile = MakeTileType();
    auto call = MakeCall("block.ub_copy", {MakeVar("dst", tile), MakeVar("src", tile)});
    auto code = RunCodegen("block.ub_copy", call);
    EXPECT_CONTAINS(code, "TMOV<");
}

TEST(BackendCCEBlockOutOps, SsbufStore)
{
    auto code = RunSsbufCodegen("block.ssbuf_store");
    EXPECT_CONTAINS(code, "class SsbufStruct {\npublic:\n    volatile int32_t value;");
    EXPECT_CONTAINS(code, "reinterpret_cast<__ssbuf__ uint32_t*>((uint64_t)(0))");
    EXPECT_CONTAINS(code, "reinterpret_cast<const uint32_t*>(&struct_var_0)");
    EXPECT_CONTAINS(code, "sizeof(struct_var_0) / sizeof(uint32_t)");
}

TEST(BackendCCEBlockOutOps, SsbufLoad)
{
    auto code = RunSsbufCodegen("block.ssbuf_load");
    EXPECT_CONTAINS(code, "class SsbufStruct {\npublic:\n    volatile int32_t value;");
    EXPECT_CONTAINS(code, "reinterpret_cast<__ssbuf__ uint32_t*>((uint64_t)(0))");
    EXPECT_CONTAINS(code, "reinterpret_cast<uint32_t*>(&struct_var_0)");
    EXPECT_CONTAINS(code, "sizeof(struct_var_0) / sizeof(uint32_t)");
}

TEST(BackendCCEBlockOutOps, Fillpad)
{
    auto tile = MakeTileType();
    auto call = MakeCall("block.fillpad", {MakeVar("dst", tile), MakeVar("src", tile)});
    EXPECT_CONTAINS(RunCodegen("block.fillpad", call), "TFILLPAD(dst, src);");
}

TEST(BackendCCEBlockOutOps, FillpadWithPadAlias)
{
    ir::HardwareInfo hw;
    hw.pad = ir::TilePad::zero;
    auto tile_with_pad = MakeTileType({16, 16}, ir::DataType::FP16, std::nullopt, hw);
    auto tile_no_pad = MakeTileType();
    auto call = MakeCall("block.fillpad", {MakeVar("dst", tile_no_pad), MakeVar("src", tile_with_pad)});
    auto code = RunCodegen("block.fillpad", call);
    EXPECT_CONTAINS(code, "using __block_fillpad_src_alias_type_");
    EXPECT_CONTAINS(code, "fillpad_src_alias");
    EXPECT_CONTAINS(code, "TASSIGN(__block_fillpad_src_alias_");
    EXPECT_CONTAINS(code, "TFILLPAD(dst, __block_fillpad_src_alias_");
}

TEST(BackendCCEBlockOutOps, FillpadInplace)
{
    auto tile = MakeTileType();
    auto call = MakeCall("block.fillpad_inplace", {MakeVar("dst", tile), MakeVar("src", tile)});
    EXPECT_CONTAINS(RunCodegen("block.fillpad_inplace", call), "TFILLPAD_INPLACE(dst, src);");
}

TEST(BackendCCEBlockOutOps, FillpadExpand)
{
    auto tile = MakeTileType();
    auto call = MakeCall("block.fillpad_expand", {MakeVar("dst", tile), MakeVar("src", tile)});
    EXPECT_CONTAINS(RunCodegen("block.fillpad_expand", call), "TFILLPAD_EXPAND(dst, src);");
}

TEST(BackendCCEBlockOutOps, FillpadExpandSameVarThrows)
{
    auto tile = MakeTileType();
    auto same_var = MakeVar("same", tile);
    auto call = MakeCall("block.fillpad_expand", {same_var, same_var});
    EXPECT_THROW(RunCodegen("block.fillpad_expand", call), ir::ValueError);
}

TEST(BackendCCEBlockOutOps, Gather3Args)
{
    auto tile = MakeTileType();
    auto call = MakeCall("block.gather", {MakeVar("out", tile), MakeVar("src", tile), MakeVar("idx", tile)});
    EXPECT_CONTAINS(RunCodegen("block.gather", call), "TGATHER(out, src, idx);");
}

TEST(BackendCCEBlockOutOps, Gather4Args)
{
    auto tile = MakeTileType();
    auto call = MakeCall("block.gather",
                         {MakeVar("out", tile), MakeVar("src", tile), MakeVar("idx", tile), MakeVar("tmp", tile)});
    EXPECT_CONTAINS(RunCodegen("block.gather", call), "TGATHER(out, src, idx, tmp);");
}

TEST(BackendCCEBlockOutOps, Gather5ArgsCmp)
{
    auto tile = MakeTileType();
    auto call = MakeCallWithKwargs(
        "block.gather",
        {MakeVar("out", tile), MakeVar("src", tile), MakeVar("k", tile), MakeVar("cdst", tile), MakeVar("tmp", tile)},
        {{"cmp_mode", 4}, {"offset", 0}});
    auto code = RunCodegen("block.gather", call);
    EXPECT_CONTAINS(code, "TGATHER<");
    EXPECT_CONTAINS(code, "CmpMode::GT");
}

TEST(BackendCCEBlockOutOps, Gather5ArgsCmpEq)
{
    auto tile = MakeTileType();
    auto call = MakeCallWithKwargs(
        "block.gather",
        {MakeVar("out", tile), MakeVar("src", tile), MakeVar("k", tile), MakeVar("cdst", tile), MakeVar("tmp", tile)},
        {{"cmp_mode", 0}, {"offset", 0}});
    auto code = RunCodegen("block.gather", call);
    EXPECT_CONTAINS(code, "CmpMode::EQ");
}

TEST(BackendCCEBlockOutOps, GatherMask)
{
    auto tile = MakeTileType();
    const std::vector<std::pair<int, std::string>> patterns = {
        {1, "P0101"}, {2, "P1010"}, {3, "P0001"}, {4, "P0010"}, {5, "P0100"}, {6, "P1000"}, {7, "P1111"},
    };
    for (const auto& [mode, name] : patterns) {
        auto call = MakeCallWithKwargs("block.gathermask", {MakeVar("out", tile), MakeVar("src", tile)},
                                       {{"pattern_mode", mode}});
        auto code = RunCodegen("block.gathermask", call);
        EXPECT_CONTAINS(code, "MaskPattern::" + name);
        EXPECT_CONTAINS(code, "TGATHER<");
    }
}

TEST(BackendCCEBlockOutOps, Sort32)
{
    auto tile = MakeTileType();
    auto call3 = MakeCall("block.sort32", {MakeVar("dst", tile), MakeVar("src", tile), MakeVar("idx", tile)});
    EXPECT_CONTAINS(RunCodegen("block.sort32", call3), "TSORT32(dst, src, idx);");

    auto call4 = MakeCall("block.sort32",
                          {MakeVar("dst", tile), MakeVar("src", tile), MakeVar("idx", tile), MakeVar("tmp", tile)});
    EXPECT_CONTAINS(RunCodegen("block.sort32", call4), "TSORT32(dst, src, idx, tmp);");
}

TEST(BackendCCEBlockOutOps, Mrgsort)
{
    auto tile = MakeTileType();
    auto call = MakeCallWithKwargs("block.mrgsort", {MakeVar("dst", tile), MakeVar("src", tile)}, {{"block_len", 16}});
    EXPECT_CONTAINS(RunCodegen("block.mrgsort", call), "TMRGSORT(dst, src, 16);");
}

TEST(BackendCCEBlockOutOps, Mrgsort2)
{
    auto tile = MakeTileType();
    auto call4 = MakeCallWithKwargs(
        "block.mrgsort2", {MakeVar("dst", tile), MakeVar("src0", tile), MakeVar("tmp", tile), MakeVar("src1", tile)},
        {{"exhausted", false}});
    auto code4 = RunCodegen("block.mrgsort2", call4);
    EXPECT_CONTAINS(code4, "MrgSortExecutedNumList");
    EXPECT_CONTAINS(code4, "TMRGSORT_IMPL<");
    EXPECT_CONTAINS(code4, "false");

    auto call5 = MakeCallWithKwargs("block.mrgsort2",
                                    {MakeVar("dst", tile), MakeVar("src0", tile), MakeVar("tmp", tile),
                                     MakeVar("src1", tile), MakeVar("src2", tile)},
                                    {{"exhausted", true}});
    auto code5 = RunCodegen("block.mrgsort2", call5);
    EXPECT_CONTAINS(code5, "true");
    EXPECT_CONTAINS(code5, "decltype(src2)");

    auto call6 = MakeCallWithKwargs("block.mrgsort2",
                                    {MakeVar("dst", tile), MakeVar("src0", tile), MakeVar("tmp", tile),
                                     MakeVar("src1", tile), MakeVar("src2", tile), MakeVar("src3", tile)},
                                    {{"exhausted", false}});
    auto code6 = RunCodegen("block.mrgsort2", call6);
    EXPECT_CONTAINS(code6, "decltype(src3)");
}

TEST(BackendCCEBlockOutOps, Histogram)
{
    auto u16_tile = MakeTileType({16, 16}, ir::DataType::UINT16);
    auto u32_tile = MakeTileType({16, 16}, ir::DataType::UINT32);

    auto callU16Msb = MakeCallWithKwargs("block.histogram",
                                         {MakeVar("dst", u16_tile), MakeVar("src", u16_tile), MakeVar("idx", u16_tile)},
                                         {{"is_msb", true}});
    EXPECT_CONTAINS(RunCodegen("block.histogram", callU16Msb), "HistByte::BYTE_1");

    auto callU16Lsb = MakeCallWithKwargs("block.histogram",
                                         {MakeVar("dst", u16_tile), MakeVar("src", u16_tile), MakeVar("idx", u16_tile)},
                                         {{"is_msb", false}});
    EXPECT_CONTAINS(RunCodegen("block.histogram", callU16Lsb), "HistByte::BYTE_0");

    auto callU32Msb = MakeCallWithKwargs("block.histogram",
                                         {MakeVar("dst", u32_tile), MakeVar("src", u32_tile), MakeVar("idx", u32_tile)},
                                         {{"is_msb", true}});
    EXPECT_CONTAINS(RunCodegen("block.histogram", callU32Msb), "HistByte::BYTE_3");

    auto callU32Lsb = MakeCallWithKwargs("block.histogram",
                                         {MakeVar("dst", u32_tile), MakeVar("src", u32_tile), MakeVar("idx", u32_tile)},
                                         {{"is_msb", false}});
    EXPECT_CONTAINS(RunCodegen("block.histogram", callU32Lsb), "HistByte::BYTE_0");
}

TEST(BackendCCEBlockOutOps, RowReduce)
{
    auto tile = MakeTileType();
    const std::vector<std::pair<int, std::string>> ops = {
        {0, "TROWSUM"},
        {1, "TROWMAX"},
        {2, "TROWMIN"},
        {3, "TROWPROD"},
    };
    for (const auto& [op_type, op_name] : ops) {
        auto call = MakeCallWithKwargs("block.row_reduce",
                                       {MakeVar("dst", tile), MakeVar("tile", tile), MakeVar("tmp", tile)},
                                       {{"op_type", op_type}});
        EXPECT_CONTAINS(RunCodegen("block.row_reduce", call), op_name + "(dst, tile, tmp);");
    }
}

TEST(BackendCCEBlockOutOps, ColReduce)
{
    auto tile = MakeTileType();
    auto call0 = MakeCallWithKwargs(
        "block.col_reduce", {MakeVar("dst", tile), MakeVar("tile", tile), MakeVar("tmp", tile)}, {{"op_type", 0}});
    EXPECT_CONTAINS(RunCodegen("block.col_reduce", call0), "TCOLSUM(dst, tile, tmp, false);");

    auto call1 = MakeCallWithKwargs(
        "block.col_reduce", {MakeVar("dst", tile), MakeVar("tile", tile), MakeVar("tmp", tile)}, {{"op_type", 1}});
    EXPECT_CONTAINS(RunCodegen("block.col_reduce", call1), "TCOLMAX(dst, tile);");

    auto call2 = MakeCallWithKwargs(
        "block.col_reduce", {MakeVar("dst", tile), MakeVar("tile", tile), MakeVar("tmp", tile)}, {{"op_type", 2}});
    EXPECT_CONTAINS(RunCodegen("block.col_reduce", call2), "TCOLMIN(dst, tile);");

    auto call3 = MakeCallWithKwargs(
        "block.col_reduce", {MakeVar("dst", tile), MakeVar("tile", tile), MakeVar("tmp", tile)}, {{"op_type", 3}});
    EXPECT_CONTAINS(RunCodegen("block.col_reduce", call3), "TCOLPROD(dst, tile);");
}

TEST(BackendCCEBlockOutOps, Quant)
{
    auto tile = MakeTileType();
    auto callSym = MakeCallWithKwargs(
        "block.quant", {MakeVar("dst", tile), MakeVar("src", tile), MakeVar("scale", tile)}, {{"mode", 0}});
    EXPECT_CONTAINS(RunCodegen("block.quant", callSym), "TQUANT<QuantType::INT8_SYM>(dst, src, scale);");

    auto callAsym = MakeCallWithKwargs(
        "block.quant", {MakeVar("dst", tile), MakeVar("src", tile), MakeVar("scale", tile), MakeVar("offset", tile)},
        {{"mode", 1}});
    EXPECT_CONTAINS(RunCodegen("block.quant", callAsym), "TQUANT<QuantType::INT8_ASYM>(dst, src, scale, offset);");
}

TEST(BackendCCEBlockOutOps, Dequant)
{
    auto tile = MakeTileType();
    auto call = MakeCall("block.dequant",
                         {MakeVar("dst", tile), MakeVar("src", tile), MakeVar("scale", tile), MakeVar("offset", tile)});
    EXPECT_CONTAINS(RunCodegen("block.dequant", call), "TDEQUANT(dst, src, scale, offset);");
}

TEST(BackendCCEBlockOutOps, StructCreate)
{
    TestableCCECodegen codegen(ir::SectionKind::Vector);
    codegen.SetCurrentTargetVar("result");
    auto tile = MakeTileType();
    auto tuple_type = std::make_shared<const ir::TupleType>(std::vector<ir::TypePtr>{tile, tile});
    auto call = std::make_shared<const ir::Call>("struct.create",
                                                 std::vector<ir::ExprPtr>{MakeVar("v0", tile), MakeVar("v1", tile)},
                                                 std::vector<std::pair<std::string, std::any>>{
                                                     {"name", std::string("MyStruct")},
                                                     {"fields", std::vector<std::string>{"f0", "f1"}},
                                                 },
                                                 tuple_type, ir::Span::Unknown());
    auto* info = BackendCCE::Instance().GetOpInfo("struct.create");
    ASSERT_NE(info, nullptr);
    info->codegen_func(call, codegen);
    auto code = codegen.GetEmittedCode();
    EXPECT_CONTAINS(code, "MyStruct result = {");
    EXPECT_CONTAINS(code, ".f0=v0");
    EXPECT_CONTAINS(code, ".f1=v1");
    EXPECT_CONTAINS(code, "};");
}

TEST(BackendCCEBlockOutOps, StructCreateCastsScalarInitializerToFieldType)
{
    TestableCCECodegen codegen(ir::SectionKind::Cube);
    codegen.SetCurrentTargetVar("run_info");
    auto index = MakeScalarType(ir::DataType::INDEX);
    auto tuple_type = std::make_shared<const ir::TupleType>(std::vector<ir::TypePtr>{index});
    auto call = std::make_shared<const ir::Call>("struct.create", std::vector<ir::ExprPtr>{MakeVar("work_id", index)},
                                                 std::vector<std::pair<std::string, std::any>>{
                                                     {"name", std::string("RunInfo")},
                                                     {"fields", std::vector<std::string>{"workId"}},
                                                 },
                                                 tuple_type, ir::Span::Unknown());

    auto* info = BackendCCE::Instance().GetOpInfo("struct.create");
    ASSERT_NE(info, nullptr);
    info->codegen_func(call, codegen);

    EXPECT_CONTAINS(codegen.GetEmittedCode(), ".workId=static_cast<int64_t>(work_id)");
}

TEST(BackendCCEBlockOutOps, GeneratesStructCreateThroughAssignStmt)
{
    auto tile = MakeTileType();
    auto v0 = MakeVar("v0", tile);
    auto v1 = MakeVar("v1", tile);
    auto tuple_type = std::make_shared<const ir::TupleType>(std::vector<ir::TypePtr>{tile, tile});
    auto result = MakeVar("result", tuple_type);
    auto create = std::make_shared<const ir::Call>("struct.create", std::vector<ir::ExprPtr>{v0, v1},
                                                   std::vector<std::pair<std::string, std::any>>{
                                                       {"name", std::string("MyStruct")},
                                                       {"fields", std::vector<std::string>{"f0", "f1"}},
                                                   },
                                                   tuple_type, ir::Span::Unknown());
    auto body = std::make_shared<const ir::AssignStmt>(result, create, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a5");

    EXPECT_CONTAINS(generated, "class MyStruct");
    EXPECT_CONTAINS(generated, "MyStruct result_0 = {");
    EXPECT_CONTAINS(generated, ".f0=v0");
    EXPECT_CONTAINS(generated, ".f1=v1");
}

TEST(BackendCCEBlockOutOps, StructSet)
{
    auto tile = MakeTileType();
    auto call = MakeCallWithKwargs("struct.set", {MakeVar("base", tile), MakeVar("value", tile)},
                                   {{"field", std::string("my_field")}});
    auto code = RunCodegen("struct.set", call);
    EXPECT_CONTAINS(code, "base.my_field = value;");
}

TEST(BackendCCEBlockOutOps, StructSet_WithIndex)
{
    auto scalar_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT64);
    auto call = MakeCallWithKwargs(
        "struct.set", {MakeVar("base", scalar_type), MakeVar("idx", scalar_type), MakeVar("value", scalar_type)},
        {{"field", std::string("arr")}});
    auto code = RunCodegen("struct.set", call);
    EXPECT_CONTAINS(code, "base.arr[idx] = value;");
}

TEST(BackendCCEBlockOutOps, StructCreate_WithArrayField)
{
    TestableCCECodegen codegen(ir::SectionKind::Vector);
    codegen.SetCurrentTargetVar("result");
    auto scalar_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT64);
    auto array_type = std::make_shared<const ir::TupleType>(
        std::vector<ir::TypePtr>{scalar_type, scalar_type, scalar_type, scalar_type});
    auto tuple_type = std::make_shared<const ir::TupleType>(std::vector<ir::TypePtr>{scalar_type, array_type});
    auto arr_init = std::make_shared<const ir::MakeTuple>(
        std::vector<ir::ExprPtr>{MakeConstInt(0), MakeConstInt(0), MakeConstInt(0), MakeConstInt(0)},
        ir::Span::Unknown());
    auto call = std::make_shared<const ir::Call>("struct.create", std::vector<ir::ExprPtr>{MakeConstInt(42), arr_init},
                                                 std::vector<std::pair<std::string, std::any>>{
                                                     {"name", std::string("MyStruct")},
                                                     {"fields", std::vector<std::string>{"x", "arr"}},
                                                 },
                                                 tuple_type, ir::Span::Unknown());
    auto* info = BackendCCE::Instance().GetOpInfo("struct.create");
    ASSERT_NE(info, nullptr);
    info->codegen_func(call, codegen);
    auto code = codegen.GetEmittedCode();
    EXPECT_CONTAINS(code, "MyStruct result = {");
    EXPECT_CONTAINS(code, ".x=static_cast<int64_t>(42)");
    EXPECT_CONTAINS(code, ".arr={static_cast<int64_t>(0), static_cast<int64_t>(0), static_cast<int64_t>(0), "
                          "static_cast<int64_t>(0)}");
    EXPECT_CONTAINS(code, "};");
}

TEST(BackendCCEBlockOutOps, StructCreate_MultipleArrayFields)
{
    TestableCCECodegen codegen(ir::SectionKind::Vector);
    codegen.SetCurrentTargetVar("result");
    auto scalar_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT64);
    auto array_type_2 = std::make_shared<const ir::TupleType>(std::vector<ir::TypePtr>{scalar_type, scalar_type});
    auto array_type_4 = std::make_shared<const ir::TupleType>(
        std::vector<ir::TypePtr>{scalar_type, scalar_type, scalar_type, scalar_type});
    auto tuple_type = std::make_shared<const ir::TupleType>(std::vector<ir::TypePtr>{array_type_2, array_type_4});
    auto a_init = std::make_shared<const ir::MakeTuple>(std::vector<ir::ExprPtr>{MakeConstInt(1), MakeConstInt(2)},
                                                        ir::Span::Unknown());
    auto b_init = std::make_shared<const ir::MakeTuple>(
        std::vector<ir::ExprPtr>{MakeConstInt(10), MakeConstInt(20), MakeConstInt(30), MakeConstInt(40)},
        ir::Span::Unknown());
    auto call = std::make_shared<const ir::Call>("struct.create", std::vector<ir::ExprPtr>{a_init, b_init},
                                                 std::vector<std::pair<std::string, std::any>>{
                                                     {"name", std::string("DualArr")},
                                                     {"fields", std::vector<std::string>{"a", "b"}},
                                                 },
                                                 tuple_type, ir::Span::Unknown());
    auto* info = BackendCCE::Instance().GetOpInfo("struct.create");
    ASSERT_NE(info, nullptr);
    info->codegen_func(call, codegen);
    auto code = codegen.GetEmittedCode();
    EXPECT_CONTAINS(code, "DualArr result = {");
    EXPECT_CONTAINS(code, ".a={static_cast<int64_t>(1), static_cast<int64_t>(2)}");
    EXPECT_CONTAINS(code, ".b={static_cast<int64_t>(10), static_cast<int64_t>(20), static_cast<int64_t>(30), "
                          "static_cast<int64_t>(40)}");
    EXPECT_CONTAINS(code, "};");
}

TEST(BackendCCEBlockOutOps, StructCreate_MixedScalarArray)
{
    TestableCCECodegen codegen(ir::SectionKind::Vector);
    codegen.SetCurrentTargetVar("result");
    auto scalar_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT64);
    auto array_type = std::make_shared<const ir::TupleType>(
        std::vector<ir::TypePtr>{scalar_type, scalar_type, scalar_type, scalar_type});
    auto tuple_type = std::make_shared<const ir::TupleType>(
        std::vector<ir::TypePtr>{scalar_type, array_type, scalar_type});
    auto arr_init = std::make_shared<const ir::MakeTuple>(
        std::vector<ir::ExprPtr>{MakeConstInt(0), MakeConstInt(0), MakeConstInt(0), MakeConstInt(0)},
        ir::Span::Unknown());
    auto call = std::make_shared<const ir::Call>("struct.create",
                                                 std::vector<ir::ExprPtr>{MakeConstInt(42), arr_init, MakeConstInt(99)},
                                                 std::vector<std::pair<std::string, std::any>>{
                                                     {"name", std::string("Mixed")},
                                                     {"fields", std::vector<std::string>{"x", "arr", "y"}},
                                                 },
                                                 tuple_type, ir::Span::Unknown());
    auto* info = BackendCCE::Instance().GetOpInfo("struct.create");
    ASSERT_NE(info, nullptr);
    info->codegen_func(call, codegen);
    auto code = codegen.GetEmittedCode();
    EXPECT_CONTAINS(code, "Mixed result = {");
    EXPECT_CONTAINS(code, ".x=static_cast<int64_t>(42)");
    EXPECT_CONTAINS(code, ".arr={static_cast<int64_t>(0), static_cast<int64_t>(0), static_cast<int64_t>(0), "
                          "static_cast<int64_t>(0)}");
    EXPECT_CONTAINS(code, ".y=static_cast<int64_t>(99)");
    EXPECT_CONTAINS(code, "};");
}

TEST(BackendCCEBlockOutOps, StructSet_WithConstIndex)
{
    auto scalar_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT64);
    auto call = MakeCallWithKwargs("struct.set",
                                   {MakeVar("base", scalar_type), MakeConstInt(2), MakeVar("value", scalar_type)},
                                   {{"field", std::string("arr")}});
    auto code = RunCodegen("struct.set", call);
    EXPECT_CONTAINS(code, "base.arr[2] = value;");
}

} // namespace
} // namespace backend
} // namespace pypto
