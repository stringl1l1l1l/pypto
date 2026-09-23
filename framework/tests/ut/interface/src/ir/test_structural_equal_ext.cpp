/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_structural_equal_ext.cpp
 * \brief Coverage tests for structural_equal.cpp — exercising all EQUAL_DISPATCH paths
 */

#include "gtest/gtest.h"

#include "ir/stmt.h"
#include "ir/transforms/structural_comparison.h"
#include "op/test_op_helpers.h"
#include "tilefwk/error.h"

namespace pypto {
namespace ir {

using test_helpers::Node;
using test_helpers::Scalar;
using test_helpers::Sp;

// ============================================================================
// Expression equality
// ============================================================================

class IRStructEqExprTest : public testing::Test {};

TEST_F(IRStructEqExprTest, TestConstAndLeafExprsEqual)
{
    // ConstFloat
    auto f1 = std::make_shared<ConstFloat>(3.14, DataType::FP32, Sp());
    auto f2 = std::make_shared<ConstFloat>(3.14, DataType::FP32, Sp());
    EXPECT_TRUE(structural_equal(Node(f1), Node(f2)));
    auto f3 = std::make_shared<ConstFloat>(2.71, DataType::FP32, Sp());
    EXPECT_FALSE(structural_equal(Node(f1), Node(f3)));

    // ConstBool
    auto b1 = std::make_shared<ConstBool>(true, Sp());
    auto b2 = std::make_shared<ConstBool>(true, Sp());
    EXPECT_TRUE(structural_equal(Node(b1), Node(b2)));
    EXPECT_FALSE(structural_equal(Node(b1), Node(std::make_shared<ConstBool>(false, Sp()))));

    // MemRef
    auto off1 = std::make_shared<ConstInt>(0, DataType::INT64, Sp());
    auto off2 = std::make_shared<ConstInt>(0, DataType::INT64, Sp());
    auto mr1 = std::make_shared<MemRef>(MemorySpace::DDR, off1, 1024, Sp());
    auto mr2 = std::make_shared<MemRef>(MemorySpace::DDR, off2, 1024, Sp());
    EXPECT_TRUE(structural_equal(Node(mr1), Node(mr2)));
    auto off3 = std::make_shared<ConstInt>(100, DataType::INT64, Sp());
    auto mr3 = std::make_shared<MemRef>(MemorySpace::DDR, off3, 1024, Sp());
    EXPECT_FALSE(structural_equal(Node(mr1), Node(mr3)));
}

TEST_F(IRStructEqExprTest, TestCallAndTupleExprsEqual)
{
    auto a1 = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto a2 = std::make_shared<ConstInt>(1, DataType::INT32, Sp());

    // Call
    auto c1 = std::make_shared<Call>("op", std::vector<ExprPtr>{a1}, Sp());
    auto c2 = std::make_shared<Call>("op", std::vector<ExprPtr>{a2}, Sp());
    EXPECT_TRUE(structural_equal(Node(c1), Node(c2)));
    auto c3 = std::make_shared<Call>("op_b", std::vector<ExprPtr>{a1}, Sp());
    EXPECT_FALSE(structural_equal(Node(c1), Node(c3)));

    // MakeTuple
    auto b1 = std::make_shared<ConstInt>(2, DataType::INT32, Sp());
    auto b2 = std::make_shared<ConstInt>(2, DataType::INT32, Sp());
    auto t1 = std::make_shared<MakeTuple>(std::vector<ExprPtr>{a1, b1}, Sp());
    auto t2 = std::make_shared<MakeTuple>(std::vector<ExprPtr>{a2, b2}, Sp());
    EXPECT_TRUE(structural_equal(Node(t1), Node(t2)));
    auto t3 = std::make_shared<MakeTuple>(std::vector<ExprPtr>{a1, a1}, Sp());
    EXPECT_FALSE(structural_equal(Node(t1), Node(t3)));

    // GetItemExpr
    auto idx1 = std::make_shared<ConstInt>(0, DataType::INDEX, Sp());
    auto idx2 = std::make_shared<ConstInt>(0, DataType::INDEX, Sp());
    auto g1 = std::make_shared<GetItemExpr>(t1, idx1, Sp());
    auto g2 = std::make_shared<GetItemExpr>(t2, idx2, Sp());
    EXPECT_TRUE(structural_equal(Node(g1), Node(g2)));

    // BinaryExpr
    auto add1 = std::make_shared<Add>(a1, b1, DataType::INT32, Sp());
    auto add2 = std::make_shared<Add>(a2, b2, DataType::INT32, Sp());
    EXPECT_TRUE(structural_equal(Node(add1), Node(add2)));
    auto sub1 = std::make_shared<Sub>(a1, b1, DataType::INT32, Sp());
    EXPECT_FALSE(structural_equal(Node(add1), Node(sub1)));

    // UnaryExpr
    auto neg1 = std::make_shared<Neg>(a1, DataType::INT32, Sp());
    auto neg2 = std::make_shared<Neg>(a2, DataType::INT32, Sp());
    EXPECT_TRUE(structural_equal(Node(neg1), Node(neg2)));
}

// ============================================================================
// Statement equality
// ============================================================================

TEST_F(IRStructEqExprTest, TestAllStmtsEqual)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto v1 = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto v2 = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto v42a = std::make_shared<ConstInt>(42, DataType::INT32, Sp());
    auto v42b = std::make_shared<ConstInt>(42, DataType::INT32, Sp());
    auto cond = std::make_shared<ConstBool>(true, Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, Sp());
    auto i = std::make_shared<Var>("i", Scalar(DataType::INT32), Sp());
    auto zero = std::make_shared<ConstInt>(0, DataType::INT32, Sp());
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto ten = std::make_shared<ConstInt>(10, DataType::INT32, Sp());
    auto call = std::make_shared<Call>("op", std::vector<ExprPtr>{v42a}, Sp());

    // AssignStmt
    EXPECT_TRUE(structural_equal(Node(std::make_shared<AssignStmt>(x, v42a, Sp())),
                                 Node(std::make_shared<AssignStmt>(x, v42b, Sp()))));

    // IfStmt without else
    EXPECT_TRUE(
        structural_equal(Node(std::make_shared<IfStmt>(cond, yield, std::nullopt, std::vector<VarPtr>{}, Sp())),
                         Node(std::make_shared<IfStmt>(cond, yield, std::nullopt, std::vector<VarPtr>{}, Sp()))));

    // IfStmt with else
    auto then = std::make_shared<AssignStmt>(x, v1, Sp());
    auto els = std::make_shared<AssignStmt>(x, v2, Sp());
    EXPECT_TRUE(structural_equal(Node(std::make_shared<IfStmt>(cond, then, els, std::vector<VarPtr>{}, Sp())),
                                 Node(std::make_shared<IfStmt>(cond, then, els, std::vector<VarPtr>{}, Sp()))));

    // YieldStmt, ReturnStmt
    EXPECT_TRUE(structural_equal(Node(std::make_shared<YieldStmt>(std::vector<ExprPtr>{v1}, Sp())),
                                 Node(std::make_shared<YieldStmt>(std::vector<ExprPtr>{v2}, Sp()))));
    EXPECT_TRUE(structural_equal(Node(std::make_shared<ReturnStmt>(std::vector<ExprPtr>{v42a}, Sp())),
                                 Node(std::make_shared<ReturnStmt>(std::vector<ExprPtr>{v42b}, Sp()))));

    // ForStmt, WhileStmt
    EXPECT_TRUE(structural_equal(Node(std::make_shared<ForStmt>(i, zero, ten, one, std::vector<IterArgPtr>{}, yield,
                                                                std::vector<VarPtr>{}, Sp())),
                                 Node(std::make_shared<ForStmt>(i, zero, ten, one, std::vector<IterArgPtr>{}, yield,
                                                                std::vector<VarPtr>{}, Sp()))));
    EXPECT_TRUE(structural_equal(
        Node(std::make_shared<WhileStmt>(cond, std::vector<IterArgPtr>{}, yield, std::vector<VarPtr>{}, Sp())),
        Node(std::make_shared<WhileStmt>(cond, std::vector<IterArgPtr>{}, yield, std::vector<VarPtr>{}, Sp()))));

    // SeqStmts
    auto a = std::make_shared<AssignStmt>(x, v1, Sp());
    EXPECT_TRUE(structural_equal(Node(std::make_shared<SeqStmts>(std::vector<StmtPtr>{a, a}, Sp())),
                                 Node(std::make_shared<SeqStmts>(std::vector<StmtPtr>{a, a}, Sp()))));

    // BreakStmt, ContinueStmt, EvalStmt
    EXPECT_TRUE(structural_equal(Node(std::make_shared<BreakStmt>(Sp())), Node(std::make_shared<BreakStmt>(Sp()))));
    EXPECT_TRUE(
        structural_equal(Node(std::make_shared<ContinueStmt>(Sp())), Node(std::make_shared<ContinueStmt>(Sp()))));
    EXPECT_TRUE(
        structural_equal(Node(std::make_shared<EvalStmt>(call, Sp())), Node(std::make_shared<EvalStmt>(call, Sp()))));
}

// ============================================================================
// Function and Program equality
// ============================================================================

TEST_F(IRStructEqExprTest, TestFunctionAndProgramEqual)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto v1 = std::make_shared<ConstInt>(42, DataType::INT32, Sp());
    auto v2 = std::make_shared<ConstInt>(42, DataType::INT32, Sp());
    auto body1 = std::make_shared<AssignStmt>(x, v1, Sp());
    auto body2 = std::make_shared<AssignStmt>(x, v2, Sp());

    // Function equal
    auto f1 = std::make_shared<Function>("f", std::vector<VarPtr>{x}, std::vector<TypePtr>{Scalar(DataType::INT32)},
                                         body1, Sp());
    auto f2 = std::make_shared<Function>("f", std::vector<VarPtr>{x}, std::vector<TypePtr>{Scalar(DataType::INT32)},
                                         body2, Sp());
    EXPECT_TRUE(structural_equal(Node(f1), Node(f2)));

    // Function not equal (different body)
    auto v99 = std::make_shared<ConstInt>(99, DataType::INT32, Sp());
    auto body3 = std::make_shared<AssignStmt>(x, v99, Sp());
    auto f3 = std::make_shared<Function>("f", std::vector<VarPtr>{x}, std::vector<TypePtr>{Scalar(DataType::INT32)},
                                         body3, Sp());
    EXPECT_FALSE(structural_equal(Node(f1), Node(f3)));

    // Program equal
    auto v = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto fbody = std::make_shared<AssignStmt>(x, v, Sp());
    auto f = std::make_shared<Function>("f", std::vector<VarPtr>{x}, std::vector<TypePtr>{}, fbody, Sp());
    auto p1 = std::make_shared<Program>(std::vector<FunctionPtr>{f}, "prog", Sp());
    auto p2 = std::make_shared<Program>(std::vector<FunctionPtr>{f}, "prog", Sp());
    EXPECT_TRUE(structural_equal(Node(p1), Node(p2)));
}

// ============================================================================
// Type equality
// ============================================================================

class IRStructEqTypeTest : public testing::Test {};

TEST_F(IRStructEqTypeTest, TestTensorAndTileTypeEqual)
{
    auto d16a = std::make_shared<ConstInt>(16, DataType::INT64, Sp());
    auto d16b = std::make_shared<ConstInt>(16, DataType::INT64, Sp());
    auto d32 = std::make_shared<ConstInt>(32, DataType::INT64, Sp());

    // TensorType equal / dtype mismatch / shape mismatch
    auto t1 = std::make_shared<TensorType>(std::vector<ExprPtr>{d16a}, DataType::FP32);
    auto t2 = std::make_shared<TensorType>(std::vector<ExprPtr>{d16b}, DataType::FP32);
    EXPECT_TRUE(structural_equal(t1, t2));
    EXPECT_FALSE(structural_equal(t1, std::make_shared<TensorType>(std::vector<ExprPtr>{d16a}, DataType::FP16)));
    EXPECT_FALSE(structural_equal(t1, std::make_shared<TensorType>(std::vector<ExprPtr>{d32}, DataType::FP32)));

    // TileType equal / dtype mismatch / rank mismatch
    auto tl1 = std::make_shared<TileType>(std::vector<ExprPtr>{d16a}, DataType::FP32);
    auto tl2 = std::make_shared<TileType>(std::vector<ExprPtr>{d16b}, DataType::FP32);
    EXPECT_TRUE(structural_equal(tl1, tl2));
    EXPECT_FALSE(structural_equal(tl1, std::make_shared<TileType>(std::vector<ExprPtr>{d16a}, DataType::FP16)));
    EXPECT_FALSE(structural_equal(tl1, std::make_shared<TileType>(std::vector<ExprPtr>{d16a, d16a}, DataType::FP32)));

    HardwareInfo compactNull(TileLayout::row_major, TileLayout::none_box, 512, TilePad::null, CompactMode::null);
    HardwareInfo compactNormal(TileLayout::row_major, TileLayout::none_box, 512, TilePad::null, CompactMode::normal);
    auto compactNullTile = std::make_shared<TileType>(std::vector<ExprPtr>{d16a}, DataType::FP32, std::nullopt,
                                                      std::nullopt, compactNull);
    auto compactNormalTile = std::make_shared<TileType>(std::vector<ExprPtr>{d16a}, DataType::FP32, std::nullopt,
                                                        std::nullopt, compactNormal);
    EXPECT_FALSE(structural_equal(compactNullTile, compactNormalTile));
}

TEST_F(IRStructEqTypeTest, TestTupleAndOtherTypes)
{
    auto s32 = std::make_shared<ScalarType>(DataType::INT32);

    // TupleType size mismatch
    EXPECT_FALSE(structural_equal(std::make_shared<TupleType>(std::vector<TypePtr>{s32}),
                                  std::make_shared<TupleType>(std::vector<TypePtr>{s32, s32})));

    // Type name mismatch
    EXPECT_FALSE(structural_equal(s32, std::make_shared<TupleType>(std::vector<TypePtr>{})));
    EXPECT_FALSE(structural_equal(
        s32, std::make_shared<TensorType>(std::vector<ExprPtr>{std::make_shared<ConstInt>(16, DataType::INT64, Sp())},
                                          DataType::FP32)));

    // MemRefType singleton
    EXPECT_TRUE(structural_equal(GetMemRefType(), GetMemRefType()));

    // TokenType kind match / mismatch
    EXPECT_TRUE(structural_equal(GetReadTokenType(), GetReadTokenType()));
    EXPECT_FALSE(structural_equal(GetReadTokenType(), GetWriteTokenType()));
}

// ============================================================================
// assert_structural_equal mismatch error paths
// ============================================================================

class IRStructEqAssertTest : public testing::Test {};

TEST_F(IRStructEqAssertTest, TestAssertExprMismatches)
{
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto b = std::make_shared<ConstInt>(2, DataType::INT32, Sp());

    // Type mismatch (ConstInt vs ConstFloat)
    EXPECT_THROW(assert_structural_equal(Node(a), Node(std::make_shared<ConstFloat>(1.0, DataType::FP32, Sp()))),
                 npu::tile_fwk::Error);

    // Null mismatch
    EXPECT_THROW(assert_structural_equal(Node(a), nullptr), npu::tile_fwk::Error);

    // Int mismatch
    EXPECT_THROW(assert_structural_equal(Node(a), Node(b)), npu::tile_fwk::Error);

    // Float mismatch
    EXPECT_THROW(assert_structural_equal(Node(std::make_shared<ConstFloat>(1.0, DataType::FP32, Sp())),
                                         Node(std::make_shared<ConstFloat>(2.0, DataType::FP32, Sp()))),
                 npu::tile_fwk::Error);

    // Bool mismatch
    EXPECT_THROW(assert_structural_equal(Node(std::make_shared<ConstBool>(true, Sp())),
                                         Node(std::make_shared<ConstBool>(false, Sp()))),
                 npu::tile_fwk::Error);

    // Call name mismatch
    EXPECT_THROW(assert_structural_equal(Node(std::make_shared<Call>("op_a", std::vector<ExprPtr>{a}, Sp())),
                                         Node(std::make_shared<Call>("op_b", std::vector<ExprPtr>{a}, Sp()))),
                 npu::tile_fwk::Error);

    // Call arg count mismatch
    EXPECT_THROW(assert_structural_equal(Node(std::make_shared<Call>("op", std::vector<ExprPtr>{a}, Sp())),
                                         Node(std::make_shared<Call>("op", std::vector<ExprPtr>{a, b}, Sp()))),
                 npu::tile_fwk::Error);
}

TEST_F(IRStructEqAssertTest, TestAssertStmtMismatches)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto cond = std::make_shared<ConstBool>(true, Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, Sp());

    // Function body mismatch
    auto body1 = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(1, DataType::INT32, Sp()), Sp());
    auto body2 = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(2, DataType::INT32, Sp()), Sp());
    EXPECT_THROW(
        assert_structural_equal(
            Node(std::make_shared<Function>("f", std::vector<VarPtr>{x}, std::vector<TypePtr>{}, body1, Sp())),
            Node(std::make_shared<Function>("f", std::vector<VarPtr>{x}, std::vector<TypePtr>{}, body2, Sp()))),
        npu::tile_fwk::Error);

    // Optional field mismatch (IfStmt with/without else)
    EXPECT_THROW(
        assert_structural_equal(Node(std::make_shared<IfStmt>(cond, yield, std::nullopt, std::vector<VarPtr>{}, Sp())),
                                Node(std::make_shared<IfStmt>(cond, yield, yield, std::vector<VarPtr>{}, Sp()))),
        npu::tile_fwk::Error);

    // Vector size mismatch (SeqStmts)
    auto a1 = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(1, DataType::INT32, Sp()), Sp());
    auto a2 = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(2, DataType::INT32, Sp()), Sp());
    EXPECT_THROW(assert_structural_equal(Node(std::make_shared<SeqStmts>(std::vector<StmtPtr>{a1}, Sp())),
                                         Node(std::make_shared<SeqStmts>(std::vector<StmtPtr>{a1, a2}, Sp()))),
                 npu::tile_fwk::Error);
}

TEST_F(IRStructEqAssertTest, TestAssertTypeMismatches)
{
    auto d = std::make_shared<ConstInt>(16, DataType::INT64, Sp());

    // Scalar dtype
    EXPECT_THROW(assert_structural_equal(std::make_shared<ScalarType>(DataType::INT32),
                                         std::make_shared<ScalarType>(DataType::FP32)),
                 npu::tile_fwk::Error);

    // Tuple elements
    EXPECT_THROW(assert_structural_equal(
                     std::make_shared<TupleType>(std::vector<TypePtr>{std::make_shared<ScalarType>(DataType::INT32)}),
                     std::make_shared<TupleType>(std::vector<TypePtr>{std::make_shared<ScalarType>(DataType::FP32)})),
                 npu::tile_fwk::Error);

    // Tensor dtype / shape rank
    EXPECT_THROW(assert_structural_equal(std::make_shared<TensorType>(std::vector<ExprPtr>{d}, DataType::FP32),
                                         std::make_shared<TensorType>(std::vector<ExprPtr>{d}, DataType::FP16)),
                 npu::tile_fwk::Error);
    EXPECT_THROW(assert_structural_equal(std::make_shared<TensorType>(std::vector<ExprPtr>{d}, DataType::FP32),
                                         std::make_shared<TensorType>(std::vector<ExprPtr>{d, d}, DataType::FP32)),
                 npu::tile_fwk::Error);

    // Tile dtype / shape rank
    EXPECT_THROW(assert_structural_equal(std::make_shared<TileType>(std::vector<ExprPtr>{d}, DataType::FP32),
                                         std::make_shared<TileType>(std::vector<ExprPtr>{d}, DataType::FP16)),
                 npu::tile_fwk::Error);
    EXPECT_THROW(assert_structural_equal(std::make_shared<TileType>(std::vector<ExprPtr>{d}, DataType::FP32),
                                         std::make_shared<TileType>(std::vector<ExprPtr>{d, d}, DataType::FP32)),
                 npu::tile_fwk::Error);

    // Tuple size
    auto s = std::make_shared<ScalarType>(DataType::INT32);
    EXPECT_THROW(assert_structural_equal(std::make_shared<TupleType>(std::vector<TypePtr>{s}),
                                         std::make_shared<TupleType>(std::vector<TypePtr>{s, s})),
                 npu::tile_fwk::Error);
}

// ============================================================================
// Auto-mapping variable equality
// ============================================================================

TEST_F(IRStructEqExprTest, TestVarAutoMappingEqual)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto y = std::make_shared<Var>("y", Scalar(DataType::INT32), Sp());
    auto v1 = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto v2 = std::make_shared<ConstInt>(1, DataType::INT32, Sp());

    // Auto-mapping: x↔y mapped, same structure
    EXPECT_TRUE(structural_equal(Node(std::make_shared<AssignStmt>(x, v1, Sp())),
                                 Node(std::make_shared<AssignStmt>(y, v2, Sp())), true));

    // Without auto-mapping, distinct vars differ
    EXPECT_FALSE(structural_equal(Node(x), Node(y), false));

    // Auto-mapping with type mismatch
    auto z = std::make_shared<Var>("z", Scalar(DataType::FP32), Sp());
    EXPECT_FALSE(structural_equal(Node(std::make_shared<AssignStmt>(x, v1, Sp())),
                                  Node(std::make_shared<AssignStmt>(z, v1, Sp())), true));
}

TEST_F(IRStructEqExprTest, TestVarConsistentMapping)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto y = std::make_shared<Var>("y", Scalar(DataType::INT32), Sp());
    auto v1 = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto v2 = std::make_shared<ConstInt>(1, DataType::INT32, Sp());

    // x used twice on left, y used twice on right — consistent mapping
    auto body1 = std::make_shared<SeqStmts>(
        std::vector<StmtPtr>{std::make_shared<AssignStmt>(x, v1, Sp()), std::make_shared<AssignStmt>(x, v2, Sp())},
        Sp());
    auto body2 = std::make_shared<SeqStmts>(
        std::vector<StmtPtr>{std::make_shared<AssignStmt>(y, v1, Sp()), std::make_shared<AssignStmt>(y, v2, Sp())},
        Sp());
    EXPECT_TRUE(structural_equal(Node(body1), Node(body2), true));
}

// ============================================================================
// ForStmt with iterArgs — IterArg initValue comparison
// ============================================================================

TEST_F(IRStructEqExprTest, TestForStmtWithIterArgs)
{
    auto i1 = std::make_shared<Var>("i", Scalar(DataType::INT32), Sp());
    auto i2 = std::make_shared<Var>("i", Scalar(DataType::INT32), Sp());
    auto zero = std::make_shared<ConstInt>(0, DataType::INT32, Sp());
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto ten = std::make_shared<ConstInt>(10, DataType::INT32, Sp());
    auto yv = std::make_shared<ConstInt>(1, DataType::INT32, Sp());

    auto ia1 = std::make_shared<IterArg>("acc", Scalar(DataType::INT32), zero, Sp());
    auto ia2 = std::make_shared<IterArg>("acc", Scalar(DataType::INT32),
                                         std::make_shared<ConstInt>(0, DataType::INT32, Sp()), Sp());
    auto rv1 = std::make_shared<Var>("out", Scalar(DataType::INT32), Sp());
    auto rv2 = std::make_shared<Var>("out", Scalar(DataType::INT32), Sp());

    auto body1 = std::make_shared<YieldStmt>(std::vector<ExprPtr>{yv}, Sp());
    auto body2 = std::make_shared<YieldStmt>(std::vector<ExprPtr>{yv}, Sp());

    // Equal with same init
    EXPECT_TRUE(structural_equal(Node(std::make_shared<ForStmt>(i1, zero, ten, one, std::vector<IterArgPtr>{ia1}, body1,
                                                                std::vector<VarPtr>{rv1}, Sp())),
                                 Node(std::make_shared<ForStmt>(i2, zero, ten, one, std::vector<IterArgPtr>{ia2}, body2,
                                                                std::vector<VarPtr>{rv2}, Sp()))));

    // Not equal with different init
    auto ia3 = std::make_shared<IterArg>("acc", Scalar(DataType::INT32), one, Sp());
    EXPECT_FALSE(structural_equal(Node(std::make_shared<ForStmt>(i1, zero, ten, one, std::vector<IterArgPtr>{ia1},
                                                                 body1, std::vector<VarPtr>{rv1}, Sp())),
                                  Node(std::make_shared<ForStmt>(i1, zero, ten, one, std::vector<IterArgPtr>{ia3},
                                                                 body1, std::vector<VarPtr>{rv1}, Sp()))));
}

// ============================================================================
// SectionStmt and OpStmts equality
// ============================================================================

TEST_F(IRStructEqExprTest, TestSectionStmtEqual)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto v1 = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto v2 = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto body1 = std::make_shared<AssignStmt>(x, v1, Sp());
    auto body2 = std::make_shared<AssignStmt>(x, v2, Sp());

    EXPECT_TRUE(structural_equal(Node(std::make_shared<SectionStmt>(SectionKind::Vector, body1, Sp())),
                                 Node(std::make_shared<SectionStmt>(SectionKind::Vector, body2, Sp()))));

    EXPECT_FALSE(structural_equal(Node(std::make_shared<SectionStmt>(SectionKind::Vector, body1, Sp())),
                                  Node(std::make_shared<SectionStmt>(SectionKind::Cube, body1, Sp()))));
}

// ============================================================================
// TileType with tileView_ and hardwareInfo_
// ============================================================================

TEST_F(IRStructEqTypeTest, TestTileTypeWithTileView)
{
    auto d16a = std::make_shared<ConstInt>(16, DataType::INT64, Sp());
    auto d16b = std::make_shared<ConstInt>(16, DataType::INT64, Sp());
    auto off0a = std::make_shared<ConstInt>(0, DataType::INT64, Sp());
    auto off0b = std::make_shared<ConstInt>(0, DataType::INT64, Sp());

    TileView tv1({d16a}, {d16a}, off0a);
    TileView tv2({d16b}, {d16b}, off0b);

    auto tl1 = std::make_shared<TileType>(std::vector<ExprPtr>{d16a}, DataType::FP32, std::nullopt, tv1);
    auto tl2 = std::make_shared<TileType>(std::vector<ExprPtr>{d16b}, DataType::FP32, std::nullopt, tv2);
    EXPECT_TRUE(structural_equal(tl1, tl2));

    auto d32 = std::make_shared<ConstInt>(32, DataType::INT64, Sp());
    TileView tv3({d32}, {d16a}, off0a);
    auto tl3 = std::make_shared<TileType>(std::vector<ExprPtr>{d16a}, DataType::FP32, std::nullopt, tv3);
    EXPECT_FALSE(structural_equal(tl1, tl3));
}

TEST_F(IRStructEqTypeTest, TestTileTypeWithHardwareInfo)
{
    auto d16 = std::make_shared<ConstInt>(16, DataType::INT64, Sp());

    HardwareInfo hw1(TileLayout::row_major, TileLayout::none_box, 512, TilePad::null);
    HardwareInfo hw2(TileLayout::row_major, TileLayout::none_box, 512, TilePad::null);

    auto tl1 = std::make_shared<TileType>(std::vector<ExprPtr>{d16}, DataType::FP32, std::nullopt, std::nullopt, hw1);
    auto tl2 = std::make_shared<TileType>(std::vector<ExprPtr>{d16}, DataType::FP32, std::nullopt, std::nullopt, hw2);
    EXPECT_TRUE(structural_equal(tl1, tl2));

    HardwareInfo hw3(TileLayout::col_major, TileLayout::none_box, 512, TilePad::null);
    auto tl3 = std::make_shared<TileType>(std::vector<ExprPtr>{d16}, DataType::FP32, std::nullopt, std::nullopt, hw3);
    EXPECT_FALSE(structural_equal(tl1, tl3));
    EXPECT_THROW(assert_structural_equal(tl1, tl3), npu::tile_fwk::Error);
}

TEST_F(IRStructEqTypeTest, TestTileTypeTileViewPresenceMismatch)
{
    auto d16 = std::make_shared<ConstInt>(16, DataType::INT64, Sp());
    auto off0 = std::make_shared<ConstInt>(0, DataType::INT64, Sp());
    TileView tv({d16}, {d16}, off0);

    auto tl1 = std::make_shared<TileType>(std::vector<ExprPtr>{d16}, DataType::FP32, std::nullopt, tv);
    auto tl2 = std::make_shared<TileType>(std::vector<ExprPtr>{d16}, DataType::FP32);
    EXPECT_FALSE(structural_equal(tl1, tl2));
}

TEST_F(IRStructEqTypeTest, TestTileTypeHardwareInfoPresenceMismatch)
{
    auto d16 = std::make_shared<ConstInt>(16, DataType::INT64, Sp());
    HardwareInfo hw(TileLayout::row_major);

    auto tl1 = std::make_shared<TileType>(std::vector<ExprPtr>{d16}, DataType::FP32, std::nullopt, std::nullopt, hw);
    auto tl2 = std::make_shared<TileType>(std::vector<ExprPtr>{d16}, DataType::FP32);
    EXPECT_FALSE(structural_equal(tl1, tl2));
}

// ============================================================================
// UnknownType and LogicalTensorType
// ============================================================================

TEST_F(IRStructEqTypeTest, TestUnknownTypeEqual)
{
    EXPECT_TRUE(structural_equal(std::make_shared<UnknownType>(), std::make_shared<UnknownType>()));
}

TEST_F(IRStructEqTypeTest, TestLogicalTensorTypeEqual)
{
    EXPECT_TRUE(structural_equal(std::make_shared<LogicalTensorType>(), std::make_shared<LogicalTensorType>()));
}

// ============================================================================
// Kwargs comparison via Call
// ============================================================================

TEST_F(IRStructEqExprTest, TestCallWithKwargsEqual)
{
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    std::vector<std::pair<std::string, std::any>> kwargs1 = {{"mode", std::any(std::string("round"))}};
    std::vector<std::pair<std::string, std::any>> kwargs2 = {{"mode", std::any(std::string("round"))}};

    auto c1 = std::make_shared<Call>("op", std::vector<ExprPtr>{a}, kwargs1, Scalar(DataType::INT32), Sp());
    auto c2 = std::make_shared<Call>("op", std::vector<ExprPtr>{a}, kwargs2, Scalar(DataType::INT32), Sp());
    EXPECT_TRUE(structural_equal(Node(c1), Node(c2)));
}

TEST_F(IRStructEqExprTest, TestCallWithKwargsMismatch)
{
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    std::vector<std::pair<std::string, std::any>> kwargs1 = {{"mode", std::any(std::string("round"))}};
    std::vector<std::pair<std::string, std::any>> kwargs2 = {{"mode", std::any(std::string("floor"))}};

    auto c1 = std::make_shared<Call>("op", std::vector<ExprPtr>{a}, kwargs1, Scalar(DataType::INT32), Sp());
    auto c2 = std::make_shared<Call>("op", std::vector<ExprPtr>{a}, kwargs2, Scalar(DataType::INT32), Sp());
    EXPECT_FALSE(structural_equal(Node(c1), Node(c2)));
    EXPECT_THROW(assert_structural_equal(Node(c1), Node(c2)), npu::tile_fwk::Error);

    std::vector<std::pair<std::string, std::any>> kwargs3 = {{"mode", std::any(std::string("round"))},
                                                             {"axis", std::any(int(0))}};
    auto c3 = std::make_shared<Call>("op", std::vector<ExprPtr>{a}, kwargs3, Scalar(DataType::INT32), Sp());
    EXPECT_FALSE(structural_equal(Node(c1), Node(c3)));
    EXPECT_THROW(assert_structural_equal(Node(c1), Node(c3)), npu::tile_fwk::Error);
}

TEST_F(IRStructEqExprTest, TestCallKwargsSizeMismatch)
{
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    std::vector<std::pair<std::string, std::any>> kwargs1 = {{"mode", std::any(std::string("round"))}};
    std::vector<std::pair<std::string, std::any>> kwargs2 = {};

    auto c1 = std::make_shared<Call>("op", std::vector<ExprPtr>{a}, kwargs1, Scalar(DataType::INT32), Sp());
    auto c2 = std::make_shared<Call>("op", std::vector<ExprPtr>{a}, kwargs2, Scalar(DataType::INT32), Sp());
    EXPECT_FALSE(structural_equal(Node(c1), Node(c2)));
    EXPECT_THROW(assert_structural_equal(Node(c1), Node(c2)), npu::tile_fwk::Error);
}

TEST_F(IRStructEqExprTest, TestCallKwargsIntValue)
{
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    std::vector<std::pair<std::string, std::any>> kwargs1 = {{"axis", std::any(int(0))}};
    std::vector<std::pair<std::string, std::any>> kwargs2 = {{"axis", std::any(int(0))}};
    std::vector<std::pair<std::string, std::any>> kwargs3 = {{"axis", std::any(int(1))}};

    auto c1 = std::make_shared<Call>("op", std::vector<ExprPtr>{a}, kwargs1, Scalar(DataType::INT32), Sp());
    auto c2 = std::make_shared<Call>("op", std::vector<ExprPtr>{a}, kwargs2, Scalar(DataType::INT32), Sp());
    auto c3 = std::make_shared<Call>("op", std::vector<ExprPtr>{a}, kwargs3, Scalar(DataType::INT32), Sp());
    EXPECT_TRUE(structural_equal(Node(c1), Node(c2)));
    EXPECT_FALSE(structural_equal(Node(c1), Node(c3)));
}

TEST_F(IRStructEqExprTest, TestCallKwargsBoolAndDataType)
{
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, Sp());

    std::vector<std::pair<std::string, std::any>> kwargs_b1 = {{"flag", std::any(true)}};
    std::vector<std::pair<std::string, std::any>> kwargs_b2 = {{"flag", std::any(true)}};
    std::vector<std::pair<std::string, std::any>> kwargs_b3 = {{"flag", std::any(false)}};
    auto cb1 = std::make_shared<Call>("op", std::vector<ExprPtr>{a}, kwargs_b1, Scalar(DataType::INT32), Sp());
    auto cb2 = std::make_shared<Call>("op", std::vector<ExprPtr>{a}, kwargs_b2, Scalar(DataType::INT32), Sp());
    auto cb3 = std::make_shared<Call>("op", std::vector<ExprPtr>{a}, kwargs_b3, Scalar(DataType::INT32), Sp());
    EXPECT_TRUE(structural_equal(Node(cb1), Node(cb2)));
    EXPECT_FALSE(structural_equal(Node(cb1), Node(cb3)));

    std::vector<std::pair<std::string, std::any>> kwargs_d1 = {{"dtype", std::any(DataType::FP32)}};
    std::vector<std::pair<std::string, std::any>> kwargs_d2 = {{"dtype", std::any(DataType::FP32)}};
    std::vector<std::pair<std::string, std::any>> kwargs_d3 = {{"dtype", std::any(DataType::FP16)}};
    auto cd1 = std::make_shared<Call>("op", std::vector<ExprPtr>{a}, kwargs_d1, Scalar(DataType::INT32), Sp());
    auto cd2 = std::make_shared<Call>("op", std::vector<ExprPtr>{a}, kwargs_d2, Scalar(DataType::INT32), Sp());
    auto cd3 = std::make_shared<Call>("op", std::vector<ExprPtr>{a}, kwargs_d3, Scalar(DataType::INT32), Sp());
    EXPECT_TRUE(structural_equal(Node(cd1), Node(cd2)));
    EXPECT_FALSE(structural_equal(Node(cd1), Node(cd3)));
}

TEST_F(IRStructEqExprTest, TestCallKwargsDoubleAndVectorInt)
{
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, Sp());

    std::vector<std::pair<std::string, std::any>> kwargs_db1 = {{"scale", std::any(1.5)}};
    std::vector<std::pair<std::string, std::any>> kwargs_db2 = {{"scale", std::any(1.5)}};
    std::vector<std::pair<std::string, std::any>> kwargs_db3 = {{"scale", std::any(2.5)}};
    auto cdb1 = std::make_shared<Call>("op", std::vector<ExprPtr>{a}, kwargs_db1, Scalar(DataType::INT32), Sp());
    auto cdb2 = std::make_shared<Call>("op", std::vector<ExprPtr>{a}, kwargs_db2, Scalar(DataType::INT32), Sp());
    auto cdb3 = std::make_shared<Call>("op", std::vector<ExprPtr>{a}, kwargs_db3, Scalar(DataType::INT32), Sp());
    EXPECT_TRUE(structural_equal(Node(cdb1), Node(cdb2)));
    EXPECT_FALSE(structural_equal(Node(cdb1), Node(cdb3)));

    std::vector<std::pair<std::string, std::any>> kwargs_v1 = {{"shape", std::any(std::vector<int>{16, 32})}};
    std::vector<std::pair<std::string, std::any>> kwargs_v2 = {{"shape", std::any(std::vector<int>{16, 32})}};
    std::vector<std::pair<std::string, std::any>> kwargs_v3 = {{"shape", std::any(std::vector<int>{8, 16})}};
    auto cv1 = std::make_shared<Call>("op", std::vector<ExprPtr>{a}, kwargs_v1, Scalar(DataType::INT32), Sp());
    auto cv2 = std::make_shared<Call>("op", std::vector<ExprPtr>{a}, kwargs_v2, Scalar(DataType::INT32), Sp());
    auto cv3 = std::make_shared<Call>("op", std::vector<ExprPtr>{a}, kwargs_v3, Scalar(DataType::INT32), Sp());
    EXPECT_TRUE(structural_equal(Node(cv1), Node(cv2)));
    EXPECT_FALSE(structural_equal(Node(cv1), Node(cv3)));
}

TEST_F(IRStructEqExprTest, TestCallKwargsKeyTypeMismatch)
{
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    std::vector<std::pair<std::string, std::any>> kwargs1 = {{"alpha", std::any(std::string("a"))}};
    std::vector<std::pair<std::string, std::any>> kwargs2 = {{"beta", std::any(std::string("a"))}};

    auto c1 = std::make_shared<Call>("op", std::vector<ExprPtr>{a}, kwargs1, Scalar(DataType::INT32), Sp());
    auto c2 = std::make_shared<Call>("op", std::vector<ExprPtr>{a}, kwargs2, Scalar(DataType::INT32), Sp());
    EXPECT_FALSE(structural_equal(Node(c1), Node(c2)));
    EXPECT_THROW(assert_structural_equal(Node(c1), Node(c2)), npu::tile_fwk::Error);
}

TEST_F(IRStructEqExprTest, TestCallKwargsValueTypeMismatch)
{
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    std::vector<std::pair<std::string, std::any>> kwargs1 = {{"x", std::any(int(1))}};
    std::vector<std::pair<std::string, std::any>> kwargs2 = {{"x", std::any(std::string("1"))}};

    auto c1 = std::make_shared<Call>("op", std::vector<ExprPtr>{a}, kwargs1, Scalar(DataType::INT32), Sp());
    auto c2 = std::make_shared<Call>("op", std::vector<ExprPtr>{a}, kwargs2, Scalar(DataType::INT32), Sp());
    EXPECT_FALSE(structural_equal(Node(c1), Node(c2)));
    EXPECT_THROW(assert_structural_equal(Node(c1), Node(c2)), npu::tile_fwk::Error);
}

// ============================================================================
// Function returnTypes_ vector mismatch
// ============================================================================

TEST_F(IRStructEqExprTest, TestFunctionReturnTypesSizeMismatch)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto body = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(1, DataType::INT32, Sp()), Sp());

    auto f1 = std::make_shared<Function>("f", std::vector<VarPtr>{x}, std::vector<TypePtr>{Scalar(DataType::INT32)},
                                         body, Sp());
    auto f2 = std::make_shared<Function>(
        "f", std::vector<VarPtr>{x}, std::vector<TypePtr>{Scalar(DataType::INT32), Scalar(DataType::FP32)}, body, Sp());
    EXPECT_FALSE(structural_equal(Node(f1), Node(f2)));
    EXPECT_THROW(assert_structural_equal(Node(f1), Node(f2)), npu::tile_fwk::Error);
}

TEST_F(IRStructEqExprTest, TestFunctionReturnTypesElementMismatch)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto body = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(1, DataType::INT32, Sp()), Sp());

    auto f1 = std::make_shared<Function>("f", std::vector<VarPtr>{x}, std::vector<TypePtr>{Scalar(DataType::INT32)},
                                         body, Sp());
    auto f2 = std::make_shared<Function>("f", std::vector<VarPtr>{x}, std::vector<TypePtr>{Scalar(DataType::FP32)},
                                         body, Sp());
    EXPECT_FALSE(structural_equal(Node(f1), Node(f2)));
}

TEST_F(IRStructEqExprTest, TestFunctionMaxThreadsMismatch)
{
    auto body = std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, Sp());
    auto f128 = std::make_shared<Function>("f", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body, Sp(),
                                           FunctionType::SIMT_VF, false,
                                           std::vector<std::pair<std::string, std::any>>{{kMaxThreadsAttr, 128}});
    auto f256 = std::make_shared<Function>("f", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body, Sp(),
                                           FunctionType::SIMT_VF, false,
                                           std::vector<std::pair<std::string, std::any>>{{kMaxThreadsAttr, 256}});
    auto unbounded = std::make_shared<Function>("f", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body, Sp(),
                                                FunctionType::SIMT_VF);

    EXPECT_FALSE(structural_equal(Node(f128), Node(f256)));
    EXPECT_FALSE(structural_equal(Node(f128), Node(unbounded)));
    EXPECT_THROW(assert_structural_equal(Node(f128), Node(f256)), npu::tile_fwk::Error);
}

// ============================================================================
// IterArg vector size mismatch
// ============================================================================

TEST_F(IRStructEqExprTest, TestForStmtIterArgSizeMismatch)
{
    auto i = std::make_shared<Var>("i", Scalar(DataType::INT32), Sp());
    auto zero = std::make_shared<ConstInt>(0, DataType::INT32, Sp());
    auto ten = std::make_shared<ConstInt>(10, DataType::INT32, Sp());
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, Sp());
    auto ia = std::make_shared<IterArg>("acc", Scalar(DataType::INT32), zero, Sp());

    auto for1 = std::make_shared<ForStmt>(i, zero, ten, one, std::vector<IterArgPtr>{}, yield, std::vector<VarPtr>{},
                                          Sp());
    auto for2 = std::make_shared<ForStmt>(i, zero, ten, one, std::vector<IterArgPtr>{ia}, yield, std::vector<VarPtr>{},
                                          Sp());

    EXPECT_FALSE(structural_equal(Node(for1), Node(for2)));
    EXPECT_THROW(assert_structural_equal(Node(for1), Node(for2)), npu::tile_fwk::Error);
}

// ============================================================================
// ScalarType INT64/INDEX equivalence in loop_var context
// ============================================================================

TEST_F(IRStructEqExprTest, TestLoopVarInt64IndexNotEquivalent)
{
    auto i1 = std::make_shared<Var>("i", Scalar(DataType::INT64), Sp());
    auto i2 = std::make_shared<Var>("i", Scalar(DataType::INDEX), Sp());
    auto zero = std::make_shared<ConstInt>(0, DataType::INT32, Sp());
    auto ten = std::make_shared<ConstInt>(10, DataType::INT32, Sp());
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, Sp());

    EXPECT_FALSE(structural_equal(Node(std::make_shared<ForStmt>(i1, zero, ten, one, std::vector<IterArgPtr>{}, yield,
                                                                 std::vector<VarPtr>{}, Sp())),
                                  Node(std::make_shared<ForStmt>(i2, zero, ten, one, std::vector<IterArgPtr>{}, yield,
                                                                 std::vector<VarPtr>{}, Sp()))));
}

// ============================================================================
// MemRef assert mode mismatch
// ============================================================================

TEST_F(IRStructEqAssertTest, TestMemRefMismatch)
{
    auto off0 = std::make_shared<ConstInt>(0, DataType::INT64, Sp());
    auto off100 = std::make_shared<ConstInt>(100, DataType::INT64, Sp());

    auto mr1 = std::make_shared<MemRef>(MemorySpace::DDR, off0, 1024, Sp());
    auto mr2 = std::make_shared<MemRef>(MemorySpace::DDR, off100, 1024, Sp());
    EXPECT_FALSE(structural_equal(Node(mr1), Node(mr2)));
    EXPECT_THROW(assert_structural_equal(Node(mr1), Node(mr2)), npu::tile_fwk::Error);

    auto mr3 = std::make_shared<MemRef>(MemorySpace::DDR, off0, 2048, Sp());
    EXPECT_FALSE(structural_equal(Node(mr1), Node(mr3)));
    EXPECT_THROW(assert_structural_equal(Node(mr1), Node(mr3)), npu::tile_fwk::Error);

    auto mr4 = std::make_shared<MemRef>(MemorySpace::Vec, off0, 1024, Sp());
    EXPECT_FALSE(structural_equal(Node(mr1), Node(mr4)));
    EXPECT_THROW(assert_structural_equal(Node(mr1), Node(mr4)), npu::tile_fwk::Error);
}

// ============================================================================
// Program map mismatch
// ============================================================================

TEST_F(IRStructEqAssertTest, TestAssertProgramMapMismatch)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto body = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(1, DataType::INT32, Sp()), Sp());
    auto f1 = std::make_shared<Function>("f", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body, Sp());
    auto f2 = std::make_shared<Function>("g", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body, Sp());

    auto p1 = std::make_shared<Program>(std::vector<FunctionPtr>{f1}, "prog", Sp());
    auto p2 = std::make_shared<Program>(std::vector<FunctionPtr>{f2}, "prog", Sp());
    EXPECT_THROW(assert_structural_equal(Node(p1), Node(p2)), npu::tile_fwk::Error);
}

TEST_F(IRStructEqAssertTest, TestAssertProgramMapSizeMismatch)
{
    auto body = std::make_shared<AssignStmt>(std::make_shared<Var>("x", Scalar(DataType::INT32), Sp()),
                                             std::make_shared<ConstInt>(1, DataType::INT32, Sp()), Sp());
    auto f1 = std::make_shared<Function>("f", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body, Sp());
    auto f2 = std::make_shared<Function>("g", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body, Sp());

    auto p1 = std::make_shared<Program>(std::vector<FunctionPtr>{f1}, "prog", Sp());
    auto p2 = std::make_shared<Program>(std::vector<FunctionPtr>{f1, f2}, "prog", Sp());
    EXPECT_THROW(assert_structural_equal(Node(p1), Node(p2)), npu::tile_fwk::Error);
}

// ============================================================================
// Equal: null node and type mismatch branches
// ============================================================================

TEST_F(IRStructEqExprTest, TestEqualOneNullNode)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    IRNodePtr null_ptr;

    EXPECT_FALSE(structural_equal(Node(x), null_ptr));
    EXPECT_FALSE(structural_equal(null_ptr, Node(x)));
    EXPECT_TRUE(structural_equal(null_ptr, null_ptr));
}

TEST_F(IRStructEqExprTest, TestEqualNodeTypeMismatch)
{
    auto ci = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto cf = std::make_shared<ConstFloat>(1.0, DataType::FP32, Sp());

    EXPECT_FALSE(structural_equal(Node(ci), Node(cf)));
}

TEST_F(IRStructEqAssertTest, TestAssertOneNullNode)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    IRNodePtr null_ptr;

    EXPECT_THROW(assert_structural_equal(Node(x), null_ptr), npu::tile_fwk::Error);
    EXPECT_THROW(assert_structural_equal(null_ptr, Node(x)), npu::tile_fwk::Error);
}

TEST_F(IRStructEqAssertTest, TestAssertNodeTypeMismatch)
{
    auto ci = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto cf = std::make_shared<ConstFloat>(1.0, DataType::FP32, Sp());

    EXPECT_THROW(assert_structural_equal(Node(ci), Node(cf)), npu::tile_fwk::Error);
}

// ============================================================================
// EqualVar: enable_auto_mapping=false branches
// ============================================================================

TEST_F(IRStructEqExprTest, TestVarNoAutoMapping_SamePointer)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto call1 = std::make_shared<Call>("op", std::vector<ExprPtr>{x}, Sp());
    auto call2 = std::make_shared<Call>("op", std::vector<ExprPtr>{x}, Sp());

    EXPECT_TRUE(structural_equal(Node(call1), Node(call2), false));
}

TEST_F(IRStructEqExprTest, TestVarNoAutoMapping_DifferentPointerSameName)
{
    auto x1 = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto x2 = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto c1 = std::make_shared<Call>("op", std::vector<ExprPtr>{x1}, Sp());
    auto c2 = std::make_shared<Call>("op", std::vector<ExprPtr>{x2}, Sp());

    EXPECT_FALSE(structural_equal(Node(c1), Node(c2), false));
}

TEST_F(IRStructEqExprTest, TestVarNoAutoMapping_InconsistentMapping)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto y1 = std::make_shared<Var>("y", Scalar(DataType::INT32), Sp());
    auto y2 = std::make_shared<Var>("z", Scalar(DataType::INT32), Sp());
    auto v1 = std::make_shared<ConstInt>(1, DataType::INT32, Sp());

    auto c1_lhs = std::make_shared<Call>("op", std::vector<ExprPtr>{x}, Sp());
    auto c2_lhs = std::make_shared<Call>("op", std::vector<ExprPtr>{x}, Sp());
    auto c1_rhs = std::make_shared<Call>("op", std::vector<ExprPtr>{y1}, Sp());
    auto c2_rhs = std::make_shared<Call>("op", std::vector<ExprPtr>{y2}, Sp());

    auto seq1 = std::make_shared<SeqStmts>(
        std::vector<StmtPtr>{std::make_shared<EvalStmt>(c1_lhs, Sp()), std::make_shared<EvalStmt>(c2_lhs, Sp())}, Sp());
    auto seq2 = std::make_shared<SeqStmts>(
        std::vector<StmtPtr>{std::make_shared<EvalStmt>(c1_rhs, Sp()), std::make_shared<EvalStmt>(c2_rhs, Sp())}, Sp());

    EXPECT_FALSE(structural_equal(Node(seq1), Node(seq2), true));
}

TEST_F(IRStructEqExprTest, TestVarAutoMapping_ReverseInconsistent)
{
    auto x1 = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto x2 = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto y = std::make_shared<Var>("y", Scalar(DataType::INT32), Sp());

    auto seq1 = std::make_shared<SeqStmts>(
        std::vector<StmtPtr>{
            std::make_shared<EvalStmt>(std::make_shared<Call>("op", std::vector<ExprPtr>{x1}, Sp()), Sp()),
            std::make_shared<EvalStmt>(std::make_shared<Call>("op", std::vector<ExprPtr>{x2}, Sp()), Sp())},
        Sp());
    auto seq2 = std::make_shared<SeqStmts>(
        std::vector<StmtPtr>{
            std::make_shared<EvalStmt>(std::make_shared<Call>("op", std::vector<ExprPtr>{y}, Sp()), Sp()),
            std::make_shared<EvalStmt>(std::make_shared<Call>("op", std::vector<ExprPtr>{y}, Sp()), Sp())},
        Sp());

    EXPECT_FALSE(structural_equal(Node(seq1), Node(seq2), true));
    EXPECT_THROW(assert_structural_equal(Node(seq1), Node(seq2)), npu::tile_fwk::Error);
}

// ============================================================================
// EqualIterArg: initValue mismatch
// ============================================================================

TEST_F(IRStructEqExprTest, TestIterArgInitValueMismatch)
{
    auto i = std::make_shared<Var>("i", Scalar(DataType::INT32), Sp());
    auto zero = std::make_shared<ConstInt>(0, DataType::INT32, Sp());
    auto one_val = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto ten = std::make_shared<ConstInt>(10, DataType::INT32, Sp());
    auto step = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{zero}, Sp());

    auto ia1 = std::make_shared<IterArg>("acc", Scalar(DataType::INT32), zero, Sp());
    auto ia2 = std::make_shared<IterArg>("acc", Scalar(DataType::INT32), one_val, Sp());
    auto rv = std::make_shared<Var>("out", Scalar(DataType::INT32), Sp());

    auto for1 = std::make_shared<ForStmt>(i, zero, ten, step, std::vector<IterArgPtr>{ia1}, yield,
                                          std::vector<VarPtr>{rv}, Sp());
    auto for2 = std::make_shared<ForStmt>(i, zero, ten, step, std::vector<IterArgPtr>{ia2}, yield,
                                          std::vector<VarPtr>{rv}, Sp());

    EXPECT_FALSE(structural_equal(Node(for1), Node(for2)));
    EXPECT_THROW(assert_structural_equal(Node(for1), Node(for2)), npu::tile_fwk::Error);
}

// ============================================================================
// IsConstIntTypeContext: INT64/INDEX equivalence in ConstInt type field
// ============================================================================

TEST_F(IRStructEqExprTest, TestConstIntTypeInt64IndexNotEquivalent)
{
    auto ci1 = std::make_shared<ConstInt>(42, DataType::INT64, Sp());
    auto ci2 = std::make_shared<ConstInt>(42, DataType::INDEX, Sp());

    EXPECT_FALSE(structural_equal(Node(ci1), Node(ci2)));
}

// ============================================================================
// AssertMode branches: specific error messages
// ============================================================================

TEST_F(IRStructEqAssertTest, TestAssertConstIntValueMismatch)
{
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto b = std::make_shared<ConstInt>(2, DataType::INT32, Sp());

    try {
        assert_structural_equal(Node(a), Node(b));
        FAIL() << "Expected npu::tile_fwk::Error";
    } catch (const npu::tile_fwk::Error& e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("ASSERT FAILED") != std::string::npos);
        EXPECT_TRUE(msg.find("INVALID_VAL") != std::string::npos);
        EXPECT_TRUE(msg.find("Structural equality assertion failed") != std::string::npos);
        EXPECT_TRUE(msg.find("int64_t") != std::string::npos);
    }
}

TEST_F(IRStructEqAssertTest, TestAssertConstFloatValueMismatch)
{
    auto a = std::make_shared<ConstFloat>(1.0, DataType::FP32, Sp());
    auto b = std::make_shared<ConstFloat>(2.0, DataType::FP32, Sp());

    try {
        assert_structural_equal(Node(a), Node(b));
        FAIL() << "Expected npu::tile_fwk::Error";
    } catch (const npu::tile_fwk::Error& e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("double") != std::string::npos);
    }
}

TEST_F(IRStructEqAssertTest, TestAssertSectionKindMismatch)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto body = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(1, DataType::INT32, Sp()), Sp());

    try {
        assert_structural_equal(Node(std::make_shared<SectionStmt>(SectionKind::Vector, body, Sp())),
                                Node(std::make_shared<SectionStmt>(SectionKind::Cube, body, Sp())));
        FAIL() << "Expected npu::tile_fwk::Error";
    } catch (const npu::tile_fwk::Error& e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("SectionKind") != std::string::npos);
    }
}

TEST_F(IRStructEqAssertTest, TestAssertVarTypeMismatch)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto z = std::make_shared<Var>("z", Scalar(DataType::FP32), Sp());
    auto v1 = std::make_shared<ConstInt>(1, DataType::INT32, Sp());

    try {
        assert_structural_equal(Node(std::make_shared<AssignStmt>(x, v1, Sp())),
                                Node(std::make_shared<AssignStmt>(z, v1, Sp())));
        FAIL() << "Expected npu::tile_fwk::Error";
    } catch (const npu::tile_fwk::Error& e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("Variable") != std::string::npos || msg.find("type") != std::string::npos);
    }
}

TEST_F(IRStructEqAssertTest, TestAssertVarMappingInconsistent)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto y1 = std::make_shared<Var>("y", Scalar(DataType::INT32), Sp());
    auto y2 = std::make_shared<Var>("z", Scalar(DataType::INT32), Sp());
    auto v1 = std::make_shared<ConstInt>(1, DataType::INT32, Sp());

    auto seq1 = std::make_shared<SeqStmts>(
        std::vector<StmtPtr>{
            std::make_shared<EvalStmt>(std::make_shared<Call>("op", std::vector<ExprPtr>{x}, Sp()), Sp()),
            std::make_shared<EvalStmt>(std::make_shared<Call>("op", std::vector<ExprPtr>{x}, Sp()), Sp())},
        Sp());
    auto seq2 = std::make_shared<SeqStmts>(
        std::vector<StmtPtr>{
            std::make_shared<EvalStmt>(std::make_shared<Call>("op", std::vector<ExprPtr>{y1}, Sp()), Sp()),
            std::make_shared<EvalStmt>(std::make_shared<Call>("op", std::vector<ExprPtr>{y2}, Sp()), Sp())},
        Sp());

    try {
        assert_structural_equal(Node(seq1), Node(seq2));
        FAIL() << "Expected npu::tile_fwk::Error";
    } catch (const npu::tile_fwk::Error& e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("Variable") != std::string::npos || msg.find("mapping") != std::string::npos);
    }
}

TEST_F(IRStructEqAssertTest, TestAssertHardwareInfoSlayoutMismatch)
{
    auto d16 = std::make_shared<ConstInt>(16, DataType::INT64, Sp());

    HardwareInfo hw1(TileLayout::row_major, TileLayout::none_box, 512, TilePad::null);
    HardwareInfo hw2(TileLayout::row_major, TileLayout::row_major, 512, TilePad::null);

    auto tl1 = std::make_shared<TileType>(std::vector<ExprPtr>{d16}, DataType::FP32, std::nullopt, std::nullopt, hw1);
    auto tl2 = std::make_shared<TileType>(std::vector<ExprPtr>{d16}, DataType::FP32, std::nullopt, std::nullopt, hw2);

    try {
        assert_structural_equal(tl1, tl2);
        FAIL() << "Expected npu::tile_fwk::Error";
    } catch (const npu::tile_fwk::Error& e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("HardwareInfo") != std::string::npos || msg.find("slayout") != std::string::npos);
    }
}

TEST_F(IRStructEqAssertTest, TestAssertHardwareInfoFractalMismatch)
{
    auto d16 = std::make_shared<ConstInt>(16, DataType::INT64, Sp());

    HardwareInfo hw1(TileLayout::row_major, TileLayout::none_box, 512, TilePad::null);
    HardwareInfo hw2(TileLayout::row_major, TileLayout::none_box, 1024, TilePad::null);

    auto tl1 = std::make_shared<TileType>(std::vector<ExprPtr>{d16}, DataType::FP32, std::nullopt, std::nullopt, hw1);
    auto tl2 = std::make_shared<TileType>(std::vector<ExprPtr>{d16}, DataType::FP32, std::nullopt, std::nullopt, hw2);

    try {
        assert_structural_equal(tl1, tl2);
        FAIL() << "Expected npu::tile_fwk::Error";
    } catch (const npu::tile_fwk::Error& e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("HardwareInfo") != std::string::npos || msg.find("fractal") != std::string::npos);
    }
}

TEST_F(IRStructEqAssertTest, TestAssertHardwareInfoPadMismatch)
{
    auto d16 = std::make_shared<ConstInt>(16, DataType::INT64, Sp());

    HardwareInfo hw1(TileLayout::row_major, TileLayout::none_box, 512, TilePad::null);
    HardwareInfo hw2(TileLayout::row_major, TileLayout::none_box, 512, TilePad::zero);

    auto tl1 = std::make_shared<TileType>(std::vector<ExprPtr>{d16}, DataType::FP32, std::nullopt, std::nullopt, hw1);
    auto tl2 = std::make_shared<TileType>(std::vector<ExprPtr>{d16}, DataType::FP32, std::nullopt, std::nullopt, hw2);

    try {
        assert_structural_equal(tl1, tl2);
        FAIL() << "Expected npu::tile_fwk::Error";
    } catch (const npu::tile_fwk::Error& e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("HardwareInfo") != std::string::npos || msg.find("pad") != std::string::npos);
    }
}

TEST_F(IRStructEqAssertTest, TestAssertTileViewStartOffsetMismatch)
{
    auto d16 = std::make_shared<ConstInt>(16, DataType::INT64, Sp());
    auto off0 = std::make_shared<ConstInt>(0, DataType::INT64, Sp());
    auto off100 = std::make_shared<ConstInt>(100, DataType::INT64, Sp());

    TileView tv1({d16}, {d16}, off0);
    TileView tv2({d16}, {d16}, off100);

    auto tl1 = std::make_shared<TileType>(std::vector<ExprPtr>{d16}, DataType::FP32, std::nullopt, tv1);
    auto tl2 = std::make_shared<TileType>(std::vector<ExprPtr>{d16}, DataType::FP32, std::nullopt, tv2);

    try {
        assert_structural_equal(tl1, tl2);
        FAIL() << "Expected npu::tile_fwk::Error";
    } catch (const npu::tile_fwk::Error& e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("TileView") != std::string::npos || msg.find("start_offset") != std::string::npos ||
                    msg.find("int64_t") != std::string::npos);
    }
}

TEST_F(IRStructEqAssertTest, TestAssertCallNameMismatch)
{
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto c1 = std::make_shared<Call>("op_a", std::vector<ExprPtr>{a}, Sp());
    auto c2 = std::make_shared<Call>("op_b", std::vector<ExprPtr>{a}, Sp());

    try {
        assert_structural_equal(Node(c1), Node(c2));
        FAIL() << "Expected npu::tile_fwk::Error";
    } catch (const npu::tile_fwk::Error& e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("String") != std::string::npos || msg.find("op_a") != std::string::npos);
    }
}

TEST_F(IRStructEqAssertTest, TestAssertTupleTypeElementMismatch)
{
    auto s1 = std::make_shared<ScalarType>(DataType::INT32);
    auto s2 = std::make_shared<ScalarType>(DataType::FP32);

    auto t1 = std::make_shared<TupleType>(std::vector<TypePtr>{s1, s1});
    auto t2 = std::make_shared<TupleType>(std::vector<TypePtr>{s1, s2});

    try {
        assert_structural_equal(t1, t2);
        FAIL() << "Expected npu::tile_fwk::Error";
    } catch (const npu::tile_fwk::Error& e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("Scalar") != std::string::npos || msg.find("dtype") != std::string::npos);
    }
}

TEST_F(IRStructEqAssertTest, TestAssertIterArgIterVarMismatch)
{
    auto i = std::make_shared<Var>("i", Scalar(DataType::INT32), Sp());
    auto zero = std::make_shared<ConstInt>(0, DataType::INT32, Sp());
    auto ten = std::make_shared<ConstInt>(10, DataType::INT32, Sp());
    auto step = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{zero}, Sp());

    auto ia1 = std::make_shared<IterArg>("acc", Scalar(DataType::INT32), zero, Sp());
    auto ia2 = std::make_shared<IterArg>("acc", Scalar(DataType::FP32), zero, Sp());
    auto rv = std::make_shared<Var>("out", Scalar(DataType::INT32), Sp());

    auto for1 = std::make_shared<ForStmt>(i, zero, ten, step, std::vector<IterArgPtr>{ia1}, yield,
                                          std::vector<VarPtr>{rv}, Sp());
    auto for2 = std::make_shared<ForStmt>(i, zero, ten, step, std::vector<IterArgPtr>{ia2}, yield,
                                          std::vector<VarPtr>{rv}, Sp());

    EXPECT_THROW(assert_structural_equal(Node(for1), Node(for2)), npu::tile_fwk::Error);
}

TEST_F(IRStructEqAssertTest, TestAssertTileTypeTileViewPresenceMismatch)
{
    auto d16 = std::make_shared<ConstInt>(16, DataType::INT64, Sp());
    auto off0 = std::make_shared<ConstInt>(0, DataType::INT64, Sp());
    TileView tv({d16}, {d16}, off0);

    auto t1 = std::make_shared<TileType>(std::vector<ExprPtr>{d16}, DataType::FP32, std::nullopt, tv);
    auto t2 = std::make_shared<TileType>(std::vector<ExprPtr>{d16}, DataType::FP32);
    EXPECT_THROW(assert_structural_equal(t1, t2), npu::tile_fwk::Error);
}

TEST_F(IRStructEqAssertTest, TestAssertTileTypeHardwareInfoPresenceMismatch)
{
    auto d16 = std::make_shared<ConstInt>(16, DataType::INT64, Sp());
    HardwareInfo hw(TileLayout::row_major);

    auto t1 = std::make_shared<TileType>(std::vector<ExprPtr>{d16}, DataType::FP32, std::nullopt, std::nullopt, hw);
    auto t2 = std::make_shared<TileType>(std::vector<ExprPtr>{d16}, DataType::FP32);
    EXPECT_THROW(assert_structural_equal(t1, t2), npu::tile_fwk::Error);
}

// ============================================================================
// AssertMode: EqualVar no-auto-mapping branches
// ============================================================================

TEST_F(IRStructEqAssertTest, TestAssertVarNoAutoMapping_PointerMismatch)
{
    auto x1 = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto x2 = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());

    auto c1 = std::make_shared<Call>("op", std::vector<ExprPtr>{x1}, Sp());
    auto c2 = std::make_shared<Call>("op", std::vector<ExprPtr>{x2}, Sp());

    EXPECT_THROW(assert_structural_equal(Node(c1), Node(c2), false), npu::tile_fwk::Error);
}

TEST_F(IRStructEqAssertTest, TestAssertVarDefFieldMappingInconsistent)
{
    // Covers "Variable mapping inconsistent (without auto-mapping)" branch (line 975).
    // With enable_auto_mapping=false at construction:
    //   - DEF fields (iterArgs) force auto_mapping=true, establishing a1↔b1 and a2↔b2
    //   - USUAL field (body) keeps auto_mapping=false; yield references a1 vs b2 (crossed),
    //     so EqualVar(a1,b2) finds a1→b1 in lhs_map AND b2→a2 in rhs_map, both present but inconsistent.
    auto zero = std::make_shared<ConstInt>(0, DataType::INT32, Sp());
    auto ten = std::make_shared<ConstInt>(10, DataType::INT32, Sp());
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, Sp());

    auto a1 = std::make_shared<Var>("a", Scalar(DataType::INT32), Sp());
    auto a2 = std::make_shared<Var>("a2", Scalar(DataType::INT32), Sp());
    auto b1 = std::make_shared<Var>("b", Scalar(DataType::INT32), Sp());
    auto b2 = std::make_shared<Var>("b2", Scalar(DataType::INT32), Sp());

    auto ia_lhs1 = std::make_shared<IterArg>(a1, zero);
    auto ia_lhs2 = std::make_shared<IterArg>(a2, zero);
    auto ia_rhs1 = std::make_shared<IterArg>(b1, zero);
    auto ia_rhs2 = std::make_shared<IterArg>(b2, zero);

    auto loop_i1 = std::make_shared<Var>("i", Scalar(DataType::INT64), Sp());
    auto loop_i2 = std::make_shared<Var>("i", Scalar(DataType::INT64), Sp());

    auto yield1 = std::make_shared<YieldStmt>(std::vector<ExprPtr>{a1, a2}, Sp());
    auto yield2 = std::make_shared<YieldStmt>(std::vector<ExprPtr>{b2, b1}, Sp());

    auto rv_lhs1 = std::make_shared<Var>("r1", Scalar(DataType::INT32), Sp());
    auto rv_lhs2 = std::make_shared<Var>("r2", Scalar(DataType::INT32), Sp());
    auto rv_rhs1 = std::make_shared<Var>("r1", Scalar(DataType::INT32), Sp());
    auto rv_rhs2 = std::make_shared<Var>("r2", Scalar(DataType::INT32), Sp());

    auto for_lhs = std::make_shared<ForStmt>(loop_i1, zero, ten, one, std::vector<IterArgPtr>{ia_lhs1, ia_lhs2}, yield1,
                                             std::vector<VarPtr>{rv_lhs1, rv_lhs2}, Sp());
    auto for_rhs = std::make_shared<ForStmt>(loop_i2, zero, ten, one, std::vector<IterArgPtr>{ia_rhs1, ia_rhs2}, yield2,
                                             std::vector<VarPtr>{rv_rhs1, rv_rhs2}, Sp());

    EXPECT_FALSE(structural_equal(Node(for_lhs), Node(for_rhs), false));
    EXPECT_THROW(assert_structural_equal(Node(for_lhs), Node(for_rhs), false), npu::tile_fwk::Error);
}

} // namespace ir
} // namespace pypto
