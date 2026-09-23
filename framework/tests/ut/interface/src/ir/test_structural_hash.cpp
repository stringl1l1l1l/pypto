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
 * \file test_structural_hash.cpp
 * \brief Coverage tests for structural_hash.cpp
 */

#include "gtest/gtest.h"

#include "ir/transforms/structural_comparison.h"
#include "op/test_op_helpers.h"

namespace pypto {
namespace ir {

using test_helpers::Node;
using test_helpers::Scalar;
using test_helpers::Sp;

class IRStructHashTest : public testing::Test {};

// ============================================================================
// Var hashing
// ============================================================================

TEST_F(IRStructHashTest, TestHashVarDeterministic)
{
    auto a = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto b = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    EXPECT_EQ(structural_hash(Node(a)), structural_hash(Node(b)));
}

TEST_F(IRStructHashTest, TestHashDifferentVarsDifferentHash)
{
    auto a = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto b = std::make_shared<Var>("y", Scalar(DataType::INT32), Sp());
    EXPECT_NE(structural_hash(Node(a)), structural_hash(Node(b)));
}

TEST_F(IRStructHashTest, TestHashVarAutoMappingEqual)
{
    auto a = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto b = std::make_shared<Var>("y", Scalar(DataType::INT32), Sp());
    EXPECT_EQ(structural_hash(Node(a), true), structural_hash(Node(b), true));
}

TEST_F(IRStructHashTest, TestHashVarNoAutoMappingByName)
{
    auto a = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto b = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    EXPECT_EQ(structural_hash(Node(a), false), structural_hash(Node(b), false));
}

// ============================================================================
// Expression node hashing
// ============================================================================

TEST_F(IRStructHashTest, TestHashConstInt)
{
    auto a = std::make_shared<ConstInt>(42, DataType::INT32, Sp());
    auto b = std::make_shared<ConstInt>(42, DataType::INT32, Sp());
    auto c = std::make_shared<ConstInt>(99, DataType::INT32, Sp());
    EXPECT_EQ(structural_hash(Node(a)), structural_hash(Node(b)));
    EXPECT_NE(structural_hash(Node(a)), structural_hash(Node(c)));
}

TEST_F(IRStructHashTest, TestHashConstFloat)
{
    auto a = std::make_shared<ConstFloat>(3.14, DataType::FP32, Sp());
    auto b = std::make_shared<ConstFloat>(3.14, DataType::FP32, Sp());
    auto c = std::make_shared<ConstFloat>(2.71, DataType::FP32, Sp());
    EXPECT_EQ(structural_hash(Node(a)), structural_hash(Node(b)));
    EXPECT_NE(structural_hash(Node(a)), structural_hash(Node(c)));
}

TEST_F(IRStructHashTest, TestHashConstBool)
{
    auto t = std::make_shared<ConstBool>(true, Sp());
    auto f = std::make_shared<ConstBool>(false, Sp());
    auto t2 = std::make_shared<ConstBool>(true, Sp());
    EXPECT_EQ(structural_hash(Node(t)), structural_hash(Node(t2)));
    EXPECT_NE(structural_hash(Node(t)), structural_hash(Node(f)));
}

TEST_F(IRStructHashTest, TestHashCall)
{
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto c1 = std::make_shared<Call>("op", std::vector<ExprPtr>{a}, Sp());
    auto c2 = std::make_shared<Call>("op", std::vector<ExprPtr>{a}, Sp());
    auto c3 = std::make_shared<Call>("other", std::vector<ExprPtr>{a}, Sp());
    EXPECT_EQ(structural_hash(Node(c1)), structural_hash(Node(c2)));
    EXPECT_NE(structural_hash(Node(c1)), structural_hash(Node(c3)));
}

TEST_F(IRStructHashTest, TestHashCallWithKwargs)
{
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    std::vector<std::pair<std::string, std::any>> kw1 = {{"mode", std::any(std::string("round"))}};
    std::vector<std::pair<std::string, std::any>> kw2 = {{"mode", std::any(std::string("round"))}};
    std::vector<std::pair<std::string, std::any>> kw3 = {{"mode", std::any(std::string("floor"))}};

    auto c1 = std::make_shared<Call>("op", std::vector<ExprPtr>{a}, kw1, Scalar(DataType::INT32), Sp());
    auto c2 = std::make_shared<Call>("op", std::vector<ExprPtr>{a}, kw2, Scalar(DataType::INT32), Sp());
    auto c3 = std::make_shared<Call>("op", std::vector<ExprPtr>{a}, kw3, Scalar(DataType::INT32), Sp());
    EXPECT_EQ(structural_hash(Node(c1)), structural_hash(Node(c2)));
    EXPECT_NE(structural_hash(Node(c1)), structural_hash(Node(c3)));
}

TEST_F(IRStructHashTest, TestHashCallKwargsAllTypes)
{
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto expect_hash_contract = [&](std::any value, std::any same_value, std::any different_value) {
        auto make_call = [&](std::any kwarg_value) {
            std::vector<std::pair<std::string, std::any>> kwargs = {{"x", std::move(kwarg_value)}};
            return std::make_shared<Call>("op", std::vector<ExprPtr>{a}, kwargs, Scalar(DataType::INT32), Sp());
        };
        auto lhs = make_call(std::move(value));
        auto equal_rhs = make_call(std::move(same_value));
        auto different_rhs = make_call(std::move(different_value));
        EXPECT_EQ(structural_hash(Node(lhs)), structural_hash(Node(equal_rhs)));
        EXPECT_NE(structural_hash(Node(lhs)), structural_hash(Node(different_rhs)));
    };

    expect_hash_contract(int(1), int(1), int(2));
    expect_hash_contract(true, true, false);
    expect_hash_contract(std::string("a"), std::string("a"), std::string("b"));
    expect_hash_contract(1.5, 1.5, 2.5);
    expect_hash_contract(1.5f, 1.5f, 2.5f);
    expect_hash_contract(DataType::FP32, DataType::FP32, DataType::FP16);
    expect_hash_contract(std::vector<int>{1, 2, 3}, std::vector<int>{1, 2, 3}, std::vector<int>{1, 2, 4});
    expect_hash_contract(std::vector<std::string>{"a", "b"}, std::vector<std::string>{"a", "b"},
                         std::vector<std::string>{"a", "c"});
}

TEST_F(IRStructHashTest, TestHashMakeTuple)
{
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto b = std::make_shared<ConstInt>(2, DataType::INT32, Sp());
    auto t1 = std::make_shared<MakeTuple>(std::vector<ExprPtr>{a, b}, Sp());
    auto t2 = std::make_shared<MakeTuple>(std::vector<ExprPtr>{a, b}, Sp());
    EXPECT_EQ(structural_hash(Node(t1)), structural_hash(Node(t2)));
}

TEST_F(IRStructHashTest, TestHashGetItemExpr)
{
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto b = std::make_shared<ConstInt>(2, DataType::INT32, Sp());
    auto tup = std::make_shared<MakeTuple>(std::vector<ExprPtr>{a, b}, Sp());
    auto idx = std::make_shared<ConstInt>(0, DataType::INDEX, Sp());
    auto g1 = std::make_shared<GetItemExpr>(tup, idx, Sp());
    auto g2 = std::make_shared<GetItemExpr>(tup, idx, Sp());
    EXPECT_EQ(structural_hash(Node(g1)), structural_hash(Node(g2)));
}

TEST_F(IRStructHashTest, TestHashBinaryExpr)
{
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto b = std::make_shared<ConstInt>(2, DataType::INT32, Sp());
    auto add1 = std::make_shared<Add>(a, b, DataType::INT32, Sp());
    auto add2 = std::make_shared<Add>(a, b, DataType::INT32, Sp());
    auto sub = std::make_shared<Sub>(a, b, DataType::INT32, Sp());
    EXPECT_EQ(structural_hash(Node(add1)), structural_hash(Node(add2)));
    EXPECT_NE(structural_hash(Node(add1)), structural_hash(Node(sub)));
}

TEST_F(IRStructHashTest, TestHashUnaryExpr)
{
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto neg1 = std::make_shared<Neg>(a, DataType::INT32, Sp());
    auto neg2 = std::make_shared<Neg>(a, DataType::INT32, Sp());
    auto abs_e = std::make_shared<Abs>(a, DataType::INT32, Sp());
    EXPECT_EQ(structural_hash(Node(neg1)), structural_hash(Node(neg2)));
    EXPECT_NE(structural_hash(Node(neg1)), structural_hash(Node(abs_e)));
}

// ============================================================================
// Statement node hashing
// ============================================================================

TEST_F(IRStructHashTest, TestHashAssignStmt)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto v = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto a1 = std::make_shared<AssignStmt>(x, v, Sp());
    auto a2 = std::make_shared<AssignStmt>(x, v, Sp());
    EXPECT_EQ(structural_hash(Node(a1)), structural_hash(Node(a2)));
}

TEST_F(IRStructHashTest, TestHashYieldAndReturn)
{
    auto v = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto y1 = std::make_shared<YieldStmt>(std::vector<ExprPtr>{v}, Sp());
    auto y2 = std::make_shared<YieldStmt>(std::vector<ExprPtr>{v}, Sp());
    EXPECT_EQ(structural_hash(Node(y1)), structural_hash(Node(y2)));

    auto r1 = std::make_shared<ReturnStmt>(std::vector<ExprPtr>{v}, Sp());
    auto r2 = std::make_shared<ReturnStmt>(std::vector<ExprPtr>{v}, Sp());
    EXPECT_EQ(structural_hash(Node(r1)), structural_hash(Node(r2)));
}

TEST_F(IRStructHashTest, TestHashIfStmt)
{
    auto cond = std::make_shared<ConstBool>(true, Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, Sp());
    auto i1 = std::make_shared<IfStmt>(cond, yield, std::nullopt, std::vector<VarPtr>{}, Sp());
    auto i2 = std::make_shared<IfStmt>(cond, yield, std::nullopt, std::vector<VarPtr>{}, Sp());
    EXPECT_EQ(structural_hash(Node(i1)), structural_hash(Node(i2)));

    auto i3 = std::make_shared<IfStmt>(cond, yield, yield, std::vector<VarPtr>{}, Sp());
    EXPECT_NE(structural_hash(Node(i1)), structural_hash(Node(i3)));
}

TEST_F(IRStructHashTest, TestHashForStmt)
{
    auto i = std::make_shared<Var>("i", Scalar(DataType::INT32), Sp());
    auto zero = std::make_shared<ConstInt>(0, DataType::INT32, Sp());
    auto ten = std::make_shared<ConstInt>(10, DataType::INT32, Sp());
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, Sp());

    auto f1 = std::make_shared<ForStmt>(i, zero, ten, one, std::vector<IterArgPtr>{}, yield, std::vector<VarPtr>{},
                                        Sp());
    auto f2 = std::make_shared<ForStmt>(i, zero, ten, one, std::vector<IterArgPtr>{}, yield, std::vector<VarPtr>{},
                                        Sp());
    EXPECT_EQ(structural_hash(Node(f1)), structural_hash(Node(f2)));
}

TEST_F(IRStructHashTest, TestHashForStmtWithIterArgs)
{
    auto i = std::make_shared<Var>("i", Scalar(DataType::INT32), Sp());
    auto zero = std::make_shared<ConstInt>(0, DataType::INT32, Sp());
    auto ten = std::make_shared<ConstInt>(10, DataType::INT32, Sp());
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto ia = std::make_shared<IterArg>("acc", Scalar(DataType::INT32), zero, Sp());
    auto rv = std::make_shared<Var>("out", Scalar(DataType::INT32), Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{zero}, Sp());

    auto f1 = std::make_shared<ForStmt>(i, zero, ten, one, std::vector<IterArgPtr>{ia}, yield, std::vector<VarPtr>{rv},
                                        Sp());
    auto f2 = std::make_shared<ForStmt>(i, zero, ten, one, std::vector<IterArgPtr>{ia}, yield, std::vector<VarPtr>{rv},
                                        Sp());
    EXPECT_EQ(structural_hash(Node(f1)), structural_hash(Node(f2)));
}

TEST_F(IRStructHashTest, TestHashWhileStmt)
{
    auto cond = std::make_shared<ConstBool>(true, Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, Sp());
    auto w1 = std::make_shared<WhileStmt>(cond, std::vector<IterArgPtr>{}, yield, std::vector<VarPtr>{}, Sp());
    auto w2 = std::make_shared<WhileStmt>(cond, std::vector<IterArgPtr>{}, yield, std::vector<VarPtr>{}, Sp());
    EXPECT_EQ(structural_hash(Node(w1)), structural_hash(Node(w2)));
}

TEST_F(IRStructHashTest, TestHashSectionStmt)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto v = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto body = std::make_shared<AssignStmt>(x, v, Sp());
    auto s1 = std::make_shared<SectionStmt>(SectionKind::Vector, body, Sp());
    auto s2 = std::make_shared<SectionStmt>(SectionKind::Vector, body, Sp());
    auto s3 = std::make_shared<SectionStmt>(SectionKind::Cube, body, Sp());
    EXPECT_EQ(structural_hash(Node(s1)), structural_hash(Node(s2)));
    EXPECT_NE(structural_hash(Node(s1)), structural_hash(Node(s3)));
}

TEST_F(IRStructHashTest, TestHashSeqStmts)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto v = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto a = std::make_shared<AssignStmt>(x, v, Sp());
    auto seq1 = std::make_shared<SeqStmts>(std::vector<StmtPtr>{a}, Sp());
    auto seq2 = std::make_shared<SeqStmts>(std::vector<StmtPtr>{a}, Sp());
    EXPECT_EQ(structural_hash(Node(seq1)), structural_hash(Node(seq2)));
}

TEST_F(IRStructHashTest, TestHashEvalBreakContinue)
{
    auto call = std::make_shared<Call>("op", std::vector<ExprPtr>{}, Sp());
    auto e1 = std::make_shared<EvalStmt>(call, Sp());
    auto e2 = std::make_shared<EvalStmt>(call, Sp());
    EXPECT_EQ(structural_hash(Node(e1)), structural_hash(Node(e2)));

    auto b1 = std::make_shared<BreakStmt>(Sp());
    auto b2 = std::make_shared<BreakStmt>(Sp());
    EXPECT_EQ(structural_hash(Node(b1)), structural_hash(Node(b2)));

    auto c1 = std::make_shared<ContinueStmt>(Sp());
    auto c2 = std::make_shared<ContinueStmt>(Sp());
    EXPECT_EQ(structural_hash(Node(c1)), structural_hash(Node(c2)));
}

// ============================================================================
// Function and Program hashing
// ============================================================================

TEST_F(IRStructHashTest, TestHashFunction)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto v = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto body = std::make_shared<AssignStmt>(x, v, Sp());
    auto f1 = std::make_shared<Function>("f", std::vector<VarPtr>{x}, std::vector<TypePtr>{Scalar(DataType::INT32)},
                                         body, Sp());
    auto f2 = std::make_shared<Function>("f", std::vector<VarPtr>{x}, std::vector<TypePtr>{Scalar(DataType::INT32)},
                                         body, Sp());
    EXPECT_EQ(structural_hash(Node(f1)), structural_hash(Node(f2)));
}

TEST_F(IRStructHashTest, TestHashFunctionMaxThreads)
{
    auto body = std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, Sp());
    auto f128a = std::make_shared<Function>("f", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body, Sp(),
                                            FunctionType::SIMT_VF, false,
                                            std::vector<std::pair<std::string, std::any>>{{kMaxThreadsAttr, 128}});
    auto f128b = std::make_shared<Function>("f", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body, Sp(),
                                            FunctionType::SIMT_VF, false,
                                            std::vector<std::pair<std::string, std::any>>{{kMaxThreadsAttr, 128}});
    auto f256 = std::make_shared<Function>("f", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body, Sp(),
                                           FunctionType::SIMT_VF, false,
                                           std::vector<std::pair<std::string, std::any>>{{kMaxThreadsAttr, 256}});
    auto unbounded = std::make_shared<Function>("f", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body, Sp(),
                                                FunctionType::SIMT_VF);

    EXPECT_EQ(structural_hash(Node(f128a)), structural_hash(Node(f128b)));
    EXPECT_NE(structural_hash(Node(f128a)), structural_hash(Node(f256)));
    EXPECT_NE(structural_hash(Node(f128a)), structural_hash(Node(unbounded)));
}

TEST_F(IRStructHashTest, TestHashProgram)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto body = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(1, DataType::INT32, Sp()), Sp());
    auto f = std::make_shared<Function>("f", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body, Sp());
    auto p1 = std::make_shared<Program>(std::vector<FunctionPtr>{f}, "prog", Sp());
    auto p2 = std::make_shared<Program>(std::vector<FunctionPtr>{f}, "prog", Sp());
    EXPECT_EQ(structural_hash(Node(p1)), structural_hash(Node(p2)));
}

// ============================================================================
// MemRef node hashing
// ============================================================================

TEST_F(IRStructHashTest, TestHashMemRefNode)
{
    auto off1 = std::make_shared<ConstInt>(0, DataType::INT64, Sp());
    auto off2 = std::make_shared<ConstInt>(0, DataType::INT64, Sp());
    auto mr1 = std::make_shared<MemRef>(MemorySpace::DDR, off1, 1024, Sp());
    auto mr2 = std::make_shared<MemRef>(MemorySpace::DDR, off2, 1024, Sp());
    EXPECT_EQ(structural_hash(Node(mr1)), structural_hash(Node(mr2)));

    auto off3 = std::make_shared<ConstInt>(100, DataType::INT64, Sp());
    auto mr3 = std::make_shared<MemRef>(MemorySpace::DDR, off3, 1024, Sp());
    EXPECT_NE(structural_hash(Node(mr1)), structural_hash(Node(mr3)));
}

// ============================================================================
// Type hashing — advanced paths
// ============================================================================

TEST_F(IRStructHashTest, TestHashScalarTypeDeterministicAndDifferent)
{
    auto t1 = std::make_shared<ScalarType>(DataType::INT32);
    auto t2 = std::make_shared<ScalarType>(DataType::INT32);
    auto t3 = std::make_shared<ScalarType>(DataType::FP32);
    EXPECT_EQ(structural_hash(t1), structural_hash(t2));
    EXPECT_NE(structural_hash(t1), structural_hash(t3));
}

TEST_F(IRStructHashTest, TestHashTensorTypeWithVarShape)
{
    auto d = std::make_shared<Var>("N", Scalar(DataType::INT64), Sp());
    auto t = std::make_shared<TensorType>(std::vector<ExprPtr>{d}, DataType::FP32);
    auto d2 = std::make_shared<Var>("N", Scalar(DataType::INT64), Sp());
    auto t2 = std::make_shared<TensorType>(std::vector<ExprPtr>{d2}, DataType::FP32);
    EXPECT_EQ(structural_hash(t, false), structural_hash(t2, false));
}

TEST_F(IRStructHashTest, TestHashTileType)
{
    auto d = std::make_shared<Var>("T", Scalar(DataType::INT64), Sp());
    auto t1 = std::make_shared<TileType>(std::vector<ExprPtr>{d}, DataType::FP32);
    auto t2 = std::make_shared<TileType>(std::vector<ExprPtr>{d}, DataType::FP16);
    EXPECT_NE(structural_hash(t1), structural_hash(t2));
}

TEST_F(IRStructHashTest, TestHashTileTypeWithTileView)
{
    auto d16 = std::make_shared<ConstInt>(16, DataType::INT64, Sp());
    auto off0 = std::make_shared<ConstInt>(0, DataType::INT64, Sp());
    TileView tv({d16}, {d16}, off0);

    auto t1 = std::make_shared<TileType>(std::vector<ExprPtr>{d16}, DataType::FP32, std::nullopt, tv);
    auto t2 = std::make_shared<TileType>(std::vector<ExprPtr>{d16}, DataType::FP32, std::nullopt, tv);
    EXPECT_EQ(structural_hash(t1), structural_hash(t2));

    auto t3 = std::make_shared<TileType>(std::vector<ExprPtr>{d16}, DataType::FP32);
    EXPECT_NE(structural_hash(t1), structural_hash(t3));
}

TEST_F(IRStructHashTest, TestHashTileTypeWithHardwareInfo)
{
    auto d16 = std::make_shared<ConstInt>(16, DataType::INT64, Sp());
    HardwareInfo hw1(TileLayout::row_major, TileLayout::none_box, 512, TilePad::null);
    HardwareInfo hw2(TileLayout::row_major, TileLayout::none_box, 512, TilePad::null);
    HardwareInfo hw3(TileLayout::col_major, TileLayout::none_box, 512, TilePad::null);
    HardwareInfo hw4(TileLayout::row_major, TileLayout::none_box, 512, TilePad::null, CompactMode::normal);

    auto t1 = std::make_shared<TileType>(std::vector<ExprPtr>{d16}, DataType::FP32, std::nullopt, std::nullopt, hw1);
    auto t2 = std::make_shared<TileType>(std::vector<ExprPtr>{d16}, DataType::FP32, std::nullopt, std::nullopt, hw2);
    auto t3 = std::make_shared<TileType>(std::vector<ExprPtr>{d16}, DataType::FP32, std::nullopt, std::nullopt, hw3);
    auto t4 = std::make_shared<TileType>(std::vector<ExprPtr>{d16}, DataType::FP32, std::nullopt, std::nullopt, hw4);
    EXPECT_EQ(structural_hash(t1), structural_hash(t2));
    EXPECT_NE(structural_hash(t1), structural_hash(t3));
    EXPECT_NE(structural_hash(t1), structural_hash(t4));
}

TEST_F(IRStructHashTest, TestHashTupleType)
{
    auto t1 = std::make_shared<TupleType>(std::vector<TypePtr>{Scalar(DataType::INT32)});
    auto t2 = std::make_shared<TupleType>(std::vector<TypePtr>{Scalar(DataType::INT32)});
    auto t3 = std::make_shared<TupleType>(std::vector<TypePtr>{Scalar(DataType::FP32)});
    EXPECT_EQ(structural_hash(t1), structural_hash(t2));
    EXPECT_NE(structural_hash(t1), structural_hash(t3));
}

TEST_F(IRStructHashTest, TestHashMemRefAndUnknownType)
{
    auto memref1 = GetMemRefType();
    auto memref2 = GetMemRefType();
    auto unknown1 = std::make_shared<UnknownType>();
    auto unknown2 = std::make_shared<UnknownType>();
    EXPECT_EQ(structural_hash(memref1), structural_hash(memref2));
    EXPECT_EQ(structural_hash(unknown1), structural_hash(unknown2));
    EXPECT_NE(structural_hash(memref1), structural_hash(unknown1));
}

// ============================================================================
// LoopVar INT64/INDEX canonicalization
// ============================================================================

TEST_F(IRStructHashTest, TestLoopVarInt64IndexDifferentHash)
{
    auto i1 = std::make_shared<Var>("i", Scalar(DataType::INT64), Sp());
    auto i2 = std::make_shared<Var>("i", Scalar(DataType::INDEX), Sp());
    auto zero = std::make_shared<ConstInt>(0, DataType::INT32, Sp());
    auto ten = std::make_shared<ConstInt>(10, DataType::INT32, Sp());
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, Sp());

    auto f1 = std::make_shared<ForStmt>(i1, zero, ten, one, std::vector<IterArgPtr>{}, yield, std::vector<VarPtr>{},
                                        Sp());
    auto f2 = std::make_shared<ForStmt>(i2, zero, ten, one, std::vector<IterArgPtr>{}, yield, std::vector<VarPtr>{},
                                        Sp());
    EXPECT_NE(structural_hash(Node(f1)), structural_hash(Node(f2)));
}

// ============================================================================
// structural_hash_with_var_identity API
// ============================================================================

TEST_F(IRStructHashTest, TestHashWithVarIdentity)
{
    auto a = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto b = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    EXPECT_EQ(structural_hash_with_var_identity(Node(a), false), structural_hash_with_var_identity(Node(b), false));

    auto t = std::make_shared<ScalarType>(DataType::INT32);
    EXPECT_EQ(structural_hash_with_var_identity(t), structural_hash(t));
}

// ============================================================================
// Hash caching (hash_value_map_ reuse)
// ============================================================================

TEST_F(IRStructHashTest, TestHashCaching)
{
    auto x = std::make_shared<Var>("x", Scalar(DataType::INT32), Sp());
    auto v = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto assign = std::make_shared<AssignStmt>(x, v, Sp());
    auto seq = std::make_shared<SeqStmts>(std::vector<StmtPtr>{assign, assign}, Sp());
    auto h1 = structural_hash(Node(seq));
    auto h2 = structural_hash(Node(seq));
    EXPECT_EQ(h1, h2);
}

} // namespace ir
} // namespace pypto
