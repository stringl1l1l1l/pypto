/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_tile_shape_resolver.cpp
 * \brief Verify TileShapeResolver (both GetInputTileShape and GetOutputTileShape) against the
 *        actual first-cut tile shape produced by the registered TileFunc (via ExpandFunction)
 *        for representative tensor ops.
 *
 * Strategy:
 *  1. Build a single-op tensor graph with the DSL inside FUNCTION. The op picks up
 *     TileShape::Current() as its op-level tile shape. The fixture holds the output Tensor(s)
 *     outside FUNCTION so their underlying LogicalTensor survives.
 *  2. Before expansion: record each input/output operand's underlying RawTensor identity and
 *     compute the resolver's expected per-input and per-output tile shape.
 *     Note: in this harness the tensor-graph op's oOperand is still empty at this point (outputs
 *     are attached lazily during the expansion pipeline), so we attach the fixture-provided
 *     output LogicalTensors to oOperand first; this is exactly what ExpandFunction consumes.
 *  3. Run ExpandFunction on the function; the tensor-graph op is replaced by tile ops whose
 *     input/output operands are Views sharing the original operands' RawTensor.
 *  4. Scan the emitted tile ops in emission order; the first input (output) operand whose
 *     RawTensor matches operand i gives the actual first-cut tile shape for input (output) i.
 *  5. Compare expected vs actual per input and per output.
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <tuple>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/function/function.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation.h"
#include "interface/operation/operation_impl.h"
#include "interface/operation/tile_shape_resolver.h"
#include "interface/program/program.h"
#include "interface/tensor/irbuilder.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/tensor_graph_pass/expand_function.h"
#include "interface/configs/config_manager.h"

using namespace npu::tile_fwk;

namespace {
// Core harness: given a function containing exactly one tensor-graph op and the output Tensors
// produced by that op (held outside FUNCTION), verify that the resolver's per-input AND per-output
// tile shape matches the actual first-cut tile shape emitted by the op's TileFunc during expansion.
static void ExpectResolverMatchesExpansion(Function* fn, std::vector<Tensor> outputs)
{
    ASSERT_NE(fn, nullptr);
    auto preOps = fn->Operations().DuplicatedOpList();
    ASSERT_EQ(preOps.size(), 1u) << "fixture must build exactly one tensor-graph op";
    Operation* op = preOps[0];

    const auto& iOperands = op->GetIOperands();
    const int nIn = static_cast<int>(iOperands.size());
    ASSERT_GE(nIn, 1) << "op has no tensor inputs to verify";

    // The tensor-graph op's oOperand is still empty here (outputs are attached lazily during the
    // expansion pipeline). Attach the fixture-provided output LogicalTensors so both the resolver
    // and ExpandFunction see the real outputs. If oOperand is already populated (e.g. inplace ops
    // whose output is an input), leave it untouched.
    auto& oOperandsMut = op->GetOOperands();
    if (oOperandsMut.empty()) {
        ASSERT_FALSE(outputs.empty()) << "fixture must provide at least one output tensor";
        for (auto& out : outputs) {
            oOperandsMut.push_back(out.GetStorage());
        }
    }
    const auto& oOperands = oOperandsMut;
    const int nOut = static_cast<int>(oOperands.size());
    ASSERT_GE(nOut, 1) << "op has no tensor outputs to verify";

    // Record each input's RawTensor identity and the resolver's expected input tile.
    std::vector<RawTensor*> inRawIds(nIn, nullptr);
    std::vector<std::vector<int64_t>> inExpected(nIn);
    for (int i = 0; i < nIn; ++i) {
        inRawIds[i] = iOperands[i]->GetRawTensor().get();
        inExpected[i] = TileShapeResolver::Instance().GetInputTileShape(*op, i).GetVecTile().tile;
    }

    // Record each output's RawTensor identity and the resolver's expected output tile.
    std::vector<RawTensor*> outRawIds(nOut, nullptr);
    std::vector<std::vector<int64_t>> outExpected(nOut);
    for (int o = 0; o < nOut; ++o) {
        outRawIds[o] = oOperands[o]->GetRawTensor().get();
        outExpected[o] = TileShapeResolver::Instance().GetOutputTileShape(*op, o).GetVecTile().tile;
    }

    // Expand: the tensor-graph op is replaced by tile ops whose input/output operands are Views
    // sharing the original operands' RawTensor. Abort the test if expansion itself fails,
    // otherwise the subsequent scan would capture partial/intermediate shapes and mask
    // the real failure as a spurious tile-shape mismatch.
    ExpandFunction expandFunction;
    ASSERT_NE(expandFunction.RunOnFunction(*fn), Status::FAILED)
        << "ExpandFunction failed; cannot verify tile shapes against a non-expanded graph";

    // Capture the first-cut tile shape per input: the first emitted tile op (in emission
    // order) that consumes a View of input i. TiledXxx emits first-cut tiles first
    // (depth-first recursion starting at offset 0), so the first match is the first cut.
    std::vector<std::vector<int64_t>> inActual(nIn);
    std::vector<bool> inCaptured(nIn, false);
    // Capture the first-cut tile shape per output: the first emitted tile op that produces
    // a View of output o (i.e. o appears among the tile op's output operands).
    std::vector<std::vector<int64_t>> outActual(nOut);
    std::vector<bool> outCaptured(nOut, false);
    for (auto& top : fn->Operations().DuplicatedOpList()) {
        for (auto& operand : top->GetIOperands()) {
            RawTensor* raw = operand->GetRawTensor().get();
            for (int i = 0; i < nIn; ++i) {
                if (!inCaptured[i] && raw == inRawIds[i]) {
                    inActual[i] = operand->GetShape();
                    inCaptured[i] = true;
                }
            }
        }
        for (auto& operand : top->GetOOperands()) {
            RawTensor* raw = operand->GetRawTensor().get();
            for (int o = 0; o < nOut; ++o) {
                if (!outCaptured[o] && raw == outRawIds[o]) {
                    outActual[o] = operand->GetShape();
                    outCaptured[o] = true;
                }
            }
        }
    }

    for (int i = 0; i < nIn; ++i) {
        EXPECT_TRUE(inCaptured[i]) << "input[" << i << "] was never consumed by an emitted tile op";
        if (!inCaptured[i]) {
            continue;
        }
        EXPECT_EQ(inExpected[i], inActual[i])
            << "input[" << i << "] first-cut tile shape mismatch (expected from resolver)";
    }
    for (int o = 0; o < nOut; ++o) {
        EXPECT_TRUE(outCaptured[o]) << "output[" << o << "] was never produced by an emitted tile op";
        if (!outCaptured[o]) {
            continue;
        }
        EXPECT_EQ(outExpected[o], outActual[o])
            << "output[" << o << "] first-cut tile shape mismatch (expected from resolver)";
    }
}
} // namespace

class TileShapeResolverTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {}
};

// ---- Elementwise (default branch) ----
TEST_F(TileShapeResolverTest, Div_Elementwise)
{
    TileShape::Current().SetVecTile(32, 32);
    Tensor a(DT_FP32, {64, 64}, "a");
    Tensor b(DT_FP32, {64, 64}, "b");
    Tensor out;
    FUNCTION("DivCase") { out = Div(a, b); }
    ExpectResolverMatchesExpansion(Program::GetInstance().GetCurrentFunction(), {out});
}

// ---- OP_GATHER: params (axis full) + indices ----
TEST_F(TileShapeResolverTest, Gather_Axis0)
{
    TileShape::Current().SetVecTile(4, 16, 16);
    Tensor params(DT_FP32, {8, 16}, "params");
    Tensor indices(DT_INT32, {4, 16}, "indices");
    Tensor out;
    FUNCTION("GatherCase") { out = Gather(params, indices, 0); }
    ExpectResolverMatchesExpansion(Program::GetInstance().GetCurrentFunction(), {out});
}

// ---- OP_SCATTER: self (axis full) + idx (axis full) + src (axis full) ----
TEST_F(TileShapeResolverTest, Scatter_Axis0)
{
    TileShape::Current().SetVecTile(8, 8);
    Tensor self(DT_FP32, {8, 16}, "self");
    Tensor indices(DT_INT32, {4, 16}, "indices");
    Tensor src(DT_FP32, {4, 16}, "src");
    Tensor out;
    FUNCTION("ScatterCase") { out = Scatter(self, indices, src, 0, ScatterMode::NONE); }
    ExpectResolverMatchesExpansion(Program::GetInstance().GetCurrentFunction(), {out});
}

// ---- OP_WHERE_TT: cond + input + other, all elementwise ----
TEST_F(TileShapeResolverTest, Where_TT)
{
    TileShape::Current().SetVecTile(32, 32);
    Tensor cond(DT_BOOL, {64, 64}, "cond");
    Tensor input(DT_FP32, {64, 64}, "input");
    Tensor other(DT_FP32, {64, 64}, "other");
    Tensor out;
    FUNCTION("WhereCase") { out = Where(cond, input, other); }
    ExpectResolverMatchesExpansion(Program::GetInstance().GetCurrentFunction(), {out});
}

// ---- OP_ONEHOT: single 1-D input, default elementwise branch ----
TEST_F(TileShapeResolverTest, OneHot_1D)
{
    TileShape::Current().SetVecTile(8, 16);
    Tensor self(DT_INT32, {8}, "self");
    Tensor out;
    FUNCTION("OneHotCase") { out = OneHot(self, 16); }
    ExpectResolverMatchesExpansion(Program::GetInstance().GetCurrentFunction(), {out});
}

// ---- OP_TOPK: single input, non-axis elementwise + axis chunked (aligned to 32) ----
TEST_F(TileShapeResolverTest, TopK_Axis1_MergeSort)
{
    TileShape::Current().SetVecTile(32, 4);
    Tensor self(DT_FP32, {64, 128}, "self");
    Tensor values, indices;
    FUNCTION("TopKCase") { std::tie(values, indices) = TopK(self, 4, 1, true, TopKAlgo::MERGE_SORT); }
    ExpectResolverMatchesExpansion(Program::GetInstance().GetCurrentFunction(), {values, indices});
}

// ---- OP_GATHER_ELEMENT: params (axis full) + indices (elementwise) ----
TEST_F(TileShapeResolverTest, GatherElements_Axis0)
{
    TileShape::Current().SetVecTile(4, 16);
    Tensor params(DT_FP32, {8, 16}, "params");
    Tensor indices(DT_INT32, {8, 16}, "indices");
    Tensor out;
    FUNCTION("GatherElementsCase") { out = GatherElements(params, indices, 0); }
    ExpectResolverMatchesExpansion(Program::GetInstance().GetCurrentFunction(), {out});
}

// ---- OP_INDEX_PUT: in-place GM write; every tile op shares the same full {result} output. ----
TEST_F(TileShapeResolverTest, IndexPut_3D)
{
    TileShape::Current().SetVecTile(8, 8, 8);
    Tensor self(DT_FP32, {128, 8, 8}, "self");
    Tensor values(DT_FP32, {128, 8}, "values");
    Tensor indices0(DT_INT32, {128}, "indices0");
    Tensor indices1(DT_INT32, {128}, "indices1");
    std::vector<Tensor> indices{indices0, indices1};
    FUNCTION("IndexPutCase") { IndexPut_(self, indices, values, false); }
    ExpectResolverMatchesExpansion(Program::GetInstance().GetCurrentFunction(), {self});
}

// ---- OP_INDEX_ADD: [self, src, indices]; self axis full, src elementwise, indices full ----
TEST_F(TileShapeResolverTest, IndexAdd_Axis0)
{
    TileShape::Current().SetVecTile(8, 8);
    Tensor self(DT_FP32, {8, 16}, "self");
    Tensor src(DT_FP32, {4, 16}, "src");
    Tensor indices(DT_INT32, {4}, "indices");
    Element alpha(DT_FP32, 1.0f);
    FUNCTION("IndexAddCase") { IndexAdd_(self, src, indices, 0, alpha); }
    ExpectResolverMatchesExpansion(Program::GetInstance().GetCurrentFunction(), {self});
}

// ---- OP_PRELU: [input, weight]; input elementwise, weight 1-D tiled on channel axis ----
TEST_F(TileShapeResolverTest, PReLU_2D)
{
    TileShape::Current().SetVecTile(8, 8);
    Tensor input(DT_FP32, {8, 16}, "input");
    Tensor weight(DT_FP32, {16}, "weight");
    Tensor out;
    FUNCTION("PReLUCase") { out = PReLU(input, weight); }
    ExpectResolverMatchesExpansion(Program::GetInstance().GetCurrentFunction(), {out});
}

// ---- OP_CAST: single input, default elementwise branch ----
TEST_F(TileShapeResolverTest, Cast_Fp32ToFp16)
{
    TileShape::Current().SetVecTile(4, 8);
    Tensor self(DT_FP32, {8, 16}, "self");
    Tensor out;
    FUNCTION("CastCase") { out = Cast(self, DataType::DT_FP16, CastMode::CAST_NONE); }
    ExpectResolverMatchesExpansion(Program::GetInstance().GetCurrentFunction(), {out});
}

// ---- OP_EXPAND: single input broadcast; default elementwise branch clamps by input's own shape ----
TEST_F(TileShapeResolverTest, Expand_Broadcast)
{
    TileShape::Current().SetVecTile(4, 16);
    Tensor self(DT_FP32, {8, 1}, "self");
    Tensor out;
    FUNCTION("ExpandCase") { out = Expand(self, {8, 16}); }
    ExpectResolverMatchesExpansion(Program::GetInstance().GetCurrentFunction(), {out});
}

// ---- OP_AXPY: [self, other] both elementwise (y += alpha*x) ----
TEST_F(TileShapeResolverTest, Axpy_Elementwise)
{
    TileShape::Current().SetVecTile(4, 8);
    Tensor self(DT_FP32, {8, 16}, "self");
    Tensor other(DT_FP32, {8, 16}, "other");
    Tensor out;
    FUNCTION("AxpyCase") { out = Axpy(self, other, 1.0f); }
    ExpectResolverMatchesExpansion(Program::GetInstance().GetCurrentFunction(), {out});
}

// ---- OP_PERMUTE: output axis i <- input axis perm[i]; output tile = VecTile permuted by perm.
//      Uses vecTile smaller than input on every axis so the default (min per output axis) would
//      differ from the correct permuted VecTile, making this a discriminating fixture. ----
TEST_F(TileShapeResolverTest, Permute_AxisSwap)
{
    TileShape::Current().SetVecTile(2, 4, 8);
    Tensor self(DT_FP32, {4, 8, 16}, "self");
    Tensor out;
    FUNCTION("PermuteCase") { out = Permute(self, {1, 0, 2}); }
    ExpectResolverMatchesExpansion(Program::GetInstance().GetCurrentFunction(), {out});
}

// ---- OP_MRGSORT: single input; sort axis (last) kept full, others min by vecTile ----
TEST_F(TileShapeResolverTest, MrgSort_Axis1)
{
    TileShape::Current().SetVecTile(4, 16);
    Tensor self(DT_FP32, {8, 16}, "self");
    Tensor out;
    FUNCTION("MrgSortCase") { out = MrgSort(self, 2); }
    ExpectResolverMatchesExpansion(Program::GetInstance().GetCurrentFunction(), {out});
}

// ---- Reduction single family (OP_ROW*_SINGLE): single input; reduce axis kept full,
//      other axes min by vecTile. FP32 + keepDim=true yields a single tensor-graph op. ----

TEST_F(TileShapeResolverTest, RowSumSingle_Axis1)
{
    TileShape::Current().SetVecTile(4, 16);
    Tensor self(DT_FP32, {8, 16}, "self");
    Tensor out;
    FUNCTION("RowSumSingleCase") { out = Sum(self, 1, true); }
    ExpectResolverMatchesExpansion(Program::GetInstance().GetCurrentFunction(), {out});
}

TEST_F(TileShapeResolverTest, RowMaxSingle_Axis0)
{
    TileShape::Current().SetVecTile(8, 4);
    Tensor self(DT_FP32, {8, 16}, "self");
    Tensor out;
    FUNCTION("RowMaxSingleCase") { out = Amax(self, 0, true); }
    ExpectResolverMatchesExpansion(Program::GetInstance().GetCurrentFunction(), {out});
}

TEST_F(TileShapeResolverTest, RowMinSingle_Axis1)
{
    TileShape::Current().SetVecTile(4, 16);
    Tensor self(DT_FP32, {8, 16}, "self");
    Tensor out;
    FUNCTION("RowMinSingleCase") { out = Amin(self, 1, true); }
    ExpectResolverMatchesExpansion(Program::GetInstance().GetCurrentFunction(), {out});
}

TEST_F(TileShapeResolverTest, RowProdSingle_Axis0)
{
    TileShape::Current().SetVecTile(8, 4);
    Tensor self(DT_FP32, {8, 16}, "self");
    Tensor out;
    FUNCTION("RowProdSingleCase") { out = Prod(self, 0, true); }
    ExpectResolverMatchesExpansion(Program::GetInstance().GetCurrentFunction(), {out});
}

TEST_F(TileShapeResolverTest, RowArgMaxSingle_Axis1)
{
    TileShape::Current().SetVecTile(4, 16);
    Tensor self(DT_FP32, {8, 16}, "self");
    Tensor out;
    FUNCTION("RowArgMaxSingleCase") { out = ArgMax(self, 1, true); }
    ExpectResolverMatchesExpansion(Program::GetInstance().GetCurrentFunction(), {out});
}

TEST_F(TileShapeResolverTest, RowArgMinSingle_Axis0)
{
    TileShape::Current().SetVecTile(8, 4);
    Tensor self(DT_FP32, {8, 16}, "self");
    Tensor out;
    FUNCTION("RowArgMinSingleCase") { out = ArgMin(self, 0, true); }
    ExpectResolverMatchesExpansion(Program::GetInstance().GetCurrentFunction(), {out});
}

// ---- OP_BITSORT (via Sort32): single input; sort axis (last) kept full, others min by vecTile. ----
TEST_F(TileShapeResolverTest, BitSort_Axis1)
{
    TileShape::Current().SetVecTile(4, 16);
    Tensor self(DT_FP32, {8, 16}, "self");
    Tensor out;
    FUNCTION("BitSortCase") { out = Sort32(self, 0); }
    ExpectResolverMatchesExpansion(Program::GetInstance().GetCurrentFunction(), {out});
}

// ---- OP_EXTRACT (via TopKExtract): single input; last axis kept full, others min by vecTile. ----
TEST_F(TileShapeResolverTest, Extract_Axis1)
{
    TileShape::Current().SetVecTile(4, 16);
    Tensor self(DT_FP32, {8, 16}, "self");
    Tensor out;
    FUNCTION("ExtractCase") { out = TopKExtract(self, 4, false); }
    ExpectResolverMatchesExpansion(Program::GetInstance().GetCurrentFunction(), {out});
}

// ---- OP_SCATTER_ELEMENT (via Scatter with Element src): [self, indices]; axis full on both. ----
TEST_F(TileShapeResolverTest, ScatterElement_Axis0)
{
    TileShape::Current().SetVecTile(8, 8);
    Tensor self(DT_FP32, {8, 16}, "self");
    Tensor indices(DT_INT32, {8, 16}, "indices");
    Element src(DT_FP32, 1.0f);
    Tensor out;
    FUNCTION("ScatterElementCase") { out = Scatter(self, indices, src, 0, ScatterMode::NONE); }
    ExpectResolverMatchesExpansion(Program::GetInstance().GetCurrentFunction(), {out});
}

namespace {

Operation& MakeMatmulOp(Function& function, const std::vector<int64_t>& shapeA, const std::vector<int64_t>& shapeB,
                        const std::vector<int64_t>& shapeC, bool transA, bool transB)
{
    auto inA = IRBuilder().CreateTensorVar(DT_FP32, shapeA, SymbolicScalar::FromConcrete(shapeA));
    auto inB = IRBuilder().CreateTensorVar(DT_FP32, shapeB, SymbolicScalar::FromConcrete(shapeB));
    auto outC = IRBuilder().CreateTensorVar(DT_FP32, shapeC, SymbolicScalar::FromConcrete(shapeC));
    auto& matmul = IRBuilder().CreateTensorOpStmt(function, Opcode::OP_A_MUL_B, {inA, inB}, {outC});
    if (transA) {
        matmul.SetAttribute(npu::tile_fwk::Matrix::A_MUL_B_TRANS_A, true);
    }
    if (transB) {
        matmul.SetAttribute(npu::tile_fwk::Matrix::A_MUL_B_TRANS_B, true);
    }
    return matmul;
}

} // namespace

// ---- OP_A_MUL_B: input tile is the L1 copy-in (closest hop to a producer VIEW),
//      not the L0 tile that later slices K off L1. A = {m[0], k[1]}, B = {k[2], n[0]}. ----
TEST_F(TileShapeResolverTest, Matmul_InputUsesL1CopyInTile)
{
    constexpr int64_t kM0 = 32;
    constexpr int64_t kK0 = 32;
    constexpr int64_t kK1 = 64;
    constexpr int64_t kK2 = 96;
    constexpr int64_t kN0 = 64;

    auto func = std::make_shared<Function>(Program::GetInstance(), "MatmulL1CopyIn", "MatmulL1CopyIn", nullptr);
    func->SetGraphType(GraphType::TENSOR_GRAPH);
    auto& matmul = MakeMatmulOp(*func, {64, 192}, {192, 128}, {64, 128}, false, false);
    matmul.GetTileShapeForSetting().SetCubeTile({kM0, kM0}, {kK0, kK1, kK2}, {kN0, kN0});

    EXPECT_EQ(TileShapeResolver::Instance().GetInputTileShape(matmul, 0).GetVecTile().tile,
              (std::vector<int64_t>{kM0, kK1}));
    EXPECT_EQ(TileShapeResolver::Instance().GetInputTileShape(matmul, 1).GetVecTile().tile,
              (std::vector<int64_t>{kK2, kN0}));
    EXPECT_EQ(TileShapeResolver::Instance().GetOutputTileShape(matmul, 0).GetVecTile().tile,
              (std::vector<int64_t>{kM0, kN0}));
}

TEST_F(TileShapeResolverTest, Matmul_InputUsesL1CopyInTileTransA)
{
    constexpr int64_t kM0 = 32;
    constexpr int64_t kK0 = 32;
    constexpr int64_t kK1 = 64;
    constexpr int64_t kN0 = 64;

    auto func = std::make_shared<Function>(Program::GetInstance(), "MatmulL1CopyInTransA", "MatmulL1CopyInTransA",
                                           nullptr);
    func->SetGraphType(GraphType::TENSOR_GRAPH);
    auto& matmul = MakeMatmulOp(*func, {192, 64}, {192, 128}, {64, 128}, true, false);
    matmul.GetTileShapeForSetting().SetCubeTile({kM0, kM0}, {kK0, kK1, kK1}, {kN0, kN0});

    EXPECT_EQ(TileShapeResolver::Instance().GetInputTileShape(matmul, 0).GetVecTile().tile,
              (std::vector<int64_t>{kK1, kM0}));
    EXPECT_EQ(TileShapeResolver::Instance().GetInputTileShape(matmul, 1).GetVecTile().tile,
              (std::vector<int64_t>{kK1, kN0}));
}

// ---- OP_INDEX_ADD_UB: [self, src, indices]; FP32 + INT32 -> single op. ----
TEST_F(TileShapeResolverTest, IndexAddUb_Axis0)
{
    TileShape::Current().SetVecTile(8, 8);
    Tensor self(DT_FP32, {8, 16}, "self");
    Tensor src(DT_FP32, {4, 16}, "src");
    Tensor indices(DT_INT32, {4}, "indices");
    Element alpha(DT_FP32, 1.0f);
    Tensor out;
    FUNCTION("IndexAddUbCase") { out = IndexAddUB(self, src, indices, 0, alpha); }
    ExpectResolverMatchesExpansion(Program::GetInstance().GetCurrentFunction(), {out});
}
