/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_expand_function.cpp
 * \brief Unit test for ExpandFunction pass.
 */

#include <algorithm>
#include <gtest/gtest.h>
#include <string>
#include <unordered_set>
#include <vector>
#include "interface/function/function.h"
#include "interface/operation/attribute.h"
#include "interface/tensor/irbuilder.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "passes/tensor_graph_pass/expand_function.h"
#include "ut_json/ut_json_tool.h"

namespace npu::tile_fwk {
class TestExpandFunction : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetHostConfig(KEY_STRATEGY, "ExpandFunctionTestStrategy");
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {}
};

TEST_F(TestExpandFunction, ExpandFunctionTest)
{
    PassManager& passManager = PassManager::Instance();
    passManager.RegisterStrategy("ExpandFunctionTestStrategy",
                                 {
                                     {"RemoveRedundantReshape", PassName::REMOVE_REDUNDANT_RESHAPE},
                                 });

    std::vector<int64_t> shape{64, 64};
    Tensor a(DT_FP32, shape, "a");
    Tensor b(DT_FP32, shape, "b");
    Tensor c(DT_FP32, shape, "c");
    constexpr int TILE_SHAPE = 32;
    TileShape::Current().SetVecTile(TILE_SHAPE, TILE_SHAPE);

    FUNCTION("A") { c = Div(a, b); }

    std::string jsonFilePath = "./config/pass/json/expand_function.json";
    bool dumpJsonFlag = true;
    if (dumpJsonFlag) {
        auto programJson = Program::GetInstance().DumpJson();
        DumpJsonFile(programJson, jsonFilePath);
    }
    Json readData = LoadJsonFile(jsonFilePath);
    Program::GetInstance().LoadJson(readData);

    Function* currentFunction = Program::GetInstance().GetCurrentFunction();

    auto opListBefore = currentFunction->Operations().DuplicatedOpList();
    int divNumBefore = 0;
    int divNumAfter = 0;
    for (auto& op : opListBefore) {
        if (op->GetOpcodeStr().find("DIV") != std::string::npos) {
            divNumBefore++;
        }
    }
    Program testProgram;
    ExpandFunction expandFunction;
    expandFunction.RunOnFunction(*currentFunction);
    auto opListAfter = currentFunction->Operations().DuplicatedOpList();
    for (auto& op : opListAfter) {
        if (op->GetOpcodeStr().find("DIV") != std::string::npos) {
            divNumAfter++;
        }
    }
    constexpr int TEST_RES1 = 1;
    constexpr int TEST_RES2 = 4;
    EXPECT_EQ(divNumBefore, TEST_RES1);
    EXPECT_EQ(divNumAfter, TEST_RES2);
}

namespace {

void SetViewOpAttribute(Operation& viewOp, const std::vector<int64_t>& offsets, const std::vector<int64_t>& shapes)
{
    auto dynOffsets = SymbolicScalar::FromConcrete(offsets);
    auto validShape = GetViewValidShape(viewOp.GetIOperands()[0]->GetDynValidShape(), offsets, dynOffsets, shapes);
    viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(offsets, dynOffsets, validShape));
}

int CountInputSliceOpsWithSecondDimOffset(Function& function, const LogicalTensorPtr& input, int64_t offset)
{
    int count = 0;
    for (auto& op : function.Operations(false)) {
        if (op.GetOpcode() != Opcode::OP_SLICE) {
            continue;
        }
        if (input == nullptr || op.GetIOperands().empty() || op.GetIOperands()[0] != input) {
            continue;
        }
        auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
        if (viewAttr == nullptr || viewAttr->GetFromOffset().size() < 2) {
            continue;
        }
        if (viewAttr->GetFromOffset()[1] == offset) {
            count++;
        }
    }
    return count;
}

int CountSliceOpsOnInputWithShape(Function& function, const LogicalTensorPtr& input, const std::vector<int64_t>& shape)
{
    int count = 0;
    for (auto& op : function.Operations(false)) {
        if (op.GetOpcode() != Opcode::OP_SLICE) {
            continue;
        }
        if (input == nullptr || op.GetIOperands().empty() || op.GetIOperands()[0] != input) {
            continue;
        }
        if (op.GetOOperands().empty() || op.GetOOperands()[0]->GetShape() != shape) {
            continue;
        }
        ++count;
    }
    return count;
}

} // namespace

TEST_F(TestExpandFunction, ViewDerivesVecTileFromCubeMatmulConsumer)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "ViewCubeTile", "ViewCubeTile", nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("ViewCubeTile", func);
    func->SetGraphType(GraphType::TENSOR_GRAPH);

    constexpr int64_t kM = 64;
    constexpr int64_t kK = 96;
    constexpr int64_t kN = 96;
    constexpr int64_t kCubeM = 64;
    constexpr int64_t kCubeK = 32;
    constexpr int64_t kCubeN = 64;

    const std::vector<int64_t> shapeA = {kM, kK};
    const std::vector<int64_t> shapeB = {kK, kN};
    const std::vector<int64_t> viewShapeA = {kM, kCubeK};
    const std::vector<int64_t> viewShapeB = {kCubeK, kN};
    const std::vector<int64_t> shapeC = {kM, kN};
    const std::vector<int64_t> offset0 = {0, 0};

    auto incastA = IRBuilder().CreateTensorVar(DT_FP32, shapeA, SymbolicScalar::FromConcrete(shapeA));
    auto incastB = IRBuilder().CreateTensorVar(DT_FP32, shapeB, SymbolicScalar::FromConcrete(shapeB));
    auto viewOutA = IRBuilder().CreateTensorVar(DT_FP32, viewShapeA, SymbolicScalar::FromConcrete(viewShapeA));
    auto viewOutB = IRBuilder().CreateTensorVar(DT_FP32, viewShapeB, SymbolicScalar::FromConcrete(viewShapeB));
    auto matmulOut = IRBuilder().CreateTensorVar(DT_FP32, shapeC, SymbolicScalar::FromConcrete(shapeC));

    auto& viewA = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_VIEW, {incastA}, {viewOutA});
    SetViewOpAttribute(viewA, offset0, viewShapeA);
    auto& viewB = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_VIEW, {incastB}, {viewOutB});
    SetViewOpAttribute(viewB, offset0, viewShapeB);

    TileShape::Current().SetCubeTile({kCubeM, kCubeM}, {kCubeK, kCubeK, kCubeK}, {kCubeN, kCubeN});
    TileShape::Current().GetVecTile().tile.clear();
    auto& matmul = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_A_MUL_B, {viewOutA, viewOutB}, {matmulOut});
    matmul.GetTileShapeForSetting().SetCubeTile({kCubeM, kCubeM}, {kCubeK, kCubeK, kCubeK}, {kCubeN, kCubeN});
    matmul.GetTileShapeForSetting().GetVecTile().tile.clear();

    ExpandFunction expandFunction;
    ASSERT_EQ(expandFunction.RunOnFunction(*func), SUCCESS);

    EXPECT_EQ(CountInputSliceOpsWithSecondDimOffset(*func, incastB, kCubeN), 1);
}

TEST_F(TestExpandFunction, ViewDerivesVecTileFromCubeMatmulL1CopyIn)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "ViewCubeL1Tile", "ViewCubeL1Tile", nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("ViewCubeL1Tile", func);
    func->SetGraphType(GraphType::TENSOR_GRAPH);

    constexpr int64_t kM = 64;
    constexpr int64_t kK = 128;
    constexpr int64_t kN = 64;
    constexpr int64_t kCubeM = 64;
    constexpr int64_t kCubeKL0 = 32;
    constexpr int64_t kCubeKL1 = 64;
    constexpr int64_t kCubeN = 64;

    const std::vector<int64_t> shapeA = {kM, kK};
    const std::vector<int64_t> shapeB = {kK, kN};
    const std::vector<int64_t> shapeC = {kM, kN};
    const std::vector<int64_t> offset0 = {0, 0};
    const std::vector<int64_t> l1SliceShapeA = {kCubeM, kCubeKL1};
    const std::vector<int64_t> l0SliceShapeA = {kCubeM, kCubeKL0};

    auto incastA = IRBuilder().CreateTensorVar(DT_FP32, shapeA, SymbolicScalar::FromConcrete(shapeA));
    auto incastB = IRBuilder().CreateTensorVar(DT_FP32, shapeB, SymbolicScalar::FromConcrete(shapeB));
    auto viewOutA = IRBuilder().CreateTensorVar(DT_FP32, shapeA, SymbolicScalar::FromConcrete(shapeA));
    auto viewOutB = IRBuilder().CreateTensorVar(DT_FP32, shapeB, SymbolicScalar::FromConcrete(shapeB));
    auto matmulOut = IRBuilder().CreateTensorVar(DT_FP32, shapeC, SymbolicScalar::FromConcrete(shapeC));

    auto& viewA = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_VIEW, {incastA}, {viewOutA});
    SetViewOpAttribute(viewA, offset0, shapeA);
    auto& viewB = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_VIEW, {incastB}, {viewOutB});
    SetViewOpAttribute(viewB, offset0, shapeB);

    TileShape::Current().SetCubeTile({kCubeM, kCubeM}, {kCubeKL0, kCubeKL1, kCubeKL1}, {kCubeN, kCubeN});
    TileShape::Current().GetVecTile().tile.clear();
    auto& matmul = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_A_MUL_B, {viewOutA, viewOutB}, {matmulOut});
    matmul.GetTileShapeForSetting().SetCubeTile({kCubeM, kCubeM}, {kCubeKL0, kCubeKL1, kCubeKL1}, {kCubeN, kCubeN});
    matmul.GetTileShapeForSetting().GetVecTile().tile.clear();

    ExpandFunction expandFunction;
    ASSERT_EQ(expandFunction.RunOnFunction(*func), SUCCESS);

    EXPECT_EQ(CountSliceOpsOnInputWithShape(*func, incastA, l1SliceShapeA), 2);
    EXPECT_EQ(CountSliceOpsOnInputWithShape(*func, incastA, l0SliceShapeA), 0);
}

TEST_F(TestExpandFunction, AssembleDerivesVecTileFromCubeMatmulProducer)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "AssembleCubeTile", "AssembleCubeTile", nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("AssembleCubeTile", func);
    func->SetGraphType(GraphType::TENSOR_GRAPH);

    constexpr int64_t kM = 64;
    constexpr int64_t kK = 32;
    constexpr int64_t kN = 256;
    constexpr int64_t kCubeM = 64;
    constexpr int64_t kCubeK = 32;
    constexpr int64_t kCubeN = 128;

    const std::vector<int64_t> shapeA = {kM, kK};
    const std::vector<int64_t> shapeB = {kK, kN};
    const std::vector<int64_t> shapeC = {kM, kN};
    const std::vector<int64_t> shapeOut = {kM, kN};
    const std::vector<int64_t> offset0 = {0, 0};
    const std::vector<int64_t> expectedSliceShape = {kCubeM, kCubeN};

    auto incastA = IRBuilder().CreateTensorVar(DT_FP32, shapeA, SymbolicScalar::FromConcrete(shapeA));
    auto incastB = IRBuilder().CreateTensorVar(DT_FP32, shapeB, SymbolicScalar::FromConcrete(shapeB));
    auto matmulOut = IRBuilder().CreateTensorVar(DT_FP32, shapeC, SymbolicScalar::FromConcrete(shapeC));
    auto outCast = IRBuilder().CreateTensorVar(DT_FP32, shapeOut, SymbolicScalar::FromConcrete(shapeOut));

    TileShape::Current().SetCubeTile({kCubeM, kCubeM}, {kCubeK, kCubeK, kCubeK}, {kCubeN, kCubeN});
    // Deliberately mismatch TileShape::Current() with matmul output tile {64, 128}.
    TileShape::Current().SetVecTile(kCubeM, kCubeM);
    auto& matmul = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_A_MUL_B, {incastA, incastB}, {matmulOut});
    matmul.GetTileShapeForSetting().SetCubeTile({kCubeM, kCubeM}, {kCubeK, kCubeK, kCubeK}, {kCubeN, kCubeN});
    matmul.GetTileShapeForSetting().GetVecTile().tile.clear();

    auto& assemble = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_ASSEMBLE, {matmulOut}, {outCast});
    assemble.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset0));
    assemble.GetTileShapeForSetting().GetVecTile().tile.clear();

    ExpandFunction expandFunction;
    ASSERT_EQ(expandFunction.RunOnFunction(*func), SUCCESS);

    EXPECT_EQ(CountSliceOpsOnInputWithShape(*func, matmulOut, expectedSliceShape), 2);
    EXPECT_EQ(CountInputSliceOpsWithSecondDimOffset(*func, matmulOut, kCubeN), 1);
    EXPECT_EQ(CountSliceOpsOnInputWithShape(*func, matmulOut, {kCubeM, kCubeM}), 0);
}

TEST_F(TestExpandFunction, ExpandedOperationsPreserveTokenContract)
{
    std::vector<int64_t> shape{64, 64};
    Tensor input(DT_FP32, shape, "input");
    Tensor inputB(DT_FP32, shape, "inputB");
    Tensor output(DT_FP32, shape, "output");
    TileShape::Current().SetVecTile(32, 32);

    FUNCTION("TokenContract") { output = Div(input, inputB); }
    auto* function = Program::GetInstance().GetFunctionByRawName("TENSOR_TokenContract");
    ASSERT_NE(function, nullptr);

    auto operations = function->Operations(false).DuplicatedOpList();
    auto source = std::find_if(operations.begin(), operations.end(),
                               [](const auto* op) { return op->GetOpcodeStr().find("DIV") != std::string::npos; });
    ASSERT_NE(source, operations.end());
    auto sourceOpcode = (*source)->GetOpcode();

    auto resultToken = IRBuilder().CreateTokenVar(ir::Span::Unknown());
    auto inputToken = IRBuilder().CreateTokenVar(ir::Span::Unknown());
    (*source)->result_token_ = {resultToken};
    (*source)->tokens_.push_back(inputToken);

    ExpandFunction expandFunction;
    ASSERT_EQ(expandFunction.RunOnFunction(*function), SUCCESS);

    std::unordered_set<ir::VarPtr> expandedResultTokens;
    size_t expandedCount = 0;
    for (auto* op : function->Operations(false).DuplicatedOpList()) {
        if (op->GetOpcode() != sourceOpcode) {
            continue;
        }
        expandedCount++;
        ASSERT_FALSE(op->result_token_.empty());
        expandedResultTokens.insert(op->result_token_.front());
        EXPECT_EQ(std::count(op->tokens_.begin(), op->tokens_.end(), inputToken), 1);
    }

    EXPECT_EQ(expandedCount, 4);
    EXPECT_EQ(expandedResultTokens.size(), expandedCount);
    EXPECT_EQ(function->GetVarDependency().GetConsumers(inputToken).size(), expandedCount);
    for (const auto& token : expandedResultTokens) {
        EXPECT_EQ(function->GetVarDependency().GetProducers(token).size(), 1);
    }
}
} // namespace npu::tile_fwk
