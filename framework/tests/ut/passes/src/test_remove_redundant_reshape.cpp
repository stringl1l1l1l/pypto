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
 * \file test_remove_redundant_reshape.cpp
 * \brief Unit test for RemoveRedundantReshape pass.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/irbuilder.h"
#include "symbolic_scalar_test_utils.h"
#include <fstream>
#include <vector>
#include <string>
#define private public
#include "passes/tensor_graph_pass/remove_redundant_reshape.h"
#include "passes/pass_utils/view_reshape_assemble_reorder_utils.h"

using namespace npu::tile_fwk;

class RemoveRedundantReshapeTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetHostConfig(KEY_STRATEGY, "ReshapeTestStrategy");
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {}

protected:
    static Operation* FindOpByOpcode(const OperationsViewer& ops, Opcode opcode)
    {
        for (auto& op : ops) {
            if (op.GetOpcode() == opcode) {
                return const_cast<Operation*>(&op);
            }
        }
        return nullptr;
    }

    static bool IsOpRemoved(const OperationsViewer& ops, int opmagic)
    {
        for (const auto& op : ops) {
            if (op.opmagic == opmagic) {
                return false;
            }
        }
        return true;
    }

    static Operation* FindOpByMagic(const OperationsViewer& ops, int opmagic)
    {
        for (auto& op : ops) {
            if (op.opmagic == opmagic) {
                return const_cast<Operation*>(&op);
            }
        }
        return nullptr;
    }
};

TEST_F(RemoveRedundantReshapeTest, TestReshapeChain)
{
    // Define Tensor shapes
    std::vector<int64_t> shape1{1, 256, 512};
    std::vector<int64_t> shape2{1, 512, 256};
    std::vector<int64_t> shape3{1, 128, 1024};
    // Create Tensors
    Tensor in_tensor(DT_FP32, shape1, "in_tensor");
    Tensor out_tensor_B(DT_FP32, shape3, "out_tensor_B");

    // Initialize PassManager
    PassManager& passManager = PassManager::Instance();
    passManager.RegisterStrategy("ReshapeTestStrategy",
                                 {
                                     {"RemoveRedundantReshape", PassName::REMOVE_REDUNDANT_RESHAPE},
                                 });
    ConfigManager::Instance();

    // Create and configure the function
    FUNCTION("ReshapeChainFunction")
    {
        Tensor out_tensor_A = Reshape(in_tensor, shape2);
        out_tensor_B = Reshape(out_tensor_A, shape3);
    }

    Function* currentFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_ReshapeChainFunction");

    auto updated_operations = currentFunction->Operations();

    EXPECT_EQ(updated_operations.size(), 3)
        << "After the Pass, there should be 3 operations (View + Assemble + Reshape)";
    Operation* view_op = FindOpByOpcode(updated_operations, Opcode::OP_VIEW);
    Operation* assemble_op = FindOpByOpcode(updated_operations, Opcode::OP_ASSEMBLE);
    Operation* reshape_op = FindOpByOpcode(updated_operations, Opcode::OP_RESHAPE);
    ASSERT_NE(view_op, nullptr) << "View operation should be kept";
    ASSERT_NE(assemble_op, nullptr) << "Assemble operation should be kept";
    ASSERT_NE(reshape_op, nullptr) << "Reshape operation should be kept";

    EXPECT_EQ(view_op->GetIOperands()[0]->shape, shape1) << "The input shape of View should be the same as shape1";
    EXPECT_EQ(view_op->GetOOperands()[0]->shape, shape1)
        << "Without matmul, view/reshape/assemble reorder should be skipped";
    EXPECT_EQ(reshape_op->GetIOperands()[0]->shape, shape1)
        << "Without matmul, final Reshape should consume the original shape";
    EXPECT_EQ(reshape_op->GetOOperands()[0]->shape, shape3)
        << "The output shape of final Reshape should be the same as shape3";
}

TEST_F(RemoveRedundantReshapeTest, TestReplaceInput)
{
    std::vector<int64_t> shape1{1, 256, 512};
    std::vector<int64_t> shape2{1, 512, 256};
    std::vector<int64_t> shape3{1, 128, 1024};

    Tensor in_tensor(DT_FP32, shape1, "in_tensor");
    Tensor out_tensor_B(DT_FP32, shape2, "out_tensor_B");
    Tensor out_tensor_C(DT_FP32, shape3, "out_tensor_C");

    PassManager& passManager = PassManager::Instance();
    passManager.RegisterStrategy("ReshapeTestStrategy",
                                 {{"RemoveRedundantReshape", PassName::REMOVE_REDUNDANT_RESHAPE}});

    TileShape::Current().SetVecTile({1, 64, 64});
    int64_t first_reshape_magic = -1;
    FUNCTION("ReplaceInputFunction")
    {
        Tensor out_tensor_A = Reshape(in_tensor, shape2);
        out_tensor_B = Reshape(out_tensor_A, shape3);
        Tensor add_tensor(DT_FP32, shape2, "add_tensor");
        out_tensor_C = Add(out_tensor_A, add_tensor);

        auto operations = Program::GetInstance().GetCurrentFunction()->Operations();
        for (const auto& op : operations) {
            if (op.GetOpcodeStr() == "RESHAPE") {
                first_reshape_magic = op.opmagic;
                break;
            }
        }
    }

    Function* currentFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_ReplaceInputFunction");
    auto updated_operations = currentFunction->Operations();

    EXPECT_EQ(updated_operations.size(), 6) << "Without matmul, view/reshape/assemble reorder should be skipped";
    EXPECT_FALSE(IsOpRemoved(updated_operations, first_reshape_magic))
        << "Without matmul, the first Reshape operation should be kept";

    Operation* add_op = FindOpByOpcode(updated_operations, Opcode::OP_ADD);
    Operation* view_op = FindOpByOpcode(updated_operations, Opcode::OP_VIEW);
    ASSERT_NE(add_op, nullptr) << "Add operation should be present";
    ASSERT_NE(view_op, nullptr) << "View operation should be present";
    EXPECT_EQ(add_op->GetIOperands()[0]->shape, shape2) << "The input shape of Add should be the same as shape2";
}

TEST_F(RemoveRedundantReshapeTest, RemovedReshapeMigratesTokenDependencies)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "RemovedReshapeMigratesTokenDependencies",
                                               "RemovedReshapeMigratesTokenDependencies", nullptr);
    ASSERT_NE(function, nullptr);

    std::vector<int64_t> shape{32, 32};
    auto input = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto producerOutput = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto reshapeOutput = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tokenInput = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tokenOutput = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& producer = IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_EXP, {input}, {producerOutput});
    auto& reshape = IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_RESHAPE, {producerOutput}, {reshapeOutput});
    auto& consumer = IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_SQRT, {reshapeOutput}, {output});
    auto& tokenProducer = IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_ABS, {tokenInput}, {tokenOutput});
    int reshapeMagic = reshape.GetOpMagic();

    auto inputToken = IRBuilder().CreateTokenVar(ir::Span::Unknown());
    auto resultToken = IRBuilder().CreateTokenVar(ir::Span::Unknown());
    auto toStmt = [](Operation& op) { return std::static_pointer_cast<const ir::Stmt>(op.shared_from_this()); };
    auto& dependency = function->GetVarDependency();
    tokenProducer.result_token_ = {inputToken};
    reshape.tokens_ = {inputToken};
    reshape.result_token_ = {resultToken};
    consumer.tokens_ = {resultToken};
    dependency.AddProducer(inputToken, toStmt(tokenProducer));
    dependency.AddConsumer(inputToken, toStmt(reshape));
    dependency.AddProducer(resultToken, toStmt(reshape));
    dependency.AddConsumer(resultToken, toStmt(consumer));
    function->inCasts_ = {input, tokenInput};
    function->outCasts_ = {output, tokenOutput};

    RemoveRedundantReshape pass;
    EXPECT_EQ(pass.RunOnFunction(*function), SUCCESS);

    EXPECT_TRUE(IsOpRemoved(function->Operations(), reshapeMagic));
    EXPECT_EQ(consumer.GetIOperands().front(), producerOutput);
    EXPECT_TRUE(dependency.HasConsumer(inputToken, toStmt(consumer)));
    EXPECT_NE(std::find(consumer.tokens_.begin(), consumer.tokens_.end(), inputToken), consumer.tokens_.end());
    ASSERT_FALSE(producer.result_token_.empty());
    auto migratedResultToken = producer.result_token_.front();
    EXPECT_NE(migratedResultToken, resultToken);
    EXPECT_TRUE(dependency.HasProducer(migratedResultToken, toStmt(producer)));
    EXPECT_TRUE(dependency.HasConsumer(migratedResultToken, toStmt(consumer)));
    EXPECT_NE(std::find(consumer.tokens_.begin(), consumer.tokens_.end(), migratedResultToken), consumer.tokens_.end());
    EXPECT_FALSE(dependency.HasDependency(resultToken));
    EXPECT_NO_THROW(function->GetSortedOperations());
}

TEST_F(RemoveRedundantReshapeTest, ViewReshapeReorderMigratesTokenDependencies)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "ViewReshapeReorderMigratesTokens",
                                               "ViewReshapeReorderMigratesTokens", nullptr);
    ASSERT_NE(function, nullptr);

    auto input = IRBuilder().CreateTensorVar(DT_FP32, {8, 16}, CreateTestConstIntVector({8, 16}));
    auto middle = IRBuilder().CreateTensorVar(DT_FP32, {4, 16}, CreateTestConstIntVector({4, 16}));
    auto reshapeOutput = IRBuilder().CreateTensorVar(DT_FP32, {64}, CreateTestConstIntVector({64}));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, {64}, CreateTestConstIntVector({64}));
    auto tokenInput = IRBuilder().CreateTensorVar(DT_FP32, {1}, CreateTestConstIntVector({1}));
    auto tokenOutput = IRBuilder().CreateTensorVar(DT_FP32, {1}, CreateTestConstIntVector({1}));

    auto& view = IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_VIEW, {input}, {middle});
    view.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& reshape = IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_RESHAPE, {middle}, {reshapeOutput});
    auto& consumer = IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_EXP, {reshapeOutput}, {output});
    auto& tokenProducer = IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_ABS, {tokenInput}, {tokenOutput});
    int viewMagic = view.GetOpMagic();
    int reshapeMagic = reshape.GetOpMagic();

    auto inputToken = IRBuilder().CreateTokenVar(ir::Span::Unknown());
    auto viewResultToken = IRBuilder().CreateTokenVar(ir::Span::Unknown());
    auto reshapeResultToken = IRBuilder().CreateTokenVar(ir::Span::Unknown());
    auto toStmt = [](Operation& op) { return std::static_pointer_cast<const ir::Stmt>(op.shared_from_this()); };
    auto& dependency = function->GetVarDependency();
    tokenProducer.result_token_ = {inputToken};
    view.tokens_ = {inputToken};
    view.result_token_ = {viewResultToken};
    reshape.tokens_ = {viewResultToken};
    reshape.result_token_ = {reshapeResultToken};
    consumer.tokens_ = {reshapeResultToken};
    dependency.AddProducer(inputToken, toStmt(tokenProducer));
    dependency.AddConsumer(inputToken, toStmt(view));
    dependency.AddProducer(viewResultToken, toStmt(view));
    dependency.AddConsumer(viewResultToken, toStmt(reshape));
    dependency.AddProducer(reshapeResultToken, toStmt(reshape));
    dependency.AddConsumer(reshapeResultToken, toStmt(consumer));

    ViewReshapeAssembleReorderUtils utils;
    EXPECT_EQ(utils.TryRecordViewReshape(*function, view), SUCCESS);
    ASSERT_EQ(utils.viewReshapeRecords_.size(), 1U);
    utils.AppendViewReshapeRecords(*function);
    utils.CleanUp(*function);

    EXPECT_TRUE(IsOpRemoved(function->Operations(), viewMagic));
    EXPECT_TRUE(IsOpRemoved(function->Operations(), reshapeMagic));
    Operation* newReshape = FindOpByOpcode(function->Operations(), Opcode::OP_RESHAPE);
    Operation* newView = FindOpByOpcode(function->Operations(), Opcode::OP_VIEW);
    ASSERT_NE(newReshape, nullptr);
    ASSERT_NE(newView, nullptr);
    EXPECT_NE(std::find(newReshape->tokens_.begin(), newReshape->tokens_.end(), inputToken), newReshape->tokens_.end());
    ASSERT_FALSE(newReshape->result_token_.empty());
    EXPECT_NE(std::find(newView->tokens_.begin(), newView->tokens_.end(), newReshape->result_token_.front()),
              newView->tokens_.end());
    ASSERT_FALSE(newView->result_token_.empty());
    EXPECT_TRUE(dependency.HasConsumer(newView->result_token_.front(), toStmt(consumer)));
    EXPECT_NE(std::find(consumer.tokens_.begin(), consumer.tokens_.end(), newView->result_token_.front()),
              consumer.tokens_.end());
    EXPECT_FALSE(dependency.HasDependency(viewResultToken));
    EXPECT_FALSE(dependency.HasDependency(reshapeResultToken));
    EXPECT_NO_THROW(function->GetSortedOperations());
}

/*
 * View->Reshape with MatMul present.
 * Before: input{32,64} -> view -> middle{16,64} -> reshape -> output{1024}
 * After:  input{32,64} -> reshape(metadata) -> newMid{2048} -> view -> output{1024}
 *         The view is pushed below the reshape and its offset is remapped to {0}.
 */
TEST_F(RemoveRedundantReshapeTest, TestViewReshapeReorderWithMatmul)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestViewReshapeReorder",
                                                      "TestViewReshapeReorder", nullptr);
    ASSERT_NE(currFunctionPtr, nullptr);

    std::vector<int64_t> inputShape = {32, 64};
    std::vector<int64_t> middleShape = {16, 64};
    std::vector<int64_t> outputShape = {1024};
    std::vector<int64_t> matmulShape = {16, 16};

    auto input = IRBuilder().CreateTensorVar(DT_FP32, inputShape, CreateTestConstIntVector(inputShape));
    auto middle = IRBuilder().CreateTensorVar(DT_FP32, middleShape, CreateTestConstIntVector(middleShape));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, outputShape, CreateTestConstIntVector(outputShape));
    auto matmulA = IRBuilder().CreateTensorVar(DT_FP32, matmulShape, CreateTestConstIntVector(matmulShape));
    auto matmulB = IRBuilder().CreateTensorVar(DT_FP32, matmulShape, CreateTestConstIntVector(matmulShape));
    auto matmulC = IRBuilder().CreateTensorVar(DT_FP32, matmulShape, CreateTestConstIntVector(matmulShape));

    auto& viewOp = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {input}, {middle});
    viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    int viewMagic = viewOp.GetOpMagic();

    auto& reshapeOp = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_RESHAPE, {middle}, {output});
    int reshapeMagic = reshapeOp.GetOpMagic();

    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_A_MUL_B, {matmulA, matmulB}, {matmulC});

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->inCasts_.push_back(matmulA);
    currFunctionPtr->inCasts_.push_back(matmulB);
    currFunctionPtr->outCasts_.push_back(output);
    currFunctionPtr->outCasts_.push_back(matmulC);

    RemoveRedundantReshape pass;
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);

    auto ops = currFunctionPtr->Operations();
    EXPECT_TRUE(IsOpRemoved(ops, viewMagic)) << "Original View should be replaced in the reordered chain";
    EXPECT_TRUE(IsOpRemoved(ops, reshapeMagic)) << "Original Reshape should be replaced in the reordered chain";

    int viewCount = 0, reshapeCount = 0, matmulCount = 0;
    Operation* newReshape = nullptr;
    Operation* newView = nullptr;
    for (auto& op : ops) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            viewCount++;
            newView = const_cast<Operation*>(&op);
        } else if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            reshapeCount++;
            newReshape = const_cast<Operation*>(&op);
        } else if (op.GetOpcode() == Opcode::OP_A_MUL_B) {
            matmulCount++;
        }
    }
    EXPECT_EQ(viewCount, 1) << "One new View should be created after reorder";
    EXPECT_EQ(reshapeCount, 1) << "One metadata Reshape should be created after reorder";
    EXPECT_EQ(matmulCount, 1) << "MatMul should be preserved";
    ASSERT_NE(newReshape, nullptr);
    ASSERT_NE(newView, nullptr);

    // New metadata reshape consumes the original input and flattens {32,64} to {2048}.
    EXPECT_EQ(newReshape->GetIOperands().front(), input);
    EXPECT_EQ(newReshape->GetOOperands().front()->GetShape(), std::vector<int64_t>({2048}));

    // New view consumes the new reshape output and reproduces the original output{1024} at offset {0}.
    EXPECT_EQ(newView->GetIOperands().front(), newReshape->GetOOperands().front());
    EXPECT_EQ(newView->GetOOperands().front(), output);
    auto newViewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(newView->GetOpAttribute());
    ASSERT_NE(newViewAttr, nullptr);
    EXPECT_EQ(newViewAttr->GetFromOffset(), std::vector<int64_t>({0}));
}

/*
 * Reshape->Assemble with MatMul present but no cascaded assemble pattern.
 * Before: input{2048} -> reshape -> middle{32,64} -> assemble -> output{32,64}
 * After:  unchanged — reorder requires cascaded pattern (RESHAPE->ASSEMBLE->ASSEMBLE),
 *         so the pass skips reorder and preserves original ops.
 */
TEST_F(RemoveRedundantReshapeTest, TestReshapeAssembleReorderWithMatmul)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeAssembleReorder",
                                                      "TestReshapeAssembleReorder", nullptr);
    ASSERT_NE(currFunctionPtr, nullptr);

    std::vector<int64_t> inputShape = {2048};
    std::vector<int64_t> middleShape = {32, 64};
    std::vector<int64_t> outputShape = {32, 64};
    std::vector<int64_t> matmulShape = {16, 16};

    auto input = IRBuilder().CreateTensorVar(DT_FP32, inputShape, CreateTestConstIntVector(inputShape));
    auto middle = IRBuilder().CreateTensorVar(DT_FP32, middleShape, CreateTestConstIntVector(middleShape));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, outputShape, CreateTestConstIntVector(outputShape));
    auto matmulA = IRBuilder().CreateTensorVar(DT_FP32, matmulShape, CreateTestConstIntVector(matmulShape));
    auto matmulB = IRBuilder().CreateTensorVar(DT_FP32, matmulShape, CreateTestConstIntVector(matmulShape));
    auto matmulC = IRBuilder().CreateTensorVar(DT_FP32, matmulShape, CreateTestConstIntVector(matmulShape));

    auto& reshapeOp = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_RESHAPE, {input}, {middle});
    int reshapeMagic = reshapeOp.GetOpMagic();

    auto& assembleOp = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_ASSEMBLE, {middle}, {output});
    assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0}));
    int assembleMagic = assembleOp.GetOpMagic();

    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_A_MUL_B, {matmulA, matmulB}, {matmulC});

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->inCasts_.push_back(matmulA);
    currFunctionPtr->inCasts_.push_back(matmulB);
    currFunctionPtr->outCasts_.push_back(output);
    currFunctionPtr->outCasts_.push_back(matmulC);

    RemoveRedundantReshape pass;
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);

    auto ops = currFunctionPtr->Operations();
    EXPECT_FALSE(IsOpRemoved(ops, reshapeMagic)) << "Reshape should be kept (no cascaded pattern, reorder skipped)";
    EXPECT_FALSE(IsOpRemoved(ops, assembleMagic)) << "Assemble should be kept (no cascaded pattern, reorder skipped)";

    int assembleCount = 0, reshapeCount = 0, matmulCount = 0;
    for (auto& op : ops) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            assembleCount++;
        } else if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            reshapeCount++;
        } else if (op.GetOpcode() == Opcode::OP_A_MUL_B) {
            matmulCount++;
        }
    }
    EXPECT_EQ(assembleCount, 1) << "Original Assemble should be preserved";
    EXPECT_EQ(reshapeCount, 1) << "Original Reshape should be preserved";
    EXPECT_EQ(matmulCount, 1) << "MatMul should be preserved";
}

/*
 * sub->reshape->assemble with MatMul present.
 * After: unchanged — reorder skipped when reshape input is produced by SUB op.
 */
TEST_F(RemoveRedundantReshapeTest, TestSubReshapeAssembleSkipReorderWithMatmul)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestSubReshapeAssemble",
                                                      "TestSubReshapeAssemble", nullptr);
    ASSERT_NE(currFunctionPtr, nullptr);
    std::vector<int64_t> subShape = {32, 64};
    std::vector<int64_t> middleShape = {2048};
    std::vector<int64_t> matmulShape = {16, 16};
    auto subIn1 = IRBuilder().CreateTensorVar(DT_FP32, subShape, CreateTestConstIntVector(subShape));
    auto subIn2 = IRBuilder().CreateTensorVar(DT_FP32, subShape, CreateTestConstIntVector(subShape));
    auto subOut = IRBuilder().CreateTensorVar(DT_FP32, subShape, CreateTestConstIntVector(subShape));
    auto middle = IRBuilder().CreateTensorVar(DT_FP32, middleShape, CreateTestConstIntVector(middleShape));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, subShape, CreateTestConstIntVector(subShape));
    auto matmulA = IRBuilder().CreateTensorVar(DT_FP32, matmulShape, CreateTestConstIntVector(matmulShape));
    auto matmulB = IRBuilder().CreateTensorVar(DT_FP32, matmulShape, CreateTestConstIntVector(matmulShape));
    auto matmulC = IRBuilder().CreateTensorVar(DT_FP32, matmulShape, CreateTestConstIntVector(matmulShape));
    int subMagic = IRBuilder()
                       .CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_SUB, {subIn1, subIn2}, {subOut})
                       .GetOpMagic();
    int reshapeMagic = IRBuilder()
                           .CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_RESHAPE, {subOut}, {middle})
                           .GetOpMagic();
    auto& assembleOp = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_ASSEMBLE, {middle}, {output});
    assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    int assembleMagic = assembleOp.GetOpMagic();
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_A_MUL_B, {matmulA, matmulB}, {matmulC});
    currFunctionPtr->inCasts_ = {subIn1, subIn2, matmulA, matmulB};
    currFunctionPtr->outCasts_ = {output, matmulC};
    RemoveRedundantReshape pass;
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    auto ops = currFunctionPtr->Operations();
    EXPECT_FALSE(IsOpRemoved(ops, subMagic)) << "Sub should be kept";
    EXPECT_FALSE(IsOpRemoved(ops, reshapeMagic)) << "Reshape should be kept";
    EXPECT_FALSE(IsOpRemoved(ops, assembleMagic)) << "Assemble should be kept";
    int subCount = 0, reshapeCount = 0, assembleCount = 0, matmulCount = 0;
    for (auto& op : ops) {
        if (op.GetOpcode() == Opcode::OP_SUB) {
            subCount++;
        } else if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            reshapeCount++;
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            assembleCount++;
        } else if (op.GetOpcode() == Opcode::OP_A_MUL_B) {
            matmulCount++;
        }
    }
    EXPECT_EQ(subCount, 1);
    EXPECT_EQ(reshapeCount, 1);
    EXPECT_EQ(assembleCount, 1);
    EXPECT_EQ(matmulCount, 1);
}

struct FanoutGraphInfo {
    std::shared_ptr<Function> func;
    LogicalTensorPtr input;
    LogicalTensorPtr fanout1;
    LogicalTensorPtr fanout2;
    int viewMagic;
    int reshapeMagic;
    int fanoutView1Magic;
    int fanoutView2Magic;
};

static FanoutGraphInfo BuildViewReshapeFanoutGraph()
{
    FanoutGraphInfo info;
    info.func = std::make_shared<Function>(Program::GetInstance(), "TestViewReshapeFanout", "TestViewReshapeFanout",
                                           nullptr);

    std::vector<int64_t> inputShape = {4, 32};
    std::vector<int64_t> middleShape = {4, 16};
    std::vector<int64_t> reshapeOutShape = {64};
    std::vector<int64_t> fanoutShape = {32};
    std::vector<int64_t> matmulShape = {16, 16};

    auto input = IRBuilder().CreateTensorVar(DT_FP32, inputShape, CreateTestConstIntVector(inputShape));
    auto middle = IRBuilder().CreateTensorVar(DT_FP32, middleShape, CreateTestConstIntVector(middleShape));
    auto reshapeOut = IRBuilder().CreateTensorVar(DT_FP32, reshapeOutShape, CreateTestConstIntVector(reshapeOutShape));
    auto fanout1 = IRBuilder().CreateTensorVar(DT_FP32, fanoutShape, CreateTestConstIntVector(fanoutShape));
    auto fanout2 = IRBuilder().CreateTensorVar(DT_FP32, fanoutShape, CreateTestConstIntVector(fanoutShape));
    auto matmulA = IRBuilder().CreateTensorVar(DT_FP32, matmulShape, CreateTestConstIntVector(matmulShape));
    auto matmulB = IRBuilder().CreateTensorVar(DT_FP32, matmulShape, CreateTestConstIntVector(matmulShape));
    auto matmulC = IRBuilder().CreateTensorVar(DT_FP32, matmulShape, CreateTestConstIntVector(matmulShape));
    info.input = input;
    info.fanout1 = fanout1;
    info.fanout2 = fanout2;

    auto& viewOp = IRBuilder().CreateTensorOpStmt(*info.func, Opcode::OP_VIEW, {input}, {middle});
    viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 8}));
    info.viewMagic = viewOp.GetOpMagic();

    auto& reshapeOp = IRBuilder().CreateTensorOpStmt(*info.func, Opcode::OP_RESHAPE, {middle}, {reshapeOut});
    info.reshapeMagic = reshapeOp.GetOpMagic();

    auto& fanoutView1 = IRBuilder().CreateTensorOpStmt(*info.func, Opcode::OP_VIEW, {reshapeOut}, {fanout1});
    fanoutView1.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0}));
    info.fanoutView1Magic = fanoutView1.GetOpMagic();

    auto& fanoutView2 = IRBuilder().CreateTensorOpStmt(*info.func, Opcode::OP_VIEW, {reshapeOut}, {fanout2});
    fanoutView2.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{32}));
    info.fanoutView2Magic = fanoutView2.GetOpMagic();

    IRBuilder().CreateTensorOpStmt(*info.func, Opcode::OP_A_MUL_B, {matmulA, matmulB}, {matmulC});

    info.func->inCasts_.push_back(input);
    info.func->inCasts_.push_back(matmulA);
    info.func->inCasts_.push_back(matmulB);
    info.func->outCasts_.push_back(fanout1);
    info.func->outCasts_.push_back(fanout2);
    info.func->outCasts_.push_back(matmulC);
    return info;
}

/*
 * View->Reshape fanout with MatMul present.
 * Before: input{4,32} -> view(offset={0,8}) -> middle{4,16} -> reshape -> reshapeOut{64}
 *         reshapeOut -> fanoutView1(offset={0})  -> fanout1{32}
 *         reshapeOut -> fanoutView2(offset={32}) -> fanout2{32}
 * After:  input{4,32} -> reshape(metadata) -> newMid{128}
 *         newMid -> view(offset={8})  -> fanout1{32}
 *         newMid -> view(offset={72}) -> fanout2{32}
 *         The view is pushed below the reshape and fanout offsets are remapped.
 */
TEST_F(RemoveRedundantReshapeTest, TestViewReshapeFanoutWithMatmul)
{
    auto info = BuildViewReshapeFanoutGraph();
    ASSERT_NE(info.func, nullptr);

    RemoveRedundantReshape pass;
    EXPECT_EQ(pass.RunOnFunction(*info.func), SUCCESS);

    auto ops = info.func->Operations();
    EXPECT_TRUE(IsOpRemoved(ops, info.viewMagic)) << "Original View should be replaced in the fanout chain";
    EXPECT_TRUE(IsOpRemoved(ops, info.reshapeMagic)) << "Original Reshape should be replaced in the fanout chain";
    EXPECT_TRUE(IsOpRemoved(ops, info.fanoutView1Magic)) << "Original fanout view1 should be replaced";
    EXPECT_TRUE(IsOpRemoved(ops, info.fanoutView2Magic)) << "Original fanout view2 should be replaced";

    int viewCount = 0, reshapeCount = 0, matmulCount = 0;
    Operation* newReshape = nullptr;
    std::vector<Operation*> newFanoutViews;
    for (auto& op : ops) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            viewCount++;
            newFanoutViews.push_back(const_cast<Operation*>(&op));
        } else if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            reshapeCount++;
            newReshape = const_cast<Operation*>(&op);
        } else if (op.GetOpcode() == Opcode::OP_A_MUL_B) {
            matmulCount++;
        }
    }
    EXPECT_EQ(viewCount, 2) << "Two new fanout views should be created after reorder";
    EXPECT_EQ(reshapeCount, 1) << "One metadata reshape should be created after reorder";
    EXPECT_EQ(matmulCount, 1) << "MatMul should be preserved";
    ASSERT_NE(newReshape, nullptr);
    ASSERT_EQ(newFanoutViews.size(), 2u);

    // New metadata reshape consumes the original input and flattens {4,32} to {128}.
    EXPECT_EQ(newReshape->GetIOperands().front(), info.input);
    EXPECT_EQ(newReshape->GetOOperands().front()->GetShape(), std::vector<int64_t>({128}));

    // Both new fanout views consume the new reshape output with remapped offsets:
    // fanout1 (compact offset {0})  -> {8};  fanout2 (compact offset {32}) -> {72}.
    bool hasFanout1View = false, hasFanout2View = false;
    for (auto* viewOp : newFanoutViews) {
        EXPECT_EQ(viewOp->GetIOperands().front(), newReshape->GetOOperands().front());
        auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(viewOp->GetOpAttribute());
        ASSERT_NE(viewAttr, nullptr);
        auto offset = viewAttr->GetFromOffset();
        if (offset == std::vector<int64_t>({8})) {
            hasFanout1View = true;
            EXPECT_EQ(viewOp->GetOOperands().front(), info.fanout1);
        } else if (offset == std::vector<int64_t>({72})) {
            hasFanout2View = true;
            EXPECT_EQ(viewOp->GetOOperands().front(), info.fanout2);
        }
    }
    EXPECT_TRUE(hasFanout1View) << "Fanout view with remapped offset {8} should exist";
    EXPECT_TRUE(hasFanout2View) << "Fanout view with remapped offset {72} should exist";
}

struct ReshapeAssembleGraphInfo {
    std::shared_ptr<Function> func;
    LogicalTensorPtr middle;
    int reshapeMagic;
    int assembleMagic;
};

static ReshapeAssembleGraphInfo BuildReshapeAssembleGraph(const std::string& name)
{
    ReshapeAssembleGraphInfo info;
    info.func = std::make_shared<Function>(Program::GetInstance(), name, name, nullptr);

    std::vector<int64_t> inputShape = {32, 64};
    std::vector<int64_t> middleShape = {2048};
    std::vector<int64_t> outputShape = {2048};
    std::vector<int64_t> matmulShape = {16, 16};

    auto input = IRBuilder().CreateTensorVar(DT_FP32, inputShape, CreateTestConstIntVector(inputShape));
    auto middle = IRBuilder().CreateTensorVar(DT_FP32, middleShape, CreateTestConstIntVector(middleShape));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, outputShape, CreateTestConstIntVector(outputShape));
    auto matmulA = IRBuilder().CreateTensorVar(DT_FP32, matmulShape, CreateTestConstIntVector(matmulShape));
    auto matmulB = IRBuilder().CreateTensorVar(DT_FP32, matmulShape, CreateTestConstIntVector(matmulShape));
    auto matmulC = IRBuilder().CreateTensorVar(DT_FP32, matmulShape, CreateTestConstIntVector(matmulShape));
    info.middle = middle;

    auto& reshapeOp = IRBuilder().CreateTensorOpStmt(*info.func, Opcode::OP_RESHAPE, {input}, {middle});
    info.reshapeMagic = reshapeOp.GetOpMagic();

    auto& assembleOp = IRBuilder().CreateTensorOpStmt(*info.func, Opcode::OP_ASSEMBLE, {middle}, {output});
    assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0}));
    info.assembleMagic = assembleOp.GetOpMagic();

    IRBuilder().CreateTensorOpStmt(*info.func, Opcode::OP_A_MUL_B, {matmulA, matmulB}, {matmulC});

    info.func->inCasts_.push_back(input);
    info.func->inCasts_.push_back(matmulA);
    info.func->inCasts_.push_back(matmulB);
    info.func->outCasts_.push_back(output);
    info.func->outCasts_.push_back(matmulC);
    return info;
}

/*
 * Reshape->Assemble with A_MUL_B present, middle tensor has symbolic DynValidShape.
 * The DynValidShape contains a non-concrete scalar (e.g., runtime variable),
 * which causes BuildAssembledValidShape to produce incorrect results.
 * The reorder should be skipped.
 */
TEST_F(RemoveRedundantReshapeTest, TestReshapeAssembleSkipReorderWithSymbolicDynValidShape)
{
    auto info = BuildReshapeAssembleGraph("TestReshapeAssembleSymbolic");
    ASSERT_NE(info.func, nullptr);

    auto symL = CreateTestScalarVar("sym_L");
    ASSERT_FALSE(symL.ConcreteValid()) << "sym_L should be non-concrete";
    info.middle->UpdateDynValidShape({symL});

    RemoveRedundantReshape pass;
    EXPECT_EQ(pass.RunOnFunction(*info.func), SUCCESS);

    auto ops = info.func->Operations();
    EXPECT_FALSE(IsOpRemoved(ops, info.reshapeMagic))
        << "Reshape should be kept (symbolic DynValidShape → reorder skipped)";
    EXPECT_FALSE(IsOpRemoved(ops, info.assembleMagic))
        << "Assemble should be kept (symbolic DynValidShape → reorder skipped)";

    int assembleCount = 0, reshapeCount = 0, matmulCount = 0;
    for (auto& op : ops) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            assembleCount++;
        } else if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            reshapeCount++;
        } else if (op.GetOpcode() == Opcode::OP_A_MUL_B) {
            matmulCount++;
        }
    }
    EXPECT_EQ(assembleCount, 1);
    EXPECT_EQ(reshapeCount, 1);
    EXPECT_EQ(matmulCount, 1);
}

/*
 * Reshape->Assemble with A_MUL_B present, middle tensor has concrete DynValidShape
 * that matches the static shape. The reorder should proceed normally.
 */
TEST_F(RemoveRedundantReshapeTest, TestReshapeAssembleReorderWithConcreteDynValidShape)
{
    auto info = BuildReshapeAssembleGraph("TestReshapeAssembleConcrete");
    ASSERT_NE(info.func, nullptr);

    info.middle->UpdateDynValidShape(CreateTestConstIntVector({2048}));

    RemoveRedundantReshape pass;
    EXPECT_EQ(pass.RunOnFunction(*info.func), SUCCESS);

    auto ops = info.func->Operations();
    int matmulCount = 0;
    for (auto& op : ops) {
        if (op.GetOpcode() == Opcode::OP_A_MUL_B) {
            matmulCount++;
        }
    }
    EXPECT_EQ(matmulCount, 1);
}

TEST_F(RemoveRedundantReshapeTest, TestInferInputDynRawShapeFromOutputRejectsNullArgs)
{
    auto input = IRBuilder().CreateTensorVar(DT_FP32, std::vector<int64_t>{-1, 64}, CreateTestConstIntVector({-1, 64}));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, std::vector<int64_t>{-1, 4, 16},
                                              CreateTestConstIntVector({-1, 4, 16}));

    std::vector<SymbolicScalar> inferred;
    EXPECT_FALSE(ViewReshapeAssembleReorderUtils::InferInputDynRawShapeFromOutput(nullptr, output, inferred));
    EXPECT_FALSE(ViewReshapeAssembleReorderUtils::InferInputDynRawShapeFromOutput(input, nullptr, inferred));
}

TEST_F(RemoveRedundantReshapeTest, TestInferInputDynRawShapeFromOutputRejectsBadDynShapeSize)
{
    auto input = IRBuilder().CreateTensorVar(DT_FP32, std::vector<int64_t>{-1, 64}, CreateTestConstIntVector({-1, 64}));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, std::vector<int64_t>{-1, 4, 16},
                                              CreateTestConstIntVector({-1, 4, 16}));
    output->GetRawTensor()->UpdateDynRawShape(CreateTestConstIntVector({4, 4}));

    std::vector<SymbolicScalar> inferred;
    EXPECT_FALSE(ViewReshapeAssembleReorderUtils::InferInputDynRawShapeFromOutput(input, output, inferred));
}

TEST_F(RemoveRedundantReshapeTest, TestInferInputDynRawShapeFromOutputRejectsIncompatibleShapes)
{
    auto input = IRBuilder().CreateTensorVar(DT_FP32, std::vector<int64_t>{-1, 64}, CreateTestConstIntVector({-1, 64}));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, std::vector<int64_t>{-1, 128},
                                              CreateTestConstIntVector({-1, 128}));
    output->GetRawTensor()->UpdateDynRawShape(CreateTestConstIntVector({4, 128}));

    std::vector<SymbolicScalar> inferred;
    EXPECT_FALSE(ViewReshapeAssembleReorderUtils::InferInputDynRawShapeFromOutput(input, output, inferred));
}

/*
 * CreateMetadataReshape: when input DynRawShape has a negative value, it is
 * inferred from the output's concrete DynRawShape and updated in-place.
 */
TEST_F(RemoveRedundantReshapeTest, TestCreateMetadataReshapeInfersInputDynRawShape)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestInferDynRawShape", "TestInferDynRawShape",
                                           nullptr);

    // input: static shape {-1, 64}, dynRawShape {-1, 64} (has negative -> triggers inference)
    auto input = IRBuilder().CreateTensorVar(DT_FP32, std::vector<int64_t>{-1, 64}, CreateTestConstIntVector({-1, 64}));
    // output: static shape {-1, 4, 16}, dynRawShape {4, 4, 16} (concrete, source for inference)
    auto output = IRBuilder().CreateTensorVar(DT_FP32, std::vector<int64_t>{-1, 4, 16},
                                              CreateTestConstIntVector({-1, 4, 16}));
    output->GetRawTensor()->UpdateDynRawShape(CreateTestConstIntVector({4, 4, 16}));

    // srcOp: a dummy reshape op for attribute copying
    auto dummyIn = IRBuilder().CreateTensorVar(DT_FP32, std::vector<int64_t>{4}, CreateTestConstIntVector({4}));
    auto dummyOut = IRBuilder().CreateTensorVar(DT_FP32, std::vector<int64_t>{4}, CreateTestConstIntVector({4}));
    auto& srcOp = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_RESHAPE, {dummyIn}, {dummyOut});

    // Before: input dynRawShape has -1 at first dim
    const auto& beforeDynRawShape = input->GetRawTensor()->GetDynRawShape();
    ASSERT_EQ(beforeDynRawShape.size(), 2u);
    EXPECT_EQ(beforeDynRawShape[0].Concrete(), -1);

    ViewReshapeAssembleReorderUtils utils;
    utils.CreateMetadataReshape(*func, input, output, CreateTestConstIntVector({4, 4, 16}), ir::Span::Unknown(),
                                Operation::ScopeInfo(), srcOp);

    // After: input dynRawShape inferred to {4, 64}
    const auto& afterDynRawShape = input->GetRawTensor()->GetDynRawShape();
    ASSERT_EQ(afterDynRawShape.size(), 2u);
    EXPECT_EQ(afterDynRawShape[0].Concrete(), 4);
    EXPECT_EQ(afterDynRawShape[1].Concrete(), 64);
}

TEST_F(RemoveRedundantReshapeTest, TestCreateMetadataReshapeUpdatesOutputDynRawShape)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestUpdateOutputDynRawShape",
                                           "TestUpdateOutputDynRawShape", nullptr);

    auto input = IRBuilder().CreateTensorVar(DT_FP32, std::vector<int64_t>{-1, 2, 128},
                                             CreateTestConstIntVector({-1, 2, 128}));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, std::vector<int64_t>{-1, 256},
                                              CreateTestConstIntVector({-1, 256}));
    auto runtimeLength = CreateTestScalarVar("runtime_length_for_reshape_output");
    std::vector<SymbolicScalar> dynShape = {runtimeLength, SymbolicScalar(256)};

    auto dummyIn = IRBuilder().CreateTensorVar(DT_FP32, std::vector<int64_t>{1}, CreateTestConstIntVector({1}));
    auto dummyOut = IRBuilder().CreateTensorVar(DT_FP32, std::vector<int64_t>{1}, CreateTestConstIntVector({1}));
    auto& srcOp = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_RESHAPE, {dummyIn}, {dummyOut});

    ASSERT_EQ(output->GetRawTensor()->GetDynRawShape().size(), 2U);
    ASSERT_TRUE(output->GetRawTensor()->GetDynRawShape()[0].ConcreteValid());
    EXPECT_EQ(output->GetRawTensor()->GetDynRawShape()[0].Concrete(), -1);

    ViewReshapeAssembleReorderUtils utils;
    utils.CreateMetadataReshape(*func, input, output, dynShape, ir::Span::Unknown(), Operation::ScopeInfo(), srcOp);

    const auto& updatedDynRawShape = output->GetRawTensor()->GetDynRawShape();
    ASSERT_EQ(updatedDynRawShape.size(), dynShape.size());
    EXPECT_EQ(updatedDynRawShape[0].Dump(), dynShape[0].Dump());
    EXPECT_EQ(updatedDynRawShape[1].Concrete(), 256);
}

/*
 * CreateMetadataReshape: when inference fails (incompatible shapes), the
 * input's original DynRawShape is preserved unchanged.
 */
TEST_F(RemoveRedundantReshapeTest, TestCreateMetadataReshapeKeepsOriginalWhenInferFails)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestInferFail", "TestInferFail", nullptr);

    // input: {-1, 64}, dynRawShape {-1, 64}
    auto input = IRBuilder().CreateTensorVar(DT_FP32, std::vector<int64_t>{-1, 64}, CreateTestConstIntVector({-1, 64}));
    // output: {-1, 128} — incompatible (64 vs 128), inference fails
    auto output = IRBuilder().CreateTensorVar(DT_FP32, std::vector<int64_t>{-1, 128},
                                              CreateTestConstIntVector({-1, 128}));
    output->GetRawTensor()->UpdateDynRawShape(CreateTestConstIntVector({4, 128}));

    auto dummyIn = IRBuilder().CreateTensorVar(DT_FP32, std::vector<int64_t>{4}, CreateTestConstIntVector({4}));
    auto dummyOut = IRBuilder().CreateTensorVar(DT_FP32, std::vector<int64_t>{4}, CreateTestConstIntVector({4}));
    auto& srcOp = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_RESHAPE, {dummyIn}, {dummyOut});

    ViewReshapeAssembleReorderUtils utils;
    utils.CreateMetadataReshape(*func, input, output, CreateTestConstIntVector({4, 128}), ir::Span::Unknown(),
                                Operation::ScopeInfo(), srcOp);

    // After: input dynRawShape unchanged (still {-1, 64})
    const auto& afterDynRawShape = input->GetRawTensor()->GetDynRawShape();
    ASSERT_EQ(afterDynRawShape.size(), 2u);
    EXPECT_EQ(afterDynRawShape[0].Concrete(), -1);
    EXPECT_EQ(afterDynRawShape[1].Concrete(), 64);
}

/*
 * CreateMetadataReshape: when input DynRawShape has no negative value
 * (already concrete), inference is skipped and DynRawShape stays unchanged.
 */
TEST_F(RemoveRedundantReshapeTest, TestCreateMetadataReshapeSkipsWhenNoNegativeDynDim)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestNoNegDim", "TestNoNegDim", nullptr);

    // input: dynRawShape already concrete {8, 64} — no negative dims
    auto input = IRBuilder().CreateTensorVar(DT_FP32, std::vector<int64_t>{-1, 64}, CreateTestConstIntVector({-1, 64}));
    input->GetRawTensor()->UpdateDynRawShape(CreateTestConstIntVector({8, 64}));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, std::vector<int64_t>{-1, 4, 16},
                                              CreateTestConstIntVector({-1, 4, 16}));
    output->GetRawTensor()->UpdateDynRawShape(CreateTestConstIntVector({4, 4, 16}));

    auto dummyIn = IRBuilder().CreateTensorVar(DT_FP32, std::vector<int64_t>{4}, CreateTestConstIntVector({4}));
    auto dummyOut = IRBuilder().CreateTensorVar(DT_FP32, std::vector<int64_t>{4}, CreateTestConstIntVector({4}));
    auto& srcOp = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_RESHAPE, {dummyIn}, {dummyOut});

    ViewReshapeAssembleReorderUtils utils;
    utils.CreateMetadataReshape(*func, input, output, CreateTestConstIntVector({4, 4, 16}), ir::Span::Unknown(),
                                Operation::ScopeInfo(), srcOp);

    // After: input dynRawShape unchanged (still {8, 64}, not overwritten by output's {4,...})
    const auto& afterDynRawShape = input->GetRawTensor()->GetDynRawShape();
    ASSERT_EQ(afterDynRawShape.size(), 2u);
    EXPECT_EQ(afterDynRawShape[0].Concrete(), 8);
    EXPECT_EQ(afterDynRawShape[1].Concrete(), 64);
}

TEST_F(RemoveRedundantReshapeTest, TestDirectReshapeAssemblePreservesOutcastAlias)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestOutcastAlias", "TestOutcastAlias", nullptr);
    ASSERT_NE(func, nullptr);

    const std::vector<int64_t> inputShape{4, 4};
    const std::vector<int64_t> middleShape{16};
    const std::vector<int64_t> outputShape{16};
    auto input = IRBuilder().CreateTensorVar(DT_FP32, inputShape, CreateTestConstIntVector(inputShape));
    auto middle = IRBuilder().CreateTensorVar(DT_FP32, middleShape, CreateTestConstIntVector(middleShape));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, outputShape, CreateTestConstIntVector(outputShape));
    output->tensor->actualRawmagic = 4242;
    output->tensor->SetSymbol("outcast_alias");

    auto& reshapeOp = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_RESHAPE, {input}, {middle});
    auto& assembleOp = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_ASSEMBLE, {middle}, {output});
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0});
    assembleOp.SetOpAttribute(assembleAttr);
    func->outCasts_.push_back(output);

    ViewReshapeAssembleReorderUtils utils;
    ViewReshapeAssembleReorderUtils::ChainMatch match{input, middle, &assembleOp, output};
    EXPECT_EQ(utils.TryRecordDirectReshapeAssemble(*func, reshapeOp, assembleOp, match, *assembleAttr, outputShape,
                                                   CreateTestConstIntVector(outputShape),
                                                   CreateTestConstIntVector(middleShape),
                                                   CreateTestConstIntVector(outputShape)),
              SUCCESS);

    ASSERT_EQ(utils.reshapeAssembleRecords_.size(), 1U);
    const auto& generated = utils.reshapeAssembleRecords_.front().assembleOutput;
    ASSERT_NE(generated, nullptr);
    EXPECT_EQ(generated->GetRawMagic(), output->GetRawMagic());
    EXPECT_EQ(generated->Symbol(), output->Symbol());
}
