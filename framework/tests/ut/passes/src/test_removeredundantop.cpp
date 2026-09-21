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

#include <gtest/gtest.h>
#include <algorithm>
#include <vector>
#include <string>
#include "interface/function/function.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_utils/pass_operation_utils.h"
#include "symbolic_scalar_test_utils.h"
#include "pass_test_utils.h"
#include "tilefwk/tilefwk.h"
#include "ut_json/ut_json_tool.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "ir/span.h"
#include "passes/tile_graph_pass/graph_optimization/infer_discontinuous_input.h"
#include "computational_graph_builder.h"

#include "interface/tensor/irbuilder.h"
#define private public
#include "passes/tile_graph_pass/graph_optimization/remove_redundant_op.h"
#include "passes/pass_utils/remove_redundant_op_utils.h"

namespace npu {
namespace tile_fwk {
static const size_t kSizeZero = 0UL;
static const size_t kSizeOne = 1UL;
static const size_t kSizeSeven = 7UL;
static const size_t kSizeEight = 8UL;
static const size_t kSizeTen = 10UL;
static const size_t kSizeEleven = 11UL;
static const size_t kSizeThirteen = 13UL;
static const size_t kSizeForteen = 14UL;
static const int32_t kNumNegOne = -1;
static const uint16_t kNumZero = 0u;
static const uint16_t kNumOne = 1u;
static const uint16_t kNumTwo = 2u;
static const uint16_t kNumThree = 3u;
static const uint16_t kNumFour = 4u;
static const uint16_t kNumFive = 5u;
static const uint16_t kNumEight = 8u;
static const uint16_t kNumExpFour = 16u;
static const uint16_t kNumExpFive = 32u;
static const uint16_t kNumExpSix = 64u;
static const uint16_t kNumExpSeven = 128u;
static const uint16_t kNumExpEight = 256u;

namespace {
ir::StmtPtr ToStmtPtr(Operation& op) { return std::static_pointer_cast<const ir::Stmt>(op.shared_from_this()); }

ir::VarPtr AddTokenEdge(Function& function, Operation& producer, Operation& consumer)
{
    if (producer.result_token_.empty()) {
        producer.result_token_ = {IRBuilder().CreateTokenVar(producer.GetSpan())};
    }
    auto token = producer.result_token_.front();
    function.GetVarDependency().AddProducer(token, ToStmtPtr(producer));
    function.GetVarDependency().AddConsumer(token, ToStmtPtr(consumer));
    if (std::find(consumer.tokens_.begin(), consumer.tokens_.end(), token) == consumer.tokens_.end()) {
        consumer.tokens_.emplace_back(token);
    }
    return token;
}

bool HasToken(const Operation& op, const ir::VarPtr& token)
{
    return std::find(op.tokens_.begin(), op.tokens_.end(), token) != op.tokens_.end();
}

} // namespace

class TestRemoveRedundantOpPass : public ::testing::Test {
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
        TileShape::Current().SetVecTile({64, 64});
    }
    void TearDown() override {}
};

/*
TESTRemoveDummyExpand
inCast{8,16}->expand->ubTensor{8,16}->exp->outCast1{8,16}
                                    ->sqrt->outCast2{8,16}
                                    ->reciprocal->outCast3{8,16}
inCast{8,16}->exp->outCast1
            ->sqrt->outCast2
            ->reciprocal->outCast3
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest1)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp",
                                                      "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXPAND, {inCast}, {ubTensor});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {ubTensor}, {outCast1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_SQRT, {ubTensor}, {outCast2});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RECIPROCAL, {ubTensor}, {outCast3});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->outCasts_.push_back(outCast2);
    currFunctionPtr->outCasts_.push_back(outCast3);

    RemoveRedundantOp removeredundantpass;
    EXPECT_EQ(removeredundantpass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t expand_num = kNumZero;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_EXPAND) {
            ++expand_num;
        } else if (op.GetOpcode() == Opcode::OP_SQRT) {
            EXPECT_EQ(op.GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(op.GetInputOperand(kSizeZero), inCast);
        } else if (op.GetOpcode() == Opcode::OP_EXP) {
            EXPECT_EQ(op.GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(op.GetInputOperand(kSizeZero), inCast);
        } else if (op.GetOpcode() == Opcode::OP_RECIPROCAL) {
            EXPECT_EQ(op.GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(op.GetInputOperand(kSizeZero), inCast);
        }
    }
    EXPECT_EQ(expand_num, kNumZero);
}

/*
TESTRemoveDummyRegCopy
inCast{8,16}->regcopy->ubTensor1{16,8}->regcopy->ubTensor2{16,8}->exp->outCast1{16,8}
inCast{8,16}->regcopy->ubTensor1{16,8}->exp->outCast1{16,8}
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest2)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp",
                                                      "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumExpFour, kNumEight};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));

    auto& regcopy = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_REGISTER_COPY, {inCast}, {ubTensor1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_REGISTER_COPY, {ubTensor1}, {ubTensor2});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {ubTensor2}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp removeredundantpass;
    EXPECT_EQ(removeredundantpass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_NE(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t regcopy_num = kNumZero;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_REGISTER_COPY) {
            EXPECT_EQ(op.GetOpMagic(), regcopy.GetOpMagic());
            EXPECT_EQ(op.GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(op.GetInputOperand(kSizeZero), inCast);
            ++regcopy_num;
        } else if (op.GetOpcode() == Opcode::OP_EXP) {
            EXPECT_EQ(op.GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(op.GetInputOperand(kSizeZero), ubTensor1);
        }
    }
    EXPECT_EQ(regcopy_num, kNumOne);
}

/*
TESTRemoveDummyAssembleDDRSpecialCase(WARNING CASE)
inCast{8,16}->exp(any legal op)->ddrTensor1{8,16}  ->exp->outCast3{8,16}
                                     ->assemble->outCast1{8,16}
                                     ->assemble->outCast2{8,16}

assembles kept: output is outcast and input has other consumers (outcast protection)
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest3)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp",
                                                      "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    ubTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto outCast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape),
                                                               TileOpFormat::TILEOP_ND, "outCast1");
    outCast1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto outCast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape),
                                                               TileOpFormat::TILEOP_ND, "outCast2");
    outCast2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto outCast3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& exp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {inCast}, {ubTensor});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {ubTensor}, {outCast1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {ubTensor}, {outCast2});
    auto& exp2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {ubTensor}, {outCast3});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->outCasts_.push_back(outCast2);
    currFunctionPtr->outCasts_.push_back(outCast3);

    RemoveRedundantOp removeredundantpass;
    EXPECT_EQ(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);
    EXPECT_NE(removeredundantpass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t assemble_num = kNumZero;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assemble_num;
        }
    }
    EXPECT_EQ(assemble_num, kNumTwo);
    EXPECT_EQ(exp1.GetOutputOperandSize(), kSizeOne);
    EXPECT_EQ(exp2.GetInputOperandSize(), kSizeOne);
}

/*
TESTRemoveDummyView(WARNING CASE)
inCast{8,16}->exp->ddrTensor1{8,16}->exp->ubTensor2{8,16}->view->ubTensor3{8,16}->exp->outCast2{8,16}
                                  ->view->outCast1{8,16}                       ->reciprocal->outCast3{8,16}
                                                                               ->sqrt->outCast4{8,16}
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest4)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp",
                                                      "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto ubTensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto outCast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto outCast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto outCast3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto outCast4 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {inCast}, {ubTensor1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {ubTensor1}, {outCast1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {ubTensor1}, {ubTensor2});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {ubTensor2}, {ubTensor3});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {ubTensor3}, {outCast2});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RECIPROCAL, {ubTensor3}, {outCast3});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_SQRT, {ubTensor3}, {outCast4});
    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->outCasts_.push_back(outCast2);
    currFunctionPtr->outCasts_.push_back(outCast3);
    currFunctionPtr->outCasts_.push_back(outCast4);
    RemoveRedundantOp removeredundantpass;
    EXPECT_NE(removeredundantpass.PreCheck(*currFunctionPtr), SUCCESS);
}

/*
TESTRemoveAssemble1
inCast{8,16}->view->ddrTensor{8,16}->assemble->outCast{1,8,16}
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest6)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp",
                                                      "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ddrTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {inCast}, {ddrTensor});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {ddrTensor}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp removeredundantpass;
    EXPECT_NE(removeredundantpass.PreCheck(*currFunctionPtr), SUCCESS);
}

/*
TESTRemoveDummyRegCopy
inCast{8,16}/{a0,16}->regcopy->ubTensor1{8,16}/{a1,16}->regcopy->ubTensor2{16,8}/{a1,16}->exp->outCast1{16,8}
inCast{8,16}/{a0,16}->regcopy->ubTensor1{8,16}/{a1,16}->exp->outCast1{16,8}
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest7)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp",
                                                      "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    inCast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    ubTensor1->SetMemoryTypeBoth(MemoryType::MEM_UB);
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    ubTensor2->SetMemoryTypeBoth(MemoryType::MEM_UB);
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    outCast->SetMemoryTypeBoth(MemoryType::MEM_UB);

    auto& regcopy = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_REGISTER_COPY, {inCast}, {ubTensor1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_REGISTER_COPY, {ubTensor1}, {ubTensor2});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {ubTensor2}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp removeredundantpass;
    EXPECT_NE(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t regcopy_num = kNumZero;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_REGISTER_COPY) {
            EXPECT_EQ(op.GetOpMagic(), regcopy.GetOpMagic());
            EXPECT_EQ(op.GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(op.GetInputOperand(kSizeZero), inCast);
            ++regcopy_num;
        } else if (op.GetOpcode() == Opcode::OP_EXP) {
            EXPECT_EQ(op.GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(op.GetInputOperand(kSizeZero), ubTensor1);
        }
    }
    EXPECT_EQ(regcopy_num, kNumOne);
}

/*
TESTRemoveAssembleDDR2
inCast{8,16}->view->ubTensor1{8,16}->assemble->outCast1{8,16}
all delete
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest10)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp",
                                                      "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    std::vector<int64_t> offset = {kNumZero, kNumZero};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    inCast->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    ubTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape),
                                                              TileOpFormat::TILEOP_ND, "outCast");
    outCast->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {inCast}, {ubTensor}, [&offset](Operation& op) {
        op.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset));
    });
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_ASSEMBLE, {ubTensor}, {outCast},
        [&offset](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset)); });

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp removeredundantpass;
    EXPECT_EQ(removeredundantpass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t assemble_num = kNumZero;
    uint32_t view_num = kNumZero;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assemble_num;
        }
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++view_num;
        }
    }
    EXPECT_EQ(assemble_num, kNumZero);
    EXPECT_EQ(view_num, kNumZero);
}

/*
TESTRemoveAssembleDDR3
inCast1{8,16}->view->ubTensor1{16,16}->assemble->outCast1{16,16}
inCast2{8,16}->view->
all delete
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest11)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp",
                                                      "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumExpFour, kNumExpFour};
    std::vector<int64_t> offset1 = {kNumZero, kNumZero};
    std::vector<int64_t> offset2 = {kNumEight, kNumZero};
    auto inCast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    inCast1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto inCast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    inCast2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    ubTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2),
                                                              TileOpFormat::TILEOP_ND, "outCast");
    outCast->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_VIEW, {inCast1}, {ubTensor},
        [&offset1](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset1)); });
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_VIEW, {inCast2}, {ubTensor},
        [&offset2](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset2)); });
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {ubTensor}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast1);
    currFunctionPtr->inCasts_.push_back(inCast2);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp removeredundantpass;
    EXPECT_EQ(removeredundantpass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t assemble_num = kNumZero;
    uint32_t view_num = kNumZero;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assemble_num;
        }
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++view_num;
        }
    }
    EXPECT_EQ(assemble_num, kNumZero);
    EXPECT_EQ(view_num, kNumTwo);
}

/*
TESTPostExpand(DynValidShape not same)
inCast{8,16}->sqrt->ubTensor1{8,16}->expand->ubTensor2{8,16}->exp->outCast1{8,16}
inCast{8,16}->sqrt->ubTensor1{8,16}->expand->ubTensor2{8,16}->exp->outCast1{8,16}
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest12)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp",
                                                      "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    std::vector<SymbolicScalar> dynValidShape1;
    std::vector<SymbolicScalar> dynValidShape2;
    dynValidShape1.push_back(CreateTestScalarVar("Tensor1"));
    dynValidShape2.push_back(CreateTestScalarVar("Tensor2"));
    ubTensor1->UpdateDynValidShape(dynValidShape1);
    ubTensor2->UpdateDynValidShape(dynValidShape2);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXPAND, {ubTensor1}, {ubTensor2});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_SQRT, {inCast}, {ubTensor1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {ubTensor2}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp removeredundantpass;
    EXPECT_EQ(removeredundantpass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t expand_num = kNumZero;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_EXPAND) {
            ++expand_num;
        }
    }
    EXPECT_EQ(expand_num, kNumOne);
}

/*
view->exp(end assemble)->view(end assemble)->expand(end assemble)->exp(end assemble)
                                                                 ->exp(end assemble)

exp(end contract*3) ->exp(end contract)
                     ->exp(end contract)
outcast assemble is no longer split in ExpandFunction (kept as view semantic);
RemoveRedundantOp later converts it, yielding one extra slice+contract pair.
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpSTest1)
{
    // Define the shape of the Tensors
    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};

    PassManager& passManager = PassManager::Instance();

    Tensor input(DT_FP32, shape, "input");
    Tensor exp(DT_FP32, shape, "exp");
    Tensor view(DT_FP32, shape, "view");
    Tensor expand(DT_FP32, shape, "expand");
    Tensor output1(DT_FP32, shape, "output1");
    Tensor output2(DT_FP32, shape, "output2");

    FUNCTION("STCase1")
    {
        exp = Exp(input);
        view = View(exp, shape, {kNumZero, kNumZero});
        expand = Expand(view, shape);
        output1 = Exp(expand);
        output2 = Exp(expand);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase1");
    EXPECT_EQ(func->Operations().size(), kSizeEleven);

    passManager.RegisterStrategy("RemoveRedundantOpTestStrategy",
                                 {
                                     {"RemoveRedundantReshape", PassName::REMOVE_REDUNDANT_RESHAPE},
                                     {"InferMemoryConflict", PassName::INFER_MEMORY_CONFLICT},
                                     {"ExpandFunction", PassName::EXPAND_FUNCTION},
                                     {"SplitReshape", PassName::SPLIT_RESHAPE},
                                     {"SplitRawTensor", PassName::SPLIT_RAW_TENSOR},
                                     {"SplitLargeFanoutTensor", PassName::SPLIT_LARGE_FANOUT_TENSOR},
                                     {"AssignMemoryType", PassName::ASSIGN_MEMORY_TYPE},
                                     {"RemoveRedundantOp", PassName::REMOVE_REDUNDANT_OP},
                                 });
    auto ret = passManager.RunPass(Program::GetInstance(), *func, "RemoveRedundantOpTestStrategy");
    EXPECT_EQ(ret, SUCCESS);

    // ================== Verify the effect of the Pass ==================
    auto updated_operations = func->Operations();

    int slice_num = kNumZero;
    int expand_num = kNumZero;
    for (const auto& op : updated_operations) {
        if (op.GetOpcode() == Opcode::OP_SLICE) {
            slice_num++;
        } else if (op.GetOpcode() == Opcode::OP_EXPAND) {
            expand_num++;
        }
    }
    EXPECT_EQ(slice_num, kNumTwo);
    EXPECT_EQ(expand_num, kNumZero);
}

/*
view->exp(end assemble)->view(end assemble)->expand(end assemble)->exp(end assemble)
                                                                 ->exp(end assemble)

exp(end contract)->view(end assemble)->expand*4(end contract) ->exp*4(end contract)
                                                              ->exp*4(end contract)
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpSTest2)
{
    // Define the shape of the Tensors
    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    std::vector<int64_t> shape2 = {kNumExpFour, 1};
    std::vector<int64_t> shape3 = {kNumExpFour, kNumExpEight};

    PassManager& passManager = PassManager::Instance();

    Tensor input(DT_FP32, shape, "input");
    Tensor exp(DT_FP32, shape, "exp");
    Tensor view(DT_FP32, shape2, "view");
    Tensor expand(DT_FP32, shape3, "expand");
    Tensor output1(DT_FP32, shape3, "output1");
    Tensor output2(DT_FP32, shape3, "output2");

    FUNCTION("STCase2")
    {
        exp = Exp(input);
        view = View(exp, shape2, {kNumZero, kNumZero});
        expand = Expand(view, shape3);
        output1 = Exp(expand);
        output2 = Exp(expand);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase2");

    passManager.RegisterStrategy("RemoveRedundantOpTestStrategy",
                                 {
                                     {"RemoveRedundantReshape", PassName::REMOVE_REDUNDANT_RESHAPE},
                                     {"InferMemoryConflict", PassName::INFER_MEMORY_CONFLICT},
                                     {"ExpandFunction", PassName::EXPAND_FUNCTION},
                                     {"SplitReshape", PassName::SPLIT_RESHAPE},
                                     {"SplitRawTensor", PassName::SPLIT_RAW_TENSOR},
                                     {"SplitLargeFanoutTensor", PassName::SPLIT_LARGE_FANOUT_TENSOR},
                                     {"AssignMemoryType", PassName::ASSIGN_MEMORY_TYPE},
                                     {"RemoveRedundantOp", PassName::REMOVE_REDUNDANT_OP},
                                 });
    auto ret = passManager.RunPass(Program::GetInstance(), *func, "RemoveRedundantOpTestStrategy");
    EXPECT_EQ(ret, SUCCESS);

    // ================== Verify the effect of the Pass ==================
    auto updated_operations = func->Operations();

    int view_num = kNumZero;
    int expand_num = kNumZero;
    for (const auto& op : updated_operations) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            view_num++;
        } else if (op.GetOpcode() == Opcode::OP_EXPAND) {
            expand_num++;
        }
    }
    EXPECT_EQ(view_num, kNumOne);
    EXPECT_EQ(expand_num, kNumFour);
}

/*
view{64,64} ->exp{64,64} ->assemble{64, 64}
view{64,64} ->view{32,64} ->exp{64, 64} ->assemble{32, 64} ->assemble{64, 64}
            ->view{32,64} ->exp{64, 64} ->assemble{32, 64}
view{64,64} ->view{32,64} ->exp{64, 64} ->assemble{32, 64}
            ->view{32,64} ->exp{64, 64} ->assemble{32, 64}
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpSTest3)
{
    // Define the shape of the Tensors
    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    std::vector<int64_t> tile_shape = {kNumExpFive, kNumExpSix};

    PassManager& passManager = PassManager::Instance();
    passManager.RegisterStrategy("ExpandFunctionTestStrategy",
                                 {
                                     {"RemoveRedundantReshape", PassName::REMOVE_REDUNDANT_RESHAPE},
                                     {"InferMemoryConflict", PassName::INFER_MEMORY_CONFLICT},
                                     {"ExpandFunction", PassName::EXPAND_FUNCTION},
                                     {"SplitReshape", PassName::SPLIT_RESHAPE},
                                     {"SplitRawTensor", PassName::SPLIT_RAW_TENSOR},
                                     {"SplitLargeFanoutTensor", PassName::SPLIT_LARGE_FANOUT_TENSOR},
                                     {"AssignMemoryType", PassName::ASSIGN_MEMORY_TYPE},
                                 });

    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");

    FUNCTION("STCase3")
    {
        TileShape::Current().SetVecTile(tile_shape);
        output = Exp(input);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase3");

    passManager.RegisterStrategy("RemoveRedundantOpTestStrategy",
                                 {
                                     {"RemoveRedundantOp", PassName::REMOVE_REDUNDANT_OP},
                                 });
    EXPECT_EQ(passManager.RunPass(Program::GetInstance(), *func, "RemoveRedundantOpTestStrategy"), SUCCESS);

    // ================== Verify the effect of the Pass ==================
    int contract_after = kNumZero;
    for (const auto& op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_CONTRACT) {
            contract_after++;
        }
    }
    EXPECT_EQ(contract_after, kNumTwo);
}
void RemoveRedundantL1DataMoveGraph(std::shared_ptr<Function>& currFunctionPtr)
{
    std::shared_ptr<LogicalTensor> input_cast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> input_cast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{64, 16}, CreateTestConstIntVector(std::vector<int64_t>{64, 16}));
    std::shared_ptr<LogicalTensor> input_cast1_view = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> input_cast2_view = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{64, 16}, CreateTestConstIntVector(std::vector<int64_t>{64, 16}));
    input_cast1_view->SetMemoryTypeBoth(MEM_L1);
    input_cast2_view->SetMemoryTypeBoth(MEM_L1);
    std::shared_ptr<LogicalTensor> op_view_L1_out1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> op_view_L1_out2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{64, 16}, CreateTestConstIntVector(std::vector<int64_t>{64, 16}));
    op_view_L1_out1->SetMemoryTypeBoth(MEM_L1);
    op_view_L1_out2->SetMemoryTypeBoth(MEM_L1);
    std::shared_ptr<LogicalTensor> view_out1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 32}, CreateTestConstIntVector(std::vector<int64_t>{32, 32}));
    std::shared_ptr<LogicalTensor> view_out2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 32}, CreateTestConstIntVector(std::vector<int64_t>{32, 32}));
    std::shared_ptr<LogicalTensor> view_out3 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 16}, CreateTestConstIntVector(std::vector<int64_t>{32, 16}));
    std::shared_ptr<LogicalTensor> view_out4 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 16}, CreateTestConstIntVector(std::vector<int64_t>{32, 16}));
    std::shared_ptr<LogicalTensor> l0a_out1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 32}, CreateTestConstIntVector(std::vector<int64_t>{32, 32}));
    std::shared_ptr<LogicalTensor> l0a_out2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 32}, CreateTestConstIntVector(std::vector<int64_t>{32, 32}));
    std::shared_ptr<LogicalTensor> l0b_out1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 16}, CreateTestConstIntVector(std::vector<int64_t>{32, 16}));
    std::shared_ptr<LogicalTensor> l0b_out2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 16}, CreateTestConstIntVector(std::vector<int64_t>{32, 16}));
    std::shared_ptr<LogicalTensor> a_mul_b_out1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 16}, CreateTestConstIntVector(std::vector<int64_t>{32, 16}));
    std::shared_ptr<LogicalTensor> a_mul_b_out2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 16}, CreateTestConstIntVector(std::vector<int64_t>{32, 16}));
    auto& head_view_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {input_cast1},
                                                         {input_cast1_view});
    std::vector<int> newoffset{0, 0};
    auto viewAttribute = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute->SetToType(MemoryType::MEM_L1);
    head_view_op1.SetOpAttribute(viewAttribute);

    auto& head_view_op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {input_cast2},
                                                         {input_cast2_view});
    head_view_op2.SetOpAttribute(viewAttribute);

    auto& view_L1_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {input_cast1_view},
                                                       {op_view_L1_out1});
    view_L1_op1.SetOpAttribute(viewAttribute);
    auto& view_L1_op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {input_cast2_view},
                                                       {op_view_L1_out2});
    view_L1_op2.SetOpAttribute(viewAttribute);

    auto& view_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {op_view_L1_out1}, {view_out1});
    view_op1.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& view_op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {op_view_L1_out1}, {view_out2});
    view_op2.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 32}));
    auto& view_op3 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {op_view_L1_out2}, {view_out3});
    view_op3.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& view_op4 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {op_view_L1_out2}, {view_out4});
    view_op4.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{32, 0}));

    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_L1_TO_L0A, {view_out1}, {l0a_out1});
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_L1_TO_L0A, {view_out2}, {l0a_out2});
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_L1_TO_L0B, {view_out3}, {l0b_out1});
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_L1_TO_L0B, {view_out4}, {l0b_out2});

    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_A_MUL_B, {l0a_out1, l0b_out1}, {a_mul_b_out1});
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_A_MUL_B, {l0a_out2, l0b_out2}, {a_mul_b_out2});

    currFunctionPtr->inCasts_.push_back(input_cast1);
    currFunctionPtr->inCasts_.push_back(input_cast2);
    currFunctionPtr->outCasts_.push_back(a_mul_b_out1);
    currFunctionPtr->outCasts_.push_back(a_mul_b_out2);
}
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpL1DataMove)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "RemoveRedundantOpL1DataMove",
                                                      "RemoveRedundantOpL1DataMove", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("RemoveRedundantOpL1DataMove", currFunctionPtr);

    RemoveRedundantL1DataMoveGraph(currFunctionPtr);

    // 验证构图
    int view_count = 0;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            view_count++;
        }
    }
    EXPECT_EQ(view_count, 8);

    std::stringstream ssBefore;
    ssBefore << "Before_RemoveRedundantOp";

    // Call the pass
    RemoveRedundantOp removeRedundantOp;
    removeRedundantOp.PreCheck(*currFunctionPtr);
    currFunctionPtr->DumpJsonFile("./config/pass/json/removeRedundant_L1DataMove_before.json");
    removeRedundantOp.RunOnFunction(*currFunctionPtr);
    currFunctionPtr->DumpJsonFile("./config/pass/json/removeRedundant_L1DataMove_after.json");
    removeRedundantOp.PostCheck(*currFunctionPtr);

    std::stringstream ss;
    ss << "After_RemoveRedundantOp";

    // Validate the results
    int view_count_after_pass = 0;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            view_count_after_pass++;
        }
    }
    EXPECT_EQ(view_count_after_pass, 6);
}

/*
RemoveReshapeChain
inCast{8,16}->reshape->ubTensor1{16,8}->reshape->ubTensor2{32,4}->sqrt->outCast{32,4}
inCast{8,16}->reshape->ubTensor2{32,4}->sqrt->outCast{32,4}
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest13)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantReshape",
                                                      "TestRemoveRedundantReshape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumExpFour, kNumEight};
    std::vector<int64_t> shape3 = {kNumExpFive, kNumFour};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape3, CreateTestConstIntVector(shape3));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape3, CreateTestConstIntVector(shape3));

    auto& reshape1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {inCast}, {ubTensor1});
    auto& reshape2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    auto& sqrt = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_SQRT, {ubTensor2}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp removeredundantpass;
    auto status = removeredundantpass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    const auto& operations = currFunctionPtr->Operations();
    uint32_t reshape_num = kNumZero;
    for (auto& op : operations) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            EXPECT_EQ(reshape2.GetOpMagic(), op.GetOpMagic());
            EXPECT_EQ(reshape2.GetInputOperand(kSizeZero), inCast);
            ++reshape_num;
        } else if (op.GetOpcode() == Opcode::OP_SQRT) {
            EXPECT_EQ(sqrt.GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(sqrt.GetInputOperand(kSizeZero), ubTensor2);
        }
    }
    EXPECT_EQ(operations.Contains(reshape1), false);
    EXPECT_EQ(reshape_num, kNumOne);
}

/*
RemoveSameReshape
inCast{8,16}->reshape->ubTensor{8,16}->sqrt->outCast{8,16}
inCast{8,16}->sqrt->outCast{8,16}
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest14)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantReshape",
                                                      "TestRemoveRedundantReshape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {inCast}, {ubTensor});
    auto& sqrt = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_SQRT, {ubTensor}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp removeredundantpass;
    auto status = removeredundantpass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    uint32_t reshape_num = kNumZero;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            ++reshape_num;
        } else if (op.GetOpcode() == Opcode::OP_SQRT) {
            EXPECT_EQ(sqrt.GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(sqrt.GetInputOperand(kSizeZero), inCast);
        }
    }
    EXPECT_EQ(reshape_num, kNumZero);
}

/*
RemoveReshapeChainSeveralConsumer(WARNING CASE)
inCast{8,16}->reshape->ubTensor{8,16}->sqrt->outCast1{8,16}
                                    ->exp->outCast2{8,16}
                                    ->reshape->outCast3{16,8}
inCast{8,16}->sqrt->outCast1{8,16}
            ->exp->outCast2{8,16}
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest15)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantReshape",
                                                      "TestRemoveRedundantReshape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumExpFour, kNumEight};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto ubTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto outCast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto outCast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto outCast3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {inCast}, {ubTensor});
    auto& sqrt = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_SQRT, {ubTensor}, {outCast1});
    auto& exp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {ubTensor}, {outCast2});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {ubTensor}, {outCast3});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->outCasts_.push_back(outCast2);
    currFunctionPtr->outCasts_.push_back(outCast3);

    RemoveRedundantOp removeredundantpass;
    EXPECT_EQ(removeredundantpass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.RunOnFunction(*currFunctionPtr), SUCCESS);

    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_SQRT) {
            EXPECT_EQ(sqrt.GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(sqrt.GetInputOperand(kSizeZero), inCast);
        } else if (op.GetOpcode() == Opcode::OP_EXP) {
            EXPECT_EQ(exp.GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(exp.GetInputOperand(kSizeZero), inCast);
        }
    }
}

/*
RemoveReshapeChainSeveralConsumer
inCast{8,16}->reshape->ubTensor1{16,8}->exp->outCast1{16,8}
                                      ->reshape->ubTensor2{32,4}->sqrt->outCast2{32,4}
inCast{8,16}->reshape->ubTensor1{16,8}->exp->outCast1{16,8}
            ->reshape->ubTensor2{32,4}->sqrt->outCast2{32,4}
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest16)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantReshape",
                                                      "TestRemoveRedundantReshape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumExpFour, kNumEight};
    std::vector<int64_t> shape3 = {kNumExpFive, kNumFour};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto outCast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape3, CreateTestConstIntVector(shape3));
    auto outCast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape3, CreateTestConstIntVector(shape3));

    auto& reshape1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {inCast}, {ubTensor1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {ubTensor1}, {outCast1});
    auto& reshape2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_SQRT, {ubTensor2}, {outCast2});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->outCasts_.push_back(outCast2);

    RemoveRedundantOp removeredundantpass;
    auto status = removeredundantpass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    uint32_t reshape_num = kNumZero;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            ++reshape_num;
        }
    }
    EXPECT_EQ(reshape1.GetInputOperand(kSizeZero), inCast);
    EXPECT_EQ(reshape2.GetInputOperand(kSizeZero), inCast);
    EXPECT_EQ(reshape_num, kNumTwo);
}

/*
TESTRemoveIterative
inCast{8,16}->view->ubTensor1{8,16}->reshape->ubTensor2{8,16}->assemble->outCast1{8,16}
all delete
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest17)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp",
                                                      "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    std::vector<int64_t> offset = {kNumZero, kNumZero};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    inCast->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape),
                                                              TileOpFormat::TILEOP_ND, "outCast");
    outCast->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_VIEW, {inCast}, {ubTensor1},
        [&offset](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset)); });
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_ASSEMBLE, {ubTensor2}, {outCast},
        [&offset](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset)); });

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp removeredundantpass;
    EXPECT_EQ(removeredundantpass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t assemble_num = kNumZero;
    uint32_t view_num = kNumZero;
    uint32_t reshape_num = kNumZero;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assemble_num;
        } else if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++view_num;
        } else if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            ++reshape_num;
        }
    }
    EXPECT_EQ(assemble_num, kNumZero);
    EXPECT_EQ(view_num, kNumZero);
    EXPECT_EQ(reshape_num, kNumZero);
}

/*
TestRemoveAssembleSpecialCase
inCast{8,16}->exp->ddrTensor1{8,16} ->assemble-> outCast{8,16}
            ->exp->ddrTensor1{8,16} ->assemble->

inCast{8,16}->exp->ddrTensor1{8,16} ->assemble-> outCast{8,16}
            ->exp->ddrTensor1{8,16} ->assemble->
*/
TEST_F(TestRemoveRedundantOpPass, TestRemoveMoreAssembleSpecialCase)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp",
                                                      "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape),
                                                              TileOpFormat::TILEOP_ND, "outCast");
    outCast->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    ubTensor3->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {inCast}, {ubTensor1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {inCast}, {ubTensor2});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {ubTensor1}, {ubTensor3});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {ubTensor2}, {ubTensor3});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {ubTensor3}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);
    RemoveRedundantOp RemoveRedundantOpPass;
    EXPECT_EQ(RemoveRedundantOpPass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(RemoveRedundantOpPass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(RemoveRedundantOpPass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t assembleNum = kNumZero;
    for (const auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assembleNum;
        }
    }
    EXPECT_EQ(assembleNum, kNumTwo);
}

/*
TestRemoveAssembleDynSpecialCase
inCast{8,16}->exp->Tensor1{8,16} ->Reshape->Tensor2{8,16} ->assemble-> outCast{8,16}

inCast{8,16}->exp->Tensor1{8,16} ->Reshape->Tensor2{16,8} ->assemble-> outCast{8,16}
*/
TEST_F(TestRemoveRedundantOpPass, TestRemoveMoreAssembleDynSpecialCase)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp",
                                                      "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape1 = {kNumExpFour, kNumEight};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1),
                                                              TileOpFormat::TILEOP_ND, "outCast");
    outCast->UpdateDynValidShape({CreateTestScalarVar("output_0_Dim_0"), CreateTestScalarVar("output_0_Dim_1")});
    outCast->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    ubTensor1->UpdateDynValidShape({CreateTestScalarVar("Reshape_0_Dim_0"), CreateTestScalarVar("Reshape_0_Dim_1")});
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    ubTensor2->UpdateDynValidShape({CreateTestScalarVar("Reshape_0_Dim_0"), CreateTestScalarVar("Reshape_0_Dim_1")});

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {inCast}, {ubTensor1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {ubTensor2}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);
    RemoveRedundantOp RemoveRedundantOpPass;

    EXPECT_EQ(RemoveRedundantOpPass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(RemoveRedundantOpPass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(RemoveRedundantOpPass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t viewNum = kNumZero;
    uint32_t assembleNum = kNumZero;
    for (const auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++viewNum;
        }
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assembleNum;
        }
    }
    EXPECT_EQ(currFunctionPtr->GetOutcast()[0]->GetDynValidShape()[0].Dump(), "Reshape_0_Dim_0");
    EXPECT_EQ(currFunctionPtr->GetOutcast()[0]->GetDynValidShape()[1].Dump(), "Reshape_0_Dim_1");
    EXPECT_EQ(viewNum, kNumZero);
    EXPECT_EQ(assembleNum, kNumZero);
}

/*
TestGenerateViewSpecialCase
inCast1{8,16}->view->Tensor1{4,16}->assemble->outCast{16,16}
             ->view->Tensor2{4,16}->assemble->
inCast2{8,16}->mul->Tenosr3{8,16}->assemble->
inCast3{8,16}

inCast1{8,16}->view->Tensor1{4,16}->assemble->outCast{16,16}
             ->view->Tensor2{4,16}->assemble->
inCast2{8,16}->mul->Tenosr3{8,16}->assemble->
inCast3{8,16}
*/
TEST_F(TestRemoveRedundantOpPass, TestGenerateViewSpecialCase)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp",
                                                      "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape1 = {kNumExpFour, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumFour, kNumExpFour};
    std::vector<int64_t> offset1 = {kNumZero, kNumZero};
    std::vector<int64_t> offset2 = {kNumFour, kNumZero};
    std::vector<int64_t> offset3 = {kNumEight, kNumZero};
    auto inCast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto inCast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto inCast3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1),
                                                              TileOpFormat::TILEOP_ND, "outCast");
    outCast->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
    auto ubTensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    ubTensor3->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);

    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_VIEW, {inCast1}, {ubTensor1},
        [&offset1](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset1)); });
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_VIEW, {inCast1}, {ubTensor2},
        [&offset2](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset2)); });
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_MUL, {inCast2, inCast3}, {ubTensor3});
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_ASSEMBLE, {ubTensor1}, {outCast},
        [&offset1](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset1)); });
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_ASSEMBLE, {ubTensor2}, {outCast},
        [&offset2](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset2)); });
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_ASSEMBLE, {ubTensor3}, {outCast},
        [&offset3](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset3)); });

    currFunctionPtr->inCasts_.push_back(inCast1);
    currFunctionPtr->inCasts_.push_back(inCast2);
    currFunctionPtr->inCasts_.push_back(inCast3);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp RemoveRedundantOpPass;
    EXPECT_EQ(RemoveRedundantOpPass.RunOnFunction(*currFunctionPtr), SUCCESS);

    uint32_t viewNum = kNumZero;
    uint32_t assembleNum = kNumZero;
    for (const auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++viewNum;
        }
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assembleNum;
        }
    }
    EXPECT_EQ(viewNum, kNumTwo);
    EXPECT_EQ(assembleNum, kNumThree);
}

/*
TestGenerateViewDynOffsetCase
inCast{8,16}->view->Tensor1{4,16}->assemble->Tensor2{4,16}->exp->outCast{4,16}

inCast{8,16}->view->Tensor1{4,16}->exp->outCast{4,16}
*/
TEST_F(TestRemoveRedundantOpPass, TestGenerateViewDynOffsetCase)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp",
                                                      "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    uint32_t dynOffset = 0;
    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape1 = {kNumFour, kNumExpFour};
    std::vector<int64_t> offset = {kNumZero, kNumZero};
    std::vector<SymbolicScalar> newDynOffset{dynOffset, dynOffset};

    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1),
                                                              TileOpFormat::TILEOP_ND, "outCast");
    outCast->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);

    auto& viewOp = PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_VIEW, {inCast}, {ubTensor1}, [&offset, &newDynOffset](Operation& op) {
            op.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset, newDynOffset));
        });
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_ASSEMBLE, {ubTensor1}, {ubTensor2},
        [&offset](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset)); });
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {ubTensor2}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp RemoveRedundantOpPass;
    EXPECT_EQ(RemoveRedundantOpPass.RunOnFunction(*currFunctionPtr), SUCCESS);

    uint32_t viewNum = kNumZero;
    uint32_t assembleNum = kNumZero;
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(viewOp.GetOpAttribute().get());
    for (const auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++viewNum;
        }
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assembleNum;
        }
    }
    EXPECT_EQ(viewNum, kNumOne);
    EXPECT_EQ(assembleNum, kNumZero);
    EXPECT_EQ(viewOpAttribute->GetFromDynOffset()[0].Dump(), "0");
    EXPECT_EQ(viewOpAttribute->GetFromDynOffset()[1].Dump(), "0");
}

/*
TestOutcastMutiConsumerCase
inCast{8,16}->view->Tensor1{4,16}->assemble->outCast1{4,16}
                                  ->exp->Tensor2{4,16}->exp->outCast2{4,16}
graph unchanged: assemble kept because output is outcast and input has other consumers
*/
TEST_F(TestRemoveRedundantOpPass, TestOutcastMutiConsumerCase)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp",
                                                      "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape1 = {kNumFour, kNumExpFour};
    std::vector<int64_t> offset = {kNumZero, kNumZero};

    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1),
                                                               TileOpFormat::TILEOP_ND, "outCast");
    auto outCast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1),
                                                               TileOpFormat::TILEOP_ND, "outCast");
    outCast1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    outCast2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ddrTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    ddrTensor1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);

    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_ASSEMBLE, {ddrTensor1}, {outCast1},
        [&offset](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset)); });
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {inCast}, {ddrTensor1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {ddrTensor1}, {ubTensor2});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {ubTensor2}, {outCast2});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->outCasts_.push_back(outCast2);

    RemoveRedundantOp RemoveRedundantOpPass;
    EXPECT_EQ(RemoveRedundantOpPass.RunOnFunction(*currFunctionPtr), SUCCESS);

    uint32_t opNum = currFunctionPtr->Operations().size();
    EXPECT_EQ(opNum, kNumFour);
}

/*
TEST DynamicOutcast
inCast{8,16}->exp->ubTensor1{8,16}->view->ubTensor1{4,16}->assemble->outCast1{-1,16}
dynamic-axis, cannot delete, trans view/assemble to slice/contract
*/
TEST_F(TestRemoveRedundantOpPass, DynamicOutcast)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp",
                                                      "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumExpFour, kNumExpFour};
    std::vector<int64_t> shape3 = {kNumNegOne, kNumExpFour};
    std::vector<int64_t> offset = {kNumZero, kNumZero};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    inCast->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape3, CreateTestConstIntVector(shape3),
                                                              TileOpFormat::TILEOP_ND, "outCast");
    outCast->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {inCast}, {ubTensor1});
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_VIEW, {ubTensor1}, {ubTensor2},
        [&offset](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset)); });
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_ASSEMBLE, {ubTensor2}, {outCast},
        [&offset](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset)); });

    RemoveRedundantOp removeredundantpass;
    EXPECT_EQ(removeredundantpass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t contract_num = kNumZero;
    uint32_t slice_num = kNumZero;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_CONTRACT) {
            ++contract_num;
        }
        if (op.GetOpcode() == Opcode::OP_SLICE) {
            ++slice_num;
        }
    }
    EXPECT_EQ(contract_num, kNumOne);
    EXPECT_EQ(slice_num, kNumOne);
}

/*
TestOutcastToOutcastViewAssembleSkip
inCast{8,16}->exp->startOutCast{8,16}->view->ubTensor{4,16}->assemble->endOutCast{8,16}
startOutCast and endOutCast are both outcasts, cannot delete view/assemble.
*/
TEST_F(TestRemoveRedundantOpPass, TestOutcastToOutcastViewAssembleSkip)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp",
                                                      "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    std::vector<int64_t> viewShape = {kNumFour, kNumExpFour};
    std::vector<int64_t> offset = {kNumZero, kNumZero};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    inCast->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto startOutCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape),
                                                                   TileOpFormat::TILEOP_ND, "startOutCast");
    startOutCast->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, viewShape, CreateTestConstIntVector(viewShape));
    ubTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto endOutCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape),
                                                                 TileOpFormat::TILEOP_ND, "endOutCast");
    endOutCast->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(startOutCast);
    currFunctionPtr->outCasts_.push_back(endOutCast);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {inCast}, {startOutCast});
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_VIEW, {startOutCast}, {ubTensor},
        [&offset](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset)); });
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_ASSEMBLE, {ubTensor}, {endOutCast},
        [&offset](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset)); });

    RemoveRedundantOp removeredundantpass;
    EXPECT_EQ(removeredundantpass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t assembleNum = kNumZero;
    uint32_t viewNum = kNumZero;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_SLICE) {
            ++assembleNum;
        }
        if (op.GetOpcode() == Opcode::OP_CONTRACT) {
            ++viewNum;
        }
    }
    EXPECT_EQ(assembleNum, kNumOne);
    EXPECT_EQ(viewNum, kNumOne);
}

TEST_F(TestRemoveRedundantOpPass, AssembleDDR)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestAssembleDDRNoConsumer",
                                           "TestAssembleDDRNoConsumer", nullptr);
    EXPECT_TRUE(func != nullptr);

    std::vector<int64_t> shape = {64, 64};

    auto inTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ddrOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    ddrOut->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    PassOperationUtils::AddOperation(*func, Opcode::OP_ASSEMBLE, {inTensor}, {ddrOut});

    func->inCasts_.push_back(inTensor);

    RemoveRedundantOp pass;
    EXPECT_EQ(pass.PreCheck(*func), FAILED);
}

TEST_F(TestRemoveRedundantOpPass, ViewOp_OutCast)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestView_OutCast_Cover75",
                                           "TestView_OutCast_Cover75", nullptr);
    EXPECT_TRUE(func != nullptr);
    std::vector<int64_t> shape = {64, 64};
    auto in = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto view_out = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    PassOperationUtils::AddOperation(*func, Opcode::OP_VIEW, {in}, {view_out});

    func->inCasts_.push_back(in);
    func->outCasts_.push_back(view_out);
    auto dummy_out = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    PassOperationUtils::AddOperation(*func, (Opcode::OP_COPY_IN), {view_out}, {dummy_out});
    RemoveRedundantOp pass;
    Status ret = pass.PreCheck(*func);

    EXPECT_EQ(ret, FAILED);
}

TEST_F(TestRemoveRedundantOpPass, RegCopyNoConsumer)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestRegCopyNoConsumer", "TestRegCopyNoConsumer",
                                           nullptr);
    EXPECT_TRUE(func != nullptr);

    std::vector<int64_t> shape = {64, 64};
    auto in = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto out = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    PassOperationUtils::AddOperation(*func, Opcode::OP_REGISTER_COPY, {in}, {out});

    func->inCasts_.push_back(in);
    RemoveRedundantOp pass;
    Status ret = pass.PreCheck(*func);
    EXPECT_EQ(ret, FAILED);
}

/*
TestDynValidShapeInference
验证删除 assemble 时 DynValidShape 的正确推导：
inCast{8,16}->exp->Tensor1{8,16}[exp_dim0, exp_dim1]->Reshape->Tensor2{16,8}[reshape_dim0,
reshape_dim1]->assemble->outCast{16,8}

验证逻辑：
1. 初始 outCast 使用 shape 作为默认 concrete DynValidShape
2. 删除 assemble 后，验证 outCast 正确继承 Reshape 输出的 DynValidShape
3. 验证 DynValidShape 推导的完整过程（从输入 tensor 继承到输出 tensor）
*/
TEST_F(TestRemoveRedundantOpPass, TestDynValidShapeInference)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDynValidShapeInference",
                                                      "TestDynValidShapeInference", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape8x16 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape16x8 = {kNumExpFour, kNumEight};

    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape8x16, CreateTestConstIntVector(shape8x16));
    inCast->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    auto expOutput = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape8x16,
                                                                CreateTestConstIntVector(shape8x16));
    expOutput->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    expOutput->UpdateDynValidShape({CreateTestScalarVar("exp_output_dim0"), CreateTestScalarVar("exp_output_dim1")});

    auto reshapeOutput = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape16x8,
                                                                    CreateTestConstIntVector(shape16x8));
    reshapeOutput->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    reshapeOutput->UpdateDynValidShape(
        {CreateTestScalarVar("reshape_output_dim0"), CreateTestScalarVar("reshape_output_dim1")});

    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape16x8, CreateTestConstIntVector(shape16x8),
                                                              TileOpFormat::TILEOP_ND, "outCast");
    outCast->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {inCast}, {expOutput});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {expOutput}, {reshapeOutput});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {reshapeOutput}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    auto outcastBeforePass = currFunctionPtr->GetOutcast()[0];
    ASSERT_EQ(outcastBeforePass->GetDynValidShape().size(), kNumTwo);
    EXPECT_EQ(outcastBeforePass->GetDynValidShape()[0].Dump(), "reshape_output_dim0");
    EXPECT_EQ(outcastBeforePass->GetDynValidShape()[1].Dump(), "reshape_output_dim1");

    RemoveRedundantOp pass;
    EXPECT_EQ(pass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t assembleNum = kNumZero;
    uint32_t reshapeNum = kNumZero;
    for (const auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assembleNum;
        }
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            ++reshapeNum;
        }
    }

    auto outcastAfterPass = currFunctionPtr->GetOutcast()[0];
    EXPECT_FALSE(outcastAfterPass->GetDynValidShape().empty());
    EXPECT_EQ(outcastAfterPass->GetDynValidShape().size(), kNumTwo);
    EXPECT_EQ(outcastAfterPass->GetDynValidShape()[0].Dump(), "reshape_output_dim0");
    EXPECT_EQ(outcastAfterPass->GetDynValidShape()[1].Dump(), "reshape_output_dim1");
    EXPECT_EQ(assembleNum, kNumZero);
    EXPECT_EQ(reshapeNum, kNumOne);
}

/*
TestNewViewDynValidShapeInference
验证插入新 view 时 DynValidShape 的正确推导（Case2: GenerateNewView）：
inCast{8,16}->view->Tensor1{4,16}[view_dim0, view_dim1]->assemble->Tensor2{4,16}->exp->outCast{4,16}
                        (offset=[0,0])

删除 assemble 后插入新 view：
inCast{8,16}->view(Tensor2)->exp->outCast{4,16}

验证逻辑：
1. viewOutput和assembleOutput共享同一个RawTensor（这是正常情况）
2. viewOutput设置了DynValidShape
3. 删除assemble并插入新view后，验证：
   - 新view的输出Tensor正确继承RawTensor的DynValidShape（即viewOutput的DynValidShape）
   - 验证DynValidShape在RawTensor层面的正确传播
*/
TEST_F(TestRemoveRedundantOpPass, TestNewViewDynValidShapeInference)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestNewViewDynValidShape",
                                                      "TestNewViewDynValidShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape8x16 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape4x16 = {kNumFour, kNumExpFour};
    std::vector<int64_t> offset = {kNumZero, kNumZero};

    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape8x16, CreateTestConstIntVector(shape8x16));
    inCast->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    auto viewOutput = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape4x16,
                                                                 CreateTestConstIntVector(shape4x16));
    viewOutput->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
    viewOutput->UpdateDynValidShape({CreateTestScalarVar("view_output_dim0"), CreateTestScalarVar("view_output_dim1")});

    auto assembleOutput = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape4x16,
                                                                     CreateTestConstIntVector(shape4x16));
    assembleOutput->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);

    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape4x16, CreateTestConstIntVector(shape4x16),
                                                              TileOpFormat::TILEOP_ND, "outCast");
    outCast->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_VIEW, {inCast}, {viewOutput},
        [&offset](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset)); });

    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_ASSEMBLE, {viewOutput}, {assembleOutput},
        [&offset](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset)); });

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {assembleOutput}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp pass;
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);

    EXPECT_EQ(CountOpcode(currFunctionPtr, Opcode::OP_VIEW), kNumOne);
    EXPECT_EQ(CountOpcode(currFunctionPtr, Opcode::OP_ASSEMBLE), kNumZero);

    const Operation* newViewOp = FindSingleOp(currFunctionPtr, Opcode::OP_VIEW);
    EXPECT_TRUE(newViewOp != nullptr);
    auto viewAttribute = dynamic_cast<ViewOpAttribute*>(newViewOp->GetOpAttribute().get());
    EXPECT_TRUE(viewAttribute != nullptr);

    auto newViewOutput = newViewOp->GetOOperands()[0];
    EXPECT_FALSE(newViewOutput->GetDynValidShape().empty());
    EXPECT_EQ(newViewOutput->GetDynValidShape().size(), kNumTwo);
    EXPECT_EQ(newViewOutput->GetDynValidShape()[0].Dump(), "view_output_dim0");
    EXPECT_EQ(newViewOutput->GetDynValidShape()[1].Dump(), "view_output_dim1");
}

TEST_F(TestRemoveRedundantOpPass, ContractSliceFullShouldRemoveBoth)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestContractSliceFull",
                                                      "TestContractSliceFull", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    std::vector<int64_t> offset = {kNumZero, kNumZero};
    auto input = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto contractOutput = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto sliceOutput = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape), TileOpFormat::TILEOP_ND,
                                               "out");

    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_CONTRACT, {input}, {contractOutput},
        [&offset](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset)); });
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_SLICE, {contractOutput}, {sliceOutput},
        [&offset](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset)); });
    auto& expOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {sliceOutput}, {outCast});

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp pass;
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);

    EXPECT_EQ(CountOpcode(currFunctionPtr, Opcode::OP_CONTRACT), kNumZero);
    EXPECT_EQ(CountOpcode(currFunctionPtr, Opcode::OP_SLICE), kNumZero);
    EXPECT_EQ(expOp.GetInputOperand(kSizeZero), input);
}

TEST_F(TestRemoveRedundantOpPass, SingleContractWithNonZeroOffsetShouldRemove)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestSingleContractNonZeroOffset",
                                                      "TestSingleContractNonZeroOffset", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    std::vector<int64_t> offset = {kNumFour, kNumZero};
    auto input = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto contractOutput = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape), TileOpFormat::TILEOP_ND,
                                               "out");

    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_CONTRACT, {input}, {contractOutput},
        [&offset](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset)); });
    auto& expOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {contractOutput}, {outCast});

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp pass;
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);

    EXPECT_EQ(CountOpcode(currFunctionPtr, Opcode::OP_CONTRACT), kNumZero);
    EXPECT_EQ(expOp.GetInputOperand(kSizeZero), input);
}

TEST_F(TestRemoveRedundantOpPass, ContractMultiSliceShouldBypassFullAndGenerateViewForPartial)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestContractMultiSlice",
                                                      "TestContractMultiSlice", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> fullShape = {kNumEight, kNumExpFour};
    std::vector<int64_t> partShape = {kNumFour, kNumExpFour};
    std::vector<int64_t> zeroOffset = {kNumZero, kNumZero};
    std::vector<int64_t> partOffset = {kNumFour, kNumZero};
    auto input = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto contractOutput = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto fullSliceOutput = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto partSliceOutput = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto fullOut = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape),
                                               TileOpFormat::TILEOP_ND, "fullOut");
    auto partOut = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape),
                                               TileOpFormat::TILEOP_ND, "partOut");

    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_CONTRACT, {input}, {contractOutput},
        [&zeroOffset](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(zeroOffset)); });
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_SLICE, {contractOutput}, {fullSliceOutput},
        [&zeroOffset](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(zeroOffset)); });
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_SLICE, {contractOutput}, {partSliceOutput},
        [&partOffset](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(partOffset)); });
    auto& fullExp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {fullSliceOutput}, {fullOut});
    auto& partExp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {partSliceOutput}, {partOut});

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(fullOut);
    currFunctionPtr->outCasts_.push_back(partOut);

    std::vector<Operation*> contractSliceNewOps;
    bool contractSliceUpdated = false;
    EXPECT_EQ(RemoveRedundantOpUtils::ProcessContractSlice(*currFunctionPtr, contractSliceNewOps, contractSliceUpdated),
              SUCCESS);

    EXPECT_EQ(CountOpcode(currFunctionPtr, Opcode::OP_CONTRACT), kNumZero);
    EXPECT_EQ(CountOpcode(currFunctionPtr, Opcode::OP_SLICE), kNumZero);
    EXPECT_EQ(CountOpcode(currFunctionPtr, Opcode::OP_VIEW), kNumOne);
    EXPECT_EQ(fullExp.GetInputOperand(kSizeZero), input);
    auto viewOp = FindSingleOp(currFunctionPtr, Opcode::OP_VIEW);
    ASSERT_NE(viewOp, nullptr);
    EXPECT_EQ(partExp.GetInputOperand(kSizeZero), viewOp->GetOOperands().front());
    auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(viewOp->GetOpAttribute());
    ASSERT_NE(viewAttr, nullptr);
    EXPECT_EQ(viewAttr->GetFromOffset(), partOffset);
}

void VerifyMatmulContractMultiSliceNotFolded(Opcode matmulOpcode)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "TestMatmulContractMultiSlice",
                                               "TestMatmulContractMultiSlice", nullptr);
    ASSERT_NE(function, nullptr);

    std::vector<int64_t> fullShape = {kNumEight, kNumExpFour};
    std::vector<int64_t> partShape = {kNumFour, kNumExpFour};
    std::vector<int64_t> zeroOffset = {kNumZero, kNumZero};
    std::vector<int64_t> partOffset = {kNumFour, kNumZero};
    auto inputA = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto inputB = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto inputAcc = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto matmulOutput = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto contractOutput = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto sliceOutput0 = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto sliceOutput1 = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto out0 = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape),
                                            TileOpFormat::TILEOP_ND, "out0");
    auto out1 = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape),
                                            TileOpFormat::TILEOP_ND, "out1");

    std::vector<LogicalTensorPtr> matmulInputs = {inputA, inputB};
    if (matmulOpcode == Opcode::OP_A_MULACC_B) {
        matmulInputs.push_back(inputAcc);
    }
    PassOperationUtils::AddOperation(*function, matmulOpcode, matmulInputs, {matmulOutput});
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_CONTRACT, {matmulOutput}, {contractOutput},
        [&zeroOffset](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(zeroOffset)); });
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_SLICE, {contractOutput}, {sliceOutput0},
        [&zeroOffset](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(zeroOffset)); });
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_SLICE, {contractOutput}, {sliceOutput1},
        [&partOffset](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(partOffset)); });
    PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {sliceOutput0}, {out0});
    PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {sliceOutput1}, {out1});

    function->inCasts_ = {inputA, inputB};
    if (matmulOpcode == Opcode::OP_A_MULACC_B) {
        function->inCasts_.push_back(inputAcc);
    }
    function->outCasts_ = {out0, out1};

    std::vector<Operation*> newOps;
    bool operationUpdated = false;
    ASSERT_EQ(RemoveRedundantOpUtils::ProcessContractSlice(*function, newOps, operationUpdated), SUCCESS);

    EXPECT_FALSE(operationUpdated);
    EXPECT_TRUE(newOps.empty());
    EXPECT_EQ(CountOpcode(function, Opcode::OP_CONTRACT), kNumOne);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_SLICE), kNumTwo);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_VIEW), kNumZero);
}

TEST_F(TestRemoveRedundantOpPass, MatmulContractMultiSliceShouldNotFold)
{
    VerifyMatmulContractMultiSliceNotFolded(Opcode::OP_A_MUL_B);
}

TEST_F(TestRemoveRedundantOpPass, MatmulAccContractMultiSliceShouldNotFold)
{
    VerifyMatmulContractMultiSliceNotFolded(Opcode::OP_A_MULACC_B);
}

TEST_F(TestRemoveRedundantOpPass, MatmulTransposeContractMultiSliceShouldNotFold)
{
    VerifyMatmulContractMultiSliceNotFolded(Opcode::OP_A_MUL_BT);
    VerifyMatmulContractMultiSliceNotFolded(Opcode::OP_A_MULACC_BT);
    VerifyMatmulContractMultiSliceNotFolded(Opcode::OP_AT_MUL_B);
    VerifyMatmulContractMultiSliceNotFolded(Opcode::OP_AT_MUL_BT);
}

void VerifyMatmulContractSingleSliceNotFolded(Opcode matmulOpcode)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "TestMatmulContractSingleSlice",
                                               "TestMatmulContractSingleSlice", nullptr);
    ASSERT_NE(function, nullptr);

    std::vector<int64_t> fullShape = {kNumEight, kNumExpFour};
    std::vector<int64_t> partShape = {kNumFour, kNumExpFour};
    std::vector<int64_t> zeroOffset = {kNumZero, kNumZero};
    std::vector<int64_t> partOffset = {kNumFour, kNumZero};
    auto inputA = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto inputB = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto inputAcc = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto matmulOutput = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto contractOutput = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto sliceOutput = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto out = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape),
                                           TileOpFormat::TILEOP_ND, "out");

    std::vector<LogicalTensorPtr> matmulInputs = {inputA, inputB};
    if (matmulOpcode == Opcode::OP_A_MULACC_B) {
        matmulInputs.push_back(inputAcc);
    }
    PassOperationUtils::AddOperation(*function, matmulOpcode, matmulInputs, {matmulOutput});
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_CONTRACT, {matmulOutput}, {contractOutput},
        [&zeroOffset](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(zeroOffset)); });
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_SLICE, {contractOutput}, {sliceOutput},
        [&partOffset](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(partOffset)); });
    PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {sliceOutput}, {out});

    function->inCasts_ = {inputA, inputB};
    if (matmulOpcode == Opcode::OP_A_MULACC_B) {
        function->inCasts_.push_back(inputAcc);
    }
    function->outCasts_ = {out};

    std::vector<Operation*> newOps;
    bool operationUpdated = false;
    ASSERT_EQ(RemoveRedundantOpUtils::ProcessContractSlice(*function, newOps, operationUpdated), SUCCESS);

    EXPECT_FALSE(operationUpdated);
    EXPECT_TRUE(newOps.empty());
    EXPECT_EQ(CountOpcode(function, Opcode::OP_CONTRACT), kNumOne);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_SLICE), kNumOne);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_VIEW), kNumZero);
}

TEST_F(TestRemoveRedundantOpPass, MatmulContractSingleSliceShouldNotFold)
{
    VerifyMatmulContractSingleSliceNotFolded(Opcode::OP_A_MUL_B);
}

TEST_F(TestRemoveRedundantOpPass, MatmulAccContractSingleSliceShouldNotFold)
{
    VerifyMatmulContractSingleSliceNotFolded(Opcode::OP_A_MULACC_B);
}

TEST_F(TestRemoveRedundantOpPass, MatmulTransposeContractSingleSliceShouldNotFold)
{
    VerifyMatmulContractSingleSliceNotFolded(Opcode::OP_A_MUL_BT);
    VerifyMatmulContractSingleSliceNotFolded(Opcode::OP_A_MULACC_BT);
    VerifyMatmulContractSingleSliceNotFolded(Opcode::OP_AT_MUL_B);
    VerifyMatmulContractSingleSliceNotFolded(Opcode::OP_AT_MUL_BT);
}

TEST_F(TestRemoveRedundantOpPass, NonMatmulContractSingleSliceShouldFold)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "TestNonMatmulContractSingleSlice",
                                               "TestNonMatmulContractSingleSlice", nullptr);
    ASSERT_NE(function, nullptr);

    std::vector<int64_t> fullShape = {kNumEight, kNumExpFour};
    std::vector<int64_t> partShape = {kNumFour, kNumExpFour};
    std::vector<int64_t> zeroOffset = {kNumZero, kNumZero};
    std::vector<int64_t> partOffset = {kNumFour, kNumZero};
    auto input = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto contractOutput = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto sliceOutput = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape),
                                              TileOpFormat::TILEOP_ND, "out");

    PassOperationUtils::AddOperation(
        *function, Opcode::OP_CONTRACT, {input}, {contractOutput},
        [&zeroOffset](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(zeroOffset)); });
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_SLICE, {contractOutput}, {sliceOutput},
        [&partOffset](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(partOffset)); });
    auto& exp = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {sliceOutput}, {output});
    function->inCasts_ = {input};
    function->outCasts_ = {output};

    std::vector<Operation*> newOps;
    bool operationUpdated = false;
    ASSERT_EQ(RemoveRedundantOpUtils::ProcessContractSlice(*function, newOps, operationUpdated), SUCCESS);
    EXPECT_TRUE(operationUpdated);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_CONTRACT), kNumZero);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_SLICE), kNumZero);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_VIEW), kNumOne);
    auto generatedView = exp.GetInputOperand(kSizeZero)->GetProducers();
    ASSERT_EQ(generatedView.size(), kNumOne);
    EXPECT_EQ((*generatedView.begin())->GetOpcode(), Opcode::OP_VIEW);
}

TEST_F(TestRemoveRedundantOpPass, MultiContractSingleSliceShouldBecomeAssemble)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestMultiContractSingleSlice",
                                                      "TestMultiContractSingleSlice", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> fullShape = {kNumEight, kNumExpFour};
    std::vector<int64_t> partShape = {kNumFour, kNumExpFour};
    std::vector<int64_t> rawShape = {kNumFour, 24};
    std::vector<int64_t> inputOffset0 = {kNumZero, kNumZero};
    std::vector<int64_t> inputOffset1 = {kNumZero, kNumEight};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    std::vector<int64_t> offset1 = {kNumFour, kNumZero};
    IRBuilder builder;
    auto rawInput0 = builder.CreateRawTensor(DT_FP32, rawShape);
    auto rawInput1 = builder.CreateRawTensor(DT_FP32, rawShape);
    auto input0 = builder.CreateTensorVar(rawInput0, inputOffset0, partShape, CreateTestConstIntVector(partShape));
    auto input1 = builder.CreateTensorVar(rawInput1, inputOffset1, partShape, CreateTestConstIntVector(partShape));
    auto contractOutput = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto sliceOutput = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto outCast = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape),
                                               TileOpFormat::TILEOP_ND, "out");

    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_CONTRACT, {input0}, {contractOutput},
        [&offset0](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset0)); });
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_CONTRACT, {input1}, {contractOutput},
        [&offset1](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset1)); });
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_SLICE, {contractOutput}, {sliceOutput},
        [&offset0](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset0)); });
    auto& expOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {sliceOutput}, {outCast});

    currFunctionPtr->inCasts_.push_back(input0);
    currFunctionPtr->inCasts_.push_back(input1);
    currFunctionPtr->outCasts_.push_back(outCast);

    std::vector<Operation*> contractSliceNewOps;
    bool contractSliceUpdated = false;
    EXPECT_EQ(RemoveRedundantOpUtils::ProcessContractSlice(*currFunctionPtr, contractSliceNewOps, contractSliceUpdated),
              SUCCESS);

    EXPECT_EQ(CountOpcode(currFunctionPtr, Opcode::OP_CONTRACT), kNumZero);
    EXPECT_EQ(CountOpcode(currFunctionPtr, Opcode::OP_SLICE), kNumZero);
    EXPECT_EQ(CountOpcode(currFunctionPtr, Opcode::OP_ASSEMBLE), kNumTwo);
    EXPECT_EQ(expOp.GetInputOperand(kSizeZero), sliceOutput);
    for (const auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        EXPECT_EQ(op.GetOOperands().front(), sliceOutput);
    }
}

TEST_F(TestRemoveRedundantOpPass, MultiContractSingleSliceWithMaterializedMatmulInputShouldNotFold)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "TestMultiContractMatmulInput",
                                               "TestMultiContractMatmulInput", nullptr);
    ASSERT_NE(function, nullptr);

    std::vector<int64_t> fullShape = {kNumEight, kNumExpFour};
    std::vector<int64_t> partShape = {kNumFour, kNumExpFour};
    std::vector<int64_t> matmulRhsShape = {kNumExpFour, kNumExpFour};
    std::vector<int64_t> contractOffset0 = {kNumZero, kNumZero};
    std::vector<int64_t> contractOffset1 = {kNumFour, kNumZero};
    auto matmulInputA = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto matmulInputB = IRBuilder().CreateTensorVar(DT_FP32, matmulRhsShape, CreateTestConstIntVector(matmulRhsShape));
    auto matmulOutput = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto matmulContractOutput = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto matmulSliceOutput = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto otherInput = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto contractOutput = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto sliceOutput = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));

    PassOperationUtils::AddOperation(*function, Opcode::OP_A_MUL_B, {matmulInputA, matmulInputB}, {matmulOutput});
    PassOperationUtils::AddOperation(*function, Opcode::OP_CONTRACT, {matmulOutput}, {matmulContractOutput},
                                     [&contractOffset0](Operation& op) {
                                         op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(contractOffset0));
                                     });
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_SLICE, {matmulContractOutput}, {matmulSliceOutput},
        [&contractOffset0](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(contractOffset0)); });
    auto& matmulOuterContract = PassOperationUtils::AddOperation(
        *function, Opcode::OP_CONTRACT, {matmulSliceOutput}, {contractOutput}, [&contractOffset0](Operation& op) {
            op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(contractOffset0));
        });
    auto& otherOuterContract = PassOperationUtils::AddOperation(
        *function, Opcode::OP_CONTRACT, {otherInput}, {contractOutput}, [&contractOffset1](Operation& op) {
            op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(contractOffset1));
        });
    auto& finalSlice = PassOperationUtils::AddOperation(
        *function, Opcode::OP_SLICE, {contractOutput}, {sliceOutput},
        [&contractOffset0](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(contractOffset0)); });
    PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {sliceOutput}, {output});

    std::vector<Operation*> newOps;
    bool operationUpdated = false;
    ASSERT_EQ(RemoveRedundantOpUtils::ProcessContractSlice(*function, newOps, operationUpdated), SUCCESS);

    // The inner materialized matmul chain (matmulOutput -> CONTRACT -> SLICE -> matmulSliceOutput)
    // is eliminated by ProcessContractSliceContract: the outer contract reads the matmul output
    // directly. The outer multi-contract-single-slice chain itself must stay untouched.
    EXPECT_TRUE(operationUpdated);
    EXPECT_EQ(matmulOuterContract.GetOpcode(), Opcode::OP_CONTRACT);
    EXPECT_EQ(matmulOuterContract.GetIOperands().front(), matmulOutput);
    EXPECT_EQ(otherOuterContract.GetOpcode(), Opcode::OP_CONTRACT);
    EXPECT_EQ(contractOutput->GetProducers().size(), kNumTwo);
    EXPECT_FALSE(finalSlice.IsDeleted());
    EXPECT_EQ(CountOpcode(function, Opcode::OP_CONTRACT), kNumTwo);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_SLICE), kNumOne);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_ASSEMBLE), kNumZero);
}

TEST_F(TestRemoveRedundantOpPass, MultiContractSingleSliceWithUnalignedInputOffsetShouldNotBecomeAssemble)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestUnalignedMultiContractSingleSlice",
                                                      "TestUnalignedMultiContractSingleSlice", nullptr);
    ASSERT_NE(currFunctionPtr, nullptr);

    std::vector<int64_t> fullShape = {kNumEight, kNumExpFour};
    std::vector<int64_t> partShape = {kNumFour, kNumExpFour};
    std::vector<int64_t> alignedRawShape = {kNumFour, kNumExpFour};
    std::vector<int64_t> unalignedRawShape = {kNumFour, 17};
    std::vector<int64_t> alignedInputOffset = {kNumZero, kNumZero};
    std::vector<int64_t> unalignedInputOffset = {kNumZero, kNumOne};
    std::vector<int64_t> contractOffset0 = {kNumZero, kNumZero};
    std::vector<int64_t> contractOffset1 = {kNumFour, kNumZero};
    IRBuilder builder;
    auto alignedRawInput = builder.CreateRawTensor(DT_FP32, alignedRawShape);
    auto unalignedRawInput = builder.CreateRawTensor(DT_FP32, unalignedRawShape);
    auto input0 = builder.CreateTensorVar(alignedRawInput, alignedInputOffset, partShape,
                                          CreateTestConstIntVector(partShape));
    auto input1 = builder.CreateTensorVar(unalignedRawInput, unalignedInputOffset, partShape,
                                          CreateTestConstIntVector(partShape));
    auto contractOutput = builder.CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto sliceOutput = builder.CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto outCast = builder.CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape),
                                           TileOpFormat::TILEOP_ND, "out");

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_CONTRACT, {input0}, {contractOutput},
                                     [&contractOffset0](Operation& op) {
                                         op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(contractOffset0));
                                     });
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_CONTRACT, {input1}, {contractOutput},
                                     [&contractOffset1](Operation& op) {
                                         op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(contractOffset1));
                                     });
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_SLICE, {contractOutput}, {sliceOutput},
        [&contractOffset0](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(contractOffset0)); });
    auto& expOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {sliceOutput}, {outCast});

    currFunctionPtr->inCasts_.push_back(input0);
    currFunctionPtr->inCasts_.push_back(input1);
    currFunctionPtr->outCasts_.push_back(outCast);

    std::vector<Operation*> contractSliceNewOps;
    bool contractSliceUpdated = false;
    EXPECT_EQ(RemoveRedundantOpUtils::ProcessContractSlice(*currFunctionPtr, contractSliceNewOps, contractSliceUpdated),
              SUCCESS);

    EXPECT_FALSE(contractSliceUpdated);
    EXPECT_EQ(CountOpcode(currFunctionPtr, Opcode::OP_CONTRACT), kNumTwo);
    EXPECT_EQ(CountOpcode(currFunctionPtr, Opcode::OP_SLICE), kNumOne);
    EXPECT_EQ(CountOpcode(currFunctionPtr, Opcode::OP_ASSEMBLE), kNumZero);
    EXPECT_EQ(expOp.GetInputOperand(kSizeZero), sliceOutput);
}

TEST_F(TestRemoveRedundantOpPass, ContractFullL1SliceShouldTransferRequirementToPrecedingSlice)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "ContractFullL1", "ContractFullL1", nullptr);
    ASSERT_NE(function, nullptr);

    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    std::vector<int64_t> offset = {kNumZero, kNumZero};
    auto input = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto precedingOutput = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto contractOutput = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto sliceOutput = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    precedingOutput->SetMemoryTypeBoth(MemoryType::MEM_UB);
    sliceOutput->SetMemoryTypeBoth(MemoryType::MEM_L1);

    auto precedingAttr = std::make_shared<ViewOpAttribute>(offset);
    precedingAttr->SetToType(MemoryType::MEM_UB);
    auto& precedingSlice = PassOperationUtils::AddOperation(
        *function, Opcode::OP_SLICE, {input}, {precedingOutput},
        [&precedingAttr](Operation& op) { op.SetOpAttribute(precedingAttr); });
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_CONTRACT, {precedingOutput}, {contractOutput}, [&offset](Operation& op) {
            op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset));
        });
    auto l1Attr = std::make_shared<ViewOpAttribute>(offset);
    l1Attr->SetToType(MemoryType::MEM_L1);
    PassOperationUtils::AddOperation(*function, Opcode::OP_SLICE, {contractOutput}, {sliceOutput},
                                     [&l1Attr](Operation& op) {
                                         op.SetOpAttribute(l1Attr);
                                         op.SetAttr(OpAttributeKey::copyInMode, static_cast<int64_t>(0));
                                         op.SetAttr("op_attr_copy_in_l1_k_index", static_cast<int64_t>(1));
                                         op.SetAttr("op_attr_copy_in_l1_padding_mode", static_cast<int64_t>(2));
                                     });
    auto& exp = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {sliceOutput}, {output});

    std::vector<Operation*> newOps;
    bool operationUpdated = false;
    ASSERT_EQ(RemoveRedundantOpUtils::ProcessContractSlice(*function, newOps, operationUpdated), SUCCESS);

    EXPECT_TRUE(operationUpdated);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_CONTRACT), kNumZero);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_SLICE), kNumOne);
    EXPECT_EQ(precedingSlice.GetOOperands().front(), sliceOutput);
    EXPECT_EQ(exp.GetIOperands().front(), sliceOutput);
    auto transferredAttr = std::dynamic_pointer_cast<ViewOpAttribute>(precedingSlice.GetOpAttribute());
    ASSERT_NE(transferredAttr, nullptr);
    EXPECT_EQ(transferredAttr->GetTo(), MemoryType::MEM_L1);
    EXPECT_EQ(sliceOutput->GetMemoryTypeOriginal(), MemoryType::MEM_L1);
    EXPECT_EQ(sliceOutput->GetMemoryTypeToBe(), MemoryType::MEM_L1);
    int64_t transferredCopyInMode = -1;
    EXPECT_TRUE(precedingSlice.GetAttr<int64_t>(OpAttributeKey::copyInMode, transferredCopyInMode));
    EXPECT_EQ(transferredCopyInMode, kNumZero);
    int64_t transferredKIndex = -1;
    EXPECT_TRUE(precedingSlice.GetAttr<int64_t>("op_attr_copy_in_l1_k_index", transferredKIndex));
    EXPECT_EQ(transferredKIndex, kNumOne);
    int64_t transferredPaddingMode = -1;
    EXPECT_TRUE(precedingSlice.GetAttr<int64_t>("op_attr_copy_in_l1_padding_mode", transferredPaddingMode));
    EXPECT_EQ(transferredPaddingMode, kNumTwo);
    for (const auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_SLICE) {
            EXPECT_EQ(op.GetOpMagic(), precedingSlice.GetOpMagic());
        }
    }
}

TEST_F(TestRemoveRedundantOpPass, ContractPartialL1SliceShouldGenerateL1View)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "ContractPartialL1", "ContractPartialL1",
                                               nullptr);
    ASSERT_NE(function, nullptr);

    std::vector<int64_t> fullShape = {kNumEight, kNumExpFour};
    std::vector<int64_t> partShape = {kNumFour, kNumExpFour};
    std::vector<int64_t> zeroOffset = {kNumZero, kNumZero};
    std::vector<int64_t> partOffset = {kNumFour, kNumZero};
    auto input = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto precedingOutput = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto contractOutput = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto sliceOutput = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    precedingOutput->SetMemoryTypeBoth(MemoryType::MEM_UB);
    sliceOutput->SetMemoryTypeBoth(MemoryType::MEM_L1);

    auto precedingAttr = std::make_shared<ViewOpAttribute>(zeroOffset);
    precedingAttr->SetToType(MemoryType::MEM_UB);
    auto& precedingSlice = PassOperationUtils::AddOperation(
        *function, Opcode::OP_SLICE, {input}, {precedingOutput},
        [&precedingAttr](Operation& op) { op.SetOpAttribute(precedingAttr); });
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_CONTRACT, {precedingOutput}, {contractOutput}, [&zeroOffset](Operation& op) {
            op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, zeroOffset));
        });
    auto l1Attr = std::make_shared<ViewOpAttribute>(partOffset);
    l1Attr->SetToType(MemoryType::MEM_L1);
    PassOperationUtils::AddOperation(*function, Opcode::OP_SLICE, {contractOutput}, {sliceOutput},
                                     [&l1Attr](Operation& op) { op.SetOpAttribute(l1Attr); });
    auto& exp = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {sliceOutput}, {output});

    std::vector<Operation*> newOps;
    bool operationUpdated = false;
    ASSERT_EQ(RemoveRedundantOpUtils::ProcessContractSlice(*function, newOps, operationUpdated), SUCCESS);

    EXPECT_EQ(CountOpcode(function, Opcode::OP_CONTRACT), kNumZero);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_SLICE), kNumOne);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_VIEW), kNumOne);
    auto transferredAttr = std::dynamic_pointer_cast<ViewOpAttribute>(precedingSlice.GetOpAttribute());
    ASSERT_NE(transferredAttr, nullptr);
    EXPECT_EQ(transferredAttr->GetTo(), MemoryType::MEM_L1);
    auto* generatedView = FindSingleOp(function, Opcode::OP_VIEW);
    ASSERT_NE(generatedView, nullptr);
    auto generatedAttr = std::dynamic_pointer_cast<ViewOpAttribute>(generatedView->GetOpAttribute());
    ASSERT_NE(generatedAttr, nullptr);
    EXPECT_EQ(generatedAttr->GetTo(), MemoryType::MEM_L1);
    EXPECT_EQ(exp.GetIOperands().front(), generatedView->GetOOperands().front());
}

TEST_F(TestRemoveRedundantOpPass, SliceContractMultiL1SlicesShouldKeepFanout)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "SliceContractMultiL1", "SliceContractMultiL1",
                                               nullptr);
    ASSERT_NE(function, nullptr);

    std::vector<int64_t> fullShape = {kNumEight, kNumExpFour};
    std::vector<int64_t> partShape = {kNumFour, kNumExpFour};
    std::vector<int64_t> zeroOffset = {kNumZero, kNumZero};
    std::vector<int64_t> partOffset = {kNumFour, kNumZero};
    auto input = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto precedingOutput = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto contractOutput = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto sliceOutput0 = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto sliceOutput1 = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto output0 = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto output1 = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    precedingOutput->SetMemoryTypeBoth(MemoryType::MEM_UB);
    contractOutput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    sliceOutput0->SetMemoryTypeBoth(MemoryType::MEM_L1);
    sliceOutput1->SetMemoryTypeBoth(MemoryType::MEM_L1);

    auto precedingAttr = std::make_shared<ViewOpAttribute>(zeroOffset);
    precedingAttr->SetToType(MemoryType::MEM_UB);
    auto& precedingSlice = PassOperationUtils::AddOperation(
        *function, Opcode::OP_SLICE, {input}, {precedingOutput},
        [&precedingAttr](Operation& op) { op.SetOpAttribute(precedingAttr); });
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_CONTRACT, {precedingOutput}, {contractOutput}, [&zeroOffset](Operation& op) {
            op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, zeroOffset));
        });
    auto l1Attr0 = std::make_shared<ViewOpAttribute>(zeroOffset);
    l1Attr0->SetToType(MemoryType::MEM_L1);
    auto& l1Slice0 = PassOperationUtils::AddOperation(
        *function, Opcode::OP_SLICE, {contractOutput}, {sliceOutput0}, [&l1Attr0](Operation& op) {
            op.SetOpAttribute(l1Attr0);
            op.SetAttr(OpAttributeKey::copyInMode, static_cast<int64_t>(0));
            op.SetAttr("op_attr_copy_in_l1_k_index", static_cast<int64_t>(1));
        });
    auto l1Attr1 = std::make_shared<ViewOpAttribute>(partOffset);
    l1Attr1->SetToType(MemoryType::MEM_L1);
    auto& l1Slice1 = PassOperationUtils::AddOperation(
        *function, Opcode::OP_SLICE, {contractOutput}, {sliceOutput1}, [&l1Attr1](Operation& op) {
            op.SetOpAttribute(l1Attr1);
            op.SetAttr(OpAttributeKey::copyInMode, static_cast<int64_t>(1));
            op.SetAttr("op_attr_copy_in_l1_k_index", static_cast<int64_t>(2));
        });
    auto& exp0 = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {sliceOutput0}, {output0});
    auto& exp1 = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {sliceOutput1}, {output1});

    function->inCasts_.push_back(input);
    function->outCasts_.push_back(output0);
    function->outCasts_.push_back(output1);

    std::vector<Operation*> newOps;
    bool operationUpdated = false;
    ASSERT_EQ(RemoveRedundantOpUtils::Process(*function, newOps, operationUpdated), SUCCESS);

    EXPECT_TRUE(operationUpdated);
    EXPECT_TRUE(newOps.empty());
    EXPECT_EQ(CountOpcode(function, Opcode::OP_CONTRACT), kNumZero);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_SLICE), kNumThree);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_VIEW), kNumZero);
    EXPECT_EQ(l1Slice0.GetIOperands().front(), precedingOutput);
    EXPECT_EQ(l1Slice1.GetIOperands().front(), precedingOutput);
    EXPECT_EQ(exp0.GetIOperands().front(), sliceOutput0);
    EXPECT_EQ(exp1.GetIOperands().front(), sliceOutput1);

    auto precedingSliceAttr = std::dynamic_pointer_cast<ViewOpAttribute>(precedingSlice.GetOpAttribute());
    ASSERT_NE(precedingSliceAttr, nullptr);
    EXPECT_EQ(precedingSliceAttr->GetTo(), MemoryType::MEM_UB);
    EXPECT_EQ(precedingOutput->GetMemoryTypeOriginal(), MemoryType::MEM_UB);

    auto sliceAttr0 = std::dynamic_pointer_cast<ViewOpAttribute>(l1Slice0.GetOpAttribute());
    auto sliceAttr1 = std::dynamic_pointer_cast<ViewOpAttribute>(l1Slice1.GetOpAttribute());
    ASSERT_NE(sliceAttr0, nullptr);
    ASSERT_NE(sliceAttr1, nullptr);
    EXPECT_EQ(sliceAttr0->GetTo(), MemoryType::MEM_L1);
    EXPECT_EQ(sliceAttr1->GetTo(), MemoryType::MEM_L1);
    EXPECT_EQ(sliceAttr0->GetFromOffset(), zeroOffset);
    EXPECT_EQ(sliceAttr1->GetFromOffset(), partOffset);
    EXPECT_EQ(sliceOutput0->GetMemoryTypeOriginal(), MemoryType::MEM_L1);
    EXPECT_EQ(sliceOutput1->GetMemoryTypeOriginal(), MemoryType::MEM_L1);

    int64_t copyInMode0 = -1;
    int64_t copyInMode1 = -1;
    int64_t kIndex0 = -1;
    int64_t kIndex1 = -1;
    EXPECT_TRUE(l1Slice0.GetAttr<int64_t>(OpAttributeKey::copyInMode, copyInMode0));
    EXPECT_TRUE(l1Slice1.GetAttr<int64_t>(OpAttributeKey::copyInMode, copyInMode1));
    EXPECT_TRUE(l1Slice0.GetAttr<int64_t>("op_attr_copy_in_l1_k_index", kIndex0));
    EXPECT_TRUE(l1Slice1.GetAttr<int64_t>("op_attr_copy_in_l1_k_index", kIndex1));
    EXPECT_EQ(copyInMode0, kNumZero);
    EXPECT_EQ(copyInMode1, kNumOne);
    EXPECT_EQ(kIndex0, kNumOne);
    EXPECT_EQ(kIndex1, kNumTwo);
}

TEST_F(TestRemoveRedundantOpPass, MultiContractL1SliceWithMultipleProducersShouldNotFold)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "MultiContractL1", "MultiContractL1", nullptr);
    ASSERT_NE(function, nullptr);

    std::vector<int64_t> fullShape = {kNumEight, kNumExpFour};
    std::vector<int64_t> partShape = {kNumFour, kNumExpFour};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    std::vector<int64_t> offset1 = {kNumFour, kNumZero};
    auto input0 = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto input1 = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto part0 = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto part1 = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto contractOutput = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto sliceOutput = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    part0->SetMemoryTypeBoth(MemoryType::MEM_UB);
    part1->SetMemoryTypeBoth(MemoryType::MEM_UB);

    auto preAttr0 = std::make_shared<ViewOpAttribute>(offset0);
    preAttr0->SetToType(MemoryType::MEM_UB);
    auto& preSlice0 = PassOperationUtils::AddOperation(*function, Opcode::OP_SLICE, {input0}, {part0},
                                                       [&preAttr0](Operation& op) { op.SetOpAttribute(preAttr0); });
    auto preAttr1 = std::make_shared<ViewOpAttribute>(offset0);
    preAttr1->SetToType(MemoryType::MEM_UB);
    auto& preSlice1 = PassOperationUtils::AddOperation(*function, Opcode::OP_SLICE, {input1}, {part1},
                                                       [&preAttr1](Operation& op) { op.SetOpAttribute(preAttr1); });
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_CONTRACT, {part0}, {contractOutput}, [&offset0](Operation& op) {
            op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset0));
        });
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_CONTRACT, {part1}, {contractOutput}, [&offset1](Operation& op) {
            op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset1));
        });
    auto l1Attr = std::make_shared<ViewOpAttribute>(offset0);
    l1Attr->SetToType(MemoryType::MEM_L1);
    PassOperationUtils::AddOperation(*function, Opcode::OP_SLICE, {contractOutput}, {sliceOutput},
                                     [&l1Attr](Operation& op) { op.SetOpAttribute(l1Attr); });
    PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {sliceOutput}, {output});

    std::vector<Operation*> newOps;
    bool operationUpdated = false;
    ASSERT_EQ(RemoveRedundantOpUtils::ProcessContractSlice(*function, newOps, operationUpdated), SUCCESS);

    EXPECT_FALSE(operationUpdated);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_CONTRACT), kNumTwo);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_SLICE), kNumThree);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_ASSEMBLE), kNumZero);
    for (auto* precedingSlice : {&preSlice0, &preSlice1}) {
        auto attr = std::dynamic_pointer_cast<ViewOpAttribute>(precedingSlice->GetOpAttribute());
        ASSERT_NE(attr, nullptr);
        EXPECT_EQ(attr->GetTo(), MemoryType::MEM_UB);
        EXPECT_EQ(precedingSlice->GetOOperands().front()->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    }
}

TEST_F(TestRemoveRedundantOpPass, ContractL1SliceWithoutPrecedingSliceShouldNotFold)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "ContractL1WithoutPreSlice",
                                               "ContractL1WithoutPreSlice", nullptr);
    ASSERT_NE(function, nullptr);

    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    std::vector<int64_t> offset = {kNumZero, kNumZero};
    auto input = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto contractOutput = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto sliceOutput = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_CONTRACT, {input}, {contractOutput}, [&offset](Operation& op) {
            op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset));
        });
    auto l1Attr = std::make_shared<ViewOpAttribute>(offset);
    l1Attr->SetToType(MemoryType::MEM_L1);
    PassOperationUtils::AddOperation(*function, Opcode::OP_SLICE, {contractOutput}, {sliceOutput},
                                     [&l1Attr](Operation& op) { op.SetOpAttribute(l1Attr); });
    PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {sliceOutput}, {output});

    std::vector<Operation*> newOps;
    bool operationUpdated = false;
    ASSERT_EQ(RemoveRedundantOpUtils::ProcessContractSlice(*function, newOps, operationUpdated), SUCCESS);

    EXPECT_FALSE(operationUpdated);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_CONTRACT), kNumOne);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_SLICE), kNumOne);
}

TEST_F(TestRemoveRedundantOpPass, ContractL1SliceWithMultipleInputProducersShouldNotFold)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "ContractL1MultiInputProducer",
                                               "ContractL1MultiInputProducer", nullptr);
    ASSERT_NE(function, nullptr);

    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    std::vector<int64_t> offset = {kNumZero, kNumZero};
    auto input0 = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto input1 = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto contractInput = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto contractOutput = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto sliceOutput = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    contractInput->SetMemoryTypeBoth(MemoryType::MEM_UB);
    sliceOutput->SetMemoryTypeBoth(MemoryType::MEM_L1);

    auto preAttr = std::make_shared<ViewOpAttribute>(offset);
    preAttr->SetToType(MemoryType::MEM_UB);
    PassOperationUtils::AddOperation(*function, Opcode::OP_SLICE, {input0}, {contractInput},
                                     [&preAttr](Operation& op) { op.SetOpAttribute(preAttr); });
    PassOperationUtils::AddOperation(*function, Opcode::OP_SLICE, {input1}, {contractInput},
                                     [&preAttr](Operation& op) { op.SetOpAttribute(preAttr); });
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_CONTRACT, {contractInput}, {contractOutput}, [&offset](Operation& op) {
            op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset));
        });
    auto l1Attr = std::make_shared<ViewOpAttribute>(offset);
    l1Attr->SetToType(MemoryType::MEM_L1);
    PassOperationUtils::AddOperation(*function, Opcode::OP_SLICE, {contractOutput}, {sliceOutput},
                                     [&l1Attr](Operation& op) { op.SetOpAttribute(l1Attr); });
    auto& exp = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {sliceOutput}, {output});

    std::vector<Operation*> newOps;
    bool operationUpdated = false;
    ASSERT_EQ(RemoveRedundantOpUtils::ProcessContractSlice(*function, newOps, operationUpdated), SUCCESS);

    EXPECT_FALSE(operationUpdated);
    EXPECT_TRUE(newOps.empty());
    EXPECT_EQ(CountOpcode(function, Opcode::OP_CONTRACT), kNumOne);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_SLICE), kNumThree);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_VIEW), kNumZero);
    EXPECT_EQ(exp.GetIOperands().front(), sliceOutput);
}

TEST_F(TestRemoveRedundantOpPass, SliceContractPartialShouldGenerateViewWithoutMemoryTransform)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestSliceContractPartialView",
                                                      "TestSliceContractPartialView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> inputShape = {kNumEight, kNumExpFour};
    std::vector<int64_t> partShape = {kNumFour, kNumExpFour};
    std::vector<int64_t> offset = {kNumZero, kNumZero};
    auto input = IRBuilder().CreateTensorVar(DT_FP32, inputShape, CreateTestConstIntVector(inputShape));
    auto sliceOutput = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto contractOutput = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto outCast = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape),
                                               TileOpFormat::TILEOP_ND, "out");

    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_SLICE, {input}, {sliceOutput},
        [&offset](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset)); });
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_CONTRACT, {sliceOutput}, {contractOutput},
        [&offset](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset)); });
    auto& expOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {contractOutput}, {outCast});

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp pass;
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);

    EXPECT_EQ(CountOpcode(currFunctionPtr, Opcode::OP_SLICE), kNumZero);
    EXPECT_EQ(CountOpcode(currFunctionPtr, Opcode::OP_CONTRACT), kNumZero);
    EXPECT_EQ(CountOpcode(currFunctionPtr, Opcode::OP_VIEW), kNumOne);
    auto viewOp = FindSingleOp(currFunctionPtr, Opcode::OP_VIEW);
    ASSERT_NE(viewOp, nullptr);
    EXPECT_EQ(expOp.GetInputOperand(kSizeZero), viewOp->GetOOperands().front());
}

TEST_F(TestRemoveRedundantOpPass, SingleSliceContractZeroOffsetOutcastShouldBecomeView)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "SingleSliceContractZeroOffsetOutcast",
                                               "SingleSliceContractZeroOffsetOutcast", nullptr);
    ASSERT_NE(function, nullptr);

    std::vector<int64_t> inputShape = {kNumEight, kNumExpFour};
    std::vector<int64_t> partShape = {kNumFour, kNumExpFour};
    std::vector<int64_t> sliceOffset = {kNumFour, kNumZero};
    std::vector<int64_t> contractOffset = {kNumZero, kNumZero};
    auto input = IRBuilder().CreateTensorVar(DT_FP32, inputShape, CreateTestConstIntVector(inputShape));
    auto sliceInput = IRBuilder().CreateTensorVar(DT_FP32, inputShape, CreateTestConstIntVector(inputShape));
    auto sliceOutput = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto contractOutput = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape),
                                                      TileOpFormat::TILEOP_ND, "out");
    input->SetMemoryTypeBoth(MemoryType::MEM_UB);
    sliceInput->SetMemoryTypeBoth(MemoryType::MEM_UB);
    sliceOutput->SetMemoryTypeBoth(MemoryType::MEM_UB);
    contractOutput->SetMemoryTypeBoth(MemoryType::MEM_UB);

    PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {input}, {sliceInput});
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_SLICE, {sliceInput}, {sliceOutput},
        [&sliceOffset](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(sliceOffset)); });
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_CONTRACT, {sliceOutput}, {contractOutput},
        [&contractOffset](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(contractOffset)); });
    function->inCasts_.push_back(input);
    function->outCasts_.push_back(contractOutput);

    RemoveRedundantOp pass;
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);

    EXPECT_EQ(CountOpcode(function, Opcode::OP_SLICE), kNumZero);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_CONTRACT), kNumZero);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_VIEW), kNumOne);
    auto* view = FindSingleOp(function, Opcode::OP_VIEW);
    ASSERT_NE(view, nullptr);
    auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(view->GetOpAttribute());
    ASSERT_NE(viewAttr, nullptr);
    EXPECT_EQ(viewAttr->GetFromOffset(), sliceOffset);
    EXPECT_EQ(view->GetIOperands().front(), sliceInput);
    EXPECT_EQ(view->GetOOperands().front(), contractOutput);
}

TEST_F(TestRemoveRedundantOpPass, SliceContractPartialShouldGenerateSliceWithMemoryTransform)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestSliceContractPartialSlice",
                                                      "TestSliceContractPartialSlice", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> inputShape = {kNumEight, kNumExpFour};
    std::vector<int64_t> partShape = {kNumFour, kNumExpFour};
    std::vector<int64_t> offset = {kNumZero, kNumZero};
    auto input = IRBuilder().CreateTensorVar(DT_FP32, inputShape, CreateTestConstIntVector(inputShape));
    input->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto sliceOutput = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    sliceOutput->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto contractOutput = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    contractOutput->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
    auto outCast = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape),
                                               TileOpFormat::TILEOP_ND, "out");

    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_SLICE, {input}, {sliceOutput},
        [&offset](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset)); });
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_CONTRACT, {sliceOutput}, {contractOutput},
        [&offset](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset)); });
    auto& expOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {contractOutput}, {outCast});

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp pass;
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);

    EXPECT_EQ(CountOpcode(currFunctionPtr, Opcode::OP_VIEW), kNumZero);
    EXPECT_EQ(CountOpcode(currFunctionPtr, Opcode::OP_CONTRACT), kNumZero);
    EXPECT_EQ(CountOpcode(currFunctionPtr, Opcode::OP_SLICE), kNumOne);
    auto sliceOp = FindSingleOp(currFunctionPtr, Opcode::OP_SLICE);
    ASSERT_NE(sliceOp, nullptr);
    EXPECT_EQ(expOp.GetInputOperand(kSizeZero), sliceOp->GetOOperands().front());
}

/*
TestGenerateViewWithNonViewProducer
inCast{8,16}->view->Tensor1{4,16}->assemble->outCast{16,16}
inCast2{8,16}->exp->Tensor2{8,16}->assemble->outCast{16,16}

endTensor(outCast) has a producer chain (inCast2->exp->Tensor2->assemble) whose producer
is NOT a VIEW op. IsNotSameViewInput should detect this and return true, so
GenerateNewView is skipped to avoid breaking the graph.

Expected: no new view generated, all original ops preserved.
*/
TEST_F(TestRemoveRedundantOpPass, TestGenerateViewWithNonViewProducer)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp",
                                                      "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape1 = {kNumExpFour, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumFour, kNumExpFour};
    std::vector<int64_t> offset1 = {kNumZero, kNumZero};
    std::vector<int64_t> offset2 = {kNumFour, kNumZero};
    std::vector<int64_t> offset3 = {kNumEight, kNumZero};
    auto inCast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto inCast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1),
                                                              TileOpFormat::TILEOP_ND, "outCast");
    outCast->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);

    // inCast1->view->ubTensor1->assemble->outCast (view-assemble chain from startTensor=inCast1)
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_VIEW, {inCast1}, {ubTensor1},
        [&offset1](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset1)); });
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_ASSEMBLE, {ubTensor1}, {outCast},
        [&offset1](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset1)); });
    // inCast2->exp->ubTensor2->assemble->outCast (non-view producer chain)
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {inCast2}, {ubTensor2});
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_ASSEMBLE, {ubTensor2}, {outCast},
        [&offset3](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset3)); });

    currFunctionPtr->inCasts_.push_back(inCast1);
    currFunctionPtr->inCasts_.push_back(inCast2);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp RemoveRedundantOpPass;
    EXPECT_EQ(RemoveRedundantOpPass.RunOnFunction(*currFunctionPtr), SUCCESS);

    // GenerateNewView should NOT be triggered because endTensor has a non-view producer (exp).
    // All original view and assemble ops should be preserved.
    uint32_t viewNum = kNumZero;
    uint32_t assembleNum = kNumZero;
    for (const auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++viewNum;
        }
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assembleNum;
        }
    }
    EXPECT_EQ(viewNum, kNumOne);
    EXPECT_EQ(assembleNum, kNumTwo);
}

TEST_F(TestRemoveRedundantOpPass, PerfectMatchWithNonViewProducerShouldNotBypass)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "PerfectMatchWithNonViewProducer",
                                               "PerfectMatchWithNonViewProducer", nullptr);
    ASSERT_NE(function, nullptr);

    std::vector<int64_t> sourceShape = {kNumTwo, kNumTwo, kNumExpFour};
    std::vector<int64_t> flatShape = {kNumFour, kNumExpFour};
    std::vector<int64_t> partShape = {kNumTwo, kNumExpFour};
    std::vector<int64_t> zeroOffset = {kNumZero, kNumZero};
    std::vector<int64_t> partOffset = {kNumTwo, kNumZero};

    auto source = IRBuilder().CreateTensorVar(DT_FP32, sourceShape, CreateTestConstIntVector(sourceShape));
    auto start = IRBuilder().CreateTensorVar(DT_FP32, flatShape, CreateTestConstIntVector(flatShape));
    auto viewOutput = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto otherInput = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto otherValue = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto assembled = IRBuilder().CreateTensorVar(DT_FP32, flatShape, CreateTestConstIntVector(flatShape));
    auto reshaped = IRBuilder().CreateTensorVar(DT_FP32, sourceShape, CreateTestConstIntVector(sourceShape));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, sourceShape, CreateTestConstIntVector(sourceShape),
                                              TileOpFormat::TILEOP_ND, "output");

    source->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    start->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    viewOutput->SetMemoryTypeBoth(MemoryType::MEM_UB);
    otherInput->SetMemoryTypeBoth(MemoryType::MEM_UB);
    otherValue->SetMemoryTypeBoth(MemoryType::MEM_UB);
    assembled->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    reshaped->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    output->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);

    PassOperationUtils::AddOperation(*function, Opcode::OP_RESHAPE, {source}, {start});
    PassOperationUtils::AddOperation(*function, Opcode::OP_VIEW, {start}, {viewOutput}, [&zeroOffset](Operation& op) {
        op.SetOpAttribute(std::make_shared<ViewOpAttribute>(zeroOffset));
    });
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_ASSEMBLE, {viewOutput}, {assembled},
        [&zeroOffset](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(zeroOffset)); });
    PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {otherInput}, {otherValue});
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_ASSEMBLE, {otherValue}, {assembled},
        [&partOffset](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(partOffset)); });
    PassOperationUtils::AddOperation(*function, Opcode::OP_RESHAPE, {assembled}, {reshaped});
    auto& finalExp = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {reshaped}, {output});

    function->inCasts_.push_back(source);
    function->inCasts_.push_back(otherInput);
    function->outCasts_.push_back(output);

    RemoveRedundantOp pass;
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);

    EXPECT_EQ(CountOpcode(function, Opcode::OP_SLICE), kNumOne);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_CONTRACT), kNumOne);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_RESHAPE), kNumTwo);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_EXP), kNumTwo);
    EXPECT_EQ(finalExp.GetIOperands().front(), reshaped);
}

/*
TestGenerateViewEndTensorLargerThanStartTensor
inCast{8,16}->view->Tensor1{4,16}->assemble->outCast{128,16}

endTensor(outCast) shape (128,16) has dim0 > startTensor shape dim0 (8), so endTensor is NOT
a part of startTensor. The shape check in IsValidViewAssemble should reject this case.

Expected: no new view generated, all original ops preserved.
*/
TEST_F(TestRemoveRedundantOpPass, TestGenerateViewEndTensorLargerThanStartTensor)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp",
                                                      "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape1 = {kNumExpSeven, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumFour, kNumExpFour};
    std::vector<int64_t> offset1 = {kNumZero, kNumZero};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1),
                                                              TileOpFormat::TILEOP_ND, "outCast");
    outCast->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);

    // inCast{8,16}->view->ubTensor1{4,16}->assemble->outCast{128,16}
    // endTensor=outCast{128,16}, startTensor=inCast{8,16}, dim0: 128 > 8 => not a part of startTensor
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_VIEW, {inCast}, {ubTensor1},
        [&offset1](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset1)); });
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_ASSEMBLE, {ubTensor1}, {outCast},
        [&offset1](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset1)); });

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp RemoveRedundantOpPass;
    EXPECT_EQ(RemoveRedundantOpPass.RunOnFunction(*currFunctionPtr), SUCCESS);

    // GenerateNewView should NOT be triggered because endTensor shape (128,16) exceeds
    // startTensor shape (8,16) in dim0.
    uint32_t viewNum = kNumZero;
    uint32_t assembleNum = kNumZero;
    for (const auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++viewNum;
        }
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assembleNum;
        }
    }
    EXPECT_EQ(viewNum, kNumOne);
    EXPECT_EQ(assembleNum, kNumOne);
}

/*
TestGenerateViewOutcastWithNonViewProducer
inCast1{8,16}->view->Tensor1{4,16}->assemble->outCast{16,16}
inCast2{8,16}->exp->Tensor2{8,16}->assemble->outCast{16,16}

endTensor(outCast) has no consumers (is outcast) and has a non-view-assemble producer
(inCast2->exp->Tensor2->assemble). GenerateNewView should create a new tensor for the
view-assemble chain instead of reusing endTensor directly, to avoid breaking the graph.

Expected: a new view op is created, non-view-assemble producer chain preserved on endTensor.
*/
TEST_F(TestRemoveRedundantOpPass, TestGenerateViewOutcastWithNonViewProducer)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp",
                                                      "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph: endTensor shape must be <= startTensor shape in every dim
    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    std::vector<int64_t> shape1 = {kNumExpFive, kNumExpSix};
    std::vector<int64_t> shape2 = {kNumExpFive, kNumExpSix};
    std::vector<int64_t> offset1 = {kNumZero, kNumZero};
    std::vector<int64_t> offset3 = {kNumExpFive, kNumZero};
    auto inCast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto inCast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1),
                                                              TileOpFormat::TILEOP_ND, "outCast");
    outCast->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);

    // inCast1{64,64}->view->ubTensor1{32,64}->assemble->outCast{32,64} (view-assemble chain from inCast1)
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_VIEW, {inCast1}, {ubTensor1},
        [&offset1](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset1)); });
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_ASSEMBLE, {ubTensor1}, {outCast},
        [&offset1](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset1)); });
    // inCast2{64,64}->exp->ubTensor2{32,64}->assemble->outCast{32,64} (non-view producer chain)
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {inCast2}, {ubTensor2});
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_ASSEMBLE, {ubTensor2}, {outCast},
        [&offset3](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset3)); });

    currFunctionPtr->inCasts_.push_back(inCast1);
    currFunctionPtr->inCasts_.push_back(inCast2);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp RemoveRedundantOpPass;
    EXPECT_EQ(RemoveRedundantOpPass.RunOnFunction(*currFunctionPtr), SUCCESS);

    // endTensor 是 outcast（无消费者），GenerateNewView 直接跳过，所有原始操作保留
    uint32_t viewNum = kNumZero;
    uint32_t assembleNum = kNumZero;
    uint32_t expNum = kNumZero;
    for (const auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++viewNum;
        }
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assembleNum;
        }
        if (op.GetOpcode() == Opcode::OP_EXP) {
            ++expNum;
        }
    }
    EXPECT_EQ(viewNum, kNumOne);
    EXPECT_EQ(assembleNum, kNumTwo);
    EXPECT_EQ(expNum, kNumOne);
}

// outcast-ASSEMBLE guard 回归测试：RESHAPE 输出 reshapeOut{256,64}(DDR) 有 3 个 consumer
// （2个VIEW+1个ASSEMBLE），该 ASSEMBLE 输出 outcast outCastZ{256,64}(DDR)，输入输出同形同 DDR，
// 看似冗余但必须被 RemoveDummyOp 的 outcast guard 保留。
TEST_F(TestRemoveRedundantOpPass, TestReshapeLoopOutcastAssembleKept)
{
    ComputationalGraphBuilder G;
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    G.AddTensor(DataType::DT_FP32, {128, 128}, MemoryType::MEM_DEVICE_DDR, "inCastA");
    G.AddTensor(DataType::DT_FP32, {128, 128}, MemoryType::MEM_DEVICE_DDR, "inCastB");
    G.AddTensor(DataType::DT_FP32, {128, 128}, "aL1");
    G.AddTensor(DataType::DT_FP32, {128, 128}, "bL1");
    G.AddTensor(DataType::DT_FP32, {128, 128}, "mulOut");
    G.AddTensor(DataType::DT_FP32, {128, 128}, MemoryType::MEM_DEVICE_DDR, "zDdr");
    G.AddTensor(DataType::DT_FP32, {256, 64}, MemoryType::MEM_DEVICE_DDR, "reshapeOut");
    G.AddTensor(DataType::DT_FP32, {128, 64}, "view0Out");
    G.AddTensor(DataType::DT_FP32, {128, 64}, "view1Out");
    G.AddTensor(DataType::DT_FP32, {128, 64}, "reg0");
    G.AddTensor(DataType::DT_FP32, {128, 64}, "reg1");
    G.AddTensor(DataType::DT_FP32, {256, 64}, MemoryType::MEM_DEVICE_DDR, "outCastTmp");
    G.AddTensor(DataType::DT_FP32, {256, 64}, MemoryType::MEM_DEVICE_DDR, "outCastZ");

    G.AddOp(Opcode::OP_VIEW, {"inCastA"}, {"aL1"}, "view_a");
    G.GetOp("view_a")->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    G.AddOp(Opcode::OP_VIEW, {"inCastB"}, {"bL1"}, "view_b");
    G.GetOp("view_b")->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    G.AddOp(Opcode::OP_A_MUL_B, {"aL1", "bL1"}, {"mulOut"}, "mul");
    G.AddOp(Opcode::OP_ASSEMBLE, {"mulOut"}, {"zDdr"}, "asm_zddr");
    G.GetOp("asm_zddr")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    G.AddOp(Opcode::OP_RESHAPE, {"zDdr"}, {"reshapeOut"}, "reshape");
    G.AddOp(Opcode::OP_VIEW, {"reshapeOut"}, {"view0Out"}, "view0");
    G.GetOp("view0")->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    G.AddOp(Opcode::OP_VIEW, {"reshapeOut"}, {"view1Out"}, "view1");
    G.GetOp("view1")->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 128}));
    G.AddOp(Opcode::OP_REGISTER_COPY, {"view0Out"}, {"reg0"}, "regcopy0");
    G.AddOp(Opcode::OP_REGISTER_COPY, {"view1Out"}, {"reg1"}, "regcopy1");
    G.AddOp(Opcode::OP_ASSEMBLE, {"reg0"}, {"outCastTmp"}, "asm_tmp0");
    G.GetOp("asm_tmp0")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    G.AddOp(Opcode::OP_ASSEMBLE, {"reg1"}, {"outCastTmp"}, "asm_tmp1");
    G.GetOp("asm_tmp1")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 128}));
    G.AddOp(Opcode::OP_ASSEMBLE, {"reshapeOut"}, {"outCastZ"}, "asm_outcastz");
    G.GetOp("asm_outcastz")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));

    G.SetInCast({"inCastA", "inCastB"});
    G.SetOutCast({"outCastTmp", "outCastZ"});

    RemoveRedundantOp removeRedundantOpPass;
    EXPECT_EQ(removeRedundantOpPass.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(removeRedundantOpPass.PostCheck(*function), SUCCESS);

    // 核心断言：outcast-ASSEMBLE guard 必须保留 reshapeOut→outCastZ 的 ASSEMBLE
    bool outcastAssembleKept = false;
    auto outCastZ = G.GetTensor("outCastZ");
    for (const auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE && op.oOperand.front() == outCastZ) {
            outcastAssembleKept = true;
            break;
        }
    }
    EXPECT_TRUE(outcastAssembleKept);

    uint32_t regCopyNum = 0;
    uint32_t viewNum = 0;
    uint32_t assembleNum = 0;
    for (const auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_REGISTER_COPY) {
            regCopyNum++;
        }
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            viewNum++;
        }
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            assembleNum++;
        }
    }
    EXPECT_EQ(regCopyNum, 0);
    EXPECT_EQ(viewNum, 4);
    EXPECT_EQ(assembleNum, 4);
}

// RemoveRedundantOp 删除 REGISTER_COPY+VIEW 后 ASSEMBLE 输入穿透到 CAST(inCast)，
// Process(function, false) 跳过 NoViewConflict，仅靠 ASSEMBLE 输出覆盖判断 → 不插 copy。
TEST_F(TestRemoveRedundantOpPass, TestSkipViewConflictAfterRemoveRedundantOp)
{
    ComputationalGraphBuilder G;
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    G.AddTensor(DataType::DT_INT8, {82816, 672}, MemoryType::MEM_DEVICE_DDR, "kvActSeqs");
    G.AddTensor(DataType::DT_INT8, {1, 16}, MemoryType::MEM_DEVICE_DDR, "bIdx");
    G.AddTensor(DataType::DT_INT8, {1, 512}, MemoryType::MEM_DEVICE_DDR, "s1Idx");
    G.AddTensor(DataType::DT_INT8, {16, 672}, "gatherOut");
    G.AddTensor(DataType::DT_INT8, {16, 512}, "view1Out");
    G.AddTensor(DataType::DT_FP16, {16, 512}, "cast1Out");
    G.AddTensor(DataType::DT_FP16, {16, 512}, "view2Out");
    G.AddTensor(DataType::DT_FP32, {16, 512}, "cast2Out");
    G.AddTensor(DataType::DT_FP32, {16, 512}, "viewAIn");
    G.AddTensor(DataType::DT_FP32, {16, 512}, "viewBIn");
    G.AddTensor(DataType::DT_FP32, {16, 512}, "regCopyAOut");
    G.AddTensor(DataType::DT_FP32, {16, 512}, "regCopyBOut");
    G.AddTensor(DataType::DT_FP32, {16, 1024}, "asmOut");
    G.AddTensor(DataType::DT_FP32, {128, 128}, "reshapeOut");
    G.AddTensor(DataType::DT_FP32, {128, 128}, "viewMulIn");
    G.AddTensor(DataType::DT_FP32, {128, 1}, "scaleIn");
    G.AddTensor(DataType::DT_FP32, {128, 1}, "viewScaleOut");
    G.AddTensor(DataType::DT_FP32, {128, 128}, "mulOut");
    G.AddTensor(DataType::DT_FP32, {128, 128}, MemoryType::MEM_DEVICE_DDR, "ddrOut");

    G.AddOp(Opcode::OP_GATHER_IN_UB, {"kvActSeqs", "bIdx", "s1Idx"}, {"gatherOut"}, "gather");
    G.AddOp(Opcode::OP_VIEW, {"gatherOut"}, {"view1Out"}, "view_167");
    G.GetOp("view_167")->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 160}));
    G.AddOp(Opcode::OP_CAST, {"view1Out"}, {"cast1Out"}, "cast_167");
    G.AddOp(Opcode::OP_VIEW, {"cast1Out"}, {"view2Out"}, "view_170");
    G.GetOp("view_170")->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    G.AddOp(Opcode::OP_CAST, {"view2Out"}, {"cast2Out"}, "cast_170");
    G.AddOp(Opcode::OP_VIEW, {"cast2Out"}, {"viewAIn"}, "view_a");
    G.GetOp("view_a")->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    G.AddOp(Opcode::OP_VIEW, {"cast2Out"}, {"viewBIn"}, "view_b");
    G.GetOp("view_b")->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    G.AddOp(Opcode::OP_REGISTER_COPY, {"viewAIn"}, {"regCopyAOut"}, "regcopy_a");
    G.AddOp(Opcode::OP_REGISTER_COPY, {"viewBIn"}, {"regCopyBOut"}, "regcopy_b");
    G.AddOp(Opcode::OP_ASSEMBLE, {"regCopyAOut"}, {"asmOut"}, "asm_0");
    G.GetOp("asm_0")->SetOpAttribute(
        std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, std::vector<int64_t>{0, 512}));
    G.AddOp(Opcode::OP_ASSEMBLE, {"regCopyBOut"}, {"asmOut"}, "asm_1");
    G.GetOp("asm_1")->SetOpAttribute(
        std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, std::vector<int64_t>{0, 0}));
    G.AddOp(Opcode::OP_RESHAPE, {"asmOut"}, {"reshapeOut"}, "reshape");
    G.AddOp(Opcode::OP_VIEW, {"reshapeOut"}, {"viewMulIn"}, "view_mul");
    G.GetOp("view_mul")->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    G.AddOp(Opcode::OP_VIEW, {"scaleIn"}, {"viewScaleOut"}, "view_scale");
    G.GetOp("view_scale")->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    G.AddOp(Opcode::OP_MUL, {"viewMulIn", "viewScaleOut"}, {"mulOut"}, "mul");
    G.AddOp(Opcode::OP_COPY_OUT, {"mulOut"}, {"ddrOut"}, "copy_out");

    G.SetInCast({"kvActSeqs", "bIdx", "s1Idx", "scaleIn"});
    G.SetOutCast({"ddrOut"});

    RemoveRedundantOp removeRedundantOpPass;
    EXPECT_EQ(removeRedundantOpPass.RunOnFunction(*function), SUCCESS);

    uint32_t regCopyNum = 0;
    uint32_t viewNum = 0;
    uint32_t assembleNum = 0;
    uint32_t castNum = 0;
    for (const auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_REGISTER_COPY) {
            regCopyNum++;
        }
        if (op.GetOpcode() == Opcode::OP_VIEW || op.GetOpcode() == Opcode::OP_SLICE) {
            viewNum++;
        }
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE || op.GetOpcode() == Opcode::OP_CONTRACT) {
            assembleNum++;
        }
        if (op.GetOpcode() == Opcode::OP_CAST) {
            castNum++;
        }
    }
    EXPECT_EQ(regCopyNum, 0);
    EXPECT_EQ(castNum, 2);
    EXPECT_EQ(viewNum, 3);
    EXPECT_EQ(assembleNum, 4);
}

TEST_F(TestRemoveRedundantOpPass, NonPerfectContiguousViewAssembleShouldGenerateSingleView)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "NonPerfectContiguousViewAssemble",
                                               "NonPerfectContiguousViewAssemble", nullptr);
    ASSERT_NE(function, nullptr);

    std::vector<int64_t> inputShape = {kNumExpEight, kNumExpSeven};
    std::vector<int64_t> partShape = {kNumExpSix, kNumExpSeven};
    std::vector<int64_t> outputShape = {kNumExpSeven, kNumExpSeven};
    std::vector<int64_t> sourceOffset0 = {kNumExpSeven, kNumZero};
    std::vector<int64_t> sourceOffset1 = {192, kNumZero};
    std::vector<int64_t> targetOffset0 = {kNumZero, kNumZero};
    std::vector<int64_t> targetOffset1 = {kNumExpSix, kNumZero};

    auto input = IRBuilder().CreateTensorVar(DT_FP32, inputShape, CreateTestConstIntVector(inputShape));
    auto part0 = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto part1 = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto assembled = IRBuilder().CreateTensorVar(DT_FP32, outputShape, CreateTestConstIntVector(outputShape));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, outputShape, CreateTestConstIntVector(outputShape),
                                              TileOpFormat::TILEOP_ND, "output");
    input->SetMemoryTypeBoth(MemoryType::MEM_UB);
    part0->SetMemoryTypeBoth(MemoryType::MEM_UB);
    part1->SetMemoryTypeBoth(MemoryType::MEM_UB);
    assembled->SetMemoryTypeBoth(MemoryType::MEM_UB);
    output->SetMemoryTypeBoth(MemoryType::MEM_UB);

    PassOperationUtils::AddOperation(*function, Opcode::OP_VIEW, {input}, {part0}, [&sourceOffset0](Operation& op) {
        op.SetOpAttribute(std::make_shared<ViewOpAttribute>(sourceOffset0));
    });
    PassOperationUtils::AddOperation(*function, Opcode::OP_VIEW, {input}, {part1}, [&sourceOffset1](Operation& op) {
        op.SetOpAttribute(std::make_shared<ViewOpAttribute>(sourceOffset1));
    });
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_ASSEMBLE, {part0}, {assembled},
        [&targetOffset0](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(targetOffset0)); });
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_ASSEMBLE, {part1}, {assembled},
        [&targetOffset1](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(targetOffset1)); });
    auto& exp = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {assembled}, {output});
    function->inCasts_.push_back(input);
    function->outCasts_.push_back(output);

    RemoveRedundantOp pass;
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);

    EXPECT_EQ(CountOpcode(function, Opcode::OP_VIEW), kNumOne);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_ASSEMBLE), kNumZero);
    auto* view = FindSingleOp(function, Opcode::OP_VIEW);
    ASSERT_NE(view, nullptr);
    auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(view->GetOpAttribute());
    ASSERT_NE(viewAttr, nullptr);
    EXPECT_EQ(viewAttr->GetFromOffset(), sourceOffset0);
    EXPECT_EQ(view->GetIOperands().front(), input);
    EXPECT_EQ(exp.GetIOperands().front(), view->GetOOperands().front());
}

TEST_F(TestRemoveRedundantOpPass, SliceContractWithPartialValidShapeShouldGenerateSingleView)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "SliceContractPartialValidShape",
                                               "SliceContractPartialValidShape", nullptr);
    ASSERT_NE(function, nullptr);

    std::vector<int64_t> inputShape = {kNumExpEight, kNumExpSeven};
    std::vector<int64_t> partShape = {kNumExpSix, kNumExpFive};
    std::vector<int64_t> outputShape = {kNumExpEight, kNumExpFive};
    std::vector<SymbolicScalar> partValidShape = CreateTestConstIntVector(partShape);
    std::vector<std::vector<int64_t>> targetOffsets = {
        {kNumZero, kNumZero}, {kNumExpSix, kNumZero}, {kNumExpSeven, kNumZero}, {192, kNumZero}};

    auto input = IRBuilder().CreateTensorVar(DT_FP32, inputShape, CreateTestConstIntVector(inputShape));
    auto assembled = IRBuilder().CreateTensorVar(DT_FP32, outputShape, partValidShape);
    auto output = IRBuilder().CreateTensorVar(DT_FP32, outputShape, partValidShape, TileOpFormat::TILEOP_ND, "output");
    input->SetMemoryTypeBoth(MemoryType::MEM_UB);
    assembled->SetMemoryTypeBoth(MemoryType::MEM_UB);
    output->SetMemoryTypeBoth(MemoryType::MEM_UB);

    for (const auto& targetOffset : targetOffsets) {
        std::vector<int64_t> sourceOffset = {targetOffset[0], kNumExpFive};
        auto part = IRBuilder().CreateTensorVar(DT_FP32, partShape, partValidShape);
        part->SetMemoryTypeBoth(MemoryType::MEM_UB);
        PassOperationUtils::AddOperation(*function, Opcode::OP_SLICE, {input}, {part}, [&sourceOffset](Operation& op) {
            op.SetOpAttribute(std::make_shared<ViewOpAttribute>(sourceOffset));
        });
        PassOperationUtils::AddOperation(
            *function, Opcode::OP_CONTRACT, {part}, {assembled},
            [&targetOffset](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(targetOffset)); });
    }
    auto& exp = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {assembled}, {output});
    function->inCasts_.push_back(input);
    function->outCasts_.push_back(output);

    RemoveRedundantOp pass;
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);

    EXPECT_EQ(CountOpcode(function, Opcode::OP_SLICE), kNumZero);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_CONTRACT), kNumZero);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_VIEW), kNumOne);
    auto* view = FindSingleOp(function, Opcode::OP_VIEW);
    ASSERT_NE(view, nullptr);
    auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(view->GetOpAttribute());
    ASSERT_NE(viewAttr, nullptr);
    EXPECT_EQ(viewAttr->GetFromOffset(), (std::vector<int64_t>{kNumZero, kNumExpFive}));
    EXPECT_EQ(view->GetIOperands().front(), input);
    EXPECT_EQ(exp.GetIOperands().front(), view->GetOOperands().front());
}

TEST_F(TestRemoveRedundantOpPass, OrderedViewAssembleWithDifferentTranslationsShouldNotFold)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "OrderedViewAssemble", "OrderedViewAssemble",
                                               nullptr);
    ASSERT_NE(function, nullptr);

    std::vector<int64_t> inputShape = {kNumExpEight, kNumExpSeven};
    std::vector<int64_t> partShape = {kNumExpSix, kNumExpSeven};
    std::vector<int64_t> outputShape = {kNumExpSeven, kNumExpSeven};
    std::vector<int64_t> sourceOffset0 = {120, kNumZero};
    std::vector<int64_t> sourceOffset1 = {192, kNumZero};
    std::vector<int64_t> targetOffset0 = {kNumZero, kNumZero};
    std::vector<int64_t> targetOffset1 = {kNumExpSix, kNumZero};

    auto input = IRBuilder().CreateTensorVar(DT_FP32, inputShape, CreateTestConstIntVector(inputShape));
    auto part0 = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto part1 = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto assembled = IRBuilder().CreateTensorVar(DT_FP32, outputShape, CreateTestConstIntVector(outputShape));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, outputShape, CreateTestConstIntVector(outputShape),
                                              TileOpFormat::TILEOP_ND, "output");
    input->SetMemoryTypeBoth(MemoryType::MEM_UB);
    part0->SetMemoryTypeBoth(MemoryType::MEM_UB);
    part1->SetMemoryTypeBoth(MemoryType::MEM_UB);
    assembled->SetMemoryTypeBoth(MemoryType::MEM_UB);
    output->SetMemoryTypeBoth(MemoryType::MEM_UB);

    PassOperationUtils::AddOperation(*function, Opcode::OP_VIEW, {input}, {part0}, [&sourceOffset0](Operation& op) {
        op.SetOpAttribute(std::make_shared<ViewOpAttribute>(sourceOffset0));
    });
    PassOperationUtils::AddOperation(*function, Opcode::OP_VIEW, {input}, {part1}, [&sourceOffset1](Operation& op) {
        op.SetOpAttribute(std::make_shared<ViewOpAttribute>(sourceOffset1));
    });
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_ASSEMBLE, {part0}, {assembled},
        [&targetOffset0](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(targetOffset0)); });
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_ASSEMBLE, {part1}, {assembled},
        [&targetOffset1](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(targetOffset1)); });
    auto& exp = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {assembled}, {output});
    function->inCasts_.push_back(input);
    function->outCasts_.push_back(output);

    RemoveRedundantOp pass;
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);

    EXPECT_EQ(CountOpcode(function, Opcode::OP_VIEW), kNumTwo);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_ASSEMBLE), kNumTwo);
    EXPECT_EQ(exp.GetIOperands().front(), assembled);
}

TEST_F(TestRemoveRedundantOpPass, ParallelViewAssembleToOutcastShouldRelinkAllProducers)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "ParallelViewAssembleToOutcast",
                                               "ParallelViewAssembleToOutcast", nullptr);
    ASSERT_NE(function, nullptr);

    std::vector<int64_t> shape = {kNumFour, kNumExpFour};
    std::vector<int64_t> partShape = {kNumOne, kNumExpFour};
    auto input0 = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto input1 = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto sharedOutput = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outcast = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape), TileOpFormat::TILEOP_ND,
                                               "outcast");
    input0->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    input1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    sharedOutput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    outcast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);

    auto& producer0 = PassOperationUtils::AddOperation(*function, Opcode::OP_ADD, {input0, input1}, {sharedOutput});
    auto& producer1 = PassOperationUtils::AddOperation(*function, Opcode::OP_ADD, {input1, input0}, {sharedOutput});
    for (int64_t idx = 0; idx < kNumFour; ++idx) {
        std::vector<int64_t> offset = {idx, kNumZero};
        auto part = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
        part->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
        PassOperationUtils::AddOperation(*function, Opcode::OP_VIEW, {sharedOutput}, {part}, [&offset](Operation& op) {
            op.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset));
        });
        PassOperationUtils::AddOperation(*function, Opcode::OP_ASSEMBLE, {part}, {outcast}, [&offset](Operation& op) {
            op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset));
        });
    }
    function->inCasts_.push_back(input0);
    function->inCasts_.push_back(input1);
    function->outCasts_.push_back(outcast);

    RemoveRedundantOp pass;
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);

    EXPECT_EQ(CountOpcode(function, Opcode::OP_ADD), kNumTwo);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_VIEW), kNumZero);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_ASSEMBLE), kNumZero);
    EXPECT_EQ(outcast->GetProducers().size(), kNumTwo);
    EXPECT_TRUE(outcast->HasProducer(&producer0));
    EXPECT_TRUE(outcast->HasProducer(&producer1));
    EXPECT_EQ(producer0.GetOOperands().front(), outcast);
    EXPECT_EQ(producer1.GetOOperands().front(), outcast);
    for (auto& op : function->Operations()) {
        if (!op.GetIOperands().empty() && !op.GetOOperands().empty()) {
            EXPECT_NE(op.GetIOperands().front(), op.GetOOperands().front());
        }
    }
}

TEST_F(TestRemoveRedundantOpPass, ParallelViewAssembleToIntermediateShouldKeepStartProducer)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "ParallelViewAssembleToIntermediate",
                                               "ParallelViewAssembleToIntermediate", nullptr);
    ASSERT_NE(function, nullptr);

    std::vector<int64_t> shape = {kNumFour, kNumExpFour};
    std::vector<int64_t> partShape = {kNumTwo, kNumExpFour};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    std::vector<int64_t> offset1 = {kNumTwo, kNumZero};
    auto input0 = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto input1 = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto start = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto part0 = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto part1 = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto assembled = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    input0->SetMemoryTypeBoth(MemoryType::MEM_UB);
    input1->SetMemoryTypeBoth(MemoryType::MEM_UB);
    start->SetMemoryTypeBoth(MemoryType::MEM_UB);
    part0->SetMemoryTypeBoth(MemoryType::MEM_UB);
    part1->SetMemoryTypeBoth(MemoryType::MEM_UB);
    assembled->SetMemoryTypeBoth(MemoryType::MEM_UB);
    output->SetMemoryTypeBoth(MemoryType::MEM_UB);

    auto& startProducer = PassOperationUtils::AddOperation(*function, Opcode::OP_ADD, {input0, input1}, {start});
    PassOperationUtils::AddOperation(*function, Opcode::OP_VIEW, {start}, {part0}, [&offset0](Operation& op) {
        op.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset0));
    });
    PassOperationUtils::AddOperation(*function, Opcode::OP_VIEW, {start}, {part1}, [&offset1](Operation& op) {
        op.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset1));
    });
    PassOperationUtils::AddOperation(*function, Opcode::OP_ASSEMBLE, {part0}, {assembled}, [&offset0](Operation& op) {
        op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset0));
    });
    PassOperationUtils::AddOperation(*function, Opcode::OP_ASSEMBLE, {part1}, {assembled}, [&offset1](Operation& op) {
        op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset1));
    });
    auto& consumer = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {assembled}, {output});
    function->inCasts_.push_back(input0);
    function->inCasts_.push_back(input1);
    function->outCasts_.push_back(output);

    RemoveRedundantOp pass;
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);

    EXPECT_EQ(CountOpcode(function, Opcode::OP_VIEW), kNumZero);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_ASSEMBLE), kNumZero);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_ADD), kNumOne);
    EXPECT_EQ(consumer.GetIOperands().front(), start);
    EXPECT_EQ(startProducer.GetOOperands().front(), start);
    EXPECT_TRUE(start->HasProducer(&startProducer));
}

TEST_F(TestRemoveRedundantOpPass, SingleAssembleMovesTokenDependency)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "SingleAssembleMovesTokenDependency",
                                               "SingleAssembleMovesTokenDependency", nullptr);
    ASSERT_NE(function, nullptr);

    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    auto input = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto assembleInput = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto assembleOutput = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tokenInput = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tokenOutput = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tokenSinkOutput = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    assembleInput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    assembleOutput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    assembleInput->UpdateDynValidShape({CreateTestScalarVar("in_valid_0"), CreateTestScalarVar("in_valid_1")});
    assembleOutput->UpdateDynValidShape({CreateTestScalarVar("out_valid_0"), CreateTestScalarVar("out_valid_1")});
    assembleOutput->offset = {kNumOne, kNumZero};

    auto& dataProducer = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {input}, {assembleInput});
    auto& assemble = PassOperationUtils::AddOperation(
        *function, Opcode::OP_ASSEMBLE, {assembleInput}, {assembleOutput},
        [](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{1, 0})); });
    auto& finalConsumer = PassOperationUtils::AddOperation(*function, Opcode::OP_SQRT, {assembleOutput}, {output});
    auto& tokenProducer = PassOperationUtils::AddOperation(*function, Opcode::OP_ABS, {tokenInput}, {tokenOutput});
    auto& tokenSink = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {tokenInput}, {tokenSinkOutput});
    auto inputToken = AddTokenEdge(*function, tokenProducer, assemble);
    auto oldResultToken = AddTokenEdge(*function, assemble, tokenSink);

    function->inCasts_ = {input, tokenInput};
    function->outCasts_ = {output, tokenOutput, tokenSinkOutput};

    auto assembleStmt = ToStmtPtr(assemble);
    RemoveRedundantOp pass;
    EXPECT_EQ(pass.PreCheck(*function), SUCCESS);
    EXPECT_EQ(pass.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(pass.PostCheck(*function), SUCCESS);

    EXPECT_EQ(CountOpcode(function, Opcode::OP_ASSEMBLE), kNumZero);
    EXPECT_EQ(finalConsumer.GetIOperands().front(), assembleInput);
    EXPECT_TRUE(HasToken(finalConsumer, inputToken));
    EXPECT_TRUE(function->GetVarDependency().HasConsumer(inputToken, ToStmtPtr(finalConsumer)));
    EXPECT_FALSE(function->GetVarDependency().HasConsumer(inputToken, assembleStmt));
    ASSERT_FALSE(dataProducer.result_token_.empty());
    EXPECT_NE(dataProducer.result_token_.front(), oldResultToken);
    EXPECT_TRUE(function->GetVarDependency().HasProducer(dataProducer.result_token_.front(), ToStmtPtr(dataProducer)));
    EXPECT_TRUE(function->GetVarDependency().HasConsumer(dataProducer.result_token_.front(), ToStmtPtr(tokenSink)));
    EXPECT_TRUE(HasToken(tokenSink, dataProducer.result_token_.front()));
    EXPECT_FALSE(function->GetVarDependency().HasDependency(oldResultToken));
}

TEST_F(TestRemoveRedundantOpPass, PerfectViewAssembleMovesTokenDependency)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "PerfectViewAssembleMovesTokenDependency",
                                               "PerfectViewAssembleMovesTokenDependency", nullptr);
    ASSERT_NE(function, nullptr);

    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    std::vector<int64_t> offset = {kNumZero, kNumZero};
    auto input = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto start = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto viewOutput = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto assembleOutput = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tokenInput = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tokenOutput = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tokenSinkOutput = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    start->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    assembleOutput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);

    auto& dataProducer = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {input}, {start});
    auto& view = PassOperationUtils::AddOperation(
        *function, Opcode::OP_VIEW, {start}, {viewOutput},
        [&offset](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset)); });
    auto& assemble = PassOperationUtils::AddOperation(
        *function, Opcode::OP_ASSEMBLE, {viewOutput}, {assembleOutput},
        [&offset](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset)); });
    auto& finalConsumer = PassOperationUtils::AddOperation(*function, Opcode::OP_SQRT, {assembleOutput}, {output});
    auto& tokenProducer = PassOperationUtils::AddOperation(*function, Opcode::OP_ABS, {tokenInput}, {tokenOutput});
    auto& tokenSink = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {tokenInput}, {tokenSinkOutput});
    auto inputToken = AddTokenEdge(*function, tokenProducer, view);
    AddTokenEdge(*function, tokenProducer, assemble);
    auto oldResultToken = AddTokenEdge(*function, assemble, tokenSink);

    function->inCasts_ = {input, tokenInput};
    function->outCasts_ = {output, tokenOutput, tokenSinkOutput};

    auto viewStmt = ToStmtPtr(view);
    auto assembleStmt = ToStmtPtr(assemble);
    RemoveRedundantOp pass;
    EXPECT_EQ(pass.PreCheck(*function), SUCCESS);
    EXPECT_EQ(pass.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(pass.PostCheck(*function), SUCCESS);

    EXPECT_EQ(CountOpcode(function, Opcode::OP_VIEW), kNumZero);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_ASSEMBLE), kNumZero);
    EXPECT_EQ(finalConsumer.GetIOperands().front(), start);
    EXPECT_TRUE(HasToken(finalConsumer, inputToken));
    EXPECT_TRUE(function->GetVarDependency().HasConsumer(inputToken, ToStmtPtr(finalConsumer)));
    EXPECT_FALSE(function->GetVarDependency().HasConsumer(inputToken, viewStmt));
    EXPECT_FALSE(function->GetVarDependency().HasConsumer(inputToken, assembleStmt));
    ASSERT_FALSE(dataProducer.result_token_.empty());
    EXPECT_NE(dataProducer.result_token_.front(), oldResultToken);
    EXPECT_TRUE(function->GetVarDependency().HasProducer(dataProducer.result_token_.front(), ToStmtPtr(dataProducer)));
    EXPECT_TRUE(function->GetVarDependency().HasConsumer(dataProducer.result_token_.front(), ToStmtPtr(tokenSink)));
    EXPECT_TRUE(HasToken(tokenSink, dataProducer.result_token_.front()));
    EXPECT_FALSE(function->GetVarDependency().HasDependency(oldResultToken));
}

TEST_F(TestRemoveRedundantOpPass, PartialViewAssembleKeepsTokenOnNewView)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "PartialViewAssembleKeepsTokenOnNewView",
                                               "PartialViewAssembleKeepsTokenOnNewView", nullptr);
    ASSERT_NE(function, nullptr);

    std::vector<int64_t> fullShape = {kNumEight, kNumExpFour};
    std::vector<int64_t> partShape = {kNumFour, kNumExpFour};
    std::vector<int64_t> offset = {kNumZero, kNumZero};
    std::vector<SymbolicScalar> dynOffset = {kNumZero, kNumZero};
    auto input = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto start = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto viewOutput = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto assembleOutput = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto tokenInput = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto tokenOutput = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto tokenSinkOutput = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    start->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    assembleOutput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);

    PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {input}, {start});
    auto& view = PassOperationUtils::AddOperation(
        *function, Opcode::OP_VIEW, {start}, {viewOutput}, [&offset, &dynOffset](Operation& op) {
            op.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset, dynOffset));
        });
    auto& assemble = PassOperationUtils::AddOperation(
        *function, Opcode::OP_ASSEMBLE, {viewOutput}, {assembleOutput},
        [&offset](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset)); });
    auto& finalConsumer = PassOperationUtils::AddOperation(*function, Opcode::OP_SQRT, {assembleOutput}, {output});
    auto& tokenProducer = PassOperationUtils::AddOperation(*function, Opcode::OP_ABS, {tokenInput}, {tokenOutput});
    auto& tokenSink = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {tokenInput}, {tokenSinkOutput});
    auto inputToken = AddTokenEdge(*function, tokenProducer, view);
    auto oldResultToken = AddTokenEdge(*function, assemble, tokenSink);

    function->inCasts_ = {input, tokenInput};
    function->outCasts_ = {output, tokenOutput, tokenSinkOutput};

    auto viewStmt = ToStmtPtr(view);
    RemoveRedundantOp pass;
    EXPECT_EQ(pass.PreCheck(*function), SUCCESS);
    EXPECT_EQ(pass.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(pass.PostCheck(*function), SUCCESS);

    Operation* newView = nullptr;
    for (auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            newView = &op;
            break;
        }
    }
    ASSERT_NE(newView, nullptr);
    EXPECT_FALSE(function->Operations().Contains(view));
    EXPECT_FALSE(function->Operations().Contains(assemble));
    EXPECT_EQ(CountOpcode(function, Opcode::OP_VIEW), kNumOne);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_ASSEMBLE), kNumZero);
    EXPECT_EQ(newView->GetIOperands().front(), start);
    EXPECT_EQ(finalConsumer.GetIOperands().front(), newView->GetOOperands().front());
    EXPECT_TRUE(HasToken(*newView, inputToken));
    EXPECT_TRUE(function->GetVarDependency().HasConsumer(inputToken, ToStmtPtr(*newView)));
    EXPECT_FALSE(function->GetVarDependency().HasConsumer(inputToken, viewStmt));
    ASSERT_FALSE(newView->result_token_.empty());
    EXPECT_NE(newView->result_token_.front(), oldResultToken);
    EXPECT_TRUE(function->GetVarDependency().HasProducer(newView->result_token_.front(), ToStmtPtr(*newView)));
    EXPECT_TRUE(function->GetVarDependency().HasConsumer(newView->result_token_.front(), ToStmtPtr(tokenSink)));
    EXPECT_TRUE(HasToken(tokenSink, newView->result_token_.front()));
    EXPECT_FALSE(function->GetVarDependency().HasDependency(oldResultToken));
}

TEST_F(TestRemoveRedundantOpPass, WarVersionAssembleIsPreserved)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "WarVersionAssembleIsPreserved",
                                               "WarVersionAssembleIsPreserved", nullptr);
    ASSERT_NE(function, nullptr);

    IRBuilder builder;
    std::vector<int64_t> shape = {kNumFour, kNumExpFour};
    auto sharedRaw = builder.CreateRawTensor(DT_FP32, Shape{kNumFour, kNumExpFour});
    auto oldSource = builder.CreateTensorVar(*function, DT_FP32, shape, CreateTestConstIntVector(shape));
    auto newSource = builder.CreateTensorVar(*function, DT_FP32, shape, CreateTestConstIntVector(shape));
    auto oldInput = builder.CreateTensorVar(*function, DT_FP32, shape, CreateTestConstIntVector(shape));
    auto newInput = builder.CreateTensorVar(*function, DT_FP32, shape, CreateTestConstIntVector(shape));
    auto oldVersion = builder.CreateTensorVar(*function, sharedRaw, {0, 0}, shape);
    auto newVersion = builder.CreateTensorVar(*function, sharedRaw, {0, 0}, shape);
    auto readOutput = builder.CreateTensorVar(*function, DT_FP32, shape, CreateTestConstIntVector(shape));
    auto finalOutput = builder.CreateTensorVar(*function, DT_FP32, shape, CreateTestConstIntVector(shape));
    oldInput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    newInput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    oldVersion->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    newVersion->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);

    builder.CreateTensorOpStmt(*function, Opcode::OP_EXP, {oldSource}, {oldInput});
    builder.CreateTensorOpStmt(*function, Opcode::OP_EXP, {newSource}, {newInput});
    auto& oldWrite = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {oldInput}, {oldVersion});
    oldWrite.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& read = builder.CreateTensorOpStmt(*function, Opcode::OP_ABS, {oldVersion}, {readOutput});
    auto& newWrite = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {newInput}, {newVersion});
    newWrite.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    builder.CreateTensorOpStmt(*function, Opcode::OP_EXP, {newVersion}, {finalOutput});
    auto warToken = AddTokenEdge(*function, read, newWrite);
    function->inCasts_ = {oldSource, newSource};
    function->outCasts_ = {readOutput, finalOutput};

    RemoveRedundantOp pass;
    EXPECT_EQ(pass.PreCheck(*function), SUCCESS);
    EXPECT_EQ(pass.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(pass.PostCheck(*function), SUCCESS);

    EXPECT_EQ(CountOpcode(function, Opcode::OP_ASSEMBLE), kNumTwo);
    EXPECT_TRUE(function->Operations().Contains(oldWrite));
    EXPECT_TRUE(function->Operations().Contains(newWrite));
    EXPECT_TRUE(function->GetVarDependency().HasProducer(warToken, ToStmtPtr(read)));
    EXPECT_TRUE(function->GetVarDependency().HasConsumer(warToken, ToStmtPtr(newWrite)));
    EXPECT_TRUE(HasToken(newWrite, warToken));
}

TEST_F(TestRemoveRedundantOpPass, ViewAssembleWarVersionBoundaryIsPreserved)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "ViewAssembleWarVersionBoundaryIsPreserved",
                                               "ViewAssembleWarVersionBoundaryIsPreserved", nullptr);
    ASSERT_NE(function, nullptr);

    IRBuilder builder;
    std::vector<int64_t> fullShape = {kNumFour, kNumExpFour};
    std::vector<int64_t> partShape = {kNumTwo, kNumExpFour};
    auto sharedRaw = builder.CreateRawTensor(DT_FP32, Shape{kNumFour, kNumExpFour});
    auto oldStart = builder.CreateTensorVar(*function, DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto newStart = builder.CreateTensorVar(*function, DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto oldPart = builder.CreateTensorVar(*function, DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto newPart = builder.CreateTensorVar(*function, DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto oldVersion = builder.CreateTensorVar(*function, sharedRaw, {0, 0}, fullShape);
    auto newVersion = builder.CreateTensorVar(*function, sharedRaw, {0, 0}, fullShape);
    auto readOutput = builder.CreateTensorVar(*function, DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto finalOutput = builder.CreateTensorVar(*function, DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    oldStart->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    newStart->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    oldVersion->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    newVersion->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);

    auto& oldView = builder.CreateTensorOpStmt(*function, Opcode::OP_VIEW, {oldStart}, {oldPart});
    oldView.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& oldWrite = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {oldPart}, {oldVersion});
    oldWrite.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& read = builder.CreateTensorOpStmt(*function, Opcode::OP_ABS, {oldVersion}, {readOutput});
    auto& newView = builder.CreateTensorOpStmt(*function, Opcode::OP_VIEW, {newStart}, {newPart});
    newView.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& newWrite = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {newPart}, {newVersion});
    newWrite.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    builder.CreateTensorOpStmt(*function, Opcode::OP_EXP, {newVersion}, {finalOutput});
    auto warToken = AddTokenEdge(*function, read, newWrite);
    function->inCasts_ = {oldStart, newStart};
    function->outCasts_ = {readOutput, finalOutput};

    RemoveRedundantOp pass;
    EXPECT_EQ(pass.PreCheck(*function), SUCCESS);
    EXPECT_EQ(pass.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(pass.PostCheck(*function), SUCCESS);

    EXPECT_EQ(CountOpcode(function, Opcode::OP_VIEW), kNumTwo);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_ASSEMBLE), kNumTwo);
    EXPECT_TRUE(function->Operations().Contains(oldView));
    EXPECT_TRUE(function->Operations().Contains(oldWrite));
    EXPECT_TRUE(function->Operations().Contains(newView));
    EXPECT_TRUE(function->Operations().Contains(newWrite));
    EXPECT_TRUE(function->GetVarDependency().HasProducer(warToken, ToStmtPtr(read)));
    EXPECT_TRUE(function->GetVarDependency().HasConsumer(warToken, ToStmtPtr(newWrite)));
}

/*
SliceContractSliceSingleFullSlice
input{8,16}->slice(preceding)->contractInput{8,16}->contract->contractOutput{8,16}
                                                      ->slice(full)->sliceOutput{8,16}->exp->out
input{8,16}->slice(preceding)->contractInput{8,16}->exp->out
The view/assemble cascade leaves the strict slice-contract-slice chain to the dedicated pass, which
removes the contract and the trailing full slice while keeping the preceding materialization slice.
*/
TEST_F(TestRemoveRedundantOpPass, SliceContractSliceSingleFullSliceShouldKeepPrecedingSlice)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "SliceContractSliceSingleFull",
                                               "SliceContractSliceSingleFull", nullptr);
    ASSERT_NE(function, nullptr);

    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    std::vector<int64_t> zeroOffset = {kNumZero, kNumZero};
    auto input = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto contractInput = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto contractOutput = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto sliceOutput = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape), TileOpFormat::TILEOP_ND,
                                              "out");

    auto& precedingSlice = PassOperationUtils::AddOperation(
        *function, Opcode::OP_SLICE, {input}, {contractInput},
        [&zeroOffset](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(zeroOffset)); });
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_CONTRACT, {contractInput}, {contractOutput},
        [&zeroOffset](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(zeroOffset)); });
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_SLICE, {contractOutput}, {sliceOutput},
        [&zeroOffset](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(zeroOffset)); });
    auto& exp = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {sliceOutput}, {output});

    function->inCasts_ = {input};
    function->outCasts_ = {output};

    std::vector<Operation*> newOps;
    bool operationUpdated = false;
    ASSERT_EQ(RemoveRedundantOpUtils::ProcessContractSlice(*function, newOps, operationUpdated), SUCCESS);

    EXPECT_TRUE(operationUpdated);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_CONTRACT), kNumZero);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_SLICE), kNumOne);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_VIEW), kNumZero);
    EXPECT_EQ(exp.GetInputOperand(kSizeZero), contractInput);
    EXPECT_FALSE(precedingSlice.IsDeleted());
    EXPECT_EQ(precedingSlice.GetOOperands().front(), contractInput);
}

/*
SliceContractSliceSinglePartialSlice
input{8,16}->slice(preceding)->contractInput{8,16}->contract->contractOutput{8,16}
                                                      ->slice(partial{4,0})->sliceOutput{4,16}->exp
input{8,16}->slice(preceding)->contractInput{8,16}->view(offset{4,0})->sliceOutput{4,16}->exp
*/
TEST_F(TestRemoveRedundantOpPass, SliceContractSliceSinglePartialSliceShouldGenerateView)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "SliceContractSliceSinglePartial",
                                               "SliceContractSliceSinglePartial", nullptr);
    ASSERT_NE(function, nullptr);

    std::vector<int64_t> fullShape = {kNumEight, kNumExpFour};
    std::vector<int64_t> partShape = {kNumFour, kNumExpFour};
    std::vector<int64_t> zeroOffset = {kNumZero, kNumZero};
    std::vector<int64_t> partOffset = {kNumFour, kNumZero};
    auto input = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto contractInput = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto contractOutput = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto sliceOutput = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape),
                                              TileOpFormat::TILEOP_ND, "out");

    PassOperationUtils::AddOperation(
        *function, Opcode::OP_SLICE, {input}, {contractInput},
        [&zeroOffset](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(zeroOffset)); });
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_CONTRACT, {contractInput}, {contractOutput},
        [&zeroOffset](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(zeroOffset)); });
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_SLICE, {contractOutput}, {sliceOutput},
        [&partOffset](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(partOffset)); });
    auto& exp = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {sliceOutput}, {output});

    std::vector<Operation*> newOps;
    bool operationUpdated = false;
    ASSERT_EQ(RemoveRedundantOpUtils::ProcessContractSlice(*function, newOps, operationUpdated), SUCCESS);

    EXPECT_TRUE(operationUpdated);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_CONTRACT), kNumZero);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_SLICE), kNumOne);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_VIEW), kNumOne);
    auto* generatedView = FindSingleOp(function, Opcode::OP_VIEW);
    ASSERT_NE(generatedView, nullptr);
    EXPECT_EQ(generatedView->GetIOperands().front(), contractInput);
    EXPECT_EQ(exp.GetIOperands().front(), generatedView->GetOOperands().front());
    auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(generatedView->GetOpAttribute());
    ASSERT_NE(viewAttr, nullptr);
    EXPECT_EQ(viewAttr->GetFromOffset(), partOffset);
}

/*
SliceContractSliceMixedFanout
input{8,16}->slice(preceding)->contractInput{8,16}->contract->contractOutput{8,16}
                                                       ->slice0{0,0}->sliceOut0{4,16}->exp0
                                                       ->slice1{4,0}(L1)->sliceOut1{4,16}->exp1
input{8,16}->slice0{0,0}->sliceOut0{4,16}->exp0
           ->slice1{4,0}(L1)->sliceOut1{4,16}->exp1
Mixed L1/UNKNOWN fan-out composes the preceding slice and contract offsets into the surviving slices
so every consumer reads directly from the original source tensor.
*/
TEST_F(TestRemoveRedundantOpPass, SliceContractSliceMixedFanoutShouldComposeOffsets)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "SliceContractSliceMixedFanout",
                                               "SliceContractSliceMixedFanout", nullptr);
    ASSERT_NE(function, nullptr);

    std::vector<int64_t> fullShape = {kNumEight, kNumExpFour};
    std::vector<int64_t> partShape = {kNumFour, kNumExpFour};
    std::vector<int64_t> zeroOffset = {kNumZero, kNumZero};
    std::vector<int64_t> partOffset = {kNumFour, kNumZero};
    auto input = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto contractInput = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto contractOutput = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto sliceOut0 = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto sliceOut1 = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto out0 = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape),
                                            TileOpFormat::TILEOP_ND, "out0");
    auto out1 = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape),
                                            TileOpFormat::TILEOP_ND, "out1");
    sliceOut1->SetMemoryTypeBoth(MemoryType::MEM_L1);

    PassOperationUtils::AddOperation(
        *function, Opcode::OP_SLICE, {input}, {contractInput},
        [&zeroOffset](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(zeroOffset)); });
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_CONTRACT, {contractInput}, {contractOutput},
        [&zeroOffset](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(zeroOffset)); });
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_SLICE, {contractOutput}, {sliceOut0},
        [&zeroOffset](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(zeroOffset)); });
    auto l1Attr = std::make_shared<ViewOpAttribute>(partOffset);
    l1Attr->SetToType(MemoryType::MEM_L1);
    PassOperationUtils::AddOperation(*function, Opcode::OP_SLICE, {contractOutput}, {sliceOut1},
                                     [&l1Attr](Operation& op) { op.SetOpAttribute(l1Attr); });
    auto& exp0 = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {sliceOut0}, {out0});
    auto& exp1 = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {sliceOut1}, {out1});

    std::vector<Operation*> newOps;
    bool operationUpdated = false;
    ASSERT_EQ(RemoveRedundantOpUtils::ProcessContractSlice(*function, newOps, operationUpdated), SUCCESS);

    // EraseOperations inside the pass destroys deleted ops (precedingSlice/contract), so their
    // references from AddOperation are dangling.  Verify everything through surviving objects only.
    EXPECT_TRUE(operationUpdated);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_CONTRACT), kNumZero);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_SLICE), kNumTwo);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_VIEW), kNumZero);
    const Operation* l1Slice = nullptr;
    uint32_t sliceInputCount = kNumZero;
    for (const auto& op : function->Operations()) {
        if (op.GetOpcode() != Opcode::OP_SLICE) {
            continue;
        }
        EXPECT_EQ(op.GetIOperands().front(), input);
        ++sliceInputCount;
        if (op.GetOOperands().front() == sliceOut1) {
            l1Slice = &op;
        }
    }
    EXPECT_EQ(sliceInputCount, kNumTwo);
    ASSERT_NE(l1Slice, nullptr);
    auto composedAttr = std::dynamic_pointer_cast<ViewOpAttribute>(l1Slice->GetOpAttribute());
    ASSERT_NE(composedAttr, nullptr);
    EXPECT_EQ(composedAttr->GetFromOffset(), partOffset);
    EXPECT_EQ(composedAttr->GetTo(), MemoryType::MEM_L1);
    EXPECT_EQ(sliceOut1->GetMemoryTypeOriginal(), MemoryType::MEM_L1);
    EXPECT_EQ(exp0.GetIOperands().front(), sliceOut0);
    EXPECT_EQ(exp1.GetIOperands().front(), sliceOut1);
}

/*
SliceContractSliceL1FanoutFoldPrecedingSlice
input{12,16}(UB)->slice(preceding{4,0},To=UB)->contractInput{8,16}(UB)->contract->contractOutput{8,16}
                                                                            ->slice0{0,0}(L1)->sliceOut0{4,16}->exp0
                                                                            ->slice1{4,0}(L1)->sliceOut1{4,16}->exp1
input{12,16}->slice0{4,0}(L1)->sliceOut0{4,16}->exp0
             ->slice1{8,0}(L1)->sliceOut1{4,16}->exp1
All-L1 fan-out with a plain same-memory preceding slice folds the preceding offset into every
consumer, so both the slice and the contract disappear.
*/
TEST_F(TestRemoveRedundantOpPass, SliceContractSliceL1FanoutShouldFoldPrecedingSlice)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "SliceContractSliceL1FanoutFold",
                                               "SliceContractSliceL1FanoutFold", nullptr);
    ASSERT_NE(function, nullptr);

    std::vector<int64_t> srcShape = {kNumEight + kNumFour, kNumExpFour};
    std::vector<int64_t> midShape = {kNumEight, kNumExpFour};
    std::vector<int64_t> partShape = {kNumFour, kNumExpFour};
    std::vector<int64_t> precedingOffset = {kNumFour, kNumZero};
    std::vector<int64_t> zeroOffset = {kNumZero, kNumZero};
    std::vector<int64_t> partOffset = {kNumFour, kNumZero};
    auto input = IRBuilder().CreateTensorVar(DT_FP32, srcShape, CreateTestConstIntVector(srcShape));
    auto contractInput = IRBuilder().CreateTensorVar(DT_FP32, midShape, CreateTestConstIntVector(midShape));
    auto contractOutput = IRBuilder().CreateTensorVar(DT_FP32, midShape, CreateTestConstIntVector(midShape));
    auto sliceOut0 = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto sliceOut1 = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto out0 = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape),
                                            TileOpFormat::TILEOP_ND, "out0");
    auto out1 = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape),
                                            TileOpFormat::TILEOP_ND, "out1");
    input->SetMemoryTypeBoth(MemoryType::MEM_UB);
    contractInput->SetMemoryTypeBoth(MemoryType::MEM_UB);
    sliceOut0->SetMemoryTypeBoth(MemoryType::MEM_L1);
    sliceOut1->SetMemoryTypeBoth(MemoryType::MEM_L1);

    auto precedingAttr = std::make_shared<ViewOpAttribute>(precedingOffset);
    precedingAttr->SetToType(MemoryType::MEM_UB);
    PassOperationUtils::AddOperation(*function, Opcode::OP_SLICE, {input}, {contractInput},
                                     [&precedingAttr](Operation& op) { op.SetOpAttribute(precedingAttr); });
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_CONTRACT, {contractInput}, {contractOutput},
        [&zeroOffset](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(zeroOffset)); });
    auto l1Attr0 = std::make_shared<ViewOpAttribute>(zeroOffset);
    l1Attr0->SetToType(MemoryType::MEM_L1);
    PassOperationUtils::AddOperation(*function, Opcode::OP_SLICE, {contractOutput}, {sliceOut0},
                                     [&l1Attr0](Operation& op) { op.SetOpAttribute(l1Attr0); });
    auto l1Attr1 = std::make_shared<ViewOpAttribute>(partOffset);
    l1Attr1->SetToType(MemoryType::MEM_L1);
    PassOperationUtils::AddOperation(*function, Opcode::OP_SLICE, {contractOutput}, {sliceOut1},
                                     [&l1Attr1](Operation& op) { op.SetOpAttribute(l1Attr1); });
    auto& exp0 = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {sliceOut0}, {out0});
    auto& exp1 = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {sliceOut1}, {out1});

    std::vector<Operation*> newOps;
    bool operationUpdated = false;
    ASSERT_EQ(RemoveRedundantOpUtils::ProcessContractSlice(*function, newOps, operationUpdated), SUCCESS);

    // EraseOperations inside the pass destroys deleted ops (precedingSlice/contract), so their
    // references from AddOperation are dangling.  Verify everything through surviving objects only.
    EXPECT_TRUE(operationUpdated);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_CONTRACT), kNumZero);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_SLICE), kNumTwo);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_VIEW), kNumZero);
    std::vector<const Operation*> survivingSlices;
    for (const auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_SLICE) {
            survivingSlices.push_back(&op);
        }
    }
    ASSERT_EQ(survivingSlices.size(), 2UL);
    for (const auto* survivingSlice : survivingSlices) {
        EXPECT_EQ(survivingSlice->GetIOperands().front(), input);
        auto attr = std::dynamic_pointer_cast<ViewOpAttribute>(survivingSlice->GetOpAttribute());
        ASSERT_NE(attr, nullptr);
        EXPECT_EQ(attr->GetTo(), MemoryType::MEM_L1);
        const auto& offset = attr->GetFromOffset();
        const bool isFoldedOffset = offset == precedingOffset || offset == (std::vector<int64_t>{kNumEight, kNumZero});
        EXPECT_TRUE(isFoldedOffset);
        EXPECT_EQ(survivingSlice->GetOOperands().front()->GetMemoryTypeOriginal(), MemoryType::MEM_L1);
    }
    EXPECT_EQ(exp0.GetIOperands().front(), sliceOut0);
    EXPECT_EQ(exp1.GetIOperands().front(), sliceOut1);
}

/*
SliceContractSliceWithMaterializedMatmulInput
inputA/inputB->A_MUL_B->mmOut{8,16}->contract(materialize)->matOut{8,16}->slice(preceding)
    ->contractInput{8,16}->contract->contractOutput{8,16}->slice{4,0}->sliceOut{4,16}->exp
inputA/inputB->A_MUL_B->mmOut{8,16}->contract(materialize)->matOut{8,16}->slice(preceding)
    ->contractInput{8,16}->view{4,0}->sliceOut{4,16}->exp
The L0C copy-out contract stays outside the chain, so the slice-contract prefix (a view expansion)
is folded into the trailing slice: the chain contract is removed and a composed view is generated.
*/
TEST_F(TestRemoveRedundantOpPass, SliceContractSliceWithMaterializedMatmulInputShouldFoldToView)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "SliceContractSliceMatmulInput",
                                               "SliceContractSliceMatmulInput", nullptr);
    ASSERT_NE(function, nullptr);

    std::vector<int64_t> fullShape = {kNumEight, kNumExpFour};
    std::vector<int64_t> partShape = {kNumFour, kNumExpFour};
    std::vector<int64_t> zeroOffset = {kNumZero, kNumZero};
    std::vector<int64_t> partOffset = {kNumFour, kNumZero};
    auto inputA = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto inputB = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto matmulOutput = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto matOut = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto contractInput = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto contractOutput = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto sliceOutput = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape),
                                              TileOpFormat::TILEOP_ND, "out");

    PassOperationUtils::AddOperation(*function, Opcode::OP_A_MUL_B, {inputA, inputB}, {matmulOutput});
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_CONTRACT, {matmulOutput}, {matOut},
        [&zeroOffset](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(zeroOffset)); });
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_SLICE, {matOut}, {contractInput},
        [&zeroOffset](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(zeroOffset)); });
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_CONTRACT, {contractInput}, {contractOutput},
        [&zeroOffset](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(zeroOffset)); });
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_SLICE, {contractOutput}, {sliceOutput},
        [&partOffset](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(partOffset)); });
    auto& exp = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {sliceOutput}, {output});

    function->inCasts_ = {inputA, inputB};
    function->outCasts_ = {output};

    std::vector<Operation*> newOps;
    bool operationUpdated = false;
    ASSERT_EQ(RemoveRedundantOpUtils::ProcessContractSlice(*function, newOps, operationUpdated), SUCCESS);

    EXPECT_TRUE(operationUpdated);
    ASSERT_EQ(newOps.size(), kNumOne);
    EXPECT_EQ(newOps.front()->GetOpcode(), Opcode::OP_VIEW);
    EXPECT_EQ(newOps.front()->GetIOperands().front(), contractInput);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_CONTRACT), kNumOne);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_SLICE), kNumOne);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_VIEW), kNumOne);
    EXPECT_EQ(exp.GetInputOperand(kSizeZero), newOps.front()->GetOOperands().front());
}

/*
MultipleContractsSameOutput
input0{4,16}->contract{0,0}->sharedOut{8,16}->slice{0,0}->sliceOut{4,16}->exp
input1{4,16}->contract{4,0}->
A contract output assembled from several contract chains has no unique layout owner, so the chains
stay materialized and the generic contract-slice rewrite must not fold them.
*/
TEST_F(TestRemoveRedundantOpPass, MultipleContractsSameOutputShouldNotFold)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "MultiContractSameOutput",
                                               "MultiContractSameOutput", nullptr);
    ASSERT_NE(function, nullptr);

    std::vector<int64_t> partShape = {kNumFour, kNumExpFour};
    std::vector<int64_t> fullShape = {kNumEight, kNumExpFour};
    std::vector<int64_t> zeroOffset = {kNumZero, kNumZero};
    std::vector<int64_t> offset1 = {kNumFour, kNumZero};
    auto input0 = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto input1 = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto sharedOut = IRBuilder().CreateTensorVar(DT_FP32, fullShape, CreateTestConstIntVector(fullShape));
    auto sliceOutput = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    auto output = IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape),
                                              TileOpFormat::TILEOP_ND, "out");

    PassOperationUtils::AddOperation(
        *function, Opcode::OP_CONTRACT, {input0}, {sharedOut},
        [&zeroOffset](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(zeroOffset)); });
    PassOperationUtils::AddOperation(*function, Opcode::OP_CONTRACT, {input1}, {sharedOut}, [&offset1](Operation& op) {
        op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset1));
    });
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_SLICE, {sharedOut}, {sliceOutput},
        [&zeroOffset](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(zeroOffset)); });
    auto& exp = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {sliceOutput}, {output});

    function->inCasts_ = {input0, input1};
    function->outCasts_ = {output};

    std::vector<Operation*> newOps;
    bool operationUpdated = false;
    ASSERT_EQ(RemoveRedundantOpUtils::ProcessContractSlice(*function, newOps, operationUpdated), SUCCESS);

    EXPECT_FALSE(operationUpdated);
    EXPECT_TRUE(newOps.empty());
    EXPECT_EQ(CountOpcode(function, Opcode::OP_CONTRACT), kNumTwo);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_SLICE), kNumOne);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_VIEW), kNumZero);
    EXPECT_EQ(exp.GetInputOperand(kSizeZero), sliceOutput);
}

/*
ContractSliceMisalignedLocalOffset (fp16)
input{8,16}->contract{0,0}->contractOutput{8,16}->slice{0,8}->sliceOutput{8,8}->exp->out
The slice offset {0,8} linearizes to 8*2=16B on the fp16 input, which is not 32B aligned.  Folding
the contract+slice pair into a direct view would let the VEC consumer read the shared buffer at a
misaligned UB address (aicore errcode 0x800), so the CONTRACT+SLICE copy path must stay.
*/
TEST_F(TestRemoveRedundantOpPass, ContractSliceMisalignedLocalOffsetShouldKeepCopyPath)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "ContractSliceMisalignedLocal",
                                               "ContractSliceMisalignedLocal", nullptr);
    ASSERT_NE(function, nullptr);

    std::vector<int64_t> fullShape = {kNumEight, kNumExpFour};
    std::vector<int64_t> partShape = {kNumEight, kNumEight};
    std::vector<int64_t> zeroOffset = {kNumZero, kNumZero};
    std::vector<int64_t> misalignedOffset = {kNumZero, kNumEight};
    auto input = IRBuilder().CreateTensorVar(DT_FP16, fullShape, CreateTestConstIntVector(fullShape));
    auto contractOutput = IRBuilder().CreateTensorVar(DT_FP16, fullShape, CreateTestConstIntVector(fullShape));
    auto sliceOutput = IRBuilder().CreateTensorVar(DT_FP16, partShape, CreateTestConstIntVector(partShape));
    auto output = IRBuilder().CreateTensorVar(DT_FP16, partShape, CreateTestConstIntVector(partShape),
                                              TileOpFormat::TILEOP_ND, "out");

    PassOperationUtils::AddOperation(
        *function, Opcode::OP_CONTRACT, {input}, {contractOutput},
        [&zeroOffset](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(zeroOffset)); });
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_SLICE, {contractOutput}, {sliceOutput},
        [&misalignedOffset](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(misalignedOffset)); });
    auto& exp = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {sliceOutput}, {output});

    function->inCasts_ = {input};
    function->outCasts_ = {output};

    std::vector<Operation*> newOps;
    bool operationUpdated = false;
    ASSERT_EQ(RemoveRedundantOpUtils::ProcessContractSlice(*function, newOps, operationUpdated), SUCCESS);

    EXPECT_FALSE(operationUpdated);
    EXPECT_TRUE(newOps.empty());
    EXPECT_EQ(CountOpcode(function, Opcode::OP_CONTRACT), kNumOne);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_SLICE), kNumOne);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_VIEW), kNumZero);
    EXPECT_EQ(exp.GetInputOperand(kSizeZero), sliceOutput);
}

/*
ContractSliceAlignedLocalOffset (fp16)
input{8,32}->contract{0,0}->contractOutput{8,32}->slice{0,16}->sliceOutput{8,16}->exp->out
The slice offset {0,16} linearizes to 16*2=32B on the fp16 input, which is 32B aligned.  The
alignment guard must not over-block: the contract+slice pair is still folded into a direct view.
*/
TEST_F(TestRemoveRedundantOpPass, ContractSliceAlignedLocalOffsetShouldGenerateView)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "ContractSliceAlignedLocal",
                                               "ContractSliceAlignedLocal", nullptr);
    ASSERT_NE(function, nullptr);

    std::vector<int64_t> fullShape = {kNumEight, kNumExpFive};
    std::vector<int64_t> partShape = {kNumEight, kNumExpFour};
    std::vector<int64_t> zeroOffset = {kNumZero, kNumZero};
    std::vector<int64_t> alignedOffset = {kNumZero, kNumExpFour};
    auto input = IRBuilder().CreateTensorVar(DT_FP16, fullShape, CreateTestConstIntVector(fullShape));
    auto contractOutput = IRBuilder().CreateTensorVar(DT_FP16, fullShape, CreateTestConstIntVector(fullShape));
    auto sliceOutput = IRBuilder().CreateTensorVar(DT_FP16, partShape, CreateTestConstIntVector(partShape));
    auto output = IRBuilder().CreateTensorVar(DT_FP16, partShape, CreateTestConstIntVector(partShape),
                                              TileOpFormat::TILEOP_ND, "out");

    PassOperationUtils::AddOperation(
        *function, Opcode::OP_CONTRACT, {input}, {contractOutput},
        [&zeroOffset](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(zeroOffset)); });
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_SLICE, {contractOutput}, {sliceOutput},
        [&alignedOffset](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(alignedOffset)); });
    auto& exp = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {sliceOutput}, {output});

    function->inCasts_ = {input};
    function->outCasts_ = {output};

    std::vector<Operation*> newOps;
    bool operationUpdated = false;
    ASSERT_EQ(RemoveRedundantOpUtils::ProcessContractSlice(*function, newOps, operationUpdated), SUCCESS);

    EXPECT_TRUE(operationUpdated);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_CONTRACT), kNumZero);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_SLICE), kNumZero);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_VIEW), kNumOne);
    auto* generatedView = FindSingleOp(function, Opcode::OP_VIEW);
    ASSERT_NE(generatedView, nullptr);
    EXPECT_EQ(generatedView->GetIOperands().front(), input);
    EXPECT_EQ(exp.GetInputOperand(kSizeZero), generatedView->GetOOperands().front());
    auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(generatedView->GetOpAttribute());
    ASSERT_NE(viewAttr, nullptr);
    EXPECT_EQ(viewAttr->GetFromOffset(), alignedOffset);
}

/*
SliceContractSliceMisalignedLocalOffset (fp16)
input{8,16}->slice{0,0}->contractInput{8,16}->contract{0,0}->contractOutput{8,16}
                                                       ->slice{0,8}->sliceOutput{8,8}->exp->out
The trailing slice reads the contract input at a 16B tail-dim offset (8 fp16 elements), so folding
the chain into a direct view would produce a misaligned UB access.  The whole slice-contract-slice
chain stays materialized.
*/
TEST_F(TestRemoveRedundantOpPass, SliceContractSliceMisalignedLocalOffsetShouldKeepCopyPath)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "SliceContractSliceMisalignedLocal",
                                               "SliceContractSliceMisalignedLocal", nullptr);
    ASSERT_NE(function, nullptr);

    std::vector<int64_t> fullShape = {kNumEight, kNumExpFour};
    std::vector<int64_t> partShape = {kNumEight, kNumEight};
    std::vector<int64_t> zeroOffset = {kNumZero, kNumZero};
    std::vector<int64_t> misalignedOffset = {kNumZero, kNumEight};
    auto input = IRBuilder().CreateTensorVar(DT_FP16, fullShape, CreateTestConstIntVector(fullShape));
    auto contractInput = IRBuilder().CreateTensorVar(DT_FP16, fullShape, CreateTestConstIntVector(fullShape));
    auto contractOutput = IRBuilder().CreateTensorVar(DT_FP16, fullShape, CreateTestConstIntVector(fullShape));
    auto sliceOutput = IRBuilder().CreateTensorVar(DT_FP16, partShape, CreateTestConstIntVector(partShape));
    auto output = IRBuilder().CreateTensorVar(DT_FP16, partShape, CreateTestConstIntVector(partShape),
                                              TileOpFormat::TILEOP_ND, "out");

    PassOperationUtils::AddOperation(
        *function, Opcode::OP_SLICE, {input}, {contractInput},
        [&zeroOffset](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(zeroOffset)); });
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_CONTRACT, {contractInput}, {contractOutput},
        [&zeroOffset](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(zeroOffset)); });
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_SLICE, {contractOutput}, {sliceOutput},
        [&misalignedOffset](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(misalignedOffset)); });
    auto& exp = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {sliceOutput}, {output});

    function->inCasts_ = {input};
    function->outCasts_ = {output};

    std::vector<Operation*> newOps;
    bool operationUpdated = false;
    ASSERT_EQ(RemoveRedundantOpUtils::ProcessContractSlice(*function, newOps, operationUpdated), SUCCESS);

    EXPECT_FALSE(operationUpdated);
    EXPECT_TRUE(newOps.empty());
    EXPECT_EQ(CountOpcode(function, Opcode::OP_CONTRACT), kNumOne);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_SLICE), kNumTwo);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_VIEW), kNumZero);
    EXPECT_EQ(exp.GetInputOperand(kSizeZero), sliceOutput);
}

namespace {
struct L1FanoutFoldGraph {
    std::shared_ptr<Function> function;
    LogicalTensorPtr input;
    LogicalTensorPtr precedingOutput;
};

// input{8,32}->slice{precedingOffset}(same-mem view)->contractInput{8,32-preOffset[1]}->contract{0,0}
// ->contractOutput->{slice{0,0}(To=L1)->exp->out0, slice{0,0}(To=L1)->exp->out1}
L1FanoutFoldGraph BuildL1FanoutFoldGraph(const std::string& name, const std::vector<int64_t>& precedingOffset)
{
    L1FanoutFoldGraph graph;
    graph.function = std::make_shared<Function>(Program::GetInstance(), name, name, nullptr);
    std::vector<int64_t> sourceShape = {kNumEight, kNumExpFive};
    std::vector<int64_t> innerShape = {kNumEight, kNumExpFive - precedingOffset[kSizeOne]};
    std::vector<int64_t> zeroOffset = {kNumZero, kNumZero};
    graph.input = IRBuilder().CreateTensorVar(DT_FP16, sourceShape, CreateTestConstIntVector(sourceShape));
    graph.precedingOutput = IRBuilder().CreateTensorVar(DT_FP16, innerShape, CreateTestConstIntVector(innerShape));
    auto contractOutput = IRBuilder().CreateTensorVar(DT_FP16, innerShape, CreateTestConstIntVector(innerShape));
    auto sliceOutput0 = IRBuilder().CreateTensorVar(DT_FP16, innerShape, CreateTestConstIntVector(innerShape));
    auto sliceOutput1 = IRBuilder().CreateTensorVar(DT_FP16, innerShape, CreateTestConstIntVector(innerShape));
    sliceOutput0->SetMemoryTypeBoth(MemoryType::MEM_L1);
    sliceOutput1->SetMemoryTypeBoth(MemoryType::MEM_L1);
    PassOperationUtils::AddOperation(
        *graph.function, Opcode::OP_SLICE, {graph.input}, {graph.precedingOutput},
        [&precedingOffset](Operation& op) { op.SetOpAttribute(std::make_shared<ViewOpAttribute>(precedingOffset)); });
    PassOperationUtils::AddOperation(
        *graph.function, Opcode::OP_CONTRACT, {graph.precedingOutput}, {contractOutput},
        [&zeroOffset](Operation& op) { op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(zeroOffset)); });
    for (const auto& sliceOutput : {sliceOutput0, sliceOutput1}) {
        auto l1Attr = std::make_shared<ViewOpAttribute>(zeroOffset);
        l1Attr->SetToType(MemoryType::MEM_L1);
        PassOperationUtils::AddOperation(*graph.function, Opcode::OP_SLICE, {contractOutput}, {sliceOutput},
                                         [&l1Attr](Operation& op) { op.SetOpAttribute(l1Attr); });
    }
    auto out0 = IRBuilder().CreateTensorVar(DT_FP16, innerShape, CreateTestConstIntVector(innerShape),
                                            TileOpFormat::TILEOP_ND, "out0");
    auto out1 = IRBuilder().CreateTensorVar(DT_FP16, innerShape, CreateTestConstIntVector(innerShape),
                                            TileOpFormat::TILEOP_ND, "out1");
    PassOperationUtils::AddOperation(*graph.function, Opcode::OP_EXP, {sliceOutput0}, {out0});
    PassOperationUtils::AddOperation(*graph.function, Opcode::OP_EXP, {sliceOutput1}, {out1});
    graph.function->inCasts_ = {graph.input};
    graph.function->outCasts_ = {out0, out1};
    return graph;
}

std::vector<const Operation*> FindSlicesWithInput(const std::shared_ptr<Function>& function,
                                                  const LogicalTensorPtr& input)
{
    std::vector<const Operation*> slices;
    for (const auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_SLICE && op.GetIOperands().front() == input) {
            slices.push_back(&op);
        }
    }
    return slices;
}
} // namespace

/*
SliceContractSliceL1FanoutMisalignedMergedOffset (fp16): graph built by BuildL1FanoutFoldGraph with
precedingOffset {0,8}, i.e. input{8,32}->slice{0,8}(same-mem view)->contractInput{8,24}->contract
->{slice{0,0}(To=L1)->exp->out0, slice{0,0}(To=L1)->exp->out1}.  All consumers are L1 slices
(keepL1Fanout).  The preceding offset {0,8} linearizes to 16B on the fp16 source, so the merged offset
{0,8} is not 32B aligned: the L1 fold must keep the preceding slice materialization and let the
consumers read the contract input at the (aligned) local offset {0,0} instead.
*/
TEST_F(TestRemoveRedundantOpPass, SliceContractSliceL1FanoutMisalignedMergedOffsetShouldKeepMaterialization)
{
    auto graph = BuildL1FanoutFoldGraph("L1FanoutMisalignedMerged", {kNumZero, kNumEight});
    std::vector<Operation*> newOps;
    bool operationUpdated = false;
    ASSERT_EQ(RemoveRedundantOpUtils::Process(*graph.function, newOps, operationUpdated), SUCCESS);
    EXPECT_TRUE(operationUpdated);
    EXPECT_EQ(CountOpcode(graph.function, Opcode::OP_CONTRACT), kNumZero);
    EXPECT_EQ(CountOpcode(graph.function, Opcode::OP_SLICE), kNumThree);
    EXPECT_EQ(CountOpcode(graph.function, Opcode::OP_VIEW), kNumZero);
    const auto consumers = FindSlicesWithInput(graph.function, graph.precedingOutput);
    ASSERT_EQ(consumers.size(), 2UL);
    for (const auto* consumer : consumers) {
        auto attr = std::dynamic_pointer_cast<ViewOpAttribute>(consumer->GetOpAttribute());
        ASSERT_NE(attr, nullptr);
        EXPECT_EQ(attr->GetTo(), MemoryType::MEM_L1);
        EXPECT_EQ(attr->GetFromOffset(), (std::vector<int64_t>{kNumZero, kNumZero}));
    }
}

/*
SliceContractSliceL1FanoutAlignedMergedOffset (fp16): same graph as the misaligned case but built with
precedingOffset {0,16} (contractInput{8,16}).  The preceding offset linearizes to 32B on the fp16
source, so the merged offset {0,16} is aligned and the L1 fold must proceed: consumers read the source
tensor directly at {0,16}, and the preceding slice and the contract both disappear.
*/
TEST_F(TestRemoveRedundantOpPass, SliceContractSliceL1FanoutAlignedMergedOffsetShouldFoldToSource)
{
    auto graph = BuildL1FanoutFoldGraph("L1FanoutAlignedMerged", {kNumZero, kNumExpFour});
    std::vector<Operation*> newOps;
    bool operationUpdated = false;
    ASSERT_EQ(RemoveRedundantOpUtils::Process(*graph.function, newOps, operationUpdated), SUCCESS);
    EXPECT_TRUE(operationUpdated);
    EXPECT_EQ(CountOpcode(graph.function, Opcode::OP_CONTRACT), kNumZero);
    EXPECT_EQ(CountOpcode(graph.function, Opcode::OP_SLICE), kNumTwo);
    EXPECT_EQ(CountOpcode(graph.function, Opcode::OP_VIEW), kNumZero);
    const auto consumers = FindSlicesWithInput(graph.function, graph.input);
    ASSERT_EQ(consumers.size(), 2UL);
    for (const auto* consumer : consumers) {
        auto attr = std::dynamic_pointer_cast<ViewOpAttribute>(consumer->GetOpAttribute());
        ASSERT_NE(attr, nullptr);
        EXPECT_EQ(attr->GetTo(), MemoryType::MEM_L1);
        EXPECT_EQ(attr->GetFromOffset(), (std::vector<int64_t>{kNumZero, kNumExpFour}));
    }
}
/*
TestReduceAccViewWithMaxZeroClampShouldRemove
viewIn{8,16}(DDR)[reduce_acc_view_dim,16]->view->viewOut{8,16}(DDR)[RUNTIME_Max(reduce_acc_view_dim,0),16]
                                                            ->reduceAcc->outCast{8,16}
view 输出仅被 ReduceAcc 消费且输入输出有效形状仅相差无效的 RUNTIME_Max(expr, 0) 截断时，
放宽比较后视为冗余 view 并删除，ReduceAcc 直连 viewIn
*/
TEST_F(TestRemoveRedundantOpPass, ReduceAccViewWithMaxZeroClampShouldRemove)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReduceAccViewClampRemove",
                                                      "TestReduceAccViewClampRemove", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    std::vector<int64_t> offset = {kNumZero, kNumZero};
    auto viewIn = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    viewIn->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto viewOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    viewOut->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto reduceDim = CreateTestScalarVar("reduce_acc_view_dim");
    viewIn->UpdateDynValidShape({reduceDim, SymbolicScalar(kNumExpFour)});
    viewOut->UpdateDynValidShape({reduceDim.Max(kNumZero), SymbolicScalar(kNumExpFour)});
    EXPECT_EQ(viewIn->GetDynValidShape()[0].Dump(), "reduce_acc_view_dim");
    EXPECT_EQ(viewOut->GetDynValidShape()[0].Dump(), "RUNTIME_Max(reduce_acc_view_dim, 0)");

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {viewIn}, {viewOut}, [&offset](Operation& op) {
        op.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset));
    });
    auto& reduceAcc = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_REDUCE_ACC, {viewOut}, {outCast});

    currFunctionPtr->inCasts_.push_back(viewIn);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp pass;
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);

    EXPECT_EQ(CountOpcode(currFunctionPtr, Opcode::OP_VIEW), kNumZero) << "view differing only by Max(expr,0) clamp "
                                                                          "for ReduceAcc should be removed";
    EXPECT_EQ(reduceAcc.GetInputOperand(kSizeZero), viewIn);
}

/*
TestReduceAccViewWithMaxZeroClampAndOtherConsumerShouldKeep
viewIn{8,16}(DDR)[reduce_acc_view_dim,16]->view->viewOut{8,16}(DDR)[RUNTIME_Max(reduce_acc_view_dim,0),16]
                                                            ->reduceAcc->outCast1{8,16}
                                                            ->exp->outCast2{8,16}
view 输出除 ReduceAcc 外还有其它消费方时不放宽有效形状比较，
有效形状不同的 view 不删除
*/
TEST_F(TestRemoveRedundantOpPass, ReduceAccViewWithMaxZeroClampAndOtherConsumerShouldKeep)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReduceAccViewClampKeep",
                                                      "TestReduceAccViewClampKeep", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    std::vector<int64_t> offset = {kNumZero, kNumZero};
    auto viewIn = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    viewIn->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto viewOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    viewOut->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto outCast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto reduceDim = CreateTestScalarVar("reduce_acc_view_dim");
    viewIn->UpdateDynValidShape({reduceDim, SymbolicScalar(kNumExpFour)});
    viewOut->UpdateDynValidShape({reduceDim.Max(kNumZero), SymbolicScalar(kNumExpFour)});

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {viewIn}, {viewOut}, [&offset](Operation& op) {
        op.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset));
    });
    auto& reduceAcc = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_REDUCE_ACC, {viewOut}, {outCast1});
    auto& expOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {viewOut}, {outCast2});

    currFunctionPtr->inCasts_.push_back(viewIn);
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->outCasts_.push_back(outCast2);

    RemoveRedundantOp pass;
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);

    EXPECT_EQ(CountOpcode(currFunctionPtr, Opcode::OP_VIEW), kNumOne) << "view with non-ReduceAcc consumer should "
                                                                         "not relax dyn valid shape compare";
    EXPECT_EQ(reduceAcc.GetInputOperand(kSizeZero), viewOut);
    EXPECT_EQ(expOp.GetInputOperand(kSizeZero), viewOut);
}
} // namespace tile_fwk
} // namespace npu

namespace {

// Build a contract₁ -> slice -> contract₂ sandwich on the output side of a cube op:
//   cubeOut  --CONTRACT(to {0,0})-->    middle[1,128]
//   cubeOut2 --CONTRACT(to {0,64})-->   middle            (optional, double producer)
//   middle   --SLICE(fromOffset, sliceShape)--> tile
//   tile     --CONTRACT(to {0,0})-->    out
// An extra non-matching slice can be attached to middle to keep it alive.
// Only references to surviving ops and tensors are kept: eliminated operations are asserted
// through CountOpcode so no dangling pointer is ever touched after EraseOperations.
struct CscSandwich {
    Operation* consumerContract = nullptr;
    LogicalTensorPtr matmulOut;
    LogicalTensorPtr middle;
    LogicalTensorPtr tile;
};

CscSandwich BuildCscSandwich(Function& function, Opcode cubeOpcode, const std::vector<int64_t>& sliceFromOffset,
                             const std::vector<int64_t>& sliceShape, bool addSecondProducer, bool addKeptSlice,
                             const std::vector<int64_t>& secondProducerOffset = {kNumZero, 64})
{
    constexpr int64_t kCscTileCols = kNumExpSix;
    constexpr int64_t kCscMiddleCols = kNumExpSeven;
    CscSandwich g;
    g.matmulOut = IRBuilder().CreateTensorVar(DT_FP32, {1, kCscTileCols}, CreateTestConstIntVector({1, kCscTileCols}));
    auto inA = IRBuilder().CreateTensorVar(DT_FP32, {1, kCscTileCols}, CreateTestConstIntVector({1, kCscTileCols}));
    auto inB = IRBuilder().CreateTensorVar(DT_FP32, {kCscTileCols, kCscTileCols},
                                           CreateTestConstIntVector({kCscTileCols, kCscTileCols}));
    IRBuilder().CreateTensorOpStmt(function, cubeOpcode, {inA, inB}, {g.matmulOut});

    g.middle = IRBuilder().CreateTensorVar(DT_FP32, {1, kCscMiddleCols}, CreateTestConstIntVector({1, kCscMiddleCols}));
    auto& producerContract = IRBuilder().CreateTensorOpStmt(function, Opcode::OP_CONTRACT, {g.matmulOut}, {g.middle});
    producerContract.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{kNumZero, kNumZero}));

    if (addSecondProducer) {
        auto matmulOut2 = IRBuilder().CreateTensorVar(DT_FP32, {1, kCscTileCols},
                                                      CreateTestConstIntVector({1, kCscTileCols}));
        auto inA2 = IRBuilder().CreateTensorVar(DT_FP32, {1, kCscTileCols},
                                                CreateTestConstIntVector({1, kCscTileCols}));
        auto inB2 = IRBuilder().CreateTensorVar(DT_FP32, {kCscTileCols, kCscTileCols},
                                                CreateTestConstIntVector({kCscTileCols, kCscTileCols}));
        IRBuilder().CreateTensorOpStmt(function, cubeOpcode, {inA2, inB2}, {matmulOut2});
        auto& producerContract2 = IRBuilder().CreateTensorOpStmt(function, Opcode::OP_CONTRACT, {matmulOut2},
                                                                 {g.middle});
        producerContract2.SetOpAttribute(std::make_shared<AssembleOpAttribute>(secondProducerOffset));
    }

    g.tile = IRBuilder().CreateTensorVar(DT_FP32, sliceShape, CreateTestConstIntVector(sliceShape));
    auto& slice = IRBuilder().CreateTensorOpStmt(function, Opcode::OP_SLICE, {g.middle}, {g.tile});
    auto dynFromOffset = SymbolicScalar::FromConcrete(sliceFromOffset);
    auto tileValidShape = GetViewValidShape(g.middle->GetDynValidShape(), sliceFromOffset, dynFromOffset, sliceShape);
    slice.SetOpAttribute(std::make_shared<ViewOpAttribute>(sliceFromOffset, dynFromOffset, tileValidShape));

    auto out = IRBuilder().CreateTensorVar(DT_FP32, {1, kCscMiddleCols}, CreateTestConstIntVector({1, kCscMiddleCols}));
    g.consumerContract = &IRBuilder().CreateTensorOpStmt(function, Opcode::OP_CONTRACT, {g.tile}, {out});
    g.consumerContract->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{kNumZero, kNumZero}));

    if (addKeptSlice) {
        auto smallTile = IRBuilder().CreateTensorVar(DT_FP32, {1, 16}, CreateTestConstIntVector({1, 16}));
        auto& keptSlice = IRBuilder().CreateTensorOpStmt(function, Opcode::OP_SLICE, {g.middle}, {smallTile});
        auto keptOffset = std::vector<int64_t>{kNumZero, 16};
        auto keptDynOffset = SymbolicScalar::FromConcrete(keptOffset);
        auto keptValid = GetViewValidShape(g.middle->GetDynValidShape(), keptOffset, keptDynOffset, {1, 16});
        keptSlice.SetOpAttribute(std::make_shared<ViewOpAttribute>(keptOffset, keptDynOffset, keptValid));
    }
    return g;
}

} // namespace

// Matmul-backed sandwich with a fully-overlapping slice region: the slice and both producer
// contracts are eliminated and the downstream contract consumes the matmul output directly.
TEST_F(TestRemoveRedundantOpPass, ContractSliceContractSandwichEliminated)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "TestCscEliminate", "TestCscEliminate", nullptr);
    ASSERT_NE(function, nullptr);
    function->SetGraphType(GraphType::TENSOR_GRAPH);
    auto g = BuildCscSandwich(*function, Opcode::OP_A_MUL_B, {kNumZero, kNumZero}, {1, kNumExpSix}, true, false);
    ASSERT_EQ(CountOpcode(function, Opcode::OP_CONTRACT), kNumThree);
    ASSERT_EQ(CountOpcode(function, Opcode::OP_SLICE), kNumOne);

    std::vector<Operation*> newOps;
    bool operationUpdated = false;
    ASSERT_EQ(RemoveRedundantOpUtils::ProcessContractSlice(*function, newOps, operationUpdated), SUCCESS);
    EXPECT_TRUE(operationUpdated);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_CONTRACT), kNumOne);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_SLICE), kNumZero);
    ASSERT_EQ(g.consumerContract->GetIOperands().size(), 1UL);
    EXPECT_EQ(g.consumerContract->GetIOperands().front(), g.matmulOut);
    // The orphaned middle producers are removed here; the second matmul's output loses its only
    // consumer and stays until the downstream DCE (RemoveRedundantOp pass) reclaims it.
    EXPECT_EQ(CountOpcode(function, Opcode::OP_A_MUL_B), kNumTwo);
}

// Two producers writing the exact same region as the slice is an overlapping-write conflict:
// the ambiguous guard must reject the elimination so the data flow is never silently rewritten.
TEST_F(TestRemoveRedundantOpPass, ContractSliceContractOverlappingProducersKept)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "TestCscAmbiguous", "TestCscAmbiguous", nullptr);
    ASSERT_NE(function, nullptr);
    function->SetGraphType(GraphType::TENSOR_GRAPH);
    auto g = BuildCscSandwich(*function, Opcode::OP_A_MUL_B, {kNumZero, kNumZero}, {1, kNumExpSix}, true, false,
                              {kNumZero, kNumZero});

    std::vector<Operation*> newOps;
    bool operationUpdated = false;
    ASSERT_EQ(RemoveRedundantOpUtils::ProcessContractSlice(*function, newOps, operationUpdated), SUCCESS);
    EXPECT_FALSE(operationUpdated);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_CONTRACT), kNumThree);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_SLICE), kNumOne);
    ASSERT_EQ(g.consumerContract->GetIOperands().size(), 1UL);
    EXPECT_EQ(g.consumerContract->GetIOperands().front(), g.tile);
}

// Non-matmul cube producer: the narrowed guard keeps the sandwich untouched.
TEST_F(TestRemoveRedundantOpPass, ContractSliceContractNonMatmulBackedKept)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "TestCscNonMatmul", "TestCscNonMatmul", nullptr);
    ASSERT_NE(function, nullptr);
    function->SetGraphType(GraphType::TENSOR_GRAPH);
    auto g = BuildCscSandwich(*function, Opcode::OP_ADD, {kNumZero, kNumZero}, {1, kNumExpSix}, true, false);

    std::vector<Operation*> newOps;
    bool operationUpdated = false;
    ASSERT_EQ(RemoveRedundantOpUtils::ProcessContractSlice(*function, newOps, operationUpdated), SUCCESS);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_CONTRACT), kNumThree);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_SLICE), kNumOne);
    ASSERT_EQ(g.consumerContract->GetIOperands().size(), 1UL);
    EXPECT_EQ(g.consumerContract->GetIOperands().front(), g.tile);
}

// A slice region spanning two producer regions is not a full-block match: nothing is eliminated.
TEST_F(TestRemoveRedundantOpPass, ContractSliceContractOffsetMismatchKept)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "TestCscOffset", "TestCscOffset", nullptr);
    ASSERT_NE(function, nullptr);
    function->SetGraphType(GraphType::TENSOR_GRAPH);
    auto g = BuildCscSandwich(*function, Opcode::OP_A_MUL_B, {kNumZero, 32}, {1, kNumExpSix}, true, false);

    std::vector<Operation*> newOps;
    bool operationUpdated = false;
    ASSERT_EQ(RemoveRedundantOpUtils::ProcessContractSlice(*function, newOps, operationUpdated), SUCCESS);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_CONTRACT), kNumThree);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_SLICE), kNumOne);
    EXPECT_EQ(g.consumerContract->GetIOperands().front(), g.tile);
}

// A partial region of a producer block (smaller shape) is not a full-block match.
TEST_F(TestRemoveRedundantOpPass, ContractSliceContractPartialShapeKept)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "TestCscPartial", "TestCscPartial", nullptr);
    ASSERT_NE(function, nullptr);
    function->SetGraphType(GraphType::TENSOR_GRAPH);
    auto g = BuildCscSandwich(*function, Opcode::OP_A_MUL_B, {kNumZero, kNumZero}, {1, 32}, false, false);

    std::vector<Operation*> newOps;
    bool operationUpdated = false;
    ASSERT_EQ(RemoveRedundantOpUtils::ProcessContractSlice(*function, newOps, operationUpdated), SUCCESS);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_CONTRACT), kNumTwo);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_SLICE), kNumOne);
    EXPECT_EQ(g.consumerContract->GetIOperands().front(), g.tile);
}

// When the middle tensor still has a live consumer, the matched slice is eliminated but the
// producer contracts must survive (they feed the remaining consumer).
TEST_F(TestRemoveRedundantOpPass, ContractSliceContractMiddleStillUsedKeepsProducers)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "TestCscMiddleUsed", "TestCscMiddleUsed",
                                               nullptr);
    ASSERT_NE(function, nullptr);
    function->SetGraphType(GraphType::TENSOR_GRAPH);
    auto g = BuildCscSandwich(*function, Opcode::OP_A_MUL_B, {kNumZero, kNumZero}, {1, kNumExpSix}, true, true);
    ASSERT_EQ(CountOpcode(function, Opcode::OP_CONTRACT), kNumThree);
    ASSERT_EQ(CountOpcode(function, Opcode::OP_SLICE), kNumTwo);

    std::vector<Operation*> newOps;
    bool operationUpdated = false;
    ASSERT_EQ(RemoveRedundantOpUtils::ProcessContractSlice(*function, newOps, operationUpdated), SUCCESS);
    EXPECT_TRUE(operationUpdated);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_CONTRACT), kNumThree);
    EXPECT_EQ(CountOpcode(function, Opcode::OP_SLICE), kNumOne);
    EXPECT_EQ(g.consumerContract->GetIOperands().front(), g.matmulOut);
}
