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
#include <chrono>
#include <vector>
#include <string>
#include "interface/function/function.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_utils/pass_attr_defs.h"
#include "passes/pass_utils/pass_operation_utils.h"
#include "symbolic_scalar_test_utils.h"
#include "tilefwk/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"

#include "interface/tensor/irbuilder.h"
#define private public
#include "passes/tile_graph_pass/graph_optimization/split_reshape.h"

namespace npu {
namespace tile_fwk {
static const uint32_t kNumZero = 0u;
static const uint32_t kNumOne = 1u;
static const uint32_t kNumTwo = 2u;
static const uint32_t kNumThree = 3u;
static const uint32_t kNumFour = 4u;
static const uint32_t kNumSix = 6u;
static const uint32_t kNumEight = 8u;
static const uint32_t kNumTen = 10u;
static const uint32_t kNumTwelve = 12u;
static const uint32_t kExpFour = 16u;
static const uint32_t kNumTwentyFour = 24u;
static const uint32_t kExpFive = 32u;
static const uint32_t kExpSix = 64u;
static const uint32_t kNumNineSix = 96u;
static const uint32_t kExpSeven = 128u;
static const uint32_t kExpEight = 256u;
static const size_t kSizeZero = 0UL;
static const size_t kSizeOne = 1UL;
static const size_t kSizeTwo = 2UL;
static const size_t kSizeThree = 3UL;
static const size_t kSizeFour = 4UL;

class TestSplitReshapePass : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetHostConfig(KEY_STRATEGY, "SplitReshapeTestStrategy");
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {}
};

TEST_F(TestSplitReshapePass, TestInit)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    SplitReshape pass;
    auto status = pass.Init();
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(pass.assembleOutToInput_.size(), kSizeZero);
    EXPECT_EQ(pass.reshapeSources_.size(), kSizeZero);
    EXPECT_EQ(pass.mapOffset_.size(), kSizeZero);
    EXPECT_EQ(pass.mapAssembleOpMagic_.size(), kSizeZero);
    EXPECT_EQ(pass.reshapeDynOutput_.size(), kSizeZero);
    EXPECT_EQ(pass.assembles_.size(), kSizeZero);
    EXPECT_EQ(pass.reshapes_.size(), kSizeZero);
    EXPECT_EQ(pass.viewOffset_.size(), kSizeZero);
    EXPECT_EQ(pass.reshapeOffset_.size(), kSizeZero);
    EXPECT_EQ(pass.reshapeRawOutputs_.size(), kSizeZero);
    EXPECT_EQ(pass.reshapeRawInputs_.size(), kSizeZero);
    EXPECT_EQ(pass.rawToAlignCache_.size(), kSizeZero);
    EXPECT_EQ(pass.sameRawInputCache_.size(), kSizeZero);
}

void BuildGraphForCollectCopyOut(std::shared_ptr<Function>& func, std::shared_ptr<LogicalTensor>& input1,
                                 std::shared_ptr<LogicalTensor>& input2, std::shared_ptr<LogicalTensor>& ubTensor,
                                 std::shared_ptr<LogicalTensor>& output, std::vector<int64_t>& offset1,
                                 std::vector<int64_t>& offset2, std::vector<SymbolicScalar>& validShape)
{
    func = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);

    std::vector<int64_t> shape = {kNumTwo, kNumOne, kNumEight};
    offset1 = {kNumZero, kNumZero, kNumZero};
    offset2 = {kNumOne, kNumZero, kNumZero};
    std::vector<int64_t> offset3 = {kNumTwo, kNumZero, kNumZero};

    std::vector<int64_t> shape1 = {kNumOne, kNumOne, kNumEight};
    std::vector<int64_t> shape2 = {kNumThree, kNumOne, kNumEight};
    std::vector<int64_t> shape3 = {kNumThree, kNumEight};

    auto ddrRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    auto input3 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor, offset3, shape1,
                                                             CreateTestConstIntVector(shape1));
    auto copyTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));

    input1 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor, offset1, shape1,
                                                        CreateTestConstIntVector(shape1));
    input2 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor, offset2, shape1,
                                                        CreateTestConstIntVector(shape1));
    ubTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape3, CreateTestConstIntVector(shape3));

    auto& assemble_op1 = PassOperationUtils::AddOperation(*func, Opcode::OP_ASSEMBLE, {input1}, {ubTensor});
    assemble_op1.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1));

    auto& assemble_op2 = PassOperationUtils::AddOperation(*func, Opcode::OP_ASSEMBLE, {input2}, {ubTensor});
    assemble_op2.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset2));

    PassOperationUtils::AddOperation(*func, Opcode::OP_COPY_OUT, {input3}, {copyTensor});

    auto& assemble_op3 = PassOperationUtils::AddOperation(*func, Opcode::OP_ASSEMBLE, {copyTensor}, {ubTensor});
    assemble_op3.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset3));

    validShape = {CreateTestScalarVar("a"), kNumEight};
    auto& reshape_op = PassOperationUtils::AddOperation(*func, Opcode::OP_RESHAPE, {ubTensor}, {output});
    reshape_op.SetAttribute(OP_ATTR_VALID_SHAPE, validShape);
}

TEST_F(TestSplitReshapePass, TestCollectCopyOut)
{
    std::shared_ptr<Function> func;
    std::shared_ptr<LogicalTensor> input1, input2, ubTensor, output;
    std::vector<int64_t> offset1, offset2;
    std::vector<SymbolicScalar> validShape;

    BuildGraphForCollectCopyOut(func, input1, input2, ubTensor, output, offset1, offset2, validShape);
    ASSERT_TRUE(func != nullptr);

    SplitReshape pass;
    EXPECT_EQ(pass.CollectCopyOut(*func), SUCCESS);

    EXPECT_EQ(pass.reshapeSources_.size(), kSizeOne);
    auto it1 = pass.reshapeSources_.find(output->tensor->rawmagic);
    EXPECT_NE(it1, pass.reshapeSources_.end());
    EXPECT_EQ(it1->second, ubTensor);

    EXPECT_EQ(pass.reshapeDynOutput_.size(), kSizeOne);
    auto it2 = pass.reshapeDynOutput_.find(output->tensor->rawmagic);
    EXPECT_NE(it2, pass.reshapeDynOutput_.end());
    for (size_t i = 0; i < kSizeTwo; ++i) {
        EXPECT_EQ(it2->second[i].Dump(), validShape[i].Dump());
    }

    EXPECT_EQ(pass.assembleOutToInput_.size(), kSizeOne);
    auto it3 = pass.assembleOutToInput_.find(ubTensor->tensor->rawmagic);
    EXPECT_NE(it3, pass.assembleOutToInput_.end());
    EXPECT_EQ(it3->second.size(), kNumThree);
    EXPECT_EQ(std::count(it3->second.begin(), it3->second.end(), input1), kNumOne);
    EXPECT_EQ(std::count(it3->second.begin(), it3->second.end(), input2), kNumOne);

    EXPECT_EQ(pass.mapOffset_.size(), kSizeThree);
    EXPECT_EQ(pass.mapOffset_[std::make_pair(input1->magic, ubTensor->magic)], offset1);
    EXPECT_EQ(pass.mapOffset_[std::make_pair(input2->magic, ubTensor->magic)], offset2);
}

TEST_F(TestSplitReshapePass, TestCheckSplit)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape = {kNumTwo, kNumOne, kNumEight};
    std::vector<int64_t> offset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> offset2 = {kNumOne, kNumZero, kNumZero};
    std::vector<int64_t> shape1 = {kNumOne, kNumOne, kNumEight};
    std::vector<int64_t> shape2 = {kNumTwo, kNumOne, kNumEight};
    std::vector<int64_t> shape3 = {kNumTwo, kNumEight};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape);

    auto case1Input1 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor1, offset1, shape1,
                                                                  CreateTestConstIntVector(shape1));
    auto case1Input2 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor1, offset2, shape1,
                                                                  CreateTestConstIntVector(shape1));
    auto case1UbTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto case1Output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape3, CreateTestConstIntVector(shape3));
    auto& assemble_op1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {case1Input1},
                                                          {case1UbTensor});
    auto assemble_Attr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assemble_op1.SetOpAttribute(assemble_Attr1);
    auto& assemble_op2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {case1Input2},
                                                          {case1UbTensor});
    auto assemble_Attr2 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset2);
    assemble_op2.SetOpAttribute(assemble_Attr2);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {case1UbTensor}, {case1Output});

    auto case2Input = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto case2Output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape3, CreateTestConstIntVector(shape3));
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {case2Input}, {case2Output});

    auto case3Input1 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor1, offset1, shape1,
                                                                  CreateTestConstIntVector(shape1));
    auto case3Input2 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor2, offset2, shape1,
                                                                  CreateTestConstIntVector(shape1));
    auto case3UbTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto case3Output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape3, CreateTestConstIntVector(shape3));
    auto& assemble_op3 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {case3Input1},
                                                          {case3UbTensor});
    auto assemble_Attr3 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assemble_op3.SetOpAttribute(assemble_Attr3);
    auto& assemble_op4 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {case3Input2},
                                                          {case3UbTensor});
    auto assemble_Attr4 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset2);
    assemble_op4.SetOpAttribute(assemble_Attr4);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {case3UbTensor}, {case3Output});

    auto case4Input1 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor1, offset1, shape1,
                                                                  CreateTestConstIntVector(shape1));
    auto case4UbTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto case4Output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape3, CreateTestConstIntVector(shape3));
    auto& assemble_op5 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {case4Input1},
                                                          {case4UbTensor});
    auto assemble_Attr5 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assemble_op5.SetOpAttribute(assemble_Attr5);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {case4UbTensor}, {case4Output});

    SplitReshape pass;
    auto status = pass.CollectCopyOut(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(pass.CheckSameRawInput(case1UbTensor), true);
    EXPECT_EQ(pass.CheckSameRawInput(case2Input), true);
    EXPECT_EQ(pass.CheckSameRawInput(case3UbTensor), false);
    EXPECT_EQ(pass.CheckSameRawInput(case4UbTensor), true);
}

TEST_F(TestSplitReshapePass, TestCheckDynStatus)
{
    std::vector<int64_t> input;
    std::vector<int64_t> output;
    std::vector<int64_t> alignedShape;
    std::vector<SymbolicScalar> dynOutput;
    SplitReshape pass;

    input = {kNumFour, kNumTwo};
    output = {kNumTwo, kNumFour};
    alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    dynOutput = {CreateTestScalarVar("a"), kNumFour};
    EXPECT_EQ(pass.CheckDynStatus(alignedShape, input, output, dynOutput), SUCCESS);

    input = {kNumTwo, kNumTwo, kNumTwo};
    output = {kNumFour, kNumTwo};
    alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    dynOutput = {CreateTestScalarVar("a"), kNumTwo};
    EXPECT_EQ(pass.CheckDynStatus(alignedShape, input, output, dynOutput), SUCCESS);

    input = {kNumFour, kNumTwo};
    output = {kNumTwo, kNumFour};
    alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    dynOutput = {kNumTwo, CreateTestScalarVar("a")};
    EXPECT_EQ(pass.CheckDynStatus(alignedShape, input, output, dynOutput), WARNING);

    input = {kNumFour, kNumTwo};
    output = {kNumTwo, kNumFour};
    alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    dynOutput = {kNumTwo, kNumOne, CreateTestScalarVar("a")};
    EXPECT_EQ(pass.CheckDynStatus(alignedShape, input, output, dynOutput), FAILED);

    input = {kNumTwo, kNumOne, kNumFour};
    output = {kNumTwo, kNumOne, kNumTwo, kNumTwo};
    alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    dynOutput = {CreateTestScalarVar("a"), kNumOne, kNumTwo, kNumTwo};
    EXPECT_EQ(pass.CheckDynStatus(alignedShape, input, output, dynOutput), SUCCESS);

    input = {kNumTwo, kNumOne, kNumTwo, kNumTwo};
    output = {kNumTwo, kNumOne, kNumFour};
    alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    dynOutput = {CreateTestScalarVar("a"), kNumOne, kNumFour};
    EXPECT_EQ(pass.CheckDynStatus(alignedShape, input, output, dynOutput), SUCCESS);

    input = {kNumTwo, kNumOne, kNumTwo, kNumTwo};
    output = {kNumTwo, kNumOne, kNumFour};
    alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    dynOutput = {kNumTwo, kNumOne, CreateTestScalarVar("a")};
    EXPECT_EQ(pass.CheckDynStatus(alignedShape, input, output, dynOutput), WARNING);

    input = {kNumTwo, kNumOne, kNumTwo, kNumTwo};
    output = {kNumTwo, kNumOne, kNumFour};
    alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    dynOutput = {kNumTwo, CreateTestScalarVar("a"), kNumFour};
    EXPECT_EQ(pass.CheckDynStatus(alignedShape, input, output, dynOutput), SUCCESS);
}

TEST_F(TestSplitReshapePass, TestShapeAlign)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    SplitReshape pass;
    Status status;
    std::vector<int64_t> inputShape;
    std::vector<int64_t> outputShape;
    std::vector<int64_t> alignedShape;
    std::vector<int64_t> expectedShape;

    alignedShape.clear();
    inputShape = {kExpSix, kExpSix};
    outputShape = {kExpFive, kExpSeven};
    expectedShape = {kExpFive, kNumTwo, kExpSix};
    status = pass.ShapeAlign(inputShape, outputShape, alignedShape);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(alignedShape, expectedShape);

    alignedShape.clear();
    inputShape = {kNumTwo, kExpFive, kExpSix, kExpSeven};
    outputShape = {kExpSix, kExpSix, kExpSeven};
    expectedShape = inputShape;
    status = pass.ShapeAlign(inputShape, outputShape, alignedShape);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(alignedShape, expectedShape);

    alignedShape.clear();
    inputShape = {kExpSix, kExpSix, kExpSeven};
    outputShape = {kNumTwo, kExpFive, kExpSix, kExpSeven};
    expectedShape = outputShape;
    status = pass.ShapeAlign(inputShape, outputShape, alignedShape);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(alignedShape, expectedShape);

    inputShape = {kNumNineSix, kNumFour};
    outputShape = {kExpSix, kNumSix};
    status = pass.ShapeAlign(inputShape, outputShape, alignedShape);
    EXPECT_EQ(status, WARNING);
}

TEST_F(TestSplitReshapePass, TestRawToAlign)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    SplitReshape pass;
    Status status;
    std::vector<int64_t> rawShape;
    std::vector<int64_t> alignedShape;
    std::vector<int64_t> tileOffset;
    std::vector<int64_t> tileShape;
    std::vector<int64_t> expectOffset;
    std::vector<int64_t> expectShape;
    std::vector<int64_t> newOffset;
    std::vector<int64_t> newShape;

    ReshapeTilePara shapePara;

    rawShape = {kNumEight};
    alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    tileOffset = {kNumTwo};
    tileShape = {kNumTwo};
    shapePara = {rawShape, alignedShape, tileOffset, tileShape};
    status = pass.RawToAlign(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    expectOffset = {kNumZero, kNumOne, kNumZero};
    expectShape = {kNumOne, kNumOne, kNumTwo};
    EXPECT_EQ(newShape, expectShape);
    EXPECT_EQ(newOffset, expectOffset);

    rawShape = {kExpFive, kNumTwo, kExpSeven};
    alignedShape = {kExpFive, kNumTwo, kNumFour, kExpFive};
    tileOffset = {kNumZero, kNumZero, kNumZero};
    tileShape = {kNumOne, kNumOne, kExpSix};
    shapePara = {rawShape, alignedShape, tileOffset, tileShape};
    status = pass.RawToAlign(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    expectOffset = {kNumZero, kNumZero, kNumZero, kNumZero};
    expectShape = {kNumOne, kNumOne, kNumTwo, kExpFive};
    EXPECT_EQ(newShape, expectShape);
    EXPECT_EQ(newOffset, expectOffset);

    rawShape = {kExpFive, kNumTwo, kExpEight};
    alignedShape = {kExpFive, kNumTwo, kNumEight, kExpFive};
    tileOffset = {kNumZero, kNumOne, kExpSeven};
    tileShape = {kNumOne, kNumOne, kExpSix};
    shapePara = {rawShape, alignedShape, tileOffset, tileShape};
    status = pass.RawToAlign(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    expectOffset = {kNumZero, kNumOne, kNumFour, kNumZero};
    expectShape = {kNumOne, kNumOne, kNumTwo, kExpFive};
    EXPECT_EQ(newShape, expectShape);
    EXPECT_EQ(newOffset, expectOffset);

    rawShape = {kExpFive, kNumFour, kExpEight};
    alignedShape = {kExpFive, kNumFour, kNumEight, kExpFive};
    tileOffset = {kNumOne, kNumOne, kExpSeven};
    tileShape = {kNumOne, kNumTwo, kExpSix};
    shapePara = {rawShape, alignedShape, tileOffset, tileShape};
    status = pass.RawToAlign(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    expectOffset = {kNumOne, kNumOne, kNumFour, kNumZero};
    expectShape = {kNumOne, kNumTwo, kNumTwo, kExpFive};
    EXPECT_EQ(newShape, expectShape);
    EXPECT_EQ(newOffset, expectOffset);

    rawShape = {kExpFive, kNumFour, kNumNineSix};
    alignedShape = {kExpFive, kNumFour, kNumFour, kExpFive};
    tileOffset = {kNumOne, kNumOne, kExpSeven};
    tileShape = {kNumOne, kNumTwo, kExpSix};
    shapePara = {rawShape, alignedShape, tileOffset, tileShape};
    status = pass.RawToAlign(shapePara, newOffset, newShape);
    EXPECT_EQ(status, WARNING);
}

TEST_F(TestSplitReshapePass, TestAlignToRaw)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    SplitReshape pass;
    Status status;
    std::vector<int64_t> alignedShape;
    std::vector<int64_t> rawShape;
    std::vector<int64_t> tileOffset;
    std::vector<int64_t> tileShape;
    std::vector<int64_t> expectOffset;
    std::vector<int64_t> expectShape;
    std::vector<int64_t> newOffset;
    std::vector<int64_t> newShape;

    ReshapeTilePara shapePara;

    rawShape = {kNumSix};
    alignedShape = {kNumTwo, kNumThree};
    tileOffset = {kNumOne, kNumZero};
    tileShape = {kNumOne, kNumThree};
    shapePara = {alignedShape, rawShape, tileOffset, tileShape};
    status = pass.AlignToRaw(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    expectOffset = {kNumThree};
    expectShape = {kNumThree};
    EXPECT_EQ(newShape, expectShape);
    EXPECT_EQ(newOffset, expectOffset);

    rawShape = {kExpFive, kNumTwo, kExpSeven};
    alignedShape = {kExpFive, kNumTwo, kNumFour, kExpFive};
    tileOffset = {kNumZero, kNumZero, kNumZero, kNumZero};
    tileShape = {kNumOne, kNumOne, kNumTwo, kExpFive};
    shapePara = {alignedShape, rawShape, tileOffset, tileShape};
    status = pass.AlignToRaw(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    expectOffset = {kNumZero, kNumZero, kNumZero};
    expectShape = {kNumOne, kNumOne, kExpSix};
    EXPECT_EQ(newShape, expectShape);
    EXPECT_EQ(newOffset, expectOffset);

    rawShape = {kExpFive, kNumTwo, kExpEight};
    alignedShape = {kExpFive, kNumTwo, kNumEight, kExpFive};
    tileOffset = {kNumZero, kNumOne, kNumFour, kNumZero};
    tileShape = {kNumOne, kNumOne, kNumTwo, kExpFive};
    shapePara = {alignedShape, rawShape, tileOffset, tileShape};
    status = pass.AlignToRaw(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    expectOffset = {kNumZero, kNumOne, kExpSeven};
    expectShape = {kNumOne, kNumOne, kExpSix};
    EXPECT_EQ(newShape, expectShape);
    EXPECT_EQ(newOffset, expectOffset);

    rawShape = {kExpFive, kNumFour, kExpEight};
    alignedShape = {kExpFive, kNumFour, kNumEight, kExpFive};
    tileOffset = {kNumOne, kNumOne, kNumFour, kNumZero};
    tileShape = {kNumOne, kNumTwo, kNumTwo, kExpFive};
    shapePara = {alignedShape, rawShape, tileOffset, tileShape};
    status = pass.AlignToRaw(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    expectOffset = {kNumOne, kNumOne, kExpSeven};
    expectShape = {kNumOne, kNumTwo, kExpSix};
    EXPECT_EQ(newShape, expectShape);
    EXPECT_EQ(newOffset, expectOffset);
}

TEST_F(TestSplitReshapePass, TestAlignToRawSpecialCase)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    SplitReshape pass;
    Status status;
    std::vector<int64_t> alignedShape;
    std::vector<int64_t> rawShape;
    std::vector<int64_t> tileOffset;
    std::vector<int64_t> tileShape;
    std::vector<int64_t> newOffset;
    std::vector<int64_t> newShape;

    ReshapeTilePara shapePara;

    rawShape = {kNumEight};
    alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    tileOffset = {kNumZero, kNumTwo, kNumOne};
    tileShape = {kNumTwo, kNumTwo, kNumTwo};
    shapePara = {alignedShape, rawShape, tileOffset, tileShape};
    status = pass.AlignToRaw(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_TRUE(newShape.empty());
    EXPECT_TRUE(newOffset.empty());

    rawShape = {kExpFive, kNumTwo, kExpEight};
    alignedShape = {kExpFive, kNumTwo, kNumEight, kExpFive};
    tileShape = {kNumOne, kNumOne, kNumTwo, kExpFive};
    tileOffset = {kNumZero, kNumOne, kNumZero, kExpFive};
    shapePara = {alignedShape, rawShape, tileOffset, tileShape};
    status = pass.AlignToRaw(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_TRUE(newShape.empty());
    EXPECT_TRUE(newOffset.empty());
}

/*
多对一场景
rawShape = {2, 4}
{2, 4} -> assemble -> {2, 4} -> reshape -> {2, 2, 2}
*/
TEST_F(TestSplitReshapePass, TestObtainCopyOutTileBeCovered)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape = {kNumTwo, kNumFour};
    std::vector<int64_t> offset1 = {kNumZero, kNumZero};
    std::vector<int64_t> offset2 = {kNumZero, kNumTwo};
    std::vector<int64_t> shape1 = {kNumTwo, kNumTwo};
    std::vector<int64_t> shape2 = {kNumTwo, kNumFour};
    std::vector<int64_t> shape3 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> newOutputTileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> newOutputTileShape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> alignedShape = {kNumTwo, kNumTwo, kNumTwo};

    std::shared_ptr<RawTensor> ddrRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    auto input1 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor, offset1, shape1,
                                                             CreateTestConstIntVector(shape1));
    auto input2 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor, offset2, shape1,
                                                             CreateTestConstIntVector(shape1));
    auto ubTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape3, CreateTestConstIntVector(shape3));

    auto& assemble_op1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {input1}, {ubTensor});
    auto assemble_Attr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assemble_op1.SetOpAttribute(assemble_Attr1);
    auto& assemble_op2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {input2}, {ubTensor});
    auto assemble_Attr2 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset2);
    assemble_op2.SetOpAttribute(assemble_Attr2);
    auto& reshapeOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {ubTensor}, {output});

    SplitReshape pass;
    LogicalTensors overlaps;
    LogicalTensors newOverlaps;
    std::vector<SymbolicScalar> validShape;
    auto newOutput = npu::tile_fwk::IRBuilder().CreateTensorVar(output->tensor, newOutputTileOffset, newOutputTileShape,
                                                                validShape);
    copyOutTilePara copyOutTile = {ubTensor, reshapeOp.GetOpMagic(), output, newOutput, alignedShape};
    OverlapStatus overlapStatus = OverlapStatus::NO_OVER_LAP;
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.ObtainCopyOutTile(*currFunctionPtr, copyOutTile, overlaps, newOverlaps, overlapStatus), SUCCESS);
    EXPECT_EQ(overlapStatus, OverlapStatus::PERFECTLY_MATCH_WITH_ALL);
    EXPECT_EQ(overlaps.size(), kSizeTwo);
    EXPECT_NE(std::find(overlaps.begin(), overlaps.end(), input1), overlaps.end());
    EXPECT_NE(std::find(overlaps.begin(), overlaps.end(), input2), overlaps.end());
    EXPECT_EQ(newOverlaps.size(), kSizeTwo);
}

/*
一对一场景
rawShape = {2, 2, 2}
{2, 2, 2} -> assemble -> {2, 2, 2} -> reshape -> {4, 2}
*/
TEST_F(TestSplitReshapePass, TestObtainCopyOutTilePerfectlyMatched)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> offset = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> shape1 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> shape2 = {kNumFour, kNumTwo};

    std::shared_ptr<RawTensor> ddrRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    auto input = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor, offset, shape1,
                                                            CreateTestConstIntVector(shape1));
    auto ubTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));

    auto& assemble_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {input}, {ubTensor});
    auto assemble_Attr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset);
    assemble_op.SetOpAttribute(assemble_Attr);
    auto& reshapeOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {ubTensor}, {output});

    SplitReshape pass;
    LogicalTensors overlaps;
    LogicalTensors newOverlaps;
    std::vector<SymbolicScalar> validShape;
    std::vector<int64_t> newOutputTileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> newOutputTileShape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    auto newOutput = npu::tile_fwk::IRBuilder().CreateTensorVar(output->tensor, newOutputTileOffset, newOutputTileShape,
                                                                validShape);
    copyOutTilePara copyOutTile = {ubTensor, reshapeOp.GetOpMagic(), output, newOutput, alignedShape};
    OverlapStatus overlapStatus = OverlapStatus::NO_OVER_LAP;
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.ObtainCopyOutTile(*currFunctionPtr, copyOutTile, overlaps, newOverlaps, overlapStatus), SUCCESS);
    EXPECT_EQ(overlapStatus, OverlapStatus::PERFECTLY_MATCH);
    EXPECT_EQ(overlaps.size(), kSizeOne);
    EXPECT_NE(std::find(overlaps.begin(), overlaps.end(), input), overlaps.end());
    EXPECT_EQ(newOverlaps.size(), kSizeOne);
    EXPECT_EQ(newOverlaps[0]->GetOffset(), newOutputTileOffset);
    EXPECT_EQ(newOverlaps[0]->GetShape(), newOutputTileShape);
}

/*
验证一对一场景下数据的处理
rawShape = {2, 2, 2}
{2, 2, 2}(ddr) -> assemble -> {2, 2, 2}(unknown) -> reshape -> {4, 2}(unknown) -> view -> {4,2}(ddr) -> OP
{2, 2, 2}(ddr) -> assemble -> {2, 2, 2}(unknown) -> reshape(一个ReshapeOp成员) -> {4, 2}(unknown) -> view -> {4,2}(ddr)
-> OP
*/
TEST_F(TestSplitReshapePass, TestUpdateForPerfectlyMatch)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> offset = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> shape1 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> shape2 = {kNumFour, kNumTwo};

    std::shared_ptr<RawTensor> ddrRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    auto input = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor, offset, shape1,
                                                            CreateTestConstIntVector(shape1));
    input->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    output->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    auto& assemble_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {input}, {ubTensor1});
    auto assemble_Attr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset);
    assemble_op.SetOpAttribute(assemble_Attr);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    auto& view_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {ubTensor2}, {output});
    std::vector<int64_t> view_offset = {0, 0};
    auto view_Attr = std::make_shared<ViewOpAttribute>(view_offset);
    view_op.SetOpAttribute(view_Attr);

    CalcOverlapPara para;
    std::vector<SymbolicScalar> validShape;
    std::vector<int64_t> newTileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> newTileShape = {kNumTwo, kNumTwo, kNumTwo};
    para.alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    para.overlaps = {input};
    para.newOverlaps = {
        npu::tile_fwk::IRBuilder().CreateTensorVar(input->tensor, newTileOffset, newTileShape, validShape)};
    para.reshapeSource = ubTensor1;
    para.input = ubTensor2;
    para.output = output;
    para.inputView = ubTensor2;

    SplitReshape pass;
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.ProcessOnetoOne(*currFunctionPtr, view_op, para), SUCCESS);
    auto newReshapeOutput = view_op.GetInputOperand(kSizeZero);
    EXPECT_EQ(pass.assembles_.size(), kSizeOne);
    auto assemble = pass.assembles_.begin();
    auto reshape = pass.reshapes_.begin()->second;
    auto newReshapeSource = reshape->input;
    EXPECT_EQ(reshape->output, newReshapeOutput);
    EXPECT_EQ(newReshapeSource->GetMemoryTypeOriginal(), MemoryType::MEM_UNKNOWN);
    EXPECT_NE(newReshapeOutput, ubTensor2);
    EXPECT_EQ(newReshapeOutput->GetMemoryTypeOriginal(), MemoryType::MEM_UNKNOWN);
    EXPECT_EQ(pass.reshapes_.size(), kSizeOne);
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(view_op.GetOpAttribute().get());
    EXPECT_EQ(viewOpAttribute->GetFromOffset(), view_offset);
    EXPECT_EQ(assemble->from, MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(assemble->toOffset, offset);
    EXPECT_EQ(assemble->input, input);
    EXPECT_EQ(assemble->output, newReshapeSource);
}

/*
验证一对一场景下动态shape数据的处理
rawShape = {2, 2, 2}
{2, 2, 2}(ddr) -> assemble -> {2, 2, 2}(unknown) -> reshape -> {4, 2}(unknown) -> view -> {4,2}(ddr) -> OP
                                                     {4, a}
{2, 2, 2}(ddr) -> assemble -> {2, 2, 2}(unknown) -> reshape(一个ReshapeOp成员) -> {4, 2}(unknown) -> view -> {4,2}(ddr)
-> OP
*/
TEST_F(TestSplitReshapePass, TestDynUpdateForPerfectlyMatch)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> offset = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> shape1 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> shape2 = {kNumFour, kNumTwo};
    std::vector<int64_t> view_offset = {kNumZero, kNumZero};
    std::vector<SymbolicScalar> validShape = {kNumFour, CreateTestScalarVar("a")};

    std::shared_ptr<RawTensor> ddrRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    auto input = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor, offset, shape1,
                                                            CreateTestConstIntVector(shape1));
    input->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    output->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    auto& assemble_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {input}, {ubTensor1});
    auto assemble_Attr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset);
    assemble_op.SetOpAttribute(assemble_Attr);
    auto& reshape_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    reshape_op.SetAttribute(OP_ATTR_VALID_SHAPE, validShape);
    auto& view_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {ubTensor2}, {output});
    auto view_Attr = std::make_shared<ViewOpAttribute>(view_offset);
    view_op.SetOpAttribute(view_Attr);

    CalcOverlapPara para;
    std::vector<int64_t> newTileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> newTileShape = {kNumTwo, kNumTwo, kNumTwo};
    para.alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    para.overlaps = {input};
    auto newOverlap = npu::tile_fwk::IRBuilder().CreateTensorVar(input->tensor, newTileOffset, newTileShape,
                                                                 CreateTestConstIntVector(newTileShape));
    para.newOverlaps = {newOverlap};
    para.reshapeSource = ubTensor1;
    para.input = ubTensor2;
    para.output = output;
    para.inputView = ubTensor2;
    auto newValidShape = GetViewValidShape(validShape, view_offset, {}, shape2);
    para.oriViewDynShape = newValidShape;

    SplitReshape pass;
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.ProcessOnetoOne(*currFunctionPtr, view_op, para), SUCCESS);
    auto newReshapeOutput = view_op.GetInputOperand(kSizeZero);
    EXPECT_NE(newReshapeOutput, ubTensor2);
    EXPECT_EQ(newReshapeOutput->GetMemoryTypeOriginal(), MemoryType::MEM_UNKNOWN);
    EXPECT_EQ(pass.reshapes_.size(), kSizeOne);
    auto reshape = pass.reshapes_.begin()->second;
    EXPECT_EQ(pass.assembles_.size(), kSizeOne);
    auto assemble = pass.assembles_.begin();
    auto newReshapeSource = reshape->input;
    EXPECT_EQ(reshape->output, newReshapeOutput);
    EXPECT_EQ(reshape->dynValidShapes.size(), kSizeOne);
    EXPECT_EQ(reshape->dynValidShapes[0].size(), kSizeTwo);
    std::vector<std::string> expectValidShape = {
        "4", "(RUNTIME_Min(RUNTIME_Max(a, 0), 2)*(RUNTIME_Min(RUNTIME_Max(a, 0), 2)!=0))"};
    for (size_t i = 0; i < kSizeTwo; ++i) {
        EXPECT_EQ(reshape->dynValidShapes[0][i].Dump(), expectValidShape[i]);
    }
    EXPECT_EQ(newReshapeSource->GetMemoryTypeOriginal(), MemoryType::MEM_UNKNOWN);
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(view_op.GetOpAttribute().get());
    EXPECT_EQ(viewOpAttribute->GetFromOffset(), view_offset);
    EXPECT_EQ(assemble->from, MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(assemble->input, input);
    EXPECT_EQ(assemble->output, newReshapeSource);
}

/*
验证一对多场景下数据的处理
rawShape = {2, 2, 2}
{2, 2, 2}(ddr) -> assemble -> {2, 2, 2} -> reshape -> {2, 4}(unknown) -> view -> {2, 2}(ddr)
                                                                      -> view -> {2, 2}(ddr)
{2, 2, 2}(ddr) -> assemble -> {2, 2, 2} -> reshape -> {2, 4}(unknown) -> view -> {2, 2}
                                                                      -> view -> {2, 2}
*/
TEST_F(TestSplitReshapePass, TestUpdateForBeCovered)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> shape2 = {kNumTwo, kNumFour};
    std::vector<int64_t> shape3 = {kNumTwo, kNumTwo};
    std::vector<int64_t> offset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> offset2 = {kNumZero, kNumZero};
    std::vector<int64_t> view_offset1 = {kNumZero, kNumZero};
    std::vector<int64_t> view_offset2 = {kNumZero, kNumTwo};

    std::shared_ptr<RawTensor> RawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto input = npu::tile_fwk::IRBuilder().CreateTensorVar(RawTensor1, offset1, shape1,
                                                            CreateTestConstIntVector(shape1));
    input->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    std::shared_ptr<RawTensor> RawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape2);
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(RawTensor2, offset2, shape2,
                                                                CreateTestConstIntVector(shape2));
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto output1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape3, CreateTestConstIntVector(shape3));
    output1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape3, CreateTestConstIntVector(shape3));
    output2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    auto& assemble_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {input}, {ubTensor1});
    auto assemble_Attr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assemble_op.SetOpAttribute(assemble_Attr);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    auto& view_op1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {ubTensor2}, {output1});
    auto view_Attr1 = std::make_shared<ViewOpAttribute>(view_offset1);
    view_op1.SetOpAttribute(view_Attr1);
    auto& view_op2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {ubTensor2}, {output2});
    auto view_Attr2 = std::make_shared<ViewOpAttribute>(view_offset2);
    view_op2.SetOpAttribute(view_Attr2);

    CalcOverlapPara para;
    std::vector<SymbolicScalar> validShape;

    SplitReshape pass;
    std::vector<int64_t> newCopyOutTileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> newCopyOutTileShape = {kNumTwo, kNumTwo, kNumTwo};
    para.alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    para.overlaps = {input};
    para.newOverlaps = {npu::tile_fwk::IRBuilder().CreateTensorVar(input->tensor, newCopyOutTileOffset,
                                                                   newCopyOutTileShape, validShape)};
    para.reshapeSource = ubTensor1;
    para.input = ubTensor2;
    para.output = output1;
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);

    std::vector<int64_t> viewOffset = {kNumZero, kNumZero};
    auto inputView = npu::tile_fwk::IRBuilder().CreateTensorVar(ubTensor2->tensor, viewOffset, shape2,
                                                                CreateTestConstIntVector(shape2));
    para.inputView = inputView;
    para.newInputViewTileShape = {kNumTwo, kNumOne, kNumTwo};
    para.newInputViewTileOffset = {kNumZero, kNumZero, kNumZero};
    EXPECT_EQ(pass.ProcessOnetoMulti(*currFunctionPtr, view_op1, para), SUCCESS);
    std::vector<int64_t> view2Offset = {kNumZero, kNumTwo};
    para.newInputViewTileShape = {kNumTwo, kNumOne, kNumTwo};
    para.newInputViewTileOffset = {kNumZero, kNumOne, kNumZero};
    EXPECT_EQ(pass.ProcessOnetoMulti(*currFunctionPtr, view_op2, para), SUCCESS);

    EXPECT_EQ(pass.reshapes_.size(), kSizeOne);
    auto newReshape = pass.reshapes_.begin()->second;
    auto newReshapeResource = newReshape->input;
    EXPECT_NE(newReshape->output, ubTensor2);
    EXPECT_EQ(newReshape->output->GetMemoryTypeOriginal(), MemoryType::MEM_UNKNOWN);
    auto viewOpAttribute1 = dynamic_cast<ViewOpAttribute*>(view_op1.GetOpAttribute().get());
    EXPECT_EQ(viewOpAttribute1->GetFromOffset(), view_offset1);
    auto viewOpAttribute2 = dynamic_cast<ViewOpAttribute*>(view_op2.GetOpAttribute().get());
    EXPECT_EQ(viewOpAttribute2->GetFromOffset(), view_offset2);
    EXPECT_EQ(pass.assembles_.size(), kSizeOne);
    auto newAssemble = pass.assembles_.begin();
    EXPECT_EQ(newAssemble->from, MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(newAssemble->toOffset, offset1);
    EXPECT_EQ(newAssemble->input, input);
    EXPECT_EQ(newAssemble->output, newReshapeResource);
}

/*
验证一对多场景下动态shape数据的处理
rawShape = {2, 2, 2}
{2, 2, 2}(ddr) -> assemble -> {2, 2, 2} -> reshape -> {2, 4}(unknown) -> view -> {2, 2}(ddr)
                                                                      -> view -> {2, 2}(ddr)
                                            {a, 4}                                                                   {a,
2}/{b, 0} {2, 2, 2}(ddr) -> assemble -> {2, 2, 2} -> reshape -> {2, 4}(unknown) -> view -> {2, 2}
                                                                      -> view -> {2, 2}
*/
TEST_F(TestSplitReshapePass, TestDynUpdateForBeCovered)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> shape2 = {kNumTwo, kNumFour};
    std::vector<int64_t> shape3 = {kNumTwo, kNumTwo};
    std::vector<int64_t> offset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> offset2 = {kNumZero, kNumZero};
    std::vector<int64_t> view_offset1 = {kNumZero, kNumZero};
    std::vector<int64_t> view_offset2 = {kNumZero, kNumTwo};
    std::vector<SymbolicScalar> validShape = {CreateTestScalarVar("a"), kNumFour};

    std::shared_ptr<RawTensor> RawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto input = npu::tile_fwk::IRBuilder().CreateTensorVar(RawTensor1, offset1, shape1,
                                                            CreateTestConstIntVector(shape1));
    input->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    std::shared_ptr<RawTensor> RawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape2);
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(RawTensor2, offset2, shape2,
                                                                CreateTestConstIntVector(shape2));
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto output1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape3, CreateTestConstIntVector(shape3));
    output1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape3, CreateTestConstIntVector(shape3));
    output2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    auto& assemble_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {input}, {ubTensor1});
    auto assemble_Attr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assemble_op.SetOpAttribute(assemble_Attr);
    auto& reshape_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    reshape_op.SetAttribute(OP_ATTR_VALID_SHAPE, validShape);
    auto& view_op1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {ubTensor2}, {output1});
    auto view_Attr1 = std::make_shared<ViewOpAttribute>(view_offset1);
    view_op1.SetOpAttribute(view_Attr1);
    auto& view_op2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {ubTensor2}, {output2});
    auto view_Attr2 = std::make_shared<ViewOpAttribute>(view_offset2);
    view_op2.SetOpAttribute(view_Attr2);

    CalcOverlapPara para;
    SplitReshape pass;
    std::vector<int64_t> newCopyOutTileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> newCopyOutTileShape = {kNumTwo, kNumTwo, kNumTwo};
    para.alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    para.overlaps = {input};
    auto newOverlap = npu::tile_fwk::IRBuilder().CreateTensorVar(
        input->tensor, newCopyOutTileOffset, newCopyOutTileShape, CreateTestConstIntVector(newCopyOutTileShape));
    para.newOverlaps = {newOverlap};
    para.reshapeSource = ubTensor1;
    para.input = ubTensor2;
    para.output = output1;
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);

    std::vector<int64_t> viewOffset = {kNumZero, kNumZero};
    auto inputView = npu::tile_fwk::IRBuilder().CreateTensorVar(ubTensor2->tensor, viewOffset, shape2,
                                                                CreateTestConstIntVector(shape2));
    para.inputView = inputView;
    para.newInputViewTileShape = {kNumTwo, kNumOne, kNumTwo};
    para.newInputViewTileOffset = {kNumZero, kNumZero, kNumZero};
    auto newValidShape1 = GetViewValidShape(validShape, view_offset1, {}, shape3);
    para.oriViewDynShape = newValidShape1;
    EXPECT_EQ(pass.ProcessOnetoMulti(*currFunctionPtr, view_op1, para), SUCCESS);
    std::vector<int64_t> view2Offset = {kNumZero, kNumTwo};
    para.newInputViewTileShape = {kNumTwo, kNumOne, kNumTwo};
    para.newInputViewTileOffset = {kNumZero, kNumOne, kNumZero};
    auto newValidShape2 = GetViewValidShape(validShape, view_offset1, {}, shape3);
    para.oriViewDynShape = newValidShape2;
    EXPECT_EQ(pass.ProcessOnetoMulti(*currFunctionPtr, view_op2, para), SUCCESS);

    std::vector<SymbolicScalar> expectShape = {CreateTestScalarVar("a") * 1, kNumFour};
    EXPECT_EQ(pass.reshapes_.size(), kSizeOne);
    auto newReshape = pass.reshapes_.begin()->second;
    auto newReshapeResource = newReshape->input;
    EXPECT_NE(newReshape->output, ubTensor2);
    auto reshapeOutput = newReshape->output;
    EXPECT_EQ(newReshape->dynValidShapes.size(), kSizeTwo);
    EXPECT_EQ(newReshape->dynValidShapes[0].size(), kSizeTwo);
    EXPECT_EQ(newReshape->dynValidShapes[1].size(), kSizeTwo);
    std::vector<std::string> expectValidShape1 = {
        "(RUNTIME_Min(RUNTIME_Max(a, 0), 2)*(RUNTIME_Min(RUNTIME_Max(a, 0), 2)!=0))", "4"};
    for (size_t i = 0; i < kSizeTwo; ++i) {
        EXPECT_EQ(newReshape->dynValidShapes[0][i].Dump(), expectValidShape1[i]);
    }
    std::vector<std::string> expectValidShape2 = {
        "(RUNTIME_Min(RUNTIME_Max(a, 0), 2)*(RUNTIME_Min(RUNTIME_Max(a, 0), 2)!=0))", "4"};
    for (size_t i = 0; i < kSizeTwo; ++i) {
        EXPECT_EQ(newReshape->dynValidShapes[1][i].Dump(), expectValidShape2[i]);
    }
    EXPECT_EQ(reshapeOutput->GetMemoryTypeOriginal(), MemoryType::MEM_UNKNOWN);
    EXPECT_EQ(pass.assembles_.size(), kSizeOne);
    auto newAssemble = pass.assembles_.begin();
    EXPECT_EQ(newAssemble->from, MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(newAssemble->toOffset, offset1);
    EXPECT_EQ(newAssemble->input, input);
    EXPECT_EQ(newAssemble->output, newReshapeResource);
    auto viewOpAttribute1 = dynamic_cast<ViewOpAttribute*>(view_op1.GetOpAttribute().get());
    EXPECT_EQ(viewOpAttribute1->GetFromOffset(), view_offset1);
    auto viewOpAttribute2 = dynamic_cast<ViewOpAttribute*>(view_op2.GetOpAttribute().get());
    EXPECT_EQ(viewOpAttribute2->GetFromOffset(), view_offset2);
}

static void BuildBasePerfectlyMatchGraph(std::shared_ptr<Function>& currFunctionPtr, LogicalTensorPtr& input1Out,
                                         LogicalTensorPtr& input2Out, LogicalTensorPtr& ubTensor1Out,
                                         LogicalTensorPtr& ubTensor2Out, LogicalTensorPtr& outputOut,
                                         Operation*& viewOpOut, std::vector<int64_t>& viewOffsetOut)
{
    std::vector<int64_t> shape1 = {kNumTwo, kNumFour};
    std::vector<int64_t> shape2 = {kNumTwo, kNumTwo};
    std::vector<int64_t> shape3 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> offset1 = {kNumZero, kNumZero};
    std::vector<int64_t> offset2 = {kNumZero, kNumTwo};
    viewOffsetOut = {kNumZero, kNumZero, kNumZero};

    std::shared_ptr<RawTensor> RawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    input1Out = npu::tile_fwk::IRBuilder().CreateTensorVar(RawTensor1, offset1, shape2,
                                                           CreateTestConstIntVector(shape2));
    input1Out->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    input2Out = npu::tile_fwk::IRBuilder().CreateTensorVar(RawTensor1, offset2, shape2,
                                                           CreateTestConstIntVector(shape2));
    input2Out->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    ubTensor1Out = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    ubTensor1Out->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    ubTensor2Out = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape3, CreateTestConstIntVector(shape3));
    ubTensor2Out->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    outputOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape3, CreateTestConstIntVector(shape3));
    outputOut->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    auto& assemble_op1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {input1Out},
                                                          {ubTensor1Out});
    auto assemble_Attr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assemble_op1.SetOpAttribute(assemble_Attr1);
    auto& assemble_op2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {input2Out},
                                                          {ubTensor1Out});
    auto assemble_Attr2 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset2);
    assemble_op2.SetOpAttribute(assemble_Attr2);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {ubTensor1Out}, {ubTensor2Out});
    auto& view_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {ubTensor2Out}, {outputOut});
    auto view_Attr = std::make_shared<ViewOpAttribute>(viewOffsetOut);
    view_op.SetOpAttribute(view_Attr);
    viewOpOut = &view_op;
}

/*
验证多对一场景下数据的处理
rawShape = {2, 4}
{2, 2}(ddr) -> assemble -> {2, 4}(unknown) -> reshape -> {2, 2, 2}(ddr) -> view -> {2, 2, 2}(ddr)
{2, 2}(ddr) -> assemble ->
{2, 2}(ddr) -> {2, 4}(unknown) -> reshape -> {2, 2, 2}(ddr) -> view -> {2, 2}
{2, 2}(ddr) ->
*/
TEST_F(TestSplitReshapePass, TestUpdateForPerfectlyMatchWithAll)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    LogicalTensorPtr input1, input2, ubTensor1, ubTensor2, output;
    Operation* viewOpPtr = nullptr;
    std::vector<int64_t> view_offset;
    BuildBasePerfectlyMatchGraph(currFunctionPtr, input1, input2, ubTensor1, ubTensor2, output, viewOpPtr, view_offset);
    auto& view_op = *viewOpPtr;

    CalcOverlapPara para;
    std::vector<SymbolicScalar> validShape;
    std::vector<int64_t> shape3 = {kNumTwo, kNumTwo, kNumTwo};

    SplitReshape pass;
    para.alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    para.overlaps = {input1, input2};
    std::vector<int64_t> newInput1TileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> newInput1TileShape = {kNumTwo, kNumOne, kNumTwo};
    auto newInput1 = npu::tile_fwk::IRBuilder().CreateTensorVar(input1->tensor, newInput1TileOffset, newInput1TileShape,
                                                                validShape);
    std::vector<int64_t> newInput2TileOffset = {kNumZero, kNumOne, kNumZero};
    std::vector<int64_t> newInput2TileShape = {kNumTwo, kNumOne, kNumTwo};
    auto newInput2 = npu::tile_fwk::IRBuilder().CreateTensorVar(input2->tensor, newInput2TileOffset, newInput2TileShape,
                                                                validShape);
    para.newOverlaps = {newInput1, newInput2};
    para.reshapeSource = ubTensor1;
    para.input = ubTensor2;
    para.output = output;
    para.newInputViewTileShape = {kNumTwo, kNumTwo, kNumTwo};
    para.newInputViewTileOffset = {kNumZero, kNumZero, kNumZero};
    auto inputView = npu::tile_fwk::IRBuilder().CreateTensorVar(ubTensor2->tensor, view_offset, shape3,
                                                                CreateTestConstIntVector(shape3));
    para.inputView = inputView;
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.ProcessMultitoOne(*currFunctionPtr, view_op, para), SUCCESS);
    EXPECT_EQ(pass.reshapes_.size(), kSizeOne);
    auto reshape = pass.reshapes_.begin()->second;
    auto newReshapeSource = reshape->input;
    auto newReshapeOutput = view_op.GetInputOperand(kSizeZero);
    EXPECT_EQ(reshape->output, newReshapeOutput);
    EXPECT_EQ(newReshapeSource->GetMemoryTypeOriginal(), MemoryType::MEM_UNKNOWN);
    EXPECT_EQ(newReshapeOutput->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(view_op.GetOpAttribute().get());
    EXPECT_EQ(viewOpAttribute->GetFromOffset(), inputView->offset);
}

/*
验证多对一场景下动态shape数据的处理
rawShape = {2, 4}
{2, 2}(ddr) -> assemble -> {2, 4}(unknown) -> reshape -> {2, 2, 2}(ddr) -> view -> {2, 2, 2}(ddr)
{2, 2}(ddr) -> assemble ->
                                             {a, 2, 2}
{2, 2}(ddr) -> {2, 4}(unknown) -> reshape -> {2, 2, 2}(ddr) -> view -> {2, 2}
{2, 2}(ddr) ->
*/
static void BuildDynUpdatePerfectlyMatchGraph(std::shared_ptr<Function>& currFunctionPtr, Operation*& viewOpOut,
                                              CalcOverlapPara& para, LogicalTensorPtr& inputViewOut,
                                              std::vector<SymbolicScalar>& validShapeOut)
{
    LogicalTensorPtr input1, input2, ubTensor1, ubTensor2, output;
    std::vector<int64_t> view_offset;
    BuildBasePerfectlyMatchGraph(currFunctionPtr, input1, input2, ubTensor1, ubTensor2, output, viewOpOut, view_offset);
    validShapeOut = {CreateTestScalarVar("a"), kNumTwo, kNumTwo};
    auto reshapeOp = *ubTensor2->GetProducers().begin();
    reshapeOp->SetAttribute(OP_ATTR_VALID_SHAPE, validShapeOut);
    std::vector<int64_t> shape3 = {kNumTwo, kNumTwo, kNumTwo};

    para.alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    para.overlaps = {input1, input2};
    std::vector<int64_t> newInput1TileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> newInput1TileShape = {kNumTwo, kNumOne, kNumTwo};
    auto newInput1 = npu::tile_fwk::IRBuilder().CreateTensorVar(input1->tensor, newInput1TileOffset, newInput1TileShape,
                                                                CreateTestConstIntVector(newInput1TileShape));
    std::vector<int64_t> newInput2TileOffset = {kNumZero, kNumOne, kNumZero};
    std::vector<int64_t> newInput2TileShape = {kNumTwo, kNumOne, kNumTwo};
    auto newInput2 = npu::tile_fwk::IRBuilder().CreateTensorVar(input2->tensor, newInput2TileOffset, newInput2TileShape,
                                                                CreateTestConstIntVector(newInput2TileShape));
    para.newOverlaps = {newInput1, newInput2};
    para.reshapeSource = ubTensor1;
    para.input = ubTensor2;
    para.output = output;
    para.newInputViewTileShape = {kNumTwo, kNumTwo, kNumTwo};
    para.newInputViewTileOffset = {kNumZero, kNumZero, kNumZero};
    para.oriViewDynShape = GetViewValidShape(validShapeOut, view_offset, {}, shape3);
    inputViewOut = npu::tile_fwk::IRBuilder().CreateTensorVar(ubTensor2->tensor, view_offset, shape3,
                                                              CreateTestConstIntVector(shape3));
    para.inputView = inputViewOut;
}

TEST_F(TestSplitReshapePass, TestDynUpdateForPerfectlyMatchWithAll)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    Operation* viewOpPtr = nullptr;
    CalcOverlapPara para;
    LogicalTensorPtr inputView;
    std::vector<SymbolicScalar> validShape;
    BuildDynUpdatePerfectlyMatchGraph(currFunctionPtr, viewOpPtr, para, inputView, validShape);
    auto& view_op = *viewOpPtr;

    SplitReshape pass;
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.ProcessMultitoOne(*currFunctionPtr, view_op, para), SUCCESS);
    EXPECT_EQ(pass.reshapes_.size(), kSizeOne);
    auto reshape = pass.reshapes_.begin()->second;
    EXPECT_EQ(reshape->dynValidShapes.size(), kNumOne);
    EXPECT_EQ(reshape->dynValidShapes[0].size(), kNumThree);
    std::vector<std::string> expectValidShape = {
        "(RUNTIME_Min(RUNTIME_Max(a, 0), 2)*(RUNTIME_Min(RUNTIME_Max(a, 0), 2)!=0))", "2", "2"};
    for (size_t i = 0; i < kNumThree; ++i) {
        EXPECT_EQ(reshape->dynValidShapes[0][i].Dump(), expectValidShape[i]);
    }
    auto newReshapeSource = reshape->input;
    auto newReshapeOutput = view_op.GetInputOperand(kSizeZero);
    EXPECT_EQ(reshape->output, newReshapeOutput);
    EXPECT_EQ(newReshapeSource->GetMemoryTypeOriginal(), MemoryType::MEM_UNKNOWN);
    EXPECT_EQ(newReshapeOutput->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(view_op.GetOpAttribute().get());
    EXPECT_EQ(viewOpAttribute->GetFromOffset(), inputView->offset);
}

TEST_F(TestSplitReshapePass, TestDynFirstAxisMergeForPerfectlyMatchWithAll)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    ASSERT_NE(func, nullptr);

    const std::vector<int64_t> reshapeInputShape = {kNumTwo, kNumFour, kNumFour};
    const std::vector<int64_t> reshapeOutputShape = {kNumEight, kNumFour};
    const std::vector<int64_t> overlapShape = {kNumOne, kNumTwo, kNumFour};
    const std::vector<int64_t> localInputShape = {kNumOne, kNumFour, kNumFour};
    const std::vector<int64_t> localOutputShape = {kNumFour, kNumFour};
    const std::vector<int64_t> firstOffset = {kNumZero, kNumZero, kNumZero};
    const std::vector<int64_t> secondOffset = {kNumZero, kNumTwo, kNumZero};
    const std::vector<int64_t> outputOffset = {kNumZero, kNumZero};
    const auto globalBatch = CreateTestScalarVar("global_b");
    const auto tailK = CreateTestScalarVar("tail_k");

    auto inputRaw = std::make_shared<RawTensor>(DT_FP32, reshapeInputShape);
    auto input0ValidShape = std::vector<SymbolicScalar>{kNumOne, std::min(tailK, SymbolicScalar(kNumTwo)), kNumFour};
    auto input1ValidShape = std::vector<SymbolicScalar>{
        kNumOne, std::min(std::max(tailK - kNumTwo, SymbolicScalar(kNumZero)), SymbolicScalar(kNumTwo)), kNumFour};
    auto input0 = IRBuilder().CreateTensorVar(inputRaw, firstOffset, overlapShape, input0ValidShape);
    auto input1 = IRBuilder().CreateTensorVar(inputRaw, secondOffset, overlapShape, input1ValidShape);
    input0->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    input1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    auto reshapeInput = IRBuilder().CreateTensorVar(DT_FP32, reshapeInputShape,
                                                    std::vector<SymbolicScalar>{globalBatch, tailK, kNumFour});
    reshapeInput->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto reshapeValidShape = std::vector<SymbolicScalar>{globalBatch * tailK, kNumFour};
    auto reshapeOutput = IRBuilder().CreateTensorVar(DT_FP32, reshapeOutputShape, reshapeValidShape);
    reshapeOutput->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto viewOutput = IRBuilder().CreateTensorVar(DT_FP32, localOutputShape, reshapeValidShape);
    viewOutput->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    auto& assemble0 = PassOperationUtils::AddOperation(*func, Opcode::OP_ASSEMBLE, {input0}, {reshapeInput});
    assemble0.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, firstOffset));
    auto& assemble1 = PassOperationUtils::AddOperation(*func, Opcode::OP_ASSEMBLE, {input1}, {reshapeInput});
    assemble1.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, secondOffset));
    auto& reshape = PassOperationUtils::AddOperation(*func, Opcode::OP_RESHAPE, {reshapeInput}, {reshapeOutput});
    reshape.SetAttribute(OP_ATTR_VALID_SHAPE, reshapeValidShape);
    auto& view = PassOperationUtils::AddOperation(*func, Opcode::OP_VIEW, {reshapeOutput}, {viewOutput});
    view.SetOpAttribute(std::make_shared<ViewOpAttribute>(outputOffset));

    auto inputView = IRBuilder().CreateTensorVar(reshapeOutput->GetRawTensor(), outputOffset, localOutputShape,
                                                 reshapeValidShape);
    CalcOverlapPara para;
    para.overlaps = {input0, input1};
    para.reshapeSource = reshapeInput;
    para.input = reshapeOutput;
    para.output = viewOutput;
    para.inputView = inputView;
    ReshapeSourcePara sourcePara = {localInputShape, firstOffset};

    SplitReshape pass;
    ASSERT_EQ(pass.CollectCopyOut(*func), SUCCESS);
    ASSERT_EQ(pass.UpdateForPerfectlyMatchWithAll(*func, view, para, sourcePara), SUCCESS);
    ASSERT_EQ(pass.reshapes_.size(), kSizeOne);
    const auto splitReshape = pass.reshapes_.begin()->second;
    ASSERT_EQ(splitReshape->dynValidShapes.size(), kSizeOne);
    ASSERT_EQ(splitReshape->dynValidShapes[0].size(), kSizeTwo);
    EXPECT_NE(splitReshape->dynValidShapes[0][0].Dump().find("tail_k"), std::string::npos);
    EXPECT_EQ(splitReshape->dynValidShapes[0][0].Dump().find("global_b"), std::string::npos);
    EXPECT_EQ(splitReshape->dynValidShapes[0][1].Dump(), std::to_string(kNumFour));
}

void RunPassStra(Function& func, const PassName passName)
{
    std::string passNameStr = PassNameStr(passName);
    std::string strategyName = passNameStr + "Strategy";
    PassManager& passManager = PassManager::Instance();
    passManager.RegisterStrategy(strategyName, {
                                                   {passNameStr, passName},
                                               });
    EXPECT_EQ(passManager.RunPass(Program::GetInstance(), func, strategyName), SUCCESS);
}

struct CheckReshapeStruct {
    const std::vector<int64_t> reshapeInputShape;
    const int reshapeInputProducerSize;
    const bool checkProducer;
    const std::vector<int64_t> reshapeInputOperandShape;
    const std::vector<int64_t> reshapeOutputShape;
    const int reshapeOutputProducerSize;
    const bool checkConsumer;
    const std::vector<int64_t> reshapeOutputOperandShape;
    const uint32_t reshapeOpNum;
};

void CheckOpReshape(Function* func, CheckReshapeStruct expectReshape)
{
    int reshapeOp = 0;
    for (auto& op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            EXPECT_EQ(op.GetInputOperandSize(), kSizeOne);
            auto reshapeInput = op.GetInputOperand(kSizeZero);
            EXPECT_NE(reshapeInput, nullptr);
            EXPECT_EQ(reshapeInput->shape, expectReshape.reshapeInputShape);
            EXPECT_EQ(reshapeInput->GetProducers().size(), expectReshape.reshapeInputProducerSize);
            for (const auto& producer : reshapeInput->GetProducers()) {
                EXPECT_EQ(producer->GetOpcode(), Opcode::OP_CONTRACT);
                if (expectReshape.checkProducer) {
                    EXPECT_EQ(producer->GetInputOperandSize(), kSizeOne);
                    EXPECT_EQ(producer->GetInputOperand(kSizeZero)->shape, expectReshape.reshapeInputOperandShape);
                }
            }
            EXPECT_EQ(op.GetOutputOperandSize(), kSizeOne);
            auto reshapeOutput = op.GetOutputOperand(kSizeZero);
            EXPECT_NE(reshapeOutput, nullptr);
            EXPECT_EQ(reshapeOutput->shape, expectReshape.reshapeOutputShape);
            EXPECT_EQ(reshapeOutput->GetConsumers().size(), expectReshape.reshapeOutputProducerSize);
            for (const auto& consumer : reshapeOutput->GetConsumers()) {
                EXPECT_EQ(consumer->GetOpcode(), Opcode::OP_SLICE);
                if (expectReshape.checkConsumer) {
                    EXPECT_EQ(consumer->GetOutputOperandSize(), kSizeOne);
                    EXPECT_EQ(consumer->GetOutputOperand(kSizeZero)->shape, expectReshape.reshapeOutputOperandShape);
                }
            }
            reshapeOp++;
        }
    }
    EXPECT_EQ(reshapeOp, expectReshape.reshapeOpNum);
}

/*
校验一对一场景(使用expandfunction作为前序pass)
1) 用例设置：
{2,2,4} -> exp -> {2,2,4} -> reshape -> {2,2,1,4} -> exp -> {2,2,1,4}
tileshape = {2,2,2,2}
2) expandfunction：
exp -> {2,2,2} -> assemble -> reshape -> {2,2,1,4} -> view -> {2,2,1,2} -> exp -> {2,2,1,2}
exp -> {2,2,2} -> assemble                         -> view -> {2,2,1,2} -> exp -> {2,2,1,2}
3) splitreshape
exp -> {2,2,2} -> assemble -> reshape -> {2,2,1,2} -> view -> {2,2,1,2} -> exp -> {2,2,1,2}
exp -> {2,2,2} -> assemble -> reshape -> {2,2,1,2} -> view -> {2,2,1,2} -> exp -> {2,2,1,2}
*/
TEST_F(TestSplitReshapePass, TestPerfectlyMatchedSTest)
{
    // Define the shape of the Tensors
    std::vector<int64_t> origShape = {kNumTwo, kNumTwo, kNumFour};
    std::vector<int64_t> reshapeShape = {kNumTwo, kNumTwo, kNumOne, kNumFour};
    std::vector<int64_t> tiledShape = {kNumTwo, kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> tiledorigShape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> tiledreshapeShape = {kNumTwo, kNumTwo, kNumOne, kNumTwo};

    TileShape::Current().SetVecTile(tiledShape);
    Tensor input(DT_FP32, origShape, "input");
    Tensor output(DT_FP32, reshapeShape, "output");

    FUNCTION("STCase1")
    {
        Tensor exp = Exp(input);
        Tensor reshape = Reshape(exp, reshapeShape);
        output = Exp(reshape);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase1");

    RunPassStra(*func, PassName::EXPAND_FUNCTION);
    CheckOpReshape(func,
                   CheckReshapeStruct{origShape, kSizeTwo, false, {}, reshapeShape, kSizeTwo, false, {}, kNumOne});

    RunPassStra(*func, PassName::SPLIT_RESHAPE);
    CheckOpReshape(
        func, CheckReshapeStruct{tiledorigShape, kSizeOne, false, {}, tiledreshapeShape, kSizeOne, false, {}, kNumTwo});
}

/*
校验一对多场景(使用expandfunction作为前序pass)
1) 用例设置：
{4,2,2} -> exp -> {4,2,2} -> reshape -> {4,4} -> exp -> {4,4}
tileshape = {2,2,2}
2) expandfunction：
exp -> {2,2,2} -> assemble -> reshape -> {4,4} -> view -> {2,2} -> exp -> {2,2}
exp -> {2,2,2} -> assemble                     -> view -> {2,2} -> exp -> {2,2}
                                               -> view -> {2,2} -> exp -> {2,2}
                                               -> view -> {2,2} -> exp -> {2,2}
3) splitreshape
                                               -> view -> {2,2} -> exp -> {2,2}
exp -> {2,2,2} -> assemble -> reshape -> {2,4} -> view -> {2,2} -> exp -> {2,2}
exp -> {2,2,2} -> assemble -> reshape -> {2,4} -> view -> {2,2} -> exp -> {2,2}
                                               -> view -> {2,2} -> exp -> {2,2}
*/
TEST_F(TestSplitReshapePass, TestBeCoveredSTest)
{
    // Define the shape of the Tensors
    std::vector<int64_t> origShape = {kNumFour, kNumTwo, kNumTwo};
    std::vector<int64_t> reshapeShape = {kNumFour, kNumFour};
    std::vector<int64_t> tiledShape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> tiledorigShape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> tiledreshapeShape = {kNumTwo, kNumFour};
    std::vector<int64_t> tiledviewShape = {kNumTwo, kNumTwo};

    TileShape::Current().SetVecTile(tiledShape);
    Tensor input(DT_FP32, origShape, "input");
    Tensor output(DT_FP32, reshapeShape, "output");

    FUNCTION("STCase2")
    {
        Tensor exp = Exp(input);
        Tensor reshape = Reshape(exp, reshapeShape);
        output = Exp(reshape);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase2");

    RunPassStra(*func, PassName::EXPAND_FUNCTION);
    CheckOpReshape(func,
                   CheckReshapeStruct{origShape, kSizeTwo, false, {}, reshapeShape, kSizeFour, false, {}, kNumOne});

    RunPassStra(*func, PassName::SPLIT_RESHAPE);
    CheckOpReshape(
        func, CheckReshapeStruct{
                  tiledorigShape, kSizeOne, false, {}, tiledreshapeShape, kSizeTwo, true, tiledviewShape, kNumTwo});
}

/*
校验多对一场景(使用expandfunction作为前序pass)
1) 用例设置：
{2,4,4} -> exp -> {2,4,4} -> reshape -> {2,4,2,2} -> exp -> {2,4,2,2}
tileshape = {2,2,2,2}
2) expandfunction：
exp -> {2,2,2} -> assemble -> reshape -> {2,4,2,2} -> view -> {2,2,2,2} -> exp -> {2,2,2,2}
exp -> {2,2,2} -> assemble                         -> view -> {2,2,2,2} -> exp -> {2,2,2,2}
exp -> {2,2,2} -> assemble
exp -> {2,2,2} -> assemble
3) splitreshape
exp -> {2,2,2} -> assemble ->
exp -> {2,2,2} -> assemble -> reshape -> {2,2,2,2} -> view -> {2,2,2,2} -> exp -> {2,2,2,2}
exp -> {2,2,2} -> assemble -> reshape -> {2,2,2,2} -> view -> {2,2,2,2} -> exp -> {2,2,2,2}
exp -> {2,2,2} -> assemble ->
*/
TEST_F(TestSplitReshapePass, TestPerfectlyMatchedWithallSTest)
{
    // Define the shape of the Tensors
    std::vector<int64_t> origShape = {kNumTwo, kNumFour, kNumFour};
    std::vector<int64_t> reshapeShape = {kNumTwo, kNumFour, kNumTwo, kNumTwo};
    std::vector<int64_t> tiledShape = {kNumTwo, kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> tiledassembleShape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> tiledreshapeShape = {kNumTwo, kNumTwo, kNumFour};
    std::vector<int64_t> tiledviewShape = {kNumTwo, kNumTwo, kNumTwo, kNumTwo};

    TileShape::Current().SetVecTile(tiledShape);
    Tensor input(DT_FP32, origShape, "input");
    Tensor output(DT_FP32, reshapeShape, "output");

    FUNCTION("STCase3")
    {
        Tensor exp = Exp(input);
        Tensor reshape = Reshape(exp, reshapeShape);
        output = Exp(reshape);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase3");

    RunPassStra(*func, PassName::EXPAND_FUNCTION);
    CheckOpReshape(func,
                   CheckReshapeStruct{origShape, kSizeFour, false, {}, reshapeShape, kSizeTwo, false, {}, kNumOne});

    RunPassStra(*func, PassName::SPLIT_RESHAPE);
    CheckOpReshape(func, CheckReshapeStruct{tiledreshapeShape, kSizeTwo, true, tiledassembleShape, tiledviewShape,
                                            kSizeOne, true, tiledviewShape, kNumTwo});
}

void CollectOperations(std::shared_ptr<Function> func, std::unordered_map<LogicalTensorPtr, int>& inputsWeight,
                       std::unordered_map<LogicalTensorPtr, Operation*>& newAssembles, const uint32_t expectReshapeOp,
                       const uint32_t expectViewOp, int expectAssembleOp)
{
    int reshapeOp = 0;
    int assembleOp = 0;
    int viewOp = 0;

    for (auto& op : func->Operations().DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_RESHAPE) {
            reshapeOp++;
        } else if (op->GetOpcode() == Opcode::OP_ASSEMBLE) {
            for (auto [input, weight] : inputsWeight) {
                if (op->GetInputOperand(kSizeZero) == input) {
                    newAssembles[input] = op;
                    assembleOp += weight;
                }
            }
        } else if (op->GetOpcode() == Opcode::OP_VIEW) {
            viewOp++;
        }
    }
    EXPECT_EQ(reshapeOp, expectReshapeOp);
    EXPECT_EQ(viewOp, expectViewOp);
    EXPECT_EQ(assembleOp, expectAssembleOp);
}

void CheckNewAssembles(std::unordered_map<LogicalTensorPtr, Operation*>& newAssembles,
                       std::unordered_map<LogicalTensorPtr, std::vector<int64_t>>& expectAssembleOffset,
                       std::vector<std::string>& expectAssembleDynShape,
                       std::unordered_map<LogicalTensorPtr, std::vector<std::string>>& expectValidShapes,
                       std::vector<SymbolicScalar>& dynInputShape, LogicalTensors& reshapeOutputs,
                       const uint32_t reshapeOutputSize = kNumFour)
{
    for (auto [input, newAssemble] : newAssembles) {
        EXPECT_NE(newAssemble, nullptr);
        auto assembleDynValidShape = dynamic_cast<AssembleOpAttribute*>(newAssemble->GetOpAttribute().get())
                                         ->GetFromDynValidShape();
        EXPECT_EQ(assembleDynValidShape.size(), kNumThree);
        for (size_t i = 0; i < kNumThree; ++i) {
            EXPECT_EQ(assembleDynValidShape[i].Dump(), dynInputShape[i].Dump());
        }
        auto assembleOpAttribute = dynamic_cast<AssembleOpAttribute*>(newAssemble->GetOpAttribute().get());
        EXPECT_EQ(assembleOpAttribute->GetToOffset(), expectAssembleOffset[input]);
        auto reshapeSource = newAssemble->GetOutputOperand(kSizeZero);
        std::vector<SymbolicScalar> assembleDynOutput = reshapeSource->GetDynValidShape();
        EXPECT_EQ(assembleDynOutput.size(), kNumThree);
        for (size_t i = 0; i < kNumThree; ++i) {
            EXPECT_EQ(assembleDynOutput[i].Dump(), expectAssembleDynShape[i]);
        }
        auto reshape = *(reshapeSource->GetConsumers().begin());
        auto reshapeOutput = reshape->GetOutputOperand(kSizeZero);
        reshapeOutputs.emplace_back(reshapeOutput);
        std::vector<SymbolicScalar> reshapeAttrValidShape;
        std::vector<SymbolicScalar> reshapeDynOutput = reshapeOutput->GetDynValidShape();
        EXPECT_TRUE(reshape->GetAttr(OP_ATTR_VALID_SHAPE, reshapeAttrValidShape));
        EXPECT_EQ(reshapeDynOutput.size(), reshapeOutputSize);
        EXPECT_EQ(reshapeAttrValidShape.size(), reshapeOutputSize);
        for (size_t i = 0; i < reshapeOutputSize; ++i) {
            EXPECT_EQ(reshapeDynOutput[i].Dump(), expectValidShapes[input][i]);
            EXPECT_EQ(reshapeAttrValidShape[i].Dump(), expectValidShapes[input][i]);
        }
    }
}

LogicalTensors BuildDynPerfectlyMatchFunc(std::shared_ptr<Function> func)
{
    std::vector<int64_t> shape1 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> shape2 = {kNumTwo, kNumTwo, kNumFour};
    std::vector<int64_t> shape3 = {kNumTwo, kNumTwo, kNumOne, kNumFour};
    std::vector<int64_t> shape4 = {kNumTwo, kNumTwo, kNumOne, kNumTwo};
    std::vector<int64_t> assembleOffset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> assembleOffset2 = {kNumZero, kNumZero, kNumTwo};
    std::vector<int64_t> viewOffset1 = {kNumZero, kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> viewOffset2 = {kNumZero, kNumZero, kNumZero, kNumTwo};
    std::vector<SymbolicScalar> validShape = {CreateTestScalarVar("a0"), CreateTestScalarVar("a1"), kNumOne, kNumFour};
    std::vector<SymbolicScalar> dynInputShape = {CreateTestScalarVar("a0"), CreateTestScalarVar("a1"), kNumTwo};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape2);
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape3);
    auto input1 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor1, assembleOffset1, shape1, dynInputShape);
    input1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto input2 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor1, assembleOffset2, shape1, dynInputShape);
    input2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape3, CreateTestConstIntVector(shape3));
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output1 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor2, viewOffset1, shape4,
                                                              CreateTestConstIntVector(shape4));
    output1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output2 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor2, viewOffset2, shape4,
                                                              CreateTestConstIntVector(shape4));
    output2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    auto& assemble_op1 = PassOperationUtils::AddOperation(*func, Opcode::OP_ASSEMBLE, {input1}, {ubTensor1});
    auto assemble_Attr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset1);
    assemble_op1.SetOpAttribute(assemble_Attr1);
    auto& assemble_op2 = PassOperationUtils::AddOperation(*func, Opcode::OP_ASSEMBLE, {input2}, {ubTensor1});
    auto assemble_Attr2 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset2);
    assemble_op2.SetOpAttribute(assemble_Attr2);
    auto& reshape_op = PassOperationUtils::AddOperation(*func, Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    reshape_op.SetAttribute(OP_ATTR_VALID_SHAPE, validShape);
    auto& view_op1 = PassOperationUtils::AddOperation(*func, Opcode::OP_VIEW, {ubTensor2}, {output1});
    auto view_Attr1 = std::make_shared<ViewOpAttribute>(viewOffset1);
    view_op1.SetOpAttribute(view_Attr1);
    auto& view_op2 = PassOperationUtils::AddOperation(*func, Opcode::OP_VIEW, {ubTensor2}, {output2});
    auto view_Attr2 = std::make_shared<ViewOpAttribute>(viewOffset2);
    view_op2.SetOpAttribute(view_Attr2);

    func->inCasts_.push_back(input1);
    func->inCasts_.push_back(input2);
    func->outCasts_.push_back(output1);
    func->outCasts_.push_back(output2);
    return {input1, input2};
}

/*
验证一对一场景下动态shape的兜底策略
因为缺乏宏构建策略，手动构造expandfunction的输出构图
1) 用例设置：
                                 {a0,a1,1,4}
{2,2,2} -> assemble -> {2,2,4} -> reshape -> {2,2,1,4} -> view -> {2,2,1,2}
{2,2,2} -> assemble                                    -> view -> {2,2,1,2}
2) splitreshape
                                 {a0,a1,1,2}
{2,2,2} -> assemble -> {2,2,2} -> reshape -> {2,2,1,2} -> view -> {2,2,1,2}
{2,2,2} -> assemble -> {2,2,2} -> reshape -> {2,2,1,2} -> view -> {2,2,1,2}
{a0,a1,2}             {a0,a1,2}              {a0,a1,1,2}
{a0,a1,2}             {a0,a1,2}              {a0,a1,1,2}
*/
TEST_F(TestSplitReshapePass, TestDynPerfectlyMatchSTest)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(func != nullptr);
    std::vector<int64_t> assembleOffset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> assembleOffset2 = {kNumZero, kNumZero, kNumTwo};
    std::vector<SymbolicScalar> dynInputShape = {CreateTestScalarVar("a0"), CreateTestScalarVar("a1"), kNumTwo};

    auto inputs = BuildDynPerfectlyMatchFunc(func);

    RunPassStra(*func, PassName::SPLIT_RESHAPE);

    std::unordered_map<LogicalTensorPtr, int> inputsWeight = {{inputs[0], 1}, {inputs[1], 10}};
    std::unordered_map<LogicalTensorPtr, Operation*> newAssembles = {{inputs[0], nullptr}, {inputs[1], nullptr}};
    CollectOperations(func, inputsWeight, newAssembles, kNumTwo, kNumTwo, 11);

    std::unordered_map<LogicalTensorPtr, std::vector<int64_t>> expectAssembleOffset = {{inputs[0], assembleOffset1},
                                                                                       {inputs[1], assembleOffset2}};
    LogicalTensors reshapeOutputs;
    std::vector<std::string> expectAssembleDynShape = {"RUNTIME_Max((a0*(a0!=0)), 0)", "RUNTIME_Max((a1*(a1!=0)), 0)",
                                                       "2"};
    std::vector<std::string> expectReshapeDynShape = {
        "RUNTIME_Max(((RUNTIME_Min(RUNTIME_Max(a0, 0), 2)*(RUNTIME_Min(RUNTIME_Max(a0, 0), 2)!=0))-0), "
        "0)",
        "RUNTIME_Max(((RUNTIME_Min(RUNTIME_Max(a1, 0), 2)*(RUNTIME_Min(RUNTIME_Max(a1, 0), 2)!=0))-0), "
        "0)",
        "1", "2"};
    std::unordered_map<LogicalTensorPtr, std::vector<std::string>> expectValidShapes = {
        {inputs[0], expectReshapeDynShape}, {inputs[1], expectReshapeDynShape}};
    CheckNewAssembles(newAssembles, expectAssembleOffset, expectAssembleDynShape, expectValidShapes, dynInputShape,
                      reshapeOutputs, kNumFour);
    EXPECT_NE(reshapeOutputs[0], reshapeOutputs[1]);
    EXPECT_NE(*(reshapeOutputs[0]->GetConsumers().begin()), *(reshapeOutputs[1]->GetConsumers().begin()));
}

LogicalTensors BuildDynBeCoveredFunc(std::shared_ptr<Function> func, const std::vector<SymbolicScalar>& validShape)
{
    std::vector<int64_t> shape1 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> shape2 = {kNumTwo, kNumTwo, kNumFour};
    std::vector<int64_t> shape3 = {kNumFour, kNumFour};
    std::vector<int64_t> shape4 = {kNumTwo, kNumTwo};
    std::vector<int64_t> assembleOffset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> assembleOffset2 = {kNumZero, kNumZero, kNumTwo};
    std::vector<int64_t> viewOffset1 = {kNumZero, kNumZero};
    std::vector<int64_t> viewOffset2 = {kNumZero, kNumTwo};
    std::vector<int64_t> viewOffset3 = {kNumTwo, kNumZero};
    std::vector<int64_t> viewOffset4 = {kNumTwo, kNumTwo};
    std::vector<SymbolicScalar> dynInputShape = {kNumTwo, kNumTwo, CreateTestScalarVar("a")};

    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape3, CreateTestConstIntVector(shape3));
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape2);
    auto input1 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor1, assembleOffset1, shape1, dynInputShape);
    input1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto input2 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor1, assembleOffset2, shape1, dynInputShape);
    input2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape3);
    auto output1 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor2, viewOffset1, shape4,
                                                              CreateTestConstIntVector(shape4));
    output1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output2 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor2, viewOffset2, shape4,
                                                              CreateTestConstIntVector(shape4));
    output2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output3 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor2, viewOffset3, shape4,
                                                              CreateTestConstIntVector(shape4));
    output3->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output4 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor2, viewOffset4, shape4,
                                                              CreateTestConstIntVector(shape4));
    output4->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    auto& assemble_op1 = PassOperationUtils::AddOperation(*func, Opcode::OP_ASSEMBLE, {input1}, {ubTensor1});
    assemble_op1.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset1));
    auto& assemble_op2 = PassOperationUtils::AddOperation(*func, Opcode::OP_ASSEMBLE, {input2}, {ubTensor1});
    assemble_op2.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset2));
    auto& reshape_op = PassOperationUtils::AddOperation(*func, Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    reshape_op.SetAttribute(OP_ATTR_VALID_SHAPE, validShape);
    auto& view_op1 = PassOperationUtils::AddOperation(*func, Opcode::OP_VIEW, {ubTensor2}, {output1});
    view_op1.SetOpAttribute(std::make_shared<ViewOpAttribute>(viewOffset1));
    auto& view_op2 = PassOperationUtils::AddOperation(*func, Opcode::OP_VIEW, {ubTensor2}, {output2});
    view_op2.SetOpAttribute(std::make_shared<ViewOpAttribute>(viewOffset2));
    auto& view_op3 = PassOperationUtils::AddOperation(*func, Opcode::OP_VIEW, {ubTensor2}, {output3});
    view_op3.SetOpAttribute(std::make_shared<ViewOpAttribute>(viewOffset3));
    auto& view_op4 = PassOperationUtils::AddOperation(*func, Opcode::OP_VIEW, {ubTensor2}, {output4});
    view_op4.SetOpAttribute(std::make_shared<ViewOpAttribute>(viewOffset4));

    func->inCasts_ = {input1, input2};
    func->outCasts_ = {output1, output2, output3, output4};
    return {input1, input2};
}

/*
验证一对多场景下动态shape的兜底策略
因为缺乏宏构建策略，手动构造expandfunction的输出构图
1) 用例设置：
{2,2,2} -> assemble -> {2,2,4} -> reshape -> {4,4} -> view -> {2,2}
{2,2,2} -> assemble                                -> view -> {2,2}
                                                   -> view -> {2,2}
                                                   -> view -> {2,2}
{2,2,a}                            {4,a}
2) splitreshape
                                    {4, Max(0, GetViewValidShapeDim(a,0,2))}
                                                   -> view -> {2,2}
{2,2,2} -> assemble -> {2,2,2} -> reshape -> {4,2} -> view -> {2,2}
{2,2,2} -> assemble -> {2,2,2} -> reshape -> {4,2} -> view -> {2,2}
                                                   -> view -> {2,2}
{2,2,a}                {2,2,a}               {4,a}
                   {4, Max(0, GetViewValidShapeDim(a,2,2))}
*/
TEST_F(TestSplitReshapePass, TestDynBeCoveredSTest)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(func != nullptr);

    std::vector<int64_t> assembleOffset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> assembleOffset2 = {kNumZero, kNumZero, kNumTwo};
    std::vector<SymbolicScalar> validShape = {kNumFour, CreateTestScalarVar("a")};
    std::vector<SymbolicScalar> dynInputShape = {kNumTwo, kNumTwo, CreateTestScalarVar("a")};

    auto inputs = BuildDynBeCoveredFunc(func, validShape);
    RunPassStra(*func, PassName::SPLIT_RESHAPE);

    std::unordered_map<LogicalTensorPtr, int> inputsWeight = {{inputs[0], 1}, {inputs[1], 10}};
    std::unordered_map<LogicalTensorPtr, Operation*> newAssembles = {{inputs[0], nullptr}, {inputs[1], nullptr}};
    CollectOperations(func, inputsWeight, newAssembles, kNumTwo, kNumFour, 11);

    LogicalTensors reshapeOutputs;
    std::vector<std::string> expectAssembleDynShape = {"2", "2", "RUNTIME_Max((a*(a!=0)), 0)"};
    std::vector<std::string> expectValidShape1 = {
        "4", "RUNTIME_Max(((RUNTIME_Min(RUNTIME_Max(a, 0), 2)*(RUNTIME_Min(RUNTIME_Max(a, 0), 2)!=0))-0), 0)"};
    std::vector<std::string> expectValidShape2 = {
        "4",
        "RUNTIME_Max((((RUNTIME_Min(RUNTIME_Max((a-2), 0), 2)+2)*(RUNTIME_Min(RUNTIME_Max((a-2), 0), 2)!=0))-2), 0)"};
    std::unordered_map<LogicalTensorPtr, std::vector<int64_t>> expectAssembleOffset = {{inputs[0], assembleOffset1},
                                                                                       {inputs[1], assembleOffset2}};
    std::unordered_map<LogicalTensorPtr, std::vector<std::string>> expectValidShapes = {{inputs[0], expectValidShape1},
                                                                                        {inputs[1], expectValidShape2}};
    CheckNewAssembles(newAssembles, expectAssembleOffset, expectAssembleDynShape, expectValidShapes, dynInputShape,
                      reshapeOutputs, kNumTwo);
    std::vector<int64_t> expectedShape = {kNumFour, kNumTwo};
    EXPECT_EQ(reshapeOutputs[0]->shape, expectedShape);
    EXPECT_EQ(reshapeOutputs[1]->shape, expectedShape);
    EXPECT_NE(reshapeOutputs[0], reshapeOutputs[1]);
    EXPECT_EQ(reshapeOutputs[0]->GetConsumers().size(), kNumTwo);
    EXPECT_EQ(reshapeOutputs[1]->GetConsumers().size(), kNumTwo);
    auto view1 = *(reshapeOutputs[0]->GetConsumers().begin());
    auto view2 = *(++(reshapeOutputs[0]->GetConsumers().begin()));
    auto view3 = *(reshapeOutputs[1]->GetConsumers().begin());
    auto view4 = *(++(reshapeOutputs[1]->GetConsumers().begin()));
    EXPECT_NE(view1, view2);
    EXPECT_NE(view1, view3);
    EXPECT_NE(view1, view4);
}

TEST_F(TestSplitReshapePass, TestSplitReshapeWithDynamicFirstAxis)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    ASSERT_NE(func, nullptr);
    std::vector<SymbolicScalar> validShape = {CreateTestScalarVar("a"), kNumFour};
    BuildDynBeCoveredFunc(func, validShape);

    SplitReshape pass;
    EXPECT_EQ(pass.RunOnFunction(*func), SUCCESS);
    ASSERT_EQ(pass.reshapes_.size(), kSizeTwo);
    const std::vector<int64_t> expectedShape = {kNumFour, kNumTwo};
    for (const auto& [hash, reshape] : pass.reshapes_) {
        (void)hash;
        EXPECT_EQ(reshape->output->shape, expectedShape);
    }
}

TEST_F(TestSplitReshapePass, TestSplitDynamicFirstAxisMergeKeepsLocalShapeAndOffset)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    ASSERT_NE(func, nullptr);

    const std::vector<int64_t> reshapeInputShape = {kNumTwo, kNumTwo, kNumFour};
    const std::vector<int64_t> reshapeOutputShape = {kNumFour, kNumFour};
    const std::vector<int64_t> inputTileShape = {kNumOne, kNumTwo, kNumFour};
    const std::vector<int64_t> outputTileShape = {kNumTwo, kNumFour};
    const std::vector<int64_t> firstInputOffset = {kNumZero, kNumZero, kNumZero};
    const std::vector<int64_t> secondInputOffset = {kNumOne, kNumZero, kNumZero};
    const std::vector<int64_t> firstOutputOffset = {kNumZero, kNumZero};
    const std::vector<int64_t> secondOutputOffset = {kNumTwo, kNumZero};
    const auto globalBatch = CreateTestScalarVar("global_b");
    const auto localBatch = CreateTestScalarVar("local_b");
    const auto dynInner = CreateTestScalarVar("inner");
    const std::vector<SymbolicScalar> inputTileValidShape = {localBatch, dynInner, kNumFour};
    const std::vector<SymbolicScalar> reshapeValidShape = {globalBatch * dynInner, kNumFour};

    auto reshapeInput = IRBuilder().CreateTensorVar(DT_FP32, reshapeInputShape,
                                                    std::vector<SymbolicScalar>{globalBatch, dynInner, kNumFour});
    reshapeInput->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto reshapeOutput = IRBuilder().CreateTensorVar(DT_FP32, reshapeOutputShape, reshapeValidShape);
    reshapeOutput->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    auto inputRaw = std::make_shared<RawTensor>(DT_FP32, reshapeInputShape);
    auto input0 = IRBuilder().CreateTensorVar(inputRaw, firstInputOffset, inputTileShape, inputTileValidShape);
    auto input1 = IRBuilder().CreateTensorVar(inputRaw, secondInputOffset, inputTileShape, inputTileValidShape);
    input0->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    input1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    auto outputRaw = std::make_shared<RawTensor>(DT_FP32, reshapeOutputShape);
    const std::vector<SymbolicScalar> firstDynOffset = {kNumZero, kNumZero};
    const std::vector<SymbolicScalar> secondDynOffset = {dynInner, kNumZero};
    auto output0ValidShape = GetViewValidShape(reshapeValidShape, firstOutputOffset, firstDynOffset, outputTileShape);
    auto output1ValidShape = GetViewValidShape(reshapeValidShape, secondOutputOffset, secondDynOffset, outputTileShape);
    auto output0 = IRBuilder().CreateTensorVar(outputRaw, firstOutputOffset, outputTileShape, output0ValidShape);
    auto output1 = IRBuilder().CreateTensorVar(outputRaw, secondOutputOffset, outputTileShape, output1ValidShape);
    output0->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    output1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    auto& assemble0 = PassOperationUtils::AddOperation(*func, Opcode::OP_ASSEMBLE, {input0}, {reshapeInput});
    assemble0.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, firstInputOffset));
    auto& assemble1 = PassOperationUtils::AddOperation(*func, Opcode::OP_ASSEMBLE, {input1}, {reshapeInput});
    assemble1.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, secondInputOffset));
    auto& reshape = PassOperationUtils::AddOperation(*func, Opcode::OP_RESHAPE, {reshapeInput}, {reshapeOutput});
    reshape.SetAttribute(OP_ATTR_VALID_SHAPE, reshapeValidShape);
    auto& view0 = PassOperationUtils::AddOperation(*func, Opcode::OP_VIEW, {reshapeOutput}, {output0});
    view0.SetOpAttribute(std::make_shared<ViewOpAttribute>(firstOutputOffset, firstDynOffset, output0ValidShape));
    auto& view1 = PassOperationUtils::AddOperation(*func, Opcode::OP_VIEW, {reshapeOutput}, {output1});
    view1.SetOpAttribute(std::make_shared<ViewOpAttribute>(secondOutputOffset, secondDynOffset, output1ValidShape));
    func->inCasts_ = {input0, input1};
    func->outCasts_ = {output0, output1};

    SplitReshape pass;
    ASSERT_EQ(pass.RunOnFunction(*func), SUCCESS);
    ASSERT_EQ(pass.reshapes_.size(), kSizeTwo);

    LogicalTensorPtr targetInput;
    LogicalTensorPtr targetOutput;
    LogicalTensors splitOutputs;
    size_t reshapeCount = 0;
    for (auto& op : func->Operations(false)) {
        if (op.GetOpcode() != Opcode::OP_RESHAPE) {
            continue;
        }
        reshapeCount++;
        auto output = op.GetOutputOperand(kSizeZero);
        splitOutputs.emplace_back(output);
        if (output != nullptr && output->GetOffset() == secondOutputOffset) {
            targetInput = op.GetInputOperand(kSizeZero);
            targetOutput = output;
        }
    }
    EXPECT_EQ(reshapeCount, kSizeTwo);
    ASSERT_EQ(splitOutputs.size(), kSizeTwo);
    EXPECT_EQ(splitOutputs[0]->GetRawTensor(), splitOutputs[1]->GetRawTensor());
    ASSERT_NE(targetInput, nullptr);
    ASSERT_NE(targetOutput, nullptr);
    EXPECT_EQ(targetInput->GetShape(), inputTileShape);
    EXPECT_EQ(targetOutput->GetShape(), outputTileShape);

    const auto& localValidShape = targetOutput->GetDynValidShape();
    ASSERT_EQ(localValidShape.size(), kSizeTwo);
    EXPECT_NE(localValidShape[0].Dump().find("local_b"), std::string::npos);
    EXPECT_NE(localValidShape[0].Dump().find("inner"), std::string::npos);
    EXPECT_EQ(localValidShape[0].Dump().find("global_b"), std::string::npos);
    EXPECT_EQ(localValidShape[1].Dump(), std::to_string(kNumFour));

    const auto& reshapeDynOffset = targetOutput->GetDynOffset();
    ASSERT_EQ(reshapeDynOffset.size(), kSizeTwo);
    EXPECT_EQ(reshapeDynOffset[0].Dump(), dynInner.Dump());
    EXPECT_EQ(reshapeDynOffset[1].Dump(), std::to_string(kNumZero));

    ASSERT_EQ(targetOutput->GetConsumers().size(), kSizeOne);
    auto targetView = *targetOutput->GetConsumers().begin();
    ASSERT_NE(targetView, nullptr);
    ASSERT_EQ(targetView->GetOpcode(), Opcode::OP_VIEW);
    auto targetViewAttr = dynamic_cast<ViewOpAttribute*>(targetView->GetOpAttribute().get());
    ASSERT_NE(targetViewAttr, nullptr);
    EXPECT_EQ(targetViewAttr->GetFromOffset(), secondOutputOffset);
    ASSERT_EQ(targetViewAttr->GetFromDynOffset().size(), kSizeTwo);
    EXPECT_EQ(targetViewAttr->GetFromDynOffset()[0].Dump(), dynInner.Dump());
    EXPECT_EQ(targetViewAttr->GetFromDynOffset()[1].Dump(), std::to_string(kNumZero));
}

TEST_F(TestSplitReshapePass, TestDynamicFirstAxisUsesViewDynOffset)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    ASSERT_NE(func, nullptr);
    std::vector<SymbolicScalar> validShape = {CreateTestScalarVar("a"), kNumFour};
    BuildDynBeCoveredFunc(func, validShape);

    Operation* targetView = nullptr;
    const std::vector<int64_t> targetOffset = {kNumTwo, kNumZero};
    for (auto& op : func->Operations(false)) {
        if (op.GetOpcode() != Opcode::OP_VIEW) {
            continue;
        }
        auto viewAttr = dynamic_cast<ViewOpAttribute*>(op.GetOpAttribute().get());
        if (viewAttr != nullptr && viewAttr->GetFromOffset() == targetOffset) {
            targetView = &op;
            break;
        }
    }
    ASSERT_NE(targetView, nullptr);

    auto viewAttr = dynamic_cast<ViewOpAttribute*>(targetView->GetOpAttribute().get());
    ASSERT_NE(viewAttr, nullptr);

    SplitReshape pass;
    ASSERT_EQ(pass.CollectCopyOut(*func), SUCCESS);
    auto input = targetView->GetIOperands().front();
    auto output = targetView->GetOOperands().front();
    auto inputView = IRBuilder().CreateTensorVar(input->GetRawTensor(), targetOffset, output->shape,
                                                 std::vector<SymbolicScalar>{});
    CheckParam checkParam = {input, output, inputView};
    CheckOutputParam checkOutputParam;
    ASSERT_EQ(pass.CheckValidOp(checkParam, checkOutputParam), SUCCESS);

    auto expectedDynShape = GetViewValidShape(validShape, targetOffset, {}, output->shape);
    ASSERT_EQ(checkOutputParam.curViewDynShape.size(), expectedDynShape.size());
    for (size_t i = 0; i < expectedDynShape.size(); ++i) {
        EXPECT_EQ(checkOutputParam.curViewDynShape[i].Dump(), expectedDynShape[i].Dump());
    }
}

LogicalTensors BuildDynPerfectlyMatchWithAllFunc(std::shared_ptr<Function> func)
{
    std::vector<int64_t> shape1 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> shape2 = {kNumTwo, kNumEight, kNumTwo};
    std::vector<int64_t> shape3 = {kNumTwo, kNumFour, kNumTwo, kNumTwo};
    std::vector<int64_t> shape4 = {kNumTwo, kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> assembleOffset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> assembleOffset2 = {kNumZero, kNumTwo, kNumZero};
    std::vector<int64_t> assembleOffset3 = {kNumZero, kNumFour, kNumZero};
    std::vector<int64_t> assembleOffset4 = {kNumZero, kNumSix, kNumZero};
    std::vector<int64_t> viewOffset1 = {kNumZero, kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> viewOffset2 = {kNumZero, kNumTwo, kNumZero, kNumZero};
    std::vector<SymbolicScalar> validShape = {kNumTwo, kNumFour, kNumTwo, CreateTestScalarVar("a")};
    std::vector<SymbolicScalar> dynInputShape = {kNumTwo, kNumTwo, CreateTestScalarVar("a")};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape2);
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape3);
    auto input1 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor1, assembleOffset1, shape1, dynInputShape);
    input1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto input2 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor1, assembleOffset2, shape1, dynInputShape);
    input2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto input3 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor1, assembleOffset3, shape1, dynInputShape);
    input3->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto input4 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor1, assembleOffset4, shape1, dynInputShape);
    input4->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape3, CreateTestConstIntVector(shape3));
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output1 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor2, viewOffset1, shape4,
                                                              CreateTestConstIntVector(shape4));
    output1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output2 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor2, viewOffset2, shape4,
                                                              CreateTestConstIntVector(shape4));
    output2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    auto& assemble_op1 = PassOperationUtils::AddOperation(*func, Opcode::OP_ASSEMBLE, {input1}, {ubTensor1});
    assemble_op1.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset1));
    auto& assemble_op2 = PassOperationUtils::AddOperation(*func, Opcode::OP_ASSEMBLE, {input2}, {ubTensor1});
    assemble_op2.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset2));
    auto& assemble_op3 = PassOperationUtils::AddOperation(*func, Opcode::OP_ASSEMBLE, {input3}, {ubTensor1});
    assemble_op3.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset3));
    auto& assemble_op4 = PassOperationUtils::AddOperation(*func, Opcode::OP_ASSEMBLE, {input4}, {ubTensor1});
    assemble_op4.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset4));
    auto& reshape_op = PassOperationUtils::AddOperation(*func, Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    reshape_op.SetAttribute(OP_ATTR_VALID_SHAPE, validShape);
    auto& view_op1 = PassOperationUtils::AddOperation(*func, Opcode::OP_VIEW, {ubTensor2}, {output1});
    view_op1.SetOpAttribute(std::make_shared<ViewOpAttribute>(viewOffset1));
    auto& view_op2 = PassOperationUtils::AddOperation(*func, Opcode::OP_VIEW, {ubTensor2}, {output2});
    view_op2.SetOpAttribute(std::make_shared<ViewOpAttribute>(viewOffset2));

    func->inCasts_ = {input1, input2, input3, input4};
    func->outCasts_ = {output1, output2};
    return {input1, input2, input3, input4};
}

/*
验证多对一场景下动态shape的兜底策略
因为缺乏宏构建策略，手动构造expandfunction的输出构图
1) 用例设置：
{2,2,2} -> assemble -> {2,8,2} -> reshape -> {2,4,2,2} -> view -> {2,2,2,2}
{2,2,2} -> assemble                                    -> view -> {2,2,2,2}
{2,2,2} -> assemble
{2,2,2} -> assemble
{2,2,a}                         {2,4,2,a}
2) splitreshape
{2,2,2} -> assemble ->
{2,2,2} -> assemble -> reshape -> {2,2,2,2} -> view -> {2,2,2,2}
{2,2,2} -> assemble -> reshape -> {2,2,2,2} -> view -> {2,2,2,2}
{2,2,2} -> assemble ->
{2,2,a}           {2,4,a}         {2,2,2,a}
                        {2,2,2,Max(0, RUNTIME_GetViewValidShapeDim(a,0,2))}
*/
TEST_F(TestSplitReshapePass, TestDynPerfectlyMatchWithAllSTest)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(func != nullptr);

    std::vector<int64_t> assembleOffset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> assembleOffset2 = {kNumZero, kNumTwo, kNumZero};
    std::vector<int64_t> assembleOffset3 = {kNumZero, kNumFour, kNumZero};
    std::vector<int64_t> assembleOffset4 = {kNumZero, kNumSix, kNumZero};
    std::vector<SymbolicScalar> dynInputShape = {kNumTwo, kNumTwo, CreateTestScalarVar("a")};

    auto inputs = BuildDynPerfectlyMatchWithAllFunc(func);
    RunPassStra(*func, PassName::SPLIT_RESHAPE);

    std::unordered_map<LogicalTensorPtr, int> inputsWeight = {
        {inputs[0], 1}, {inputs[1], 10}, {inputs[2], 100}, {inputs[3], 1000}};
    std::unordered_map<LogicalTensorPtr, Operation*> newAssembles = {
        {inputs[0], nullptr}, {inputs[1], nullptr}, {inputs[2], nullptr}, {inputs[3], nullptr}};
    CollectOperations(func, inputsWeight, newAssembles, kNumTwo, kNumTwo, 1111);

    std::unordered_map<LogicalTensorPtr, std::vector<int64_t>> expectAssembleOffset = {{inputs[0], assembleOffset1},
                                                                                       {inputs[1], assembleOffset2},
                                                                                       {inputs[2], assembleOffset3},
                                                                                       {inputs[3], assembleOffset4}};
    LogicalTensors reshapeOutputs;

    std::vector<std::string> expectAssembleDynShape = {"2", "4", "RUNTIME_Max((a*(a!=0)), 0)"};
    std::vector<std::string> expectValidShape = {
        "2", "2", "2",
        "RUNTIME_Max(((RUNTIME_Min(RUNTIME_Max(a, 0), 2)*(RUNTIME_Min(RUNTIME_Max(a, 0), 2)!=0))-0), 0)"};
    std::unordered_map<LogicalTensorPtr, std::vector<std::string>> expectValidShapes = {{inputs[0], expectValidShape},
                                                                                        {inputs[1], expectValidShape},
                                                                                        {inputs[2], expectValidShape},
                                                                                        {inputs[3], expectValidShape}};
    CheckNewAssembles(newAssembles, expectAssembleOffset, expectAssembleDynShape, expectValidShapes, dynInputShape,
                      reshapeOutputs, kNumFour);
    int eqCount = (reshapeOutputs[0] == reshapeOutputs[1]) + (reshapeOutputs[0] == reshapeOutputs[2]) +
                  (reshapeOutputs[0] == reshapeOutputs[3]) + (reshapeOutputs[1] == reshapeOutputs[2]) +
                  (reshapeOutputs[1] == reshapeOutputs[3]) + (reshapeOutputs[2] == reshapeOutputs[3]);
    const int expectEqCount = 2;
    EXPECT_EQ(eqCount, expectEqCount); // reshapeOutput共四个，两两相等，两组之间不等，所以预期相等数量为2。
    for (const auto& outputTensor : reshapeOutputs) {
        EXPECT_EQ(outputTensor->GetConsumers().size(), kNumOne);
    }
}

namespace {
const int scopeId = 10;
const std::string keyInt = "key_int";
const int attrInt = 0;
const std::string keyBool = "key_bool";
const bool attrBool = true;
const std::string keyElement = "key_element";
const DataType eleDatatype = DT_INT4;
const int32_t eleValue = 1;
const Element attrElement = Element(eleDatatype, eleValue);
const std::string keyInt64 = "key_int64";
const int64_t attrInt64 = 2;
const std::string keyCastmode = "key_castmode";
const CastMode attrCastmode = CAST_FLOOR;
const std::string keyString = "key_string";
const std::string attrString = "attr_string";
const std::string keySymbolicScalar = "key_symbolicScalar";
const SymbolicScalar attrSymbolicScalar = IRBuilder().CreateConstInt(4);
} // namespace

// 测试新生成的op是否继承了原op的scopeID和attribute。
TEST_F(TestSplitReshapePass, TestInheritAttribute)
{
    std::vector<int64_t> origShape = {kNumFour, kNumTwo, kNumTwo};
    std::vector<int64_t> reshapeShape = {kNumFour, kNumFour};
    std::vector<int64_t> tiledShape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> tiledorigShape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> tiledreshapeShape = {kNumTwo, kNumFour};
    std::vector<int64_t> tiledviewShape = {kNumTwo, kNumTwo};
    TileShape::Current().SetVecTile(tiledShape);
    Tensor input(DT_FP32, origShape, "input");
    Tensor output(DT_FP32, reshapeShape, "output");

    FUNCTION("STCase4")
    {
        Tensor exp = Exp(input);
        Tensor reshape = Reshape(exp, reshapeShape);
        output = Exp(reshape);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase4");
    RunPassStra(*func, PassName::EXPAND_FUNCTION);
    for (auto& op : func->Operations()) {
        op.SetScopeId(scopeId);
        op.SetAttribute(keyInt, attrInt);
        op.SetAttribute(keyBool, attrBool);
        op.SetAttribute(keyElement, attrElement);
        op.SetAttribute(keyInt64, attrInt64);
        op.SetAttribute(keyCastmode, attrCastmode);
        op.SetAttribute(keyString, attrString);
        op.SetAttribute(keySymbolicScalar, attrSymbolicScalar);
    }
    RunPassStra(*func, PassName::SPLIT_RESHAPE);
    for (const auto& op : func->Operations()) {
        EXPECT_EQ(scopeId, op.GetScopeId());
        EXPECT_EQ(attrInt, op.GetIntAttribute(keyInt));
        EXPECT_EQ(attrBool, op.GetBoolAttribute(keyBool));
        EXPECT_EQ(eleDatatype, op.GetElementAttribute(keyElement).GetDataType());
        EXPECT_EQ(eleValue, op.GetElementAttribute(keyElement).GetUnsignedData());
        EXPECT_EQ(attrInt64, op.GetIntAttribute(keyInt64));
        EXPECT_EQ(attrCastmode, op.GetCastModeAttribute(keyCastmode));
        EXPECT_EQ(attrString, op.GetStringAttribute(keyString));
        EXPECT_EQ(attrSymbolicScalar, op.GetSymbolicScalarAttribute(keySymbolicScalar));
    }
}

// check the number of reshape operations
// return the number of total operations
int CheckOpNum(Function* func, const uint32_t expectReshapeNum)
{
    int reshapeOp = 0;
    int OpNum = 0;
    for (auto& op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            reshapeOp++;
        }
        OpNum++;
    }
    EXPECT_EQ(reshapeOp, expectReshapeNum);
    return OpNum;
}

/*
splitreshape pass不起作用的场景
一对多场景(使用expandfunction作为前序pass)
assemble的tile输入无法被映射为一个完整inputView的tile
1) 用例设置：
{1,1,2,8} -> exp -> {1,1,2,8} -> reshape -> {1,16} -> exp -> {1,16}
tileshape1 = {2,1,2,4}
tileshape2 = {1,4}
2) expandfunction：
exp -> {1,1,2,4} -> assemble -> reshape -> {1,16} -> view -> {1,4} -> exp -> {1,4}
exp -> {1,1,2,4} -> assemble                      -> view -> {1,4} -> exp -> {1,4}
                                                  -> view -> {1,4} -> exp -> {1,4}
                                                  -> view -> {1,4} -> exp -> {1,4}
*/
TEST_F(TestSplitReshapePass, TestExceptionCase1)
{
    // Define the shape of the Tensors
    std::vector<int64_t> origShape = {kNumOne, kNumOne, kNumTwo, kNumEight};
    std::vector<int64_t> reshapeShape = {kNumOne, kExpFour};
    std::vector<int64_t> tiledShape1 = {kNumTwo, kNumOne, kNumTwo, kNumFour};
    std::vector<int64_t> tiledShape2 = {kNumOne, kNumFour};

    Tensor input(DT_FP32, origShape, "input");
    Tensor output(DT_FP32, reshapeShape, "output");

    FUNCTION("STCase5")
    {
        TileShape::Current().SetVecTile(tiledShape1);
        Tensor exp = Exp(input);
        Tensor reshape = Reshape(exp, reshapeShape);
        TileShape::Current().SetVecTile(tiledShape2);
        output = Exp(reshape);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase5");

    RunPassStra(*func, PassName::EXPAND_FUNCTION);
    int OpNum = CheckOpNum(func, kNumOne);
    RunPassStra(*func, PassName::SPLIT_RESHAPE);
    int AfterOpNum = CheckOpNum(func, kNumOne);
    EXPECT_EQ(AfterOpNum, OpNum);
}

/*
splitreshape pass不起作用的场景
使用expandfunction作为前序pass
无法计算reshape前后的加细
1) 用例设置：
{64,6} -> exp -> {64,6} -> reshape -> {96,4} -> exp -> {96,4}
tileshape = {32,2}
2) expandfunction：
exp -> {32,2} -> assemble -> {64,6} -> reshape -> {96,4} -> view -> {32,2} -> exp -> {32,2}
exp -> {32,2} -> assemble                                -> view -> {32,2} -> exp -> {32,2}
exp -> {32,2} -> assemble                                -> view -> {32,2} -> exp -> {32,2}
exp -> {32,2} -> assemble                                -> view -> {32,2} -> exp -> {32,2}
exp -> {32,2} -> assemble                                -> view -> {32,2} -> exp -> {32,2}
exp -> {32,2} -> assemble                                -> view -> {32,2} -> exp -> {32,2}
*/
TEST_F(TestSplitReshapePass, TestExceptionCase2)
{
    // Define the shape of the Tensors
    std::vector<int64_t> origShape = {kExpSix, kNumSix};
    std::vector<int64_t> reshapeShape = {kNumNineSix, kNumFour};
    std::vector<int64_t> tiledShape = {kExpFive, kNumTwo};

    TileShape::Current().SetVecTile(tiledShape);
    Tensor input(DT_FP32, origShape, "input");
    Tensor output(DT_FP32, reshapeShape, "output");

    FUNCTION("STCase6")
    {
        Tensor exp = Exp(input);
        Tensor reshape = Reshape(exp, reshapeShape);
        output = Exp(reshape);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase6");

    RunPassStra(*func, PassName::EXPAND_FUNCTION);

    CheckOpNum(func, kNumOne);
    RunPassStra(*func, PassName::SPLIT_RESHAPE);
    CheckOpNum(func, kNumOne);
}

/*
splitreshape pass不起作用的场景
多对多的场景，前后的tile即非cover也非covered
1) 用例设置：
{8,8} -> exp -> {8,8} -> reshape -> {16,4} -> exp -> {16,4}
tileshape1 = {2,4}
tileshape1 = {4,2}
2) expandfunction：
exp -> {2,4} -> assemble -> {8,8} -> reshape -> {16,4} -> view -> {4,2} -> exp -> {4,2}
exp -> {2,4} -> assemble                               -> view -> {4,2} -> exp -> {4,2}
exp -> {2,4} -> assemble                               -> view -> {4,2} -> exp -> {4,2}
exp -> {2,4} -> assemble                               -> view -> {4,2} -> exp -> {4,2}
exp -> {2,4} -> assemble                               -> view -> {4,2} -> exp -> {4,2}
exp -> {2,4} -> assemble                               -> view -> {4,2} -> exp -> {4,2}
*/
TEST_F(TestSplitReshapePass, TestExceptionCase3)
{
    // Define the shape of the Tensors
    std::vector<int64_t> origShape = {kNumEight, kNumEight};
    std::vector<int64_t> reshapeShape = {kExpFour, kNumFour};
    std::vector<int64_t> tiledShape1 = {kNumTwo, kNumFour};
    std::vector<int64_t> tiledShape2 = {kNumFour, kNumTwo};

    Tensor input(DT_FP32, origShape, "input");
    Tensor output(DT_FP32, reshapeShape, "output");

    FUNCTION("STCase7")
    {
        TileShape::Current().SetVecTile(tiledShape1);
        Tensor exp = Exp(input);
        Tensor reshape = Reshape(exp, reshapeShape);
        TileShape::Current().SetVecTile(tiledShape2);
        output = Exp(reshape);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase7");

    RunPassStra(*func, PassName::EXPAND_FUNCTION);

    CheckOpNum(func, kNumOne);
    RunPassStra(*func, PassName::SPLIT_RESHAPE);
    CheckOpNum(func, kNumOne);
}

/*
splitreshape pass不起作用的场景
动态shape位于变化轴
                                 {2,2,2,a}
{2,2,2} -> assemble -> {2,2,4} -> reshape -> {2,2,2,2} -> view -> {2,2,1,2}
{2,2,2} -> assemble                                    -> view -> {2,2,1,2}
*/
TEST_F(TestSplitReshapePass, TestExceptionCase4)
{
    // Define the shape of the Tensors
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(func != nullptr);

    std::vector<int64_t> shape1 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> shape2 = {kNumTwo, kNumTwo, kNumFour};
    std::vector<int64_t> shape3 = {kNumTwo, kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> shape4 = {kNumTwo, kNumTwo, kNumOne, kNumTwo};
    std::vector<int64_t> assembleOffset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> assembleOffset2 = {kNumZero, kNumZero, kNumTwo};
    std::vector<int64_t> viewOffset1 = {kNumZero, kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> viewOffset2 = {kNumZero, kNumZero, kNumOne, kNumZero};
    std::vector<SymbolicScalar> validShape = {kNumTwo, kNumTwo, kNumTwo, CreateTestScalarVar("a")};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape2);
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape3);
    auto input1 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor1, assembleOffset1, shape1,
                                                             CreateTestConstIntVector(shape1));
    input1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto input2 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor1, assembleOffset2, shape1,
                                                             CreateTestConstIntVector(shape1));
    input2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape3, CreateTestConstIntVector(shape3));
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output1 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor2, viewOffset1, shape4,
                                                              CreateTestConstIntVector(shape4));
    output1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output2 = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor2, viewOffset2, shape4,
                                                              CreateTestConstIntVector(shape4));
    output2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    auto& reshape_op = PassOperationUtils::AddOperation(*func, Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    reshape_op.SetAttribute(OP_ATTR_VALID_SHAPE, validShape);
    auto& view_op1 = PassOperationUtils::AddOperation(*func, Opcode::OP_VIEW, {ubTensor2}, {output1});
    view_op1.SetOpAttribute(std::make_shared<ViewOpAttribute>(viewOffset1));
    auto& view_op2 = PassOperationUtils::AddOperation(*func, Opcode::OP_VIEW, {ubTensor2}, {output2});
    view_op2.SetOpAttribute(std::make_shared<ViewOpAttribute>(viewOffset2));
    auto& assemble_op1 = PassOperationUtils::AddOperation(*func, Opcode::OP_ASSEMBLE, {input1}, {ubTensor1});
    assemble_op1.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset1));
    auto& assemble_op2 = PassOperationUtils::AddOperation(*func, Opcode::OP_ASSEMBLE, {input2}, {ubTensor1});
    assemble_op2.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset2));

    func->inCasts_.push_back(input1);
    func->inCasts_.push_back(input2);
    func->outCasts_.push_back(output1);
    func->outCasts_.push_back(output2);

    CheckOpNum(func.get(), kNumOne);
    RunPassStra(*func, PassName::SPLIT_RESHAPE);
    CheckOpNum(func.get(), kNumOne);
}

/*
splitreshape pass不起作用的场景
多对一场景(使用expandfunction作为前序pass)
assemble的tile输入无法被映射为一个完整inputView的tile
exp -> {1,4} -> assemble -> {1,16} -> reshape -> {1,1,2,8} -> view -> {1,1,2,4} -> exp -> {1,1,2,4}
exp -> {1,4} -> assemble                                   -> view -> {1,1,2,4} -> exp -> {1,1,2,4}
exp -> {1,4} -> assemble
exp -> {1,4} -> assemble
*/
TEST_F(TestSplitReshapePass, TestExceptionCase5)
{
    // Define the shape of the Tensors
    std::vector<int64_t> origShape = {kNumOne, kExpFour};
    std::vector<int64_t> reshapeShape = {kNumOne, kNumOne, kNumTwo, kNumEight};
    std::vector<int64_t> tiledShape1 = {kNumOne, kNumFour};
    std::vector<int64_t> tiledShape2 = {kNumTwo, kNumOne, kNumTwo, kNumFour};
    std::vector<int64_t> tiledreshapeShape = {kNumOne, kNumFour};
    std::vector<int64_t> tiledassembleShape = {kNumOne, kNumOne, kNumOne, kNumFour};
    std::vector<int64_t> tiledviewShape = {kNumOne, kNumOne, kNumTwo, kNumFour};

    Tensor input(DT_FP32, origShape, "input");
    Tensor output(DT_FP32, reshapeShape, "output");

    FUNCTION("STCase8")
    {
        TileShape::Current().SetVecTile(tiledShape1);
        Tensor exp = Exp(input);
        Tensor reshape = Reshape(exp, reshapeShape);
        TileShape::Current().SetVecTile(tiledShape2);
        output = Exp(reshape);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase8");

    RunPassStra(*func, PassName::EXPAND_FUNCTION);
    int OpNum = CheckOpNum(func, kNumOne);
    RunPassStra(*func, PassName::SPLIT_RESHAPE);
    int AfterOpNum = CheckOpNum(func, kNumOne);
    EXPECT_EQ(AfterOpNum, OpNum);
}

TEST_F(TestSplitReshapePass, TestShapeAlignIndexExceeded)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    ASSERT_TRUE(currFunctionPtr != nullptr);
    SplitReshape pass;
    // shape1 单维先耗尽：第一轮 i1=0,i2=0 得到 prod1=1,i1=1,i2=1；下一轮 prod1==1&&prod2==1 时 i1>=shape1.size() 触发
    // line 393
    std::vector<int64_t> shape1 = {kNumTwo};          // 1 dim
    std::vector<int64_t> shape2 = {kNumTwo, kNumTwo}; // 2 dims
    std::vector<int64_t> alignedShape;
    Status status = pass.ShapeAlign(shape1, shape2, alignedShape);
    EXPECT_EQ(status, FAILED); // line 393: i1 >= shape1.size() || i2 >= shape2.size()
}

TEST_F(TestSplitReshapePass, TestRawToAlignUpdateShapeOffsetTailCond)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    ASSERT_TRUE(currFunctionPtr != nullptr);
    SplitReshape pass;
    ReshapeTilePara shapePara;
    std::vector<int64_t> rawShape = {kNumEight};
    std::vector<int64_t> alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> tileOffset = {kNumThree}; // currentOffset=3
    std::vector<int64_t> tileShape = {kNumTwo};    // currentShape=2, stride becomes 4 -> 3+2>4
    shapePara = {rawShape, alignedShape, tileOffset, tileShape};
    std::vector<int64_t> newOffset, newShape;
    Status status = pass.RawToAlign(shapePara, newOffset, newShape);
    EXPECT_EQ(status, WARNING); // line 420: currentOffset + currentShape > stride
}

TEST_F(TestSplitReshapePass, TestRawToAlignUpdateShapeOffsetOffsetNotDivisible)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    ASSERT_TRUE(currFunctionPtr != nullptr);
    SplitReshape pass;
    ReshapeTilePara shapePara;
    std::vector<int64_t> rawShape = {kNumEight};
    std::vector<int64_t> alignedShape = {kNumTwo, kNumFour};
    std::vector<int64_t> tileOffset = {kNumThree}; // currentOffset=3, stride becomes 4 -> 3%4!=0
    std::vector<int64_t> tileShape = {kNumEight};
    shapePara = {rawShape, alignedShape, tileOffset, tileShape};
    std::vector<int64_t> newOffset, newShape;
    Status status = pass.RawToAlign(shapePara, newOffset, newShape);
    EXPECT_EQ(status, WARNING); // line 426: currentOffset % stride != 0
}

TEST_F(TestSplitReshapePass, TestRawToAlignUpdateShapeOffsetShapeNotDivisible)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    ASSERT_TRUE(currFunctionPtr != nullptr);
    SplitReshape pass;
    ReshapeTilePara shapePara;
    std::vector<int64_t> rawShape = {kNumEight};
    std::vector<int64_t> alignedShape = {kNumTwo, kNumFour};
    std::vector<int64_t> tileOffset = {kNumZero};
    std::vector<int64_t> tileShape = {kNumSix}; // currentShape=5 would be 5%4!=0; use 6 so first block no, then 6%4!=0
    shapePara = {rawShape, alignedShape, tileOffset, tileShape};
    std::vector<int64_t> newOffset, newShape;
    Status status = pass.RawToAlign(shapePara, newOffset, newShape);
    EXPECT_EQ(status, WARNING); // line 430: currentShape % stride != 0
}

TEST_F(TestSplitReshapePass, TestRawToAlignConstructShapeOffsetStrideNotDivisible)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    ASSERT_TRUE(currFunctionPtr != nullptr);
    SplitReshape pass;
    ReshapeTilePara shapePara;
    static const int64_t kNine = 9;
    std::vector<int64_t> rawShape = {kNine}; // 9 % 2 != 0
    std::vector<int64_t> alignedShape = {kNumTwo, kNumTwo};
    std::vector<int64_t> tileOffset = {kNumZero};
    std::vector<int64_t> tileShape = {kNine};
    shapePara = {rawShape, alignedShape, tileOffset, tileShape};
    std::vector<int64_t> newOffset, newShape;
    Status status = pass.RawToAlign(shapePara, newOffset, newShape);
    EXPECT_EQ(status, WARNING); // line 454: stride % alignedShape[i] != 0
}

TEST_F(TestSplitReshapePass, TestRawToAlignShapeOneTileInvalid)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    ASSERT_TRUE(currFunctionPtr != nullptr);
    SplitReshape pass;
    ReshapeTilePara shapePara;
    std::vector<int64_t> rawShape = {kNumOne, kNumEight};
    std::vector<int64_t> alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> tileOffset = {kNumOne, kNumTwo}; // shape[0]=1 but tileOffset[0]=1 (should be 0)
    std::vector<int64_t> tileShape = {kNumOne, kNumTwo};
    shapePara = {rawShape, alignedShape, tileOffset, tileShape};
    std::vector<int64_t> newOffset, newShape;
    Status status = pass.RawToAlign(shapePara, newOffset, newShape);
    EXPECT_EQ(status, FAILED); // line 492: Shape[%zu] is 1, but tileOffset/tileShape invalid
}

TEST_F(TestSplitReshapePass, TestAlignToRawRawShapeZero)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    ASSERT_TRUE(currFunctionPtr != nullptr);
    SplitReshape pass;
    ReshapeTilePara shapePara;
    std::vector<int64_t> alignedShape = {kNumZero, kNumTwo}; // rawShape in AlignToRaw = shapePara.shape
    std::vector<int64_t> rawShape = {kNumSix};
    std::vector<int64_t> tileOffset = {kNumZero, kNumZero};
    std::vector<int64_t> tileShape = {kNumZero, kNumThree};
    shapePara = {alignedShape, rawShape, tileOffset, tileShape};
    std::vector<int64_t> newOffset, newShape;
    Status status = pass.AlignToRaw(shapePara, newOffset, newShape);
    EXPECT_EQ(status, FAILED); // line 537: rawShape[i] is 0
}

TEST_F(TestSplitReshapePass, TestAlignToRawStrideNotDivisibleByRawShape)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    ASSERT_TRUE(currFunctionPtr != nullptr);
    SplitReshape pass;
    ReshapeTilePara shapePara;
    std::vector<int64_t> alignedShape = {kNumFour}; // rawShape = [4]
    std::vector<int64_t> rawShape = {kNumSix};      // newRawshape = [6], stride=6, 6%4!=0
    std::vector<int64_t> tileOffset = {kNumZero};
    std::vector<int64_t> tileShape = {kNumFour};
    shapePara = {alignedShape, rawShape, tileOffset, tileShape};
    std::vector<int64_t> newOffset, newShape;
    Status status = pass.AlignToRaw(shapePara, newOffset, newShape);
    EXPECT_EQ(status, FAILED); // line 541: stride % rawShape[i] != 0
}

/*
 * Test CollectCopyOut with input that has CopyOut producer.
 * After removing CheckProducerCopyOut check, assemble ops whose input has CopyOut producer
 * should also be collected.
 * Graph: ubTensor(UB) -> CopyOut -> ddrTensor(DDR) -> Assemble -> ddrTensor2(DDR) -> Reshape -> output
 * CopyOut: UB -> DDR
 */
TEST_F(TestSplitReshapePass, TestCollectCopyOutWithCopyOutProducer)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit",
                                                      nullptr);
    ASSERT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {kNumTwo, kNumFour};
    std::vector<int64_t> reshapeOutputShape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> offset = {kNumZero, kNumZero};

    auto ubTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    ubTensor->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);

    auto ddrRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    auto ddrTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(ddrRawTensor, offset, shape,
                                                                CreateTestConstIntVector(shape));
    ddrTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    auto ddrTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    ddrTensor2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    auto reshapeOutput = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshapeOutputShape,
                                                                    CreateTestConstIntVector(reshapeOutputShape));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {ubTensor}, {ddrTensor});

    auto& assembleOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {ddrTensor},
                                                        {ddrTensor2});
    assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset));

    auto& reshapeOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {ddrTensor2},
                                                       {reshapeOutput});
    std::vector<SymbolicScalar> validShape = {kNumTwo, kNumTwo, kNumTwo};
    reshapeOp.SetAttribute(OP_ATTR_VALID_SHAPE, validShape);

    SplitReshape pass;
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);

    EXPECT_EQ(pass.assembleOutToInput_.size(), kSizeOne);
    auto it = pass.assembleOutToInput_.find(ddrTensor2->tensor->rawmagic);
    EXPECT_NE(it, pass.assembleOutToInput_.end());
    EXPECT_EQ(it->second.size(), kNumOne);
    EXPECT_EQ(std::count(it->second.begin(), it->second.end(), ddrTensor), kNumOne);

    EXPECT_EQ(pass.mapOffset_.size(), kSizeOne);
    EXPECT_EQ(pass.mapOffset_[std::make_pair(ddrTensor->magic, ddrTensor2->magic)], offset);
}

/*
 * Test SplitReshape with multiple parallel CopyOut->Assemble paths.
 * Multiple independent UB tensors go through CopyOut->Assemble to different offsets
 * of the same DDR tensor, then Reshape changes shape from [6,4] to [3,8],
 * followed by multiple Views from different offsets, each consumed by ADDS.
 */
struct MultipleParallelCopyOutAssembleShapes {
    std::vector<int64_t> ubShape{kNumTwo, kNumFour};
    std::vector<int64_t> ddrBigShape{kNumSix, kNumFour};
    std::vector<int64_t> reshapeShape{kNumThree, kNumEight};
    std::vector<int64_t> viewOutShape{kNumOne, kNumEight};
    std::vector<std::vector<int64_t>> assembleOffsets{{kNumZero, kNumZero}, {kNumTwo, kNumZero}, {kNumFour, kNumZero}};
    std::vector<std::vector<int64_t>> viewOffsets{{kNumZero, kNumZero}, {kNumOne, kNumZero}, {kNumTwo, kNumZero}};
};

void BuildMultipleParallelCopyOutAssembleGraph(std::shared_ptr<Function>& func,
                                               const MultipleParallelCopyOutAssembleShapes& shapes,
                                               std::vector<std::shared_ptr<LogicalTensor>>& inputs,
                                               std::vector<std::shared_ptr<LogicalTensor>>& outputs)
{
    auto sharedDdrRawTensor = std::make_shared<RawTensor>(DT_FP32, shapes.ddrBigShape);
    std::vector<std::shared_ptr<LogicalTensor>> ddrTensors;
    for (size_t i = 0; i < kNumThree; i++) {
        auto input = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapes.ubShape,
                                                                CreateTestConstIntVector(shapes.ubShape));
        input->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
        inputs.push_back(input);
        auto ddrTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(
            sharedDdrRawTensor, shapes.assembleOffsets[i], shapes.ubShape, CreateTestConstIntVector(shapes.ubShape));
        ddrTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
        ddrTensors.push_back(ddrTensor);
        PassOperationUtils::AddOperation(*func, Opcode::OP_COPY_OUT, {input}, {ddrTensor});
    }
    auto ddrBigTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapes.ddrBigShape,
                                                                   CreateTestConstIntVector(shapes.ddrBigShape));
    ddrBigTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    for (size_t i = 0; i < kNumThree; i++) {
        auto& assembleOp = PassOperationUtils::AddOperation(*func, Opcode::OP_ASSEMBLE, {ddrTensors[i]},
                                                            {ddrBigTensor});
        assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, shapes.assembleOffsets[i]));
    }
    auto ddrReshape = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapes.reshapeShape,
                                                                 CreateTestConstIntVector(shapes.reshapeShape));
    ddrReshape->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto& reshapeOp = PassOperationUtils::AddOperation(*func, Opcode::OP_RESHAPE, {ddrBigTensor}, {ddrReshape});
    reshapeOp.SetAttribute(OP_ATTR_VALID_SHAPE, std::vector<SymbolicScalar>{kNumThree, kNumEight});
    for (size_t i = 0; i < kNumThree; i++) {
        auto ubOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapes.viewOutShape,
                                                                CreateTestConstIntVector(shapes.viewOutShape));
        ubOut->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
        auto& viewOp = PassOperationUtils::AddOperation(*func, Opcode::OP_VIEW, {ddrReshape}, {ubOut});
        viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(shapes.viewOffsets[i]));
        auto output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapes.viewOutShape,
                                                                 CreateTestConstIntVector(shapes.viewOutShape));
        PassOperationUtils::AddOperation(*func, Opcode::OP_ADDS, {ubOut}, {output});
        outputs.push_back(output);
    }
    func->inCasts_ = inputs;
    func->outCasts_ = outputs;
}

std::vector<std::shared_ptr<LogicalTensor>> BuildReduceAccAssembleDdrTensors(
    std::shared_ptr<Function>& func, const MultipleParallelCopyOutAssembleShapes& shapes,
    std::vector<std::shared_ptr<LogicalTensor>>& inputs)
{
    auto sharedDdrRawTensor = std::make_shared<RawTensor>(DT_FP32, shapes.ddrBigShape);
    std::vector<std::shared_ptr<LogicalTensor>> ddrTensors;
    for (size_t i = 0; i < kNumThree; i++) {
        auto ddrTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(
            sharedDdrRawTensor, shapes.assembleOffsets[i], shapes.ubShape, CreateTestConstIntVector(shapes.ubShape));
        ddrTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
        ddrTensors.push_back(ddrTensor);
        if (i == 0) {
            auto reduceInput0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapes.ubShape,
                                                                           CreateTestConstIntVector(shapes.ubShape));
            auto reduceInput1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapes.ubShape,
                                                                           CreateTestConstIntVector(shapes.ubShape));
            reduceInput0->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
            reduceInput1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
            inputs.push_back(reduceInput0);
            inputs.push_back(reduceInput1);
            PassOperationUtils::AddOperation(*func, Opcode::OP_REDUCE_ACC, {reduceInput0, reduceInput1}, {ddrTensor});
            continue;
        }
        auto input = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapes.ubShape,
                                                                CreateTestConstIntVector(shapes.ubShape));
        input->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
        inputs.push_back(input);
        PassOperationUtils::AddOperation(*func, Opcode::OP_COPY_OUT, {input}, {ddrTensor});
    }
    return ddrTensors;
}

void BuildReduceAccAssembleInputGraph(std::shared_ptr<Function>& func,
                                      const MultipleParallelCopyOutAssembleShapes& shapes,
                                      std::vector<std::shared_ptr<LogicalTensor>>& inputs,
                                      std::vector<std::shared_ptr<LogicalTensor>>& outputs)
{
    auto ddrTensors = BuildReduceAccAssembleDdrTensors(func, shapes, inputs);
    auto ddrBigTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapes.ddrBigShape,
                                                                   CreateTestConstIntVector(shapes.ddrBigShape));
    ddrBigTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    for (size_t i = 0; i < kNumThree; i++) {
        auto& assembleOp = PassOperationUtils::AddOperation(*func, Opcode::OP_ASSEMBLE, {ddrTensors[i]},
                                                            {ddrBigTensor});
        assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, shapes.assembleOffsets[i]));
    }
    auto ddrReshape = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapes.reshapeShape,
                                                                 CreateTestConstIntVector(shapes.reshapeShape));
    ddrReshape->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto& reshapeOp = PassOperationUtils::AddOperation(*func, Opcode::OP_RESHAPE, {ddrBigTensor}, {ddrReshape});
    reshapeOp.SetAttribute(OP_ATTR_VALID_SHAPE, std::vector<SymbolicScalar>{kNumThree, kNumEight});
    for (size_t i = 0; i < kNumThree; i++) {
        auto ubOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapes.viewOutShape,
                                                                CreateTestConstIntVector(shapes.viewOutShape));
        ubOut->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
        auto& viewOp = PassOperationUtils::AddOperation(*func, Opcode::OP_VIEW, {ddrReshape}, {ubOut});
        viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(shapes.viewOffsets[i]));
        auto output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapes.viewOutShape,
                                                                 CreateTestConstIntVector(shapes.viewOutShape));
        PassOperationUtils::AddOperation(*func, Opcode::OP_ADDS, {ubOut}, {output});
        outputs.push_back(output);
    }
    func->inCasts_ = inputs;
    func->outCasts_ = outputs;
}

void BuildReduceAccContractSliceInputGraph(std::shared_ptr<Function>& func,
                                           const MultipleParallelCopyOutAssembleShapes& shapes,
                                           std::vector<std::shared_ptr<LogicalTensor>>& inputs,
                                           std::vector<std::shared_ptr<LogicalTensor>>& outputs)
{
    auto ddrTensors = BuildReduceAccAssembleDdrTensors(func, shapes, inputs);
    auto ddrBigTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapes.ddrBigShape,
                                                                   CreateTestConstIntVector(shapes.ddrBigShape));
    ddrBigTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    for (size_t i = 0; i < kNumThree; i++) {
        auto& contractOp = PassOperationUtils::AddOperation(*func, Opcode::OP_CONTRACT, {ddrTensors[i]},
                                                            {ddrBigTensor});
        contractOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, shapes.assembleOffsets[i]));
    }
    auto ddrReshape = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapes.reshapeShape,
                                                                 CreateTestConstIntVector(shapes.reshapeShape));
    ddrReshape->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto& reshapeOp = PassOperationUtils::AddOperation(*func, Opcode::OP_RESHAPE, {ddrBigTensor}, {ddrReshape});
    reshapeOp.SetAttribute(OP_ATTR_PREFIX + "validShape", std::vector<SymbolicScalar>{kNumThree, kNumEight});
    for (size_t i = 0; i < kNumThree; i++) {
        auto ubOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapes.viewOutShape,
                                                                CreateTestConstIntVector(shapes.viewOutShape));
        ubOut->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
        auto& sliceOp = PassOperationUtils::AddOperation(*func, Opcode::OP_SLICE, {ddrReshape}, {ubOut});
        sliceOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(shapes.viewOffsets[i]));
        auto output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapes.viewOutShape,
                                                                 CreateTestConstIntVector(shapes.viewOutShape));
        PassOperationUtils::AddOperation(*func, Opcode::OP_ADDS, {ubOut}, {output});
        outputs.push_back(output);
    }
    func->inCasts_ = inputs;
    func->outCasts_ = outputs;
}

std::vector<int64_t> GetLargeScaleAssembleOffset(size_t pathIndex, const std::vector<int64_t>& ubShape)
{
    return {static_cast<int64_t>(pathIndex) * ubShape.front(), 0};
}

void BuildLargeScaleCopyOutInputs(std::shared_ptr<Function>& func, size_t pathCount,
                                  const std::vector<int64_t>& ubShape,
                                  const std::shared_ptr<RawTensor>& sharedDdrRawTensor,
                                  std::vector<std::shared_ptr<LogicalTensor>>& inputs,
                                  std::vector<std::shared_ptr<LogicalTensor>>& ddrTensors)
{
    inputs.reserve(pathCount);
    ddrTensors.reserve(pathCount);
    for (size_t i = 0; i < pathCount; ++i) {
        auto input = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, ubShape, CreateTestConstIntVector(ubShape));
        input->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
        inputs.push_back(input);
        auto assembleOffset = GetLargeScaleAssembleOffset(i, ubShape);
        auto ddrTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(sharedDdrRawTensor, assembleOffset, ubShape,
                                                                    CreateTestConstIntVector(ubShape));
        ddrTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
        ddrTensors.push_back(ddrTensor);
        PassOperationUtils::AddOperation(*func, Opcode::OP_COPY_OUT, {input}, {ddrTensor});
    }
}

std::shared_ptr<LogicalTensor> BuildLargeScaleAssembleAndReshape(
    std::shared_ptr<Function>& func, size_t pathCount, const std::vector<int64_t>& ubShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& ddrTensors, int64_t reshapeRows, int64_t reshapeCols)
{
    std::vector<int64_t> ddrBigShape = {reshapeRows * ubShape.front(), ubShape.back()};
    auto ddrBigTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, ddrBigShape,
                                                                   CreateTestConstIntVector(ddrBigShape));
    ddrBigTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    for (size_t i = 0; i < pathCount; ++i) {
        auto assembleOffset = GetLargeScaleAssembleOffset(i, ubShape);
        auto& assembleOp = PassOperationUtils::AddOperation(*func, Opcode::OP_ASSEMBLE, {ddrTensors[i]},
                                                            {ddrBigTensor});
        assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset));
    }
    std::vector<int64_t> reshapeShape = {reshapeRows, reshapeCols};
    auto ddrReshape = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshapeShape,
                                                                 CreateTestConstIntVector(reshapeShape));
    ddrReshape->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto& reshapeOp = PassOperationUtils::AddOperation(*func, Opcode::OP_RESHAPE, {ddrBigTensor}, {ddrReshape});
    reshapeOp.SetAttribute(OP_ATTR_VALID_SHAPE, std::vector<SymbolicScalar>{reshapeRows, reshapeCols});
    return ddrReshape;
}

void BuildLargeScaleViewOutputs(std::shared_ptr<Function>& func, size_t pathCount, int64_t reshapeCols,
                                const std::shared_ptr<LogicalTensor>& ddrReshape,
                                std::vector<std::shared_ptr<LogicalTensor>>& inputs,
                                std::vector<std::shared_ptr<LogicalTensor>>& outputs)
{
    outputs.reserve(pathCount);
    for (size_t i = 0; i < pathCount; ++i) {
        auto ubOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, {1, reshapeCols}, {1, reshapeCols});
        ubOut->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
        auto& viewOp = PassOperationUtils::AddOperation(*func, Opcode::OP_VIEW, {ddrReshape}, {ubOut});
        viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{static_cast<int64_t>(i), 0}));
        auto output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, {1, reshapeCols}, {1, reshapeCols});
        PassOperationUtils::AddOperation(*func, Opcode::OP_ADDS, {ubOut}, {output});
        outputs.push_back(output);
    }
    func->inCasts_ = inputs;
    func->outCasts_ = outputs;
}

void BuildLargeScaleParallelCopyOutAssembleGraph(std::shared_ptr<Function>& func, size_t pathCount,
                                                 const std::vector<int64_t>& ubShape,
                                                 std::vector<std::shared_ptr<LogicalTensor>>& inputs,
                                                 std::vector<std::shared_ptr<LogicalTensor>>& outputs)
{
    int64_t reshapeRows = static_cast<int64_t>(pathCount);
    int64_t reshapeCols = ubShape.front() * ubShape.back();
    auto sharedDdrRawTensor = std::make_shared<RawTensor>(
        DT_FP32, std::vector<int64_t>{reshapeRows * ubShape.front(), ubShape.back()});
    std::vector<std::shared_ptr<LogicalTensor>> ddrTensors;
    BuildLargeScaleCopyOutInputs(func, pathCount, ubShape, sharedDdrRawTensor, inputs, ddrTensors);
    auto ddrReshape = BuildLargeScaleAssembleAndReshape(func, pathCount, ubShape, ddrTensors, reshapeRows, reshapeCols);
    BuildLargeScaleViewOutputs(func, pathCount, reshapeCols, ddrReshape, inputs, outputs);
}

void VerifyMultipleParallelCopyOutAssembleSplit(std::shared_ptr<Function>& func,
                                                const std::vector<int64_t>& expectedInputShape,
                                                const std::vector<int64_t>& expectedOutputShape, int expectedCount)
{
    int reshapeOpCount = 0;
    for (auto& op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            reshapeOpCount++;
            auto reshapeInput = op.GetInputOperand(kSizeZero);
            EXPECT_NE(reshapeInput, nullptr);
            EXPECT_EQ(reshapeInput->shape, expectedInputShape);
            auto reshapeOutput = op.GetOutputOperand(kSizeZero);
            EXPECT_NE(reshapeOutput, nullptr);
            EXPECT_EQ(reshapeOutput->shape, expectedOutputShape);
            EXPECT_EQ(reshapeInput->GetProducers().size(), kSizeOne);
            for (const auto& producer : reshapeInput->GetProducers()) {
                EXPECT_EQ(producer->GetOpcode(), Opcode::OP_ASSEMBLE);
            }
            EXPECT_GE(reshapeOutput->GetConsumers().size(), kSizeOne);
        }
    }
    EXPECT_EQ(reshapeOpCount, expectedCount);
}

TEST_F(TestSplitReshapePass, TestSplitReshapeWithMultipleParallelCopyOutAssemble)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    ASSERT_TRUE(func != nullptr);

    MultipleParallelCopyOutAssembleShapes shapes;
    std::vector<std::shared_ptr<LogicalTensor>> inputs;
    std::vector<std::shared_ptr<LogicalTensor>> outputs;
    BuildMultipleParallelCopyOutAssembleGraph(func, shapes, inputs, outputs);

    int reshapeCountBefore = 0;
    for (auto& op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            reshapeCountBefore++;
        }
    }
    EXPECT_EQ(reshapeCountBefore, kNumOne);

    SplitReshape pass;
    pass.Init();
    EXPECT_EQ(pass.RunOnFunction(*func), SUCCESS);
    VerifyMultipleParallelCopyOutAssembleSplit(func, shapes.ubShape, shapes.viewOutShape, kNumThree);
}

TEST_F(TestSplitReshapePass, TestSkipSplitReshapeWhenAssembleInputProducedByReduceAcc)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestReduceAccReshapeSplit",
                                           "TestReduceAccReshapeSplit", nullptr);
    ASSERT_TRUE(func != nullptr);

    MultipleParallelCopyOutAssembleShapes shapes;
    std::vector<std::shared_ptr<LogicalTensor>> inputs;
    std::vector<std::shared_ptr<LogicalTensor>> outputs;
    BuildReduceAccAssembleInputGraph(func, shapes, inputs, outputs);

    SplitReshape pass;
    pass.Init();
    func->DumpJsonFile("./config/pass/json/"
                       "TestSplitReshapePass_TestSkipSplitReshapeWhenAssembleInputProducedByReduceAcc_before1.json");
    EXPECT_EQ(pass.RunOnFunction(*func), SUCCESS);
    func->DumpJsonFile(
        "./config/pass/json/TestSplitReshapePass_TestSkipSplitReshapeWhenAssembleInputProducedByReduceAcc_after1.json");
    EXPECT_EQ(pass.reshapes_.size(), kSizeZero);
    EXPECT_EQ(pass.assembles_.size(), kSizeZero);

    int reshapeOpCount = 0;
    bool hasReduceAccAssembleInput = false;
    for (auto& op : func->Operations()) {
        if (op.GetOpcode() != Opcode::OP_RESHAPE) {
            continue;
        }
        reshapeOpCount++;
        auto reshapeInput = op.GetInputOperand(kSizeZero);
        ASSERT_NE(reshapeInput, nullptr);
        EXPECT_EQ(reshapeInput->shape, shapes.ddrBigShape);
        EXPECT_EQ(reshapeInput->GetProducers().size(), kSizeThree);
        for (const auto& producer : reshapeInput->GetProducers()) {
            ASSERT_NE(producer, nullptr);
            EXPECT_EQ(producer->GetOpcode(), Opcode::OP_ASSEMBLE);
            auto assembleInput = producer->GetInputOperand(kSizeZero);
            ASSERT_NE(assembleInput, nullptr);
            for (const auto& assembleInputProducer : assembleInput->GetProducers()) {
                ASSERT_NE(assembleInputProducer, nullptr);
                hasReduceAccAssembleInput |= assembleInputProducer->GetOpcode() == Opcode::OP_REDUCE_ACC;
            }
        }
        auto reshapeOutput = op.GetOutputOperand(kSizeZero);
        ASSERT_NE(reshapeOutput, nullptr);
        EXPECT_EQ(reshapeOutput->shape, shapes.reshapeShape);
    }
    EXPECT_TRUE(hasReduceAccAssembleInput);
    EXPECT_EQ(reshapeOpCount, kNumOne);
}

TEST_F(TestSplitReshapePass, TestSkipSplitReshapeWhenContractInputProducedByReduceAcc)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestReduceAccContractReshapeSplit",
                                           "TestReduceAccContractReshapeSplit", nullptr);
    ASSERT_TRUE(func != nullptr);

    MultipleParallelCopyOutAssembleShapes shapes;
    std::vector<std::shared_ptr<LogicalTensor>> inputs;
    std::vector<std::shared_ptr<LogicalTensor>> outputs;
    BuildReduceAccContractSliceInputGraph(func, shapes, inputs, outputs);

    SplitReshape pass;
    pass.Init();
    EXPECT_EQ(pass.RunOnFunction(*func), SUCCESS);
    EXPECT_EQ(pass.reshapes_.size(), kSizeZero);
    EXPECT_EQ(pass.assembles_.size(), kSizeZero);

    int reshapeOpCount = 0;
    bool hasReduceAccContractInput = false;
    for (auto& op : func->Operations()) {
        if (op.GetOpcode() != Opcode::OP_RESHAPE) {
            continue;
        }
        reshapeOpCount++;
        auto reshapeInput = op.GetInputOperand(kSizeZero);
        ASSERT_NE(reshapeInput, nullptr);
        EXPECT_EQ(reshapeInput->shape, shapes.ddrBigShape);
        EXPECT_EQ(reshapeInput->GetProducers().size(), kSizeThree);
        for (const auto& producer : reshapeInput->GetProducers()) {
            ASSERT_NE(producer, nullptr);
            EXPECT_EQ(producer->GetOpcode(), Opcode::OP_CONTRACT);
            auto contractInput = producer->GetInputOperand(kSizeZero);
            ASSERT_NE(contractInput, nullptr);
            for (const auto& contractInputProducer : contractInput->GetProducers()) {
                ASSERT_NE(contractInputProducer, nullptr);
                hasReduceAccContractInput |= contractInputProducer->GetOpcode() == Opcode::OP_REDUCE_ACC;
            }
        }
        auto reshapeOutput = op.GetOutputOperand(kSizeZero);
        ASSERT_NE(reshapeOutput, nullptr);
        EXPECT_EQ(reshapeOutput->shape, shapes.reshapeShape);
    }
    EXPECT_TRUE(hasReduceAccContractInput);
    EXPECT_EQ(reshapeOpCount, kNumOne);
}

TEST_F(TestSplitReshapePass, TestSkipSplitReshapeWhenReshapeOutputContainsNegativeOne)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestNegativeOneReshapeSplit",
                                           "TestNegativeOneReshapeSplit", nullptr);
    ASSERT_TRUE(func != nullptr);

    MultipleParallelCopyOutAssembleShapes shapes;
    shapes.reshapeShape = {kNumThree, -1};
    std::vector<std::shared_ptr<LogicalTensor>> inputs;
    std::vector<std::shared_ptr<LogicalTensor>> outputs;
    BuildMultipleParallelCopyOutAssembleGraph(func, shapes, inputs, outputs);

    SplitReshape pass;
    pass.Init();
    EXPECT_EQ(pass.RunOnFunction(*func), SUCCESS);
    EXPECT_EQ(pass.reshapes_.size(), kSizeZero);
    EXPECT_EQ(pass.assembles_.size(), kSizeZero);

    int reshapeOpCount = 0;
    for (auto& op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            reshapeOpCount++;
        }
    }
    EXPECT_EQ(reshapeOpCount, kNumOne);
}

void CheckReshapeOutputValidShape(Function* func, const std::vector<int64_t>& originValidShape)
{
    ASSERT_NE(func, nullptr);
    int reshapeOpCount = 0;
    for (auto& op : func->Operations()) {
        if (op.GetOpcode() != Opcode::OP_RESHAPE) {
            continue;
        }
        auto reshapeOutput = op.GetOutputOperand(kSizeZero);
        ASSERT_NE(reshapeOutput, nullptr);
        auto outputOffset = reshapeOutput->GetOffset();
        ASSERT_EQ(originValidShape.size(), reshapeOutput->shape.size());
        ASSERT_EQ(originValidShape.size(), outputOffset.size());

        std::vector<SymbolicScalar> attrValidShape;
        EXPECT_TRUE(op.GetAttr(OP_ATTR_VALID_SHAPE, attrValidShape));
        auto tensorValidShape = reshapeOutput->GetDynValidShape();
        ASSERT_EQ(attrValidShape.size(), reshapeOutput->shape.size());
        ASSERT_EQ(tensorValidShape.size(), reshapeOutput->shape.size());
        for (size_t i = 0; i < reshapeOutput->shape.size(); ++i) {
            auto expectedDim = std::min(std::max(originValidShape[i] - outputOffset[i], static_cast<int64_t>(0)),
                                        reshapeOutput->shape[i]);
            EXPECT_EQ(attrValidShape[i].Dump(), std::to_string(expectedDim));
            EXPECT_EQ(tensorValidShape[i].Dump(), std::to_string(expectedDim));
        }
        reshapeOpCount++;
    }
    EXPECT_GT(reshapeOpCount, 0);
}

int CountReshapeOps(Function* func)
{
    if (func == nullptr) {
        return 0;
    }
    int reshapeOpCount = 0;
    for (auto& op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            reshapeOpCount++;
        }
    }
    return reshapeOpCount;
}

/*
exp -> {1,16} -> assemble -> {1,64} -> reshape -> {1,1,64}  -> view -> {1,1,10} -> exp -> {1,1,10}
exp -> {1,16} -> assemble                                   -> view -> {1,1,10} -> exp -> {1,1,10}
exp -> {1,16} -> assemble                                   -> view -> {1,1,10} -> exp -> {1,1,10}
exp -> {1,16} -> assemble                                   -> view -> {1,1,10} -> exp -> {1,1,10}
                                                            -> view -> {1,1,10} -> exp -> {1,1,10}
                                                            -> view -> {1,1,10} -> exp -> {1,1,10}
                                                            -> view -> {1,1,4} -> exp -> {1,1,4}
*/
TEST_F(TestSplitReshapePass, TestSplitReshapeWithTail)
{
    // Define the shape of the Tensors
    std::vector<int64_t> origShape = {kNumOne, kExpSix};
    std::vector<int64_t> reshapeShape = {kNumOne, kNumOne, kExpSix};
    std::vector<int64_t> tiledShape1 = {kNumOne, kExpFour};
    std::vector<int64_t> tiledShape2 = {kNumTwo, kNumOne, kNumTen};

    Tensor input(DT_FP32, origShape, "input");
    Tensor output(DT_FP32, reshapeShape, "output");

    FUNCTION("TestSplitReshapeWithTail")
    {
        TileShape::Current().SetVecTile(tiledShape1);
        Tensor exp = Exp(input);
        Tensor reshape = Reshape(exp, reshapeShape);
        TileShape::Current().SetVecTile(tiledShape2);
        output = Exp(reshape);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_TestSplitReshapeWithTail");

    RunPassStra(*func, PassName::EXPAND_FUNCTION);
    RunPassStra(*func, PassName::SPLIT_RESHAPE);
    EXPECT_GT(CountReshapeOps(func), 1);
    CheckReshapeOutputValidShape(func, reshapeShape);
}

TEST_F(TestSplitReshapePass, TestSplitReshapeWithValidShapeTail)
{
    std::vector<int64_t> origShape = {kNumOne, kExpSix};
    std::vector<int64_t> reshapeShape = {kNumOne, kNumOne, kExpSix};
    std::vector<int64_t> reshapeValidShape = {kNumOne, kNumOne, kNumTwentyFour};
    std::vector<SymbolicScalar> reshapeDynValidShape = {kNumOne, kNumOne, kNumTwentyFour};
    std::vector<int64_t> tiledShape1 = {kNumOne, kExpFour};
    std::vector<int64_t> tiledShape2 = {kNumTwo, kNumOne, kNumTen};
    std::vector<SymbolicScalar> inputValidShape = {kNumOne, kNumTwentyFour};
    std::vector<SymbolicScalar> inputOffset = {kNumZero, kNumZero};

    Tensor input(DT_FP32, origShape, "input");
    Tensor output(DT_FP32, reshapeShape, "output");

    FUNCTION("TestSplitReshapeWithValidShapeTail")
    {
        Tensor validInput = View(input, origShape, inputValidShape, inputOffset);
        TileShape::Current().SetVecTile(tiledShape1);
        Tensor exp = Exp(validInput);
        Tensor reshape = Reshape(exp, reshapeShape, reshapeDynValidShape);
        TileShape::Current().SetVecTile(tiledShape2);
        output = Exp(reshape);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_TestSplitReshapeWithValidShapeTail");

    RunPassStra(*func, PassName::EXPAND_FUNCTION);
    RunPassStra(*func, PassName::SPLIT_RESHAPE);
    EXPECT_GT(CountReshapeOps(func), 1);
    CheckReshapeOutputValidShape(func, reshapeValidShape);
}

TEST_F(TestSplitReshapePass, TestSplitReshapeLargeScaleRunTimeLimit)
{
    constexpr size_t pathCount = 2048;
    constexpr int64_t maxElapsedMs = 3000;
    const std::vector<int64_t> ubShape{kNumTwo, kNumFour};
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestLargeScaleReshapeSplit",
                                           "TestLargeScaleReshapeSplit", nullptr);
    ASSERT_TRUE(func != nullptr);

    std::vector<std::shared_ptr<LogicalTensor>> inputs;
    std::vector<std::shared_ptr<LogicalTensor>> outputs;
    BuildLargeScaleParallelCopyOutAssembleGraph(func, pathCount, ubShape, inputs, outputs);

    SplitReshape pass;
    pass.Init();
    auto start = std::chrono::steady_clock::now();
    EXPECT_EQ(pass.RunOnFunction(*func), SUCCESS);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
    VerifyMultipleParallelCopyOutAssembleSplit(func, ubShape, {kNumOne, kNumEight}, static_cast<int>(pathCount));
    EXPECT_LT(elapsed.count(), maxElapsedMs);
}

TEST_F(TestSplitReshapePass, TestSkipSplitReshapeWithGroupReshapeNoSplitFlag)
{
    std::vector<int64_t> origShape = {kNumOne, kExpSix};
    std::vector<int64_t> reshapeShape = {kNumOne, kNumOne, kExpSix};
    std::vector<int64_t> tiledShape1 = {kNumOne, kExpFour};
    std::vector<int64_t> tiledShape2 = {kNumTwo, kNumOne, kNumTen};

    Tensor input(DT_FP32, origShape, "input");
    Tensor output(DT_FP32, reshapeShape, "output");

    FUNCTION("TestSkipSplitReshapeWithGroupNoSplitFlag")
    {
        TileShape::Current().SetVecTile(tiledShape1);
        Tensor exp = Exp(input);
        Tensor reshape = Reshape(exp, reshapeShape);
        TileShape::Current().SetVecTile(tiledShape2);
        output = Exp(reshape);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_TestSkipSplitReshapeWithGroupNoSplitFlag");

    RunPassStra(*func, PassName::EXPAND_FUNCTION);
    for (auto& op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            op.SetAttribute(OpAttributeKey::groupReshapeNoSplit, true);
        }
    }
    int reshapeCountBefore = CountReshapeOps(func);
    RunPassStra(*func, PassName::SPLIT_RESHAPE);
    EXPECT_EQ(CountReshapeOps(func), reshapeCountBefore);
}

// 标志位在 tensor 图（ExpandFunction 之前）设置，验证展开后标志位不丢失且 SplitReshape 跳过拆分。
TEST_F(TestSplitReshapePass, TestGroupReshapeNoSplitFlagSurviveExpandFunction)
{
    std::vector<int64_t> origShape = {kNumOne, kExpSix};
    std::vector<int64_t> reshapeShape = {kNumOne, kNumOne, kExpSix};
    std::vector<int64_t> tiledShape1 = {kNumOne, kExpFour};
    std::vector<int64_t> tiledShape2 = {kNumTwo, kNumOne, kNumTen};

    Tensor input(DT_FP32, origShape, "input");
    Tensor output(DT_FP32, reshapeShape, "output");

    FUNCTION("TestGroupNoSplitFlagBeforeExpand")
    {
        TileShape::Current().SetVecTile(tiledShape1);
        Tensor exp = Exp(input);
        Tensor reshape = Reshape(exp, reshapeShape);
        TileShape::Current().SetVecTile(tiledShape2);
        output = Exp(reshape);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_TestGroupNoSplitFlagBeforeExpand");

    for (auto& op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            op.SetAttribute(OpAttributeKey::groupReshapeNoSplit, true);
        }
    }
    int reshapeCountBeforeExpand = CountReshapeOps(func);

    RunPassStra(*func, PassName::EXPAND_FUNCTION);

    int flagCount = 0;
    for (auto& op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            EXPECT_TRUE(op.HasAttr(OpAttributeKey::groupReshapeNoSplit));
            EXPECT_TRUE(op.GetBoolAttribute(OpAttributeKey::groupReshapeNoSplit));
            flagCount++;
        }
    }
    EXPECT_EQ(flagCount, reshapeCountBeforeExpand);

    int reshapeCountBefore = CountReshapeOps(func);
    RunPassStra(*func, PassName::SPLIT_RESHAPE);
    EXPECT_EQ(CountReshapeOps(func), reshapeCountBefore);
}

} // namespace tile_fwk
} // namespace npu
