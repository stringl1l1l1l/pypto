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
 * \file test_assign_memory_type_unalign.cpp
 * \brief Unit test for RemoveRedundantReshape pass.
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <fstream>
#include <vector>
#include <string>
#include "computational_graph_builder.h"
#include "interface/function/function.h"
#include "interface/tensor/irbuilder.h"
#include "symbolic_scalar_test_utils.h"
#define private public
#define protected public
#include "passes/tile_graph_pass/graph_constraint/remove_unaligned_reshape_op.h"
#undef private
#undef protected
#include "passes/tile_graph_pass/graph_constraint/pad_local_buffer.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/irbuilder.h"

using namespace npu::tile_fwk;

static const uint16_t kNumOne = 1u;

class TestRemoveUnalignedReshapeOp : public ::testing::Test {
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

struct BranchReshapeTensors {
    std::shared_ptr<LogicalTensor> incast;
    std::shared_ptr<LogicalTensor> copyInTensor1;
    std::shared_ptr<LogicalTensor> copyOutTensor1;
    std::shared_ptr<LogicalTensor> assembleTensor;
    std::shared_ptr<LogicalTensor> reshapeTensor;
    std::shared_ptr<LogicalTensor> copyInTensor2;
    std::shared_ptr<LogicalTensor> copyInTensor3;
    std::shared_ptr<LogicalTensor> outcast1;
    std::shared_ptr<LogicalTensor> outcast2;
};

std::shared_ptr<CopyOpAttribute> CreateUbCopyOutAttr(const std::vector<int64_t>& shape)
{
    return std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({0, 0}), OpImmediate::Specified(shape),
                                             OpImmediate::Specified(shape), std::vector<npu::tile_fwk::OpImmediate>());
}

BranchReshapeTensors CreateBranchReshapeTensors(DataType dataType, const std::vector<int64_t>& inShape,
                                                const std::vector<int64_t>& reshapeShape,
                                                const std::vector<int64_t>& outShape)
{
    BranchReshapeTensors tensors;
    tensors.incast = npu::tile_fwk::IRBuilder().CreateTensorVar(dataType, inShape, CreateTestConstIntVector(inShape));
    tensors.incast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    tensors.copyInTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(dataType, inShape,
                                                                       CreateTestConstIntVector(inShape));
    tensors.copyInTensor1->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    tensors.copyOutTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(dataType, inShape,
                                                                        CreateTestConstIntVector(inShape));
    tensors.copyOutTensor1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    tensors.assembleTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(dataType, inShape,
                                                                        CreateTestConstIntVector(inShape));
    tensors.assembleTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    tensors.assembleTensor->UpdateDynValidShape(
        {CreateTestScalarVar("Input_1_Dim_0"), CreateTestScalarVar("Input_1_Dim_1")});
    tensors.reshapeTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(dataType, reshapeShape,
                                                                       CreateTestConstIntVector(reshapeShape));
    tensors.reshapeTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    tensors.reshapeTensor->UpdateDynValidShape(
        {CreateTestScalarVar("Input_2_Dim_0"), CreateTestScalarVar("Input_2_Dim_1")});
    tensors.copyInTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(dataType, reshapeShape,
                                                                       CreateTestConstIntVector(reshapeShape));
    tensors.copyInTensor2->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    tensors.copyInTensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(dataType, reshapeShape,
                                                                       CreateTestConstIntVector(reshapeShape));
    tensors.copyInTensor3->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    tensors.outcast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(dataType, outShape,
                                                                  CreateTestConstIntVector(outShape));
    tensors.outcast1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    tensors.outcast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(dataType, outShape,
                                                                  CreateTestConstIntVector(outShape));
    tensors.outcast2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    return tensors;
}

void AddBranchCopyOutReshapePath(const std::shared_ptr<Function>& currFunctionPtr, const BranchReshapeTensors& tensors,
                                 const std::shared_ptr<CopyOpAttribute>& copyoutAttr, bool branchFromAssemble)
{
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {tensors.incast}, {tensors.copyInTensor1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {tensors.copyInTensor1},
                                     {tensors.copyOutTensor1},
                                     [&copyoutAttr](Operation& op) { op.SetOpAttribute(copyoutAttr); });
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {tensors.copyOutTensor1},
                                     {tensors.assembleTensor});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {tensors.assembleTensor},
                                     {tensors.reshapeTensor});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {tensors.reshapeTensor},
                                     {tensors.copyInTensor2});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {tensors.copyInTensor2},
                                     {tensors.outcast1});
    auto branchInput = branchFromAssemble ? tensors.assembleTensor : tensors.copyOutTensor1;
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {branchInput}, {tensors.outcast2});
}

struct MultiConsumerReshapeTensors {
    std::shared_ptr<LogicalTensor> incast;
    std::shared_ptr<LogicalTensor> copyinTensor;
    std::shared_ptr<LogicalTensor> copyoutTensor;
    std::shared_ptr<LogicalTensor> reshapeTensor;
    std::shared_ptr<LogicalTensor> addInput2;
    std::shared_ptr<LogicalTensor> addTensor;
    std::shared_ptr<LogicalTensor> mulInput2;
    std::shared_ptr<LogicalTensor> mulTensor;
    std::shared_ptr<LogicalTensor> outcast1;
    std::shared_ptr<LogicalTensor> outcast2;
};

MultiConsumerReshapeTensors BuildMultiConsumerReshapeGraph(const std::shared_ptr<Function>& currFunctionPtr,
                                                           const std::vector<int64_t>& inShape,
                                                           const std::vector<int64_t>& reshapeShape)
{
    MultiConsumerReshapeTensors t;
    t.incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inShape, CreateTestConstIntVector(inShape));
    t.incast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    t.copyinTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inShape, CreateTestConstIntVector(inShape));
    t.copyinTensor->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    t.copyoutTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inShape, CreateTestConstIntVector(inShape));
    t.copyoutTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    t.copyoutTensor->UpdateDynValidShape({CreateTestScalarVar("Input_0_Dim_0"), CreateTestScalarVar("Input_0_Dim_1")});
    t.reshapeTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshapeShape,
                                                                 CreateTestConstIntVector(reshapeShape));
    t.reshapeTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    t.reshapeTensor->UpdateDynValidShape({CreateTestScalarVar("Input_1_Dim_0"), CreateTestScalarVar("Input_1_Dim_1")});
    t.addInput2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshapeShape,
                                                             CreateTestConstIntVector(reshapeShape));
    t.addInput2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    t.addTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshapeShape,
                                                             CreateTestConstIntVector(reshapeShape));
    t.addTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    t.mulInput2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshapeShape,
                                                             CreateTestConstIntVector(reshapeShape));
    t.mulInput2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    t.mulTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshapeShape,
                                                             CreateTestConstIntVector(reshapeShape));
    t.mulTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    t.outcast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshapeShape,
                                                            CreateTestConstIntVector(reshapeShape));
    t.outcast1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    t.outcast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshapeShape,
                                                            CreateTestConstIntVector(reshapeShape));
    t.outcast2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {t.incast}, {t.copyinTensor});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {t.copyinTensor}, {t.copyoutTensor});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {t.copyoutTensor}, {t.reshapeTensor});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ADD, {t.reshapeTensor, t.addInput2}, {t.addTensor});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_MUL, {t.reshapeTensor, t.mulInput2}, {t.mulTensor});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {t.addTensor}, {t.outcast1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {t.mulTensor}, {t.outcast2});

    currFunctionPtr->inCasts_.push_back(t.incast);
    currFunctionPtr->outCasts_.push_back(t.outcast1);
    currFunctionPtr->outCasts_.push_back(t.outcast2);
    return t;
}

struct SiblingCopyInReshapeGraph {
    Operation* copyOutOp;
    Operation* outputCopyInOp;
    Operation* siblingCopyInOp;
};

SiblingCopyInReshapeGraph BuildDDRReshapeGraphWithSiblingCopyIn(const std::shared_ptr<Function>& currFunctionPtr,
                                                                MemoryType siblingCopyInOutputMemType)
{
    std::vector<int64_t> inShape = {8, 8};
    std::vector<int64_t> reshapeShape = {4, 16};
    auto incast = IRBuilder().CreateTensorVar(DT_FP32, inShape, CreateTestConstIntVector(inShape));
    incast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto ubTensor = IRBuilder().CreateTensorVar(DT_FP32, inShape, CreateTestConstIntVector(inShape));
    ubTensor->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto reshapeInput = IRBuilder().CreateTensorVar(DT_FP32, inShape, CreateTestConstIntVector(inShape));
    reshapeInput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    reshapeInput->UpdateDynValidShape({CreateTestScalarVar("Input_0_Dim_0"), CreateTestScalarVar("Input_0_Dim_1")});
    auto reshapeOutput = IRBuilder().CreateTensorVar(DT_FP32, reshapeShape, CreateTestConstIntVector(reshapeShape));
    reshapeOutput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    reshapeOutput->UpdateDynValidShape({CreateTestScalarVar("Input_1_Dim_0"), CreateTestScalarVar("Input_1_Dim_1")});
    auto outputCopyInTensor = IRBuilder().CreateTensorVar(DT_FP32, reshapeShape,
                                                          CreateTestConstIntVector(reshapeShape));
    outputCopyInTensor->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto siblingCopyInTensor = IRBuilder().CreateTensorVar(DT_FP32, inShape, CreateTestConstIntVector(inShape));
    siblingCopyInTensor->SetMemoryTypeBoth(siblingCopyInOutputMemType, true);
    auto outcast = IRBuilder().CreateTensorVar(DT_FP32, reshapeShape, CreateTestConstIntVector(reshapeShape));
    outcast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto siblingOutcast = IRBuilder().CreateTensorVar(DT_FP32, inShape, CreateTestConstIntVector(inShape));
    siblingOutcast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {ubTensor});
    auto& copyOutOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {ubTensor},
                                                       {reshapeInput});
    copyOutOp.SetOpAttribute(
        std::make_shared<CopyOpAttribute>(MemoryType::MEM_UB, OpImmediate::Specified(reshapeInput->GetTensorOffset()),
                                          OpImmediate::Specified(reshapeInput->GetShape()),
                                          OpImmediate::Specified(reshapeInput->GetRawTensor()->GetDynRawShape())));
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {reshapeInput}, {reshapeOutput});
    auto& outputCopyInOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {reshapeOutput},
                                                            {outputCopyInTensor});
    outputCopyInOp.SetOpAttribute(
        std::make_shared<CopyOpAttribute>(OpImmediate::Specified(reshapeOutput->GetTensorOffset()), MemoryType::MEM_UB,
                                          OpImmediate::Specified(reshapeOutput->GetShape()),
                                          OpImmediate::Specified(reshapeOutput->GetRawTensor()->GetDynRawShape())));
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {outputCopyInTensor}, {outcast});
    auto& siblingCopyInOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {reshapeInput},
                                                             {siblingCopyInTensor});
    siblingCopyInOp.SetOpAttribute(
        std::make_shared<CopyOpAttribute>(OpImmediate::Specified(reshapeInput->GetTensorOffset()),
                                          siblingCopyInOutputMemType, OpImmediate::Specified(reshapeInput->GetShape()),
                                          OpImmediate::Specified(reshapeInput->GetRawTensor()->GetDynRawShape())));
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {siblingCopyInTensor}, {siblingOutcast});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);
    currFunctionPtr->outCasts_.push_back(siblingOutcast);
    return {&copyOutOp, &outputCopyInOp, &siblingCopyInOp};
}

inline void ConstructGraphWithCopyInAndReshape(std::shared_ptr<Function>& currFunctionPtr,
                                               const std::vector<int64_t>& shape,
                                               const std::vector<int64_t>& reshape_shape)
{
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    ubTensor1->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshape_shape,
                                                                CreateTestConstIntVector(reshape_shape));
    ubTensor2->SetMemoryTypeBoth(MEM_UB);
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshape_shape,
                                                              CreateTestConstIntVector(reshape_shape));
    outCast->UpdateDynValidShape({CreateTestScalarVar("output_0_Dim_0"), CreateTestScalarVar("output_0_Dim_1")});
    auto& copy_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {incast1}, {ubTensor1});
    auto copyin1Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme,
                                                         std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {OpImmediate(CreateTestScalarVar("Input_0_Dim_0")),
                                                            OpImmediate(CreateTestScalarVar("Input_0_Dim_1"))};
    copyin1Attr->SetToDynValidShape(toValidShape);
    copy_op1.SetOpAttribute(copyin1Attr);
    auto& reshape_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    (void)reshape_op;
    auto& copy_out_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_OUT, {ubTensor2}, {outCast});
    (void)copy_out_op;
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outCast);
}

inline void ConstructGraph1(std::shared_ptr<Function>& currFunctionPtr)
{
    // Prepare the graph
    ConstructGraphWithCopyInAndReshape(currFunctionPtr, {7, 15}, {15, 7});
}

inline void ConstructGraph2(std::shared_ptr<Function>& currFunctionPtr)
{
    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    std::vector<int64_t> reshape_shape = {16, 8};
    std::vector<int64_t> expect_shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    ubTensor1->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshape_shape,
                                                                CreateTestConstIntVector(reshape_shape));
    ubTensor2->SetMemoryTypeBoth(MEM_UB);
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshape_shape,
                                                              CreateTestConstIntVector(reshape_shape));
    outCast->UpdateDynValidShape({CreateTestScalarVar("output_0_Dim_0"), CreateTestScalarVar("output_0_Dim_1")});
    auto& copy_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {incast1}, {ubTensor1});
    auto copyin1Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme,
                                                         std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {OpImmediate(CreateTestScalarVar("Input_0_Dim_0")),
                                                            OpImmediate(CreateTestScalarVar("Input_0_Dim_1"))};
    copyin1Attr->SetToDynValidShape(toValidShape);
    copy_op1.SetOpAttribute(copyin1Attr);
    auto& reshape_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    (void)reshape_op;
    auto& copy_out_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_OUT, {ubTensor2}, {outCast});
    (void)copy_out_op;
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outCast);
}

inline void ConstructGraph3(std::shared_ptr<Function>& currFunctionPtr)
{
    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    std::vector<int64_t> reshape_shape = {16, 8};
    std::vector<int64_t> expect_shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    incast1->SetMemoryTypeBoth(MEM_DEVICE_DDR);
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    ubTensor1->SetMemoryTypeBoth(MEM_UB);
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshape_shape,
                                                              CreateTestConstIntVector(reshape_shape));
    outCast->UpdateDynValidShape({CreateTestScalarVar("output_0_Dim_0"), CreateTestScalarVar("output_0_Dim_1")});
    auto& reshape_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_RESHAPE, {incast1}, {ubTensor1});
    (void)reshape_op;
    auto& copy_out_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_OUT, {ubTensor1}, {outCast});
    (void)copy_out_op;
}

inline void ConstructGraph4(std::shared_ptr<Function>& currFunctionPtr)
{
    // Prepare the graph
    std::vector<int64_t> shape = {64, 1};
    std::vector<int64_t> reshape_shape = {1, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    incast1->SetMemoryTypeBoth(MEM_DEVICE_DDR);
    auto ubTensor0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    ubTensor0->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshape_shape,
                                                                CreateTestConstIntVector(reshape_shape));
    ubTensor1->SetMemoryTypeBoth(MEM_UB);
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshape_shape,
                                                              CreateTestConstIntVector(reshape_shape));
    outCast->UpdateDynValidShape({CreateTestScalarVar("output_0_Dim_0"), CreateTestScalarVar("output_0_Dim_1")});
    auto& copy_in_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {incast1}, {ubTensor0});
    (void)copy_in_op;
    auto& reshape_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_RESHAPE, {ubTensor0}, {ubTensor1});
    (void)reshape_op;
    auto& copy_out_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_OUT, {ubTensor1}, {outCast});
    (void)copy_out_op;
}
} // namespace
/*
before:
    copyin
    [7,15]
     |
    reshape
    [15,7]
     |
    copyout
    [15,7]

after:
    copyin
    [7,15]
      |
    copyout
    [7,15]
      |
    reshape
    [15,7]
      |
    copyin
    [15,8]
      |
    copyout
    [15,7]
*/
TEST_F(TestRemoveUnalignedReshapeOp, reshaped_padded_ub)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPadLocalBuffer",
                                                      "TestPadLocalBuffer", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {7, 15};
    std::vector<int64_t> reshape_shape = {15, 7};
    std::vector<int64_t> expect_shape = {7, 16};
    ConstructGraph1(currFunctionPtr);
    PadLocalBuffer padLocalBufferTest;
    padLocalBufferTest.RunOnFunction(*currFunctionPtr);
    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr);
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            for (auto& out : op.oOperand) {
                if (out->oriShape == reshape_shape) {
                    EXPECT_EQ(out->shape, reshape_shape);
                    EXPECT_EQ(out->tensor->rawshape, reshape_shape);
                }
                EXPECT_EQ(out->GetConsumers().size(), 1);
                auto consumer = *(out->GetConsumers().begin());
                EXPECT_EQ(consumer->GetOpcode(), Opcode::OP_COPY_IN);
            }
            for (auto& in : op.iOperand) {
                EXPECT_EQ(in->GetProducers().size(), 1);
                if (in->oriShape == shape) {
                    EXPECT_EQ(in->shape, shape);
                    EXPECT_EQ(in->tensor->rawshape, shape);
                }
                auto producer = *(in->GetProducers().begin());
                EXPECT_EQ(producer->GetOpcode(), Opcode::OP_COPY_OUT);
            }
        }
    }
}

/*
before:
    copyin
    [8,16]
     |
    reshape
    [16,8]
     |
    copyout
    [16,8]

after:
    copyin
    [8,16]
     |
    reshape
    [16,8]
     |
    copyout
    [16,8]
*/
TEST_F(TestRemoveUnalignedReshapeOp, reshaped_unpadded_ub)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPadLocalBuffer",
                                                      "TestPadLocalBuffer", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {8, 16};
    std::vector<int64_t> reshape_shape = {16, 8};
    std::vector<int64_t> expect_shape = {8, 16};
    ConstructGraph2(currFunctionPtr);
    PadLocalBuffer padLocalBufferTest;
    padLocalBufferTest.RunOnFunction(*currFunctionPtr);
    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr);
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            for (auto& in : op.iOperand) {
                EXPECT_EQ(in->GetProducers().size(), 1);
                auto producer = *(in->GetProducers().begin());
                EXPECT_NE(producer->GetOpcode(), Opcode::OP_COPY_OUT);
                if (in->oriShape == shape) {
                    EXPECT_EQ(in->shape, shape);
                    EXPECT_EQ(in->tensor->rawshape, shape);
                }
            }
            for (auto& out : op.oOperand) {
                auto consumer = *(out->GetConsumers().begin());
                EXPECT_EQ(out->GetConsumers().size(), 1);
                if (out->oriShape == reshape_shape) {
                    EXPECT_EQ(out->shape, reshape_shape);
                    EXPECT_EQ(out->tensor->rawshape, reshape_shape);
                }
                EXPECT_NE(consumer->GetOpcode(), Opcode::OP_COPY_IN);
            }
        }
    }
}

/*
before:
    incast(gm)
    [8,16]
     |
    reshape
    [16,8]
     |
    copyout
    [16,8]

after:
    incast(gm)
    [8,16]
     |
    reshape
    [16,8]
     |
    copyout
    [16,8]
*/
TEST_F(TestRemoveUnalignedReshapeOp, reshaped_unpadded_ub_gm)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPadLocalBuffer",
                                                      "TestPadLocalBuffer", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {8, 16};
    std::vector<int64_t> reshape_shape = {16, 8};
    std::vector<int64_t> expect_shape = {8, 16};
    ConstructGraph3(currFunctionPtr);
    PadLocalBuffer padLocalBufferTest;
    padLocalBufferTest.RunOnFunction(*currFunctionPtr);
    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr);
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            for (auto& out : op.oOperand) {
                EXPECT_EQ(out->GetConsumers().size(), 1);
                auto consumer = *(out->GetConsumers().begin());
                EXPECT_NE(consumer->GetOpcode(), Opcode::OP_COPY_IN);
                if (out->oriShape == reshape_shape) {
                    EXPECT_EQ(out->shape, reshape_shape);
                    EXPECT_EQ(out->tensor->rawshape, reshape_shape);
                }
            }
        }
    }
}

/*
问题用例
before:
    copyin
    [64,1]
     |
    reshape
    [1, 64]
     |
    copyout
    [1, 64]

after:
    copyin
    [64, 1]
      |
    copyout
    [64, 1]
      |
    reshape
    [1, 64]
      |
    copyin
    [1, 64]
      |
    copyout
    [1, 64]
*/
TEST_F(TestRemoveUnalignedReshapeOp, reshaped_unpadded_ub_gm_last_dim_1)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPadLocalBuffer",
                                                      "TestPadLocalBuffer", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {64, 1};
    std::vector<int64_t> reshape_shape = {1, 64};
    ConstructGraph4(currFunctionPtr);
    PadLocalBuffer padLocalBufferTest;
    padLocalBufferTest.RunOnFunction(*currFunctionPtr);
    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr);
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            for (auto& in : op.iOperand) {
                EXPECT_EQ(in->GetProducers().size(), 1);
                auto producer = *(in->GetProducers().begin());
                EXPECT_EQ(producer->GetOpcode(), Opcode::OP_COPY_OUT);
                if (in->oriShape == shape) {
                    EXPECT_EQ(in->shape, shape);
                    EXPECT_EQ(in->tensor->rawshape, shape);
                }
            }
            for (auto& out : op.oOperand) {
                EXPECT_EQ(out->GetConsumers().size(), 1);
                auto consumer = *(out->GetConsumers().begin());
                EXPECT_EQ(consumer->GetOpcode(), Opcode::OP_COPY_IN);
                if (out->oriShape == reshape_shape) {
                    EXPECT_EQ(out->shape, reshape_shape);
                    EXPECT_EQ(out->tensor->rawshape, reshape_shape);
                }
            }
        }
    }
}

// in - COPYIN - COPYOUT - RESHAPE - COPYIN - COPYOUT - out
//                                 - COPYIN - COPYOUT - out

// in - COPYIN - COPYOUT - COPYIN - RESHAPECOPYOUT - RESHAPE - RESHAPECOPYIN - COPYOUT - COPYIN - COPYOUT - out
//                                                           - RESHAPECOPYIN - COPYOUT - COPYIN - COPYOUT - out
TEST_F(TestRemoveUnalignedReshapeOp, TestCopyToReshapeCopyOnL1)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestCopyToReshapeCopy",
                                                      "TestCopyToReshapeCopy", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inShape = {4, 4};
    std::vector<int64_t> reshapeShape = {2, 8};
    std::vector<int64_t> outShape = {4, 8};
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inShape, CreateTestConstIntVector(inShape));
    auto copyinTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inShape,
                                                                    CreateTestConstIntVector(inShape));
    copyinTensor1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto copyoutTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inShape,
                                                                     CreateTestConstIntVector(inShape));
    copyoutTensor1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape1 = {CreateTestScalarVar("Input_0_Dim_0"),
                                               CreateTestScalarVar("Input_0_Dim_1")};
    copyoutTensor1->UpdateDynValidShape(validShape1);
    auto reshapeTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshapeShape,
                                                                    CreateTestConstIntVector(reshapeShape));
    reshapeTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape2 = {CreateTestScalarVar("Input_1_Dim_0"),
                                               CreateTestScalarVar("Input_1_Dim_1")};
    reshapeTensor->UpdateDynValidShape(validShape2);
    auto copyinTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshapeShape,
                                                                    CreateTestConstIntVector(reshapeShape));
    copyinTensor2->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto copyinTensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshapeShape,
                                                                    CreateTestConstIntVector(reshapeShape));
    copyinTensor3->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outShape, CreateTestConstIntVector(outShape));
    outcast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {copyinTensor1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copyinTensor1}, {copyoutTensor1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {copyoutTensor1}, {reshapeTensor});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {reshapeTensor}, {copyinTensor2});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {reshapeTensor}, {copyinTensor3});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copyinTensor2}, {outcast});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copyinTensor3}, {outcast});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    int curSize = currFunctionPtr->Operations().size();
    EXPECT_EQ(removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(currFunctionPtr->Operations().size(), curSize);
    int l1ReshapeCopyInCount = 0;
    for (const auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_L1_RESHAPE_COPY_IN) {
            l1ReshapeCopyInCount++;
        }
    }
    EXPECT_EQ(l1ReshapeCopyInCount, 2);
}

// in - COPYIN - COPYOUT - RESHAPE - COPYIN - COPYOUT - out
//                                 - COPYIN - COPYOUT - out

// in - COPYIN - RESHAPECOPYOUT - RESHAPE - RESHAPECOPYIN - COPYOUT - out
//                                        - RESHAPECOPYIN - COPYOUT - out
TEST_F(TestRemoveUnalignedReshapeOp, TestCopyToReshapeCopyOnUB)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestCopyToReshapeCopy",
                                                      "TestCopyToReshapeCopy", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inShape = {4, 4};
    std::vector<int64_t> reshapeShape = {2, 8};
    std::vector<int64_t> outShape = {4, 8};
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inShape, CreateTestConstIntVector(inShape));
    auto copyin_Tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inShape,
                                                                     CreateTestConstIntVector(inShape));
    copyin_Tensor1->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto copyout_Tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inShape,
                                                                      CreateTestConstIntVector(inShape));
    copyout_Tensor1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape1 = {CreateTestScalarVar("Input_00_Dim_0"),
                                               CreateTestScalarVar("Input_00_Dim_1")};
    copyout_Tensor1->UpdateDynValidShape(validShape1);
    auto reshape_Tensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshapeShape,
                                                                     CreateTestConstIntVector(reshapeShape));
    reshape_Tensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape2 = {CreateTestScalarVar("Input_01_Dim_0"),
                                               CreateTestScalarVar("Input_01_Dim_1")};
    reshape_Tensor->UpdateDynValidShape(validShape2);
    auto copyin_Tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshapeShape,
                                                                     CreateTestConstIntVector(reshapeShape));
    copyin_Tensor2->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto copyin_Tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshapeShape,
                                                                     CreateTestConstIntVector(reshapeShape));
    copyin_Tensor3->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outShape, CreateTestConstIntVector(outShape));
    outcast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {copyin_Tensor1});
    auto& copyoutOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copyin_Tensor1},
                                                       {copyout_Tensor1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {copyout_Tensor1}, {reshape_Tensor});
    auto& copyinOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {reshape_Tensor},
                                                       {copyin_Tensor2});
    auto& copyinOp2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {reshape_Tensor},
                                                       {copyin_Tensor3});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copyin_Tensor2}, {outcast});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copyin_Tensor3}, {outcast});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    int curOpSize = currFunctionPtr->Operations().size();
    EXPECT_EQ(removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(copyoutOp.GetOpcode() == Opcode::OP_RESHAPE_COPY_OUT, true);
    EXPECT_EQ(copyinOp1.GetOpcode() == Opcode::OP_RESHAPE_COPY_IN, true);
    EXPECT_EQ(copyinOp2.GetOpcode() == Opcode::OP_RESHAPE_COPY_IN, true);
    EXPECT_EQ(currFunctionPtr->Operations().size(), curOpSize);
}

// in - COPYIN - COPYOUT - RESHAPE - COPYIN - COPYOUT - out
//    - COPYIN - COPYOUT           - COPYIN - COPYOUT - out

// 不变
TEST_F(TestRemoveUnalignedReshapeOp, TestCopyToReshapeBeforeMultCopyOutOnL1)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestCopyToReshapeCopy",
                                                      "TestCopyToReshapeCopy", nullptr);
    // Prepare the graph
    std::vector<int64_t> copyoutShape = {2, 4};
    std::vector<int64_t> reshapeShape = {4, 2};
    std::vector<int64_t> inShape = {2, 2};
    std::vector<int64_t> outShape = {4, 4};

    auto copyin_Tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inShape,
                                                                     CreateTestConstIntVector(inShape));
    copyin_Tensor1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inShape, CreateTestConstIntVector(inShape));
    auto copyout_Tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, copyoutShape,
                                                                      CreateTestConstIntVector(copyoutShape));
    copyout_Tensor1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape1 = {CreateTestScalarVar("Input_00_Dim_0"),
                                               CreateTestScalarVar("Input_00_Dim_1")};
    copyout_Tensor1->UpdateDynValidShape(validShape1);
    auto copyin_Tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inShape,
                                                                     CreateTestConstIntVector(inShape));
    copyin_Tensor2->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto reshape_Tensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshapeShape,
                                                                     CreateTestConstIntVector(reshapeShape));
    reshape_Tensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape3 = {CreateTestScalarVar("Input_02_Dim_0"),
                                               CreateTestScalarVar("Input_02_Dim_1")};
    reshape_Tensor->UpdateDynValidShape(validShape3);
    auto copyin_Tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshapeShape,
                                                                     CreateTestConstIntVector(reshapeShape));
    copyin_Tensor3->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto copyin_Tensor4 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshapeShape,
                                                                     CreateTestConstIntVector(reshapeShape));
    copyin_Tensor4->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outShape, CreateTestConstIntVector(outShape));
    outcast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {copyin_Tensor2});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {copyin_Tensor1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {copyout_Tensor1}, {reshape_Tensor});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {reshape_Tensor}, {copyin_Tensor3});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {reshape_Tensor}, {copyin_Tensor4});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copyin_Tensor1}, {copyout_Tensor1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copyin_Tensor2}, {copyout_Tensor1});

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copyin_Tensor4}, {outcast});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copyin_Tensor3}, {outcast});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    int curOpSize = currFunctionPtr->Operations().size();
    EXPECT_EQ(removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr), SUCCESS);
    int expOpSize = curOpSize;
    EXPECT_EQ(currFunctionPtr->Operations().size(), expOpSize);
    int l1ReshapeCopyInCount = 0;
    for (const auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_L1_RESHAPE_COPY_IN) {
            l1ReshapeCopyInCount++;
        }
    }
    EXPECT_EQ(l1ReshapeCopyInCount, 2);
}

// in - COPYIN - COPYOUT - RESHAPE - COPYIN   - COPYOUT - out1
//                                 - ASSEMBLE - out2

// in - COPYIN - RESHAPECOPYOUT - RESHAPE - RESHAPECOPYIN - COPYOUT - out1
//                                        - RESHAPECOPYIN - COPYOUT - ASSEMBLE - out2
TEST_F(TestRemoveUnalignedReshapeOp, TestCopyToReshapeConsumerAssembleOnUB)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestCopyToReshapeCopy",
                                                      "TestCopyToReshapeCopy", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inShape = {8, 8};
    std::vector<int64_t> reshapeShape = {4, 16};
    std::vector<int64_t> outShape = {8, 16};
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inShape, CreateTestConstIntVector(inShape));
    auto copyInTensor_1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inShape,
                                                                     CreateTestConstIntVector(inShape));
    copyInTensor_1->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto copyOutTensor_1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inShape,
                                                                      CreateTestConstIntVector(inShape));
    copyOutTensor_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape_1 = {CreateTestScalarVar("Input_0_Dim_0"),
                                                CreateTestScalarVar("Input_0_Dim_1")};
    copyOutTensor_1->UpdateDynValidShape(validShape_1);
    auto reshapeTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshapeShape,
                                                                    CreateTestConstIntVector(reshapeShape));
    reshapeTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape_2 = {CreateTestScalarVar("Input_1_Dim_0"),
                                                CreateTestScalarVar("Input_1_Dim_1")};
    reshapeTensor->UpdateDynValidShape(validShape_2);
    auto copyInTensor_2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshapeShape,
                                                                     CreateTestConstIntVector(reshapeShape));
    copyInTensor_2->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto outcast_1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outShape, CreateTestConstIntVector(outShape));
    outcast_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto outcast_2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outShape, CreateTestConstIntVector(outShape));
    outcast_2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {copyInTensor_1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copyInTensor_1}, {copyOutTensor_1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {copyOutTensor_1}, {reshapeTensor});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {reshapeTensor}, {copyInTensor_2});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copyInTensor_2}, {outcast_1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {reshapeTensor}, {outcast_2});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast_1);
    currFunctionPtr->outCasts_.push_back(outcast_2);

    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    int curSize = currFunctionPtr->Operations().size();
    EXPECT_EQ(removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(currFunctionPtr->Operations().size(), curSize + 2);
}
// in - COPYIN - COPYOUT - ASSEMBLE - RESHAPE - COPYIN - COPYOUT - out1
//                       - ASSEMBLE - out2
TEST_F(TestRemoveUnalignedReshapeOp, TestBranchBetweenCopyOutReshape1)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestBranchBetweenCopyOutReshape1",
                                                      "TestBranchBetweenCopyOutReshape1", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> inShape = {8, 8};
    std::vector<int64_t> reshapeShape = {4, 16};
    std::vector<int64_t> outShape = {8, 16};
    DataType dataType = DT_FP32;
    auto tensors = CreateBranchReshapeTensors(dataType, inShape, reshapeShape, outShape);
    tensors.copyInTensor1->UpdateDynValidShape(
        {CreateTestScalarVar("Input_0_Dim_0"), CreateTestScalarVar("Input_0_Dim_1")});
    auto copyoutAttr = CreateUbCopyOutAttr(inShape);
    AddBranchCopyOutReshapePath(currFunctionPtr, tensors, copyoutAttr, false);

    currFunctionPtr->inCasts_.push_back(tensors.incast);
    currFunctionPtr->outCasts_.push_back(tensors.outcast1);
    currFunctionPtr->outCasts_.push_back(tensors.outcast2);

    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    int curSize = currFunctionPtr->Operations().size();
    EXPECT_EQ(removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr), SUCCESS);
    int expSize = curSize + 1;
    EXPECT_EQ(currFunctionPtr->Operations().size(), expSize);
}

/*
before:
    inCast(DDR) - COPYIN(UB) - COPYOUT(DDR) - RESHAPE(DDR) - ADD(DDR) - outCast(DDR)

after:
    inCast(DDR) - COPYIN(UB) - RESHAPECOPYOUT(DDR) - RESHAPE(DDR) - RESHAPECOPYIN(UB) - COPYOUT(DDR) - ADD(DDR) -
outCast(DDR)

Test HandleNoCopyInConsumer: reshape output has no COPY_IN consumer, only ADD consumer.
*/
TEST_F(TestRemoveUnalignedReshapeOp, TestHandleNoCopyInConsumerWithAddOp)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestHandleNoCopyInConsumer",
                                                      "TestHandleNoCopyInConsumer", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> inShape = {8, 8};
    std::vector<int64_t> reshapeShape = {4, 16};
    std::vector<int64_t> outShape = {4, 16};
    DataType dataType = DT_FP32;

    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(dataType, inShape, CreateTestConstIntVector(inShape));
    incast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto copyinTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(dataType, inShape,
                                                                   CreateTestConstIntVector(inShape));
    copyinTensor->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto copyoutTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(dataType, inShape,
                                                                    CreateTestConstIntVector(inShape));
    copyoutTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape1 = {CreateTestScalarVar("Input_0_Dim_0"),
                                               CreateTestScalarVar("Input_0_Dim_1")};
    copyoutTensor->UpdateDynValidShape(validShape1);
    auto reshapeTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(dataType, reshapeShape,
                                                                    CreateTestConstIntVector(reshapeShape));
    reshapeTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape2 = {CreateTestScalarVar("Input_1_Dim_0"),
                                               CreateTestScalarVar("Input_1_Dim_1")};
    reshapeTensor->UpdateDynValidShape(validShape2);
    auto addInput2 = npu::tile_fwk::IRBuilder().CreateTensorVar(dataType, reshapeShape,
                                                                CreateTestConstIntVector(reshapeShape));
    addInput2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto addTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(dataType, reshapeShape,
                                                                CreateTestConstIntVector(reshapeShape));
    addTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(dataType, outShape, CreateTestConstIntVector(outShape));
    outcast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {copyinTensor});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copyinTensor}, {copyoutTensor});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {copyoutTensor}, {reshapeTensor});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ADD, {reshapeTensor, addInput2}, {addTensor});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {addTensor}, {outcast});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    int curSize = currFunctionPtr->Operations().size();
    EXPECT_EQ(removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr), SUCCESS);
    int expSize = curSize + 2;
    EXPECT_EQ(currFunctionPtr->Operations().size(), expSize);

    uint32_t reshapeCopyinNum = 0;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE_COPY_IN) {
            reshapeCopyinNum++;
        }
    }
    EXPECT_EQ(reshapeCopyinNum, kNumOne);
}

/*
before:
    inCast(DDR) - COPYIN(UB) - COPYOUT(DDR) - RESHAPE(DDR) - ADD(DDR) - outCast1(DDR)
                                            - MUL(DDR) - outCast2(DDR)

after:
    inCast(DDR) - COPYIN(UB) - RESHAPECOPYOUT(DDR) - RESHAPE(DDR) - RESHAPECOPYIN(UB) - COPYOUT(DDR) - ADD(DDR) -
outCast1(DDR)
                                                                                                     - MUL(DDR) -
outCast2(DDR)

Test HandleNoCopyInConsumer: reshape output has no COPY_IN consumer, has multiple other consumers (ADD, MUL).
*/
TEST_F(TestRemoveUnalignedReshapeOp, TestHandleNoCopyInConsumerWithMultiOps)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestHandleNoCopyInConsumerMulti",
                                                      "TestHandleNoCopyInConsumerMulti", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> inShape = {8, 8};
    std::vector<int64_t> reshapeShape = {4, 16};

    auto tensors = BuildMultiConsumerReshapeGraph(currFunctionPtr, inShape, reshapeShape);

    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    EXPECT_EQ(removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr), SUCCESS);

    uint32_t reshapeCopyinNum = 0;
    uint32_t reshapeCopyoutNum = 0;

    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE_COPY_IN) {
            reshapeCopyinNum++;
        }
        if (op.GetOpcode() == Opcode::OP_RESHAPE_COPY_OUT) {
            reshapeCopyoutNum++;
        }
    }
    EXPECT_EQ(reshapeCopyoutNum, kNumOne);
    EXPECT_EQ(reshapeCopyinNum, kNumOne);
}

// in - COPYIN - COPYOUT - ASSEMBLE - RESHAPE - COPYIN - COPYOUT - out1
//                                  - ASSEMBLE - out2
TEST_F(TestRemoveUnalignedReshapeOp, TestBranchBetweenCopyOutReshape2)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestBranchBetweenCopyOutReshape2",
                                                      "TestBranchBetweenCopyOutReshape2", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> inShape = {8, 8};
    std::vector<int64_t> reshapeShape = {4, 16};
    std::vector<int64_t> outShape = {8, 16};
    auto tensors = CreateBranchReshapeTensors(DT_FP32, inShape, reshapeShape, outShape);
    tensors.assembleTensor->UpdateDynValidShape(
        {CreateTestScalarVar("Input_0_Dim_0"), CreateTestScalarVar("Input_0_Dim_1")});
    tensors.reshapeTensor->UpdateDynValidShape(
        {CreateTestScalarVar("Input_1_Dim_0"), CreateTestScalarVar("Input_1_Dim_1")});
    auto copyoutAttr = CreateUbCopyOutAttr(inShape);
    AddBranchCopyOutReshapePath(currFunctionPtr, tensors, copyoutAttr, true);

    currFunctionPtr->inCasts_.push_back(tensors.incast);
    currFunctionPtr->outCasts_.push_back(tensors.outcast1);
    currFunctionPtr->outCasts_.push_back(tensors.outcast2);

    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    int curSize = currFunctionPtr->Operations().size();
    EXPECT_EQ(removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr), SUCCESS);
    int expSize = curSize + 2;
    EXPECT_EQ(currFunctionPtr->Operations().size(), expSize);
}

inline void CheckTestMutiProducerBetweenCopyOutReshape(std::shared_ptr<Function> currFunctionPtr)
{
    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    int curSize = currFunctionPtr->Operations().size();
    EXPECT_EQ(removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr), SUCCESS);
    int expSize = curSize + 1;
    EXPECT_EQ(currFunctionPtr->Operations().size(), expSize);
}
// in1                    - ASSEMBLE \*
// in2 - COPYIN - COPYOUT - ASSEMBLE - RESHAPE - COPYIN - COPYOUT - out1
//                        - ASSEMBLE - out2

TEST_F(TestRemoveUnalignedReshapeOp, TestMutiProducerBetweenCopyOutReshape)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestMutiProducerBetweenCopyOutReshape",
                                                      "TestMutiProducerBetweenCopyOutReshape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inShape = {8, 8};
    std::vector<int64_t> reshapeShape = {4, 16};
    std::vector<int64_t> outShape = {8, 16};
    DataType dataTypeFp32 = DT_FP32;
    auto incast_1 = npu::tile_fwk::IRBuilder().CreateTensorVar(dataTypeFp32, inShape,
                                                               CreateTestConstIntVector(inShape));
    incast_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto incast_2 = npu::tile_fwk::IRBuilder().CreateTensorVar(dataTypeFp32, inShape,
                                                               CreateTestConstIntVector(inShape));
    incast_2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto copyInTensor_1 = npu::tile_fwk::IRBuilder().CreateTensorVar(dataTypeFp32, inShape,
                                                                     CreateTestConstIntVector(inShape));
    copyInTensor_1->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto copyOutTensor_1 = npu::tile_fwk::IRBuilder().CreateTensorVar(dataTypeFp32, inShape,
                                                                      CreateTestConstIntVector(inShape));
    copyOutTensor_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto assembleTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(dataTypeFp32, inShape,
                                                                     CreateTestConstIntVector(inShape));
    assembleTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape_1 = {CreateTestScalarVar("Input_0_Dim_0"),
                                                CreateTestScalarVar("Input_0_Dim_1")};
    assembleTensor->UpdateDynValidShape(validShape_1);
    auto reshapeTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(dataTypeFp32, reshapeShape,
                                                                    CreateTestConstIntVector(reshapeShape));
    reshapeTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape_2 = {CreateTestScalarVar("Input_1_Dim_0"),
                                                CreateTestScalarVar("Input_1_Dim_1")};
    reshapeTensor->UpdateDynValidShape(validShape_2);
    auto copyInTensor_2 = npu::tile_fwk::IRBuilder().CreateTensorVar(dataTypeFp32, reshapeShape,
                                                                     CreateTestConstIntVector(reshapeShape));
    copyInTensor_2->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto copyInTensor_3 = npu::tile_fwk::IRBuilder().CreateTensorVar(dataTypeFp32, reshapeShape,
                                                                     CreateTestConstIntVector(reshapeShape));
    copyInTensor_3->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto outcast_1 = npu::tile_fwk::IRBuilder().CreateTensorVar(dataTypeFp32, outShape,
                                                                CreateTestConstIntVector(outShape));
    outcast_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto outcast_2 = npu::tile_fwk::IRBuilder().CreateTensorVar(dataTypeFp32, outShape,
                                                                CreateTestConstIntVector(outShape));
    outcast_2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast_1}, {copyInTensor_1});
    auto copyoutAttr = CreateUbCopyOutAttr(inShape);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copyInTensor_1}, {copyOutTensor_1},
                                     [&copyoutAttr](Operation& op) { op.SetOpAttribute(copyoutAttr); });
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {copyOutTensor_1}, {assembleTensor});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {incast_2}, {assembleTensor});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {assembleTensor}, {reshapeTensor});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {reshapeTensor}, {copyInTensor_2});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copyInTensor_2}, {outcast_1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {copyOutTensor_1}, {outcast_2});

    currFunctionPtr->inCasts_.push_back(incast_1);
    currFunctionPtr->inCasts_.push_back(incast_2);
    currFunctionPtr->outCasts_.push_back(outcast_1);
    currFunctionPtr->outCasts_.push_back(outcast_2);
    CheckTestMutiProducerBetweenCopyOutReshape(currFunctionPtr);
}
inline void CheckTestMutiConsumersBetweenCopyOutReshape(std::shared_ptr<Function> currFunctionPtr)
{
    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    int curSize = currFunctionPtr->Operations().size();
    EXPECT_EQ(removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr), SUCCESS);
    int expSize = curSize + 2;
    EXPECT_EQ(currFunctionPtr->Operations().size(), expSize);
}

// in - COPYIN - COPYOUT - ASSEMBLE - RESHAPE - COPYIN - COPYOUT - out1
//                                  - ASSEMBLE - out3
//                       - ASSEMBLE - out2

TEST_F(TestRemoveUnalignedReshapeOp, TestMutiConsumersBetweenCopyOutReshape)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestMutiProducerBetweenCopyOutReshape",
                                                      "TestMutiProducerBetweenCopyOutReshape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inShape = {8, 8};
    std::vector<int64_t> reshapeShape = {4, 16};
    std::vector<int64_t> outShape = {8, 16};
    DataType dataTypeFP32 = DT_FP32;
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(dataTypeFP32, inShape, CreateTestConstIntVector(inShape));
    incast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto copyInTensor_1 = npu::tile_fwk::IRBuilder().CreateTensorVar(dataTypeFP32, inShape,
                                                                     CreateTestConstIntVector(inShape));
    copyInTensor_1->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto copyOutTensor_1 = npu::tile_fwk::IRBuilder().CreateTensorVar(dataTypeFP32, inShape,
                                                                      CreateTestConstIntVector(inShape));
    copyOutTensor_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto assembleTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(dataTypeFP32, inShape,
                                                                     CreateTestConstIntVector(inShape));
    assembleTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape_1 = {CreateTestScalarVar("Input_0_Dim_0"),
                                                CreateTestScalarVar("Input_0_Dim_1")};
    assembleTensor->UpdateDynValidShape(validShape_1);
    auto reshapeTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(dataTypeFP32, reshapeShape,
                                                                    CreateTestConstIntVector(reshapeShape));
    reshapeTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape_2 = {CreateTestScalarVar("Input_1_Dim_0"),
                                                CreateTestScalarVar("Input_1_Dim_1")};
    reshapeTensor->UpdateDynValidShape(validShape_2);
    auto copyInTensor_2 = npu::tile_fwk::IRBuilder().CreateTensorVar(dataTypeFP32, reshapeShape,
                                                                     CreateTestConstIntVector(reshapeShape));
    copyInTensor_2->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto copyInTensor_3 = npu::tile_fwk::IRBuilder().CreateTensorVar(dataTypeFP32, reshapeShape,
                                                                     CreateTestConstIntVector(reshapeShape));
    copyInTensor_3->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto outcast_1 = npu::tile_fwk::IRBuilder().CreateTensorVar(dataTypeFP32, outShape,
                                                                CreateTestConstIntVector(outShape));
    outcast_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto outcast_2 = npu::tile_fwk::IRBuilder().CreateTensorVar(dataTypeFP32, outShape,
                                                                CreateTestConstIntVector(outShape));
    outcast_2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto outcast_3 = npu::tile_fwk::IRBuilder().CreateTensorVar(dataTypeFP32, inShape,
                                                                CreateTestConstIntVector(inShape));
    outcast_3->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {copyInTensor_1});
    auto copyoutAttr = CreateUbCopyOutAttr(inShape);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copyInTensor_1}, {copyOutTensor_1},
                                     [&copyoutAttr](Operation& op) { op.SetOpAttribute(copyoutAttr); });
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {copyOutTensor_1}, {assembleTensor});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {assembleTensor}, {reshapeTensor});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {reshapeTensor}, {copyInTensor_2});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copyInTensor_2}, {outcast_1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {copyOutTensor_1}, {outcast_2});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {assembleTensor}, {outcast_3});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast_1);
    currFunctionPtr->outCasts_.push_back(outcast_2);
    currFunctionPtr->outCasts_.push_back(outcast_3);
    CheckTestMutiConsumersBetweenCopyOutReshape(currFunctionPtr);
}

// in - COPYIN - COPYOUT - RESHAPE - COPYIN - COPYOUT - out
//                                 - COPYIN - COPYOUT - out

// 搬运超过了UB限额，不进行修改
TEST_F(TestRemoveUnalignedReshapeOp, TestCopyToReshapeCopyOnL1OverUB)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestCopyToReshapeCopyOnL1OverUB",
                                                      "TestCopyToReshapeCopyOnL1OverUB", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inShapeOverUB = {2024, 32};
    std::vector<int64_t> reshapeShapeOverUB = {1024, 64};
    std::vector<int64_t> outShapeOverUB = {2024, 64};
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inShapeOverUB,
                                                             CreateTestConstIntVector(inShapeOverUB));
    auto copyinTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inShapeOverUB,
                                                                    CreateTestConstIntVector(inShapeOverUB));
    copyinTensor1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto copyoutTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inShapeOverUB,
                                                                     CreateTestConstIntVector(inShapeOverUB));
    copyoutTensor1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape1 = {CreateTestScalarVar("Input_0_Dim_0"),
                                               CreateTestScalarVar("Input_0_Dim_1")};
    copyoutTensor1->UpdateDynValidShape(validShape1);
    auto reshapeTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshapeShapeOverUB,
                                                                    CreateTestConstIntVector(reshapeShapeOverUB));
    reshapeTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape2 = {CreateTestScalarVar("Input_1_Dim_0"),
                                               CreateTestScalarVar("Input_1_Dim_1")};
    reshapeTensor->UpdateDynValidShape(validShape2);
    auto copyinTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshapeShapeOverUB,
                                                                    CreateTestConstIntVector(reshapeShapeOverUB));
    copyinTensor2->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto copyinTensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshapeShapeOverUB,
                                                                    CreateTestConstIntVector(reshapeShapeOverUB));
    copyinTensor3->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto outcast_1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outShapeOverUB,
                                                                CreateTestConstIntVector(outShapeOverUB));
    outcast_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto outcast_2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outShapeOverUB,
                                                                CreateTestConstIntVector(outShapeOverUB));
    outcast_2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {copyinTensor1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copyinTensor1}, {copyoutTensor1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {copyoutTensor1}, {reshapeTensor});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {reshapeTensor}, {copyinTensor2});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {reshapeTensor}, {copyinTensor3});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copyinTensor3}, {outcast_1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copyinTensor2}, {outcast_2});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast_1);
    currFunctionPtr->outCasts_.push_back(outcast_2);

    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    int curSize = currFunctionPtr->Operations().size();
    EXPECT_EQ(removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(currFunctionPtr->Operations().size(), curSize);
}

TEST_F(TestRemoveUnalignedReshapeOp, TestCopyToReshapeCopyOnL0CToL1)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestCopyToReshapeCopyOnL0CToL1",
                                                      "TestCopyToReshapeCopyOnL0CToL1", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> inShape = {4, 4};
    std::vector<int64_t> reshapeShape = {2, 8};
    auto incast = IRBuilder().CreateTensorVar(DT_FP32, inShape, CreateTestConstIntVector(inShape));
    auto l0cTensor = IRBuilder().CreateTensorVar(DT_FP32, inShape, CreateTestConstIntVector(inShape));
    l0cTensor->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    auto reshapeInput = IRBuilder().CreateTensorVar(DT_FP32, inShape, CreateTestConstIntVector(inShape));
    reshapeInput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    reshapeInput->UpdateDynValidShape({CreateTestScalarVar("Input_0_Dim_0"), CreateTestScalarVar("Input_0_Dim_1")});
    auto reshapeOutput = IRBuilder().CreateTensorVar(DT_FP32, reshapeShape, CreateTestConstIntVector(reshapeShape));
    reshapeOutput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    reshapeOutput->UpdateDynValidShape({CreateTestScalarVar("Input_1_Dim_0"), CreateTestScalarVar("Input_1_Dim_1")});
    auto l1Tensor = IRBuilder().CreateTensorVar(DT_FP32, reshapeShape, CreateTestConstIntVector(reshapeShape));
    l1Tensor->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto outcast = IRBuilder().CreateTensorVar(DT_FP32, reshapeShape, CreateTestConstIntVector(reshapeShape));
    outcast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {l0cTensor});
    auto& copyOutOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {l0cTensor},
                                                       {reshapeInput});
    copyOutOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        l0cTensor->GetMemoryTypeOriginal(), OpImmediate::Specified(reshapeInput->GetTensorOffset()),
        OpImmediate::Specified(reshapeInput->GetShape()),
        OpImmediate::Specified(reshapeInput->GetRawTensor()->GetDynRawShape())));
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {reshapeInput}, {reshapeOutput});
    auto& copyInOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {reshapeOutput},
                                                      {l1Tensor});
    copyInOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(reshapeOutput->GetTensorOffset()), l1Tensor->GetMemoryTypeOriginal(),
        OpImmediate::Specified(reshapeOutput->GetShape()),
        OpImmediate::Specified(reshapeOutput->GetRawTensor()->GetDynRawShape())));
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {l1Tensor}, {outcast});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    int curSize = currFunctionPtr->Operations().size();
    EXPECT_EQ(removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(copyOutOp.GetOpcode(), Opcode::OP_L0C_RESHAPE_COPY_OUT);
    EXPECT_EQ(copyInOp.GetOpcode(), Opcode::OP_L1_RESHAPE_COPY_IN);
    EXPECT_EQ(currFunctionPtr->Operations().size(), curSize);

    auto copyOutAttr = std::dynamic_pointer_cast<CopyOpAttribute>(copyOutOp.GetOpAttribute());
    ASSERT_NE(copyOutAttr, nullptr);
    EXPECT_FALSE(copyOutAttr->GetToDynValidShape().empty());
    auto copyInAttr = std::dynamic_pointer_cast<CopyOpAttribute>(copyInOp.GetOpAttribute());
    ASSERT_NE(copyInAttr, nullptr);
    EXPECT_FALSE(copyInAttr->GetFromDynValidShape().empty());
}

TEST_F(TestRemoveUnalignedReshapeOp, TestSiblingCopyInConsumerToReshapeCopyInOnUB)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(),
                                                      "TestSiblingCopyInConsumerToReshapeCopyInOnUB",
                                                      "TestSiblingCopyInConsumerToReshapeCopyInOnUB", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    auto ops = BuildDDRReshapeGraphWithSiblingCopyIn(currFunctionPtr, MemoryType::MEM_UB);

    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    int curSize = currFunctionPtr->Operations().size();
    EXPECT_EQ(removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(currFunctionPtr->Operations().size(), curSize + 1);
    EXPECT_EQ(ops.copyOutOp->GetOpcode(), Opcode::OP_COPY_OUT);
    EXPECT_EQ(ops.outputCopyInOp->GetOpcode(), Opcode::OP_RESHAPE_COPY_IN);
    EXPECT_EQ(ops.siblingCopyInOp->GetOpcode(), Opcode::OP_RESHAPE_COPY_IN);

    auto siblingCopyInAttr = std::dynamic_pointer_cast<CopyOpAttribute>(ops.siblingCopyInOp->GetOpAttribute());
    ASSERT_NE(siblingCopyInAttr, nullptr);
    EXPECT_FALSE(siblingCopyInAttr->GetFromDynValidShape().empty());
}

TEST_F(TestRemoveUnalignedReshapeOp, TestSiblingCopyInConsumerToL1ReshapeCopyIn)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(),
                                                      "TestSiblingCopyInConsumerToL1ReshapeCopyIn",
                                                      "TestSiblingCopyInConsumerToL1ReshapeCopyIn", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    auto ops = BuildDDRReshapeGraphWithSiblingCopyIn(currFunctionPtr, MemoryType::MEM_L1);

    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    int curSize = currFunctionPtr->Operations().size();
    EXPECT_EQ(removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(currFunctionPtr->Operations().size(), curSize + 1);
    EXPECT_EQ(ops.copyOutOp->GetOpcode(), Opcode::OP_COPY_OUT);
    EXPECT_EQ(ops.outputCopyInOp->GetOpcode(), Opcode::OP_RESHAPE_COPY_IN);
    EXPECT_EQ(ops.siblingCopyInOp->GetOpcode(), Opcode::OP_L1_RESHAPE_COPY_IN);

    auto siblingCopyInAttr = std::dynamic_pointer_cast<CopyOpAttribute>(ops.siblingCopyInOp->GetOpAttribute());
    ASSERT_NE(siblingCopyInAttr, nullptr);
    EXPECT_FALSE(siblingCopyInAttr->GetFromDynValidShape().empty());
}

TEST_F(TestRemoveUnalignedReshapeOp, TestValidShapeInfer)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestValidShapeInfer",
                                                      "TestValidShapeInfer", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, std::vector<int64_t>{128, 64},
                                                             CreateTestConstIntVector(std::vector<int64_t>{128, 64}));
    inCast->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
    inCast->UpdateDynValidShape({IRBuilder().CreateConstInt(128), IRBuilder().CreateConstInt(64)});

    auto reshapeInput = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{128, 64}, CreateTestConstIntVector(std::vector<int64_t>{128, 64}));
    reshapeInput->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
    reshapeInput->UpdateDynValidShape({IRBuilder().CreateConstInt(128), IRBuilder().CreateConstInt(64)});

    auto reshapeOutput = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{64, 128}, CreateTestConstIntVector(std::vector<int64_t>{64, 128}));
    reshapeOutput->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
    reshapeOutput->UpdateDynValidShape({IRBuilder().CreateConstInt(64), IRBuilder().CreateConstInt(128)});

    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, std::vector<int64_t>{64, 128},
                                                              CreateTestConstIntVector(std::vector<int64_t>{64, 128}),
                                                              TileOpFormat::TILEOP_ND, "outCast");
    outCast->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    auto& reshapeOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {reshapeInput},
                                                       {reshapeOutput});
    reshapeOp.UpdateSubgraphID(0);

    auto& copyOutOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {reshapeOutput},
                                                       {outCast});
    copyOutOp.UpdateSubgraphID(0);

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    EXPECT_EQ(removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr), SUCCESS);

    for (const auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE_COPY_OUT || op.GetOpcode() == Opcode::OP_RESHAPE_COPY_IN) {
            auto oOperand = op.GetOOperands().front();
            auto dynValidShape = oOperand->GetDynValidShape();
            EXPECT_FALSE(dynValidShape.empty());
        }
    }
}

// in - COPYIN - COPYOUT - RESHAPE - COPYIN - COPYOUT - out
//    - COPYIN - COPYOUT           - COPYIN - COPYOUT - out
TEST_F(TestRemoveUnalignedReshapeOp, TestCopyToReshapeBeforeMultCopyOutOnUB)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestCopyToReshapeCopy",
                                                      "TestCopyToReshapeCopy", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inShape = {2, 2};
    std::vector<int64_t> copyoutShape = {2, 4};
    std::vector<int64_t> reshapeShape = {4, 2};
    std::vector<int64_t> outShape = {4, 4};
    IRBuilder builder;
    auto incast = builder.CreateTensorVar(DT_FP32, inShape, CreateTestConstIntVector(inShape));

    auto copyin_Tensor1 = builder.CreateTensorVar(DT_FP32, inShape, CreateTestConstIntVector(inShape));
    copyin_Tensor1->SetMemoryTypeBoth(MemoryType::MEM_UB, true);

    auto copyout_Tensor1 = builder.CreateTensorVar(
        DT_FP32, copyoutShape, {CreateTestScalarVar("Input_00_Dim_0"), CreateTestScalarVar("Input_00_Dim_1")});
    copyout_Tensor1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    auto copyin_Tensor2 = builder.CreateTensorVar(DT_FP32, inShape, CreateTestConstIntVector(inShape));
    copyin_Tensor2->SetMemoryTypeBoth(MemoryType::MEM_UB, true);

    auto reshape_Tensor = builder.CreateTensorVar(
        DT_FP32, reshapeShape, {CreateTestScalarVar("Input_02_Dim_0"), CreateTestScalarVar("Input_02_Dim_1")});
    reshape_Tensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    auto copyin_Tensor3 = builder.CreateTensorVar(DT_FP32, reshapeShape, CreateTestConstIntVector(reshapeShape));
    copyin_Tensor3->SetMemoryTypeBoth(MemoryType::MEM_UB, true);

    auto copyin_Tensor4 = builder.CreateTensorVar(DT_FP32, reshapeShape, CreateTestConstIntVector(reshapeShape));
    copyin_Tensor4->SetMemoryTypeBoth(MemoryType::MEM_UB, true);

    auto outcast = builder.CreateTensorVar(DT_FP32, outShape, CreateTestConstIntVector(outShape));
    outcast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {copyin_Tensor1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {copyin_Tensor2});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copyin_Tensor1}, {copyout_Tensor1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copyin_Tensor2}, {copyout_Tensor1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {copyout_Tensor1}, {reshape_Tensor});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {reshape_Tensor}, {copyin_Tensor3});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {reshape_Tensor}, {copyin_Tensor4});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copyin_Tensor4}, {outcast});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copyin_Tensor3}, {outcast});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    EXPECT_EQ(removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr), SUCCESS);
}

TEST_F(TestRemoveUnalignedReshapeOp, TestMultiCopyOutProducerPreserveToken)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestMultiCopyOutProducerPreserveToken",
                                                      "TestMultiCopyOutProducerPreserveToken", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> inShape = {2, 4};
    std::vector<int64_t> reshapeShape = {4, 2};
    IRBuilder builder;
    auto incast = builder.CreateTensorVar(*currFunctionPtr, DT_FP32, inShape, CreateTestConstIntVector(inShape));
    incast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto ubTensor1 = builder.CreateTensorVar(*currFunctionPtr, DT_FP32, inShape, CreateTestConstIntVector(inShape));
    ubTensor1->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto ubTensor2 = builder.CreateTensorVar(*currFunctionPtr, DT_FP32, inShape, CreateTestConstIntVector(inShape));
    ubTensor2->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto reshapeInput = builder.CreateTensorVar(
        *currFunctionPtr, DT_FP32, inShape,
        {CreateTestScalarVar("Input_0_Dim_0"), CreateTestScalarVar("Input_0_Dim_1")});
    reshapeInput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto reshapeOutput = builder.CreateTensorVar(
        *currFunctionPtr, DT_FP32, reshapeShape,
        {CreateTestScalarVar("Input_1_Dim_0"), CreateTestScalarVar("Input_1_Dim_1")});
    reshapeOutput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto ubOutput = builder.CreateTensorVar(*currFunctionPtr, DT_FP32, reshapeShape,
                                            CreateTestConstIntVector(reshapeShape));
    ubOutput->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto outcast = builder.CreateTensorVar(*currFunctionPtr, DT_FP32, reshapeShape,
                                           CreateTestConstIntVector(reshapeShape));
    outcast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {ubTensor1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {ubTensor2});
    auto& firstCopyOut = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {ubTensor1},
                                                          {reshapeInput});
    auto& secondCopyOut = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {ubTensor2},
                                                           {reshapeInput});
    auto token = AddTokenEdge(*currFunctionPtr, firstCopyOut, secondCopyOut);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {reshapeInput}, {reshapeOutput});
    auto& copyIn = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {reshapeOutput}, {ubOutput});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {ubOutput}, {outcast});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    int curSize = currFunctionPtr->Operations().size();
    EXPECT_EQ(removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(currFunctionPtr->Operations().size(), curSize);
    EXPECT_EQ(firstCopyOut.GetOpcode(), Opcode::OP_RESHAPE_COPY_OUT);
    EXPECT_EQ(secondCopyOut.GetOpcode(), Opcode::OP_RESHAPE_COPY_OUT);
    EXPECT_EQ(copyIn.GetOpcode(), Opcode::OP_RESHAPE_COPY_IN);
    EXPECT_EQ(firstCopyOut.result_token_.front(), token);
    EXPECT_NE(std::find(secondCopyOut.tokens_.begin(), secondCopyOut.tokens_.end(), token),
              secondCopyOut.tokens_.end());
    EXPECT_TRUE(currFunctionPtr->GetVarDependency().HasProducer(token, ToStmtPtr(firstCopyOut)));
    EXPECT_TRUE(currFunctionPtr->GetVarDependency().HasConsumer(token, ToStmtPtr(secondCopyOut)));
}

TEST_F(TestRemoveUnalignedReshapeOp, TestRawIdClosureAcrossTensorVersionsPreserveToken)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(),
                                                      "TestRawIdClosureAcrossTensorVersionsPreserveToken",
                                                      "TestRawIdClosureAcrossTensorVersionsPreserveToken", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> inShape = {8, 8};
    std::vector<int64_t> reshapeShape = {4, 16};
    IRBuilder builder;
    auto incast = builder.CreateTensorVar(*currFunctionPtr, DT_FP32, inShape, CreateTestConstIntVector(inShape));
    incast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto oldUb = builder.CreateTensorVar(*currFunctionPtr, DT_FP32, inShape, CreateTestConstIntVector(inShape));
    oldUb->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto currentUb = builder.CreateTensorVar(*currFunctionPtr, DT_FP32, inShape, CreateTestConstIntVector(inShape));
    currentUb->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto oldVersion = builder.CreateTensorVar(
        *currFunctionPtr, DT_FP32, inShape,
        {CreateTestScalarVar("Input_0_Dim_0"), CreateTestScalarVar("Input_0_Dim_1")});
    oldVersion->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto currentVersion = builder.CreateTensorVar(
        *currFunctionPtr, oldVersion->GetRawTensor(), {0, 0}, inShape,
        {CreateTestScalarVar("Input_2_Dim_0"), CreateTestScalarVar("Input_2_Dim_1")});
    currentVersion->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto oldCopyInOutput = builder.CreateTensorVar(*currFunctionPtr, DT_FP32, inShape,
                                                   CreateTestConstIntVector(inShape));
    oldCopyInOutput->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto reshapeOutput = builder.CreateTensorVar(
        *currFunctionPtr, DT_FP32, reshapeShape,
        {CreateTestScalarVar("Input_1_Dim_0"), CreateTestScalarVar("Input_1_Dim_1")});
    reshapeOutput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto reshapeCopyInOutput = builder.CreateTensorVar(*currFunctionPtr, DT_FP32, reshapeShape,
                                                       CreateTestConstIntVector(reshapeShape));
    reshapeCopyInOutput->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto outcast = builder.CreateTensorVar(*currFunctionPtr, DT_FP32, reshapeShape,
                                           CreateTestConstIntVector(reshapeShape));
    outcast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto oldOutcast = builder.CreateTensorVar(*currFunctionPtr, DT_FP32, inShape, CreateTestConstIntVector(inShape));
    oldOutcast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {oldUb});
    auto& oldCopyOut = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {oldUb}, {oldVersion});
    auto& oldCopyIn = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {oldVersion},
                                                       {oldCopyInOutput});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {oldCopyInOutput}, {oldOutcast});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {currentUb});
    auto& currentCopyOut = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {currentUb},
                                                            {currentVersion});
    auto token = AddTokenEdge(*currFunctionPtr, oldCopyOut, currentCopyOut);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {currentVersion}, {reshapeOutput});
    auto& reshapeCopyIn = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {reshapeOutput},
                                                           {reshapeCopyInOutput});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {reshapeCopyInOutput}, {outcast});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);
    currFunctionPtr->outCasts_.push_back(oldOutcast);

    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    int curSize = currFunctionPtr->Operations().size();
    EXPECT_EQ(removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(currFunctionPtr->Operations().size(), curSize);
    EXPECT_EQ(oldCopyOut.GetOpcode(), Opcode::OP_RESHAPE_COPY_OUT);
    EXPECT_EQ(currentCopyOut.GetOpcode(), Opcode::OP_RESHAPE_COPY_OUT);
    EXPECT_EQ(oldCopyIn.GetOpcode(), Opcode::OP_RESHAPE_COPY_IN);
    EXPECT_EQ(reshapeCopyIn.GetOpcode(), Opcode::OP_RESHAPE_COPY_IN);
    EXPECT_EQ(oldCopyOut.result_token_.front(), token);
    EXPECT_NE(std::find(currentCopyOut.tokens_.begin(), currentCopyOut.tokens_.end(), token),
              currentCopyOut.tokens_.end());
    EXPECT_TRUE(currFunctionPtr->GetVarDependency().HasProducer(token, ToStmtPtr(oldCopyOut)));
    EXPECT_TRUE(currFunctionPtr->GetVarDependency().HasConsumer(token, ToStmtPtr(currentCopyOut)));
}

namespace {
constexpr int64_t FOLD_RAW_ROWS = 285;
constexpr int64_t FOLD_RAW_COLS = 64;
constexpr int64_t FOLD_TILE_ROWS = 5;
constexpr int64_t FOLD_TILE_COLS = 32;
} // namespace

// FoldConsumerViewOffsetIntoCopyIn：视图动态偏移下沉进reshape copy-in的看护用例
class FoldConsumerViewOffsetTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        func = std::make_shared<Function>(Program::GetInstance(), "test_fold_func", "test_fold_func", nullptr);
        func->SetGraphType(GraphType::BLOCK_GRAPH);
        func->SetFunctionType(FunctionType::STATIC);
        pass = std::make_unique<RemoveUnalignedReshape>();
    }

    void TearDown() override
    {
        pass.reset();
        func.reset();
    }

    // 构造 RESHAPE_COPY_IN: GM[285,64] → UB output，from-offset全0、shape=[285,64]
    Operation& CreateReshapeCopyIn(const std::shared_ptr<LogicalTensor>& output)
    {
        auto input = npu::tile_fwk::IRBuilder().CreateTensorVar(
            DT_FP32, {FOLD_RAW_ROWS, FOLD_RAW_COLS}, CreateTestConstIntVector({FOLD_RAW_ROWS, FOLD_RAW_COLS}));
        input->SetMemoryTypeBoth(MEM_DEVICE_DDR);
        auto& copyInOp = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_RESHAPE_COPY_IN, {input}, {output});
        auto copyAttr = std::make_shared<CopyOpAttribute>(
            OpImmediate::Specified(CreateTestConstIntVector({0, 0})), MEM_DEVICE_DDR,
            OpImmediate::Specified(std::vector<int64_t>{FOLD_RAW_ROWS, FOLD_RAW_COLS}),
            OpImmediate::Specified(std::vector<int64_t>{FOLD_RAW_ROWS, FOLD_RAW_COLS}),
            OpImmediate::Specified(output->GetDynValidShape()));
        copyInOp.SetOpAttribute(copyAttr);
        return copyInOp;
    }

    // 构造VIEW消费者：from = (rowDyn, colConst)，fromDynOffset为权威值
    Operation& CreateViewConsumer(const std::shared_ptr<LogicalTensor>& input,
                                  const std::shared_ptr<LogicalTensor>& output, const SymbolicScalar& rowOffset,
                                  int64_t colOffset)
    {
        auto& viewOp = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_VIEW, {input}, {output});
        auto viewAttr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, colOffset});
        viewAttr->SetFromOffset(std::vector<int64_t>{0, colOffset}, {rowOffset, SymbolicScalar(colOffset)});
        viewOp.SetOpAttribute(viewAttr);
        return viewOp;
    }

    std::shared_ptr<LogicalTensor> MakeUbTensor(const std::vector<int64_t>& shape,
                                                const std::vector<SymbolicScalar>& validShape)
    {
        auto tensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, validShape);
        tensor->SetMemoryTypeBoth(MEM_UB);
        return tensor;
    }

protected:
    std::shared_ptr<Function> func;
    std::unique_ptr<RemoveUnalignedReshape> pass;
};

// 正常下沉：动态行偏移+常量列偏移，copy-in属性收缩、view退化为常量列偏移
TEST_F(FoldConsumerViewOffsetTest, FoldWithDynamicRowOffset)
{
    auto rowOffset = CreateTestScalarVar("loop_idx_b_idx_mul_5");
    auto output = MakeUbTensor({FOLD_RAW_ROWS, FOLD_RAW_COLS},
                               CreateTestConstIntVector({FOLD_RAW_ROWS, FOLD_RAW_COLS}));
    auto& copyInOp = CreateReshapeCopyIn(output);

    auto view1Out = MakeUbTensor({FOLD_TILE_ROWS, FOLD_TILE_COLS},
                                 CreateTestConstIntVector({FOLD_TILE_ROWS, FOLD_TILE_COLS}));
    auto view2Out = MakeUbTensor({FOLD_TILE_ROWS, FOLD_TILE_COLS},
                                 CreateTestConstIntVector({FOLD_TILE_ROWS, FOLD_TILE_COLS}));
    auto& view1 = CreateViewConsumer(output, view1Out, rowOffset, 0);
    auto& view2 = CreateViewConsumer(output, view2Out, rowOffset, FOLD_TILE_COLS);
    (void)view1;
    (void)view2;

    pass->FoldConsumerViewOffsetIntoCopyIn(copyInOp);

    // copy-in输出收缩为view并集box [5,64]
    ASSERT_EQ(copyInOp.GetOOperands().size(), 1);
    auto shrunk = copyInOp.GetOOperands().front();
    EXPECT_EQ(shrunk->GetShape(), (std::vector<int64_t>{FOLD_TILE_ROWS, FOLD_RAW_COLS}));

    // from-offset折入动态行偏移、列为并集起点0
    auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(copyInOp.GetOpAttribute());
    ASSERT_NE(copyAttr, nullptr);
    auto fromOffset = OpImmediate::ToSpecified(copyAttr->GetFromOffset());
    ASSERT_EQ(fromOffset.size(), 2);
    EXPECT_FALSE(fromOffset[0].IsImmediate()) << "Row offset should keep the dynamic component";
    EXPECT_EQ(fromOffset[1].Concrete(), 0);

    // 两个view退化为纯常量列偏移
    auto attr1 = std::dynamic_pointer_cast<ViewOpAttribute>(view1.GetOpAttribute());
    auto attr2 = std::dynamic_pointer_cast<ViewOpAttribute>(view2.GetOpAttribute());
    ASSERT_NE(attr1, nullptr);
    ASSERT_NE(attr2, nullptr);
    EXPECT_EQ(attr1->GetFromOffset(), (std::vector<int64_t>{0, 0}));
    EXPECT_EQ(attr2->GetFromOffset(), (std::vector<int64_t>{0, FOLD_TILE_COLS}));
    // view输入已重接到shrunk
    EXPECT_EQ(view1.GetIOperands().front(), shrunk);
    EXPECT_EQ(view2.GetIOperands().front(), shrunk);
    EXPECT_TRUE(output->GetConsumers().empty()) << "Old output should have no consumers after fold";
}

// 混合非VIEW消费者早退：图保持不变
TEST_F(FoldConsumerViewOffsetTest, FoldSkippedWithNonViewConsumer)
{
    auto rowOffset = CreateTestScalarVar("loop_idx_b_idx_mul_5");
    auto output = MakeUbTensor({FOLD_RAW_ROWS, FOLD_RAW_COLS},
                               CreateTestConstIntVector({FOLD_RAW_ROWS, FOLD_RAW_COLS}));
    auto& copyInOp = CreateReshapeCopyIn(output);

    auto viewOut = MakeUbTensor({FOLD_TILE_ROWS, FOLD_TILE_COLS},
                                CreateTestConstIntVector({FOLD_TILE_ROWS, FOLD_TILE_COLS}));
    auto& view = CreateViewConsumer(output, viewOut, rowOffset, 0);
    auto castOut = MakeUbTensor({FOLD_RAW_ROWS, FOLD_RAW_COLS},
                                CreateTestConstIntVector({FOLD_RAW_ROWS, FOLD_RAW_COLS}));
    auto& castOp = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_CAST, {output}, {castOut});
    (void)castOp;

    pass->FoldConsumerViewOffsetIntoCopyIn(copyInOp);

    EXPECT_EQ(copyInOp.GetOOperands().front(), output) << "Copy-in output should stay unchanged";
    EXPECT_EQ(view.GetIOperands().front(), output) << "View input should stay unchanged";
}

// 末维偏移非常量早退：图保持不变
TEST_F(FoldConsumerViewOffsetTest, FoldSkippedWithDynamicLastDimOffset)
{
    auto rowOffset = CreateTestScalarVar("loop_idx_b_idx_mul_5");
    auto colOffset = CreateTestScalarVar("dynamic_col_offset");
    auto output = MakeUbTensor({FOLD_RAW_ROWS, FOLD_RAW_COLS},
                               CreateTestConstIntVector({FOLD_RAW_ROWS, FOLD_RAW_COLS}));
    auto& copyInOp = CreateReshapeCopyIn(output);

    auto viewOut = MakeUbTensor({FOLD_TILE_ROWS, FOLD_TILE_COLS},
                                CreateTestConstIntVector({FOLD_TILE_ROWS, FOLD_TILE_COLS}));
    auto& view = CreateViewConsumer(output, viewOut, rowOffset, 0);
    // 末维改为动态偏移
    auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(view.GetOpAttribute());
    viewAttr->SetFromOffset(std::vector<int64_t>{0, 0}, {rowOffset, colOffset});

    pass->FoldConsumerViewOffsetIntoCopyIn(copyInOp);

    EXPECT_EQ(copyInOp.GetOOperands().front(), output) << "Copy-in output should stay unchanged";
    EXPECT_EQ(view.GetIOperands().front(), output) << "View input should stay unchanged";
}

// 多view列并集box计算：中间列起点+不同列宽，box覆盖并集且view列偏移相对化
TEST_F(FoldConsumerViewOffsetTest, FoldWithMultiViewColumnUnion)
{
    auto rowOffset = CreateTestScalarVar("loop_idx_b_idx_mul_5");
    auto output = MakeUbTensor({FOLD_RAW_ROWS, FOLD_RAW_COLS},
                               CreateTestConstIntVector({FOLD_RAW_ROWS, FOLD_RAW_COLS}));
    auto& copyInOp = CreateReshapeCopyIn(output);

    // 三个view：列起点0/16/48，宽16/32/16，并集[0,64)
    auto view1Out = MakeUbTensor({FOLD_TILE_ROWS, 16}, CreateTestConstIntVector({FOLD_TILE_ROWS, 16}));
    auto view2Out = MakeUbTensor({FOLD_TILE_ROWS, 32}, CreateTestConstIntVector({FOLD_TILE_ROWS, 32}));
    auto view3Out = MakeUbTensor({FOLD_TILE_ROWS, 16}, CreateTestConstIntVector({FOLD_TILE_ROWS, 16}));
    auto& view1 = CreateViewConsumer(output, view1Out, rowOffset, 0);
    auto& view2 = CreateViewConsumer(output, view2Out, rowOffset, 16);
    auto& view3 = CreateViewConsumer(output, view3Out, rowOffset, 48);
    (void)view1;
    (void)view2;
    (void)view3;

    pass->FoldConsumerViewOffsetIntoCopyIn(copyInOp);

    auto shrunk = copyInOp.GetOOperands().front();
    EXPECT_EQ(shrunk->GetShape(), (std::vector<int64_t>{FOLD_TILE_ROWS, FOLD_RAW_COLS}))
        << "Box should cover column union";

    auto attr1 = std::dynamic_pointer_cast<ViewOpAttribute>(view1.GetOpAttribute());
    auto attr2 = std::dynamic_pointer_cast<ViewOpAttribute>(view2.GetOpAttribute());
    auto attr3 = std::dynamic_pointer_cast<ViewOpAttribute>(view3.GetOpAttribute());
    ASSERT_NE(attr1, nullptr);
    ASSERT_NE(attr2, nullptr);
    ASSERT_NE(attr3, nullptr);
    EXPECT_EQ(attr1->GetFromOffset(), (std::vector<int64_t>{0, 0}));
    EXPECT_EQ(attr2->GetFromOffset(), (std::vector<int64_t>{0, 16}));
    EXPECT_EQ(attr3->GetFromOffset(), (std::vector<int64_t>{0, 48}));
}

// Split views may carry the same dynamic row base plus different static row/column tile offsets.
TEST_F(FoldConsumerViewOffsetTest, FoldWithDynamicBaseAndTwoDimensionalTileOffsets)
{
    auto rowBase = CreateTestScalarVar("loop_idx_b_idx_mul_actual_b");
    auto validRows = CreateTestScalarVar("actual_b");
    auto output = MakeUbTensor({FOLD_RAW_ROWS, FOLD_RAW_COLS},
                               CreateTestConstIntVector({FOLD_RAW_ROWS, FOLD_RAW_COLS}));
    auto& copyInOp = CreateReshapeCopyIn(output);

    auto topLeftOut = MakeUbTensor({16, 32}, {validRows.Min(SymbolicScalar(16)), SymbolicScalar(32)});
    auto topRightOut = MakeUbTensor({16, 32}, {validRows.Min(SymbolicScalar(16)), SymbolicScalar(32)});
    auto bottomLeftOut = MakeUbTensor(
        {2, 32}, {(validRows - 16).Max(SymbolicScalar(0)).Min(SymbolicScalar(2)), SymbolicScalar(32)});
    auto bottomRightOut = MakeUbTensor(
        {2, 32}, {(validRows - 16).Max(SymbolicScalar(0)).Min(SymbolicScalar(2)), SymbolicScalar(32)});

    auto& topLeft = CreateViewConsumer(output, topLeftOut, rowBase, 0);
    auto& topRight = CreateViewConsumer(output, topRightOut, rowBase, 32);
    auto& bottomLeft = CreateViewConsumer(output, bottomLeftOut, rowBase + 16, 0);
    auto& bottomRight = CreateViewConsumer(output, bottomRightOut, rowBase + 16, 32);
    std::dynamic_pointer_cast<ViewOpAttribute>(bottomLeft.GetOpAttribute())
        ->SetFromOffset({16, 0}, {rowBase + 16, SymbolicScalar(0)});
    std::dynamic_pointer_cast<ViewOpAttribute>(bottomRight.GetOpAttribute())
        ->SetFromOffset({16, 32}, {rowBase + 16, SymbolicScalar(32)});

    pass->FoldConsumerViewOffsetIntoCopyIn(copyInOp);

    auto shrunk = copyInOp.GetOOperands().front();
    EXPECT_EQ(shrunk->GetShape(), (std::vector<int64_t>{18, 64}));
    auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(copyInOp.GetOpAttribute());
    ASSERT_NE(copyAttr, nullptr);
    auto fromOffset = OpImmediate::ToSpecified(copyAttr->GetFromOffset());
    ASSERT_EQ(fromOffset.size(), 2);
    EXPECT_EQ(fromOffset[0].Dump(), rowBase.Dump());
    EXPECT_EQ(fromOffset[1].Concrete(), 0);
    EXPECT_EQ(std::dynamic_pointer_cast<ViewOpAttribute>(topLeft.GetOpAttribute())->GetFromOffset(),
              (std::vector<int64_t>{0, 0}));
    EXPECT_EQ(std::dynamic_pointer_cast<ViewOpAttribute>(topRight.GetOpAttribute())->GetFromOffset(),
              (std::vector<int64_t>{0, 32}));
    EXPECT_EQ(std::dynamic_pointer_cast<ViewOpAttribute>(bottomLeft.GetOpAttribute())->GetFromOffset(),
              (std::vector<int64_t>{16, 0}));
    EXPECT_EQ(std::dynamic_pointer_cast<ViewOpAttribute>(bottomRight.GetOpAttribute())->GetFromOffset(),
              (std::vector<int64_t>{16, 32}));
}

// 无消费者早退
TEST_F(FoldConsumerViewOffsetTest, FoldSkippedWithNoConsumer)
{
    auto output = MakeUbTensor({FOLD_RAW_ROWS, FOLD_RAW_COLS},
                               CreateTestConstIntVector({FOLD_RAW_ROWS, FOLD_RAW_COLS}));
    auto& copyInOp = CreateReshapeCopyIn(output);

    pass->FoldConsumerViewOffsetIntoCopyIn(copyInOp);

    EXPECT_EQ(copyInOp.GetOOperands().front(), output) << "No consumer means no fold";
}

// 纯常量偏移不触发（无动态分量）
TEST_F(FoldConsumerViewOffsetTest, FoldSkippedWithAllConstantOffsets)
{
    auto output = MakeUbTensor({FOLD_RAW_ROWS, FOLD_RAW_COLS},
                               CreateTestConstIntVector({FOLD_RAW_ROWS, FOLD_RAW_COLS}));
    auto& copyInOp = CreateReshapeCopyIn(output);

    auto viewOut = MakeUbTensor({FOLD_TILE_ROWS, FOLD_TILE_COLS},
                                CreateTestConstIntVector({FOLD_TILE_ROWS, FOLD_TILE_COLS}));
    auto& view = CreateViewConsumer(output, viewOut, SymbolicScalar(5), 0);
    (void)view;

    pass->FoldConsumerViewOffsetIntoCopyIn(copyInOp);

    EXPECT_EQ(copyInOp.GetOOperands().front(), output) << "All-constant offsets should not trigger fold";
}
