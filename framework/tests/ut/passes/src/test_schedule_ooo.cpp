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
 * \file test_schedule_ooo.cpp
 * \brief Unit test for OoOSchedule.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/irbuilder.h"
#include "interface/utils/simt_utils.h"
#include "passes/pass_mgr/pass_manager.h"
#include "symbolic_scalar_test_utils.h"
#include "tilefwk/platform.h"
#include "tilefwk/tilefwk.h"
#define private public
#define protected public
#include "computational_graph_builder.h"
#include "passes/block_graph_pass/schedule_ooo/common/dep_manager.h"
#include "passes/block_graph_pass/schedule_ooo/common/iso_matcher.h"
#include "passes/block_graph_pass/schedule_ooo/post_schedule/buffer_rearrange.h"
#include "passes/block_graph_pass/schedule_ooo/pre_schedule/cluster_list_sort.h"
#include "passes/block_graph_pass/schedule_ooo/pre_schedule/core_assign.h"
#include "passes/block_graph_pass/schedule_ooo/pre_schedule/prior_dfs_sort.h"
#include "passes/block_graph_pass/schedule_ooo/pre_schedule/task_splitter.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/block_graph_pass/schedule_ooo/schedule_ooo.h"
#include "passes/tile_graph_pass/graph_constraint/infer_dyn_shape.h"

namespace npu::tile_fwk {
constexpr int OOO_NUM2 = 2;
constexpr int OOO_NUM209 = 209;
constexpr int UBPoolSize = 192 * 1024;
std::unordered_map<Opcode, int> preNodePriority = {
    {Opcode::OP_UB_ALLOC, 0},
    {Opcode::OP_L1_ALLOC, 0},
    {Opcode::OP_L0A_ALLOC, 0},
    {Opcode::OP_L0B_ALLOC, 0},
    {Opcode::OP_L0C_ALLOC, 0},
    {Opcode::OP_BT_ALLOC, 0},
    {Opcode::OP_FIX_ALLOC, 0},
    {Opcode::OP_L1_TO_L0A, 1},
    {Opcode::OP_L1_TO_L0B, 1},
    {Opcode::OP_L1_TO_L0_AT, 1},
    {Opcode::OP_L1_TO_L0_BT, 1},
    {Opcode::OP_L1_TO_FIX, 1},
    {Opcode::OP_L1_TO_FIX_QUANT_PRE, 1},
    {Opcode::OP_L1_TO_FIX_RELU_PRE, 1},
    {Opcode::OP_L1_TO_FIX_RELU_POST, 1},
    {Opcode::OP_L1_TO_FIX_QUANT_POST, 1},
    {Opcode::OP_L1_TO_FIX_ELT_ANTIQ, 1},
    {Opcode::OP_L1_TO_FIX_MTE2_ANTIQ, 1},
    {Opcode::OP_L1_TO_BT, 1},
    {Opcode::OP_COPY_IN, 2},
    {Opcode::OP_UB_COPY_IN, 2},
    {Opcode::OP_L1_COPY_IN, 2},
    {Opcode::OP_L1_COPY_IN_FRACTAL_Z, 2},
    {Opcode::OP_L1_COPY_UB, 2},
    {Opcode::OP_L0C_COPY_UB, 2},
    {Opcode::OP_UB_COPY_L1, 2},
};

class ScheduleOoOTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override { Program::GetInstance().Reset(); }
    void TearDown() override {}
};

void SetTensorAttr(LogicalTensorPtr tensor, MemoryType memType, int memId)
{
    tensor->SetMemoryTypeOriginal(memType);
    tensor->SetMemoryTypeToBe(memType);
    tensor->memoryrange.memId = memId;
    tensor->UpdateDynValidShape({CreateTestScalarVar("S0"), CreateTestScalarVar("S1")});
}

void SetAllocAttr(Operation& alloc, int latency) { alloc.UpdateLatency(latency); }

LogicalTensorPtr CreateTensor(DataType dateType, std::vector<int64_t> shape, MemoryType memType, int memId)
{
    LogicalTensorPtr tensor = npu::tile_fwk::IRBuilder().CreateTensorVar(dateType, shape,
                                                                         CreateTestConstIntVector(shape));
    SetTensorAttr(tensor, memType, memId);
    return tensor;
}

Operation& CreateAllocOp(Function& currFunction, LogicalTensorPtr tensor, int latency)
{
    Operation& alloc = PassOperationUtils::AddOperation(currFunction, Opcode::OP_UB_ALLOC, {},
                                                        LogicalTensors({tensor}));
    SetAllocAttr(alloc, latency);
    return alloc;
}

Operation& CreateCopyOp(Function& currFunction, Opcode opcode, LogicalTensorPtr inTensor, LogicalTensorPtr outTensor,
                        std::vector<int64_t> shape)
{
    std::vector<int64_t> offset = {0, 0};
    auto& copy = PassOperationUtils::AddOperation(currFunction, opcode, LogicalTensors({inTensor}),
                                                  LogicalTensors({outTensor}));
    auto shapeImme = OpImmediate::Specified(shape);
    if (opcode == Opcode::OP_COPY_IN) {
        copy.SetOpAttribute(
            std::make_shared<CopyOpAttribute>(OpImmediate::Specified(offset), MEM_UB, shapeImme, shapeImme));
    }
    if (opcode == Opcode::OP_COPY_OUT) {
        copy.SetOpAttribute(
            std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified(offset), shapeImme, shapeImme));
    }
    return copy;
}

Operation& CreateAddOp(Function& currFunction, LogicalTensorPtr inTensor1, LogicalTensorPtr inTensor2,
                       LogicalTensorPtr outTensor)
{
    auto& add = PassOperationUtils::AddOperation(currFunction, Opcode::OP_ADD, LogicalTensors({inTensor1, inTensor2}),
                                                 LogicalTensors({outTensor}));
    return add;
}

void ReorderOperations(Function& function)
{
    auto opList = function.Operations().DuplicatedOpList();
    std::vector<Operation*> newOperations;
    for (auto& op : opList) {
        if (op->GetOpcodeStr().find("ALLOC") != std::string::npos) {
            newOperations.insert(newOperations.begin(), op);
        } else {
            newOperations.push_back(op);
        }
    }
    function.ScheduleBy(newOperations);
}

TEST_F(ScheduleOoOTest, TestMainScheduleOoO)
{
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestParams", "TestParams", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestOOO", "TestOOO", rootFuncPtr.get());
    currFunctionPtr->paramConfigs_.OoOPreScheduleMethod = "PriorDFS";
    auto emptyOpFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "", "", rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    EXPECT_TRUE(emptyOpFunctionPtr != nullptr);
    currFunctionPtr->SetGraphType(GraphType::BLOCK_GRAPH);
    emptyOpFunctionPtr->SetGraphType(GraphType::BLOCK_GRAPH);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());
    rootFuncPtr->rootFunc_->programs_.emplace(emptyOpFunctionPtr->GetFuncMagic(), emptyOpFunctionPtr.get());
    std::vector<int64_t> shape = {128, 128};
    auto shapeImme = OpImmediate::Specified(shape);

    auto tensor1 = CreateTensor(DataType::DT_FP32, shape, MEM_DEVICE_DDR, 0);
    auto tensor2 = CreateTensor(DataType::DT_FP32, shape, MEM_DEVICE_DDR, 1);
    auto tensor3 = CreateTensor(DataType::DT_FP32, shape, MEM_UB, 2);
    auto tensor4 = CreateTensor(DataType::DT_FP32, shape, MEM_UB, 3);
    auto tensor5 = CreateTensor(DataType::DT_FP32, shape, MEM_UB, 4);
    auto tensor6 = CreateTensor(DataType::DT_FP32, shape, MEM_UB, 5);
    auto tensor7 = CreateTensor(DataType::DT_FP32, shape, MEM_DEVICE_DDR, 6);
    auto tensor8 = CreateTensor(DataType::DT_FP32, shape, MEM_UB, 7);
    auto tensor9 = CreateTensor(DataType::DT_FP32, shape, MEM_UB, 8);
    auto& alloc1 = CreateAllocOp(*currFunctionPtr, tensor3, 1);
    auto& alloc2 = CreateAllocOp(*currFunctionPtr, tensor4, 1);
    auto& alloc3 = CreateAllocOp(*currFunctionPtr, tensor5, 1);
    auto& alloc4 = CreateAllocOp(*currFunctionPtr, tensor6, 1);
    auto& alloc5 = CreateAllocOp(*currFunctionPtr, tensor8, 1);
    auto& alloc6 = CreateAllocOp(*currFunctionPtr, tensor9, 1);
    auto& copyin1 = CreateCopyOp(*currFunctionPtr, Opcode::OP_COPY_IN, tensor1, tensor3, shape);
    auto& copyin2 = CreateCopyOp(*currFunctionPtr, Opcode::OP_COPY_IN, tensor2, tensor4, shape);
    auto& add1 = CreateAddOp(*currFunctionPtr, tensor3, tensor4, tensor5);
    auto& add2 = CreateAddOp(*currFunctionPtr, tensor3, tensor4, tensor6);
    auto& add3 = CreateAddOp(*currFunctionPtr, tensor6, tensor4, tensor8);
    auto& add4 = CreateAddOp(*currFunctionPtr, tensor8, tensor5, tensor9);
    (void)alloc1, (void)alloc2, (void)alloc3, (void)alloc4, (void)alloc5, (void)alloc6, (void)copyin1, (void)copyin2,
        (void)add1, (void)add2, (void)add3, (void)add4;
    for (auto& program : rootFuncPtr->rootFunc_->programs_) {
        ReorderOperations(*(program.second));
    }
    currFunctionPtr->EndFunction(nullptr);
    emptyOpFunctionPtr->EndFunction(nullptr);
    OoOSchedule oooSchedule;
    EXPECT_EQ(oooSchedule.PreCheck(*rootFuncPtr), SUCCESS);
    oooSchedule.RunOnFunction(*rootFuncPtr);
    EXPECT_EQ(oooSchedule.PostCheck(*rootFuncPtr), SUCCESS);
}

static bool CheckSkipOps(std::vector<Operation*>& skipOps, Operation* op)
{
    for (auto skipOp : skipOps) {
        if (skipOp == op) {
            return true;
        }
    }
    return false;
}

TEST_F(ScheduleOoOTest, TestDependencies)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_ROWMAX_SINGLE,
                                Opcode::OP_ADD,      Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t3"}, {"t2"}, {"t4", "t5"}, {"t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t4"}, {"t5"},       {"t6"}, {"t8"},
                                                    {"t2"}, {"t4"}, {"t5", "t6"}, {"t8"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3",  "Alloc4", "Alloc5",
                                     "Copyin1", "Copyin2", "RowMax1", "Add1",   "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    Operation* op = subGraph.GetOp("RowMax1");
    EXPECT_NE(op, nullptr);
    EXPECT_EQ(ooOScheduler.state_.depManager.GetPredecessors(op).size(), 3);
    EXPECT_TRUE(ooOScheduler.state_.depManager.GetPredecessors(op).count(subGraph.GetOp("Alloc4")) > 0);
    EXPECT_EQ(ooOScheduler.state_.depManager.GetSuccessors(op).size(), 2);
    EXPECT_TRUE(ooOScheduler.state_.depManager.GetSuccessors(op).count(subGraph.GetOp("Add1")) > 0);
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestTokenDependency)
{
    ComputationalGraphBuilder graph;
    EXPECT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"in1", "out1", "in2", "out2"}));
    EXPECT_TRUE(graph.AddOp(Opcode::OP_ADDS, {"in1"}, {"out1"}, "Producer", true));
    EXPECT_TRUE(graph.AddOp(Opcode::OP_MULS, {"in2"}, {"out2"}, "Consumer", true));
    auto* function = graph.GetFunction();
    auto* producer = graph.GetOp("Producer");
    auto* consumer = graph.GetOp("Consumer");
    ASSERT_NE(function, nullptr);
    ASSERT_NE(producer, nullptr);
    ASSERT_NE(consumer, nullptr);

    producer->result_token_ = {IRBuilder().CreateTokenVar(producer->GetSpan())};
    consumer->tokens_.push_back(producer->result_token_.front());
    function->GetVarDependency().AddProducer(producer->result_token_.front(),
                                             std::static_pointer_cast<const ir::Stmt>(producer->shared_from_this()));
    function->GetVarDependency().AddConsumer(producer->result_token_.front(),
                                             std::static_pointer_cast<const ir::Stmt>(consumer->shared_from_this()));

    OoOScheduler scheduler(*function);
    ASSERT_EQ(scheduler.Init(function->Operations().DuplicatedOpList()), SUCCESS);
    EXPECT_EQ(scheduler.state_.depManager.GetPredecessors(consumer).count(producer), 1U);
    EXPECT_EQ(scheduler.state_.depManager.GetSuccessors(producer).count(consumer), 1U);
}

TEST_F(ScheduleOoOTest, TokenDependencySkipsUnregisteredProducer)
{
    ComputationalGraphBuilder graph;
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"in1", "out1", "in2", "out2"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_ADDS, {"in1"}, {"out1"}, "Producer", true));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_MULS, {"in2"}, {"out2"}, "Consumer", true));
    auto* function = graph.GetFunction();
    auto* producer = graph.GetOp("Producer");
    auto* consumer = graph.GetOp("Consumer");
    ASSERT_NE(function, nullptr);
    ASSERT_NE(producer, nullptr);
    ASSERT_NE(consumer, nullptr);

    auto token = IRBuilder().CreateTokenVar(producer->GetSpan());
    producer->result_token_ = {token};
    consumer->tokens_.push_back(token);
    function->GetVarDependency().AddProducer(token,
                                             std::static_pointer_cast<const ir::Stmt>(producer->shared_from_this()));
    function->GetVarDependency().AddConsumer(token,
                                             std::static_pointer_cast<const ir::Stmt>(consumer->shared_from_this()));

    DependencyManager depManager;
    ASSERT_EQ(depManager.InitDependencies({consumer}, false), SUCCESS);
    EXPECT_TRUE(depManager.HasOp(consumer));
    EXPECT_FALSE(depManager.HasOp(producer));
    EXPECT_TRUE(depManager.GetPredecessors(consumer).empty());
}

// 任务切分直读 IR 连边、不查依赖图, 所以 token 边要单独补一趟。
TEST_F(ScheduleOoOTest, TaskSplitSeesTokenEdge)
{
    ComputationalGraphBuilder graph;
    EXPECT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"in1", "out1", "in2", "out2"}));
    EXPECT_TRUE(graph.AddOp(Opcode::OP_ADDS, {"in1"}, {"out1"}, "TokenProducer", true));
    EXPECT_TRUE(graph.AddOp(Opcode::OP_MULS, {"in2"}, {"out2"}, "TokenConsumer", true));
    auto* function = graph.GetFunction();
    auto* producer = graph.GetOp("TokenProducer");
    auto* consumer = graph.GetOp("TokenConsumer");
    ASSERT_NE(function, nullptr);
    ASSERT_NE(producer, nullptr);
    ASSERT_NE(consumer, nullptr);

    // 两条链之间没有任何数据边, 唯一的联系就是这条 token。
    auto token = IRBuilder().CreateTokenVar(producer->GetSpan());
    producer->result_token_ = {token};
    consumer->tokens_.push_back(token);
    function->GetVarDependency().AddProducer(token,
                                             std::static_pointer_cast<const ir::Stmt>(producer->shared_from_this()));
    function->GetVarDependency().AddConsumer(token,
                                             std::static_pointer_cast<const ir::Stmt>(consumer->shared_from_this()));

    TaskSplitter splitter;
    splitter.opList_ = function->Operations(false).DuplicatedOpList();
    splitter.BuildOpGraph();
    // 一个 op 一个 task, 这样 task 图上的边就是 op 之间的边, 判据不受聚簇策略干扰。
    int opNum = static_cast<int>(splitter.opList_.size());
    std::vector<int> clusterIds(opNum);
    for (int i = 0; i < opNum; i++) {
        clusterIds[i] = i;
    }
    std::vector<std::set<int>> inGraph;
    std::vector<std::set<int>> outGraph;
    splitter.BuildInOutGraph(inGraph, outGraph, clusterIds, opNum);

    int producerIdx = splitter.opMagicToIdx_[producer->GetOpMagic()];
    int consumerIdx = splitter.opMagicToIdx_[consumer->GetOpMagic()];
    EXPECT_EQ(outGraph[producerIdx].count(consumerIdx), 1U);
    EXPECT_EQ(inGraph[consumerIdx].count(producerIdx), 1U);
}

TEST_F(ScheduleOoOTest, SimtOpUses216KBForUbScheduling)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_ROWMAX_SINGLE,
                                Opcode::OP_ADD,      Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t3"}, {"t2"}, {"t4", "t5"}, {"t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t4"}, {"t5"},       {"t6"}, {"t8"},
                                                    {"t2"}, {"t4"}, {"t5", "t6"}, {"t8"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3",  "Alloc4", "Alloc5",
                                     "Copyin1", "Copyin2", "RowMax1", "Add1",   "Copyout1"};
    ASSERT_TRUE(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0));
    ASSERT_TRUE(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true));
    Function* function = subGraph.GetFunction();
    ASSERT_NE(function, nullptr);
    Operation* simtOp = subGraph.GetOp("Add1");
    ASSERT_NE(simtOp, nullptr);
    simtOp->SetAttribute(OP_ATTR_PREFIX + "requires_simt", true);

    OoOScheduler scheduler(*function);
    ASSERT_EQ(scheduler.Init(function->Operations().DuplicatedOpList()), SUCCESS);
    EXPECT_EQ(scheduler.state_.localMemSize.at(MemoryType::MEM_UB), A5_SIMT_DYNAMIC_UB_SIZE);
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(ScheduleOoOTest, TestDependenciesView)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_VIEW,
                                Opcode::OP_VIEW,     Opcode::OP_VIEW,     Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {"t1"}, {"t2"}, {"t2"}, {"t2"}, {"t3", "t4"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t6"}, {"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Copyin1", "View1", "View2", "View3", "Add1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t3");
    tensor1->memoryrange.memId = subGraph.GetTensor("t2")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t4");
    tensor2->memoryrange.memId = subGraph.GetTensor("t2")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor3 = subGraph.GetTensor("t5");
    tensor3->memoryrange.memId = subGraph.GetTensor("t2")->memoryrange.memId;
    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    Operation* copyin = subGraph.GetOp("Copyin1");
    EXPECT_NE(copyin, nullptr);
    Operation* add = subGraph.GetOp("Add1");
    EXPECT_NE(add, nullptr);
    EXPECT_TRUE(ooOScheduler.state_.depManager.GetPredecessors(copyin).count(subGraph.GetOp("Alloc1")) > 0);
    EXPECT_TRUE(ooOScheduler.state_.depManager.GetPredecessors(add).count(subGraph.GetOp("Alloc2")) > 0);
    EXPECT_TRUE(ooOScheduler.state_.depManager.GetPredecessors(add).count(subGraph.GetOp("Copyin1")) > 0);
    EXPECT_TRUE(CheckSkipOps(ooOScheduler.GetSkipOps(add), subGraph.GetOp("View1")));
    EXPECT_TRUE(CheckSkipOps(ooOScheduler.GetSkipOps(add), subGraph.GetOp("View2")));
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestDependenciesReshape)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_RESHAPE,
                                Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {"t1"}, {"t2"}, {"t3", "t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t4"}, {"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Copyin1", "Reshape1", "Add1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    std::shared_ptr<LogicalTensor> reshapeOut = subGraph.GetTensor("t3");
    reshapeOut->memoryrange.memId = subGraph.GetTensor("t2")->memoryrange.memId;

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);

    Operation* add = subGraph.GetOp("Add1");
    Operation* reshape = subGraph.GetOp("Reshape1");
    EXPECT_NE(add, nullptr);
    EXPECT_NE(reshape, nullptr);
    EXPECT_TRUE(ooOScheduler.state_.depManager.GetPredecessors(add).count(subGraph.GetOp("Copyin1")) > 0);
    EXPECT_TRUE(ooOScheduler.state_.depManager.GetPredecessors(add).count(reshape) == 0);
    EXPECT_TRUE(CheckSkipOps(ooOScheduler.GetSkipOps(add), reshape));
    EXPECT_EQ(ooOScheduler.GetSkipOps(add).size(), 1UL);
}

// copyin -> t2 -> reshape1 -> t3 -+-> reshape2 -> t4 -> add(t4, t4)
//                                 +-> mul(t3, t3)
// t3 是链中间的张量且有旁支消费者 mul: 克隆范围必须覆盖到链尾 reshape2, 且只动 add 那一支。
TEST_F(ScheduleOoOTest, TestSpillClonesSharedReshapeChain)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,
                                Opcode::OP_RESHAPE,  Opcode::OP_RESHAPE,  Opcode::OP_ADD,      Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {"t1"}, {"t2"}, {"t3"}, {"t4", "t4"}, {"t3", "t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t5"}, {"t6"}, {"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Copyin1", "Reshape1", "Reshape2", "Add1", "Mul1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    int spillMemId = subGraph.GetTensor("t2")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> midTensor = subGraph.GetTensor("t3");
    std::shared_ptr<LogicalTensor> tailOut = subGraph.GetTensor("t4");

    OoOScheduler ooOScheduler(*function);
    EXPECT_EQ(ooOScheduler.Init(function->Operations().DuplicatedOpList()), SUCCESS);

    Operation* reshape1 = subGraph.GetOp("Reshape1");
    Operation* reshape2 = subGraph.GetOp("Reshape2");
    Operation* add = subGraph.GetOp("Add1");
    Operation* mul = subGraph.GetOp("Mul1");
    LogicalTensorPtr reshape1InBefore = reshape1->GetInputOperand(0);

    midTensor->memoryrange.memId = spillMemId;
    tailOut->memoryrange.memId = spillMemId;
    EXPECT_EQ(ooOScheduler.GetSkipOps(add).size(), 2UL);

    LogicalTensorPtr reloadTensor = subGraph.GetTensor("t2")->Clone(*function, true);
    reloadTensor->memoryrange.memId = spillMemId + 1000;
    EXPECT_EQ(ooOScheduler.spillEngine_.UpdateOperationInput(add, subGraph.GetOp("Copyin1"), reloadTensor, spillMemId),
              SUCCESS);

    EXPECT_EQ(reshape1->GetInputOperand(0), reshape1InBefore);
    EXPECT_EQ(reshape2->GetInputOperand(0), midTensor);
    EXPECT_EQ(mul->GetIOperands()[0], midTensor);
    EXPECT_EQ(midTensor->memoryrange.memId, spillMemId);
    EXPECT_NE(add->GetIOperands()[0], tailOut);
    EXPECT_EQ(add->GetIOperands()[0]->memoryrange.memId, reloadTensor->memoryrange.memId);
    EXPECT_EQ(add->GetIOperands()[1]->memoryrange.memId, reloadTensor->memoryrange.memId);
    EXPECT_EQ(add->GetIOperands()[0]->Datatype(), tailOut->Datatype());
    EXPECT_EQ(add->GetIOperands()[0]->Format(), tailOut->Format());
    EXPECT_EQ(add->GetIOperands()[0]->GetShape(), tailOut->GetShape());
    EXPECT_EQ(add->GetIOperands()[0]->GetRawTensor()->rawshape, tailOut->GetRawTensor()->rawshape);

    // add 两个 operand 都改指到克隆链, 原链整条从 add 自己的 skipOps 摘掉;
    // 链首 reshape1 还有旁支 mul 读着所以不标删, 由 mul 的 skipOps 带走。
    auto& skipOps = ooOScheduler.GetSkipOps(add);
    EXPECT_EQ(skipOps.size(), 4UL);
    EXPECT_FALSE(CheckSkipOps(skipOps, reshape1));
    EXPECT_FALSE(CheckSkipOps(skipOps, reshape2));
    EXPECT_TRUE(CheckSkipOps(ooOScheduler.GetSkipOps(mul), reshape1));
    EXPECT_FALSE(reshape1->IsDeleted());
    EXPECT_TRUE(reshape2->IsDeleted());

    Operation* cloneTail = *add->GetIOperands()[0]->GetProducers().begin();
    Operation* cloneHead = *cloneTail->GetInputOperand(0)->GetProducers().begin();
    EXPECT_NE(cloneTail, reshape2);
    EXPECT_NE(cloneHead, reshape1);
    EXPECT_TRUE(CheckSkipOps(skipOps, cloneTail));
    EXPECT_TRUE(CheckSkipOps(skipOps, cloneHead));
    EXPECT_EQ(cloneHead->GetInputOperand(0), reloadTensor);
    EXPECT_EQ(cloneTail->GetOutputOperand(0)->memoryrange.memId, reloadTensor->memoryrange.memId);
    // 克隆体持有私有 RawTensor, 改它不会连坐原链。
    EXPECT_NE(cloneTail->GetOutputOperand(0)->GetRawTensor(), tailOut->GetRawTensor());
    EXPECT_TRUE(tailOut->GetConsumers().empty());
    // 被 spill 的 t2 不受连坐。
    EXPECT_EQ(subGraph.GetTensor("t2")->memoryrange.memId, spillMemId);
}

TEST_F(ScheduleOoOTest, TestDependenciesAssemble)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_DEVICE_DDR};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_SUB,      Opcode::OP_SUB,      Opcode::OP_SUB,
                                Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{}, {"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}};
    std::vector<std::vector<std::string>> ooperands{{"t4"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"t7"}, {"t7"}, {"t8"}};
    std::vector<std::string> opNames{"Alloc1", "Sub1", "Sub2", "Sub3", "Assemble1", "Assemble2", "Assemble3", "Mul1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    for (size_t i = 3; i < tensorNames.size() - 1; i++) {
        EXPECT_NE(subGraph.GetTensor(tensorNames[i]), nullptr);
        std::shared_ptr<LogicalTensor> tensor = subGraph.GetTensor(tensorNames[i]);
        tensor->memoryrange.memId = subGraph.GetTensor("t4")->memoryrange.memId;
    }

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    Operation* alloc = subGraph.GetOp("Alloc1");
    EXPECT_NE(alloc, nullptr);
    Operation* sub = subGraph.GetOp("Sub1");
    EXPECT_NE(sub, nullptr);
    EXPECT_TRUE(ooOScheduler.state_.depManager.GetSuccessors(alloc).count(subGraph.GetOp("Sub3")) > 0);
    EXPECT_TRUE(ooOScheduler.state_.depManager.GetPredecessors(sub).count(subGraph.GetOp("Alloc1")) > 0);
    EXPECT_TRUE(ooOScheduler.state_.depManager.GetSuccessors(sub).count(subGraph.GetOp("Assemble1")) > 0);
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestDependenciesInplace)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_ADD, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {"t1"}, {"t2"}, {"t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"Alloc1", "Copyin1", "Add1", "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t3"), nullptr);
    std::shared_ptr<LogicalTensor> tensor = subGraph.GetTensor("t3");
    tensor->memoryrange.memId = subGraph.GetTensor("t2")->memoryrange.memId;

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    Operation* add = subGraph.GetOp("Add1");
    EXPECT_NE(add, nullptr);
    EXPECT_TRUE(ooOScheduler.state_.depManager.GetSuccessors(add).count(subGraph.GetOp("Copyout1")) > 0);
    EXPECT_TRUE(ooOScheduler.state_.depManager.GetPredecessors(add).count(subGraph.GetOp("Copyin1")) > 0);
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestDependenciesTrue)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR};
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN, Opcode::OP_UB_ALLOC, Opcode::OP_ADD, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {}, {"t2"}, {"t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"Copyin1", "Alloc1", "Add1", "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t3"), nullptr);
    std::shared_ptr<LogicalTensor> tensor = subGraph.GetTensor("t3");
    tensor->memoryrange.memId = subGraph.GetTensor("t2")->memoryrange.memId;

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    std::rotate(ooOScheduler.state_.orderedOps.begin(), ooOScheduler.state_.orderedOps.begin() + 1,
                ooOScheduler.state_.orderedOps.end());
    res = ooOScheduler.state_.depManager.InitDependencies(ooOScheduler.state_.orderedOps, false);
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestSpillCopyIn)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_ADD,
                                Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{},     {},           {},           {},           {},    {"t1"},
                                                    {"t2"}, {"t3", "t4"}, {"t3", "t5"}, {"t4", "t6"}, {"t8"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t3"},
                                                    {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2", "Alloc3", "Alloc4", "Alloc5",  "Copyin1",
                                     "Copyin2", "Add1",   "Add2",   "Add3",   "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SeqSchedule();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(ooOScheduler.state_.orderedOps[8]->GetOpcodeStr(), "UB_ALLOC");
    EXPECT_EQ(ooOScheduler.state_.orderedOps[9]->GetOpcodeStr(), "COPY_IN");
}

TEST_F(ScheduleOoOTest, TestSpill)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,
                                Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD,
                                Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{
        {}, {}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3", "t4"}, {"t3", "t4"}, {"t4", "t6"}, {"t5", "t8"}, {"t9"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t9"}, {"t3"},
                                                    {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t9"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Alloc6",  "Copyin1",
                                     "Copyin2", "Add1",   "Add2",   "Add3",   "Add4",   "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SeqSchedule();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(ooOScheduler.state_.orderedOps[8]->GetOpcodeStr(), "COPY_OUT");
    EXPECT_EQ(ooOScheduler.state_.orderedOps[13]->GetOpcodeStr(), "UB_ALLOC");
    EXPECT_EQ(ooOScheduler.state_.orderedOps[14]->GetOpcodeStr(), "COPY_IN");
}

TEST_F(ScheduleOoOTest, TestSpillInplace)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,
                                Opcode::OP_COPY_IN,  Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{},     {},     {},     {},           {},           {"t1"},
                                                    {"t2"}, {"t3"}, {"t4"}, {"t5", "t6"}, {"t7", "t8"}, {"t9", "t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t6"}, {"t7"}, {"t8"},  {"t9"}, {"t5"},
                                                    {"t6"}, {"t7"}, {"t8"}, {"t10"}, {"t9"}, {"t11"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3",  "Alloc4", "Alloc5", "Copyin1",
                                     "Copyin2", "Copyin3", "Copyin4", "Add1",   "Add2",   "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t11"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t10");
    tensor1->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t11");
    tensor2->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    std::rotate(ooOScheduler.state_.orderedOps.begin(), ooOScheduler.state_.orderedOps.begin() + 6,
                ooOScheduler.state_.orderedOps.begin() + 11);
    res = ooOScheduler.SeqSchedule();
    EXPECT_EQ(res, SUCCESS);
    Operation* add1 = subGraph.GetOp("Add1");
    EXPECT_NE(add1, nullptr);
    Operation* add3 = subGraph.GetOp("Add3");
    EXPECT_NE(add3, nullptr);
    EXPECT_EQ((*ooOScheduler.state_.depManager.GetSuccessors(add1).begin())->GetOpcodeStr(), "COPY_OUT");
    EXPECT_EQ(ooOScheduler.state_.depManager.GetPredecessors(add3).size(), 3);
}

TEST_F(ScheduleOoOTest, TestSpillMultiTensor)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,
                                Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_ADD,
                                Opcode::OP_ADD,      Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{},
                                                    {},
                                                    {},
                                                    {},
                                                    {},
                                                    {},
                                                    {},
                                                    {"t1"},
                                                    {"t2"},
                                                    {"t3"},
                                                    {"t4"},
                                                    {"t7", "t8"},
                                                    {"t5", "t6", "t9"},
                                                    {"t7", "t8", "t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}, {"t10"}, {"t11"},
                                                    {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}, {"t10"}, {"t11"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3",  "Alloc4",  "Alloc5", "Alloc6", "Alloc7",
                                     "Copyin1", "Copyin2", "Copyin3", "Copyin4", "Add1",   "Add2",   "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {50, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t6"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t5");
    tensor1->shape = {128, 128};
    tensor1->tensor->rawshape = {128, 128};
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->shape = {128, 128};
    tensor2->tensor->rawshape = {128, 128};

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SeqSchedule();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(ooOScheduler.state_.orderedOps.size(), 21);
    for (auto* op : ooOScheduler.state_.orderedOps) {
        if (op->GetOpcode() != Opcode::OP_COPY_IN) {
            continue;
        }
        auto output = op->GetOutputOperand(0);
        ASSERT_NE(output, nullptr);
        size_t liveConsumers = 0;
        for (auto* consumer : output->GetConsumers()) {
            if (consumer != nullptr && !consumer->IsDeleted()) {
                liveConsumers++;
            }
        }
        EXPECT_GT(liveConsumers, 0u) << "copyin without live consumer: " << op->Dump();
    }
}

TEST_F(ScheduleOoOTest, TestSpillView)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_VIEW,
                                Opcode::OP_VIEW,     Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{},     {},     {},     {},     {},     {"t1"},
                                                    {"t2"}, {"t3"}, {"t3"}, {"t5"}, {"t6"}, {"t4", "t7"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t7"}, {"t8"}, {"t9"}, {"t3"},
                                                    {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1",
                                     "Copyin2", "View1",  "View2",  "Add1",   "Add2",   "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t6"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t5");
    tensor1->memoryrange.memId = subGraph.GetTensor("t3")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->memoryrange.memId = subGraph.GetTensor("t3")->memoryrange.memId;

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SeqSchedule();
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestSpillAssemble)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3",  "t4",  "t5",  "t6",  "t7",
                                         "t8", "t9", "t10", "t11", "t12", "t13", "t14"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD,
                                Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_SUB,
                                Opcode::OP_SUB,      Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE,
                                Opcode::OP_ASSEMBLE, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{},     {},     {},      {},      {},      {"t1"},
                                                    {"t2"}, {"t3"}, {"t4"},  {"t5"},  {"t6"},  {"t7"},
                                                    {"t8"}, {"t9"}, {"t10"}, {"t11"}, {"t12"}, {"t13"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"},  {"t6"},  {"t7"},  {"t8"},  {"t12"}, {"t5"},
                                                    {"t6"},  {"t7"},  {"t8"},  {"t9"},  {"t10"}, {"t11"},
                                                    {"t12"}, {"t13"}, {"t13"}, {"t13"}, {"t13"}, {"t14"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",    "Alloc3",    "Alloc4",    "Alloc5",    "Copyin1",
                                     "Copyin2", "Copyin3",   "Copyin4",   "Add1",      "Add2",      "Sub1",
                                     "Sub2",    "Assemble1", "Assemble2", "Assemble3", "Assemble4", "Mul1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 64}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t9"), nullptr);
    std::shared_ptr<LogicalTensor> tensor0 = subGraph.GetTensor("t9");
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t10");
    tensor0->tensor->rawshape = {128, 192};
    tensor1->tensor->rawshape = {128, 192};
    tensor1->memoryrange.memId = subGraph.GetTensor("t9")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t11");
    tensor2->memoryrange.memId = subGraph.GetTensor("t9")->memoryrange.memId;
    tensor2->shape = {128, 128};
    tensor2->tensor->rawshape = {128, 192};
    std::shared_ptr<LogicalTensor> tensor3 = subGraph.GetTensor("t12");
    tensor3->memoryrange.memId = subGraph.GetTensor("t9")->memoryrange.memId;
    tensor3->shape = {128, 128};
    tensor3->tensor->rawshape = {128, 192};
    std::shared_ptr<LogicalTensor> tensor4 = subGraph.GetTensor("t13");
    tensor4->memoryrange.memId = subGraph.GetTensor("t9")->memoryrange.memId;
    tensor4->shape = {128, 192};
    tensor4->tensor->rawshape = {128, 192};
    std::shared_ptr<LogicalTensor> tensor5 = subGraph.GetTensor("t14");
    tensor5->memoryrange.memId = subGraph.GetTensor("t9")->memoryrange.memId;
    tensor5->shape = {128, 192};
    tensor5->tensor->rawshape = {128, 192};
    std::shared_ptr<LogicalTensor> tensor6 = subGraph.GetTensor("t7");
    tensor6->shape = {128, 128};
    tensor6->tensor->rawshape = {128, 128};
    std::shared_ptr<LogicalTensor> tensor7 = subGraph.GetTensor("t8");
    tensor7->shape = {128, 128};
    tensor7->tensor->rawshape = {128, 128};
    std::vector<int64_t> offset1 = {0, 0};
    std::vector<int64_t> offset2 = {0, 128};
    std::vector<int64_t> offset3 = {64, 0};
    std::vector<int64_t> offset4 = {64, 128};
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset1);
    auto assemble1 = subGraph.GetOp("Assemble1");
    assemble1->SetOpAttribute(assembleAttr);
    auto assembleAttr2 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset2);
    auto assemble2 = subGraph.GetOp("Assemble2");
    assemble2->SetOpAttribute(assembleAttr2);
    auto assembleAttr3 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset3);
    auto assemble3 = subGraph.GetOp("Assemble3");
    assemble3->SetOpAttribute(assembleAttr3);
    auto assembleAttr4 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset4);
    auto assemble4 = subGraph.GetOp("Assemble4");
    assemble4->SetOpAttribute(assembleAttr4);

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SeqSchedule();
    EXPECT_EQ(res, SUCCESS);
}

// 写后写不变量: 凡是排在回载之后又跟它写同一片字节的 op, 回载都得是它的前驱。
static void ExpectReloadOrderedBeforeConflictingWrites(ScheduleState& state)
{
    for (auto* copyin : state.orderedOps) {
        if (COPY_IN_OPS.count(copyin->GetOpcode()) == 0) {
            continue;
        }
        for (const auto& reloaded : copyin->GetOOperands()) {
            if (reloaded == nullptr) {
                continue;
            }
            for (auto* other : state.orderedOps) {
                if (other == copyin || state.IsOpRetired(other) || USE_LESS_OPS.count(other->GetOpcode()) != 0 ||
                    state.GetExecOrder(other) <= state.GetExecOrder(copyin)) {
                    continue;
                }
                bool conflicts = false;
                for (const auto& written : other->GetOOperands()) {
                    if (written == nullptr || written->memoryrange.memId != reloaded->memoryrange.memId ||
                        written->shape.size() != reloaded->shape.size()) {
                        continue;
                    }
                    conflicts = conflicts || IsOverlapping(*reloaded, *written);
                }
                if (!conflicts) {
                    continue;
                }
                EXPECT_EQ(state.depManager.GetSuccessors(copyin).count(other), 1U)
                    << "reload " << state.GetOpInfo(copyin) << " is not ordered before conflicting write "
                    << state.GetOpInfo(other);
            }
        }
    }
}

// 分片回载会把还没执行的原写改指到新 buffer, 于是它和回载写同一块地 —— 这正是规则 R 要接住的冲突。
TEST_F(ScheduleOoOTest, SpillReloadOrderedBeforeConflictingWrite)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,
                                Opcode::OP_COPY_IN,  Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{},     {},     {},     {},           {},           {"t1"},
                                                    {"t2"}, {"t3"}, {"t4"}, {"t5", "t6"}, {"t7", "t8"}, {"t9", "t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t6"}, {"t7"}, {"t8"},  {"t9"}, {"t5"},
                                                    {"t6"}, {"t7"}, {"t8"}, {"t10"}, {"t9"}, {"t11"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3",  "Alloc4", "Alloc5", "Copyin1",
                                     "Copyin2", "Copyin3", "Copyin4", "Add1",   "Add2",   "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    ASSERT_NE(function, nullptr);

    // t10/t11 复用 t5 的地: 腾走 t5 之后这几个写会被改指到回载出来的新 buffer。
    subGraph.GetTensor("t10")->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;
    subGraph.GetTensor("t11")->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    ASSERT_EQ(sort.SortOps(), SUCCESS);
    OoOScheduler ooOScheduler(*function);
    ASSERT_EQ(ooOScheduler.Init(sort.operations), SUCCESS);
    std::rotate(ooOScheduler.state_.orderedOps.begin(), ooOScheduler.state_.orderedOps.begin() + 6,
                ooOScheduler.state_.orderedOps.begin() + 11);
    ASSERT_EQ(ooOScheduler.SeqSchedule(), SUCCESS);

    ExpectReloadOrderedBeforeConflictingWrites(ooOScheduler.state_);
}

TEST_F(ScheduleOoOTest, TestSchedule)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,
                                Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD,
                                Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{
        {}, {}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3", "t4"}, {"t3", "t4"}, {"t4", "t6"}, {"t5", "t8"}, {"t9"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t9"}, {"t3"},
                                                    {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t9"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Alloc6",  "Copyin1",
                                     "Copyin2", "Add1",   "Add2",   "Add3",   "Add4",   "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Schedule(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    Operation* add = subGraph.GetOp("Add2");
    EXPECT_NE(add, nullptr);
    EXPECT_EQ(add->oOperand[0]->memoryrange.start, 32768);
    EXPECT_EQ(add->oOperand[0]->memoryrange.end, 49152);
}

TEST_F(ScheduleOoOTest, TestScheduleInplace)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,
                                Opcode::OP_COPY_IN,  Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{},     {},     {},     {},           {},           {"t1"},
                                                    {"t2"}, {"t3"}, {"t4"}, {"t5", "t6"}, {"t7", "t8"}, {"t9", "t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t6"}, {"t7"}, {"t8"},  {"t9"}, {"t5"},
                                                    {"t6"}, {"t7"}, {"t8"}, {"t10"}, {"t9"}, {"t11"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3",  "Alloc4", "Alloc5", "Copyin1",
                                     "Copyin2", "Copyin3", "Copyin4", "Add1",   "Add2",   "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t11"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t10");
    tensor1->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t11");
    tensor2->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Schedule(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    Operation* copyin = subGraph.GetOp("Copyin1");
    EXPECT_NE(copyin, nullptr);
    Operation* add1 = subGraph.GetOp("Add1");
    EXPECT_NE(add1, nullptr);
    Operation* add3 = subGraph.GetOp("Add3");
    EXPECT_NE(add3, nullptr);
    EXPECT_EQ(copyin->oOperand[0]->memoryrange.start, 49152);
    EXPECT_EQ(copyin->oOperand[0]->memoryrange.end, 65536);
    EXPECT_EQ(add1->oOperand[0]->memoryrange.start, 49152);
    EXPECT_EQ(add1->oOperand[0]->memoryrange.end, 65536);
    EXPECT_EQ(add3->oOperand[0]->memoryrange.start, 49152);
    EXPECT_EQ(add3->oOperand[0]->memoryrange.end, 65536);
}

TEST_F(ScheduleOoOTest, TestScheduleView)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_VIEW,
                                Opcode::OP_VIEW,     Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{},     {},     {},     {},     {},     {"t1"},
                                                    {"t2"}, {"t3"}, {"t3"}, {"t5"}, {"t6"}, {"t4", "t7"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t7"}, {"t8"}, {"t9"}, {"t3"},
                                                    {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1",
                                     "Copyin2", "View1",  "View2",  "Add1",   "Add2",   "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t6"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t5");
    tensor1->memoryrange.memId = subGraph.GetTensor("t3")->memoryrange.memId;
    tensor1->shape = {32, 32};
    tensor1->tensor->rawshape = {64, 64};
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->memoryrange.memId = subGraph.GetTensor("t3")->memoryrange.memId;
    tensor2->shape = {32, 32};
    tensor2->tensor->rawshape = {64, 64};

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Schedule(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    Operation* copyin = subGraph.GetOp("Copyin1");
    EXPECT_NE(copyin, nullptr);
    EXPECT_EQ(copyin->oOperand[0]->memoryrange.start, 0);
    EXPECT_EQ(copyin->oOperand[0]->memoryrange.end, 16384);
    EXPECT_EQ(subGraph.GetOp("View1")->GetOutputOperand(0)->memoryrange.start, 0);
    EXPECT_EQ(subGraph.GetOp("View1")->GetOutputOperand(0)->memoryrange.end, 16384);
    EXPECT_EQ(subGraph.GetOp("View2")->GetOutputOperand(0)->memoryrange.start, 0);
    EXPECT_EQ(subGraph.GetOp("View2")->GetOutputOperand(0)->memoryrange.end, 16384);
}

TEST_F(ScheduleOoOTest, TestScheduleAssemble)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,
                                Opcode::OP_COPY_IN,  Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_ADD,
                                Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{},     {},     {},     {},     {},           {"t1"},       {"t2"},
                                                    {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7", "t8"}, {"t9", "t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t7"}, {"t8"}, {"t10"}, {"t11"}, {"t5"}, {"t6"},
                                                    {"t7"}, {"t8"}, {"t9"}, {"t9"},  {"t10"}, {"t11"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3",    "Alloc4",    "Alloc5", "Copyin1", "Copyin2",
                                     "Copyin3", "Copyin4", "Assemble1", "Assemble2", "Add1",   "Add2"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t9"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t9");
    tensor1->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;
    tensor2->shape = {32, 32};
    tensor2->tensor->rawshape = {64, 64};
    std::shared_ptr<LogicalTensor> tensor3 = subGraph.GetTensor("t5");
    tensor3->shape = {32, 32};
    tensor3->tensor->rawshape = {64, 64};

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Schedule(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    Operation* copyin1 = subGraph.GetOp("Copyin1");
    EXPECT_NE(copyin1, nullptr);
    Operation* copyin2 = subGraph.GetOp("Copyin2");
    EXPECT_NE(copyin2, nullptr);
    Operation* assemble1 = subGraph.GetOp("Assemble1");
    EXPECT_NE(assemble1, nullptr);
    Operation* assemble2 = subGraph.GetOp("Assemble2");
    EXPECT_NE(assemble2, nullptr);
    EXPECT_EQ(copyin1->oOperand[0]->memoryrange.start, 0);
    EXPECT_EQ(copyin1->oOperand[0]->memoryrange.end, 16384);
    EXPECT_EQ(copyin2->oOperand[0]->memoryrange.start, 0);
    EXPECT_EQ(copyin2->oOperand[0]->memoryrange.end, 16384);
    EXPECT_EQ(assemble1->oOperand[0]->memoryrange.start, 0);
    EXPECT_EQ(assemble1->oOperand[0]->memoryrange.end, 16384);
    EXPECT_EQ(assemble2->oOperand[0]->memoryrange.start, 0);
    EXPECT_EQ(assemble2->oOperand[0]->memoryrange.end, 16384);
}

TEST_F(ScheduleOoOTest, TestScheduleSpillCopyIn)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_ADD,
                                Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{},     {},           {},           {},           {},    {"t1"},
                                                    {"t2"}, {"t3", "t4"}, {"t3", "t5"}, {"t4", "t6"}, {"t8"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t3"},
                                                    {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2", "Alloc3", "Alloc4", "Alloc5",  "Copyin1",
                                     "Copyin2", "Add1",   "Add2",   "Add3",   "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(ooOScheduler.state_.newOperations[8]->GetOpcodeStr(), "UB_ALLOC");
    EXPECT_EQ(ooOScheduler.state_.newOperations[8]->oOperand[0]->memoryrange.start, 0);
    EXPECT_EQ(ooOScheduler.state_.newOperations[8]->oOperand[0]->memoryrange.end, 65536);
    EXPECT_EQ(ooOScheduler.state_.newOperations[10]->GetOpcodeStr(), "COPY_IN");
    EXPECT_EQ(ooOScheduler.state_.newOperations[10]->oOperand[0]->memoryrange.start, 0);
    EXPECT_EQ(ooOScheduler.state_.newOperations[10]->oOperand[0]->memoryrange.end, 65536);
}

TEST_F(ScheduleOoOTest, TestScheduleSpill)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,
                                Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD,
                                Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{
        {}, {}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3", "t4"}, {"t3", "t4"}, {"t4", "t6"}, {"t5", "t8"}, {"t9"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t9"}, {"t3"},
                                                    {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t9"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Alloc6",  "Copyin1",
                                     "Copyin2", "Add1",   "Add2",   "Add3",   "Add4",   "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(ooOScheduler.state_.newOperations[8]->GetOpcodeStr(), "COPY_OUT");
    EXPECT_EQ(ooOScheduler.state_.newOperations[8]->oOperand[0]->memoryrange.start, 0);
    EXPECT_EQ(ooOScheduler.state_.newOperations[8]->oOperand[0]->memoryrange.end, 65536);
    EXPECT_EQ(ooOScheduler.state_.newOperations[13]->GetOpcodeStr(), "UB_ALLOC");
    EXPECT_EQ(ooOScheduler.state_.newOperations[13]->oOperand[0]->memoryrange.start, 65536);
    EXPECT_EQ(ooOScheduler.state_.newOperations[13]->oOperand[0]->memoryrange.end, 131072);
    EXPECT_EQ(ooOScheduler.state_.newOperations[15]->GetOpcodeStr(), "COPY_IN");
    EXPECT_EQ(ooOScheduler.state_.newOperations[15]->oOperand[0]->memoryrange.start, 65536);
    EXPECT_EQ(ooOScheduler.state_.newOperations[15]->oOperand[0]->memoryrange.end, 131072);
}

TEST_F(ScheduleOoOTest, TestScheduleSpillInplace)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,
                                Opcode::OP_COPY_IN,  Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{},     {},     {},     {},           {},           {"t1"},
                                                    {"t2"}, {"t3"}, {"t4"}, {"t5", "t6"}, {"t7", "t8"}, {"t9", "t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t6"}, {"t7"}, {"t8"},  {"t9"}, {"t5"},
                                                    {"t6"}, {"t7"}, {"t8"}, {"t10"}, {"t9"}, {"t11"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3",  "Alloc4", "Alloc5", "Copyin1",
                                     "Copyin2", "Copyin3", "Copyin4", "Add1",   "Add2",   "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t11"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t10");
    tensor1->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t11");
    tensor2->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    std::rotate(ooOScheduler.state_.orderedOps.begin(), ooOScheduler.state_.orderedOps.begin() + 6,
                ooOScheduler.state_.orderedOps.begin() + 11);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(ooOScheduler.state_.newOperations[6]->GetOpcodeStr(), "COPY_OUT");
    EXPECT_EQ(ooOScheduler.state_.newOperations[13]->GetOpcodeStr(), "COPY_IN");
    EXPECT_EQ(ooOScheduler.state_.newOperations[13]->oOperand[0]->memoryrange.start, 65536);
    EXPECT_EQ(ooOScheduler.state_.newOperations[13]->oOperand[0]->memoryrange.end, 131072);
    EXPECT_EQ(ooOScheduler.state_.newOperations[14]->GetOpcodeStr(), "ADD");
    EXPECT_EQ(ooOScheduler.state_.newOperations[14]->oOperand[0]->memoryrange.start, 65536);
    EXPECT_EQ(ooOScheduler.state_.newOperations[14]->oOperand[0]->memoryrange.end, 131072);
}

TEST_F(ScheduleOoOTest, TestScheduleSpillView)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_VIEW,
                                Opcode::OP_VIEW,     Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{},     {},     {},     {},     {},     {"t1"},
                                                    {"t2"}, {"t3"}, {"t3"}, {"t5"}, {"t6"}, {"t4", "t7"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t7"}, {"t8"}, {"t9"}, {"t3"},
                                                    {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1",
                                     "Copyin2", "View1",  "View2",  "Add1",   "Add2",   "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t6"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t5");
    tensor1->memoryrange.memId = subGraph.GetTensor("t3")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->memoryrange.memId = subGraph.GetTensor("t3")->memoryrange.memId;

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestScheduleSpillAssemble)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,
                                Opcode::OP_COPY_IN,  Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_ADD,
                                Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{},     {},     {},     {},     {},           {"t1"},       {"t2"},
                                                    {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7", "t8"}, {"t9", "t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t7"}, {"t8"}, {"t10"}, {"t11"}, {"t5"}, {"t6"},
                                                    {"t7"}, {"t8"}, {"t9"}, {"t9"},  {"t10"}, {"t11"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3",    "Alloc4",    "Alloc5", "Copyin1", "Copyin2",
                                     "Copyin3", "Copyin4", "Assemble1", "Assemble2", "Add1",   "Add2"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t9"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t9");
    tensor1->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor5 = subGraph.GetTensor("t5");
    std::shared_ptr<LogicalTensor> tensor6 = subGraph.GetTensor("t6");
    tensor5->shape = {64, 128};
    tensor5->tensor->rawshape = {128, 128};
    tensor6->shape = {64, 128};
    tensor6->tensor->rawshape = {128, 128};
    std::vector<int64_t> offset1 = {0, 0};
    std::vector<int64_t> offset2 = {64, 0};
    auto assembleAttr1 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset1);
    auto assemble1 = subGraph.GetOp("Assemble1");
    assemble1->SetOpAttribute(assembleAttr1);
    auto assembleAttr2 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset2);
    auto assemble2 = subGraph.GetOp("Assemble2");
    assemble2->SetOpAttribute(assembleAttr2);

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestSpillMultiProducerBuffer)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"DDR1", "DDR2", "DDR3", "UB1",  "UB2",  "UB3",
                                         "UB4",  "L1_1", "L0C1", "L0C2", "L1_2", "L0C3"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_L1,
        MemoryType::MEM_L0C,        MemoryType::MEM_L0C,        MemoryType::MEM_L1,         MemoryType::MEM_L0C};
    std::vector<Opcode> opCodes{
        Opcode::OP_COPY_IN,    Opcode::OP_COPY_IN,    Opcode::OP_UB_COPY_ND2NZ, Opcode::OP_UB_COPY_ND2NZ,
        Opcode::OP_UB_COPY_L1, Opcode::OP_UB_COPY_L1, Opcode::OP_A_MUL_B,       Opcode::OP_A_MUL_B,
        Opcode::OP_COPY_IN,    Opcode::OP_A_MUL_B,    Opcode::OP_UB_ALLOC,      Opcode::OP_UB_ALLOC,
        Opcode::OP_UB_ALLOC,   Opcode::OP_UB_ALLOC,   Opcode::OP_L1_ALLOC,      Opcode::OP_L1_ALLOC,
        Opcode::OP_L0C_ALLOC,  Opcode::OP_L0C_ALLOC,  Opcode::OP_L0C_ALLOC};
    std::vector<std::vector<std::string>> ioperands{{"DDR1"}, {"DDR2"}, {"UB1"},  {"UB2"}, {"UB3"}, {"UB4"}, {"L1_1"},
                                                    {"L1_1"}, {"DDR3"}, {"L1_2"}, {},      {},      {},      {},
                                                    {},       {},       {},       {},      {}};
    std::vector<std::vector<std::string>> ooperands{
        {"UB1"}, {"UB2"}, {"UB3"}, {"UB4"}, {"L1_1"}, {"L1_1"}, {"L0C1"}, {"L0C2"}, {"L1_2"}, {"L0C3"},
        {"UB1"}, {"UB2"}, {"UB3"}, {"UB4"}, {"L1_1"}, {"L1_2"}, {"L0C1"}, {"L0C2"}, {"L0C3"}};
    std::vector<std::string> opNames{"copyin1",  "copyin2",   "ubNd2nz1",  "ubNd2nz2", "ub_l11",
                                     "ub_l12",   "aMulB1",    "aMulB2",    "copyin3",  "aMulB3",
                                     "ubAlloc1", "ubAlloc2",  "ubAlloc3",  "ubAlloc4", "l1Alloc1",
                                     "l1Alloc2", "l0cAlloc1", "l0cAlloc2", "l0cAlloc3"};
    subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0);
    subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true);
    Function* function = subGraph.GetFunction();
    std::shared_ptr<LogicalTensor> tensor5 = subGraph.GetTensor("L1_1");
    std::shared_ptr<LogicalTensor> tensor6 = subGraph.GetTensor("L1_2");
    tensor5->shape = {256, 300};
    tensor5->tensor->rawshape = {256, 300};
    tensor6->shape = {256, 300};
    tensor6->tensor->rawshape = {256, 300};
    auto* ubToL11 = subGraph.GetOp("ub_l11");
    ubToL11->SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MemoryType::MEM_L1,
                                                              OpImmediate::Specified({64, 64}),
                                                              OpImmediate::Specified({256, 256})));
    auto* ubToL12 = subGraph.GetOp("ub_l12");
    ubToL12->SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({64, 0}), MemoryType::MEM_L1,
                                                              OpImmediate::Specified({64, 64}),
                                                              OpImmediate::Specified({256, 256})));
    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    EXPECT_EQ(sort.SortOps(), SUCCESS);
    OoOScheduler ooOScheduler(*function);
    rotate(sort.operations.begin() + 12, sort.operations.begin() + 15, sort.operations.begin() + 18);
    EXPECT_EQ(ooOScheduler.Init(sort.operations), SUCCESS);
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Platform::Instance().ReloadMemoryPaths("3510");
    EXPECT_FALSE(Platform::Instance().GetDie().HasDirectPath(MemoryType::MEM_L1, MemoryType::MEM_DEVICE_DDR));
    EXPECT_EQ(ooOScheduler.SeqSchedule(), SUCCESS);
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
    Platform::Instance().ReloadMemoryPaths("2201");
    EXPECT_TRUE(Platform::Instance().GetDie().HasDirectPath(MemoryType::MEM_L1, MemoryType::MEM_DEVICE_DDR));
}

TEST_F(ScheduleOoOTest, TestSpillMultiProducerBufferNotReady)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"DDR1", "DDR2", "DDR3", "UB1",  "UB2", "UB3",
                                         "UB4",  "L1_1", "L0C1", "L0C2", "L1_2"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_L1,
        MemoryType::MEM_L0C,        MemoryType::MEM_L0C,        MemoryType::MEM_L1};
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN,       Opcode::OP_COPY_IN,    Opcode::OP_UB_COPY_ND2NZ,
                                Opcode::OP_UB_COPY_ND2NZ, Opcode::OP_UB_COPY_L1, Opcode::OP_UB_COPY_L1,
                                Opcode::OP_A_MUL_B,       Opcode::OP_A_MUL_B,    Opcode::OP_L1_COPY_UB,
                                Opcode::OP_UB_ALLOC,      Opcode::OP_UB_ALLOC,   Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC,      Opcode::OP_L1_ALLOC,   Opcode::OP_L1_ALLOC,
                                Opcode::OP_L0C_ALLOC,     Opcode::OP_L0C_ALLOC};
    std::vector<std::vector<std::string>> ioperands{{"DDR1"}, {"DDR2"}, {"UB1"},  {"UB2"}, {"UB3"}, {"UB4"},
                                                    {"L1_1"}, {"L1_1"}, {"L1_2"}, {},      {},      {},
                                                    {},       {},       {},       {},      {}};
    std::vector<std::vector<std::string>> ooperands{{"UB1"},  {"L1_2"}, {"UB3"},  {"UB4"},  {"L1_1"}, {"L1_1"},
                                                    {"L0C1"}, {"L0C2"}, {"UB2"},  {"UB1"},  {"UB2"},  {"UB3"},
                                                    {"UB4"},  {"L1_1"}, {"L1_2"}, {"L0C1"}, {"L0C2"}};
    std::vector<std::string> opNames{"copyin1",  "copyin2",  "ubNd2nz1", "ubNd2nz2",  "ub_l11",   "ub_l12",
                                     "aMulB1",   "aMulB2",   "l1CopyUb", "ubAlloc1",  "ubAlloc2", "ubAlloc3",
                                     "ubAlloc4", "l1Alloc1", "l1Alloc2", "l0cAlloc1", "l0cAlloc2"};
    subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0);
    subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true);
    Function* function = subGraph.GetFunction();
    std::shared_ptr<LogicalTensor> tensor5 = subGraph.GetTensor("L1_1");
    std::shared_ptr<LogicalTensor> tensor6 = subGraph.GetTensor("L1_2");
    tensor5->shape = {256, 300};
    tensor5->tensor->rawshape = {256, 300};
    tensor6->shape = {256, 300};
    tensor6->tensor->rawshape = {256, 300};
    auto* ubToL11 = subGraph.GetOp("ub_l11");
    ubToL11->SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MemoryType::MEM_L1,
                                                              OpImmediate::Specified({64, 64}),
                                                              OpImmediate::Specified({256, 256})));
    auto* ubToL12 = subGraph.GetOp("ub_l12");
    ubToL12->SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({64, 0}), MemoryType::MEM_L1,
                                                              OpImmediate::Specified({64, 64}),
                                                              OpImmediate::Specified({256, 256})));
    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    EXPECT_EQ(sort.SortOps(), SUCCESS);
    OoOScheduler ooOScheduler(*function);
    rotate(sort.operations.begin(), sort.operations.begin() + 8, sort.operations.begin() + 14);
    rotate(sort.operations.begin() + 4, sort.operations.begin() + 12, sort.operations.begin() + 13);
    EXPECT_EQ(ooOScheduler.Init(sort.operations), SUCCESS);
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Platform::Instance().ReloadMemoryPaths("3510");
    EXPECT_FALSE(Platform::Instance().GetDie().HasDirectPath(MemoryType::MEM_L1, MemoryType::MEM_DEVICE_DDR));
    EXPECT_EQ(ooOScheduler.SeqSchedule(), SUCCESS);
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
    Platform::Instance().ReloadMemoryPaths("2201");
    EXPECT_TRUE(Platform::Instance().GetDie().HasDirectPath(MemoryType::MEM_L1, MemoryType::MEM_DEVICE_DDR));
}

// Spill L1 across a reshape/view/viewType skip chain: the copyout must read the real UB
// source, and the skip chain must land in newOperations, in order, before the copyout --
// otherwise ScheduleBy's op-count invariant breaks.
TEST_F(ScheduleOoOTest, TestSpillWalkUpThroughSkipOps)
{
    ComputationalGraphBuilder graph;
    graph.AddTensors(DT_FP32, {16, 32}, {MEM_DEVICE_DDR, MEM_UB}, {"ddr", "source"});
    graph.AddTensors(DT_FP32, {8, 64}, {MEM_UB, MEM_UB, MEM_UB, MEM_UB, MEM_L1},
                     {"reshape", "view", "viewType", "nz", "l1"});
    for (const auto& name : {"reshape", "view", "viewType"}) {
        graph.GetTensor(name)->tensor = graph.GetTensor("source")->tensor;
        graph.GetTensor(name)->memoryrange = graph.GetTensor("source")->memoryrange;
    }
    graph.AddOp(Opcode::OP_UB_ALLOC, {}, {"source"}, "allocSource");
    graph.AddOp(Opcode::OP_COPY_IN, {"ddr"}, {"source"}, "write");
    graph.AddOp(Opcode::OP_RESHAPE, {"source"}, {"reshape"}, "reshape");
    graph.AddOp(Opcode::OP_VIEW, {"reshape"}, {"view"}, "view");
    graph.AddOp(Opcode::OP_VIEW_TYPE, {"view"}, {"viewType"}, "viewType");
    graph.AddOp(Opcode::OP_UB_ALLOC, {}, {"nz"}, "allocNz");
    graph.AddOp(Opcode::OP_UB_COPY_ND2NZ, {"viewType"}, {"nz"}, "nd2nz");
    graph.AddOp(Opcode::OP_L1_ALLOC, {}, {"l1"}, "allocL1");
    graph.AddOp(Opcode::OP_UB_COPY_L1, {"nz"}, {"l1"}, "copyL1");

    OoOScheduler scheduler(*graph.GetFunction());
    ASSERT_EQ(scheduler.Init(graph.GetFunction()->Operations().DuplicatedOpList()), SUCCESS);
    scheduler.state_.schedInfoMap[graph.GetOp("write")].isRetired = true;

    SpillPlan plan;
    ASSERT_EQ(scheduler.spillEngine_.CollectWalkUpSources(graph.GetTensor("l1"), plan), SUCCESS);
    graph.AddTensor(DT_FP32, {8, 64}, MEM_DEVICE_DDR, "mirror");
    SpillContext ctx;
    ASSERT_EQ(scheduler.spillEngine_.SaveSourcesToDDR(graph.GetTensor("mirror"), ctx, plan), SUCCESS);
    auto* copyout = *graph.GetTensor("mirror")->GetProducers().begin();
    EXPECT_EQ(copyout->GetInputOperand(0), graph.GetTensor("viewType"));

    // ND2NZ is absent from newOperations, so no HandleSkipOp ran for the chain.
    scheduler.state_.newOperations = {graph.GetOp("allocSource"), graph.GetOp("write"), graph.GetOp("allocNz"),
                                      graph.GetOp("allocL1"), graph.GetOp("copyL1")};
    ASSERT_EQ(scheduler.ApplySpillContext(ctx, graph.GetOp("allocL1")), SUCCESS);

    auto& ops = scheduler.state_.newOperations;
    auto at = [&](Operation* op) { return std::find(ops.begin(), ops.end(), op) - ops.begin(); };
    EXPECT_LT(at(graph.GetOp("reshape")), at(graph.GetOp("view")));
    EXPECT_LT(at(graph.GetOp("view")), at(graph.GetOp("viewType")));
    EXPECT_LT(at(graph.GetOp("viewType")), at(copyout));
}

// 直接写者就是 ND2NZ 时, WalkUp 第一跳要认出跨分形 (旧逻辑只查第二跳, crossedNd2nz 漏置为假)。
TEST_F(ScheduleOoOTest, TestSpillWalkUpFirstHopIsNd2nz)
{
    ComputationalGraphBuilder graph;
    graph.AddTensors(DT_FP32, {16, 32}, {MEM_DEVICE_DDR, MEM_UB, MEM_UB, MEM_L1}, {"ddr", "nd", "nz", "l1"});
    graph.AddOp(Opcode::OP_UB_ALLOC, {}, {"nd"}, "allocNd");
    graph.AddOp(Opcode::OP_COPY_IN, {"ddr"}, {"nd"}, "write");
    graph.AddOp(Opcode::OP_UB_ALLOC, {}, {"nz"}, "allocNz");
    graph.AddOp(Opcode::OP_UB_COPY_ND2NZ, {"nd"}, {"nz"}, "nd2nz");
    graph.AddOp(Opcode::OP_L1_ALLOC, {}, {"l1"}, "allocL1");
    graph.AddOp(Opcode::OP_UB_COPY_L1, {"nz"}, {"l1"}, "copyL1");

    OoOScheduler scheduler(*graph.GetFunction());
    ASSERT_EQ(scheduler.Init(graph.GetFunction()->Operations().DuplicatedOpList()), SUCCESS);
    scheduler.state_.schedInfoMap[graph.GetOp("write")].isRetired = true;

    SpillPlan plan;
    ASSERT_EQ(scheduler.spillEngine_.CollectWalkUpSources(graph.GetTensor("nz"), plan), SUCCESS);
    EXPECT_TRUE(plan.crossedNd2nz);
    ASSERT_EQ(plan.sources.size(), 1u);
    EXPECT_EQ(plan.sources[0].tensor, graph.GetTensor("nd"));
}

TEST_F(ScheduleOoOTest, TestSpillWalkUpPreservesSymbolicReloadOffset)
{
    ComputationalGraphBuilder graph;
    graph.AddTensors(DT_FP32, {128, 64}, {MEM_DEVICE_DDR, MEM_UB}, {"ddr", "source"});
    graph.AddTensors(DT_FP32, {64, 64}, {MEM_L1, MEM_DEVICE_DDR, MEM_L1}, {"l1", "mirror", "reload"});
    graph.AddOp(Opcode::OP_UB_ALLOC, {}, {"source"}, "allocSource");
    graph.AddOp(Opcode::OP_COPY_IN, {"ddr"}, {"source"}, "write");
    graph.AddOp(Opcode::OP_L1_ALLOC, {}, {"l1"}, "allocL1");
    graph.AddOp(Opcode::OP_UB_COPY_L1, {"source"}, {"l1"}, "copyL1");

    SymbolicScalar runtimeOffset = SymbolicScalar(AddRuntimeCoaPrefix("GET_PARAM_OFFSET"))(
        SymbolicScalar(2), SymbolicScalar(17), SymbolicScalar(0));
    std::vector<OpImmediate> reloadOffset = {OpImmediate::Specified(runtimeOffset), OpImmediate::Parameter(18)};
    graph.GetOp("copyL1")->SetOpAttribute(std::make_shared<CopyOpAttribute>(
        reloadOffset, MemoryType::MEM_L1, OpImmediate::Specified({64, 64}), OpImmediate::Specified({128, 64})));

    OoOScheduler scheduler(*graph.GetFunction());
    ASSERT_EQ(scheduler.Init(graph.GetFunction()->Operations().DuplicatedOpList()), SUCCESS);
    scheduler.state_.schedInfoMap[graph.GetOp("write")].isRetired = true;

    SpillPlan plan;
    ASSERT_EQ(scheduler.spillEngine_.CollectWalkUpSources(graph.GetTensor("l1"), plan), SUCCESS);
    ASSERT_EQ(plan.reloadOffset.size(), reloadOffset.size());
    EXPECT_TRUE(plan.reloadOffset[0].IsSpecified());
    EXPECT_EQ(plan.reloadOffset[0].GetSpecifiedValue().Dump(), runtimeOffset.Dump());
    EXPECT_TRUE(plan.reloadOffset[1].IsParameter());
    EXPECT_EQ(plan.reloadOffset[1].GetParameterIndex(), 18);

    Operation* reload = scheduler.spillEngine_.CreateWholeReload(graph.GetTensor("mirror"), graph.GetTensor("reload"),
                                                                 plan);
    auto attr = std::dynamic_pointer_cast<CopyOpAttribute>(reload->GetOpAttribute());
    ASSERT_NE(attr, nullptr);
    const auto& recreatedOffset = attr->GetFromOffset();
    ASSERT_EQ(recreatedOffset.size(), reloadOffset.size());
    EXPECT_TRUE(recreatedOffset[0].IsSpecified());
    EXPECT_EQ(recreatedOffset[0].GetSpecifiedValue().Dump(), runtimeOffset.Dump());
    EXPECT_TRUE(recreatedOffset[1].IsParameter());
    EXPECT_EQ(recreatedOffset[1].GetParameterIndex(), 18);
}

TEST_F(ScheduleOoOTest, TestSpillL0CMultiConsumer)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::tuple<DataType, MemoryType, std::string>> tensors{
        {DT_FP32, MEM_L0A, "L0A"},          {DT_FP32, MEM_L0B, "L0B"},  {DT_FP32, MEM_L0C, "L0C1"},
        {DT_FP32, MEM_L0C, "L0C2"},         {DT_FP16, MEM_UB, "UBDst"}, {DT_FP32, MEM_L1, "L1Dst"},
        {DT_FP32, MEM_DEVICE_DDR, "DDROut"}};
    for (auto& [dt, mem, name] : tensors) {
        subGraph.AddTensor(dt, {128, 128}, mem, name, 0);
    }
    std::vector<Opcode> opCodes{Opcode::OP_L0A_ALLOC, Opcode::OP_L0B_ALLOC, Opcode::OP_L0C_ALLOC,
                                Opcode::OP_A_MUL_B,   Opcode::OP_UB_ALLOC,  Opcode::OP_L0C_COPY_UB,
                                Opcode::OP_L1_ALLOC,  Opcode::OP_L0C_TO_L1, Opcode::OP_L0C_COPY_OUT,
                                Opcode::OP_L0C_ALLOC, Opcode::OP_A_MUL_B};
    std::vector<std::vector<std::string>> ioperands{{},       {},       {}, {"L0A", "L0B"}, {}, {"L0C1"}, {},
                                                    {"L0C1"}, {"L0C1"}, {}, {"L0A", "L0B"}};
    std::vector<std::vector<std::string>> ooperands{{"L0A"},   {"L0B"},   {"L0C1"},   {"L0C1"}, {"UBDst"}, {"UBDst"},
                                                    {"L1Dst"}, {"L1Dst"}, {"DDROut"}, {"L0C2"}, {"L0C2"}};
    std::vector<std::string> opNames{"L0AAlloc", "L0BAlloc", "L0CAlloc1",  "Matmul1",   "UBAlloc", "L0CCopyUB",
                                     "L1Alloc",  "L0CToL1",  "L0CCopyOut", "L0CAlloc2", "Matmul2"};
    subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true);
    Function* function = subGraph.GetFunction();

    auto shapeImme = OpImmediate::Specified(std::vector<int64_t>{128, 128});
    auto* copyOutOp = subGraph.GetOp("L0CCopyOut");
    copyOutOp->SetOpAttribute(std::make_shared<CopyOpAttribute>(
        MemoryType::MEM_L0C, OpImmediate::Specified(std::vector<int64_t>{0, 0}), shapeImme, shapeImme));

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    EXPECT_EQ(sort.SortOps(), SUCCESS);
    OoOScheduler ooOScheduler(*function);
    EXPECT_EQ(ooOScheduler.Init(sort.operations), SUCCESS);

    ooOScheduler.state_.bufferManagerMap[CoreLocationType::AIC][MemoryType::MEM_L0C] = BufferPool(MemoryType::MEM_L0C,
                                                                                                  64 * 1024);
    ooOScheduler.state_.orderedOps = {subGraph.GetOp("L0AAlloc"),
                                      subGraph.GetOp("L0BAlloc"),
                                      subGraph.GetOp("L0CAlloc1"),
                                      subGraph.GetOp("Matmul1"),
                                      copyOutOp,
                                      subGraph.GetOp("L0CAlloc2"),
                                      subGraph.GetOp("Matmul2"),
                                      subGraph.GetOp("UBAlloc"),
                                      subGraph.GetOp("L0CCopyUB"),
                                      subGraph.GetOp("L1Alloc"),
                                      subGraph.GetOp("L0CToL1")};
    for (size_t i = 0; i < ooOScheduler.state_.orderedOps.size(); i++) {
        ooOScheduler.state_.schedInfoMap[ooOScheduler.state_.orderedOps[i]].execOrder = static_cast<int>(i);
    }

    EXPECT_EQ(ooOScheduler.SeqSchedule(), SUCCESS);

    auto spillCopyOut = std::count_if(ooOScheduler.state_.orderedOps.begin(), ooOScheduler.state_.orderedOps.end(),
                                      [](Operation* op) { return op->GetOpcodeStr() == "COPY_OUT"; });
    EXPECT_EQ(spillCopyOut, 2);
}

TEST_F(ScheduleOoOTest, TestScheduleSpillWithInplaceView)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11", "t12"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,
                                Opcode::OP_COPY_IN,  Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD,
                                Opcode::OP_VIEW};
    std::vector<std::vector<std::string>> ioperands{
        {}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t5", "t6"}, {"t7", "t8"}, {"t9", "t10"}, {"t11"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t6"}, {"t7"},  {"t8"}, {"t9"},  {"t5"}, {"t6"},
                                                    {"t7"}, {"t8"}, {"t10"}, {"t9"}, {"t11"}, {"t12"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3", "Alloc4", "Alloc5", "Copyin1", "Copyin2",
                                     "Copyin3", "Copyin4", "Add1",   "Add2",   "Add3",   "View1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t11"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t10");
    tensor1->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t11");
    tensor2->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor3 = subGraph.GetTensor("t12");
    tensor3->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;

    std::vector<int64_t> offset1 = {0, 0};
    auto viewAttr1 = std::make_shared<ViewOpAttribute>(offset1, MemoryType::MEM_UB);
    auto view1 = subGraph.GetOp("View1");
    view1->SetOpAttribute(viewAttr1);
    auto inputTensor = view1->GetIOperands()[0];
    auto outputTensor = view1->GetOOperands()[0];
    EXPECT_EQ(outputTensor->memoryrange.memId, inputTensor->memoryrange.memId);
    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    std::rotate(ooOScheduler.state_.orderedOps.begin(), ooOScheduler.state_.orderedOps.begin() + 6,
                ooOScheduler.state_.orderedOps.begin() + 11);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, SUCCESS);
    // The tensor's memId remains the same before and after the view operation following a spill.
    EXPECT_EQ(outputTensor->memoryrange.memId, inputTensor->memoryrange.memId);
}

TEST_F(ScheduleOoOTest, TestScheduleSpillFragFailed)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {"t1"}, {"t2"}, {"t4", "t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t3", "t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Copyin1", "Copyin2", "Add"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t3"), nullptr);
    std::shared_ptr<LogicalTensor> tensor = subGraph.GetTensor("t3");
    tensor->shape = {32, 32};
    tensor->tensor->rawshape = {32, 32};

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestEmptyOplist)
{
    Function function(Program::GetInstance(), "", "", nullptr);
    std::vector<Operation*> scheduleOpList;
    OoOScheduler ooOScheduler(function);
    Status res = ooOScheduler.Schedule(scheduleOpList);
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestScheduleReshape)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR};
    std::vector<Opcode> opCodes{Opcode::OP_RESHAPE};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"Reshape1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Schedule(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestSingleCopyin1)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"Copyin1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Schedule(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, FAILED);
}

TEST_F(ScheduleOoOTest, TestSingleCopyin2)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN};
    std::vector<std::vector<std::string>> ioperands{{}, {"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t2"}};
    std::vector<std::string> opNames{"Alloc1", "Copyin1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Schedule(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, FAILED);
}

TEST_F(ScheduleOoOTest, TestDelBufCount)
{
    Function function(Program::GetInstance(), "", "", nullptr);
    OoOScheduler oooSchedule(function);
    oooSchedule.state_.DelBufRefCount(-1);
}

TEST_F(ScheduleOoOTest, TestDelBufCount_1)
{
    Function function(Program::GetInstance(), "", "", nullptr);
    OoOScheduler oooSchedule(function);
    oooSchedule.state_.bufRefCount[1] = -1;
    oooSchedule.state_.DelBufRefCount(1);
}

TEST_F(ScheduleOoOTest, TestGetSpillTensor)
{
    Function function(Program::GetInstance(), "", "", nullptr);
    std::vector<Operation*> scheduleOpList;

    std::vector<int64_t> shape = {128, 128};
    std::shared_ptr<LogicalTensor> tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DataType::DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor3->SetMemoryTypeOriginal(MEM_UB);
    tensor3->SetMemoryTypeToBe(MEM_UB);
    tensor3->memoryrange.memId = 3;

    auto& alloc1 = PassOperationUtils::AddOperation(function, Opcode::OP_UB_ALLOC, {}, {tensor3});
    alloc1.UpdateLatency(1);

    OoOScheduler oooSchedule(function);
    LogicalTensorPtr tensor = oooSchedule.spillEngine_.GetSpillTensor(&alloc1, 1);
    EXPECT_EQ(tensor, nullptr);
}

TEST_F(ScheduleOoOTest, TestCheckAllocIssue)
{
    Function function(Program::GetInstance(), "", "", nullptr);
    std::vector<Operation*> scheduleOpList;

    std::vector<int64_t> shape = {128, 128};
    std::shared_ptr<LogicalTensor> tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DataType::DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor3->SetMemoryTypeOriginal(MEM_UB);
    tensor3->SetMemoryTypeToBe(MEM_UB);
    tensor3->memoryrange.memId = 3;

    std::shared_ptr<LogicalTensor> tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DataType::DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor3->SetMemoryTypeOriginal(MEM_UB);
    tensor3->SetMemoryTypeToBe(MEM_UB);
    tensor3->memoryrange.memId = 1;

    auto& alloc1 = PassOperationUtils::AddOperation(function, Opcode::OP_UB_ALLOC, {}, {tensor3, tensor2});
    alloc1.UpdateLatency(1);

    OoOScheduler oooSchedule(function);
    auto opList = function.Operations().DuplicatedOpList();
    oooSchedule.Init(opList);
}

TEST_F(ScheduleOoOTest, TestBufferUsage)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {"t1"}, {"t2"}, {"t4", "t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t3", "t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Copyin1", "Copyin2", "Add1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    OoOScheduleStatistic testCheck;
    ooOScheduler.AddObserver(&testCheck);
    res = ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, SUCCESS);
    std::unordered_map<MemoryType, uint64_t> invalidBufferTotalUsage = {{MemoryType::MEM_UB, 0},
                                                                        {MemoryType::MEM_L1, 0},
                                                                        {MemoryType::MEM_L0A, 0},
                                                                        {MemoryType::MEM_L0B, 0},
                                                                        {MemoryType::MEM_L0C, 0}};
    std::unordered_map<MemoryType, uint64_t> invalidBufferMaxUsage = {{MemoryType::MEM_UB, 0},
                                                                      {MemoryType::MEM_L1, 0},
                                                                      {MemoryType::MEM_L0A, 0},
                                                                      {MemoryType::MEM_L0B, 0},
                                                                      {MemoryType::MEM_L0C, 0}};
    EXPECT_NE(testCheck.bufferTotalUsage, invalidBufferTotalUsage);
    EXPECT_NE(testCheck.bufferMaxUsage, invalidBufferMaxUsage);

    testCheck.clock = 3; // 模拟数据
    res = testCheck.HealthCheckOoOSchedule();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_NE(testCheck.report, nullptr);
}

TEST_F(ScheduleOoOTest, TestScheduleGenSpillInfiniteLoop)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
                                           MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_SUB,
                                Opcode::OP_ADD,      Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{},     {},     {},     {},           {},
                                                    {"t1"}, {"t2"}, {"t3"}, {"t4", "t5"}, {"t3", "t6"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"},
                                                    {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3", "Alloc4", "Alloc5",
                                     "Copyin1", "Copyin2", "Sub1",   "Add1",   "Add2"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP16, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t3"), nullptr);
    std::shared_ptr<LogicalTensor> tensor = subGraph.GetTensor("t3");
    tensor->shape = {80, 128};
    tensor->tensor->rawshape = {80, 128};

    EXPECT_NE(subGraph.GetTensor("t4"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t4");
    tensor1->shape = {176, 256};
    tensor1->tensor->rawshape = {176, 256};

    EXPECT_NE(subGraph.GetTensor("t5"), nullptr);
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t5");
    tensor2->shape = {176, 256};
    tensor2->tensor->rawshape = {176, 256};

    EXPECT_NE(subGraph.GetTensor("t6"), nullptr);
    std::shared_ptr<LogicalTensor> tensor3 = subGraph.GetTensor("t6");
    tensor3->shape = {64, 128};
    tensor3->tensor->rawshape = {64, 128};

    EXPECT_NE(subGraph.GetTensor("t7"), nullptr);
    std::shared_ptr<LogicalTensor> tensor4 = subGraph.GetTensor("t7");
    tensor4->shape = {16, 16};
    tensor4->tensor->rawshape = {16, 16};

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SeqSchedule();
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestCheckOpBufferSize)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_ROWMAX_SINGLE,
                                Opcode::OP_ADD,      Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t3"}, {"t2"}, {"t4", "t5"}, {"t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t4"}, {"t5"},       {"t6"}, {"t8"},
                                                    {"t2"}, {"t4"}, {"t5", "t6"}, {"t8"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3",  "Alloc4", "Alloc5",
                                     "Copyin1", "Copyin2", "RowMax1", "Add1",   "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {144, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, FAILED);
}

TEST_F(ScheduleOoOTest, TestInitLocalBufferFailed)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_ROWMAX_SINGLE,
                                Opcode::OP_ADD,      Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t3"}, {"t2"}, {"t4", "t5"}, {"t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t4"}, {"t5"},       {"t6"}, {"t8"},
                                                    {"t2"}, {"t4"}, {"t5", "t6"}, {"t8"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3",  "Alloc4", "Alloc5",
                                     "Copyin1", "Copyin2", "RowMax1", "Add1",   "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {62, 69}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestCheckAllocBufferSize)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_ROWMAX_SINGLE,
                                Opcode::OP_ADD,      Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t3"}, {"t2"}, {"t4", "t5"}, {"t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t4"}, {"t5"},       {"t6"}, {"t8"},
                                                    {"t2"}, {"t4"}, {"t5", "t6"}, {"t8"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3",  "Alloc4", "Alloc5",
                                     "Copyin1", "Copyin2", "RowMax1", "Add1",   "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {256, 256}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, FAILED);
}

TEST_F(ScheduleOoOTest, TestOoORollbackMix)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"T3",  "T1",   "T6",   "T8",   "T11",  "T13",  "T16",  "T18",
                                         "T21", "DDR1", "DDR2", "DDR3", "DDR4", "DDR5", "DDR6", "DDR7"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_L1,         MemoryType::MEM_L1,         MemoryType::MEM_L1,         MemoryType::MEM_L1,
        MemoryType::MEM_L1,         MemoryType::MEM_L1,         MemoryType::MEM_L1,         MemoryType::MEM_L1,
        MemoryType::MEM_L1,         MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR};
    std::vector<std::string> tensorNames_L0AB{"T4", "T2", "T7", "T9", "T12", "T14", "T19", "T17"};
    std::vector<MemoryType> tensorMemTypes_L0AB{MemoryType::MEM_L0B, MemoryType::MEM_L0A, MemoryType::MEM_L0A,
                                                MemoryType::MEM_L0B, MemoryType::MEM_L0A, MemoryType::MEM_L0B,
                                                MemoryType::MEM_L0B, MemoryType::MEM_L0A};
    std::vector<std::string> tensorNames_L0C{"T5", "T10", "T15", "T20"};
    std::vector<MemoryType> tensorMemTypes_L0C{MemoryType::MEM_L0C, MemoryType::MEM_L0C, MemoryType::MEM_L0C,
                                               MemoryType::MEM_L0C};
    std::vector<Opcode> opCodes{
        Opcode::OP_L1_ALLOC,  Opcode::OP_L1_ALLOC,  Opcode::OP_L1_ALLOC,  Opcode::OP_L1_ALLOC,  Opcode::OP_L1_ALLOC,
        Opcode::OP_L1_ALLOC,  Opcode::OP_L1_ALLOC,  Opcode::OP_L1_ALLOC,  Opcode::OP_L0A_ALLOC, Opcode::OP_L1_ALLOC,
        Opcode::OP_L0A_ALLOC, Opcode::OP_L0A_ALLOC, Opcode::OP_L0A_ALLOC, Opcode::OP_L0B_ALLOC, Opcode::OP_L0B_ALLOC,
        Opcode::OP_L0B_ALLOC, Opcode::OP_L0B_ALLOC, Opcode::OP_L0C_ALLOC, Opcode::OP_L0C_ALLOC, Opcode::OP_L0C_ALLOC,
        Opcode::OP_COPY_IN,   Opcode::OP_COPY_IN,   Opcode::OP_COPY_IN,   Opcode::OP_COPY_IN,   Opcode::OP_COPY_IN,
        Opcode::OP_COPY_IN,   Opcode::OP_L1_TO_L0A, Opcode::OP_L1_TO_L0A, Opcode::OP_L1_TO_L0A, Opcode::OP_L1_TO_L0A,
        Opcode::OP_L1_TO_L0B, Opcode::OP_L1_TO_L0B, Opcode::OP_L1_TO_L0B, Opcode::OP_L1_TO_L0B, Opcode::OP_L0C_TO_L1,
        Opcode::OP_L0C_TO_L1, Opcode::OP_L0C_TO_L1, Opcode::OP_A_MUL_B,   Opcode::OP_A_MUL_B,   Opcode::OP_A_MUL_B,
        Opcode::OP_COPY_OUT,  Opcode::OP_A_MULACC_B};
    std::vector<std::vector<std::string>> inputoperands{{},           {},
                                                        {},           {},
                                                        {},           {},
                                                        {},           {},
                                                        {},           {},
                                                        {},           {},
                                                        {},           {},
                                                        {},           {},
                                                        {},           {},
                                                        {},           {},
                                                        {"DDR2"},     {"DDR1"},
                                                        {"DDR3"},     {"DDR4"},
                                                        {"DDR5"},     {"DDR6"},
                                                        {"T1"},       {"T6"},
                                                        {"T11"},      {"T16"},
                                                        {"T3"},       {"T8"},
                                                        {"T13"},      {"T18"},
                                                        {"T5"},       {"T15"},
                                                        {"T20"},      {"T2", "T4"},
                                                        {"T7", "T9"}, {"T12", "T14"},
                                                        {"T21"},      {"T10", "T17", "T19"}};
    std::vector<std::vector<std::string>> outputoperands{
        {"T1"},  {"T3"},  {"T6"},  {"T8"},  {"T11"}, {"T13"}, {"T16"}, {"T18"},  {"T2"},  {"T21"}, {"T7"},
        {"T12"}, {"T17"}, {"T4"},  {"T9"},  {"T14"}, {"T19"}, {"T5"},  {"T10"},  {"T15"}, {"T3"},  {"T1"},
        {"T8"},  {"T11"}, {"T13"}, {"T18"}, {"T2"},  {"T7"},  {"T12"}, {"T17"},  {"T4"},  {"T9"},  {"T14"},
        {"T19"}, {"T6"},  {"T16"}, {"T21"}, {"T5"},  {"T10"}, {"T15"}, {"DDR7"}, {"T20"}};
    std::vector<std::string> operationNames{
        "L1_Alloc1",      "L1_Alloc2",      "L1_Alloc3",      "L1_Alloc4",      "L1_Alloc5",      "L1_Alloc6",
        "L1_Alloc7",      "L1_Alloc8",      "L0A_Alloc1",     "L1_Alloc9",      "L0A_Alloc2",     "L0A_Alloc3",
        "L0A_Alloc4",     "L0B_Alloc1",     "L0B_Alloc2",     "L0B_Alloc3",     "L0B_Alloc4",     "L0C_Alloc1",
        "L0C_Alloc2",     "L0C_Alloc3",     "Copyin2",        "Copyin1",        "Copyin3",        "Copyin4",
        "Copyin5",        "Copyin6",        "OP_L1_TO_L0A_1", "OP_L1_TO_L0A_2", "OP_L1_TO_L0A_3", "OP_L1_TO_L0A_4",
        "OP_L1_TO_L0B_1", "OP_L1_TO_L0B_2", "OP_L1_TO_L0B_3", "OP_L1_TO_L0B_4", "OP_L0C_TO_L1_1", "OP_L0C_TO_L1_2",
        "OP_L0C_TO_L1_3", "OP_A_MUL_B_1",   "OP_A_MUL_B_2",   "OP_A_MUL_B_3",   "Copyout",        "OP_A_MULACC_B"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes_L0AB, tensorNames_L0AB, 0), true);
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 256}, tensorMemTypes_L0C, tensorNames_L0C, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, inputoperands, outputoperands, operationNames, true), true);
    Function* function = subGraph.GetFunction();
    std::shared_ptr<LogicalTensor> tensor = subGraph.GetTensor("T10");
    tensor->memoryrange.memId = subGraph.GetTensor("T20")->memoryrange.memId;
    EXPECT_NE(function, nullptr);

    OptimizeSort optimizeSort(function->Operations().DuplicatedOpList(), *function);
    Status res = optimizeSort.SortOps();
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestHasEnoughBuffer)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC};
    std::vector<std::vector<std::string>> ioperands{{}};
    std::vector<std::vector<std::string>> ooperands{{"t1", "t2"}};
    std::vector<std::string> opNames{"Alloc1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    auto op = subGraph.GetOp("Alloc1");
    auto tensor1 = subGraph.GetTensor("t1");
    int memId = tensor1->memoryrange.memId;

    OoOScheduler ooOScheduler(*function);
    ooOScheduler.state_.orderedOps.push_back(op);
    ooOScheduler.state_.schedInfoMap[op].isAlloc = true;
    ooOScheduler.state_.depManager.GetSuccessors(op).clear();
    ooOScheduler.state_.opReqMemIdsMap[op] = {memId};
    ooOScheduler.SetCoreLocation(op, CoreLocationType::AIV0);
    EXPECT_EQ(ooOScheduler.state_.InitLocalBuffer(tensor1, memId), SUCCESS);
    ooOScheduler.state_.bufferManagerMap[CoreLocationType::AIV0][MemoryType::MEM_UB] = BufferPool(MemoryType::MEM_UB,
                                                                                                  0);
    bool res = ooOScheduler.HasEnoughBuffer(op, MemoryType::MEM_UB);
    EXPECT_EQ(res, false);
}

TEST_F(ScheduleOoOTest, TestHasEnoughBufferAddMemId)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN};
    std::vector<std::vector<std::string>> ioperands{{}, {"t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t1"}};
    std::vector<std::string> opNames{"Alloc1", "COPY_IN"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    auto op = subGraph.GetOp("Alloc1");
    auto opCopyIn = subGraph.GetOp("COPY_IN");
    auto tensor1 = subGraph.GetTensor("t1");
    auto tensor2 = subGraph.GetTensor("t2");
    int memId1 = tensor1->memoryrange.memId;
    op->GetOutputOperand(0)->ClearAllProducers();
    op->GetOutputOperand(0)->AddProducer(*opCopyIn);
    OoOScheduler ooOScheduler(*function);
    ooOScheduler.state_.orderedOps.push_back(op);
    ooOScheduler.state_.orderedOps.push_back(opCopyIn);
    ooOScheduler.state_.schedInfoMap[op].isAlloc = true;
    ooOScheduler.state_.depManager.InsertSuccessor(op, opCopyIn);
    ooOScheduler.state_.opReqMemIdsMap[opCopyIn] = {1};
    ooOScheduler.state_.opReqMemIdsMap[op] = {memId1};
    ooOScheduler.SetCoreLocation(op, CoreLocationType::AIV0);
    EXPECT_EQ(ooOScheduler.state_.InitLocalBuffer(tensor1, memId1), SUCCESS);
    ooOScheduler.state_.bufferManagerMap[CoreLocationType::AIV0][MemoryType::MEM_UB] = BufferPool(MemoryType::MEM_UB,
                                                                                                  0);
    EXPECT_EQ(ooOScheduler.state_.InitLocalBuffer(tensor2, 1), SUCCESS);
    bool res = ooOScheduler.HasEnoughBuffer(op, MemoryType::MEM_UB);
    EXPECT_EQ(res, false);
}

TEST_F(ScheduleOoOTest, TestCoreAssign)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10"};
    std::vector<Opcode> opCodes{Opcode::OP_A_MUL_B, Opcode::OP_ADDS,      Opcode::OP_ADDS, Opcode::OP_ADDS,
                                Opcode::OP_ADDS,    Opcode::OP_A_MUL_B,   Opcode::OP_ADDS, Opcode::OP_A_MUL_B,
                                Opcode::OP_ADD,     Opcode::OP_A_MULACC_B};
    std::vector<std::vector<std::string>> ioperands{
        {"t0", "t0"}, {"t1"}, {"t1"},       {"t1"},       {"t2", "t2"},
        {"t2"},       {"t4"}, {"t5", "t5"}, {"t3", "t6"}, {"t7", "t8", "t9"}};
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t2"}, {"t8"}, {"t9"}, {"t3"},
                                                    {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"t10"}};
    std::vector<std::string> opNames{"op1", "op2", "op3", "op4", "op5", "op6", "op7", "op8", "op9", "op10"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {256, 256}, tensorNames), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    auto opList = function->Operations(false).DuplicatedOpList();
    TaskSplitter splitter;
    splitter.SplitGraph(opList);
    CoreScheduler coreScheduler;
    coreScheduler.Schedule(splitter.GetTaskGraph());
    const int taskNum = 10;
    EXPECT_EQ(splitter.GetTaskGraph().tasks.size(), taskNum);
    splitter.MergeTask();
    OoOScheduler ooOScheduler(*function);
    splitter.MarkInternalSubgraphID();
    EXPECT_EQ(splitter.GetMergedOperations().size(), opList.size());
}

TEST_F(ScheduleOoOTest, TestAtomicScopeMerge)
{
    constexpr int atomicScopeId = 1;
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t0", "t1", "t2", "t3", "t4"};
    std::vector<Opcode> opCodes{Opcode::OP_A_MUL_B, Opcode::OP_ADDS, Opcode::OP_ADDS, Opcode::OP_EXP};
    std::vector<std::vector<std::string>> ioperands{{"t0", "t0"}, {"t1"}, {"t2"}, {"t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"op1", "op2", "op3", "op4"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {256, 256}, tensorNames), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    auto op2 = subGraph.GetOp("op2");
    auto op3 = subGraph.GetOp("op3");
    auto op4 = subGraph.GetOp("op4");
    ASSERT_NE(op2, nullptr);
    ASSERT_NE(op3, nullptr);
    ASSERT_NE(op4, nullptr);
    op2->SetAtomicScopeId(atomicScopeId);
    op4->SetAtomicScopeId(atomicScopeId);
    int magic2 = op2->GetOpMagic();
    int magic3 = op3->GetOpMagic();
    int magic4 = op4->GetOpMagic();

    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    auto opList = function->Operations(false).DuplicatedOpList();
    EXPECT_EQ(opList.size(), 4U);

    TaskSplitter splitter;
    splitter.SplitGraph(opList);
    auto& tasks = splitter.GetTaskGraph().tasks;

    bool foundMerged23 = false;
    bool foundMerged24 = false;
    for (auto& task : tasks) {
        bool hasOp2 = false, hasOp3 = false, hasOp4 = false;
        for (auto* op : task.opList_) {
            if (op->GetOpMagic() == magic2)
                hasOp2 = true;
            if (op->GetOpMagic() == magic3)
                hasOp3 = true;
            if (op->GetOpMagic() == magic4)
                hasOp4 = true;
        }
        if (hasOp2 && hasOp3) {
            foundMerged23 = true;
        }
        if (hasOp2 && hasOp4) {
            foundMerged24 = true;
        }
    }
    EXPECT_TRUE(!foundMerged23) << "op2 and op3 should not be merged into one task by atomic_scope";
    EXPECT_TRUE(foundMerged24) << "op2 and op4 should be merged into one task by atomic_scope";
}

TEST_F(ScheduleOoOTest, TestBufferPollRearrange)
{
    BufferPool pool;
    pool.memSize_ = UBPoolSize;
    BufferSlice s1(32768, 65536);
    BufferSlice s2(98304, 98304);
    pool.bufferSlices[1] = s1;
    pool.bufferSlices[2] = s2;
    EXPECT_FALSE(pool.CheckBufferSlicesOverlap());

    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}};
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t2"}, {"t3"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    auto alloc1 = subGraph.GetOp("Alloc1");
    auto alloc2 = subGraph.GetOp("Alloc2");
    auto alloc3 = subGraph.GetOp("Alloc3");

    OoOScheduler oooSchedule(*function);
    auto corePair = CoreLocationType::AIV0;
    oooSchedule.state_.schedInfoMap[alloc3].coreLocation = corePair;
    oooSchedule.state_.bufferManagerMap[corePair][MemoryType::MEM_UB] = pool;
    oooSchedule.state_.tensorOccupyMap[1] = alloc1;
    oooSchedule.state_.tensorOccupyMap[2] = alloc2;

    oooSchedule.state_.localBufferMap[1] = std::make_shared<LocalBuffer>(1, 65536, 0, MemoryType::MEM_UB);
    oooSchedule.state_.localBufferMap[2] = std::make_shared<LocalBuffer>(2, 98304, 0, MemoryType::MEM_UB);
    EXPECT_EQ(oooSchedule.RearrangeBuffer(alloc3, MemoryType::MEM_UB), SUCCESS);
    auto& ubPool = oooSchedule.state_.bufferManagerMap[corePair][MemoryType::MEM_UB];
    EXPECT_EQ(ubPool.GetBufferSize(1), 65536);
    EXPECT_EQ(ubPool.GetBufferSize(2), 98304);
    EXPECT_EQ(ubPool.GetBufferOffset(1), 98304);
    EXPECT_EQ(ubPool.GetBufferOffset(2), 0);
}

TEST_F(ScheduleOoOTest, TestBufferPoolMakeBufferSliceAlreadyAlloc)
{
    BufferPool pool(MemoryType::MEM_UB, 1024);
    auto tensor = std::make_shared<LocalBuffer>(1, 64, 0, MemoryType::MEM_UB);
    BufferSlice slice1(0, 64);
    EXPECT_EQ(pool.MakeBufferSlice(tensor, slice1), SUCCESS);
    BufferSlice slice2(128, 64);
    EXPECT_EQ(pool.MakeBufferSlice(tensor, slice2), FAILED);
}

TEST_F(ScheduleOoOTest, TestBufferPoolAllocateNoFreeSpace)
{
    BufferPool pool(MemoryType::MEM_UB, 256);
    auto tensor1 = std::make_shared<LocalBuffer>(1, 256, 0, MemoryType::MEM_UB);
    EXPECT_EQ(pool.Allocate(tensor1), SUCCESS);
    auto tensor2 = std::make_shared<LocalBuffer>(2, 64, 0, MemoryType::MEM_UB);
    EXPECT_EQ(pool.Allocate(tensor2), FAILED);
}

TEST_F(ScheduleOoOTest, TestBufferRearrangeSingleBubble)
{
    BufferPool pool(MemoryType::MEM_UB, 100);
    auto tensor = std::make_shared<LocalBuffer>(1, 50, 0, MemoryType::MEM_UB);
    BufferSlice s1(0, 50);
    EXPECT_EQ(pool.MakeBufferSlice(tensor, s1), SUCCESS);
    RearrangeScheme scheme = GetRearrangeScheme(pool, 50);
    EXPECT_EQ(scheme.cost, static_cast<size_t>(INT_MAX));
}

TEST_F(ScheduleOoOTest, TestSchedulerAllocTensorMemRangeNonViewOp)
{
    ComputationalGraphBuilder subGraph;
    subGraph.AddTensor(DataType::DT_FP32, {8, 8}, "a");
    subGraph.AddTensor(DataType::DT_FP32, {8, 8}, "b");
    subGraph.AddTensor(DataType::DT_FP32, {8, 8}, "c");
    subGraph.AddOp(Opcode::OP_ADD, {"a", "b"}, {"c"}, "add1");
    Function* function = subGraph.GetFunction();
    OoOScheduler oooSchedule(*function);
    auto addOp = subGraph.GetOp("add1");
    oooSchedule.GetSkipOps(addOp).push_back(addOp);
    EXPECT_EQ(oooSchedule.AllocTensorMemRange(addOp), FAILED);
}

TEST_F(ScheduleOoOTest, TestSpillOnBlockFailedAtL0)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_L0A, MemoryType::MEM_L0A, MemoryType::MEM_L0B,
                                           MemoryType::MEM_L0B};
    std::vector<Opcode> opCodes{Opcode::OP_L1_TO_L0A, Opcode::OP_L0A_ALLOC, Opcode::OP_L1_TO_L0B, Opcode::OP_L0B_ALLOC};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}};
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"L1toL0A", "AllocL0A", "L1toL0B", "AllocL0B"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP16, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    auto L1toL0A = subGraph.GetOp("L1toL0A");
    auto L1toL0B = subGraph.GetOp("L1toL0B");
    auto AllocL0A = subGraph.GetOp("AllocL0A");
    auto AllocL0B = subGraph.GetOp("AllocL0B");
    OoOScheduler oooSchedule(*function);
    oooSchedule.state_.SetOpMemIds(AllocL0A, {3});
    oooSchedule.state_.SetOpMemIds(AllocL0B, {4});
    auto corePair = CoreLocationType::AIC;
    oooSchedule.state_.allocIssueQueue[corePair][MemoryType::MEM_L0A].Insert(AllocL0A);
    oooSchedule.state_.allocIssueQueue[corePair][MemoryType::MEM_L0B].Insert(AllocL0B);
    oooSchedule.state_.tensorOccupyMap.emplace(1, L1toL0A);
    oooSchedule.state_.tensorOccupyMap.emplace(2, L1toL0B);
    oooSchedule.state_.localBufferMap[1] = std::make_shared<LocalBuffer>(1, 32768, 0, MemoryType::MEM_L0A);
    oooSchedule.state_.localBufferMap[2] = std::make_shared<LocalBuffer>(2, 32768, 0, MemoryType::MEM_L0B);
    oooSchedule.state_.localBufferMap[3] = std::make_shared<LocalBuffer>(3, 32768, 0, MemoryType::MEM_L0A);
    oooSchedule.state_.localBufferMap[4] = std::make_shared<LocalBuffer>(4, 32768, 0, MemoryType::MEM_L0B);
    oooSchedule.state_.localBufferMap[1]->start = 512;
    oooSchedule.state_.localBufferMap[1]->end = 33280;
    oooSchedule.state_.localBufferMap[2]->start = 512;
    oooSchedule.state_.localBufferMap[2]->end = 33280;
    EXPECT_EQ(oooSchedule.SpillOnBlock(), FAILED);
}

TEST_F(ScheduleOoOTest, TestOoO1C2V)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1",   "t3",   "t6",   "t7",   "t8",  "t10",
                                         "DDR1", "DDR2", "DDR3", "DDR4", "t11", "t12"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_L1,         MemoryType::MEM_L1,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<std::string> tensorNames_L0{"t2", "t4", "t5"};
    std::vector<MemoryType> tensorMemTypes_L0AB{MemoryType::MEM_L0A, MemoryType::MEM_L0B, MemoryType::MEM_L0C};

    std::vector<Opcode> opCodes{
        Opcode::OP_L1_ALLOC, Opcode::OP_L1_ALLOC,  Opcode::OP_L0A_ALLOC,  Opcode::OP_L0B_ALLOC, Opcode::OP_L0C_ALLOC,
        Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,  Opcode::OP_UB_ALLOC,   Opcode::OP_UB_ALLOC,  Opcode::OP_COPY_IN,
        Opcode::OP_COPY_IN,  Opcode::OP_L1_TO_L0A, Opcode::OP_L1_TO_L0B,  Opcode::OP_A_MUL_B,   Opcode::OP_L0C_COPY_UB,
        Opcode::OP_ADDS,     Opcode::OP_COPY_OUT,  Opcode::OP_L1_COPY_UB, Opcode::OP_ADDS,      Opcode::OP_COPY_OUT,
        Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,  Opcode::OP_ADDS,       Opcode::OP_UB_COPY_L1};
    std::vector<std::vector<std::string>> ioperands{
        {},     {},           {},     {},     {},     {},     {},     {},      {}, {"DDR1"}, {"DDR2"}, {"t1"},
        {"t3"}, {"t2", "t4"}, {"t5"}, {"t6"}, {"t7"}, {"t3"}, {"t8"}, {"t10"}, {}, {},       {"t11"},  {"t12"}};
    std::vector<std::vector<std::string>> ooperands{
        {"t1"}, {"t3"}, {"t2"}, {"t4"}, {"t5"},   {"t6"}, {"t7"},  {"t8"},   {"t10"}, {"t11"}, {"t3"},  {"t2"},
        {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"DDR3"}, {"t8"}, {"t10"}, {"DDR4"}, {"t11"}, {"t12"}, {"t12"}, {"t1"}};
    std::vector<std::string> opNames{"L1_Alloc1", "L1_Alloc2", "L0A_Alloc1",  "L0B_Alloc1", "L0C_Alloc1", "UB_Alloc1",
                                     "UB_Alloc2", "UB_Alloc3", "UB_Alloc4",   "COPY_IN1",   "COPY_IN2",   "L1_TO_L0A",
                                     "L1_TO_L0B", "A_MUL_B",   "L0C_COPY_UB", "ADDS1",      "COPY_OUT1",  "L1_COPY_UB",
                                     "ADDS2",     "COPY_OUT2", "UB_Alloc5",   "UB_Alloc6",  "ADDS3",      "UB_COPY_L1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes_L0AB, tensorNames_L0, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    subGraph.GetOp("ADDS1")->SetAttribute(OpAttributeKey::isCube, false);
    Function* function = subGraph.GetFunction();
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    auto op1 = subGraph.GetOp("ADDS3");
    auto op2 = subGraph.GetOp("ADDS2");
    auto op3 = subGraph.GetOp("ADDS1");
    auto op4 = subGraph.GetOp("L1_TO_L0A");
    OoOSchedule oooSchedule;
    std::pair<uint64_t, Function*> functionPair = std::make_pair(0, function);
    int64_t size = 0;
    std::vector<Operation*> opList = function->Operations().DuplicatedOpList();
    auto res = oooSchedule.Schedule(opList, *function, functionPair, size);
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(op1->GetInternalSubgraphID(), 1);
    EXPECT_EQ(op2->GetInternalSubgraphID(), 2);
    EXPECT_EQ(op3->GetInternalSubgraphID(), 1);
    EXPECT_EQ(op4->GetInternalSubgraphID(), 0);
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

void SetInternalSubgraphIDAndAIVCore(Operation* op, int id)
{
    op->UpdateInternalSubgraphID(id);
    if (id == 0) {
        op->SetAIVCore(AIVCore::AIV0);
    }
}

void SetAttribute(ComputationalGraphBuilder& subGraph, OoOScheduler& oooSchedule, Operation*& ubCopyL1,
                  Operation*& alloc3)
{
    Operation* adds = subGraph.GetOp("ADDS");
    ubCopyL1 = subGraph.GetOp("UB_COPY_L1");
    Operation* copyin1 = subGraph.GetOp("COPY_IN1");
    Operation* copyin2 = subGraph.GetOp("COPY_IN2");
    Operation* copyin3 = subGraph.GetOp("COPY_IN3");
    Operation* copyin4 = subGraph.GetOp("COPY_IN4");
    Operation* copyout1 = subGraph.GetOp("COPY_OUT1");
    Operation* copyout2 = subGraph.GetOp("COPY_OUT2");

    Operation* alloc1 = subGraph.GetOp("UB_Alloc2");
    Operation* alloc2 = subGraph.GetOp("L1_Alloc1");
    alloc3 = subGraph.GetOp("L1_Alloc3");
    Operation* alloc4 = subGraph.GetOp("L1_Alloc2");

    Operation* alloc5 = subGraph.GetOp("UB_Alloc1");
    Operation* alloc6 = subGraph.GetOp("L0A_Alloc1");
    Operation* alloc7 = subGraph.GetOp("L0A_Alloc2");

    SetInternalSubgraphIDAndAIVCore(adds, 0);
    SetInternalSubgraphIDAndAIVCore(alloc5, 0);
    SetInternalSubgraphIDAndAIVCore(alloc1, 0);
    SetInternalSubgraphIDAndAIVCore(ubCopyL1, 0);

    SetInternalSubgraphIDAndAIVCore(alloc2, 1);
    SetInternalSubgraphIDAndAIVCore(alloc3, 1);
    SetInternalSubgraphIDAndAIVCore(alloc4, 1);
    SetInternalSubgraphIDAndAIVCore(alloc6, 1);
    SetInternalSubgraphIDAndAIVCore(alloc7, 1);
    SetInternalSubgraphIDAndAIVCore(copyin1, 1);
    SetInternalSubgraphIDAndAIVCore(copyin2, 1);
    SetInternalSubgraphIDAndAIVCore(copyin3, 1);
    SetInternalSubgraphIDAndAIVCore(copyin4, 1);
    SetInternalSubgraphIDAndAIVCore(copyout1, 1);
    SetInternalSubgraphIDAndAIVCore(copyout2, 1);

    oooSchedule.SetIsRetired(alloc5, true);
    oooSchedule.SetIsRetired(adds, true);
    oooSchedule.SetIsRetired(alloc1, true);
    oooSchedule.SetIsRetired(ubCopyL1, true);
    oooSchedule.SetIsRetired(alloc2, true);
    oooSchedule.SetIsRetired(alloc7, true);
    oooSchedule.SetIsRetired(copyin2, true);

    auto localBuffer1 = oooSchedule.state_.localBufferMap[0];
    auto coreAIC = CoreLocationType::AIC;
    oooSchedule.state_.bufferManagerMap[coreAIC][MemoryType::MEM_L1].Allocate(localBuffer1);
    oooSchedule.state_.tensorOccupyMap.emplace(0, copyin2);
}

TEST_F(ScheduleOoOTest, TensorMemTypeMismatch)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestMemTypeMismatch", "TestMemTypeMismatch",
                                           nullptr);
    std::vector<int64_t> shape = {16, 16};

    auto t = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    t->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    t->SetMemoryTypeToBe(MemoryType::MEM_UB); // 不一�?

    PassOperationUtils::AddOperation(*func, Opcode::OP_NOP, {t}, {t});
    func->inCasts_.push_back(t);

    OoOScheduleChecker checker;
    bool ok = checker.PreCheckTensorInfo(t);
    EXPECT_FALSE(ok);
}

TEST_F(ScheduleOoOTest, TensorMemIdInvalid)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestMemIdInvalid", "TestMemIdInvalid", nullptr);
    std::vector<int64_t> shape = {16, 16};

    auto t = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    t->SetMemoryTypeOriginal(MemoryType::MEM_UB);
    t->SetMemoryTypeToBe(MemoryType::MEM_UB);
    t->memoryrange.memId = -1; // 非法

    OoOScheduleChecker checker;
    bool ok = checker.PreCheckTensorInfo(t);
    EXPECT_FALSE(ok);
}

TEST_F(ScheduleOoOTest, CallOpNotAllowed)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames = {"t1", "t2"};
    std::vector<MemoryType> memTypes = {MEM_DEVICE_DDR, MEM_DEVICE_DDR};
    subGraph.AddTensors(DT_FP32, {16, 16}, memTypes, tensorNames, 0);

    std::vector<Opcode> opCodes = {Opcode::OP_CALL};
    std::vector<std::vector<std::string>> ins = {{"t1"}};
    std::vector<std::vector<std::string>> outs = {{"t2"}};
    std::vector<std::string> opNames = {"CALL_OP"};
    subGraph.AddOps(opCodes, ins, outs, opNames, true);

    Function* function = subGraph.GetFunction();
    OoOScheduleChecker checker;
    bool ret = checker.PreCheckOpInfo(function->Operations().DuplicatedOpList()[0]);
    EXPECT_FALSE(ret);
}

TEST_F(ScheduleOoOTest, ViewMemIdMismatch)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "ViewMemIdMismatch", "ViewMemIdMismatch", nullptr);
    std::vector<int64_t> shape = {16, 16};
    auto inTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    inTensor->SetMemoryTypeOriginal(MemoryType::MEM_UB);
    inTensor->SetMemoryTypeToBe(MemoryType::MEM_UB);
    inTensor->memoryrange.memId = 0;
    auto outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    outTensor->SetMemoryTypeOriginal(MemoryType::MEM_UB);
    outTensor->SetMemoryTypeToBe(MemoryType::MEM_UB);
    outTensor->memoryrange.memId = 1;
    PassOperationUtils::AddOperation(*func, Opcode::OP_VIEW, {inTensor}, {outTensor});
    OoOScheduleChecker checker;
    bool ret = checker.PreCheckOpInfo(func->Operations().DuplicatedOpList()[0]);
    EXPECT_FALSE(ret);
}

namespace dualdst_ut {

constexpr int64_t TILE_M = 64;
constexpr int64_t TILE_N = 64;
constexpr size_t SMALL_UB_POOL = 8 * 1024;

void SetCopyL0cToUbAttr(Operation& op, const std::vector<int64_t>& fromOff, const std::vector<int64_t>& tileShape)
{
    auto fromOffImme = OpImmediate::Specified(fromOff);
    auto shapeImme = OpImmediate::Specified(tileShape);
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(fromOffImme, MemoryType::MEM_UB, shapeImme, shapeImme));
}

void InjectStaticValidShape(Operation& op, const std::vector<int64_t>& vals)
{
    op.SetAttribute(OpAttributeKey::staticValidShape, vals);
}

struct DualDstGraph {
    std::shared_ptr<ComputationalGraphBuilder> builder;
    Function* func{nullptr};
    Operation* allocL0c{nullptr};
    Operation* allocUb0{nullptr};
    Operation* allocUb1{nullptr};
    Operation* allocOut0{nullptr};
    Operation* allocOut1{nullptr};
    Operation* copyinL0c{nullptr};
    Operation* copy0{nullptr};
    Operation* copy1{nullptr};
    Operation* add0{nullptr};
    Operation* add1{nullptr};
};

DualDstGraph BuildDualDstGraph(const std::vector<int64_t>& l0cShape, const std::vector<int64_t>& tileShape,
                               const std::vector<int64_t>& fromOff0, const std::vector<int64_t>& fromOff1)
{
    DualDstGraph g;
    g.builder = std::make_shared<ComputationalGraphBuilder>();
    EXPECT_EQ(g.builder->AddTensor(DataType::DT_FP32, l0cShape, MemoryType::MEM_L0C, "t_l0c"), true);
    EXPECT_EQ(g.builder->AddTensor(DataType::DT_FP32, tileShape, MemoryType::MEM_UB, "t_ub0"), true);
    EXPECT_EQ(g.builder->AddTensor(DataType::DT_FP32, tileShape, MemoryType::MEM_UB, "t_ub1"), true);
    EXPECT_EQ(g.builder->AddTensor(DataType::DT_FP32, tileShape, MemoryType::MEM_UB, "t_out0"), true);
    EXPECT_EQ(g.builder->AddTensor(DataType::DT_FP32, tileShape, MemoryType::MEM_UB, "t_out1"), true);

    EXPECT_EQ(g.builder->AddOp(Opcode::OP_L0C_ALLOC, {}, {"t_l0c"}, "alloc_l0c"), true);
    EXPECT_EQ(g.builder->AddOp(Opcode::OP_UB_ALLOC, {}, {"t_ub0"}, "alloc_ub0"), true);
    EXPECT_EQ(g.builder->AddOp(Opcode::OP_UB_ALLOC, {}, {"t_ub1"}, "alloc_ub1"), true);
    EXPECT_EQ(g.builder->AddOp(Opcode::OP_UB_ALLOC, {}, {"t_out0"}, "alloc_out0"), true);
    EXPECT_EQ(g.builder->AddOp(Opcode::OP_UB_ALLOC, {}, {"t_out1"}, "alloc_out1"), true);
    EXPECT_EQ(g.builder->AddOp(Opcode::OP_L0C_COPY_UB, {"t_l0c"}, {"t_ub0"}, "copy0"), true);
    EXPECT_EQ(g.builder->AddOp(Opcode::OP_L0C_COPY_UB, {"t_l0c"}, {"t_ub1"}, "copy1"), true);
    EXPECT_EQ(g.builder->AddOp(Opcode::OP_ADD, {"t_ub0", "t_ub0"}, {"t_out0"}, "add0"), true);
    EXPECT_EQ(g.builder->AddOp(Opcode::OP_ADD, {"t_ub1", "t_ub1"}, {"t_out1"}, "add1"), true);

    g.func = g.builder->GetFunction();
    g.allocL0c = g.builder->GetOp("alloc_l0c");
    g.allocUb0 = g.builder->GetOp("alloc_ub0");
    g.allocUb1 = g.builder->GetOp("alloc_ub1");
    g.allocOut0 = g.builder->GetOp("alloc_out0");
    g.allocOut1 = g.builder->GetOp("alloc_out1");
    g.copy0 = g.builder->GetOp("copy0");
    g.copy1 = g.builder->GetOp("copy1");
    g.add0 = g.builder->GetOp("add0");
    g.add1 = g.builder->GetOp("add1");

    SetCopyL0cToUbAttr(*g.copy0, fromOff0, tileShape);
    SetCopyL0cToUbAttr(*g.copy1, fromOff1, tileShape);
    return g;
}

DualDstGraph BuildDualDstGraph_2(const std::vector<int64_t>& l0cShape, const std::vector<int64_t>& tileShape,
                                 const std::vector<int64_t>& fromOff0, const std::vector<int64_t>& fromOff1)
{
    DualDstGraph g;
    g.builder = std::make_shared<ComputationalGraphBuilder>();
    EXPECT_EQ(g.builder->AddTensor(DataType::DT_FP32, l0cShape, MemoryType::MEM_DEVICE_DDR, "t_ddr"), true);
    EXPECT_EQ(g.builder->AddTensor(DataType::DT_FP32, l0cShape, MemoryType::MEM_L0C, "t_l0c"), true);
    EXPECT_EQ(g.builder->AddTensor(DataType::DT_FP32, tileShape, MemoryType::MEM_UB, "t_ub0"), true);
    EXPECT_EQ(g.builder->AddTensor(DataType::DT_FP32, tileShape, MemoryType::MEM_UB, "t_ub1"), true);
    for (int side = 0; side < 2; ++side) {
        for (int idx = 0; idx < 5; ++idx) {
            EXPECT_EQ(g.builder->AddTensor(DataType::DT_FP32, tileShape, MemoryType::MEM_UB,
                                           "t_softmax" + std::to_string(side) + "_out" + std::to_string(idx)),
                      true);
        }
    }

    EXPECT_EQ(g.builder->AddOp(Opcode::OP_L0C_ALLOC, {}, {"t_l0c"}, "alloc_l0c"), true);
    EXPECT_EQ(g.builder->AddOp(Opcode::OP_UB_ALLOC, {}, {"t_ub0"}, "alloc_ub0"), true);
    EXPECT_EQ(g.builder->AddOp(Opcode::OP_UB_ALLOC, {}, {"t_ub1"}, "alloc_ub1"), true);
    for (int side = 0; side < 2; ++side) {
        for (int idx = 0; idx < 5; ++idx) {
            EXPECT_EQ(g.builder->AddOp(Opcode::OP_UB_ALLOC, {},
                                       {"t_softmax" + std::to_string(side) + "_out" + std::to_string(idx)},
                                       "alloc_softmax" + std::to_string(side) + "_out" + std::to_string(idx)),
                      true);
        }
    }
    EXPECT_EQ(g.builder->AddOp(Opcode::OP_COPY_IN, {"t_ddr"}, {"t_l0c"}, "copyin_l0c"), true);
    EXPECT_EQ(g.builder->AddOp(Opcode::OP_L0C_COPY_UB, {"t_l0c"}, {"t_ub0"}, "copy0"), true);
    EXPECT_EQ(g.builder->AddOp(Opcode::OP_L0C_COPY_UB, {"t_l0c"}, {"t_ub1"}, "copy1"), true);
    EXPECT_EQ(g.builder->AddOp(
                  Opcode::OP_ONLINE_SOFTMAX, {"t_ub0"},
                  {"t_softmax0_out0", "t_softmax0_out1", "t_softmax0_out2", "t_softmax0_out3", "t_softmax0_out4"},
                  "softmax0"),
              true);
    EXPECT_EQ(g.builder->AddOp(
                  Opcode::OP_ONLINE_SOFTMAX, {"t_ub1"},
                  {"t_softmax1_out0", "t_softmax1_out1", "t_softmax1_out2", "t_softmax1_out3", "t_softmax1_out4"},
                  "softmax1"),
              true);

    g.func = g.builder->GetFunction();
    g.allocL0c = g.builder->GetOp("alloc_l0c");
    g.allocUb0 = g.builder->GetOp("alloc_ub0");
    g.allocUb1 = g.builder->GetOp("alloc_ub1");
    g.allocOut0 = g.builder->GetOp("alloc_softmax0_out0");
    g.allocOut1 = g.builder->GetOp("alloc_softmax1_out0");
    g.copyinL0c = g.builder->GetOp("copyin_l0c");
    g.copy0 = g.builder->GetOp("copy0");
    g.copy1 = g.builder->GetOp("copy1");
    g.add0 = g.builder->GetOp("softmax0");
    g.add1 = g.builder->GetOp("softmax1");

    SetCopyL0cToUbAttr(*g.copy0, fromOff0, tileShape);
    SetCopyL0cToUbAttr(*g.copy1, fromOff1, tileShape);
    auto copyinOffImme = OpImmediate::Specified(std::vector<int64_t>{0, 0});
    auto copyinShapeImme = OpImmediate::Specified(l0cShape);
    g.copyinL0c->SetOpAttribute(
        std::make_shared<CopyOpAttribute>(copyinOffImme, MemoryType::MEM_L0C, copyinShapeImme, copyinShapeImme));
    return g;
}

void InjectCoreMap(OoOScheduler& s, const DualDstGraph& g, bool sameCoreForAdds = false)
{
    s.state_.schedInfoMap[g.copy0].coreLocation = CoreLocationType::AIC;
    s.state_.schedInfoMap[g.copy1].coreLocation = CoreLocationType::AIC;
    s.state_.schedInfoMap[g.add0].coreLocation = CoreLocationType::AIV0;
    s.state_.schedInfoMap[g.add1].coreLocation = sameCoreForAdds ? CoreLocationType::AIV0 : CoreLocationType::AIV1;
    s.state_.schedInfoMap[g.allocL0c].coreLocation = CoreLocationType::AIC;
    if (g.copyinL0c != nullptr) {
        s.state_.schedInfoMap[g.copyinL0c].coreLocation = CoreLocationType::AIC;
    }
    s.state_.schedInfoMap[g.allocUb0].coreLocation = CoreLocationType::AIV0;
    s.state_.schedInfoMap[g.allocUb1].coreLocation = CoreLocationType::AIV1;
    s.state_.schedInfoMap[g.allocOut0].coreLocation = CoreLocationType::AIV0;
    s.state_.schedInfoMap[g.allocOut1].coreLocation = CoreLocationType::AIV1;
}

void UpdateCopyDynValidShape(DualDstGraph& g)
{
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape({SymbolicScalar(TILE_M), SymbolicScalar(TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape({SymbolicScalar(TILE_M), SymbolicScalar(TILE_N)});
}

DualDstGraph BuildOnlineSoftmaxDualDstGraph()
{
    auto g = BuildDualDstGraph_2({TILE_M, TILE_N * 2}, {TILE_M, TILE_N}, {0, 0}, {0, TILE_N});
    UpdateCopyDynValidShape(g);
    return g;
}

bool TopoSortInPlace(DependencyManager& depManager, std::vector<Operation*>& ops)
{
    std::vector<Operation*> sorted;
    sorted.reserve(ops.size());
    std::unordered_set<Operation*> placed;
    while (sorted.size() < ops.size()) {
        bool progressed = false;
        for (auto* op : ops) {
            if (placed.count(op) != 0) {
                continue;
            }
            bool ready = true;
            for (auto* pred : depManager.GetPredecessors(op)) {
                if (pred != nullptr && placed.count(pred) == 0 &&
                    std::find(ops.begin(), ops.end(), pred) != ops.end()) {
                    ready = false;
                    break;
                }
            }
            if (!ready) {
                continue;
            }
            sorted.push_back(op);
            placed.insert(op);
            progressed = true;
        }
        if (!progressed) {
            return false;
        }
    }
    ops = sorted;
    return true;
}

Status InitDualDstScheduler(OoOScheduler& s, const DualDstGraph& g)
{
    Status st = s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO);
    if (st != SUCCESS)
        return st;
    InjectCoreMap(s, g);
    s.SetEnableDualDst(true);
    return SUCCESS;
}

bool HasAiv0AndAiv1Tasks(TaskSplitter& splitter)
{
    bool hasAiv0Task = false;
    bool hasAiv1Task = false;
    for (const auto& task : splitter.GetTaskGraph().tasks) {
        hasAiv0Task = hasAiv0Task || task.targetCoreType == TargetCoreType::AIV0;
        hasAiv1Task = hasAiv1Task || task.targetCoreType == TargetCoreType::AIV1;
    }
    return hasAiv0Task && hasAiv1Task;
}

bool HasIsoAllocPair(const std::unordered_map<Operation*, Operation*>& pairs, Operation* lhs, Operation* rhs)
{
    auto lhsIt = pairs.find(lhs);
    if (lhsIt != pairs.end() && lhsIt->second == rhs)
        return true;
    auto rhsIt = pairs.find(rhs);
    return rhsIt != pairs.end() && rhsIt->second == lhs;
}

int FindTaskIndex(const TaskGraph& graph, Operation* target)
{
    for (const auto& task : graph.tasks) {
        if (std::find(task.opList_.begin(), task.opList_.end(), target) != task.opList_.end())
            return task.idx;
    }
    return -1;
}

Operation* AddAivUbAlloc(Function& function, OoOScheduler& scheduler, CoreLocationType core, uint64_t allocSize)
{
    std::vector<int64_t> shape{8, 8};
    auto tensor = IRBuilder().CreateTensorVar(DataType::DT_FP32, shape, CreateTestConstIntVector(shape));
    int memId = tensor->GetMagic();
    SetTensorAttr(tensor, MemoryType::MEM_UB, memId);
    auto* alloc = &PassOperationUtils::AddOperation(function, Opcode::OP_UB_ALLOC, {}, LogicalTensors({tensor}));
    auto& info = scheduler.state_.schedInfoMap[alloc];
    info.isAlloc = true;
    info.isRetired = false;
    info.coreLocation = core;
    scheduler.state_.SetOpMemIds(alloc, {memId});
    scheduler.state_.localBufferMap[memId] = std::make_shared<LocalBuffer>(memId, allocSize, 0, MemoryType::MEM_UB);
    scheduler.state_.bufRefCount[memId] = 2;
    return alloc;
}

class DualDstSpillObserver : public ScheduleObserver {
public:
    void OnSpill(const SpillEvent& event) override { events.push_back(event); }

    std::vector<SpillEvent> events;
};

Operation* FindDualDstOp(Function& func)
{
    for (auto& op : func.Operations()) {
        if (op.GetOpcode() == Opcode::OP_L0C_COPY_UB_DUAL_DST)
            return &op;
    }
    return nullptr;
}

struct FuseSnapshot {
    size_t functionOpCount;
    size_t orderedOpCount;
    size_t firstCopyIndex;
    LogicalTensorPtr l0cTensor;
    LogicalTensorPtr ubTensor0;
    LogicalTensorPtr ubTensor1;
    int l0cMemId;
    int ubMemId0;
    int ubMemId1;
    int l0cRefCount;
    int ubRefCount0;
    int ubRefCount1;
};

FuseSnapshot CaptureFuseSnapshot(OoOScheduler& scheduler, const DualDstGraph& graph)
{
    auto l0c = graph.copy0->GetInputOperand(0);
    auto ub0 = graph.copy0->GetOutputOperand(0);
    auto ub1 = graph.copy1->GetOutputOperand(0);
    auto copy0 = std::find(scheduler.state_.orderedOps.begin(), scheduler.state_.orderedOps.end(), graph.copy0);
    auto copy1 = std::find(scheduler.state_.orderedOps.begin(), scheduler.state_.orderedOps.end(), graph.copy1);
    size_t firstCopy = static_cast<size_t>(std::min(copy0, copy1) - scheduler.state_.orderedOps.begin());
    return {graph.func->Operations().size(),
            scheduler.state_.orderedOps.size(),
            firstCopy,
            l0c,
            ub0,
            ub1,
            l0c->memoryrange.memId,
            ub0->memoryrange.memId,
            ub1->memoryrange.memId,
            scheduler.state_.bufRefCount.at(l0c->memoryrange.memId),
            scheduler.state_.bufRefCount.at(ub0->memoryrange.memId),
            scheduler.state_.bufRefCount.at(ub1->memoryrange.memId)};
}

void ExpectFusedGraph(OoOScheduler& scheduler, const DualDstGraph& graph, const FuseSnapshot& before)
{
    Operation* dual = FindDualDstOp(*graph.func);
    ASSERT_NE(dual, nullptr);
    EXPECT_EQ(before.functionOpCount, graph.func->Operations().size() + 1);
    EXPECT_EQ(before.orderedOpCount, scheduler.state_.orderedOps.size() + 1);
    ASSERT_LT(before.firstCopyIndex, scheduler.state_.orderedOps.size());
    EXPECT_EQ(scheduler.state_.orderedOps[before.firstCopyIndex], dual);
    EXPECT_EQ(std::count(scheduler.state_.orderedOps.begin(), scheduler.state_.orderedOps.end(), graph.copy0), 0);
    EXPECT_EQ(std::count(scheduler.state_.orderedOps.begin(), scheduler.state_.orderedOps.end(), graph.copy1), 0);
    EXPECT_EQ(dual->GetInputOperand(0), before.l0cTensor);
    ASSERT_EQ(dual->GetOOperands().size(), 2u);
    EXPECT_EQ(dual->GetOutputOperand(0), before.ubTensor0);
    EXPECT_EQ(dual->GetOutputOperand(1), before.ubTensor1);
    auto functionOps = graph.func->Operations().DuplicatedOpList();
    EXPECT_EQ(std::count(functionOps.begin(), functionOps.end(), graph.copy0), 0);
    EXPECT_EQ(std::count(functionOps.begin(), functionOps.end(), graph.copy1), 0);
    const auto& l0cConsumers = before.l0cTensor->GetConsumers();
    ASSERT_EQ(l0cConsumers.size(), 1u);
    EXPECT_EQ(*l0cConsumers.begin(), dual);
    const auto& ubProducers0 = before.ubTensor0->GetProducers();
    ASSERT_EQ(ubProducers0.size(), 2u);
    EXPECT_NE(std::find(ubProducers0.begin(), ubProducers0.end(), graph.allocUb0), ubProducers0.end());
    EXPECT_NE(std::find(ubProducers0.begin(), ubProducers0.end(), dual), ubProducers0.end());
    const auto& ubProducers1 = before.ubTensor1->GetProducers();
    ASSERT_EQ(ubProducers1.size(), 2u);
    EXPECT_NE(std::find(ubProducers1.begin(), ubProducers1.end(), graph.allocUb1), ubProducers1.end());
    EXPECT_NE(std::find(ubProducers1.begin(), ubProducers1.end(), dual), ubProducers1.end());
    auto preds = scheduler.state_.depManager.GetPredecessors(dual);
    auto succs = scheduler.state_.depManager.GetSuccessors(dual);
    EXPECT_NE(std::find(preds.begin(), preds.end(), graph.allocUb0), preds.end());
    EXPECT_NE(std::find(preds.begin(), preds.end(), graph.allocUb1), preds.end());
    EXPECT_NE(std::find(succs.begin(), succs.end(), graph.add0), succs.end());
    EXPECT_NE(std::find(succs.begin(), succs.end(), graph.add1), succs.end());
}

void ExpectFusedMetadata(OoOScheduler& scheduler, const DualDstGraph& graph, const FuseSnapshot& before)
{
    Operation* dual = FindDualDstOp(*graph.func);
    ASSERT_NE(dual, nullptr);
    EXPECT_TRUE(scheduler.state_.IsDualDstAlloc(graph.allocUb0));
    EXPECT_TRUE(scheduler.state_.IsDualDstAlloc(graph.allocUb1));
    EXPECT_EQ(scheduler.state_.schedInfoMap[graph.allocUb0].pairedDualDstAlloc, graph.allocUb1);
    EXPECT_EQ(scheduler.state_.schedInfoMap[graph.allocUb1].pairedDualDstAlloc, graph.allocUb0);
    EXPECT_EQ(scheduler.state_.schedInfoMap.count(graph.copy0), 0u);
    EXPECT_EQ(scheduler.state_.schedInfoMap.count(graph.copy1), 0u);
    EXPECT_EQ(scheduler.state_.opReqMemIdsMap.count(graph.copy0), 0u);
    EXPECT_EQ(scheduler.state_.opReqMemIdsMap.count(graph.copy1), 0u);
    EXPECT_EQ(scheduler.state_.opReqMemIdsMap.count(dual), 1u);
    EXPECT_EQ(before.l0cTensor->memoryrange.memId, before.l0cMemId);
    EXPECT_EQ(before.ubTensor0->memoryrange.memId, before.ubMemId0);
    EXPECT_EQ(before.ubTensor1->memoryrange.memId, before.ubMemId1);
    ASSERT_EQ(scheduler.state_.bufRefCount.count(before.l0cMemId), 1u);
    ASSERT_EQ(scheduler.state_.bufRefCount.count(before.ubMemId0), 1u);
    ASSERT_EQ(scheduler.state_.bufRefCount.count(before.ubMemId1), 1u);
    EXPECT_EQ(scheduler.state_.bufRefCount.at(before.l0cMemId), before.l0cRefCount - 1);
    EXPECT_EQ(scheduler.state_.bufRefCount.at(before.ubMemId0), before.ubRefCount0);
    EXPECT_EQ(scheduler.state_.bufRefCount.at(before.ubMemId1), before.ubRefCount1);
}

bool HasDualDstOp(const std::vector<Operation*>& ops)
{
    return std::any_of(ops.begin(), ops.end(), [](Operation* op) {
        return op != nullptr && op->GetOpcode() == Opcode::OP_L0C_COPY_UB_DUAL_DST;
    });
}

Operation* FindUbAllocPred(OoOScheduler& s, Operation* op)
{
    for (auto* pred : s.state_.depManager.GetPredecessors(op)) {
        if (pred != nullptr && pred->GetOpcodeStr().find("UB_ALLOC") != std::string::npos)
            return pred;
    }
    return nullptr;
}

Status FillAivPoolsWithPlaceholderBuffers(OoOScheduler& s, const DualDstGraph& g, size_t needSize, int memIdA,
                                          int memIdB)
{
    auto& poolA = s.state_.bufferManagerMap[CoreLocationType::AIV0][MemoryType::MEM_UB];
    auto& poolB = s.state_.bufferManagerMap[CoreLocationType::AIV1][MemoryType::MEM_UB];
    if (poolA.GetMemSize() < needSize || poolB.GetMemSize() < needSize)
        return FAILED;
    auto bufHolderA = std::make_shared<LocalBuffer>(memIdA, needSize, 0, MemoryType::MEM_UB);
    auto bufHolderB = std::make_shared<LocalBuffer>(memIdB, needSize, 0, MemoryType::MEM_UB);
    if (poolA.AllocateAtOffset(bufHolderA, 0) != SUCCESS || poolB.AllocateAtOffset(bufHolderB, 0) != SUCCESS)
        return FAILED;
    s.state_.tensorOccupyMap[memIdA] = g.add0;
    s.state_.tensorOccupyMap[memIdB] = g.add1;
    return SUCCESS;
}

struct SpillTestGraph {
    std::shared_ptr<ComputationalGraphBuilder> builder;
    Function* function{nullptr};
    Operation* liveAlloc0{nullptr};
    Operation* liveAlloc1{nullptr};
    Operation* needAlloc0{nullptr};
    Operation* needAlloc1{nullptr};
    Operation* copyIn0{nullptr};
    Operation* copyIn1{nullptr};
    Operation* add0{nullptr};
    Operation* add1{nullptr};
};

SpillTestGraph BuildSpillTestGraph()
{
    SpillTestGraph graph;
    graph.builder = std::make_shared<ComputationalGraphBuilder>();
    for (const auto& name : {"ddr0", "ddr1"})
        EXPECT_TRUE(graph.builder->AddTensor(DataType::DT_FP32, {TILE_M, TILE_N}, MemoryType::MEM_DEVICE_DDR, name));
    for (const auto& name : {"live0", "live1", "need0", "need1", "out0", "out1"}) {
        EXPECT_TRUE(graph.builder->AddTensor(DataType::DT_FP32, {TILE_M, TILE_N}, MemoryType::MEM_UB, name));
        EXPECT_TRUE(graph.builder->AddOp(Opcode::OP_UB_ALLOC, {}, {name}, "alloc_" + std::string(name)));
    }
    graph.function = graph.builder->GetFunction();
    graph.copyIn0 = &CreateCopyOp(*graph.function, Opcode::OP_COPY_IN, graph.builder->GetTensor("ddr0"),
                                  graph.builder->GetTensor("live0"), {TILE_M, TILE_N});
    graph.copyIn1 = &CreateCopyOp(*graph.function, Opcode::OP_COPY_IN, graph.builder->GetTensor("ddr1"),
                                  graph.builder->GetTensor("live1"), {TILE_M, TILE_N});
    graph.add0 = &CreateAddOp(*graph.function, graph.builder->GetTensor("live0"), graph.builder->GetTensor("live0"),
                              graph.builder->GetTensor("out0"));
    graph.add1 = &CreateAddOp(*graph.function, graph.builder->GetTensor("live1"), graph.builder->GetTensor("live1"),
                              graph.builder->GetTensor("out1"));
    graph.liveAlloc0 = graph.builder->GetOp("alloc_live0");
    graph.liveAlloc1 = graph.builder->GetOp("alloc_live1");
    graph.needAlloc0 = graph.builder->GetOp("alloc_need0");
    graph.needAlloc1 = graph.builder->GetOp("alloc_need1");
    return graph;
}

Status ConfigureSpillTestState(OoOScheduler& scheduler, const SpillTestGraph& graph)
{
    auto setCore = [&scheduler](Operation* op, CoreLocationType core) {
        scheduler.state_.schedInfoMap[op].coreLocation = core;
    };
    for (auto* op : {graph.liveAlloc0, graph.needAlloc0, graph.copyIn0, graph.add0, graph.builder->GetOp("alloc_out0")})
        setCore(op, CoreLocationType::AIV0);
    for (auto* op : {graph.liveAlloc1, graph.needAlloc1, graph.copyIn1, graph.add1, graph.builder->GetOp("alloc_out1")})
        setCore(op, CoreLocationType::AIV1);
    scheduler.state_.schedInfoMap[graph.needAlloc0].isDualDstAlloc = true;
    scheduler.state_.schedInfoMap[graph.needAlloc0].pairedDualDstAlloc = graph.needAlloc1;
    scheduler.state_.schedInfoMap[graph.needAlloc1].isDualDstAlloc = true;
    scheduler.state_.schedInfoMap[graph.needAlloc1].pairedDualDstAlloc = graph.needAlloc0;
    scheduler.state_.allocIssueQueue[CoreLocationType::AIV0][MemoryType::MEM_UB].Insert(graph.needAlloc0);
    scheduler.state_.allocIssueQueue[CoreLocationType::AIV1][MemoryType::MEM_UB].Insert(graph.needAlloc1);
    int memId0 = graph.builder->GetTensor("live0")->memoryrange.memId;
    int memId1 = graph.builder->GetTensor("live1")->memoryrange.memId;
    auto& pool0 = scheduler.state_.bufferManagerMap[CoreLocationType::AIV0][MemoryType::MEM_UB];
    auto& pool1 = scheduler.state_.bufferManagerMap[CoreLocationType::AIV1][MemoryType::MEM_UB];
    if (pool0.Allocate(scheduler.state_.localBufferMap[memId0]) != SUCCESS ||
        pool1.Allocate(scheduler.state_.localBufferMap[memId1]) != SUCCESS)
        return FAILED;
    scheduler.state_.tensorOccupyMap[memId0] = graph.copyIn0;
    scheduler.state_.tensorOccupyMap[memId1] = graph.copyIn1;
    scheduler.state_.newOperations = {graph.liveAlloc0, graph.copyIn0, graph.liveAlloc1, graph.copyIn1};
    return SUCCESS;
}

void ExpectTwoAivSpills(const DualDstSpillObserver& observer, int memId0, int memId1)
{
    ASSERT_EQ(observer.events.size(), 2u);
    std::set<int> cores;
    for (const auto& event : observer.events) {
        EXPECT_EQ(event.coreLocation.coreType, CoreClass::AIV);
        cores.insert(event.coreLocation.coreIdx);
        if (event.coreLocation.coreIdx == 0) {
            EXPECT_EQ(event.spillMemId, memId0);
        } else if (event.coreLocation.coreIdx == 1) {
            EXPECT_EQ(event.spillMemId, memId1);
        } else {
            ADD_FAILURE() << "Unexpected AIV core index: " << event.coreLocation.coreIdx;
        }
    }
    EXPECT_EQ(cores, std::set<int>({0, 1}));
}

struct MainLoopReuseGraph {
    DualDstGraph dualDst;
    LogicalTensorPtr dualTensor0;
    LogicalTensorPtr dualTensor1;
    Operation* releaseAlloc0{nullptr};
    Operation* releaseAlloc1{nullptr};
    Operation* releaseCopyin0{nullptr};
    Operation* releaseCopyin1{nullptr};
    Operation* releaseCopy0{nullptr};
    Operation* releaseCopy1{nullptr};
};

bool BuildMainLoopReuseGraph(MainLoopReuseGraph& graph)
{
    graph.dualDst = BuildDualDstGraph({TILE_M, TILE_N * 2}, {TILE_M, TILE_N}, {0, 0}, {0, TILE_N});
    if (graph.dualDst.func == nullptr)
        return false;
    UpdateCopyDynValidShape(graph.dualDst);
    graph.dualTensor0 = graph.dualDst.copy0->GetOutputOperand(0);
    graph.dualTensor1 = graph.dualDst.copy1->GetOutputOperand(0);
    for (const auto& name : {"release0", "release1"}) {
        if (!graph.dualDst.builder->AddTensor(DataType::DT_FP32, {TILE_M, TILE_N}, MemoryType::MEM_UB, name) ||
            !graph.dualDst.builder->AddOp(Opcode::OP_UB_ALLOC, {}, {name}, "alloc_" + std::string(name))) {
            return false;
        }
    }
    for (const auto& name : {"release_in0", "release_in1", "release_out0", "release_out1"}) {
        if (!graph.dualDst.builder->AddTensor(DataType::DT_FP32, {TILE_M, TILE_N}, MemoryType::MEM_DEVICE_DDR, name))
            return false;
    }
    auto& builder = *graph.dualDst.builder;
    graph.releaseCopyin0 = &CreateCopyOp(*graph.dualDst.func, Opcode::OP_COPY_IN, builder.GetTensor("release_in0"),
                                         builder.GetTensor("release0"), {TILE_M, TILE_N});
    graph.releaseCopyin1 = &CreateCopyOp(*graph.dualDst.func, Opcode::OP_COPY_IN, builder.GetTensor("release_in1"),
                                         builder.GetTensor("release1"), {TILE_M, TILE_N});
    graph.releaseCopy0 = &CreateCopyOp(*graph.dualDst.func, Opcode::OP_COPY_OUT, builder.GetTensor("release0"),
                                       builder.GetTensor("release_out0"), {TILE_M, TILE_N});
    graph.releaseCopy1 = &CreateCopyOp(*graph.dualDst.func, Opcode::OP_COPY_OUT, builder.GetTensor("release1"),
                                       builder.GetTensor("release_out1"), {TILE_M, TILE_N});
    graph.releaseAlloc0 = builder.GetOp("alloc_release0");
    graph.releaseAlloc1 = builder.GetOp("alloc_release1");
    return graph.dualTensor0 != nullptr && graph.dualTensor1 != nullptr && graph.releaseAlloc0 != nullptr &&
           graph.releaseAlloc1 != nullptr;
}

std::vector<Operation*> BuildMainLoopReuseOpList(const MainLoopReuseGraph& graph)
{
    std::vector<Operation*> result{graph.releaseAlloc0, graph.releaseAlloc1, graph.dualDst.allocUb0,
                                   graph.dualDst.allocUb1};
    std::unordered_set<Operation*> prefix(result.begin(), result.end());
    for (auto* op : graph.dualDst.func->Operations().DuplicatedOpList()) {
        if (prefix.count(op) == 0)
            result.push_back(op);
    }
    return result;
}

void ConfigureMainLoopReuseCores(OoOScheduler& scheduler, const MainLoopReuseGraph& graph)
{
    InjectCoreMap(scheduler, graph.dualDst);
    auto setCore = [&scheduler](Operation* op, CoreLocationType core) {
        scheduler.state_.schedInfoMap[op].coreLocation = core;
    };
    for (auto* op : {graph.releaseAlloc0, graph.releaseCopyin0, graph.releaseCopy0})
        setCore(op, CoreLocationType::AIV0);
    for (auto* op : {graph.releaseAlloc1, graph.releaseCopyin1, graph.releaseCopy1})
        setCore(op, CoreLocationType::AIV1);
}

size_t OperationIndex(const std::vector<Operation*>& operations, Operation* target)
{
    auto it = std::find(operations.begin(), operations.end(), target);
    return it == operations.end() ? operations.size() : static_cast<size_t>(it - operations.begin());
}

void ExpectMainLoopReuse(OoOScheduler& scheduler, const MainLoopReuseGraph& graph)
{
    const auto& ops = scheduler.state_.newOperations;
    // 普通 alloc 和 DualDst alloc 不在同一轮 BufferAllocStage 下发：
    // COPY_IN 是 LaunchIssueStage 产物，出现在普通 alloc 之后、DualDst alloc 之前，
    // 证明 DualDst alloc 被推到了下一轮。
    EXPECT_LT(OperationIndex(ops, graph.releaseAlloc0), OperationIndex(ops, graph.releaseCopyin0));
    EXPECT_LT(OperationIndex(ops, graph.releaseCopyin0), OperationIndex(ops, graph.dualDst.allocUb0));
    EXPECT_LT(OperationIndex(ops, graph.releaseAlloc1), OperationIndex(ops, graph.releaseCopyin1));
    EXPECT_LT(OperationIndex(ops, graph.releaseCopyin1), OperationIndex(ops, graph.dualDst.allocUb1));
}

struct MixedIsoGraph {
    std::shared_ptr<ComputationalGraphBuilder> builder;
    Operation* allocUbA{nullptr};
    Operation* allocUbB{nullptr};
    Operation* allocL1A{nullptr};
    Operation* allocL1B{nullptr};
    Operation* copyL1A{nullptr};
    Operation* copyL1B{nullptr};
    Operation* allocL0c{nullptr};
    std::vector<Operation*> opsA;
    std::vector<Operation*> opsB;
};

MixedIsoGraph BuildMixedIsoGraph()
{
    MixedIsoGraph graph;
    graph.builder = std::make_shared<ComputationalGraphBuilder>();
    for (const auto& side : {"a", "b"}) {
        EXPECT_TRUE(
            graph.builder->AddTensor(DataType::DT_FP32, {4, 4}, MemoryType::MEM_UB, "t_ub_" + std::string(side)));
        EXPECT_TRUE(
            graph.builder->AddTensor(DataType::DT_FP32, {4, 4}, MemoryType::MEM_L1, "t_l1_" + std::string(side)));
        EXPECT_TRUE(graph.builder->AddOp(Opcode::OP_UB_ALLOC, {}, {"t_ub_" + std::string(side)},
                                         "alloc_ub_" + std::string(side)));
        EXPECT_TRUE(graph.builder->AddOp(Opcode::OP_L1_ALLOC, {}, {"t_l1_" + std::string(side)},
                                         "alloc_l1_" + std::string(side)));
        EXPECT_TRUE(graph.builder->AddOp(Opcode::OP_UB_COPY_L1, {"t_ub_" + std::string(side)},
                                         {"t_l1_" + std::string(side)}, "copy_l1_" + std::string(side)));
    }
    EXPECT_TRUE(graph.builder->AddTensor(DataType::DT_FP32, {4, 4}, MemoryType::MEM_L0C, "t_l0c"));
    EXPECT_TRUE(graph.builder->AddOp(Opcode::OP_L0C_ALLOC, {}, {"t_l0c"}, "alloc_l0c"));
    graph.allocUbA = graph.builder->GetOp("alloc_ub_a");
    graph.allocUbB = graph.builder->GetOp("alloc_ub_b");
    graph.allocL1A = graph.builder->GetOp("alloc_l1_a");
    graph.allocL1B = graph.builder->GetOp("alloc_l1_b");
    graph.copyL1A = graph.builder->GetOp("copy_l1_a");
    graph.copyL1B = graph.builder->GetOp("copy_l1_b");
    graph.allocL0c = graph.builder->GetOp("alloc_l0c");
    graph.opsA = {graph.allocUbA, graph.allocL1A, graph.copyL1A};
    graph.opsB = {graph.allocUbB, graph.allocL1B, graph.copyL1B};
    return graph;
}

// 两侧各有 2 个同签名入口链的镜像图,用于覆盖 IsoMatch 多候选消歧分支。
// 两个入口均为 UB_ALLOC -> ADD,入口局部签名相同;更深一层分别接 MUL/SUB,
// 只有完整子图 hash 能唯一得到 a0<->b0、a1<->b1。
struct MultiEntryIsoGraph {
    std::shared_ptr<ComputationalGraphBuilder> builder;
    std::vector<Operation*> opsA;
    std::vector<Operation*> opsB;
};

MultiEntryIsoGraph BuildMultiEntryIsoGraph()
{
    MultiEntryIsoGraph graph;
    graph.builder = std::make_shared<ComputationalGraphBuilder>();
    for (const auto& side : {"a", "b"}) {
        for (const auto& idx : {"0", "1"}) {
            std::string s = std::string(side) + idx;
            EXPECT_TRUE(graph.builder->AddTensor(DataType::DT_FP32, {4, 4}, MemoryType::MEM_UB, "t_ub_" + s));
            EXPECT_TRUE(graph.builder->AddTensor(DataType::DT_FP32, {4, 4}, MemoryType::MEM_UB, "t_mid_" + s));
            EXPECT_TRUE(graph.builder->AddTensor(DataType::DT_FP32, {4, 4}, MemoryType::MEM_UB, "t_out_" + s));
            EXPECT_TRUE(graph.builder->AddOp(Opcode::OP_UB_ALLOC, {}, {"t_ub_" + s}, "alloc_" + s));
            EXPECT_TRUE(graph.builder->AddOp(Opcode::OP_ADD, {"t_ub_" + s, "t_ub_" + s}, {"t_mid_" + s}, "add_" + s));
            Opcode tailOpcode = std::string(idx) == "0" ? Opcode::OP_MUL : Opcode::OP_SUB;
            EXPECT_TRUE(graph.builder->AddOp(tailOpcode, {"t_mid_" + s, "t_mid_" + s}, {"t_out_" + s}, "tail_" + s));
        }
    }
    graph.opsA = {graph.builder->GetOp("alloc_a0"), graph.builder->GetOp("add_a0"), graph.builder->GetOp("tail_a0"),
                  graph.builder->GetOp("alloc_a1"), graph.builder->GetOp("add_a1"), graph.builder->GetOp("tail_a1")};
    graph.opsB = {graph.builder->GetOp("alloc_b0"), graph.builder->GetOp("add_b0"), graph.builder->GetOp("tail_b0"),
                  graph.builder->GetOp("alloc_b1"), graph.builder->GetOp("add_b1"), graph.builder->GetOp("tail_b1")};
    return graph;
}

void BuildMixedIsoTasks(TaskSplitter& splitter, const MixedIsoGraph& graph)
{
    auto& taskGraph = splitter.GetTaskGraph();
    int aiv0 = taskGraph.AddTask("aiv0", ScheduleCoreType::AIV, 1);
    int aic = taskGraph.AddTask("aic", ScheduleCoreType::AIC, 1);
    int aiv1 = taskGraph.AddTask("aiv1", ScheduleCoreType::AIV, 1);
    taskGraph.tasks[aiv0].opList_ = graph.opsA;
    taskGraph.tasks[aiv0].targetCoreType = TargetCoreType::AIV0;
    taskGraph.tasks[aic].opList_ = {graph.allocL0c};
    taskGraph.tasks[aic].targetCoreType = TargetCoreType::AIC;
    taskGraph.tasks[aiv1].opList_ = graph.opsB;
    taskGraph.tasks[aiv1].targetCoreType = TargetCoreType::AIV1;
}
} // namespace dualdst_ut

// 亲和 dualdst 候选判定：一对 matmul（L0C）后接两个几何相邻的 L0C_COPY_UB，
// 分别喂给两个不连通、等耗时的 AIV task。跑 task 划分 + HLF 调度后，
// branchCandidate_ 应建立一对 (branch→AIV0, branch→AIV1)，方向按 offset 小→AIV0。
TEST_F(ScheduleOoOTest, DualDst_AssignCandidates_SeedsBranchPair_OnHLF)
{
    // fromOff0={0,0} 与 fromOff1={0,TILE_N} 沿 N 方向相邻一个 tile，offset0 < offset1。
    auto g = dualdst_ut::BuildDualDstGraph(
        /*l0cShape*/ {dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
        /*tileShape*/ {dualdst_ut::TILE_M, dualdst_ut::TILE_N},
        /*fromOff0*/ {0, 0},
        /*fromOff1*/ {0, dualdst_ut::TILE_N});

    auto opList = g.func->Operations(false).DuplicatedOpList();
    TaskSplitter splitter;
    splitter.SplitGraph(opList);
    auto& taskGraph = splitter.GetTaskGraph();

    int tCopy0 = dualdst_ut::FindTaskIndex(taskGraph, g.copy0);
    int tCopy1 = dualdst_ut::FindTaskIndex(taskGraph, g.copy1);
    int tAdd0 = dualdst_ut::FindTaskIndex(taskGraph, g.add0);
    int tAdd1 = dualdst_ut::FindTaskIndex(taskGraph, g.add1);
    ASSERT_GE(tCopy0, 0);
    ASSERT_GE(tAdd0, 0);
    ASSERT_GE(tAdd1, 0);
    // 两个 copy 落在同一个 AIC task（同核 union），add 落在各自独立的 AIV task。
    EXPECT_EQ(tCopy0, tCopy1);
    EXPECT_EQ(taskGraph.tasks[tCopy0].coreType, ScheduleCoreType::AIC);
    EXPECT_EQ(taskGraph.tasks[tAdd0].coreType, ScheduleCoreType::AIV);
    EXPECT_EQ(taskGraph.tasks[tAdd1].coreType, ScheduleCoreType::AIV);
    ASSERT_NE(tAdd0, tAdd1);
    // 前提：两 AIV task 不连通（不同 branch）且等耗时。
    int bAdd0 = taskGraph.tasks[tAdd0].vecBranchId;
    int bAdd1 = taskGraph.tasks[tAdd1].vecBranchId;
    ASSERT_NE(bAdd0, bAdd1);
    ASSERT_EQ(taskGraph.tasks[tAdd0].latency, taskGraph.tasks[tAdd1].latency);

    CoreScheduler coreScheduler;
    coreScheduler.Schedule(taskGraph, "HLF");

    // 候选应成对建立：add0(offset 小)→AIV0，add1(offset 大)→AIV1。
    ASSERT_EQ(coreScheduler.branchCandidate_.size(), 2u);
    auto it0 = coreScheduler.branchCandidate_.find(bAdd0);
    auto it1 = coreScheduler.branchCandidate_.find(bAdd1);
    ASSERT_NE(it0, coreScheduler.branchCandidate_.end());
    ASSERT_NE(it1, coreScheduler.branchCandidate_.end());
    EXPECT_EQ(it0->second, TargetCoreType::AIV0);
    EXPECT_EQ(it1->second, TargetCoreType::AIV1);
}

TEST_F(ScheduleOoOTest, DualDst_DynShapeEq_DumpEqual_HitsIdentify)
{
    auto g = dualdst_ut::BuildDualDstGraph_2(
        /*l0cShape*/ {dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
        /*tileShape*/ {dualdst_ut::TILE_M, dualdst_ut::TILE_N},
        /*fromOff0*/ {0, 0},
        /*fromOff1*/ {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape({SymbolicScalar("S0"), SymbolicScalar("S1")});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape({SymbolicScalar("S0"), SymbolicScalar("S1")});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);

    std::vector<DualDstPair> pairs;
    EXPECT_EQ(s.dualDstEngine_.IdentifyDualDstPairs(pairs), SUCCESS);
    EXPECT_EQ(pairs.size(), 1u);
}

TEST_F(ScheduleOoOTest, DualDst_DynShapeEq_ConcreteEqualButDifferentDump_StillHits)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar("a", dualdst_ut::TILE_M), SymbolicScalar("b", dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar("c", dualdst_ut::TILE_M), SymbolicScalar("d", dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);

    std::vector<DualDstPair> pairs;
    EXPECT_EQ(s.dualDstEngine_.IdentifyDualDstPairs(pairs), SUCCESS);
    EXPECT_EQ(pairs.size(), 1u);
}

TEST_F(ScheduleOoOTest, DualDst_DynShapeEq_DumpDifferAndNoConcrete_NoPair)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape({SymbolicScalar("X0"), SymbolicScalar("X1")});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape({SymbolicScalar("Y0"), SymbolicScalar("Y1")});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);

    std::vector<DualDstPair> pairs;
    EXPECT_EQ(s.dualDstEngine_.IdentifyDualDstPairs(pairs), SUCCESS);
    EXPECT_EQ(pairs.size(), 0u);
}
TEST_F(ScheduleOoOTest, DualDst_ReadGeometry_PrefersStaticValidShape)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape({SymbolicScalar("X0"), SymbolicScalar("X1")});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape({SymbolicScalar("Y0"), SymbolicScalar("Y1")});
    dualdst_ut::InjectStaticValidShape(*g.copy0, {dualdst_ut::TILE_M, dualdst_ut::TILE_N});
    dualdst_ut::InjectStaticValidShape(*g.copy1, {dualdst_ut::TILE_M, dualdst_ut::TILE_N});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);

    std::vector<DualDstPair> pairs;
    EXPECT_EQ(s.dualDstEngine_.IdentifyDualDstPairs(pairs), SUCCESS);
    EXPECT_EQ(pairs.size(), 1u);
}

TEST_F(ScheduleOoOTest, DualDst_Identify_SplitN_HappyPath)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);

    std::vector<DualDstPair> pairs;
    EXPECT_EQ(s.dualDstEngine_.IdentifyDualDstPairs(pairs), SUCCESS);
    ASSERT_EQ(pairs.size(), 1u);
    EXPECT_EQ(pairs[0].opEarly, g.copy0);
    EXPECT_EQ(pairs[0].opLate, g.copy1);
    EXPECT_NE(pairs[0].allocEarly, nullptr);
    EXPECT_NE(pairs[0].allocLate, nullptr);
}

TEST_F(ScheduleOoOTest, DualDst_ShouldEnableDualDst_WithOnlineSoftmaxTasks)
{
    auto g = dualdst_ut::BuildOnlineSoftmaxDualDstGraph();
    auto opList = g.func->Operations().DuplicatedOpList();
    TaskSplitter splitter;
    splitter.SplitGraph(opList);
    OoOSchedule oooSchedule;
    ASSERT_EQ(oooSchedule.TaskSchedule(opList, *g.func, splitter), SUCCESS);

    EXPECT_TRUE(dualdst_ut::HasAiv0AndAiv1Tasks(splitter));
    ASSERT_TRUE(oooSchedule.ShouldEnableDualDst(splitter));
    ASSERT_FALSE(oooSchedule.dualDstPairs_.empty());
    EXPECT_TRUE(dualdst_ut::HasIsoAllocPair(oooSchedule.dualDstPairs_, g.allocUb0, g.allocUb1) ||
                dualdst_ut::HasIsoAllocPair(oooSchedule.dualDstPairs_, g.allocOut0, g.allocOut1));
    EXPECT_TRUE(dualdst_ut::HasIsoAllocPair(oooSchedule.dualDstOpPairs_, g.add0, g.add1));

    OoOScheduler scheduler(*g.func);
    ASSERT_EQ(dualdst_ut::InitDualDstScheduler(scheduler, g), SUCCESS);
    scheduler.SetDualDstPairs(oooSchedule.dualDstPairs_);
    scheduler.SetDualDstOpPairs(oooSchedule.dualDstOpPairs_);
    std::vector<DualDstPair> pairs;
    EXPECT_EQ(scheduler.dualDstEngine_.IdentifyDualDstPairs(pairs), SUCCESS);
    ASSERT_TRUE(scheduler.state_.enableDualDst);
    ASSERT_EQ(pairs.size(), 1u);
    EXPECT_EQ(pairs[0].opEarly, g.copy0);
    EXPECT_EQ(pairs[0].opLate, g.copy1);
    EXPECT_EQ(scheduler.dualDstEngine_.RunDualDstFuse(scheduler.state_.orderedOps), SUCCESS);
    EXPECT_TRUE(dualdst_ut::HasDualDstOp(scheduler.state_.orderedOps));
}

// 已存在同 startTime 的合法 AIV0/AIV1 pair 时，额外的单侧 AIV task 仍应关闭 DualDst。
TEST_F(ScheduleOoOTest, DualDst_ShouldEnableDualDst_RejectsSingleSidedStartTime)
{
    auto graph = dualdst_ut::BuildMixedIsoGraph();
    TaskSplitter splitter;
    dualdst_ut::BuildMixedIsoTasks(splitter, graph);
    auto& taskGraph = splitter.GetTaskGraph();
    int singleAiv = taskGraph.AddTask("single_aiv", ScheduleCoreType::AIV, 1);
    taskGraph.tasks[singleAiv].opList_ = {graph.allocUbA};
    taskGraph.tasks[singleAiv].targetCoreType = TargetCoreType::AIV0;
    taskGraph.tasks[singleAiv].startTime = 1;

    OoOSchedule schedule;
    EXPECT_FALSE(schedule.ShouldEnableDualDst(splitter));
    EXPECT_TRUE(schedule.dualDstPairs_.empty());
    EXPECT_TRUE(schedule.dualDstOpPairs_.empty());
}

TEST_F(ScheduleOoOTest, DualDst_Realign_MovesConsumersWithAllocs)
{
    auto g = dualdst_ut::BuildOnlineSoftmaxDualDstGraph();
    OoOScheduler s(*g.func);
    ASSERT_EQ(dualdst_ut::InitDualDstScheduler(s, g), SUCCESS);
    s.SetDualDstPairs({{g.allocUb0, g.allocUb1}});
    s.SetDualDstOpPairs({{g.add0, g.add1}});

    auto& ops = s.state_.orderedOps;

    // 构造出的 opList 并非拓扑序（copy 排在自己的 UB_ALLOC 之前），拓扑守卫会直接
    // 否掉整次重排。先归位成拓扑序，让重排本身有机会执行。
    ASSERT_TRUE(dualdst_ut::TopoSortInPlace(s.state_.depManager, ops));

    auto slotOf = [&ops](Operation* op) {
        return static_cast<int>(std::find(ops.begin(), ops.end(), op) - ops.begin());
    };
    int allocSlot = slotOf(g.allocUb1);
    int addSlot = slotOf(g.add1);
    ASSERT_LT(addSlot, static_cast<int>(ops.size()));
    ASSERT_LT(allocSlot, addSlot);
    ASSERT_LT(slotOf(g.allocUb0), slotOf(g.add0));

    std::swap(ops[allocSlot], ops[addSlot]);
    auto before = ops;
    ASSERT_EQ(s.dualDstEngine_.RealignAllocByIso(ops), SUCCESS);

    ASSERT_EQ(before.size(), ops.size());
    EXPECT_EQ(std::unordered_set<Operation*>(before.begin(), before.end()),
              std::unordered_set<Operation*>(ops.begin(), ops.end()));
    EXPECT_EQ(slotOf(g.allocUb1), allocSlot);
    EXPECT_EQ(slotOf(g.add1), addSlot);
}

TEST_F(ScheduleOoOTest, DualDst_Realign_SkipsWhenTopoOrderWouldBreak)
{
    auto g = dualdst_ut::BuildOnlineSoftmaxDualDstGraph();
    OoOScheduler s(*g.func);
    ASSERT_EQ(dualdst_ut::InitDualDstScheduler(s, g), SUCCESS);
    // Pairing an alloc with a consumer of the other core forces the mirrored order to
    // place a consumer ahead of its own producer, which the topo guard must reject.
    s.SetDualDstPairs({{g.allocUb0, g.allocUb1}});
    s.SetDualDstOpPairs({{g.add0, g.allocOut1}});

    auto before = s.state_.orderedOps;
    EXPECT_EQ(s.dualDstEngine_.RealignAllocByIso(s.state_.orderedOps), SUCCESS);
    EXPECT_EQ(before, s.state_.orderedOps);
}

TEST_F(ScheduleOoOTest, DualDst_Realign_SkipsPairsOutsideOpList)
{
    auto g = dualdst_ut::BuildOnlineSoftmaxDualDstGraph();
    auto other = dualdst_ut::BuildOnlineSoftmaxDualDstGraph();
    OoOScheduler s(*g.func);
    ASSERT_EQ(dualdst_ut::InitDualDstScheduler(s, g), SUCCESS);
    s.SetDualDstPairs({{g.allocUb0, g.allocUb1}});
    s.SetDualDstOpPairs({{other.add0, other.add1}});

    auto before = s.state_.orderedOps;
    EXPECT_EQ(s.dualDstEngine_.RealignAllocByIso(s.state_.orderedOps), SUCCESS);
    EXPECT_EQ(before, s.state_.orderedOps);
}

TEST_F(ScheduleOoOTest, DualDst_Realign_SkipsWhenOneAiv1OpHasTwoAiv0Pairs)
{
    auto g = dualdst_ut::BuildOnlineSoftmaxDualDstGraph();
    OoOScheduler s(*g.func);
    ASSERT_EQ(dualdst_ut::InitDualDstScheduler(s, g), SUCCESS);
    s.SetDualDstPairs({{g.allocUb0, g.allocUb1}});
    s.SetDualDstOpPairs({{g.add0, g.allocUb1}});

    auto before = s.state_.orderedOps;
    EXPECT_EQ(s.dualDstEngine_.RealignAllocByIso(s.state_.orderedOps), SUCCESS);
    EXPECT_EQ(before, s.state_.orderedOps);
}

TEST_F(ScheduleOoOTest, DualDst_Identify_SplitM_HappyPath)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M * 2, dualdst_ut::TILE_N},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {dualdst_ut::TILE_M, 0});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);

    std::vector<DualDstPair> pairs;
    EXPECT_EQ(s.dualDstEngine_.IdentifyDualDstPairs(pairs), SUCCESS);
    ASSERT_EQ(pairs.size(), 1u);
    EXPECT_EQ(pairs[0].opEarly, g.copy0); // fromM 0 < TILE_M
    EXPECT_EQ(pairs[0].opLate, g.copy1);
    auto l0c = g.copy0->GetInputOperand(0);
    EXPECT_EQ(s.dualDstEngine_.dualDstL0CDirection_.count(l0c), 1u);
    EXPECT_EQ(s.dualDstEngine_.dualDstL0CDirection_[l0c], 0);
}

TEST_F(ScheduleOoOTest, DualDst_Identify_NotAdjacent_NoPair)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 4},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0},
                                             {0, dualdst_ut::TILE_N * 2});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);

    std::vector<DualDstPair> pairs;
    EXPECT_EQ(s.dualDstEngine_.IdentifyDualDstPairs(pairs), SUCCESS);
    EXPECT_EQ(pairs.size(), 0u);
}

// 同一 L0C 存在三个 L0C_COPY_UB 时，当前策略为保守禁用配对，
// 避免多个候选间的 Copy/UB_ALLOC 顺序无法唯一确定。
TEST_F(ScheduleOoOTest, DualDst_Identify_ThreeCopyUbsForOneL0C_NoPair)
{
    auto g = dualdst_ut::BuildDualDstGraph({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 3},
                                           {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    EXPECT_TRUE(
        g.builder->AddTensor(DataType::DT_FP32, {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, MemoryType::MEM_UB, "t_ub2"));
    EXPECT_TRUE(g.builder->AddTensor(DataType::DT_FP32, {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, MemoryType::MEM_UB,
                                     "t_out2"));
    EXPECT_TRUE(g.builder->AddOp(Opcode::OP_UB_ALLOC, {}, {"t_ub2"}, "alloc_ub2"));
    EXPECT_TRUE(g.builder->AddOp(Opcode::OP_UB_ALLOC, {}, {"t_out2"}, "alloc_out2"));
    EXPECT_TRUE(g.builder->AddOp(Opcode::OP_L0C_COPY_UB, {"t_l0c"}, {"t_ub2"}, "copy2"));
    EXPECT_TRUE(g.builder->AddOp(Opcode::OP_ADD, {"t_ub2", "t_ub2"}, {"t_out2"}, "add2"));

    Operation* copy2 = g.builder->GetOp("copy2");
    Operation* allocUb2 = g.builder->GetOp("alloc_ub2");
    Operation* allocOut2 = g.builder->GetOp("alloc_out2");
    Operation* add2 = g.builder->GetOp("add2");
    ASSERT_NE(copy2, nullptr);
    ASSERT_NE(allocUb2, nullptr);
    ASSERT_NE(allocOut2, nullptr);
    ASSERT_NE(add2, nullptr);
    dualdst_ut::SetCopyL0cToUbAttr(*copy2, {0, dualdst_ut::TILE_N * 2}, {dualdst_ut::TILE_M, dualdst_ut::TILE_N});
    dualdst_ut::UpdateCopyDynValidShape(g);
    copy2->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    OoOScheduler scheduler(*g.func);
    ASSERT_EQ(scheduler.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(scheduler, g);
    scheduler.state_.schedInfoMap[copy2].coreLocation = CoreLocationType::AIC;
    scheduler.state_.schedInfoMap[allocUb2].coreLocation = CoreLocationType::AIV0;
    scheduler.state_.schedInfoMap[allocOut2].coreLocation = CoreLocationType::AIV0;
    scheduler.state_.schedInfoMap[add2].coreLocation = CoreLocationType::AIV0;

    std::vector<DualDstPair> pairs;
    EXPECT_EQ(scheduler.dualDstEngine_.IdentifyDualDstPairs(pairs), SUCCESS);
    EXPECT_TRUE(pairs.empty());
}

// 验证候选识别不依赖 ONLINE_SOFTMAX 等特定消费者：
// 两个相邻 Copy 后接普通 ADD，只要分别落在 AIV0/AIV1，仍应识别出一对。
TEST_F(ScheduleOoOTest, DualDst_Identify_AddConsumer_HitsPair)
{
    auto g = dualdst_ut::BuildDualDstGraph({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                           {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);

    std::vector<DualDstPair> pairs;
    EXPECT_EQ(s.dualDstEngine_.IdentifyDualDstPairs(pairs), SUCCESS);
    ASSERT_EQ(pairs.size(), 1u);
    EXPECT_EQ(pairs[0].opEarly, g.copy0);
    EXPECT_EQ(pairs[0].opLate, g.copy1);
}

TEST_F(ScheduleOoOTest, DualDst_Identify_SameConsumerCore_NoPair)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g, /*sameCoreForAdds=*/true); // both consumers AIV0

    std::vector<DualDstPair> pairs;
    EXPECT_EQ(s.dualDstEngine_.IdentifyDualDstPairs(pairs), SUCCESS);
    EXPECT_EQ(pairs.size(), 0u);
}

TEST_F(ScheduleOoOTest, DualDst_RunDualDstFuse_DisabledIsNoOp)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);

    s.SetEnableDualDst(false);
    EXPECT_EQ(s.dualDstEngine_.RunDualDstFuse(s.state_.orderedOps), SUCCESS);
}

TEST_F(ScheduleOoOTest, DualDst_RunDualDstFuse_SingleAivPoolEarlyExit)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_ONE), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);

    s.SetEnableDualDst(true);
    EXPECT_EQ(s.dualDstEngine_.RunDualDstFuse(s.state_.orderedOps), SUCCESS);
}

// 验证融合改图的完整结果：两个旧 Copy 被一个 DualDst Copy 替换，
// 插入位置、tensor 边、依赖、alloc 配对元数据及 bufRefCount 均保持正确。
TEST_F(ScheduleOoOTest, DualDst_RunDualDstFuse_ActuallyFusesAndMutatesFunction)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);
    ASSERT_NE(g.copy0->GetInputOperand(0), nullptr);
    ASSERT_NE(g.copy0->GetOutputOperand(0), nullptr);
    ASSERT_NE(g.copy1->GetOutputOperand(0), nullptr);
    ASSERT_NE(std::find(s.state_.orderedOps.begin(), s.state_.orderedOps.end(), g.copy0), s.state_.orderedOps.end());
    ASSERT_NE(std::find(s.state_.orderedOps.begin(), s.state_.orderedOps.end(), g.copy1), s.state_.orderedOps.end());
    auto before = dualdst_ut::CaptureFuseSnapshot(s, g);
    s.SetEnableDualDst(true);
    EXPECT_EQ(s.dualDstEngine_.RunDualDstFuse(s.state_.orderedOps), SUCCESS);
    dualdst_ut::ExpectFusedGraph(s, g, before);
    dualdst_ut::ExpectFusedMetadata(s, g, before);
}

TEST_F(ScheduleOoOTest, DualDst_FusedCopyPreservesDynamicValidShapeOnNonSplitAxis)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M * 2, dualdst_ut::TILE_N},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {dualdst_ut::TILE_M, 0});
    const std::vector<SymbolicScalar> validShape = {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar("n")};
    for (Operation* copy : {g.copy0, g.copy1}) {
        copy->GetOutputOperand(0)->UpdateDynValidShape(validShape);
        auto attr = std::dynamic_pointer_cast<CopyOpAttribute>(copy->GetOpAttribute());
        ASSERT_NE(attr, nullptr);
        attr->SetToDynValidShape(OpImmediate::Specified(validShape));
    }

    OoOScheduler s(*g.func);
    ASSERT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);
    s.SetEnableDualDst(true);
    ASSERT_EQ(s.dualDstEngine_.RunDualDstFuse(s.state_.orderedOps), SUCCESS);

    Operation* dual = dualdst_ut::FindDualDstOp(*g.func);
    ASSERT_NE(dual, nullptr);
    auto attr = std::dynamic_pointer_cast<CopyOpAttribute>(dual->GetOpAttribute());
    ASSERT_NE(attr, nullptr);
    const auto fusedValid = OpImmediate::ToSpecified(attr->GetToDynValidShape());
    ASSERT_EQ(fusedValid.size(), 2u);
    ASSERT_TRUE(fusedValid[0].ConcreteValid());
    EXPECT_EQ(fusedValid[0].Concrete(), dualdst_ut::TILE_M * 2);
    EXPECT_EQ(fusedValid[1].Dump(), "n");
}

TEST_F(ScheduleOoOTest, DualDst_FusedCopyPreservesDynamicValidShapeOnNonSplitAxis_DirectionN)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    const std::vector<SymbolicScalar> validShape = {SymbolicScalar("m"), SymbolicScalar(dualdst_ut::TILE_N)};
    for (Operation* copy : {g.copy0, g.copy1}) {
        copy->GetOutputOperand(0)->UpdateDynValidShape(validShape);
        auto attr = std::dynamic_pointer_cast<CopyOpAttribute>(copy->GetOpAttribute());
        ASSERT_NE(attr, nullptr);
        attr->SetToDynValidShape(OpImmediate::Specified(validShape));
    }

    OoOScheduler s(*g.func);
    ASSERT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);
    s.SetEnableDualDst(true);
    ASSERT_EQ(s.dualDstEngine_.RunDualDstFuse(s.state_.orderedOps), SUCCESS);

    Operation* dual = dualdst_ut::FindDualDstOp(*g.func);
    ASSERT_NE(dual, nullptr);
    auto attr = std::dynamic_pointer_cast<CopyOpAttribute>(dual->GetOpAttribute());
    ASSERT_NE(attr, nullptr);
    const auto fusedValid = OpImmediate::ToSpecified(attr->GetToDynValidShape());
    ASSERT_EQ(fusedValid.size(), 2u);
    EXPECT_EQ(fusedValid[0].Dump(), "m");
    ASSERT_TRUE(fusedValid[1].ConcreteValid());
    EXPECT_EQ(fusedValid[1].Concrete(), dualdst_ut::TILE_N * 2);
}

// 验证融合后两个 UB_ALLOC 互相记录 pairedDualDstAlloc，
// 且 L0C_ALLOC 不会被误标记为 DualDst alloc。
TEST_F(ScheduleOoOTest, DualDst_AllocQueryHelpers_AfterFuse)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);
    s.SetEnableDualDst(true);
    EXPECT_EQ(s.dualDstEngine_.RunDualDstFuse(s.state_.orderedOps), SUCCESS);

    Operation* dual = nullptr;
    for (auto& op : g.func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_L0C_COPY_UB_DUAL_DST) {
            dual = &op;
            break;
        }
    }
    ASSERT_NE(dual, nullptr);

    std::vector<Operation*> dualAllocs;
    for (auto* pred : s.state_.depManager.GetPredecessors(dual)) {
        if (pred != nullptr && pred->GetOpcodeStr().find("UB_ALLOC") != std::string::npos) {
            dualAllocs.push_back(pred);
        }
    }
    ASSERT_EQ(dualAllocs.size(), 2u);
    EXPECT_TRUE(s.state_.IsDualDstAlloc(dualAllocs[0]));
    EXPECT_TRUE(s.state_.IsDualDstAlloc(dualAllocs[1]));
    EXPECT_EQ(s.state_.schedInfoMap[dualAllocs[0]].pairedDualDstAlloc, dualAllocs[1]);
    EXPECT_EQ(s.state_.schedInfoMap[dualAllocs[1]].pairedDualDstAlloc, dualAllocs[0]);

    EXPECT_FALSE(s.state_.IsDualDstAlloc(g.allocL0c));
}

// 验证普通 AIV UB alloc 的跨核地址对齐：AIV0 先在非零 offset 分配，
// AIV1 随后必须使用相同 offset，并在匹配完成后清空两侧 alloc 记录。
TEST_F(ScheduleOoOTest, DualDst_AivUbAllocUsesMatchedPeerOffset)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    s.SetEnableDualDst(true);

    constexpr int kPlaceholderMemId = 92003;
    constexpr uint64_t kAllocSize = 256;
    constexpr uint64_t kPlaceholderSize = 128;
    constexpr uint64_t kPoolSize = 2048;

    s.state_.bufferManagerMap[CoreLocationType::AIV0][MemoryType::MEM_UB] = BufferPool(MemoryType::MEM_UB, kPoolSize);
    s.state_.bufferManagerMap[CoreLocationType::AIV1][MemoryType::MEM_UB] = BufferPool(MemoryType::MEM_UB, kPoolSize);
    auto& aiv0Pool = s.state_.bufferManagerMap[CoreLocationType::AIV0][MemoryType::MEM_UB];
    auto& aiv1Pool = s.state_.bufferManagerMap[CoreLocationType::AIV1][MemoryType::MEM_UB];
    auto placeholder = std::make_shared<LocalBuffer>(kPlaceholderMemId, kPlaceholderSize, 0, MemoryType::MEM_UB);
    ASSERT_EQ(aiv0Pool.AllocateAtOffset(placeholder, 0), SUCCESS);

    Operation* aiv0Alloc = dualdst_ut::AddAivUbAlloc(*g.func, s, CoreLocationType::AIV0, kAllocSize);
    Operation* aiv1Alloc = dualdst_ut::AddAivUbAlloc(*g.func, s, CoreLocationType::AIV1, kAllocSize);
    int aiv0MemId = aiv0Alloc->GetOutputOperand(0)->memoryrange.memId;
    int aiv1MemId = aiv1Alloc->GetOutputOperand(0)->memoryrange.memId;

    uint64_t commitCnt = 0;
    bool allocated = false;
    EXPECT_EQ(s.TryRegularAllocOnce(aiv0Alloc, MemoryType::MEM_UB, CoreLocationType::AIV0,
                                    s.state_.GetOpMemIds(aiv0Alloc), commitCnt, allocated),
              SUCCESS);
    ASSERT_TRUE(allocated);
    ASSERT_EQ(aiv0Pool.GetBufferOffset(aiv0MemId), kPlaceholderSize);
    ASSERT_EQ(s.dualDstEngine_.aiv0UbAllocRecords_.size(), 1u);
    ASSERT_TRUE(s.dualDstEngine_.aiv1UbAllocRecords_.empty());

    allocated = false;
    EXPECT_EQ(s.TryRegularAllocOnce(aiv1Alloc, MemoryType::MEM_UB, CoreLocationType::AIV1,
                                    s.state_.GetOpMemIds(aiv1Alloc), commitCnt, allocated),
              SUCCESS);
    ASSERT_TRUE(allocated);
    EXPECT_EQ(aiv1Pool.GetBufferOffset(aiv1MemId), kPlaceholderSize);
    EXPECT_NE(aiv1Pool.GetBufferOffset(aiv1MemId), 0u);
    EXPECT_TRUE(s.dualDstEngine_.aiv0UbAllocRecords_.empty());
    EXPECT_TRUE(s.dualDstEngine_.aiv1UbAllocRecords_.empty());
}

// 验证普通 alloc 与 DualDst alloc 强制分属两次 BufferAllocStage：
// 第一次调用只退休普通 alloc 并停在 DualDst 边界，第二次调用才联合执行 DualDst pair。
TEST_F(ScheduleOoOTest, DualDst_AivUbAllocSeparatesRegularAndDualDstStages)
{
    auto graph = dualdst_ut::BuildOnlineSoftmaxDualDstGraph();
    OoOScheduler scheduler(*graph.func);
    ASSERT_EQ(dualdst_ut::InitDualDstScheduler(scheduler, graph), SUCCESS);
    ASSERT_EQ(scheduler.dualDstEngine_.RunDualDstFuse(scheduler.state_.orderedOps), SUCCESS);

    constexpr uint64_t kRegularSize = 256;
    Operation* regular0 = dualdst_ut::AddAivUbAlloc(*graph.func, scheduler, CoreLocationType::AIV0, kRegularSize);
    Operation* regular1 = dualdst_ut::AddAivUbAlloc(*graph.func, scheduler, CoreLocationType::AIV1, kRegularSize);
    scheduler.state_.schedInfoMap[regular0].execOrder = 0;
    scheduler.state_.schedInfoMap[regular1].execOrder = 0;
    scheduler.state_.schedInfoMap[graph.allocUb0].execOrder = 1;
    scheduler.state_.schedInfoMap[graph.allocUb1].execOrder = 1;

    auto& queue0 = scheduler.state_.allocIssueQueue[CoreLocationType::AIV0][MemoryType::MEM_UB];
    auto& queue1 = scheduler.state_.allocIssueQueue[CoreLocationType::AIV1][MemoryType::MEM_UB];
    queue0.queue.clear();
    queue1.queue.clear();
    queue0.Insert(regular0);
    queue0.Insert(graph.allocUb0);
    queue1.Insert(regular1);
    queue1.Insert(graph.allocUb1);

    uint64_t commitCnt = 0;
    ASSERT_EQ(scheduler.ExecuteAivUbAllocRound(commitCnt), SUCCESS);
    EXPECT_TRUE(scheduler.state_.continueAllocStage);
    EXPECT_TRUE(scheduler.state_.schedInfoMap[regular0].isRetired);
    EXPECT_TRUE(scheduler.state_.schedInfoMap[regular1].isRetired);
    EXPECT_FALSE(scheduler.state_.schedInfoMap[graph.allocUb0].isRetired);
    EXPECT_FALSE(scheduler.state_.schedInfoMap[graph.allocUb1].isRetired);
    ASSERT_EQ(queue0.Front(), graph.allocUb0);
    ASSERT_EQ(queue1.Front(), graph.allocUb1);

    ASSERT_EQ(scheduler.ExecuteAivUbAllocRound(commitCnt), SUCCESS);
    EXPECT_FALSE(scheduler.state_.continueAllocStage);
    EXPECT_TRUE(scheduler.state_.schedInfoMap[graph.allocUb0].isRetired);
    EXPECT_TRUE(scheduler.state_.schedInfoMap[graph.allocUb1].isRetired);
    EXPECT_TRUE(queue0.Empty());
    EXPECT_TRUE(queue1.Empty());
}

// 验证两侧队头类型不一致时拒绝执行：AIV0 为普通 alloc、AIV1 为 DualDst alloc，
// 应返回 FAILED，且不退休 op、不弹出队列。
TEST_F(ScheduleOoOTest, DualDst_AivUbAllocRejectsMismatchedFrontKinds)
{
    auto graph = dualdst_ut::BuildOnlineSoftmaxDualDstGraph();
    OoOScheduler scheduler(*graph.func);
    ASSERT_EQ(dualdst_ut::InitDualDstScheduler(scheduler, graph), SUCCESS);
    ASSERT_EQ(scheduler.dualDstEngine_.RunDualDstFuse(scheduler.state_.orderedOps), SUCCESS);
    constexpr uint64_t kRegularSize = 256;
    Operation* regular0 = dualdst_ut::AddAivUbAlloc(*graph.func, scheduler, CoreLocationType::AIV0, kRegularSize);
    auto& queue0 = scheduler.state_.allocIssueQueue[CoreLocationType::AIV0][MemoryType::MEM_UB];
    auto& queue1 = scheduler.state_.allocIssueQueue[CoreLocationType::AIV1][MemoryType::MEM_UB];
    queue0.queue.clear();
    queue1.queue.clear();
    queue0.Insert(regular0);
    queue1.Insert(graph.allocUb1);

    uint64_t commitCnt = 0;
    size_t newOperationCount = scheduler.state_.newOperations.size();
    EXPECT_EQ(scheduler.ExecuteAivUbAllocRound(commitCnt), FAILED);
    EXPECT_EQ(commitCnt, 0u);
    EXPECT_EQ(scheduler.state_.newOperations.size(), newOperationCount);
    EXPECT_EQ(queue0.Front(), regular0);
    EXPECT_EQ(queue1.Front(), graph.allocUb1);
    EXPECT_FALSE(scheduler.state_.schedInfoMap[regular0].isRetired);
    EXPECT_FALSE(scheduler.state_.schedInfoMap[graph.allocUb1].isRetired);
    int regularMemId = regular0->GetOutputOperand(0)->memoryrange.memId;
    int dualMemId = graph.allocUb1->GetOutputOperand(0)->memoryrange.memId;
    auto& pool0 = scheduler.state_.bufferManagerMap[CoreLocationType::AIV0][MemoryType::MEM_UB];
    auto& pool1 = scheduler.state_.bufferManagerMap[CoreLocationType::AIV1][MemoryType::MEM_UB];
    EXPECT_FALSE(pool0.isAllocate(regularMemId));
    EXPECT_FALSE(pool1.isAllocate(dualMemId));
}

// 验证仅一侧存在待执行 alloc 时拒绝执行，避免单核独自推进破坏两侧 UB 布局一致性。
TEST_F(ScheduleOoOTest, DualDst_AivUbAllocRejectsUnsynchronizedQueues)
{
    auto graph = dualdst_ut::BuildOnlineSoftmaxDualDstGraph();
    OoOScheduler scheduler(*graph.func);
    ASSERT_EQ(dualdst_ut::InitDualDstScheduler(scheduler, graph), SUCCESS);
    ASSERT_EQ(scheduler.dualDstEngine_.RunDualDstFuse(scheduler.state_.orderedOps), SUCCESS);
    auto& queue0 = scheduler.state_.allocIssueQueue[CoreLocationType::AIV0][MemoryType::MEM_UB];
    auto& queue1 = scheduler.state_.allocIssueQueue[CoreLocationType::AIV1][MemoryType::MEM_UB];
    queue0.queue.clear();
    queue1.queue.clear();
    queue0.Insert(graph.allocUb0);
    auto& pool0 = scheduler.state_.bufferManagerMap[CoreLocationType::AIV0][MemoryType::MEM_UB];
    auto& pool1 = scheduler.state_.bufferManagerMap[CoreLocationType::AIV1][MemoryType::MEM_UB];
    auto pool0Before = pool0.GetSortedAllocatedBufs();
    auto pool1Before = pool1.GetSortedAllocatedBufs();

    uint64_t commitCnt = 0;
    size_t newOperationCount = scheduler.state_.newOperations.size();
    EXPECT_EQ(scheduler.ExecuteAivUbAllocRound(commitCnt), FAILED);
    EXPECT_EQ(commitCnt, 0u);
    EXPECT_EQ(scheduler.state_.newOperations.size(), newOperationCount);
    EXPECT_EQ(queue0.Front(), graph.allocUb0);
    EXPECT_TRUE(queue1.Empty());
    EXPECT_FALSE(scheduler.state_.schedInfoMap[graph.allocUb0].isRetired);
    EXPECT_EQ(pool0.GetSortedAllocatedBufs(), pool0Before);
    EXPECT_EQ(pool1.GetSortedAllocatedBufs(), pool1Before);
}

// 验证两侧虽然都是 DualDst alloc，但 pairing 元数据不互指时拒绝执行。
TEST_F(ScheduleOoOTest, DualDst_AivUbAllocRejectsUnpairedFronts)
{
    auto graph = dualdst_ut::BuildOnlineSoftmaxDualDstGraph();
    OoOScheduler scheduler(*graph.func);
    ASSERT_EQ(dualdst_ut::InitDualDstScheduler(scheduler, graph), SUCCESS);
    ASSERT_EQ(scheduler.dualDstEngine_.RunDualDstFuse(scheduler.state_.orderedOps), SUCCESS);
    auto& queue0 = scheduler.state_.allocIssueQueue[CoreLocationType::AIV0][MemoryType::MEM_UB];
    auto& queue1 = scheduler.state_.allocIssueQueue[CoreLocationType::AIV1][MemoryType::MEM_UB];
    queue0.queue.clear();
    queue1.queue.clear();
    queue0.Insert(graph.allocUb0);
    queue1.Insert(graph.allocUb1);
    scheduler.state_.schedInfoMap[graph.allocUb0].pairedDualDstAlloc = nullptr;
    auto& pool0 = scheduler.state_.bufferManagerMap[CoreLocationType::AIV0][MemoryType::MEM_UB];
    auto& pool1 = scheduler.state_.bufferManagerMap[CoreLocationType::AIV1][MemoryType::MEM_UB];
    auto pool0Before = pool0.GetSortedAllocatedBufs();
    auto pool1Before = pool1.GetSortedAllocatedBufs();

    uint64_t commitCnt = 0;
    size_t newOperationCount = scheduler.state_.newOperations.size();
    EXPECT_EQ(scheduler.ExecuteAivUbAllocRound(commitCnt), FAILED);
    EXPECT_EQ(commitCnt, 0u);
    EXPECT_EQ(scheduler.state_.newOperations.size(), newOperationCount);
    EXPECT_EQ(queue0.Front(), graph.allocUb0);
    EXPECT_EQ(queue1.Front(), graph.allocUb1);
    EXPECT_FALSE(scheduler.state_.schedInfoMap[graph.allocUb0].isRetired);
    EXPECT_FALSE(scheduler.state_.schedInfoMap[graph.allocUb1].isRetired);
    EXPECT_EQ(pool0.GetSortedAllocatedBufs(), pool0Before);
    EXPECT_EQ(pool1.GetSortedAllocatedBufs(), pool1Before);
}

// 验证 DualDst pair 因空间不足而停止时不要求继续 alloc stage，
// 两侧 alloc 均不退休且仍保留在队首，供主循环触发 SpillOnBlock 后重试。
TEST_F(ScheduleOoOTest, DualDst_AivUbAllocReportsBufferFull)
{
    auto graph = dualdst_ut::BuildOnlineSoftmaxDualDstGraph();
    OoOScheduler scheduler(*graph.func);
    ASSERT_EQ(dualdst_ut::InitDualDstScheduler(scheduler, graph), SUCCESS);
    ASSERT_EQ(scheduler.dualDstEngine_.RunDualDstFuse(scheduler.state_.orderedOps), SUCCESS);
    auto& queue0 = scheduler.state_.allocIssueQueue[CoreLocationType::AIV0][MemoryType::MEM_UB];
    auto& queue1 = scheduler.state_.allocIssueQueue[CoreLocationType::AIV1][MemoryType::MEM_UB];
    queue0.queue.clear();
    queue1.queue.clear();
    queue0.Insert(graph.allocUb0);
    queue1.Insert(graph.allocUb1);
    size_t poolSize = scheduler.state_.bufferManagerMap[CoreLocationType::AIV0][MemoryType::MEM_UB].GetMemSize();
    constexpr int kPlaceholderMemId0 = 90003;
    constexpr int kPlaceholderMemId1 = 90004;
    ASSERT_EQ(dualdst_ut::FillAivPoolsWithPlaceholderBuffers(scheduler, graph, poolSize, kPlaceholderMemId0,
                                                             kPlaceholderMemId1),
              SUCCESS);

    uint64_t commitCnt = 0;
    size_t newOperationCount = scheduler.state_.newOperations.size();
    ASSERT_EQ(scheduler.ExecuteAivUbAllocRound(commitCnt), SUCCESS);
    EXPECT_FALSE(scheduler.state_.continueAllocStage);
    EXPECT_EQ(commitCnt, 0u);
    EXPECT_EQ(scheduler.state_.newOperations.size(), newOperationCount);
    EXPECT_FALSE(scheduler.state_.schedInfoMap[graph.allocUb0].isRetired);
    EXPECT_FALSE(scheduler.state_.schedInfoMap[graph.allocUb1].isRetired);
    EXPECT_EQ(queue0.Front(), graph.allocUb0);
    EXPECT_EQ(queue1.Front(), graph.allocUb1);
    int memId0 = graph.allocUb0->GetOutputOperand(0)->memoryrange.memId;
    int memId1 = graph.allocUb1->GetOutputOperand(0)->memoryrange.memId;
    auto& pool0 = scheduler.state_.bufferManagerMap[CoreLocationType::AIV0][MemoryType::MEM_UB];
    auto& pool1 = scheduler.state_.bufferManagerMap[CoreLocationType::AIV1][MemoryType::MEM_UB];
    EXPECT_FALSE(pool0.isAllocate(memId0));
    EXPECT_FALSE(pool1.isAllocate(memId1));
    EXPECT_EQ(scheduler.CheckAivUbPoolSlicesEqual(), SUCCESS);
}

// 验证 DualDst pair 成功后若队列仍有 alloc，应继续下一 alloc stage，
// 而不是把尚未执行的普通 alloc 误判为 spill 阻塞。
TEST_F(ScheduleOoOTest, DualDst_AivUbAllocContinuesAfterDualDstStage)
{
    auto graph = dualdst_ut::BuildOnlineSoftmaxDualDstGraph();
    OoOScheduler scheduler(*graph.func);
    ASSERT_EQ(dualdst_ut::InitDualDstScheduler(scheduler, graph), SUCCESS);
    ASSERT_EQ(scheduler.dualDstEngine_.RunDualDstFuse(scheduler.state_.orderedOps), SUCCESS);
    constexpr uint64_t kRegularSize = 1024;
    Operation* regular0 = dualdst_ut::AddAivUbAlloc(*graph.func, scheduler, CoreLocationType::AIV0, kRegularSize);
    Operation* regular1 = dualdst_ut::AddAivUbAlloc(*graph.func, scheduler, CoreLocationType::AIV1, kRegularSize);
    scheduler.state_.schedInfoMap[graph.allocUb0].execOrder = 0;
    scheduler.state_.schedInfoMap[graph.allocUb1].execOrder = 0;
    scheduler.state_.schedInfoMap[regular0].execOrder = 1;
    scheduler.state_.schedInfoMap[regular1].execOrder = 1;
    auto& queue0 = scheduler.state_.allocIssueQueue[CoreLocationType::AIV0][MemoryType::MEM_UB];
    auto& queue1 = scheduler.state_.allocIssueQueue[CoreLocationType::AIV1][MemoryType::MEM_UB];
    queue0.queue.clear();
    queue1.queue.clear();
    queue0.Insert(graph.allocUb0);
    queue0.Insert(regular0);
    queue1.Insert(graph.allocUb1);
    queue1.Insert(regular1);

    uint64_t commitCnt = 0;
    size_t newOperationCount = scheduler.state_.newOperations.size();
    ASSERT_EQ(scheduler.ExecuteAivUbAllocRound(commitCnt), SUCCESS);
    EXPECT_TRUE(scheduler.state_.continueAllocStage);
    EXPECT_EQ(commitCnt, 2u);
    EXPECT_EQ(scheduler.state_.newOperations.size(), newOperationCount + 2);
    EXPECT_TRUE(scheduler.state_.schedInfoMap[graph.allocUb0].isRetired);
    EXPECT_TRUE(scheduler.state_.schedInfoMap[graph.allocUb1].isRetired);
    int memId0 = graph.allocUb0->GetOutputOperand(0)->memoryrange.memId;
    int memId1 = graph.allocUb1->GetOutputOperand(0)->memoryrange.memId;
    auto& pool0 = scheduler.state_.bufferManagerMap[CoreLocationType::AIV0][MemoryType::MEM_UB];
    auto& pool1 = scheduler.state_.bufferManagerMap[CoreLocationType::AIV1][MemoryType::MEM_UB];
    EXPECT_TRUE(pool0.isAllocate(memId0));
    EXPECT_TRUE(pool1.isAllocate(memId1));
    EXPECT_EQ(pool0.GetBufferOffset(memId0), pool1.GetBufferOffset(memId1));
    EXPECT_EQ(queue0.Front(), regular0);
    EXPECT_EQ(queue1.Front(), regular1);
    ASSERT_EQ(scheduler.ExecuteAivUbAllocRound(commitCnt), SUCCESS);
    EXPECT_FALSE(scheduler.state_.continueAllocStage);
    EXPECT_TRUE(queue0.Empty());
    EXPECT_TRUE(queue1.Empty());
}

// 验证一对 DualDst alloc 可从任一侧触发联合分配；
// 一侧已退休后从另一侧再次查询，也应识别为已完成而非报错。
TEST_F(ScheduleOoOTest, DualDst_AllocateDualDstAtCurrent_HappyPath)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);
    s.SetEnableDualDst(true);
    EXPECT_EQ(s.dualDstEngine_.RunDualDstFuse(s.state_.orderedOps), SUCCESS);

    Operation* dual = nullptr;
    for (auto& op : g.func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_L0C_COPY_UB_DUAL_DST) {
            dual = &op;
            break;
        }
    }
    ASSERT_NE(dual, nullptr);
    Operation* survivingUbAlloc = nullptr;
    for (auto* pred : s.state_.depManager.GetPredecessors(dual)) {
        if (pred != nullptr && pred->GetOpcodeStr().find("UB_ALLOC") != std::string::npos) {
            survivingUbAlloc = pred;
            break;
        }
    }
    ASSERT_NE(survivingUbAlloc, nullptr);

    int memIdA = survivingUbAlloc->GetOutputOperand(0)->memoryrange.memId;
    Operation* pairedAlloc = s.state_.schedInfoMap[survivingUbAlloc].pairedDualDstAlloc;
    ASSERT_NE(pairedAlloc, nullptr);
    int memIdB = pairedAlloc->GetOutputOperand(0)->memoryrange.memId;
    ASSERT_NE(s.state_.localBufferMap.find(memIdA), s.state_.localBufferMap.end());
    ASSERT_NE(s.state_.localBufferMap.find(memIdB), s.state_.localBufferMap.end());

    bool allocated = false;
    EXPECT_EQ(s.dualDstEngine_.AllocateDualDstAtCurrent(survivingUbAlloc, allocated), SUCCESS);
    EXPECT_TRUE(allocated);

    s.state_.schedInfoMap[survivingUbAlloc].isRetired = true;
    allocated = false;
    EXPECT_EQ(s.dualDstEngine_.AllocateDualDstAtCurrent(pairedAlloc, allocated), SUCCESS);
    EXPECT_TRUE(allocated);
}

// 验证底层 SelectSpillBuffers 只从触发 alloc 所在的单个 AIV pool 选取候选，
// DualDst 两侧联合 spill 由更上层 SpillOnBlock/GenBufferSpill 协调。
TEST_F(ScheduleOoOTest, DualDst_SelectSpillBuffers_UsesOnlyTriggerAllocPool)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    dualdst_ut::UpdateCopyDynValidShape(g);

    OoOScheduler s(*g.func);
    ASSERT_EQ(dualdst_ut::InitDualDstScheduler(s, g), SUCCESS);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);
    s.SetEnableDualDst(true);
    EXPECT_EQ(s.dualDstEngine_.RunDualDstFuse(s.state_.orderedOps), SUCCESS);

    Operation* dual = dualdst_ut::FindDualDstOp(*g.func);
    ASSERT_NE(dual, nullptr);
    Operation* survivingUbAlloc = dualdst_ut::FindUbAllocPred(s, dual);
    ASSERT_NE(survivingUbAlloc, nullptr);
    ASSERT_TRUE(s.state_.IsDualDstAlloc(survivingUbAlloc));

    int memIdA = survivingUbAlloc->GetOutputOperand(0)->memoryrange.memId;
    ASSERT_NE(s.state_.localBufferMap.find(memIdA), s.state_.localBufferMap.end());
    size_t needSize = s.state_.localBufferMap[memIdA]->size;

    constexpr int kPlaceholderMemIdA = 90001;
    constexpr int kPlaceholderMemIdB = 90002;
    ASSERT_EQ(dualdst_ut::FillAivPoolsWithPlaceholderBuffers(s, g, needSize, kPlaceholderMemIdA, kPlaceholderMemIdB),
              SUCCESS);

    auto spillGroup = s.SelectSpillBuffers(survivingUbAlloc);
    ASSERT_EQ(spillGroup.size(), 1u);
    const int expectedMemId = s.state_.schedInfoMap[survivingUbAlloc].coreLocation == CoreLocationType::AIV0 ?
                                  kPlaceholderMemIdA :
                                  kPlaceholderMemIdB;
    EXPECT_EQ(spillGroup[0], expectedMemId);
}

TEST_F(ScheduleOoOTest, DualDst_SelectSpillBuffers_EmptyPoolsReturnEmpty)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);
    s.SetEnableDualDst(true);
    EXPECT_EQ(s.dualDstEngine_.RunDualDstFuse(s.state_.orderedOps), SUCCESS);

    Operation* dual = nullptr;
    for (auto& op : g.func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_L0C_COPY_UB_DUAL_DST) {
            dual = &op;
            break;
        }
    }
    ASSERT_NE(dual, nullptr);
    Operation* survivingUbAlloc = nullptr;
    for (auto* pred : s.state_.depManager.GetPredecessors(dual)) {
        if (pred != nullptr && pred->GetOpcodeStr().find("UB_ALLOC") != std::string::npos) {
            survivingUbAlloc = pred;
            break;
        }
    }
    ASSERT_NE(survivingUbAlloc, nullptr);
    ASSERT_TRUE(s.state_.IsDualDstAlloc(survivingUbAlloc));

    auto spillGroup = s.SelectSpillBuffers(survivingUbAlloc);
    EXPECT_TRUE(spillGroup.empty());
}

// 直接验证 DualDst 边界上的 SpillOnBlock：一次调用应分别 spill AIV0/AIV1，
// 清除两侧占用记录，并保持两个 UB pool 的切片布局一致。
TEST_F(ScheduleOoOTest, DualDst_SpillOnBlockOnceSpillsBothAivPools)
{
    auto graph = dualdst_ut::BuildSpillTestGraph();
    ASSERT_NE(graph.function, nullptr);
    ASSERT_NE(graph.liveAlloc0, nullptr);
    ASSERT_NE(graph.liveAlloc1, nullptr);
    ASSERT_NE(graph.needAlloc0, nullptr);
    ASSERT_NE(graph.needAlloc1, nullptr);
    OoOScheduler scheduler(*graph.function);
    ASSERT_EQ(scheduler.Init(graph.function->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    scheduler.SetEnableDualDst(true);
    ASSERT_EQ(dualdst_ut::ConfigureSpillTestState(scheduler, graph), SUCCESS);

    int liveMemId0 = graph.builder->GetTensor("live0")->memoryrange.memId;
    int liveMemId1 = graph.builder->GetTensor("live1")->memoryrange.memId;
    auto& queue0 = scheduler.state_.allocIssueQueue[CoreLocationType::AIV0][MemoryType::MEM_UB];
    auto& queue1 = scheduler.state_.allocIssueQueue[CoreLocationType::AIV1][MemoryType::MEM_UB];
    auto& pool0 = scheduler.state_.bufferManagerMap[CoreLocationType::AIV0][MemoryType::MEM_UB];
    auto& pool1 = scheduler.state_.bufferManagerMap[CoreLocationType::AIV1][MemoryType::MEM_UB];
    ASSERT_EQ(queue0.Front(), graph.needAlloc0);
    ASSERT_EQ(queue1.Front(), graph.needAlloc1);
    ASSERT_TRUE(pool0.isAllocate(liveMemId0));
    ASSERT_TRUE(pool1.isAllocate(liveMemId1));
    ASSERT_EQ(pool0.GetBufferOffset(liveMemId0), pool1.GetBufferOffset(liveMemId1));

    dualdst_ut::DualDstSpillObserver observer;
    scheduler.AddObserver(&observer);
    ASSERT_EQ(scheduler.SpillOnBlock(), SUCCESS);

    dualdst_ut::ExpectTwoAivSpills(observer, liveMemId0, liveMemId1);
    EXPECT_FALSE(pool0.isAllocate(liveMemId0));
    EXPECT_FALSE(pool1.isAllocate(liveMemId1));
    EXPECT_EQ(scheduler.state_.tensorOccupyMap.count(liveMemId0), 0u);
    EXPECT_EQ(scheduler.state_.tensorOccupyMap.count(liveMemId1), 0u);
    EXPECT_EQ(scheduler.CheckAivUbPoolSlicesEqual(), SUCCESS);
    EXPECT_NE(std::find(queue0.queue.begin(), queue0.queue.end(), graph.needAlloc0), queue0.queue.end());
    EXPECT_NE(std::find(queue1.queue.begin(), queue1.queue.end(), graph.needAlloc1), queue1.queue.end());
    EXPECT_FALSE(scheduler.state_.schedInfoMap[graph.needAlloc0].isRetired);
    EXPECT_FALSE(scheduler.state_.schedInfoMap[graph.needAlloc1].isRetired);
}

// 验证两个 AIV UB pool 的 slice 数量或 offset 不一致时拒绝 DualDst alloc。
TEST_F(ScheduleOoOTest, DualDst_CheckAivUbPoolSlicesEqual_RejectsDivergentSlices)
{
    auto graph = dualdst_ut::BuildSpillTestGraph();
    ASSERT_NE(graph.function, nullptr);

    OoOScheduler scheduler(*graph.function);
    ASSERT_EQ(scheduler.Init(graph.function->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);

    int memId0 = graph.builder->GetTensor("live0")->memoryrange.memId;
    int memId1 = graph.builder->GetTensor("live1")->memoryrange.memId;
    auto buffer0 = scheduler.state_.localBufferMap.at(memId0);
    auto buffer1 = scheduler.state_.localBufferMap.at(memId1);
    auto& pool0 = scheduler.state_.bufferManagerMap[CoreLocationType::AIV0][MemoryType::MEM_UB];
    auto& pool1 = scheduler.state_.bufferManagerMap[CoreLocationType::AIV1][MemoryType::MEM_UB];

    // slice 数量不同。
    ASSERT_EQ(pool0.AllocateAtOffset(buffer0, 0), SUCCESS);
    EXPECT_EQ(scheduler.CheckAivUbPoolSlicesEqual(), FAILED);
    ASSERT_EQ(pool0.Free(memId0), SUCCESS);

    // slice 数量相同，但 offset 不同。
    ASSERT_EQ(pool0.AllocateAtOffset(buffer0, 0), SUCCESS);
    ASSERT_EQ(pool1.AllocateAtOffset(buffer1, buffer0->size), SUCCESS);
    EXPECT_EQ(scheduler.CheckAivUbPoolSlicesEqual(), FAILED);
}

// 验证非 MainLoop 的 DualDst spill：一次 GenBufferSpill 调用应联合选择
// AIV0/AIV1 的 spill 对象，并通过 ApplyDualSpill 释放两侧对应 buffer。
TEST_F(ScheduleOoOTest, DualDst_GenBufferSpill_NonMainLoopSpillsBothAivPools)
{
    auto graph = dualdst_ut::BuildSpillTestGraph();
    ASSERT_NE(graph.function, nullptr);
    ASSERT_NE(graph.needAlloc0, nullptr);
    ASSERT_NE(graph.needAlloc1, nullptr);

    OoOScheduler scheduler(*graph.function);
    ASSERT_EQ(scheduler.Init(graph.function->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    scheduler.SetEnableDualDst(true);
    ASSERT_EQ(dualdst_ut::ConfigureSpillTestState(scheduler, graph), SUCCESS);

    ASSERT_TRUE(scheduler.state_.IsDualDstAlloc(graph.needAlloc0));
    ASSERT_TRUE(scheduler.state_.IsDualDstAlloc(graph.needAlloc1));
    ASSERT_EQ(scheduler.state_.schedInfoMap[graph.needAlloc0].pairedDualDstAlloc, graph.needAlloc1);
    ASSERT_EQ(scheduler.state_.schedInfoMap[graph.needAlloc1].pairedDualDstAlloc, graph.needAlloc0);

    int liveMemId0 = graph.builder->GetTensor("live0")->memoryrange.memId;
    int liveMemId1 = graph.builder->GetTensor("live1")->memoryrange.memId;
    auto& pool0 = scheduler.state_.bufferManagerMap[CoreLocationType::AIV0][MemoryType::MEM_UB];
    auto& pool1 = scheduler.state_.bufferManagerMap[CoreLocationType::AIV1][MemoryType::MEM_UB];
    ASSERT_TRUE(pool0.isAllocate(liveMemId0));
    ASSERT_TRUE(pool1.isAllocate(liveMemId1));
    ASSERT_EQ(pool0.GetBufferOffset(liveMemId0), pool1.GetBufferOffset(liveMemId1));

    dualdst_ut::DualDstSpillObserver observer;
    scheduler.AddObserver(&observer);

    SpillContext ctx;
    ASSERT_EQ(scheduler.GenBufferSpill(graph.needAlloc0, ctx, /*isMainLoop=*/false), SUCCESS);

    dualdst_ut::ExpectTwoAivSpills(observer, liveMemId0, liveMemId1);
    EXPECT_FALSE(pool0.isAllocate(liveMemId0));
    EXPECT_FALSE(pool1.isAllocate(liveMemId1));
    EXPECT_EQ(scheduler.state_.tensorOccupyMap.count(liveMemId0), 0u);
    EXPECT_EQ(scheduler.state_.tensorOccupyMap.count(liveMemId1), 0u);
    EXPECT_EQ(ctx.spillMemIds, std::vector<int>{liveMemId0});
}

// 验证 MainLoop 的 continueAllocStage 分段下发：
// AIV0/AIV1 的 UB alloc queue 中普通 alloc 在前、DualDst alloc 在后。
// 普通 alloc 在第一轮 BufferAllocStage 下发后，遇 DualDst boundary 停止，
// DualDst alloc 被推到下一轮 BufferAllocStage 才下发。
// 判定依据：同一轮 BufferAllocStage 下发的 alloc 会在 LaunchIssueStage 之前连续提交，
// 若 DualDst alloc 出现在 COPY_IN（LaunchIssueStage 产物）之后，则证明被推到了下一轮。
TEST_F(ScheduleOoOTest, DualDst_MainLoopReusesRegularAllocReleasedInPreviousRound)
{
    dualdst_ut::MainLoopReuseGraph graph;
    ASSERT_TRUE(dualdst_ut::BuildMainLoopReuseGraph(graph));
    ASSERT_NE(graph.releaseAlloc0, nullptr);
    ASSERT_NE(graph.releaseAlloc1, nullptr);
    ASSERT_NE(graph.dualTensor0, nullptr);
    ASSERT_NE(graph.dualTensor1, nullptr);
    OoOScheduler scheduler(*graph.dualDst.func);
    ASSERT_EQ(scheduler.Init(dualdst_ut::BuildMainLoopReuseOpList(graph), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::ConfigureMainLoopReuseCores(scheduler, graph);
    scheduler.SetEnableDualDst(true);
    ASSERT_EQ(scheduler.dualDstEngine_.RunDualDstFuse(scheduler.state_.orderedOps), SUCCESS);
    ASSERT_TRUE(scheduler.state_.IsDualDstAlloc(graph.dualDst.allocUb0));
    ASSERT_TRUE(scheduler.state_.IsDualDstAlloc(graph.dualDst.allocUb1));
    dualdst_ut::DualDstSpillObserver observer;
    scheduler.AddObserver(&observer);
    ASSERT_EQ(scheduler.ScheduleMainLoop(), SUCCESS);
    EXPECT_TRUE(observer.events.empty());
    dualdst_ut::ExpectMainLoopReuse(scheduler, graph);
}

// 验证双池 spill 选择可找到相同起始地址的候选区间，
// 并在结果中分别保留 AIV0 和 AIV1 对应的 memId。
TEST_F(ScheduleOoOTest, DualDst_GetDualSpillGroup_FindsSharedStartAddrCandidate)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);

    auto& poolA = s.state_.bufferManagerMap[CoreLocationType::AIV0][MemoryType::MEM_UB];
    auto& poolB = s.state_.bufferManagerMap[CoreLocationType::AIV1][MemoryType::MEM_UB];

    constexpr int kBufMemIdA = 80001;
    constexpr int kBufMemIdB = 80002;
    constexpr size_t kBufSize = 1024;
    constexpr size_t kNeedSize = 512;
    ASSERT_GE(poolA.GetMemSize(), kBufSize);
    ASSERT_GE(poolB.GetMemSize(), kBufSize);

    auto bufA = std::make_shared<LocalBuffer>(kBufMemIdA, kBufSize, 0, MemoryType::MEM_UB);
    auto bufB = std::make_shared<LocalBuffer>(kBufMemIdB, kBufSize, 0, MemoryType::MEM_UB);
    ASSERT_EQ(poolA.AllocateAtOffset(bufA, 0), SUCCESS);
    ASSERT_EQ(poolB.AllocateAtOffset(bufB, 0), SUCCESS);

    auto groups = s.GetDualSpillGroup(poolA, poolB, kNeedSize);
    ASSERT_EQ(groups.size(), 1u);
    EXPECT_EQ(groups[0].aiv0MemIds, std::vector<int>({kBufMemIdA}));
    EXPECT_EQ(groups[0].aiv1MemIds, std::vector<int>({kBufMemIdB}));
}

TEST_F(ScheduleOoOTest, DualDst_GetDualSpillGroup_NeedSizeExceedsPoolReturnsEmpty)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);

    auto& poolA = s.state_.bufferManagerMap[CoreLocationType::AIV0][MemoryType::MEM_UB];
    auto& poolB = s.state_.bufferManagerMap[CoreLocationType::AIV1][MemoryType::MEM_UB];

    constexpr int kBufMemIdA = 80003;
    constexpr int kBufMemIdB = 80004;
    auto bufA = std::make_shared<LocalBuffer>(kBufMemIdA, 1024, 0, MemoryType::MEM_UB);
    auto bufB = std::make_shared<LocalBuffer>(kBufMemIdB, 1024, 0, MemoryType::MEM_UB);
    ASSERT_EQ(poolA.AllocateAtOffset(bufA, 0), SUCCESS);
    ASSERT_EQ(poolB.AllocateAtOffset(bufB, 0), SUCCESS);

    size_t needSize = poolA.GetMemSize() + 1024;
    auto groups = s.GetDualSpillGroup(poolA, poolB, needSize);
    EXPECT_TRUE(groups.empty());
}

TEST_F(ScheduleOoOTest, DualDst_InferDynShape_RecordsStaticValidShape)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    InferDynShape pass;
    pass.RecordStaticValidShapeOnL0CCopyUB(*g.func);

    EXPECT_TRUE(g.copy0->HasAttribute(OpAttributeKey::staticValidShape));
    EXPECT_TRUE(g.copy1->HasAttribute(OpAttributeKey::staticValidShape));
    auto v0 = g.copy0->GetVectorIntAttribute<int64_t>(OpAttributeKey::staticValidShape);
    EXPECT_EQ(v0.size(), 2u);
    EXPECT_EQ(v0[0], dualdst_ut::TILE_M);
    EXPECT_EQ(v0[1], dualdst_ut::TILE_N);
}

TEST_F(ScheduleOoOTest, DualDst_InferDynShape_SkipsDynamicValidShape)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape({SymbolicScalar("dyn0"), SymbolicScalar(dualdst_ut::TILE_N)});

    InferDynShape pass;
    pass.RecordStaticValidShapeOnL0CCopyUB(*g.func);

    EXPECT_FALSE(g.copy0->HasAttribute(OpAttributeKey::staticValidShape));
}

TEST_F(ScheduleOoOTest, DualDst_GetNewOperations_DedupePreservesFirstOccurrence)
{
    auto g = dualdst_ut::BuildDualDstGraph_2({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                             {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);

    s.state_.newOperations.clear();
    s.state_.newOperations.push_back(g.copy0);
    s.state_.newOperations.push_back(g.copy1);
    s.state_.newOperations.push_back(g.copy0); // dup
    s.state_.newOperations.push_back(nullptr); // null

    auto uniq = s.GetNewOperations();
    EXPECT_EQ(uniq.size(), 2u);
    EXPECT_EQ(uniq[0], g.copy0);
    EXPECT_EQ(uniq[1], g.copy1);
}

TEST_F(ScheduleOoOTest, IsoMatch_IsoMatchChains_RootSignatureMismatch)
{
    auto builder = std::make_shared<ComputationalGraphBuilder>();
    EXPECT_EQ(builder->AddTensor(DataType::DT_FP32, {4, 4}, MemoryType::MEM_UB, "ta0"), true);
    EXPECT_EQ(builder->AddTensor(DataType::DT_FP32, {4, 4}, MemoryType::MEM_UB, "tb0"), true);
    EXPECT_EQ(builder->AddTensor(DataType::DT_FP32, {4, 4}, MemoryType::MEM_UB, "ta_out"), true);
    EXPECT_EQ(builder->AddTensor(DataType::DT_FP32, {4, 4}, MemoryType::MEM_UB, "tb_out"), true);
    EXPECT_EQ(builder->AddOp(Opcode::OP_UB_ALLOC, {}, {"ta0"}, "alloc_a0"), true);
    EXPECT_EQ(builder->AddOp(Opcode::OP_UB_ALLOC, {}, {"tb0"}, "alloc_b0"), true);
    EXPECT_EQ(builder->AddOp(Opcode::OP_ADD, {"ta0", "ta0"}, {"ta_out"}, "add_a0"), true);
    EXPECT_EQ(builder->AddOp(Opcode::OP_SUB, {"tb0", "tb0"}, {"tb_out"}, "sub_b0"), true);

    std::vector<Operation*> opsA = {builder->GetOp("alloc_a0"), builder->GetOp("add_a0")};
    std::vector<Operation*> opsB = {builder->GetOp("alloc_b0"), builder->GetOp("sub_b0")};
    std::unordered_set<Operation*> setA(opsA.begin(), opsA.end());
    std::unordered_set<Operation*> setB(opsB.begin(), opsB.end());

    auto rootsA = FindTaskEntryOps(opsA, setA);
    auto rootsB = FindTaskEntryOps(opsB, setB);
    auto res = IsoMatchChains(rootsA, rootsB, setA, setB);
    EXPECT_FALSE(res.rootIsomorphic);
    EXPECT_EQ(res.pairs.size(), 0u);
}

// 验证 AIV0-AIC-AIV1 混合 task 的同构匹配只收集两侧 UB_ALLOC，
// L1_ALLOC/L0C_ALLOC 不参与地址对齐，非 alloc 的同构 op 仍正常配对。
TEST_F(ScheduleOoOTest, IsoMatch_MixedAivAicTask_CollectsOnlyUbAllocPairs)
{
    auto graph = dualdst_ut::BuildMixedIsoGraph();
    ASSERT_NE(graph.allocUbA, nullptr);
    ASSERT_NE(graph.allocUbB, nullptr);
    ASSERT_NE(graph.allocL1A, nullptr);
    ASSERT_NE(graph.allocL1B, nullptr);
    ASSERT_NE(graph.copyL1A, nullptr);
    ASSERT_NE(graph.copyL1B, nullptr);
    ASSERT_NE(graph.allocL0c, nullptr);
    std::unordered_set<Operation*> setA(graph.opsA.begin(), graph.opsA.end());
    std::unordered_set<Operation*> setB(graph.opsB.begin(), graph.opsB.end());
    auto result = IsoMatchChains(FindTaskEntryOps(graph.opsA, setA), FindTaskEntryOps(graph.opsB, setB), setA, setB);

    ASSERT_TRUE(result.rootIsomorphic);
    ASSERT_EQ(result.allocPairs.size(), 1u);
    EXPECT_EQ(result.allocPairs[0].opA, graph.allocUbA);
    EXPECT_EQ(result.allocPairs[0].opB, graph.allocUbB);
    EXPECT_TRUE(std::none_of(result.allocPairs.begin(), result.allocPairs.end(), [](const IsoPair& pair) {
        return pair.opA->GetOpcode() == Opcode::OP_L1_ALLOC || pair.opB->GetOpcode() == Opcode::OP_L1_ALLOC;
    }));
    ASSERT_EQ(result.pairs.size(), 1u);
    EXPECT_EQ(result.pairs[0].opA, graph.copyL1A);
    EXPECT_EQ(result.pairs[0].opB, graph.copyL1B);

    TaskSplitter splitter;
    dualdst_ut::BuildMixedIsoTasks(splitter, graph);
    OoOSchedule schedule;
    ASSERT_TRUE(schedule.ShouldEnableDualDst(splitter));
    ASSERT_TRUE(dualdst_ut::HasIsoAllocPair(schedule.dualDstPairs_, graph.allocUbA, graph.allocUbB));
    EXPECT_FALSE(dualdst_ut::HasIsoAllocPair(schedule.dualDstPairs_, graph.allocL1A, graph.allocL1B));
    EXPECT_TRUE(dualdst_ut::HasIsoAllocPair(schedule.dualDstOpPairs_, graph.copyL1A, graph.copyL1B));
}

// 两侧各有 2 个同签名入口(ub_alloc)时,旧逻辑在 depth=0 判为歧义、截断不下探,
// 导致下游 op 配不齐而关闭 dualdst。新逻辑靠下游子图 hash 消歧后,应完整配对。
TEST_F(ScheduleOoOTest, IsoMatch_MultiEntrySameSig_DisambiguatesBySubgraphHash)
{
    auto graph = dualdst_ut::BuildMultiEntryIsoGraph();
    std::unordered_set<Operation*> setA(graph.opsA.begin(), graph.opsA.end());
    std::unordered_set<Operation*> setB(graph.opsB.begin(), graph.opsB.end());
    auto result = IsoMatchChains(FindTaskEntryOps(graph.opsA, setA), FindTaskEntryOps(graph.opsB, setB), setA, setB);

    ASSERT_TRUE(result.rootIsomorphic);
    auto hasPair = [](const std::vector<IsoPair>& pairs, Operation* opA, Operation* opB) {
        return std::any_of(pairs.begin(), pairs.end(),
                           [opA, opB](const IsoPair& pair) { return pair.opA == opA && pair.opB == opB; });
    };
    // 两条 alloc->add->tail 链都应完整配对:2 个 alloc pair + 4 个非 alloc pair。
    EXPECT_EQ(result.allocPairs.size(), 2u);
    EXPECT_EQ(result.pairs.size(), 4u);
    EXPECT_TRUE(hasPair(result.allocPairs, graph.builder->GetOp("alloc_a0"), graph.builder->GetOp("alloc_b0")));
    EXPECT_TRUE(hasPair(result.allocPairs, graph.builder->GetOp("alloc_a1"), graph.builder->GetOp("alloc_b1")));
    EXPECT_TRUE(hasPair(result.pairs, graph.builder->GetOp("tail_a0"), graph.builder->GetOp("tail_b0")));
    EXPECT_TRUE(hasPair(result.pairs, graph.builder->GetOp("tail_a1"), graph.builder->GetOp("tail_b1")));
    // 确认确实走了多候选分支(两个同签名入口),而非碰巧单候选。
    EXPECT_GT(result.truncatedCount, 0u);
}

// === upstream/master: ModifyAllocOrder tests ===

static int IndexOf(const std::vector<Operation*>& opList, Operation* op)
{
    for (size_t i = 0; i < opList.size(); i++) {
        if (opList[i] == op) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

TEST_F(ScheduleOoOTest, TestModifyAllocOrderPushesEarlyAllocDown)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> names{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<MemoryType> memTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR,
                                     MemoryType::MEM_UB,         MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, memTypes, names, 0), true);
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,
                                Opcode::OP_COPY_IN,  Opcode::OP_ADD,      Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ins{{}, {}, {}, {"t1"}, {"t3"}, {"t2", "t4"}, {"t5"}};
    std::vector<std::vector<std::string>> outs{{"t2"}, {"t4"}, {"t5"}, {"t2"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"AllocT2", "AllocT4", "AllocT5", "Copyin1", "Copyin2", "Add1", "Copyout1"};
    EXPECT_EQ(subGraph.AddOps(opCodes, ins, outs, opNames, true), true);
    Function* function = subGraph.GetFunction();
    ASSERT_NE(function, nullptr);

    std::vector<Operation*> opList{subGraph.GetOp("AllocT2"), subGraph.GetOp("AllocT4"), subGraph.GetOp("AllocT5"),
                                   subGraph.GetOp("Copyin1"), subGraph.GetOp("Copyin2"), subGraph.GetOp("Add1"),
                                   subGraph.GetOp("Copyout1")};
    EXPECT_EQ(IndexOf(opList, subGraph.GetOp("Copyin1")) - IndexOf(opList, subGraph.GetOp("AllocT2")), 3);

    OoOSchedule oooSchedule;
    EXPECT_EQ(oooSchedule.ModifyAllocOrder(opList), SUCCESS);

    EXPECT_EQ(opList.size(), 7u);
    EXPECT_EQ(IndexOf(opList, subGraph.GetOp("Copyin1")) - IndexOf(opList, subGraph.GetOp("AllocT2")), 1);
    EXPECT_EQ(IndexOf(opList, subGraph.GetOp("Copyin2")) - IndexOf(opList, subGraph.GetOp("AllocT4")), 1);
    EXPECT_EQ(IndexOf(opList, subGraph.GetOp("Add1")) - IndexOf(opList, subGraph.GetOp("AllocT5")), 1);
}

TEST_F(ScheduleOoOTest, TestModifyAllocOrderPullsLateAllocUp)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> names{"t1", "t2", "t3"};
    std::vector<MemoryType> memTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, memTypes, names, 0), true);
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ins{{"t1"}, {}, {"t2"}};
    std::vector<std::vector<std::string>> outs{{"t2"}, {"t2"}, {"t3"}};
    std::vector<std::string> opNames{"Copyin1", "AllocT2", "Copyout1"};
    EXPECT_EQ(subGraph.AddOps(opCodes, ins, outs, opNames, true), true);
    Function* function = subGraph.GetFunction();
    ASSERT_NE(function, nullptr);

    std::vector<Operation*> opList{subGraph.GetOp("Copyin1"), subGraph.GetOp("AllocT2"), subGraph.GetOp("Copyout1")};
    OoOSchedule oooSchedule;
    EXPECT_EQ(oooSchedule.ModifyAllocOrder(opList), SUCCESS);

    EXPECT_EQ(opList.size(), 3u);
    EXPECT_EQ(IndexOf(opList, subGraph.GetOp("AllocT2")), 0);
    EXPECT_EQ(IndexOf(opList, subGraph.GetOp("Copyin1")), 1);
}

TEST_F(ScheduleOoOTest, TestModifyAllocOrderKeepsUnreferencedAllocInPlace)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> names{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> memTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR,
                                     MemoryType::MEM_UB};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, memTypes, names, 0), true);
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ins{{}, {}, {"t1"}, {"t2"}};
    std::vector<std::vector<std::string>> outs{{"t4"}, {"t2"}, {"t2"}, {"t3"}};
    std::vector<std::string> opNames{"AllocT4", "AllocT2", "Copyin1", "Copyout1"};
    EXPECT_EQ(subGraph.AddOps(opCodes, ins, outs, opNames, true), true);
    Function* function = subGraph.GetFunction();
    ASSERT_NE(function, nullptr);

    std::vector<Operation*> opList{subGraph.GetOp("AllocT4"), subGraph.GetOp("AllocT2"), subGraph.GetOp("Copyin1"),
                                   subGraph.GetOp("Copyout1")};
    OoOSchedule oooSchedule;
    EXPECT_EQ(oooSchedule.ModifyAllocOrder(opList), SUCCESS);

    EXPECT_EQ(opList.size(), 4u);
    EXPECT_EQ(IndexOf(opList, subGraph.GetOp("AllocT4")), 0);
    EXPECT_EQ(IndexOf(opList, subGraph.GetOp("Copyin1")) - IndexOf(opList, subGraph.GetOp("AllocT2")), 1);
}

TEST_F(ScheduleOoOTest, TestModifyAllocOrderInsertsAllocOnceForInPlaceOp)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> names{"t1", "t2", "t3"};
    std::vector<MemoryType> memTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, memTypes, names, 0), true);
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_ADD, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ins{{}, {"t1"}, {"t2", "t2"}, {"t2"}};
    std::vector<std::vector<std::string>> outs{{"t2"}, {"t2"}, {"t2"}, {"t3"}};
    std::vector<std::string> opNames{"AllocT2", "Copyin1", "AddInPlace", "Copyout1"};
    EXPECT_EQ(subGraph.AddOps(opCodes, ins, outs, opNames, true), true);
    Function* function = subGraph.GetFunction();
    ASSERT_NE(function, nullptr);

    std::vector<Operation*> opList{subGraph.GetOp("AllocT2"), subGraph.GetOp("AddInPlace"), subGraph.GetOp("Copyout1")};
    OoOSchedule oooSchedule;
    EXPECT_EQ(oooSchedule.ModifyAllocOrder(opList), SUCCESS);

    EXPECT_EQ(opList.size(), 3u);
    EXPECT_EQ(IndexOf(opList, subGraph.GetOp("AllocT2")), 0);
    EXPECT_EQ(IndexOf(opList, subGraph.GetOp("AddInPlace")), 1);
}

// === vecSortAlgo: OptimizeSort factory interface tests ===

// 用按 mode 实例化正确的 sorter 子类。
TEST_F(ScheduleOoOTest, OptimizeSortCreateFactory)
{
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestParams", "TestParams", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestCreate", "TestCreate",
                                                      rootFuncPtr.get());
    ASSERT_NE(currFunctionPtr, nullptr);

    std::vector<Operation*> emptyOps;
    auto priordfs = OptimizeSort::Create(emptyOps, *currFunctionPtr, "PriorDFS");
    EXPECT_NE(priordfs, nullptr);
    EXPECT_NE(dynamic_cast<PriorDFSSort*>(priordfs.get()), nullptr);

    auto clusterList = OptimizeSort::Create(emptyOps, *currFunctionPtr, "ClusterList");
    EXPECT_NE(clusterList, nullptr);
    EXPECT_NE(dynamic_cast<ClusterListSort*>(clusterList.get()), nullptr);

    auto unknown = OptimizeSort::Create(emptyOps, *currFunctionPtr, "UnknownMode");
    EXPECT_EQ(unknown, nullptr);
}

// SortOps 端到端——oooSortModeAiv="ClusterList" + 纯 AIV 图 → SUCCESS。
TEST_F(ScheduleOoOTest, SortOpsClusterListMode)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_ADD,
                                Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {"t1"}, {"t2"}, {"t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Copyin1", "Add1", "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    ASSERT_NE(function, nullptr);
    function->paramConfigs_.oooSortModeAiv = "ClusterList";

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    EXPECT_EQ(sort.SortOps(), SUCCESS);
    auto result = sort.GetOperations();
    EXPECT_EQ(result.size(), 5U);
    Operation* copyin = subGraph.GetOp("Copyin1");
    Operation* add = subGraph.GetOp("Add1");
    Operation* copyout = subGraph.GetOp("Copyout1");
    auto pos = [&](Operation* target) {
        for (size_t i = 0; i < result.size(); ++i) {
            if (result[i] == target) {
                return i;
            }
        }
        return result.size();
    };
    EXPECT_LT(pos(copyin), pos(add));
    EXPECT_LT(pos(add), pos(copyout));
}

// SortOps 端到端——oooSortModeAiv="" + 纯 AIV 图 → SUCCESS（默认走 PriorDFS）。
TEST_F(ScheduleOoOTest, SortOpsDefaultPriorDFSMode)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_ADD,
                                Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {"t1"}, {"t2"}, {"t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Copyin1", "Add1", "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    ASSERT_NE(function, nullptr);
    function->paramConfigs_.oooSortModeAiv = "";

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    EXPECT_EQ(sort.SortOps(), SUCCESS);
    auto result = sort.GetOperations();
    EXPECT_EQ(result.size(), 5U);
}

// SortOps 端到端——空 operations → SUCCESS。
TEST_F(ScheduleOoOTest, SortOpsEmptyInput)
{
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestParams", "TestParams", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestEmpty", "TestEmpty",
                                                      rootFuncPtr.get());
    ASSERT_NE(currFunctionPtr, nullptr);

    std::vector<Operation*> emptyOps;
    OptimizeSort sort(emptyOps, *currFunctionPtr);
    EXPECT_EQ(sort.SortOps(), SUCCESS);
    EXPECT_TRUE(sort.GetOperations().empty());
}

// 参考真实场景 mla_prolog_quant Unroll128 的 kr_cache 量化写回(dump: op[38965/38968/38971/38974]),
// 逐边对照真实来源链做等比简化, 图中不存在悬空 tensor:
//   src 链(真实: incast->COPY_IN->CAST/MUL/ADD->RESHAPE->src)  简化为  incast(b0,b1) -> COPY_IN -> ADD -> src_i
//   idx 链(真实: incast[3919]->COPY_IN->idx[28797])            简化为  incast(kr_idx) -> COPY_IN -> idx_i
//   cache 链(真实: incast[3901] 直连 INDEX_OUTCAST 第3输入)     保持    incast(kr_cache) 直连
// 4 路 INDEX_OUTCAST(src_i, idx_i, kr_cache) -> kr_cache_out: 共享 kr_cache 并共写 kr_cache_out(WAW)
static void BuildKrCacheOutcastGraph(ComputationalGraphBuilder& subGraph, Function** function)
{
    // 函数输入 incast: b0/b1(量化数据), kr_idx(散写索引), kr_cache(写回基址)
    // 中间: cin0..3(add的COPY_IN输入) add0..3(src链) idx0..idx3(idx的COPY_IN输出)
    //       kr_cache_out: 函数输出(outcast)
    std::vector<std::string> tensorNames{"b0",   "b1",   "cin0",   "cin1",     "cin2",        "cin3",
                                         "add0", "add1", "add2",   "add3",     "idx0",        "idx1",
                                         "idx2", "idx3", "kr_idx", "kr_cache", "kr_cache_out"};
    std::vector<Opcode> opCodes{
        Opcode::OP_COPY_IN,       Opcode::OP_COPY_IN,       Opcode::OP_COPY_IN,       Opcode::OP_COPY_IN,
        Opcode::OP_ADD,           Opcode::OP_ADD,           Opcode::OP_ADD,           Opcode::OP_ADD,
        Opcode::OP_COPY_IN,       Opcode::OP_COPY_IN,       Opcode::OP_COPY_IN,       Opcode::OP_COPY_IN,
        Opcode::OP_INDEX_OUTCAST, Opcode::OP_INDEX_OUTCAST, Opcode::OP_INDEX_OUTCAST, Opcode::OP_INDEX_OUTCAST};
    std::vector<std::vector<std::string>> ioperands{{"b0"},
                                                    {"b1"},
                                                    {"b0"},
                                                    {"b1"},
                                                    {"cin0", "cin1"},
                                                    {"cin1", "cin2"},
                                                    {"cin2", "cin3"},
                                                    {"cin3", "cin0"},
                                                    {"kr_idx"},
                                                    {"kr_idx"},
                                                    {"kr_idx"},
                                                    {"kr_idx"},
                                                    {"add0", "idx0", "kr_cache"},
                                                    {"add1", "idx1", "kr_cache"},
                                                    {"add2", "idx2", "kr_cache"},
                                                    {"add3", "idx3", "kr_cache"}};
    std::vector<std::vector<std::string>> ooperands{
        {"cin0"}, {"cin1"}, {"cin2"}, {"cin3"}, {"add0"},         {"add1"},         {"add2"},         {"add3"},
        {"idx0"}, {"idx1"}, {"idx2"}, {"idx3"}, {"kr_cache_out"}, {"kr_cache_out"}, {"kr_cache_out"}, {"kr_cache_out"}};
    std::vector<std::string> opNames{"cpin0",    "cpin1",    "cpin2",    "cpin3",   "add0",  "add1",
                                     "add2",     "add3",     "cpin4",    "cpin5",   "cpin6", "cpin7",
                                     "outcast0", "outcast1", "outcast2", "outcast3"};
    // 内存布局(对照真实图): incast与outcast在DDR, COPY_IN输出与计算中间量在UB
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, // b0 (incast)
        MemoryType::MEM_DEVICE_DDR, // b1 (incast)
        MemoryType::MEM_UB,         // cin0
        MemoryType::MEM_UB,         // cin1
        MemoryType::MEM_UB,         // cin2
        MemoryType::MEM_UB,         // cin3
        MemoryType::MEM_UB,         // add0
        MemoryType::MEM_UB,         // add1
        MemoryType::MEM_UB,         // add2
        MemoryType::MEM_UB,         // add3
        MemoryType::MEM_UB,         // idx0
        MemoryType::MEM_UB,         // idx1
        MemoryType::MEM_UB,         // idx2
        MemoryType::MEM_UB,         // idx3
        MemoryType::MEM_DEVICE_DDR, // kr_idx (incast)
        MemoryType::MEM_DEVICE_DDR, // kr_cache (incast)
        MemoryType::MEM_DEVICE_DDR, // kr_cache_out (outcast)
    };
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {256, 256}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    // 函数IO(对照真实 dump: 数据 incast/3919/3901 在 incasts, 3943 在 outcasts)
    EXPECT_EQ(subGraph.SetInCast({"b0", "b1", "kr_idx", "kr_cache"}), true);
    EXPECT_EQ(subGraph.SetOutCast({"kr_cache_out"}), true);
    *function = subGraph.GetFunction();
    EXPECT_NE(*function, nullptr);
}

// 在 task 图中定位 op 所在 task, 未找到返回 -1
static int FindTaskIdOfOp(const TaskGraph& tasks, const Operation* op)
{
    for (size_t t = 0; t < tasks.tasks.size(); t++) {
        for (auto opPtr : tasks.tasks[t].opList_) {
            if (opPtr == op) {
                return static_cast<int>(t);
            }
        }
    }
    return -1;
}

TEST_F(ScheduleOoOTest, TaskSplitterUnionIndexOutcastOnSameOutcast)
{
    ComputationalGraphBuilder subGraph;
    Function* function = nullptr;
    BuildKrCacheOutcastGraph(subGraph, &function);
    // 反向对照: 第 5 路 INDEX_OUTCAST 写独立 outcast(不同 memId)且走独立计算链, 不应与共享组被误合并
    std::vector<std::string> extraNames{"cin4", "add4", "idx4", "kr_idx2", "kr_cache2", "kr_cache_out2"};
    std::vector<MemoryType> extraMemTypes{MemoryType::MEM_UB,         MemoryType::MEM_UB,
                                          MemoryType::MEM_UB,         MemoryType::MEM_DEVICE_DDR,
                                          MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {256, 256}, extraMemTypes, extraNames, 0), true);
    EXPECT_EQ(subGraph.AddOps({Opcode::OP_COPY_IN, Opcode::OP_ADD, Opcode::OP_COPY_IN, Opcode::OP_INDEX_OUTCAST},
                              {{"b0"}, {"cin4", "b1"}, {"kr_idx2"}, {"add4", "idx4", "kr_cache2"}},
                              {{"cin4"}, {"add4"}, {"idx4"}, {"kr_cache_out2"}}, {"cpin8", "add4", "cpin9", "outcast4"},
                              true),
              true);
    EXPECT_EQ(subGraph.SetOutCast({"kr_cache_out2"}), true);
    std::vector<Operation*> outcasts;
    for (const std::string& name : std::vector<std::string>{"outcast0", "outcast1", "outcast2", "outcast3"}) {
        auto op = subGraph.GetOp(name);
        ASSERT_NE(op, nullptr);
        outcasts.push_back(op);
    }
    auto outcast4 = subGraph.GetOp("outcast4");
    ASSERT_NE(outcast4, nullptr);

    auto opList = function->Operations(false).DuplicatedOpList();
    TaskSplitter splitter;
    splitter.SplitGraph(opList);
    // 验证共享同一 outcast 的 4 路 INDEX_OUTCAST 落入同一 task(修复前各自独立 cluster, 必不满足)
    int firstTaskId = -1;
    for (auto op : outcasts) {
        int taskId = FindTaskIdOfOp(splitter.GetTaskGraph(), op);
        ASSERT_NE(taskId, -1);
        if (firstTaskId == -1) {
            firstTaskId = taskId;
        }
        EXPECT_EQ(taskId, firstTaskId);
    }
    // 反向断言: 写不同 outcast 的 outcast4 不得与共享组合并(防分组键放宽导致过度合并)
    int task4Id = FindTaskIdOfOp(splitter.GetTaskGraph(), outcast4);
    ASSERT_NE(task4Id, -1);
    EXPECT_NE(task4Id, firstTaskId);
    // 同 task 即保证同核: MarkInternalSubgraphID 按 task 整体赋 AIVCore(纯 AIV 图下其 targetTypes>1
    // 前置约束不满足, 故此处不调用), task 内 op 的核一致性由 task 归属直接保证
}

// 图结构：
//   copy_in1 -> add1 -> add2 -> add3 -> copy_out1   (depth=4, 源点={copy_in1})
//   copy_in1 -> mul1 -> copy_out2                    (depth=2, 源点={copy_in1})
//   copy_in2 -> sub1 -> sub2 -> copy_out3            (depth=3, 源点={copy_in2})
//
// 三个出口节点：copy_out1(depth=4), copy_out2(depth=2), copy_out3(depth=3)
// copy_out1 和 copy_out2 同源（源点都是 copy_in1），copy_out3 不同源（源点是 copy_in2）
// 正确排序：copy_out1(4), copy_out2(2), copy_out3(3)  — 同源排一起，组间按最大深度降序
// 错误排序：copy_out1(4), copy_out3(3), copy_out2(2)  — 纯深度排序，同源被拆开
TEST_F(ScheduleOoOTest, PriorDFSSort_SortOutNodeQueue_SameSourceGrouped)
{
    ComputationalGraphBuilder b;
    // t0:DDR(输入)  t1:UB(copy_in1输出)  t2:UB(add1输出)  t3:UB(add2输出)  t4:UB(add3输出/copy_out1输入)
    // t5:UB(mul1输出/copy_out2输入)  t6:DDR(输入)  t7:UB(copy_in2输出)  t8:UB(sub1输出)  t9:UB(sub2输出/copy_out3输入)
    // t10:DDR(copy_out输出)
    std::vector<std::string> tensorNames{"t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10"};
    std::vector<MemoryType> memTypes{MEM_DEVICE_DDR, MEM_UB, MEM_UB, MEM_UB, MEM_UB,        MEM_UB,
                                     MEM_DEVICE_DDR, MEM_UB, MEM_UB, MEM_UB, MEM_DEVICE_DDR};
    EXPECT_TRUE(b.AddTensors(DT_FP32, {64, 64}, memTypes, tensorNames, 0));

    // 每个UB张量都需要一个alloc
    std::vector<Opcode> opcodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,
                                Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_COPY_OUT,
                                Opcode::OP_MUL,      Opcode::OP_COPY_OUT, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,
                                Opcode::OP_SUB,      Opcode::OP_SUB,      Opcode::OP_COPY_OUT};
    // Alloc1->t1, Alloc2->t2, Alloc3->t3, Alloc4->t4, Alloc5->t5, Alloc6->t7, Alloc7->t8
    // CopyIn1: t0->t1, Add1: t1->t2, Add2: t2->t3, Add3: t3->t4, CopyOut1: t4->t10
    // Mul1: t1->t5, CopyOut2: t5->t10
    // Alloc8->t7(已有Alloc6), 这里t9也需要alloc
    // CopyIn2: t6->t7, Sub1: t7->t8, Sub2: t8->t9, CopyOut3: t9->t10
    std::vector<std::vector<std::string>> ins{{},     {},     {},     {},     {},     {},     {},
                                              {"t0"}, {"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t1"}, {"t5"},
                                              {},     {"t6"}, {"t7"}, {"t8"}, {"t9"}};
    std::vector<std::vector<std::string>> outs{{"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t5"},  {"t7"}, {"t8"},
                                               {"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t10"}, {"t5"}, {"t10"},
                                               {"t9"}, {"t7"}, {"t8"}, {"t9"}, {"t10"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3", "Alloc4", "Alloc5",   "Alloc6", "Alloc7",
                                     "CopyIn1", "Add1",    "Add2",   "Add3",   "CopyOut1", "Mul1",   "CopyOut2",
                                     "Alloc8",  "CopyIn2", "Sub1",   "Sub2",   "CopyOut3"};
    EXPECT_TRUE(b.AddOps(opcodes, ins, outs, opNames, true));
    Function* func = b.GetFunction();
    ASSERT_NE(func, nullptr);

    auto ops = func->Operations().DuplicatedOpList();
    PriorDFSSort sorter(ops, *func);
    sorter.state_.Init(ops);

    std::vector<Operation*> outNodeQueue;
    sorter.depthCache_.clear();
    for (auto* op : ops) {
        if (sorter.state_.depManager.GetSuccessors(op).empty()) {
            outNodeQueue.emplace_back(op);
        }
    }
    ASSERT_EQ(outNodeQueue.size(), 3U);

    sorter.SortOutNodeQueue(outNodeQueue);

    EXPECT_EQ(outNodeQueue[0], b.GetOp("CopyOut1"));
    EXPECT_EQ(outNodeQueue[1], b.GetOp("CopyOut2"));
    EXPECT_EQ(outNodeQueue[2], b.GetOp("CopyOut3"));
}

} // namespace npu::tile_fwk
