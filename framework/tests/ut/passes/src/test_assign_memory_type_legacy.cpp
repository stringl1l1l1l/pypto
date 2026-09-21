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
 * \file test_assign_memory_type_legacy.cpp
 * \brief Unit test for assign_memory_type legacy pass (enable_slice=false).
 */

#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "interface/tensor/irbuilder.h"
#include "symbolic_scalar_test_utils.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "passes/tile_graph_pass/data_path/assign_memory_type.h"
#include "passes/pass_mgr/pass_manager.h"
#include "computational_graph_builder.h"
#include "interface/tensor/irbuilder.h"

using namespace npu::tile_fwk;

namespace npu {
namespace tile_fwk {
const int NUM_1 = 1;
const int NUM_2 = 2;
const int NUM_8 = 8;
const int NUM_16 = 16;
const int NUM_32 = 32;
const int NUM_48 = 48;
const int NUM_64 = 64;
const int NUM_128 = 128;
const int NUM_256 = 256;
const int NUM_512 = 512;
const int NUM_1024 = 1024;
constexpr float F_1 = 1.0;
constexpr float F_3 = 3.0;

class LegacyAssignMemoryTypeTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetPlatformConfig(KEY_TEST_IS_TIG, true);
        config::SetPassOption(ENABLE_SLICE, false);
    }
    void TearDown() override {}

    void SetHalfwayStrategy()
    {
        PassManager& passManager = PassManager::Instance();
        passManager.RegisterStrategy("AssignMemoryTypeTestStrategy",
                                     {
                                         {"RemoveRedundantReshape", PassName::REMOVE_REDUNDANT_RESHAPE},
                                         {"InferMemoryConflict", PassName::INFER_MEMORY_CONFLICT},
                                         {"ExpandFunction", PassName::EXPAND_FUNCTION},
                                         {"DuplicateOp", PassName::DUPLICATE_OP},
                                         {"MergeViewAssemble", PassName::MERGE_VIEW_ASSEMBLE},
                                     });
        ConfigManager::Instance();
    }

    void SetTestStrategy()
    {
        PassManager& passManager = PassManager::Instance();
        passManager.RegisterStrategy("AssignMemoryTypeTestStrategy",
                                     {
                                         {"InferMemoryConflict", PassName::INFER_MEMORY_CONFLICT},
                                         {"ExpandFunction", PassName::EXPAND_FUNCTION},
                                         {"AssignMemoryType", PassName::ASSIGN_MEMORY_TYPE},
                                     });
        ConfigManager::Instance();
    }

    void SetFullTestStrategy()
    {
        PassManager& passManager = PassManager::Instance();
        passManager.RegisterStrategy("AssignMemoryTypeTestStrategy",
                                     {
                                         {"RemoveRedundantReshape", PassName::REMOVE_REDUNDANT_RESHAPE},
                                         {"AutoCast", PassName::AUTO_CAST},
                                         {"InferMemoryConflict", PassName::INFER_MEMORY_CONFLICT},
                                         {"RemoveUndrivenView", PassName::REMOVE_UNDRIVEN_VIEW},
                                         {"ExpandFunction", PassName::EXPAND_FUNCTION},
                                         {"MergeViewAssemble", PassName::MERGE_VIEW_ASSEMBLE},
                                         {"SplitReshape", PassName::SPLIT_RESHAPE},
                                         {"SplitRawTensor", PassName::SPLIT_RAW_TENSOR},
                                         {"SplitLargeFanoutTensor", PassName::SPLIT_LARGE_FANOUT_TENSOR},
                                         {"DuplicateOp", PassName::DUPLICATE_OP},
                                         {"AssignMemoryType", PassName::ASSIGN_MEMORY_TYPE},
                                     });
        ConfigManager::Instance();
    }

    void CheckConvertOp(const Operation& op, bool verbose = false)
    {
        /*
        1. 单输入单输出
        2. 输入/输出的tensor mem类型唯一
        3. 输入和输出的tensor的mem类型不同
        */
        EXPECT_EQ(op.GetIOperands().size(), 1) << "OP_CONVERT should have ONLY ONE input.";
        EXPECT_EQ(op.GetOOperands().size(), 1) << "OP_CONVERT should have ONLY ONE input.";
        auto input = op.GetIOperands().front();
        ASSERT_NE(input, nullptr) << "OP_CONVERT input is nullptr";
        auto output = op.GetOOperands().front();
        ASSERT_NE(output, nullptr) << "OP_CONVERT output is nullptr";
        auto inputMemOri = input->GetMemoryTypeOriginal();
        auto inputMemTobe = input->GetMemoryTypeToBe();
        auto outputMemOri = output->GetMemoryTypeOriginal();
        auto outputMemTobe = output->GetMemoryTypeToBe();
        if (verbose) {
            std::cout << "\t|--- iOperand " << input->magic;
            std::cout << ", mem ori: " << BriefMemoryTypeToString(inputMemOri);
            std::cout << ", tobe: " << BriefMemoryTypeToString(inputMemTobe) << std::endl;
            std::cout << "\t|--- oOperand " << output->magic;
            std::cout << ", mem ori: " << BriefMemoryTypeToString(outputMemOri);
            std::cout << ", tobe: " << BriefMemoryTypeToString(outputMemTobe) << std::endl;
        }
        EXPECT_EQ(inputMemOri, inputMemTobe) << "OP_CONVERT input Memory Ori should be the same as Memory Tobe.";
        EXPECT_EQ(outputMemOri, outputMemTobe) << "OP_CONVERT output Memory Ori should be the same as Memory Tobe.";
        EXPECT_NE(inputMemOri, outputMemOri) << "OP_CONVERT input should have different memory type from output.";
    }

    int CountAndCheckNewOps(Function& function, const std::vector<int64_t>& beforeMagic,
                            const std::vector<Opcode>& targetOpcodes)
    {
        int convertNum = 0;
        for (const auto& op : function.Operations()) {
            if (std::find(beforeMagic.begin(), beforeMagic.end(), op.opmagic) != beforeMagic.end()) {
                continue;
            }
            if (std::find(targetOpcodes.begin(), targetOpcodes.end(), op.GetOpcode()) == targetOpcodes.end()) {
                continue;
            }
            std::cout << op.GetOpcodeStr() << " " << op.GetOpMagic() << std::endl;
            CheckConvertOp(op, true);
            convertNum++;
        }
        return convertNum;
    }

    bool HasViewMemoryPath(Function* function, MemoryType from, MemoryType to)
    {
        for (const auto& op : function->Operations()) {
            if (op.GetOpcode() != Opcode::OP_VIEW || op.GetIOperands().empty() || op.GetOOperands().empty()) {
                continue;
            }
            auto input = op.GetIOperands().front();
            auto output = op.GetOOperands().front();
            if (input->GetMemoryTypeOriginal() == from && output->GetMemoryTypeOriginal() == to) {
                return true;
            }
        }
        return false;
    }

    void BuildAssembleL1ViewAndNonViewConsumerGraph(ComputationalGraphBuilder& G)
    {
        const Shape shape{NUM_32, NUM_32};
        G.AddTensor(DataType::DT_FP16, shape, MemoryType::MEM_DEVICE_DDR, "incast");
        G.AddTensor(DataType::DT_FP16, shape, MemoryType::MEM_UNKNOWN, "add_out");
        G.AddTensor(DataType::DT_FP16, shape, MemoryType::MEM_UNKNOWN, "assemble1_out");
        G.AddTensor(DataType::DT_FP16, shape, MemoryType::MEM_UNKNOWN, "assemble2_out");
        G.AddTensor(DataType::DT_FP16, shape, MemoryType::MEM_UNKNOWN, "view1_out");
        G.AddTensor(DataType::DT_FP16, shape, MemoryType::MEM_UNKNOWN, "view2_out");
        G.AddTensor(DataType::DT_FP16, shape, MemoryType::MEM_UNKNOWN, "view4_out");
        G.AddTensor(DataType::DT_FP16, shape, MemoryType::MEM_UNKNOWN, "view6_out");
        G.AddTensor(DataType::DT_FP16, shape, MemoryType::MEM_UNKNOWN, "l1_in1");
        G.AddTensor(DataType::DT_FP16, shape, MemoryType::MEM_UNKNOWN, "l1_in2");
        G.AddTensors(DataType::DT_FP16, shape, {MemoryType::MEM_L0B, MemoryType::MEM_L0B}, {"l0b_in1", "l0b_in2"});
        G.AddTensors(DataType::DT_FP16, shape, {MemoryType::MEM_L0C, MemoryType::MEM_L0C},
                     {"matmul1_out", "matmul2_out"});
        G.AddTensors(DataType::DT_FP16, shape, {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR},
                     {"outcast1", "outcast2"});

        G.AddOp(Opcode::OP_ADDS, {"incast"}, {"add_out"}, "add_gather");
        G.AddOp(Opcode::OP_ASSEMBLE, {"add_out"}, {"assemble1_out"}, "assemble1");
        G.GetOp("assemble1")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(Offset{0, 0}));
        G.AddOp(Opcode::OP_ASSEMBLE, {"assemble1_out"}, {"assemble2_out"}, "assemble2");
        G.GetOp("assemble2")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(Offset{0, 0}));

        G.AddOp(Opcode::OP_VIEW, {"assemble1_out"}, {"view1_out"}, "view1");
        G.GetOp("view1")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_L1));
        G.AddOp(Opcode::OP_VIEW, {"assemble2_out"}, {"view2_out"}, "view2");
        G.GetOp("view2")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_L1));
        G.AddOp(Opcode::OP_VIEW, {"view2_out"}, {"view4_out"}, "view4");
        G.GetOp("view4")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_L0A));
        G.AddOp(Opcode::OP_VIEW, {"view1_out"}, {"view6_out"}, "view6");
        G.GetOp("view6")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_L0A));
        G.AddOp(Opcode::OP_VIEW, {"incast"}, {"l1_in1"}, "gather_to_l1_1");
        G.GetOp("gather_to_l1_1")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_L1));
        G.AddOp(Opcode::OP_VIEW, {"l1_in1"}, {"l0b_in1"}, "l1_to_l0b_1");
        G.GetOp("l1_to_l0b_1")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_L0B));
        G.AddOp(Opcode::OP_VIEW, {"incast"}, {"l1_in2"}, "gather_to_l1_2");
        G.GetOp("gather_to_l1_2")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_L1));
        G.AddOp(Opcode::OP_VIEW, {"l1_in2"}, {"l0b_in2"}, "l1_to_l0b_2");
        G.GetOp("l1_to_l0b_2")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_L0B));

        G.AddOp(Opcode::OP_A_MUL_B, {"view4_out", "l0b_in1"}, {"matmul1_out"}, "amulb1");
        G.AddOp(Opcode::OP_A_MUL_B, {"view6_out", "l0b_in2"}, {"matmul2_out"}, "amulb2");
        G.AddOp(Opcode::OP_ASSEMBLE, {"matmul1_out"}, {"outcast1"}, "assemble_out1");
        G.GetOp("assemble_out1")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(Offset{0, 0}));
        G.AddOp(Opcode::OP_ASSEMBLE, {"matmul2_out"}, {"outcast2"}, "assemble_out2");
        G.GetOp("assemble_out2")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(Offset{0, 0}));
        G.SetInCast({"incast"});
        G.SetOutCast({"outcast1", "outcast2"});
    }
};

TEST_F(LegacyAssignMemoryTypeTest, AddReshape)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TILE_AddReshape", "TILE_AddReshape",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    Program::GetInstance().InsertFuncToFunctionMap("TILE_AddReshape", currFunctionPtr);

    constexpr int opMagic0 = 1001;
    constexpr int opMagic1 = 1002;
    constexpr int opMagic2 = 1003;
    constexpr int opMagic3 = 1004;
    constexpr int opMagic4 = 1005;

    constexpr int tensorMagic0 = 1;
    constexpr int tensorMagic1 = 2;
    constexpr int tensorMagic2 = 3;
    constexpr int tensorMagic3 = 4;
    constexpr int tensorMagic4 = 5;
    constexpr int tensorMagic5 = 6;
    constexpr int tensorMagic6 = 7;
    // Prepare the graph
    std::vector<int64_t> shape = {16, 32};
    std::vector<int64_t> shape1 = {32, 16};
    std::shared_ptr<LogicalTensor> input_tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape, CreateTestConstIntVector(shape));
    input_tensor1->SetMemoryTypeBoth(MEM_DEVICE_DDR);
    input_tensor1->SetMagic(tensorMagic0);

    std::shared_ptr<LogicalTensor> input_tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape, CreateTestConstIntVector(shape));
    input_tensor2->SetMemoryTypeBoth(MEM_UNKNOWN);
    input_tensor2->SetMagic(tensorMagic1);

    std::shared_ptr<LogicalTensor> view_output1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape, CreateTestConstIntVector(shape));
    view_output1->SetMemoryTypeBoth(MEM_UNKNOWN);
    view_output1->SetMagic(tensorMagic6);

    std::shared_ptr<LogicalTensor> view_output2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape, CreateTestConstIntVector(shape));
    view_output2->SetMemoryTypeBoth(MEM_UNKNOWN);
    view_output2->SetMagic(tensorMagic2);

    std::shared_ptr<LogicalTensor> add_output = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape, CreateTestConstIntVector(shape));
    add_output->SetMemoryTypeBoth(MEM_UNKNOWN);
    add_output->SetMagic(tensorMagic3);

    std::shared_ptr<LogicalTensor> reshape_output = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape1, CreateTestConstIntVector(shape1));
    reshape_output->SetMemoryTypeBoth(MEM_UNKNOWN);
    reshape_output->SetMagic(tensorMagic4);

    std::shared_ptr<LogicalTensor> assemble_output = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape1, CreateTestConstIntVector(shape1));
    assemble_output->SetMemoryTypeBoth(MEM_UNKNOWN);
    assemble_output->SetMagic(tensorMagic5);

    auto& view_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {input_tensor1}, {view_output1});
    view_op1.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    view_op1.opmagic = opMagic0;

    auto& view_op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {input_tensor2}, {view_output2});
    view_op2.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    view_op2.opmagic = opMagic3;

    auto& add_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_ADD, {view_output1, view_output2},
                                                  {add_output});
    add_op.opmagic = opMagic1;

    auto& reshape_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_RESHAPE, {add_output},
                                                      {reshape_output});
    reshape_op.opmagic = opMagic4;

    auto& assemble_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_ASSEMBLE, {reshape_output},
                                                       {assemble_output});
    assemble_op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    assemble_op.opmagic = opMagic2;

    currFunctionPtr->inCasts_.push_back(input_tensor1);
    currFunctionPtr->inCasts_.push_back(input_tensor2);
    currFunctionPtr->outCasts_.push_back(assemble_output);

    std::stringstream ssBefore;
    ssBefore << "Before_AssignMemoryType";

    // Call the pass
    AssignMemoryType assignMemoryType;
    assignMemoryType.PreCheck(*currFunctionPtr);
    assignMemoryType.RunOnFunction(*currFunctionPtr);
    assignMemoryType.PostCheck(*currFunctionPtr);

    std::stringstream ss;
    ss << "After_AssignMemoryType";

    // Validate the results, 所有op的输入输出memory类型唯一
    std::cout << "========== op size: " << currFunctionPtr->Operations().size() << std::endl;
    int convertNum = 0;
    for (auto& op : currFunctionPtr->Operations()) {
        std::cout << op.GetOpcodeStr() << " " << op.GetOpMagic() << std::endl;
        for (auto& input : op.GetIOperands()) {
            auto memOri = input->GetMemoryTypeOriginal();
            auto memTobe = input->GetMemoryTypeToBe();
            std::cout << "\t|--- iOperand " << input->magic;
            std::cout << ", mem ori: " << BriefMemoryTypeToString(memOri);
            std::cout << ", tobe: " << BriefMemoryTypeToString(memTobe) << std::endl;
            EXPECT_EQ(memOri, memTobe) << " input Memory Ori should be the same as Memory Tobe";
        }
        for (auto& output : op.GetOOperands()) {
            auto memOri = output->GetMemoryTypeOriginal();
            auto memTobe = output->GetMemoryTypeToBe();
            std::cout << "\t|--- oOperand " << output->magic;
            std::cout << ", mem ori: " << BriefMemoryTypeToString(memOri);
            std::cout << ", tobe: " << BriefMemoryTypeToString(memTobe) << std::endl;
            EXPECT_EQ(memOri, memTobe) << " output Memory Ori should be the same as Memory Tobe";
        }
        if (op.GetOpcode() == Opcode::OP_CONVERT || (op.GetOpcode() == Opcode::OP_ASSEMBLE && op.opmagic != opMagic2)) {
            convertNum++;
            CheckConvertOp(op);
        }
    }
    constexpr int expextedConvertNum = 1;
    EXPECT_EQ(convertNum, expextedConvertNum) << "ONLY ONE OP_CONVERT.";
}

TEST_F(LegacyAssignMemoryTypeTest, TestVecToCubeV2)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shape0 = {256, 128};
    std::vector<int64_t> shape1 = {128, 64};
    std::vector<int64_t> shape2 = {256, 64};
    PROGRAM("AssignMemoryTest")
    {
        Tensor input1(DataType::DT_FP32, shape0, "A");
        Tensor input2(DataType::DT_FP32, shape0, "B");
        Tensor weight(DataType::DT_FP32, shape1, "weight");
        Tensor out(DataType::DT_FP32, shape2, "output");
        SetHalfwayStrategy();
        Function* originFunction = nullptr;

        config::SetBuildStatic(true);
        FUNCTION("TestVecToCubeV2", {input1, input2, weight, out})
        {
            config::SetPassStrategy("AssignMemoryTypeTestStrategy");
            TileShape::Current().SetVecTile(NUM_128, NUM_128);
            Tensor addRes = Add(input1, input2); // 256 * 128
            TileShape::Current().SetCubeTile({NUM_32, NUM_32}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor mmRes = Matrix::Matmul(out.GetDataType(), addRes, weight); // (256 * 128) @ (128 * 64) = (256 * 64)
            TileShape::Current().SetVecTile(NUM_128, NUM_128);
            Tensor sumRes = Sum(addRes, 1, true);
            TileShape::Current().SetVecTile(NUM_64, NUM_64);
            out = Add(mmRes, sumRes);
        }

        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_TestVecToCubeV2"); // Tensor_{Function名字}
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        std::vector<int64_t> beforeMagic;
        for (const auto& op : originFunction->Operations()) {
            if (op.GetOpcode() == Opcode::OP_CONVERT || op.GetOpcode() == Opcode::OP_VIEW ||
                op.GetOpcode() == Opcode::OP_ASSEMBLE) {
                beforeMagic.push_back(op.opmagic);
            }
        }
        // Call the pass
        AssignMemoryType assignMemoryType;
        assignMemoryType.PreCheck(*originFunction);
        assignMemoryType.RunOnFunction(*originFunction);
        assignMemoryType.PostCheck(*originFunction);
        // ================== Verify Pass Effect ==================
        auto updatedOperations = originFunction->Operations();
        int convertNum = 0;
        for (const auto& op : updatedOperations) {
            if (op.GetOpcode() == Opcode::OP_CONVERT || op.GetOpcode() == Opcode::OP_VIEW ||
                op.GetOpcode() == Opcode::OP_ASSEMBLE) {
                if (std::find(beforeMagic.begin(), beforeMagic.end(), op.opmagic) != beforeMagic.end()) {
                    continue;
                }
                convertNum++;
                std::cout << op.GetOpcodeStr() << " " << op.GetOpMagic() << std::endl;
                CheckConvertOp(op, true);
            }
        }
        constexpr int expextedConvertNum = 0;
        EXPECT_EQ(convertNum, expextedConvertNum) << "0 operations should be Convert";
    }
}

TEST_F(LegacyAssignMemoryTypeTest, TestCubeToCube)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shape0 = {256, 128};
    std::vector<int64_t> shape1 = {128, 64};
    std::vector<int64_t> shape2 = {256, 256};
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputQ(DataType::DT_BF16, shape0, "Q");
        Tensor inputK(DataType::DT_BF16, shape0, "K");
        Tensor weight(DataType::DT_BF16, shape1, "weight");
        Tensor out(DataType::DT_FP32, shape2, "output");
        SetHalfwayStrategy();
        Function* originFunction = nullptr;

        config::SetBuildStatic(true);
        FUNCTION("TestCubeToCube", {inputQ, inputK, weight, out})
        {
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor qUpdate = Matrix::Matmul(out.GetDataType(), inputQ, weight); // (256 * 128) @ (128 * 64) = (256 * 64)
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor kUpdate = Matrix::Matmul(out.GetDataType(), inputK, weight); // (256 * 128) @ (128 * 64) = (256 * 64)
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_64, NUM_64}, {NUM_128, NUM_128});
            Tensor QKT = Matrix::Matmul(out.GetDataType(), qUpdate, kUpdate, false,
                                        true); // (256 * 64) @ (64 * 256) = (256 * 256)
            TileShape::Current().SetVecTile(NUM_64, NUM_64);
            out = Sub(QKT, Element(DataType::DT_FP32, F_3));
        }

        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_TestCubeToCube"); // Tensor_{Function名字}
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        std::vector<int64_t> beforeMagic;
        for (const auto& op : originFunction->Operations()) {
            if (op.GetOpcode() == Opcode::OP_CONVERT || op.GetOpcode() == Opcode::OP_VIEW ||
                op.GetOpcode() == Opcode::OP_ASSEMBLE) {
                beforeMagic.push_back(op.opmagic);
            }
        }
        // Call the pass
        AssignMemoryType assignMemoryType;
        assignMemoryType.PreCheck(*originFunction);
        assignMemoryType.RunOnFunction(*originFunction);
        assignMemoryType.PostCheck(*originFunction);
        // ================== Verify Pass Effect ==================
        int convertNum = CountAndCheckNewOps(*originFunction, beforeMagic, {Opcode::OP_CONVERT});
        constexpr int expextedConvertNum = 0;
        EXPECT_EQ(convertNum, expextedConvertNum) << "0 operations should be Convert";
    }
}

TEST_F(LegacyAssignMemoryTypeTest, TestCubeToCubeV2)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shape0 = {256, 128};
    std::vector<int64_t> shape1 = {128, 64};
    std::vector<int64_t> shape2 = {256, 256};
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputQ(DataType::DT_FP32, shape0, "Q");
        Tensor inputK(DataType::DT_FP32, shape0, "K");
        Tensor weight(DataType::DT_FP32, shape1, "weight");
        Tensor out(DataType::DT_FP32, shape2, "output");
        SetHalfwayStrategy();
        Function* originFunction = nullptr;

        config::SetBuildStatic(true);
        FUNCTION("TestCubeToCubeV2", {inputQ, inputK, weight, out})
        {
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor qUpdate = Matrix::Matmul(out.GetDataType(), inputQ, weight); // (256 * 128) @ (128 * 64) = (256 * 64)
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor kUpdate = Matrix::Matmul(out.GetDataType(), inputK, weight); // (256 * 128) @ (128 * 64) = (256 * 64)
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_64, NUM_64}, {NUM_128, NUM_128});
            Tensor QKT = Matrix::Matmul(out.GetDataType(), qUpdate, kUpdate, false,
                                        true); // (256 * 64) @ (64 * 256) = (256 * 256)
            TileShape::Current().SetVecTile(NUM_64, NUM_64);
            out = Add(QKT, Element(DataType::DT_FP32, F_1));
        }

        originFunction = Program::GetInstance().GetFunctionByRawName(
            "TENSOR_TestCubeToCubeV2"); // Tensor_{Function名字}
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        std::vector<int64_t> beforeMagic;
        for (const auto& op : originFunction->Operations()) {
            if (op.GetOpcode() == Opcode::OP_CONVERT || op.GetOpcode() == Opcode::OP_VIEW ||
                op.GetOpcode() == Opcode::OP_ASSEMBLE) {
                beforeMagic.push_back(op.opmagic);
            }
        }
        // Call the pass
        AssignMemoryType assignMemoryType;
        assignMemoryType.PreCheck(*originFunction);
        assignMemoryType.RunOnFunction(*originFunction);
        assignMemoryType.PostCheck(*originFunction);
        // ================== Verify Pass Effect ==================
        int convertNum = CountAndCheckNewOps(*originFunction, beforeMagic,
                                             {Opcode::OP_CONVERT, Opcode::OP_VIEW, Opcode::OP_ASSEMBLE});
        constexpr int expextedConvertNum = 0;
        EXPECT_EQ(convertNum, expextedConvertNum) << "0 operations should be Convert";
    }
}

TEST_F(LegacyAssignMemoryTypeTest, TestCubeToVec)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shape0 = {NUM_64, NUM_128};
    std::vector<int64_t> shape1 = {NUM_128, NUM_64};
    std::vector<int64_t> shape2 = {NUM_64, NUM_64};
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputA1(DataType::DT_FP32, shape0, "A1");
        Tensor inputB1(DataType::DT_FP32, shape1, "B1");
        Tensor inputA2(DataType::DT_FP32, shape0, "A2");
        Tensor inputB2(DataType::DT_FP32, shape1, "B2");
        Tensor inputV1(DataType::DT_FP32, shape0, "B2");
        Tensor inputV2(DataType::DT_FP32, shape0, "B2");
        Tensor out(DataType::DT_FP32, shape2, "output");
        SetHalfwayStrategy();
        Function* originFunction = nullptr;
        config::SetBuildStatic(true);
        FUNCTION("TestCubeToVec", {inputA1, inputB1, inputA2, inputB2, inputV1, inputV2, out})
        {
            TileShape::Current().SetCubeTile({NUM_64, NUM_64}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor C1 = Matrix::Matmul(out.GetDataType(), inputA1, inputB1); // (64 * 128) @ (128 * 64) = (64 * 64)
            TileShape::Current().SetCubeTile({NUM_64, NUM_64}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor C2 = Matrix::Matmul(out.GetDataType(), inputA2, inputB2); // (64 * 128) @ (128 * 64) = (64 * 64)
            Assemble(C1, {0, 0}, inputV1);
            Assemble(C2, {0, NUM_64}, inputV1);
            TileShape::Current().SetVecTile(NUM_64, NUM_128);
            out = Add(inputV1, inputV2);
        }
        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_TestCubeToVec"); // Tensor_{Function名字}
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        int64_t beforeViewNum = 0;
        for (const auto& op : originFunction->Operations()) {
            if (op.GetOpcode() == Opcode::OP_VIEW) {
                ++beforeViewNum;
            }
        }
        // Call the pass
        AssignMemoryType assignMemoryType;
        assignMemoryType.PreCheck(*originFunction);
        assignMemoryType.RunOnFunction(*originFunction);
        assignMemoryType.PostCheck(*originFunction);
        // ================== Verify Pass Effect ==================
        int64_t afterViewNum = 0;
        for (const auto& op : originFunction->Operations()) {
            if (op.GetOpcode() == Opcode::OP_VIEW) {
                ++afterViewNum;
                auto viewOpAttr = std::dynamic_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
                EXPECT_TRUE(viewOpAttr->GetTo() == MemoryType::MEM_L1 || viewOpAttr->GetTo() == MemoryType::MEM_UB ||
                            viewOpAttr->GetTo() == MemoryType::MEM_L0A || viewOpAttr->GetTo() == MemoryType::MEM_L0B)
                    << "View to either l1, ub, l0a or l0b";
            }
        }
        EXPECT_EQ(afterViewNum, beforeViewNum);
    }
}

static void GetInvalidPatternGraph(std::shared_ptr<Function>& currFunctionPtr)
{
    constexpr int opMagic0 = 1001;
    constexpr int opMagic1 = 1002;
    constexpr int opMagic2 = 1003;
    constexpr int opMagic3 = 1004;
    constexpr int opMagic4 = 1005;
    constexpr int opMagic5 = 1006;
    constexpr int opMagic6 = 1007;
    constexpr int opMagic7 = 1008;

    constexpr int tensorMagic0 = 1;
    constexpr int tensorMagic1 = 2;
    constexpr int tensorMagic2 = 3;
    constexpr int tensorMagic3 = 4;
    constexpr int tensorMagic4 = 5;
    constexpr int tensorMagic5 = 6;
    constexpr int tensorMagic6 = 7;
    constexpr int tensorMagic7 = 8;
    // Prepare the graph
    std::vector<int64_t> shape = {16, 32};
    std::vector<int64_t> shape1 = {32, 16};
    std::vector<int64_t> shape2 = {8, 32};
    std::vector<int64_t> shape3 = {32, 8};
    std::shared_ptr<LogicalTensor> input_cast = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape1, CreateTestConstIntVector(shape1));
    input_cast->SetMagic(tensorMagic0);

    std::shared_ptr<LogicalTensor> input_tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape, CreateTestConstIntVector(shape));
    input_tensor1->SetMagic(tensorMagic1);

    std::shared_ptr<LogicalTensor> view_output1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape2, CreateTestConstIntVector(shape2));
    view_output1->SetMagic(tensorMagic2);

    std::shared_ptr<LogicalTensor> view_output2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape2, CreateTestConstIntVector(shape2));
    view_output2->SetMagic(tensorMagic3);

    std::shared_ptr<LogicalTensor> reshape_output1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape3, CreateTestConstIntVector(shape3));
    reshape_output1->SetMagic(tensorMagic4);

    std::shared_ptr<LogicalTensor> reshape_output2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape3, CreateTestConstIntVector(shape3));
    reshape_output2->SetMagic(tensorMagic5);

    std::shared_ptr<LogicalTensor> assemble_output = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape1, CreateTestConstIntVector(shape1));
    assemble_output->SetMagic(tensorMagic6);

    std::shared_ptr<LogicalTensor> output_cast = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape1, CreateTestConstIntVector(shape1));
    output_cast->SetMagic(tensorMagic7);

    auto& reshape_op0 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_RESHAPE, {input_cast},
                                                       {input_tensor1});
    reshape_op0.opmagic = opMagic0;

    auto& view_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {input_tensor1}, {view_output1});
    view_op1.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    view_op1.opmagic = opMagic1;

    auto& view_op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {input_tensor1}, {view_output2});
    view_op2.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{8, 0}));
    view_op2.opmagic = opMagic2;

    auto& reshape_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_RESHAPE, {view_output1},
                                                       {reshape_output1});
    reshape_op1.opmagic = opMagic3;

    auto& reshape_op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_RESHAPE, {view_output2},
                                                       {reshape_output2});
    reshape_op2.opmagic = opMagic4;

    auto& assemble_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_ASSEMBLE, {reshape_output1},
                                                        {assemble_output});
    assemble_op1.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    assemble_op1.opmagic = opMagic5;

    auto& assemble_op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_ASSEMBLE, {reshape_output2},
                                                        {assemble_output});
    assemble_op2.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{8, 0}));
    assemble_op2.opmagic = opMagic6;

    auto& view_op3 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {assemble_output},
                                                    {output_cast});
    view_op3.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    view_op3.opmagic = opMagic7;

    currFunctionPtr->inCasts_.push_back(input_cast);
    currFunctionPtr->outCasts_.push_back(output_cast);
}

static void CallAndVerify(std::shared_ptr<Function>& currFunctionPtr, const MemoryType type)
{
    std::stringstream ssBefore;
    ssBefore << "Before_AssignMemoryType";

    // Call the pass
    AssignMemoryType assignMemoryType;
    assignMemoryType.PreCheck(*currFunctionPtr);
    assignMemoryType.RunOnFunction(*currFunctionPtr);
    assignMemoryType.PostCheck(*currFunctionPtr);

    std::stringstream ss;
    ss << "After_AssignMemoryType";

    std::string josnFilePath = "./config/pass/json/assign_mem_type_invalidpattern.json";
    currFunctionPtr->DumpJsonFile(josnFilePath);

    // Validate the results
    std::cout << "========== op size: " << currFunctionPtr->Operations().size() << std::endl;
    for (auto& op : currFunctionPtr->Operations()) {
        std::cout << op.GetOpcodeStr() << " " << op.GetOpMagic() << std::endl;
        for (auto& input : op.GetIOperands()) {
            std::cout << "\t|--- iOperand " << input->magic;
            EXPECT_EQ(input->GetMemoryTypeOriginal(), type) << " Unexpected memory type.";
            EXPECT_EQ(input->GetMemoryTypeOriginal(), input->GetMemoryTypeToBe()) << " iOperand has two memory type.";
        }
        for (auto& output : op.GetOOperands()) {
            std::cout << "\t|--- oOperand " << output->magic << std::endl;
            EXPECT_EQ(output->GetMemoryTypeOriginal(), type) << " Unexpected memory type.";
            EXPECT_EQ(output->GetMemoryTypeOriginal(), output->GetMemoryTypeToBe()) << " oOperand has two memory type.";
        }
    }
}

TEST_F(LegacyAssignMemoryTypeTest, InValidOpPattern)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "InValidOpPattern", "InValidOpPattern",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    Program::GetInstance().InsertFuncToFunctionMap("InValidOpPattern", currFunctionPtr);

    GetInvalidPatternGraph(currFunctionPtr);

    CallAndVerify(currFunctionPtr, MemoryType::MEM_DEVICE_DDR);
}

static void GetViewReshapeGraph(std::shared_ptr<Function>& currFunctionPtr)
{
    constexpr int opMagic0 = 1001;
    constexpr int opMagic1 = 1002;
    constexpr int opMagic2 = 1003;
    constexpr int opMagic3 = 1004;
    constexpr int opMagic4 = 1005;

    constexpr int tensorMagic0 = 1;
    constexpr int tensorMagic1 = 2;
    constexpr int tensorMagic2 = 3;
    constexpr int tensorMagic3 = 4;
    constexpr int tensorMagic4 = 5;
    constexpr int tensorMagic5 = 6;

    // Prepare the graph
    std::vector<int64_t> shape = {16, 32};
    std::vector<int64_t> shape1 = {32, 16};
    std::vector<int64_t> shape2 = {1, 32};
    std::vector<int64_t> shape3 = {8, 32};
    std::shared_ptr<LogicalTensor> input_cast = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape, CreateTestConstIntVector(shape));
    input_cast->SetMagic(tensorMagic0);

    std::shared_ptr<LogicalTensor> transpose_out = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape1, CreateTestConstIntVector(shape1));
    transpose_out->SetMagic(tensorMagic1);

    std::shared_ptr<LogicalTensor> view_output1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape1, CreateTestConstIntVector(shape1));
    view_output1->SetMagic(tensorMagic2);

    std::shared_ptr<LogicalTensor> reshape_output = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape, CreateTestConstIntVector(shape));
    reshape_output->SetMagic(tensorMagic3);

    std::shared_ptr<LogicalTensor> view_output2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape2, CreateTestConstIntVector(shape2));
    view_output2->SetMagic(tensorMagic4);

    std::shared_ptr<LogicalTensor> output_cast = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape3, CreateTestConstIntVector(shape3));
    output_cast->SetMagic(tensorMagic5);

    auto& transpose_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_TRANSPOSE_VNCHWCONV, {input_cast},
                                                        {transpose_out});
    transpose_op.opmagic = opMagic0;

    auto& view_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {transpose_out}, {view_output1});
    view_op1.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    view_op1.opmagic = opMagic1;

    auto& reshape_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_RESHAPE, {view_output1},
                                                      {reshape_output});
    reshape_op.opmagic = opMagic2;

    auto& view_op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {reshape_output},
                                                    {view_output2});
    view_op2.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    view_op2.opmagic = opMagic3;

    auto& expand_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_EXPAND, {view_output2},
                                                     {output_cast});
    expand_op.opmagic = opMagic4;
}
TEST_F(LegacyAssignMemoryTypeTest, ViewReshape)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "ViewReshape", "ViewReshape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    Program::GetInstance().InsertFuncToFunctionMap("ViewReshape", currFunctionPtr);

    GetViewReshapeGraph(currFunctionPtr);
    CallAndVerify(currFunctionPtr, MemoryType::MEM_UB);
}
static void L1DataMoveGraph(std::shared_ptr<Function>& currFunctionPtr)
{
    std::shared_ptr<LogicalTensor> input_cast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> input_cast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{64, 16}, CreateTestConstIntVector(std::vector<int64_t>{64, 16}));
    std::shared_ptr<LogicalTensor> op_view_L1_out1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> op_view_L1_out2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{64, 16}, CreateTestConstIntVector(std::vector<int64_t>{64, 16}));
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
    // std::shared_ptr<LogicalTensor> output_cast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape,
    // CreateTestConstIntVector(shape));
    auto& view_L1_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {input_cast1},
                                                       {op_view_L1_out1});
    std::vector<int> newoffset{0, 0};
    auto viewAttribute = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute->SetToType(MemoryType::MEM_L1);
    view_L1_op1.SetOpAttribute(viewAttribute);

    auto& view_L1_op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {input_cast2},
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
TEST_F(LegacyAssignMemoryTypeTest, L1DataMove)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "L1DataMove", "L1DataMove", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("L1DataMove", currFunctionPtr);

    L1DataMoveGraph(currFunctionPtr);

    std::stringstream ssBefore;
    ssBefore << "Before_AssignMemoryType";

    // Call the pass
    AssignMemoryType assignMemoryType;
    assignMemoryType.PreCheck(*currFunctionPtr);
    currFunctionPtr->DumpJsonFile("./config/pass/json/assign_mem_type_L1DataMove_before.json");
    assignMemoryType.RunOnFunction(*currFunctionPtr);
    currFunctionPtr->DumpJsonFile("./config/pass/json/assign_mem_type_L1DataMove_after.json");
    assignMemoryType.PostCheck(*currFunctionPtr);

    std::stringstream ss;
    ss << "After_AssignMemoryType";

    // Validate the results
    std::cout << "========== op size: " << currFunctionPtr->Operations().size() << std::endl;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() != Opcode::OP_VIEW) {
            continue;
        } else {
            auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(op.GetOpAttribute().get());
            auto mem_to = viewOpAttribute->GetTo();
            if (mem_to != MemoryType::MEM_L1) {
                continue;
            } else {
                EXPECT_EQ(op.GetIOperands().size(), 1) << "View op has more than one input!";
                EXPECT_EQ(op.GetOOperands().size(), 1) << "View op has more than one output!";
                auto input = op.GetIOperands().front();
                auto output = op.GetOOperands().front();
                std::cout << "\t|--- MEM_L1 VIEW iOperand " << input->GetMagic() << std::endl;
                std::cout << "\t|--- MEM_L1 VIEW oOperand " << output->GetMagic() << std::endl;
                // EXPECT_EQ(input->GetMemoryTypeToBe(),MemoryType::MEM_L1) << "View op input has unexpected memory
                // type!";
                EXPECT_EQ(output->GetMemoryTypeOriginal(), MemoryType::MEM_L1)
                    << "View op input has unexpected memory type!";
            }
        }
    }
}

static void AssignViewTensorWithAttr(std::shared_ptr<Function>& currFunctionPtr)
{
    std::shared_ptr<LogicalTensor> view_in1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> view_out1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> view_in2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> tensor4 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> view_out2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> view_in3 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> tensor5 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> view_out3 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> view_in4 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> tensor6 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> view_out4 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> output = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));

    auto& view_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {view_in1}, {tensor1});
    auto viewAttribute1 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute1->SetToType(MemoryType::MEM_L1);
    view_op1.SetOpAttribute(viewAttribute1);
    auto& view_op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {tensor1}, {view_out1});
    auto viewAttribute2 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute2->SetToType(MemoryType::MEM_BT);
    view_op2.SetOpAttribute(viewAttribute2);
    auto& view_op3 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {view_in2}, {tensor2});
    auto viewAttribute3 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute3->SetToType(MemoryType::MEM_L1);
    view_op3.SetOpAttribute(viewAttribute3);
    auto& view_op4 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {tensor2}, {view_out2});
    auto viewAttribute4 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute4->SetToType(MemoryType::MEM_FIX_QUANT_PRE);
    view_op4.SetOpAttribute(viewAttribute4);
    auto& view_op5 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {view_in3}, {tensor3});
    auto viewAttribute5 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute5->SetToType(MemoryType::MEM_L1);
    view_op5.SetOpAttribute(viewAttribute5);
    auto& view_op6 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {tensor3}, {view_out3});
    auto viewAttribute6 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute6->SetToType(MemoryType::MEM_L0A);
    view_op6.SetOpAttribute(viewAttribute6);
    auto& view_op7 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {view_in4}, {tensor4});
    auto viewAttribute7 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute7->SetToType(MemoryType::MEM_L1);
    view_op7.SetOpAttribute(viewAttribute7);
    auto& view_op8 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {tensor4}, {view_out4});
    auto viewAttribute8 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute8->SetToType(MemoryType::MEM_L0B);
    view_op8.SetOpAttribute(viewAttribute7);

    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_A_MUL_B, {view_out3, view_out4, view_out1, view_out2},
                                   {output});

    currFunctionPtr->inCasts_.push_back(view_in1);
    currFunctionPtr->inCasts_.push_back(view_in2);
    currFunctionPtr->inCasts_.push_back(view_in3);
    currFunctionPtr->inCasts_.push_back(view_in4);
    currFunctionPtr->outCasts_.push_back(output);
}

TEST_F(LegacyAssignMemoryTypeTest, TestViewWithAttr)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestViewWithAttr", "TestViewWithAttr",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("TestViewWithAttr", currFunctionPtr);

    AssignViewTensorWithAttr(currFunctionPtr);

    std::stringstream ssBefore;
    ssBefore << "Before_AssignMemoryType";

    // Call the pass
    AssignMemoryType assignMemoryType;
    assignMemoryType.PreCheck(*currFunctionPtr);
    currFunctionPtr->DumpJsonFile("./config/pass/json/assignMemoryType_TestViewWithAttr_before.json");
    assignMemoryType.RunOnFunction(*currFunctionPtr);
    currFunctionPtr->DumpJsonFile("./config/pass/json/assignMemoryType_TestViewWithAttr_after.json");
    assignMemoryType.PostCheck(*currFunctionPtr);

    std::stringstream ss;
    ss << "After_AssignMemoryType";

    // Validate the results
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(op.GetOpAttribute().get());
            MemoryType attrToType = viewOpAttribute->GetTo();
            auto output = op.GetOOperands().front();
            auto outputMemOri = output->GetMemoryTypeOriginal();
            auto outputMemTobe = output->GetMemoryTypeToBe();
            std::cout << "\t|--- oOperand " << output->magic;
            std::cout << ", mem ori: " << BriefMemoryTypeToString(outputMemOri);
            std::cout << ", tobe: " << BriefMemoryTypeToString(outputMemTobe) << std::endl;
            EXPECT_EQ(attrToType, outputMemOri);
            EXPECT_EQ(attrToType, outputMemTobe);
        }
    }
}

TEST_F(LegacyAssignMemoryTypeTest, TestPostcheckFailWhenTensorMemUnknown)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPostcheckFailWhenTensorMemUnknown",
                                                      "TestPostcheckFailWhenTensorMemUnknown", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("TestPostcheckFailWhenTensorMemUnknown", currFunctionPtr);
    AssignViewTensorWithAttr(currFunctionPtr);
    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.PostCheck(*currFunctionPtr), FAILED);
}

TEST_F(LegacyAssignMemoryTypeTest, TestPostcheckFailWhenPathUnreachable)
{
    std::vector<int64_t> shape1{NUM_32, NUM_32};
    std::vector<int64_t> shape2{NUM_64, NUM_64};
    std::vector<int64_t> shape3{NUM_128, NUM_128};
    ComputationalGraphBuilder G;

    G.AddTensor(DataType::DT_FP32, shape3, "input");
    auto tensorInput = G.GetTensor("input");
    tensorInput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(DataType::DT_FP32, shape2, "a");
    auto tensorA = G.GetTensor("a");
    tensorA->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    G.AddTensor(DataType::DT_FP32, shape1, "b");
    auto tensorB = G.GetTensor("b");
    tensorB->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(DataType::DT_FP32, shape3, "output");
    auto tensorOutput = G.GetTensor("output");
    tensorOutput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.AddOp(Opcode::OP_VIEW, {"input"}, {"a"}, "view1");
    G.GetOp("view1")->SetOpAttribute(std::make_shared<ViewOpAttribute>(shape3, MemoryType::MEM_L0C));
    G.AddOp(Opcode::OP_VIEW, {"a"}, {"b"}, "view2");
    G.GetOp("view2")->SetOpAttribute(std::make_shared<ViewOpAttribute>(shape2, MemoryType::MEM_DEVICE_DDR));
    G.AddOp(Opcode::OP_ASSEMBLE, {"b"}, {"output"}, "assemble1");
    G.GetOp("assemble1")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, shape2));

    G.SetInCast({"input"});
    G.SetOutCast({"output"});

    Function* function = G.GetFunction();

    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.PostCheck(*function), FAILED);
}

TEST_F(LegacyAssignMemoryTypeTest, UnalignedAssembleBeforeReshapeFallbackDdr)
{
    ComputationalGraphBuilder G;
    Shape shape{NUM_16, NUM_32};
    Shape reshapeShape{NUM_32, NUM_16};
    G.AddTensor(DataType::DT_FP16, shape, MemoryType::MEM_UNKNOWN, "vec_dup_out");
    G.AddTensor(DataType::DT_FP16, shape, MemoryType::MEM_UNKNOWN, "assemble_out");
    G.AddTensor(DataType::DT_FP16, reshapeShape, MemoryType::MEM_UNKNOWN, "reshape_out");
    G.AddTensor(DataType::DT_FP16, reshapeShape, MemoryType::MEM_UNKNOWN, "view_out");

    G.AddOp(Opcode::OP_VEC_DUP, {}, {"vec_dup_out"}, "vec_dup");
    G.AddOp(Opcode::OP_ASSEMBLE, {"vec_dup_out"}, {"assemble_out"}, "assemble");
    G.GetOp("assemble")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(Offset{0, 1}));
    G.AddOp(Opcode::OP_RESHAPE, {"assemble_out"}, {"reshape_out"}, "reshape");
    G.AddOp(Opcode::OP_VIEW, {"reshape_out"}, {"view_out"}, "view");
    G.GetOp("view")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_UB));

    Function* func = G.GetFunction();
    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.RunOnFunction(*func), SUCCESS);
    EXPECT_EQ(assignMemoryType.PostCheck(*func), SUCCESS);

    EXPECT_EQ(G.GetTensor("vec_dup_out")->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    EXPECT_EQ(G.GetTensor("assemble_out")->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(G.GetTensor("reshape_out")->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(G.GetTensor("view_out")->GetMemoryTypeOriginal(), MemoryType::MEM_UB);

    auto assembleOpAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(G.GetOp("assemble")->GetOpAttribute());
    ASSERT_NE(assembleOpAttr, nullptr);
    EXPECT_EQ(assembleOpAttr->GetFrom(), MemoryType::MEM_UB);
}

TEST_F(LegacyAssignMemoryTypeTest, AssembleAndReshapeAfterAssemble)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shape0 = {NUM_64, NUM_32};
    std::vector<int64_t> shape1 = {NUM_32, NUM_64};
    PROGRAM("AssignMemoryTest")
    {
        Tensor input1(DataType::DT_FP32, shape0, "In1");
        Tensor input2(DataType::DT_FP32, shape0, "In2");
        Tensor input3(DataType::DT_FP32, shape1, "In3");
        Tensor output1(DataType::DT_FP32, shape1, "Out1");
        Tensor output2(DataType::DT_FP32, shape0, "Out2");
        SetTestStrategy();
        Function* originFunction = nullptr;
        config::SetBuildStatic(true);
        FUNCTION("AssembleAndReshapeAfterAssemble", {input1, input2, output1, output2})
        {
            TileShape::Current().SetVecTile(NUM_256, NUM_128);
            Tensor t1 = Add(input1, input2);
            Tensor t2(DT_FP32, shape0, "t2");
            Tensor t3(DT_FP32, shape0, "t2");
            Assemble(t1, {0, 0}, t2);
            Assemble(t2, {0, 0}, t3);
            Assemble(t3, {0, 0}, output2);
            Tensor r1 = Reshape(t2, shape1);
            output1 = Add(r1, input3);
        }
        originFunction = Program::GetInstance().GetFunctionByRawName(
            "TENSOR_AssembleAndReshapeAfterAssemble"); // Tensor_{Function名字}
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        for (auto& op : originFunction->Operations()) {
            if (op.GetOpcode() == Opcode::OP_RESHAPE) {
                EXPECT_EQ(op.iOperand[0]->GetMemoryTypeOriginal(), op.oOperand[0]->GetMemoryTypeOriginal());
            }
        }
    }
}

static int CountL0c2l1Num(Function* originFunction)
{
    int l0c2l1Count = 0;
    for (auto& op : originFunction->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE || op.GetOpcode() == Opcode::OP_CONVERT ||
            op.GetOpcode() == Opcode::OP_VIEW) {
            if (op.GetIOperands().front()->GetMemoryTypeOriginal() == MemoryType::MEM_L0C &&
                op.GetOOperands().front()->GetMemoryTypeOriginal() == MemoryType::MEM_L1) {
                l0c2l1Count++;
                EXPECT_TRUE((*op.ProducerOps().begin())->GetOpcode() == Opcode::OP_A_MUL_B ||
                            (*op.ProducerOps().begin())->GetOpcode() == Opcode::OP_A_MULACC_B);
            }
        }
    }
    return l0c2l1Count;
}

static int CountMemoryPath(Function* originFunction, MemoryType from, MemoryType to)
{
    int pathCount = 0;
    for (auto& op : originFunction->Operations()) {
        if (op.GetOpcode() != Opcode::OP_ASSEMBLE && op.GetOpcode() != Opcode::OP_CONVERT &&
            op.GetOpcode() != Opcode::OP_VIEW) {
            continue;
        }
        if (op.GetIOperands().empty() || op.GetOOperands().empty()) {
            continue;
        }
        if (op.GetIOperands().front()->GetMemoryTypeOriginal() == from &&
            op.GetOOperands().front()->GetMemoryTypeOriginal() == to) {
            ++pathCount;
        }
    }
    return pathCount;
}

struct L0C2UBTestShapes {
    std::vector<int64_t> shapeA = {NUM_64, NUM_128};
    std::vector<int64_t> shapeB = {NUM_128, NUM_64};
    std::vector<int64_t> shapeC = {NUM_64, NUM_64};
};

static L0C2UBTestShapes PrepareA5Platform()
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Platform::Instance().ReloadMemoryPaths("3510");
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    return {};
}

TEST_F(LegacyAssignMemoryTypeTest, TestL0C2L1EqualShape)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shapeA1 = {NUM_64, NUM_32};
    std::vector<int64_t> shapeA2 = {NUM_128, NUM_64};
    std::vector<int64_t> shapeB1 = {NUM_32, NUM_16};
    std::vector<int64_t> shapeC2 = {NUM_128, NUM_16};
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputA1(DataType::DT_FP16, shapeA1, "A1");
        Tensor inputA2(DataType::DT_FP16, shapeA2, "A2");
        Tensor inputB1(DataType::DT_FP16, shapeB1, "B1");
        Tensor outC2(DataType::DT_FP16, shapeC2, "C2");
        SetFullTestStrategy();
        Function* originFunction = nullptr;

        config::SetBuildStatic(true);
        FUNCTION("TestL0C2L1EqualShape", {inputA1, inputB1, inputA2, outC2})
        {
            TileShape::Current().SetCubeTile({NUM_32, NUM_32}, {NUM_16, NUM_16}, {NUM_16, NUM_16});
            Tensor inputB2 = Matrix::Matmul(outC2.GetDataType(), inputA1, inputB1); // (64 * 32) @ (32 * 16) = (64 * 16)
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_32, NUM_32}, {NUM_16, NUM_16});
            outC2 = Matrix::Matmul(outC2.GetDataType(), inputA2, inputB2); // (128 * 64) @ (64 * 16) = (128 * 16)
        }

        originFunction = Program::GetInstance().GetFunctionByRawName(
            "TENSOR_TestL0C2L1EqualShape"); // Tensor_{Function名字}
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        EXPECT_EQ(CountL0c2l1Num(originFunction), 2);
    }
}

TEST_F(LegacyAssignMemoryTypeTest, TestL0C2L1ParallelDdrFallback)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shapeA1 = {NUM_64, NUM_32};
    std::vector<int64_t> shapeA2 = {NUM_128, NUM_64};
    std::vector<int64_t> shapeB1 = {NUM_32, NUM_16};
    std::vector<int64_t> shapeC1 = {NUM_64, NUM_16};
    std::vector<int64_t> shapeC2 = {NUM_128, NUM_16};
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputA1(DataType::DT_FP16, shapeA1, "A1");
        Tensor inputA2(DataType::DT_FP16, shapeA2, "A2");
        Tensor inputB1(DataType::DT_FP16, shapeB1, "B1");
        Tensor outC1(DataType::DT_FP16, shapeC1, "C1");
        Tensor outC2(DataType::DT_FP16, shapeC2, "C2");
        SetFullTestStrategy();
        Function* originFunction = nullptr;

        config::SetBuildStatic(true);
        FUNCTION("TestL0C2L1ParallelDdrFallback", {inputA1, inputB1, inputA2, outC1, outC2})
        {
            TileShape::Current().SetCubeTile({NUM_32, NUM_32}, {NUM_16, NUM_16}, {NUM_16, NUM_16});
            Tensor inputB2 = Matrix::Matmul(outC1.GetDataType(), inputA1, inputB1);
            Assemble(inputB2, {0, 0}, outC1);
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_32, NUM_32}, {NUM_16, NUM_16});
            outC2 = Matrix::Matmul(outC2.GetDataType(), inputA2, inputB2);
        }

        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_TestL0C2L1ParallelDdrFallback");
        ASSERT_NE(originFunction, nullptr) << "Function pointer is null";
        EXPECT_EQ(CountMemoryPath(originFunction, MemoryType::MEM_L0C, MemoryType::MEM_L1), 0);
        EXPECT_GE(CountMemoryPath(originFunction, MemoryType::MEM_L0C, MemoryType::MEM_DEVICE_DDR), 1);
        EXPECT_GE(CountMemoryPath(originFunction, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_L1), 1);
    }
}

TEST_F(LegacyAssignMemoryTypeTest, TestL0C2L1LargeToSmall)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shapeA1 = {NUM_64, NUM_32};
    std::vector<int64_t> shapeB1 = {NUM_32, NUM_32};
    std::vector<int64_t> shapeA2 = {NUM_128, NUM_64};
    std::vector<int64_t> shapeC2 = {NUM_128, NUM_32};
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputA1(DataType::DT_FP16, shapeA1, "A1");
        Tensor inputB1(DataType::DT_FP16, shapeB1, "B1");
        Tensor inputA2(DataType::DT_FP16, shapeA2, "A2");
        Tensor outC2(DataType::DT_FP16, shapeC2, "C2");
        SetFullTestStrategy();
        Function* originFunction = nullptr;

        config::SetBuildStatic(true);
        FUNCTION("TestL0C2L1LargeToSmall", {inputA1, inputB1, inputA2, outC2})
        {
            TileShape::Current().SetCubeTile({NUM_32, NUM_32}, {NUM_16, NUM_16}, {NUM_32, NUM_32});
            Tensor inputB2 = Matrix::Matmul(outC2.GetDataType(), inputA1, inputB1); // (64 * 32) @ (32 * 32) = (64 * 32)
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_32, NUM_32}, {NUM_16, NUM_16});
            outC2 = Matrix::Matmul(outC2.GetDataType(), inputA2, inputB2); // (128 * 64) @ (64 * 32) = (128 * 32)
        }

        originFunction = Program::GetInstance().GetFunctionByRawName(
            "TENSOR_TestL0C2L1LargeToSmall"); // Tensor_{Function名字}
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        EXPECT_EQ(CountL0c2l1Num(originFunction), 4);
    }
}

TEST_F(LegacyAssignMemoryTypeTest, TestL0C2L1SmallToLarge)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shapeB1 = {NUM_32, NUM_32};
    std::vector<int64_t> shapeA1 = {NUM_64, NUM_32};
    std::vector<int64_t> shapeA2 = {NUM_128, NUM_64};
    std::vector<int64_t> shapeC2 = {NUM_128, NUM_16};
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputB1(DataType::DT_FP16, shapeB1, "B1");
        Tensor inputA1(DataType::DT_FP16, shapeA1, "A1");
        Tensor inputA2(DataType::DT_FP16, shapeA2, "A2");
        Tensor outC2(DataType::DT_FP16, shapeC2, "C2");
        SetFullTestStrategy();
        Function* originFunction = nullptr;

        config::SetBuildStatic(true);
        FUNCTION("TestL0C2L1SmallToLarge", {inputA1, inputB1, inputA2, outC2})
        {
            TileShape::Current().SetCubeTile({NUM_32, NUM_32}, {NUM_16, NUM_16}, {NUM_16, NUM_16});
            Tensor inputB2 = Matrix::Matmul(outC2.GetDataType(), inputA1, inputB1); // (64 * 32) @ (32 * 32) = (64 * 32)
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_32, NUM_32}, {NUM_32, NUM_32});
            outC2 = Matrix::Matmul(outC2.GetDataType(), inputA2, inputB2); // (128 * 64) @ (64 * 32) = (128 * 32)
        }

        originFunction = Program::GetInstance().GetFunctionByRawName(
            "TENSOR_TestL0C2L1SmallToLarge"); // Tensor_{Function名字}
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        EXPECT_EQ(CountL0c2l1Num(originFunction), 4);
    }
}

TEST_F(LegacyAssignMemoryTypeTest, TestL0C2L1Dim0Mismatch)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shapeB1 = {NUM_32, NUM_16};
    std::vector<int64_t> shapeA1 = {NUM_64, NUM_32};
    std::vector<int64_t> shapeA2 = {NUM_128, NUM_64};
    std::vector<int64_t> shapeC1 = {NUM_128, NUM_16};
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputB1(DataType::DT_FP16, shapeB1, "B1");
        Tensor inputA1(DataType::DT_FP16, shapeA1, "A1");
        Tensor inputA2(DataType::DT_FP16, shapeA2, "A2");
        Tensor outC2(DataType::DT_FP16, shapeC1, "C2");
        SetFullTestStrategy();
        Function* originFunction = nullptr;

        config::SetBuildStatic(true);
        FUNCTION("TestL0C2L1Dim0Mismatch", {inputA1, inputB1, inputA2, outC2})
        {
            TileShape::Current().SetCubeTile({NUM_16, NUM_16}, {NUM_16, NUM_16}, {NUM_16, NUM_16});
            Tensor inputB2 = Matrix::Matmul(outC2.GetDataType(), inputA1, inputB1); // (64 * 32) @ (32 * 16) = (64 * 16)
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_32, NUM_32}, {NUM_16, NUM_16});
            outC2 = Matrix::Matmul(outC2.GetDataType(), inputA2, inputB2); // (128 * 64) @ (64 * 16) = (128 * 16)
        }

        originFunction = Program::GetInstance().GetFunctionByRawName(
            "TENSOR_TestL0C2L1Dim0Mismatch"); // Tensor_{Function名字}
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        EXPECT_EQ(CountL0c2l1Num(originFunction), 0);
    }
}

static void ConstructL0C2L1GraphWithNonImmediateValidShape(std::shared_ptr<Function>& currFunctionPtr)
{
    std::vector<int64_t> shapeA1 = {NUM_64, NUM_32};
    std::vector<int64_t> shapeB1 = {NUM_32, NUM_16};
    std::vector<int64_t> shapeC1 = {NUM_64, NUM_16};
    std::vector<int64_t> shapeA2 = {NUM_128, NUM_64};
    std::vector<int64_t> shapeC2 = {NUM_128, NUM_16};

    auto inputA1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP16, shapeA1, CreateTestConstIntVector(shapeA1));
    auto inputB1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP16, shapeB1, CreateTestConstIntVector(shapeB1));
    auto inputA2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP16, shapeA2, CreateTestConstIntVector(shapeA2));
    auto inputB2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP16, shapeC1, CreateTestConstIntVector(shapeC1));
    auto outputC2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP16, shapeC2, CreateTestConstIntVector(shapeC2));

    auto matmul1Output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP16, shapeC1,
                                                                    CreateTestConstIntVector(shapeC1));
    auto viewL1Output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP16, shapeC1, CreateTestConstIntVector(shapeC1));
    auto viewL0AOutput = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP16, shapeC1,
                                                                    CreateTestConstIntVector(shapeC1));

    auto& matmul1Op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_A_MUL_B, {inputA1, inputB1},
                                                     {matmul1Output});
    matmul1Op.opmagic = 1001;

    auto& viewL1Op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {matmul1Output}, {viewL1Output});
    auto viewL1Attr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewL1Attr->SetToType(MemoryType::MEM_L1);
    viewL1Op.SetOpAttribute(viewL1Attr);
    viewL1Op.opmagic = 1002;

    auto& viewL0AOp = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {viewL1Output},
                                                     {viewL0AOutput});
    auto viewL0AAttr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewL0AAttr->SetToType(MemoryType::MEM_L0A);
    viewL0AOp.SetOpAttribute(viewL0AAttr);
    viewL0AOp.opmagic = 1003;

    auto& matmul2Op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_A_MUL_B, {inputA2, viewL0AOutput},
                                                     {outputC2});
    matmul2Op.opmagic = 1004;

    currFunctionPtr->inCasts_.push_back(inputA1);
    currFunctionPtr->inCasts_.push_back(inputB1);
    currFunctionPtr->inCasts_.push_back(inputA2);
    currFunctionPtr->inCasts_.push_back(inputB2);
    currFunctionPtr->outCasts_.push_back(outputC2);

    std::vector<SymbolicScalar> dynValidShape;
    dynValidShape.push_back(CreateTestScalarVar("dim0"));
    dynValidShape.push_back(CreateTestScalarVar("dim1"));
    matmul1Output->UpdateDynValidShape(dynValidShape);
}

// 当tensor存在非立即数dynValidShape时，直接走L0C2L1通道进行后续矩阵乘会有精度问题，当前走DDR规避
TEST_F(LegacyAssignMemoryTypeTest, TestL0C2L1WithNonImmediateValidShape)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestL0C2L1WithNonImmediateValidShape",
                                                      "TestL0C2L1WithNonImmediateValidShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("TestL0C2L1WithNonImmediateValidShape", currFunctionPtr);

    ConstructL0C2L1GraphWithNonImmediateValidShape(currFunctionPtr);

    AssignMemoryType assignMemoryType;
    assignMemoryType.PreCheck(*currFunctionPtr);
    assignMemoryType.RunOnFunction(*currFunctionPtr);
    assignMemoryType.PostCheck(*currFunctionPtr);

    int l0c2l1Count = 0;
    int ddr2l1Count = 0;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_CONVERT || op.GetOpcode() == Opcode::OP_VIEW) {
            if (op.GetIOperands().size() > 0 && op.GetOOperands().size() > 0) {
                auto inputMem = op.GetIOperands().front()->GetMemoryTypeOriginal();
                auto outputMem = op.GetOOperands().front()->GetMemoryTypeOriginal();
                if (inputMem == MemoryType::MEM_L0C && outputMem == MemoryType::MEM_L1) {
                    l0c2l1Count++;
                }
                if (inputMem == MemoryType::MEM_DEVICE_DDR && outputMem == MemoryType::MEM_L1) {
                    ddr2l1Count++;
                }
            }
        }
    }
    EXPECT_EQ(l0c2l1Count, 0) << "Should not use L0C2L1 path when validShape is non-immediate";
    EXPECT_GT(ddr2l1Count, 0) << "Should use DDR2L1 path when validShape is non-immediate";
}

TEST_F(LegacyAssignMemoryTypeTest, TestL0C2L1UnsupportDataType)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shapeA1 = {NUM_64, NUM_32};
    std::vector<int64_t> shapeA2 = {NUM_128, NUM_64};
    std::vector<int64_t> shapeB1 = {NUM_32, NUM_16};
    std::vector<int64_t> shapeC2 = {NUM_128, NUM_16};
    PROGRAM("AssignMemoryTest")
    {
        SetFullTestStrategy();
        Function* originFunction = nullptr;
        Tensor inputA1(DataType::DT_FP32, shapeA1, "A1");
        Tensor inputA2(DataType::DT_FP32, shapeA2, "A2");
        Tensor inputB1(DataType::DT_FP32, shapeB1, "B1");
        Tensor outC2(DataType::DT_FP32, shapeC2, "C2");

        config::SetBuildStatic(true);
        FUNCTION("TestL0C2L1UnsupportDataType", {inputA1, inputB1, inputA2, outC2})
        {
            TileShape::Current().SetCubeTile({NUM_32, NUM_32}, {NUM_16, NUM_16}, {NUM_16, NUM_16});
            Tensor inputB2 = Matrix::Matmul(outC2.GetDataType(), inputA1, inputB1); // (64 * 32) @ (32 * 16) = (64 * 16)
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_64, NUM_64}, {NUM_16, NUM_16});
            outC2 = Matrix::Matmul(outC2.GetDataType(), inputA2, inputB2); // (128 * 64) @ (64 * 16) = (128 * 16)
        }

        originFunction = Program::GetInstance().GetFunctionByRawName(
            "TENSOR_TestL0C2L1UnsupportDataType"); // Tensor_{Function名字}
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        EXPECT_EQ(CountL0c2l1Num(originFunction), 0);
    }
}

TEST_F(LegacyAssignMemoryTypeTest, TestL0C2L1UnsupportDataShape)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shapeB1 = {NUM_32, NUM_16};
    std::vector<int64_t> shapeA1 = {NUM_64, NUM_32};
    std::vector<int64_t> shapeA2 = {NUM_128, NUM_64};
    std::vector<int64_t> shapeC2 = {NUM_128, NUM_16};
    PROGRAM("AssignMemoryTest")
    {
        SetFullTestStrategy();
        Function* originFunction = nullptr;
        Tensor inputB1(DataType::DT_FP16, shapeB1, "B1");
        Tensor inputA1(DataType::DT_FP16, shapeA1, "A1");
        Tensor inputA2(DataType::DT_FP16, shapeA2, "A2");
        Tensor outC2(DataType::DT_FP16, shapeC2, "C2");

        config::SetBuildStatic(true);
        FUNCTION("TestL0C2L1UnsupportDataShape", {inputA1, inputB1, inputA2, outC2})
        {
            TileShape::Current().SetCubeTile({NUM_8, NUM_8}, {NUM_16, NUM_16}, {NUM_16, NUM_16});
            Tensor inputB2 = Matrix::Matmul(outC2.GetDataType(), inputA1, inputB1); // (64 * 32) @ (32 * 16) = (64 * 16)
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_64, NUM_64}, {NUM_16, NUM_16});
            outC2 = Matrix::Matmul(outC2.GetDataType(), inputA2, inputB2); // (128 * 64) @ (64 * 16) = (128 * 16)
        }

        originFunction = Program::GetInstance().GetFunctionByRawName(
            "TENSOR_TestL0C2L1UnsupportDataShape"); // Tensor_{Function名字}
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        EXPECT_EQ(CountL0c2l1Num(originFunction), 0);
    }
}

TEST_F(LegacyAssignMemoryTypeTest, TestL0C2L1NoSupportNotMultipleCase)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shapeA1 = {NUM_64, NUM_32};
    std::vector<int64_t> shapeB1 = {NUM_32, NUM_16};
    std::vector<int64_t> shapeA2 = {NUM_128, NUM_64};
    std::vector<int64_t> shapeC2 = {NUM_128, NUM_16};
    PROGRAM("AssignMemoryTest")
    {
        SetFullTestStrategy();
        Function* originFunction = nullptr;
        Tensor inputA1(DataType::DT_FP16, shapeA1, "A1");
        Tensor inputB1(DataType::DT_FP16, shapeB1, "B1");
        Tensor inputA2(DataType::DT_FP16, shapeA2, "A2");
        Tensor outC2(DataType::DT_FP16, shapeC2, "C2");

        config::SetBuildStatic(true);
        FUNCTION("TestL0C2L1NoSupportNotMultipleCase", {inputA1, inputB1, inputA2, outC2})
        {
            TileShape::Current().SetCubeTile({NUM_48, NUM_48}, {NUM_16, NUM_16}, {NUM_16, NUM_16});
            Tensor inputB2 = Matrix::Matmul(outC2.GetDataType(), inputA1, inputB1); // (64 * 32) @ (32 * 16) = (64 * 16)
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_64, NUM_64}, {NUM_16, NUM_16});
            outC2 = Matrix::Matmul(outC2.GetDataType(), inputA2, inputB2); // (128 * 64) @ (64 * 16) = (128 * 16)
        }

        originFunction = Program::GetInstance().GetFunctionByRawName(
            "TENSOR_TestL0C2L1NoSupportNotMultipleCase"); // Tensor_{Function名字}
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        EXPECT_EQ(CountL0c2l1Num(originFunction), 0);
    }
}

TEST_F(LegacyAssignMemoryTypeTest, TestCascadingAssembleViewNoDDR2L0C)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shapeA1 = {NUM_16, NUM_32};
    std::vector<int64_t> shapeB1 = {NUM_32, NUM_64};
    std::vector<int64_t> shapeC1 = {NUM_16, NUM_64};
    std::vector<int64_t> shapeT1 = {NUM_32, NUM_64};
    std::vector<int64_t> shapeT2 = {NUM_32, NUM_32};
    std::vector<int64_t> shapeA2 = {NUM_64, NUM_32};
    std::vector<int64_t> shapeB2 = {NUM_32, NUM_16};
    std::vector<int64_t> shapeC2 = {NUM_64, NUM_16};
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputA11(DataType::DT_FP16, shapeA1, "A11");
        Tensor inputB11(DataType::DT_FP16, shapeB1, "B11");
        Tensor inputA12(DataType::DT_FP16, shapeA1, "A12");
        Tensor inputB12(DataType::DT_FP16, shapeB1, "B12");
        Tensor inputA13(DataType::DT_FP16, shapeA1, "A13");
        Tensor inputB13(DataType::DT_FP16, shapeB1, "B13");
        Tensor inputA14(DataType::DT_FP16, shapeA1, "A14");
        Tensor inputB14(DataType::DT_FP16, shapeB1, "B14");
        Tensor inputB2(DataType::DT_FP16, shapeB2, "B2");
        Tensor outC2(DataType::DT_FP16, shapeC2, "C2");
        SetFullTestStrategy();
        Function* originFunction = nullptr;

        config::SetBuildStatic(true);
        FUNCTION("TestCascadingAssembleViewNoDDR2L0C",
                 {inputA11, inputB11, inputA12, inputB12, inputA13, inputB13, inputA14, inputB14, inputB2, outC2})
        {
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_128, NUM_128}, {NUM_128, NUM_128});
            Tensor C11 = Matrix::Matmul(outC2.GetDataType(), inputA11, inputB11); // (16, 32) @ (32, 64) = (16, 64)
            Tensor C12 = Matrix::Matmul(outC2.GetDataType(), inputA12, inputB12); // (16, 32) @ (32, 64) = (16, 64)
            Tensor C13 = Matrix::Matmul(outC2.GetDataType(), inputA13, inputB13); // (16, 32) @ (32, 64) = (16, 64)
            Tensor C14 = Matrix::Matmul(outC2.GetDataType(), inputA14, inputB14); // (16, 32) @ (32, 64) = (16, 64)
            Tensor T11(DT_FP16, shapeT1, "T11");                                  // (32, 64)
            Tensor T12(DT_FP16, shapeT1, "T12");                                  // (32, 64)
            Assemble(C11, {0, 0}, T11);
            Assemble(C12, {16, 0}, T11);
            Assemble(C13, {0, 0}, T12);
            Assemble(C14, {16, 0}, T12);
            Tensor T21 = View(T11, shapeT2, {0, 0}); // (32, 32)
            Tensor T22 = View(T12, shapeT2, {0, 0}); // (32, 32)
            Tensor A2(DT_FP16, shapeA2, "A2");       // (64, 32)
            Assemble(T21, {0, 0}, A2);
            Assemble(T22, {32, 0}, A2);
            outC2 = Matrix::Matmul(outC2.GetDataType(), A2, inputB2); // (64, 32) @ (32, 16) = (64, 16)
        }
        originFunction = Program::GetInstance().GetFunctionByRawName(
            "TENSOR_TestCascadingAssembleViewNoDDR2L0C"); // Tensor_{Function名字}
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        AssignMemoryType assignMemoryType;
        EXPECT_EQ(assignMemoryType.PostCheck(*originFunction),
                  SUCCESS); // postcheck中包含对DDR到L0C的不合理通路校验，直接调用
    }
}

static void ConstructMultiDataLoadGraphBranch(ComputationalGraphBuilder& G, std::string name)
{
    G.AddTensor(DataType::DT_FP32, {NUM_128, NUM_1, NUM_128}, MemoryType::MEM_UNKNOWN, "in" + name);
    G.AddTensor(DataType::DT_FP32, {NUM_128, NUM_1, NUM_128}, MemoryType::MEM_UNKNOWN, "t1" + name);
    G.AddOp(Opcode::OP_VIEW, {"in" + name}, {"t1" + name}, "v1" + name);
    G.GetOp("v1" + name)
        ->SetOpAttribute(
            std::make_shared<ViewOpAttribute>(std::vector<int64_t>{NUM_128, NUM_1, NUM_128}, MemoryType::MEM_UNKNOWN));
    G.AddTensor(DataType::DT_FP32, {NUM_128, NUM_128}, MemoryType::MEM_UNKNOWN, "t2" + name);
    G.AddOp(Opcode::OP_RESHAPE, {"t1" + name}, {"t2" + name}, "r" + name);
    G.AddTensor(DataType::DT_FP32, {NUM_128, NUM_128}, MemoryType::MEM_UNKNOWN, "t4" + name);
    G.AddOp(Opcode::OP_VIEW, {"t2" + name}, {"t4" + name}, "v2" + name);
    G.GetOp("v2" + name)
        ->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{NUM_128, NUM_128}, MemoryType::MEM_L1));
    G.AddTensor(DataType::DT_FP32, {NUM_128, NUM_128}, MemoryType::MEM_UNKNOWN, "t5" + name);
    G.AddOp(Opcode::OP_VIEW, {"t4" + name}, {"t5" + name}, "v3" + name);
}

static void ConstructMultiDataLoadGraph(ComputationalGraphBuilder& G)
{
    ConstructMultiDataLoadGraphBranch(G, "a");
    ConstructMultiDataLoadGraphBranch(G, "b");
    G.AddTensor(DataType::DT_FP32, {NUM_128, NUM_128}, MemoryType::MEM_UNKNOWN, "t6");
    G.AddOp(Opcode::OP_A_MUL_B, {"t5a", "t5b"}, {"t6"}, "amulb");
    G.AddTensor(DataType::DT_FP32, {NUM_128, NUM_128}, MemoryType::MEM_UNKNOWN, "t3b");
    G.AddOp(Opcode::OP_MUL, {"t2b", "t2b"}, {"t3b"}, "mulb");
    G.GetOp("v3a")->SetOpAttribute(
        std::make_shared<ViewOpAttribute>(std::vector<int64_t>{NUM_128, NUM_128}, MemoryType::MEM_L0A));
    G.GetOp("v3b")->SetOpAttribute(
        std::make_shared<ViewOpAttribute>(std::vector<int64_t>{NUM_128, NUM_128}, MemoryType::MEM_L0B));
}

static void MultiDataLoadCheck(Function* func)
{
    for (const auto& op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            EXPECT_TRUE(op.iOperand.front()->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR);
        }
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            EXPECT_FALSE(op.iOperand.front()->GetMemoryTypeOriginal() == MemoryType::MEM_UB &&
                         op.oOperand.front()->GetMemoryTypeOriginal() == MemoryType::MEM_L1);
        }
    }
}
TEST_F(LegacyAssignMemoryTypeTest, TestMultiDataLoad)
{
    ComputationalGraphBuilder G;
    ConstructMultiDataLoadGraph(G);
    Function* func = G.GetFunction();
    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.PostCheck(*func), FAILED);
    EXPECT_EQ(assignMemoryType.RunOnFunction(*func), SUCCESS);
    EXPECT_EQ(assignMemoryType.PostCheck(*func), SUCCESS);
    MultiDataLoadCheck(func);
}
TEST_F(LegacyAssignMemoryTypeTest, TestMultiDataLoad1)
{
    ComputationalGraphBuilder G;
    ConstructMultiDataLoadGraph(G);
    G.GetOp("rb")->SetOpCode(Opcode::OP_ADDS);
    Function* func = G.GetFunction();
    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.PostCheck(*func), FAILED);
    EXPECT_EQ(assignMemoryType.RunOnFunction(*func), SUCCESS);
    EXPECT_EQ(assignMemoryType.PostCheck(*func), SUCCESS);
    MultiDataLoadCheck(func);
}
TEST_F(LegacyAssignMemoryTypeTest, TestMultiDataLoad2)
{
    ComputationalGraphBuilder G;
    ConstructMultiDataLoadGraph(G);
    Shape s{NUM_128, NUM_128};
    G.GetOp("rb")->SetOpCode(Opcode::OP_A_MUL_B);
    G.AddTensor(DataType::DT_FP32, s, MemoryType::MEM_UNKNOWN, "inb2");
    G.AddTensor(DataType::DT_FP32, s, MemoryType::MEM_UNKNOWN, "t1b2");
    G.AddOp(Opcode::OP_VIEW, {"inb2"}, {"t1b2"}, "v1b2");
    G.GetOp("v1b2")->SetOpAttribute(std::make_shared<ViewOpAttribute>(s, MemoryType::MEM_L1));
    G.AddTensor(DataType::DT_FP32, s, MemoryType::MEM_UNKNOWN, "t2b2");
    G.AddOp(Opcode::OP_VIEW, {"t1b2"}, {"t2b2"}, "v2b2");
    G.GetOp("v2b2")->SetOpAttribute(std::make_shared<ViewOpAttribute>(s, MemoryType::MEM_L0A));
    G.GetTensor("t2b2")->AddConsumer(G.GetOp("rb"));
    G.GetOp("rb")->iOperand = {G.GetTensor("t1b"), G.GetTensor("t2b2")};
    // rb b
    G.GetTensor("inb")->shape = s;
    G.GetTensor("inb")->tensor->rawshape = s;
    G.GetTensor("t1b")->shape = s;
    G.GetTensor("t1b")->tensor->rawshape = s;
    G.AddTensor(DataType::DT_FP32, s, MemoryType::MEM_UNKNOWN, "t2b22");
    G.GetOp("v1b")->ReplaceInput(G.GetTensor("t2b22"), G.GetTensor("inb"));
    G.GetTensor("inb")->RemoveConsumer(G.GetOp("v1b"));
    G.AddOp(Opcode::OP_VIEW, {"inb"}, {"t2b22"}, "v2b22");
    G.GetOp("v2b22")->SetOpAttribute(std::make_shared<ViewOpAttribute>(s, MemoryType::MEM_L1));
    G.GetOp("v1b")->SetOpAttribute(std::make_shared<ViewOpAttribute>(s, MemoryType::MEM_L0B));

    Function* func = G.GetFunction();
    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.PostCheck(*func), FAILED);
    EXPECT_EQ(assignMemoryType.RunOnFunction(*func), SUCCESS);
    EXPECT_EQ(assignMemoryType.PostCheck(*func), SUCCESS);
    MultiDataLoadCheck(func);
}

TEST_F(LegacyAssignMemoryTypeTest, TestMatmulL0COutputToL1ToL0A)
{
    /*
     * Before:
     *   A_MUL_B1(l0a_in, l0b_in) -> l0c_out1(MEM_L0C) -> L1_TO_L0A -> l0a_mid(MEM_L0A) -> A_MUL_B2 -> l0c_out2
     *
     * After:
     *   A_MUL_B1 -> l0c_out1(MEM_L0C) -> OP_CONVERT(L0C->L1) -> tensorL1(MEM_L1) -> L1_TO_L0A -> l0a_mid(MEM_L0A) ->
     * A_MUL_B2 -> l0c_out2
     */
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP16, {NUM_16, NUM_16}, MemoryType::MEM_L0A, "l0a_in");
    G.AddTensor(DataType::DT_FP16, {NUM_16, NUM_16}, MemoryType::MEM_L0B, "l0b_in");
    G.AddTensor(DataType::DT_FP16, {NUM_16, NUM_16}, MemoryType::MEM_L0C, "l0c_out1");
    G.AddTensor(DataType::DT_FP16, {NUM_16, NUM_16}, MemoryType::MEM_L0A, "l0a_mid");
    G.AddTensor(DataType::DT_FP16, {NUM_16, NUM_16}, MemoryType::MEM_L0B, "l0b_in2");
    G.AddTensor(DataType::DT_FP16, {NUM_16, NUM_16}, MemoryType::MEM_L0C, "l0c_out2");

    G.AddOp(Opcode::OP_A_MUL_B, {"l0a_in", "l0b_in"}, {"l0c_out1"}, "matmul1");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l0c_out1"}, {"l0a_mid"}, "l1_to_l0a");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0a_mid", "l0b_in2"}, {"l0c_out2"}, "matmul2");

    constexpr int producerScopeId = 42;
    G.GetOp("matmul1")->SetScopeId(producerScopeId);

    G.SetInCast({"l0a_in", "l0b_in", "l0b_in2"});
    G.SetOutCast({"l0c_out2"});

    Function* func = G.GetFunction();
    size_t beforeOpCount = func->Operations().size();
    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.RunOnFunction(*func), SUCCESS);
    EXPECT_EQ(assignMemoryType.PostCheck(*func), SUCCESS);
    auto opList = func->Operations();
    int convertNum = 0;
    bool foundL1ToL0A = false;
    for (const auto& op : opList) {
        if (op.GetOpcode() == Opcode::OP_CONVERT) {
            convertNum++;
            auto input = op.GetIOperands().front();
            auto output = op.GetOOperands().front();
            EXPECT_EQ(input->GetMemoryTypeOriginal(), MemoryType::MEM_L0C) << "convert input should be MEM_L0C";
            EXPECT_EQ(output->GetMemoryTypeOriginal(), MemoryType::MEM_L1) << "convert output should be MEM_L1";
            EXPECT_EQ(op.GetScopeId(), producerScopeId) << "convert should inherit scopeId from its producer (matmul1)";
        }
        if (op.GetOpcode() == Opcode::OP_L1_TO_L0A) {
            foundL1ToL0A = true;
            auto l1ToL0AInput = op.GetIOperands().front();
            EXPECT_EQ(l1ToL0AInput->GetMemoryTypeOriginal(), MemoryType::MEM_L1)
                << "L1_TO_L0A input should be reconnected to MEM_L1 tensor (convert output)";
        }
    }
    EXPECT_EQ(convertNum, 1) << "should insert exactly 1 OP_CONVERT";
    EXPECT_TRUE(foundL1ToL0A) << "L1_TO_L0A should still exist after insertion";
    EXPECT_EQ(opList.size(), beforeOpCount + 1) << "should have exactly 1 more op after insertion";
}

TEST_F(LegacyAssignMemoryTypeTest, TestAmulBInputInvalidProducer)
{
    ComputationalGraphBuilder G;
    Shape s{NUM_128, NUM_128};
    G.AddTensor(DataType::DT_FP32, s, MemoryType::MEM_DEVICE_DDR, "input");
    G.AddTensor(DataType::DT_FP32, s, MemoryType::MEM_DEVICE_DDR, "temp");
    G.AddTensor(DataType::DT_FP32, s, MemoryType::MEM_DEVICE_DDR, "output");
    G.AddOp(Opcode::OP_ADD, {"input"}, {"temp"}, "add_op");
    G.AddOp(Opcode::OP_A_MUL_B, {"temp"}, {"output"}, "a_mul_b_op");

    Function* func = G.GetFunction();
    AssignMemoryType assignMemoryType;

    EXPECT_EQ(assignMemoryType.PreCheck(*func), FAILED);
}

TEST_F(LegacyAssignMemoryTypeTest, TestConvertScopeId)
{
    ComputationalGraphBuilder G;
    Shape s1{NUM_16, NUM_128};
    Shape s{NUM_16, NUM_16};
    Offset o1{0, 112};
    Offset o2{0, 0};
    G.AddTensor(DataType::DT_FP32, s1, MemoryType::MEM_UNKNOWN, "input");
    G.AddTensor(DataType::DT_FP32, s1, MemoryType::MEM_UNKNOWN, "t1");
    G.AddOp(Opcode::OP_VIEW, {"input"}, {"t1"}, "view1");
    G.GetOp("view1")->SetOpAttribute(std::make_shared<ViewOpAttribute>(o2, MemoryType::MEM_UNKNOWN));

    G.AddTensor(DataType::DT_FP32, s1, MemoryType::MEM_UNKNOWN, "t11");
    G.AddOp(Opcode::OP_ADDS, {"t1"}, {"t11"}, "adds1");

    G.AddTensor(DataType::DT_FP32, s, MemoryType::MEM_UNKNOWN, "t12");
    G.AddOp(Opcode::OP_VIEW, {"t11"}, {"t12"}, "view12");
    G.GetOp("view12")->SetOpAttribute(std::make_shared<ViewOpAttribute>(o1, MemoryType::MEM_UNKNOWN));

    G.AddTensor(DataType::DT_FP32, s, MemoryType::MEM_UNKNOWN, "t2");
    G.AddOp(Opcode::OP_ADDS, {"t12"}, {"t2"}, "adds");

    G.AddTensor(DataType::DT_FP32, s, MemoryType::MEM_UNKNOWN, "t3");
    G.AddOp(Opcode::OP_VIEW, {"t2"}, {"t3"}, "view2");
    G.GetOp("view2")->SetOpAttribute(std::make_shared<ViewOpAttribute>(o2, MemoryType::MEM_L1));

    G.AddTensor(DataType::DT_FP32, s, MemoryType::MEM_UNKNOWN, "t4");
    G.AddOp(Opcode::OP_VIEW, {"t3"}, {"t4"}, "view3");
    G.GetOp("view3")->SetOpAttribute(std::make_shared<ViewOpAttribute>(o2, MemoryType::MEM_L0A));

    G.AddTensor(DataType::DT_FP32, s, MemoryType::MEM_UNKNOWN, "input_b");
    G.AddTensor(DataType::DT_FP32, s, MemoryType::MEM_UNKNOWN, "t5");
    G.AddOp(Opcode::OP_VIEW, {"input_b"}, {"t5"}, "view4");
    G.GetOp("view4")->SetOpAttribute(std::make_shared<ViewOpAttribute>(o2, MemoryType::MEM_L1));

    G.AddTensor(DataType::DT_FP32, s, MemoryType::MEM_UNKNOWN, "t6");
    G.AddOp(Opcode::OP_VIEW, {"t5"}, {"t6"}, "view5");
    G.GetOp("view5")->SetOpAttribute(std::make_shared<ViewOpAttribute>(o2, MemoryType::MEM_L0B));

    G.AddTensor(DataType::DT_FP32, s, MemoryType::MEM_UNKNOWN, "output");
    G.AddOp(Opcode::OP_A_MUL_B, {"t4", "t6"}, {"output"}, "a_mul_b");

    constexpr int scopeId1 = 1;
    G.GetOp("adds")->SetScopeId(scopeId1);
    G.GetOp("adds1")->SetScopeId(scopeId1);

    G.SetInCast({"input", "input_b"});
    G.SetOutCast({"output"});

    Function* func = G.GetFunction();
    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.RunOnFunction(*func), SUCCESS);
    EXPECT_EQ(assignMemoryType.PostCheck(*func), SUCCESS);

    for (auto& op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            EXPECT_EQ(op.GetScopeId(), scopeId1);
        }
    }
}

TEST_F(LegacyAssignMemoryTypeTest, TestTobeMapOrdering)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shape = {NUM_256, NUM_128};
    std::vector<int64_t> shape1 = {NUM_128, NUM_64};
    std::vector<int64_t> shape2 = {NUM_64, NUM_256};
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputA(DataType::DT_FP32, shape, "A");
        Tensor inputB(DataType::DT_FP32, shape, "B");
        Tensor weight(DataType::DT_FP32, shape1, "weight");
        Tensor out(DataType::DT_FP32, shape2, "output");
        SetFullTestStrategy();
        config::SetBuildStatic(true);
        FUNCTION("TestTobeMapOrdering", {inputA, inputB, weight, out})
        {
            TileShape::Current().SetCubeTile({NUM_256, NUM_256}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor mmRes = Matrix::Matmul(out.GetDataType(), inputA, weight);
            Tensor reshapeRes = Reshape(mmRes, shape2);
            TileShape::Current().SetVecTile(NUM_256, NUM_256);
            Tensor add1Out = Add(reshapeRes, Element(DataType::DT_FP32, 1.0));
            Tensor add2Out = Add(reshapeRes, Element(DataType::DT_FP32, 2.0));
            Tensor expOut = Exp(reshapeRes);
            Tensor out1 = Add(add2Out, expOut);
            out = Add(out1, add1Out);
        }
        Function* originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_TestTobeMapOrdering");
        ASSERT_NE(originFunction, nullptr) << "Function pointer is null";
        std::vector<std::pair<uint64_t, uint64_t>> tensorOpMagicPairs;
        for (const auto& op : originFunction->Operations()) {
            if (op.GetOpcode() != Opcode::OP_VIEW) {
                continue;
            }
            auto output = op.GetOOperands().front();
            auto consumer = *output->GetConsumers().begin();
            if (output->GetMemoryTypeOriginal() != MemoryType::MEM_UB) {
                continue;
            }
            tensorOpMagicPairs.emplace_back(output->GetMagic(), consumer->GetOpMagic());
        }
        for (size_t i = 0; i < tensorOpMagicPairs.size(); ++i) {
            for (size_t j = i + 1; j < tensorOpMagicPairs.size(); ++j) {
                if (tensorOpMagicPairs[i].second < tensorOpMagicPairs[j].second) {
                    ASSERT_LE(tensorOpMagicPairs[i].first, tensorOpMagicPairs[j].first)
                        << "TobeMap ordering violation: OpMagic " << tensorOpMagicPairs[i].second << " (TensorMagic "
                        << tensorOpMagicPairs[i].first << ") < OpMagic " << tensorOpMagicPairs[j].second
                        << " (TensorMagic " << tensorOpMagicPairs[j].first << ")";
                }
            }
        }
    }
}

TEST_F(LegacyAssignMemoryTypeTest, TestOverSizeUb)
{
    config::SetHostConfig(KEY_STRATEGY, "TestOverSizeUb");
    std::vector<int64_t> shape = {NUM_256, NUM_256};
    PROGRAM("TestOverSizeUb")
    {
        Tensor input1(DataType::DT_FP32, shape, "input1");
        Tensor input2(DataType::DT_FP32, shape, "input2");
        Tensor input3(DataType::DT_FP32, shape, "input3");
        Tensor input4(DataType::DT_FP32, shape, "input4");
        Tensor output(DataType::DT_FP32, shape, "output");
        SetFullTestStrategy();
        config::SetBuildStatic(true);
        FUNCTION("TestOverSizeUb", {input1, input2, input3, input4, output})
        {
            TileShape::Current().SetCubeTile({NUM_256, NUM_256}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor mmRes = Matrix::Matmul(input4.GetDataType(), input1, input2);
            Assemble(mmRes, {0, 0}, input4);
            TileShape::Current().SetVecTile(NUM_256, NUM_256);
            output = Add(input4, input3);
        }
        Function* originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_TestOverSizeUb");
        ASSERT_NE(originFunction, nullptr) << "Function pointer is null";
        int beforeViewNum = 0;
        for (const auto& op : originFunction->Operations()) {
            if (op.GetOpcode() == Opcode::OP_VIEW) {
                beforeViewNum++;
            }
        }
        AssignMemoryType assignMemoryType;
        EXPECT_EQ(assignMemoryType.RunOnFunction(*originFunction), SUCCESS);
        int afterViewNum = 0;
        for (const auto& op : originFunction->Operations()) {
            if (op.GetOpcode() == Opcode::OP_VIEW) {
                afterViewNum++;
            }
        }
        EXPECT_EQ(afterViewNum, beforeViewNum + 2);
    }
}

TEST_F(LegacyAssignMemoryTypeTest, OversizedViewInputRequirementFallback)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> fullShape = {NUM_1024, NUM_1024};
    std::vector<int64_t> viewShape = {NUM_1024, NUM_512};
    PROGRAM("OversizedViewInputRequirementFallback")
    {
        Tensor input(DataType::DT_FP32, fullShape, "input");
        Tensor output(DataType::DT_FP32, viewShape, "output");
        SetFullTestStrategy();
        Function* originFunction = nullptr;
        config::SetBuildStatic(true);
        FUNCTION("OversizedViewInputRequirementFallback", {input, output})
        {
            TileShape::Current().SetVecTile(NUM_1024, NUM_1024);
            Tensor expOut = Exp(input);
            Tensor viewOut = View(expOut, viewShape, {0, 0});
            output = Exp(viewOut);
        }
        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_OversizedViewInputRequirementFallback");
        ASSERT_NE(originFunction, nullptr) << "Function pointer is null";
        EXPECT_FALSE(HasViewMemoryPath(originFunction, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR))
            << "Oversized view should not represent UB->DDR movement.";
    }
}

TEST_F(LegacyAssignMemoryTypeTest, TestL0C2UBSmallToLarge)
{
    auto shapes = PrepareA5Platform();
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputA(DataType::DT_FP16, shapes.shapeA, "A");
        Tensor inputB(DataType::DT_FP16, shapes.shapeB, "B");
        Tensor inputC(DataType::DT_FP32, shapes.shapeC, "C");
        Tensor out(DataType::DT_FP32, shapes.shapeC, "output");
        SetFullTestStrategy();
        Function* originFunction = nullptr;
        config::SetBuildStatic(true);
        FUNCTION("TestL0C2UBSmallToLarge", {inputA, inputB, inputC, out})
        {
            // 设置 Cube tile shape，使 matmul 输出 L0C
            TileShape::Current().SetCubeTile({NUM_32, NUM_32}, {NUM_64, NUM_64}, {NUM_64, NUM_64});
            Tensor ab = Matrix::Matmul(out.GetDataType(), inputA, inputB);
            // 设置 Vec tile shape，使后续 Vector 操作需要 UB 输入
            TileShape::Current().SetVecTile(NUM_64, NUM_64);
            Tensor result = Add(ab, inputC);
            out = result;
        }
        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_TestL0C2UBSmallToLarge");
        ASSERT_NE(originFunction, nullptr) << "Function pointer is null";
        // 验证存在 L0C -> UB 的转换
        bool hasL0C2UB = false;
        for (auto& op : originFunction->Operations()) {
            // 检查 Convert: L0C -> UB
            if (op.GetOpcode() == Opcode::OP_CONVERT) {
                auto input = op.GetIOperands().front();
                auto output = op.GetOOperands().front();
                if (input->GetMemoryTypeOriginal() == MEM_L0C && output->GetMemoryTypeOriginal() == MEM_UB) {
                    hasL0C2UB = true;
                }
            }
            // 或者检查 Assemble 输出为 UB（小搬大场景）
            if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
                auto output = op.GetOOperands().front();
                if (output->GetMemoryTypeOriginal() == MEM_UB) {
                    hasL0C2UB = true;
                }
            }
        }
        EXPECT_TRUE(hasL0C2UB) << "Should have L0C->UB data path for matmul then add";
    }
    // 恢复平台设置
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
    Platform::Instance().ReloadMemoryPaths("2201");
}

TEST_F(LegacyAssignMemoryTypeTest, TestL0C2UBParallelDdrFallback)
{
    auto shapes = PrepareA5Platform();
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputA(DataType::DT_FP16, shapes.shapeA, "A");
        Tensor inputB(DataType::DT_FP16, shapes.shapeB, "B");
        Tensor inputC(DataType::DT_FP32, shapes.shapeC, "C");
        Tensor ddrOut(DataType::DT_FP32, shapes.shapeC, "DdrOut");
        Tensor out(DataType::DT_FP32, shapes.shapeC, "output");
        SetFullTestStrategy();
        Function* originFunction = nullptr;

        config::SetBuildStatic(true);
        FUNCTION("TestL0C2UBParallelDdrFallback", {inputA, inputB, inputC, ddrOut, out})
        {
            TileShape::Current().SetCubeTile({NUM_32, NUM_32}, {NUM_64, NUM_64}, {NUM_64, NUM_64});
            Tensor ab = Matrix::Matmul(out.GetDataType(), inputA, inputB);
            Assemble(ab, {0, 0}, ddrOut);
            TileShape::Current().SetVecTile(NUM_64, NUM_64);
            out = Add(ab, inputC);
        }

        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_TestL0C2UBParallelDdrFallback");
        ASSERT_NE(originFunction, nullptr) << "Function pointer is null";
        EXPECT_EQ(CountMemoryPath(originFunction, MemoryType::MEM_L0C, MemoryType::MEM_UB), 0);
        EXPECT_GE(CountMemoryPath(originFunction, MemoryType::MEM_L0C, MemoryType::MEM_DEVICE_DDR), 1);
        EXPECT_GE(CountMemoryPath(originFunction, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB), 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
    Platform::Instance().ReloadMemoryPaths("2201");
}

TEST_F(LegacyAssignMemoryTypeTest, TestUB2L1SmallToLarge)
{
    // 设置 A5 平台
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Platform::Instance().ReloadMemoryPaths("3510");
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    // m=32, k=64, n=64
    std::vector<int64_t> shapeA = {32, 64};
    std::vector<int64_t> shapeB = {64, 64};
    std::vector<int64_t> shapeC = {32, 64};
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputA1(DataType::DT_FP32, shapeA, "A1");
        Tensor inputA2(DataType::DT_FP32, shapeA, "A2");
        Tensor inputB1(DataType::DT_FP32, shapeB, "B1");
        Tensor inputB2(DataType::DT_FP32, shapeB, "B2");
        Tensor out(DataType::DT_FP32, shapeC, "output");
        SetFullTestStrategy();
        Function* originFunction = nullptr;

        config::SetBuildStatic(true);
        FUNCTION("TestUB2L1SmallToLarge", {inputA1, inputA2, inputB1, inputB2, out})
        {
            // 1. Vector 操作: Add 输出 UB
            TileShape::Current().SetVecTile(16, 32); // vec_tile_shapes = (16, 32)
            Tensor add1 = Add(inputA1, inputA2);     // (32, 64) UB
            Tensor add2 = Add(inputB1, inputB2);     // (64, 64) UB

            // 2. Cube 操作: MatMul 需要 L1 输入，触发 UB->L1 转换
            TileShape::Current().SetCubeTile({32, 32}, {64, 64},
                                             {64, 64}); // cube_tile_shapes = ([32,32], [64,64], [64,64])
            Tensor result = Matrix::Matmul(out.GetDataType(), add1, add2); // (32, 64) @ (64, 64) = (32, 64)
            out = result;
        }
        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_TestUB2L1SmallToLarge");
        ASSERT_NE(originFunction, nullptr) << "Function pointer is null";
        // 验证存在 UB->L1 转换
        bool hasUB2L1 = false;
        for (auto& op : originFunction->Operations()) {
            // 检查 Assemble 的输出是否为 L1（小搬大场景）
            if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
                auto output = op.GetOOperands().front();
                if (output->GetMemoryTypeOriginal() == MEM_L1) {
                    hasUB2L1 = true;
                }
            }
            // 检查 Convert: UB -> L1
            if (op.GetOpcode() == Opcode::OP_CONVERT) {
                auto input = op.GetIOperands().front();
                auto output = op.GetOOperands().front();
                if (input->GetMemoryTypeOriginal() == MEM_UB && output->GetMemoryTypeOriginal() == MEM_L1) {
                    hasUB2L1 = true;
                }
            }
        }
        EXPECT_TRUE(hasUB2L1) << "Should have UB->L1 data path for add_then_matmul";
    }
    // 恢复平台设置
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
    Platform::Instance().ReloadMemoryPaths("2201");
}

// 负向用例（A5 + view/assemble legacy 路径）：UB2L1 不支持的 dtype（int32）应降级为经 DDR 搬运，不再生成 UB->L1 直连
TEST_F(LegacyAssignMemoryTypeTest, TestUB2L1UnsupportDataTypeDdrFallback)
{
    // 设置 A5 平台
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Platform::Instance().ReloadMemoryPaths("3510");
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shapeA = {NUM_32, NUM_64};
    std::vector<int64_t> shapeB = {NUM_64, NUM_64};
    std::vector<int64_t> shapeC = {NUM_32, NUM_64};
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputA1(DataType::DT_INT32, shapeA, "A1");
        Tensor inputA2(DataType::DT_INT32, shapeA, "A2");
        Tensor inputB1(DataType::DT_INT32, shapeB, "B1");
        Tensor inputB2(DataType::DT_INT32, shapeB, "B2");
        Tensor out(DataType::DT_INT32, shapeC, "output");
        SetFullTestStrategy();
        Function* originFunction = nullptr;

        config::SetBuildStatic(true);
        FUNCTION("TestUB2L1UnsupportDataTypeDdrFallback", {inputA1, inputA2, inputB1, inputB2, out})
        {
            // 1. Vector 操作: Add 输出 UB（int32 不支持 UB2L1 直连）
            TileShape::Current().SetVecTile(NUM_16, NUM_32);
            Tensor add1 = Add(inputA1, inputA2); // (32, 64) UB
            Tensor add2 = Add(inputB1, inputB2); // (64, 64) UB

            // 2. Cube 操作: MatMul 需要 L1 输入，int32 不支持 UB->L1 直连，应经 DDR 中转
            TileShape::Current().SetCubeTile({NUM_32, NUM_32}, {NUM_64, NUM_64}, {NUM_64, NUM_64});
            Tensor result = Matrix::Matmul(out.GetDataType(), add1, add2); // (32, 64) @ (64, 64) = (32, 64)
            out = result;
        }
        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_TestUB2L1UnsupportDataTypeDdrFallback");
        ASSERT_NE(originFunction, nullptr) << "Function pointer is null";
        // 验证不存在 UB->L1 直连转换，且经 DDR 中转搬运
        EXPECT_EQ(CountMemoryPath(originFunction, MemoryType::MEM_UB, MemoryType::MEM_L1), 0);
        EXPECT_GE(CountMemoryPath(originFunction, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_L1), 1);
    }
    // 恢复平台设置
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
    Platform::Instance().ReloadMemoryPaths("2201");
}

// 回归用例（2201/A2A3）：platforminfo.ini 中 2201 无 UB->L1 直连边，任何 dtype 都不允许生成 UB->L1 直连
TEST_F(LegacyAssignMemoryTypeTest, TestUB2L1On2201NoDirectPath)
{
    // 设置 2201 平台
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_2201);
    Platform::Instance().ReloadMemoryPaths("2201");
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shapeA = {NUM_32, NUM_64};
    std::vector<int64_t> shapeB = {NUM_64, NUM_64};
    std::vector<int64_t> shapeC = {NUM_32, NUM_64};
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputA1(DataType::DT_FP16, shapeA, "A1");
        Tensor inputA2(DataType::DT_FP16, shapeA, "A2");
        Tensor inputB1(DataType::DT_FP16, shapeB, "B1");
        Tensor inputB2(DataType::DT_FP16, shapeB, "B2");
        Tensor out(DataType::DT_FP16, shapeC, "output");
        SetFullTestStrategy();
        Function* originFunction = nullptr;

        config::SetBuildStatic(true);
        FUNCTION("TestUB2L1On2201NoDirectPath", {inputA1, inputA2, inputB1, inputB2, out})
        {
            // 1. Vector 操作: Add 输出 UB（fp16 属于 UB2L1 支持 dtype，但 2201 无直连边）
            TileShape::Current().SetVecTile(NUM_16, NUM_32);
            Tensor add1 = Add(inputA1, inputA2); // (32, 64) UB
            Tensor add2 = Add(inputB1, inputB2); // (64, 64) UB

            // 2. Cube 操作: MatMul 需要 L1 输入，2201 无 UB->L1 直连边，应经 DDR 中转
            TileShape::Current().SetCubeTile({NUM_32, NUM_32}, {NUM_64, NUM_64}, {NUM_64, NUM_64});
            Tensor result = Matrix::Matmul(out.GetDataType(), add1, add2); // (32, 64) @ (64, 64) = (32, 64)
            out = result;
        }
        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_TestUB2L1On2201NoDirectPath");
        ASSERT_NE(originFunction, nullptr) << "Function pointer is null";
        // 验证不存在 UB->L1 直连转换
        EXPECT_EQ(CountMemoryPath(originFunction, MemoryType::MEM_UB, MemoryType::MEM_L1), 0);
        EXPECT_GE(CountMemoryPath(originFunction, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_L1), 1);
    }
    // 恢复平台设置
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
    Platform::Instance().ReloadMemoryPaths("2201");
}

TEST_F(LegacyAssignMemoryTypeTest, TestHf8CastRightMatmulUB2L1)
{
    // FP32 -> HF8 cast is only supported on A5, and this case verifies the A5 UB->L1 path.
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Platform::Instance().ReloadMemoryPaths("3510");
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> castInputShape = {NUM_64, NUM_1};
    std::vector<int64_t> leftMatrixShape = {NUM_8, NUM_1};
    std::vector<int64_t> outputShape = {NUM_8, NUM_64};
    PROGRAM("AssignMemoryTest")
    {
        Tensor castInput(DataType::DT_FP32, castInputShape, "castInput");
        Tensor leftMatrix(DataType::DT_HF8, leftMatrixShape, "leftMatrix");
        Tensor out(DataType::DT_FP32, outputShape, "output");
        SetFullTestStrategy();
        Function* originFunction = nullptr;
        config::SetBuildStatic(true);
        FUNCTION("TestHf8CastRightMatmulUB2L1", {castInput, leftMatrix, out})
        {
            TileShape::Current().SetVecTile(NUM_16, NUM_1);
            Tensor castRightMatrix = Cast(castInput, DataType::DT_HF8);
            // HF8 kL0=1 fails frontend 32-byte alignment checks before AssignMemoryType.
            TileShape::Current().SetCubeTile({NUM_8, NUM_8}, {NUM_32, NUM_32}, {NUM_32, NUM_32});
            out = Matrix::Matmul(out.GetDataType(), leftMatrix, castRightMatrix, false, true);
        }
        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_TestHf8CastRightMatmulUB2L1");
        ASSERT_NE(originFunction, nullptr) << "Function pointer is null";

        bool hasUb2L1 = false;
        for (auto& op : originFunction->Operations()) {
            if (op.GetOpcode() != Opcode::OP_CONVERT && op.GetOpcode() != Opcode::OP_VIEW &&
                op.GetOpcode() != Opcode::OP_ASSEMBLE) {
                continue;
            }
            if (op.GetIOperands().empty() || op.GetOOperands().empty()) {
                continue;
            }
            auto input = op.GetIOperands().front();
            auto output = op.GetOOperands().front();
            if (input->GetMemoryTypeOriginal() == MemoryType::MEM_UB &&
                output->GetMemoryTypeOriginal() == MemoryType::MEM_L1) {
                hasUb2L1 = true;
            }
        }
        EXPECT_TRUE(hasUb2L1) << "Cast result should use UB->L1 before feeding right matrix into matmul.";
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
    Platform::Instance().ReloadMemoryPaths("2201");
}

// 负向约束用例（A5 + view/assemble legacy 路径）：MX matmul 的 L1 拷入带 MX_PADDING_MODE 属性且
// K 轴（k=1）未按 64 对齐，UB->L1 直连不执行 MX K 向补齐，cast 产生的 UB 输入应回退经 DDR 搬运
TEST_F(LegacyAssignMemoryTypeTest, TestMXMatmulUB2L1DdrFallback)
{
    // FP32 -> FP8E4M3 cast 后作为 MX matmul 右矩阵（B 转置，n=64, k=1），MX 要求 kL0 按 64 对齐
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Platform::Instance().ReloadMemoryPaths("3510");
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> castInputShape = {NUM_64, NUM_1};
    std::vector<int64_t> leftMatrixShape = {NUM_8, NUM_1};
    std::vector<int64_t> leftScaleShape = {NUM_8, NUM_1, NUM_2};
    std::vector<int64_t> rightScaleShape = {NUM_1, NUM_64, NUM_2};
    std::vector<int64_t> outputShape = {NUM_8, NUM_64};
    PROGRAM("AssignMemoryTest")
    {
        Tensor castInput(DataType::DT_FP32, castInputShape, "castInput");
        Tensor leftMatrix(DataType::DT_FP8E4M3, leftMatrixShape, "leftMatrix");
        Tensor leftScale(DataType::DT_FP8E8M0, leftScaleShape, "leftScale");
        Tensor rightScale(DataType::DT_FP8E8M0, rightScaleShape, "rightScale");
        Tensor out(DataType::DT_FP32, outputShape, "output");
        SetFullTestStrategy();
        Function* originFunction = nullptr;
        config::SetBuildStatic(true);
        FUNCTION("TestMXMatmulUB2L1DdrFallback", {castInput, leftMatrix, leftScale, rightScale, out})
        {
            TileShape::Current().SetVecTile(NUM_16, NUM_1);
            Tensor castRightMatrix = Cast(castInput, DataType::DT_FP8E4M3);
            TileShape::Current().SetCubeTile({NUM_8, NUM_64}, {NUM_64, NUM_64}, {NUM_64, NUM_64});
            out = Matrix::MatmulMX(out.GetDataType(), leftMatrix, leftScale, castRightMatrix, rightScale, false, false,
                                   true, false, false);
        }
        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_TestMXMatmulUB2L1DdrFallback");
        ASSERT_NE(originFunction, nullptr) << "Function pointer is null";
        EXPECT_EQ(CountMemoryPath(originFunction, MemoryType::MEM_UB, MemoryType::MEM_L1), 0)
            << "MX_PADDING_MODE copy-in with K not 64-aligned must not use UB->L1 direct path";
        EXPECT_GE(CountMemoryPath(originFunction, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_L1), 1)
            << "MX_PADDING_MODE copy-in with K not 64-aligned should fall back to DDR transit path";
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
    Platform::Instance().ReloadMemoryPaths("2201");
}

// 正向用例（A5 + view/assemble legacy 路径）：MX matmul 的 K 轴（k=64）concrete 且按 64 对齐时
// 无需 K 向补齐，cast 产生的 UB 输入允许保持 UB->L1 直连
TEST_F(LegacyAssignMemoryTypeTest, TestMXMatmulUB2L1KAlignedDirectPath)
{
    // FP32 -> FP8E4M3 cast 后作为 MX matmul 右矩阵（B 转置，n=64, k=64）
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Platform::Instance().ReloadMemoryPaths("3510");
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> castInputShape = {NUM_64, NUM_64};
    std::vector<int64_t> leftMatrixShape = {NUM_8, NUM_64};
    std::vector<int64_t> leftScaleShape = {NUM_8, NUM_1, NUM_2};
    std::vector<int64_t> rightScaleShape = {NUM_1, NUM_64, NUM_2};
    std::vector<int64_t> outputShape = {NUM_8, NUM_64};
    PROGRAM("AssignMemoryTest")
    {
        Tensor castInput(DataType::DT_FP32, castInputShape, "castInput");
        Tensor leftMatrix(DataType::DT_FP8E4M3, leftMatrixShape, "leftMatrix");
        Tensor leftScale(DataType::DT_FP8E8M0, leftScaleShape, "leftScale");
        Tensor rightScale(DataType::DT_FP8E8M0, rightScaleShape, "rightScale");
        Tensor out(DataType::DT_FP32, outputShape, "output");
        SetFullTestStrategy();
        Function* originFunction = nullptr;
        config::SetBuildStatic(true);
        FUNCTION("TestMXMatmulUB2L1KAlignedDirectPath", {castInput, leftMatrix, leftScale, rightScale, out})
        {
            TileShape::Current().SetVecTile(NUM_16, NUM_64);
            Tensor castRightMatrix = Cast(castInput, DataType::DT_FP8E4M3);
            TileShape::Current().SetCubeTile({NUM_8, NUM_64}, {NUM_64, NUM_64}, {NUM_64, NUM_64});
            out = Matrix::MatmulMX(out.GetDataType(), leftMatrix, leftScale, castRightMatrix, rightScale, false, false,
                                   true, false, false);
        }
        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_TestMXMatmulUB2L1KAlignedDirectPath");
        ASSERT_NE(originFunction, nullptr) << "Function pointer is null";
        EXPECT_GE(CountMemoryPath(originFunction, MemoryType::MEM_UB, MemoryType::MEM_L1), 1)
            << "MX_PADDING_MODE copy-in with 64-aligned K should keep UB->L1 direct path";
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
    Platform::Instance().ReloadMemoryPaths("2201");
}

static void BuildConvertDynShapeGraph(std::shared_ptr<Function>& currFunctionPtr)
{
    currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestConvertDynShape", "TestConvertDynShape",
                                                 nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("TestConvertDynShape", currFunctionPtr);

    std::vector<int64_t> shape = {16, 32};
    std::vector<int64_t> shape1 = {32, 16};
    std::vector<SymbolicScalar> dynShape = {IRBuilder().CreateConstInt(6), IRBuilder().CreateConstInt(6)};
    std::shared_ptr<LogicalTensor> input_tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, dynShape);
    std::shared_ptr<LogicalTensor> input_tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, dynShape);
    std::shared_ptr<LogicalTensor> view_output1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, dynShape);
    std::shared_ptr<LogicalTensor> view_output2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, dynShape);
    std::shared_ptr<LogicalTensor> add_output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, dynShape);
    std::shared_ptr<LogicalTensor> reshape_output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1,
                                                                                               dynShape);
    std::shared_ptr<LogicalTensor> assem_output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, dynShape);

    auto& view_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {input_tensor1}, {view_output1});
    view_op1.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& view_op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {input_tensor2}, {view_output2});
    view_op2.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_ADD, {view_output1, view_output2}, {add_output});
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_RESHAPE, {add_output}, {reshape_output});
    auto& assemble_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_ASSEMBLE, {reshape_output},
                                                       {assem_output});
    assemble_op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));

    currFunctionPtr->inCasts_.push_back(input_tensor1);
    currFunctionPtr->inCasts_.push_back(input_tensor2);
    currFunctionPtr->outCasts_.push_back(assem_output);
}

TEST_F(LegacyAssignMemoryTypeTest, TestConvertOpsHaveDynValidShape)
{
    std::shared_ptr<Function> currFunctionPtr;
    BuildConvertDynShapeGraph(currFunctionPtr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    AssignMemoryType assignMemoryType;
    assignMemoryType.PreCheck(*currFunctionPtr);
    assignMemoryType.RunOnFunction(*currFunctionPtr);
    assignMemoryType.PostCheck(*currFunctionPtr);

    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            auto input = op.iOperand[0];
            auto output = op.oOperand[0];
            const auto& in_shape = input->GetDynValidShape();
            const auto& out_shape = output->GetDynValidShape();
            ASSERT_EQ(in_shape.size(), out_shape.size()) << "Size mismatch";
            for (size_t i = 0; i < in_shape.size(); ++i) {
                EXPECT_EQ(in_shape[i].Dump(), out_shape[i].Dump()) << "Mismatch at dimension " << i;
            }
        }
    }
}
} // namespace tile_fwk

struct VecDupViewMatmulGraph {
    LogicalTensorPtr vecDupOutput;
    LogicalTensorPtr viewOutput;
    LogicalTensorPtr inputB;
    LogicalTensorPtr matmulOutput;
    Operation* viewOp = nullptr;
};

TEST_F(LegacyAssignMemoryTypeTest, ReshapeOutputUsesUbWithMixedViewConsumers)
{
    ComputationalGraphBuilder G;
    Shape shape{NUM_16, NUM_32};
    Shape reshapeShape{NUM_32, NUM_16};
    G.AddTensor(DataType::DT_FP16, shape, MemoryType::MEM_UNKNOWN, "vec_dup_out");
    G.AddTensor(DataType::DT_FP16, reshapeShape, MemoryType::MEM_UNKNOWN, "reshape_out");
    G.AddTensor(DataType::DT_FP16, reshapeShape, MemoryType::MEM_UNKNOWN, "mul_out");
    G.AddTensor(DataType::DT_FP16, reshapeShape, MemoryType::MEM_UNKNOWN, "view_l1_out");
    G.AddTensor(DataType::DT_FP16, reshapeShape, MemoryType::MEM_UNKNOWN, "l0a_out");

    G.AddOp(Opcode::OP_VEC_DUP, {}, {"vec_dup_out"}, "vec_dup");
    G.AddOp(Opcode::OP_RESHAPE, {"vec_dup_out"}, {"reshape_out"}, "reshape");
    G.AddOp(Opcode::OP_MUL, {"reshape_out", "reshape_out"}, {"mul_out"}, "mul");
    G.AddOp(Opcode::OP_VIEW, {"reshape_out"}, {"view_l1_out"}, "view_l1");
    G.GetOp("view_l1")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_L1));
    G.AddOp(Opcode::OP_L1_TO_L0A, {"view_l1_out"}, {"l0a_out"}, "l1_to_l0a");

    Function* func = G.GetFunction();
    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.RunOnFunction(*func), SUCCESS);
    EXPECT_EQ(assignMemoryType.PostCheck(*func), SUCCESS);

    auto reshapeInput = G.GetTensor("vec_dup_out");
    auto reshapeOutput = G.GetTensor("reshape_out");
    auto mulOutput = G.GetTensor("mul_out");
    auto l1ViewOutput = G.GetTensor("view_l1_out");
    auto l0aOutput = G.GetTensor("l0a_out");
    EXPECT_EQ(reshapeInput->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    EXPECT_EQ(reshapeOutput->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    EXPECT_EQ(mulOutput->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    EXPECT_EQ(l1ViewOutput->GetMemoryTypeOriginal(), MemoryType::MEM_L1);
    EXPECT_EQ(l0aOutput->GetMemoryTypeOriginal(), MemoryType::MEM_L0A);

    auto viewOpAttr = std::dynamic_pointer_cast<ViewOpAttribute>(G.GetOp("view_l1")->GetOpAttribute());
    ASSERT_NE(viewOpAttr, nullptr);
    EXPECT_EQ(viewOpAttr->GetTo(), MemoryType::MEM_L1);
}

TEST_F(LegacyAssignMemoryTypeTest, CastAssembleReshapeViewMatmulKeepsReshapeOutputUb)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Platform::Instance().ReloadMemoryPaths("3510");
    ComputationalGraphBuilder G;
    Shape inputShape{NUM_16, NUM_32};
    Shape matrixAShape{NUM_32, NUM_16};
    Shape matrixBShape{NUM_16, NUM_16};

    G.AddTensor(DataType::DT_FP32, inputShape, MemoryType::MEM_DEVICE_DDR, "cast_in");
    G.AddTensor(DataType::DT_FP16, inputShape, MemoryType::MEM_UNKNOWN, "cast_out");
    G.AddTensor(DataType::DT_FP16, inputShape, MemoryType::MEM_UNKNOWN, "assemble_out");
    G.AddTensor(DataType::DT_FP16, matrixAShape, MemoryType::MEM_UNKNOWN, "reshape_out");
    G.AddTensor(DataType::DT_FP16, matrixAShape, MemoryType::MEM_UNKNOWN, "view_l1_out");
    G.AddTensor(DataType::DT_FP16, matrixAShape, MemoryType::MEM_UNKNOWN, "view_l0a_out");
    G.AddTensor(DataType::DT_FP16, matrixBShape, MemoryType::MEM_DEVICE_DDR, "matrix_b");
    G.AddTensor(DataType::DT_FP16, matrixBShape, MemoryType::MEM_UNKNOWN, "matrix_b_l1");
    G.AddTensor(DataType::DT_FP16, matrixBShape, MemoryType::MEM_UNKNOWN, "matrix_b_l0b");
    G.AddTensor(DataType::DT_FP16, matrixAShape, MemoryType::MEM_UNKNOWN, "matmul_out");
    G.AddTensor(DataType::DT_FP16, matrixAShape, MemoryType::MEM_DEVICE_DDR, "output");

    G.AddOp(Opcode::OP_CAST, {"cast_in"}, {"cast_out"}, "cast");
    G.AddOp(Opcode::OP_ASSEMBLE, {"cast_out"}, {"assemble_out"}, "assemble");
    G.GetOp("assemble")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(Offset{0, 0}));
    G.AddOp(Opcode::OP_RESHAPE, {"assemble_out"}, {"reshape_out"}, "reshape");
    G.AddOp(Opcode::OP_VIEW, {"reshape_out"}, {"view_l1_out"}, "view_l1");
    G.GetOp("view_l1")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_L1));
    G.AddOp(Opcode::OP_VIEW, {"view_l1_out"}, {"view_l0a_out"}, "view_l0a");
    G.GetOp("view_l0a")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_L0A));
    G.AddOp(Opcode::OP_VIEW, {"matrix_b"}, {"matrix_b_l1"}, "matrix_b_to_l1");
    G.GetOp("matrix_b_to_l1")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_L1));
    G.AddOp(Opcode::OP_VIEW, {"matrix_b_l1"}, {"matrix_b_l0b"}, "matrix_b_to_l0b");
    G.GetOp("matrix_b_to_l0b")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_L0B));
    G.AddOp(Opcode::OP_A_MUL_B, {"view_l0a_out", "matrix_b_l0b"}, {"matmul_out"}, "matmul");
    G.AddOp(Opcode::OP_ASSEMBLE, {"matmul_out"}, {"output"}, "output_assemble");
    G.GetOp("output_assemble")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(Offset{0, 0}));
    G.SetInCast({"cast_in", "matrix_b"});
    G.SetOutCast({"output"});

    Function* func = G.GetFunction();
    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.RunOnFunction(*func), SUCCESS);
    EXPECT_EQ(assignMemoryType.PostCheck(*func), SUCCESS);

    EXPECT_EQ(G.GetTensor("cast_out")->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    EXPECT_EQ(G.GetTensor("assemble_out")->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    EXPECT_EQ(G.GetTensor("reshape_out")->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    EXPECT_EQ(G.GetTensor("view_l1_out")->GetMemoryTypeOriginal(), MemoryType::MEM_L1);
    EXPECT_EQ(G.GetTensor("view_l0a_out")->GetMemoryTypeOriginal(), MemoryType::MEM_L0A);

    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
    Platform::Instance().ReloadMemoryPaths("2201");
}

static void BuildVecDupViewMatmulNoAssembleGraph(std::shared_ptr<Function>& currFunctionPtr,
                                                 VecDupViewMatmulGraph& graph)
{
    std::vector<int64_t> shape = {32, 16};
    std::vector<int64_t> shapeB = {16, 16};
    std::vector<int64_t> shapeC = {32, 16};
    graph.vecDupOutput = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP16, shape, CreateTestConstIntVector(shape));
    graph.vecDupOutput->SetMagic(100);
    graph.viewOutput = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP16, shape, CreateTestConstIntVector(shape));
    graph.viewOutput->SetMagic(101);
    graph.inputB = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP16, shapeB, CreateTestConstIntVector(shapeB));
    graph.inputB->SetMagic(102);
    graph.matmulOutput = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP16, shapeC, CreateTestConstIntVector(shapeC));
    graph.matmulOutput->SetMagic(103);
    auto& vecDupOp = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VEC_DUP, {}, {graph.vecDupOutput});
    vecDupOp.opmagic = 2001;
    auto& viewOp = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {graph.vecDupOutput},
                                                  {graph.viewOutput});
    viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    viewOp.opmagic = 2002;
    graph.viewOp = &viewOp;
    auto& matmulOp = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_A_MUL_B,
                                                    {graph.viewOutput, graph.inputB}, {graph.matmulOutput});
    matmulOp.opmagic = 2003;
    currFunctionPtr->inCasts_.push_back(graph.inputB);
    currFunctionPtr->outCasts_.push_back(graph.matmulOutput);
}

TEST_F(LegacyAssignMemoryTypeTest, VecDupViewMatmulNoAssemble)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "VecDupViewMatmulNoAssemble",
                                                      "VecDupViewMatmulNoAssemble", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("VecDupViewMatmulNoAssemble", currFunctionPtr);

    VecDupViewMatmulGraph graph;
    BuildVecDupViewMatmulNoAssembleGraph(currFunctionPtr, graph);
    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(assignMemoryType.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(assignMemoryType.PostCheck(*currFunctionPtr), SUCCESS);

    auto viewOpAttr = std::dynamic_pointer_cast<ViewOpAttribute>(graph.viewOp->GetOpAttribute());
    ASSERT_NE(viewOpAttr, nullptr);
    EXPECT_EQ(viewOpAttr->GetTo(), MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(graph.vecDupOutput->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    EXPECT_EQ(graph.vecDupOutput->GetMemoryTypeToBe(), MemoryType::MEM_UB);
    EXPECT_EQ(graph.viewOutput->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(graph.viewOutput->GetMemoryTypeToBe(), MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(graph.inputB->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(graph.inputB->GetMemoryTypeToBe(), MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(graph.matmulOutput->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(graph.matmulOutput->GetMemoryTypeToBe(), MemoryType::MEM_DEVICE_DDR);

    int assembleCount = 0;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            assembleCount++;
            auto assembleOpAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(op.GetOpAttribute());
            ASSERT_NE(assembleOpAttr, nullptr);
            EXPECT_EQ(assembleOpAttr->GetFrom(), MemoryType::MEM_UB);
            ASSERT_FALSE(op.GetIOperands().empty());
            ASSERT_FALSE(op.GetOOperands().empty());
            EXPECT_EQ(op.GetIOperands().front()->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
            EXPECT_EQ(op.GetOOperands().front()->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
            EXPECT_EQ(op.GetOOperands().front()->GetMemoryTypeToBe(), MemoryType::MEM_DEVICE_DDR);
        }
    }
    EXPECT_EQ(assembleCount, 1) << "Should insert one assemble when vec_dup(UB) -> view -> A_MUL_B(DDR)";
}

TEST_F(LegacyAssignMemoryTypeTest, DdrViewMatmulNoAssemble)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "DdrViewMatmulNoAssemble",
                                                      "DdrViewMatmulNoAssemble", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("DdrViewMatmulNoAssemble", currFunctionPtr);
    std::vector<int64_t> shapeA = {32, 16};
    std::vector<int64_t> shapeB = {16, 16};
    std::vector<int64_t> shapeC = {32, 16};
    auto inputA = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP16, shapeA, CreateTestConstIntVector(shapeA));
    inputA->SetMagic(110);
    auto viewOutput = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP16, shapeA, CreateTestConstIntVector(shapeA));
    viewOutput->SetMagic(111);
    auto inputB = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP16, shapeB, CreateTestConstIntVector(shapeB));
    inputB->SetMagic(112);
    auto matmulOutput = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP16, shapeC, CreateTestConstIntVector(shapeC));
    matmulOutput->SetMagic(113);
    auto& viewOp = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {inputA}, {viewOutput});
    viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    viewOp.opmagic = 2012;
    auto& matmulOp = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_A_MUL_B, {viewOutput, inputB},
                                                    {matmulOutput});
    matmulOp.opmagic = 2013;
    currFunctionPtr->inCasts_.push_back(inputA);
    currFunctionPtr->inCasts_.push_back(inputB);
    currFunctionPtr->outCasts_.push_back(matmulOutput);

    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(assignMemoryType.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(assignMemoryType.PostCheck(*currFunctionPtr), SUCCESS);

    int assembleCount = 0;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            assembleCount++;
        }
    }
    EXPECT_EQ(assembleCount, 0) << "Should not insert assemble when view input is already DDR";
    auto viewOpAttr = std::dynamic_pointer_cast<ViewOpAttribute>(viewOp.GetOpAttribute());
    ASSERT_NE(viewOpAttr, nullptr);
    EXPECT_EQ(viewOpAttr->GetTo(), MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(inputA->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(viewOutput->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
}

// VIEW_TYPE 输出 toBeMap 未知、后继 VIEW 无 toAttr 时，验证 InferTargetTypeThroughForwardViews
// 沿未推导视图链向前查得有效内存类型。
// 链路：inA/inB/inC(DDR) -> VIEW -> GATHER_IN_UB -> VIEW -> VIEW_TYPE -> VIEW -> ADDS -> ASSEMBLE -> outcast
TEST_F(LegacyAssignMemoryTypeTest, ViewTypeReusesForwardViewRequirement)
{
    ComputationalGraphBuilder G;
    // 头部输入（DDR incast）：nope_cache / topk_indices / block_table
    G.AddTensor(DataType::DT_INT8, {NUM_16, NUM_32}, MemoryType::MEM_DEVICE_DDR, "inA");
    G.AddTensor(DataType::DT_INT32, {1, NUM_16}, MemoryType::MEM_DEVICE_DDR, "inB");
    G.AddTensor(DataType::DT_INT32, {1, NUM_16}, MemoryType::MEM_DEVICE_DDR, "inC");
    // 视图输出（均无 toAttr）
    G.AddTensor(DataType::DT_INT8, {NUM_16, NUM_32}, MemoryType::MEM_UNKNOWN, "t1792");
    G.AddTensor(DataType::DT_INT32, {1, NUM_16}, MemoryType::MEM_UNKNOWN, "t1793");
    G.AddTensor(DataType::DT_INT32, {1, NUM_16}, MemoryType::MEM_UNKNOWN, "t1794");
    G.AddTensor(DataType::DT_INT8, {NUM_16, NUM_32}, MemoryType::MEM_UNKNOWN, "t1795");
    G.AddTensor(DataType::DT_INT8, {NUM_16, NUM_32}, MemoryType::MEM_UNKNOWN, "t3280");
    // VIEW_TYPE 做字节等价重解释：INT8[16,32]=512B -> FP32[16,8]=512B
    G.AddTensor(DataType::DT_FP32, {NUM_16, NUM_8}, MemoryType::MEM_UNKNOWN, "t3281");
    G.AddTensor(DataType::DT_FP32, {NUM_16, NUM_8}, MemoryType::MEM_UNKNOWN, "t3536");
    G.AddTensor(DataType::DT_FP32, {NUM_16, NUM_8}, MemoryType::MEM_UNKNOWN, "t3537");
    G.AddTensor(DataType::DT_FP32, {NUM_16, NUM_8}, MemoryType::MEM_UNKNOWN, "t8307");

    // inA/inB/inC -> VIEW[16138/16010/15881]
    G.AddOp(Opcode::OP_VIEW, {"inA"}, {"t1792"}, "view16138");
    G.GetOp("view16138")
        ->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}, MemoryType::MEM_UNKNOWN));
    G.AddOp(Opcode::OP_VIEW, {"inB"}, {"t1793"}, "view16010");
    G.GetOp("view16010")
        ->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}, MemoryType::MEM_UNKNOWN));
    G.AddOp(Opcode::OP_VIEW, {"inC"}, {"t1794"}, "view15881");
    G.GetOp("view15881")
        ->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}, MemoryType::MEM_UNKNOWN));
    // VIEW* -> GATHER_IN_UB[10211] -> t1795
    G.AddOp(Opcode::OP_GATHER_IN_UB, {"t1792", "t1793", "t1794"}, {"t1795"}, "gather10211");
    // t1795 -> VIEW[16394] -> t3280
    G.AddOp(Opcode::OP_VIEW, {"t1795"}, {"t3280"}, "view16394");
    G.GetOp("view16394")
        ->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}, MemoryType::MEM_UNKNOWN));
    // t3280 -> VIEW_TYPE[12343] -> t3281
    G.AddOp(Opcode::OP_VIEW_TYPE, {"t3280"}, {"t3281"}, "viewtype12343");
    // t3281 -> VIEW[12726] -> t3536
    G.AddOp(Opcode::OP_VIEW, {"t3281"}, {"t3536"}, "view12726");
    G.GetOp("view12726")
        ->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}, MemoryType::MEM_UNKNOWN));
    // t3536 -> ADDS[12727] -> t3537
    G.AddOp(Opcode::OP_ADDS, {"t3536"}, {"t3537"}, "adds12727");
    // t3537 -> ASSEMBLE[19009] -> t8307（后接 outcast）
    G.AddOp(Opcode::OP_ASSEMBLE, {"t3537"}, {"t8307"}, "assemble19009");
    G.GetOp("assemble19009")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));

    G.SetInCast({"inA", "inB", "inC"});
    G.SetOutCast({"t8307"});

    Function* function = G.GetFunction();
    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(assignMemoryType.PostCheck(*function), SUCCESS);

    // 核心：VIEW_TYPE[12343] 输出由后继视图链前推得到 UB（改动前会回退 DDR）
    auto viewTypeOut = G.GetTensor("t3281");
    ASSERT_NE(viewTypeOut, nullptr);
    EXPECT_EQ(viewTypeOut->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    EXPECT_EQ(viewTypeOut->GetMemoryTypeToBe(), MemoryType::MEM_UB);

    // 后继 VIEW[12726] 的 toAttr 同步为 UB
    auto view12726Attr = std::dynamic_pointer_cast<ViewOpAttribute>(G.GetOp("view12726")->GetOpAttribute());
    ASSERT_NE(view12726Attr, nullptr);
    EXPECT_EQ(view12726Attr->GetTo(), MemoryType::MEM_UB);

    // 生产者 VIEW[16394] 的 toAttr 同步为 UB
    auto view16394Attr = std::dynamic_pointer_cast<ViewOpAttribute>(G.GetOp("view16394")->GetOpAttribute());
    ASSERT_NE(view16394Attr, nullptr);
    EXPECT_EQ(view16394Attr->GetTo(), MemoryType::MEM_UB);

    // ASSEMBLE[19009] 的 fromAttr 推导为 UB
    auto assembleAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(G.GetOp("assemble19009")->GetOpAttribute());
    ASSERT_NE(assembleAttr, nullptr);
    EXPECT_EQ(assembleAttr->GetFrom(), MemoryType::MEM_UB);
}

TEST_F(LegacyAssignMemoryTypeTest, TestL0C2UBAssembleDirectPathNotDdrFallback)
{
    auto shapes = PrepareA5Platform();
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputA(DataType::DT_FP16, shapes.shapeA, "A");
        Tensor inputB(DataType::DT_FP16, shapes.shapeB, "B");
        Tensor inputC(DataType::DT_FP32, shapes.shapeC, "C");
        Tensor ubAssembleOut(DataType::DT_FP32, shapes.shapeC, "ubAssembleOut");
        Tensor out(DataType::DT_FP32, shapes.shapeC, "output");
        SetFullTestStrategy();
        Function* originFunction = nullptr;

        config::SetBuildStatic(true);
        FUNCTION("TestL0C2UBAssembleDirectPathNotDdrFallback", {inputA, inputB, inputC, ubAssembleOut, out})
        {
            TileShape::Current().SetCubeTile({NUM_32, NUM_32}, {NUM_64, NUM_64}, {NUM_64, NUM_64});
            Tensor ab = Matrix::Matmul(out.GetDataType(), inputA, inputB);
            Assemble(ab, {0, 0}, ubAssembleOut);
            TileShape::Current().SetVecTile(NUM_64, NUM_64);
            ubAssembleOut = Exp(ubAssembleOut);
            out = Add(ab, inputC);
        }

        originFunction = Program::GetInstance().GetFunctionByRawName(
            "TENSOR_TestL0C2UBAssembleDirectPathNotDdrFallback");
        ASSERT_NE(originFunction, nullptr) << "Function pointer is null";
        EXPECT_EQ(CountMemoryPath(originFunction, MemoryType::MEM_L0C, MemoryType::MEM_DEVICE_DDR), 0);
        EXPECT_GE(CountMemoryPath(originFunction, MemoryType::MEM_L0C, MemoryType::MEM_UB), 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
    Platform::Instance().ReloadMemoryPaths("2201");
}

TEST_F(LegacyAssignMemoryTypeTest, TestUB2L1AssembleDirectPathNotDdrFallback)
{
    PrepareA5Platform();
    std::vector<int64_t> ub2l1ShapeA = {32, 64};
    std::vector<int64_t> ub2l1ShapeB = {64, 64};
    std::vector<int64_t> ub2l1ShapeC = {32, 64};
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputA1(DataType::DT_FP32, ub2l1ShapeA, "A1");
        Tensor inputA2(DataType::DT_FP32, ub2l1ShapeA, "A2");
        Tensor inputB1(DataType::DT_FP32, ub2l1ShapeB, "B1");
        Tensor inputB2(DataType::DT_FP32, ub2l1ShapeB, "B2");
        Tensor l1AssembleOut(DataType::DT_FP32, ub2l1ShapeC, "l1AssembleOut");
        Tensor out(DataType::DT_FP32, ub2l1ShapeC, "output");
        SetFullTestStrategy();
        Function* originFunction = nullptr;

        config::SetBuildStatic(true);
        FUNCTION("TestUB2L1AssembleDirectPathNotDdrFallback", {inputA1, inputA2, inputB1, inputB2, l1AssembleOut, out})
        {
            TileShape::Current().SetVecTile(16, 32);
            Tensor add1 = Add(inputA1, inputA2);
            Assemble(add1, {0, 0}, l1AssembleOut);
            Tensor add2 = Add(inputB1, inputB2);
            TileShape::Current().SetCubeTile({32, 32}, {64, 64}, {64, 64});
            l1AssembleOut = Matrix::Matmul(out.GetDataType(), l1AssembleOut, add2);
            out = Matrix::Matmul(out.GetDataType(), add1, add2);
        }

        originFunction = Program::GetInstance().GetFunctionByRawName(
            "TENSOR_TestUB2L1AssembleDirectPathNotDdrFallback");
        ASSERT_NE(originFunction, nullptr) << "Function pointer is null";
        EXPECT_EQ(CountMemoryPath(originFunction, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR), 0);
        EXPECT_GE(CountMemoryPath(originFunction, MemoryType::MEM_UB, MemoryType::MEM_L1), 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
    Platform::Instance().ReloadMemoryPaths("2201");
}

TEST_F(LegacyAssignMemoryTypeTest, AssembleL1ViewAndNonViewConsumerFallbackDdr)
{
    ComputationalGraphBuilder G;
    BuildAssembleL1ViewAndNonViewConsumerGraph(G);

    AssignMemoryType assignMemoryType;
    Function* function = G.GetFunction();
    EXPECT_EQ(assignMemoryType.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(assignMemoryType.PostCheck(*function), SUCCESS);

    ASSERT_EQ(G.GetOp("assemble2")->GetIOperands().size(), 1);
    EXPECT_NE(G.GetOp("assemble2")->GetIOperands().front()->GetMemoryTypeOriginal(), MemoryType::MEM_L1);
}

// 为 conv 算子增加的临时规避动作测试，规避动作删除后此测试跟随删除。
// 场景：permute(incast/DDR) -> view -> transData(outcast/DDR + UB temp)
// 验证：view 输入被强制 DDR，permute 输出(UB)与 view 输入(DDR)间插入 assemble(UB->DDR)
TEST_F(LegacyAssignMemoryTypeTest, PermuteViewTransDataForceDdr)
{
    ComputationalGraphBuilder G;
    Shape shape{NUM_1, NUM_16, NUM_32, NUM_32};
    G.AddTensors(DataType::DT_FP16, shape, {"incast", "permute_out", "view_out", "outcast", "ub_temp"});
    G.AddOps({Opcode::OP_PERMUTE, Opcode::OP_VIEW, Opcode::OP_NCHW2NC1HWC0},
             {{"incast"}, {"permute_out"}, {"view_out"}}, {{"permute_out"}, {"view_out"}, {"outcast", "ub_temp"}},
             {"permute", "view", "transdata"});
    G.GetOp("permute")->SetAttribute(OpAttributeKey::perm, std::vector<int>{0, 2, 1, 3});
    G.GetOp("view")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0, 0, 0}));
    G.GetOp("transdata")->SetAttribute(OpAttributeKey::transDataOffset, CreateTestConstIntVector({0, 0, 0, 0}));
    G.SetInCast({"incast"});
    G.SetOutCast({"outcast"});
    Function* func = G.GetFunction();
    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.RunOnFunction(*func), SUCCESS);
    EXPECT_EQ(assignMemoryType.PostCheck(*func), SUCCESS);
    int assembleCount = 0;
    for (auto& op : func->Operations()) {
        if (op.GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        assembleCount++;
        EXPECT_EQ(op.GetIOperands().front()->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
        EXPECT_EQ(op.GetOOperands().front()->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
    }
    EXPECT_EQ(assembleCount, 1) << "Should insert assemble(UB->DDR) before view for permute->view->transData";
    EXPECT_EQ(G.GetTensor("permute_out")->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    EXPECT_EQ(G.GetTensor("view_out")->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
}

// 为 conv 算子增加的临时规避动作测试，规避动作删除后此测试跟随删除。
// 场景：permute(incast/DDR) -> view -> register_copy -> assemble -> transData(outcast/DDR + UB temp)
// 验证：view 输入被强制 DDR，permute 输出(UB)与 view 输入(DDR)间插入 assemble(UB->DDR)
TEST_F(LegacyAssignMemoryTypeTest, PermuteViewRegisterCopyAssembleTransDataForceDdr)
{
    ComputationalGraphBuilder G;
    Shape shape{NUM_1, NUM_16, NUM_32, NUM_32};
    G.AddTensors(DataType::DT_FP16, shape,
                 {"incast", "permute_out", "view_out", "rc_out", "asm_out", "outcast", "ub_temp"});
    G.AddOps(
        {Opcode::OP_PERMUTE, Opcode::OP_VIEW, Opcode::OP_REGISTER_COPY, Opcode::OP_ASSEMBLE, Opcode::OP_NCHW2NC1HWC0},
        {{"incast"}, {"permute_out"}, {"view_out"}, {"rc_out"}, {"asm_out"}},
        {{"permute_out"}, {"view_out"}, {"rc_out"}, {"asm_out"}, {"outcast", "ub_temp"}},
        {"permute", "view", "reg_copy", "assemble", "transdata"});
    G.GetOp("permute")->SetAttribute(OpAttributeKey::perm, std::vector<int>{0, 2, 1, 3});
    G.GetOp("view")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0, 0, 0}));
    G.GetOp("assemble")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(Offset{0, 0, 0, 0}));
    G.GetOp("transdata")->SetAttribute(OpAttributeKey::transDataOffset, CreateTestConstIntVector({0, 0, 0, 0}));
    G.SetInCast({"incast"});
    G.SetOutCast({"outcast"});
    Function* func = G.GetFunction();
    std::vector<int64_t> beforeMagics;
    for (auto& op : func->Operations()) {
        beforeMagics.push_back(op.GetOpMagic());
    }
    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.RunOnFunction(*func), SUCCESS);
    EXPECT_EQ(assignMemoryType.PostCheck(*func), SUCCESS);
    int newAssembleCount = 0;
    for (auto& op : func->Operations()) {
        if (op.GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        if (std::find(beforeMagics.begin(), beforeMagics.end(), op.GetOpMagic()) != beforeMagics.end()) {
            continue;
        }
        newAssembleCount++;
        EXPECT_EQ(op.GetIOperands().front()->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
        EXPECT_EQ(op.GetOOperands().front()->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
    }
    EXPECT_EQ(newAssembleCount, 1)
        << "Should insert assemble(UB->DDR) before view for permute->view->rc->asm->transData";
    EXPECT_EQ(G.GetTensor("permute_out")->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    EXPECT_EQ(G.GetTensor("view_out")->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
}

// DAV_3510 L0C -> UB 直连穿透 REGISTER_COPY 时，REGISTER_COPY 后的真实消费者 shape
// 不满足端点逐维整除关系，必须回退 DDR。
TEST_F(LegacyAssignMemoryTypeTest, ShapeTransportRegisterCopyShapeMismatchFallbackDdr)
{
    config::SetPassOption(ENABLE_SLICE, false);
    ComputationalGraphBuilder G;
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Platform::Instance().ReloadMemoryPaths("3510");
    G.AddTensor(DataType::DT_FP16, {NUM_32, NUM_32}, MemoryType::MEM_L0A, "l0a_in");
    G.AddTensor(DataType::DT_FP16, {NUM_32, NUM_128}, MemoryType::MEM_L0B, "l0b_in");
    G.AddTensor(DataType::DT_FP16, {NUM_32, NUM_128}, MemoryType::MEM_L0C, "matmul_out");
    G.AddTensor(DataType::DT_FP16, {NUM_32, NUM_128}, MemoryType::MEM_L0C, "view_l0c_out");
    G.AddTensor(DataType::DT_FP16, {NUM_32, NUM_128}, MemoryType::MEM_UB, "asm_out");
    G.AddTensor(DataType::DT_FP16, {NUM_32, NUM_128}, MemoryType::MEM_UB, "rc_out");
    G.AddTensor(DataType::DT_FP16, {NUM_16, NUM_256}, MemoryType::MEM_UB, "view_ub_out");
    G.AddTensor(DataType::DT_FP16, {NUM_16, NUM_256}, MemoryType::MEM_UB, "add_in2");
    G.AddTensor(DataType::DT_FP16, {NUM_16, NUM_256}, MemoryType::MEM_UB, "add_out");

    G.AddOp(Opcode::OP_A_MUL_B, {"l0a_in", "l0b_in"}, {"matmul_out"}, "matmul");
    G.AddOp(Opcode::OP_VIEW, {"matmul_out"}, {"view_l0c_out"}, "view_l0c");
    G.GetOp("view_l0c")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_L0C));
    G.AddOp(Opcode::OP_ASSEMBLE, {"view_l0c_out"}, {"asm_out"}, "assemble");
    G.GetOp("assemble")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(Offset{0, 0}));
    G.AddOp(Opcode::OP_REGISTER_COPY, {"asm_out"}, {"rc_out"}, "register_copy");
    G.AddOp(Opcode::OP_VIEW, {"rc_out"}, {"view_ub_out"}, "view_ub");
    G.GetOp("view_ub")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_UB));
    G.AddOp(Opcode::OP_ADD, {"view_ub_out", "add_in2"}, {"add_out"}, "add");
    G.SetInCast({"l0a_in", "l0b_in", "add_in2"});
    G.SetOutCast({"add_out"});

    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.RunOnFunction(*G.GetFunction()), SUCCESS);
    EXPECT_EQ(CountMemoryPath(G.GetFunction(), MemoryType::MEM_L0C, MemoryType::MEM_UB), 0);
    EXPECT_GT(CountMemoryPath(G.GetFunction(), MemoryType::MEM_L0C, MemoryType::MEM_DEVICE_DDR), 0);
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
    Platform::Instance().ReloadMemoryPaths("2201");
}

// 与上例对应的正向场景：REGISTER_COPY 输入 shape 与端点较小 shape 相同，保留 L0C -> L1 直连。
TEST_F(LegacyAssignMemoryTypeTest, ShapeTransportRegisterCopyShapeMultipleKeepsDirectPath)
{
    ComputationalGraphBuilder G;
    G.AddTensors(DataType::DT_FP16, {NUM_64, NUM_64},
                 {MemoryType::MEM_L0A, MemoryType::MEM_L0B, MemoryType::MEM_L0C, MemoryType::MEM_L1},
                 {"l0a_in", "l0b_in", "matmul_out", "l1_transport_out"});
    G.AddTensors(
        DataType::DT_FP16, {NUM_32, NUM_64},
        {MemoryType::MEM_L1, MemoryType::MEM_L1, MemoryType::MEM_L0A, MemoryType::MEM_L0B, MemoryType::MEM_L0C},
        {"l1_view_out", "rc_out", "l0a_view_out", "l0b_in_2", "out"});

    G.AddOp(Opcode::OP_A_MUL_B, {"l0a_in", "l0b_in"}, {"matmul_out"}, "matmul_1");
    G.AddOp(Opcode::OP_VIEW, {"matmul_out"}, {"l1_transport_out"}, "l0c_to_l1");
    G.GetOp("l0c_to_l1")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_L1));
    G.AddOp(Opcode::OP_VIEW, {"l1_transport_out"}, {"l1_view_out"}, "l1_view");
    G.GetOp("l1_view")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_L1));
    G.AddOp(Opcode::OP_REGISTER_COPY, {"l1_view_out"}, {"rc_out"}, "register_copy");
    G.AddOp(Opcode::OP_VIEW, {"rc_out"}, {"l0a_view_out"}, "l1_to_l0a");
    G.GetOp("l1_to_l0a")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_L0A));
    G.AddOp(Opcode::OP_A_MUL_B, {"l0a_view_out", "l0b_in_2"}, {"out"}, "matmul_2");
    G.SetInCast({"l0a_in", "l0b_in", "l0b_in_2"});
    G.SetOutCast({"out"});

    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.RunOnFunction(*G.GetFunction()), SUCCESS);
    EXPECT_GT(CountMemoryPath(G.GetFunction(), MemoryType::MEM_L0C, MemoryType::MEM_L1), 0);
    EXPECT_EQ(CountMemoryPath(G.GetFunction(), MemoryType::MEM_L0C, MemoryType::MEM_DEVICE_DDR), 0);
}

// REGISTER_COPY 的输入维数与端点 shape 不一致时，即使端点之间可整除，也必须回退 DDR。
TEST_F(LegacyAssignMemoryTypeTest, ShapeTransportRegisterCopyRankMismatchFallbackDdr)
{
    ComputationalGraphBuilder G;
    G.AddTensors(DataType::DT_FP16, {NUM_64, NUM_64}, {MemoryType::MEM_L0A, MemoryType::MEM_L0B, MemoryType::MEM_L0C},
                 {"l0a_in", "l0b_in", "matmul_out"});
    G.AddTensor(DataType::DT_FP16, {NUM_1024}, MemoryType::MEM_L0C, "l0c_view_out");
    G.AddTensor(DataType::DT_FP16, {NUM_32, NUM_64}, MemoryType::MEM_L0C, "rc_out");
    G.AddTensors(DataType::DT_FP16, {NUM_32, NUM_64},
                 {MemoryType::MEM_L1, MemoryType::MEM_L0A, MemoryType::MEM_L0B, MemoryType::MEM_L0C},
                 {"l1_transport_out", "l0a_view_out", "l0b_in_2", "out"});

    G.AddOp(Opcode::OP_A_MUL_B, {"l0a_in", "l0b_in"}, {"matmul_out"}, "matmul_1");
    G.AddOp(Opcode::OP_VIEW, {"matmul_out"}, {"l0c_view_out"}, "l0c_view");
    G.GetOp("l0c_view")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_L0C));
    G.AddOp(Opcode::OP_REGISTER_COPY, {"l0c_view_out"}, {"rc_out"}, "register_copy");
    G.AddOp(Opcode::OP_VIEW, {"rc_out"}, {"l1_transport_out"}, "l0c_to_l1");
    G.GetOp("l0c_to_l1")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_L1));
    G.AddOp(Opcode::OP_VIEW, {"l1_transport_out"}, {"l0a_view_out"}, "l1_to_l0a");
    G.GetOp("l1_to_l0a")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_L0A));
    G.AddOp(Opcode::OP_A_MUL_B, {"l0a_view_out", "l0b_in_2"}, {"out"}, "matmul_2");
    G.SetInCast({"l0a_in", "l0b_in", "l0b_in_2"});
    G.SetOutCast({"out"});

    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.RunOnFunction(*G.GetFunction()), SUCCESS);
    EXPECT_EQ(CountMemoryPath(G.GetFunction(), MemoryType::MEM_L0C, MemoryType::MEM_L1), 0);
    EXPECT_GT(CountMemoryPath(G.GetFunction(), MemoryType::MEM_L0C, MemoryType::MEM_DEVICE_DDR), 0);
}

// REGISTER_COPY 的输入维数相同但不是端点较小 shape 的整数倍时，必须回退 DDR。
TEST_F(LegacyAssignMemoryTypeTest, ShapeTransportRegisterCopyNonMultipleFallbackDdr)
{
    ComputationalGraphBuilder G;
    G.AddTensors(DataType::DT_FP16, {NUM_64, NUM_64}, {MemoryType::MEM_L0A, MemoryType::MEM_L0B, MemoryType::MEM_L0C},
                 {"l0a_in", "l0b_in", "matmul_out"});
    G.AddTensor(DataType::DT_FP16, {NUM_48, NUM_64}, MemoryType::MEM_L0C, "l0c_view_out");
    G.AddTensor(DataType::DT_FP16, {NUM_32, NUM_64}, MemoryType::MEM_L0C, "rc_out");
    G.AddTensors(DataType::DT_FP16, {NUM_32, NUM_64},
                 {MemoryType::MEM_L1, MemoryType::MEM_L0A, MemoryType::MEM_L0B, MemoryType::MEM_L0C},
                 {"l1_transport_out", "l0a_view_out", "l0b_in_2", "out"});

    G.AddOp(Opcode::OP_A_MUL_B, {"l0a_in", "l0b_in"}, {"matmul_out"}, "matmul_1");
    G.AddOp(Opcode::OP_VIEW, {"matmul_out"}, {"l0c_view_out"}, "l0c_view");
    G.GetOp("l0c_view")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_L0C));
    G.AddOp(Opcode::OP_REGISTER_COPY, {"l0c_view_out"}, {"rc_out"}, "register_copy");
    G.AddOp(Opcode::OP_VIEW, {"rc_out"}, {"l1_transport_out"}, "l0c_to_l1");
    G.GetOp("l0c_to_l1")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_L1));
    G.AddOp(Opcode::OP_VIEW, {"l1_transport_out"}, {"l0a_view_out"}, "l1_to_l0a");
    G.GetOp("l1_to_l0a")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_L0A));
    G.AddOp(Opcode::OP_A_MUL_B, {"l0a_view_out", "l0b_in_2"}, {"out"}, "matmul_2");
    G.SetInCast({"l0a_in", "l0b_in", "l0b_in_2"});
    G.SetOutCast({"out"});

    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.RunOnFunction(*G.GetFunction()), SUCCESS);
    EXPECT_EQ(CountMemoryPath(G.GetFunction(), MemoryType::MEM_L0C, MemoryType::MEM_L1), 0);
    EXPECT_GT(CountMemoryPath(G.GetFunction(), MemoryType::MEM_L0C, MemoryType::MEM_DEVICE_DDR), 0);
}

// 场景：view 的 fromOffset 在外层维度有偏移 [1,0]，线性偏移 = 1*8+0 = 8 元素
// FP16: 2*8=16 bytes，16 % 32 != 0 -> 不对齐，view 输入回退 DDR
// 修复前仅检查 fromOffset.back()=0，会错误判定为对齐
TEST_F(LegacyAssignMemoryTypeTest, UnalignedViewOuterDimOffsetFallbackDdr)
{
    ComputationalGraphBuilder G;
    Shape shape{NUM_16, NUM_8};
    G.AddTensor(DataType::DT_FP16, shape, MemoryType::MEM_UNKNOWN, "vec_out");
    G.AddTensor(DataType::DT_FP16, shape, MemoryType::MEM_UNKNOWN, "view_out");
    G.AddTensor(DataType::DT_FP16, shape, MemoryType::MEM_UNKNOWN, "adds_out");

    G.AddOp(Opcode::OP_VEC_DUP, {}, {"vec_out"}, "vec_dup");
    G.AddOp(Opcode::OP_VIEW, {"vec_out"}, {"view_out"}, "view");
    G.GetOp("view")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{1, 0}, MemoryType::MEM_UB));
    G.AddOp(Opcode::OP_ADDS, {"view_out"}, {"adds_out"}, "adds");

    Function* func = G.GetFunction();
    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.RunOnFunction(*func), SUCCESS);
    EXPECT_EQ(assignMemoryType.PostCheck(*func), SUCCESS);

    EXPECT_EQ(G.GetTensor("vec_out")->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    EXPECT_EQ(G.GetTensor("view_out")->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    int assembleCount = 0;
    for (auto& op : func->Operations()) {
        if (op.GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        assembleCount++;
        EXPECT_EQ(op.GetIOperands().front()->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
        EXPECT_EQ(op.GetOOperands().front()->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
    }
    EXPECT_EQ(assembleCount, 1) << "Should insert assemble(UB->DDR) for unaligned view fromOffset outer dim";
}

// gdr_bwd 实际场景：shape [128,1], fromOffset [127,0]
// 线性偏移 = 127，FP16: 2*127=254, 254%32=30 != 0 -> 不对齐，view 输入回退 DDR
TEST_F(LegacyAssignMemoryTypeTest, UnalignedViewOffset127GdrBwdFallbackDdr)
{
    ComputationalGraphBuilder G;
    Shape inShape{NUM_128, NUM_1};
    Shape outShape{NUM_1, NUM_1};
    G.AddTensor(DataType::DT_FP16, inShape, MemoryType::MEM_UNKNOWN, "vec_out");
    G.AddTensor(DataType::DT_FP16, outShape, MemoryType::MEM_UNKNOWN, "view_out");
    G.AddTensor(DataType::DT_FP16, outShape, MemoryType::MEM_UNKNOWN, "adds_out");

    G.AddOp(Opcode::OP_VEC_DUP, {}, {"vec_out"}, "vec_dup");
    G.AddOp(Opcode::OP_VIEW, {"vec_out"}, {"view_out"}, "view");
    G.GetOp("view")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{127, 0}, MemoryType::MEM_UB));
    G.AddOp(Opcode::OP_ADDS, {"view_out"}, {"adds_out"}, "adds");

    Function* func = G.GetFunction();
    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.RunOnFunction(*func), SUCCESS);
    EXPECT_EQ(assignMemoryType.PostCheck(*func), SUCCESS);

    EXPECT_EQ(G.GetTensor("vec_out")->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    EXPECT_EQ(G.GetTensor("view_out")->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    int assembleCount = 0;
    for (auto& op : func->Operations()) {
        if (op.GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        assembleCount++;
        EXPECT_EQ(op.GetIOperands().front()->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
        EXPECT_EQ(op.GetOOperands().front()->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
    }
    EXPECT_EQ(assembleCount, 1) << "Should insert assemble(UB->DDR) for view fromOffset [127,0]";
}

// cube 数据加载路径豁免场景，源自 ScaledMmMxMNSplitWithBiasFp8e4m3 失败用例：
// A5 平台 FP8 MatmulMX（含 scale），K 方向切分后首个 A_MUL_B 产出 L0C 累加器，
// 后续 A_MULACC_B 沿 K 累加。scaleA 在 L1 中按 K 组切片后经 view 加载到 L0AMX，
// 切片偏移不在最内层维度，字节不对齐；此类 cube 加载路径若回退 DDR 会级联产生
// 无路径的 L1->UB CONVERT，故豁免对齐检查，保持 L1->L0AMX 管线完整。
TEST_F(LegacyAssignMemoryTypeTest, UnalignedViewToCubePathSkipsDdrFallback)
{
    const NPUArch savedArch = Platform::Instance().GetSoc().GetNPUArch();
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Platform::Instance().ReloadMemoryPaths("3510");
    ComputationalGraphBuilder G;
    // M-split 后的 tile：matA/scaleA/matB/scaleB（k64 = 192/64 = 3）
    Shape matShape{NUM_64, 192};
    Shape scaleAShape{NUM_64, 3, 2};
    Shape matBShape{96, 192};
    Shape scaleBShape{3, 96, 2};
    G.AddTensor(DataType::DT_FP8E4M3, matShape, MemoryType::MEM_UNKNOWN, "input_a");
    G.AddTensor(DataType::DT_FP8E8M0, scaleAShape, MemoryType::MEM_UNKNOWN, "input_sa");
    G.AddTensor(DataType::DT_FP8E4M3, matBShape, MemoryType::MEM_UNKNOWN, "input_b");
    G.AddTensor(DataType::DT_FP8E8M0, scaleBShape, MemoryType::MEM_UNKNOWN, "input_sb");
    G.AddTensor(DataType::DT_FP8E4M3, matShape, MemoryType::MEM_UNKNOWN, "t_a");
    G.AddTensor(DataType::DT_FP8E8M0, scaleAShape, MemoryType::MEM_UNKNOWN, "t_sa");
    G.AddTensor(DataType::DT_FP8E4M3, matBShape, MemoryType::MEM_UNKNOWN, "t_b");
    G.AddTensor(DataType::DT_FP8E8M0, scaleBShape, MemoryType::MEM_UNKNOWN, "t_sb");

    G.AddOp(Opcode::OP_VIEW, {"input_a"}, {"t_a"}, "view_a_l1");
    G.GetOp("view_a_l1")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_L1));
    G.AddOp(Opcode::OP_VIEW, {"input_sa"}, {"t_sa"}, "view_sa_l1");
    G.GetOp("view_sa_l1")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0, 0}, MemoryType::MEM_L1));
    G.AddOp(Opcode::OP_VIEW, {"input_b"}, {"t_b"}, "view_b_l1");
    G.GetOp("view_b_l1")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_L1));
    G.AddOp(Opcode::OP_VIEW, {"input_sb"}, {"t_sb"}, "view_sb_l1");
    G.GetOp("view_sb_l1")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0, 0}, MemoryType::MEM_L1));

    // K-slice #0：A_MUL_B 产出初始 L0C 累加器
    G.AddTensor(DataType::DT_FP8E4M3, Shape{NUM_64, NUM_64}, MemoryType::MEM_UNKNOWN, "a0");
    G.AddTensor(DataType::DT_FP8E4M3, Shape{NUM_64, 96}, MemoryType::MEM_UNKNOWN, "b0");
    G.AddTensor(DataType::DT_FP8E8M0, Shape{NUM_64, 1, 2}, MemoryType::MEM_UNKNOWN, "sa0");
    G.AddTensor(DataType::DT_FP8E8M0, Shape{1, 96, 2}, MemoryType::MEM_UNKNOWN, "sb0");
    G.AddTensor(DataType::DT_FP16, Shape{NUM_64, 96}, MemoryType::MEM_UNKNOWN, "c0");
    G.AddTensor(DataType::DT_FP8E4M3, Shape{NUM_64, NUM_64}, MemoryType::MEM_UNKNOWN, "a1");
    G.AddTensor(DataType::DT_FP8E4M3, Shape{NUM_64, 96}, MemoryType::MEM_UNKNOWN, "b1");
    G.AddTensor(DataType::DT_FP8E8M0, Shape{NUM_64, 1, 2}, MemoryType::MEM_UNKNOWN, "sa1");
    G.AddTensor(DataType::DT_FP8E8M0, Shape{1, 96, 2}, MemoryType::MEM_UNKNOWN, "sb1");
    G.AddTensor(DataType::DT_FP16, Shape{NUM_64, 96}, MemoryType::MEM_UNKNOWN, "out");

    G.AddOp(Opcode::OP_VIEW, {"t_a"}, {"a0"}, "view_a0");
    G.GetOp("view_a0")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_L0A));
    G.AddOp(Opcode::OP_VIEW, {"t_b"}, {"b0"}, "view_b0");
    G.GetOp("view_b0")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_L0B));
    G.AddOp(Opcode::OP_VIEW, {"t_sa"}, {"sa0"}, "view_sa0");
    G.GetOp("view_sa0")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0, 0}, MemoryType::MEM_L0AMX));
    G.AddOp(Opcode::OP_VIEW, {"t_sb"}, {"sb0"}, "view_sb0");
    G.GetOp("view_sb0")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0, 0}, MemoryType::MEM_L0BMX));
    G.AddOp(Opcode::OP_A_MUL_B, {"a0", "b0", "sa0", "sb0"}, {"c0"}, "mmad_k0");

    // K-slice #1：A_MULACC_B 沿 K 累加，scaleA 切片偏移不在最内层维度即豁免命中点
    G.AddOp(Opcode::OP_VIEW, {"t_a"}, {"a1"}, "view_a1");
    G.GetOp("view_a1")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, NUM_64}, MemoryType::MEM_L0A));
    G.AddOp(Opcode::OP_VIEW, {"t_b"}, {"b1"}, "view_b1");
    G.GetOp("view_b1")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, NUM_64}, MemoryType::MEM_L0B));
    G.AddOp(Opcode::OP_VIEW, {"t_sa"}, {"sa1"}, "view_sa1");
    G.GetOp("view_sa1")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 1, 0}, MemoryType::MEM_L0AMX));
    G.AddOp(Opcode::OP_VIEW, {"t_sb"}, {"sb1"}, "view_sb1");
    G.GetOp("view_sb1")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{1, 0, 0}, MemoryType::MEM_L0BMX));
    G.AddOp(Opcode::OP_A_MULACC_B, {"a1", "b1", "c0", "sa1", "sb1"}, {"out"}, "mmacc_k1");

    G.SetInCast({"input_a", "input_sa", "input_b", "input_sb"});
    G.SetOutCast({"out"});

    Function* func = G.GetFunction();
    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.RunOnFunction(*func), SUCCESS);
    EXPECT_EQ(assignMemoryType.PostCheck(*func), SUCCESS);

    // cube 加载管线完整保留：L1 中转 + L0A/L0B/L0AMX/L0BMX 就位，无 DDR 回退
    EXPECT_EQ(G.GetTensor("t_a")->GetMemoryTypeOriginal(), MemoryType::MEM_L1);
    EXPECT_EQ(G.GetTensor("t_sa")->GetMemoryTypeOriginal(), MemoryType::MEM_L1);
    EXPECT_EQ(G.GetTensor("t_b")->GetMemoryTypeOriginal(), MemoryType::MEM_L1);
    EXPECT_EQ(G.GetTensor("t_sb")->GetMemoryTypeOriginal(), MemoryType::MEM_L1);
    EXPECT_EQ(G.GetTensor("a1")->GetMemoryTypeOriginal(), MemoryType::MEM_L0A);
    EXPECT_EQ(G.GetTensor("b1")->GetMemoryTypeOriginal(), MemoryType::MEM_L0B);
    EXPECT_EQ(G.GetTensor("sa1")->GetMemoryTypeOriginal(), MemoryType::MEM_L0AMX);
    EXPECT_EQ(G.GetTensor("sb1")->GetMemoryTypeOriginal(), MemoryType::MEM_L0BMX);
    EXPECT_EQ(G.GetTensor("c0")->GetMemoryTypeOriginal(), MemoryType::MEM_L0C);
    EXPECT_EQ(G.GetTensor("out")->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
    int assembleCount = 0;
    for (auto& op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            assembleCount++;
        }
    }
    EXPECT_EQ(assembleCount, 0) << "Cube path view (to L0AMX) should skip unaligned DDR fallback";
    Platform::Instance().GetSoc().SetNPUArch(savedArch);
    Platform::Instance().ReloadMemoryPaths("2201");
}

} // namespace npu
