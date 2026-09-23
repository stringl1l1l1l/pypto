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
 * \file test_generate_move_op.cpp
 * \brief Unit test for Generate Move Op pass.
 */

#include <gtest/gtest.h>
#include "computational_graph_builder.h"
#include "interface/function/function.h"
#include "interface/tensor/irbuilder.h"
#include "symbolic_scalar_test_utils.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "ut_json/ut_json_tool.h"
#include "passes/tile_graph_pass/data_path/generate_move_op.h"
#include "passes/tile_graph_pass/data_path/convert_op_inserter.h"
#include "interface/configs/config_manager.h"
#include <fstream>
#include <vector>
#include <string>
#include "interface/tensor/irbuilder.h"

using namespace npu::tile_fwk;

namespace npu {
namespace tile_fwk {
const int NUM_32 = 32;
const int NUM_64 = 64;
const int NUM_128 = 128;
constexpr float F_3 = 3.0;

class GenerateMoveOpPassTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetHostConfig(KEY_STRATEGY, "GenerateMoveOpPassTestStrategy");
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {}
};

TEST_F(GenerateMoveOpPassTest, AssembleViewToCopy)
{
    PROGRAM("GenerateMoveOpPassTest")
    {
        std::vector<int64_t> shape1{256, 256};
        std::vector<int64_t> shape2{128, 128};
        TileShape::Current().SetVecTile({128, 128});
        Tensor input_a(DT_FP32, shape1, "input_a");
        Tensor input_b(DT_FP32, shape1, "input_b");
        Tensor output(DT_FP32, shape2, "output");
        PassManager& passManager = PassManager::Instance();
        passManager.RegisterStrategy("GenerateMoveOpPassTestStrategy",
                                     {
                                         {"RemoveRedundantReshape", PassName::REMOVE_REDUNDANT_RESHAPE},
                                         {"InferMemoryConflict", PassName::INFER_MEMORY_CONFLICT},
                                         {"ExpandFunction", PassName::EXPAND_FUNCTION},
                                         {"DuplicateOp", PassName::DUPLICATE_OP},
                                         {"MergeViewAssemble", PassName::MERGE_VIEW_ASSEMBLE},
                                         {"AssignMemoryType", PassName::ASSIGN_MEMORY_TYPE},
                                         {"SplitLargeFanoutTensor", PassName::SPLIT_LARGE_FANOUT_TENSOR},
                                         {"SplitReshape", PassName::SPLIT_RESHAPE},
                                         {"RemoveRedundantOp", PassName::REMOVE_REDUNDANT_OP},

                                     });
        ConfigManager::Instance();

        Function* originFunction = nullptr;
        std::vector<int> originOpmagic;
        config::SetBuildStatic(true);
        FUNCTION("ADD", {input_a, input_b, output})
        {
            config::SetPassStrategy("GenerateMoveOpPassTestStrategy");

            auto tmp_a_0 = View(input_a, shape2, {0, 0});
            auto tmp_b_1 = View(input_b, shape2, {0, 0});

            output = Add(tmp_a_0, tmp_b_1);
        }

        std::string jsonFilePath = "./config/pass/json/generate_move_op_assemble_view_to_copy.json";
        bool dumpJsonFlag = true;
        if (dumpJsonFlag) {
            auto programJson = Program::GetInstance().DumpJson();
            DumpJsonFile(programJson, jsonFilePath);
        }
        Json readData = LoadJsonFile(jsonFilePath);
        Program::GetInstance().LoadJson(readData);

        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_ADD");

        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        auto operations = originFunction->Operations();
        for (const auto& op : operations) {
            std::cout << "opmagic: " << op.opmagic << "op type " << op.GetOpcodeStr() << std::endl;
            originOpmagic.emplace_back(op.opmagic);
        }
        GenerateMoveOp generateMoveOp;
        generateMoveOp.RunOnFunction(*originFunction);

        // ================== Verify Pass Effect ==================
        auto updatedOperations = Program::GetInstance().GetFunctionByRawName("TENSOR_ADD")->Operations();
        constexpr int expectedOperations = 4;
        EXPECT_EQ(updatedOperations.size(), expectedOperations)
            << "4 operations should remain View + Convert + Add + Assemble";
        int assemble_num = 0;
        int view_num = 0;
        int copy_in_num = 0;
        int copy_out_num = 0;
        for (const auto& updatedOperation : updatedOperations) {
            switch (updatedOperation.GetOpcode()) {
                case Opcode::OP_COPY_IN: {
                    copy_in_num++;
                    break;
                }
                case Opcode::OP_ASSEMBLE: {
                    assemble_num++;
                    break;
                }
                case Opcode::OP_COPY_OUT: {
                    copy_out_num++;
                    break;
                }
                case Opcode::OP_VIEW: {
                    view_num++;
                    break;
                }
                default:
                    break;
            }
        }
        constexpr int expectedAssemble = 0;
        constexpr int expectedView = 0;
        constexpr int expectedCopyIn = 2;
        constexpr int expectedCopyOut = 1;
        EXPECT_EQ(assemble_num, expectedAssemble) << "0 operations should be OP_ASSEMBLE";
        EXPECT_EQ(view_num, expectedView) << "0 operations should be OP_VIEW";
        EXPECT_EQ(copy_in_num, expectedCopyIn) << "2 operations should be OP_COPY_IN";
        EXPECT_EQ(copy_out_num, expectedCopyOut) << "1 operations should be OP_COPY_OUT";
    }
}

TEST_F(GenerateMoveOpPassTest, ConvertToCopy)
{
    PROGRAM("GenerateMoveOpPassTest")
    {
        std::vector<int64_t> shape1{256, 256};
        std::vector<int64_t> shape2{128, 128};
        TileShape::Current().SetVecTile({128, 128});
        Tensor input_a(DT_FP32, shape1, "input_a");
        Tensor input_b(DT_FP32, shape1, "input_b");
        Tensor output(DT_FP32, shape2, "output");
        PassManager& passManager = PassManager::Instance();
        passManager.RegisterStrategy("GenerateMoveOpPassTestStrategy",
                                     {
                                         {"RemoveRedundantReshape", PassName::REMOVE_REDUNDANT_RESHAPE},
                                         {"InferMemoryConflict", PassName::INFER_MEMORY_CONFLICT},
                                         {"ExpandFunction", PassName::EXPAND_FUNCTION},
                                         {"DuplicateOp", PassName::DUPLICATE_OP},
                                         {"MergeViewAssemble", PassName::MERGE_VIEW_ASSEMBLE},
                                         {"SplitLargeFanoutTensor", PassName::SPLIT_LARGE_FANOUT_TENSOR},
                                         {"SplitReshape", PassName::SPLIT_RESHAPE},
                                         {"AssignMemoryType", PassName::ASSIGN_MEMORY_TYPE},
                                         {"RemoveRedundantOp", PassName::REMOVE_REDUNDANT_OP},
                                         {"GenerateMoveOp", PassName::GENERATE_MOVE_OP},
                                     });
        ConfigManager::Instance();

        std::vector<int> originOpmagic;
        config::SetBuildStatic(true);
        FUNCTION("ADD", {input_a, input_b, output})
        {
            config::SetPassStrategy("GenerateMoveOpPassTestStrategy");

            auto tmp_a_0 = View(input_a, shape2, {0, 0});
            tmp_a_0.GetStorage()->SetMemoryTypeBoth(MEM_L1, true);
            auto tmp_b_1 = View(input_b, shape2, {0, 0});
            tmp_b_1.GetStorage()->SetMemoryTypeBoth(MEM_L1, true);

            output = Add(tmp_a_0, tmp_b_1);
        }

        // ================== Verify Pass Effect ==================
        auto updatedOperations = Program::GetInstance().GetFunctionByRawName("TENSOR_ADD")->Operations();
        constexpr int expectedOperations = 4;
        EXPECT_EQ(updatedOperations.size(), expectedOperations)
            << "4 operations should remain View + Convert + Add + Assemble";
        int assemble_num = 0;
        int view_num = 0;
        int copy_in_num = 0;
        int copy_out_num = 0;
        for (const auto& updatedOperation : updatedOperations) {
            switch (updatedOperation.GetOpcode()) {
                case Opcode::OP_ASSEMBLE: {
                    assemble_num++;
                    break;
                }
                case Opcode::OP_VIEW: {
                    view_num++;
                    break;
                }
                case Opcode::OP_COPY_OUT: {
                    copy_out_num++;
                    break;
                }
                case Opcode::OP_COPY_IN: {
                    copy_in_num++;
                    break;
                }
                default:
                    break;
            }
        }
        constexpr int expectedView = 0;
        constexpr int expectedAssemble = 0;
        constexpr int expectedCopyIn = 2;
        constexpr int expectedCopyOut = 1;
        EXPECT_EQ(view_num, expectedView) << "0 operations should be OP_VIEW";
        EXPECT_EQ(assemble_num, expectedAssemble) << "0 operations should be OP_ASSEMBLE";
        EXPECT_EQ(copy_in_num, expectedCopyIn) << "2 operations should be OP_COPY_IN";
        EXPECT_EQ(copy_out_num, expectedCopyOut) << "3 operations should be OP_COPY_OUT";
    }
}

TEST_F(GenerateMoveOpPassTest, Transpose)
{
    PROGRAM("GenerateMoveOpPassTest")
    {
        std::vector<int64_t> shape{1, 32, 32, 2};
        Tensor a(DT_FP32, shape, "a");
        Tensor a_trans(DT_FP32, shape, "a_trans");

        constexpr int dim0 = 1, dim1 = 16, dim2 = 16, dim3 = 2;
        TileShape::Current().SetVecTile(dim0, dim1, dim2, dim3);

        PassManager& passManager = PassManager::Instance();
        passManager.RegisterStrategy("GenerateMoveOpPassTestStrategy",
                                     {
                                         {"RemoveRedundantReshape", PassName::REMOVE_REDUNDANT_RESHAPE},
                                         {"InferMemoryConflict", PassName::INFER_MEMORY_CONFLICT},
                                         {"ExpandFunction", PassName::EXPAND_FUNCTION},
                                         {"MergeViewAssemble", PassName::MERGE_VIEW_ASSEMBLE},
                                         {"SplitLargeFanoutTensor", PassName::SPLIT_LARGE_FANOUT_TENSOR},
                                         {"SplitReshape", PassName::SPLIT_RESHAPE},
                                         {"DuplicateOp", PassName::DUPLICATE_OP},
                                         {"AssignMemoryType", PassName::ASSIGN_MEMORY_TYPE},
                                         {"RemoveRedundantOp", PassName::REMOVE_REDUNDANT_OP},

                                     });
        ConfigManager::Instance();

        FUNCTION("Tranpose") { a_trans = Transpose(a, {1, 2}); }
        std::string jsonFilePath = "./config/pass/json/generate_move_op_transpose.json";

        bool dumpJsonFlag = true;
        if (dumpJsonFlag) {
            auto programJson = Program::GetInstance().DumpJson();
            DumpJsonFile(programJson, jsonFilePath);
        }
        Json readData = LoadJsonFile(jsonFilePath);
        Program::GetInstance().LoadJson(readData);

        Function* originFunction = Program::GetInstance().GetCurrentFunction();
        GenerateMoveOp generateMoveOp;
        originFunction->DumpJsonFile("./config/pass/json/GenerateMoveOpPassTest_Transpose_before1.json");
        generateMoveOp.RunOnFunction(*originFunction);
        originFunction->DumpJsonFile("./config/pass/json/GenerateMoveOpPassTest_Transpose_after1.json");

        // ================== Verify Pass Effect ==================
        auto updatedOperations = Program::GetInstance().GetFunctionByRawName("TENSOR_Tranpose")->Operations();
        constexpr int expectedOperations = 2;
        EXPECT_EQ(updatedOperations.size(), expectedOperations) << "total 2 operations";
        int assemble_num = 0;
        int view_num = 0;
        int copy_in_num = 0;
        int copy_out_num = 0;
        int transpose_datamove_num = 0;
        int slice_num = 0;
        int contract_num = 0;
        for (const auto& updatedOperation : updatedOperations) {
            switch (updatedOperation.GetOpcode()) {
                case Opcode::OP_VIEW: {
                    view_num++;
                    break;
                }
                case Opcode::OP_COPY_IN: {
                    copy_in_num++;
                    break;
                }
                case Opcode::OP_TRANSPOSE_MOVEOUT: {
                    transpose_datamove_num++;
                    break;
                }
                case Opcode::OP_COPY_OUT: {
                    copy_out_num++;
                    break;
                }
                case Opcode::OP_ASSEMBLE: {
                    assemble_num++;
                    break;
                }
                case Opcode::OP_SLICE: {
                    slice_num++;
                    break;
                }
                case Opcode::OP_CONTRACT: {
                    contract_num++;
                    break;
                }
                default:
                    break;
            }
        }
        constexpr int expectedView = 0;
        constexpr int expectedCopyIn = 1;
        constexpr int expectedAssemble = 0;
        constexpr int expectedCopyOut = 0;
        constexpr int expectedSlice = 0;
        constexpr int expectedContract = 0;
        constexpr int expectedTransposeMoveout = 1;
        EXPECT_EQ(assemble_num, expectedAssemble) << "0 operations should be OP_ASSEMBLE";
        EXPECT_EQ(transpose_datamove_num, expectedTransposeMoveout) << "1 operations should be OP_TRANSPOSE_MOVEOUT";
        EXPECT_EQ(view_num, expectedView) << "0 operations should be OP_VIEW";
        EXPECT_EQ(slice_num, expectedSlice) << "0 operations should be OP_SLICE";
        EXPECT_EQ(contract_num, expectedContract) << "0 operations should be OP_CONTRACT";
        EXPECT_EQ(copy_in_num, expectedCopyIn) << "1 operations should be OP_COPY_IN";
        EXPECT_EQ(copy_out_num, expectedCopyOut) << "0 operations should be OP_COPY_OUT";
    }
}

// TODO: removeRedundantOp 存在疑似未合并，需要确认removeRedundantOp逻辑是否存在问题。
TEST_F(GenerateMoveOpPassTest, ScatterUpdate)
{
    PROGRAM("GenerateMoveOpPassTest")
    {
        int row = 64, col = 32;
        TileShape::Current().SetVecTile(row, col);

        PassManager& passManager = PassManager::Instance();
        passManager.RegisterStrategy("GenerateMoveOpPassTestStrategy",
                                     {
                                         {"RemoveRedundantReshape", PassName::REMOVE_REDUNDANT_RESHAPE},
                                         {"InferMemoryConflict", PassName::INFER_MEMORY_CONFLICT},
                                         {"ExpandFunction", PassName::EXPAND_FUNCTION},
                                         {"MergeViewAssemble", PassName::MERGE_VIEW_ASSEMBLE},
                                         {"SplitLargeFanoutTensor", PassName::SPLIT_LARGE_FANOUT_TENSOR},
                                         {"SplitReshape", PassName::SPLIT_RESHAPE},
                                         {"DuplicateOp", PassName::DUPLICATE_OP},
                                         {"AssignMemoryType", PassName::ASSIGN_MEMORY_TYPE},
                                         {"RemoveRedundantOp", PassName::REMOVE_REDUNDANT_OP},
                                     });
        ConfigManager::Instance();

        int b = 2, s = 64, numExpertsPerTok = 2, h = 128, minus_two = -2;
        Tensor output(DT_FP32, {b * s * numExpertsPerTok, h}, "output");
        Tensor idxs(DT_INT64, {1, b * s * numExpertsPerTok}, "idxs");
        Tensor key_states(DT_FP32, {b * s * numExpertsPerTok, h}, "key_states");
        FUNCTION("ScatterUpdate") { output = ScatterUpdate(output, {idxs}, key_states, minus_two); }
        std::string jsonFilePath = "./config/pass/json/generate_move_op_scatter_update.json";
        bool dumpJsonFlag = false;
        if (dumpJsonFlag) {
            auto programJson = Program::GetInstance().DumpJson();
            DumpJsonFile(programJson, jsonFilePath);
        }

        Function* originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_ScatterUpdate");
        GenerateMoveOp generateMoveOp;
        generateMoveOp.RunOnFunction(*originFunction);

        // ================== Verify Pass Effect ==================
        // auto updatedOperations = Program::GetInstance().GetFunctionByRawName("TENSOR_ScatterUpdate")->Operations();
        // constexpr int expectedOperations = 112;
        // EXPECT_EQ(updatedOperations.size(), expectedOperations) << "total 112 operations";
        // int assemble_num = 0;
        // int view_num = 0;
        // int copy_in_num = 0;
        // int copy_out_num = 0;
        // int index_outcast_num = 0;
        // int slice_num = 0;
        // int contract_num = 0;
        // for (const auto& updatedOperation : updatedOperations) {
        //     switch (updatedOperation.GetOpcode()) {
        //         case Opcode::OP_INDEX_OUTCAST: {
        //             index_outcast_num++;
        //             break;
        //         }
        //         case Opcode::OP_ASSEMBLE: {
        //             assemble_num++;
        //             break;
        //         }
        //         case Opcode::OP_COPY_IN: {
        //             copy_in_num++;
        //             break;
        //         }
        //         case Opcode::OP_COPY_OUT: {
        //             copy_out_num++;
        //             break;
        //         }
        //         case Opcode::OP_VIEW: {
        //             view_num++;
        //             break;
        //         }
        //         case Opcode::OP_SLICE: {
        //             slice_num++;
        //             break;
        //         }
        //         case Opcode::OP_CONTRACT: {
        //             contract_num++;
        //             break;
        //         }
        //         default:
        //             break;
        //     }
        // }
        // constexpr int expectedAssemble = 12;
        // constexpr int expectedView = 4;
        // constexpr int expectedIndexOutcast = 4;
        // constexpr int expectedCopyOut = 0;
        // EXPECT_EQ(assemble_num, expectedAssemble) << "12 operations should be OP_ASSEMBLE";
        // EXPECT_EQ(view_num, expectedView) << "4 operations should be OP_VIEW";
        // EXPECT_EQ(index_outcast_num, expectedIndexOutcast) << "4 operations should be OP_INDEX_OUTCAST";
        // EXPECT_EQ(copy_out_num, expectedCopyOut) << "0 operations should be OP_COPY_OUT";
    }
}

TEST_F(GenerateMoveOpPassTest, L1TOL0)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "L1TOL0", "L1TOL0", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    Program::GetInstance().InsertFuncToFunctionMap("L1TOL0", currFunctionPtr);
    constexpr int opMagic0 = 1001;
    constexpr int opMagic1 = 1002;
    constexpr int opMagic2 = 1003;

    constexpr int tensorMagic0 = 1;
    constexpr int tensorMagic1 = 2;
    constexpr int tensorMagic2 = 3;
    constexpr int tensorMagic3 = 4;
    constexpr int tensorMagic4 = 5;

    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    std::vector<int64_t> shape1 = {16, 8};
    std::vector<int64_t> shape2 = {8, 8};
    std::shared_ptr<LogicalTensor> input_a = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape, CreateTestConstIntVector(shape));
    input_a->SetMagic(tensorMagic0);
    input_a->SetMemoryTypeOriginal(MemoryType::MEM_L1);

    std::shared_ptr<LogicalTensor> tmp_a = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape,
                                                                                      CreateTestConstIntVector(shape));
    tmp_a->SetMagic(tensorMagic1);
    tmp_a->SetMemoryTypeOriginal(MemoryType::MEM_L0A);

    std::shared_ptr<LogicalTensor> input_b = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape1, CreateTestConstIntVector(shape1));
    input_b->SetMagic(tensorMagic2);
    input_b->SetMemoryTypeOriginal(MemoryType::MEM_L1);

    std::shared_ptr<LogicalTensor> tmp_b = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1,
                                                                                      CreateTestConstIntVector(shape1));
    tmp_b->SetMagic(tensorMagic3);
    tmp_b->SetMemoryTypeOriginal(MemoryType::MEM_L0B);

    std::shared_ptr<LogicalTensor> output_c = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape2, CreateTestConstIntVector(shape2));
    output_c->SetMagic(tensorMagic4);

    auto& convert_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CONVERT, {input_a}, {tmp_a});
    convert_op1.opmagic = opMagic0;
    convert_op1.SetOpAttribute(std::make_shared<ConvertOpAttribute>(MemoryType::MEM_L1, MemoryType::MEM_L0A));

    auto& convert_op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CONVERT, {input_b}, {tmp_b});
    convert_op2.opmagic = opMagic1;
    convert_op2.SetOpAttribute(std::make_shared<ConvertOpAttribute>(MemoryType::MEM_L1, MemoryType::MEM_L0B));

    auto& matmul_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_A_MUL_B, {tmp_a, tmp_b}, {output_c});
    matmul_op.opmagic = opMagic2;

    currFunctionPtr->inCasts_.push_back(input_a);
    currFunctionPtr->inCasts_.push_back(input_b);
    currFunctionPtr->outCasts_.push_back(output_c);

    std::stringstream ssBefore;
    ssBefore << "Before_GenerateMoveOp";

    // Call the pass
    ConvertInserter inserter;
    inserter.CreateMoveOpForConvert(convert_op1);
    inserter.CreateMoveOpForConvert(convert_op2);
    GenerateMoveOp generateMoveOp;
    generateMoveOp.PreCheck(*currFunctionPtr);
    generateMoveOp.RunOnFunction(*currFunctionPtr);
    generateMoveOp.PostCheck(*currFunctionPtr);

    std::stringstream ss;
    ss << "After_GenerateMoveOp";

    // Validate the results
    std::cout << "========== op size: " << currFunctionPtr->Operations().size() << std::endl;
    int convert_num = 0;
    int l1tol0a_num = 0;
    int l1tol0B_num = 0;
    for (auto& op : currFunctionPtr->Operations()) {
        std::cout << op.GetOpcodeStr() << " " << op.GetOpMagic() << std::endl;
        for (auto& input : op.GetIOperands()) {
            std::cout << "\t|--- iOperand " << input->magic;
        }
        for (auto& output : op.GetOOperands()) {
            std::cout << "\t|--- oOperand " << output->magic << std::endl;
        }
        if (op.GetOpcode() == Opcode::OP_CONVERT) {
            convert_num++;
        } else if (op.GetOpcode() == Opcode::OP_L1_TO_L0A) {
            l1tol0a_num++;
        } else if (op.GetOpcode() == Opcode::OP_L1_TO_L0B) {
            l1tol0B_num++;
        }
    }
    constexpr int expectedConvert = 0;
    constexpr int expectedL1tol0a = 1;
    constexpr int expectedL1tol0b = 1;
    EXPECT_EQ(convert_num, expectedConvert) << "0 operations shoulde be OP_VIEW.";
    EXPECT_EQ(l1tol0a_num, expectedL1tol0a) << "1 operations shoulde be OP_COPY_IN.";
    EXPECT_EQ(l1tol0B_num, expectedL1tol0b) << "1 operations shoulde be OP_COPY_OUT.";
}
void TransViewTensorWithAttr(std::shared_ptr<Function>& currFunctionPtr)
{
    std::shared_ptr<LogicalTensor> view_in1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    view_in1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    std::shared_ptr<LogicalTensor> tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    tensor1->SetMemoryTypeOriginal(MemoryType::MEM_L1);
    std::shared_ptr<LogicalTensor> view_out1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    view_out1->SetMemoryTypeOriginal(MemoryType::MEM_BT);

    std::shared_ptr<LogicalTensor> view_in2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    view_in2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    std::shared_ptr<LogicalTensor> tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    tensor2->SetMemoryTypeOriginal(MemoryType::MEM_L1);
    std::shared_ptr<LogicalTensor> view_out2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    view_out2->SetMemoryTypeOriginal(MemoryType::MEM_FIX_QUANT_PRE);

    std::shared_ptr<LogicalTensor> output = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    output->SetMemoryTypeOriginal(MemoryType::MEM_L0C);

    auto& view_op4 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_SLICE, {tensor2}, {view_out2});
    auto viewAttribute4 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute4->SetToType(MemoryType::MEM_FIX_QUANT_PRE);
    view_op4.SetOpAttribute(viewAttribute4);
    auto& view_op3 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_SLICE, {view_in2}, {tensor2});
    auto viewAttribute3 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute3->SetToType(MemoryType::MEM_L1);
    view_op3.SetOpAttribute(viewAttribute3);
    auto& view_op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_SLICE, {tensor1}, {view_out1});
    auto viewAttribute2 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute2->SetToType(MemoryType::MEM_BT);
    view_op2.SetOpAttribute(viewAttribute2);
    auto& view_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_SLICE, {view_in1}, {tensor1});
    auto viewAttribute1 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute1->SetToType(MemoryType::MEM_L1);
    view_op1.SetOpAttribute(viewAttribute1);

    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_A_MUL_B, {view_out1, view_out2}, {output});

    currFunctionPtr->inCasts_.push_back(view_in1);
    currFunctionPtr->inCasts_.push_back(view_in2);
    currFunctionPtr->outCasts_.push_back(output);
}
TEST_F(GenerateMoveOpPassTest, TransViewWithAttr)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TransViewWithAttr", "TransViewWithAttr",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("TransViewWithAttr", currFunctionPtr);

    TransViewTensorWithAttr(currFunctionPtr);

    std::stringstream ssBefore;
    ssBefore << "Before_GenerateMoveOp";

    // Call the pass
    GenerateMoveOp generateMoveOp;
    generateMoveOp.PreCheck(*currFunctionPtr);
    currFunctionPtr->DumpJsonFile("./config/pass/json/generateMoveOp_TransViewWithAttr_before.json");
    generateMoveOp.RunOnFunction(*currFunctionPtr);
    currFunctionPtr->DumpJsonFile("./config/pass/json/generateMoveOp_TransViewWithAttr_after.json");

    std::stringstream ss;
    ss << "After_GenerateMoveOp";

    // Validate the results
    int copyIn_count_after_pass = 0;
    int l12Bt_count_after_pass = 0;
    int l12Fb_count_after_pass = 0;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_COPY_IN) {
            copyIn_count_after_pass++;
        }
        if (op.GetOpcode() == Opcode::OP_L1_TO_BT) {
            l12Bt_count_after_pass++;
        }
        if (op.GetOpcode() == Opcode::OP_L1_TO_FIX_QUANT_PRE) {
            l12Fb_count_after_pass++;
        }
    }
    constexpr int expectedCopyIn = 2;
    constexpr int expectedL1toBt = 1;
    constexpr int expectedL1toFb = 1;
    EXPECT_EQ(copyIn_count_after_pass, expectedCopyIn) << "2 operations shoulde be OP_COPY_IN.";
    EXPECT_EQ(l12Bt_count_after_pass, expectedL1toBt) << "1 operations shoulde be OP_L1_TO_BT.";
    EXPECT_EQ(l12Fb_count_after_pass, expectedL1toFb) << "1 operations shoulde be OP_L1_TO_FIX_QUANT_PRE.";
}
void ViewconnectAssemble(std::shared_ptr<Function>& currFunctionPtr)
{
    std::shared_ptr<LogicalTensor> input = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    input->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    std::shared_ptr<LogicalTensor> tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    tensor1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    std::shared_ptr<LogicalTensor> output = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    output->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);

    auto& view_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {input}, {tensor1});
    auto viewAttribute = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute->SetToType(MemoryType::MEM_DEVICE_DDR);
    view_op.SetOpAttribute(viewAttribute);
    auto& assemble_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_ASSEMBLE, {tensor1}, {output});
    auto assembleAttribute = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0});
    assemble_op.SetOpAttribute(assembleAttribute);

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(output);
}
TEST_F(GenerateMoveOpPassTest, ViewconnectAssemble)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "ViewconnectAssemble",
                                                      "ViewconnectAssemble", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("ViewconnectAssemble", currFunctionPtr);

    ViewconnectAssemble(currFunctionPtr);

    std::stringstream ssBefore;
    ssBefore << "Before_GenerateMoveOp";

    // Call the pass
    GenerateMoveOp generateMoveOp;
    generateMoveOp.PreCheck(*currFunctionPtr);
    currFunctionPtr->DumpJsonFile("./config/pass/json/generateMoveOp_ViewconnectAssemble_before.json");
    generateMoveOp.RunOnFunction(*currFunctionPtr);
    currFunctionPtr->DumpJsonFile("./config/pass/json/generateMoveOp_ViewconnectAssemble_after.json");

    std::stringstream ss;
    ss << "After_GenerateMoveOp";

    // Validate the results
    int check_Op_inputsMemType = 0;
    for (auto& op : currFunctionPtr->Operations()) {
        auto consumerOps = op.oOperand[0]->GetConsumers();
        for (auto childOp : consumerOps) {
            auto opcode = childOp->GetOpcode();
            const auto& inputsMemType = OpcodeManager::Inst().GetInputsMemType(opcode);
            if (inputsMemType.empty()) {
                check_Op_inputsMemType++;
            }
        }
    }
    constexpr int expectedcheck = 1;
    EXPECT_EQ(check_Op_inputsMemType, expectedcheck) << "1 operation inputsMemType shoulde be OP_COPY_IN.";
}

// 辅助函数：构造测试用LogicalTensor
std::shared_ptr<LogicalTensor> CreateTestLogicalTensor(MemoryType memType, TileOpFormat format,
                                                       const std::vector<int64_t>& shape)
{
    auto rawTensor = std::make_shared<RawTensor>(DT_FP32, shape, format);
    std::vector<int64_t> offset(shape.size(), 0);
    auto logicalTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(rawTensor, offset, shape,
                                                                    CreateTestConstIntVector(shape));
    logicalTensor->SetMemoryTypeBoth(memType);
    return logicalTensor;
}

// 辅助函数：构造测试用Operation
std::shared_ptr<Operation> CreateTestOperation(Opcode opcode, Function& func,
                                               const std::vector<std::shared_ptr<LogicalTensor>>& inputs,
                                               const std::vector<std::shared_ptr<LogicalTensor>>& outputs)
{
    auto op = std::make_shared<Operation>(func, opcode);
    op->iOperand = inputs;
    op->oOperand = outputs;
    op->opmagic = 1001;
    return op;
}
// ========== 测试用例1：PadUB函数全覆盖 ==========
TEST_F(GenerateMoveOpPassTest, PadUBFullCoverage)
{
    PROGRAM("PadUBFullCoverage")
    {
        // 覆盖PadUB所有分支场景
        // 场景1：刚好对齐
        EXPECT_EQ(GenerateMoveOp::PadUB(32, 32), 32);
        // 场景2：需要向上取整（33→64）
        EXPECT_EQ(GenerateMoveOp::PadUB(33, 32), 64);
        // 场景3：边界值（15→16）
        EXPECT_EQ(GenerateMoveOp::PadUB(15, 16), 16);
        // 场景4：dim=0
        EXPECT_EQ(GenerateMoveOp::PadUB(0, 32), 0);
        // 场景5：pad=1（最小有效值）
        EXPECT_EQ(GenerateMoveOp::PadUB(5, 1), 5);

        // 验证字节对齐场景（模拟DT_FLOAT=4字节）
        int64_t dtypeBytes = 4;
        EXPECT_EQ(GenerateMoveOp::PadUB(30, 32 / dtypeBytes), 32); // 30→32（8字节对齐）
        EXPECT_EQ(GenerateMoveOp::PadUB(15, 16), 16);              // 15→16（16字节对齐）
    }
}

// ========== 测试用例2：ProcessUB2L1函数全覆盖（ND→NZ格式转换） ==========
TEST_F(GenerateMoveOpPassTest, ProcessUB2L1FullCoverage)
{
    PROGRAM("ProcessUB2L1FullCoverage")
    {
        std::vector<int64_t> ndShape{2, 30, 15}; // inner=30, outer=15（非对齐）
        Tensor a(DT_FP32, ndShape, "a");
        Tensor b(DT_FP32, ndShape, "b");

        // 设置TileShape（模拟真实场景）
        TileShape::Current().SetVecTile(2, 30, 15, 1);

        // 注册基础Pass策略（模拟真实流程）
        PassManager& passManager = PassManager::Instance();
        passManager.RegisterStrategy("ProcessUB2L1Strategy", {{"AssignMemoryType", PassName::ASSIGN_MEMORY_TYPE},
                                                              {"GenerateMoveOp", PassName::GENERATE_MOVE_OP}});

        FUNCTION("ProcessUB2L1Func")
        {
            b = View(a, ndShape, {0, 0, 0}); // 构造VIEW OP作为基础
        }

        // 导出/导入JSON（模拟真实流程）
        std::string jsonPath = "./config/pass/json/process_ub2l1.json";
        auto programJson = Program::GetInstance().DumpJson();
        DumpJsonFile(programJson, jsonPath);
        Json readData = LoadJsonFile(jsonPath);
        Program::GetInstance().LoadJson(readData);

        // 获取Function并构造OP_UB_COPY_L1操作
        Function* func = Program::GetInstance().GetCurrentFunction();
        auto inputTensor = CreateTestLogicalTensor(MEM_UB, TileOpFormat::TILEOP_ND, ndShape);
        auto outputTensor = CreateTestLogicalTensor(MEM_L1, TileOpFormat::TILEOP_NZ, ndShape);
        auto ubCopyL1Op = CreateTestOperation(Opcode::OP_UB_COPY_L1, *func, {inputTensor}, {outputTensor});
        ubCopyL1Op->UpdateSubgraphID(0);

        // 调用ProcessUB2L1
        GenerateMoveOp generateMoveOp;
        generateMoveOp.ProcessUB2L1(*func, *ubCopyL1Op);

        // ================== 验证核心逻辑 ==================

        // 1. 验证插入OP_UB_COPY_ND2NZ节点
        int ub2ubNum = 0;
        for (const auto& op : func->Operations()) {
            if (op.GetOpcode() == Opcode::OP_UB_COPY_ND2NZ) {
                ub2ubNum++;
                EXPECT_EQ(op.GetSubgraphID(), 0);
                EXPECT_EQ(op.iOperand.front()->GetMemoryTypeOriginal(), MEM_UB);
                EXPECT_EQ(op.iOperand.front()->Format(), TileOpFormat::TILEOP_ND);
            }
        }
        EXPECT_EQ(ub2ubNum, 1) << "Must insert one OP_UB_COPY_ND2NZ op";

        // 2. 验证新UB NZ Tensor
        auto newInputTensor = ubCopyL1Op->iOperand.front();
        EXPECT_EQ(newInputTensor->Format(), TileOpFormat::TILEOP_NZ);
    }
}

// ========== 测试用例3：CreateMoveOpForView触发OP_UB_COPY_L1（106行覆盖） ==========
TEST_F(GenerateMoveOpPassTest, CreateMoveOpForViewUB2L1)
{
    PROGRAM("CreateMoveOpForViewUB2L1")
    {
        std::vector<int64_t> shape{2, 30, 15};
        Tensor a(DT_FP32, shape, "a");
        Tensor b(DT_FP32, shape, "b");

        TileShape::Current().SetVecTile(2, 30, 15, 1);

        // 注册Pass策略
        PassManager& passManager = PassManager::Instance();
        passManager.RegisterStrategy("GenerateMoveOpViewStrategy",
                                     {{"AssignMemoryType", PassName::ASSIGN_MEMORY_TYPE},
                                      {"MergeViewAssemble", PassName::MERGE_VIEW_ASSEMBLE},
                                      {"GenerateMoveOp", PassName::GENERATE_MOVE_OP}});

        config::SetHostOption(COMPILE_STAGE, CS_TENSOR_GRAPH);
        FUNCTION("View2UBCopyL1Func")
        {
            b = View(a, shape, {0, 0, 0}); // 构造VIEW OP
        }
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);

        // 导出/导入JSON
        std::string jsonPath = "./config/pass/json/view_ub2l1.json";
        auto programJson = Program::GetInstance().DumpJson();
        DumpJsonFile(programJson, jsonPath);
        Json readData = LoadJsonFile(jsonPath);
        Program::GetInstance().LoadJson(readData);

        // 获取Function并修改VIEW OP的内存类型（UB→L1）
        Function* func = Program::GetInstance().GetCurrentFunction();
        Operation* viewOp = nullptr;
        for (auto& op : func->Operations()) {
            if (op.GetOpcode() == Opcode::OP_VIEW) {
                viewOp = &op;
                // 设置输入为UB，输出为L1（触发OP_UB_COPY_L1）
                op.iOperand.front()->SetMemoryTypeBoth(MEM_UB);
                op.oOperand.front()->SetMemoryTypeBoth(MEM_L1);
                break;
            }
        }
        ASSERT_NE(viewOp, nullptr) << "VIEW op not found";

        // 执行CreateMoveOpForView
        GenerateMoveOp generateMoveOp;
        Status status = generateMoveOp.A23CreateMoveOpForView(*func, *viewOp);

        // ================== 验证核心逻辑 ==================
        EXPECT_EQ(status, SUCCESS);
        EXPECT_EQ(viewOp->GetOpcode(), Opcode::OP_UB_COPY_L1) << "VIEW should convert to OP_UB_COPY_L1";

        // 验证ProcessUB2L1被调用（ND→NZ格式转换）
        auto inputTensor = viewOp->iOperand.front();
        EXPECT_EQ(inputTensor->Format(), TileOpFormat::TILEOP_NZ);
    }
}

// ========== 测试用例4：ProcessUB2L1非ND格式（无操作分支） ==========
TEST_F(GenerateMoveOpPassTest, ProcessUB2L1NonNDFormat)
{
    PROGRAM("ProcessUB2L1NonNDFormat")
    {
        std::vector<int64_t> nzShape{2, 32, 16};
        Tensor a(DT_FP32, nzShape, "a");
        Tensor b(DT_FP32, nzShape, "b");

        TileShape::Current().SetVecTile(2, 32, 16, 1);

        FUNCTION("ProcessUB2L1NonNDFunc") { b = View(a, nzShape, {0, 0, 0}); }

        // 导出/导入JSON
        std::string jsonPath = "./config/pass/json/ub2l1_non_nd.json";
        auto programJson = Program::GetInstance().DumpJson();
        DumpJsonFile(programJson, jsonPath);
        Json readData = LoadJsonFile(jsonPath);
        Program::GetInstance().LoadJson(readData);

        // 构造NZ格式UB Tensor的OP_UB_COPY_L1
        Function* func = Program::GetInstance().GetCurrentFunction();
        auto inputTensor = CreateTestLogicalTensor(MEM_UB, TileOpFormat::TILEOP_NZ, nzShape);
        auto outputTensor = CreateTestLogicalTensor(MEM_L1, TileOpFormat::TILEOP_NZ, nzShape);
        auto ubCopyL1Op = CreateTestOperation(Opcode::OP_UB_COPY_L1, *func, {inputTensor}, {outputTensor});

        // 调用ProcessUB2L1
        GenerateMoveOp generateMoveOp;
        generateMoveOp.ProcessUB2L1(*func, *ubCopyL1Op);

        // 验证：无新节点插入，Tensor格式不变
        int ub2ubNum = 0;
        for (const auto& op : func->Operations()) {
            if (op.GetOpcode() == Opcode::OP_UB_COPY_ND2NZ) {
                ub2ubNum++;
            }
        }
        EXPECT_EQ(ub2ubNum, 0) << "No OP_UB_COPY_ND2NZ should be inserted for NZ format";
        EXPECT_EQ(ubCopyL1Op->iOperand.front()->Format(), TileOpFormat::TILEOP_NZ);
    }
}

TEST_F(GenerateMoveOpPassTest, ProcessDefault_L0C_UB_SetIsCube)
{
    PROGRAM("ProcessDefault_L0C_UB_SetIsCube")
    {
        std::vector<int64_t> shape{16, 16};
        auto func = std::make_shared<Function>(Program::GetInstance(), "ProcessDefault_L0C_UB_Func",
                                               "ProcessDefault_L0C_UB_Func", nullptr);
        Program::GetInstance().InsertFuncToFunctionMap("ProcessDefault_L0C_UB_Func", func);

        // 创建输入tensor（L0C）
        auto l0cRawTensor = std::make_shared<RawTensor>(DT_FP32, shape, TileOpFormat::TILEOP_ND);
        std::vector<int64_t> l0cOffset(shape.size(), 0);
        auto l0cTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(l0cRawTensor, l0cOffset, shape,
                                                                    CreateTestConstIntVector(shape));
        l0cTensor->SetMemoryTypeOriginal(MEM_L0C);
        l0cTensor->SetMemoryTypeToBe(MEM_L0C);

        // 创建输出tensor（UB）
        auto ubRawTensor = std::make_shared<RawTensor>(DT_FP32, shape, TileOpFormat::TILEOP_ND);
        std::vector<int64_t> ubOffset(shape.size(), 0);
        auto ubTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(ubRawTensor, ubOffset, shape,
                                                                   CreateTestConstIntVector(shape));
        ubTensor->SetMemoryTypeOriginal(MEM_UB);
        ubTensor->SetMemoryTypeToBe(MEM_UB);

        // 使用AddRawOperation将OP_SLICE添加到func中
        auto& viewOp = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_SLICE, {l0cTensor}, {ubTensor});

        // 验证输入输出内存类型不同
        ASSERT_EQ(viewOp.iOperand.front()->GetMemoryTypeOriginal(), MEM_L0C) << "Input memory type should be L0C";
        ASSERT_EQ(viewOp.oOperand.front()->GetMemoryTypeOriginal(), MEM_UB) << "Output memory type should be UB";

        // 设置ViewOpAttribute
        auto viewAttr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}, MemoryType::MEM_UB);
        viewOp.SetOpAttribute(viewAttr);

        // 直接调用ProcessDefault函数
        GenerateMoveOp generateMoveOp;
        Status status = generateMoveOp.ProcessDefault(*func, viewOp, viewAttr.get());
        ASSERT_EQ(status, SUCCESS);

        // 验证OP_VIEW已被转换为OP_L0C_COPY_UB
        EXPECT_EQ(viewOp.GetOpcode(), Opcode::OP_L0C_COPY_UB) << "VIEW should convert to OP_L0C_COPY_UB";

        // 验证isCube属性已设置为true
        ASSERT_TRUE(viewOp.HasAttribute(OpAttributeKey::isCube)) << "OP_L0C_COPY_UB should have isCube attribute";
        EXPECT_TRUE(viewOp.GetBoolAttribute(OpAttributeKey::isCube)) << "isCube should be true for OP_L0C_COPY_UB";
    }
}

TEST_F(GenerateMoveOpPassTest, SetOpcodeByMemPath)
{
    GenerateMoveOp generateMoveOp;
    Program& program = Program::GetInstance();
    std::shared_ptr<Function> testFunc = std::make_shared<Function>(program, "test_func_magic", "test_func_raw",
                                                                    nullptr);

    LogicalTensors emptyIOperands;
    LogicalTensors emptyOOperands;
    Operation& testOp = IRBuilder().CreateTensorOpStmt(*testFunc, Opcode::OP_VIEW, emptyIOperands, emptyOOperands);
    Status ret = generateMoveOp.SetOpcodeByMemPath(testOp, MemoryType::MEM_L0AMX, MemoryType::MEM_L0BMX);
    EXPECT_EQ(ret, FAILED);
}

TEST_F(GenerateMoveOpPassTest, CreateMoveOpForAssemble_L0C2UB)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "L0C2UB", "L0C2UB", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("L0C2UB", currFunctionPtr);

    // 创建输入 tensor (L0C)
    std::vector<int64_t> shape{32, 64};
    auto inputTensor = CreateTestLogicalTensor(MEM_L0C, TileOpFormat::TILEOP_NZ, shape);
    // 创建输出 tensor (UB)
    auto outputTensor = CreateTestLogicalTensor(MEM_UB, TileOpFormat::TILEOP_ND, shape);

    // 创建 Contract 操作
    auto& assembleOp = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CONTRACT, {inputTensor},
                                                      {outputTensor});
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0});
    assembleOp.SetOpAttribute(assembleAttr);

    // 调用 CreateMoveOpForAssemble
    GenerateMoveOp generateMoveOp;
    generateMoveOp.CreateMoveOpForAssemble(*currFunctionPtr, assembleOp);

    // 验证：Opcode 应变为 OP_L0C_COPY_UB
    EXPECT_EQ(assembleOp.GetOpcode(), Opcode::OP_L0C_COPY_UB);
    EXPECT_TRUE(assembleOp.HasAttribute(OpAttributeKey::isCube));
}

TEST_F(GenerateMoveOpPassTest, CreateMoveOpForAssemble_UB2L1)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "UB2L1", "UB2L1", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("UB2L1", currFunctionPtr);
    std::vector<int64_t> shape{32, 64};
    auto inputTensor = CreateTestLogicalTensor(MEM_UB, TileOpFormat::TILEOP_ND, shape);
    // 创建输出 tensor (L1)
    auto outputTensor = CreateTestLogicalTensor(MEM_L1, TileOpFormat::TILEOP_NZ, shape);

    // 创建 Contract 操作
    auto& assembleOp = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CONTRACT, {inputTensor},
                                                      {outputTensor});
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0});
    assembleOp.SetOpAttribute(assembleAttr);

    // 调用 CreateMoveOpForAssemble
    GenerateMoveOp generateMoveOp;
    generateMoveOp.CreateMoveOpForAssemble(*currFunctionPtr, assembleOp);

    // 验证：Opcode 应变为 OP_UB_COPY_L1
    EXPECT_EQ(assembleOp.GetOpcode(), Opcode::OP_UB_COPY_L1);
}

TEST_F(GenerateMoveOpPassTest, CreateMoveOpForAssemble_A5BypassRedundantL1)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "A5BypassRedundantL1",
                                                      "A5BypassRedundantL1", nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("A5BypassRedundantL1", currFunctionPtr);

    std::vector<int64_t> shape{32, 64};
    auto ubTensor = CreateTestLogicalTensor(MEM_UB, TileOpFormat::TILEOP_ND, shape);
    auto l1Tensor = CreateTestLogicalTensor(MEM_L1, TileOpFormat::TILEOP_NZ, shape);
    auto gmTensor = CreateTestLogicalTensor(MEM_DEVICE_DDR, TileOpFormat::TILEOP_ND, shape);
    auto& ubCopyL1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_UB_COPY_L1, {ubTensor}, {l1Tensor});
    auto& assembleOp = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CONTRACT, {l1Tensor}, {gmTensor});
    assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_L1, std::vector<int64_t>{0, 0}));

    auto oldArch = Platform::Instance().GetSoc().GetNPUArch();
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    GenerateMoveOp().CreateMoveOpForAssemble(*currFunctionPtr, assembleOp);
    Platform::Instance().GetSoc().SetNPUArch(oldArch);

    EXPECT_EQ(assembleOp.GetOpcode(), Opcode::OP_COPY_OUT);
    EXPECT_EQ(assembleOp.GetIOperands().front(), ubTensor);
    auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(assembleOp.GetOpAttribute());
    EXPECT_NE(copyAttr, nullptr);
    if (copyAttr != nullptr) {
        EXPECT_EQ(copyAttr->GetCopyOutAttr().first, MEM_UB);
    }
    EXPECT_TRUE(ubCopyL1.IsDeleted());
}
// ========== ProcessL1CopyInConv 测试 ==========

TEST_F(GenerateMoveOpPassTest, l1CopyInConvOffsetAccumulation)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "l1CopyInConvAcc", "l1CopyInConvAcc", nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("l1CopyInConvAcc", func);

    std::vector<int64_t> shape{16, 16};
    auto input = CreateTestLogicalTensor(MEM_DEVICE_DDR, TileOpFormat::TILEOP_ND, shape);
    auto mid = CreateTestLogicalTensor(MEM_L1, TileOpFormat::TILEOP_ND, shape);
    auto output = CreateTestLogicalTensor(MEM_L1, TileOpFormat::TILEOP_ND, shape);

    // SLICE: fromOffset=[1, 2]
    auto& viewOp = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_SLICE, {input}, {mid});
    viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{1, 2}));

    // L1_COPY_IN_CONV: fromOffset=[3, 4]
    auto& copyOp = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_L1_COPY_IN_CONV, {mid}, {output});
    std::vector<OpImmediate> shapeImm = {OpImmediate::Specified(IRBuilder().CreateConstInt(shape[0])),
                                         OpImmediate::Specified(IRBuilder().CreateConstInt(shape[1]))};
    std::vector<OpImmediate> fromOffset = {OpImmediate::Specified(IRBuilder().CreateConstInt(3)),
                                           OpImmediate::Specified(IRBuilder().CreateConstInt(4))};
    copyOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(fromOffset, MEM_L1, shapeImm, shapeImm));

    GenerateMoveOp pass;
    EXPECT_EQ(pass.ProcessL1CopyInConv(copyOp), SUCCESS);

    // 验证: VIEW 被删除，输入重连
    EXPECT_TRUE(viewOp.IsDeleted());
    EXPECT_EQ(copyOp.GetIOperands()[0], input);

    // 验证: fromOffset 累加 [1+3, 2+4] = [4, 6]
    auto mergedAttr = std::dynamic_pointer_cast<CopyOpAttribute>(copyOp.GetOpAttribute());
    ASSERT_NE(mergedAttr, nullptr);
    auto scalars = OpImmediate::ToSpecified(mergedAttr->GetFromOffset());
    ASSERT_EQ(scalars.size(), 2);
    EXPECT_TRUE(scalars[0].ConcreteValid());
    EXPECT_EQ(scalars[0].Concrete(), 4);
    EXPECT_TRUE(scalars[1].ConcreteValid());
    EXPECT_EQ(scalars[1].Concrete(), 6);
}

TEST_F(GenerateMoveOpPassTest, l1CopyInConvSymbolicScalarOffsetAccumulation)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "l1CopyInConvSym", "l1CopyInConvSym", nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("l1CopyInConvSym", func);

    std::vector<int64_t> shape{16, 16};
    auto input = CreateTestLogicalTensor(MEM_DEVICE_DDR, TileOpFormat::TILEOP_ND, shape);
    auto mid = CreateTestLogicalTensor(MEM_L1, TileOpFormat::TILEOP_ND, shape);
    auto output = CreateTestLogicalTensor(MEM_L1, TileOpFormat::TILEOP_ND, shape);

    auto viewDynA = CreateTestScalarVar("a");
    auto viewDynB = CreateTestScalarVar("b");
    auto copyOffsetC = CreateTestScalarVar("c");
    auto copyOffsetD = CreateTestScalarVar("d");

    // SLICE: fromOffset=[1, 2] + fromDynOffset=[Sym("a"), Sym("b")]
    auto& viewOp = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_SLICE, {input}, {mid});
    auto viewAttr = std::make_shared<ViewOpAttribute>(
        std::vector<int64_t>{1, 2}, std::vector<SymbolicScalar>{viewDynA, viewDynB}, std::vector<SymbolicScalar>{});
    viewOp.SetOpAttribute(viewAttr);

    // L1_COPY_IN_CONV: fromOffset=[Sym("c"), Sym("d")]
    auto& copyOp = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_L1_COPY_IN_CONV, {mid}, {output});
    std::vector<OpImmediate> shapeImm = {OpImmediate::Specified(IRBuilder().CreateConstInt(shape[0])),
                                         OpImmediate::Specified(IRBuilder().CreateConstInt(shape[1]))};
    std::vector<OpImmediate> fromOffset = {OpImmediate::Specified(copyOffsetC), OpImmediate::Specified(copyOffsetD)};
    copyOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(fromOffset, MEM_L1, shapeImm, shapeImm));

    GenerateMoveOp pass;
    EXPECT_EQ(pass.ProcessL1CopyInConv(copyOp), SUCCESS);

    // 验证: VIEW 被删除，输入重连
    EXPECT_TRUE(viewOp.IsDeleted());
    EXPECT_EQ(copyOp.GetIOperands()[0], input);

    // 验证: fromOffset 已累加 (包含符号表达式)
    auto mergedAttr = std::dynamic_pointer_cast<CopyOpAttribute>(copyOp.GetOpAttribute());
    ASSERT_NE(mergedAttr, nullptr);
    auto& mergedOffset = mergedAttr->GetFromOffset();
    ASSERT_EQ(mergedOffset.size(), 2);
    auto scalars = OpImmediate::ToSpecified(mergedOffset);
    EXPECT_EQ(scalars[0].Dump(), "(" + viewDynA.Dump() + "+" + copyOffsetC.Dump() + ")");
    EXPECT_EQ(scalars[1].Dump(), "(" + viewDynB.Dump() + "+" + copyOffsetD.Dump() + ")");
}

TEST_F(GenerateMoveOpPassTest, l1CopyInConvDynRawShapePropagation)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "l1CopyInConvDynRaw", "l1CopyInConvDynRaw", nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("l1CopyInConvDynRaw", func);

    std::vector<int64_t> inputShape{-1, -1};
    std::vector<int64_t> viewShape{16, 16};
    auto input = CreateTestLogicalTensor(MEM_DEVICE_DDR, TileOpFormat::TILEOP_ND, inputShape);
    input->tensor->UpdateDynRawShape({CreateTestScalarVar("N"), CreateTestScalarVar("C")});
    auto mid = CreateTestLogicalTensor(MEM_L1, TileOpFormat::TILEOP_ND, viewShape);
    auto output = CreateTestLogicalTensor(MEM_L1, TileOpFormat::TILEOP_ND, viewShape);

    auto& viewOp = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_SLICE, {input}, {mid});
    viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{1, 2}));

    auto& copyOp = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_L1_COPY_IN_CONV, {mid}, {output});
    std::vector<OpImmediate> viewShapeImm = {OpImmediate::Specified(IRBuilder().CreateConstInt(viewShape[0])),
                                             OpImmediate::Specified(IRBuilder().CreateConstInt(viewShape[1]))};
    std::vector<OpImmediate> fromOffset = {OpImmediate::Specified(IRBuilder().CreateConstInt(3)),
                                           OpImmediate::Specified(IRBuilder().CreateConstInt(4))};
    copyOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(fromOffset, MEM_L1, viewShapeImm,
                                                            OpImmediate::Specified(mid->tensor->GetDynRawShape())));

    GenerateMoveOp pass;
    EXPECT_EQ(pass.ProcessL1CopyInConv(copyOp), SUCCESS);

    EXPECT_TRUE(viewOp.IsDeleted());
    EXPECT_EQ(copyOp.GetIOperands()[0], input);

    auto mergedAttr = std::dynamic_pointer_cast<CopyOpAttribute>(copyOp.GetOpAttribute());
    ASSERT_NE(mergedAttr, nullptr);
    auto rawShapeScalars = OpImmediate::ToSpecified(mergedAttr->GetRawShape());
    ASSERT_EQ(rawShapeScalars.size(), 2);
    EXPECT_EQ(rawShapeScalars[0].Dump(), "N");
    EXPECT_EQ(rawShapeScalars[1].Dump(), "C");
}

// ========== ProcessL0CCopyOutConv 测试 ==========

TEST_F(GenerateMoveOpPassTest, l0CCopyOutConvOffsetAccumulation)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "l0CCopyOutConvAcc", "l0CCopyOutConvAcc", nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("l0CCopyOutConvAcc", func);

    std::vector<int64_t> shape{16, 16};
    auto input = CreateTestLogicalTensor(MEM_L0C, TileOpFormat::TILEOP_ND, shape);
    auto mid = CreateTestLogicalTensor(MEM_L0C, TileOpFormat::TILEOP_ND, shape);
    auto output = CreateTestLogicalTensor(MEM_DEVICE_DDR, TileOpFormat::TILEOP_ND, shape);

    // L0C_COPY_OUT_CONV: toOffset=[1, 2]
    auto& copyOp = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_L0C_COPY_OUT_CONV, {input}, {mid});
    std::vector<OpImmediate> shapeImm = {OpImmediate::Specified(IRBuilder().CreateConstInt(shape[0])),
                                         OpImmediate::Specified(IRBuilder().CreateConstInt(shape[1]))};
    std::vector<OpImmediate> toOffset = {OpImmediate::Specified(IRBuilder().CreateConstInt(1)),
                                         OpImmediate::Specified(IRBuilder().CreateConstInt(2))};
    copyOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_L0C, toOffset, shapeImm, shapeImm));

    // CONTRACT: toOffset=[3, 4]
    auto& assembleOp = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_CONTRACT, {mid}, {output});
    assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{3, 4}));

    GenerateMoveOp pass;
    EXPECT_EQ(pass.ProcessL0CCopyOutConv(copyOp), SUCCESS);

    // 验证: ASSEMBLE 被删除，输出重连
    EXPECT_TRUE(assembleOp.IsDeleted());
    EXPECT_EQ(copyOp.GetOOperands()[0], output);

    // 验证: toOffset 累加 [1+3, 2+4] = [4, 6]
    auto mergedAttr = std::dynamic_pointer_cast<CopyOpAttribute>(copyOp.GetOpAttribute());
    ASSERT_NE(mergedAttr, nullptr);
    auto scalars = OpImmediate::ToSpecified(mergedAttr->GetToOffset());
    ASSERT_EQ(scalars.size(), 2);
    EXPECT_TRUE(scalars[0].ConcreteValid());
    EXPECT_EQ(scalars[0].Concrete(), 4);
    EXPECT_TRUE(scalars[1].ConcreteValid());
    EXPECT_EQ(scalars[1].Concrete(), 6);
}

TEST_F(GenerateMoveOpPassTest, l0CCopyOutConvSymbolicScalarOffsetAccumulation)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "l0CCopyOutConvSym", "l0CCopyOutConvSym", nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("l0CCopyOutConvSym", func);

    std::vector<int64_t> shape{16, 16};
    auto input = CreateTestLogicalTensor(MEM_L0C, TileOpFormat::TILEOP_ND, shape);
    auto mid = CreateTestLogicalTensor(MEM_L0C, TileOpFormat::TILEOP_ND, shape);
    auto output = CreateTestLogicalTensor(MEM_DEVICE_DDR, TileOpFormat::TILEOP_ND, shape);

    auto copyOffsetA = CreateTestScalarVar("a");
    auto copyOffsetB = CreateTestScalarVar("b");
    auto assembleDynC = CreateTestScalarVar("c");
    auto assembleDynD = CreateTestScalarVar("d");

    // L0C_COPY_OUT_CONV: toOffset=[Sym("a"), Sym("b")]
    auto& copyOp = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_L0C_COPY_OUT_CONV, {input}, {mid});
    std::vector<OpImmediate> shapeImm = {OpImmediate::Specified(IRBuilder().CreateConstInt(shape[0])),
                                         OpImmediate::Specified(IRBuilder().CreateConstInt(shape[1]))};
    std::vector<OpImmediate> toOffset = {OpImmediate::Specified(copyOffsetA), OpImmediate::Specified(copyOffsetB)};
    copyOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_L0C, toOffset, shapeImm, shapeImm));

    // CONTRACT: toOffset=[1, 2] + toDynOffset=[Sym("c"), Sym("d")]
    auto& assembleOp = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_CONTRACT, {mid}, {output});
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{1, 2},
                                                              std::vector<SymbolicScalar>{assembleDynC, assembleDynD});
    assembleOp.SetOpAttribute(assembleAttr);

    GenerateMoveOp pass;
    EXPECT_EQ(pass.ProcessL0CCopyOutConv(copyOp), SUCCESS);

    // 验证: ASSEMBLE 被删除，输出重连
    EXPECT_TRUE(assembleOp.IsDeleted());
    EXPECT_EQ(copyOp.GetOOperands()[0], output);

    // 验证: toOffset 已累加 (包含符号表达式，用字符串验证)
    auto mergedAttr = std::dynamic_pointer_cast<CopyOpAttribute>(copyOp.GetOpAttribute());
    ASSERT_NE(mergedAttr, nullptr);
    auto& mergedOffset = mergedAttr->GetToOffset();
    ASSERT_EQ(mergedOffset.size(), 2);
    auto scalars = OpImmediate::ToSpecified(mergedOffset);
    EXPECT_EQ(scalars[0].Dump(), "(" + assembleDynC.Dump() + "+" + copyOffsetA.Dump() + ")");
    EXPECT_EQ(scalars[1].Dump(), "(" + assembleDynD.Dump() + "+" + copyOffsetB.Dump() + ")");
}

TEST_F(GenerateMoveOpPassTest, l0CCopyOutConvDynRawShapePropagation)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "l0CCopyOutConvDynRaw", "l0CCopyOutConvDynRaw",
                                           nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("l0CCopyOutConvDynRaw", func);

    std::vector<int64_t> l0cShape{16, 16};
    std::vector<int64_t> outputShape{-1, -1};
    auto input = CreateTestLogicalTensor(MEM_L0C, TileOpFormat::TILEOP_ND, l0cShape);
    auto mid = CreateTestLogicalTensor(MEM_L0C, TileOpFormat::TILEOP_ND, l0cShape);
    auto output = CreateTestLogicalTensor(MEM_DEVICE_DDR, TileOpFormat::TILEOP_ND, outputShape);
    output->tensor->UpdateDynRawShape({CreateTestScalarVar("N"), CreateTestScalarVar("C")});

    auto& copyOp = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_L0C_COPY_OUT_CONV, {input}, {mid});
    std::vector<OpImmediate> l0cShapeImm = {OpImmediate::Specified(IRBuilder().CreateConstInt(l0cShape[0])),
                                            OpImmediate::Specified(IRBuilder().CreateConstInt(l0cShape[1]))};
    std::vector<OpImmediate> toOffset = {OpImmediate::Specified(IRBuilder().CreateConstInt(1)),
                                         OpImmediate::Specified(IRBuilder().CreateConstInt(2))};
    copyOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_L0C, toOffset, l0cShapeImm,
                                                            OpImmediate::Specified(mid->tensor->GetDynRawShape())));

    auto& assembleOp = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_CONTRACT, {mid}, {output});
    assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{3, 4}));

    GenerateMoveOp pass;
    EXPECT_EQ(pass.ProcessL0CCopyOutConv(copyOp), SUCCESS);

    EXPECT_TRUE(assembleOp.IsDeleted());
    EXPECT_EQ(copyOp.GetOOperands()[0], output);

    auto mergedAttr = std::dynamic_pointer_cast<CopyOpAttribute>(copyOp.GetOpAttribute());
    ASSERT_NE(mergedAttr, nullptr);
    auto rawShapeScalars = OpImmediate::ToSpecified(mergedAttr->GetRawShape());
    ASSERT_EQ(rawShapeScalars.size(), 2);
    EXPECT_EQ(rawShapeScalars[0].Dump(), "N");
    EXPECT_EQ(rawShapeScalars[1].Dump(), "C");
}

// ========== 测试用例5：ProcessUB2L1 GEMV 分支（跳过 ND2NZ 插入） ==========
TEST_F(GenerateMoveOpPassTest, ProcessUB2L1GemvSkipND2NZ)
{
    std::vector<int64_t> ndShape{2, 30, 15}; // inner=30, outer=15（非对齐）
    PROGRAM("ProcessUB2L1GemvSkipND2NZ")
    {
        Tensor a(DT_FP32, ndShape, "a");
        Tensor b(DT_FP32, ndShape, "b");

        TileShape::Current().SetVecTile(2, 30, 15, 1);

        PassManager& passManager = PassManager::Instance();
        passManager.RegisterStrategy("ProcessUB2L1GemvStrategy", {{"AssignMemoryType", PassName::ASSIGN_MEMORY_TYPE},
                                                                  {"GenerateMoveOp", PassName::GENERATE_MOVE_OP}});

        FUNCTION("ProcessUB2L1GemvFunc")
        {
            b = View(a, ndShape, {0, 0, 0}); // 构造VIEW OP作为基础
        }
    }

    // 导出/导入JSON（模拟真实流程）
    std::string jsonPath = "./config/pass/json/process_ub2l1_gemv.json";
    auto programJson = Program::GetInstance().DumpJson();
    DumpJsonFile(programJson, jsonPath);
    Json readData = LoadJsonFile(jsonPath);
    Program::GetInstance().LoadJson(readData);

    // 获取Function并构造OP_UB_COPY_L1操作（带 isGemv 属性）
    Function* func = Program::GetInstance().GetCurrentFunction();
    auto inputTensor = CreateTestLogicalTensor(MEM_UB, TileOpFormat::TILEOP_ND, ndShape);
    auto outputTensor = CreateTestLogicalTensor(MEM_L1, TileOpFormat::TILEOP_NZ, ndShape);
    auto ubCopyL1Op = CreateTestOperation(Opcode::OP_UB_COPY_L1, *func, {inputTensor}, {outputTensor});
    ubCopyL1Op->UpdateSubgraphID(0);
    ubCopyL1Op->SetAttr(OpAttributeKey::isGemv, static_cast<int64_t>(1));

    // 调用ProcessUB2L1
    GenerateMoveOp generateMoveOp;
    generateMoveOp.ProcessUB2L1(*func, *ubCopyL1Op);

    // 验证：GEMV 分支跳过 ND2NZ 插入（不产生 OP_UB_COPY_ND2NZ），且保留 isGemv 属性
    int ub2ubNum = 0;
    for (const auto& op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_UB_COPY_ND2NZ) {
            ub2ubNum++;
        }
    }
    EXPECT_EQ(ub2ubNum, 0) << "GEMV branch must not insert OP_UB_COPY_ND2NZ";
    int64_t isGemv = 0;
    bool hasIsGemv = ubCopyL1Op->GetAttr<int64_t>(OpAttributeKey::isGemv, isGemv);
    EXPECT_TRUE(hasIsGemv && isGemv != 0);
}

} // namespace tile_fwk
} // namespace npu
