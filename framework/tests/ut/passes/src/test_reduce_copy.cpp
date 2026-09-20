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
 * \file test_reduce_copy.cpp
 * \brief Unit test for ReduceCopy pass.
 */

#include <vector>
#include <string>
#include "gtest/gtest.h"
#include "tilefwk/data_type.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/platform.h"
#include "interface/function/function.h"
#define private public
#include "passes/tile_graph_pass/graph_partition/reduce_copy.h"
#undef private
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "computational_graph_builder.h"

namespace npu {
namespace tile_fwk {

class ReduceCopyTest : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetHostConfig(KEY_STRATEGY, "ReduceCopyTestStrategy");
        Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    }
    void TearDown() override { Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN); }
};

void BuildMatmulAddBranch(ComputationalGraphBuilder& G, int brId, std::vector<std::string>& incasts,
                          std::vector<std::string>& outcasts)
{
    std::vector<int64_t> tileShape{16, 16};
    std::string br = std::to_string(brId);
    std::vector<std::string> tensorNames{"tRA" + br, "tRB" + br, "tL1A" + br, "tL1B" + br,
                                         "tA" + br,  "tB" + br,  "tC" + br,   "tUB" + br};
    std::vector<Opcode> opCodes{Opcode::OP_VIEW,      Opcode::OP_VIEW,    Opcode::OP_L1_TO_L0A,
                                Opcode::OP_L1_TO_L0B, Opcode::OP_A_MUL_B, Opcode::OP_CONVERT};
    std::vector<std::vector<std::string>> ioperands{{"tRA" + br},  {"tRB" + br},           {"tL1A" + br},
                                                    {"tL1B" + br}, {"tA" + br, "tB" + br}, {"tC" + br}};
    std::vector<std::vector<std::string>> ooperands{{"tL1A" + br}, {"tL1B" + br}, {"tA" + br},
                                                    {"tB" + br},   {"tC" + br},   {"tUB" + br}};
    std::vector<std::string> opNames{"view" + br, "view2" + br, "toA" + br, "toB" + br, "matmul" + br, "convert" + br};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    for (auto aicOp : opNames) {
        G.GetOp(aicOp)->SetAttr(OpAttributeKey::isCube, true);
    }
    incasts.push_back("tRA" + br);
    incasts.push_back("tRB" + br);
    const int Num50 = 50;
    for (auto opName : opNames) {
        G.GetOp(opName)->UpdateSubgraphID(brId);
        G.GetOp(opName)->UpdateLatency(Num50);
    }
    const int Num2 = 2;
    for (int k = 0; k < Num2; k++) {
        std::string brv1 = std::to_string(brId + 1 + k);
        std::vector<std::string> tensorNamesV1{"add1" + brv1, "add2" + brv1, "out" + brv1};
        std::vector<Opcode> opCodesV1{Opcode::OP_ADDS, Opcode::OP_ADDS, Opcode::OP_ASSEMBLE};
        std::vector<std::vector<std::string>> ioperandsV1{
            {"tUB" + br},
            {"add1" + brv1},
            {"add2" + brv1},
        };
        std::vector<std::vector<std::string>> ooperandsV1{
            {"add1" + brv1},
            {"add2" + brv1},
            {"out" + brv1},
        };
        std::vector<std::string> opNamesV1{"add1" + brv1, "add2" + brv1, "assemble" + brv1};
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNamesV1), true);
        EXPECT_EQ(G.AddOps(opCodesV1, ioperandsV1, ooperandsV1, opNamesV1, true), true);
        outcasts.push_back("out" + brv1);
        for (auto opName : opNamesV1) {
            G.GetOp(opName)->UpdateSubgraphID(brId + 1 + k);
            G.GetOp(opName)->UpdateLatency(Num50);
            G.GetOp(opName)->SetAttr(OpAttributeKey::isCube, false);
        }
    }
}

void BuildMatmulAddsGraph(ComputationalGraphBuilder& G)
{
    std::vector<std::string> incasts;
    std::vector<std::string> outcasts;
    const int Num3 = 3;
    BuildMatmulAddBranch(G, 0, incasts, outcasts);
    BuildMatmulAddBranch(G, Num3, incasts, outcasts);
    Function* function = G.GetFunction();
    const int Num6 = 6;
    function->SetTotalSubGraphCount(Num6);
    EXPECT_EQ(G.SetInCast(incasts), true);
    EXPECT_EQ(G.SetOutCast(outcasts), true);
}

TEST_F(ReduceCopyTest, TestCase0)
{
    ComputationalGraphBuilder G;
    BuildMatmulAddsGraph(G);
    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);
    ReduceCopyMerge merger;
    function->paramConfigs_.autoMixPartition = 1;
    EXPECT_EQ(merger.RunOnFunction(*function), SUCCESS);
    const int Num2 = 2;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num2);
}

void BuildConnectMatmul(ComputationalGraphBuilder& G, int brId, std::vector<std::string>& incasts,
                        std::vector<std::string>& outcasts)
{
    std::vector<int64_t> tileShape{16, 16};
    const int Num50 = 50;
    std::string br = std::to_string(brId);
    std::vector<std::string> tensorNames{"tRA" + br, "tRB" + br, "tL1A" + br, "tL1B" + br,
                                         "tA" + br,  "tB" + br,  "tC" + br,   "tGM" + br};
    std::vector<Opcode> opCodes{Opcode::OP_VIEW,      Opcode::OP_VIEW,    Opcode::OP_L1_TO_L0A,
                                Opcode::OP_L1_TO_L0B, Opcode::OP_A_MUL_B, Opcode::OP_ASSEMBLE};
    std::vector<std::vector<std::string>> ioperands{{"tRA" + br},  {"tRB" + br},           {"tL1A" + br},
                                                    {"tL1B" + br}, {"tA" + br, "tB" + br}, {"tC" + br}};
    std::vector<std::vector<std::string>> ooperands{{"tL1A" + br}, {"tL1B" + br}, {"tA" + br},
                                                    {"tB" + br},   {"tC" + br},   {"tGM" + br}};
    std::vector<std::string> opNames{"view" + br, "view2" + br, "toA" + br, "toB" + br, "matmul" + br, "convert" + br};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    incasts.push_back("tRA" + br);
    incasts.push_back("tRB" + br);
    for (auto opName : opNames) {
        G.GetOp(opName)->UpdateSubgraphID(brId);
        G.GetOp(opName)->UpdateLatency(Num50);
        G.GetOp(opName)->SetAttr(OpAttributeKey::isCube, true);
    }
    std::string br2 = std::to_string(brId + 1);
    std::vector<std::string> tensorNames2{"tRB" + br2, "tL1A" + br2, "tL1B" + br2, "tA" + br2,
                                          "tB" + br2,  "tC" + br2,   "tGM" + br2};
    std::vector<Opcode> opCodes2{Opcode::OP_VIEW,      Opcode::OP_VIEW,    Opcode::OP_L1_TO_L0A,
                                 Opcode::OP_L1_TO_L0B, Opcode::OP_A_MUL_B, Opcode::OP_ASSEMBLE};
    std::vector<std::vector<std::string>> ioperands2{
        {"tGM" + br}, {"tRB" + br2}, {"tL1A" + br2}, {"tL1B" + br2}, {"tA" + br2, "tB" + br2}, {"tC" + br2}};
    std::vector<std::vector<std::string>> ooperands2{{"tL1A" + br2}, {"tL1B" + br2}, {"tA" + br2},
                                                     {"tB" + br2},   {"tC" + br2},   {"tGM" + br2}};
    std::vector<std::string> opNames2{"view" + br2, "view2" + br2,  "toA" + br2,
                                      "toB" + br2,  "matmul" + br2, "convert" + br2};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames2), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands2, ooperands2, opNames2, true), true);
    incasts.push_back("tRB" + br2);
    outcasts.push_back("tGM" + br2);
    for (auto opName : opNames2) {
        G.GetOp(opName)->UpdateSubgraphID(brId + 1);
        G.GetOp(opName)->UpdateLatency(Num50);
    }
}

void BuildConnectVector(ComputationalGraphBuilder& G, int brId, std::vector<std::string>& incasts,
                        std::vector<std::string>& outcasts)
{
    std::vector<int64_t> tileShape{16, 16};
    std::string br = std::to_string(brId);
    std::vector<std::string> tensorNames{"tin" + br, "tadds1" + br, "tout" + br};
    std::vector<Opcode> opCodes{Opcode::OP_ADDS, Opcode::OP_ADDS};
    std::vector<std::vector<std::string>> ioperands{{"tin" + br}, {"tadds1" + br}};
    std::vector<std::vector<std::string>> ooperands{{"tadds1" + br}, {"tout" + br}};
    std::vector<std::string> opNames{"adds1" + br, "adds2" + br};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    incasts.push_back("tin" + br);
    outcasts.push_back("tout" + br);
    const int Num50 = 50;
    G.GetOp("adds1" + br)->UpdateSubgraphID(brId);
    G.GetOp("adds1" + br)->UpdateLatency(Num50);
    G.GetOp("adds1" + br)->SetAttr(OpAttributeKey::isCube, false);
    G.GetOp("adds2" + br)->UpdateSubgraphID(brId + 1);
    G.GetOp("adds2" + br)->UpdateLatency(Num50);
    G.GetOp("adds2" + br)->SetAttr(OpAttributeKey::isCube, false);
}

void BuildConnect(ComputationalGraphBuilder& G)
{
    std::vector<std::string> incasts;
    std::vector<std::string> outcasts;
    const int Num2 = 2;
    BuildConnectMatmul(G, 0, incasts, outcasts);
    BuildConnectVector(G, Num2, incasts, outcasts);
    Function* function = G.GetFunction();
    const int Num4 = 4;
    function->SetTotalSubGraphCount(Num4);
    EXPECT_EQ(G.SetInCast(incasts), true);
    EXPECT_EQ(G.SetOutCast(outcasts), true);
}

TEST_F(ReduceCopyTest, TestCase1)
{
    ComputationalGraphBuilder G;
    BuildConnect(G);
    Function* function = G.GetFunction();
    function->paramConfigs_.autoMixPartition = 1;
    ReduceCopyMerge merger;
    merger.RunOnFunction(*function);
    const int Num3 = 3;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num3);
}

TEST_F(ReduceCopyTest, TestCase2)
{
    ComputationalGraphBuilder G;
    BuildConnect(G);
    Function* function = G.GetFunction();
    function->paramConfigs_.autoMixPartition = 1;
    G.GetOp("adds12")->scopeInfo_.cvFuseId = 0;
    G.GetOp("adds22")->scopeInfo_.cvFuseId = 0;
    ReduceCopyMerge merger;
    merger.RunOnFunction(*function);
    // 存在 CV 混合 scope(cvFuseId>=0)时禁用 auto-mix: 仅 {2,3} 经 enforce 路径合并, {0,1} 不再自动合并
    const int Num3 = 3;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num3);
}

TEST_F(ReduceCopyTest, PreserveOriginalSubgraphId)
{
    ComputationalGraphBuilder G;
    BuildMatmulAddsGraph(G);
    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);
    auto* add11 = G.GetOp("add11");
    ASSERT_NE(add11, nullptr);
    const int64_t add11SubgraphId = static_cast<int64_t>(add11->GetSubgraphID());
    auto* add14 = G.GetOp("add14");
    ASSERT_NE(add14, nullptr);
    const int64_t add14SubgraphId = static_cast<int64_t>(add14->GetSubgraphID());
    function->paramConfigs_.autoMixPartition = 1;
    ReduceCopyMerge merger;
    EXPECT_EQ(merger.RunOnFunction(*function), SUCCESS);

    int64_t preSubgraphId = -1;
    ASSERT_TRUE(add11->GetAttr(OpAttributeKey::reduceCopyPreSubgraphId, preSubgraphId));
    EXPECT_EQ(preSubgraphId, add11SubgraphId);
    ASSERT_TRUE(add14->GetAttr(OpAttributeKey::reduceCopyPreSubgraphId, preSubgraphId));
    EXPECT_EQ(preSubgraphId, add14SubgraphId);
}

TEST_F(ReduceCopyTest, TestCase3)
{
    ComputationalGraphBuilder G;
    BuildConnect(G);
    Function* function = G.GetFunction();
    function->paramConfigs_.autoMixPartition = 1;
    const int largeNum = 2e7; // latency超过阈值的子图不会合并
    G.GetOp("matmul0")->UpdateLatency(largeNum);
    ReduceCopyMerge merger;
    merger.RunOnFunction(*function);
    const int Num4 = 4;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num4);
}

// ============================================================================
// ST: automix sink 保护与输出级旁路 (简化自 glm / gqa / mha 调试用例)
static std::string AddCubeMatmulFrom(ComputationalGraphBuilder& G, int sg, const std::string& inA,
                                     const std::string& inB);
static std::string AddCubeMatmulSG(ComputationalGraphBuilder& G, int sg, std::vector<std::string>& incasts)
{
    std::vector<int64_t> sh{16, 16};
    std::string b = std::to_string(sg);
    std::string inA = "tRA" + b;
    std::string inB = "tRB" + b;
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, sh, {inA, inB}), true);
    incasts.push_back(inA);
    incasts.push_back(inB);
    return AddCubeMatmulFrom(G, sg, inA, inB);
}

static std::string AddIncast(ComputationalGraphBuilder& G, const std::string& name, std::vector<std::string>& incasts)
{
    std::vector<int64_t> sh{16, 16};
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, sh, name), true);
    incasts.push_back(name);
    return name;
}

static std::string AddCubeMatmulFrom(ComputationalGraphBuilder& G, int sg, const std::string& inA,
                                     const std::string& inB)
{
    std::vector<int64_t> sh{16, 16};
    std::string b = std::to_string(sg);
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, sh, {"tVA" + b, "tVB" + b, "tL0A" + b, "tL0B" + b, "tC" + b}), true);
    EXPECT_EQ(
        G.AddOps({Opcode::OP_VIEW, Opcode::OP_VIEW, Opcode::OP_L1_TO_L0A, Opcode::OP_L1_TO_L0B, Opcode::OP_A_MUL_B},
                 {{inA}, {inB}, {"tVA" + b}, {"tVB" + b}, {"tL0A" + b, "tL0B" + b}},
                 {{"tVA" + b}, {"tVB" + b}, {"tL0A" + b}, {"tL0B" + b}, {"tC" + b}},
                 {"viewA" + b, "viewB" + b, "toA" + b, "toB" + b, "mm" + b}, true),
        true);
    const int Num50 = 50;
    for (auto& n : std::vector<std::string>{"viewA" + b, "viewB" + b, "toA" + b, "toB" + b, "mm" + b}) {
        G.GetOp(n)->UpdateSubgraphID(sg);
        G.GetOp(n)->UpdateLatency(Num50);
        G.GetOp(n)->SetAttr(OpAttributeKey::isCube, true);
    }
    return "tC" + b;
}

static std::string AddVecSG(ComputationalGraphBuilder& G, int sg, const std::string& in, const std::string& out)
{
    std::vector<int64_t> sh{16, 16};
    std::string b = std::to_string(sg);
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, sh, {"vAdd" + b, out}), true);
    std::string addsName = "vadds" + b;
    std::string asmName = "vasm" + b;
    EXPECT_EQ(G.AddOps({Opcode::OP_ADDS, Opcode::OP_ASSEMBLE}, {{in}, {"vAdd" + b}}, {{"vAdd" + b}, {out}},
                       {addsName, asmName}, true),
              true);
    const int Num50 = 50;
    for (auto& n : std::vector<std::string>{addsName, asmName}) {
        G.GetOp(n)->UpdateSubgraphID(sg);
        G.GetOp(n)->UpdateLatency(Num50);
        G.GetOp(n)->SetAttr(OpAttributeKey::isCube, false);
    }
    return out;
}

static void AddVecSinkSG(ComputationalGraphBuilder& G, int sg, const std::vector<std::string>& inputs,
                         const std::string& outName, std::vector<std::string>& outcasts)
{
    std::vector<int64_t> sh{16, 16};
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, sh, outName), true);
    const int Num50 = 50;
    std::vector<std::string> addOuts;
    for (size_t i = 0; i < inputs.size(); i++) {
        std::string b = std::to_string(sg) + "_" + std::to_string(i);
        std::string tName = "sinkadd" + b;
        std::string opName = "vadds" + b;
        EXPECT_EQ(G.AddTensor(DataType::DT_FP32, sh, tName), true);
        EXPECT_EQ(G.AddOp(Opcode::OP_ADDS, {inputs[i]}, {tName}, opName, true), true);
        G.GetOp(opName)->UpdateSubgraphID(sg);
        G.GetOp(opName)->UpdateLatency(Num50);
        G.GetOp(opName)->SetAttr(OpAttributeKey::isCube, false);
        addOuts.push_back(tName);
    }
    std::string asmName = "vasm" + std::to_string(sg);
    EXPECT_EQ(G.AddOp(Opcode::OP_ASSEMBLE, addOuts, {outName}, asmName, true), true);
    G.GetOp(asmName)->UpdateSubgraphID(sg);
    G.GetOp(asmName)->UpdateLatency(Num50);
    G.GetOp(asmName)->SetAttr(OpAttributeKey::isCube, false);
    outcasts.push_back(outName);
}

// glm 场景: 两条等价 attention 分支 (C1->V1->C2) 汇入单个 sink, 串行损失 600 > 8×sink15
// -> sink 被保护, 两 C2 root 不归一, rootInDeg=2。
TEST_F(ReduceCopyTest, SinkProtectedWhenMultiProducerRoots)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> incasts;
    std::vector<std::string> outcasts;
    AddCubeMatmulFrom(G, 0, AddIncast(G, "QA0", incasts), AddIncast(G, "QB0", incasts));
    AddVecSG(G, 1, "tC0", "v1a");
    std::string c2a = AddCubeMatmulFrom(G, 2, "v1a", AddIncast(G, "VA0", incasts));
    AddCubeMatmulFrom(G, 3, AddIncast(G, "QA1", incasts), AddIncast(G, "QB1", incasts));
    AddVecSG(G, 4, "tC3", "v1b");
    std::string c2b = AddCubeMatmulFrom(G, 5, "v1b", AddIncast(G, "VA1", incasts));
    AddVecSinkSG(G, 6, {c2a, c2b}, "sinkGlmOut", outcasts);
    // 降低 sink 延迟使串行损失超阈值: sink 3-op×5=15, 分支各 600, serialLoss=600 > 8×15
    const int Num5 = 5;
    for (auto& n : std::vector<std::string>{"vadds6_0", "vadds6_1", "vasm6"}) {
        G.GetOp(n)->UpdateLatency(Num5);
    }
    Function* function = G.GetFunction();
    function->SetTotalSubGraphCount(7);
    ASSERT_EQ(G.SetInCast(incasts), true);
    ASSERT_EQ(G.SetOutCast(outcasts), true);
    function->paramConfigs_.autoMixPartition = 1;
    ReduceCopyMerge merger;
    EXPECT_EQ(merger.RunOnFunction(*function), SUCCESS);
    const int Num3 = 3;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num3);
}

// gqa 场景: 一个 cube fan-out 到两个 vec 再汇入 sink, vec 先并入 cube 归一 -> sink rootInDeg=1 放行。
TEST_F(ReduceCopyTest, SinkMergesWhenProducersCollapse)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> incasts;
    std::vector<std::string> outcasts;
    std::string c = AddCubeMatmulSG(G, 0, incasts);
    std::string v1 = AddVecSG(G, 1, c, "v1");
    std::string v2 = AddVecSG(G, 2, c, "v2");
    AddVecSinkSG(G, 3, {v1, v2}, "sinkGqaOut", outcasts);
    Function* function = G.GetFunction();
    function->SetTotalSubGraphCount(4);
    ASSERT_EQ(G.SetInCast(incasts), true);
    ASSERT_EQ(G.SetOutCast(outcasts), true);
    function->paramConfigs_.autoMixPartition = 1;
    ReduceCopyMerge merger;
    EXPECT_EQ(merger.RunOnFunction(*function), SUCCESS);
    const int Num1 = 1;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num1);
}

// mha 场景: 两个 cube 各 fan-out 到两个 sink (2:2 完全二分), 2 个出度0 sink -> 输出级旁路放行。
TEST_F(ReduceCopyTest, OutputStageBypassMergesAllSinks)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> incasts;
    std::vector<std::string> outcasts;
    std::string c0 = AddCubeMatmulSG(G, 0, incasts);
    std::string c1 = AddCubeMatmulSG(G, 1, incasts);
    AddVecSinkSG(G, 2, {c0, c1}, "sinkA", outcasts);
    AddVecSinkSG(G, 3, {c0, c1}, "sinkB", outcasts);
    Function* function = G.GetFunction();
    function->SetTotalSubGraphCount(4);
    ASSERT_EQ(G.SetInCast(incasts), true);
    ASSERT_EQ(G.SetOutCast(outcasts), true);
    function->paramConfigs_.autoMixPartition = 1;
    ReduceCopyMerge merger;
    EXPECT_EQ(merger.RunOnFunction(*function), SUCCESS);
    const int Num1 = 1;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num1);
}

// ============================================================================
// MixGraphMerger 缓存优化 UT

static MergeInput BuildSimpleMergeInput(int numSubgraph, const std::vector<std::set<int>>& outGraph,
                                        const std::vector<std::vector<int>>& groups)
{
    MergeInput input;
    input.numSubgraph = numSubgraph;
    input.maxLatency = 1e7;
    input.aivRatio = {1e-6, 1e6};
    input.subgraphAICLatency.assign(numSubgraph, 50);
    input.subgraphAIVLatency.assign(numSubgraph, 50);
    input.subgraphAICOpNum.assign(numSubgraph, 1);
    input.subgraphAIVOpNum.assign(numSubgraph, 1);
    input.maxSubgraphAICOpNum = 2000;
    input.maxSubgraphAIVOpNum = 2240;
    input.subGraphOutGraph = outGraph;
    input.mergeGroup = groups;
    input.isEnforceMergeGroup.assign(groups.size(), true);
    input.isValidMergeGroup.assign(groups.size(), true);
    return input;
}

// 收紧 AIC op 数阈值为 50: {0,1} AIC 总 60 超限拒绝, {2,3} AIC 总 10 限内通过
TEST_F(ReduceCopyTest, MixGraphMerger_AICOpNumLimitRejectsMerge)
{
    MergeInput input = BuildSimpleMergeInput(4, {{1}, {}, {3}, {}}, {{0, 1}, {2, 3}});
    input.subgraphAICOpNum = {30, 30, 5, 5};
    input.subgraphAIVOpNum = {30, 30, 5, 5};
    input.maxSubgraphAICOpNum = 50;
    MixGraphMerger merger;
    merger.mInput = input;
    merger.mParent = {0, 1, 2, 3};
    EXPECT_FALSE(merger.CheckLatencyConstraint({0, 1}));
    EXPECT_TRUE(merger.CheckLatencyConstraint({2, 3}));
}

// 收紧 AIV op 数阈值为 50: {0,1} AIV 总 60 超限拒绝, {2,3} AIV 总 10 限内通过
TEST_F(ReduceCopyTest, MixGraphMerger_AIVOpNumLimitRejectsMerge)
{
    MergeInput input = BuildSimpleMergeInput(4, {{1}, {}, {3}, {}}, {{0, 1}, {2, 3}});
    input.subgraphAICOpNum = {30, 30, 5, 5};
    input.subgraphAIVOpNum = {30, 30, 5, 5};
    input.maxSubgraphAIVOpNum = 50;
    MixGraphMerger merger;
    merger.mInput = input;
    merger.mParent = {0, 1, 2, 3};
    EXPECT_FALSE(merger.CheckLatencyConstraint({0, 1}));
    EXPECT_TRUE(merger.CheckLatencyConstraint({2, 3}));
}

// 上限边界: aicOps/aivOps 恰等于阈值时放行(仅超过才拒绝)
TEST_F(ReduceCopyTest, MixGraphMerger_OpNumLimitBoundaryEqualPasses)
{
    MergeInput input = BuildSimpleMergeInput(4, {{1}, {}, {3}, {}}, {{0, 1}, {2, 3}});
    input.subgraphAICOpNum = {30, 30, 5, 5};
    input.subgraphAIVOpNum = {30, 30, 5, 5};
    input.maxSubgraphAICOpNum = 60;
    input.maxSubgraphAIVOpNum = 60;
    MixGraphMerger merger;
    merger.mInput = input;
    merger.mParent = {0, 1, 2, 3};
    EXPECT_TRUE(merger.CheckLatencyConstraint({0, 1}));
    EXPECT_TRUE(merger.CheckLatencyConstraint({2, 3}));
}

// 两对独立子图 0->1, 2->3, 强制合并 {0,1} 和 {2,3}
// 验证 ApplyMergeToGraph 在 UnionSets 之后用 FindParent 获取正确 root, 增量更新缓存图
TEST_F(ReduceCopyTest, MixGraphMerger_ApplyMergeToGraphUpdatesCache)
{
    std::vector<std::set<int>> outGraph{{1}, {}, {3}, {}};
    MergeInput input = BuildSimpleMergeInput(4, outGraph, {{0, 1}, {2, 3}});
    MixGraphMerger merger;
    MergeOutput output = merger.Merge(input);
    EXPECT_EQ(output.numSubgraphUpdated, 2);
    EXPECT_EQ(output.subgraphIdUpdated[0], output.subgraphIdUpdated[1]);
    EXPECT_EQ(output.subgraphIdUpdated[2], output.subgraphIdUpdated[3]);
    EXPECT_NE(output.subgraphIdUpdated[0], output.subgraphIdUpdated[2]);
}

// 0->1->2, 0->3, 强制合并 {0,2}(因成环被拒) 然后 {0,1}(成功)
// 验证 CanMergeWithoutCycle 被拒后 save/restore 正确恢复缓存, 随后 {0,1} 合并成功
TEST_F(ReduceCopyTest, MixGraphMerger_CacheRestoreAfterRejection)
{
    std::vector<std::set<int>> outGraph{{1, 3}, {2}, {}, {}};
    MergeInput input = BuildSimpleMergeInput(4, outGraph, {{0, 2}, {0, 1}});
    MixGraphMerger merger;
    MergeOutput output = merger.Merge(input);
    const int Num2 = 2;
    EXPECT_EQ(output.numSubgraphUpdated, Num2);
    EXPECT_EQ(output.subgraphIdUpdated[0], output.subgraphIdUpdated[1]);
    EXPECT_EQ(output.subgraphIdUpdated[1], output.subgraphIdUpdated[2]);
    EXPECT_NE(output.subgraphIdUpdated[0], output.subgraphIdUpdated[3]);
}

// RunOnFunction 端到端: 非法自定义值(3~100)回退 default 档, 合图正常进行, 子图合并到 2
TEST_F(ReduceCopyTest, RunOnFunction_InvalidCustomFallsBackToDefaultLevel)
{
    ComputationalGraphBuilder G;
    BuildMatmulAddsGraph(G);
    Function* function = G.GetFunction();
    function->paramConfigs_.autoMixPartition = 50;
    ReduceCopyMerge merger;
    EXPECT_EQ(merger.RunOnFunction(*function), SUCCESS);
    const int Num2 = 2;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num2);
}

// RunOnFunction 端到端自定义档: N=101 时合并组 {cube,vec,vec} 共 12 op(AIC 6 + AIV 6) 远小于
// 总 op 上限, 合图正常进行, 子图合并到 2
TEST_F(ReduceCopyTest, RunOnFunction_CustomTotalOpNumMerges)
{
    ComputationalGraphBuilder G;
    BuildMatmulAddsGraph(G);
    Function* function = G.GetFunction();
    function->paramConfigs_.autoMixPartition = 101;
    ReduceCopyMerge merger;
    EXPECT_EQ(merger.RunOnFunction(*function), SUCCESS);
    const int Num2 = 2;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num2);
}

// 经历多轮「合并-拒绝-合并」后, 直接对比缓存图与 BuildMergedGraph 全量重建结果
// 守护「增量更新 ≡ 全量重建」这一核心不变量
TEST_F(ReduceCopyTest, MixGraphMerger_CacheConsistentWithFullRebuild)
{
    std::vector<std::set<int>> outGraph{{1, 3}, {2}, {}, {}};
    MergeInput input = BuildSimpleMergeInput(4, outGraph, {{0, 2}, {0, 1}});
    MixGraphMerger merger;
    merger.Merge(input);
    std::vector<std::set<int>> freshOut, freshIn;
    merger.BuildMergedGraph(freshOut, freshIn);
    for (int i = 0; i < input.numSubgraph; i++) {
        EXPECT_EQ(merger.mCachedOutGraph[i], freshOut[i]) << "outGraph mismatch at node " << i;
        EXPECT_EQ(merger.mCachedInGraph[i], freshIn[i]) << "inGraph mismatch at node " << i;
    }
}

// 强制合并也不能让 DDR tensor 同时具有组内读写和组外端点，否则该 tensor 会在合并后
// 变成仍被组外子图使用的内部 tensor。组外 producer/consumer 同时覆盖 WARN 诊断的分类路径。
TEST_F(ReduceCopyTest, MixGraphMerger_EnforcedMergeRejectsExternalDdrTensorUse)
{
    std::vector<std::set<int>> outGraph{{1}, {}, {}};
    MergeInput input = BuildSimpleMergeInput(3, outGraph, {{0, 1}});
    input.boundaryTensors = {{101, {0, 2}, {0, 2}, true, {}, {}}};
    input.subgraphToBoundaryTensorIds = {{0}, {}, {0}};

    MixGraphMerger merger;
    MergeOutput output = merger.Merge(input);

    const int Num3 = 3;
    EXPECT_EQ(output.numSubgraphUpdated, Num3);
    EXPECT_NE(output.subgraphIdUpdated[0], output.subgraphIdUpdated[1]);
    EXPECT_NE(output.subgraphIdUpdated[0], output.subgraphIdUpdated[2]);
}

// 复刻 issue #3160 原始形态: producer 全部组内, 仅 consumer 部分组外(组内读写 + 组外消费)。
// 防止误改判定条件(例如要求必须存在组外 producer 才拒绝)后守护失效。
TEST_F(ReduceCopyTest, MixGraphMerger_EnforcedMergeRejectsExternalConsumerOnly)
{
    std::vector<std::set<int>> outGraph{{1}, {}, {}};
    MergeInput input = BuildSimpleMergeInput(3, outGraph, {{0, 1}});
    input.boundaryTensors = {{103, {0}, {0, 2}, true, {}, {}}};
    input.subgraphToBoundaryTensorIds = {{0}, {}, {0}};

    MixGraphMerger merger;
    MergeOutput output = merger.Merge(input);

    const int Num3 = 3;
    EXPECT_EQ(output.numSubgraphUpdated, Num3);
    EXPECT_NE(output.subgraphIdUpdated[0], output.subgraphIdUpdated[1]);
    EXPECT_NE(output.subgraphIdUpdated[0], output.subgraphIdUpdated[2]);
}

// 没有组外端点时，强制合并仍应正常完成，防止新增的 external-use 检查误拦截。
TEST_F(ReduceCopyTest, MixGraphMerger_EnforcedMergeAllowsInternalDdrTensorUse)
{
    std::vector<std::set<int>> outGraph{{1}, {}};
    MergeInput input = BuildSimpleMergeInput(2, outGraph, {{0, 1}});
    input.boundaryTensors = {{102, {0}, {1}, true, {}, {}}};
    input.subgraphToBoundaryTensorIds = {{0}, {0}};

    MixGraphMerger merger;
    MergeOutput output = merger.Merge(input);

    const int Num1 = 1;
    EXPECT_EQ(output.numSubgraphUpdated, Num1);
    EXPECT_EQ(output.subgraphIdUpdated[0], output.subgraphIdUpdated[1]);
}

// 同 scope 的组外端点豁免: tensor 202 的组外 producer 2 与组内端点(producer 0/consumer 1)携带相同
// CvFuseId(5), scope 融合机制保证其终将同组 -> 不计为组外 -> 不构成拒绝 -> 合并放行。
// 复刻 scope 互锁场景: [48,49,52,53] 与 [49,50,52,53] 各被对方 tensor 卡住, 豁免后可先后合并。
TEST_F(ReduceCopyTest, MixGraphMerger_EnforcedMergeAllowsSameScopeExternalProducer)
{
    std::vector<std::set<int>> outGraph{{1}, {}, {}};
    MergeInput input = BuildSimpleMergeInput(3, outGraph, {{0, 1}});
    input.boundaryTensors = {{202, {0, 2}, {1}, true, {5, 5}, {5}}};
    input.subgraphToBoundaryTensorIds = {{0}, {0}, {0}};

    MixGraphMerger merger;
    MergeOutput output = merger.Merge(input);

    const int Num2 = 2;
    EXPECT_EQ(output.numSubgraphUpdated, Num2);
    EXPECT_EQ(output.subgraphIdUpdated[0], output.subgraphIdUpdated[1]);
    EXPECT_NE(output.subgraphIdUpdated[0], output.subgraphIdUpdated[2]);
}

// 组外 consumer 与组内端点属于同一 scope 时应豁免，覆盖 consumer 侧的同 scope 路径。
TEST_F(ReduceCopyTest, MixGraphMerger_EnforcedMergeAllowsSameScopeExternalConsumer)
{
    std::vector<std::set<int>> outGraph{{1}, {}, {}};
    MergeInput input = BuildSimpleMergeInput(3, outGraph, {{0, 1}});
    input.boundaryTensors = {{205, {0}, {1, 2}, true, {5}, {5, 5}}};
    input.subgraphToBoundaryTensorIds = {{0}, {0}, {0}};

    MixGraphMerger merger;
    MergeOutput output = merger.Merge(input);

    const int Num2 = 2;
    EXPECT_EQ(output.numSubgraphUpdated, Num2);
    EXPECT_EQ(output.subgraphIdUpdated[0], output.subgraphIdUpdated[1]);
    EXPECT_NE(output.subgraphIdUpdated[0], output.subgraphIdUpdated[2]);
}

// 组外 producer 虽与组内 scope 相同，但同时存在异 scope 的组外 consumer 时不得豁免，
// 防止豁免路径吞掉 foreign endpoint 的拒绝条件。
TEST_F(ReduceCopyTest, MixGraphMerger_EnforcedMergeRejectsMixedScopeExternalEndpoints)
{
    std::vector<std::set<int>> outGraph{{1}, {}, {}};
    MergeInput input = BuildSimpleMergeInput(3, outGraph, {{0, 1}});
    input.boundaryTensors = {{206, {0, 2}, {1, 2}, true, {5, 5}, {5, 7}}};
    input.subgraphToBoundaryTensorIds = {{0}, {0}, {0}};

    MixGraphMerger merger;
    MergeOutput output = merger.Merge(input);

    const int Num3 = 3;
    EXPECT_EQ(output.numSubgraphUpdated, Num3);
    EXPECT_NE(output.subgraphIdUpdated[0], output.subgraphIdUpdated[1]);
    EXPECT_NE(output.subgraphIdUpdated[0], output.subgraphIdUpdated[2]);
}

// 异 scope 的组外端点不豁免: 组内端点(producer 0/consumer 1)为 scope 5, 组外 producer 2 为 scope 7,
// 无融合保证 -> 仍计为组外 -> 组内读写 + 异 scope 组外端点 -> 拒绝。
TEST_F(ReduceCopyTest, MixGraphMerger_EnforcedMergeRejectsForeignScopeExternalProducer)
{
    std::vector<std::set<int>> outGraph{{1}, {}, {}};
    MergeInput input = BuildSimpleMergeInput(3, outGraph, {{0, 1}});
    input.boundaryTensors = {{201, {0, 2}, {1}, true, {5, 7}, {5}}};
    input.subgraphToBoundaryTensorIds = {{0}, {0}, {0}};

    MixGraphMerger merger;
    MergeOutput output = merger.Merge(input);

    const int Num3 = 3;
    EXPECT_EQ(output.numSubgraphUpdated, Num3);
    EXPECT_NE(output.subgraphIdUpdated[0], output.subgraphIdUpdated[1]);
    EXPECT_NE(output.subgraphIdUpdated[0], output.subgraphIdUpdated[2]);
}

// 无 scope(-1)的组外端点不豁免: 组内端点为 scope 5, 组外 producer 2 为 -1, 无融合保证 -> 拒绝。
TEST_F(ReduceCopyTest, MixGraphMerger_EnforcedMergeRejectsScopelessExternalProducer)
{
    std::vector<std::set<int>> outGraph{{1}, {}, {}};
    MergeInput input = BuildSimpleMergeInput(3, outGraph, {{0, 1}});
    input.boundaryTensors = {{203, {0, 2}, {1}, true, {5, -1}, {5}}};
    input.subgraphToBoundaryTensorIds = {{0}, {0}, {0}};

    MixGraphMerger merger;
    MergeOutput output = merger.Merge(input);

    const int Num3 = 3;
    EXPECT_EQ(output.numSubgraphUpdated, Num3);
    EXPECT_NE(output.subgraphIdUpdated[0], output.subgraphIdUpdated[1]);
    EXPECT_NE(output.subgraphIdUpdated[0], output.subgraphIdUpdated[2]);
}

// 组内端点含 -1(无 scope)同样不豁免: 组内 producer 0 为 scope 5、组内 consumer 1 为 -1, 组外 producer 2
// 为 scope 5。任一端点为 -1 即 scope 信息不完整 -> 豁免整体失效 -> 组外 producer 2 按 foreign 处理 -> 拒绝。
TEST_F(ReduceCopyTest, MixGraphMerger_EnforcedMergeRejectsScopelessInnerEndpoint)
{
    std::vector<std::set<int>> outGraph{{1}, {}, {}};
    MergeInput input = BuildSimpleMergeInput(3, outGraph, {{0, 1}});
    input.boundaryTensors = {{204, {0, 2}, {1}, true, {5, 5}, {-1}}};
    input.subgraphToBoundaryTensorIds = {{0}, {0}, {0}};

    MixGraphMerger merger;
    MergeOutput output = merger.Merge(input);

    const int Num3 = 3;
    EXPECT_EQ(output.numSubgraphUpdated, Num3);
    EXPECT_NE(output.subgraphIdUpdated[0], output.subgraphIdUpdated[1]);
    EXPECT_NE(output.subgraphIdUpdated[0], output.subgraphIdUpdated[2]);
}

// 正向: sg0 同时是 T 的 producer(ASSEMBLE) 和 consumer(CAST), sg1 是外部 ASSEMBLE producer.
// 复刻 gdr_fwd tensor 522 结构: producer 和 consumer 在同一子图, 外部 producer 在另一子图.
// T(UB) 多 ASSEMBLE producer(全 MOVE_LOCAL) -> WillBeDdrWithoutNewCopyOp 命中 -> isDDR=true ->
// 合并 sg0+sg2 时 T 满足三条件(producer in sg0 + consumer in sg0 + external sg1) -> 拒绝.
TEST_F(ReduceCopyTest, DdrPredictRejectsMultiAssembleProducerOnUbTensor)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> incasts;
    std::vector<std::string> outcasts;
    std::vector<int64_t> sh{16, 16};
    // sg0: cube matmul(AIC) -> ASSEMBLE 写 T(UB); CAST 读 T -> ASSEMBLE 写 out (AIV)
    std::string c0 = AddCubeMatmulSG(G, 0, incasts);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, sh, MemoryType::MEM_UB, "T"), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ASSEMBLE, {c0}, {"T"}, "asm0", true), true);
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, sh, {"tCast0", "out0"}), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_CAST, {"T"}, {"tCast0"}, "cast0", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ASSEMBLE, {"tCast0"}, {"out0"}, "asmOut0", true), true);
    // sg1: 外部 ASSEMBLE producer of T (多 producer, MOVE_LOCAL -> WillBeDdr 命中)
    AddIncast(G, "in1", incasts);
    EXPECT_EQ(G.AddOp(Opcode::OP_ASSEMBLE, {"in1"}, {"T"}, "asm1", true), true);
    // sg2: 独立 vec 子图消费 out0 (与 sg0 构成合并候选)
    AddVecSG(G, 2, "out0", "v2");
    outcasts.push_back("v2");
    const int Num50 = 50;
    G.GetOp("asm0")->UpdateSubgraphID(0);
    G.GetOp("asm0")->UpdateLatency(Num50);
    G.GetOp("asm0")->SetAttr(OpAttributeKey::isCube, true);
    for (auto& n : std::vector<std::string>{"cast0", "asmOut0"}) {
        G.GetOp(n)->UpdateSubgraphID(0);
        G.GetOp(n)->UpdateLatency(Num50);
        G.GetOp(n)->SetAttr(OpAttributeKey::isCube, false);
    }
    const int largeNum = 2e7;
    G.GetOp("asm1")->UpdateSubgraphID(1);
    G.GetOp("asm1")->UpdateLatency(largeNum);
    G.GetOp("asm1")->SetAttr(OpAttributeKey::isCube, false);
    Function* function = G.GetFunction();
    function->SetTotalSubGraphCount(3);
    ASSERT_EQ(G.SetInCast(incasts), true);
    ASSERT_EQ(G.SetOutCast(outcasts), true);
    function->paramConfigs_.autoMixPartition = 1;
    ReduceCopyMerge merger;
    EXPECT_EQ(merger.RunOnFunction(*function), SUCCESS);
    // T(UB) 多 ASSEMBLE producer -> WillBeDdr 命中 -> isDDR=true ->
    // inner-external-use 拒绝 sg0+sg2 合并(sg1 为外部 producer 端点) -> 子图数 == 3.
    const int Num3 = 3;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num3);
}

// 反向: 同上结构, 但外部 producer 换成 ADD(calcType=BROADCAST, 非 MOVE) ->
// WillBeDdrWithoutNewCopyOp 返回 false -> isDDR=false -> 跳过 inner-external-use -> 合并放行.
TEST_F(ReduceCopyTest, DdrPredictNotTriggeredWhenProducerMixNonMoveOp)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> incasts;
    std::vector<std::string> outcasts;
    std::vector<int64_t> sh{16, 16};
    std::string c0 = AddCubeMatmulSG(G, 0, incasts);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, sh, MemoryType::MEM_UB, "T"), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ASSEMBLE, {c0}, {"T"}, "asm0", true), true);
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, sh, {"tCast0", "out0"}), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_CAST, {"T"}, {"tCast0"}, "cast0", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ASSEMBLE, {"tCast0"}, {"out0"}, "asmOut0", true), true);
    // sg1: 外部 ADD producer of T (BROADCAST -> WillBeDdr 返回 false)
    AddIncast(G, "in1", incasts);
    EXPECT_EQ(G.AddOp(Opcode::OP_ADD, {"in1", "in1"}, {"T"}, "add1", true), true);
    AddVecSG(G, 2, "out0", "v2");
    outcasts.push_back("v2");
    const int Num50 = 50;
    G.GetOp("asm0")->UpdateSubgraphID(0);
    G.GetOp("asm0")->UpdateLatency(Num50);
    G.GetOp("asm0")->SetAttr(OpAttributeKey::isCube, true);
    for (auto& n : std::vector<std::string>{"cast0", "asmOut0"}) {
        G.GetOp(n)->UpdateSubgraphID(0);
        G.GetOp(n)->UpdateLatency(Num50);
        G.GetOp(n)->SetAttr(OpAttributeKey::isCube, false);
    }
    const int largeNum = 2e7;
    G.GetOp("add1")->UpdateSubgraphID(1);
    G.GetOp("add1")->UpdateLatency(largeNum);
    G.GetOp("add1")->SetAttr(OpAttributeKey::isCube, false);
    Function* function = G.GetFunction();
    function->SetTotalSubGraphCount(3);
    ASSERT_EQ(G.SetInCast(incasts), true);
    ASSERT_EQ(G.SetOutCast(outcasts), true);
    function->paramConfigs_.autoMixPartition = 1;
    ReduceCopyMerge merger;
    EXPECT_EQ(merger.RunOnFunction(*function), SUCCESS);
    // T(UB) 多 producer 混入 ADD(BROADCAST) -> WillBeDdr 返回 false -> isDDR=false ->
    // 跳过 inner-external-use -> sg0+sg2 合并放行(sg1 高 latency 不参与) -> 子图数 == 2.
    const int Num2 = 2;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num2);
}

// 单 producer 跨核 COPY_OUT(L1->UB): sg0 cube matmul 产 L1 tensor, COPY_OUT(L1->UB) 写 T(UB).
TEST_F(ReduceCopyTest, DdrPredictRejectsCrossCoreCopyOutProducerOnUbTensor)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> incasts;
    std::vector<std::string> outcasts;
    std::vector<int64_t> sh{16, 16};
    // sg0: cube matmul(AIC) -> L1 tensor -> 跨核 COPY_OUT(L1->UB) 写 T(UB), 单 producer
    std::string c0 = AddCubeMatmulSG(G, 0, incasts);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, sh, MemoryType::MEM_L1, "tL1"), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_COPY_OUT, {c0}, {"tL1"}, "cpOut0", true), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, sh, MemoryType::MEM_UB, "T"), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_COPY_OUT, {"tL1"}, {"T"}, "crossCoreCopy", true), true);
    // sg0: CAST 读 T -> ASSEMBLE 写 out (AIV, 与 cube 构成混合子图)
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, sh, {"tCast0", "out0"}), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_CAST, {"T"}, {"tCast0"}, "cast0", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ASSEMBLE, {"tCast0"}, {"out0"}, "asmOut0", true), true);
    // sg1: 外部 CAST consumer of T (外部端点, 高 latency 阻止合并)
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, sh, {"tCast1", "out1"}), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_CAST, {"T"}, {"tCast1"}, "cast1", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ASSEMBLE, {"tCast1"}, {"out1"}, "asmOut1", true), true);
    // sg2: 独立 vec 子图消费 out0 (与 sg0 构成合并候选)
    AddVecSG(G, 2, "out0", "v2");
    outcasts.push_back("v2");
    const int Num50 = 50;
    for (auto& n : std::vector<std::string>{"cpOut0", "crossCoreCopy"}) {
        G.GetOp(n)->UpdateSubgraphID(0);
        G.GetOp(n)->UpdateLatency(Num50);
        G.GetOp(n)->SetAttr(OpAttributeKey::isCube, true);
    }
    for (auto& n : std::vector<std::string>{"cast0", "asmOut0"}) {
        G.GetOp(n)->UpdateSubgraphID(0);
        G.GetOp(n)->UpdateLatency(Num50);
        G.GetOp(n)->SetAttr(OpAttributeKey::isCube, false);
    }
    const int largeNum = 2e7;
    for (auto& n : std::vector<std::string>{"cast1", "asmOut1"}) {
        G.GetOp(n)->UpdateSubgraphID(1);
        G.GetOp(n)->UpdateLatency(largeNum);
        G.GetOp(n)->SetAttr(OpAttributeKey::isCube, false);
    }
    Function* function = G.GetFunction();
    function->SetTotalSubGraphCount(3);
    ASSERT_EQ(G.SetInCast(incasts), true);
    ASSERT_EQ(G.SetOutCast(outcasts), true);
    function->paramConfigs_.autoMixPartition = 1;
    ReduceCopyMerge merger;
    EXPECT_EQ(merger.RunOnFunction(*function), SUCCESS);
    // T(UB) 单 producer 跨核 COPY_OUT(L1->UB) -> isDDR=true -> 拒绝 sg0+sg2 合并 -> 子图数 == 3.
    const int Num3 = 3;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num3);
}

// 单 producer OP_ASSEMBLE 写 UB tensor T: WillBeDdrWithoutNewCopyOp 单 producer 分支命中 -> isDDR=true -> 拒绝合并.
TEST_F(ReduceCopyTest, DdrPredictRejectsSingleAssembleProducerOnUbTensor)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> incasts;
    std::vector<std::string> outcasts;
    std::vector<int64_t> sh{16, 16};
    // sg0: cube matmul(AIC) -> ASSEMBLE 写 T(UB), 单 producer
    std::string c0 = AddCubeMatmulSG(G, 0, incasts);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, sh, MemoryType::MEM_UB, "T"), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ASSEMBLE, {c0}, {"T"}, "asm0", true), true);
    // sg0: CAST 读 T -> ASSEMBLE 写 out (AIV, 与 cube 构成混合子图)
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, sh, {"tCast0", "out0"}), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_CAST, {"T"}, {"tCast0"}, "cast0", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ASSEMBLE, {"tCast0"}, {"out0"}, "asmOut0", true), true);
    // sg1: 外部 CAST consumer of T (外部端点, 高 latency 阻止合并)
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, sh, {"tCast1", "out1"}), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_CAST, {"T"}, {"tCast1"}, "cast1", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ASSEMBLE, {"tCast1"}, {"out1"}, "asmOut1", true), true);
    // sg2: 独立 vec 子图消费 out0 (与 sg0 构成合并候选)
    AddVecSG(G, 2, "out0", "v2");
    outcasts.push_back("v2");
    const int Num50 = 50;
    G.GetOp("asm0")->UpdateSubgraphID(0);
    G.GetOp("asm0")->UpdateLatency(Num50);
    G.GetOp("asm0")->SetAttr(OpAttributeKey::isCube, true);
    for (auto& n : std::vector<std::string>{"cast0", "asmOut0"}) {
        G.GetOp(n)->UpdateSubgraphID(0);
        G.GetOp(n)->UpdateLatency(Num50);
        G.GetOp(n)->SetAttr(OpAttributeKey::isCube, false);
    }
    const int largeNum = 2e7;
    for (auto& n : std::vector<std::string>{"cast1", "asmOut1"}) {
        G.GetOp(n)->UpdateSubgraphID(1);
        G.GetOp(n)->UpdateLatency(largeNum);
        G.GetOp(n)->SetAttr(OpAttributeKey::isCube, false);
    }
    Function* function = G.GetFunction();
    function->SetTotalSubGraphCount(3);
    ASSERT_EQ(G.SetInCast(incasts), true);
    ASSERT_EQ(G.SetOutCast(outcasts), true);
    function->paramConfigs_.autoMixPartition = 1;
    ReduceCopyMerge merger;
    EXPECT_EQ(merger.RunOnFunction(*function), SUCCESS);
    // T(UB) 单 ASSEMBLE producer -> WillBeDdr 命中 -> isDDR=true -> 拒绝 sg0+sg2 合并 -> 子图数 == 3.
    const int Num3 = 3;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num3);
}

// 存在 CV 混合 scope(cvFuseId>=0)时禁用 auto-mix: {0,1} 为该 scope 涉及的子图组(enforce 路径),
// {2,3} 为满足全部约束的 auto-mix 候选组。对照组(无 scope 标记)中 {2,3} 可被 auto-mix 合并;
// 存在 scope 标记时仅 {0,1} 经 enforce 路径合并, {2,3} 保持独立, 防止 auto-mix 改变 scope 的切分结果。
TEST_F(ReduceCopyTest, MixGraphMerger_AutoMixSkippedWhenScopedOpExists)
{
    std::vector<std::set<int>> outGraph{{1}, {}, {3}, {}};
    MergeInput input = BuildSimpleMergeInput(4, outGraph, {{0, 1}, {2, 3}});
    input.isEnforceMergeGroup = {true, false};
    input.hasScopedOp = true;
    input.boundaryTensors = {{300, {0}, {1}, false, {}, {}}, {301, {2}, {3}, false, {}, {}}};
    input.subgraphToBoundaryTensorIds = {{0}, {0}, {1}, {1}};

    // 对照: 无 scope 标记时 auto-mix 同时合并 {0,1} 和 {2,3}
    MergeInput noScopeInput = input;
    noScopeInput.hasScopedOp = false;
    noScopeInput.isEnforceMergeGroup = {false, false};
    MixGraphMerger noScopeMerger;
    MergeOutput noScopeOutput = noScopeMerger.Merge(noScopeInput);
    const int Num2 = 2;
    EXPECT_EQ(noScopeOutput.numSubgraphUpdated, Num2);

    // 存在 scope 标记时 auto-mix 被禁用, 仅 CV 混合 scope 涉及的 {0,1} 经 enforce 路径合并
    MixGraphMerger merger;
    MergeOutput output = merger.Merge(input);
    const int Num3 = 3;
    EXPECT_EQ(output.numSubgraphUpdated, Num3);
    EXPECT_EQ(output.subgraphIdUpdated[0], output.subgraphIdUpdated[1]);
    EXPECT_NE(output.subgraphIdUpdated[2], output.subgraphIdUpdated[3]);
    EXPECT_NE(output.subgraphIdUpdated[0], output.subgraphIdUpdated[2]);
}

// scope 标记存在但未形成任何 enforce 子图组(如 boundary tensor 带组外端点, IsEnforceMergeBoundary 为
// false)时, auto-mix 同样被禁用: 门控以 cvFuseId>=0 为准, 不依赖 enforce 子图组的过滤结果, 候选全部跳过。
TEST_F(ReduceCopyTest, MixGraphMerger_AutoMixSkippedWithoutEnforceGroup)
{
    std::vector<std::set<int>> outGraph{{1}, {}, {3}, {}};
    MergeInput input = BuildSimpleMergeInput(4, outGraph, {{0, 1}, {2, 3}});
    input.isEnforceMergeGroup = {false, false};
    input.hasScopedOp = true;
    input.boundaryTensors = {{300, {0}, {1}, false, {}, {}}, {301, {2}, {3}, false, {}, {}}};
    input.subgraphToBoundaryTensorIds = {{0}, {0}, {1}, {1}};

    MixGraphMerger merger;
    MergeOutput output = merger.Merge(input);

    const int Num4 = 4;
    EXPECT_EQ(output.numSubgraphUpdated, Num4);
    EXPECT_NE(output.subgraphIdUpdated[0], output.subgraphIdUpdated[1]);
    EXPECT_NE(output.subgraphIdUpdated[2], output.subgraphIdUpdated[3]);
}

// enforce 合并超 op 数上限时不拦截, 只打 WARN; 子图照常合并为 1
TEST_F(ReduceCopyTest, MixGraphMerger_EnforceOpNumExceedsWarnsButMerges)
{
    std::vector<std::set<int>> outGraph{{1}, {}};
    MergeInput input = BuildSimpleMergeInput(2, outGraph, {{0, 1}});
    input.subgraphAICOpNum = {1500, 600};
    input.subgraphAIVOpNum = {800, 500};
    input.maxSubgraphAICOpNum = 2000;
    input.maxSubgraphAIVOpNum = 1000;

    MixGraphMerger merger;
    merger.mInput = input;
    merger.mParent = {0, 1};
    merger.WarnIfEnforceOpNumExceeds({0, 1});

    MergeOutput output = merger.Merge(input);
    const int Num1 = 1;
    EXPECT_EQ(output.numSubgraphUpdated, Num1);
    EXPECT_EQ(output.subgraphIdUpdated[0], output.subgraphIdUpdated[1]);
}

// ============================================================================
// 本次修改 UT: 250k 时延护栏 + slot-scope carry 识别 + 同通路门控
// 对应 reduce_copy.cpp 的 kAutoMixMaxMergeLatency / MarkFeedbackSubgraphs 系列 /
// CheckLoopPathConsistency / MergeInput::subgraphLoopPaths
// ============================================================================

namespace {
constexpr int kExpLatencyCap = 250000; // 与 reduce_copy.cpp 的 kAutoMixMaxMergeLatency 保持一致
} // namespace

// 时延护栏边界: 组总延迟(AIC+AIV) 260020 > 250k 拒绝, 120020 限内放行
TEST_F(ReduceCopyTest, MixGraphMerger_LatencyCap250kBoundaryCheck)
{
    std::vector<std::set<int>> outGraph{{1}, {}, {3}, {}};
    MergeInput input = BuildSimpleMergeInput(4, outGraph, {{0, 1}, {2, 3}});
    input.maxLatency = kExpLatencyCap;
    input.subgraphAICLatency = {130000, 130000, 60000, 60000};
    input.subgraphAIVLatency = {10, 10, 10, 10};
    MixGraphMerger merger;
    merger.mInput = input;
    merger.mParent = {0, 1, 2, 3};
    EXPECT_FALSE(merger.CheckLatencyConstraint({0, 1}));
    EXPECT_TRUE(merger.CheckLatencyConstraint({2, 3}));
}

// 自动合并路径: 链 {0,1,2} 总延迟 360030 > 250k, 合并被拒, 子图数保持 3
// (结构检查已放行: 边界张量构成 1:1 链, 拒绝只来自时延护栏)
TEST_F(ReduceCopyTest, Merge_LatencyCap250kBlocksOversizeChain)
{
    std::vector<std::set<int>> outGraph{{1}, {2}, {}};
    MergeInput input = BuildSimpleMergeInput(3, outGraph, {{0, 1, 2}});
    input.isEnforceMergeGroup = {false};
    input.maxLatency = kExpLatencyCap;
    input.subgraphAICLatency = {120000, 120000, 120000};
    input.subgraphAIVLatency = {10, 10, 10};
    input.boundaryTensors = {{100, {0}, {1}, true, {}, {}}, {101, {1}, {2}, true, {}, {}}};
    input.subgraphToBoundaryTensorIds = {{0}, {0, 1}, {1}};
    MixGraphMerger merger;
    MergeOutput output = merger.Merge(input);
    EXPECT_EQ(output.numSubgraphUpdated, 3);
}

// 同图总延迟 180030 <= 250k: 合并放行, 收敛到 1 子图 (gqa 全融合组 213934 cycles 的场景抽象)
TEST_F(ReduceCopyTest, Merge_LatencyCap250kAllowsChainUnderLimit)
{
    std::vector<std::set<int>> outGraph{{1}, {2}, {}};
    MergeInput input = BuildSimpleMergeInput(3, outGraph, {{0, 1, 2}});
    input.isEnforceMergeGroup = {false};
    input.maxLatency = kExpLatencyCap;
    input.subgraphAICLatency = {60000, 60000, 60000};
    input.subgraphAIVLatency = {10, 10, 10};
    input.boundaryTensors = {{100, {0}, {1}, true, {}, {}}, {101, {1}, {2}, true, {}, {}}};
    input.subgraphToBoundaryTensorIds = {{0}, {0, 1}, {1}};
    MixGraphMerger merger;
    MergeOutput output = merger.Merge(input);
    EXPECT_EQ(output.numSubgraphUpdated, 1);
}

// 同通路门控: carry 环上成员(sg0/sg1, path={7})与环外成员(sg2/sg3, path={})混合的候选组拒绝;
// 同环链与全环外组合放行
TEST_F(ReduceCopyTest, MixGraphMerger_LoopPathConsistency_MixedMembershipRejected)
{
    MergeInput input = BuildSimpleMergeInput(4, {{1}, {2}, {3}, {}}, {});
    input.subgraphLoopPaths = {{7}, {7}, {}, {}}; // sg0/sg1 在 carry 环 7 上, sg2/sg3 环外
    MixGraphMerger merger;
    merger.mInput = input;
    merger.mRootLoopPaths = input.subgraphLoopPaths;
    merger.mParent = {0, 1, 2, 3};
    merger.mRootToBoundaryTensorIds.assign(input.numSubgraph, {});
    EXPECT_FALSE(merger.CheckLoopPathConsistency({0, 1, 2}));
    EXPECT_FALSE(merger.CheckLoopPathConsistency({0, 2}));
    EXPECT_TRUE(merger.CheckLoopPathConsistency({0, 1}));
    EXPECT_TRUE(merger.CheckLoopPathConsistency({2, 3}));
}

// 无 slot 信息(subgraphLoopPaths 为空, 如 mha_grad / 无 feedback slot 的 kernel)时门控不启用
TEST_F(ReduceCopyTest, MixGraphMerger_LoopPathConsistencyDisabledWithoutSlotInfo)
{
    MergeInput input = BuildSimpleMergeInput(3, {{1}, {2}, {}}, {});
    input.subgraphLoopPaths = {};
    MixGraphMerger merger;
    merger.mInput = input;
    merger.mRootLoopPaths = input.subgraphLoopPaths;
    merger.mParent = {0, 1, 2};
    EXPECT_TRUE(merger.CheckLoopPathConsistency({0, 1, 2}));
}

// 数据耦合豁免: 环外成员(空集)挂在环上成员的直接边界 tensor 数据边上放行;
// 环上成员与无数据边的环外成员混合仍拒绝
TEST_F(ReduceCopyTest, MixGraphMerger_LoopPathConsistencyDataEdgeExempted)
{
    MergeInput input = BuildSimpleMergeInput(4, {{1}, {2}, {}, {}}, {});
    input.subgraphLoopPaths = {{7}, {7}, {}, {}}; // sg0/sg1 在 carry 环 7 上, sg2/sg3 环外
    // sg0 -> sg2 直接数据边 (boundary tensor 100: producer sg0, consumer sg2)
    input.boundaryTensors = {{100, {0}, {2}, true, {}, {}}};
    input.subgraphToBoundaryTensorIds = {{0}, {}, {0}, {}};
    MixGraphMerger merger;
    merger.mInput = input;
    merger.mRootLoopPaths = input.subgraphLoopPaths;
    merger.mParent = {0, 1, 2, 3};
    merger.mRootToBoundaryTensorIds = input.subgraphToBoundaryTensorIds;
    EXPECT_TRUE(merger.CheckLoopPathConsistency({0, 2}));  // 环外成员挂在环上成员数据边, 豁免放行
    EXPECT_FALSE(merger.CheckLoopPathConsistency({1, 2})); // sg1 与 sg2 既不共环也无数据边, 仍拒绝
    EXPECT_TRUE(merger.CheckLoopPathConsistency({0, 1}));  // 同环, 放行不变
}

// 共环豁免: loop 集合不同但有交集(共享子图的多 slot 链场景)的成员放行;
// 与完全不相交的空集成员混合仍拒绝
TEST_F(ReduceCopyTest, MixGraphMerger_LoopPathConsistencySharedLoopExempted)
{
    MergeInput input = BuildSimpleMergeInput(3, {{1}, {2}, {}}, {});
    // sg0 在环 {7}, sg1 在环 {7,8}(多条 slot 链共享子图): 集合不同但共环 7
    input.subgraphLoopPaths = {{7}, {7, 8}, {}};
    MixGraphMerger merger;
    merger.mInput = input;
    merger.mRootLoopPaths = input.subgraphLoopPaths;
    merger.mParent = {0, 1, 2};
    merger.mRootToBoundaryTensorIds.assign(input.numSubgraph, {});
    EXPECT_TRUE(merger.CheckLoopPathConsistency({0, 1}));  // 共环豁免放行
    EXPECT_FALSE(merger.CheckLoopPathConsistency({0, 2})); // sg2 空集, 不共环无数据边, 拒绝
}

// 跨环拒绝: 非空不相交(两个独立 loop)的成员即使存在直接数据边也拒绝,
// 防止两个独立 loop body 被融合进同一 slot-scope 子图(两环 trip count 可不同)
TEST_F(ReduceCopyTest, MixGraphMerger_LoopPathConsistencyCrossRingDataEdgeRejected)
{
    MergeInput input = BuildSimpleMergeInput(2, {{1}, {}}, {});
    input.subgraphLoopPaths = {{7}, {8}};                    // sg0 环7, sg1 环8, 互不相交
    input.boundaryTensors = {{100, {0}, {1}, true, {}, {}}}; // sg0 -> sg1 直接数据边
    input.subgraphToBoundaryTensorIds = {{0}, {0}};
    MixGraphMerger merger;
    merger.mInput = input;
    merger.mRootLoopPaths = input.subgraphLoopPaths;
    merger.mParent = {0, 1};
    merger.mRootToBoundaryTensorIds = input.subgraphToBoundaryTensorIds;
    EXPECT_FALSE(merger.CheckLoopPathConsistency({0, 1}));
}

// hinge guard(扇出): 扇出源 sg0 喂 3 个互不可达的组内分支(>= kHingeBranchThreshold),
// 串行化独立并行分支拒绝; 组内含简单 tensor edge(0->1/0->2/0->3), 无 hinge guard 时本组合并归 1
TEST_F(ReduceCopyTest, MixGraphMerger_HingeGuardRejectsParallelFanout)
{
    std::vector<std::set<int>> outGraph{{1, 2, 3}, {}, {}, {}};
    MergeInput input = BuildSimpleMergeInput(4, outGraph, {{0, 1, 2, 3}});
    input.isEnforceMergeGroup = {false};
    input.boundaryTensors = {
        {100, {0}, {1}, true, {}, {}}, {101, {0}, {2}, true, {}, {}}, {102, {0}, {3}, true, {}, {}}};
    input.subgraphToBoundaryTensorIds = {{0, 1, 2}, {0}, {1}, {2}};
    MixGraphMerger merger;
    MergeOutput output = merger.Merge(input);
    EXPECT_EQ(output.numSubgraphUpdated, 4); // 分支互不可达, 保持独立
}

// hinge guard(链式): 扇出源 sg0 -> sg1 -> sg2 -> sg3 链式依赖(组内可达),
// 不属串行化独立并行分支, 放行归 1
TEST_F(ReduceCopyTest, MixGraphMerger_HingeGuardAllowsChainedBranches)
{
    std::vector<std::set<int>> outGraph{{1}, {2}, {3}, {}};
    MergeInput input = BuildSimpleMergeInput(4, outGraph, {{0, 1, 2, 3}});
    input.isEnforceMergeGroup = {false};
    input.boundaryTensors = {
        {100, {0}, {1}, true, {}, {}}, {101, {1}, {2}, true, {}, {}}, {102, {2}, {3}, true, {}, {}}};
    input.subgraphToBoundaryTensorIds = {{0}, {0, 1}, {1, 2}, {2}};
    MixGraphMerger merger;
    MergeOutput output = merger.Merge(input);
    EXPECT_EQ(output.numSubgraphUpdated, 1);
}

// hinge guard(同环豁免): 同环({7})三链扇出(online softmax 共享读端的多写链抽象),
// 分支被环迭代耦合, 非独立并行分支不拦; 无豁免时 3 分支互不可达会被误拒
TEST_F(ReduceCopyTest, MixGraphMerger_HingeGuardExemptsSameRingBranches)
{
    std::vector<std::set<int>> outGraph{{1, 2, 3}, {}, {}, {}};
    MergeInput input = BuildSimpleMergeInput(4, outGraph, {{0, 1, 2, 3}});
    input.isEnforceMergeGroup = {false};
    input.subgraphLoopPaths = {{7}, {7}, {7}, {7}};
    input.boundaryTensors = {
        {100, {0}, {1}, true, {}, {}}, {101, {0}, {2}, true, {}, {}}, {102, {0}, {3}, true, {}, {}}};
    input.subgraphToBoundaryTensorIds = {{0, 1, 2}, {0}, {1}, {2}};
    MixGraphMerger merger;
    MergeOutput output = merger.Merge(input);
    EXPECT_EQ(output.numSubgraphUpdated, 1);
}

// 全 Merge 集成: carry 环上(sg0/sg1)与环外支路(sg2)混合的候选组被门控拦截, 保持 3 子图
TEST_F(ReduceCopyTest, Merge_LoopPathGateBlocksBranchAbsorption)
{
    std::vector<std::set<int>> outGraph{{1}, {2}, {}};
    MergeInput input = BuildSimpleMergeInput(3, outGraph, {{0, 1, 2}});
    input.isEnforceMergeGroup = {false};
    input.subgraphLoopPaths = {{7}, {7}, {}};
    input.boundaryTensors = {{100, {0}, {1}, true, {}, {}}, {101, {1}, {2}, true, {}, {}}};
    input.subgraphToBoundaryTensorIds = {{0}, {0, 1}, {1}};
    MixGraphMerger merger;
    MergeOutput output = merger.Merge(input);
    EXPECT_EQ(output.numSubgraphUpdated, 3);
}

// 同环全成员: 链内合并不受门控影响, 收敛到 1 子图 (gqa 29->1 场景抽象)
TEST_F(ReduceCopyTest, Merge_LoopPathGateAllowsSamePathChain)
{
    std::vector<std::set<int>> outGraph{{1}, {2}, {}};
    MergeInput input = BuildSimpleMergeInput(3, outGraph, {{0, 1, 2}});
    input.isEnforceMergeGroup = {false};
    input.subgraphLoopPaths = {{7}, {7}, {7}};
    input.boundaryTensors = {{100, {0}, {1}, true, {}, {}}, {101, {1}, {2}, true, {}, {}}};
    input.subgraphToBoundaryTensorIds = {{0}, {0, 1}, {1}};
    MixGraphMerger merger;
    MergeOutput output = merger.Merge(input);
    EXPECT_EQ(output.numSubgraphUpdated, 1);
}

// 无 slot 信息: 门控不启用 -> 收敛到 1 子图 (mla / gqa-PATH0 等无 feedback slot 场景)
TEST_F(ReduceCopyTest, Merge_NoSlotInfoGateDisabledMergesAll)
{
    std::vector<std::set<int>> outGraph{{1}, {2}, {}};
    MergeInput input = BuildSimpleMergeInput(3, outGraph, {{0, 1, 2}});
    input.isEnforceMergeGroup = {false};
    input.boundaryTensors = {{100, {0}, {1}, true, {}, {}}, {101, {1}, {2}, true, {}, {}}};
    input.subgraphToBoundaryTensorIds = {{0}, {0, 1}, {1}};
    MixGraphMerger merger;
    MergeOutput output = merger.Merge(input);
    EXPECT_EQ(output.numSubgraphUpdated, 1);
}

// 端到端(RunOnFunction): feedback slot 7 标记 carry 链 sg0(读 incast)->sg1(写 outcast);
// 环外 sg2 挂在 carryOut 数据边上(生产者-消费者耦合), 数据耦合豁免允许并入 carry 链,
// sg2->sg3 为链式下游(组内可达) -> 全图归 1
TEST_F(ReduceCopyTest, RunOnFunction_CarrySlotGateAbsorbsDataCoupledBranch)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> incasts;
    std::vector<std::string> outcasts;
    std::string carryIn = AddIncast(G, "carryIn", incasts);
    std::string k0 = AddIncast(G, "k0", incasts);
    std::string c0 = AddCubeMatmulFrom(G, 0, carryIn, k0); // sg0: carry reader (AIC)
    std::string carryOut = AddVecSG(G, 1, c0, "carryOut"); // sg1: carry writer (AIV)
    outcasts.push_back(carryOut);
    std::string v2 = AddVecSG(G, 2, carryOut, "v2"); // sg2: 环外支路 (AIV)
    outcasts.push_back(v2);
    std::string k3 = AddIncast(G, "k3", incasts);
    (void)AddCubeMatmulFrom(G, 3, v2, k3); // sg3: 环外支路 (AIC)
    Function* function = G.GetFunction();
    function->SetTotalSubGraphCount(4);
    ASSERT_EQ(G.SetInCast(incasts), true);
    ASSERT_EQ(G.SetOutCast(outcasts), true);
    auto slotScope = std::make_shared<TensorSlotScope>(function);
    slotScope->ioslot.incastSlot = {{7}, {}};
    slotScope->ioslot.outcastSlot = {{7}, {}};
    function->SetSlotScope(slotScope);
    function->paramConfigs_.autoMixPartition = 1;
    ReduceCopyMerge merger;
    EXPECT_EQ(merger.RunOnFunction(*function), SUCCESS);
    // sg0/sg1(同环)与环外 sg2(carryOut 数据边挂靠)/sg3(链式下游)依次并入 -> 1 子图
    const int Num1 = 1;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num1);
}

// 双 feedback slot(7/8) 共享读端子图 sg0 —— online softmax 的 out/sum/max 同环抽象:
// 两条 carry 链共享端点 -> 并查集归为同一环, sg0/sg1/sg2 同 pathId, 链间合并放行 -> 1 子图
// (若按链独立归类, sg0={7,8}/sg1={7}/sg2={8} 通路混杂会被误拒 -> 3 子图)
TEST_F(ReduceCopyTest, RunOnFunction_SharedEndpointSlotsGroupIntoSameRing)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> incasts;
    std::vector<std::string> outcasts;
    std::string carryInA = AddIncast(G, "carryInA", incasts);     // slot 7 incast
    std::string carryInB = AddIncast(G, "carryInB", incasts);     // slot 8 incast
    std::string c0 = AddCubeMatmulFrom(G, 0, carryInA, carryInB); // sg0: 两条链共享的 carry reader
    std::string o7 = AddVecSG(G, 1, c0, "o7");                    // sg1: slot 7 writer
    outcasts.push_back(o7);
    std::string o8 = AddVecSG(G, 2, c0, "o8"); // sg2: slot 8 writer
    outcasts.push_back(o8);
    Function* function = G.GetFunction();
    function->SetTotalSubGraphCount(3);
    ASSERT_EQ(G.SetInCast(incasts), true);
    ASSERT_EQ(G.SetOutCast(outcasts), true);
    auto slotScope = std::make_shared<TensorSlotScope>(function);
    slotScope->ioslot.incastSlot = {{7}, {8}};
    slotScope->ioslot.outcastSlot = {{7}, {8}};
    function->SetSlotScope(slotScope);
    function->paramConfigs_.autoMixPartition = 1;
    ReduceCopyMerge merger;
    EXPECT_EQ(merger.RunOnFunction(*function), SUCCESS);
    const int Num1 = 1;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num1);
}

// 双 feedback slot 无共享端点(独立环): 环1 sg0(读i7)->sg1(写o7), 环2 sg2(读i8)->sg3(写o8);
// o7 同时作为跨环边被 sg2 消费, 候选组 {0,1,2,3} 通路混杂({7},{7},{8},{8})被门控拒绝,
// 同环子组 {0,1}/{2,3} 各自合并 -> 2 子图 (无 slot 信息时同图会合并到 1)
TEST_F(ReduceCopyTest, RunOnFunction_IndependentRingsRejectCrossRingMerge)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> incasts;
    std::vector<std::string> outcasts;
    std::string carryInA = AddIncast(G, "carryInA", incasts); // slot 7
    std::string k0 = AddIncast(G, "k0", incasts);
    std::string c0 = AddCubeMatmulFrom(G, 0, carryInA, k0); // sg0: 环1 reader
    std::string o7 = AddVecSG(G, 1, c0, "o7");              // sg1: 环1 writer
    outcasts.push_back(o7);
    std::string carryInB = AddIncast(G, "carryInB", incasts); // slot 8
    std::string c2 = AddCubeMatmulFrom(G, 2, o7, carryInB);   // sg2: 环2 reader, 消费 o7 形成跨环边
    std::string o8 = AddVecSG(G, 3, c2, "o8");                // sg3: 环2 writer
    outcasts.push_back(o8);
    Function* function = G.GetFunction();
    function->SetTotalSubGraphCount(4);
    ASSERT_EQ(G.SetInCast(incasts), true);
    ASSERT_EQ(G.SetOutCast(outcasts), true);
    auto slotScope = std::make_shared<TensorSlotScope>(function);
    slotScope->ioslot.incastSlot = {{7}, {}, {8}};
    slotScope->ioslot.outcastSlot = {{7}, {8}};
    function->SetSlotScope(slotScope);
    function->paramConfigs_.autoMixPartition = 1;
    ReduceCopyMerge merger;
    EXPECT_EQ(merger.RunOnFunction(*function), SUCCESS);
    const int Num2 = 2;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num2);
}

// 多轮合并含 root 轮转的通路继承: 先合并 {1,2}(root=1, 秩1), 再合并 {0,1} —— GetActualGroup
// 升序返回 {0,1}, actualGroup[0]=0(秩0) 按秩挂在 1 下, 真实 root=sg1; 异构通路集合 {9}/{7}
// 必须完整累计到真实 root(写错到 actualGroup[0]=0 的条目会使 root 残缺为 {7})
TEST_F(ReduceCopyTest, Merge_RootRotationAccumulatesLoopPathsOnRealRoot)
{
    std::vector<std::set<int>> outGraph{{1}, {2}, {}};
    MergeInput input = BuildSimpleMergeInput(3, outGraph, {{1, 2}, {0, 1}});
    input.subgraphLoopPaths = {{9}, {7}, {7}};
    MixGraphMerger merger;
    MergeOutput output = merger.Merge(input);
    EXPECT_EQ(output.numSubgraphUpdated, 1);
    EXPECT_EQ(output.subgraphIdUpdated[0], output.subgraphIdUpdated[1]);
    EXPECT_EQ(output.subgraphIdUpdated[1], output.subgraphIdUpdated[2]);
    EXPECT_EQ(merger.FindParent(0), 1); // 根轮转到 sg1, actualGroup[0]=0 已不是集合根
    std::set<int> expected{7, 9};
    EXPECT_EQ(merger.mRootLoopPaths[merger.FindParent(0)], expected);
}

// glm 场景变体: 分支均衡但未达悬殊 (分支 600 < 35×sink150), sink 放行合并 -> 全图归一。
TEST_F(ReduceCopyTest, SinkMergesWhenBalancedNotDwarfing)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> incasts;
    std::vector<std::string> outcasts;
    AddCubeMatmulFrom(G, 0, AddIncast(G, "QA0", incasts), AddIncast(G, "QB0", incasts));
    AddVecSG(G, 1, "tC0", "v1a");
    std::string c2a = AddCubeMatmulFrom(G, 2, "v1a", AddIncast(G, "VA0", incasts));
    AddCubeMatmulFrom(G, 3, AddIncast(G, "QA1", incasts), AddIncast(G, "QB1", incasts));
    AddVecSG(G, 4, "tC3", "v1b");
    std::string c2b = AddCubeMatmulFrom(G, 5, "v1b", AddIncast(G, "VA1", incasts));
    AddVecSinkSG(G, 6, {c2a, c2b}, "sinkGlmOut", outcasts);
    Function* function = G.GetFunction();
    function->SetTotalSubGraphCount(7);
    ASSERT_EQ(G.SetInCast(incasts), true);
    ASSERT_EQ(G.SetOutCast(outcasts), true);
    function->paramConfigs_.autoMixPartition = 1;
    ReduceCopyMerge merger;
    EXPECT_EQ(merger.RunOnFunction(*function), SUCCESS);
    const int Num1 = 1;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num1);
}

// 纯汇合点: sink 延迟清零(rootLatency=0), 无收益来源, 分支 600/600 串行损失 600 > 0 -> 拒绝合并。
TEST_F(ReduceCopyTest, SinkRejectsPureConvergencePoint)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> incasts;
    std::vector<std::string> outcasts;
    AddCubeMatmulFrom(G, 0, AddIncast(G, "QA0", incasts), AddIncast(G, "QB0", incasts));
    AddVecSG(G, 1, "tC0", "v1a");
    std::string c2a = AddCubeMatmulFrom(G, 2, "v1a", AddIncast(G, "VA0", incasts));
    AddCubeMatmulFrom(G, 3, AddIncast(G, "QA1", incasts), AddIncast(G, "QB1", incasts));
    AddVecSG(G, 4, "tC3", "v1b");
    std::string c2b = AddCubeMatmulFrom(G, 5, "v1b", AddIncast(G, "VA1", incasts));
    AddVecSinkSG(G, 6, {c2a, c2b}, "sinkGlmOut", outcasts);
    Function* function = G.GetFunction();
    // sink(sg6) 延迟清零成纯汇合点
    const int Num0 = 0;
    for (auto& op : function->Operations()) {
        if (op.GetSubgraphID() == 6) {
            op.UpdateLatency(Num0);
        }
    }
    function->SetTotalSubGraphCount(7);
    ASSERT_EQ(G.SetInCast(incasts), true);
    ASSERT_EQ(G.SetOutCast(outcasts), true);
    function->paramConfigs_.autoMixPartition = 1;
    ReduceCopyMerge merger;
    EXPECT_EQ(merger.RunOnFunction(*function), SUCCESS);
    const int Num3 = 3;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num3);
}

// 不均衡分支: 分支 A=600/B=60, 串行损失 60 < 8×sink150 可被消化 -> 放行(不再要求均衡)。
TEST_F(ReduceCopyTest, SinkMergesWhenUnbalancedLossAbsorbable)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> incasts;
    std::vector<std::string> outcasts;
    AddCubeMatmulFrom(G, 0, AddIncast(G, "QA0", incasts), AddIncast(G, "QB0", incasts));
    AddVecSG(G, 1, "tC0", "v1a");
    std::string c2a = AddCubeMatmulFrom(G, 2, "v1a", AddIncast(G, "VA0", incasts));
    AddCubeMatmulFrom(G, 3, AddIncast(G, "QA1", incasts), AddIncast(G, "QB1", incasts));
    AddVecSG(G, 4, "tC3", "v1b");
    std::string c2b = AddCubeMatmulFrom(G, 5, "v1b", AddIncast(G, "VA1", incasts));
    AddVecSinkSG(G, 6, {c2a, c2b}, "sinkGlmOut", outcasts);
    Function* function = G.GetFunction();
    // 分支 B(sg3-5) 压到 12-op×5=60, 与分支 A(600) 不均衡
    const int Num5 = 5;
    for (auto& op : function->Operations()) {
        if (op.GetSubgraphID() >= 3 && op.GetSubgraphID() <= 5) {
            op.UpdateLatency(Num5);
        }
    }
    function->SetTotalSubGraphCount(7);
    ASSERT_EQ(G.SetInCast(incasts), true);
    ASSERT_EQ(G.SetOutCast(outcasts), true);
    function->paramConfigs_.autoMixPartition = 1;
    ReduceCopyMerge merger;
    EXPECT_EQ(merger.RunOnFunction(*function), SUCCESS);
    const int Num1 = 1;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num1);
}

// 边界: 分支 120/120, sink 15, serialLoss=120 == 8×15, 严格大于才拒绝 -> 放行。
TEST_F(ReduceCopyTest, SinkBoundaryLossEqualRatioAllows)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> incasts;
    std::vector<std::string> outcasts;
    AddCubeMatmulFrom(G, 0, AddIncast(G, "QA0", incasts), AddIncast(G, "QB0", incasts));
    AddVecSG(G, 1, "tC0", "v1a");
    std::string c2a = AddCubeMatmulFrom(G, 2, "v1a", AddIncast(G, "VA0", incasts));
    AddCubeMatmulFrom(G, 3, AddIncast(G, "QA1", incasts), AddIncast(G, "QB1", incasts));
    AddVecSG(G, 4, "tC3", "v1b");
    std::string c2b = AddCubeMatmulFrom(G, 5, "v1b", AddIncast(G, "VA1", incasts));
    AddVecSinkSG(G, 6, {c2a, c2b}, "sinkGlmOut", outcasts);
    Function* function = G.GetFunction();
    // 分支(sg0-5) 压到 12-op×10=120, sink(sg6) 压到 3-op×5=15
    const int Num10 = 10;
    const int Num5 = 5;
    for (auto& op : function->Operations()) {
        if (op.GetSubgraphID() <= 5) {
            op.UpdateLatency(Num10);
        } else if (op.GetSubgraphID() == 6) {
            op.UpdateLatency(Num5);
        }
    }
    function->SetTotalSubGraphCount(7);
    ASSERT_EQ(G.SetInCast(incasts), true);
    ASSERT_EQ(G.SetOutCast(outcasts), true);
    function->paramConfigs_.autoMixPartition = 1;
    ReduceCopyMerge merger;
    EXPECT_EQ(merger.RunOnFunction(*function), SUCCESS);
    const int Num1 = 1;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num1);
}

// ============================================================================
// 本次修改 UT: hinge guard 纯 V 轻量分支规模豁免
// 对应 reduce_copy.cpp CheckMergeBenefitByStructuralPattern 的豁免分支:
// 独立分支全纯 V(无 AIC op) 且串行损失(Σbranch − max) <= kSerialLossRatio(8) × hinge
// 端点 root 总 latency 时放行; 含 C 分支或损失超限仍拒绝
// ============================================================================

// 豁免放行(扇出): hinge 端点 sg0(含 C, 900) 扇出 3 个互不可达纯 V 轻量分支(各 50),
// 串行损失 150-50=100 <= 8x900 -> 豁免放行归 1
// (同构型无豁免时被拒保持 4, 见 MixGraphMerger_HingeGuardRejectsParallelFanout)
TEST_F(ReduceCopyTest, MixGraphMerger_HingeGuardExemptsPureVecLightFanout)
{
    std::vector<std::set<int>> outGraph{{1, 2, 3}, {}, {}, {}};
    MergeInput input = BuildSimpleMergeInput(4, outGraph, {{0, 1, 2, 3}});
    input.isEnforceMergeGroup = {false};
    input.boundaryTensors = {
        {100, {0}, {1}, true, {}, {}}, {101, {0}, {2}, true, {}, {}}, {102, {0}, {3}, true, {}, {}}};
    input.subgraphToBoundaryTensorIds = {{0, 1, 2}, {0}, {1}, {2}};
    input.subgraphAICOpNum = {2, 0, 0, 0}; // 仅 hinge 端点含 AIC op, 分支全纯 V
    input.subgraphAICLatency = {800, 0, 0, 0};
    input.subgraphAIVLatency = {100, 50, 50, 50};
    MixGraphMerger merger;
    MergeOutput output = merger.Merge(input);
    EXPECT_EQ(output.numSubgraphUpdated, 1);
}

// 含 C 分支不豁免(扇出): 分支 sg1 含 AIC op, 串行损失同上 100 <= 8x900 本可达标,
// 但豁免限纯 V 分支 -> 拒绝保持 4 (钉住豁免判据的纯 V 子句)
TEST_F(ReduceCopyTest, MixGraphMerger_HingeGuardKeepsCubeBranchFanoutRejected)
{
    std::vector<std::set<int>> outGraph{{1, 2, 3}, {}, {}, {}};
    MergeInput input = BuildSimpleMergeInput(4, outGraph, {{0, 1, 2, 3}});
    input.isEnforceMergeGroup = {false};
    input.boundaryTensors = {
        {100, {0}, {1}, true, {}, {}}, {101, {0}, {2}, true, {}, {}}, {102, {0}, {3}, true, {}, {}}};
    input.subgraphToBoundaryTensorIds = {{0, 1, 2}, {0}, {1}, {2}};
    input.subgraphAICOpNum = {2, 1, 0, 0}; // 分支 sg1 含 AIC op, 破坏全纯 V
    input.subgraphAICLatency = {800, 50, 0, 0};
    input.subgraphAIVLatency = {100, 50, 50, 50};
    MixGraphMerger merger;
    MergeOutput output = merger.Merge(input);
    EXPECT_EQ(output.numSubgraphUpdated, 4);
}

// 纯 V 但串行损失超限(扇出): hinge 40, 分支 500/400/300, 损失 1200-500=700 > 8x40=320
// -> 不豁免拒绝保持 4 (钉住豁免判据的损失子句)
TEST_F(ReduceCopyTest, MixGraphMerger_HingeGuardRejectsHeavyPureVecFanout)
{
    std::vector<std::set<int>> outGraph{{1, 2, 3}, {}, {}, {}};
    MergeInput input = BuildSimpleMergeInput(4, outGraph, {{0, 1, 2, 3}});
    input.isEnforceMergeGroup = {false};
    input.boundaryTensors = {
        {100, {0}, {1}, true, {}, {}}, {101, {0}, {2}, true, {}, {}}, {102, {0}, {3}, true, {}, {}}};
    input.subgraphToBoundaryTensorIds = {{0, 1, 2}, {0}, {1}, {2}};
    input.subgraphAICOpNum = {2, 0, 0, 0};
    input.subgraphAICLatency = {20, 0, 0, 0};
    input.subgraphAIVLatency = {20, 500, 400, 300};
    MixGraphMerger merger;
    MergeOutput output = merger.Merge(input);
    EXPECT_EQ(output.numSubgraphUpdated, 4);
}

// 边界(扇出): hinge 25, 分支 500/100/100, 损失 700-500=200 == 8x25, 等号放行归 1
TEST_F(ReduceCopyTest, MixGraphMerger_HingeGuardPureVecBoundaryEqualRatioAllows)
{
    std::vector<std::set<int>> outGraph{{1, 2, 3}, {}, {}, {}};
    MergeInput input = BuildSimpleMergeInput(4, outGraph, {{0, 1, 2, 3}});
    input.isEnforceMergeGroup = {false};
    input.boundaryTensors = {
        {100, {0}, {1}, true, {}, {}}, {101, {0}, {2}, true, {}, {}}, {102, {0}, {3}, true, {}, {}}};
    input.subgraphToBoundaryTensorIds = {{0, 1, 2}, {0}, {1}, {2}};
    input.subgraphAICOpNum = {2, 0, 0, 0};
    input.subgraphAICLatency = {25, 0, 0, 0};
    input.subgraphAIVLatency = {0, 500, 100, 100};
    MixGraphMerger merger;
    MergeOutput output = merger.Merge(input);
    EXPECT_EQ(output.numSubgraphUpdated, 1);
}

// 扇入(sparse_attention_antiquant 型): 3 个分支产 tensor 汇入 hinge 端点 sg0(含 C, 900)。
// 分支 sg1 含 AIC: sink 串行损失门放行(100 <= 8x900)后 hinge 豁免因非纯 V 拒绝 -> 保持 4
TEST_F(ReduceCopyTest, MixGraphMerger_HingeGuardKeepsCubeBranchFaninRejected)
{
    std::vector<std::set<int>> outGraph{{}, {0}, {0}, {0}};
    MergeInput input = BuildSimpleMergeInput(4, outGraph, {{0, 1, 2, 3}});
    input.isEnforceMergeGroup = {false};
    input.boundaryTensors = {
        {100, {1}, {0}, true, {}, {}}, {101, {2}, {0}, true, {}, {}}, {102, {3}, {0}, true, {}, {}}};
    input.subgraphToBoundaryTensorIds = {{0, 1, 2}, {0}, {1}, {2}};
    input.subgraphAICOpNum = {2, 1, 0, 0};
    input.subgraphAICLatency = {800, 50, 0, 0};
    input.subgraphAIVLatency = {100, 50, 50, 50};
    MixGraphMerger merger;
    MergeOutput output = merger.Merge(input);
    EXPECT_EQ(output.numSubgraphUpdated, 4);
}

// 扇入豁免放行: 同上拓扑但分支全纯 V 轻量(各 50), 损失 100 <= 8x900, sink 门与 hinge
// 豁免同标尺均放行 -> 归 1 (无 hinge 豁免时该形态被拒保持 4)
TEST_F(ReduceCopyTest, MixGraphMerger_HingeGuardExemptsPureVecLightFanin)
{
    std::vector<std::set<int>> outGraph{{}, {0}, {0}, {0}};
    MergeInput input = BuildSimpleMergeInput(4, outGraph, {{0, 1, 2, 3}});
    input.isEnforceMergeGroup = {false};
    input.boundaryTensors = {
        {100, {1}, {0}, true, {}, {}}, {101, {2}, {0}, true, {}, {}}, {102, {3}, {0}, true, {}, {}}};
    input.subgraphToBoundaryTensorIds = {{0, 1, 2}, {0}, {1}, {2}};
    input.subgraphAICOpNum = {2, 0, 0, 0};
    input.subgraphAICLatency = {800, 0, 0, 0};
    input.subgraphAIVLatency = {100, 50, 50, 50};
    MixGraphMerger merger;
    MergeOutput output = merger.Merge(input);
    EXPECT_EQ(output.numSubgraphUpdated, 1);
}

// 档位映射: 0=关闭(high 上限仅作 enforce WARN 观测), 1=high 档(旧值兼容, 行为同旧版),
// 2=default 档, 3~100=非法自定义值回退 default 档, >100=自定义总 op 数上限(AIC/AIV 上限置 0 表示不启用)
// 数值与 reduce_copy.cpp 的 kAutoMixHighMaxAICOpNum/kAutoMixHighMaxAIVOpNum/
// kAutoMixDefaultMaxAICOpNum/kAutoMixDefaultMaxAIVOpNum 保持一致
TEST_F(ReduceCopyTest, ReduceCopyMerge_MapAutoMixPartitionToLimits)
{
    int maxAIC = 0;
    int maxAIV = 0;
    int maxTotal = 0;
    EXPECT_FALSE(ReduceCopyMerge::MapAutoMixPartitionToLimits(0, maxAIC, maxAIV, maxTotal));
    EXPECT_EQ(maxAIC, 2000);
    EXPECT_EQ(maxAIV, 2240);
    EXPECT_EQ(maxTotal, 0);
    EXPECT_TRUE(ReduceCopyMerge::MapAutoMixPartitionToLimits(1, maxAIC, maxAIV, maxTotal));
    EXPECT_EQ(maxAIC, 2000);
    EXPECT_EQ(maxAIV, 2240);
    EXPECT_EQ(maxTotal, 0);
    EXPECT_TRUE(ReduceCopyMerge::MapAutoMixPartitionToLimits(2, maxAIC, maxAIV, maxTotal));
    EXPECT_EQ(maxAIC, 1700);
    EXPECT_EQ(maxAIV, 400);
    EXPECT_EQ(maxTotal, 0);
    // 3~100 非法自定义值: 回退 default 档上限
    EXPECT_TRUE(ReduceCopyMerge::MapAutoMixPartitionToLimits(3, maxAIC, maxAIV, maxTotal));
    EXPECT_EQ(maxAIC, 1700);
    EXPECT_EQ(maxAIV, 400);
    EXPECT_EQ(maxTotal, 0);
    EXPECT_TRUE(ReduceCopyMerge::MapAutoMixPartitionToLimits(100, maxAIC, maxAIV, maxTotal));
    EXPECT_EQ(maxAIC, 1700);
    EXPECT_EQ(maxAIV, 400);
    EXPECT_EQ(maxTotal, 0);
    // >100 自定义档: 仅约束总 op 数
    EXPECT_TRUE(ReduceCopyMerge::MapAutoMixPartitionToLimits(101, maxAIC, maxAIV, maxTotal));
    EXPECT_EQ(maxAIC, 0);
    EXPECT_EQ(maxAIV, 0);
    EXPECT_EQ(maxTotal, 101);
}

// 总 op 数上限独立检查: {0,1} 总 op 120 超限拒绝, {2,3} 总 op 20 限内通过(自定义档不分别限制 AIC/AIV)
TEST_F(ReduceCopyTest, MixGraphMerger_TotalOpNumLimitRejectsMerge)
{
    MergeInput input = BuildSimpleMergeInput(4, {{1}, {}, {3}, {}}, {{0, 1}, {2, 3}});
    input.subgraphAICOpNum = {30, 30, 5, 5};
    input.subgraphAIVOpNum = {30, 30, 5, 5};
    input.maxSubgraphTotalOpNum = 50;
    MixGraphMerger merger;
    merger.mInput = input;
    merger.mParent = {0, 1, 2, 3};
    EXPECT_FALSE(merger.CheckLatencyConstraint({0, 1}));
    EXPECT_TRUE(merger.CheckLatencyConstraint({2, 3}));
}

// 对照: 同图走 default 档(独立上限 1700/400)不受影响, 子图合并到 2
TEST_F(ReduceCopyTest, RunOnFunction_DefaultLevelMerges)
{
    ComputationalGraphBuilder G;
    BuildMatmulAddsGraph(G);
    Function* function = G.GetFunction();
    function->paramConfigs_.autoMixPartition = 2;
    ReduceCopyMerge merger;
    EXPECT_EQ(merger.RunOnFunction(*function), SUCCESS);
    const int Num2 = 2;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num2);
}

// 对照: 同图走 high 档(独立上限 2000/2240, 旧值 1 兼容)不受影响, 子图合并到 2
TEST_F(ReduceCopyTest, RunOnFunction_HighLevelMerges)
{
    ComputationalGraphBuilder G;
    BuildMatmulAddsGraph(G);
    Function* function = G.GetFunction();
    function->paramConfigs_.autoMixPartition = 1;
    ReduceCopyMerge merger;
    EXPECT_EQ(merger.RunOnFunction(*function), SUCCESS);
    const int Num2 = 2;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num2);
}

// 自定义档单侧上限置 0 不启用: 单侧检查跳过(不因 0 上限误拒), 总 op 数上限仍生效:
// {0,1} 总 op 120 超总限 50 拒绝, {2,3} 总 op 20 限内通过
TEST_F(ReduceCopyTest, MixGraphMerger_ZeroSideLimitSkipsSideCheck)
{
    MergeInput input = BuildSimpleMergeInput(4, {{1}, {}, {3}, {}}, {{0, 1}, {2, 3}});
    input.subgraphAICOpNum = {30, 30, 5, 5};
    input.subgraphAIVOpNum = {30, 30, 5, 5};
    input.maxSubgraphAICOpNum = 0;
    input.maxSubgraphAIVOpNum = 0;
    input.maxSubgraphTotalOpNum = 50;
    MixGraphMerger merger;
    merger.mInput = input;
    merger.mParent = {0, 1, 2, 3};
    EXPECT_FALSE(merger.CheckLatencyConstraint({0, 1}));
    EXPECT_TRUE(merger.CheckLatencyConstraint({2, 3}));
}

// 对照: 同图关闭 auto-mix(0), 子图数保持 6
TEST_F(ReduceCopyTest, RunOnFunction_DisabledKeepsSubgraphs)
{
    ComputationalGraphBuilder G;
    BuildMatmulAddsGraph(G);
    Function* function = G.GetFunction();
    function->paramConfigs_.autoMixPartition = 0;
    ReduceCopyMerge merger;
    EXPECT_EQ(merger.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(function->GetTotalSubGraphCount(), 6);
}

// 对照: 同上拓扑无 slotScope 时门控不启用, 全部合并到 1 子图 (mla / gqa-PATH0 场景)
TEST_F(ReduceCopyTest, RunOnFunction_NoSlotScopeMergesAll)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> incasts;
    std::vector<std::string> outcasts;
    std::string carryIn = AddIncast(G, "carryIn", incasts);
    std::string k0 = AddIncast(G, "k0", incasts);
    std::string c0 = AddCubeMatmulFrom(G, 0, carryIn, k0);
    std::string carryOut = AddVecSG(G, 1, c0, "carryOut");
    outcasts.push_back(carryOut);
    std::string v2 = AddVecSG(G, 2, carryOut, "v2");
    outcasts.push_back(v2);
    std::string k3 = AddIncast(G, "k3", incasts);
    (void)AddCubeMatmulFrom(G, 3, v2, k3);
    Function* function = G.GetFunction();
    function->SetTotalSubGraphCount(4);
    ASSERT_EQ(G.SetInCast(incasts), true);
    ASSERT_EQ(G.SetOutCast(outcasts), true);
    function->paramConfigs_.autoMixPartition = 1;
    ReduceCopyMerge merger;
    EXPECT_EQ(merger.RunOnFunction(*function), SUCCESS);
    const int Num1 = 1;
    EXPECT_EQ(function->GetTotalSubGraphCount(), Num1);
}

} // namespace tile_fwk
} // namespace npu
