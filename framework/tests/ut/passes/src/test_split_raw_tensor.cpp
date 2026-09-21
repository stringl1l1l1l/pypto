/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_split_raw_tensor.cpp
 * \brief Unit test for SplitRawTensor pass.
 */

#include <vector>
#include "gtest/gtest.h"
#include "interface/function/function.h"
#include "interface/configs/config_manager.h"
#include "ut_json/ut_json_tool.h"
#include "passes/tile_graph_pass/graph_optimization/split_raw.h"
#include "computational_graph_builder.h"

using namespace npu::tile_fwk;

class SplitLargeLocalRawTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};

void AddTensor(ComputationalGraphBuilder& G, const std::string name, const Shape& shape, const Shape& rawShape,
               const Offset& offset, const MemoryType memType)
{
    std::string nameRaw = name + "_raw";
    G.AddTensor(DataType::DT_FP32, rawShape, nameRaw);
    auto tensorRaw = G.GetTensor(nameRaw);
    tensorRaw->SetMemoryTypeBoth(memType, true);
    G.AddTensor(DataType::DT_FP32, shape, name);
    auto tensorLogic = G.GetTensor(name);
    tensorLogic->SetMemoryTypeBoth(memType, true);
    tensorLogic->tensor = tensorRaw->tensor;
    tensorLogic->UpdateOffset(offset);
}

// input -> view1 -> a -> view2 -> b -> assemble1 -> output
TEST_F(SplitLargeLocalRawTest, TestSplitRawTensorCheceker)
{
    int NUM_64 = 64;
    int NUM_128 = 128;
    int NUM_256 = 256;
    int NUM_512 = 512;
    std::vector<int64_t> shape1{NUM_64, NUM_64};
    std::vector<int64_t> shape2{NUM_128, NUM_128};
    std::vector<int64_t> shape3{NUM_256, NUM_256};
    std::vector<int64_t> shape4{NUM_512, NUM_512};
    ComputationalGraphBuilder G;

    AddTensor(G, "input", shape3, shape4, shape3, MemoryType::MEM_DEVICE_DDR);
    AddTensor(G, "a", shape2, shape3, shape2, MemoryType::MEM_DEVICE_DDR);
    AddTensor(G, "b", shape1, shape2, shape1, MemoryType::MEM_UB);
    AddTensor(G, "output", shape3, shape4, shape3, MemoryType::MEM_DEVICE_DDR);

    G.AddOp(Opcode::OP_VIEW, {"input"}, {"a"}, "view1");
    G.GetOp("view1")->SetOpAttribute(std::make_shared<ViewOpAttribute>(shape3, MemoryType::MEM_DEVICE_DDR));
    G.AddOp(Opcode::OP_VIEW, {"a"}, {"b"}, "view2");
    G.GetOp("view2")->SetOpAttribute(std::make_shared<ViewOpAttribute>(shape2, MemoryType::MEM_UB));
    G.AddOp(Opcode::OP_ASSEMBLE, {"b"}, {"output"}, "assemble1");
    G.GetOp("assemble1")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, shape2));

    G.SetInCast({"input"});
    G.SetOutCast({"output"});

    Function* function = G.GetFunction();
    npu::tile_fwk::SplitRawTensor splitLargeLocalRawPass;
    EXPECT_EQ(splitLargeLocalRawPass.PreCheck(*function), SUCCESS);
    EXPECT_EQ(splitLargeLocalRawPass.PostCheck(*function), FAILED);
    EXPECT_EQ(splitLargeLocalRawPass.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(splitLargeLocalRawPass.PostCheck(*function), SUCCESS);
}

TEST_F(SplitLargeLocalRawTest, TestSplitRawTensorUpdateShmemGetToOffset)
{
    constexpr int64_t num64 = 64;
    constexpr int64_t num128 = 128;
    constexpr int64_t num256 = 256;
    constexpr int64_t num512 = 512;
    Shape shape{num128, num128};
    Shape rawShape{num512, num512};
    Offset tensorOffset{num64, num64};
    Offset shmemGetToOffset{num256, num256};
    Offset expectedToOffset{num256 - num64, num256 - num64};
    ComputationalGraphBuilder G;

    AddTensor(G, "pred", {1}, {1}, {0}, MemoryType::MEM_UB);
    AddTensor(G, "shmem_data", shape, shape, {0, 0}, MemoryType::MEM_DEVICE_DDR);
    AddTensor(G, "output", shape, rawShape, tensorOffset, MemoryType::MEM_DEVICE_DDR);
    AddTensor(G, "ub_buffer", shape, shape, {0, 0}, MemoryType::MEM_UB);

    G.AddOp(Opcode::OP_SHMEM_GET, {"pred", "shmem_data"}, {"output", "ub_buffer"}, "shmem_get");
    auto opAttr = std::make_shared<CopyOpAttribute>(
        MemoryType::MEM_DEVICE_DDR, OpImmediate::Specified(shmemGetToOffset), OpImmediate::Specified(shape),
        OpImmediate::Specified(rawShape), OpImmediate::Specified(shape));
    G.GetOp("shmem_get")->SetOpAttribute(opAttr);

    Function* function = G.GetFunction();
    npu::tile_fwk::SplitRawTensor splitLargeLocalRawPass;
    EXPECT_EQ(splitLargeLocalRawPass.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(splitLargeLocalRawPass.PostCheck(*function), SUCCESS);

    auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(G.GetOp("shmem_get")->GetOpAttribute());
    const auto& toOffset = copyAttr->GetToOffset();
    ASSERT_EQ(toOffset.size(), expectedToOffset.size());
    for (size_t i = 0; i < expectedToOffset.size(); i++) {
        ASSERT_TRUE(toOffset[i].IsSpecified());
        ASSERT_TRUE(toOffset[i].GetSpecifiedValue().ConcreteValid());
        EXPECT_EQ(toOffset[i].GetSpecifiedValue().Concrete(), expectedToOffset[i]);
    }
}

TEST_F(SplitLargeLocalRawTest, TestSplitRawTensorWithSymbolicOffset)
{
    constexpr int64_t num128 = 128;
    constexpr int64_t num256 = 256;
    const Shape shape{num128, num128};
    const Shape rawShape{num256, num128};
    const Offset tensorOffset{num128, 0};
    const Offset zeroOffset{0, 0};
    const SymbolicScalar dynBase("dyn_offset");
    const std::vector<SymbolicScalar> dynOffset{(dynBase + num128).Simplify(), 0};
    ComputationalGraphBuilder G;

    AddTensor(G, "input", rawShape, rawShape, zeroOffset, MemoryType::MEM_UB);
    AddTensor(G, "middle", shape, rawShape, tensorOffset, MemoryType::MEM_UB);
    AddTensor(G, "output", shape, shape, zeroOffset, MemoryType::MEM_UB);
    auto middle = G.GetTensor("middle");
    middle->UpdateOffset(TensorOffset(tensorOffset, dynOffset));

    G.AddOp(Opcode::OP_RESHAPE, {"input"}, {"middle"}, "reshape");
    G.AddOp(Opcode::OP_VIEW, {"middle"}, {"output"}, "view");
    G.GetOp("view")->SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset, dynOffset));
    G.SetInCast({"input"});

    Function* function = G.GetFunction();
    SplitRawTensor splitRawTensorPass;
    ASSERT_EQ(splitRawTensorPass.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(splitRawTensorPass.PostCheck(*function), SUCCESS);
    EXPECT_EQ(middle->GetOffset(), zeroOffset);
    EXPECT_TRUE(middle->GetDynOffset().empty());

    auto viewAttr = std::static_pointer_cast<ViewOpAttribute>(G.GetOp("view")->GetOpAttribute());
    EXPECT_EQ(viewAttr->GetFromOffset(), zeroOffset);
    ASSERT_EQ(viewAttr->GetFromDynOffset().size(), zeroOffset.size());
    for (const auto& offset : viewAttr->GetFromDynOffset()) {
        ASSERT_TRUE(offset.IsImmediate());
        EXPECT_EQ(offset.Concrete(), 0);
    }
}
