/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_codegen_dyn_indexadd.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "tilefwk/platform.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_common.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk {

class TestCodegenDynIndexAdd : public CodegenTestBase {
public:
    TestCodegenDynIndexAdd() : CodegenTestBase({.compileStage = CS_CODEGEN_INSTRUCTION}) {}

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }
};

TEST_F(TestCodegenDynIndexAdd, TestIndexAddUB)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
    constexpr const int S1 = 32;
    constexpr const int D = 64;
    constexpr const int S2 = 16;
    std::vector<int64_t> shape0 = {S1, D};
    std::vector<int64_t> shape1 = {S2, D};
    int axis = 0;
    std::vector<int64_t> shape2 = {shape1[axis]};

    TileShape::Current().SetVecTile({S2, D});

    Tensor inputSrc0(DT_FP32, shape0, "x1");
    Tensor inputSrc1(DT_FP32, shape1, "x2");
    Tensor inputIndex(DT_INT32, shape2, "indices");
    Tensor output(DT_FP32, shape0, "output");
    Element alphaVal(DataType::DT_FP32, 1.0);

    std::string funcName = "TestIndexAddUB";
    FUNCTION(funcName, {inputSrc0, inputSrc1, inputIndex}, {output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = IndexAddUB(inputSrc0, inputSrc1, inputIndex, axis, alphaVal);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);
    (void)GenCodeByFunction(*function);
}

TEST_F(TestCodegenDynIndexAdd, TestIndexAddUBLayout)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    constexpr const int S1 = 16;
    constexpr const int D = 32;
    constexpr const int S2 = 8;
    std::vector<int64_t> shape0 = {S1, D};
    std::vector<int64_t> shape1 = {S2, D};
    int axis = 0;
    std::vector<int64_t> shape2 = {shape1[axis]};

    TileShape::Current().SetVecTile({S2, D});

    Tensor inputSrc0(DT_FP32, shape0, "x1");
    Tensor inputSrc1(DT_FP32, shape1, "x2");
    Tensor inputIndex(DT_INT32, shape2, "indices");
    Tensor output(DT_FP32, shape0, "output");
    Element alphaVal(DataType::DT_FP32, 1.0);

    ConfigManager::Instance();
    std::string funcName = "IndexAddUBLayout";
    FUNCTION(funcName, {inputSrc0, inputSrc1, inputIndex}, {output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = IndexAddUB(inputSrc0, inputSrc1, inputIndex, axis, alphaVal);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);
    (void)GenCodeByFunction(*function);
}

TEST_F(TestCodegenDynIndexAdd, TestIndexAddLayout)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    constexpr const int S1 = 16;
    constexpr const int D = 32;
    constexpr const int S2 = 8;
    std::vector<int64_t> shape0 = {S1, D};
    std::vector<int64_t> shape1 = {S2, D};
    int axis = 0;
    std::vector<int64_t> shape2 = {shape1[axis]};

    TileShape::Current().SetVecTile({S2, D});

    Tensor inputSrc0(DT_FP32, shape0, "x1");
    Tensor inputSrc1(DT_FP32, shape1, "x2");
    Tensor inputIndex(DT_INT32, shape2, "indices");
    Tensor output(DT_FP32, shape0, "output");
    Element alphaVal(DataType::DT_FP32, 1.0);

    ConfigManager::Instance();
    std::string funcName = "IndexAddLayout";
    FUNCTION(funcName, {inputSrc0, inputSrc1, inputIndex}, {output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            IndexAdd_(inputSrc0, inputSrc1, inputIndex, axis, alphaVal);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);
    std::string res = GenCodeByFunction(*function);
    std::string expect =
        R"!!!(TIndexAdd<3>(gmTensor_4, gmTensor_6, ubTensor_0, ubTensor_2, ubTensor_5, Coord2Dim(0, 0), (float)1.f);)!!!";
    CheckStringExist(expect, res);
}
class TestCodegenDynIndexAddSimt : public TestCodegenDynIndexAdd,
                                   public testing::WithParamInterface<std::tuple<NPUArch, DataType, DataType, int>> {
public:
    void SetUp() override
    {
        TestCodegenDynIndexAdd::SetUp();
        savedArch_ = Platform::Instance().GetSoc().GetNPUArch();
        Platform::Instance().GetSoc().SetNPUArch(std::get<0>(GetParam()));
    }

    void TearDown() override
    {
        Platform::Instance().GetSoc().SetNPUArch(savedArch_);
        TestCodegenDynIndexAdd::TearDown();
    }

private:
    NPUArch savedArch_ = NPUArch::DAV_UNKNOWN;
};

TEST_P(TestCodegenDynIndexAddSimt, SelectImplementation)
{
    auto [arch, dtype, indexType, axis] = GetParam();
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    TileShape::Current().SetVecTile({4, 16});
    Tensor target(dtype, {9, 33}, "target");
    Tensor source(dtype, axis == 0 ? Shape{7, 33} : Shape{9, 19}, "source");
    Tensor indices(indexType, {source.GetShape()[axis]}, "indices");
    const std::string funcName = "IndexAddSimtLayout";
    FUNCTION(funcName, {target, source, indices}, {target})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            IndexAdd_(target, source, indices, axis, Element(dtype, 2));
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);
    auto code = GenCodeByFunction(*function);
    bool useSimt = arch == NPUArch::DAV_3510 && indexType != DT_INT64 && dtype != DT_INT32;
    CheckStringExist(std::string(useSimt ? "TIndexAddSimt<" : "TIndexAdd<") + std::to_string(axis + 3) + ">", code);
    if (!useSimt) {
        EXPECT_EQ(code.find("TIndexAddSimt<"), std::string::npos);
    }
}

INSTANTIATE_TEST_SUITE_P(A5AndFallback, TestCodegenDynIndexAddSimt,
                         testing::Values(std::make_tuple(NPUArch::DAV_3510, DT_FP32, DT_INT32, 0),
                                         std::make_tuple(NPUArch::DAV_3510, DT_FP16, DT_INT32, 1),
                                         std::make_tuple(NPUArch::DAV_3510, DT_BF16, DT_INT32, 1),
                                         std::make_tuple(NPUArch::DAV_3510, DT_FP32, DT_INT64, 1),
                                         std::make_tuple(NPUArch::DAV_3510, DT_INT32, DT_INT32, 0),
                                         std::make_tuple(NPUArch::DAV_2201, DT_FP32, DT_INT32, 1)));

} // namespace npu::tile_fwk
