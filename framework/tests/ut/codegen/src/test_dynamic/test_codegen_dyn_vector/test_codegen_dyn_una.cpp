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
 * \file test_codegen_dyn_una.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/npu/cloudnpu/codegen_op_cloudnpu.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk {
constexpr const unsigned OP_MAGIC3 = 3;
constexpr const unsigned OP_MAGIC4 = 4;
class TestCodegenDynUna : public CodegenTestBase {
public:
    TestCodegenDynUna()
        : CodegenTestBase({.compileStage = CS_EXECUTE_GRAPH,
                           .setTileTensor = true,
                           .tileTensorValue = true,
                           .resetTileTensorOnTearDown = true})
    {}
};

TEST_F(TestCodegenDynUna, TestAbsDynamic)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);

    int S0 = 8;
    int S1 = 4608;
    int D0 = 8;
    int D1 = 4608;

    std::vector<int64_t> srcShape = {S0, S1};
    std::vector<int64_t> dstShape = {D0, D1};

    TileShape::Current().SetVecTile({8, 128});
    Tensor input_a(DataType::DT_FP16, srcShape, "A");
    Tensor output(DataType::DT_FP16, dstShape, "C");

    std::string funcName = "TestAbsDynamic";
    FUNCTION(funcName, {input_a, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = Abs(input_a);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);

    std::string res = GenCodeByFunction(*function);
    std::string expect =
        R"!!!(TileOp::DynTabs_<half, /*DS*/ 1, 8, 128, /*SS*/ 1, 8, 128>((__ubuf__ half*)UB_S0_E2048, (__ubuf__ half*)UB_S0_E2048, 1, 1,)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynUna, TestDynExpand)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);

    std::vector<int64_t> shape = {64, 64};
    std::vector<int64_t> shape1 = {1, 64};
    auto function = GenMockFuncDyn("TestDynExpand");
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    std::vector<SymbolicScalar> dynValidShape1 = {1, 64};
    auto localTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape1, dynValidShape1});
    auto localOutTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});

    auto& op = function->AddOperation(Opcode::OP_EXPAND, {localTensor}, {localOutTensor});
    op.SetAttribute(OpAttributeKey::expandDims, std::vector<int>{0});

    std::string res = GenOpCodeFromOp(*function, op);
    std::string expect =
        R"!!!(TileOp::DynTexpand_<float, /*DS*/ 1, 64, 64, /*SS*/ 1, 1, 64, 2>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 1, 1, 64, 64, 1, 1, 1, 64);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynUna, TestAtanFP32)
{
    std::vector<int64_t> shape = {32, 32};
    TileShape::Current().SetVecTile({32, 32});
    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "output");

    std::string funcName = "TestAtanFP32";
    FUNCTION(funcName, {input, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = Atan(input);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);

    const std::string res = GenCodeByFunction(*function);
    std::string expect = R"!!!(TAtan(ubTensor_2, ubTensor_3, ubTensor_0);
)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynUna, TestPackInt32)
{
    std::vector<int64_t> input_shape = {32, 32};
    std::vector<int64_t> output_shape = {4096};
    TileShape::Current().SetVecTile({32, 32});
    Tensor input(DataType::DT_INT32, input_shape, "input");
    Tensor output(DataType::DT_UINT8, output_shape, "output");

    std::string funcName = "TestPackInt32";
    FUNCTION(funcName, {input, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = Pack(input);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);

    const std::string res = GenCodeByFunction(*function);
    std::string expect = R"!!!(TPack(ubTensor_2, ubTensor_0);
)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynUna, TestUnPackInt32)
{
    std::vector<int64_t> input_shape = {1024};
    std::vector<int64_t> output_shape = {256};
    TileShape::Current().SetVecTile({32});
    Tensor input(DataType::DT_UINT8, input_shape, "input");
    Tensor output(DataType::DT_INT32, output_shape, "output");

    std::string funcName = "TestUnPackInt32";
    FUNCTION(funcName, {input}, {output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = UnPack(input, DataType::DT_INT32);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);

    const std::string res = GenCodeByFunction(*function);
    std::string expect = R"!!!(TUnPack(ubTensor_2, ubTensor_0);)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynUna, TestPadDynamic)
{
    int S0 = 6;
    int S1 = 12;
    int D0 = 12;
    int D1 = 20;

    std::vector<int64_t> srcShape = {S0, S1};
    std::vector<int64_t> dstShape = {D0, D1};

    TileShape::Current().SetVecTile({32, 32});
    Tensor input(DataType::DT_FP32, srcShape, "input");
    Tensor output(DataType::DT_FP32, dstShape, "output");

    std::string funcName = "TestPadDynamic";
    FUNCTION(funcName, {input, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = Pad(input, {0, 6, 0, 8}, "constant", Element(DT_FP32, 2.0));
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);

    const std::string res = GenCodeByFunction(*function);
    std::string expect = R"!!!(TPad<pto::PadValueCustom((float)2.f)>(ubTensor_2, ubTensor_0);)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynUna, TestPadDynamicFP16)
{
    int S0 = 6;
    int S1 = 12;
    int D0 = 12;
    int D1 = 20;

    std::vector<int64_t> srcShape = {S0, S1};
    std::vector<int64_t> dstShape = {D0, D1};

    TileShape::Current().SetVecTile({32, 32});
    Tensor input(DataType::DT_FP16, srcShape, "input");
    Tensor output(DataType::DT_FP16, dstShape, "output");

    std::string funcName = "TestPadDynamicFP16";
    FUNCTION(funcName, {input, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = Pad(input, {0, 6, 0, 8}, "constant", Element(DT_FP16, std::numeric_limits<float>::infinity()));
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);

    const std::string res = GenCodeByFunction(*function);
    std::string expect = R"!!!(TPad<pto::PadValue::Max>(ubTensor_2, ubTensor_0);)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynUna, TestFillPadDynamicBF16)
{
    int S0 = 12;
    int S1 = 20;
    int D0 = 12;
    int D1 = 20;

    std::vector<int64_t> srcShape = {S0, S1};
    std::vector<int64_t> dstShape = {D0, D1};

    TileShape::Current().SetVecTile({32, 32});
    Tensor input(DataType::DT_BF16, srcShape, "input");
    Tensor output(DataType::DT_BF16, dstShape, "output");

    std::string funcName = "TestFillPadDynamicBF16";
    FUNCTION(funcName, {input, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = FillPad(input, "constant", Element(DT_BF16, 0.0f));
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);

    const std::string res = GenCodeByFunction(*function);
    std::string expect = R"!!!(TFillPad<pto::PadValueCustom((bfloat16_t)0.f), 20>(ubTensor_2, ubTensor_0);)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynUna, TestFillPadDynamic)
{
    int S0 = 12;
    int S1 = 20;
    int D0 = 12;
    int D1 = 20;

    std::vector<int64_t> srcShape = {S0, S1};
    std::vector<int64_t> dstShape = {D0, D1};

    TileShape::Current().SetVecTile({32, 32});
    Tensor input(DataType::DT_FP32, srcShape, "input");
    Tensor output(DataType::DT_FP32, dstShape, "output");

    std::string funcName = "TestFillPadDynamic";
    FUNCTION(funcName, {input, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = FillPad(input, "constant", Element(DT_FP32, 0.0f));
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);

    const std::string res = GenCodeByFunction(*function);
    std::string expect = R"!!!(TFillPad<pto::PadValueCustom((float)0.f), 20>(ubTensor_2, ubTensor_0);)!!!";
    CheckStringExist(expect, res);
}

} // namespace npu::tile_fwk
