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
 * \file test_symbolic_scalar_evaluate.cpp
 * \brief Unit test for symbolic scalar evaluate.
 */

#include <vector>
#include <string>
#include <memory>
#include "gtest/gtest.h"
#include "interface/tensor/symbolic_scalar.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/interpreter/function.h"
#include "tilefwk/data_type.h"
#include "interface/tensor/symbolic_scalar_evaluate.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {

class TestSymbolicScalarEvaluate : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(TestSymbolicScalarEvaluate, EvaluateSymbolicCallRuntimeGetInputShapeDimSize)
{
    EvaluateSymbol evaluator;
    std::vector<ScalarImmediateType> dataList = {0};
    std::vector<int64_t> shape = {2, 3, 4};
    auto rawData = std::make_shared<RawTensorData>(DataType::DT_INT32, shape);
    auto tensorData = std::make_shared<LogicalTensorData>(rawData);
    std::vector<std::shared_ptr<LogicalTensorData>> inputList = {tensorData};
    evaluator.InitInputDataViewList(inputList);
    auto ret = evaluator.EvaluateSymbolicCall("RUNTIME_GetInputShapeDimSize", dataList, {});
    EXPECT_EQ(ret, 3);
}

TEST_F(TestSymbolicScalarEvaluate, EvaluateSymbolicCallRuntimeGetInputShapeDim)
{
    EvaluateSymbol evaluator;
    std::vector<ScalarImmediateType> dataList = {0, 1};
    std::vector<int64_t> shape = {2, 3, 4};
    auto rawData = std::make_shared<RawTensorData>(DataType::DT_INT32, shape);
    auto tensorData = std::make_shared<LogicalTensorData>(rawData);
    std::vector<std::shared_ptr<LogicalTensorData>> inputList = {tensorData};
    evaluator.InitInputDataViewList(inputList);
    auto ret = evaluator.EvaluateSymbolicCall("RUNTIME_GetInputShapeDim", dataList, {});
    EXPECT_EQ(ret, 3);
}

template <typename T>
void testGetInputData(DataType dtype, int dim)
{
    EvaluateSymbol evaluator;
    std::vector<int64_t> shape(dim, 5);
    auto rawData = std::make_shared<RawTensorData>(dtype, shape);
    for (int i = 0; i < 5; i++) {
        rawData->Get<T>(i) = i * 10;
    }
    auto tensorData = std::make_shared<LogicalTensorData>(rawData);
    std::vector<std::shared_ptr<LogicalTensorData>> inputList = {tensorData};
    evaluator.InitInputDataViewList(inputList);

    std::vector<ScalarImmediateType> dataList = {0, (int64_t)dtype};
    for (int i = 0; i < dim - 1; i++) {
        dataList.emplace_back(0);
    }
    dataList.emplace_back(2);
    auto ret = evaluator.EvaluateSymbolicCall("RUNTIME_GetInputData", dataList, {});
    EXPECT_EQ(ret, 20);
}

TEST_F(TestSymbolicScalarEvaluate, GetInputData)
{
    for (auto dim = 1; dim < 5; dim++) {
        testGetInputData<int8_t>(DT_BOOL, dim);
        testGetInputData<int8_t>(DT_INT8, dim);
        testGetInputData<int16_t>(DT_INT16, dim);
        testGetInputData<int32_t>(DT_INT32, dim);
        testGetInputData<int64_t>(DT_INT64, dim);
        testGetInputData<uint8_t>(DT_UINT8, dim);
        testGetInputData<uint16_t>(DT_UINT16, dim);
        testGetInputData<uint32_t>(DT_UINT32, dim);
        testGetInputData<uint64_t>(DT_UINT64, dim);
    }
}

TEST_F(TestSymbolicScalarEvaluate, EvaluateSymbolicCallRuntimeIsLoopBegin)
{
    EvaluateSymbol evaluator;
    std::vector<ScalarImmediateType> dataList = {5, 5};
    auto ret = evaluator.EvaluateSymbolicCall("RUNTIME_IsLoopBegin", dataList, {});
    EXPECT_EQ(ret, 1);

    dataList = {5, 6};
    ret = evaluator.EvaluateSymbolicCall("RUNTIME_IsLoopBegin", dataList, {});
    EXPECT_EQ(ret, 0);
}

TEST_F(TestSymbolicScalarEvaluate, EvaluateSymbolicCallRuntimeIsLoopEnd)
{
    EvaluateSymbol evaluator;
    std::vector<ScalarImmediateType> dataList = {10, 10};
    auto ret = evaluator.EvaluateSymbolicCall("RUNTIME_IsLoopEnd", dataList, {});
    EXPECT_EQ(ret, 1);

    dataList = {9, 10};
    ret = evaluator.EvaluateSymbolicCall("RUNTIME_IsLoopEnd", dataList, {});
    EXPECT_EQ(ret, 0);
}

TEST_F(TestSymbolicScalarEvaluate, EvaluateSymbolicCallRuntimeTernaryOP)
{
    EvaluateSymbol evaluator;
    std::vector<ScalarImmediateType> dataList = {1, 7, 9};
    auto ret = evaluator.EvaluateSymbolicCall("RUNTIME_TernaryOP", dataList, {});
    EXPECT_EQ(ret, 7);

    dataList = {0, 7, 9};
    ret = evaluator.EvaluateSymbolicCall("RUNTIME_TernaryOP", dataList, {});
    EXPECT_EQ(ret, 9);

    dataList = {3, 7, 9};
    ret = evaluator.EvaluateSymbolicCall("RUNTIME_TernaryOP", dataList, {});
    EXPECT_EQ(ret, 7);
}

TEST_F(TestSymbolicScalarEvaluate, EvaluateSymbolicCallRuntimeGetViewValidShapeDim)
{
    EvaluateSymbol evaluator;
    std::vector<ScalarImmediateType> dataList = {10, 2, 8};
    auto ret = evaluator.EvaluateSymbolicCall("RUNTIME_GetViewValidShapeDim", dataList, {});
    EXPECT_EQ(ret, 8);

    dataList = {5, 2, 8};
    ret = evaluator.EvaluateSymbolicCall("RUNTIME_GetViewValidShapeDim", dataList, {});
    EXPECT_EQ(ret, 3);
}

TEST_F(TestSymbolicScalarEvaluate, EvaluateSymbolicCallRuntimeGetTensorDataInt)
{
    EvaluateSymbol evaluator;
    std::vector<int64_t> shape = {1};
    auto rawData = std::make_shared<RawTensorData>(DataType::DT_INT32, shape);
    rawData->Get<int32_t>(0) = 42;
    auto tensorData = std::make_shared<LogicalTensorData>(rawData);
    std::vector<std::shared_ptr<LogicalTensorData>> incastList = {tensorData};
    std::vector<std::shared_ptr<LogicalTensorData>> outcastList;
    auto inoutDataPair = std::make_shared<FunctionIODataPair>(incastList, outcastList);
    evaluator.UpdateIODataPair(inoutDataPair);

    auto ret = evaluator.EvaluateSymbolicCall("RUNTIME_COA_GET_PARAM_ADDR", {}, {});
    EXPECT_EQ(ret, 0);
}

TEST_F(TestSymbolicScalarEvaluate, EvaluateSymbolicCallRuntimeGetTensorDataInt32Dim1)
{
    EvaluateSymbol evaluator;
    std::vector<int64_t> shape = {5};
    auto rawData = std::make_shared<RawTensorData>(DataType::DT_INT32, shape);
    for (int i = 0; i < 5; i++) {
        rawData->Get<int32_t>(i) = i * 100;
    }
    auto tensorData = std::make_shared<LogicalTensorData>(rawData);
    std::vector<std::shared_ptr<LogicalTensorData>> incastList = {tensorData};
    std::vector<std::shared_ptr<LogicalTensorData>> outcastList;
    auto inoutDataPair = std::make_shared<FunctionIODataPair>(incastList, outcastList);
    evaluator.UpdateIODataPair(inoutDataPair);

    std::vector<ScalarImmediateType> dataList = {0, 0, 0, 2};
    auto ret = evaluator.EvaluateSymbolicCall("RUNTIME_GetTensorDataInt32Dim1", dataList, {});
    EXPECT_EQ(ret, 0);
}

TEST_F(TestSymbolicScalarEvaluate, EvaluateSymbolicCallRuntimeGetTensorDataInt32Dim2)
{
    EvaluateSymbol evaluator;
    std::vector<int64_t> shape = {3, 4};
    auto rawData = std::make_shared<RawTensorData>(DataType::DT_INT32, shape);
    for (int i = 0; i < 12; i++) {
        rawData->Get<int32_t>(i) = i * 50;
    }
    auto tensorData = std::make_shared<LogicalTensorData>(rawData);
    std::vector<std::shared_ptr<LogicalTensorData>> incastList;
    std::vector<std::shared_ptr<LogicalTensorData>> outcastList = {tensorData};
    auto inoutDataPair = std::make_shared<FunctionIODataPair>(incastList, outcastList);
    evaluator.UpdateIODataPair(inoutDataPair);

    std::vector<ScalarImmediateType> dataList = {1, 1, 0, 1, 2};
    auto ret = evaluator.EvaluateSymbolicCall("RUNTIME_GetTensorDataInt32Dim2", dataList, {});
    EXPECT_EQ(ret, 0);
}

TEST_F(TestSymbolicScalarEvaluate, EvaluateSymbolicCallRuntimeGetTensorDataInt32Dim3)
{
    EvaluateSymbol evaluator;
    std::vector<int64_t> shape = {2, 2, 2};
    auto rawData = std::make_shared<RawTensorData>(DataType::DT_INT32, shape);
    for (int i = 0; i < 8; i++) {
        rawData->Get<int32_t>(i) = i * 20;
    }
    auto tensorData = std::make_shared<LogicalTensorData>(rawData);
    std::vector<std::shared_ptr<LogicalTensorData>> incastList = {tensorData};
    std::vector<std::shared_ptr<LogicalTensorData>> outcastList;
    auto inoutDataPair = std::make_shared<FunctionIODataPair>(incastList, outcastList);
    evaluator.UpdateIODataPair(inoutDataPair);

    std::vector<ScalarImmediateType> dataList = {0, 0, 0, 1, 0, 1};
    auto ret = evaluator.EvaluateSymbolicCall("RUNTIME_GetTensorDataInt32Dim3", dataList, {});
    EXPECT_EQ(ret, 0);
}

TEST_F(TestSymbolicScalarEvaluate, EvaluateSymbolicCallRuntimeCoaGetValidShape)
{
    EvaluateSymbol evaluator;
    std::vector<ScalarImmediateType> dataList = {0, 1, 2};
    SymbolicScalar ss1(10);
    SymbolicScalar ss2(20);
    SymbolicScalar ss3(30);
    SymbolicScalar ss4(40);
    SymbolicScalar ss5(50);
    SymbolicScalar ss6(60);
    std::vector<SymbolicScalar> linearArgList = {ss1, ss2, ss3, ss4, ss5, ss6};
    auto ret = evaluator.EvaluateSymbolicCall("RUNTIME_COA_GET_PARAM_VALID_SHAPE", dataList, linearArgList);
    EXPECT_EQ(ret, 50);
}

TEST_F(TestSymbolicScalarEvaluate, EvaluateSymbolicCallRuntimeCoaGetOffset)
{
    EvaluateSymbol evaluator;
    std::vector<ScalarImmediateType> dataList = {0, 1, 1};
    SymbolicScalar ss1(100);
    SymbolicScalar ss2(200);
    SymbolicScalar ss3(300);
    SymbolicScalar ss4(400);
    std::vector<SymbolicScalar> linearArgList = {ss1, ss2, ss3, ss4};
    auto ret = evaluator.EvaluateSymbolicCall("RUNTIME_COA_GET_PARAM_OFFSET", dataList, linearArgList);
    EXPECT_EQ(ret, 400);
}

TEST_F(TestSymbolicScalarEvaluate, EvaluateSymbolicCallRuntimeCoaGetParam)
{
    EvaluateSymbol evaluator;
    std::vector<ScalarImmediateType> dataList = {37};
    std::vector<SymbolicScalar> linearArgList(38, SymbolicScalar(0));
    linearArgList[37] = SymbolicScalar(123);
    auto ret = evaluator.EvaluateSymbolicCall("RUNTIME_COA_GET_PARAM", dataList, linearArgList);
    EXPECT_EQ(ret, 123);
}

TEST_F(TestSymbolicScalarEvaluate, EvaluateSymbolicCallRuntimeCoaGetRawShape)
{
    EvaluateSymbol evaluator;
    // RUNTIME_COA_GET_PARAM_RAW_SHAPE(2, 1, idx) -> coaIndex = 1 + 1 + 2*2 + idx = 6 + idx
    std::vector<ScalarImmediateType> dataList = {2, 1, 0};
    std::vector<SymbolicScalar> linearArgList(8, SymbolicScalar(0));
    linearArgList[6] = SymbolicScalar(77);
    linearArgList[7] = SymbolicScalar(88);
    auto ret = evaluator.EvaluateSymbolicCall("RUNTIME_COA_GET_PARAM_RAW_SHAPE", dataList, linearArgList);
    EXPECT_EQ(ret, 77);
    dataList = {2, 1, 1};
    ret = evaluator.EvaluateSymbolicCall("RUNTIME_COA_GET_PARAM_RAW_SHAPE", dataList, linearArgList);
    EXPECT_EQ(ret, 88);
}

TEST_F(TestSymbolicScalarEvaluate, EvaluateSymbolicCallInvalid)
{
    EvaluateSymbol evaluator;
    std::vector<ScalarImmediateType> dataList = {1, 2};
    EXPECT_THROW(evaluator.EvaluateSymbolicCall("INVALID_CALL", dataList, {}), std::exception);
}

TEST_F(TestSymbolicScalarEvaluate, EvaluateSymbolicScalarImmediate)
{
    EvaluateSymbol evaluator;
    SymbolicScalar ss(42);
    auto ret = evaluator.EvaluateSymbolicScalar(ss);
    EXPECT_EQ(ret, 42);
}

TEST_F(TestSymbolicScalarEvaluate, EvaluateSymbolicScalarSymbol)
{
    EvaluateSymbol evaluator;
    SymbolicScalar ss("test_symbol");
    evaluator.UpdateSymbolDict("test_symbol", 99);
    auto ret = evaluator.EvaluateSymbolicScalar(ss);
    EXPECT_EQ(ret, 99);
}

TEST_F(TestSymbolicScalarEvaluate, EvaluateSymbolicScalarExpressionUnary)
{
    EvaluateSymbol evaluator;
    SymbolicScalar ss(10);
    SymbolicScalar neg = ss.Neg();
    auto ret = evaluator.EvaluateSymbolicScalar(neg);
    EXPECT_EQ(ret, -10);
}

TEST_F(TestSymbolicScalarEvaluate, EvaluateSymbolicScalarExpressionBinary)
{
    EvaluateSymbol evaluator;
    SymbolicScalar ss1(5);
    SymbolicScalar ss2(3);
    SymbolicScalar add = ss1.Add(ss2);
    auto ret = evaluator.EvaluateSymbolicScalar(add);
    EXPECT_EQ(ret, 8);
}

TEST_F(TestSymbolicScalarEvaluate, EvaluateSymbolicScalarExpressionMultipleMax)
{
    EvaluateSymbol evaluator;
    SymbolicScalar ss1(10);
    SymbolicScalar ss2(20);
    SymbolicScalar ss3(15);
    SymbolicScalar max = std::max({ss1, ss2, ss3});
    auto ret = evaluator.EvaluateSymbolicScalar(max);
    EXPECT_EQ(ret, 20);
}

TEST_F(TestSymbolicScalarEvaluate, EvaluateSymbolicScalarExpressionMultipleMin)
{
    EvaluateSymbol evaluator;
    SymbolicScalar ss1(10);
    SymbolicScalar ss2(20);
    SymbolicScalar ss3(15);
    SymbolicScalar min = std::min({ss1, ss2, ss3});
    auto ret = evaluator.EvaluateSymbolicScalar(min);
    EXPECT_EQ(ret, 10);
}

} // namespace npu::tile_fwk
