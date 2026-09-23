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
 * \file symbolic_scalar_evaluate.cpp
 * \brief
 */

#include "interface/tensor/symbolic_scalar_evaluate.h"
#include "interface/interpreter/function.h"
#include "interface/operation/attribute.h"
#include "tensor/symbolic_scalar.h"

#include <string>

namespace npu::tile_fwk {
namespace {
ScalarImmediateType EvaluateSymbolicCallRuntimeGetInputShapeDimSize(EvaluateSymbol* evaluateSymbol,
                                                                    const std::vector<ScalarImmediateType>& dataList)
{
    FE_ASSERT(dataList.size() == 1);
    auto inputIndex = dataList[0];
    auto input = evaluateSymbol->GetInputDataViewList()[inputIndex];

    auto ret = input->GetShape().size();
    return ret;
}

ScalarImmediateType EvaluateSymbolicCallRuntimeGetInputShapeDim(EvaluateSymbol* evaluateSymbol,
                                                                const std::vector<ScalarImmediateType>& dataList)
{
    FE_ASSERT(dataList.size() == SIZE_TWO);
    auto inputIndex = dataList[0];
    auto input = evaluateSymbol->GetInputDataViewList()[inputIndex];
    auto n = dataList[1];

    auto ret = input->GetShape()[n];
    return ret;
}

ScalarImmediateType EvaluateSymbolicCallRuntimeGetInputData(EvaluateSymbol* evaluateSymbol,
                                                            const std::vector<ScalarImmediateType>& dataList)
{
    auto inputIndex = dataList[0];
    auto input = evaluateSymbol->GetInputDataViewList()[inputIndex];
    int64_t offset = 0;
    for (size_t i = 0; i < input->GetStride().size(); i++) {
        offset += dataList[i + 2] * input->GetStride(i); // 2 skip inputIndex dtype
    }
    auto elt = input->GetElement(offset);
    auto ret = static_cast<ScalarImmediateType>(elt.Cast<int64_t>());
    return ret;
}

ScalarImmediateType EvaluateSymbolicCallRuntimeGetTensorDataInt32(EvaluateSymbol* evaluateSymbol, int ioType,
                                                                  int ioTypeIndex,
                                                                  const std::vector<ScalarImmediateType>& offsetList)
{
    UNUSED(offsetList);
    auto inoutDataPair = evaluateSymbol->GetInoutDataPair();
    FE_ASSERT(inoutDataPair != nullptr) << "GetTensorData eval failed: current frame has no FunctionIODataPair, ioType="
                                        << ioType << ", ioTypeIndex=" << ioTypeIndex;
    FE_ASSERT(ioType == GET_TENSOR_DATA_OPERAND_IOTYPE_INCAST || ioType == GET_TENSOR_DATA_OPERAND_IOTYPE_OUTCAST)
        << "GetTensorData eval failed: invalid ioType=" << ioType << ", ioTypeIndex=" << ioTypeIndex;
    const auto& viewList = (ioType == GET_TENSOR_DATA_OPERAND_IOTYPE_INCAST) ? inoutDataPair->GetIncastDataViewList() :
                                                                               inoutDataPair->GetOutcastDataViewList();
    FE_ASSERT(ioTypeIndex >= 0 && static_cast<size_t>(ioTypeIndex) < viewList.size())
        << "GetTensorData eval failed: " << (ioType == GET_TENSOR_DATA_OPERAND_IOTYPE_INCAST ? "incast" : "outcast")
        << " index " << ioTypeIndex << " out of range (list size=" << viewList.size()
        << "); the (ioType, ioTypeIndex) of RUNTIME_GetTensorDataInt32 symbol is not "
           "remapped after graph transform (op moved to another function)";
    std::shared_ptr<LogicalTensorData> view = viewList[ioTypeIndex];
    FE_ASSERT(view != nullptr) << "GetTensorData eval failed: "
                               << (ioType == GET_TENSOR_DATA_OPERAND_IOTYPE_INCAST ? "incast" : "outcast") << " index "
                               << ioTypeIndex << " resolved a null data view";
    auto elt = view->GetElement(0);
    auto ret = static_cast<ScalarImmediateType>(elt.Cast<int64_t>());
    return ret;
}

ScalarImmediateType EvaluateSymbolicCallRuntimeGetTensorDataInt32Dim1(EvaluateSymbol* evaluateSymbol,
                                                                      const std::vector<ScalarImmediateType>& dataList)
{
    auto ioType = dataList[GET_TENSOR_DATA_OPERAND_INDEX_IOTYPE - 1];
    auto ioTypeIndex = dataList[GET_TENSOR_DATA_OPERAND_INDEX_IOTYPE_INDEX - 1];
    std::vector<ScalarImmediateType> offsetList(dataList.begin() + GET_TENSOR_DATA_OPERAND_INDEX_ADDRESS - 1 + 1,
                                                dataList.end());

    auto ret = EvaluateSymbolicCallRuntimeGetTensorDataInt32(evaluateSymbol, ioType, ioTypeIndex, offsetList);
    return ret;
}

ScalarImmediateType EvaluateSymbolicCallRuntimeGetTensorDataInt32Dim2(EvaluateSymbol* evaluateSymbol,
                                                                      const std::vector<ScalarImmediateType>& dataList)
{
    auto ioType = dataList[GET_TENSOR_DATA_OPERAND_INDEX_IOTYPE - 1];
    auto ioTypeIndex = dataList[GET_TENSOR_DATA_OPERAND_INDEX_IOTYPE_INDEX - 1];
    std::vector<ScalarImmediateType> offsetList(dataList.begin() + GET_TENSOR_DATA_OPERAND_INDEX_ADDRESS - 1 + 1,
                                                dataList.end());

    auto ret = EvaluateSymbolicCallRuntimeGetTensorDataInt32(evaluateSymbol, ioType, ioTypeIndex, offsetList);
    return ret;
}

ScalarImmediateType EvaluateSymbolicCallRuntimeGetTensorDataInt32Dim3(EvaluateSymbol* evaluateSymbol,
                                                                      const std::vector<ScalarImmediateType>& dataList)
{
    auto ioType = dataList[GET_TENSOR_DATA_OPERAND_INDEX_IOTYPE - 1];
    auto ioTypeIndex = dataList[GET_TENSOR_DATA_OPERAND_INDEX_IOTYPE_INDEX - 1];
    std::vector<ScalarImmediateType> offsetList(dataList.begin() + GET_TENSOR_DATA_OPERAND_INDEX_ADDRESS - 1 + 1,
                                                dataList.end());

    auto ret = EvaluateSymbolicCallRuntimeGetTensorDataInt32(evaluateSymbol, ioType, ioTypeIndex, offsetList);
    return ret;
}

ScalarImmediateType EvaluateSymbolicCallGetHcclRankId(EvaluateSymbol* evaluateSymbol,
                                                      const std::vector<ScalarImmediateType>& dataList)
{
    int hcclIdx = static_cast<int>(dataList[0]);
    return evaluateSymbol->EvaluateRankId(hcclIdx);
}

ScalarImmediateType EvaluateSymbolicCallGetParaAddr(EvaluateSymbol*, const std::vector<ScalarImmediateType>&)
{
    // not used by getTensorData
    return 0;
}

ScalarImmediateType EvaluateSymbolicCallRuntimeIsLoopBegin(EvaluateSymbol* evaluateSymbol,
                                                           const std::vector<ScalarImmediateType>& dataList)
{
    FE_ASSERT(dataList.size() == SIZE_TWO);
    auto ret = evaluateSymbol->RuntimeIsLoopBegin(dataList[0], dataList[1]);
    return ret;
}

ScalarImmediateType EvaluateSymbolicCallRuntimeIsLoopEnd(EvaluateSymbol* evaluateSymbol,
                                                         const std::vector<ScalarImmediateType>& dataList)
{
    FE_ASSERT(dataList.size() == SIZE_TWO);
    auto ret = evaluateSymbol->RuntimeIsLoopEnd(dataList[0], dataList[1]);
    return ret;
}

ScalarImmediateType EvaluateSymbolicCallRuntimeTernaryOP(EvaluateSymbol* evaluateSymbol,
                                                         const std::vector<ScalarImmediateType>& dataList)
{
    UNUSED(evaluateSymbol);
    FE_ASSERT(dataList.size() == SIZE_THREE);
    // Matches RUNTIME_TernaryOP(cond, lhs, rhs) in aicpu_runtime.h: ((cond) ? (lhs) : (rhs))
    return dataList[0] != 0 ? dataList[1] : dataList[2];
}

ScalarImmediateType EvaluateSymbolicCallRuntimeGetViewValidShapeDim(EvaluateSymbol* evaluateSymbol,
                                                                    const std::vector<ScalarImmediateType>& dataList)
{
    UNUSED(evaluateSymbol);
    FE_ASSERT(dataList.size() == SIZE_THREE);
    auto validshape = dataList[0];
    auto viewOffset = dataList[1];
    auto viewshape = dataList[2];
    validshape -= viewOffset;
    if (validshape > viewshape)
        validshape = viewshape;
    else if (validshape < 0)
        validshape = 0;
    return validshape;
}

ScalarImmediateType EvaluateSymbolicCallRuntimeCoaGetValidShape(EvaluateSymbol* evaluateSymbol,
                                                                const std::vector<ScalarImmediateType>& dataList,
                                                                const std::vector<SymbolicScalar>& linearArgList)
{
    FE_ASSERT(linearArgList.size()) << "linearArgList is null";
    auto coaIndex = dataList[1] + COA_INDEX_DIM_BASE + dataList[0] * 3 + dataList[2];
    return evaluateSymbol->EvaluateSymbolicScalar(linearArgList[coaIndex]);
}

ScalarImmediateType EvaluateSymbolicCallRuntimeCoaGetOffset(EvaluateSymbol* evaluateSymbol,
                                                            const std::vector<ScalarImmediateType>& dataList,
                                                            const std::vector<SymbolicScalar>& linearArgList)
{
    FE_ASSERT(linearArgList.size()) << "linearArgList is null";

    auto coaIndex = dataList[1] + COA_INDEX_DIM_BASE + dataList[2];
    return evaluateSymbol->EvaluateSymbolicScalar(linearArgList[coaIndex]);
}

ScalarImmediateType EvaluateSymbolicCallRuntimeCoaGetRawShape(EvaluateSymbol* evaluateSymbol,
                                                              const std::vector<ScalarImmediateType>& dataList,
                                                              const std::vector<SymbolicScalar>& linearArgList)
{
    FE_ASSERT(linearArgList.size()) << "linearArgList is null";
    // Matches RUNTIME_COA_GET_PARAM_RAW_SHAPE(dim, base, idx) in aicore_runtime.h:
    // GetCoa(param, ((base) + 1) + 2 * (dim) + idx)
    auto coaIndex = dataList[1] + COA_INDEX_DIM_BASE + dataList[0] * 2 + dataList[2];
    return evaluateSymbol->EvaluateSymbolicScalar(linearArgList[coaIndex]);
}

ScalarImmediateType EvaluateSymbolicCallRuntimeCoaGetParam(EvaluateSymbol* evaluateSymbol,
                                                           const std::vector<ScalarImmediateType>& dataList,
                                                           const std::vector<SymbolicScalar>& linearArgList)
{
    FE_ASSERT(linearArgList.size()) << "linearArgList is null";
    FE_ASSERT(dataList.size() == 1);
    // Matches RUNTIME_COA_GET_PARAM(idx) in aicore_runtime.h: GetCoa<0, 0>(param, idx)
    auto coaIndex = dataList[0];
    return evaluateSymbol->EvaluateSymbolicScalar(linearArgList[coaIndex]);
}
} // namespace

ScalarImmediateType EvaluateSymbol::EvaluateSymbolicCall(const std::string& name,
                                                         const std::vector<ScalarImmediateType>& dataList,
                                                         const std::vector<SymbolicScalar>& linearArgList)
{
    using CallEntry = ScalarImmediateType (*)(EvaluateSymbol*, const std::vector<ScalarImmediateType>& dataList);
    static std::unordered_map<std::string, CallEntry> callEntryDict = {
        {"RUNTIME_GetInputShapeDimSize", EvaluateSymbolicCallRuntimeGetInputShapeDimSize},
        {"RUNTIME_GetInputShapeDim", EvaluateSymbolicCallRuntimeGetInputShapeDim},
        {"RUNTIME_GetInputData", EvaluateSymbolicCallRuntimeGetInputData},
        {"RUNTIME_IsLoopBegin", EvaluateSymbolicCallRuntimeIsLoopBegin},
        {"RUNTIME_IsLoopEnd", EvaluateSymbolicCallRuntimeIsLoopEnd},
        {"RUNTIME_TernaryOP", EvaluateSymbolicCallRuntimeTernaryOP},
        {"RUNTIME_GetViewValidShapeDim", EvaluateSymbolicCallRuntimeGetViewValidShapeDim},
        {"RUNTIME_GetTensorDataInt32Dim1", EvaluateSymbolicCallRuntimeGetTensorDataInt32Dim1},
        {"RUNTIME_GetTensorDataInt32Dim2", EvaluateSymbolicCallRuntimeGetTensorDataInt32Dim2},
        {"RUNTIME_GetTensorDataInt32Dim3", EvaluateSymbolicCallRuntimeGetTensorDataInt32Dim3},
        {"RUNTIME_COA_GET_PARAM_ADDR", EvaluateSymbolicCallGetParaAddr},
        {"RUNTIME_GetHcclRankId", EvaluateSymbolicCallGetHcclRankId},
    };
    using CallWithLinerArgsEntry = ScalarImmediateType (*)(EvaluateSymbol*,
                                                           const std::vector<ScalarImmediateType>& dataList,
                                                           const std::vector<SymbolicScalar>& linearArgList);
    static std::unordered_map<std::string, CallWithLinerArgsEntry> CallWithLinerArgsEntryDict = {
        {"RUNTIME_COA_GET_PARAM", EvaluateSymbolicCallRuntimeCoaGetParam},
        {"RUNTIME_COA_GET_PARAM_VALID_SHAPE", EvaluateSymbolicCallRuntimeCoaGetValidShape},
        {"RUNTIME_COA_GET_PARAM_OFFSET", EvaluateSymbolicCallRuntimeCoaGetOffset},
        {"RUNTIME_COA_GET_PARAM_RAW_SHAPE", EvaluateSymbolicCallRuntimeCoaGetRawShape},
    };
    ScalarImmediateType ret{0};
    if (callEntryDict.count(name)) {
        auto callEntry = callEntryDict[name];
        ret = callEntry(this, dataList);
    } else if (CallWithLinerArgsEntryDict.count(name)) {
        auto callEntry = CallWithLinerArgsEntryDict[name];
        ret = callEntry(this, dataList, linearArgList);
    } else {
        FE_ASSERT(false) << "Symbolic call not found: " << name;
    }
    return ret;
}

static std::map<std::string, long> constSymbolDict_ = {
    {"RUNTIME_int8_t", DataType::DT_INT8},     {"RUNTIME_int16_t", DataType::DT_INT16},
    {"RUNTIME_int32_t", DataType::DT_INT32},   {"RUNTIME_int64_t", DataType::DT_INT64},
    {"RUNTIME_uint8_t", DataType::DT_UINT8},   {"RUNTIME_uint16_t", DataType::DT_UINT16},
    {"RUNTIME_uint32_t", DataType::DT_UINT32}, {"RUNTIME_uint64_t", DataType::DT_UINT64},
    {"RUNTIME_bool_t", DataType::DT_BOOL},
};

ScalarImmediateType EvaluateSymbol::EvaluateSymbolicScalar(const RawSymbolicScalarPtr& ss,
                                                           const std::vector<SymbolicScalar>& linearArgList)
{
    ScalarImmediateType result{0};
    switch (ss->Kind()) {
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE: {
            std::shared_ptr<RawSymbolicImmediate> imm = std::static_pointer_cast<RawSymbolicImmediate>(ss);
            result = imm->Immediate();
        } break;
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL: {
            std::shared_ptr<RawSymbolicSymbol> sym = std::static_pointer_cast<RawSymbolicSymbol>(ss);
            if (constSymbolDict_.count(sym->Name())) {
                result = constSymbolDict_[sym->Name()];
            } else {
                FE_ASSERT(symbolDict_.count(sym->Name()));
                result = symbolDict_.at(sym->Name());
            }
        } break;
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION: {
            std::shared_ptr<RawSymbolicExpression> expr = std::static_pointer_cast<RawSymbolicExpression>(ss);
            if (expr->Opcode() == SymbolicOpcode::T_MOP_CALL) {
                std::vector<ScalarImmediateType> dataList;
                for (size_t i = 1; i < expr->OperandList().size(); i++) {
                    dataList.emplace_back(EvaluateSymbolicScalar(expr->OperandList()[i], linearArgList));
                }
                std::string name = std::static_pointer_cast<RawSymbolicSymbol>(expr->OperandList()[0])->Name();
                result = EvaluateSymbolicCall(name, dataList, linearArgList);
            } else {
                std::vector<ScalarImmediateType> dataList;
                for (size_t i = 0; i < expr->OperandList().size(); i++) {
                    dataList.emplace_back(EvaluateSymbolicScalar(expr->OperandList()[i], linearArgList));
                }
                if (SymbolicOpcode::T_UOP_BEGIN <= expr->Opcode() && expr->Opcode() < SymbolicOpcode::T_UOP_END) {
                    result = RawSymbolicExpression::GetSymbolicCalcUnary(expr->Opcode())(dataList[0]);
                } else if (RawSymbolicExpression::IsBinaryCalcOpcode(expr->Opcode())) {
                    result = dataList[0];
                    for (size_t i = 1; i < dataList.size(); i++) {
                        result = RawSymbolicExpression::GetSymbolicCalcBinary(expr->Opcode())(result, dataList[i]);
                    }
                } else if (expr->Opcode() == SymbolicOpcode::T_MOP_MAX || expr->Opcode() == SymbolicOpcode::T_MOP_MIN) {
                    return RawSymbolicExpression::GetSymbolicCalcMultiple(expr->Opcode())(dataList);
                }
            }
        } break;
        default:
            FE_ASSERT(false);
            break;
    }
    return result;
}

} // namespace npu::tile_fwk
