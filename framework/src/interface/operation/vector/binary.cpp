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
 * \file binary.cpp
 * \brief
 */

#include "tilefwk/data_type.h"
#include "unary.h"
#include "binary.h"
#include "binary_tiled.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/error_code.h"
#include "passes/tile_graph_pass/graph_constraint/axis_combine.h"
namespace npu::tile_fwk {
std::vector<int64_t> BinaryOperationResultShape(LogicalTensorPtr operand1, LogicalTensorPtr operand2)
{
    std::vector<int64_t> resultShape(operand1->shape.size());
    for (size_t i = 0; i < resultShape.size(); i++) {
        resultShape[i] = std::max(operand1->shape[i], operand2->shape[i]);
    }
    return resultShape;
}

LogicalTensorPtr BinaryOperationBroadCast(const LogicalTensorPtr& operand, const std::vector<int>& broadCastShape)
{
    if (operand->shape.size() < broadCastShape.size()) {
        auto broadCastDims = broadCastShape.size() - operand->shape.size();
        std::vector<int64_t> unsqueezeShape(operand->shape);
        unsqueezeShape.insert(unsqueezeShape.begin(), broadCastDims, 1);
        auto tmpOperand = Reshape(operand, unsqueezeShape).GetStorage();
        return tmpOperand;
    }
    return operand;
}

void CheckOperandsValid(const LogicalTensorPtr& operand1, const LogicalTensorPtr& operand2)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, operand1->shape.size() == operand2->shape.size())
        << "The shape size of the two input tensors must be equal";
}

void CheckBinOpOperandsValid(const LogicalTensorPtr& operand1, const LogicalTensorPtr& operand2)
{
    CheckOperandsValid(operand1, operand2);
    for (size_t i = 0; i < operand1->shape.size(); ++i) {
        if (operand1->shape[i] != operand2->shape[i] && (operand1->shape[i] != 1 && operand2->shape[i] != 1)) {
            CHECK(VectorErrorCode::ERR_PARAM_INVALID, false) << "shape not support binary operation";
        }
    }
}

void BroadcastOperandTensor(LogicalTensorPtr& operand, LogicalTensorPtr& other, LogicalTensorPtr result,
                            Function& function, const TileShape& tileShape, std::vector<int64_t> dstShape)
{
    if (dstShape.empty()) {
        dstShape = result->shape;
    }
    if (operand->shape == dstShape) {
        return;
    }
    auto expanded = std::make_shared<LogicalTensor>(function, operand->Datatype(), dstShape);
    Expand(function, tileShape, operand, {other}, expanded);
    operand = expanded;
}

void BinaryOperationOperandCheck(const std::vector<LogicalTensorPtr>& iOperand,
                                 const std::vector<LogicalTensorPtr>& oOperand)
{
    constexpr size_t inOpSize = NUM_VALUE_2;
    constexpr size_t outOpSize = 1;
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, iOperand.size() == inOpSize) << "iOperand size should be 2";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, oOperand.size() == outOpSize) << "oOperand size should be 1";
}

// Identify which operand need brc at a specific axis counting from the first
// Return value 0 = NONE, 1 = LEFT_OPERAND, 2 = RIGHT_OPERAND
int BrcAxisBinaryOp(LogicalTensorPtr operand1, LogicalTensorPtr operand2, int64_t axisNum)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, operand1->shape.size() == operand2->shape.size()) << "Dims not match";
    int64_t shapeSize = operand1->shape.size();
    int operandNum = 0;

    int64_t idx = (axisNum < 0) ? (shapeSize + axisNum) : axisNum;
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, idx >= 0 && idx < shapeSize)
        << "axisNum " << axisNum << " out of range for shapeSize " << shapeSize;
    if ((operand1->shape[idx] != 1) && (operand2->shape[idx] == 1)) {
        operandNum = NUM_VALUE_2;
    } else if ((operand1->shape[idx] == 1) && (operand2->shape[idx] != 1)) {
        operandNum = 1;
    }
    return operandNum;
}

Tensor Add(const Tensor& self, const Tensor& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Add");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Add");

    CheckTensorsDataTypeConsistency(self.GetStorage(), other.GetStorage(), "ADD");

    const auto& supportedTypes = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_ADD);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "ADD");
    CheckInt64Broadcast(self.GetStorage(), other.GetStorage(), "ADD");
    RETURN_CALL(BinaryOperation<BinaryOpType::ADD>, *Program::GetInstance().GetCurrentFunction(), self, other);
}

Tensor Sub(const Tensor& self, const Tensor& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Sub");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Sub");

    CheckTensorsDataTypeConsistency(self.GetStorage(), other.GetStorage(), "SUB");

    const auto& supportedTypes = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_SUB);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "SUB");
    CheckInt64Broadcast(self.GetStorage(), other.GetStorage(), "SUB");
    RETURN_CALL(BinaryOperation<BinaryOpType::SUB>, *Program::GetInstance().GetCurrentFunction(), self, other);
}

Tensor Mul(const Tensor& self, const Tensor& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Mul");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Mul");

    CheckTensorsDataTypeConsistency(self.GetStorage(), other.GetStorage(), "MUL");

    const auto& supportedTypes = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_MUL);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "MUL");
    CheckInt64Broadcast(self.GetStorage(), other.GetStorage(), "MUL");
    RETURN_CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), self, other);
}

Tensor Div(const Tensor& self, const Tensor& other, PrecisionType precisionType)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Div");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Div");

    CheckTensorsDataTypeConsistency(self.GetStorage(), other.GetStorage(), "DIV");

    const auto& supportedTypes = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_DIV);
    auto isDivSupportedInt = [](DataType dt) { return dt == DT_INT16 || dt == DT_INT32 || dt == DT_INT64; };
    CheckTensorDataType(self.GetStorage(), supportedTypes, "DIV");

    if (isDivSupportedInt(self.GetDataType())) {
        Tensor castSelf = Cast(self, DT_FP32, CastMode::CAST_NONE, SaturationMode::ON);
        Tensor castOther = Cast(other, DT_FP32, CastMode::CAST_NONE, SaturationMode::ON);
        auto [castResult, op] = TensorBinaryOperationWithOp<BinaryOpType::DIV>(
            *Program::GetInstance().GetCurrentFunction(), castSelf, castOther);
        op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
        return Tensor(castResult);
    }

    auto [result, op] = TensorBinaryOperationWithOp<BinaryOpType::DIV>(*Program::GetInstance().GetCurrentFunction(),
                                                                       self, other);
    op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    return Tensor(result);
}

Tensor Fmod(const Tensor& self, const Tensor& other, PrecisionType precisionType)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Fmod");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Fmod");

    CheckTensorsDataTypeConsistency(self.GetStorage(), other.GetStorage(), "MOD");
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "MOD");
    auto selfDtype = self.GetDataType();
    if (selfDtype == DT_FP16) {
        Tensor castSelf = Cast(self, DataType::DT_FP32, CastMode::CAST_NONE);
        Tensor castOther = Cast(other, DataType::DT_FP32, CastMode::CAST_NONE);
        auto [castResult, op] = TensorBinaryOperationWithOp<BinaryOpType::MOD>(
            *Program::GetInstance().GetCurrentFunction(), castSelf, castOther);
        op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
        return Cast(Tensor(castResult), selfDtype, CastMode::CAST_NONE);
    }
    auto [result, op] = TensorBinaryOperationWithOp<BinaryOpType::MOD>(*Program::GetInstance().GetCurrentFunction(),
                                                                       self, other);
    op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    return Tensor(result);
}

Tensor Remainder(const Tensor& self, const Tensor& other, PrecisionType precisionType)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Remainder");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Remainder");

    CheckTensorsDataTypeConsistency(self.GetStorage(), other.GetStorage(), "REM");
    const auto& supportedTypes = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_REM);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "REM");
    CheckInt64Broadcast(self.GetStorage(), other.GetStorage(), "REM");
    auto selfDtype = self.GetDataType();
    bool isA5Architecture = (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510);
    if ((!isA5Architecture && selfDtype == DT_INT16) || selfDtype == DT_FP16) {
        Tensor castSelf = Cast(self, DT_FP32, CastMode::CAST_NONE, SaturationMode::ON);
        Tensor castOther = Cast(other, DT_FP32, CastMode::CAST_NONE, SaturationMode::ON);
        auto [result, op] = TensorBinaryOperationWithOp<BinaryOpType::REM>(*Program::GetInstance().GetCurrentFunction(),
                                                                           castSelf, castOther);
        op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
        Tensor castedResult = Cast(Tensor(result), selfDtype, CastMode::CAST_NONE, SaturationMode::ON);
        return castedResult;
    }
    auto [result, op] = TensorBinaryOperationWithOp<BinaryOpType::REM>(*Program::GetInstance().GetCurrentFunction(),
                                                                       self, other);
    op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    return Tensor(result);
}

Tensor Maximum(const Tensor& operand1, const Tensor& operand2)
{
    DECLARE_TRACER();
    CheckTensorFormat(operand1.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Maximum");
    CheckTensorFormat(operand2.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Maximum");

    CheckTensorsDataTypeConsistency(operand1.GetStorage(), operand2.GetStorage(), "MAXIMUM");

    const auto& supportedTypes = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_MAXIMUM);
    CheckTensorDataType(operand1.GetStorage(), supportedTypes, "MAXIMUM");
    CheckInt64Broadcast(operand1.GetStorage(), operand2.GetStorage(), "MAXIMUM");
    RETURN_CALL(BinaryOperation<BinaryOpType::MAXIMUM>, *Program::GetInstance().GetCurrentFunction(), operand1,
                operand2);
}

Tensor Minimum(const Tensor& operand1, const Tensor& operand2)
{
    DECLARE_TRACER();
    CheckTensorFormat(operand1.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Minimum");
    CheckTensorFormat(operand2.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Minimum");

    CheckTensorsDataTypeConsistency(operand1.GetStorage(), operand2.GetStorage(), "MINIMUM");

    const auto& supportedTypes = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_MINIMUM);
    CheckTensorDataType(operand1.GetStorage(), supportedTypes, "MINIMUM");
    CheckInt64Broadcast(operand1.GetStorage(), operand2.GetStorage(), "MINIMUM");
    RETURN_CALL(BinaryOperation<BinaryOpType::MINIMUM>, *Program::GetInstance().GetCurrentFunction(), operand1,
                operand2);
}

Tensor Clip(const Tensor& self, const Element& min, const Element& max)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Clip");

    static const std::unordered_set<DataType> CLIP_A2A3_TYPES = {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16};
    static const std::unordered_set<DataType> CLIP_A5_TYPES = {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16, DT_INT64};
    const auto& supportedTypes = (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) ? CLIP_A5_TYPES :
                                                                                                     CLIP_A2A3_TYPES;
    CheckTensorDataType(self.GetStorage(), supportedTypes, "CLIP");
    CheckTensorShapeSize(self.GetStorage(), "CLIP");

    Element min_ = min, max_ = max;

    Tensor result = self;
    if (min_.GetDataType() != DT_BOTTOM) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, min_.GetDataType() == self.GetDataType())
            << "The datatype of inputs should be same";
        result = Maximum(result, min_);
    }
    if (max_.GetDataType() != DT_BOTTOM) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, max_.GetDataType() == self.GetDataType())
            << "The datatype of inputs should be same";
        result = Minimum(result, max_);
    }
    result.GetStorage()->UpdateDynValidShape(self.GetStorage()->GetDynValidShape());
    return result;
}

Tensor Clip(const Tensor& self, const Tensor& min, const Tensor& max)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Clip");

    static const std::unordered_set<DataType> CLIP_A2A3_TYPES = {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16};
    static const std::unordered_set<DataType> CLIP_A5_TYPES = {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16, DT_INT64};
    const auto& supportedTypes = (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) ? CLIP_A5_TYPES :
                                                                                                     CLIP_A2A3_TYPES;
    CheckTensorDataType(self.GetStorage(), supportedTypes, "CLIP");
    CheckTensorShapeSize(self.GetStorage(), "CLIP");

    Tensor result = self;
    if (min.GetStorage() != nullptr) {
        CheckTensorFormat(min.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Clip");
        CheckTensorShapeSize(min.GetStorage(), "CLIP");
        CheckTensorsFormatConsistency(self.GetStorage(), min.GetStorage(), "CLIP");
        std::vector minBroadcastAxes = GetBroadcastAxes(min.GetShape(), self.GetShape());
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, minBroadcastAxes.size() <= 1)
            << "minBroadcastAxes size should be <= 1";
        CheckInt64Broadcast(self.GetStorage(), min.GetStorage(), "CLIP");
        result = Maximum(result, min);
    }
    if (max.GetStorage() != nullptr) {
        CheckTensorFormat(max.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Clip");
        CheckTensorShapeSize(max.GetStorage(), "CLIP");
        CheckTensorsFormatConsistency(self.GetStorage(), max.GetStorage(), "CLIP");
        std::vector maxBroadcastAxes = GetBroadcastAxes(max.GetShape(), self.GetShape());
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, maxBroadcastAxes.size() <= 1)
            << "maxBroadcastAxes size should be <= 1";
        CheckInt64Broadcast(self.GetStorage(), max.GetStorage(), "CLIP");
        result = Minimum(result, max);
    }
    result.GetStorage()->UpdateDynValidShape(self.GetStorage()->GetDynValidShape());
    return result;
}

Tensor Gcd(const Tensor& self, const Tensor& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Gcd");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Gcd");

    CheckTensorsDataTypeConsistency(self.GetStorage(), other.GetStorage(), "GCD");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "GCD");
    std::unordered_set<DataType> supportedTypes = {DT_INT8, DT_INT16, DT_INT32, DT_UINT8};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "GCD");
    return TensorBinaryOperation<BinaryOpType::GCD>(*Program::GetInstance().GetCurrentFunction(), self, other);
}

Tensor FloorDiv(const Tensor& self, const Tensor& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "FLOORDIV");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "FLOORDIV");

    const auto& supportedTypes = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_FLOORDIV);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "FLOORDIV");
    CheckTensorDataType(other.GetStorage(), supportedTypes, "FLOORDIV");

    CheckTensorsDataTypeConsistency(self.GetStorage(), other.GetStorage(), "FLOORDIV");
    RETURN_CALL(BinaryOperation<BinaryOpType::FLOORDIV>, *Program::GetInstance().GetCurrentFunction(), self, other);
}

Tensor CeilDiv(const Tensor& self, const Tensor& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "CeilDiv");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "CeilDiv");

    CheckTensorsDataTypeConsistency(self.GetStorage(), other.GetStorage(), "CEILDIV");
    std::unordered_set<DataType> supportedTypes = {DT_INT32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "CEILDIV");

    Tensor selfFp32 = Cast(self, DataType::DT_FP32);
    Tensor otherFp32 = Cast(other, DataType::DT_FP32);
    Tensor resultFp32 = Div(selfFp32, otherFp32);
    resultFp32 = Ceil(resultFp32);
    Tensor result = Cast(resultFp32, DT_INT32);
    Tensor target = Mul(result, other);
    Tensor diff = Sub(self, target);
    Tensor sgn_diff = Sign(diff);
    Tensor sgn_other = Sign(other);
    Tensor product_sgn = Mul(sgn_diff, sgn_other);
    Tensor inc = Clip(product_sgn, Element(DT_INT32, 0), Element(DT_INT32, 1));
    result = Add(result, inc);
    return result;
}

// OP_ADD OP_SUB OP_MUL OP_DIV OP_MAX OP_BITWISEAND OP_BITWISEOR OP_BITWISEXOR
int64_t GetOperationPrecisionType(const Operation& op)
{
    if (op.HasAttr(OpAttributeKey::precisionType)) {
        return op.GetIntAttribute(OpAttributeKey::precisionType);
    }
    return static_cast<int64_t>(PrecisionType::INTRINSIC);
}

REGISTER_OPERATION_TILED_FUNC(OP_ADD, Opcode::OP_ADD, BinaryOperationTileFunc<BinaryOpType::ADD>);
REGISTER_OPERATION_TILED_FUNC(OP_SUB, Opcode::OP_SUB, BinaryOperationTileFunc<BinaryOpType::SUB>);
REGISTER_OPERATION_TILED_FUNC(OP_MUL, Opcode::OP_MUL, BinaryOperationTileFunc<BinaryOpType::MUL>);
REGISTER_OPERATION_TILED_FUNC(OP_DIV, Opcode::OP_DIV, BinaryOperationTileFunc<BinaryOpType::DIV>);
REGISTER_OPERATION_TILED_FUNC(OP_MAXIMUM, Opcode::OP_MAXIMUM, BinaryOperationTileFunc<BinaryOpType::MAXIMUM>);
REGISTER_OPERATION_TILED_FUNC(OP_MINIMUM, Opcode::OP_MINIMUM, BinaryOperationTileFunc<BinaryOpType::MINIMUM>);
REGISTER_OPERATION_TILED_FUNC(OP_MOD, Opcode::OP_MOD, BinaryOperationTileFunc<BinaryOpType::MOD>);
REGISTER_OPERATION_TILED_FUNC(OP_REM, Opcode::OP_REM, BinaryOperationTileFunc<BinaryOpType::REM>);
REGISTER_OPERATION_TILED_FUNC(OP_GCD, Opcode::OP_GCD, BinaryOperationTileFunc<BinaryOpType::GCD>);
REGISTER_OPERATION_TILED_FUNC(OP_FLOORDIV, Opcode::OP_FLOORDIV, BinaryOperationTileFunc<BinaryOpType::FLOORDIV>);

} // namespace npu::tile_fwk
