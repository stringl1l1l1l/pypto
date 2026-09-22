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
 * \file binary_scalar.cpp
 * \brief Binary scalar operation implementations.
 */

#include <limits>
#include "tilefwk/data_type.h"
#include "unary.h"
#include "binary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {

static Element CastScalarToDtype(const Element& scalar, DataType dtype)
{
    if (dtype == DT_INT64) {
        return Element(dtype, scalar.Cast<int64_t>());
    }
    return Element(dtype, scalar.Cast<float>());
}

static std::pair<int64_t, int64_t> GetIntegerScalarRange(DataType dtype)
{
    switch (dtype) {
        case DT_INT8:
            return {std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()};
        case DT_INT16:
            return {std::numeric_limits<int16_t>::min(), std::numeric_limits<int16_t>::max()};
        case DT_INT32:
            return {std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()};
        case DT_UINT8:
            return {0LL, static_cast<int64_t>(std::numeric_limits<uint8_t>::max())};
        default:
            return {std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()};
    }
}

Tensor Gcd(const Tensor& self, const Element& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Gcd");

    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "GCD");
    std::unordered_set<DataType> supportedTypes = {DT_INT8, DT_INT16, DT_INT32, DT_UINT8};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "GCD");
    auto [elemMin, elemMax] = GetIntegerScalarRange(self.GetDataType());
    if (other.GetSignedData() < elemMin || other.GetSignedData() > elemMax) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, false)
            << "The value range for the element is incorrect! expected [" << elemMin << ", " << elemMax << "], got "
            << other.GetSignedData();
    }
    return TensorBinaryOperationScalar<BinaryOpType::GCD>(*Program::GetInstance().GetCurrentFunction(),
                                                          self.GetStorage(), other);
}

LogicalTensorPtr GenAllOneTensor(const Shape& shape, std::vector<SymbolicScalar> validShape, const DataType& dataType)
{
    auto result = CALL(FullOperation, *Program::GetInstance().GetCurrentFunction(), Element(DataType::DT_FP32, 1.0),
                       SymbolicScalar(), DataType::DT_FP32, shape, validShape);
    if (dataType != DataType::DT_FP32) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), result.GetStorage(),
                    dataType, CastMode::CAST_NONE);
    }
    return result.GetStorage();
}

LogicalTensorPtr PowSCalc(const LogicalTensorPtr& self, const Element& other, PrecisionType precisionType)
{
    double exponent = other.Cast<double>();
    if (std::abs(exponent - NUM_VALUE_0_5) < NUM_VALUE_EPS) { // sqrt(x)
        RETURN_CALL(UnaryOperation<UnaryOpType::SQRT>, *Program::GetInstance().GetCurrentFunction(), self);
    } else if (std::abs(exponent + NUM_VALUE_0_5) < NUM_VALUE_EPS) { // 1 / sqrt(x)
        auto sqrt = CALL(UnaryOperation<UnaryOpType::SQRT>, *Program::GetInstance().GetCurrentFunction(), self);
        auto ones = GenAllOneTensor(self->shape, self->GetDynValidShape(), DT_FP32);
        auto [result, op] = TensorBinaryOperationWithOp<BinaryOpType::DIV>(*Program::GetInstance().GetCurrentFunction(),
                                                                           ones, sqrt);
        op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(PrecisionType::HIGH_PRECISION));
        return result;
    } else if (std::abs(exponent - NUM_VALUE_3) < NUM_VALUE_EPS) { // x * x * x
        auto mul = CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), self, self);
        RETURN_CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), mul, self);
    } else if (std::abs(exponent - NUM_VALUE_2) < NUM_VALUE_EPS) { // x * x
        RETURN_CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), self, self);
    } else if (std::abs(exponent + NUM_VALUE_2) < NUM_VALUE_EPS) { // 1 / (x * x)
        auto mul = CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), self, self);
        auto ones = GenAllOneTensor(self->shape, self->GetDynValidShape(), DT_FP32);
        auto [result, op] = TensorBinaryOperationWithOp<BinaryOpType::DIV>(*Program::GetInstance().GetCurrentFunction(),
                                                                           ones, mul);
        op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(PrecisionType::HIGH_PRECISION));
        return result;
    } else if (std::abs(exponent + NUM_VALUE_1) < NUM_VALUE_EPS) { // 1 / x
        auto ones = GenAllOneTensor(self->shape, self->GetDynValidShape(), DT_FP32);
        auto [result, op] = TensorBinaryOperationWithOp<BinaryOpType::DIV>(*Program::GetInstance().GetCurrentFunction(),
                                                                           ones, self);
        op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(PrecisionType::HIGH_PRECISION));
        return result;
    } else if (self->Datatype() == DT_INT32) {
        auto [res, op] = TensorBinaryOperationScalarWithOp<BinaryOpType::POW>(
            *Program::GetInstance().GetCurrentFunction(), self, Element(DT_INT32, other.Cast<int>()));
        op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
        return res;
    } else if (self->Datatype() == DT_FP32) {
        auto otherTensor = CALL(FullOperation, *Program::GetInstance().GetCurrentFunction(),
                                Element(DataType::DT_FP32, exponent), SymbolicScalar(), DataType::DT_FP32, self->shape,
                                self->GetDynValidShape());
        auto [res, op] = TensorBinaryOperationWithOp<BinaryOpType::POW>(*Program::GetInstance().GetCurrentFunction(),
                                                                        self, otherTensor);
        op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
        return res;
    }
    return self;
}

void PowSCheck(const Tensor& self)
{
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT32, DT_FP32, DT_INT8, DT_UINT8, DT_INT16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "POW");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "POW");
    CheckTensorShapeSize(self.GetStorage(), "POW");
}

Tensor Pow(const Tensor& self, const Element& other, PrecisionType precisionType)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Pow");

    PowSCheck(self);
    LogicalTensorPtr castSelf = self.GetStorage();
    if (self.GetDataType() == DT_INT32 && !IsInteger(other.GetDataType())) {
        castSelf = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), castSelf,
                        DataType::DT_FP32, CastMode::CAST_NONE);
    }
    if (IsInteger(castSelf->Datatype())) {
        CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, IsInteger(other.GetDataType()))
            << "Scalar dtype incorrect. When self dtype in UINT32/INT16/UINT16/INT8/UINT8, "
            << "Scalar dtype should be int, actual is: " << DataType2String(other.GetDataType());
        auto [res, op] = TensorBinaryOperationScalarWithOp<BinaryOpType::POW>(
            *Program::GetInstance().GetCurrentFunction(), castSelf, Element(DT_INT32, other.Cast<int>()));
        op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(PrecisionType::INTRINSIC));
        return res;
    }
    if (std::abs(other.Cast<double>()) < NUM_VALUE_EPS) {
        return GenAllOneTensor(self.GetShape(), self.GetStorage()->GetDynValidShape(), self.GetDataType());
    }
    DataType dataType = castSelf->Datatype();
    bool shouldUpToFp32 = dataType == DT_FP16 || dataType == DT_BF16;
    if (shouldUpToFp32) {
        castSelf = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), castSelf,
                        DataType::DT_FP32, CastMode::CAST_NONE);
    }
    auto result = PowSCalc(castSelf, other, precisionType);
    if (shouldUpToFp32) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), result, dataType,
                    CastMode::CAST_NONE);
    }
    return result;
}

template <BinaryOpType T>
void TiledBinaryOperationScalar(Function& function, const TileShape& tileShape, size_t cur, LogicalInput& input1,
                                Element& value, const LogicalTensorPtr& result, TileInfo& resultTileInfo,
                                bool reverseOperand, int64_t precisionType)
{
    auto opNameCode = GetBinaryOpNameCode<T, true>();
    if (cur == input1.tensor->GetShape().size()) {
        auto inputTile1 = input1.tensor->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        if (opNameCode == Opcode::OP_BITWISEXORS || opNameCode == Opcode::OP_POWS) {
            std::vector<int64_t> tmpShape(resultTileInfo.shape);
            auto alignSize = BLOCK_SIZE / BytesOf(input1.tensor->Datatype());
            tmpShape[resultTileInfo.shape.size() - 1] = AlignUp(tmpShape[resultTileInfo.shape.size() - 1], alignSize);
            auto tempTensor = std::make_shared<LogicalTensor>(function, input1.tensor->Datatype(), tmpShape);
            auto& tmpOp = function.AddOperation(opNameCode, {inputTile1}, {resultTile, tempTensor});
            tmpOp.SetAttribute(OpAttributeKey::scalar, value);
            tmpOp.SetAttribute(OP_ATTR_PREFIX + "reverseOperand", reverseOperand);
            return;
        } else if (opNameCode == Opcode::OP_FLOORDIVS) {
            auto alignSize = BLOCK_SIZE / BytesOf(input1.tensor->Datatype());
            auto tmpShape = resultTileInfo.shape;
            tmpShape.back() = AlignUp(tmpShape.back(), alignSize);
            tmpShape.front() *= NUM_VALUE_6;
            auto tmpType = input1.tensor->Datatype() == DT_INT64 ? DT_INT64 : DT_FP32;
            auto tempTensor = std::make_shared<LogicalTensor>(function, tmpType, tmpShape);
            auto& tmpOp = function.AddOperation(opNameCode, {inputTile1}, {resultTile, tempTensor});
            tmpOp.SetAttribute(OpAttributeKey::scalar, value);
            tmpOp.SetAttribute(OP_ATTR_PREFIX + "reverseOperand", reverseOperand);
            return;
        } else if (opNameCode == Opcode::OP_GCDS) {
            auto vecTile = tileShape.GetVecTile().tile;
            int64_t tileW = vecTile.empty() ? 1 : vecTile.back();
            int64_t tileH = vecTile.size() > 1 ? vecTile[vecTile.size() - NUM_VALUE_2] : 1;
            int64_t tileFootprint = tileH * tileW;
            int64_t maskCols = ((tileW + NUM_VALUE_7) / NUM_VALUE_8 + NUM_VALUE_31) / BLOCK_SIZE * BLOCK_SIZE;
            int64_t intermediateBytes = (NUM_VALUE_6 * tileFootprint * BytesOf(DT_INT32)) +
                                        (NUM_VALUE_2 * tileFootprint * BytesOf(DT_FP32)) +
                                        NUM_VALUE_4 * tileH * maskCols + GCD_TSEL_TMP_BYTES;
            Shape tempShape(resultTile->shape.size(), 1);
            tempShape[tempShape.size() - 1] = intermediateBytes;
            auto tempTensor = std::make_shared<LogicalTensor>(function, DT_UINT8, tempShape);
            auto& tmpOp = function.AddOperation(opNameCode, {inputTile1}, {resultTile, tempTensor});
            tmpOp.SetAttribute(OpAttributeKey::scalar, value);
            tmpOp.SetAttribute(OP_ATTR_PREFIX + "reverseOperand", reverseOperand);
            return;
        }
        // 确认接口
        auto& op = function.AddOperation(opNameCode, {inputTile1}, {resultTile});
        op.SetAttribute(OpAttributeKey::scalar, value);
        op.SetAttribute(OP_ATTR_PREFIX + "reverseOperand", reverseOperand);
        if constexpr (T == BinaryOpType::DIV || T == BinaryOpType::MOD || T == BinaryOpType::POW) {
            op.SetAttribute(OpAttributeKey::precisionType, precisionType);
        }
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        input1.tileInfo.offset[cur] = i % input1.tensor->GetShape()[cur];
        input1.tileInfo.shape[cur] = std::min(input1.tensor->GetShape()[cur] - input1.tileInfo.offset[cur],
                                              vecTile[cur]);

        TiledBinaryOperationScalar<T>(function, tileShape, cur + 1, input1, value, result, resultTileInfo,
                                      reverseOperand, precisionType);
    }
}

template <BinaryOpType T>
void TiledBinaryOperationScalar(Function& function, const TileShape& tileShape, LogicalTensorPtr operand1,
                                Element value, const LogicalTensorPtr& result, bool reverseOperand,
                                int64_t precisionType)
{
    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input1 = LogicalInput{operand1, tileInfo1};
    TiledBinaryOperationScalar<T>(function, tileShape, 0, input1, value, result, resultTileInfo, reverseOperand,
                                  precisionType);
}

template <BinaryOpType T>
void TiledRemainderSOperation(Function& function, const TileShape& tileShape, size_t cur, LogicalInput& input1,
                              Element& value, const LogicalTensorPtr& result, TileInfo& resultTileInfo,
                              bool reverseOperand, int64_t precisionType)
{
    auto opNameCode = GetBinaryOpNameCode<T, true>();
    if (cur == input1.tensor->GetShape().size()) {
        auto inputTile1 = input1.tensor->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        int64_t shapeSize = resultTileInfo.shape.size();
        int64_t alignSize = static_cast<int64_t>(NUM_VALUE_256 / BytesOf(result->Datatype()));
        int64_t alignedLast = AlignUp(resultTileInfo.shape[shapeSize - 1], alignSize);
        int64_t maskBytes = (alignedLast + NUM_VALUE_8 - 1) / NUM_VALUE_8;
        int64_t maskFloats = AlignUp(maskBytes, NUM_VALUE_256) / static_cast<int64_t>(BytesOf(result->Datatype()));
        int64_t tmpCols = alignedLast + maskFloats + alignSize;
        std::vector<int64_t> tmpShape{1, tmpCols};
        if (opNameCode == Opcode::OP_REMRS) {
            tmpShape[0] = shapeSize > 1 ? resultTileInfo.shape[shapeSize - NUM_VALUE_2] + 1 : NUM_VALUE_2;
        }
        auto tmpTensor = std::make_shared<LogicalTensor>(function, input1.tensor->Datatype(), tmpShape);
        auto& tmpOp = function.AddOperation(opNameCode, {inputTile1}, {resultTile, tmpTensor});
        tmpOp.SetAttribute(OpAttributeKey::scalar, value);
        tmpOp.SetAttribute(OP_ATTR_PREFIX + "reverseOperand", reverseOperand);
        tmpOp.SetAttribute(OpAttributeKey::precisionType, precisionType);
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        input1.tileInfo.offset[cur] = i % input1.tensor->GetShape()[cur];
        input1.tileInfo.shape[cur] = std::min(input1.tensor->GetShape()[cur] - input1.tileInfo.offset[cur],
                                              vecTile[cur]);
        TiledRemainderSOperation<T>(function, tileShape, cur + 1, input1, value, result, resultTileInfo, reverseOperand,
                                    precisionType);
    }
}

template <BinaryOpType T>
void TiledRemainderSOperation(Function& function, const TileShape& tileShape, LogicalTensorPtr operand1, Element value,
                              const LogicalTensorPtr& result, bool reverseOperand = false,
                              int64_t precisionType = static_cast<int64_t>(PrecisionType::INTRINSIC))
{
    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input1 = LogicalInput{operand1, tileInfo1};
    TiledRemainderSOperation<T>(function, tileShape, 0, input1, value, result, resultTileInfo, reverseOperand,
                                precisionType);
}

Tensor Add(const Tensor& self, const Element& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Add");

    const auto& supportedTypes = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_ADDS);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "ADD");
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::ADD>, *Program::GetInstance().GetCurrentFunction(),
                self.GetStorage(), other);
}

Tensor Sub(const Tensor& self, const Element& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Sub");

    const auto& supportedTypes = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_SUBS);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "SUB");
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::SUB>, *Program::GetInstance().GetCurrentFunction(),
                self.GetStorage(), other);
}

Tensor Mul(const Tensor& self, const Element& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Mul");

    const auto& supportedTypes = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_MULS);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "MUL");
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(),
                self.GetStorage(), other);
}

Tensor Div(const Tensor& self, const Element& other, PrecisionType precisionType)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Div");

    const auto& supportedTypes = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_DIVS);
    auto isDivSupportedInt = [](DataType dt) { return dt == DT_INT16 || dt == DT_INT32 || dt == DT_INT64; };
    auto isDivSupportedFloat = [](DataType dt) { return dt == DT_FP16 || dt == DT_FP32 || dt == DT_BF16; };
    auto isDivSupportedScalar = [&](DataType dt) { return isDivSupportedFloat(dt) || isDivSupportedInt(dt); };
    CheckTensorDataType(self.GetStorage(), supportedTypes, "DIV");

    if (isDivSupportedInt(self.GetDataType())) {
        CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, isDivSupportedInt(other.GetDataType()))
            << "Scalar dtype incorrect. When self dtype is integer (DT_INT16/DT_INT32/DT_INT64), "
            << "scalar must be DT_INT16, DT_INT32 or DT_INT64, actual is: " << DataType2String(other.GetDataType());
        Tensor castSelf = Cast(self, DT_FP32, CastMode::CAST_NONE, SaturationMode::ON);
        Element castOther(DT_FP32, other.Cast<float>());
        auto [castResult, op] = TensorBinaryOperationScalarWithOp<BinaryOpType::DIV>(
            *Program::GetInstance().GetCurrentFunction(), castSelf.GetStorage(), castOther);
        op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
        return Tensor(castResult);
    }

    CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, isDivSupportedScalar(other.GetDataType()))
        << "Scalar dtype incorrect. When self dtype is float, "
        << "scalar must be float (DT_FP16/DT_FP32/DT_BF16) or integer (DT_INT16/DT_INT32/DT_INT64), "
        << "actual is: " << DataType2String(other.GetDataType());

    if (isDivSupportedInt(other.GetDataType())) {
        Tensor castSelf = Cast(self, DT_FP32, CastMode::CAST_NONE, SaturationMode::ON);
        Element castOther(DT_FP32, other.Cast<float>());
        auto [result, op] = TensorBinaryOperationScalarWithOp<BinaryOpType::DIV>(
            *Program::GetInstance().GetCurrentFunction(), castSelf.GetStorage(), castOther);
        op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
        return Cast(Tensor(result), self.GetDataType(), CastMode::CAST_RINT, SaturationMode::OFF);
    }

    auto [result, op] = TensorBinaryOperationScalarWithOp<BinaryOpType::DIV>(
        *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), other);
    op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    return Tensor(result);
}

Tensor Fmod(const Tensor& self, const Element& other, PrecisionType precisionType)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Fmod");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "MOD");
    auto [result, op] = TensorBinaryOperationScalarWithOp<BinaryOpType::MOD>(
        *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), other);
    op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    return Tensor(result);
}

Tensor Remainder(const Tensor& self, const Element& other, PrecisionType precisionType)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Remainder");

    const auto& supportedTypes = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_REMS);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "REM");
    auto selfDtype = self.GetDataType();
    Element castOther = CastScalarToDtype(other, selfDtype);
    bool isA5Architecture = (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510);
    if ((!isA5Architecture && selfDtype == DT_INT16) || selfDtype == DT_FP16) {
        Tensor castSelf = Cast(self, DT_FP32, CastMode::CAST_NONE, SaturationMode::ON);
        auto [result, op] = TensorBinaryOperationScalarWithOp<BinaryOpType::REM>(
            *Program::GetInstance().GetCurrentFunction(), castSelf.GetStorage(), castOther);
        op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
        Tensor castedResult = Cast(Tensor(result), selfDtype, CastMode::CAST_NONE, SaturationMode::ON);
        return castedResult;
    }
    auto [result, op] = TensorBinaryOperationScalarWithOp<BinaryOpType::REM>(
        *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), castOther);
    op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    return Tensor(result);
}

Tensor Remainder(const Element& self, const Tensor& other, PrecisionType precisionType)
{
    DECLARE_TRACER();
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Remainder");

    const auto& supportedTypes = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_REMRS);
    CheckTensorDataType(other.GetStorage(), supportedTypes, "REM");
    auto otherDtype = other.GetDataType();
    Element castSelf = CastScalarToDtype(self, otherDtype);
    bool isA5Architecture = (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510);
    if ((!isA5Architecture && otherDtype == DT_INT16) || otherDtype == DT_FP16) {
        Tensor castOther = Cast(other, DT_FP32, CastMode::CAST_NONE, SaturationMode::ON);
        auto [result, op] = TensorBinaryOperationScalarWithOp<BinaryOpType::REMR>(
            *Program::GetInstance().GetCurrentFunction(), castOther.GetStorage(), castSelf);
        op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
        op->SetAttribute(OP_ATTR_PREFIX + "reverseOperand", true);
        Tensor castedResult = Cast(Tensor(result), otherDtype, CastMode::CAST_NONE, SaturationMode::ON);
        return castedResult;
    }
    auto [result, op] = TensorBinaryOperationScalarWithOp<BinaryOpType::REMR>(
        *Program::GetInstance().GetCurrentFunction(), other.GetStorage(), castSelf);
    op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    op->SetAttribute(OP_ATTR_PREFIX + "reverseOperand", true);
    return Tensor(result);
}

Tensor BitwiseAnd(const Tensor& self, const Element& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "BitwiseAnd");

    const auto& supportedTypes = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_BITWISEANDS);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "BITWISEAND");
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::BITWISEAND>, *Program::GetInstance().GetCurrentFunction(),
                self.GetStorage(), other);
}

Tensor BitwiseOr(const Tensor& self, const Element& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "BitwiseOr");

    const auto& supportedTypes = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_BITWISEORS);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "BITWISEOR");
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::BITWISEOR>, *Program::GetInstance().GetCurrentFunction(),
                self.GetStorage(), other);
}

Tensor BitwiseXor(const Tensor& self, const Element& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "BitwiseXor");

    const auto& supportedTypes = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_BITWISEXORS);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "BITWISEXOR");
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::BITWISEXOR>, *Program::GetInstance().GetCurrentFunction(),
                self.GetStorage(), other);
}

Tensor Maximum(const Tensor& operand1, const Element& operand2)
{
    DECLARE_TRACER();
    CheckTensorFormat(operand1.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Maximum");

    const auto& supportedTypes = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_MAXS);
    CheckTensorDataType(operand1.GetStorage(), supportedTypes, "MAX");
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::MAX>, *Program::GetInstance().GetCurrentFunction(),
                operand1.GetStorage(), operand2);
}

Tensor Minimum(const Tensor& operand1, const Element& operand2)
{
    DECLARE_TRACER();
    CheckTensorFormat(operand1.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Minimum");

    const auto& supportedTypes = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_MINS);
    CheckTensorDataType(operand1.GetStorage(), supportedTypes, "MIN");
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::MIN>, *Program::GetInstance().GetCurrentFunction(),
                operand1.GetStorage(), operand2);
}

Tensor LReLU(const Tensor& self, const Element& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "LReLU");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "LRELU");
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::LRELU>, *Program::GetInstance().GetCurrentFunction(),
                self.GetStorage(), other);
}

Tensor CeilDiv(const Tensor& self, const Element& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "CeilDiv");

    std::unordered_set<DataType> supportedTypes = {DT_INT32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "CEILDIV");

    Tensor selfFp32 = Cast(self, DataType::DT_FP32);
    Element otherFp32(DT_FP32, other.Cast<float>());
    Tensor resultFp32 = Div(selfFp32, otherFp32);
    resultFp32 = Ceil(resultFp32);
    Tensor result = Cast(resultFp32, DT_INT32);
    Tensor target = Mul(result, other);
    Tensor diff = Sub(self, target);
    Tensor sgn_diff = Sign(diff);
    Element sgn_other;
    if (other.GetSignedData() > 0) {
        sgn_other = Element(DT_INT32, 1);
    } else {
        sgn_other = Element(DT_INT32, -1);
    }
    Tensor product_sgn = Mul(sgn_diff, sgn_other);
    Tensor inc = Clip(product_sgn, Element(DT_INT32, 0), Element(DT_INT32, 1));
    result = Add(result, inc);
    return result;
}

Tensor FloorDiv(const Tensor& self, const Element& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "FLOORDIV");

    const auto& supportedTypes = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_FLOORDIVS);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "FLOORDIV");
    CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, other.GetDataType() == self.GetDataType())
        << "Scalar dtype incorrect. Scalar dtype should be same as self dtype, self dtype is: "
        << DataType2String(self.GetDataType()) << ", actual scalar dtype is: " << DataType2String(other.GetDataType());
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::FLOORDIV>, *Program::GetInstance().GetCurrentFunction(),
                self.GetStorage(), other);
}

template <BinaryOpType T>
void BinaryOperationScalarTileFunc(Function& function, const TileShape& tileShape,
                                   const std::vector<LogicalTensorPtr>& iOperand,
                                   const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    int64_t precisionType = static_cast<int64_t>(PrecisionType::INTRINSIC);
    if constexpr (T == BinaryOpType::DIV || T == BinaryOpType::MOD || T == BinaryOpType::POW) {
        precisionType = GetOperationPrecisionType(op);
    }
    TiledBinaryOperationScalar<T>(function, tileShape, iOperand[0], op.GetElementAttribute(OpAttributeKey::scalar),
                                  oOperand[0], false, precisionType);
}

template <BinaryOpType T>
void BinaryOperationScalarResTileFunc(Function& function, const TileShape& tileShape,
                                      const std::vector<LogicalTensorPtr>& iOperand,
                                      const std::vector<LogicalTensorPtr>& oOperand,
                                      [[maybe_unused]] const Operation& op)
{
    int64_t precisionType = static_cast<int64_t>(PrecisionType::INTRINSIC);
    if constexpr (T == BinaryOpType::DIV || T == BinaryOpType::MOD) {
        precisionType = GetOperationPrecisionType(op);
    }
    TiledBinaryOperationScalar<T>(function, tileShape, iOperand[0], op.GetElementAttribute(OpAttributeKey::scalar),
                                  oOperand[0], op.GetBoolAttribute(OP_ATTR_PREFIX + "reverseOperand"), precisionType);
}

template <BinaryOpType T>
void RemainderSTileFunc(Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
                        const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    int64_t precisionType = GetOperationPrecisionType(op);
    TiledRemainderSOperation<T>(function, tileShape, iOperand[0], op.GetElementAttribute(OpAttributeKey::scalar),
                                oOperand[0], op.GetBoolAttribute(OP_ATTR_PREFIX + "reverseOperand"), precisionType);
}

// OP_S_ADDS OP_S_SUBS OP_S_MULS OP_S_DIVS OP_S_MAXS

REGISTER_OPERATION_TILED_FUNC(OP_ADDS, Opcode::OP_ADDS, BinaryOperationScalarTileFunc<BinaryOpType::ADD>);
REGISTER_OPERATION_TILED_FUNC(OP_SUBS, Opcode::OP_SUBS, BinaryOperationScalarTileFunc<BinaryOpType::SUB>);
REGISTER_OPERATION_TILED_FUNC(OP_MULS, Opcode::OP_MULS, BinaryOperationScalarTileFunc<BinaryOpType::MUL>);
REGISTER_OPERATION_TILED_FUNC(OP_DIVS, Opcode::OP_DIVS, BinaryOperationScalarTileFunc<BinaryOpType::DIV>);
REGISTER_OPERATION_TILED_FUNC(OP_MAXS, Opcode::OP_MAXS, BinaryOperationScalarTileFunc<BinaryOpType::MAX>);
REGISTER_OPERATION_TILED_FUNC(OP_MINS, Opcode::OP_MINS, BinaryOperationScalarTileFunc<BinaryOpType::MIN>);
REGISTER_OPERATION_TILED_FUNC(OP_POWS, Opcode::OP_POWS, BinaryOperationScalarTileFunc<BinaryOpType::POW>);
REGISTER_OPERATION_TILED_FUNC(OP_LRELU, Opcode::OP_LRELU, BinaryOperationScalarTileFunc<BinaryOpType::LRELU>);
REGISTER_OPERATION_TILED_FUNC(OP_MODS, Opcode::OP_MODS, BinaryOperationScalarTileFunc<BinaryOpType::MOD>);
REGISTER_OPERATION_TILED_FUNC(OP_BITWISEANDS, Opcode::OP_BITWISEANDS,
                              BinaryOperationScalarTileFunc<BinaryOpType::BITWISEAND>);
REGISTER_OPERATION_TILED_FUNC(OP_BITWISEORS, Opcode::OP_BITWISEORS,
                              BinaryOperationScalarTileFunc<BinaryOpType::BITWISEOR>);
REGISTER_OPERATION_TILED_FUNC(OP_BITWISEXORS, Opcode::OP_BITWISEXORS,
                              BinaryOperationScalarTileFunc<BinaryOpType::BITWISEXOR>);
REGISTER_OPERATION_TILED_FUNC(OP_GCDS, Opcode::OP_GCDS, BinaryOperationScalarTileFunc<BinaryOpType::GCD>);
REGISTER_OPERATION_TILED_FUNC(OP_REMS, Opcode::OP_REMS, RemainderSTileFunc<BinaryOpType::REM>);
REGISTER_OPERATION_TILED_FUNC(OP_REMRS, Opcode::OP_REMRS, RemainderSTileFunc<BinaryOpType::REMR>);
REGISTER_OPERATION_TILED_FUNC(OP_FLOORDIVS, Opcode::OP_FLOORDIVS,
                              BinaryOperationScalarResTileFunc<BinaryOpType::FLOORDIV>);

} // namespace npu::tile_fwk
