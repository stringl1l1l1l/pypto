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
 * \\file index_add.cpp
 * \\brief
 */

#include <limits>
#include <cmath>
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/operation/operation_common.h"
#include "interface/operation/vector/gather_mask_common.h"
#include "tensor_transformation.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {

constexpr float FP16_MAX = 65504.0f;

struct IndexAddPara {
    const LogicalTensorPtr& selfInput;
    const LogicalTensorPtr& srcInput;
    const LogicalTensorPtr& indicesInput;
    const LogicalTensorPtr& dstTensor;
    const int axis;
    const Element& alpha;
};

struct IndexAddTileInfoPara {
    TileInfo selfTileInfo;
    TileInfo srcTileInfo;
    TileInfo indicesTileInfo;
    TileInfo dstTileInfo;
};

Shape GetTempShape(Shape shape, size_t axis)
{
    Shape newShape(shape.size(), 1);
    for (size_t i = axis + 1; i < shape.size(); ++i) {
        newShape[i] = shape[i];
    }
    auto alignSize = BLOCK_SIZE / BytesOf(DT_BF16);
    newShape[shape.size() - 1] = (newShape[shape.size() - 1] + alignSize - 1) / alignSize * alignSize;
    return newShape;
}

// IndexAdd in UB
void IndexAddUBExpandFunc(Function& function, const IndexAddPara& indexaddPara, IndexAddTileInfoPara& indexaddTileInfo)
{
    const LogicalTensorPtr& selfInput = indexaddPara.selfInput;
    const LogicalTensorPtr& srcInput = indexaddPara.srcInput;
    const LogicalTensorPtr& indicesInput = indexaddPara.indicesInput;
    const LogicalTensorPtr& dstTensor = indexaddPara.dstTensor;
    const int axis = indexaddPara.axis;
    const Element& alpha = indexaddPara.alpha;

    auto dstTile = dstTensor->View(function, indexaddTileInfo.dstTileInfo.shape, indexaddTileInfo.dstTileInfo.offset);
    auto selfTile = selfInput->View(function, indexaddTileInfo.selfTileInfo.shape,
                                    indexaddTileInfo.selfTileInfo.offset);
    auto srcTile = srcInput->View(function, indexaddTileInfo.srcTileInfo.shape, indexaddTileInfo.srcTileInfo.offset);
    indexaddTileInfo.indicesTileInfo.offset = {
        indexaddTileInfo.srcTileInfo.offset[axis]}; // 按照srcShape所在的axis轴切分
    indexaddTileInfo.indicesTileInfo.shape = {indexaddTileInfo.srcTileInfo.shape[axis]};
    auto indexTile = indicesInput->View(function, indexaddTileInfo.indicesTileInfo.shape,
                                        indexaddTileInfo.indicesTileInfo.offset);
    Shape tempShape(dstTile->GetShape().size(), 1);
    auto alignSize = BLOCK_SIZE / BytesOf(DT_BF16);
    tempShape[dstTile->GetShape().size() - 1] = (tempShape[dstTile->GetShape().size() - 1] + alignSize - 1) /
                                                alignSize * alignSize;
    auto tempBuffer = std::make_shared<LogicalTensor>(function, DT_BF16, tempShape);

    if (selfTile->Datatype() == DT_BF16 || (selfTile->Datatype() == DT_FP16 && indexTile->Datatype() == DT_INT64 &&
                                            (std::abs(alpha.Cast<float>() - 1) < 1e-6f))) {
        // vector和scalar均不支持BF16直接计算; alpha=1,且index类型为int64时逻辑不一样
        LogicalTensorPtr selfConvertedTile = std::make_shared<LogicalTensor>(function, DT_FP32, selfTile->GetShape());
        Operation& castSelfOp = function.AddOperation(Opcode::OP_CAST, {selfTile}, {selfConvertedTile});
        selfConvertedTile->UpdateDynValidShape(selfTile->GetDynValidShape());
        castSelfOp.SetAttribute(OP_ATTR_PREFIX + "mode", CastMode::CAST_NONE);
        LogicalTensorPtr srcConvertedTile = std::make_shared<LogicalTensor>(function, DT_FP32, srcTile->GetShape());
        Operation& castSrcOp = function.AddOperation(Opcode::OP_CAST, {srcTile}, {srcConvertedTile});
        srcConvertedTile->UpdateDynValidShape(srcTile->GetDynValidShape());
        castSrcOp.SetAttribute(OP_ATTR_PREFIX + "mode", CastMode::CAST_NONE);
        LogicalTensorPtr dstConvertedTile = std::make_shared<LogicalTensor>(function, DT_FP32, dstTile->GetShape());
        tempBuffer = std::make_shared<LogicalTensor>(function, DT_BF16, GetTempShape(dstTile->GetShape(), axis));
        auto& op = function.AddOperation(Opcode::OP_INDEX_ADD_UB, {selfConvertedTile, srcConvertedTile, indexTile},
                                         {dstConvertedTile, tempBuffer});
        dstConvertedTile->UpdateDynValidShape(dstTile->GetDynValidShape());
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        op.SetAttribute(OpAttributeKey::scalar, alpha);
        Operation& castDstOp = function.AddOperation(Opcode::OP_CAST, {dstConvertedTile}, {dstTile});
        castDstOp.SetAttribute(OP_ATTR_PREFIX + "mode", CastMode::CAST_RINT);
    } else {
        auto& op = function.AddOperation(Opcode::OP_INDEX_ADD_UB, {selfTile, srcTile, indexTile},
                                         {dstTile, tempBuffer});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        op.SetAttribute(OpAttributeKey::scalar, alpha);
    }
}

void InnerTiledIndexAddUB(size_t cur, Function& function, const TileShape& tileShape, const IndexAddPara& indexaddPara,
                          IndexAddTileInfoPara& indexaddTileInfo)
{
    if (cur == indexaddPara.dstTensor->shape.size()) {
        IndexAddUBExpandFunc(function, indexaddPara, indexaddTileInfo);
        return;
    }

    auto& vecTile = tileShape.GetVecTile();
    int64_t tmpTile = vecTile[cur];
    // axis 维度不参与切分，也不循环
    if (static_cast<int>(cur) == indexaddPara.axis) {
        indexaddTileInfo.dstTileInfo.offset[cur] = 0;
        indexaddTileInfo.dstTileInfo.shape[cur] = indexaddPara.dstTensor->GetShape()[cur];
        indexaddTileInfo.selfTileInfo.offset[cur] = 0;
        indexaddTileInfo.selfTileInfo.shape[cur] = indexaddPara.selfInput->GetShape()[cur];
        indexaddTileInfo.srcTileInfo.offset[cur] = 0;
        indexaddTileInfo.srcTileInfo.shape[cur] = indexaddPara.srcInput->GetShape()[cur];
        InnerTiledIndexAddUB(cur + 1, function, tileShape, indexaddPara, indexaddTileInfo);
        return;
    }

    // 非 axis 维度正常切分
    for (int64_t i = 0; i < indexaddPara.srcInput->GetShape()[cur]; i += tmpTile) {
        indexaddTileInfo.dstTileInfo.offset[cur] = i;
        indexaddTileInfo.dstTileInfo.shape[cur] = std::min(indexaddPara.dstTensor->GetShape()[cur] - i, tmpTile);

        indexaddTileInfo.selfTileInfo.offset[cur] = i;
        indexaddTileInfo.selfTileInfo.shape[cur] = std::min(indexaddPara.selfInput->GetShape()[cur] - i, tmpTile);

        indexaddTileInfo.srcTileInfo.offset[cur] = i;
        indexaddTileInfo.srcTileInfo.shape[cur] = std::min(indexaddPara.srcInput->GetShape()[cur] - i, tmpTile);

        InnerTiledIndexAddUB(cur + 1, function, tileShape, indexaddPara, indexaddTileInfo);
    }
}

void TiledIndexAddUB(Function& function, const TileShape& tileShape, const IndexAddPara& indexaddPara)
{
    // Check Operands Valid
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          indexaddPara.selfInput->GetShape().size() == indexaddPara.selfInput->GetOffset().size())
        << "The size of indexaddPara selfinput shape and selfinput offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          indexaddPara.srcInput->GetShape().size() == indexaddPara.srcInput->GetOffset().size())
        << "The size of indexaddPara srcInput shape and srcInput offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          indexaddPara.indicesInput->GetShape().size() == indexaddPara.indicesInput->GetOffset().size())
        << "The size of indexaddPara indicesInput shape and indicesInput offset should be equal";

    IndexAddTileInfoPara indexaddTileInfo{
        TileInfo(indexaddPara.selfInput->GetShape().size(), indexaddPara.selfInput->GetOffset().size()),
        TileInfo(indexaddPara.srcInput->GetShape().size(), indexaddPara.srcInput->GetOffset().size()),
        TileInfo(indexaddPara.indicesInput->GetShape().size(), indexaddPara.indicesInput->GetOffset().size()),
        TileInfo(indexaddPara.dstTensor->GetShape().size(), indexaddPara.dstTensor->GetOffset().size())};
    InnerTiledIndexAddUB(0, function, tileShape, indexaddPara, indexaddTileInfo);
}

void TensorIndexAddUB(Function& function, const IndexAddPara& indexaddPara)
{
    auto& op = GraphUtils::AddDynOperation(function, Opcode::OP_INDEX_ADD_UB,
                                           {indexaddPara.selfInput, indexaddPara.srcInput, indexaddPara.indicesInput},
                                           {indexaddPara.dstTensor});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", indexaddPara.axis);
    op.SetAttribute(OpAttributeKey::scalar, indexaddPara.alpha);
}

bool CheckAlphaOverflow(Element alpha, DataType dtype)
{
    double value = alpha.Cast<double>();
    if (std::isnan(value) || std::isinf(value))
        return true;
    switch (dtype) {
        case DT_INT8:
            return value < std::numeric_limits<int8_t>::min() || value > std::numeric_limits<int8_t>::max();
        case DT_INT16:
            return value < std::numeric_limits<int16_t>::min() || value > std::numeric_limits<int16_t>::max();
        case DT_INT32:
            return value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max();
        case DT_FP16:
            return std::abs(value) > FP16_MAX;
        case DT_BF16:
            return std::abs(value) > std::numeric_limits<float>::max();
        case DT_FP32:
            return std::abs(value) > std::numeric_limits<float>::max();
        default:
            return false;
    }
}

void CheckIndexAddParamsInvalid(const Tensor& self, const Tensor& src, const Tensor& indices, const int axis,
                                const Element& alpha, const Opcode& opCode)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          axis < static_cast<int>(self.GetShape().size()) && axis >= -static_cast<int>(self.GetShape().size()))
        << "axis out of range of shape size";

    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_5, "INDEXADD");
    CheckTensorDimRange(indices.GetStorage(), 1, 1, "INDEXADD");
    CheckTensorShapeSize(self.GetStorage(), "INDEXADD");
    CheckTensorShapeSize(src.GetStorage(), "INDEXADD");
    CheckTensorShapeSize(indices.GetStorage(), "INDEXADD");
    std::vector<LogicalTensorPtr> tensors = {self.GetStorage(), src.GetStorage()};
    CheckTensorsDimConsistency(tensors, "INDEXADD");
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, src.GetShape()[axis] == indices.GetShape()[0])
        << "src shape[axis] and indices[0] must equal";
    for (size_t i = 0; i < self.GetShape().size(); ++i) {
        if (static_cast<int>(i) == axis) {
            continue;
        }
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, src.GetShape()[i] == self.GetShape()[i])
            << "src shape and self shape should be equal";
    }

    // 检查数据类型和格式
    std::unordered_set<DataType> supportedTypes = {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16};
    if (opCode == Opcode::OP_INDEX_ADD) {
        supportedTypes.insert(DT_INT8);
    }
    CheckTensorDataType(self.GetStorage(), supportedTypes, "INDEXADD");
    CheckTensorsDataTypeConsistency(self.GetStorage(), src.GetStorage(), "INDEXADD");
    CheckTensorsFormatConsistency(self.GetStorage(), src.GetStorage(), "INDEXADD");
    std::unordered_set<DataType> indexSupportedTypes = {DT_INT32, DT_INT64};
    CheckTensorDataType(indices.GetStorage(), indexSupportedTypes, "INDEXADD");

    // 检验 alpha 溢出
    if (CheckAlphaOverflow(alpha, self.GetDataType())) {
        CHECK(VectorErrorCode::ERR_RUNTIME_LOGIC, false)
            << "Value cannot be converted to type " << DataType2String(self.GetDataType()) << " without overflow!";
    }
}

Tensor IndexAddUB(const Tensor& self, const Tensor& src, const Tensor& indices, int axis, const Element& alpha)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "IndexAddUB");
    CheckTensorFormat(src.GetStorage(), {TileOpFormat::TILEOP_NZ}, "IndexAddUB");
    CheckTensorFormat(indices.GetStorage(), {TileOpFormat::TILEOP_NZ}, "IndexAddUB");

    CheckAxisRange(self, axis);
    CheckIndexAddParamsInvalid(self, src, indices, axis, alpha, Opcode::OP_INDEX_ADD_UB);
    DataType selfDataType = self.GetDataType();
    Element alpha_ = Element(selfDataType, alpha.Cast<float>());
    Tensor result(selfDataType, self.GetShape());
    result.GetStorage()->UpdateDynValidShape(self.GetStorage()->GetDynValidShape());
    CALL(IndexAddUB, *Program::GetInstance().GetCurrentFunction(),
         {self.GetStorage(), src.GetStorage(), indices.GetStorage(), result.GetStorage(), axis, alpha_});
    return result;
}

// IndexAdd in GM
void IndexAddExpandFunc(Function& function, const IndexAddPara& indexaddPara, IndexAddTileInfoPara& indexaddTileInfo,
                        const LogicalTensorPtr& cachedDstTile = nullptr,
                        const LogicalTensorPtr& cachedSelfTile = nullptr)
{
    const LogicalTensorPtr& selfInput = indexaddPara.selfInput;
    const LogicalTensorPtr& srcInput = indexaddPara.srcInput;
    const LogicalTensorPtr& indicesInput = indexaddPara.indicesInput;
    const LogicalTensorPtr& dstTensor = indexaddPara.dstTensor;
    const int axis = indexaddPara.axis;

    auto selfTile = cachedSelfTile ? cachedSelfTile :
                                     selfInput->View(function, indexaddTileInfo.selfTileInfo.shape,
                                                     indexaddTileInfo.selfTileInfo.offset);
    auto dstTile = cachedDstTile ? cachedDstTile :
                                   dstTensor->View(function, indexaddTileInfo.dstTileInfo.shape,
                                                   indexaddTileInfo.dstTileInfo.offset);
    auto srcTile = srcInput->View(function, indexaddTileInfo.srcTileInfo.shape, indexaddTileInfo.srcTileInfo.offset);
    indexaddTileInfo.indicesTileInfo.offset = {indexaddTileInfo.srcTileInfo.offset[axis]};
    indexaddTileInfo.indicesTileInfo.shape = {indexaddTileInfo.srcTileInfo.shape[axis]};
    auto indexTile = indicesInput->View(function, indexaddTileInfo.indicesTileInfo.shape,
                                        indexaddTileInfo.indicesTileInfo.offset);
    Shape tmpShape(NUM_VALUE_2, 1);
    auto alignSize = BLOCK_SIZE / BytesOf(srcTile->Datatype());
    tmpShape[1] = AlignUp(srcTile->GetShape()[srcTile->GetShape().size() - 1], alignSize);
    bool useSimt = Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510 &&
                   (srcTile->Datatype() == DT_FP32 || srcTile->Datatype() == DT_FP16 ||
                    srcTile->Datatype() == DT_BF16) &&
                   indexTile->Datatype() == DT_INT32;
    // Separate rows for element offsets and alpha-scaled values; both start at a block boundary.
    if (useSimt) {
        tmpShape[0] = NUM_VALUE_2;
    }
    auto tmpTile = std::make_shared<LogicalTensor>(function, useSimt ? DT_FP32 : srcTile->Datatype(), tmpShape);

    auto& op = function.AddOperation(Opcode::OP_INDEX_ADD, {selfTile, srcTile, indexTile}, {dstTile, tmpTile});
    op.SetAttribute(OpAttributeKey::inplaceIdx, 0);
    op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
    op.SetAttribute(OpAttributeKey::scalar, indexaddPara.alpha);
    if (useSimt) {
        op.SetAttribute(OP_ATTR_PREFIX + "requires_simt", true);
    }
}

using TileCache = std::unordered_map<int64_t, std::pair<LogicalTensorPtr, LogicalTensorPtr>>;

void InnerTiledIndexAdd(size_t cur, Function& function, const TileShape& tileShape, const IndexAddPara& indexaddPara,
                        IndexAddTileInfoPara& indexaddTileInfo, TileCache& tileCache, int64_t encodeKey = 0)
{
    if (cur == indexaddPara.dstTensor->shape.size()) {
        auto it = tileCache.find(encodeKey);
        if (it == tileCache.end()) {
            auto selfTile = indexaddPara.selfInput->View(function, indexaddTileInfo.selfTileInfo.shape,
                                                         indexaddTileInfo.selfTileInfo.offset);
            auto dstTile = indexaddPara.dstTensor->View(function, indexaddTileInfo.dstTileInfo.shape,
                                                        indexaddTileInfo.dstTileInfo.offset);
            it = tileCache.emplace(encodeKey, std::make_pair(dstTile, selfTile)).first;
        }
        // 调用缓存的dstTile创建子图
        IndexAddExpandFunc(function, indexaddPara, indexaddTileInfo, it->second.first, it->second.second);
        return;
    }
    const auto& vecTile = tileShape.GetVecTile();
    int64_t tileStep = vecTile[cur];
    const auto& srcShape = indexaddPara.srcInput->GetShape();
    const auto& dstShape = indexaddPara.dstTensor->GetShape();
    int64_t numTilesInCurDim = (srcShape[cur] + tileStep - 1) / tileStep;
    if (static_cast<int>(cur) == indexaddPara.axis) {
        // self和dst都在GM上，在axis轴不切分
        indexaddTileInfo.dstTileInfo.offset[cur] = 0;
        indexaddTileInfo.dstTileInfo.shape[cur] = dstShape[cur];
        indexaddTileInfo.selfTileInfo.offset[cur] = 0;
        indexaddTileInfo.selfTileInfo.shape[cur] = dstShape[cur];
        for (int i = 0; i < srcShape[cur]; i += tileStep) {
            indexaddTileInfo.srcTileInfo.offset[cur] = i;
            indexaddTileInfo.srcTileInfo.shape[cur] = std::min(srcShape[cur] - i, tileStep);
            // axis维度不参与编码，使用同一个encodeKey
            InnerTiledIndexAdd(cur + 1, function, tileShape, indexaddPara, indexaddTileInfo, tileCache, encodeKey);
        }
    } else {
        // 非 axis 维度，dst、self、src都切块
        int64_t tileIndex = 0; // 当前维度块索引
        for (int i = 0; i < srcShape[cur]; i += tileStep) {
            indexaddTileInfo.dstTileInfo.offset[cur] = i;
            indexaddTileInfo.dstTileInfo.shape[cur] = std::min(dstShape[cur] - i, tileStep);
            indexaddTileInfo.selfTileInfo.offset[cur] = i;
            indexaddTileInfo.selfTileInfo.shape[cur] = std::min(dstShape[cur] - i, tileStep);
            indexaddTileInfo.srcTileInfo.offset[cur] = i;
            indexaddTileInfo.srcTileInfo.shape[cur] = std::min(srcShape[cur] - i, tileStep);
            // 使用混合基数编码
            int64_t newKey = encodeKey * numTilesInCurDim + tileIndex;
            tileIndex++;
            InnerTiledIndexAdd(cur + 1, function, tileShape, indexaddPara, indexaddTileInfo, tileCache, newKey);
        }
    }
}

void TiledIndexAdd(Function& function, const TileShape& tileShape, const IndexAddPara& indexaddPara)
{
    // Check Operands Valid
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          indexaddPara.selfInput->GetShape().size() == indexaddPara.selfInput->GetOffset().size())
        << "The size of indexaddPara selfinput shape and selfinput offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          indexaddPara.srcInput->GetShape().size() == indexaddPara.srcInput->GetOffset().size())
        << "The size of indexaddPara srcInput shape and srcInput offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          indexaddPara.indicesInput->GetShape().size() == indexaddPara.indicesInput->GetOffset().size())
        << "The size of indexaddPara indicesInput shape and indicesInput offset should be equal";

    IndexAddTileInfoPara indexaddTileInfo{
        TileInfo(indexaddPara.selfInput->GetShape().size(), indexaddPara.selfInput->GetOffset().size()),
        TileInfo(indexaddPara.srcInput->GetShape().size(), indexaddPara.srcInput->GetOffset().size()),
        TileInfo(indexaddPara.indicesInput->GetShape().size(), indexaddPara.indicesInput->GetOffset().size()),
        TileInfo(indexaddPara.dstTensor->GetShape().size(), indexaddPara.dstTensor->GetOffset().size())};
    TileCache tileCache;
    InnerTiledIndexAdd(0, function, tileShape, indexaddPara, indexaddTileInfo, tileCache);
}

void TensorIndexAdd(Function& function, const IndexAddPara& indexaddPara)
{
    auto& op = GraphUtils::AddDynOperation(function, Opcode::OP_INDEX_ADD,
                                           {indexaddPara.selfInput, indexaddPara.srcInput, indexaddPara.indicesInput},
                                           {indexaddPara.dstTensor});
    op.SetAttribute(OpAttributeKey::inplaceIdx, 0);
    op.SetAttribute(OP_ATTR_PREFIX + "axis", indexaddPara.axis);
    op.SetAttribute(OpAttributeKey::scalar, indexaddPara.alpha);
}

void IndexAdd_(Tensor& self, const Tensor& src, const Tensor& indices, int axis, const Element& alpha)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "IndexAdd_");
    CheckTensorFormat(src.GetStorage(), {TileOpFormat::TILEOP_NZ}, "IndexAdd_");
    CheckTensorFormat(indices.GetStorage(), {TileOpFormat::TILEOP_NZ}, "IndexAdd_");

    CheckAxisRange(self, axis);
    CheckIndexAddParamsInvalid(self, src, indices, axis, alpha, Opcode::OP_INDEX_ADD);
    DataType selfDataType = self.GetDataType();
    Element castedAlpha = Element(selfDataType, alpha.Cast<float>());
    Tensor result(selfDataType, self.GetShape());
    CALL(IndexAdd, *Program::GetInstance().GetCurrentFunction(),
         {self.GetStorage(), src.GetStorage(), indices.GetStorage(), result.GetStorage(), axis, castedAlpha});
    self = result;
}

void IndexAddUBOperationTileFunc(Function& function, const TileShape& tileShape,
                                 const std::vector<LogicalTensorPtr>& iOperand,
                                 const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    int axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");
    Element alpha = op.GetElementAttribute(OpAttributeKey::scalar);
    TiledIndexAddUB(function, tileShape, {iOperand[0], iOperand[1], iOperand[NUM_VALUE_2], oOperand[0], axis, alpha});
}

void IndexAddOperationTileFunc(Function& function, const TileShape& tileShape,
                               const std::vector<LogicalTensorPtr>& iOperand,
                               const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    int axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");
    Element alpha = op.GetElementAttribute(OpAttributeKey::scalar);
    TiledIndexAdd(function, tileShape, {iOperand[0], iOperand[1], iOperand[NUM_VALUE_2], oOperand[0], axis, alpha});
}

REGISTER_OPERATION_TILED_FUNC(OP_INDEX_ADD_UB, Opcode::OP_INDEX_ADD_UB, IndexAddUBOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_INDEX_ADD, Opcode::OP_INDEX_ADD, IndexAddOperationTileFunc);

} // namespace npu::tile_fwk
