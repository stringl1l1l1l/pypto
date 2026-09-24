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
 * \file operation_impl.cpp
 * \brief
 */

#include "operation_impl.h"
#include <memory>
#include <climits>
#include "tilefwk/data_type.h"
#include "interface/operation/operation.h"
#include "interface/operation/vector/unary.h"
#include "interface/utils/error.h"
#include "distributed/distributed_expand.h"
#include "interface/function/function.h"
#include "tilefwk/symbolic_scalar.h"
#include "tilefwk/tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "interface/utils/common.h"
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"
#include "passes/tensor_graph_pass/expand_function.h"
#include "interface/tensor/irbuilder.h"

using namespace npu::tile_fwk;

namespace {

void CheckFwkOpTileShape(const VecTile& vecTile, const std::shared_ptr<LogicalTensor>& tensor,
                         const std::string& fwkOpName)
{
    const auto& tensorShape = tensor->GetShape();
    CHECK_OP(vecTile.size() >= tensorShape.size()) << "FwkOp tile shape dimension mismatch! "
                                                   << "Creating FwkOp: " << fwkOpName << ", "
                                                   << "Tile dims: " << vecTile.size() << ", "
                                                   << "Tensor dims: " << tensorShape.size() << ", "
                                                   << "Dump tensor: " << tensor->Dump();

    DataType dataType = tensor->Datatype();
    size_t lastDimBytes = vecTile[vecTile.size() - 1] * BytesOf(dataType);
    CHECK_OP(lastDimBytes % BLOCK_SIZE == 0) << "FwkOp tile shape's last dim is not aligned. "
                                             << "Creating FwkOp: " << fwkOpName << ", "
                                             << "Last dim bytes: " << lastDimBytes << ", "
                                             << "BLOCK_SIZE: " << BLOCK_SIZE << ", "
                                             << "Dump tensor: " << tensor->Dump();
}

void CheckViewValidShapesConstraint(const std::vector<SymbolicScalar>& newValidShapes,
                                    const std::vector<int64_t>& shapes, const std::vector<int64_t>& operandShape)
{
    CHECK_OP(newValidShapes.size() == operandShape.size())
        << "View operation failed: newValidShapes dimension count must match original tensor's dimension count. "
        << "newValidShapes has " << newValidShapes.size() << " dimensions, "
        << "but original tensor has " << operandShape.size() << " dimensions.";

    auto validShapesConcrete = SymbolicScalar::Concrete(newValidShapes, 0);

    for (size_t i = 0; i < shapes.size(); i++) {
        CHECK_OP(operandShape[i] == -1 || validShapesConcrete[i] <= operandShape[i])
            << "View operation failed: newValidShapes cannot exceed original tensor's shape at dimension " << i << ". "
            << "newValidShapes[" << i << "] = " << validShapesConcrete[i] << ", "
            << "but original tensor shape[" << i << "] = " << operandShape[i]
            << (operandShape[i] == -1 ? " (dynamic dimension)" : "") << ". "
            << "Note: -1 indicates a dynamic dimension that can be any size.";
    }
}

TileShape GetOpEffectiveTileShape(const Operation& op)
{
    if (op.GetTileShape().GetVecTile().valid()) {
        return op.GetTileShape();
    }
    return TileShape::Current();
}

TileShape GetViewExpandTileShape(const Operation& op)
{
    TileShape tileShape = GetOpEffectiveTileShape(op);
    if (!tileShape.GetVecTile().valid() && !op.GetOOperands().empty()) {
        // 直接写 vec tile，避免 SetVecTile 污染全局 Current()。
        tileShape.GetVecTile().tile = op.GetOOperands()[0]->GetShape();
    }
    return tileShape;
}

TileShape GetAssembleExpandTileShape(const Operation& op)
{
    TileShape tileShape = GetOpEffectiveTileShape(op);
    if (!tileShape.GetVecTile().valid() && !op.GetIOperands().empty()) {
        // 直接写 vec tile，避免 SetVecTile 污染全局 Current()。
        tileShape.GetVecTile().tile = op.GetIOperands()[0]->GetShape();
    }
    return tileShape;
}

void PropagateExpandedViewOpAttrs(Function& function, const Operation& sourceOp, size_t opListPreSize)
{
    auto opListPost = function.Operations(false);
    for (size_t i = opListPreSize; i < opListPost.size(); ++i) {
        auto& newOp = opListPost[i];
        if (newOp.GetOpcode() != Opcode::OP_SLICE && newOp.GetOpcode() != Opcode::OP_CONTRACT) {
            continue;
        }
        newOp.SetScopeInfo(sourceOp.GetScopeInfo());
        newOp.CopyAttrFrom(sourceOp, OP_EMUOP_PREFIX);
        if (sourceOp.HasAttribute(OpAttributeKey::inplaceIdx)) {
            newOp.SetAttribute(OpAttributeKey::inplaceIdx, sourceOp.GetIntAttribute(OpAttributeKey::inplaceIdx));
        }
        if (sourceOp.HasAttribute(OpAttributeKey::isGlobalInput)) {
            newOp.SetAttribute(OpAttributeKey::isGlobalInput, sourceOp.GetBoolAttribute(OpAttributeKey::isGlobalInput));
        }
    }
}

void PropagateExpandedAssembleOpAttrs(Function& function, const Operation& sourceOp, size_t opListPreSize)
{
    auto opListPost = function.Operations(false);
    for (size_t i = opListPreSize; i < opListPost.size(); ++i) {
        auto& newOp = opListPost[i];
        if (newOp.GetOpcode() != Opcode::OP_SLICE && newOp.GetOpcode() != Opcode::OP_CONTRACT) {
            continue;
        }
        newOp.SetScopeInfo(sourceOp.GetScopeInfo());
        newOp.CopyAttrFrom(sourceOp, OP_EMUOP_PREFIX);
        if (sourceOp.HasAttribute(OpAttributeKey::inplaceIdx)) {
            newOp.SetAttribute(OpAttributeKey::inplaceIdx, sourceOp.GetIntAttribute(OpAttributeKey::inplaceIdx));
        }
        if (sourceOp.HasAttribute(OpAttributeKey::rmwMode)) {
            newOp.SetAttribute(OpAttributeKey::rmwMode, sourceOp.GetIntAttribute(OpAttributeKey::rmwMode));
        }
    }
}

void AttachExpandResultTokens(Operation& sliceOp, Operation& contractOp, Operation* currentTileOp,
                              const LogicalTensorPtr& matchReadOn)
{
    if (currentTileOp == nullptr) {
        return;
    }
    for (const auto& token : currentTileOp->result_token_) {
        if (token == nullptr) {
            continue;
        }
        auto tokenType = std::dynamic_pointer_cast<const ir::TokenType>(token->GetType());
        auto normal = ExpandFunction::GetNormalToken(token);
        if (tokenType == nullptr || normal == nullptr) {
            continue;
        }
        if (tokenType->kind_ == ir::TokenKind::READ) {
            if (matchReadOn != nullptr && matchReadOn->GetReadToken() != token) {
                continue;
            }
            sliceOp.result_token_.push_back(normal);
        } else if (tokenType->kind_ == ir::TokenKind::WRITE) {
            contractOp.result_token_.push_back(normal);
        }
    }
}

void AttachExpandInputTokens(Operation& contractOp, Operation* currentTileOp)
{
    if (currentTileOp == nullptr) {
        return;
    }
    for (const auto& token : currentTileOp->tokens_) {
        auto normal = ExpandFunction::GetNormalToken(token);
        if (normal != nullptr) {
            contractOp.tokens_.push_back(normal);
        }
    }
}

void TiledAssembleRecursive(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                            const std::shared_ptr<LogicalTensor>& result, std::shared_ptr<AssembleOpAttribute> attr)
{
    if (cur == input.tensor.GetShape().size()) {
        auto operand = input.tensor.GetStorage();
        auto inputTile = std::make_shared<LogicalTensor>(function, operand->Datatype(), input.tileInfo.shape,
                                                         operand->Format(), operand->Symbol());

        std::vector<int64_t> fromOffset = input.tileInfo.offset;
        std::vector<SymbolicScalar> fromDynOffset;
        fromDynOffset.reserve(fromOffset.size());
        for (size_t i = 0; i < fromOffset.size(); ++i) {
            fromDynOffset.emplace_back(SymbolicScalar(fromOffset[i]));
        }
        auto tileValidShape = GetViewValidShape(operand->GetDynValidShape(), fromOffset, fromDynOffset,
                                                input.tileInfo.shape);
        inputTile->UpdateDynValidShape(tileValidShape);

        auto& sliceOp = function.AddRawOperation(Opcode::OP_SLICE, {operand}, {inputTile});
        sliceOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(fromOffset, fromDynOffset, tileValidShape));

        auto& contractOp = function.AddRawOperation(Opcode::OP_CONTRACT, {inputTile}, {result});
        contractOp.SetAttr("NeedCopy", true);
        auto* currentTileOp = ExpandFunction::GetCurrentTileOp();
        AttachExpandResultTokens(sliceOp, contractOp, currentTileOp, operand);
        AttachExpandInputTokens(contractOp, currentTileOp);
        inputTile->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN);
        const auto& baseToOffset = attr->GetToOffset();
        const auto& baseToDynOffset = attr->GetToDynOffset();
        size_t offsetSize = std::min(baseToOffset.size(), input.tileInfo.offset.size());
        std::vector<int64_t> newToOffset(offsetSize);
        std::vector<SymbolicScalar> newDynOffset;
        newDynOffset.reserve(baseToDynOffset.empty() ? offsetSize : std::min(baseToDynOffset.size(), offsetSize));
        for (size_t i = 0; i < offsetSize; ++i) {
            newToOffset[i] = baseToOffset[i] + input.tileInfo.offset[i];
            if (baseToDynOffset.empty()) {
                newDynOffset.emplace_back(newToOffset[i]);
            } else {
                newDynOffset.push_back(baseToDynOffset[i] + SymbolicScalar(input.tileInfo.offset[i]));
            }
        }
        contractOp.SetOpAttribute(
            std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, newToOffset, newDynOffset));
        return;
    }

    auto& vecTile = tileShape.GetVecTile();
    int64_t tileDim = (cur < vecTile.size()) ? vecTile[cur] : vecTile[vecTile.size() - 1];
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += tileDim) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, tileDim);
        input.tileInfo.offset[cur] = i;
        TiledAssembleRecursive(function, tileShape, cur + 1, input, result, attr);
    }
}

void TiledViewOperationRecursive(Function& function, const TileShape& tileShape, size_t cur, Input& output,
                                 const std::shared_ptr<LogicalTensor>& operand, std::shared_ptr<ViewOpAttribute> attr)
{
    if (cur == output.tensor.GetShape().size()) {
        auto resultParent = output.tensor.GetStorage();
        auto resultTile = std::make_shared<LogicalTensor>(function, resultParent->Datatype(), output.tileInfo.shape,
                                                          resultParent->Format(), resultParent->Symbol());
        const auto& baseFromOffset = attr->GetFromOffset();
        const auto& baseFromDynOffset = attr->GetFromDynOffset();
        size_t offsetSize = std::min(baseFromOffset.size(), output.tileInfo.offset.size());
        std::vector<int64_t> newFromOffset(offsetSize);
        std::vector<SymbolicScalar> newFromDynOffset;
        newFromDynOffset.reserve(baseFromDynOffset.empty() ? offsetSize :
                                                             std::min(baseFromDynOffset.size(), offsetSize));
        for (size_t i = 0; i < offsetSize; ++i) {
            newFromOffset[i] = baseFromOffset[i] + output.tileInfo.offset[i];
            if (baseFromDynOffset.empty()) {
                newFromDynOffset.emplace_back(newFromOffset[i]);
            } else {
                newFromDynOffset.push_back(baseFromDynOffset[i] + SymbolicScalar(output.tileInfo.offset[i]));
            }
        }
        auto tileValidShape = GetViewValidShape(attr->GetToDynValidShape(), output.tileInfo.offset, {},
                                                output.tileInfo.shape);
        resultTile->UpdateDynValidShape(tileValidShape);
        auto& viewOp = function.AddRawOperation(Opcode::OP_SLICE, {operand}, {resultTile});
        auto viewAttr = std::make_shared<ViewOpAttribute>(newFromOffset, newFromDynOffset, tileValidShape);
        if (attr->GetTo() != MemoryType::MEM_UNKNOWN) {
            viewAttr->SetToType(attr->GetTo());
        }
        viewOp.SetOpAttribute(viewAttr);

        auto& contractOp = function.AddRawOperation(Opcode::OP_CONTRACT, {resultTile}, {resultParent});
        contractOp.SetAttr("NeedCopy", true);
        auto* currentTileOp = ExpandFunction::GetCurrentTileOp();
        if (currentTileOp != nullptr) {
            size_t readCount = 0;
            for (const auto& token : currentTileOp->result_token_) {
                auto tokenType = token ? std::dynamic_pointer_cast<const ir::TokenType>(token->GetType()) : nullptr;
                if (tokenType != nullptr && tokenType->kind_ == ir::TokenKind::READ) {
                    ++readCount;
                }
            }
            FE_ASSERT(FeError::INVALID_VAL, readCount <= 1)
                << "VIEW can have at most one READ result token, got " << readCount;
        }
        AttachExpandResultTokens(viewOp, contractOp, currentTileOp, nullptr);
        resultTile->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN);
        std::vector<SymbolicScalar> toDynOffset;
        toDynOffset.reserve(output.tileInfo.offset.size());
        for (size_t i = 0; i < output.tileInfo.offset.size(); ++i) {
            toDynOffset.emplace_back(SymbolicScalar(output.tileInfo.offset[i]));
        }
        contractOp.SetOpAttribute(
            std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, output.tileInfo.offset, toDynOffset));
        return;
    }

    auto& vecTile = tileShape.GetVecTile();
    int64_t tileDim = (cur < vecTile.size()) ? vecTile[cur] : vecTile[vecTile.size() - 1];
    for (int i = 0; i < output.tensor.GetShape()[cur]; i += tileDim) {
        output.tileInfo.shape[cur] = std::min(output.tensor.GetShape()[cur] - i, tileDim);
        output.tileInfo.offset[cur] = i;
        TiledViewOperationRecursive(function, tileShape, cur + 1, output, operand, attr);
    }
}

} // namespace

namespace npu::tile_fwk {

void TiledAssemble(Function& function, const TileShape& tileShape, const std::shared_ptr<LogicalTensor>& operand,
                   const std::shared_ptr<LogicalTensor>& result, std::shared_ptr<AssembleOpAttribute> attr)
{
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, operand->shape.size() == operand->offset.size())
        << "operand's shape size and offset size should be equal";

    TileInfo tileInfo(operand->shape.size(), operand->offset.size());
    auto input = Input{operand, tileInfo};
    TiledAssembleRecursive(function, tileShape, 0, input, result, attr);
}

void TiledViewOperation(Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand,
                        const LogicalTensorPtr& result, const Operation& op)
{
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
    FE_ASSERT(FeError::INVALID_VAL, viewOpAttribute != nullptr)
        << "View op missing ViewOpAttribute, opmagic=" << op.GetOpMagic();
    FE_ASSERT(FeError::INVALID_VAL, result->shape.size() == result->offset.size())
        << "result's shape size and offset size should be equal";

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto output = Input{result, tileInfo};
    TiledViewOperationRecursive(function, tileShape, 0, output, operand, viewOpAttribute);
}

constexpr int NCHW_DIM_NUM = 4;
constexpr int NC1HWC0_DIM_NUM = 5;
constexpr int STRIDE_DIM_NUM = 2;
constexpr int PADS_DIM_NUM = 4;
constexpr int WEIGHT_DIM_NUM = 4;
constexpr int BIAS_DIM_NUM = 1;
constexpr int SMALL_CHANNEL_4 = 4;
constexpr int SMALL_CHANNEL_8 = 8;
constexpr int SMALL_CHANNEL_16 = 16;
static const std::vector<NPUArch> ONLINE_SOFTMAX_SUPPORTED_ARCHITECTURES = {NPUArch::DAV_3510};

void TiledMaxpool(Function& function, const TileShape& tileShape, const std::shared_ptr<LogicalTensor>& input,
                  const std::shared_ptr<LogicalTensor>& output, const Operation& op)
{
    const int dimN = output->shape[NUM_VALUE_0];
    const int dimC1 = output->shape[NUM_VALUE_1];
    const int dimOutH = output->shape[NUM_VALUE_2];
    const int dimOutW = output->shape[NUM_VALUE_3];
    const int dimInH = input->shape[NUM_VALUE_2];
    const int dimInW = input->shape[NUM_VALUE_3];
    const int c0 = output->shape[NUM_VALUE_4];
    const int paddingLeft = op.GetIntAttribute(ConvOpAttributeKey::paddingLeft);
    const int paddingTop = op.GetIntAttribute(ConvOpAttributeKey::paddingTop);
    const int paddingRight = op.GetIntAttribute(ConvOpAttributeKey::paddingRight);
    const int paddingBottom = op.GetIntAttribute(ConvOpAttributeKey::paddingBottom);
    const int strideH = op.GetIntAttribute(ConvOpAttributeKey::strideh);
    const int strideW = op.GetIntAttribute(ConvOpAttributeKey::stridew);
    const int poolH = op.GetIntAttribute(PoolOpAttributeKey::poolh);
    const int poolW = op.GetIntAttribute(PoolOpAttributeKey::poolw);

    auto& vecTile = tileShape.GetVecTile();
    int tileOutH = vecTile[NUM_VALUE_0];
    int tileOutW = vecTile[NUM_VALUE_1];
    bool isOnlyNeedCopy = strideH == 1 && strideW == 1 && poolH == 1 && poolW == 1;

    for (int n = 0; n < dimN; n++) {
        const int tileN = 1;
        for (int c1 = 0; c1 < dimC1; c1++) {
            const int tileC1 = 1;
            for (int h = 0; h < dimOutH; h += tileOutH) {
                const int tileHOut = Min(dimOutH - h, tileOutH);
                int startHIn = -paddingTop + h * strideH;
                int curStartHIn = startHIn > 0 ? startHIn : 0;
                int endHIn = -paddingTop + (h + tileHOut - 1) * strideH + poolH - 1;
                int curEndHIn = endHIn < dimInH ? endHIn : dimInH - 1;
                int tileHIn = curEndHIn - curStartHIn + 1;
                const int curPaddingTop = startHIn > 0 ? 0 : paddingTop;
                const int curPaddingBottom = endHIn < dimInH ? 0 : paddingBottom;
                for (int w = 0; w < dimOutW; w += tileOutW) {
                    const int tileWOut = Min(dimOutW - w, tileOutW);
                    int startWIn = -paddingLeft + w * strideW;
                    int curStartWIn = startWIn > 0 ? startWIn : 0;
                    int endWIn = -paddingLeft + (w + tileWOut - 1) * strideW + poolW - 1;
                    int curEndWIn = endWIn < dimInW ? endWIn : dimInW - 1;
                    int tileWIn = curEndWIn - curStartWIn + 1;
                    const int curPaddingLeft = startWIn > 0 ? 0 : paddingLeft;
                    const int curPaddingRight = endWIn < dimInW ? 0 : paddingRight;

                    auto inTile = input->View(function, {tileN, tileC1, tileHIn, tileWIn, c0},
                                              {n, c1, curStartHIn, curStartWIn, 0});
                    auto outTile = output->View(function, {tileN, tileC1, tileHOut, tileWOut, c0}, {n, c1, h, w, 0});
                    if (isOnlyNeedCopy) {
                        function.AddOperation(Opcode::OP_COPY_UB_TO_UB, {inTile}, {outTile});
                        continue;
                    }

                    auto& maxpoolOp = function.AddOperation(Opcode::OP_MAX_POOL, {inTile}, {outTile});

                    maxpoolOp.SetAttribute(ConvOpAttributeKey::paddingLeft, curPaddingLeft);
                    maxpoolOp.SetAttribute(ConvOpAttributeKey::paddingTop, curPaddingTop);
                    maxpoolOp.SetAttribute(ConvOpAttributeKey::paddingRight, curPaddingRight);
                    maxpoolOp.SetAttribute(ConvOpAttributeKey::paddingBottom, curPaddingBottom);
                    maxpoolOp.SetAttribute(ConvOpAttributeKey::strideh, strideH);
                    maxpoolOp.SetAttribute(ConvOpAttributeKey::stridew, strideW);
                    maxpoolOp.SetAttribute(PoolOpAttributeKey::poolh, poolH);
                    maxpoolOp.SetAttribute(PoolOpAttributeKey::poolw, poolH);
                }
            }
        }
    }
}

void TensorMaxpool(Function& function, const std::shared_ptr<LogicalTensor>& operand,
                   const std::shared_ptr<LogicalTensor>& result, const std::vector<int>& pools,
                   const std::vector<int>& strides, const std::vector<int>& paddings)
{
    const int paddingLeft = paddings[NUM_VALUE_0];
    const int paddingTop = paddings[NUM_VALUE_1];
    const int paddingRight = paddings[NUM_VALUE_2];
    const int paddingBottom = paddings[NUM_VALUE_3];
    const int strideH = strides[NUM_VALUE_0];
    const int strideW = strides[NUM_VALUE_1];
    const int poolH = pools[NUM_VALUE_0];
    const int poolW = pools[NUM_VALUE_1];

    auto& maxpoolTensorOp = function.AddOperation(Opcode::OP_MAX_POOL, {operand}, {result});
    maxpoolTensorOp.SetAttribute(ConvOpAttributeKey::paddingLeft, paddingLeft);
    maxpoolTensorOp.SetAttribute(ConvOpAttributeKey::paddingTop, paddingTop);
    maxpoolTensorOp.SetAttribute(ConvOpAttributeKey::paddingRight, paddingRight);
    maxpoolTensorOp.SetAttribute(ConvOpAttributeKey::paddingBottom, paddingBottom);
    maxpoolTensorOp.SetAttribute(ConvOpAttributeKey::strideh, strideH);
    maxpoolTensorOp.SetAttribute(ConvOpAttributeKey::stridew, strideW);
    maxpoolTensorOp.SetAttribute(PoolOpAttributeKey::poolh, poolH);
    maxpoolTensorOp.SetAttribute(PoolOpAttributeKey::poolw, poolW);
}

Tensor Maxpool(const Tensor& operand, const std::vector<int>& pools, const std::vector<int>& strides,
               const std::vector<int>& paddings)
{
    DECLARE_TRACER();
    // 目前只支持5D操作
    CHECK_OP((operand.GetShape().size() == NC1HWC0_DIM_NUM) && pools.size() == NUM_VALUE_2 &&
             strides.size() == STRIDE_DIM_NUM && paddings.size() == PADS_DIM_NUM);

    const int inDimH = operand.GetShape()[NUM_VALUE_2];
    const int inDimW = operand.GetShape()[NUM_VALUE_3];
    const int paddingLeft = paddings[NUM_VALUE_0];
    const int paddingTop = paddings[NUM_VALUE_1];
    const int paddingRight = paddings[NUM_VALUE_2];
    const int paddingBottom = paddings[NUM_VALUE_3];
    const int strideH = strides[NUM_VALUE_0];
    const int strideW = strides[NUM_VALUE_1];
    const int kh = pools[NUM_VALUE_0];
    const int kw = pools[NUM_VALUE_1];
    const int outHeight = CeilDiv(inDimH + paddingTop + paddingBottom - kh + 1, strideH);
    const int outWidth = CeilDiv(inDimW + paddingLeft + paddingRight - kw + 1, strideW);
    const std::vector<int64_t> outShape = {operand.GetShape()[NUM_VALUE_0], operand.GetShape()[NUM_VALUE_1], outHeight,
                                           outWidth, operand.GetShape()[NUM_VALUE_4]};
    Tensor result(operand.GetStorage()->tensor->datatype, outShape);
    CALL(Maxpool, *Program::GetInstance().GetCurrentFunction(), operand.GetStorage(), result.GetStorage(), pools,
         strides, paddings);

    return result;
}

static std::vector<int64_t> GetLineReduceWorkspaceShape(const std::vector<int64_t>& sourceShape, int axis)
{
    return {(sourceShape[axis] + 1) / NUM_VALUE_2,
            (sourceShape[sourceShape.size() - 1] + BLOCK_NUM - 1) / BLOCK_NUM * BLOCK_NUM};
}

static int64_t GetOnlineSoftmaxFp32AlignedColumns(int64_t columns)
{
    return AlignUp(columns, BLOCK_SIZE / BytesOf(DT_FP32));
}

static std::vector<int64_t> GetOnlineSoftmaxUpdateWorkspaceShape(const std::vector<int64_t>& outputShape)
{
    return {outputShape[0] + NUM_VALUE_3, GetOnlineSoftmaxFp32AlignedColumns(outputShape[1])};
}

static void CheckOnlineSoftmaxScores(const Tensor& scores)
{
    CheckTensorDataType(scores.GetStorage(), {DT_FP32}, "ONLINE_SOFTMAX");
    CheckTensorDimRange(scores.GetStorage(), NUM_VALUE_2, NUM_VALUE_2, "ONLINE_SOFTMAX");
    CheckTensorShapeSize(scores.GetStorage(), "ONLINE_SOFTMAX");
}

std::tuple<Tensor, Tensor, Tensor> OnlineSoftmax(const Tensor& scores, float scale)
{
    DECLARE_TRACER();
    CheckSupportedNPUArch(ONLINE_SOFTMAX_SUPPORTED_ARCHITECTURES, "OnlineSoftmax");
    CheckOnlineSoftmaxScores(scores);

    auto columnStatisticShape = scores.GetShape();
    columnStatisticShape[0] = 1;
    Tensor expScoresBf16(DataType::DT_BF16, scores.GetShape());
    Tensor columnMax(DataType::DT_FP32, columnStatisticShape);
    Tensor columnSum(DataType::DT_FP32, columnStatisticShape);
    Tensor scaledScores(DataType::DT_FP32, scores.GetShape());
    Tensor reduceWorkspace(DataType::DT_FP32, GetLineReduceWorkspaceShape(scores.GetShape(), 0));

    auto scoresValidShape = scores.GetStorage()->GetDynValidShape();
    if (!scoresValidShape.empty()) {
        auto columnStatisticValidShape = scoresValidShape;
        columnStatisticValidShape[0] = SymbolicScalar(1);
        expScoresBf16.GetStorage()->UpdateDynValidShape(scoresValidShape);
        columnMax.GetStorage()->UpdateDynValidShape(columnStatisticValidShape);
        columnSum.GetStorage()->UpdateDynValidShape(columnStatisticValidShape);
        scaledScores.GetStorage()->UpdateDynValidShape(scoresValidShape);
        reduceWorkspace.GetStorage()->UpdateDynValidShape(SymbolicScalar::FromConcrete(reduceWorkspace.GetShape()));
    }

    auto& op = Program::GetInstance().GetCurrentFunction()->AddOperation(
        Opcode::OP_ONLINE_SOFTMAX, {scores.GetStorage()},
        {expScoresBf16.GetStorage(), columnMax.GetStorage(), columnSum.GetStorage(), scaledScores.GetStorage(),
         reduceWorkspace.GetStorage()});
    op.SetAttribute(OpAttributeKey::scalar, Element(DataType::DT_FP32, scale));
    op.SetAttribute(OpAttributeKey::excludeBufferReuse, true);
    return {expScoresBf16, columnMax, columnSum};
}

static void CheckOnlineSoftmaxUpdateStats(const Tensor& previousMax, const Tensor& previousSum,
                                          const Tensor& previousOutput, const Tensor& currentMax,
                                          const Tensor& currentSum, const Tensor& currentOutput)
{
    const std::vector<Tensor> inputTensors = {previousMax, previousSum, previousOutput,
                                              currentMax,  currentSum,  currentOutput};
    for (const auto& tensor : inputTensors) {
        CheckTensorDataType(tensor.GetStorage(), {DT_FP32}, "ONLINE_SOFTMAX_UPDATE");
        CheckTensorDimRange(tensor.GetStorage(), NUM_VALUE_2, NUM_VALUE_2, "ONLINE_SOFTMAX_UPDATE");
        CheckTensorShapeSize(tensor.GetStorage(), "ONLINE_SOFTMAX_UPDATE");
    }
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, previousOutput.GetShape() == currentOutput.GetShape())
        << "ONLINE_SOFTMAX_UPDATE requires previousOutput and currentOutput to have the same shape.";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, previousMax.GetShape() == previousSum.GetShape() &&
                                                   previousMax.GetShape() == currentMax.GetShape() &&
                                                   previousMax.GetShape() == currentSum.GetShape())
        << "ONLINE_SOFTMAX_UPDATE requires previous/current max and sum tensors to have the same shape.";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID,
           previousMax.GetShape()[0] == 1 && previousMax.GetShape()[1] == previousOutput.GetShape()[1])
        << "ONLINE_SOFTMAX_UPDATE requires max/sum tensors with shape [1, q_len].";
}

std::tuple<Tensor, Tensor, Tensor> OnlineSoftmaxUpdate(const Tensor& previousMax, const Tensor& previousSum,
                                                       const Tensor& previousOutput, const Tensor& currentMax,
                                                       const Tensor& currentSum, const Tensor& currentOutput)
{
    DECLARE_TRACER();
    CheckSupportedNPUArch(ONLINE_SOFTMAX_SUPPORTED_ARCHITECTURES, "OnlineSoftmaxUpdate");
    CheckOnlineSoftmaxUpdateStats(previousMax, previousSum, previousOutput, currentMax, currentSum, currentOutput);

    Tensor updatedMax(DataType::DT_FP32, previousMax.GetShape());
    Tensor updatedSum(DataType::DT_FP32, previousSum.GetShape());
    Tensor updatedOutput(DataType::DT_FP32, previousOutput.GetShape());
    Tensor updateWorkspace(DataType::DT_FP32, GetOnlineSoftmaxUpdateWorkspaceShape(previousOutput.GetShape()));
    updatedMax.GetStorage()->UpdateDynValidShape(previousMax.GetStorage()->GetDynValidShape());
    updatedSum.GetStorage()->UpdateDynValidShape(previousSum.GetStorage()->GetDynValidShape());
    updatedOutput.GetStorage()->UpdateDynValidShape(previousOutput.GetStorage()->GetDynValidShape());
    updateWorkspace.GetStorage()->UpdateDynValidShape(SymbolicScalar::FromConcrete(updateWorkspace.GetShape()));

    auto& op = Program::GetInstance().GetCurrentFunction()->AddOperation(
        Opcode::OP_ONLINE_SOFTMAX_UPDATE,
        {previousMax.GetStorage(), previousSum.GetStorage(), previousOutput.GetStorage(), currentMax.GetStorage(),
         currentSum.GetStorage(), currentOutput.GetStorage()},
        {updatedMax.GetStorage(), updatedSum.GetStorage(), updatedOutput.GetStorage(), updateWorkspace.GetStorage()});
    op.SetAttribute(OpAttributeKey::excludeBufferReuse, true);
    return {updatedMax, updatedSum, updatedOutput};
}

static void CheckOnlineSoftmaxTileShape(const std::vector<int64_t>& viewShape, const VecTile& vecTile,
                                        const std::string& opName)
{
    ASSERT(VectorErrorCode::ERR_CONFIG_TILE, vecTile.size() >= viewShape.size())
        << opName << " expects vec tile rank >= tensor rank.";
    ASSERT(VectorErrorCode::ERR_CONFIG_TILE, viewShape[0] <= vecTile[0])
        << opName << " v1 does not split reduce/head axis; shape[0] (" << viewShape[0] << ") must be <= vec tile[0] ("
        << vecTile[0] << ").";
}

void OnlineSoftmaxOperationTileFunc(Function& function, const TileShape& tileShape,
                                    const std::vector<LogicalTensorPtr>& iOperand,
                                    const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, iOperand.size() == 1 && oOperand.size() == NUM_VALUE_5)
        << "ONLINE_SOFTMAX expects 1 input and 5 outputs including workspace tensors.";
    auto& vecTile = tileShape.GetVecTile();
    auto scores = iOperand[0];
    auto expScoresBf16 = oOperand[0];
    auto columnMax = oOperand[1];
    auto columnSum = oOperand[2];
    auto scaledScores = oOperand[NUM_VALUE_3];
    auto reduceWorkspace = oOperand[NUM_VALUE_4];
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, scores->shape.size() == NUM_VALUE_2)
        << "ONLINE_SOFTMAX only supports 2D tensor in v1.";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID,
           expScoresBf16->shape == scores->shape && scaledScores->shape == scores->shape)
        << "ONLINE_SOFTMAX requires expScoresBf16/scaledScores shape to match scores shape.";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID,
           columnMax->shape == columnSum->shape && columnMax->shape[0] == 1 && columnMax->shape[1] == scores->shape[1])
        << "ONLINE_SOFTMAX requires columnMax/columnSum shape [1, q_len].";
    CheckOnlineSoftmaxTileShape(scores->shape, vecTile, "ONLINE_SOFTMAX");

    TileInfo scoreTileInfo(scores->shape, std::vector<int64_t>(scores->shape.size(), 0));
    TileInfo statisticTileInfo(columnMax->shape, std::vector<int64_t>(columnMax->shape.size(), 0));
    for (int64_t col = 0; col < scores->shape[1]; col += vecTile[1]) {
        scoreTileInfo.offset = {0, col};
        scoreTileInfo.shape = {scores->shape[0], std::min<int64_t>(vecTile[1], scores->shape[1] - col)};
        statisticTileInfo.offset = {0, col};
        statisticTileInfo.shape = {1, scoreTileInfo.shape[1]};
        auto reduceWorkspaceShape = GetLineReduceWorkspaceShape(scoreTileInfo.shape, 0);
        auto reduceWorkspaceOffset = std::vector<int64_t>{0, col};
        auto scoreTile = scores->View(function, scoreTileInfo.shape, scoreTileInfo.offset);
        auto expScoresTile = expScoresBf16->View(function, scoreTileInfo.shape, scoreTileInfo.offset);
        auto columnMaxTile = columnMax->View(function, statisticTileInfo.shape, statisticTileInfo.offset);
        auto columnSumTile = columnSum->View(function, statisticTileInfo.shape, statisticTileInfo.offset);
        auto scaledScoresTile = scaledScores->View(function, scoreTileInfo.shape, scoreTileInfo.offset);
        auto reduceWorkspaceTile = reduceWorkspace->View(function, reduceWorkspaceShape, reduceWorkspaceOffset);
        auto& tileOp = function.AddOperation(
            Opcode::OP_ONLINE_SOFTMAX, {scoreTile},
            {expScoresTile, columnMaxTile, columnSumTile, scaledScoresTile, reduceWorkspaceTile});
        tileOp.SetAttribute(OpAttributeKey::scalar, op.GetElementAttribute(OpAttributeKey::scalar));
        tileOp.SetAttribute(OpAttributeKey::excludeBufferReuse, true);
    }
}

struct OnlineSoftmaxUpdateTileOperands {
    LogicalTensorPtr previousMax;
    LogicalTensorPtr previousSum;
    LogicalTensorPtr previousOutput;
    LogicalTensorPtr currentMax;
    LogicalTensorPtr currentSum;
    LogicalTensorPtr currentOutput;
    LogicalTensorPtr updatedMax;
    LogicalTensorPtr updatedSum;
    LogicalTensorPtr updatedOutput;
    LogicalTensorPtr updateWorkspace;
};

static OnlineSoftmaxUpdateTileOperands GetOnlineSoftmaxUpdateTileOperands(const std::vector<LogicalTensorPtr>& iOperand,
                                                                          const std::vector<LogicalTensorPtr>& oOperand)
{
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, iOperand.size() == NUM_VALUE_6 && oOperand.size() == NUM_VALUE_4)
        << "ONLINE_SOFTMAX_UPDATE expects 6 inputs and 4 outputs including workspace tensor.";
    return {iOperand[0],           iOperand[1], iOperand[NUM_VALUE_2], iOperand[NUM_VALUE_3], iOperand[NUM_VALUE_4],
            iOperand[NUM_VALUE_5], oOperand[0], oOperand[1],           oOperand[NUM_VALUE_2], oOperand[NUM_VALUE_3]};
}

static void CheckOnlineSoftmaxUpdateTileOperands(const OnlineSoftmaxUpdateTileOperands& tensors, const VecTile& vecTile)
{
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, tensors.previousOutput->shape.size() == NUM_VALUE_2)
        << "ONLINE_SOFTMAX_UPDATE only supports 2D O tensor in v1.";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, tensors.currentOutput->shape == tensors.previousOutput->shape &&
                                                   tensors.updatedOutput->shape == tensors.previousOutput->shape)
        << "ONLINE_SOFTMAX_UPDATE requires previous/current/updated output tensors to have the same shape.";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, tensors.previousMax->shape == tensors.previousSum->shape &&
                                                   tensors.previousMax->shape == tensors.currentMax->shape &&
                                                   tensors.previousMax->shape == tensors.currentSum->shape &&
                                                   tensors.previousMax->shape == tensors.updatedMax->shape &&
                                                   tensors.previousMax->shape == tensors.updatedSum->shape &&
                                                   tensors.previousMax->shape[0] == 1 &&
                                                   tensors.previousMax->shape[1] == tensors.previousOutput->shape[1])
        << "ONLINE_SOFTMAX_UPDATE requires all statistic tensors with shape [1, q_len].";
    ASSERT(
        VectorErrorCode::ERR_PARAM_INVALID,
        tensors.updateWorkspace->shape.size() == NUM_VALUE_2 &&
            tensors.updateWorkspace->shape[0] == tensors.previousOutput->shape[0] + NUM_VALUE_3 &&
            tensors.updateWorkspace->shape[1] == GetOnlineSoftmaxFp32AlignedColumns(tensors.previousOutput->shape[1]))
        << "ONLINE_SOFTMAX_UPDATE requires workspace tensor with shape [head_dim + 3, aligned_q_len].";
    CheckOnlineSoftmaxTileShape(tensors.previousOutput->shape, vecTile, "ONLINE_SOFTMAX_UPDATE");
    ASSERT(VectorErrorCode::ERR_CONFIG_TILE, vecTile[1] % (BLOCK_SIZE / BytesOf(DT_FP32)) == 0)
        << "ONLINE_SOFTMAX_UPDATE requires vec tile last dim to be FP32 32-byte aligned.";
}

static void AddOnlineSoftmaxUpdateTileOp(Function& function, const OnlineSoftmaxUpdateTileOperands& tensors,
                                         int64_t col, int64_t tileColumns)
{
    const std::vector<int64_t> outputTileShape = {tensors.previousOutput->shape[0], tileColumns};
    const std::vector<int64_t> outputTileOffset = {0, col};
    const std::vector<int64_t> statisticTileShape = {1, tileColumns};
    const std::vector<int64_t> statisticTileOffset = {0, col};
    auto previousMaxTile = tensors.previousMax->View(function, statisticTileShape, statisticTileOffset);
    auto previousSumTile = tensors.previousSum->View(function, statisticTileShape, statisticTileOffset);
    auto previousOutputTile = tensors.previousOutput->View(function, outputTileShape, outputTileOffset);
    auto currentMaxTile = tensors.currentMax->View(function, statisticTileShape, statisticTileOffset);
    auto currentSumTile = tensors.currentSum->View(function, statisticTileShape, statisticTileOffset);
    auto currentOutputTile = tensors.currentOutput->View(function, outputTileShape, outputTileOffset);
    auto updatedMaxTile = tensors.updatedMax->View(function, statisticTileShape, statisticTileOffset);
    auto updatedSumTile = tensors.updatedSum->View(function, statisticTileShape, statisticTileOffset);
    auto updatedOutputTile = tensors.updatedOutput->View(function, outputTileShape, outputTileOffset);
    auto alignedTileColumns = GetOnlineSoftmaxFp32AlignedColumns(tileColumns);
    auto updateWorkspaceTile = tensors.updateWorkspace->View(
        function, {outputTileShape[0] + NUM_VALUE_3, alignedTileColumns}, {0, col});
    auto& tileOp = function.AddOperation(
        Opcode::OP_ONLINE_SOFTMAX_UPDATE,
        {previousMaxTile, previousSumTile, previousOutputTile, currentMaxTile, currentSumTile, currentOutputTile},
        {updatedMaxTile, updatedSumTile, updatedOutputTile, updateWorkspaceTile});
    tileOp.SetAttribute(OpAttributeKey::excludeBufferReuse, true);
}

void OnlineSoftmaxUpdateOperationTileFunc(Function& function, const TileShape& tileShape,
                                          const std::vector<LogicalTensorPtr>& iOperand,
                                          const std::vector<LogicalTensorPtr>& oOperand,
                                          [[maybe_unused]] const Operation& op)
{
    const auto& vecTile = tileShape.GetVecTile();
    auto tensors = GetOnlineSoftmaxUpdateTileOperands(iOperand, oOperand);
    CheckOnlineSoftmaxUpdateTileOperands(tensors, vecTile);
    for (int64_t col = 0; col < tensors.previousOutput->shape[1]; col += vecTile[1]) {
        auto tileColumns = std::min<int64_t>(vecTile[1], tensors.previousOutput->shape[1] - col);
        AddOnlineSoftmaxUpdateTileOp(function, tensors, col, tileColumns);
    }
}

REGISTER_OPERATION_TILED_FUNC(OP_ONLINE_SOFTMAX, Opcode::OP_ONLINE_SOFTMAX, OnlineSoftmaxOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ONLINE_SOFTMAX_UPDATE, Opcode::OP_ONLINE_SOFTMAX_UPDATE,
                              OnlineSoftmaxUpdateOperationTileFunc);

Tensor Compact(const Tensor& operand)
{
    DECLARE_TRACER();

    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, operand.GetShape().size() == operand.GetStorage()->offset.size())
        << "operand's shape size and offset size should be equal";
    Tensor result(operand.GetStorage()->tensor->datatype, {operand.GetShape()[0], 1});
    Tensor workspace(operand.GetStorage()->tensor->datatype, {operand.GetShape()[0], NUM_VALUE_8});
    Program::GetInstance().AddOperation(Opcode::OP_COMPACT, {operand.GetStorage()}, {result.GetStorage()});
    return result;
}

void experimental::Print(SymbolicScalar cond, const std::string& format, const std::vector<Tensor>& tensors,
                         const std::vector<SymbolicScalar>& scalars)
{
    auto function = Program::GetInstance().GetCurrentFunction();
    std::vector<LogicalTensorPtr> inputs;
    for (auto& t : tensors) {
        inputs.push_back(t.GetStorage());
    }
    auto& op = function->AddOperation(Opcode::OP_PRINT, inputs, {});
    op.SetAttr(OP_ATTR_PREFIX + "format", format);
    op.SetAttr(OP_ATTR_PREFIX + "scalars", scalars);
    op.SetAttribute(OP_ATTR_PREFIX + "cond", cond);
}

void ToFile(const Tensor& operand, const std::string& fname, const std::vector<SymbolicScalar>& scalars,
            SymbolicScalar cond)
{
    auto function = Program::GetInstance().GetCurrentFunction();
    auto& op = function->AddOperation(Opcode::OP_PRINT, {operand.GetStorage()}, {});
    CHECK_OP(!fname.empty()) << "Invalid file name";
    op.SetAttribute(OP_ATTR_PREFIX + "fname", fname);
    op.SetAttribute(OP_ATTR_PREFIX + "scalars", scalars);
    op.SetAttribute(OP_ATTR_PREFIX + "cond", cond);
}

Tensor Unsqueeze(const Tensor& old, int unsqueezeDimNum)
{
    DECLARE_TRACER();

    CHECK_OP(unsqueezeDimNum < static_cast<int>(old.GetShape().size()) + 1 &&
             unsqueezeDimNum >= -static_cast<int>(old.GetShape().size()) - 1);
    size_t unsqueezeDim = unsqueezeDimNum;
    if (unsqueezeDimNum < 0) {
        unsqueezeDim = unsqueezeDimNum + old.GetShape().size() + 1;
    }
    std::vector<int64_t> newShape(old.GetStorage()->shape);
    newShape.insert(newShape.begin() + unsqueezeDim, 1);
    auto validShape = old.GetStorage()->GetDynValidShape();
    CHECK_OP(!validShape.empty());
    validShape.insert(validShape.begin() + unsqueezeDim, 1);
    return Reshape(old, newShape, validShape);
}

static void SqueezeParamsValidCheck(const Tensor& input, std::vector<int>& dim)
{
    Shape oriShape = input.GetShape();
    size_t shapeSize = oriShape.size();
    CHECK_OP(shapeSize <= SHAPE_DIM4) << "The input dimension only supports 1~4. Cur dimension is " << shapeSize;

    if (dim.empty()) {
        for (size_t i = 0; i < shapeSize; i++) {
            dim.push_back(static_cast<int>(i));
        }
    }
    CHECK_OP(dim.size() <= shapeSize) << "The dim.size <= input.dim is not matched. dim.size is " << dim.size()
                                      << ", input.dim is " << shapeSize;
    std::set<int> dupDimSet(dim.begin(), dim.end());
    CHECK_OP(dupDimSet.size() == dim.size()) << "There are duplicate elements in dim";
    for (size_t i = 0; i < dim.size(); i++) {
        CHECK_OP(dim[i] < static_cast<int>(shapeSize) && dim[i] >= -(static_cast<int>(shapeSize)))
            << "dim " << i << " in dim is out of range";
        if (dim[i] < 0) {
            dim[i] = dim[i] + static_cast<int>(shapeSize);
        }
    }
    std::sort(dim.begin(), dim.end());
}

Tensor Squeeze(const Tensor& input, const std::vector<int>& dim)
{
    DECLARE_TRACER();

    Shape oriShape = input.GetShape();
    Shape dstShape(oriShape.begin(), oriShape.end());
    size_t shapeSize = oriShape.size();
    std::vector<SymbolicScalar> validShape;
    std::vector<int> innerDim(dim.begin(), dim.end());

    if (shapeSize == 1) {
        return input;
    }
    SqueezeParamsValidCheck(input, innerDim);
    for (auto shape : input.GetStorage()->GetDynValidShape()) {
        validShape.push_back(shape);
    }

    CHECK_OP(!validShape.empty()) << "The input validshape should not be empty.";

    for (auto it = innerDim.rbegin(); it != innerDim.rend(); ++it) {
        int axis = *it;
        if (oriShape[axis] == 1) {
            dstShape.erase(dstShape.begin() + axis);
            validShape.erase(validShape.begin() + axis);
        }
    }
    if (dstShape.empty()) {
        dstShape.push_back(1);
    }
    if (validShape.empty()) {
        validShape.push_back(1);
    }

    if (dstShape.size() == shapeSize) {
        return input;
    } else {
        return Reshape(input, dstShape, validShape);
    }
}

void TensorInnerAssign(Function& function, const LogicalTensorPtr& operand, const LogicalTensorPtr& result)
{
    function.AddOperation(Opcode::OP_REGISTER_COPY, {operand}, {result});
}

Tensor Assign(const Tensor& operand)
{
    Tensor result(operand.GetStorage()->Datatype(), operand.GetShape(), "", operand.Format());
    result.GetStorage()->UpdateDynValidShape(operand.GetStorage()->GetDynValidShape());
    CALL(InnerAssign, *Program::GetInstance().GetCurrentFunction(), operand.GetStorage(), result.GetStorage());
    return result;
}

#define CALL(n, ...) Tensor##n(__VA_ARGS__)

void TiledInnerRegisterCopy(const int dimIdx, Function& function, const TileShape& tileShape,
                            const LogicalTensorPtr& operand, const LogicalTensorPtr& result,
                            std::vector<int64_t> actTileShape, std::vector<int64_t> actOffset)
{
    if (static_cast<size_t>(dimIdx) == result->GetShape().size()) {
        auto inputTile = operand->View(function, actTileShape, actOffset);
        auto resultTile = result->View(function, actTileShape, actOffset);
        function.AddOperation("TILE_REGISTER_COPY", {inputTile}, {resultTile});
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    CheckFwkOpTileShape(vecTile, result, "TILE_REGISTER_COPY");

    for (auto i = 0; i < result->GetShape()[dimIdx]; i += vecTile[dimIdx]) {
        actTileShape[dimIdx] = std::min(result->GetShape()[dimIdx] - i, vecTile[dimIdx]);
        actOffset[dimIdx] = i;
        TiledInnerRegisterCopy(dimIdx + 1, function, tileShape, operand, result, actTileShape, actOffset);
    }
}

void TiledInnerRegisterCopy(Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand,
                            const LogicalTensorPtr& result)
{
    std::vector<int64_t> actOffset(result->GetShape().size(), 0);
    std::vector<int64_t> actTileShape(result->GetShape().size(), 1);
    TiledInnerRegisterCopy(0, function, tileShape, operand, result, actTileShape, actOffset);
}

void TiledInnerCompact(Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand,
                       const LogicalTensorPtr& result)
{
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, operand->shape.size() == operand->offset.size())
        << "operand's shape size and offset size should be equal";
    auto workspace = std::make_shared<LogicalTensor>(function, operand->tensor->datatype,
                                                     std::vector<int64_t>{operand->shape[0], NUM_VALUE_8});

    // 目前只支持2维操作
    if (operand->shape.size() != 2) {
        ASSERT(VectorErrorCode::ERR_PARAM_SHAPE_DIM_UNSUPPORTED, false) << "unsupported dimension";
    }
    auto& vecTile = tileShape.GetVecTile();
    int tileShape1 = std::min(operand->shape[1], vecTile[1]);
    for (int i = 0; i < operand->shape[0]; i += vecTile[0]) {
        int tileShape0 = std::min(operand->shape[0] - i, vecTile[0]);
        auto inputTile = operand->View(function, {tileShape0, tileShape1}, {i, 0});
        auto resultTile = result->View(function, {tileShape0, 1}, {i, 0});
        auto workspaceTile = workspace->View(function, {tileShape0, NUM_VALUE_8}, {i, 0});
        function.AddOperation("TILE_COMPACT", {inputTile, workspaceTile}, {resultTile});
    }
}

void TensorInnerCompact(Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand,
                        const LogicalTensorPtr& result)
{
    TiledInnerCompact(function, tileShape, operand, result);
}

Tensor NewCompact(const Tensor& operand)
{
    DECLARE_TRACER();

    Tensor result(operand.GetStorage()->tensor->datatype, {operand.GetShape()[0], 1});
    CALL(InnerCompact, *Program::GetInstance().GetCurrentFunction(), TileShape::Current(), operand.GetStorage(),
         result.GetStorage());
    return result;
}

/* Begin: Start for Reduce*/

Tensor Reduce(const std::vector<Tensor>& aggregation, const ReduceMode reduceMode)
{
    DECLARE_TRACER();
    // Support Reduce::Add only
    if (reduceMode != ReduceMode::ATOMIC_ADD) {
        return Tensor();
    }
    std::vector<LogicalTensorPtr> iOperand;
    std::vector<LogicalTensorPtr> oOperand;
    iOperand.reserve(aggregation.size());
    std::transform(aggregation.begin(), aggregation.end(), std::back_inserter(iOperand),
                   [](const Tensor& elem) { return elem.GetStorage(); });
    auto o0 = iOperand[0];
    Tensor result(o0->Datatype(), o0->shape, "", o0->Format());
    auto& op = Program::GetInstance().AddOperation(Opcode::OP_REDUCE_ACC, iOperand, {result.GetStorage()});
    op.SetAttribute(Matrix::ACC_A_MUL_B, 1);
    return result;
}

void TiledReduceAcc(Function& function, const TileShape& tileShape, size_t cur, std::vector<Input> inputVec,
                    const LogicalTensorPtr& result, TileInfo& resultTileInfo)
{
    if (cur == inputVec[0].tensor.GetShape().size()) {
        std::vector<LogicalTensorPtr> inputTileVec;
        for (size_t index = 0; index < inputVec.size(); ++index) {
            auto inputTile = inputVec[index].tensor.GetStorage()->View(function, inputVec[index].tileInfo.shape,
                                                                       inputVec[index].tileInfo.offset);
            inputTileVec.emplace_back(inputTile);
        }

        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto& op = function.AddOperation(Opcode::OP_REDUCE_ACC, inputTileVec, {resultTile});
        op.SetAttribute(Matrix::ACC_A_MUL_B, 1);
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (auto i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        for (size_t index = 0; index < inputVec.size(); ++index) {
            inputVec[index].tileInfo.offset[cur] = i % inputVec[index].tensor.GetShape()[cur];
            inputVec[index].tileInfo.shape[cur] = std::min(
                inputVec[index].tensor.GetShape()[cur] - inputVec[index].tileInfo.offset[cur], vecTile[cur]);
        }
        TiledReduceAcc(function, tileShape, cur + 1, inputVec, result, resultTileInfo);
    }
}

void TiledReduceAcc(Function& function, const TileShape& tileShape, std::vector<LogicalTensorPtr> operandVec,
                    const LogicalTensorPtr& result)
{
    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo tileInfo2(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    std::vector<Input> inputVec;
    for (size_t index = 0; index < operandVec.size(); ++index) {
        TileInfo tileInfo(result->shape.size(), result->offset.size());
        Input input = Input{operandVec[index], tileInfo};
        inputVec.push_back(input);
    }
    TiledReduceAcc(function, tileShape, 0, inputVec, result, resultTileInfo);
}

// parallel sort
const std::string SORT_ORDER = OP_ATTR_PREFIX + "order";
const std::string SORT_START_INDEX = OP_ATTR_PREFIX + "start_index";
const std::string SORT_FULL = OP_ATTR_PREFIX + "full_sort";

void TiledSort(Function& function, const LogicalTensorPtr& x, const LogicalTensorPtr& y, const LogicalTensorPtr& yIdx,
               const LogicalTensorPtr& temp, int idxStart, int descending)
{
    auto& op = function.AddOperation(Opcode::OP_SORT, {x}, {y, yIdx, temp});
    op.SetAttribute(SORT_START_INDEX, static_cast<int>(idxStart));
    op.SetAttribute(SORT_ORDER, static_cast<int>(descending));
    std::map<int, int> inplaceInfo = {{0, 0}};
    op.SetAttr(OpAttributeKey::inplaceInfo, inplaceInfo);
}

std::tuple<Tensor, Tensor, Tensor> L1Sort(const Tensor& x, int idxStart, bool descending)
{
    constexpr int32_t kFactorSize = NUM_VALUE_4;
    auto tempShape = x.GetShape();
    tempShape[1] *= kFactorSize;
    auto y = Tensor(x.GetStorage()->tensor->datatype, x.GetShape());
    auto yIdx = Tensor(DataType::DT_INT32, x.GetShape());
    auto temp = Tensor(x.GetStorage()->tensor->datatype, tempShape);
    TiledSort(*Program::GetInstance().GetCurrentFunction(), x.GetStorage(), y.GetStorage(), yIdx.GetStorage(),
              temp.GetStorage(), idxStart, descending);
    return std::tie(y, yIdx, temp);
}

void TiledCompareAndSwap(Function& function, const LogicalTensorPtr& x0, const LogicalTensorPtr& idx0,
                         const LogicalTensorPtr& x1, const LogicalTensorPtr& idx1, const LogicalTensorPtr& y0,
                         const LogicalTensorPtr& yIdx0, const LogicalTensorPtr& y1, const LogicalTensorPtr& yIdx1,
                         int descending)
{
    auto& op = function.AddOperation(Opcode::OP_COMPARE_SWAP, {x0, idx0, x1, idx1}, {y0, yIdx0, y1, yIdx1});
    op.SetAttribute(SORT_ORDER, static_cast<int>(descending));
    std::map<int, int> inplaceInfo = {{0, 0}, {1, 1}};
    op.SetAttr(OpAttributeKey::inplaceInfo, inplaceInfo);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> L1CompareAndSwap(const Tensor& x0, const Tensor& idx0, const Tensor& x1,
                                                            const Tensor& idx1, bool descending)
{
    Tensor y0(x0.GetStorage()->Datatype(), x0.GetShape());
    Tensor yIdx0(idx0.GetStorage()->Datatype(), idx0.GetShape());
    Tensor y1(x1.GetStorage()->Datatype(), x1.GetShape());
    Tensor yIdx1(idx1.GetStorage()->Datatype(), idx1.GetShape());
    TiledCompareAndSwap(*Program::GetInstance().GetCurrentFunction(), x0.GetStorage(), idx0.GetStorage(),
                        x1.GetStorage(), idx1.GetStorage(), y0.GetStorage(), yIdx0.GetStorage(), y1.GetStorage(),
                        yIdx1.GetStorage(), descending);
    return std::tie(y0, yIdx0, y1, yIdx1);
}

void TiledMerge(Function& function, const LogicalTensorPtr& x, const LogicalTensorPtr& idx, const LogicalTensorPtr& y,
                const LogicalTensorPtr& yIdx, const LogicalTensorPtr& temp, int fullSort, int descending)
{
    auto& op = function.AddOperation(Opcode::OP_MERGE, {x, idx}, {y, yIdx, temp});
    op.SetAttribute(SORT_ORDER, static_cast<int>(descending));
    op.SetAttribute(SORT_FULL, static_cast<int>(fullSort));
    std::map<int, int> inplaceInfo = {{0, 0}, {1, 1}};
    op.SetAttr(OpAttributeKey::inplaceInfo, inplaceInfo);
}

std::tuple<Tensor, Tensor, Tensor> L1Merge(const Tensor& x, const Tensor& idx, bool descending, bool fullSort)
{
    constexpr int32_t kFactorSize = NUM_VALUE_4;
    auto tempShape = x.GetShape();
    tempShape[1] *= kFactorSize;
    auto y = Tensor(x.GetStorage()->tensor->datatype, x.GetShape());
    auto yIdx = Tensor(idx.GetStorage()->tensor->datatype, idx.GetShape());
    auto temp = Tensor(x.GetStorage()->tensor->datatype, tempShape);
    TiledMerge(*Program::GetInstance().GetCurrentFunction(), x.GetStorage(), idx.GetStorage(), y.GetStorage(),
               yIdx.GetStorage(), temp.GetStorage(), fullSort, descending);
    return std::tie(y, yIdx, temp);
}

using SortTileMap = std::map<int, std::tuple<Tensor, Tensor>>;

bool IsMaxTile(SortTileMap& map, int index) { return map.find(index) == map.end(); }

void CompareAndSwapStep(SortTileMap& tileMap, int offset, int mergeSize, bool descending)
{
    int nTile = mergeSize;
    for (int step = nTile; step >= NUM2; step /= NUM2) {
        for (int start = 0; start < nTile * NUM2; start += step * NUM2) {
            // within each swap size = step * tileSize
            for (int i = 0; i < step; i++) {
                int idx0 = offset + start + i;
                int idx1 = idx0 + step;

                // no need to comp & swap
                if (IsMaxTile(tileMap, idx0) && descending) {
                    continue;
                }
                if (IsMaxTile(tileMap, idx1) && !descending) {
                    continue;
                }
                if (IsMaxTile(tileMap, idx0) && !descending) {
                    tileMap[idx0] = tileMap[idx1];
                    tileMap.erase(idx1);
                    continue;
                } else if (IsMaxTile(tileMap, idx1) && descending) {
                    tileMap[idx1] = tileMap[idx0];
                    tileMap.erase(idx0);
                    continue;
                }

                // use L1CompareAndSwap
                auto [x0, xIdx0] = tileMap[idx0];
                auto [x1, xIdx1] = tileMap[idx1];
                auto [y0, yIdx0, y1, yIdx1] = L1CompareAndSwap(x0, xIdx0, x1, xIdx1, descending);
                tileMap[idx0] = std::tie(y0, yIdx0);
                tileMap[idx1] = std::tie(y1, yIdx1);
            }
        }
    }
}

void MergeStep(SortTileMap& tileMap, int offset, int mergeSize, int tileSize, bool descending)
{
    // Compare & Swap
    CompareAndSwapStep(tileMap, offset, mergeSize, descending);

    // maxStep is the minimum orders of 2 that >= n
    int n = tileMap.size() / NUM2;
    int maxStep = 1;
    while (maxStep < n) {
        maxStep <<= 1;
    }
    int halfSize = tileSize / NUM2;

    // Merge within each tile
    for (int i = 0; i < mergeSize; i++) {
        int idx0 = offset + NUM2 * i;
        int idx1 = idx0 + 1;
        if (IsMaxTile(tileMap, idx0) || IsMaxTile(tileMap, idx1)) {
            continue;
        }
        auto [x0, xIdx0] = tileMap[idx0];
        auto [x1, xIdx1] = tileMap[idx1];
        Tensor src(x0.GetDataType(), {1, tileSize});
        Tensor srcIdx(DT_INT32, {1, tileSize});
        Assemble(x0, {0, 0}, src);
        Assemble(x1, {0, halfSize}, src);
        Assemble(xIdx0, {0, 0}, srcIdx);
        Assemble(xIdx1, {0, halfSize}, srcIdx);
        auto mergeResult = L1Merge(src, srcIdx, descending, false);
        auto res = std::get<0>(mergeResult);
        auto resIdx = std::get<1>(mergeResult);

        if (mergeSize < maxStep) {
            tileMap[idx0] = {View(res, {1, halfSize}, {0, 0}), View(resIdx, {1, halfSize}, {0, 0})};
            tileMap[idx1] = {View(res, {1, halfSize}, {0, halfSize}), View(resIdx, {1, halfSize}, {0, halfSize})};
        } else {
            // For assemble, no need to split into half
            tileMap[idx0] = {res, resIdx};
        }
    }
}

bool IsPowerOfTwo(int n) { return (n & (n - 1)) == 0; }

int NextPowerofTwo(int n)
{
    int power = 1;
    while (power < n) {
        power <<= 1;
    }
    return power;
}

std::tuple<Tensor, Tensor> Sort(const Tensor& x, bool descending)
{
    DECLARE_TRACER();
    FE_ASSERT(x.GetShape().size() == NUM2);
    FE_ASSERT(x.GetShape()[0] == 1);
    auto& vecTile = TileShape::Current().GetVecTile();
    FE_ASSERT(vecTile.size() == NUM2);
    FE_ASSERT(vecTile[0] == 1);
    auto tileSize = vecTile[1];
    FE_ASSERT(IsPowerOfTwo(tileSize));
    int length = x.GetShape()[1];
    int padLength = NextPowerofTwo(length);

    int nTile = padLength / tileSize;
    int halfSize = tileSize / NUM2;
    SortTileMap tileMap;

    if (nTile <= 1) {
        auto res = L1Sort(x, 0, descending);
        auto y = std::get<0>(res);
        auto yIdx = std::get<1>(res);
        return std::tie(y, yIdx);
    }

    // Tile Sort
    for (int i = 0; i < nTile; i++) {
        bool flag = (i % NUM2 == (descending ? 0 : 1));
        int idxStart = i;
        auto src = View(x, {1, tileSize}, {0, tileSize * i});
        auto sortResult = L1Sort(src, idxStart, flag);
        auto res = std::get<0>(sortResult);
        auto resIdx = std::get<1>(sortResult);
        tileMap[i * NUM2] = {View(res, {1, halfSize}, {0, 0}), View(resIdx, {1, halfSize}, {0, 0})};
        tileMap[i * NUM2 + 1] = {View(res, {1, halfSize}, {0, halfSize}), View(resIdx, {1, halfSize}, {0, halfSize})};
    }

    // Merge
    for (int step = NUM2; step <= nTile; step *= NUM2) {
        for (int i = 0; i < nTile / step; ++i) {
            int offset = i * step * NUM2;
            bool flag = (i % NUM2 == 0) ? descending : !descending;
            MergeStep(tileMap, offset, step, tileSize, flag);
        }
    }

    // Assemble result
    Tensor y(x.GetDataType(), {1, length});
    Tensor yIdx(DT_INT32, {1, length});
    for (int i = 0; i < nTile; i++) {
        if (IsMaxTile(tileMap, NUM2 * i)) {
            continue;
        }
        auto [res, resIdx] = tileMap[NUM2 * i];
        Assemble(res, {0, i * tileSize}, y);
        Assemble(resIdx, {0, i * tileSize}, yIdx);
    }
    return std::tie(y, yIdx);
}

std::tuple<Tensor, Tensor> SortWithIndex(const Tensor& x, const Tensor& idx, bool descending)
{
    DECLARE_TRACER();
    FE_ASSERT(x.GetShape().size() == NUM2);
    FE_ASSERT(x.GetShape()[0] == 1);
    auto& vecTile = TileShape::Current().GetVecTile();
    FE_ASSERT(vecTile.size() == NUM2);
    FE_ASSERT(vecTile[0] == 1);
    auto tileSize = vecTile[1];
    FE_ASSERT(IsPowerOfTwo(tileSize));
    int length = x.GetShape()[1];
    int padLength = NextPowerofTwo(length);
    int nTile = padLength / tileSize;
    int halfSize = tileSize / NUM2;
    SortTileMap tileMap;

    if (nTile <= 1) {
        auto res = L1Merge(x, idx, descending, true); // L1Sort with index
        auto y = std::get<0>(res);
        auto yIdx = std::get<1>(res);
        return std::tie(y, yIdx);
    }

    // Tile Sort
    for (int i = 0; i < nTile; i++) {
        bool flag = (i % NUM2 == (descending ? 0 : 1));
        auto src = View(x, {1, tileSize}, {0, tileSize * i});
        auto srcIdx = View(idx, {1, tileSize}, {0, tileSize * i});
        auto sortResult = L1Merge(src, srcIdx, flag, true); // L1Sort with index
        auto res = std::get<0>(sortResult);
        auto resIdx = std::get<1>(sortResult);
        tileMap[i * NUM2] = {View(res, {1, halfSize}, {0, 0}), View(resIdx, {1, halfSize}, {0, 0})};
        tileMap[i * NUM2 + 1] = {View(res, {1, halfSize}, {0, halfSize}), View(resIdx, {1, halfSize}, {0, halfSize})};
    }

    // Merge
    for (int step = NUM2; step <= nTile; step *= NUM2) {
        for (int i = 0; i < nTile / step; ++i) {
            int offset = i * step * NUM2;
            bool flag = (i % NUM2 == 0) ? descending : !descending;
            MergeStep(tileMap, offset, step, tileSize, flag);
        }
    }

    // Assemble result
    Tensor y(x.GetDataType(), {1, length});
    Tensor yIdx(idx.GetDataType(), {1, length});
    for (int i = 0; i < nTile; i++) {
        if (IsMaxTile(tileMap, NUM2 * i)) {
            continue;
        }
        auto [res, resIdx] = tileMap[NUM2 * i];
        Assemble(res, {0, i * tileSize}, y);
        Assemble(resIdx, {0, i * tileSize}, yIdx);
    }
    return std::tie(y, yIdx);
}

// topk for ds3.2-Day0
const std::string TOPK_START_INDEX = OP_ATTR_PREFIX + "start_index";
const std::string TOPK_MERGE_SIZE = OP_ATTR_PREFIX + "merge_size";
const std::string TOPK_INDEX = OP_ATTR_PREFIX + "is_index";
const std::string TOPK_K = OP_ATTR_PREFIX + "k";

void TiledTopKSort(Function& function, const LogicalTensorPtr& x, const LogicalTensorPtr& y,
                   const LogicalTensorPtr& temp, const SymbolicScalar& dynValue, int idxStart)
{
    auto& op = function.AddOperation(Opcode::OP_TOPK_SORT, {x}, {y, temp});
    op.SetAttribute(TOPK_START_INDEX, idxStart);
    if (dynValue.IsValid()) {
        op.SetAttribute(OpAttributeKey::dynScalar, dynValue);
    }
}

std::tuple<Tensor, Tensor> TopKSort(const Tensor& x, int idxStart)
{
    constexpr int32_t kFactorSize = NUM_VALUE_2;
    auto shape = x.GetShape();
    shape[1] *= kFactorSize;
    auto y = Tensor(x.GetStorage()->tensor->datatype, shape);
    auto temp = Tensor(x.GetStorage()->tensor->datatype, shape);
    TiledTopKSort(*Program::GetInstance().GetCurrentFunction(), x.GetStorage(), y.GetStorage(), temp.GetStorage(),
                  SymbolicScalar(), idxStart);
    return std::tie(y, temp);
}

std::tuple<Tensor, Tensor> TopKSort(const Tensor& x, const SymbolicScalar& idxStart)
{
    constexpr int32_t kFactorSize = NUM_VALUE_2;
    auto shape = x.GetShape();
    shape[1] *= kFactorSize;
    auto y = Tensor(x.GetStorage()->tensor->datatype, shape);
    auto temp = Tensor(x.GetStorage()->tensor->datatype, shape);
    TiledTopKSort(*Program::GetInstance().GetCurrentFunction(), x.GetStorage(), y.GetStorage(), temp.GetStorage(),
                  idxStart, 0);
    return std::tie(y, temp);
}

void TiledTopKMerge(Function& function, const LogicalTensorPtr& x, const LogicalTensorPtr& y, int mergeSize)
{
    auto& op = function.AddOperation(Opcode::OP_TOPK_MERGE, {x}, {y});
    op.SetAttribute(TOPK_MERGE_SIZE, mergeSize);
}

Tensor TopKMerge(const Tensor& x, int mergeSize)
{
    auto y = Tensor(x.GetStorage()->tensor->datatype, x.GetShape());
    TiledTopKMerge(*Program::GetInstance().GetCurrentFunction(), x.GetStorage(), y.GetStorage(), mergeSize);
    return y;
}

void TiledTopKExtract(Function& function, const LogicalTensorPtr& x, const LogicalTensorPtr& y, int k, bool isIndex)
{
    auto& op = function.AddOperation(Opcode::OP_TOPK_EXTRACT, {x}, {y});
    op.SetAttribute(TOPK_K, k);
    op.SetAttribute(TOPK_INDEX, static_cast<int>(isIndex));
}

// view op
Tensor View(const Tensor& operand, const std::vector<int64_t>& shapes, const std::vector<int64_t>& offsets)
{
    DECLARE_TRACER();
    Tensor result(operand.GetStorage()->Datatype(), shapes, "View_" + operand.GetStorage()->GetRawTensor()->GetSymbol(),
                  operand.Format());
    auto& op = Program::GetInstance().GetCurrentFunction()->AddOperation(Opcode::OP_VIEW, {operand.GetStorage()},
                                                                         {result.GetStorage()});
    auto validShape = GetViewValidShape(operand.GetStorage()->GetDynValidShape(), offsets, {}, shapes);
    result.GetStorage()->UpdateDynValidShape(validShape);
    auto newOffsets = SymbolicScalar::FromConcrete(offsets);
    op.SetOpAttribute(std::make_shared<ViewOpAttribute>(offsets, newOffsets, validShape));
    result.GetStorage()->SetAttr("ORIGINAL_SHAPE", operand.GetShape());
    const auto& operandDynVS = operand.GetStorage()->GetDynValidShape();
    if (!operandDynVS.empty()) {
        result.GetStorage()->SetAttr("ORIGINAL_DYN_VALID_SHAPE", operandDynVS);
    }
    return result;
}

bool isInteger(float num)
{
    const float epsilon = 1e-6f;
    double intPart;
    double fracPart = std::modf(num, &intPart);
    return std::abs(fracPart) < epsilon || std::abs(1 - std::abs(fracPart)) < epsilon;
}

void FactorCheck(const Tensor& operand, const float factor)
{
    FE_ASSERT(factor > 0) << "factor must > 0";
    if (factor > 1) {
        FE_ASSERT(isInteger(factor)) << "factor must be int";
    } else if (factor < 1) {
        auto lastDim = operand.GetShape()[operand.GetShape().size() - 1];
        FE_ASSERT(isInteger(lastDim * factor))
            << "lastDim * factor must be int,  lastDim = " << lastDim << ", factor = " << factor;
    }
}

// viewtype op
Tensor View(const Tensor& operand, const DataType dstDataType)
{
    DECLARE_TRACER();
    auto originDType = operand.GetStorage()->Datatype();
    float factor = (float)BytesOf(originDType) /
                   (float)BytesOf(dstDataType); // factor就代表了目标tensor尾部维度要扩展的倍数
    FactorCheck(operand, factor);

    auto dstShape = operand.GetShape();
    dstShape[dstShape.size() - 1] = int(dstShape[dstShape.size() - 1] * factor);

    auto validShape = operand.GetStorage()->GetDynValidShape();
    auto changedDim = validShape[validShape.size() - 1] * BytesOf(originDType) / BytesOf(dstDataType);
    validShape[validShape.size() - 1] = changedDim;

    Tensor result(dstDataType, dstShape, "ViewType_" + operand.GetStorage()->GetRawTensor()->GetSymbol(),
                  operand.Format());
    result.GetStorage()->UpdateDynValidShape(validShape);

    Program::GetInstance().GetCurrentFunction()->AddOperation(Opcode::OP_VIEW_TYPE, {operand.GetStorage()},
                                                              {result.GetStorage()});
    return result;
}

void TiledViewTypeOperation(Function& function, const TileShape& tileShape, const int cur, Input& input, float factor,
                            const LogicalTensorPtr& result)
{
    if (cur == static_cast<int>(input.tensor.GetShape().size())) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);

        auto outputShape = input.tileInfo.shape;
        outputShape[outputShape.size() - 1] = int(outputShape[outputShape.size() - 1] * factor);
        auto outputOffset = input.tileInfo.offset;
        outputOffset[outputOffset.size() - 1] = int(outputOffset[outputOffset.size() - 1] * factor);

        auto resultTile = result->View(function, outputShape, outputOffset);
        function.AddOperation(Opcode::OP_VIEW_TYPE, {tile}, {resultTile});
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    CheckFwkOpTileShape(vecTile, input.tensor.GetStorage(), "OP_VIEW_TYPE");
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledViewTypeOperation(function, tileShape, cur + 1, input, factor, result);
    }
}

void TiledViewTypeOperation(Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand,
                            const LogicalTensorPtr& result)
{
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, operand->shape.size() == operand->offset.size())
        << "operand's shape size and offset size should be equal";

    TileInfo operandTileInfo(operand->shape.size(), operand->offset.size());
    auto input = Input{operand, operandTileInfo};

    float factor = (float)BytesOf(operand->tensor->datatype) / (float)BytesOf(result->tensor->datatype);
    // 检查TileShape是否符合要求
    if (factor < 1) {
        auto& vecTile = tileShape.GetVecTile();
        CHECK_OP(vecTile.valid()) << "OP_VIEW_TYPE tile shape is empty or invalid, please call set_vec_tile_shapes "
                                  << "before creating the view op. operand: " << operand->Dump();
        auto lastDim = vecTile[vecTile.size() - 1];
        FE_ASSERT(isInteger(lastDim * factor)) << "TileShape lastDim * factor must be int";
    }
    TiledViewTypeOperation(function, tileShape, 0, input, factor, result);
}

Tensor View(const Tensor& operand, const std::vector<int64_t>& shapes, const std::vector<SymbolicScalar>& newOffsets,
            const void* lr)
{
    DECLARE_TRACERX(lr);
    Tensor result(operand.GetStorage()->Datatype(), shapes, "View_" + operand.GetStorage()->GetRawTensor()->GetSymbol(),
                  operand.Format());
    result.GetStorage()->UpdateDynValidShape(SymbolicScalar::FromConcrete(shapes));
    auto function = Program::GetInstance().GetCurrentFunction();
    auto& op = function->AddOperation(Opcode::OP_VIEW, {operand.GetStorage()}, {result.GetStorage()});
    auto validShape = GetViewValidShape(operand.GetStorage()->GetDynValidShape(), {}, newOffsets, shapes);
    result.GetStorage()->UpdateDynValidShape(validShape);
    std::vector<int64_t> newOffsetsConcrete = SymbolicScalar::Concrete(newOffsets, 0);
    op.SetOpAttribute(std::make_shared<ViewOpAttribute>(newOffsetsConcrete, newOffsets, validShape));
    result.GetStorage()->SetAttr("ORIGINAL_SHAPE", operand.GetShape());
    const auto& operandDynVS2 = operand.GetStorage()->GetDynValidShape();
    if (!operandDynVS2.empty()) {
        result.GetStorage()->SetAttr("ORIGINAL_DYN_VALID_SHAPE", operandDynVS2);
    }
    return result;
}

Tensor View(const Tensor& operand, const std::vector<int64_t>& shapes, const std::vector<SymbolicScalar>& newOffset)
{
    return View(operand, shapes, newOffset, __builtin_return_address(0));
}

// 重载View，initializer_list避免歧义
Tensor View(const Tensor& operand, const std::vector<int64_t>& shapes,
            const std::initializer_list<SymbolicScalar>& newOffsets)
{
    return View(operand, shapes, std::vector<SymbolicScalar>(newOffsets), __builtin_return_address(0));
}

Tensor View(const Tensor& operand, const std::vector<int64_t>& shapes,
            const std::vector<SymbolicScalar>& newValidShapes, const std::vector<SymbolicScalar>& newOffsets)
{
    DECLARE_TRACER();

    const auto& operandShape = operand.GetShape();
    CheckViewValidShapesConstraint(newValidShapes, shapes, operandShape);

    Tensor result(operand.GetStorage()->Datatype(), shapes, "View_" + operand.GetStorage()->GetRawTensor()->GetSymbol(),
                  operand.Format());
    auto function = Program::GetInstance().GetCurrentFunction();
    auto& op = function->AddOperation(Opcode::OP_VIEW, {operand.GetStorage()}, {result.GetStorage()});
    std::vector<int64_t> newOffsetsConcrete = SymbolicScalar::Concrete(newOffsets, 0);
    op.SetOpAttribute(std::make_shared<ViewOpAttribute>(newOffsetsConcrete, newOffsets, newValidShapes));
    result.GetStorage()->UpdateDynValidShape(newValidShapes);
    result.GetStorage()->SetAttr("ORIGINAL_SHAPE", operand.GetShape());
    const auto& operandDynVS3 = operand.GetStorage()->GetDynValidShape();
    if (!operandDynVS3.empty()) {
        result.GetStorage()->SetAttr("ORIGINAL_DYN_VALID_SHAPE", operandDynVS3);
    }
    return result;
}

void TensorInnerAssemble(Function& function, const LogicalTensorPtr& operand, const LogicalTensorPtr& result,
                         const std::vector<int64_t>& offset)
{
    auto& op = function.AddOperation(Opcode::OP_ASSEMBLE, {operand}, {result});
    op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset));
}

void InnerAssemble(Function& function, const LogicalTensorPtr& operand, const LogicalTensorPtr& result,
                   const std::vector<int64_t>& offset)
{
    CALL(InnerAssemble, function, operand, result, offset);
}

Tensor Assemble(const std::vector<std::pair<Tensor, std::vector<int64_t>>>& tensors)
{
    DECLARE_TRACER();

    CHECK_OP(!tensors.empty());
    std::vector<int64_t> shape = tensors.front().first.GetShape();
    TileOpFormat format = tensors.front().first.Format();
    for (const auto& [tensor, offset] : tensors) {
        // 目前只支持2维操作
        CHECK_OP(tensor.GetShape().size() == 2) << "only support rank 2";
        CHECK_OP(tensor.GetShape().size() == tensor.GetStorage()->offset.size());
        CHECK_OP(tensor.GetShape().size() == offset.size());
        CHECK_OP(tensor.Format() == format);
    }

    auto shapeSize = tensors[0].first.GetShape().size(); // 2

    std::vector<int64_t> rawShape(shapeSize, 0);

    std::set<std::vector<int64_t>> position;
    for (const auto& [tensor, offset] : tensors) {
        (void)tensor;
        CHECK_OP(position.find(offset) == position.end());
        position.emplace(offset);
    }
    CHECK_OP(position.find(std::vector<int64_t>(shapeSize, 0)) != position.end());

    for (const auto& [tensor, offset] : tensors) {
        for (int j = 0; static_cast<size_t>(j) < shapeSize; j++) {
            rawShape[j] = std::max(rawShape[j], tensor.GetShape()[j] + offset[j]);
            CHECK_OP(offset[j] % shape[j] == 0);
            if (offset[j] > 0) {
                auto tmpOffset = offset;
                tmpOffset[j] -= shape[j];
                CHECK_OP(position.find(tmpOffset) != position.end());
            }
        }
    }

    for (int i = 0; static_cast<size_t>(i) < shapeSize; i++) {
        CHECK_OP(rawShape[i] > 0);
    }

    Tensor result(tensors[0].first.GetStorage()->Datatype(), rawShape, "Assemble", tensors[0].first.Format());
    auto& curFunc = *Program::GetInstance().GetCurrentFunction();
    for (const auto& [tensor, offset] : tensors) {
        InnerAssemble(curFunc, tensor.GetStorage(), result.GetStorage(), offset);
    }
    Program::GetInstance().GetTensorSlotManager()->TensorWrite(result, SlotProperty::ASSEMBLE_DST);
    return result;
}

Operation& TensorDInnerAssemble(Function& function, const LogicalTensorPtr& operand, const LogicalTensorPtr& result,
                                const std::vector<SymbolicScalar>& dynOffset, bool parallel)
{
    std::vector<int64_t> offset = SymbolicScalar::Concrete(dynOffset, 0);
    auto& op = function.AddOperation(Opcode::OP_ASSEMBLE, {operand}, {result});
    op.SetAssembleOpAttribute(offset, dynOffset);
    op.SetAttribute("dassemble", true);
    op.SetAttribute(OpAttributeKey::parallel, parallel);
    return op;
}

void Assemble(const Tensor& tensor, const std::vector<SymbolicScalar>& dynOffset, Tensor& dest, bool parallel)
{
    DECLARE_TRACER();

    const auto& srcShape = tensor.GetShape();
    const auto& dstShape = dest.GetShape();
    CHECK_OP(dstShape.size() == srcShape.size()) << "Assemble: src and dest requires same shape";
    CHECK_OP(dstShape.size() == dynOffset.size()) << "Assemble: dynOffset and dest requires same shape";

    for (size_t dimIdx = 0; dimIdx < srcShape.size(); ++dimIdx) {
        CHECK_OP(srcShape[dimIdx] > 0) << "Assemble: shape of src tensor requires integer";
    }

    CHECK_OP(dest.GetStorage(false)->Format() == tensor.GetStorage(false)->Format())
        << "Assemble: src and dest requires same format";
    CHECK_OP(dest.GetDataType() == tensor.GetDataType()) << "Assemble: src and dest requires same dtype";

    auto& func = *Program::GetInstance().GetCurrentFunction();

    if (IRContext::Get().AssembleNewLogicalTensor()) {
        std::vector<ir::VarPtr> tokens;
        auto newDest = dest.GetStorage(false)->NextVersion(func, tokens);
        TensorDInnerAssemble(func, tensor.GetStorage(), newDest, dynOffset, parallel);
        Program::GetInstance().GetTensorSlotManager()->TensorWrite(dest, SlotProperty::ASSEMBLE_DST);
        dest = Tensor(newDest);
        return;
    }

    TensorDInnerAssemble(func, tensor.GetStorage(), dest.GetStorage(), dynOffset, parallel);
    Program::GetInstance().GetTensorSlotManager()->TensorWrite(dest, SlotProperty::ASSEMBLE_DST);
}

void AtomicRMW(const Tensor& t, const std::vector<SymbolicScalar>& dynOffset, Tensor& dest, AtomicRMWMode mode)
{
    DECLARE_TRACER();
    CHECK_OP(dest.Format() == t.Format()) << "AtomicRMW: src and dest requires same format";
    CHECK_OP(dest.GetShape().size() == t.GetShape().size()) << "AtomicRMW: src and dest requires same shape";
    CHECK_OP(dest.GetShape().size() == dynOffset.size()) << "AtomicRMW: dynOffset and dest requires same shape";
    CHECK_OP(dest.GetDataType() == t.GetDataType()) << "AtomicRMW: src and dest requires same dtype";

    auto& func = *Program::GetInstance().GetCurrentFunction();
    auto offset = SymbolicScalar::Concrete(dynOffset, 0);

    Tensor res(t.GetDataType(), t.GetShape(), "", t.Format());
    auto& op = func.AddOperation(Opcode::OP_ATOMIC_RMW, {t.GetStorage()}, {dest.GetStorage()});
    op.SetAssembleOpAttribute(offset, dynOffset);
    op.SetAttribute(OpAttributeKey::rmwMode, (int)mode);

    Program::GetInstance().GetTensorSlotManager()->TensorWrite(dest, SlotProperty::ASSEMBLE_DST);
}

void TiledInnerAssemble(Function& function, const TileShape& tileShape, size_t cur,
                        const std::vector<SymbolicScalar>& initialOffsets, const LogicalTensorPtr& src,
                        const LogicalTensorPtr& dst, const LogicalTensorPtr& result, TileInfo& tileInfo)
{
    if (cur == src->GetShape().size()) {
        auto srcTile = src->View(function, tileInfo.shape, tileInfo.offset);
        auto& op = function.AddOperation(Opcode::OP_ASSEMBLE_SSA, {srcTile, dst}, {result});
        auto srcTileOffset = initialOffsets;
        CHECK_OP(initialOffsets.size() == tileInfo.offset.size());
        for (size_t i = 0; i < srcTileOffset.size(); i++) {
            srcTileOffset[i] = srcTileOffset[i] + tileInfo.offset[i];
        }
        Offset staticSrcTileOffsets = SymbolicScalar::Concrete(srcTileOffset, 0);
        op.SetAssembleOpAttribute(staticSrcTileOffsets, srcTileOffset);
        op.SetAttribute(OpAttributeKey::inplaceIdx, 1);
        return;
    }
    const auto& vecTile = tileShape.GetVecTile();
    CHECK_OP(vecTile.size() >= src->shape.size());
    CheckFwkOpTileShape(vecTile, src, "OP_ASSEMBLE_SSA");

    for (auto i = 0; i < src->shape[cur]; i += vecTile[cur]) {
        tileInfo.offset[cur] = i;
        tileInfo.shape[cur] = std::min(src->shape[cur] - tileInfo.offset[cur], vecTile[cur]);
        TiledInnerAssemble(function, tileShape, cur + 1, initialOffsets, src, dst, result, tileInfo);
    }
}

void TiledInnerAssemble(Function& function, const TileShape& tileShape, const Operation& op)
{
    CHECK_OP(op.GetIOperands().size() == NUM_VALUE_2);
    CHECK_OP(op.GetOOperands().size() == 1);
    CHECK_OP(op.HasAttribute(OpAttributeKey::inplaceIdx));
    auto src = op.GetInputOperand(0);
    auto dst = op.GetInputOperand(1);
    auto result = op.GetOutputOperand(0);
    auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(op.GetOpAttribute());
    CHECK_OP(assembleOpAttribute != nullptr);
    const auto& initialOffsets = assembleOpAttribute->GetToDynOffset();
    TileInfo tileInfo(src->GetShape().size(), src->GetOffset().size());
    TiledInnerAssemble(function, tileShape, 0, initialOffsets, src, dst, result, tileInfo);
}

void TensorInnerAssemble(Function& function, const LogicalTensorPtr& value, const std::vector<SymbolicScalar>& offsets,
                         const LogicalTensorPtr& dst, const LogicalTensorPtr& result)
{
    Offset staticOffsets = SymbolicScalar::Concrete(offsets, 0);
    auto& op = function.AddOperation(Opcode::OP_ASSEMBLE_SSA, {value, dst}, {result});
    op.SetAssembleOpAttribute(staticOffsets, offsets);
    op.SetAttribute(OpAttributeKey::inplaceIdx, 1);
}

void Assemble(const std::vector<AssembleItem>& items, Tensor& src, bool parallelInAssemble)
{
    DECLARE_TRACER();

    CHECK_OP(!items.empty());

    for (const auto& item : items) {
        CHECK_OP(src.GetStorage(false)->Format() == item.tensor.GetStorage(false)->Format())
            << "Assemble: src and dest requires same format";
        CHECK_OP(src.GetShape().size() == item.tensor.GetShape().size())
            << "Assemble: src and dest requires same shape size";
        CHECK_OP(src.GetShape().size() == item.offsets.size()) << "Assemble: offsets and dest requires same shape size";
        CHECK_OP(src.GetDataType() == item.tensor.GetDataType()) << "Assemble: src and dest requires same dtype";
    }

    if (parallelInAssemble) {
        Tensor result(src.GetDataType(), src.GetShape(), "assemble_parallel_out", src.GetStorage()->Format());
        auto shapes = result.GetStorage()->GetShape();
        if (std::find(shapes.begin(), shapes.end(), -1) != shapes.end()) {
            result = Tensor(src.GetDataType(), src.GetStorage()->GetDynValidShape(), "assemble_parallel_out",
                            src.GetStorage()->Format());
        }
        for (const auto& item : items) {
            auto viewTensor = View(src.GetStorage(), item.tensor.GetShape(),
                                   item.tensor.GetStorage()->GetDynValidShape(), item.offsets);
            TensorInnerAssemble(*Program::GetInstance().GetCurrentFunction(), item.tensor.GetStorage(), item.offsets,
                                viewTensor.GetStorage(), result.GetStorage());
        }
        Program::GetInstance().GetCurrentFunction()->SetSameMemId(src.GetStorage(), result.GetStorage());
        src = result;
        return;
    }

    auto preResult = src.GetStorage();
    int i = 0;
    for (const auto& item : items) {
        auto viewTensor = View(preResult, item.tensor.GetShape(), item.tensor.GetStorage()->GetDynValidShape(),
                               item.offsets);
        Tensor curResult(src.GetDataType(), src.GetShape(), "assemble_seq_out" + std::to_string(i),
                         src.GetStorage()->Format());
        auto shapes = curResult.GetStorage()->GetShape();
        if (std::find(shapes.begin(), shapes.end(), -1) != shapes.end()) {
            curResult = Tensor(src.GetDataType(), src.GetStorage()->GetDynValidShape(), "assemble_seq_out",
                               src.GetStorage()->Format());
        }
        TensorInnerAssemble(*Program::GetInstance().GetCurrentFunction(), item.tensor.GetStorage(), item.offsets,
                            viewTensor.GetStorage(), curResult.GetStorage());
        preResult = curResult.GetStorage();
        i++;
    }
    Program::GetInstance().GetCurrentFunction()->SetSameMemId(src.GetStorage(), preResult);
    src = preResult;
    return;
}

template <bool isB, bool isTrans>
void TiledGatherInL1(Function& function, const TileShape& tileShape, const LogicalTensorPtr& src,
                     const LogicalTensorPtr& offsets, const LogicalTensorPtr& blockTable, const LogicalTensorPtr& dst,
                     int blockSize)
{
    const auto& cubeTile = tileShape.GetCubeTile();

    auto [firstDimTileShape, secondDimTileShape] = !isB ? std::pair<int64_t, int64_t>{cubeTile.m[1], cubeTile.k[1]} :
                                                          std::pair<int64_t, int64_t>{cubeTile.k[1], cubeTile.n[1]};
    if constexpr (isTrans) {
        std::swap(firstDimTileShape, secondDimTileShape);
    }

    for (int64_t i = 0; i < dst->GetShape()[0]; i += firstDimTileShape) {
        auto shape0 = std::min(dst->GetShape()[0] - i, firstDimTileShape);
        for (int64_t j = 0; j < dst->GetShape()[1]; j += secondDimTileShape) {
            auto shape1 = std::min(dst->GetShape()[1] - j, secondDimTileShape);
            auto dstTile = dst->View(function, {shape0, shape1}, {i, j});
            auto offsetsTile = offsets->View(function, {1, shape0}, {0, i});
            auto blockTableTile = blockTable->View(function, {blockTable->GetShape()[0], blockTable->GetShape()[1]},
                                                   {0, 0});
            auto& op = function.AddOperation(Opcode::OP_GATHER_IN_L1, {src, offsetsTile, blockTableTile}, {dstTile});
            op.SetAttribute(OpAttributeKey::startOffset, j);
            op.SetAttribute(OP_ATTR_PREFIX + "blocksize", blockSize);
        }
    }
}

template <bool isB, bool isTrans>
Tensor experimental::GatherInL1(const Tensor& src, const Tensor& offsets, const Tensor& blockTable, int blockSize,
                                const int size)
{
    constexpr int32_t NUM_SIZE = 2;
    CHECK_OP(src.GetShape().size() == NUM_SIZE);
    CHECK_OP(offsets.GetShape().size() == NUM_SIZE); // offsets必须是两维是因为不支持1维的Tensor
    CHECK_OP(offsets.GetShape()[0] == 1);
    CHECK_OP(size <= src.GetShape()[1]);
    CHECK_OP(!offsets.GetStorage()->GetDynValidShape().empty());

    Tensor dst(src.GetStorage()->Datatype(), {offsets.GetShape()[1], size});
    if (!offsets.GetStorage()->GetDynValidShape().empty()) {
        dst.GetStorage()->UpdateDynValidShape(
            {offsets.GetStorage()->GetDynValidShape()[1], src.GetStorage()->GetDynValidShape()[1]});
    }
    auto& op = Program::GetInstance().GetCurrentFunction()->AddOperation(
        Opcode::OP_GATHER_IN_L1, {src.GetStorage(), offsets.GetStorage(), blockTable.GetStorage()}, {dst.GetStorage()});
    op.SetAttribute("isB", isB);
    op.SetAttribute("isTrans", isTrans);
    op.SetAttribute(OP_ATTR_PREFIX + "blocksize", blockSize);
    return dst;
}

template Tensor experimental::GatherInL1<false, false>(const Tensor&, const Tensor&, const Tensor&, int, int);
template Tensor experimental::GatherInL1<false, true>(const Tensor&, const Tensor&, const Tensor&, int, int);
template Tensor experimental::GatherInL1<true, false>(const Tensor&, const Tensor&, const Tensor&, int, int);
template Tensor experimental::GatherInL1<true, true>(const Tensor&, const Tensor&, const Tensor&, int, int);

static int64_t CalculateCapacity(const std::vector<int64_t>& shape)
{
    int64_t capacity = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        capacity = capacity * shape[i];
    }
    return capacity;
}

void TiledInnerReshape(Function& function, const LogicalTensorPtr& operand, const LogicalTensorPtr& result,
                       const bool isInplace = false, const bool groupReshapeNoSplit = false)
{
    auto& op = function.AddOperation("TILE_RESHAPE", {operand}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "isInplace", isInplace);
    op.SetAttribute(OpAttributeKey::groupReshapeNoSplit, groupReshapeNoSplit);
    op.SetAttribute(OP_ATTR_PREFIX + "validShape", result->GetDynValidShape());
    op.oOperand.front()->SetIsDummy();
}

void TensorInnerReshape(Function& function, const LogicalTensorPtr& operand, const LogicalTensorPtr& result,
                        const std::vector<SymbolicScalar>& validShape)
{
    auto& operation = function.AddOperation(Opcode::OP_RESHAPE, {operand}, {result});
    if (validShape.empty()) {
        result->UpdateDynValidShape(SymbolicScalar::FromConcrete(result->GetShape()));
    } else {
        result->UpdateDynValidShape(validShape);
    }
    operation.SetAttribute("reshape", result->shape);
}

static std::vector<int64_t> CheckAndInferShape(const std::vector<int64_t>& oriShape,
                                               const std::vector<int64_t>& dstshape)
{
    for (size_t i = 0; i < oriShape.size(); i++) {
        CHECK_OP(oriShape[i] != -1)
            << "Reshape does not support dynamic axis in input shape. Input shape contains -1 at index " << i;
    }

    int negIdx = -1;
    std::vector<int64_t> newShape = dstshape;
    auto capacity = CalculateCapacity(oriShape);

    // Special case: dstshape == [-1], flatten to 1D
    if (dstshape.size() == 1 && dstshape[0] == -1) {
        return std::vector<int64_t>{capacity};
    }

    for (size_t i = 0; i < newShape.size(); i++) {
        int x = newShape[i];
        CHECK_OP(x >= -1) << "Invalid shape " << x;
        if (x == -1) {
            CHECK_OP(negIdx == -1) << "Only one dim can be inferred";
            negIdx = i;
        }
        CHECK_OP(capacity % x == 0) << "Invalid dstshape";
        capacity /= x;
    }

    if (negIdx != -1) {
        newShape[negIdx] = -capacity;
        capacity = 1;
    }
    CHECK_OP(capacity == 1) << "Shape size not match";
    return newShape;
}

bool MatchInputTailProductOutputLastDim(const std::vector<int64_t>& inputShape, const std::vector<int64_t>& outputShape)
{
    if (outputShape.empty() || inputShape.size() != outputShape.size() + 1 || outputShape.back() <= 0) {
        return false;
    }

    int64_t tailProduct = 1;
    for (size_t i = 1; i < inputShape.size(); ++i) {
        if (inputShape[i] <= 0 || tailProduct > LLONG_MAX / inputShape[i]) {
            return false;
        }
        tailProduct *= inputShape[i];
    }
    return tailProduct == outputShape.back();
}

// batch MatMul优化pattern，不插入register copy
bool MatchBatchMatMulPattern(const std::vector<int64_t>& inputShape, const std::vector<int64_t>& outputShape)
{
    constexpr size_t DIMENSIONS_2D = 2;
    constexpr size_t DIMENSIONS_3D = 3;
    constexpr size_t DIMENSIONS_4D = 4;
    constexpr size_t DIMENSIONS_5D = 5;
    // 定义所有有效的模式：{input_size, output_size, 验证函数}
    using Validator = std::function<bool(const std::vector<int64_t>&, const std::vector<int64_t>&)>;

    static const std::vector<std::pair<std::pair<size_t, size_t>, Validator>> patterns = {
        {{DIMENSIONS_3D, DIMENSIONS_2D},
         [](const auto& in, const auto& out) { return in[0] == 1 && in[1] == out[0] && in[2] == out[1]; }},
        {{DIMENSIONS_2D, DIMENSIONS_3D},
         [](const auto& in, const auto& out) { return out[0] == 1 && in[0] == out[1] && in[1] == out[2]; }},
        {{DIMENSIONS_4D, DIMENSIONS_2D},
         [](const auto& in, const auto& out) {
             return in[0] == 1 && in[1] == 1 && in[2] == out[0] && in[3] == out[1];
         }},
        {{DIMENSIONS_2D, DIMENSIONS_4D},
         [](const auto& in, const auto& out) {
             return out[0] == 1 && out[1] == 1 && in[0] == out[2] && in[1] == out[3];
         }},
        {{DIMENSIONS_4D, DIMENSIONS_3D},
         [](const auto& in, const auto& out) {
             return in[0] == 1 && in[1] == out[0] && in[2] == out[1] && in[3] == out[2];
         }},
        {{DIMENSIONS_3D, DIMENSIONS_4D},
         [](const auto& in, const auto& out) {
             return out[0] == 1 && in[0] == out[1] && in[1] == out[2] && in[2] == out[3];
         }},
        {{DIMENSIONS_5D, DIMENSIONS_3D},
         [](const auto& in, const auto& out) {
             return in[0] == 1 && in[1] == 1 && in[2] == out[0] && in[3] == out[1] && in[4] == out[2];
         }},
        {{DIMENSIONS_3D, DIMENSIONS_5D}, [](const auto& in, const auto& out) {
             return out[0] == 1 && out[1] == 1 && in[0] == out[2] && in[1] == out[3] && in[2] == out[4];
         }}};

    for (const auto& [sizes, validator] : patterns) {
        if (inputShape.size() == sizes.first && outputShape.size() == sizes.second &&
            validator(inputShape, outputShape)) {
            return true;
        }
    }

    return MatchInputTailProductOutputLastDim(inputShape, outputShape);
}

static bool ReshapeNeedCopy(const Tensor& operand)
{
    if (operand.GetShape() != operand.GetStorage()->tensor->rawshape) {
        return true;
    }
    if (operand.GetStorage()->GetProducers().empty()) {
        return false;
    }

    auto op = *operand.GetStorage()->GetProducers().begin();
    while (op->GetOpcode() == Opcode::OP_VIEW || op->GetOpcode() == Opcode::OP_SLICE) {
        if (op->GetInputOperand(0)->GetShape() != op->GetOutputOperand(0)->GetShape()) {
            return true;
        }
        if (op->GetInputOperand(0) != nullptr && !op->GetInputOperand(0)->GetProducers().empty()) {
            op = *op->GetInputOperand(0)->GetProducers().begin();
        } else {
            break;
        }
    }
    return false;
}

Tensor Reshape(const Tensor& operand, const std::vector<int64_t>& dstshape,
               const std::vector<SymbolicScalar>& validShape, const bool inplace, const void* lr)
{
    DECLARE_TRACERX(lr);
    CHECK_OP(!inplace)
        << "Reshape(Inplace=true) is not supported for tensors derived from dynamic view. Please set inplace=false "
           "or do not pass valid_shape.";
    std::vector<SymbolicScalar> validShapeDefault = validShape;
    auto newShape = CheckAndInferShape(operand.GetShape(), dstshape);
    if (validShape.empty()) {
        validShapeDefault = SymbolicScalar::FromConcrete(newShape);
    } else {
        for (auto validShapeItem : validShape) {
            if (validShapeItem.IsImmediate() && validShapeItem == -1) {
                CHECK_OP(false) << "Not supported: validShape contains -1";
            }
        }
    }

    if (ReshapeNeedCopy(operand) && !MatchBatchMatMulPattern(operand.GetShape(), dstshape)) {
        Tensor copyOperand(operand.GetStorage()->Datatype(), operand.GetShape(), "", operand.Format());
        copyOperand.GetStorage()->UpdateDynValidShape(operand.GetStorage()->GetDynValidShape());
        CALL(InnerAssign, *Program::GetInstance().GetCurrentFunction(), operand.GetStorage(), copyOperand.GetStorage());
        Tensor result(copyOperand.GetStorage()->Datatype(), newShape, "", operand.Format());
        CALL(InnerReshape, *Program::GetInstance().GetCurrentFunction(), copyOperand.GetStorage(), result.GetStorage(),
             validShapeDefault);
        return result;
    } else {
        Tensor result(operand.GetStorage()->Datatype(), newShape, "", operand.Format());
        CALL(InnerReshape, *Program::GetInstance().GetCurrentFunction(), operand.GetStorage(), result.GetStorage(),
             validShapeDefault);
        return result;
    }
}

Tensor Reshape(const Tensor& operand, const std::vector<int64_t>& dstshape,
               const std::vector<SymbolicScalar>& validShape, const bool inplace)
{
    return Reshape(operand, dstshape, validShape, inplace, __builtin_return_address(0));
}

Tensor Reshape(const Tensor& operand, const std::initializer_list<int64_t>& dstshape,
               const std::initializer_list<SymbolicScalar>& validShape, const bool inplace)
{
    return Reshape(operand, std::vector<int64_t>(dstshape), std::vector<SymbolicScalar>(validShape), inplace,
                   __builtin_return_address(0));
}

Tensor Reshape(const Tensor& operand, const std::vector<SymbolicScalar>& dstShape, const bool inplace)
{
    CHECK_OP(inplace) << "The 'inplace' parameter must be true !!!";
    Tensor dst(operand.GetStorage()->Datatype(), dstShape, "", operand.Format());
    auto slotManager = Program::GetInstance().GetTensorSlotManager();
    auto& operation = Program::GetInstance().GetCurrentFunction()->AddOperation(
        Opcode::OP_RESHAPE, {operand.GetStorage()}, {dst.GetStorage()});
    operation.SetAttribute(OP_ATTR_PREFIX + "isInplace", true);
    slotManager->TensorWrite(dst);
    Program::GetInstance().GetCurrentFunction()->SetSameMemId(operand.GetStorage(), dst.GetStorage());
    if (slotManager->GetOutputIndex(dst) != -1) {
        slotManager->SetSameSlot(operand, dst);
    }
    return dst;
}

constexpr int64_t GATHER_IN_UB_INDICES_LEADING_DIM = 1;
constexpr int GATHER_IN_UB_SUPPORTED_AXIS = -2;

void TiledGatherInUB(Function& function, const TileShape& tileShape, const LogicalTensorPtr& param,
                     const LogicalTensorPtr& indices, const LogicalTensorPtr& blockTable,
                     const LogicalTensorPtr& result, int blockSize)
{
    const auto& vecTile = tileShape.GetVecTile();
    CHECK(VectorErrorCode::ERR_CONFIG_TILE, vecTile.size() >= NUM_VALUE_2)
        << "GatherInUB requires at least two vec_tile dimensions. "
        << "Please call pypto.set_vec_tile_shapes(rows, cols) before gather_in_ub; got " << vecTile.size()
        << " dimensions.";
    CHECK(VectorErrorCode::ERR_CONFIG_TILE, vecTile[0] > 0 && vecTile[1] > 0)
        << "GatherInUB requires positive vec_tile dimensions. "
        << "Please call pypto.set_vec_tile_shapes(rows, cols) with rows > 0 and cols > 0; got " << vecTile[0] << ", "
        << vecTile[1] << ".";
    const int64_t firstDimTileShape = vecTile[0];
    const int64_t secondDimTileShape = vecTile[1];
    for (int64_t i = 0; i < result->GetShape()[0]; i += firstDimTileShape) {
        auto shape0 = std::min(result->GetShape()[0] - i, firstDimTileShape);
        for (int64_t j = 0; j < result->GetShape()[1]; j += secondDimTileShape) {
            auto shape1 = std::min(result->GetShape()[1] - j, secondDimTileShape);
            auto paramTile = param->View(function, {param->GetShape()[0], shape1}, {0, j});
            auto indicesTile = indices->View(function, {GATHER_IN_UB_INDICES_LEADING_DIM, shape0}, {0, i});
            auto blockTableTile = blockTable->View(function, {blockTable->GetShape()[0], blockTable->GetShape()[1]},
                                                   {0, 0});
            auto resultTile = result->View(function, {shape0, shape1}, {i, j});
            auto& op = function.AddOperation(Opcode::OP_GATHER_IN_UB, {paramTile, indicesTile, blockTableTile},
                                             {resultTile});
            op.SetAttribute(OpAttributeKey::blockSize, blockSize);
            (void)op;
        }
    }
}

/**
 * 定制版本，暂不拓展性，支撑ds v3.2
 * 支撑功能
 * param [a,b]
 * indices [1,c]
 * axis = -2
 * result [c,b]
 */
Tensor experimental::GatherInUB(const Tensor& params, const Tensor& indices, const Tensor& blockTable, int blockSize,
                                int axis)
{
    const auto& supportedTypes = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_GATHER_IN_UB);
    CheckTensorDataType(params.GetStorage(), supportedTypes, "GatherInUB");

    CheckTensorDimRange(params.GetStorage(), NUM_VALUE_2, NUM_VALUE_2, "GatherInUB params");
    CheckTensorDimRange(indices.GetStorage(), NUM_VALUE_2, NUM_VALUE_2, "GatherInUB indices");
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, indices.GetShape()[0] == GATHER_IN_UB_INDICES_LEADING_DIM)
        << "GatherInUB requires indices with shape [1, c]; got first dimension " << indices.GetShape()[0] << ".";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, axis == GATHER_IN_UB_SUPPORTED_AXIS)
        << "GatherInUB only supports axis = " << GATHER_IN_UB_SUPPORTED_AXIS << "; got " << axis << ".";
    Tensor result{params.GetStorage()->Datatype(), {indices.GetShape()[1], params.GetShape()[1]}};
    if (!indices.GetStorage()->GetDynValidShape().empty()) {
        result.GetStorage()->UpdateDynValidShape(
            {indices.GetStorage()->GetDynValidShape()[1], params.GetStorage()->GetDynValidShape()[1]});
    }
    auto& op = Program::GetInstance().GetCurrentFunction()->AddOperation(
        Opcode::OP_GATHER_IN_UB, {params.GetStorage(), indices.GetStorage(), blockTable.GetStorage()},
        {result.GetStorage()});
    op.SetAttribute(OpAttributeKey::blockSize, blockSize);
    (void)op;
    return result;
}

void Reshape(const Tensor& operand, Tensor& dst)
{
    CHECK_OP(operand.Format() == dst.Format()) << "Tensor format not match";
    auto slotManager = Program::GetInstance().GetTensorSlotManager();
    auto& operation = Program::GetInstance().GetCurrentFunction()->AddOperation(
        Opcode::OP_RESHAPE, {operand.GetStorage()}, {dst.GetStorage()});
    operation.SetAttribute(OP_ATTR_PREFIX + "isInplace", true);
    slotManager->TensorWrite(dst);
    Program::GetInstance().GetCurrentFunction()->SetSameMemId(operand.GetStorage(), dst.GetStorage());
    if (slotManager->GetOutputIndex(dst) != -1) {
        slotManager->SetSameSlot(operand, dst);
    }
}

void ExpandOperationInto(Function& function, const TileShape& tileShape, Opcode opCode,
                         const std::vector<LogicalTensorPtr>& iOperand, const std::vector<LogicalTensorPtr>& oOperand,
                         const Operation& op)
{
    auto tileFunc = TiledFuncRegistry::GetInstance().GetTiledFunc(opCode);
    if (tileFunc != nullptr) {
        return tileFunc(function, tileShape, iOperand, oOperand, op);
    }
    switch (opCode) {
        case Opcode::OP_GATHER_IN_L1: {
            bool isB = op.GetBoolAttribute("isB");
            bool isTrans = op.GetBoolAttribute("isTrans");
            int blocksize = op.GetIntAttribute(OP_ATTR_PREFIX + "blocksize");
            if (isB) {
                if (isTrans) {
                    TiledGatherInL1<true, true>(function, tileShape, iOperand[0], iOperand[1], iOperand[2], oOperand[0],
                                                blocksize);
                } else {
                    TiledGatherInL1<true, false>(function, tileShape, iOperand[0], iOperand[1], iOperand[2],
                                                 oOperand[0], blocksize);
                }
            } else {
                if (isTrans) {
                    TiledGatherInL1<false, true>(function, tileShape, iOperand[0], iOperand[1], iOperand[2],
                                                 oOperand[0], blocksize);
                } else {
                    TiledGatherInL1<false, false>(function, tileShape, iOperand[0], iOperand[1], iOperand[2],
                                                  oOperand[0], blocksize);
                }
            }

            break;
        }
        case Opcode::OP_GATHER_IN_UB: {
            int blocksize = op.GetIntAttribute(OpAttributeKey::blockSize);
            TiledGatherInUB(function, tileShape, iOperand[0], iOperand[1], iOperand[2], oOperand[0], blocksize);
            break;
        }
        case Opcode::OP_REGISTER_COPY: {
            UnaryOperationOperandCheck(iOperand, oOperand);
            TiledInnerRegisterCopy(function, tileShape, iOperand[0], oOperand[0]);
            break;
        }
        case Opcode::OP_A_MUL_B: {
            Matrix::ConstructTileGraph(function, tileShape, iOperand, oOperand[0], op);
            break;
        }
        case Opcode::OP_CONV2D:
        case Opcode::OP_CONV3D: {
            Conv::ConstructTileGraph(function, tileShape, iOperand, oOperand[0], op);
            break;
        }
        case Opcode::OP_CONV_BP_INPUT_2D:
        case Opcode::OP_CONV_BP_INPUT_3D: {
            ConvBp::ConstructTileGraph(function, tileShape, iOperand, oOperand[0], op);
            break;
        }
        case Opcode::OP_FAKE_TRANS: {
            function.AddOperation(Opcode::OP_FAKE_TRANS, iOperand, oOperand);
            break;
        }
        case Opcode::OP_TOPK_SORT: {
            int idxStart = op.GetIntAttribute(TOPK_START_INDEX);
            SymbolicScalar dynIdxStart;
            if (op.HasAttr(OpAttributeKey::dynScalar)) {
                dynIdxStart = op.GetSymbolicScalarAttribute(OpAttributeKey::dynScalar);
            }
            TiledTopKSort(function, iOperand[0], oOperand[0], oOperand[1], dynIdxStart, idxStart);
            break;
        }
        case Opcode::OP_TOPK_MERGE: {
            int mergeSize = op.GetIntAttribute(TOPK_MERGE_SIZE);
            TiledTopKMerge(function, iOperand[0], oOperand[0], mergeSize);
            break;
        }
        case Opcode::OP_TOPK_EXTRACT: {
            int k = op.GetIntAttribute(TOPK_K);
            int isIndex = op.GetIntAttribute(TOPK_INDEX);
            TiledTopKExtract(function, iOperand[0], oOperand[0], k, isIndex);
            break;
        }
        case Opcode::OP_SORT: {
            int idxStart = op.GetIntAttribute(SORT_START_INDEX);
            int descending = op.GetIntAttribute(SORT_ORDER);
            TiledSort(function, iOperand[0], oOperand[0], oOperand[1], oOperand[2], idxStart, descending);
            break;
        }
        case Opcode::OP_COMPARE_SWAP: {
            int descending = op.GetIntAttribute(SORT_ORDER);
            TiledCompareAndSwap(function, iOperand[0], iOperand[1], iOperand[2], iOperand[3], oOperand[0], oOperand[1],
                                oOperand[2], oOperand[3], descending);
            break;
        }
        case Opcode::OP_MERGE: {
            int descending = op.GetIntAttribute(SORT_ORDER);
            int fullSort = op.GetIntAttribute(SORT_FULL);
            TiledMerge(function, iOperand[0], iOperand[1], oOperand[0], oOperand[1], oOperand[2], fullSort, descending);
            break;
        }
        case Opcode::OP_REDUCE_ACC: {
            TiledReduceAcc(function, tileShape, iOperand, oOperand[0]);
            break;
        }
        case Opcode::OP_VIEW: {
            size_t opListPreSize = function.Operations(false).size();
            TiledViewOperation(function, GetViewExpandTileShape(op), iOperand[0], oOperand[0], op);
            PropagateExpandedViewOpAttrs(function, op, opListPreSize);
            break;
        }
        case Opcode::OP_ASSEMBLE:
        case Opcode::OP_CONTRACT: {
            size_t opListPreSize = function.Operations(false).size();
            auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(op.GetOpAttribute());
            if (assembleOpAttribute == nullptr) {
                const auto& inputShape = iOperand[0]->GetShape();
                assembleOpAttribute = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>(inputShape.size(), 0));
            }
            TiledAssemble(function, GetAssembleExpandTileShape(op), iOperand[0], oOperand[0], assembleOpAttribute);
            PropagateExpandedAssembleOpAttrs(function, op, opListPreSize);
            break;
        }
        case Opcode::OP_ASSEMBLE_SSA: {
            TiledInnerAssemble(function, tileShape, op);
            break;
        }
        case Opcode::OP_RESHAPE: {
            bool isInplace = false;
            op.GetAttr(OP_ATTR_PREFIX + "isInplace", isInplace);
            bool groupReshapeNoSplit = false;
            op.GetAttr(OpAttributeKey::groupReshapeNoSplit, groupReshapeNoSplit);
            TiledInnerReshape(function, iOperand[0], oOperand[0], isInplace, groupReshapeNoSplit);
            break;
        }
        case Opcode::OP_MAX_POOL: {
            TiledMaxpool(function, tileShape, iOperand[0], oOperand[0], op);
            break;
        }
        case Opcode::OP_SHMEM_PUT: {
            npu::tile_fwk::Distributed::TiledShmemPut(function, tileShape, iOperand, oOperand, op);
            break;
        }
        case Opcode::OP_SHMEM_STORE: {
            npu::tile_fwk::Distributed::TiledShmemStore(function, tileShape, iOperand, oOperand, op);
            break;
        }
        case Opcode::OP_SHMEM_GET: {
            npu::tile_fwk::Distributed::TiledShmemGet(function, tileShape, iOperand, oOperand, op);
            break;
        }
        case Opcode::OP_SHMEM_LOAD: {
            npu::tile_fwk::Distributed::TiledShmemLoad(function, tileShape, iOperand, oOperand, op);
            break;
        }
        case Opcode::OP_SHMEM_SIGNAL: {
            npu::tile_fwk::Distributed::TiledShmemSignal(function, tileShape, iOperand, oOperand, op);
            break;
        }
        case Opcode::OP_SHMEM_WAIT_UNTIL: {
            npu::tile_fwk::Distributed::TiledShmemWaitUntil(function, tileShape, iOperand, oOperand, op);
            break;
        }
        case Opcode::OP_BIND_TENSOR: {
            npu::tile_fwk::Distributed::TiledShmemBindTensor(function, tileShape, iOperand, oOperand, op);
            break;
        }
        case Opcode::OP_SHMEM_SET: {
            npu::tile_fwk::Distributed::TiledShmemSet(function, tileShape, iOperand, oOperand, op);
            break;
        }
        case Opcode::OP_VIEW_TYPE: {
            TiledViewTypeOperation(function, tileShape, iOperand[0], oOperand[0]);
            break;
        }
        case Opcode::OP_BLOCK_CALL: {
            auto& newOp = function.AddRawOperation(Opcode::OP_BLOCK_CALL, iOperand, oOperand);
            newOp.SetOpAttribute(op.GetOpAttribute());
            newOp.SetAttr(OpAttributeKey::dontTouch, true);
            break;
        }
        default: {
            FE_LOGE(FeError::NOT_EXIST, "Unsupported opcode %d, opmagic is %d", static_cast<int>(opCode),
                    op.GetOpMagic());
            FE_ASSERT(false) << "Unsupported opcode " << static_cast<int>(opCode) << ", opmagic is " << op.GetOpMagic();
        }
    }
}

Tensor Nop(const std::vector<Tensor>& inTensors)
{
    auto& function = *Program::GetInstance().GetCurrentFunction();
    auto out = std::make_shared<LogicalTensor>(function, DT_INT32, Shape{1, 1});
    LogicalTensors iOperands;
    for (const Tensor& inTensor : inTensors) {
        iOperands.emplace_back(inTensor.GetStorage());
    }
    function.AddOperation(Opcode::OP_NOP, iOperands, {out});
    return out;
}
} // namespace npu::tile_fwk
