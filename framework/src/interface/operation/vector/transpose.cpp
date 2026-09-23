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
 * \file transpose.cpp
 * \brief
 */

#include "unary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "tilefwk/error_code.h"
#include "tilefwk/platform.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {

constexpr int64_t MERGED_TILE_BUDGET_BYTES = 48 * 1024;

static bool IsStaticFunction() { return Program::GetInstance().GetCurrentFunction()->IsStatic(); }

enum class TransposeOpType {
    TRANSPOSE_MOVEIN,
    TRANSPOSE_MOVEOUT,
    TRANSPOSE_VNCHWCONV,
};

template <TransposeOpType T>
Opcode GetTransposeOpName()
{
#define CASE(X)              \
    case TransposeOpType::X: \
        return Opcode::OP_##X
    switch (T) {
        CASE(TRANSPOSE_MOVEOUT);
        CASE(TRANSPOSE_MOVEIN);
        CASE(TRANSPOSE_VNCHWCONV);
        default:
            CHECK(VectorErrorCode::ERR_PARAM_INVALID, false) << "unknown transpose op type";
    }
#undef CASE
}

inline void UnalignPadTmpBufTile(std::vector<int64_t>& shape, int blockElem, DataType dtype)
{
    // tmpbuf按16 8对齐
    auto size = shape.size();
    if (size >= NUM_VALUE_2) {
        int64_t alignSize = VNCHWCONV_REPEAT;
        if (BytesOf(dtype) == 1) {
            alignSize = BLOCK_SIZE;
        }
        if (BytesOf(dtype) == NUM_VALUE_2 && shape[size - 1] % VNCHWCONV_REPEAT == 0 &&
            shape[size - 1] < 2 * VNCHWCONV_REPEAT) {
            shape[size - 1] = 2 * VNCHWCONV_REPEAT;
        }
        shape[size - NUM_VALUE_2] = AlignUp(shape[size - NUM_VALUE_2], alignSize);
        shape[size - 1] = AlignUp(shape[size - 1], blockElem);
    }
}

template <TransposeOpType T>
void TiledInnerTranspose(Function& function, const TileShape& tileShape, const int cur, Input& input,
                         const LogicalTensorPtr& result, const std::vector<int>& shape, int64_t maxMerge,
                         int64_t baseBytes)
{
    int shapeSize = input.tensor.GetShape().size();
    if (cur == shapeSize) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        std::vector<int64_t> resultTileShape(input.tileInfo.shape);
        std::swap(resultTileShape[shape[0]], resultTileShape[shape[1]]);
        std::vector<int64_t> resultTileOfs(input.tileInfo.offset);
        std::swap(resultTileOfs[shape[0]], resultTileOfs[shape[1]]);
        auto resultTile = result->View(function, resultTileShape, resultTileOfs);
        if (T == TransposeOpType::TRANSPOSE_MOVEOUT || T == TransposeOpType::TRANSPOSE_MOVEIN) {
            auto& op = function.AddOperation(GetTransposeOpName<T>(), {tile}, {resultTile});
            op.SetAttribute(OP_ATTR_PREFIX + "shape", shape);
        } else {
            int64_t blockElem = BLOCK_SIZE / static_cast<int>(BytesOf(tile->Datatype()));
            std::vector<int64_t> tmpShape;
            if (BytesOf(tile->Datatype()) == NUM_VALUE_2 &&
                Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_2201 &&
                input.tensor.GetShape().size() == NUM_VALUE_2 && input.tensor.GetShape()[shape[1]] >= 512) {
                tmpShape = {AlignUp(input.tileInfo.shape[shape[0]], (int64_t)VNCHWCONV_REPEAT), 2 * VNCHWCONV_REPEAT};
            } else {
                tmpShape = input.tileInfo.shape;
                UnalignPadTmpBufTile(tmpShape, blockElem, tile->Datatype());
            }
            auto tempTensor = std::make_shared<LogicalTensor>(function, tile->Datatype(), tmpShape);
            tempTensor->dynValidShape_ = SymbolicScalar::FromConcrete(tmpShape);
            auto& op = function.AddOperation(GetTransposeOpName<T>(), {tile}, {resultTile, tempTensor});
            op.SetAttribute(OP_ATTR_PREFIX + "shape", shape);
        }
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    int64_t step = vecTile[cur];
    if ((cur == shape[0] || (cur == shape[1] && input.tensor.GetShape().size() >= NUM_VALUE_3)) && maxMerge > 1) {
        step = vecTile[cur] *
               std::min<int64_t>(maxMerge, (input.tensor.GetShape()[cur] + vecTile[cur] - 1) / vecTile[cur]);
        if (cur == shape[0] && T == TransposeOpType::TRANSPOSE_VNCHWCONV) {
            int64_t tilesInRow = (input.tensor.GetShape()[cur] + vecTile[cur] - 1) / vecTile[cur];
            int64_t step16 = (step / VNCHWCONV_REPEAT) * VNCHWCONV_REPEAT;
            if (step16 >= vecTile[cur]) {
                step = step16;
            } else if (input.tensor.GetShape().size() >= NUM_VALUE_3) {
                int64_t rows16 = std::min<int64_t>(VNCHWCONV_REPEAT, tilesInRow * vecTile[cur]);
                step = std::max(step, rows16);
            } else if (tilesInRow * vecTile[cur] >= VNCHWCONV_REPEAT) {
                step = VNCHWCONV_REPEAT;
            } else {
                step = vecTile[cur];
            }
        }
    }
    if (cur == shape[1] && T == TransposeOpType::TRANSPOSE_VNCHWCONV) {
        int64_t rowTile = input.tileInfo.shape[shape[0]];
        int64_t dtypeBytes = BytesOf(input.tensor.GetDataType());
        int64_t blockElem = BLOCK_SIZE / dtypeBytes;
        int64_t padRows = AlignUp(rowTile, blockElem);
        constexpr int64_t OP_UB_BUDGET_BYTES = 160 * 1024;
        int64_t batchElems = 1;
        for (int d = 0; d < cur; ++d) {
            if (d != shape[0]) {
                batchElems *= input.tileInfo.shape[d];
            }
        }
        int64_t alignSize = (dtypeBytes == 1) ? BLOCK_SIZE : VNCHWCONV_REPEAT;
        int64_t tmpRowsAligned = AlignUp(rowTile, alignSize);
        bool b16SmallTmp = (dtypeBytes == NUM_VALUE_2) && (input.tensor.GetShape().size() == NUM_VALUE_2) &&
                           (input.tensor.GetShape()[shape[1]] >= 512);
        int64_t tmpCoef = b16SmallTmp ? 0 : tmpRowsAligned;
        int64_t bytesPerCol = batchElems * (rowTile + padRows + tmpCoef) * dtypeBytes;
        int64_t colStepMax = OP_UB_BUDGET_BYTES / bytesPerCol;
        colStepMax = (colStepMax / blockElem) * blockElem;
        if (BytesOf(input.tensor.GetDataType()) == NUM_VALUE_2 && rowTile < VNCHWCONV_REPEAT &&
            input.tensor.GetShape()[cur] >= 4096) {
            colStepMax = std::min<int64_t>(colStepMax, 1024);
        }
        step = std::min(step, std::max<int64_t>(colStepMax, blockElem));
    }
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += step) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, step);
        input.tileInfo.offset[cur] = i;
        TiledInnerTranspose<T>(function, tileShape, cur + 1, input, result, shape, maxMerge, baseBytes);
    }
}

template <TransposeOpType T>
void TiledInnerTranspose(Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand,
                         const LogicalTensorPtr& result, const std::vector<int>& shape)
{
    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand, tileInfo};
    int64_t baseBytes = BytesOf(operand->Datatype());
    const auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < (int)operand->shape.size(); ++i) {
        baseBytes *= std::min<int64_t>(operand->shape[i], vecTile[i]);
    }
    int64_t maxMerge = baseBytes > 0 ? MERGED_TILE_BUDGET_BYTES / baseBytes : 1;
    if (maxMerge > 1) {
        auto dimMergeFactor = [&](int d) {
            return std::min<int64_t>(maxMerge, (operand->shape[d] + vecTile[d] - 1) / vecTile[d]);
        };
        while (maxMerge > 1 &&
               baseBytes * dimMergeFactor(shape[0]) * dimMergeFactor(shape[1]) > MERGED_TILE_BUDGET_BYTES) {
            --maxMerge;
        }
    }
    TiledInnerTranspose<T>(function, tileShape, 0, input, result, shape, maxMerge, baseBytes);
}

void TensorInnerTranspose(Function& function, const LogicalTensorPtr& self, const LogicalTensorPtr& result,
                          std::vector<int> perm)
{
    if (perm[0] != (int)self->shape.size() - 1 && perm[1] != (int)self->shape.size() - 1) {
        auto& operation = function.AddOperation(Opcode::OP_TRANSPOSE_MOVEOUT, {self}, {result});
        operation.SetAttribute(OP_ATTR_PREFIX + "shape", perm);
        return;
    }

    if (perm[0] == (int)self->shape.size() - NUM_VALUE_2 && // last 2 dims transpose
        perm[1] == (int)self->shape.size() - 1) {
        auto& operation = function.AddOperation(Opcode::OP_TRANSPOSE_VNCHWCONV, {self}, {result});
        operation.SetAttribute(OP_ATTR_PREFIX + "shape", perm);
        return;
    }

    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          self->shape.size() == NUM_VALUE_3 || self->shape.size() == NUM_VALUE_4) // input should be 3 or 4 dims
        << "Transpose shape should be [A1,T1,A2,T2] or [T1,A2,T2]";

    // [A1,T1,A2,T2] to [A1,A2,T1,T2] or [T1,A2,T2] to [A2,T1,T2]
    auto oldVecTileShapes = TileShape::Current().GetVecTile();
    auto newVecTileShape = oldVecTileShapes;
    std::vector<int64_t> tmpShape(self->shape);
    int dim1 = (tmpShape.size() == NUM_VALUE_3) ? 0 : 1;           // if input is 3 dims, dim1 = 0, otherwise dim1 = 1
    int dim2 = (tmpShape.size() == NUM_VALUE_3) ? 1 : NUM_VALUE_2; // if input is 3 dims, dim2 = 1, otherwise dim2 = 2
    std::swap(tmpShape[dim1], tmpShape[dim2]);
    std::swap(newVecTileShape[dim1], newVecTileShape[dim2]);
    auto outValidShapes = self->GetDynValidShape();
    std::swap(outValidShapes[dim1], outValidShapes[dim2]);
    auto moveInResult = std::make_shared<LogicalTensor>(function, self->Datatype(), tmpShape, outValidShapes);
    auto& inOp = function.AddOperation(Opcode::OP_TRANSPOSE_MOVEIN, {self}, {moveInResult});
    inOp.SetAttribute(OP_ATTR_PREFIX + "shape", std::vector<int>{dim1, dim2});
    TileShape::Current().SetVecTile(newVecTileShape);

    // [A1,A2,T1,T2] to [A1,A2,T2,T1] or [A2,T1,T2] to [A2,T2,T1]
    tmpShape = moveInResult->shape;
    dim1 = (tmpShape.size() == NUM_VALUE_3) ? 1 : NUM_VALUE_2; // if input is 3 dims, dim1 = 1, otherwise dim1 = 2
    dim2 = (tmpShape.size() == NUM_VALUE_3) ? NUM_VALUE_2 : NUM_VALUE_3;
    // if input is 3 dims, dim2 = 2, otherwise dim2 = 3
    std::swap(tmpShape[dim1], tmpShape[dim2]);
    std::swap(newVecTileShape[dim1], newVecTileShape[dim2]);
    std::swap(outValidShapes[dim1], outValidShapes[dim2]);
    auto vnchwconvResult = std::make_shared<LogicalTensor>(function, self->Datatype(), tmpShape, outValidShapes);
    auto& convOp = function.AddOperation(Opcode::OP_TRANSPOSE_VNCHWCONV, {moveInResult}, {vnchwconvResult});
    convOp.SetAttribute(OP_ATTR_PREFIX + "shape", std::vector<int>{dim1, dim2});
    TileShape::Current().SetVecTile(newVecTileShape);

    // [A1,A2,T2,T1] to [A1,T2,A2,T1] or [A2,T2,T1] to [T2,A2,T1]
    tmpShape = vnchwconvResult->shape;
    dim1 = (tmpShape.size() == NUM_VALUE_3) ? 0 : 1;           // if input is 3 dims, dim1 = 0, otherwise dim1 = 1
    dim2 = (tmpShape.size() == NUM_VALUE_3) ? 1 : NUM_VALUE_2; // if input is 3 dims, dim2 = 1, otherwise dim2 = 2
    std::swap(tmpShape[dim1], tmpShape[dim2]);
    auto& outOp = function.AddOperation(Opcode::OP_TRANSPOSE_MOVEOUT, {vnchwconvResult}, {result});
    outOp.SetAttribute(OP_ATTR_PREFIX + "shape", std::vector<int>{dim1, dim2});
    TileShape::Current().SetVecTile(oldVecTileShapes);
}

bool MergeTransposeAxis(const Tensor& operand, std::vector<int64_t>& inputShape, std::vector<int64_t>& vecTileShape,
                        std::vector<SymbolicScalar>& validShape, std::vector<int>& transposeShape)
{
    auto oldTransposeShape = transposeShape;
    int64_t pre = 1;
    int64_t mid = 1;
    int64_t after = 1;
    int64_t preTileShape = 1;
    int64_t midTileShape = 1;
    int64_t afterTileShape = 1;
    SymbolicScalar preValidShape = 1;
    SymbolicScalar midValidShape = 1;
    SymbolicScalar afterValidShape = 1;
    int preNum = 0;
    int midNum = 0;
    int afterNum = 0;
    auto oldVecTileShapes = TileShape::Current().GetVecTile();
    auto oldValidShapes = validShape;
    for (int i = 0; i < (int)operand.GetShape().size(); i++) {
        if (i < oldTransposeShape[0]) {
            pre *= operand.GetShape()[i];
            preTileShape *= oldVecTileShapes[i];
            preValidShape = preValidShape * oldValidShapes[i];
            preNum++;
        } else if (i < oldTransposeShape[1] && i > oldTransposeShape[0]) {
            mid *= operand.GetShape()[i];
            midTileShape *= oldVecTileShapes[i];
            midValidShape = midValidShape * oldValidShapes[i];
            midNum++;
        } else if (i > oldTransposeShape[1]) {
            after *= operand.GetShape()[i];
            afterTileShape *= oldVecTileShapes[i];
            afterValidShape = afterValidShape * oldValidShapes[i];
            afterNum++;
        }
    }

    if (operand.GetShape().size() <= NUM_VALUE_5 &&                             // tileop支持5维
        oldTransposeShape[0] == (int)operand.GetShape().size() - NUM_VALUE_2 && // 最后2维转置
        oldTransposeShape[1] == (int)operand.GetShape().size() - 1 &&
        (IsStaticFunction() || BytesOf(operand.GetStorage()->Datatype()) != NUM_VALUE_2) &&
        operand.GetShape()[oldTransposeShape[0]] > 1 && operand.GetShape()[oldTransposeShape[0]] < VNCHWCONV_REPEAT &&
        (pre * operand.GetShape()[oldTransposeShape[0]] >= VNCHWCONV_REPEAT ||
         operand.GetShape()[oldTransposeShape[1]] < 4096) &&
        oldVecTileShapes[oldTransposeShape[1]] >= VNCHWCONV_REPEAT) {
        if (preNum != 1 || afterNum > 0) {
            return false;
        }
        inputShape.clear();
        vecTileShape.clear();
        validShape.clear();
        inputShape.push_back(pre * operand.GetShape()[oldTransposeShape[0]]);
        vecTileShape.push_back(preTileShape * oldVecTileShapes[oldTransposeShape[0]]);
        validShape.push_back(preValidShape * oldValidShapes[oldTransposeShape[0]]);
        inputShape.push_back(operand.GetShape()[oldTransposeShape[1]]);
        vecTileShape.push_back(oldVecTileShapes[oldTransposeShape[1]]);
        validShape.push_back(oldValidShapes[oldTransposeShape[1]]);
        transposeShape[0] = 0;
        transposeShape[1] = 1;
        return true;
    }

    if (preNum <= 1 && midNum <= 1 && afterNum <= 1) {
        return false;
    }
    // [A1,T1,A2,T2,A3]
    validShape.clear();
    if (preNum > 0) {
        inputShape.push_back(pre);
        vecTileShape.push_back(preTileShape);
        validShape.push_back(preValidShape);
        transposeShape[0] -= (preNum - 1);
        transposeShape[1] -= (preNum - 1);
    }
    inputShape.push_back(operand.GetShape()[oldTransposeShape[0]]);
    vecTileShape.push_back(oldVecTileShapes[oldTransposeShape[0]]);
    validShape.push_back(oldValidShapes[oldTransposeShape[0]]);
    if (midNum > 0) {
        inputShape.push_back(mid);
        vecTileShape.push_back(midTileShape);
        validShape.push_back(midValidShape);
        transposeShape[1] -= (midNum - 1);
    }
    inputShape.push_back(operand.GetShape()[oldTransposeShape[1]]);
    vecTileShape.push_back(oldVecTileShapes[oldTransposeShape[1]]);
    validShape.push_back(oldValidShapes[oldTransposeShape[1]]);
    if (afterNum > 0) {
        inputShape.push_back(after);
        vecTileShape.push_back(afterTileShape);
        validShape.push_back(afterValidShape);
    }
    return true;
}

Tensor Transpose(const Tensor& self, std::vector<int> perm)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Transpose");

    const auto& supportedTypes = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_TRANSPOSE_VNCHWCONV);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "TRANSPOSE");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_5, "TRANSPOSE");
    CheckTensorShapeSize(self.GetStorage(), "TRANSPOSE");
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, perm.size() == NUM_VALUE_2)
        << "Transpose dim num should be 2."; // perm should be 2 dims
    int shapeSize = self.GetShape().size();
    if (perm[0] < 0) {
        perm[0] += shapeSize;
    }
    if (perm[1] < 0) {
        perm[1] += shapeSize;
    }
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, perm[0] < shapeSize && perm[0] >= 0) << "Transpose dim 0 is invalid.";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, perm[1] < shapeSize && perm[1] >= 0) << "Transpose dim 1 is invalid.";

    std::sort(perm.begin(), perm.end());
    if (self.GetShape()[perm[0]] == 1 || self.GetShape()[perm[1]] == 1 || perm[0] == perm[1]) {
        if (self.GetShape()[perm[0]] == self.GetShape()[perm[1]]) {
            return self;
        }
        const auto& vecTile = TileShape::Current().GetVecTile();
        bool tileCoversBatch = true;
        for (int d = 0; d < shapeSize; ++d) {
            if (d != perm[0] && d != perm[1] && vecTile[d] < self.GetShape()[d]) {
                tileCoversBatch = false;
                break;
            }
        }
        int otherTransposedDim = (self.GetShape()[perm[0]] == 1) ? perm[1] : perm[0];
        bool otherDimNotCovered = vecTile[otherTransposedDim] < self.GetShape()[otherTransposedDim];
        if (tileCoversBatch && otherDimNotCovered) {
            std::vector<int64_t> resultShape(self.GetShape());
            std::swap(resultShape[perm[0]], resultShape[perm[1]]);
            auto validShapes = self.GetStorage()->GetDynValidShape();
            if (validShapes.empty()) {
                validShapes = SymbolicScalar::FromConcrete(self.GetShape());
            }
            std::swap(validShapes[perm[0]], validShapes[perm[1]]);
            return Reshape(self, resultShape, validShapes);
        }
    }

    auto oldVecTileShapes = TileShape::Current().GetVecTile();
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, (int)oldVecTileShapes.size() == shapeSize)
        << "TileShape dim num should be the same as input.";
    auto oldValidShapes = self.GetStorage()->GetDynValidShape();
    if (oldValidShapes.empty()) {
        oldValidShapes = SymbolicScalar::FromConcrete(self.GetShape());
    }
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, (int)oldValidShapes.size() == shapeSize)
        << "ValidShape dim num should be the same as input.";

    std::vector<int64_t> newInputShape;
    std::vector<int64_t> newVecTileShape;
    std::vector<int> newTransposeShape = perm;
    std::vector<SymbolicScalar> newValidShape = oldValidShapes;
    std::swap(oldValidShapes[perm[0]], oldValidShapes[perm[1]]);
    std::vector<int64_t> resultShape(self.GetShape());
    std::swap(resultShape[perm[0]], resultShape[perm[1]]);
    if (!MergeTransposeAxis(self, newInputShape, newVecTileShape, newValidShape, newTransposeShape)) {
        Tensor result(self.GetStorage()->Datatype(), resultShape);
        result.GetStorage()->UpdateDynValidShape(oldValidShapes);
        CALL(InnerTranspose, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), result.GetStorage(),
             perm);
        return result;
    }

    auto tmpInputTensor = Reshape(self, newInputShape, newValidShape);
    TileShape::Current().SetVecTile(newVecTileShape);
    auto tmpOutputTensor = Transpose(tmpInputTensor, newTransposeShape);
    TileShape::Current().SetVecTile(oldVecTileShapes);
    if (newInputShape.size() == NUM_VALUE_2 && !IsStaticFunction()) {
        int64_t rows = self.GetShape()[perm[0]];
        int64_t cols = self.GetShape()[perm[1]];
        int64_t pre = newInputShape[0] / rows;
        std::vector<int64_t> tmp3DShape = {cols, pre, rows};
        auto tmp3D = Reshape(tmpOutputTensor, tmp3DShape, SymbolicScalar::FromConcrete(tmp3DShape));
        Tensor result(self.GetStorage()->Datatype(), resultShape);
        result.GetStorage()->UpdateDynValidShape(oldValidShapes);
        int64_t alignRows = BytesOf(self.GetStorage()->Datatype()) == NUM_VALUE_2 ? VNCHWCONV_REPEAT : 4;
        int64_t rowSliceBytes = pre * AlignUp(rows, alignRows) * BytesOf(self.GetStorage()->Datatype());
        int64_t kT = std::min(cols, 196608 / 3 / std::max<int64_t>(1, rowSliceBytes));
        kT = std::max<int64_t>(1, kT / VNCHWCONV_REPEAT * VNCHWCONV_REPEAT);
        TileShape::Current().SetVecTile({kT, pre, rows});
        CALL(InnerTranspose, *Program::GetInstance().GetCurrentFunction(), tmp3D.GetStorage(), result.GetStorage(),
             std::vector<int>{0, 1});
        TileShape::Current().SetVecTile(oldVecTileShapes);
        return result;
    }
    return Reshape(tmpOutputTensor, resultShape, oldValidShapes);
}

void MoveOutOperationTileFunc(Function& function, const TileShape& tileShape,
                              const std::vector<LogicalTensorPtr>& iOperand,
                              const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    auto shape = op.GetVectorIntAttribute<int>(OP_ATTR_PREFIX + "shape");
    TiledInnerTranspose<TransposeOpType::TRANSPOSE_MOVEOUT>(function, tileShape, iOperand[0], oOperand[0], shape);
}

void MoveInOperationTileFunc(Function& function, const TileShape& tileShape,
                             const std::vector<LogicalTensorPtr>& iOperand,
                             const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    auto shape = op.GetVectorIntAttribute<int>(OP_ATTR_PREFIX + "shape");
    TiledInnerTranspose<TransposeOpType::TRANSPOSE_MOVEIN>(function, tileShape, iOperand[0], oOperand[0], shape);
}

void VnchwconvOperationTileFunc(Function& function, const TileShape& tileShape,
                                const std::vector<LogicalTensorPtr>& iOperand,
                                const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    auto shape = op.GetVectorIntAttribute<int>(OP_ATTR_PREFIX + "shape");
    TiledInnerTranspose<TransposeOpType::TRANSPOSE_VNCHWCONV>(function, tileShape, iOperand[0], oOperand[0], shape);
}

REGISTER_OPERATION_TILED_FUNC(OP_TRANSPOSE_MOVEOUT, Opcode::OP_TRANSPOSE_MOVEOUT, MoveOutOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_TRANSPOSE_MOVEIN, Opcode::OP_TRANSPOSE_MOVEIN, MoveInOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_TRANSPOSE_VNCHWCONV, Opcode::OP_TRANSPOSE_VNCHWCONV, VnchwconvOperationTileFunc);

} // namespace npu::tile_fwk
