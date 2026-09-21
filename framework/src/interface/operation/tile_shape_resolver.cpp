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
 * \file tile_shape_resolver.cpp
 * \brief Resolve the per-input / per-output tile shape of an op from the op (opcode + op-level tile shape + shapes).
 */

#include "tile_shape_resolver.h"

#include <algorithm>

#include "interface/utils/error.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation_common.h"
#include "interface/operation/operation_impl.h"
#include "interface/operation/vector/gather_mask_common.h"
#include "interface/tensor/logical_tensor.h"

namespace npu::tile_fwk {

TileShapeResolver& TileShapeResolver::Instance()
{
    static TileShapeResolver inst;
    return inst;
}

namespace {
// Build a TileShape whose VecTile is `vec`, leaving other sub-tiles default.
TileShape MakeTileShapeFromVec(const std::vector<int64_t>& vec)
{
    TileShape tile;
    tile.SetVecTile(vec);
    return tile;
}

// Per-axis tile size for an elementwise operand: min(operand shape, op vecTile).
// Axes beyond the op vecTile keep the operand's full shape (not tiled).
std::vector<int64_t> ElementwiseInputVecTile(const std::vector<int64_t>& inShape, const VecTile& opVecTile)
{
    std::vector<int64_t> tile(inShape.size());
    for (size_t axis = 0; axis < inShape.size(); ++axis) {
        if (axis < opVecTile.tile.size()) {
            tile[axis] = std::min(inShape[axis], opVecTile.tile[axis]);
        } else {
            tile[axis] = inShape[axis];
        }
    }
    return tile;
}

// Like ElementwiseInputVecTile but the `groupAxis` is tiled within its per-group slice
// (curShapeLen = inShape[groupAxis] / group), mirroring InnerTransData/InnerTransDataND.
std::vector<int64_t> TransDataInputVecTile(const std::vector<int64_t>& inShape, const VecTile& opVecTile, int64_t group,
                                           int64_t groupAxis)
{
    if (group <= 0) {
        group = 1;
    }
    std::vector<int64_t> tile(inShape.size());
    for (size_t axis = 0; axis < inShape.size(); ++axis) {
        int64_t curShapeLen = (static_cast<int64_t>(axis) == groupAxis) ? inShape[axis] / group : inShape[axis];
        if (axis < opVecTile.tile.size()) {
            tile[axis] = std::min(curShapeLen, opVecTile.tile[axis]);
        } else {
            tile[axis] = curShapeLen;
        }
    }
    return tile;
}

// Like ElementwiseInputVecTile but `fullAxis` is kept at its full shape (not tiled),
// and per-axis clamping uses `clampShape` instead of `inShape` (some ops clamp an
// input by another operand's shape, e.g. SCATTER clamps self/src by idx.shape).
std::vector<int64_t> VecTileAxisFull(const std::vector<int64_t>& inShape, const std::vector<int64_t>& clampShape,
                                     const VecTile& opVecTile, int64_t fullAxis)
{
    std::vector<int64_t> tile(inShape.size());
    for (size_t axis = 0; axis < inShape.size(); ++axis) {
        if (static_cast<int64_t>(axis) == fullAxis) {
            tile[axis] = inShape[axis];
        } else if (axis < opVecTile.tile.size()) {
            tile[axis] = std::min(clampShape[axis], opVecTile.tile[axis]);
        } else {
            tile[axis] = clampShape[axis];
        }
    }
    return tile;
}

} // namespace

TileShape TileShapeResolver::GetInputTileShape(const Operation& op, int index) const
{
    const auto& iOperands = op.GetIOperands();
    FE_ASSERT(FeError::OUT_OF_RANGE, index >= 0 && static_cast<size_t>(index) < iOperands.size())
        << "input tile shape index out of range: " << index << ", input operand size=" << iOperands.size();

    const auto& opTileShape = op.GetTileShape();
    const auto& opVecTile = opTileShape.GetVecTile();
    const auto& inShape = iOperands[index]->GetShape();

    switch (op.GetOpcode()) {
        case Opcode::OP_INDEX_ADD_UB:
        case Opcode::OP_INDEX_ADD: {
            int axis = op.HasAttr(OP_ATTR_PREFIX + "axis") ?
                           static_cast<int>(op.GetIntAttribute(OP_ATTR_PREFIX + "axis")) :
                           0;
            if (index == 0) {
                return MakeTileShapeFromVec(VecTileAxisFull(inShape, inShape, opVecTile, axis));
            }
            if (index == 1) {
                return MakeTileShapeFromVec(ElementwiseInputVecTile(inShape, opVecTile));
            }
            return MakeTileShapeFromVec(inShape); // indices: full
        }
        case Opcode::OP_GATHER: {
            int axis = op.HasAttr(OP_ATTR_PREFIX + "axis") ?
                           static_cast<int>(op.GetIntAttribute(OP_ATTR_PREFIX + "axis")) :
                           0;
            if (axis < 0) {
                axis += static_cast<int>(iOperands[0]->GetShape().size());
            }
            const auto& paramsShape = iOperands[0]->GetShape();
            const auto& indicesShape = iOperands[1]->GetShape();
            auto indicesDim = static_cast<int64_t>(indicesShape.size());
            if (index == 0) {
                std::vector<int64_t> tile(paramsShape.size());
                for (size_t a = 0; a < paramsShape.size(); ++a) {
                    if (static_cast<int64_t>(a) == axis) {
                        tile[a] = paramsShape[a];
                    } else {
                        int64_t cur = (static_cast<int64_t>(a) < axis) ? static_cast<int64_t>(a) :
                                                                         static_cast<int64_t>(a) + indicesDim - 1;
                        tile[a] = (cur < static_cast<int64_t>(opVecTile.tile.size())) ?
                                      std::min(paramsShape[a], opVecTile.tile[cur]) :
                                      paramsShape[a];
                    }
                }
                return MakeTileShapeFromVec(tile);
            }
            if (index == 1) {
                std::vector<int64_t> tile(indicesShape.size());
                for (size_t j = 0; j < indicesShape.size(); ++j) {
                    int64_t cur = axis + static_cast<int64_t>(j);
                    tile[j] = (cur < static_cast<int64_t>(opVecTile.tile.size())) ?
                                  std::min(indicesShape[j], opVecTile.tile[cur]) :
                                  indicesShape[j];
                }
                return MakeTileShapeFromVec(tile);
            }
            return opTileShape;
        }
        case Opcode::OP_GATHER_ELEMENT: {
            int axis = op.HasAttr(OP_ATTR_PREFIX + "axis") ?
                           static_cast<int>(op.GetIntAttribute(OP_ATTR_PREFIX + "axis")) :
                           0;
            if (index == 0) {
                return MakeTileShapeFromVec(VecTileAxisFull(inShape, inShape, opVecTile, axis));
            }
            return MakeTileShapeFromVec(ElementwiseInputVecTile(inShape, opVecTile));
        }
        case Opcode::OP_SCATTER_ELEMENT: {
            int axis = op.HasAttr(OP_ATTR_PREFIX + "axis") ?
                           static_cast<int>(op.GetIntAttribute(OP_ATTR_PREFIX + "axis")) :
                           0;
            const auto& idxShape = iOperands[1]->GetShape();
            if (index == 0) {
                return MakeTileShapeFromVec(VecTileAxisFull(inShape, idxShape, opVecTile, axis));
            }
            return MakeTileShapeFromVec(VecTileAxisFull(idxShape, idxShape, opVecTile, axis));
        }
        case Opcode::OP_SCATTER: {
            int axis = op.HasAttr(OP_ATTR_PREFIX + "axis") ?
                           static_cast<int>(op.GetIntAttribute(OP_ATTR_PREFIX + "axis")) :
                           0;
            const auto& idxShape = iOperands[1]->GetShape();
            if (index == 0) {
                return MakeTileShapeFromVec(VecTileAxisFull(inShape, idxShape, opVecTile, axis));
            }
            return MakeTileShapeFromVec(VecTileAxisFull(idxShape, idxShape, opVecTile, axis));
        }
        case Opcode::OP_INDEX_OUTCAST: {
            // iOperand = [src, index, dst]; axis attr (stored under the bare key "axis", like
            // OP_ARGSORT — see indexing.cpp SetAttribute("axis", ...)). dst is the inplace GM tensor.
            // src: axis full, non-axis min(src.shape, vecTile). dst: axis full, non-axis min(dst.shape, vecTile).
            // index: 2-D. Per TiledScatterUpdate: the axis dim is full; the literal axis 0 is tiled by
            // vecTile[0] only when axis != 0 (i.e. when axis 0 is a non-reduce dim); all other non-axis
            // dims are full. (Best-effort; see TiledScatterUpdate "only cut index first axis".)
            int axis = op.HasAttr("axis") ? static_cast<int>(op.GetIntAttribute("axis")) : 0;
            if (index == 0 || index == 2) {
                return MakeTileShapeFromVec(VecTileAxisFull(inShape, inShape, opVecTile, axis));
            }
            // index == 1: 2-D index tensor
            std::vector<int64_t> tile = inShape;
            for (size_t a = 0; a < inShape.size(); ++a) {
                if (static_cast<int64_t>(a) == axis) {
                    tile[a] = inShape[a]; // axis dim: full
                } else if (a == 0 && axis != 0 && a < opVecTile.tile.size()) {
                    tile[a] = std::min(inShape[a], opVecTile.tile[a]); // literal axis 0 (non-reduce): tiled
                } else {
                    tile[a] = inShape[a]; // other non-axis dims: full
                }
            }
            return MakeTileShapeFromVec(tile);
        }
        case Opcode::OP_RANGE:
            // No tensor inputs (start/step/size are attrs); unreachable for a valid index.
            return opTileShape;
        case Opcode::OP_ROWMAX_SINGLE:
        case Opcode::OP_ROWMIN_SINGLE:
        case Opcode::OP_ROWSUM_SINGLE:
        case Opcode::OP_ROWPROD_SINGLE:
        case Opcode::OP_ROWARGMAX_SINGLE:
        case Opcode::OP_ROWARGMIN_SINGLE: {
            int axis = op.HasAttr(OP_ATTR_PREFIX + "AXIS") ?
                           static_cast<int>(op.GetIntAttribute(OP_ATTR_PREFIX + "AXIS")) :
                           0;
            if (axis < 0) {
                axis += static_cast<int>(inShape.size());
            }
            return MakeTileShapeFromVec(VecTileAxisFull(inShape, inShape, opVecTile, axis));
        }
        case Opcode::OP_ROWEXPMAX:
        case Opcode::OP_ROWEXPSUM: {
            return MakeTileShapeFromVec(VecTileAxisFull(inShape, inShape, opVecTile, 1));
        }
        case Opcode::OP_WHERE_TT:
        case Opcode::OP_WHERE_TS:
        case Opcode::OP_WHERE_ST:
        case Opcode::OP_WHERE_SS: {
            if (index == 0) {
                auto tile = ElementwiseInputVecTile(inShape, opVecTile);
                if (iOperands[0]->Datatype() == DataType::DT_UINT8 && !tile.empty() &&
                    opVecTile.tile.size() >= tile.size()) {
                    tile.back() = std::min(inShape.back(), opVecTile.tile[tile.size() - 1] / 8);
                }
                return MakeTileShapeFromVec(tile);
            }
            return MakeTileShapeFromVec(ElementwiseInputVecTile(inShape, opVecTile));
        }
        case Opcode::OP_BITSORT:
        case Opcode::OP_MRGSORT: {
            int axis = op.HasAttr(OP_ATTR_PREFIX + "axis") ?
                           static_cast<int>(op.GetIntAttribute(OP_ATTR_PREFIX + "axis")) :
                           0;
            if (axis < 0) {
                axis += static_cast<int>(inShape.size());
            }
            return MakeTileShapeFromVec(VecTileAxisFull(inShape, inShape, opVecTile, axis));
        }
        case Opcode::OP_ARGSORT: {
            int axis = op.HasAttr("axis") ? static_cast<int>(op.GetIntAttribute("axis")) : 0;
            if (axis < 0) {
                axis += static_cast<int>(inShape.size());
            }
            return MakeTileShapeFromVec(VecTileAxisFull(inShape, inShape, opVecTile, axis));
        }
        case Opcode::OP_EXTRACT: {
            int64_t last = static_cast<int64_t>(inShape.size()) - 1;
            return MakeTileShapeFromVec(VecTileAxisFull(inShape, inShape, opVecTile, last));
        }
        case Opcode::OP_TOPK: {
            int axis = op.HasAttr(OP_ATTR_PREFIX + "axis") ?
                           static_cast<int>(op.GetIntAttribute(OP_ATTR_PREFIX + "axis")) :
                           0;
            if (axis < 0) {
                axis += static_cast<int>(inShape.size());
            }
            std::vector<int64_t> tile = ElementwiseInputVecTile(inShape, opVecTile);
            if (axis >= static_cast<int>(opVecTile.tile.size())) {
                return MakeTileShapeFromVec(tile);
            }
            int64_t shapeAxis = inShape[axis];
            int64_t align = opVecTile.tile[axis];
            int64_t nonAxisProd = 1;
            for (size_t a = 0; a < inShape.size(); ++a) {
                if (static_cast<int>(a) == axis) {
                    continue;
                }
                nonAxisProd *= tile[a];
            }
            int64_t sourceShapeSize = nonAxisProd * shapeAxis;
            constexpr int64_t kBlockSize = 32;
            constexpr int64_t maxNumValue = 8192;
            if (shapeAxis > align * 2) {
                int64_t tileShapeSize = nonAxisProd * align;
                if (sourceShapeSize < maxNumValue) {
                    align = shapeAxis;
                } else if (tileShapeSize < maxNumValue) {
                    align = std::max<int64_t>(kBlockSize, (maxNumValue / nonAxisProd / kBlockSize) * kBlockSize);
                }
            }
            align = (align + kBlockSize - 1) / kBlockSize * kBlockSize;
            tile[axis] = std::min(align, shapeAxis);
            return MakeTileShapeFromVec(tile);
        }
        case Opcode::OP_SORT_UB: {
            int axis = op.HasAttr(OP_ATTR_PREFIX + "axis") ?
                           static_cast<int>(op.GetIntAttribute(OP_ATTR_PREFIX + "axis")) :
                           0;
            if (axis < 0) {
                axis += static_cast<int>(inShape.size());
            }
            return MakeTileShapeFromVec(VecTileAxisFull(inShape, inShape, opVecTile, axis));
        }
        case Opcode::OP_VEC_DUP:
            return opTileShape;
        case Opcode::OP_NCHW2NC1HWC0: {
            int64_t group = op.HasAttr(OP_ATTR_PREFIX + "group") ? op.GetIntAttribute(OP_ATTR_PREFIX + "group") : 1;
            return MakeTileShapeFromVec(TransDataInputVecTile(inShape, opVecTile, group, 1));
        }
        case Opcode::OP_NCHW2Fractal_Z: {
            int64_t group = op.HasAttr(OP_ATTR_PREFIX + "group") ? op.GetIntAttribute(OP_ATTR_PREFIX + "group") : 1;
            return MakeTileShapeFromVec(TransDataInputVecTile(inShape, opVecTile, group, 0));
        }
        case Opcode::OP_NC1HWC02NCHW: {
            int64_t group = op.HasAttr(OP_ATTR_PREFIX + "group") ? op.GetIntAttribute(OP_ATTR_PREFIX + "group") : 1;
            return MakeTileShapeFromVec(TransDataInputVecTile(inShape, opVecTile, group, 1));
        }
        case Opcode::OP_NCDHW2NDC1HWC0: {
            int64_t group = op.HasAttr(OP_ATTR_PREFIX + "group") ? op.GetIntAttribute(OP_ATTR_PREFIX + "group") : 1;
            return MakeTileShapeFromVec(TransDataInputVecTile(inShape, opVecTile, group, 2));
        }
        case Opcode::OP_NCDHW2FRACTAL_Z_3D: {
            int64_t group = op.HasAttr(OP_ATTR_PREFIX + "group") ? op.GetIntAttribute(OP_ATTR_PREFIX + "group") : 1;
            return MakeTileShapeFromVec(TransDataInputVecTile(inShape, opVecTile, group, 0));
        }
        case Opcode::OP_NDC1HWC02NCDHW: {
            int64_t group = op.HasAttr(OP_ATTR_PREFIX + "group") ? op.GetIntAttribute(OP_ATTR_PREFIX + "group") : 1;
            return MakeTileShapeFromVec(TransDataInputVecTile(inShape, opVecTile, group, 2));
        }
        case Opcode::OP_UNIFORM:
            return opTileShape;
        case Opcode::OP_PACK: {
            int64_t tile = 1;
            for (size_t i = 0; i < opVecTile.tile.size(); ++i) {
                tile *= opVecTile.tile[i];
            }
            int64_t axis0 = inShape.empty() ? tile : std::min(inShape[0], tile);
            return MakeTileShapeFromVec({axis0});
        }
        case Opcode::OP_PRELU: {
            if (index == 0) {
                return MakeTileShapeFromVec(ElementwiseInputVecTile(inShape, opVecTile));
            }
            const auto& weightShape = iOperands[1]->GetShape();
            if (inShape.size() == 1) {
                return MakeTileShapeFromVec({1});
            }
            int64_t wTile = (opVecTile.tile.size() >= 2) ? std::min(weightShape[0], opVecTile.tile[1]) : weightShape[0];
            return MakeTileShapeFromVec({wTile});
        }
        case Opcode::OP_CUBE_CONV_D2S:
        case Opcode::OP_CUBE_CONCAT_C:
            return opTileShape;
        case Opcode::OP_INDEX_PUT: {
            const auto& selfShape = iOperands[0]->GetShape();
            const auto& valuesShape = iOperands[1]->GetShape();
            auto selfDim = static_cast<int64_t>(selfShape.size());
            auto valuesDim = static_cast<int64_t>(valuesShape.size());
            if (index == 1) {
                return MakeTileShapeFromVec(ElementwiseInputVecTile(valuesShape, opVecTile));
            }
            if (index >= 2) {
                int64_t axis0Tile = opVecTile.tile.empty() ? inShape[0] : std::min(inShape[0], opVecTile.tile[0]);
                return MakeTileShapeFromVec({axis0Tile});
            }
            std::vector<int64_t> tile(selfDim);
            int64_t leading = selfDim - valuesDim;
            for (int64_t axis = 0; axis < selfDim; ++axis) {
                if (axis <= leading) {
                    tile[axis] = selfShape[axis];
                } else {
                    int64_t cur = axis - leading;
                    if (cur < static_cast<int64_t>(opVecTile.tile.size())) {
                        tile[axis] = std::min(selfShape[axis], opVecTile.tile[cur]);
                    } else {
                        tile[axis] = selfShape[axis];
                    }
                }
            }
            return MakeTileShapeFromVec(tile);
        }
        case Opcode::OP_A_MUL_B:
        case Opcode::OP_A_MUL_BT:
        case Opcode::OP_A_MULACC_B:
        case Opcode::OP_A_MULACC_BT: {
            const auto& cubeTile = opTileShape.GetCubeTile();
            auto transA = op.HasAttr(Matrix::A_MUL_B_TRANS_A) && op.GetBoolAttribute(Matrix::A_MUL_B_TRANS_A);
            auto transB = op.HasAttr(Matrix::A_MUL_B_TRANS_B) && op.GetBoolAttribute(Matrix::A_MUL_B_TRANS_B);
            // L1 copy-in is the first cut of the matmul inputs (closest hop to a producer VIEW).
            // ConstructTileGraph loops M/N at L0 (m[0]/n[0]) and slices A/B from the original
            // tensor at {m[0], k[1]} / {k[2], n[0]} (k[1]=tileKAL1, k[2]=tileKBL1). L0 then
            // further slices K off that L1 buffer. k[2]==0 falls back to k[1], matching SetCubeTile.
            constexpr int kBL1Idx = 2;
            const int64_t tileKAL1 = cubeTile.k[1];
            const int64_t tileKBL1 = (cubeTile.k[kBL1Idx] > 0) ? cubeTile.k[kBL1Idx] : tileKAL1;
            if (index == 0) {
                std::vector<int64_t> tileA = {cubeTile.m[0], tileKAL1};
                if (transA) {
                    std::reverse(tileA.begin(), tileA.end());
                }
                return MakeTileShapeFromVec(tileA);
            }
            if (index == 1) {
                std::vector<int64_t> tileB = {tileKBL1, cubeTile.n[0]};
                if (transB) {
                    std::reverse(tileB.begin(), tileB.end());
                }
                return MakeTileShapeFromVec(tileB);
            }
            auto hasMX = op.HasAttr(Matrix::A_MUL_B_MX_ATTR) && op.GetBoolAttribute(Matrix::A_MUL_B_MX_ATTR);
            auto hasBias = op.HasAttr(Matrix::A_MUL_B_BIAS_ATTR) && op.GetBoolAttribute(Matrix::A_MUL_B_BIAS_ATTR);
            auto hasScale = op.HasAttr(Matrix::A_MUL_B_VECTOR_QUANT_FLAG) &&
                            op.GetBoolAttribute(Matrix::A_MUL_B_VECTOR_QUANT_FLAG);
            int base = 2;
            int aScaleIdx = -1, bScaleIdx = -1, biasIdx = -1, scaleIdx = -1;
            if (hasMX) {
                aScaleIdx = base;
                bScaleIdx = base + 1;
                base += 2;
            }
            if (hasBias) {
                biasIdx = base++;
            }
            if (hasScale) {
                scaleIdx = base;
            }
            const auto& aShape = iOperands[0]->GetShape();
            const auto& bShape = iOperands[1]->GetShape();
            int64_t m = transA ? aShape[1] : aShape[0];
            int64_t k = transA ? aShape[0] : aShape[1];
            int64_t n = transB ? bShape[0] : bShape[1];
            auto ceilDiv = [](int64_t x, int64_t a) { return (x + a - 1) / a; };
            if (index == biasIdx || index == scaleIdx) {
                return MakeTileShapeFromVec({1, std::min(n, cubeTile.n[0])});
            }
            if (hasMX && index == aScaleIdx) {
                auto transAScale = op.HasAttr(Matrix::A_MUL_B_SCALE_A_COPY_IN_MODE) &&
                                   op.GetIntAttribute(Matrix::A_MUL_B_SCALE_A_COPY_IN_MODE) ==
                                       static_cast<int64_t>(Matrix::CopyInMode::DN2NZ);
                int64_t mL0 = std::min(m, cubeTile.m[0]);
                int64_t kScale = ceilDiv(std::min(k, cubeTile.k[1]), 64);
                std::vector<int64_t> tile = transAScale ? std::vector<int64_t>{kScale, mL0, 2} :
                                                          std::vector<int64_t>{mL0, kScale, 2};
                return MakeTileShapeFromVec(tile);
            }
            if (hasMX && index == bScaleIdx) {
                auto transBScale = op.HasAttr(Matrix::A_MUL_B_SCALE_B_COPY_IN_MODE) &&
                                   op.GetIntAttribute(Matrix::A_MUL_B_SCALE_B_COPY_IN_MODE) ==
                                       static_cast<int64_t>(Matrix::CopyInMode::DN2NZ);
                int64_t nL0 = std::min(n, cubeTile.n[0]);
                int64_t kScale = ceilDiv(std::min(k, cubeTile.k[1]), 64);
                std::vector<int64_t> tile = transBScale ? std::vector<int64_t>{nL0, kScale, 2} :
                                                          std::vector<int64_t>{kScale, nL0, 2};
                return MakeTileShapeFromVec(tile);
            }
            return opTileShape;
        }
        case Opcode::OP_RESHAPE:
            return MakeTileShapeFromVec(inShape);
        default:
            return MakeTileShapeFromVec(ElementwiseInputVecTile(inShape, opVecTile));
    }
}

TileShape TileShapeResolver::GetOutputTileShape(const Operation& op, int index) const
{
    const auto& oOperands = op.GetOOperands();
    FE_ASSERT(FeError::OUT_OF_RANGE, index >= 0 && static_cast<size_t>(index) < oOperands.size())
        << "output tile shape index out of range: " << index << ", output operand size=" << oOperands.size();

    const auto& opTileShape = op.GetTileShape();
    const auto& opVecTile = opTileShape.GetVecTile();
    const auto& outShape = oOperands[index]->GetShape();

    switch (op.GetOpcode()) {
        case Opcode::OP_A_MUL_B:
        case Opcode::OP_A_MUL_BT:
        case Opcode::OP_A_MULACC_B:
        case Opcode::OP_A_MULACC_BT: {
            // Output C is tiled by the L0 cube tile {m[0], n[0]} (see SetVecTileBasedOnUbSize
            // in cube_operation_impl.cpp), clamped to the actual {M, N}.
            const auto& cubeTile = opTileShape.GetCubeTile();
            const auto& iOperands = op.GetIOperands();
            auto transA = op.HasAttr(Matrix::A_MUL_B_TRANS_A) && op.GetBoolAttribute(Matrix::A_MUL_B_TRANS_A);
            auto transB = op.HasAttr(Matrix::A_MUL_B_TRANS_B) && op.GetBoolAttribute(Matrix::A_MUL_B_TRANS_B);
            const auto& aShape = iOperands[0]->GetShape();
            const auto& bShape = iOperands[1]->GetShape();
            int64_t m = transA ? aShape[1] : aShape[0];
            int64_t n = transB ? bShape[0] : bShape[1];
            return MakeTileShapeFromVec({std::min(m, cubeTile.m[0]), std::min(n, cubeTile.n[0])});
        }
        case Opcode::OP_NCHW2NC1HWC0:
        case Opcode::OP_NCHW2Fractal_Z:
        case Opcode::OP_NC1HWC02NCHW:
        case Opcode::OP_NCDHW2NDC1HWC0:
        case Opcode::OP_NCDHW2FRACTAL_Z_3D:
        case Opcode::OP_NDC1HWC02NCDHW: {
            // TransData output is in a transformed layout (C0/N0/group); its dim count differs
            // from the op-level VecTile (which follows the input), so a per-axis elementwise
            // clamp does not apply. Handle*Format passes the FULL dstTensor as the tile op's
            // oOperand (not a sliced View), so every emitted tile op writes the whole output and
            // the first-cut output tile equals the full output shape.
            return MakeTileShapeFromVec(outShape);
        }
        case Opcode::OP_PACK: {
            // Pack output is 1-D bytes: outShape[0] = inShape[0] * byte. PackOperationTileFunc
            // emits outputShape{ min(inShape[0], tile) * byte } with tile = prod(vecTile), so the
            // first-cut output tile = { min(outShape[0], tile * byte) }.
            const auto& iOperands = op.GetIOperands();
            int64_t byte = iOperands.empty() ? 1 : static_cast<int64_t>(BytesOf(iOperands[0]->Datatype()));
            int64_t tile = 1;
            for (size_t i = 0; i < opVecTile.tile.size(); ++i) {
                tile *= opVecTile.tile[i];
            }
            int64_t out0 = outShape.empty() ? tile * byte : std::min(outShape[0], tile * byte);
            return MakeTileShapeFromVec({out0});
        }
        case Opcode::OP_CUBE_CONV_D2S:
        case Opcode::OP_CUBE_CONCAT_C:
            // OpCalcType::CONV with no registered TileFunc: not expanded by ExpandFunction, tiling
            // is ConvTile-driven. Out of this resolver's scope; return op-level TileShape as a safe
            // placeholder (GetOutputTileShape is not exercised on these via the tensor-graph path).
            return opTileShape;
        case Opcode::OP_PERMUTE:
        case Opcode::OP_PERMUTE_ELEMENT: {
            // Output axis i <- input axis perm[i]; the op-level VecTile follows the INPUT axis
            // order, so the output tile is the VecTile permuted by `perm`, clamped to outShape.
            // (TiledPermuteOperation: resultTileShape = PermuteTileVector(inputTile.shape, perm),
            //  inputTile.shape = min(inShape, vecTile) in input axis order.)
            auto perm = op.GetVectorIntAttribute<int>(OpAttributeKey::perm);
            std::vector<int64_t> tile(outShape.size());
            for (size_t i = 0; i < outShape.size(); ++i) {
                int64_t srcAxis = (i < perm.size()) ? static_cast<int64_t>(perm[i]) : static_cast<int64_t>(i);
                if (srcAxis < 0) {
                    srcAxis += static_cast<int64_t>(outShape.size());
                }
                int64_t v = (srcAxis < static_cast<int64_t>(opVecTile.tile.size())) ? opVecTile.tile[srcAxis] :
                                                                                      outShape[i];
                tile[i] = std::min(outShape[i], v);
            }
            return MakeTileShapeFromVec(tile);
        }
        case Opcode::OP_TRANSPOSE_MOVEOUT:
        case Opcode::OP_TRANSPOSE_MOVEIN:
        case Opcode::OP_TRANSPOSE_VNCHWCONV: {
            // A 2-axis transpose: output tile = op-level VecTile with axes shape[0] and shape[1]
            // swapped, clamped to outShape. (TiledInnerTranspose: resultTileShape = inputTile.shape
            // with shape[0]/shape[1] swapped; inputTile.shape = min(inShape, vecTile) in input order.)
            auto swp = op.GetVectorIntAttribute<int64_t>(OP_ATTR_PREFIX + "shape");
            std::vector<int64_t> tile(outShape.size());
            for (size_t a = 0; a < outShape.size(); ++a) {
                int64_t srcAxis = static_cast<int64_t>(a);
                if (swp.size() == 2) {
                    if (a == static_cast<size_t>(swp[0])) {
                        srcAxis = swp[1];
                    } else if (a == static_cast<size_t>(swp[1])) {
                        srcAxis = swp[0];
                    }
                }
                int64_t v = (srcAxis < static_cast<int64_t>(opVecTile.tile.size())) ? opVecTile.tile[srcAxis] :
                                                                                      outShape[a];
                tile[a] = std::min(outShape[a], v);
            }
            return MakeTileShapeFromVec(tile);
        }
        case Opcode::OP_UNPACK: {
            // Output 1-D: outShape[0] = inShape[0]/dstByte. UnPackOperationTileFunc emits
            // outputShape{ min(inShape[0], vecTile[0]) / dstByte }, dstByte = BytesOf(output dtype).
            const auto& iOperands = op.GetIOperands();
            int64_t dstByte = static_cast<int64_t>(BytesOf(oOperands[index]->Datatype()));
            if (iOperands.empty() || opVecTile.tile.empty() || dstByte <= 0) {
                return opTileShape;
            }
            int64_t in0 = iOperands[0]->GetShape().empty() ? 0 : iOperands[0]->GetShape()[0];
            return MakeTileShapeFromVec({std::min(in0, opVecTile.tile[0]) / dstByte});
        }
        case Opcode::OP_DEINTERLEAVE_SINGLE: {
            // Splits the last axis in half: outShape[last] = inShape[last]/2. The single-input
            // path tiles the last axis by vecTile[last]/2, others by vecTile. (Both outputs share
            // the same tile shape.)
            std::vector<int64_t> tile(outShape.size());
            for (size_t a = 0; a < outShape.size(); ++a) {
                if (a == outShape.size() - 1 && a < opVecTile.tile.size()) {
                    tile[a] = std::min(outShape[a], opVecTile.tile[a] / 2);
                } else if (a < opVecTile.tile.size()) {
                    tile[a] = std::min(outShape[a], opVecTile.tile[a]);
                } else {
                    tile[a] = outShape[a];
                }
            }
            return MakeTileShapeFromVec(tile);
        }
        case Opcode::OP_EXTRACT: {
            // Last axis is not tiled (full); other axes min(outShape, vecTile). (TiledExtract jumps
            // the last axis, leaving it at full outShape[last].)
            int64_t last = static_cast<int64_t>(outShape.size()) - 1;
            return MakeTileShapeFromVec(VecTileAxisFull(outShape, outShape, opVecTile, last));
        }
        case Opcode::OP_BITSORT:
        case Opcode::OP_MRGSORT: {
            // The sort axis (TOPK_AXIS = OP_ATTR_PREFIX+"axis") is not tiled (full); other axes
            // min(outShape, vecTile). (TiledBitSort/TiledMrgSort jump the sort axis.)
            int axis = op.HasAttr(OP_ATTR_PREFIX + "axis") ?
                           static_cast<int>(op.GetIntAttribute(OP_ATTR_PREFIX + "axis")) :
                           0;
            if (axis < 0) {
                axis += static_cast<int>(outShape.size());
            }
            return MakeTileShapeFromVec(VecTileAxisFull(outShape, outShape, opVecTile, axis));
        }
        case Opcode::OP_GATHER_MASK_BUILDIN: {
            // Last axis tile shrinks by patternMode: /2 for mode 1/2, /4 for mode 3-6, else full min;
            // other axes min(outShape, vecTile). (TiledGatherMaskBuildIn.)
            uint8_t patternMode = static_cast<uint8_t>(op.GetIntAttribute(OP_ATTR_PREFIX + "patternMode"));
            const auto pattern = static_cast<GatherMaskPattern>(patternMode);
            int64_t divisor = NUM_VALUE_1;
            if (IsHalfGatherMaskPattern(pattern)) {
                divisor = NUM_VALUE_2;
            } else if (IsQuarterGatherMaskPattern(pattern)) {
                divisor = NUM_VALUE_4;
            }
            std::vector<int64_t> tile(outShape.size());
            for (size_t a = 0; a < outShape.size(); ++a) {
                if (a == outShape.size() - 1 && a < opVecTile.tile.size()) {
                    tile[a] = std::min(outShape[a], opVecTile.tile[a] / divisor);
                } else if (a < opVecTile.tile.size()) {
                    tile[a] = std::min(outShape[a], opVecTile.tile[a]);
                } else {
                    tile[a] = outShape[a];
                }
            }
            return MakeTileShapeFromVec(tile);
        }
        case Opcode::OP_RESHAPE:
            return MakeTileShapeFromVec(outShape);
        case Opcode::OP_INDEX_PUT:
            // TiledIndexPut emits every tile op with the same full {result} buffer (in-place GM
            // write), not a per-tile output View.
            return MakeTileShapeFromVec(outShape);
        case Opcode::OP_ASSEMBLE_SSA:
            // TiledInnerAssemble emits {srcTile, dst}, {result} with the same full result buffer
            // on every tile cut (inplaceIdx=1).
            return MakeTileShapeFromVec(outShape);
        case Opcode::OP_INDEX_OUTCAST: {
            const auto& iOperands = op.GetIOperands();
            int axis = op.HasAttr("axis") ? static_cast<int>(op.GetIntAttribute("axis")) : 0;
            if (op.HasAttr(OpAttributeKey::cacheMode) &&
                op.GetStringAttribute(OpAttributeKey::cacheMode) == "PA_BSND") {
                // TiledScatterUpdateFor2Dims/4Dims: every tile op writes the full {result}.
                return MakeTileShapeFromVec(outShape);
            }
            if (!iOperands.empty()) {
                const auto& srcShape = iOperands[0]->GetShape();
                // TiledIndexScatterUpdate (2-D index cut): output is the full dst/result tensor.
                if (axis == 1 && srcShape.size() == 2 && opVecTile.tile.size() >= 2 &&
                    opVecTile.tile[1] == srcShape[1]) {
                    return MakeTileShapeFromVec(outShape);
                }
            }
            // TiledScatterUpdate: output is resultTile = View(result, dstTileInfo); same tiling
            // as the result/dst buffer (typically iOperand[2] shares the shape).
            if (iOperands.size() >= 3) {
                const auto& dstShape = iOperands[2]->GetShape();
                return MakeTileShapeFromVec(VecTileAxisFull(dstShape, dstShape, opVecTile, axis));
            }
            return MakeTileShapeFromVec(ElementwiseInputVecTile(outShape, opVecTile));
        }
        default:
            // Default: output tiled by the op-level VecTile, per-axis clamped to the output's own
            // shape. Covers:
            //  - elementwise ops (Add/Div/Cast/Expand/Axpy/Where/OneHot/PReLU ...);
            //  - reductions: the tensor-graph reduce op always carries the reduced axis as dim 1
            //    (ProcessResultShape squeezes keepDim=false via a SEPARATE OP_RESHAPE, which is not
            //    a tensor op), so min(1, vecTile[axis]) = 1 and the default is correct;
            //  - gather/scatter/topk (op-level VecTile already matches the output dim count);
            //  - generated 1-D ops OP_RANGE / OP_UNIFORM / OP_VEC_DUP: each tiles a 1-D output by
            //    vecTile[0], which is exactly min(outShape[0], vecTile[0]).
            return MakeTileShapeFromVec(ElementwiseInputVecTile(outShape, opVecTile));
    }
}

} // namespace npu::tile_fwk
