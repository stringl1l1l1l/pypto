/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fillpad.h
 * \brief FillPad operator tile implementation
 */

#ifndef TILEOP_TILE_OPERATOR_FILLPAD__H
#define TILEOP_TILE_OPERATOR_FILLPAD__H

#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <pto::PadValue padValue, size_t dstCols, typename DstTensor, typename SrcTensor>
TILEOP void TFillPad(DstTensor dst, SrcTensor src)
{
    constexpr auto dstShapeSize = Std::tuple_size<typename DstTensor::Shape>::value;
    constexpr auto srcShapeSize = Std::tuple_size<typename SrcTensor::Shape>::value;
    static_assert(srcShapeSize == dstShapeSize, "FillPad: Src and Dst rank mismatch");

    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto dstStride0 = dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto dstStride1 = dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto dstStride2 = dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    auto dstStride3 = dstLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();

    const auto srcLayout = src.GetLayout();
    auto srcShape3 = srcLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto srcShape4 = srcLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto srcStride0 = srcLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto srcStride1 = srcLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto srcStride2 = srcLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    auto srcStride3 = srcLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();

    using SrcDtype = typename SrcTensor::Type;
    using DstDtype = typename DstTensor::Type;
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<DstTensor, DIM_4TH, MAX_DIMS>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<DstTensor, DIM_5TH, MAX_DIMS>();
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<SrcTensor, DIM_4TH, MAX_DIMS>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<SrcTensor, DIM_5TH, MAX_DIMS>();
    constexpr size_t PAD_TILE_INNER_SIZE = 512;
    constexpr size_t blockElements = 32 / sizeof(DstDtype);
    constexpr size_t rowTileW = (dstCols + blockElements - 1) / blockElements * blockElements;
    using DstTileType = pto::Tile<pto::TileType::Vec, DstDtype, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1,
                                  pto::SLayout::NoneBox, PAD_TILE_INNER_SIZE, padValue>;
    using SrcTileType = pto::Tile<pto::TileType::Vec, SrcDtype, srcTileH, srcTileW, pto::BLayout::RowMajor, -1, -1>;

    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                if constexpr (dstTileH >= srcTileH && dstTileW >= srcTileW && dstTileW == rowTileW) {
                    DstTileType dstTile(dstShape3, dstShape4);
                    auto dstAddr = dst.GetAddr() + dstOffset * sizeof(DstDtype);
                    pto::TASSIGN(dstTile, dstAddr);
                    SrcTileType srcTile(srcShape3, srcShape4);
                    auto srcAddr = src.GetAddr() + srcOffset * sizeof(SrcDtype);
                    pto::TASSIGN(srcTile, srcAddr);
                    pto::TFILLPAD_EXPAND(dstTile, srcTile);
                } else {
                    // A subview keeps the parent stride, but may only pad its own columns.
                    // TFILLPAD_EXPAND pads to static Cols, irrespective of the valid columns.
                    using DstRowTileType = pto::Tile<pto::TileType::Vec, DstDtype, 1, rowTileW, pto::BLayout::RowMajor,
                                                     -1, -1, pto::SLayout::NoneBox, PAD_TILE_INNER_SIZE, padValue>;
                    using SrcRowTileType = pto::Tile<pto::TileType::Vec, SrcDtype, 1, rowTileW, pto::BLayout::RowMajor,
                                                     -1, -1>;
                    for (LoopVar row = 0; row < dstShape3; ++row) {
                        auto srcValidRow = row < srcShape3 ? 1 : 0;
                        auto srcValidCol = srcValidRow == 0 ? 0 : srcShape4;
                        DstRowTileType dstTile(1, dstShape4);
                        SrcRowTileType srcTile(srcValidRow, srcValidCol);
                        auto dstAddr = dst.GetAddr() + (dstOffset + row * dstStride3) * sizeof(DstDtype);
                        auto srcAddr = src.GetAddr() + (srcOffset + row * srcStride3) * sizeof(SrcDtype);
                        pto::TASSIGN(dstTile, dstAddr);
                        pto::TASSIGN(srcTile, srcAddr);
                        pto::TFILLPAD_EXPAND(dstTile, srcTile);
                    }
                }
            }
        }
    }
}

#endif
