/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file index_outcast.h
 * \brief
 */
#ifndef TILEOP_TILE_OPERATOR_INDEX_OUTCAST__H
#define TILEOP_TILE_OPERATOR_INDEX_OUTCAST__H

#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <typename T0, typename T1, typename T2, int srcrawShape1, int srcTileH, int srcTileW, int src1SAligned,
          typename DstDtype, typename SrcDtype, typename IdxDtype>
TILEOP void TIndexOutcastMode2(T0 dst, T1 src, T2 src1, unsigned b, unsigned s, unsigned srcShape3, unsigned srcShape4,
                               uint64_t offset)
{
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

    __ubuf__ SrcDtype* srcBase = reinterpret_cast<__ubuf__ SrcDtype*>(src.GetAddr());
    __ubuf__ IdxDtype* idxBase = reinterpret_cast<__ubuf__ IdxDtype*>(src1.GetAddr());
    __gm__ DstDtype* dstBase = reinterpret_cast<__gm__ DstDtype*>(dst.GetAddr()) + offset;

    __ubuf__ SrcDtype* curSrc = srcBase;
    __ubuf__ IdxDtype* dstIdx = idxBase;

    constexpr auto srcNdAligned = srcTileW;

    for (LoopVar i = 0; i < b; ++i) {
        for (LoopVar j = 0; j < s; ++j) {
            int64_t targetRow = static_cast<int64_t>(*dstIdx);
            if (targetRow >= 0) {
                __gm__ DstDtype* curDst = dstBase + targetRow * srcShape4;
                using SrcTileDefine = pto::Tile<pto::TileType::Vec, SrcDtype, srcTileH, srcTileW,
                                                pto::BLayout::RowMajor, -1, -1>;
                SrcTileDefine srcTile(srcShape3, srcShape4);
                pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(curSrc));
                using ShapeDim5 = pto::Shape<-1, -1, -1, -1, -1>;
                using StrideDim5 = pto::Stride<-1, -1, -1, -1, -1>;
                using DstGlobal = pto::GlobalTensor<DstDtype, ShapeDim5, StrideDim5>;
                DstGlobal dstGlobal(curDst, pto::Shape(1, 1, 1, 1, srcShape4), pto::Stride(1, 1, 1, 1, 1));
                pto::TSTORE(dstGlobal, srcTile);
            }
            curSrc += srcNdAligned;
            dstIdx++;
        }
        curSrc += (srcrawShape1 - s) * srcNdAligned;
        dstIdx += src1SAligned - s;
    }
}

template <unsigned cacheMode, unsigned blockSize, typename T0, typename T1, typename T2, typename C>
TILEOP void TIndexOutcast(T0 dst, T1 src, T2 src1, C coordinate)
{
    const auto uLayout = src.GetLayout();
    auto srcShape1 = uLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto srcShape2 = uLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto srcShape3 = uLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto srcShape4 = uLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    const auto iLayout = src1.GetLayout();
    auto src1Shape3 = iLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto src1Shape4 = iLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    const auto dLayout = dst.GetLayout();
    auto dstShape3 = dLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    auto offset = dLayout.template GetGmOffset<C, MAX_DIMS>(coordinate);

    using DstDtype = typename T0::Type;
    using SrcDtype = typename T1::Type;
    using IdxDtype = typename T2::Type;

    constexpr auto srcrawShape1 = TileOp::GetTensorTileShapeDim<T1, DIM_3RD, MAX_DIMS>();
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, DIM_4TH, MAX_DIMS>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, DIM_5TH, MAX_DIMS>();
    constexpr auto src1SAligned = TileOp::GetTensorTileShapeDim<T2, DIM_5TH, MAX_DIMS>();

    if (srcShape1 == 0 || srcShape2 == 0 || srcShape4 == 0 || src1Shape3 == 0 || src1Shape4 == 0) {
        return;
    }

    constexpr int FULL_ROW_CACHE_MODE = 2;
    if constexpr (cacheMode == FULL_ROW_CACHE_MODE) {
        TIndexOutcastMode2<T0, T1, T2, srcrawShape1, srcTileH, srcTileW, src1SAligned, DstDtype, SrcDtype, IdxDtype>(
            dst, src, src1, src1Shape3, src1Shape4, srcShape3, srcShape4, offset);
        return;
    }

    auto alignTS2TS3 = srcTileH * srcTileW;
    auto alignSrc1 = src1SAligned;
    __ubuf__ SrcDtype* srcBase = reinterpret_cast<__ubuf__ SrcDtype*>(src.GetAddr());
    __ubuf__ IdxDtype* src1Base = reinterpret_cast<__ubuf__ IdxDtype*>(src1.GetAddr());
    __gm__ DstDtype* dstBase = reinterpret_cast<__gm__ DstDtype*>(dst.GetAddr());
    dstBase += offset;

    for (LoopVar i = 0; i < srcShape1; ++i) {
        for (LoopVar j = 0; j < srcShape2; ++j) {
            for (LoopVar k = 0; k < src1Shape4; ++k) {
                set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
                set_flag(PIPE_V, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

                auto curValue = *(reinterpret_cast<__ubuf__ IdxDtype*>(src1Base + k));
                int64_t idxVal = static_cast<int64_t>(curValue);
                if (idxVal < 0) {
                    continue;
                }
                __ubuf__ SrcDtype* srcPtr = srcBase + k * srcrawShape1;

                if constexpr (cacheMode == 1) {
                    auto blockCount = curValue / blockSize;
                    auto index = curValue % blockSize;
                    __gm__ DstDtype* newDst = dstBase + blockCount * blockSize * dstShape4 +
                                              index * TileOp::BLOCK_SIZE / sizeof(DstDtype);
                    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);

                    using SrcTileDefine = pto::Tile<pto::TileType::Vec, SrcDtype, srcTileH, srcTileW,
                                                    pto::BLayout::RowMajor, -1, -1>;
                    SrcTileDefine srcTile(srcShape3, srcShape4);
                    pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(srcPtr));
                    using ShapeDim = pto::Shape<-1, -1, -1, -1, -1>;
                    using StrideDim = pto::Stride<-1, -1, -1, -1, -1>;
                    using DstGlobalType = pto::GlobalTensor<DstDtype, ShapeDim, StrideDim>;
                    DstGlobalType dstGlobal(newDst, pto::Shape(1, 1, 1, 1, srcShape4), pto::Stride(1, 1, 1, 1, 1));
                    pto::TSTORE(dstGlobal, srcTile);

                } else {
                    __gm__ DstDtype* newDst = dstBase + static_cast<unsigned>(curValue) * dstShape4;
                    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);

                    using SrcTileDefine = pto::Tile<pto::TileType::Vec, SrcDtype, srcTileH, srcTileW,
                                                    pto::BLayout::RowMajor, -1, -1>;
                    SrcTileDefine srcTile(srcShape3, srcShape4);
                    pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(srcPtr));
                    using ShapeDim5 = pto::Shape<-1, -1, -1, -1, -1>;
                    using StrideDim5 = pto::Stride<-1, -1, -1, -1, -1>;
                    using DstGlobalType = pto::GlobalTensor<DstDtype, ShapeDim5, StrideDim5>;
                    DstGlobalType dstGlobal(newDst, pto::Shape(1, 1, 1, 1, srcShape4), pto::Stride(1, 1, 1, 1, 1));
                    pto::TSTORE(dstGlobal, srcTile);
                }
            }
        }
        srcBase += alignTS2TS3;
        src1Base += alignSrc1;
        dstBase += dstShape3 * dstShape4;
    }
}
#endif // TILEOP_TILE_OPERATOR_INDEX_OUTCAST__H
