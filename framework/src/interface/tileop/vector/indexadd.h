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
 * \file indexadd.h
 * \brief
 */
#ifndef TILEOP_TILE_OPERATOR_INDEXADD_H
#define TILEOP_TILE_OPERATOR_INDEXADD_H
#include "utils/sync.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

namespace indexadd_detail {
constexpr size_t MERGED_TILE_AXIS_OFFSET = 3;
} // namespace indexadd_detail

template <typename T0, typename T2, typename T3, typename dstTileDefine, typename tempTileDefine,
          typename src1TileDefine, typename Scalar>
TILEOP void IndexAddUBNotLastAxisCompute(dstTileDefine dstTile, tempTileDefine tempTile, src1TileDefine src1Tile,
                                         Scalar alpha, __ubuf__ typename T0::Type* dstAddr,
                                         __ubuf__ bfloat16_t* tempAddr, __ubuf__ typename T2::Type* src1Addr,
                                         size_t dstOffset, size_t src1Offset)
{
    pto::TASSIGN(dstTile, (uint64_t)(dstAddr + dstOffset));
    pto::TASSIGN(src1Tile, (uint64_t)(src1Addr + src1Offset));

    if constexpr (Std::is_same_v<Scalar, bfloat16_t>) {
        pto::TASSIGN(tempTile, (uint64_t)(tempAddr));
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        if (abs(static_cast<float>(alpha) - 1) > TileOp::EPSILON) {
            pto::TMULS(src1Tile, src1Tile, alpha);
            SyncV();
            pto::TCVT(tempTile, src1Tile, pto::RoundMode::CAST_RINT); // fp32->bf16
            SyncV();
            pto::TCVT(src1Tile, tempTile, pto::RoundMode::CAST_NONE); // bf16->fp32
            SyncV();
        }
        pto::TADD(dstTile, dstTile, src1Tile);
        // 当alpha不为1或index为int32类型时，需要在每一步运算后转换为bf16类型
        if (Std::is_same_v<typename T3::Type, int32_t> || abs(static_cast<float>(alpha) - 1) > TileOp::EPSILON) {
            SyncV();
            pto::TCVT(tempTile, dstTile, pto::RoundMode::CAST_RINT); // fp32->bf16
            SyncV();
            pto::TCVT(dstTile, tempTile, pto::RoundMode::CAST_NONE); // bf16->fp32
        }
    } else {
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        if (abs(static_cast<float>(alpha) - 1) > TileOp::EPSILON) {
            pto::TMULS(src1Tile, src1Tile, alpha);
            SyncV();
        }
        pto::TADD(dstTile, dstTile, src1Tile);
    }
}

template <typename T0, typename T2, typename T3, typename Scalar>
TILEOP void IndexAddUBLastAxisCompute(T0 dst, T2 src1, T3 src2, Scalar alpha, size_t src1Shape0, size_t src1Shape1,
                                      size_t src1Shape2, size_t src1Shape3, size_t src1Shape4, size_t dstStride0,
                                      size_t dstStride1, size_t dstStride2, size_t dstStride3, size_t src1Stride0,
                                      size_t src1Stride1, size_t src1Stride2, size_t src1Stride3)
{
    auto dstAddr = (__ubuf__ typename T0::Type*)((uint64_t)(dst.GetAddr()));
    auto src1Addr = (__ubuf__ typename T2::Type*)((uint64_t)(src1.GetAddr()));
    auto idxAddr = (__ubuf__ typename T3::Type*)((uint64_t)(src2.GetAddr()));
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    uint64_t dstOffset = 0;
    uint64_t src1Offset = 0;
    if (abs(static_cast<float>(alpha) - 1) > TileOp::EPSILON) {
        for (LoopVar i = 0; i < src1Shape0; ++i) {
            for (LoopVar j = 0; j < src1Shape1; ++j) {
                for (LoopVar k = 0; k < src1Shape2; ++k) {
                    for (LoopVar l = 0; l < src1Shape3; ++l) {
                        for (LoopVar idx = 0; idx < src1Shape4; ++idx) {
                            auto index = *(idxAddr + idx);
                            dstOffset = i * dstStride0 + j * dstStride1 + k * dstStride2 + l * dstStride3 + index;
                            src1Offset = i * src1Stride0 + j * src1Stride1 + k * src1Stride2 + l * src1Stride3 + idx;
                            if constexpr (Std::is_same_v<Scalar, half>) { // half
                                float mulsResult = static_cast<float>(src1Addr[src1Offset]) * static_cast<float>(alpha);
                                src1Addr[src1Offset] = static_cast<half>(mulsResult);
                            } else if constexpr (Std::is_same_v<Scalar, bfloat16_t>) { // bf16
                                float mulsResult = src1Addr[src1Offset] * TileOp::Bf16ToFp32(alpha);
                                bfloat16_t mulsResBf16 = TileOp::Fp32ToBf16R(mulsResult);
                                src1Addr[src1Offset] = TileOp::Bf16ToFp32(mulsResBf16);
                            } else { // int8,int16,int32,float32
                                Scalar mulsResult = static_cast<Scalar>(src1Addr[src1Offset]) * alpha;
                                src1Addr[src1Offset] = static_cast<typename T2::Type>(mulsResult);
                            }
                        }
                    }
                }
            }
        }
    }
    for (LoopVar i = 0; i < src1Shape0; ++i) {
        for (LoopVar j = 0; j < src1Shape1; ++j) {
            for (LoopVar k = 0; k < src1Shape2; ++k) {
                for (LoopVar l = 0; l < src1Shape3; ++l) {
                    for (LoopVar idx = 0; idx < src1Shape4; ++idx) {
                        auto index = *(idxAddr + idx);
                        dstOffset = i * dstStride0 + j * dstStride1 + k * dstStride2 + l * dstStride3 + index;
                        src1Offset = i * src1Stride0 + j * src1Stride1 + k * src1Stride2 + l * src1Stride3 + idx;
                        if constexpr (Std::is_same_v<Scalar, half>) {
                            float addResult = static_cast<float>(dstAddr[dstOffset]) +
                                              static_cast<float>(src1Addr[src1Offset]);
                            if (abs(static_cast<float>(alpha) - 1) < TileOp::EPSILON &&
                                Std::is_same_v<typename T3::Type, int64_t>) {
                                dstAddr[dstOffset] = addResult; // 不需要转换
                            } else {
                                dstAddr[dstOffset] = static_cast<half>(addResult);
                            }
                        } else if constexpr (Std::is_same_v<Scalar, bfloat16_t>) {
                            float addResult = dstAddr[dstOffset] + src1Addr[src1Offset];
                            if (abs(static_cast<float>(alpha) - 1) < TileOp::EPSILON &&
                                Std::is_same_v<typename T3::Type, int64_t>) {
                                dstAddr[dstOffset] = addResult; // 不需要转换
                            } else {
                                bfloat16_t addResBf16 = TileOp::Fp32ToBf16R(addResult);
                                dstAddr[dstOffset] = TileOp::Bf16ToFp32(addResBf16);
                            }
                        } else { // int8,int16,int32,float32
                            Scalar addResult = static_cast<Scalar>(dstAddr[dstOffset]) +
                                               static_cast<Scalar>(src1Addr[src1Offset]);
                            dstAddr[dstOffset] = static_cast<typename T0::Type>(addResult);
                        }
                    }
                }
            }
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
}

/*
src0:self
src1:source
src2:index
axis是泛化成5维后的值，实际值为 axis + shapeSize - 5
*/
template <int axis, typename T0, typename T1, typename T2, typename T3, typename T4, typename Scalar>
TILEOP void TIndexAddUB(T0 dst, T1 src0, T2 src1, T3 src2, T4 tempTensor, Scalar alpha)
{                                                                          // T0: tileTensor
    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value; // support 2-5
    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>(); // validShape
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto dstStride0 = dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto dstStride1 = dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto dstStride2 = dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    auto dstStride3 = dstLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();

    const auto src1Layout = src1.GetLayout();
    auto src1Shape0 = src1Layout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto src1Shape1 = src1Layout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto src1Shape2 = src1Layout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto src1Shape3 = src1Layout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto src1Shape4 = src1Layout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    auto src1Stride0 = src1Layout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto src1Stride1 = src1Layout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto src1Stride2 = src1Layout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    auto src1Stride3 = src1Layout.template GetStrideDim<DIM_4TH, MAX_DIMS>();

    auto dstAddr = (__ubuf__ typename T0::Type*)((uint64_t)(dst.GetAddr()));
    auto tempAddr = (__ubuf__ typename T4::Type*)((uint64_t)(tempTensor.GetAddr()));
    auto src1Addr = (__ubuf__ typename T2::Type*)((uint64_t)(src1.GetAddr()));
    auto idxAddr = (__ubuf__ typename T3::Type*)((uint64_t)(src2.GetAddr()));
    if (!dstShape0 || !dstShape1 || !dstShape2 || !dstShape3 || !dstShape4) {
        return;
    }
    if constexpr (axis == DIM_5TH) { // 尾轴
        IndexAddUBLastAxisCompute(dst, src1, src2, alpha, src1Shape0, src1Shape1, src1Shape2, src1Shape3, src1Shape4,
                                  dstStride0, dstStride1, dstStride2, dstStride3, src1Stride0, src1Stride1, src1Stride2,
                                  src1Stride3);
    } else {
        constexpr auto dstTileW = TileOp::GetAnyAxisMergeResult<
            axis + shapeSize - indexadd_detail::MERGED_TILE_AXIS_OFFSET, shapeSize, typename T0::TileShape>();
        constexpr auto tempTileW = TileOp::GetAnyAxisMergeResult<
            axis + shapeSize - indexadd_detail::MERGED_TILE_AXIS_OFFSET, shapeSize, typename T4::TileShape>();
        constexpr auto src1TileW = TileOp::GetAnyAxisMergeResult<
            axis + shapeSize - indexadd_detail::MERGED_TILE_AXIS_OFFSET, shapeSize, typename T2::TileShape>();
        using dstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, 1, dstTileW, pto::BLayout::RowMajor>;
        using tempTileDefine = pto::Tile<pto::TileType::Vec, bfloat16_t, 1, tempTileW, pto::BLayout::RowMajor>;
        using src1TileDefine = pto::Tile<pto::TileType::Vec, typename T2::Type, 1, src1TileW, pto::BLayout::RowMajor>;
        dstTileDefine dstTile;
        tempTileDefine tempTile;
        src1TileDefine src1Tile;
        if constexpr (axis == 0) { // 从第2轴开始合轴
            for (LoopVar i = 0; i < src1Shape0; ++i) {
                set_flag(PIPE_V, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
                auto index = *(idxAddr + i);
                auto dstOffset = index * dstStride0;
                auto src1Offset = i * src1Stride0;
                IndexAddUBNotLastAxisCompute<T0, T2, T3>(dstTile, tempTile, src1Tile, alpha, dstAddr, tempAddr,
                                                         src1Addr, dstOffset, src1Offset);
            }
        } else if constexpr (axis == 1) { // 从第3轴开始合轴
            for (LoopVar i = 0; i < src1Shape0; ++i) {
                for (LoopVar j = 0; j < src1Shape1; ++j) {
                    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
                    auto index = *(idxAddr + j);
                    auto dstOffset = i * dstStride0 + index * dstStride1;
                    auto src1Offset = i * src1Stride0 + j * src1Stride1;
                    IndexAddUBNotLastAxisCompute<T0, T2, T3>(dstTile, tempTile, src1Tile, alpha, dstAddr, tempAddr,
                                                             src1Addr, dstOffset, src1Offset);
                }
            }
        } else if constexpr (axis == DIM_3RD) { // 从第4轴开始合轴
            for (LoopVar i = 0; i < src1Shape0; ++i) {
                for (LoopVar j = 0; j < src1Shape1; ++j) {
                    for (LoopVar k = 0; k < src1Shape2; ++k) {
                        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
                        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
                        auto index = *(idxAddr + k);
                        auto dstOffset = i * dstStride0 + j * dstStride1 + index * dstStride2;
                        auto src1Offset = i * src1Stride0 + j * src1Stride1 + k * src1Stride2;
                        IndexAddUBNotLastAxisCompute<T0, T2, T3>(dstTile, tempTile, src1Tile, alpha, dstAddr, tempAddr,
                                                                 src1Addr, dstOffset, src1Offset);
                    }
                }
            }
        } else {
            for (LoopVar i = 0; i < src1Shape0; ++i) {
                for (LoopVar j = 0; j < src1Shape1; ++j) {
                    for (LoopVar k = 0; k < src1Shape2; ++k) {
                        for (LoopVar l = 0; l < src1Shape3; ++l) {
                            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
                            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
                            auto index = *(idxAddr + l);
                            auto dstOffset = i * dstStride0 + j * dstStride1 + k * dstStride2 + index * dstStride3;
                            auto src1Offset = i * src1Stride0 + j * src1Stride1 + k * src1Stride2 + l * src1Stride3;
                            IndexAddUBNotLastAxisCompute<T0, T2, T3>(dstTile, tempTile, src1Tile, alpha, dstAddr,
                                                                     tempAddr, src1Addr, dstOffset, src1Offset);
                        }
                    }
                }
            }
        }
    }
}

// indexadd in GM
template <typename T0, typename T2, typename dstGlobalData, typename tmpTileDefine, typename src1TileDefine,
          typename Scalar>
TILEOP void IndexAddNotLastAxisCompute(dstGlobalData dstGlobal, tmpTileDefine tmpTile, src1TileDefine src1Tile,
                                       Scalar alpha, __gm__ typename T0::Type* dstAddr,
                                       __ubuf__ typename T2::Type* tmpAddr, __ubuf__ typename T2::Type* src1Addr,
                                       size_t dstOffset, size_t src1Offset)
{
    pto::TASSIGN(dstGlobal, dstAddr + dstOffset);
    pto::TASSIGN(src1Tile, (uint64_t)(src1Addr + src1Offset));
    pto::TASSIGN(tmpTile, (uint64_t)(tmpAddr));
    if (abs(static_cast<float>(alpha) - 1) > TileOp::EPSILON) {
        if constexpr (Std::is_same_v<Scalar, int8_t>) {
            for (LoopVar idx = 0; idx < src1Tile.GetValidCol(); ++idx) {
                auto newSrc1Offset = src1Offset + idx;
                Scalar mulsResult = static_cast<Scalar>(src1Addr[newSrc1Offset]) * alpha;
                tmpAddr[idx] = mulsResult;
            }
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        } else {
            // int16,int32,fp16,fp32
            set_flag(PIPE_S, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
            pto::TMULS(tmpTile, src1Tile, alpha);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        }
        pto::TSTORE<tmpTileDefine, dstGlobalData, pto::AtomicType::AtomicAdd>(dstGlobal, tmpTile);
    } else {
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        pto::TSTORE<src1TileDefine, dstGlobalData, pto::AtomicType::AtomicAdd>(dstGlobal, src1Tile);
    }
}

template <typename T0, typename T2, typename dstGlobalData, typename tmpTileDefine, typename Scalar>
TILEOP void IndexAddLastAxisCompute(dstGlobalData dstGlobal, tmpTileDefine tmpTile, Scalar alpha,
                                    __gm__ typename T0::Type* dstAddr, __ubuf__ typename T2::Type* tmpAddr,
                                    __ubuf__ typename T2::Type* src1Addr, size_t dstOffset, size_t src1Offset)
{
    // 将src1单个元素加载到tmpTile起始位置
    if (abs(static_cast<float>(alpha) - 1) > TileOp::EPSILON) {
        if constexpr (Std::is_same_v<Scalar, half>) { // half
            float mulsResult = static_cast<float>(src1Addr[src1Offset]) * static_cast<float>(alpha);
            tmpAddr[0] = mulsResult;
        } else if constexpr (Std::is_same_v<Scalar, bfloat16_t>) { // bf16
            float mulsResult = src1Addr[src1Offset] * TileOp::Bf16ToFp32(alpha);
            tmpAddr[0] = mulsResult;
        } else { // int8,int16,int32,float32
            Scalar mulsResult = static_cast<Scalar>(src1Addr[src1Offset]) * alpha;
            tmpAddr[0] = mulsResult;
        }
    } else {
        tmpAddr[0] = src1Addr[src1Offset];
    }
    pto::TASSIGN(dstGlobal, dstAddr + dstOffset);
    pto::TASSIGN(tmpTile, (uint64_t)tmpAddr);
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    pto::TSTORE<tmpTileDefine, dstGlobalData, pto::AtomicType::AtomicAdd>(dstGlobal, tmpTile);
}

template <int axis, typename T3>
TILEOP size_t GetTileOffset(size_t dstStrides[], size_t idx[], __ubuf__ typename T3::Type* idxAddr)
{
    size_t dstOffset = 0;
    if constexpr (axis == 0) {
        dstOffset = *(idxAddr + idx[DIM_1ST]) * dstStrides[DIM_1ST] + idx[DIM_2ND] * dstStrides[DIM_2ND] +
                    idx[DIM_3RD] * dstStrides[DIM_3RD] + idx[DIM_4TH] * dstStrides[DIM_4TH];
    } else if constexpr (axis == 1) {
        dstOffset = idx[DIM_1ST] * dstStrides[DIM_1ST] + *(idxAddr + idx[DIM_2ND]) * dstStrides[DIM_2ND] +
                    idx[DIM_3RD] * dstStrides[DIM_3RD] + idx[DIM_4TH] * dstStrides[DIM_4TH];
    } else if constexpr (axis == DIM_3RD) {
        dstOffset = idx[DIM_1ST] * dstStrides[DIM_1ST] + idx[DIM_2ND] * dstStrides[DIM_2ND] +
                    *(idxAddr + idx[DIM_3RD]) * dstStrides[DIM_3RD] + idx[DIM_4TH] * dstStrides[DIM_4TH];
    } else if constexpr (axis == DIM_4TH) {
        dstOffset = idx[DIM_1ST] * dstStrides[DIM_1ST] + idx[DIM_2ND] * dstStrides[DIM_2ND] +
                    idx[DIM_3RD] * dstStrides[DIM_3RD] + *(idxAddr + idx[DIM_4TH]) * dstStrides[DIM_4TH];
    } else {
        dstOffset = idx[DIM_1ST] * dstStrides[DIM_1ST] + idx[DIM_2ND] * dstStrides[DIM_2ND] +
                    idx[DIM_3RD] * dstStrides[DIM_3RD] + idx[DIM_4TH] * dstStrides[DIM_4TH];
    }
    return dstOffset;
}

// A5 SIMT atomic scatter, one valid source row per launch.
#if defined(__DAV_V310)
template <int axis, typename T0, typename T1, typename T2, typename T3, typename T4, typename C, typename Scalar>
TILEOP void TIndexAddSimt(T0 dst, T1 src0, T2 src, T3 indices, T4 tmp, C coord, Scalar alpha)
{
    (void)src0;
    using ValueType = typename T2::Type;
    using IndexType = typename T3::Type;
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T2, DIM_5TH, MAX_DIMS>();
    constexpr auto tmpW = TileOp::GetTensorTileShapeDim<T4, DIM_5TH, MAX_DIMS>();
    using ValueTile = pto::Tile<pto::TileType::Vec, ValueType, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
    using OffsetTile = pto::Tile<pto::TileType::Vec, IndexType, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
    using TableShape = pto::Shape<1, 1, 1, 1, -1>;
    using TableStride = pto::Stride<-1, -1, -1, -1, 1>;
    using TableGlobal = pto::GlobalTensor<ValueType, TableShape, TableStride>;
    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    size_t dstShapes[] = {
        dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>(), dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>(),
        dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>(), dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>(),
        dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>()};
    size_t dstStrides[] = {
        dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>(), dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>(),
        dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>(), dstLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>(),
        dstLayout.template GetStrideDim<DIM_5TH, MAX_DIMS>()};
    size_t srcShapes[] = {
        srcLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>(), srcLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>(),
        srcLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>(), srcLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>(),
        srcLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>()};
    size_t srcStrides[] = {
        srcLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>(), srcLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>(),
        srcLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>(), srcLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>(),
        srcLayout.template GetStrideDim<DIM_5TH, MAX_DIMS>()};
    for (size_t dim = 0; dim < MAX_DIMS; ++dim) {
        if (dstShapes[dim] == 0 || srcShapes[dim] == 0) {
            return;
        }
    }
    auto dstAddr = (__gm__ ValueType*)dst.GetAddr() + dstLayout.template GetGmOffset<C, MAX_DIMS>(coord);
    auto srcAddr = (__ubuf__ ValueType*)src.GetAddr();
    auto idxAddr = (__ubuf__ IndexType*)indices.GetAddr();
    auto offsetAddr = (__ubuf__ IndexType*)tmp.GetAddr();
    auto valueAddr = (__ubuf__ ValueType*)((__ubuf__ float*)tmp.GetAddr() + tmpW);
    const unsigned validCol = static_cast<unsigned>(srcShapes[DIM_5TH]);
    ValueTile valuesTile(1U, validCol);
    ValueTile scaledTile(1U, validCol);
    OffsetTile offsetTile(1U, validCol);
    OffsetTile indicesTile(1U, validCol);
    pto::TASSIGN(offsetTile, (uint64_t)offsetAddr);
    pto::TASSIGN(scaledTile, (uint64_t)valueAddr);
    // OP_INDEX_ADD exposes MTE3 to the scheduler; bridge input readiness to V/S.
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    if constexpr (axis != DIM_5TH) {
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    }
    if constexpr (axis == DIM_5TH) {
        pto::TASSIGN(indicesTile, (uint64_t)idxAddr);
        pto::TMULS(offsetTile, indicesTile, static_cast<IndexType>(dstStrides[DIM_5TH]));
    } else {
        pto::TCI<OffsetTile, IndexType, 0>(offsetTile, static_cast<IndexType>(0));
        // A5 TCI writes UB on the scalar pipe; TMULS consumes it on the vector pipe.
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        pto::TMULS(offsetTile, offsetTile, static_cast<IndexType>(dstStrides[DIM_5TH]));
    }
    float alphaValue;
    if constexpr (Std::is_same_v<Scalar, bfloat16_t>) {
        alphaValue = TileOp::Bf16ToFp32(alpha);
    } else {
        alphaValue = static_cast<float>(alpha);
    }
    // MSCATTER Elem uses flat element offsets and ignores GlobalTensor strides.
    const auto rowSpan = (dstShapes[DIM_5TH] - 1) * dstStrides[DIM_5TH] + 1;
    for (LoopVar i = 0; i < srcShapes[0]; ++i) {
        for (LoopVar j = 0; j < srcShapes[1]; ++j) {
            for (LoopVar k = 0; k < srcShapes[DIM_3RD]; ++k) {
                for (LoopVar l = 0; l < srcShapes[DIM_4TH]; ++l) {
                    size_t row[] = {i, j, k, l};
                    auto dstOffset = GetTileOffset<axis, T3>(dstStrides, row, idxAddr);
                    auto srcOffset = GetTileOffset<DIM_5TH, T3>(srcStrides, row, idxAddr);
                    TableGlobal tableGM(dstAddr + dstOffset, TableShape(1, 1, 1, 1, rowSpan),
                                        TableStride(rowSpan, rowSpan, rowSpan, rowSpan, 1));
                    pto::TASSIGN(valuesTile, (uint64_t)(srcAddr + srcOffset));
                    // Vector and SIMT work share V; keep row-buffer dependencies on that pipe.
                    if (alphaValue != 1.0f) {
                        pto::TMULS(scaledTile, valuesTile, alpha);
                    }
                    if (alphaValue != 1.0f) {
                        pto::MSCATTER<pto::Coalesce::Elem, pto::ScatterAtomicOp::Add, pto::ScatterOOB::Skip,
                                      pto::ScatterConflict::Default>(tableGM, scaledTile, offsetTile);
                    } else {
                        pto::MSCATTER<pto::Coalesce::Elem, pto::ScatterAtomicOp::Add, pto::ScatterOOB::Skip,
                                      pto::ScatterConflict::Default>(tableGM, valuesTile, offsetTile);
                    }
                }
            }
        }
    }
    // Publish completion of vector/SIMT work on the scheduler-visible output pipe.
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
}
#endif

/*
dst: dst in GM
src0: self in GM
src1: source in UB
src2: index in UB
axis是泛化成5维后的值，实际值为 axis + shapeSize - 5
*/
template <int axis, typename T0, typename T1, typename T2, typename T3, typename T4, typename C, typename Scalar>
TILEOP void TIndexAdd(T0 dst, T1 src0, T2 src1, T3 src2, T4 tmpTensor, C coord, Scalar alpha)
{                                                                          // T0: tileTensor
    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value; // support 2-5
    const auto dstLayout = dst.GetLayout();
    size_t dstShapes[] = {
        dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>(), dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>(),
        dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>(), dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>(),
        dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>()}; // validShape
    if (!dstShapes[DIM_1ST] || !dstShapes[DIM_2ND] || !dstShapes[DIM_3RD] || !dstShapes[DIM_4TH] ||
        !dstShapes[DIM_5TH]) {
        return;
    }
    size_t dstStrides[] = {
        dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>(), dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>(),
        dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>(), dstLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>()};

    const auto src1Layout = src1.GetLayout();
    size_t src1Shapes[] = {
        src1Layout.template GetShapeDim<DIM_1ST, MAX_DIMS>(), src1Layout.template GetShapeDim<DIM_2ND, MAX_DIMS>(),
        src1Layout.template GetShapeDim<DIM_3RD, MAX_DIMS>(), src1Layout.template GetShapeDim<DIM_4TH, MAX_DIMS>(),
        src1Layout.template GetShapeDim<DIM_5TH, MAX_DIMS>()};
    size_t src1Strides[] = {
        src1Layout.template GetStrideDim<DIM_1ST, MAX_DIMS>(), src1Layout.template GetStrideDim<DIM_2ND, MAX_DIMS>(),
        src1Layout.template GetStrideDim<DIM_3RD, MAX_DIMS>(), src1Layout.template GetStrideDim<DIM_4TH, MAX_DIMS>()};
    using dstType = typename T0::Type;
    using src1Type = typename T2::Type;
    using idxType = typename T3::Type;
    using tmpType = typename T4::Type;
    auto dstAddr = (__gm__ dstType*)((uint64_t)(dst.GetAddr()));
    auto tmpAddr = (__ubuf__ tmpType*)((uint64_t)(tmpTensor.GetAddr()));
    auto src1Addr = (__ubuf__ src1Type*)((uint64_t)(src1.GetAddr()));
    auto idxAddr = (__ubuf__ idxType*)((uint64_t)(src2.GetAddr()));
    size_t gmOffset = static_cast<size_t>(dstLayout.template GetGmOffset<C, MAX_DIMS>(coord));
    dstAddr += gmOffset;
    constexpr auto tmpTileW = Std::tuple_element<1, typename T4::TileShape>::type::value;
    constexpr auto src1TileW = Std::tuple_element<shapeSize - 1, typename T2::TileShape>::type::value;
    using dstGlobalData = pto::GlobalTensor<dstType, pto::Shape<-1, -1, -1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>>;
    using tmpTileDefine = pto::Tile<pto::TileType::Vec, tmpType, 1, tmpTileW, pto::BLayout::RowMajor, -1, -1>;
    using src1TileDefine = pto::Tile<pto::TileType::Vec, src1Type, 1, src1TileW, pto::BLayout::RowMajor, -1, -1>;
    if constexpr (axis == DIM_5TH) { // 尾轴
        dstGlobalData dstGlobal(dstAddr, pto::Shape(1, 1, 1, 1, 1), pto::Stride(0, 0, 0, 0, 0));
        tmpTileDefine tmpTile(1, 1);
        for (LoopVar i = 0; i < src1Shapes[0]; ++i) {
            for (LoopVar j = 0; j < src1Shapes[1]; ++j) {
                for (LoopVar k = 0; k < src1Shapes[DIM_3RD]; ++k) {
                    for (LoopVar l = 0; l < src1Shapes[DIM_4TH]; ++l) {
                        for (LoopVar m = 0; m < src1Shapes[DIM_5TH]; ++m) {
                            set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                            wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                            size_t idx[] = {i, j, k, l};
                            auto dstOffset = GetTileOffset<axis, T3>(dstStrides, idx, idxAddr) + idxAddr[m];
                            auto src1Offset = GetTileOffset<axis, T3>(src1Strides, idx, idxAddr) + m;
                            IndexAddLastAxisCompute<T0, T2>(dstGlobal, tmpTile, alpha, dstAddr, tmpAddr, src1Addr,
                                                            dstOffset, src1Offset);
                        }
                    }
                }
            }
        }
    } else {
        dstGlobalData dstGlobal(dstAddr, pto::Shape(1, 1, 1, 1, dstShapes[DIM_5TH]), pto::Stride(0, 0, 0, 0, 0));
        tmpTileDefine tmpTile(1, src1Shapes[DIM_5TH]);
        src1TileDefine src1Tile(1, src1Shapes[DIM_5TH]);
        for (LoopVar i = 0; i < src1Shapes[0]; ++i) {
            for (LoopVar j = 0; j < src1Shapes[1]; ++j) {
                for (LoopVar k = 0; k < src1Shapes[DIM_3RD]; ++k) {
                    for (LoopVar l = 0; l < src1Shapes[DIM_4TH]; ++l) {
                        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                        size_t idx[] = {i, j, k, l};
                        auto dstOffset = GetTileOffset<axis, T3>(dstStrides, idx, idxAddr);
                        auto src1Offset = GetTileOffset<DIM_5TH, T3>(src1Strides, idx, idxAddr);
                        IndexAddNotLastAxisCompute<T0, T2>(dstGlobal, tmpTile, src1Tile, alpha, dstAddr, tmpAddr,
                                                           src1Addr, dstOffset, src1Offset);
                    }
                }
            }
        }
    }
}
#endif
