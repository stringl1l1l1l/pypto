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
 * \file tile_tensor.h
 * \brief
 */

#ifndef TILEOP_UTILS_TILE_TENSOR_H
#define TILEOP_UTILS_TILE_TENSOR_H

#include "common_type.h"
#include <type_traits>

template <typename... Indexs>
using Offsets = Std::tuple<Indexs...>;

using TileOffset = Offsets<size_t, size_t, size_t>;
using TileOffset4Dim = Offsets<size_t, size_t, size_t, size_t>;

template <typename T, typename LA, Hardware FMT = Hardware::UB>
struct TileTensor {
    using Type = T;
    using LayoutType = LA;
    using Shape = typename LA::Shape;
    using Stride = typename LA::Stride;
    using TileShape = typename LA::TileShape;
    static constexpr Hardware FORMAT = FMT;

    __aicore__ inline TileTensor() {}
    __aicore__ inline TileTensor(uint64_t addr, LA layout) : addr_(addr), layout_(layout) {}
    __aicore__ inline TileTensor(uint64_t addr, Shape shape) : addr_(addr), layout_(LA(shape)) {}

    __aicore__ inline TileTensor(uint64_t addr) : addr_(addr) {}
    __aicore__ inline uint64_t GetAddr() { return addr_; }
    __aicore__ inline void SetAddr(uint64_t addr) { addr_ = addr; }
    __aicore__ inline LA GetLayout() { return layout_; }
    __aicore__ inline const LA GetLayout() const { return layout_; }
    __aicore__ inline static constexpr bool IsStaticLayout() { return LayoutType::IsStaticLayout(); }
    __aicore__ inline constexpr Hardware GetPhyType() { return FORMAT; }
    __aicore__ inline Shape GetShape() { return layout_.GetShape(); }
    __aicore__ inline Stride GetStride() { return layout_.GetStride(); }

    __aicore__ inline uint64_t GetLinearAddr(const TileOffset& offsets = TileOffset(0, 0, 0))
    {
        size_t offset = Std::get<DIM_1ST>(offsets) * layout_.template GetStrideDim<DIM_1ST, MAX_DIMS>();
        offset += Std::get<DIM_2ND>(offsets) * layout_.template GetStrideDim<DIM_2ND, MAX_DIMS>();
        offset += Std::get<DIM_3RD>(offsets) * layout_.template GetStrideDim<DIM_3RD, MAX_DIMS>();
#if defined PTO_NPU_ARCH_A5
        if constexpr (std::is_same_v<Type, float4_e1m2x2_t> || std::is_same_v<Type, float4_e2m1x2_t>) {
            offset >>= 1;
        }
#endif
        return addr_ + offset * sizeof(Type);
    }

private:
    uint64_t addr_;
    LA layout_;
};

template <typename T, typename LA>
struct TileTensor<T, LA, Hardware::GM> {
    using Type = T;
    using LayoutType = LA;
    using Shape = typename LA::Shape;
    using Stride = typename LA::Stride;
    using TileShape = typename LA::TileShape;
    static constexpr Hardware FORMAT = Hardware::GM;

    __aicore__ inline TileTensor() {}
    __aicore__ inline TileTensor(T* addr, LA layout) : addr_(addr), layout_(layout) {}
    __aicore__ inline TileTensor(T* addr, Shape shape) : addr_(addr), layout_(LA(shape)) {}

    __aicore__ inline TileTensor(T* addr) : addr_(addr) {}
    __aicore__ inline T* GetAddr() { return addr_; }
    __aicore__ inline void SetAddr(T* addr) { addr_ = addr; }
    __aicore__ inline LA GetLayout() { return layout_; }
    __aicore__ inline const LA GetLayout() const { return layout_; }
    __aicore__ inline static constexpr bool IsStaticLayout() { return LayoutType::IsStaticLayout(); }
    __aicore__ inline constexpr Hardware GetPhyType() { return FORMAT; }
    __aicore__ inline Shape GetShape() { return layout_.GetShape(); }
    __aicore__ inline Stride GetStride() { return layout_.GetStride(); }

private:
    T* addr_;
    LA layout_;
};

template <typename T>
struct TensorTileInfo {
    static constexpr auto tile0 = TileOp::GetTensorTileShapeDim<T, DIM_1ST, MAX_DIMS>();
    static constexpr auto tile1 = TileOp::GetTensorTileShapeDim<T, DIM_2ND, MAX_DIMS>();
    static constexpr auto tile2 = TileOp::GetTensorTileShapeDim<T, DIM_3RD, MAX_DIMS>();
    static constexpr auto tileH = TileOp::GetTensorTileShapeDim<T, DIM_4TH, MAX_DIMS>();
    static constexpr auto tileW = TileOp::GetTensorTileShapeDim<T, DIM_5TH, MAX_DIMS>();
};

template <typename T, typename LA, Hardware FMT = Hardware::UB>
TILEOP TileTensor<T, LA, FMT> MakeTensor(__ubuf__ T* addr, LA layout)
{
    return TileTensor<T, LA, FMT>(addr, layout);
}
#endif // TILEOP_UTILS_TILE_TENSOR_H
