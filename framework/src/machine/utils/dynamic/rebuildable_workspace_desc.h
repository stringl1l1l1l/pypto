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
 * \file rebuildable_workspace_desc.h
 * \brief Encode-time WorkspaceDesc snapshot and stitch-pool helpers attached via RebuildableAttribute.
 */

#pragma once

#include <cstdint>
#include <string>

#include "interface/function/rebuildable_attribute.h"
#include "tilefwk/workspace_desc.h"

namespace npu::tile_fwk {

inline constexpr uint32_t STITCH_POOL_ROOT_INNER = 0;
inline constexpr uint32_t STITCH_POOL_ASSEMBLE_OUTCAST = 1;
inline constexpr uint32_t STITCH_POOL_EXCLUSIVE_OUTCAST = 2;

inline bool HasPreciseStitchFunctionNumPerPool(const uint32_t (&pool)[STITCH_FUNCTION_NUM_PER_POOL_SIZE])
{
    return pool[STITCH_POOL_ROOT_INNER] != 0 || pool[STITCH_POOL_ASSEMBLE_OUTCAST] != 0 ||
           pool[STITCH_POOL_EXCLUSIVE_OUTCAST] != 0;
}

inline uint32_t MaxOfStitchFunctionNumPerPool(const uint32_t (&pool)[STITCH_FUNCTION_NUM_PER_POOL_SIZE])
{
    uint32_t maxNum = pool[STITCH_POOL_ROOT_INNER];
    if (pool[STITCH_POOL_ASSEMBLE_OUTCAST] > maxNum) {
        maxNum = pool[STITCH_POOL_ASSEMBLE_OUTCAST];
    }
    if (pool[STITCH_POOL_EXCLUSIVE_OUTCAST] > maxNum) {
        maxNum = pool[STITCH_POOL_EXCLUSIVE_OUTCAST];
    }
    return maxNum;
}

inline void FillStitchFunctionNumPerPool(uint32_t (&pool)[STITCH_FUNCTION_NUM_PER_POOL_SIZE], uint32_t value)
{
    pool[STITCH_POOL_ROOT_INNER] = value;
    pool[STITCH_POOL_ASSEMBLE_OUTCAST] = value;
    pool[STITCH_POOL_EXCLUSIVE_OUTCAST] = value;
}

inline void CopyStitchFunctionNumPerPool(uint32_t (&dst)[STITCH_FUNCTION_NUM_PER_POOL_SIZE],
                                         const uint32_t (&src)[STITCH_FUNCTION_NUM_PER_POOL_SIZE])
{
    dst[STITCH_POOL_ROOT_INNER] = src[STITCH_POOL_ROOT_INNER];
    dst[STITCH_POOL_ASSEMBLE_OUTCAST] = src[STITCH_POOL_ASSEMBLE_OUTCAST];
    dst[STITCH_POOL_EXCLUSIVE_OUTCAST] = src[STITCH_POOL_EXCLUSIVE_OUTCAST];
}

struct RebuildableWorkspaceDesc : RebuildableAttribute<WorkspaceDesc> {
    const char* Name() const override { return "WorkspaceDesc"; }
    uint64_t GetSizeForCheckOnly(uint64_t maxDynamicAssembleOutcastMem, uint64_t debugSize) const;
    std::string PrettyDumpSize(uint64_t maxDynamicAssembleOutcastMem, uint64_t debugSize) const;
};

} // namespace npu::tile_fwk
