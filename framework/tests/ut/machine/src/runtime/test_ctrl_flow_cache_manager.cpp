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
 * \file test_ctrl_flow_cache_manager.cpp
 * \brief UT for machine/runtime/launcher/ctrl_flow_cache_manager.h — struct and singleton coverage
 */

#include <gtest/gtest.h>
#define private public
#include "machine/runtime/launcher/ctrl_flow_cache_manager.h"
#undef private

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

TEST(CtrlFlowCacheManagerTest, Singleton)
{
    auto& inst1 = CtrlFlowCacheManager::Instance();
    auto& inst2 = CtrlFlowCacheManager::Instance();
    EXPECT_EQ(&inst1, &inst2);
}

TEST(CtrlFlowCacheManagerTest, ControlFlowCache_Hash_DeviceTensorData)
{
    std::vector<DeviceTensorData> empty;
    auto hashEmpty = ControlFlowCache::Hash(empty);
    EXPECT_NE(hashEmpty, 0);

    std::vector<DeviceTensorData> datas = {{DT_FP32, nullptr, {2, 4}}};
    auto hashNonEmpty = ControlFlowCache::Hash(datas);
    EXPECT_NE(hashNonEmpty, hashEmpty);

    std::vector<DeviceTensorData> sameDatas = {{DT_FP32, nullptr, {2, 4}}};
    auto hashSame = ControlFlowCache::Hash(sameDatas);
    EXPECT_EQ(hashNonEmpty, hashSame);
}

TEST(CtrlFlowCacheManagerTest, ControlFlowCache_Hash_Shapes)
{
    std::vector<std::vector<int64_t>> empty;
    auto hashEmpty = ControlFlowCache::Hash(empty);

    std::vector<std::vector<int64_t>> shapes = {{2, 4}};
    auto hashNonEmpty = ControlFlowCache::Hash(shapes);
    EXPECT_NE(hashNonEmpty, hashEmpty);

    std::vector<std::vector<int64_t>> sameShapes = {{2, 4}};
    auto hashSame = ControlFlowCache::Hash(sameShapes);
    EXPECT_EQ(hashNonEmpty, hashSame);
}

TEST(CtrlFlowCacheManagerTest, ControlFlowCache_Constructor)
{
    std::vector<DeviceTensorData> datas = {{DT_FP32, nullptr, {2, 4}}, {DT_INT32, nullptr, {8}}};
    auto* cachePtr = reinterpret_cast<uint8_t*>(0x2000);
    ControlFlowCache cache(datas, cachePtr);
    EXPECT_EQ(cache.devCache, cachePtr);
    EXPECT_EQ(cache.inputs.size(), 2u);
    EXPECT_NE(cache.hash, 0);
}

TEST(CtrlFlowCacheManagerTest, HostControlFlowCache_Constructor)
{
    std::vector<DeviceTensorData> datas = {{DT_FP32, nullptr, {2, 4}}};
    CtrlFlowCacheBlob hostCache = {1, 2, 3, 4, 5};
    HostControlFlowCache cache(datas, std::move(hostCache));
    EXPECT_EQ(cache.hostCache.size(), 5u);
    EXPECT_EQ(cache.hostCache[0], 1);
    EXPECT_NE(cache.hash, 0);
}
