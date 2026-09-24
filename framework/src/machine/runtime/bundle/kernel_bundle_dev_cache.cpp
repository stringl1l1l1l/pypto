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
 * \file kernel_bundle_dev_cache.cpp
 * \brief Implementation of the per-bundle control-flow-cache device buffer singleton.
 */

#include "machine/runtime/bundle/kernel_bundle_dev_cache.h"

#include "machine/runtime/memory_utils/device_memory_utils.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk::bundle {

KernelBundleDevCache& KernelBundleDevCache::Instance()
{
    static KernelBundleDevCache instance;
    return instance;
}

uint8_t* KernelBundleDevCache::GetOrCopy(uint64_t key, const std::vector<uint8_t>& hostCtrlCache)
{
    if (hostCtrlCache.empty()) {
        return nullptr; // launch path rebuilds the cache at runtime when absent
    }
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        return it->second; // reuse the device buffer copied on a previous launch
    }
    // Huge-page path -> DevMemoryPool, so the destructor free stays consistent with the sentinel-checked teardown.
    dynamic::DeviceMemoryUtils devMem(true);
    uint8_t* dev = devMem.CopyToDev(const_cast<uint8_t*>(hostCtrlCache.data()), hostCtrlCache.size(), nullptr);
    cache_[key] = dev;
    MACHINE_LOGI("[kernel-bundle] dev-cache: copied ctrl-flow cache key=%#lx size=%zuB (cached, reused hereafter)", key,
                 hostCtrlCache.size());
    // 维测：bundle 来路的 device blob 基址对齐。走 DevMemoryPool（huge page，2MB 对齐）+
    // MemoryBlock::Allocate 每块只服务一次直接返回 baseAddr，故天然满足；此处留观测点。
    MACHINE_LOGI("#ctrl.cache.align: site=KernelBundleDevCache.GetOrCopy base=%p size=%zu align64=%lu %s",
                 static_cast<void*>(dev), hostCtrlCache.size(),
                 static_cast<unsigned long>(reinterpret_cast<uintptr_t>(dev) % npu::tile_fwk::DUPPED_STITCH_NODE_ALIGN),
                 reinterpret_cast<uintptr_t>(dev) % npu::tile_fwk::DUPPED_STITCH_NODE_ALIGN == 0 ? "OK" : "MISALIGNED");
    return dev;
}

uint8_t* KernelBundleDevCache::GetOrAllocCellMatch(uint64_t key, uint64_t bytes)
{
    if (bytes == 0) {
        return nullptr; // op has no dynamic cell-match table for this shape
    }
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = cellMatchCache_.find(key);
    if (it != cellMatchCache_.end() && it->second.addr != nullptr && it->second.capacity >= bytes) {
        return it->second.addr; // reuse the buffer allocated on a previous launch
    }
    dynamic::DeviceMemoryUtils devMem(true);
    if (it != cellMatchCache_.end() && it->second.addr != nullptr) {
        devMem.Free(it->second.addr); // a later shape needs a bigger pool: release the smaller buffer first
    }
    uint8_t* dev = devMem.AllocDev(static_cast<size_t>(bytes), nullptr);
    cellMatchCache_[key] = CellMatchBuf{dev, bytes};
    MACHINE_LOGI("[kernel-bundle] dev-cache: alloc cell-match pool key=%#lx size=%luB (cached, reused hereafter)", key,
                 static_cast<unsigned long>(bytes));
    return dev;
}

KernelBundleDevCache::~KernelBundleDevCache()
{
    // No free here: DevMemoryPool is a static destroyed before this one, and its DestroyPool() has already
    // released every block. Freeing again would only hit an empty addrToBlock_ and raise FREE_FAILED.
    cache_.clear();
    cellMatchCache_.clear();
}

} // namespace npu::tile_fwk::bundle
