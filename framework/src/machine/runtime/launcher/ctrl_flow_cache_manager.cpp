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
 * \file ctrl_flow_cache_manager.cpp
 * \brief Implementation of CtrlFlowCacheManager.
 */

#include "machine/runtime/launcher/ctrl_flow_cache_manager.h"

#include <sstream>

#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_code.h"
#include "interface/function/function.h"
#include "machine/runtime/launcher/device_launcher.h"
#include "machine/runtime/runner/kernel_binary.h"
#include "machine/runtime/runner/runtime_utils.h"
#include "machine/utils/dynamic/dev_encode_program.h"
#include "machine/utils/dynamic/dev_encode_program_ctrlflow_cache.h"

namespace {

uint64_t ComputeValueDependHash(const std::vector<npu::tile_fwk::dynamic::DeviceTensorData>& tensors,
                                const std::vector<size_t>& indices)
{
    uint64_t h = 14695981039346656037ull;
    for (size_t idx : indices) {
        if (idx >= tensors.size()) {
            continue;
        }
        const auto& shape = tensors[idx].GetShape();
        for (auto dim : shape) {
            h ^= static_cast<uint64_t>(dim);
            h *= 1099511628211ull;
        }
        auto addr = reinterpret_cast<const uint8_t*>(tensors[idx].GetAddr());
        int64_t dataSize = tensors[idx].GetDataSize();
        if (addr != nullptr && dataSize > 0) {
            for (int64_t i = 0; i < dataSize; i++) {
                h ^= addr[i];
                h *= 1099511628211ull;
            }
        }
    }
    return h;
}

} // namespace

namespace npu::tile_fwk::dynamic {

CtrlFlowCacheManager& CtrlFlowCacheManager::Instance()
{
    static CtrlFlowCacheManager instance;
    return instance;
}

uint8_t* CtrlFlowCacheManager::FindOrBuildDevCache(KernelBinary* kernel, std::vector<DeviceTensorData>& tensors,
                                                   bool isCaptureMode)
{
    // Device RT entry: skip find/build entirely for value-depend programs.
    // Actual Emulation build path is separately gated in EmulationLauncher.
    if (kernel->DisableHostCtrlFlowCacheBuild()) {
        COMPILER_LOGI("Skip host control flow cache build due to disableCtrlFlowCache.");
        kernel->SetCtrlFlowCacheReplay(false);
        return nullptr;
    }
    const auto& valueDependIndices = kernel->GetValueDependInputIndices();
    if (!kernel->HasValueDepend()) {
        // No value-depend: original shape-based cache reuse logic (unchanged).
        auto devCache = kernel->FindCtrlFlowCache(tensors, true);
        if (devCache == nullptr) {
            AclModeGuard guard(AclMdlRICaptureMode::RELAXED);
            devCache = kernel->BuildControlFlowCache(tensors, true);
        }
        // Host-built cache is already isActivated when copied to device; restore runs on first launch.
        kernel->SetCtrlFlowCacheReplay(devCache != nullptr);
        COMPILER_LOGD("find ctrlflow cache: %p", devCache);
        return devCache;
    }
    if (isCaptureMode) {
        return nullptr;
    }
    // Value-depend with cpu tensors: single-slot cache keyed by data+shape hash.
    uint64_t currentHash = ComputeValueDependHash(tensors, valueDependIndices);
    uint64_t cachedHash = kernel->GetCachedCtrlFlowHash();
    if (cachedHash == currentHash && kernel->GetValueDependDevCache() != nullptr) {
        kernel->SetCtrlFlowCacheReplay(true);
        return kernel->GetValueDependDevCache();
    }
    // Hash mismatch or first build: free old, rebuild, update hash.
    kernel->FreeAndClearValueDependCache();
    AclModeGuard guard(AclMdlRICaptureMode::RELAXED);
    auto devCache = kernel->BuildControlFlowCache(tensors, true, false);
    kernel->SetValueDependDevCache(devCache);
    kernel->SetCachedCtrlFlowHash(currentHash);
    kernel->SetCtrlFlowCacheReplay(devCache != nullptr);
    return devCache;
}

DevControlFlowCache* CtrlFlowCacheManager::GetHostCtrlFlowCache(KernelBinary* kernel,
                                                                std::vector<DeviceTensorData>& tensors,
                                                                uint8_t* devCache, CtrlFlowCacheBlob& hostCache)
{
    DevControlFlowCache* ctrlCache = FindHostCtrlFlowCache(kernel, tensors, hostCache);
    if (ctrlCache == nullptr && devCache != nullptr) {
        auto devProg = reinterpret_cast<DevAscendProgram*>(
            kernel->GetFunction()->GetDyndevAttribute()->devProgBinary.data());
        size_t ctrlCacheSize = devProg->ctrlFlowCacheSize;
        CtrlFlowCacheBlob hostCacheVec;
        hostCacheVec.resize(ctrlCacheSize);
        // 维测：blob 基址必须 64B 对齐（CtrlFlowCacheBlob 用 AlignedAllocator<uint8_t,0x40> 保证）。
        MACHINE_LOGI("#ctrl.cache.align: site=GetHostCtrlFlowCache.build base=%p size=%zu align64=%lu %s",
                     static_cast<void*>(hostCacheVec.data()), ctrlCacheSize,
                     static_cast<unsigned long>(reinterpret_cast<uintptr_t>(hostCacheVec.data()) %
                                                npu::tile_fwk::DUPPED_STITCH_NODE_ALIGN),
                     reinterpret_cast<uintptr_t>(hostCacheVec.data()) % npu::tile_fwk::DUPPED_STITCH_NODE_ALIGN == 0 ?
                         "OK" :
                         "MISALIGNED");
        AclModeGuard guard(AclMdlRICaptureMode::RELAXED);
        RuntimeMemcpy(hostCacheVec.data(), ctrlCacheSize, devCache, ctrlCacheSize, RtMemcpyKind::DEVICE_TO_HOST);
        AddHostCtrlFlowCache(kernel, tensors, std::move(hostCacheVec));
        ctrlCache = FindHostCtrlFlowCache(kernel, tensors, hostCache);
    }
    return ctrlCache;
}

DevControlFlowCache* CtrlFlowCacheManager::FindHostCtrlFlowCache(KernelBinary* kernel,
                                                                 std::vector<DeviceTensorData>& tensors,
                                                                 CtrlFlowCacheBlob& hostCache)
{
    int64_t hash = ControlFlowCache::Hash(tensors);
    for (auto& cache : kernel->GetHostCtrlFlowCaches()) {
        if (cache.hash == hash) {
            hostCache = cache.hostCache;
            // 维测：值拷贝后 data() 是新地址，必须仍然 64B 对齐（同一 allocator，故成立）
            MACHINE_LOGI("#ctrl.cache.align: site=FindHostCtrlFlowCache.copy base=%p size=%zu align64=%lu %s",
                         static_cast<void*>(hostCache.data()), hostCache.size(),
                         static_cast<unsigned long>(reinterpret_cast<uintptr_t>(hostCache.data()) %
                                                    npu::tile_fwk::DUPPED_STITCH_NODE_ALIGN),
                         reinterpret_cast<uintptr_t>(hostCache.data()) % npu::tile_fwk::DUPPED_STITCH_NODE_ALIGN == 0 ?
                             "OK" :
                             "MISALIGNED");
            return reinterpret_cast<DevControlFlowCache*>(hostCache.data());
        }
    }
    return nullptr;
}

void CtrlFlowCacheManager::AddHostCtrlFlowCache(KernelBinary* kernel, std::vector<DeviceTensorData>& tensors,
                                                CtrlFlowCacheBlob&& hostCache)
{
    kernel->GetHostCtrlFlowCaches().emplace_back(tensors, std::move(hostCache));
}

} // namespace npu::tile_fwk::dynamic
