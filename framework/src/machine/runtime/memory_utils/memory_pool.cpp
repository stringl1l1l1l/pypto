/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "machine/runtime/memory_utils/memory_pool.h"
#include <iomanip>
#include <optional>
#include <sstream>
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_code.h"
#include "adapter/api/runtime_api.h"
#include "machine/runtime/launcher/device_launcher.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {
namespace {
inline constexpr int RTMALLOC_SUCCESS = 0;
inline constexpr uint64_t SENTINEL_VALUE = 0xDEADBEEFDEADBEEF;
inline constexpr uint32_t SENTINEL_NUM = 64;
inline constexpr uint32_t SENTINEL_MEM_SIZE = 512;
inline uint64_t MemSizeAlign(const uint64_t bytes, const uint32_t aligns = 512U)
{
    const uint64_t alignSize = (aligns == 0U) ? sizeof(uintptr_t) : aligns;
    return (((bytes + alignSize) - 1U) / alignSize) * alignSize;
}
} // namespace

RtError NormalizedRtMemcpy(void* dst, uint64_t destMax, const void* src, uint64_t cnt, RtMemcpyKind kind)
{
    std::optional<dynamic::AclModeGuard> captureRelaxGuard;
    if (dynamic::DeviceLauncher::IsCaptureMode()) {
        captureRelaxGuard.emplace(AclMdlRICaptureMode::RELAXED);
    }
    return RuntimeMemcpyDirect(dst, destMax, src, cnt, kind);
}

MemoryBlock::MemoryBlock(void* addr, size_t size) : baseAddr(addr), blockSize(size), usedSize(0) {}

void* MemoryBlock::Allocate(uint64_t alignSize)
{
    if (usedSize == 0 && blockSize >= alignSize) {
        usedSize = blockSize;
        return baseAddr;
    }
    return nullptr;
}

DevMemoryPool& DevMemoryPool::Instance()
{
    static DevMemoryPool memoryPool;
    return memoryPool;
}

DevMemoryPool::DevMemoryPool()
{
    needMemCheck_ = config::IsRuntimeDebugAllEnabled();
    sentinelVec_ = std::vector<uint64_t>(SENTINEL_NUM, SENTINEL_VALUE);
}

DevMemoryPool::~DevMemoryPool()
{
    CheckAllSentinels();
    DestroyPool();
}

void DevMemoryPool::AllocDevAddr(uint8_t** devAddr, const uint64_t size)
{
    if (!AllocDevAddrInPool(devAddr, size)) {
        MACHINE_LOGE(DevCommonErr::ALLOC_FAILED, "AllocDevAddrInPool failed for size %lu bytes", size);
    } else {
        MACHINE_LOGI("RuntimeAgentMemory: Alloc success %p", *devAddr);
    }
}

bool DevMemoryPool::AllocDevAddrInPool(uint8_t** devAddr, const uint64_t size)
{
    if (size == 0)
        return false;
    if (devAddr == nullptr) {
        MACHINE_LOGE(DevCommonErr::NULLPTR, "devAddr is nullptr");
        return false;
    }
    auto alignSize = MemSizeAlign(size);
    if (needMemCheck_) {
        alignSize += SENTINEL_MEM_SIZE;
    }

    for (auto& block : memoryBlocks_) {
        void* ptr = block->Allocate(alignSize);
        if (ptr != nullptr) {
            *devAddr = static_cast<uint8_t*>(ptr);
            RecordAllocation(ptr, block);
            PutSentinelAddr(*devAddr, size);
            return true;
        }
    }

    MemoryBlock* newBlock = CreateNewBlock(alignSize);
    if (newBlock != nullptr) {
        void* ptr = newBlock->Allocate(alignSize);
        if (ptr != nullptr) {
            *devAddr = static_cast<uint8_t*>(ptr);
            RecordAllocation(ptr, newBlock);
            PutSentinelAddr(*devAddr, size);
            return true;
        }
    }

    MACHINE_LOGE(DevCommonErr::ALLOC_FAILED, "Allocate failed: size=%lu bytes", size);
    return false;
}

void DevMemoryPool::FreeDevAddr(void* ptr)
{
    if (ptr == nullptr) {
        MACHINE_LOGE(DevCommonErr::NULLPTR, "Freeing nullptr");
        return;
    }
    CheckSentinel(static_cast<uint8_t*>(ptr), true);

    auto it = addrToBlock_.find(ptr);
    if (it == addrToBlock_.end()) {
        MACHINE_LOGE(DevCommonErr::FREE_FAILED, "Freeing unknown pointer: %p", ptr);
        return;
    }

    MemoryBlock* block = it->second;

    MACHINE_LOGI("Directly freeing 2MB block: addr=%p.", block->baseAddr);
    FreeMemBlock(block);
    for (auto vecIt = memoryBlocks_.begin(); vecIt != memoryBlocks_.end(); ++vecIt) {
        if (*vecIt == block) {
            memoryBlocks_.erase(vecIt);
            break;
        }
    }

    addrToBlock_.erase(it);
}

void DevMemoryPool::PutSentinelAddr(uint8_t* baseAddr, uint64_t baseSize)
{
    if (needMemCheck_) {
        uint8_t* sentinelAddr = baseAddr + baseSize;
        if (NormalizedRtMemcpy(sentinelAddr, SENTINEL_MEM_SIZE, sentinelVec_.data(), SENTINEL_MEM_SIZE,
                               RtMemcpyKind::HOST_TO_DEVICE) != 0) {
            MACHINE_LOGW("Memory copy sentinel value failed! Do not check memory.");
            return;
        }
        MACHINE_LOGI("Base addr add: baseAddr=%p, sentinelAddr=%p.", baseAddr, sentinelAddr);
        sentinelValMap_[baseAddr].push_back(sentinelAddr);
    }
}

bool DevMemoryPool::CheckAllSentinels()
{
    if (!needMemCheck_) {
        return true;
    }
    bool allGood = true;
    for (auto& iter : sentinelValMap_) {
        if (!CheckSentinel(iter.first, false)) {
            allGood = false;
        }
    }
    if (!allGood) {
        MACHINE_LOGW("CheckAllSentinels failed.");
    }

    return allGood;
}

void DevMemoryPool::PrintSentinelVal(std::vector<uint64_t>& sentinelVal, uint8_t* sentinelAddr)
{
    std::ostringstream oss;
    uint8_t* bytePtr = reinterpret_cast<uint8_t*>(sentinelVal.data());
    oss << "Print Sentinel val in hex with ori val[" << std::hex << "0x" << SENTINEL_VALUE << "]" << std::endl;
    MACHINE_LOGW("%s", oss.str().c_str());
    oss.str("");
    for (uint32_t i = 0; i < SENTINEL_MEM_SIZE; ++i) {
        oss << std::hex << std::setw(2) << std::setfill('0') << (int)bytePtr[i];
        if ((i + 1) % 16 == 0) {
            oss << std::endl;
        } else {
            oss << " ";
        }
        if ((i + 1) % 64 == 0) {
            MACHINE_LOGW("Sentinel Addr:%p Val:[\n%s]", sentinelAddr + i, oss.str().c_str());
            oss.str("");
        }
    }
}

bool DevMemoryPool::CheckSentinel(uint8_t* baseAddr, bool remove)
{
    if (!needMemCheck_ || sentinelValMap_.empty()) {
        return true;
    }
    if (baseAddr == reinterpret_cast<uint8_t*>(0x12345678)) {
        return true;
    }
    auto iter = sentinelValMap_.find(baseAddr);
    if (iter == sentinelValMap_.end()) {
        MACHINE_LOGW("Base addr %p not found in map, need check code.", baseAddr);
        return false;
    }
    std::vector<uint64_t> sentinelVal(SENTINEL_NUM, 0);
    bool allGood = true;
    auto& sentinelVec = iter->second;
    for (auto sentinelAddr : sentinelVec) {
        MACHINE_LOGI("Check Sentinel: baseAddr=%p, sentinelAddr=%p.", baseAddr, sentinelAddr);
        if (NormalizedRtMemcpy(sentinelVal.data(), SENTINEL_MEM_SIZE, sentinelAddr, SENTINEL_MEM_SIZE,
                               RtMemcpyKind::DEVICE_TO_HOST) != 0) {
            MACHINE_LOGW("Memory copy D2H failed! Do not check memory.");
            break;
        }
        if (memcmp(sentinelVal.data(), sentinelVec_.data(), SENTINEL_MEM_SIZE) != 0) {
            PrintSentinelVal(sentinelVal, sentinelAddr);
            allGood = false;
        }
    }
    if (!allGood) {
        MACHINE_LOGW("BaseAddr:%p check sentinel failed.", baseAddr);
    } else {
        MACHINE_LOGI("BaseAddr:%p check sentinel Ok.", baseAddr);
    }
    if (remove) {
        sentinelValMap_.erase(baseAddr);
    }
    return allGood;
}

void DevMemoryPool::DynamicRecycle()
{
    auto it = memoryBlocks_.begin();
    while (it != memoryBlocks_.end()) {
        if ((*it)->usedSize == 0) {
            MACHINE_LOGI("Recycling empty block: addr=%p", (*it)->baseAddr);
            FreeMemBlock(*it);
            it = memoryBlocks_.erase(it);
        } else {
            ++it;
        }
    }
}

void DevMemoryPool::DestroyPool()
{
    for (auto& block : memoryBlocks_) {
        if (block != nullptr) {
            FreeMemBlock(block);
        }
    }
    memoryBlocks_.clear();
    addrToBlock_.clear();
    MACHINE_LOGI("MemPool destroyed, all memory freed");
}

void DevMemoryPool::PrintPoolStatus() const
{
    size_t total = 0;
    size_t used = 0;
    MACHINE_LOGI("========== [Memory Pool Status] ==========");
    for (size_t i = 0; i < memoryBlocks_.size(); ++i) {
        auto* blk = memoryBlocks_[i];
        total += blk->blockSize;
        used += blk->usedSize;

        double rate = blk->blockSize ? (double)blk->usedSize * 100.0 / blk->blockSize : 0;
        MACHINE_LOGI("Block[%lu] | Addr: %p | Used: %.1f%%", i, blk->baseAddr, rate);
    }
    MACHINE_LOGI("Summary: Blocks: %lu | Used/Total: %lu/%lu MB", memoryBlocks_.size(), used >> 20, total >> 20);
}

void DevMemoryPool::FreeMemBlock(MemoryBlock* block)
{
    if (block == nullptr) {
        return;
    }

    if (block->baseAddr != nullptr) {
        MACHINE_LOGI("Releasing physical memory: addr=%p, size=%lu bytes", block->baseAddr, block->blockSize);
        RuntimeFree(block->baseAddr);
        block->baseAddr = nullptr;
    }
    delete block;
    block = nullptr;
}

void DevMemoryPool::RecordAllocation(void* ptr, MemoryBlock* block) { addrToBlock_[ptr] = block; }

MemoryBlock* DevMemoryPool::CreateNewBlock(uint64_t alignSize)
{
    uint8_t* devAddr = nullptr;

    if (RuntimeMalloc((void**)&devAddr, alignSize, TWO_MB_HUGE_PAGE_FLAGS, PYPTO) == RTMALLOC_SUCCESS) {
        MemoryBlock* block = new MemoryBlock(devAddr, alignSize);
        memoryBlocks_.push_back(block);
        return block;
    }

    MACHINE_LOGE(DevCommonErr::ALLOC_FAILED, "All memory alloc strategies failed");
    return nullptr;
}

void* DevAlloc(const uint64_t size)
{
    uint8_t* devPtr = nullptr;
    DevMemoryPool::Instance().AllocDevAddr(&devPtr, size);
    if (devPtr == nullptr) {
        MACHINE_LOGE(RtErr::RT_MALLOC_FAILED, "Failed to alloc dev addr of size[%lu] bytes.", size);
        return nullptr;
    }
    if (RuntimeMemset(devPtr, size, 0, size) != RT_SUCCESS) {
        DevMemoryPool::Instance().FreeDevAddr(devPtr);
        MACHINE_LOGE(RtErr::RT_MEMSET_FAILED, "RuntimeMemset failed size=%lu bytes.", size);
        return nullptr;
    }
    return devPtr;
}

void* CopyDataToDevice(const void* dataPtr, const uint64_t dataSize)
{
    void* devAddr = DevAlloc(dataSize);
    if (devAddr == nullptr) {
        MACHINE_LOGE(DevCommonErr::ALLOC_FAILED, "Failed to alloc dev memory of size %lu bytes", dataSize);
        return nullptr;
    }
    if (RuntimeMemcpyDirect(devAddr, dataSize, dataPtr, dataSize, RtMemcpyKind::HOST_TO_DEVICE) != RT_SUCCESS) {
        DevMemoryPool::Instance().FreeDevAddr(devAddr);
        MACHINE_LOGE(DevCommonErr::ALLOC_FAILED, "Failed to copy data to dev of size %lu bytes", dataSize);
        return nullptr;
    }
    return devAddr;
}
} // namespace npu::tile_fwk
