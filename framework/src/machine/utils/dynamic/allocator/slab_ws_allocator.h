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
 * \file seq_ws_allocator.h
 * \brief
 */

#pragma once
#include <cstddef>
#include <cstdint>
#include <algorithm>

#include "machine/utils/device_log.h"
#include "machine/device/dynamic/device_utils.h"
#include "ws_allocator_basics.h"

namespace npu::tile_fwk::dynamic {
enum class WsAicpuSlabMemType : uint8_t {
    DUPPED_FUNC_DATA = 0,
    DYN_FUNC_DATA,
    VEC_STITCHED_LIST,
    DEV_DYN_TASK,
    SHMEM_WAIT_UNTIL_CACHE,
    COHERENT_SLAB_MEM_TYPE_BUTT, // add new slabmemtype should be above this type
    READY_QUE,
    DIE_READY_QUE,
    WRAP_QUEUE,
    PER_CORE_PENDING_QUE,
    LOCAL_READY_QUE,
    PRED_COUNT,
    DUPPED_STITCH, // stitch pool memory
    SLAB_MEM_TYPE_BUTT
};
constexpr int SLAB_ALLOCATOR_MAX_CACHES = 16;
constexpr uint32_t SLAB_ALIGN_GRANULARITY = 64;

constexpr uint64_t SLAB_OBJ_MAGIC = 0xDEADBEEFCAFEBABEULL;
constexpr uint32_t SLAB_MAGIC_SIZE = sizeof(uint64_t);
constexpr uint32_t SLAB_OBJ_META_SIZE = sizeof(void*) + SLAB_MAGIC_SIZE;

template <typename T>
constexpr T AlignUpSlab(T n)
{
    return (n + static_cast<T>(SLAB_ALIGN_GRANULARITY) - 1) & ~(static_cast<T>(SLAB_ALIGN_GRANULARITY) - 1);
}

struct StageAllocInfo {
    void* heads[SLAB_ALLOCATOR_MAX_CACHES];
    void* tails[SLAB_ALLOCATOR_MAX_CACHES];
    uint32_t objCnt[SLAB_ALLOCATOR_MAX_CACHES];
    uint32_t objSizes[SLAB_ALLOCATOR_MAX_CACHES];
    uint32_t slabAlignSize{0};
    uint8_t* slabBase{nullptr};
};

class SlabWsAllocator {
private:
    struct SlabHeader;
    struct SlabCache {
        uint32_t objSize{0};
        uint32_t slotStride{0};
        void* freeList{nullptr};
        void* freeListTail{nullptr};
        SlabHeader* activeSlab{nullptr};
        SlabWsAllocator* allocator{nullptr};

        // Per-cache allocation tracking
        void* stageAllocHead{nullptr};
        void* stageAllocTail{nullptr};

        // Cache-level statistics
        uint32_t slabCount{0};
        uint32_t totalObjCount{0};
        uint32_t allocatedObjCount{0};
        uint32_t unPopAllocatedObjCount{0};

        SlabCache() {}
        SlabCache(uint32_t size, uint32_t stride, SlabWsAllocator* alloc)
            : objSize(size),
              slotStride(stride),
              freeList(nullptr),
              activeSlab(nullptr),
              allocator(alloc),
              stageAllocHead(nullptr),
              stageAllocTail(nullptr),
              slabCount(0),
              totalObjCount(0),
              allocatedObjCount(0),
              unPopAllocatedObjCount(0)
        {}
    };

    struct SlabHeader {
        SlabCache* cache; // extend
        uint32_t allocatedCount;
        uint32_t totalCount;
        uint64_t magic;
    };

public:
    SlabWsAllocator() = default;

    void Init(void* baseAddr, uint32_t totalSize, uint32_t alignSize, ArchInfo arch = ArchInfo::DAV_2201)
    {
        uintptr_t rawAddr = reinterpret_cast<uintptr_t>(baseAddr);
        uintptr_t alignedAddr = AlignUpSlab(rawAddr);
        memBaseaddr_ = reinterpret_cast<uint8_t*>(alignedAddr);
        uint32_t padding = static_cast<uint32_t>(alignedAddr - rawAddr);
        totalMemSize_ = (totalSize > padding) ? (totalSize - padding) : 0;
        slabAlignSize_ = AlignUpSlab(alignSize);
        nextFreeSlabAddr_ = memBaseaddr_;
        freeSlabList_ = nullptr;
        numCaches_ = 0;
        archInfo_ = arch;

        totalSlabCount_ = totalMemSize_ / slabAlignSize_;
        allocatedSlabCount_ = 0;
        DEV_DEBUG("[SlabWsAllocator]Init SlabWsAllocator: base=%p, size=%u, align=%u.\n", memBaseaddr_, totalMemSize_,
                  slabAlignSize_);
    }

    // 对齐后的池基址：池内对象相对此地址的偏移恒 < 4GB（totalMemSize_ 为 u32），
    // 供 u32 偏移方案（如 DrcoGlobalStitchNodeMatrix 槽位）换算节点地址
    uint64_t GetMemBase() const { return reinterpret_cast<uint64_t>(memBaseaddr_); }

    bool RegistCache(uint32_t type, uint32_t objSize)
    {
        if (type >= SLAB_ALLOCATOR_MAX_CACHES || objSize == 0) {
            return false;
        }

        if (caches_[type].objSize != 0) {
            if (caches_[type].objSize >= objSize) {
                DEV_DEBUG("[SlabWsAllocator]Slab cache exists: objsize=%u, cacheType=%u.\n", objSize, type);
                return true;
            }
            DEV_ERROR(WsErr::SLAB_ADD_CACHE_FAILED,
                      "workspace.slab.add_cache: [SlabWsAllocator]Add cache failed: type=%u, objsize=%u", type,
                      objSize);
            return false;
        }
        uint32_t slotStride = CalcSlotStride(objSize);
        caches_[type] = SlabCache(objSize, slotStride, this);
        numCaches_++;
        DEV_DEBUG("[SlabWsAllocator]Add slab cache: objsize=%u, slotStride=%u, type=%u.\n", objSize, slotStride, type);
        return true;
    }

    bool ExistCache(uint32_t cacheType, uint32_t objSize)
    {
        if (cacheType >= SLAB_ALLOCATOR_MAX_CACHES) {
            return false;
        }
        if (caches_[cacheType].objSize >= objSize) {
            return true;
        }
        return false;
    }

    static constexpr uint32_t CalcSlotStride(uint32_t objSize)
    {
        return AlignUpSlab(SLAB_OBJ_META_SIZE + objSize + SLAB_MAGIC_SIZE);
    }

    static constexpr uint32_t FirstObjBaseOffset()
    {
        return AlignUpSlab(sizeof(SlabHeader) + SLAB_OBJ_META_SIZE) - SLAB_OBJ_META_SIZE;
    }

    static uint32_t CalcObjsPerSlab(uint32_t slabAlignSize, uint32_t objSize)
    {
        if (objSize == 0 || slabAlignSize == 0) {
            return 0;
        }
        uint32_t firstObjBase = FirstObjBaseOffset();
        uint32_t slotStride = CalcSlotStride(objSize);
        if (slabAlignSize <= firstObjBase) {
            return 0;
        }
        return (slabAlignSize - firstObjBase) / slotStride;
    }

    void AfterAllocSuccess(SlabCache& cache, void* obj, uint32_t objSize)
    {
        *static_cast<void**>(obj) = nullptr;
        *reinterpret_cast<uint64_t*>(static_cast<uint8_t*>(obj) + sizeof(void*)) = SLAB_OBJ_MAGIC;
        *reinterpret_cast<uint64_t*>(static_cast<uint8_t*>(obj) + SLAB_OBJ_META_SIZE + objSize) = SLAB_OBJ_MAGIC;
        if (!cache.stageAllocHead) {
            cache.stageAllocHead = obj;
        } else {
            *static_cast<void**>(cache.stageAllocTail) = obj;
        }
        cache.stageAllocTail = obj;
        cache.allocatedObjCount++;
        cache.unPopAllocatedObjCount++;
        totalAllocNum_++;
        totalMemReq_ += objSize;
        return;
    }

    void* Alloc(uint32_t cacheType)
    {
        if (cacheType >= SLAB_ALLOCATOR_MAX_CACHES) {
            return nullptr;
        }

        SlabCache& cache = caches_[cacheType];
        uint32_t objSize = cache.objSize;
        if (objSize == 0) {
            return nullptr;
        }

        void* obj = nullptr;
        if (cache.freeList) {
            obj = cache.freeList;
            cache.freeList = *static_cast<void**>(obj);
            if (cache.freeList == nullptr) {
                cache.freeListTail = nullptr;
            }
            DEV_VERBOSE_DEBUG("[SlabWsAllocator]Alloc from slab free list: objsize = %u .\n", objSize);
        } else if (cache.activeSlab && cache.activeSlab->allocatedCount < cache.activeSlab->totalCount) {
            SlabHeader* slab = cache.activeSlab;
            obj = static_cast<uint8_t*>(static_cast<void*>(slab)) + FirstObjBaseOffset() +
                  slab->allocatedCount * cache.slotStride;
            slab->allocatedCount++;
            DEV_VERBOSE_DEBUG("[SlabWsAllocator]Alloc from active slab: slab = %p, objsize = %u, allocCnt=%u .\n", slab,
                              objSize, slab->allocatedCount);
        } else {
            void* slabMem = get_free_slab();
            if (!slabMem) {
                DEV_IF_DEBUG { DumpMemoryStatusWhenAbnormal("alloc null:"); }

                DEV_DEBUG("[SlabWsAllocator]Alloc memory cacheType=%u not enough: objsize=%u.\n", cacheType, objSize);
                return nullptr; // memory not enough
            }

            SlabHeader* header = new (slabMem) SlabHeader();
            header->cache = &cache;
            header->allocatedCount = 1;
            header->totalCount = CalcObjsPerSlab(slabAlignSize_, objSize);
            header->magic = SLAB_OBJ_MAGIC;
            cache.activeSlab = header;
            obj = static_cast<uint8_t*>(slabMem) + FirstObjBaseOffset();

            // Update statistics for new slab
            cache.slabCount++;
            cache.totalObjCount += header->totalCount;
            allocatedSlabCount_++;

            DEV_VERBOSE_DEBUG("[SlabWsAllocator]Alloc from new slab: slab = %p, objsize = %u, totalCnt=%u .\n", header,
                              objSize, header->totalCount);
        }

        AfterAllocSuccess(cache, obj, objSize);
        DEV_VERBOSE_DEBUG("[SlabWsAllocator]Alloc success obj = %p cacheType = %u size = %u .\n", obj, cacheType,
                          objSize);
        return static_cast<uint8_t*>(obj) + SLAB_OBJ_META_SIZE;
    }

    void ProcessKeepTailCase(uint32_t cacheIdx, StageAllocInfo& info)
    {
        if (caches_[cacheIdx].stageAllocHead == caches_[cacheIdx].stageAllocTail) {
            info.heads[cacheIdx] = nullptr;
            info.tails[cacheIdx] = nullptr;
            info.objCnt[cacheIdx] = 0;
            return;
        }

        void* temp = caches_[cacheIdx].stageAllocHead;
        if (temp == nullptr) {
            DEV_ERROR(WsErr::SLAB_STAGE_LIST_INCONSISTENT,
                      "workspace.slab.stage: stageAllocHead is null for cacheIndex=%u\n", cacheIdx);
        }
        DEV_ASSERT(WsErr::SLAB_STAGE_LIST_INCONSISTENT, temp != nullptr);

        TIMEOUT_CHECK_INIT_WARN_ONLY(archInfo_);
        while (*static_cast<void**>(temp) != caches_[cacheIdx].stageAllocTail) {
            __PYPTO_TIMEOUT_CHECK_WARN_ONLY("#workspace.slab.stage: Stage alloc traversal for cacheIndex=%u.",
                                            cacheIdx);
            temp = *static_cast<void**>(temp);
        }

        if (temp == nullptr) {
            DEV_ERROR(WsErr::SLAB_STAGE_LIST_INCONSISTENT,
                      "workspace.slab.stage: stageAllocHead is null after loop for cacheIndex=%u, "
                      "stageAllocTail=%p\n",
                      cacheIdx, caches_[cacheIdx].stageAllocTail);
        }
        DEV_ASSERT(WsErr::SLAB_STAGE_LIST_INCONSISTENT, temp != nullptr);

        *static_cast<void**>(temp) = nullptr;
        info.tails[cacheIdx] = temp;
        DEV_VERBOSE_DEBUG("Keep tail not pop %p \n", caches_[cacheIdx].stageAllocTail);
        caches_[cacheIdx].stageAllocHead = caches_[cacheIdx].stageAllocTail;
        info.objCnt[cacheIdx] = caches_[cacheIdx].allocatedObjCount - 1;
        caches_[cacheIdx].unPopAllocatedObjCount = 1;
    }

    StageAllocInfo PopStageAllocMem(bool keepTail, uint32_t memType)
    {
        StageAllocInfo info;
        info.slabAlignSize = slabAlignSize_;
        info.slabBase = memBaseaddr_;
        for (uint32_t i = 0; i < SLAB_ALLOCATOR_MAX_CACHES; i++) {
            info.heads[i] = caches_[i].stageAllocHead;
            info.tails[i] = caches_[i].stageAllocTail;
            info.objSizes[i] = caches_[i].objSize;
            if (!keepTail || i != memType) {
                caches_[i].stageAllocHead = nullptr;
                caches_[i].stageAllocTail = nullptr;
                info.objCnt[i] = caches_[i].allocatedObjCount;
                caches_[i].unPopAllocatedObjCount = 0;
                continue;
            }
            ProcessKeepTailCase(i, info);
        }

        return info;
    }

    void FreeStageAllocMem(const StageAllocInfo& info)
    {
        for (int i = 0; i < SLAB_ALLOCATOR_MAX_CACHES; i++) {
            SlabCache& cache = caches_[i];
            cache.allocatedObjCount -= info.objCnt[i];

            DEV_IF_VERBOSE_DEBUG
            {
                void* temp = info.heads[i];
                while (temp) {
                    DEV_VERBOSE_DEBUG("[SlabWsAllocator]recycle success obj = %p cacheType = %d size = %u .\n", temp, i,
                                      cache.objSize);
                    temp = *static_cast<void**>(temp);
                }
            }

            /* The recently released ones are placed at the back
               to ensure that newly allocated addresses are from the historical pool. */
            if (cache.freeListTail) {
                *static_cast<void**>(cache.freeListTail) = info.heads[i];
            } else {
                cache.freeList = info.heads[i];
            }
            if (info.tails[i]) {
                cache.freeListTail = info.tails[i];
            }
        }
    }

    static void VerifyStageAllocMem(const StageAllocInfo& info, const char* tag)
    {
        for (uint32_t i = 0; i < SLAB_ALLOCATOR_MAX_CACHES; i++) {
            void* curr = info.heads[i];
            uint32_t traversed = 0;
            while (curr) {
                auto* base = static_cast<uint8_t*>(curr);
                uint64_t headMagic = *reinterpret_cast<uint64_t*>(base + sizeof(void*));
                DEV_ASSERT_MSG(WsErr::SLAB_MAGIC_CORRUPTED, headMagic == SLAB_OBJ_MAGIC,
                               "tag=%s, cache=%u, obj=%p, headMagic expected=0x%llX, actual=0x%llX", tag, i, curr,
                               (unsigned long long)SLAB_OBJ_MAGIC, (unsigned long long)headMagic);
                uint64_t tailMagic = *reinterpret_cast<uint64_t*>(base + SLAB_OBJ_META_SIZE + info.objSizes[i]);
                DEV_ASSERT_MSG(WsErr::SLAB_MAGIC_CORRUPTED, tailMagic == SLAB_OBJ_MAGIC,
                               "tag=%s, cache=%u, obj=%p, tailMagic expected=0x%llX, actual=0x%llX", tag, i, curr,
                               (unsigned long long)SLAB_OBJ_MAGIC, (unsigned long long)tailMagic);
                uintptr_t objAddr = reinterpret_cast<uintptr_t>(curr);
                uintptr_t baseAddr = reinterpret_cast<uintptr_t>(info.slabBase);
                uintptr_t slabAddr = baseAddr + ((objAddr - baseAddr) / info.slabAlignSize) * info.slabAlignSize;
                const SlabHeader* slab = reinterpret_cast<const SlabHeader*>(slabAddr);
                DEV_ASSERT_MSG(WsErr::SLAB_MAGIC_CORRUPTED, slab->magic == SLAB_OBJ_MAGIC,
                               "tag=%s, cache=%u, slab=%p, slabMagic expected=0x%llX, actual=0x%llX", tag, i, slab,
                               (unsigned long long)SLAB_OBJ_MAGIC, (unsigned long long)slab->magic);
                curr = *static_cast<void**>(curr);
                traversed++;
                DEV_ASSERT_MSG(WsErr::SLAB_STAGE_LIST_INCONSISTENT, traversed <= info.objCnt[i],
                               "tag=%s, cache=%u, traversed=%u > objCnt=%u, list may be corrupted", tag, i, traversed,
                               info.objCnt[i]);
            }
        }
    }

    /* ==================== DFX Statistics Methods ==================== */
    // Get allocator-level statistics
    struct AllocatorStats {
        uint32_t totalSlabCount;
        uint32_t allocatedSlabCount;
        uint32_t freeSlabCount;
        uint32_t slabSize;
        double usage;
    };

    AllocatorStats GetAllocatorStats() const
    {
        AllocatorStats stats;
        stats.totalSlabCount = totalSlabCount_;
        stats.allocatedSlabCount = allocatedSlabCount_;
        stats.freeSlabCount = totalSlabCount_ - allocatedSlabCount_;
        stats.slabSize = slabAlignSize_;
        stats.usage = totalSlabCount_ > 0 ? (static_cast<double>(stats.allocatedSlabCount) / stats.totalSlabCount) :
                                            0.0;
        return stats;
    }

    // Get cache-level statistics
    struct CacheStats {
        uint32_t objSize;
        uint32_t slabCount;
        uint32_t totalObjCount;
        uint32_t allocatedObjCount;
        uint32_t freeObjCount;
        double usage;
    };

    CacheStats GetCacheStats(uint32_t cacheType) const
    {
        if (cacheType >= SLAB_ALLOCATOR_MAX_CACHES) {
            return CacheStats{};
        }

        const SlabCache& cache = caches_[cacheType];
        CacheStats stats;
        stats.objSize = cache.objSize;
        stats.slabCount = cache.slabCount;
        stats.totalObjCount = cache.totalObjCount;
        stats.allocatedObjCount = cache.allocatedObjCount;
        stats.freeObjCount = cache.totalObjCount - cache.allocatedObjCount;
        stats.usage = cache.totalObjCount > 0 ? (static_cast<double>(cache.allocatedObjCount) / cache.totalObjCount) :
                                                0.0;
        return stats;
    }

    // Get statistics for all caches
    void GetAllCacheStats(CacheStats stats[SLAB_ALLOCATOR_MAX_CACHES]) const
    {
        for (int i = 0; i < SLAB_ALLOCATOR_MAX_CACHES; i++) {
            stats[i] = GetCacheStats(i);
        }
    }
    void DumpSlabUsage(const char* title) const
    {
        int percent = 100;
        AllocatorStats allocStats = GetAllocatorStats();
        uint64_t usedBytes = static_cast<uint64_t>(allocStats.allocatedSlabCount) * allocStats.slabSize;
        uint64_t freeBytes = static_cast<uint64_t>(allocStats.freeSlabCount) * allocStats.slabSize;
        DEV_DEBUG("[workspaceSize.slab.usage] %s totalBytes=%u, usedBytes=%lu, freeBytes=%lu, totalSlabs=%u, "
                  "allocatedSlabs=%u, freeSlabs=%u, slabSize=%u, usage=%.2f%%.",
                  title, totalMemSize_, usedBytes, freeBytes, allocStats.totalSlabCount, allocStats.allocatedSlabCount,
                  allocStats.freeSlabCount, allocStats.slabSize, allocStats.usage * percent);
    }
    void DumpMemoryStatusWhenAbnormal(const char* title) const
    {
        DEV_WARN("[SlabWsAllocator]%s\n", title);
        int percent = 100;

        // Dump allocator-level statistics
        DumpSlabUsage(title);
        // Dump cache-level statistics
        for (int i = 0; i < SLAB_ALLOCATOR_MAX_CACHES; i++) {
            if (caches_[i].objSize == 0)
                continue;

            CacheStats cacheStats = GetCacheStats(i);
            DEV_DEBUG("Slab cache[%d]: ObjSize=%u, AlloCatedSlabs=%u, TotalObjs=%u,"
                      "AllocatedObjs=%u, FreeObjs=%u, Usage=%.2f%%\n",
                      i, cacheStats.objSize, cacheStats.slabCount, cacheStats.totalObjCount,
                      cacheStats.allocatedObjCount, cacheStats.freeObjCount, cacheStats.usage * percent);
        }
    }

    void DumpMemoryUsage(const char* hint, const char* title) const
    {
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT
        DEV_MEM_DUMP("%s memory usage (%s)\n", title, hint);
        DEV_MEM_DUMP("            Memory pool size: %10u bytes\n", totalMemSize_);
        DEV_MEM_DUMP("    Total memory requirement: %10u bytes\n", totalMemReq_);
        DEV_MEM_DUMP("      Total allocation count: %10u\n", totalAllocNum_);
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT
        (void)title;
        (void)hint;
    }

private:
    void* get_free_slab()
    {
        if (freeSlabList_) {
            void* slab = freeSlabList_;
            freeSlabList_ = *static_cast<void**>(freeSlabList_);
            return slab;
        }

        if (nextFreeSlabAddr_ + slabAlignSize_ <= memBaseaddr_ + totalMemSize_) {
            void* slab = nextFreeSlabAddr_;
            nextFreeSlabAddr_ += slabAlignSize_;
            return slab;
        }

        return nullptr;
    }

private:
    uint8_t* memBaseaddr_{nullptr};
    uint8_t* nextFreeSlabAddr_{nullptr};
    uint32_t totalMemSize_{0};
    uint32_t slabAlignSize_{0};
    void* freeSlabList_{nullptr};
    ArchInfo archInfo_{ArchInfo::DAV_2201};

    SlabCache caches_[SLAB_ALLOCATOR_MAX_CACHES];
    int numCaches_{0};

    // Allocator-level statistics
    uint32_t totalSlabCount_{0};
    uint32_t allocatedSlabCount_{0};
    uint32_t totalMemReq_{0};
    uint32_t totalAllocNum_{0};
};

struct WsSlabStageAllocMem {
    std::atomic_bool canFree{false};
    std::atomic_bool* ctrlNotFreeFlag{nullptr};
    StageAllocInfo generalMetadataStageMem;
    StageAllocInfo stitchStageMem;

    WsSlabStageAllocMem() = default;
    WsSlabStageAllocMem(const WsSlabStageAllocMem& other)
        : canFree(other.canFree.load(std::memory_order_relaxed)),
          ctrlNotFreeFlag(other.ctrlNotFreeFlag),
          generalMetadataStageMem(other.generalMetadataStageMem),
          stitchStageMem(other.stitchStageMem)
    {}

    WsSlabStageAllocMem& operator=(const WsSlabStageAllocMem& other)
    {
        if (this != &other) {
            canFree.store(other.canFree.load(std::memory_order_relaxed), std::memory_order_relaxed);
            ctrlNotFreeFlag = other.ctrlNotFreeFlag;
            generalMetadataStageMem = other.generalMetadataStageMem;
            stitchStageMem = other.stitchStageMem;
        }
        return *this;
    }

    void VerifyAll() const
    {
#if SLAB_OBJ_MAGIC_VERIFY
        SlabWsAllocator::VerifyStageAllocMem(generalMetadataStageMem, "general");
        SlabWsAllocator::VerifyStageAllocMem(stitchStageMem, "stitch");
#endif
    }
};
} // namespace npu::tile_fwk::dynamic
