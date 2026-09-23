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
 * \file dev_encode_program_ctrlflow_cache.h
 * \brief
 */

#pragma once

#include <cinttypes>
#include <type_traits>
#include "machine/utils/dynamic/dev_encode_types.h"
#include "machine/utils/dynamic/dev_encode_function.h"
#include "machine/utils/dynamic/dev_encode_function_dupped_data.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/utils/dynamic/item_pool.h"
#include "machine/utils/dynamic/runtime_outcast_tensor.h"

namespace npu::tile_fwk::dynamic {
#define INVALID_STITCH_IDX (static_cast<uint32_t>(-1))

constexpr size_t READY_QUEUE_SIZE = 3UL;
constexpr size_t DIE_READY_QUEUE_SIZE = 2UL;
inline constexpr size_t MAX_STITCH_FUNC_NUM = 1024;

inline constexpr size_t DEFAULT_STITCH_CFGCACHE_SIZE = 100000000;

struct ReadyQueueCache {
    uint32_t coreFunctionCnt;
    ReadyCoreFunctionQueueUnsafe queueList[READY_QUEUE_SIZE];
    uint32_t readyTaskNum;
    uint32_t* pingElem[READY_QUEUE_SIZE];
    uint32_t* pongElem[READY_QUEUE_SIZE];

    npu::tile_fwk::PerCorePendingQueue* perCorePendingQueueList[npu::tile_fwk::MAX_AICORE_NUM_FOR_QUEUE]{};
};

struct DieReadyQueueCache {
    uint32_t coreFunctionCnt;
    ReadyCoreFunctionQueueUnsafe queueList[DIE_READY_QUEUE_SIZE * DIE_NUM];
    uint32_t readyTaskNum;

    /* Ping-pong element buffers, same protocol as ReadyQueueCache. */
    uint32_t* pingElem[DIE_READY_QUEUE_SIZE * DIE_NUM];
    uint32_t* pongElem[DIE_READY_QUEUE_SIZE * DIE_NUM];
};

struct MixTaskDataCache {
    WrapInfoQueue queue;
    uint64_t wrapIdNum;
    uint64_t opWrapList[MAX_STITCH_FUNC_NUM];
};

struct DynFuncDataCache {
    DevAscendFunction* devFunc;
    predcount_t* predCount;
    predcount_t* predCountPingPong[PRED_COUNT_PINGPONG_NUM];
    int* calleeList;
    DevAscendFunctionDuppedData* duppedData;

    const DynFuncDataCache& At(size_t index) const { return this[index]; }
    DynFuncDataCache& At(size_t index) { return this[index]; }
};

struct DynFuncDataWorkspaceAddressBackup {
    uint64_t runtimeWorkspace;
    uint64_t runtimeOutcastWorkspace;
    uint64_t workspaceAddr;
    uint64_t stackWorkspaceAddr;
};

// POD-like: no NSDMI so DynDeviceTaskBase::dynFuncDataBackupList[MAX_STITCH_FUNC_NUM]
// default-init is a no-op (avoids ~1024x store loop in MakeDynDeviceTask).
// Bitmap/predcount/raw addr fields are filled on ctrlflow cache backup paths.
struct DynFuncDataBackup {
    predcount_t* predCountBackup;
    uint64_t* rawTensorAddrBackup;
    uint64_t* deadEndHubBitmapBackup;
    uint64_t* tailTaskBitmapBackup;
    size_t bitmapByteSize;

    DynFuncDataWorkspaceAddressBackup workspaceAddressBackup;

    const DynFuncDataBackup& At(size_t index) const { return this[index]; }
    DynFuncDataBackup& At(size_t index) { return this[index]; }
};
static_assert(std::is_trivially_default_constructible_v<DynFuncDataBackup>,
              "DynFuncDataBackup must stay trivially default-constructible for MakeDynDeviceTask hot path");

struct ParallelInfo {
    uint32_t parallelism{1};
    uint32_t forId{0};
    uint32_t iterId{0};
    uint32_t wsId{0}; // parallel dev task use different workspace
};

struct DynDeviceTaskBase {
    DeviceTask devTask;
    DynFuncHeader* dynFuncDataList{nullptr};
    npu::tile_fwk::DrcoRootFuncList* drcoRootFuncList{nullptr};
    // DRCO precount 备份：构建期写入，cache 重放时恢复到 rootFuncList（不重算）
    uint32_t drcoPrecountExecuted[npu::tile_fwk::DRCO_QUEUE_MAX]{};
    uint32_t drcoPrecountFinishFlag[npu::tile_fwk::DRCO_QUEUE_MAX]{};

    ReadyCoreFunctionQueue* readyQueue[READY_QUEUE_SIZE];
    DynFuncDataCache dynFuncDataCacheList[MAX_STITCH_FUNC_NUM];
    uint64_t dynFuncDataCacheListSize;

    const DevCceBinary* cceBinary;
    const DevAicpuLeafBinary* aicpuLeafBinary;

    ReadyQueueCache* readyQueueBackup;
    DieReadyQueueCache* dieReadyQueueBackup{nullptr};
    MixTaskDataCache* mixTaskDataBackup{nullptr};
    DynFuncDataBackup dynFuncDataBackupList[MAX_STITCH_FUNC_NUM];
    void* shmemWaitUntilCacheBackup{nullptr};
    bool isLastTask{false};
    bool isParallelSameIterLastTask{false};
    ParallelInfo parallelInfo;

    DynFuncHeader* GetDynFuncDataList() const { return dynFuncDataList; }
    DynFuncHeader* GetDynFuncDataList() { return dynFuncDataList; }
    const DynFuncDataCache* GetDynFuncDataCacheList() const { return dynFuncDataCacheList; }
    DynFuncDataCache* GetDynFuncDataCacheList() { return dynFuncDataCacheList; }

    uint64_t GetIndex() { return GetDynFuncDataList()->GetIndex(); }
    inline bool IsLastTask() const { return isLastTask; }
    void SetLastTask(bool b) { isLastTask = b; }
    uint32_t ParallelForId() { return parallelInfo.forId; }
    uint32_t ParallelIterId() { return parallelInfo.iterId; }
    uint32_t ParallelWsId() { return parallelInfo.wsId; }
    bool SupportParallel() { return parallelInfo.forId != 0; }
    void SetParallelInfo(ParallelInfo info) { parallelInfo = info; }
    bool IsParallelSameIterLastDevTask() { return isParallelSameIterLastTask; }
    void SetParallelSameIterLastDevTask(bool isLast) { isParallelSameIterLastTask = isLast; }
    uint32_t maxC_{0};
    uint32_t maxV_{0};

    uint32_t GetMaxC() const { return maxC_; }
    uint32_t GetMaxV() const { return maxV_; }
    void SetMaxCV(uint32_t maxC, uint32_t maxV)
    {
        maxC_ = maxC;
        maxV_ = maxV;
    }
};

struct DeviceTaskCache {
    DynDeviceTaskBase* dynTaskBase;
};

struct DeviceExecuteSlot {
    ItemPoolIter rtOutcastIter{ITEM_POOL_INVALID_INDEX};
    bool isOutputSlot{false};
    bool isAssembleSlot{false};
    bool isAssembleSlotNeedAlloc{false};
    bool isPartialUpdateStitch{false};
    uint32_t stitchDupIdx{INVALID_STITCH_IDX};
    uint32_t stitchOutcastIdx;
    uint32_t slotAllocIterId{0};
    StitchCtrlBitMask stitchCtrlBitMask{STITCH_CTRL_NONE};

    DevAscendProgramUpdate* partialUpdate{nullptr};

    bool IsOutputAddress() const { return isOutputSlot; }
    bool IsAssembleAddress() const { return isAssembleSlot; }
    void ChangeSlotAllocIterId()
    {
        slotAllocIterId++;
        if (slotAllocIterId == MAX_STITCH_FUNC_NUM - 1) {
            slotAllocIterId = 0;
        }
    }
};

struct DevControlFlowCacheRuntime {
    struct DeviceWorkspaceAllocator {
        struct {
            SeqWsAllocator rootInner;
            SeqWsAllocator devTaskInnerExclusiveOutcasts;
            WsSlotAllocator devTaskBoundaryOutcasts;
            WsSlotAllocator devTaskInnerTemporalOutcasts;
            DevRelocVector<WsSlotAllocator::BlockHeader> slottedOutcastsBlockList;
        } tensorAllocators[SCH_DEVTASK_MAX_PARALLELISM];
        DevRelocVector<ItemPool<RuntimeOutcastTensor>::ItemBlock> runtimeOutcastTensorPool;
        ItemPoolMeta itemPoolMeta;
    } workspace;
    struct DeviceSlotContext {
        DevRelocVector<DeviceExecuteSlot> slotList;
    } slotContext;
};

template <typename T>
inline T* RelocControlFlowCachePointer(T*& ptrRef, const RelocRange& relocProgram)
{
    T* result = nullptr;
    if (relocProgram.GetDst() == 0) {
        result = ptrRef;
        relocProgram.Reloc(ptrRef);
    } else {
        relocProgram.Reloc(ptrRef);
        result = ptrRef;
    }
    return result;
}

struct DevControlFlowCache {
    uint64_t allCacheSize{0};
    /* actual used cache size */
    uint64_t usedCacheSize{0};
    /* Filled by user, true means try to allocate in cache. */
    bool isRecording{false};
    /* Filled by user, true means activate in cache. */
    bool isActivated{false};
    /* reloc meta at device */
    bool isRelocMetaDev{false};
    /* reloc data at device */
    bool isRelocDataDev{false};
    /* cache shape is origin or infer shape */
    bool isCacheOriginShape{true};
    /* Filled in caching */
    DevRelocVector<DevTensorData> inputTensorDataList;
    /* Filled in caching */
    DevRelocVector<DevTensorData> outputTensorDataList;
    /* Filled in caching for runtime */
    DevControlFlowCacheRuntime runtimeBackup;

    /* Filled in caching, true means some metadata is not cached. */
    bool isRecordingStopped;
    /* Filled in caching */
    uint64_t deviceTaskCount;
    /* Filled in caching */
    uint64_t rootTaskCount;
    /* Filled in caching */
    uint64_t cacheDataOffset;
    /* Filled in caching */
    uint64_t deviceTaskSkippedCount;
    /* Filled in caching */
    uint64_t contextWorkspaceAddr;
    /* Filled in caching */
    DevRelocVector<DeviceTaskCache> deviceTaskCacheList;

    bool bitmapRestoreDone{false};
    /* Filled in caching */
    DevRelocVector<uint8_t> cacheData;

    uint64_t workspaceAddr;

    uint64_t taskRelocTableOffset{0};
    uint32_t taskRelocTableCount{0};
#define ctrlFlowLastField cacheData
    uint64_t dataSize;
    uint8_t data[0];

    bool inline IsRecording() const
    {
        if (IsDeviceMode()) {
            return false;
        }
        return isRecording;
    }

    bool inline IsRecordingStopped() const { return isRecordingStopped; }

    bool inline IsCacheOriginShape() const { return isCacheOriginShape; }

    void inline StopRecording() { isRecordingStopped = true; }

    void inline CalcUsedCacheSize()
    {
        usedCacheSize = reinterpret_cast<uintptr_t>(cacheData.Data()) + cacheDataOffset -
                        reinterpret_cast<uintptr_t>(this);
    }

    // slottedOutcastBlockCount: live devTask boundary-outcast block headers backed up for ctrl-flow cache.
    void Init(void* dyndevAttrPtr, uint64_t cacheSize, uint64_t runtimeOutcastPoolSize, uint64_t& initOffset,
              uint64_t slottedOutcastBlockCount);
    uint64_t GetSize() const
    {
        return reinterpret_cast<uintptr_t>(ctrlFlowLastField.End()) - reinterpret_cast<uintptr_t>(this);
    }

#define CFGCACHE_ALIGN 64
    void* AllocateCache(uint64_t size)
    {
        void* result = nullptr;
        if (cacheDataOffset + size < cacheData.size()) {
            result = &cacheData[cacheDataOffset];
            /* make cache 64 byte aligned */
            cacheDataOffset += (size + CFGCACHE_ALIGN - 1) / CFGCACHE_ALIGN * CFGCACHE_ALIGN;
            DEV_VERBOSE_DEBUG("cacheDataOffset is: %lu", cacheDataOffset);
        } else {
            DEV_DEBUG("[ctrl.cache.record] Recording is stopped, requestSize=%lu, "
                      "cacheDataOffset=%lu, cacheDataSize=%lu.",
                      size, cacheDataOffset, cacheData.size());

            isRecordingStopped = true;
        }
        return result;
    }

    bool AppendDeviceTask(DynDeviceTaskBase* base)
    {
        if (!isRecordingStopped && (deviceTaskCount < deviceTaskCacheList.size())) {
            deviceTaskCacheList[deviceTaskCount].dynTaskBase = base;
            deviceTaskCount += 1;
            rootTaskCount += base->dynFuncDataList->Size();
            DEV_DEBUG("deviceTaskCount=%lu", deviceTaskCount);
            return true;
        } else {
            deviceTaskSkippedCount += 1;
            return false;
        }
    }

    void InitInputOutput(DevStartArgsBase* startArgs)
    {
        for (size_t i = 0; i < inputTensorDataList.size(); i++) {
            inputTensorDataList[i] = startArgs->GetInputTensor(i);
        }
        for (size_t i = 0; i < outputTensorDataList.size(); i++) {
            outputTensorDataList[i] = startArgs->GetOutputTensor(i);
        }
    }

    void MatchInputOutputDump(DevStartArgsBase* startArgs) const;

    inline bool MatchInputOutput(DevStartArgsBase* startArgs) const
    {
        MatchInputOutputDump(startArgs);

        if (inputTensorDataList.size() != startArgs->inputTensorSize) {
            return false;
        }
        if (outputTensorDataList.size() != startArgs->outputTensorSize) {
            return false;
        }
        // support infer controlflow cache now, cached shape and realshape may not match now
        return true;
    }

    inline bool IsActivatedFullCache(DevStartArgsBase* startArgs) const
    {
        if (!isActivated) {
            return false;
        }
        if (deviceTaskSkippedCount != 0) {
            return false;
        }
        if (isRecordingStopped != 0) {
            return false;
        }
        if (!MatchInputOutput(startArgs)) {
            return false;
        }
        return true;
    }

    inline bool IsActivatedCache(DevStartArgsBase* startArgs) const
    {
        if (!isActivated) {
            return false;
        }
        if (deviceTaskCount == 0) {
            return false;
        }
        if (!MatchInputOutput(startArgs)) {
            return false;
        }
        return true;
    }

    void BitmapDataBackup(DynFuncDataCache* dynDataCache, DynFuncDataBackup* dynDataBackup)
    {
        auto* devFunc = dynDataCache->devFunc;
        size_t bitmapBytes = devFunc->GetBitmapByteSize();
        if (bitmapBytes == 0) {
            dynDataBackup->deadEndHubBitmapBackup = nullptr;
            dynDataBackup->tailTaskBitmapBackup = nullptr;
            dynDataBackup->bitmapByteSize = 0;
            return;
        }

        uint64_t* deadEndBuf = reinterpret_cast<uint64_t*>(AllocateCache(bitmapBytes));
        uint64_t* tailBuf = reinterpret_cast<uint64_t*>(AllocateCache(bitmapBytes));
        if (deadEndBuf == nullptr || tailBuf == nullptr) {
            dynDataBackup->deadEndHubBitmapBackup = nullptr;
            dynDataBackup->tailTaskBitmapBackup = nullptr;
            dynDataBackup->bitmapByteSize = 0;
            return;
        }

        dynDataBackup->deadEndHubBitmapBackup = deadEndBuf;
        dynDataBackup->tailTaskBitmapBackup = tailBuf;
        dynDataBackup->bitmapByteSize = bitmapBytes;
        devFunc->BackupBitmapTo(deadEndBuf, tailBuf, bitmapBytes);
    }

    void BitmapDataRestore(DynFuncDataCache* dynDataCache, DynFuncDataBackup* dynDataBackup)
    {
        if (dynDataBackup->bitmapByteSize == 0) {
            return;
        }
        dynDataCache->devFunc->RestoreBitmapFrom(dynDataBackup->deadEndHubBitmapBackup,
                                                 dynDataBackup->tailTaskBitmapBackup, dynDataBackup->bitmapByteSize);
    }

    void PredCountDataBackup(DynDeviceTaskBase* base);

    void PredCountPingPongSwap(DynDeviceTaskBase* base);

    void DrcoPredCountDataRestore(DynDeviceTaskBase* base);

    void PredCountPingPongRestore(DynDeviceTaskBase* base);

    void BitmapDataRestoreTask(DynDeviceTaskBase* base);

    void ReadyQueueDataBackup(DynDeviceTaskBase* base);

    void ReadyQueueDataPingPongSwap(DynDeviceTaskBase* base, uint32_t nrValidAic);

    void ReadyQueueDataPingPongRestore(DynDeviceTaskBase* base);

    void DrcoReadyQueueDataRestore(DynDeviceTaskBase* base, uint32_t nrValidAic);

    void DieReadyQueueDataBackup(DynDeviceTaskBase* base);

    void DieReadyQueuePingPongSwap(DynDeviceTaskBase* base, uint32_t nrValidAic);

    void DieReadyQueuePingPongRestore(DynDeviceTaskBase* base);

    void MixTaskDataBackup(DynDeviceTaskBase* base);

    void MixTaskDataRestore(DynDeviceTaskBase* base);

    static void RelocBuildInputOutputDesc(std::unordered_map<uint64_t, AddressDescriptor>& cacheInputOutputDict,
                                          DevStartArgsBase* devStartArgs);

    static void RelocBuildInputOutputDesc(std::unordered_map<uint64_t, AddressDescriptor>& cacheInputOutputDict,
                                          DevRelocVector<DevTensorData> inputTensorDataList,
                                          DevRelocVector<DevTensorData> outputTensorDataList);

    static void RelocDescToCache(AddressDescriptor& desc, const RelocRange& relocWorkspace,
                                 std::unordered_map<uint64_t, AddressDescriptor>& cacheInputOutputDict);

    static void RelocDescFromCache(const AddressDescriptor& desc, AddressDescriptor& result,
                                   const RelocRange& relocWorkspace, DevStartArgsBase* devStartArgs);

    struct IncastOutcastRelocBatch {
        uint64_t srcBaseOffset; /* rawTensorAddrBackup (cache-form descriptors, read-only) */
        uint64_t dstBaseOffset; /* rawTensorAddr (live addresses, rewritten per launch) */
        uint64_t idxListOffset; /* uint16 descriptor indices covered by this batch */
        uint32_t descCount;     /* number of descriptors in this batch */
        AddressCacheKind kind;
    };

    struct TaskRelocEntry {
        uint64_t batchListOffset;
        uint64_t dynFuncHeaderOffset;
        uint32_t batchCount;
    };

    void BuildIncastOutcastRelocTable();
    void IncastOutcastAddrBackup(DynDeviceTaskBase* base);

    void TaskAddrBackupWorkspace(DynDeviceTaskBase* base);

    void TaskAddrRestoreWorkspace(DynDeviceTaskBase* base);

    void TaskAddrRestoreWorkspace();

    void TaskAddrRelocWorkspace(uint64_t srcWorkspace, uint64_t dstWorkspace, DevStartArgsBase* devStartArgs);

    void IncastOutcastAddrReloc(uint64_t srcWorkspace, uint64_t dstWorkspace, DevStartArgsBase* devStartArgs);

    void RelocIncastOutcastTask(uint32_t taskIndex, uint64_t srcWorkspace, uint64_t dstWorkspace,
                                DevStartArgsBase* devStartArgs);

    void RelocIncastOutcastTaskStructural(uint32_t taskIndex, uint64_t srcWorkspace, uint64_t dstWorkspace,
                                          DevStartArgsBase* devStartArgs);

    void RuntimeAddrBackup(DeviceExecuteSlot* runtimeSlotList, ItemPool<RuntimeOutcastTensor>* runtimeOutcastTensorPool,
                           uint64_t slotSize, uint64_t runtimeOutcastTensorSize, TensorAllocator* allocator,
                           uint32_t parallelism);

    void RuntimeAddrRestore(DeviceExecuteSlot* runtimeSlotList,
                            ItemPool<RuntimeOutcastTensor>* runtimeOutcastTensorPool, uint64_t slotSize,
                            uint64_t runtimeOutcastTensorSize, TensorAllocator* allocator, uint32_t parallelism);

    void RuntimeAddrRelocProgram(uint64_t srcProgram, uint64_t dstProgram);

    void RuntimeAddrRelocWorkspace(uint64_t srcWorkspace, uint64_t dstWorkspace, DevStartArgsBase* devStartArgs,
                                   DeviceExecuteSlot* runtimeSlotList,
                                   ItemPool<RuntimeOutcastTensor>::ItemBlock* runtimeOutcastTensorPool,
                                   uint32_t parallelism, TensorAllocator* allocator = nullptr);

    void MixTaskDataReloc(RelocRange& relocCtrlCache, RelocRange& relocProgram, DynDeviceTaskBase* dynTaskBase,
                          DynFuncHeader* dynFuncDataList);

    void DieReadyQueueReloc(RelocRange& relocCtrlCache, DynDeviceTaskBase* dynTaskBase);
    void RelocDrcoRootFuncList(RelocRange& relocCtrlCache, DynDeviceTaskBase* dynTaskBase);
    void RelocDuppedDataAndDynFuncData(RelocRange& relocProgram, RelocRange& relocCtrlCache,
                                       DevAscendFunctionDuppedData* duppedData, DynFuncData* dynData,
                                       DynFuncDataCache* dynDataCache, DynFuncDataBackup* dynDataBackup);

    /* Host-to-cache: devStartArgs should be nullptr. Cache-to-Device: devStartArgs should be filled */
    void TaskAddrRelocProgramAndCtrlCache(uint64_t srcProgram, uint64_t srcCtrlCache, uint64_t dstProgram,
                                          uint64_t dstCtrlCache);

    template <typename Ty>
    typename Ty::ElementType* RelocOffset(intptr_t shift, void*& offset, Ty& list)
    {
        typename Ty::ElementType* ptr = reinterpret_cast<typename Ty::ElementType*>(offset);
        offset = (void*)((uintptr_t)(offset) + list.ElementSize() * list.size());
        list.DeviceRelocData(shift);
        return ptr;
    }

    void RelocMetaCache(uint64_t srcCache, uint64_t dstCache);
};

#define ControlFlowAllocateSlab(devProg, size, expr)               \
    ({                                                             \
        WsAllocation ws;                                           \
        DevControlFlowCache* c = (devProg)->GetControlFlowCache(); \
        if (c->IsRecording()) {                                    \
            void* ptr = c->AllocateCache(size);                    \
            if (ptr != nullptr) {                                  \
                ws.ptr = reinterpret_cast<uintdevptr_t>(ptr);      \
            } else {                                               \
                ws = (expr);                                       \
            }                                                      \
        } else {                                                   \
            ws = (expr);                                           \
        }                                                          \
        ws;                                                        \
    })
} // namespace npu::tile_fwk::dynamic
