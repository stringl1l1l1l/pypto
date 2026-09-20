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
 * \file dev_workspace.h
 * \brief
 */

#ifndef DEV_WORKSPACE_H
#define DEV_WORKSPACE_H

#include "dev_start_args.h"
#include "device_task.h"
#include "item_pool.h"
#include "spsc_queue.h"
#include "../machine_ws_intf.h"
#include "allocator/allocators.h"
#include "machine/device/dynamic/device_perf.h"
#include "machine/utils/dynamic/runtime_outcast_tensor.h"
#include "machine/device/distributed/shmem_wait_until.h"

namespace npu::tile_fwk::dynamic {
inline constexpr int64_t TENSOR_ADDR_ALIGNMENT = 512;
inline constexpr uint32_t SUBMMIT_TASK_QUE_SIZE = 512;
constexpr int32_t ALLOC_NUM_ONE_SLAB = 4;

enum class FunctionMemoryAllocFailure : uint8_t {
    OK = 0,
    EXCLUSIVE_OUTCAST,
    INNER_TEMPORAL_OUTCAST,
    RUNTIME_OUTCAST_METADATA,
};

inline constexpr const char* FunctionMemoryAllocFailureName(FunctionMemoryAllocFailure failure)
{
    switch (failure) {
        case FunctionMemoryAllocFailure::OK:
            return "ok";
        case FunctionMemoryAllocFailure::EXCLUSIVE_OUTCAST:
            return "exclusive-outcast";
        case FunctionMemoryAllocFailure::INNER_TEMPORAL_OUTCAST:
            return "inner-temporal-outcast";
        case FunctionMemoryAllocFailure::RUNTIME_OUTCAST_METADATA:
            return "runtime-outcast-metadata";
        default:
            return "unknown";
    }
}

class DeviceWorkspaceAllocator {
public:
    DeviceWorkspaceAllocator() = default;
    explicit DeviceWorkspaceAllocator(DevAscendProgram* base) : devProg_(base) {}
    ~DeviceWorkspaceAllocator() = default;
    void Init(DevStartArgs* devStartArgs);

    uintdevptr_t StackWorkspaceAddr() const { return stackWorkspaceBase_; }
    uint64_t StandardStackWorkspacePerCore() const { return standardStackWorkspacePerCore_; }

    // Persist epoch on DevAscendProgram (workspace allocator is stack-local per launch).
    // Epoch in entry high bits invalidates stale cells; skip full-cache memset on the hot path.
    // Packed epoch 0 is skipped: memset once then continue with the next non-zero packed value.
    // Also called when packed taskId (low 16 bits) wraps so old entries cannot false-match.
    void AdvanceStitchCacheEpoch()
    {
        uint32_t nextEpoch = devProg_->stitchCacheEpoch_ + 1U;
        if (unlikely((nextEpoch & 0xFFFFU) == 0)) {
            nextEpoch++;
            if (stitchCacheAddr_ != 0) {
                (void)memset_s(reinterpret_cast<void*>(stitchCacheAddr_), devProg_->memBudget.metadata.stitchCacheSize,
                               0, devProg_->memBudget.metadata.stitchCacheSize);
            }
        }
        devProg_->stitchCacheEpoch_ = nextEpoch;
    }
    void AllocateStitchCache()
    {
        stitchCacheAddr_ = metadataAllocators_.general.Malloc(devProg_->memBudget.metadata.stitchCacheSize).ptr;
        rootFuncMaxCallOpsize_ = devProg_->rootFuncMaxCallOpsize;
        AdvanceStitchCacheEpoch();
    }
    uint64_t* StitchCacheAddr() const { return reinterpret_cast<uint64_t*>(stitchCacheAddr_); }
    uint32_t RootFuncMaxCallOpsize() const { return rootFuncMaxCallOpsize_; }
    uint16_t StitchCacheEpoch() const { return static_cast<uint16_t>(devProg_->stitchCacheEpoch_ & 0xFFFFU); }

    DevAscendProgram* GetDevProg() const { return devProg_; }

#if DEBUG_INFINITE_LIFETIME
    uintdevptr_t DumpTensorWsBaseAddr() const { return dumpTensorWsAllocator_.MemBaseAddr(); }
    uint64_t DumpTensorWsSize() const { return dumpTensorWsAllocator_.Capacity(); }
#endif
    template <typename T, WsMemCategory category, typename WsAllocator_T>
    void SetupVector(Vector<T, category, WsAllocator_T>& vector)
    {
        if constexpr (std::is_same_v<WsAllocator_T, npu::tile_fwk::dynamic::DeviceWorkspaceAllocator>) {
            vector.InitAllocator((*this));
        } else {
            vector.InitAllocator(metadataAllocators_.general);
        }
    }

    template <typename T>
    void SetupItemPool(ItemPool<T>& pool, size_t count, WsMemCategory category)
    {
        pool.Init(metadataAllocators_.general, count, category);
    }

    void SwitchWParallelWorkSpace(uint32_t parallelWsId) { curParallelWsId = parallelWsId; }

private:
    struct MemoryInfo {
        uintdevptr_t ptr;
        size_t size;
        DevAscendFunctionDupped dup;
        size_t stitchedListIndex;
        size_t rawIndex;

        void DumpError() const;
    };

    WsMemoryState VerifyTensorMemoryState(uintdevptr_t ptr, size_t size) const
    {
        return tensorWsVerifier_.Verify(ptr, size);
    }

    bool IsValidWsTensor(uintdevptr_t ptr, size_t memSize) const
    {
        return slotVerifier_.Verify(ptr, memSize) == WsMemoryState::INSIDE ||
               dassembleDestsTensorVerifier_.Verify(ptr, memSize) == WsMemoryState::INSIDE ||
               rootInnerWsVerifier_.Verify(ptr, memSize) == WsMemoryState::INSIDE ||
               devTaskInnerExclusiveOutcastsWsVerifier_.Verify(ptr, memSize) == WsMemoryState::INSIDE;
    }

public:
    void VerifyStitchedListMemory(DevStartArgs& args, const DevAscendFunctionDupped* stitchedList, size_t size);

private:
    void AllocateFunctionInnerWorkspace(DevAscendFunctionDupped dup, uint64_t rootInnerMemReq,
                                        [[maybe_unused]] WsAllocatorCounter* dfxCounter);

    // Helper: allocate outcast workspace for a duplicated root function
    void AllocateOutcastWorkspaceForDup(DevAscendFunctionDupped devRootDup,
                                        [[maybe_unused]] WsAllocatorCounter* pDfxCounter);

    // Helper: allocate inner workspace for a duplicated root function
    void AllocateInnerWorkspaceForDup(DevAscendFunctionDupped devRootDup, WsAllocatorCounter* pDfxCounter);

    // Helper: assign incast address descriptors for a duplicated root function
    void AssignIncastAddresses(DevAscendFunctionDupped devRootDup, DeviceExecuteSlot* slotList);

    // Helper: assign outcast address descriptors for a duplicated root function
    void ResolveOutcastAddress(DevAscendFunctionDupped devRootDup, DevAscendFunction* devRootSrc,
                               DeviceExecuteSlot* slotList, size_t outcastIdx, int outputSlotIndex,
                               int assembleSlotIndex, uintdevptr_t outcastBaseAddr, AddressDescriptor& outcastDesc);

    void AssignOutcastAddresses(DevAscendFunctionDupped devRootDup, DeviceExecuteSlot* slotList);

    FunctionMemoryAllocFailure CanAllocateFunctionMemory(DevAscendFunctionDupped devRootDup,
                                                         DeviceExecuteSlot* slotList);

    void TryAllocateDynamicCellMatchForAssembleSlot(DeviceExecuteSlot& slot);

public:
#if DEBUG_INFINITE_LIFETIME
    WsAllocation DebugDumpTensorAllocate(size_t memReq, WsMemCategory category = WsMemCategory::UNCLASSIFIED)
    {
        DEV_ASSERT_MSG(WsErr::WORKSPACE_INIT_RESOURCE_ERROR, dumpTensorWsAllocator_.CanAllocate(memReq),
                       "dumpTensorWsAllocator_ cannot allocate requested memory unexpectedly, memReq=%zu", memReq);
        WsAllocation allocation = dumpTensorWsAllocator_.Malloc(memReq, category);
        *dumpTensorWsAllocatorCounter_ = dumpTensorWsAllocator_.AllocatedSize();
        return allocation;
    }
#endif

    FunctionMemoryAllocFailure TryAllocateFunctionMemory(DevAscendFunctionDupped devRootDup,
                                                         DeviceExecuteSlot* slotList)
    {
        AutoScopedPerf asp(PERF_EVT_ALLOCATE_WORKSPACE);

        devRootDup.DupDataForDynFuncData()->InitDuppedRuntimeTail();

        const FunctionMemoryAllocFailure status = CanAllocateFunctionMemory(devRootDup, slotList);
        if (status != FunctionMemoryAllocFailure::OK) {
            DEV_INFO("[Workspace Runtime Alloc Failure] reason=%s", FunctionMemoryAllocFailureName(status));
            return status;
        }

        WsAllocatorCounter* pDfxCounter = nullptr;
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        WsAllocatorCounter funcAllocDfx;
        pDfxCounter = &funcAllocDfx;
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL

        // Allocate required workspaces and assign descriptors using helpers
        AllocateOutcastWorkspaceForDup(devRootDup, pDfxCounter);
        AllocateInnerWorkspaceForDup(devRootDup, pDfxCounter);

        AssignIncastAddresses(devRootDup, slotList);
        AssignOutcastAddresses(devRootDup, slotList);
        DEV_IF_INFO { ObserveTensorPoolUsage(curParallelWsId); }

#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        funcAllocDfx.DelayedDumpAsRootFuncAndReset(wsMemDelayedDumper_, devRootDup.GetSource()->GetRawName());
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        return FunctionMemoryAllocFailure::OK;
    }

    bool IsValidSlotMemRequirement(uint64_t memReq) const;

    bool HasDynamicCellMatchSlots() const { return devProg_->memBudget.metadata.dynamicCellMatchSlotNum != 0; }
    uint64_t DynamicCellMatchSlotByteSize() const { return devProg_->memBudget.metadata.maxDynamicCellMatchTableMem; }
    uint64_t DynamicCellMatchSlotCellCapacity() const { return DynamicCellMatchSlotByteSize() / sizeof(uint64_t); }

    WsAllocation AllocateBoundaryOutcastSlot([[maybe_unused]] const char* rootFuncName = nullptr)
    {
        return AllocateFromOutcastSlotPool(tensorAllocators_[curParallelWsId].devTaskBoundaryOutcasts, rootFuncName);
    }

    WsAllocation AllocateInnerTemporalOutcastSlot([[maybe_unused]] const char* rootFuncName = nullptr)
    {
        return AllocateFromOutcastSlotPool(tensorAllocators_[curParallelWsId].devTaskInnerTemporalOutcasts,
                                           rootFuncName);
    }

    ItemPoolIter MakeRuntimeOutcastTensor(WsAllocation allocation, RuntimeTensorMemProperty property)
    {
        return runtimeOutcastTensorPool_.Allocate(allocation, property, 1);
    }

    // Virgin bump: Fill I/O is the first Allocate on a freshly Init()'d pool, so the
    // freelist is empty. Skip ItemAt bounds check and freelist branch per item.
    ItemPoolIter MakeRuntimeOutcastTensorBump(WsAllocation allocation, RuntimeTensorMemProperty property)
    {
        return runtimeOutcastTensorPool_.AllocateBump(allocation, property, 1);
    }

    ItemPool<RuntimeOutcastTensor>::ItemBlock* GetRuntimeOutcastTensorPoolBase()
    {
        return reinterpret_cast<ItemPool<RuntimeOutcastTensor>::ItemBlock*>(&runtimeOutcastTensorPool_.At(0));
    }

    ItemPool<RuntimeOutcastTensor>* GetRuntimeOutcastTensorPool() { return &runtimeOutcastTensorPool_; }

    RuntimeOutcastTensor& GetRuntimeOutcastTensor(ItemPoolIter iter)
    {
        DEV_ASSERT(WsErr::WORKSPACE_ITER_INVALID, iter != ITEM_POOL_INVALID_INDEX);
        return runtimeOutcastTensorPool_.At(iter);
    }

    void RuntimeOutcastTensorDeref(ItemPoolIter iter)
    {
        DEV_ASSERT(WsErr::WORKSPACE_ITER_INVALID, iter != ITEM_POOL_INVALID_INDEX);
        RuntimeOutcastTensorDeref(runtimeOutcastTensorPool_.At(iter));
    }

    void RuntimeOutcastTensorDeref(RuntimeOutcastTensor& outcast)
    {
        DEV_ASSERT(WsErr::WORKSPACE_REFCOUNT_INVALID, outcast.refCnt > 0);
        outcast.refCnt--;
        if (outcast.refCnt == 0) {
            RuntimeOutcastTensorDestruct(outcast);
        }
    }

    void RuntimeOutcastTensorRef(ItemPoolIter iter)
    {
        DEV_ASSERT(WsErr::WORKSPACE_ITER_INVALID, iter != ITEM_POOL_INVALID_INDEX);
        auto& outcast = runtimeOutcastTensorPool_.At(iter);
        DEV_ASSERT_MSG(WsErr::WORKSPACE_REFCOUNT_INVALID, outcast.refCnt > 0,
                       "Shouldn't ref a possibly destroyed tensor, iter=%" PRId64, iter);
        outcast.refCnt++;
    }

    void RuntimeOutcastTensorDerefSafe(ItemPoolIter iter)
    {
        if (iter != ITEM_POOL_INVALID_INDEX) {
            RuntimeOutcastTensorDeref(iter);
        }
    }

    void RuntimeOutcastTensorRefSafe(ItemPoolIter iter)
    {
        if (iter != ITEM_POOL_INVALID_INDEX) {
            RuntimeOutcastTensorRef(iter);
        }
    }

    void RuntimeOutcastTensorAssign(ItemPoolIter& dst, ItemPoolIter src)
    {
        if (dst == src) {
            return;
        }
        RuntimeOutcastTensorDerefSafe(dst);
        dst = src;
        RuntimeOutcastTensorRefSafe(src);
    }

    void RuntimeOutcastTensorReplaceAddrWithoutRecycle(ItemPoolIter iter, WsAllocation allocation,
                                                       RuntimeTensorMemProperty property)
    {
        DEV_ASSERT(WsErr::WORKSPACE_ITER_INVALID, iter != ITEM_POOL_INVALID_INDEX);
        auto& outcast = runtimeOutcastTensorPool_.At(iter);
        outcast.allocation = allocation;
        outcast.property = property;
    }

private:
    void RuntimeOutcastTensorDestruct(RuntimeOutcastTensor& outcast)
    {
#if !DEBUG_INFINITE_LIFETIME
        if (outcast.property == RuntimeTensorMemProperty::BOUNDARY_OUTCAST) {
            rtBoundaryOutcastToBeFree_.push_back(outcast);
        }
#endif // !DEBUG_INFINITE_LIFETIME
        runtimeOutcastTensorPool_.Destroy(&outcast);
    }

public:
    void TriggerDelayedRecycle();

    void RecycleDevFuncWorkspace();

    DevAscendFunctionDupped DuplicateRoot(DevAscendFunction* func);

    void DestroyDuppedFunc(DevAscendFunctionDupped& dup);

    DynDeviceTask* MakeDynDeviceTask();

    DevAscendFunctionDuppedStitch* AllocateStitch();

    DynFuncHeader* AllocateDynFuncData(uint64_t size);

    npu::tile_fwk::PerCorePendingQueue* AllocatePerCorePendingQueue(uint64_t size);

    npu::tile_fwk::DrcoLocalReadyQueue* AllocateDrcoLocalReadyQueue(uint64_t size);

    npu::tile_fwk::DrcoLocalReadyMatrix* AllocateDrcoLocalReadyMatrix(uint64_t size);

    npu::tile_fwk::DrcoGlobalStitchNodeMatrix* AllocateDrcoStitchNodeMatrix(uint64_t size);

    void ResetAicpuMemCounter();

    void RewindMemoryDumper();

    void MarkAsNewStitchWindow();

    void DumpMemoryUsage(const char* hint) const;

    void LogTuningUsage(uint64_t taskId, size_t stitchCount) const;

    void LogTuningSummary() const;

    void ResetTuningDeviceTaskPeaks();

    void InitMetadataSlabAllocator();

    static uint64_t CalcMetadataItemPoolMemSize(const DevAscendProgram* devProg);

    static uint64_t CalcMetadataVectorMemSize(const DevAscendProgram* devProg);

    static uint64_t CalcMetadataSlotAllocatorMemSize(const DevAscendProgram* devProg);

    uint32_t CalcSlabMemObjmaxSize();

    uint32_t CalcStitchSlabMemObjmaxSize(uint32_t* slabCapacity);

    void CalculateSlabCapacityPerType(uint32_t slabSize, uint32_t* slabCapacity, uint32_t slabTypeNum);
    WsAllocation SlabAlloc(uint32_t objSize, WsAicpuSlabMemType type);

    WsSlabStageAllocMem SlabGetStageAllocMem(bool keepTail, WsAicpuSlabMemType keepType);
    void DumpSlabUsageBeforeSubmit(uint32_t taskId, DynDeviceTask* devTask);

    void SlabStageAllocMemSubmmit(DynDeviceTask* devTask);

    /* support vector allocator,so need have this fucntion member */
    template <typename T>
    WsAllocation Allocate(uint64_t count, WsMemCategory category)
    {
        DEV_ASSERT_MSG(WsErr::WORKSPACE_CATEGORY_INVALID, category == WsMemCategory::VECTOR_STITCHED_LIST,
                       "Unexpected category=%s", GetCategoryName(category));
        uint64_t size = count * sizeof(T);
        return ControlFlowAllocateSlab(devProg_, size, SlabAlloc(size, WsAicpuSlabMemType::VEC_STITCHED_LIST));
    }

    void Deallocate(WsAllocation) {} // just for support vector allocator,so need have this fucntion member

private:
    void InitMetadataAllocators(DevAscendProgram* devProg, DevStartArgs* devStartArgs);

    void InitTensorAllocators(uintdevptr_t workspaceAddr, uint64_t tensorWorkspaceSize, DevAscendProgram* devProg);

    void InitAICoreSpilledMemory(uintdevptr_t workspaceAddr, DevAscendProgram* devProg);

    uint32_t DevFunctionDuppedSlabMemObjSize();

    /*计算使用vector的元数据的数据结构大小*/
    static uint64_t CalculateVectorCapacity(uint64_t size);

    /* 按照devicetask最大支持stitch阈值分配对象 */
    uint32_t DynFuncDataSlabMemObjSize();

    /* 按照devicetask最大支持stitch阈值分配对象 */
    uint32_t VecStitchListSLabMemObjSize();

    uint32_t DynDevTaskSlabMemObjSize();

    uint32_t ShmemWaitUntilCacheSlabMemObjSize();

    uint32_t DuppedStitchSlabMemObjSize();

    uint32_t ReadyQueSlabMemObjSize();

    uint32_t DieReadyQueSlabMemObjSize();

    uint32_t WrapQueSlabMemObjSize();

    uint32_t PerCorePendingQueSlabMemObjSize();

    uint32_t LocalReadyQueSlabMemObjSize();

    uint32_t PredCountSlabMemObjSize();

    uint32_t (DeviceWorkspaceAllocator::*slabMemObjSizeFunc[ToUnderlying(WsAicpuSlabMemType::SLAB_MEM_TYPE_BUTT)])() = {
        &DeviceWorkspaceAllocator::DevFunctionDuppedSlabMemObjSize,
        &DeviceWorkspaceAllocator::DynFuncDataSlabMemObjSize,
        &DeviceWorkspaceAllocator::VecStitchListSLabMemObjSize,
        &DeviceWorkspaceAllocator::DynDevTaskSlabMemObjSize,
        &DeviceWorkspaceAllocator::ShmemWaitUntilCacheSlabMemObjSize,
        nullptr, // invalid type
        &DeviceWorkspaceAllocator::ReadyQueSlabMemObjSize,
        &DeviceWorkspaceAllocator::DieReadyQueSlabMemObjSize,
        &DeviceWorkspaceAllocator::WrapQueSlabMemObjSize,
        &DeviceWorkspaceAllocator::PerCorePendingQueSlabMemObjSize,
        &DeviceWorkspaceAllocator::LocalReadyQueSlabMemObjSize,
        &DeviceWorkspaceAllocator::PredCountSlabMemObjSize,
        &DeviceWorkspaceAllocator::DuppedStitchSlabMemObjSize,
    };

    /* 根据当前算子的业务模型分析计算出slab 管理内存页大小, 基于当前可评估的所有内存类型的最大值评估 */
    uint32_t CalcAicpuMetaSlabAlloctorSlabMemObjmaxSize();
    uint32_t CalcAicpuMetaSlabAlloctorSlabPageSize(uint32_t totalMemSize);

    void InitAicpuStitchSlabAllocator(void* memBase, uint32_t totalSize);

    void SlabTryDynAddCache(WsAicpuSlabMemType type, uint32_t objSize);

public:
    TensorAllocator* GetTensorAllocator() { return tensorAllocators_; }

    // stitch pool 对齐后基址（设备地址）：由 DrcoRootFuncList::stitchNodeBase 透传给 aicore 侧，
    // 以 base + u32 偏移还原 stitch 节点地址（偏移 < 4GB 由池 u32 总大小保证）
    uint64_t GetStitchPoolBase() const { return metadataAllocators_.stitchSlab.GetMemBase(); }

private:
    bool DeviceTaskMemTryRecycle();

    void ObserveTensorPoolUsage(uint32_t parallelId);

private:
    WsAllocation AllocateFromOutcastSlotPool(WsSlotAllocator& pool, [[maybe_unused]] const char* rootFuncName = nullptr)
    {
        WsAllocation allocation;
#if !DEBUG_INFINITE_LIFETIME
        allocation = pool.Allocate();
        allocation.parallelWsId = curParallelWsId;
#else
        allocation = DebugDumpTensorAllocate(devProg_->memBudget.tensor.MaxOutcastMem(),
                                             WsMemCategory::TENSOR_ROOTFUNC_OUTCAST_SLOT);
#endif
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        wsMemDelayedDumper_.LogTensorMalloc(rootFuncName == nullptr ? "unspecified_root" : rootFuncName, allocation);
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        return allocation;
    }

    uint32_t curParallelWsId{0};
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
    DelayedDumper wsMemDelayedDumper_;
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL

    MetadataAllocator metadataAllocators_;
    TensorAllocator tensorAllocators_[SCH_DEVTASK_MAX_PARALLELISM];

#if DEBUG_INFINITE_LIFETIME
    SeqWsAllocator dumpTensorWsAllocator_;
    uint64_t* dumpTensorWsAllocatorCounter_; // used in host-side when reading back npu memory
#endif

    uintdevptr_t stackWorkspaceBase_{0};
    uint64_t standardStackWorkspacePerCore_{0};
    uint64_t stackWorkspaceSize_{0};

    uint32_t maxDevFuncDuppedSize_{0};
    DevAscendProgram* devProg_{nullptr};

    WsMemoryVerifier tensorWsVerifier_;
    WsMemoryVerifier slotVerifier_;
    WsMemoryVerifier dassembleDestsTensorVerifier_;
    WsMemoryVerifier rootInnerWsVerifier_;
    WsMemoryVerifier devTaskInnerExclusiveOutcastsWsVerifier_;

    Vector<RuntimeOutcastTensor, WsMemCategory::VECTOR_RUNTIME_OUTCAST_RECYCLE_LIST> rtBoundaryOutcastToBeFree_;
    SPSCQueue<DynDeviceTask*, SUBMMIT_TASK_QUE_SIZE> submmitTaskQueue_;

    ItemPool<RuntimeOutcastTensor> runtimeOutcastTensorPool_;

    struct TensorPoolUsagePeak {
        uint64_t rootInnerBytes;
        uint64_t exclusiveOutcastBytes;
        uint64_t innerTemporalOutcastSlots;
    };

    struct WorkspaceTuningState {
        TensorPoolUsagePeak deviceTaskUsagePeak[SCH_DEVTASK_MAX_PARALLELISM];
        TensorPoolUsagePeak launchUsagePeak[SCH_DEVTASK_MAX_PARALLELISM];
    };

    WorkspaceTuningState tuningState_{};

    uint64_t stitchCacheAddr_{0};
    uint32_t rootFuncMaxCallOpsize_{0};
};
} // namespace npu::tile_fwk::dynamic
#endif
