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
 * \file dev_workspace.cpp
 * \brief
 */

#include "machine/utils/dynamic/dev_workspace.h"

#include <set>
#include <string>

namespace npu::tile_fwk::dynamic {
namespace {
uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return divisor == 0 ? 0 : value / divisor + static_cast<uint64_t>(value % divisor != 0);
}

uint64_t AtLeastRequiredDepth(uint64_t value, uint64_t unitRequirement)
{
    return value == 0 && unitRequirement != 0 ? 1 : value;
}

// Conservative Assemble preflight submits when remaining slots < one depth. Recommend one extra
// depth so filling stitch_function_num_per_pool from this value does not immediately re-trigger submit.
uint64_t RecommendAssembleOutcastDepth(uint64_t observedDepth, uint64_t assembleSlotsPerDepth)
{
    uint64_t recommended = AtLeastRequiredDepth(observedDepth, assembleSlotsPerDepth);
    if (observedDepth != 0) {
        recommended += 1;
    }
    if (recommended > MAX_STITCH_FUNC_NUM) {
        recommended = MAX_STITCH_FUNC_NUM;
    }
    return recommended;
}
} // namespace

void DeviceWorkspaceAllocator::Init(DevStartArgs* devStartArgs)
{
    uintdevptr_t baseAddr = devStartArgs->contextWorkspaceAddr;
    DevAscendProgram* devProg = devStartArgs->devProg;
    devProg_ = devProg;
    DEV_IF_INFO { tuningState_ = {}; }
    // Host coherent allocators MUST be initialized EARLIEST since some other allocators might depend on them
    InitMetadataAllocators(devProg, devStartArgs);

    InitAICoreSpilledMemory(baseAddr, devProg);
    baseAddr += devProg->memBudget.aicoreSpilled.Total();

    // dassembleDests contains dynamic workspace, put it to the end
    InitTensorAllocators(baseAddr, devProg->memBudget.tensor.Total(), devProg);
    baseAddr += devProg->memBudget.tensor.Total();

#if DEBUG_INFINITE_LIFETIME
    dumpTensorWsAllocator_.InitTensorAllocator(baseAddr, devProg->memBudget.debug.dumpTensor);
    DEV_DEBUG("[DumpTensor] dumpTensorWsAllocator_: ptr=0x%lx, size=%lu", baseAddr,
              devProg->memBudget.debug.dumpTensor);
    baseAddr += devProg->memBudget.debug.dumpTensor;

    // Allocate 512 for address alignment
    dumpTensorWsAllocatorCounter_ = dumpTensorWsAllocator_.Malloc(TENSOR_ADDR_ALIGNMENT).As<uint64_t>();
    *dumpTensorWsAllocatorCounter_ = dumpTensorWsAllocator_.AllocatedSize();
#endif
    SetupVector(rtBoundaryOutcastToBeFree_);
    rtBoundaryOutcastToBeFree_.reserve(devProg->memBudget.tensor.BoundaryAndInnerTemporalOutcastSlotNum());

    SetupItemPool(runtimeOutcastTensorPool_, devProg->memBudget.tensor.runtimeOutcastPoolSize,
                  WsMemCategory::ITEMPOOL_RUNTIME_OUTCAST);
    DEV_IF_INFO { ResetTuningDeviceTaskPeaks(); }
}

void DeviceWorkspaceAllocator::ObserveTensorPoolUsage(uint32_t parallelId)
{
    auto& allocators = tensorAllocators_[parallelId];
    auto& deviceTaskPeak = tuningState_.deviceTaskUsagePeak[parallelId];
    auto& launchPeak = tuningState_.launchUsagePeak[parallelId];
    const uint64_t rootInnerBytes = allocators.rootInner.AllocatedSize();
    const uint64_t exclusiveBytes = allocators.devTaskInnerExclusiveOutcasts.AllocatedSize();
    const uint64_t innerTemporalSlots = devProg_->memBudget.tensor.devTaskInnerTemporalOutcastNum -
                                        allocators.devTaskInnerTemporalOutcasts.AvailableSlots();
    deviceTaskPeak.rootInnerBytes = rootInnerBytes > deviceTaskPeak.rootInnerBytes ? rootInnerBytes :
                                                                                     deviceTaskPeak.rootInnerBytes;
    deviceTaskPeak.exclusiveOutcastBytes = exclusiveBytes > deviceTaskPeak.exclusiveOutcastBytes ?
                                               exclusiveBytes :
                                               deviceTaskPeak.exclusiveOutcastBytes;
    deviceTaskPeak.innerTemporalOutcastSlots = innerTemporalSlots > deviceTaskPeak.innerTemporalOutcastSlots ?
                                                   innerTemporalSlots :
                                                   deviceTaskPeak.innerTemporalOutcastSlots;
    launchPeak.rootInnerBytes = rootInnerBytes > launchPeak.rootInnerBytes ? rootInnerBytes : launchPeak.rootInnerBytes;
    launchPeak.exclusiveOutcastBytes = exclusiveBytes > launchPeak.exclusiveOutcastBytes ?
                                           exclusiveBytes :
                                           launchPeak.exclusiveOutcastBytes;
    launchPeak.innerTemporalOutcastSlots = innerTemporalSlots > launchPeak.innerTemporalOutcastSlots ?
                                               innerTemporalSlots :
                                               launchPeak.innerTemporalOutcastSlots;
}

void DeviceWorkspaceAllocator::ResetTuningDeviceTaskPeaks()
{
    for (uint32_t i = 0; i < devProg_->GetParallelism(); ++i) {
        auto& allocators = tensorAllocators_[i];
        auto& peak = tuningState_.deviceTaskUsagePeak[i];
        peak.rootInnerBytes = allocators.rootInner.AllocatedSize();
        peak.exclusiveOutcastBytes = allocators.devTaskInnerExclusiveOutcasts.AllocatedSize();
        peak.innerTemporalOutcastSlots = devProg_->memBudget.tensor.devTaskInnerTemporalOutcastNum -
                                         allocators.devTaskInnerTemporalOutcasts.AvailableSlots();
    }
}

void DeviceWorkspaceAllocator::LogTuningUsage(uint64_t taskId, size_t stitchCount) const
{
    const auto& tensorBudget = devProg_->memBudget.tensor;
    uint64_t maxRootInnerDepth = 0;
    uint64_t maxExclusiveDepth = 0;
    uint64_t maxInnerTemporalDepth = 0;
    for (uint32_t i = 0; i < devProg_->GetParallelism(); ++i) {
        const auto& allocators = tensorAllocators_[i];
        const auto& peak = tuningState_.deviceTaskUsagePeak[i];
        const uint64_t rootCurrent = allocators.rootInner.AllocatedSize();
        const uint64_t rootCapacity = allocators.rootInner.Capacity();
        const uint64_t exclusiveCurrent = allocators.devTaskInnerExclusiveOutcasts.AllocatedSize();
        const uint64_t exclusiveCapacity = allocators.devTaskInnerExclusiveOutcasts.Capacity();
        const uint64_t innerCapacity = tensorBudget.devTaskInnerTemporalOutcastNum;
        const uint64_t innerCurrent = innerCapacity - allocators.devTaskInnerTemporalOutcasts.AvailableSlots();
        DEV_INFO("[Workspace Runtime Pool] parallelId=%u, "
                 "rootInner={current=%" PRIu64 ",peak=%" PRIu64 ",capacity=%" PRIu64 "}, "
                 "exclusiveOutcast={current=%" PRIu64 ",peak=%" PRIu64 ",capacity=%" PRIu64 "}, "
                 "assembleOutcast={currentSlots=%" PRIu64 ",peakSlots=%" PRIu64 ",capacitySlots=%" PRIu64 "}.",
                 i, rootCurrent, peak.rootInnerBytes, rootCapacity, exclusiveCurrent, peak.exclusiveOutcastBytes,
                 exclusiveCapacity, innerCurrent, peak.innerTemporalOutcastSlots, innerCapacity);
        const uint64_t rootInnerDepth = CeilDiv(peak.rootInnerBytes, tensorBudget.workspacePool.rootInnerUnitBytes);
        const uint64_t exclusiveDepth = CeilDiv(peak.exclusiveOutcastBytes,
                                                tensorBudget.workspacePool.exclusiveOutcastUnitBytes);
        const uint64_t innerTemporalDepth = CeilDiv(peak.innerTemporalOutcastSlots,
                                                    tensorBudget.workspacePool.assembleSlotsPerDepth);
        maxExclusiveDepth = exclusiveDepth > maxExclusiveDepth ? exclusiveDepth : maxExclusiveDepth;
        maxRootInnerDepth = rootInnerDepth > maxRootInnerDepth ? rootInnerDepth : maxRootInnerDepth;
        maxInnerTemporalDepth = innerTemporalDepth > maxInnerTemporalDepth ? innerTemporalDepth : maxInnerTemporalDepth;
    }
    DEV_INFO("[Workspace Runtime Tuning] taskId=%" PRIu64 ", stitchCount=%zu, "
             "configuredDepths={rootInner=%u,innerTemporal=%u,exclusive=%u}, "
             "recommendedDepths={rootInner=%" PRIu64 ",assembleOutcast=%" PRIu64 ",exclusive=%" PRIu64 "}.",
             taskId, stitchCount, tensorBudget.workspacePool.stitchFunctionNumPerPool[0],
             tensorBudget.workspacePool.stitchFunctionNumPerPool[1],
             tensorBudget.workspacePool.stitchFunctionNumPerPool[2],
             AtLeastRequiredDepth(maxRootInnerDepth, tensorBudget.workspacePool.rootInnerUnitBytes),
             RecommendAssembleOutcastDepth(maxInnerTemporalDepth, tensorBudget.workspacePool.assembleSlotsPerDepth),
             AtLeastRequiredDepth(maxExclusiveDepth, tensorBudget.workspacePool.exclusiveOutcastUnitBytes));
}

void DeviceWorkspaceAllocator::MemoryInfo::DumpError() const
{
    std::string ioPropertyDump;
    switch (dup.GetSource()->GetRawTensor(rawIndex)->ioProperty) {
        case DevIOProperty::ROOT_INCAST:
            ioPropertyDump = " (Root Incast)";
            break;
        case DevIOProperty::ROOT_OUTCAST:
            ioPropertyDump = " (Root Outcast)";
            break;
        default:
            break;
    }
    DEV_INFO("  Func (%2zu) %16s rawTensor[%2zu], @%" PRIx64 " [%zu bytes]%s.", stitchedListIndex,
             dup.GetSource()->GetRawName(), rawIndex, ptr, size, ioPropertyDump.c_str());
}

void DeviceWorkspaceAllocator::VerifyStitchedListMemory(DevStartArgs& args, const DevAscendFunctionDupped* stitchedList,
                                                        size_t size)
{
    std::set<uintdevptr_t> inoutAddr;
    for (int i = 0; i < args.GetInputTensorSize(); i++) {
        inoutAddr.insert(args.GetInputTensor(i).address);
    }
    for (int i = 0; i < args.GetOutputTensorSize(); i++) {
        inoutAddr.insert(args.GetOutputTensor(i).address);
    }

    bool verificationSuccess = true;
    for (size_t i = 0; i < size; i++) {
        const auto& dup = stitchedList[i];

        size_t rawTensorCount = dup.GetSource()->GetRawTensorSize();
        for (size_t j = 0; j < rawTensorCount; j++) {
            auto* rawTensor = dup.GetSource()->GetRawTensor(j);
            auto memReq = rawTensor->GetMemoryRequirement(dup.GetExpressionAddr());
            MemoryInfo memInfo{
                dup.GetRawTensorAddr(j), rawTensor->ioProperty == DevIOProperty::NONE ? 0 : memReq, dup, i, j,
            };
            switch (VerifyTensorMemoryState(memInfo.ptr, memInfo.size)) {
                case WsMemoryState::INSIDE:
                    if (!IsValidWsTensor(memInfo.ptr, memInfo.size)) {
                        DEV_ERROR(WsErr::WS_TENSOR_ADDRESS_OUT_OF_RANGE,
                                  "workspace.verify.tensor: Invalid workspace tensor (not completely inside any "
                                  "workspace segment):");
                        memInfo.DumpError();
                        verificationSuccess = false;
                    }
                    break;
                case WsMemoryState::CROSS_BOUNDARY:
                    DEV_ERROR(WsErr::WS_TENSOR_ADDRESS_OUT_OF_RANGE,
                              "workspace.verify.tensor: Memory crossing workspace boundary:");
                    memInfo.DumpError();
                    verificationSuccess = false;
                    break;
                default:
                    if (!inoutAddr.count(memInfo.ptr)) {
                        DEV_ERROR(WsErr::WS_TENSOR_ADDRESS_OUT_OF_RANGE,
                                  "workspace.verify.tensor: Non input/output tensor outside of workspace:");
                        memInfo.DumpError();
                        verificationSuccess = false;
                    }
                    break;
            }
        }
    }
    DEV_ASSERT(WsErr::WORKSPACE_INIT_RESOURCE_ERROR, verificationSuccess);
}

void DeviceWorkspaceAllocator::AllocateFunctionInnerWorkspace(DevAscendFunctionDupped dup, uint64_t rootInnerMemReq,
                                                              [[maybe_unused]] WsAllocatorCounter* dfxCounter)
{
    if (!tensorAllocators_[curParallelWsId].rootInner.CanAllocate(rootInnerMemReq)) {
        DEV_IF_INFO { ObserveTensorPoolUsage(curParallelWsId); }
        tensorAllocators_[curParallelWsId].rootInner.ResetPool();
        DEV_ASSERT_MSG(WsErr::WORKSPACE_INIT_RESOURCE_ERROR,
                       tensorAllocators_[curParallelWsId].rootInner.CanAllocate(rootInnerMemReq),
                       "After reset, still cannot allocate root inner workspace unexpectedly, memReq=%" PRIu64,
                       rootInnerMemReq);
    }
    WsAllocation allocation = tensorAllocators_[curParallelWsId].rootInner.Malloc(
        rootInnerMemReq, WsMemCategory::TENSOR_ROOTFUNC_INTERNAL);
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
    if (dfxCounter) {
        dfxCounter->LogMalloc(allocation);
    }
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
    dup.RuntimeWorkspace() = allocation.ptr;
    auto& reuseInfo = dup.GetRuntimeReuseInfo();
    reuseInfo.poolResetTimes = tensorAllocators_[curParallelWsId].rootInner.ResetTimes();
}

// Helper: allocate outcast workspace for a duplicated root function
void DeviceWorkspaceAllocator::AllocateOutcastWorkspaceForDup(DevAscendFunctionDupped devRootDup,
                                                              [[maybe_unused]] WsAllocatorCounter* pDfxCounter)
{
    DevAscendFunction* devRootSrc = devRootDup.GetSource();
    size_t outcastMemReq = devRootSrc->exclusiveOutcastWsMemoryRequirement;
    if (outcastMemReq != 0) {
        DEV_ASSERT(WsErr::WORKSPACE_INIT_RESOURCE_ERROR,
                   tensorAllocators_[curParallelWsId].devTaskInnerExclusiveOutcasts.CanAllocate(outcastMemReq));
        WsAllocation allocation = tensorAllocators_[curParallelWsId].devTaskInnerExclusiveOutcasts.Malloc(
            outcastMemReq, WsMemCategory::TENSOR_ROOTFUNC_INTERNAL);
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        if (pDfxCounter) {
            pDfxCounter->LogMalloc(allocation);
        }
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
#if DEBUG_INFINITE_LIFETIME
        allocation = DebugDumpTensorAllocate(outcastMemReq, WsMemCategory::TENSOR_ROOTFUNC_INTERNAL);
#endif
        devRootDup.RuntimeOutcastBase() = allocation.ptr;
    } else {
        devRootDup.RuntimeOutcastBase() = 0;
    }
}

// Helper: allocate inner workspace for a duplicated root function
void DeviceWorkspaceAllocator::AllocateInnerWorkspaceForDup(DevAscendFunctionDupped devRootDup,
                                                            WsAllocatorCounter* pDfxCounter)
{
    DevAscendFunction* devRootSrc = devRootDup.GetSource();
    size_t rootInnerMemReq = devRootSrc->rootInnerTensorWsMemoryRequirement;
    if (rootInnerMemReq != 0) {
        AllocateFunctionInnerWorkspace(devRootDup, rootInnerMemReq, pDfxCounter);
#if DEBUG_INFINITE_LIFETIME
        WsAllocation allocation = DebugDumpTensorAllocate(rootInnerMemReq, WsMemCategory::TENSOR_ROOTFUNC_INTERNAL);
        devRootDup.RuntimeWorkspace() = allocation.ptr;
#endif
    } else {
        devRootDup.RuntimeWorkspace() = 0;
    }
}

// Helper: assign incast address descriptors for a duplicated root function
void DeviceWorkspaceAllocator::AssignIncastAddresses(DevAscendFunctionDupped devRootDup, DeviceExecuteSlot* slotList)
{
    DevAscendFunction* devRootSrc = devRootDup.GetSource();
    const size_t incastSize = devRootSrc->GetIncastSize();
    for (size_t i = 0; i < incastSize; ++i) {
        auto& incast = devRootSrc->GetIncast(i);
        DEV_ASSERT_MSG(WsErr::WORKSPACE_ITER_INVALID, incast.fromSlotList.size() > 0,
                       "Root [%s] Incast %zu has no fromSlotList.", devRootSrc->GetRawName(), i);

        int slotIndex = devRootSrc->At(incast.fromSlotList, 0);
        DEV_ASSERT_MSG(WsErr::WORKSPACE_ITER_INVALID, slotList[slotIndex].rtOutcastIter != ITEM_POOL_INVALID_INDEX,
                       "Root[%s] incast %zu  slotIndex %d read from empty address.", devRootSrc->GetRawName(), i,
                       slotIndex);
        auto& incastDesc = devRootDup.GetIncastAddress(i);
        incastDesc = AddressDescriptor::MakeFromRtOutcast(slotList[slotIndex].rtOutcastIter);
        RuntimeOutcastTensorRef(incastDesc.GetRtOutcastIter());
        DEV_VERBOSE_DEBUG("get incast %zu, from slot %d address %s.", i, slotIndex, incastDesc.Dump().c_str());
    }
}

// Helper: assign outcast address descriptors for a duplicated root function
void DeviceWorkspaceAllocator::ResolveOutcastAddress(DevAscendFunctionDupped devRootDup, DevAscendFunction* devRootSrc,
                                                     DeviceExecuteSlot* slotList, size_t outcastIdx,
                                                     int outputSlotIndex, int assembleSlotIndex,
                                                     uintdevptr_t outcastBaseAddr, AddressDescriptor& outcastDesc)
{
    auto rawTensor = devRootSrc->GetOutcastRawTensor(outcastIdx);
    if (outputSlotIndex != -1) {
        /* Output tensor: allocate dynamic cell-match table when encode marked the slot */
        if (slotList[outputSlotIndex].isPartialUpdateStitch &&
            slotList[outputSlotIndex].partialUpdate->stitchCtrlBitMask != STITCH_CTRL_NONE) {
            TryAllocateDynamicCellMatchForAssembleSlot(slotList[outputSlotIndex]);
        }
        outcastDesc = AddressDescriptor::MakeFromRtOutcast(slotList[outputSlotIndex].rtOutcastIter);
        RuntimeOutcastTensorRef(outcastDesc.GetRtOutcastIter());
    } else if (rawTensor->linkedIncastId != -1) {
        /* reshape inplace or something */
        auto& incastDesc = devRootDup.GetIncastAddress(rawTensor->linkedIncastId);
        DEV_ASSERT(WsErr::WORKSPACE_CATEGORY_INVALID, incastDesc.IsRtOutcast());
        DEV_ASSERT(WsErr::WORKSPACE_ITER_INVALID, incastDesc.GetRtOutcastIter() != ITEM_POOL_INVALID_INDEX);
        outcastDesc = incastDesc;
        RuntimeOutcastTensorRef(outcastDesc.GetRtOutcastIter());
    } else if (assembleSlotIndex != -1) {
        /* assemble outcast tensor: isAssembleSlotNeedAlloc = local buffer alloc only;
         * cell-match uses stitchCtrlBitMask on partialUpdate (Partial slots only). */
        if (slotList[assembleSlotIndex].isAssembleSlotNeedAlloc) {
            RuntimeOutcastTensorDerefSafe(slotList[assembleSlotIndex].rtOutcastIter);
            slotList[assembleSlotIndex].rtOutcastIter = MakeRuntimeOutcastTensor(
                AllocateInnerTemporalOutcastSlot(devRootSrc->GetRawName()), RuntimeTensorMemProperty::BOUNDARY_OUTCAST);
            slotList[assembleSlotIndex].isAssembleSlotNeedAlloc = false;
            if (slotList[assembleSlotIndex].isPartialUpdateStitch &&
                slotList[assembleSlotIndex].partialUpdate->stitchCtrlBitMask != STITCH_CTRL_NONE) {
                TryAllocateDynamicCellMatchForAssembleSlot(slotList[assembleSlotIndex]);
                slotList[assembleSlotIndex]
                    .ChangeSlotAllocIterId(); // tensor memory changed → invalidate cell-match tagId
            }
        } else {
            DEV_ASSERT_MSG(WsErr::WORKSPACE_ITER_INVALID,
                           slotList[assembleSlotIndex].rtOutcastIter != ITEM_POOL_INVALID_INDEX,
                           "Missing RUNTIME_SlotMarkNeedAlloc for assemble slot %d.", assembleSlotIndex);
        }
        outcastDesc = AddressDescriptor::MakeFromRtOutcast(slotList[assembleSlotIndex].rtOutcastIter);
        RuntimeOutcastTensorRef(outcastDesc.GetRtOutcastIter());
    } else if (devRootSrc->GetOutcast(outcastIdx).exprListIndex != -1) {
        /* something like an expression address, probably shmem */
        uint64_t* exprTbl = devRootDup.GetExpressionAddr();
        uint64_t addr = exprTbl[devRootSrc->GetOutcast(outcastIdx).exprListIndex];
        outcastDesc = AddressDescriptor::MakeFromRtOutcast(
            MakeRuntimeOutcastTensor(WsAllocation(addr, curParallelWsId), RuntimeTensorMemProperty::EXTERNAL));
    } else {
        outcastDesc = AddressDescriptor::MakeFromRtOutcast(MakeRuntimeOutcastTensor(
            WsAllocation(outcastBaseAddr + devRootSrc->GetOutcastRawTensor(outcastIdx)->addrOffset, curParallelWsId),
            RuntimeTensorMemProperty::DEVTASK_INNER_OUTCAST));
    }
}

void DeviceWorkspaceAllocator::AssignOutcastAddresses(DevAscendFunctionDupped devRootDup, DeviceExecuteSlot* slotList)
{
    DevAscendFunction* devRootSrc = devRootDup.GetSource();
    uintdevptr_t outcastBaseAddr = devRootDup.RuntimeOutcastBase();
    const size_t outcastSize = devRootSrc->GetOutcastSize();
    for (size_t i = 0; i < outcastSize; ++i) {
        int outputSlotIndex = -1;
        int assembleSlotIndex = -1;
        auto& toSlotList = devRootSrc->GetOutcast(i).toSlotList;
        const size_t toCnt = toSlotList.size();
        if (toCnt > 0) {
            auto* toSlots = &devRootSrc->At(toSlotList, 0);
            for (size_t k = 0; k < toCnt; ++k) {
                auto idx = toSlots[k];
                if (slotList[idx].IsOutputAddress()) {
                    outputSlotIndex = idx;
                } else if (slotList[idx].IsAssembleAddress()) {
                    assembleSlotIndex = idx;
                }
            }
        }

        AddressDescriptor& outcastDesc = devRootDup.GetOutcastAddress(i);
        ResolveOutcastAddress(devRootDup, devRootSrc, slotList, i, outputSlotIndex, assembleSlotIndex, outcastBaseAddr,
                              outcastDesc);

        DEV_VERBOSE_DEBUG("get outcast %zu slot %d/%d address %s.", i, outputSlotIndex, assembleSlotIndex,
                          outcastDesc.Dump().c_str());
    }
}

FunctionMemoryAllocFailure DeviceWorkspaceAllocator::CanAllocateFunctionMemory(DevAscendFunctionDupped devRootDup,
                                                                               DeviceExecuteSlot* slotList)
{
    (void)slotList;
    DevAscendFunction* devRootSrc = devRootDup.GetSource();
    const size_t outcastSize = devRootSrc->GetOutcastSize();

    const size_t outcastMemReq = devRootSrc->exclusiveOutcastWsMemoryRequirement;
    if (!tensorAllocators_[curParallelWsId].devTaskInnerExclusiveOutcasts.CanAllocate(outcastMemReq)) {
        return FunctionMemoryAllocFailure::EXCLUSIVE_OUTCAST;
    }

    auto& innerTemporalPool = tensorAllocators_[curParallelWsId].devTaskInnerTemporalOutcasts;
    const size_t availableInnerTemporalSlots = innerTemporalPool.AvailableSlots();
    const uint64_t assembleSlotsPerDepth = devProg_->memBudget.tensor.workspacePool.assembleSlotsPerDepth;
    // Conservative: treat one assemble depth as the next-root requirement. Exact per-root assemble
    // slot counting belongs on Host; the extra recommended assemble depth covers this over-estimate.
    if (outcastSize != 0 && availableInnerTemporalSlots < assembleSlotsPerDepth) {
        return FunctionMemoryAllocFailure::INNER_TEMPORAL_OUTCAST;
    }

    if (outcastSize > runtimeOutcastTensorPool_.FreeItemNum()) {
        return FunctionMemoryAllocFailure::RUNTIME_OUTCAST_METADATA;
    }

    return FunctionMemoryAllocFailure::OK;
}

void DeviceWorkspaceAllocator::TryAllocateDynamicCellMatchForAssembleSlot(DeviceExecuteSlot& slot)
{
    // Caller: isPartialUpdateStitch (⇒ partialUpdate set) && mask != NONE.
    auto* partialUpdate = slot.partialUpdate;
    if (partialUpdate->cellMatchRuntimePartialUpdateTable.Data() != nullptr) {
        return;
    }
    auto& desc = partialUpdate->cellMatchTableDesc;
    int dim = desc.GetDimensionSize();
    if (dim <= 0) {
        return;
    }
    auto dynamicCellMatchSlotBytes = DynamicCellMatchSlotByteSize();
    DEV_ASSERT_MSG(WsErr::WORKSPACE_INIT_PARAM_INVALID, dynamicCellMatchSlotBytes > 0,
                   "Dynamic cell match slot bytes invalid, slotBytes=%" PRIu64, dynamicCellMatchSlotBytes);
    DEV_ASSERT_MSG(WsErr::WORKSPACE_INIT_RESOURCE_ERROR, metadataAllocators_.dynamicCellMatch.AvailableSlots() > 0,
                   "Dynamic cell match allocator exhausted, available=%zu",
                   metadataAllocators_.dynamicCellMatch.AvailableSlots());
    WsAllocation dynamicCellMatchAlloc = metadataAllocators_.dynamicCellMatch.Allocate();
    DEV_ASSERT_MSG(WsErr::WORKSPACE_INIT_RESOURCE_ERROR, dynamicCellMatchAlloc.ptr != 0,
                   "Dynamic cell match metadata alloc failed, sizeBytes=%" PRIu64, dynamicCellMatchSlotBytes);
    dynamicCellMatchAlloc.parallelWsId = curParallelWsId;
    partialUpdate->cellMatchRuntimePartialUpdateTable = DevRelocVector<uint64_t>(
        0, reinterpret_cast<uint64_t*>(dynamicCellMatchAlloc.ptr));
    auto& runtimeOutcastTensor = GetRuntimeOutcastTensor(slot.rtOutcastIter);
    runtimeOutcastTensor.dynamicCellMatchAllocation = dynamicCellMatchAlloc;
}

bool DeviceWorkspaceAllocator::IsValidSlotMemRequirement(uint64_t memReq) const
{
    auto& boundaryPool = tensorAllocators_[curParallelWsId].devTaskBoundaryOutcasts;
    auto& innerTemporalPool = tensorAllocators_[curParallelWsId].devTaskInnerTemporalOutcasts;
    return boundaryPool.IsValidSlotMemRequirement(memReq) || innerTemporalPool.IsValidSlotMemRequirement(memReq);
}

void DeviceWorkspaceAllocator::TriggerDelayedRecycle()
{
    for (auto&& outcast : rtBoundaryOutcastToBeFree_) {
        auto& boundaryPool = tensorAllocators_[outcast.allocation.parallelWsId].devTaskBoundaryOutcasts;
        auto& innerTemporalPool = tensorAllocators_[outcast.allocation.parallelWsId].devTaskInnerTemporalOutcasts;
        if (innerTemporalPool.ContainsPtr(outcast.allocation.ptr)) {
            innerTemporalPool.Deallocate(outcast.allocation.ptr);
        } else {
            DEV_ASSERT_MSG(WsErr::WS_TENSOR_ADDRESS_OUT_OF_RANGE, boundaryPool.ContainsPtr(outcast.allocation.ptr),
                           "Boundary outcast pointer is not in boundary or inner temporal pool, ptr=0x%lx",
                           outcast.allocation.ptr);
            boundaryPool.Deallocate(outcast.allocation.ptr);
        }
        if (outcast.dynamicCellMatchAllocation.ptr != 0) {
            metadataAllocators_.dynamicCellMatch.Deallocate(outcast.dynamicCellMatchAllocation.ptr);
        }
    }
    rtBoundaryOutcastToBeFree_.clear();
}

void DeviceWorkspaceAllocator::RecycleDevFuncWorkspace()
{
    tensorAllocators_[curParallelWsId].devTaskInnerExclusiveOutcasts.ResetPool();
    tensorAllocators_[curParallelWsId].rootInner.ResetPool();
}

DevAscendFunctionDupped DeviceWorkspaceAllocator::DuplicateRoot(DevAscendFunction* func)
{
    WsAllocation tinyAlloc = ControlFlowAllocateSlab(
        devProg_, func->GetDuppedDataAllocSize(),
        SlabAlloc(func->GetDuppedDataAllocSize(), WsAicpuSlabMemType::DUPPED_FUNC_DATA));
    return DevAscendFunctionDupped::DuplicateRoot(func, tinyAlloc);
}

void DeviceWorkspaceAllocator::DestroyDuppedFunc(DevAscendFunctionDupped& dup)
{
    dup.ReleaseDuppedMemory(metadataAllocators_.general);
}

DynDeviceTask* DeviceWorkspaceAllocator::MakeDynDeviceTask()
{
    WsAllocation alloc = ControlFlowAllocateSlab(devProg_, sizeof(DynDeviceTask),
                                                 SlabAlloc(sizeof(DynDeviceTask), WsAicpuSlabMemType::DEV_DYN_TASK));
    DynDeviceTask* dynTask = new (reinterpret_cast<void*>(alloc.ptr)) DynDeviceTask(*this);
    dynTask->selfAlloc = alloc;
    return dynTask;
}

DevAscendFunctionDuppedStitch* DeviceWorkspaceAllocator::AllocateStitch()
{
    WsAllocation allocation = ControlFlowAllocateSlab(
        devProg_, sizeof(DevAscendFunctionDuppedStitch),
        SlabAlloc(sizeof(DevAscendFunctionDuppedStitch), WsAicpuSlabMemType::DUPPED_STITCH));
    DevAscendFunctionDuppedStitch* stitch = allocation.As<DevAscendFunctionDuppedStitch>();
    uint64_t* clear = PtrToPtr<DevAscendFunctionDuppedStitch, uint64_t>(stitch);
    clear[0] = 0;
    clear[1] = 0;
    return stitch;
}

DynFuncHeader* DeviceWorkspaceAllocator::AllocateDynFuncData(uint64_t size)
{
    WsAllocation allocation = ControlFlowAllocateSlab(devProg_, size,
                                                      SlabAlloc(size, WsAicpuSlabMemType::DYN_FUNC_DATA));
    DynFuncHeader* header = allocation.As<DynFuncHeader>();
    return header;
}

npu::tile_fwk::PerCorePendingQueue* DeviceWorkspaceAllocator::AllocatePerCorePendingQueue(uint64_t size)
{
    WsAllocation allocation = ControlFlowAllocateSlab(devProg_, size,
                                                      SlabAlloc(size, WsAicpuSlabMemType::PER_CORE_PENDING_QUE));
    return allocation.As<npu::tile_fwk::PerCorePendingQueue>();
}

npu::tile_fwk::DrcoLocalReadyQueue* DeviceWorkspaceAllocator::AllocateDrcoLocalReadyQueue(uint64_t size)
{
    WsAllocation allocation = ControlFlowAllocateSlab(devProg_, size,
                                                      SlabAlloc(size, WsAicpuSlabMemType::LOCAL_READY_QUE));
    return allocation.As<npu::tile_fwk::DrcoLocalReadyQueue>();
}

npu::tile_fwk::DrcoLocalReadyMatrix* DeviceWorkspaceAllocator::AllocateDrcoLocalReadyMatrix(uint64_t size)
{
    WsAllocation allocation = ControlFlowAllocateSlab(devProg_, size,
                                                      SlabAlloc(size, WsAicpuSlabMemType::LOCAL_READY_MATRIX));
    return allocation.As<npu::tile_fwk::DrcoLocalReadyMatrix>();
}

// 共享 stitch 节点矩阵分配（单实例，与 localReadyMatrix 共用 LOCAL_READY_MATRIX 池），
// 数量预算见 CalcStitchWorkspace 的 objUsedNum
npu::tile_fwk::DrcoGlobalStitchNodeMatrix* DeviceWorkspaceAllocator::AllocateDrcoStitchNodeMatrix(uint64_t size)
{
    WsAllocation allocation = ControlFlowAllocateSlab(devProg_, size,
                                                      SlabAlloc(size, WsAicpuSlabMemType::LOCAL_READY_MATRIX));
    return allocation.As<npu::tile_fwk::DrcoGlobalStitchNodeMatrix>();
}

void DeviceWorkspaceAllocator::ResetAicpuMemCounter()
{
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
    metadataAllocators_.general.ResetCounter();
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
}

void DeviceWorkspaceAllocator::RewindMemoryDumper()
{
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
    wsMemDelayedDumper_.Rewind();
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
}

void DeviceWorkspaceAllocator::MarkAsNewStitchWindow()
{
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
    metadataAllocators_.general.DelayedDumpAndResetCounter(wsMemDelayedDumper_);
    aicpuStitchAllocator_.DelayedDumpAndResetCounter(wsMemDelayedDumper_);
    wsMemDelayedDumper_.MarkAsNewStitchWindow();
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
}

void DeviceWorkspaceAllocator::LogTuningSummary() const
{
    const auto& tensorBudget = devProg_->memBudget.tensor;
    uint64_t maxRootInnerDepth = 0;
    uint64_t maxExclusiveDepth = 0;
    uint64_t maxInnerTemporalDepth = 0;
    for (uint32_t i = 0; i < devProg_->GetParallelism(); ++i) {
        const auto& peak = tuningState_.launchUsagePeak[i];
        const auto& allocators = tensorAllocators_[i];
        const uint64_t rootCapacity = allocators.rootInner.Capacity();
        const uint64_t exclusiveCapacity = allocators.devTaskInnerExclusiveOutcasts.Capacity();
        DEV_INFO("[Workspace Runtime Summary] parallelId=%u, "
                 "rootInner={peak=%" PRIu64 ",capacity=%" PRIu64 "}, "
                 "exclusiveOutcast={peak=%" PRIu64 ",capacity=%" PRIu64 "}, "
                 "assembleOutcast={peakSlots=%" PRIu64 ",capacitySlots=%" PRIu64 "}.",
                 i, peak.rootInnerBytes, rootCapacity, peak.exclusiveOutcastBytes, exclusiveCapacity,
                 peak.innerTemporalOutcastSlots, tensorBudget.devTaskInnerTemporalOutcastNum);
        const uint64_t rootInnerDepth = CeilDiv(peak.rootInnerBytes, tensorBudget.workspacePool.rootInnerUnitBytes);
        const uint64_t exclusiveDepth = CeilDiv(peak.exclusiveOutcastBytes,
                                                tensorBudget.workspacePool.exclusiveOutcastUnitBytes);
        const uint64_t innerTemporalDepth = CeilDiv(peak.innerTemporalOutcastSlots,
                                                    tensorBudget.workspacePool.assembleSlotsPerDepth);
        maxExclusiveDepth = exclusiveDepth > maxExclusiveDepth ? exclusiveDepth : maxExclusiveDepth;
        maxRootInnerDepth = rootInnerDepth > maxRootInnerDepth ? rootInnerDepth : maxRootInnerDepth;
        maxInnerTemporalDepth = innerTemporalDepth > maxInnerTemporalDepth ? innerTemporalDepth : maxInnerTemporalDepth;
    }
    const uint64_t slotBytes = tensorBudget.MaxOutcastMem();
    const uint64_t rootInnerDepthBytes = tensorBudget.workspacePool.rootInnerUnitBytes;
    const uint64_t innerTemporalDepthBytes = tensorBudget.workspacePool.assembleSlotsPerDepth * slotBytes;
    const uint64_t exclusiveDepthBytes = tensorBudget.workspacePool.exclusiveOutcastUnitBytes;
    const uint64_t tensorRawPerParallel = tensorBudget.rootInnerSpilledMem +
                                          tensorBudget.devTaskInnerExclusiveOutcasts +
                                          tensorBudget.BoundaryAndInnerTemporalOutcastSlotNum() * slotBytes;
    const uint64_t tensorAlignedPerParallel = AlignUp(tensorRawPerParallel, ALIGNMENT_32K);
    const auto nextDepthTotalBytes = [&](uint64_t depthBytes) {
        return (AlignUp(tensorRawPerParallel + depthBytes, ALIGNMENT_32K) - tensorAlignedPerParallel) *
               tensorBudget.parallelism;
    };
    DEV_INFO("[Workspace Runtime Recommendation] "
             "actualDepths={rootInner=%u,innerTemporal=%u,exclusive=%u}, "
             "recommendedDepths={rootInner=%" PRIu64 ",assembleOutcast=%" PRIu64 ",exclusive=%" PRIu64 "}, "
             "depthMemory={rawPerParallelBytes={rootInner=%" PRIu64 ",innerTemporal=%" PRIu64 ",exclusive=%" PRIu64
             "}, nextDepthTotalBytes={rootInner=%" PRIu64 ",assembleOutcast=%" PRIu64 ",exclusive=%" PRIu64
             "}, parallelism=%u}.",
             tensorBudget.workspacePool.stitchFunctionNumPerPool[0],
             tensorBudget.workspacePool.stitchFunctionNumPerPool[1],
             tensorBudget.workspacePool.stitchFunctionNumPerPool[2],
             AtLeastRequiredDepth(maxRootInnerDepth, tensorBudget.workspacePool.rootInnerUnitBytes),
             RecommendAssembleOutcastDepth(maxInnerTemporalDepth, tensorBudget.workspacePool.assembleSlotsPerDepth),
             AtLeastRequiredDepth(maxExclusiveDepth, tensorBudget.workspacePool.exclusiveOutcastUnitBytes),
             rootInnerDepthBytes, innerTemporalDepthBytes, exclusiveDepthBytes,
             nextDepthTotalBytes(rootInnerDepthBytes), nextDepthTotalBytes(innerTemporalDepthBytes),
             nextDepthTotalBytes(exclusiveDepthBytes), tensorBudget.parallelism);
}

void DeviceWorkspaceAllocator::DumpMemoryUsage(const char* hint) const
{
    (void)hint;
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
    wsMemDelayedDumper_.DumpStitchWindowMemoryUsage();
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL

#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT
    metadataAllocators_.general.DumpMemoryUsage(hint, "Metadata");
    metadataAllocators_.generalSlab.DumpMemoryUsage(hint, "Metadata slab allocator");
    metadataAllocators_.stitchSlab.DumpMemoryUsage(hint, "Metadata Stitch slab allocator");

    for (uint32_t i = 0; i < devProg_->GetParallelism(); i++) {
        DEV_MEM_DUMP("Parallel workspace %u.", i);
        tensorAllocators_[i].rootInner.DumpMemoryUsage(hint, "Tensor (root inner) workspace");
        tensorAllocators_[i].devTaskInnerExclusiveOutcasts.DumpMemoryUsage(
            hint, "Tensor (DeviceTask inner outcasts) workspace");
        tensorAllocators_[i].devTaskBoundaryOutcasts.DumpMemoryUsage(hint);
        tensorAllocators_[i].devTaskInnerTemporalOutcasts.DumpMemoryUsage(hint);
    }

    // Dump stack memory
    DEV_MEM_DUMP("Stack workspace memory usage (%s)\n", hint);
    DEV_MEM_DUMP("            Memory pool size: %10lu bytes\n", stackWorkspaceSize_);
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT
}

void DeviceWorkspaceAllocator::InitMetadataSlabAllocator()
{
    DEV_ASSERT(WsErr::WORKSPACE_CAPACITY_INSUFFICIENT, metadataAllocators_.general.FreeMemorySize() > 0);
    uint64_t memBase = metadataAllocators_.general.MemBaseAddr() + metadataAllocators_.general.AllocatedSize();
    uint64_t realMemBase = AlignUp(memBase, sizeof(uint64_t));
    uint32_t metaSlabMemSize = metadataAllocators_.general.FreeMemorySize() - (realMemBase - memBase);
    uint32_t slabSize = CalcAicpuMetaSlabAlloctorSlabPageSize(metaSlabMemSize);
    metadataAllocators_.generalSlab.Init(reinterpret_cast<void*>(realMemBase), metaSlabMemSize, slabSize,
                                         devProg_->devArgs.archInfo);
    for (size_t i = 0; i < ToUnderlying(WsAicpuSlabMemType::COHERENT_SLAB_MEM_TYPE_BUTT); i++) {
        if ((slabMemObjSizeFunc[i] != nullptr) && ((this->*slabMemObjSizeFunc[i])() != 0)) {
            [[maybe_unused]] bool registCacheRes = metadataAllocators_.generalSlab.RegistCache(
                i, (this->*slabMemObjSizeFunc[i])());
            DEV_ASSERT(WsErr::WORKSPACE_ALLOCATOR_REGIST_FAILED, registCacheRes);
        }
    }
}

uint64_t DeviceWorkspaceAllocator::CalcMetadataItemPoolMemSize(const DevAscendProgram* devProg)
{
    size_t itemBlockSize = sizeof(ItemPool<RuntimeOutcastTensor>::ItemBlock);
    DEV_DEBUG("itemBlockSize=%zu, OutcastPoolSize=%u", itemBlockSize, devProg->memBudget.tensor.runtimeOutcastPoolSize);
    uint64_t itemPoolMemSize = itemBlockSize * devProg->memBudget.tensor.runtimeOutcastPoolSize;
    return itemPoolMemSize;
}

uint64_t DeviceWorkspaceAllocator::CalcMetadataVectorMemSize(const DevAscendProgram* devProg)
{
    // 1. symbolTable
    uint64_t symbolTableCapacity = CalculateVectorCapacity(devProg->symbolTable.size());
    uint64_t symbolTableMemory = symbolTableCapacity * sizeof(int64_t);
    DEV_DEBUG("symbolTableMemory=%lu bytes.", symbolTableMemory);
    // 2. slotList_
    uint64_t slotListCapacity = CalculateVectorCapacity(devProg->slotSize);
    uint64_t slotListMemory = slotListCapacity * sizeof(DeviceExecuteSlot);
    DEV_DEBUG("slotListMemory=%lu bytes.", slotListMemory);
    // 3. rtBoundaryOutcastToBeFree_
    uint64_t boundaryOutcastToFreeListSize = CalculateVectorCapacity(
        devProg->memBudget.tensor.BoundaryAndInnerTemporalOutcastSlotNum());
    uint64_t boundaryOutcastToFreeMemory = boundaryOutcastToFreeListSize * sizeof(RuntimeOutcastTensor);
    DEV_DEBUG("boundaryOutcastToFreeMemory=%lu bytes.", boundaryOutcastToFreeMemory);
    // total
    uint64_t totalSetupVectorMemory = symbolTableMemory + slotListMemory + boundaryOutcastToFreeMemory;
    return totalSetupVectorMemory;
}

uint64_t DeviceWorkspaceAllocator::CalcMetadataSlotAllocatorMemSize(const DevAscendProgram* devProg)
{
    size_t blockHeaderSize = sizeof(WsSlotAllocator::BlockHeader);
    uint64_t boundaryOutcastSlotNum = devProg->memBudget.tensor.BoundaryAndInnerTemporalOutcastSlotNum();
    uint64_t dynamicCellMatchSlotNum = devProg->memBudget.metadata.dynamicCellMatchSlotNum;
    DEV_DEBUG("boundaryOutcastSlotNum=%lu, dynamicCellMatchSlotNum=%lu", boundaryOutcastSlotNum,
              dynamicCellMatchSlotNum);
    uint64_t blockHeadersBytes = (boundaryOutcastSlotNum + dynamicCellMatchSlotNum) * blockHeaderSize;
    return blockHeadersBytes;
}

uint32_t DeviceWorkspaceAllocator::CalcSlabMemObjmaxSize()
{
    uint32_t slabMemObjmaxSize = CalcAicpuMetaSlabAlloctorSlabMemObjmaxSize();
    DEV_DEBUG("[workspaceSize] slabMemObjmaxSize=%u", slabMemObjmaxSize);
    return slabMemObjmaxSize;
}

uint32_t DeviceWorkspaceAllocator::CalcStitchSlabMemObjmaxSize(uint32_t* slabCapacity)
{
    uint32_t slabMemObjmaxSize = 0;
    size_t start = ToUnderlying(WsAicpuSlabMemType::READY_QUE);
    size_t end = ToUnderlying(WsAicpuSlabMemType::DUPPED_STITCH);
    for (size_t i = start; i < end; ++i) {
        if (slabMemObjSizeFunc[i] != nullptr) {
            uint32_t currentSize = (this->*slabMemObjSizeFunc[i])();
            if (currentSize > slabMemObjmaxSize) {
                slabMemObjmaxSize = currentSize;
            }
        }
    }
    slabMemObjmaxSize = SlabWsAllocator::CalcSlotStride(slabMemObjmaxSize);
    slabMemObjmaxSize = AlignUpSlab(slabMemObjmaxSize * ALLOC_NUM_ONE_SLAB + SlabWsAllocator::FirstObjBaseOffset());
    slabMemObjmaxSize = std::max(slabMemObjmaxSize, (uint32_t)MEBI);
    DEV_INFO("Stitch slab size=%u bytes", slabMemObjmaxSize);
    devProg_->memBudget.metadata.stitchSlabSize = slabMemObjmaxSize;
    size_t j = 0;
    for (size_t i = start; i < end; ++i) {
        if (slabMemObjSizeFunc[i] == nullptr) {
            continue;
        }
        uint32_t currentSize = (this->*slabMemObjSizeFunc[i])();
        if (currentSize == 0) {
            slabCapacity[j] = 0;
        } else {
            slabCapacity[j] = SlabWsAllocator::CalcObjsPerSlab(slabMemObjmaxSize, currentSize);
        }
        ++j;
    }
    return slabMemObjmaxSize;
}

void DeviceWorkspaceAllocator::CalculateSlabCapacityPerType(uint32_t slabSize, uint32_t* slabCapacity,
                                                            uint32_t slabTypeNum)
{
    if (slabCapacity == nullptr) {
        DEV_ERROR(WsErr::WORKSPACE_INIT_PARAM_INVALID, "#workspace.init.resource: slabCapacity is nullptr");
        return;
    }
    constexpr uint32_t maxSlabTypes = ToUnderlying(WsAicpuSlabMemType::COHERENT_SLAB_MEM_TYPE_BUTT);
    if (slabTypeNum > maxSlabTypes) {
        DEV_ERROR(WsErr::SLAB_TYPE_INVALID, "#workspace.init.check: slabTypeNum exceeds the allowed maxSlabTypes=%u",
                  maxSlabTypes);
        return;
    }
    for (size_t i = 0; i < slabTypeNum; ++i) {
        if (slabMemObjSizeFunc[i] != nullptr && (this->*slabMemObjSizeFunc[i])() != 0) {
            DEV_DEBUG("WsAicpuSlabMemType[%zu]=%u", i, (this->*slabMemObjSizeFunc[i])());
            slabCapacity[i] = SlabWsAllocator::CalcObjsPerSlab(slabSize, (this->*slabMemObjSizeFunc[i])());
        }
    }
}

WsAllocation DeviceWorkspaceAllocator::SlabAlloc(uint32_t objSize, WsAicpuSlabMemType type)
{
    void* ptr = nullptr;
    DEV_VERBOSE_DEBUG("SlabAlloc type = %u, size = %u.", ToUnderlying(type), objSize);
    SlabTryDynAddCache(type, objSize);

    TIMEOUT_CHECK_INIT(devProg_->devArgs.archInfo, TIMEOUT_20MIN);
    uint64_t inner_start = GetCycles();

    do {
        if (type < WsAicpuSlabMemType::COHERENT_SLAB_MEM_TYPE_BUTT) {
            ptr = metadataAllocators_.generalSlab.Alloc(ToUnderlying(type));
        } else if (type < WsAicpuSlabMemType::SLAB_MEM_TYPE_BUTT) {
            ptr = metadataAllocators_.stitchSlab.Alloc(ToUnderlying(type));
        }
        if (ptr != nullptr) {
            break;
        }

        if (submmitTaskQueue_.IsEmpty()) {
            metadataAllocators_.generalSlab.DumpMemoryStatusWhenAbnormal("SlabAlloc null");
            metadataAllocators_.stitchSlab.DumpMemoryStatusWhenAbnormal("SlabAlloc null");
            DEV_ERROR(WsErr::SLAB_ADD_CACHE_FAILED, "#workspace.init.check: Slab alloc null,type=%u,objsize=%u.",
                      ToUnderlying(type), objSize);
            DEV_ASSERT_MSG(WsErr::SLAB_ADD_CACHE_FAILED, false, "Slab alloc null,type=%u,objsize=%u.",
                           ToUnderlying(type), objSize);
        }

        while (!DeviceTaskMemTryRecycle()) {
            if ((GetCycles() - inner_start) > timeout_cycles) {
                DEV_ERROR(WsErr::SLAB_ADD_CACHE_FAILED,
                          "#workspace.alloc.inner: Inner recycle timeout, type=%u, objSize=%u.", ToUnderlying(type),
                          objSize);
                break;
            }
            if ((GetCycles() - inner_start) % warn_interval == 0) {
                DEV_WARN("#workspace.alloc.inner: Inner recycle still waiting, type=%u, objSize=%u.",
                         ToUnderlying(type), objSize);
            }
        };

        __PYPTO_TIMEOUT_CHECK(
            WsErr::SLAB_ADD_CACHE_FAILED,
            {
                WsAllocation emptyAlloc;
                emptyAlloc.ptr = 0;
                return emptyAlloc;
            },
            "#workspace.alloc: SlabAlloc, type=%u, objSize=%u.", ToUnderlying(type), objSize);
    } while (true);

    WsAllocation allocation;
    allocation.ptr = reinterpret_cast<uintdevptr_t>(ptr);
    return allocation;
}

WsSlabStageAllocMem DeviceWorkspaceAllocator::SlabGetStageAllocMem(bool keepTail, WsAicpuSlabMemType keepType)
{
    WsSlabStageAllocMem stageMem;
    stageMem.generalMetadataStageMem = metadataAllocators_.generalSlab.PopStageAllocMem(keepTail,
                                                                                        ToUnderlying(keepType));
    stageMem.stitchStageMem = metadataAllocators_.stitchSlab.PopStageAllocMem(false,
                                                                              0); // not support keep alloc memory
    return stageMem;
}

void DeviceWorkspaceAllocator::DumpSlabUsageBeforeSubmit(uint32_t taskId, DynDeviceTask* devTask)
{
    DEV_VERBOSE_DEBUG("[workspace.slab.usage] before submit devTask, taskId=%u, devTask=%p. ", taskId, devTask);
    metadataAllocators_.generalSlab.DumpSlabUsage("General metadata slab");
    metadataAllocators_.stitchSlab.DumpSlabUsage("Stitch pool slab");
}

void DeviceWorkspaceAllocator::SlabStageAllocMemSubmmit(DynDeviceTask* devTask)
{
    TIMEOUT_CHECK_INIT(devProg_->devArgs.archInfo, TIMEOUT_20MIN);
    while (!submmitTaskQueue_.TryEnqueue(devTask)) {
        DeviceTaskMemTryRecycle();

        __PYPTO_TIMEOUT_CHECK(WsErr::SLAB_ADD_CACHE_FAILED, return, "#workspace.submit: SlabStageAllocMemSubmit.");
    }
    return;
}

void DeviceWorkspaceAllocator::InitMetadataAllocators(DevAscendProgram* devProg, DevStartArgs* devStartArgs)
{
    // Initialize aicpu memory
    uint64_t generalAddr = devStartArgs->deviceRuntimeDataDesc.generalAddr;
    metadataAllocators_.general.InitMetadataAllocator(generalAddr, devProg->memBudget.metadata.general);
    DEV_TRACE_DEBUG(CtrlEvent(
        none(), WorkspaceMetadataGeneral(Range(generalAddr, generalAddr + devProg->memBudget.metadata.general))));

    uint64_t stitchPoolAddr = devStartArgs->deviceRuntimeDataDesc.stitchPoolAddr;
    // stitch 池预算 < 4GB 仅 AICore 解依赖（DRCO）需要：池内对象以 u32 偏移被引用
    // （DrcoGlobalStitchNodeMatrix 槽位），且下方传参存在 u64 -> u32 截断；
    if (devProg->devArgs.enableAicoreResolve) {
        DEV_ASSERT_MSG(WsErr::WORKSPACE_INIT_PARAM_INVALID, devProg->memBudget.metadata.stitchPool <= UINT32_MAX,
                       "Stitch pool size %lu exceeds u32 limit", devProg->memBudget.metadata.stitchPool);
    }
    InitAicpuStitchSlabAllocator(reinterpret_cast<void*>(stitchPoolAddr), devProg->memBudget.metadata.stitchPool);
    DEV_TRACE_DEBUG(CtrlEvent(none(), WorkspaceMetadataStitch(Range(
                                          stitchPoolAddr, stitchPoolAddr + devProg->memBudget.metadata.stitchPool))));

    uint64_t dynamicCellMatchSlotNum = devProg->memBudget.metadata.dynamicCellMatchSlotNum;
    uint64_t dynamicCellMatchSlotBytes = devProg->memBudget.metadata.maxDynamicCellMatchTableMem;
    uint64_t dynamicCellMatchAddr = devStartArgs->deviceRuntimeDataDesc.dynamicCellMatchAddr;
    uint64_t dynamicCellMatchBytes = devProg->memBudget.metadata.dynamicCellMatch;
    if (dynamicCellMatchSlotNum == 0 || dynamicCellMatchSlotBytes == 0 || dynamicCellMatchBytes == 0) {
        return;
    }
    DEV_ASSERT_MSG(WsErr::WORKSPACE_INIT_PARAM_INVALID, dynamicCellMatchAddr != 0,
                   "Dynamic cell match addr is null while bytes=%lu", dynamicCellMatchBytes);
    DEV_ASSERT_MSG(WsErr::WORKSPACE_INIT_PARAM_INVALID,
                   dynamicCellMatchBytes == dynamicCellMatchSlotNum * dynamicCellMatchSlotBytes,
                   "Dynamic cell match pool bytes mismatch, budget=%lu, calc=%lu", dynamicCellMatchBytes,
                   dynamicCellMatchSlotNum * dynamicCellMatchSlotBytes);
    WsAllocation dynamicCellMatchBase(dynamicCellMatchAddr, dynamicCellMatchBytes);
    metadataAllocators_.dynamicCellMatch.InitTensorAllocator(dynamicCellMatchBase.ptr, dynamicCellMatchSlotNum,
                                                             dynamicCellMatchSlotBytes, metadataAllocators_.general);
}

void DeviceWorkspaceAllocator::InitTensorAllocators(uintdevptr_t workspaceAddr, uint64_t tensorWorkspaceSize,
                                                    DevAscendProgram* devProg)
{
    uint64_t baseAddr = workspaceAddr;

    // Initialize tensor workspace memory verifier
    tensorWsVerifier_.Init(baseAddr, tensorWorkspaceSize);

    uint32_t paallelism = devProg->GetParallelism();

    // Initialize root function slotted outcast tensor memory (boundary pool then inner temporal pool)
    uint64_t boundarySlotNum = devProg->memBudget.tensor.devTaskBoundaryOutcastNum;
    uint64_t innerTemporalSlotNum = devProg->memBudget.tensor.devTaskInnerTemporalOutcastNum;
    uint64_t slotMem = devProg->memBudget.tensor.MaxOutcastMem();
    uint64_t boundaryOutcastBudgetPerParallel = boundarySlotNum * slotMem;
    uint64_t innerTemporalOutcastBudgetPerParallel = innerTemporalSlotNum * slotMem;
    slotVerifier_.Init(baseAddr,
                       paallelism * (boundaryOutcastBudgetPerParallel + innerTemporalOutcastBudgetPerParallel));
    for (uint32_t parallelIdx = 0; parallelIdx < paallelism; parallelIdx++) {
        auto& boundaryPool = tensorAllocators_[parallelIdx].devTaskBoundaryOutcasts;
        auto& innerTemporalPool = tensorAllocators_[parallelIdx].devTaskInnerTemporalOutcasts;
        boundaryPool.InitTensorAllocator(baseAddr, boundarySlotNum, slotMem, metadataAllocators_.general);
        DEV_TRACE_DEBUG(CtrlEvent(
            none(), WorkspaceCrossDeviceTaskOutcast(Range(baseAddr, baseAddr + boundaryOutcastBudgetPerParallel))));
        baseAddr += boundaryOutcastBudgetPerParallel;
        innerTemporalPool.InitTensorAllocator(baseAddr, innerTemporalSlotNum, slotMem, metadataAllocators_.general);
        DEV_TRACE_DEBUG(CtrlEvent(none(), WorkspaceCrossDeviceTaskOutcast(
                                              Range(baseAddr, baseAddr + innerTemporalOutcastBudgetPerParallel))));
        baseAddr += innerTemporalOutcastBudgetPerParallel;
    }

    // Initialize root function non-outcast tensor memory
    auto rootInnerBudget = devProg->memBudget.tensor.rootInnerSpilledMem;
    rootInnerWsVerifier_.Init(baseAddr, paallelism * rootInnerBudget);
    for (uint32_t parallelIdx = 0; parallelIdx < paallelism; parallelIdx++) {
        tensorAllocators_[parallelIdx].rootInner.InitTensorAllocator(baseAddr, rootInnerBudget);
        DEV_TRACE_DEBUG(CtrlEvent(none(), WorkspaceInnerTensor(Range(baseAddr, baseAddr + rootInnerBudget))));
        baseAddr += rootInnerBudget;
    }

    // Initialize root function sequential outcast tensor memory
    auto devTaskInnerOutcastBudget = devProg->memBudget.tensor.devTaskInnerExclusiveOutcasts;
    devTaskInnerExclusiveOutcastsWsVerifier_.Init(baseAddr, paallelism * devTaskInnerOutcastBudget);
    for (uint32_t parallelIdx = 0; parallelIdx < paallelism; parallelIdx++) {
        tensorAllocators_[parallelIdx].devTaskInnerExclusiveOutcasts.InitTensorAllocator(baseAddr,
                                                                                         devTaskInnerOutcastBudget);
        DEV_TRACE_DEBUG(
            CtrlEvent(none(), WorkspaceInDeviceTaskOutcast(Range(baseAddr, baseAddr + devTaskInnerOutcastBudget))));
        baseAddr += devTaskInnerOutcastBudget;
    }

    DEV_ASSERT(WsErr::WORKSPACE_BASE_ADDR_OUT_OF_RANGE,
               workspaceAddr <= baseAddr && baseAddr <= workspaceAddr + tensorWorkspaceSize);
}

void DeviceWorkspaceAllocator::InitAICoreSpilledMemory(uintdevptr_t workspaceAddr, DevAscendProgram* devProg)
{
    uint64_t coreNum = devProg->devArgs.GetBlockNum();
    if (coreNum == 0) {
        return;
    }
    // Compile time `aicoreSpilled` per single core is required to be aligned by 512.
    // This formula will never result into a value smaller than compile time one.
    uint64_t perCoreMem = devProg->memBudget.aicoreSpilled.Total() / TENSOR_ADDR_ALIGNMENT / coreNum *
                          TENSOR_ADDR_ALIGNMENT;

    // Initialize in-core stack memory
    stackWorkspaceBase_ = workspaceAddr;
    standardStackWorkspacePerCore_ = perCoreMem;
    stackWorkspaceSize_ = devProg->memBudget.aicoreSpilled.Total();
    DEV_TRACE_DEBUG(
        CtrlEvent(none(), WorkspaceSpill(mem(perCoreMem), coreNum,
                                         Range(stackWorkspaceBase_, stackWorkspaceBase_ + stackWorkspaceSize_))));
}

uint32_t DeviceWorkspaceAllocator::DevFunctionDuppedSlabMemObjSize()
{
    // Allocator is stack-local per launch, so maxDevFuncDuppedSize_ resets to 0.
    // DevProg is shared across launches on this Ctrl thread; cache the scan there.
    if (maxDevFuncDuppedSize_ == 0) {
        static DevAscendProgram* cachedProg = nullptr;
        static uint32_t cachedSize = 0;
        if (cachedProg == devProg_ && cachedSize != 0) {
            maxDevFuncDuppedSize_ = cachedSize;
            return maxDevFuncDuppedSize_;
        }
        for (uint32_t i = 0; i < devProg_->GetFunctionSize(); i++) {
            uint64_t curSize = devProg_->GetFunction(i)->GetDuppedDataAllocSize();
            if (curSize > maxDevFuncDuppedSize_) {
                maxDevFuncDuppedSize_ = curSize;
            }
        }
        cachedProg = devProg_;
        cachedSize = maxDevFuncDuppedSize_;
    }

    return maxDevFuncDuppedSize_;
}

/*计算使用vector的元数据的数据结构大小*/
uint64_t DeviceWorkspaceAllocator::CalculateVectorCapacity(uint64_t size)
{
    if (size == 0) {
        return 0;
    }
    constexpr uint64_t MIN_CAPACITY = 8;
    uint64_t capacity = std::max(MIN_CAPACITY, size);
    // 向上取整到 2 的幂次
    capacity = (capacity == 0) ? 0 : (1ULL << (64 - __builtin_clzll(capacity - 1)));
    return capacity;
}

/* 按照devicetask最大支持stitch阈值分配对象 */
uint32_t DeviceWorkspaceAllocator::DynFuncDataSlabMemObjSize()
{
    return (sizeof(DynFuncHeader) + MAX_STITCH_FUNC_NUM * sizeof(DynFuncData));
}

/* 按照devicetask最大支持stitch阈值分配对象 */
uint32_t DeviceWorkspaceAllocator::VecStitchListSLabMemObjSize()
{
    return MAX_STITCH_FUNC_NUM * sizeof(DevAscendFunctionDupped);
}

uint32_t DeviceWorkspaceAllocator::DynDevTaskSlabMemObjSize() { return sizeof(struct DynDeviceTask); }

uint32_t DeviceWorkspaceAllocator::ShmemWaitUntilCacheSlabMemObjSize()
{
    if (devProg_->devArgs.hasAicpuTask) {
        return sizeof(npu::tile_fwk::Distributed::ShmemWaitUntilCache);
    }
    return 0;
}

uint32_t DeviceWorkspaceAllocator::DuppedStitchSlabMemObjSize() { return sizeof(DevAscendFunctionDuppedStitch); }

uint32_t DeviceWorkspaceAllocator::ReadyQueSlabMemObjSize()
{
    return sizeof(ReadyCoreFunctionQueue) + devProg_->stitchFunctionsize * sizeof(uint32_t);
}

uint32_t DeviceWorkspaceAllocator::DieReadyQueSlabMemObjSize()
{
    if (devProg_->devArgs.archInfo == ArchInfo::DAV_3510) {
        return sizeof(ReadyCoreFunctionQueue) + devProg_->stitchFunctionsize * sizeof(uint32_t);
    } else {
        return 0;
    }
}

uint32_t DeviceWorkspaceAllocator::WrapQueSlabMemObjSize()
{
    if (devProg_->devArgs.archInfo == ArchInfo::DAV_3510) {
        return sizeof(WrapInfoQueue) + devProg_->stitchFunctionsize * sizeof(WrapInfo);
    } else {
        return 0;
    }
}

uint32_t DeviceWorkspaceAllocator::PerCorePendingQueSlabMemObjSize()
{
    return sizeof(npu::tile_fwk::PerCorePendingQueue) + devProg_->stitchFunctionsize * sizeof(uint32_t);
}

uint32_t DeviceWorkspaceAllocator::LocalReadyQueSlabMemObjSize()
{
    return sizeof(npu::tile_fwk::DrcoLocalReadyQueue) + devProg_->stitchFunctionsize * sizeof(uint32_t);
}

uint32_t DeviceWorkspaceAllocator::LocalReadyMatrixSlabMemObjSize()
{
    // LOCAL_READY_MATRIX 池承载 local ready matrix（N×N 槽位数组，随 LOCAL_GROUP_SIZE 增大）
    // 与全局共享 stitch 节点矩阵（108 行 × 64B cacheline 对齐行），注册 objSize 取两者最大值
    uint32_t localMatrixSize = sizeof(npu::tile_fwk::DrcoLocalReadyMatrix);
    uint32_t stitchMatrixSize = sizeof(npu::tile_fwk::DrcoGlobalStitchNodeMatrix);
    return std::max(localMatrixSize, stitchMatrixSize);
}

uint32_t DeviceWorkspaceAllocator::PredCountSlabMemObjSize()
{
    // predCount
    return devProg_->rootFuncMaxCallOpsize * sizeof(int32_t);
}

/* 根据当前算子的业务模型分析计算出slab 管理内存页大小, 基于当前可评估的所有内存类型的最大值评估 */
uint32_t DeviceWorkspaceAllocator::CalcAicpuMetaSlabAlloctorSlabMemObjmaxSize()
{
    uint32_t slabMemObjmaxSize = 0;
    for (size_t i = 0; i < ToUnderlying(WsAicpuSlabMemType::COHERENT_SLAB_MEM_TYPE_BUTT); ++i) {
        if (slabMemObjSizeFunc[i] != nullptr) {
            uint32_t currentSize = (this->*slabMemObjSizeFunc[i])();
            if (currentSize > slabMemObjmaxSize) {
                slabMemObjmaxSize = currentSize;
            }
        }
    }
    slabMemObjmaxSize = SlabWsAllocator::CalcSlotStride(slabMemObjmaxSize);
    devProg_->memBudget.metadata.generalSlabSize = slabMemObjmaxSize;
    DEV_INFO("General slab size=%u bytes", slabMemObjmaxSize);
    return slabMemObjmaxSize;
}

uint32_t DeviceWorkspaceAllocator::CalcAicpuMetaSlabAlloctorSlabPageSize(uint32_t totalMemSize)
{
    uint32_t allocNumOneSlab = 4; // default
    uint32_t slabSize = devProg_->memBudget.metadata.generalSlabSize;
    uint32_t leastSlabReqMem = (ToUnderlying(WsAicpuSlabMemType::COHERENT_SLAB_MEM_TYPE_BUTT)) * slabSize;
    DEV_ASSERT_MSG(WsErr::WORKSPACE_CAPACITY_INSUFFICIENT, leastSlabReqMem < totalMemSize,
                   "leastSlabReqMem=%u >= totalMemSize=%u", leastSlabReqMem, totalMemSize);
    uint32_t realMaxAllocNum = totalMemSize / leastSlabReqMem;
    if (realMaxAllocNum < allocNumOneSlab) {
        allocNumOneSlab = realMaxAllocNum;
    }
    return AlignUpSlab(slabSize * allocNumOneSlab + SlabWsAllocator::FirstObjBaseOffset());
}

void DeviceWorkspaceAllocator::InitAicpuStitchSlabAllocator(void* memBase, uint32_t totalSize)
{
    DEV_ASSERT_MSG(WsErr::WORKSPACE_INIT_PARAM_INVALID, memBase != nullptr && totalSize > 0,
                   "memBase %s null, totalSize=%u", memBase == nullptr ? "is" : "is not", totalSize);
    uint32_t slabSize = devProg_->memBudget.metadata.stitchSlabSize;
    metadataAllocators_.stitchSlab.Init(memBase, totalSize, slabSize, devProg_->devArgs.archInfo);
    for (size_t i = ToUnderlying(WsAicpuSlabMemType::COHERENT_SLAB_MEM_TYPE_BUTT) + 1;
         i < ToUnderlying(WsAicpuSlabMemType::SLAB_MEM_TYPE_BUTT); ++i) {
        if ((slabMemObjSizeFunc[i] != nullptr) && ((this->*slabMemObjSizeFunc[i])() != 0)) {
            uint32_t objSize = (this->*slabMemObjSizeFunc[i])();
            DEV_ASSERT_MSG(WsErr::WORKSPACE_CAPACITY_INSUFFICIENT, slabSize > objSize, "slabSize=%u <= objSize=%u",
                           slabSize, objSize);
            [[maybe_unused]] bool registCacheRes = metadataAllocators_.stitchSlab.RegistCache(i, objSize);
            DEV_ASSERT(WsErr::WORKSPACE_ALLOCATOR_REGIST_FAILED, registCacheRes);
        }
    }
}

void DeviceWorkspaceAllocator::SlabTryDynAddCache(WsAicpuSlabMemType type, uint32_t objSize)
{
    if (type < WsAicpuSlabMemType::COHERENT_SLAB_MEM_TYPE_BUTT) {
        if (!metadataAllocators_.generalSlab.ExistCache(ToUnderlying(type), objSize)) {
            [[maybe_unused]] bool registCacheRes = metadataAllocators_.generalSlab.RegistCache(ToUnderlying(type),
                                                                                               objSize);
            DEV_ASSERT(WsErr::WORKSPACE_ALLOCATOR_REGIST_FAILED, registCacheRes);
        }
    } else if (type < WsAicpuSlabMemType::SLAB_MEM_TYPE_BUTT) {
        if (!metadataAllocators_.stitchSlab.ExistCache(ToUnderlying(type), objSize)) {
            [[maybe_unused]] bool registCacheRes = metadataAllocators_.stitchSlab.RegistCache(ToUnderlying(type),
                                                                                              objSize);
            DEV_ASSERT(WsErr::WORKSPACE_ALLOCATOR_REGIST_FAILED, registCacheRes);
        }
    } else {
        DEV_ERROR(WsErr::SLAB_TYPE_INVALID, "#workspace.init.check: Invalid slab memory type: %u", (unsigned int)type);
        DEV_ASSERT(WsErr::SLAB_TYPE_INVALID, false);
    }
}

bool DeviceWorkspaceAllocator::DeviceTaskMemTryRecycle()
{
    auto FreeTaskSlabMemfunc = [this](DynDeviceTask* deviceTask, bool& continueNext) -> bool {
        if (deviceTask == nullptr) {
            continueNext = true;
            return true;
        }

        if (!deviceTask->taskStageAllocMem.canFree.load(std::memory_order_relaxed)) {
            auto* dynFuncHeader = deviceTask->dynFuncDataList;
            if (dynFuncHeader != nullptr) {
                auto* rootFuncList = deviceTask->drcoRootFuncList;
                if (rootFuncList != nullptr) {
                    uint32_t finished = __atomic_load_n(&rootFuncList->devTaskFinished, __ATOMIC_ACQUIRE);
                    if (finished == 1 || rootFuncList->totalTaskCount == 0) {
                        deviceTask->taskStageAllocMem.canFree.store(true, std::memory_order_release);
                    }
                }
            }
        }

        if (deviceTask->taskStageAllocMem.canFree.load(std::memory_order_relaxed)) {
            if (deviceTask->taskStageAllocMem.ctrlNotFreeFlag != nullptr) {
                deviceTask->taskStageAllocMem.ctrlNotFreeFlag->store(false, std::memory_order_release);
            }
            // recycle slab alloc memory
            metadataAllocators_.generalSlab.FreeStageAllocMem(deviceTask->taskStageAllocMem.generalMetadataStageMem);
            metadataAllocators_.stitchSlab.FreeStageAllocMem(deviceTask->taskStageAllocMem.stitchStageMem);
            continueNext = true;
            return true;
        }

        // parallel device task need continue check next
        // devtask(iter1)(canFree = true), devtask(iter1)(canFree = false), ... devtask(iter2)(canFree = true),
        // devtask(iter2)
        continueNext = deviceTask->SupportParallel();
        return false;
    };

    // try free finished task and recycle aicpu meta memory
    return submmitTaskQueue_.FreeUntil(FreeTaskSlabMemfunc);
}

} // namespace npu::tile_fwk::dynamic
