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
 * \file device_execute_context.cpp
 * \brief
 */

#include "machine/device/dynamic/context/device_execute_context.h"
#include "machine/device/distributed/shmem_wait_until.h"
#include "tilefwk/aicpu_runtime.h"
#include "tileop/distributed/comm_context.h"

#include <cinttypes>

namespace npu::tile_fwk::dynamic {
bool DeviceExecuteContext::DuppedRootCached()
{
    if (!controlFlowCacheActivated) {
        return false;
    }
    return duppedRootCount < devProg->ctrlFlowCacheAnchor->rootTaskCount;
}

bool DeviceExecuteContext::DuppedRootUpdateAndCachedAllSubmitted()
{
    if (!controlFlowCacheActivated) {
        return false;
    }
    duppedRootCount++;
    return duppedRootCount == devProg->ctrlFlowCacheAnchor->rootTaskCount;
}

int DeviceExecuteContext::RunInit(DevStartArgs* startArgs, PushTaskEntry tPushTask)
{
    PerfBegin(PERF_EVT_CONTROL_FLOW_INIT);
    this->pushTask = tPushTask;
    this->args = startArgs;
    this->devProg = startArgs->devProg;
    parallelCtx.InitParallel(devProg->GetParallelism());
    workspace.Init(startArgs);
    stitchTaskLoopNumThreshold = devProg->stitchMaxFunctionNum;

    slotContext.InitAllocator(workspace, devProg->slotSize);
    slotContext.FillInputOutputSlot(devProg, startArgs);

    stitchContext.Init(devProg, workspace);

    taskContext.InitAllocator(devProg, workspace, startArgs);

    workspace.SetupVector(symbolTable);
    symbolTable.resize(devProg->symbolTable.size());
    for (int index = 0; index < startArgs->GetInputSymbolSize(); ++index) {
        DevInputSymbol& param = startArgs->GetInputSymbol(index);
        int inputSymbolIndex = this->devProg->startArgsInputSymbolIndexList[index];
        symbolTable[inputSymbolIndex] = param.value;
        DEV_INFO("Param %d Symbol Table %d = %ld.", index, inputSymbolIndex, param.value);
    }

    workspace.AllocateStitchCache();

    /* This initialization must only occur after all other AICPU workspace meta memory allocations have completed.
        The remaining portion of AICPU workspace meta memory must support reclamation. */
    workspace.InitMetadataSlabAllocator();

    PerfEnd(PERF_EVT_CONTROL_FLOW_INIT);
    DEV_INFO("Image size is %lu bytes.", devProg->GetSize());
    return DEVICE_MACHINE_OK;
}

DeviceExecuteContext::DeviceExecuteContext(DevStartArgs* startArgs)
{
    PerfBegin(PERF_EVT_INIT);
    this->devProg = startArgs->devProg;

    DEV_IF_VERBOSE_DEBUG
    {
        std::string dump = devProg->Dump(0, true);
        DEV_VERBOSE_DEBUG_SPLIT("[DEVICE] %s.", dump.c_str());
    }

    PerfBegin(PERF_EVT_CONTROL_FLOW_MAPEXE);
    // Prefer injected entry (device custom ctrl / host UT stubs).
    // Production already resolves AOT CF in InitDevProgram and sets controlFlowEntry before constructing.
    // Do not fall back to AOT here when entry is unset: DevStartArgs.devProg may be null/uninit in host UT.
    if (startArgs->controlFlowEntry != nullptr && devProg != nullptr) {
        execProg = DeviceExecuteProgram(devProg, reinterpret_cast<AOTBinaryControlFlow::controlFlowEntry>(
                                                     const_cast<void*>(startArgs->controlFlowEntry)));
    }
    PerfEnd(PERF_EVT_CONTROL_FLOW_MAPEXE);
    PerfEnd(PERF_EVT_INIT);
}

void DeviceExecuteContext::PushTask(DynDeviceTask* dynTask)
{
    DEV_IF_VERBOSE_DEBUG { taskContext.TraceFirstBatchResolve(dynTask); }
    pushTask(dynTask, this);
    taskId++;
    // Stitch-cache entry packs only taskId[15:0]; bump epoch on wrap so reused low bits cannot
    // collide with earlier tasks in the same launch (see CheckStitchCacheDuplicate).
    if (unlikely((taskId & 0xFFFFULL) == 0)) {
        workspace.AdvanceStitchCacheEpoch();
    }
    AdvanceCellMatchTagSeq(devProg);
}

void DeviceExecuteContext::ShowStats()
{
    taskContext.ShowStats();
    workspace.DumpMemoryUsage("End ExecDyn");
}

void DeviceExecuteContext::GELaunchRunCached(DevStartArgs* startArgs, PushTaskEntry tPushTask)
{
    PerfBegin(PERF_EVT_CONTROL_FLOW_INIT);
    this->pushTask = tPushTask;
    this->args = startArgs;
    this->devProg = startArgs->devProg;
    PerfEnd(PERF_EVT_CONTROL_FLOW_INIT);
    PerfBegin(PERF_EVT_CONTROL_FLOW);
    const bool fullCache = devProg->ctrlFlowCacheAnchor->IsActivatedFullCache(startArgs);
    for (size_t index = 0; index < devProg->ctrlFlowCacheAnchor->deviceTaskCount; index++) {
        DynDeviceTask* dynTask = reinterpret_cast<DynDeviceTask*>(
            devProg->ctrlFlowCacheAnchor->deviceTaskCacheList[index].dynTaskBase);
        devProg->ctrlFlowCacheAnchor->RelocIncastOutcastTask(static_cast<uint32_t>(index), 0,
                                                             startArgs->contextWorkspaceAddr, startArgs);
        devProg->ctrlFlowCacheAnchor->PredCountPingPongSwap(dynTask);
        if (startArgs->devProg->devArgs.enableAicoreResolve) {
            devProg->ctrlFlowCacheAnchor->DrcoPredCountDataRestore(dynTask);
        }
        // The Bitmap remains unchanged in the sche.
        // For the full cache scenario, BitmapDataRestoreTask only needs to be executed once.
        if (!fullCache || !devProg->ctrlFlowCacheAnchor->bitmapRestoreDone) {
            devProg->ctrlFlowCacheAnchor->BitmapDataRestoreTask(dynTask);
        }
        devProg->ctrlFlowCacheAnchor->ReadyQueueDataPingPongSwap(dynTask, startArgs->nrValidAic);
        devProg->ctrlFlowCacheAnchor->DieReadyQueuePingPongSwap(dynTask, startArgs->nrValidAic);
        if (dynTask->drcoRootFuncList != nullptr) {
            npu::tile_fwk::DrcoRootFuncListRestorePrecount(dynTask->drcoRootFuncList, dynTask->drcoPrecountExecuted,
                                                           dynTask->drcoPrecountFinishFlag);
        }
        devProg->ctrlFlowCacheAnchor->MixTaskDataRestore(dynTask);
        taskContext.UpdateReadyTaskNum(dynTask->readyQueueBackup->readyTaskNum);

        // dynamic devtask building need inherit cached last dev task parallinfo
        parallelCtx.info = dynTask->parallelInfo;
        if (parallelCtx.info.forId != 0) {
            parallelCtx.isInParallelForScope = true;
        }

        PROF_STAGE_BEGIN(PERF_EVT_STAGE_PUSH_TASK, "push.before\n");
        DumpDeviceTask(taskId, dynTask);
        PerfMtTrace(PERF_TRACE_DEV_TASK_BUILD, CTRL_CPU_THREAD_IDX);
        PushTask(dynTask);
        PROF_STAGE_END(PERF_EVT_STAGE_PUSH_TASK, "push.after\n");
    }
    PerfEnd(PERF_EVT_CONTROL_FLOW);
}

void DeviceExecuteContext::PingPongRestoreAll(DevStartArgs* startArgs)
{
    const bool isFullCache = devProg->ctrlFlowCacheAnchor->IsActivatedFullCache(startArgs);
    for (size_t index = 0; index < devProg->ctrlFlowCacheAnchor->deviceTaskCount; index++) {
        DynDeviceTask* dynTask = reinterpret_cast<DynDeviceTask*>(
            devProg->ctrlFlowCacheAnchor->deviceTaskCacheList[index].dynTaskBase);
        devProg->ctrlFlowCacheAnchor->PredCountPingPongRestore(dynTask);
        devProg->ctrlFlowCacheAnchor->ReadyQueueDataPingPongRestore(dynTask);
        devProg->ctrlFlowCacheAnchor->DieReadyQueuePingPongRestore(dynTask);
    }
    if (isFullCache) {
        devProg->ctrlFlowCacheAnchor->bitmapRestoreDone = true;
    }
}

int DeviceExecuteContext::RunControlFlow(DevStartArgs* startArgs)
{
    PerfBegin(PERF_EVT_CONTROL_FLOW);
    RuntimeCallEntryType runtimeCallList[static_cast<uint32_t>(RuntimeCallStage::T_RUNTIME_CALL_MAX)] = {
        DeviceExecuteRuntimeCallRootAlloc,
        DeviceExecuteRuntimeCallRootStitch,
        DeviceExecuteRuntimeCallLog,
        DeviceExecuteRuntimeCallShmemAllocator,
        DeviceExecuteRuntimeCallSlotMarkNeedAlloc,
        DeviceExecuteRuntimeCallGetLoopDieId,
        DeviceExecuteRuntimeCallSetLoopDieId,
    };
    int originalErrorState = this->GetErrorState();
    execProg.controlFlowBinary.CallControlFlow(this, symbolTable.data(), runtimeCallList, startArgs);
    int finalErrorState = this->GetErrorState();
    if (finalErrorState != originalErrorState && finalErrorState != DEVICE_MACHINE_OK) {
        DEV_ERROR(CtrlErr::CTRL_FLOW_EXEC_FAILED,
                  "#ctrl.ctrlflow.leave: Control flow execution failed with error code: %d", finalErrorState);
        return finalErrorState;
    }
    PerfEnd(PERF_EVT_CONTROL_FLOW);
    return DEVICE_MACHINE_OK;
}

int DeviceExecuteContext::GELaunchFullCacheRunControlFlow(DevStartArgs* startArgs, PushTaskEntry tPushTask)
{
    int ret = DEVICE_MACHINE_OK;
    ret = RunInit(startArgs, tPushTask);
    if (unlikely(ret != DEVICE_MACHINE_OK)) {
        return DEVICE_MACHINE_ERROR;
    }
    ret = RunControlFlow(startArgs);
    if (unlikely(ret != DEVICE_MACHINE_OK)) {
        return DEVICE_MACHINE_ERROR;
    }
    return ret;
}

void DeviceExecuteContext::GELaunchFullCache(DevStartArgs* startArgs, PushTaskEntry tPushTask)
{
    if (devProg->ctrlFlowCacheAnchor->IsActivatedFullCache(startArgs)) {
        DEV_TRACE_DEBUG(CtrlEvent(none(), ControlFlowCacheFullRunCache()));
        GELaunchRunCached(startArgs, tPushTask);
    } else {
        DEV_TRACE_DEBUG(CtrlEvent(none(), ControlFlowCacheFullRunControl()));
        GELaunchFullCacheRunControlFlow(startArgs, tPushTask);
    }
}

int DeviceExecuteContext::GELaunch(DevStartArgs* startArgs, PushTaskEntry tPushTask)
{
    int ret = DEVICE_MACHINE_OK;
    if (devProg->ctrlFlowCacheAnchor->IsRecording()) {
        devProg->ctrlFlowCacheAnchor->InitInputOutput(startArgs);
    }
    ret = GELaunchPartialCache(startArgs, tPushTask);
    if (unlikely(ret != DEVICE_MACHINE_OK)) {
        return DEVICE_MACHINE_ERROR;
    }
    return DEVICE_MACHINE_OK;
}

int DeviceExecuteContext::GELaunchPartialCache(DevStartArgs* startArgs, PushTaskEntry tPushTask)
{
    int ret = DEVICE_MACHINE_OK;
    const bool isActivatedCache = devProg->ctrlFlowCacheAnchor->IsActivatedCache(startArgs);
    DEV_TRACE_DEBUG(
        CtrlEvent(none(), Workspace(Range(startArgs->contextWorkspaceAddr,
                                          startArgs->contextWorkspaceAddr + startArgs->contextWorkspaceSize))));

    if (isActivatedCache) {
        controlFlowCacheActivated = true;
        DEV_TRACE_DEBUG(CtrlEvent(none(), ControlFlowCachePartRunCache(devProg->ctrlFlowCacheAnchor->deviceTaskCount,
                                                                       devProg->ctrlFlowCacheAnchor->rootTaskCount)));
        GELaunchRunCached(startArgs, tPushTask);
    }
    DEV_TRACE_DEBUG(CtrlEvent(none(), ControlFlowCacheFullRunControl()));
    if (!devProg->ctrlFlowCacheAnchor->IsActivatedFullCache(startArgs)) {
        ret = RunInit(startArgs, tPushTask);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return DEVICE_MACHINE_ERROR;
        }
        ret = RunControlFlow(startArgs);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return DEVICE_MACHINE_ERROR;
        }
        DEV_IF_INFO { workspace.LogTuningSummary(); }
    }
    if (isActivatedCache) {
        PingPongRestoreAll(startArgs);
    }
    return ret;
}

bool DeviceExecuteContext::AiCoreFree()
{
    return false; // extend check point
}

void DeviceExecuteContext::DumpDeviceTask(uint64_t taskId, DynDeviceTask* deviceTask)
{
    DEV_IF_VERBOSE_DEBUG {}
    else
    {
        return;
    }
    for (uint64_t dupIdx = 0; dupIdx < deviceTask->dynFuncDataCacheListSize; dupIdx++) {
        DevAscendFunctionDuppedData* dupped = deviceTask->dynFuncDataCacheList[dupIdx].duppedData;
        DEV_TRACE_DEBUG(
            REvent(RUid(taskId, dupIdx, dupped->GetSource()->GetRootIndex()), dupped->SchemaGetWorkspace()));
        size_t incastSize = dupped->GetSource()->GetIncastSize();
        DEV_TRACE_DEBUG(REvent(RUid(taskId, dupIdx, dupped->GetSource()->GetRootIndex()), RActIncastCount(incastSize)));
        for (size_t i = 0; i < incastSize; ++i) {
            DEV_TRACE_DEBUG(REvent(RUid(taskId, dupIdx, dupped->GetSource()->GetRootIndex()),
                                   RActIncast(i, dupped->SchemaGetIncastRange(i))));
        }

        size_t outcastSize = dupped->GetSource()->GetOutcastSize();
        DEV_TRACE_DEBUG(
            REvent(RUid(taskId, dupIdx, dupped->GetSource()->GetRootIndex()), RActOutcastCount(outcastSize)));
        for (size_t i = 0; i < outcastSize; ++i) {
            DEV_TRACE_DEBUG(REvent(RUid(taskId, dupIdx, dupped->GetSource()->GetRootIndex()),
                                   RActOutcast(i, dupped->SchemaGetOutcastRange(i))));
        }
        DEV_TRACE_DEBUG(REvent(RUid(taskId, dupIdx, dupped->GetSource()->GetRootIndex()),
                               RActExpressionCount(dupped->GetExpressionSize())));
        DEV_TRACE_DEBUG_SPLIT(
            REvent(RUid(taskId, dupIdx, dupped->GetSource()->GetRootIndex()), expr(dupped->SchemaGetExpressionList())));
    }
}

void DeviceExecuteContext::ProcessControlFlowCacheRecord(DynDeviceTask* dynTask)
{
    if (devProg->ctrlFlowCacheAnchor->IsRecording()) {
        if (!devProg->ctrlFlowCacheAnchor->IsRecordingStopped()) {
            devProg->ctrlFlowCacheAnchor->PredCountDataBackup(dynTask);
            devProg->ctrlFlowCacheAnchor->ReadyQueueDataBackup(dynTask);
            devProg->ctrlFlowCacheAnchor->DieReadyQueueDataBackup(dynTask);
            devProg->ctrlFlowCacheAnchor->MixTaskDataBackup(dynTask);
            devProg->ctrlFlowCacheAnchor->IncastOutcastAddrBackup(dynTask);
            devProg->ctrlFlowCacheAnchor->TaskAddrBackupWorkspace(dynTask);
            devProg->ctrlFlowCacheAnchor->RuntimeAddrBackup(slotContext.GetSlotList(),
                                                            workspace.GetRuntimeOutcastTensorPool(), devProg->slotSize,
                                                            devProg->memBudget.tensor.runtimeOutcastPoolSize,
                                                            workspace.GetTensorAllocator(), devProg->GetParallelism());
        }
        devProg->ctrlFlowCacheAnchor->AppendDeviceTask(dynTask);
    }
}

void DeviceExecuteContext::CalcControlMaxAicore()
{
    // 在 Submit 时遍历 stitchContext 累加所有真正执行的 devRoot 的 maxCV
    currentMaxC_ = 0;
    currentMaxV_ = 0;
    const auto& stitchedList = stitchContext.GetStitchedList();
    const size_t stitchedSize = stitchedList.size();
    for (size_t i = 0; i < stitchedSize; i++) {
        const DevAscendFunction* sourceFunc = stitchedList[i].GetSource();
        if (sourceFunc != nullptr) {
            currentMaxC_ += sourceFunc->GetMaxC();
            currentMaxV_ += sourceFunc->GetMaxV();
        }
    }

    // InitDyn already resolved per-round nrValidAic onto DevStartArgs; Submit only runs after RunInit sets args.
    const uint32_t nrValidAic = args->nrValidAic;
    if (currentMaxC_ == 0 && currentMaxV_ == 0) {
        currentMaxC_ = nrValidAic;
        currentMaxV_ = currentMaxC_ * AIV_NUM_PER_AI_CORE;
        return;
    }

    uint32_t oriAivNum = nrValidAic * AIV_NUM_PER_AI_CORE;
    currentMaxC_ = currentMaxC_ >= nrValidAic ? nrValidAic : currentMaxC_;
    currentMaxV_ = currentMaxV_ >= oriAivNum ? oriAivNum : currentMaxV_;
    if (devProg->devArgs.archInfo == ArchInfo::DAV_3510) {
        if (currentMaxC_ * AIV_NUM_PER_AI_CORE >= currentMaxV_) {
            currentMaxV_ = currentMaxC_ * AIV_NUM_PER_AI_CORE;
        } else {
            currentMaxV_ = (currentMaxV_ & 1) ? currentMaxV_ + 1 : currentMaxV_;
            currentMaxC_ = currentMaxV_ / AIV_NUM_PER_AI_CORE;
        }
    }
    DEV_INFO("[CalcControlMaxAicore] stitchedSize=%u, maxC=%u, maxV=%u", static_cast<uint32_t>(stitchedSize),
             currentMaxC_, currentMaxV_);
}

int DeviceExecuteContext::PrepareShmemWaitUntilTasks(DynDeviceTask* dynTask)
{
    auto funcHeader = dynTask->GetDynFuncDataList();
    auto funcDataList = reinterpret_cast<npu::tile_fwk::DynFuncData*>(funcHeader + 1);
    auto hcclContextAddr = funcHeader->startArgs->commContexts;
    size_t cacheSize = sizeof(npu::tile_fwk::Distributed::ShmemWaitUntilCache);
    WsAllocation alloc = ControlFlowAllocateSlab(
        devProg, cacheSize, workspace.SlabAlloc(cacheSize, WsAicpuSlabMemType::SHMEM_WAIT_UNTIL_CACHE));
    auto cache = alloc.As<npu::tile_fwk::Distributed::ShmemWaitUntilCache>();
    if (cache == nullptr) {
        DEV_ERROR(CtrlErr::CTRL_FLOW_EXEC_FAILED, "AllocateCache failed for ShmemWaitUntilCache");
        return DEVICE_MACHINE_ERROR;
    }

    uint32_t taskCount = 0;

    for (uint64_t funcId = 0; funcId < dynTask->dynFuncDataCacheListSize; ++funcId) {
        auto callList = dynTask->dynFuncDataCacheList[funcId].calleeList;
        for (size_t opIndex = 0; opIndex < dynTask->dynFuncDataCacheList[funcId].devFunc->GetOperationSize();
             ++opIndex) {
            auto coreType = dynTask->cceBinary[callList[opIndex]].coreType;
            if (coreType == static_cast<int>(MachineType::AICPU)) {
                if (taskCount >= npu::tile_fwk::Distributed::AICPU_TASK_ARRAY_SIZE) {
                    DEV_ERROR(CtrlErr::CTRL_FLOW_EXEC_FAILED,
                              "PrepareShmemWaitUntilTasks: taskCount=%u exceeds AICPU_TASK_ARRAY_SIZE=%lu", taskCount,
                              npu::tile_fwk::Distributed::AICPU_TASK_ARRAY_SIZE);
                    return DEVICE_MACHINE_ERROR;
                }
                uint32_t aicpuTaskId = (static_cast<uint32_t>(funcId) << TASKID_TASK_BITS) |
                                       static_cast<uint32_t>(opIndex);
                auto& code = dynTask->aicpuLeafBinary[callList[opIndex]].aicpuLeafCode;
                auto prepareRet = npu::tile_fwk::Distributed::ShmemWaitUntilImpl::PrepareTask(
                    aicpuTaskId, code, cache->taskArray, taskCount, funcDataList, hcclContextAddr);
                if (prepareRet != DEVICE_MACHINE_OK) {
                    DEV_ERROR(CtrlErr::CTRL_FLOW_EXEC_FAILED, "PrepareTask failed: aicpuTaskId=%u ret=%d", aicpuTaskId,
                              prepareRet);
                    return DEVICE_MACHINE_ERROR;
                }
                ++taskCount;
            }
        }
    }
    cache->taskCount = taskCount;
    npu::tile_fwk::Distributed::ShmemWaitUntilImpl::BuildHashTable(cache, taskCount);
    dynTask->shmemWaitUntilCacheBackup = cache;
    DEV_INFO("[ControlFlow] PrepareShmemWaitUntilTasks END: prepared %u tasks, cache=%p", taskCount, cache);
    return DEVICE_MACHINE_OK;
}

int DeviceExecuteContext::SubmitToAicoreAndRecycleMemory(bool withoutTail, bool isLastTask, bool isParallelIterLastTask)
{
    int ret = DEVICE_MACHINE_OK;
    DEV_VERBOSE_DEBUG("Submit stitch task");
    DEV_TRACE_DEBUG(DEvent(taskId, DActSubmit(stitchContext.Size())));
    AutoScopedPerf asp(PERF_EVT_SUBMIT_AICORE);
    if (stitchContext.Empty()) {
        DEV_INFO("Stitch context is empty.");
        return ret;
    }

    PROF_STAGE_BEGIN(PERF_EVT_DECIDE_SLOT_ADDRESS, "slotaddr.before\n");
    stitchContext.DecideSlotAddress(slotContext.GetSlotList(), slotContext.GetSlotSize());
    PROF_STAGE_END(PERF_EVT_DECIDE_SLOT_ADDRESS, "slotaddr.after\n");
    if (unlikely(ret != DEVICE_MACHINE_OK)) {
        return DEVICE_MACHINE_ERROR;
    }

    PROF_STAGE_BEGIN(PERF_EVT_DECIDE_INCAST_ADDRESS, "incastaddr.before\n");
    ret = stitchContext.DecideIncastOutcast(taskId);
    PROF_STAGE_END(PERF_EVT_DECIDE_INCAST_ADDRESS, "incastaddr.after\n");
    if (unlikely(ret != DEVICE_MACHINE_OK)) {
        return DEVICE_MACHINE_ERROR;
    }

    DEV_IF_VERBOSE_DEBUG
    {
        stitchContext.DumpStitchInfo();
#if !DEBUG_INFINITE_LIFETIME
        stitchContext.VerifyStitchedListMemory(*args);
#endif
    }

#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
    workspace.MarkAsNewStitchWindow();
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL

    CalcControlMaxAicore();
    DEV_IF_INFO { workspace.LogTuningUsage(taskId, stitchContext.Size()); }
    PROF_STAGE_BEGIN(PERF_EVT_STAGE_BUILD_TASK, "BuildDeviceTaskData.before\n");
    DynDeviceTask* dynTask = taskContext.BuildDeviceTaskData(stitchContext, taskId, devProg, withoutTail);
    if (dynTask == nullptr) {
        DEV_ERROR(DevCommonErr::NULLPTR, "#ctrl.buildtask.leave: Build device task data failed.");
        return DEVICE_MACHINE_ERROR;
    }
    dynTask->SetMaxCV(currentMaxC_, currentMaxV_);

    if (parallelCtx.isInParallelForScope) {
        dynTask->SetParallelInfo(parallelCtx.info);
    }

    if (!devProg->ctrlFlowCacheAnchor->IsRecording() ||
        (devProg->ctrlFlowCacheAnchor->IsRecording() && devProg->ctrlFlowCacheAnchor->IsCacheOriginShape())) {
        dynTask->SetLastTask(isLastTask);
        dynTask->SetParallelSameIterLastDevTask(isParallelIterLastTask);
    }
    DEV_INFO("devProg->ctrlFlowCacheAnchor->isActivated : %d", devProg->ctrlFlowCacheAnchor->isActivated);
    PROF_STAGE_END(PERF_EVT_STAGE_BUILD_TASK, "BuildDeviceTaskData.after\n");
    if (!devProg->ctrlFlowCacheAnchor->isActivated && devProg->devArgs.hasAicpuTask) {
        PrepareShmemWaitUntilTasks(dynTask);
    }

    PROF_STAGE_BEGIN(PERF_EVT_DEALLOCATE_WORKSPACE, "RecycleTensorWorkspace.before\n");
    // Memory recycling
    stitchContext.RecycleTensorWorkspace();

    // Reset stitch context
    stitchContext.Reset();
    slotContext.ClearDirty();
    DEV_IF_INFO { workspace.ResetTuningDeviceTaskPeaks(); }
    PROF_STAGE_END(PERF_EVT_DEALLOCATE_WORKSPACE, "RecycleTensorWorkspace.after\n");

    ProcessControlFlowCacheRecord(dynTask);

    PROF_STAGE_BEGIN(PERF_EVT_STAGE_PUSH_TASK, "push.before\n");
    DumpDeviceTask(taskId, dynTask);
    PerfMtTrace(PERF_TRACE_DEV_TASK_BUILD, CTRL_CPU_THREAD_IDX);
    PushTask(dynTask);
    PROF_STAGE_END(PERF_EVT_STAGE_PUSH_TASK, "push.after\n");

    currentMaxC_ = 0;
    currentMaxV_ = 0;
    return ret;
}

schema::RUid DeviceExecuteContext::GetRuid(uint64_t rootKey, bool afterAppend)
{
    int64_t dupIndex = stitchContext.Size();
    if (afterAppend) {
        dupIndex -= 1;
    }
    schema::RUid ruid(taskId, dupIndex, rootKey);
    return ruid;
}

int DeviceExecuteContext::ControlFlowCacheStopCache(uint64_t rootKey)
{
    int ret = DEVICE_MACHINE_OK;
    ret = SubmitToAicoreAndRecycleMemory(false);
    if (unlikely(ret != DEVICE_MACHINE_OK)) {
        return DEVICE_MACHINE_ERROR;
    }
    devProg->ctrlFlowCacheAnchor->StopRecording();
    DEV_INFO("[Stitch Finish] Stop recording ctrl flow cache. rootKey=%" PRIu64 ".", rootKey);
    return ret;
}

void* DeviceExecuteContext::CallRootFunctionAlloc(uint64_t rootKey)
{
    int ret = DEVICE_MACHINE_OK;
    DevAscendFunction* devRoot = devProg->GetFunction(rootKey);
    DEV_DEBUG("Alloc one func %lu %p %s.", rootKey, devRoot, devRoot->GetRawName());
    const bool stitchByMemoryOnly = devProg->memBudget.tensor.memoryDrivenWorkspace != 0;
    const uint16_t realStitchNumThreshold = parallelCtx.isInParallelForScope ? MAX_STITCH_FUNC_NUM :
                                                                               stitchTaskLoopNumThreshold;
    const bool stitchCountExceeded = !stitchByMemoryOnly && (stitchContext.Size() >= realStitchNumThreshold);
    const bool callOpExceeded = stitchContext.stitchedCallOpSize() + devRoot->GetOperationSize() >
                                devProg->stitchFunctionsize;
    const bool rootFuncCountExceeded = stitchContext.Size() >= MAX_STITCH_FUNC_NUM;
    if (stitchCountExceeded || callOpExceeded || rootFuncCountExceeded) {
        DEV_INFO("[Stitch Finish] Stitch Limit Exceeded. memoryDriven=%d rootCount=%zu (limit=%u, hardLimit=%zu) "
                 "rootKey=%lu, func=%s, #callop=%u+%zu (limit=%u).",
                 static_cast<int>(stitchByMemoryOnly), stitchContext.Size(), realStitchNumThreshold,
                 MAX_STITCH_FUNC_NUM, rootKey, devRoot->GetRawName(), stitchContext.stitchedCallOpSize(),
                 devRoot->GetOperationSize(), devProg->stitchFunctionsize);
        ret = SubmitToAicoreAndRecycleMemory(false);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return RUNTIME_FUNCKEY_ERROR;
        }
    }
    DEV_TRACE_DEBUG(REvent(GetRuid(rootKey), RActDup(devRoot->GetRawName())));

    PROF_STAGE_BEGIN(PERF_EVT_STAGE_DUP_ROOT, "dup.before\n");
    currDevRootDup = workspace.DuplicateRoot(devRoot);
    PROF_STAGE_END(PERF_EVT_STAGE_DUP_ROOT, "dup.after\n");
    return reinterpret_cast<void*>(&currDevRootDup.GetExpression(0));
}

bool DeviceExecuteContext::NeedSubmmitDevTask(uint64_t rootkey)
{
    return (rootkey == RUNTIME_FUNCKEY_FINISH || rootkey == RUNTIME_FUNCKEY_LOOP_BARRIER ||
            rootkey == RUNTIME_FUNCKEY_PARALLEL_FOR_END || rootkey == RUNTIME_FUNCKEY_PARALLEL_FOR_BEGIN);
}

void DeviceExecuteContext::ParallelForBegin()
{
    parallelCtx.Begin();
    workspace.SwitchWParallelWorkSpace(parallelCtx.info.wsId);
}

void* DeviceExecuteContext::CallRootFunctionStitch(uint64_t rootKey)
{
    int ret = DEVICE_MACHINE_OK;
    DEV_DEBUG("Root stitch %lu.", rootKey);

    if (rootKey == RUNTIME_FUNCKEY_CACHESTOP) {
        if (devProg->ctrlFlowCacheAnchor->IsRecording()) {
            ret = ControlFlowCacheStopCache(rootKey);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return RUNTIME_FUNCKEY_ERROR;
            }
            return RUNTIME_FUNCRET_CACHESTOP_RETURN;
        } else {
            return RUNTIME_FUNCRET_CACHESTOP_CONTINUE;
        }
    }

    if (NeedSubmmitDevTask(rootKey)) {
        ret = SubmitToAicoreAndRecycleMemory(false, rootKey == RUNTIME_FUNCKEY_FINISH,
                                             rootKey == RUNTIME_FUNCKEY_PARALLEL_FOR_END);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return RUNTIME_FUNCKEY_ERROR;
        }

        switch (rootKey) {
            case RUNTIME_FUNCKEY_LOOP_BARRIER: {
                parallelCtx.ChangeForId();
                break;
            }
            case RUNTIME_FUNCKEY_PARALLEL_FOR_BEGIN: {
                ParallelForBegin();
                break;
            }
            case RUNTIME_FUNCKEY_PARALLEL_FOR_END: {
                parallelCtx.End();
                break;
            }
            default: {
                break;
            }
        }

        DEV_INFO("[Stitch Finish] Finish Signal or Barrier. rootKey=%" PRIu64 ".", rootKey);
        return nullptr;
    }

    DEV_TRACE_DEBUG(REvent(GetRuid(rootKey), currDevRootDup.SchemaGetExpressionTable()));
    // dyn rawshape size depend expresstable calculated
    FunctionMemoryAllocFailure allocStatus = workspace.TryAllocateFunctionMemory(currDevRootDup,
                                                                                 slotContext.GetSlotList());
    while (allocStatus != FunctionMemoryAllocFailure::OK) {
        const char* failureName = FunctionMemoryAllocFailureName(allocStatus);
        if (stitchContext.Empty()) {
            DEV_ERROR(CtrlErr::CTRL_FLOW_EXEC_FAILED,
                      "#ctrl.workspace.preflight: Root cannot fit in an empty DeviceTask, rootKey=%" PRIu64
                      ", func=%s, reason=%s.",
                      rootKey, currDevRootDup.GetSource()->GetRawName(), failureName);
            return RUNTIME_FUNCKEY_ERROR;
        }
        ret = SubmitToAicoreAndRecycleMemory(true);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return RUNTIME_FUNCKEY_ERROR;
        }
        allocStatus = workspace.TryAllocateFunctionMemory(currDevRootDup, slotContext.GetSlotList());
    }

    if (AiCoreFree()) {
        ret = SubmitToAicoreAndRecycleMemory(false);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return RUNTIME_FUNCKEY_ERROR;
        }
        DEV_INFO("[Stitch Finish] AICore Free.");
    }

    DEV_TRACE_DEBUG(DEvent(taskId, DActStitchStart(GetRuid(rootKey))));
    PROF_STAGE_BEGIN(PERF_EVT_STAGE_STITCH, "stitch.before\n");
    size_t devNextIdx = stitchContext.Size();
    const uint32_t cellMatchTagSeq = devProg->GetCellMatchTagSeq();
    stitchContext.Stitch(slotContext, currDevRootDup, taskId, devNextIdx, cellMatchTagSeq);

    uint32_t updateErrCode = slotContext.UpdateSlots(currDevRootDup, taskId, devNextIdx, cellMatchTagSeq);
    if (updateErrCode == static_cast<uint32_t>(CtrlErr::CELL_MATCH_FILL_OP_NOT_ENOUGH)) {
        DEV_WARN("UpdateSlots stitch cell failed with error code %u, force submit devtask", updateErrCode);
        ret = SubmitToAicoreAndRecycleMemory(false);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return RUNTIME_FUNCKEY_ERROR;
        }
    }
    PROF_STAGE_END(PERF_EVT_STAGE_STITCH, "stitch.after\n");
    DEV_TRACE_DEBUG(DEvent(taskId, DActStitchFinish(GetRuid(rootKey, true))));
    return nullptr;
}

void DeviceExecuteContext::MarkSlotNeedAlloc(int slotIndex)
{
    DEV_ASSERT_MSG(
        DevCommonErr::PARAM_INVALID, slotIndex >= 0 && slotIndex < static_cast<int>(slotContext.GetSlotSize()),
        "MarkSlotNeedAlloc: Invalid slot index %d, valid range [0, %zu).", slotIndex, slotContext.GetSlotSize());
    slotContext.GetSlotList()[slotIndex].isAssembleSlotNeedAlloc = true;
    return;
}

void DeviceExecuteContext::SetLoopDieId(int8_t dieId)
{
    if (DuppedRootCached()) {
        return;
    }
    currDevRootDup.DupDataForDynFuncData()->loopDieId_ = dieId;
}

void* DeviceExecuteContext::DeviceExecuteRuntimeCallRootAlloc(void* ctx_, uint64_t rootKey)
{
    DeviceExecuteContext* ctx = (DeviceExecuteContext*)ctx_;
    if (ctx == nullptr) {
        DEV_ERROR(CtrlErr::ROOT_ALLOC_CTX_NULL, "#ctrl.ctrlflow.call.root_alloc: invalid ctx.");
        return nullptr;
    }
    PerfBegin(PERF_EVT_ROOT_FUNC);
    void* result = nullptr;
    if (ctx->DuppedRootCached()) {
        result = nullptr;
    } else if (ctx->devProg->ctrlFlowCacheAnchor->IsRecording() &&
               ctx->devProg->ctrlFlowCacheAnchor->IsRecordingStopped()) {
        result = nullptr;
    } else {
        result = ctx->CallRootFunctionAlloc(rootKey);
        if (result == RUNTIME_FUNCKEY_ERROR) {
            ctx->SetErrorState(DEVICE_MACHINE_ERROR);
        }
    }

    PerfEnd(PERF_EVT_ROOT_FUNC);
    return result;
}

bool IsSpecialRootKey(uint64_t rootKey)
{
    if (rootKey == RUNTIME_FUNCKEY_FINISH || rootKey == RUNTIME_FUNCKEY_CACHESTOP ||
        rootKey == RUNTIME_FUNCKEY_LOOP_BARRIER || rootKey == RUNTIME_FUNCKEY_PARALLEL_FOR_BEGIN ||
        rootKey == RUNTIME_FUNCKEY_PARALLEL_FOR_END) {
        return true;
    }
    return false;
}

void* DeviceExecuteContext::DeviceExecuteRuntimeCallRootStitch(void* ctx_, uint64_t rootKey)
{
    DeviceExecuteContext* ctx = (DeviceExecuteContext*)ctx_;
    if (ctx == nullptr) {
        DEV_ERROR(CtrlErr::ROOT_STITCH_CTX_NULL, "#ctrl.ctrlflow.call.root_stitch: invalid ctx.");
        return nullptr;
    }
    PerfBegin(PERF_EVT_ROOT_FUNC);
    void* result = nullptr;
    if (ctx->DuppedRootCached()) {
        result = nullptr;
    } else if (ctx->devProg->ctrlFlowCacheAnchor->IsRecording() &&
               ctx->devProg->ctrlFlowCacheAnchor->IsRecordingStopped()) {
        result = nullptr;
    } else {
        result = ctx->CallRootFunctionStitch(rootKey);
        if (result == RUNTIME_FUNCKEY_ERROR) {
            ctx->SetErrorState(DEVICE_MACHINE_ERROR);
        }
    }

    if (result == nullptr && IsSpecialRootKey(rootKey)) {
        return result;
    }

    PerfEnd(PERF_EVT_ROOT_FUNC);
    if (ctx->DuppedRootUpdateAndCachedAllSubmitted()) {
        DEV_TRACE_DEBUG(CtrlEvent(none(), ControlFlowCachePartRunControlContinue()));
        // forcely break device task
        auto ctrlFlowCacheAnchor = ctx->devProg->ctrlFlowCacheAnchor;
        ctrlFlowCacheAnchor->RuntimeAddrRestore(ctx->slotContext.GetSlotList(),
                                                ctx->workspace.GetRuntimeOutcastTensorPool(), ctx->devProg->slotSize,
                                                ctx->devProg->memBudget.tensor.runtimeOutcastPoolSize,
                                                ctx->workspace.GetTensorAllocator(), ctx->devProg->GetParallelism());
        ctrlFlowCacheAnchor->RuntimeAddrRelocWorkspace(
            0, ctx->args->contextWorkspaceAddr, ctx->args, ctx->slotContext.GetSlotList(),
            ctx->workspace.GetRuntimeOutcastTensorPoolBase(), ctx->devProg->GetParallelism(),
            ctx->workspace.GetTensorAllocator());
    }
    return result;
}

void* DeviceExecuteContext::DeviceExecuteRuntimeCallLog(void* ctx_, uint64_t value)
{
    (void)ctx_;
    DEV_DEBUG("DeviceExecuteRuntimeCallLog -> Value: %lu", value);
    return nullptr;
}

void* DeviceExecuteContext::DeviceExecuteRuntimeCallShmemAllocator(void* ctx_, uint64_t value)
{
    uint64_t groupIndex = (reinterpret_cast<uint64_t*>(value))[0];
    uint64_t memType = (reinterpret_cast<uint64_t*>(value))[1];
    uint64_t size = (reinterpret_cast<uint64_t*>(value))[2];
    uint64_t maxTileNum = (reinterpret_cast<uint64_t*>(value))[3];
    constexpr uint64_t memTypeCount = 2;

    DEV_ASSERT(DevCommonErr::PARAM_CHECK_FAILED, memType < memTypeCount);
    DeviceExecuteContext* ctx = (DeviceExecuteContext*)ctx_;
    DEV_ASSERT(DevCommonErr::PARAM_CHECK_FAILED, groupIndex < ctx->args->commGroupNum);
    auto hcclOpParam = reinterpret_cast<TileOp::CommContext*>(ctx->args->commContexts[groupIndex]);
    uint64_t winSize = memType == 0 ? hcclOpParam->winDataSize : hcclOpParam->winStatusSize;
    static uint64_t staticShmemAddrEndOffset[MAX_SHMEM_GROUP_NUM][SHMEM_MEM_TYPE_NUM] = {0};
    uint64_t shmemAddrEndOffset = staticShmemAddrEndOffset[groupIndex][memType] + size;
    if (shmemAddrEndOffset > winSize) {
        staticShmemAddrEndOffset[groupIndex][memType] = 0UL;
    }
    uint64_t vaddr = TileOp::Distributed::EncodeShmemAddr(staticShmemAddrEndOffset[groupIndex][memType], maxTileNum,
                                                          groupIndex, memType);
    staticShmemAddrEndOffset[groupIndex][memType] += size;
    return reinterpret_cast<void*>(vaddr);
}

void* DeviceExecuteContext::DeviceExecuteRuntimeCallSlotMarkNeedAlloc(void* ctx_, uint64_t slotIndex)
{
    DeviceExecuteContext* ctx = (DeviceExecuteContext*)ctx_;
    ctx->MarkSlotNeedAlloc(slotIndex);
    return nullptr;
}

void* DeviceExecuteContext::DeviceExecuteRuntimeCallGetLoopDieId(void* ctx_, uint64_t rootKey)
{
    (void)rootKey;
    DeviceExecuteContext* ctx = (DeviceExecuteContext*)ctx_;
    return static_cast<void*>(&ctx->loopDieId_);
}

void* DeviceExecuteContext::DeviceExecuteRuntimeCallSetLoopDieId(void* ctx_, uint64_t rootKey)
{
    (void)rootKey;
    DeviceExecuteContext* ctx = (DeviceExecuteContext*)ctx_;
    ctx->SetLoopDieId(ctx->loopDieId_);
    DEV_VERBOSE_DEBUG("Set loop die id:%d:", ctx->loopDieId_);
    return nullptr;
}
} // namespace npu::tile_fwk::dynamic
