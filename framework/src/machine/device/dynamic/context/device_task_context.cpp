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
 * \file device_task_context.cpp
 * \brief
 */

#include "machine/device/dynamic/context/device_task_context.h"
#include "machine/device/dynamic/eslmodel_aicore_hal.h"
#include "machine/device/dump/dump_device_memory.h"
#ifndef __DEVICE__
#include "interface/configs/config_manager.h"
#endif

namespace npu::tile_fwk::dynamic {
namespace {
const uint32_t OP_ATTRS_PRE_NUM = 8;
const uint32_t OP_ATTRS_OFFSET_PRE_NUM = 4;
const uint32_t EXPR_TABLE_PRE_NUM = 8;
const uint32_t RAW_TENSOR_ADDR_MASK = 8;
const uint32_t CCE_BINARY_MOD = 8;
const size_t DUP_PRED_COUNT_LOOP_MAX = 8;
const size_t DUP_PRED_COUNT_PRE_LOOP_CNT = 4;

inline void CheckAddrAligned(uint64_t addr, uint32_t align, const char* msg)
{
    DEV_ASSERT_MSG(ProgEncodeErr::DYNFUNC_DATA_ALIGNMENT_ERROR, addr % align == 0, "%s", msg);
}

inline void FillOneDynFuncData(DynFuncData* dyndata, DevAscendFunctionDupped& dupFunc, uint64_t& leafFuncDataSize,
                               uint64_t& leafFuncNum)
{
    auto* source = dupFunc.GetSource();
    dyndata->opAttrs = reinterpret_cast<uint64_t*>(const_cast<SymInt*>(source->GetSymoffset(0)));
    dyndata->opAtrrOffsets = source->GetOpAttrOffsetAddr();
    dyndata->exprTbl = dupFunc.GetExpressionAddr();
    dyndata->rawTensorAddr = reinterpret_cast<uint64_t*>(&dupFunc.GetIncastAddress(0));
    dyndata->rawTensorDesc = source->GetRawTensorDesc(0);
    dyndata->workspaceAddr = dupFunc.RuntimeWorkspace();
    dyndata->drcoRootFuncData = {};
    DEV_IF_NONDEVICE
    {
        dyndata->exprNum = source->expressionList.size();
        dyndata->opAttrSize = source->GetOpAttrSize();
        dyndata->rawTensorAddrSize = source->GetIncastSize() + source->GetOutcastSize();
        dyndata->rawTensorDescSize = source->GetRawTensorDescSize();
    }

    DEV_IF_DEBUG
    {
        CheckAddrAligned(reinterpret_cast<uint64_t>(dyndata->opAttrs), OP_ATTRS_PRE_NUM,
                         "#ctrl.task.pre.dynfunc.process: opAttrs address is not aligned.");
        CheckAddrAligned(reinterpret_cast<uint64_t>(dyndata->opAtrrOffsets), OP_ATTRS_OFFSET_PRE_NUM,
                         "#ctrl.task.pre.dynfunc.process: opAtrrOffsets address is not aligned.");
        CheckAddrAligned(reinterpret_cast<uint64_t>(dyndata->exprTbl), EXPR_TABLE_PRE_NUM,
                         "#ctrl.task.pre.dynfunc.process: exprTbl address is not aligned.");
        CheckAddrAligned(reinterpret_cast<uint64_t>(dyndata->rawTensorAddr), RAW_TENSOR_ADDR_MASK,
                         "#ctrl.task.pre.dynfunc.process: rawTensorAddr address is not aligned.");
        leafFuncDataSize += source->GetOpAttrSize() * sizeof(SymInt);
        leafFuncDataSize += source->GetOperationSize() * sizeof(int32_t);
        leafFuncDataSize += source->expressionList.size() * sizeof(int64_t);
        leafFuncDataSize += source->GetRawTensorSize() * sizeof(DevRawTensorDesc);
        leafFuncNum += source->GetOperationSize();
    }
    dupFunc.SetFuncData(dyndata);
}
} // namespace

void DeviceTaskContext::InitAllocator(DevAscendProgram* devProg, DeviceWorkspaceAllocator& workspace,
                                      npu::tile_fwk::DevStartArgsBase* startArgs)
{
    devProg_ = devProg;
    workspace_ = &workspace;
    startArgs_ = startArgs;
    enableAicoreResolve_ = devProg->devArgs.enableAicoreResolve;
}

DynDeviceTask* DeviceTaskContext::BuildDeviceTaskData(DeviceStitchContext& stitchContext, uint32_t taskId,
                                                      DevAscendProgram* devProg, bool withoutTail)
{
    int ret = DEVICE_MACHINE_OK;
    PerfBegin(PERF_EVT_ALLOCATE_TASK);
    DynDeviceTask* dynTask = workspace_->MakeDynDeviceTask();
    ret = stitchContext.MoveTo(dynTask);
    if (unlikely(ret != DEVICE_MACHINE_OK)) {
        return nullptr;
    }
    PerfEnd(PERF_EVT_ALLOCATE_TASK);

    PerfBegin(PERF_EVT_BUILD_TASK_DATA);
    ret = BuildDeviceTaskDataAndReadyQueue(dynTask, taskId, devProg);
    if (unlikely(ret != DEVICE_MACHINE_OK)) {
        return nullptr;
    }
    PerfEnd(PERF_EVT_BUILD_TASK_DATA);

    PerfBegin(PERF_EVT_SLAB_MEM_SUBMIT);
    // cache allocated memory , when task finish will recycle
    dynTask->taskStageAllocMem = workspace_->SlabGetStageAllocMem(withoutTail, WsAicpuSlabMemType::DUPPED_FUNC_DATA);
    DEV_IF_DEBUG { workspace_->DumpSlabUsageBeforeSubmit(taskId, dynTask); }
    if (!devProg->ctrlFlowCacheAnchor->IsRecording()) {
        workspace_->SlabStageAllocMemSubmmit(dynTask);
    } else {
        DEV_VERBOSE_DEBUG("[workspace.slab.submit] skip slab stage submit in ctrlflow cache recording, taskId=%u.",
                          taskId);
    }
    PerfEnd(PERF_EVT_SLAB_MEM_SUBMIT);
    return dynTask;
}

void DeviceTaskContext::ReleaseFinishedTasks(int perfEvtReleaseFinishTask, int perfEvtDeallocateTask)
{
    (void)perfEvtReleaseFinishTask;
    (void)perfEvtDeallocateTask;
}

void DeviceTaskContext::AppendFinishTask(DynDeviceTask* dynTask) { (void)dynTask; }

void DeviceTaskContext::ShowStats()
{
    DEV_DEBUG("   Stitched function count: %10lu.", stitchedFuncNum);
    DEV_DEBUG("       Root function count: %10lu.", rootFuncNum);
    DEV_DEBUG("       Leaf function count: %10lu.", leafFuncNum);
    DEV_DEBUG("   Initial ready task count: %10lu.", readyTaskNum);
    DEV_DEBUG(" Static function data size: %10lu bytes.", dynFuncDataSize);
    DEV_DEBUG("   Leaf function data size: %10lu bytes.", leafFuncDataSize);
}

void DeviceTaskContext::InitReadyCoreFunctionQueue(ReadyCoreFunctionQueue* q, uint32_t capacity)
{
    taskid_t* elem = reinterpret_cast<taskid_t*>(q + 1); // elems are immediately after ReadyCoreFunctionQueue struct
    new (q) ReadyCoreFunctionQueue(capacity, elem);
}

int DeviceTaskContext::InitReadyQueues(DynDeviceTask* dyntask, DevAscendProgram* devProg,
                                       ReadyCoreFunctionQueue* queue[READY_QUEUE_SIZE])
{
    uint32_t size = sizeof(ReadyCoreFunctionQueue) + dyntask->devTask.coreFunctionCnt * sizeof(taskid_t);
    if (dyntask->devTask.coreFunctionCnt > devProg->stitchFunctionsize) {
        DEV_ERROR(
            CtrlErr::DEVICE_TASK_BUILD_FAILED,
            "#ctrl.task.pre.queue.init: coreFunctionCnt (%lu) exceeds stitchFunctionsize (%u), cannot build ready "
            "queue.",
            dyntask->devTask.coreFunctionCnt, devProg->stitchFunctionsize);
        return DEVICE_MACHINE_ERROR;
    }

    for (size_t i = 0; i < READY_QUEUE_SIZE; ++i) {
        WsAllocation qalloc = ControlFlowAllocateSlab(devProg_, size,
                                                      workspace_->SlabAlloc(size, WsAicpuSlabMemType::READY_QUE));
        ReadyCoreFunctionQueue* q = qalloc.As<ReadyCoreFunctionQueue>();
        InitReadyCoreFunctionQueue(q, dyntask->devTask.coreFunctionCnt);
        queue[i] = q;
        dyntask->readyQueue[i] = q;
    }

    if (enableAicoreResolve_) {
        InitDrcoRootFuncList(dyntask);
    }
    return DEVICE_MACHINE_OK;
}

void DeviceTaskContext::ProcessAivBatchTasks(ReadyCoreFunctionQueue* aivQueue, size_t totalZeroPredAIVBatchEnd,
                                             const predcount_t* dupPredCountList, size_t funcIndex)
{
    uint32v8 one = {1, 1, 1, 1, 1, 1, 1, 1};
    uint32v8 base = {0, 1, 2, 3, 4, 5, 6, 7};

    for (size_t opIndex = 0; opIndex < totalZeroPredAIVBatchEnd; opIndex += DUP_PRED_COUNT_LOOP_MAX) {
        if (likely((*reinterpret_cast<const uint64_t*>(&dupPredCountList[opIndex]) |
                    *reinterpret_cast<const uint64_t*>(&dupPredCountList[opIndex + DUP_PRED_COUNT_PRE_LOOP_CNT])) ==
                   0)) {
            uint32v8 taskidv8 = (one * MakeTaskID(funcIndex, 0)) | (base + static_cast<uint32_t>(opIndex));
#ifdef __x86_64__
            aivQueue->UnsafeEnqueue(reinterpret_cast<ReadyCoreFunctionQueue::ValueType*>(&taskidv8),
                                    DUP_PRED_COUNT_LOOP_MAX);
#else
            aivQueue->UnsafeEnqueueSimd(taskidv8);
#endif
        } else {
            for (size_t idx = 0; idx < DUP_PRED_COUNT_LOOP_MAX; ++idx) {
                if (likely(dupPredCountList[opIndex + idx] == 0)) {
                    const auto taskId = MakeTaskID(funcIndex, opIndex + idx);
                    aivQueue->UnsafeEnqueue(taskId);
                }
            }
        }
    }
}

void DeviceTaskContext::UpdateDeviceTaskQueueInfo(DynDeviceTask* dyntask, ReadyCoreFunctionQueue* aicpuQueue,
                                                  ReadyCoreFunctionQueue* aivQueue, ReadyCoreFunctionQueue* aicQueue,
                                                  WrapInfoQueue* wrapQueue)
{
    dyntask->devTask.readyAivCoreFunctionQue = PtrToValue(aivQueue);
    dyntask->devTask.readyAicCoreFunctionQue = PtrToValue(aicQueue);
    dyntask->devTask.readyAicpuFunctionQue = PtrToValue(aicpuQueue);
    dyntask->devTask.mixTaskData.readyWrapCoreFunctionQue = PtrToValue(wrapQueue);
}

int DeviceTaskContext::ProcessZeroPredTask(DynDeviceTask* dyntask, WrapInfoQueue* wrapQueue, bool isNeedWrap)
{
    int wrapTaskNum = 0;
    size_t funcSize = dyntask->dynFuncDataCacheListSize;
    for (size_t funcIndex = 0; funcIndex < funcSize; ++funcIndex) {
        BuildReadyQueueForFunc(dyntask, funcIndex, isNeedWrap, wrapQueue, wrapTaskNum);
    }
    return wrapTaskNum;
}

int DeviceTaskContext::BuildReadyQueue(DynDeviceTask* dyntask, DevAscendProgram* devProg)
{
    PerfBegin(PERF_EVT_READY_QUEUE_IN);

    ReadyCoreFunctionQueue* queue[READY_QUEUE_SIZE];
    if (InitReadyQueues(dyntask, devProg, queue) != DEVICE_MACHINE_OK) {
        return DEVICE_MACHINE_ERROR;
    }
    ReadyCoreFunctionQueue* aicpuQueue = queue[DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AICPU)];
    ReadyCoreFunctionQueue* aivQueue = queue[DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIV)];
    ReadyCoreFunctionQueue* aicQueue = queue[DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIC)];

    InitDieReadyQueues(dyntask, devProg);

    bool isNeedWrap = IsNeedWrapProcess(dyntask, devProg);
    WrapInfoQueue* wrapQueue = nullptr;
    if (isNeedWrap) {
        wrapQueue = AllocWrapQueue(dyntask);
    }

    int wrapTaskNum = ProcessZeroPredTask(dyntask, wrapQueue, isNeedWrap);
    UpdateDeviceTaskQueueInfo(dyntask, aicpuQueue, aivQueue, aicQueue, wrapQueue);
    readyTaskNum += static_cast<uint64_t>(aivQueue->UnsafeSize() + aicQueue->UnsafeSize() + aicpuQueue->UnsafeSize() +
                                          wrapTaskNum);
    PerfEnd(PERF_EVT_READY_QUEUE_IN);
    return DEVICE_MACHINE_OK;
}

void DeviceTaskContext::BuildReadyQueueForFunc(DynDeviceTask* dyntask, size_t funcIndex, bool isNeedWrap,
                                               WrapInfoQueue* wrapQueue, int& wrapTaskNum)
{
    ReadyCoreFunctionQueue* aicpuQueue = dyntask
                                             ->readyQueue[DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AICPU)];
    ReadyCoreFunctionQueue* aivQueue = dyntask->readyQueue[DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIV)];
    ReadyCoreFunctionQueue* aicQueue = dyntask->readyQueue[DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIC)];
    ReadyCoreFunctionQueue* targetAivQueue = aivQueue;
    ReadyCoreFunctionQueue* targetAicQueue = aicQueue;
    if (IsMultiDie(devProg_) && (GetLoopDieId(dyntask, funcIndex) >= 0)) {
        auto dieId = GetLoopDieId(dyntask, funcIndex);
        targetAivQueue = reinterpret_cast<ReadyCoreFunctionQueue*>(
            dyntask->devTask.dieReadyFunctionQue.readyDieAivCoreFunctionQue[dieId]);
        targetAicQueue = reinterpret_cast<ReadyCoreFunctionQueue*>(
            dyntask->devTask.dieReadyFunctionQue.readyDieAicCoreFunctionQue[dieId]);
    }
    int32_t* opWrapList = reinterpret_cast<int32_t*>(dyntask->devTask.mixTaskData.opWrapList[funcIndex]);
    DynFuncDataCache* dynFuncDataCacheList = dyntask->GetDynFuncDataCacheList();
    DevAscendFunctionDuppedData* duppedData = dynFuncDataCacheList->At(funcIndex).duppedData;
    predcount_t* dupPredCountList = &duppedData->GetOperationCurrPredCount(0);
    auto& predInfo = duppedData->GetSource()->GetPredInfo();
    size_t totalZeroPredAIVBatchEnd = isNeedWrap ?
                                          0 :
                                          predInfo.totalZeroPredAIV & ~0x7; // wrap doesnt support batch process
    ProcessAivBatchTasks(targetAivQueue, totalZeroPredAIVBatchEnd, &duppedData->GetOperationCurrPredCount(0),
                         funcIndex);

    for (size_t opIndex = totalZeroPredAIVBatchEnd; opIndex < predInfo.totalZeroPredAIV; ++opIndex) {
        if (likely(dupPredCountList[opIndex] == 0)) {
            if (isNeedWrap && opWrapList != nullptr && opWrapList[opIndex] != -1) {
                ProcessWrapQueue(dyntask, MakeMixWrapID(funcIndex, static_cast<uint32_t>(opWrapList[opIndex])),
                                 funcIndex, opIndex, wrapQueue);
                wrapTaskNum++;
            } else {
                targetAivQueue->UnsafeEnqueue(MakeTaskID(funcIndex, opIndex));
            }
        }
    }

    // process aic task
    auto aicEnd = predInfo.totalZeroPredAIV + predInfo.totalZeroPredAIC;
    for (size_t opIndex = predInfo.totalZeroPredAIV; opIndex < aicEnd; ++opIndex) {
        if (likely(dupPredCountList[opIndex] == 0)) {
            if (isNeedWrap && opWrapList != nullptr && opWrapList[opIndex] != -1) {
                ProcessWrapQueue(dyntask, MakeMixWrapID(funcIndex, static_cast<uint32_t>(opWrapList[opIndex])),
                                 funcIndex, opIndex, wrapQueue);
                wrapTaskNum++;
            } else {
                targetAicQueue->UnsafeEnqueue(MakeTaskID(funcIndex, opIndex));
            }
        }
    }

    // process aicpu task
    auto aicpuEnd = predInfo.totalZeroPredAIV + predInfo.totalZeroPredAIC + predInfo.totalZeroPredAicpu;
    for (size_t opIndex = aicEnd; opIndex < aicpuEnd; ++opIndex) {
        if (likely(dupPredCountList[opIndex] == 0)) {
            aicpuQueue->UnsafeEnqueue(MakeTaskID(funcIndex, opIndex));
        }
    }
}

int DeviceTaskContext::BuildDynFuncData(DynDeviceTask* dyntask, uint32_t taskId, DevAscendFunctionDupped* stitchedList,
                                        uint64_t stitchedSize)
{
    size_t headerSize = sizeof(DynFuncHeader) + stitchedSize * sizeof(DynFuncData);
    auto funcHeader = workspace_->AllocateDynFuncData(headerSize);
    dyntask->dynFuncDataList = funcHeader;
    auto dyndata = &funcHeader->At(0);

    stitchedFuncNum++;
    funcHeader->funcSize = headerSize;
    funcHeader->seqNo = taskId;
    funcHeader->funcNum = stitchedSize;
    funcHeader->cceBinary = reinterpret_cast<DynFuncBin*>(const_cast<DevCceBinary*>(dyntask->cceBinary));
    funcHeader->stackWorkSpaceSize = workspace_->StandardStackWorkspacePerCore();
    funcHeader->stackWorkSpaceAddr = workspace_->StackWorkspaceAddr();
    funcHeader->startArgs = startArgs_;
    CheckAddrAligned(reinterpret_cast<uint64_t>(funcHeader->cceBinary), CCE_BINARY_MOD,
                     "#ctrl.task.pre.task.build: cceBinary address  is not aligned.");

    rootFuncNum += stitchedSize;
    for (size_t funcIdx = 0; funcIdx < stitchedSize; ++funcIdx) {
        FillOneDynFuncData(dyndata, stitchedList[funcIdx], leafFuncDataSize, leafFuncNum);
        DEV_IF_NONDEVICE { mem_dump::DumpRootMemory(taskId, static_cast<uint32_t>(funcIdx), stitchedList[funcIdx]); }
        dyndata++;
    }
    if (enableAicoreResolve_) {
        auto* drcoDyndata = &funcHeader->At(0);
        for (size_t funcIdx = 0; funcIdx < stitchedSize; ++funcIdx) {
            BuildDrcoRootFuncData(drcoDyndata, stitchedList[funcIdx]);
            drcoDyndata++;
        }
    }
    dynFuncDataSize += headerSize * sizeof(int64_t);
    return DEVICE_MACHINE_OK;
}

void DeviceTaskContext::ResolveEarlyDepends(DynDeviceTask* dyntask, size_t funcIndex, size_t opIdx, bool hasWrap,
                                            bool isMultiDie)
{
    const auto& cache = dyntask->dynFuncDataCacheList[funcIndex];
    auto cceBinary = dyntask->cceBinary;
    auto func = cache.devFunc;
    auto predList = cache.predCount;
    size_t succSize;
    auto succList = func->GetOperationDepGraphSuccAddr(static_cast<int>(opIdx), succSize);
    auto callList = cache.calleeList;
    DEV_VERBOSE_DEBUG("ResolveEarlyDepends:: funcdup %p", &dyntask->stitchedList[funcIndex]);
    for (size_t index = 0; index < succSize; ++index) {
        auto succIdx = succList[index];
        DEV_VERBOSE_DEBUG("ResolveEarlyDepends inner func %d opindex %d succfunc %d succOpIdx %u",
                          static_cast<int>(funcIndex), static_cast<int>(opIdx), static_cast<int>(funcIndex), succIdx);
        doResolve(dyntask, cceBinary[callList[succIdx]].coreType, funcIndex, succIdx, predList, hasWrap, isMultiDie);
    }

    if (func->GetOperationStitchIndex(static_cast<int>(opIdx)) == 0) {
        return;
    }
    auto& stitchList = cache.duppedData->GetOperationStitch(static_cast<int>(opIdx));
    DEV_VERBOSE_DEBUG("ResolveEarlyDepends:: stitchList %p", &stitchList);
    for (auto* node = stitchList.Head(); node != nullptr; node = node->Next()) {
        DEV_VERBOSE_DEBUG("ResolveEarlyDepends:: node %p", node);
        uint32_t listSize = node->Size();
        for (uint32_t index = 0; index < listSize; ++index) {
            uint32_t id = node->At(index);
            auto succFuncIdx = FuncID(id);
            auto succIdx = TaskID(id);
            predList = dyntask->dynFuncDataCacheList[succFuncIdx].predCount;
            callList = dyntask->dynFuncDataCacheList[succFuncIdx].calleeList;
            DEV_VERBOSE_DEBUG("ResolveEarlyDepends inner stitch func %d opindex %d succfunc %d succOpIdx %d",
                              static_cast<int>(funcIndex), static_cast<int>(opIdx), static_cast<int>(succFuncIdx),
                              static_cast<int>(succIdx));
            doResolve(dyntask, cceBinary[callList[succIdx]].coreType, succFuncIdx, succIdx, predList, hasWrap,
                      isMultiDie);
        }
    }
}

void DeviceTaskContext::ResolveEarlyDepends(DynDeviceTask* dyntask)
{
    const bool hasWrap = dyntask->devTask.mixTaskData.wrapIdNum > 0;
    const bool isMultiDie = IsMultiDie(devProg_);
    size_t funcSize = dyntask->dynFuncDataCacheListSize;
    for (size_t funcIdx = 0; funcIdx < funcSize; ++funcIdx) {
        auto func = dyntask->dynFuncDataCacheList[funcIdx].devFunc;
        auto predList = dyntask->dynFuncDataCacheList[funcIdx].predCount;
        auto& predInfo = func->GetPredInfo();
        if (predInfo.totalZeroPredHub == 0) {
            continue;
        }
        auto opIndex = predInfo.totalZeroPredAIC + predInfo.totalZeroPredAIV + predInfo.totalZeroPredAicpu;
        while (opIndex < predInfo.totalZeroPred) {
            if (predList[opIndex] == 0) {
                DEV_VERBOSE_DEBUG("ResolveEarlyDepends func %d opindex %lu", static_cast<int>(funcIdx), opIndex);
                ResolveEarlyDepends(dyntask, funcIdx, opIndex, hasWrap, isMultiDie);
            }
            opIndex++;
        }
    }
}

void DeviceTaskContext::DumpReadyQueue(DynDeviceTask* dynTask, const char* prefix)
{
    DEV_DEBUG("%s: coreFunctionCnt: %d", prefix, (int)dynTask->devTask.coreFunctionCnt);
    int aivIndex = DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIV);
    int aicIndex = DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIC);
    int aicpuIndex = DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AICPU);

    DEV_DEBUG("%s: ready queue aiv: %s [%s]", prefix, dynTask->readyQueue[aivIndex]->Str().c_str(),
              dynTask->readyQueue[aivIndex]->Dump().c_str());
    DEV_DEBUG("%s: ready queue aic: %s [%s]", prefix, dynTask->readyQueue[aicIndex]->Str().c_str(),
              dynTask->readyQueue[aicIndex]->Dump().c_str());
    DEV_DEBUG("%s: ready queue aicpu: %s [%s]", prefix, dynTask->readyQueue[aicpuIndex]->Str().c_str(),
              dynTask->readyQueue[aicpuIndex]->Dump().c_str());
}

void DeviceTaskContext::TraceFirstBatchResolve(DynDeviceTask* dynTask)
{
    for (size_t i = 0; i < READY_QUEUE_SIZE; i++) {
        if (dynTask->readyQueue[i] == nullptr)
            continue;
        for (uint32_t task : *dynTask->readyQueue[i]) {
            DEV_VERBOSE_DEBUG("#trace.ltask.resolve: tid=%d task=%lu firstBatch=1 dtaskId=%lu",
                              static_cast<int>(CTRL_CPU_THREAD_IDX), (uint64_t)task, dynTask->GetIndex());
        }
    }
}

void DeviceTaskContext::DumpDepend(DynDeviceTask* dyntask, DevAscendProgram* devProg, DevStartArgs* startArgs,
                                   const char* prefix)
{
    (void)devProg;
    (void)startArgs;
    int total = 0;
    for (size_t i = 0; i < READY_QUEUE_SIZE; i++) {
        ReadyCoreFunctionQueue* q = dyntask->readyQueue[i];
        total += q->UnsafeSize();
    }
    DEV_DEBUG("%s: ready total:%d", prefix, total);
    for (size_t i = 0; i < READY_QUEUE_SIZE; i++) {
        ReadyCoreFunctionQueue* q = dyntask->readyQueue[i];
        int k = 0;
        for (auto taskId : *q) {
            uint32_t dupIndex = FuncID(taskId);
            uint32_t opIndex = TaskID(taskId);
            DEV_DEBUG("%s: ready %d-%d:L(%d,%d,%d)\n", prefix, (int)i, k++, (int)dyntask->GetDynFuncDataList()->seqNo,
                      (int)dupIndex, (int)opIndex);
        }
    }
    DEV_DEBUG("%s: workspace:%llx", prefix, (unsigned long long)startArgs->contextWorkspaceAddr);
    for (size_t i = 0; i < startArgs->inputTensorSize; i++) {
        DEV_DEBUG("%s: input-%d:%llx", prefix, (int)i, (unsigned long long)startArgs->GetInputTensor(i).address);
    }
    for (size_t i = 0; i < startArgs->outputTensorSize; i++) {
        DEV_DEBUG("%s: output-%d:%llx", prefix, (int)i, (unsigned long long)startArgs->GetOutputTensor(i).address);
    }
    std::unordered_map<uint64_t, AddressDescriptor> cacheInputOutputDict;
    DevControlFlowCache::RelocBuildInputOutputDesc(cacheInputOutputDict, startArgs);
    RelocRange relocWorkspace(startArgs->contextWorkspaceAddr, 0);

    DynFuncHeader* dynFuncDataList = dyntask->GetDynFuncDataList();
    int deviceIndex = dynFuncDataList->seqNo;
    DynFuncDataCache* dynFuncDataCacheList = dyntask->GetDynFuncDataCacheList();
    for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); dupIndex++) {
        DynFuncData& dynFuncData = dynFuncDataList->At(dupIndex);
        DynFuncDataCache& dynFuncDataCache = dynFuncDataCacheList->At(dupIndex);

        DevAscendFunctionDuppedData* duppedData = dynFuncDataCache.duppedData;

        predcount_t* pred = &duppedData->GetOperationCurrPredCount(0);
        for (size_t opIndex = 0; opIndex < duppedData->GetOperationSize(); opIndex++) {
            DEV_DEBUG("%s: L(%d,%d,%d) pred:%d\n", prefix, (int)deviceIndex, (int)dupIndex, (int)opIndex,
                      (int)pred[opIndex]);
        }
        for (size_t stitchIndex = 1; stitchIndex < duppedData->GetStitchSize(); stitchIndex++) {
            DevAscendFunctionDuppedStitchList stitchList = duppedData->GetStitch(stitchIndex);
            stitchList.ForEach([&](int succTaskId) {
                uint32_t succDupIndex = FuncID(succTaskId);
                uint32_t succOpIndex = TaskID(succTaskId);
                DEV_DEBUG("%s: R(%d,%d).succ-%d: L(%d,%d,%d)\n", prefix, (int)deviceIndex, (int)dupIndex,
                          (int)stitchIndex, (int)deviceIndex, (int)succDupIndex, (int)succOpIndex);
            });
        }
        for (size_t exprIndex = 0; exprIndex < duppedData->GetExpressionSize(); exprIndex++) {
            DEV_DEBUG("%s: R(%d,%d).expr-%d: %lld\n", prefix, (int)deviceIndex, (int)dupIndex, (int)exprIndex,
                      (long long)duppedData->GetExpression(exprIndex));
        }
        for (size_t incastIndex = 0; incastIndex < duppedData->GetIncastSize(); incastIndex++) {
            AddressDescriptor addr = duppedData->GetIncastAddress(incastIndex);
            AddressDescriptor addrDesc = addr;
            DevControlFlowCache::RelocDescToCache(addrDesc, relocWorkspace, cacheInputOutputDict);

            DEV_DEBUG("%s: R(%d,%d).incast-%d: 0x%llx - 0x%llx\n", prefix, (int)deviceIndex, (int)dupIndex,
                      (int)incastIndex, (unsigned long long)addrDesc.GetAddressValue(),
                      (unsigned long long)addr.GetAddressValue());
        }
        for (size_t outcastIndex = 0; outcastIndex < duppedData->GetOutcastSize(); outcastIndex++) {
            AddressDescriptor addr = duppedData->GetOutcastAddress(outcastIndex);
            AddressDescriptor addrDesc = addr;
            DevControlFlowCache::RelocDescToCache(addrDesc, relocWorkspace, cacheInputOutputDict);
            DEV_DEBUG("%s: R(%d,%d).outcast-%d: 0x%llx - 0x%llx\n", prefix, (int)deviceIndex, (int)dupIndex,
                      (int)outcastIndex, (unsigned long long)addrDesc.GetAddressValue(),
                      (unsigned long long)addr.GetAddressValue());
        }
        DEV_DEBUG("%s: R(%d,%d).workspace: 0x%llx\n", prefix, (int)deviceIndex, (int)dupIndex,
                  (unsigned long long)duppedData->GetRuntimeWorkspace());
        DEV_DEBUG("%s: R(%d,%d).outcastWorkspace: 0x%llx\n", prefix, (int)deviceIndex, (int)dupIndex,
                  (unsigned long long)duppedData->GetRuntimeOutcastWorkspace());

        DEV_DEBUG("%s: R(%d,%d).opAttrList: 0x%llx\n", prefix, (int)deviceIndex, (int)dupIndex,
                  (unsigned long long)dynFuncData.opAttrs);
        DEV_DEBUG("%s: R(%d,%d).opAttrList:Dupped: 0x%llx\n", prefix, (int)deviceIndex, (int)dupIndex,
                  (unsigned long long)duppedData->GetSource()->GetSymoffset(0));

        DEV_DEBUG("%s: R(%d,%d).opAttrOffsetList: 0x%llx\n", prefix, (int)deviceIndex, (int)dupIndex,
                  (unsigned long long)dynFuncData.opAtrrOffsets);
        DEV_DEBUG("%s: R(%d,%d).opAttrOffsetList:Dupped: 0x%llx\n", prefix, (int)deviceIndex, (int)dupIndex,
                  (unsigned long long)duppedData->GetSource()->GetOpAttrOffsetAddr());

        DEV_DEBUG("%s: R(%d,%d).exprTbl: 0x%llx\n", prefix, (int)deviceIndex, (int)dupIndex,
                  (unsigned long long)dynFuncData.exprTbl);
        DEV_DEBUG("%s: R(%d,%d).exprTbl:Dupped: 0x%llx\n", prefix, (int)deviceIndex, (int)dupIndex,
                  (unsigned long long)duppedData->GetExpressionAddr());

        DEV_DEBUG("%s: R(%d,%d).rawTensorDesc: 0x%llx\n", prefix, (int)deviceIndex, (int)dupIndex,
                  (unsigned long long)dynFuncData.rawTensorDesc);
        DEV_DEBUG("%s: R(%d,%d).rawTensorDesc:Dupped: 0x%llx\n", prefix, (int)deviceIndex, (int)dupIndex,
                  (unsigned long long)duppedData->GetSource()->GetRawTensorDesc(0));

        DEV_DEBUG("%s: R(%d,%d).rawTensorDesc: 0x%llx\n", prefix, (int)deviceIndex, (int)dupIndex,
                  (unsigned long long)dynFuncData.rawTensorAddr);
        DEV_DEBUG("%s: R(%d,%d).rawTensorDesc:Dupped: 0x%llx\n", prefix, (int)deviceIndex, (int)dupIndex,
                  (unsigned long long)&duppedData->GetIncastAddress(0));
    }
}

int DeviceTaskContext::BuildDeviceTaskDataAndReadyQueue(DynDeviceTask* dyntask, uint32_t taskId,
                                                        DevAscendProgram* devProg)
{
    int result = DEVICE_MACHINE_OK;
    dyntask->cceBinary = devProg->GetCceBinary(0);
    dyntask->aicpuLeafBinary = devProg->GetAicpuLeafBinary(0);
    DeviceStitchContext::CheckStitch(dyntask);

    DEV_VERBOSE_DEBUG("Build ready queue");
    PerfBegin(PERF_EVT_READY_QUEUE);
    result = BuildReadyQueue(dyntask, devProg);
    if (unlikely(result != DEVICE_MACHINE_OK)) {
        return DEVICE_MACHINE_ERROR;
    }
    PerfEnd(PERF_EVT_READY_QUEUE);

    PerfBegin(PERF_EVT_RESOLVE_EARLY);
    ResolveEarlyDepends(dyntask);
    PerfEnd(PERF_EVT_RESOLVE_EARLY);

    DEV_VERBOSE_DEBUG("Build func data");
    PerfBegin(PERF_EVT_CORE_FUNCDATA);
    result = BuildDynFuncData(dyntask, taskId, &dyntask->stitchedList[0], dyntask->stitchedList.size());
    if (unlikely(result != DEVICE_MACHINE_OK)) {
        return DEVICE_MACHINE_ERROR;
    }
    PerfEnd(PERF_EVT_CORE_FUNCDATA);
    DEV_DEBUG("Finish build a new device task");

    if (enableAicoreResolve_) {
        DispatchReadyQueueToCores(dyntask, devProg);
        DispatchDieReadyQueueToCores(dyntask, devProg);
        npu::tile_fwk::DrcoRootFuncListPrecountPerCoreTasks(dyntask->drcoRootFuncList, devProg->devArgs.nrValidAic,
                                                            dyntask->drcoPrecountExecuted,
                                                            dyntask->drcoPrecountFinishFlag);
        DEV_IF_NONDEVICE { __atomic_store_n(&dyntask->drcoRootFuncList->devTaskFinished, 1u, __ATOMIC_RELEASE); }
    }

#ifndef __DEVICE__
    DEV_IF_NONDEVICE
    {
        if (devProg->devArgs.enableEslModel ||
            config::GetDebugOption<int64_t>(CFG_RUNTIME_DBEUG_MODE) != CFG_DEBUG_NONE ||
            config::GetRuntimeOption<int64_t>(CFG_RUN_MODE) == CFG_RUN_MODE_SIM) {
            dyntask->DumpTopo(devProg->devArgs.enableVFFusion);
        }
    }

    HandleEslModelTransmission(devProg, dyntask, dynFuncDataSize);
#endif

#if DEBUG_INFINITE_LIFETIME
    DEV_IF_DEVICE { dyntask->DumpTensorAddrInfo(workspace_->DumpTensorWsBaseAddr(), workspace_->DumpTensorWsSize()); }
#endif
    DEV_IF_VERBOSE_DEBUG { dyntask->DumpLeafs(); }

    DEV_IF_DEBUG
    {
        int funcIdx = 0;
        for (auto& func : dyntask->stitchedList) {
            DEV_DEBUG_SPLIT("func %d %s.", funcIdx, func.DumpDyn(funcIdx, dyntask->cceBinary).c_str());
            DEV_DEBUG_SPLIT("func %d %s.", funcIdx, func.DumpMainBlockFlag().c_str());
            funcIdx++;
            (void)func;
        }
    }
    dyntask->stitchedList.clear();
    return result;
}
} // namespace npu::tile_fwk::dynamic
