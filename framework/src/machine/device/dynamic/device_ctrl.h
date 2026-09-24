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
 * \file device_ctrl.h
 * \brief
 */

#pragma once

#include "device_common.h"
#include <cstdint>
#include <cstdlib>
#include "device_utils.h"
#include "device_perf.h"
#include "machine/device/dynamic/context/device_execute_context.h"
#include "machine/utils/dynamic/dev_tensor_creator.h"
#include "machine/utils/dynamic/dev_encode_tensor.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/utils/device_log.h"
#include "machine/utils/barrier.h"
#include "machine/device/dynamic/perf_event_sampler.h"
#include "machine/device/dynamic/aicore_prof.h"
#include "device_trace.h"

#ifdef __USE_CUSTOM_CTRLFLOW__
extern "C" __attribute__((visibility("default"))) void* GetCtrlFlowFunc();
#endif

extern "C" __attribute__((weak)) int dlog_setlevel(int32_t moduled, int32_t level, int32_t enableEvent);

namespace npu::tile_fwk::dynamic {

class DeviceCtrlMachine {
public:
    DeviceTaskCtrl* InitTaskCtrl(int idx, DeviceTask* devTask, DeviceExecuteContext* ctx)
    {
        if (ctx == nullptr) {
            DEV_ERROR(CtrlErr::ROOT_ALLOC_CTX_NULL,
                      "#ctrl.push.init_dtask: Init Task control failed, which ctx is null.");
            return nullptr;
        }
        if (idx < 0 || idx >= MAX_DEVICE_TASK_NUM) {
            DEV_ERROR(CtrlErr::CTRL_FLOW_EXEC_FAILED,
                      "#ctrl.push.init_dtask: Init Task control failed, idx=%d out of bounds.", idx);
            return nullptr;
        }
        auto taskCtrl = &GetTaskCtrlInPool(idx);
        taskCtrl->BindTask(devTask);
        taskCtrl->InitSchNumCnt();
        devTask->aicoreModel = reinterpret_cast<uint64_t>(ctx->aicoreModel);
        if (ctx->costModelData != nullptr) {
            devTask->costModelData = reinterpret_cast<uint64_t>(ctx->costModelData);
        }
        return taskCtrl;
    }

    int AllocNewTaskCtrl()
    {
        uint32_t& taskCtrlIndex = devStartArgs_->devCtrlState.taskCtrlIndex;

        TIMEOUT_CHECK_INIT(devStartArgs_->devProg->devArgs.archInfo, TIMEOUT_1MIN);

        while (true) {
            if (taskCtrlIndex == MAX_DEVICE_TASK_NUM)
                taskCtrlIndex = 0;
            if (!GetTaskCtrlInPool(taskCtrlIndex).IsNotFree()) {
                return taskCtrlIndex++;
            }
            taskCtrlIndex++;

            __PYPTO_TIMEOUT_CHECK(CtrlErr::CTRL_ALLOC_TIMEOUT, return DEVICE_MACHINE_ERROR,
                                  "#ctrl.alloc: AllocNewTaskCtrl, taskCtrlIndex=%u.", taskCtrlIndex);
        }
    }

    bool SameParallelIterTaskCtrl(DeviceTaskCtrl* srcTaskCtrl, DeviceTaskCtrl* dstTaskCtrl)
    {
        if (!srcTaskCtrl->SupportParallel() || !dstTaskCtrl->SupportParallel()) {
            return false;
        }

        if (srcTaskCtrl->ParallelForId() != dstTaskCtrl->ParallelForId() ||
            srcTaskCtrl->ParallelIterId() != dstTaskCtrl->ParallelIterId()) {
            return false;
        }

        return true;
    }

    int PushTask(DynDeviceTask* dynTask, DeviceExecuteContext* ctx)
    {
        DEV_VERBOSE_DEBUG("#trace.dtask.built: dtaskId %lu", dynTask->GetIndex());

        if (AicoreResolveEnabled()) {
            if (!ctx->devProg->ctrlFlowCacheAnchor->IsRecording()) {
                DrcoDeviceTaskReadyQueueAppend(dynTask->dynFuncDataList, dynTask->drcoRootFuncList);
                DEV_DEBUG("DRCO append task=%p root=%p qtail=%u", dynTask->dynFuncDataList, dynTask->drcoRootFuncList,
                          GetDeviceTaskReadyQueue().tail);
            }
            return 0;
        }

        auto idx = AllocNewTaskCtrl();
        if (idx < 0) {
            DEV_ERROR(CtrlErr::CTRL_ALLOC_TIMEOUT, "#ctrl.push.alloc: AllocNewTaskCtrl failed, idx=%d.", idx);
            return idx;
        }
        bool appendLastTaskCtrl = false;
        DeviceTaskCtrl* newTaskCtrl = InitTaskCtrl(idx, &dynTask->devTask, ctx);
        if (lastTaskCtrl_ && lastTaskCtrl_->SupportParallel()) {
            if (SameParallelIterTaskCtrl(lastTaskCtrl_, newTaskCtrl)) {
                lastTaskCtrl_->existNextSameIterTask = true;
                lastTaskCtrl_->nextSameIterTaskCtrl = reinterpret_cast<uint64_t>(newTaskCtrl);
                appendLastTaskCtrl = true;
            } else {
                lastTaskCtrl_->existNextSameIterTask = false; // Correct the flag
            }
        }
        lastTaskCtrl_ = newTaskCtrl;

        if (dynTask->IsParallelSameIterLastDevTask()) {
            lastTaskCtrl_->existNextSameIterTask = false;
        }

        if (!appendLastTaskCtrl && !ctx->devProg->ctrlFlowCacheAnchor->IsRecording()) {
            for (uint32_t i = 0; i < GetScheAicpuNum(); ++i) {
                GetTaskQueue(i).Enqueue(newTaskCtrl);
            }
        }

        return idx;
    }

    void StopAicoreManager()
    {
        if (AicoreResolveEnabled()) {
            DrcoDeviceTaskReadyQueueAppend(nullptr, nullptr);
        } else {
            for (uint32_t i = 0; i < GetScheAicpuNum(); ++i) {
                GetTaskQueue(i).Enqueue(nullptr);
            }
        }
    }

    void RegisterTaskInspector(DeviceTaskInspectorEntry inspectorEntry, void* inspector)
    {
        inspectorEntry_ = inspectorEntry;
        inspector_ = inspector;
    }

    void InitTaskPipeWithSched(DevAscendProgram* devProg)
    {
        if (AicoreResolveEnabled()) {
            GetDeviceTaskReadyQueue().Reset();
        } else {
            for (uint32_t i = 0; i < devProg->devArgs.scheCpuNum; ++i) {
                GetTaskQueue(i).ResetEmpty();
            }
        }
    }

    void InitCtrlFlowCache(DevAscendProgram* devProg, DevControlFlowCache* ctrlFlowCache, DevStartArgs* devStartArgs,
                           bool firstInit)
    {
        DevControlFlowCache* devCtrlFlowCache = nullptr;
        devCtrlFlowCache = &devProg->controlFlowCache;
        if (devProg->controlFlowCache.isRecording) {
            DEV_INFO("Init dev program cache");
            devProg->controlFlowCache.contextWorkspaceAddr = devStartArgs->contextWorkspaceAddr;
        } else if (ctrlFlowCache != nullptr) {
            DEV_INFO("Init independent anchor program cache %p.", ctrlFlowCache);
            if (ctrlFlowCache->isRecording) {
                DEV_ASSERT_MSG(CtrlErr::CTRL_FLOW_EXEC_FAILED, !devProg->controlFlowCache.isRecording,
                               "#ctrl.flow.exec: dev program ctr cache should not record");
                ctrlFlowCache->contextWorkspaceAddr = devStartArgs->contextWorkspaceAddr;
            } else {
                DEV_ASSERT_MSG(
                    CtrlErr::CTRL_FLOW_EXEC_FAILED,
                    !devProg->controlFlowCache.isActivated && ctrlFlowCache->isActivated,
                    "#ctrl.flow.exec: should not active dev program cache and independent ctrl cache at same time");
            }
            devCtrlFlowCache = ctrlFlowCache;
            if (devCtrlFlowCache->isActivated && !devCtrlFlowCache->isRelocMetaDev) {
                DEV_INFO("ControlFlowCache: reloc meta cache");
                devCtrlFlowCache->isRelocMetaDev = true;
                devCtrlFlowCache->RelocMetaCache(0, reinterpret_cast<uint64_t>(devCtrlFlowCache));
            }
        }

        DEV_INFO("ControlFlowCache: deviceTaskCount=%d, firstInit=%d.", (int)devCtrlFlowCache->deviceTaskCount,
                 (int)firstInit);

        /* Currently, sche does not use ctrlFlowCacheAnchor, so that we could record it in devProgram.
         * However, it should be moved into the execute context. */
        devProg->ctrlFlowCacheAnchor = devCtrlFlowCache;
        if (devCtrlFlowCache->deviceTaskCount == 0) {
            if (!firstInit) {
                devProg->ResetRerun(); // Clean the dirty data of cell match table from the last launch
            }
            DEV_INFO("ControlFlowCache: cache has no devtask, ignore it");
            return;
        }

        if (devCtrlFlowCache->IsActivatedCache(devStartArgs)) {
            DEV_INFO("ControlFlowCache: activated");
            // 维测：cache 生效前最后一个卡点，记录最终参与重定位的 blob 基址对齐情况与来路。
            // isEmbedded 区分"内嵌在 DevAscendProgram 里的 controlFlowCache"与"外部传入的独立 blob"，
            // 便于把错位直接归因到具体产生点（配合各 site= 日志）。
            DEV_INFO("#ctrl.cache.align: site=InitCtrlFlowCache.consume base=%p embedded=%p incoming=%p "
                     "isEmbedded=%d align64=%lu progAlign64=%lu %s",
                     reinterpret_cast<void*>(devCtrlFlowCache), reinterpret_cast<void*>(&devProg->controlFlowCache),
                     reinterpret_cast<void*>(ctrlFlowCache),
                     static_cast<int>(devCtrlFlowCache == &devProg->controlFlowCache),
                     static_cast<unsigned long>(reinterpret_cast<uintptr_t>(devCtrlFlowCache) %
                                                npu::tile_fwk::DUPPED_STITCH_NODE_ALIGN),
                     static_cast<unsigned long>(reinterpret_cast<uintptr_t>(devProg) %
                                                npu::tile_fwk::DUPPED_STITCH_NODE_ALIGN),
                     reinterpret_cast<uintptr_t>(devCtrlFlowCache) % npu::tile_fwk::DUPPED_STITCH_NODE_ALIGN == 0 ?
                         "OK" :
                         "MISALIGNED");
            // Actual run
            if (!devCtrlFlowCache->isRelocDataDev) {
                devCtrlFlowCache->isRelocDataDev = true;
                devCtrlFlowCache->TaskAddrRelocProgramAndCtrlCache(0, 0, reinterpret_cast<uint64_t>(devProg),
                                                                   reinterpret_cast<uint64_t>(devCtrlFlowCache));
                devCtrlFlowCache->RuntimeAddrRelocProgram(0, reinterpret_cast<uint64_t>(devProg));
            }
            if (devCtrlFlowCache->workspaceAddr != devStartArgs->contextWorkspaceAddr) {
                devCtrlFlowCache->workspaceAddr = devStartArgs->contextWorkspaceAddr;
                devCtrlFlowCache->TaskAddrRestoreWorkspace();
                devCtrlFlowCache->TaskAddrRelocWorkspace(0, devStartArgs->contextWorkspaceAddr, devStartArgs);
            }
            if (!devCtrlFlowCache->IsActivatedFullCache(devStartArgs)) {
                devProg->ResetRerun();
            }
        }
    }

    bool InitDevProgram(DevAscendProgram* devProg)
    {
        bool firstInit = false;
        if (devProg->controlFlowBinaryAddr == nullptr) {
            devProg->RelocProgram(0, reinterpret_cast<uint64_t>(devProg), true);

            RuntimeDataRingBufferHead* ringBufferHead = reinterpret_cast<RuntimeDataRingBufferHead*>(
                devProg->GetRuntimeDataList());
            ringBufferHead->Initialize(devProg->GetDeviceRuntimeOffset().size, devProg->GetDeviceRuntimeOffset().count,
                                       devProg->devArgs.archInfo);

            devProg->runtimeDataRingBufferInited = true;
            firstInit = true;
        }
        memBarrier();
#ifdef __USE_CUSTOM_CTRLFLOW__
        if (devProg->controlFlowBinaryAddr == nullptr) {
            DEV_INFO("Use built in ctrl flow func.");
            devProg->controlFlowBinaryAddr = GetCtrlFlowFunc();
        }
#else
        // Resolve CF by DevProg hash: already in pool and not overwritten -> skip memcpy.
        {
            auto execProg = DeviceExecuteProgram(devProg, nullptr);
            devProg->controlFlowBinaryAddr = execProg.GetControlFlowEntry();
        }
#endif
        return firstInit;
    }

    int InitDyn(DeviceKernelArgs* kargs)
    {
        DEV_INFO("AscendCppDyInitTask begin");

        DevAscendProgram* devProg = PtrToPtr<int64_t, DevAscendProgram>(kargs->cfgdata);
        ApplyDynamicCellMatchDescPatchesFromLaunchArgs(devProg, kargs->inputs);
        auto& tensorBudget = devProg->memBudget.tensor;
        auto& metadataBudget = devProg->memBudget.metadata;
        if (kargs->maxDynamicAssembleOutcastMem != 0) {
            tensorBudget.maxDynamicAssembleOutcastMem = kargs->maxDynamicAssembleOutcastMem;
        }
        if (kargs->maxDynamicCellMatchTableMem != 0) {
            metadataBudget.maxDynamicCellMatchTableMem = kargs->maxDynamicCellMatchTableMem;
            uint64_t totalDynamicCellMatchSlotNum = metadataBudget.dynamicCellMatchSlotNum;
            metadataBudget.dynamicCellMatch = totalDynamicCellMatchSlotNum * metadataBudget.maxDynamicCellMatchTableMem;
            if (kargs->runtimeDynamicCellMatchCapacity != 0) {
                devProg->devArgs.dynamicCellMatchAddr = kargs->runtimeDynamicCellMatchAddr;
                devProg->devArgs.dynamicCellMatchCapacity = kargs->runtimeDynamicCellMatchCapacity;
            }
        }
        PerfBegin(PERF_EVT_INIT);
        bool firstInit = InitDevProgram(devProg);
        PerfEnd(PERF_EVT_INIT);

        RuntimeDataRingBufferHead* ringBufferHead = devProg->GetRuntimeDataList();

        DEV_INFO("AllocatePrepare begin runtime data: %lu, %lu %lu", ringBufferHead->GetIndexFinished(),
                 ringBufferHead->GetIndexPending(), ringBufferHead->GetRuntimeDataCount());
        DevStartArgs* devStartArgs = reinterpret_cast<DevStartArgs*>(ringBufferHead->AllocatePrepare());
        DEV_INFO("AllocatePrepare end");

        devStartArgs->syncFlag = 0;
        devStartArgs->InitProgram(devProg, reinterpret_cast<uint64_t>(devStartArgs));
        // Bind 控核结果到本 round 的 ring slot，避免改写共享 DevProg.devArgs.nrValidAic（多 shape 重叠会竞态）
        const uint32_t capacityNrValidAic = devProg->devArgs.nrValidAic;
        const uint32_t capacityScheCpuNum = devProg->devArgs.scheCpuNum;
        devStartArgs->nrValidAic = ResolveRoundNrValidAic(kargs->parameter.ctrlBlockNum, capacityNrValidAic);
        devStartArgs->scheCpuNum = ResolveRoundScheCpuNum(devStartArgs->nrValidAic, capacityScheCpuNum,
                                                          devProg->devArgs.archInfo);
        DEV_INFO("Round topology: ctrlBlockNum=%lu nrValidAic=%u scheCpuNum=%u (capacity aic=%u sche=%u)",
                 kargs->parameter.ctrlBlockNum, devStartArgs->nrValidAic, devStartArgs->scheCpuNum, capacityNrValidAic,
                 capacityScheCpuNum);
        devStartArgs->devCtrlState.schAicpuNum = devStartArgs->scheCpuNum;
        devStartArgs->devCtrlState.taskCtrlIndex = 0;
        devStartArgs->devScheState.threadIdx = CTRL_THREAD_INDEX;
        devStartArgs->devScheState.finished = 0;

        devStartArgs_ = devStartArgs;

        InitTaskPipeWithSched(devProg);

        devStartArgs->controlFlowEntry = devProg->controlFlowBinaryAddr;

        uint64_t inputSize = *kargs->inputs;
        uint64_t outputSize = *(kargs->inputs + 1);
        auto inputPtr = PtrToPtr<int64_t, DevTensorData>(kargs->inputs + DEV_TENSOR_DATA_OFFSET);
        DEV_INFO("inputSize=%lu, outputSize=%lu, tensorListPtr=%p.", inputSize, outputSize, inputPtr);
        devStartArgs->devTensorList = inputPtr;
        devStartArgs->inputTensorSize = static_cast<uint64_t>(inputSize);
        devStartArgs->outputTensorSize = static_cast<uint64_t>(outputSize);

        devStartArgs->contextWorkspaceAddr = PtrToValue(kargs->workspace);
        devStartArgs->contextWorkspaceSize = devProg->workspaceSize;

        devStartArgs->inputSymbolList = nullptr;
        devStartArgs->inputSymbolSize = 0;
        devStartArgs->commGroupNum = (kargs->commContexts == nullptr) ? 0 : static_cast<uint64_t>(*kargs->commContexts);
        devStartArgs->commContexts = (devStartArgs->commGroupNum == 0) ? nullptr : kargs->commContexts + 1;

        DevControlFlowCache* ctrlFlowCacheBase = reinterpret_cast<DevControlFlowCache*>(kargs->ctrlFlowCache);
        DevControlFlowCache* ctrlFlowCache;
        if (ctrlFlowCacheBase == nullptr) {
            ctrlFlowCache = ctrlFlowCacheBase;
        } else if (ctrlFlowCacheBase->IsRecording()) {
            ctrlFlowCache = ctrlFlowCacheBase;
        } else {
            ctrlFlowCache = reinterpret_cast<DevControlFlowCache*>(reinterpret_cast<uint8_t*>(kargs->ctrlFlowCache) +
                                                                   ctrlFlowCacheBase->usedCacheSize *
                                                                       ringBufferHead->GetIndexPendingIndex());
        }
        InitCtrlFlowCache(devProg, ctrlFlowCache, devStartArgs, firstInit);

        ringBufferHead->AllocateSubmit();
        lastTaskCtrl_ = nullptr;
        DEV_INFO("AscendCppDyInitTask done.");
        return 0;
    }

    int ExecDyn(npu::tile_fwk::DeviceKernelArgs* args)
    {
        DEV_INFO("start control flow.");
        auto devProg = PtrToPtr<int64_t, DevAscendProgram>(args->cfgdata);
        auto devStartArgs = (DevStartArgs*)devProg->GetRuntimeDataList()->GetRuntimeDataPending();

        DeviceExecuteContext ctx(devStartArgs);
        ctx.costModelData = reinterpret_cast<CostModel::ModelData*>(args->costmodeldata);
        ctx.aicoreModel = args->aicoreModel;
        PerfBegin(PERF_EVT_EXEC_DYN);
        PerfBegin(PERF_EVT_CONTROL_FLOW_CALL);
        int ret = ctx.GELaunch(devStartArgs, [this](DynDeviceTask* dynTask, DeviceExecuteContext* exeCtx) {
            if (unlikely(inspectorEntry_ != nullptr)) {
                inspectorEntry_(inspector_, exeCtx, dynTask);
            }
            DEV_IF_DEBUG { DumpTask(dynTask->GetIndex(), (DeviceTask*)dynTask, true); }
            PushTask(dynTask, exeCtx);
        });
        PerfEnd(PERF_EVT_CONTROL_FLOW_CALL);
        if (ret != DEVICE_MACHINE_OK) {
            DeviceTrace::GetInstance().ReportTraceMsg();
            return ret;
        }
        DEV_INFO("end control flow.");
        PerfBegin(PERF_EVT_STAGE_STOP_AICORE);
        if (!devProg->ctrlFlowCacheAnchor->IsRecording()) {
            // host cache not need to enque
            StopAicoreManager();
        }
        PerfEnd(PERF_EVT_STAGE_STOP_AICORE);
        DEV_INFO("aicore manager stopped");
        PerfEnd(PERF_EVT_EXEC_DYN);
#if ENABLE_PERF_EVT
        ctx.ShowStats();
        PerfEvtMgr::Instance().Dump();
        PerfettoMgr::Instance().Dump("/tmp/perfetto.txt");
#endif
        return ret;
    }

    int EntryInit(DeviceKernelArgs* kargs)
    {
        PerfBegin(PERF_EVT_DEVICE_MACHINE_INIT_DYN);
        if (kargs == nullptr) {
            return -1;
        }
        if (kargs->inputs == nullptr || kargs->cfgdata == nullptr) {
            DEV_ERROR(DevCommonErr::NULLPTR, "#ctrl.init: Args has null in inputs[%p] work[%p] or cfg[%p].\n",
                      kargs->inputs, kargs->workspace, kargs->cfgdata);
            return -1;
        }
        InitDyn(kargs);
        PerfEnd(PERF_EVT_DEVICE_MACHINE_INIT_DYN);
        return 0;
    }

    int EntryMain(DeviceKernelArgs* kargs)
    {
        int rc = ExecDyn(kargs);
        if (rc == npu::tile_fwk::dynamic::DEVICE_MACHINE_OK) {
            return 0;
        }
        return -1;
    }

private:
    static void DumpTaskDetail(DeviceTask* devTask, bool isDyn)
    {
        DEV_DEBUG("===== ready aic func =====");
        ReadyCoreFunctionQueue* readyFunc = reinterpret_cast<ReadyCoreFunctionQueue*>(devTask->readyAicCoreFunctionQue);
        DEV_DEBUG("aic [%s]", readyFunc->Dump().c_str());

        DEV_DEBUG("===== ready aiv func =====");
        readyFunc = reinterpret_cast<ReadyCoreFunctionQueue*>(devTask->readyAivCoreFunctionQue);
        DEV_DEBUG("aiv [%s]", readyFunc->Dump().c_str());

        DEV_DEBUG("===== ready aicpu func =====");
        readyFunc = reinterpret_cast<ReadyCoreFunctionQueue*>(devTask->readyAicpuFunctionQue);
        DEV_DEBUG("aicpu [%s]", readyFunc->Dump().c_str());

        if (isDyn) {
            DEV_DEBUG("===== dyn info =====");
            auto dyntask = PtrToPtr<DeviceTask, DynDeviceTask>(devTask);
            int funcIdx = 0;
            for (auto& func : dyntask->stitchedList) {
                DEV_DEBUG("funcIdx=%d, %s.", funcIdx, func.DumpDyn(funcIdx, dyntask->cceBinary).c_str());
                funcIdx++;
                (void)func;
            }
        } else {
            auto coreFunc = reinterpret_cast<CoreFunctionWsAddr*>(devTask->coreFuncData.coreFunctionWsAddr);
            DEV_DEBUG("===== core func =====");
            for (uint64_t i = 0; i < devTask->coreFunctionCnt; i++) {
                DEV_DEBUG("taskId[%lu]: binAddr=%#lx, invokeEntry=%#lx, topo=%#lx.", i, coreFunc[i].functionBinAddr,
                          coreFunc[i].invokeEntryAddr, coreFunc[i].topoAddr);
                auto topo = reinterpret_cast<CoreFunctionTopo*>(coreFunc[i].topoAddr);
                DEV_DEBUG("  topo: coreType=%lu, psgId=%lu, readyCount=%ld, depNum=%lu.", topo->coreType, topo->psgId,
                          topo->readyCount, topo->depNum);
                (void)topo;
            }
            DEV_DEBUG("===== ready state =====");
            auto readyState = reinterpret_cast<CoreFunctionReadyState*>(devTask->coreFunctionReadyStateAddr);
            for (uint64_t i = 0; i < devTask->coreFunctionCnt; i++) {
                DEV_DEBUG("taskId[%lu]: readyCount=%ld, coreType=%lu.", i, readyState[i].readyCount,
                          readyState[i].coreType);
            }
            (void)(readyState);
        }
        DEV_DEBUG("===== dev task end =====");
    }

    static void DumpTask(int64_t taskId, DeviceTask* devTask, bool isDyn)
    {
        DEV_DEBUG("taskId=%ld, devTask=%p, isDyn=%d.", taskId, devTask, static_cast<int>(isDyn));
        if (devTask == nullptr) {
            return;
        }

        DEV_DEBUG("devtask { coreFunctionCnt=%lu, readyStateAddr=%#lx, "
                  "readyAicQue=%#lx, readyAivQue=%#lx, readyAicpuQue=%#lx, "
                  "coreFuncWsAddr=%#lx, stackWsAddr=%#lx, stackWsSize=%lu }.",
                  devTask->coreFunctionCnt, devTask->coreFunctionReadyStateAddr, devTask->readyAicCoreFunctionQue,
                  devTask->readyAivCoreFunctionQue, devTask->readyAicpuFunctionQue,
                  devTask->coreFuncData.coreFunctionWsAddr, devTask->coreFuncData.stackWorkSpaceAddr,
                  devTask->coreFuncData.stackWorkSpaceSize);

        DumpTaskDetail(devTask, isDyn);
    }

private:
    DeviceTaskCtrl& GetTaskCtrlInPool(int index) { return devStartArgs_->deviceRuntimeDataDesc.taskCtrlPool[index]; }
    DeviceTaskCtrlQueue& GetTaskQueue(int index) { return devStartArgs_->deviceRuntimeDataDesc.taskQueueList[index]; }
    npu::tile_fwk::DrcoDeviceTaskReadyQueue& GetDeviceTaskReadyQueue()
    {
        return *devStartArgs_->drcoDeviceTaskReadyQueue;
    }
    int32_t DrcoDeviceTaskReadyQueueAppend(npu::tile_fwk::DynFuncHeader* dynFuncDataList,
                                           npu::tile_fwk::DrcoRootFuncList* rootFuncList)
    {
        TIMEOUT_CHECK_INIT(devStartArgs_->devProg->devArgs.archInfo, TIMEOUT_1MIN);
        while (!GetDeviceTaskReadyQueue().TryAppend(dynFuncDataList, rootFuncList)) {
            __PYPTO_TIMEOUT_CHECK_EXIT_ONLY(CtrlErr::CTRL_ALLOC_TIMEOUT, return DEVICE_MACHINE_ERROR,
                                            "#drco.queue.full: DrcoDeviceTaskReadyQueue full, tail=%u head=%u.",
                                            GetDeviceTaskReadyQueue().tail,
                                            __atomic_load_n(&GetDeviceTaskReadyQueue().head, __ATOMIC_ACQUIRE));
        }
        return DEVICE_MACHINE_OK;
    }
    bool AicoreResolveEnabled() const
    {
        return devStartArgs_ != nullptr && devStartArgs_->devProg != nullptr &&
               devStartArgs_->devProg->devArgs.enableAicoreResolve;
    }
    uint32_t GetScheAicpuNum()
    {
        uint64_t arbitratedScehNum = devStartArgs_->devCtrlState.arbitratedScehNum.load();
        if (arbitratedScehNum != 0 && arbitratedScehNum < devStartArgs_->devCtrlState.schAicpuNum) {
            return arbitratedScehNum;
        }
        return devStartArgs_->devCtrlState.schAicpuNum;
    }

private:
    DevStartArgs* devStartArgs_{nullptr};
    DeviceTaskCtrl* lastTaskCtrl_{nullptr};

    /* inspector entry */
    DeviceTaskInspectorEntry inspectorEntry_;
    void* inspector_;
};
} // namespace npu::tile_fwk::dynamic
