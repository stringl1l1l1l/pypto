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
 * \file aicore_manager.h
 * \brief
 */

#pragma once
#include <cstdint>
#include <sys/ioctl.h>
#include <functional>
#include <vector>
#include <map>
#include <atomic>
#include <array>
#include <semaphore.h>
#include "machine/utils/dynamic/dev_start_args.h"
#include "securec.h"
#include "device_common.h"
#include "tilefwk/config.h"
#include "interface/utils/common.h"
#include "interface/operation/opcode.h"
#include "interface/schema/schema.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "machine/utils/dynamic/small_array.h"
#include "machine/utils/dynamic/spsc_queue.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/utils/device_log.h"
#include "machine/device/dynamic/perf_event_sampler.h"
#include "machine/device/dynamic/aicore_prof.h"
#include "machine/device/dynamic/aicore_hal.h"
#include "machine/device/dynamic/aicpu_task_manager.h"
#include "machine/device/dynamic/device_utils.h"
#include "machine/device/dynamic/wrap_manager.h"
#include "machine/device/dynamic/eslmodel_manager.h"
#include "machine/device/dump/aicore_dump.h"
#include "machine/device/debug/schema_trace_utils.h"
#include "device_trace.h"

namespace npu::tile_fwk::dynamic {

const uint32_t AICORE_STATUS_INIT = 0xFFFFFFFFU;

constexpr uint32_t REG_31_BITS = 0x7FFFFFFF;
constexpr uint32_t REG_32_BITS = 0xFFFFFFFF;
#define REG_LOW_TASK_ID(regVal) (regVal) & REG_31_BITS            // 低31位存储的taskid
#define REG_LOW_TASK_STATE(regVal) ((regVal) & REG_32_BITS) >> 31 // 低32位存储的task的状态
constexpr uint32_t TASK_FIN_STATE = 1;                            // 任务执行完成完成
constexpr uint32_t TASK_ACK_STATE = 0;                            // 收到任务状态，没执行完成
constexpr uint32_t REG_TASK_NUM = 2;                              // 一次寄存器task个数

struct TaskInfo {
    int coreIdx;
    uint64_t taskId;
    uint32_t dtaskId;
    TaskInfo(int idx, uint64_t id, uint32_t devtaskid) : coreIdx(idx), taskId(id), dtaskId(devtaskid) {}
};
struct ResolveTaskContext {
    uint32_t finishIds{0};
    uint32_t resolveIndexBase{0};
    int finishCoreIdx{0};
};
class AiCoreManager {
public:
    explicit AiCoreManager(SchThreadStatus& schThreadStatus) : threadStatus(schThreadStatus), aicoreProf_(*this) {};
    ~AiCoreManager() {};

    void InitCostModelFuncData(SchDeviceTaskContext* devTaskCtx)
    {
        int64_t funcdata;
        auto dyntask = (DynDeviceTask*)devTaskCtx->GetDeviceTask();
        funcdata = static_cast<int64_t>(PtrToValue(dyntask->GetDynFuncDataList()));
        ForEachManageAicore([&](int coreIdx) { aicoreHal_.InitCostModelDevTaskData(coreIdx, funcdata); });
    }

    void InitCostModelFuncDataForOneCore(SchDeviceTaskContext* devTaskCtx, int coreIdx)
    {
        auto dyntask = (DynDeviceTask*)devTaskCtx->GetDeviceTask();
        aicoreHal_.InitCostModelDevTaskData(coreIdx, static_cast<int64_t>(PtrToValue(dyntask->GetDynFuncDataList())));
    }

    void InitAicoreParallelDevTask(ParallelSchDeviceTaskContext* parallelCtx)
    {
        int64_t localFuncData[npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM];
        uint32_t localDevTaskIds[npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM];
        parallelCtx->GatherParallelDevTaskData(localFuncData, localDevTaskIds);

        DEV_IF_DEVICE
        {
            ForEachManageAicore([&](int coreIdx) {
                aicoreHal_.InitKernelArgs(coreIdx, 0);
                FillKernelArgsParallexDevTaskPreGathered(parallelCtx, coreIdx, localFuncData, localDevTaskIds);
            });
        }
        else
        {
            if (aicoreHal_.IsHostSimMode()) {
                ForEachManageAicore([&](int coreIdx) {
                    aicoreHal_.InitKernelArgs(coreIdx, 0);
                    FillKernelArgsParallexDevTaskPreGathered(parallelCtx, coreIdx, localFuncData, localDevTaskIds);
                });
            }
        }
    }

    void FillKernelArgsParallexDevTaskPreGathered(ParallelSchDeviceTaskContext* parallelCtx, int coreIdx,
                                                  const int64_t* localFuncData, const uint32_t* localDevTaskIds)
    {
        volatile ParallelDevTask* kernelParallDevTask = aicoreHal_.GetParallelDevTask(coreIdx);
        for (uint32_t idx = parallelCtx->front; idx < parallelCtx->rear; idx++) {
            auto slot = idx % npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM;
            aicoreHal_.SetParallelDevTask(kernelParallDevTask, idx, localFuncData[slot], localDevTaskIds[slot]);
        }
        aicoreHal_.SetParallelDevTaskSize(kernelParallDevTask, parallelCtx->front, parallelCtx->rear);
        aicoreHal_.SetParallelDevTaskCtxVersion(coreIdx, parallelCtx->Version());
        DEV_VERBOSE_DEBUG("Fill parallel dev task for core %d, ver:%u", coreIdx, parallelCtx->Version());
    }

    inline void SetSchduleContext(SchduleContext* context) { this->context_ = context; }

    inline void SetSchedSyncMode(uint8_t usingSynModel)
    {
        releaseCoreByRegValFn_ = usingSynModel == 1 ? &AiCoreManager::ReleaseCoreByRegValBySyncMode : // 串行执行模式
                                     &AiCoreManager::ReleaseCoreByRegValByAsyncMode; // 并行执行模式[默认执行模式]
    }

    inline bool CheckAndResetReg()
    {
        if (!validGetPgMask_) {
            return true;
        }
        bool isValid = true;
        DEV_IF_DEVICE
        {
            if (aicoreHal_.GetRegSprDataMainBase() == DAV_3510::REG_SPR_DATA_MAIN_BASE) {
                return true;
            }
            auto regAddrs = aicoreHal_.GetRegAddrs();
            uint32_t regNum = aicoreHal_.GetregNum();
            for (uint32_t coreIdx = 0; coreIdx < regNum; ++coreIdx) {
                if (regAddrs[coreIdx] == 0) {
                    continue;
                }
                uint32_t currentStatus = *(
                    reinterpret_cast<volatile uint32_t*>(regAddrs[coreIdx] + REG_SPR_FAST_PATH_ENABLE));
                if (currentStatus != REG_SPR_FAST_PATH_CLOSE) {
                    isValid = false;
                    *(reinterpret_cast<volatile uint32_t*>(regAddrs[coreIdx] +
                                                           REG_SPR_FAST_PATH_ENABLE)) = REG_SPR_FAST_PATH_CLOSE;
                }
            }
        }
        return isValid;
    }

    inline void InitDevTask(SchDeviceTaskContext* deviceTaskCtx)
    {
        auto devTask = deviceTaskCtx->GetDeviceTask();
        aicoreHal_.SetModel(devTask->aicoreModel);
        deviceTaskCtx->wrapManager.Init(deviceTaskCtx, deviceTaskCtx->GetDeviceTask(), &context_->coreStatusMgr,
                                        pendingIds_.data(), runningIds_.data(), aicValidNum_, aicStart_, adjAicEnd_,
                                        [&](SchDeviceTaskContext* devTaskCtx, CoreType coreType, int arg1,
                                            uint64_t arg2) { SendTaskToAiCore(devTaskCtx, coreType, arg1, arg2); });

        context_->coreStatusMgr.SetLastPendReadyCoreIdx(static_cast<int>(CoreType::AIV),
                                                        static_cast<uint32_t>(aivStart_));
        context_->coreStatusMgr.SetLastPendReadyCoreIdx(static_cast<int>(CoreType::AIC),
                                                        static_cast<uint32_t>(aicStart_));
        InitCostModelFuncData(deviceTaskCtx);
    }

    template <bool enableAicpuTask = false>
    inline int32_t RunCoreTask(SchDeviceTaskContext* devTaskCtx)
    {
        int32_t ret = DEVICE_MACHINE_OK;
        devTaskCtx->GetWrapManager().DispatchMixCoreTask();
        ret = DispatchAiCoreTask(devTaskCtx, CoreType::AIC, devTaskCtx->readyAicCoreFunctionQue, aicStart_, adjAicEnd_);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return ret;
        }
        ret = DispatchAiCoreTask(devTaskCtx, CoreType::AIV, devTaskCtx->readyAivCoreFunctionQue, aivStart_, adjAivEnd_);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return ret;
        }

        uint64_t aicpuTaskSent = 0UL;
        if constexpr (enableAicpuTask) {
            if (IsNeedProcAicpuTask()) {
                ret = ResolveDepForAicpuTask(aicpuTaskSent);
                if (unlikely(ret != DEVICE_MACHINE_OK)) {
                    return ret;
                }
            }
        }

        DEV_IF_VERBOSE_DEBUG
        {
            procAicCoreFunctionCnt_ += devTaskCtx->CurCoreTaskSent(CoreType::AIC);
            procAivCoreFunctionCnt_ += devTaskCtx->CurCoreTaskSent(CoreType::AIV);
            procAicpuFunctionCnt_ += aicpuTaskSent;
        }

        devTaskCtx->SetAicpuTaskSent(static_cast<uint32_t>(aicpuTaskSent));
        devTaskCtx->CountCoreTaskSent(context_->coreStatusMgr.WaitTaskCnt(static_cast<int>(CoreType::AIC)),
                                      context_->coreStatusMgr.WaitTaskCnt(static_cast<int>(CoreType::AIV)));
        return ret;
    }

    inline int RunTask(SchDeviceTaskContext* deviceTaskCtx)
    {
        int32_t ret = DEVICE_MACHINE_OK;
        DEV_VERBOSE_DEBUG("Run device task entry stage : %d", ToUnderlying(deviceTaskCtx->CurStage()));
        while (true) {
            bool isStageFinish = false;
            switch (deviceTaskCtx->CurStage()) {
                case DevTaskExecStage::INIT: {
                    ret = PreProcessTask(deviceTaskCtx, isStageFinish);
                    if (isStageFinish) {
                        deviceTaskCtx->EntryStage(DevTaskExecStage::SEND_CORE_TASK);
                    }
                    break;
                }
                case DevTaskExecStage::SEND_CORE_TASK: {
                    ret = ProcessTaskLoop(deviceTaskCtx, isStageFinish);
                    if (isStageFinish) {
                        deviceTaskCtx->EntryStage(DevTaskExecStage::WAIT_TAIL_TASK_FINISH);
                        PerfMtTrace(PERF_TRACE_DEV_TASK_SCHED_EXEC, aicpuIdx_);
                    }
                    break;
                }
                case DevTaskExecStage::WAIT_TAIL_TASK_FINISH: {
                    PerfMtBegin(PERF_EVT_SYNC_AICORE, aicpuIdx_);
                    ret = SyncTaskFinish(deviceTaskCtx, isStageFinish);
                    PerfMtEnd(PERF_EVT_SYNC_AICORE, aicpuIdx_);
                    if (isStageFinish) {
                        PerfMtTrace(PERF_TRACE_DEV_TASK_SYNC_CORE_STOP, aicpuIdx_);
                        if (deviceTaskCtx->GetDeviceTaskCtrl()->Finish(!deviceTaskCtx->IsParallel(), aicpuNum_)) {
                            PerfMtTrace(PERF_TRACE_DEV_TASK_RSP, aicpuIdx_);
                            deviceTaskCtx->EntryStage(DevTaskExecStage::FINISH);
                        } else {
                            deviceTaskCtx->EntryStage(DevTaskExecStage::WAIT_ALL_SCH_FINISH);
                        }
                    }
                    break;
                }
                case DevTaskExecStage::WAIT_ALL_SCH_FINISH: {
                    isStageFinish = deviceTaskCtx->GetDeviceTaskCtrl()->TryWaitAllSchFinish(aicpuNum_);
                    if (isStageFinish) {
                        deviceTaskCtx->EntryStage(DevTaskExecStage::FINISH);
                        PerfMtTrace(PERF_TRACE_DEV_TASK_RSP, aicpuIdx_);
                    }
                    break;
                }
                case DevTaskExecStage::FINISH: {
                    return DEVICE_MACHINE_OK;
                }
                default:
                    DEV_ERROR(SchedErr::FSM_STATUS_ERROR, "Invalid stage %d.", ToUnderlying(deviceTaskCtx->CurStage()));
                    ret = ToUnderlying(SchedErr::FSM_STATUS_ERROR);
                    break;
            }

            if (ret != DEVICE_MACHINE_OK) {
                break;
            }

            if (deviceTaskCtx->IsRunFinish()) {
                break;
            }

            if (!isStageFinish && deviceTaskCtx->IsParallel()) {
                DEV_VERBOSE_DEBUG("Run device task leave stage : %d", ToUnderlying(deviceTaskCtx->CurStage()));
                return ret; // wait parallel scheduled next time
            }
        }
        DEV_DEBUG("aicpu %d proc finish devtask(%lu),aic: %lu, aiv: %lu, aicpu: %lu, stage:%d, ret: %d.", aicpuIdx_,
                  deviceTaskCtx->TaskId(), procAicCoreFunctionCnt_, procAivCoreFunctionCnt_, procAicpuFunctionCnt_,
                  ToUnderlying(deviceTaskCtx->CurStage()), ret);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            DEV_ERROR(SchedErr::CORE_TASK_PROCESS_FAILED,
                      "#sche.dtask.leave: Aicpu[%d] proc finish: finishedFunctionCnt=%lu, "
                      "coreFunctionCnt=%lu, taskId=%lu, but timeout !.",
                      aicpuIdx_, deviceTaskCtx->GetDeviceTaskCtrl()->finishedFunctionCnt.load(),
                      deviceTaskCtx->GetDeviceTask()->coreFunctionCnt, deviceTaskCtx->TaskId());
        }
        return ret;
    }

    inline int32_t PreProcessTask(SchDeviceTaskContext* deviceTaskCtx, bool& isExecFinish)
    {
        int32_t ret = DEVICE_MACHINE_OK;
        DEV_INFO("receive new task %lu, firstTaskSend=%d.", deviceTaskCtx->TaskId(), deviceTaskCtx->isFirstTaskSend);

        // The initialization of aicpu tasks takes time, so to reduce headroom overhead, a batch of tasks is sent first.
        if (!deviceTaskCtx->isFirstTaskSend) {
            DEV_VERBOSE_DEBUG("#trace.dtask.start: tid=%d taskId=%lu coreFunctionCnt=%lu isLast=%d", schedIdx_,
                              deviceTaskCtx->TaskId(), deviceTaskCtx->GetDeviceTask()->coreFunctionCnt,
                              reinterpret_cast<DynDeviceTask*>(deviceTaskCtx->GetDeviceTask())->IsLastTask() ? 1 : 0);
            InitDevTask(deviceTaskCtx);
            ret = RunCoreTask(deviceTaskCtx);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }
        }

        if (IsNeedProcAicpuTask()) {
            const bool profSwitch = aicoreProf_.ProfIsEnable();
            uint32_t parallelIdx = deviceTaskCtx->parallelIdx;
            ret = aicpuTaskManager_.Init(reinterpret_cast<DynDeviceTask*>(deviceTaskCtx->GetDeviceTask()), profSwitch,
                                         parallelIdx);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }
        }

        isExecFinish = true;
        return DEVICE_MACHINE_OK;
    }

    inline int ProcessTaskLoop(SchDeviceTaskContext* deviceTaskCtx, bool& isFinish)
    {
        TIMEOUT_CHECK_INIT(archInfo_, TIMEOUT_DISPATCH);
        while (!deviceTaskCtx->IsCoreTaskSendFinish()) {
            int32_t ret = RunCoreTask<true>(deviceTaskCtx);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }

            if (deviceTaskCtx->IsParallel()) {
                deviceTaskCtx->SyncAllSchCoreTaskSent();
                isFinish = deviceTaskCtx->IsCoreTaskSendFinish();
                return ret;
            }

            DEV_IF_DEVICE
            {
                __PYPTO_TIMEOUT_CHECK_EXIT_ONLY(SchedErr::TASK_WAIT_TIMEOUT, return DEVICE_MACHINE_TIMEOUT_CORETASK,
                                                "#sche.task.loop: ProcessTaskLoop.");
            }
            DEV_IF_NONDEVICE
            {
                if (aicoreHal_.IsHostSimMode() && !enableEslModel_) {
                    __PYPTO_TIMEOUT_CHECK_EXIT_ONLY(SchedErr::TASK_WAIT_TIMEOUT, return DEVICE_MACHINE_TIMEOUT_CORETASK,
                                                    "#sche.task.loop: ProcessTaskLoop.");
                }
            }
        }
        deviceTaskCtx->SyncAllSchCoreTaskSent();
        isFinish = true;

        return DEVICE_MACHINE_OK;
    }

    inline void DumpLastWord(int coreIdx)
    {
        uint64_t status = aicoreHal_.GetAicoreStatus(coreIdx);
        if (pendingIds_[coreIdx] != AICORE_TASK_INIT) {
            DEV_ERROR(SchedErr::ABNOMAL_LAST_WORD,
                      "coreid=%d status=%lu, warningstatus=%lu ,lastwordstatus=%lu,"
                      "pending taskid=%x, parallelIdx=%u, coreVer=%u",
                      coreIdx, status, aicoreHal_.GetAicoreWarningStatus(coreIdx),
                      aicoreHal_.GetAicoreStatusLastWord(coreIdx), pendingIds_[coreIdx],
                      npu::tile_fwk::ParallelIndex(pendingIds_[coreIdx]),
                      aicoreHal_.ParallelDevTaskCtxVersion(coreIdx));
        }
        if (runningIds_[coreIdx] != AICORE_TASK_INIT) {
            DEV_ERROR(SchedErr::ABNOMAL_LAST_WORD,
                      "coreid=%d, status=%lu, warningstatus=%lu, lastwordstatus=%lu,"
                      "running taskid=%x, parallelIdx=%u, coreVer=%u",
                      coreIdx, status, aicoreHal_.GetAicoreWarningStatus(coreIdx),
                      aicoreHal_.GetAicoreStatusLastWord(coreIdx), runningIds_[coreIdx],
                      npu::tile_fwk::ParallelIndex(runningIds_[coreIdx]),
                      aicoreHal_.ParallelDevTaskCtxVersion(coreIdx));
        }
    }

    void ResetRegAll()
    {
        ForEachManageAicore([this](int coreIdx) {
            if (aicoreHal_.ReadPathReg(coreIdx) == REG_SPR_FAST_PATH_OPEN) {
                aicoreHal_.SetReadyQueue(coreIdx, AICORE_TASK_STOP + 1);
                aicoreHal_.WriteReg32(coreIdx, REG_SPR_FAST_PATH_ENABLE, REG_SPR_FAST_PATH_CLOSE);
            } else {
                aicoreHal_.SetReadyQueue(coreIdx, AICORE_TASK_STOP + 1);
            }
        });
    }

    inline void DumpPostRunSnapshot()
    {
        DEV_IF_VERBOSE_DEBUG
        {
            DEV_VERBOSE_DEBUG(
                "#trace.ctrlcore: schedIdx=%d ctrlCoreDisabled=%d aicStart=%d aicEnd=%d aivStart=%d aivEnd=%d "
                "runReadyAic=%u runReadyAiv=%u pendReadyAic=%u pendReadyAiv=%u",
                schedIdx_, disableControlCore_ ? 1 : 0, aicStart_, aicEnd_, aivStart_, aivEnd_,
                context_->coreStatusMgr.GetCoreRunReadyCnt(static_cast<int>(CoreType::AIC)),
                context_->coreStatusMgr.GetCoreRunReadyCnt(static_cast<int>(CoreType::AIV)),
                context_->coreStatusMgr.GetCorePendReadyCnt(static_cast<int>(CoreType::AIC)),
                context_->coreStatusMgr.GetCorePendReadyCnt(static_cast<int>(CoreType::AIV)));

            ForEachManageAicore([&](int coreIdx) {
                DEV_VERBOSE_DEBUG("#trace.aicore.status: schedIdx=%d core=%d type=%d runningId=%u pendingId=%u "
                                  "aicoreStatus=%lu lastwordStatus=%lu finishedTaskReg=%lu",
                                  schedIdx_, coreIdx, static_cast<int>(AicoreType(coreIdx)), runningIds_[coreIdx],
                                  pendingIds_[coreIdx], aicoreHal_.GetAicoreStatus(coreIdx),
                                  aicoreHal_.GetAicoreStatusLastWord(coreIdx), aicoreHal_.GetFinishedTask(coreIdx));
            });

            auto& parallelDevTaskCtx = context_->schParallelDevTaskCtx;
            for (uint32_t i = parallelDevTaskCtx.front; i != parallelDevTaskCtx.rear; ++i) {
                auto& taskCtx = parallelDevTaskCtx.elements[i % SCH_DEVTASK_MAX_PARALLELISM];
                if (taskCtx.readyAicCoreFunctionQue != nullptr) {
                    DEV_VERBOSE_DEBUG("#trace.queue: schedIdx=%d name=%s size=%u", schedIdx_, "readyAicCoreFunctionQue",
                                      taskCtx.readyAicCoreFunctionQue->UnsafeAtomicSize());
                }
                if (taskCtx.readyAivCoreFunctionQue != nullptr) {
                    DEV_VERBOSE_DEBUG("#trace.queue: schedIdx=%d name=%s size=%u", schedIdx_, "readyAivCoreFunctionQue",
                                      taskCtx.readyAivCoreFunctionQue->UnsafeAtomicSize());
                }
                auto& wrapMgr = taskCtx.GetWrapManager();
                if (wrapMgr.readyWrapCoreFunctionQue_ != nullptr) {
                    DEV_VERBOSE_DEBUG(
                        "#trace.queue: schedIdx=%d name=%s size=%u", schedIdx_, "readyWrapCoreFunctionQue",
                        wrapMgr.readyWrapCoreFunctionQue_->tail - wrapMgr.readyWrapCoreFunctionQue_->head);
                }
            }
        }
    }

    inline void PostRun(int ret)
    {
        if (ret) {
            DEV_ERROR(SchedErr::ABNOMAL_LAST_WORD,
                      "#sche.dtask.leave.post: execute error=%d, skip rest tasks, "
                      "runreadyAiv=%u runreadyaic=%u, %u, %u, cycle:%lu",
                      ret, context_->coreStatusMgr.GetCoreRunReadyCnt(0), context_->coreStatusMgr.GetCoreRunReadyCnt(1),
                      context_->coreStatusMgr.GetCorePendReadyCnt(0), context_->coreStatusMgr.GetCorePendReadyCnt(1),
                      GetCycles());
            if constexpr (IsDeviceMode()) {
                ForEachManageAicore([&](int coreIdx) { DumpLastWord(coreIdx); });
            }

            DumpPostRunSnapshot();
            // skip device task of current parallel ctx
            auto& parallelDevTaskCtx = context_->schParallelDevTaskCtx;
            for (uint32_t i = parallelDevTaskCtx.front; i != parallelDevTaskCtx.rear; ++i) {
                auto& taskCtx = parallelDevTaskCtx.elements[i % SCH_DEVTASK_MAX_PARALLELISM];
                if (!taskCtx.IsFree()) {
                    DEV_VERBOSE_DEBUG("#trace.dtask.abnormalend: tid=%d taskId=%lu finishedFunctionCnt=%lu", schedIdx_,
                                      taskCtx.TaskId(),
                                      taskCtx.GetDeviceTaskCtrl()->finishedFunctionCnt.load(std::memory_order_relaxed));
                    taskCtx.GetDeviceTaskCtrl()->Finish(true, aicpuNum_);
                }
                DEV_ERROR(SchedErr::ABNOMAL_LAST_WORD, "Force finish parallel ctx  parallelidx:%u.",
                          i % SCH_DEVTASK_MAX_PARALLELISM);
                taskCtx.Dump();
            }
            DumpAiCoreStatus();
            // skip device tash of ctrl quene
            DeviceTaskCtrl* taskCtrl = nullptr;
            while (!taskQueue_->IsEmpty()) {
                if ((taskCtrl = taskQueue_->Dequeue())) {
                    taskCtrl->Finish(true, aicpuNum_);
                }
            };

            if constexpr (IsDeviceMode()) {
                NormalStop(); // some core maybe timeout
            }
        }

        if constexpr (IsDeviceMode()) {
            PerfMtTrace(PERF_TRACE_WAIT_CORE_EXIT, aicpuIdx_);
            ProfStop();
        }
        DEV_INFO("Aicpu[%d] stop: ret=%d, procAicTaskCnt=%lu, procAivTaskCnt=%lu.", aicpuIdx_, ret,
                 procAicCoreFunctionCnt_, procAivCoreFunctionCnt_);
    }

    inline int RunManager(int threadIdx, DevStartArgs* devStartArgs, DeviceArgs* deviceArgs, int schedIdx,
                          int arbitratedScheNum)
    {
        int ret = DEVICE_MACHINE_OK;
        DEV_DEBUG("schedule run threadIdx=%d", threadIdx);
        Init(threadIdx, devStartArgs, deviceArgs, schedIdx, arbitratedScheNum);
        PerfMtTrace(PERF_TRACE_INIT, threadIdx);
        DEV_DEBUG("Schedule run init succ");
        DeviceTaskCtrl* taskCtrl = nullptr;
        taskQueue_ = &(devStartArgs->deviceRuntimeDataDesc.taskQueueList[schedIdx_]);
        if constexpr (IsDeviceMode()) {
            ret = HandShake(devStartArgs);
            PerfMtTrace(PERF_TRACE_CORE_HAND_SHAKE, threadIdx);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                DEV_ERROR(SchedErr::HANDSHAKE_TIMEOUT, "#sche.handshake.error: hand shake timeout.");
                AbnormalStop();
                while ((taskCtrl = taskQueue_->Dequeue())) {
                    taskCtrl->Finish(true, aicpuNum_);
                }
                return ret;
            }
            aicoreProf_.ProfStart();
        } else if (aicoreHal_.IsHostSimMode()) {
            ret = HandShake(devStartArgs);
            PerfMtTrace(PERF_TRACE_CORE_HAND_SHAKE, threadIdx);
            if (unlikely(ret != 0)) {
                while ((taskCtrl = taskQueue_->Dequeue())) {
                    taskCtrl->Finish(true, aicpuNum_);
                }
                return ret;
            }
        }
        DEV_DEBUG("Schedule run start succ");
        uint64_t lastDevTaskFinCycle = 0;
        PROF_STAGE_BEGIN_MTSAFE(PERF_EVT_STAGE_SCHEDULE, threadIdx, "dispatch.before\n");
        TIMEOUT_CHECK_INIT(archInfo_, TIMEOUT_DISPATCH);
        uint32_t lastParallelVer = context_->PrallelVersion();
        while (ret == 0) {
            FillParallelDevtaskCtx();

            ret = ProcessParallelDevTasks();
            if (ret != DEVICE_MACHINE_OK)
                break;
            lastDevTaskFinCycle = GetCycles();
            if (context_->DevTaskEmpty() && taskCtrlDequeFinish) {
                PerfMtTrace(PERF_TRACE_WAIT_ALL_DEV_TASK_FINISH, aicpuIdx_, lastDevTaskFinCycle);
                if (!isSendStop) {
                    DEV_INFO("Send all core stop.");
                    SendAllCoreStop();
                }
                break;
            }

            if (lastParallelVer != context_->PrallelVersion()) {
                __PYPTO_TIMEOUT_CHECK_RESET;
                lastParallelVer = context_->PrallelVersion();
            }

            DEV_IF_DEVICE
            {
                __PYPTO_TIMEOUT_CHECK_EXIT_ONLY(
                    SchedErr::SCH_PARALLEL_DEVTASK_TIMEOUT,
                    { ret = ToUnderlying(SchedErr::SCH_PARALLEL_DEVTASK_TIMEOUT); },
                    "#sche.parallel.devtask: Schedule parallel devtask, dequeueFinish=%d.", taskCtrlDequeFinish);
            }
            DEV_IF_NONDEVICE
            {
                if (aicoreHal_.IsHostSimMode() && !enableEslModel_) {
                    __PYPTO_TIMEOUT_CHECK_EXIT_ONLY(
                        SchedErr::SCH_PARALLEL_DEVTASK_TIMEOUT,
                        { ret = ToUnderlying(SchedErr::SCH_PARALLEL_DEVTASK_TIMEOUT); },
                        "#sche.parallel.devtask: Schedule parallel devtask, dequeueFinish=%d.", taskCtrlDequeFinish);
                }
            }
        }
        PROF_STAGE_END_MTSAFE(PERF_EVT_STAGE_SCHEDULE, threadIdx, "dispatch.after\n");

        PostRun(ret);
        return ret;
    }

    int32_t ProcessCompletedAicpuTask(uint64_t taskId)
    {
        int32_t ret = ResolveDepDyn(context_->GetCurSchDevTaskCtx(), taskId);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return ret;
        }
        return BatchPushReadyQueue(context_->GetCurSchDevTaskCtx());
    }

private:
    inline void DumpTaskProf()
    {
        for (int i = aicStart_; i < aicEnd_; ++i) {
            aicoreHal_.DumpTaskProf(i, CoreType::AIC);
        }
        for (int i = aivStart_; i < aivEnd_; ++i) {
            aicoreHal_.DumpTaskProf(i, CoreType::AIV);
        }
    }

    inline void ProfStop()
    {
        if (aicoreProf_.ProfIsEnable()) {
            DumpTaskProf();
        }

        if (unlikely(isOpenPerf_)) {
            for (int i = aicStart_; i < aicEnd_; ++i) {
                aicoreHal_.SetAicorePerfTrace(i, CoreType::AIC, aicpuIdx_);
            }
            for (int i = aivStart_; i < aivEnd_; ++i) {
                aicoreHal_.SetAicorePerfTrace(i, CoreType::AIV, aicpuIdx_);
            }
        }

        aicoreProf_.ProfStop();
    }

    inline void DumpAiCoreStatus() const
    {
        DEV_IF_VERBOSE_DEBUG
        {
            ForEachManageAicore([this](int coreIdx) {
                if constexpr (IsDeviceMode()) {
                    aicoreHal_.DumpAicoreStatus(coreIdx);
                }
                DEV_ERROR(SchedErr::ABNOMAL_LAST_WORD, "reg low task: runningid(%u) pendingid(%u)",
                          runningIds_[coreIdx], pendingIds_[coreIdx]);

                DEV_ERROR(SchedErr::ABNOMAL_LAST_WORD,
                          "send task info ~~~~~~~~~~~~~~~~~~~~~~~~~~~count:%lu~~~~~~~~~~~~~~~~~~~~~~~~~~~.",
                          sendTask_[coreIdx].size());
                for (size_t i = 0; i < sendTask_[coreIdx].size(); i++) {
                    DEV_ERROR(
                        SchedErr::ABNOMAL_LAST_WORD,
                        "send task: seqno %d, taskId %lu, refreshDevTask %x parallelModifyFlag:%x, deviceTaskId %u",
                        (int)i, sendTask_[coreIdx][i].taskId & 0xFFFFFFFF,
                        DevTaskId(sendTask_[coreIdx][i].taskId >> REG_HIGH_DTASKID_SHIFT),
                        ParallelDevTaskModifyFlag(sendTask_[coreIdx][i].taskId >> REG_HIGH_DTASKID_SHIFT),
                        sendTask_[coreIdx][i].dtaskId);
                }

                DEV_ERROR(SchedErr::ABNOMAL_LAST_WORD,
                          "recv finish task info ~~~~~~~~~~~~~~~~~~~~~~~~count:%lu~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~.",
                          recvFinTask_[coreIdx].size());
                for (size_t i = 0; i < recvFinTask_[coreIdx].size(); i++) {
                    DEV_ERROR(SchedErr::ABNOMAL_LAST_WORD, "recv task: seqno %d, taskId %lu, deviceTaskId %u", (int)i,
                              recvFinTask_[coreIdx][i].taskId, recvFinTask_[coreIdx][i].dtaskId);
                }

                DEV_ERROR(SchedErr::ABNOMAL_LAST_WORD,
                          "recv ack task info ~~~~~~~~~~~~~~~~~~~~~~~~~~~count:%lu~~~~~~~~~~~~~~~~~~~~~~~~~~~.",
                          recvAckTask_[coreIdx].size());
                for (size_t i = 0; i < recvAckTask_[coreIdx].size(); i++) {
                    DEV_ERROR(SchedErr::ABNOMAL_LAST_WORD, "recv ack task: seqno %d, taskId %lu", static_cast<int>(i),
                              recvAckTask_[coreIdx][i].taskId);
                }
            });
        }
    }

    bool NeedProcCoreTaskRsp(SchDeviceTaskContext* devTaskCtx, int coreIdx)
    {
        if ((pendingIds_[coreIdx] != AICORE_TASK_INIT &&
             ParallelIndex(pendingIds_[coreIdx]) == devTaskCtx->parallelIdx) ||
            (runningIds_[coreIdx] != AICORE_TASK_INIT &&
             ParallelIndex(runningIds_[coreIdx]) == devTaskCtx->parallelIdx)) {
            DEV_VERBOSE_DEBUG("core %d have tail un-finished task %x  %x.", coreIdx, pendingIds_[coreIdx],
                              runningIds_[coreIdx]);
            return true;
        }

        return false;
    }

    inline bool CheckStopTaskCanBeSent(SchDeviceTaskContext* devTaskCtx, int coreIdx, bool isLastDevTask,
                                       uint32_t& resloveParallelIdx)
    {
        if (!NeedProcCoreTaskRsp(devTaskCtx, coreIdx)) {
            return true;
        }

        uint64_t finTaskVal = aicoreHal_.GetFinishedTask(coreIdx);
        uint32_t regLFinTaskId = REG_LOW_TASK_ID(finTaskVal);
        uint32_t regLFinTaskState = REG_LOW_TASK_STATE(finTaskVal);
        bool bMatch = false;

        auto& pendingIdRef = pendingIds_[coreIdx];
        auto& runningIdRef = runningIds_[coreIdx];
        int type = static_cast<int>(AicoreType(coreIdx));
        if (likely(regLFinTaskState == TASK_FIN_STATE)) {
            if (pendingIdRef == regLFinTaskId) {
                bMatch = true;
                context_->coreStatusMgr.AddRunAndPendCoreIdx(coreIdx, type);

                if (runningIdRef != AICORE_TASK_INIT) {
                    if (ParallelIndex(runningIdRef) == devTaskCtx->parallelIdx) {
                        DfxProcAfterFinishTask(devTaskCtx, coreIdx, runningIdRef);
                    } else {
                        // attention: if finished task belong to other devicetask, need to do the dependency resolution
                        (void)ResolveDepWithDfx(static_cast<CoreType>(type), coreIdx, runningIdRef,
                                                runningResolveIndexList_[coreIdx], resloveParallelIdx);
                    }
                }

                if (ParallelIndex(regLFinTaskId) == devTaskCtx->parallelIdx) {
                    DfxProcAfterFinishTask(devTaskCtx, coreIdx, regLFinTaskId);
                } else {
                    // attention: if finished task belong to other devicetask, need to do the dependency resolution
                    (void)ResolveDepWithDfx(static_cast<CoreType>(type), coreIdx, regLFinTaskId,
                                            pendingResolveIndexList_[coreIdx], resloveParallelIdx);
                }

                DEV_VERBOSE_DEBUG("rcv final pending task finish, pendtask: %u", regLFinTaskId);
            } else if (runningIdRef == regLFinTaskId && pendingIdRef == AICORE_TASK_INIT) {
                bMatch = true;
                context_->coreStatusMgr.AddRunReadyCoreIdx(coreIdx, type);
                DfxProcAfterFinishTask(devTaskCtx, coreIdx, regLFinTaskId);
                DEV_VERBOSE_DEBUG("rcv final running task finish, runningtask: %u", regLFinTaskId);
            }
        } else if (isLastDevTask && regLFinTaskState == TASK_ACK_STATE && pendingIdRef == regLFinTaskId) {
            // The core stop task can be sent once the last task ACK is received, without waiting for finish rsp.
            // The execution of the final task and the sending of the final core stop task can be parallelized.
            bMatch = true;
            context_->coreStatusMgr.AddRunAndPendCoreIdx(coreIdx, type);
            DfxProcAfterFinishTask(devTaskCtx, coreIdx, regLFinTaskId);
            if (runningIds_[coreIdx] != AICORE_TASK_INIT) {
                DfxProcAfterFinishTask(devTaskCtx, coreIdx, runningIds_[coreIdx]);
            }
            DEV_VERBOSE_DEBUG("rcv final pending task ack, pendtask: %u", regLFinTaskId);
        }

        if (bMatch) {
            pendingIdRef = AICORE_TASK_INIT;
            pendingResolveIndexList_[coreIdx] = 0;
            runningIdRef = AICORE_TASK_INIT;
            runningResolveIndexList_[coreIdx] = 0;
            DEV_VERBOSE_DEBUG("core %d tail task finished.", coreIdx);
            return true;
        }

        return false;
    }

    inline void MarkCoreStoped(SchDeviceTaskContext* devTaskCtx, int coreIdx)
    {
        devTaskCtx->coreTaskFinished[coreIdx] = 1;
        devTaskCtx->coreFinishedNum++;
        DEV_VERBOSE_DEBUG("Core %d finished, finishnum = %u. ", coreIdx, devTaskCtx->coreFinishedNum);
    }

    inline void AicoreDevTaskFinishProc(SchDeviceTaskContext* devTaskCtx, int coreIdx, bool isLastDevTask,
                                        uint32_t& resloveParallelIdx)
    {
        if (CheckStopTaskCanBeSent(devTaskCtx, coreIdx, isLastDevTask, resloveParallelIdx)) {
            MarkCoreStoped(devTaskCtx, coreIdx);
        }
        return;
    }

    inline void DumpDfxWhenCoreNotStop(SchDeviceTaskContext* devTaskCtx)
    {
        for (int i = aicStart_; i < aicEnd_; i++) {
            if (!devTaskCtx->coreTaskFinished[i]) {
                DEV_ERROR(
                    SchedErr::TASK_WAIT_TIMEOUT,
                    "#sche.task.end.sync.timeout: left aic core %d not stop, pending:%x, running:%x, regfinishid: %lx,"
                    "core last status:%lu, warning status:%lu, lastword status:%lu",
                    i, pendingIds_[i], runningIds_[i], aicoreHal_.GetFinishedTask(i), aicoreHal_.GetAicoreStatus(i),
                    aicoreHal_.GetAicoreWarningStatus(i), aicoreHal_.GetAicoreStatusLastWord(i));
            }
        }

        for (int i = aivStart_; i < aivEnd_; i++) {
            if (!devTaskCtx->coreTaskFinished[i]) {
                DEV_ERROR(
                    SchedErr::TASK_WAIT_TIMEOUT,
                    "#sche.task.end.sync.timeout: left aiv core %d not stop, pending:%x, running:%x, regfinishid: %lx,"
                    "core last status:%lu, warning status:%lu, lastword status:%lu",
                    i, pendingIds_[i], runningIds_[i], aicoreHal_.GetFinishedTask(i), aicoreHal_.GetAicoreStatus(i),
                    aicoreHal_.GetAicoreWarningStatus(i), aicoreHal_.GetAicoreStatusLastWord(i));
            }
        }
    }

    void SendAllCoreStop()
    {
        if (!NeedsHwStopOnLastDevTask()) {
            return;
        }
        BatchStopAllManagedCores();
    }

    inline int SyncTaskFinish(SchDeviceTaskContext* devTaskCtx, bool& isFinish, bool forceStop = false)
    {
        int aicNum = aicEnd_ - aicStart_;
        int aivNum = aivEnd_ - aivStart_;
        uint32_t mngCoreNum = static_cast<uint32_t>(aicNum + aivNum);
        bool aicAllStop = false;
        bool aivAllStop = false;
        bool isLastDevTask = false;
        auto curDevTask = devTaskCtx->GetDeviceTask();
        if (!forceStop) {
            isLastDevTask = reinterpret_cast<DynDeviceTask*>(curDevTask)->IsLastTask();
        } else {
            isLastDevTask = true;
        }

        if (isLastDevTask && (context_->DeviceTaskCtxNum() == 1)) {
            isSendStop = true;
        }

        TIMEOUT_CHECK_INIT(archInfo_, TIMEOUT_DISPATCH);
        uint32_t resloveParallelIdx = 0;
        while (devTaskCtx->coreFinishedNum < mngCoreNum) {
            bool curIterAicAllStop = true;
            bool curIterAivAllStop = true;
            for (int i = aicStart_; (!aicAllStop) && i < aicEnd_; ++i) {
                if (devTaskCtx->coreTaskFinished[i]) {
                    continue;
                }

                AicoreDevTaskFinishProc(devTaskCtx, i, isSendStop, resloveParallelIdx);
                curIterAicAllStop = false;
            }
            aicAllStop = curIterAicAllStop;

            for (int i = aivStart_; (!aivAllStop) && i < aivEnd_; ++i) {
                if (devTaskCtx->coreTaskFinished[i]) {
                    continue;
                }
                AicoreDevTaskFinishProc(devTaskCtx, i, isSendStop, resloveParallelIdx);
                curIterAivAllStop = false;
            }
            aivAllStop = curIterAivAllStop;

            if (devTaskCtx->IsParallel()) {
                (void)BatchPushReadyQueForParallel(resloveParallelIdx);
                break;
            }

            DEV_IF_DEVICE
            {
                __PYPTO_TIMEOUT_CHECK_EXIT_ONLY(
                    SchedErr::TASK_WAIT_TIMEOUT,
                    {
                        DumpDfxWhenCoreNotStop(devTaskCtx);
                        return DEVICE_MACHINE_TIMEOUT_SYNC_CORE_FINISH;
                    },
                    "#sche.task.end.sync: SyncAicoreDevTaskFinish, notstopNum=%u.",
                    mngCoreNum - devTaskCtx->coreFinishedNum);
            }
            DEV_IF_NONDEVICE
            {
                if (aicoreHal_.IsHostSimMode() && !enableEslModel_) {
                    __PYPTO_TIMEOUT_CHECK_EXIT_ONLY(
                        SchedErr::TASK_WAIT_TIMEOUT,
                        {
                            DumpDfxWhenCoreNotStop(devTaskCtx);
                            return DEVICE_MACHINE_TIMEOUT_SYNC_CORE_FINISH;
                        },
                        "#sche.task.end.sync: SyncAicoreDevTaskFinish, notstopNum=%u.",
                        mngCoreNum - devTaskCtx->coreFinishedNum);
                }
            }
        }

        if (isSendStop && NeedsHwStopOnLastDevTask() && devTaskCtx->coreFinishedNum == mngCoreNum) {
            BatchStopAllManagedCores();
        }

        if (devTaskCtx->coreFinishedNum == mngCoreNum) {
            isFinish = true;
            return SyncAicpuTaskFinish(devTaskCtx);
        }
        return DEVICE_MACHINE_OK;
    }

    inline int32_t SyncAicpuTaskFinish(SchDeviceTaskContext* devTaskCtx)
    {
        if (IsNeedProcAicpuTask()) {
            uint32_t parallelIdx = devTaskCtx->parallelIdx;
            auto ret = aicpuTaskManager_.SyncAicpuTaskFinish(this, parallelIdx);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }
        }
        return DEVICE_MACHINE_OK;
    }

    // 检查是否进入了尾批，当剩余任务数小于等于管理核心数时，认为进入了尾批
    inline bool CheckIsTailBatch(SchDeviceTaskContext* devTaskCtx, CoreType type, uint64_t& remaining)
    {
        if (devTaskCtx->IsParallel() || aicpuNum_ <= 1) {
            return false;
        }

        remaining = devTaskCtx->GetDeviceTask()->coreFunctionCnt -
                    devTaskCtx->GetDeviceTaskCtrl()->finishedFunctionCnt.load(std::memory_order_relaxed);
        uint32_t totalCoreNum = (type == CoreType::AIC) ? static_cast<uint32_t>(aicNum_) :
                                                          static_cast<uint32_t>(aivNum_);
        return (remaining > 0 && remaining <= static_cast<uint64_t>(totalCoreNum));
    }

    // 当进入尾批时，也选择保守策略，只分配完全空闲的核心
    inline uint16_t GetReadyCoreNum(CoreType type, bool isTail = false)
    {
        if ((enableFairSch_ || isTail) && IsExistOtherAicpuIdle(type)) {
            return static_cast<uint16_t>(context_->coreStatusMgr.GetCoreRunReadyCnt(static_cast<int>(type)));
        }
        return static_cast<uint16_t>(context_->coreStatusMgr.GetCorePendReadyCnt(static_cast<int>(type)));
    }

    inline uint16_t GetRunReadyCoreNum(CoreType type)
    {
        return static_cast<uint16_t>(context_->coreStatusMgr.GetCoreRunReadyCnt(static_cast<int>(type)));
    }

    [[gnu::hot]] inline uint64_t TryBatchSendTask(SchDeviceTaskContext* devTaskCtx, CoreType type,
                                                  ReadyCoreFunctionQueue* readyQue, int coreIdxStart, int coreIdxEnd)
    {
        if (readyQue->UnsafeAtomicSize() == 0) {
            DEV_VERBOSE_DEBUG("AiCpud:%d, can not send task currently. ready Task: 0", aicpuIdx_);
            return 0;
        }
        const int typeInt = static_cast<int>(type);
        uint64_t remaining = 0;
        bool isTail = CheckIsTailBatch(devTaskCtx, type, remaining);
        uint32_t ready = GetReadyCoreNum(type, isTail);
        if (ready == 0) {
            DEV_VERBOSE_DEBUG("AiCpud:%d, can not send task currently. ready Core: %u.", aicpuIdx_, ready);
            return 0;
        }
        PerfMtBegin(PERF_EVT_SEND_AIC_TASK, aicpuIdx_);
        const bool isRealLifo = (enableL2CacheSch_ && !firstLock[typeInt]);
        uint32_t readyId[MAX_MANAGER_AIV_NUM];
        auto readyTasksRange = isRealLifo ? readyQue->DequeueTail(ready, readyId) : readyQue->Dequeue(ready);
        const uint32_t taskCount = readyTasksRange.second - readyTasksRange.first;
        if (taskCount == 0) {
            DEV_VERBOSE_DEBUG("AiCpud:%d, taskCount is zero", aicpuIdx_);
            PerfMtEnd(PERF_EVT_SEND_AIC_TASK, aicpuIdx_);
            return 0;
        }

        DEV_VERBOSE_DEBUG("AiCpud:%d, pop all new task count: %u", aicpuIdx_, taskCount);
        BatchSendTask(devTaskCtx, type, isRealLifo ? readyTasksRange.second - 1 : readyTasksRange.first, taskCount,
                      coreIdxStart, coreIdxEnd, isRealLifo);
        DEV_VERBOSE_DEBUG("core ready cnt: %u", context_->coreStatusMgr.GetCorePendReadyCnt(typeInt));
        firstLock[typeInt] = false;
        PerfMtEnd(PERF_EVT_SEND_AIC_TASK, aicpuIdx_);
        return taskCount;
    }

    [[gnu::hot]] inline uint32_t BatchSendTask(SchDeviceTaskContext* devTaskCtx, CoreType type, const uint32_t* newTask,
                                               uint32_t taskCount, int coreIdxStart, int coreIdxEnd, bool isLifo)
    {
        const int typeInt = static_cast<int>(type);
        uint32_t sendCnt = 0;
        uint32_t coreNum = coreIdxEnd - coreIdxStart;
        uint32_t coreRunReadyCnt = context_->coreStatusMgr.GetCoreRunReadyCnt(typeInt);
        uint32_t corePendReadyCnt = context_->coreStatusMgr.GetCorePendReadyCnt(typeInt);
        DEV_VERBOSE_DEBUG("Begin Batch send %s task: corerunreadycnt:%u, pendreadyCnt:%u, taskCount:%u.",
                          type == CoreType::AIC ? "AIC" : "AIV", coreRunReadyCnt, corePendReadyCnt, taskCount);
        if (unlikely(coreRunReadyCnt > coreNum || corePendReadyCnt > coreNum)) {
            DEV_ERROR(SchedErr::CORE_INFO_INVALID,
                      "Core info is invalid aicpu %d, %s coreNum: %u, coreRunReadCnt: %u,"
                      "corePendReadyCnt: %u",
                      aicpuIdx_, type == CoreType::AIC ? "AIC" : "AIV", coreNum, coreRunReadyCnt, corePendReadyCnt);
            return DEVICE_MACHINE_ERROR;
        }
        while (sendCnt < static_cast<uint64_t>(coreRunReadyCnt) && sendCnt < taskCount) {
            uint32_t coreIdx = context_->coreStatusMgr.GetRunReadyCoreIdx(
                typeInt, context_->coreStatusMgr.GetCoreRunReadyCnt(typeInt) - 1);
            context_->coreStatusMgr.RemoveReadyCoreIdxTail(coreIdx, typeInt);
            SendTaskToAiCore(devTaskCtx, type, coreIdx, isLifo ? *newTask-- : *newTask++);
            sendCnt++;
        }
        context_->coreStatusMgr.BatchRemovePendReadyCoreIdx(typeInt, sendCnt);

        uint32_t idx = context_->coreStatusMgr.GetLastPendReadyCoreIdx(typeInt);
        uint32_t lastProcCore = idx;
        DEV_VERBOSE_DEBUG("  ## send task left pend ready cnt %u , last core index:%u.",
                          context_->coreStatusMgr.GetCorePendReadyCnt(typeInt), idx);
        while (context_->coreStatusMgr.GetCorePendReadyCnt(typeInt) > 0 && sendCnt < taskCount) {
            if (pendingIds_[idx] == AICORE_TASK_INIT) {
                DEV_VERBOSE_DEBUG("  ## send task use pendready core %u.", idx);
                SendTaskToAiCore(devTaskCtx, type, idx, isLifo ? *newTask-- : *newTask++);
                sendCnt++;
                context_->coreStatusMgr.RemovePendReadyCoreIdx(typeInt);
                lastProcCore = idx;
            }
            idx = coreIdxStart + (idx - coreIdxStart + 1) % coreNum;
        }

        if (lastProcCore != context_->coreStatusMgr.GetLastPendReadyCoreIdx(typeInt)) {
            context_->coreStatusMgr.SetLastPendReadyCoreIdx(typeInt,
                                                            coreIdxStart + (lastProcCore - coreIdxStart + 1) % coreNum);
        }
        DEV_VERBOSE_DEBUG("  ## finish send task left runreadycnt:%u pendreadycnt %u, last coreindex:%u.",
                          context_->coreStatusMgr.GetCoreRunReadyCnt(typeInt),
                          context_->coreStatusMgr.GetCorePendReadyCnt(typeInt), idx);
        return sendCnt;
    }

    inline int32_t DispatchAiCoreTask(SchDeviceTaskContext* devTaskCtx, CoreType type, ReadyCoreFunctionQueue* readyQue,
                                      int coreIdxStart, int coreIdxEnd)
    {
        int32_t ret = DEVICE_MACHINE_OK;
        if (context_->coreStatusMgr.WaitTaskCnt(static_cast<int>(type)) > 0) {
            ret = ResolveDepForAllAiCore(devTaskCtx, type, coreIdxStart, coreIdxEnd);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }
        }
        auto& wrapManager = devTaskCtx->GetWrapManager();
        if (wrapManager.IsMixArch() && wrapManager.GetDieId() != DieId::DIE_MIX) {
            ReadyCoreFunctionQueue* dieReadyQue = (type == CoreType::AIC) ? wrapManager.GetDieReadyAicQue() :
                                                                            wrapManager.GetDieReadyAivQue();
            if (dieReadyQue != readyQue) {
                TryBatchSendTask(devTaskCtx, type, dieReadyQue, coreIdxStart, coreIdxEnd);
            }
        }
        TryBatchSendTask(devTaskCtx, type, readyQue, coreIdxStart, coreIdxEnd);
        // DIE_MIX only: GetDieReady*Que() == default, so die tasks would starve; fall back when default empty.
        if (wrapManager.IsMixArch() && wrapManager.GetDieId() == DieId::DIE_MIX && readyQue->UnsafeAtomicSize() == 0) {
            for (size_t i = 0; i < DIE_NUM; ++i) {
                ReadyCoreFunctionQueue* dieReadyQue = wrapManager.GetDieReadyQue(type, i);
                if (dieReadyQue != nullptr && dieReadyQue != readyQue) {
                    TryBatchSendTask(devTaskCtx, type, dieReadyQue, coreIdxStart, coreIdxEnd);
                }
            }
        }
        if (enableFairSch_) {
            if (context_->coreStatusMgr.GetCoreRunReadyCnt(static_cast<int>(type)) > 0) {
                AicpuIsIdle(type);
            } else {
                AicpuIsBusy(type);
            }
        }
        return ret;
    }

    uint64_t UpdateParallelCtxAndCalcModifyFlag(int coreIdx, uint32_t coreParallelVersion)
    {
        uint64_t modifyFlag = 0;
        auto& parallelCtx = context_->schParallelDevTaskCtx;
        volatile ParallelDevTask* coreParallelDevTask = aicoreHal_.GetParallelDevTask(coreIdx);

        for (uint32_t i = parallelCtx.front; i < parallelCtx.rear; ++i) {
            uint32_t idx = i % npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM;

            SchDeviceTaskContext* devTaskCtx = parallelCtx.Element(i);
            if (devTaskCtx->IsFree()) {
                continue;
            }

            auto* dyntask = reinterpret_cast<DynDeviceTask*>(devTaskCtx->GetDeviceTask());
            int64_t funcData = static_cast<int64_t>(PtrToValue(dyntask->GetDynFuncDataList()));
            if (devTaskCtx->bindParallelCtxVersion > coreParallelVersion) {
                modifyFlag |= (1ULL << idx);
                aicoreHal_.SetParallelDevTask(coreParallelDevTask, idx, funcData, dyntask->GetIndex());
            }
        }
        return modifyFlag;
    }

    /*
     |--------8bit--------------|----24bit----|----1bit----|-----1bit------|------1bit-----|-----3bit--------|---10bit---|---16bit--|
     |-parallel ctx modifyflag--|--devtaskid--|----rspflag-|--pingpongflag-|---dcci flag---|--prallel index--|--func
     id--|--opindex-|
    */
    [[gnu::hot]] uint64_t EncodeTaskId(SchDeviceTaskContext* devTaskCtx, int coreIdx, uint64_t newTask)
    {
        uint32_t shift = TASKID_TASK_BITS + TASKID_FUNC_BITS;

        // encode parallel index
        uint64_t encodeTaskId = newTask | (devTaskCtx->parallelIdx << shift);
        uint32_t coreParallelVersion = aicoreHal_.ParallelDevTaskCtxVersion(coreIdx);

        // devicetask context not compatible， need notify aicore dcci
        if (devTaskCtx->bindParallelCtxVersion > coreParallelVersion) {
            DEV_INFO("Notify aicore(%d) refresh parallel devtask, devtaskVer:%u > coreVer:%u, newestVer: %u.", coreIdx,
                     devTaskCtx->bindParallelCtxVersion, aicoreHal_.ParallelDevTaskCtxVersion(coreIdx),
                     context_->PrallelVersion());

            // encode dcci flag
            shift += TASKID_PARALLEL_INDEX_BITS;
            encodeTaskId |= 1 << shift;

            // encode pingpong flag make sure encoded taskid is different with last time
            shift += TASKID_DEVTASK_DCCI_BITS;
            encodeTaskId |= pingPongFlag_[coreIdx] << shift;
            pingPongFlag_[coreIdx] ^= 1;

            // encode device taskid, aicore will validate it from taskid
            shift += 2;
            encodeTaskId |= devTaskCtx->TaskId() << shift;

            DEV_IF_DEVICE
            {
                // encode parallel ctx modify flag
                shift += REG_VAL_DEVTASK_ID_BITS;
                encodeTaskId |= UpdateParallelCtxAndCalcModifyFlag(coreIdx, coreParallelVersion) << shift;
            }
            else
            {
                if (enableEslModel_) {
                    // encode parallel ctx modify flag
                    shift += REG_VAL_DEVTASK_ID_BITS;
                    encodeTaskId |= UpdateParallelCtxAndCalcModifyFlag(coreIdx, coreParallelVersion) << shift;
                } else {
                    // costmodel don't support parallel devtask
                    InitCostModelFuncDataForOneCore(devTaskCtx, coreIdx);
                }
            }

            aicoreHal_.SetParallelDevTaskCtxVersion(coreIdx, context_->PrallelVersion());
        }
        return encodeTaskId;
    }

    [[gnu::hot]] inline void SendTaskToAiCore(SchDeviceTaskContext* devTaskCtx, CoreType type, int coreIdx,
                                              uint64_t newTask)
    {
        DEV_TRACE_DEBUG(LEvent(LUid(devTaskCtx->TaskId(), FuncID(newTask), GetRootIndex(devTaskCtx, newTask),
                                    TaskID(newTask), GetLeafIndex(devTaskCtx, newTask)),
                               LActStart(coreIdx)));
#if ENABLE_DUMP_OPERATION
        SchemaDumpUtil::DumpSchemaOperationInfo(devTaskCtx, newTask);
#endif

#if ENABLE_TENSOR_DUMP
        DEV_IF_DEVICE
        {
            // dump input tensor
            if (unlikely(isEnableDump)) {
                aicoreDump_.DoDump(devTaskCtx->GetDeviceTask(), "input", newTask, coreIdx);
            }
        }
#endif
        uint64_t encodeTaskId = EncodeTaskId(devTaskCtx, coreIdx, newTask);
        if (!devTaskCtx->isFirstTaskSend) {
            PerfMtTrace(PERF_TRACE_DEV_TASK_SEND_FIRST_LEAF_TASK, aicpuIdx_);
            devTaskCtx->isFirstTaskSend = 1;
        }
        aicoreHal_.SetReadyQueue(coreIdx, (encodeTaskId + 1));
        pendingIds_[coreIdx] = static_cast<uint32_t>(encodeTaskId & 0xFFFFFFFF);
        pendingResolveIndexList_[coreIdx] = 0;
        devTaskCtx->sendCnt[static_cast<int>(type)]++;

        DEV_IF_VERBOSE_DEBUG
        {
            sendTask_[coreIdx].push_back(TaskInfo(coreIdx, encodeTaskId, devTaskCtx->TaskId()));
            auto dyntask = reinterpret_cast<DynDeviceTask*>(devTaskCtx->GetDeviceTask());
            auto cceIndex = GetLeafIndex(devTaskCtx, static_cast<uint32_t>(newTask));
            uint64_t leafHash = dyntask->cceBinary[cceIndex].funcHash;
            DEV_VERBOSE_DEBUG("#trace.ltask.send: tid=%d task=%lu coreIdx=%d coreType=%d dtaskId=%lu leafHash=%lu",
                              schedIdx_, (uint64_t)REG_LOW_TASK_ID(newTask), coreIdx, static_cast<int>(type),
                              (uint64_t)devTaskCtx->TaskId(), leafHash);
        }
        DEV_VERBOSE_DEBUG("Send task %lx, origin taskid %lx, at core %d ,type:%d.", encodeTaskId, newTask, coreIdx,
                          static_cast<int>(type));
    }

    inline int32_t PushReadyQue(ReadyCoreFunctionQueue* readyQue, void* idList, uint32_t idCnt) const
    {
        const bool res = readyQue->TryEnqueue(reinterpret_cast<uint32_t*>(idList), idCnt);
        DEV_IF_NONDEVICE
        {
            if (!res) {
                DEV_ERROR(SchedErr::READY_QUEUE_OVERFLOW, "#sche.resolve.enqueue: readyQue: %s",
                          readyQue->Str().c_str());
                return DEVICE_MACHINE_ERROR;
            }
        }
        DEV_ASSERT(SchedErr::READY_QUEUE_OVERFLOW, res); // fail on queue overflow
        return DEVICE_MACHINE_OK;
    }

    [[gnu::hot]] inline int32_t ResolveDepForAllAiCore(SchDeviceTaskContext* devTaskCtx, CoreType type,
                                                       int coreIdxStart, int coreIdxEnd)
    {
        int32_t ret = DEVICE_MACHINE_OK;
        uint32_t resloveParallelIdx = 0;
        PerfMtBegin(static_cast<int>(PERF_EVT_RESOLVE_DEPENDENCE), aicpuIdx_);
        ResolveTaskContext resolveCtx[MAX_RESOLVE_TASK_NUM];
        uint32_t finishCnt = 0;
        for (int i = coreIdxStart; i < coreIdxEnd; i++) {
            if (pendingIds_[i] != AICORE_TASK_INIT || runningIds_[i] != AICORE_TASK_INIT) {
                // release finish core
                ret = ReleaseCoreByRegVal(type, i, resolveCtx, finishCnt, resloveParallelIdx);
                if (unlikely(ret != DEVICE_MACHINE_OK)) {
                    return ret;
                }
            }
        }

        auto& wrapManager = devTaskCtx->GetWrapManager();
        wrapManager.DispatchMixCoreTask();

        const bool isParallel = devTaskCtx->IsParallel();
        if (!enableL2CacheSch_ && !isParallel) {
            // send task to available core
            ReadyCoreFunctionQueue* readyQue = (type == CoreType::AIC) ? devTaskCtx->readyAicCoreFunctionQue :
                                                                         devTaskCtx->readyAivCoreFunctionQue;
            if (wrapManager.IsMixArch()) {
                ReadyCoreFunctionQueue* dieReadyQue = (type == CoreType::AIC) ? wrapManager.GetDieReadyAicQue() :
                                                                                wrapManager.GetDieReadyAivQue();
                if (dieReadyQue != readyQue) {
                    TryBatchSendTask(devTaskCtx, type, dieReadyQue, coreIdxStart, coreIdxEnd);
                }
            }
            TryBatchSendTask(devTaskCtx, type, readyQue, coreIdxStart, coreIdxEnd);
        }

        // resolve resolveCtx
        for (uint32_t i = 0; i < finishCnt; i++) {
            ret = ResolveDepWithDfx(type, resolveCtx[i].finishCoreIdx, resolveCtx[i].finishIds,
                                    resolveCtx[i].resolveIndexBase, resloveParallelIdx);
            if (enableFairSch_ && (resloveParallelIdx & (1U << devTaskCtx->parallelIdx))) {
                if (devTaskCtx->readyAicCoreFunctionQue->UnsafeSize() == 0 ||
                    devTaskCtx->readyAivCoreFunctionQue->UnsafeSize() == 0) {
                    ret = BatchPushReadyQueue(devTaskCtx);
                    if (unlikely(ret != DEVICE_MACHINE_OK)) {
                        return ret;
                    }
                }
            }
        }

        ret = BatchPushReadyQueForParallel(resloveParallelIdx);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return ret;
        }
        PerfMtEnd(static_cast<int>(PERF_EVT_RESOLVE_DEPENDENCE), aicpuIdx_);
        return ret;
    }

    int32_t BatchPushReadyQueForParallel(uint32_t resloveParallelIdx)
    {
        int32_t ret = DEVICE_MACHINE_OK;
        auto& parallelCtx = context_->schParallelDevTaskCtx;
        for (uint32_t i = parallelCtx.front; i < parallelCtx.rear; ++i) { // make sure the devtask priority
            uint32_t idx = i % npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM;
            if (resloveParallelIdx & (1U << idx)) {
                ret = BatchPushReadyQueue(context_->ParallelDeviceTaskCtx(idx));
                if (unlikely(ret != DEVICE_MACHINE_OK)) {
                    return ret;
                }
            }
        }
        return ret;
    }

    [[gnu::hot]] inline int32_t BatchPushReadyQueue(SchDeviceTaskContext* devTaskCtx)
    {
        int32_t ret = DEVICE_MACHINE_OK;
        uint32_t aicIndex = static_cast<uint32_t>(CoreType::AIC);
        uint32_t aivIndex = static_cast<uint32_t>(CoreType::AIV);
        auto& wrapManager = devTaskCtx->GetWrapManager();
        const bool isMixArch = wrapManager.IsMixArch();
        if (devTaskCtx->readyCount[aicIndex] > 0) {
            uint32_t needSendCnt = std::min(GetRunReadyCoreNum(CoreType::AIC), devTaskCtx->readyCount[aicIndex]);
            if (needSendCnt > 0) {
                devTaskCtx->readyCount[aicIndex] -= BatchSendTask(
                    devTaskCtx, CoreType::AIC, &devTaskCtx->readyIds[aicIndex][devTaskCtx->readyCount[aicIndex] - 1],
                    needSendCnt, aicStart_, adjAicEnd_, true);
            }
            DEV_VERBOSE_DEBUG("resolved new task, aic ready count: %u coretype:%u.", devTaskCtx->readyCount[aicIndex],
                              aicIndex);
            if (devTaskCtx->readyCount[aicIndex] > 0) {
                ReadyCoreFunctionQueue* targetReadyQue = devTaskCtx->readyAicCoreFunctionQue;
                if (isMixArch && EnableDieScheduling(devTaskCtx, CoreType::AIC, devTaskCtx->readyIds[aicIndex][0])) {
                    targetReadyQue = wrapManager.GetDieReadyAicQue();
                }
                ret = PushReadyQue(targetReadyQue, devTaskCtx->readyIds[aicIndex], devTaskCtx->readyCount[aicIndex]);
                if (unlikely(ret != DEVICE_MACHINE_OK)) {
                    return ret;
                }
            }
            devTaskCtx->readyCount[aicIndex] = 0;
        }

        if (devTaskCtx->readyCount[aivIndex] > 0) {
            uint32_t needSendCnt = std::min(GetRunReadyCoreNum(CoreType::AIV), devTaskCtx->readyCount[aivIndex]);
            if (needSendCnt > 0) {
                devTaskCtx->readyCount[aivIndex] -= BatchSendTask(
                    devTaskCtx, CoreType::AIV, &devTaskCtx->readyIds[aivIndex][devTaskCtx->readyCount[aivIndex] - 1],
                    needSendCnt, aivStart_, adjAivEnd_, true);
            }
            DEV_VERBOSE_DEBUG("resolved new task, aiv ready count: %u coretype: %u.", devTaskCtx->readyCount[aivIndex],
                              aivIndex);
            if (devTaskCtx->readyCount[aivIndex] > 0) {
                ReadyCoreFunctionQueue* targetReadyQue = devTaskCtx->readyAivCoreFunctionQue;
                if (isMixArch && EnableDieScheduling(devTaskCtx, CoreType::AIV, devTaskCtx->readyIds[aivIndex][0])) {
                    targetReadyQue = wrapManager.GetDieReadyAivQue();
                }
                ret = PushReadyQue(targetReadyQue, devTaskCtx->readyIds[aivIndex], devTaskCtx->readyCount[aivIndex]);
                if (unlikely(ret != DEVICE_MACHINE_OK)) {
                    return ret;
                }
            }
            devTaskCtx->readyCount[aivIndex] = 0;
        }
        return ret;
    }

    inline int32_t ResolveDepForAicpuTask(uint64_t& taskCount)
    {
        auto curSchDevTaskCtx = context_->GetCurSchDevTaskCtx();
        uint32_t parallelIdx = curSchDevTaskCtx->parallelIdx;
        auto deviceTask = reinterpret_cast<DynDeviceTask*>(curSchDevTaskCtx->GetDeviceTask());
        int32_t ret = aicpuTaskManager_.TaskProcess(taskCount, deviceTask, parallelIdx);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return ret;
        }
        return aicpuTaskManager_.TaskPoll(this, parallelIdx);
    }

    [[gnu::hot]] inline int32_t ResolveWhenSyncMode(CoreType type, uint32_t finTaskId, uint32_t finTaskState,
                                                    int coreIdx, uint32_t& resloveParallelIdx)
    {
        int32_t ret = DEVICE_MACHINE_OK;
        if (finTaskId == pendingIds_[coreIdx] && finTaskState == TASK_FIN_STATE) {
            DEV_VERBOSE_DEBUG("core index: %d, PendingTask Finished."
                              " pending: %x.",
                              coreIdx, pendingIds_[coreIdx]);
            ret = ResolveDepWithDfx(type, coreIdx, finTaskId, 0, resloveParallelIdx);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }
            pendingIds_[coreIdx] = AICORE_TASK_INIT;
            pendingResolveIndexList_[coreIdx] = 0;
            context_->coreStatusMgr.AddRunAndPendCoreIdx(coreIdx, static_cast<int>(type));
        }
        return ret;
    }

    static uint64_t RuntimeCopyOutResolveCounterDecode(uint64_t aicpuCallCode) { return aicpuCallCode & 0xffff; }

    inline void RecordResolveTask(ResolveTaskContext* ctx, uint32_t& finishCnt, int coreIdx, uint32_t taskId,
                                  int indexBase)
    {
        if (unlikely(finishCnt >= MAX_RESOLVE_TASK_NUM)) {
            DEV_ERROR(SchedErr::CORE_INFO_INVALID,
                      "#sche.resolve.overflow: resolveCtx overflow guarded: aicpu[%d] schedIdx=%d aicpuNum=%d "
                      "finishCnt=%u cap=%u coreIdx=%d taskId=%#x aic[%d,%d) aiv[%d,%d). DROPPED.",
                      aicpuIdx_, schedIdx_, aicpuNum_, finishCnt, static_cast<uint32_t>(MAX_RESOLVE_TASK_NUM), coreIdx,
                      taskId, aicStart_, aicEnd_, aivStart_, aivEnd_);
            return;
        }
        ctx[finishCnt].finishIds = taskId;
        ctx[finishCnt].resolveIndexBase = indexBase;
        ctx[finishCnt].finishCoreIdx = coreIdx;
        finishCnt++;
    }

    inline int32_t ReleaseCoreByRegVal(CoreType type, int coreIdx, [[maybe_unused]] ResolveTaskContext* ctx,
                                       [[maybe_unused]] uint32_t& finishCnt, uint32_t& resloveParallelIdx)
    {
        uint64_t finTaskRegVal = aicoreHal_.GetFinishedTask(coreIdx);
        [[maybe_unused]] uint32_t aicpuCallCode = finTaskRegVal >> 32;
        uint32_t finTaskId = REG_LOW_TASK_ID(finTaskRegVal);
        uint32_t finTaskState = REG_LOW_TASK_STATE(finTaskRegVal);
        DEV_VERBOSE_DEBUG("resolve task core index: %d, finishtaskid:%x, finishstate: %u.", coreIdx, finTaskId,
                          finTaskState);

        return (this->*releaseCoreByRegValFn_)(type, coreIdx, ctx, finishCnt, resloveParallelIdx, finTaskRegVal,
                                               aicpuCallCode, finTaskId, finTaskState);
    }

    inline int32_t ReleaseCoreByRegValByAsyncMode(CoreType type, int coreIdx, ResolveTaskContext* ctx,
                                                  uint32_t& finishCnt, uint32_t& resloveParallelIdx,
                                                  uint64_t finTaskRegVal, uint32_t aicpuCallCode, uint32_t finTaskId,
                                                  uint32_t finTaskState)
    {
        int32_t ret = DEVICE_MACHINE_OK;
        auto& pendingIdRef = pendingIds_[coreIdx];
        auto& pendingResolveIndexBaseRef = pendingResolveIndexList_[coreIdx];
        auto& runningIdRef = runningIds_[coreIdx];
        auto& runningResolveIndexBaseRef = runningResolveIndexList_[coreIdx];
        if (likely(finTaskId == pendingIdRef && finTaskState == TASK_FIN_STATE)) {
            // pending task is finished, resolve both running and pending task.
            DEV_VERBOSE_DEBUG("Pending Finished: core:%d pending:%x,%d running:%x,%d", coreIdx, pendingIdRef,
                              pendingResolveIndexBaseRef, runningIdRef, runningResolveIndexBaseRef);
            uint32_t runningIdValue = runningIdRef;
            int runningResolveIndexBaseValue = runningResolveIndexBaseRef;
            uint32_t pendingIdValue = pendingIdRef;
            int pendingResolveIndexBaseValue = pendingResolveIndexBaseRef;
            runningIdRef = AICORE_TASK_INIT;
            runningResolveIndexBaseRef = 0;
            pendingIdRef = AICORE_TASK_INIT; // ResolveDepWithDfx depend this line
            pendingResolveIndexBaseRef = 0;
            context_->coreStatusMgr.AddRunAndPendCoreIdx(coreIdx, static_cast<int>(type));
            if (runningIdValue != AICORE_TASK_INIT) {
                RecordResolveTask(ctx, finishCnt, coreIdx, runningIdValue, runningResolveIndexBaseValue);
            }
            RecordResolveTask(ctx, finishCnt, coreIdx, pendingIdValue, pendingResolveIndexBaseValue);
        } else if (unlikely(finTaskId == pendingIdRef && aicpuCallCode != 0)) {
            // pending task is copyout, reolve both running and pending task.
            DEV_VERBOSE_DEBUG("Pending Copyout: core:%d pending:%x,%d running:%x,%d", coreIdx, pendingIdRef,
                              pendingResolveIndexBaseRef, runningIdRef, runningResolveIndexBaseRef);
            uint32_t copyOutResolveCounter = RuntimeCopyOutResolveCounterDecode(aicpuCallCode);
            uint32_t runningIdValueCopyout = runningIdRef;
            int runningResolveIndexBaseValueCopyout = runningResolveIndexBaseRef;
            uint32_t pendingIdValue = pendingIdRef;
            int pendingResolveIndexBaseValue = pendingResolveIndexBaseRef;
            runningIdRef = pendingIdRef;
            runningResolveIndexBaseRef = copyOutResolveCounter + 1;
            pendingIdRef = AICORE_TASK_INIT; // ResolveDepWithDfx depend this line
            pendingResolveIndexBaseRef = 0;
            context_->coreStatusMgr.AddPendReadyCoreIdx(static_cast<int>(type));
            if (runningIdValueCopyout != AICORE_TASK_INIT) {
                RecordResolveTask(ctx, finishCnt, coreIdx, runningIdValueCopyout, runningResolveIndexBaseValueCopyout);
            }
            ret = ResolveCopyOutDepDyn(copyOutResolveCounter, pendingIdValue, pendingResolveIndexBaseValue,
                                       resloveParallelIdx);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }
        } else if (finTaskId == pendingIdRef && finTaskState == TASK_ACK_STATE) {
            // pending task is acknowledged, resolve running task. And move pending to running
            DEV_VERBOSE_DEBUG("Pending Acknowledged: core:%d pending:%x,%d running:%x,%d", coreIdx, pendingIdRef,
                              pendingResolveIndexBaseRef, runningIdRef, runningResolveIndexBaseRef);
            DEV_IF_VERBOSE_DEBUG { recvAckTask_[coreIdx].push_back(TaskInfo(coreIdx, finTaskId, 0xFFFFFFFF)); }
            DEV_VERBOSE_DEBUG("#trace.ltask.ack: tid=%d task=%u dtaskId=%lu coreIdx=%d", schedIdx_, finTaskId,
                              context_->ParallelDeviceTaskCtx(ParallelIndex(finTaskId))->TaskId(), coreIdx);
            uint32_t runningIdValueAck = runningIdRef;
            int runningResolveIndexBaseValueAck = runningResolveIndexBaseRef;
            runningIdRef = finTaskId;
            runningResolveIndexBaseRef = pendingResolveIndexBaseRef;
            pendingIdRef = AICORE_TASK_INIT; // ResolveDepWithDfx depend this line
            pendingResolveIndexBaseRef = 0;
            context_->coreStatusMgr.AddPendReadyCoreIdx(static_cast<int>(type));
            if (runningIdValueAck != AICORE_TASK_INIT) {
                RecordResolveTask(ctx, finishCnt, coreIdx, runningIdValueAck, runningResolveIndexBaseValueAck);
            }
        } else if (finTaskId == runningIdRef && finTaskState == TASK_FIN_STATE) {
            // running task is finished, resolve running task. Pending task is unmodified
            DEV_VERBOSE_DEBUG("Running finished: core:%d pending:%x,%d running:%x,%d", coreIdx, pendingIdRef,
                              pendingResolveIndexBaseRef, runningIdRef, runningResolveIndexBaseRef);
            uint32_t runningIdValue = runningIdRef;
            int runningResolveIndexBaseValue = runningResolveIndexBaseRef;
            runningIdRef = AICORE_TASK_INIT;
            runningResolveIndexBaseRef = 0;
            if (pendingIdRef == AICORE_TASK_INIT) {
                context_->coreStatusMgr.AddRunReadyCoreIdx(coreIdx, static_cast<int>(type));
            }
            RecordResolveTask(ctx, finishCnt, coreIdx, runningIdValue, runningResolveIndexBaseValue);
        } else if (unlikely(finTaskId == runningIdRef && aicpuCallCode != 0)) {
            // running task is copyout, resolve running task. Pending task is unmodified
            DEV_VERBOSE_DEBUG("Running copyout: core:%d pending:%x,%d running:%x,%d", coreIdx, pendingIdRef,
                              pendingResolveIndexBaseRef, runningIdRef, runningResolveIndexBaseRef);
            uint32_t copyOutResolveCounter = RuntimeCopyOutResolveCounterDecode(aicpuCallCode);
            uint32_t runningIdValue = runningIdRef;
            int runningResolveIndexBaseValue = runningResolveIndexBaseRef;
            runningResolveIndexBaseRef = copyOutResolveCounter + 1;
            ret = ResolveCopyOutDepDyn(copyOutResolveCounter, runningIdValue, runningResolveIndexBaseValue,
                                       resloveParallelIdx);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }
        } else {
            DEV_VERBOSE_DEBUG("Warning, maybe inconsistent state. coreidx: %d,finTask: %lx,pending: %x,running: %x.",
                              coreIdx, finTaskRegVal, pendingIdRef, runningIdRef);
        }
        return ret;
    }

    inline int32_t ReleaseCoreByRegValBySyncMode(CoreType type, int coreIdx, [[maybe_unused]] ResolveTaskContext* ctx,
                                                 [[maybe_unused]] uint32_t& finishCnt, uint32_t& resloveParallelIdx,
                                                 [[maybe_unused]] uint64_t finTaskRegVal,
                                                 [[maybe_unused]] uint32_t aicpuCallCode, uint32_t finTaskId,
                                                 uint32_t finTaskState)
    {
        int32_t ret = ResolveWhenSyncMode(type, finTaskId, finTaskState, coreIdx, resloveParallelIdx);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return ret;
        }
        return ret;
    }

    inline void PushAicpuTaskQueue(SchDeviceTaskContext* devTaskCtx, uint64_t taskId)
    {
        DEV_VERBOSE_DEBUG("PushAicpuTaskQueue taskId = %lu", taskId);
        PushReadyQue(devTaskCtx->readyAicpuFunctionQue, &taskId, 1);
    }

    inline bool TrySendTaskDirectly(SchDeviceTaskContext* devTaskCtx, int coreType, uint32_t taskId)
    {
        if (context_->coreStatusMgr.GetCoreRunReadyCnt(coreType) > 0) {
            uint32_t coreIdx = context_->coreStatusMgr.GetRunReadyCoreIdx(
                coreType, context_->coreStatusMgr.GetCoreRunReadyCnt(coreType) - 1);
            context_->coreStatusMgr.RemoveReadyCoreIdxTail(coreIdx, coreType);
            context_->coreStatusMgr.RemovePendReadyCoreIdx(coreType);
            DEV_VERBOSE_DEBUG("Direct send task when task ready %x.", taskId);
            SendTaskToAiCore(devTaskCtx, static_cast<CoreType>(coreType), coreIdx, taskId);
            return true;
        }

        if (context_->coreStatusMgr.GetCorePendReadyCnt(coreType) == 0) {
            return false;
        }

        if (enableFairSch_ && IsExistOtherAicpuIdle(static_cast<CoreType>(coreType))) {
            return false;
        }

        int startIdx;
        int coreNum;
        int idx = static_cast<int>(context_->coreStatusMgr.GetLastPendReadyCoreIdx(coreType));
        if (coreType == static_cast<int>(CoreType::AIC)) {
            startIdx = aicStart_;
            coreNum = adjAicEnd_ - aicStart_;
        } else {
            startIdx = aivStart_;
            coreNum = adjAivEnd_ - aivStart_;
        }
        while (pendingIds_[idx] != AICORE_TASK_INIT) {
            idx = startIdx + (idx - startIdx + 1) % (coreNum);
        }
        context_->coreStatusMgr.SetLastPendReadyCoreIdx(
            coreType, static_cast<uint32_t>(startIdx + (idx - startIdx + 1) % (coreNum)));
        context_->coreStatusMgr.RemovePendReadyCoreIdx(coreType);
        DEV_VERBOSE_DEBUG("Direct send task when task ready %x.", taskId);
        SendTaskToAiCore(devTaskCtx, static_cast<CoreType>(coreType), idx, taskId);
        return true;
    }

    inline int32_t PushReadyTask(SchDeviceTaskContext* devTaskCtx, int coreType, uint64_t taskId)
    {
        DEV_VERBOSE_DEBUG("#trace.ltask.resolve: tid=%d task=%lu firstBatch=%d dtaskId=%lu", schedIdx_, taskId, 0,
                          devTaskCtx->TaskId());
        int32_t ret = DEVICE_MACHINE_OK;
        if (enableL2CacheSch_ && (!devTaskCtx->IsParallel()) && TrySendTaskDirectly(devTaskCtx, coreType, taskId)) {
            return DEVICE_MACHINE_OK;
        }

        if (unlikely(devTaskCtx->readyCount[coreType] == READY_ID_FIX_CACHE_NUM)) {
            ReadyCoreFunctionQueue* readyQue = coreType == static_cast<int>(CoreType::AIC) ?
                                                   devTaskCtx->readyAicCoreFunctionQue :
                                                   devTaskCtx->readyAivCoreFunctionQue;
            auto& wrapManager = devTaskCtx->GetWrapManager();
            if (wrapManager.IsMixArch() &&
                EnableDieScheduling(devTaskCtx, static_cast<CoreType>(coreType), devTaskCtx->readyIds[coreType][0])) {
                readyQue = coreType == static_cast<int>(CoreType::AIC) ? wrapManager.GetDieReadyAicQue() :
                                                                         wrapManager.GetDieReadyAivQue();
            }
            ret = PushReadyQue(readyQue, devTaskCtx->readyIds[coreType], devTaskCtx->readyCount[coreType]);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }
            devTaskCtx->readyCount[coreType] = 0;
        }
        devTaskCtx->readyIds[coreType][devTaskCtx->readyCount[coreType]++] = taskId;
        return ret;
    }

    inline uint64_t GetCostModelTaskTime(uint64_t coreIdx, uint64_t taskId, uint64_t currentTime)
    {
        DeviceTask* curDevTask = context_->GetCurSchDevTaskCtx()->GetDeviceTask();
        auto funcId = FuncID(taskId);
        auto dyntask = reinterpret_cast<DynDeviceTask*>(curDevTask);
        auto costModelData = reinterpret_cast<CostModel::ModelData*>(curDevTask->costModelData);
        if (costModelData == nullptr)
            return 0;
        auto source = dyntask->GetDynFuncDataCacheList()[funcId].devFunc;
        auto opIndex = TaskID(taskId);
        auto leafFunctionIdx = source->GetOperationAttrCalleeIndex(opIndex);
        auto timeCost = costModelData->functionTime[leafFunctionIdx];
        auto header = dyntask->GetDynFuncDataList();
        auto dyndata = reinterpret_cast<DynFuncData*>(&header->At(0));
        auto opAttrs = &dyndata->opAttrs[dyndata->opAtrrOffsets[TaskID(taskId)]];
        auto psgId = opAttrs[0];
        // dtaskId - funcId - leaf function Id - psgId
        std::string name = std::to_string(context_->GetCurSchDevTaskCtx()->TaskId()) + '-' + std::to_string(funcId) +
                           '-' + std::to_string(opIndex) + '-' + std::to_string(psgId);
        PerfMtEvent(PERF_EVT_TASK, coreIdx + PERF_AICORE_THREAD_START, currentTime, currentTime + timeCost, name);
        return timeCost;
    }

    inline int32_t ResolveDynStitched(SchDeviceTaskContext* deviceTaskCtx, DynDeviceTask* dyntask, int origfunc,
                                      int origop, int coreIdx = 0)
    {
        int32_t ret = DEVICE_MACHINE_OK;
        auto& duppedData = dyntask->GetDynFuncDataCacheList()[origfunc].duppedData;
        auto& stitchList = duppedData->GetOperationStitch(origop);
        auto cceBinary = dyntask->cceBinary;
        for (auto* node = stitchList.Head(); node != nullptr; node = node->Next()) {
            uint32_t listSize = node->Size();
            for (uint32_t i = 0; i < listSize; i++) {
                uint32_t id = node->At(i);
                auto funcId = FuncID(id);
                auto opIndex = TaskID(id);
                auto predCounts = dyntask->dynFuncDataCacheList[funcId].predCount;
                bool needProcess = predCounts[opIndex] == 1 ||
                                   __atomic_sub_fetch(&predCounts[opIndex], 1, __ATOMIC_RELAXED) == 0;
                if (!needProcess) {
                    continue;
                }

                auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;
                auto coreType = cceBinary[callList[opIndex]].coreType;
                if (unlikely(coreType == static_cast<int>(CoreType::HUB))) {
                    ret = ResolveDepDyn(deviceTaskCtx, id, 0, coreIdx);
                    if (unlikely(ret != DEVICE_MACHINE_OK)) {
                        return ret;
                    }
                    deviceTaskCtx->resolveHubCnt++;
                } else if (unlikely(coreType == static_cast<int>(CoreType::HUB_MIX))) {
                    ResolveHubMixDepDyn(deviceTaskCtx, id, coreIdx);
                    deviceTaskCtx->resolveHubCnt++;
                } else if (coreType == static_cast<int>(MachineType::AICPU)) {
                    PushAicpuTaskQueue(deviceTaskCtx, id);
                } else {
                    ret = PushReadyTask(deviceTaskCtx, static_cast<int>(coreType), id);
                    if (unlikely(ret != DEVICE_MACHINE_OK)) {
                        return ret;
                    }
                }
            }
        }
        return ret;
    }

    inline int GetRootIndex(SchDeviceTaskContext* deviceTaskCtx, uint32_t taskId) const
    {
        auto dyntask = reinterpret_cast<DynDeviceTask*>(deviceTaskCtx->GetDeviceTask());
        auto funcId = FuncID(taskId);
        auto func = dyntask->dynFuncDataCacheList[funcId].devFunc;
        return func->GetRootIndex();
    }

    inline int GetLeafIndex(SchDeviceTaskContext* deviceTaskCtx, uint32_t taskId) const
    {
        auto dyntask = reinterpret_cast<DynDeviceTask*>(deviceTaskCtx->GetDeviceTask());
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;
        return callList[opIndex];
    }

    inline DevAscendFunctionDuppedData* GetDuppedData(DeviceTask* deviceTask, uint32_t taskId) const
    {
        auto dyntask = reinterpret_cast<DynDeviceTask*>(deviceTask);
        auto funcId = FuncID(taskId);
        return dyntask->dynFuncDataCacheList[funcId].duppedData;
    }

    inline void ResolveHubMixDepDynStitched(SchDeviceTaskContext* deviceTaskCtx, DynDeviceTask* dyntask, int origfunc,
                                            int origop, int coreIdx = 0)
    {
        auto& wrapManager = deviceTaskCtx->GetWrapManager();
        auto cceBinary = dyntask->cceBinary;
        auto& duppedData = dyntask->GetDynFuncDataCacheList()[origfunc].duppedData;
        auto& stitchList = duppedData->GetOperationStitch(origop);
        for (auto* node = stitchList.Head(); node != nullptr; node = node->Next()) {
            uint32_t listSize = node->Size();
            uint32_t taskIds[MAX_WRAP_TASK_NUM];
            uint8_t mixResoruceType = 0;
            for (uint32_t i = 0; i < listSize; i++) {
                uint32_t id = node->At(i);
                auto funcId = FuncID(id);
                auto opIndex = TaskID(id);
                auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;
                auto curBinary = cceBinary[callList[opIndex]];
                int32_t wrapAicoreIdx = WrapManager::GetWrapAicoreIdx(curBinary.coreType, curBinary.wrapVecId);
                taskIds[wrapAicoreIdx] = id;
                mixResoruceType = curBinary.mixResourceType;
            }
            DEV_VERBOSE_DEBUG("ResolveHubMixDepDynStitched listSize = %u, should equal mixTaskNum: %u", listSize,
                              GetTaskNumByMixResType(mixResoruceType));
            wrapManager.ResolveDepForOneMix(taskIds, mixResoruceType, coreIdx);
        }
    }

    inline void ResolveHubMixDepDyn(SchDeviceTaskContext* deviceTaskCtx, uint64_t hubMixId, int coreIdx = 0)
    {
        auto dyntask = reinterpret_cast<DynDeviceTask*>(deviceTaskCtx->GetDeviceTask());
        auto funcId = FuncID(hubMixId);
        auto opIndex = TaskID(hubMixId);
        auto& wrapManager = deviceTaskCtx->GetWrapManager();
        auto cceBinary = dyntask->cceBinary;
        auto func = dyntask->dynFuncDataCacheList[funcId].devFunc;
        auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;
        size_t succSize;
        auto succList = func->GetOperationDepGraphSuccAddr(opIndex, succSize);
        if (succSize != 0) {
            uint32_t taskIds[MAX_WRAP_TASK_NUM];
            uint8_t mixResoruceType = 0;
            for (size_t i = 0; i < succSize; i++) {
                auto succIdx = succList[i];
                auto curBinary = cceBinary[callList[succIdx]];
                int32_t wrapAicoreIdx = WrapManager::GetWrapAicoreIdx(curBinary.coreType, curBinary.wrapVecId);
                taskIds[wrapAicoreIdx] = MakeTaskID(funcId, succIdx);
                mixResoruceType = curBinary.mixResourceType;
            }
            DEV_VERBOSE_DEBUG("ResolveHubMixDepDyn succSize = %lu, should equal mixTaskNum: %u", succSize,
                              GetTaskNumByMixResType(mixResoruceType));
            wrapManager.ResolveDepForOneMix(taskIds, mixResoruceType, coreIdx);
        }
        ResolveHubMixDepDynStitched(deviceTaskCtx, dyntask, funcId, opIndex, coreIdx);
    }

    inline int32_t ResolveDepDyn(SchDeviceTaskContext* deviceTaskCtx, uint64_t finishId, size_t resolveIndexBase = 0,
                                 int coreIdx = 0)
    {
        int32_t ret = DEVICE_MACHINE_OK;
        auto dyntask = reinterpret_cast<DynDeviceTask*>(deviceTaskCtx->GetDeviceTask());
        auto funcId = FuncID(finishId);
        auto opIndex = TaskID(finishId);

        auto cceBinary = dyntask->cceBinary;
        auto func = dyntask->dynFuncDataCacheList[funcId].devFunc;
        if (unlikely(func->IsTailTask(opIndex))) {
            return DEVICE_MACHINE_OK;
        }
        auto predCounts = dyntask->dynFuncDataCacheList[funcId].predCount;
        auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;

        size_t succIndexSize;
        const int* succIndexList = func->GetOperationDepGraphCopyOutResolveSuccIndexAddr(opIndex, succIndexSize);
        size_t succSize;
        auto succList = func->GetOperationDepGraphSuccAddr(opIndex, succSize);
        for (size_t i = succIndexList[resolveIndexBase]; i < succSize; i++) {
            auto succIdx = succList[i];
            if (predCounts[succIdx] == 1 || __atomic_sub_fetch(&predCounts[succIdx], 1, __ATOMIC_RELAXED) == 0) {
                auto id = MakeTaskID(funcId, succIdx);
                auto coreType = cceBinary[callList[succIdx]].coreType;
                if (unlikely(coreType == static_cast<int>(CoreType::HUB))) {
                    if (!func->IsDeadEndHub(succIdx)) {
                        ret = ResolveDepDyn(deviceTaskCtx, id, resolveIndexBase, coreIdx);
                    }
                    if (unlikely(ret != DEVICE_MACHINE_OK)) {
                        return ret;
                    }
                    deviceTaskCtx->resolveHubCnt++;
                } else if (unlikely(coreType == static_cast<int>(CoreType::HUB_MIX))) {
                    ResolveHubMixDepDyn(deviceTaskCtx, id, coreIdx);
                    deviceTaskCtx->resolveHubCnt++;
                } else if (unlikely(coreType == static_cast<int>(MachineType::AICPU))) {
                    PushAicpuTaskQueue(deviceTaskCtx, id);
                } else {
                    ret = PushReadyTask(deviceTaskCtx, static_cast<int>(coreType), id);
                    if (unlikely(ret != DEVICE_MACHINE_OK)) {
                        return ret;
                    }
                }
            }
        }

        ret = ResolveDynStitched(deviceTaskCtx, dyntask, funcId, opIndex, coreIdx);
        return ret;
    }

    inline int32_t ResolveCopyOutDepDyn(uint32_t currResolveIndex, uint64_t taskId, uint32_t resolveIndexBase,
                                        uint32_t& resloveParallelIdx)
    {
        uint32_t taskParallelIndex = ParallelIndex(taskId);
        resloveParallelIdx |= (1U << taskParallelIndex);
        SchDeviceTaskContext* deviceTaskCtx = context_->ParallelDeviceTaskCtx(taskParallelIndex);

        int32_t ret = DEVICE_MACHINE_OK;
        auto dyntask = reinterpret_cast<DynDeviceTask*>(deviceTaskCtx->GetDeviceTask());
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);

        auto cceBinary = dyntask->cceBinary;
        auto func = dyntask->dynFuncDataCacheList[funcId].devFunc;
        auto predCounts = dyntask->dynFuncDataCacheList[funcId].predCount;
        auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;

        size_t succIndexSize;
        const int* succIndexList = func->GetOperationDepGraphCopyOutResolveSuccIndexAddr(opIndex, succIndexSize);
        size_t succSize;
        const uint32_t* succList = func->GetOperationDepGraphSuccAddr(opIndex, succSize);
        // here we don't use resolveIndexBase + 1, because at the beginning, resolveIndexBase is 0. And we resolve from
        // 0.
        for (int i = succIndexList[resolveIndexBase]; i < succIndexList[currResolveIndex + 1]; i++) {
            auto succIdx = succList[i];
            if (predCounts[succIdx] == 1 || __atomic_sub_fetch(&predCounts[succIdx], 1, __ATOMIC_RELAXED) == 0) {
                auto id = MakeTaskID(funcId, succIdx);
                auto coreType = cceBinary[callList[succIdx]].coreType;
                if (unlikely(coreType == static_cast<int>(CoreType::HUB))) {
                    ret = ResolveDepDyn(deviceTaskCtx, id);
                    if (unlikely(ret != DEVICE_MACHINE_OK)) {
                        return ret;
                    }
                    deviceTaskCtx->resolveHubCnt++;
                } else if (unlikely(coreType == static_cast<int>(CoreType::HUB_MIX))) {
                    ResolveHubMixDepDyn(deviceTaskCtx, id);
                    deviceTaskCtx->resolveHubCnt++;
                } else if (unlikely(coreType == static_cast<int>(MachineType::AICPU))) {
                    PushAicpuTaskQueue(deviceTaskCtx, id);
                } else {
                    ret = PushReadyTask(deviceTaskCtx, static_cast<int>(coreType), id);
                    if (unlikely(ret != DEVICE_MACHINE_OK)) {
                        return ret;
                    }
                }
            }
        }
        return ret;
    }

    inline int32_t ResolveDepWithDfx(CoreType type, int coreIdx, uint64_t finishId, size_t resolveIndexBase,
                                     uint32_t& resloveParallelIdx)
    {
        uint32_t taskParallelIndex = ParallelIndex(finishId);
        resloveParallelIdx |= (1U << taskParallelIndex);
        SchDeviceTaskContext* deviceTaskCtx = context_->ParallelDeviceTaskCtx(taskParallelIndex);
        int32_t ret = DEVICE_MACHINE_OK;
        ret = ResolveDepDyn(deviceTaskCtx, finishId, resolveIndexBase, coreIdx);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return ret;
        }
        DEV_VERBOSE_DEBUG("[Call]: Core %d Dispatch Task: %lu, %u, %u, %u", coreIdx, deviceTaskCtx->TaskId(),
                          FuncID(finishId), TaskID(finishId), DevTaskDcciFlag(finishId));
        DfxProcAfterFinishTask(deviceTaskCtx, coreIdx, finishId);
        context_->coreStatusMgr.WaitTaskCnt(static_cast<int>(type))--;
        return ret;
    }

    inline bool IsExistOtherAicpuIdle(CoreType type)
    {
        int idx = (schedIdx_ + 1) % aicpuNum_;
        while (idx != schedIdx_) {
            if (threadStatus.isAicpuIdle[static_cast<int>(type)][idx].load(std::memory_order_relaxed) == true) {
                return true;
            }
            idx = (idx + 1) % aicpuNum_;
        }
        return false;
    }

    inline bool EnableDieScheduling(SchDeviceTaskContext* deviceTaskCtx, CoreType type, uint32_t taskId)
    {
        auto duppedData = GetDuppedData(deviceTaskCtx->GetDeviceTask(), taskId);
        auto loopDieId = duppedData->loopDieId_;
        if (loopDieId < 0 ||
            (loopDieId !=
             static_cast<int8_t>(deviceTaskCtx->GetWrapManager()
                                     .GetDieId()))) { // prevent parallel_loop incorrectly, task depends on other die
            return false;
        }
        if (!enableFairSch_) {
            return true;
        }
        int schedStart = 0;
        int schedEnd = 0;
        deviceTaskCtx->GetWrapManager().GetDieSchedIdRange(schedStart, schedEnd, aicpuNum_);
        const auto& idleMap = threadStatus.isAicpuIdle[static_cast<int>(type)];
        for (int idx = schedStart; idx < schedEnd; idx++) {
            if (idleMap[idx].load(std::memory_order_relaxed) == true) {
                return true;
            }
        }
        return false;
    }

    inline void AicpuIsBusy(CoreType type)
    {
        if (threadStatus.isAicpuIdle[static_cast<int>(type)][schedIdx_] != false) {
            threadStatus.isAicpuIdle[static_cast<int>(type)][schedIdx_].store(false, std::memory_order_relaxed);
        }
    }

    inline void AicpuIsIdle(CoreType type)
    {
        if (threadStatus.isAicpuIdle[static_cast<int>(type)][schedIdx_] != true) {
            threadStatus.isAicpuIdle[static_cast<int>(type)][schedIdx_].store(true, std::memory_order_relaxed);
        }
    }

    inline void Init(int threadIdx, DevStartArgs* startArgs, DeviceArgs* deviceArgs, int schedIdx,
                     int arbitratedScheNum)
    {
        archInfo_ = deviceArgs->archInfo;
        aicNum_ = static_cast<int32_t>(deviceArgs->nrAic);
        aivNum_ = static_cast<int32_t>(deviceArgs->nrAiv);
        aicpuNum_ = arbitratedScheNum;
        aicpuIdx_ = threadIdx;
        schedIdx_ = schedIdx;
        aicValidNum_ = deviceArgs->nrValidAic;
        hasAicpuTask_ = deviceArgs->hasAicpuTask;
        enableEslModel_ = deviceArgs->enableEslModel;
        // Force-disable post-handshake control-core.
        constexpr bool kForceDisableControlCore = true;
        disableControlCore_ = kForceDisableControlCore || (startArgs->devProg->GetParallelism() > 1);
        aicoreHal_.Init(deviceArgs, &aicoreProf_);
        validGetPgMask_ = deviceArgs->validGetPgMask;
        runningIds_.fill(AICORE_STATUS_INIT);
        pendingIds_.fill(AICORE_STATUS_INIT);
        runningResolveIndexList_.fill(0);
        pendingResolveIndexList_.fill(0);
        pingPongFlag_.fill(0);
        isSendStop = false;
        taskCtrlDequeFinish = false;
        DEV_IF_DEVICE { isOpenPerf_ = ((DevDfxArgs*)deviceArgs->devDfxArgAddr)->isOpenPerfTrace; }
        if (IsNeedProcAicpuTask()) {
            DEV_VERBOSE_DEBUG("Init aicpu task manager");
            aicpuTaskManager_.InitDeviceArgs(deviceArgs);
        }
        context_->Init(deviceArgs, schedIdx);

#if ENABLE_TENSOR_DUMP
        DEV_IF_DEVICE
        {
            isEnableDump = startArgs->devProg->devArgs.hostPid != 0;
            if (unlikely(isEnableDump)) {
                aicoreDump_.Init(startArgs, schedIdx);
            }
        }
#endif

        if (deviceArgs->machineConfig != static_cast<uint8_t>(MachineScheduleConfig::DEFAULT_SCH)) {
            if (aicpuNum_ > 1) {
                enableFairSch_ = static_cast<uint8_t>(deviceArgs->machineConfig) &
                                 static_cast<uint8_t>(MachineScheduleConfig::MULTI_CORE_FAIR_SCH);
            }
            enableL2CacheSch_ = static_cast<uint8_t>(deviceArgs->machineConfig) &
                                static_cast<uint8_t>(MachineScheduleConfig::L2CACHE_AFFINITY_SCH);
        }
        UpdateAiCoreBlockIndexSection();
        if constexpr (IsDeviceMode()) {
            aicoreHal_.MapRegistersForAllCores(aicNum_);
            aicoreProf_.ProfInit(deviceArgs);
        } else {
            aicoreHal_.SetTaskTimeCost([this](uint64_t coreIdx, uint64_t taskId, uint64_t time) {
                return GetCostModelTaskTime(coreIdx, taskId, time);
            });
            eslModelReplayMgr_.Init(context_);
            aicoreHal_.SetEslModelReplayManager(&eslModelReplayMgr_);
        }
        firstLock[static_cast<int>(CoreType::AIC)] = true;
        firstLock[static_cast<int>(CoreType::AIV)] = true;
        aicoreDevTaskInited = false;
        DEV_INFO("Init aicore manager: aicNum=%d, aivNum=%d, schAicpuNum=%d, aicpuIdx=%d, "
                 "aicValidNum=%d, aicoreHal.regAddrs=%p, sharedBuffer=%p, machineConfig=%u.",
                 aicNum_, aivNum_, aicpuNum_, aicpuIdx_, aicValidNum_, aicoreHal_.GetRegAddrs(),
                 (void*)aicoreHal_.GetSharedBuffer(), static_cast<uint8_t>(deviceArgs->machineConfig));
    }

    inline SchDeviceTaskContext* HandShakeTryPreFetchDevTask(bool& needSendAic, bool& needSendAiv)
    {
        FillParallelDevtaskCtx();
        if (!context_->DevTaskEmpty()) {
            auto deviceTaskCtx = context_->FrontDevTaskCtx();
            DEV_VERBOSE_DEBUG("#trace.dtask.start: tid=%d taskId=%lu coreFunctionCnt=%lu isLast=%d", schedIdx_,
                              deviceTaskCtx->TaskId(), deviceTaskCtx->GetDeviceTask()->coreFunctionCnt,
                              reinterpret_cast<DynDeviceTask*>(deviceTaskCtx->GetDeviceTask())->IsLastTask() ? 1 : 0);
            InitDevTask(deviceTaskCtx);
            needSendAic = (deviceTaskCtx->readyAicCoreFunctionQue->UnsafeSize() > 0);
            needSendAiv = (deviceTaskCtx->readyAivCoreFunctionQue->UnsafeSize() > 0);
            DEV_DEBUG("hand shake prefetch dev task success: needSendAic=%d, needSendAiv=%d", needSendAic, needSendAiv);
            return deviceTaskCtx;
        }
        return nullptr;
    }

    inline void HandShakePostProc(SchDeviceTaskContext* schDeviceTaskCtx, bool needSendAic, bool needSendAiv)
    {
        // send task by left ready core
        if (needSendAic) {
            __sync_synchronize();
            TryBatchSendTask(schDeviceTaskCtx, CoreType::AIC, schDeviceTaskCtx->readyAicCoreFunctionQue, aicStart_,
                             adjAicEnd_);
        }
        if (needSendAiv) {
            __sync_synchronize();
            TryBatchSendTask(schDeviceTaskCtx, CoreType::AIV, schDeviceTaskCtx->readyAivCoreFunctionQue, aivStart_,
                             adjAivEnd_);
        }

        if (schDeviceTaskCtx) {
            schDeviceTaskCtx->CountCoreTaskSent(context_->coreStatusMgr.WaitTaskCnt(static_cast<int>(CoreType::AIC)),
                                                context_->coreStatusMgr.WaitTaskCnt(static_cast<int>(CoreType::AIV)));
            DEV_DEBUG("hand shake presend task cnt : aic=%u, aiv=%u",
                      context_->coreStatusMgr.WaitTaskCnt(static_cast<int>(CoreType::AIC)),
                      context_->coreStatusMgr.WaitTaskCnt(static_cast<int>(CoreType::AIV)));
        }
    }

    inline void DumpAicoreStatusWhenTimeout(bool* handFlag)
    {
        for (int i = aicStart_; i < aicEnd_; i++) {
            if (handFlag[i]) {
                DEV_INFO("Aic core[%d] hand shake success, phyid=%d.", i, aicoreHal_.GetPhyIdByBlockId(i));
            } else {
                DEV_ERROR(SchedErr::HANDSHAKE_TIMEOUT,
                          "#sche.handshake.timeout: Aic core[%d] hand shake timeout, status=%lu.", i,
                          aicoreHal_.GetAicoreStatus(i));
            }
        }

        for (int i = aivStart_; i < aivEnd_; i++) {
            if (handFlag[i]) {
                DEV_INFO("Aiv core[%d] hand shake success, phyid=%d.", i, aicoreHal_.GetPhyIdByBlockId(i));
            } else {
                DEV_ERROR(SchedErr::HANDSHAKE_TIMEOUT,
                          "#sche.handshake.timeout: Aiv core[%d] hand shake timeout, status=%lu.", i,
                          aicoreHal_.GetAicoreStatus(i));
            }
        }
    }

    inline void HandShakeByGmForAic(bool& aicAllSuccess, bool (&handFlag)[MAX_AICORE_NUM], int& handShakeNum,
                                    int& aicSucessCnt)
    {
        bool curIterAllAicSuccess = true;
        for (int i = aicEnd_ - 1; (!aicAllSuccess) && i >= aicStart_; i--) {
            if (handFlag[i]) {
                continue;
            }
            if (aicoreHal_.TryHandShakeByGm(i, dotStatus_)) {
                handShakeNum++;
                aicSucessCnt++;
                handFlag[i] = true;
                if (i < adjAicEnd_) {
                    context_->coreStatusMgr.AddRunAndPendCoreIdx(i, static_cast<int>(CoreType::AIC));
                }
                DEV_VERBOSE_DEBUG("#trace.handshake: tid=%d core=%d type=%d phyId=%d success=%d", schedIdx_, i,
                                  static_cast<int>(CoreType::AIC), aicoreHal_.GetPhyIdByBlockId(i), 1);
            } else {
                curIterAllAicSuccess = false;
            }
        }
        aicAllSuccess = curIterAllAicSuccess;
    }

    inline void HandShakeByGmForAiv(bool& aivAllSuccess, bool (&handFlag)[MAX_AICORE_NUM], int& handShakeNum,
                                    int& aivSucessCnt)
    {
        bool curIterAllAivSuccess = true;
        for (int i = aivEnd_ - 1; (!aivAllSuccess) && i >= aivStart_; i--) {
            if (handFlag[i]) {
                continue;
            }
            if (aicoreHal_.TryHandShakeByGm(i, dotStatus_)) {
                handShakeNum++;
                aivSucessCnt++;
                handFlag[i] = true;
                if (i < adjAivEnd_) {
                    context_->coreStatusMgr.AddRunAndPendCoreIdx(i, static_cast<int>(CoreType::AIV));
                }
                DEV_VERBOSE_DEBUG("#trace.handshake: tid=%d core=%d type=%d phyId=%d success=%d", schedIdx_, i,
                                  static_cast<int>(CoreType::AIV), aicoreHal_.GetPhyIdByBlockId(i), 1);
            } else {
                curIterAllAivSuccess = false;
            }
        }
        aivAllSuccess = curIterAllAivSuccess;
    }

    inline void HandShakeCorrectReadyCore(CoreType coreType)
    {
        uint8_t adjAicoreEnd = coreType == CoreType::AIV ? adjAivEnd_ : adjAicEnd_;
        uint8_t preAdjAicoreEnd = coreType == CoreType::AIV ? aivEnd_ : aicEnd_;
        for (int i = preAdjAicoreEnd - 1; i >= adjAicoreEnd; i--) {
            if (context_->coreStatusMgr.GetCoreIdxPosition(i) != INVALID_COREIDX_POSITION) {
                context_->coreStatusMgr.RemovePendReadyCoreIdx(static_cast<int>(coreType));
            }
            context_->coreStatusMgr.RemoveRunReadyCoreIdx(i, static_cast<int>(coreType));
        }
    }

    inline int HandShakeByGmWithPreSendTask(DevStartArgs* devStartArgs)
    {
        int handShakeNum = 0;
        int mngAicoreNum = aicEnd_ - aicStart_ + aivEnd_ - aivStart_;
        bool handFlag[MAX_AICORE_NUM] = {false};
        TIMEOUT_CHECK_INIT(archInfo_, TIMEOUT_HAND_SHAKE);
        bool needSendAic = false;
        bool needSendAiv = false;
        bool aicAllSuccess = false;
        bool aivAllSuccess = false;
        bool needSetSync = true;
        int aicSucessCnt = 0;
        int aivSucessCnt = 0;
        int aicTreshold = 4;
        int aivThreshold = 4;
        SchDeviceTaskContext* deviceCtx = nullptr;
        while (handShakeNum < mngAicoreNum) {
            if (deviceCtx == nullptr) {
                deviceCtx = HandShakeTryPreFetchDevTask(needSendAic, needSendAiv);
                if (!disableControlCore_) {
                    CalcAdjAicoreEnd(deviceCtx, false);
                }
            }

            HandShakeByGmForAic(aicAllSuccess, handFlag, handShakeNum, aicSucessCnt);
            if (unlikely(needSetSync && (handShakeNum > 0))) {
                devStartArgs->syncFlag = 1;
                needSetSync = false;
            }

            if (needSendAic && aicSucessCnt >= aicTreshold) {
                __sync_synchronize();
                HandShakeCorrectReadyCore(CoreType::AIC);
                TryBatchSendTask(deviceCtx, CoreType::AIC, deviceCtx->readyAicCoreFunctionQue, aicStart_, adjAicEnd_);
                aicSucessCnt = 0;
            }

            HandShakeByGmForAiv(aivAllSuccess, handFlag, handShakeNum, aivSucessCnt);
            if (needSendAiv && aivSucessCnt >= aivThreshold) {
                __sync_synchronize();
                HandShakeCorrectReadyCore(CoreType::AIV);
                TryBatchSendTask(deviceCtx, CoreType::AIV, deviceCtx->readyAivCoreFunctionQue, aivStart_, adjAivEnd_);
                aivSucessCnt = 0;
            }

            __PYPTO_TIMEOUT_CHECK(
                SchedErr::HANDSHAKE_TIMEOUT,
                {
                    DumpAicoreStatusWhenTimeout(handFlag);
                    return DEVICE_MACHINE_ERROR;
                },
                "#sche.handshake: HandShakeByGmWithPreSendTask, notHandshakeNum=%d.", mngAicoreNum - handShakeNum);
        }
        HandShakeCorrectReadyCore(CoreType::AIV);
        HandShakeCorrectReadyCore(CoreType::AIC);
        HandShakePostProc(deviceCtx, needSendAic, needSendAiv);
        return DEVICE_MACHINE_OK;
    }

    inline int HandShake(DevStartArgs* devStartArgs)
    {
        DEV_INFO("aicpu[%d] handshake start.", aicpuIdx_);
        int rc = HandShakeByGmWithPreSendTask(devStartArgs);
        if (rc != DEVICE_MACHINE_OK) {
            DEV_ERROR(SchedErr::HANDSHAKE_TIMEOUT, "#sche.handshake.presend: Aicpu[%d] handshake failed.", aicpuIdx_);
            return rc;
        }
        DEV_INFO("Aicpu[%d] handshake success.", aicpuIdx_);
        return 0;
    }

    /* assign aic and aiv core index section for this aicpu */
    inline void UpdateAiCoreBlockIndexSection()
    {
        auto f = [](int total, int idx, int part, int count, int& start, int& end) {
            int perCpu = (total / part) * count;
            int remain = total % part;
            start = idx * perCpu + ((idx < remain) ? idx * count : remain * count);
            end = start + perCpu + ((idx < remain) ? count : 0);
        };

        f(aicValidNum_, schedIdx_, aicpuNum_, 1, aicStart_, aicEnd_);
        if (archInfo_ == ArchInfo::DAV_3510) {
            f(aicValidNum_, schedIdx_, aicpuNum_, AIV_NUM_PER_AI_CORE, aivStart_, aivEnd_);
        } else {
            f(AIV_NUM_PER_AI_CORE * aicValidNum_, schedIdx_, aicpuNum_, 1, aivStart_, aivEnd_);
        }

        aivStart_ += aicValidNum_;
        aivEnd_ += aicValidNum_;
        adjAicEnd_ = aicEnd_;
        adjAivEnd_ = aivEnd_;

        DEV_IF_NONDEVICE
        {
            if (!aicoreHal_.IsHostSimMode()) {
                context_->coreStatusMgr.SetCorePendReadyCnt(static_cast<int>(CoreType::AIC), aicEnd_ - aicStart_);
                context_->coreStatusMgr.SetCorePendReadyCnt(static_cast<int>(CoreType::AIV), aivEnd_ - aivStart_);
                ForEachManageAicoreReverse([this](int coreIdx) {
                    int coreType = static_cast<int>(AicoreType(coreIdx));
                    context_->coreStatusMgr.AddRunReadyCoreIdx(coreIdx, coreType);
                });
            }
        }

        context_->coreStatusMgr.SetLastPendReadyCoreIdx(static_cast<int>(CoreType::AIV),
                                                        static_cast<uint32_t>(aivStart_));
        context_->coreStatusMgr.SetLastPendReadyCoreIdx(static_cast<int>(CoreType::AIC),
                                                        static_cast<uint32_t>(aicStart_));
        aicoreHal_.SetMngCoreBlockId(aicStart_, aicEnd_, aivStart_, aivEnd_);
        DEV_VERBOSE_DEBUG("#trace.arbitration: tid=%d threadIdx=%d schedIdx=%d arbitratedScheNum=%d "
                          "aicStart=%d aicEnd=%d aivStart=%d aivEnd=%d",
                          schedIdx_, aicpuIdx_, schedIdx_, aicpuNum_, aicStart_, aicEnd_, aivStart_, aivEnd_);
        DEV_DEBUG("assign core aic coreindex section: start=%d, end=%d.", aicStart_, aicEnd_);
        DEV_DEBUG("assign core aiv coreindex section: start=%d, end=%d.", aivStart_, aivEnd_);
    }

    inline int GetPhyIdByBlockId(int coreIdx) { return aicoreHal_.GetPhyIdByBlockId(coreIdx); }

    inline void ForEachManageAicore(std::function<void(int coreIdx)> func) const
    {
        for (int i = aicStart_; i < aicEnd_; ++i) {
            func(i);
        }
        for (int i = aivStart_; i < aivEnd_; ++i) {
            func(i);
        }
    }

    inline void ForEachManageAicoreReverse(std::function<void(int coreIdx)> func) const
    {
        for (int i = aicEnd_ - 1; i >= aicStart_; --i) {
            func(i);
        }
        for (int i = aivEnd_ - 1; i >= aivStart_; --i) {
            func(i);
        }
    }

    inline int ForEachManageAicoreWithRet(std::function<int(int coreIdx)> func) const
    {
        int ret = DEVICE_MACHINE_OK;
        for (int i = aicStart_; i < aicEnd_; ++i) {
            ret = func(i);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                DEV_ERROR(SchedErr::CORE_TASK_PROCESS_FAILED, "#sche.check.aic.process: proc aicore aic[%d] failed.",
                          i);
                return ret;
            }
        }
        for (int i = aivStart_; i < aivEnd_; ++i) {
            ret = func(i);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                DEV_ERROR(SchedErr::CORE_TASK_PROCESS_FAILED, "#sche.check.aiv.process: proc aicore aiv[%d] failed.",
                          i);
                return ret;
            }
        }
        return ret;
    }

    inline bool NeedsHwStopOnLastDevTask() const
    {
        DEV_IF_DEVICE { return true; }
        return aicoreHal_.IsHostSimMode();
    }

    void BatchStopAllManagedCores()
    {
        aicoreHal_.SetReadyQueue(aicStart_, aicEnd_, AICORE_TASK_STOP + 1);
        aicoreHal_.SetReadyQueue(aivStart_, aivEnd_, AICORE_TASK_STOP + 1);
        /* write to MAINBASE reg must be done before close 0x18;
         * also make STOP visible before clearing parallelDevTask. */
        __sync_synchronize();
        if (aicoreHal_.NeedsFastPathRegClose()) {
            aicoreHal_.CloseFastPathReg(aicStart_, aicEnd_);
            aicoreHal_.CloseFastPathReg(aivStart_, aivEnd_);
        }
        // Destask + HELLO zeros after STOP is visible, fence, then GOODBYE.
        // HELLO must be globally cleared before GOODBYE; a post-GOODBYE
        // shakeBuffer[0]=0 can wipe the next launch's HELLO.
        aicoreHal_.ResetCoreStopSlot(aicStart_, aicEnd_);
        aicoreHal_.ResetCoreStopSlot(aivStart_, aivEnd_);
        __sync_synchronize();
        aicoreHal_.SendWaveGoodbye(aicStart_, aicEnd_);
        aicoreHal_.SendWaveGoodbye(aivStart_, aivEnd_);
    }

    inline void AbnormalStop()
    {
        ResetRegAll();
        CheckAndResetReg();
        DEV_INFO("aicore manager[%d] abnormally stopped.", aicpuIdx_);
    }

    inline void NormalStop()
    {
        DEV_INFO("aicore manager[%d] try normal stop.", aicpuIdx_);
        BatchStopAllManagedCores();
        DEV_INFO("aicore manager[%d] normally stopped.", aicpuIdx_);
    }

    inline int GetAllAiCoreNum() { return aicNum_ + aivNum_; }
    inline void SetDotStatus(int64_t status) { dotStatus_ = status; }
    inline CoreType AicoreType(int coreIdx) const { return coreIdx < aicEnd_ ? CoreType::AIC : CoreType::AIV; }

    // DFX
    inline void DfxProcAfterFinishTask(SchDeviceTaskContext* deviceTaskCtx, int coreIdx, uint64_t taskId)
    {
        DEV_TRACE_DEBUG(LEvent(LUid(deviceTaskCtx->TaskId(), FuncID(taskId), GetRootIndex(deviceTaskCtx, taskId),
                                    TaskID(taskId), GetLeafIndex(deviceTaskCtx, taskId)),
                               LActFinish(coreIdx)));
        if constexpr (!IsDeviceMode())
            return;

        aicoreProf_.ProfGetPmu(coreIdx, 0, static_cast<uint32_t>(taskId & TASKID_FROM_CTRL_TOPO_MASK),
                               deviceTaskCtx->TaskId());

#if ENABLE_TENSOR_DUMP
        DEV_IF_DEVICE
        {
            // dump output tensor
            if (unlikely(isEnableDump)) {
                aicoreDump_.DoDump(deviceTaskCtx->GetDeviceTask(), "output", taskId, coreIdx);
            }
        }
#endif

        DEV_IF_VERBOSE_DEBUG { recvFinTask_[coreIdx].push_back(TaskInfo(coreIdx, taskId, deviceTaskCtx->TaskId())); }
        DEV_VERBOSE_DEBUG("#trace.ltask.finish: tid=%d task=%lu coreIdx=%d dtaskId=%lu", schedIdx_, taskId, coreIdx,
                          deviceTaskCtx->TaskId());
    }

    inline bool IsNeedProcAicpuTask() { return hasAicpuTask_; }

private:
    void ReuseUpdateDeviceCtx(SchDeviceTaskContext* devTaskCtx, DeviceTaskCtrl* newDevTask)
    {
        devTaskCtx->BindTaskCtrl(newDevTask);

        // update version mark parallel context modified
        context_->UpdateParallelVersion();
        devTaskCtx->BindParallelCtxVersion(context_->PrallelVersion());
    }

    void FillParallelDevtaskCtx()
    {
        auto& parallelDevTaskCtx = context_->schParallelDevTaskCtx;
        while (!parallelDevTaskCtx.Full()) {
            DeviceTaskCtrl* taskCtrl = nullptr;
            if (!taskQueue_->TempDequeue(taskCtrl)) {
                break; // no device task
            }

            if (taskCtrl == nullptr) {
                taskQueue_->PopFront();
                taskCtrlDequeFinish = true;
                break;
            }

            if (parallelDevTaskCtx.Empty()) {
                PerfMtTrace(PERF_TRACE_DEV_TASK_RCV, aicpuIdx_);
                context_->EnqueueParallelCtx(taskCtrl); // if empty,  equeue directly
                taskQueue_->PopFront();
                continue;
            }

            if (!context_->CurSupportParallel()) {
                DEV_DEBUG("Cur ctx cannot support parallel, fill stop.");
                break; // non-parallel context just support one devtask schedule
            }

            if (!taskCtrl->SupportParallel()) {
                DEV_DEBUG("Device task(%lu) cannot support parallel, fill stop.", taskCtrl->taskId);
                break; // non-parallel devtask cannot scheduled with prallel dev task
            }

            if (!context_->CanParallelWith(taskCtrl)) {
                DEV_DEBUG("Cur ctx cannot parallel with device task, %lu, forid=%u", taskCtrl->taskId,
                          taskCtrl->ParallelForId());
                break; // different parallel-forid devtask cannot scheduled together
            }

            taskCtrlDequeFinish = reinterpret_cast<DynDeviceTask*>(taskCtrl->devTask)->IsLastTask();
            context_->EnqueueParallelCtx(taskCtrl);
            taskQueue_->PopFront();
        }

        if (!aicoreDevTaskInited && !parallelDevTaskCtx.Empty()) {
            DEV_INFO("Begin init aicore parallel devtask data.");
            InitAicoreParallelDevTask(&parallelDevTaskCtx);
            aicoreDevTaskInited = true; // just need init one time
        }
        return;
    }

    inline void UpdateRunReadyCoreNum(int preAdjAicEnd, int preAdjAivEnd)
    {
        if (preAdjAicEnd > adjAicEnd_) {
            for (int i = preAdjAicEnd - 1; i >= adjAicEnd_; i--) {
                context_->coreStatusMgr.RemoveRunReadyCoreIdx(i, static_cast<int>(CoreType::AIC));
            }
        } else if (preAdjAicEnd < adjAicEnd_) {
            for (int i = preAdjAicEnd; i < adjAicEnd_; i++) {
                context_->coreStatusMgr.AddRunReadyCoreIdx(i, static_cast<int>(CoreType::AIC));
            }
        }

        if (preAdjAivEnd > adjAivEnd_) {
            for (int i = preAdjAivEnd - 1; i >= adjAivEnd_; i--) {
                context_->coreStatusMgr.RemoveRunReadyCoreIdx(i, static_cast<int>(CoreType::AIV));
            }
        } else if (preAdjAivEnd < adjAivEnd_) {
            for (int i = preAdjAivEnd; i < adjAivEnd_; i++) {
                context_->coreStatusMgr.AddRunReadyCoreIdx(i, static_cast<int>(CoreType::AIV));
            }
        }
    }

    static void UpdateAicoreEnd(int total, int idx, int part, int count, int start, int& end)
    {
        if (part == 0) {
            end = start;
            return;
        }

        int perCpu = (total / part) * count;
        int remain = total % part;
        end = start + perCpu + ((idx < remain) ? count : 0);
    }

    // 4-way remainder fills low schedIdx first. On g20 (aicpuNum=4):
    // maxC==1 → only tid0; maxC==2 → only tid0/1; die1 gets 0 cores. maxC>=3 is fine.
    static void ScanBoundLoopDieIds(DynDeviceTask* dynTask, bool& hasDie0, bool& hasDie1)
    {
        hasDie0 = false;
        hasDie1 = false;
        if (dynTask == nullptr) {
            return;
        }
        const size_t cacheListSize = dynTask->dynFuncDataCacheListSize;
        const auto* cacheList = dynTask->dynFuncDataCacheList;
        constexpr int8_t kDie0 = static_cast<int8_t>(DieId::DIE_0);
        constexpr int8_t kDie1 = static_cast<int8_t>(DieId::DIE_1);
        for (size_t i = 0; i < cacheListSize; ++i) {
            const auto* duppedData = cacheList[i].duppedData;
            if (duppedData == nullptr) {
                continue;
            }
            if (duppedData->loopDieId_ == kDie0) {
                hasDie0 = true;
            } else if (duppedData->loopDieId_ == kDie1) {
                hasDie1 = true;
            }
            if (hasDie0 && hasDie1) {
                return;
            }
        }
    }

    bool PrepareDieLocalSched(SchDeviceTaskContext* devTaskCtx, int& schedPart, int& localSchedIdx)
    {
        auto& wrapManager = devTaskCtx->GetWrapManager();
        if (!wrapManager.IsMixArch()) {
            return false;
        }
        DieId dieId = wrapManager.GetDieId();
        if (dieId == DieId::DIE_MIX || dieId == DieId::DIE_UNKNOWN) {
            return false;
        }
        int dieSchedStart = 0;
        int dieSchedEnd = 0;
        wrapManager.GetDieSchedIdRange(dieSchedStart, dieSchedEnd, aicpuNum_);
        schedPart = dieSchedEnd - dieSchedStart;
        localSchedIdx = schedIdx_ - dieSchedStart;
        return schedPart > 0;
    }

    // Only for mix-arch DevTask with maxC==1 or 2. Caller falls back to 4-way when this returns false.
    // Die queues are exclusive: a thread only sees its own die queue plus the global queue.
    bool TryApplySmallMaxCDieFallback(SchDeviceTaskContext* devTaskCtx, int maxC)
    {
        if (maxC <= 0 || maxC > 2) {
            return false;
        }
        int schedPart = 0;
        int localSchedIdx = 0;
        if (!PrepareDieLocalSched(devTaskCtx, schedPart, localSchedIdx)) {
            return false;
        }

        auto* dynTask = reinterpret_cast<DynDeviceTask*>(devTaskCtx->GetDeviceTask());
        bool hasDie0 = false;
        bool hasDie1 = false;
        ScanBoundLoopDieIds(dynTask, hasDie0, hasDie1);
        // maxC==2: 4-way puts both cores on die0; 1 per die keeps total=2 and unblocks die1.
        // maxC==1 with both die queues: one core cannot see both queues; 1+1 uses one extra core.
        if (maxC == 2 || (hasDie0 && hasDie1)) {
            ApplyCoreBudgetToAdjRange(1, 1, schedPart, localSchedIdx);
            return true;
        }
        // No die1-queued work: tid0 already covers die0 and the global queue. Keep 4-way.
        if (!hasDie1) {
            return false;
        }
        // maxC==1 and only die1 has a die queue: move the single core to die1.
        // Die1 threads take it; die0 threads get adj==start so they do not occupy that core.
        if (devTaskCtx->GetWrapManager().GetDieId() == DieId::DIE_1) {
            ApplyCoreBudgetToAdjRange(1, 1, schedPart, localSchedIdx);
        } else {
            adjAicEnd_ = aicStart_;
            adjAivEnd_ = aivStart_;
        }
        return true;
    }

    void ApplyCoreBudgetToAdjRange(int aicBudget, int aivBudget, int schedPart, int localSchedIdx)
    {
        UpdateAicoreEnd(aicBudget, localSchedIdx, schedPart, 1, aicStart_, adjAicEnd_);
        if (archInfo_ == ArchInfo::DAV_3510) {
            UpdateAicoreEnd(aicBudget, localSchedIdx, schedPart, AIV_NUM_PER_AI_CORE, aivStart_, adjAivEnd_);
        } else {
            UpdateAicoreEnd(aivBudget, localSchedIdx, schedPart, 1, aivStart_, adjAivEnd_);
        }
        adjAicEnd_ = std::min(adjAicEnd_, aicEnd_);
        adjAivEnd_ = std::min(adjAivEnd_, aivEnd_);
    }

    void ApplyGlobalCoreBudgetToAdjRange(int maxC, int maxV)
    {
        UpdateAicoreEnd(maxC, schedIdx_, aicpuNum_, 1, aicStart_, adjAicEnd_);
        if (archInfo_ == ArchInfo::DAV_3510) {
            UpdateAicoreEnd(maxC, schedIdx_, aicpuNum_, AIV_NUM_PER_AI_CORE, aivStart_, adjAivEnd_);
        } else {
            UpdateAicoreEnd(maxV, schedIdx_, aicpuNum_, 1, aivStart_, adjAivEnd_);
        }
    }

    inline void CalcAdjAicoreEnd(SchDeviceTaskContext* devTaskCtx, bool isNeedUpdateCoreNum = true)
    {
        if constexpr (!IsDeviceMode()) {
            if (!enableEslModel_)
                return;
        }

        int preAdjAicEnd = adjAicEnd_;
        int preAdjAivEnd = adjAivEnd_;
        if (devTaskCtx == nullptr || devTaskCtx->GetDeviceTaskCtrl() == nullptr) {
            adjAicEnd_ = aicEnd_;
            adjAivEnd_ = aivEnd_;
        } else {
            auto taskCtrl = devTaskCtx->GetDeviceTaskCtrl();
            int maxC = static_cast<int>(taskCtrl->GetMaxC());
            int maxV = static_cast<int>(taskCtrl->GetMaxV());
            int maxAivNum = aicValidNum_ * static_cast<int>(AIV_NUM_PER_AI_CORE);
            if ((maxC >= aicValidNum_ && maxV >= maxAivNum) || maxC == 0 || maxV == 0) {
                adjAicEnd_ = aicEnd_;
                adjAivEnd_ = aivEnd_;
            } else {
                maxC = maxC >= aicValidNum_ ? aicValidNum_ : maxC;
                maxV = maxV >= maxAivNum ? maxAivNum : maxV;
                if (!TryApplySmallMaxCDieFallback(devTaskCtx, maxC)) {
                    ApplyGlobalCoreBudgetToAdjRange(maxC, maxV);
                }
            }
        }

        if (!isNeedUpdateCoreNum || (preAdjAicEnd == adjAicEnd_ && preAdjAivEnd == adjAivEnd_)) {
            return;
        }

        context_->coreStatusMgr.SetCorePendReadyCnt(static_cast<int>(CoreType::AIC), adjAicEnd_ - aicStart_);
        context_->coreStatusMgr.SetCorePendReadyCnt(static_cast<int>(CoreType::AIV), adjAivEnd_ - aivStart_);
        UpdateRunReadyCoreNum(preAdjAicEnd, preAdjAivEnd);
    }

    inline void ProcessParallellDevTasksFinish(SchDeviceTaskContext* devTaskCtx)
    {
        // continue bind the next parallel devtaskctrl wich have the same forid & iterid to this sch context
        DeviceTaskCtrl* curTaskCtrl = devTaskCtx->GetDeviceTaskCtrl();
        if (curTaskCtrl->ExistNextSameIterTask()) {
            DeviceTaskCtrl* nextTaskCtrl = curTaskCtrl->NextSameIterTaskCtrl();
            if (nextTaskCtrl != nullptr) {
                // reuse this task ctrl and task context for next same iterid device task
                ReuseUpdateDeviceCtx(devTaskCtx, nextTaskCtrl);
                curTaskCtrl->Free(aicpuNum_); // parallel device taskctrl need free manually
                DEV_DEBUG("Sch dev ctx bind next same parallel iter device task(%lu), forid %u, iterid %u",
                          nextTaskCtrl->taskId, nextTaskCtrl->ParallelForId(), nextTaskCtrl->ParallelIterId());
            } else {
                DEV_VERBOSE_DEBUG("Wait ctrl build same parallel iter device task, forid %u, iterid %u.",
                                  curTaskCtrl->ParallelForId(), curTaskCtrl->ParallelIterId());
            }
        } else {
            // parallel devicetaskctrl and device context need set free manually, wait recycle
            devTaskCtx->Free(aicpuNum_);
        }
    }

    inline int32_t ProcessParallelDevTasks()
    {
        int32_t ret = DEVICE_MACHINE_OK;
        auto& parallelDevTaskCtx = context_->schParallelDevTaskCtx;
        for (uint32_t i = parallelDevTaskCtx.front; i < parallelDevTaskCtx.rear; ++i) {
            SchDeviceTaskContext* devTaskCtx = parallelDevTaskCtx.Element(i);
            if (devTaskCtx->IsFree()) {
                DEV_VERBOSE_DEBUG("Device task ctx(%u) wait recycle.", devTaskCtx->parallelIdx);
                continue; // maybe have some non-consecutiv free ctx wait recycle
            }
            context_->SetCurSchDevTaskCtx(devTaskCtx);
            PerfMtBegin(PERF_EVT_RUN_TASK, aicpuIdx_);
            if (!disableControlCore_) {
                CalcAdjAicoreEnd(devTaskCtx);
            }
            ret = RunTask(devTaskCtx);
            PerfMtEnd(PERF_EVT_RUN_TASK, aicpuIdx_);
            if (ret != DEVICE_MACHINE_OK)
                break;

            if (devTaskCtx->IsRunFinish()) {
                DEV_VERBOSE_DEBUG("#trace.dtask.end: tid=%d taskId=%lu finishedFunctionCnt=%lu coreFunctionCnt=%lu",
                                  schedIdx_, devTaskCtx->TaskId(),
                                  devTaskCtx->GetDeviceTaskCtrl()->finishedFunctionCnt.load(),
                                  devTaskCtx->GetDeviceTask()->coreFunctionCnt);
                if (unlikely(devTaskCtx->IsParallel())) {
                    ProcessParallellDevTasksFinish(devTaskCtx);
                } else {
                    parallelDevTaskCtx.PopFront(); // non-parallel tasks can only exist one at a time.
                }
            }
        }

        parallelDevTaskCtx.RecycleFreeContexts();
        return ret;
    }

private:
    AicoreHAL aicoreHal_;
    EslModelReplayManager eslModelReplayMgr_;
    bool aicoreDevTaskInited{false};
    bool firstLock[AICORE_TYPE_NUM]{true, true};
    int aicNum_{0};
    int aivNum_{0};
    int aicValidNum_{0}; // 有效的aic，根据pgmask计算host传过来
    int aicpuIdx_{0};
    int schedIdx_{0};
    int aicpuNum_{MAX_SCHEDULE_AICPU_NUM};
    int aicStart_{0};
    int aicEnd_{0};
    int adjAicEnd_{0};
    int aivStart_{0};
    int aivEnd_{0};
    int adjAivEnd_{0};
    uint64_t procAicCoreFunctionCnt_{0};
    uint64_t procAivCoreFunctionCnt_{0};
    uint64_t procAicpuFunctionCnt_{0};
    bool enableL2CacheSch_{false};
    bool enableFairSch_{false};
    bool validGetPgMask_{true};

    std::array<uint32_t, MAX_AICORE_NUM> runningIds_;
    std::array<uint32_t, MAX_AICORE_NUM> pendingIds_;
    std::array<int, MAX_AICORE_NUM> runningResolveIndexList_;
    std::array<int, MAX_AICORE_NUM> pendingResolveIndexList_;

    SchduleContext* context_{nullptr};
    ArchInfo archInfo_{ArchInfo::DAV_2201};
    SchThreadStatus& threadStatus;

    bool taskCtrlDequeFinish{false};
    SPSCQueue<DeviceTaskCtrl*, DEFAULT_QUEUE_SIZE>* taskQueue_{nullptr};
    AicpuTaskManager aicpuTaskManager_;

    AiCoreProf aicoreProf_;
    AicoreDump aicoreDump_;
    bool isEnableDump{false};
    int64_t dotStatus_{0};
    bool isSendStop{false};
    bool isOpenPerf_{false};
    std::array<uint8_t, MAX_AICORE_NUM> pingPongFlag_;

    std::vector<TaskInfo> sendTask_[MAX_AICORE_NUM];
    std::vector<TaskInfo> recvFinTask_[MAX_AICORE_NUM];
    std::vector<TaskInfo> recvAckTask_[MAX_AICORE_NUM];

    friend class AiCoreProf;

    bool enableEslModel_;
    bool disableControlCore_{false};
    bool hasAicpuTask_{false};

    using ReleaseCoreByRegValFn = int32_t (AiCoreManager::*)(CoreType type, int coreIdx, ResolveTaskContext* ctx,
                                                             uint32_t& finishCnt, uint32_t& resloveParallelIdx,
                                                             uint64_t finTaskRegVal, uint32_t aicpuCallCode,
                                                             uint32_t finTaskId, uint32_t finTaskState);
    ReleaseCoreByRegValFn releaseCoreByRegValFn_{nullptr};
};
} // namespace npu::tile_fwk::dynamic
