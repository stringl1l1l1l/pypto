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
 * \file aicore_hal.h
 * \brief
 */

#pragma once

#include "tilefwk/aicpu_common.h"
#include "machine/device/dynamic/aicore_constants.h"
#include "machine/device/dynamic/aicore_runtime_model.h"
#include "machine/device/dynamic/aicore_prof.h"
#include "machine/device/dynamic/costmodel_utils.h"
#include "machine/device/dynamic/eslmodel_aicore_hal.h"
#include "machine/simulation/aicore_hardware.h"

namespace npu::tile_fwk::dynamic {

const int32_t CORE_QUEUE_MODE_NUM_8 = 8;
const int32_t CORE_QUEUE_MODE_NUM_7 = 7;
const int32_t CORE_QUEUE_MODE_NUM_6 = 6;
const int32_t CORE_QUEUE_MODE_NUM_5 = 5;
const int32_t CORE_QUEUE_MODE_NUM_4 = 4;
const int32_t CORE_QUEUE_MODE_NUM_3 = 3;
const int32_t CORE_QUEUE_MODE_NUM_2 = 2;
const int32_t CORE_QUEUE_MODE_NUM_1 = 1;

const uint32_t REG_SPR_MAGIC = 0x78;

namespace DAV_2201 {
const uint32_t REG_SPR_DATA_MAIN_BASE = 0xA0;
const uint32_t REG_SPR_COND = 0x4C8;
} // namespace DAV_2201

namespace DAV_3510 {
const uint32_t REG_SPR_DATA_MAIN_BASE = 0xD0;
const uint32_t REG_SPR_COND = 0x5108;
} // namespace DAV_3510

class AicoreHAL {
public:
    inline void Init(DeviceArgs* deviceArgs, AiCoreProf* aicoreProf)
    {
        aicoreProf_ = aicoreProf;
        sharedBuffer_ = deviceArgs->sharedBuffer;
        regAddrs_ = reinterpret_cast<int64_t*>(deviceArgs->coreRegAddr);
        regNum_ = deviceArgs->nrAic + deviceArgs->nrAiv;
        freq_ = GetFreq() / (NSEC_PER_SEC / NSEC_PER_USEC);
        archInfo_ = deviceArgs->archInfo;
        readyRegQueues_.fill(nullptr);
        finishRegQueues_.fill(nullptr);
        blockIdToPhyCoreId_.fill(-1);
        args_.fill(nullptr);
        if (deviceArgs->archInfo == ArchInfo::DAV_3510) {
            regSprDataMainBase_ = DAV_3510::REG_SPR_DATA_MAIN_BASE;
            regSprCond_ = DAV_3510::REG_SPR_COND;
            isNeedWriteRegForFastPath_ = false;
        }
        enableEslModel_ = deviceArgs->enableEslModel;
        DEV_IF_NONDEVICE
        {
            if (enableEslModel_) {
                eslModel_.Init();
            }
            costModelAdapter_.SetSendTask([this](int coreIdx, uint64_t taskId) { CostModelSendTask(coreIdx, taskId); });
            costModelAdapter_.SetGetTask([this](int coreIdx) { return CostModelGetTask(coreIdx); });
            costModelAdapter_.SetActualModel(costModel_);
        }
    }

    inline uint32_t GetRegSprDataMainBase() { return regSprDataMainBase_; }

    inline void SetMngCoreBlockId(int aicStart, int aicEnd, int aivStart, int aivEnd)
    {
        aicStart_ = aicStart;
        aicEnd_ = aicEnd;
        aivStart_ = aivStart;
        aivEnd_ = aivEnd;
    }

    inline void SetModel(uint64_t costModel)
    {
        costModel_ = reinterpret_cast<CostModel::AiCoreModel*>(costModel);
        costModelAdapter_.SetActualModel(costModel_);
    }

    int64_t* GetRegAddrs() const { return regAddrs_; }
    uint32_t GetregNum() const { return regNum_; }
    inline int GetHostSimPhyId(int coreIdx)
    {
        int phyId = GetPhyIdByBlockId(coreIdx);
        return (phyId >= 0) ? phyId : coreIdx;
    }

    inline bool IsHostSimMode() const
    {
        if constexpr (IsDeviceMode()) {
            return false;
        } else {
            return enableEslModel_ || AicoreHardware::Global().CoreNum() > 0;
        }
    }

    inline uint32_t ReadReg32(int coreIdx, int offset)
    {
        auto idx = GetPhyIdByBlockId(coreIdx);
        if (idx != -1) {
            return *(reinterpret_cast<volatile uint32_t*>(regAddrs_[idx] + offset));
        }
        return 0;
    }

    inline uint32_t ReadPathReg(int coreIdx)
    {
        if (!isNeedWriteRegForFastPath_) {
            return 0;
        }

        return ReadReg32(coreIdx, REG_SPR_FAST_PATH_ENABLE);
    }

    inline bool NeedsFastPathRegClose() const { return isNeedWriteRegForFastPath_; }

    inline void CloseFastPathReg(int coreStart, int coreEnd)
    {
        WriteReg32(coreStart, coreEnd, REG_SPR_FAST_PATH_ENABLE, static_cast<uint32_t>(REG_SPR_FAST_PATH_CLOSE));
    }

    inline void WriteReg32(int coreIdx, int offset, uint32_t val)
    {
        auto idx = GetPhyIdByBlockId(coreIdx);
        if (idx != -1) {
            *(reinterpret_cast<volatile uint32_t*>(regAddrs_[idx] + offset)) = val;
        }
        return;
    }

    inline void WriteReg32(int coreStart, int coreEnd, int offset, uint32_t val);

    inline void WriteReg32All(int aicNum, int aivNum, int offset, uint32_t val)
    {
        for (int i = 0; i < aicNum + aivNum; ++i) {
            if (regAddrs_[i] != 0) {
                *(reinterpret_cast<volatile uint32_t*>(regAddrs_[i] + offset)) = val;
            }
        }
    }

    inline bool IsSpecialTask(uint32_t taskId)
    {
        return taskId == AICORE_TASK_INIT || taskId == AICORE_TASK_STOP || (taskId & 0xFFFFFFFF) == AICORE_FUNC_STOP;
    }

    inline void SetReadyQueue(int coreIdx, uint64_t value)
    {
        if constexpr (IsDeviceMode()) {
            *readyRegQueues_[GetPhyIdByBlockId(coreIdx)] = value;
        } else {
            GetActiveModel()->SetReadyQueue(coreIdx, GetHostSimPhyId(coreIdx), value);
        }
    }

    inline void SetReadyQueue(int coreStart, int coreEnd, uint32_t val);

    inline void GetFinishQueue(const uint32_t* coreIdx, uint32_t* vals, int n);

    inline uint64_t GetFinishedTask(int coreIdx)
    {
        if constexpr (IsDeviceMode()) {
            return *(finishRegQueues_[GetPhyIdByBlockId(coreIdx)]);
        } else {
            return GetActiveModel()->GetFinishedTask(coreIdx, GetHostSimPhyId(coreIdx));
        }
    }

    void SetTaskTimeCost(std::function<uint64_t(uint64_t, uint64_t, uint64_t)> func) { getTaskTimeCost = func; }

    void SetEslModelReplayManager(EslModelReplayManager* replayMgr) { eslModel_.SetEslModelReplayManager(replayMgr); }

    uint64_t CostModelGetTask(int coreIdx)
    {
        auto currentTime = GetCycles();
        DEV_DEBUG("CostModel AICore polling: aicoreIdx=%d, time(cycles)=%lu.", coreIdx, currentTime);
        if (taskIds[coreIdx].empty())
            return AICORE_FUNC_STOP | AICORE_FIN_MASK;
        uint64_t taskId = 0;
        while (!taskIds[coreIdx].empty() && currentTime >= taskTimes[coreIdx].front()) {
            taskId = taskIds[coreIdx].front();
            taskTimes[coreIdx].pop_front();
            taskIds[coreIdx].pop_front();
        }
        if (taskIds[coreIdx].empty()) {
            DEV_DEBUG("CostModel AICore finish task: aicoreIdx=%d, taskId=%#lx, currentTime(cycles)=%lu.", coreIdx,
                      taskId, currentTime);
            return taskId | AICORE_FIN_MASK;
        }
        DEV_DEBUG("CostModel AICore running task: aicoreIdx=%d, taskId=%#lx, currentTime(cycles)=%lu, "
                  "finishTime(cycles)=%lu.",
                  coreIdx, taskIds[coreIdx].front(), currentTime, taskTimes[coreIdx].front());
        return taskIds[coreIdx].front();
    }

    void CostModelSendTask(int coreIdx, uint64_t taskId)
    {
        uint64_t time = taskIds[coreIdx].empty() ? GetCycles() : taskTimes[coreIdx].back();
        uint64_t timeCost = getTaskTimeCost == nullptr ? 0 : getTaskTimeCost(coreIdx, taskId, time);
        taskTimes[coreIdx].push_back(time + timeCost);
        taskIds[coreIdx].push_back(taskId);
        if (costModel_) {
            costModel_->SendTask(coreIdx, taskId);
        }
        DEV_DEBUG("CostModel AICore add task: aicoreIdx=%d, taskId=%#lx, newQueueSize=%lu, finishTime=%lu.", coreIdx,
                  taskId, taskIds[coreIdx].size(), time + timeCost);
    }

    int64_t GetSharedBuffer() { return sharedBuffer_; }

    inline void MapRegistersForAllCores(int aicNum)
    {
        for (uint32_t idx = 0; idx < static_cast<u_int32_t>(aicNum * CORE_NUM_PER_AI_CORE); idx++) {
            void* addr = reinterpret_cast<void*>(regAddrs_[idx]);
            if (addr == nullptr) {
                continue;
            }
            DEV_VERBOSE_DEBUG("phy core %u Addr is %p.", idx, addr);
            volatile uint64_t* reqQueueReg = reinterpret_cast<volatile uint64_t*>(static_cast<uint8_t*>(addr) +
                                                                                  regSprDataMainBase_);
            readyRegQueues_[idx] = reqQueueReg;
            volatile uint64_t* finishQueueReg = reinterpret_cast<volatile uint64_t*>(static_cast<uint8_t*>(addr) +
                                                                                     regSprCond_);
            finishRegQueues_[idx] = finishQueueReg;
        }
    }

    inline int& GetPhyIdByBlockId(int coreIdx) { return blockIdToPhyCoreId_[coreIdx]; }

    Metrics* GetMetrics(int coreIdx)
    {
        volatile KernelArgs* arg = reinterpret_cast<KernelArgs*>(sharedBuffer_ + coreIdx * SHARED_BUFFER_SIZE);
        volatile Metrics* metric = reinterpret_cast<Metrics*>(arg->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);
        DEV_INFO("aicoreIdx=%d host alloc metric memory: %p.", coreIdx, metric);
        if (metric == nullptr) {
            DEV_ERROR(DevCommonErr::NULLPTR, "#sche.prof.aicore.getaddr: aicoreIdx=%d null metric.", coreIdx);
            return nullptr;
        }

        TIMEOUT_CHECK_INIT(archInfo_, TIMEOUT_DISPATCH);
        volatile int stopFlag = metric->isMetricStop;
        while (stopFlag != 1) {
            __PYPTO_TIMEOUT_CHECK_EXIT_ONLY(DevCommonErr::NULLPTR, return nullptr,
                                            "#sche.prof.aicore.wait_finish: wait metrics done.");
            stopFlag = metric->isMetricStop;
        }

        return reinterpret_cast<Metrics*>(arg->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);
    }

    int DumpTaskProf(int coreIdx, CoreType coreType)
    {
        Metrics* metric = GetMetrics(coreIdx);
        if (metric == nullptr) {
            return DEVICE_MACHINE_ERROR;
        }
        metric->coreType = static_cast<int16_t>(coreType);
        DEV_VERBOSE_DEBUG("Dump core %d coreType[%s] prof data , task cnt %ld, metric:%p.", coreIdx,
                          coreType == CoreType::AIC ? "AIC" : "AIV", metric->taskCount, metric);
        for (int i = 0; i < metric->taskCount; i++) {
            volatile TaskStat* stat = &metric->tasks[i];
            aicoreProf_->ProfGetLog(coreIdx, &(metric)->tasks[i]);
            DEV_VERBOSE_DEBUG("  Dump prof for task %d, execstart: %ld execend :%ld.", stat->taskId, stat->execStart,
                              stat->execEnd);
        }
        return 0;
    }

    int SetAicorePerfTrace(int coreIdx, CoreType coreType, int16_t aicpuIdx)
    {
        Metrics* metric = GetMetrics(coreIdx);
        if (metric == nullptr) {
            return DEVICE_MACHINE_ERROR;
        }
        metric->coreType = static_cast<int16_t>(coreType);
        DEV_VERBOSE_DEBUG("Aicore %d coreType[%s]", coreIdx, coreType == CoreType::AIC ? "AIC" : "AIV");
        metric->scheCpuIdx = aicpuIdx;
        return 0;
    }

    void DumpAicoreStatus(int coreIdx) const
    {
        volatile KernelArgs* arg = reinterpret_cast<KernelArgs*>(sharedBuffer_ + coreIdx * SHARED_BUFFER_SIZE);
        DEV_ERROR(SchedErr::ABNOMAL_LAST_WORD,
                  "!!***********************aicore %d last status **************************!!", coreIdx);
        DEV_ERROR(SchedErr::ABNOMAL_LAST_WORD, "hello status %ld.", arg->dfxBuffer[0]);
        DEV_ERROR(SchedErr::ABNOMAL_LAST_WORD, "last_taskId %ld task status [%ld, %ld, %ld, %ld].",
                  arg->dfxBuffer[NUM_ONE], arg->dfxBuffer[NUM_TWO], arg->dfxBuffer[NUM_THREE], arg->dfxBuffer[NUM_FOUR],
                  arg->dfxBuffer[NUM_FIVE]);
    }

    uint64_t GetAicoreStatus(int coreIdx) const
    {
        int aicoreStatusIndex = 2;
        volatile KernelArgs* arg = reinterpret_cast<KernelArgs*>(sharedBuffer_ + coreIdx * SHARED_BUFFER_SIZE);
        return arg->dfxBuffer[aicoreStatusIndex];
    }

    uint64_t GetAicoreStatusLastWord(int coreIdx) const
    {
        int aicoreStatusIndex = 3;
        volatile KernelArgs* arg = reinterpret_cast<KernelArgs*>(sharedBuffer_ + coreIdx * SHARED_BUFFER_SIZE);
        return arg->dfxBuffer[aicoreStatusIndex];
    }

    uint64_t GetAicoreWarningStatus(int coreIdx) const
    {
        int aicoreWarningIndex = 6;
        volatile KernelArgs* arg = reinterpret_cast<KernelArgs*>(sharedBuffer_ + coreIdx * SHARED_BUFFER_SIZE);
        return arg->dfxBuffer[aicoreWarningIndex];
    }

    bool TryHandShakeByGm(int coreIdx, int64_t dotStatus)
    {
        if constexpr (IsDeviceMode()) {
            auto args = reinterpret_cast<KernelArgs*>((static_cast<uint64_t>(sharedBuffer_)) +
                                                      SHARED_BUFFER_SIZE * coreIdx);
            volatile int64_t* shakeBuffer = args->shakeBuffer;
            if ((*shakeBuffer & 0xFFFFFFFF) != AICORE_SAY_HELLO) {
                return false;
            }

            args_[coreIdx] = args;
            args->taskEntry.reserved[0] = static_cast<uint32_t>(dotStatus);
            GetPhyIdByBlockId(coreIdx) = (*shakeBuffer >> NUM_THIRTY_TWO) & AICORE_COREID_MASK;
            if (isNeedWriteRegForFastPath_) {
                WriteReg32(coreIdx, REG_SPR_FAST_PATH_ENABLE, REG_SPR_FAST_PATH_OPEN);
            }
            SetReadyQueue(coreIdx, (uint64_t)0);
            // make sure reset wave goodbye flag after hand shake ,orelse impact last aicore exit through wavegoodbye
            // flag
            args_[coreIdx]->waveBufferCpuToCore[CPU_TO_CORE_SHAK_BUF_GOODBYE_INDEX] = 0;
            DEV_VERBOSE_DEBUG("hand shake success coreIdx=%d", coreIdx);
            return true;
        } else {
            int32_t phyId = -1;
            volatile KernelArgs* arg = reinterpret_cast<KernelArgs*>(sharedBuffer_ + coreIdx * SHARED_BUFFER_SIZE);
            args_[coreIdx] = arg;
            bool handShakeSuccess = GetActiveModel()->TryHandShakeByGm(arg, dotStatus, phyId);
            if (!handShakeSuccess) {
                return false;
            }
            args_[coreIdx] = arg;
            GetPhyIdByBlockId(coreIdx) = phyId;
            SetReadyQueue(coreIdx, (uint64_t)0);
            return true;
        }
    }

    // Device: publish GOODBYE only. Destask + HELLO zeros are in ResetCoreStopSlot.
    void SendWaveGoodbye(int coreIdx)
    {
        if constexpr (IsDeviceMode()) {
            SendWaveGoodbyeDeviceOne(coreIdx);
        } else {
            ResetCoreStopSlot(coreIdx);
            GetActiveModel()->ResetShakeBuf(args_[coreIdx]);
        }
    }

    void SendWaveGoodbye(int coreStart, int coreEnd);

    void ResetCoreStopSlot(int coreStart, int coreEnd);

    inline void InitKernelArgs(int coreIdx, int64_t buffer)
    {
        (void)buffer;
        if constexpr (IsDeviceMode()) {
            if (args_[coreIdx] == nullptr) {
                args_[coreIdx] = reinterpret_cast<KernelArgs*>((static_cast<uint64_t>(sharedBuffer_)) +
                                                               SHARED_BUFFER_SIZE * coreIdx);
            }
        } else {
            GetActiveModel()->InitKernelArgs(args_[coreIdx], coreIdx, sharedBuffer_, buffer);
        }
    }

    volatile ParallelDevTask* GetParallelDevTask(int coreIdx)
    {
        volatile KernelArgs* arg = args_[coreIdx];
        return &arg->parallelDevTask;
    }

    void SetParallelDevTask(volatile ParallelDevTask* kernelParallDevTask, int parallelIdx, int64_t funcData,
                            uint32_t devTaskId)
    {
        DEV_IF_DEVICE
        {
            kernelParallDevTask->ptrElements[parallelIdx % npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM] = funcData;
            kernelParallDevTask->idElements[parallelIdx % npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM] = devTaskId;
        }
        else
        {
            GetActiveModel()->SetParallelDevTask(kernelParallDevTask, parallelIdx, funcData, devTaskId);
        }
    }

    void SetParallelDevTaskSize(volatile ParallelDevTask* kernelParallDevTask, uint32_t front, uint32_t rear)
    {
        DEV_IF_DEVICE
        {
            kernelParallDevTask->front = front;
            kernelParallDevTask->rear = rear;
        }
        else
        {
            GetActiveModel()->SetParallelDevTaskSize(kernelParallDevTask, front, rear);
        }
    }

    inline uint32_t ParallelDevTaskCtxVersion(int coreIdx) { return parallelDevTaskCtxVersion[coreIdx]; }

    inline void SetParallelDevTaskCtxVersion(int coreIdx, uint32_t version)
    {
        parallelDevTaskCtxVersion[coreIdx] = version;
        DEV_IF_DEVICE
        {
            volatile KernelArgs* arg = args_[coreIdx];
            arg->parallelDevTask.version = version;
        }
        else
        {
            GetActiveModel()->SetParallelDevTaskCtxVersion(args_[coreIdx], version);
        }

        DEV_VERBOSE_DEBUG("Refresh core %d parall version %u", coreIdx, version);
    }

    void ResetCoreStopSlot(int coreIdx)
    {
        DEV_IF_DEVICE { ResetCoreStopSlotDeviceOne(coreIdx); }
        else
        {
            GetActiveModel()->ResetParallelDevTask(args_[coreIdx]);
        }
    }

    inline void InitCostModelDevTaskData(int coreIdx, int64_t funcData)
    {
        if constexpr (!IsDeviceMode()) {
            GetActiveModel()->InitCostModelDevTaskData(coreIdx, funcData);
        }
    }

private:
    // Device BatchStop: GOODBYE only. HELLO zeros are in ResetCoreStopSlotDeviceOne
    // so they become visible before this store (see BatchStopAllManagedCores fence).
    void SendWaveGoodbyeDeviceOne(int coreIdx)
    {
        args_[coreIdx]->waveBufferCpuToCore[CPU_TO_CORE_SHAK_BUF_GOODBYE_INDEX] = AICORE_SAY_GOODBYE;
    }

    void ResetCoreStopSlotDeviceOne(int coreIdx);

    inline ModelBase* GetActiveModel()
    {
        if constexpr (IsDeviceMode()) {
            return nullptr;
        } else {
            if (enableEslModel_) {
                return &eslModel_;
            }
            if (AicoreHardware::Global().CoreNum() > 0) {
                return &aicoreModel_;
            }
            return &costModelAdapter_;
        }
    }

    int64_t sharedBuffer_;
    int64_t* regAddrs_{nullptr};
    int aicStart_{0};
    int aicEnd_{0};
    int aivStart_{0};
    int aivEnd_{0};
    uint32_t regNum_{0};
    uint64_t freq_{50};

    std::array<volatile KernelArgs*, MAX_AICORE_NUM> args_;

    std::array<volatile uint64_t*, MAX_AICORE_NUM> readyRegQueues_;
    std::array<volatile uint64_t*, MAX_AICORE_NUM> finishRegQueues_;

    // cost model aicore
    std::function<uint64_t(uint64_t, uint64_t, uint64_t)> getTaskTimeCost{nullptr};
    std::array<std::deque<uint64_t>, MAX_AICORE_NUM> taskIds;
    std::array<std::deque<uint64_t>, MAX_AICORE_NUM> taskTimes;

    std::array<int, MAX_AICORE_NUM> blockIdToPhyCoreId_;
    std::array<uint32_t, MAX_AICORE_NUM> parallelDevTaskCtxVersion;

    uint32_t regSprDataMainBase_{DAV_2201::REG_SPR_DATA_MAIN_BASE};
    uint32_t regSprCond_{DAV_2201::REG_SPR_COND};

    bool isNeedWriteRegForFastPath_{true};
    AiCoreProf* aicoreProf_{nullptr};
    CostModel::AiCoreModel* costModel_{nullptr};

    bool enableEslModel_;
    ArchInfo archInfo_{ArchInfo::DAV_2201};
    AicoreModel aicoreModel_;
    EslModel eslModel_;
    CostModelAdapter costModelAdapter_;
};

} // namespace npu::tile_fwk::dynamic

#include "machine/device/dynamic/aicore_hal_inl.h"
