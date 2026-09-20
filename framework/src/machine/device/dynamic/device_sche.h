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
 * \file device_machine.h
 * \brief
 */

#pragma once

#include <signal.h>
#include <sys/ucontext.h>
#include "device_sche_context.h"
#include "device_common.h"
#include "aicore_manager.h"
#include "aicore_constants.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/utils/device_log.h"
#include "machine/device/dynamic/aicore_prof.h"
#include "device_trace.h"
#include "device_sche_alloc_thread.h"
#include "device_sche_wait_ctrl.h"

constexpr uint32_t LAUNCH_AICPU_NUM = 5;
constexpr int MAX_RETRIES = 100;

namespace npu::tile_fwk::dynamic {

typedef void (*sig_act_f)(int signum, siginfo_t* info, void* act);

class DeviceSchedMachine {
public:
    DeviceSchedMachine()
    {
        for (uint32_t i = 0; i < MAX_SCHEDULE_AICPU_NUM; ++i) {
            aicoreManager_[i] = std::make_unique<AiCoreManager>(schThreadStatus);
        }
    }

    void SetStachSchduleContext(int schedIdx, SchduleContext* context)
    {
        aicoreManager_[schedIdx]->SetSchduleContext(context);
    }

    bool CheckAndResetReg() { return aicoreManager_[0]->CheckAndResetReg(); }

    void init(uint32_t schNum)
    {
        schAicpuNum_ = schNum;
        schThreadStatus.Init();
    }

    void SetDevSchedSyncMode(uint8_t devScheSyncMode)
    {
        devScheSyncMode_ = devScheSyncMode;
        DEV_INFO("Current front set devSchedSynMode is %s", devScheSyncMode == 1 ? "SynMode" : "AsyncMode");
    }

    int RunThread(int threadIdx, DevStartArgs* devStartArgs, DeviceArgs* args, int schedIdx, int arbitratedScheNum)
    {
        int ret = 0;
        if (args->nrAic == 0 || args->nrValidAic == 0 || args->nrAicpu < args->scheCpuNum) {
            DEV_ERROR(DevCommonErr::PARAM_INVALID,
                      "#sche.thread.init: Device machine run invalid args: aicNum=%u, blockdim=%u, launchAicpuNum=%u, "
                      "launchScheAicpuNum=%u",
                      args->nrAic, args->nrValidAic, args->nrAicpu, args->scheCpuNum);
            return DEVICE_MACHINE_ERROR;
        }

        if (static_cast<uint32_t>(schedIdx) >= args->scheCpuNum) {
            DEV_INFO("Thread start ignored");
            return DEVICE_MACHINE_OK;
        }
        aicoreManager_[schedIdx]->SetSchedSyncMode(devScheSyncMode_);
        ret = aicoreManager_[schedIdx]->RunManager(threadIdx, devStartArgs, args, schedIdx, arbitratedScheNum);
        DEV_INFO("threadIdx=%d end, ret=%d", threadIdx, ret);
        return ret;
    }

    void ResetRegAll()
    {
        sleep(1);
        DEV_INFO("ResetRegAll");
        for (uint32_t i = 0; i < schAicpuNum_; ++i) {
            aicoreManager_[i]->ResetRegAll();
        }
        sleep(1);
        aicoreManager_[0]->CheckAndResetReg();
        DEV_INFO("Exception register reset completed.");
    }

private:
    uint8_t devScheSyncMode_{0};
    SchThreadStatus schThreadStatus;
    uint32_t schAicpuNum_{MAX_SCHEDULE_AICPU_NUM};
    std::unique_ptr<AiCoreManager> aicoreManager_[MAX_SCHEDULE_AICPU_NUM];
};

constexpr int CPUS_PER_CLUSTER = 4;
static constexpr uint64_t SIGNAL_DELAY_SECONDS = 2;
constexpr int SCHE_THREAD_START_IDX = 1;

struct DynMachineManager {
    struct KernelCtrlEntry {
        int (*kernelCtrlServerInit)(void* targ);
        int (*kernelCtrlServer)(void* targ);
    };

    static int AllocThreadIdxForDav3510(DeviceArgs* devArgs, int cpu, int& curThreadIdx, std::atomic<int>& threadIdx,
                                        std::atomic<uint64_t>& cpumask, int& arbitratedScheNum,
                                        std::atomic<int>& arbitrationLevel, std::atomic<uint64_t>& arbitrationCpumask,
                                        std::atomic<uint64_t>& threadIdxBitmap)
    {
        return AllocThreadIdxForDav3510Impl(devArgs, cpu, curThreadIdx, threadIdx, cpumask, arbitratedScheNum,
                                            arbitrationLevel, arbitrationCpumask, threadIdxBitmap);
    }

    static int AllocThreadIdxForDav2201(DeviceArgs* devArgs, int cpu, int& curThreadIdx, std::atomic<int>& threadIdx,
                                        std::atomic<uint64_t>& cpumask, int& arbitratedScheNum,
                                        std::atomic<int>& arbitrationLevel)
    {
        return AllocThreadIdxForDav2201Impl(devArgs, cpu, curThreadIdx, threadIdx, cpumask, arbitratedScheNum,
                                            arbitrationLevel);
    }

    static int AllocThreadIdx(DeviceArgs* devArgs, int& curThreadIdx, std::atomic<int>& threadIdx,
                              std::atomic<uint64_t>& cpumask, int& arbitratedScheNum,
                              std::atomic<int>& arbitrationLevel, std::atomic<int>& simCpuId,
                              std::atomic<uint64_t>& arbitrationCpumask, std::atomic<uint64_t>& threadIdxBitmap)
    {
        int ret = npu::tile_fwk::dynamic::DEVICE_MACHINE_OK;

#ifdef __DEVICE__
        int cpu = sched_getcpu();
        (void)simCpuId;
#else
        int cpu = ++simCpuId;
#endif
        if (devArgs->archInfo == ArchInfo::DAV_3510) {
            ret = AllocThreadIdxForDav3510(devArgs, cpu, curThreadIdx, threadIdx, cpumask, arbitratedScheNum,
                                           arbitrationLevel, arbitrationCpumask, threadIdxBitmap);
        } else if (devArgs->archInfo == ArchInfo::DAV_2201) {
            ret = AllocThreadIdxForDav2201(devArgs, cpu, curThreadIdx, threadIdx, cpumask, arbitratedScheNum,
                                           arbitrationLevel);
        } else {
            curThreadIdx = ++threadIdx;
        }
        return ret;
    }

    void SignalReg(const sig_act_f sigAct)
    {
        DEV_INFO("Signal register completed.");
        struct sigaction myAct;
        (void)memset_s(&myAct, sizeof(myAct), 0, sizeof(myAct));
        sigemptyset(&myAct.sa_mask);
        myAct.sa_flags = SA_SIGINFO;
        myAct.sa_sigaction = sigAct;
        sigaction(SIGFPE, &myAct, &oriFPEAct_);
        sigaction(SIGBUS, &myAct, &oriBUSAct_);
        sigaction(SIGSEGV, &myAct, &oriSEGVAct_);
        sigaction(SIGPIPE, &myAct, &oriPIPEAct_);
        sigaction(SIGILL, &myAct, &oriILLAct_);
        sigaction(SIGABRT, &myAct, &oriBordAct_);
        return;
    }

    int RunCtrl(DeviceKernelArgs* kargs, const KernelCtrlEntry& entry, int threadIdx)
    {
        DEV_TRACE_DEBUG(schema::CtrlEvent(threadIdx, schema::ThreadStart()));

        DEV_INFO("ThreadCtrlEnter idx=%d", threadIdx);

        int ret = entry.kernelCtrlServer(static_cast<void*>(kargs));

        DEV_INFO("ThreadCtrlLeave idx=%d ret=%d", threadIdx, ret);
        return ret;
    }

    int RunSche(DeviceKernelArgs* kargs, const KernelCtrlEntry& entry, int threadIdx, int arbitratedScheNum,
                DeviceArgs* roundDevArgs)
    {
        UNUSED(entry);

        DEV_INFO("DeviceMode=%s, isDeviceMode=%d, stage=%s, threadIdx=%d", IsDeviceMode() ? "device" : "sim",
                 IsDeviceMode(), "RunSche.before", threadIdx);
        DEV_INFO("ThreadScheEnter idx=%d", threadIdx);

        DEV_INFO("TaskType=%d, threadIdx=%d, aicNum=%u, aivNum=%u, aicpuNum=%u, validAicNum=%u, scheCpuNum=%u.",
                 static_cast<int>(roundDevArgs->taskType), threadIdx, roundDevArgs->nrAic, roundDevArgs->nrAiv,
                 roundDevArgs->nrAicpu, roundDevArgs->nrValidAic, roundDevArgs->scheCpuNum);
        DEV_INFO("devQueueAddr=%#lx, sharedBuffer=%#lx, coreRegAddr=%#lx, corePmuAddr=%#lx.",
                 roundDevArgs->devQueueAddr, roundDevArgs->sharedBuffer, roundDevArgs->coreRegAddr,
                 roundDevArgs->corePmuAddr);
        DEV_TRACE_DEBUG(schema::ScheEvent(threadIdx, schema::ThreadStart()));

        roundDevArgs->toSubMachineConfig = kargs->toSubMachineConfig;
        SchduleContext localContext;
        int schedIdx = threadIdx - SCHE_THREAD_START_IDX;
        schMachine_.SetStachSchduleContext(schedIdx, &localContext);
        DevAscendProgram* devProg = reinterpret_cast<DevAscendProgram*>(kargs->cfgdata);
        DevStartArgs* devStartArgs = reinterpret_cast<DevStartArgs*>(
            devProg->GetRuntimeDataList()->GetRuntimeDataCurrent());
        schMachine_.SetDevSchedSyncMode(kargs->schedSyncMode);
        int ret = schMachine_.RunThread(threadIdx, devStartArgs, roundDevArgs, schedIdx, arbitratedScheNum);

        DEV_INFO("ThreadScheLeave idx=%d ret=%d", threadIdx, ret);
        return ret;
    }

    void RunSchInit(uint32_t capacityScheCpuNum)
    {
        if (initSch_.load()) {
            return;
        }
        schMachine_.init(capacityScheCpuNum);
        initSch_.store(true);
    }

    void RunSchDeInit()
    {
        cpumask_ = 0;
        threadIdxBitmap_.store(0, std::memory_order_release);
        schExitNum_ = 0;
        arbitrationLevel_.store(ARBIT_UNSET, std::memory_order_release);
        ctrlWaitLevel_.store(CTRL_WAIT_UNSET, std::memory_order_release);
        initSch_.store(false);
        globalThreadIdx_.store(0);
        simCpuId_.store(0);
    }

    void RunSchPost(DevAscendProgram* devProg)
    {
        ReleaseRuntimeDataRingBuffer(devProg);
        DEV_INFO("All schedule exited, destroy the machine.");
    }

    int SyncSchExit(DevAscendProgram* devProg, const DeviceArgs& devArgs, int ret, DevStartArgs* runtimeDataCurrent,
                    int arbitratedScheNum)
    {
        if (++schExitNum_ == devArgs.nrAicpu) {
            scheFinishRound_.fetch_add(1, std::memory_order_acq_rel);
            // Queue array is sized by capacity scheCpuNum on shared DevProg, not per-round scheCpuNum.
            ResetDroppedThreadTaskQueues(runtimeDataCurrent, devProg->devArgs.scheCpuNum, arbitratedScheNum);
            UpdateScheNumForCtrl(runtimeDataCurrent, MAX_SCHEDULE_AICPU_NUM);
            RunSchPost(devProg);
            RunSchDeInit();
            PerfEvtMgr::Instance().AddScheduleTurn();
            DEV_INFO("All sche cpu exited.");
            return ret;
        }
        return ret;
    }

    int RunCtrlInitNoLock(DeviceKernelArgs* kargs, const KernelCtrlEntry& entry)
    {
        int ret = entry.kernelCtrlServerInit(kargs);
        return ret;
    }

    __sighandler_t GetSigHandle(int signum)
    {
        __sighandler_t handle = nullptr;
        if (signum == static_cast<int>(SIGFPE)) {
            handle = oriFPEAct_.sa_handler;
        } else if (signum == static_cast<int>(SIGBUS)) {
            handle = oriBUSAct_.sa_handler;
        } else if (signum == static_cast<int>(SIGSEGV)) {
            handle = oriSEGVAct_.sa_handler;
        } else if (signum == static_cast<int>(SIGPIPE)) {
            handle = oriPIPEAct_.sa_handler;
        } else if (signum == static_cast<int>(SIGILL)) {
            handle = oriILLAct_.sa_handler;
        } else if (signum == static_cast<int>(SIGABRT)) {
            handle = oriBordAct_.sa_handler;
        }
        return handle;
    }

    void SigAct(int signum, siginfo_t* info, void* act)
    {
        (void)info;
        (void)act;
        DEV_ERROR(ThreadErr::SIGNAL_HANDLER_ABNORMAL, "#sche.except.signal: Exception Signum[%d] Act.", signum);
        PrintBacktrace(ThreadErr::SIGNAL_HANDLER_ABNORMAL, "signal " + std::to_string(signum));
        if (reset_.load()) {
            DEV_WARN("#sche.except.reset: Exception Already reset.");
            sleep(SIGNAL_DELAY_SECONDS);
            return;
        }
        reset_.store(true);
        if (!initSch_.load() && !initCtrl_.load()) {
            DEV_ERROR(ThreadErr::SIGNAL_HANDLER_ABNORMAL, "#sche.except.signal: Exception call ori sigact.");
            __sighandler_t handle = GetSigHandle(signum);
            if (handle == SIG_DFL) {
                DEV_ERROR(ThreadErr::SIGNAL_HANDLER_ABNORMAL, "#sche.except.signal: Ori sigact SIG_DFL.");
                signal(signum, SIG_DFL);
                raise(signum);
            } else if (handle == SIG_IGN) {
                DEV_ERROR(ThreadErr::SIGNAL_HANDLER_ABNORMAL, "#sche.except.signal: Ori sigact SIG_IGN.");
            } else if (handle != nullptr) {
                DEV_ERROR(ThreadErr::SIGNAL_HANDLER_ABNORMAL, "#sche.except.signal: Call Ori sigact.");
                handle(signum);
            }
            return;
        }
        schMachine_.ResetRegAll();
        sigaction(SIGFPE, &oriFPEAct_, nullptr);
        sigaction(SIGBUS, &oriBUSAct_, nullptr);
        sigaction(SIGSEGV, &oriSEGVAct_, nullptr);
        sigaction(SIGPIPE, &oriPIPEAct_, nullptr);
        sigaction(SIGILL, &oriILLAct_, nullptr);
        sigaction(SIGABRT, &oriBordAct_, nullptr);
        (void)raise(signum);
        return;
    }

    void ReleaseRuntimeDataRingBuffer(DevAscendProgram* devProg)
    {
        RuntimeDataRingBufferHead* runtimeDataList = devProg->GetRuntimeDataList();
        runtimeDataList->Deallocate(runtimeDataList->GetRuntimeDataCurrent());
        DEV_INFO("Runtime data: %lu, %lu", runtimeDataList->GetIndexFinished(), runtimeDataList->GetIndexPending());
    }

    int EntrySplittedStreamCtrl(DeviceKernelArgs* kargs, const KernelCtrlEntry& entry)
    {
        // ctrl start only one thread
        PerfMtTrace(PERF_TRACE_BEGIN, 0);
        DEV_INFO("Ctrl enter round=%d", (int)kargs->parameter.globalRound);
        DEV_VERBOSE_DEBUG("#trace.round.start: round=%lu", kargs->parameter.globalRound);
        ctrlStartRound_.fetch_add(1, std::memory_order_acq_rel);
        initCtrl_.store(true);
        // Round topology (nrValidAic/scheCpuNum) is bound in InitDyn -> DevStartArgs ring slot.
        int ret = RunCtrlInitNoLock(kargs, entry);
        if (ret != 0) {
            initCtrl_.store(false);
            DeviceTrace::GetInstance().ReportTraceMsg();
            return ret;
        }
        PerfMtTrace(PERF_TRACE_INIT, CTRL_CPU_THREAD_IDX);
        kargs->taskWastTime = GetCycles();
        ret = RunCtrl(kargs, entry, 0);
        PerfMtTrace(PERF_TRACE_EXIT, 0);
        DEV_INFO("Ctrl leave ret=%d", ret);
        DEV_VERBOSE_DEBUG("#trace.round.end: round=%lu ret=%d", kargs->parameter.globalRound, ret);
        if (ret != DEVICE_MACHINE_OK) {
            DeviceTrace::GetInstance().ReportTraceMsg();
        }
        initCtrl_.store(false);
        PerfEvtMgr::Instance().AddCtrlTurn();
        return ret;
    }

    static int WaitForCtrlAndRingBuffer(DevAscendProgram* devProg, ArchInfo archInfo, int& threadIdx,
                                        int& arbitratedScheNum, std::atomic<int>& ctrlWaitLevel,
                                        std::atomic<uint64_t>& ctrlRound, std::atomic<uint64_t>& scheRound)
    {
        int ctrlDecisionRet = WaitForCtrlDecision(archInfo, threadIdx, arbitratedScheNum, ctrlWaitLevel, ctrlRound,
                                                  scheRound);
        if (ctrlDecisionRet != DEVICE_MACHINE_OK) {
            DEV_ERROR(SchedErr::WAIT_CTRL_TIMEOUT,
                      "#sche.wait.ctrl_decision: WaitForCtrlDecision failed, ret=%d, threadIdx=%d, "
                      "arbitratedScheNum=%d, ctrlWaitLevel=%d, ctrlRound=%lu, scheRound=%lu. "
                      "Will skip SyncSchExit/RunSchDeInit.",
                      ctrlDecisionRet, threadIdx, arbitratedScheNum, ctrlWaitLevel.load(std::memory_order_acquire),
                      ctrlRound.load(std::memory_order_acquire), scheRound.load(std::memory_order_acquire));
            DeviceTrace::GetInstance().ReportTraceMsg();
            return ctrlDecisionRet;
        }
        if (threadIdx != -1) {
            int scheWaitRet = SplittedInfo::ScheWait(devProg);
            if (scheWaitRet != DEVICE_MACHINE_OK) {
                DEV_ERROR(SchedErr::RINGBUFFER_WAIT_TIMEOUT,
                          "#sche.wait.ring: ScheWait failed, ret=%d, threadIdx=%d, arbitratedScheNum=%d, "
                          "ctrlRound=%lu, scheRound=%lu. Will skip SyncSchExit/RunSchDeInit.",
                          scheWaitRet, threadIdx, arbitratedScheNum, ctrlRound.load(std::memory_order_acquire),
                          scheRound.load(std::memory_order_acquire));
                DeviceTrace::GetInstance().ReportTraceMsg();
                return scheWaitRet;
            }
        }
        return DEVICE_MACHINE_OK;
    }

    static void UpdateScheNumForCtrl(DevStartArgs* runtimeDataCurrent, int scheAiCpuNum)
    {
        runtimeDataCurrent->devCtrlState.arbitratedScehNum.store(scheAiCpuNum);
    }

    static void ResetDroppedThreadTaskQueues(DevStartArgs* runtimeDataCurrent, uint32_t scheCpuNum,
                                             int arbitratedScheNum)
    {
        for (size_t i = arbitratedScheNum; i < scheCpuNum; i++) {
            runtimeDataCurrent->deviceRuntimeDataDesc.taskQueueList[i].ResetEmpty();
        }
    }

    int EntrySplittedStreamSche(DeviceKernelArgs* kargs, const KernelCtrlEntry& entry)
    {
        DevAscendProgram* devProg = PtrToPtr<int64_t, DevAscendProgram>(kargs->cfgdata);
        auto beginTime = GetCycles(); // After wait, the devStartArgs should be ready.
        // One local DeviceArgs: capacity fields from shared DevProg, then overlay per-round topology.
        DeviceArgs roundArgs = devProg->devArgs;
        const uint32_t capacityScheCpuNum = roundArgs.scheCpuNum;
        roundArgs.nrValidAic = ResolveRoundNrValidAic(kargs->parameter.ctrlBlockNum, roundArgs.nrValidAic);
        roundArgs.scheCpuNum = ResolveRoundScheCpuNum(roundArgs.nrValidAic, capacityScheCpuNum, roundArgs.archInfo);
        DEV_INFO("Sche round topology: ctrlBlockNum=%lu nrValidAic=%u scheCpuNum=%u (capacity sche=%u)",
                 kargs->parameter.ctrlBlockNum, roundArgs.nrValidAic, roundArgs.scheCpuNum, capacityScheCpuNum);

        int threadIdx = -1;
        // schMachine_ init once with capacity so later rounds can still use up to capacity sche threads.
        RunSchInit(capacityScheCpuNum);
        int arbitratedScheNum = static_cast<int>(roundArgs.scheCpuNum);
        if (AllocThreadIdx(&roundArgs, threadIdx, globalThreadIdx_, cpumask_, arbitratedScheNum, arbitrationLevel_,
                           simCpuId_, arbitrationCpumask_, threadIdxBitmap_) != DEVICE_MACHINE_OK) {
            DEV_ERROR(
                ThreadErr::THREAD_CPU_ALLOC_FAILED,
                "#sche.thread.init: Current cpu[%d] alloc thread failed. round=%lu scheCpuNum=%u nrAicpu=%u "
                "arbitratedScheNum=%d cpumask=0x%lx arbitrationCpumask=0x%lx arbitrationLevel=%d "
                "threadIdxBitmap=0x%lx schExitNum=%u ctrlStartRound=%lu scheFinishRound=%lu. "
                "Will skip SyncSchExit/RunSchDeInit.",
                sched_getcpu(), kargs->parameter.globalRound, roundArgs.scheCpuNum, roundArgs.nrAicpu,
                arbitratedScheNum, cpumask_.load(std::memory_order_acquire),
                arbitrationCpumask_.load(std::memory_order_acquire), arbitrationLevel_.load(std::memory_order_acquire),
                threadIdxBitmap_.load(std::memory_order_acquire), schExitNum_.load(std::memory_order_acquire),
                ctrlStartRound_.load(std::memory_order_acquire), scheFinishRound_.load(std::memory_order_acquire));
            DEV_ATRACE("Schedule Current cpu[%d] alloc thread failed", sched_getcpu());
            DeviceTrace::GetInstance().ReportTraceMsg();
            return npu::tile_fwk::dynamic::DEVICE_MACHINE_ERROR;
        }
        if (threadIdx != -1) {
            int waitCtrlRet = WaitForCtrlAndRingBuffer(devProg, roundArgs.archInfo, threadIdx, arbitratedScheNum,
                                                       ctrlWaitLevel_, ctrlStartRound_, scheFinishRound_);
            if (waitCtrlRet != DEVICE_MACHINE_OK) {
                DEV_ERROR(
                    SchedErr::WAIT_CTRL_TIMEOUT,
                    "#sche.wait: WaitForCtrlAndRingBuffer failed: ret=%d threadIdx=%d round=%lu "
                    "arbitratedScheNum=%d ctrlStartRound=%lu scheFinishRound=%lu cpumask=0x%lx "
                    "arbitrationLevel=%d threadIdxBitmap=0x%lx schExitNum=%u. "
                    "Will skip SyncSchExit/RunSchDeInit.",
                    waitCtrlRet, threadIdx, kargs->parameter.globalRound, arbitratedScheNum,
                    ctrlStartRound_.load(std::memory_order_acquire), scheFinishRound_.load(std::memory_order_acquire),
                    cpumask_.load(std::memory_order_acquire), arbitrationLevel_.load(std::memory_order_acquire),
                    threadIdxBitmap_.load(std::memory_order_acquire), schExitNum_.load(std::memory_order_acquire));
                return waitCtrlRet;
            }
        }
        PerfMtTrace(PERF_TRACE_ALLOC_THREAD_ID, threadIdx);
        PerfMtTrace(PERF_TRACE_BEGIN, threadIdx, beginTime);
        int ret = DEVICE_MACHINE_OK;
        DevStartArgs* runtimeDataCurrent = reinterpret_cast<DevStartArgs*>(
            devProg->GetRuntimeDataList()->GetRuntimeDataCurrent());
        if (threadIdx != -1 && threadIdx <= arbitratedScheNum) {
            // Topology already from this launch's ctrlBlockNum (same as CTRL InitDyn); no need to re-read slot.
            DEV_VERBOSE_DEBUG("#trace.round.start: round=%lu tid=%d", kargs->parameter.globalRound, threadIdx);
            DEV_INFO("SchedThreadEnter idx=%d round=%d nrValidAic=%u scheCpuNum=%u", threadIdx,
                     (int)kargs->parameter.globalRound, roundArgs.nrValidAic, roundArgs.scheCpuNum);
            UpdateScheNumForCtrl(runtimeDataCurrent, arbitratedScheNum);
            ResetDroppedThreadTaskQueues(runtimeDataCurrent, capacityScheCpuNum, arbitratedScheNum);
            ret = RunSche(kargs, entry, threadIdx, arbitratedScheNum, &roundArgs);
            DEV_INFO("SchedThreadLeave idx=%d ret=%d", threadIdx, ret);
            if (ret != DEVICE_MACHINE_OK) {
                DeviceTrace::GetInstance().ReportTraceMsg();
            }
            PerfMtTrace(PERF_TRACE_EXIT, threadIdx);
            DEV_VERBOSE_DEBUG("#trace.round.end: round=%lu tid=%d ret=%d", kargs->parameter.globalRound, threadIdx,
                              ret);
        }
        return SyncSchExit(devProg, roundArgs, ret, runtimeDataCurrent, arbitratedScheNum);
    }

    int Entry(DeviceKernelArgs* kargs, const KernelCtrlEntry& entry)
    {
        switch (kargs->parameter.runMode) {
            case RUN_SPLITTED_STREAM_CTRL:
                return EntrySplittedStreamCtrl(kargs, entry);
                break;
            case RUN_SPLITTED_STREAM_SCHE:
                return EntrySplittedStreamSche(kargs, entry);
                break;
            default:
                DEV_ERROR(DevCommonErr::PARAM_INVALID, "#dev.entry.invalid_mode: Invalid run mode: %d\n",
                          (int)kargs->parameter.runMode);
                break;
        }
        return DEVICE_MACHINE_INVALID_RUN_MODE;
    }

    int LastFinishThreadIdx_{0};
    std::atomic<uint64_t> cpumask_{0};
    std::atomic<uint64_t> arbitrationCpumask_{0}; // 仲裁线程在 WaitForCpuMaskReadyForArbitration 中写入，
                                                  // 所有线程在仲裁完成后通过 PerformArbitrationDav3510 的
                                                  // globalArbitrationLevel release-acquire 保证 happens-before 后读取
    std::atomic<uint64_t> threadIdxBitmap_{0}; // threadIdx 分配位图，用于检测重复分配
    std::atomic<uint32_t> schExitNum_{0};
    std::atomic<int> arbitrationLevel_{ARBIT_UNSET};
    std::atomic<int> ctrlWaitLevel_{CTRL_WAIT_UNSET};
    std::atomic<int> simCpuId_{0};
    DeviceSchedMachine schMachine_;
    struct sigaction oriFPEAct_;
    struct sigaction oriBUSAct_;
    struct sigaction oriSEGVAct_;
    struct sigaction oriPIPEAct_;
    struct sigaction oriILLAct_;
    struct sigaction oriBordAct_;
    std::atomic<bool> reset_{false};
    std::atomic<bool> initCtrl_{false};
    std::atomic<bool> initSch_{false};
    std::atomic<bool> schRunFailed_{false};
    std::atomic<int> globalThreadIdx_{0};
    std::atomic<uint64_t> ctrlStartRound_{0};
    std::atomic<uint64_t> scheFinishRound_{0};

    struct SplittedInfo {
        static int ScheWait(DevAscendProgram* devProg)
        {
            TIMEOUT_CHECK_INIT(devProg->devArgs.archInfo, TIMEOUT_1MIN);

            while (unlikely(!devProg->runtimeDataRingBufferInited)) {
                RuntimeYield(0);

                __PYPTO_TIMEOUT_CHECK(SchedErr::RINGBUFFER_WAIT_TIMEOUT, return DEVICE_MACHINE_ERROR,
                                      "#sche.wait: RingBuffer init.");
            }
            RuntimeDataRingBufferHead* ringBufferHead = devProg->GetRuntimeDataList();
            start = GetCycles();

            while (unlikely(ringBufferHead->Empty())) {
                RuntimeYield(0);

                __PYPTO_TIMEOUT_CHECK(SchedErr::RINGBUFFER_WAIT_TIMEOUT, return DEVICE_MACHINE_ERROR,
                                      "#sche.wait: RingBuffer data.");
            }

            return DEVICE_MACHINE_OK;
        }
    } splittedInfo_;
};

} // namespace npu::tile_fwk::dynamic
