/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "machine/runtime/launcher/aicore_model_launcher.h"

#include <cstdint>
#include <stdint.h>
#include <pthread.h>
#include <thread>
#include <vector>
#include <unistd.h>
#include "machine/utils/device_log.h"
#include "machine/runtime/runner/runtime_utils.h"
#include "tilefwk/aicpu_perf.h"
#if defined(__linux__)
#include <sched.h>
#endif
#include "tilefwk/pypto_fwk_log.h"
#include "machine/simulation/host_aicore_entry_adapter.h"
#include "interface/machine/device/tilefwk/aicore_entry.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/device/dynamic/device_utils.h"
#include "machine/simulation/host_core_context.h"
#include "machine/simulation/aicore_hardware.h"
#include "machine/simulation/aicore_model_kernel_meta_hook.h"
#include "machine/runtime/launcher/device_launcher.h"

extern "C" int DynTileFwkBackendKernelServer(void* targ);

namespace npu::tile_fwk::dynamic {

constexpr int AIV_NUM_PER_AI_CORE_LOCAL = 2;
constexpr int SCHE_CPU_NUM = 3;
constexpr int AIC_PER_SCHE = 8;
constexpr int AICORE_PER_SCHE = 24;
constexpr int CTRL_THREAD_CPU_BASE = 0;
constexpr int SCHE_THREAD_CPU_BASE = 1;
constexpr int AICORE_THREAD_CPU_BASE = 4;
constexpr int AICORE_CORES_PER_SCHE_36CORE = 12;
constexpr int AICORE_CORES_PER_SCHE_24CORE = 7;

constexpr uint32_t AICORE_MODEL_NR_AIC = 24;
constexpr uint32_t AICORE_MODEL_NR_AIV = 48;
constexpr uint32_t AICORE_MODEL_NR_VALID_AIC = 24;
constexpr uint32_t AICORE_MODEL_NR_AICPU = 4;
constexpr uint32_t AICORE_MODEL_SCHE_CPU_NUM = 3;
constexpr int AICORE_MODEL_AICPU_THREAD_NUM = MAX_LAUNCH_SCHEDULE_AICPU_NUM +
                                              static_cast<int>(dynamic::MAX_CONTROL_FLOW_AICPU_NUM);

static int GetSystemCpuCount()
{
#if defined(__linux__)
    cpu_set_t cpuSet;
    CPU_ZERO(&cpuSet);
    if (sched_getaffinity(0, sizeof(cpu_set_t), &cpuSet) == 0) {
        return CPU_COUNT(&cpuSet);
    }
#endif
    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    return (nprocs > 0) ? static_cast<int>(nprocs) : 24;
}

static int CalcBindCpuForAicore(int aicoreIdx, int nrValidAic, int totalCpuCores, int scheCpuNum)
{
    (void)nrValidAic;
    (void)scheCpuNum;
    int cpuBase = AICORE_THREAD_CPU_BASE;
    int bindCoreNum = totalCpuCores - cpuBase;
    if (bindCoreNum <= 0) {
        bindCoreNum = totalCpuCores;
        cpuBase = 0;
    }
    int hostCpu = cpuBase + (aicoreIdx % bindCoreNum);
    return hostCpu;
}

static void SetThreadAffinity(int cpuId, const char* threadName)
{
#if defined(__linux__)
    if (cpuId < 0) {
        return;
    }
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpuId, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (threadName != nullptr) {
        pthread_setname_np(pthread_self(), threadName);
    }
#else
    (void)cpuId;
    (void)threadName;
#endif
}

static void InitHostSimRegAddrArray(DevAscendProgram* devProg, int aicoreNum)
{
    if (devProg == nullptr || aicoreNum <= 0 || devProg->devArgs.coreRegAddr == 0) {
        return;
    }
    uint64_t* coreRegAddr = reinterpret_cast<uint64_t*>(static_cast<uintptr_t>(devProg->devArgs.coreRegAddr));
    if (coreRegAddr == nullptr) {
        return;
    }
    for (int i = 0; i < aicoreNum; ++i) {
        coreRegAddr[i] = AicoreHardware::Global().GetReg32Addr(static_cast<size_t>(i));
    }
}

static void AicoreModelWorker(DeviceKernelArgs kArgs, int blockId, int phyId, int bindCpu, int nrValidAic,
                              int scheCpuNum)
{
    char name[32] = {0};
    bool isAic = (blockId < nrValidAic);
    const char* role = isAic ? "aic" : "aiv";
    const char* threadName = nullptr;
    (void)scheCpuNum;
    if (sprintf_s(name, sizeof(name), "aicore_model_%s_%d", role, blockId) < 0) {
        DEV_WARN("sprintf_s failed, role: %s, blockId: %d", role, blockId);
    }
    threadName = name;
    SetThreadAffinity(bindCpu, threadName);

    HostCoreContext ctx;
    ctx.blockId = blockId;
    ctx.phyId = phyId;
    HostCoreCtx::SetCurrent(ctx);
    KernelEntry(0, 0, 0, 0, 0, reinterpret_cast<int64_t>(kArgs.cfgdata));
}

static int InitAicoreModelRuntime(DevAscendProgram* devProg)
{
    if (devProg == nullptr) {
        return 0;
    }
    devProg->devArgs.nrAic = AICORE_MODEL_NR_AIC;
    devProg->devArgs.nrAiv = AICORE_MODEL_NR_AIV;
    devProg->devArgs.nrValidAic = AICORE_MODEL_NR_VALID_AIC;
    devProg->devArgs.nrAicpu = AICORE_MODEL_NR_AICPU;
    devProg->devArgs.scheCpuNum = AICORE_MODEL_SCHE_CPU_NUM;
    int aicoreNum = static_cast<int>(devProg->devArgs.nrAic + devProg->devArgs.nrAiv);
    AicoreHardware::Global().Reset(static_cast<size_t>(aicoreNum));
    InitHostSimRegAddrArray(devProg, aicoreNum);
    size_t shmSize = DEVICE_TASK_CTRL_POOL_SIZE + DEVICE_TASK_QUEUE_SIZE * devProg->devArgs.scheCpuNum;
    auto deviceTaskCtrlPoolAddr = devProg->GetRuntimeDataList()->GetRuntimeData() + DEV_ARGS_SIZE;
    (void)memset_s(reinterpret_cast<void*>(deviceTaskCtrlPoolAddr), shmSize, 0, shmSize);
    devProg->devArgs.aicpuPerfAddr = 0UL;
    return aicoreNum;
}

static int AicoreModelLaunchOnce(DeviceKernelArgs& kArgs)
{
    std::thread aicpuThreadList[AICORE_MODEL_AICPU_THREAD_NUM];
    int aicpuResultList[AICORE_MODEL_AICPU_THREAD_NUM] = {0};
    auto* devProg = (DevAscendProgram*)(kArgs.cfgdata);
    int aicoreNum = InitAicoreModelRuntime(devProg);

    std::vector<std::thread> aicoreThreadList;
    aicoreThreadList.reserve(static_cast<size_t>(aicoreNum));
    int launchAiCpuNum = static_cast<int>(devProg->devArgs.nrAicpu + dynamic::MAX_CONTROL_FLOW_AICPU_NUM);
    int nrValidAic = static_cast<int>(devProg->devArgs.nrValidAic);
    int totalCpuCores = GetSystemCpuCount();
    bool enableBind = (totalCpuCores >= 16);

    auto threadFun = [&](int threadIndex, uint32_t runMode, int bindCpu) -> void {
        char name[64];
        const char* role = (runMode == RUN_SPLITTED_STREAM_CTRL) ? "ctrl" : "sche";
        int schedIdx = (runMode == RUN_SPLITTED_STREAM_CTRL) ? 0 : threadIndex;
        if (sprintf_s(name, sizeof(name), "aicore_model_%s_%d", role, schedIdx) < 0) {
            DEV_WARN("sprintf_s failed, role: %s, schedIdx: %d", role, schedIdx);
        }
        SetThreadAffinity(bindCpu, name);
        DeviceKernelArgs localArgs = kArgs;
        localArgs.parameter.runMode = runMode;
        aicpuResultList[threadIndex] = DynTileFwkBackendKernelServer(&localArgs);
    };

    int ctrlBindCpu = enableBind ? CTRL_THREAD_CPU_BASE : -1;
    aicpuThreadList[0] = std::thread(threadFun, 0, RUN_SPLITTED_STREAM_CTRL, ctrlBindCpu);

    for (int i = 1; i < launchAiCpuNum; i++) {
        int scheBindCpu = enableBind ? (SCHE_THREAD_CPU_BASE + (i - 1) % SCHE_CPU_NUM) : -1;
        aicpuThreadList[i] = std::thread(threadFun, i, RUN_SPLITTED_STREAM_SCHE, scheBindCpu);
    }

    int scheCpuNum = static_cast<int>(devProg->devArgs.scheCpuNum);
    for (int i = 0; i < aicoreNum; ++i) {
        int bindCpu = enableBind ? CalcBindCpuForAicore(i, nrValidAic, totalCpuCores, scheCpuNum) : -1;
        aicoreThreadList.emplace_back(AicoreModelWorker, kArgs, i, i, bindCpu, nrValidAic, scheCpuNum);
    }

    for (int i = 0; i < AICORE_MODEL_AICPU_THREAD_NUM; i++) {
        if (aicpuThreadList[i].joinable()) {
            aicpuThreadList[i].join();
        }
    }

    for (auto& t : aicoreThreadList) {
        if (t.joinable()) {
            t.join();
        }
    }

    for (int result : aicpuResultList) {
        if (result != 0)
            return result;
    }
    return 0;
}

int AicoreModelLauncher::AicoreModelLaunchOnceWithHostTensorData(
    Function* function, const std::vector<DeviceTensorData>& inputList, const std::vector<DeviceTensorData>& outputList,
    DevControlFlowCache* ctrlCache, AicoreModelMemoryUtils& memUtils, const DeviceLauncherConfig& config)
{
    MACHINE_LOGI("!!! Launch AicoreModel Host Sim\n");
    DeviceKernelArgs kArgs;
    auto dynAttr = function->GetDyndevAttribute();
    DeviceLauncher::DeviceInitDistributedContext(memUtils, dynAttr->commGroupNames, kArgs);
    DeviceLauncher::DeviceInitTilingData(memUtils, kArgs, dynAttr->devProgBinary, ctrlCache, config, nullptr);
    AicoreModelInitKernelMetaDeviceArgs(memUtils, kArgs, dynAttr->devProgBinary);
    DeviceLauncher::DeviceInitKernelInOuts(memUtils, kArgs, inputList, outputList, dynAttr->disableL2List);
    int rc = AicoreModelLaunchOnce(kArgs);
    return rc;
}
int AicoreModelLauncher::AicoreModelRunOnce(Function* function, DevControlFlowCache* inputCtrlCache,
                                            const DeviceLauncherConfig& config)
{
    auto& inputDataList = ProgramData::GetInstance().GetInputDataList();
    auto& outputDataList = ProgramData::GetInstance().GetOutputDataList();
    std::vector<DeviceTensorData> inputDeviceDataList;
    std::vector<DeviceTensorData> outputDeviceDataList;
    AicoreModelMemoryUtils memUtils;
    std::tie(inputDeviceDataList, outputDeviceDataList) = DeviceLauncher::BuildInputOutputFromHost(
        memUtils, inputDataList, outputDataList);
    DevControlFlowCache* launchCtrlFlowCache = nullptr;
    if (inputCtrlCache != nullptr) {
        // stitch 节点 nodeNext 低 6 位编码子链长度 R：cache 副本基址须 64B 对齐，
        // 重定位平移（dst - src）才能保留下低 6 位
        launchCtrlFlowCache = reinterpret_cast<DevControlFlowCache*>(memUtils.AllocZero(
            inputCtrlCache->usedCacheSize, nullptr, static_cast<size_t>(npu::tile_fwk::DUPPED_STITCH_NODE_ALIGN)));
        if (launchCtrlFlowCache) {
            MACHINE_LOGI(
                "#ctrl.cache.align: site=AicoreModelRunOnce base=%p size=%llu align64=%lu %s",
                static_cast<void*>(launchCtrlFlowCache), static_cast<unsigned long long>(inputCtrlCache->usedCacheSize),
                static_cast<unsigned long>(reinterpret_cast<uintptr_t>(launchCtrlFlowCache) %
                                           npu::tile_fwk::DUPPED_STITCH_NODE_ALIGN),
                reinterpret_cast<uintptr_t>(launchCtrlFlowCache) % npu::tile_fwk::DUPPED_STITCH_NODE_ALIGN == 0 ?
                    "OK" :
                    "MISALIGNED");
            MemcpyS(launchCtrlFlowCache, inputCtrlCache->usedCacheSize, inputCtrlCache, inputCtrlCache->usedCacheSize);
        }
    }
    int rc = AicoreModelLaunchOnceWithHostTensorData(function, inputDeviceDataList, outputDeviceDataList,
                                                     launchCtrlFlowCache, memUtils, config);
    return rc;
}

static std::vector<DeviceTensorData> toHostTensorData(const std::vector<DeviceTensorData>& devDataList, bool isInput)
{
    std::vector<DeviceTensorData> hostDataList;
    for (auto& devData : devDataList) {
        int64_t size = devData.GetDataSize();
        if (size == 0 || size > 0x500000000) {
            MACHINE_LOGE(DevCommonErr::PARAM_INVALID, "AllocDev failed: size=%ld bytes", size);
            return {};
        }
        void* ptr = malloc(size);
        if (isInput) {
            RuntimeMemcpy(ptr, size, devData.GetAddr(), size, RtMemcpyKind::DEVICE_TO_HOST);
        }
        hostDataList.emplace_back(devData.GetDataType(), ptr, devData.GetShape());
    }
    return hostDataList;
}

static void freeHostTensorData(const std::vector<DeviceTensorData>& hostDataList)
{
    for (auto& hostData : hostDataList) {
        free(hostData.GetAddr());
    }
}
int AicoreModelLauncher::AicoreModelLaunchDeviceTensorData(Function* function,
                                                           const std::vector<DeviceTensorData>& inDevList,
                                                           const std::vector<DeviceTensorData>& outDevList,
                                                           const DeviceLauncherConfig& config,
                                                           DevControlFlowCache* ctrlCache)
{
    AicoreModelMemoryUtils memUtils;
    ExchangeCaptureModeRelax();
    auto inList = toHostTensorData(inDevList, true);
    auto outList = toHostTensorData(outDevList, false);
    ExchangeCaptureModeGlobal();
    int rc = AicoreModelLaunchOnceWithHostTensorData(function, inList, outList, ctrlCache, memUtils, config);
    freeHostTensorData(inList);
    freeHostTensorData(outList);
    return rc;
}
} // namespace npu::tile_fwk::dynamic
