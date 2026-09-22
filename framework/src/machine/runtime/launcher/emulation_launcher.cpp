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
 * \file emulation_launcher.cpp
 * \brief
 */

#include "machine/runtime/launcher/emulation_launcher.h"
#include <map>
#include <thread>
#include "tilefwk/error_code.h"
#include "tilefwk/error.h"
#include "machine/host/backend.h"
#include "machine/runtime/launcher/device_launcher.h"
#include "machine/runtime/runner/runtime_utils.h"
#include "machine/runtime/launcher/device_launcher_binding.h"
#include "machine/runtime/bundle/pack/kernel_bundle_pack.h"

extern "C" int DynTileFwkBackendKernelServer(void* targ);

namespace npu::tile_fwk::dynamic {
static int EmulationLaunchOnce(DeviceKernelArgs& kArgs)
{
    constexpr int threadNum = MAX_LAUNCH_SCHEDULE_AICPU_NUM + static_cast<int>(dynamic::MAX_CONTROL_FLOW_AICPU_NUM);
    std::thread aicpuThreadList[threadNum];
    int aicpuResultList[threadNum] = {0};
    std::atomic<int> idx{0};
    auto* devProg = (DevAscendProgram*)(kArgs.cfgdata);
    size_t shmSize = DEVICE_TASK_CTRL_POOL_SIZE + DEVICE_TASK_QUEUE_SIZE * devProg->devArgs.scheCpuNum;
    auto deviceTaskCtrlPoolAddr = devProg->GetRuntimeDataList()->GetRuntimeData() + DEV_ARGS_SIZE;
    (void)memset_s(reinterpret_cast<void*>(deviceTaskCtrlPoolAddr), shmSize, 0, shmSize);
    devProg->devArgs.aicpuPerfAddr = 0UL;
    int launchAiCpuNum = static_cast<int>(devProg->devArgs.nrAicpu + dynamic::MAX_CONTROL_FLOW_AICPU_NUM);

    auto threadFun = [&](int threadIndex, uint32_t runMode) {
        int tidx = idx++;
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(tidx, &cpuset);
        char name[64];
        (void)sprintf_s(name, sizeof(name), "aicput%d", tidx);
        MACHINE_LOGD("start thread: %s ", name);
        pthread_setname_np(pthread_self(), name);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        DeviceKernelArgs localArgs = kArgs;
        localArgs.parameter.runMode = runMode;
        localArgs.parameter.ctrlBlockNum = devProg->ctrlBlockDim;
        aicpuResultList[threadIndex] = DynTileFwkBackendKernelServer(&localArgs);
    };

    aicpuThreadList[0] = std::thread(threadFun, 0, RUN_SPLITTED_STREAM_CTRL);

    if (!devProg->devArgs.enableAicoreResolve) {
        for (int i = 1; i < launchAiCpuNum; i++) {
            aicpuThreadList[i] = std::thread(threadFun, i, RUN_SPLITTED_STREAM_SCHE);
        }
    }

    for (int i = 0; i < threadNum; i++) {
        if (aicpuThreadList[i].joinable()) {
            aicpuThreadList[i].join();
        }
    }

    for (int i = 0; i < threadNum; i++) {
        if (aicpuResultList[i] != 0) {
            return aicpuResultList[i];
        }
    }
    return 0;
}

std::map<size_t, uintptr_t> BuildDevIdxToCpuAddrMap(const std::shared_ptr<DyndevFunctionAttribute>& dynAttr,
                                                    const std::vector<DeviceTensorData>& inputList)
{
    std::map<size_t, uintptr_t> devIdxToCpuAddr;
    if (dynAttr == nullptr || dynAttr->readyOnHostCpuPairs.empty()) {
        return devIdxToCpuAddr;
    }
    auto inputLogicalSize = dynAttr->startArgsInputLogicalTensorList.size();
    std::map<std::string, size_t> nameToIdx;
    for (size_t i = 0; i < inputLogicalSize && i < inputList.size(); i++) {
        nameToIdx[dynAttr->startArgsInputLogicalTensorList[i]->Symbol()] = i;
    }
    for (const auto& [deviceName, cpuName] : dynAttr->readyOnHostCpuPairs) {
        auto devIt = nameToIdx.find(deviceName);
        auto cpuIt = nameToIdx.find(cpuName);
        if (devIt != nameToIdx.end() && cpuIt != nameToIdx.end()) {
            devIdxToCpuAddr[devIt->second] = reinterpret_cast<uintptr_t>(inputList[cpuIt->second].GetAddr());
        }
    }
    return devIdxToCpuAddr;
}

int EmulationLauncher::EmulationBuildControlFlowCache(DeviceKernelArgs& kArgs)
{
    auto* devProg = (DevAscendProgram*)(kArgs.cfgdata);
    size_t shmSize = DEVICE_TASK_CTRL_POOL_SIZE + DEVICE_TASK_QUEUE_SIZE * devProg->devArgs.scheCpuNum;
    auto deviceTaskCtrlPoolAddr = devProg->GetRuntimeDataList()->GetRuntimeData() + DEV_ARGS_SIZE;
    (void)memset_s(reinterpret_cast<void*>(deviceTaskCtrlPoolAddr), shmSize, 0, shmSize);
    devProg->devArgs.aicpuPerfAddr = 0UL;
    kArgs.parameter.runMode = RUN_SPLITTED_STREAM_CTRL;
    return DynTileFwkBackendKernelServer(&kArgs);
    ;
}

int EmulationLauncher::EmulationLaunchOnceWithHostTensorData(
    Function* function, const std::vector<DeviceTensorData>& inputList, const std::vector<DeviceTensorData>& outputList,
    DevControlFlowCache* ctrlCache, EmulationMemoryUtils& memUtils, const DeviceLauncherConfig& config)
{
    MACHINE_LOGD("!!! Emulation Launch\n");
    DeviceKernelArgs kArgs;
    auto dynAttr = function->GetDyndevAttribute();
    DeviceLauncher::DeviceInitDistributedContext(memUtils, dynAttr->commGroupNames, kArgs);
    DeviceLauncher::DeviceInitTilingData(memUtils, kArgs, dynAttr->devProgBinary, ctrlCache, config, nullptr);
    DeviceLauncher::DeviceInitKernelInOuts(memUtils, kArgs, inputList, outputList, dynAttr->disableL2List);
    int rc = EmulationLaunchOnce(kArgs);
    return rc;
}

int EmulationLauncher::EmulationRunOnce(Function* function, DevControlFlowCache* inputCtrlCache,
                                        const DeviceLauncherConfig& config)
{
    auto& inputDataList = ProgramData::GetInstance().GetInputDataList();
    auto& outputDataList = ProgramData::GetInstance().GetOutputDataList();
    std::vector<DeviceTensorData> inputDeviceDataList;
    std::vector<DeviceTensorData> outputDeviceDataList;
    EmulationMemoryUtils memUtils;
    std::tie(inputDeviceDataList, outputDeviceDataList) = DeviceLauncher::BuildInputOutputFromHost(
        memUtils, inputDataList, outputDataList);
    DevControlFlowCache* launchCtrlFlowCache = nullptr;
    if (inputCtrlCache != nullptr) {
        launchCtrlFlowCache = reinterpret_cast<DevControlFlowCache*>(
            memUtils.AllocZero(inputCtrlCache->usedCacheSize, nullptr));
        if (launchCtrlFlowCache != nullptr) {
            MemcpyS(launchCtrlFlowCache, inputCtrlCache->usedCacheSize, inputCtrlCache, inputCtrlCache->usedCacheSize);
        }
    }
    int rc = EmulationLaunchOnceWithHostTensorData(function, inputDeviceDataList, outputDeviceDataList,
                                                   launchCtrlFlowCache, memUtils, config);
    return rc;
}

static void InitHostCtrlFlowCacheLayout(DevControlFlowCache& cache, DevAscendProgram* devProg,
                                        DyndevFunctionAttribute* dyndevAttr, uintdevptr_t& initOffset)
{
    cache.Init(dyndevAttr, devProg->ctrlFlowCacheSize, devProg->memBudget.tensor.runtimeOutcastPoolSize, initOffset,
               devProg->GetCtrlFlowCacheSlottedOutcastBlockCount(dyndevAttr->inoutLink.totalSlot));
}

DevControlFlowCache* EmulationLauncher::CreateHostCtrlFlowCache(DevAscendProgram* devProg, Function* function,
                                                                EmulationMemoryUtils& memUtils)
{
    auto* dyndevAttr = function->GetDyndevAttribute().get();
    DevControlFlowCache encodeCtrlCache;
    uintdevptr_t initOffset = reinterpret_cast<uintdevptr_t>(encodeCtrlCache.data);
    InitHostCtrlFlowCacheLayout(encodeCtrlCache, devProg, dyndevAttr, initOffset);
    uint32_t ctrlCacheAllocSize = encodeCtrlCache.GetSize();
    // stichNode在保持controlFlow 64B 对齐。
    DevControlFlowCache* hostCtrlFlowCache = reinterpret_cast<DevControlFlowCache*>(
        memUtils.AllocZero(ctrlCacheAllocSize, nullptr, static_cast<size_t>(npu::tile_fwk::DUPPED_STITCH_NODE_ALIGN)));
    if (hostCtrlFlowCache == nullptr) {
        return nullptr;
    }
    hostCtrlFlowCache->allCacheSize = ctrlCacheAllocSize;
    initOffset = reinterpret_cast<uintdevptr_t>(hostCtrlFlowCache->data);
    InitHostCtrlFlowCacheLayout(*hostCtrlFlowCache, devProg, dyndevAttr, initOffset);
    return hostCtrlFlowCache;
}

static void TryResetBlockDimForPreLaunch(DevControlFlowCache* ctrlFlowCache, DevAscendProgram* devProg,
                                         const DeviceLauncherConfig& config)
{
    // Force-disable host pre-launch control-core.
    constexpr bool kForceDisablePreLaunchControlCore = true;
    devProg->ctrlBlockDim = static_cast<uint32_t>(config.blockdim);
    if (kForceDisablePreLaunchControlCore) {
        return;
    }

    if (ctrlFlowCache == nullptr || ctrlFlowCache->deviceTaskCount > 1) {
        return; // close host control core when deviceTaskCount > 1
    }

    auto devStartArgs = (DevStartArgs*)devProg->GetRuntimeDataList()->GetRuntimeDataPending();
    if (!ctrlFlowCache->IsActivatedFullCache(devStartArgs) || ctrlFlowCache->isRecordingStopped) {
        return;
    }
    // 获取实际运行时的 maxC/maxV（从缓存的 DynDeviceTask 中）
    uint32_t actualMaxC = 0;
    uint32_t actualMaxV = 0;
    for (size_t i = 0; i < ctrlFlowCache->deviceTaskCount; i++) {
        DynDeviceTaskBase* dynTaskBase = ctrlFlowCache->deviceTaskCacheList[i].dynTaskBase;
        if (dynTaskBase == nullptr) {
            continue;
        }
        uint32_t taskMaxC = dynTaskBase->GetMaxC();
        uint32_t taskMaxV = dynTaskBase->GetMaxV();
        if (taskMaxC > actualMaxC) {
            actualMaxC = taskMaxC;
        }
        if (taskMaxV > actualMaxV) {
            actualMaxV = taskMaxV;
        }
    }

    actualMaxV = (actualMaxV & 1) ? actualMaxV + 1 : actualMaxV;
    uint32_t ctualMaxCore = std::max(actualMaxC, actualMaxV / dynamic::AIV_NUM_PER_AI_CORE);
    if ((actualMaxC != 0 && actualMaxV != 0) && ctualMaxCore <= static_cast<uint32_t>(config.blockdim)) {
        devProg->ctrlBlockDim = ctualMaxCore;
        MACHINE_LOGI("control aicore before launch, nrValidAic changed to %u\n", devProg->devArgs.nrValidAic);
    }
}

int EmulationLauncher::BuildControlFlowCacheWithEmulationTensorData(
    Function* function, const std::vector<DeviceTensorData>& inputList, const std::vector<DeviceTensorData>& outputList,
    CachedOperator* cachedOperator, DevControlFlowCache** outCtrlFlowCache, EmulationMemoryUtils& memUtils,
    const DeviceLauncherConfig& config)
{
    (void)cachedOperator;
    auto dynAttr = function->GetDyndevAttribute();
    DevAscendProgram* devProg = DeviceLauncher::GetDevProg(function);
    // Value-depend: host must not record ctrl-flow cache (DevProg.disableCtrlFlowCache from backend).
    // Single choke point for Emulation/Python/ST callers; device RT path is gated earlier in
    // CtrlFlowCacheManager::FindOrBuildDevCache via KernelBinary::DisableHostCtrlFlowCacheBuild().
    if (devProg != nullptr && devProg->disableCtrlFlowCache != 0) {
        MACHINE_LOGI("Skip host control flow cache build due to disableCtrlFlowCache.");
        if (outCtrlFlowCache != nullptr) {
            *outCtrlFlowCache = nullptr;
        }
        return 0;
    }
    DevControlFlowCache* hostCtrlFlowCache = CreateHostCtrlFlowCache(devProg, function, memUtils);
    if (hostCtrlFlowCache == nullptr) {
        MACHINE_LOGE(CtrlErr::CTRL_SIM_FAILED, "Failed to allocate control flow cache");
        return -1;
    }
    hostCtrlFlowCache->isRecording = true;
    hostCtrlFlowCache->isCacheOriginShape = config.isCacheOriginShape;
    DeviceKernelArgs kArgs;
    DeviceLauncher::DeviceInitDistributedContext(memUtils, dynAttr->commGroupNames, kArgs);
    DeviceLauncher::DeviceInitTilingData(memUtils, kArgs, dynAttr->devProgBinary, hostCtrlFlowCache, config, nullptr);
    DeviceLauncher::DeviceInitKernelInOuts(memUtils, kArgs, inputList, outputList, dynAttr->disableL2List);
    int rc = EmulationBuildControlFlowCache(kArgs);

    hostCtrlFlowCache->isRecording = false;
    hostCtrlFlowCache->isActivated = true;
    TryResetBlockDimForPreLaunch(hostCtrlFlowCache, devProg, config);
    uint64_t contextWorkspaceAddr = hostCtrlFlowCache->contextWorkspaceAddr;
    hostCtrlFlowCache->IncastOutcastAddrReloc(contextWorkspaceAddr, 0, nullptr);
    hostCtrlFlowCache->BuildIncastOutcastRelocTable();
    hostCtrlFlowCache->CalcUsedCacheSize();
    hostCtrlFlowCache->RuntimeAddrRelocWorkspace(contextWorkspaceAddr, 0, nullptr, nullptr, nullptr,
                                                 devProg->GetParallelism());
    hostCtrlFlowCache->RuntimeAddrRelocProgram(reinterpret_cast<uint64_t>(devProg), 0);
    hostCtrlFlowCache->TaskAddrRelocWorkspace(contextWorkspaceAddr, 0, nullptr);
    hostCtrlFlowCache->TaskAddrRelocProgramAndCtrlCache(reinterpret_cast<uint64_t>(devProg),
                                                        reinterpret_cast<uint64_t>(hostCtrlFlowCache), 0, 0);
    hostCtrlFlowCache->RelocMetaCache(reinterpret_cast<uint64_t>(hostCtrlFlowCache), 0);
    devProg->ctrlFlowCacheAnchor = nullptr;
    devProg->ctrlFlowCacheSize = hostCtrlFlowCache->usedCacheSize;
    devProg->ResetFromLaunch();
    if (outCtrlFlowCache) {
        *outCtrlFlowCache = hostCtrlFlowCache;
    }

    // Hand the freshly relocated base-0 cache to the pack hook. Packing itself is decided at the launch choke
    // point (DeviceLauncher::PrepareLaunch), not here -- see KernelBundlePackHook.
    bundle::KernelBundlePackHook::Instance().StashCtrlFlowCache(
        dynAttr.get(), reinterpret_cast<const uint8_t*>(hostCtrlFlowCache), hostCtrlFlowCache->usedCacheSize);
    return rc;
}

int EmulationLauncher::BuildControlFlowCache(Function* function, DevControlFlowCache** outCtrlFlowCache,
                                             EmulationMemoryUtils& memUtils, const DeviceLauncherConfig& config)
{
    auto& inputDataList = ProgramData::GetInstance().GetInputDataList();
    auto& outputDataList = ProgramData::GetInstance().GetOutputDataList();
    std::vector<DeviceTensorData> inputDeviceDataList;
    std::vector<DeviceTensorData> outputDeviceDataList;
    std::tie(inputDeviceDataList, outputDeviceDataList) = DeviceLauncher::BuildInputOutputFromHost(
        memUtils, inputDataList, outputDataList);
    return BuildControlFlowCacheWithEmulationTensorData(function, inputDeviceDataList, outputDeviceDataList, nullptr,
                                                        outCtrlFlowCache, memUtils, config);
}

int EmulationLauncher::BuildControlFlowCache(Function* function, EmulationMemoryUtils& memUtils,
                                             const std::vector<DeviceTensorData>& inputList,
                                             const std::vector<DeviceTensorData>& outputList,
                                             DevControlFlowCache** outCtrlFlowCache, const DeviceLauncherConfig& config)
{
    auto getShapeString = [&](const std::vector<DeviceTensorData>& inputTensor) {
        std::stringstream ss;
        for (size_t i = 0; i < inputTensor.size(); ++i) {
            const auto& shape = inputTensor[i].GetShape();

            ss << "[";
            for (size_t j = 0; j < shape.size(); ++j) {
                ss << shape[j];
                if (j != shape.size() - 1) {
                    ss << ",";
                }
            }
            ss << "]";

            if (i != inputTensor.size() - 1) {
                ss << " ";
            }
        }
        return ss.str();
    };
    MACHINE_LOGI("!!! Emulation ControlFlowCache shape {%s}\n", getShapeString(inputList).c_str());

    /* python front end use inputs/output as unified tensors, outputList is always null */
    if (inputList.size() == 0 && outputList.size() == 0) {
        return BuildControlFlowCache(function, outCtrlFlowCache, memUtils, config);
    } else {
        std::vector<DeviceTensorData> inputDeviceDataList;
        std::vector<DeviceTensorData> outputDeviceDataList;

        auto devIdxToCpuAddr = BuildDevIdxToCpuAddrMap(function->GetDyndevAttribute(), inputList);

        uintptr_t index = 0;
        for (size_t i = 0; i < inputList.size(); i++) {
            uintptr_t addr = CONTROL_FLOW_CACHE_BASE_ADDR + index * CONTROL_FLOW_CACHE_TENSOR_SIZE;
            auto it = devIdxToCpuAddr.find(i);
            if (it != devIdxToCpuAddr.end()) {
                addr = it->second;
            }
            inputDeviceDataList.emplace_back(inputList[i].GetDataType(), addr, inputList[i].GetShape());
            index++;
        }
        for (auto& output : outputList) {
            outputDeviceDataList.emplace_back(output.GetDataType(),
                                              CONTROL_FLOW_CACHE_BASE_ADDR + index * CONTROL_FLOW_CACHE_TENSOR_SIZE,
                                              output.GetShape());
            index++;
        }
        return BuildControlFlowCacheWithEmulationTensorData(function, inputDeviceDataList, outputDeviceDataList,
                                                            nullptr, outCtrlFlowCache, memUtils, config);
    }
}

static std::vector<DeviceTensorData> toHostTensorData(const std::vector<DeviceTensorData>& devDataList, bool isInput)
{
    std::vector<DeviceTensorData> hostDataList;
    for (auto& devData : devDataList) {
        auto size = devData.GetDataSize();
        if (size == 0) {
            // Optional placeholder (e.g. empty dequant_scale); no device buffer — skip D2H memcpy.
            MACHINE_LOGW("toHostTensorData: skip empty tensor, isInput=%d", isInput);
            hostDataList.emplace_back(devData.GetDataType(), nullptr, devData.GetShape());
            continue;
        }
        void* dataPtr = malloc(size);
        if (isInput) {
            RuntimeMemcpy(dataPtr, size, devData.GetAddr(), size, RtMemcpyKind::DEVICE_TO_HOST);
        }
        hostDataList.emplace_back(devData.GetDataType(), dataPtr, devData.GetShape());
    }
    return hostDataList;
}

static void FreeHostTensorData(const std::vector<DeviceTensorData>& hostDataList)
{
    for (auto& hostData : hostDataList) {
        free(hostData.GetAddr());
    }
}

int EmulationLauncher::EmulationLaunchDeviceTensorData(Function* function,
                                                       const std::vector<DeviceTensorData>& inDevList,
                                                       const std::vector<DeviceTensorData>& outDevList,
                                                       const DeviceLauncherConfig& config,
                                                       DevControlFlowCache* ctrlCache)
{
    EmulationMemoryUtils memUtils;
    ExchangeCaptureModeRelax();
    auto inList = toHostTensorData(inDevList, true);
    auto outList = toHostTensorData(outDevList, false);
    ExchangeCaptureModeGlobal();
    int rc = EmulationLaunchOnceWithHostTensorData(function, inList, outList, ctrlCache, memUtils, config);
    FreeHostTensorData(inList);
    FreeHostTensorData(outList);
    return rc;
}
} // namespace npu::tile_fwk::dynamic
