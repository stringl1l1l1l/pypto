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
 * \file kernel_binary.cpp
 * \brief Implementation of KernelBinary class.
 */

#include "machine/runtime/runner/kernel_binary.h"

#include <cstdlib>
#include <mutex>
#include <sstream>

#include "tilefwk/platform.h"
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_code.h"
#include "adapter/api/msprof_api.h"
#include "adapter/api/acl_api.h"
#include "adapter/api/runtime_api.h"
#include "machine/utils/dynamic/rebuildable_workspace_desc.h"
#include "interface/utils/common.h"
#include "interface/configs/config_manager.h"
#include "interface/machine/host/perf_analysis.h"
#include "machine/runtime/runner/host_prof.h"
#include "machine/runtime/runner/device_perf.h"
#include "machine/runtime/runner/runtime_agent.h"
#include "machine/runtime/runner/device_error_tracking.h"
#include "machine/runtime/runner/device_exception_dump.h"
#include "machine/runtime/runner/device_dfx.h"
#include "machine/runtime/runner/pmu_common.h"
#include "machine/device/dump/dump_device_memory.h"
#include "machine/device/dynamic/device_common.h"
#include "machine/device/tilefwk/aicore_print_base.h"
#include "machine/runtime/launcher/cell_match_dynamic.h"
#include "machine/runtime/launcher/device_launcher.h"
#include "machine/runtime/launcher/device_launcher_binding.h"
#include "machine/runtime/launcher/emulation_launcher.h"
#include "machine/runtime/memory_utils/device_memory_utils.h"
#include "machine/runtime/memory_utils/eslmodel_memory_utils.h"
#include "machine/runtime/memory_utils/memory_pool.h"
#include "machine/runtime/runner/runtime_utils.h"
#include "machine/utils/dynamic/dev_encode_program.h"
#include "machine/utils/dynamic/dev_encode_program_ctrlflow_cache.h"
#include "machine/utils/dynamic/dev_start_args.h"
#include "machine/utils/dynamic/device_task.h"
#include "interface/configs/config_manager_ng.h"
#include "utils/file_utils.h"
#include "securec.h"

namespace npu::tile_fwk::dynamic {
namespace {
constexpr uint32_t SUB_CORE = 3;
constexpr uint32_t AIV_PER_AICORE = 2;
constexpr uint32_t AICPU_NUM_OF_RUN_AICPU_TASKS = 1;
// AiCpu backend .so bytes from a kernel bundle; when non-empty, InitAiCpuSoBin uses these instead of the
// on-disk copy. Set before init runs, so no locking is needed.
std::vector<uint8_t> g_aicpuSoOverride;
DeviceArgs g_deviceArgs;
std::once_flag g_initOnce;
} // namespace

KernelBinary::KernelBinary(std::shared_ptr<Function> func, std::vector<std::shared_ptr<Function>> pinnedGraph)
    : dynFunc(std::move(func)), pinnedGraph_(std::move(pinnedGraph))
{
    dynAttr = dynFunc->GetDyndevAttribute().get();
    devProg = (DevAscendProgram*)dynAttr->devProgBinary.data();
    kernelBin = RegisterKernelBinary(dynAttr->kernelBinary);
    workspaceSize = devProg->memBudget.Total();
    if (!IsLiteNPU(Platform::Instance().GetSoc().GetNPUArch())) { // litenpu does not need init device args
        InitCachedArgs();
        InitLaunchArgs();
        InitDeviceArgs();
        auto aicpuArgs = (AiCpuArgs*)aicpuArgBuf.data();
        DeviceLauncher::FillSwimLaneEnableInfo(toSubMachineConfig_);
        if (config::GetRuntimeOption<int64_t>(CFG_RUN_MODE) == CFG_RUN_MODE_SIM) {
            EslModelMemoryUtils eslMemoryUtils{true, true};
            DeviceLauncher::FillDeviceKernelArgs(eslMemoryUtils, dynAttr->devProgBinary, aicpuArgs->kArgs,
                                                 dynAttr->commGroupNames);
        } else {
            DeviceMemoryUtils deviceMemoryUtils;
            DeviceLauncher::FillDeviceKernelArgs(deviceMemoryUtils, dynAttr->devProgBinary, aicpuArgs->kArgs,
                                                 dynAttr->commGroupNames);
        }
    }
    kernelName_ = "PyPTO_" + dynFunc->GetOriginalRawName();
}

KernelBinary::~KernelBinary()
{
    if (runtimeDynamicCellMatchOwned_ && runtimeDynamicCellMatchAddr_ != 0) {
        DevMemoryPool::Instance().FreeDevAddr(reinterpret_cast<uint8_t*>(runtimeDynamicCellMatchAddr_));
    }
    if (runtimeDynamicCellMatchHostOwned_ && runtimeDynamicCellMatchHostAddr_ != 0) {
        std::free(reinterpret_cast<void*>(runtimeDynamicCellMatchHostAddr_));
    }
    UnregisterKernelBinary(kernelBin);
    FreeAndClearValueDependCache();
    for (auto& cache : originShapeCaches) {
        DeviceLauncher::FreeControlFlowCache(cache.devCache);
    }
    for (auto& cache : inferShapeCaches) {
        DeviceLauncher::FreeControlFlowCache(cache.devCache);
    }
}

ToSubMachineConfig& KernelBinary::GetMachineConfig() { return toSubMachineConfig_; }

bool KernelBinary::BeginRingLaunch(uintptr_t epoch, bool isCapture, int64_t& outSequence)
{
    BeginRingEpoch(epoch, isCapture);
    if (!EnsureRingEventsCreated(isCapture)) {
        return false;
    }
    outSequence = ++ringSequence_;
    return true;
}

void KernelBinary::BeginRingEpoch(uintptr_t epoch, bool isCapture)
{
    if (ringEpochInited_ && ringEpoch_ == epoch) {
        return;
    }
    if (isCapture) {
        // Previous graph keeps its events for replay; this capture gets a new pool.
        graphEventsCreated_ = false;
    }
    ringSequence_ = 0;
    ringEpoch_ = epoch;
    ringEpochInited_ = true;
}

bool KernelBinary::EnsureRingEventsCreated(bool isCapture)
{
    AclRtEvent* events = isCapture ? graphEvents_ : eagerEvents_;
    bool& created = isCapture ? graphEventsCreated_ : eagerEventsCreated_;
    if (created) {
        return true;
    }
    for (int64_t i = 0; i < kRingPingPongCount; ++i) {
        if (AclRtCreateEventExWithFlag(&events[i], ACL_EVENT_SYNC) < 0) {
            MACHINE_LOGE(RtErr::RT_EVENT_FAILED, "AclRtCreateEvent failed for ping-pong event %ld.", i);
            return false;
        }
    }
    created = true;
    return true;
}

uint8_t* KernelBinary::FindCtrlFlowCache(std::vector<std::vector<int64_t>>& inputs, bool isOriginShape)
{
    int64_t inHash = ControlFlowCache::Hash(inputs);
    auto& caches = isOriginShape ? originShapeCaches : inferShapeCaches;
    for (auto& cache : caches) {
        if (cache.hash == inHash) {
            return cache.devCache;
        }
    }
    return nullptr;
}

uint8_t* KernelBinary::FindCtrlFlowCache(std::vector<DeviceTensorData>& inputs, bool isOriginShape)
{
    int64_t inHash = ControlFlowCache::Hash(inputs);
    auto& caches = isOriginShape ? originShapeCaches : inferShapeCaches;
    for (auto& cache : caches) {
        if (cache.hash == inHash) {
            return cache.devCache;
        }
    }
    return nullptr;
}

uint8_t* KernelBinary::BuildControlFlowCache(std::vector<DeviceTensorData>& inputs, bool isOriginShape,
                                             bool storeToCache)
{
    DeviceLauncherConfig config;
    DeviceLauncher::DeviceLauncherConfigFillDeviceInfo(config);
    DevControlFlowCache* ctrlCache = nullptr;
    devProg->ctrlFlowCacheSize = DEFAULT_STITCH_CFGCACHE_SIZE;
    config.isCacheOriginShape = isOriginShape;
    EmulationMemoryUtils memUtils;
    size_t inputCount = dynAttr->startArgsInputTensorList.size();
    std::vector<DeviceTensorData> inputList(inputs.begin(), inputs.begin() + inputCount);
    std::vector<DeviceTensorData> outputList(inputs.begin() + inputCount, inputs.end());
    int ret = EmulationLauncher::BuildControlFlowCache(dynFunc.get(), memUtils, inputList, outputList, &ctrlCache,
                                                       config);
    if (ret != 0) {
        COMPILER_LOGE(CtrlErr::DEVICE_TASK_BUILD_FAILED, "control flow cache failed %d", ret);
        return nullptr;
    }
    // Emulation returns success + nullptr when DevProg.disableCtrlFlowCache is set.
    if (ctrlCache == nullptr) {
        return nullptr;
    }

    uint8_t* devCache = DeviceLauncher::CopyControlFlowCache(ctrlCache);
    COMPILER_LOGD("control flow cache: %p", devCache);
    if (storeToCache) {
        if (isOriginShape) {
            originShapeCaches.emplace_back(inputs, devCache);
        } else {
            inferShapeCaches.emplace_back(inputs, devCache);
        }
    }
    return devCache;
}

int64_t KernelBinary::GetWorkspaceSize(const std::vector<DeviceTensorData>& tensors)
{
    static const std::vector<DeviceTensorData> kEmptyOutputs;
    Evaluator eval{dynAttr->inputSymbolDict, &tensors, &kEmptyOutputs};
    dynamicCellMatchDescPatches_ = PrepareDynamicCellMatchDescPatches(*dynAttr, eval);
    PatchHostDynamicCellMatchTableDesc(devProg, dynamicCellMatchDescPatches_);
    if (dynAttr->maxDynamicAssembleOutcastMem.IsValid() || dynAttr->maxDynamicCellMatchTableMem.IsValid()) {
        if (dynAttr->maxDynamicAssembleOutcastMem.IsValid()) {
            devProg->memBudget.tensor.maxDynamicAssembleOutcastMem = eval.Evaluate(
                dynAttr->maxDynamicAssembleOutcastMem);
        }
        if (dynAttr->maxDynamicCellMatchTableMem.IsValid()) {
            devProg->memBudget.metadata.maxDynamicCellMatchTableMem = eval.Evaluate(
                dynAttr->maxDynamicCellMatchTableMem);
            uint64_t totalDynamicCellMatchSlotNum = devProg->memBudget.metadata.dynamicCellMatchSlotNum;
            devProg->memBudget.metadata.dynamicCellMatch = totalDynamicCellMatchSlotNum *
                                                           devProg->memBudget.metadata.maxDynamicCellMatchTableMem;
            ValidateDynamicCellMatchTableMemBudget(*dynAttr, devProg);
        }
        if (devProg->memBudget.metadata.dynamicCellMatch != lastPreparedDynamicCellMatchBytes_) {
            RefreshRuntimeDynamicCellMatchMeta(devProg->memBudget.metadata.dynamicCellMatch);
            lastPreparedDynamicCellMatchBytes_ = devProg->memBudget.metadata.dynamicCellMatch;
            devProg->devArgs.dynamicCellMatchAddr = runtimeDynamicCellMatchAddr_;
            devProg->devArgs.dynamicCellMatchCapacity = runtimeDynamicCellMatchCapacity_;
        }
        PatchHostDynamicCellMatchAddr(devProg);
        workspaceSize = devProg->memBudget.Total();

        // check and pretty print total workspace consumption
        auto* wsChecker = RebuildableAttributeManager::GetInstance().GetAttr<RebuildableWorkspaceDesc>(dynFunc.get());
        MACHINE_LOGI_FULL("Memory Consumption: size=%ld bytes\n%s\n", workspaceSize,
                          wsChecker
                              ->PrettyDumpSize(devProg->memBudget.tensor.maxDynamicAssembleOutcastMem,
                                               devProg->memBudget.debug.Total())
                              .c_str());
        MACHINE_ASSERT(uint64_t(workspaceSize) ==
                       wsChecker->GetSizeForCheckOnly(devProg->memBudget.tensor.maxDynamicAssembleOutcastMem,
                                                      devProg->memBudget.debug.Total()));
        mem_dump::DumpMemoryOverview(*devProg, dynFunc.get(), static_cast<uint64_t>(workspaceSize));
    }
    return workspaceSize;
}

std::pair<AiCpuArgs*, int64_t> KernelBinary::BuildKernelArgs(const std::vector<DeviceTensorData>& tensors)
{
    auto& disableL2List = dynAttr->disableL2List;
    auto aicpuArgs = (AiCpuArgs*)aicpuArgBuf.data();
    int64_t* inputp = (int64_t*)(aicpuArgs + 1);
    auto tensorData = (DevTensorData*)(inputp + 2);
    const int64_t totalTensorCount = inputp[0] + inputp[1];
    MACHINE_ASSERT((int64_t)tensors.size() == totalTensorCount) << "mismatch tensor size";
    for (size_t i = 0; i < (size_t)totalTensorCount; ++i) {
        auto& t = tensors[i];
        auto addr = (uint64_t)t.GetAddr();
        if (unlikely(addr && disableL2List.size() && disableL2List[i])) {
            COMPILER_LOGI("mismatch tensor addr");
            addr += l2Offset;
        }
        tensorData->address = addr;
        tensorData->dataType = tensors[i].GetDataType();
        auto& shape = t.GetShape();
        tensorData->shape.dimSize = shape.size();
        for (int j = 0; j < tensorData->shape.dimSize; ++j) {
            tensorData->shape.dim[j] = shape[j];
        }
        tensorData++;
    }

    WriteDynamicCellMatchStridePatchesToLaunchArgs(inputp, dynamicCellMatchDescPatches_);

    return {aicpuArgs, aicpuArgBuf.size() * sizeof(int64_t)};
}

bool KernelBinary::CheckArgs(const std::vector<DeviceTensorData>& tensors) const
{
    if (tensors.size() != argTypes.size()) {
        return false;
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
        auto& t = tensors[i];
        auto& type = argTypes[i];
        if (unlikely(t.GetDataType() != type.GetDataType())) {
            return false;
        }
        if (unlikely(t.Format() != type.Format())) {
            return false;
        }
        auto& shape1 = type.GetShape();
        auto& shape2 = t.GetShape();
        if (unlikely(shape1.size() != shape2.size())) {
            return false;
        }
        for (size_t j = 0; j < shape1.size(); ++j) {
            if (unlikely((shape1[j] != -1) && (shape1[j] != shape2[j]))) {
                return false;
            }
        }
    }
    return true;
}

void* KernelBinary::GetKernelBin() { return kernelBin; }

Function* KernelBinary::GetFunction() { return dynFunc.get(); }

const std::string& KernelBinary::GetKernelname() const { return kernelName_; }

std::vector<size_t> KernelBinary::emptyIndices_;

bool KernelBinary::DisableHostCtrlFlowCacheBuild() const
{
    if (devProg == nullptr) {
        return true;
    }
    return devProg->disableCtrlFlowCache != 0;
}

const std::vector<size_t>& KernelBinary::GetValueDependInputIndices() const
{
    return devProg != nullptr ? devProg->valueDependInputIndices : emptyIndices_;
}

void KernelBinary::FreeAndClearValueDependCache()
{
    if (valueDependDevCache_ != nullptr) {
        DeviceLauncher::FreeControlFlowCache(valueDependDevCache_);
        valueDependDevCache_ = nullptr;
    }
}

uint64_t KernelBinary::GetMaxDynamicAssembleOutcastMem() const
{
    return devProg->memBudget.tensor.maxDynamicAssembleOutcastMem;
}

uint64_t KernelBinary::GetMaxDynamicCellMatchTableMem() const
{
    return devProg->memBudget.metadata.maxDynamicCellMatchTableMem;
}

uint64_t KernelBinary::GetRuntimeDynamicCellMatchAddr() const { return runtimeDynamicCellMatchAddr_; }

uint64_t KernelBinary::GetRuntimeDynamicCellMatchCapacity() const { return runtimeDynamicCellMatchCapacity_; }

void KernelBinary::ResetRuntimeDynamicCellMatchPool(bool useHostMirror) const
{
    if (runtimeDynamicCellMatchCapacity_ == 0) {
        return;
    }
    if (useHostMirror) {
        if (runtimeDynamicCellMatchHostAddr_ == 0) {
            return;
        }
        // Host mirror: fill each uint64 with AICORE_TASK_INIT (not byte 0xFF).
        ResetRuntimeDynamicCellMatchPoolHost(runtimeDynamicCellMatchHostAddr_, runtimeDynamicCellMatchCapacity_, false);
        return;
    }
    if (runtimeDynamicCellMatchAddr_ == 0) {
        return;
    }
    ResetRuntimeDynamicCellMatchPoolHost(runtimeDynamicCellMatchAddr_, runtimeDynamicCellMatchCapacity_, true);
}

void KernelBinary::SetSyncMode(uint8_t syncModel) { scheSyncModel_ = syncModel; }

uint8_t KernelBinary::GetSyncMode() { return scheSyncModel_; }

void KernelBinary::PatchHostDynamicCellMatchAddr(DevAscendProgram* hostProg)
{
    if (hostProg == nullptr) {
        return;
    }
    hostProg->devArgs.dynamicCellMatchAddr = runtimeDynamicCellMatchHostAddr_;
    hostProg->devArgs.dynamicCellMatchCapacity = runtimeDynamicCellMatchCapacity_;
}

void KernelBinary::InitCachedArgs()
{
    auto argNum = dynAttr->startArgsInputLogicalTensorList.size() + dynAttr->startArgsOutputLogicalTensorList.size();
    const uint64_t maxPatchCount = dynAttr->dynamicCellMatchLaunchMetaList.size();
    auto argSize = sizeof(AiCpuArgs) + 2 * sizeof(int64_t) + argNum * sizeof(DevTensorData) + sizeof(uint64_t) +
                   maxPatchCount * sizeof(DevDynamicCellMatchStridePatch);
    MACHINE_ASSERT(argSize % 0x8 == 0);
    aicpuArgBuf.resize(argSize / 0x8);

    auto aicpuArgs = new (aicpuArgBuf.data()) AiCpuArgs();
    aicpuArgs->kArgs.inputs = nullptr;
    aicpuArgs->kArgs.outputs = nullptr;

    int64_t* inputp = (int64_t*)(aicpuArgs + 1);
    inputp[0] = dynAttr->startArgsInputLogicalTensorList.size();
    inputp[1] = dynAttr->startArgsOutputLogicalTensorList.size();
    const uint64_t tensorCount = static_cast<uint64_t>(inputp[0]) + static_cast<uint64_t>(inputp[1]);
    *reinterpret_cast<uint64_t*>(reinterpret_cast<DevTensorData*>(inputp + 2) + tensorCount) = 0;

    l2Offset = GetRuntimeL2Offset();

    for (auto& t : dynAttr->startArgsInputLogicalTensorList) {
        if (t == nullptr) {
            continue;
        }
        argTypes.emplace_back(t->Datatype(), nullptr, t->GetShape(), t->Format());
    }
    for (auto& t : dynAttr->startArgsOutputLogicalTensorList) {
        if (t == nullptr) {
            continue;
        }
        argTypes.emplace_back(t->Datatype(), nullptr, t->GetShape(), t->Format());
    }
}

void KernelBinary::InitLaunchArgs()
{
    aicpuLaunchDesc_ = {};
    aicpuLaunchDesc_.hostInputNum = 1;
    hostInfo_.addrOffset = offsetof(AiCpuArgs, kArgs.inputs);
    hostInfo_.dataOffset = sizeof(AiCpuArgs);
    aicpuLaunchDesc_.hostInputs = &hostInfo_;
    aicpuLaunchDesc_.timeout = AICPU_EXECUTE_TIMEOUT;
    memset_s(&rtAicoreArgs_, sizeof(RtArgsEx), 0, sizeof(RtArgsEx));
    kernelArgs_.resize(0x7, nullptr);
    rtAicoreArgs_.args = kernelArgs_.data();
    rtAicoreArgs_.argsSize = kernelArgs_.size() * sizeof(void*);

    memset_s(&rtTaskCfg_, sizeof(RtTaskCfgInfo), 0, sizeof(RtTaskCfgInfo));
    rtTaskCfg_.schemMode = static_cast<uint8_t>(RtSchemModeType::BATCH);
    rtTaskCfg_.localMemorySize = GetAicoreLocalMemorySize(static_cast<NPUArch>(devProg->devArgs.archInfo),
                                                          devProg->requiresSimt);
}

void KernelBinary::RefreshRuntimeDynamicCellMatchMeta(uint64_t needBytes)
{
    if (needBytes == 0) {
        if (runtimeDynamicCellMatchOwned_ && runtimeDynamicCellMatchAddr_ != 0) {
            DevMemoryPool::Instance().FreeDevAddr(reinterpret_cast<uint8_t*>(runtimeDynamicCellMatchAddr_));
        }
        if (runtimeDynamicCellMatchHostOwned_ && runtimeDynamicCellMatchHostAddr_ != 0) {
            std::free(reinterpret_cast<void*>(runtimeDynamicCellMatchHostAddr_));
        }
        runtimeDynamicCellMatchAddr_ = 0;
        runtimeDynamicCellMatchHostAddr_ = 0;
        runtimeDynamicCellMatchCapacity_ = 0;
        runtimeDynamicCellMatchOwned_ = false;
        runtimeDynamicCellMatchHostOwned_ = false;
        return;
    }
    if (runtimeDynamicCellMatchAddr_ != 0 && runtimeDynamicCellMatchHostAddr_ != 0 &&
        runtimeDynamicCellMatchCapacity_ >= needBytes) {
        return;
    }
    uint64_t oldAddr = runtimeDynamicCellMatchAddr_;
    uint64_t oldHostAddr = runtimeDynamicCellMatchHostAddr_;
    bool oldOwned = runtimeDynamicCellMatchOwned_;
    bool oldHostOwned = runtimeDynamicCellMatchHostOwned_;
    DeviceMemoryUtils deviceMemoryUtils;
    auto* newPtr = deviceMemoryUtils.AllocDev(needBytes, nullptr);
    if (newPtr == nullptr) {
        ASSERT(DevCommonErr::ALLOC_FAILED, false) << "alloc dynamic cell match meta failed, needBytes=" << needBytes;
        return;
    }
    auto* newHostPtr = static_cast<uint8_t*>(std::malloc(static_cast<size_t>(needBytes)));
    if (newHostPtr == nullptr) {
        DevMemoryPool::Instance().FreeDevAddr(newPtr);
        ASSERT(DevCommonErr::MALLOC_FAILED, false)
            << "alloc host dynamic cell match meta failed, needBytes=" << needBytes;
        return;
    }
    runtimeDynamicCellMatchAddr_ = reinterpret_cast<uint64_t>(newPtr);
    runtimeDynamicCellMatchHostAddr_ = reinterpret_cast<uint64_t>(newHostPtr);
    runtimeDynamicCellMatchCapacity_ = needBytes;
    runtimeDynamicCellMatchOwned_ = true;
    runtimeDynamicCellMatchHostOwned_ = true;
    ResetRuntimeDynamicCellMatchPool(true);
    ResetRuntimeDynamicCellMatchPool(false);
    if (oldOwned && oldAddr != 0) {
        DevMemoryPool::Instance().FreeDevAddr(reinterpret_cast<uint8_t*>(oldAddr));
    }
    if (oldHostOwned && oldHostAddr != 0) {
        std::free(reinterpret_cast<void*>(oldHostAddr));
    }
}

void KernelBinary::SetAiCpuSoOverride(std::vector<uint8_t> soBytes) { g_aicpuSoOverride = std::move(soBytes); }

void KernelBinary::InitAiCpuSoBin(DeviceArgs& devArgs)
{
    std::vector<uint8_t> buffer;
    std::string source;
    if (!g_aicpuSoOverride.empty()) {
        buffer = g_aicpuSoOverride; // bundle-supplied .so takes precedence over the on-disk copy
        source = "kernel-bundle AICPU_SO";
    } else {
        std::string fileName = GetPyptoLibPath() + "/libtilefwk_backend_server.so";
        source = fileName;
        buffer = ReadFile(fileName);
    }
    if (buffer.empty()) {
        MACHINE_LOGE(DevCommonErr::FILE_ERROR,
                     "Read bin from tilefwk_backend_server.so failed, please check the so[%s]", source.c_str());
        return;
    }
    void* devBufferPtr = CopyDataToDevice(buffer.data(), buffer.size());
    if (devBufferPtr == nullptr) {
        MACHINE_LOGE(DevCommonErr::MEMCPY_FAILED, "Failed to copy buffer of [%s] to device.", source.c_str());
        return;
    }
    devArgs.aicpuSoBin = reinterpret_cast<uint64_t>(devBufferPtr);
    devArgs.aicpuSoLen = buffer.size();
    MACHINE_LOGI("[aicpu-so] init backend server .so from %s, len=%zu", source.c_str(), buffer.size());
    HOST_PERF_TRACE(TracePhase::RunDevKernelInitAicpuSo);
}

void KernelBinary::GetAicoreRegs(const ArchInfo archInfo, std::vector<int64_t>& regs, std::vector<int64_t>& regsPmu)
{
    if (archInfo == ArchInfo::DAV_3510) {
        RuntimeAgent::GetAicoreRegInfoForDAV3510(regs, regsPmu);
        return;
    }
    if (archInfo == ArchInfo::DAV_2201) {
        std::vector<int64_t> aiv;
        std::vector<int64_t> aic;
        if (RuntimeAgent::GetAgent().GetAicoreRegInfo(aic, aiv, ADDR_MAP_TYPE_REG_AIC_CTRL) != 0) {
            return;
        }
        regs.insert(regs.end(), aic.begin(), aic.end());
        regs.insert(regs.end(), aiv.begin(), aiv.end());

        std::vector<int64_t> aivPmu;
        std::vector<int64_t> aicPmu;
        if (RuntimeAgent::GetAgent().GetAicoreRegInfo(aicPmu, aivPmu, ADDR_MAP_TYPE_REG_AIC_PMU_CTRL) != 0) {
            return;
        }
        regsPmu.insert(regsPmu.end(), aicPmu.begin(), aicPmu.end());
        regsPmu.insert(regsPmu.end(), aivPmu.begin(), aivPmu.end());
    }
}

int KernelBinary::InitDeviceArgsCore(DeviceArgs& args)
{
    std::vector<int64_t> regs;
    std::vector<int64_t> regsPmu;
    GetAicoreRegs(args.archInfo, regs, regsPmu);

    args.nrAic = regs.size() / SUB_CORE;
    args.nrAiv = args.nrAic * AIV_PER_AICORE;
    uint64_t nrCore = regs.size() + AICPU_NUM_OF_RUN_AICPU_TASKS;
    args.sharedBuffer = reinterpret_cast<uint64_t>(DevAlloc(nrCore * SHARED_BUFFER_SIZE));
    args.corePmuAddr = reinterpret_cast<uint64_t>(DevAlloc(nrCore * PMU_BUFFER_SIZE));
    if (args.sharedBuffer == 0 || args.corePmuAddr == 0) {
        MACHINE_LOGE(DevCommonErr::ALLOC_FAILED, "Fail alloc sharedBuffer[%lu] or corePmuAddr[%lu].", args.sharedBuffer,
                     args.corePmuAddr);
        return static_cast<int>(DevCommonErr::ALLOC_FAILED);
    }

    // core reg
    size_t regSize = regs.size() * sizeof(uint64_t);
    args.coreRegAddr = reinterpret_cast<uint64_t>(CopyDataToDevice(regs.data(), regSize));
    if (args.coreRegAddr == 0) {
        MACHINE_LOGE(DevCommonErr::MEMCPY_FAILED, "Fail to copy aicore reg data from host to device.");
        return static_cast<int>(DevCommonErr::MEMCPY_FAILED);
    }

    // core reg pmu
    args.corePmuRegAddr = reinterpret_cast<uint64_t>(CopyDataToDevice(regsPmu.data(), regSize));
    if (args.corePmuRegAddr == 0) {
        MACHINE_LOGE(DevCommonErr::MEMCPY_FAILED, "Fail to copy aicore pmu reg data from host to device.");
        return static_cast<int>(DevCommonErr::MEMCPY_FAILED);
    }
    MACHINE_LOGI("Dev args :aic %u aiv %u, sharedBuffer %lx coreRegAddr %lx corePmuRegAddr %lx", args.nrAic, args.nrAiv,
                 args.sharedBuffer, args.coreRegAddr, args.corePmuRegAddr);

    args.taskWastTime = reinterpret_cast<uint64_t>(DevAlloc(sizeof(uint64_t)));
    size_t shmSize = sizeof(dynamic::RuntimeDataRingBufferHead) + dynamic::DEVICE_SHM_SIZE +
                     dynamic::DEVICE_TASK_QUEUE_SIZE * args.nrAicpu;
    args.runtimeDataRingBufferAddr = reinterpret_cast<uint64_t>(DevAlloc(shmSize));

    // pmu evt info
    std::vector<int64_t> pmuEvtType;
    PmuCommon::InitPmuEventType(args.archInfo, pmuEvtType);
    args.pmuEventAddr = reinterpret_cast<uint64_t>(
        CopyDataToDevice(pmuEvtType.data(), pmuEvtType.size() * sizeof(int64_t)));
    if (args.pmuEventAddr == 0) {
        MACHINE_LOGE(DevCommonErr::MEMCPY_FAILED, "Fail to copy pmu evt type from host to device.");
        return static_cast<int>(DevCommonErr::MEMCPY_FAILED);
    }

    if (!DeviceDfx::GetInstance().Init(args)) {
        MACHINE_LOGE(DevCommonErr::INIT_FAILED, "Device dfx info init not success.");
        return static_cast<int>(DevCommonErr::INIT_FAILED);
    }
    return 0;
}

int KernelBinary::InitDeviceArgs(DeviceArgs& args)
{
    memset_s(&args, sizeof(args), 0, sizeof(args));
    args.deviceId = GetLogDeviceId();
    args.archInfo = static_cast<ArchInfo>(Platform::Instance().GetSoc().GetNPUArch());
    uint32_t aicpuNum = args.archInfo == ArchInfo::DAV_3510 ? dynamic::DEVICE_MAX_AICPU_NUM : 5;
    uint32_t maxAicpuNum = static_cast<uint32_t>(Platform::Instance().GetSoc().GetAICPUNum());
    args.nrValidAic = GetCfgBlockdim();
    args.nrAicpu = std::min(aicpuNum, maxAicpuNum);
    args.scheCpuNum = dynamic::CalcSchAicpuNumByBlockDim(args.nrValidAic, args.nrAicpu, args.archInfo);
    MACHINE_LOGD("DevArgs: block dim[%u], aicpu num[%u], max aicpu num[%u], sche cpu num[%u].", args.nrValidAic,
                 args.nrAicpu, maxAicpuNum, args.scheCpuNum);

    InitAiCpuSoBin(args);

    return InitDeviceArgsCore(args);
}

void KernelBinary::InitDeviceArgs()
{
    std::call_once(g_initOnce, []() {
        HostProf::GetInstance().RegHostProf();
        InitializeErrorCallback();
        ASSERT(DevCommonErr::INIT_FAILED, InitDeviceArgs(g_deviceArgs) == 0);
        int64_t* devArgsAddr = static_cast<int64_t*>(CopyDataToDevice(&g_deviceArgs, sizeof(DeviceArgs)));
        if (devArgsAddr == nullptr) {
            MACHINE_LOGE(DevCommonErr::MEMCPY_FAILED, "Failed to copy args to device.");
            ASSERT(false);
        }
        if (config::GetRuntimeOption<int64_t>(CFG_RUN_MODE) != CFG_RUN_MODE_SIM) {
            if (LoadAicpuOp::GetInstance().GetBuiltInOpBinHandle(devArgsAddr) != 0) {
                MACHINE_LOGE(DevCommonErr::GET_HANDLE_FAILED, "Get builtInOp Funchandle failed\n");
                ASSERT(false);
            }
            DevicePerf::GetInstance().InitAndStartDumpThread(g_deviceArgs);
            AdumpRegExceptionDump();
        }
    });
}

void KernelBinary::InitMetaData(DeviceArgs& devArgs)
{
    InitDeviceArgs();
    devArgs.runtimeDataRingBufferAddr = g_deviceArgs.runtimeDataRingBufferAddr;
    devArgs.sharedBuffer = g_deviceArgs.sharedBuffer;
    devArgs.coreRegAddr = g_deviceArgs.coreRegAddr;
    devArgs.nrAic = g_deviceArgs.nrAic;
    devArgs.nrAiv = g_deviceArgs.nrAiv;
    devArgs.corePmuRegAddr = g_deviceArgs.corePmuRegAddr;
    devArgs.corePmuAddr = g_deviceArgs.corePmuAddr;
    devArgs.taskWastTime = g_deviceArgs.taskWastTime;
    devArgs.pmuEventAddr = g_deviceArgs.pmuEventAddr;
    devArgs.aicpuPerfAddr = g_deviceArgs.aicpuPerfAddr;
    devArgs.devDfxArgAddr = g_deviceArgs.devDfxArgAddr;
}

bool KernelBinary::GetEnableDumpDevPref()
{
    InitDeviceArgs();
    return g_deviceArgs.aicpuPerfAddr != 0;
}

} // namespace npu::tile_fwk::dynamic
