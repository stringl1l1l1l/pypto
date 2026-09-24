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
 * \file device_launcher.cpp
 * \brief
 */

#include "machine/runtime/launcher/device_launcher.h"

#include "tilefwk/aicore_print_base.h"
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_code.h"
#include "adapter/api/msprof_api.h"
#include "adapter/api/acl_api.h"
#include "adapter/api/adump_api.h"
#include "adapter/api/runtime_api.h"
#include "interface/utils/op_info_manager.h"
#include "interface/utils/common.h"
#include "interface/configs/config_manager.h"
#include "machine/runtime/memory_utils/eslmodel_memory_utils.h"
#include "interface/configs/config_manager_ng.h"
#include "machine/runtime/context/stream_context.h"
#include "machine/runtime/context/device_launcher_context.h"
#include "machine/runtime/runner/runtime_utils.h"
#include "machine/runtime/runner/device_dfx.h"
#include "machine/runtime/runner/kernel_binary.h"
#include "machine/runtime/launcher/device_launcher_driver_gate.h"
#include "machine/runtime/launcher/emulation_launcher.h"
#include "machine/runtime/launcher/aicore_model_launcher.h"
#include "machine/runtime/launcher/ctrl_flow_cache_manager.h"
#include "machine/runtime/bundle/pack/kernel_bundle_pack.h"
#include "machine/host/perf_analysis.h"
#include "interface/program/program.h"

namespace npu::tile_fwk::dynamic {
namespace {

bool IsEarlyLaunchActive(bool isCaptureMode, int launchEarlyMode)
{
    if (launchEarlyMode == 1) { // early launch in all modes (capture + merge/eager)
        return true;
    }
    if (launchEarlyMode == 0 && isCaptureMode) { // early launch only in capture mode
        return true;
    }
    return false;
}

bool IsRingEventSyncEnabled(int launchEarlyMode, bool ctrlFlowCacheReplay)
{
    if (launchEarlyMode == 2) {
        return false;
    }
    // Slot-reuse restore race only applies when host ctrl-flow cache replays on device.
    if (!ctrlFlowCacheReplay) {
        return false;
    }
    return true;
}

uintptr_t ResolveRingEpoch(AclRtStream aicoreStream, bool isCapture)
{
    if (!isCapture) {
        return 0;
    }
    AclMdlRI rtModel = nullptr;
    bool streamCapture = false;
    if (GetStreamCaptureInfo(aicoreStream, rtModel, streamCapture) && streamCapture && rtModel != nullptr) {
        return reinterpret_cast<uintptr_t>(rtModel);
    }
    return 1; // capturing, but model handle is unavailable
}

// Early-launch + CF-cache replay 时，ctrl 会复用 ring slot / 改 rawTensorAddr，而上一轮 aicore 可能还在读。
// 用每 KernelBinary 一对 ping-pong event：ctrl 在 launch 前 wait「waitDepth 轮之前」的 aicore record，
// 保证 slot 复用发生时，旧消费者已退场。waitDepth = min(ping-pong(=2), ringbuf)。
//
// ringbuf>=2（waitDepth=2）时序：第 3 轮 ctrl 等第 1 轮 aicore；第 2 轮仍可与第 1 轮重叠。
//
//   launch:     L1          L2          L3              L4
//   ctrl:    --[c1]------[c2]----[c3 wait E0]----[c4 wait E1]-->
//   aicore:  ----[a1]------[a2]------[a3]----------[a4]-------->
//   record:       E0^       E1^       E0^           E1^
//                          \__________/
//                     L3.ctrl 等到 a1 完成后才进
//
// ringbuf==1（waitDepth=1）时序：第 2 轮起就会复用唯一 slot，故 L2.ctrl 就要等 L1.aicore。
//
//   launch:     L1              L2
//   ctrl:    --[c1]----[c2 wait E0]-->
//   aicore:  ----[a1]------[a2]------>
//   record:       E0^       E1^
//
int RunRingEventWaitBeforeCtrl(int64_t sequence, RtStream ctrlStream, KernelBinary* kernel, bool isCapture)
{
    const int64_t ringBufSize = static_cast<int64_t>(DEFAULT_RUNTIME_DATA_RING_BUFFER_COUNT);
    const int64_t waitDepth = KernelBinary::kRingPingPongCount < ringBufSize ? KernelBinary::kRingPingPongCount :
                                                                               ringBufSize;
    if (sequence <= waitDepth) {
        return 0;
    }
    const int64_t idx = (sequence - waitDepth - 1) % KernelBinary::kRingPingPongCount;
    int rc = AclRtStreamWaitEvent(ctrlStream, kernel->RingEvent(isCapture, idx));
    if (rc < 0) {
        MACHINE_LOGE(RtErr::RT_EVENT_FAILED, "AclRtStreamWaitEvent (ctrl) failed %d\n", rc);
        return rc;
    }
    return 0;
}

int RunRingEventRecordAfterAicore(int64_t sequence, AclRtStream aicoreStream, KernelBinary* kernel, bool isCapture)
{
    const int64_t idx = (sequence - 1) % KernelBinary::kRingPingPongCount;
    int rc = AclRtRecordEvent(kernel->RingEvent(isCapture, idx), aicoreStream);
    if (rc < 0) {
        MACHINE_LOGE(RtErr::RT_EVENT_FAILED, "AclRtRecordEvent failed %d\n", rc);
        return rc;
    }
    return 0;
}

} // namespace

void DeviceLauncher::InitDevArgs(DeviceArgs& devArgs) { KernelBinary::InitMetaData(devArgs); }
bool DeviceLauncher::inited_ = false;
std::vector<uint8_t> DeviceLauncher::tensorInfo_(kDefaultTensorinfoSize);
std::unordered_map<Function*, DeviceLauncher::DeviceRunCacheInfo> DeviceLauncher::cacheInfoDict_;
std::atomic<int64_t> DeviceLauncher::sequence_(0);

void DeviceLauncher::CheckAscendDriverVersionOnboard() { AscendDriverVersionGate::EnsureDriverVersionForOnboardOnce(); }

int DeviceLauncher::SetCaptureStream(RtStream aicoreStream, RtStream aicpuStream, bool& isCapture)
{
    AclMdlRI rtModel = nullptr;

    if (!GetStreamCaptureInfo(aicoreStream, rtModel, isCapture)) {
        return -1;
    }
    DeviceLauncherContext::Get().SetCaptureMode(isCapture);

    if (isCapture) {
        if (rtModel == nullptr) {
            MACHINE_LOGE(DevCommonErr::NULLPTR, "rtModel is null!");
            return -1;
        }
        RtError ret = RuntimeStreamAddToModel(aicpuStream, rtModel);
        if (ret != 0) {
            MACHINE_LOGE(RtErr::RT_LAUNCH_FAILED, "RuntimeStreamAddToModel failed, return[%d]", ret);
            return -1;
        }
    }
    return 0;
}

int DeviceLauncher::RunWithProfile(RtStream aicoreStream, RtStream aicpuStream, bool isCapture)
{
    if (config::IsRuntimeDebugAllEnabled()) {
        if (isCapture) {
            MACHINE_LOGW("The swimlane function is not currently supported in CaptureMode. The contents of "
                         "tilefwk_L1_prof_data may be empty.");
            return 0;
        }
        int rc = DynamicLaunchSynchronize(aicpuStream, nullptr, aicoreStream);
        if (rc < 0) {
            return rc;
        }
        DevicePerf::GetInstance().SyncProfData(true);
        DevicePerf::GetInstance().ResetPerData();
    }
    return 0;
}

int DeviceLauncher::DynamicLaunchSynchronize(RtStream schedStream, RtStream ctrlStream, RtStream aicoreStream)
{
    int rcAicore = RuntimeStreamSynchronize(aicoreStream);
    int rcAicpu = IsAicoreResolveEnabled() ? 0 : RuntimeStreamSynchronize(schedStream);
    int rcCtrl = 0;
    if (ctrlStream != nullptr) {
        rcCtrl = RuntimeStreamSynchronize(ctrlStream);
    }
    if (IsPtoDataDumpEnabled()) {
        MACHINE_LOGD("DataDumpServerUnInit is called \n");
        (void)AdxDumpDataDumpServerUnInit();
    }
    int retAicorePrint = DeviceDfx::GetInstance().DumpAicoreLog();
    if (rcAicore != 0 || rcAicpu != 0 || rcCtrl != 0 || retAicorePrint != 0) {
        MACHINE_LOGW("sync stream failed aicpu:%d aicore:%d ctrl cpu:%d, aicorePrint: %d", rcAicpu, rcAicore, rcCtrl,
                     retAicorePrint);
    }
    return rcAicore + rcAicpu + rcCtrl;
}

int DeviceLauncher::DeviceLaunchOnceWithDeviceTensorData(
    Function* function, const std::vector<DeviceTensorData>& inputList, const std::vector<DeviceTensorData>& outputList,
    RtStream aicoreStream, bool streamSynchronize, [[maybe_unused]] CachedOperator* cachedOperator,
    [[maybe_unused]] DevControlFlowCache* inputDevCtrlCache, const DeviceLauncherConfig& config)
{
    MACHINE_LOGI("Kernel Launch");
    aicoreStream = aicoreStream == nullptr ? GetContextAiCoreStream() : aicoreStream;
    KernelLaunchInfo launchInfo(GetContextScheStream(), GetContextCtrlStream(), aicoreStream, config.blockdim,
                                config.aicpuNum);
    // 1.Add stream to capture model
    int rc = SetCaptureStream(launchInfo.aicoreStream, launchInfo.schedStream, launchInfo.isCaptureActivate);
    if (rc < 0) {
        return rc;
    }

    // 2. Change capture mode to relaxed
    if (launchInfo.isCaptureActivate) {
        ExchangeCaptureModeRelax();
    }
    HOST_PERF_TRACE(TracePhase::RunDeviceSetCapture);

    HostProf::GetInstance().SetProfFunction(function);
    rc = AclInit(nullptr);
    if (rc != 0 && rc != ACLRT_ERROR_REPEAT_INITIALIZE) {
        return rc;
    }
    HOST_PERF_TRACE(TracePhase::RunDeviceInit);

    CheckAscendDriverVersionOnboard();
    CheckDeviceId();

    auto kernel = std::make_unique<KernelBinary>(Program::GetInstance().GetFunctionSharedPtr(function));

    std::vector<DeviceTensorData> tensors;
    tensors.reserve(inputList.size() + outputList.size());
    tensors.insert(tensors.end(), inputList.begin(), inputList.end());
    tensors.insert(tensors.end(), outputList.begin(), outputList.end());

    int64_t wsSize = kernel->GetWorkspaceSize(tensors);
    HOST_PERF_TRACE(TracePhase::RunDevInitInOutTensor);

    int64_t* wsAddr = nullptr;
    if (wsSize > 0) {
        DeviceMemoryUtils devMem;
        wsAddr = reinterpret_cast<int64_t*>(devMem.AllocDev(static_cast<size_t>(wsSize), nullptr));
        if (wsAddr == nullptr) {
            MACHINE_LOGE(RtErr::RT_MALLOC_FAILED, "Failed to alloc workspace of size %ld bytes", wsSize);
            return -1;
        }
    }

    uint8_t* ctrlFlowCache = PrepareLaunch(kernel.get(), tensors, nullptr, LaunchMode::DEVICE_RT);

    DataDumpInit();
    rc = LaunchKernel(aicoreStream, ctrlFlowCache, kernel.get(), wsAddr, tensors, false, 0);
    if (rc < 0) {
        return rc;
    }

    rc = RunWithProfile(aicoreStream, launchInfo.schedStream, launchInfo.isCaptureActivate);
    if (rc < 0) {
        return rc;
    }
    if (streamSynchronize) {
        rc = DynamicLaunchSynchronize(launchInfo.schedStream, launchInfo.ctrlStream, aicoreStream);
        ASSERT(DevCommonErr::PARAM_CHECK_FAILED, DevMemoryPool::Instance().CheckAllSentinels());
    }
    MACHINE_LOGI("finish Kernel Launch.");

    HOST_PERF_TRACE(TracePhase::RunDevRunProfile);
    DataDumpUnInit();
    return rc;
}

int DeviceLauncher::DeviceSynchronize(RtStream aicpuStream, RtStream aicoreStream)
{
    int rc = DynamicLaunchSynchronize(aicpuStream, nullptr, aicoreStream);
    return rc;
}

int DeviceLauncher::DeviceRunOnce(Function* function, DevControlFlowCache* hostCtrlCache,
                                  const DeviceLauncherConfig& config)
{
    auto& inputDataList = ProgramData::GetInstance().GetInputDataList();
    auto& outputDataList = ProgramData::GetInstance().GetOutputDataList();
    std::vector<DeviceTensorData> inputDeviceDataList;
    std::vector<DeviceTensorData> outputDeviceDataList;
    DeviceMemoryUtils devMemoryUtilis(true);
    std::tie(inputDeviceDataList, outputDeviceDataList) = BuildInputOutputFromHost(devMemoryUtilis, inputDataList,
                                                                                   outputDataList);

    DeviceMemoryUtils devMemory(false);
    uint8_t* devCtrlCache = nullptr;
    if (hostCtrlCache) {
        devCtrlCache = devMemory.CopyToDev(reinterpret_cast<uint8_t*>(hostCtrlCache), hostCtrlCache->usedCacheSize,
                                           nullptr);
    }

    int rc = DeviceLaunchOnceWithDeviceTensorData(function, inputDeviceDataList, outputDeviceDataList, nullptr, true,
                                                  nullptr, reinterpret_cast<DevControlFlowCache*>(devCtrlCache),
                                                  config);
    CopyFromDev(DeviceMemoryUtils(), outputDataList);
    if (HasInplaceArgs(function) || outputDataList.size() == 0) {
        CopyFromDev(DeviceMemoryUtils(), inputDataList);
    }
    for (const auto& data : inputDeviceDataList) {
        devMemoryUtilis.Free(static_cast<uint8_t*>(data.GetAddr()));
    }
    for (const auto& data : outputDeviceDataList) {
        devMemoryUtilis.Free(static_cast<uint8_t*>(data.GetAddr()));
    }
    devMemory.Free(devCtrlCache);
    return rc;
}

void DeviceLauncher::SetDevRunCacheKernelEnable(Function* func, bool enabled)
{
    cacheInfoDict_[func].devProgEnabled = enabled;
}

bool DeviceLauncher::IsDevRunCacheKernelEnable(Function* func) { return cacheInfoDict_[func].devProgEnabled; }

void DeviceLauncher::SetDevRunCacheKernel(Function* func, uint8_t* devProg)
{
    if (!IsDevRunCacheKernelEnable(func)) {
        return;
    }
    *CachedOperator::GetCfgDataDevAddrHolder(&(cacheInfoDict_[func].cacheOperator)) = devProg;
}

CachedOperator* DeviceLauncher::GetDevRunCacheOperator(Function* func)
{
    if (!IsDevRunCacheKernelEnable(func)) {
        return nullptr;
    }
    return &(cacheInfoDict_[func].cacheOperator);
}

void DeviceLauncher::DataDumpInit()
{
    if (IsPtoDataDumpEnabled()) {
        MACHINE_LOGD("DataDumpServerInit is called \n");
        int sf = AdxDumpDataDumpServerInit();
        if (sf != 0) {
            MACHINE_LOGW("ERROR AdxDataDumpServerInit failed \n");
        }
    }
}

void DeviceLauncher::DataDumpUnInit()
{
    if (IsPtoDataDumpEnabled()) {
        MACHINE_LOGD("DataDumpServerUnInit is called \n");
        int sf = AdxDumpDataDumpServerUnInit();
        if (sf != 0) {
            MACHINE_LOGW("AdxDataDumpServerUnInit failed, ret=%d \n", sf);
        }
    }
}

int32_t DataFormat2CannFormat(const TileOpFormat format)
{
    constexpr int32_t GE_FORMAT_ND = 2;
    constexpr int32_t GE_FORMAT_NZ = 29;
    switch (format) {
        case TileOpFormat::TILEOP_ND:
            return GE_FORMAT_ND;
        case TileOpFormat::TILEOP_NZ:
            return GE_FORMAT_NZ;
        default:
            throw std::invalid_argument("Unknown Format");
    }
}

void DeviceLauncher::DumpIOTensorsWithCann(AclRtStream stream, std::vector<DeviceTensorData>& tensors,
                                           const std::string& funcName)
{
    if (AdxDumpGetDumpSwitch(AdxDumpType::OPERATOR) != 0) {
        std::vector<AdxTensorInfoV2> dumpTensors;
        for (auto& tensor : tensors) {
            AdxTensorInfoV2 info;
            info.type = AdxTensorType::INPUT;
            info.addrType = AdxAddressType::TRADITIONAL;
            info.tensorSize = static_cast<size_t>(tensor.GetDataSize());
            info.format = DataFormat2CannFormat(tensor.Format());
            info.dataType = static_cast<int32_t>(DataType2CannType(tensor.GetDataType()));
            info.tensorAddr = static_cast<int64_t*>(tensor.GetAddr());
            info.placement = static_cast<int32_t>(AdxTensorPlacement::kOnDeviceHbm);
            info.shape = tensor.GetShape();
            info.originShape = tensor.GetShape();
            dumpTensors.push_back(info);
        }
        AdxDumpDumpTensorV2(funcName, funcName, dumpTensors, stream);
    }
}

uint8_t* DeviceLauncher::CopyControlFlowCache(DevControlFlowCache* ctrlCache)
{
    uint8_t* devCache = nullptr;
    auto cacheSize = ctrlCache->usedCacheSize;
    auto bufNum = DEFAULT_RUNTIME_DATA_RING_BUFFER_COUNT;

    int ret = RuntimeMalloc((void**)&devCache, cacheSize * bufNum, RT_MEMORY_HBM, PYPTO);
    if (devCache == nullptr) {
        MACHINE_LOGE(RtErr::RT_MALLOC_FAILED, "control flow cache malloc failed");
        return nullptr;
    }
    // 维测：device 侧 blob 基址对齐。RuntimeMalloc(HBM) 实测 512B 对齐，但无文档契约；
    // 另外 ring slot i 的基址是 devCache + i*cacheSize，cacheSize 非 64 倍数时 slot 1..3 会错位
    // （当前 device 只用 slot 0，见 LaunchKernel 里 kArgs.ctrlFlowCache 直接传基址）。
    MACHINE_LOGI(
        "#ctrl.cache.align: site=CopyControlFlowCache base=%p cacheSize=%llu bufNum=%d align64=%lu "
        "slotStrideAlign64=%lu %s",
        static_cast<void*>(devCache), static_cast<unsigned long long>(cacheSize), bufNum,
        static_cast<unsigned long>(reinterpret_cast<uintptr_t>(devCache) % npu::tile_fwk::DUPPED_STITCH_NODE_ALIGN),
        static_cast<unsigned long>(cacheSize % npu::tile_fwk::DUPPED_STITCH_NODE_ALIGN),
        reinterpret_cast<uintptr_t>(devCache) % npu::tile_fwk::DUPPED_STITCH_NODE_ALIGN == 0 ? "OK" : "MISALIGNED");

    for (int i = 0; i < bufNum; ++i) {
        ret = static_cast<int>(RuntimeMemcpyDirect(devCache + i * cacheSize, cacheSize, ctrlCache, cacheSize,
                                                   RtMemcpyKind::HOST_TO_DEVICE));
        if (ret != 0) {
            MACHINE_LOGE(RtErr::RT_MEMCPY_FAILED, "control flow cache memcpy failed, ret: %d", ret);
            RuntimeFree(devCache);
            return nullptr;
        }
    }
    return devCache;
}

void DeviceLauncher::FreeControlFlowCache(uint8_t* ctrlCache)
{
    if (ctrlCache != nullptr) {
        RuntimeFree(ctrlCache);
    }
}

void DeviceLauncher::AddAicpuStream(const bool isCapture, AclMdlRI& rtModel)
{
    if (isCapture) {
        RuntimeStreamAddToModel(GetContextCtrlStream(), rtModel);
        RuntimeStreamAddToModel(GetContextScheStream(), rtModel);
    }
}

void DeviceLauncher::SaveStream(AclRtStream aicoreStream)
{
    // 存储 current stream，后续控核接口需使用current stream
    GetStreamContext().SetCurrentStream(aicoreStream);
}

void DeviceLauncher::GetCaptureInfo(AclRtStream aicoreStream, AclMdlRI& rtModel)
{
    bool isCapture = false;
    (void)GetStreamCaptureInfo(aicoreStream, rtModel, isCapture);
    DeviceLauncherContext::Get().SetCaptureMode(isCapture);
}

bool DeviceLauncher::IsCaptureMode() { return DeviceLauncherContext::Get().IsCaptureMode(); }

int DeviceLauncher::SetDevPerfAddr(const bool debugEnable, const bool isCaptureMode,
                                   const ToSubMachineConfig& profLevelConfig)
{
    int ret = 0;
    if (debugEnable || KernelBinary::GetEnableDumpDevPref() || HostProf::GetInstance().GetHostProfType() == 1) {
        if (isCaptureMode) {
            ExchangeCaptureModeRelax();
        }
        ret = DevicePerf::GetInstance().SetDebugEnable(profLevelConfig);
        if (isCaptureMode) {
            ExchangeCaptureModeGlobal();
        }
    }
    return ret;
}

int DeviceLauncher::LaunchSyncTask(AclRtStream aicoreStream, bool isCaptureMode, int launchEarlyMode)
{
    if (IsEarlyLaunchActive(isCaptureMode, launchEarlyMode)) {
        return 0;
    }

    //  close early launch
    auto schedStream = GetStreamContext().GetScheStream();
    auto ctrlStream = GetStreamContext().GetCtrlStream();
    return RunPreSync(IsAicoreResolveEnabled() ? nullptr : schedStream, ctrlStream, aicoreStream);
}

int DeviceLauncher::RunPreSync(RtStream scheStream, RtStream ctrlStream, RtStream aicoreStream)
{
    AclRtEvent event;
    if (AclRtCreateEventExWithFlag(&event, ACL_EVENT_SYNC) < 0) {
        MACHINE_LOGE(RtErr::RT_EVENT_FAILED, "AclRtCreateEvent failed.");
        return -1;
    }
    int rc = AclRtRecordEvent(event, aicoreStream);
    if (rc < 0) {
        MACHINE_LOGE(RtErr::RT_EVENT_FAILED, "AclRtRecordEvent failed %d\n", rc);
        return rc;
    }

    if (scheStream) {
        rc = AclRtStreamWaitEvent(scheStream, event);
        if (rc < 0) {
            MACHINE_LOGE(RtErr::RT_EVENT_FAILED, "AclRtStreamWaitEvent failed %d\n", rc);
            return rc;
        }
    }

    rc = AclRtStreamWaitEvent(ctrlStream, event);
    if (rc < 0) {
        MACHINE_LOGE(RtErr::RT_EVENT_FAILED, "AclRtStreamWaitEvent failed %d\n", rc);
        return rc;
    }
    return 0;
}

int DeviceLauncher::LaunchAicpuKernel(AicpuLaunchDesc& launchDesc, [[maybe_unused]] bool debugEnable,
                                      [[maybe_unused]] Function* function, const std::vector<DeviceTensorData>& tensors)
{
    auto ctrlStream = GetStreamContext().GetCtrlStream();
    auto schedStream = GetStreamContext().GetScheStream();
    HostProf::GetInstance().SetProfFunction(function, tensors);
    int ret = 0;
    auto args = static_cast<AiCpuArgs*>(launchDesc.args);
    const int nrAicpu = static_cast<int>(DeviceLauncher::GetDevProg(function)->devArgs.nrAicpu);
    const bool launchSchedSameCluster = static_cast<int>(
        DeviceLauncher::GetDevProg(function)->devArgs.launchSchedSameCluster);
    if (launchSchedSameCluster) {
        MACHINE_LOGW("When available AICPUs are insufficient, execute export PYPTO_LAUNCH_SCHED_SAME_CLUSTER=false "
                     "to disable the constraint that forces scheduling threads onto the same cluster.");
    }
    args->kArgs.parameter.ctrlBlockNum = static_cast<int>(DeviceLauncher::GetDevProg(function)->ctrlBlockDim);
    auto startTime = MspfSysCycleTime();
    args->kArgs.parameter.runMode = RUN_SPLITTED_STREAM_CTRL;
    launchDesc.stream = ctrlStream;
    launchDesc.blockDim = 1U;
    ret = LoadAicpuOp::GetInstance().LaunchBuiltInOpWithHostArgs(launchDesc, "PyptoRun");
    HostProf::GetInstance().ReportHostProfInfo(ctrlStream, startTime, 1, MSPF_GE_TASK_TYPE_AI_CPU, false);
    if (ret != RT_SUCCESS) {
        return ret;
    }
    if (IsAicoreResolveEnabled()) {
        return ret;
    }
    args->kArgs.parameter.runMode = RUN_SPLITTED_STREAM_SCHE;
    startTime = MspfSysCycleTime();
    const int scheCpuNum = static_cast<int>(DeviceLauncher::GetDevProg(function)->devArgs.scheCpuNum);
    launchDesc.stream = schedStream;
    launchDesc.blockDim = static_cast<uint32_t>(nrAicpu);
    ret = LoadAicpuOp::GetInstance().LaunchBuiltInOpWithHostArgs(launchDesc, "PyptoRun");
    HostProf::GetInstance().ReportHostProfInfo(schedStream, startTime, scheCpuNum, MSPF_GE_TASK_TYPE_AI_CPU, false);
    return ret;
}

int DeviceLauncher::LaunchAicoreKernel(AclRtStream aicoreStream, void* kernel, RtArgsEx& rtArgs,
                                       RtTaskCfgInfo& rtTaskCfg, bool debugEnable, [[maybe_unused]] Function* function)
{
    auto tilingKey = OpInfoManager::GetInstance().GetOpTilingKey();
    int blockDim = static_cast<int>(DeviceLauncher::GetDevProg(function)->ctrlBlockDim);
    if (blockDim == 0) {
        blockDim = static_cast<int>(DeviceLauncher::GetDevProg(function)->devArgs.nrValidAic);
    }
    auto startTime = MspfSysCycleTime();
    auto ret = RuntimeKernelLaunchWithHandleV2(kernel, tilingKey, blockDim, &rtArgs, nullptr, aicoreStream, &rtTaskCfg);
    HostProf::GetInstance().ReportHostProfInfo(aicoreStream, startTime, blockDim, MSPF_GE_TASK_TYPE_MIX_AIC, true);
    constexpr bool isOpenAicorePrint = static_cast<bool>(ENABLE_AICORE_PRINT);
    if (debugEnable || (!IsCaptureMode() && isOpenAicorePrint) || IsPtoDataDumpEnabled()) {
        int rc = 0;
        auto scheStream = GetStreamContext().GetScheStream();
        rc = DeviceSynchronize(scheStream, aicoreStream);
        if (rc != 0) {
            MACHINE_LOGE(HostLauncherErr::SYNC_FAILED, "stream sync failed");
            return rc;
        }
    }
    if (debugEnable) {
        DevicePerf::GetInstance().SyncProfData(debugEnable);
        ASSERT(DevCommonErr::PARAM_CHECK_FAILED, DevMemoryPool::Instance().CheckAllSentinels());
    }
    if (IsPtoDataDumpEnabled()) {
        uint32_t hostPid = GetProcessId();
        std::string sourceDir = "output/dump_tensor_" + std::to_string(hostPid);
        std::string targetDir = config::LogTopFolder() + "/dump_tensor_" + std::to_string(hostPid);
        if (IsPathExist(sourceDir)) {
            std::rename(sourceDir.c_str(), targetDir.c_str());
        }
    }
    return ret;
}

// 在这个范围之类的流不受ASCENT_RT_LAUNCH_BLOCKING的影响, 依旧是异步
int32_t DeviceLauncher::SetLaunchNoBlocking(const AclRtStream& aicoreStream)
{
    auto schedStream = GetStreamContext().GetScheStream();
    auto ctrlStream = GetStreamContext().GetCtrlStream();
    int32_t ret = 0;
    AclRtStreamAttrValue attrValue;
    attrValue.launchBlockingMode = 1;
    ret = AclRtSetStreamAttribute(schedStream, npu::tile_fwk::AclRtStreamAttr::STREAM_ATTR_LAUNCH_BLOCKING, &attrValue);
    if (ret != 0) {
        MACHINE_LOGW("ScheStream Launch set no blocking failed, ret: %d", ret);
        return static_cast<int32_t>(MachineError::HOST_LAUNCHER);
    }
    ret = AclRtSetStreamAttribute(ctrlStream, npu::tile_fwk::AclRtStreamAttr::STREAM_ATTR_LAUNCH_BLOCKING, &attrValue);
    if (ret != 0) {
        MACHINE_LOGW("CtrlStream Launch set no blocking failed, ret: %d", ret);
        return static_cast<int32_t>(MachineError::HOST_LAUNCHER);
    }
    ret = AclRtSetStreamAttribute(aicoreStream, npu::tile_fwk::AclRtStreamAttr::STREAM_ATTR_LAUNCH_BLOCKING,
                                  &attrValue);
    if (ret != 0) {
        MACHINE_LOGW("AicoreStream Launch set no blocking failed, ret: %d", ret);
        return static_cast<int32_t>(MachineError::HOST_LAUNCHER);
    }
    return ret;
}

int DeviceLauncher::LaunchKernel(AclRtStream aicoreStream, uint8_t* ctrlFlowCache, KernelBinary* kernel,
                                 int64_t* workspace, const std::vector<DeviceTensorData>& tensors, bool isDebugMode,
                                 int launchEarlyMode)
{
    auto& aicpuLaunchDesc = kernel->GetAicpuLaunchDesc();
    auto& rtAicoreArgs = kernel->GetRtAicoreArgs();
    auto& rtTaskCfg = kernel->GetRtTaskCfg();
    auto& kernelArgs = kernel->GetKernelArgs();

    auto [args, argsSize] = kernel->BuildKernelArgs(tensors);
    aicpuLaunchDesc.args = args;
    aicpuLaunchDesc.argsSize = argsSize;

    args->kArgs.ctrlFlowCache = (int64_t*)ctrlFlowCache;
    args->kArgs.workspace = workspace;
    args->kArgs.parameter.globalRound = ++sequence_;
    args->kArgs.maxDynamicAssembleOutcastMem = kernel->GetMaxDynamicAssembleOutcastMem();
    args->kArgs.maxDynamicCellMatchTableMem = kernel->GetMaxDynamicCellMatchTableMem();
    args->kArgs.runtimeDynamicCellMatchAddr = kernel->GetRuntimeDynamicCellMatchAddr();
    args->kArgs.runtimeDynamicCellMatchCapacity = kernel->GetRuntimeDynamicCellMatchCapacity();
    args->kArgs.schedSyncMode = kernel->GetSyncMode();
    auto isCaptureMode = DeviceLauncher::IsCaptureMode();
    bool debugEnable = !isCaptureMode && isDebugMode;
    const bool ringSync = IsRingEventSyncEnabled(launchEarlyMode, kernel->IsCtrlFlowCacheReplay());
    int64_t ringSequence = 0;
    if (ringSync) {
        MACHINE_ASSERT(
            kernel->BeginRingLaunch(ResolveRingEpoch(aicoreStream, isCaptureMode), isCaptureMode, ringSequence))
            << "create ping-pong events failed";
    }

    int ret = LaunchSyncTask(aicoreStream, isCaptureMode, launchEarlyMode);
    MACHINE_ASSERT(ret == RT_SUCCESS) << "launch pre sync failed: " << ret;

    if (ringSync) {
        ret = RunRingEventWaitBeforeCtrl(ringSequence, GetStreamContext().GetCtrlStream(), kernel, isCaptureMode);
        MACHINE_ASSERT(ret == RT_SUCCESS) << "launch ring event wait failed: " << ret;
    }

    ret = DeviceLauncher::SetDevPerfAddr(debugEnable, isCaptureMode, kernel->GetMachineConfig());
    MACHINE_ASSERT(ret == 0) << "set dev perf addr failed: " << ret;
    if (!isCaptureMode) {
        args->kArgs.toSubMachineConfig = kernel->GetMachineConfig();
    }
    bool isEnableBlocking = IsLaunchBlockingEnabled();
    if (!isCaptureMode && isEnableBlocking) {
        // 更新runtime 包场景这里失败需要中断，当前考虑CI和本地无runtime 包更新场景暂时warn 提示下
        if (SetLaunchNoBlocking(aicoreStream) != 0) {
            MACHINE_LOGW("May exect failed, please update CANN Package");
        }
    }
    ret = LaunchAicpuKernel(aicpuLaunchDesc, debugEnable, kernel->GetFunction(), tensors);
    MACHINE_ASSERT(ret == RT_SUCCESS) << "launch aicpu failed: " << ret;

    kernelArgs[5] = args->kArgs.cfgdata;
    kernelArgs[0] = const_cast<char*>(kernel->GetKernelname().c_str());
    kernelArgs[4] = (int64_t*)(args + 1);
    kernelArgs[6] = (DevTensorData*)((int64_t*)(args + 1) + 2);
    ret = LaunchAicoreKernel(aicoreStream, kernel->GetKernelBin(), rtAicoreArgs, rtTaskCfg, debugEnable,
                             kernel->GetFunction());
    MACHINE_ASSERT(ret == RT_SUCCESS) << "launch aicore failed: " << ret;

    if (ringSync) {
        ret = RunRingEventRecordAfterAicore(ringSequence, aicoreStream, kernel, isCaptureMode);
        MACHINE_ASSERT(ret == RT_SUCCESS) << "launch ring event record failed: " << ret;
    }
    return ret;
}

void DeviceLauncher::EmulationLaunch(Function* function, const std::vector<DeviceTensorData>& tensors,
                                     DevControlFlowCache* ctrlCache, LaunchMode launchMode)
{
    DeviceLauncherConfig config;
    DeviceLauncherConfigFillDeviceInfo(config);
    int ret = 0;
    if (launchMode == LaunchMode::EMULATION) {
        ret = EmulationLauncher::EmulationLaunchDeviceTensorData(function, tensors, {}, config, ctrlCache);
        MACHINE_ASSERT(ret == RT_SUCCESS) << "emulation run failed: " << ret;
    } else if (launchMode == LaunchMode::AICORE_MODEL) {
        ret = AicoreModelLauncher::AicoreModelLaunchDeviceTensorData(function, tensors, {}, config, ctrlCache);
        MACHINE_ASSERT(ret == RT_SUCCESS) << "aicore model run failed: " << ret;
    }
}

uint8_t* DeviceLauncher::PrepareLaunch(KernelBinary* kernel, std::vector<DeviceTensorData>& tensors, AclMdlRI rtModel,
                                       LaunchMode launchMode)
{
    AddAicpuStream(IsCaptureMode(), rtModel);
    HOST_PERF_TRACE(TracePhase::LaunchAttachStream);
    auto& cacheMgr = CtrlFlowCacheManager::Instance();
    // findOrBuildDevCache===>kermode FindCtrlFlowcache
    uint8_t* ctrlFlowCache = cacheMgr.FindOrBuildDevCache(kernel, tensors, IsCaptureMode());
    HOST_PERF_TRACE(TracePhase::FindCtrlFlowCache);
    // The one place kernel-bundle packing is decided, for every launch mode and both cache flavours. The cache
    // build above has by now stashed its bytes with the hook (value-dependent ops stash nothing, which is what
    // makes their bundle cacheless). No-op unless PYPTO_ENABLE_KERNEL_BUNDLE=1; packs once per op.
    bundle::KernelBundlePackHook::Instance().MaybePack(kernel->GetFunction()->GetDyndevAttribute().get());
    // emulation launch
    if (launchMode == LaunchMode::DEVICE_RT) {
        return ctrlFlowCache;
    }
    CtrlFlowCacheBlob hostCache;
    DevControlFlowCache* ctrlCache = cacheMgr.GetHostCtrlFlowCache(kernel, tensors, ctrlFlowCache, hostCache);
    EmulationLaunch(kernel->GetFunction(), tensors, ctrlCache, launchMode);
    return ctrlFlowCache;
}

} // namespace npu::tile_fwk::dynamic
