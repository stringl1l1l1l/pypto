/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "machine/utils/dynamic/dev_encode_workspace.h"
#include "machine/utils/dynamic/rebuildable_workspace_desc.h"

#include "machine/utils/dynamic/dev_encode_program_ctrlflow_cache.h"
#include "machine/utils/dynamic/dev_encode_function.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "machine/utils/dynamic/dev_cell_match_mem_layout.h"
#include "machine/utils/dynamic/workspace_budget_calculator.h"
#include "interface/operation/operation.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "interface/configs/config_manager_ng.h"
#include "interface/utils/common.h"
#include "tilefwk/platform.h"
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_code.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace npu::tile_fwk {
namespace dynamic {

constexpr int32_t MAX_AICORE_NUM_2210 = 75;
constexpr int32_t MAX_AICORE_NUM_3510 = 108;
static constexpr uint64_t GENERAL_METADATA_SIZE_MIN = 2 * MEBI;
constexpr size_t CALC_STITCH_NUM = ToUnderlying(WsAicpuSlabMemType::DUPPED_STITCH) -
                                   ToUnderlying(WsAicpuSlabMemType::READY_QUE);
uint32_t EffectiveUnrollTimes(const DevAscendFunction* devFunc)
{
    return devFunc != nullptr ? devFunc->unrollTimes : 1u;
}

uint32_t ParseUnrollTimesFromName(const std::string& rawName)
{
    const static std::string UNROLL_MARKS[2] = {"_LoopUnroll", "_Unroll"};
    uint32_t unrollTimes = 1;
    for (const auto& unrollMask : UNROLL_MARKS) {
        auto unrollPos = rawName.rfind(unrollMask);
        if (unrollPos == std::string::npos) {
            continue;
        }
        std::string suffix = rawName.substr(unrollPos + unrollMask.length());
        if (!suffix.empty() && std::isdigit(static_cast<unsigned char>(suffix.front()))) {
            int value = 0;
            char* endPtr = nullptr;
            value = static_cast<int>(strtol(suffix.c_str(), &endPtr, 10));
            if (endPtr != suffix.c_str() && value > 0) {
                unrollTimes *= static_cast<uint32_t>(value);
            }
        }
    }
    return std::max(unrollTimes, 1u);
}

uint32_t ComputeMaxUnrollTimesFromDevEncodeList(const std::vector<std::vector<uint8_t>>& devEncodeListInput)
{
    uint32_t maxUnroll = 1;
    for (const auto& devEncodeBin : devEncodeListInput) {
        if (devEncodeBin.empty()) {
            continue;
        }
        const auto* devFunc = reinterpret_cast<const DevAscendFunction*>(devEncodeBin.data());
        maxUnroll = std::max(maxUnroll, EffectiveUnrollTimes(devFunc));
    }
    return maxUnroll;
}

uint32_t ComputeMaxUnrollTimesFromDevProg(const DevAscendProgram& devProg)
{
    uint32_t maxUnroll = 1;
    for (size_t i = 0; i < devProg.devEncodeList.size(); ++i) {
        const auto* devFunc = reinterpret_cast<const DevAscendFunction*>(devProg.devEncodeList[i].Data());
        maxUnroll = std::max(maxUnroll, EffectiveUnrollTimes(devFunc));
    }
    return maxUnroll;
}

uint32_t ConfiguredStitchFunctionMaxNum()
{
    uint16_t stitchFunctionMaxNum = config::GetRuntimeOption<uint16_t>(STITCH_FUNCTION_MAX_NUM);
    return std::min(static_cast<uint32_t>(stitchFunctionMaxNum), static_cast<uint32_t>(MAX_STITCH_FUNC_NUM));
}

uint32_t EffectiveStitchNumMax(uint32_t maxUnrollTimes)
{
    return std::max(ConfiguredStitchFunctionMaxNum(), maxUnrollTimes);
}

void CalcWorkspaceConfig(WorkspaceDesc& wsDesc) { wsDesc.config.parallelism = config::GetDeviceSchedParallelism(); }

void CalcWorkspacePlatform(WorkspaceDesc& wsDesc)
{
    wsDesc.platform.aicoreCount = static_cast<uint64_t>(GetPlatformMaxAicoreNum());
}

RuntimeWorkspaceConfig LoadRuntimeWorkspaceConfig(uint32_t maxUnrollTimes)
{
    RuntimeWorkspaceConfig cfg;
    cfg.parallelism = config::GetDeviceSchedParallelism();
    cfg.maxWorkspaceBytes = GetMaxWorkspaceBytes();
    cfg.stitchNumMax = (cfg.maxWorkspaceBytes == 0) ? ConfiguredStitchFunctionMaxNum() :
                                                      EffectiveStitchNumMax(maxUnrollTimes);
    // stitch_function_num_per_pool is an experimental runtime option configured via
    // pypto.experimental.set_runtime_options(stitch_function_num_per_pool=[...]).
    // Default [0, 0, 0] comes from tile_fwk_config.json under the "runtime" section.
    const auto stitchFunctionNumPerPool = config::GetRuntimeOption<std::vector<int64_t>>(STITCH_FUNCTION_NUM_PER_POOL);
    if (stitchFunctionNumPerPool.size() != STITCH_FUNCTION_NUM_PER_POOL_SIZE) {
        throw std::runtime_error("stitch_function_num_per_pool must contain exactly 3 elements");
    }
    for (size_t i = 0; i < STITCH_FUNCTION_NUM_PER_POOL_SIZE; ++i) {
        const int64_t depth = stitchFunctionNumPerPool[i];
        if (depth < 0 || depth > static_cast<int64_t>(MAX_STITCH_FUNC_NUM)) {
            throw std::runtime_error("stitch_function_num_per_pool elements must be in range [0, 1024]");
        }
        cfg.stitchFunctionNumPerPool[i] = static_cast<uint32_t>(depth);
    }
    if (HasPreciseStitchFunctionNumPerPool(cfg.stitchFunctionNumPerPool)) {
        cfg.stitchNumMax = static_cast<uint32_t>(MAX_STITCH_FUNC_NUM);
    }
    return cfg;
}

uint64_t GetMaxWorkspaceBytes()
{
    const int64_t maxWorkspaceKb = config::GetRuntimeOption<int64_t>(MAX_WORKSPACE_KB);
    return maxWorkspaceKb <= 0 ? UINT64_C(0) : static_cast<uint64_t>(maxWorkspaceKb) * UINT64_C(1024);
}

static void EmitWorkspaceUserMessage(const std::string& msg)
{
    MACHINE_LOGI("%s", msg.c_str());
    (void)fprintf(stdout, "%s\n", msg.c_str());
    (void)fflush(stdout);
}

void ValidateMaxWorkspaceOrThrow(uint64_t maxWorkspaceBytes, uint64_t workspaceStitchMin)
{
    if (maxWorkspaceBytes > 0 && maxWorkspaceBytes <= workspaceStitchMin) {
        const uint64_t maxKb = WorkspaceBytesToKbCeil(maxWorkspaceBytes);
        const uint64_t minKb = WorkspaceBytesToKbCeil(workspaceStitchMin);
        std::ostringstream oss;
        oss << "[workspaceSize] max_workspace_kb=" << maxKb
            << "KB must be greater than the minimum runnable workspace=" << minKb << "KB.";
        const std::string msg = oss.str();
        EmitWorkspaceUserMessage(msg);
        MACHINE_LOGE(ExternalError::INVALID_VAL, "%s", msg.c_str());
        throw std::runtime_error(msg);
    }
}

void ApplyTensorWorkspaceResult(DevAscendProgram* devProg, const WorkspaceDesc& wsDesc)
{
    devProg->memBudget.tensor.slottableOutcastSlotSize = wsDesc.totalExclusiveOutcastSlot +
                                                         wsDesc.totalAssembleOutcastSlot;
    devProg->memBudget.tensor.rootInnerSpilledMem = wsDesc.maxRootInnerSpilledMem;
    devProg->memBudget.tensor.devTaskInnerExclusiveOutcasts = wsDesc.maxRootTotalExclusiveOutcastMem;
    devProg->memBudget.tensor.maxStaticOutcastMem = wsDesc.maxStaticOutcastMem;
    devProg->memBudget.tensor.devTaskBoundaryOutcastNum = wsDesc.devTaskBoundaryOutcastNum;
    devProg->memBudget.tensor.devTaskInnerTemporalOutcastNum = wsDesc.devTaskInnerTemporalOutcastNum;
    devProg->memBudget.aicoreSpilled.perCoreSpilledMem = wsDesc.maxLeafPerCoreSpilledMem;
    devProg->memBudget.metadata.dynamicCellMatchSlotNum = wsDesc.cellMatch.dynamicCellMatchSlotNum;
}

uint64_t CalcGeneralMetadataSlotWorkspace(DevAscendProgram* devProg)
{
    uint64_t generalMetadataSlotSize = 0;
    uint64_t itemPoolMemSize = DeviceWorkspaceAllocator::CalcMetadataItemPoolMemSize(devProg);
    uint64_t vectorMemSize = DeviceWorkspaceAllocator::CalcMetadataVectorMemSize(devProg);
    uint64_t slotAllocatorMemSize = DeviceWorkspaceAllocator::CalcMetadataSlotAllocatorMemSize(devProg);
    MACHINE_LOGI("[workspaceSize] ItemPoolMemSize is: %lu bytes, vectorMemSize is: %lu bytes, slotAllocatorMemSize is "
                 "%lu bytes.",
                 itemPoolMemSize, vectorMemSize, slotAllocatorMemSize);
    static constexpr uint64_t AICPU_SLOT_STATIC_MEMSIZE = 2 * MEBI;
    generalMetadataSlotSize = itemPoolMemSize + vectorMemSize + slotAllocatorMemSize + AICPU_SLOT_STATIC_MEMSIZE;
    MACHINE_LOGI("[workspaceSize] Workspace of generalMetadataSlotSize is %lu bytes., ", generalMetadataSlotSize);
    return generalMetadataSlotSize;
}

uint64_t CalcGeneralMetadataSlabWorkspace(DevAscendProgram* devProg)
{
    DeviceWorkspaceAllocator workspace(devProg);
    uint64_t generalMetadataSlabSize = 0;
    uint32_t slabSize = AlignUpSlab(workspace.CalcSlabMemObjmaxSize() * ALLOC_NUM_ONE_SLAB +
                                    SlabWsAllocator::FirstObjBaseOffset());
    uint32_t slabCapacity[ToUnderlying(WsAicpuSlabMemType::COHERENT_SLAB_MEM_TYPE_BUTT)];
    size_t objUsedNum[ToUnderlying(WsAicpuSlabMemType::COHERENT_SLAB_MEM_TYPE_BUTT)]{
        MAX_STITCH_FUNC_NUM, // DevFunctionDupped: protocol max; runtime submits before exceeding
        1,                   // DynFuncData
        1,                   // VecStitchList
        1,                   // DynDevTask
    };
    workspace.CalculateSlabCapacityPerType(slabSize, slabCapacity,
                                           ToUnderlying(WsAicpuSlabMemType::COHERENT_SLAB_MEM_TYPE_BUTT));

    for (int i = 0; i < ToUnderlying(WsAicpuSlabMemType::COHERENT_SLAB_MEM_TYPE_BUTT); i++) {
        MACHINE_LOGI("SlabCapacity[%d] is %u.", i, slabCapacity[i]);
        if (slabCapacity[i] == 0) {
            continue;
        }
        uint32_t requiredSlabNum = (objUsedNum[i] + slabCapacity[i] - 1) / slabCapacity[i];
        if (i == ToUnderlying(WsAicpuSlabMemType::DUPPED_FUNC_DATA)) {
            requiredSlabNum += 3;
        }
        MACHINE_LOGI("[workspaceSize] RequiredSlabNum[%d] is %u.", i, requiredSlabNum);
        generalMetadataSlabSize += static_cast<uint64_t>(requiredSlabNum) * slabSize;
    }
    MACHINE_LOGI("[workspaceSize] General->MetadataSlabSize is %lu bytes.",
                 static_cast<unsigned long>(generalMetadataSlabSize));
    generalMetadataSlabSize = (generalMetadataSlabSize < GENERAL_METADATA_SIZE_MIN) ? GENERAL_METADATA_SIZE_MIN :
                                                                                      generalMetadataSlabSize;
    return (generalMetadataSlabSize + SLAB_ALIGN_GRANULARITY) * devProg->GetParallelism();
}

uint64_t CalcStitchWorkspace(DevAscendProgram& devProg, bool aicoreResolve)
{
    DeviceWorkspaceAllocator workspace(&devProg);
    uint32_t slabCapacity[CALC_STITCH_NUM] = {0};
    uint32_t objUsedNum[CALC_STITCH_NUM] = {READY_QUEUE_SIZE,
                                            DIE_READY_QUEUE_SIZE * DIE_NUM,
                                            1,
                                            aicoreResolve ? MAX_AICORE_NUM_FOR_QUEUE : 0,
                                            aicoreResolve ? DRCO_QUEUE_MAX * NUM_LOCAL_GROUPS : 0,
                                            aicoreResolve ? DRCO_QUEUE_MAX * NUM_LOCAL_GROUPS + 1 : 0,
                                            aicoreResolve ? static_cast<uint32_t>(MAX_STITCH_FUNC_NUM) : 0};
    uint32_t slabSize = workspace.CalcStitchSlabMemObjmaxSize(slabCapacity);
    uint64_t stitchPoolSize = slabSize << 4;

    for (size_t i = 0; i < CALC_STITCH_NUM; ++i) {
        if (slabCapacity[i] == 0) {
            continue;
        }
        uint32_t requiredSlabNum = ((objUsedNum[i] << 2) + slabCapacity[i] - 1) / slabCapacity[i];
        stitchPoolSize += slabSize * requiredSlabNum;
    }
    MACHINE_LOGD("[workspaceSize] Stitch pool size is %lu bytes, with slab size:%u bytes.", stitchPoolSize, slabSize);
    return (stitchPoolSize + SLAB_ALIGN_GRANULARITY) * devProg.GetParallelism();
}

uint64_t DumpTensorWorkspace()
{
#if DEBUG_INFINITE_LIFETIME
    static constexpr uint64_t DUMP_TENSOR_WORKSPACE = 8 * GIBI;
    return DUMP_TENSOR_WORKSPACE;
#else
    return 0;
#endif
}

uint64_t LeafDumpWorkspace()
{
    if (IsPtoDataDumpEnabled()) {
        static constexpr uint64_t LEAFDUMP_WORKSPACE = 12 * MEBI;
        return LEAFDUMP_WORKSPACE;
    }
    return 0;
}

uint64_t CalcStitchCacheSize(DevAscendProgram* devProg)
{
    for (uint32_t i = 0; i < static_cast<uint32_t>(devProg->GetFunctionSize()); i++) {
        DevAscendFunction* func = devProg->GetFunction(static_cast<int>(i));
        uint32_t callOpsize = static_cast<uint32_t>(func->GetOperationSize());
        if (callOpsize > devProg->rootFuncMaxCallOpsize) {
            devProg->rootFuncMaxCallOpsize = callOpsize;
        }
    }
    uint64_t cacheSize = static_cast<uint64_t>(devProg->rootFuncMaxCallOpsize) * devProg->rootFuncMaxCallOpsize *
                         sizeof(uint64_t);
    MACHINE_LOGI("stitchCacheSize: %lu bytes, maxCallOpsize: %u", cacheSize, devProg->rootFuncMaxCallOpsize);
    return cacheSize;
}

int32_t GetPlatformMaxAicoreNum()
{
    if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) {
        return MAX_AICORE_NUM_3510;
    }
    return MAX_AICORE_NUM_2210;
}

void ApplyStitchDepthConfig(DevAscendProgram* devProg, WorkspaceDesc& wsDesc, const StitchDepthConfig& config,
                            uint64_t totalSlot)
{
    ApplyTensorWorkspaceResult(devProg, wsDesc);
    devProg->memBudget.aicoreSpilled.aicoreCount = wsDesc.platform.aicoreCount;
    devProg->memBudget.tensor.memoryDrivenWorkspace = config.memoryDrivenWorkspace;
    auto& workspacePool = devProg->memBudget.tensor.workspacePool;
    workspacePool.rootInnerUnitBytes = wsDesc.config.rootInnerUnitBytes;
    workspacePool.exclusiveOutcastUnitBytes = wsDesc.config.exclusiveOutcastUnitBytes;
    workspacePool.assembleSlotsPerDepth = wsDesc.config.assembleSlotsPerDepth;
    CopyStitchFunctionNumPerPool(workspacePool.stitchFunctionNumPerPool, wsDesc.stitchFunctionNumPerPool);
    workspacePool.preciseWorkspaceEnabled = wsDesc.config.preciseWorkspaceEnabled;
    devProg->stitchMaxFunctionNum = config.stitchMaxFunctionNum;
    devProg->memBudget.tensor.runtimeOutcastPoolSize = totalSlot * (config.runtimeOutcastPoolDepth + 1) *
                                                       devProg->GetParallelism();
}

void LogWorkspaceEncodeSummary(int kMin, const DevAscendProgram& devProg, const StitchDepthConfig& depthConfig,
                               const RuntimeWorkspaceConfig& runtimeCfg, const WorkspaceDesc& wsDesc)
{
    MACHINE_LOGI(
        "[workspaceSize] stitch depth: k_min=%d, stitchNumMax=%u, k_eff=%u, outcastCacheDepth=%u, "
        "runtimeOutcastPoolDepth=%u, devTaskBoundaryOutcastNum=%lu, devTaskInnerTemporalOutcastNum=%lu, "
        "stitchMax=%u, stitch_1=%lu bytes, encoded_tensor_budget=%lu bytes, context_workspace=%lu bytes, metadata=%lu "
        "bytes, "
        "max_workspace_kb=%lu KB, memory_driven=%d, precise_workspace=%u, "
        "rootInnerDepth=%u, assembleOutcast=%u, exclusiveDepth=%u, "
        "rootInnerUnitBytes=%lu, runtimeOutcastPoolSize=%u bytes.",
        kMin, runtimeCfg.stitchNumMax, depthConfig.kEff, depthConfig.outcastCacheDepth,
        depthConfig.runtimeOutcastPoolDepth, devProg.memBudget.tensor.devTaskBoundaryOutcastNum,
        devProg.memBudget.tensor.devTaskInnerTemporalOutcastNum, devProg.stitchMaxFunctionNum,
        runtimeCfg.workspaceStitchMin, depthConfig.encodedWorkspaceSize, devProg.workspaceSize,
        devProg.memBudget.metadata.Total(), WorkspaceBytesToKbCeil(runtimeCfg.maxWorkspaceBytes),
        static_cast<int>(depthConfig.memoryDrivenWorkspace), depthConfig.preciseWorkspaceEnabled,
        wsDesc.stitchFunctionNumPerPool[STITCH_POOL_ROOT_INNER],
        wsDesc.stitchFunctionNumPerPool[STITCH_POOL_ASSEMBLE_OUTCAST],
        wsDesc.stitchFunctionNumPerPool[STITCH_POOL_EXCLUSIVE_OUTCAST],
        devProg.memBudget.tensor.workspacePool.rootInnerUnitBytes, devProg.memBudget.tensor.runtimeOutcastPoolSize);
}

struct FlexSlotInfo {
    bool asWriteSlot{false};
    RuntimeSlotKindSet kindSet;
    uint64_t maxAssembleDstMemReq{0};
    SymbolicScalar dynMemReq;
};

static void MarkOutcastSlotsFromEncodeList(const std::vector<std::vector<uint8_t>>& hostDevEncodeList,
                                           std::vector<FlexSlotInfo>& slotInfoList)
{
    for (const auto& devEncodeBin : hostDevEncodeList) {
        if (devEncodeBin.empty()) {
            continue;
        }
        DevAscendFunction* devFunc = reinterpret_cast<DevAscendFunction*>(const_cast<uint8_t*>(devEncodeBin.data()));
        for (size_t outcastIdx = 0; outcastIdx < devFunc->GetOutcastSize(); outcastIdx++) {
            auto& toSlotList = devFunc->GetOutcast(outcastIdx).toSlotList;
            bool isInputOutputSlot = false;
            bool isAssembleOutcastSlot = false;
            for (size_t j = 0; j < toSlotList.size(); j++) {
                FlexSlotInfo& slotInfo = slotInfoList[devFunc->At(toSlotList, j)];
                if (slotInfo.kindSet.Contains(RuntimeSlotKind::INPUT) ||
                    slotInfo.kindSet.Contains(RuntimeSlotKind::OUTPUT)) {
                    isInputOutputSlot = true;
                } else if (slotInfo.kindSet.Contains(RuntimeSlotKind::ASSEMBLE_OUTCAST)) {
                    isAssembleOutcastSlot = true;
                }
            }
            if (isInputOutputSlot) {
            } else if (isAssembleOutcastSlot) {
                for (size_t j = 0; j < toSlotList.size(); j++) {
                    FlexSlotInfo& slotInfo = slotInfoList[devFunc->At(toSlotList, j)];
                    slotInfo.kindSet.Add(RuntimeSlotKind::ASSEMBLE_OUTCAST);
                    slotInfo.maxAssembleDstMemReq = std::max(slotInfo.maxAssembleDstMemReq,
                                                             devFunc->GetOutcastRawTensor(outcastIdx)->maxStaticMemReq);
                }
            } else {
                for (size_t j = 0; j < toSlotList.size(); j++) {
                    FlexSlotInfo& slotInfo = slotInfoList[devFunc->At(toSlotList, j)];
                    slotInfo.kindSet.Add(RuntimeSlotKind::EXCLUSIVE_OUTCAST);
                }
            }
        }
    }
}

static std::vector<FlexSlotInfo> MarkInputOutputAssembleSlots(
    uint32_t slotSize, const std::vector<int>& inputSlotIdxList, const std::vector<int>& outputSlotIdxList,
    const std::vector<int>& assembleSlotIdxList, const std::vector<std::vector<uint8_t>>& hostDevEncodeList)
{
    std::vector<FlexSlotInfo> slotInfoList(slotSize);
    for (int inputSlotIdx : inputSlotIdxList) {
        slotInfoList[inputSlotIdx].kindSet.Add(RuntimeSlotKind::INPUT);
    }
    for (int outputSlotIdx : outputSlotIdxList) {
        slotInfoList[outputSlotIdx].kindSet.Add(RuntimeSlotKind::OUTPUT);
    }
    for (int slotIdx : assembleSlotIdxList) {
        slotInfoList[slotIdx].kindSet.Add(RuntimeSlotKind::ASSEMBLE_OUTCAST);
    }
    MarkOutcastSlotsFromEncodeList(hostDevEncodeList, slotInfoList);
    return slotInfoList;
}

static std::vector<FlexSlotInfo> MarkInputOutputAssembleSlots(DevAscendProgram& devProg)
{
    std::vector<std::vector<uint8_t>> hostDevEncodeList;
    hostDevEncodeList.reserve(devProg.devEncodeList.size());
    for (size_t i = 0; i < devProg.devEncodeList.size(); ++i) {
        const uint8_t* data = devProg.devEncodeList[i].Data();
        const size_t size = devProg.devEncodeList[i].size();
        hostDevEncodeList.emplace_back(data, data + size);
    }
    return MarkInputOutputAssembleSlots(devProg.slotSize, devProg.GetInputTensorSlotIndexList(),
                                        devProg.GetOutputTensorSlotIndexList(),
                                        devProg.GetAssembleTensorSlotIndexList(), hostDevEncodeList);
}

static bool IsInputOutputSlot(const std::vector<FlexSlotInfo>& slotInfoList, DevAscendFunction* func, size_t idx)
{
    auto& toSlotList = func->GetOutcast(idx).toSlotList;
    for (size_t j = 0; j < toSlotList.size(); j++) {
        int slotIdx = func->At(toSlotList, j);
        if (slotInfoList[slotIdx].kindSet.Contains(RuntimeSlotKind::INPUT) ||
            slotInfoList[slotIdx].kindSet.Contains(RuntimeSlotKind::OUTPUT)) {
            return true;
        }
    }
    return false;
}

static bool IsAssembleSlot(std::vector<FlexSlotInfo>& slots, DevAscendFunction* func, size_t idx)
{
    auto& toSlotList = func->GetOutcast(idx).toSlotList;
    for (size_t j = 0; j < toSlotList.size(); j++) {
        int slotIdx = func->At(toSlotList, j);
        if (slots[slotIdx].kindSet.Contains(RuntimeSlotKind::ASSEMBLE_OUTCAST)) {
            return true;
        }
    }
    return false;
}

static SymbolicScalar GetDynRawTensorSize(Function* dynFunc, int funcKey, int idx)
{
    auto dynAttr = dynFunc->GetDyndevAttribute();
    Function* devRoot = nullptr;

    for (auto& func : dynAttr->funcGroup.devRootList) {
        if (dynAttr->funcGroup.devRootList.GetIndex(func) == funcKey) {
            devRoot = func;
            break;
        }
    }
    ASSERT(DevCommonErr::PARAM_CHECK_FAILED, devRoot) << "func " << funcKey << " missing";

    auto rawTensor = devRoot->GetOutcast()[idx]->GetRawTensor();
    ASSERT(DevCommonErr::PARAM_CHECK_FAILED, !rawTensor->GetDynRawShape().empty()) << "Not dynamic shape tensor";

    SymbolicScalar size = BitsOf(rawTensor->GetDataType());
    for (auto x : rawTensor->GetDynRawShape()) {
        size = size * x;
    }
    // The total number of bits must be byte aligned.
    return size / 0x8;
}

static void ProcessAssembleOutcast(Function* func, DevAscendFunction* devFunc, size_t outIdx,
                                   std::vector<FlexSlotInfo>& slots, uint64_t staticMemReq)
{
    SymbolicScalar dynMemReq;
    if (devFunc->GetOutcastRawTensor(outIdx)->memoryRequirement == 0) {
        dynMemReq = GetDynRawTensorSize(func, devFunc->funcKey, outIdx);
    }
    auto& toSlotList = devFunc->GetOutcast(outIdx).toSlotList;
    for (size_t j = 0; j < toSlotList.size(); j++) {
        int slotIdx = devFunc->At(toSlotList, j);
        if (dynMemReq.IsValid() || slots[slotIdx].dynMemReq.IsValid()) {
            if (!dynMemReq.IsValid()) {
                dynMemReq = staticMemReq;
            }
            if (!slots[slotIdx].dynMemReq.IsValid()) {
                slots[slotIdx].dynMemReq = slots[slotIdx].maxAssembleDstMemReq;
                slots[slotIdx].maxAssembleDstMemReq = 0;
            }
            slots[slotIdx].dynMemReq = std::max(dynMemReq, slots[slotIdx].dynMemReq);
        } else {
            slots[slotIdx].maxAssembleDstMemReq = std::max(slots[slotIdx].maxAssembleDstMemReq, staticMemReq);
        }
    }
}

static void ProcessExclusiveOutcast(DevAscendFunction* devFunc, size_t outIdx, std::vector<FlexSlotInfo>& slots)
{
    auto& toSlotList = devFunc->GetOutcast(outIdx).toSlotList;
    for (size_t j = 0; j < toSlotList.size(); j++) {
        int slotIdx = devFunc->At(toSlotList, j);
        slots[slotIdx].asWriteSlot = true;
    }
}

static void ProcessDevFunctionOutcasts(WorkspaceDesc::WorkspacePerRootFunctionDesc& rootMem, Function* func,
                                       DevAscendFunction* devFunc, std::vector<FlexSlotInfo>& slots,
                                       uint64_t& maxExclusiveOutcastMem)
{
    rootMem.func = func;
    rootMem.devFuncName = devFunc->GetRawName();
    rootMem.unroll = devFunc->unrollTimes;
    uint64_t maxStaticMemReq = 0;
    int64_t maxStaticMemReqIdx = -1;
    for (size_t i = 0; i < devFunc->GetOutcastSize(); i++) {
        if (IsInputOutputSlot(slots, devFunc, i)) {
            continue;
        }

        uint64_t staticMemReq = devFunc->GetOutcastRawTensor(i)->maxStaticMemReq;
        if (IsAssembleSlot(slots, devFunc, i)) {
            ProcessAssembleOutcast(func, devFunc, i, slots, staticMemReq);
        } else {
            ProcessExclusiveOutcast(devFunc, i, slots);
            maxStaticMemReq = std::max(maxStaticMemReq, staticMemReq);
            maxStaticMemReqIdx = static_cast<int64_t>(i);
        }
    }
    rootMem.rootMaxExclusiveOutcastMem = maxStaticMemReq;
    rootMem.rootMaxExclusiveOutcastIdx = maxStaticMemReqIdx;
    rootMem.rootInnerSpilledRawMem = devFunc->rootInnerTensorWsMemoryRequirement;
    rootMem.rootTotalExclusiveOutcastRawMem = devFunc->exclusiveOutcastWsMemoryRequirement;
    rootMem.leafPerCoreSpilledMem = static_cast<uint64_t>(devFunc->stackWorkSpaceSize);
    maxExclusiveOutcastMem = std::max(maxExclusiveOutcastMem, maxStaticMemReq);
    MACHINE_LOGD("[workspaceSize] RootInnerTensorWsMemoryRequirement is %lu bytes, exclusiveOutcastWsMemoryRequirement "
                 "is %lu bytes.",
                 devFunc->rootInnerTensorWsMemoryRequirement, devFunc->exclusiveOutcastWsMemoryRequirement);
}

// Only slots codegen marked with RUNTIME_SlotMarkNeedAlloc get device-side IT memory, i.e. the runtime slots in
// DyndevFunctionAttribute::constructAssembleNeedAllocRuntimeSlots (same numbering as slots); others must not count.
static std::pair<uint64_t, SymbolicScalar> ComputeAssembleOutcastMem(
    const std::vector<FlexSlotInfo>& slots, const std::unordered_set<int>& constructAssembleNeedAllocSlots)
{
    uint64_t maxStaticAssembleOutcastMem = 0;
    SymbolicScalar maxDynamicAssembleOutcastMem(0);
    for (int slotIdx : constructAssembleNeedAllocSlots) {
        const FlexSlotInfo& slot = slots[slotIdx];
        maxStaticAssembleOutcastMem = std::max(maxStaticAssembleOutcastMem, slot.maxAssembleDstMemReq);
        if (slot.dynMemReq.IsValid()) {
            maxDynamicAssembleOutcastMem = std::max(maxDynamicAssembleOutcastMem, slot.dynMemReq);
        }
    }
    return {maxStaticAssembleOutcastMem, maxDynamicAssembleOutcastMem};
}

static SymbolicScalar ComputeDynamicCellMatchTableBytesForOutcast(const DevAscendProgramUpdate& partial,
                                                                  const std::shared_ptr<RawTensor>& rawTensor)
{
    SymbolicScalar cellCount(1);
    auto dynShape = rawTensor->GetDynRawShape();
    int dimSize = partial.cellMatchTableDesc.GetDimensionSize();
    for (int d = 0; d < dimSize; ++d) {
        int64_t cellDim = std::max<int64_t>(partial.cellMatchTableDesc.GetCellShape(d), 1);
        SymbolicScalar tensorDim(1);
        if (!dynShape.empty() && d < static_cast<int>(dynShape.size())) {
            tensorDim = dynShape[d];
        } else {
            auto rawShape = rawTensor->GetRawShape();
            if (d < static_cast<int>(rawShape.size())) {
                tensorDim = SymbolicScalar(rawShape[d]);
            }
        }
        cellCount = cellCount * ((tensorDim + SymbolicScalar(cellDim - 1)) / SymbolicScalar(cellDim));
    }

    return cellCount * SymbolicScalar(static_cast<int64_t>(partial.cellMatchTableDesc.cellUint64Size)) *
           SymbolicScalar(static_cast<int64_t>(sizeof(uint64_t)));
}

static bool IsRuntimeDynamicPartialWithSlotRoot(const DevAscendProgramUpdate& partial,
                                                const std::shared_ptr<DyndevFunctionAttribute>& dynAttr)
{
    return partial.cellMatchRuntimePartialUpdateTable.size() == 0 &&
           partial.cellMatchTableDesc.GetDimensionSize() > 0 &&
           dynAttr->slotRootOutcastDict.count(partial.slotIndex) != 0;
}

static bool IsRuntimeDynamicPartialNeedAlloc(const DevAscendProgramUpdate& partial,
                                             const std::shared_ptr<DyndevFunctionAttribute>& dynAttr)
{
    return partial.stitchCtrlBitMask != STITCH_CTRL_NONE && IsRuntimeDynamicPartialWithSlotRoot(partial, dynAttr);
}

static SymbolicScalar ComputeMaxDynamicCellMatchTableMemPerSlot(Function* func, DevAscendProgram& devProg)
{
    auto dynAttr = func->GetDyndevAttribute();
    SymbolicScalar maxDynamicTableMem(0);
    for (size_t i = 0; i < devProg.updateList.size(); ++i) {
        auto& partial = devProg.At(devProg.updateList, i);
        if (!IsRuntimeDynamicPartialNeedAlloc(partial, dynAttr)) {
            continue;
        }
        int slotIndex = partial.slotIndex;

        SymbolicScalar slotMaxTableMem(0);
        for (const auto& [root, outcastIndex] : dynAttr->slotRootOutcastDict.at(slotIndex)) {
            auto rootIt = dynAttr->rootFuncKeyDict.find(root);
            ASSERT(DevCommonErr::PARAM_CHECK_FAILED, rootIt != dynAttr->rootFuncKeyDict.end())
                << "root not found in rootFuncKeyDict, slotIndex=" << slotIndex;
            auto rawTensor = root->GetOutcast()[outcastIndex]->GetRawTensor();
            SymbolicScalar tableBytes = ComputeDynamicCellMatchTableBytesForOutcast(partial, rawTensor);
            slotMaxTableMem = std::max(slotMaxTableMem, tableBytes);
        }
        maxDynamicTableMem = std::max(maxDynamicTableMem, slotMaxTableMem);
    }
    return maxDynamicTableMem;
}

void BuildDynamicCellMatchLaunchMeta(Function* func, DevAscendProgram& devProg)
{
    auto dynAttr = func->GetDyndevAttribute();
    dynAttr->dynamicCellMatchLaunchMetaList.clear();
    const auto* devProgBase = reinterpret_cast<const uint8_t*>(&devProg);
    for (size_t i = 0; i < devProg.updateList.size(); ++i) {
        auto& partial = devProg.At(devProg.updateList, i);
        if (partial.stitchCtrlBitMask == STITCH_CTRL_NONE || !IsRuntimeDynamicPartialWithSlotRoot(partial, dynAttr)) {
            continue;
        }
        int slotIndex = partial.slotIndex;
        DyndevFunctionAttribute::DynamicCellMatchLaunchMeta meta;
        meta.slotIndex = slotIndex;
        meta.descOffset = static_cast<uint64_t>(reinterpret_cast<const uint8_t*>(&partial.cellMatchTableDesc) -
                                                devProgBase);
        int dim = partial.cellMatchTableDesc.GetDimensionSize();
        meta.cellShape.resize(dim);
        for (int d = 0; d < dim; ++d) {
            meta.cellShape[d] = partial.cellMatchTableDesc.GetCellShape(d);
        }
        for (const auto& [root, outcastIndex] : dynAttr->slotRootOutcastDict.at(slotIndex)) {
            auto rawTensor = root->GetOutcast()[outcastIndex]->GetRawTensor();
            auto dynShape = rawTensor->GetDynRawShape();
            auto rawShape = rawTensor->GetRawShape();
            std::vector<SymbolicScalar> dims(dim, SymbolicScalar(1));
            for (int d = 0; d < dim; ++d) {
                if (!dynShape.empty() && d < static_cast<int>(dynShape.size())) {
                    dims[d] = dynShape[d];
                } else if (d < static_cast<int>(rawShape.size())) {
                    dims[d] = SymbolicScalar(rawShape[d]);
                }
            }
            meta.candidateRawDims.push_back(std::move(dims));
        }
        dynAttr->dynamicCellMatchLaunchMetaList.push_back(std::move(meta));
    }
}

static void AccumulateRootFunctionIntoWorkspaceDesc(WorkspaceDesc& desc, Function* func, DevAscendFunction* devFunc,
                                                    std::vector<FlexSlotInfo>& slots, uint64_t& maxExclusiveOutcastMem,
                                                    uint64_t& maxRootMaxExclusiveOutcastMem,
                                                    uint64_t& maxPerCoreSpilledMem)
{
    WorkspaceDesc::WorkspacePerRootFunctionDesc rootMem;
    ProcessDevFunctionOutcasts(rootMem, func, devFunc, slots, maxExclusiveOutcastMem);
    maxPerCoreSpilledMem = std::max(maxPerCoreSpilledMem, rootMem.leafPerCoreSpilledMem);
    maxRootMaxExclusiveOutcastMem = std::max(maxRootMaxExclusiveOutcastMem, rootMem.rootMaxExclusiveOutcastMem);
    desc.rootFuncDescList.push_back(std::move(rootMem));
}

static void FinalizeWorkspaceDescSlotBudgets(WorkspaceDesc& desc, const std::vector<FlexSlotInfo>& slots,
                                             const std::unordered_set<int>& constructAssembleNeedAllocSlots,
                                             uint64_t maxPerCoreSpilledMem, uint64_t maxRootMaxExclusiveOutcastMem)
{
    auto [maxStaticAssembleOutcastMem, maxDynamicAssembleOutcastMem] = ComputeAssembleOutcastMem(
        slots, constructAssembleNeedAllocSlots);
    desc.maxLeafPerCoreSpilledMem = AlignUp(maxPerCoreSpilledMem, TENSOR_ADDR_ALIGNMENT);
    desc.maxStaticOutcastMem = std::max(maxRootMaxExclusiveOutcastMem, maxStaticAssembleOutcastMem);
    desc.maxDynamicAssembleOutcastMem = maxDynamicAssembleOutcastMem;
    desc.totalExclusiveOutcastSlot = std::count_if(slots.begin(), slots.end(), [](const FlexSlotInfo& slot) {
        return slot.kindSet.Contains(RuntimeSlotKind::EXCLUSIVE_OUTCAST);
    });
    // One inner-temporal assemble slot per codegen-emitted RUNTIME_SlotMarkNeedAlloc.
    desc.totalAssembleOutcastSlot = constructAssembleNeedAllocSlots.size();
}

WorkspaceDesc CollectWorkspaceDesc(Function* func, DevAscendProgram& devProg,
                                   const std::unordered_set<int>& constructAssembleNeedAllocSlots)
{
    auto dynAttr = func->GetDyndevAttribute();
    std::vector<FlexSlotInfo> slots = MarkInputOutputAssembleSlots(devProg);

    WorkspaceDesc desc;
    desc.maxUnrollTimes = ComputeMaxUnrollTimesFromDevProg(devProg);
    CalcWorkspaceConfig(desc);
    CalcWorkspacePlatform(desc);

    uint64_t maxExclusiveOutcastMem = 0;
    uint64_t maxRootMaxExclusiveOutcastMem = 0;
    uint64_t maxPerCoreSpilledMem = 0;

    for (size_t i = 0; i < devProg.devEncodeList.size(); ++i) {
        const uint8_t* data = devProg.devEncodeList[i].Data();
        if (data == nullptr || devProg.devEncodeList[i].size() == 0) {
            continue;
        }
        DevAscendFunction* devFunc = reinterpret_cast<DevAscendFunction*>(const_cast<uint8_t*>(data));
        AccumulateRootFunctionIntoWorkspaceDesc(desc, func, devFunc, slots, maxExclusiveOutcastMem,
                                                maxRootMaxExclusiveOutcastMem, maxPerCoreSpilledMem);
    }

    FinalizeWorkspaceDescSlotBudgets(desc, slots, constructAssembleNeedAllocSlots, maxPerCoreSpilledMem,
                                     maxRootMaxExclusiveOutcastMem);

    uint64_t dynamicCellMatchSlotNum = 0;
    for (size_t i = 0; i < devProg.updateList.size(); ++i) {
        auto& partial = devProg.At(devProg.updateList, i);
        if (IsRuntimeDynamicPartialNeedAlloc(partial, dynAttr)) {
            dynamicCellMatchSlotNum++;
        }
    }
    desc.cellMatch.dynamicCellMatchSlotNum = dynamicCellMatchSlotNum;
    desc.cellMatch.maxDynamicCellMatchTableMem = ComputeMaxDynamicCellMatchTableMemPerSlot(func, devProg);
    return desc;
}

WorkspaceDesc CollectWorkspaceDescFromHostEncodeList(Function* func, const DyndevFunctionAttribute& dyndevAttr,
                                                     const std::unordered_set<int>& constructAssembleNeedAllocSlots)
{
    const auto& link = dyndevAttr.inoutLink;
    const auto& hostDevEncodeList = dyndevAttr.devEncodeList;
    std::vector<FlexSlotInfo> slots = MarkInputOutputAssembleSlots(link.totalSlot, link.inputSlotIndexList,
                                                                   link.outputSlotIndexList, link.assembleSlotIndexList,
                                                                   hostDevEncodeList);

    WorkspaceDesc desc;
    desc.maxUnrollTimes = ComputeMaxUnrollTimesFromDevEncodeList(hostDevEncodeList);
    CalcWorkspaceConfig(desc);
    CalcWorkspacePlatform(desc);

    uint64_t maxExclusiveOutcastMem = 0;
    uint64_t maxRootMaxExclusiveOutcastMem = 0;
    uint64_t maxPerCoreSpilledMem = 0;

    for (const auto& devEncodeBin : hostDevEncodeList) {
        if (devEncodeBin.empty()) {
            continue;
        }
        DevAscendFunction* devFunc = reinterpret_cast<DevAscendFunction*>(const_cast<uint8_t*>(devEncodeBin.data()));
        AccumulateRootFunctionIntoWorkspaceDesc(desc, func, devFunc, slots, maxExclusiveOutcastMem,
                                                maxRootMaxExclusiveOutcastMem, maxPerCoreSpilledMem);
    }

    FinalizeWorkspaceDescSlotBudgets(desc, slots, constructAssembleNeedAllocSlots, maxPerCoreSpilledMem,
                                     maxRootMaxExclusiveOutcastMem);

    desc.cellMatch.dynamicCellMatchSlotNum = 0;
    desc.cellMatch.maxDynamicCellMatchTableMem = SymbolicScalar(0);
    return desc;
}

uint64_t DevAscendProgram::GetCtrlFlowCacheSlottedOutcastBlockCount(uint64_t totalSlot,
                                                                    uint32_t outcastCacheDepthFallback) const
{
    const uint64_t liveSlotCount = memBudget.tensor.devTaskBoundaryOutcastNum +
                                   memBudget.tensor.devTaskInnerTemporalOutcastNum;
    if (liveSlotCount != 0) {
        return liveSlotCount;
    }
    const uint32_t depth = outcastCacheDepthFallback > 0 ?
                               outcastCacheDepthFallback :
                               (stitchMaxFunctionNum > 0 ? stitchMaxFunctionNum :
                                                           static_cast<uint32_t>(MAX_STITCH_FUNC_NUM));
    return EstimateCtrlFlowCacheSlottedBlockCount(totalSlot, depth);
}

} // namespace dynamic

#define ComputeSize(name)                                                                                \
    constexpr uint64_t ALIGNMENT_32K = 32 * 1024;                                                        \
    uint64_t tensorRootInnerSpill = data.maxRootInnerSpilledMem;                                         \
    uint64_t tensorDevTaskInnerExclusiveOutcast = data.maxRootTotalExclusiveOutcastMem;                  \
    uint64_t tensorMaxOutcast = std::max(data.maxStaticOutcastMem, maxDynamicAssembleOutcastMem);        \
    uint64_t tensorInnerTemporalOutcastNum = data.devTaskInnerTemporalOutcastNum;                        \
    uint64_t tensorBoundaryAndInnerTemporalOutcastNum = data.devTaskBoundaryOutcastNum +                 \
                                                        tensorInnerTemporalOutcastNum;                   \
    uint64_t tensorPerOutcast = tensorMaxOutcast * tensorBoundaryAndInnerTemporalOutcastNum;             \
    uint64_t tensorTotal = tensorRootInnerSpill + tensorDevTaskInnerExclusiveOutcast + tensorPerOutcast; \
    uint64_t tensorTotalAlloc = AlignUp(tensorTotal, ALIGNMENT_32K) * data.config.parallelism;           \
    uint64_t leafSpill = data.platform.aicoreCount * data.maxLeafPerCoreSpilledMem;                      \
    uint64_t name = tensorTotalAlloc + leafSpill + debugSize

uint64_t RebuildableWorkspaceDesc::GetSizeForCheckOnly(uint64_t maxDynamicAssembleOutcastMem, uint64_t debugSize) const
{
    ComputeSize(workspaceSize);
    return workspaceSize;
}

std::string RebuildableWorkspaceDesc::PrettyDumpSize(uint64_t maxDynamicAssembleOutcastMem, uint64_t debugSize) const
{
    std::ostringstream oss;
    oss << "Config:\n"
        << "  " << std::setw(30) << std::left << "parallelism:" << std::setw(10) << std::right
        << data.config.parallelism << "\n";
    oss << "Root:\n";
    for (auto& rootFuncDesc : data.rootFuncDescList) {
        oss << "  " << "name: " << rootFuncDesc.devFuncName << "\n";
        oss << "    " << "unroll:" << std::setw(3) << rootFuncDesc.unroll << " innerSpilledRaw:" << std::setw(10)
            << rootFuncDesc.rootInnerSpilledRawMem << " leafSpilled:" << std::setw(7)
            << rootFuncDesc.leafPerCoreSpilledMem << " totalOutcastRaw:" << std::setw(10)
            << rootFuncDesc.rootTotalExclusiveOutcastRawMem << " staticOutcast:" << std::setw(10)
            << rootFuncDesc.rootMaxExclusiveOutcastMem << "\n";
    }

    ComputeSize(workspaceSize);

    auto print = [&](int indent, const std::string& name, uint64_t value) {
        oss << std::setw(indent * 2) << " " << std::setw(50 - indent * 2) << std::left << name << std::setw(10)
            << std::right << value << "\n";
    };
    print(1, "workspace:", workspaceSize);
    print(2, "tensorAlloc:", tensorTotalAlloc);
    print(3, "tensor:", tensorTotal);
    print(4, "rootInnerSpilled:", tensorRootInnerSpill);
    print(4, "devTaskInnerOutcast:", tensorDevTaskInnerExclusiveOutcast);
    print(4, "devTaskBoundaryAndInnerOutcast:", tensorPerOutcast);
    print(5, "maxOutcast:", tensorMaxOutcast);
    print(6, "staticOutcast:", data.maxStaticOutcastMem);
    print(6, "dynamicOutcast:", maxDynamicAssembleOutcastMem);
    print(5, "devTaskBoundaryAndInnerOutcastCount:", tensorBoundaryAndInnerTemporalOutcastNum);
    print(6, "devTaskBoundaryOutcastCount:", data.devTaskBoundaryOutcastNum);
    print(6, "devTaskInnerOutcastCount:", tensorInnerTemporalOutcastNum);
    print(4, "preciseWorkspaceEnabled:", data.config.preciseWorkspaceEnabled);
    print(4, "stitchFunctionNumPerPool[0]:", data.stitchFunctionNumPerPool[STITCH_POOL_ROOT_INNER]);
    print(4, "stitchFunctionNumPerPool[1]:", data.stitchFunctionNumPerPool[STITCH_POOL_ASSEMBLE_OUTCAST]);
    print(4, "stitchFunctionNumPerPool[2]:", data.stitchFunctionNumPerPool[STITCH_POOL_EXCLUSIVE_OUTCAST]);
    print(4, "rootInnerUnitBytes:", data.config.rootInnerUnitBytes);
    print(4, "exclusiveOutcastUnitBytes:", data.config.exclusiveOutcastUnitBytes);
    print(4, "assembleSlotsPerDepth:", data.config.assembleSlotsPerDepth);
    print(3, "parallel:", data.config.parallelism);
    print(2, "leafSpill:", leafSpill);
    print(3, "aicoreCount:", data.platform.aicoreCount);
    print(3, "maxLeafPerCoreSpilled:", data.maxLeafPerCoreSpilledMem);
    print(2, "debug:", debugSize);
    return oss.str();
}

RBUILDABLE_ATTRIBUTE_REGISTER(RebuildableWorkspaceDesc);

} // namespace npu::tile_fwk
