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
 * \file cell_match_dynamic.cpp
 * \brief Per-launch evaluation and refresh of dynamic CellMatchTableDesc.stride on host.
 */

#include "machine/runtime/launcher/cell_match_dynamic.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "adapter/api/runtime_api.h"
#include "interface/machine/device/tilefwk/aicpu_common.h"
#include "interface/machine/device/tilefwk/aikernel_data.h"
#include "machine/runtime/launcher/device_launcher_binding.h"
#include "machine/runtime/memory_utils/memory_pool.h"
#include "machine/utils/dynamic/dev_encode_function_stitch.h"
#include "tilefwk/error_code.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk::dynamic {

void ResetRuntimeDynamicCellMatchPoolHost(uint64_t addr, uint64_t capacityBytes, bool isDevice)
{
    if (addr == 0 || capacityBytes == 0) {
        return;
    }
    // Byte 0xFF memset yields 0xFFFFFFFFFFFFFFFF per uint64, which is NOT AICORE_TASK_INIT
    // (0x00000000FFFFFFFF). Stitch compares full uint64 opId against AICORE_TASK_INIT.
    ASSERT(ProgEncodeErr::CELL_MATCH_PARAM_INVALID, (capacityBytes % sizeof(uint64_t)) == 0)
        << "ResetRuntimeDynamicCellMatchPoolHost: capacity not uint64 aligned, addr=" << addr
        << " capacityBytes=" << capacityBytes;
    const size_t numWords = static_cast<size_t>(capacityBytes / sizeof(uint64_t));

    if (!isDevice) {
        auto* table = reinterpret_cast<uint64_t*>(addr);
        for (size_t i = 0; i < numWords; ++i) {
            table[i] = AICORE_TASK_INIT;
        }
        return;
    }

    // RuntimeMemset only fills a byte pattern; stage AICORE_TASK_INIT words then H2D copy.
    // Use NormalizedRtMemcpy so aclgraph capture switches to RELAXED (direct H2D is forbidden).
    constexpr size_t kChunkWords = 512; // 4 KiB
    uint64_t chunk[kChunkWords];
    for (size_t i = 0; i < kChunkWords; ++i) {
        chunk[i] = AICORE_TASK_INIT;
    }
    auto* dst = reinterpret_cast<uint8_t*>(addr);
    size_t remaining = numWords;
    size_t byteOffset = 0;
    while (remaining > 0) {
        const size_t n = std::min(remaining, kChunkWords);
        const uint64_t bytes = static_cast<uint64_t>(n * sizeof(uint64_t));
        auto ret = NormalizedRtMemcpy(dst + byteOffset, bytes, chunk, bytes, RtMemcpyKind::HOST_TO_DEVICE);
        if (ret != RT_SUCCESS) {
            ASSERT(DevCommonErr::MEMCPY_FAILED, false)
                << "ResetRuntimeDynamicCellMatchPoolHost device memcpy failed, addr=" << addr
                << " capacityBytes=" << capacityBytes << " offsetBytes=" << byteOffset << " ret=" << ret;
            return;
        }
        remaining -= n;
        byteOffset += static_cast<size_t>(bytes);
    }
}

namespace {

bool TryBuildDynamicCellMatchDesc(const DyndevFunctionAttribute::DynamicCellMatchLaunchMeta& launchMeta,
                                  Evaluator& eval, DevCellMatchTableDesc& patchedDesc)
{
    patchedDesc.SetCellShape(launchMeta.cellShape);
    const int dim = patchedDesc.GetDimensionSize();
    if (launchMeta.candidateRawDims.empty() || dim > DEV_SHAPE_DIM_MAX) {
        return false;
    }

    bool consistent = true;
    int64_t refStride[DEV_SHAPE_DIM_MAX]{0};
    for (size_t c = 0; c < launchMeta.candidateRawDims.size(); ++c) {
        int64_t currentStride[DEV_SHAPE_DIM_MAX]{0};
        for (int d = dim - 1; d >= 0; --d) {
            auto expr = launchMeta.candidateRawDims[c][d];
            int64_t tensorDim = eval.Evaluate(expr);
            int64_t cellDim = std::max<int64_t>(patchedDesc.GetCellShape(d), 1);
            int64_t tile = (tensorDim + cellDim - 1) / cellDim;
            ASSERT(ProgEncodeErr::CELL_MATCH_PARAM_INVALID, tile > 0)
                << "Invalid tile for dynamic cell match slot=" << launchMeta.slotIndex << ", dim=" << d
                << ", tile=" << tile << ", expected > 0";
            currentStride[d] = tile;
        }
        if (c == 0) {
            for (int d = 0; d < dim; ++d) {
                refStride[d] = currentStride[d];
            }
            continue;
        }
        for (int d = 0; d < dim; ++d) {
            if (refStride[d] != currentStride[d]) {
                consistent = false;
                break;
            }
        }
        if (!consistent) {
            break;
        }
    }

    if (!consistent) {
        return false;
    }

    std::vector<int> strideShape(dim);
    for (int d = 0; d < dim; ++d) {
        strideShape[d] = static_cast<int>(refStride[d]);
    }
    patchedDesc.SetStrideShape(strideShape);
    return true;
}

} // namespace

std::vector<DevDynamicCellMatchStridePatch> PrepareDynamicCellMatchDescPatches(const DyndevFunctionAttribute& dynAttr,
                                                                               Evaluator& eval)
{
    return PrepareDynamicCellMatchDescPatches(dynAttr.dynamicCellMatchLaunchMetaList, eval);
}

std::vector<DevDynamicCellMatchStridePatch> PrepareDynamicCellMatchDescPatches(
    const std::vector<DyndevFunctionAttribute::DynamicCellMatchLaunchMeta>& launchMetaList, Evaluator& eval)
{
    std::vector<DevDynamicCellMatchStridePatch> patches;
    if (launchMetaList.empty()) {
        return patches;
    }

    for (const auto& launchMeta : launchMetaList) {
        DevCellMatchTableDesc patchedDesc;
        bool ready = TryBuildDynamicCellMatchDesc(launchMeta, eval, patchedDesc);
        if (!ready) {
            ASSERT(ProgEncodeErr::CELL_MATCH_LAUNCH_PREPARE_FAILED, false)
                << "Dynamic cell match launch prepare failed, slot=" << launchMeta.slotIndex
                << ", stride inconsistent across candidates or candidateRawDims invalid";
        }
        DevDynamicCellMatchStridePatch patch;
        patch.descOffset = launchMeta.descOffset;
        patch.stride = patchedDesc.stride;
        patches.push_back(patch);
    }
    return patches;
}

void PatchHostDynamicCellMatchTableDesc(DevAscendProgram* hostDevProg,
                                        const std::vector<DevDynamicCellMatchStridePatch>& patches)
{
    if (hostDevProg == nullptr || patches.empty()) {
        return;
    }
    auto* cfgBytes = reinterpret_cast<uint8_t*>(hostDevProg);
    for (const auto& patch : patches) {
        auto* dstDesc = reinterpret_cast<DevCellMatchTableDesc*>(cfgBytes + patch.descOffset);
        dstDesc->stride = patch.stride;
    }
}

void WriteDynamicCellMatchStridePatchesToLaunchArgs(int64_t* launchInputs,
                                                    const std::vector<DevDynamicCellMatchStridePatch>& patches)
{
    if (launchInputs == nullptr) {
        return;
    }
    const uint64_t inputCount = static_cast<uint64_t>(launchInputs[0]);
    const uint64_t outputCount = static_cast<uint64_t>(launchInputs[1]);
    auto* patchCountPtr = reinterpret_cast<uint64_t*>(
        reinterpret_cast<DevTensorData*>(launchInputs + DEV_TENSOR_DATA_OFFSET) + inputCount + outputCount);
    *patchCountPtr = patches.size();
    auto* patchArr = reinterpret_cast<DevDynamicCellMatchStridePatch*>(patchCountPtr + 1);
    for (size_t i = 0; i < patches.size(); ++i) {
        patchArr[i] = patches[i];
    }
}

void ValidateDynamicCellMatchTableMemBudget(const DyndevFunctionAttribute& dynAttr, DevAscendProgram* hostDevProg)
{
    if (hostDevProg == nullptr || hostDevProg->memBudget.metadata.dynamicCellMatchSlotNum == 0) {
        return;
    }
    const auto* cfgBytes = reinterpret_cast<const uint8_t*>(hostDevProg);
    for (const auto& launchMeta : dynAttr.dynamicCellMatchLaunchMetaList) {
        // Validate every launch-meta slot that encode marked stitchCtrlBitMask != 0 (via NeedAlloc/budget).
        const auto* desc = reinterpret_cast<const DevCellMatchTableDesc*>(cfgBytes + launchMeta.descOffset);
        const uint64_t cellMatchStride0 = desc->stride.dimStride[0];
        if (cellMatchStride0 > static_cast<uint64_t>(MAX_CELLMATCHSSTRIDE)) {
            MACHINE_LOGD("Cell-match table exceeds limit: cell_shape=%s cellMatchStride=%s table_entries=%lu limit=%lu",
                         DumpShape(desc->cellShape).c_str(), DumpStride(desc->stride).c_str(),
                         static_cast<unsigned long>(cellMatchStride0),
                         static_cast<unsigned long>(MAX_CELLMATCHSSTRIDE));
            MACHINE_LOGE(ProgEncodeErr::ASSEMBLE_STITCH_MEMORY_EXCESS,
                         "Excessive memory consumption due to insufficient tile shapes. "
                         "Consider increasing tile shapes (e.g. set_vec_tile_shapes).");
        }
    }
}

void RefillDynamicMemBudgets(DevAscendProgram* hostDevProg, DyndevFunctionAttribute& dynAttr, Evaluator& eval)
{
    if (hostDevProg == nullptr) {
        return;
    }
    if (dynAttr.maxDynamicAssembleOutcastMem.IsValid()) {
        hostDevProg->memBudget.tensor.maxDynamicAssembleOutcastMem = eval.Evaluate(
            dynAttr.maxDynamicAssembleOutcastMem);
    }
    if (!dynAttr.maxDynamicCellMatchTableMem.IsValid()) {
        return;
    }
    hostDevProg->memBudget.metadata.maxDynamicCellMatchTableMem = eval.Evaluate(dynAttr.maxDynamicCellMatchTableMem);
    uint64_t totalDynamicCellMatchSlotNum = hostDevProg->memBudget.metadata.dynamicCellMatchSlotNum;
    hostDevProg->memBudget.metadata.dynamicCellMatch = totalDynamicCellMatchSlotNum *
                                                       hostDevProg->memBudget.metadata.maxDynamicCellMatchTableMem;
}

std::vector<DevDynamicCellMatchStridePatch> PrepareHostDynamicCellMatchForLaunch(DyndevFunctionAttribute& dynAttr,
                                                                                 Evaluator& eval,
                                                                                 DevAscendProgram* hostDevProg)
{
    auto patches = PrepareDynamicCellMatchDescPatches(dynAttr, eval);
    PatchHostDynamicCellMatchTableDesc(hostDevProg, patches);
    RefillDynamicMemBudgets(hostDevProg, dynAttr, eval);
    if (dynAttr.maxDynamicCellMatchTableMem.IsValid()) {
        ValidateDynamicCellMatchTableMemBudget(dynAttr, hostDevProg);
    }
    return patches;
}

} // namespace npu::tile_fwk::dynamic
