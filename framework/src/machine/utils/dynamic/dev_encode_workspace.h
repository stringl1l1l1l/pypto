/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include "machine/utils/dynamic/dev_encode_program.h"
#include "machine/utils/dynamic/workspace_budget_calculator.h"
#include "tilefwk/workspace_desc.h"
#include "interface/function/function.h"

#include <cstdint>
#include <unordered_set>
#include <vector>

namespace npu::tile_fwk::dynamic {

uint32_t EffectiveUnrollTimes(const DevAscendFunction* devFunc);
uint32_t ParseUnrollTimesFromName(const std::string& rawName);
uint32_t ComputeMaxUnrollTimesFromDevEncodeList(const std::vector<std::vector<uint8_t>>& devEncodeListInput);
uint32_t ComputeMaxUnrollTimesFromDevProg(const DevAscendProgram& devProg);
uint32_t ConfiguredStitchFunctionMaxNum();
uint32_t EffectiveStitchNumMax(uint32_t maxUnrollTimes);
int32_t GetPlatformMaxAicoreNum();

void CalcWorkspaceConfig(WorkspaceDesc& wsDesc);
void CalcWorkspacePlatform(WorkspaceDesc& wsDesc);

RuntimeWorkspaceConfig LoadRuntimeWorkspaceConfig(uint32_t maxUnrollTimes);
WorkspaceDesc CollectWorkspaceDesc(Function* func, DevAscendProgram& devProg,
                                   const std::unordered_set<int>& constructAssembleNeedAllocSlots);

// Size-only encode: host-side devEncodeList (fillContent=false devProg has invalid Data() pointers).
WorkspaceDesc CollectWorkspaceDescFromHostEncodeList(Function* func, const DyndevFunctionAttribute& dyndevAttr,
                                                     const std::unordered_set<int>& constructAssembleNeedAllocSlots);

void ValidateMaxWorkspaceOrThrow(uint64_t maxWorkspaceBytes, uint64_t workspaceStitchMin);
void ApplyStitchDepthConfig(DevAscendProgram* devProg, WorkspaceDesc& wsDesc, const StitchDepthConfig& config,
                            uint64_t totalSlot);
void ApplyTensorWorkspaceResult(DevAscendProgram* devProg, const WorkspaceDesc& wsDesc);
void LogWorkspaceEncodeSummary(int kMin, const DevAscendProgram& devProg, const StitchDepthConfig& depthConfig,
                               const RuntimeWorkspaceConfig& runtimeCfg, const WorkspaceDesc& wsDesc);

uint64_t CalcGeneralMetadataSlotWorkspace(DevAscendProgram* devProg);
uint64_t CalcGeneralMetadataSlabWorkspace(DevAscendProgram* devProg);
uint64_t CalcStitchWorkspace(DevAscendProgram& devProg, bool aicoreResolve);
uint64_t DumpTensorWorkspace();
uint64_t LeafDumpWorkspace();
uint64_t CalcStitchCacheSize(DevAscendProgram* devProg);
uint64_t GetMaxWorkspaceBytes();
void BuildDynamicCellMatchLaunchMeta(Function* func, DevAscendProgram& devProg);

} // namespace npu::tile_fwk::dynamic
