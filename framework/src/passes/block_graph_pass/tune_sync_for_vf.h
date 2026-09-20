/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tune_sync_for_vf.h
 * \brief
 */

#ifndef TUNE_SYNC_FOR_VF_H
#define TUNE_SYNC_FOR_VF_H

#include "passes/pass_interface/pass.h"
#include "interface/program/program.h"
#include "interface/function/function.h"
#include "passes/pass_utils/pass_utils.h"

namespace npu::tile_fwk {
constexpr float vfPrarm = 0.8f;
class TuneSyncForVF : public Pass {
public:
    TuneSyncForVF() : Pass("TuneSyncForVF")
    {
        SetSupportedArches({NPUArch::DAV_3510, NPUArch::DAV_3003, NPUArch::DAV_3113});
    }
    ~TuneSyncForVF() override = default;

    Status RunOnFunction(Function& function) override;

private:
    Status ChangeOpSeq(Function* subGraphFunc, bool isAIV1);
    bool NeedAdjustSetFlag(Function* subGraphFunc, Operation* vecTileOp0, Operation* vecTileOp1, Operation* setFlag);
    bool NeedAdjustWaitFlag(Function* subGraphFunc, Operation* vecTileOp0, Operation* vecTileOp1, Operation* waitFlag);
    Status AdjustSetWaitFlag(Function* subGraphFunc, std::vector<Operation*>& setFlagList,
                             std::vector<Operation*>& waitFlagList, size_t vecTileOp0Idx, size_t vecTileOp1Idx,
                             int groupNum);
    void GenPipeOpMap(Function* subGraphFunc);
    void FindPipeVIdx(std::vector<size_t>& pipeVIdx, AIVCore coreType);
    bool IsMergeable(size_t left, size_t right, std::vector<Operation*>& setFlagList,
                     std::vector<Operation*>& waitFlagList);
    bool NeedAdjustOpSeq(Function* subGraphFunc, const std::vector<Operation*>& setFlagList,
                         const std::vector<Operation*>& waitFlagList, size_t left, size_t right);
    void AddVecTileopsToGroup(int& groupNum, size_t left, size_t right);
    size_t MoveOpsForMerge(size_t vecTileOp0Idx, size_t vecTileOp1Idx, int groupNum,
                           std::vector<Operation*>& setFlagList, std::vector<Operation*>& waitFlagList);
    Status UpdatePipeVTime(Operation* vecTileOp1, int groupNum, size_t mergedSize, int& curVFStartTime,
                           int& curVecTileOp1EndTime);
    Status UpdateSetPipeTime(Function* subGraphFunc, std::vector<Operation*>& setFlagList,
                             const int& curVecTileOp1EndTime);
    Status UpdateWaitPipeTime(Function* subGraphFunc, std::vector<Operation*>& waitFlagList, const int& curVFStartTime,
                              int& maxMoveBackDist);
    Status MoveBackPipeVOps(int groupNum, const int& maxMoveBackDist);
    void LogOpList(const std::string& label);
    std::vector<Operation*> opList_;
    std::vector<std::vector<Operation*>> mergedOps;
    std::unordered_map<PipeType, std::vector<Operation*>> pipeOpMap;
};
} // namespace npu::tile_fwk
#endif // TUNE_SYNC_FOR_VF_H
