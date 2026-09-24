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
 * \file ooo_scheduler.cpp
 * \brief
 */

#include "ooo_scheduler.h"

#include "buffer_rearrange.h"
#include "passes/pass_log/pass_log.h"
#include "interface/utils/simt_utils.h"

#ifdef MODULE_NAME
#undef MODULE_NAME
#endif
#define MODULE_NAME "OoOSchedule"

namespace npu::tile_fwk {

inline std::string coreTypeToString(CoreLocationType coreLocation)
{
    switch (coreLocation) {
        case CoreLocationType::AIV0:
            return "AIV0";
        case CoreLocationType::AIV1:
            return "AIV1";
        case CoreLocationType::AIC:
            return "AIC";
        default:
            return "MEM_UNKNOWN";
    }
}

inline bool IsMixGraph(const std::vector<Operation*>& operations)
{
    bool hasAIC = false;
    bool hasAIV = false;
    for (auto opPtr : operations) {
        if (OpcodeManager::Inst().GetCoreType(opPtr->GetOpcode()) == OpCoreType::AIV) {
            hasAIV = true;
        } else if (OpcodeManager::Inst().GetCoreType(opPtr->GetOpcode()) == OpCoreType::AIC) {
            hasAIC = true;
        }
        if (hasAIC && hasAIV) {
            return true;
        }
    }
    return false;
}

void OoOScheduler::PrintOpList(std::vector<Operation*> opList)
{
    APASS_LOG_INFO_F(Elements::Operation, "====================OoO OP_LIST =====================");
    bool needMark = false;
    if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510 || !IsMixGraph(opList)) {
        needMark = true;
    }
    for (auto& op : opList) {
        if (needMark) {
            bool isCubeComponent = op->HasAttribute(OpAttributeKey::isCube) &&
                                   op->GetBoolAttribute(OpAttributeKey::isCube);
            if (!isCubeComponent) {
                op->SetAIVCore(AIVCore::AIV0);
            }
        }
        if (!op->oOperand.empty()) {
            APASS_LOG_INFO_F(Elements::Operation, "%s[%d], range[%zu, %zu]", op->GetOpcodeStr().c_str(),
                             op->GetOpMagic(), op->oOperand[0]->memoryrange.start, op->oOperand[0]->memoryrange.end);
        } else {
            APASS_LOG_INFO_F(Elements::Operation, "%s[%d]", op->GetOpcodeStr().c_str(), op->GetOpMagic());
        }
    }
}

void OoOScheduler::UpdateIssueExecOrder()
{
    for (size_t idx = 0; idx < state_.orderedOps.size(); idx++) {
        state_.schedInfoMap[state_.orderedOps[idx]].execOrder = idx;
    }
}

Status OoOScheduler::CheckAndUpdateLifecycle()
{
    for (const auto& op : state_.orderedOps) {
        if (!state_.schedInfoMap[op].isRetired) {
            APASS_LOG_ERROR_F(Elements::Operation, "Unexecuted op: %s. %s", state_.GetOpInfo(op).c_str(),
                              GetFormatBacktrace(*op).c_str());
            return FAILED;
        }
        if (state_.schedInfoMap[op].isAlloc) {
            op->GetOutputOperand(0)->memoryrange.lifeStart = state_.localBufferMap[state_.GetOpMemIds(op)[0]]
                                                                 ->startCycle;
            op->GetOutputOperand(0)->memoryrange.lifeEnd = state_.localBufferMap[state_.GetOpMemIds(op)[0]]
                                                               ->retireCycle;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::SpillOnCoreBlock(std::pair<CoreLocationType, MemoryType> orderFirstPair)
{
    APASS_LOG_INFO_F(Elements::Operation, "Start to spillOnBlock at %s",
                     coreTypeToString(orderFirstPair.first).c_str());
    SpillContext ctx;
    if (GenBufferSpill(state_.allocIssueQueue[orderFirstPair.first][orderFirstPair.second].Front(), ctx, true) !=
        SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "SpillOnBlock failed at GenBufferSpill.");
        return FAILED;
    }
    if (ApplySpillContext(ctx, state_.allocIssueQueue[orderFirstPair.first][orderFirstPair.second].Front()) !=
        SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "ApplySpillContext failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::FindCoreLocationMemoryType(CoreLocationType coreLocation, MemoryType& spillMemType)
{
    bool anyNotEmpty = false;
    for (auto& kv : state_.allocIssueQueue[coreLocation]) {
        if (!kv.second.Empty()) {
            anyNotEmpty = true;
            break;
        }
    }
    if (!anyNotEmpty) {
        return FAILED;
    }
    if (!state_.allocIssueQueue[coreLocation][MemoryType::MEM_UB].Empty()) {
        spillMemType = MemoryType::MEM_UB;
    } else if (!state_.allocIssueQueue[coreLocation][MemoryType::MEM_L1].Empty()) {
        spillMemType = MemoryType::MEM_L1;
    } else if (coreLocation == CoreLocationType::AIC &&
               !state_.allocIssueQueue[coreLocation][MemoryType::MEM_L0C].Empty()) {
        spillMemType = MemoryType::MEM_L0C;
    } else {
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::FindFirstOrder(std::pair<CoreLocationType, MemoryType>& orderFirstPair)
{
    std::unordered_map<CoreLocationType, MemoryType> spillMemTypeMap;
    std::vector<CoreLocationType> coreVec;
    for (auto& coreLocation : state_.coreInitConfigs) {
        MemoryType spillMemType;
        if (FindCoreLocationMemoryType(coreLocation, spillMemType) != SUCCESS) {
            APASS_LOG_INFO_F(Elements::Operation, "FindCoreLocationMemoryType %s failed.",
                             coreTypeToString(coreLocation).c_str());
            continue;
        }
        spillMemTypeMap[coreLocation] = spillMemType;
        coreVec.push_back(coreLocation);
    }
    if (spillMemTypeMap.empty() || coreVec.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "All coreLocation have no spillMemType.");
        return FAILED;
    }
    std::sort(coreVec.begin(), coreVec.end(),
              [&spillMemTypeMap, this](const CoreLocationType& a, const CoreLocationType& b) {
                  return state_.schedInfoMap[state_.allocIssueQueue[a][spillMemTypeMap[a]].Front()].execOrder <
                         state_.schedInfoMap[state_.allocIssueQueue[b][spillMemTypeMap[b]].Front()].execOrder;
              });
    orderFirstPair = std::make_pair(coreVec[0], spillMemTypeMap[coreVec[0]]);
    return SUCCESS;
}

Status OoOScheduler::SpillOnBlock()
{
    std::pair<CoreLocationType, MemoryType> orderFirstPair;
    if (FindFirstOrder(orderFirstPair) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "FindFirstOrder failed.");
        return FAILED;
    }

    if (DetectSpillDeadLoop() != SUCCESS)
        return FAILED;

    bool isAivUb = orderFirstPair.second == MemoryType::MEM_UB &&
                   (orderFirstPair.first == CoreLocationType::AIV0 || orderFirstPair.first == CoreLocationType::AIV1);
    if (state_.enableDualDst && isAivUb)
        return SpillDualDstAivUbBlock();

    if (SpillOnCoreBlock(orderFirstPair) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "SpillOnCoreBlock failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::DetectSpillDeadLoop()
{
    int curPreSpillTotalQSize = 0;
    for (auto& corePair : state_.allocIssueQueue) {
        for (auto& memPair : corePair.second) {
            curPreSpillTotalQSize += static_cast<int>(memPair.second.Size());
        }
    }
    if (state_.lastPreSpillTotalQSize >= 0) {
        if (curPreSpillTotalQSize >= state_.lastPreSpillTotalQSize) {
            state_.spillNoProgressCnt++;
            if (state_.spillNoProgressCnt >= MAX_SPILL_NO_PROGRESS_ROUNDS) {
                APASS_LOG_ERROR_F(Elements::Operation,
                                  "Spill dead-loop detected: %d consecutive spills without total queue size reduction "
                                  "(current=%d, last=%d).",
                                  state_.spillNoProgressCnt, curPreSpillTotalQSize, state_.lastPreSpillTotalQSize);
                return FAILED;
            }
        } else {
            state_.spillNoProgressCnt = 0;
        }
    }
    state_.lastPreSpillTotalQSize = curPreSpillTotalQSize;
    return SUCCESS;
}

Status OoOScheduler::SpillDualDstAivUbBlock()
{
    auto& queue0 = state_.allocIssueQueue[CoreLocationType::AIV0][MemoryType::MEM_UB];
    auto& queue1 = state_.allocIssueQueue[CoreLocationType::AIV1][MemoryType::MEM_UB];
    if (queue0.Empty() || queue1.Empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "AIV UB spill queues are not synchronized.");
        return FAILED;
    }
    Operation* front0 = queue0.Front();
    Operation* front1 = queue1.Front();
    bool isDual0 = state_.IsDualDstAlloc(front0);
    bool isDual1 = state_.IsDualDstAlloc(front1);
    if (isDual0 != isDual1 || (isDual0 && (state_.schedInfoMap[front0].pairedDualDstAlloc != front1 ||
                                           state_.schedInfoMap[front1].pairedDualDstAlloc != front0))) {
        APASS_LOG_ERROR_F(Elements::Operation, "AIV UB spill queue fronts are inconsistent.");
        return FAILED;
    }
    if (SpillOnCoreBlock({CoreLocationType::AIV0, MemoryType::MEM_UB}) != SUCCESS ||
        SpillOnCoreBlock({CoreLocationType::AIV1, MemoryType::MEM_UB}) != SUCCESS) {
        return FAILED;
    }
    if (CheckAivUbPoolSlicesEqual() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Tensor, "AIV0/AIV1 UB pool slices diverged after spill.");
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::AllocSkipTensorMemRange(Operation& operation)
{
    auto outTensor = operation.GetOOperands()[0];
    int memId = outTensor->memoryrange.memId;
    if (state_.localBufferMap.find(memId) == state_.localBufferMap.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] cannot find in state_.localBufferMap.", memId);
        return FAILED;
    }
    outTensor->memoryrange = TileRange(state_.localBufferMap[memId]->start, state_.localBufferMap[memId]->end, memId);
    return SUCCESS;
}

Status OoOScheduler::AllocTensorMemRange(Operation* op)
{
    auto& skipOps = state_.schedInfoMap[op].skipOps;
    for (auto& skipOp : skipOps) {
        if (!IsSkipOp(*skipOp)) {
            APASS_LOG_ERROR_F(Elements::Operation, "%s is not a skip op.", state_.GetOpInfo(skipOp).c_str());
            return FAILED;
        }
        if (AllocSkipTensorMemRange(*skipOp) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "AllocSkipTensorMemRange failed on %s.",
                              state_.GetOpInfo(skipOp).c_str());
            return FAILED;
        }
    }
    for (auto& outTensor : op->GetOOperands()) {
        MemoryType memType = outTensor->GetMemoryTypeOriginal();
        if (memType >= MemoryType::MEM_DEVICE_DDR) {
            continue;
        }
        int memId = outTensor->memoryrange.memId;
        if (state_.tensorOccupyMap.find(memId) == state_.tensorOccupyMap.end()) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] cannot find in tensorOccupyMap.", memId);
            return FAILED;
        }
        if (state_.localBufferMap.find(memId) == state_.localBufferMap.end()) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] cannot find in localBufferMap.", memId);
            return FAILED;
        }
        APASS_LOG_DEBUG_F(Elements::Tensor, "REALLOC Tensor[%d] %s --> %s.", memId,
                          state_.GetOpInfo(state_.tensorOccupyMap[memId]).c_str(), state_.GetOpInfo(op).c_str());
        state_.tensorOccupyMap[memId] = op;
        outTensor->memoryrange = TileRange(state_.localBufferMap[memId]->start, state_.localBufferMap[memId]->end,
                                           memId);
    }
    return SUCCESS;
}

void OoOScheduler::HandleSkipOp(Operation* op)
{
    auto& skipOps = state_.schedInfoMap[op].skipOps;
    for (auto& skipOp : skipOps) {
        if (std::find(state_.newOperations.begin(), state_.newOperations.end(), skipOp) != state_.newOperations.end()) {
            continue;
        }
        state_.newOperations.emplace_back(skipOp);
    }
}

// 闸门判定：关闭时仅放行活动 scope 自身的 op（scope op 天然只在 PIPE_V 上，其余 op 留在队首
// 等待；其 execOrder 均在活动 scope 窗口外，后到就绪的 scope op 会插到它们前面，不会饿死）。
// 未关时队首 vf op 即本 scope 首 op，须等 scope 全部外部输入 tensor 就绪，否则首 op issue 立即
// 关闸，尚未执行的输入 copyin 被闸门饿死 → 调度死锁。
bool OoOScheduler::GateAllowsIssue(CoreLocationType coreLocation, PipeType pipeType, Operation* op)
{
    int activeScope = gateActiveScope_[coreLocation];
    if (activeScope >= VF_CLUSTER_ID_START) {
        return op->GetAtomicScopeId() == activeScope;
    }
    if (pipeType == PipeType::PIPE_V && op->GetAtomicScopeId() >= VF_CLUSTER_ID_START) {
        return IsScopeInputsReady(op->GetAtomicScopeId());
    }
    return true;
}

// 关闸前置 flush 判定: 首 scope op 即将关闸时, 略等更该先发射的搬运 op。
// drain 期间 V pipe 被 hold, 本 core 不再产生 compute retire, MTE3 新唤醒随之停止, 排空有界。
bool OoOScheduler::ShouldDelayScopeGate(CoreLocationType coreLocation, Operation* scopeFirstOp)
{
    if (!issueQueues[coreLocation][PipeType::PIPE_MTE3].Empty()) {
        return true;
    }
    auto& mte2Queue = issueQueues[coreLocation][PipeType::PIPE_MTE2];
    if (!mte2Queue.Empty() &&
        state_.schedInfoMap[mte2Queue.Front()].execOrder < state_.schedInfoMap[scopeFirstOp].execOrder) {
        return true;
    }
    return false;
}

// 发射后维护闸门状态：发射 scope op 即占闸并计数，全部 scope op 已发射（发射位序已固化）则开闸。
void OoOScheduler::UpdateVfScopeGate(CoreLocationType coreLocation, PipeType pipeType, Operation* op)
{
    if (op->GetAtomicScopeId() < VF_CLUSTER_ID_START || pipeType != PipeType::PIPE_V) {
        return;
    }
    int scopeId = op->GetAtomicScopeId();
    gateActiveScope_[coreLocation] = scopeId;
    scopeLaunchedCnt_[scopeId]++;
    if (scopeLaunchedCnt_[scopeId] >= scopeLaunchTotal_[scopeId]) {
        gateActiveScope_[coreLocation] = -1;
        ClearScopeTensorRanges(scopeId);
    }
}

Status OoOScheduler::LaunchIssueStage(int& nextCycle)
{
    for (auto coreLocation : state_.coreInitConfigs) {
        for (auto& pipeEntry : issueQueues[coreLocation]) {
            auto& pipe = pipeEntry.second;
            if (pipe.Empty() || pipe.busy) {
                continue;
            }
            Operation* op = pipe.Front();
            bool allows = GateAllowsIssue(coreLocation, pipeEntry.first, op);
            if (allows && pipeEntry.first == PipeType::PIPE_V && gateActiveScope_[coreLocation] < VF_CLUSTER_ID_START &&
                op->GetAtomicScopeId() >= VF_CLUSTER_ID_START && ShouldDelayScopeGate(coreLocation, op)) {
                APASS_LOG_DEBUG_F(Elements::Operation, "Delay vf scope %d gate close: drain MTE on %s.",
                                  op->GetAtomicScopeId(), coreTypeToString(coreLocation).c_str());
                continue;
            }
            if (!allows) {
                op = pipe.PopAllowing([this, coreLocation, &pipeEntry](Operation* cand) {
                    return GateAllowsIssue(coreLocation, pipeEntry.first, cand);
                });
                if (op == nullptr) {
                    continue;
                }
            } else {
                pipe.PopFront();
            }
            // 标注op的生命周期
            op->cycleStart = state_.clock;
            op->cycleEnd = state_.clock + op->GetLatency();
            pipe.busy = true;
            pipe.curOp = op;
            pipe.curOpRetireCycle = state_.clock + op->GetLatency();
            NotifyOpLaunch(op, op->cycleEnd);
            HandleSkipOp(op);
            state_.newOperations.emplace_back(op);
            UpdateVfScopeGate(coreLocation, pipeEntry.first, op);
            if (nextCycle == -1 || nextCycle > pipe.curOpRetireCycle) {
                nextCycle = pipe.curOpRetireCycle;
            }
            if (AllocTensorMemRange(op) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "AllocTensorMemRangeOp failed at coreType: %s. %s",
                                  coreTypeToString(coreLocation).c_str(), GetFormatBacktrace(*op).c_str());
                return FAILED;
            }
            APASS_LOG_DEBUG_F(Elements::Operation, "Insert: %s.", state_.GetOpInfo(op).c_str());
        }
    }
    return SUCCESS;
}

Status OoOScheduler::TryDualDstAllocPair(Operation* aiv0Alloc, Operation* aiv1Alloc, uint64_t& commitCnt,
                                         bool& allocated)
{
    if (dualDstEngine_.AllocateDualDstAtCurrent(aiv0Alloc, allocated) != SUCCESS)
        return FAILED;
    if (!allocated)
        return SUCCESS; // 当前 Full,由调用方 break 触发 SpillOnBlock

    for (Operation* alloc : {aiv0Alloc, aiv1Alloc}) {
        state_.newOperations.push_back(alloc);
        APASS_LOG_DEBUG_F(Elements::Operation, "Insert(dualdst): %s.", state_.GetOpInfo(alloc).c_str());
        if (RetireOpAndAwakeSucc(alloc, commitCnt) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "RetireOpAndAwakeSucc failed for dualdst alloc. %s",
                              GetFormatBacktrace(*alloc).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

// 查询 alloc 的落点约束: dualdst 指定 offset + vf scope 的 avoidRanges。
Status OoOScheduler::QueryAllocPlacement(Operation* op, bool& hasMatchedOffset, uint64_t& matchedOffset,
                                         std::vector<std::pair<uint64_t, uint64_t>>& avoidRanges)
{
    if (dualDstEngine_.GetMatchedAivUbAllocOffset(op, hasMatchedOffset, matchedOffset) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "GetMatchedAivUbAllocOffset failed. %s",
                          GetFormatBacktrace(*op).c_str());
        return FAILED;
    }
    int vfScopeId = op->GetAtomicScopeId();
    if (vfScopeId >= VF_CLUSTER_ID_START) {
        auto it = scopeTensorRanges_.find(vfScopeId);
        if (it != scopeTensorRanges_.end()) {
            avoidRanges = it->second;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::TryRegularAllocOnce(Operation* op, MemoryType memType, CoreLocationType coreLocation,
                                         const std::vector<int>& reqMemIds, uint64_t& commitCnt, bool& allocated)
{
    auto& pool = state_.bufferManagerMap[coreLocation][memType];
    auto buf = state_.localBufferMap[reqMemIds[0]];
    bool hasMatchedOffset = false;
    uint64_t matchedOffset = 0;
    std::vector<std::pair<uint64_t, uint64_t>> avoidRanges;
    if (QueryAllocPlacement(op, hasMatchedOffset, matchedOffset, avoidRanges) != SUCCESS) {
        return FAILED;
    }
    int vfScopeId = op->GetAtomicScopeId();
    if (!hasMatchedOffset && pool.IsFull(buf, true, avoidRanges)) {
        allocated = false;
        return SUCCESS;
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "ALLOCATE: %s.", state_.GetOpInfo(op).c_str());
    Status allocStatus = hasMatchedOffset ? pool.AllocateAtOffset(buf, matchedOffset) : pool.Allocate(buf, avoidRanges);
    if (allocStatus != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Allocate Tensor[%d] failed.", reqMemIds[0]);
        return FAILED;
    }
    if (vfScopeId >= VF_CLUSTER_ID_START) {
        RecordScopeTensorRange(vfScopeId, reqMemIds[0]);
    }
    // pool-evt 记录单池 alloc 成功时的 clock、memId 和 size，用于重建池占用曲线。
    APASS_LOG_DEBUG_F(Elements::Operation, "[pool-evt] alloc clock=%llu cycles core=%d mt=%d memId=%d size=%llu bytes",
                      (unsigned long long)state_.clock, static_cast<int>(coreLocation), static_cast<int>(memType),
                      reqMemIds[0], (unsigned long long)buf->size);
    NotifyAllocExec(op, reqMemIds[0]);
    state_.tensorOccupyMap[reqMemIds[0]] = op;
    buf->startCycle = state_.clock;
    if (op->GetOutputOperand(0) == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Alloc[%d] cannot find oOperand[0]. %s", op->GetOpMagic(),
                          GetFormatBacktrace(*op).c_str());
        return FAILED;
    }
    if (dualDstEngine_.RecordAivUbAlloc(op) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "RecordAivUbAlloc failed. %s", GetFormatBacktrace(*op).c_str());
        return FAILED;
    }
    state_.newOperations.push_back(op);
    APASS_LOG_DEBUG_F(Elements::Operation, "Insert: %s.", state_.GetOpInfo(op).c_str());
    if (RetireOpAndAwakeSucc(op, commitCnt) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "RetireOpAndAwakeSuccOp failed. %s", GetFormatBacktrace(*op).c_str());
        return FAILED;
    }
    allocated = true;
    return SUCCESS;
}

Status OoOScheduler::ExecuteAllocIssue(uint64_t& commitCnt, MemoryType memType, OpQueue& pipe)
{
    while (!pipe.Empty()) {
        Operation* op = pipe.Front();
        if (state_.enableDualDst && memType == MemoryType::MEM_UB && state_.IsDualDstAlloc(op))
            break;
        auto& coreLocation = state_.schedInfoMap[op].coreLocation;
        auto& reqMemIds = state_.GetOpMemIds(op);
        bool allocated = false;
        allocActivatedOp_ = false;
        Status st = TryRegularAllocOnce(op, memType, coreLocation, reqMemIds, commitCnt, allocated);
        if (st != SUCCESS)
            return FAILED;
        if (!allocated)
            break; // Full;退出循环交给 SpillOnBlock
        pipe.PopFront();
        // vf alloc 激活了后继 op：停发，剩余 alloc 留给下一轮（激活即停，避免提前占位破坏 exact-reuse）
        if (op->GetAtomicScopeId() >= VF_CLUSTER_ID_START && allocActivatedOp_) {
            break;
        }
    }
    return SUCCESS;
}

// GetSortedAllocatedBufs 返回的元组布局为 (memId, 起始地址, 结束地址)。
constexpr size_t BUF_START_ADDR_IDX = 1;
constexpr size_t BUF_END_ADDR_IDX = 2;

Status OoOScheduler::CheckAivUbPoolSlicesEqual()
{
    auto& pool0 = state_.bufferManagerMap.at(CoreLocationType::AIV0).at(MemoryType::MEM_UB);
    auto& pool1 = state_.bufferManagerMap.at(CoreLocationType::AIV1).at(MemoryType::MEM_UB);
    if (pool0.GetMemSize() != pool1.GetMemSize())
        return FAILED;
    auto slices0 = pool0.GetSortedAllocatedBufs();
    auto slices1 = pool1.GetSortedAllocatedBufs();
    if (slices0.size() != slices1.size())
        return FAILED;
    for (size_t i = 0; i < slices0.size(); ++i) {
        if (std::get<BUF_START_ADDR_IDX>(slices0[i]) != std::get<BUF_START_ADDR_IDX>(slices1[i]) ||
            std::get<BUF_END_ADDR_IDX>(slices0[i]) != std::get<BUF_END_ADDR_IDX>(slices1[i])) {
            return FAILED;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::ExecuteAivRegularAllocStage(uint64_t& commitCnt, OpQueue& queue0, OpQueue& queue1)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "AIV UB regular alloc stage begin: AIV0 queue=%zu, AIV1 queue=%zu.",
                      queue0.Size(), queue1.Size());
    if (ExecuteAllocIssue(commitCnt, MemoryType::MEM_UB, queue0) != SUCCESS)
        return FAILED;
    if (ExecuteAllocIssue(commitCnt, MemoryType::MEM_UB, queue1) != SUCCESS)
        return FAILED;
    bool bothAtDualDst = !queue0.Empty() && !queue1.Empty() && state_.IsDualDstAlloc(queue0.Front()) &&
                         state_.IsDualDstAlloc(queue1.Front());
    if (bothAtDualDst) {
        state_.continueAllocStage = true;
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "AIV UB regular alloc stage end at DualDst boundary; defer pair[%d/%d] to next stage.",
                          queue0.Front()->GetOpMagic(), queue1.Front()->GetOpMagic());
    }
    return SUCCESS;
}

Status OoOScheduler::ExecuteAivUbAllocRound(uint64_t& commitCnt)
{
    state_.continueAllocStage = false;
    auto& queue0 = state_.allocIssueQueue[CoreLocationType::AIV0][MemoryType::MEM_UB];
    auto& queue1 = state_.allocIssueQueue[CoreLocationType::AIV1][MemoryType::MEM_UB];
    if (queue0.Empty() && queue1.Empty())
        return SUCCESS;
    if (queue0.Empty() != queue1.Empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "AIV UB alloc queues are not synchronized.");
        return FAILED;
    }
    Operation* front0 = queue0.Front();
    Operation* front1 = queue1.Front();
    bool isDual0 = state_.IsDualDstAlloc(front0);
    bool isDual1 = state_.IsDualDstAlloc(front1);
    if (isDual0 != isDual1) {
        APASS_LOG_ERROR_F(Elements::Operation, "AIV UB alloc queue fronts have different DualDst kinds.");
        return FAILED;
    }
    if (!isDual0)
        return ExecuteAivRegularAllocStage(commitCnt, queue0, queue1);
    if (state_.schedInfoMap[front0].pairedDualDstAlloc != front1 ||
        state_.schedInfoMap[front1].pairedDualDstAlloc != front0) {
        APASS_LOG_ERROR_F(Elements::Operation, "AIV UB DualDst queue fronts are not paired.");
        return FAILED;
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "AIV UB DualDst alloc stage begin: pair[%d/%d].", front0->GetOpMagic(),
                      front1->GetOpMagic());
    if (CheckAivUbPoolSlicesEqual() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Tensor, "AIV0/AIV1 UB pool slices diverged before DualDst alloc.");
        return FAILED;
    }
    bool allocated = false;
    if (TryDualDstAllocPair(front0, front1, commitCnt, allocated) != SUCCESS)
        return FAILED;
    if (!allocated) {
        APASS_LOG_DEBUG_F(Elements::Operation, "AIV UB DualDst alloc stage stopped: buffer full for pair[%d/%d].",
                          front0->GetOpMagic(), front1->GetOpMagic());
        return SUCCESS;
    }
    queue0.PopFront();
    queue1.PopFront();
    if (CheckAivUbPoolSlicesEqual() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Tensor, "AIV0/AIV1 UB pool slices diverged after DualDst alloc.");
        return FAILED;
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "AIV UB DualDst alloc stage end: pair[%d/%d].", front0->GetOpMagic(),
                      front1->GetOpMagic());
    if (!queue0.Empty() || !queue1.Empty())
        state_.continueAllocStage = true;
    return SUCCESS;
}

Status OoOScheduler::BufferAllocStage(uint64_t& commitCnt)
{
    if (state_.enableDualDst) {
        if (ExecuteAivUbAllocRound(commitCnt) != SUCCESS)
            return FAILED;
    }
    for (auto coreLocation : state_.coreInitConfigs) {
        for (auto& [memType, pipe] : state_.allocIssueQueue[coreLocation]) {
            if (state_.enableDualDst && memType == MemoryType::MEM_UB &&
                (coreLocation == CoreLocationType::AIV0 || coreLocation == CoreLocationType::AIV1)) {
                continue;
            }
            if (pipe.Empty()) {
                continue;
            }
            // 不断按顺序执行alloc指令，直到buffer被占满为止。
            if (ExecuteAllocIssue(commitCnt, memType, pipe) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "ExecuteAllocIssue failed at coreType: %s.",
                                  coreTypeToString(coreLocation).c_str());
                return FAILED;
            }
            if (!pipe.Empty())
                state_.continueAllocStage = false;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::FreeBuffer(Operation* op, std::vector<int>& freedMemIds)
{
    auto& reqMemIds = state_.GetOpMemIds(op);
    for (auto memId : reqMemIds) {
        if (state_.DelBufRefCount(memId) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Tensor, "DelBufRefCount tensor [%d] failed.", memId);
            return FAILED;
        }
        if (state_.bufRefCount[memId] == 0) {
            CoreLocationType coreLocation = state_.ResolveCoreForFree(memId);
            auto memType = state_.localBufferMap[memId]->memType;
            uint64_t freedSize = state_.localBufferMap[memId]->size;
            if (state_.bufferManagerMap[coreLocation][memType].Free(state_.localBufferMap[memId]->id) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Free tensor [%d] failed.", memId);
                return FAILED;
            }
            // pool-evt 记录 free 事件。
            APASS_LOG_DEBUG_F(
                Elements::Operation, "[pool-evt] free clock=%llu core=%d mt=%d memId=%d size=%llu freerOp=%s",
                (unsigned long long)state_.clock, static_cast<int>(coreLocation), static_cast<int>(memType), memId,
                (unsigned long long)freedSize, state_.GetOpInfo(op).c_str());
            freedMemIds.push_back(memId);
            if (state_.tensorOccupyMap.erase(memId) == 0) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Erase tensor[%d] failed.", memId);
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

std::vector<Operation*> OoOScheduler::GetNewOperations()
{
    std::vector<Operation*> uniq;
    uniq.reserve(state_.newOperations.size());
    std::unordered_set<Operation*> seen;
    for (auto* op : state_.newOperations) {
        if (op == nullptr)
            continue;
        if (seen.insert(op).second) {
            uniq.push_back(op);
        } else {
            // 命中即代表上游某条 push 路径漏了去重 (期望随后续根因修复降为 0)。
            APASS_LOG_WARN_F(Elements::Operation, "GetNewOperations dedupe drop duplicate op[%d] %s", op->GetOpMagic(),
                             op->GetOpcodeStr().c_str());
        }
    }
    return uniq;
}

Status OoOScheduler::RetireOpAndAwakeSucc(Operation* op, uint64_t& commitCnt)
{
    commitCnt++;
    state_.schedInfoMap[op].isRetired = true;
    std::vector<int> freedMemIds;
    if (FreeBuffer(op, freedMemIds) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "FreeBufferOp failed. %s", GetFormatBacktrace(*op).c_str());
        return FAILED;
    }
    for (auto memId : freedMemIds) {
        state_.localBufferMap[memId]->retireCycle = state_.clock;
    }
    NotifyOpRetire(op, freedMemIds);

    auto& successors = state_.depManager.GetSuccessors(op);
    for (auto succOp : successors) {
        if (state_.schedInfoMap[succOp].isRetired) {
            continue;
        }
        bool ready = true;
        auto& preds = state_.depManager.GetPredecessors(succOp);
        for (auto predOp : preds) {
            if (!state_.schedInfoMap[predOp].isRetired) {
                ready = false;
                break;
            }
        }
        if (ready) {
            issueQueues[state_.schedInfoMap[succOp].coreLocation][state_.schedInfoMap[succOp].pipeType].Insert(succOp);
            allocActivatedOp_ = true;
            APASS_LOG_DEBUG_F(Elements::Operation, "    Wakeup: %s, execOrder: %d", state_.GetOpInfo(succOp).c_str(),
                              state_.schedInfoMap[succOp].execOrder);
        }
    }
    return SUCCESS;
}

Status OoOScheduler::RetireCoreIssue(CoreLocationType coreLocation, uint64_t& commitCnt, int& nextCycle)
{
    for (auto& [pipeType, pipe] : issueQueues[coreLocation]) {
        if (!pipe.busy) {
            continue;
        }
        if (!state_.pipeEndTime.count(pipeType)) {
            state_.pipeEndTime.emplace(pipeType, pipe.curOpRetireCycle);
        } else {
            auto curEndTime = state_.pipeEndTime[pipeType];
            state_.pipeEndTime[pipeType] = std::max(curEndTime, pipe.curOpRetireCycle);
        }
        if (pipe.curOpRetireCycle <= state_.clock) { // 如果该pipe内当前正在执行op，在clock的时刻已经执行完毕。
            Operation* op = pipe.curOp;
            pipe.busy = false;
            pipe.curOp = nullptr;
            APASS_LOG_DEBUG_F(Elements::Operation, "EXECUTE END: %s", state_.GetOpInfo(op).c_str());
            if (RetireOpAndAwakeSucc(op, commitCnt) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "RetireOpAndAwakeSuccOp failed at coreType: %s! %s",
                                  coreTypeToString(coreLocation).c_str(), GetFormatBacktrace(*op).c_str());
                return FAILED;
            }
            continue;
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "EXECUTING[%d]: %s", pipe.curOpRetireCycle,
                          state_.GetOpInfo(pipe.curOp).c_str());
        if (nextCycle == -1 || nextCycle > pipe.curOpRetireCycle) {
            nextCycle = pipe.curOpRetireCycle;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::RetireIssueStage(uint64_t& commitCnt, int& nextCycle)
{
    for (auto coreLocation : state_.coreInitConfigs) {
        if (RetireCoreIssue(coreLocation, commitCnt, nextCycle) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "RetireIssueStage failed");
            return FAILED;
        }
    }
    return SUCCESS;
}

void OoOScheduler::LaunchReadyIssue()
{
    // 初始化 Queue
    for (auto& op : state_.orderedOps) {
        auto& coreLocation = state_.schedInfoMap[op].coreLocation;
        if (USE_LESS_OPS.find(op->GetOpcode()) != USE_LESS_OPS.end() && state_.depManager.GetPredecessors(op).empty()) {
            issueQueues[coreLocation][state_.schedInfoMap[op].pipeType].Insert(op);
        }
        if (state_.schedInfoMap[op].isAlloc) {
            auto& reqMemIds = state_.GetOpMemIds(op);
            if (!reqMemIds.empty()) {
                auto memType = state_.localBufferMap[reqMemIds[0]]->memType;
                state_.allocIssueQueue[coreLocation][memType].Insert(op);
            }
        }
    }
}

Status OoOScheduler::ScheduleMainLoop()
{
    LOG_SCOPE_BEGIN(tScheduleMainLoop, Elements::Function, "ScheduleMainLoop");
    auto ret = RunSchedulerMainLoop(*this);
    LOG_SCOPE_END(tScheduleMainLoop);
    return ret;
}

Status OoOScheduler::PreMainLoop()
{
    UpdateIssueExecOrder();
    LaunchReadyIssue();
    state_.numTotalIssues = state_.orderedOps.size();
    state_.originalNumTotalIssues = state_.numTotalIssues;
    NotifyMainLoopBegin();
    return SUCCESS;
}

Status OoOScheduler::PostMainLoop()
{
    NotifyMainLoopEnd();
    return SUCCESS;
}

Status OoOScheduler::RetireIssue(Operation* op)
{
    state_.schedInfoMap[op].isRetired = true;
    std::vector<int> freedMemIds;
    return FreeBuffer(op, freedMemIds);
}

Status OoOScheduler::AllocateAfterSpill(Operation* op, LocalBufferPtr allocBuffer, CoreLocationType coreLocation,
                                        bool isDualDst)
{
    if (!isDualDst) {
        if (state_.bufferManagerMap[coreLocation][allocBuffer->memType].Allocate(allocBuffer) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Allocate tensor[%d] failed.", allocBuffer->id);
            return FAILED;
        }
        return SUCCESS;
    }

    bool allocated = false;
    if (dualDstEngine_.AllocateDualDstAtCurrent(op, allocated) != SUCCESS)
        return FAILED;
    if (!allocated) {
        APASS_LOG_ERROR_F(Elements::Operation, "DualDst alloc still infeasible after spill. %s",
                          GetFormatBacktrace(*op).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::ExecuteAllocIssue(Operation* op, size_t& pcIdx)
{
    if (state_.localBufferMap.find(state_.GetOpMemIds(op)[0]) == state_.localBufferMap.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] cannot find in localBufferMap!", state_.GetOpMemIds(op)[0]);
        return FAILED;
    }
    LocalBufferPtr allocBuffer = state_.localBufferMap[state_.GetOpMemIds(op)[0]];
    auto coreLocation = state_.schedInfoMap[op].coreLocation;
    const bool isDualDst = (allocBuffer->memType == MemoryType::MEM_UB) && state_.IsDualDstAlloc(op);

    bool needSpill = false;
    if (isDualDst) {
        bool allocated = false;
        if (dualDstEngine_.AllocateDualDstAtCurrent(op, allocated) != SUCCESS)
            return FAILED;
        if (allocated) {
            return SUCCESS;
        }
        needSpill = true;
    } else {
        needSpill = state_.bufferManagerMap[coreLocation][allocBuffer->memType].IsFull(allocBuffer, false);
    }

    if (needSpill) {
        SpillContext ctx;
        if (GenBufferSpill(op, ctx, false) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "GenBufferSpill failed at ExecuteAllocIssue. %s",
                              GetFormatBacktrace(*op).c_str());
            return FAILED;
        }
        if (isDualDst) {
            auto currentIt = std::find(state_.orderedOps.begin(), state_.orderedOps.end(), op);
            if (currentIt == state_.orderedOps.end())
                return FAILED;
            pcIdx = static_cast<size_t>(currentIt - state_.orderedOps.begin());
        } else {
            pcIdx += ctx.newCopyoutOps.size() - ctx.deleteRetiredOpSize;
        }
    }

    return AllocateAfterSpill(op, allocBuffer, coreLocation, isDualDst);
}

Status OoOScheduler::SeqSchedule()
{
    UpdateIssueExecOrder();
    state_.originalNumTotalIssues = state_.orderedOps.size();
    size_t pcIdx = 0;
    LOG_SCOPE_BEGIN(tGenSpillSchedule, Elements::Function, "SeqSchedule");
    while (pcIdx < state_.orderedOps.size()) {
        auto op = state_.orderedOps[pcIdx];
        if (state_.IsDualDstAlloc(op) && state_.schedInfoMap[op].isRetired) {
            pcIdx += 1;
            continue;
        }
        if (ProcessSeqOp(op, pcIdx) != SUCCESS)
            return FAILED;
    }
    LOG_SCOPE_END(tGenSpillSchedule);
    for (auto bufRef : state_.bufRefCount) {
        if (bufRef.second != 0) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] bufRefCount not equal to 0!", bufRef.first);
            return FAILED;
        }
    }
    if (state_.InitBufRefCount(state_.orderedOps) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InitBufRefCount failed!");
        return FAILED;
    }
    for (const auto& op : state_.orderedOps) {
        state_.schedInfoMap[op].isRetired = false;
    }
    // 更新依赖关系
    if (state_.depManager.InitDependencies(state_.orderedOps, false) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InitDependencies failed!");
        return FAILED;
    }
    state_.depManager.PrintDependencies(state_.orderedOps);
    return SUCCESS;
}

Status OoOScheduler::ProcessSeqOp(Operation* op, size_t& pcIdx)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "Launch %s", state_.GetOpInfo(op).c_str());
    if (state_.schedInfoMap[op].isAlloc) {
        if (ExecuteAllocIssue(op, pcIdx) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "ExecuteAllocIssue failed! %s", GetFormatBacktrace(*op).c_str());
            return FAILED;
        }
    }
    for (auto& outTensor : op->GetOOperands()) {
        if (outTensor->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR) {
            state_.tensorOccupyMap[outTensor->memoryrange.memId] = op;
        }
    }
    Operation* peer = nullptr;
    if (state_.IsDualDstAlloc(op)) {
        peer = state_.schedInfoMap[op].pairedDualDstAlloc;
        if (peer == nullptr || state_.schedInfoMap.find(peer) == state_.schedInfoMap.end() ||
            state_.schedInfoMap[peer].isRetired) {
            APASS_LOG_ERROR_F(Elements::Operation, "Resolve paired DualDst alloc failed for %s.",
                              GetFormatBacktrace(*op).c_str());
            return FAILED;
        }
    }
    if (RetireIssue(op) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "RetireIssue failed! %s", GetFormatBacktrace(*op).c_str());
        return FAILED;
    }
    if (peer != nullptr && RetireIssue(peer) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Retire paired DualDst alloc failed! %s",
                          GetFormatBacktrace(*peer).c_str());
        return FAILED;
    }
    pcIdx += 1;
    if (state_.orderedOps.size() > state_.originalNumTotalIssues * MAX_NUM_TOTAL_ISSUES_RATIO) {
        APASS_LOG_ERROR_F(Elements::Operation, "Spill dead-loop.");
        return FAILED;
    }
    return SUCCESS;
}

void OoOScheduler::InitIssueQueuesAndBufferManager()
{
    // 设置比较函数用于OpQueue排序
    auto compareFunc = [this](Operation* a, Operation* b) {
        return state_.schedInfoMap[a].execOrder > state_.schedInfoMap[b].execOrder;
    };

    // 初始化
    for (auto coreLocation : state_.coreInitConfigs) {
        for (size_t i = 0; i <= static_cast<int>(PipeType::PIPE_FIX); i++) {
            OpQueue queue;
            queue.SetCompareFunc(compareFunc);
            issueQueues[coreLocation][static_cast<PipeType>(i)] = queue;
        }
    }

    state_.bufferManagerMap.clear();
    for (auto coreLocation : state_.coreInitConfigs) {
        if (coreLocation == CoreLocationType::AIV0 || coreLocation == CoreLocationType::AIV1) {
            OpQueue queue;
            queue.SetCompareFunc(compareFunc);
            state_.allocIssueQueue[coreLocation][MemoryType::MEM_UB] = queue;
            state_.bufferManagerMap[coreLocation].insert(
                {MemoryType::MEM_UB, BufferPool(MemoryType::MEM_UB, state_.localMemSize[MemoryType::MEM_UB])});
            continue;
        }
        for (size_t i = 1; i < static_cast<int>(MemoryType::MEM_DEVICE_DDR); i++) {
            OpQueue queue;
            queue.SetCompareFunc(compareFunc);
            state_.allocIssueQueue[coreLocation][static_cast<MemoryType>(i)] = queue;
            if (state_.localMemSize.find(static_cast<MemoryType>(i)) != state_.localMemSize.end()) {
                state_.bufferManagerMap[coreLocation].insert(
                    {static_cast<MemoryType>(i),
                     BufferPool(static_cast<MemoryType>(i), state_.localMemSize[static_cast<MemoryType>(i)])});
            }
        }
    }
}

void OoOScheduler::InitVfScopeGate()
{
    gateActiveScope_.clear();
    scopeLaunchedCnt_.clear();
    scopeLaunchTotal_.clear();
    scopeTensorRanges_.clear();
    state_.vfScopeTensorMemIds.clear();
    state_.vfScopeOpsByScope.clear();
    for (auto* op : state_.orderedOps) {
        int scopeId = op->GetAtomicScopeId();
        if (scopeId >= VF_CLUSTER_ID_START) {
            state_.vfScopeOpsByScope[scopeId].push_back(op);
            if (!state_.schedInfoMap[op].isAlloc) {
                scopeLaunchTotal_[scopeId]++;
            }
            for (auto& iOperand : op->GetIOperands()) {
                if (iOperand->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR) {
                    state_.vfScopeTensorMemIds[scopeId].insert(iOperand->memoryrange.memId);
                }
            }
            for (auto& oOperand : op->GetOOperands()) {
                if (oOperand->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR) {
                    state_.vfScopeTensorMemIds[scopeId].insert(oOperand->memoryrange.memId);
                }
            }
        }
    }
    for (auto& scope : scopeLaunchTotal_) {
        scopeLaunchedCnt_[scope.first] = 0;
    }
    APASS_LOG_INFO_F(Elements::Operation, "InitVfScopeGate: %zu vf scopes with launch counting.",
                     scopeLaunchTotal_.size());
}

// scope 全部外部输入 tensor（producer 不在本 scope、且非 ALLOC）的 producer 已 retire
bool OoOScheduler::IsScopeInputsReady(int scopeId)
{
    auto it = state_.vfScopeOpsByScope.find(scopeId);
    if (it == state_.vfScopeOpsByScope.end()) {
        return true;
    }
    for (auto* op : it->second) {
        for (auto& iop : op->GetIOperands()) {
            if (iop->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            for (auto* producer : iop->GetProducers()) {
                if (producer == nullptr || IsAllocOpCode(producer->GetOpcode())) {
                    continue;
                }
                if (producer->GetAtomicScopeId() == scopeId) {
                    continue;
                }
                auto schedIt = state_.schedInfoMap.find(producer);
                if (schedIt == state_.schedInfoMap.end()) {
                    continue;
                }
                if (!schedIt->second.isRetired) {
                    return false;
                }
            }
        }
    }
    return true;
}

void OoOScheduler::RecordScopeTensorRange(int scopeId, int memId)
{
    auto bufIt = state_.localBufferMap.find(memId);
    if (bufIt == state_.localBufferMap.end() || bufIt->second == nullptr) {
        return;
    }
    auto& buf = bufIt->second;
    scopeTensorRanges_[scopeId].push_back({buf->start, buf->end});
    APASS_LOG_DEBUG_F(Elements::Operation, "RecordScopeTensorRange: scope=%d memId=%d range=[%lu, %lu].", scopeId,
                      memId, buf->start, buf->end);
}

void OoOScheduler::ClearScopeTensorRanges(int scopeId)
{
    auto it = scopeTensorRanges_.find(scopeId);
    if (it == scopeTensorRanges_.end()) {
        return;
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "ClearScopeTensorRanges: scope=%d, %zu ranges cleared.", scopeId,
                      it->second.size());
    scopeTensorRanges_.erase(it);
}

void OoOScheduler::InitTensorCoreMap()
{
    // 不存在 no producer情况
    for (auto op : state_.orderedOps) {
        if (state_.schedInfoMap[op].isAlloc) {
            auto memId = op->GetOutputOperand(0)->memoryrange.memId;
            state_.tensorAllocMap[memId] = op;
        }
    }
}

void OoOScheduler::InitCoreConfig(const std::vector<Operation*>& opList)
{
    if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510 || !IsMixGraph(opList)) {
        state_.coreInitConfigs = CORE_INIT_CONFIGS_HARDWARE_ONE;
    } else {
        state_.coreInitConfigs = CORE_INIT_CONFIGS_HARDWARE_TWO;
    }
}

// skipOps 按生产者在前存放: HandleSkipOp 照这个次序写回调度序列, 反了的话消费者会排在生产者之前。
// 向上走出的祖先路径天然是消费者在前, 每条子链单独翻转后追加; 去重命中意味着该 op 连同它的整条
// 祖先前缀已经在更前面放好了, 所以跨 operand 共享祖先时次序依然成立。
void OoOScheduler::InitSkipOps(Operation* op)
{
    if (op == nullptr)
        return;
    std::vector<Operation*> skipOps;
    for (auto iOperand : op->GetIOperands()) {
        for (auto pre : iOperand->GetProducers()) {
            std::vector<Operation*> chain;
            while (pre != nullptr && IsSkipOp(*pre) &&
                   pre->GetOutputOperand(0)->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR) {
                chain.push_back(pre);
                const auto& upProducers = pre->GetInputOperand(0)->GetProducers();
                pre = upProducers.empty() ? nullptr : *upProducers.begin();
            }
            for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
                if (std::find(skipOps.begin(), skipOps.end(), *it) == skipOps.end()) {
                    skipOps.push_back(*it);
                }
            }
        }
    }
    state_.schedInfoMap[op].skipOps = skipOps;
}

Status OoOScheduler::InitOpCoreType(Operation* op)
{
    if (op == nullptr)
        return FAILED;
    std::unordered_map<int, CoreLocationType> opCoreMap = {
        {0, CoreLocationType::AIC}, {1, CoreLocationType::AIV0}, {2, CoreLocationType::AIV1}};
    CoreLocationType coreLocation;
    int internalSubgraphID = op->GetInternalSubgraphID();
    if (internalSubgraphID != -1) {
        coreLocation = opCoreMap[op->GetInternalSubgraphID()];
    } else if (op->GetCoreType() == CoreType::AIC) {
        coreLocation = CoreLocationType::AIC;
    } else if (op->GetCoreType() == CoreType::AIV) {
        coreLocation = CoreLocationType::AIV0;
    } else {
        // 对 ANY 类型进行处理
        if (op->GetOutputOperand(0)->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
            coreLocation = CoreLocationType::AIV0;
        } else if (op->GetOutputOperand(0)->GetMemoryTypeOriginal() <= MemoryType::MEM_BT) {
            coreLocation = CoreLocationType::AIC;
        } else if (op->GetOutputOperand(0)->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
            if (op->GetIOperands().size() == 0 ||
                op->GetInputOperand(0)->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
                coreLocation = CoreLocationType::AIC;
            } else if (op->GetInputOperand(0)->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
                coreLocation = CoreLocationType::AIV0;
            } else if (op->GetInputOperand(0)->GetMemoryTypeOriginal() <= MemoryType::MEM_BT) {
                coreLocation = CoreLocationType::AIC;
            } else {
                APASS_LOG_ERROR_F(Elements::Operation, "%s init coreLocation failed.", state_.GetOpInfo(op).c_str());
                return FAILED;
            }
        } else {
            APASS_LOG_ERROR_F(Elements::Operation, "%s init coreLocation failed.", state_.GetOpInfo(op).c_str());
            return FAILED;
        }
    }
    state_.schedInfoMap[op].coreLocation = coreLocation;
    return SUCCESS;
}

Status OoOScheduler::InitOpEntry(Operation* op)
{
    if (op == nullptr)
        return FAILED;

    // skip op 由消费者的 skipOps 带走; 输出在 DDR 上的没有消费者接管, 攒起来等定稿时按拓扑落位。
    if (IsSkipOp(*op)) {
        if (op->GetOutputOperand(0)->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
            ddrSkipOps_.push_back(op);
        }
        return SUCCESS;
    }
    if (state_.CheckOpBufferSize(op) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] CheckOpBufferSize failed! %s", op->GetOpcodeStr().c_str(),
                          op->GetOpMagic(), GetFormatBacktrace(*op).c_str());
        return FAILED;
    }

    // 初始化Operation属性到map
    int order = static_cast<int>(state_.orderedOps.size());
    state_.orderedOps.push_back(op);
    state_.schedInfoMap[op].execOrder = order;
    state_.schedInfoMap[op].pipeType = RescheduleUtils::GetOpPipeType(op);
    state_.schedInfoMap[op].isAlloc = (op->GetOpcodeStr().find("ALLOC") != std::string::npos);
    state_.schedInfoMap[op].isRetired = false;
    state_.SetOpMemIds(op, {});

    InitSkipOps(op);

    // 初始化核属性
    if (InitOpCoreType(op) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Operation %s init coreType failed!", state_.GetOpInfo(op).c_str());
        return FAILED;
    }

    APASS_LOG_DEBUG_F(Elements::Operation, "issue: %s, coreType: %s", state_.GetOpInfo(op).c_str(),
                      coreTypeToString(state_.schedInfoMap[op].coreLocation).c_str());
    return SUCCESS;
}

Status OoOScheduler::Init(const std::vector<Operation*>& opList,
                          const std::unordered_set<CoreLocationType> fixCoreConfig)
{
    state_.orderedOps.clear();
    state_.schedInfoMap.clear();
    state_.ClearAllOpMemIds();
    state_.localBufferMap.clear();
    ddrSkipOps_.clear();
    LOG_SCOPE_BEGIN(tInit, Elements::Function, "Init");
    // 初始化芯片各buffer大小
    state_.localMemSize = CommonUtils::GetLocalMemorySize();
    const std::string simtAttr = OP_ATTR_PREFIX + "requires_simt";
    for (auto& op : opList) {
        bool requiresSimt = op->HasAttribute(simtAttr) && op->GetBoolAttribute(simtAttr);
        if (requiresSimt) {
            FE_ASSERT(Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510)
                << "requires_simt is only supported on DAV_3510";
            state_.localMemSize[MemoryType::MEM_UB] = A5_SIMT_DYNAMIC_UB_SIZE;
            break;
        }
    }
    if (fixCoreConfig.empty()) {
        InitCoreConfig(opList);
    } else {
        state_.coreInitConfigs = fixCoreConfig;
    }
    // 校验并初始化Operation
    for (const auto& op : opList) {
        if (InitOpEntry(op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Operation %s[%d] init issue failed!", op->GetOpcodeStr().c_str(),
                              op->GetOpMagic());
            return FAILED;
        }
    }
    state_.numTotalIssues = state_.orderedOps.size();

    if (state_.InitBufRefCount(state_.orderedOps) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InitBufRefCount failed!");
        return FAILED;
    }
    // 初始化依赖关系
    if (state_.depManager.InitDependencies(state_.orderedOps, false) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InitDependencies failed!");
        return FAILED;
    }
    state_.depManager.PrintDependencies(state_.orderedOps);
    InitTensorCoreMap();
    // 初始化内存管理器
    InitIssueQueuesAndBufferManager();
    InitVfScopeGate();
    LOG_SCOPE_END(tInit);
    return SUCCESS;
}

void OoOScheduler::UpdateL0MXMap(const std::vector<Operation*>& opList)
{
    for (size_t i = 0; i < opList.size(); i++) {
        if (opList[i]->GetOpcode() == Opcode::OP_L1_TO_L0B_SCALE) {
            auto l0MxOut = opList[i]->GetOOperands()[0];
            auto consOp = *l0MxOut->GetConsumers().begin();
            LogicalTensorPtr l0ATensor;
            LogicalTensorPtr l0BTensor;
            LogicalTensorPtr l0AMXTensor;
            LogicalTensorPtr l0BMXTensor;
            for (auto& l0Tensor : consOp->GetIOperands()) {
                if (l0Tensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0A) {
                    l0ATensor = l0Tensor;
                } else if (l0Tensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0B) {
                    l0BTensor = l0Tensor;
                } else if (l0Tensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0AMX) {
                    l0AMXTensor = l0Tensor;
                } else if (l0Tensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0BMX) {
                    l0BMXTensor = l0Tensor;
                }
            }
            dualDstEngine_.GetL02L0MXMap()[l0ATensor] = l0AMXTensor;
            dualDstEngine_.GetL02L0MXMap()[l0BTensor] = l0BMXTensor;
        }
    }
}

void OoOScheduler::UpdateDualDstL0MXRange()
{
    constexpr int kL0mxAddrShiftBits = 4;
    for (auto& entry : dualDstEngine_.GetL02L0MXMap()) {
        auto l0Tensor = entry.first;
        auto l0MXTensor = entry.second;
        int l0MemID = l0Tensor->memoryrange.memId;
        int l0MemMXID = l0MXTensor->memoryrange.memId;
        l0MXTensor->memoryrange = TileRange(state_.localBufferMap[l0MemID]->start >> kL0mxAddrShiftBits,
                                            state_.localBufferMap[l0MemID]->end >> kL0mxAddrShiftBits, l0MemMXID);
    }
}

// 紧跟最后一个生产者插入: 位置唯一确定, 且不会把 skip op 推得比必要更晚。
// 一个生产者都不在序列里的留在头部 —— 无依赖可违背。
void OoOScheduler::PlaceDdrSkipOps()
{
    for (auto* skipOp : ddrSkipOps_) {
        // 已标删的不再落位: 插进 newOperations 就成了"既发射又删除", 数量守恒当场不成立。
        if (skipOp->IsDeleted()) {
            continue;
        }
        size_t insertPos = 0;
        for (auto iOperand : skipOp->GetIOperands()) {
            if (iOperand == nullptr) {
                continue;
            }
            for (auto* prod : iOperand->GetProducers()) {
                // 生产者自己也可能不参与调度 (USE_LESS_OPS), 不在序列里就没有位置约束可言。
                auto it = std::find(state_.newOperations.begin(), state_.newOperations.end(), prod);
                if (it == state_.newOperations.end()) {
                    continue;
                }
                size_t prodPos = static_cast<size_t>(std::distance(state_.newOperations.begin(), it));
                insertPos = std::max(insertPos, prodPos + 1);
            }
        }
        state_.newOperations.insert(state_.newOperations.begin() + static_cast<long>(insertPos), skipOp);
        APASS_LOG_DEBUG_F(Elements::Operation, "Place DDR skip op %s at %zu.", state_.GetOpInfo(skipOp).c_str(),
                          insertPos);
    }
}

Status OoOScheduler::FinalizeScheduleResult(const std::vector<Operation*>& opList)
{
    PlaceDdrSkipOps();
    // 消费者可能在链标删之前就发射过, 那时 HandleSkipOp 已把它写进 newOperations。
    auto& newOps = state_.newOperations;
    newOps.erase(std::remove_if(newOps.begin(), newOps.end(), [](Operation* op) { return op->IsDeleted(); }),
                 newOps.end());
    function_.EraseOperations(false, false);
    UpdateL0MXMap(opList);
    UpdateDualDstL0MXRange();
    PrintOpList(state_.newOperations);
    if (dualDstEngine_.VerifyDualDstSameOffset() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "VerifyDualDstSameOffset failed.");
        return FAILED;
    }
    function_.SetStackWorkespaceSize(state_.workspaceOffset);
    function_.pipeEndTime = state_.pipeEndTime;
    return SUCCESS;
}
Status OoOScheduler::Schedule(const std::vector<Operation*>& opList,
                              const std::unordered_set<CoreLocationType> fixCoreConfig)
{
    struct EndGuard {
        OoOScheduler* self;
        bool success{false};
        ~EndGuard() { self->NotifyScheduleEnd(success); }
    } guard{this};
    if (opList.empty()) {
        guard.success = true;
        return SUCCESS;
    }

    if (Init(opList, fixCoreConfig) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Init failed!");
        return FAILED;
    }
    AllocWorkspaceGM(opList);
    UpdateIssueExecOrder();
    if (dualDstEngine_.RunDualDstFuse(state_.orderedOps) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "RunDualDstFuse failed!");
        return FAILED;
    }
    NotifyInitDDRBuffers();
    if (SeqSchedule() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "GenSpillSchedule failed!");
        return FAILED;
    }
    if (ScheduleMainLoop() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "ScheduleMainLoop failed!");
        return FAILED;
    }
    if (CheckAndUpdateLifecycle() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "CheckAndUpdateLifecycle failed!");
        return FAILED;
    }
    if (FinalizeScheduleResult(opList) != SUCCESS)
        return FAILED;
    guard.success = true;
    return SUCCESS;
}

void OoOScheduler::AllocWorkspaceGM(const std::vector<Operation*>& opList)
{
    std::set<int> allocedRawmagic;
    for (auto& inCast : function_.GetIncast()) {
        allocedRawmagic.insert(inCast->tensor->GetRawMagic());
    }
    for (auto& outCast : function_.GetOutcast()) {
        allocedRawmagic.insert(outCast->tensor->GetRawMagic());
    }
    std::map<int, TileRange> rawMagicRange;
    std::map<int, int64_t> rawMagicOffset;
    for (auto& op : opList) {
        for (auto& iOperand : op->GetIOperands()) {
            if (allocedRawmagic.count(iOperand->tensor->GetRawMagic()) &&
                rawMagicRange.count(iOperand->tensor->GetRawMagic())) {
                iOperand->memoryrange = rawMagicRange[iOperand->tensor->GetRawMagic()];
                iOperand->SetAttr(OpAttributeKey::workspaceBaseOffset, rawMagicOffset[iOperand->tensor->GetRawMagic()]);
            } else if (iOperand->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
                       !allocedRawmagic.count(iOperand->tensor->GetRawMagic())) {
                allocedRawmagic.insert(iOperand->tensor->GetRawMagic());
                iOperand->SetAttr(OpAttributeKey::workspaceBaseOffset, state_.workspaceOffset);
                iOperand->memoryrange = TileRange(state_.workspaceOffset,
                                                  state_.workspaceOffset + iOperand->tensor->GetRawDataSize(),
                                                  iOperand->tensor->GetRawMagic());
                rawMagicOffset[iOperand->tensor->GetRawMagic()] = state_.workspaceOffset;
                state_.workspaceOffset += iOperand->tensor->GetRawDataSize();
                rawMagicRange[iOperand->tensor->GetRawMagic()] = iOperand->memoryrange;
                ddrKindMap_[iOperand->memoryrange.memId] = DDRBufferKind::FUNCTION_TEMP;
            }
        }
        for (auto& oOperand : op->GetOOperands()) {
            if (allocedRawmagic.count(oOperand->tensor->GetRawMagic()) &&
                rawMagicRange.count(oOperand->tensor->GetRawMagic())) {
                oOperand->memoryrange = rawMagicRange[oOperand->tensor->GetRawMagic()];
                oOperand->SetAttr(OpAttributeKey::workspaceBaseOffset, rawMagicOffset[oOperand->tensor->GetRawMagic()]);
            } else if (oOperand->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
                       !allocedRawmagic.count(oOperand->tensor->GetRawMagic())) {
                allocedRawmagic.insert(oOperand->tensor->GetRawMagic());
                oOperand->SetAttr(OpAttributeKey::workspaceBaseOffset, state_.workspaceOffset);
                oOperand->memoryrange = TileRange(state_.workspaceOffset,
                                                  state_.workspaceOffset + oOperand->tensor->GetRawDataSize(),
                                                  oOperand->tensor->GetRawMagic());
                rawMagicOffset[oOperand->tensor->GetRawMagic()] = state_.workspaceOffset;
                state_.workspaceOffset += oOperand->tensor->GetRawDataSize();
                rawMagicRange[oOperand->tensor->GetRawMagic()] = oOperand->memoryrange;
                ddrKindMap_[oOperand->memoryrange.memId] = DDRBufferKind::FUNCTION_TEMP;
            }
        }
    }
}

bool OoOScheduler::HasEnoughBuffer(Operation* allocOp, MemoryType memType)
{
    return !state_.bufferManagerMap[state_.schedInfoMap[allocOp].coreLocation][memType].IsFull(
        state_.localBufferMap[state_.opReqMemIdsMap[allocOp][0]], false);
}

Status OoOScheduler::RearrangeBuffer(Operation* allocOp, MemoryType memType)
{
    std::vector<int> memIds = state_.bufferManagerMap[state_.schedInfoMap[allocOp].coreLocation][memType]
                                  .GetAddrSortedBufs();
    for (auto memId : memIds) {
        auto op = spillEngine_.GetSpillOp(memId);
        if (op == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d] latest write op.", memId);
            return FAILED;
        }
        if (op->GetOpcodeStr().find("ALLOC") == std::string::npos) {
            return SUCCESS;
        }
    }
    std::vector<BufferAddrChange> changes;
    auto status = state_.bufferManagerMap[state_.schedInfoMap[allocOp].coreLocation][memType].CompactBufferSlices(
        state_.localBufferMap, changes);
    if (status == SUCCESS) {
        NotifyBufferRearrange(allocOp, memType, std::move(changes));
    }
    return status;
}

std::vector<std::vector<int>> OoOScheduler::GetSpillGroup(BufferPool& pool, size_t sizeNeedSpill,
                                                          const std::vector<std::pair<uint64_t, uint64_t>>& avoidRanges)
{
    return pool.GetSpillGroup(sizeNeedSpill, avoidRanges);
}

std::vector<OoOScheduler::DualSpillGroup> OoOScheduler::GetDualSpillGroup(BufferPool& poolA, BufferPool& poolB,
                                                                          size_t sizeNeedSpill)
{
    std::vector<DualSpillGroup> result;
    auto bufsA = poolA.GetSortedAllocatedBufs();
    auto bufsB = poolB.GetSortedAllocatedBufs();

    size_t iA = 0;
    while (iA < bufsA.size()) {
        size_t startAddrA = poolA.ObtainStartAddr(iA, bufsA);
        if ((poolA.GetMemSize() - startAddrA) < sizeNeedSpill) {
            break;
        }
        size_t jA = poolA.UpdateIdx(iA, sizeNeedSpill, startAddrA, bufsA);
        size_t iB = 0;
        while (iB < bufsB.size()) {
            size_t startAddrB = poolB.ObtainStartAddr(iB, bufsB);
            if (startAddrB != startAddrA) {
                if (startAddrB < startAddrA) {
                    iB++;
                    continue;
                }
                break;
            }
            if ((poolB.GetMemSize() - startAddrB) < sizeNeedSpill) {
                break;
            }
            size_t jB = poolB.UpdateIdx(iB, sizeNeedSpill, startAddrB, bufsB);
            if (jA < iA || jB < iB || (jA == iA && jB == iB)) {
                iB++;
                continue;
            }
            DualSpillGroup group;
            group.aiv0MemIds.reserve(jA - iA);
            group.aiv1MemIds.reserve(jB - iB);
            for (size_t k = iA; k < jA; k++) {
                group.aiv0MemIds.push_back(std::get<0>(bufsA[k]));
            }
            for (size_t k = iB; k < jB; k++) {
                group.aiv1MemIds.push_back(std::get<0>(bufsB[k]));
            }
            result.push_back(std::move(group));
            iB++;
        }
        iA++;
    }
    return result;
}

Status OoOScheduler::GetGroupNextUseTime(std::vector<int> group, Operation* allocOp, std::vector<int>& groupNextUseTime,
                                         std::unordered_map<int, size_t>& nextUseTimeCache)
{
    size_t minNextUseTime = INT_MAX;
    for (auto& memId : group) {
        Operation* spillOp = spillEngine_.GetSpillOp(memId);
        if (spillOp == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d] occupy op.", memId);
            return FAILED;
        }
        if (spillEngine_.IsBelongSpillBlackList(memId, allocOp)) {
            groupNextUseTime.push_back(-1);
            return SUCCESS;
        }
        if (nextUseTimeCache.find(memId) != nextUseTimeCache.end()) {
            minNextUseTime = std::min(minNextUseTime, nextUseTimeCache[memId]);
        } else {
            int nextUseTime = spillEngine_.GetBufNextUseTime(memId);
            if (nextUseTime == -1) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find Tensor[%d] next used time.", memId);
                return FAILED;
            }
            nextUseTimeCache[memId] = static_cast<size_t>(nextUseTime);
            minNextUseTime = std::min(minNextUseTime, static_cast<size_t>(nextUseTime));
        }
    }
    groupNextUseTime.push_back(minNextUseTime);
    return SUCCESS;
}

std::vector<int> OoOScheduler::SelectSpillBuffers(Operation* allocOp)
{
    LocalBufferPtr allocBuffer = state_.localBufferMap[state_.opReqMemIdsMap[allocOp][0]];
    auto coreType = state_.schedInfoMap[allocOp].coreLocation;
    auto& pool = state_.bufferManagerMap[coreType][allocBuffer->memType];
    // vf 受限分配的 avoidRanges: SeqSchedule 阶段登记表尚未登记, 天然为空。
    std::vector<std::pair<uint64_t, uint64_t>> avoidRanges;
    if (allocOp->GetAtomicScopeId() >= VF_CLUSTER_ID_START) {
        auto it = scopeTensorRanges_.find(allocOp->GetAtomicScopeId());
        if (it != scopeTensorRanges_.end()) {
            avoidRanges = it->second;
        }
    }
    std::vector<std::vector<int>> canSpillGroups = GetSpillGroup(pool, allocBuffer->size, avoidRanges);
    if (canSpillGroups.empty()) {
        return pool.GetAddrSortedBufs();
    }
    std::unordered_map<int, size_t> nextUseTimeCache;
    std::vector<int> groupNextUseTime;
    for (auto& group : canSpillGroups) {
        if (GetGroupNextUseTime(group, allocOp, groupNextUseTime, nextUseTimeCache) != SUCCESS) {
            APASS_LOG_WARN_F(Elements::Operation, "Get group next use time failed, begin spill all tensor.");
            return pool.GetAddrSortedBufs();
        }
    }
    size_t groupSel = std::max_element(groupNextUseTime.begin(), groupNextUseTime.end()) - groupNextUseTime.begin();
    if (groupNextUseTime[groupSel] == -1) {
        APASS_LOG_WARN_F(Elements::Tensor, "Cannot find tensor to spill, begin spill all tensor.");
        return pool.GetAddrSortedBufs();
    }
    return canSpillGroups[groupSel];
}

Status OoOScheduler::GenBufferSpill(Operation* allocOp, SpillContext& ctx, bool isMainLoop)
{
    auto reqMemType = state_.localBufferMap[state_.opReqMemIdsMap[allocOp][0]]->memType;
    auto reqSize = state_.localBufferMap[state_.opReqMemIdsMap[allocOp][0]]->size;
    std::vector<int> spillGroup;
    SpillContext peerCtx;
    Operation* peerAlloc = nullptr;
    DualSpillGroup dualGroup;
    bool useDualGroup = !isMainLoop && state_.IsDualDstAlloc(allocOp);
    if (useDualGroup) {
        peerAlloc = state_.schedInfoMap[allocOp].pairedDualDstAlloc;
        if (SelectDualDstSpillTargets(allocOp, peerAlloc, dualGroup) != SUCCESS)
            return FAILED;
    } else {
        spillGroup = SelectSpillBuffers(allocOp);
    }
    if ((!useDualGroup && spillGroup.empty()) ||
        (useDualGroup && dualGroup.aiv0MemIds.empty() && dualGroup.aiv1MemIds.empty())) {
        APASS_LOG_ERROR_F(Elements::Operation, "Select buffer to spill failed.");
        NotifyAllocFail(allocOp, reqMemType, reqSize);
        return FAILED;
    }
    // DualDst 需要共同 offset，不使用单侧 compact 和容量检查。
    if (useDualGroup) {
        if (ApplyDualSpill(allocOp, peerAlloc, dualGroup, ctx, peerCtx, reqMemType, reqSize) != SUCCESS)
            return FAILED;
    } else {
        if (ApplySingleSpill(allocOp, spillGroup, ctx, reqMemType, reqSize) != SUCCESS)
            return FAILED;
        if (FinalizeSingleSideSpill(allocOp, reqMemType, reqSize) != SUCCESS)
            return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::SelectDualDstSpillTargets(Operation* allocOp, Operation* peerAlloc, DualSpillGroup& dualGroup)
{
    DualDstAllocCtx allocCtx;
    if (peerAlloc == nullptr || dualDstEngine_.ResolveDualDstAllocCtx(allocOp, allocCtx) != SUCCESS)
        return FAILED;
    auto& pool0 = state_.bufferManagerMap[CoreLocationType::AIV0][MemoryType::MEM_UB];
    auto& pool1 = state_.bufferManagerMap[CoreLocationType::AIV1][MemoryType::MEM_UB];
    auto candidates = GetDualSpillGroup(pool0, pool1, allocCtx.bufA->size);
    if (candidates.empty()) {
        dualGroup.aiv0MemIds = pool0.GetAddrSortedBufs();
        dualGroup.aiv1MemIds = pool1.GetAddrSortedBufs();
    } else {
        long long bestScore = -1;
        for (auto& candidate : candidates) {
            std::vector<int> score0;
            std::vector<int> score1;
            std::unordered_map<int, size_t> cache0;
            std::unordered_map<int, size_t> cache1;
            bool allocOnAiv0 = state_.schedInfoMap[allocOp].coreLocation == CoreLocationType::AIV0;
            Operation* alloc0 = allocOnAiv0 ? allocOp : peerAlloc;
            Operation* alloc1 = allocOnAiv0 ? peerAlloc : allocOp;
            if (GetGroupNextUseTime(candidate.aiv0MemIds, alloc0, score0, cache0) != SUCCESS ||
                GetGroupNextUseTime(candidate.aiv1MemIds, alloc1, score1, cache1) != SUCCESS || score0.empty() ||
                score1.empty()) {
                continue;
            }
            long long score = std::min<long long>(score0.back(), score1.back());
            if (score > bestScore) {
                bestScore = score;
                dualGroup = candidate;
            }
        }
        if (bestScore < 0) {
            dualGroup.aiv0MemIds = pool0.GetAddrSortedBufs();
            dualGroup.aiv1MemIds = pool1.GetAddrSortedBufs();
        }
    }
    return SUCCESS;
}

Status OoOScheduler::ApplyDualSpill(Operation* allocOp, Operation* peerAlloc, const DualSpillGroup& dualGroup,
                                    SpillContext& ctx, SpillContext& peerCtx, MemoryType reqMemType, uint64_t reqSize)
{
    auto spillOneSide = [this](const std::vector<int>& memIds, Operation* trigger, SpillContext& spillCtx) {
        spillCtx.spillMemIds = memIds;
        for (int memId : memIds) {
            if (spillEngine_.SpillBuffer(memId, trigger, spillCtx) != SUCCESS)
                return FAILED;
        }
        return SUCCESS;
    };
    Operation* alloc0 = state_.schedInfoMap[allocOp].coreLocation == CoreLocationType::AIV0 ? allocOp : peerAlloc;
    Operation* alloc1 = alloc0 == allocOp ? peerAlloc : allocOp;
    SpillContext& ctx0 = alloc0 == allocOp ? ctx : peerCtx;
    SpillContext& ctx1 = alloc1 == allocOp ? ctx : peerCtx;
    if (spillOneSide(dualGroup.aiv0MemIds, alloc0, ctx0) != SUCCESS ||
        spillOneSide(dualGroup.aiv1MemIds, alloc1, ctx1) != SUCCESS) {
        NotifyAllocFail(allocOp, reqMemType, reqSize);
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::ApplySingleSpill(Operation* allocOp, const std::vector<int>& spillGroup, SpillContext& ctx,
                                      MemoryType reqMemType, uint64_t reqSize)
{
    ctx.spillMemIds = spillGroup;
    for (auto& memId : spillGroup) {
        if (spillEngine_.SpillBuffer(memId, allocOp, ctx) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Spill tensor[%d] for %s failed!", memId,
                              state_.GetOpInfo(allocOp).c_str());
            NotifyAllocFail(allocOp, reqMemType, reqSize);
            return FAILED;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::FinalizeSingleSideSpill(Operation* allocOp, MemoryType reqMemType, uint64_t reqSize)
{
    if (RearrangeBuffer(allocOp, reqMemType) != SUCCESS) {
        APASS_LOG_WARN_F(Elements::Operation, "RearrangeBuffer failed for %s.", GetFormatBacktrace(*allocOp).c_str());
    }
    if (!HasEnoughBuffer(allocOp, reqMemType)) {
        APASS_LOG_ERROR_F(Elements::Operation, "Spill all buffer failed! %s", GetFormatBacktrace(*allocOp).c_str());
        if (PrintSpillFailedInfo(allocOp) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PrintSpillFailedInfo failed!");
            NotifyAllocFail(allocOp, reqMemType, reqSize);
            return FAILED;
        }
        APASS_LOG_ERROR_F(
            Elements::Operation,
            "Possible causes: incorrect memory reuse, memory fragmentation, or spill not supported for "
            "L0C_COPY_TO_L1."
            "Please check tile shape and OOO spill failed info. Consider avoiding cube-aligned matrix sizes.");
        NotifyAllocFail(allocOp, reqMemType, reqSize);
        return FAILED;
    }
    return SUCCESS;
}

// Inserts the copyout's skip chain (reshape/view/viewType producers, if not already
// present) starting at copyoutPos, in order. Returns the position right after the last
// inserted skip op -- where the copyout itself must go so it lands after its producers.
size_t OoOScheduler::InsertSkipOpsBeforeCopyout(Operation* copyoutOp, size_t copyoutPos)
{
    auto& skipOps = state_.schedInfoMap[copyoutOp].skipOps;
    size_t insertPos = copyoutPos;
    for (auto* skipOp : skipOps) {
        auto existing = std::find(state_.newOperations.begin(), state_.newOperations.end(), skipOp);
        if (existing == state_.newOperations.end()) {
            state_.newOperations.insert(state_.newOperations.begin() + static_cast<long>(insertPos), skipOp);
            insertPos++;
        }
    }
    return insertPos;
}

Status OoOScheduler::InsertSpillCopyoutOp(Operation* copyoutOp)
{
    int spillMemid = copyoutOp->GetInputOperand(0)->memoryrange.memId;
    Operation* insertOp = nullptr;
    for (auto op : state_.newOperations) {
        auto& reqMemids = state_.GetOpMemIds(op);
        if (std::find(reqMemids.begin(), reqMemids.end(), spillMemid) != reqMemids.end()) {
            insertOp = op;
        }
    }
    if (insertOp == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Insert spillout %s failed.", state_.GetOpInfo(copyoutOp).c_str());
        return FAILED;
    }
    if (insertOp->GetAtomicScopeId() >= VF_CLUSTER_ID_START) {
        int scopeId = insertOp->GetAtomicScopeId();
        auto opsIt = state_.vfScopeOpsByScope.find(scopeId);
        if (opsIt == state_.vfScopeOpsByScope.end() || opsIt->second.empty()) {
            APASS_LOG_ERROR_F(Elements::Operation, "Insert spillout %s failed: unknown vf scope %d.",
                              state_.GetOpInfo(copyoutOp).c_str(), scopeId);
            return FAILED;
        }
        std::unordered_set<Operation*> scopeOpSet(opsIt->second.begin(), opsIt->second.end());
        Operation* scopeLastOp = nullptr;
        size_t foundCnt = 0;
        for (auto op : state_.newOperations) {
            if (scopeOpSet.count(op) > 0) {
                foundCnt++;
                scopeLastOp = op;
            }
        }
        if (foundCnt != scopeOpSet.size()) {
            APASS_LOG_ERROR_F(Elements::Operation, "Insert spillout %s failed: vf scope %d not fully launch(%zu/%zu).",
                              state_.GetOpInfo(copyoutOp).c_str(), scopeId, foundCnt, scopeOpSet.size());
            return FAILED;
        }
        auto lastIt = find(state_.newOperations.begin(), state_.newOperations.end(), scopeLastOp);
        size_t copyoutPos = static_cast<size_t>(std::distance(state_.newOperations.begin(), lastIt)) + 1;
        size_t insertPos = InsertSkipOpsBeforeCopyout(copyoutOp, copyoutPos);
        state_.newOperations.insert(state_.newOperations.begin() + static_cast<long>(insertPos), copyoutOp);
        return SUCCESS;
    }
    auto it = find(state_.newOperations.begin(), state_.newOperations.end(), insertOp);
    if (it != state_.newOperations.end()) {
        size_t copyoutPos = static_cast<size_t>(std::distance(state_.newOperations.begin(), it)) + 1;
        size_t insertPos = InsertSkipOpsBeforeCopyout(copyoutOp, copyoutPos);
        state_.newOperations.insert(state_.newOperations.begin() + static_cast<long>(insertPos), copyoutOp);
    }
    return SUCCESS;
}

Status OoOScheduler::ApplySpillContext(SpillContext& ctx, Operation* allocOp)
{
    for (auto* copyoutOp : ctx.newCopyoutOps) {
        if (InsertSpillCopyoutOp(copyoutOp) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "ApplySpillContext: insert copyout %s failed.",
                              state_.GetOpInfo(copyoutOp).c_str());
            return FAILED;
        }
    }
    state_.numTotalIssues += ctx.newNotRetiredCopyOutSize - ctx.deleteNotRetiredOpSize;
    MemoryType memType = allocOp->GetOutputOperand(0)->GetMemoryTypeOriginal();

    for (auto deleteOp : ctx.deleteAllocOps) {
        state_.allocIssueQueue[std::get<2>(deleteOp)][std::get<1>(deleteOp)].DeleteOp(std::get<0>(deleteOp));
    }
    for (auto& newAllocOp : ctx.newAllocOps) {
        state_.allocIssueQueue[state_.schedInfoMap[newAllocOp].coreLocation][memType].Insert(newAllocOp);
    }

    for (auto memId : ctx.spillMemIds) {
        state_.localBufferMap[memId]->retireCycle = state_.clock;
    }
    return SUCCESS;
}

Status OoOScheduler::PrintSpillFailedInfo(Operation* allocOp)
{
    auto memType = state_.localBufferMap[state_.GetOpMemIds(allocOp)[0]]->memType;
    APASS_LOG_ERROR_F(Elements::Operation, "======== OoO Spill failed info ===========");
    APASS_LOG_ERROR_F(Elements::Operation, "Spill failed memoryType: %s. %s", MemoryTypeToString(memType).c_str(),
                      GetFormatBacktrace(*allocOp).c_str());

    APASS_LOG_ERROR_F(Elements::Operation, "---- alloc request ----");
    APASS_LOG_ERROR_F(Elements::Operation, "op:%s need buffer size: %lu. %s", state_.GetOpInfo(allocOp).c_str(),
                      state_.localBufferMap[state_.GetOpMemIds(allocOp)[0]]->size,
                      GetFormatBacktrace(*allocOp).c_str());

    APASS_LOG_ERROR_F(Elements::Operation, "---- current buffer occupancy ----");
    for (auto& [memId, occupyOp] : state_.tensorOccupyMap) {
        if (state_.localBufferMap[memId]->memType != memType) {
            continue;
        }
        APASS_LOG_ERROR_F(Elements::Operation, "Tensor[%d], size:%lu, range[%lu,%lu], last writer: %s. %s", memId,
                          state_.localBufferMap[memId]->size, state_.localBufferMap[memId]->start,
                          state_.localBufferMap[memId]->end, state_.GetOpInfo(occupyOp).c_str(),
                          GetFormatBacktrace(*occupyOp).c_str());
    }
    return SUCCESS;
}

} // namespace npu::tile_fwk
