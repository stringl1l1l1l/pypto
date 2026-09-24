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
 * \file dualdst_engine.cpp
 * \brief DualDstEngine 实现：OP_L0C_COPY_UB 到 OP_L0C_COPY_UB_DUAL_DST 的融合识别、图改写和 UB 双池分配。
 */

#include "dualdst_engine.h"
#include "interface/operation/attribute.h"
#include <sstream>
#include <string>

#ifdef MODULE_NAME
#undef MODULE_NAME
#endif
#define MODULE_NAME "DualDstEngine"

namespace npu::tile_fwk {

namespace {

bool ShapeEq(const std::vector<int64_t>& a, const std::vector<int64_t>& b) { return a == b; }

bool DynShapeEq(const std::vector<SymbolicScalar>& a, const std::vector<SymbolicScalar>& b)
{
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i].Dump() == b[i].Dump())
            continue;
        if (a[i].ConcreteValid() && b[i].ConcreteValid() && a[i].Concrete() == b[i].Concrete()) {
            continue;
        }
        return false;
    }
    return true;
}

} // namespace

int64_t DualDstEngine::SpecifiedInt(const OpImmediate& imm)
{
    if (!imm.IsSpecified()) {
        return kInvalidCoord;
    }
    const auto& s = imm.GetSpecifiedValue();
    if (!s.ConcreteValid()) {
        return kInvalidCoord;
    }
    return s.Concrete();
}

bool DualDstEngine::ReadGeometry(Operation* op, CopyUbGeometry& g)
{
    if (op == nullptr)
        return false;
    if (op->GetIOperands().size() != 1 || op->GetOOperands().size() != 1)
        return false;
    auto attr = std::dynamic_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
    if (attr == nullptr)
        return false;

    const auto& fromOff = attr->GetFromOffset();
    if (fromOff.size() != kCopyUbGeometryDimCount)
        return false;
    g.fromM = SpecifiedInt(fromOff[0]);
    g.fromN = SpecifiedInt(fromOff[1]);
    if (g.fromM == kInvalidCoord || g.fromN == kInvalidCoord)
        return false;

    const auto& shape = attr->GetShape();
    if (shape.size() != kCopyUbGeometryDimCount)
        return false;
    g.tileM = SpecifiedInt(shape[0]);
    g.tileN = SpecifiedInt(shape[1]);
    if (g.tileM <= 0 || g.tileN <= 0)
        return false;

    g.ubOut = op->GetOutputOperand(0);
    if (g.ubOut == nullptr)
        return false;
    g.ubShape = g.ubOut->GetShape();
    if (op->HasAttribute(OpAttributeKey::staticValidShape)) {
        auto staticVals = op->GetVectorIntAttribute<int64_t>(OpAttributeKey::staticValidShape);
        g.ubValidShape.clear();
        g.ubValidShape.reserve(staticVals.size());
        for (auto v : staticVals)
            g.ubValidShape.emplace_back(v);
    } else {
        APASS_LOG_INFO_F(Elements::Operation, "DualDst op[%d] is dynValidshape", op->GetOpMagic());
        g.ubValidShape = g.ubOut->GetDynValidShape();
    }
    return true;
}

bool DualDstEngine::LoadGeometries(const std::vector<Operation*>& copyUbs, std::vector<CopyUbGeometry>& geos)
{
    geos.assign(copyUbs.size(), CopyUbGeometry{});
    int okCnt = 0;
    for (size_t i = 0; i < copyUbs.size(); i++) {
        if (ReadGeometry(copyUbs[i], geos[i]))
            okCnt++;
    }
    return okCnt >= kMinDualDstPairCount;
}

// ===== 调度后诊断：DUAL_DST 同 offset 不变量校验 =====
// 如果最终 offset 不等，跨 AIV 读取可能拿到错误语义的张量。
Status DualDstEngine::VerifyDualDstSameOffset()
{
    if (!state_.enableDualDst)
        return SUCCESS;
    int total = 0;
    int matched = 0;
    int mismatched = 0;
    for (auto* op : state_.newOperations) {
        if (op == nullptr)
            continue;
        if (op->GetOpcode() != Opcode::OP_L0C_COPY_UB_DUAL_DST)
            continue;
        const auto& outs = op->GetOOperands();
        if (outs.size() != kDualDstOutputCount)
            continue;
        if (outs[0] == nullptr || outs[1] == nullptr)
            continue;
        int memIdA = outs[0]->memoryrange.memId;
        int memIdB = outs[1]->memoryrange.memId;
        auto itA = state_.localBufferMap.find(memIdA);
        auto itB = state_.localBufferMap.find(memIdB);
        if (itA == state_.localBufferMap.end() || itB == state_.localBufferMap.end())
            continue;
        if (itA->second == nullptr || itB->second == nullptr)
            continue;
        size_t offA = itA->second->start;
        size_t offB = itB->second->start;
        size_t sizeA = itA->second->size;
        size_t sizeB = itB->second->size;
        total++;
        if (offA == offB) {
            matched++;
        } else {
            mismatched++;
            CoreLocationType coreA = state_.ResolveCoreForFree(memIdA);
            CoreLocationType coreB = state_.ResolveCoreForFree(memIdB);
            APASS_LOG_ERROR_F(Elements::Operation,
                              "[dualdst-verify] mismatch op=%d | memIdA=%d coreA=%d off=%zu size=%zu | "
                              "memIdB=%d coreB=%d off=%zu size=%zu",
                              op->GetOpMagic(), memIdA, static_cast<int>(coreA), offA, sizeA, memIdB,
                              static_cast<int>(coreB), offB, sizeB);
        }
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "[dualdst-verify] DUAL_DST total=%d matched=%d mismatched=%d", total,
                      matched, mismatched);
    return (mismatched > 0) ? FAILED : SUCCESS;
}

void DualDstEngine::GreedyNonOverlapPick(std::vector<CandidatePair>& cands, std::vector<CandidatePair>& picked)
{
    std::sort(cands.begin(), cands.end(),
              [](const CandidatePair& a, const CandidatePair& b) { return a.earlyOffsetOnAxis < b.earlyOffsetOnAxis; });
    std::unordered_set<Operation*> used;
    for (auto& c : cands) {
        if (used.count(c.opEarly) || used.count(c.opLate))
            continue;
        picked.push_back(c);
        used.insert(c.opEarly);
        used.insert(c.opLate);
    }
}

CoreLocationType DualDstEngine::ConsumerCore(Operation* copyUbOp)
{
    auto out = copyUbOp->GetOutputOperand(0);
    if (out == nullptr)
        return CoreLocationType::UNKNOWN;
    const auto& cons = out->GetConsumers();
    if (cons.empty())
        return CoreLocationType::UNKNOWN;
    Operation* cur = *cons.begin();
    for (int hop = 0; hop < kMaxConsumerSearchDepth && cur != nullptr; ++hop) {
        auto it = state_.schedInfoMap.find(cur);
        if (it != state_.schedInfoMap.end() &&
            (it->second.coreLocation == CoreLocationType::AIV0 || it->second.coreLocation == CoreLocationType::AIV1)) {
            return it->second.coreLocation;
        }
        if (cur->GetOOperands().empty())
            break;
        auto outT = cur->GetOutputOperand(0);
        if (outT == nullptr)
            break;
        const auto& nextCons = outT->GetConsumers();
        if (nextCons.size() != 1)
            break;
        cur = *nextCons.begin();
    }
    auto it = state_.schedInfoMap.find(*cons.begin());
    return (it == state_.schedInfoMap.end()) ? CoreLocationType::UNKNOWN : it->second.coreLocation;
}

Operation* DualDstEngine::FindAllocPred(Operation* op)
{
    for (auto* pre : state_.depManager.GetPredecessors(op)) {
        if (pre == nullptr)
            continue;
        auto it = state_.schedInfoMap.find(pre);
        if (it != state_.schedInfoMap.end() && it->second.isAlloc) {
            return pre;
        }
    }
    return nullptr;
}

// 最终融合判定比 core_assign.cpp::Adjacent2D 的亲和预筛选更严格；修改基础二维相邻规则时需同步两处。
void DualDstEngine::BuildAdjacencyCandidates(const std::vector<Operation*>& copyUbs,
                                             const std::vector<CopyUbGeometry>& geos, std::vector<CandidatePair>& candM,
                                             std::vector<CandidatePair>& candN)
{
    auto consumerSplit = [this](Operation* early, Operation* late) {
        return ConsumerCore(early) == CoreLocationType::AIV0 && ConsumerCore(late) == CoreLocationType::AIV1;
    };
    for (size_t i = 0; i < copyUbs.size(); i++) {
        if (geos[i].tileM <= 0)
            continue;
        for (size_t j = i + 1; j < copyUbs.size(); j++) {
            if (geos[j].tileM <= 0)
                continue;
            const auto& a = geos[i];
            const auto& b = geos[j];
            if (!ShapeEq(a.ubShape, b.ubShape))
                continue;
            if (!DynShapeEq(a.ubValidShape, b.ubValidShape))
                continue;
            if (a.tileM != b.tileM || a.tileN != b.tileN)
                continue;
            const int64_t tileM = a.tileM;
            const int64_t tileN = a.tileN;
            Operation* opA = copyUbs[i];
            Operation* opB = copyUbs[j];
            if (a.fromN == b.fromN && std::abs(a.fromM - b.fromM) == tileM) {
                Operation* early = (a.fromM < b.fromM) ? opA : opB;
                Operation* late = (a.fromM < b.fromM) ? opB : opA;
                if (consumerSplit(early, late)) {
                    candM.push_back({early, late, std::min(a.fromM, b.fromM)});
                }
            }
            if (a.fromM == b.fromM && std::abs(a.fromN - b.fromN) == tileN) {
                Operation* early = (a.fromN < b.fromN) ? opA : opB;
                Operation* late = (a.fromN < b.fromN) ? opB : opA;
                if (consumerSplit(early, late)) {
                    candN.push_back({early, late, std::min(a.fromN, b.fromN)});
                }
            }
        }
    }
}

bool DualDstEngine::IsAivUbAllocAlignmentCheckEnabled() const
{
    return state_.enableDualDst && state_.coreInitConfigs.find(CoreLocationType::AIV1) != state_.coreInitConfigs.end();
}

Status DualDstEngine::ResolveAivUbAllocRecordInput(Operation* allocOp, bool& shouldCheck,
                                                   CoreLocationType& coreLocation, int& memId, LocalBufferPtr& buf)
{
    shouldCheck = false;
    memId = -1;
    buf = nullptr;
    if (!IsAivUbAllocAlignmentCheckEnabled()) {
        return SUCCESS;
    }
    if (allocOp == nullptr || state_.IsDualDstAlloc(allocOp)) {
        return SUCCESS;
    }
    auto schedIt = state_.schedInfoMap.find(allocOp);
    if (schedIt == state_.schedInfoMap.end() || !schedIt->second.isAlloc) {
        return SUCCESS;
    }
    coreLocation = schedIt->second.coreLocation;
    if (coreLocation != CoreLocationType::AIV0 && coreLocation != CoreLocationType::AIV1) {
        return SUCCESS;
    }

    auto& memIds = state_.GetOpMemIds(allocOp);
    if (memIds.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "AIV UB alloc alignment check cannot find memId for alloc %s.",
                          state_.GetOpInfo(allocOp).c_str());
        return FAILED;
    }
    memId = memIds[0];
    auto bufIt = state_.localBufferMap.find(memId);
    if (bufIt == state_.localBufferMap.end() || bufIt->second == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "AIV UB alloc alignment check cannot find local buffer for alloc %s, memId[%d].",
                          state_.GetOpInfo(allocOp).c_str(), memId);
        return FAILED;
    }
    buf = bufIt->second;
    if (buf->memType != MemoryType::MEM_UB) {
        return SUCCESS;
    }
    shouldCheck = true;
    return SUCCESS;
}

std::string DualDstEngine::FormatAivUbAllocRecord(const AivUbAllocRecord& record) const
{
    std::ostringstream oss;
    oss << "op: " << (record.op == nullptr ? "null" : state_.GetOpInfo(record.op)) << ", memId: " << record.memId
        << ", offset: " << record.offset << ", size: " << record.size;
    return oss.str();
}

Status DualDstEngine::TryCancelAivUbAllocRecords()
{
    // 两侧普通 AIV UB alloc 按下发顺序一一对应，不重排也不跳过记录。
    while (!aiv0UbAllocRecords_.empty() && !aiv1UbAllocRecords_.empty()) {
        const auto& aiv0Record = aiv0UbAllocRecords_.front();
        const auto& aiv1Record = aiv1UbAllocRecords_.front();
        if (aiv0Record.offset != aiv1Record.offset || aiv0Record.size != aiv1Record.size) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "AIV UB alloc alignment violated at clock[%d]. AIV0 {%s}; AIV1 {%s}; "
                              "queue sizes: aiv0=%zu, aiv1=%zu.",
                              state_.clock, FormatAivUbAllocRecord(aiv0Record).c_str(),
                              FormatAivUbAllocRecord(aiv1Record).c_str(), aiv0UbAllocRecords_.size(),
                              aiv1UbAllocRecords_.size());
            return FAILED;
        }
        aiv0UbAllocRecords_.pop_front();
        aiv1UbAllocRecords_.pop_front();
    }
    return SUCCESS;
}

Status DualDstEngine::RecordAivUbAlloc(Operation* allocOp)
{
    bool shouldCheck = false;
    CoreLocationType coreLocation = CoreLocationType::UNKNOWN;
    int memId = -1;
    LocalBufferPtr buf = nullptr;
    if (ResolveAivUbAllocRecordInput(allocOp, shouldCheck, coreLocation, memId, buf) != SUCCESS) {
        return FAILED;
    }
    if (!shouldCheck) {
        return SUCCESS;
    }

    AivUbAllocRecord record{allocOp, memId, static_cast<uint64_t>(buf->start), static_cast<uint64_t>(buf->size)};
    if (coreLocation == CoreLocationType::AIV0) {
        aiv0UbAllocRecords_.push_back(record);
    } else {
        aiv1UbAllocRecords_.push_back(record);
    }
    return TryCancelAivUbAllocRecords();
}

Status DualDstEngine::GetMatchedAivUbAllocOffset(Operation* allocOp, bool& hasMatchedOffset, uint64_t& matchedOffset)
{
    hasMatchedOffset = false;
    matchedOffset = 0;
    bool shouldCheck = false;
    CoreLocationType coreLocation = CoreLocationType::UNKNOWN;
    int memId = -1;
    LocalBufferPtr buf = nullptr;
    if (ResolveAivUbAllocRecordInput(allocOp, shouldCheck, coreLocation, memId, buf) != SUCCESS) {
        return FAILED;
    }
    if (!shouldCheck) {
        return SUCCESS;
    }

    auto& peerRecords = coreLocation == CoreLocationType::AIV0 ? aiv1UbAllocRecords_ : aiv0UbAllocRecords_;
    if (peerRecords.empty()) {
        return SUCCESS;
    }
    const auto& peerRecord = peerRecords.front();
    if (peerRecord.size != static_cast<uint64_t>(buf->size)) {
        APASS_LOG_ERROR_F(
            Elements::Operation,
            "AIV UB alloc alignment size mismatch before alloc %s. Current memId: %d, size: %lu; peer {%s}.",
            state_.GetOpInfo(allocOp).c_str(), memId, buf->size, FormatAivUbAllocRecord(peerRecord).c_str());
        return FAILED;
    }
    hasMatchedOffset = true;
    matchedOffset = peerRecord.offset;
    return SUCCESS;
}

void DualDstEngine::EraseFromOrderedOps(Operation* op)
{
    if (op == nullptr)
        return;
    auto it = std::find(state_.orderedOps.begin(), state_.orderedOps.end(), op);
    if (it != state_.orderedOps.end()) {
        state_.orderedOps.erase(it);
    }
    state_.schedInfoMap.erase(op);
    state_.opReqMemIdsMap.erase(op);
    state_.inOutOperandsCache.erase(op);
}

bool DualDstEngine::IsSupportedDualDstDtype(DataType dtype)
{
    return dtype == DataType::DT_FP32 || dtype == DataType::DT_INT32;
}

bool DualDstEngine::CheckDualDstDtype(LogicalTensorPtr l0cTensor, const std::vector<Operation*>& copyUbs)
{
    if (l0cTensor == nullptr) {
        APASS_LOG_DEBUG_F(Elements::Operation, "DualDst condition failed: l0c tensor is null.");
        return false;
    }

    auto l0cDtype = l0cTensor->Datatype();
    for (auto* copyUb : copyUbs) {
        if (copyUb == nullptr || copyUb->GetOOperands().empty() || copyUb->GetOutputOperand(0) == nullptr) {
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "DualDst condition failed: copyUb output tensor is invalid, copyUb=%s.",
                              state_.GetOpInfo(copyUb).c_str());
            return false;
        }

        auto ubTensor = copyUb->GetOutputOperand(0);
        auto ubDtype = ubTensor->Datatype();
        if (l0cDtype != ubDtype || !IsSupportedDualDstDtype(l0cDtype)) {
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "DualDst condition failed: l0c/ub dtype unsupported or mismatched, "
                              "l0cTensor[%d], l0cDtype=%d, copyUb=%s, ubTensor[%d], ubDtype=%d.",
                              l0cTensor->GetMagic(), static_cast<int>(l0cDtype), state_.GetOpInfo(copyUb).c_str(),
                              ubTensor->GetMagic(), static_cast<int>(ubDtype));
            return false;
        }
    }

    return true;
}

void DualDstEngine::AppendDualDstPairs(const std::vector<CandidatePair>& chosen, std::vector<DualDstPair>& pairs)
{
    for (auto& cp : chosen) {
        DualDstPair pair;
        pair.opEarly = cp.opEarly;
        pair.opLate = cp.opLate;
        pair.tensorEarly = cp.opEarly->GetOutputOperand(0);
        pair.tensorLate = cp.opLate->GetOutputOperand(0);
        pair.allocEarly = FindAllocPred(cp.opEarly);
        pair.allocLate = FindAllocPred(cp.opLate);
        if (pair.allocEarly == nullptr || pair.allocLate == nullptr) {
            APASS_LOG_WARN_F(Elements::Operation, "DualDst skip pair: cannot find alloc preds for op[%d]/op[%d]",
                             cp.opEarly->GetOpMagic(), cp.opLate->GetOpMagic());
            continue;
        }
        pairs.push_back(pair);
    }
}

void DualDstEngine::IdentifyPairsForOneL0C(LogicalTensorPtr l0cTensor, const std::vector<Operation*>& copyUbs,
                                           std::vector<DualDstPair>& pairs)
{
    APASS_LOG_INFO_F(Elements::Operation,
                     "DualDst l0cTensor->GetShape().size: %zu, copyUbs.size: %zu) for L0C tensor[%d]",
                     l0cTensor->GetShape().size(), copyUbs.size(), l0cTensor->GetMagic());
    if (l0cTensor->GetShape().size() != kCopyUbGeometryDimCount)
        return;
    if (copyUbs.size() != kDualDstOutputCount) {
        APASS_LOG_INFO_F(Elements::Operation,
                         "DualDst skip L0C tensor[%d]: expected exactly %zu L0C_COPY_UB consumers, but got %zu.",
                         l0cTensor->GetMagic(), kDualDstOutputCount, copyUbs.size());
        return;
    }
    if (!CheckDualDstDtype(l0cTensor, copyUbs))
        return;

    std::vector<CopyUbGeometry> geos;
    if (!LoadGeometries(copyUbs, geos))
        return;

    std::vector<CandidatePair> candM;
    std::vector<CandidatePair> candN;
    BuildAdjacencyCandidates(copyUbs, geos, candM, candN);

    std::vector<CandidatePair> pickedM;
    std::vector<CandidatePair> pickedN;
    GreedyNonOverlapPick(candM, pickedM);
    GreedyNonOverlapPick(candN, pickedN);

    bool chooseM = (pickedM.size() >= pickedN.size());
    std::vector<CandidatePair>& chosen = chooseM ? pickedM : pickedN;
    if (!chosen.empty()) {
        dualDstL0CDirection_[l0cTensor] = chooseM ? 0 : 1;
    }
    APASS_LOG_INFO_F(Elements::Operation, "DualDst pick direction: %s (M=%zu, N=%zu) for L0C tensor[%d]",
                     chooseM ? "SplitM" : "SplitN", pickedM.size(), pickedN.size(), l0cTensor->GetMagic());

    AppendDualDstPairs(chosen, pairs);
}

Status DualDstEngine::IdentifyDualDstPairs(std::vector<DualDstPair>& pairs)
{
    pairs.clear();
    std::unordered_map<LogicalTensorPtr, std::vector<Operation*>> l0cToCopyUb;
    for (auto* op : state_.orderedOps) {
        if (op == nullptr)
            continue;
        if (op->GetOpcode() != Opcode::OP_L0C_COPY_UB)
            continue;
        if (op->GetIOperands().empty())
            continue;
        auto l0cIn = op->GetInputOperand(0);
        if (l0cIn == nullptr)
            continue;
        l0cToCopyUb[l0cIn].push_back(op);
    }
    for (auto& kv : l0cToCopyUb) {
        IdentifyPairsForOneL0C(kv.first, kv.second, pairs);
    }
    APASS_LOG_INFO_F(Elements::Operation, "DualDst identify done: %zu pairs.", pairs.size());
    return SUCCESS;
}
Status DualDstEngine::FuseDualDstPairs(const std::vector<DualDstPair>& pairs)
{
    if (pairs.empty())
        return SUCCESS;
    size_t fusedCnt = 0;
    for (const auto& p : pairs) {
        if (FuseOnePair(p) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "DualDst graph rewrite failed after fusion started.");
            return FAILED;
        }
        fusedCnt++;
    }
    if (fusedCnt > 0) {
        function_.EraseOperations(false, true);
    }
    APASS_LOG_INFO_F(Elements::Operation, "DualDst fuse done: %zu / %zu pairs fused.", fusedCnt, pairs.size());
    return SUCCESS;
}

Operation* DualDstEngine::CreateDualDstFusedOp(const DualDstPair& p, LogicalTensorPtr l0cIn)
{
    Operation& cRef = function_.AddRawOperation(Opcode::OP_L0C_COPY_UB_DUAL_DST, {l0cIn},
                                                {p.tensorEarly, p.tensorLate});
    Operation* C = &cRef;
    C->UpdateInternalSubgraphID(p.opEarly->GetInternalSubgraphID());
    C->SetAttribute(OpAttributeKey::isCube, true);
    return C;
}

void DualDstEngine::SetDualDstCopyAttr(Operation* C, LogicalTensorPtr l0cIn, const DualDstPair& p,
                                       std::shared_ptr<CopyOpAttribute> attrE, std::shared_ptr<CopyOpAttribute> attrL)
{
    // realShape = 沿 SplitMN 轴对 opEarly/opLate 各自 shape 求和。
    // 其余字段沿用 attrE，dstValidShape 由 realShape 派生。
    auto eShapeImms = attrE->GetShape();
    auto lShapeImms = attrL->GetShape();
    if (eShapeImms.size() != kCopyUbGeometryDimCount || lShapeImms.size() != kCopyUbGeometryDimCount) {
        APASS_LOG_WARN_F(Elements::Operation, "DualDst SetCopyAttr: expect 2D shape, got E=%zu L=%zu",
                         eShapeImms.size(), lShapeImms.size());
        return;
    }
    int64_t eM = SpecifiedInt(eShapeImms[0]);
    int64_t eN = SpecifiedInt(eShapeImms[1]);
    int64_t lM = SpecifiedInt(lShapeImms[0]);
    int64_t lN = SpecifiedInt(lShapeImms[1]);
    if (eM <= 0 || eN <= 0 || lM <= 0 || lN <= 0) {
        APASS_LOG_WARN_F(Elements::Operation, "DualDst SetCopyAttr: shape not specified for op[%d]", C->GetOpMagic());
        return;
    }
    int64_t direction = dualDstL0CDirection_.count(l0cIn) ? dualDstL0CDirection_[l0cIn] : 0;
    std::vector<int64_t> realShape = (direction == 0) ? std::vector<int64_t>{eM + lM, eN} :
                                                        std::vector<int64_t>{eM, eN + lN};
    auto validShape = attrE->GetToDynValidShape();
    if (validShape.size() != kCopyUbGeometryDimCount) {
        validShape = OpImmediate::Specified(realShape);
    } else {
        // Only the split axis becomes the combined full tile.  The orthogonal
        // axis may still be dynamic and must keep the original valid shape.
        validShape[direction] = OpImmediate::Specified(realShape[direction]);
    }

    auto copyAttr = std::make_shared<CopyOpAttribute>(
        attrE->GetFromOffset(), p.tensorEarly->GetMemoryTypeOriginal(), OpImmediate::Specified(realShape),
        OpImmediate::Specified(l0cIn->tensor->GetDynRawShape()), validShape);
    copyAttr->SetToOffset(attrE->GetToOffset());
    C->SetOpAttribute(copyAttr);
    C->SetAttribute(OpAttributeKey::splitMN, direction);
}

void DualDstEngine::RewireEdgesForFusedOp(Operation* opEarly, Operation* opLate, Operation* A, Operation* B,
                                          Operation* C)
{
    auto rewireInOut = [this, A, B, C](Operation* op) {
        auto preds = state_.depManager.GetPredecessors(op);
        auto succs = state_.depManager.GetSuccessors(op);
        for (auto* pre : preds) {
            if (pre != A && pre != B) {
                if (pre->GetOpcodeStr().find("ALLOC") != std::string::npos) {
                    state_.depManager.AddAllocDependency(pre, C);
                } else {
                    state_.depManager.AddDependency(pre, C);
                }
            }
            state_.depManager.RemoveDependency(pre, op);
        }
        for (auto* suc : succs) {
            if (suc->GetOpcodeStr().find("ALLOC") != std::string::npos) {
                state_.depManager.AddAllocDependency(C, suc);
            } else {
                state_.depManager.AddDependency(C, suc);
            }
            state_.depManager.RemoveDependency(op, suc);
        }
    };
    rewireInOut(opEarly);
    rewireInOut(opLate);
    state_.depManager.AddAllocDependency(A, C);
    state_.depManager.AddAllocDependency(B, C);
}

void DualDstEngine::DetachOldOpsFromTensors(const DualDstPair& p, LogicalTensorPtr l0cIn)
{
    p.tensorEarly->RemoveProducer(p.opEarly);
    p.tensorLate->RemoveProducer(p.opLate);
    l0cIn->RemoveConsumer(p.opEarly);
    l0cIn->RemoveConsumer(p.opLate);
}

void DualDstEngine::SyncBufRefCountForFuse(const DualDstPair& p, Operation* C)
{
    auto sub = [this](Operation* op) {
        auto it = state_.opReqMemIdsMap.find(op);
        if (it == state_.opReqMemIdsMap.end())
            return;
        for (int mid : it->second) {
            auto rit = state_.bufRefCount.find(mid);
            if (rit != state_.bufRefCount.end())
                rit->second--;
        }
    };
    sub(p.opEarly);
    sub(p.opLate);

    std::vector<int> cMemIds;
    std::unordered_set<int> seen;
    auto add = [this, &cMemIds, &seen](LogicalTensorPtr t) {
        if (t == nullptr)
            return;
        if (t->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR)
            return;
        int mid = t->memoryrange.memId;
        if (!seen.insert(mid).second)
            return;
        cMemIds.push_back(mid);
        state_.bufRefCount[mid]++;
    };
    for (auto& t : C->GetOOperands())
        add(t);
    for (auto& t : C->GetIOperands())
        add(t);
    state_.SetOpMemIds(C, cMemIds);
}

Status DualDstEngine::FuseOnePair(const DualDstPair& p)
{
    if (p.opEarly == nullptr || p.opLate == nullptr || p.allocEarly == nullptr || p.allocLate == nullptr) {
        return FAILED;
    }
    auto l0cIn = p.opEarly->GetInputOperand(0);
    if (l0cIn == nullptr || l0cIn != p.opLate->GetInputOperand(0)) {
        APASS_LOG_WARN_F(Elements::Operation, "DualDst skip pair: l0c input mismatch op[%d] vs op[%d]",
                         p.opEarly->GetOpMagic(), p.opLate->GetOpMagic());
        return FAILED;
    }
    auto attrE = std::dynamic_pointer_cast<CopyOpAttribute>(p.opEarly->GetOpAttribute());
    auto attrL = std::dynamic_pointer_cast<CopyOpAttribute>(p.opLate->GetOpAttribute());
    if (attrE == nullptr || attrL == nullptr) {
        APASS_LOG_WARN_F(Elements::Operation, "DualDst skip pair: missing CopyOpAttribute op[%d]/op[%d]",
                         p.opEarly->GetOpMagic(), p.opLate->GetOpMagic());
        return FAILED;
    }

    Operation* A = p.allocEarly;
    Operation* B = p.allocLate;
    if (std::find(state_.orderedOps.begin(), state_.orderedOps.end(), A) == state_.orderedOps.end() ||
        std::find(state_.orderedOps.begin(), state_.orderedOps.end(), B) == state_.orderedOps.end()) {
        return FAILED;
    }

    Operation* C = CreateDualDstFusedOp(p, l0cIn);
    SetDualDstCopyAttr(C, l0cIn, p, attrE, attrL);
    RewireEdgesForFusedOp(p.opEarly, p.opLate, A, B, C);
    DetachOldOpsFromTensors(p, l0cIn);
    SyncBufRefCountForFuse(p, C);

    size_t replaceIdx = 0;
    if (!SpliceFusedOpIntoOrderedOps(p, C, replaceIdx))
        return FAILED;
    MarkDualDstAllocPair(A, B);

    p.opEarly->SetAsDeleted();
    p.opLate->SetAsDeleted();
    EraseFromOrderedOps(p.opEarly);
    EraseFromOrderedOps(p.opLate);
    for (size_t i = replaceIdx; i < state_.orderedOps.size(); ++i)
        state_.schedInfoMap[state_.orderedOps[i]].execOrder = static_cast<int>(i);

    APASS_LOG_INFO_F(Elements::Operation, "DualDst fused: opEarly[%d] + opLate[%d] -> dualOp[%d]; alloc pair[%d/%d]",
                     p.opEarly->GetOpMagic(), p.opLate->GetOpMagic(), C->GetOpMagic(), A->GetOpMagic(),
                     B->GetOpMagic());
    return SUCCESS;
}

void DualDstEngine::MarkDualDstAllocPair(Operation* A, Operation* B)
{
    state_.schedInfoMap[A].isDualDstAlloc = true;
    state_.schedInfoMap[A].pairedDualDstAlloc = B;
    state_.schedInfoMap[B].isDualDstAlloc = true;
    state_.schedInfoMap[B].pairedDualDstAlloc = A;
}

bool DualDstEngine::SpliceFusedOpIntoOrderedOps(const DualDstPair& p, Operation* C, size_t& replaceIdx)
{
    auto earlyCopyIt = std::find(state_.orderedOps.begin(), state_.orderedOps.end(), p.opEarly);
    auto lateCopyIt = std::find(state_.orderedOps.begin(), state_.orderedOps.end(), p.opLate);
    if (earlyCopyIt == state_.orderedOps.end() || lateCopyIt == state_.orderedOps.end())
        return false;
    auto replaceCopyIt = std::min(earlyCopyIt, lateCopyIt);
    auto eraseCopyIt = std::max(earlyCopyIt, lateCopyIt);
    replaceIdx = static_cast<size_t>(replaceCopyIt - state_.orderedOps.begin());
    OpSchedInfo fusedInfo = state_.schedInfoMap[*replaceCopyIt];
    *replaceCopyIt = C;
    state_.schedInfoMap[C] = fusedInfo;
    state_.schedInfoMap[C].execOrder = static_cast<int>(replaceIdx);
    state_.orderedOps.erase(eraseCopyIt);
    return true;
}

Status DualDstEngine::ResolveDualDstMemAndBuf(Operation* allocOp, DualDstAllocCtx& ctx)
{
    if (allocOp == nullptr || allocOp->GetOOperands().empty())
        return FAILED;
    auto infoIt = state_.schedInfoMap.find(allocOp);
    Operation* pairedAlloc = infoIt == state_.schedInfoMap.end() ? nullptr : infoIt->second.pairedDualDstAlloc;
    if (pairedAlloc == nullptr || pairedAlloc->GetOOperands().empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "DualDst[%d]: cannot resolve paired memId.", allocOp->GetOpMagic());
        return FAILED;
    }
    ctx.memIdA = allocOp->GetOutputOperand(0)->memoryrange.memId;
    ctx.memIdB = pairedAlloc->GetOutputOperand(0)->memoryrange.memId;
    ctx.bufA = state_.localBufferMap[ctx.memIdA];
    ctx.bufB = state_.localBufferMap[ctx.memIdB];
    if (ctx.bufA == nullptr || ctx.bufB == nullptr || ctx.bufA->size != ctx.bufB->size) {
        APASS_LOG_ERROR_F(Elements::Tensor,
                          "DualDst[%d]: missing localBuffer or size mismatch (A=%lu bytes B=%lu bytes).",
                          allocOp->GetOpMagic(), ctx.bufA ? ctx.bufA->size : 0, ctx.bufB ? ctx.bufB->size : 0);
        return FAILED;
    }
    return SUCCESS;
}

Status DualDstEngine::ResolveDualDstCores(Operation* allocOp, DualDstAllocCtx& ctx)
{
    auto infoIt = state_.schedInfoMap.find(allocOp);
    Operation* pairedAlloc = infoIt == state_.schedInfoMap.end() ? nullptr : infoIt->second.pairedDualDstAlloc;
    auto peerIt = state_.schedInfoMap.find(pairedAlloc);
    if (infoIt == state_.schedInfoMap.end() || peerIt == state_.schedInfoMap.end()) {
        return FAILED;
    }
    ctx.coreA = infoIt->second.coreLocation;
    ctx.coreB = peerIt->second.coreLocation;
    if (ctx.coreA == CoreLocationType::UNKNOWN || ctx.coreB == CoreLocationType::UNKNOWN || ctx.coreA == ctx.coreB) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "DualDst[%d]: paired memIds[%d/%d] not split across AIV0/AIV1 pools "
                          "(consumer core: %d / %d).",
                          allocOp->GetOpMagic(), ctx.memIdA, ctx.memIdB, static_cast<int>(ctx.coreA),
                          static_cast<int>(ctx.coreB));
        return FAILED;
    }
    return SUCCESS;
}

Status DualDstEngine::ResolveDualDstAllocCtx(Operation* allocOp, DualDstAllocCtx& ctx)
{
    if (ResolveDualDstMemAndBuf(allocOp, ctx) != SUCCESS)
        return FAILED;
    if (ResolveDualDstCores(allocOp, ctx) != SUCCESS)
        return FAILED;
    return SUCCESS;
}

void DualDstEngine::CommitDualDstAlloc(Operation* allocA, const DualDstAllocCtx& ctx, uint64_t off)
{
    Operation* allocB = state_.schedInfoMap[allocA].pairedDualDstAlloc;
    state_.tensorOccupyMap[ctx.memIdA] = allocA;
    state_.tensorOccupyMap[ctx.memIdB] = allocB;

    ctx.bufA->startCycle = state_.clock;
    ctx.bufB->startCycle = state_.clock;
    // dualdst 会同时在两个 UB 池分配，两侧各打一条事件。
    APASS_LOG_DEBUG_F(
        Elements::Operation, "[pool-evt] alloc clock=%llu cycles core=%d mt=0 memId=%d size=%llu bytes dualdst=A",
        (unsigned long long)state_.clock, static_cast<int>(ctx.coreA), ctx.memIdA, (unsigned long long)ctx.bufA->size);
    APASS_LOG_DEBUG_F(
        Elements::Operation, "[pool-evt] alloc clock=%llu cycles core=%d mt=0 memId=%d size=%llu bytes dualdst=B",
        (unsigned long long)state_.clock, static_cast<int>(ctx.coreB), ctx.memIdB, (unsigned long long)ctx.bufB->size);
    APASS_LOG_DEBUG_F(Elements::Operation, "DualDst alloc[%d]: placed memId[%d]/[%d] at offset %lu (size %lu).",
                      allocA->GetOpMagic(), ctx.memIdA, ctx.memIdB, off, ctx.bufA->size);
}

std::optional<uint64_t> DualDstEngine::FindCommonFreeOffset(BufferPool& poolA, BufferPool& poolB, uint64_t size)
{
    if (size == 0) {
        return std::optional<uint64_t>{0};
    }
    auto listA = poolA.GetSortedFreeIntervals();
    auto listB = poolB.GetSortedFreeIntervals();
    size_t i = 0;
    size_t j = 0;
    while (i < listA.size() && j < listB.size()) {
        uint64_t s = std::max(listA[i].first, listB[j].first);
        uint64_t e = std::min(listA[i].second, listB[j].second);
        if (e >= s && (e - s) >= size) {
            return std::optional<uint64_t>{s};
        }
        if (listA[i].second <= listB[j].second) {
            i++;
        } else {
            j++;
        }
    }
    return std::nullopt;
}

Status DualDstEngine::AllocateDualDstAtCurrent(Operation* allocA, bool& allocated)
{
    allocated = false;
    auto infoIt = state_.schedInfoMap.find(allocA);
    if (infoIt == state_.schedInfoMap.end() || !infoIt->second.isDualDstAlloc ||
        infoIt->second.pairedDualDstAlloc == nullptr) {
        return FAILED;
    }
    if (state_.IsOpRetired(infoIt->second.pairedDualDstAlloc)) {
        allocated = true;
        return SUCCESS;
    }
    DualDstAllocCtx ctx;
    if (ResolveDualDstAllocCtx(allocA, ctx) != SUCCESS)
        return FAILED;

    auto& poolForA = state_.bufferManagerMap[ctx.coreA][MemoryType::MEM_UB];
    auto& poolForB = state_.bufferManagerMap[ctx.coreB][MemoryType::MEM_UB];
    auto off = FindCommonFreeOffset(poolForA, poolForB, ctx.bufA->size);
    if (!off.has_value()) {
        // 两个 UB 池不存在共同连续空闲段，交给调用方触发常规 spill。
        LogDualDstAllocMiss(allocA, poolForA, poolForB, ctx.bufA->size);
        return SUCCESS;
    }
    if (AllocateBothPoolsAtOffset(ctx, *off, allocA) != SUCCESS)
        return FAILED;
    CommitDualDstAlloc(allocA, ctx, *off);
    allocated = true;
    return SUCCESS;
}

void DualDstEngine::LogDualDstAllocMiss(Operation* allocA, BufferPool& poolA, BufferPool& poolB, uint64_t size)
{
    auto freeA = poolA.GetSortedFreeIntervals();
    auto freeB = poolB.GetSortedFreeIntervals();
    std::string sA, sB;
    for (auto& [s, e] : freeA) {
        sA += "[" + std::to_string(s) + "," + std::to_string(e) + ") ";
    }
    for (auto& [s, e] : freeB) {
        sB += "[" + std::to_string(s) + "," + std::to_string(e) + ") ";
    }
    APASS_LOG_DEBUG_F(
        Elements::Tensor,
        "[dualdst-debug] DualDst alloc[%d] FindCommonFreeOffset miss: size=%lu, poolA free=%s, poolB free=%s",
        allocA->GetOpMagic(), size, sA.c_str(), sB.c_str());
}

Status DualDstEngine::AllocateBothPoolsAtOffset(const DualDstAllocCtx& ctx, uint64_t off, Operation* allocA)
{
    auto& poolForA = state_.bufferManagerMap[ctx.coreA][MemoryType::MEM_UB];
    auto& poolForB = state_.bufferManagerMap[ctx.coreB][MemoryType::MEM_UB];
    if (poolForA.AllocateAtOffset(ctx.bufA, off) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Tensor, "DualDst alloc[%d]: AllocateAtOffset poolForA failed at offset %lu bytes.",
                          allocA->GetOpMagic(), off);
        return FAILED;
    }
    if (poolForB.AllocateAtOffset(ctx.bufB, off) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Tensor,
                          "DualDst alloc[%d]: AllocateAtOffset poolForB failed at offset %lu bytes, rollback.",
                          allocA->GetOpMagic(), off);
        (void)poolForA.Free(ctx.memIdA);
        return FAILED;
    }
    return SUCCESS;
}

// ===== 核心 Override 查询 =====

Status DualDstEngine::RunDualDstFuse(std::vector<Operation*>& opList)
{
    if (!state_.enableDualDst) {
        return SUCCESS;
    }
    if (state_.coreInitConfigs.find(CoreLocationType::AIV1) == state_.coreInitConfigs.end()) {
        return SUCCESS;
    }
    dualDstL0CDirection_.clear();
    std::vector<DualDstPair> pairs;
    if (IdentifyDualDstPairs(pairs) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "IdentifyDualDstPairs failed.");
        return FAILED;
    }
    if (pairs.empty()) {
        state_.enableDualDst = false;
        return SUCCESS;
    }
    if (RealignAllocByIso(opList) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "RealignAllocByIso failed!");
        return FAILED;
    }
    for (size_t i = 0; i < opList.size(); ++i) {
        if (opList[i] != nullptr) {
            state_.schedInfoMap[opList[i]].execOrder = static_cast<int>(i);
        }
    }
    if (FuseDualDstPairs(pairs) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "FuseDualDstPairs failed.");
        return FAILED;
    }
    return SUCCESS;
}

bool DualDstEngine::BuildIsoReorderPlan(const std::vector<Operation*>& opList,
                                        const std::unordered_map<Operation*, Operation*>& isoPairs,
                                        std::vector<Operation*>& values, std::vector<size_t>& slots)
{
    std::unordered_map<Operation*, size_t> pos;
    pos.reserve(opList.size());
    for (size_t i = 0; i < opList.size(); ++i) {
        if (opList[i] != nullptr)
            pos[opList[i]] = i;
    }

    std::vector<std::pair<Operation*, Operation*>> ordered;
    ordered.reserve(isoPairs.size());
    for (const auto& p : isoPairs) {
        if (pos.find(p.first) == pos.end() || pos.find(p.second) == pos.end())
            continue;
        ordered.emplace_back(p.first, p.second);
    }
    std::sort(ordered.begin(), ordered.end(),
              [&](const auto& x, const auto& y) { return pos.at(x.first) < pos.at(y.first); });

    values.clear();
    slots.clear();
    values.reserve(ordered.size());
    slots.reserve(ordered.size());
    std::unordered_set<Operation*> seenValues;
    for (const auto& p : ordered) {
        if (!seenValues.insert(p.second).second) {
            APASS_LOG_INFO_F(Elements::Operation,
                             "DualDst iso reorder skip: AIV1 op %s is paired with more than one AIV0 op.",
                             state_.GetOpInfo(p.second).c_str());
            return false;
        }
        values.push_back(p.second);
        slots.push_back(pos.at(p.second));
    }
    std::sort(slots.begin(), slots.end());
    return true;
}

bool DualDstEngine::IsTopoOrderPreserved(const std::vector<Operation*>& opList)
{
    std::unordered_map<Operation*, size_t> pos;
    pos.reserve(opList.size());
    for (size_t i = 0; i < opList.size(); ++i) {
        if (opList[i] != nullptr)
            pos[opList[i]] = i;
    }
    for (size_t i = 0; i < opList.size(); ++i) {
        if (opList[i] == nullptr)
            continue;
        for (auto* pred : state_.depManager.GetPredecessors(opList[i])) {
            auto it = pos.find(pred);
            if (it != pos.end() && it->second > i) {
                APASS_LOG_DEBUG_F(Elements::Operation, "DualDst iso reorder skip: %s would precede its producer %s.",
                                  state_.GetOpInfo(opList[i]).c_str(), state_.GetOpInfo(pred).c_str());
                return false;
            }
        }
    }
    return true;
}

// 返回是否真的重排了。跳过（无可重排的 pair、置换非法、会破坏拓扑序）都不是失败，
// 调用方无需区分成功/失败，只需知道 opList 有没有被改。
bool DualDstEngine::ReorderAiv1ToAiv0Order(std::vector<Operation*>& opList,
                                           const std::unordered_map<Operation*, Operation*>& isoPairs)
{
    if (isoPairs.empty())
        return false;

    std::vector<Operation*> values;
    std::vector<size_t> slots;
    if (!BuildIsoReorderPlan(opList, isoPairs, values, slots)) {
        return false;
    }
    APASS_LOG_INFO_F(Elements::Operation, "DualDst iso reorder: %zu of %zu pairs in this opList.", values.size(),
                     isoPairs.size());

    // AIV1 侧同构 op 在自己原有的槽位集合内置换，使其相对顺序镜像 AIV0 侧。
    // 槽位集合不变，所以这批 op 占用的位置范围不会外扩；alloc 和它的消费者同属这批，
    // 不会再出现只搬 alloc、消费者留在原地而拉长 buffer 生命周期的情况。
    std::vector<Operation*> candidate = opList;
    for (size_t k = 0; k < values.size(); ++k) {
        candidate[slots[k]] = values[k];
    }

    if (!IsTopoOrderPreserved(candidate)) {
        APASS_LOG_INFO_F(Elements::Operation, "DualDst iso reorder skipped: topo order would break.");
        return false;
    }
    opList = std::move(candidate);

    APASS_LOG_INFO_F(Elements::Operation, "DualDst iso reorder: aligned %zu AIV1 ops to AIV0 order.", values.size());
    return true;
}

Status DualDstEngine::RealignAllocByIso(std::vector<Operation*>& opList)
{
    if (!state_.enableDualDst)
        return SUCCESS;

    std::unordered_map<Operation*, Operation*> isoPairs = state_.dualDstPairs;
    isoPairs.insert(state_.dualDstOpPairs.begin(), state_.dualDstOpPairs.end());
    if (isoPairs.empty())
        return SUCCESS;

    bool reordered = ReorderAiv1ToAiv0Order(opList, isoPairs);
    APASS_LOG_INFO_F(Elements::Operation, "DualDst iso realign: %zu pairs (%zu alloc + %zu op), reordered=%d.",
                     isoPairs.size(), state_.dualDstPairs.size(), state_.dualDstOpPairs.size(),
                     static_cast<int>(reordered));
    return SUCCESS;
}
} // namespace npu::tile_fwk
