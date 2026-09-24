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
 * \file dualdst_engine.h
 * \brief DualDst execution engine — DualDst fuse identification, graph modification,
 *        and dual-buffer allocation extracted from OoOScheduler.
 *        Public interface = RunDualDstFuse + AllocateDualDstAtCurrent.
 *        ResolveCoreForFree / IsDualDstAlloc / enableDualDst moved to ScheduleState
 *        (shared by SpillEngine / OoOScheduler).
 */

#ifndef PASS_DUALDST_ENGINE_H
#define PASS_DUALDST_ENGINE_H

#include <deque>
#include <optional>
#include <memory>
#include <set>
#include <string>
#include "passes/block_graph_pass/schedule_ooo/common/schedule_state.h"
#include "interface/operation/attribute.h"

#ifdef MODULE_NAME
#undef MODULE_NAME
#endif
#define MODULE_NAME "DualDstEngine"

namespace npu::tile_fwk {

class DualDstEngine {
public:
    DualDstEngine(ScheduleState& state, Function& function) : state_(state), function_(function) {}
    ~DualDstEngine() {}

    Status RunDualDstFuse(std::vector<Operation*>& opList);
    Status AllocateDualDstAtCurrent(Operation* allocA, bool& allocated);
    Status ResolveDualDstAllocCtx(Operation* allocOp, DualDstAllocCtx& ctx);
    Status GetMatchedAivUbAllocOffset(Operation* allocOp, bool& hasMatchedOffset, uint64_t& matchedOffset);
    Status RecordAivUbAlloc(Operation* allocOp);

    bool IsDualDstEnabled() const { return state_.enableDualDst; }
    void SetDualDstL0CDirection(const std::unordered_map<LogicalTensorPtr, int64_t>& dir)
    {
        dualDstL0CDirection_ = dir;
    }
    const auto& GetL02L0MXMap() const { return l02L0MXMap_; }
    auto& GetL02L0MXMap() { return l02L0MXMap_; }

    // 诊断：调度完成后，校验 OP_L0C_COPY_UB_DUAL_DST 的两个输出 memId。
    // 两个输出 buffer 在最终 BufferPool 中的 start offset 应保持相等，违例返回 FAILED。
    Status VerifyDualDstSameOffset();
    // DualDstFuse 后，对两个输出 tensor 的下游链做同构判断。
    // 然后把 AIV1 侧 alloc 在 orderedOps 中的顺序对齐 AIV0 侧顺序。
    Status RealignAllocByIso(std::vector<Operation*>& opList);

private:
    static constexpr int64_t kInvalidCoord = INT64_MIN;
    static constexpr int kCopyUbGeometryDimCount = 2;
    static constexpr int kMinDualDstPairCount = 2;
    static constexpr size_t kDualDstOutputCount = 2;
    static constexpr int kMaxConsumerSearchDepth = 16;

    struct CopyUbGeometry {
        int64_t fromM{kInvalidCoord};
        int64_t fromN{kInvalidCoord};
        int64_t tileM{kInvalidCoord};
        int64_t tileN{kInvalidCoord};
        std::vector<int64_t> ubShape;
        std::vector<SymbolicScalar> ubValidShape;
        LogicalTensorPtr ubOut;
    };

    struct CandidatePair {
        Operation* opEarly;
        Operation* opLate;
        int64_t earlyOffsetOnAxis;
    };

    struct AivUbAllocRecord {
        Operation* op{nullptr};
        int memId{-1};
        uint64_t offset{0};
        uint64_t size{0};
    };

    static int64_t SpecifiedInt(const OpImmediate& imm);
    static bool ReadGeometry(Operation* op, CopyUbGeometry& g);
    static bool LoadGeometries(const std::vector<Operation*>& copyUbs, std::vector<CopyUbGeometry>& geos);
    static void GreedyNonOverlapPick(std::vector<CandidatePair>& cands, std::vector<CandidatePair>& picked);
    static bool IsSupportedDualDstDtype(DataType dtype);

    CoreLocationType ConsumerCore(Operation* copyUbOp);
    Operation* FindAllocPred(Operation* op);
    bool CheckDualDstDtype(LogicalTensorPtr l0cTensor, const std::vector<Operation*>& copyUbs);
    void BuildAdjacencyCandidates(const std::vector<Operation*>& copyUbs, const std::vector<CopyUbGeometry>& geos,
                                  std::vector<CandidatePair>& candM, std::vector<CandidatePair>& candN);
    void AppendDualDstPairs(const std::vector<CandidatePair>& chosen, std::vector<DualDstPair>& pairs);
    void EraseFromOrderedOps(Operation* op);
    void IdentifyPairsForOneL0C(LogicalTensorPtr l0cTensor, const std::vector<Operation*>& copyUbs,
                                std::vector<DualDstPair>& pairs);
    Status IdentifyDualDstPairs(std::vector<DualDstPair>& pairs);
    Status FuseDualDstPairs(const std::vector<DualDstPair>& pairs);
    Operation* CreateDualDstFusedOp(const DualDstPair& p, LogicalTensorPtr l0cIn);
    void SetDualDstCopyAttr(Operation* C, LogicalTensorPtr l0cIn, const DualDstPair& p,
                            std::shared_ptr<CopyOpAttribute> attrE, std::shared_ptr<CopyOpAttribute> attrL);
    void RewireEdgesForFusedOp(Operation* opEarly, Operation* opLate, Operation* A, Operation* B, Operation* C);
    void DetachOldOpsFromTensors(const DualDstPair& p, LogicalTensorPtr l0cIn);
    void SyncBufRefCountForFuse(const DualDstPair& p, Operation* C);
    Status FuseOnePair(const DualDstPair& p);
    void MarkDualDstAllocPair(Operation* A, Operation* B);
    bool SpliceFusedOpIntoOrderedOps(const DualDstPair& p, Operation* C, size_t& replaceIdx);
    Status ResolveDualDstMemAndBuf(Operation* allocOp, DualDstAllocCtx& ctx);
    Status ResolveDualDstCores(Operation* allocOp, DualDstAllocCtx& ctx);
    void CommitDualDstAlloc(Operation* allocA, const DualDstAllocCtx& ctx, uint64_t off);
    void LogDualDstAllocMiss(Operation* allocA, BufferPool& poolA, BufferPool& poolB, uint64_t size);
    Status AllocateBothPoolsAtOffset(const DualDstAllocCtx& ctx, uint64_t off, Operation* allocA);
    std::optional<uint64_t> FindCommonFreeOffset(BufferPool& poolA, BufferPool& poolB, uint64_t size);
    bool IsAivUbAllocAlignmentCheckEnabled() const;
    Status ResolveAivUbAllocRecordInput(Operation* allocOp, bool& shouldCheck, CoreLocationType& coreLocation,
                                        int& memId, LocalBufferPtr& buf);
    Status TryCancelAivUbAllocRecords();
    std::string FormatAivUbAllocRecord(const AivUbAllocRecord& record) const;
    // 把 AIV1 侧同构 op 在 orderedOps 中重排为与 AIV0 侧一致，AIV0 侧保持原始顺序。
    // 返回是否真的重排了，跳过不算失败。
    bool ReorderAiv1ToAiv0Order(std::vector<Operation*>& opList,
                                const std::unordered_map<Operation*, Operation*>& isoPairs);
    bool BuildIsoReorderPlan(const std::vector<Operation*>& opList,
                             const std::unordered_map<Operation*, Operation*>& isoPairs,
                             std::vector<Operation*>& values, std::vector<size_t>& slots);
    bool IsTopoOrderPreserved(const std::vector<Operation*>& opList);

    ScheduleState& state_;
    Function& function_;
    std::unordered_map<LogicalTensorPtr, int64_t> dualDstL0CDirection_;
    std::unordered_map<LogicalTensorPtr, LogicalTensorPtr> l02L0MXMap_;
    std::deque<AivUbAllocRecord> aiv0UbAllocRecords_;
    std::deque<AivUbAllocRecord> aiv1UbAllocRecords_;
};

} // namespace npu::tile_fwk
#endif // PASS_DUALDST_ENGINE_H
