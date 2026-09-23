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
 * \file spill_engine.h
 * \brief Spill 执行层: SpillBuffer 执行一次 spill, 分四个阶段 —— 定计划、落盘、回载、收尾。
 *        选哪块 buffer、要不要 spill 这些决策留在 OoOScheduler。
 */

#ifndef PASS_SPILL_ENGINE_H
#define PASS_SPILL_ENGINE_H

#include <set>
#include <vector>

#include "passes/block_graph_pass/schedule_ooo/common/schedule_state.h"
#include "passes/statistics/schedule_observer.h"

#ifdef MODULE_NAME
#undef MODULE_NAME
#endif
#define MODULE_NAME "SpillEngine"

namespace npu::tile_fwk {

// 一次 spill 的形态, 阶段①判完就不再变。下游照它走, 不从字段空不空反推。
enum class SpillKind {
    ReuseDDR,       // 归还这一级的写就是 copyin: 克隆它, 连镜像都不建
    InPlaceWhole,   // 这一级落得了盘、数据齐: 一条整块 copyout + 一条整块 copyin
    InPlacePartial, // 这一级落得了盘、数据没齐: 带脏字节整块落, 分片回载 + assemble
    WalkUp,         // 这一级落不了盘: 每个写各走一跳, 改从它们的输入落
};

// 一份真实数据源: 落盘镜像的一个写入区间从哪里读、copyout 挂在谁身上。
struct SpillSource {
    LogicalTensorPtr tensor;             // 从这里读
    std::vector<Operation*> producerOps; // 产出这块数据的那些写; 定 copyout 的落位、retired 与 scale
    std::vector<OpImmediate> saveOffset; // copyout 写进镜像的哪个位置; 就地落盘时空 (落点即原点)
    bool producedInPast{false};          // 源已产出 -> copyout 插回历史序列; 否则进后续调度
};

// 数据不完备时的一片回载: 已执行的从镜像读回再 assemble 装进整块; 未执行的那段字节是脏的, 改指向。
struct SpillPartialWrite {
    Operation* writeOp{nullptr};   // 这一片是谁写的
    bool producedInPast{false};    // 已执行 -> 分片读回; 否则改指向
    std::vector<int64_t> toOffset; // 这一片在整块里的落位
    std::vector<SymbolicScalar> toDynOffset;
    std::vector<SymbolicScalar> fromDynValidShape;
};

// 一份落盘镜像。dtype 决定要几份: 随路转换只能落在 copyout 上, 不同 dtype 不能共用一份镜像。
struct SpillMirror {
    DataType dtype;
    std::vector<Operation*> consumers;  // 要顶替的消费者; 换输入路为空
    LogicalTensorPtr gmTensor{nullptr}; // 阶段②填
};

// 一次 spill 的局部状态: ① 定计划, ②③ 往里填产物。跨多次 spill 的累积在 SpillContext。
struct SpillPlan {
    SpillKind kind{SpillKind::InPlaceWhole};
    std::vector<SpillSource> sources;
    std::vector<SpillMirror> mirrors;
    std::vector<SpillPartialWrite> partialWrites; // 只有 InPlacePartial 非空
    Operation* cloneCopyinFrom{nullptr};          // 只有 ReuseDDR 非空
    std::vector<OpImmediate> reloadOffset;        // 只有 WalkUp 单写非空, 取那一跳的 fromOffset
    bool crossedNd2nz{false};                     // 上溯走过了 ND2NZ -> 回载要重做分形
    bool replaceInput{false}; // 回载能从 DDR 直搬回这一级 -> 换输入; 否则 (L0C) 顶替消费者
    SingleSpillCreatedOps created;
    std::vector<Operation*> reloadCopyinOps; // 本轮回载的 copyin, 分片时多条; 补 token 用
    std::vector<Operation*> saveCopyoutOps;  // 本轮存盘的 copyout; seed 用
};

// 插进调度序列的一批 op, 每个带自己占的 memId 列表。
using OpMemIdMap = std::vector<std::pair<Operation*, std::vector<int>>>;

// 一次收割的全部产物: 失去读者的写、它们腾空的 buffer, 以及那些 buffer 的 memId。
struct OrphanedOps {
    std::vector<Operation*> ops;
    std::vector<LogicalTensorPtr> tensors;
    std::set<int> memIds;
};

class SpillEngine {
public:
    SpillEngine(ScheduleState& state, Function& function) : state_(state), function_(function) {}
    ~SpillEngine() {}

    Status SpillBuffer(int memId, Operation* spillAllocOp, SpillContext& ctx);

    void EmitInitDDRBuffer(const LogicalTensorPtr& t, DDRBufferKind kind);

    Operation* GetSpillOp(int memId);
    int GetBufNextUseTime(int curMemId);
    bool IsBelongSpillBlackList(int memId, Operation* op);

private:
    bool IsVfTensorSpillBlocked(int memId, int triggerScopeId) const;
    ScheduleState& state_;
    Function& function_;
    IRBuilder irBuilder_;
    std::unordered_map<int, DDRBufferKind> ddrKindMap_;
    void EraseSchedulerSideMaps(Operation* op);
    void FindFilterLtags(Operation* allocOp, std::set<Operation*>& filterLtags);

    LogicalTensorPtr CreateLocalTensor(LogicalTensorPtr spillTensor);
    LogicalTensorPtr CreateGMTensor(LogicalTensorPtr ref, int spillMemId, DataType dtype);
    Operation* CreateAllocOp(LogicalTensorPtr oOperand);
    void RegisterLocalBuffer(const LogicalTensorPtr& localTensor);
    void RegisterTensorAllocOp(Operation* allocOp);
    Status ReserveWorkspaceRange(int spillMemId, int64_t size, int64_t& baseOffset, TileRange& range);
    Operation* CloneCopyinOp(Operation* spillOp, LogicalTensorPtr iOperand, LogicalTensorPtr oOperand);
    Operation* CreateCopyinOp(LogicalTensorPtr iOperand, LogicalTensorPtr oOperand, std::vector<OpImmediate> offset,
                              bool isND2NZ = false);
    Operation* CreateCopyoutOp(Operation* spillOp, LogicalTensorPtr iOperand, LogicalTensorPtr oOperand,
                               std::vector<OpImmediate> offset);

    static LogicalTensorPtr GetMirrorRef(const SpillPlan& plan, LogicalTensorPtr spillTensor);

    void TakeOverScheduleSlot(Operation* oldOp, Operation* newOp);

    LogicalTensorPtr GetSpillTensor(Operation* spillOp, int spillMemId);

    Status PlanSpill(int memId, LogicalTensorPtr spillTensor, SpillPlan& plan);
    Status SaveToDDR(int memId, LogicalTensorPtr spillTensor, SpillPlan& plan, SpillContext& ctx);
    Status ReloadFromDDR(int memId, LogicalTensorPtr spillTensor, Operation* spillOp, Operation* spillAllocOp,
                         SpillPlan& plan, SpillContext& ctx);
    Status FinalizeSpill(int memId, LogicalTensorPtr spillTensor, Operation* spillAllocOp, SpillPlan& plan,
                         SpillContext& ctx);

    Status CollectMirrorGroups(LogicalTensorPtr spillTensor, bool replaceInput, std::vector<SpillMirror>& mirrors);
    Status CollectSpillSources(LogicalTensorPtr spillTensor, SpillPlan& plan);
    Status CollectPartialWrites(LogicalTensorPtr spillTensor, std::vector<SpillPartialWrite>& partialWrites);
    static Status GetPartialWriteReplayAttr(Operation* writeOp, SpillPartialWrite& partial);

    SpillKind DispatchSpill(LogicalTensorPtr spillTensor);
    static bool IsReadFromDDR(Operation* op);
    Status CollectInPlaceSource(LogicalTensorPtr spillTensor, SpillPlan& plan);
    Status CollectWalkUpSources(LogicalTensorPtr spillTensor, SpillPlan& plan);
    LogicalTensorPtr WalkUpOneHop(Operation* writeOp, SpillPlan& plan);
    std::vector<Operation*> CollectDataWrites(LogicalTensorPtr tensor);
    std::vector<Operation*> CollectSourceProducers(LogicalTensorPtr source);
    bool IsDataComplete(LogicalTensorPtr tensor);
    static Operation* GetScaleDonor(const SpillSource& source);
    int ComputeCopyoutExecOrder(const SpillSource& source, Operation* copyoutOp);
    static bool IsPureMove(Operation* op);
    static bool IsLayoutMove(Operation* op);
    static bool ReloadReappliesLayout(MemoryType memType);
    bool HasLayoutWrite(LogicalTensorPtr tensor);
    bool CanSaveTensorToDDR(LogicalTensorPtr tensor);
    static std::vector<OpImmediate> GetSaveOffset(Operation* writeOp);
    static std::vector<OpImmediate> GetReloadOffset(Operation* moveOp);
    static bool CanReplayOffset(const std::vector<OpImmediate>& offset);

    Status ReplaceConsumersWithCopyin(const SpillMirror& mirror, Operation* spillAllocOp,
                                      SingleSpillCreatedOps& created, std::vector<Operation*>& seeds);
    Status PrepareSpillMirror(int spillMemId, const SpillPlan& plan, LogicalTensorPtr spillTensor, SpillMirror& mirror);
    Status SaveSourcesToDDR(LogicalTensorPtr gmTensor, SpillContext& ctx, SpillPlan& plan);
    Status ReloadIntoNewBuffer(int spillMemId, LogicalTensorPtr spillTensor, Operation* spillOp,
                               Operation* spillAllocOp, SpillPlan& plan, SpillContext& ctx);
    Operation* CreateWholeReload(LogicalTensorPtr gmTensor, LogicalTensorPtr localTensor, const SpillPlan& plan);
    Operation* CreateReFractalReload(LogicalTensorPtr gmTensor, LogicalTensorPtr localTensor, const SpillPlan& plan,
                                     OpMemIdMap& opMemIdMap);
    LogicalTensorPtr CreateNdMidTensor(LogicalTensorPtr gmTensor, LogicalTensorPtr localTensor);
    Operation* CreateFractalOp(LogicalTensorPtr iOperand, LogicalTensorPtr oOperand);
    void NormalizeAssembleAllocOutput(LogicalTensorPtr spillTensor);
    void CreatePartialReloads(LogicalTensorPtr spillTensor, LogicalTensorPtr localTensor, LogicalTensorPtr gmTensor,
                              const std::vector<SpillPartialWrite>& partialWrites, OpMemIdMap& opMemIdMap);
    void CreateOnePartialReload(LogicalTensorPtr localTensor, LogicalTensorPtr gmTensor,
                                const SpillPartialWrite& partial, OpMemIdMap& opMemIdMap);
    LogicalTensorPtr CreatePartialTensor(LogicalTensorPtr shapeFrom, LogicalTensorPtr wholeTensor,
                                         const std::vector<int64_t>& toOffset);
    Operation* CreateAssembleOp(LogicalTensorPtr iOperand, LogicalTensorPtr oOperand, const SpillPartialWrite& partial);
    Status DropWritesWithoutReaders(LogicalTensorPtr spillTensor, SpillPlan& plan, SpillContext& ctx);
    OrphanedOps CollectOrphanedChain(LogicalTensorPtr spillTensor);
    static bool HasLiveConsumer(const LogicalTensorPtr& tensor);
    bool HasLiveReader(Operation* op);
    bool DeleteOneOp(Operation* op, const std::set<int>& orphanedMemIds);
    bool EraseFromExecOrder(Operation* op);
    void ReleaseOpBufRefs(Operation* op, const std::set<int>& orphanedMemIds);
    void DetachOrphanedProducers(const OrphanedOps& orphaned);
    Status FreeSpilledBuffer(int memId, CoreLocationType freeCore);

    Status UpdateSpillOpDepend(Operation* spillOp, LogicalTensorPtr newTensor, int spillMemId,
                               std::vector<Operation*>& rewired);

    Status UpdateOperationInput(Operation* targetOp, Operation* spillOp, LogicalTensorPtr reloadTensor, int spillMemId);
    Status UpdateSkipOpInput(Operation* chainTail, Operation* spillOp, Operation* targetOp,
                             LogicalTensorPtr reloadTensor, size_t index);
    LogicalTensorPtr CloneSkipChain(Operation* targetOp, const std::vector<Operation*>& chain,
                                    LogicalTensorPtr reloadTensor);
    void UnregisterSkipOp(Operation* targetOp, Operation* skipOp);
    void DetachOrphanedSkipChain(const std::vector<Operation*>& chain, Operation* targetOp);
    void ReplaceSkipOpChainMemId(LogicalTensorPtr startTensor, int oldMemId, int newMemId);
    bool RemapOpReqMemId(Operation* op, int oldMemId, int newMemId);
    bool ReplaceTensorMemId(Operation* op, int oldMemId, int newMemId);
    Status UpdateRemainMemid(int oldMemId, int newMemId, std::vector<Operation*>& remapped);
    void UpdateOpInternalSubgraphID(Operation& op, Operation* srcOp);

    Status UpdateCopyoutScheduleInfo(Operation* op, const SpillSource& source, int sourceMemId);
    void UpdateOpScheduleInfo(Operation* op, std::vector<int> memIds, Operation* AllocOp);
    Status InsertOps(OpMemIdMap opMemidMap, Operation* spillAllocOp, int memId);
    Status UpdateScheduleStatus(OpMemIdMap opMemidMap, int memId, Operation* spillAllocOp, LogicalTensorPtr localTensor,
                                Operation* spillOp, const SpillPlan& plan);
};

} // namespace npu::tile_fwk
#endif // PASS_SPILL_ENGINE_H
