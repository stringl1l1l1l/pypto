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
 * \file spill_engine.cpp
 * \brief Spill execution engine — pure execution layer only.
 *        Orchestration methods (GenBufferSpill, SelectSpillBuffers, HasEnoughBuffer,
 *        RearrangeBuffer, ApplySpillContext, PrintSpillFailedInfo, GetSpillGroup,
 *        GetDualSpillGroup, GetGroupNextUseTime) are in OoOScheduler (spill_buffer.cpp).
 *        SpillEngine only retains execution-layer methods that work purely through
 *        state_ / notifier_ / function_ / irBuilder_ / ddrKindMap_.
 */

#include "spill_engine.h"
#include <algorithm>
#include "tilefwk/symbolic_scalar.h"
#include "passes/pass_utils/reschedule_utils.h"
#include "passes/pass_utils/mem_path_utils.h"
#include "passes/pass_utils/pass_token_utils.h"
#include "passes/pass_utils/pass_utils.h"

namespace npu::tile_fwk {

constexpr int32_t DEFAULT_LATENCY = 511;

namespace {

// 写后写冲突判据: 同一块地 + 都在写 + 区域重叠, 不按 opcode 名字判。
bool ConflictsWith(const LogicalTensorPtr& reloaded, Operation* op, int memId)
{
    for (const auto& written : op->GetOOperands()) {
        if (written == nullptr || written->memoryrange.memId != memId) {
            continue;
        }
        // 维数不等会让 IsOverlapping 越界, 判不出就当重叠。
        if (written->shape.size() != reloaded->shape.size() || IsOverlapping(*reloaded, *written)) {
            return true;
        }
    }
    return false;
}

// 回载写的那片字节, 往后还写它的 op 必须排在回载之后, 否则旧数据盖掉新结果。
void OrderReloadBeforeConflictingWrites(ScheduleState& state, Function& function,
                                        const std::vector<Operation*>& reloadCopyins)
{
    // 回载 copyin 恒为单输出, 空 vector 取 front 是 UB, 故只在此处判一次。
    if (reloadCopyins.empty() || reloadCopyins.front()->GetOOperands().empty()) {
        return;
    }
    // 本轮回载共用一个 memId(分片抄自整块), 扫一遍收候选写者就够。
    int memId = reloadCopyins.front()->GetOOperands().front()->memoryrange.memId;
    std::unordered_set<Operation*> reloadSet(reloadCopyins.begin(), reloadCopyins.end());
    std::vector<Operation*> writers;
    for (auto* op : state.orderedOps) {
        // 同轮分片写的是互不相交的片, 彼此不用排序。
        if (reloadSet.count(op) != 0 || state.IsOpRetired(op) || USE_LESS_OPS.count(op->GetOpcode()) != 0) {
            continue;
        }
        for (const auto& written : op->GetOOperands()) {
            if (written != nullptr && written->memoryrange.memId == memId) {
                writers.push_back(op);
                break;
            }
        }
    }
    for (auto* copyin : reloadCopyins) {
        const LogicalTensorPtr& reloaded = copyin->GetOOperands().front();
        int copyinOrder = state.GetExecOrder(copyin);
        auto& successors = state.depManager.GetSuccessors(copyin);
        for (auto* op : writers) {
            // 判据按代价从低到高短路: 查表, 查已有边, 最后才逐 operand 判重叠。
            if (state.GetExecOrder(op) <= copyinOrder || successors.count(op) != 0 ||
                !ConflictsWith(reloaded, op, memId)) {
                continue;
            }
            // token 落 IR 供下次重建复现, AddDependency 让本轮立刻生效。
            PassTokenUtils::LinkTokenDependency(function, *copyin, *op);
            state.depManager.AddDependency(copyin, op);
            APASS_LOG_DEBUG_F(Elements::Operation, "Spill: order reload %s before conflicting write %s.",
                              state.GetOpInfo(copyin).c_str(), state.GetOpInfo(op).c_str());
        }
    }
}

} // namespace

void SpillEngine::EmitInitDDRBuffer(const LogicalTensorPtr& t, DDRBufferKind kind)
{
    if (t == nullptr)
        return;
    int memId = t->memoryrange.memId;
    if (ddrKindMap_.count(memId) != 0)
        return;
    ddrKindMap_[memId] = kind;
    if (!state_.HasObservers())
        return;
    InitDDRBufferEvent event;
    event.clock = -1;
    event.memId = memId;
    event.kind = kind;
    event.magic = t->GetMagic();
    event.dtype = t->Datatype();
    auto dynShape = t->GetDynValidShape();
    if (!dynShape.empty()) {
        for (const auto& s : dynShape) {
            event.shape.push_back(s.Dump());
        }
    } else {
        for (auto d : t->GetShape()) {
            event.shape.push_back(std::to_string(d));
        }
    }
    for (auto* obs : state_.observers_) {
        obs->OnInitDDRBuffer(event);
    }
}

bool SpillEngine::IsBelongSpillBlackList(int memId, Operation* op)
{
    Operation* spillOp = GetSpillOp(memId);
    if (spillOp == nullptr) {
        return true;
    }
    std::set<Operation*> filterLtags;
    FindFilterLtags(op, filterLtags);
    if (state_.IsOpAllocInSchedInfo(spillOp) || filterLtags.count(spillOp) != 0) {
        return true;
    }
    // 拿不到张量是账对不上, 那是 bug 不是规避, 留给 SpillBuffer 报错。
    LogicalTensorPtr spillTensor = GetSpillTensor(spillOp, memId);
    if (spillTensor == nullptr) {
        return false;
    }
    SpillPlan plan;
    if (PlanSpill(memId, spillTensor, plan) != SUCCESS) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Spill: skip tensor[%d], cannot plan a spill for it.", memId);
        return true;
    }
    if (op->GetAtomicScopeId() >= VF_CLUSTER_ID_START && IsVfTensorSpillBlocked(memId, op->GetAtomicScopeId())) {
        APASS_LOG_DEBUG_F(Elements::Operation, "VF spill blacklist: block memId=%d (scope=%d) triggered by alloc %s.",
                          memId, op->GetAtomicScopeId(), state_.GetOpInfo(op).c_str());
        return true;
    }
    return false;
}

bool SpillEngine::IsVfTensorSpillBlocked(int memId, int triggerScopeId) const
{
    auto scopeIt = state_.vfScopeTensorMemIds.find(triggerScopeId);
    if (scopeIt == state_.vfScopeTensorMemIds.end()) {
        return false;
    }
    return scopeIt->second.count(memId) > 0;
}

void SpillEngine::FindFilterLtags(Operation* allocOp, std::set<Operation*>& filterLtags)
{
    auto dstOpList = state_.depManager.GetSuccessors(allocOp);
    for (auto dstOp : dstOpList) {
        if (COPY_IN_OPS.find(dstOp->GetOpcode()) == COPY_IN_OPS.end()) {
            for (auto& inOp : state_.depManager.GetPredecessors(dstOp)) {
                filterLtags.insert(inOp);
            }
            continue;
        }
        for (auto& dstOpId : state_.depManager.GetSuccessors(dstOp)) {
            auto dstOp_level0 = dstOpId;
            for (auto& inOp : state_.depManager.GetPredecessors(dstOp_level0)) {
                filterLtags.insert(inOp);
            }
        }
    }
}

LogicalTensorPtr SpillEngine::CreateLocalTensor(LogicalTensorPtr spillTensor)
{
    LogicalTensorPtr localTensor = irBuilder_.CreateTensorVar(spillTensor->Datatype(), spillTensor->GetShape(),
                                                              std::vector<SymbolicScalar>{}, spillTensor->Format());
    localTensor->SetMemoryTypeToBe(spillTensor->GetMemoryTypeOriginal());
    localTensor->SetMemoryTypeOriginal(spillTensor->GetMemoryTypeOriginal());
    localTensor->UpdateDynValidShape(spillTensor->GetDynValidShape());
    localTensor->tensor->rawshape = spillTensor->tensor->rawshape;
    localTensor->memoryrange.memId = localTensor->GetRawTensor()->GetRawMagic();
    localTensor->offset = std::vector<int64_t>(localTensor->GetShape().size(), 0);
    APASS_LOG_DEBUG_F(Elements::Operation, "Create local tensor[%d].", localTensor->memoryrange.memId);
    return localTensor;
}

void SpillEngine::RegisterLocalBuffer(const LogicalTensorPtr& localTensor)
{
    int memId = localTensor->memoryrange.memId;
    state_.localBufferMap[memId] = std::make_shared<LocalBuffer>(memId, localTensor->tensor->GetRawDataSize(),
                                                                 state_.CalcTensorMemoryPaddingSize(localTensor),
                                                                 localTensor->GetMemoryTypeOriginal());
}

// rawshape 用镜像里躺的那份数据的: 上溯单源存的是那一跳的输入, 与 spillTensor 隔着 ND2NZ, 排布不同。
LogicalTensorPtr SpillEngine::GetMirrorRef(const SpillPlan& plan, LogicalTensorPtr spillTensor)
{
    if (plan.kind == SpillKind::WalkUp && plan.sources.size() == 1) {
        return plan.sources.front().tensor;
    }
    return spillTensor;
}

// 新开的 workspace 由镜像整块独占, 原点与覆盖范围认自己的。
LogicalTensorPtr SpillEngine::CreateGMTensor(LogicalTensorPtr ref, int spillMemId, DataType dtype)
{
    const std::vector<int64_t>& gmShape = ref->tensor->rawshape;
    std::shared_ptr<RawTensor> gmRawTensor = std::make_shared<RawTensor>(dtype, gmShape, TileOpFormat::TILEOP_ND,
                                                                         "WorkspaceGm");
    LogicalTensorPtr gmTensor = irBuilder_.CreateTensorVar(gmRawTensor, std::vector<int64_t>(gmShape.size(), 0),
                                                           gmShape, std::vector<SymbolicScalar>{});
    gmTensor->SetMemoryTypeToBe(MemoryType::MEM_DEVICE_DDR);
    gmTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    gmTensor->UpdateDynValidShape(ref->GetDynValidShape());
    int64_t baseOffset = 0;
    TileRange range;
    if (ReserveWorkspaceRange(spillMemId, gmTensor->tensor->GetRawDataSize(), baseOffset, range) != SUCCESS) {
        return nullptr;
    }
    gmTensor->SetAttr(OpAttributeKey::workspaceBaseOffset, baseOffset);
    gmTensor->memoryrange = range;
    EmitInitDDRBuffer(gmTensor, DDRBufferKind::SPILL_TEMP);
    APASS_LOG_DEBUG_F(Elements::Operation, "Spill: Create gm tensor[%d].", gmTensor->memoryrange.memId);
    return gmTensor;
}

Status SpillEngine::ReserveWorkspaceRange(int spillMemId, int64_t size, int64_t& baseOffset, TileRange& range)
{
    if (state_.localBufferMap.find(spillMemId) == state_.localBufferMap.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find Tensor[%d] in localBufferMap.", spillMemId);
        return FAILED;
    }
    baseOffset = state_.workspaceOffset;
    range = TileRange(state_.workspaceOffset, state_.workspaceOffset + size, state_.workspaceMemId++);
    state_.workspaceOffset += size;
    return SUCCESS;
}

Operation* SpillEngine::CreateAllocOp(LogicalTensorPtr oOperand)
{
    Opcode opcode = oOperand->GetMemoryTypeOriginal() == MemoryType::MEM_UB ? Opcode::OP_UB_ALLOC : Opcode::OP_L1_ALLOC;
    Operation& allocOp = irBuilder_.CreateTensorOpStmt(function_, opcode, {}, {oOperand});
    allocOp.UpdateLatency(1);
    APASS_LOG_DEBUG_F(Elements::Operation, "Spill: Create %s", state_.GetOpInfo(&allocOp).c_str());
    return &allocOp;
}

void SpillEngine::RegisterTensorAllocOp(Operation* allocOp)
{
    state_.tensorAllocMap[allocOp->GetOutputOperand(0)->memoryrange.memId] = allocOp;
}

Operation* SpillEngine::CloneCopyinOp(Operation* spillOp, LogicalTensorPtr iOperand, LogicalTensorPtr oOperand)
{
    Operation& copyinOp = spillOp->CloneOperation(function_, {iOperand}, {oOperand});
    copyinOp.SetIOpAtt(0, spillOp->GetIOpAttrOffset(0));
    copyinOp.SetOpAttribute(spillOp->GetOpAttribute()->Clone());
    copyinOp.inParamLocation_ = spillOp->inParamLocation_;
    copyinOp.UpdateLatency(DEFAULT_LATENCY);
    APASS_LOG_DEBUG_F(Elements::Operation, "Spill: Clone %s", state_.GetOpInfo(&copyinOp).c_str());
    return &copyinOp;
}

Operation* SpillEngine::CreateCopyinOp(LogicalTensorPtr iOperand, LogicalTensorPtr oOperand,
                                       std::vector<OpImmediate> offset, bool isND2NZ)
{
    Operation& copyinOp = irBuilder_.CreateTensorOpStmt(function_, Opcode::OP_COPY_IN, {iOperand}, {oOperand});
    copyinOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        offset, oOperand->GetMemoryTypeOriginal(), OpImmediate::Specified(oOperand->GetShape()),
        OpImmediate::Specified(oOperand->tensor->GetDynRawShape()),
        OpImmediate::Specified(oOperand->GetDynValidShape())));
    copyinOp.UpdateLatency(DEFAULT_LATENCY);
    bool isCube = true;
    if (oOperand->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
        isCube = false;
    }
    copyinOp.SetAttribute(OpAttributeKey::isCube, isCube);
    if (oOperand->GetMemoryTypeOriginal() == MemoryType::MEM_L1) {
        // 镜像里存的是 ND 才分形: 上溯走过了 ND2NZ, 或 DAV_3510 上 L1 一律按 NZ 存。
        bool needND2NZ = Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510 || isND2NZ;
        auto mode = needND2NZ ? Matrix::CopyInMode::ND2NZ : Matrix::CopyInMode::ND2ND;
        copyinOp.SetAttribute(OpAttributeKey::copyInMode, static_cast<int64_t>(mode));
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "Spill: Create %s", state_.GetOpInfo(&copyinOp).c_str());
    return &copyinOp;
}

Operation* SpillEngine::CreateCopyoutOp(Operation* spillOp, LogicalTensorPtr iOperand, LogicalTensorPtr oOperand,
                                        std::vector<OpImmediate> offset)
{
    Operation& copyoutOp = irBuilder_.CreateTensorOpStmt(function_, Opcode::OP_COPY_OUT, {iOperand}, {oOperand});
    copyoutOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        iOperand->GetMemoryTypeOriginal(), offset, OpImmediate::Specified(iOperand->GetShape()),
        OpImmediate::Specified(iOperand->GetRawTensor()->GetDynRawShape()),
        OpImmediate::Specified(iOperand->GetDynValidShape())));
    if (spillOp->HasAttribute(OpAttributeKey::scaleValue)) {
        Element scaleValue = Element(DataType::DT_UINT64, 0);
        spillOp->GetAttr(OpAttributeKey::scaleValue, scaleValue);
        copyoutOp.SetAttribute(OpAttributeKey::scaleValue, scaleValue);
    }
    bool isCube = true;
    if (iOperand->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
        isCube = false;
    }
    copyoutOp.SetAttribute(OpAttributeKey::isCube, isCube);
    if (iOperand->GetMemoryTypeOriginal() == MemoryType::MEM_L0C) {
        copyoutOp.SetAttribute(OpAttributeKey::copyIsNZ, 0);
    } else if (iOperand->GetMemoryTypeOriginal() == MemoryType::MEM_L1) {
        // L1 的数据本就是 ND 排布, 落盘不做变换。
        copyoutOp.SetAttribute(OpAttributeKey::copyOutMode, static_cast<int64_t>(Matrix::CopyOutMode::ND2ND));
    }
    copyoutOp.UpdateLatency(DEFAULT_LATENCY);
    APASS_LOG_DEBUG_F(Elements::Operation, "Spill: Create %s", state_.GetOpInfo(&copyoutOp).c_str());
    return &copyoutOp;
}

// 整块里的一片视图: 落盘时描述"整块里已产出的那一片", 回载时描述"整块里要装回的那一片"。
LogicalTensorPtr SpillEngine::CreatePartialTensor(LogicalTensorPtr shapeFrom, LogicalTensorPtr wholeTensor,
                                                  const std::vector<int64_t>& toOffset)
{
    LogicalTensorPtr partialTensor = irBuilder_.CreateTensorVar(wholeTensor->Datatype(), shapeFrom->GetShape(),
                                                                std::vector<SymbolicScalar>{}, wholeTensor->Format());
    partialTensor->SetMemoryTypeToBe(wholeTensor->GetMemoryTypeToBe());
    partialTensor->SetMemoryTypeOriginal(wholeTensor->GetMemoryTypeOriginal());
    partialTensor->tensor = wholeTensor->tensor;
    partialTensor->memoryrange.memId = wholeTensor->memoryrange.memId;
    partialTensor->UpdateDynValidShape(shapeFrom->GetDynValidShape());
    partialTensor->offset = toOffset;
    APASS_LOG_DEBUG_F(Elements::Operation, "Spill: Create partial tensor[%d].", partialTensor->memoryrange.memId);
    return partialTensor;
}

Operation* SpillEngine::CreateAssembleOp(LogicalTensorPtr iOperand, LogicalTensorPtr oOperand,
                                         const SpillPartialWrite& partial)
{
    Operation& assembleOp = irBuilder_.CreateTensorOpStmt(function_, Opcode::OP_ASSEMBLE, {iOperand}, {oOperand});
    assembleOp.UpdateLatency(1);
    assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(iOperand->GetMemoryTypeOriginal(), partial.toOffset,
                                                                    partial.toDynOffset, partial.fromDynValidShape));
    assembleOp.SetAttribute(OpAttributeKey::isCube, iOperand->GetMemoryTypeOriginal() != MemoryType::MEM_UB);
    APASS_LOG_DEBUG_F(Elements::Operation, "Spill: Create %s", state_.GetOpInfo(&assembleOp).c_str());
    return &assembleOp;
}

Operation* SpillEngine::GetSpillOp(int memId)
{
    if (state_.tensorOccupyMap.count(memId)) {
        return state_.tensorOccupyMap[memId];
    }
    return nullptr;
}

LogicalTensorPtr SpillEngine::GetSpillTensor(Operation* spillOp, int spillMemId)
{
    for (size_t i = 0; i < spillOp->GetOOperands().size(); i++) {
        if (spillOp->GetOOperands()[i]->memoryrange.memId == spillMemId) {
            return spillOp->GetOutputOperand(i);
        }
    }
    return nullptr;
}

// 没设过与全零都归一成空 —— 都是"这一跳不带偏移"。运行期值原样保留, 后续校验能否在同一参数域重放。
static std::vector<OpImmediate> NormalizeOffset(const std::vector<OpImmediate>& offset)
{
    for (const auto& imm : offset) {
        if (imm.IsParameter()) {
            return offset;
        }
        if (imm.IsSpecified() &&
            (!imm.GetSpecifiedValue().ConcreteValid() || imm.GetSpecifiedValue().Concrete() != 0)) {
            return offset;
        }
    }
    return {};
}

// 归一后的空只能用于判断, 造 op 得写具体值: GM 越界检查要求偏移维数和 rawShape 对齐。
static std::vector<OpImmediate> MirrorOffset(const std::vector<OpImmediate>& offset, LogicalTensorPtr gmTensor)
{
    return offset.empty() ? OpImmediate::Specified(gmTensor->GetOffset()) : offset;
}

// 这个写往它的输出的哪个位置写 (INSERT): 非空说明只填了一片, 落盘要落到镜像的同一位置。
std::vector<OpImmediate> SpillEngine::GetSaveOffset(Operation* writeOp)
{
    if (writeOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
        auto attr = std::dynamic_pointer_cast<AssembleOpAttribute>(writeOp->GetOpAttribute());
        if (attr != nullptr) {
            return NormalizeOffset(OpImmediate::Specified(attr->GetToTensorOffset()));
        }
    }
    auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(writeOp->GetOpAttribute());
    return copyAttr != nullptr ? NormalizeOffset(copyAttr->GetToOffset()) : std::vector<OpImmediate>{};
}

// 这个搬运从它的输入的哪个位置读: 源比落盘目标大时非空 (抠子块), 回载 copyin 要重放同一个偏移。
std::vector<OpImmediate> SpillEngine::GetReloadOffset(Operation* moveOp)
{
    auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(moveOp->GetOpAttribute());
    return copyAttr != nullptr ? NormalizeOffset(copyAttr->GetFromOffset()) : std::vector<OpImmediate>{};
}

// 新 copy 仍在同一个 Function 中: Specified 表达式保留同一组符号, Parameter 索引仍引用同一线性 COA 参数域。
// 只有无效 immediate 或无效参数索引无法安全重放。
bool SpillEngine::CanReplayOffset(const std::vector<OpImmediate>& offset)
{
    for (const auto& imm : offset) {
        if (imm.IsSpecified()) {
            if (!imm.GetSpecifiedValue().IsValid()) {
                return false;
            }
            continue;
        }
        if (!imm.IsParameter() || imm.GetParameterIndex() < 0) {
            return false;
        }
    }
    return true;
}

// 唯一允许穿过的排布变换, 回载 copyin 会重做这一步分形。
// 认 opcode 不认 format: format 默认就是 ND, 按它判会放行 NZ->NZ, 二次分形静默错值。
bool SpillEngine::IsLayoutMove(Operation* op) { return op != nullptr && op->GetOpcode() == Opcode::OP_UB_COPY_ND2NZ; }

// 只搬不算的写才能穿过去上溯: 算子的输出是算出来的, 它的输入里没有这份数据。
// 白名单而非反查通路表 —— 放行一条得先确认这一跳 shape 语义一致、偏移读得出来。
// OP_ASSEMBLE 不收: 同级的不进调度, 穿过它可能溯回同一块 buffer, 等于没换级。
bool SpillEngine::IsPureMove(Operation* op)
{
    static const std::set<Opcode> PURE_MOVE_OPS = {Opcode::OP_UB_COPY_L1, Opcode::OP_L0C_TO_L1, Opcode::OP_L0C_COPY_UB,
                                                   Opcode::OP_COPY_IN};
    if (op->GetInputOperand(0) == nullptr || op->GetOutputOperand(0) == nullptr || op->GetIOperands().size() != 1 ||
        op->GetOOperands().size() != 1) {
        return false;
    }
    return IsLayoutMove(op) || PURE_MOVE_OPS.count(op->GetOpcode()) != 0;
}

// 逐写问: 多写张量没有唯一生产者, 只问唯一生产者会把 NZ 数据漏判成可落盘。
bool SpillEngine::HasLayoutWrite(LogicalTensorPtr tensor)
{
    for (auto* producer : tensor->GetProducers()) {
        if (IsLayoutMove(producer)) {
            return true;
        }
    }
    return false;
}

// 能不能就地落盘: 已经在 DDR, 或有 DDR 通路且排布在回载侧复现得出来。
bool SpillEngine::CanSaveTensorToDDR(LogicalTensorPtr tensor)
{
    if (tensor->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
        return true;
    }
    return MemPathUtils::CanSaveToDDR(tensor->GetMemoryTypeOriginal()) && !HasLayoutWrite(tensor);
}

bool SpillEngine::IsDataComplete(LogicalTensorPtr tensor)
{
    for (auto* writeOp : CollectDataWrites(tensor)) {
        if (!state_.IsOpRetired(writeOp)) {
            return false;
        }
    }
    return true;
}

// 这个写读的就是 DDR, 那份数据不必重新落盘。只认单输入: 克隆时只接一个 iOperand。
bool SpillEngine::IsReadFromDDR(Operation* op)
{
    if (op == nullptr || !OpcodeManager::Inst().IsCopyIn(op->GetOpcode()) || op->GetIOperands().size() != 1 ||
        op->GetOOperands().size() != 1) {
        return false;
    }
    LogicalTensorPtr input = op->GetInputOperand(0);
    return input != nullptr && input->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR;
}

// 一次判完形态, 后三个阶段照它走。通路优先: 落得了盘就在这一级落, 数据齐不齐只影响怎么回载。
SpillKind SpillEngine::DispatchSpill(LogicalTensorPtr spillTensor)
{
    std::vector<Operation*> writes = CollectDataWrites(spillTensor);
    // 认输入在不在 DDR 而不只认 opcode 是 copyin: L1_TO_BT 这类读的是 L1,
    // 那块地随时会被腾走, 克隆它等于把镜像建在活不到回载的 buffer 上。
    if (writes.size() == 1 && IsReadFromDDR(writes[0])) {
        return SpillKind::ReuseDDR;
    }
    if (!CanSaveTensorToDDR(spillTensor)) {
        return SpillKind::WalkUp;
    }
    return IsDataComplete(spillTensor) ? SpillKind::InPlaceWhole : SpillKind::InPlacePartial;
}

Status SpillEngine::CollectSpillSources(LogicalTensorPtr spillTensor, SpillPlan& plan)
{
    if (plan.kind == SpillKind::ReuseDDR) {
        plan.cloneCopyinFrom = CollectDataWrites(spillTensor).front();
        return SUCCESS;
    }
    if (plan.kind == SpillKind::WalkUp) {
        return CollectWalkUpSources(spillTensor, plan);
    }
    return CollectInPlaceSource(spillTensor, plan);
}

// 就地落盘: 镜像与这块 buffer 同排布, 整块搬出整块搬回, 所以只有一份源、不带偏移。
// 只取已执行的写当锚: 未执行的写会改指新 buffer 或随失去读者被删, 与这条 copyout 无关。
Status SpillEngine::CollectInPlaceSource(LogicalTensorPtr spillTensor, SpillPlan& plan)
{
    std::vector<Operation*> retiredWrites;
    for (auto* writeOp : CollectDataWrites(spillTensor)) {
        if (state_.IsOpRetired(writeOp)) {
            retiredWrites.push_back(writeOp);
        }
    }
    if (retiredWrites.empty()) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Spill: tensor[%d] holds nothing produced yet.",
                          spillTensor->memoryrange.memId);
        return FAILED;
    }
    plan.sources.push_back({spillTensor, retiredWrites, {}, true});
    return SUCCESS;
}

// 上溯: 每个写各走自己那一跳, 改从它的输入落盘, 各源带各自的 saveOffset 落进镜像的对应片。
// 上一级的全部写都是锚, 一个都不能漏 —— 漏掉未执行的那个, copyout 会排在它前面搬走缺片的镜像。
Status SpillEngine::CollectWalkUpSources(LogicalTensorPtr spillTensor, SpillPlan& plan)
{
    std::vector<Operation*> writes = CollectDataWrites(spillTensor);
    if (writes.empty()) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Spill: tensor[%d] cannot reach DDR and has no write to walk up.",
                          spillTensor->memoryrange.memId);
        return FAILED;
    }
    // 回载是一条整块 copyin, 带不了逐源各一份偏移, 所以大搬小的偏移整次 spill 只有一个。
    if (writes.size() == 1) {
        plan.reloadOffset = GetReloadOffset(writes[0]);
        if (!CanReplayOffset(plan.reloadOffset)) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Spill: %s has an invalid reload offset.",
                              state_.GetOpInfo(writes[0]).c_str());
            return FAILED;
        }
    }
    for (auto* writeOp : writes) {
        LogicalTensorPtr source = WalkUpOneHop(writeOp, plan);
        if (source == nullptr) {
            return FAILED;
        }
        std::vector<Operation*> anchors = CollectSourceProducers(source);
        if (anchors.empty()) {
            APASS_LOG_DEBUG_F(Elements::Tensor, "Spill: tensor[%d] has no data producer.", source->memoryrange.memId);
            return FAILED;
        }
        bool allRetired = std::all_of(anchors.begin(), anchors.end(),
                                      [this](Operation* anchor) { return state_.IsOpRetired(anchor); });
        plan.sources.push_back({source, anchors, GetSaveOffset(writeOp), allRetired});
    }
    return SUCCESS;
}

// 往上走一跳: 输出是搬来的, 同一份数据在输入里还有一份, 改从输入落盘。
// 一跳后必是 UB 或 L0C, 两者都有 DDR 通路, 所以"没通路"这个原因一步消耗完;
// 剩下唯一落不了盘的是已分形, 那个 op 就是 ND2NZ, 再多走一步取它变换前的 ND。
LogicalTensorPtr SpillEngine::WalkUpOneHop(Operation* writeOp, SpillPlan& plan)
{
    if (!IsPureMove(writeOp)) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Spill: %s computes into a level with no DDR path.",
                          state_.GetOpInfo(writeOp).c_str());
        return nullptr;
    }
    if (!CanReplayOffset(GetSaveOffset(writeOp))) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Spill: %s has an invalid save offset.",
                          state_.GetOpInfo(writeOp).c_str());
        return nullptr;
    }
    LogicalTensorPtr source = writeOp->GetInputOperand(0);
    std::vector<Operation*> writes = CollectDataWrites(source);
    if (writes.size() == 1 && IsLayoutMove(writes[0])) {
        plan.crossedNd2nz = true;
        source = writes[0]->GetInputOperand(0);
    }
    // 溯到 DDR 就放弃: 复用它得让镜像认别人的地, 落位和生命期都不再由这次 spill 说了算。
    if (source->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Spill: tensor[%d] walks up onto DDR, not reused.",
                          source->memoryrange.memId);
        return nullptr;
    }
    if (!CanSaveTensorToDDR(source)) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Spill: tensor[%d] still cannot reach DDR after one hop.",
                          source->memoryrange.memId);
        return nullptr;
    }
    return source;
}

// Anchor the copyout to real writes, not to skip ops (aliases have no schedule state).
// A non-skip write is itself the producer; a skip write reduces through SkipChain to the
// writes feeding its chain tail.
std::vector<Operation*> SpillEngine::CollectSourceProducers(LogicalTensorPtr source)
{
    std::set<Operation*, Operation::OperationComparator> seen;
    std::vector<Operation*> producers;
    for (auto* writeOp : CollectDataWrites(source)) {
        Operation* chainTail = SkipChain(writeOp);
        std::vector<Operation*> realProducers = chainTail == nullptr ? std::vector<Operation*>{writeOp} :
                                                                       CollectDataWrites(chainTail->GetInputOperand(0));
        for (auto* producer : realProducers) {
            if (seen.insert(producer).second) {
                producers.push_back(producer);
            }
        }
    }
    return producers;
}

std::vector<Operation*> SpillEngine::CollectDataWrites(LogicalTensorPtr tensor)
{
    std::vector<Operation*> writes;
    for (auto* producer : tensor->GetProducers()) {
        if (producer != nullptr && !state_.IsOpAllocInSchedInfo(producer)) {
            writes.push_back(producer);
        }
    }
    return writes;
}

// 量化 scale 长在算出这块数据的那个写上 (matmul), 搬运 op 从来不带, 所以供体取生产者。
Operation* SpillEngine::GetScaleDonor(const SpillSource& source)
{
    return source.producerOps.empty() ? nullptr : source.producerOps.front();
}

// 换输入路恒一份镜像; 顶替消费者路 (L0C) 按消费者输出 dtype 分组, 每组一份 ——
// 随路转换只能落在 copyout 上, 不同 dtype 不能共用一份镜像。
Status SpillEngine::CollectMirrorGroups(LogicalTensorPtr spillTensor, bool replaceInput,
                                        std::vector<SpillMirror>& mirrors)
{
    if (replaceInput) {
        mirrors.push_back({spillTensor->Datatype(), {}, nullptr});
        return SUCCESS;
    }
    std::vector<Operation*> consumers;
    for (auto* consumer : spillTensor->GetConsumers()) {
        if (consumer == nullptr || state_.IsOpRetired(consumer)) {
            continue;
        }
        auto output = consumer->GetOutputOperand(0);
        if (output == nullptr) {
            APASS_LOG_WARN_F(Elements::Operation, "L0C spill: skip consumer %s without output operand.",
                             state_.GetOpInfo(consumer).c_str());
            continue;
        }
        auto outMem = output->GetMemoryTypeOriginal();
        if (outMem == MemoryType::MEM_DEVICE_DDR) {
            continue;
        }
        if (outMem != MemoryType::MEM_UB && outMem != MemoryType::MEM_L1) {
            APASS_LOG_WARN_F(Elements::Operation, "L0C spill: skip consumer %s with output memType %s.",
                             state_.GetOpInfo(consumer).c_str(), MemoryTypeToString(outMem).c_str());
            continue;
        }
        consumers.push_back(consumer);
    }
    if (consumers.empty()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Spill: tensor[%d] has no pure-move consumer to replace.",
                          spillTensor->memoryrange.memId);
        return FAILED;
    }
    std::sort(consumers.begin(), consumers.end(),
              [this](Operation* a, Operation* b) { return state_.GetExecOrder(a) < state_.GetExecOrder(b); });
    std::map<DataType, std::vector<Operation*>> groups;
    for (auto* consumer : consumers) {
        groups[consumer->GetOutputOperand(0)->Datatype()].push_back(consumer);
    }
    for (auto& [dtype, group] : groups) {
        mirrors.push_back({dtype, group, nullptr});
    }
    return SUCCESS;
}

Status SpillEngine::UpdateOperationInput(Operation* targetOp, Operation* spillOp, LogicalTensorPtr reloadTensor,
                                         int spillMemId)
{
    for (size_t index = 0; index < targetOp->GetIOperands().size(); index++) {
        if (targetOp->GetIOperands()[index]->memoryrange.memId != spillMemId) {
            continue;
        }
        for (auto& inOp : targetOp->GetIOperands()[index]->GetProducers()) {
            if (IsSkipOp(*inOp)) {
                if (UpdateSkipOpInput(inOp, spillOp, targetOp, reloadTensor, index) != SUCCESS) {
                    return FAILED;
                }
            } else if (inOp == spillOp) {
                targetOp->UpdateInputOperand(index, reloadTensor);
            }
        }
    }
    return SUCCESS;
}

// 一律重建一条链, 不判是否被共享: 原链留给别的读者 (含已发射的) 永远安全,
// 代价只是多几个 skip op —— 它们零 codegen、不申请内存、不进流水线。
Status SpillEngine::UpdateSkipOpInput(Operation* chainTail, Operation* spillOp, Operation* targetOp,
                                      LogicalTensorPtr reloadTensor, size_t index)
{
    // 从链尾往上取: 链中间允许有旁支消费者, 沿 consumers 走会在分叉处停下。
    std::vector<Operation*> chain = SkipChainPath(chainTail);
    if (chain.empty()) {
        return SUCCESS;
    }
    std::reverse(chain.begin(), chain.end());
    const auto& producers = chain.front()->GetInputOperand(0)->GetProducers();
    if (producers.find(spillOp) == producers.end()) {
        return SUCCESS;
    }

    LogicalTensorPtr clonedTail = CloneSkipChain(targetOp, chain, reloadTensor);
    if (clonedTail == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Clone skip chain failed for %s.", state_.GetOpInfo(targetOp).c_str());
        return FAILED;
    }
    targetOp->UpdateInputOperand(index, clonedTail);
    DetachOrphanedSkipChain(chain, targetOp);
    return SUCCESS;
}

void SpillEngine::UnregisterSkipOp(Operation* targetOp, Operation* skipOp)
{
    auto& skipOps = state_.schedInfoMap[targetOp].skipOps;
    auto pos = std::find(skipOps.begin(), skipOps.end(), skipOp);
    if (pos != skipOps.end()) {
        skipOps.erase(pos);
    }
}

// 从链尾往链首扫: 摘掉尾巴才会让它的上游变成无读者, 一趟连锁清干净。
void SpillEngine::DetachOrphanedSkipChain(const std::vector<Operation*>& chain, Operation* targetOp)
{
    for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
        Operation* oldOp = *it;
        UnregisterSkipOp(targetOp, oldOp);
        const auto& consumers = oldOp->GetOutputOperand(0)->GetConsumers();
        if (std::any_of(consumers.begin(), consumers.end(), [](Operation* c) { return !c->IsDeleted(); })) {
            continue;
        }
        EraseSchedulerSideMaps(oldOp);
        oldOp->SetAsDeleted();
        APASS_LOG_DEBUG_F(Elements::Operation, "Detached orphaned skip op %s.", state_.GetOpInfo(oldOp).c_str());
    }
}

// Clone(function_, true) 已深拷 shape/offset/dtype/format 等, 这里只覆写 memId。
LogicalTensorPtr SpillEngine::CloneSkipChain(Operation* targetOp, const std::vector<Operation*>& chain,
                                             LogicalTensorPtr reloadTensor)
{
    std::vector<Operation*> clones;
    LogicalTensorPtr inTensor = reloadTensor;
    for (Operation* op : chain) {
        LogicalTensorPtr outTensor = op->GetOutputOperand(0)->Clone(function_, true);
        if (outTensor == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Clone skip op %s operand failed.", state_.GetOpInfo(op).c_str());
            return nullptr;
        }
        outTensor->memoryrange.memId = reloadTensor->memoryrange.memId;
        Operation& cloneOp = op->CloneOperation(function_, {inTensor}, {outTensor});
        UpdateOpInternalSubgraphID(cloneOp, op);
        clones.push_back(&cloneOp);
        inTensor = outTensor;
    }
    auto& skipOps = state_.schedInfoMap[targetOp].skipOps;
    skipOps.insert(skipOps.end(), clones.begin(), clones.end());
    return inTensor;
}

void SpillEngine::ReplaceSkipOpChainMemId(LogicalTensorPtr startTensor, int oldMemId, int newMemId)
{
    std::vector<Operation*> skipConsumers;
    for (auto* consumer : startTensor->GetConsumers()) {
        if (IsSkipOp(*consumer)) {
            skipConsumers.push_back(consumer);
        }
    }

    while (!skipConsumers.empty()) {
        Operation* skipOp = skipConsumers.back();
        skipConsumers.pop_back();
        auto skipOutTensor = skipOp->GetOutputOperand(0);
        if (skipOutTensor == nullptr) {
            continue;
        }
        if (skipOutTensor->memoryrange.memId == oldMemId) {
            skipOutTensor->memoryrange.memId = newMemId;
        }
        for (auto* consumer : skipOutTensor->GetConsumers()) {
            if (IsSkipOp(*consumer)) {
                skipConsumers.push_back(consumer);
            }
        }
    }
}

bool SpillEngine::RemapOpReqMemId(Operation* op, int oldMemId, int newMemId)
{
    bool changed = false;
    auto& reqMemIds = state_.opReqMemIdsMap[op];
    for (auto memId : reqMemIds) {
        if (memId == oldMemId || memId == newMemId) {
            state_.bufRefCount[newMemId]++;
        }
        if (memId == oldMemId) {
            std::replace(reqMemIds.begin(), reqMemIds.end(), oldMemId, newMemId);
            changed = true;
        }
    }
    return changed;
}

bool SpillEngine::ReplaceTensorMemId(Operation* op, int oldMemId, int newMemId)
{
    bool changed = false;
    for (auto& outTensor : op->GetOOperands()) {
        if (outTensor->memoryrange.memId == oldMemId) {
            outTensor->memoryrange.memId = newMemId;
            ReplaceSkipOpChainMemId(outTensor, oldMemId, newMemId);
            changed = true;
        }
    }
    return changed;
}

void SpillEngine::UpdateOpInternalSubgraphID(Operation& op, Operation* srcOp)
{
    if (srcOp->GetInternalSubgraphID() != NOT_IN_SUBGRAPH) {
        op.UpdateInternalSubgraphID(srcOp->GetInternalSubgraphID());
        op.SetAIVCore(srcOp->GetAIVCore());
    }
}

Status SpillEngine::UpdateSpillOpDepend(Operation* spillOp, LogicalTensorPtr newTensor, int spillMemId,
                                        std::vector<Operation*>& rewired)
{
    const auto successors = state_.depManager.GetSuccessors(spillOp);
    for (auto succOp : successors) {
        if (!state_.schedInfoMap[succOp].isRetired) {
            auto& reqMemIds = state_.opReqMemIdsMap[succOp];
            if (std::count(reqMemIds.begin(), reqMemIds.end(), spillMemId) > 0) {
                if (UpdateOperationInput(succOp, spillOp, newTensor, spillMemId) != SUCCESS) {
                    return FAILED;
                }
                rewired.push_back(succOp);
            }
        }
    }
    return SUCCESS;
}

Status SpillEngine::SpillBuffer(int memId, Operation* spillAllocOp, SpillContext& ctx)
{
    Operation* spillOp = GetSpillOp(memId);
    if (spillOp == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d] occupy op.", memId);
        return FAILED;
    }
    if (state_.IsOpAllocInSchedInfo(spillOp)) {
        return SUCCESS;
    }
    LogicalTensorPtr spillTensor = GetSpillTensor(spillOp, memId);
    if (spillTensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Find %s spill tensor[%d] failed.", state_.GetOpInfo(spillOp).c_str(),
                          memId);
        return FAILED;
    }

    SpillPlan plan;
    // 兜底 spill-all 走的是全池, 不过候选筛选, 所以这里会收到排不出计划的 buffer。
    // 腾不动就跳过这块、换下一块, 不挂整个调度。
    if (PlanSpill(memId, spillTensor, plan) != SUCCESS) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Spill: skip tensor[%d], cannot plan a spill for it.", memId);
        return SUCCESS;
    }
    if (SaveToDDR(memId, spillTensor, plan, ctx) != SUCCESS ||
        ReloadFromDDR(memId, spillTensor, spillOp, spillAllocOp, plan, ctx) != SUCCESS) {
        return FAILED;
    }
    if (FinalizeSpill(memId, spillTensor, spillAllocOp, plan, ctx) != SUCCESS) {
        return FAILED;
    }
    // 必须在 FinalizeSpill 之后: 更早 memId 还没刷完。
    OrderReloadBeforeConflictingWrites(state_, function_, plan.reloadCopyinOps);
    return SUCCESS;
}

// 阶段①: 解析源、定镜像分组、必要时定逐片回载的计划, 纯读。
// 回载侧按通路图分派: 能从 DDR 直搬回这一级就换输入, 否则顶替消费者。
// 排不出计划就是"这块存不下去", 只记 DEBUG, 报不报错由调用侧定。
Status SpillEngine::PlanSpill(int memId, LogicalTensorPtr spillTensor, SpillPlan& plan)
{
    auto bufIt = state_.localBufferMap.find(memId);
    if (bufIt == state_.localBufferMap.end() || bufIt->second == nullptr) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Spill: tensor[%d] has no local buffer record.", memId);
        return FAILED;
    }
    plan.replaceInput = MemPathUtils::CanReloadFromDDR(bufIt->second->memType);
    plan.kind = DispatchSpill(spillTensor);
    // 顶替消费者 (L0C) 时新数据直接写进消费者的输出, 没有整块可拼, 分不了片。
    if (plan.kind == SpillKind::InPlacePartial && !plan.replaceInput) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Spill: tensor[%d] cannot reload partially while replacing consumers.",
                          memId);
        return FAILED;
    }
    if (plan.kind == SpillKind::ReuseDDR && !plan.replaceInput) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Spill: tensor[%d] cannot reuse DDR while replacing consumers.", memId);
        return FAILED;
    }
    if (CollectSpillSources(spillTensor, plan) != SUCCESS) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Spill: plan sources of tensor[%d] failed.", memId);
        return FAILED;
    }
    if (plan.kind != SpillKind::ReuseDDR &&
        CollectMirrorGroups(spillTensor, plan.replaceInput, plan.mirrors) != SUCCESS) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Spill: plan mirrors of tensor[%d] failed.", memId);
        return FAILED;
    }
    if (plan.kind == SpillKind::InPlacePartial && CollectPartialWrites(spillTensor, plan.partialWrites) != SUCCESS) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Spill: tensor[%d] cannot replay its partial writes.", memId);
        return FAILED;
    }
    APASS_LOG_DEBUG_F(Elements::Tensor, "Spill plan: tensor[%d] kind %d, %zu sources, crossedNd2nz %d.", memId,
                      static_cast<int>(plan.kind), plan.sources.size(), static_cast<int>(plan.crossedNd2nz));
    for (const auto& source : plan.sources) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Spill source: tensor[%d] from %s, saveOffset %zu, producedInPast %d.",
                          source.tensor->memoryrange.memId, state_.GetOpInfo(GetScaleDonor(source)).c_str(),
                          source.saveOffset.size(), static_cast<int>(source.producedInPast));
    }
    return SUCCESS;
}

// 取这个写在整块里的落点, 只有两种表达: assemble 的记在 AssembleOpAttribute, 搬运的记在 toOffset。
Status SpillEngine::GetPartialWriteReplayAttr(Operation* writeOp, SpillPartialWrite& partial)
{
    if (writeOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
        auto attr = std::dynamic_pointer_cast<AssembleOpAttribute>(writeOp->GetOpAttribute());
        if (attr == nullptr) {
            return FAILED;
        }
        partial.toOffset = attr->GetToOffset();
        partial.toDynOffset = attr->GetToDynOffset();
        partial.fromDynValidShape = attr->GetFromDynValidShape();
        return SUCCESS;
    }
    auto attr = std::dynamic_pointer_cast<CopyOpAttribute>(writeOp->GetOpAttribute());
    if (attr == nullptr) {
        return FAILED;
    }
    // 符号值搬进新建的 assemble 属性后不保证仍指向同一处。
    for (const auto& imm : attr->GetToOffset()) {
        if (!imm.IsSpecified() || !imm.GetSpecifiedValue().ConcreteValid()) {
            return FAILED;
        }
        partial.toOffset.push_back(imm.GetSpecifiedValue().Concrete());
    }
    partial.fromDynValidShape = writeOp->GetInputOperand(0)->GetDynValidShape();
    if (partial.fromDynValidShape.empty()) {
        partial.fromDynValidShape = OpImmediate::ToSpecified(attr->GetToDynValidShape());
    }
    return SUCCESS;
}

// 数据没齐时逐片回载的计划: 一个写一片。已执行的从镜像读回, 未执行的改指向新 buffer。
Status SpillEngine::CollectPartialWrites(LogicalTensorPtr spillTensor, std::vector<SpillPartialWrite>& partialWrites)
{
    for (auto* writeOp : CollectDataWrites(spillTensor)) {
        SpillPartialWrite partial;
        partial.writeOp = writeOp;
        partial.producedInPast = state_.IsOpRetired(writeOp);
        if (GetPartialWriteReplayAttr(writeOp, partial) != SUCCESS) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Spill: %s cannot be replayed as a partial write.",
                              state_.GetOpInfo(writeOp).c_str());
            return FAILED;
        }
        partialWrites.push_back(std::move(partial));
    }
    return SUCCESS;
}

// 阶段②: 每份镜像一块 workspace, 各源的 copyout 往各自区间写。ReuseDDR 不落盘。
Status SpillEngine::SaveToDDR(int memId, LogicalTensorPtr spillTensor, SpillPlan& plan, SpillContext& ctx)
{
    if (plan.kind == SpillKind::ReuseDDR) {
        return SUCCESS;
    }
    for (auto& mirror : plan.mirrors) {
        if (PrepareSpillMirror(memId, plan, spillTensor, mirror) != SUCCESS ||
            SaveSourcesToDDR(mirror.gmTensor, ctx, plan) != SUCCESS) {
            return FAILED;
        }
    }
    return SUCCESS;
}

// 阶段③: 能从 DDR 搬回这一级就换输入 (镜像恒只有一份), 搬不回去就按 dtype 逐份顶替消费者。
Status SpillEngine::ReloadFromDDR(int memId, LogicalTensorPtr spillTensor, Operation* spillOp, Operation* spillAllocOp,
                                  SpillPlan& plan, SpillContext& ctx)
{
    if (plan.replaceInput) {
        return ReloadIntoNewBuffer(memId, spillTensor, spillOp, spillAllocOp, plan, ctx);
    }
    std::vector<Operation*> seeds = plan.saveCopyoutOps;
    for (const auto& mirror : plan.mirrors) {
        if (ReplaceConsumersWithCopyin(mirror, spillAllocOp, plan.created, seeds) != SUCCESS) {
            return FAILED;
        }
    }
    if (state_.depManager.RefreshDependencies(seeds, state_.tensorAllocMap) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "RefreshDependencies failed.");
        return FAILED;
    }
    state_.bufRefCount[memId] = 0;
    return SUCCESS;
}

// 阶段④: 删掉没有活读者的写, 通知观察者, 释放这块 buffer。
Status SpillEngine::FinalizeSpill(int memId, LogicalTensorPtr spillTensor, Operation* spillAllocOp, SpillPlan& plan,
                                  SpillContext& ctx)
{
    // 释放的是被腾空那块地所在的核, 这个事实在收割之前就已确定; 收割会抹掉 tensorAllocMap
    // 的条目, 之后再解析就查不到 alloc 了。
    CoreLocationType freeCore = state_.enableDualDst ? state_.ResolveCoreForFree(memId) :
                                                       state_.GetCoreLocation(spillAllocOp);
    if (DropWritesWithoutReaders(spillTensor, plan, ctx) != SUCCESS) {
        return FAILED;
    }
    NotifySpill(state_, spillTensor, memId, spillAllocOp, plan.created);
    return FreeSpilledBuffer(memId, freeCore);
}

// ③ 的顶替消费者版: 消费者本是纯搬运, 由 copyin 从 DDR 直写它的输出即等价, 整个换掉。
Status SpillEngine::ReplaceConsumersWithCopyin(const SpillMirror& mirror, Operation* spillAllocOp,
                                               SingleSpillCreatedOps& created, std::vector<Operation*>& seeds)
{
    for (auto* consumer : mirror.consumers) {
        auto oOperand = consumer->GetOutputOperand(0);
        Operation* copyinOp = CreateCopyinOp(mirror.gmTensor, oOperand,
                                             OpImmediate::Specified(mirror.gmTensor->GetOffset()), true);
        UpdateOpScheduleInfo(copyinOp, {oOperand->memoryrange.memId}, spillAllocOp);
        for (auto* pred : state_.depManager.GetPredecessors(consumer)) {
            seeds.push_back(pred);
        }
        for (auto* succ : state_.depManager.GetSuccessors(consumer)) {
            seeds.push_back(succ);
        }
        TakeOverScheduleSlot(consumer, copyinOp);
        consumer->SetAsDeleted();
        seeds.push_back(copyinOp);
    }
    created.Record(nullptr, nullptr, nullptr, mirror.gmTensor);
    return SUCCESS;
}

Status SpillEngine::PrepareSpillMirror(int spillMemId, const SpillPlan& plan, LogicalTensorPtr spillTensor,
                                       SpillMirror& mirror)
{
    mirror.gmTensor = CreateGMTensor(GetMirrorRef(plan, spillTensor), spillMemId, mirror.dtype);
    if (mirror.gmTensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Spill: create mirror for tensor[%d] failed.", spillMemId);
        return FAILED;
    }
    return SUCCESS;
}

Status SpillEngine::SaveSourcesToDDR(LogicalTensorPtr gmTensor, SpillContext& ctx, SpillPlan& plan)
{
    for (const auto& source : plan.sources) {
        int sourceMemId = source.tensor->memoryrange.memId;
        Operation* copyoutOp = CreateCopyoutOp(GetScaleDonor(source), source.tensor, gmTensor,
                                               MirrorOffset(source.saveOffset, gmTensor));
        if (UpdateCopyoutScheduleInfo(copyoutOp, source, sourceMemId) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Spill: update copyout schedule info failed.");
            return FAILED;
        }
        plan.created.Record(copyoutOp);
        plan.saveCopyoutOps.push_back(copyoutOp);
        if (source.producedInPast) {
            ctx.newCopyoutOps.push_back(copyoutOp);
        } else {
            // 源还没产出: 先给源 buffer 加引用, 保它活到 copyout 执行。
            state_.bufRefCount[sourceMemId]++;
            ctx.newNotRetiredCopyOutSize++;
        }
    }
    return SUCCESS;
}

// 新 buffer 与 spillTensor 同 memType、同 rawshape, 消费者原本那套 offset/shape 因此继续有效。
Status SpillEngine::ReloadIntoNewBuffer(int spillMemId, LogicalTensorPtr spillTensor, Operation* spillOp,
                                        Operation* spillAllocOp, SpillPlan& plan, SpillContext& ctx)
{
    // ReuseDDR 没建镜像, 数据源就是原来那条 copyin 读的那块 DDR。
    LogicalTensorPtr gmTensor = plan.kind == SpillKind::ReuseDDR ? plan.cloneCopyinFrom->GetInputOperand(0) :
                                                                   plan.mirrors.front().gmTensor;
    LogicalTensorPtr localTensor = CreateLocalTensor(spillTensor);
    RegisterLocalBuffer(localTensor);
    Operation* allocOp = CreateAllocOp(localTensor);
    RegisterTensorAllocOp(allocOp);
    OpMemIdMap opMemIdMap = {{allocOp, {localTensor->memoryrange.memId}}};
    Operation* wholeCopyin = nullptr;
    if (plan.kind == SpillKind::InPlacePartial) {
        CreatePartialReloads(spillTensor, localTensor, gmTensor, plan.partialWrites, opMemIdMap);
    } else {
        wholeCopyin = CreateWholeReload(gmTensor, localTensor, plan);
        opMemIdMap.push_back({wholeCopyin, {localTensor->memoryrange.memId}});
    }
    if (UpdateScheduleStatus(opMemIdMap, spillMemId, spillAllocOp, localTensor, spillOp, plan) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Spill: update schedule status failed.");
        return FAILED;
    }
    ctx.newAllocOps.push_back(allocOp);
    // 分片回载下 created.copyinOp 为空, opMemIdMap 才记全了本轮产物。
    for (const auto& entry : opMemIdMap) {
        if (entry.first == allocOp || USE_LESS_OPS.count(entry.first->GetOpcode()) != 0) {
            continue;
        }
        plan.reloadCopyinOps.push_back(entry.first);
    }
    plan.created.Record(nullptr, allocOp, wholeCopyin, gmTensor);
    return SUCCESS;
}

// 上溯单写时镜像存的是那一跳的输入 (更大), reloadOffset 重放它抠子块的偏移; 其余读镜像原点。
// crossedNd2nz 时镜像里是变换前的 ND, 回载要重做分形, 否则消费者按 NZ 读到排布不对的数据。
Operation* SpillEngine::CreateWholeReload(LogicalTensorPtr gmTensor, LogicalTensorPtr localTensor,
                                          const SpillPlan& plan)
{
    if (plan.kind == SpillKind::ReuseDDR) {
        return CloneCopyinOp(plan.cloneCopyinFrom, gmTensor, localTensor);
    }
    std::vector<OpImmediate> fromOffset = MirrorOffset(plan.reloadOffset, gmTensor);
    return CreateCopyinOp(gmTensor, localTensor, fromOffset, plan.crossedNd2nz);
}

// assemble 上挂着的 alloc 认的是分片视图, 而未执行的写马上按整块改指向, alloc 得跟着认整块。
void SpillEngine::NormalizeAssembleAllocOutput(LogicalTensorPtr spillTensor)
{
    for (auto* producer : spillTensor->GetProducers()) {
        if (producer == nullptr || producer->GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        for (auto* pre : producer->ProducerOps()) {
            if (state_.IsOpAllocInSchedInfo(pre)) {
                pre->UpdateOutputOperand(0, spillTensor);
            }
        }
    }
}

// 改指放在最后: 分片都靠老 spillTensor 的生产者关系定位, 改早了就找不着了。
void SpillEngine::CreatePartialReloads(LogicalTensorPtr spillTensor, LogicalTensorPtr localTensor,
                                       LogicalTensorPtr gmTensor, const std::vector<SpillPartialWrite>& partialWrites,
                                       OpMemIdMap& opMemIdMap)
{
    NormalizeAssembleAllocOutput(spillTensor);
    std::vector<Operation*> writesToRedirect;
    for (const auto& partial : partialWrites) {
        if (partial.producedInPast) {
            CreateOnePartialReload(localTensor, gmTensor, partial, opMemIdMap);
        } else {
            writesToRedirect.push_back(partial.writeOp);
        }
    }
    for (auto* writeOp : writesToRedirect) {
        writeOp->ReplaceOutput(localTensor, spillTensor);
        APASS_LOG_DEBUG_F(Elements::Operation, "Spill: redirect %s to the reloaded buffer.",
                          state_.GetOpInfo(writeOp).c_str());
    }
}

// assemble 读写同一块 buffer, 所以它的 memId 要报两次。
void SpillEngine::CreateOnePartialReload(LogicalTensorPtr localTensor, LogicalTensorPtr gmTensor,
                                         const SpillPartialWrite& partial, OpMemIdMap& opMemIdMap)
{
    int memId = localTensor->memoryrange.memId;
    LogicalTensorPtr writeInput = partial.writeOp->GetInputOperand(0);
    LogicalTensorPtr partialTensor = CreatePartialTensor(writeInput, localTensor, partial.toOffset);
    Operation* copyinOp = CreateCopyinOp(gmTensor, partialTensor, OpImmediate::Specified(partial.toOffset));
    Operation* assembleOp = CreateAssembleOp(partialTensor, localTensor, partial);
    // 这一片在镜像里是 NZ 还是 ND, 由写它的那个 op 决定, 读回来的两条都得照着认。
    int64_t isNZ = 0;
    partial.writeOp->GetAttr(OpAttributeKey::copyIsNZ, isNZ);
    copyinOp->SetAttr(OpAttributeKey::copyIsNZ, isNZ);
    assembleOp->SetAttr(OpAttributeKey::copyIsNZ, isNZ);
    opMemIdMap.push_back({copyinOp, {memId}});
    opMemIdMap.push_back({assembleOp, {memId, memId}});
}

bool SpillEngine::HasLiveConsumer(const LogicalTensorPtr& tensor)
{
    if (tensor == nullptr) {
        return false;
    }
    const auto& consumers = tensor->GetConsumers();
    return std::any_of(consumers.begin(), consumers.end(),
                       [](Operation* consumer) { return consumer != nullptr && !consumer->IsDeleted(); });
}

bool SpillEngine::HasLiveReader(Operation* op)
{
    for (const auto& outTensor : op->GetOOperands()) {
        if (HasLiveConsumer(outTensor)) {
            return true;
        }
    }
    return false;
}

// 消费者全改指到新 buffer 后, 原来往这块地写的搬运就白干了, 连同它们腾空的 buffer 一起收走。
// 就地落盘那两种形态的 copyout 读的就是这块地, 恒有活读者, 进不来。
// 改指向已经把这些写的输出 memId 换成了新 buffer 的, 所以擦引用记录时要把新 buffer 摘出去:
// 它刚建好、还没退休, 记录一抹回载的 alloc 退休时就找不到自己的引用。
Status SpillEngine::DropWritesWithoutReaders(LogicalTensorPtr spillTensor, SpillPlan& plan, SpillContext& ctx)
{
    if (HasLiveConsumer(spillTensor)) {
        return SUCCESS;
    }
    OrphanedOps orphaned = CollectOrphanedChain(spillTensor);
    if (orphaned.ops.empty()) {
        return SUCCESS;
    }
    if (plan.created.allocOp != nullptr && plan.created.allocOp->GetOutputOperand(0) != nullptr) {
        orphaned.memIds.erase(plan.created.allocOp->GetOutputOperand(0)->memoryrange.memId);
    }
    size_t retiredNum = 0;
    for (auto* op : orphaned.ops) {
        if (state_.IsOpAllocInSchedInfo(op)) {
            ctx.deleteAllocOps.push_back(
                {op, op->GetOutputOperand(0)->GetMemoryTypeOriginal(), state_.schedInfoMap[op].coreLocation});
        }
        if (DeleteOneOp(op, orphaned.memIds)) {
            retiredNum++;
        }
    }
    ctx.deleteRetiredOpSize += static_cast<int>(retiredNum);
    ctx.deleteNotRetiredOpSize += static_cast<int>(orphaned.ops.size() - retiredNum);
    for (int memId : orphaned.memIds) {
        state_.tensorAllocMap.erase(memId);
        state_.bufRefCount.erase(memId);
    }
    DetachOrphanedProducers(orphaned);
    // EraseOperations 不碰 VarDependency, 残留的死 op 会在下一轮重建时挂死调度。
    PassTokenUtils::CleanupDeletedTokenDependency(function_, orphaned.ops);
    function_.EraseOperations(false, false);
    APASS_LOG_DEBUG_F(Elements::Operation, "Spill: dropped %zu writes without readers.", orphaned.ops.size());
    return SUCCESS;
}

// 逐级上溯: 删掉一层搬运, 它的输入那一级也可能失去唯一读者, 不继续删就漏一块回不来的 buffer。
// 天然停在数据源上 —— 源上挂着这次 spill 新插的 copyout, 永远有活读者。
// 还有别的活消费者的那一级整个留下: 它仍在服役, 抹掉 memId 会让别人退休时找不到引用记录。
OrphanedOps SpillEngine::CollectOrphanedChain(LogicalTensorPtr spillTensor)
{
    OrphanedOps orphaned;
    std::vector<LogicalTensorPtr> pending = {spillTensor};
    std::set<LogicalTensor*> visited;
    while (!pending.empty()) {
        LogicalTensorPtr tensor = pending.back();
        pending.pop_back();
        if (tensor == nullptr || !visited.insert(tensor.get()).second || HasLiveConsumer(tensor)) {
            continue;
        }
        orphaned.tensors.push_back(tensor);
        orphaned.memIds.insert(tensor->memoryrange.memId);
        for (auto* writeOp : tensor->GetProducers()) {
            if (writeOp == nullptr || HasLiveReader(writeOp)) {
                continue;
            }
            orphaned.ops.push_back(writeOp);
            // 先标删: 它不再算活读者, 上一级下一轮才判得出自己也没读者了。
            writeOp->SetAsDeleted();
            if (!state_.IsOpAllocInSchedInfo(writeOp)) {
                pending.push_back(writeOp->GetInputOperand(0));
            }
        }
    }
    return orphaned;
}

// op 的存在分布在五处: 执行序、buffer 引用、调度侧各表、依赖边、新增 op 台账, 一处不摘就留下野记录。
// 返回它是否已执行过, 供调用侧区分两类计数。
bool SpillEngine::DeleteOneOp(Operation* op, const std::set<int>& orphanedMemIds)
{
    bool wasRetired = EraseFromExecOrder(op);
    ReleaseOpBufRefs(op, orphanedMemIds);
    EraseSchedulerSideMaps(op);

    auto newOpsIt = std::find(state_.newOperations.begin(), state_.newOperations.end(), op);
    if (newOpsIt != state_.newOperations.end()) {
        state_.newOperations.erase(newOpsIt);
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "Spill: deleted op %s.", state_.GetOpInfo(op).c_str());
    return wasRetired;
}

// 执行序是连号的, 抽掉一个后面每个都要挪一位。
bool SpillEngine::EraseFromExecOrder(Operation* op)
{
    auto it = std::find(state_.orderedOps.begin(), state_.orderedOps.end(), op);
    if (it == state_.orderedOps.end()) {
        return false;
    }
    bool wasRetired = state_.schedInfoMap[op].isRetired;
    int deletedOrder = state_.schedInfoMap[op].execOrder;
    auto nextIt = state_.orderedOps.erase(it);
    for (auto adjustIt = nextIt; adjustIt != state_.orderedOps.end(); adjustIt++) {
        if (state_.schedInfoMap.count(*adjustIt) > 0 && state_.schedInfoMap[*adjustIt].execOrder > deletedOrder) {
            state_.schedInfoMap[*adjustIt].execOrder--;
        }
    }
    return wasRetired;
}

// 未执行的写被删 -> 它占的引用不会再由退休来释放, 就地还掉。
// 整块腾空的那些跳过: 引用记录随后整条抹除, 逐个减到 0 反而会让别处误判已释放。
void SpillEngine::ReleaseOpBufRefs(Operation* op, const std::set<int>& orphanedMemIds)
{
    if (state_.schedInfoMap[op].isRetired) {
        return;
    }
    for (int memId : state_.GetOpMemIds(op)) {
        if (orphanedMemIds.count(memId) > 0) {
            continue;
        }
        auto refIt = state_.bufRefCount.find(memId);
        if (refIt != state_.bufRefCount.end() && refIt->second > 0) {
            refIt->second--;
        }
    }
}

// 已删的写还挂在各级 tensor 的生产/消费表里, 不摘掉重建依赖时会走到已删对象。
void SpillEngine::DetachOrphanedProducers(const OrphanedOps& orphaned)
{
    for (const auto& tensor : orphaned.tensors) {
        for (auto* op : orphaned.ops) {
            tensor->RemoveProducer(op);
            tensor->RemoveConsumer(op);
        }
    }
}

Status SpillEngine::FreeSpilledBuffer(int memId, CoreLocationType freeCore)
{
    if (state_.bufferManagerMap[freeCore][state_.localBufferMap[memId]->memType].Free(memId) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Free spill tensor[%d] failed!", memId);
        return FAILED;
    }
    state_.tensorOccupyMap.erase(memId);
    return SUCCESS;
}

void SpillEngine::TakeOverScheduleSlot(Operation* oldOp, Operation* newOp)
{
    // 顶替不是删除, oldOp 两侧的序约束要整体过户。
    PassTokenUtils::TransferTokenDependency(function_, *oldOp, *newOp);
    std::replace(state_.orderedOps.begin(), state_.orderedOps.end(), oldOp, newOp);
    APASS_LOG_DEBUG_F(Elements::Operation, "Replace %s with %s in exec order.", state_.GetOpInfo(oldOp).c_str(),
                      state_.GetOpInfo(newOp).c_str());
    EraseSchedulerSideMaps(oldOp);
}

void SpillEngine::EraseSchedulerSideMaps(Operation* op)
{
    auto it = std::find(state_.orderedOps.begin(), state_.orderedOps.end(), op);
    if (it != state_.orderedOps.end()) {
        state_.orderedOps.erase(it);
    }
    state_.schedInfoMap.erase(op);
    state_.opReqMemIdsMap.erase(op);
    state_.inOutOperandsCache.erase(op);
    state_.depManager.RemoveOp(op);
}

int SpillEngine::GetBufNextUseTime(int curMemId)
{
    for (size_t i = 0; i < state_.orderedOps.size(); i++) {
        auto& op = state_.orderedOps[i];
        if (state_.schedInfoMap[op].isRetired)
            continue;
        auto& reqMemids = state_.GetOpMemIds(op);
        if (std::find(reqMemids.begin(), reqMemids.end(), curMemId) == reqMemids.end())
            continue;

        if (op->GetAtomicScopeId() >= VF_CLUSTER_ID_START) {
            auto scopeIt = state_.vfScopeOpsByScope.find(op->GetAtomicScopeId());
            if (scopeIt != state_.vfScopeOpsByScope.end() && !scopeIt->second.empty()) {
                return state_.schedInfoMap[scopeIt->second.front()].execOrder;
            }
        }

        int opExecOrder = state_.schedInfoMap[op].execOrder;
        int minAlloc = INT_MAX;
        int allocCnt = 0;
        for (auto pre : state_.depManager.GetPredecessors(op)) {
            if (state_.schedInfoMap[pre].isRetired)
                continue;
            if (state_.schedInfoMap[pre].isAlloc) {
                int preOrder = state_.schedInfoMap[pre].execOrder;
                if (preOrder < minAlloc)
                    minAlloc = preOrder;
                allocCnt++;
            }
        }
        if (allocCnt == 0) {
            return opExecOrder;
        }
        return (opExecOrder - minAlloc <= allocCnt) ? minAlloc : opExecOrder;
    }
    return -1;
}

Status SpillEngine::UpdateCopyoutScheduleInfo(Operation* op, const SpillSource& source, int sourceMemId)
{
    LogicalTensorPtr spillTensor = source.tensor;
    state_.opReqMemIdsMap[op] = {sourceMemId};
    state_.schedInfoMap[op].isRetired = source.producedInPast;
    state_.schedInfoMap[op].isAlloc = false;
    state_.schedInfoMap[op].pipeType = RescheduleUtils::GetOpPipeType(op);
    // The original move may be removed by spill. Its new copyout must carry the
    // source alias chain into the final schedule, even when it is the only reader.
    auto& skipOps = state_.schedInfoMap[op].skipOps;
    for (auto* producer : spillTensor->GetProducers()) {
        auto path = SkipChainPath(producer);
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            if (std::find(skipOps.begin(), skipOps.end(), *it) == skipOps.end()) {
                skipOps.push_back(*it);
            }
        }
    }
    state_.depManager.RegisterOp(op);
    // 落位跟 buffer 的归属者 (alloc) 而非生产者: 跨核写同一块 buffer 是允许的, 拿生产者会把核定错。
    auto allocIt = state_.tensorAllocMap.find(spillTensor->memoryrange.memId);
    if (allocIt == state_.tensorAllocMap.end() || allocIt->second == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Spill: tensor[%d] has no alloc op to locate its copyout.",
                          spillTensor->memoryrange.memId);
        return FAILED;
    }
    Operation* refOp = allocIt->second;
    state_.schedInfoMap[op].coreLocation = state_.schedInfoMap[refOp].coreLocation;
    UpdateOpInternalSubgraphID(*op, refOp);
    state_.schedInfoMap[op].execOrder = ComputeCopyoutExecOrder(source, op) + 1;
    state_.InsertOrdered(op);
    return SUCCESS;
}

// copyout 排在数据齐之后: 取各生产者里最晚的那个。
// 同一块 buffer 上已有的 retired copyout 也要让开, 否则两个 copyout 抢同一格。
// 生产 anchor 属于 vf scope 时重定位到该 scope 的静态末 op 之后: SeqSchedule 顺序趟没有发射流可扫,
// 插在 scope 中间会影响 vf 连续流, 后续 UpdateIssueExecOrder 重编号会把错位固化。
int SpillEngine::ComputeCopyoutExecOrder(const SpillSource& source, Operation* copyoutOp)
{
    const std::vector<Operation*>& anchors = source.producerOps;
    int execOrder = state_.schedInfoMap[anchors.front()].execOrder;
    int vfScopeLastOrder = -1;
    for (auto* anchor : anchors) {
        execOrder = std::max(execOrder, state_.schedInfoMap[anchor].execOrder);
        int scopeId = anchor->GetAtomicScopeId();
        if (scopeId >= VF_CLUSTER_ID_START) {
            auto scopeIt = state_.vfScopeOpsByScope.find(scopeId);
            if (scopeIt != state_.vfScopeOpsByScope.end() && !scopeIt->second.empty()) {
                vfScopeLastOrder = std::max(vfScopeLastOrder, state_.schedInfoMap[scopeIt->second.back()].execOrder);
            }
        }
        for (auto* succOp : state_.depManager.GetSuccessors(anchor)) {
            if (!state_.schedInfoMap[succOp].isRetired || succOp == copyoutOp) {
                continue;
            }
            if (OpcodeManager::Inst().IsCopyOut(succOp->GetOpcode())) {
                execOrder = std::max(execOrder, state_.schedInfoMap[succOp].execOrder);
            }
        }
    }
    return std::max(execOrder, vfScopeLastOrder);
}

void SpillEngine::UpdateOpScheduleInfo(Operation* op, std::vector<int> memIds, Operation* AllocOp)
{
    state_.schedInfoMap[op].pipeType = RescheduleUtils::GetOpPipeType(op);
    state_.schedInfoMap[op].isAlloc = op->GetOpcodeStr().find("ALLOC") != std::string::npos;
    state_.schedInfoMap[op].isRetired = false;
    state_.opReqMemIdsMap[op] = memIds;
    state_.depManager.RegisterOp(op);
    state_.schedInfoMap[op].coreLocation = state_.schedInfoMap[AllocOp].coreLocation;
    UpdateOpInternalSubgraphID(*op, AllocOp);
    state_.numTotalIssues++;
}

Status SpillEngine::InsertOps(OpMemIdMap opMemidMap, Operation* spillAllocOp, int memId)
{
    if (memId == -1) {
        APASS_LOG_ERROR_F(Elements::Tensor, "MemId: %d illegal.", memId);
        return FAILED;
    }
    int bufNextUseTime = GetBufNextUseTime(memId);
    if (bufNextUseTime == -1) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Get Tensor[%d] next use time failed.", memId);
        return FAILED;
    }
    bufNextUseTime = bufNextUseTime <= state_.schedInfoMap[spillAllocOp].execOrder ?
                         state_.schedInfoMap[spillAllocOp].execOrder + 1 :
                         bufNextUseTime;
    for (auto& op : opMemidMap) {
        state_.schedInfoMap[op.first].execOrder = bufNextUseTime++;
        state_.InsertOrdered(op.first);
    }
    return SUCCESS;
}

Status SpillEngine::UpdateScheduleStatus(OpMemIdMap opMemidMap, int memId, Operation* spillAllocOp,
                                         LogicalTensorPtr localTensor, Operation* spillOp, const SpillPlan& plan)
{
    std::vector<Operation*> seeds;
    for (auto& [op, memid] : opMemidMap) {
        UpdateOpScheduleInfo(op, memid, spillAllocOp);
        seeds.push_back(op);
    }
    seeds.push_back(spillOp);
    seeds.push_back(spillAllocOp);
    seeds.insert(seeds.end(), plan.saveCopyoutOps.begin(), plan.saveCopyoutOps.end());

    if (InsertOps(opMemidMap, spillAllocOp, memId) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InsertOps failed.");
        return FAILED;
    }
    if (UpdateSpillOpDepend(spillOp, localTensor, memId, seeds) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateSpillOpDepend failed.");
        return FAILED;
    }
    if (UpdateRemainMemid(memId, localTensor->memoryrange.memId, seeds) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateRemainMemid failed.");
        return FAILED;
    }
    if (state_.depManager.RefreshDependencies(seeds, state_.tensorAllocMap) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "RefreshDependencies failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status SpillEngine::UpdateRemainMemid(int oldMemId, int newMemId, std::vector<Operation*>& remapped)
{
    if (state_.bufRefCount.find(oldMemId) == state_.bufRefCount.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "bufRefCount cannot find Tensor[%d]. ", oldMemId);
        return FAILED;
    }
    state_.bufRefCount[newMemId] = 0;
    state_.bufRefCount[oldMemId] = 0;
    for (auto& op : state_.orderedOps) {
        if (state_.schedInfoMap[op].isRetired) {
            continue;
        }
        bool reqChanged = RemapOpReqMemId(op, oldMemId, newMemId);
        bool tensorChanged = ReplaceTensorMemId(op, oldMemId, newMemId);
        if (reqChanged || tensorChanged) {
            remapped.push_back(op);
        }
    }
    return SUCCESS;
}

} // namespace npu::tile_fwk
