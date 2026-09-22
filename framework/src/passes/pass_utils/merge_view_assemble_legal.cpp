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
 * \file merge_view_assemble_utils.cpp
 * \brief utils of view and assemble operation merging
 */

#include "merge_view_assemble_legal.h"
#include <optional>
#include "interface/tensor/irbuilder.h"
#include "interface/operation/attribute.h"
#include "passes/pass_utils/dead_operation_eliminate.h"
#include "passes/pass_utils/infer_shape_utils.h"
#include "passes/pass_utils/pass_attr_defs.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/pass_log/pass_log.h"
#include "tilefwk/tilefwk_op.h"

#define MODULE_NAME "LegalMergeAssembleUtils"

namespace npu::tile_fwk {
namespace {
struct RmwModeAttrState {
    bool conflict = false;
    std::optional<AtomicRMWMode> mode;
};

struct AtomicSemanticAttrState {
    bool fromReduceAcc = false;
    bool fromExplicitRmw = false;
};

RmwModeAttrState MergeRmwModeAttr(const RmwModeAttrState& current, const RmwModeAttrState& next)
{
    if (current.conflict || next.conflict) {
        return {true, std::nullopt};
    }
    if (!current.mode.has_value()) {
        return next;
    }
    if (!next.mode.has_value() || current.mode == next.mode) {
        return current;
    }
    return {true, std::nullopt};
}

RmwModeAttrState GetRmwModeAttr(const Operation& op)
{
    RmwModeAttrState rmwModeAttr;
    if (op.HasAttr(RMW_MODE_ATTR_ADD)) {
        rmwModeAttr = MergeRmwModeAttr(rmwModeAttr, {false, AtomicRMWMode::ADD});
    }
    if (op.HasAttr(RMW_MODE_ATTR_MIN)) {
        rmwModeAttr = MergeRmwModeAttr(rmwModeAttr, {false, AtomicRMWMode::MIN});
    }
    if (op.HasAttr(RMW_MODE_ATTR_MAX)) {
        rmwModeAttr = MergeRmwModeAttr(rmwModeAttr, {false, AtomicRMWMode::MAX});
    }
    return rmwModeAttr;
}

RmwModeAttrState GetChainRmwModeAttr(const std::vector<Operation*>& chain)
{
    RmwModeAttrState chainRmwModeAttr;
    for (const auto* op : chain) {
        if (op == nullptr) {
            return {true, std::nullopt};
        }
        chainRmwModeAttr = MergeRmwModeAttr(chainRmwModeAttr, GetRmwModeAttr(*op));
        if (chainRmwModeAttr.conflict) {
            return chainRmwModeAttr;
        }
    }
    return chainRmwModeAttr;
}

bool IsRmwModeAttrCompatible(const std::vector<Operation*>& chain, const Operation& consumer)
{
    auto chainRmwModeAttr = GetChainRmwModeAttr(chain);
    auto consumerRmwModeAttr = GetRmwModeAttr(consumer);
    return !MergeRmwModeAttr(chainRmwModeAttr, consumerRmwModeAttr).conflict;
}

std::string GetRmwModeAttrKey(const RmwModeAttrState& rmwModeAttr)
{
    if (!rmwModeAttr.mode.has_value() || rmwModeAttr.conflict) {
        return "";
    }
    switch (*rmwModeAttr.mode) {
        case AtomicRMWMode::ADD:
            return RMW_MODE_ATTR_ADD;
        case AtomicRMWMode::MIN:
            return RMW_MODE_ATTR_MIN;
        case AtomicRMWMode::MAX:
            return RMW_MODE_ATTR_MAX;
        default:
            return "";
    }
}

bool IsViewLikeOpcode(Opcode opcode) { return opcode == Opcode::OP_VIEW || opcode == Opcode::OP_SLICE; }

bool IsAssembleLikeOpcode(Opcode opcode) { return opcode == Opcode::OP_ASSEMBLE || opcode == Opcode::OP_CONTRACT; }

bool ChainHasOpcode(const std::vector<Operation*>& chain, Opcode opcode)
{
    for (const auto* op : chain) {
        if (op != nullptr && op->GetOpcode() == opcode) {
            return true;
        }
    }
    return false;
}

bool CanMergeViewLikeChain(const std::vector<Operation*>& chain, Opcode nextOpcode)
{
    return !(ChainHasOpcode(chain, Opcode::OP_SLICE) && nextOpcode == Opcode::OP_SLICE);
}

bool CanMergeAssembleLikeChain(const std::vector<Operation*>& chain, Opcode nextOpcode)
{
    return !(ChainHasOpcode(chain, Opcode::OP_CONTRACT) && nextOpcode == Opcode::OP_CONTRACT);
}

Opcode GetMergedViewOpcode(const std::vector<Operation*>& chain)
{
    return ChainHasOpcode(chain, Opcode::OP_SLICE) ? Opcode::OP_SLICE : Opcode::OP_VIEW;
}

Opcode GetMergedAssembleOpcode(const std::vector<Operation*>& chain)
{
    return ChainHasOpcode(chain, Opcode::OP_CONTRACT) ? Opcode::OP_CONTRACT : Opcode::OP_ASSEMBLE;
}

AtomicSemanticAttrState GetChainAtomicSemanticAttr(const std::vector<Operation*>& chain)
{
    AtomicSemanticAttrState attr;
    for (const auto* op : chain) {
        if (op == nullptr) {
            continue;
        }
        attr.fromReduceAcc = attr.fromReduceAcc || op->HasAttr(ATOMIC_FROM_REDUCE_ACC_ATTR);
        attr.fromExplicitRmw = attr.fromExplicitRmw || op->HasAttr(ATOMIC_FROM_EXPLICIT_RMW_ATTR);
    }
    return attr;
}
} // namespace

Status LegalMergeAssembleUtils::MergeViewAssemble(Function& function)
{
    LegalMergeAssembleUtils LegalMergeAssembleUtils;
    Status status = LegalMergeAssembleUtils.Process(function);
    return status;
}

Status LegalMergeAssembleUtils::Process(Function& function)
{
    Status status = Initialize();
    if (status != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "LegalMergeAssembleUtils initialization failed.");
        return status;
    }
    status = ProcessOperations(function);
    if (status != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Processing operations failed.");
        return status;
    }
    status = CleanUp(function);
    if (status != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Cleanup phase failed.");
        return status;
    }
    return SUCCESS;
}

Status LegalMergeAssembleUtils::Initialize()
{
    visitedOp_.clear();
    viewOpToAppend_.clear();
    assembleOpToAppend_.clear();
    consumerCache_.clear();
    tensorConsumerCache_.clear();
    candidateOps_.clear();
    return SUCCESS;
}

const LegalMergeAssembleUtils::ConsumerCacheEntry& LegalMergeAssembleUtils::BuildTensorConsumerCache(
    Function& function, const LogicalTensorPtr& tensor)
{
    static const ConsumerCacheEntry emptyEntry;
    if (tensor == nullptr) {
        return emptyEntry;
    }
    auto tensorMagic = tensor->GetMagic();
    auto cached = tensorConsumerCache_.find(tensorMagic);
    if (cached != tensorConsumerCache_.end()) {
        return cached->second;
    }

    auto iter = tensorConsumerCache_.emplace(tensorMagic, ConsumerCacheEntry{}).first;
    auto& cacheEntry = iter->second;
    for (auto* consumer : tensor->GetConsumers()) {
        if (consumer == nullptr || consumer->BelongTo() != &function || consumer->IsDeleted()) {
            continue;
        }
        if (IsViewLikeOpcode(consumer->GetOpcode())) {
            cacheEntry.viewConsumers.emplace_back(consumer);
            cacheEntry.hasAssembleChainStopper = true;
        } else if (IsAssembleLikeOpcode(consumer->GetOpcode())) {
            cacheEntry.assembleConsumers.emplace_back(consumer);
        } else {
            cacheEntry.hasAssembleChainStopper = true;
        }
    }
    return cacheEntry;
}

Status LegalMergeAssembleUtils::BuildConsumerCache(Function& function)
{
    auto operations = function.Operations(false);
    consumerCache_.reserve(operations.size());
    tensorConsumerCache_.reserve(operations.size());
    candidateOps_.reserve(operations.size());
    for (auto& operation : operations) {
        if (!IsViewLikeOpcode(operation.GetOpcode()) && !IsAssembleLikeOpcode(operation.GetOpcode())) {
            continue;
        }
        candidateOps_.emplace_back(&operation);
        if (operation.oOperand.empty()) {
            continue;
        }
        consumerCache_[operation.GetOpMagic()] = &BuildTensorConsumerCache(function, operation.oOperand.front());
    }
    return SUCCESS;
}

const LegalMergeAssembleUtils::ConsumerCacheEntry& LegalMergeAssembleUtils::GetConsumers(
    const Operation& operation) const
{
    static const ConsumerCacheEntry emptyEntry;
    auto iter = consumerCache_.find(operation.GetOpMagic());
    if (iter == consumerCache_.end() || iter->second == nullptr) {
        return emptyEntry;
    }
    return *iter->second;
}

Status LegalMergeAssembleUtils::ProcessOperations(Function& function)
{
    function.SortOperations(SortOperationsMode::LIGHTWEIGHT);
    Status status = BuildConsumerCache(function);
    if (status != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "BuildConsumerCache failed.");
        return status;
    }
    for (auto* op : candidateOps_) {
        if (op == nullptr || op->IsDeleted()) {
            continue;
        }
        if (visitedOp_.count(op->GetOpMagic()) != 0) {
            continue;
        }
        Status processStatus = SUCCESS;
        std::vector<Operation*> chain;
        if (IsViewLikeOpcode(op->GetOpcode())) {
            processStatus = MergeViewChain(function, *op, chain);
        } else if (IsAssembleLikeOpcode(op->GetOpcode())) {
            processStatus = MergeAssembleChain(function, *op, chain);
        }
        if (processStatus != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "ProcessOperations failed.");
            return processStatus;
        }
    }
    status = AppendMergedViewOperations(function);
    if (status != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "AppendMergedViewOperations phase failed.");
        return status;
    }
    status = AppendMergedAssembleOperations(function);
    if (status != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "AppendMergedAssembleOperations phase failed.");
        return FAILED;
    }
    return status;
}

Status LegalMergeAssembleUtils::AppendMergedViewOperations(Function& function)
{
    /* Process View ops first to avoid View output being cleared in View-Assemble scenarios */
    for (auto& viewOp : viewOpToAppend_) {
        auto attr = std::make_shared<ViewOpAttribute>(viewOp.offset, viewOp.toType, viewOp.dynOffset,
                                                      viewOp.dynValidShape);
        if (!attr) {
            APASS_LOG_ERROR_F(Elements::Function, "Failed to create ViewOpAttribute.");
            return FAILED;
        }
        auto& mergedViewOp = irBuilder_.CreateTensorOpStmt(function, viewOp.opcode, {viewOp.input}, {viewOp.output},
                                                           viewOp.span);
        mergedViewOp.SetScopeInfo(viewOp.scopeInfo);
        mergedViewOp.SetOpAttribute(attr);
        // 继承op_attr_copy_in_mode属性
        if (viewOp.hasCopyInMode) {
            mergedViewOp.SetAttr("op_attr_copy_in_mode", viewOp.copyInModeValue);
        }
        // 继承op_attr_copy_in_l1_padding_mode属性
        if (viewOp.hasL1PaddingMode) {
            mergedViewOp.SetAttr("op_attr_copy_in_l1_padding_mode", viewOp.l1PaddingMode);
        }
        // 继承op_attr_copy_in_l1_k_index属性
        if (viewOp.hasKIndex) {
            mergedViewOp.SetAttr("op_attr_copy_in_l1_k_index", viewOp.kIndex);
        }
        // 继承op_attr_is_gemv属性
        if (viewOp.hasIsGemv) {
            mergedViewOp.SetAttr(OpAttributeKey::isGemv, viewOp.isGemvValue);
        }
        viewOp.output->UpdateDynValidShape(viewOp.dynValidShape);
    }
    return SUCCESS;
}

Status LegalMergeAssembleUtils::AppendMergedAssembleOperations(Function& function)
{
    for (const auto& assembleOp : assembleOpToAppend_) {
        auto attr = std::make_shared<AssembleOpAttribute>(assembleOp.offset, assembleOp.dynOffset);
        if (!attr) {
            return FAILED;
        }
        auto& mergedAssembleOp = irBuilder_.CreateTensorOpStmt(function, assembleOp.opcode, {assembleOp.input},
                                                               {assembleOp.output}, assembleOp.span);
        mergedAssembleOp.SetScopeInfo(assembleOp.scopeInfo);
        mergedAssembleOp.SetOpAttribute(attr);
        if (!assembleOp.rmwModeAttr.empty()) {
            mergedAssembleOp.SetAttribute(assembleOp.rmwModeAttr, 1L);
        }
        if (assembleOp.atomicFromReduceAcc) {
            mergedAssembleOp.SetAttribute(ATOMIC_FROM_REDUCE_ACC_ATTR, true);
        }
        if (assembleOp.atomicFromExplicitRmw) {
            mergedAssembleOp.SetAttribute(ATOMIC_FROM_EXPLICIT_RMW_ATTR, true);
        }
    }
    return SUCCESS;
}

Status LegalMergeAssembleUtils::CleanUp(Function& function)
{
    Status status = EraseRedundantAssemble(function);
    if (status != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "EraseRedundantAssemble failed.");
        return status;
    }
    DeadOperationEliminator eliminator;
    eliminator.EliminateOperation(function, false, false);
    function.SortOperations(SortOperationsMode::LIGHTWEIGHT);
    return SUCCESS;
}

ir::Span LegalMergeAssembleUtils::GetFirstSpan(const std::vector<Operation*>& chain)
{
    ir::Span firstSpan;
    for (auto* op : chain) {
        auto loc = op->GetSpan();
        if (!loc.IsUnknown()) {
            firstSpan = loc;
            break;
        }
    }
    return firstSpan;
}

Operation::ScopeInfo LegalMergeAssembleUtils::GetChainScopeInfo(const std::vector<Operation*>& chain)
{
    for (auto* op : chain) {
        if (op->GetScopeId() != -1) {
            return op->GetScopeInfo();
        }
    }
    return Operation::ScopeInfo();
}

Status LegalMergeAssembleUtils::MergeViewChain(Function& function, Operation& operation, std::vector<Operation*>& chain,
                                               int effectiveScopeId)
{
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
    // 1. 初始化操作链
    InitOperationChain(operation, chain);

    int newScopeId = operation.GetScopeId();
    if (effectiveScopeId == -1 && newScopeId != -1) {
        effectiveScopeId = newScopeId;
    }

    // 2. 处理消费者链
    const auto& consumers = GetConsumers(operation);
    bool chainEnd = true;
    Status status = ProcessConsumerChain(function, consumers, chain, chainEnd, effectiveScopeId);
    if (status != SUCCESS) {
        return status;
    }

    // 3. 处理链尾情况
    if (chainEnd && chain.size() > 1) {
        return ProcessChainEnd(function, chain);
    }

    return SUCCESS;
}

void LegalMergeAssembleUtils::InitOperationChain(Operation& operation, std::vector<Operation*>& chain)
{
    visitedOp_.insert(operation.opmagic);
    chain.emplace_back(&operation);
}

Status LegalMergeAssembleUtils::ProcessConsumerChain(Function& function, const ConsumerCacheEntry& consumers,
                                                     std::vector<Operation*>& chain, bool& chainEnd,
                                                     int effectiveScopeId)
{
    if (consumers.viewConsumers.empty()) {
        return SUCCESS;
    }
    Operation* currentOp = chain.back();
    auto currentViewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(currentOp->GetOpAttribute());
    if (!currentViewAttr) {
        APASS_LOG_ERROR_F(Elements::Function, "Failed to get current view attribute.");
        return FAILED;
    }
    MemoryType currentMemType = currentViewAttr->GetTo();
    for (auto& op : consumers.viewConsumers) {
        if (!op) {
            return FAILED;
        }
        if (IsViewLikeOpcode(op->GetOpcode())) {
            auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(op->GetOpAttribute());
            if (viewOpAttribute == nullptr) {
                APASS_LOG_ERROR_F(Elements::Function, "View operation has null viewOpAttribute.");
                return FAILED;
            }
            auto memoryTo = viewOpAttribute->GetTo();
            // 根据新的合并原则判断是否可以合并
            bool canMerge = false;
            if (currentMemType == MemoryType::MEM_UNKNOWN || currentMemType == memoryTo) {
                // 1.unknown memType 可以向它之后的view合并 2.相同memType的view可以合并
                canMerge = true;
            }
            canMerge = canMerge && CanMergeViewLikeChain(chain, op->GetOpcode());
            if (canMerge) {
                int consumerScopeId = op->GetScopeId();
                if (effectiveScopeId != -1 && consumerScopeId != -1 && effectiveScopeId != consumerScopeId) {
                    chainEnd = true;
                    continue;
                }
                chainEnd = false;
                Status status = MergeViewChain(function, *op, chain, effectiveScopeId);
                if (status != SUCCESS) {
                    return status;
                }
                chain.pop_back();
            } else {
                chainEnd = true;
            }
        }
    }
    return SUCCESS;
}

Status LegalMergeAssembleUtils::ProcessChainEnd(Function& function, std::vector<Operation*>& chain)
{
    // 1. 验证链的有效性
    Operation* startOp = chain.front();
    Operation* endOp = chain.back();
    if (startOp->iOperand.empty()) {
        APASS_LOG_ERROR_F(Elements::Function, "First operation in chain has no input operands.");
        return FAILED;
    }
    if (endOp->oOperand.empty()) {
        APASS_LOG_ERROR_F(Elements::Function, "Last operation in chain has no output operands.");
        return FAILED;
    }
    auto& startTensor = startOp->iOperand.front();
    auto& endTensor = endOp->oOperand.front();
    if (!startTensor) {
        APASS_LOG_ERROR_F(Elements::Function, "Null input tensor found for first operation in chain.");
        return FAILED;
    }
    if (!endTensor) {
        APASS_LOG_ERROR_F(Elements::Function, "Null output tensor found for last operation in chain.");
        return FAILED;
    }
    std::vector<int64_t> newOffset;
    std::vector<SymbolicScalar> newDynOffset;
    std::vector<SymbolicScalar> newDynValidShape;
    Status status = CalculateMergedOffsets(chain, newOffset, newDynOffset, newDynValidShape);
    if (status != SUCCESS) {
        return status;
    }
    // 获取链路上第一个非空的span
    ir::Span firstSpan = GetFirstSpan(chain);
    Operation::ScopeInfo chainScopeInfo = GetChainScopeInfo(chain);
    // 记录合并操作
    RecordMergedViewOperation(endOp, startTensor, endTensor, newOffset, newDynOffset, newDynValidShape, firstSpan,
                              chainScopeInfo, GetMergedViewOpcode(chain));

    // 清理链尾
    endOp->oOperand.clear();
    function.GetTensorMap().Erase(endTensor);
    return SUCCESS;
}

Status LegalMergeAssembleUtils::CalculateMergedOffsets(const std::vector<Operation*>& chain,
                                                       std::vector<int64_t>& newOffset,
                                                       std::vector<SymbolicScalar>& newDynOffset,
                                                       std::vector<SymbolicScalar>& newDynValidShape)
{
    for (size_t i = 0; i < chain.size(); ++i) {
        const auto& view = chain[i];
        if (!view) {
            APASS_LOG_ERROR_F(Elements::Function, "Null view operation in chain.");
            return FAILED;
        }
        auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(view->GetOpAttribute());
        if (!viewOpAttribute) {
            APASS_LOG_ERROR_F(Elements::Function, "Failed to get ViewOpAttribute.");
            return FAILED;
        }
        if (i == 0) {
            newOffset = viewOpAttribute->GetFromOffset();
            newDynOffset = viewOpAttribute->GetFromDynOffset();
            if (!viewOpAttribute->GetToDynValidShape().empty()) {
                newDynValidShape = viewOpAttribute->GetToDynValidShape();
            }
            continue;
        }
        auto ret = TensorOffset::Add(newOffset, newDynOffset, viewOpAttribute->GetFromOffset(),
                                     viewOpAttribute->GetFromDynOffset());
        if (!ret.first.empty()) {
            newOffset = ret.first;
            newDynOffset = ret.second;
        }
        if (!viewOpAttribute->GetToDynValidShape().empty()) {
            newDynValidShape = viewOpAttribute->GetToDynValidShape();
            continue;
        }
        newDynValidShape = GetViewValidShape(newDynValidShape, viewOpAttribute->GetFromOffset(),
                                             viewOpAttribute->GetFromDynOffset(), view->GetOOperands()[0]->GetShape());
    }
    return SUCCESS;
}

void LegalMergeAssembleUtils::RecordMergedViewOperation(
    Operation* lastViewOp, const std::shared_ptr<LogicalTensor>& startTensor,
    const std::shared_ptr<LogicalTensor>& endTensor, const std::vector<int64_t>& newOffset,
    const std::vector<SymbolicScalar>& newDynOffset, const std::vector<SymbolicScalar>& newDynValidShape,
    const ir::Span& span, const Operation::ScopeInfo& scopeInfo, Opcode opcode)
{
    // 获取最后一个VIEW的属性
    auto lastViewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(lastViewOp->GetOpAttribute());
    if (!lastViewAttr) {
        return;
    }
    // 获取特定的 op_attr_copy_in_mode 属性
    int64_t copyInModeValue = 0;
    bool hasCopyInMode = lastViewOp->GetAttr<int64_t>("op_attr_copy_in_mode", copyInModeValue);
    // 获取特定的 op_attr_copy_in_l1_padding_mode 属性
    int64_t l1PaddingMode = 0;
    bool hasL1PaddingMode = lastViewOp->GetAttr<int64_t>("op_attr_copy_in_l1_padding_mode", l1PaddingMode);
    // 获取特定的 op_attr_copy_in_l1_k_index 属性
    int64_t kIndex = 0;
    bool hasKIndex = lastViewOp->GetAttr<int64_t>("op_attr_copy_in_l1_k_index", kIndex);
    // 获取特定的 op_attr_is_gemv 属性
    int64_t isGemv = 0;
    bool hasIsGemv = lastViewOp->GetAttr<int64_t>(OpAttributeKey::isGemv, isGemv);
    // 清理消费者关系
    endTensor->GetProducers().clear();
    // 记录合并op
    viewOpToAppend_.emplace_back(ViewOp{startTensor, endTensor, newOffset, newDynOffset, newDynValidShape,
                                        lastViewAttr->GetTo(), hasCopyInMode, std::move(copyInModeValue),
                                        hasL1PaddingMode, std::move(l1PaddingMode), hasKIndex, kIndex, hasIsGemv,
                                        std::move(isGemv), span, scopeInfo, opcode});
}

Status LegalMergeAssembleUtils::MergeAssembleChain(Function& function, Operation& operation,
                                                   std::vector<Operation*>& chain, int effectiveScopeId)
{
    // 1. 初始化操作链
    InitAssembleChain(operation, chain);

    int newScopeId = operation.GetScopeId();
    if (effectiveScopeId == -1 && newScopeId != -1) {
        effectiveScopeId = newScopeId;
    }

    // 2. 处理消费者
    const auto& consumers = GetConsumers(operation);
    bool chainEnd = consumers.assembleConsumers.empty() || consumers.hasAssembleChainStopper;
    Status status = ProcessAssembleConsumers(function, consumers, chain, chainEnd, effectiveScopeId);
    if (status != SUCCESS) {
        return status;
    }

    // 3. 处理链尾情况
    if (chainEnd && chain.size() > 1) {
        status = ProcessAssembleChainEnd(function, chain, operation);
        if (status != SUCCESS) {
            return status;
        }
    }

    chain.pop_back();
    return SUCCESS;
}

void LegalMergeAssembleUtils::InitAssembleChain(Operation& operation, std::vector<Operation*>& chain)
{
    visitedOp_.insert(operation.opmagic);
    chain.emplace_back(&operation);
}

Status LegalMergeAssembleUtils::ProcessAssembleConsumers(Function& function, const ConsumerCacheEntry& consumers,
                                                         std::vector<Operation*>& chain, bool& chainEnd,
                                                         int effectiveScopeId)
{
    if (consumers.assembleConsumers.empty()) {
        return SUCCESS;
    }
    if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510 && consumers.hasAssembleChainStopper) {
        return SUCCESS;
    }
    for (auto& op : consumers.assembleConsumers) {
        if (!op) {
            APASS_LOG_ERROR_F(Elements::Function, "Null consumer operation found.");
            return FAILED;
        }
        if (IsAssembleLikeOpcode(op->GetOpcode())) {
            int consumerScopeId = op->GetScopeId();
            if (effectiveScopeId != -1 && consumerScopeId != -1 && effectiveScopeId != consumerScopeId) {
                chainEnd = true;
                continue;
            }
            if (!CanMergeAssembleLikeChain(chain, op->GetOpcode())) {
                chainEnd = true;
                continue;
            }
            if (!IsRmwModeAttrCompatible(chain, *op)) {
                chainEnd = true;
                continue;
            }
            Status status = MergeAssembleChain(function, *op, chain, effectiveScopeId);
            if (status != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Function, "Run MergeAssembleChain failed.");
                return status;
            }
            continue;
        }
        chainEnd = true;
    }
    return SUCCESS;
}

Status LegalMergeAssembleUtils::ProcessAssembleChainEnd(Function& function, std::vector<Operation*>& chain,
                                                        Operation& operation)
{
    // 验证链有效性
    if (chain.front()->iOperand.empty() || chain.back()->oOperand.empty()) {
        APASS_LOG_ERROR_F(Elements::Function, "Invalid chain operations.");
        return FAILED;
    }
    auto& startTensor = chain.front()->iOperand.front();
    auto& endTensor = chain.back()->oOperand.front();
    if (!startTensor || !endTensor) {
        APASS_LOG_ERROR_F(Elements::Function, "Null tensor found in chain.");
        return FAILED;
    }
    // 计算合并offset
    auto [newOffset, newDynOffset] = CalculateAssembleOffsets(chain, startTensor->offset.size());
    // 获取链路上第一个非空的span
    ir::Span firstSpan = GetFirstSpan(chain);
    Operation::ScopeInfo chainScopeInfo = GetChainScopeInfo(chain);
    RmwModeAttrState rmwModeAttr = GetChainRmwModeAttr(chain);
    if (rmwModeAttr.conflict) {
        APASS_LOG_ERROR_F(Elements::Function, "Assemble chain has conflicting rmw mode attributes.");
        return FAILED;
    }
    AtomicSemanticAttrState atomicSemanticAttr = GetChainAtomicSemanticAttr(chain);
    // 4. 记录并清理
    RecordAssembleOperation(startTensor, endTensor, newOffset, newDynOffset, firstSpan, chainScopeInfo,
                            GetRmwModeAttrKey(rmwModeAttr), GetMergedAssembleOpcode(chain),
                            atomicSemanticAttr.fromReduceAcc, atomicSemanticAttr.fromExplicitRmw);
    function.GetTensorMap().Erase(endTensor);
    operation.SetAsDeleted();

    return SUCCESS;
}

std::pair<std::vector<int64_t>, std::vector<SymbolicScalar>> LegalMergeAssembleUtils::CalculateAssembleOffsets(
    const std::vector<Operation*>& chain, size_t offsetSize)
{
    std::vector<int64_t> newOffset(offsetSize, 0);
    std::vector<SymbolicScalar> newDynOffset;
    for (size_t i = 0; i < chain.size(); ++i) {
        const auto& assemble = chain[i];
        if (!assemble) {
            return {};
        }
        auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(assemble->GetOpAttribute());
        if (!assembleOpAttribute) {
            return {};
        }
        if (i == 0) {
            newOffset = assembleOpAttribute->GetToOffset();
            newDynOffset = assembleOpAttribute->GetToDynOffset();
            continue;
        }
        auto ret = TensorOffset::Add(newOffset, newDynOffset, assembleOpAttribute->GetToOffset(),
                                     assembleOpAttribute->GetToDynOffset());
        if (!ret.first.empty()) {
            newOffset = ret.first;
            newDynOffset = ret.second;
        }
    }
    return {newOffset, newDynOffset};
}

void LegalMergeAssembleUtils::RecordAssembleOperation(const std::shared_ptr<LogicalTensor>& input,
                                                      const std::shared_ptr<LogicalTensor>& output,
                                                      const std::vector<int64_t>& offset,
                                                      const std::vector<SymbolicScalar>& dynOffset,
                                                      const ir::Span& span, const Operation::ScopeInfo& scopeInfo,
                                                      const std::string& rmwModeAttr, Opcode opcode,
                                                      bool atomicFromReduceAcc, bool atomicFromExplicitRmw)
{
    assembleOpToAppend_.emplace_back(AssembleOp{input, output, offset, dynOffset, span, scopeInfo, rmwModeAttr, opcode,
                                                atomicFromReduceAcc, atomicFromExplicitRmw});
}

Status LegalMergeAssembleUtils::EraseRedundantAssemble(Function& function) const
{
    std::unordered_set<Operation*> redundantAssembles;
    for (auto& op : function.Operations(false)) {
        if (op.GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        if (op.iOperand.empty()) {
            APASS_LOG_ERROR_F(Elements::Function, "Assemble operation with no input operands.");
            return FAILED;
        }
        if (op.iOperand.front()->GetProducers().empty()) {
            redundantAssembles.emplace(&op);
        }
    }
    for (auto& ele : redundantAssembles) {
        if (!ele) {
            continue;
        }
        ele->SetAsDeleted();
    }
    function.EraseOperations(true, false);
    return SUCCESS;
}
} // namespace npu::tile_fwk
