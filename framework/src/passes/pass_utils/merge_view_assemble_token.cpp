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

#include "merge_view_assemble_token.h"
#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <queue>
#include <unordered_set>
#include "interface/tensor/irbuilder.h"
#include "interface/operation/attribute.h"
#include "passes/pass_utils/dead_operation_eliminate.h"
#include "passes/pass_utils/infer_shape_utils.h"
#include "passes/pass_utils/pass_attr_defs.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/pass_log/pass_log.h"
#include "tilefwk/tilefwk_op.h"

#define MODULE_NAME "TokenMergeAssembleUtils"

namespace npu::tile_fwk {
namespace {
ir::StmtPtr ToStmtPtr(Operation& op) { return std::static_pointer_cast<const ir::Stmt>(op.shared_from_this()); }

Operation* ToOperation(const ir::StmtPtr& stmt) { return static_cast<Operation*>(const_cast<ir::Stmt*>(stmt.get())); }

template <typename T>
void AddUnique(std::vector<T>& values, const T& value)
{
    if (value == nullptr || std::find(values.begin(), values.end(), value) != values.end()) {
        return;
    }
    values.emplace_back(value);
}

void RemoveToken(Operation& op, const ir::VarPtr& token)
{
    op.tokens_.erase(std::remove(op.tokens_.begin(), op.tokens_.end(), token), op.tokens_.end());
}

void AddTokenConsumer(Function& function, const ir::VarPtr& token, Operation& consumer)
{
    if (token == nullptr) {
        return;
    }
    function.GetVarDependency().AddConsumer(token, ToStmtPtr(consumer));
    AddUnique(consumer.tokens_, token);
}

struct TokenSnapshot {
    ir::VarPtr token;
    Operation* producer = nullptr;
    std::vector<Operation*> consumers;
};

// Collects every token touching the affected ops: from the ops' own result/input token
// lists plus the var dependency registry entries whose producer or consumer is affected.
std::vector<ir::VarPtr> CollectAffectedTokens(Function& function, const std::unordered_set<Operation*>& affected)
{
    std::vector<ir::VarPtr> tokens;
    for (auto* op : affected) {
        if (op == nullptr) {
            continue;
        }
        for (const auto& token : op->result_token_) {
            AddUnique(tokens, token);
        }
        for (const auto& token : op->tokens_) {
            AddUnique(tokens, token);
        }
    }
    for (const auto& [token, entry] : function.GetVarDependency().GetAllDependencies()) {
        bool touchesAffected = false;
        for (const auto& producer : entry.producers) {
            if (affected.count(ToOperation(producer)) != 0) {
                touchesAffected = true;
                break;
            }
        }
        if (!touchesAffected) {
            for (const auto& consumer : entry.consumers) {
                if (affected.count(ToOperation(consumer)) != 0) {
                    touchesAffected = true;
                    break;
                }
            }
        }
        if (touchesAffected) {
            AddUnique(tokens, token);
        }
    }
    return tokens;
}

// Builds the producer/consumers snapshot of one token relative to the affected ops.
TokenSnapshot BuildTokenSnapshot(Function& function, const ir::VarPtr& token,
                                 const std::unordered_set<Operation*>& affected)
{
    TokenSnapshot snapshot;
    snapshot.token = token;
    const auto& producers = function.GetVarDependency().GetProducers(token);
    if (!producers.empty()) {
        snapshot.producer = ToOperation(*producers.begin());
    } else {
        for (auto* op : affected) {
            if (op != nullptr &&
                std::find(op->result_token_.begin(), op->result_token_.end(), token) != op->result_token_.end()) {
                snapshot.producer = op;
                break;
            }
        }
    }
    for (const auto& consumer : function.GetVarDependency().GetConsumers(token)) {
        AddUnique(snapshot.consumers, ToOperation(consumer));
    }
    for (auto* op : affected) {
        if (op != nullptr && std::find(op->tokens_.begin(), op->tokens_.end(), token) != op->tokens_.end()) {
            AddUnique(snapshot.consumers, op);
        }
    }
    return snapshot;
}

std::vector<TokenSnapshot> CollectTokenSnapshots(Function& function, const std::unordered_set<Operation*>& affected)
{
    std::vector<TokenSnapshot> snapshots;
    for (const auto& token : CollectAffectedTokens(function, affected)) {
        snapshots.emplace_back(BuildTokenSnapshot(function, token, affected));
    }
    return snapshots;
}

bool HasTokenDependency(const std::vector<Operation*>& operations)
{
    return std::any_of(operations.begin(), operations.end(), [](const Operation* op) {
        return op != nullptr && (!op->result_token_.empty() || !op->tokens_.empty());
    });
}

bool IsFullTensorAssemble(const Operation& operation)
{
    return operation.GetOpcode() == Opcode::OP_ASSEMBLE && operation.GetIOperands().size() == 1 &&
           operation.GetOOperands().size() == 1 && operation.GetIOperands().front() != nullptr &&
           operation.GetOOperands().front() != nullptr &&
           operation.GetIOperands().front()->GetShape() == operation.GetOOperands().front()->GetShape();
}

void AddDependencyEdge(std::unordered_map<Operation*, std::unordered_set<Operation*>>& adjacency, Operation* producer,
                       Operation* consumer)
{
    if (producer == nullptr || consumer == nullptr || producer == consumer) {
        return;
    }
    adjacency[producer].insert(consumer);
    adjacency.try_emplace(consumer);
}

bool HasDependencyCycle(const std::unordered_map<Operation*, std::unordered_set<Operation*>>& adjacency)
{
    std::unordered_map<Operation*, size_t> indegree;
    indegree.reserve(adjacency.size());
    for (const auto& [op, consumers] : adjacency) {
        indegree.try_emplace(op, 0);
        for (auto* consumer : consumers) {
            ++indegree[consumer];
        }
    }
    std::queue<Operation*> ready;
    for (const auto& [op, degree] : indegree) {
        if (degree == 0) {
            ready.push(op);
        }
    }
    size_t visited = 0;
    while (!ready.empty()) {
        auto* op = ready.front();
        ready.pop();
        ++visited;
        auto iter = adjacency.find(op);
        if (iter == adjacency.end()) {
            continue;
        }
        for (auto* consumer : iter->second) {
            if (--indegree[consumer] == 0) {
                ready.push(consumer);
            }
        }
    }
    return visited != indegree.size();
}

bool WouldCreateCycleAfterContraction(Function& function, const std::unordered_map<Operation*, Operation*>& mapping,
                                      bool dagVerified)
{
    // See VerifyFunctionIsDag: an acyclic graph stays acyclic under edge-removing merges.
    if (dagVerified) {
        return false;
    }
    auto mapOp = [&mapping](Operation* op) {
        auto iter = mapping.find(op);
        return iter == mapping.end() ? op : iter->second;
    };
    std::unordered_map<Operation*, std::unordered_set<Operation*>> adjacency;
    for (auto& op : function.Operations(false)) {
        adjacency.try_emplace(mapOp(&op));
        for (const auto& input : op.GetIOperands()) {
            for (auto* producer : input->GetProducers()) {
                if (producer->BelongTo() == &function && !producer->IsDeleted()) {
                    AddDependencyEdge(adjacency, mapOp(producer), mapOp(&op));
                }
            }
        }
    }
    for (const auto& [token, entry] : function.GetVarDependency().GetAllDependencies()) {
        (void)token;
        for (const auto& producerStmt : entry.producers) {
            for (const auto& consumerStmt : entry.consumers) {
                AddDependencyEdge(adjacency, mapOp(ToOperation(producerStmt)), mapOp(ToOperation(consumerStmt)));
            }
        }
    }
    return HasDependencyCycle(adjacency);
}

// Cycle check for assemble chain merges. The merged op reads the chain head's input and
// writes the chain tail's output, bypassing the intermediates: the tail's input edges
// (reading the middle) disappear, so they must not be attributed to the representative.
// Keeping them would fabricate cycles through sibling producers of the middle.
bool WouldCreateAssembleChainCycle(Function& function, const std::vector<Operation*>& chain, bool dagVerified)
{
    // See VerifyFunctionIsDag: an acyclic graph stays acyclic under edge-removing merges.
    if (dagVerified) {
        return false;
    }
    if (chain.empty()) {
        return false;
    }
    std::unordered_set<Operation*> chainSet(chain.begin(), chain.end());
    Operation* representative = chain.front();
    std::unordered_map<Operation*, std::unordered_set<Operation*>> adjacency;
    for (auto& op : function.Operations(false)) {
        bool inChain = chainSet.count(&op) != 0;
        if (inChain && &op != representative) {
            // The tail's read of the middle is bypassed; intermediate ops keep their edges
            // only when side consumers keep them alive, and mapping them onto the
            // representative stays conservative.
            continue;
        }
        adjacency.try_emplace(inChain ? representative : &op);
        for (const auto& input : op.GetIOperands()) {
            for (auto* producer : input->GetProducers()) {
                if (producer->BelongTo() == &function && !producer->IsDeleted()) {
                    AddDependencyEdge(adjacency, chainSet.count(producer) != 0 ? representative : producer,
                                      inChain ? representative : &op);
                }
            }
        }
    }
    for (const auto& [token, entry] : function.GetVarDependency().GetAllDependencies()) {
        (void)token;
        for (const auto& producerStmt : entry.producers) {
            auto* producer = ToOperation(producerStmt);
            for (const auto& consumerStmt : entry.consumers) {
                auto* consumer = ToOperation(consumerStmt);
                AddDependencyEdge(adjacency, chainSet.count(producer) != 0 ? representative : producer,
                                  chainSet.count(consumer) != 0 ? representative : consumer);
            }
        }
    }
    return HasDependencyCycle(adjacency);
}

void CollectLinearTokenDependency(Function& function, const std::vector<Operation*>& chain,
                                  TokenMergeAssembleUtils::TokenDependency& tokenDependency)
{
    std::unordered_set<Operation*> chainSet(chain.begin(), chain.end());
    for (const auto& snapshot : CollectTokenSnapshots(function, chainSet)) {
        AddUnique(tokenDependency.touchedTokens, snapshot.token);
        bool producerInChain = chainSet.count(snapshot.producer) != 0;
        bool consumerInChain = std::any_of(snapshot.consumers.begin(), snapshot.consumers.end(),
                                           [&chainSet](Operation* consumer) { return chainSet.count(consumer) != 0; });
        if (!producerInChain && consumerInChain) {
            AddUnique(tokenDependency.inputTokens, snapshot.token);
            continue;
        }
        if (!producerInChain) {
            continue;
        }
        std::vector<Operation*> externalConsumers;
        for (auto* consumer : snapshot.consumers) {
            if (chainSet.count(consumer) == 0) {
                AddUnique(externalConsumers, consumer);
            }
        }
        if (externalConsumers.empty()) {
            continue;
        }
        AddUnique(tokenDependency.resultTokens, snapshot.token);
        tokenDependency.resultTokenConsumers.emplace_back();
        for (auto* consumer : externalConsumers) {
            AddUnique(tokenDependency.resultTokenConsumers.back(), ToStmtPtr(*consumer));
        }
    }
}

void ClearLinearTokenDependency(Function& function, const std::vector<Operation*>& chain,
                                const TokenMergeAssembleUtils::TokenDependency& tokenDependency)
{
    std::unordered_set<Operation*> chainSet(chain.begin(), chain.end());
    auto snapshots = CollectTokenSnapshots(function, chainSet);
    auto& dependency = function.GetVarDependency();
    for (const auto& snapshot : snapshots) {
        bool producerInChain = chainSet.count(snapshot.producer) != 0;
        bool isResultToken = std::find(tokenDependency.resultTokens.begin(), tokenDependency.resultTokens.end(),
                                       snapshot.token) != tokenDependency.resultTokens.end();
        if (isResultToken) {
            // 保留旧 token 的 varDependency：扇出场景下同一 producer 的 result token 会被多条
            // 合并链共享，后续链仍需收集它及其外部消费者；统一由 CleanupLegacyResultTokens 移除。
            if (producerInChain && snapshot.producer != nullptr) {
                auto& producerTokens = snapshot.producer->result_token_;
                producerTokens.erase(std::remove(producerTokens.begin(), producerTokens.end(), snapshot.token),
                                     producerTokens.end());
            }
            continue;
        }
        for (auto* consumer : snapshot.consumers) {
            if (chainSet.count(consumer) != 0 || producerInChain) {
                // Keep the varDependency consumer entry: sibling chains passing through
                // the same chain op must still be able to collect this token, and every
                // such chain's merged op waits it independently. The op-level list is
                // cleared below; the stale entry of the deleted op is inert and gets
                // cleaned up with the op itself.
                RemoveToken(*consumer, snapshot.token);
            }
        }
        if (producerInChain && snapshot.producer != nullptr) {
            dependency.RemoveProducer(snapshot.token, ToStmtPtr(*snapshot.producer));
        }
        if (producerInChain) {
            dependency.RemoveVar(snapshot.token);
        }
    }
    for (auto* op : chain) {
        op->tokens_.clear();
        op->result_token_.clear();
    }
}

void ApplyLinearTokenDependency(Function& function, Operation& mergedOp,
                                const TokenMergeAssembleUtils::TokenDependency& tokenDependency)
{
    for (const auto& token : tokenDependency.inputTokens) {
        AddTokenConsumer(function, token, mergedOp);
    }
    for (size_t index = 0; index < tokenDependency.resultTokens.size(); ++index) {
        const auto& resultToken = tokenDependency.resultTokens[index];
        if (resultToken == nullptr) {
            continue;
        }
        // token 只允许单生产者：merged op 产出新 token 接管该旧 token 的全部外部消费者。
        // 扇出场景下同一旧 token 被多条链共享时，每条链的 merged op 各产一个新 token，
        // 消费者等待全部新 token，与原语义（等旧 token 的唯一生产者）保持一致。
        auto newToken = IRBuilder().CreateTokenVar(mergedOp.GetSpan());
        AddUnique(mergedOp.result_token_, newToken);
        function.GetVarDependency().AddProducer(newToken, ToStmtPtr(mergedOp));
        if (index < tokenDependency.resultTokenConsumers.size()) {
            for (const auto& consumerStmt : tokenDependency.resultTokenConsumers[index]) {
                AddTokenConsumer(function, newToken, *ToOperation(consumerStmt));
            }
        }
    }
}

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

bool HasDataPath(Operation* producer, Operation* consumer)
{
    if (producer == nullptr || consumer == nullptr) {
        return false;
    }
    std::vector<Operation*> pending{consumer};
    std::unordered_set<Operation*> visited;
    while (!pending.empty()) {
        auto* current = pending.back();
        pending.pop_back();
        if (!visited.insert(current).second) {
            continue;
        }
        for (auto* predecessor : current->ProducerOps()) {
            if (predecessor == producer) {
                return true;
            }
            if (predecessor != nullptr && predecessor->BelongTo() == consumer->BelongTo() &&
                !predecessor->IsDeleted()) {
                pending.emplace_back(predecessor);
            }
        }
    }
    return false;
}

} // namespace

Status TokenMergeAssembleUtils::MergeViewAssemble(Function& function)
{
    TokenMergeAssembleUtils TokenMergeAssembleUtils;
    Status status = TokenMergeAssembleUtils.Process(function);
    return status;
}

Status TokenMergeAssembleUtils::Process(Function& function)
{
    Status status = Initialize();
    if (status != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "TokenMergeAssembleUtils initialization failed.");
        return status;
    }
    DeadOperationEliminator eliminator;
    eliminator.EliminateOperation(function, false, false);
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

Status TokenMergeAssembleUtils::Initialize()
{
    visitedOp_.clear();
    viewOpToAppend_.clear();
    assembleOpToAppend_.clear();
    consumerCache_.clear();
    tensorConsumerCache_.clear();
    rawTensorVersions_.clear();
    candidateOps_.clear();
    recordedMergeInputMagics_.clear();
    coverageCache_.clear();
    dagVerified_ = false;
    return SUCCESS;
}

const TokenMergeAssembleUtils::ConsumerCacheEntry& TokenMergeAssembleUtils::BuildTensorConsumerCache(
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
    cacheEntry.producerCount = tensor->GetProducers().size();
    cacheEntry.allProducersAreAssembleLike = cacheEntry.producerCount != 0;
    for (auto* producer : tensor->GetProducers()) {
        if (producer == nullptr || producer->BelongTo() != &function || producer->IsDeleted() ||
            !IsAssembleLikeOpcode(producer->GetOpcode())) {
            cacheEntry.allProducersAreAssembleLike = false;
            break;
        }
    }
    for (auto* consumer : tensor->GetConsumers()) {
        if (consumer == nullptr || consumer->BelongTo() != &function || consumer->IsDeleted()) {
            continue;
        }
        if (IsViewLikeOpcode(consumer->GetOpcode())) {
            cacheEntry.viewConsumers.emplace_back(consumer);
            cacheEntry.hasAssembleChainStopper = true;
        } else if (IsAssembleLikeOpcode(consumer->GetOpcode())) {
            cacheEntry.assembleConsumers.emplace_back(consumer);
            cacheEntry.hasViewChainStopper |= !consumer->result_token_.empty() || !consumer->tokens_.empty() ||
                                              IsFullTensorAssemble(*consumer);
        } else {
            cacheEntry.hasViewChainStopper = true;
            cacheEntry.hasAssembleChainStopper = true;
        }
    }
    return cacheEntry;
}

Status TokenMergeAssembleUtils::BuildConsumerCache(Function& function)
{
    hasTokenDependencies_ = false;
    auto operations = function.Operations(false);
    consumerCache_.reserve(operations.size());
    tensorConsumerCache_.reserve(operations.size());
    candidateOps_.reserve(operations.size());
    auto recordTensor = [this](const LogicalTensorPtr& tensor) {
        if (tensor == nullptr) {
            return;
        }
        auto& versions = rawTensorVersions_[tensor->GetRawMagic()];
        if (std::none_of(versions.begin(), versions.end(), [&tensor](const LogicalTensorPtr& existing) {
                return existing->GetMagic() == tensor->GetMagic();
            })) {
            versions.emplace_back(tensor);
        }
    };
    for (const auto& incast : function.GetIncast()) {
        recordTensor(incast);
    }
    for (const auto& outcast : function.GetOutcast()) {
        recordTensor(outcast);
    }
    for (auto& operation : operations) {
        hasTokenDependencies_ |= !operation.tokens_.empty() || !operation.result_token_.empty();
        for (const auto& input : operation.GetIOperands()) {
            recordTensor(input);
        }
        for (const auto& output : operation.GetOOperands()) {
            recordTensor(output);
        }
        if (!IsViewLikeOpcode(operation.GetOpcode()) && !IsAssembleLikeOpcode(operation.GetOpcode())) {
            continue;
        }
        candidateOps_.emplace_back(&operation);
        if (operation.oOperand.empty()) {
            continue;
        }
        consumerCache_[operation.GetOpMagic()] = &BuildTensorConsumerCache(function, operation.oOperand.front());
    }
    // The LIGHTWEIGHT sort at the head of ProcessOperations is itself a full DAG
    // verification over data and token edges (it aborts on cycles), so the graph is
    // acyclic here. Chain merging only removes edges from this point on, letting every
    // per-chain cycle check short-circuit for the rest of this pass run.
    dagVerified_ = true;
    return SUCCESS;
}

const TokenMergeAssembleUtils::ConsumerCacheEntry& TokenMergeAssembleUtils::GetConsumers(
    const Operation& operation) const
{
    static const ConsumerCacheEntry emptyEntry;
    auto iter = consumerCache_.find(operation.GetOpMagic());
    if (iter == consumerCache_.end() || iter->second == nullptr) {
        return emptyEntry;
    }
    return *iter->second;
}

namespace {
struct CoverageRegion {
    std::vector<int64_t> begin;
    std::vector<int64_t> end;
};

// Builds the write regions of the producers on the middle tensor. Returns false when a
// producer is not a statically placed single-in single-out assemble (regions unresolvable).
bool BuildCoverageRegions(const std::vector<int64_t>& targetShape, const std::vector<Operation*>& producers,
                          std::vector<CoverageRegion>& regions)
{
    for (auto* producer : producers) {
        if (producer == nullptr || !IsAssembleLikeOpcode(producer->GetOpcode()) || producer->iOperand.size() != 1 ||
            producer->oOperand.size() != 1) {
            return false;
        }
        auto attr = std::dynamic_pointer_cast<AssembleOpAttribute>(producer->GetOpAttribute());
        if (attr == nullptr) {
            return false;
        }
        auto offset = attr->GetToOffset();
        const auto& dynOffset = attr->GetToDynOffset();
        if (!dynOffset.empty()) {
            if (dynOffset.size() != targetShape.size() ||
                std::any_of(dynOffset.begin(), dynOffset.end(),
                            [](const SymbolicScalar& value) { return !value.ConcreteValid(); })) {
                return false;
            }
            offset.clear();
            std::transform(dynOffset.begin(), dynOffset.end(), std::back_inserter(offset),
                           [](const SymbolicScalar& value) { return value.Concrete(); });
        }
        const auto& shape = producer->iOperand.front()->GetShape();
        if (offset.size() != targetShape.size() || shape.size() != targetShape.size()) {
            return false;
        }
        CoverageRegion region{offset, offset};
        for (size_t dim = 0; dim < targetShape.size(); ++dim) {
            if (offset[dim] < 0 || shape[dim] <= 0 || offset[dim] > targetShape[dim] - shape[dim]) {
                return false;
            }
            region.end[dim] += shape[dim];
        }
        regions.emplace_back(std::move(region));
    }
    return true;
}

// Grid-cell check: every cell of the boundary grid must fall inside at least one region.
bool CheckRegionsCoverCells(const std::vector<std::vector<int64_t>>& boundaries,
                            const std::vector<CoverageRegion>& regions)
{
    std::vector<int64_t> point(boundaries.size(), 0);
    std::function<bool(size_t)> checkCells = [&](size_t dim) {
        if (dim == boundaries.size()) {
            return std::any_of(regions.begin(), regions.end(), [&point](const CoverageRegion& region) {
                for (size_t index = 0; index < point.size(); ++index) {
                    if (point[index] < region.begin[index] || point[index] >= region.end[index]) {
                        return false;
                    }
                }
                return true;
            });
        }
        for (size_t index = 0; index + 1 < boundaries[dim].size(); ++index) {
            point[dim] = boundaries[dim][index];
            if (!checkCells(dim + 1)) {
                return false;
            }
        }
        return true;
    };
    return checkCells(0);
}
} // namespace

bool TokenMergeAssembleUtils::HasChainOutSurvivor(const std::vector<Operation*>& chain) const
{
    for (auto it = chain.begin(); it != chain.end() - 1; ++it) {
        for (const auto& output : (*it)->GetOOperands()) {
            for (auto* consumer : output->GetConsumers()) {
                if (consumer == nullptr || consumer->IsDeleted() ||
                    std::find(chain.begin(), chain.end(), consumer) != chain.end()) {
                    continue;
                }
                return true;
            }
            // A merge recorded for an earlier chain still reads this output: the merged
            // op is only created in the append phase, after all chain processing.
            if (recordedMergeInputMagics_.count(output->GetMagic()) != 0) {
                return true;
            }
        }
    }
    return false;
}

bool TokenMergeAssembleUtils::HasCompleteStaticCoverage(const LogicalTensorPtr& middle,
                                                        const std::vector<Operation*>& producers) const
{
    if (middle == nullptr) {
        return false;
    }
    auto cached = coverageCache_.find(middle->GetMagic());
    if (cached != coverageCache_.end()) {
        return cached->second;
    }
    bool result = ComputeCompleteStaticCoverage(middle, producers);
    coverageCache_.emplace(middle->GetMagic(), result);
    return result;
}

bool TokenMergeAssembleUtils::ComputeCompleteStaticCoverage(const LogicalTensorPtr& middle,
                                                            const std::vector<Operation*>& producers) const
{
    if (middle == nullptr || middle->GetShape().empty() || producers.empty()) {
        return false;
    }
    const auto& targetShape = middle->GetShape();
    std::vector<CoverageRegion> regions;
    if (!BuildCoverageRegions(targetShape, producers, regions)) {
        return false;
    }
    // Collect per-dimension boundary points (region edges plus tensor bounds) to form a
    // grid; cap the total cell count to keep the check bounded.
    std::vector<std::vector<int64_t>> boundaries(targetShape.size());
    for (size_t dim = 0; dim < targetShape.size(); ++dim) {
        if (targetShape[dim] <= 0) {
            return false;
        }
        boundaries[dim] = {0, targetShape[dim]};
    }
    for (const auto& region : regions) {
        for (size_t dim = 0; dim < targetShape.size(); ++dim) {
            boundaries[dim].push_back(region.begin[dim]);
            boundaries[dim].push_back(region.end[dim]);
        }
    }
    size_t cellCount = 1;
    for (auto& dimensionBoundaries : boundaries) {
        std::sort(dimensionBoundaries.begin(), dimensionBoundaries.end());
        dimensionBoundaries.erase(std::unique(dimensionBoundaries.begin(), dimensionBoundaries.end()),
                                  dimensionBoundaries.end());
        if (dimensionBoundaries.size() < 2 || cellCount > 100000 / (dimensionBoundaries.size() - 1)) {
            return false;
        }
        cellCount *= dimensionBoundaries.size() - 1;
    }
    return CheckRegionsCoverCells(boundaries, regions);
}

bool TokenMergeAssembleUtils::HasSplitVersionContribution(const LogicalTensorPtr& middle,
                                                          const std::vector<Operation*>& currentProducers) const
{
    if (middle == nullptr) {
        return false;
    }
    auto versionsIter = rawTensorVersions_.find(middle->GetRawMagic());
    if (versionsIter == rawTensorVersions_.end() || versionsIter->second.size() <= 1) {
        return false;
    }
    if (HasCompleteStaticCoverage(middle, currentProducers)) {
        return false;
    }
    for (const auto& version : versionsIter->second) {
        if (version == nullptr || version->GetMagic() == middle->GetMagic()) {
            continue;
        }
        for (auto* producer : version->GetProducers()) {
            if (producer != nullptr && !producer->IsDeleted() && IsAssembleLikeOpcode(producer->GetOpcode())) {
                return true;
            }
        }
    }
    return false;
}

Status TokenMergeAssembleUtils::ProcessOperations(Function& function)
{
    function.SortOperations(SortOperationsMode::LIGHTWEIGHT);
    Status status = BuildConsumerCache(function);
    if (status != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "BuildConsumerCache failed.");
        return status;
    }
    status = ProcessCandidateChains(function);
    if (status != SUCCESS) {
        return status;
    }
    return AppendMergedOperations(function);
}

Status TokenMergeAssembleUtils::ProcessCandidateChains(Function& function)
{
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
    return SUCCESS;
}

Status TokenMergeAssembleUtils::AppendMergedOperations(Function& function)
{
    Status status = AppendMergedViewOperations(function);
    if (status != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "AppendMergedViewOperations phase failed.");
        return status;
    }
    status = AppendMergedAssembleOperations(function);
    if (status != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "AppendMergedAssembleOperations phase failed.");
        return FAILED;
    }
    CleanupLegacyResultTokens(function);
    return status;
}

void TokenMergeAssembleUtils::CleanupLegacyResultTokens(Function& function)
{
    auto& dependency = function.GetVarDependency();
    std::vector<ir::VarPtr> legacyTokens;
    auto collectTokens = [&legacyTokens](const TokenDependency& tokenDependency) {
        for (const auto& token : tokenDependency.resultTokens) {
            AddUnique(legacyTokens, token);
        }
    };
    for (const auto& viewOp : viewOpToAppend_) {
        collectTokens(viewOp.tokenDependency);
    }
    for (const auto& assembleOp : assembleOpToAppend_) {
        collectTokens(assembleOp.tokenDependency);
    }
    for (const auto& token : legacyTokens) {
        if (token == nullptr) {
            continue;
        }
        // 拷贝一份消费者：RemoveVar 会使 GetConsumers 返回的引用失效。
        auto consumers = dependency.GetConsumers(token);
        for (const auto& consumerStmt : consumers) {
            auto* consumer = ToOperation(consumerStmt);
            if (consumer != nullptr) {
                RemoveToken(*consumer, token);
            }
        }
        dependency.RemoveVar(token);
    }
}

Status TokenMergeAssembleUtils::AppendMergedViewOperations(Function& function)
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
        ApplyLinearTokenDependency(function, mergedViewOp, viewOp.tokenDependency);
        viewOp.output->UpdateDynValidShape(viewOp.dynValidShape);
    }
    return SUCCESS;
}

Status TokenMergeAssembleUtils::AppendMergedAssembleOperations(Function& function)
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
        if (assembleOp.subgraphId != -1) {
            mergedAssembleOp.UpdateSubgraphID(assembleOp.subgraphId);
        }
        if (!assembleOp.rmwModeAttr.empty()) {
            mergedAssembleOp.SetAttribute(assembleOp.rmwModeAttr, 1L);
        }
        ApplyLinearTokenDependency(function, mergedAssembleOp, assembleOp.tokenDependency);
        if (assembleOp.atomicFromReduceAcc) {
            mergedAssembleOp.SetAttribute(ATOMIC_FROM_REDUCE_ACC_ATTR, true);
        }
        if (assembleOp.atomicFromExplicitRmw) {
            mergedAssembleOp.SetAttribute(ATOMIC_FROM_EXPLICIT_RMW_ATTR, true);
        }
    }
    return SUCCESS;
}

Status TokenMergeAssembleUtils::CleanUp(Function& function)
{
    function.EraseOperations(true, false);
    DeadOperationEliminator eliminator;
    eliminator.EliminateOperation(function, false, false);
    function.SortOperations(SortOperationsMode::LIGHTWEIGHT);
    return SUCCESS;
}

ir::Span TokenMergeAssembleUtils::GetFirstSpan(const std::vector<Operation*>& chain)
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

Operation::ScopeInfo TokenMergeAssembleUtils::GetChainScopeInfo(const std::vector<Operation*>& chain)
{
    for (auto* op : chain) {
        if (op->GetScopeId() != -1) {
            return op->GetScopeInfo();
        }
    }
    return Operation::ScopeInfo();
}

Status TokenMergeAssembleUtils::MergeViewChain(Function& function, Operation& operation, std::vector<Operation*>& chain,
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

void TokenMergeAssembleUtils::InitOperationChain(Operation& operation, std::vector<Operation*>& chain)
{
    visitedOp_.insert(operation.opmagic);
    chain.emplace_back(&operation);
}

Status TokenMergeAssembleUtils::ProcessConsumerChain(Function& function, const ConsumerCacheEntry& consumers,
                                                     std::vector<Operation*>& chain, bool& chainEnd,
                                                     int effectiveScopeId)
{
    bool hasActiveAssembleConsumer = std::any_of(
        consumers.assembleConsumers.begin(), consumers.assembleConsumers.end(),
        [this](Operation* op) { return op != nullptr && visitedOp_.count(op->GetOpMagic()) == 0; });
    if (consumers.viewConsumers.empty() || consumers.hasViewChainStopper || hasActiveAssembleConsumer) {
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
        if (!IsViewLikeOpcode(op->GetOpcode())) {
            chainEnd = true;
            continue;
        }
        auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(op->GetOpAttribute());
        if (viewOpAttribute == nullptr) {
            APASS_LOG_ERROR_F(Elements::Function, "View operation has null viewOpAttribute.");
            return FAILED;
        }
        // 1.unknown memType 可以向它之后的view合并 2.相同memType的view可以合并
        bool canMerge = CanMergeViewLikeChain(chain, op->GetOpcode()) &&
                        (currentMemType == MemoryType::MEM_UNKNOWN || currentMemType == viewOpAttribute->GetTo());
        if (!canMerge) {
            chainEnd = true;
            continue;
        }
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
    }
    return SUCCESS;
}

Status TokenMergeAssembleUtils::ProcessChainEnd(Function& function, std::vector<Operation*>& chain)
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
    if (!startTensor || !endTensor) {
        APASS_LOG_ERROR_F(Elements::Function, "Null tensor found in chain.");
        return FAILED;
    }
    if (HasTokenDependency(chain)) {
        std::unordered_map<Operation*, Operation*> contraction;
        for (auto* op : chain) {
            contraction.emplace(op, chain.front());
        }
        if (WouldCreateCycleAfterContraction(function, contraction, dagVerified_)) {
            return SUCCESS;
        }
    }
    TokenDependency tokenDependency;
    CollectLinearTokenDependency(function, chain, tokenDependency);
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
                              chainScopeInfo, GetMergedViewOpcode(chain), tokenDependency);
    ClearLinearTokenDependency(function, chain, tokenDependency);
    // 清理链尾
    endOp->oOperand.clear();
    function.GetTensorMap().Erase(endTensor);
    return SUCCESS;
}

Status TokenMergeAssembleUtils::CalculateMergedOffsets(const std::vector<Operation*>& chain,
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

void TokenMergeAssembleUtils::RecordMergedViewOperation(
    Operation* lastViewOp, const std::shared_ptr<LogicalTensor>& startTensor,
    const std::shared_ptr<LogicalTensor>& endTensor, const std::vector<int64_t>& newOffset,
    const std::vector<SymbolicScalar>& newDynOffset, const std::vector<SymbolicScalar>& newDynValidShape,
    const ir::Span& span, const Operation::ScopeInfo& scopeInfo, Opcode opcode, const TokenDependency& tokenDependency)
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
                                        std::move(isGemv), span, scopeInfo, opcode, tokenDependency});
    if (startTensor != nullptr) {
        recordedMergeInputMagics_.insert(startTensor->GetMagic());
    }
}

Status TokenMergeAssembleUtils::MergeAssembleChain(Function& function, Operation& operation,
                                                   std::vector<Operation*>& chain, int effectiveScopeId)
{
    // 1. 初始化操作链
    InitAssembleChain(operation, chain);

    int newScopeId = operation.GetScopeId();
    if (effectiveScopeId == -1 && newScopeId != -1) {
        effectiveScopeId = newScopeId;
    }

    // 2. 处理消费者；token 模式下 stopper 仅在图内含 token 时停链（放宽语义），
    // legacy 模式恢复无条件停链（bcb1ceacf 之前的原始语义，case68 基线）。
    const auto& consumers = GetConsumers(operation);
    bool chainEnd = consumers.assembleConsumers.empty() || (hasTokenDependencies_ && consumers.hasAssembleChainStopper);
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

void TokenMergeAssembleUtils::InitAssembleChain(Operation& operation, std::vector<Operation*>& chain)
{
    visitedOp_.insert(operation.opmagic);
    chain.emplace_back(&operation);
}

bool TokenMergeAssembleUtils::CanExtendThroughMiddle(Function& function, Operation& currentOp)
{
    if (currentOp.oOperand.empty()) {
        return false;
    }
    const auto& middle = currentOp.oOperand.front();
    std::vector<Operation*> currentProducers(middle->GetProducers().begin(), middle->GetProducers().end());
    if (HasSplitVersionContribution(middle, currentProducers)) {
        return false;
    }
    if (currentProducers.size() <= 1) {
        return true;
    }
    // Multi-producer (fan-in) middles need extra care: extending the chain through the
    // middle redirects the chain start's region into the consumer's output while the
    // sibling producers' regions are redirected by their own chains. Require assemble-like
    // producers with complete static coverage so no region of the middle loses its copy.
    // Producers already deleted by a sibling chain keep counting: their regions are taken
    // over by that chain's merged op.
    bool allProducersAreAssembleLike = true;
    for (auto* producer : currentProducers) {
        if (producer == nullptr || producer->BelongTo() != &function || !IsAssembleLikeOpcode(producer->GetOpcode())) {
            allProducersAreAssembleLike = false;
            break;
        }
    }
    return allProducersAreAssembleLike && HasCompleteStaticCoverage(middle, currentProducers);
}

// Middle gate for chain extension. Token mode checks split versions plus fan-in
// coverage; legacy stops at multi-producer middles (handled by producer-group fusion)
// and at split-version contributions.
bool TokenMergeAssembleUtils::StopsAtMiddle(Function& function, const ConsumerCacheEntry& consumers,
                                            Operation* currentOp)
{
    (void)consumers;
    if (currentOp == nullptr || currentOp->oOperand.empty()) {
        return true;
    }
    // Both modes share the chain-share middle gate: producer-group fusion claims the
    // canonical fan-in middles first; chains penetrate the remaining multi-producer
    // middles when producers are assemble-like with complete static coverage.
    return !CanExtendThroughMiddle(function, *currentOp);
}

Status TokenMergeAssembleUtils::ProcessAssembleConsumers(Function& function, const ConsumerCacheEntry& consumers,
                                                         std::vector<Operation*>& chain, bool& chainEnd,
                                                         int effectiveScopeId)
{
    if (consumers.assembleConsumers.empty()) {
        return SUCCESS;
    }
    if ((hasTokenDependencies_ && consumers.hasAssembleChainStopper) || consumers.assembleConsumers.empty()) {
        return SUCCESS;
    }
    if (StopsAtMiddle(function, consumers, chain.back())) {
        chainEnd = true;
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

bool TokenMergeAssembleUtils::HasBlockingChainCycle(Function& function, const std::vector<Operation*>& chain)
{
    if (!HasTokenDependency(chain)) {
        return false;
    }
    return WouldCreateAssembleChainCycle(function, chain, dagVerified_);
}

bool TokenMergeAssembleUtils::HasIncompatibleSiblingProducer(Function& function, const std::vector<Operation*>& chain)
{
    // All-or-nothing across sibling producers: deleting the tail severs the only path of
    // any live sibling producer of an intermediate output that cannot itself merge
    // through the tail (contract/scope/rmw rules).
    Operation* tailOp = chain.back();
    for (size_t index = 0; index + 1 < chain.size(); ++index) {
        Operation* current = chain[index];
        if (current->oOperand.empty()) {
            continue;
        }
        for (auto* sibling : current->oOperand.front()->GetProducers()) {
            if (sibling == nullptr || sibling == current || sibling->IsDeleted() || sibling->BelongTo() != &function ||
                std::find(chain.begin(), chain.end(), sibling) != chain.end()) {
                continue;
            }
            int siblingScopeId = sibling->GetScopeId();
            int tailScopeId = tailOp->GetScopeId();
            if (!CanMergeAssembleLikeChain({sibling}, tailOp->GetOpcode()) ||
                !IsRmwModeAttrCompatible({sibling}, *tailOp) ||
                (siblingScopeId != -1 && tailScopeId != -1 && siblingScopeId != tailScopeId)) {
                return true;
            }
        }
    }
    return false;
}

void TokenMergeAssembleUtils::FilterOwnedInputTokens(Function& function, TokenDependency& tokenDependency,
                                                     const std::vector<Operation*>& chain)
{
    // Data-covered precision for input tokens: a token whose producer feeds this chain's
    // start is waited by this chain's merged op only; tokens feeding a sibling producer
    // of an intermediate output are owned by that sibling's chain and skipped here.
    // Tokens feeding no producer of the middle are waited collectively.
    std::vector<ir::VarPtr> preciseInputTokens;
    for (const auto& token : tokenDependency.inputTokens) {
        Operation* tokenProducer = nullptr;
        for (const auto& producerStmt : function.GetVarDependency().GetProducers(token)) {
            tokenProducer = ToOperation(producerStmt);
            break;
        }
        if (tokenProducer != nullptr && HasDataPath(tokenProducer, chain.front())) {
            preciseInputTokens.emplace_back(token);
            continue;
        }
        bool coveredBySibling = false;
        if (tokenProducer != nullptr) {
            for (size_t index = 0; index + 1 < chain.size() && !coveredBySibling; ++index) {
                Operation* current = chain[index];
                if (current->oOperand.empty()) {
                    continue;
                }
                for (auto* sibling : current->oOperand.front()->GetProducers()) {
                    if (sibling == nullptr || sibling == current ||
                        std::find(chain.begin(), chain.end(), sibling) != chain.end()) {
                        continue;
                    }
                    if (HasDataPath(tokenProducer, sibling)) {
                        coveredBySibling = true;
                        break;
                    }
                }
            }
        }
        if (!coveredBySibling) {
            preciseInputTokens.emplace_back(token);
        }
    }
    tokenDependency.inputTokens = std::move(preciseInputTokens);
}

void TokenMergeAssembleUtils::DeleteMergedChainOps(const std::vector<Operation*>& chain,
                                                   const std::shared_ptr<LogicalTensor>& endTensor)
{
    (void)endTensor;
    // Keep intermediate writes needed by side consumers, as in the legacy chain fusion.
    const bool hasSideConsumer = std::any_of(chain.begin(), chain.end() - 1, [this](const Operation* op) {
        return GetConsumers(*op).hasAssembleChainStopper;
    });
    // A consumer outside the chain may still read an intermediate output: its own chain
    // merge can be rejected (rmw/scope conflict) after this chain already merged. Keep
    // the intermediates alive so those readers keep a live producer.
    const bool hasChainOutSurvivor = HasChainOutSurvivor(chain);
    for (auto* op : chain) {
        if ((!hasSideConsumer && !hasChainOutSurvivor) || op == chain.back()) {
            op->SetAsDeleted();
        }
    }
}

Status TokenMergeAssembleUtils::ProcessAssembleChainEnd(Function& function, std::vector<Operation*>& chain,
                                                        Operation& operation)
{
    (void)operation;
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
    // Safety gates: cycle avoidance (token-linked chains) and sibling all-or-nothing
    // (live sibling producers that cannot merge through the tail).
    if (HasBlockingChainCycle(function, chain) || HasIncompatibleSiblingProducer(function, chain)) {
        return SUCCESS;
    }
    TokenDependency tokenDependency;
    CollectLinearTokenDependency(function, chain, tokenDependency);
    FilterOwnedInputTokens(function, tokenDependency, chain);
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
                            GetRmwModeAttrKey(rmwModeAttr), GetMergedAssembleOpcode(chain), tokenDependency,
                            atomicSemanticAttr.fromReduceAcc, atomicSemanticAttr.fromExplicitRmw,
                            chain.back()->GetSubgraphID());
    ClearLinearTokenDependency(function, chain, tokenDependency);
    DeleteMergedChainOps(chain, endTensor);
    function.GetTensorMap().Erase(endTensor);

    return SUCCESS;
}

std::pair<std::vector<int64_t>, std::vector<SymbolicScalar>> TokenMergeAssembleUtils::CalculateAssembleOffsets(
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

void TokenMergeAssembleUtils::RecordAssembleOperation(
    const std::shared_ptr<LogicalTensor>& input, const std::shared_ptr<LogicalTensor>& output,
    const std::vector<int64_t>& offset, const std::vector<SymbolicScalar>& dynOffset, const ir::Span& span,
    const Operation::ScopeInfo& scopeInfo, const std::string& rmwModeAttr, Opcode opcode,
    const TokenDependency& tokenDependency, bool atomicFromReduceAcc, bool atomicFromExplicitRmw, int subgraphId)
{
    AssembleOp assembleOp{input,
                          output,
                          offset,
                          dynOffset,
                          span,
                          scopeInfo,
                          rmwModeAttr,
                          opcode,
                          tokenDependency,
                          atomicFromReduceAcc,
                          atomicFromExplicitRmw};
    // The merged op takes over the chain tail's write; ops created after graph partitioning
    // must carry a valid subgraph id.
    assembleOp.subgraphId = subgraphId;
    if (input != nullptr) {
        recordedMergeInputMagics_.insert(input->GetMagic());
    }
    assembleOpToAppend_.emplace_back(std::move(assembleOp));
}

} // namespace npu::tile_fwk
