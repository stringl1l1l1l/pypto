/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file remove_redundant_op_internal.cpp
 * \brief Internal shared helpers for RemoveRedundantOp and RemoveRedundantOpUtils.
 */

#include "passes/pass_utils/remove_redundant_op_internal.h"

#include <algorithm>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include "interface/tensor/irbuilder.h"
#include "tilefwk/tilefwk_op.h"

namespace npu::tile_fwk::remove_redundant_op_internal {
namespace {

bool IsViewLikeOpcode(Opcode opcode) { return opcode == Opcode::OP_VIEW || opcode == Opcode::OP_SLICE; }

bool IsAssembleLikeOpcode(Opcode opcode) { return opcode == Opcode::OP_ASSEMBLE || opcode == Opcode::OP_CONTRACT; }

ir::StmtPtr ToStmtPtr(Operation& op) { return std::static_pointer_cast<const ir::Stmt>(op.shared_from_this()); }

Operation* ToOperation(const ir::StmtPtr& stmt)
{
    if (stmt == nullptr) {
        return nullptr;
    }
    return static_cast<Operation*>(const_cast<ir::Stmt*>(stmt.get()));
}

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
    if (token == nullptr) {
        return;
    }
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

ir::VarPtr EnsureResultToken(Function& function, Operation& producer)
{
    if (producer.result_token_.empty()) {
        producer.result_token_.push_back(IRBuilder().CreateTokenVar(producer.GetSpan()));
    }
    function.GetVarDependency().AddProducer(producer.result_token_.front(), ToStmtPtr(producer));
    return producer.result_token_.front();
}

bool IsTokenProducedByRemovedOp(Function& function, const ir::VarPtr& token,
                                const std::unordered_set<Operation*>& removedOps)
{
    if (token == nullptr) {
        return false;
    }
    for (const auto& producerStmt : function.GetVarDependency().GetProducers(token)) {
        if (removedOps.count(ToOperation(producerStmt)) != 0) {
            return true;
        }
    }
    return false;
}

std::vector<Operation*> GetExternalTokenConsumers(Function& function, const ir::VarPtr& token,
                                                  const std::unordered_set<Operation*>& removedOps)
{
    std::vector<Operation*> consumers;
    if (token == nullptr) {
        return consumers;
    }
    for (const auto& consumerStmt : function.GetVarDependency().GetConsumers(token)) {
        auto* consumer = ToOperation(consumerStmt);
        if (consumer == nullptr || removedOps.count(consumer) != 0) {
            continue;
        }
        AddUnique(consumers, consumer);
    }
    return consumers;
}

std::vector<Operation*> GetExternalTokenProducers(Function& function, const ir::VarPtr& token,
                                                  const std::unordered_set<Operation*>& removedOps)
{
    std::vector<Operation*> producers;
    if (token == nullptr) {
        return producers;
    }
    for (const auto& producerStmt : function.GetVarDependency().GetProducers(token)) {
        auto* producer = ToOperation(producerStmt);
        if (producer == nullptr || removedOps.count(producer) != 0) {
            continue;
        }
        AddUnique(producers, producer);
    }
    return producers;
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
        indegree.try_emplace(op, 0U);
        for (auto* consumer : consumers) {
            ++indegree[consumer];
        }
    }
    std::queue<Operation*> ready;
    for (const auto& [op, degree] : indegree) {
        if (degree == 0U) {
            ready.push(op);
        }
    }
    size_t visited = 0U;
    while (!ready.empty()) {
        auto* op = ready.front();
        ready.pop();
        ++visited;
        auto iter = adjacency.find(op);
        if (iter == adjacency.end()) {
            continue;
        }
        for (auto* consumer : iter->second) {
            if (--indegree[consumer] == 0U) {
                ready.push(consumer);
            }
        }
    }
    return visited != indegree.size();
}

bool WouldCreateCycleAfterTokenMigration(Function& function, const std::vector<Operation*>& removedOps,
                                         const std::vector<Operation*>& targetConsumers,
                                         const std::vector<Operation*>& targetProducers)
{
    std::unordered_set<Operation*> removedSet(removedOps.begin(), removedOps.end());
    std::unordered_map<Operation*, std::unordered_set<Operation*>> adjacency;
    for (auto& op : function.Operations(false)) {
        if (removedSet.count(&op) != 0) {
            continue;
        }
        adjacency.try_emplace(&op);
        for (const auto& input : op.GetIOperands()) {
            if (input == nullptr) {
                continue;
            }
            for (auto* producer : input->GetProducers()) {
                if (producer != nullptr && producer->BelongTo() == &function && !producer->IsDeleted() &&
                    removedSet.count(producer) == 0) {
                    AddDependencyEdge(adjacency, producer, &op);
                }
            }
        }
    }
    for (const auto& [token, entry] : function.GetVarDependency().GetAllDependencies()) {
        (void)token;
        for (const auto& producerStmt : entry.producers) {
            auto* producer = ToOperation(producerStmt);
            if (removedSet.count(producer) != 0) {
                continue;
            }
            for (const auto& consumerStmt : entry.consumers) {
                auto* consumer = ToOperation(consumerStmt);
                if (removedSet.count(consumer) != 0) {
                    continue;
                }
                AddDependencyEdge(adjacency, producer, consumer);
            }
        }
    }
    for (auto* producer : targetProducers) {
        for (auto* consumer : targetConsumers) {
            AddDependencyEdge(adjacency, producer, consumer);
        }
    }
    for (auto* removedOp : removedOps) {
        if (removedOp == nullptr) {
            continue;
        }
        for (const auto& token : removedOp->tokens_) {
            if (IsTokenProducedByRemovedOp(function, token, removedSet)) {
                continue;
            }
            for (auto* tokenProducer : GetExternalTokenProducers(function, token, removedSet)) {
                for (auto* consumer : targetConsumers) {
                    AddDependencyEdge(adjacency, tokenProducer, consumer);
                }
            }
        }
        for (const auto& resultToken : removedOp->result_token_) {
            for (auto* tokenConsumer : GetExternalTokenConsumers(function, resultToken, removedSet)) {
                for (auto* producer : targetProducers) {
                    AddDependencyEdge(adjacency, producer, tokenConsumer);
                }
            }
        }
    }
    return HasDependencyCycle(adjacency);
}
} // namespace

std::vector<Operation*> GetTensorConsumers(const LogicalTensorPtr& tensor)
{
    std::vector<Operation*> consumers;
    if (tensor == nullptr) {
        return consumers;
    }
    for (auto* consumer : tensor->GetConsumers()) {
        AddUnique(consumers, consumer);
    }
    return consumers;
}

std::vector<Operation*> GetTensorProducers(const LogicalTensorPtr& tensor)
{
    std::vector<Operation*> producers;
    if (tensor == nullptr) {
        return producers;
    }
    for (auto* producer : tensor->GetProducers()) {
        AddUnique(producers, producer);
    }
    return producers;
}

std::vector<Operation*> CollectViewAssembleCascadeOps(const LogicalTensorPtr& startTensor,
                                                      const LogicalTensorPtr& endTensor)
{
    std::vector<Operation*> cascadeOps;
    std::unordered_set<Operation*> visited;
    if (startTensor == nullptr || endTensor == nullptr) {
        return cascadeOps;
    }
    for (auto* viewOp : startTensor->GetConsumers()) {
        if (viewOp == nullptr || !IsViewLikeOpcode(viewOp->GetOpcode()) || viewOp->GetOOperands().empty()) {
            continue;
        }
        auto tempTensor = viewOp->GetOOperands().front();
        bool onlyMatchedAssembleConsumers = true;
        bool hasMatchedAssemble = false;
        for (auto* assembleOp : tempTensor->GetConsumers()) {
            if (assembleOp == nullptr || !IsAssembleLikeOpcode(assembleOp->GetOpcode()) ||
                assembleOp->GetOOperands().empty() || assembleOp->GetOOperands().front() != endTensor) {
                onlyMatchedAssembleConsumers = false;
                continue;
            }
            hasMatchedAssemble = true;
            if (visited.insert(assembleOp).second) {
                cascadeOps.emplace_back(assembleOp);
            }
        }
        if (hasMatchedAssemble && onlyMatchedAssembleConsumers && visited.insert(viewOp).second) {
            cascadeOps.emplace_back(viewOp);
        }
    }
    return cascadeOps;
}

bool CanMigrateRemovedOpsTokenDependency(Function& function, const std::vector<Operation*>& removedOps,
                                         const std::vector<Operation*>& targetConsumers,
                                         const std::vector<Operation*>& targetProducers)
{
    std::unordered_set<Operation*> removedSet(removedOps.begin(), removedOps.end());
    for (auto* removedOp : removedOps) {
        if (removedOp == nullptr) {
            continue;
        }
        for (const auto& token : removedOp->tokens_) {
            if (!IsTokenProducedByRemovedOp(function, token, removedSet) && targetConsumers.empty()) {
                return false;
            }
        }
        bool hasExternalTokenConsumer = false;
        for (const auto& resultToken : removedOp->result_token_) {
            if (!GetExternalTokenConsumers(function, resultToken, removedSet).empty()) {
                hasExternalTokenConsumer = true;
                break;
            }
        }
        if (hasExternalTokenConsumer && targetProducers.empty()) {
            return false;
        }
    }
    // Removing one operation only contracts existing dependency paths:
    // producer -> removed op -> consumer. The migrated producer -> consumer
    // edges therefore cannot introduce a cycle into an acyclic graph.
    if (removedOps.size() == 1U) {
        return true;
    }
    return !WouldCreateCycleAfterTokenMigration(function, removedOps, targetConsumers, targetProducers);
}

void MigrateRemovedOpsTokenDependency(Function& function, const std::vector<Operation*>& removedOps,
                                      const std::vector<Operation*>& targetConsumers,
                                      const std::vector<Operation*>& targetProducers)
{
    std::unordered_set<Operation*> removedSet(removedOps.begin(), removedOps.end());
    auto& dependency = function.GetVarDependency();

    for (auto* removedOp : removedOps) {
        if (removedOp == nullptr) {
            continue;
        }
        auto inputTokens = removedOp->tokens_;
        for (const auto& token : inputTokens) {
            dependency.RemoveConsumer(token, ToStmtPtr(*removedOp));
            RemoveToken(*removedOp, token);
            if (IsTokenProducedByRemovedOp(function, token, removedSet)) {
                continue;
            }
            for (auto* consumer : targetConsumers) {
                AddTokenConsumer(function, token, *consumer);
            }
        }
        removedOp->tokens_.clear();
    }

    for (auto* removedOp : removedOps) {
        if (removedOp == nullptr) {
            continue;
        }
        auto oldTokens = removedOp->result_token_;
        if (oldTokens.empty()) {
            continue;
        }
        std::vector<Operation*> externalConsumers;
        for (const auto& oldToken : oldTokens) {
            for (const auto& consumerStmt : function.GetVarDependency().GetConsumers(oldToken)) {
                auto* consumer = ToOperation(consumerStmt);
                if (consumer != nullptr && removedSet.count(consumer) == 0) {
                    AddUnique(externalConsumers, consumer);
                }
            }
        }
        for (auto* producer : targetProducers) {
            auto newToken = EnsureResultToken(function, *producer);
            for (auto* consumer : externalConsumers) {
                AddTokenConsumer(function, newToken, *consumer);
            }
        }
        for (const auto& oldToken : oldTokens) {
            auto oldTokenConsumers = function.GetVarDependency().GetConsumers(oldToken);
            for (const auto& consumerStmt : oldTokenConsumers) {
                auto* consumer = ToOperation(consumerStmt);
                dependency.RemoveConsumer(oldToken, consumerStmt);
                if (consumer != nullptr) {
                    RemoveToken(*consumer, oldToken);
                }
            }
            dependency.RemoveProducer(oldToken, ToStmtPtr(*removedOp));
            dependency.RemoveVar(oldToken);
        }
        removedOp->result_token_.clear();
    }
}

bool HasOtherAssembleOutputOnSameRaw(Function& function, const LogicalTensorPtr& output, const Operation* ignoredOp)
{
    if (output == nullptr) {
        return false;
    }
    for (auto& op : function.Operations(false)) {
        if (&op == ignoredOp || op.IsDeleted() || !IsAssembleLikeOpcode(op.GetOpcode())) {
            continue;
        }
        for (const auto& opOutput : op.GetOOperands()) {
            if (opOutput != nullptr && opOutput->GetRawMagic() == output->GetRawMagic() &&
                opOutput->GetMagic() != output->GetMagic()) {
                return true;
            }
        }
    }
    return false;
}

} // namespace npu::tile_fwk::remove_redundant_op_internal
