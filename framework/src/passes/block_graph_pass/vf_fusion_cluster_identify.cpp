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
 * \file vf_fusion_cluster_identify.cpp
 * \brief
 */

#include "passes/block_graph_pass/vf_fusion_cluster_identify.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <string>
#include <unordered_set>
#include <utility>

#include "interface/configs/config_manager.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/pass_utils.h"

#undef MODULE_NAME
#define MODULE_NAME "VFFusionClusterIdentify"

namespace npu::tile_fwk {

namespace {
// Cube-task prefix bitset helpers. Cube tasks are indexed in [0, cubeTaskNum).
constexpr size_t PREFIX_WORD_BITS = 64;

size_t PrefixWordNum(size_t taskNum) { return (taskNum + PREFIX_WORD_BITS - 1) / PREFIX_WORD_BITS; }

void SetPrefixBit(std::vector<uint64_t>& bits, size_t taskIndex)
{
    bits[taskIndex / PREFIX_WORD_BITS] |= (uint64_t{1} << (taskIndex % PREFIX_WORD_BITS));
}

bool IsPrefixBitSet(const std::vector<uint64_t>& bits, size_t taskIndex)
{
    return (bits[taskIndex / PREFIX_WORD_BITS] & (uint64_t{1} << (taskIndex % PREFIX_WORD_BITS))) != 0;
}

bool IsPrefixSubsetOf(const std::vector<uint64_t>& sub, const std::vector<uint64_t>& super)
{
    for (size_t w = 0; w < sub.size(); w++) {
        if ((sub[w] & ~super[w]) != 0) {
            return false;
        }
    }
    return true;
}

// Renders a cube-task prefix bitset as the set of task indices, e.g. "{0,2}".
std::string PrefixToString(const std::vector<uint64_t>& bits)
{
    std::string text = "{";
    bool first = true;
    for (size_t w = 0; w < bits.size(); w++) {
        for (size_t b = 0; b < PREFIX_WORD_BITS; b++) {
            if ((bits[w] & (uint64_t{1} << b)) != 0) {
                if (!first) {
                    text += ",";
                }
                text += std::to_string(w * PREFIX_WORD_BITS + b);
                first = false;
            }
        }
    }
    text += "}";
    return text;
}

bool IsPassDebugEnabled()
{
    const auto& logFunc = LogFuncInfo::Instance();
    return logFunc.checkLevel != nullptr && logFunc.checkLevel(PYPTO, DLOG_DEBUG, LogModule::PASS);
}
} // namespace

void AncestorBits::SetBit(std::vector<uint64_t>& bits, size_t bitIndex)
{
    bits[bitIndex / BITSET_WORD_BITS] |= (uint64_t{1} << (bitIndex % BITSET_WORD_BITS));
}

// Transitive closure bitset: O(N²·D/64) time, O(N²/64) space. D = average producer count.
// Suitable for leaf functions with N < 10000 ops (Build ~5-12ms, ~12MB). N > 50000 will
// cause significant memory and time growth; consider alternative algorithms if needed.
void AncestorBits::Build(const std::vector<std::vector<size_t>>& producers, const std::vector<size_t>& topoOrder)
{
    const size_t wordNum = (producers.size() + BITSET_WORD_BITS - 1) / BITSET_WORD_BITS;
    bits_.assign(producers.size(), std::vector<uint64_t>(wordNum, 0));
    for (size_t opIndex : topoOrder) {
        if (opIndex >= producers.size()) {
            continue;
        }
        for (size_t producerIndex : producers[opIndex]) {
            if (producerIndex >= producers.size()) {
                continue;
            }
            SetBit(bits_[opIndex], producerIndex);
            for (size_t i = 0; i < wordNum; i++) {
                bits_[opIndex][i] |= bits_[producerIndex][i];
            }
        }
    }
}

bool AncestorBits::IsAncestor(size_t descendantIndex, size_t ancestorIndex) const
{
    if (descendantIndex >= bits_.size() || ancestorIndex >= bits_.size()) {
        return false;
    }
    return (bits_[descendantIndex][ancestorIndex / BITSET_WORD_BITS] &
            (uint64_t{1} << (ancestorIndex % BITSET_WORD_BITS))) != 0;
}

void VFFusionClusterIdentify::ResetGeneratedClusterIds(const GraphContext& graph)
{
    for (auto* op : graph.ops) {
        if (IsVfFusionCluster(*op)) {
            op->SetAtomicScopeId(-1);
            SetVfFusionCluster(*op, false);
        }
    }
}

bool VFFusionClusterIdentify::IsUserScopedOp(const Operation& op) const
{
    return op.GetAtomicScopeId() > 0 && !IsVfFusionCluster(op);
}

bool VFFusionClusterIdentify::IsVfFusionCluster(const Operation& op) const
{
    return vfFusionClusterOps_.count(&op) > 0;
}

void VFFusionClusterIdentify::SetVfFusionCluster(Operation& op, bool isVfFusionCluster)
{
    if (isVfFusionCluster) {
        vfFusionClusterOps_.insert(&op);
        return;
    }
    vfFusionClusterOps_.erase(&op);
}

std::vector<int> VFFusionClusterIdentify::GetInputClusterIds(const GraphContext& graph, size_t opIndex) const
{
    std::vector<int> clusterIds;
    for (size_t producerIndex : graph.producers[opIndex]) {
        int scopeId = graph.ops[producerIndex]->GetAtomicScopeId();
        if (!IsVfFusionCluster(*graph.ops[producerIndex])) {
            continue;
        }
        clusterIds.emplace_back(scopeId);
    }
    std::sort(clusterIds.begin(), clusterIds.end());
    clusterIds.erase(std::unique(clusterIds.begin(), clusterIds.end()), clusterIds.end());
    return clusterIds;
}

void VFFusionClusterIdentify::SortOpIndicesByMagic(const GraphContext& graph, std::vector<size_t>& opIndices) const
{
    auto byMagic = [&graph](size_t lhs, size_t rhs) {
        if (graph.ops[lhs]->GetOpMagic() != graph.ops[rhs]->GetOpMagic()) {
            return graph.ops[lhs]->GetOpMagic() < graph.ops[rhs]->GetOpMagic();
        }
        return lhs < rhs;
    };
    std::sort(opIndices.begin(), opIndices.end(), byMagic);
}

bool VFFusionClusterIdentify::ReleaseConsumers(const GraphContext& graph, size_t producerIndex,
                                               std::vector<size_t>& remainingProducers,
                                               std::vector<size_t>& releasedConsumers) const
{
    releasedConsumers.clear();
    for (size_t consumerIndex : graph.consumers[producerIndex]) {
        if (remainingProducers[consumerIndex] == 0) {
            APASS_LOG_ERROR_F(Elements::Graph,
                              "Topo order build failed: producer count underflow, producer=%s[%d], "
                              "consumer=%s[%d].",
                              graph.ops[producerIndex]->GetOpcodeStr().c_str(), graph.ops[producerIndex]->GetOpMagic(),
                              graph.ops[consumerIndex]->GetOpcodeStr().c_str(), graph.ops[consumerIndex]->GetOpMagic());
            return false;
        }
        remainingProducers[consumerIndex]--;
        if (remainingProducers[consumerIndex] == 0) {
            releasedConsumers.emplace_back(consumerIndex);
        }
    }
    return true;
}

bool VFFusionClusterIdentify::BuildDfsPriorityTopoOrder(const GraphContext& graph, std::vector<size_t>& topoOrder) const
{
    topoOrder.clear();
    topoOrder.reserve(graph.ops.size());
    std::vector<size_t> remainingProducers(graph.ops.size(), 0);
    for (size_t i = 0; i < graph.producers.size(); i++) {
        remainingProducers[i] = graph.producers[i].size();
    }

    // A plain DFS can emit a consumer before another producer on a multi-input op. Requiring all producers
    // to be processed first keeps the DFS-priority order topologically valid.
    std::vector<size_t> readyStack;
    readyStack.reserve(graph.ops.size());
    std::vector<size_t> initialReadyNodes;
    initialReadyNodes.reserve(graph.ops.size());
    for (size_t i = 0; i < graph.ops.size(); i++) {
        if (remainingProducers[i] == 0) {
            initialReadyNodes.emplace_back(i);
        }
    }
    // Sort ready nodes by a stable graph property so the traversal does not depend on the original oplist order.
    SortOpIndicesByMagic(graph, initialReadyNodes);
    for (auto iter = initialReadyNodes.rbegin(); iter != initialReadyNodes.rend(); iter++) {
        readyStack.emplace_back(*iter);
    }
    APASS_LOG_DEBUG_F(Elements::Graph, "Build DFS-priority topo order: ops=%zu, initialReady=%zu.", graph.ops.size(),
                      initialReadyNodes.size());

    while (!readyStack.empty()) {
        size_t opIndex = readyStack.back();
        readyStack.pop_back();
        topoOrder.emplace_back(opIndex);
        std::vector<size_t> releasedConsumers;
        if (!ReleaseConsumers(graph, opIndex, remainingProducers, releasedConsumers)) {
            return false;
        }
        SortOpIndicesByMagic(graph, releasedConsumers);
        for (auto iter = releasedConsumers.rbegin(); iter != releasedConsumers.rend(); iter++) {
            readyStack.emplace_back(*iter);
        }
    }
    if (topoOrder.size() != graph.ops.size()) {
        APASS_LOG_ERROR_F(Elements::Graph, "Topo order build failed: emitted=%zu, expected=%zu.", topoOrder.size(),
                          graph.ops.size());
        return false;
    }
    return true;
}

bool VFFusionClusterIdentify::MergeClusters(int targetClusterId, int inputClusterId,
                                            std::unordered_map<int, std::vector<Operation*>>& clusters)
{
    if (targetClusterId == inputClusterId) {
        return true;
    }

    auto targetIter = clusters.find(targetClusterId);
    auto inputIter = clusters.find(inputClusterId);
    if (targetIter == clusters.end() || inputIter == clusters.end()) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Merge VF clusters failed: targetCluster=%d exists=%d, "
                          "inputCluster=%d exists=%d.",
                          targetClusterId, targetIter != clusters.end(), inputClusterId, inputIter != clusters.end());
        return false;
    }

    auto& targetOps = targetIter->second;
    size_t inputSize = inputIter->second.size();
    for (auto* op : inputIter->second) {
        op->SetAtomicScopeId(targetClusterId);
        SetVfFusionCluster(*op, true);
        targetOps.emplace_back(op);
    }
    clusters.erase(inputIter);
    APASS_LOG_DEBUG_F(Elements::Operation, "Merged VF cluster %d into %d, movedOps=%zu, targetSize=%zu.",
                      inputClusterId, targetClusterId, inputSize, targetOps.size());
    return true;
}

void VFFusionClusterIdentify::DissolveSingletonClusters(std::unordered_map<int, std::vector<Operation*>>& clusters)
{
    for (auto iter = clusters.begin(); iter != clusters.end();) {
        if (iter->second.size() == 1) {
            auto* op = iter->second.front();
            op->SetAtomicScopeId(-1);
            SetVfFusionCluster(*op, false);
            iter = clusters.erase(iter);
            continue;
        }
        ++iter;
    }
}

bool VFFusionClusterIdentify::AssignClusterToOp(Operation& op,
                                                std::unordered_map<int, std::vector<Operation*>>& clusters,
                                                int& nextClusterId)
{
    if (IsVfFusionCluster(op)) {
        auto iter = clusters.find(op.GetAtomicScopeId());
        if (iter == clusters.end()) {
            APASS_LOG_ERROR_F(Elements::Operation, "Append op %s[%d] to VF cluster failed: cluster %d is missing.",
                              op.GetOpcodeStr().c_str(), op.GetOpMagic(), op.GetAtomicScopeId());
            return false;
        }
        iter->second.emplace_back(&op);
        return true;
    }

    while (clusters.count(nextClusterId) > 0) {
        nextClusterId++;
    }
    int clusterId = nextClusterId++;
    op.SetAtomicScopeId(clusterId);
    SetVfFusionCluster(op, true);
    clusters[clusterId].emplace_back(&op);
    return true;
}

bool VFFusionClusterIdentify::VerifyScheduleTopology(const GraphContext& graph,
                                                     const std::vector<Operation*>& schedule) const
{
    if (!VerifyVfClusterContiguity(schedule)) {
        return false;
    }

    std::unordered_map<Operation*, size_t> scheduledIndex;
    size_t index = 0;
    for (auto* op : schedule) {
        scheduledIndex[op] = index++;
    }
    if (scheduledIndex.size() != graph.ops.size()) {
        APASS_LOG_ERROR_F(Elements::Graph,
                          "Schedule topology verification failed: scheduledUniqueOps=%zu, scheduledOps=%zu, "
                          "graphOps=%zu.",
                          scheduledIndex.size(), schedule.size(), graph.ops.size());
        return false;
    }

    for (size_t consumerIndex = 0; consumerIndex < graph.ops.size(); consumerIndex++) {
        auto consumerIter = scheduledIndex.find(graph.ops[consumerIndex]);
        if (consumerIter == scheduledIndex.end()) {
            APASS_LOG_ERROR_F(Elements::Graph, "Schedule topology verification failed: missing consumer %s[%d].",
                              graph.ops[consumerIndex]->GetOpcodeStr().c_str(), graph.ops[consumerIndex]->GetOpMagic());
            return false;
        }
        for (size_t producerIndex : graph.producers[consumerIndex]) {
            auto producerIter = scheduledIndex.find(graph.ops[producerIndex]);
            if (producerIter == scheduledIndex.end() || producerIter->second >= consumerIter->second) {
                APASS_LOG_ERROR_F(Elements::Graph,
                                  "Schedule topology verification failed: producer %s[%d] position=%zu, "
                                  "consumer %s[%d] position=%zu.",
                                  graph.ops[producerIndex]->GetOpcodeStr().c_str(),
                                  graph.ops[producerIndex]->GetOpMagic(),
                                  producerIter == scheduledIndex.end() ? schedule.size() : producerIter->second,
                                  graph.ops[consumerIndex]->GetOpcodeStr().c_str(),
                                  graph.ops[consumerIndex]->GetOpMagic(), consumerIter->second);
                return false;
            }
        }
    }
    return true;
}

bool VFFusionClusterIdentify::VerifyVfClusterContiguity(const std::vector<Operation*>& schedule) const
{
    std::unordered_map<int, size_t> clusterLastPosition;
    for (size_t position = 0; position < schedule.size(); position++) {
        auto* op = schedule[position];
        if (!IsVfFusionCluster(*op)) {
            continue;
        }

        int clusterId = op->GetAtomicScopeId();
        auto clusterIter = clusterLastPosition.find(clusterId);
        if (clusterIter != clusterLastPosition.end() && clusterIter->second + 1 != position) {
            APASS_LOG_ERROR_F(Elements::Graph,
                              "Schedule topology verification failed: VF cluster %d is non-contiguous, "
                              "previousPosition=%zu, currentPosition=%zu.",
                              clusterId, clusterIter->second, position);
            return false;
        }
        clusterLastPosition[clusterId] = position;
    }
    return true;
}

std::vector<size_t> VFFusionClusterIdentify::GetClusterOpIndices(const GraphContext& graph,
                                                                 const std::vector<Operation*>& clusterOps) const
{
    std::vector<size_t> opIndices;
    opIndices.reserve(clusterOps.size());
    for (auto* op : clusterOps) {
        auto iter = graph.opToIndex.find(op);
        if (iter != graph.opToIndex.end()) {
            opIndices.emplace_back(iter->second);
        } else {
            APASS_LOG_WARN_F(Elements::Operation, "Cannot find graph index for cluster op %s[%d].",
                             op->GetOpcodeStr().c_str(), op->GetOpMagic());
        }
    }
    return opIndices;
}

size_t VFFusionClusterIdentify::CalculateClusterExternalMemory(const GraphContext& graph,
                                                               const std::unordered_set<size_t>& clusterOpIndices) const
{
    std::unordered_set<const LogicalTensor*> externalTensors;
    for (size_t opIndex : clusterOpIndices) {
        if (opIndex >= graph.ops.size()) {
            continue;
        }
        for (const auto& input : graph.ops[opIndex]->GetIOperands()) {
            if (input == nullptr) {
                continue;
            }
            bool hasExternalProducer = input->GetProducers().empty();
            for (auto* producer : input->GetProducers()) {
                auto producerIter = graph.opToIndex.find(producer);
                if (producerIter == graph.opToIndex.end() || clusterOpIndices.count(producerIter->second) == 0) {
                    hasExternalProducer = true;
                    break;
                }
            }
            if (hasExternalProducer) {
                externalTensors.insert(input.get());
            }
        }
    }

    size_t externalMemory = 0;
    for (const auto* tensor : externalTensors) {
        const size_t tensorSize = tensor->MemorySize();
        if (externalMemory > std::numeric_limits<size_t>::max() - tensorSize) {
            return std::numeric_limits<size_t>::max();
        }
        externalMemory += tensorSize;
    }
    return externalMemory;
}

bool VFFusionClusterIdentify::IsFusionResourceWithinLimits(const GraphContext& graph,
                                                           const std::vector<size_t>& existingOpIndices,
                                                           const std::vector<size_t>& addedOpIndices) const
{
    std::unordered_set<size_t> mergedOpIndices;
    mergedOpIndices.reserve(existingOpIndices.size() + addedOpIndices.size());
    mergedOpIndices.insert(existingOpIndices.begin(), existingOpIndices.end());
    mergedOpIndices.insert(addedOpIndices.begin(), addedOpIndices.end());

    const size_t externalMemory = CalculateClusterExternalMemory(graph, mergedOpIndices);
    const size_t ubSize = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB);
    const size_t externalMemoryLimit = ubSize * VF_CLUSTER_EXTERNAL_UB_MEMORY_RATIO_NUMERATOR /
                                       VF_CLUSTER_EXTERNAL_UB_MEMORY_RATIO_DENOMINATOR;
    if (externalMemory > externalMemoryLimit) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Cannot fuse VF ops: external memory %zu exceeds %zu%% of UB (%zu).",
                          externalMemory,
                          VF_CLUSTER_EXTERNAL_UB_MEMORY_RATIO_NUMERATOR * PERCENT_SCALE /
                              VF_CLUSTER_EXTERNAL_UB_MEMORY_RATIO_DENOMINATOR,
                          externalMemoryLimit);
        return false;
    }
    return true;
}

bool VFFusionClusterIdentify::HasDirectDependencyFromCluster(const GraphContext& graph, size_t consumerIndex,
                                                             const std::vector<Operation*>& clusterOps) const
{
    auto clusterOpIndices = GetClusterOpIndices(graph, clusterOps);
    return std::any_of(
        graph.producers[consumerIndex].begin(), graph.producers[consumerIndex].end(), [&](size_t producerIndex) {
            return std::find(clusterOpIndices.begin(), clusterOpIndices.end(), producerIndex) != clusterOpIndices.end();
        });
}

bool VFFusionClusterIdentify::IsFusableOp(const Operation& op) const
{
    if (SUPPORT_VF_FUSE_OPS.count(op.GetOpcode()) == 0) {
        return false;
    }
    auto opcfg = OpcodeManager::Inst().GetTileOpCfg(op.GetOpcode());
    return opcfg.pipeIdStart_ == PipeType::PIPE_V && opcfg.coreType_ == CoreType::AIV;
}

void VFFusionClusterIdentify::RefreshScheduleIndex()
{
    schedulePosition_.clear();
    for (size_t index = 0; index < scheduleOrder_.size(); index++) {
        schedulePosition_[scheduleOrder_[index]] = index;
    }
}

bool VFFusionClusterIdentify::RefreshScheduleIndex(size_t firstPosition, size_t lastPosition)
{
    if (firstPosition > lastPosition || lastPosition >= scheduleOrder_.size()) {
        APASS_LOG_WARN_F(Elements::Graph, "Refresh schedule index failed: first=%zu, last=%zu, scheduleSize=%zu.",
                         firstPosition, lastPosition, scheduleOrder_.size());
        return false;
    }
    for (size_t index = firstPosition; index <= lastPosition; index++) {
        schedulePosition_[scheduleOrder_[index]] = index;
    }
    return true;
}

bool VFFusionClusterIdentify::CanPlaceBlockBeforeCandidates(const ScheduleBlock& block,
                                                            const std::unordered_set<size_t>& candidateOpIndices,
                                                            const AncestorBits& ancestorBits) const
{
    for (size_t opIndex : block.opIndices) {
        for (size_t candidateIndex : candidateOpIndices) {
            if (ancestorBits.IsAncestor(opIndex, candidateIndex)) {
                return false;
            }
        }
    }
    return true;
}

bool VFFusionClusterIdentify::CanPlaceBlockAfterCandidates(const ScheduleBlock& block,
                                                           const std::unordered_set<size_t>& candidateOpIndices,
                                                           const AncestorBits& ancestorBits) const
{
    for (size_t opIndex : block.opIndices) {
        for (size_t candidateIndex : candidateOpIndices) {
            if (ancestorBits.IsAncestor(candidateIndex, opIndex)) {
                return false;
            }
        }
    }
    return true;
}

bool VFFusionClusterIdentify::ValidateScheduleWindowOps(const ScheduleWindow& window) const
{
    if (window.firstPosition > window.lastPosition || window.lastPosition >= scheduleOrder_.size() ||
        window.ops.size() != window.lastPosition - window.firstPosition + 1) {
        APASS_LOG_WARN_F(Elements::Graph,
                         "Validate schedule window ops failed: first=%zu, last=%zu, windowOps=%zu, "
                         "scheduleSize=%zu.",
                         window.firstPosition, window.lastPosition, window.ops.size(), scheduleOrder_.size());
        return false;
    }

    // The replacement must be a permutation of the old window. Pulling in an outside op would require
    // refreshing positions outside [firstPosition, lastPosition], which this incremental interface avoids.
    std::unordered_set<Operation*> expectedOps;
    expectedOps.reserve(window.ops.size());
    for (size_t index = window.firstPosition; index <= window.lastPosition; index++) {
        if (!expectedOps.insert(scheduleOrder_[index]).second) {
            APASS_LOG_WARN_F(Elements::Graph, "Validate schedule window ops failed: duplicated original op %s[%d].",
                             scheduleOrder_[index]->GetOpcodeStr().c_str(), scheduleOrder_[index]->GetOpMagic());
            return false;
        }
    }
    for (auto* op : window.ops) {
        if (expectedOps.erase(op) == 0) {
            APASS_LOG_WARN_F(Elements::Graph, "Validate schedule window ops failed: replacement has outside op %s[%d].",
                             op->GetOpcodeStr().c_str(), op->GetOpMagic());
            return false;
        }
    }
    if (!expectedOps.empty()) {
        APASS_LOG_WARN_F(Elements::Graph, "Validate schedule window ops failed: missing %zu original ops.",
                         expectedOps.size());
        return false;
    }
    return true;
}

bool VFFusionClusterIdentify::ValidateScheduleWindow(const GraphContext& graph, const ScheduleWindow& window) const
{
    if (!ValidateScheduleWindowOps(window)) {
        return false;
    }

    std::unordered_map<Operation*, size_t> windowPosition;
    windowPosition.reserve(window.ops.size());
    for (size_t i = 0; i < window.ops.size(); i++) {
        auto iter = graph.opToIndex.find(window.ops[i]);
        if (iter == graph.opToIndex.end() || windowPosition.count(window.ops[i]) > 0) {
            APASS_LOG_WARN_F(Elements::Graph,
                             "Validate schedule window failed: op %s[%d] missingFromGraph=%d duplicated=%d.",
                             window.ops[i]->GetOpcodeStr().c_str(), window.ops[i]->GetOpMagic(),
                             iter == graph.opToIndex.end(), windowPosition.count(window.ops[i]) > 0);
            return false;
        }
        windowPosition[window.ops[i]] = i;
    }

    for (size_t i = 0; i < window.ops.size(); i++) {
        auto consumerIter = graph.opToIndex.find(window.ops[i]);
        if (consumerIter == graph.opToIndex.end()) {
            APASS_LOG_WARN_F(Elements::Graph, "Validate schedule window failed: consumer %s[%d] missing from graph.",
                             window.ops[i]->GetOpcodeStr().c_str(), window.ops[i]->GetOpMagic());
            return false;
        }
        for (size_t producerIndex : graph.producers[consumerIter->second]) {
            auto producerIter = windowPosition.find(graph.ops[producerIndex]);
            if (producerIter != windowPosition.end() && producerIter->second >= i) {
                APASS_LOG_DEBUG_F(Elements::Graph,
                                  "Schedule window rejected: producer %s[%d] position=%zu after consumer "
                                  "%s[%d] position=%zu.",
                                  graph.ops[producerIndex]->GetOpcodeStr().c_str(),
                                  graph.ops[producerIndex]->GetOpMagic(), producerIter->second,
                                  window.ops[i]->GetOpcodeStr().c_str(), window.ops[i]->GetOpMagic(), i);
                return false;
            }
        }
    }
    return true;
}

bool VFFusionClusterIdentify::TryBuildCompactedScheduleWindow(const GraphContext& graph,
                                                              const std::unordered_set<size_t>& candidateOpIndices,
                                                              const AncestorBits& ancestorBits,
                                                              ScheduleWindow& window) const
{
    window.ops.clear();
    size_t firstPosition = 0;
    size_t lastPosition = 0;
    if (!GetScheduleWindowRange(graph, candidateOpIndices, firstPosition, lastPosition)) {
        return false;
    }
    window.firstPosition = firstPosition;
    window.lastPosition = lastPosition;

    std::vector<ScheduleBlock> beforeBlocks;
    std::vector<Operation*> candidateOps;
    std::vector<ScheduleBlock> afterBlocks;
    window.ops.reserve(lastPosition - firstPosition + 1);
    candidateOps.reserve(candidateOpIndices.size());
    for (size_t scheduleIndex = firstPosition; scheduleIndex <= lastPosition; scheduleIndex++) {
        auto indexIter = graph.opToIndex.find(scheduleOrder_[scheduleIndex]);
        if (indexIter == graph.opToIndex.end()) {
            APASS_LOG_WARN_F(
                Elements::Graph, "Build compacted schedule window failed: scheduled op %s[%d] missing from graph.",
                scheduleOrder_[scheduleIndex]->GetOpcodeStr().c_str(), scheduleOrder_[scheduleIndex]->GetOpMagic());
            return false;
        }
        size_t opIndex = indexIter->second;
        if (candidateOpIndices.count(opIndex) > 0) {
            candidateOps.emplace_back(scheduleOrder_[scheduleIndex]);
            continue;
        }

        ScheduleBlock block;
        if (!BuildScheduleBlock(graph, scheduleIndex, lastPosition, block) ||
            !ClassifyScheduleBlock(block, candidateOpIndices, ancestorBits, beforeBlocks, afterBlocks)) {
            return false;
        }
    }

    AssembleScheduleWindow(beforeBlocks, candidateOps, afterBlocks, window);
    return ValidateScheduleWindow(graph, window);
}

bool VFFusionClusterIdentify::GetScheduleWindowRange(const GraphContext& graph,
                                                     const std::unordered_set<size_t>& candidateOpIndices,
                                                     size_t& firstPosition, size_t& lastPosition) const
{
    if (candidateOpIndices.empty() || scheduleOrder_.empty()) {
        return false;
    }
    firstPosition = scheduleOrder_.size();
    lastPosition = 0;
    for (size_t candidateIndex : candidateOpIndices) {
        if (candidateIndex >= graph.ops.size()) {
            APASS_LOG_WARN_F(Elements::Graph,
                             "Build compacted schedule window failed: candidate index %zu out of graph size %zu.",
                             candidateIndex, graph.ops.size());
            return false;
        }
        auto positionIter = schedulePosition_.find(graph.ops[candidateIndex]);
        if (positionIter == schedulePosition_.end()) {
            APASS_LOG_WARN_F(
                Elements::Graph, "Build compacted schedule window failed: candidate %s[%d] has no schedule position.",
                graph.ops[candidateIndex]->GetOpcodeStr().c_str(), graph.ops[candidateIndex]->GetOpMagic());
            return false;
        }
        firstPosition = std::min(firstPosition, positionIter->second);
        lastPosition = std::max(lastPosition, positionIter->second);
    }
    if (firstPosition == scheduleOrder_.size()) {
        APASS_LOG_WARN_F(Elements::Graph, "Build compacted schedule window failed: no valid candidate position.");
        return false;
    }
    return true;
}

bool VFFusionClusterIdentify::BuildScheduleBlock(const GraphContext& graph, size_t& scheduleIndex, size_t lastPosition,
                                                 ScheduleBlock& block) const
{
    block = {};
    int clusterId = scheduleOrder_[scheduleIndex]->GetAtomicScopeId();
    while (scheduleIndex <= lastPosition) {
        auto* op = scheduleOrder_[scheduleIndex];
        auto indexIter = graph.opToIndex.find(op);
        if (indexIter == graph.opToIndex.end()) {
            APASS_LOG_WARN_F(Elements::Graph,
                             "Build compacted schedule window failed: clustered op %s[%d] missing from graph.",
                             op->GetOpcodeStr().c_str(), op->GetOpMagic());
            return false;
        }
        block.opIndices.emplace_back(indexIter->second);
        block.ops.emplace_back(op);
        if (!IsVfFusionCluster(*op) || scheduleIndex == lastPosition ||
            !IsVfFusionCluster(*scheduleOrder_[scheduleIndex + 1]) ||
            scheduleOrder_[scheduleIndex + 1]->GetAtomicScopeId() != clusterId) {
            return true;
        }
        scheduleIndex++;
    }
    return true;
}

bool VFFusionClusterIdentify::ClassifyScheduleBlock(const ScheduleBlock& block,
                                                    const std::unordered_set<size_t>& candidateOpIndices,
                                                    const AncestorBits& ancestorBits,
                                                    std::vector<ScheduleBlock>& beforeBlocks,
                                                    std::vector<ScheduleBlock>& afterBlocks) const
{
    if (CanPlaceBlockBeforeCandidates(block, candidateOpIndices, ancestorBits)) {
        beforeBlocks.emplace_back(block);
        return true;
    }
    if (CanPlaceBlockAfterCandidates(block, candidateOpIndices, ancestorBits)) {
        afterBlocks.emplace_back(block);
        return true;
    }
    APASS_LOG_DEBUG_F(Elements::Graph,
                      "Build compacted schedule window failed: middle block headed by %s[%d] cannot move "
                      "before or after candidates.",
                      block.ops.front()->GetOpcodeStr().c_str(), block.ops.front()->GetOpMagic());
    return false;
}

void VFFusionClusterIdentify::AssembleScheduleWindow(const std::vector<ScheduleBlock>& beforeBlocks,
                                                     const std::vector<Operation*>& candidateOps,
                                                     const std::vector<ScheduleBlock>& afterBlocks,
                                                     ScheduleWindow& window) const
{
    for (const auto& block : beforeBlocks) {
        window.ops.insert(window.ops.end(), block.ops.begin(), block.ops.end());
    }
    window.ops.insert(window.ops.end(), candidateOps.begin(), candidateOps.end());
    for (const auto& block : afterBlocks) {
        window.ops.insert(window.ops.end(), block.ops.begin(), block.ops.end());
    }
}

bool VFFusionClusterIdentify::IsCubeOp(const Operation& op) const
{
    return op.HasAttribute(OpAttributeKey::isCube) && op.GetBoolAttribute(OpAttributeKey::isCube);
}

// Chain-merge directly connected cube ops into one cube task, mirroring the cube side of
// TaskSplitter::UnionSameCoreOps (including its L1 special-case: a cube producer whose first
// output tensor lives in L1 with multiple consumers is not unioned with an L1_TO_L0 consumer).
// Cube ops separated by vector ops stay in different tasks. A one-op group forms no cube task
// only when the op is a pure DDR->L1 move (all inputs on DDR, all outputs on L1): such an
// isolated L1 copy (e.g. one shared L1_COPY_IN feeding several branches) carries no
// compute-chain semantics and would only fake a shared prefix for all downstream branches.
// Other singletons (e.g. a lone MATMUL) still form their own task.
void VFFusionClusterIdentify::BuildCubeTasks(const GraphContext& graph, std::vector<int>& cubeTaskOfOp,
                                             std::vector<std::vector<size_t>>& taskMembers,
                                             std::vector<std::vector<std::string>>& taskMatMulOps) const
{
    const size_t opNum = graph.ops.size();
    cubeTaskOfOp.assign(opNum, -1);
    taskMembers.clear();
    taskMatMulOps.clear();
    std::vector<size_t> parent(opNum, 0);
    for (size_t i = 0; i < opNum; i++) {
        parent[i] = i;
    }
    auto findRoot = [&parent](size_t idx) {
        while (parent[idx] != idx) {
            parent[idx] = parent[parent[idx]];
            idx = parent[idx];
        }
        return idx;
    };
    // Same premise as TaskSplitter::UnionSameCoreOps: L1_TO_L0 ops are cube ops generated by
    // the tile-graph move-op pass, so an L1 multi-consumer cube producer can meet them here.
    auto isL1MultiConsumerProducer = [&graph](size_t producerIndex) {
        const auto output = graph.ops[producerIndex]->GetOutputOperand(0);
        return output != nullptr && output->GetMemoryTypeOriginal() == MemoryType::MEM_L1 &&
               graph.consumers[producerIndex].size() > 1;
    };
    auto isL1ToL0Op = [&graph](size_t opIndex) {
        return graph.ops[opIndex]->GetOpcodeStr().find("L1_TO_L0") != std::string::npos;
    };
    auto isDdrToL1Op = [&graph](size_t opIndex) {
        const auto& inputs = graph.ops[opIndex]->GetIOperands();
        const auto& outputs = graph.ops[opIndex]->GetOOperands();
        if (inputs.empty() || outputs.empty()) {
            return false;
        }
        for (const auto& input : inputs) {
            if (input == nullptr || input->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
                return false;
            }
        }
        for (const auto& output : outputs) {
            if (output == nullptr || output->GetMemoryTypeOriginal() != MemoryType::MEM_L1) {
                return false;
            }
        }
        return true;
    };
    std::unordered_map<size_t, size_t> rootMemberCount;
    for (size_t producerIndex = 0; producerIndex < opNum; producerIndex++) {
        if (!IsCubeOp(*graph.ops[producerIndex])) {
            continue;
        }
        rootMemberCount[findRoot(producerIndex)]++;
        const bool l1MultiConsumerSkip = isL1MultiConsumerProducer(producerIndex);
        for (size_t consumerIndex : graph.consumers[producerIndex]) {
            if (!IsCubeOp(*graph.ops[consumerIndex])) {
                continue;
            }
            if (l1MultiConsumerSkip && isL1ToL0Op(consumerIndex)) {
                continue;
            }
            const size_t producerRoot = findRoot(producerIndex);
            const size_t consumerRoot = findRoot(consumerIndex);
            if (producerRoot == consumerRoot) {
                continue;
            }
            auto consumerCountIter = rootMemberCount.find(consumerRoot);
            if (consumerCountIter != rootMemberCount.end()) {
                rootMemberCount[producerRoot] += consumerCountIter->second;
                rootMemberCount.erase(consumerCountIter);
            }
            parent[consumerRoot] = producerRoot;
        }
    }
    std::unordered_map<size_t, int> rootToTask;
    size_t cubeOpNum = 0;
    size_t skippedSingletonNum = 0;
    const bool debugEnabled = IsPassDebugEnabled();
    for (size_t i = 0; i < opNum; i++) {
        if (!IsCubeOp(*graph.ops[i])) {
            continue;
        }
        cubeOpNum++;
        size_t root = findRoot(i);
        // Singleton rule, see function comment: skip only one-op groups that are pure DDR->L1
        // moves; they stay out of the prefix context (cubeTaskOfOp == -1).
        if (rootMemberCount[root] == 1 && isDdrToL1Op(i)) {
            skippedSingletonNum++;
            continue;
        }
        auto iter = rootToTask.find(root);
        if (iter == rootToTask.end()) {
            iter = rootToTask.emplace(root, static_cast<int>(taskMembers.size())).first;
            taskMembers.emplace_back();
            taskMatMulOps.emplace_back();
        }
        const size_t taskIndex = static_cast<size_t>(iter->second);
        cubeTaskOfOp[i] = iter->second;
        taskMembers[taskIndex].emplace_back(i);
        if (debugEnabled && OpcodeManager::Inst().GetOpCalcType(graph.ops[i]->GetOpcode()) == OpCalcType::MATMUL) {
            taskMatMulOps[taskIndex].emplace_back(graph.ops[i]->GetOpcodeStr() + "[" +
                                                  std::to_string(graph.ops[i]->GetOpMagic()) + "]");
        }
    }
    APASS_LOG_DEBUG_F(Elements::Graph, "Built cube tasks: cubeOps=%zu, cubeTasks=%zu, skippedSingletons=%zu.",
                      cubeOpNum, taskMembers.size(), skippedSingletonNum);
    if (debugEnabled) {
        for (size_t taskIndex = 0; taskIndex < taskMembers.size(); taskIndex++) {
            std::string matMulOps = "[";
            for (size_t i = 0; i < taskMatMulOps[taskIndex].size(); i++) {
                if (i != 0) {
                    matMulOps += ",";
                }
                matMulOps += taskMatMulOps[taskIndex][i];
            }
            matMulOps += "]";
            APASS_LOG_DEBUG_F(Elements::Graph, "Cube task[%zu]: members=%zu, matmulOps=%s.", taskIndex,
                              taskMembers[taskIndex].size(), matMulOps.c_str());
        }
    }
}

void VFFusionClusterIdentify::ComputeOpPrefixBits(const GraphContext& graph, const std::vector<size_t>& topoOrder,
                                                  const std::vector<int>& cubeTaskOfOp, size_t cubeTaskNum,
                                                  std::vector<std::vector<uint64_t>>& opPrefix) const
{
    const size_t wordNum = PrefixWordNum(cubeTaskNum);
    opPrefix.assign(graph.ops.size(), std::vector<uint64_t>(wordNum, 0));
    for (size_t opIndex : topoOrder) {
        if (opIndex >= graph.ops.size()) {
            continue;
        }
        auto& bits = opPrefix[opIndex];
        for (size_t producerIndex : graph.producers[opIndex]) {
            if (producerIndex >= graph.ops.size()) {
                continue;
            }
            const auto& producerBits = opPrefix[producerIndex];
            for (size_t w = 0; w < wordNum; w++) {
                bits[w] |= producerBits[w];
            }
            int task = cubeTaskOfOp[producerIndex];
            if (task >= 0) {
                SetPrefixBit(bits, static_cast<size_t>(task));
            }
        }
    }
}

void VFFusionClusterIdentify::ComputeTaskPrefixBits(const std::vector<std::vector<size_t>>& taskMembers,
                                                    const std::vector<std::vector<uint64_t>>& opPrefix,
                                                    size_t cubeTaskNum,
                                                    std::vector<std::vector<uint64_t>>& taskPrefix) const
{
    taskPrefix.assign(taskMembers.size(), std::vector<uint64_t>(PrefixWordNum(cubeTaskNum), 0));
    for (size_t taskIndex = 0; taskIndex < taskMembers.size(); taskIndex++) {
        auto& bits = taskPrefix[taskIndex];
        for (size_t memberIndex : taskMembers[taskIndex]) {
            if (memberIndex >= opPrefix.size()) {
                continue;
            }
            for (size_t w = 0; w < bits.size(); w++) {
                bits[w] |= opPrefix[memberIndex][w];
            }
        }
        // A task never counts itself as its own prefix member.
        bits[taskIndex / PREFIX_WORD_BITS] &= ~(uint64_t{1} << (taskIndex % PREFIX_WORD_BITS));
    }
}

void VFFusionClusterIdentify::BuildPrefixContext(const GraphContext& graph, const std::vector<size_t>& topoOrder,
                                                 PrefixContext& prefixCtx) const
{
    BuildCubeTasks(graph, prefixCtx.cubeTaskOfOp, prefixCtx.taskMembers, prefixCtx.taskMatMulOps);
    const size_t cubeTaskNum = prefixCtx.taskMembers.size();
    ComputeOpPrefixBits(graph, topoOrder, prefixCtx.cubeTaskOfOp, cubeTaskNum, prefixCtx.opPrefix);
    ComputeTaskPrefixBits(prefixCtx.taskMembers, prefixCtx.opPrefix, cubeTaskNum, prefixCtx.taskPrefix);
    APASS_LOG_DEBUG_F(Elements::Graph, "Built prefix context: ops=%zu, cubeTasks=%zu.", graph.ops.size(), cubeTaskNum);
}

// Union of the cube-task prefixes of a set of ops: the side's collective prefix. All side ops are
// vector ops (cluster members and mergeable consumers are gated by IsFusableOp), so the union of
// their ancestor bitsets is exactly the set of cube tasks the side depends on.
std::vector<uint64_t> VFFusionClusterIdentify::UnionOpPrefixes(const PrefixContext& prefixCtx,
                                                               const std::vector<size_t>& opIndices) const
{
    const size_t wordNum = PrefixWordNum(prefixCtx.taskMembers.size());
    std::vector<uint64_t> unionBits(wordNum, 0);
    for (size_t opIndex : opIndices) {
        const auto& prefix = prefixCtx.opPrefix[opIndex];
        for (size_t w = 0; w < wordNum; w++) {
            unionBits[w] |= prefix[w];
        }
    }
    return unionBits;
}

// Rule A: prefixes are equal (both mutual subsets). For different prefixes, every task in either
// difference set must still be connected to the opposite side: one of its ancestor tasks must be
// contained in the opposite prefix, or it shares a cube-task prefix with one of the opposite
// side's tasks. A difference task with an empty own prefix is an independent parallel root and
// never provides a safe fusion anchor.
bool VFFusionClusterIdentify::IsPrefixSetCompatible(const PrefixContext& prefixCtx,
                                                    const std::vector<uint64_t>& prefixA,
                                                    const std::vector<uint64_t>& prefixB) const
{
    const bool aSubB = IsPrefixSubsetOf(prefixA, prefixB);
    const bool bSubA = IsPrefixSubsetOf(prefixB, prefixA);
    if (aSubB && bSubA) {
        return true; // Rule A: equal prefixes.
    }
    auto hasAnyBit = [](const std::vector<uint64_t>& bits) {
        return std::any_of(bits.begin(), bits.end(), [](uint64_t word) { return word != 0; });
    };
    auto hasCommonBit = [](const std::vector<uint64_t>& lhs, const std::vector<uint64_t>& rhs) {
        const size_t wordNum = std::min(lhs.size(), rhs.size());
        for (size_t wordIndex = 0; wordIndex < wordNum; wordIndex++) {
            if ((lhs[wordIndex] & rhs[wordIndex]) != 0) {
                return true;
            }
        }
        return false;
    };

    const size_t taskNum = prefixCtx.taskMembers.size();
    // Precomputed once per side: the union of the own prefixes of all cube tasks on that side.
    // Intersecting a difference task's own prefix with this closure is semantically identical to
    // intersecting it with every opposite task's own prefix one by one, and keeps the difference
    // loop O(T * wordNum) instead of O(T^2 * wordNum).
    auto buildSideAncestorClosure = [&](const std::vector<uint64_t>& prefix) {
        std::vector<uint64_t> closure(prefix.size(), 0);
        for (size_t taskIndex = 0; taskIndex < taskNum; taskIndex++) {
            if (!IsPrefixBitSet(prefix, taskIndex)) {
                continue;
            }
            const auto& taskOwnPrefix = prefixCtx.taskPrefix[taskIndex];
            const size_t wordNum = std::min(closure.size(), taskOwnPrefix.size());
            for (size_t wordIndex = 0; wordIndex < wordNum; wordIndex++) {
                closure[wordIndex] |= taskOwnPrefix[wordIndex];
            }
        }
        return closure;
    };
    const auto closureA = buildSideAncestorClosure(prefixA);
    const auto closureB = buildSideAncestorClosure(prefixB);
    auto isDifferenceTaskAnchored = [&](size_t taskIndex, const std::vector<uint64_t>& oppositePrefix,
                                        const std::vector<uint64_t>& oppositeClosure) {
        const auto& taskOwnPrefix = prefixCtx.taskPrefix[taskIndex];
        if (!hasAnyBit(taskOwnPrefix)) {
            return false;
        }
        // The difference task is anchored when at least one of its ancestor tasks is already
        // present on the opposite side. The remaining ancestors may belong to another branch;
        // requiring the complete own prefix to be a subset is unnecessarily strict for a fork
        // that still has a cross-side dependency.
        if (hasCommonBit(taskOwnPrefix, oppositePrefix)) {
            return true;
        }
        // It may also be a sibling extension: even when its complete ancestor set is not present
        // on the opposite side, sharing an ancestor prefix with one of the opposite cube tasks
        // means the two sides are rooted in the same dependency chain.
        return hasCommonBit(taskOwnPrefix, oppositeClosure);
    };

    for (size_t taskIndex = 0; taskIndex < taskNum; taskIndex++) {
        const bool inA = IsPrefixBitSet(prefixA, taskIndex);
        const bool inB = IsPrefixBitSet(prefixB, taskIndex);
        if (inA == inB) {
            continue;
        }
        if (!isDifferenceTaskAnchored(taskIndex, inA ? prefixB : prefixA, inA ? closureB : closureA)) {
            const auto& taskOwnPrefix = prefixCtx.taskPrefix[taskIndex];
            if (!hasAnyBit(taskOwnPrefix)) {
                APASS_LOG_DEBUG_F(Elements::Operation,
                                  "Cannot fuse prefix sets: difference task %zu has an empty own prefix "
                                  "(independent parallel root).",
                                  taskIndex);
            } else {
                APASS_LOG_DEBUG_F(Elements::Operation,
                                  "Cannot fuse prefix sets: difference task %zu prefix %s has no "
                                  "ancestor or shared prefix on the opposite side.",
                                  taskIndex, PrefixToString(taskOwnPrefix).c_str());
            }
            return false;
        }
    }
    return true;
}

// Compare the two sides as wholes: collect each side's union of cube-task prefixes, then apply
// the prefix compatibility rules once on the unions. Bail out immediately on the first unsatisfied condition.
bool VFFusionClusterIdentify::PrefixCompatibleForMerge(const PrefixContext& prefixCtx, const std::vector<size_t>& sideA,
                                                       const std::vector<size_t>& sideB) const
{
    for (size_t opIndex : sideA) {
        if (opIndex >= prefixCtx.opPrefix.size()) {
            return false;
        }
    }
    for (size_t opIndex : sideB) {
        if (opIndex >= prefixCtx.opPrefix.size()) {
            return false;
        }
    }
    const auto prefixA = UnionOpPrefixes(prefixCtx, sideA);
    const auto prefixB = UnionOpPrefixes(prefixCtx, sideB);
    return IsPrefixSetCompatible(prefixCtx, prefixA, prefixB);
}

bool VFFusionClusterIdentify::CanMergeToCluster(const GraphContext& graph, size_t consumerIndex, int clusterId,
                                                const std::unordered_map<int, std::vector<Operation*>>& clusters,
                                                const AncestorBits& ancestorBits, const PrefixContext& prefixCtx) const
{
    auto clusterIter = clusters.find(clusterId);
    auto* consumer = graph.ops[consumerIndex];
    if (clusterIter == clusters.end()) {
        APASS_LOG_WARN_F(Elements::Operation, "Cannot merge %s[%d] into VF cluster %d: cluster is missing.",
                         consumer->GetOpcodeStr().c_str(), consumer->GetOpMagic(), clusterId);
        return false;
    }
    if (!IsFusableOp(*consumer)) {
        return false;
    }
    if (IsUserScopedOp(*consumer)) {
        return false;
    }
    if (clusterIter->second.size() + 1 > VF_CLUSTER_SIZE_LIMIT) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Cannot merge %s[%d] into VF cluster %d: size %zu exceeds limit %zu.",
                          consumer->GetOpcodeStr().c_str(), consumer->GetOpMagic(), clusterId,
                          clusterIter->second.size() + 1, VF_CLUSTER_SIZE_LIMIT);
        return false;
    }
    auto clusterOpIndices = GetClusterOpIndices(graph, clusterIter->second);
    if (!PrefixCompatibleForMerge(prefixCtx, {consumerIndex}, clusterOpIndices)) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Cannot merge %s[%d] into VF cluster %d: cube-task prefix misaligned.",
                          consumer->GetOpcodeStr().c_str(), consumer->GetOpMagic(), clusterId);
        return false;
    }
    if (!IsFusionResourceWithinLimits(graph, clusterOpIndices, {consumerIndex})) {
        return false;
    }
    if (!HasDirectDependencyFromCluster(graph, consumerIndex, clusterIter->second)) {
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "Cannot merge %s[%d] into VF cluster %d: no direct dependency from cluster.",
                          consumer->GetOpcodeStr().c_str(), consumer->GetOpMagic(), clusterId);
        return false;
    }
    for (size_t producerIndex : graph.producers[consumerIndex]) {
        if (std::find(clusterOpIndices.begin(), clusterOpIndices.end(), producerIndex) != clusterOpIndices.end() &&
            OpcodeManager::Inst().GetOpCalcType(graph.ops[producerIndex]->GetOpcode()) == OpCalcType::REDUCE) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Cannot merge %s[%d] into VF cluster %d: producer is reduce op.",
                              consumer->GetOpcodeStr().c_str(), consumer->GetOpMagic(), clusterId);
            return false;
        }
    }
    std::unordered_set<size_t> candidateOpIndices{consumerIndex};
    candidateOpIndices.insert(clusterOpIndices.begin(), clusterOpIndices.end());
    if (!IsAdjustableAdjacent(graph, candidateOpIndices, ancestorBits)) {
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "Cannot merge %s[%d] into VF cluster %d: candidates cannot be compacted.",
                          consumer->GetOpcodeStr().c_str(), consumer->GetOpMagic(), clusterId);
        return false;
    }
    return true;
}

bool VFFusionClusterIdentify::CanMergeClusterIntoTarget(
    const GraphContext& graph, size_t consumerIndex, int targetClusterId, int inputClusterId,
    const std::unordered_map<int, std::vector<Operation*>>& clusters, const AncestorBits& ancestorBits,
    const PrefixContext& prefixCtx) const
{
    if (targetClusterId == inputClusterId) {
        return true;
    }

    auto targetIter = clusters.find(targetClusterId);
    auto inputIter = clusters.find(inputClusterId);
    if (targetIter == clusters.end() || inputIter == clusters.end()) {
        APASS_LOG_WARN_F(Elements::Operation,
                         "Cannot merge input VF cluster %d into target %d for consumer %s[%d]: "
                         "targetExists=%d, inputExists=%d.",
                         inputClusterId, targetClusterId, graph.ops[consumerIndex]->GetOpcodeStr().c_str(),
                         graph.ops[consumerIndex]->GetOpMagic(), targetIter != clusters.end(),
                         inputIter != clusters.end());
        return false;
    }
    if (targetIter->second.size() + inputIter->second.size() > VF_CLUSTER_SIZE_LIMIT) {
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "Cannot merge input VF cluster %d into target %d for consumer %s[%d]: size %zu exceeds "
                          "limit %zu.",
                          inputClusterId, targetClusterId, graph.ops[consumerIndex]->GetOpcodeStr().c_str(),
                          graph.ops[consumerIndex]->GetOpMagic(), targetIter->second.size() + inputIter->second.size(),
                          VF_CLUSTER_SIZE_LIMIT);
        return false;
    }

    auto targetOpIndices = GetClusterOpIndices(graph, targetIter->second);
    auto inputOpIndices = GetClusterOpIndices(graph, inputIter->second);
    if (!PrefixCompatibleForMerge(prefixCtx, targetOpIndices, inputOpIndices)) {
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "Cannot merge input VF cluster %d into target %d for consumer %s[%d]: cube-task prefix "
                          "misaligned.",
                          inputClusterId, targetClusterId, graph.ops[consumerIndex]->GetOpcodeStr().c_str(),
                          graph.ops[consumerIndex]->GetOpMagic());
        return false;
    }
    if (!IsFusionResourceWithinLimits(graph, targetOpIndices, inputOpIndices)) {
        return false;
    }
    for (size_t producerIndex : graph.producers[consumerIndex]) {
        if (std::find(targetOpIndices.begin(), targetOpIndices.end(), producerIndex) != targetOpIndices.end() &&
            OpcodeManager::Inst().GetOpCalcType(graph.ops[producerIndex]->GetOpcode()) == OpCalcType::REDUCE) {
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "Cannot merge input VF cluster %d into target %d for consumer %s[%d]: "
                              "producer is reduce op.",
                              inputClusterId, targetClusterId, graph.ops[consumerIndex]->GetOpcodeStr().c_str(),
                              graph.ops[consumerIndex]->GetOpMagic());
            return false;
        }
    }

    std::unordered_set<size_t> candidateOpIndices{consumerIndex};
    candidateOpIndices.insert(targetOpIndices.begin(), targetOpIndices.end());
    candidateOpIndices.insert(inputOpIndices.begin(), inputOpIndices.end());
    if (!IsAdjustableAdjacent(graph, candidateOpIndices, ancestorBits)) {
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "Cannot merge input VF cluster %d into target %d for consumer %s[%d]: candidates cannot "
                          "be compacted.",
                          inputClusterId, targetClusterId, graph.ops[consumerIndex]->GetOpcodeStr().c_str(),
                          graph.ops[consumerIndex]->GetOpMagic());
        return false;
    }
    return true;
}

bool VFFusionClusterIdentify::IsAdjustableAdjacent(const GraphContext& graph,
                                                   const std::unordered_set<size_t>& candidateOpIndices,
                                                   const AncestorBits& ancestorBits) const
{
    if (candidateOpIndices.empty() || scheduleOrder_.empty()) {
        return false;
    }

    ScheduleWindow window;
    return TryBuildCompactedScheduleWindow(graph, candidateOpIndices, ancestorBits, window);
}

bool VFFusionClusterIdentify::ApplyValidatedScheduleWindow(const ScheduleWindow& window)
{
    if (window.firstPosition > window.lastPosition || window.lastPosition >= scheduleOrder_.size() ||
        window.ops.size() != window.lastPosition - window.firstPosition + 1) {
        APASS_LOG_WARN_F(Elements::Graph,
                         "Apply schedule window failed: first=%zu, last=%zu, windowOps=%zu, scheduleSize=%zu.",
                         window.firstPosition, window.lastPosition, window.ops.size(), scheduleOrder_.size());
        return false;
    }
    for (size_t i = 0; i < window.ops.size(); i++) {
        scheduleOrder_[window.firstPosition + i] = window.ops[i];
    }
    if (!RefreshScheduleIndex(window.firstPosition, window.lastPosition)) {
        return false;
    }
    return true;
}

bool VFFusionClusterIdentify::CompactScheduleForCluster(const GraphContext& graph,
                                                        const std::vector<Operation*>& clusterOps,
                                                        const AncestorBits& ancestorBits)
{
    std::unordered_set<size_t> candidateOpIndices;
    candidateOpIndices.reserve(clusterOps.size());
    for (auto* op : clusterOps) {
        auto iter = graph.opToIndex.find(op);
        if (iter == graph.opToIndex.end()) {
            APASS_LOG_WARN_F(Elements::Operation, "Compact VF cluster schedule failed: op %s[%d] missing from graph.",
                             op->GetOpcodeStr().c_str(), op->GetOpMagic());
            return false;
        }
        candidateOpIndices.insert(iter->second);
    }
    if (candidateOpIndices.empty()) {
        APASS_LOG_WARN_F(Elements::Operation, "Compact VF cluster schedule failed: clusterOps is empty.");
        return false;
    }

    ScheduleWindow window;
    if (!TryBuildCompactedScheduleWindow(graph, candidateOpIndices, ancestorBits, window)) {
        return false;
    }
    if (!ApplyValidatedScheduleWindow(window)) {
        return false;
    }
    return true;
}

bool VFFusionClusterIdentify::CanMergeMultiInputClusters(
    const GraphContext& graph, size_t consumerIndex, const std::vector<int>& inputClusterIds,
    const std::unordered_map<int, std::vector<Operation*>>& clusters, const AncestorBits& ancestorBits,
    const PrefixContext& prefixCtx, std::vector<int>& mergeableClusterIds) const
{
    mergeableClusterIds.clear();
    for (int clusterId : inputClusterIds) {
        if (CanMergeToCluster(graph, consumerIndex, clusterId, clusters, ancestorBits, prefixCtx)) {
            mergeableClusterIds.emplace_back(clusterId);
        }
    }
    if (mergeableClusterIds.size() != inputClusterIds.size()) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Partial VF merge for %s[%d]: inputClusters=%zu, mergeableClusters=%zu.",
                          graph.ops[consumerIndex]->GetOpcodeStr().c_str(), graph.ops[consumerIndex]->GetOpMagic(),
                          inputClusterIds.size(), mergeableClusterIds.size());
    }
    return !mergeableClusterIds.empty();
}

bool VFFusionClusterIdentify::CanMergeConsumerWithInputClusters(
    const GraphContext& graph, size_t consumerIndex, const std::vector<int>& inputClusterIds,
    const std::unordered_map<int, std::vector<Operation*>>& clusters, const AncestorBits& ancestorBits,
    const PrefixContext& prefixCtx, std::vector<int>& mergeableClusterIds) const
{
    mergeableClusterIds.clear();
    if (inputClusterIds.size() == 1) {
        if (!CanMergeToCluster(graph, consumerIndex, inputClusterIds.front(), clusters, ancestorBits, prefixCtx)) {
            return false;
        }
        mergeableClusterIds.emplace_back(inputClusterIds.front());
        return true;
    }
    return CanMergeMultiInputClusters(graph, consumerIndex, inputClusterIds, clusters, ancestorBits, prefixCtx,
                                      mergeableClusterIds);
}

VFFusionClusterIdentify::GraphContext VFFusionClusterIdentify::BuildGraph(Function& function) const
{
    GraphContext graph;
    graph.ops = function.Operations(false).DuplicatedOpList();
    graph.producers.resize(graph.ops.size());
    graph.consumers.resize(graph.ops.size());
    for (size_t i = 0; i < graph.ops.size(); i++) {
        graph.opToIndex[graph.ops[i]] = i;
    }

    auto addDependency = [&graph](size_t producerIndex, size_t consumerIndex) {
        if (producerIndex >= graph.ops.size() || consumerIndex >= graph.ops.size() || producerIndex == consumerIndex) {
            return;
        }
        auto& producers = graph.producers[consumerIndex];
        if (std::find(producers.begin(), producers.end(), producerIndex) == producers.end()) {
            producers.emplace_back(producerIndex);
            graph.consumers[producerIndex].emplace_back(consumerIndex);
        }
    };

    auto addProducerDependency = [&graph, &addDependency](size_t consumerIndex, Operation* producer) {
        auto iter = graph.opToIndex.find(producer);
        if (iter == graph.opToIndex.end()) {
            return;
        }
        addDependency(iter->second, consumerIndex);
    };

    for (size_t i = 0; i < graph.ops.size(); i++) {
        for (auto* producer : graph.ops[i]->ProducerOpsOrdered()) {
            addProducerDependency(i, producer);
        }
    }

    for (size_t i = 0; i < graph.ops.size(); i++) {
        SortOpIndicesByMagic(graph, graph.producers[i]);
        SortOpIndicesByMagic(graph, graph.consumers[i]);
    }
    APASS_LOG_DEBUG_F(Elements::Graph, "Built VF graph for function[%s]: ops=%zu.", function.GetMagicName().c_str(),
                      graph.ops.size());
    return graph;
}

std::vector<Operation*> VFFusionClusterIdentify::BuildScheduledList(const GraphContext& graph) const
{
    std::vector<ScheduleGroup> groups;
    std::unordered_map<Operation*, size_t> opToGroup;
    size_t clusterGroupNum = 0;
    if (!BuildScheduleGroups(graph, groups, opToGroup, clusterGroupNum)) {
        return {};
    }

    std::vector<std::unordered_set<size_t>> groupConsumers(groups.size());
    std::vector<size_t> inDegree(groups.size(), 0);
    if (!BuildScheduleGroupGraph(graph, opToGroup, groupConsumers, inDegree)) {
        return {};
    }
    auto scheduledOps = EmitScheduleGroups(groups, groupConsumers, std::move(inDegree));
    if (scheduledOps.size() != scheduleOrder_.size()) {
        return {};
    }
    APASS_LOG_DEBUG_F(Elements::Graph,
                      "Built scheduled list with VF cluster groups: groups=%zu, clusterGroups=%zu, scheduledOps=%zu.",
                      groups.size(), clusterGroupNum, scheduledOps.size());
    return scheduledOps;
}

bool VFFusionClusterIdentify::BuildScheduleGroups(const GraphContext& graph, std::vector<ScheduleGroup>& groups,
                                                  std::unordered_map<Operation*, size_t>& opToGroup,
                                                  size_t& clusterGroupNum) const
{
    std::unordered_map<int, size_t> clusterToGroup;
    for (size_t scheduleIndex = 0; scheduleIndex < scheduleOrder_.size(); scheduleIndex++) {
        auto* op = scheduleOrder_[scheduleIndex];
        if (graph.opToIndex.count(op) == 0) {
            APASS_LOG_WARN_F(Elements::Graph, "Build scheduled list failed: scheduled op %s[%d] missing from graph.",
                             op->GetOpcodeStr().c_str(), op->GetOpMagic());
            return false;
        }
        size_t groupIndex = groups.size();
        if (IsVfFusionCluster(*op)) {
            int clusterId = op->GetAtomicScopeId();
            auto clusterIter = clusterToGroup.find(clusterId);
            if (clusterIter == clusterToGroup.end()) {
                clusterToGroup[clusterId] = groupIndex;
                groups.push_back({{}, scheduleIndex});
            } else {
                groupIndex = clusterIter->second;
            }
        } else {
            groups.push_back({{}, scheduleIndex});
        }
        groups[groupIndex].ops.emplace_back(op);
        opToGroup[op] = groupIndex;
    }
    clusterGroupNum = clusterToGroup.size();
    return true;
}

bool VFFusionClusterIdentify::BuildScheduleGroupGraph(const GraphContext& graph,
                                                      const std::unordered_map<Operation*, size_t>& opToGroup,
                                                      std::vector<std::unordered_set<size_t>>& groupConsumers,
                                                      std::vector<size_t>& inDegree) const
{
    for (size_t consumerIndex = 0; consumerIndex < graph.ops.size(); consumerIndex++) {
        auto consumerGroupIter = opToGroup.find(graph.ops[consumerIndex]);
        if (consumerGroupIter == opToGroup.end()) {
            APASS_LOG_WARN_F(Elements::Graph, "Build scheduled list failed: consumer %s[%d] has no schedule group.",
                             graph.ops[consumerIndex]->GetOpcodeStr().c_str(), graph.ops[consumerIndex]->GetOpMagic());
            return false;
        }
        size_t consumerGroup = consumerGroupIter->second;
        for (size_t producerIndex : graph.producers[consumerIndex]) {
            auto producerGroupIter = opToGroup.find(graph.ops[producerIndex]);
            if (producerGroupIter == opToGroup.end()) {
                APASS_LOG_WARN_F(Elements::Graph, "Build scheduled list failed: producer %s[%d] has no schedule group.",
                                 graph.ops[producerIndex]->GetOpcodeStr().c_str(),
                                 graph.ops[producerIndex]->GetOpMagic());
                return false;
            }
            size_t producerGroup = producerGroupIter->second;
            if (producerGroup == consumerGroup) {
                continue;
            }
            if (groupConsumers[producerGroup].insert(consumerGroup).second) {
                inDegree[consumerGroup]++;
            }
        }
    }
    return true;
}

std::vector<Operation*> VFFusionClusterIdentify::EmitScheduleGroups(
    const std::vector<ScheduleGroup>& groups, const std::vector<std::unordered_set<size_t>>& groupConsumers,
    std::vector<size_t> inDegree) const
{
    std::priority_queue<ReadyNode, std::vector<ReadyNode>, ReadyNodeCompare> readyNodes;
    for (size_t groupIndex = 0; groupIndex < groups.size(); groupIndex++) {
        if (inDegree[groupIndex] == 0) {
            readyNodes.push({groupIndex, groups[groupIndex].orderIndex});
        }
    }

    std::vector<Operation*> scheduledOps;
    scheduledOps.reserve(scheduleOrder_.size());
    size_t emittedGroupNum = 0;
    while (!readyNodes.empty()) {
        size_t groupIndex = readyNodes.top().opIndex;
        readyNodes.pop();
        emittedGroupNum++;
        scheduledOps.insert(scheduledOps.end(), groups[groupIndex].ops.begin(), groups[groupIndex].ops.end());
        for (size_t consumerGroup : groupConsumers[groupIndex]) {
            if (inDegree[consumerGroup] == 0) {
                APASS_LOG_WARN_F(Elements::Graph, "Build scheduled list failed: group %zu indegree underflow.",
                                 consumerGroup);
                return {};
            }
            inDegree[consumerGroup]--;
            if (inDegree[consumerGroup] == 0) {
                readyNodes.push({consumerGroup, groups[consumerGroup].orderIndex});
            }
        }
    }
    if (emittedGroupNum != groups.size()) {
        APASS_LOG_WARN_F(Elements::Graph, "Build scheduled list failed: emittedGroups=%zu, totalGroups=%zu.",
                         emittedGroupNum, groups.size());
        return {};
    }
    return scheduledOps;
}

bool VFFusionClusterIdentify::InitializeLeafSchedule(const GraphContext& graph, std::vector<size_t>& topoOrder,
                                                     AncestorBits& ancestorBits)
{
    if (!BuildDfsPriorityTopoOrder(graph, topoOrder)) {
        return false;
    }
    ancestorBits.Build(graph.producers, topoOrder);
    scheduleOrder_.reserve(topoOrder.size());
    for (size_t opIndex : topoOrder) {
        scheduleOrder_.emplace_back(graph.ops[opIndex]);
    }
    RefreshScheduleIndex();
    return true;
}

Status VFFusionClusterIdentify::ProcessFusableOps(const GraphContext& graph, const std::vector<size_t>& topoOrder,
                                                  const AncestorBits& ancestorBits,
                                                  std::unordered_map<int, std::vector<Operation*>>& clusters,
                                                  int& nextClusterId)
{
    PrefixContext prefixCtx;
    BuildPrefixContext(graph, topoOrder, prefixCtx);
    for (size_t opIndex : topoOrder) {
        auto* op = graph.ops[opIndex];
        APASS_LOG_DEBUG_F(Elements::Operation, "Begin process op:%s[%d].", op->GetOpcodeStr().c_str(),
                          op->GetOpMagic());
        if (!IsFusableOp(*op)) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Skip op:%s[%d], reason=not_fusable.", op->GetOpcodeStr().c_str(),
                              op->GetOpMagic());
            continue;
        }
        if (IsUserScopedOp(*op)) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Skip op:%s[%d], reason=user_scoped, atomicScopeId=%d.",
                              op->GetOpcodeStr().c_str(), op->GetOpMagic(), op->GetAtomicScopeId());
            continue;
        }
        std::vector<int> inputClusterIds = GetInputClusterIds(graph, opIndex);
        std::vector<int> mergeableClusterIds;
        if (!inputClusterIds.empty() &&
            CanMergeConsumerWithInputClusters(graph, opIndex, inputClusterIds, clusters, ancestorBits, prefixCtx,
                                              mergeableClusterIds)) {
            if (MergeConsumerIntoClusters(graph, opIndex, mergeableClusterIds, ancestorBits, prefixCtx, clusters) !=
                SUCCESS) {
                return FAILED;
            }
            continue;
        }
        if (!AssignClusterToOp(*op, clusters, nextClusterId)) {
            APASS_LOG_ERROR_F(Elements::Operation, "VFFusionClusterIdentify failed to assign VF cluster for %s[%d].",
                              op->GetOpcodeStr().c_str(), op->GetOpMagic());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status VFFusionClusterIdentify::MergeConsumerIntoClusters(const GraphContext& graph, size_t opIndex,
                                                          const std::vector<int>& mergeableClusterIds,
                                                          const AncestorBits& ancestorBits,
                                                          const PrefixContext& prefixCtx,
                                                          std::unordered_map<int, std::vector<Operation*>>& clusters)
{
    auto* op = graph.ops[opIndex];
    int targetClusterId = mergeableClusterIds.front();
    op->SetAtomicScopeId(targetClusterId);
    SetVfFusionCluster(*op, true);
    clusters[targetClusterId].emplace_back(op);
    if (!CompactScheduleForCluster(graph, clusters[targetClusterId], ancestorBits)) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "VFFusionClusterIdentify failed to compact cluster %d after adding %s[%d].", targetClusterId,
                          op->GetOpcodeStr().c_str(), op->GetOpMagic());
        return FAILED;
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "Merged consumer %s[%d] into VF cluster %d, clusterSize=%zu.",
                      op->GetOpcodeStr().c_str(), op->GetOpMagic(), targetClusterId, clusters[targetClusterId].size());
    for (size_t i = 1; i < mergeableClusterIds.size(); i++) {
        int inputClusterId = mergeableClusterIds[i];
        if (!CanMergeClusterIntoTarget(graph, opIndex, targetClusterId, inputClusterId, clusters, ancestorBits,
                                       prefixCtx)) {
            continue;
        }
        if (!MergeClusters(targetClusterId, inputClusterId, clusters)) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "VFFusionClusterIdentify failed to merge VF cluster %d into %d for %s[%d].",
                              inputClusterId, targetClusterId, op->GetOpcodeStr().c_str(), op->GetOpMagic());
            return FAILED;
        }
        if (!CompactScheduleForCluster(graph, clusters[targetClusterId], ancestorBits)) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "VFFusionClusterIdentify failed to compact merged VF cluster %d for %s[%d].",
                              targetClusterId, op->GetOpcodeStr().c_str(), op->GetOpMagic());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status VFFusionClusterIdentify::ProcessLeafFunction(Function& function, int& nextClusterId)
{
    scheduleOrder_.clear();
    schedulePosition_.clear();
    APASS_LOG_DEBUG_F(Elements::Function, "Start VFFusionClusterIdentify for leaf function[%s].",
                      function.GetMagicName().c_str());
    GraphContext graph = BuildGraph(function);
    if (graph.ops.empty()) {
        APASS_LOG_DEBUG_F(Elements::Function, "Skip VFFusionClusterIdentify for empty leaf function[%s].",
                          function.GetMagicName().c_str());
        return SUCCESS;
    }
    ResetGeneratedClusterIds(graph);
    vfFusionClusterOps_.clear();

    std::vector<size_t> topoOrder;
    AncestorBits ancestorBits;
    if (!InitializeLeafSchedule(graph, topoOrder, ancestorBits)) {
        return FAILED;
    }

    std::unordered_map<int, std::vector<Operation*>> clusters;
    if (ProcessFusableOps(graph, topoOrder, ancestorBits, clusters, nextClusterId) != SUCCESS) {
        return FAILED;
    }

    DissolveSingletonClusters(clusters);
    auto scheduledOps = BuildScheduledList(graph);
    if (!VerifyScheduleTopology(graph, scheduledOps)) {
        return FAILED;
    }

    function.ScheduleBy(scheduledOps, true);
    DumpVfClusters(clusters, scheduledOps, function);
    return SUCCESS;
}

void VFFusionClusterIdentify::DumpVfClusters(const std::unordered_map<int, std::vector<Operation*>>& clusters,
                                             const std::vector<Operation*>& scheduledOps,
                                             const Function& function) const
{
    size_t clusterOpNum = 0;
    for (const auto& cluster : clusters) {
        clusterOpNum += cluster.second.size();
    }
    if (IsPassDebugEnabled()) {
        int lastClusterId = -1;
        for (auto* op : scheduledOps) {
            if (!IsVfFusionCluster(*op)) {
                continue;
            }
            int clusterId = op->GetAtomicScopeId();
            if (clusterId != lastClusterId) {
                lastClusterId = clusterId;
                auto clusterIter = clusters.find(clusterId);
                size_t opNum = clusterIter != clusters.end() ? clusterIter->second.size() : 0;
                APASS_LOG_DEBUG_F(Elements::Function, "VF cluster (id=%d) in leaf function[%s]: opNum=%zu.", clusterId,
                                  function.GetMagicName().c_str(), opNum);
            }
            std::string operands = "inputs[";
            for (const auto& input : op->GetIOperands()) {
                if (input == nullptr) {
                    continue;
                }
                if (operands.back() != '[') {
                    operands += ", ";
                }
                operands += "magic=" + std::to_string(input->GetMagic()) +
                            ", shape=" + CommonUtils::ContainerToStr(input->GetShape());
            }
            operands += "], outputs[";
            for (const auto& output : op->GetOOperands()) {
                if (output == nullptr) {
                    continue;
                }
                if (operands.back() != '[') {
                    operands += ", ";
                }
                operands += "magic=" + std::to_string(output->GetMagic()) +
                            ", shape=" + CommonUtils::ContainerToStr(output->GetShape());
            }
            operands += "]";
            APASS_LOG_DEBUG_F(Elements::Operation, "op %s[%d]: %s", op->GetOpcodeStr().c_str(), op->GetOpMagic(),
                              operands.c_str());
        }
    }
    APASS_LOG_INFO_F(Elements::Function,
                     "Finish VFFusionClusterIdentify for leaf function[%s]: clusters=%zu, clusteredOps=%zu, "
                     "scheduledOps=%zu.",
                     function.GetMagicName().c_str(), clusters.size(), clusterOpNum, scheduledOps.size());
}

Status VFFusionClusterIdentify::RunOnFunction(Function& function)
{
    if (!config::GetPassGlobalConfig(KEY_ENABLE_VF, false)) {
        APASS_LOG_DEBUG_F(Elements::Function, "VFFusionClusterIdentify is skipped for ENABLE_VF is false.");
        return SUCCESS;
    }
    if (function.paramConfigs_.oooSchedMode != "HLF") {
        APASS_LOG_DEBUG_F(Elements::Function, "VFFusionClusterIdentify is skipped for ooo_sched_mode[%s] is not HLF.",
                          function.paramConfigs_.oooSchedMode.c_str());
        return SUCCESS;
    }
    const auto arch = Platform::Instance().GetSoc().GetNPUArch();
    if (arch != NPUArch::DAV_3510 && !IsLiteNPU(arch)) {
        APASS_LOG_DEBUG_F(Elements::Function, "VFFusionClusterIdentify is skipped for unsupported architecture.");
        return SUCCESS;
    }
    if (function.rootFunc_ == nullptr) {
        APASS_LOG_ERROR_F(Elements::Function, "VFFusionClusterIdentify failed for root function is null.");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Function, "Start VFFusionClusterIdentify for root function[%s]: leafFunctions=%zu.",
                     function.rootFunc_->GetMagicName().c_str(), function.rootFunc_->programs_.size());
    int nextClusterId = VF_CLUSTER_ID_START;
    for (auto& program : function.rootFunc_->programs_) {
        if (ProcessLeafFunction(*program.second, nextClusterId) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "VFFusionClusterIdentify failed for leaf function[%s].",
                              program.second->GetMagicName().c_str());
            return FAILED;
        }
    }
    APASS_LOG_INFO_F(Elements::Function, "Finish VFFusionClusterIdentify for root function[%s].",
                     function.rootFunc_->GetMagicName().c_str());
    return SUCCESS;
}

} // namespace npu::tile_fwk
