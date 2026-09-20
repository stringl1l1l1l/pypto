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
 * \file vf_fusion_cluster_identify.h
 * \brief Identify VF fusion clusters.
 */

#ifndef VF_FUSION_CLUSTER_IDENTIFY_H
#define VF_FUSION_CLUSTER_IDENTIFY_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "interface/function/function.h"
#include "passes/pass_interface/pass.h"
#include "passes/pass_utils/pass_common_defs.h"

namespace npu::tile_fwk {

constexpr size_t VF_CLUSTER_SIZE_LIMIT = 32;
constexpr size_t VF_CLUSTER_EXTERNAL_UB_MEMORY_RATIO_NUMERATOR = 3;
constexpr size_t VF_CLUSTER_EXTERNAL_UB_MEMORY_RATIO_DENOMINATOR = 5;
constexpr size_t PERCENT_SCALE = 100;
// Ensure VF cluster ID space never overlaps with user atomic_scope encoding.
static_assert(VF_CLUSTER_ID_START > 10000 * 10000 + 9999,
              "VF_CLUSTER_ID_START must exceed max user atomic_scope encoding");

class AncestorBits {
public:
    void Build(const std::vector<std::vector<size_t>>& producers, const std::vector<size_t>& topoOrder);
    bool IsAncestor(size_t descendantIndex, size_t ancestorIndex) const;
    size_t Size() const { return bits_.size(); }

private:
    static constexpr size_t BITSET_WORD_BITS = 64;

    static void SetBit(std::vector<uint64_t>& bits, size_t bitIndex);

    std::vector<std::vector<uint64_t>> bits_;
};

class VFFusionClusterIdentify : public Pass {
public:
    VFFusionClusterIdentify() : Pass("VFFusionClusterIdentify")
    {
        SetSupportedArches({NPUArch::DAV_3510, NPUArch::DAV_3003, NPUArch::DAV_3113});
    }
    ~VFFusionClusterIdentify() override = default;

    Status RunOnFunction(Function& function) override;

private:
    friend class VFFusionClusterIdentifyTestAccessor;

    // Graph topology context: op list + producer/consumer adjacency tables.
    struct GraphContext {
        std::vector<Operation*> ops;
        std::unordered_map<Operation*, size_t> opToIndex;
        std::vector<std::vector<size_t>> producers;
        std::vector<std::vector<size_t>> consumers;
    };

    // Ready node for priority queue scheduling: opIndex + topological orderIndex.
    struct ReadyNode {
        size_t opIndex;
        size_t orderIndex;
    };

    struct ReadyNodeCompare {
        bool operator()(const ReadyNode& lhs, const ReadyNode& rhs) const { return lhs.orderIndex > rhs.orderIndex; }
    };

    // Schedule block: a contiguous range of ops with the same clusterId in scheduleOrder_.
    struct ScheduleBlock {
        std::vector<size_t> opIndices;
        std::vector<Operation*> ops;
    };

    // Schedule group: ops merged for group-level topological sort, ordered by orderIndex.
    struct ScheduleGroup {
        std::vector<Operation*> ops;
        size_t orderIndex;
    };

    // Replaces a contiguous range in scheduleOrder_. Ops before firstPosition and after lastPosition keep both
    // their relative order and their cached positions, so only this window needs validation and index refresh.
    struct ScheduleWindow {
        size_t firstPosition{0};
        size_t lastPosition{0};
        std::vector<Operation*> ops;
    };

    // Cube-task prefix context for prefix-based fusion rules: cube ops are chain-merged into cube
    // tasks, each op gets a bitset of transitively-depended cube tasks, and each cube task gets
    // its own dependency bitset. Two vector sides may fuse when their cube-task prefixes align
    // (Rule A: equal; Rule B: subset + cross-side anchor; Rule C: cross-side anchor/shared prefix).
    struct PrefixContext {
        std::vector<int> cubeTaskOfOp;                       // opIndex -> cube task id (-1 if op is not cube)
        std::vector<std::vector<size_t>> taskMembers;        // cube task id -> member op indices
        std::vector<std::vector<std::string>> taskMatMulOps; // cube task id -> MATMUL op labels
        std::vector<std::vector<uint64_t>> opPrefix;         // opIndex -> cube-task prefix bitset
        std::vector<std::vector<uint64_t>> taskPrefix;       // cube task id -> its own cube-task prefix bitset
    };

    Status ProcessLeafFunction(Function& function, int& nextClusterId);
    GraphContext BuildGraph(Function& function) const;
    bool BuildDfsPriorityTopoOrder(const GraphContext& graph, std::vector<size_t>& topoOrder) const;
    void SortOpIndicesByMagic(const GraphContext& graph, std::vector<size_t>& opIndices) const;
    bool ReleaseConsumers(const GraphContext& graph, size_t producerIndex, std::vector<size_t>& remainingProducers,
                          std::vector<size_t>& releasedConsumers) const;
    std::vector<Operation*> BuildScheduledList(const GraphContext& graph) const;
    bool BuildScheduleGroups(const GraphContext& graph, std::vector<ScheduleGroup>& groups,
                             std::unordered_map<Operation*, size_t>& opToGroup, size_t& clusterGroupNum) const;
    bool BuildScheduleGroupGraph(const GraphContext& graph, const std::unordered_map<Operation*, size_t>& opToGroup,
                                 std::vector<std::unordered_set<size_t>>& groupConsumers,
                                 std::vector<size_t>& inDegree) const;
    std::vector<Operation*> EmitScheduleGroups(const std::vector<ScheduleGroup>& groups,
                                               const std::vector<std::unordered_set<size_t>>& groupConsumers,
                                               std::vector<size_t> inDegree) const;
    void RefreshScheduleIndex();
    bool RefreshScheduleIndex(size_t firstPosition, size_t lastPosition);

    // Prefix context construction: cube-task clustering (chain merge, mirroring the cube side of
    // TaskSplitter::UnionSameCoreOps, L1 special-case included; singleton groups form no task) +
    // prefix bitset propagation in topological order.
    bool IsCubeOp(const Operation& op) const;
    void BuildCubeTasks(const GraphContext& graph, std::vector<int>& cubeTaskOfOp,
                        std::vector<std::vector<size_t>>& taskMembers,
                        std::vector<std::vector<std::string>>& taskMatMulOps) const;
    void ComputeOpPrefixBits(const GraphContext& graph, const std::vector<size_t>& topoOrder,
                             const std::vector<int>& cubeTaskOfOp, size_t cubeTaskNum,
                             std::vector<std::vector<uint64_t>>& opPrefix) const;
    void ComputeTaskPrefixBits(const std::vector<std::vector<size_t>>& taskMembers,
                               const std::vector<std::vector<uint64_t>>& opPrefix, size_t cubeTaskNum,
                               std::vector<std::vector<uint64_t>>& taskPrefix) const;
    void BuildPrefixContext(const GraphContext& graph, const std::vector<size_t>& topoOrder,
                            PrefixContext& prefixCtx) const;

    // Prefix gate added on top of the legacy merge checks: each merging side is reduced to the
    // union of all its members' cube-task prefixes and the two unions are compared as a whole
    // (Rule A: equal; Rule B: subset with cross-side anchor; Rule C: cross-side anchor/shared
    // prefix). Comparing member-by-member is wrong for cluster merges: members with
    // complementary prefix subsets of the same cluster-level prefix would be rejected pairwise
    // even though the sides as wholes align.
    std::vector<uint64_t> UnionOpPrefixes(const PrefixContext& prefixCtx, const std::vector<size_t>& opIndices) const;
    bool IsPrefixSetCompatible(const PrefixContext& prefixCtx, const std::vector<uint64_t>& prefixA,
                               const std::vector<uint64_t>& prefixB) const;
    bool PrefixCompatibleForMerge(const PrefixContext& prefixCtx, const std::vector<size_t>& sideA,
                                  const std::vector<size_t>& sideB) const;

    std::vector<int> GetInputClusterIds(const GraphContext& graph, size_t opIndex) const;
    bool CanMergeConsumerWithInputClusters(const GraphContext& graph, size_t consumerIndex,
                                           const std::vector<int>& inputClusterIds,
                                           const std::unordered_map<int, std::vector<Operation*>>& clusters,
                                           const AncestorBits& ancestorBits, const PrefixContext& prefixCtx,
                                           std::vector<int>& mergeableClusterIds) const;
    bool CanMergeMultiInputClusters(const GraphContext& graph, size_t consumerIndex,
                                    const std::vector<int>& inputClusterIds,
                                    const std::unordered_map<int, std::vector<Operation*>>& clusters,
                                    const AncestorBits& ancestorBits, const PrefixContext& prefixCtx,
                                    std::vector<int>& mergeableClusterIds) const;
    bool CanMergeToCluster(const GraphContext& graph, size_t consumerIndex, int clusterId,
                           const std::unordered_map<int, std::vector<Operation*>>& clusters,
                           const AncestorBits& ancestorBits, const PrefixContext& prefixCtx) const;
    bool CanMergeClusterIntoTarget(const GraphContext& graph, size_t consumerIndex, int targetClusterId,
                                   int inputClusterId, const std::unordered_map<int, std::vector<Operation*>>& clusters,
                                   const AncestorBits& ancestorBits, const PrefixContext& prefixCtx) const;
    bool IsAdjustableAdjacent(const GraphContext& graph, const std::unordered_set<size_t>& candidateOpIndices,
                              const AncestorBits& ancestorBits) const;

    bool VerifyScheduleTopology(const GraphContext& graph, const std::vector<Operation*>& schedule) const;
    bool ValidateScheduleWindowOps(const ScheduleWindow& window) const;
    bool ValidateScheduleWindow(const GraphContext& graph, const ScheduleWindow& window) const;
    bool TryBuildCompactedScheduleWindow(const GraphContext& graph,
                                         const std::unordered_set<size_t>& candidateOpIndices,
                                         const AncestorBits& ancestorBits, ScheduleWindow& window) const;
    bool CanPlaceBlockBeforeCandidates(const ScheduleBlock& block, const std::unordered_set<size_t>& candidateOpIndices,
                                       const AncestorBits& ancestorBits) const;
    bool CanPlaceBlockAfterCandidates(const ScheduleBlock& block, const std::unordered_set<size_t>& candidateOpIndices,
                                      const AncestorBits& ancestorBits) const;
    bool VerifyVfClusterContiguity(const std::vector<Operation*>& schedule) const;
    bool ApplyValidatedScheduleWindow(const ScheduleWindow& window);
    bool CompactScheduleForCluster(const GraphContext& graph, const std::vector<Operation*>& clusterOps,
                                   const AncestorBits& ancestorBits);
    bool GetScheduleWindowRange(const GraphContext& graph, const std::unordered_set<size_t>& candidateOpIndices,
                                size_t& firstPosition, size_t& lastPosition) const;
    bool BuildScheduleBlock(const GraphContext& graph, size_t& scheduleIndex, size_t lastPosition,
                            ScheduleBlock& block) const;
    bool ClassifyScheduleBlock(const ScheduleBlock& block, const std::unordered_set<size_t>& candidateOpIndices,
                               const AncestorBits& ancestorBits, std::vector<ScheduleBlock>& beforeBlocks,
                               std::vector<ScheduleBlock>& afterBlocks) const;
    void AssembleScheduleWindow(const std::vector<ScheduleBlock>& beforeBlocks,
                                const std::vector<Operation*>& candidateOps,
                                const std::vector<ScheduleBlock>& afterBlocks, ScheduleWindow& window) const;
    bool HasDirectDependencyFromCluster(const GraphContext& graph, size_t consumerIndex,
                                        const std::vector<Operation*>& clusterOps) const;
    std::vector<size_t> GetClusterOpIndices(const GraphContext& graph, const std::vector<Operation*>& clusterOps) const;
    size_t CalculateClusterExternalMemory(const GraphContext& graph,
                                          const std::unordered_set<size_t>& clusterOpIndices) const;
    bool IsFusionResourceWithinLimits(const GraphContext& graph, const std::vector<size_t>& existingOpIndices,
                                      const std::vector<size_t>& addedOpIndices) const;
    bool AssignClusterToOp(Operation& op, std::unordered_map<int, std::vector<Operation*>>& clusters,
                           int& nextClusterId);
    bool InitializeLeafSchedule(const GraphContext& graph, std::vector<size_t>& topoOrder, AncestorBits& ancestorBits);
    Status ProcessFusableOps(const GraphContext& graph, const std::vector<size_t>& topoOrder,
                             const AncestorBits& ancestorBits,
                             std::unordered_map<int, std::vector<Operation*>>& clusters, int& nextClusterId);
    Status MergeConsumerIntoClusters(const GraphContext& graph, size_t opIndex,
                                     const std::vector<int>& mergeableClusterIds, const AncestorBits& ancestorBits,
                                     const PrefixContext& prefixCtx,
                                     std::unordered_map<int, std::vector<Operation*>>& clusters);
    bool MergeClusters(int targetClusterId, int inputClusterId,
                       std::unordered_map<int, std::vector<Operation*>>& clusters);
    void DissolveSingletonClusters(std::unordered_map<int, std::vector<Operation*>>& clusters);
    // DFX: log every VF cluster (op opcode/magic, tensor shape + magic) and the Finish summary.
    // Op details follow the scheduled topological order; cluster ids are unique across leaf
    // functions and serve directly as the dump index. Cluster/op details are DEBUG-only; only the
    // Finish summary is INFO.
    void DumpVfClusters(const std::unordered_map<int, std::vector<Operation*>>& clusters,
                        const std::vector<Operation*>& scheduledOps, const Function& function) const;
    void ResetGeneratedClusterIds(const GraphContext& graph);

    bool IsFusableOp(const Operation& op) const;
    bool IsUserScopedOp(const Operation& op) const;
    bool IsVfFusionCluster(const Operation& op) const;
    void SetVfFusionCluster(Operation& op, bool isVfFusionCluster);

    std::vector<Operation*> scheduleOrder_;
    std::unordered_map<Operation*, size_t> schedulePosition_;
    std::unordered_set<const Operation*> vfFusionClusterOps_;
};

} // namespace npu::tile_fwk

#endif // VF_FUSION_CLUSTER_IDENTIFY_H
