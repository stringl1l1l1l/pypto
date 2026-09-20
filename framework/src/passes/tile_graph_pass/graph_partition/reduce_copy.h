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
 * \file reduce_copy.h
 * \brief
 */

#ifndef PASS_REDUCE_COPY_H_
#define PASS_REDUCE_COPY_H_

#include "passes/pass_interface/pass.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "passes/pass_utils/pass_utils.h"
#include "interface/tensor/logical_tensor.h"
#include <vector>
#include <set>
#include <unordered_set>
#include <queue>
#include <utility>

namespace npu::tile_fwk {
struct BoundaryTensorInfo {
    int tensorMagic;
    std::vector<int> producerSubgraphs;
    std::vector<int> consumerSubgraphs;
    bool isDDR{false};                  // 目的: 仅 DDR tensor 参与 inner-external-use 检查
    std::vector<int> producerCvFuseIds; // 与 producerSubgraphs 逐项平行, 端点级 CvFuseId, 未分配为 -1
    std::vector<int> consumerCvFuseIds; // 与 consumerSubgraphs 逐项平行
};

struct MergeInput {
    int numSubgraph{0};
    int maxLatency{0};
    int maxSubgraphAICOpNum{0}; // 合并后子图 AIC op 数上限, 0=不启用(自定义档仅约束总 op 数)
    int maxSubgraphAIVOpNum{0}; // 合并后子图 AIV op 数上限, 0=不启用(自定义档仅约束总 op 数)
    // 自定义档(auto_mix_partition>100)的总 op 数上限: 合并后子图 AIC+AIV op 总数不超过该值,
    // 此时 AIC/AIV 不再各自受限; 0=不启用(0/1/2 档位走 AIC/AIV 独立上限)
    int maxSubgraphTotalOpNum{0};
    std::pair<double, double> aivRatio;
    std::vector<int> subgraphAICLatency;
    std::vector<int> subgraphAIVLatency;
    std::vector<int> subgraphAICOpNum;
    std::vector<int> subgraphAIVOpNum;
    std::vector<std::set<int>> subGraphInGraph;
    std::vector<std::set<int>> subGraphOutGraph;
    std::vector<std::vector<int>> mergeGroup;
    std::vector<bool> isEnforceMergeGroup;
    std::vector<bool> isValidMergeGroup;
    bool hasScopedOp{false}; // 图中存在 cvFuseId>=0 的 op(仅 CV 混合 scope 分配 cvFuseId, 纯 scope 保持 -1)
    std::vector<BoundaryTensorInfo> boundaryTensors;
    std::vector<std::vector<int>> subgraphToBoundaryTensorIds;
    // 每个 sg 所属的 feedback loop-carry 通路集合(空=不在任何环路上), 通路按"环"归类:
    // 共享子图的多条 slot 链属同一环(如 online softmax 的 out/sum/max), 同环内可互相合并。
    // 环路上的 sg 只允许与"通路归属完全相同"的 sg 合并, 防止通路外分支被吞进 carry 链。
    // 空向量 = 前端无 slot scope / 无 feedback slot, 门控不启用。
    std::vector<std::set<int>> subgraphLoopPaths;
};

struct MergeOutput {
    int numSubgraphUpdated;
    std::vector<int> subgraphIdUpdated;
};

class MixGraphMerger {
public:
    MixGraphMerger() = default;
    ~MixGraphMerger() = default;
    MergeOutput Merge(const MergeInput& input);
    bool enableAutoMix{true};

private:
    MergeInput mInput;
    MergeOutput mOutput;
    std::vector<int> mParent;
    std::vector<int> mRank;
    std::vector<std::vector<int>> mRootToBoundaryTensorIds;
    std::vector<int> mTensorVisitStamp;
    int mVisitStamp{0};
    std::unordered_set<int> mGlobalOutputSinks; // 出度0子图: 不作为任何 boundary tensor producer, 即最终输出端点
    std::unordered_set<int> mWarnedInnerTensorMagics; // 已输出过 WARN 的 tensor, 跨 merge loop 迭代去重防打屏
    // 每个 root 的 loop-carry 通路集合(成员原始集合的并集), 供同通路门控使用
    std::vector<std::set<int>> mRootLoopPaths;
    // cached merged graph (avoid redundant rebuild in CanMergeWithoutCycle)
    std::vector<std::set<int>> mCachedOutGraph;
    std::vector<std::set<int>> mCachedInGraph;
    // HasCycle scratch buffers (avoid per-call reallocation)
    std::vector<int> mInDegreeBuf;
    std::vector<bool> mIsRootBuf;
    std::queue<int> mQueueBuf;

    void Initialize(const MergeInput& input);
    void InitBoundaryTensorIndex();
    int FindParent(int x);
    void UnionSets(int x, int y);
    bool CanMergeWithoutCycle(const std::vector<int>& actualGroup);
    void WarnIfEnforceOpNumExceeds(const std::vector<int>& actualGroup);
    bool CanMergeWithConstraints(const std::vector<int>& actualGroup, bool allowSinkMerge);
    void PerformMerge(const std::vector<int>& actualGroup);
    void UpdateBoundaryTensorIndex(const std::vector<int>& actualGroup);
    void ApplyMergeToGraph(const std::vector<int>& actualGroup);
    void MergeNodesInCachedGraph(int root, const std::set<int>& del);
    void UpdateOutput();
    bool CheckLatencyConstraint(const std::vector<int>& actualGroup);
    bool CheckLoopPathConsistency(const std::vector<int>& actualGroup);
    bool CheckMergeBenefitByStructuralPattern(const std::vector<int>& actualGroup, bool allowSinkMerge);
    // 判断汇聚 sink 的合并是否无收益需拒绝(串行损失模型): 串行化损失 Σbranch − max(branch)
    // 超过 kSerialLossRatio × 合入root 总 latency 时拒绝; 调用方保证入边数 >= 2
    bool IsSinkMergeUnbeneficial(int root, const std::set<int>& incomingRoots);
    bool CheckNoExternalUseOfMergedInnerTensor(const std::vector<int>& actualGroup, bool checkByCvFuseId = false);
    bool IsInvalidMergedInnerTensor(int tensorId, const std::unordered_set<int>& mergedRoots, std::vector<int>& prodIn,
                                    std::vector<int>& prodOut, std::vector<int>& consIn, std::vector<int>& consOut);
    bool IsInvalidMergedInnerTensorByCvFuseId(int tensorId, const std::unordered_set<int>& mergedRoots,
                                              std::vector<int>& prodIn, std::vector<int>& prodOut,
                                              std::vector<int>& consIn, std::vector<int>& consOut);
    std::vector<int> GetActualGroup(const std::vector<int>& group);
    void BuildMergedGraph(std::vector<std::set<int>>& outGraph, std::vector<std::set<int>>& inGraph);
    bool HasCycle(const std::vector<std::set<int>>& outGraph, const std::vector<std::set<int>>& inGraph);
};

class ReduceCopyMerge : public Pass {
public:
    ReduceCopyMerge() : Pass("ReduceCopyMerge") { SetSupportedArches({NPUArch::DAV_3510}); }
    ~ReduceCopyMerge() override = default;
    // 将 auto_mix_partition 配置映射为合图 op 数上限, 返回是否开启自动 CV Mix 合图:
    // 0=关闭(上限仅作 enforce 路径 WARN 观测阈值); 1=high 档(旧值兼容, 行为同旧版);
    // 2=default 档(推荐收紧值); >100=自定义总 op 数上限; 3~100=非法自定义值, 回退 default 档
    static bool MapAutoMixPartitionToLimits(int autoMixPartition, int& maxSubgraphAICOpNum, int& maxSubgraphAIVOpNum,
                                            int& maxSubgraphTotalOpNum);

private:
    Status BuildGraph(Function& function, MergeInput& mergeInput);
    Status BuildMergeGroup(Function& function, MergeInput& mergeInput);
    void CombineForkSubgraph(Function& function, MergeInput& mergeInput);
    Status MarkNoMergeSubgraph(Function& function, MergeInput& mergeInput);
    void MarkFeedbackSubgraphs(Function& function, MergeInput& mergeInput);
    void UpdateConnectRecord(Function& function, MergeInput& mergeInput);
    void UpdateBoundaryTensorSize(LogicalTensorPtr& tensor, int tensorSize);
    void RecordBoundaryTensorInfo(LogicalTensorPtr& tensor, MergeInput& mergeInput, const std::set<int>& connectGraphs);
    void UpdateMergeInput(MergeInput& mergeInput, std::multimap<int, std::vector<int>>& sortedMergeGroup);
    bool IsEnforceMergeBoundary(LogicalTensorPtr& tensor);
    Status RunOnFunction(Function& function) override;
    Status PostCheck(Function& function) override;
    std::unordered_map<int, std::vector<int>> subgraphToOutputTensors;
    std::unordered_map<int, std::vector<int>> subgraphToInputTensors;
    std::unordered_map<int, std::vector<int>> tensorToMergeGroup;
    std::unordered_set<int> noMergeSubgraph;
    std::unordered_set<int> noMergeSubgraphEnforce;
    std::map<std::vector<int>, int> mergeGroupToPriority;
    std::set<std::vector<int>> enforceMergeGroup;
    std::vector<int> subgraphInputSize;
    std::vector<int> subgraphOutputSize;
};

} // namespace npu::tile_fwk
#endif
