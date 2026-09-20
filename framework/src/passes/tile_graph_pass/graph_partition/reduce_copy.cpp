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
 * \file reduce_copy.cpp
 * \brief
 */

#include "passes/tile_graph_pass/graph_partition/reduce_copy.h"
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/graph_utils.h"
#include "interface/utils/common.h"
#include <vector>
#include <numeric>
#include <algorithm>
#include <map>
#include <string>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <tuple>

#define MODULE_NAME "ReduceCopy"

namespace npu::tile_fwk {

// 自动合并护栏(实测依据: progress1/dump_graphs 四 kernel Before dump, 2026-09-04):
// 单次合并组的总工作量(AIC+AIV)上限。候选组超过该规模意味着串行 carry 脊柱与并行分支
// 已被融进同一巨型 kernel(mla sg143: 1.24M cycles, 4858-op, e2e +244%/+327%), 拒绝。
// 取值依据: 必须放行 gqa 全融合组 213,934 cycles(合并到 1 最优, 1163us)与
// mha_grad 101,220 cycles; 拒绝 mla 巨型组与 sparse 跨拷贝组(>250k)。
constexpr int kAutoMixMaxMergeLatency = 250000;

// auto_mix_partition 档位编码: 0=关闭自动 CV Mix 合图(上限仅作 enforce/CV scope 路径的 WARN 观测阈值);
// 1=high 档(旧值兼容: 旧版本 =1 开启即此上限, 行为保持不变); 2=default 档(推荐收紧值);
// >100=自定义档, 约束合并后子图总 op 数(AIC+AIV)不超过配置值, 不再分别限制 AIC/AIV;
// 3~100 为非法自定义值, 回退 default 档并打 WARN(对外文档仅暴露 'off'/'default'/'high' 与 N>100)
constexpr int kAutoMixPartitionDefaultLevel = 2;
constexpr int kAutoMixCustomMinOpNum = 100;
// default 档独立上限: 自历史默认 2000/2240 收紧, 控制合并后 kernel 规模与编译时间
constexpr int kAutoMixDefaultMaxAICOpNum = 1700;
constexpr int kAutoMixDefaultMaxAIVOpNum = 400;
// high 档独立上限: 沿用历史默认值
constexpr int kAutoMixHighMaxAICOpNum = 2000;
constexpr int kAutoMixHighMaxAIVOpNum = 2240;

// 串行损失阈值: 汇聚 sink 合并的串行化损失(Σbranch − max)超过合入 root 总 latency 的
// 该倍数时, 为省边界通信付出的代价远超 root 可消化的规模, 拒绝合并(经验调优值)。
constexpr int64_t kSerialLossRatio = 8;

// 目的: 日志可读性——上限 0 表示不启用, 打印为 unlimited 而非裸 0, 避免与有效上限数值混淆造成误解
static std::string LimitToStr(int limit) { return limit > 0 ? std::to_string(limit) : "unlimited"; }

static bool IsValidMergeGroup(const std::vector<int>& mergeGroup, const std::unordered_set<int>& noMergeSubgraph)
{
    for (int subidx : mergeGroup) {
        if (noMergeSubgraph.count(subidx) > 0) {
            return false;
        }
    }
    return true;
}

bool ReduceCopyMerge::MapAutoMixPartitionToLimits(int autoMixPartition, int& maxSubgraphAICOpNum,
                                                  int& maxSubgraphAIVOpNum, int& maxSubgraphTotalOpNum)
{
    maxSubgraphTotalOpNum = 0;
    if (autoMixPartition > kAutoMixCustomMinOpNum) {
        // 自定义档: 仅约束合并后子图总 op 数(AIC+AIV), AIC/AIV 上限置 0 表示不启用
        maxSubgraphAICOpNum = 0;
        maxSubgraphAIVOpNum = 0;
        maxSubgraphTotalOpNum = autoMixPartition;
    } else if (autoMixPartition > kAutoMixPartitionDefaultLevel) {
        // 拦截 3~100 的非法自定义值: 回退 default 档, WARN 观测不阻断编译
        APASS_LOG_WARN_F(Elements::Operation,
                         "Invalid auto_mix_partition=%d: custom total op limit must be > %d, fallback to default "
                         "level.",
                         autoMixPartition, kAutoMixCustomMinOpNum);
        maxSubgraphAICOpNum = kAutoMixDefaultMaxAICOpNum;
        maxSubgraphAIVOpNum = kAutoMixDefaultMaxAIVOpNum;
    } else if (autoMixPartition == kAutoMixPartitionDefaultLevel) {
        maxSubgraphAICOpNum = kAutoMixDefaultMaxAICOpNum;
        maxSubgraphAIVOpNum = kAutoMixDefaultMaxAIVOpNum;
    } else {
        // 0(关闭)与 1(旧值兼容的 high 档)共用 high 上限: 关闭时上限仅作 enforce 路径的 WARN 观测阈值,
        // 1 保持旧版开启行为不变
        maxSubgraphAICOpNum = kAutoMixHighMaxAICOpNum;
        maxSubgraphAIVOpNum = kAutoMixHighMaxAIVOpNum;
    }
    return autoMixPartition != 0;
}

Status ReduceCopyMerge::RunOnFunction(Function& function)
{
    APASS_LOG_DEBUG_F(Elements::Operation,
                      "***************************** ReduceCopy Pass start, graph: %s *****************************",
                      function.GetMagicName().c_str());
    if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510) {
        APASS_LOG_INFO_F(Elements::Operation, "Platform not support CV mix graph, skip ReduceCopy Pass.");
        return SUCCESS;
    }
    size_t subgraphNumBefore = function.GetTotalSubGraphCount();
    MergeInput mergeInput;
    mergeInput.maxLatency = kAutoMixMaxMergeLatency;
    int autoMixPartition = function.paramConfigs_.autoMixPartition;
    bool enableAutoMix = MapAutoMixPartitionToLimits(autoMixPartition, mergeInput.maxSubgraphAICOpNum,
                                                     mergeInput.maxSubgraphAIVOpNum, mergeInput.maxSubgraphTotalOpNum);
    mergeInput.aivRatio = {1e-6, 1e6};
    APASS_LOG_DEBUG_F(
        Elements::Operation,
        "Merge limits: graph=%s auto_mix_partition=%d, maxSubgraphAICOpNum=%s, maxSubgraphAIVOpNum=%s, "
        "maxSubgraphTotalOpNum=%s.",
        function.GetMagicName().c_str(), autoMixPartition, LimitToStr(mergeInput.maxSubgraphAICOpNum).c_str(),
        LimitToStr(mergeInput.maxSubgraphAIVOpNum).c_str(), LimitToStr(mergeInput.maxSubgraphTotalOpNum).c_str());
    APASS_LOG_INFO_F(Elements::Operation, "Subgraph Info before ReduceCopy Pass:");
    BuildGraph(function, mergeInput);
    MarkNoMergeSubgraph(function, mergeInput);
    BuildMergeGroup(function, mergeInput);
    CombineForkSubgraph(function, mergeInput);
    MixGraphMerger merger;
    APASS_LOG_INFO_F(Elements::Operation, "Enable auto CV mix partition: %s", enableAutoMix ? "True" : "False");
    merger.enableAutoMix = enableAutoMix;
    MergeOutput output = merger.Merge(mergeInput);
    function.SetTotalSubGraphCount(output.numSubgraphUpdated);
    for (auto& op : function.Operations()) {
        int src = op.GetSubgraphID();
        if (src >= static_cast<int>(output.subgraphIdUpdated.size())) {
            APASS_LOG_ERROR_F(Elements::Operation, "Current op subgraphID not in ReduceCopy subgraph update record.");
            return FAILED;
        }
        if (op.HasAttr(OpAttributeKey::isCube) && (op.GetBoolAttribute(OpAttributeKey::isCube) == false)) {
            op.SetAttr(OpAttributeKey::reduceCopyPreSubgraphId, static_cast<int64_t>(src));
        }
        op.UpdateSubgraphID(output.subgraphIdUpdated[src]);
    }
    MergeInput mergeInputTmp;
    APASS_LOG_DEBUG_F(Elements::Operation, "Subgraph Info after ReduceCopy Pass:");
    BuildGraph(function, mergeInputTmp);
    // 存在 CV 混合 scope 时 auto-mix 被有意跳过, 不属于"性能评估后不合并"的场景, 不告警
    if (enableAutoMix && !mergeInput.hasScopedOp && function.GetTotalSubGraphCount() == subgraphNumBefore) {
        APASS_LOG_WARN_F(Elements::Operation, "CV mix merging was not performed, since fusion would degrade "
                                              "performance or may cause loop on this computation graph.");
    }
    APASS_LOG_INFO_F(Elements::Operation, "ReduceCopy result: graph=%s autoMix=%s before=%zu after=%zu",
                     function.GetMagicName().c_str(), enableAutoMix ? "True" : "False", subgraphNumBefore,
                     function.GetTotalSubGraphCount());
    return SUCCESS;
}

void ReduceCopyMerge::CombineForkSubgraph(Function& function, MergeInput& mergeInput)
{
    std::unordered_map<int, std::unordered_set<int>> forkSubgraphsMap;
    for (auto& op : function.Operations()) {
        std::unordered_set<int> forkSubgraphIds;
        if (IsAssembleLike(op.GetOpcode())) {
            for (auto& prod : op.ProducerOps()) {
                forkSubgraphIds.insert(prod->GetSubgraphID());
            }
        }
        if (forkSubgraphIds.size() <= 1) {
            continue;
        }
        for (auto& idx : forkSubgraphIds) {
            for (auto& idy : forkSubgraphIds) {
                if (idx != idy) {
                    forkSubgraphsMap[idx].insert(idy);
                }
            }
        }
    }
    for (int i = 0; i < static_cast<int>(mergeInput.mergeGroup.size()); i++) {
        size_t originalSize = mergeInput.mergeGroup[i].size();
        std::set<int> combinedGroup(mergeInput.mergeGroup[i].begin(), mergeInput.mergeGroup[i].end());
        for (auto& subgraphId : mergeInput.mergeGroup[i]) {
            if (forkSubgraphsMap.count(subgraphId) > 0) {
                combinedGroup.insert(forkSubgraphsMap[subgraphId].begin(), forkSubgraphsMap[subgraphId].end());
            }
        }
        if (combinedGroup.size() != originalSize) {
            mergeInput.mergeGroup[i].clear();
            mergeInput.mergeGroup[i].insert(mergeInput.mergeGroup[i].end(), combinedGroup.begin(), combinedGroup.end());
            mergeInput.isValidMergeGroup[i] = IsValidMergeGroup(mergeInput.mergeGroup[i], noMergeSubgraph);
            APASS_LOG_DEBUG_F(Elements::Operation, "merge group %d after combine fork: %s, isEnforce: %s, isValid: %s.",
                              i, IntVecToStr(mergeInput.mergeGroup[i]).c_str(),
                              mergeInput.isEnforceMergeGroup[i] ? "True" : "False",
                              mergeInput.isValidMergeGroup[i] ? "True" : "False");
        }
    }
}

static void MarkCrossSubgraph(Function& function, std::unordered_set<int>& noMergeSubgraph)
{
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_RESHAPE) {
            continue;
        }
        int src = op.GetSubgraphID();
        bool hasSameConsumerId = false;
        bool hasOtherConsumerId = false;
        for (auto& consumerOp : op.ConsumerOps()) {
            if (consumerOp->GetSubgraphID() == src) {
                hasSameConsumerId = true;
            } else {
                hasOtherConsumerId = true;
            }
        }
        if (hasSameConsumerId && hasOtherConsumerId) {
            noMergeSubgraph.insert(src);
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "Subgraph %d is not mergeable because it has inner tensor used by other subgraph.", src);
        }
        bool hasSameProducerId = false;
        bool hasOtherProducerId = false;
        for (auto& producerOp : op.ProducerOps()) {
            if (producerOp->GetSubgraphID() == src) {
                hasSameProducerId = true;
            } else {
                hasOtherProducerId = true;
            }
        }
        if (hasSameProducerId && hasOtherProducerId) {
            noMergeSubgraph.insert(src);
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "Subgraph %d is not mergeable because it has inner tensor used by other subgraph.", src);
        }
    }
}

static void MarkLoopCarryPathSubgraphs(const MergeInput& mergeInput, const std::set<int>& readerSgs,
                                       const std::set<int>& writerSgs, int pathId,
                                       std::vector<std::set<int>>& subgraphLoopPaths)
{
    if (readerSgs.empty() || writerSgs.empty()) {
        return;
    }
    std::vector<bool> forwardReach(mergeInput.numSubgraph, false);
    std::vector<bool> backwardReach(mergeInput.numSubgraph, false);
    std::vector<int> stack;
    for (int sg : readerSgs) {
        forwardReach[sg] = true;
        stack.push_back(sg);
    }
    while (!stack.empty()) {
        int cur = stack.back();
        stack.pop_back();
        for (int next : mergeInput.subGraphOutGraph[cur]) {
            if (!forwardReach[next]) {
                forwardReach[next] = true;
                stack.push_back(next);
            }
        }
    }
    for (int sg : writerSgs) {
        backwardReach[sg] = true;
        stack.push_back(sg);
    }
    while (!stack.empty()) {
        int cur = stack.back();
        stack.pop_back();
        for (int prev : mergeInput.subGraphInGraph[cur]) {
            if (!backwardReach[prev]) {
                backwardReach[prev] = true;
                stack.push_back(prev);
            }
        }
    }
    for (int sg = 0; sg < mergeInput.numSubgraph; ++sg) {
        if (forwardReach[sg] && backwardReach[sg]) {
            subgraphLoopPaths[sg].insert(pathId);
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "Subgraph %d is on feedback loop-carry path of slot %d (merge only within same path).",
                              sg, pathId);
        }
    }
}

// 目的: 一次遍历收集全部 feedback slot 的写端/读端子图集合(写端=产出该 slot outcast
// tensor 的算子所在子图, 读端=消费该 slot incast tensor 的算子所在子图), 按 slot 分发,
// 避免逐 slot 全量扫描; 越界防御收敛到循环条件(槽位表与 IO 表约定平行, 见 RemoveOutcast)
static void CollectFeedbackSlotEndSgs(Function& function, const std::set<int>& feedbackSlots,
                                      std::map<int, std::set<int>>& slotWriters,
                                      std::map<int, std::set<int>>& slotReaders)
{
    auto slotScope = function.GetSlotScope();
    if (slotScope == nullptr) {
        return;
    }
    auto& incastSlots = slotScope->ioslot.incastSlot;
    auto& outcastSlots = slotScope->ioslot.outcastSlot;
    auto& incasts = function.GetIncast();
    auto& outcasts = function.GetOutcast();
    // 槽位表与 IO 表的平行约定由多个生产方维护(BuildIncastOutcastSlot/RemoveOutcast/LoopUnroll/
    // LoadJson), 失配时尾部条目被跳过会导致对应 slot 的通路不标记、门控静默失效, 留 WARN 便于定位
    if (outcasts.size() != outcastSlots.size()) {
        APASS_LOG_WARN_F(Elements::Operation, "outcasts(%zu) vs outcastSlot(%zu) size mismatch, tail entries skipped.",
                         outcasts.size(), outcastSlots.size());
    }
    if (incasts.size() != incastSlots.size()) {
        APASS_LOG_WARN_F(Elements::Operation, "incasts(%zu) vs incastSlot(%zu) size mismatch, tail entries skipped.",
                         incasts.size(), incastSlots.size());
    }
    for (size_t i = 0; i < outcasts.size() && i < outcastSlots.size(); ++i) {
        for (int s : outcastSlots[i]) {
            if (feedbackSlots.count(s) > 0) {
                for (auto& op : outcasts[i]->GetProducers()) {
                    slotWriters[s].insert(op->GetSubgraphID());
                }
            }
        }
    }
    for (size_t i = 0; i < incasts.size() && i < incastSlots.size(); ++i) {
        for (int s : incastSlots[i]) {
            if (feedbackSlots.count(s) > 0) {
                for (auto& op : incasts[i]->GetConsumers()) {
                    slotReaders[s].insert(op->GetSubgraphID());
                }
            }
        }
    }
}

// 目的: 通路归属按"环"而非单条 slot 链归类 —— 同一环的多条 carry 链(如 online softmax 的
// out/sum/max)迭代域相同, 链间合并不改变执行次数语义, 按链归类会误判为异路而阻断融合;
// 判定同环的依据是两条链存在共享子图(某子图同时位于两条链的读写端), 共享子图会将环的
// 迭代耦合, 归一安全; 互相独立的环仍保持不同通路。
// 不变量: 前端 pypto.frontend.jit 的 loop_unroll + 原位赋值版本化保证同一条环上的多条
// carry 链必共享至少一个端点子图(同迭代体内读写三件套的算子落在同一 sg), 因此"共享端点
// 归组"能覆盖所有合法同环形态; 不共享端点的两条 slot 链属于不同迭代环, 归为异路正确。
static void MarkLoopGroupedCarryPaths(const MergeInput& mergeInput, const std::vector<int>& slotList,
                                      const std::vector<std::set<int>>& readers,
                                      const std::vector<std::set<int>>& writers,
                                      std::vector<std::set<int>>& subgraphLoopPaths)
{
    int slotNum = static_cast<int>(slotList.size());
    std::vector<int> parent(slotNum);
    std::iota(parent.begin(), parent.end(), 0);
    auto findRoot = [&parent](int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    std::unordered_map<int, int> sgToSlot;
    for (int i = 0; i < slotNum; ++i) {
        std::set<int> ends = readers[i];
        ends.insert(writers[i].begin(), writers[i].end());
        for (int sg : ends) {
            auto it = sgToSlot.find(sg);
            if (it != sgToSlot.end()) {
                parent[findRoot(it->second)] = findRoot(i);
            } else {
                sgToSlot[sg] = i;
            }
        }
    }
    for (int i = 0; i < slotNum; ++i) {
        if (findRoot(i) != i) {
            continue;
        }
        std::set<int> loopReaders;
        std::set<int> loopWriters;
        for (int j = 0; j < slotNum; ++j) {
            if (findRoot(j) == i) {
                loopReaders.insert(readers[j].begin(), readers[j].end());
                loopWriters.insert(writers[j].begin(), writers[j].end());
            }
        }
        MarkLoopCarryPathSubgraphs(mergeInput, loopReaders, loopWriters, slotList[i], subgraphLoopPaths);
    }
}

Status ReduceCopyMerge::MarkNoMergeSubgraph(Function& function, MergeInput& mergeInput)
{
    noMergeSubgraph.clear();
    noMergeSubgraphEnforce.clear();
    int subgraphNum = function.GetTotalSubGraphCount();
    std::vector<int> subgraphOpNum(subgraphNum, 0);
    std::vector<bool> subgraphHasReshape(subgraphNum, false);
    std::vector<bool> subgraphHasInnerDDR(subgraphNum, false);
    for (auto& op : function.Operations()) {
        int src = op.GetSubgraphID();
        // noMergeSubgraphEnforce
        if (OpcodeManager::Inst().GetCoreType(op.GetOpcode()) == OpCoreType::AICPU) {
            noMergeSubgraphEnforce.insert(src);
            APASS_LOG_DEBUG_F(Elements::Operation, "Subgraph %d is not mergeable because it has AICPU op.", src);
        }
        // noMergeSubgraph
        subgraphOpNum[src] += 1;
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            subgraphHasReshape[src] = true;
        }
        for (auto& operand : op.GetOOperands()) {
            if (operand->GetMemoryTypeToBe() != MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            for (auto& consumer : operand->GetConsumers()) {
                if (consumer->GetSubgraphID() == src) {
                    subgraphHasInnerDDR[src] = true;
                    break;
                }
            }
        }
    }
    MarkCrossSubgraph(function, noMergeSubgraph);
    for (int i = 0; i < subgraphNum; i++) {
        if (subgraphOpNum[i] == 1 && subgraphHasReshape[i] == true) {
            // reshape-only 子图不再一刀切 noMerge: 该类子图为前端 broadcast lower 自动生成的纯形状
            // 适配(asis==tobe 无 memtype 变迁), 真正危险的跨子图 inner-tensor 形态已由 MarkCrossSubgraph
            // 单独标记. 通用规则导致 automix 路径 ~60% 的候选组被静默拒绝(gqa 全融合 69->1 的根因).
            APASS_LOG_DEBUG_F(Elements::Operation, "Subgraph %d is reshape-only, treated as mergeable.", i);
        } else if (subgraphHasInnerDDR[i]) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Subgraph %d has inner DDR tensor.", i);
        }
    }
    MarkFeedbackSubgraphs(function, mergeInput);
    return SUCCESS;
}

void ReduceCopyMerge::MarkFeedbackSubgraphs(Function& function, MergeInput& mergeInput)
{
    auto slotScope = function.GetSlotScope();
    if (slotScope == nullptr) {
        return;
    }
    auto& incastSlots = slotScope->ioslot.incastSlot;
    auto& outcastSlots = slotScope->ioslot.outcastSlot;

    std::set<int> incastSlotSet;
    for (auto& slots : incastSlots) {
        for (int s : slots) {
            incastSlotSet.insert(s);
        }
    }
    std::set<int> outcastSlotSet;
    for (auto& slots : outcastSlots) {
        for (int s : slots) {
            outcastSlotSet.insert(s);
        }
    }

    std::set<int> feedbackSlots;
    std::set_intersection(incastSlotSet.begin(), incastSlotSet.end(), outcastSlotSet.begin(), outcastSlotSet.end(),
                          std::inserter(feedbackSlots, feedbackSlots.begin()));
    if (feedbackSlots.empty()) {
        return;
    }

    mergeInput.subgraphLoopPaths.assign(mergeInput.numSubgraph, std::set<int>());
    std::map<int, std::set<int>> slotWriters;
    std::map<int, std::set<int>> slotReaders;
    CollectFeedbackSlotEndSgs(function, feedbackSlots, slotWriters, slotReaders);
    std::vector<int> slotList(feedbackSlots.begin(), feedbackSlots.end());
    std::vector<std::set<int>> readers(slotList.size());
    std::vector<std::set<int>> writers(slotList.size());
    for (size_t i = 0; i < slotList.size(); ++i) {
        readers[i] = std::move(slotReaders[slotList[i]]);
        writers[i] = std::move(slotWriters[slotList[i]]);
    }
    MarkLoopGroupedCarryPaths(mergeInput, slotList, readers, writers, mergeInput.subgraphLoopPaths);
}

Status ReduceCopyMerge::BuildGraph(Function& function, MergeInput& mergeInput)
{
    int subgraphNum = function.GetTotalSubGraphCount();
    mergeInput.numSubgraph = subgraphNum;
    mergeInput.subgraphAICLatency.clear();
    mergeInput.subgraphAICLatency.resize(subgraphNum, 0);
    mergeInput.subgraphAIVLatency.clear();
    mergeInput.subgraphAIVLatency.resize(subgraphNum, 0);
    mergeInput.subgraphAICOpNum.clear();
    mergeInput.subgraphAICOpNum.resize(subgraphNum, 0);
    mergeInput.subgraphAIVOpNum.clear();
    mergeInput.subgraphAIVOpNum.resize(subgraphNum, 0);
    mergeInput.subGraphOutGraph.clear();
    mergeInput.subGraphOutGraph.resize(subgraphNum);
    mergeInput.subGraphInGraph.clear();
    mergeInput.subGraphInGraph.resize(subgraphNum);
    for (auto& op : function.Operations()) {
        int src = op.GetSubgraphID();
        if (op.GetCvFuseId() >= 0) {
            mergeInput.hasScopedOp = true;
        }
        bool isCube = op.HasAttr(OpAttributeKey::isCube) && op.GetBoolAttribute(OpAttributeKey::isCube);
        int opLatency = op.GetLatency();
        if (isCube) {
            mergeInput.subgraphAICOpNum[src] += 1;
            mergeInput.subgraphAICLatency[src] += opLatency;
        } else {
            mergeInput.subgraphAIVOpNum[src] += 1;
            mergeInput.subgraphAIVLatency[src] += opLatency;
        }
        for (auto& consumer : op.ConsumerOps()) {
            int dst = consumer->GetSubgraphID();
            if (src != dst) {
                mergeInput.subGraphOutGraph[src].insert(dst);
                mergeInput.subGraphInGraph[dst].insert(src);
            }
        }
    }
    for (int i = 0; i < subgraphNum; i++) {
        APASS_LOG_INFO_F(Elements::Operation, "Subgraph %d : AIC Latency %d cycles, AIV Latency %d cycles.", i,
                         mergeInput.subgraphAICLatency[i], mergeInput.subgraphAIVLatency[i]);
    }
    return SUCCESS;
}

bool ReduceCopyMerge::IsEnforceMergeBoundary(LogicalTensorPtr& tensor)
{
    std::unordered_set<int> boundaryScopeIds;
    for (auto& op : tensor->GetProducers()) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Boundary tensor %d has produce %d with scopeInfoCvFuseId %d.",
                          tensor->GetMagic(), op->GetOpMagic(), op->GetCvFuseId());
        if (op->GetCvFuseId() == -1) {
            return false;
        }
        boundaryScopeIds.insert(op->GetCvFuseId());
    }
    for (auto& op : tensor->GetConsumers()) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Boundary tensor %d has consumer %d with scopeInfoCvFuseId %d.",
                          tensor->GetMagic(), op->GetOpMagic(), op->GetCvFuseId());
        if (op->GetCvFuseId() == -1) {
            return false;
        }
        boundaryScopeIds.insert(op->GetCvFuseId());
    }
    if (boundaryScopeIds.size() == 1) {
        return true;
    }
    return false;
}

void ReduceCopyMerge::UpdateBoundaryTensorSize(LogicalTensorPtr& tensor, int tensorSize)
{
    for (auto& op : tensor->GetProducers()) {
        subgraphOutputSize[op->GetSubgraphID()] += tensorSize;
        subgraphToOutputTensors[op->GetSubgraphID()].push_back(tensor->GetMagic());
    }
    for (auto& op : tensor->GetConsumers()) {
        subgraphInputSize[op->GetSubgraphID()] += tensorSize;
        subgraphToInputTensors[op->GetSubgraphID()].push_back(tensor->GetMagic());
    }
}

// 预判非 DDR boundary tensor 是否会被 IntraSubgraphAdapter 在 producer 侧不插新搬运 op 而
// 直接 DDR 化(写路径无 ASSEMBLE 隔离). consumer 侧插不插 VIEW 都不改变 tensor 本体被无条件
// DDR 化的事实, 故判定只看 producer.
static bool WillBeDdrWithoutNewCopyOp(LogicalTensorPtr tensor)
{
    const auto& producers = tensor->GetProducers();
    if (producers.size() > 1) {
        for (const auto& producer : producers) {
            OpCalcType calcType = OpcodeManager::Inst().GetOpCalcType(producer->GetOpcode());
            if (calcType != OpCalcType::MOVE_OUT && calcType != OpCalcType::MOVE_LOCAL) {
                return false;
            }
        }
    } else if (producers.size() == 1) {
        Operation* producer = *(producers.begin());
        if (!IsAssembleLike(producer->GetOpcode()) && producer->GetOpcode() != Opcode::OP_COPY_OUT &&
            !GraphUtils::IsCrossCoreMoveOp(producer)) {
            return false;
        }
    }
    return true;
}

void ReduceCopyMerge::RecordBoundaryTensorInfo(LogicalTensorPtr& tensor, MergeInput& mergeInput,
                                               const std::set<int>& connectGraphs)
{
    BoundaryTensorInfo tensorInfo;
    tensorInfo.tensorMagic = tensor->GetMagic();
    // 目的: 除当前已是 DDR 的 tensor 外, 预判 IntraSubgraphAdapter 中"不插新搬运 op 直接 DDR 化"的
    // boundary tensor 也须参与 inner-external-use 检查(其后必落地 DDR, 且无新 op 隔离)
    tensorInfo.isDDR = tensor->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR ||
                       tensor->GetMemoryTypeToBe() == MemoryType::MEM_DEVICE_DDR || WillBeDdrWithoutNewCopyOp(tensor);
    std::set<std::pair<int, int>> producerEndpoints;
    std::set<std::pair<int, int>> consumerEndpoints;
    for (auto& op : tensor->GetProducers()) {
        producerEndpoints.insert({op->GetSubgraphID(), op->GetCvFuseId()});
    }
    for (auto& op : tensor->GetConsumers()) {
        consumerEndpoints.insert({op->GetSubgraphID(), op->GetCvFuseId()});
    }
    for (const auto& endpoint : producerEndpoints) {
        tensorInfo.producerSubgraphs.push_back(endpoint.first);
        tensorInfo.producerCvFuseIds.push_back(endpoint.second);
    }
    for (const auto& endpoint : consumerEndpoints) {
        tensorInfo.consumerSubgraphs.push_back(endpoint.first);
        tensorInfo.consumerCvFuseIds.push_back(endpoint.second);
    }
    int tensorId = static_cast<int>(mergeInput.boundaryTensors.size());
    mergeInput.boundaryTensors.push_back(tensorInfo);
    for (int subgraphId : connectGraphs) {
        mergeInput.subgraphToBoundaryTensorIds[subgraphId].push_back(tensorId);
    }
}

void ReduceCopyMerge::UpdateConnectRecord(Function& function, MergeInput& mergeInput)
{
    for (auto tensor : GraphUtils::GetAllTensors(function)) {
        int tensorSize = tensor->MemorySize();
        if (tensor->GetProducers().size() == 0 || tensor->GetConsumers().size() == 0) {
            continue;
        }
        std::set<int> connectGraphs;
        for (auto& op : tensor->GetProducers()) {
            connectGraphs.insert(op->GetSubgraphID());
        }
        for (auto& op : tensor->GetConsumers()) {
            connectGraphs.insert(op->GetSubgraphID());
        }
        if (connectGraphs.size() <= 1) {
            continue;
        }
        int tensorMagic = tensor->GetMagic();
        UpdateBoundaryTensorSize(tensor, tensorSize);
        RecordBoundaryTensorInfo(tensor, mergeInput, connectGraphs);
        std::vector<int> mergeGroup(connectGraphs.begin(), connectGraphs.end());
        tensorToMergeGroup[tensorMagic] = mergeGroup;
        APASS_LOG_DEBUG_F(Elements::Operation, "Found boundary tensor %d of subgraphs %s.", tensor->GetMagic(),
                          IntVecToStr(mergeGroup).c_str());
        mergeGroupToPriority[mergeGroup] += tensorSize;
        if (IsEnforceMergeBoundary(tensor)) {
            APASS_LOG_DEBUG_F(Elements::Operation, "----boundary tensor %d is marked as enforced.", tensor->GetMagic());
            enforceMergeGroup.insert(mergeGroup);
        }
    }
}

Status ReduceCopyMerge::BuildMergeGroup(Function& function, MergeInput& mergeInput)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "Build merge group before mix subgraph merge start.");
    subgraphInputSize.resize(mergeInput.numSubgraph, 0);
    subgraphOutputSize.resize(mergeInput.numSubgraph, 0);
    mergeInput.boundaryTensors.clear();
    mergeInput.subgraphToBoundaryTensorIds.clear();
    mergeInput.subgraphToBoundaryTensorIds.resize(mergeInput.numSubgraph);
    UpdateConnectRecord(function, mergeInput);
    std::multimap<int, std::vector<int>> sortedMergeGroup;
    for (auto& pair : mergeGroupToPriority) {
        std::vector<int> boundaryGroup = pair.first;
        std::sort(boundaryGroup.begin(), boundaryGroup.end());
        sortedMergeGroup.insert({pair.second, boundaryGroup});
    }
    for (int subIdx = 0; subIdx < mergeInput.numSubgraph; subIdx++) {
        if (subgraphToOutputTensors.count(subIdx) == 0) {
            continue;
        }
        std::set<int> localGroup;
        for (auto localBoundary : subgraphToOutputTensors[subIdx]) {
            localGroup.insert(tensorToMergeGroup[localBoundary].begin(), tensorToMergeGroup[localBoundary].end());
        }
        if (localGroup.size() > 1) {
            std::vector<int> group(localGroup.begin(), localGroup.end());
            sortedMergeGroup.insert({subgraphOutputSize[subIdx], group});
        }
    }
    for (int subIdx = 0; subIdx < mergeInput.numSubgraph; subIdx++) {
        if (subgraphToInputTensors.count(subIdx) == 0) {
            continue;
        }
        std::set<int> localGroup;
        for (auto localBoundary : subgraphToInputTensors[subIdx]) {
            localGroup.insert(tensorToMergeGroup[localBoundary].begin(), tensorToMergeGroup[localBoundary].end());
        }
        if (localGroup.size() > 1) {
            std::vector<int> group(localGroup.begin(), localGroup.end());
            sortedMergeGroup.insert({subgraphInputSize[subIdx], group});
        }
    }
    UpdateMergeInput(mergeInput, sortedMergeGroup);
    return SUCCESS;
}

void ReduceCopyMerge::UpdateMergeInput(MergeInput& mergeInput, std::multimap<int, std::vector<int>>& sortedMergeGroup)
{
    int groupIdx = 0;
    std::set<std::vector<int>> visitedMergeGroup;
    for (auto it = sortedMergeGroup.rbegin(); it != sortedMergeGroup.rend(); it++) {
        if (visitedMergeGroup.count(it->second) > 0 || !IsValidMergeGroup(it->second, noMergeSubgraphEnforce)) {
            continue;
        }
        visitedMergeGroup.insert(it->second);
        mergeInput.mergeGroup.push_back(it->second);
        bool isEnforce = enforceMergeGroup.count(it->second) > 0 ? true : false;
        mergeInput.isEnforceMergeGroup.push_back(isEnforce);
        mergeInput.isValidMergeGroup.push_back(IsValidMergeGroup(it->second, noMergeSubgraph));
        APASS_LOG_DEBUG_F(Elements::Operation, "merge group %d: %s, isEnforce: %s.", groupIdx,
                          IntVecToStr(it->second).c_str(), isEnforce ? "True" : "False");
        groupIdx++;
    }
}

Status ReduceCopyMerge::PostCheck(Function& function)
{
    (void)function;
    APASS_LOG_INFO_F(Elements::Function, "PostCheck for ReduceCopy.");
    if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510) {
        APASS_LOG_INFO_F(Elements::Operation, "Platform not support CV mix graph, skip PostCheck for ReduceCopy Pass.");
        return SUCCESS;
    }
    return SUCCESS;
}

static bool ValidateInput(const MergeInput& input)
{
    if (input.numSubgraph < 0 || input.maxLatency < 0) {
        return false;
    }
    int n = input.numSubgraph;
    if ((int)input.subgraphAICLatency.size() != n || (int)input.subgraphAIVLatency.size() != n ||
        (int)input.subgraphAICOpNum.size() != n || (int)input.subgraphAIVOpNum.size() != n ||
        (int)input.subGraphOutGraph.size() != n) {
        return false;
    }
    if (input.mergeGroup.size() != input.isEnforceMergeGroup.size()) {
        return false;
    }
    if (!input.subgraphToBoundaryTensorIds.empty() && static_cast<int>(input.subgraphToBoundaryTensorIds.size()) != n) {
        return false;
    }
    if (!input.subgraphLoopPaths.empty() && static_cast<int>(input.subgraphLoopPaths.size()) != n) {
        return false;
    }
    for (int i = 0; i < n; ++i) {
        if (input.subgraphAICLatency[i] < 0 || input.subgraphAIVLatency[i] < 0) {
            return false;
        }
    }
    return true;
}

void MixGraphMerger::Initialize(const MergeInput& input)
{
    mInput = input;
    mWarnedInnerTensorMagics.clear();
    mParent.resize(input.numSubgraph);
    mRank.resize(input.numSubgraph, 0);
    for (int i = 0; i < input.numSubgraph; ++i) {
        mParent[i] = i;
    }
    // root 的 loop-carry 通路集合: 初始为各原始子图自身集合, 合并时取并集
    mRootLoopPaths = input.subgraphLoopPaths;
    mOutput.numSubgraphUpdated = input.numSubgraph;
    mOutput.subgraphIdUpdated.resize(input.numSubgraph);
    for (int i = 0; i < input.numSubgraph; ++i) {
        mOutput.subgraphIdUpdated[i] = i;
    }
    InitBoundaryTensorIndex();
    BuildMergedGraph(mCachedOutGraph, mCachedInGraph);
}

void MixGraphMerger::InitBoundaryTensorIndex()
{
    mRootToBoundaryTensorIds.assign(mInput.numSubgraph, std::vector<int>());
    mTensorVisitStamp.assign(mInput.boundaryTensors.size(), 0);
    mVisitStamp = 0;
    if (mInput.subgraphToBoundaryTensorIds.empty()) {
        return;
    }
    mRootToBoundaryTensorIds = mInput.subgraphToBoundaryTensorIds;
    // 出度0 sink: 不作为任何 boundary tensor 的 producer 的子图 (即最终输出端点)
    std::vector<bool> isProducer(mInput.numSubgraph, false);
    for (const auto& bt : mInput.boundaryTensors) {
        for (int s : bt.producerSubgraphs) {
            if (s >= 0 && s < mInput.numSubgraph) {
                isProducer[s] = true;
            }
        }
    }
    mGlobalOutputSinks.clear();
    for (int i = 0; i < mInput.numSubgraph; ++i) {
        if (!isProducer[i]) {
            mGlobalOutputSinks.insert(i);
        }
    }
}

int MixGraphMerger::FindParent(int x)
{
    if (mParent[x] != x) {
        mParent[x] = FindParent(mParent[x]);
    }
    return mParent[x];
}

void MixGraphMerger::UnionSets(int x, int y)
{
    int px = FindParent(x);
    int py = FindParent(y);
    if (px == py) {
        return;
    }
    if (mRank[px] < mRank[py]) {
        mParent[px] = py;
    } else if (mRank[px] > mRank[py]) {
        mParent[py] = px;
    } else {
        mParent[py] = px;
        mRank[px]++;
    }
}

std::vector<int> MixGraphMerger::GetActualGroup(const std::vector<int>& group)
{
    std::set<int> actualSet;
    for (int idx : group) {
        if (idx >= 0 && idx < mInput.numSubgraph) {
            actualSet.insert(FindParent(idx));
        }
    }
    return std::vector<int>(actualSet.begin(), actualSet.end());
}

void MixGraphMerger::BuildMergedGraph(std::vector<std::set<int>>& outGraph, std::vector<std::set<int>>& inGraph)
{
    int n = mInput.numSubgraph;
    outGraph.assign(n, std::set<int>());
    inGraph.assign(n, std::set<int>());
    for (int i = 0; i < n; ++i) {
        int pi = FindParent(i);
        for (int j : mInput.subGraphOutGraph[i]) {
            int pj = FindParent(j);
            if (pi != pj) {
                outGraph[pi].insert(pj);
                inGraph[pj].insert(pi);
            }
        }
    }
}

bool MixGraphMerger::HasCycle(const std::vector<std::set<int>>& outGraph, const std::vector<std::set<int>>& inGraph)
{
    int n = mInput.numSubgraph;
    mInDegreeBuf.assign(n, 0);
    mIsRootBuf.assign(n, false);
    std::vector<int>& inDegree = mInDegreeBuf;
    std::vector<bool>& isRoot = mIsRootBuf;
    int rootCount = 0;
    for (int i = 0; i < n; ++i) {
        if (FindParent(i) == i) {
            isRoot[i] = true;
            inDegree[i] = inGraph[i].size();
            rootCount++;
        }
    }
    mQueueBuf = {};
    std::queue<int>& q = mQueueBuf;
    int count = 0;
    for (int i = 0; i < n; ++i) {
        if (isRoot[i] && inDegree[i] == 0) {
            q.push(i);
        }
    }
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        count++;
        for (int v : outGraph[u]) {
            if (isRoot[v]) {
                inDegree[v]--;
                if (inDegree[v] == 0) {
                    q.push(v);
                }
            }
        }
    }
    return count != rootCount;
}

// 目的: enforce 合并不受 op 数上限拦截, 超限时仅打 WARN 用于观测编译时间风险, 不改变合并行为
void MixGraphMerger::WarnIfEnforceOpNumExceeds(const std::vector<int>& actualGroup)
{
    int totalAICOpNum = 0;
    int totalAIVOpNum = 0;
    for (int root : actualGroup) {
        for (int i = 0; i < mInput.numSubgraph; ++i) {
            if (FindParent(i) == root) {
                totalAICOpNum += mInput.subgraphAICOpNum[i];
                totalAIVOpNum += mInput.subgraphAIVOpNum[i];
            }
        }
    }
    if (mInput.maxSubgraphAICOpNum > 0 && totalAICOpNum > mInput.maxSubgraphAICOpNum) {
        APASS_LOG_WARN_F(Elements::Operation, "Enforce merged group=%s aicOps=%d exceeds maxSubgraphAICOpNum=%d.",
                         IntVecToStr(actualGroup).c_str(), totalAICOpNum, mInput.maxSubgraphAICOpNum);
    }
    if (mInput.maxSubgraphAIVOpNum > 0 && totalAIVOpNum > mInput.maxSubgraphAIVOpNum) {
        APASS_LOG_WARN_F(Elements::Operation, "Enforce merged group=%s aivOps=%d exceeds maxSubgraphAIVOpNum=%d.",
                         IntVecToStr(actualGroup).c_str(), totalAIVOpNum, mInput.maxSubgraphAIVOpNum);
    }
    int totalOpNum = totalAICOpNum + totalAIVOpNum;
    if (mInput.maxSubgraphTotalOpNum > 0 && totalOpNum > mInput.maxSubgraphTotalOpNum) {
        APASS_LOG_WARN_F(Elements::Operation, "Enforce merged group=%s totalOps=%d exceeds maxSubgraphTotalOpNum=%d.",
                         IntVecToStr(actualGroup).c_str(), totalOpNum, mInput.maxSubgraphTotalOpNum);
    }
}

bool MixGraphMerger::CanMergeWithoutCycle(const std::vector<int>& actualGroup)
{
    if (actualGroup.size() <= 1) {
        return false;
    }
    int root = actualGroup[0];
    std::set<int> del(actualGroup.begin() + 1, actualGroup.end());
    std::unordered_set<int> affectedNodes;
    affectedNodes.insert(root);
    for (int d : del) {
        affectedNodes.insert(d);
        for (int u : mCachedInGraph[d]) {
            affectedNodes.insert(u);
        }
        for (int v : mCachedOutGraph[d]) {
            affectedNodes.insert(v);
        }
    }
    std::unordered_map<int, std::set<int>> savedOut, savedIn;
    for (int node : affectedNodes) {
        savedOut[node] = mCachedOutGraph[node];
        savedIn[node] = mCachedInGraph[node];
    }
    MergeNodesInCachedGraph(root, del);
    bool hasCycle = HasCycle(mCachedOutGraph, mCachedInGraph);
    for (auto& kv : savedOut) {
        mCachedOutGraph[kv.first] = std::move(kv.second);
    }
    for (auto& kv : savedIn) {
        mCachedInGraph[kv.first] = std::move(kv.second);
    }
    if (hasCycle) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Merge skipped: detect cycle.");
        return false;
    }
    return true;
}

bool MixGraphMerger::CheckLatencyConstraint(const std::vector<int>& actualGroup)
{
    int totalAIC = 0;
    int totalAIV = 0;
    int totalAICOpNum = 0;
    int totalAIVOpNum = 0;
    for (int root : actualGroup) {
        for (int i = 0; i < mInput.numSubgraph; ++i) {
            if (FindParent(i) == root) {
                totalAIC += mInput.subgraphAICLatency[i];
                totalAIV += mInput.subgraphAIVLatency[i];
                totalAICOpNum += mInput.subgraphAICOpNum[i];
                totalAIVOpNum += mInput.subgraphAIVOpNum[i];
            }
        }
    }
    if (totalAIC == 0 || totalAIV == 0) {
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "Merge skipped: merged subgraph must be mixed (both AIC and AIV non-zero).");
        return false;
    }
    if (mInput.maxSubgraphAICOpNum > 0 && totalAICOpNum > mInput.maxSubgraphAICOpNum) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Merge skipped: group=%s aicOps=%d exceeds maxSubgraphAICOpNum=%d.",
                          IntVecToStr(actualGroup).c_str(), totalAICOpNum, mInput.maxSubgraphAICOpNum);
        return false;
    }
    if (mInput.maxSubgraphAIVOpNum > 0 && totalAIVOpNum > mInput.maxSubgraphAIVOpNum) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Merge skipped: group=%s aivOps=%d exceeds maxSubgraphAIVOpNum=%d.",
                          IntVecToStr(actualGroup).c_str(), totalAIVOpNum, mInput.maxSubgraphAIVOpNum);
        return false;
    }
    int totalOpNum = totalAICOpNum + totalAIVOpNum;
    if (mInput.maxSubgraphTotalOpNum > 0 && totalOpNum > mInput.maxSubgraphTotalOpNum) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Merge skipped: group=%s totalOps=%d exceeds maxSubgraphTotalOpNum=%d.",
                          IntVecToStr(actualGroup).c_str(), totalOpNum, mInput.maxSubgraphTotalOpNum);
        return false;
    }
    int totalLatency = totalAIC + totalAIV;
    if (totalLatency > mInput.maxLatency) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Merge skipped: total latency %d cycles exceeds max latency %d cycles.",
                          totalLatency, mInput.maxLatency);
        return false;
    }
    double ratio = (double)totalAIV / (double)totalAIC;
    if (ratio < mInput.aivRatio.first || ratio > mInput.aivRatio.second) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Merge skipped: AIV/AIC ratio %.2f out of range [%.2f, %.2f]", ratio,
                          mInput.aivRatio.first, mInput.aivRatio.second);
        return false;
    }
    return true;
}

bool MixGraphMerger::IsInvalidMergedInnerTensor(int tensorId, const std::unordered_set<int>& mergedRoots,
                                                std::vector<int>& prodIn, std::vector<int>& prodOut,
                                                std::vector<int>& consIn, std::vector<int>& consOut)
{
    const auto& tensorInfo = mInput.boundaryTensors[tensorId];
    if (!tensorInfo.isDDR) {
        return false;
    }
    bool hasProducerInMix = false;
    bool hasConsumerInMix = false;
    bool hasExternalEndpoint = false;
    for (int subgraphId : tensorInfo.producerSubgraphs) {
        int root = FindParent(subgraphId);
        if (mergedRoots.count(root) > 0) {
            hasProducerInMix = true;
            prodIn.push_back(subgraphId);
        } else {
            hasExternalEndpoint = true;
            prodOut.push_back(subgraphId);
        }
    }
    for (int subgraphId : tensorInfo.consumerSubgraphs) {
        int root = FindParent(subgraphId);
        if (mergedRoots.count(root) > 0) {
            hasConsumerInMix = true;
            consIn.push_back(subgraphId);
        } else {
            hasExternalEndpoint = true;
            consOut.push_back(subgraphId);
        }
    }
    return hasProducerInMix && hasConsumerInMix && hasExternalEndpoint;
}

// Enforce merge 只对全端点属于同一个有效 scope 的 tensor 进行豁免；其它情况沿用原子图端点检查。
// 该前提与 IsEnforceMergeBoundary 的单一 CvFuseId 判定一致；无法获得完整 scope 的端点不会进入豁免。
static bool HasSingleValidCvFuseId(const BoundaryTensorInfo& tensorInfo)
{
    if (tensorInfo.producerSubgraphs.size() != tensorInfo.producerCvFuseIds.size() ||
        tensorInfo.consumerSubgraphs.size() != tensorInfo.consumerCvFuseIds.size()) {
        return false;
    }
    int cvFuseId = -1;
    bool hasEndpoint = false;
    auto collect = [&](const std::vector<int>& ids) {
        for (int id : ids) {
            if (id < 0) {
                return false;
            }
            if (!hasEndpoint) {
                cvFuseId = id;
                hasEndpoint = true;
            } else if (cvFuseId != id) {
                return false;
            }
        }
        return true;
    };
    return collect(tensorInfo.producerCvFuseIds) && collect(tensorInfo.consumerCvFuseIds) && hasEndpoint;
}

bool MixGraphMerger::IsInvalidMergedInnerTensorByCvFuseId(int tensorId, const std::unordered_set<int>& mergedRoots,
                                                          std::vector<int>& prodIn, std::vector<int>& prodOut,
                                                          std::vector<int>& consIn, std::vector<int>& consOut)
{
    const auto& tensorInfo = mInput.boundaryTensors[tensorId];
    if (!tensorInfo.isDDR || HasSingleValidCvFuseId(tensorInfo)) {
        return false;
    }
    return IsInvalidMergedInnerTensor(tensorId, mergedRoots, prodIn, prodOut, consIn, consOut);
}

bool MixGraphMerger::CheckNoExternalUseOfMergedInnerTensor(const std::vector<int>& actualGroup, bool checkByCvFuseId)
{
    std::unordered_set<int> mergedRoots(actualGroup.begin(), actualGroup.end());
    ++mVisitStamp;
    for (int root : actualGroup) {
        for (int tensorId : mRootToBoundaryTensorIds[root]) {
            if (mTensorVisitStamp[tensorId] == mVisitStamp) {
                continue;
            }
            mTensorVisitStamp[tensorId] = mVisitStamp;
            std::vector<int> prodIn, prodOut, consIn, consOut;
            bool isInvalid = false;
            if (checkByCvFuseId) {
                isInvalid = IsInvalidMergedInnerTensorByCvFuseId(tensorId, mergedRoots, prodIn, prodOut, consIn,
                                                                 consOut);
            } else {
                isInvalid = IsInvalidMergedInnerTensor(tensorId, mergedRoots, prodIn, prodOut, consIn, consOut);
            }
            if (isInvalid) {
                const auto& tinfo = mInput.boundaryTensors[tensorId];
                if (mWarnedInnerTensorMagics.insert(tinfo.tensorMagic).second) {
                    bool hasScopelessEndpoint = tinfo.producerSubgraphs.size() != tinfo.producerCvFuseIds.size() ||
                                                tinfo.consumerSubgraphs.size() != tinfo.consumerCvFuseIds.size();
                    for (int cvFuseId : tinfo.producerCvFuseIds) {
                        hasScopelessEndpoint = hasScopelessEndpoint || cvFuseId < 0;
                    }
                    for (int cvFuseId : tinfo.consumerCvFuseIds) {
                        hasScopelessEndpoint = hasScopelessEndpoint || cvFuseId < 0;
                    }
                    const char* cause = checkByCvFuseId ? (hasScopelessEndpoint ?
                                                               "scope information is missing (CvFuseId=-1)" :
                                                               "producer/consumer carry different scopes (CvFuseId)") :
                                                          "producer/consumer carry different scopes (CvFuseId)";
                    APASS_LOG_WARN_F(Elements::Operation,
                                     "Merge group [%s] not merged: tensor %d producer(ingroup[%s] outgroup[%s]) "
                                     "consumer(ingroup[%s] outgroup[%s]) has both in-group and out-group endpoints. "
                                     "Likely cause: %s.",
                                     IntVecToStr(actualGroup).c_str(), tinfo.tensorMagic, IntVecToStr(prodIn).c_str(),
                                     IntVecToStr(prodOut).c_str(), IntVecToStr(consIn).c_str(),
                                     IntVecToStr(consOut).c_str(), cause);
                } else {
                    APASS_LOG_DEBUG_F(Elements::Operation,
                                      "Merge group [%s] not merged: tensor %d blocked by inner-external-use again, "
                                      "see previous WARN for detail.",
                                      IntVecToStr(actualGroup).c_str(), tinfo.tensorMagic);
                }
                return false;
            }
        }
    }
    return true;
}

// hinge 护栏分支数门槛: 2 路扇出为最常见良性拓扑(gqa cube->2 vec / mha 输出级 2:2 /
// softmax 共享读端双写链), 已由 sink 定量门控与同通路门控把关; >=3 路互不可达扇出
// 才构成 Unroll 型(mla_prolog_quant 8~16 路)并行度损失的主体。
static constexpr int kHingeBranchThreshold = 3;

// 两个 loop-carry 通路集合是否存在共享环
static bool HasSharedLoopPath(const std::set<int>& a, const std::set<int>& b)
{
    for (int path : a) {
        if (b.count(path) > 0) {
            return true;
        }
    }
    return false;
}

// 组内 root 图可达性: 从 from 沿 succGraph 有向边 DFS, 判断是否能到达 targets 中除自身外的节点
static bool HasReachableRoot(const std::unordered_map<int, std::set<int>>& succGraph, int from,
                             const std::set<int>& targets, std::set<int>& visited)
{
    if (visited.count(from) > 0) {
        return false;
    }
    visited.insert(from);
    auto it = succGraph.find(from);
    if (it == succGraph.end()) {
        return false;
    }
    for (int next : it->second) {
        if (next != from && targets.count(next) > 0) {
            return true;
        }
        if (HasReachableRoot(succGraph, next, targets, visited)) {
            return true;
        }
    }
    return false;
}

bool MixGraphMerger::CheckMergeBenefitByStructuralPattern(const std::vector<int>& actualGroup, bool allowSinkMerge)
{
    // 两层结构 benefit 判定(不依赖 latency):
    // tensor 层: 分侧统计 prodRoots/consRoots, 仅 prodRoots>1 && consRoots>1 拒绝为 N:M tensor;
    //            1:1/1:N/N:1 均视为简单 tensor edge, 建 root 级方向边。
    // root 图层: 用 rootPreds/rootSuccs 识别整体 N:M 并拒绝; 1:1/N:1/1:N 均通过。
    // sink 保护: actualGroup 内不产出 boundary tensor 的 root,
    //            统计其当前所有 producer root, >1 且串行损失(Σbranch − max)超过
    //            kSerialLossRatio × 合入root 才拒绝; 其余(损失可被 root 消化/全部归一 root)放行。
    const int multiBranchThreshold = 2; // 2+ 个 root/branch 视为多分支(fan-in/fan-out)
    std::unordered_set<int> mergedRoot(actualGroup.begin(), actualGroup.end());
    ++mVisitStamp;
    std::set<int> allProdRoots;
    std::unordered_map<int, std::set<int>> rootPreds;
    std::unordered_map<int, std::set<int>> rootSuccs;
    bool hasSimpleTensorEdge = false;
    for (int root : actualGroup) {
        for (int tensorId : mRootToBoundaryTensorIds[root]) {
            if (mTensorVisitStamp[tensorId] == mVisitStamp) {
                continue;
            }
            mTensorVisitStamp[tensorId] = mVisitStamp;
            const auto& info = mInput.boundaryTensors[tensorId];
            // 分侧收集 in-group producer/consumer 当前 root
            std::set<int> prodRoots;
            std::set<int> consRoots;
            for (int s : info.producerSubgraphs) {
                int parent = FindParent(s);
                if (mergedRoot.count(parent) > 0) {
                    prodRoots.insert(parent);
                    allProdRoots.insert(parent);
                }
            }
            for (int s : info.consumerSubgraphs) {
                int parent = FindParent(s);
                if (mergedRoot.count(parent) > 0) {
                    consRoots.insert(parent);
                }
            }
            if (prodRoots.empty() || consRoots.empty()) {
                continue;
            }
            // N:M tensor: 双侧均多 root
            if (prodRoots.size() >= multiBranchThreshold && consRoots.size() >= multiBranchThreshold) {
                APASS_LOG_DEBUG_F(Elements::Operation, "Structural merge skipped: N:M tensor %d (prod=%zu, cons=%zu).",
                                  info.tensorMagic, prodRoots.size(), consRoots.size());
                return false;
            }
            // 建 root 级方向边 (prodRoot -> consRoot), 跳过 self-loop
            for (int prodRoot : prodRoots) {
                for (int consRoot : consRoots) {
                    if (prodRoot == consRoot) {
                        continue;
                    }
                    hasSimpleTensorEdge = true;
                    rootPreds[consRoot].insert(prodRoot);
                    rootSuccs[prodRoot].insert(consRoot);
                }
            }
        }
    }
    // sink 保护: 不产出 boundary tensor 的 root, 当前 producer root 数 > 1 才拒绝。
    // 若所有 producer 已经归到同一个 root，合入 sink 不再减少并行分支。
    // 串行损失保护: 分支串行化损失(Σbranch − max)超过 kSerialLossRatio × 合入root 时才拒绝,
    // 损失可被 root 规模消化时放行。
    // 输出级 consolidate 旁路: 全图 ≥2 个出度0 sink 且本候选含其全部时,
    // 视为合并并行输出级(下游无并行可损失), 跳过 sink 保护。
    // 推迟门控: 常规合并轮凡涉及 sink 的候选组一律推迟, 防止 sink 提前合入后逐轮吞并大量小子图;
    // 仅常规合并收敛后的 sink 尝试轮(allowSinkMerge=true)才执行上述检查并尝试合并。
    bool skipSinkProtection = false;
    if (mGlobalOutputSinks.size() >= multiBranchThreshold) {
        skipSinkProtection = true;
        for (int sinkSg : mGlobalOutputSinks) {
            if (mergedRoot.count(FindParent(sinkSg)) == 0) {
                skipSinkProtection = false;
                break;
            }
        }
        if (skipSinkProtection) {
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "Structural merge allowed: candidate consolidates all %zu output sinks.",
                              mGlobalOutputSinks.size());
        }
    }
    if (skipSinkProtection) {
        if (!allowSinkMerge) {
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "Structural merge deferred: candidate consolidates all output sinks, "
                              "wait for sink merge round.");
            return false;
        }
    } else {
        for (int root : actualGroup) {
            if (allProdRoots.count(root) > 0) {
                continue;
            }

            std::set<int> incomingRoots;
            for (int tensorId : mRootToBoundaryTensorIds[root]) {
                const auto& info = mInput.boundaryTensors[tensorId];
                for (int s : info.producerSubgraphs) {
                    int predRoot = FindParent(s);
                    if (predRoot != root) {
                        incomingRoots.insert(predRoot);
                    }
                }
            }

            if (incomingRoots.size() < multiBranchThreshold) {
                continue;
            }
            if (!allowSinkMerge) {
                APASS_LOG_DEBUG_F(Elements::Operation,
                                  "Structural merge deferred: convergence sink at root %d (rootInDeg=%zu), "
                                  "wait for sink merge round.",
                                  root, incomingRoots.size());
                return false;
            }
            // sink 收益保护(相对 root 标尺): 分支串行化损失(Σ-max)超过 root 总 latency 的
            // kSerialLossRatio 倍时拒绝。详见 IsSinkMergeUnbeneficial。
            if (IsSinkMergeUnbeneficial(root, incomingRoots)) {
                APASS_LOG_DEBUG_F(Elements::Operation,
                                  "Structural merge skipped: convergence sink at root %d (rootInDeg=%zu).", root,
                                  incomingRoots.size());
                return false;
            }
        }
    }
    // N:M 保护: 整体 root 图同时存在 N:1 和 1:N 则拒绝
    bool hasNto1 = false;
    bool has1toN = false;
    for (const auto& pair : rootPreds) {
        if (pair.second.size() >= multiBranchThreshold) {
            hasNto1 = true;
            break;
        }
    }
    for (const auto& pair : rootSuccs) {
        if (pair.second.size() >= multiBranchThreshold) {
            has1toN = true;
            break;
        }
    }
    if (hasNto1 && has1toN) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Structural merge skipped: N:M root pattern.");
        return false;
    }
    // 并行分支串行化防护(实测依据: mla_prolog_quant automix, Unroll 展开后的扇出/扇入组把
    // 8~16 路可跨核并行的分支串进单 kernel, e2e +6.6%): 同一 hinge 端点(组内或组外的生产者/
    // 消费者)聚合挂着 >= kHingeBranchThreshold 个组内 root, 且这些 root 在组内 root 图上两两
    // 互不可达(无依赖路径)、亦非同环耦合时, 判定为并行分支被串行化, 拒绝; 分支间存在组内
    // 依赖路径的链式合并(可打通 L0C2UB 的 cube+glue 链)与同环多链不受影响。
    ++mVisitStamp;
    std::unordered_map<int, std::set<int>> hingeToBranchRoots;
    for (int root : actualGroup) {
        for (int tensorId : mRootToBoundaryTensorIds[root]) {
            if (mTensorVisitStamp[tensorId] == mVisitStamp) {
                continue;
            }
            mTensorVisitStamp[tensorId] = mVisitStamp;
            const auto& info = mInput.boundaryTensors[tensorId];
            for (int s : info.producerSubgraphs) {
                int prodRoot = FindParent(s);
                for (int c : info.consumerSubgraphs) {
                    int consRoot = FindParent(c);
                    if (consRoot == prodRoot) {
                        continue;
                    }
                    if (mergedRoot.count(consRoot) > 0) {
                        hingeToBranchRoots[prodRoot].insert(consRoot);
                    }
                    if (mergedRoot.count(prodRoot) > 0) {
                        hingeToBranchRoots[consRoot].insert(prodRoot);
                    }
                }
            }
        }
    }
    for (const auto& kv : hingeToBranchRoots) {
        if (kv.second.size() < kHingeBranchThreshold) {
            continue;
        }
        // 同环豁免: 分支两两共享 loop-carry 通路(同一 ring 的多链, 如 online softmax
        // 共享读端的多条写链)时, 执行序已被环迭代约束, 非独立并行分支, 不属本护栏
        if (!mRootLoopPaths.empty()) {
            bool ringCoupled = true;
            for (int branch : kv.second) {
                for (int other : kv.second) {
                    if (branch != other && !HasSharedLoopPath(mRootLoopPaths[branch], mRootLoopPaths[other])) {
                        ringCoupled = false;
                        break;
                    }
                }
                if (!ringCoupled) {
                    break;
                }
            }
            if (ringCoupled) {
                continue;
            }
        }
        bool hasDependentPair = false;
        for (int branch : kv.second) {
            std::set<int> others = kv.second;
            others.erase(branch);
            std::set<int> visited;
            if (HasReachableRoot(rootSuccs, branch, others, visited)) {
                hasDependentPair = true;
                break;
            }
        }
        if (!hasDependentPair) {
            // 规模豁免(限纯 V 分支): 独立分支全部为纯 vector(无 AIC op)且串行化损失(Σbranch − max)
            // 不超过 hinge 端点 root 总 latency 的 kSerialLossRatio 倍时, 属轻量准备段扇入
            // (gather/索引/反量化碎片, ~20op 级, sparse_attention_antiquant: 64 碎片致 65 kernel
            // vs 基线 5), 边界与调度收益远大于微小串行损失, 放行; 含 cube 的实质并行分支
            // (mla 型 8~16 路 Unroll 体)不豁免, 保持完整护栏。判据与 sink 串行损失门同标尺。
            bool allBranchPureVec = true;
            int64_t hingeLatency = 0;
            int64_t sumBranchLatency = 0;
            int64_t maxBranchLatency = 0;
            for (size_t i = 0; i < static_cast<size_t>(mInput.numSubgraph); ++i) {
                int root = FindParent(static_cast<int>(i));
                if (mInput.subgraphAICOpNum[i] > 0 && kv.second.count(root) > 0 && root != kv.first) {
                    allBranchPureVec = false;
                    break;
                }
                int64_t lat = static_cast<int64_t>(mInput.subgraphAICLatency[i]) + mInput.subgraphAIVLatency[i];
                if (root == kv.first) {
                    hingeLatency += lat;
                } else if (kv.second.count(root) > 0) {
                    sumBranchLatency += lat;
                    maxBranchLatency = std::max(maxBranchLatency, lat);
                }
            }
            if (allBranchPureVec && sumBranchLatency - maxBranchLatency <= kSerialLossRatio * hingeLatency) {
                continue;
            }
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "Structural merge skipped: hinge root %d fans to %zu independent parallel branches [%s].",
                              kv.first, kv.second.size(),
                              IntVecToStr(std::vector<int>(kv.second.begin(), kv.second.end())).c_str());
            return false;
        }
    }
    // 通过条件: 至少存在一个简单 tensor edge
    if (!hasSimpleTensorEdge) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Structural merge skipped: no simple tensor edge benefit.");
        return false;
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "Structural merge allowed: has simple tensor edge benefit.");
    return true;
}

// 判断汇聚 sink 的合并是否无收益需拒绝(整体收益估算, 相对 root 标尺):
// 合并 = root 吞并并行分支: 收益(省边界拷贝/调度)与 root 自身规模同量级;
// 损失 = 分支串行化新增耗时 ΣbranchLatency - max(branchLatency)。
// 串行损失达到 root 总 latency 的 kSerialLossRatio 倍时, 为省边界通信付出的代价
// 远超 root 可消化的规模, 拒绝; 损失可被 root 规模消化时放行。
// 相对判据随算子规模自适应, 无需绝对阈值标定。
// 调用方需保证入边数 ≥ 2(multiBranchThreshold 门控)。
bool MixGraphMerger::IsSinkMergeUnbeneficial(int root, const std::set<int>& incomingRoots)
{
    int64_t rootLatency = 0;
    for (size_t i = 0; i < static_cast<size_t>(mInput.numSubgraph); ++i) {
        if (FindParent(static_cast<int>(i)) == root) {
            rootLatency += static_cast<int64_t>(mInput.subgraphAICLatency[i]) + mInput.subgraphAIVLatency[i];
        }
    }
    int64_t sumLatency = 0;
    int64_t maxLatency = 0;
    for (int incomingRoot : incomingRoots) {
        int64_t latency = 0;
        for (size_t i = 0; i < static_cast<size_t>(mInput.numSubgraph); ++i) {
            if (FindParent(static_cast<int>(i)) == incomingRoot) {
                latency += static_cast<int64_t>(mInput.subgraphAICLatency[i]) + mInput.subgraphAIVLatency[i];
            }
        }
        sumLatency += latency;
        maxLatency = std::max(maxLatency, latency);
    }
    const int64_t serialLoss = sumLatency - maxLatency;
    // root 无算子(纯汇合点)时无收益来源, 任何串行损失都无法消化, 直接拒绝
    if (rootLatency <= 0) {
        return serialLoss > 0;
    }
    APASS_LOG_DEBUG_F(Elements::Operation,
                      "Sink merge gate: root=%d rootLatency=%lld branches=%zu serialLoss=%lld lossPerRoot=%lld, "
                      "merge %s.",
                      root, static_cast<long long>(rootLatency), incomingRoots.size(),
                      static_cast<long long>(serialLoss), static_cast<long long>(serialLoss / rootLatency),
                      serialLoss > rootLatency * kSerialLossRatio ? "rejected" : "allowed");
    return serialLoss > rootLatency * kSerialLossRatio;
}

bool MixGraphMerger::CheckLoopPathConsistency(const std::vector<int>& actualGroup)
{
    // 同通路门控(carry 约束原则的语义化实现):
    // 候选组内 loop-carry 通路归属混杂时, 逐对校验"集合不同的成员对"的耦合性 ——
    //   全体集合相同(含全空): 链内/普通合并, 串行性原本就存在, 放行;
    //   混杂 + 成员对共环(集合相交): 同环多链, 链内合并, 放行;
    //   混杂 + 空集(环外)成员挂在环上成员的直接边界 tensor 数据边:
    //     生产者-消费者耦合, 执行次序已被依赖链约束(如 reshape-only 形状适配挂靠), 放行;
    //   混杂 + 非空不相交(跨环)成员对: 即使存在数据边也拒绝 —— 融合两个独立 loop body
    //     破坏 slot-scope 循环体完整性(两环 trip count 可不同), 正确性护栏;
    //   混杂 + 空集成员无数据边挂靠: 真并行分支被吞进 carry 链,
    //     串行链拖住并行体, 拒绝(护栏: mla 独立分支与脊柱融合 e2e +244%)。
    // 贪心合并按邻接展开, 非邻接的独立分支不会进入候选组, 该判据与合并动态自洽。
    // subgraphLoopPaths 为空向量表示前端无 slot scope/feedback slot, 门控不启用。
    if (mRootLoopPaths.empty()) {
        return true;
    }
    const std::set<int>* ref = nullptr;
    bool mixed = false;
    for (int root : actualGroup) {
        const auto& paths = mRootLoopPaths[root];
        if (ref == nullptr) {
            ref = &paths;
            continue;
        }
        if (*ref != paths) {
            mixed = true;
            break;
        }
    }
    if (!mixed) {
        return true;
    }
    auto hasDirectDataEdge = [this](int rootA, int rootB) {
        for (int tensorId : mRootToBoundaryTensorIds[rootA]) {
            const auto& tensorInfo = mInput.boundaryTensors[tensorId];
            for (int sg : tensorInfo.producerSubgraphs) {
                if (FindParent(sg) == rootB) {
                    return true;
                }
            }
            for (int sg : tensorInfo.consumerSubgraphs) {
                if (FindParent(sg) == rootB) {
                    return true;
                }
            }
        }
        return false;
    };
    for (size_t i = 0; i < actualGroup.size(); ++i) {
        for (size_t j = i + 1; j < actualGroup.size(); ++j) {
            const auto& pathsA = mRootLoopPaths[actualGroup[i]];
            const auto& pathsB = mRootLoopPaths[actualGroup[j]];
            if (pathsA == pathsB) {
                continue;
            }
            // 数据边豁免仅限环外(空集)成员挂靠; 非空不相交(跨环)即使有数据边也拒绝:
            // 融合两个独立 loop body 破坏 slot-scope 循环体完整性(两环 trip count 可不同)
            bool offRingCoupled = (pathsA.empty() || pathsB.empty()) &&
                                  hasDirectDataEdge(actualGroup[i], actualGroup[j]);
            if (HasSharedLoopPath(pathsA, pathsB) || offRingCoupled) {
                continue;
            }
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "Merge skipped: loop-carry path mismatch inside group [%s] "
                              "(mismatched pair is cross-ring or uncoupled).",
                              IntVecToStr(actualGroup).c_str());
            return false;
        }
    }
    return true;
}

bool MixGraphMerger::CanMergeWithConstraints(const std::vector<int>& actualGroup, bool allowSinkMerge)
{
    if (actualGroup.size() <= 1) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Merge skipped: already merged.");
        return false;
    }
    if (!CheckLoopPathConsistency(actualGroup)) {
        return false;
    }
    if (!CanMergeWithoutCycle(actualGroup)) {
        return false;
    }
    if (!CheckNoExternalUseOfMergedInnerTensor(actualGroup)) {
        return false;
    }
    if (!CheckLatencyConstraint(actualGroup)) {
        return false;
    }
    return CheckMergeBenefitByStructuralPattern(actualGroup, allowSinkMerge);
}

void MixGraphMerger::UpdateBoundaryTensorIndex(const std::vector<int>& actualGroup)
{
    std::vector<int> mergedTensorIds;
    for (int root : actualGroup) {
        mergedTensorIds.insert(mergedTensorIds.end(), mRootToBoundaryTensorIds[root].begin(),
                               mRootToBoundaryTensorIds[root].end());
        mRootToBoundaryTensorIds[root].clear();
    }
    int newRoot = FindParent(actualGroup[0]);
    mRootToBoundaryTensorIds[newRoot].swap(mergedTensorIds);
}

void MixGraphMerger::PerformMerge(const std::vector<int>& actualGroup)
{
    if (actualGroup.size() <= 1) {
        return;
    }
    int root = actualGroup[0];
    for (size_t i = 1; i < actualGroup.size(); ++i) {
        UnionSets(root, actualGroup[i]);
    }
    // 合并后真实 root 继承组内全部成员的通路归属。union-by-rank 可能使集合根轮转离开
    // actualGroup[0], 必须以 FindParent 重取的根为落点(与 ApplyMergeToGraph 同一时机取根),
    // 否则 enforce 路径(不经过同通路门控, 集合可异构)真实 root 会漏继承, 后续门控静默失效
    if (!mRootLoopPaths.empty()) {
        int newRoot = FindParent(actualGroup[0]);
        for (int sg : actualGroup) {
            if (sg != newRoot) {
                mRootLoopPaths[newRoot].insert(mRootLoopPaths[sg].begin(), mRootLoopPaths[sg].end());
            }
        }
    }
    ApplyMergeToGraph(actualGroup);
    UpdateBoundaryTensorIndex(actualGroup);
}

void MixGraphMerger::ApplyMergeToGraph(const std::vector<int>& actualGroup)
{
    if (actualGroup.size() <= 1) {
        return;
    }
    int root = FindParent(actualGroup[0]);
    std::set<int> del;
    for (int d : actualGroup) {
        if (d != root) {
            del.insert(d);
        }
    }
    MergeNodesInCachedGraph(root, del);
}

void MixGraphMerger::MergeNodesInCachedGraph(int root, const std::set<int>& del)
{
    for (int d : del) {
        for (int u : mCachedInGraph[d]) {
            mCachedOutGraph[u].erase(d);
            mCachedOutGraph[u].insert(root);
        }
        for (int v : mCachedOutGraph[d]) {
            mCachedInGraph[v].erase(d);
            mCachedInGraph[v].insert(root);
        }
        mCachedInGraph[root].insert(mCachedInGraph[d].begin(), mCachedInGraph[d].end());
        mCachedOutGraph[root].insert(mCachedOutGraph[d].begin(), mCachedOutGraph[d].end());
        mCachedInGraph[d].clear();
        mCachedOutGraph[d].clear();
    }
    mCachedInGraph[root].erase(root);
    mCachedOutGraph[root].erase(root);
}

void MixGraphMerger::UpdateOutput()
{
    std::vector<int> mapping(mInput.numSubgraph, -1);
    int newId = 0;
    for (int i = 0; i < mInput.numSubgraph; ++i) {
        int root = FindParent(i);
        if (mapping[root] == -1) {
            mapping[root] = newId++;
        }
        mOutput.subgraphIdUpdated[i] = mapping[root];
    }
    mOutput.numSubgraphUpdated = newId;
}

static bool ValidateOutput(const MergeOutput& output, int numSubgraph)
{
    if (output.numSubgraphUpdated <= 0 || output.numSubgraphUpdated > numSubgraph) {
        return false;
    }
    if ((int)output.subgraphIdUpdated.size() != numSubgraph) {
        return false;
    }
    std::set<int> ids;
    for (int i = 0; i < numSubgraph; ++i) {
        int id = output.subgraphIdUpdated[i];
        if (id < 0 || id >= output.numSubgraphUpdated) {
            return false;
        }
        ids.insert(id);
    }
    for (int i = 0; i < output.numSubgraphUpdated; ++i) {
        if (ids.find(i) == ids.end()) {
            return false;
        }
    }
    return true;
}

MergeOutput MixGraphMerger::Merge(const MergeInput& input)
{
    if (!ValidateInput(input)) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Invalid input parameters");
        return mOutput;
    }
    Initialize(input);
    // 存在 CV 混合 scope(cvFuseId>=0)时禁用 auto-mix, 该 scope 涉及的子图由 enforce 路径合并为 Mix 子图,
    // 防止 auto-mix 改变手动 scope 的切分结果
    if (input.hasScopedOp && enableAutoMix) {
        APASS_LOG_INFO_F(Elements::Operation, "Auto mix partition is skipped since scoped op exists.");
    }
    const int mergeLoopNum = 5;
    // sink 合并推迟门控: 常规轮(allowSinkMerge=false)不允许合并含汇聚 sink 的候选组(避免提前吞并大量小子图);
    // 常规轮收敛或常规轮数耗尽后切换为 sink 尝试轮, 且仅执行一轮, 无论该轮是否有更新都结束合并
    bool allowSinkMerge = false;
    for (int mergeLoopStep = 0; mergeLoopStep < mergeLoopNum + 1; mergeLoopStep++) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Enter merge loop %d (sink merge allowed: %s).", mergeLoopStep,
                          allowSinkMerge ? "True" : "False");
        bool hasUpdated = false;
        for (size_t i = 0; i < input.mergeGroup.size(); ++i) {
            const auto& group = input.mergeGroup[i];
            std::vector<int> actualGroup = GetActualGroup(group);
            if (actualGroup.size() <= 1) {
                APASS_LOG_DEBUG_F(Elements::Operation, "Skip merge group %zu: already merged", i);
                continue;
            }
            if (input.isEnforceMergeGroup[i]) {
                // true: enforce 组改走 CvFuseId 判定, 同 scope 的组外端点豁免(终将并入同一混合子图),
                // 防止 enforce 组互相等待无法合并; auto-mix 路径保持默认 subgraphId 判定不变
                if (CanMergeWithoutCycle(actualGroup) && CheckNoExternalUseOfMergedInnerTensor(actualGroup, true)) {
                    // enforce 路径不受 op 数上限拦截, 但超限时打 WARN 提示编译时间风险(观测用, 不改变合并行为)
                    WarnIfEnforceOpNumExceeds(actualGroup);
                    APASS_LOG_DEBUG_F(Elements::Operation, "Merge group %zu succeeded: actualGroup=%s.", i,
                                      IntVecToStr(actualGroup).c_str());
                    PerformMerge(actualGroup);
                    hasUpdated = true;
                } else {
                    APASS_LOG_DEBUG_F(Elements::Operation,
                                      "Merge group %zu [%s] skipped due to constraints, see above for reason.", i,
                                      IntVecToStr(actualGroup).c_str());
                }
            } else if (!input.hasScopedOp && enableAutoMix && mergeLoopStep != 0 && input.isValidMergeGroup[i] &&
                       CanMergeWithConstraints(actualGroup, allowSinkMerge)) {
                APASS_LOG_DEBUG_F(Elements::Operation, "Merge group %zu succeeded: actualGroup=%s.", i,
                                  IntVecToStr(actualGroup).c_str());
                PerformMerge(actualGroup);
                hasUpdated = true;
            } else if (mergeLoopStep != 0) {
                APASS_LOG_DEBUG_F(Elements::Operation, "Merge group %zu [%s] skipped due to constraints.", i,
                                  IntVecToStr(actualGroup).c_str());
            }
        }
        if (allowSinkMerge) {
            // sink 尝试轮仅一轮, 执行完即结束合并(后续轮不再重试 sink 候选组)
            break;
        } else if (mergeLoopStep > 0 && (!hasUpdated || mergeLoopStep == mergeLoopNum - 1)) {
            // 常规合并收敛, 或常规轮数耗尽(保证 sink 尝试轮必被执行): 切换为 sink 尝试轮,
            // 对被推迟的 sink 候选组做均衡/悬殊检查并尝试合并
            allowSinkMerge = true;
        }
    }
    UpdateOutput();
    if (!ValidateOutput(mOutput, input.numSubgraph)) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Invalid output detected");
    }
    return mOutput;
}

} // namespace npu::tile_fwk
