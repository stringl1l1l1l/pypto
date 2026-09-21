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
 * \file remove_redundant_op_utils.h
 * \brief utils for redundant view/assemble and slice/contract elimination
 */

#ifndef PASS_REMOVE_REDUNDANT_OP_UTILS_H_
#define PASS_REMOVE_REDUNDANT_OP_UTILS_H_

#include "interface/function/function.h"
#include "interface/tensor/irbuilder.h"
#include "interface/tensor/logical_tensor.h"

namespace npu {
namespace tile_fwk {
class RemoveRedundantOpUtils {
public:
    RemoveRedundantOpUtils() = default;
    ~RemoveRedundantOpUtils() = default;

    static Status Process(Function& function, std::vector<Operation*>& newOps, bool& operationUpdated);
    static Status ProcessViewAssembleLike(Function& function, std::vector<Operation*>& newOps, bool& operationUpdated);
    static Status ProcessContractSlice(Function& function, std::vector<Operation*>& newOps, bool& operationUpdated);
    static Status ProcessViewCopyout(Function& function, bool& operationUpdated);
    static Status ProcessCopyinAssemble(Function& function, bool& operationUpdated);

private:
    struct SliceContractSliceChain {
        Operation* precedingSlice = nullptr;
        Operation* contract = nullptr;
        LogicalTensorPtr sourceTensor;
        LogicalTensorPtr contractInput;
        LogicalTensorPtr contractOutput;
        std::vector<Operation*> consumers;
    };
    struct SliceContractSliceRewrite {
        Operation* consumer = nullptr;
        std::vector<int64_t> localOffset;
        std::vector<SymbolicScalar> localDynOffset;
        std::vector<int64_t> composedOffset;
        bool isFullSlice = false;
    };
    struct ContractSliceRewriteInfo {
        Operation* consumer = nullptr;
        std::vector<int64_t> localOffset;
        std::vector<SymbolicScalar> localDynOffset;
        bool isFullSlice = false;
    };
    using MergedOffset = std::pair<std::vector<int64_t>, std::vector<SymbolicScalar>>;

    Status ProcessImpl(Function& function, std::vector<Operation*>& newOps, bool& operationUpdated);
    Status ProcessViewAssembleLikeImpl(Function& function, std::vector<Operation*>& newOps, bool& operationUpdated);
    Status ProcessContractSliceImpl(Function& function, std::vector<Operation*>& newOps, bool& operationUpdated);
    Status ProcessSliceContractSlice(Function& function, std::vector<Operation*>& newOps, bool& operationUpdated);
    Status ProcessContractSliceContract(Function& function, bool& operationUpdated);
    static bool MatchContractSliceContract(Operation& sliceOp, Operation*& matchedProducer);
    static bool IsMatmulBackedContractProducers(const std::set<Operation*, LogicalTensor::CompareOp>& producers);
    static void EraseOrphanMiddleProducers(const LogicalTensorPtr& middle);
    Status ProcessMultiContractSingleSlice(Function& function, bool& operationUpdated);
    Status ProcessSingleContractMultiSlice(Function& function, std::vector<Operation*>& newOps, bool& operationUpdated);

    static bool CollectSliceContractSliceChain(const Operation& contractOp, SliceContractSliceChain& chain);
    static bool ComposeSliceContractOffset(const SliceContractSliceChain& chain, const Operation& consumer,
                                           std::vector<int64_t>& composedOffset);
    static bool CollectSliceContractSliceRewrites(const SliceContractSliceChain& chain, bool isFanout,
                                                  bool keepL1Fanout, std::vector<SliceContractSliceRewrite>& rewrites,
                                                  bool& canComposeOffsets);
    static bool CollectL1FoldMergedOffsets(const SliceContractSliceChain& chain,
                                           const std::vector<SliceContractSliceRewrite>& rewrites,
                                           std::vector<MergedOffset>& mergedOffsets);
    static void FoldL1FanoutChain(const SliceContractSliceChain& chain,
                                  const std::vector<SliceContractSliceRewrite>& rewrites);
    static void RewriteComposableFanoutChain(const SliceContractSliceChain& chain,
                                             const std::vector<SliceContractSliceRewrite>& rewrites);
    bool RewriteDynamicFanoutFallback(Function& function, const SliceContractSliceChain& chain,
                                      std::vector<SliceContractSliceRewrite>& rewrites, bool hasL1Consumer,
                                      std::vector<Operation*>& newOps, bool& operationUpdated);
    void RewriteSliceContractSliceConsumer(Function& function, Operation& contract, Operation& consumer,
                                           bool isFullSlice, const std::vector<int64_t>& localOffset,
                                           const std::vector<SymbolicScalar>& localDynOffset,
                                           const LogicalTensorPtr& contractInput, LogicalTensorPtr& survivingInput,
                                           bool redirectEnabled, const LogicalTensorPtr& redirectTarget,
                                           std::vector<Operation*>& newOps, bool& operationUpdated);
    bool ProcessSliceContractSliceFanout(Function& function, const SliceContractSliceChain& chain,
                                         std::vector<SliceContractSliceRewrite>& rewrites, bool keepL1Fanout,
                                         bool canComposeOffsets, std::vector<Operation*>& newOps,
                                         bool& operationUpdated);
    void ProcessSliceContractSliceSingle(Function& function, const SliceContractSliceChain& chain,
                                         const SliceContractSliceRewrite& rewrite, std::vector<Operation*>& newOps,
                                         bool& operationUpdated);
    static bool ShouldSkipContractMultiSlice(const Operation& op, const LogicalTensorPtr& contractInput,
                                             const LogicalTensorPtr& contractOutput,
                                             const std::set<Operation*, LogicalTensor::CompareOp>& consumers);
    static bool CollectContractSliceRewriteInfos(Operation& op, const LogicalTensorPtr& contractInput,
                                                 const std::set<Operation*, LogicalTensor::CompareOp>& consumers,
                                                 std::vector<ContractSliceRewriteInfo>& rewriteInfos,
                                                 std::set<Operation*>& precedingSlices);
    static LogicalTensorPtr GetMixedConsumerRedirectTarget(const LogicalTensorPtr& contractInput);
    void RewriteContractMultiSliceConsumers(Function& function, Operation& op, const LogicalTensorPtr& contractInput,
                                            std::vector<ContractSliceRewriteInfo>& rewriteInfos,
                                            std::set<Operation*>& precedingSlices, std::vector<Operation*>& newOps,
                                            bool& operationUpdated);

    void ProcessPerfectMatch(Function& function, LogicalTensorPtr& startTensor, LogicalTensorPtr& endTensor,
                             bool endTensorIsOutcast, bool& operationUpdated);
    void RemoveViewAssembleForOutcast(LogicalTensorPtr& startTensor, LogicalTensorPtr& endTensor,
                                      bool& operationUpdated);
    void CalculateViewOffset(Operation& op, LogicalTensorPtr& startTensor, LogicalTensorPtr& endTensor,
                             std::vector<int64_t>& newOffset, std::vector<SymbolicScalar>& newDynOffset);
    void GenerateNewViewLike(Function& function, Operation& op, LogicalTensorPtr& startTensor,
                             LogicalTensorPtr& endTensor, Opcode newOpcode, std::vector<Operation*>& newOps,
                             bool& operationUpdated);
    void GenerateContractSliceView(Function& function, Operation& sliceOp, const LogicalTensorPtr& inputTensor,
                                   const std::vector<int64_t>& fromOffset,
                                   const std::vector<SymbolicScalar>& fromDynOffset, std::vector<Operation*>& newOps,
                                   bool& operationUpdated);

    bool IsNotSameViewInput(LogicalTensorPtr& startTensor, LogicalTensorPtr& endTensor) const;
    bool IsDataReplace(LogicalTensorPtr& endTensor) const;
    bool IsValidViewAssemble(LogicalTensorPtr& startTensor, LogicalTensorPtr& endTensor) const;

    IRBuilder irBuilder_;
};
} // namespace tile_fwk
} // namespace npu
#endif // PASS_REMOVE_REDUNDANT_OP_UTILS_H_
