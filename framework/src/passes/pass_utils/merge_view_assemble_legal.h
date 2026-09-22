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
 * \file merge_view_assemble_utils.h
 * \brief utils of view and assemble operation merging
 */

#ifndef PASS_MERGE_VIEW_ASSEMBLE_LEGAL_H_
#define PASS_MERGE_VIEW_ASSEMBLE_LEGAL_H_

#include <unordered_map>

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/irbuilder.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {
class LegalMergeAssembleUtils {
public:
    LegalMergeAssembleUtils() = default;
    ~LegalMergeAssembleUtils() = default;

    struct ViewOp {
        std::shared_ptr<LogicalTensor> input;
        std::shared_ptr<LogicalTensor> output;
        std::vector<int64_t> offset;
        std::vector<SymbolicScalar> dynOffset;
        std::vector<SymbolicScalar> dynValidShape;
        MemoryType toType = MemoryType::MEM_UNKNOWN;
        bool hasCopyInMode;       // 是否有copy_in_mode属性
        std::any copyInModeValue; // copy_in_mode属性值
        bool hasL1PaddingMode;    // 是否有copy_in_l1_padding_mode属性
        std::any l1PaddingMode;   // copy_in_l1_padding_mode属性值
        bool hasKIndex;           // 是否有copy_in_l1_k_index属性
        std::any kIndex;          // copy_in_l1_k_index属性值（K维度在dynValidShape中的索引，0或1）
        bool hasIsGemv;           // 是否有 isGemv 属性
        std::any isGemvValue;     // isGemv 属性值（0=false，非0=true）
        ir::Span span;            // 链路最早操作的span
        Operation::ScopeInfo scopeInfo;
        Opcode opcode = Opcode::OP_VIEW;
    };
    struct AssembleOp {
        std::shared_ptr<LogicalTensor> input;
        std::shared_ptr<LogicalTensor> output;
        std::vector<int64_t> offset;
        std::vector<SymbolicScalar> dynOffset;
        ir::Span span; // 链路最早操作的span
        Operation::ScopeInfo scopeInfo;
        std::string rmwModeAttr;
        Opcode opcode = Opcode::OP_ASSEMBLE;
        bool atomicFromReduceAcc = false;
        bool atomicFromExplicitRmw = false;
    };
    struct ConsumerCacheEntry {
        std::vector<Operation*> viewConsumers;
        std::vector<Operation*> assembleConsumers;
        // Any non-assemble consumer stops an assemble chain at this tensor.
        bool hasAssembleChainStopper = false;
    };

    static Status MergeViewAssemble(Function& function);

    Status Process(Function& function);

    // View chain processing methods
    /**
     * @brief Merge a chain of view operations into a single view.
     *
     * @param function the target function for the operation to be processed.
     * @param operation the starting operation of the view chain.
     * @param chain the list of operations in the view chain.
     * @return Status indicating success or failed.
     */
    Status MergeViewChain(Function& function, Operation& operation, std::vector<Operation*>& chain,
                          int effectiveScopeId = -1);

    void InitOperationChain(Operation& operation, std::vector<Operation*>& chain);

    /**
     * @brief Process the consumer chain of a view.
     *
     * @param function the target function for the operation to be processed.
     * @param consumers the consumers for the view to be processed.
     * @param chain the list of operations in the view chain.
     * @param chainEnd a flag indicating whether the chain has ended.
     * @return Status indicating success or failed.
     */
    Status ProcessConsumerChain(Function& function, const ConsumerCacheEntry& consumers, std::vector<Operation*>& chain,
                                bool& chainEnd, int effectiveScopeId);

    Status ProcessChainEnd(Function& function, std::vector<Operation*>& chain);

    /**
     * @brief Calculate the merged offsets and dynamic vaildshapes for the chain of a view.
     *
     * @param chain the list of operations in the view chain.
     * @param newOffset the calculated newoffset.
     * @param newDynOffset the calculated newDynOffset.
     * @param newDynValidShape the calculated newDynValidShape.
     * @return Status indicating success or failed.
     */
    Status CalculateMergedOffsets(const std::vector<Operation*>& chain, std::vector<int64_t>& newOffset,
                                  std::vector<SymbolicScalar>& newDynOffset,
                                  std::vector<SymbolicScalar>& newDynValidShape);

    /**
     * @brief Recode the merged offsets and dynamic vaildshapes for the chain of a view.
     *
     * @param lastViewOp the list of operations in the view chain.
     * @param startTensor the start tensor of the chain.
     * @param endTensor the end tensor of the chain.
     * @param newOffset the calculated newoffset.
     * @param newDynOffset the calculated newDynOffset.
     * @param newDynValidShape the calculated newDynValidShape.
     */
    void RecordMergedViewOperation(Operation* lastViewOp, const std::shared_ptr<LogicalTensor>& startTensor,
                                   const std::shared_ptr<LogicalTensor>& endTensor,
                                   const std::vector<int64_t>& newOffset,
                                   const std::vector<SymbolicScalar>& newDynOffset,
                                   const std::vector<SymbolicScalar>& newDynValidShape, const ir::Span& span,
                                   const Operation::ScopeInfo& scopeInfo, Opcode opcode);

    // Assemble chain processing methods
    /**
     * @brief Merge a chain of assemble operations into a single assemble.
     *
     * @param function the target function for the operation to be processed.
     * @param operation the starting operation of the assemble chain.
     * @param chain the list of operations in the assemble chain.
     * @return Status indicating success or failed.
     */
    Status MergeAssembleChain(Function& function, Operation& operation, std::vector<Operation*>& chain,
                              int effectiveScopeId = -1);

    void InitAssembleChain(Operation& operation, std::vector<Operation*>& chain);

    /**
     * @brief Process the consumer chain of a assemble.
     *
     * @param function the target function for the operation to be processed.
     * @param consumers the consumers for the assemble to be processed.
     * @param chain the list of operations in the assemble chain.
     * @param chainEnd a flag indicating whether the chain has ended.
     * @return Status indicating success or failed.
     */
    Status ProcessAssembleConsumers(Function& function, const ConsumerCacheEntry& consumers,
                                    std::vector<Operation*>& chain, bool& chainEnd, int effectiveScopeId);

    Status ProcessAssembleChainEnd(Function& function, std::vector<Operation*>& chain, Operation& operation);

    std::pair<std::vector<int64_t>, std::vector<SymbolicScalar>> CalculateAssembleOffsets(
        const std::vector<Operation*>& chain, size_t offsetSize);

    void RecordAssembleOperation(const std::shared_ptr<LogicalTensor>& input,
                                 const std::shared_ptr<LogicalTensor>& output, const std::vector<int64_t>& offset,
                                 const std::vector<SymbolicScalar>& dynOffset, const ir::Span& span,
                                 const Operation::ScopeInfo& scopeInfo, const std::string& rmwModeAttr, Opcode opcode,
                                 bool atomicFromReduceAcc, bool atomicFromExplicitRmw);

    // Common methods
    Status Initialize();
    Status BuildConsumerCache(Function& function);
    const ConsumerCacheEntry& BuildTensorConsumerCache(Function& function, const LogicalTensorPtr& tensor);
    const ConsumerCacheEntry& GetConsumers(const Operation& operation) const;
    static ir::Span GetFirstSpan(const std::vector<Operation*>& chain);
    static Operation::ScopeInfo GetChainScopeInfo(const std::vector<Operation*>& chain);

    // Processing methods
    Status ProcessOperations(Function& function);

    // Operation appending methods
    Status AppendMergedViewOperations(Function& function);
    Status AppendMergedAssembleOperations(Function& function);

    // Cleanup methods
    Status CleanUp(Function& function);
    Status EraseRedundantAssemble(Function& function) const;
    std::unordered_set<int> visitedOp_;
    std::unordered_map<int, const ConsumerCacheEntry*> consumerCache_;
    std::unordered_map<int, ConsumerCacheEntry> tensorConsumerCache_;
    std::vector<Operation*> candidateOps_;
    std::vector<ViewOp> viewOpToAppend_;
    std::vector<AssembleOp> assembleOpToAppend_;
    IRBuilder irBuilder_;
};
} // namespace npu::tile_fwk
#endif // PASS_MERGE_VIEW_ASSEMBLE_LEGAL_H_
