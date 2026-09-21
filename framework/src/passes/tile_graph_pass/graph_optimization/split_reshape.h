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
 * \file split_reshape.h
 * \brief
 */

#ifndef PASS_SPLIT_RESHAPE_H_
#define PASS_SPLIT_RESHAPE_H_

#include <unordered_set>
#include "interface/function/function.h"
#include "interface/interpreter/function.h"
#include "interface/tensor/irbuilder.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/configs/config_manager.h"
#include "passes/pass_utils/pass_common_defs.h"
#include "passes/pass_utils/dead_operation_eliminate.h"
#include "passes/pass_interface/pass.h"
#include "passes/pass_check/split_reshape_checker.h"

namespace npu::tile_fwk {
using InputMaigc = int;
using OutputMaigc = int;
using OverlaprawMagic = int;

struct UpdatePara {
    int64_t ShapeVal;
    int64_t OffsetVal;
    int64_t stride;
};

struct ReshapeTilePara {
    std::vector<int64_t> shape;
    std::vector<int64_t> newShape;
    std::vector<int64_t> tileOffset;
    std::vector<int64_t> tileShape;
};

struct CheckParam {
    LogicalTensorPtr input;
    LogicalTensorPtr output;
    LogicalTensorPtr inputView;
};

struct copyOutTilePara {
    LogicalTensorPtr reshapeSource;
    int reshapeOpMagic;
    LogicalTensorPtr inputView;
    LogicalTensorPtr newInputView;
    std::vector<int64_t> alignedShape;
};

struct PerfectlyMatchPara {
    LogicalTensorPtr input;
    LogicalTensorPtr output;
    LogicalTensorPtr overlap;
    LogicalTensorPtr reshapeSource;
    LogicalTensorPtr reshapeOutput;
    std::vector<SymbolicScalar> viewDynShape;
};

struct BeCoveredPara {
    LogicalTensorPtr overlap;
    LogicalTensorPtr input;
    LogicalTensorPtr reshapeOutput;
    LogicalTensorPtr reshapeSource;
    std::vector<int64_t> newOffset;
    std::vector<SymbolicScalar> viewDynShape;
};

struct PerfectlyMatchWithAllPara {
    LogicalTensorPtr input;
    LogicalTensorPtr output;
    LogicalTensorPtr overlap;
    LogicalTensorPtr reshapeOutput;
    LogicalTensorPtr newReshapeSource;
    std::vector<SymbolicScalar> viewDynShape;
    std::vector<SymbolicScalar> localInputDynShape;
};

struct AssemblePara {
    LogicalTensorPtr input;
    LogicalTensorPtr output;
    LogicalTensorPtr reshapeSource;
    LogicalTensorPtr newInput;
    LogicalTensorPtr newReshapeOutput;
    LogicalTensorPtr inputView;
    LogicalTensorPtr overlap;
    std::vector<int64_t> newReshapeOutputTileOffset;
};

struct OpPara {
    LogicalTensorPtr oldInput;
    LogicalTensorPtr oldOutput;
    LogicalTensorPtr newInput;
    LogicalTensorPtr newOutput;
    std::vector<SymbolicScalar> viewDynShape;
};

struct CalcOverlapPara {
    std::vector<int64_t> alignedShape;
    LogicalTensorPtr reshapeSource;
    std::vector<int64_t> newInputViewTileOffset;
    std::vector<int64_t> newInputViewTileShape;
    std::vector<SymbolicScalar> oriViewDynShape;
    // 原始 copy-out source，保留 reshape 前的 offset/shape，用于后续回填 assemble 和改图。
    LogicalTensors overlaps;
    // overlaps 映射到 aligned/reshape 后坐标系里的 tile，用于 overlap 判断和 tile 推导。
    LogicalTensors newOverlaps;
    LogicalTensorPtr input;
    LogicalTensorPtr inputView;
    LogicalTensorPtr output;
};

struct CheckOutputParam {
    LogicalTensorPtr reshapeSource;
    std::vector<int64_t> alignedShape;
    std::vector<int64_t> newInputViewTileOffset;
    std::vector<int64_t> newInputViewTileShape;
    std::vector<SymbolicScalar> curViewDynShape;
};

struct ReshapeSourcePara {
    std::vector<int64_t> newReshapeSourceTileShape;
    std::vector<int64_t> newReshapeSourceTileOffset;
};

struct AlignResult {
    Status st = SUCCESS;
    LogicalTensorPtr newCopyOutSource;
};

class SplitReshape : public Pass, public DeadOperationEliminator {
public:
    SplitReshape() : Pass("SplitReshape") {}
    ~SplitReshape() override = default;

private:
    Status RunOnFunction(Function& function) override;
    Status Init();
    Status CollectCopyOut(Function& function);
    Status CollectReshapeInfo(const Operation& op);
    Status CollectAssembleInfo(const Operation& op, const std::unordered_set<int64_t>& multiLogicalTensorRawMagics);
    void DeduplicateCopySources();
    Status CheckCopyIn(Function& function);
    Status AddOperation(Function& function);
    Status SetMemoryType(Function& function);

    Status ObtainReshapeSource(Function& function, const OpPara& para, LogicalTensorPtr& newReshapeSource);
    Status ObtainCopyOutTile(Function& function, const copyOutTilePara& copyOutTile, LogicalTensors& overlaps,
                             LogicalTensors& newOverlaps, OverlapStatus& overlapStatus);
    Status ConstructShapeOffset(const ReshapeTilePara& shapePara, size_t& i, size_t j, std::vector<int64_t>& newOffset,
                                std::vector<int64_t>& newShape);

    Status CalcTileInfo(const CalcOverlapPara& para, std::vector<int64_t>& newShape, std::vector<int64_t>& newOffset,
                        std::vector<int64_t>& reshapeTileShape, std::vector<int64_t>& reshapeTileOffset);
    Status CheckReshapeSkip(const LogicalTensorPtr& input, const LogicalTensorPtr& output,
                            CheckOutputParam& checkOutputParam);
    Status CheckValidOp(const CheckParam& para, CheckOutputParam& checkOutputParam);
    Status CheckOp(Function& function, Operation& op);
    Status UpdateReshapeOp(Function& function, Operation& op, const OverlapStatus& status,
                           const CalcOverlapPara& calcpara);
    Status ProcessPerfectlyMatch(Function& function, Operation& op, const PerfectlyMatchPara& para);
    Status ProcessOnetoOne(Function& function, Operation& op, const CalcOverlapPara& para);
    Status ProcessBeCovered(Function& function, Operation& op, const BeCoveredPara& para);
    Status ProcessOnetoMulti(Function& function, Operation& op, const CalcOverlapPara& para);
    Status ProcessPerfectlyMatchWithAll(Function& function, Operation& op, const PerfectlyMatchWithAllPara& para);
    Status UpdateForPerfectlyMatchWithAll(Function& function, Operation& op, const CalcOverlapPara& para,
                                          const ReshapeSourcePara& sourcePara);
    Status ProcessMultitoOne(Function& function, Operation& op, const CalcOverlapPara& para);
    Status AddReshapeRawInputs(const int overlapRawMagic, const LogicalTensorPtr overlap);

    bool CheckSameRawInput(const LogicalTensorPtr& reshapeSource);
    std::shared_ptr<ReshapeOp> ReshapeOperationExist(const std::shared_ptr<ReshapeOp>& isAddReshapeop);
    unsigned long ComputeReshapeHash(const LogicalTensorPtr& input, const LogicalTensorPtr& output) const;
    unsigned long ComputeReshapeHashOrderless(const LogicalTensorPtr& input, const LogicalTensorPtr& output) const;
    std::vector<int64_t> ObtainMapOffset(const LogicalTensorPtr& input, const LogicalTensorPtr& output) const;
    bool HasAssembleLikeInputProducedByReduceAcc(const LogicalTensorPtr& reshapeSource, int& assembleLikeOpMagic) const;

    Status AddAssembleOp(const MemoryType& memoryType, const std::vector<int64_t>& outputOffset,
                         const LogicalTensorPtr& input, const LogicalTensorPtr& output, const Operation* originOp);
    Status GetAssembleDynShape(const LogicalTensorPtr& input, const LogicalTensorPtr& output,
                               const std::vector<int64_t>& toOffset, std::vector<SymbolicScalar>& dynValidShape);
    Status GetReshapeDynShape(const std::shared_ptr<ReshapeOp>& op, std::vector<SymbolicScalar>& dynValidShape);
    Status GroupReshapeOffset(const std::shared_ptr<ReshapeOp>& isAddReshapeop, const std::vector<int64_t>& offset);
    Status UpdateDynShape(const std::shared_ptr<ReshapeOp>& reshapeOp,
                          const std::vector<SymbolicScalar>& localInputDynShape = {});
    bool InferDynFirstAxisMergeShape(const std::shared_ptr<ReshapeOp>& reshapeOp,
                                     const std::vector<SymbolicScalar>& localInputDynShape,
                                     std::vector<SymbolicScalar>& localOutputDynShape) const;
    Status UpdateReshapeOutputDynOffset(const LogicalTensorPtr& reshapeOutput,
                                        const ViewOpAttribute& viewOpAttribute) const;
    Status ObtainChangingAxis(std::vector<int64_t> alignedShape, std::vector<int64_t> input,
                              std::vector<bool>& ChangingAxis);
    Status CheckDynStatus(std::vector<int64_t> alignedShape, std::vector<int64_t> input, std::vector<int64_t> output,
                          std::vector<SymbolicScalar> dynOutput);
    Status CheckDynOutputAlignment(const std::vector<int64_t>& alignedShape, const std::vector<int64_t>& output,
                                   const std::vector<SymbolicScalar>& dynOutput, const std::vector<bool>& changingAxis);
    Status UpdateShapeOffset(UpdatePara& para, bool& flag, int& currentShape, int& currentOffset);
    Status ShapeAlign(std::vector<int64_t> shape1, std::vector<int64_t> shape2, std::vector<int64_t>& alignedShape);
    Status RawToAlign(const ReshapeTilePara& shapePara, std::vector<int64_t>& newOffset,
                      std::vector<int64_t>& newShape);
    Status AlignToRaw(const ReshapeTilePara& shapePara, std::vector<int64_t>& newOffset,
                      std::vector<int64_t>& newShape);

    Status DefaultEnabledPreCheck(Function& function) override;
    SplitReshapeChecker checker_;

    std::unordered_map<int, std::vector<LogicalTensorPtr>> assembleOutToInput_;
    std::unordered_map<std::pair<int, int>, std::vector<int64_t>, PairHash> mapOffset_;
    std::unordered_map<std::pair<int, int>, int, PairHash> mapAssembleOpMagic_;
    std::unordered_map<int, LogicalTensorPtr> reshapeSources_;
    std::unordered_set<int> groupReshapeNoSplitRawMagics_;
    std::unordered_map<int, std::vector<SymbolicScalar>> reshapeDynOutput_;
    std::vector<AssembleOp> assembles_;
    std::unordered_map<unsigned long, std::shared_ptr<ReshapeOp>> reshapes_;
    std::unordered_map<std::shared_ptr<ReshapeOp>, std::vector<int64_t>> viewOffset_;
    std::unordered_map<LogicalTensorPtr, std::vector<int64_t>> reshapeOffset_;
    std::unordered_map<OverlaprawMagic, std::shared_ptr<RawTensor>> reshapeRawOutputs_;
    std::unordered_map<OverlaprawMagic, std::shared_ptr<RawTensor>> reshapeRawInputs_;
    IRBuilder irBuilder_;
    // 记录所有op_reshape的指针，键值为reshape的输出Operand的magic。
    std::unordered_map<int, const Operation*> reshapeOpPtrs_;
    // 记录assemble输入Operand和输出Operand对应的op_assemble指针。
    std::unordered_map<std::pair<int, int>, const Operation*, PairHash> assembleOpPtrs_;
    std::unordered_map<std::pair<int, int>, AlignResult, PairHash> rawToAlignCache_;
    std::unordered_map<LogicalTensorPtr, bool> sameRawInputCache_;
};

} // namespace npu::tile_fwk
#endif // PASS_SPLIT_RESHAPE_H_
