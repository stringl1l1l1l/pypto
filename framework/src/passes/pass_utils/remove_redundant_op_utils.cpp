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
 * \file remove_redundant_op_utils.cpp
 * \brief utils for redundant view/assemble and slice/contract elimination
 */

#include "remove_redundant_op_utils.h"

#include <algorithm>
#include <climits>
#include <limits>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "interface/operation/attribute.h"
#include "interface/tensor/tensor_offset.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/dead_operation_eliminate.h"
#include "passes/pass_utils/pass_operation_utils.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/pass_utils/remove_redundant_op_internal.h"

#define MODULE_NAME "RemoveRedundantOpUtils"

namespace npu {
namespace tile_fwk {
namespace {
bool IsViewLikeOpcode(Opcode opcode) { return opcode == Opcode::OP_VIEW || opcode == Opcode::OP_SLICE; }

bool IsAssembleLikeOpcode(Opcode opcode) { return opcode == Opcode::OP_ASSEMBLE || opcode == Opcode::OP_CONTRACT; }

bool IsMatmulOpcode(Opcode opcode)
{
    static const std::unordered_set<Opcode> kMatmulOps = {Opcode::OP_A_MUL_B,  Opcode::OP_A_MULACC_B,
                                                          Opcode::OP_A_MUL_BT, Opcode::OP_A_MULACC_BT,
                                                          Opcode::OP_AT_MUL_B, Opcode::OP_AT_MUL_BT};
    return kMatmulOps.count(opcode) > 0;
}

Operation* GetSingleProducer(const LogicalTensorPtr& tensor)
{
    if (tensor == nullptr || tensor->GetProducers().size() != 1) {
        return nullptr;
    }
    return *tensor->GetProducers().begin();
}

bool IsMatmulBackedContractInput(const LogicalTensorPtr& contractInput)
{
    auto* inputProducer = GetSingleProducer(contractInput);
    if (inputProducer == nullptr) {
        return false;
    }
    if (IsMatmulOpcode(inputProducer->GetOpcode())) {
        return true;
    }
    if (inputProducer->GetOpcode() != Opcode::OP_SLICE || inputProducer->GetIOperands().size() != 1) {
        return false;
    }
    auto* materializeOp = GetSingleProducer(inputProducer->GetIOperands().front());
    if (materializeOp == nullptr || materializeOp->GetOpcode() != Opcode::OP_CONTRACT ||
        materializeOp->GetIOperands().size() != 1) {
        return false;
    }
    auto* sourceOp = GetSingleProducer(materializeOp->GetIOperands().front());
    return sourceOp != nullptr && IsMatmulOpcode(sourceOp->GetOpcode());
}

bool IsSupportedViewAssembleCascade(Opcode viewOpcode, Opcode assembleOpcode)
{
    return (viewOpcode == Opcode::OP_VIEW && assembleOpcode == Opcode::OP_ASSEMBLE) ||
           (viewOpcode == Opcode::OP_SLICE && assembleOpcode == Opcode::OP_CONTRACT);
}

bool IsLegacyViewAssembleCascade(Opcode viewOpcode, Opcode assembleOpcode)
{
    return viewOpcode == Opcode::OP_VIEW && assembleOpcode == Opcode::OP_ASSEMBLE;
}

bool IsZeroOffset(const std::vector<int64_t>& offset)
{
    return std::all_of(offset.begin(), offset.end(), [](int64_t value) { return value == 0; });
}

bool IsZeroDynOffset(const std::vector<SymbolicScalar>& dynOffset)
{
    return std::all_of(dynOffset.begin(), dynOffset.end(),
                       [](const SymbolicScalar& value) { return value.ConcreteValid() && value.Concrete() == 0; });
}

bool IsZeroTensorOffset(const std::vector<int64_t>& offset, const std::vector<SymbolicScalar>& dynOffset)
{
    return IsZeroOffset(offset) && (dynOffset.empty() || IsZeroDynOffset(dynOffset));
}

bool IsInitialDynOffset(const std::vector<SymbolicScalar>& dynOffset)
{
    return std::all_of(dynOffset.begin(), dynOffset.end(), [](const SymbolicScalar& value) {
        return value.ConcreteValid() && value.Concrete() == INT_MAX;
    });
}

std::shared_ptr<ViewOpAttribute> GetViewAttr(const Operation& op);
std::shared_ptr<AssembleOpAttribute> GetAssembleAttr(const Operation& op);

struct RegionMapping {
    std::vector<int64_t> sourceOffset;
    std::vector<int64_t> targetOffset;
    std::vector<int64_t> shape;
    std::vector<int64_t> translation;
    bool hasDynOffset{false};
};

bool IsConcreteDynOffsetConsistent(const std::vector<int64_t>& staticOffset,
                                   const std::vector<SymbolicScalar>& dynOffset)
{
    if (dynOffset.empty()) {
        return true;
    }
    if (staticOffset.size() != dynOffset.size()) {
        return false;
    }
    for (size_t idx = 0; idx < staticOffset.size(); ++idx) {
        auto simplified = dynOffset[idx].Simplify();
        if (!simplified.ConcreteValid() || simplified.Concrete() != staticOffset[idx]) {
            return false;
        }
    }
    return true;
}

bool IsConcreteDynValidShapeWithinShape(const std::vector<int64_t>& shape,
                                        const std::vector<SymbolicScalar>& dynValidShape)
{
    if (dynValidShape.empty()) {
        return true;
    }
    if (shape.size() != dynValidShape.size()) {
        return false;
    }
    for (size_t idx = 0; idx < shape.size(); ++idx) {
        auto simplified = dynValidShape[idx].Simplify();
        if (shape[idx] <= 0 || !simplified.ConcreteValid() || simplified.Concrete() <= 0 ||
            simplified.Concrete() > shape[idx]) {
            return false;
        }
    }
    return true;
}

// 获取 tensor 的 concrete valid shape（真实搬运 extent）。
// 若 valid shape 为动态符号（如 view 的 len），说明该 copy 是部分搬运，无法安全与 assemble 合并。
bool GetConcreteValidShape(const LogicalTensorPtr& tensor, std::vector<int64_t>& validShape)
{
    if (tensor == nullptr) {
        return false;
    }
    const auto& dynValidShape = tensor->GetDynValidShape();
    validShape.clear();
    if (dynValidShape.empty()) {
        validShape = tensor->GetShape();
        return !validShape.empty();
    }
    if (dynValidShape.size() != tensor->GetShape().size()) {
        return false;
    }
    validShape.reserve(dynValidShape.size());
    for (const auto& dim : dynValidShape) {
        auto simplified = dim.Simplify();
        if (!simplified.ConcreteValid()) {
            return false;
        }
        validShape.push_back(simplified.Concrete());
    }
    return true;
}

bool IsTensorOffset32BAligned(const LogicalTensorPtr& tensor)
{
    constexpr int64_t kOffsetAlignmentBytes = 32;
    if (tensor == nullptr || tensor->GetRawTensor() == nullptr) {
        return false;
    }
    const auto& rawShape = tensor->GetRawTensor()->GetRawShape();
    const auto& offset = tensor->GetOffset();
    if (rawShape.empty() || rawShape.size() != offset.size() ||
        !IsConcreteDynOffsetConsistent(offset, tensor->GetDynOffset())) {
        return false;
    }
    const auto dataSize = static_cast<int64_t>(BytesOf(tensor->Datatype()));
    if (dataSize <= 0) {
        return false;
    }

    int64_t linearOffsetModulo = 0;
    for (size_t idx = 0; idx < rawShape.size(); ++idx) {
        if (rawShape[idx] <= 0 || offset[idx] < 0 || offset[idx] >= rawShape[idx]) {
            return false;
        }
        linearOffsetModulo = (linearOffsetModulo * (rawShape[idx] % kOffsetAlignmentBytes) + offset[idx]) %
                             kOffsetAlignmentBytes;
    }
    return (linearOffsetModulo * dataSize) % kOffsetAlignmentBytes == 0;
}

// VEC 指令要求 UB 操作数地址 32B 对齐。contract-slice 对消会用偏移 view 直接读取上游共享
// buffer，当 view 的 fromOffset 按 view 输入的连续布局线性化后非 32B 对齐时（如 fp16 输入上
// 尾轴偏移 8 元素 = 16B），后续视图折叠会让 VEC consumer 产生非对齐访问（aicore errcode
// 0x800）。此时应保留 CONTRACT+SLICE 拷贝路径（SLICE 输出为独立对齐 buffer），不做对消。
bool IsViewFromOffset32BAligned(const LogicalTensorPtr& viewInput, const std::vector<int64_t>& fromOffset)
{
    if (viewInput == nullptr || fromOffset.size() != viewInput->GetShape().size()) {
        return false;
    }
    const auto elemBytes = static_cast<int64_t>(BytesOf(viewInput->Datatype()));
    if (elemBytes <= 0) {
        return false;
    }
    constexpr int64_t kOffsetAlignmentBytes = 32;
    if (elemBytes % kOffsetAlignmentBytes == 0) {
        return true;
    }
    const auto& shape = viewInput->GetShape();
    int64_t linearOffsetModulo = 0;
    int64_t strideModulo = elemBytes % kOffsetAlignmentBytes;
    for (size_t dim = fromOffset.size(); dim-- > 0;) {
        if (fromOffset[dim] < 0) {
            return false;
        }
        linearOffsetModulo = (linearOffsetModulo + (fromOffset[dim] % kOffsetAlignmentBytes) * strideModulo) %
                             kOffsetAlignmentBytes;
        if (dim > 0) {
            if (shape[dim] <= 0) {
                return false;
            }
            strideModulo = (strideModulo * (shape[dim] % kOffsetAlignmentBytes)) % kOffsetAlignmentBytes;
            if (strideModulo == 0) {
                break;
            }
        }
    }
    return linearOffsetModulo == 0;
}

// 带动态偏移的版本：动态符号可化简为常量且与静态值一致时按静态值判定；真动态（无法化简）
// 或 static/dyn 不一致时保守处理——真动态保持既有行为放行，不一致视为不可信直接拦截。
bool IsViewFromOffset32BAligned(const LogicalTensorPtr& viewInput, const std::vector<int64_t>& fromOffset,
                                const std::vector<SymbolicScalar>& fromDynOffset)
{
    if (!fromDynOffset.empty()) {
        if (fromDynOffset.size() != fromOffset.size()) {
            return false;
        }
        for (size_t idx = 0; idx < fromDynOffset.size(); ++idx) {
            auto simplified = fromDynOffset[idx].Simplify();
            if (simplified.ConcreteValid() && simplified.Concrete() != fromOffset[idx]) {
                return false;
            }
        }
    }
    return IsViewFromOffset32BAligned(viewInput, fromOffset);
}

bool IsRegionWithinShape(const std::vector<int64_t>& offset, const std::vector<int64_t>& shape,
                         const std::vector<int64_t>& outerShape)
{
    if (offset.size() != shape.size() || shape.size() != outerShape.size()) {
        return false;
    }
    for (size_t idx = 0; idx < shape.size(); ++idx) {
        if (offset[idx] < 0 || shape[idx] <= 0 || outerShape[idx] <= 0 || offset[idx] > outerShape[idx] - shape[idx]) {
            return false;
        }
    }
    return true;
}

bool RegionsOverlap(const RegionMapping& lhs, const RegionMapping& rhs)
{
    for (size_t idx = 0; idx < lhs.shape.size(); ++idx) {
        if (lhs.targetOffset[idx] + lhs.shape[idx] <= rhs.targetOffset[idx] ||
            rhs.targetOffset[idx] + rhs.shape[idx] <= lhs.targetOffset[idx]) {
            return false;
        }
    }
    return true;
}

bool TryCalculateVolume(const std::vector<int64_t>& shape, uint64_t& volume)
{
    volume = 1;
    for (auto dim : shape) {
        if (dim <= 0 || volume > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dim)) {
            return false;
        }
        volume *= static_cast<uint64_t>(dim);
    }
    return true;
}

bool AreTranslatedEndpointsCompatible(const LogicalTensorPtr& startTensor, const LogicalTensorPtr& endTensor)
{
    return startTensor != nullptr && endTensor != nullptr &&
           startTensor->GetShape().size() == endTensor->GetShape().size() &&
           startTensor->Datatype() == endTensor->Datatype() && startTensor->Format() == endTensor->Format() &&
           IsConcreteDynValidShapeWithinShape(startTensor->GetShape(), startTensor->GetDynValidShape()) &&
           IsConcreteDynValidShapeWithinShape(endTensor->GetShape(), endTensor->GetDynValidShape());
}

bool HasConsumerOutsideViewAssembleCascade(const LogicalTensorPtr& startTensor, const LogicalTensorPtr& endTensor)
{
    return std::any_of(startTensor->GetConsumers().begin(), startTensor->GetConsumers().end(),
                       [&endTensor](const Operation* startConsumer) {
                           if (startConsumer == nullptr || !IsViewLikeOpcode(startConsumer->GetOpcode())) {
                               return true;
                           }
                           const auto consumerOps = startConsumer->ConsumerOps();
                           return consumerOps.empty() ||
                                  std::any_of(consumerOps.begin(), consumerOps.end(),
                                              [startConsumer, &endTensor](const Operation* endProducer) {
                                                  return endProducer == nullptr ||
                                                         endProducer->GetOOperands().empty() ||
                                                         endProducer->GetOOperands().front() != endTensor ||
                                                         !IsSupportedViewAssembleCascade(startConsumer->GetOpcode(),
                                                                                         endProducer->GetOpcode());
                                              });
                       });
}

bool GetTranslatedMappingChain(const Operation* assembleOp, const LogicalTensorPtr& startTensor,
                               const LogicalTensorPtr& endTensor, LogicalTensorPtr& tempTensor, Operation*& viewOp)
{
    if (assembleOp == nullptr || !IsAssembleLikeOpcode(assembleOp->GetOpcode()) ||
        assembleOp->GetIOperands().size() != 1 || assembleOp->GetOOperands().empty() ||
        assembleOp->GetOOperands().front() != endTensor) {
        return false;
    }
    tempTensor = assembleOp->GetIOperands().front();
    if (tempTensor == nullptr || tempTensor->GetProducers().size() != 1 ||
        tempTensor->GetShape().size() != startTensor->GetShape().size() ||
        tempTensor->Datatype() != startTensor->Datatype() || tempTensor->Format() != startTensor->Format() ||
        !IsConcreteDynOffsetConsistent(tempTensor->GetShape(), tempTensor->GetDynValidShape())) {
        return false;
    }
    viewOp = *tempTensor->GetProducers().begin();
    return viewOp != nullptr && IsViewLikeOpcode(viewOp->GetOpcode()) &&
           IsSupportedViewAssembleCascade(viewOp->GetOpcode(), assembleOp->GetOpcode()) &&
           viewOp->GetIOperands().size() == 1 && viewOp->GetIOperands().front() == startTensor &&
           !viewOp->GetOOperands().empty() && viewOp->GetOOperands().front() == tempTensor;
}

bool CalculateRegionTranslation(const std::vector<int64_t>& sourceOffset, const std::vector<int64_t>& targetOffset,
                                std::vector<int64_t>& translation)
{
    translation.resize(sourceOffset.size());
    for (size_t idx = 0; idx < sourceOffset.size(); ++idx) {
        if (sourceOffset[idx] < targetOffset[idx]) {
            return false;
        }
        translation[idx] = sourceOffset[idx] - targetOffset[idx];
    }
    return true;
}

bool TryBuildRegionMapping(const Operation* assembleOp, const LogicalTensorPtr& startTensor,
                           const LogicalTensorPtr& endTensor, RegionMapping& mapping)
{
    LogicalTensorPtr tempTensor;
    Operation* viewOp = nullptr;
    if (!GetTranslatedMappingChain(assembleOp, startTensor, endTensor, tempTensor, viewOp)) {
        return false;
    }
    auto viewAttr = GetViewAttr(*viewOp);
    auto assembleAttr = GetAssembleAttr(*assembleOp);
    if (viewAttr == nullptr || assembleAttr == nullptr) {
        return false;
    }

    mapping.sourceOffset = viewAttr->GetFromOffset();
    mapping.targetOffset = assembleAttr->GetToOffset();
    mapping.shape = tempTensor->GetShape();
    const size_t rank = startTensor->GetShape().size();
    if (mapping.sourceOffset.size() != rank || mapping.targetOffset.size() != rank ||
        !IsConcreteDynOffsetConsistent(mapping.sourceOffset, viewAttr->GetFromDynOffset()) ||
        !IsConcreteDynOffsetConsistent(mapping.targetOffset, assembleAttr->GetToDynOffset()) ||
        !IsRegionWithinShape(mapping.sourceOffset, mapping.shape, startTensor->GetShape()) ||
        !IsRegionWithinShape(mapping.targetOffset, mapping.shape, endTensor->GetShape()) ||
        !CalculateRegionTranslation(mapping.sourceOffset, mapping.targetOffset, mapping.translation)) {
        return false;
    }
    mapping.hasDynOffset = !viewAttr->GetFromDynOffset().empty() || !assembleAttr->GetToDynOffset().empty();
    return true;
}

bool CollectRegionMappings(const LogicalTensorPtr& startTensor, const LogicalTensorPtr& endTensor,
                           std::vector<RegionMapping>& mappings, std::vector<int64_t>& commonTranslation,
                           bool& hasDynOffset)
{
    for (auto* assembleOp : endTensor->GetProducers()) {
        RegionMapping mapping;
        if (!TryBuildRegionMapping(assembleOp, startTensor, endTensor, mapping)) {
            return false;
        }
        if (mappings.empty()) {
            commonTranslation = mapping.translation;
        } else if (mapping.translation != commonTranslation) {
            return false;
        }
        hasDynOffset = hasDynOffset || mapping.hasDynOffset;
        mappings.push_back(std::move(mapping));
    }
    return !mappings.empty();
}

bool HasCompleteRegionCoverage(const std::vector<RegionMapping>& mappings, const std::vector<int64_t>& endShape)
{
    uint64_t coveredVolume = 0;
    for (size_t lhs = 0; lhs < mappings.size(); ++lhs) {
        uint64_t regionVolume = 0;
        if (!TryCalculateVolume(mappings[lhs].shape, regionVolume) ||
            coveredVolume > std::numeric_limits<uint64_t>::max() - regionVolume) {
            return false;
        }
        coveredVolume += regionVolume;
        for (size_t rhs = lhs + 1; rhs < mappings.size(); ++rhs) {
            if (RegionsOverlap(mappings[lhs], mappings[rhs])) {
                return false;
            }
        }
    }
    uint64_t endVolume = 0;
    return TryCalculateVolume(endShape, endVolume) && coveredVolume == endVolume;
}

bool TryCalculateTranslatedViewOffset(const LogicalTensorPtr& startTensor, const LogicalTensorPtr& endTensor,
                                      std::vector<int64_t>& newOffset, std::vector<SymbolicScalar>& newDynOffset)
{
    if (!AreTranslatedEndpointsCompatible(startTensor, endTensor)) {
        return false;
    }
    std::vector<RegionMapping> mappings;
    std::vector<int64_t> commonTranslation;
    bool hasDynOffset = false;
    if (!CollectRegionMappings(startTensor, endTensor, mappings, commonTranslation, hasDynOffset) ||
        IsZeroOffset(commonTranslation) ||
        !IsRegionWithinShape(commonTranslation, endTensor->GetShape(), startTensor->GetShape()) ||
        !HasCompleteRegionCoverage(mappings, endTensor->GetShape())) {
        return false;
    }

    newOffset = commonTranslation;
    newDynOffset = hasDynOffset ? SymbolicScalar::FromConcrete(commonTranslation) : std::vector<SymbolicScalar>{};
    return true;
}

std::shared_ptr<ViewOpAttribute> GetViewAttr(const Operation& op)
{
    return std::dynamic_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
}

std::shared_ptr<AssembleOpAttribute> GetAssembleAttr(const Operation& op)
{
    return std::dynamic_pointer_cast<AssembleOpAttribute>(op.GetOpAttribute());
}

bool GetConcreteImmediateValues(const std::vector<OpImmediate>& values, std::vector<int64_t>& result)
{
    result.clear();
    result.reserve(values.size());
    for (const auto& value : values) {
        if (!value.IsSpecified()) {
            return false;
        }
        auto simplified = value.GetSpecifiedValue().Simplify();
        if (!simplified.ConcreteValid()) {
            return false;
        }
        result.push_back(simplified.Concrete());
    }
    return true;
}

bool GetSpecifiedImmediateValues(const std::vector<OpImmediate>& values, std::vector<SymbolicScalar>& result)
{
    result.clear();
    result.reserve(values.size());
    for (const auto& value : values) {
        if (!value.IsSpecified()) {
            return false;
        }
        result.push_back(value.GetSpecifiedValue().Simplify());
    }
    return true;
}

bool IsTensorZeroOffset(const LogicalTensorPtr& tensor)
{
    return tensor != nullptr && IsZeroTensorOffset(tensor->GetOffset(), tensor->GetDynOffset());
}

bool HasPossiblyNonZeroTensorOffset(const LogicalTensorPtr& tensor)
{
    if (tensor == nullptr || !IsZeroOffset(tensor->GetOffset())) {
        return tensor != nullptr;
    }
    return std::any_of(tensor->GetDynOffset().begin(), tensor->GetDynOffset().end(), [](const SymbolicScalar& value) {
        auto simplified = value.Simplify();
        return !simplified.ConcreteValid() || simplified.Concrete() != 0;
    });
}

bool IsSameTensorTypeAndRank(const LogicalTensorPtr& lhs, const LogicalTensorPtr& rhs)
{
    return lhs != nullptr && rhs != nullptr && lhs->GetShape().size() == rhs->GetShape().size() &&
           lhs->Datatype() == rhs->Datatype() && lhs->Format() == rhs->Format();
}

bool IsOffsetEqualToSum(const std::vector<SymbolicScalar>& actual, const std::vector<SymbolicScalar>& base,
                        const std::vector<int64_t>& delta)
{
    if (actual.size() != base.size() || base.size() != delta.size()) {
        return false;
    }
    for (size_t idx = 0; idx < actual.size(); ++idx) {
        auto difference = (actual[idx] - base[idx]).Simplify();
        if (!difference.ConcreteValid() || difference.Concrete() != delta[idx]) {
            return false;
        }
    }
    return true;
}

bool IsCopyShapeCompatible(const CopyOpAttribute& copyAttr, const LogicalTensorPtr& copiedTensor, bool isCopyOut)
{
    if (copiedTensor == nullptr || copiedTensor->GetRawTensor() == nullptr) {
        return false;
    }
    std::vector<int64_t> copyShape;
    if (!GetConcreteImmediateValues(copyAttr.GetShape(), copyShape) || copyShape != copiedTensor->GetShape() ||
        copyAttr.GetRawShape().size() != copiedTensor->GetRawTensor()->GetDynRawShape().size()) {
        return false;
    }
    const auto& validShape = isCopyOut ? copyAttr.GetFromDynValidShape() : copyAttr.GetToDynValidShape();
    return validShape.empty() || validShape.size() == copiedTensor->GetShape().size();
}

struct ViewCopyoutBranch {
    Operation* viewOp{nullptr};
    Operation* copyOutOp{nullptr};
    LogicalTensorPtr viewOutput;
    std::shared_ptr<ViewOpAttribute> viewAttr;
    std::shared_ptr<CopyOpAttribute> copyAttr;
    std::vector<int64_t> viewOffset;
    std::vector<SymbolicScalar> copyOffset;
};

bool BuildViewCopyoutBranch(const LogicalTensorPtr& viewInput, const LogicalTensorPtr& copyOutOutput, Operation& viewOp,
                            Operation& copyOutOp, ViewCopyoutBranch& branch)
{
    if (viewInput == nullptr || copyOutOutput == nullptr || viewOp.GetOpcode() != Opcode::OP_VIEW ||
        viewOp.GetIOperands().size() != 1 || viewOp.GetOOperands().size() != 1 ||
        viewOp.GetIOperands().front() != viewInput || copyOutOp.GetIOperands().size() != 1 ||
        copyOutOp.GetOOperands().size() != 1 || copyOutOp.GetOOperands().front() != copyOutOutput ||
        !OpcodeManager::Inst().IsCopyOut(copyOutOp.GetOpcode())) {
        return false;
    }
    auto viewOutput = viewOp.GetOOperands().front();
    if (viewOutput == nullptr || copyOutOp.GetIOperands().front() != viewOutput ||
        !IsSameTensorTypeAndRank(viewInput, viewOutput) || !IsSameTensorTypeAndRank(viewInput, copyOutOutput) ||
        viewInput->GetMemoryTypeOriginal() != viewOutput->GetMemoryTypeOriginal() ||
        copyOutOutput->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR ||
        viewOp.GetSubgraphID() != copyOutOp.GetSubgraphID()) {
        return false;
    }
    auto viewAttr = GetViewAttr(viewOp);
    auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(copyOutOp.GetOpAttribute());
    if (viewAttr == nullptr || copyAttr == nullptr || !copyAttr->IsCopyOut() ||
        copyAttr->GetCopyOutAttr().first != viewInput->GetMemoryTypeOriginal() ||
        (viewAttr->GetTo() != MemoryType::MEM_UNKNOWN && viewAttr->GetTo() != viewOutput->GetMemoryTypeOriginal()) ||
        !IsConcreteDynOffsetConsistent(viewAttr->GetFromOffset(), viewAttr->GetFromDynOffset()) ||
        !GetSpecifiedImmediateValues(copyAttr->GetToOffset(), branch.copyOffset) ||
        !IsCopyShapeCompatible(*copyAttr, viewOutput, true)) {
        return false;
    }
    branch.viewOffset = viewAttr->GetFromOffset();
    branch.viewOp = &viewOp;
    branch.copyOutOp = &copyOutOp;
    branch.viewOutput = viewOutput;
    branch.viewAttr = std::move(viewAttr);
    branch.copyAttr = std::move(copyAttr);
    return true;
}

bool CanMergeViewCopyoutGroup(const LogicalTensorPtr& viewInput, const LogicalTensorPtr& copyOutOutput,
                              const std::vector<std::pair<Operation*, Operation*>>& candidates,
                              std::vector<ViewCopyoutBranch>& branches, size_t& retainedIndex)
{
    if (!IsTensorZeroOffset(viewInput) || candidates.empty()) {
        return false;
    }
    std::vector<RegionMapping> mappings;
    int zeroOffsetCount = 0;
    int groupSubgraph = candidates.front().first->GetSubgraphID();
    Opcode copyOpcode = candidates.front().second->GetOpcode();
    for (const auto& [viewOp, copyOutOp] : candidates) {
        ViewCopyoutBranch branch;
        if (viewOp == nullptr || copyOutOp == nullptr || viewOp->GetSubgraphID() != groupSubgraph ||
            copyOutOp->GetSubgraphID() != groupSubgraph || copyOutOp->GetOpcode() != copyOpcode ||
            !BuildViewCopyoutBranch(viewInput, copyOutOutput, *viewOp, *copyOutOp, branch) ||
            !IsRegionWithinShape(branch.viewOffset, branch.viewOutput->GetShape(), viewInput->GetShape())) {
            return false;
        }
        if (IsZeroOffset(branch.viewOffset)) {
            retainedIndex = branches.size();
            ++zeroOffsetCount;
        }
        RegionMapping mapping;
        mapping.targetOffset = branch.viewOffset;
        mapping.shape = branch.viewOutput->GetShape();
        mappings.push_back(std::move(mapping));
        branches.push_back(std::move(branch));
    }
    if (zeroOffsetCount != 1 || !HasCompleteRegionCoverage(mappings, viewInput->GetShape())) {
        return false;
    }
    const auto& retainedToOffset = branches[retainedIndex].copyOffset;
    return std::all_of(branches.begin(), branches.end(), [&retainedToOffset](const ViewCopyoutBranch& branch) {
        return IsOffsetEqualToSum(branch.copyOffset, retainedToOffset, branch.viewOffset);
    });
}

void MergeViewCopyoutGroup(const LogicalTensorPtr& viewInput, std::vector<ViewCopyoutBranch>& branches,
                           size_t retainedIndex, bool& operationUpdated)
{
    auto& retained = branches[retainedIndex];
    retained.copyOutOp->ReplaceIOperand(0, viewInput);
    retained.copyAttr->SetShape(OpImmediate::Specified(viewInput->GetShape()));
    // rawShape describes the destination GM layout. Keep the retained copy-out value unchanged.
    retained.copyAttr->SetFromDynValidShape(OpImmediate::Specified(viewInput->GetDynValidShape()));
    for (size_t idx = 0; idx < branches.size(); ++idx) {
        branches[idx].viewOp->SetAsDeleted();
        if (idx != retainedIndex) {
            branches[idx].copyOutOp->SetAsDeleted();
        }
    }
    operationUpdated = true;
}

struct CopyinAssembleBranch {
    Operation* copyInOp{nullptr};
    Operation* assembleOp{nullptr};
    LogicalTensorPtr copyInOutput;
    std::shared_ptr<CopyOpAttribute> copyAttr;
    std::shared_ptr<AssembleOpAttribute> assembleAttr;
    std::vector<SymbolicScalar> copyOffset;
    std::vector<int64_t> assembleOffset;
};

bool BuildCopyinAssembleBranch(const LogicalTensorPtr& copyInInput, const LogicalTensorPtr& assembleOutput,
                               Operation& copyInOp, Operation& assembleOp, CopyinAssembleBranch& branch)
{
    if (copyInInput == nullptr || assembleOutput == nullptr || assembleOp.GetOpcode() != Opcode::OP_ASSEMBLE ||
        assembleOp.GetIOperands().size() != 1 || assembleOp.GetOOperands().size() != 1 ||
        assembleOp.GetOOperands().front() != assembleOutput || copyInOp.GetIOperands().size() != 1 ||
        copyInOp.GetOOperands().size() != 1 || copyInOp.GetIOperands().front() != copyInInput ||
        !OpcodeManager::Inst().IsCopyIn(copyInOp.GetOpcode())) {
        return false;
    }
    auto copyInOutput = copyInOp.GetOOperands().front();
    if (copyInOutput == nullptr || assembleOp.GetIOperands().front() != copyInOutput ||
        !IsSameTensorTypeAndRank(copyInInput, copyInOutput) || !IsSameTensorTypeAndRank(copyInInput, assembleOutput) ||
        copyInInput->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR ||
        copyInOutput->GetMemoryTypeOriginal() != assembleOutput->GetMemoryTypeOriginal() ||
        copyInOp.GetSubgraphID() != assembleOp.GetSubgraphID()) {
        return false;
    }
    auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(copyInOp.GetOpAttribute());
    auto assembleAttr = GetAssembleAttr(assembleOp);
    if (copyAttr == nullptr || assembleAttr == nullptr || copyAttr->IsCopyOut() ||
        copyAttr->GetCopyInAttr().second != assembleOutput->GetMemoryTypeOriginal() ||
        (assembleAttr->GetFrom() != MemoryType::MEM_UNKNOWN &&
         assembleAttr->GetFrom() != copyInOutput->GetMemoryTypeOriginal()) ||
        !GetSpecifiedImmediateValues(copyAttr->GetFromOffset(), branch.copyOffset) ||
        !IsConcreteDynOffsetConsistent(assembleAttr->GetToOffset(), assembleAttr->GetToDynOffset()) ||
        !IsCopyShapeCompatible(*copyAttr, copyInOutput, false)) {
        return false;
    }
    branch.assembleOffset = assembleAttr->GetToOffset();
    branch.copyInOp = &copyInOp;
    branch.assembleOp = &assembleOp;
    branch.copyInOutput = copyInOutput;
    branch.copyAttr = std::move(copyAttr);
    branch.assembleAttr = std::move(assembleAttr);
    return true;
}

bool CanMergeCopyinAssembleGroup(const LogicalTensorPtr& copyInInput, const LogicalTensorPtr& assembleOutput,
                                 const std::vector<std::pair<Operation*, Operation*>>& candidates,
                                 std::vector<CopyinAssembleBranch>& branches, size_t& retainedIndex)
{
    if (candidates.empty()) {
        return false;
    }
    std::vector<RegionMapping> mappings;
    int zeroOffsetCount = 0;
    int groupSubgraph = candidates.front().first->GetSubgraphID();
    Opcode copyOpcode = candidates.front().first->GetOpcode();
    for (const auto& [copyInOp, assembleOp] : candidates) {
        CopyinAssembleBranch branch;
        if (copyInOp == nullptr || assembleOp == nullptr || copyInOp->GetSubgraphID() != groupSubgraph ||
            assembleOp->GetSubgraphID() != groupSubgraph || copyInOp->GetOpcode() != copyOpcode ||
            !BuildCopyinAssembleBranch(copyInInput, assembleOutput, *copyInOp, *assembleOp, branch)) {
            return false;
        }
        std::vector<int64_t> copyValidShape;
        if (!GetConcreteValidShape(branch.copyInOutput, copyValidShape) ||
            !IsRegionWithinShape(branch.assembleOffset, copyValidShape, assembleOutput->GetShape())) {
            return false;
        }
        if (IsZeroOffset(branch.assembleOffset)) {
            retainedIndex = branches.size();
            ++zeroOffsetCount;
        }
        RegionMapping mapping;
        mapping.targetOffset = branch.assembleOffset;
        mapping.shape = std::move(copyValidShape);
        mappings.push_back(std::move(mapping));
        branches.push_back(std::move(branch));
    }
    if (zeroOffsetCount != 1 || !HasCompleteRegionCoverage(mappings, assembleOutput->GetShape())) {
        return false;
    }
    const auto& retainedFromOffset = branches[retainedIndex].copyOffset;
    return std::all_of(branches.begin(), branches.end(), [&retainedFromOffset](const CopyinAssembleBranch& branch) {
        return IsOffsetEqualToSum(branch.copyOffset, retainedFromOffset, branch.assembleOffset);
    });
}

void MergeCopyinAssembleGroup(const LogicalTensorPtr& copyInInput, const LogicalTensorPtr& assembleOutput,
                              std::vector<CopyinAssembleBranch>& branches, size_t retainedIndex, bool& operationUpdated)
{
    auto& retained = branches[retainedIndex];
    retained.copyInOp->ReplaceOOperand(0, assembleOutput);
    retained.copyAttr->SetShape(OpImmediate::Specified(assembleOutput->GetShape()));
    retained.copyAttr->SetRawShape(OpImmediate::Specified(copyInInput->GetRawTensor()->GetDynRawShape()));
    retained.copyAttr->SetToDynValidShape(OpImmediate::Specified(assembleOutput->GetDynValidShape()));
    for (size_t idx = 0; idx < branches.size(); ++idx) {
        branches[idx].assembleOp->SetAsDeleted();
        if (idx != retainedIndex) {
            branches[idx].copyInOp->SetAsDeleted();
        }
    }
    operationUpdated = true;
}

std::string IntVectorToString(const std::vector<int64_t>& values)
{
    std::ostringstream stream;
    stream << "[";
    for (size_t idx = 0; idx < values.size(); ++idx) {
        if (idx != 0) {
            stream << ", ";
        }
        stream << values[idx];
    }
    stream << "]";
    return stream.str();
}

std::string SymbolicVectorToString(const std::vector<SymbolicScalar>& values)
{
    std::ostringstream stream;
    stream << "[";
    for (size_t idx = 0; idx < values.size(); ++idx) {
        if (idx != 0) {
            stream << ", ";
        }
        stream << values[idx].Dump();
    }
    stream << "]";
    return stream.str();
}

void WarnUnresolvedCopyoutOffset(Function& function)
{
    for (auto& op : function.Operations()) {
        if (op.IsDeleted() || !OpcodeManager::Inst().IsCopyOut(op.GetOpcode()) || op.GetIOperands().empty()) {
            continue;
        }
        const auto& input = op.GetIOperands().front();
        if (!HasPossiblyNonZeroTensorOffset(input)) {
            continue;
        }
        const auto rawShape = input->GetRawTensor() == nullptr ? std::vector<int64_t>{} :
                                                                 input->GetRawTensor()->GetRawShape();
        APASS_LOG_WARN_F(Elements::Operation,
                         "CopyOut op[%d] still has an input tensor with offset. shape=%s, rawShape=%s, offset=%s, "
                         "dynOffset=%s. CCE may not handle this offset correctly and could cause a precision issue.",
                         op.GetOpMagic(), IntVectorToString(input->GetShape()).c_str(),
                         IntVectorToString(rawShape).c_str(), IntVectorToString(input->GetOffset()).c_str(),
                         SymbolicVectorToString(input->GetDynOffset()).c_str());
    }
}

bool IsEqualShapeWithDynShape(const LogicalTensorPtr& lhs, const LogicalTensorPtr& rhs)
{
    if (lhs == nullptr || rhs == nullptr || lhs->GetShape() != rhs->GetShape()) {
        return false;
    }
    const auto& lhsDynShape = lhs->GetDynValidShape();
    const auto& rhsDynShape = rhs->GetDynValidShape();
    if (lhsDynShape.empty() && rhsDynShape.empty()) {
        return true;
    }
    if (lhsDynShape.size() != rhsDynShape.size()) {
        return false;
    }
    for (size_t idx = 0; idx < lhsDynShape.size(); ++idx) {
        if (lhsDynShape[idx].Dump() != rhsDynShape[idx].Dump()) {
            return false;
        }
    }
    return true;
}

bool IsSingleZeroOffsetSliceContract(const Operation& sliceOp, const Operation& contractOp)
{
    if (sliceOp.GetOpcode() != Opcode::OP_SLICE || contractOp.GetOpcode() != Opcode::OP_CONTRACT ||
        sliceOp.GetIOperands().size() != 1 || sliceOp.GetOOperands().size() != 1 ||
        contractOp.GetIOperands().size() != 1 || contractOp.GetOOperands().size() != 1 ||
        sliceOp.GetOOperands().front() != contractOp.GetIOperands().front()) {
        return false;
    }

    const auto& sliceOutput = sliceOp.GetOOperands().front();
    const auto& contractOutput = contractOp.GetOOperands().front();
    const auto& sliceConsumers = sliceOutput->GetConsumers();
    const auto& contractOutputProducers = contractOutput->GetProducers();
    if (sliceConsumers.size() != 1 || *sliceConsumers.begin() != &contractOp || contractOutputProducers.size() != 1 ||
        *contractOutputProducers.begin() != &contractOp || !IsEqualShapeWithDynShape(sliceOutput, contractOutput)) {
        return false;
    }

    auto contractAttr = GetAssembleAttr(contractOp);
    return contractAttr != nullptr && IsZeroTensorOffset(contractAttr->GetToOffset(), contractAttr->GetToDynOffset());
}

void GenerateSingleSliceContractView(Function& function, Operation& sliceOp, Operation& contractOp,
                                     const LogicalTensorPtr& startTensor, const LogicalTensorPtr& endTensor,
                                     std::vector<Operation*>& newOps, bool& operationUpdated)
{
    auto sliceAttr = GetViewAttr(sliceOp);
    if (sliceAttr == nullptr) {
        return;
    }

    auto viewAttribute = std::make_shared<ViewOpAttribute>(sliceAttr->GetFromOffset(), sliceAttr->GetFromDynOffset(),
                                                           sliceAttr->GetToDynValidShape());
    viewAttribute->SetToType(endTensor->GetMemoryTypeToBe());
    auto& newViewOp = PassOperationUtils::AddOperation(
        function, Opcode::OP_VIEW, {startTensor}, {endTensor},
        [&viewAttribute](Operation& newOp) { newOp.SetOpAttribute(viewAttribute); });

    sliceOp.SetAsDeleted();
    contractOp.SetAsDeleted();
    newOps.push_back(&newViewOp);
    operationUpdated = true;
}

bool HasMemoryTypeTransform(const LogicalTensorPtr& inputTensor, const LogicalTensorPtr& outputTensor)
{
    if (inputTensor == nullptr || outputTensor == nullptr) {
        return false;
    }
    auto inputMem = inputTensor->GetMemoryTypeOriginal();
    auto outputMem = outputTensor->GetMemoryTypeOriginal();
    if (inputMem != MemoryType::MEM_UNKNOWN && outputMem != MemoryType::MEM_UNKNOWN && inputMem != outputMem) {
        return true;
    }
    auto outputToBe = outputTensor->GetMemoryTypeToBe();
    return inputMem != MemoryType::MEM_UNKNOWN && outputToBe != MemoryType::MEM_UNKNOWN && inputMem != outputToBe;
}

Opcode GetViewLikeReplacementOpcode(const LogicalTensorPtr& inputTensor, const LogicalTensorPtr& outputTensor)
{
    return HasMemoryTypeTransform(inputTensor, outputTensor) ? Opcode::OP_SLICE : Opcode::OP_VIEW;
}

bool CanBypassPerfectMatch(const LogicalTensorPtr& inputTensor, const LogicalTensorPtr& outputTensor)
{
    return inputTensor != nullptr && outputTensor != nullptr &&
           inputTensor->GetMemoryTypeOriginal() == outputTensor->GetMemoryTypeOriginal();
}

bool IsSliceEqualSizeMove(const Operation& op)
{
    if (op.GetOpcode() != Opcode::OP_SLICE || op.GetIOperands().empty() || op.GetOOperands().empty()) {
        return false;
    }
    if (!IsEqualShapeWithDynShape(op.GetIOperands().front(), op.GetOOperands().front())) {
        return false;
    }
    auto sliceAttr = GetViewAttr(op);
    return sliceAttr != nullptr && IsZeroTensorOffset(sliceAttr->GetFromOffset(), sliceAttr->GetFromDynOffset());
}

bool SliceRequiresL1(const Operation& sliceOp)
{
    auto sliceAttr = GetViewAttr(sliceOp);
    return sliceAttr != nullptr && sliceAttr->GetTo() == MemoryType::MEM_L1;
}

bool IsSingleContractWithMultipleL1SliceConsumers(const Operation& contractOp)
{
    if (contractOp.GetOpcode() != Opcode::OP_CONTRACT || contractOp.GetOOperands().empty()) {
        return false;
    }
    auto contractOutput = contractOp.GetOOperands().front();
    if (contractOutput == nullptr) {
        return false;
    }
    const auto& producers = contractOutput->GetProducers();
    if (producers.size() != 1 || *producers.begin() != &contractOp) {
        return false;
    }
    const auto& consumers = contractOutput->GetConsumers();
    return consumers.size() > 1 && std::all_of(consumers.begin(), consumers.end(), [](const Operation* consumer) {
               return consumer != nullptr && consumer->GetOpcode() == Opcode::OP_SLICE && SliceRequiresL1(*consumer);
           });
}

bool IsInputProducedBySingleSlice(const LogicalTensorPtr& tensor)
{
    if (tensor == nullptr || tensor->GetProducers().size() != 1) {
        return false;
    }
    auto* producer = *tensor->GetProducers().begin();
    return producer != nullptr && producer->GetOpcode() == Opcode::OP_SLICE;
}

template <typename ContractOps>
bool CollectPrecedingSlicesForL1Transfer(const Operation& sliceOp, const ContractOps& contractOps,
                                         std::set<Operation*>& precedingSlices)
{
    if (!SliceRequiresL1(sliceOp)) {
        return true;
    }
    for (auto* contractOp : contractOps) {
        if (contractOp == nullptr || contractOp->GetIOperands().size() != 1) {
            return false;
        }
        auto contractInput = contractOp->GetIOperands().front();
        if (contractInput == nullptr || contractInput->GetProducers().size() != 1) {
            return false;
        }
        auto* precedingSlice = *contractInput->GetProducers().begin();
        if (precedingSlice == nullptr || precedingSlice->GetOpcode() != Opcode::OP_SLICE ||
            precedingSlice->GetOOperands().empty() || precedingSlice->GetOOperands().front() != contractInput ||
            GetViewAttr(*precedingSlice) == nullptr) {
            return false;
        }
        precedingSlices.insert(precedingSlice);
    }
    return true;
}

void TransferL1Requirement(const std::set<Operation*>& precedingSlices)
{
    for (auto* precedingSlice : precedingSlices) {
        auto sliceAttr = GetViewAttr(*precedingSlice);
        auto newAttr = std::dynamic_pointer_cast<ViewOpAttribute>(sliceAttr->Clone());
        newAttr->SetToType(MemoryType::MEM_L1);
        precedingSlice->SetOpAttribute(newAttr);
        precedingSlice->GetOOperands().front()->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    }
}

void TransferL1CopyAttrs(const Operation& l1Slice, Operation& precedingSlice)
{
    precedingSlice.CopyAttrFrom(l1Slice, OP_ATTR_PREFIX + "copy_in_l1_");
    int64_t copyInMode = 0;
    if (l1Slice.GetAttr<int64_t>(OpAttributeKey::copyInMode, copyInMode)) {
        precedingSlice.SetAttr(OpAttributeKey::copyInMode, copyInMode);
    }
    int64_t isGemv = 0;
    if (l1Slice.GetAttr<int64_t>(OpAttributeKey::isGemv, isGemv)) {
        precedingSlice.SetAttr(OpAttributeKey::isGemv, isGemv);
    }
}

bool IsContractInputEqualSliceOutput(const Operation& contractOp, const Operation& sliceOp)
{
    if (contractOp.GetOpcode() != Opcode::OP_CONTRACT || contractOp.GetIOperands().empty() ||
        sliceOp.GetOpcode() != Opcode::OP_SLICE || sliceOp.GetOOperands().empty()) {
        return false;
    }
    auto contractInput = contractOp.GetIOperands().front();
    auto sliceOutput = sliceOp.GetOOperands().front();
    return IsEqualShapeWithDynShape(contractInput, sliceOutput);
}

bool CalculateContractLocalSliceOffset(const LogicalTensorPtr& inputTensor, const LogicalTensorPtr& outputTensor,
                                       const ViewOpAttribute& viewAttr, const AssembleOpAttribute& assembleAttr,
                                       std::vector<int64_t>& localOffset, std::vector<SymbolicScalar>& localDynOffset)
{
    if (inputTensor == nullptr || outputTensor == nullptr) {
        return false;
    }
    const auto& inputShape = inputTensor->GetShape();
    const auto& outputShape = outputTensor->GetShape();
    const auto& sliceOffset = viewAttr.GetFromOffset();
    const auto& contractOffset = assembleAttr.GetToOffset();
    if (inputShape.size() != outputShape.size() || inputShape.size() != sliceOffset.size() ||
        inputShape.size() != contractOffset.size() ||
        !IsConcreteDynOffsetConsistent(sliceOffset, viewAttr.GetFromDynOffset()) ||
        !IsConcreteDynOffsetConsistent(contractOffset, assembleAttr.GetToDynOffset())) {
        return false;
    }
    localOffset = TensorOffset::Sub(sliceOffset, contractOffset);
    for (size_t idx = 0; idx < inputShape.size(); ++idx) {
        if (inputShape[idx] < 0 || outputShape[idx] < 0) {
            return false;
        }
        if (localOffset[idx] < 0 || localOffset[idx] + outputShape[idx] > inputShape[idx]) {
            return false;
        }
    }
    localDynOffset = viewAttr.GetFromDynOffset().empty() && assembleAttr.GetToDynOffset().empty() ?
                         std::vector<SymbolicScalar>{} :
                         SymbolicScalar::FromConcrete(localOffset);
    return true;
}

bool IsFullSliceOfContractInput(const LogicalTensorPtr& inputTensor, const LogicalTensorPtr& outputTensor,
                                const std::vector<int64_t>& localOffset,
                                const std::vector<SymbolicScalar>& localDynOffset)
{
    return inputTensor != nullptr && outputTensor != nullptr && inputTensor->GetShape() == outputTensor->GetShape() &&
           IsZeroTensorOffset(localOffset, localDynOffset);
}

Operation* GetReplaceablePrecedingSlice(Operation& contractOp, const Operation& sliceOp,
                                        const LogicalTensorPtr& contractInput, const LogicalTensorPtr& sliceOutput)
{
    if (!SliceRequiresL1(sliceOp) || contractInput == nullptr || sliceOutput == nullptr ||
        contractInput->GetShape() != sliceOutput->GetShape()) {
        return nullptr;
    }
    const auto& contractInputConsumers = contractInput->GetConsumers();
    if (contractInputConsumers.size() != 1 || *contractInputConsumers.begin() != &contractOp) {
        return nullptr;
    }
    const auto& producers = contractInput->GetProducers();
    if (producers.size() != 1) {
        return nullptr;
    }
    auto* precedingSlice = *producers.begin();
    if (precedingSlice == nullptr || precedingSlice->GetOpcode() != Opcode::OP_SLICE ||
        precedingSlice->GetOOperands().size() != 1 || precedingSlice->GetOOperands().front() != contractInput) {
        return nullptr;
    }
    return precedingSlice;
}

void ReplaceConsumers(const LogicalTensorPtr& oldTensor, const LogicalTensorPtr& newTensor)
{
    if (oldTensor == nullptr || newTensor == nullptr) {
        return;
    }
    auto consumers = oldTensor->GetConsumers();
    for (auto* consumer : consumers) {
        if (consumer == nullptr) {
            continue;
        }
        consumer->ReplaceInput(newTensor, oldTensor);
    }
}

} // namespace

bool RemoveRedundantOpUtils::CollectSliceContractSliceChain(const Operation& contractOp, SliceContractSliceChain& chain)
{
    if (contractOp.IsDeleted() || contractOp.GetOpcode() != Opcode::OP_CONTRACT ||
        contractOp.GetIOperands().size() != 1 || contractOp.GetOOperands().size() != 1 ||
        GetAssembleAttr(contractOp) == nullptr) {
        return false;
    }

    chain.contract = const_cast<Operation*>(&contractOp);
    chain.contractInput = contractOp.GetIOperands().front();
    chain.contractOutput = contractOp.GetOOperands().front();
    if (chain.contractInput == nullptr || chain.contractOutput == nullptr ||
        chain.contractInput->GetProducers().size() != 1 || chain.contractOutput->GetProducers().size() != 1 ||
        *chain.contractOutput->GetProducers().begin() != &contractOp) {
        return false;
    }

    chain.precedingSlice = *chain.contractInput->GetProducers().begin();
    if (chain.precedingSlice == nullptr || chain.precedingSlice->IsDeleted() ||
        chain.precedingSlice->GetOpcode() != Opcode::OP_SLICE || chain.precedingSlice->GetIOperands().size() != 1 ||
        chain.precedingSlice->GetOOperands().size() != 1 ||
        chain.precedingSlice->GetOOperands().front() != chain.contractInput ||
        GetViewAttr(*chain.precedingSlice) == nullptr || chain.contractInput->GetConsumers().size() != 1 ||
        *chain.contractInput->GetConsumers().begin() != &contractOp) {
        return false;
    }
    chain.sourceTensor = chain.precedingSlice->GetIOperands().front();
    if (chain.sourceTensor == nullptr || chain.contractOutput->GetConsumers().empty()) {
        return false;
    }

    for (auto* consumer : chain.contractOutput->GetConsumers()) {
        if (consumer == nullptr || consumer->IsDeleted() || consumer->GetOpcode() != Opcode::OP_SLICE ||
            consumer->GetIOperands().size() != 1 || consumer->GetOOperands().size() != 1 ||
            consumer->GetIOperands().front() != chain.contractOutput || GetViewAttr(*consumer) == nullptr) {
            return false;
        }
        chain.consumers.push_back(consumer);
    }
    // Keep this matcher deliberately strict: a contract output may only be consumed by slices.  In
    // particular, if several contract operations produce the same tensor, the producer-count check
    // above rejects the graph because there is no single slice-contract chain whose offsets can be
    // composed safely.
    return !chain.consumers.empty();
}

bool RemoveRedundantOpUtils::ComposeSliceContractOffset(const SliceContractSliceChain& chain, const Operation& consumer,
                                                        std::vector<int64_t>& composedOffset)
{
    auto precedingAttr = GetViewAttr(*chain.precedingSlice);
    auto contractAttr = GetAssembleAttr(*chain.contract);
    auto consumerAttr = GetViewAttr(consumer);
    if (precedingAttr == nullptr || contractAttr == nullptr || consumerAttr == nullptr ||
        !precedingAttr->GetFromDynOffset().empty() || !contractAttr->GetToDynOffset().empty() ||
        !consumerAttr->GetFromDynOffset().empty()) {
        return false;
    }

    std::vector<int64_t> localOffset;
    std::vector<SymbolicScalar> localDynOffset;
    auto consumerOutput = consumer.GetOOperands().front();
    if (!CalculateContractLocalSliceOffset(chain.contractInput, consumerOutput, *consumerAttr, *contractAttr,
                                           localOffset, localDynOffset) ||
        !localDynOffset.empty()) {
        return false;
    }

    const auto& sourceOffset = precedingAttr->GetFromOffset();
    if (sourceOffset.size() != localOffset.size() || sourceOffset.size() != chain.sourceTensor->GetShape().size()) {
        return false;
    }
    composedOffset.resize(sourceOffset.size());
    for (size_t idx = 0; idx < sourceOffset.size(); ++idx) {
        if ((localOffset[idx] > 0 && sourceOffset[idx] > std::numeric_limits<int64_t>::max() - localOffset[idx]) ||
            (localOffset[idx] < 0 && sourceOffset[idx] < std::numeric_limits<int64_t>::min() - localOffset[idx])) {
            return false;
        }
        composedOffset[idx] = sourceOffset[idx] + localOffset[idx];
    }
    return IsRegionWithinShape(composedOffset, consumerOutput->GetShape(), chain.sourceTensor->GetShape());
}

Status RemoveRedundantOpUtils::Process(Function& function, std::vector<Operation*>& newOps, bool& operationUpdated)
{
    RemoveRedundantOpUtils utils;
    return utils.ProcessImpl(function, newOps, operationUpdated);
}

Status RemoveRedundantOpUtils::ProcessViewAssembleLike(Function& function, std::vector<Operation*>& newOps,
                                                       bool& operationUpdated)
{
    RemoveRedundantOpUtils utils;
    return utils.ProcessViewAssembleLikeImpl(function, newOps, operationUpdated);
}

Status RemoveRedundantOpUtils::ProcessContractSlice(Function& function, std::vector<Operation*>& newOps,
                                                    bool& operationUpdated)
{
    RemoveRedundantOpUtils utils;
    return utils.ProcessContractSliceImpl(function, newOps, operationUpdated);
}

Status RemoveRedundantOpUtils::ProcessViewCopyout(Function& function, bool& operationUpdated)
{
    std::unordered_set<int> visitedViewMagics;
    auto opList = function.Operations().DuplicatedOpList();
    for (auto& op : opList) {
        if (op == nullptr || op->IsDeleted() || op->GetOpcode() != Opcode::OP_VIEW ||
            visitedViewMagics.count(op->GetOpMagic()) != 0 || op->GetIOperands().size() != 1) {
            continue;
        }
        const auto& viewInput = op->GetIOperands().front();
        if (viewInput == nullptr) {
            continue;
        }
        std::unordered_map<LogicalTensorPtr, std::vector<std::pair<Operation*, Operation*>>> outputGroups;
        bool familyValid = true;
        for (auto* siblingView : viewInput->GetConsumers()) {
            if (siblingView == nullptr || siblingView->IsDeleted() || siblingView->GetOpcode() != Opcode::OP_VIEW) {
                continue;
            }
            visitedViewMagics.insert(siblingView->GetOpMagic());
            if (siblingView->GetOOperands().size() != 1 || siblingView->GetOOperands().front() == nullptr) {
                continue;
            }
            const auto& viewOutput = siblingView->GetOOperands().front();
            std::vector<Operation*> copyOutConsumers;
            for (auto* consumer : viewOutput->GetConsumers()) {
                if (consumer != nullptr && !consumer->IsDeleted() &&
                    OpcodeManager::Inst().IsCopyOut(consumer->GetOpcode())) {
                    copyOutConsumers.push_back(consumer);
                }
            }
            if (copyOutConsumers.empty()) {
                continue;
            }
            if (viewOutput->GetConsumers().size() != 1 || copyOutConsumers.size() != 1 ||
                copyOutConsumers.front()->GetOOperands().size() != 1 ||
                copyOutConsumers.front()->GetOOperands().front() == nullptr) {
                familyValid = false;
                break;
            }
            outputGroups[copyOutConsumers.front()->GetOOperands().front()].push_back(
                {siblingView, copyOutConsumers.front()});
        }
        if (!familyValid) {
            continue;
        }
        for (auto& [copyOutOutput, candidates] : outputGroups) {
            std::vector<ViewCopyoutBranch> branches;
            size_t retainedIndex = 0;
            if (!CanMergeViewCopyoutGroup(viewInput, copyOutOutput, candidates, branches, retainedIndex)) {
                continue;
            }
            MergeViewCopyoutGroup(viewInput, branches, retainedIndex, operationUpdated);
        }
    }
    WarnUnresolvedCopyoutOffset(function);
    return SUCCESS;
}

Status RemoveRedundantOpUtils::ProcessCopyinAssemble(Function& function, bool& operationUpdated)
{
    std::unordered_set<int> visitedAssembleMagics;
    auto opList = function.Operations().DuplicatedOpList();
    for (auto& op : opList) {
        if (op == nullptr || op->IsDeleted() || op->GetOpcode() != Opcode::OP_ASSEMBLE ||
            visitedAssembleMagics.count(op->GetOpMagic()) != 0 || op->GetOOperands().size() != 1) {
            continue;
        }
        const auto& assembleOutput = op->GetOOperands().front();
        if (assembleOutput == nullptr) {
            continue;
        }
        std::unordered_map<LogicalTensorPtr, std::vector<std::pair<Operation*, Operation*>>> inputGroups;
        std::unordered_set<LogicalTensorPtr> invalidInputs;
        for (auto* siblingAssemble : assembleOutput->GetProducers()) {
            if (siblingAssemble == nullptr || siblingAssemble->IsDeleted() ||
                siblingAssemble->GetOpcode() != Opcode::OP_ASSEMBLE) {
                continue;
            }
            visitedAssembleMagics.insert(siblingAssemble->GetOpMagic());
            if (siblingAssemble->GetIOperands().size() != 1 || siblingAssemble->GetIOperands().front() == nullptr) {
                continue;
            }
            const auto& copyInOutput = siblingAssemble->GetIOperands().front();
            if (copyInOutput->GetProducers().size() != 1) {
                continue;
            }
            auto* copyInOp = *copyInOutput->GetProducers().begin();
            if (copyInOp == nullptr || copyInOp->IsDeleted() ||
                !OpcodeManager::Inst().IsCopyIn(copyInOp->GetOpcode()) || copyInOp->GetIOperands().size() != 1 ||
                copyInOp->GetIOperands().front() == nullptr) {
                continue;
            }
            const auto& copyInInput = copyInOp->GetIOperands().front();
            if (copyInOutput->GetConsumers().size() != 1 || *copyInOutput->GetConsumers().begin() != siblingAssemble) {
                invalidInputs.insert(copyInInput);
                continue;
            }
            inputGroups[copyInInput].push_back({copyInOp, siblingAssemble});
        }
        for (auto& [copyInInput, candidates] : inputGroups) {
            if (invalidInputs.count(copyInInput) != 0) {
                continue;
            }
            std::vector<CopyinAssembleBranch> branches;
            size_t retainedIndex = 0;
            if (!CanMergeCopyinAssembleGroup(copyInInput, assembleOutput, candidates, branches, retainedIndex)) {
                continue;
            }
            MergeCopyinAssembleGroup(copyInInput, assembleOutput, branches, retainedIndex, operationUpdated);
        }
    }
    return SUCCESS;
}

Status RemoveRedundantOpUtils::ProcessImpl(Function& function, std::vector<Operation*>& newOps, bool& operationUpdated)
{
    if (ProcessViewAssembleLikeImpl(function, newOps, operationUpdated) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "ProcessViewAssembleLike failed.");
        return FAILED;
    }
    if (ProcessContractSliceImpl(function, newOps, operationUpdated) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "ProcessContractSlice failed.");
        return FAILED;
    }
    DeadOperationEliminator::EliminateDeadOperation(function);
    return SUCCESS;
}

Status RemoveRedundantOpUtils::ProcessViewAssembleLikeImpl(Function& function, std::vector<Operation*>& newOps,
                                                           bool& operationUpdated)
{
    auto opList = function.Operations().DuplicatedOpList();
    bool hasDeletedOperations = false;
    for (auto& op : opList) {
        if (op == nullptr || !IsViewLikeOpcode(op->GetOpcode()) || op->iOperand.empty() || op->oOperand.empty()) {
            continue;
        }
        auto& startTensor = op->iOperand.front();
        auto consumers = op->oOperand.front()->GetConsumers();
        for (const auto& consumer : consumers) {
            if (consumer == nullptr || !IsSupportedViewAssembleCascade(op->GetOpcode(), consumer->GetOpcode()) ||
                consumer->oOperand.empty()) {
                continue;
            }
            auto& endTensor = consumer->oOperand.front();
            if (function.IsFromInCast(startTensor) && function.IsFromOutCast(endTensor)) {
                continue;
            }
            if (function.IsFromOutCast(startTensor) && function.IsFromOutCast(endTensor)) {
                continue;
            }
            if (IsLegacyViewAssembleCascade(op->GetOpcode(), consumer->GetOpcode()) &&
                startTensor->GetMemoryTypeOriginal() != endTensor->GetMemoryTypeOriginal()) {
                continue;
            }
            if (remove_redundant_op_internal::HasOtherAssembleOutputOnSameRaw(function, endTensor)) {
                APASS_LOG_DEBUG_F(Elements::Tensor,
                                  "endTensor[%d] has another assemble output on the same raw tensor, skip.",
                                  endTensor->GetMagic());
                continue;
            }
            if (op->GetOpcode() == Opcode::OP_SLICE && consumer->GetOpcode() == Opcode::OP_CONTRACT &&
                consumer->GetIOperands().size() == 1 &&
                IsInputProducedBySingleSlice(consumer->GetIOperands().front())) {
                // Leave a strict slice-contract-slice chain to ProcessSliceContractSlice.  The
                // dedicated pass must see all L1/UNKNOWN fan-out variants before this generic
                // view/assemble cascade pass can rewrite the prefix.
                SliceContractSliceChain chain;
                const auto& contractOutput = consumer->GetOOperands().front();
                const bool hasOnlySliceConsumers = contractOutput != nullptr &&
                                                   !contractOutput->GetConsumers().empty() &&
                                                   std::all_of(contractOutput->GetConsumers().begin(),
                                                               contractOutput->GetConsumers().end(),
                                                               [](const Operation* outputConsumer) {
                                                                   return outputConsumer != nullptr &&
                                                                          !outputConsumer->IsDeleted() &&
                                                                          outputConsumer->GetOpcode() ==
                                                                              Opcode::OP_SLICE;
                                                               });
                if (CollectSliceContractSliceChain(*consumer, chain) ||
                    IsSingleContractWithMultipleL1SliceConsumers(*consumer) || hasOnlySliceConsumers) {
                    continue;
                }
            }
            if (startTensor->shape == endTensor->shape && startTensor->offset == endTensor->offset) {
                if (!CanBypassPerfectMatch(startTensor, endTensor)) {
                    continue;
                }
                if (function.IsFromOutCast(endTensor) &&
                    HasConsumerOutsideViewAssembleCascade(startTensor, endTensor)) {
                    continue;
                }
                APASS_LOG_DEBUG_F(
                    Elements::Operation, "CASE1: Process %s[%d]'s input and %s[%d]'s output perfectMatch.",
                    OpcodeManager::Inst().GetOpcodeStr(op->GetOpcode()).c_str(), op->opmagic,
                    OpcodeManager::Inst().GetOpcodeStr(consumer->GetOpcode()).c_str(), consumer->GetOpMagic());
                ProcessPerfectMatch(function, startTensor, endTensor, function.IsFromOutCast(endTensor),
                                    operationUpdated);
            } else {
                if ((function.IsFromInCast(startTensor) && CommonUtils::ContainsNegativeOne(startTensor->GetShape())) ||
                    (function.IsFromOutCast(endTensor) && CommonUtils::ContainsNegativeOne(endTensor->GetShape()))) {
                    continue;
                }
                APASS_LOG_DEBUG_F(Elements::Operation, "CASE2: Process %s[%d]'s input is a part of %s[%d]'s output.",
                                  OpcodeManager::Inst().GetOpcodeStr(op->GetOpcode()).c_str(), op->opmagic,
                                  OpcodeManager::Inst().GetOpcodeStr(consumer->GetOpcode()).c_str(),
                                  consumer->GetOpMagic());
                if (IsSingleZeroOffsetSliceContract(*op, *consumer) &&
                    !HasMemoryTypeTransform(startTensor, endTensor)) {
                    GenerateSingleSliceContractView(function, *op, *consumer, startTensor, endTensor, newOps,
                                                    operationUpdated);
                    hasDeletedOperations = true;
                    continue;
                }
                GenerateNewViewLike(function, *op, startTensor, endTensor,
                                    GetViewLikeReplacementOpcode(startTensor, endTensor), newOps, operationUpdated);
            }
        }
    }
    if (hasDeletedOperations) {
        function.EraseOperations(true, false);
    }
    return SUCCESS;
}

void RemoveRedundantOpUtils::RemoveViewAssembleForOutcast(LogicalTensorPtr& startTensor, LogicalTensorPtr& endTensor,
                                                          bool& operationUpdated)
{
    bool canRemove = false;
    std::set<Operation*, LogicalTensor::CompareOp> removeOps;
    for (auto& startConsumer : startTensor->GetConsumers()) {
        if (startConsumer == nullptr || !IsViewLikeOpcode(startConsumer->GetOpcode())) {
            continue;
        }
        canRemove = true;
        for (auto& endProducer : startConsumer->ConsumerOps()) {
            if (endProducer->GetOOperands().empty() || endProducer->GetOOperands().front() != endTensor ||
                !IsSupportedViewAssembleCascade(startConsumer->GetOpcode(), endProducer->GetOpcode())) {
                canRemove = false;
            } else {
                removeOps.insert(endProducer);
            }
        }
        if (canRemove) {
            removeOps.insert(startConsumer);
        }
    }

    if (removeOps.empty()) {
        return;
    }

    auto startProducers = startTensor->GetProducers();
    for (auto* op : removeOps) {
        for (const auto& input : op->GetIOperands()) {
            input->RemoveConsumer(*op);
        }
        for (const auto& output : op->GetOOperands()) {
            output->RemoveProducer(*op);
        }
    }

    if (startTensor != endTensor) {
        for (auto* producer : startProducers) {
            producer->ReplaceOutputOperand(startTensor, endTensor);
            endTensor->AddProducer(*producer);
        }
        startTensor->GetProducers().clear();
        if (!startTensor->GetDynValidShape().empty()) {
            endTensor->UpdateDynValidShape(startTensor->GetDynValidShape());
        }
    }
    operationUpdated = true;
}

void RemoveRedundantOpUtils::ProcessPerfectMatch(Function& function, LogicalTensorPtr& startTensor,
                                                 LogicalTensorPtr& endTensor, bool endTensorIsOutcast,
                                                 bool& operationUpdated)
{
    if (!IsValidViewAssemble(startTensor, endTensor)) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Not valid view-assemble-like case.");
        return;
    }
    auto removedOps = remove_redundant_op_internal::CollectViewAssembleCascadeOps(startTensor, endTensor);
    auto targetConsumers = remove_redundant_op_internal::GetTensorConsumers(endTensor);
    auto targetProducers = remove_redundant_op_internal::GetTensorProducers(startTensor);
    if (!remove_redundant_op_internal::CanMigrateRemovedOpsTokenDependency(function, removedOps, targetConsumers,
                                                                           targetProducers)) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Cannot migrate token dependency for view-assemble perfect match.");
        return;
    }
    remove_redundant_op_internal::MigrateRemovedOpsTokenDependency(function, removedOps, targetConsumers,
                                                                   targetProducers);
    if (endTensor->GetConsumers().size() == 0) {
        if (endTensorIsOutcast) {
            RemoveViewAssembleForOutcast(startTensor, endTensor, operationUpdated);
        }
    } else {
        auto consumers = endTensor->GetConsumers();
        for (auto& assembleConsumer : consumers) {
            assembleConsumer->ReplaceInput(startTensor, endTensor);
        }
        operationUpdated = true;
    }
}

bool RemoveRedundantOpUtils::IsNotSameViewInput(LogicalTensorPtr& startTensor, LogicalTensorPtr& endTensor) const
{
    for (auto& assembleOp : endTensor->GetProducers()) {
        if (assembleOp == nullptr || !IsAssembleLikeOpcode(assembleOp->GetOpcode()) ||
            assembleOp->GetIOperands().empty()) {
            return true;
        }
        auto& tempTensor = assembleOp->GetIOperands().front();
        auto producers = tempTensor->GetProducers();
        if (producers.empty()) {
            return true;
        }
        auto& producerOps = tempTensor->GetProducers();
        for (auto& producerOp : producerOps) {
            if (producerOp == nullptr || producerOp->GetIOperands().empty() ||
                !IsViewLikeOpcode(producerOp->GetOpcode())) {
                return true;
            }
            auto& inTensor = producerOp->GetIOperands().front();
            if (inTensor != startTensor) {
                return true;
            }
        }
    }
    return false;
}

bool RemoveRedundantOpUtils::IsDataReplace(LogicalTensorPtr& endTensor) const
{
    for (auto& assembleOp : endTensor->GetProducers()) {
        if (assembleOp == nullptr || !IsAssembleLikeOpcode(assembleOp->GetOpcode()) ||
            assembleOp->GetIOperands().empty()) {
            return true;
        }
        auto& tempTensor = assembleOp->GetIOperands().front();
        auto producers = tempTensor->GetProducers();
        if (producers.empty()) {
            return true;
        }
        auto& viewOps = tempTensor->GetProducers();
        for (auto& viewOp : viewOps) {
            if (viewOp == nullptr || viewOp->GetIOperands().empty() || !IsViewLikeOpcode(viewOp->GetOpcode())) {
                return true;
            }
            auto viewOpAttribute = GetViewAttr(*viewOp);
            auto assembleOpAttribute = GetAssembleAttr(*assembleOp);
            if (viewOpAttribute == nullptr || assembleOpAttribute == nullptr) {
                return true;
            }
            auto viewOffset = viewOpAttribute->GetFrom();
            auto assembleOffset = assembleOpAttribute->GetToOffset();
            if (viewOffset != assembleOffset) {
                return true;
            }
        }
    }
    return false;
}

bool RemoveRedundantOpUtils::IsValidViewAssemble(LogicalTensorPtr& startTensor, LogicalTensorPtr& endTensor) const
{
    bool isNotSameViewInput = IsNotSameViewInput(startTensor, endTensor);
    if (isNotSameViewInput) {
        APASS_LOG_DEBUG_F(Elements::Tensor,
                          "Assemble-like output endTensor[%d] has different input except startTensor[%d].",
                          endTensor->magic, startTensor->magic);
        return false;
    }
    if (endTensor->shape.size() > startTensor->shape.size()) {
        return false;
    }
    for (size_t i = 0; i < endTensor->shape.size(); ++i) {
        if (endTensor->shape[i] > startTensor->shape[i]) {
            return false;
        }
    }
    bool isDataReplace = IsDataReplace(endTensor);
    if (isDataReplace) {
        APASS_LOG_DEBUG_F(Elements::Tensor,
                          "Assemble-like output endTensor[%d] is replaced comparing with startTensor[%d].",
                          endTensor->magic, startTensor->magic);
        return false;
    }
    return true;
}

void RemoveRedundantOpUtils::CalculateViewOffset(Operation& op, LogicalTensorPtr& startTensor,
                                                 LogicalTensorPtr& endTensor, std::vector<int64_t>& newOffset,
                                                 std::vector<SymbolicScalar>& newDynOffset)
{
    for (size_t idx = 0; idx < op.iOperand[0]->offset.size(); idx++) {
        for (auto& consumerView : startTensor->GetConsumers()) {
            if (consumerView == nullptr || !IsViewLikeOpcode(consumerView->GetOpcode()) ||
                consumerView->GetOOperands().empty()) {
                continue;
            }
            auto& tempTensor = consumerView->GetOOperands().front();
            bool leadsToCurrentEndTensor = false;
            for (auto& consumerAssemble : tempTensor->GetConsumers()) {
                if (consumerAssemble == nullptr || !IsAssembleLikeOpcode(consumerAssemble->GetOpcode())) {
                    continue;
                }
                if (!consumerAssemble->GetOOperands().empty() &&
                    consumerAssemble->GetOOperands().front() == endTensor) {
                    leadsToCurrentEndTensor = true;
                    break;
                }
            }
            if (!leadsToCurrentEndTensor) {
                continue;
            }
            auto viewOpAttribute = GetViewAttr(*consumerView);
            if (viewOpAttribute != nullptr) {
                auto viewOffset = viewOpAttribute->GetFromOffset();
                auto viewDynOffset = viewOpAttribute->GetFromDynOffset();
                newOffset[idx] = std::min(newOffset[idx], viewOffset[idx]);
                if (idx < viewDynOffset.size()) {
                    newDynOffset[idx] = std::min(newDynOffset[idx], viewDynOffset[idx]);
                }
            }
        }
    }
}

void RemoveRedundantOpUtils::GenerateNewViewLike(Function& function, Operation& op, LogicalTensorPtr& startTensor,
                                                 LogicalTensorPtr& endTensor, Opcode newOpcode,
                                                 std::vector<Operation*>& newOps, bool& operationUpdated)
{
    std::vector<int64_t> newOffset;
    std::vector<SymbolicScalar> newDynOffset;
    bool isExistingValidCase = IsValidViewAssemble(startTensor, endTensor);
    if (!isExistingValidCase && !TryCalculateTranslatedViewOffset(startTensor, endTensor, newOffset, newDynOffset)) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Not valid view-assemble-like case.");
        return;
    }
    // endTensor 是 outcast（无消费者）时不处理，避免图断裂
    if (endTensor->GetConsumers().empty()) {
        return;
    }
    if (isExistingValidCase) {
        newOffset.assign(op.iOperand[0]->offset.size(), INT_MAX);
        newDynOffset.assign(op.iOperand[0]->offset.size(), INT_MAX);
        CalculateViewOffset(op, startTensor, endTensor, newOffset, newDynOffset);
        if (IsInitialDynOffset(newDynOffset)) {
            newDynOffset.clear();
        }
    }
    LogicalTensorPtr newViewTensor;
    std::vector<int64_t> curOffset(endTensor->shape.size(), 0);
    newViewTensor = irBuilder_.CreateTensorVar(endTensor->GetRawTensor(), curOffset, endTensor->shape,
                                               std::vector<SymbolicScalar>{});
    newViewTensor->SetMemoryTypeBoth(endTensor->GetMemoryTypeOriginal());
    auto viewAttribute = std::make_shared<ViewOpAttribute>(newOffset, newDynOffset, newViewTensor->GetDynValidShape());
    viewAttribute->SetToType(endTensor->GetMemoryTypeToBe());
    auto& newViewOp = PassOperationUtils::AddOperation(
        function, newOpcode, {startTensor}, {newViewTensor},
        [&viewAttribute](Operation& newOp) { newOp.SetOpAttribute(viewAttribute); });
    auto consumers = endTensor->GetConsumers();
    for (auto* consumer : consumers) {
        consumer->ReplaceInput(newViewTensor, endTensor);
    }
    auto removedOps = remove_redundant_op_internal::CollectViewAssembleCascadeOps(startTensor, endTensor);
    remove_redundant_op_internal::MigrateRemovedOpsTokenDependency(function, removedOps, {&newViewOp}, {&newViewOp});
    newOps.push_back(&newViewOp);
    operationUpdated = true;
}

Status RemoveRedundantOpUtils::ProcessContractSliceImpl(Function& function, std::vector<Operation*>& newOps,
                                                        bool& operationUpdated)
{
    bool sliceContractSliceUpdated = false;
    if (ProcessSliceContractSlice(function, newOps, sliceContractSliceUpdated) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "ProcessSliceContractSlice failed.");
        return FAILED;
    }
    if (sliceContractSliceUpdated) {
        function.EraseOperations(true, false);
        operationUpdated = true;
    }
    bool contractSliceContractUpdated = false;
    // 顺序约束：该消除必须早于 AssignMemoryType 执行（本函数经 SplitLargeFanoutTensor 挂接于
    // PVC2_OOO 流水线，先于 ASSIGN_MEMORY_TYPE），否则 L0C2UB pattern 的固定深度生产者窗口
    // （cube op -> contract -> reshape）仍会因三明治失配，matmul 结果静默退回 DDR 中转。
    if (ProcessContractSliceContract(function, contractSliceContractUpdated) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "ProcessContractSliceContract failed.");
        return FAILED;
    }
    if (contractSliceContractUpdated) {
        function.EraseOperations(true, false);
        operationUpdated = true;
    }
    if (ProcessMultiContractSingleSlice(function, operationUpdated) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "ProcessMultiContractSingleSlice failed.");
        return FAILED;
    }
    if (operationUpdated) {
        function.EraseOperations(true, false);
    }
    if (ProcessSingleContractMultiSlice(function, newOps, operationUpdated) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "ProcessSingleContractMultiSlice failed.");
        return FAILED;
    }
    function.EraseOperations(true, false);
    return SUCCESS;
}

// ===== 收窄守卫（临时）：仅处理 matmul 输出侧场景 =====
// middle 的 producers 必须全是单输入单输出 CONTRACT，且每个 CONTRACT 的输入直接产自
// matmul（A_MUL_B/A_MULACC_B）。非 matmul 来源的 CONTRACT→SLICE→CONTRACT 结构保持原状；
// 验证充分后删除本函数中的 matmul 检查即可放开到通用 CONTRACT 来源。
bool RemoveRedundantOpUtils::IsMatmulBackedContractProducers(
    const std::set<Operation*, LogicalTensor::CompareOp>& producers)
{
    if (producers.empty()) {
        return false;
    }
    for (auto* producer : producers) {
        if (producer == nullptr || producer->GetOpcode() != Opcode::OP_CONTRACT ||
            producer->GetIOperands().size() != 1 || producer->GetIOperands().front() == nullptr ||
            GetAssembleAttr(*producer) == nullptr) {
            return false;
        }
        const auto& inputProducers = producer->GetIOperands().front()->GetProducers();
        if (inputProducers.empty()) {
            return false;
        }
        for (auto* inputProducer : inputProducers) {
            if (inputProducer == nullptr || inputProducer->IsDeleted() ||
                (inputProducer->GetOpcode() != Opcode::OP_A_MUL_B &&
                 inputProducer->GetOpcode() != Opcode::OP_A_MULACC_B)) {
                return false;
            }
        }
    }
    return true;
}

// 匹配 CONTRACT₁ → SLICE → CONTRACT₂ 三明治（整块匹配）：SLICE 从 middle 切出的区域与
// 某个 CONTRACT₁ 的写入区域完全重合（shape/validshape/offset 均相同）。
// 匹配成功时输出该 CONTRACT₁，否则返回 false。
bool RemoveRedundantOpUtils::MatchContractSliceContract(Operation& sliceOp, Operation*& matchedProducer)
{
    matchedProducer = nullptr;
    auto sliceAttr = GetViewAttr(sliceOp);
    auto middle = sliceOp.GetIOperands().front();
    auto tile = sliceOp.GetOOperands().front();
    if (sliceAttr == nullptr || sliceAttr->GetFromOffset().size() != middle->GetShape().size() ||
        !IsConcreteDynOffsetConsistent(sliceAttr->GetFromOffset(), sliceAttr->GetFromDynOffset()) ||
        !IsMatmulBackedContractProducers(middle->GetProducers())) {
        return false;
    }
    const auto& tileConsumers = tile->GetConsumers();
    if (tileConsumers.empty() || std::any_of(tileConsumers.begin(), tileConsumers.end(), [](Operation* consumer) {
            return consumer == nullptr || consumer->GetOpcode() != Opcode::OP_CONTRACT ||
                   consumer->GetIOperands().size() != 1;
        })) {
        return false;
    }
    bool ambiguous = false;
    for (auto* producer : middle->GetProducers()) {
        auto contractAttr = GetAssembleAttr(*producer);
        auto producerInput = producer->GetIOperands().front();
        if (!IsConcreteDynOffsetConsistent(contractAttr->GetToOffset(), contractAttr->GetToDynOffset()) ||
            !IsEqualShapeWithDynShape(producerInput, tile) ||
            contractAttr->GetToOffset() != sliceAttr->GetFromOffset()) {
            continue;
        }
        if (matchedProducer != nullptr) {
            ambiguous = true;
            break;
        }
        matchedProducer = producer;
    }
    return matchedProducer != nullptr && !ambiguous;
}

// middle 不再有存活消费者时，其全部 producer 均为死写，一并删除
void RemoveRedundantOpUtils::EraseOrphanMiddleProducers(const LogicalTensorPtr& middle)
{
    for (auto* consumer : middle->GetConsumers()) {
        if (consumer != nullptr && !consumer->IsDeleted()) {
            return;
        }
    }
    for (auto* producer : middle->GetProducers()) {
        if (producer != nullptr && !producer->IsDeleted()) {
            producer->SetAsDeleted();
        }
    }
}

// 消除 CONTRACT₁ → SLICE → CONTRACT₂ 三明治（整块匹配）。
// 该三明治由恒等 ASSEMBLE/VIEW 的 tile 展开产生：SLICE 只是把 CONTRACT₁ 刚拼入 middle 的
// 数据原样切回。消除时让 SLICE 下游的 CONTRACT₂ 直接消费 CONTRACT₁ 的输入（搬运 shape 与
// 写偏移均不变），删除 SLICE；middle 死亡后其全部 producer 一并删除。消除后 cube 计算结果
// 与 reshape 之间不再有切分/拼装往返，data path pattern（如 L0C2UB）的固定深度窗口才能匹配。
Status RemoveRedundantOpUtils::ProcessContractSliceContract(Function& function, bool& operationUpdated)
{
    std::vector<LogicalTensorPtr> eliminatedMiddles;
    auto opList = function.Operations().DuplicatedOpList();
    for (auto* op : opList) {
        if (op == nullptr || op->GetOpcode() != Opcode::OP_SLICE || op->GetIOperands().size() != 1 ||
            op->GetOOperands().size() != 1) {
            continue;
        }
        Operation* matchedProducer = nullptr;
        if (!MatchContractSliceContract(*op, matchedProducer)) {
            continue;
        }
        auto middle = op->GetIOperands().front();
        auto matchedInput = matchedProducer->GetIOperands().front();
        auto tileConsumers = op->GetOOperands().front()->GetConsumers();
        for (auto* consumer : tileConsumers) {
            consumer->ReplaceIOperand(0, matchedInput);
        }
        op->SetAsDeleted();
        operationUpdated = true;
        eliminatedMiddles.emplace_back(middle);
    }
    std::set<LogicalTensorPtr> processedMiddles;
    for (auto& middle : eliminatedMiddles) {
        if (processedMiddles.insert(middle).second) {
            EraseOrphanMiddleProducers(middle);
        }
    }
    if (operationUpdated) {
        APASS_LOG_INFO_F(Elements::Function, "Eliminated %zu contract-slice-contract sandwich(es).",
                         processedMiddles.size());
    }
    return SUCCESS;
}

bool RemoveRedundantOpUtils::CollectSliceContractSliceRewrites(const SliceContractSliceChain& chain, bool isFanout,
                                                               bool keepL1Fanout,
                                                               std::vector<SliceContractSliceRewrite>& rewrites,
                                                               bool& canComposeOffsets)
{
    rewrites.reserve(chain.consumers.size());
    for (auto* consumer : chain.consumers) {
        auto consumerAttr = GetViewAttr(*consumer);
        auto contractAttr = GetAssembleAttr(*chain.contract);
        std::vector<int64_t> localOffset;
        std::vector<SymbolicScalar> localDynOffset;
        if (consumerAttr == nullptr || contractAttr == nullptr ||
            !CalculateContractLocalSliceOffset(chain.contractInput, consumer->GetOOperands().front(), *consumerAttr,
                                               *contractAttr, localOffset, localDynOffset)) {
            return false;
        }
        // 对消产生的 view 以 localOffset 直接读取 contractInput 共享 buffer；偏移非 32B 对齐时
        // VEC consumer 会产生非对齐 UB 访问，此时保留 CONTRACT+SLICE 拷贝路径不做对消。
        if (!IsViewFromOffset32BAligned(chain.contractInput, localOffset, localDynOffset)) {
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "Skip slice-contract-slice rewrite: localOffset %s of CONTRACT[%d] consumer SLICE[%d] "
                              "is not 32B aligned.",
                              IntVectorToString(localOffset).c_str(), chain.contract->GetOpMagic(),
                              consumer->GetOpMagic());
            return false;
        }

        SliceContractSliceRewrite rewrite;
        rewrite.consumer = consumer;
        rewrite.localOffset = std::move(localOffset);
        rewrite.localDynOffset = std::move(localDynOffset);
        rewrite.isFullSlice = IsFullSliceOfContractInput(chain.contractInput, consumer->GetOOperands().front(),
                                                         rewrite.localOffset, rewrite.localDynOffset);
        if (isFanout && !keepL1Fanout) {
            if (rewrite.isFullSlice && !IsContractInputEqualSliceOutput(*chain.contract, *consumer)) {
                // Mirror the single-consumer rejection: bypassing a full slice whose dynamic valid shape
                // differs from the contract input would drop that shape from the surviving graph.
                return false;
            }
            if (!ComposeSliceContractOffset(chain, *consumer, rewrite.composedOffset)) {
                // The direct source rewrite requires concrete offsets.  Keep the legacy surviving-input
                // path available for dynamic/otherwise non-composable chains.
                canComposeOffsets = false;
            } else if (!IsViewFromOffset32BAligned(chain.sourceTensor, rewrite.composedOffset)) {
                // Direct source views with misaligned composed offsets fall back to the legacy
                // surviving-input path, which reads the contract input at the (already checked)
                // local offset instead.
                APASS_LOG_DEBUG_F(Elements::Operation,
                                  "Composed offset %s of CONTRACT[%d] consumer SLICE[%d] direct-source view is not "
                                  "32B aligned, fall back to the legacy surviving-input path.",
                                  IntVectorToString(rewrite.composedOffset).c_str(), chain.contract->GetOpMagic(),
                                  consumer->GetOpMagic());
                canComposeOffsets = false;
            }
        }
        rewrites.push_back(std::move(rewrite));
    }
    return true;
}

// 计算 L1 fanout 折叠所需的 mergedOffset（前序 slice 偏移 + localOffset）。precedingSlice 非同
// 内存普通视图、合并偏移越界或非 32B 对齐时返回 false，调用方应保持 precedingSlice 物化不折叠。
bool RemoveRedundantOpUtils::CollectL1FoldMergedOffsets(const SliceContractSliceChain& chain,
                                                        const std::vector<SliceContractSliceRewrite>& rewrites,
                                                        std::vector<MergedOffset>& mergedOffsets)
{
    auto precedingAttr = GetViewAttr(*chain.precedingSlice);
    bool canFoldPrecedingSlice = precedingAttr->GetFromOffset().size() == chain.sourceTensor->GetShape().size() &&
                                 chain.sourceTensor->GetShape().size() == chain.contractInput->GetShape().size() &&
                                 chain.sourceTensor->GetMemoryTypeOriginal() ==
                                     chain.contractInput->GetMemoryTypeOriginal() &&
                                 (precedingAttr->GetTo() == MemoryType::MEM_UNKNOWN ||
                                  precedingAttr->GetTo() == chain.sourceTensor->GetMemoryTypeOriginal()) &&
                                 !HasMemoryTypeTransform(chain.sourceTensor, chain.contractInput) &&
                                 IsConcreteDynOffsetConsistent(precedingAttr->GetFromOffset(),
                                                               precedingAttr->GetFromDynOffset());
    if (!canFoldPrecedingSlice) {
        return false;
    }
    for (const auto& rewrite : rewrites) {
        auto mergedOffset = TensorOffset::Add(precedingAttr->GetFromOffset(), precedingAttr->GetFromDynOffset(),
                                              rewrite.localOffset, rewrite.localDynOffset);
        if (!IsRegionWithinShape(mergedOffset.first, rewrite.consumer->GetOOperands().front()->GetShape(),
                                 chain.sourceTensor->GetShape())) {
            return false;
        }
        // 折叠后 consumer 以 mergedOffset 直读 sourceTensor 共享 buffer；偏移非 32B 对齐时 VEC
        // consumer 会产生非对齐 UB 访问，此时不折叠 precedingSlice，consumer 改读 contractInput
        // （localOffset 已在 collect 阶段校验对齐）。
        if (!IsViewFromOffset32BAligned(chain.sourceTensor, mergedOffset.first, mergedOffset.second)) {
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "Merged offset %s of CONTRACT[%d] consumer SLICE[%d] L1 fold is not 32B aligned, "
                              "keep the preceding slice materialization.",
                              IntVectorToString(mergedOffset.first).c_str(), chain.contract->GetOpMagic(),
                              rewrite.consumer->GetOpMagic());
            return false;
        }
        mergedOffsets.push_back(std::move(mergedOffset));
    }
    return true;
}

void RemoveRedundantOpUtils::FoldL1FanoutChain(const SliceContractSliceChain& chain,
                                               const std::vector<SliceContractSliceRewrite>& rewrites)
{
    // Multiple L1 consumers share the preceding L1/UB materialization.  Fold the preceding slice into
    // every consumer when it is a plain same-memory view, so the slice and the contract both disappear;
    // otherwise keep the slice and remove only the redundant contract.
    std::vector<MergedOffset> mergedOffsets;
    const bool canFoldPrecedingSlice = CollectL1FoldMergedOffsets(chain, rewrites, mergedOffsets);
    for (size_t idx = 0; idx < rewrites.size(); ++idx) {
        const auto& rewrite = rewrites[idx];
        auto* consumer = rewrite.consumer;
        auto consumerAttr = GetViewAttr(*consumer);
        auto newAttr = std::dynamic_pointer_cast<ViewOpAttribute>(consumerAttr->Clone());
        if (canFoldPrecedingSlice) {
            newAttr->SetFromOffset(mergedOffsets[idx].first, mergedOffsets[idx].second);
        } else {
            newAttr->SetFromOffset(rewrite.localOffset, rewrite.localDynOffset);
        }
        newAttr->SetToType(MemoryType::MEM_L1);
        consumer->SetOpAttribute(newAttr);
        consumer->GetOOperands().front()->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
        consumer->ReplaceInput(canFoldPrecedingSlice ? chain.sourceTensor : chain.contractInput, chain.contractOutput);
    }
    if (canFoldPrecedingSlice) {
        chain.precedingSlice->SetAsDeleted();
    }
    chain.contract->SetAsDeleted();
}

void RemoveRedundantOpUtils::RewriteComposableFanoutChain(const SliceContractSliceChain& chain,
                                                          const std::vector<SliceContractSliceRewrite>& rewrites)
{
    // For UNKNOWN consumers and mixed L1/UNKNOWN fan-out, compose the preceding slice and contract
    // offsets into each remaining slice.  This is the only safe way to eliminate the common
    // slice1-contract1 prefix: every consumer then reads directly from the one original source tensor.
    for (const auto& rewrite : rewrites) {
        auto* consumer = rewrite.consumer;
        auto consumerAttr = GetViewAttr(*consumer);
        auto newAttr = std::dynamic_pointer_cast<ViewOpAttribute>(consumerAttr->Clone());
        newAttr->SetFromOffset(rewrite.composedOffset);
        consumer->SetOpAttribute(newAttr);
        consumer->ReplaceInput(chain.sourceTensor, chain.contractOutput);
    }
    chain.precedingSlice->SetAsDeleted();
    chain.contract->SetAsDeleted();
}

bool RemoveRedundantOpUtils::RewriteDynamicFanoutFallback(Function& function, const SliceContractSliceChain& chain,
                                                          std::vector<SliceContractSliceRewrite>& rewrites,
                                                          bool hasL1Consumer, std::vector<Operation*>& newOps,
                                                          bool& operationUpdated)
{
    // Dynamic offsets cannot be composed into a static source slice.  Keep the original surviving-input
    // rewrite for this unsupported direct-composition case.
    std::stable_sort(rewrites.begin(), rewrites.end(),
                     [](const SliceContractSliceRewrite& lhs, const SliceContractSliceRewrite& rhs) {
                         return SliceRequiresL1(*lhs.consumer) && !SliceRequiresL1(*rhs.consumer);
                     });
    std::set<Operation*> precedingSlices;
    for (const auto& rewrite : rewrites) {
        if (!CollectPrecedingSlicesForL1Transfer(*rewrite.consumer, std::vector<Operation*>{chain.contract},
                                                 precedingSlices)) {
            return false;
        }
    }
    TransferL1Requirement(precedingSlices);
    LogicalTensorPtr survivingInput = chain.contractInput;
    for (const auto& rewrite : rewrites) {
        RewriteSliceContractSliceConsumer(function, *chain.contract, *rewrite.consumer, rewrite.isFullSlice,
                                          rewrite.localOffset, rewrite.localDynOffset, chain.contractInput,
                                          survivingInput, hasL1Consumer, chain.sourceTensor, newOps, operationUpdated);
    }
    chain.contract->SetAsDeleted();
    return true;
}

void RemoveRedundantOpUtils::RewriteSliceContractSliceConsumer(
    Function& function, Operation& contract, Operation& consumer, bool isFullSlice,
    const std::vector<int64_t>& localOffset, const std::vector<SymbolicScalar>& localDynOffset,
    const LogicalTensorPtr& contractInput, LogicalTensorPtr& survivingInput, bool redirectEnabled,
    const LogicalTensorPtr& redirectTarget, std::vector<Operation*>& newOps, bool& operationUpdated)
{
    auto sliceOutput = consumer.GetOOperands().front();
    if (SliceRequiresL1(consumer) && contractInput->GetProducers().size() == 1) {
        TransferL1CopyAttrs(consumer, **contractInput->GetProducers().begin());
    }
    if (isFullSlice) {
        auto* precedingSlice = GetReplaceablePrecedingSlice(contract, consumer, survivingInput, sliceOutput);
        if (precedingSlice != nullptr) {
            sliceOutput->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
            // ReplaceOOperand detaches the old output from its producer.  Migrate all users first so
            // later operations do not retain a dangling tensor reference.
            ReplaceConsumers(precedingSlice->GetOOperands().front(), sliceOutput);
            precedingSlice->ReplaceOOperand(0, sliceOutput);
            survivingInput = sliceOutput;
        } else if (redirectEnabled && !SliceRequiresL1(consumer) && redirectTarget != nullptr) {
            // A non-L1 consumer cannot consume the L1 materialization directly in a mixed fan-out.
            // Redirect its full-slice users to the original source instead.
            ReplaceConsumers(sliceOutput, redirectTarget);
        } else {
            ReplaceConsumers(sliceOutput, survivingInput);
        }
    } else {
        GenerateContractSliceView(function, consumer, survivingInput, localOffset, localDynOffset, newOps,
                                  operationUpdated);
    }
    consumer.SetAsDeleted();
}

bool RemoveRedundantOpUtils::ProcessSliceContractSliceFanout(Function& function, const SliceContractSliceChain& chain,
                                                             std::vector<SliceContractSliceRewrite>& rewrites,
                                                             bool keepL1Fanout, bool canComposeOffsets,
                                                             std::vector<Operation*>& newOps, bool& operationUpdated)
{
    const bool hasL1Consumer = std::any_of(rewrites.begin(), rewrites.end(), [](const SliceContractSliceRewrite& r) {
        return SliceRequiresL1(*r.consumer);
    });
    const bool hasNonL1Consumer = std::any_of(rewrites.begin(), rewrites.end(), [](const SliceContractSliceRewrite& r) {
        return !SliceRequiresL1(*r.consumer);
    });
    // A dynamic mixed-memory fan-out cannot share a surviving input safely: transferring the L1
    // requirement to the common preceding slice would also make partial vector views read from L1.
    // Keep the materialization boundary until all offsets can be composed back to the source tensor.
    if (!keepL1Fanout && !canComposeOffsets && hasL1Consumer && hasNonL1Consumer) {
        return false;
    }
    if (keepL1Fanout) {
        FoldL1FanoutChain(chain, rewrites);
    } else if (canComposeOffsets) {
        RewriteComposableFanoutChain(chain, rewrites);
    } else if (!RewriteDynamicFanoutFallback(function, chain, rewrites, hasL1Consumer, newOps, operationUpdated)) {
        return false;
    }
    return true;
}

void RemoveRedundantOpUtils::ProcessSliceContractSliceSingle(Function& function, const SliceContractSliceChain& chain,
                                                             const SliceContractSliceRewrite& rewrite,
                                                             std::vector<Operation*>& newOps, bool& operationUpdated)
{
    // There is one consumer.  Preserve the established single-consumer behavior (including L1
    // requirement transfer and the partial-slice VIEW fallback), but keep it in this dedicated
    // slice-contract-slice pass so ProcessSingleContractMultiSlice never handles this pattern.
    auto* consumer = rewrite.consumer;
    if (rewrite.isFullSlice && !IsContractInputEqualSliceOutput(*chain.contract, *consumer)) {
        return;
    }

    std::set<Operation*> precedingSlices;
    if (!CollectPrecedingSlicesForL1Transfer(*consumer, std::vector<Operation*>{chain.contract}, precedingSlices)) {
        return;
    }
    TransferL1Requirement(precedingSlices);

    // Keep the replacement tensor explicit.  Replacing a full L1 slice may detach the original contract
    // input from its preceding slice; any later rewrite must consume the tensor that actually survived.
    LogicalTensorPtr survivingInput = chain.contractInput;
    RewriteSliceContractSliceConsumer(function, *chain.contract, *consumer, rewrite.isFullSlice, rewrite.localOffset,
                                      rewrite.localDynOffset, chain.contractInput, survivingInput, false, nullptr,
                                      newOps, operationUpdated);
    chain.contract->SetAsDeleted();
    operationUpdated = true;
}

Status RemoveRedundantOpUtils::ProcessSliceContractSlice(Function& function, std::vector<Operation*>& newOps,
                                                         bool& operationUpdated)
{
    auto opList = function.Operations().DuplicatedOpList();
    for (auto* op : opList) {
        if (op == nullptr || op->GetOpcode() != Opcode::OP_CONTRACT) {
            continue;
        }

        SliceContractSliceChain chain;
        if (!CollectSliceContractSliceChain(*op, chain)) {
            continue;
        }

        const bool isFanout = chain.consumers.size() > 1;
        const bool keepL1Fanout = isFanout && std::all_of(chain.consumers.begin(), chain.consumers.end(),
                                                          [](const Operation* consumer) {
                                                              return consumer != nullptr && SliceRequiresL1(*consumer);
                                                          });
        // A slice-contract prefix is the expansion of a view (or an equal-size assemble): the preceding
        // slice is always a real copy.  When its source is a materialized matmul result, the L0C copy-out
        // contract stays outside the chain, and every rewrite below only reads linear-memory tensors
        // (sourceTensor/contractInput), so folding is safe for single- and mixed-fanout consumers too.

        std::vector<SliceContractSliceRewrite> rewrites;
        bool canComposeOffsets = true;
        if (!CollectSliceContractSliceRewrites(chain, isFanout, keepL1Fanout, rewrites, canComposeOffsets)) {
            continue;
        }

        if (isFanout) {
            if (ProcessSliceContractSliceFanout(function, chain, rewrites, keepL1Fanout, canComposeOffsets, newOps,
                                                operationUpdated)) {
                operationUpdated = true;
            }
            continue;
        }
        ProcessSliceContractSliceSingle(function, chain, rewrites.front(), newOps, operationUpdated);
    }
    return SUCCESS;
}

Status RemoveRedundantOpUtils::ProcessMultiContractSingleSlice(Function& function, bool& operationUpdated)
{
    auto opList = function.Operations().DuplicatedOpList();
    for (auto& op : opList) {
        if (op == nullptr || op->GetOpcode() != Opcode::OP_SLICE || op->GetIOperands().empty() ||
            op->GetOOperands().empty() || !IsSliceEqualSizeMove(*op)) {
            continue;
        }
        auto sliceInput = op->GetIOperands().front();
        auto sliceOutput = op->GetOOperands().front();
        auto sliceInputConsumers = sliceInput->GetConsumers();
        if (sliceInputConsumers.size() != 1 || *sliceInputConsumers.begin() != op) {
            continue;
        }
        auto producers = sliceInput->GetProducers();
        if (producers.size() <= 1) {
            continue;
        }
        if (SliceRequiresL1(*op)) {
            continue;
        }
        bool canProcess = true;
        std::set<Operation*> precedingSlices;
        for (auto* producer : producers) {
            if (producer == nullptr || producer->GetOpcode() != Opcode::OP_CONTRACT ||
                producer->GetIOperands().size() != 1 || GetAssembleAttr(*producer) == nullptr ||
                !IsTensorOffset32BAligned(producer->GetIOperands().front())) {
                canProcess = false;
                break;
            }
            // SplitLargeFanoutTensor may materialize a matmul result as contract-slice before this fan-in.
            // Folding the outer chain would transfer its copy-out requirement across that L0C boundary.
            if (IsMatmulBackedContractInput(producer->GetIOperands().front())) {
                canProcess = false;
                break;
            }
        }
        if (canProcess) {
            canProcess = CollectPrecedingSlicesForL1Transfer(*op, producers, precedingSlices);
        }
        if (!canProcess) {
            continue;
        }
        TransferL1Requirement(precedingSlices);
        for (auto* producer : producers) {
            if (SliceRequiresL1(*op)) {
                auto contractInput = producer->GetIOperands().front();
                if (contractInput != nullptr && contractInput->GetProducers().size() == 1) {
                    TransferL1CopyAttrs(*op, **contractInput->GetProducers().begin());
                }
                auto contractAttr = GetAssembleAttr(*producer);
                auto newAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(contractAttr->Clone());
                newAttr->SetFromType(MemoryType::MEM_L1);
                producer->SetOpAttribute(newAttr);
            }
            producer->SetOpCode(Opcode::OP_ASSEMBLE);
            producer->ReplaceOOperand(0, sliceOutput);
        }
        op->SetAsDeleted();
        operationUpdated = true;
    }
    return SUCCESS;
}

void RemoveRedundantOpUtils::GenerateContractSliceView(Function& function, Operation& sliceOp,
                                                       const LogicalTensorPtr& inputTensor,
                                                       const std::vector<int64_t>& fromOffset,
                                                       const std::vector<SymbolicScalar>& fromDynOffset,
                                                       std::vector<Operation*>& newOps, bool& operationUpdated)
{
    auto sliceAttr = GetViewAttr(sliceOp);
    if (sliceAttr == nullptr || sliceOp.GetOOperands().empty()) {
        return;
    }
    auto sliceOutput = sliceOp.GetOOperands().front();
    LogicalTensorPtr newViewTensor = sliceOutput;
    if (!sliceOutput->GetConsumers().empty()) {
        std::vector<int64_t> curOffset(sliceOutput->shape.size(), 0);
        newViewTensor = irBuilder_.CreateTensorVar(sliceOutput->GetRawTensor(), curOffset, sliceOutput->shape,
                                                   std::vector<SymbolicScalar>{});
        newViewTensor->SetMemoryTypeBoth(sliceOutput->GetMemoryTypeOriginal());
        ReplaceConsumers(sliceOutput, newViewTensor);
    }
    auto viewAttribute = std::make_shared<ViewOpAttribute>(fromOffset, fromDynOffset, sliceAttr->GetToDynValidShape());
    auto targetMemory = SliceRequiresL1(sliceOp) ? MemoryType::MEM_L1 : sliceOutput->GetMemoryTypeToBe();
    if (targetMemory == MemoryType::MEM_L1) {
        newViewTensor->SetMemoryTypeBoth(targetMemory, true);
    }
    viewAttribute->SetToType(targetMemory);
    auto& newViewOp = PassOperationUtils::AddOperation(
        function, Opcode::OP_VIEW, {inputTensor}, {newViewTensor},
        [&viewAttribute](Operation& newOp) { newOp.SetOpAttribute(viewAttribute); });
    newOps.push_back(&newViewOp);
    operationUpdated = true;
}

bool RemoveRedundantOpUtils::ShouldSkipContractMultiSlice(
    const Operation& op, const LogicalTensorPtr& contractInput, const LogicalTensorPtr& contractOutput,
    const std::set<Operation*, LogicalTensor::CompareOp>& consumers)
{
    // A contract fed by a single slice belongs to the dedicated slice-contract-slice pass.  Do not let
    // the generic contract-multi-slice rewrite fold that three-op chain a second time.
    if (IsInputProducedBySingleSlice(contractInput) && contractOutput != nullptr) {
        const bool hasOnlySliceConsumers = !consumers.empty() &&
                                           std::all_of(consumers.begin(), consumers.end(),
                                                       [](const Operation* consumer) {
                                                           return consumer != nullptr && !consumer->IsDeleted() &&
                                                                  consumer->GetOpcode() == Opcode::OP_SLICE;
                                                       });
        if (hasOnlySliceConsumers) {
            return true;
        }
    }
    // A matmul result uses the L0C layout. Folding its contract-slice chain into views would make later
    // copy-outs interpret logical slice offsets as linear L0C pointer offsets.  Do not require the input
    // tensor to have exactly one producer here: after fanout splitting/reconnection, the producer set can
    // contain additional entries while an A_MUL_B result is still one of the producers.
    const auto& producers = contractInput->GetProducers();
    const bool hasMatmulProducer = std::any_of(producers.begin(), producers.end(), [](const Operation* producer) {
        return producer != nullptr && IsMatmulOpcode(producer->GetOpcode());
    });
    // Keep a contract-slice chain materialized when the contract consumes an A_MUL_B result.  The
    // contract may currently have only one slice consumer (for example, after fanout splitting), so
    // checking consumers.size() > 1 would let that chain through and fold CONTRACT+SLICE into VIEW.
    if (hasMatmulProducer && std::all_of(consumers.begin(), consumers.end(), [](const Operation* consumer) {
            return consumer != nullptr && consumer->GetOpcode() == Opcode::OP_SLICE;
        })) {
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "Skip CONTRACT[%d] with SLICE consumers because its input has a matmul producer.",
                          op.GetOpMagic());
        return true;
    }
    return false;
}

bool RemoveRedundantOpUtils::CollectContractSliceRewriteInfos(
    Operation& op, const LogicalTensorPtr& contractInput,
    const std::set<Operation*, LogicalTensor::CompareOp>& consumers,
    std::vector<ContractSliceRewriteInfo>& rewriteInfos, std::set<Operation*>& precedingSlices)
{
    if (std::any_of(consumers.begin(), consumers.end(),
                    [](const Operation* consumer) {
                        return consumer != nullptr && consumer->GetOpcode() == Opcode::OP_SLICE &&
                               SliceRequiresL1(*consumer);
                    }) &&
        contractInput->GetProducers().size() != 1) {
        return false;
    }
    for (auto* consumer : consumers) {
        if (consumer == nullptr || consumer->GetOpcode() != Opcode::OP_SLICE || consumer->GetOOperands().empty()) {
            return false;
        }
        auto sliceAttr = GetViewAttr(*consumer);
        auto contractAttr = GetAssembleAttr(op);
        std::vector<int64_t> localOffset;
        std::vector<SymbolicScalar> localDynOffset;
        if (sliceAttr == nullptr || contractAttr == nullptr ||
            !CalculateContractLocalSliceOffset(contractInput, consumer->GetOOperands().front(), *sliceAttr,
                                               *contractAttr, localOffset, localDynOffset)) {
            return false;
        }
        // 对消产生的 view 直接以 localOffset 读取 contractInput 共享 buffer；偏移非 32B 对齐时
        // VEC consumer 会产生非对齐 UB 访问，此时保留 CONTRACT+SLICE 拷贝路径不做对消。
        if (!IsViewFromOffset32BAligned(contractInput, localOffset, localDynOffset)) {
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "Skip contract-slice rewrite: localOffset %s of CONTRACT[%d] consumer SLICE[%d] "
                              "is not 32B aligned.",
                              IntVectorToString(localOffset).c_str(), op.GetOpMagic(), consumer->GetOpMagic());
            return false;
        }
        bool isFullSlice = IsFullSliceOfContractInput(contractInput, consumer->GetOOperands().front(), localOffset,
                                                      localDynOffset);
        if (isFullSlice && !IsContractInputEqualSliceOutput(op, *consumer)) {
            return false;
        }
        if (!CollectPrecedingSlicesForL1Transfer(*consumer, std::vector<Operation*>{&op}, precedingSlices)) {
            return false;
        }
        rewriteInfos.push_back({consumer, std::move(localOffset), std::move(localDynOffset), isFullSlice});
    }
    return true;
}

LogicalTensorPtr RemoveRedundantOpUtils::GetMixedConsumerRedirectTarget(const LogicalTensorPtr& contractInput)
{
    if (contractInput == nullptr || contractInput->GetProducers().size() != 1) {
        return nullptr;
    }
    auto* producer = *contractInput->GetProducers().begin();
    if (producer == nullptr || producer->GetOpcode() != Opcode::OP_SLICE || producer->GetIOperands().size() != 1) {
        return nullptr;
    }
    return producer->GetIOperands().front();
}

void RemoveRedundantOpUtils::RewriteContractMultiSliceConsumers(Function& function, Operation& op,
                                                                const LogicalTensorPtr& contractInput,
                                                                std::vector<ContractSliceRewriteInfo>& rewriteInfos,
                                                                std::set<Operation*>& precedingSlices,
                                                                std::vector<Operation*>& newOps, bool& operationUpdated)
{
    // Process L1 consumers first.  Their requirement transfer can change the materialization tensor
    // used by the remaining contract-slice consumers.
    std::stable_sort(rewriteInfos.begin(), rewriteInfos.end(),
                     [](const ContractSliceRewriteInfo& lhs, const ContractSliceRewriteInfo& rhs) {
                         return SliceRequiresL1(*lhs.consumer) && !SliceRequiresL1(*rhs.consumer);
                     });
    TransferL1Requirement(precedingSlices);

    // A full-slice consumer may replace the output of the preceding materialization slice.  Keep the
    // tensor that remains connected to the graph so subsequent consumers and generated views are never
    // redirected through a detached contract input.
    const bool hasL1Consumer = std::any_of(
        rewriteInfos.begin(), rewriteInfos.end(),
        [](const ContractSliceRewriteInfo& rewriteInfo) { return SliceRequiresL1(*rewriteInfo.consumer); });
    LogicalTensorPtr redirectTarget = GetMixedConsumerRedirectTarget(contractInput);
    LogicalTensorPtr survivingInput = contractInput;
    for (auto& rewriteInfo : rewriteInfos) {
        RewriteSliceContractSliceConsumer(function, op, *rewriteInfo.consumer, rewriteInfo.isFullSlice,
                                          rewriteInfo.localOffset, rewriteInfo.localDynOffset, contractInput,
                                          survivingInput, hasL1Consumer, redirectTarget, newOps, operationUpdated);
        operationUpdated = true;
    }
    op.SetAsDeleted();
    operationUpdated = true;
}

Status RemoveRedundantOpUtils::ProcessSingleContractMultiSlice(Function& function, std::vector<Operation*>& newOps,
                                                               bool& operationUpdated)
{
    auto opList = function.Operations().DuplicatedOpList();
    for (auto& op : opList) {
        if (op == nullptr || op->IsDeleted() || op->GetOpcode() != Opcode::OP_CONTRACT || op->GetIOperands().empty() ||
            op->GetOOperands().empty() || GetAssembleAttr(*op) == nullptr) {
            continue;
        }
        auto contractInput = op->GetIOperands().front();
        auto contractOutput = op->GetOOperands().front();
        if (contractInput == nullptr || contractOutput == nullptr || contractOutput->GetProducers().size() != 1 ||
            *contractOutput->GetProducers().begin() != op) {
            // A contract output assembled from several slice-contract chains has no unique layout owner.
            // Keeping the chains materialized is required because their offsets cannot be composed independently.
            continue;
        }
        auto consumers = contractOutput->GetConsumers();
        if (consumers.empty()) {
            continue;
        }
        if (ShouldSkipContractMultiSlice(*op, contractInput, contractOutput, consumers)) {
            continue;
        }
        std::vector<ContractSliceRewriteInfo> rewriteInfos;
        std::set<Operation*> precedingSlices;
        if (!CollectContractSliceRewriteInfos(*op, contractInput, consumers, rewriteInfos, precedingSlices)) {
            continue;
        }
        RewriteContractMultiSliceConsumers(function, *op, contractInput, rewriteInfos, precedingSlices, newOps,
                                           operationUpdated);
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu
