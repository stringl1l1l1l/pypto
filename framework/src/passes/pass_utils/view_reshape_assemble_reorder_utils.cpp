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
 * \file view_reshape_assemble_reorder_utils.cpp
 * \brief utils of view/assemble and reshape operation reordering
 */

#include "view_reshape_assemble_reorder_utils.h"

#include <algorithm>
#include <climits>

#include "interface/operation/attribute.h"
#include "interface/operation/attr_holder.h"
#include "merge_view_assemble_utils.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/pass_token_utils.h"

#define MODULE_NAME "ViewReshapeAssembleReorderUtils"

namespace npu::tile_fwk {
namespace {
constexpr int64_t UNKNOWN_DIM = -1;

bool IsSameDim(int64_t lhs, int64_t rhs) { return lhs == rhs || (lhs == UNKNOWN_DIM && rhs == UNKNOWN_DIM); }

bool ProductShape(const Shape& shape, size_t begin, size_t end, int64_t& product)
{
    product = 1;
    for (size_t i = begin; i < end; ++i) {
        if (shape[i] == UNKNOWN_DIM) {
            product = UNKNOWN_DIM;
            return true;
        }
        if (shape[i] <= 0) {
            return false;
        }
        product *= shape[i];
    }
    return true;
}

SymbolicScalar ProductDynShape(const SymbolicShape& shape, size_t begin, size_t end)
{
    SymbolicScalar product(1);
    for (size_t i = begin; i < end; ++i) {
        product = (product * shape[i]).Simplify();
    }
    return product;
}

bool HasOnlyFirstUnknownDim(const Shape& shape)
{
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == UNKNOWN_DIM && i != 0) {
            return false;
        }
        if (shape[i] <= 0 && shape[i] != UNKNOWN_DIM) {
            return false;
        }
    }
    return true;
}

bool HasConcreteShape(const Shape& shape)
{
    return std::all_of(shape.begin(), shape.end(), [](int64_t dim) { return dim > 0; });
}

bool HasAtMostFirstUnknownDim(const Shape& shape) { return HasConcreteShape(shape) || HasOnlyFirstUnknownDim(shape); }

bool HasNegativeDynDim(const SymbolicShape& dynShape)
{
    return std::any_of(dynShape.begin(), dynShape.end(),
                       [](const SymbolicScalar& dim) { return dim.ConcreteValid() && dim.Concrete() < 0; });
}

bool HasNegativeDynRawShape(const LogicalTensorPtr& tensor, const SymbolicShape& dynShape)
{
    if (tensor == nullptr) {
        return false;
    }
    const auto& rawTensor = tensor->GetRawTensor();
    return rawTensor != nullptr && rawTensor->GetDynRawShape().size() == dynShape.size() &&
           HasNegativeDynDim(rawTensor->GetDynRawShape());
}

bool HasTargetMatmulOp(Function& function)
{
    auto opList = function.Operations().DuplicatedOpList();
    return std::any_of(opList.begin(), opList.end(), [](Operation* op) {
        return op->GetOpcode() == Opcode::OP_A_MULACC_B || op->GetOpcode() == Opcode::OP_A_MUL_B;
    });
}

bool CanReorderFunction(Function& function)
{
    for (const auto& incast : function.GetIncast()) {
        if (!HasAtMostFirstUnknownDim(incast->GetShape())) {
            return false;
        }
    }
    for (const auto& outcast : function.GetOutcast()) {
        if (!HasAtMostFirstUnknownDim(outcast->GetShape())) {
            return false;
        }
    }
    return true;
}

bool HasSingleInput(const Operation& op) { return op.GetIOperands().size() == 1; }

bool HasSingleOutput(const Operation& op) { return op.GetOOperands().size() == 1; }

Operation* GetSingleProducer(const LogicalTensorPtr& tensor)
{
    if (tensor == nullptr || tensor->GetProducers().size() != 1) {
        return nullptr;
    }
    return *tensor->GetProducers().begin();
}

Operation* GetSingleConsumer(const LogicalTensorPtr& tensor)
{
    if (tensor == nullptr || tensor->GetConsumers().size() != 1) {
        return nullptr;
    }
    return *tensor->GetConsumers().begin();
}

bool HasCascadedViewsBeforeReshape(Function& function)
{
    for (const auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_RESHAPE || !HasSingleInput(op)) {
            continue;
        }
        auto input = op.GetIOperands().front();
        Operation* firstView = GetSingleProducer(input);
        if (firstView == nullptr || firstView->GetOpcode() != Opcode::OP_VIEW || !HasSingleInput(*firstView)) {
            continue;
        }
        auto firstViewInput = firstView->GetIOperands().front();
        Operation* secondView = GetSingleProducer(firstViewInput);
        if (secondView != nullptr && secondView->GetOpcode() == Opcode::OP_VIEW) {
            return true;
        }
    }
    return false;
}

bool HasCascadedAssemblesAfterReshape(Function& function)
{
    for (const auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_RESHAPE) {
            continue;
        }
        if (!HasSingleOutput(op)) {
            continue;
        }
        auto output = op.GetOOperands().front();
        if (output == nullptr) {
            continue;
        }
        for (Operation* consumer : output->GetConsumers()) {
            if (consumer == nullptr || consumer->GetOpcode() != Opcode::OP_ASSEMBLE) {
                continue;
            }
            if (!HasSingleOutput(*consumer)) {
                continue;
            }
            auto assembleOutput = consumer->GetOOperands().front();
            if (assembleOutput == nullptr) {
                continue;
            }
            for (Operation* nextConsumer : assembleOutput->GetConsumers()) {
                if (nextConsumer != nullptr && nextConsumer->GetOpcode() == Opcode::OP_ASSEMBLE) {
                    return true;
                }
            }
        }
    }
    return false;
}

bool IsRegionWithinShape(const Offset& offset, const Shape& regionShape, const Shape& baseShape)
{
    if (offset.size() != regionShape.size() || regionShape.size() != baseShape.size() ||
        !HasConcreteShape(regionShape) || !HasConcreteShape(baseShape)) {
        return false;
    }
    for (size_t idx = 0; idx < baseShape.size(); ++idx) {
        if (offset[idx] < 0 || offset[idx] + regionShape[idx] > baseShape[idx]) {
            return false;
        }
    }
    return true;
}

SymbolicShape ShapeToSymbolic(const Shape& shape) { return SymbolicScalar::FromConcrete(shape); }

Offset SymbolicToStaticOffset(const SymbolicOffset& dynOffset)
{
    Offset staticOffset;
    staticOffset.reserve(dynOffset.size());
    for (const auto& item : dynOffset) {
        staticOffset.emplace_back(item.ConcreteValid() ? item.Concrete() : 0);
    }
    return staticOffset;
}

std::vector<SymbolicScalar> BuildStrides(const SymbolicShape& dynShape)
{
    std::vector<SymbolicScalar> strides(dynShape.size());
    SymbolicScalar curStride(1);
    for (size_t i = dynShape.size(); i > 0; --i) {
        strides[i - 1] = curStride;
        curStride = (curStride * dynShape[i - 1]).Simplify();
    }
    return strides;
}

SymbolicScalar LinearizeOffset(const SymbolicOffset& dynOffset, const SymbolicShape& dynShape)
{
    auto strides = BuildStrides(dynShape);
    SymbolicScalar linearIndex(0);
    for (size_t i = 0; i < dynOffset.size(); ++i) {
        linearIndex = (linearIndex + dynOffset[i] * strides[i]).Simplify();
    }
    return linearIndex;
}

SymbolicOffset DelinearizeOffset(SymbolicScalar linearIndex, const SymbolicShape& dynShape)
{
    auto strides = BuildStrides(dynShape);
    SymbolicOffset dynOffset(dynShape.size());
    for (size_t i = 0; i < dynShape.size(); ++i) {
        dynOffset[i] = (linearIndex / strides[i]).Simplify();
        linearIndex = (linearIndex % strides[i]).Simplify();
    }
    return dynOffset;
}

SymbolicOffset AddOffsets(const SymbolicOffset& lhs, const SymbolicOffset& rhs)
{
    SymbolicOffset result;
    result.reserve(lhs.size());
    for (size_t i = 0; i < lhs.size(); ++i) {
        result.emplace_back((lhs[i] + rhs[i]).Simplify());
    }
    return result;
}

void SetMetadataReshapeAttrs(Operation& reshapeOp, const LogicalTensorPtr& output, const SymbolicShape& dynShape)
{
    reshapeOp.SetAttribute("reshape", output->GetShape());
    reshapeOp.SetAttribute(OP_ATTR_PREFIX + "isInplace", true);
    reshapeOp.SetAttribute(OP_ATTR_PREFIX + "validShape", dynShape);
    output->UpdateDynValidShape(dynShape);
}

void MoveTokenDependencyToReplacement(Function& function, Operation& source, Operation& replacement)
{
    for (const auto& token : source.tokens_) {
        PassTokenUtils::AddTokenConsumer(function, token, replacement);
    }
    PassTokenUtils::MoveResultTokensToProducers(function, {&source}, {&replacement}, {});
}
} // namespace

Status ViewReshapeAssembleReorderUtils::ReorderViewReshapeAssemble(Function& function)
{
    ViewReshapeAssembleReorderUtils utils;
    return utils.Process(function);
}

bool ViewReshapeAssembleReorderUtils::RemapOffsetBackwardThroughReshape(
    const LogicalTensorPtr& reshapeInput, const LogicalTensorPtr& reshapeOutput, const Shape& outputBaseShape,
    const SymbolicShape& outputBaseDynShape, const Offset& outputOffset, const SymbolicOffset& outputDynOffset,
    Shape& inputBaseShape, SymbolicShape& inputBaseDynShape, Offset& inputOffset, SymbolicOffset& inputDynOffset)
{
    if (reshapeInput == nullptr || reshapeOutput == nullptr || outputBaseShape.size() != outputBaseDynShape.size()) {
        return false;
    }
    AxisPlan axisPlan;
    if (!BuildAxisPlan(reshapeInput->GetShape(), reshapeOutput->GetShape(), axisPlan) ||
        !ApplyBackwardShape(outputBaseShape, outputBaseDynShape, axisPlan, inputBaseShape, inputBaseDynShape)) {
        return false;
    }
    RemapResult remap;
    if (!RemapOffset(outputOffset, outputDynOffset, outputBaseShape, outputBaseDynShape, inputBaseShape,
                     inputBaseDynShape, remap)) {
        return false;
    }
    inputOffset = std::move(remap.staticOffset);
    inputDynOffset = std::move(remap.dynOffset);
    return true;
}

Status ViewReshapeAssembleReorderUtils::Process(Function& function)
{
    if (!HasTargetMatmulOp(function)) {
        return SUCCESS;
    }
    if (!CanReorderFunction(function)) {
        return SUCCESS;
    }

    ClearRecords();
    Status status = ProcessOperations(function);
    if (status != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "ProcessOperations failed.");
        return status;
    }
    if (!HasRecords()) {
        return SUCCESS;
    }
    if (HasCascadedPattern(function)) {
        status = MergeViewAssembleUtils::MergeViewAssemble(function);
        if (status != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "Merge assemble and view failed.");
            return status;
        }
    }

    ClearRecords();
    status = ProcessOperations(function);
    if (status != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "ProcessOperations after MergeViewAssemble failed.");
        return status;
    }
    if (!HasRecords()) {
        return SUCCESS;
    }
    AppendViewReshapeRecords(function);
    AppendViewReshapeFanoutRecords(function);
    AppendReshapeAssembleRecords(function);
    AppendReshapeAssembleFaninRecords(function);
    CleanUp(function);
    return SUCCESS;
}

void ViewReshapeAssembleReorderUtils::ClearRecords()
{
    visitedOp_.clear();
    viewReshapeRecords_.clear();
    viewReshapeFanoutRecords_.clear();
    reshapeAssembleRecords_.clear();
    reshapeAssembleFaninRecords_.clear();
}

bool ViewReshapeAssembleReorderUtils::HasRecords() const
{
    return !viewReshapeRecords_.empty() || !viewReshapeFanoutRecords_.empty() || !reshapeAssembleRecords_.empty() ||
           !reshapeAssembleFaninRecords_.empty();
}

Status ViewReshapeAssembleReorderUtils::ProcessOperations(Function& function)
{
    for (auto& op : function.Operations()) {
        if (visitedOp_.count(op.GetOpMagic()) != 0) {
            continue;
        }
        if (op.GetOpcode() != Opcode::OP_RESHAPE) {
            continue;
        }
        Operation* precedingViewOp = GetPrecedingViewOp(op);
        Status status;
        if (precedingViewOp != nullptr) {
            status = TryRecordViewReshape(function, *precedingViewOp);
        } else {
            Operation* followingAssembleOp = GetFollowingAssembleOp(op);
            if (followingAssembleOp == nullptr) {
                continue;
            }
            status = TryRecordReshapeAssemble(function, op);
        }
        if (status != SUCCESS) {
            return status;
        }
    }
    return SUCCESS;
}

bool ViewReshapeAssembleReorderUtils::GetChainMatch(Operation& firstOp, Opcode secondOpcode, ChainMatch& match)
{
    if (!HasSingleInput(firstOp) || !HasSingleOutput(firstOp)) {
        return false;
    }
    match.input = firstOp.GetIOperands().front();
    match.middle = firstOp.GetOOperands().front();
    if (match.input == nullptr || match.middle == nullptr || GetSingleProducer(match.middle) == nullptr ||
        GetSingleConsumer(match.middle) == nullptr) {
        return false;
    }
    match.secondOp = GetSingleConsumer(match.middle);
    if (match.secondOp->GetOpcode() != secondOpcode || !HasSingleInput(*match.secondOp) ||
        !HasSingleOutput(*match.secondOp)) {
        return false;
    }
    match.output = match.secondOp->GetOOperands().front();
    return match.output != nullptr;
}

Operation* ViewReshapeAssembleReorderUtils::GetPrecedingViewOp(Operation& reshapeOp)
{
    if (!HasSingleInput(reshapeOp)) {
        return nullptr;
    }
    auto input = reshapeOp.GetIOperands().front();
    Operation* producer = GetSingleProducer(input);
    if (producer == nullptr || producer->GetOpcode() != Opcode::OP_VIEW) {
        return nullptr;
    }
    return producer;
}

Operation* ViewReshapeAssembleReorderUtils::GetFollowingAssembleOp(Operation& reshapeOp)
{
    if (!HasSingleOutput(reshapeOp)) {
        return nullptr;
    }
    auto output = reshapeOp.GetOOperands().front();
    Operation* consumer = GetSingleConsumer(output);
    if (consumer == nullptr || consumer->GetOpcode() != Opcode::OP_ASSEMBLE) {
        return nullptr;
    }
    return consumer;
}

bool ViewReshapeAssembleReorderUtils::ValidateChainShapes(const ChainMatch& match)
{
    return HasOnlyFirstUnknownDim(match.input->GetShape()) && HasOnlyFirstUnknownDim(match.middle->GetShape()) &&
           HasOnlyFirstUnknownDim(match.output->GetShape());
}

Status ViewReshapeAssembleReorderUtils::TryRecordViewReshape(Function& function, Operation& viewOp)
{
    ChainMatch match;
    if (!GetChainMatch(viewOp, Opcode::OP_RESHAPE, match) || !IsScopeCompatible(viewOp, *match.secondOp)) {
        return SUCCESS;
    }
    auto& reshapeOp = *match.secondOp;
    auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(viewOp.GetOpAttribute());
    if (viewAttr == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "View op[%d] missing ViewOpAttribute.", viewOp.GetOpMagic());
        return FAILED;
    }
    if (!ValidateChainShapes(match)) {
        return SUCCESS;
    }
    AxisPlan axisPlan;
    if (!BuildAxisPlan(match.middle->GetShape(), match.output->GetShape(), axisPlan)) {
        return SUCCESS;
    }
    SymbolicShape inputDynShape;
    if (!GetSymbolicShape(match.input, inputDynShape)) {
        return SUCCESS;
    }
    Shape reshapeOutputShape;
    SymbolicShape reshapeDynShape;
    if (!ApplyForwardShape(match.input->GetShape(), inputDynShape, axisPlan, reshapeOutputShape, reshapeDynShape)) {
        return SUCCESS;
    }
    if (!AreCollapsedGroupsContiguous(viewAttr->GetFromOffset(), match.middle->GetShape(), match.input->GetShape(),
                                      axisPlan, true)) {
        return TryRecordViewReshapeFanout(function, viewOp, reshapeOp, match, *viewAttr, reshapeOutputShape,
                                          reshapeDynShape, inputDynShape);
    }
    RemapResult remap;
    if (!RemapOffset(viewAttr->GetFromOffset(), viewAttr->GetFromDynOffset(), match.input->GetShape(), inputDynShape,
                     reshapeOutputShape, reshapeDynShape, remap)) {
        return SUCCESS;
    }
    int64_t copyInModeValue = 0;
    bool hasCopyInMode = viewOp.GetAttr<int64_t>("op_attr_copy_in_mode", copyInModeValue);
    viewReshapeRecords_.emplace_back(ViewReshapeRecord{
        &viewOp, &reshapeOp, match.input, match.output,
        irBuilder_.CreateTensorVar(function, match.input->Datatype(), reshapeOutputShape, reshapeDynShape,
                                   match.input->Format()),
        remap.staticOffset, remap.dynOffset, reshapeDynShape, viewAttr->GetTo(), hasCopyInMode,
        std::move(copyInModeValue), GetFirstSpan(viewOp, reshapeOp), GetChainScopeInfo(viewOp, reshapeOp)});
    visitedOp_.insert(viewOp.GetOpMagic());
    visitedOp_.insert(reshapeOp.GetOpMagic());
    return SUCCESS;
}

Status ViewReshapeAssembleReorderUtils::TryRecordViewReshapeFanout(Function& function, Operation& viewOp,
                                                                   Operation& reshapeOp, const ChainMatch& match,
                                                                   const ViewOpAttribute& viewAttr,
                                                                   const Shape& reshapeOutputShape,
                                                                   const SymbolicShape& reshapeDynShape,
                                                                   const SymbolicShape& inputDynShape)
{
    auto consumers = match.output->GetConsumers();
    if (consumers.size() <= 1) {
        return SUCCESS;
    }
    SymbolicShape middleDynShape;
    if (!GetSymbolicShape(match.middle, middleDynShape)) {
        return SUCCESS;
    }
    SymbolicShape compactDynShape;
    if (!GetSymbolicShape(match.output, compactDynShape)) {
        return SUCCESS;
    }

    ViewReshapeFanoutRecord record{&viewOp,
                                   &reshapeOp,
                                   match.input,
                                   irBuilder_.CreateTensorVar(function, match.input->Datatype(), reshapeOutputShape,
                                                              reshapeDynShape, match.input->Format()),
                                   reshapeDynShape,
                                   GetFirstSpan(viewOp, reshapeOp),
                                   GetChainScopeInfo(viewOp, reshapeOp),
                                   {}};

    for (Operation* consumer : consumers) {
        if (consumer == nullptr) {
            return SUCCESS;
        }
        FanoutViewRecord fanoutRecord;
        bool canReorder = false;
        Status status = TryCollectFanoutViewRecord(reshapeOp, *consumer, match, viewAttr, compactDynShape,
                                                   middleDynShape, inputDynShape, reshapeOutputShape, reshapeDynShape,
                                                   fanoutRecord, canReorder);
        if (status != SUCCESS) {
            return status;
        }
        if (!canReorder) {
            return SUCCESS;
        }
        record.fanoutViews.emplace_back(std::move(fanoutRecord));
    }
    viewReshapeFanoutRecords_.emplace_back(std::move(record));
    MarkViewReshapeFanoutVisited(viewOp, reshapeOp, viewReshapeFanoutRecords_.back());
    return SUCCESS;
}

Status ViewReshapeAssembleReorderUtils::TryCollectFanoutViewRecord(
    Operation& reshapeOp, Operation& consumer, const ChainMatch& match, const ViewOpAttribute& viewAttr,
    const SymbolicShape& compactDynShape, const SymbolicShape& middleDynShape, const SymbolicShape& inputDynShape,
    const Shape& reshapeOutputShape, const SymbolicShape& reshapeDynShape, FanoutViewRecord& fanoutRecord,
    bool& canReorder)
{
    canReorder = false;
    if (visitedOp_.count(consumer.GetOpMagic()) != 0 || consumer.GetOpcode() != Opcode::OP_VIEW ||
        !IsScopeCompatible(reshapeOp, consumer) || !HasSingleInput(consumer) || !HasSingleOutput(consumer)) {
        return SUCCESS;
    }
    auto fanoutAttr = std::dynamic_pointer_cast<ViewOpAttribute>(consumer.GetOpAttribute());
    if (fanoutAttr == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "View op[%d] missing ViewOpAttribute.", consumer.GetOpMagic());
        return FAILED;
    }
    auto fanoutOutput = consumer.GetOOperands().front();
    if (fanoutOutput == nullptr) {
        return SUCCESS;
    }
    RemapResult remap;
    if (!RemapFanoutViewOffset(viewAttr.GetFromOffset(), viewAttr.GetFromDynOffset(), fanoutAttr->GetFromOffset(),
                               fanoutAttr->GetFromDynOffset(), match.output->GetShape(), compactDynShape,
                               match.middle->GetShape(), middleDynShape, match.input->GetShape(), inputDynShape,
                               reshapeOutputShape, reshapeDynShape, remap)) {
        return SUCCESS;
    }
    auto outputDynShape = GetSymbolicShapeOrStatic(fanoutOutput);
    int64_t copyInModeValue = 0;
    bool hasCopyInMode = consumer.GetAttr<int64_t>("op_attr_copy_in_mode", copyInModeValue);
    fanoutRecord = FanoutViewRecord{&consumer,
                                    fanoutOutput,
                                    remap.staticOffset,
                                    remap.dynOffset,
                                    outputDynShape,
                                    fanoutAttr->GetTo(),
                                    hasCopyInMode,
                                    std::move(copyInModeValue),
                                    GetFirstSpan(reshapeOp, consumer),
                                    GetChainScopeInfo(reshapeOp, consumer)};
    canReorder = true;
    return SUCCESS;
}

Status ViewReshapeAssembleReorderUtils::TryRecordReshapeAssemble(Function& function, Operation& reshapeOp)
{
    ChainMatch match;
    if (!GetChainMatch(reshapeOp, Opcode::OP_ASSEMBLE, match) || !IsScopeCompatible(reshapeOp, *match.secondOp)) {
        return SUCCESS;
    }
    auto& assembleOp = *match.secondOp;
    auto assembleAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(assembleOp.GetOpAttribute());
    if (assembleAttr == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Assemble op[%d] missing AssembleOpAttribute.", assembleOp.GetOpMagic());
        return FAILED;
    }
    if (!ValidateChainShapes(match)) {
        return SUCCESS;
    }
    // Skip reorder when reshape input is produced by SUB op (sub->reshape->assemble),
    // reordering such chains breaks the read-modify-write dependency on state tensors.
    Operation* producer = GetSingleProducer(match.input);
    if (producer != nullptr && producer->GetOpcode() == Opcode::OP_SUB) {
        return SUCCESS;
    }
    SymbolicShape inputDynShape;
    SymbolicShape middleDynShape;
    AxisPlan axisPlan;
    if (!BuildAxisPlan(match.input->GetShape(), match.middle->GetShape(), axisPlan)) {
        return SUCCESS;
    }
    SymbolicShape outputDynShape;
    if (!GetChainSymbolicShapes(match, inputDynShape, middleDynShape, outputDynShape)) {
        return SUCCESS;
    }
    Shape assembleOutputShape;
    SymbolicShape assembleDynShape;
    if (!ApplyBackwardShape(match.output->GetShape(), outputDynShape, axisPlan, assembleOutputShape,
                            assembleDynShape)) {
        return SUCCESS;
    }
    if (!AreCollapsedGroupsContiguous(assembleAttr->GetToOffset(), match.middle->GetShape(), match.output->GetShape(),
                                      axisPlan, false)) {
        return TryRecordReshapeAssembleFanin(function, reshapeOp, assembleOp, match, *assembleAttr, axisPlan);
    }
    return TryRecordDirectReshapeAssemble(function, reshapeOp, assembleOp, match, *assembleAttr, assembleOutputShape,
                                          assembleDynShape, middleDynShape, outputDynShape);
}

Status ViewReshapeAssembleReorderUtils::TryRecordDirectReshapeAssemble(
    Function& function, Operation& reshapeOp, Operation& assembleOp, const ChainMatch& match,
    const AssembleOpAttribute& assembleAttr, const Shape& assembleOutputShape, const SymbolicShape& assembleDynShape,
    const SymbolicShape& middleDynShape, const SymbolicShape& outputDynShape)
{
    RemapResult remap;
    if (!RemapOffset(assembleAttr.GetToOffset(), assembleAttr.GetToDynOffset(), match.output->GetShape(),
                     outputDynShape, assembleOutputShape, assembleDynShape, remap)) {
        return SUCCESS;
    }
    for (size_t i = 0; i < middleDynShape.size(); ++i) {
        if (!middleDynShape[i].ConcreteValid()) {
            return SUCCESS;
        }
    }
    SymbolicShape finalOutputDynShape;
    if (!BuildAssembledValidShape(assembleAttr.GetToOffset(), assembleAttr.GetToDynOffset(), middleDynShape,
                                  match.output->GetShape().size(), finalOutputDynShape)) {
        return SUCCESS;
    }
    LogicalTensorPtr assembleOutput = irBuilder_.CreateTensorVar(
        function, match.output->Datatype(), assembleOutputShape, assembleDynShape, match.output->Format());
    const auto& outcasts = function.GetOutcast();
    if (std::find(outcasts.begin(), outcasts.end(), match.output) != outcasts.end()) {
        assembleOutput->tensor->actualRawmagic = match.output->GetRawMagic();
        assembleOutput->tensor->SetSymbol(match.output->Symbol());
    }
    reshapeAssembleRecords_.emplace_back(
        ReshapeAssembleRecord{&reshapeOp, &assembleOp, match.input, match.output, assembleOutput, remap.staticOffset,
                              remap.dynOffset, assembleDynShape, finalOutputDynShape, assembleAttr.GetFrom(),
                              GetFirstSpan(reshapeOp, assembleOp), GetChainScopeInfo(reshapeOp, assembleOp)});
    visitedOp_.insert(reshapeOp.GetOpMagic());
    visitedOp_.insert(assembleOp.GetOpMagic());
    return SUCCESS;
}

Status ViewReshapeAssembleReorderUtils::TryRecordReshapeAssembleFanin(Function& function, Operation& reshapeOp,
                                                                      Operation& assembleOp, const ChainMatch& match,
                                                                      const AssembleOpAttribute& assembleAttr,
                                                                      const AxisPlan& axisPlan)
{
    auto producers = match.input->GetProducers();
    if (producers.size() <= 1 || GetSingleConsumer(match.input) == nullptr) {
        return SUCCESS;
    }
    SymbolicShape inputDynShape;
    SymbolicShape middleDynShape;
    SymbolicShape outputDynShape;
    if (!GetChainSymbolicShapes(match, inputDynShape, middleDynShape, outputDynShape)) {
        return SUCCESS;
    }
    Shape reshapeInputShape;
    SymbolicShape reshapeDynShape;
    if (!ApplyBackwardShape(match.output->GetShape(), outputDynShape, axisPlan, reshapeInputShape, reshapeDynShape)) {
        return SUCCESS;
    }
    SymbolicShape finalOutputDynShape;
    if (!BuildAssembledValidShape(assembleAttr.GetToOffset(), assembleAttr.GetToDynOffset(), middleDynShape,
                                  match.output->GetShape().size(), finalOutputDynShape)) {
        return SUCCESS;
    }

    ReshapeAssembleFaninRecord record{&reshapeOp,
                                      &assembleOp,
                                      match.output,
                                      irBuilder_.CreateTensorVar(function, match.output->Datatype(), reshapeInputShape,
                                                                 reshapeDynShape, match.output->Format()),
                                      reshapeDynShape,
                                      finalOutputDynShape,
                                      GetFirstSpan(reshapeOp, assembleOp),
                                      GetChainScopeInfo(reshapeOp, assembleOp),
                                      {}};

    for (Operation* producer : producers) {
        if (producer == nullptr) {
            return SUCCESS;
        }
        FaninAssembleRecord faninRecord;
        bool canReorder = false;
        Status status = TryCollectFaninAssembleRecord(reshapeOp, assembleOp, *producer, match, assembleAttr,
                                                      inputDynShape, middleDynShape, outputDynShape, reshapeInputShape,
                                                      reshapeDynShape, faninRecord, canReorder);
        if (status != SUCCESS) {
            return status;
        }
        if (!canReorder) {
            return SUCCESS;
        }
        record.faninAssembles.emplace_back(std::move(faninRecord));
    }

    reshapeAssembleFaninRecords_.emplace_back(std::move(record));
    MarkReshapeAssembleFaninVisited(reshapeOp, assembleOp, reshapeAssembleFaninRecords_.back());
    return SUCCESS;
}

void ViewReshapeAssembleReorderUtils::MarkViewReshapeFanoutVisited(Operation& viewOp, Operation& reshapeOp,
                                                                   const ViewReshapeFanoutRecord& record)
{
    visitedOp_.insert(viewOp.GetOpMagic());
    visitedOp_.insert(reshapeOp.GetOpMagic());
    for (const auto& fanoutRecord : record.fanoutViews) {
        visitedOp_.insert(fanoutRecord.viewOp->GetOpMagic());
    }
}

void ViewReshapeAssembleReorderUtils::MarkReshapeAssembleFaninVisited(Operation& reshapeOp, Operation& assembleOp,
                                                                      const ReshapeAssembleFaninRecord& record)
{
    visitedOp_.insert(reshapeOp.GetOpMagic());
    visitedOp_.insert(assembleOp.GetOpMagic());
    for (const auto& faninRecord : record.faninAssembles) {
        visitedOp_.insert(faninRecord.assembleOp->GetOpMagic());
    }
}

Status ViewReshapeAssembleReorderUtils::TryCollectFaninAssembleRecord(
    Operation& reshapeOp, Operation& assembleOp, Operation& producer, const ChainMatch& match,
    const AssembleOpAttribute& assembleAttr, const SymbolicShape& inputDynShape, const SymbolicShape& middleDynShape,
    const SymbolicShape& outputDynShape, const Shape& reshapeInputShape, const SymbolicShape& reshapeDynShape,
    FaninAssembleRecord& faninRecord, bool& canReorder)
{
    canReorder = false;
    if (visitedOp_.count(producer.GetOpMagic()) != 0 || producer.GetOpcode() != Opcode::OP_ASSEMBLE ||
        !IsScopeCompatible(producer, reshapeOp) || !IsScopeCompatible(producer, assembleOp) ||
        !HasSingleInput(producer) || !HasSingleOutput(producer) || producer.GetOOperands().front() != match.input) {
        return SUCCESS;
    }
    auto producerAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(producer.GetOpAttribute());
    if (producerAttr == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Assemble op[%d] missing AssembleOpAttribute.", producer.GetOpMagic());
        return FAILED;
    }
    auto producerInput = producer.GetIOperands().front();
    if (producerInput == nullptr ||
        !IsRegionWithinShape(producerAttr->GetToOffset(), producerInput->GetShape(), match.input->GetShape())) {
        return SUCCESS;
    }
    auto producerInputDynShape = GetSymbolicShapeOrStatic(producerInput);
    RemapResult remap;
    if (!RemapFaninAssembleOffset(producerAttr->GetToOffset(), producerAttr->GetToDynOffset(),
                                  assembleAttr.GetToOffset(), assembleAttr.GetToDynOffset(), match.input->GetShape(),
                                  inputDynShape, match.middle->GetShape(), middleDynShape, match.output->GetShape(),
                                  outputDynShape, reshapeInputShape, reshapeDynShape, remap)) {
        return SUCCESS;
    }
    if (!IsRegionWithinShape(remap.staticOffset, producerInput->GetShape(), reshapeInputShape)) {
        return SUCCESS;
    }
    faninRecord = FaninAssembleRecord{&producer,
                                      producerInput,
                                      remap.staticOffset,
                                      remap.dynOffset,
                                      producerInputDynShape,
                                      producerAttr->GetFrom(),
                                      GetFirstSpan(producer, assembleOp),
                                      GetChainScopeInfo(producer, assembleOp)};
    canReorder = true;
    return SUCCESS;
}

bool ViewReshapeAssembleReorderUtils::BuildAxisPlanAllowFirstUnknown(const Shape& srcShape, const Shape& dstShape,
                                                                     AxisPlan& axisPlan)
{
    if (!HasOnlyFirstUnknownDim(srcShape) || !HasOnlyFirstUnknownDim(dstShape)) {
        return false;
    }
    size_t srcIdx = 0;
    size_t dstIdx = 0;
    while (srcIdx < srcShape.size() && dstIdx < dstShape.size()) {
        size_t srcBegin = srcIdx;
        size_t dstBegin = dstIdx;
        int64_t srcProduct = srcShape[srcIdx++];
        int64_t dstProduct = dstShape[dstIdx++];
        while (srcProduct != dstProduct) {
            if (srcProduct == UNKNOWN_DIM || dstProduct == UNKNOWN_DIM) {
                return false;
            }
            if (srcProduct < dstProduct) {
                if (srcIdx >= srcShape.size() || srcShape[srcIdx] == UNKNOWN_DIM ||
                    srcProduct > LLONG_MAX / srcShape[srcIdx]) {
                    return false;
                }
                srcProduct *= srcShape[srcIdx++];
                continue;
            }
            if (dstIdx >= dstShape.size() || dstShape[dstIdx] == UNKNOWN_DIM ||
                dstProduct > LLONG_MAX / dstShape[dstIdx]) {
                return false;
            }
            dstProduct *= dstShape[dstIdx++];
        }
        axisPlan.emplace_back(AxisGroup{srcBegin, srcIdx, dstBegin, dstIdx});
    }
    return srcIdx == srcShape.size() && dstIdx == dstShape.size();
}

bool ViewReshapeAssembleReorderUtils::InferInputDynRawShapeFromOutput(const LogicalTensorPtr& input,
                                                                      const LogicalTensorPtr& output,
                                                                      SymbolicShape& inferredInputDynRawShape)
{
    if (input == nullptr || output == nullptr || input->GetRawTensor() == nullptr ||
        output->GetRawTensor() == nullptr) {
        return false;
    }
    const auto& outputDynRawShape = output->GetRawTensor()->GetDynRawShape();
    if (outputDynRawShape.size() != output->GetShape().size()) {
        return false;
    }
    AxisPlan axisPlan;
    if (!BuildAxisPlanAllowFirstUnknown(input->GetShape(), output->GetShape(), axisPlan)) {
        return false;
    }
    Shape inferredStaticShape;
    if (!ApplyBackwardShape(output->GetShape(), outputDynRawShape, axisPlan, inferredStaticShape,
                            inferredInputDynRawShape)) {
        return false;
    }
    return inferredInputDynRawShape.size() == input->GetShape().size();
}

Operation& ViewReshapeAssembleReorderUtils::CreateMetadataReshape(Function& function, const LogicalTensorPtr& input,
                                                                  const LogicalTensorPtr& output,
                                                                  const SymbolicShape& dynShape, const ir::Span& span,
                                                                  const Operation::ScopeInfo& scopeInfo,
                                                                  Operation& srcOp)
{
    if (HasNegativeDynRawShape(output, dynShape)) {
        output->GetRawTensor()->UpdateDynRawShape(dynShape);
    }
    if (input != nullptr) {
        const auto& inputRaw = input->GetRawTensor();
        if (inputRaw != nullptr && HasNegativeDynDim(inputRaw->GetDynRawShape())) {
            SymbolicShape inferredInputDynRawShape;
            if (InferInputDynRawShapeFromOutput(input, output, inferredInputDynRawShape)) {
                inputRaw->UpdateDynRawShape(inferredInputDynRawShape);
            } else {
                APASS_LOG_WARN_F(
                    Elements::Operation,
                    "Failed to infer input DynRawShape from output for metadata reshape; keep original DynRawShape.");
            }
        }
    }
    auto& newReshape = irBuilder_.CreateTensorOpStmt(function, Opcode::OP_RESHAPE, {input}, {output}, span);
    newReshape.SetScopeInfo(scopeInfo);
    newReshape.CopyAttrFrom(srcOp, "");
    SetMetadataReshapeAttrs(newReshape, output, dynShape);
    return newReshape;
}

Operation& ViewReshapeAssembleReorderUtils::CreateView(Function& function, const LogicalTensorPtr& input,
                                                       const LogicalTensorPtr& output, const Offset& offset,
                                                       const SymbolicOffset& dynOffset,
                                                       const SymbolicShape& outputDynShape, MemoryType toType,
                                                       bool hasCopyInMode, const std::any& copyInModeValue,
                                                       const ir::Span& span, const Operation::ScopeInfo& scopeInfo)
{
    auto viewAttr = std::make_shared<ViewOpAttribute>(offset, toType, dynOffset, outputDynShape);
    auto& newView = irBuilder_.CreateTensorOpStmt(function, Opcode::OP_VIEW, {input}, {output}, span);
    newView.SetScopeInfo(scopeInfo);
    newView.SetOpAttribute(viewAttr);
    if (hasCopyInMode) {
        newView.SetAttr("op_attr_copy_in_mode", copyInModeValue);
    }
    output->UpdateDynValidShape(outputDynShape);
    return newView;
}

void ViewReshapeAssembleReorderUtils::CreateAssemble(Function& function, const LogicalTensorPtr& input,
                                                     const LogicalTensorPtr& output, const Offset& offset,
                                                     const SymbolicOffset& dynOffset,
                                                     const SymbolicShape& inputDynShape, MemoryType fromType,
                                                     const ir::Span& span, const Operation::ScopeInfo& scopeInfo,
                                                     Operation& srcOp)
{
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(fromType, offset, dynOffset, inputDynShape);
    auto& newAssemble = irBuilder_.CreateTensorOpStmt(function, Opcode::OP_ASSEMBLE, {input}, {output}, span);
    newAssemble.SetScopeInfo(scopeInfo);
    newAssemble.CopyAttrFrom(srcOp, "");
    newAssemble.SetOpAttribute(assembleAttr);
}

void ViewReshapeAssembleReorderUtils::AppendViewReshapeRecords(Function& function)
{
    for (const auto& record : viewReshapeRecords_) {
        auto& newReshape = CreateMetadataReshape(function, record.input, record.reshapeOutput, record.reshapeDynShape,
                                                 record.span, record.scopeInfo, *record.reshapeOp);

        auto outputDynShape = GetSymbolicShapeOrStatic(record.output);
        auto& newView = CreateView(function, record.reshapeOutput, record.output, record.viewOffset,
                                   record.viewDynOffset, outputDynShape, record.toType, record.hasCopyInMode,
                                   record.copyInModeValue, record.span, record.scopeInfo);
        MoveTokenDependencyToReplacement(function, *record.viewOp, newReshape);
        MoveTokenDependencyToReplacement(function, *record.reshapeOp, newView);
        PassTokenUtils::CleanupDeletedTokenDependency(function, {record.viewOp, record.reshapeOp});
        record.viewOp->SetAsDeleted();
        record.reshapeOp->SetAsDeleted();
    }
}

void ViewReshapeAssembleReorderUtils::AppendViewReshapeFanoutRecords(Function& function)
{
    for (const auto& record : viewReshapeFanoutRecords_) {
        CreateMetadataReshape(function, record.input, record.reshapeOutput, record.reshapeDynShape, record.span,
                              record.scopeInfo, *record.reshapeOp);
        for (const auto& fanoutView : record.fanoutViews) {
            CreateView(function, record.reshapeOutput, fanoutView.output, fanoutView.viewOffset,
                       fanoutView.viewDynOffset, fanoutView.outputDynShape, fanoutView.toType, fanoutView.hasCopyInMode,
                       fanoutView.copyInModeValue, fanoutView.span, fanoutView.scopeInfo);
            fanoutView.viewOp->SetAsDeleted();
        }
        record.viewOp->SetAsDeleted();
        record.reshapeOp->SetAsDeleted();
    }
}

void ViewReshapeAssembleReorderUtils::AppendReshapeAssembleRecords(Function& function)
{
    std::unordered_set<const ReshapeAssembleRecord*> processedRecords;
    for (const auto& record : reshapeAssembleRecords_) {
        if (processedRecords.count(&record) != 0) {
            continue;
        }

        std::vector<const ReshapeAssembleRecord*> group;
        for (const auto& candidate : reshapeAssembleRecords_) {
            if (candidate.output == record.output &&
                candidate.assembleOutput->GetShape() == record.assembleOutput->GetShape()) {
                group.emplace_back(&candidate);
            }
        }

        SymbolicShape assembleOutputDynShape;
        SymbolicShape finalOutputDynShape;
        for (const auto* groupRecord : group) {
            CreateAssemble(function, groupRecord->input, record.assembleOutput, groupRecord->assembleOffset,
                           groupRecord->assembleDynOffset, groupRecord->input->GetDynValidShape(),
                           groupRecord->fromType, groupRecord->span, groupRecord->scopeInfo, *groupRecord->assembleOp);
            MergeValidShape(groupRecord->reshapeDynShape, assembleOutputDynShape);
            MergeValidShape(groupRecord->outputDynShape, finalOutputDynShape);
            groupRecord->reshapeOp->SetAsDeleted();
            groupRecord->assembleOp->SetAsDeleted();
            processedRecords.insert(groupRecord);
        }
        if (!assembleOutputDynShape.empty()) {
            record.assembleOutput->UpdateDynValidShape(assembleOutputDynShape);
            if (HasNegativeDynRawShape(record.assembleOutput, assembleOutputDynShape)) {
                record.assembleOutput->GetRawTensor()->UpdateDynRawShape(assembleOutputDynShape);
            }
        }
        if (finalOutputDynShape.empty()) {
            finalOutputDynShape = record.outputDynShape;
        }
        CreateMetadataReshape(function, record.assembleOutput, record.output, finalOutputDynShape, record.span,
                              record.scopeInfo, *record.reshapeOp);
    }
}

void ViewReshapeAssembleReorderUtils::AppendReshapeAssembleFaninRecords(Function& function)
{
    for (const auto& record : reshapeAssembleFaninRecords_) {
        for (const auto& faninAssemble : record.faninAssembles) {
            CreateAssemble(function, faninAssemble.input, record.reshapeInput, faninAssemble.assembleOffset,
                           faninAssemble.assembleDynOffset, faninAssemble.inputDynShape, faninAssemble.fromType,
                           faninAssemble.span, faninAssemble.scopeInfo, *record.assembleOp);
            faninAssemble.assembleOp->SetAsDeleted();
        }
        SymbolicShape reshapeInputDynShape;
        for (const auto& faninAssemble : record.faninAssembles) {
            SymbolicShape candidate;
            if (BuildAssembledValidShape(faninAssemble.assembleOffset, faninAssemble.assembleDynOffset,
                                         faninAssemble.inputDynShape, record.reshapeInput->GetShape().size(),
                                         candidate)) {
                MergeValidShape(candidate, reshapeInputDynShape);
            }
        }
        if (!reshapeInputDynShape.empty()) {
            record.reshapeInput->UpdateDynValidShape(reshapeInputDynShape);
        }

        CreateMetadataReshape(function, record.reshapeInput, record.output, record.outputDynShape, record.span,
                              record.scopeInfo, *record.reshapeOp);
        record.reshapeOp->SetAsDeleted();
        record.assembleOp->SetAsDeleted();
    }
}

void ViewReshapeAssembleReorderUtils::CleanUp(Function& function)
{
    function.EraseOperations(true, false);
    function.SortOperations(SortOperationsMode::LIGHTWEIGHT);
}

bool ViewReshapeAssembleReorderUtils::BuildAxisPlan(const Shape& srcShape, const Shape& dstShape, AxisPlan& axisPlan)
{
    if (!HasConcreteShape(srcShape) || !HasConcreteShape(dstShape)) {
        return false;
    }
    return BuildAxisPlanAllowFirstUnknown(srcShape, dstShape, axisPlan);
}

bool ViewReshapeAssembleReorderUtils::ApplyForwardShape(const Shape& baseShape, const SymbolicShape& baseDynShape,
                                                        const AxisPlan& axisPlan, Shape& newShape,
                                                        SymbolicShape& newDynShape)
{
    newShape.clear();
    newDynShape.clear();
    for (const auto& group : axisPlan) {
        if (group.srcEnd > baseShape.size() || group.srcEnd > baseDynShape.size()) {
            return false;
        }
        size_t srcLen = group.srcEnd - group.srcBegin;
        size_t dstLen = group.dstEnd - group.dstBegin;
        if (srcLen == dstLen) {
            for (size_t idx = group.srcBegin; idx < group.srcEnd; ++idx) {
                newShape.emplace_back(baseShape[idx]);
                newDynShape.emplace_back(baseDynShape[idx]);
            }
            continue;
        }
        if (dstLen != 1) {
            return false;
        }
        int64_t shapeProduct = 1;
        if (!ProductShape(baseShape, group.srcBegin, group.srcEnd, shapeProduct)) {
            return false;
        }
        newShape.emplace_back(shapeProduct);
        newDynShape.emplace_back(ProductDynShape(baseDynShape, group.srcBegin, group.srcEnd));
    }
    return HasOnlyFirstUnknownDim(newShape);
}

bool ViewReshapeAssembleReorderUtils::ApplyBackwardShape(const Shape& baseShape, const SymbolicShape& baseDynShape,
                                                         const AxisPlan& axisPlan, Shape& newShape,
                                                         SymbolicShape& newDynShape)
{
    newShape.clear();
    newDynShape.clear();
    for (const auto& group : axisPlan) {
        if (group.dstEnd > baseShape.size() || group.dstEnd > baseDynShape.size()) {
            return false;
        }
        size_t srcLen = group.srcEnd - group.srcBegin;
        size_t dstLen = group.dstEnd - group.dstBegin;
        if (srcLen == dstLen) {
            for (size_t idx = group.dstBegin; idx < group.dstEnd; ++idx) {
                newShape.emplace_back(baseShape[idx]);
                newDynShape.emplace_back(baseDynShape[idx]);
            }
            continue;
        }
        if (srcLen != 1) {
            return false;
        }
        int64_t shapeProduct = 1;
        if (!ProductShape(baseShape, group.dstBegin, group.dstEnd, shapeProduct)) {
            return false;
        }
        newShape.emplace_back(shapeProduct);
        newDynShape.emplace_back(ProductDynShape(baseDynShape, group.dstBegin, group.dstEnd));
    }
    return HasOnlyFirstUnknownDim(newShape);
}

bool ViewReshapeAssembleReorderUtils::RemapOffset(const Offset& oldOffset, const SymbolicOffset& oldDynOffset,
                                                  const Shape& oldShape, const SymbolicShape& oldDynShape,
                                                  const Shape& newShape, const SymbolicShape& newDynShape,
                                                  RemapResult& result)
{
    if (oldOffset.size() != oldShape.size() || oldShape.size() != oldDynShape.size() ||
        newShape.size() != newDynShape.size()) {
        return false;
    }
    result.staticOffset = SymbolicToStaticOffset(DelinearizeOffset(
        LinearizeOffset(ShapeToSymbolic(oldOffset), ShapeToSymbolic(oldShape)), ShapeToSymbolic(newShape)));
    auto dynOffset = NormalizeDynOffset(oldOffset, oldDynOffset);
    result.dynOffset = DelinearizeOffset(LinearizeOffset(dynOffset, oldDynShape), newDynShape);
    return true;
}

bool ViewReshapeAssembleReorderUtils::RemapFanoutViewOffset(
    const Offset& baseViewOffset, const SymbolicOffset& baseViewDynOffset, const Offset& fanoutOffset,
    const SymbolicOffset& fanoutDynOffset, const Shape& compactShape, const SymbolicShape& compactDynShape,
    const Shape& middleShape, const SymbolicShape& middleDynShape, const Shape& inputShape,
    const SymbolicShape& inputDynShape, const Shape& newShape, const SymbolicShape& newDynShape, RemapResult& result)
{
    if (fanoutOffset.size() != compactShape.size() || compactShape.size() != compactDynShape.size() ||
        baseViewOffset.size() != middleShape.size() || middleShape.size() != middleDynShape.size() ||
        inputShape.size() != inputDynShape.size() || inputShape.size() != middleShape.size() ||
        newShape.size() != newDynShape.size()) {
        return false;
    }

    auto staticFanoutDynOffset = ShapeToSymbolic(fanoutOffset);
    auto staticMiddleOffset = DelinearizeOffset(LinearizeOffset(staticFanoutDynOffset, ShapeToSymbolic(compactShape)),
                                                ShapeToSymbolic(middleShape));
    auto staticBaseDynOffset = ShapeToSymbolic(baseViewOffset);
    result.staticOffset = SymbolicToStaticOffset(DelinearizeOffset(
        LinearizeOffset(AddOffsets(staticBaseDynOffset, staticMiddleOffset), ShapeToSymbolic(inputShape)),
        ShapeToSymbolic(newShape)));

    auto normalizedFanoutDynOffset = NormalizeDynOffset(fanoutOffset, fanoutDynOffset);
    auto middleOffset = DelinearizeOffset(LinearizeOffset(normalizedFanoutDynOffset, compactDynShape), middleDynShape);

    auto baseDynOffset = NormalizeDynOffset(baseViewOffset, baseViewDynOffset);
    result.dynOffset = DelinearizeOffset(LinearizeOffset(AddOffsets(baseDynOffset, middleOffset), inputDynShape),
                                         newDynShape);
    return true;
}

bool ViewReshapeAssembleReorderUtils::RemapFaninAssembleOffset(
    const Offset& inputAssembleOffset, const SymbolicOffset& inputAssembleDynOffset, const Offset& outputAssembleOffset,
    const SymbolicOffset& outputAssembleDynOffset, const Shape& compactShape, const SymbolicShape& compactDynShape,
    const Shape& middleShape, const SymbolicShape& middleDynShape, const Shape& outputShape,
    const SymbolicShape& outputDynShape, const Shape& newShape, const SymbolicShape& newDynShape, RemapResult& result)
{
    if (inputAssembleOffset.size() != compactShape.size() || compactShape.size() != compactDynShape.size() ||
        outputAssembleOffset.size() != middleShape.size() || middleShape.size() != middleDynShape.size() ||
        outputShape.size() != outputDynShape.size() || outputShape.size() != middleShape.size() ||
        newShape.size() != newDynShape.size()) {
        return false;
    }

    auto staticInputOffset = ShapeToSymbolic(inputAssembleOffset);
    auto staticMiddleOffset = DelinearizeOffset(LinearizeOffset(staticInputOffset, ShapeToSymbolic(compactShape)),
                                                ShapeToSymbolic(middleShape));
    auto staticOutputAssembleOffset = ShapeToSymbolic(outputAssembleOffset);
    result.staticOffset = SymbolicToStaticOffset(DelinearizeOffset(
        LinearizeOffset(AddOffsets(staticOutputAssembleOffset, staticMiddleOffset), ShapeToSymbolic(outputShape)),
        ShapeToSymbolic(newShape)));

    auto normalizedInputDynOffset = NormalizeDynOffset(inputAssembleOffset, inputAssembleDynOffset);
    auto middleOffset = DelinearizeOffset(LinearizeOffset(normalizedInputDynOffset, compactDynShape), middleDynShape);

    auto normalizedOutputAssembleDynOffset = NormalizeDynOffset(outputAssembleOffset, outputAssembleDynOffset);
    result.dynOffset = DelinearizeOffset(
        LinearizeOffset(AddOffsets(normalizedOutputAssembleDynOffset, middleOffset), outputDynShape), newDynShape);
    return true;
}

bool ViewReshapeAssembleReorderUtils::IsContiguousRegion(const Offset& offset, const Shape& regionShape,
                                                         const Shape& baseShape)
{
    if (offset.size() != regionShape.size() || regionShape.size() != baseShape.size()) {
        return false;
    }
    for (size_t axis = 0; axis < baseShape.size(); ++axis) {
        bool valid = true;
        for (size_t idx = 0; idx < axis; ++idx) {
            if (regionShape[idx] != 1) {
                valid = false;
                break;
            }
        }
        if (!valid) {
            continue;
        }
        for (size_t idx = axis + 1; idx < baseShape.size(); ++idx) {
            if (offset[idx] != 0 || !IsSameDim(regionShape[idx], baseShape[idx])) {
                valid = false;
                break;
            }
        }
        if (valid) {
            return true;
        }
    }
    return baseShape.empty();
}

bool ViewReshapeAssembleReorderUtils::IsLinearizedContiguousRegion(const Offset& offset, const Shape& regionShape,
                                                                   const Shape& baseShape)
{
    if (offset.size() != regionShape.size() || regionShape.size() != baseShape.size() ||
        !HasConcreteShape(regionShape) || !HasConcreteShape(baseShape)) {
        return false;
    }

    bool crossesBaseDim = false;
    int64_t regionElements = 1;
    int64_t baseElements = 1;
    int64_t linearOffset = 0;
    int64_t stride = 1;
    for (int64_t i = static_cast<int64_t>(baseShape.size()) - 1; i >= 0; --i) {
        if (offset[i] < 0 || offset[i] >= baseShape[i]) {
            return false;
        }
        crossesBaseDim = crossesBaseDim || (offset[i] + regionShape[i] > baseShape[i]);
        regionElements *= regionShape[i];
        baseElements *= baseShape[i];
        linearOffset += offset[i] * stride;
        stride *= baseShape[i];
    }
    return crossesBaseDim && linearOffset + regionElements <= baseElements;
}

bool ViewReshapeAssembleReorderUtils::AreCollapsedGroupsContiguous(const Offset& offset, const Shape& regionShape,
                                                                   const Shape& baseShape, const AxisPlan& axisPlan,
                                                                   bool useSrcGroup)
{
    if (offset.size() != regionShape.size() || regionShape.size() != baseShape.size()) {
        return false;
    }
    if (IsLinearizedContiguousRegion(offset, regionShape, baseShape)) {
        return true;
    }
    for (const auto& group : axisPlan) {
        size_t begin = useSrcGroup ? group.srcBegin : group.dstBegin;
        size_t end = useSrcGroup ? group.srcEnd : group.dstEnd;
        size_t srcLen = group.srcEnd - group.srcBegin;
        size_t dstLen = group.dstEnd - group.dstBegin;
        if (srcLen == dstLen) {
            continue;
        }
        if (begin > end || end > offset.size()) {
            return false;
        }
        Offset groupOffset(offset.begin() + begin, offset.begin() + end);
        Shape groupRegionShape(regionShape.begin() + begin, regionShape.begin() + end);
        Shape groupBaseShape(baseShape.begin() + begin, baseShape.begin() + end);
        if (!IsContiguousRegion(groupOffset, groupRegionShape, groupBaseShape)) {
            return false;
        }
    }
    return true;
}

bool ViewReshapeAssembleReorderUtils::GetSymbolicShape(const LogicalTensorPtr& tensor, SymbolicShape& dynShape)
{
    if (tensor == nullptr) {
        return false;
    }
    if (!HasOnlyFirstUnknownDim(tensor->GetShape())) {
        return false;
    }
    dynShape = tensor->GetDynValidShape();
    if (dynShape.size() == tensor->GetShape().size()) {
        return true;
    }
    if (tensor->GetRawTensor() != nullptr &&
        tensor->GetRawTensor()->GetDynRawShape().size() == tensor->GetShape().size()) {
        dynShape = tensor->GetRawTensor()->GetDynRawShape();
        return true;
    }
    if (HasConcreteShape(tensor->GetShape())) {
        dynShape = ShapeToSymbolic(tensor->GetShape());
        return true;
    }
    return false;
}

bool ViewReshapeAssembleReorderUtils::GetChainSymbolicShapes(const ChainMatch& match, SymbolicShape& inputDynShape,
                                                             SymbolicShape& middleDynShape,
                                                             SymbolicShape& outputDynShape)
{
    return GetSymbolicShape(match.input, inputDynShape) && GetSymbolicShape(match.middle, middleDynShape) &&
           GetSymbolicShape(match.output, outputDynShape);
}

SymbolicShape ViewReshapeAssembleReorderUtils::GetSymbolicShapeOrStatic(const LogicalTensorPtr& tensor)
{
    SymbolicShape dynShape;
    if (GetSymbolicShape(tensor, dynShape)) {
        return dynShape;
    }
    return tensor == nullptr ? SymbolicShape() : ShapeToSymbolic(tensor->GetShape());
}

SymbolicOffset ViewReshapeAssembleReorderUtils::NormalizeDynOffset(const Offset& offset,
                                                                   const SymbolicOffset& dynOffset)
{
    if (dynOffset.size() == offset.size()) {
        return dynOffset;
    }
    return ShapeToSymbolic(offset);
}

bool ViewReshapeAssembleReorderUtils::BuildAssembledValidShape(const Offset& offset, const SymbolicOffset& dynOffset,
                                                               const SymbolicShape& inputDynShape, size_t outputRank,
                                                               SymbolicShape& outputDynShape)
{
    if (offset.size() != outputRank || inputDynShape.size() != outputRank) {
        return false;
    }
    auto normalizedDynOffset = NormalizeDynOffset(offset, dynOffset);
    if (normalizedDynOffset.size() != outputRank) {
        return false;
    }
    outputDynShape.clear();
    outputDynShape.reserve(outputRank);
    for (size_t i = 0; i < outputRank; ++i) {
        outputDynShape.emplace_back(((inputDynShape[i] + normalizedDynOffset[i]) * (inputDynShape[i] != 0)).Simplify());
    }
    return true;
}

bool ViewReshapeAssembleReorderUtils::MergeValidShape(const SymbolicShape& candidate, SymbolicShape& merged)
{
    if (merged.empty()) {
        merged = candidate;
        return true;
    }
    if (candidate.size() != merged.size()) {
        return false;
    }
    for (size_t i = 0; i < merged.size(); ++i) {
        merged[i] = merged[i].Max(candidate[i]).Simplify();
    }
    return true;
}

ir::Span ViewReshapeAssembleReorderUtils::GetFirstSpan(Operation& first, Operation& second)
{
    if (!first.GetSpan().IsUnknown()) {
        return first.GetSpan();
    }
    return second.GetSpan();
}

Operation::ScopeInfo ViewReshapeAssembleReorderUtils::GetChainScopeInfo(Operation& first, Operation& second)
{
    if (first.GetScopeId() != -1) {
        return first.GetScopeInfo();
    }
    if (second.GetScopeId() != -1) {
        return second.GetScopeInfo();
    }
    return Operation::ScopeInfo();
}

bool ViewReshapeAssembleReorderUtils::IsScopeCompatible(Operation& first, Operation& second)
{
    return first.GetScopeId() == -1 || second.GetScopeId() == -1 || first.GetScopeId() == second.GetScopeId();
}

bool ViewReshapeAssembleReorderUtils::HasCascadedPattern(Function& function)
{
    return HasCascadedViewsBeforeReshape(function) || HasCascadedAssemblesAfterReshape(function);
}
} // namespace npu::tile_fwk
