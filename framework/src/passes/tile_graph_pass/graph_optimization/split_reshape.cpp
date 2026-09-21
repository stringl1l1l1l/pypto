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
 * \file split_reshape.cpp
 * \brief
 */

#include "split_reshape.h"

#include <climits>

#include "interface/tensor/irbuilder.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_utils/pass_attr_defs.h"
#include "passes/pass_utils/graph_utils.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "SplitReshape"

namespace npu::tile_fwk {
namespace {

void InheritTokenFromOldAssemble(Function& function, const Operation& oldAssemble, Operation& newAssemble)
{
    if (oldAssemble.result_token_.empty()) {
        return;
    }
    IRBuilder builder;
    auto newToken = builder.CreateTokenVar(newAssemble.GetSpan());
    newAssemble.result_token_.push_back(newToken);

    auto& varDependency = function.GetVarDependency();
    varDependency.AddProducer(newToken, std::static_pointer_cast<const ir::Stmt>(newAssemble.shared_from_this()));

    auto oldConsumers = oldAssemble.ConsumerOpsByToken();
    for (auto* consumer : oldConsumers) {
        consumer->tokens_.push_back(newToken);
        varDependency.AddConsumer(newToken, std::static_pointer_cast<const ir::Stmt>(consumer->shared_from_this()));
    }
}

void RemoveObsoleteTokensAfterSplit(Function& function)
{
    auto& varDependency = function.GetVarDependency();
    std::vector<ir::VarPtr> tokensToRemove;

    for (auto& [token, entry] : varDependency.GetAllDependencies()) {
        bool hasAssembleProducer = false;
        bool hasAssembleConsumer = false;
        Operation* producerAssemble = nullptr;
        Operation* consumerAssemble = nullptr;

        for (auto& stmt : entry.producers) {
            auto* op = const_cast<Operation*>(static_cast<const Operation*>(stmt.get()));
            if (op != nullptr && op->GetOpcode() == Opcode::OP_ASSEMBLE) {
                hasAssembleProducer = true;
                producerAssemble = op;
            }
        }
        for (auto& stmt : entry.consumers) {
            auto* op = const_cast<Operation*>(static_cast<const Operation*>(stmt.get()));
            if (op != nullptr && op->GetOpcode() == Opcode::OP_ASSEMBLE) {
                hasAssembleConsumer = true;
                consumerAssemble = op;
            }
        }

        if (hasAssembleProducer && hasAssembleConsumer && producerAssemble != nullptr && consumerAssemble != nullptr) {
            auto producerOutput = producerAssemble->oOperand.empty() ? nullptr : producerAssemble->oOperand.front();
            auto consumerOutput = consumerAssemble->oOperand.empty() ? nullptr : consumerAssemble->oOperand.front();
            if (producerOutput != nullptr && consumerOutput != nullptr &&
                producerOutput->tensor != consumerOutput->tensor) {
                tokensToRemove.push_back(token);
            }
        }
    }

    for (const auto& token : tokensToRemove) {
        const auto& entry = varDependency.GetAllDependencies().find(token);
        if (entry == varDependency.GetAllDependencies().end()) {
            continue;
        }
        for (auto& stmt : entry->second.consumers) {
            auto* op = const_cast<Operation*>(static_cast<const Operation*>(stmt.get()));
            if (op != nullptr) {
                op->tokens_.erase(std::remove(op->tokens_.begin(), op->tokens_.end(), token), op->tokens_.end());
            }
        }
        varDependency.RemoveVar(token);
    }
}

std::string GetStr(const std::vector<int64_t>& vec)
{
    std::string ret;
    for (const auto& val : vec) {
        ret += std::to_string(val) + ", ";
    }
    return "{" + ret + "}";
}

std::string GetStr(const ReshapeTilePara& para)
{
    std::string ret;
    ret += "shape = " + GetStr(para.shape);
    ret += ", newShape = " + GetStr(para.newShape);
    ret += ", shape's tile = " + GetStr(para.tileShape);
    ret += ", shape's offset = " + GetStr(para.tileOffset);
    return ret;
}

std::string GetStr(const std::vector<SymbolicScalar>& vec)
{
    std::string ret;
    for (const auto& val : vec) {
        ret += val.Dump() + ", ";
    }
    return "{" + ret + "}";
}

Opcode InferAssembleOpcode(const Operation* originOp)
{
    if (originOp == nullptr) {
        APASS_LOG_WARN_F(Elements::Operation, "Origin assemble-family op is nullptr, fallback to OP_ASSEMBLE.");
        return Opcode::OP_ASSEMBLE;
    }
    if (!IsAssembleLike(originOp->GetOpcode())) {
        APASS_LOG_WARN_F(Elements::Operation, "Origin op[%d] is not assemble-family, fallback to OP_ASSEMBLE.",
                         originOp->GetOpMagic());
        return Opcode::OP_ASSEMBLE;
    }
    return originOp->GetOpcode();
}
} // namespace

Status SplitReshape::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start SplitReshapeOp.");
    if (Init() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Init failed!");
        return FAILED;
    }
    if (CollectCopyOut(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "CollectCopyOut failed!");
        return FAILED;
    }
    if (CheckCopyIn(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "CheckCopyIn failed!");
        return FAILED;
    }
    if (AddOperation(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "AddOperation failed!");
        return FAILED;
    }
    EliminateOperation(function, false, false);
    RemoveObsoleteTokensAfterSplit(function);
    function.SortOperations(SortOperationsMode::LIGHTWEIGHT);
    if (SetMemoryType(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "SetMemoryType failed!");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End SplitReshapeOp.");
    return SUCCESS;
}

Status SplitReshape::Init()
{
    assembleOutToInput_.clear();
    reshapeSources_.clear();
    groupReshapeNoSplitRawMagics_.clear();
    mapOffset_.clear();
    mapAssembleOpMagic_.clear();
    reshapeDynOutput_.clear();
    assembles_.clear();
    reshapes_.clear();
    viewOffset_.clear();
    reshapeOffset_.clear();
    reshapeRawOutputs_.clear();
    reshapeRawInputs_.clear();
    reshapeOpPtrs_.clear();
    assembleOpPtrs_.clear();
    rawToAlignCache_.clear();
    sameRawInputCache_.clear();
    return SUCCESS;
}

Status SplitReshape::ObtainChangingAxis(std::vector<int64_t> alignedShape, std::vector<int64_t> input,
                                        std::vector<bool>& ChangingAxis)
{
    int64_t curVal;
    int alignIdx = 0;
    for (size_t i = 0; i < input.size(); ++i) {
        if (alignedShape[alignIdx] == input[i]) {
            alignIdx++;
            continue;
        }
        curVal = input[i];
        while (curVal > 1) {
            if (alignedShape[alignIdx] == 0) {
                APASS_LOG_ERROR_F(
                    Elements::Tensor,
                    "Incorrect alignedShape, alignedShape[%d] is 0; Please check whether the alignshape generated by "
                    "the ShapeAlign function is corrent.",
                    alignIdx);
                return FAILED;
            }
            curVal /= alignedShape[alignIdx];
            ChangingAxis[alignIdx] = true;
            alignIdx++;
        }
    }
    return SUCCESS;
}

// 用于屏蔽无法处理的动态shape场景
// 当某根reshapeop的validShape的动态shape轴涉及发生改变的轴（如被合轴，被分轴，被部分合轴）时触发屏蔽逻辑
// 目前只判断reshapeop的validshape信息，其他属性不做判断
// SUCCESS: 动态特征合法，可以执行屏蔽操作
// FAILED: 不合法的validShape，存在构图问题，需要报错
// WARNING：动态shape涉及非首轴的变轴，跳过该reshapeOp的pass
Status SplitReshape::CheckDynOutputAlignment(const std::vector<int64_t>& alignedShape,
                                             const std::vector<int64_t>& output,
                                             const std::vector<SymbolicScalar>& dynOutput,
                                             const std::vector<bool>& changingAxis)
{
    size_t alignIdx = 0;
    for (size_t i = 0; i < output.size(); ++i) {
        if (alignIdx < alignedShape.size() && alignedShape[alignIdx] == output[i]) {
            if (i != 0U && changingAxis[alignIdx] && !dynOutput[i].IsImmediate()) {
                APASS_LOG_DEBUG_F(Elements::Tensor,
                                  "Found undetermined axis from dynOutput[%zu], which is also a merged/split axis.", i);
                return WARNING;
            }
            alignIdx++;
            continue;
        }
        if (output[i] == 1) {
            continue;
        }
        if (i != 0U && !dynOutput[i].IsImmediate()) {
            APASS_LOG_DEBUG_F(Elements::Tensor,
                              "Found undetermined axis from dynOutput[%zu], which is also a merged/split axis.", i);
            return WARNING;
        }
        int64_t curVal = output[i];
        while (curVal > 1) {
            if (alignedShape[alignIdx] == 0) {
                APASS_LOG_ERROR_F(
                    Elements::Tensor,
                    "Incorrect alignedShape, alignedShape[%zu] is 0; Please check whether the alignshape generated by "
                    "the ShapeAlign function is corrent.",
                    alignIdx);
                return FAILED;
            }
            curVal /= alignedShape[alignIdx++];
        }
    }
    return SUCCESS;
}

Status SplitReshape::CheckDynStatus(std::vector<int64_t> alignedShape, std::vector<int64_t> input,
                                    std::vector<int64_t> output, std::vector<SymbolicScalar> dynOutput)
{
    if (dynOutput.empty()) {
        return SUCCESS;
    }
    if (output.size() != dynOutput.size()) {
        APASS_LOG_ERROR_F(
            Elements::Tensor,
            "The dim of output %zu does not equal to dynOutput %zu; Make sure the dim of output equal to dynOutput.",
            output.size(), dynOutput.size());
        return FAILED;
    }
    input.erase(std::remove(input.begin(), input.end(), 1), input.end());
    std::vector<bool> changingAxis(alignedShape.size());
    if (ObtainChangingAxis(alignedShape, input, changingAxis) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Tensor, "ObtainChangingAxis failed.");
        return FAILED;
    }
    return CheckDynOutputAlignment(alignedShape, output, dynOutput, changingAxis);
}

bool SplitReshape::CheckSameRawInput(const LogicalTensorPtr& reshapeSource)
{
    if (auto it = sameRawInputCache_.find(reshapeSource); it != sameRawInputCache_.end()) {
        return it->second;
    }

    const auto reshapeInputRawMagic = reshapeSource->GetRawTensor()->GetRawMagic();
    bool result = true;
    if (auto it = assembleOutToInput_.find(reshapeInputRawMagic); it != assembleOutToInput_.end()) {
        const auto& copySources = it->second;
        if (!copySources.empty()) {
            const auto firstAssInputRawMagic = (*copySources.begin())->GetRawTensor()->GetRawMagic();
            result = std::all_of(copySources.begin(), copySources.end(), [&](const auto& src) {
                return src->GetRawTensor()->GetRawMagic() == firstAssInputRawMagic;
            });
        }
    }

    sameRawInputCache_.emplace(reshapeSource, result);
    return result;
}

std::shared_ptr<ReshapeOp> SplitReshape::ReshapeOperationExist(const std::shared_ptr<ReshapeOp>& isAddReshapeop)
{
    const auto hashKey = ComputeReshapeHash(isAddReshapeop->input, isAddReshapeop->output);
    auto it = reshapes_.find(hashKey);
    if (it != reshapes_.end()) {
        return it->second;
    }
    reshapes_[hashKey] = isAddReshapeop;
    return nullptr;
}

// 动态首轴由多个输入轴合并时，局部reshape的validShape应由局部输入validShape推导。
bool SplitReshape::InferDynFirstAxisMergeShape(const std::shared_ptr<ReshapeOp>& reshapeOp,
                                               const std::vector<SymbolicScalar>& localInputDynShape,
                                               std::vector<SymbolicScalar>& localOutputDynShape) const
{
    if (reshapeOp == nullptr || reshapeOp->originOpPtr == nullptr || reshapeOp->input == nullptr ||
        reshapeOp->output == nullptr || localInputDynShape.size() != reshapeOp->input->shape.size()) {
        return false;
    }
    std::vector<SymbolicScalar> originDynShape;
    if (!reshapeOp->originOpPtr->GetAttr(OP_ATTR_VALID_SHAPE, originDynShape) || originDynShape.empty() ||
        originDynShape.front().IsImmediate()) {
        return false;
    }
    const auto& originInputs = reshapeOp->originOpPtr->GetIOperands();
    const auto& originOutputs = reshapeOp->originOpPtr->GetOOperands();
    if (originInputs.empty() || originOutputs.empty() || originInputs.front() == nullptr ||
        originOutputs.front() == nullptr || originInputs.front()->GetRawTensor() == nullptr ||
        originOutputs.front()->GetRawTensor() == nullptr) {
        return false;
    }
    const auto& originInputShape = originInputs.front()->GetRawTensor()->GetRawShape();
    const auto& originOutputShape = originOutputs.front()->GetRawTensor()->GetRawShape();
    const auto outerAxis = std::find_if(originInputShape.begin(), originInputShape.end(),
                                        [](int64_t dim) { return dim != 1; });
    if (outerAxis == originInputShape.end() || originOutputShape.empty() || originOutputShape.front() <= *outerAxis) {
        return false;
    }

    const auto& inputShape = reshapeOp->input->shape;
    const auto& outputShape = reshapeOp->output->shape;
    if (inputShape.empty() || outputShape.empty() || outputShape.front() <= 0) {
        return false;
    }
    int64_t mergedShape = 1;
    size_t mergedAxisCount = 0;
    while (mergedAxisCount < inputShape.size() && mergedShape < outputShape.front()) {
        const int64_t dim = inputShape[mergedAxisCount];
        if (dim <= 0 || mergedShape > LLONG_MAX / dim) {
            return false;
        }
        mergedShape *= dim;
        mergedAxisCount++;
    }
    if (mergedShape != outputShape.front() || inputShape.size() - mergedAxisCount != outputShape.size() - 1) {
        return false;
    }
    for (size_t i = 1; i < outputShape.size(); ++i) {
        if (inputShape[mergedAxisCount + i - 1] != outputShape[i]) {
            return false;
        }
    }

    SymbolicScalar mergedDynShape(1);
    for (size_t i = 0; i < mergedAxisCount; ++i) {
        mergedDynShape = (mergedDynShape * localInputDynShape[i]).Simplify();
    }
    localOutputDynShape = {mergedDynShape};
    localOutputDynShape.insert(localOutputDynShape.end(), localInputDynShape.begin() + mergedAxisCount,
                               localInputDynShape.end());
    return true;
}

// 更新reshapeOp输出的dynShape信息
// 新增reshape的输出validShape应由拆分前reshape输出的validShape和新增输出tensor的offset/shape推导。
// dynValidShapes中暂存的是原reshape输出坐标系下的有效终点，GetReshapeDynShape中再减去新输出tensor的offset。
// SUCCESS: 正确地更新了节点的dynValidShapes
// FAILED: validShape与输出tensor维度不同，一般情况下不会出现
Status SplitReshape::UpdateDynShape(const std::shared_ptr<ReshapeOp>& reshapeOp,
                                    const std::vector<SymbolicScalar>& localInputDynShape)
{
    std::vector<SymbolicScalar> dynShape;
    if (reshapeOp->originOpPtr == nullptr) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Skip updating dynValidShape because origin reshape op is null.");
        return SUCCESS;
    }
    if (!reshapeOp->originOpPtr->GetAttr(OP_ATTR_VALID_SHAPE, dynShape)) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Skip updating dynValidShape because reshape op[%d] has no validShape.",
                          reshapeOp->originOpPtr->GetOpMagic());
        return SUCCESS;
    }
    if (dynShape.empty()) {
        return SUCCESS;
    }
    auto output = reshapeOp->output;
    if (output == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "The output of reshape op is nullptr.");
        return FAILED;
    }
    const auto& outputOffset = output->GetOffset();
    if (outputOffset.size() != dynShape.size() || output->shape.size() != dynShape.size()) {
        APASS_LOG_ERROR_F(Elements::Tensor,
                          "The dim of output offset %s or output shape %s does not equal to dynShape %s.",
                          GetStr(outputOffset).c_str(), GetStr(output->shape).c_str(), GetStr(dynShape).c_str());
        return FAILED;
    }
    std::vector<SymbolicScalar> outputDynShape;
    if (!InferDynFirstAxisMergeShape(reshapeOp, localInputDynShape, outputDynShape)) {
        // GetViewValidShape returns the valid size in the new output coordinates; add offset back to get the original
        // end.
        outputDynShape = GetViewValidShape(dynShape, outputOffset, {}, output->shape);
    }
    std::vector<SymbolicScalar> curDynShape;
    for (size_t i = 0; i < outputDynShape.size(); i++) {
        curDynShape.emplace_back((outputDynShape[i] + outputOffset[i]) * (outputDynShape[i] != 0));
    }
    reshapeOp->dynValidShapes.emplace_back(curDynShape);
    return SUCCESS;
}

Status SplitReshape::UpdateReshapeOutputDynOffset(const LogicalTensorPtr& reshapeOutput,
                                                  const ViewOpAttribute& viewOpAttribute) const
{
    const auto& viewDynOffset = viewOpAttribute.GetFromDynOffset();
    if (viewDynOffset.empty()) {
        return SUCCESS;
    }
    if (reshapeOutput == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "The output of reshape op is nullptr.");
        return FAILED;
    }
    const auto& reshapeOffset = reshapeOutput->GetOffset();
    const auto& viewOffset = viewOpAttribute.GetFromOffset();
    if (reshapeOffset.size() != viewOffset.size() || reshapeOffset.size() != viewDynOffset.size()) {
        APASS_LOG_ERROR_F(Elements::Tensor,
                          "The dim of reshape offset %s, view offset %s or view dynamic offset %s is inconsistent.",
                          GetStr(reshapeOffset).c_str(), GetStr(viewOffset).c_str(), GetStr(viewDynOffset).c_str());
        return FAILED;
    }
    std::vector<SymbolicScalar> reshapeDynOffset;
    reshapeDynOffset.reserve(reshapeOffset.size());
    for (size_t i = 0; i < reshapeOffset.size(); ++i) {
        reshapeDynOffset.emplace_back((viewDynOffset[i] + reshapeOffset[i] - viewOffset[i]).Simplify());
    }
    reshapeOutput->UpdateOffset(TensorOffset(reshapeOffset, reshapeDynOffset));
    return SUCCESS;
}

// 实现reshape偏移值的分组
// viewOffset_是一个ReshapeOp的sharedptr至下标的映射，用来表示reshape所涉及的tile块的最小偏移值
// 该函数会寻找并更新ReshapeOp的偏移信息，确定reshape所涉及的tile的起始偏移位置
// SUCCESS: 执行成功，更新viewOffset_信息
// FAILED：执行失败，偏移值的维度不一致，通常不会出现
Status SplitReshape::GroupReshapeOffset(const std::shared_ptr<ReshapeOp>& isAddReshapeop,
                                        const std::vector<int64_t>& offset)
{
    auto iter = viewOffset_.find(isAddReshapeop);
    if (iter == viewOffset_.end()) {
        viewOffset_[isAddReshapeop] = offset;
        return SUCCESS;
    }
    auto curStartIdx = iter->second;
    std::vector<int64_t> upperleftIdx(curStartIdx.size(), 0);
    if (!offset.empty()) {
        upperleftIdx = offset;
    }
    if (curStartIdx.size() != upperleftIdx.size()) {
        APASS_LOG_ERROR_F(
            Elements::Tensor,
            "The dim of curStartIdx (%s, offset %s) does not equal to upperleftIdx; Please check the dim of the "
            "curStartIdx and upperleftIdx.",
            GetStr(curStartIdx).c_str(), GetStr(offset).c_str());
        return FAILED;
    }
    for (size_t i = 0; i < offset.size(); ++i) {
        upperleftIdx[i] = std::min(curStartIdx[i], offset[i]);
    }
    viewOffset_[isAddReshapeop] = upperleftIdx;
    return SUCCESS;
}

unsigned long SplitReshape::ComputeReshapeHash(const LogicalTensorPtr& input, const LogicalTensorPtr& output) const
{
    unsigned long operationHash = ComputeReshapeHashOrderless(input, output);
    return operationHash;
}

unsigned long SplitReshape::ComputeReshapeHashOrderless(const LogicalTensorPtr& input,
                                                        const LogicalTensorPtr& output) const
{
    std::stringstream ss;
    ss << "[i";
    ss << "$" << input->tensor->DumpSSA(false, false);
    ss << input->DumpType();
    ss << "(";
    for (size_t i = 0; i < input->offset.size(); ++i) {
        ss << input->offset[i];
        if (i != input->offset.size() - 1) {
            ss << ", ";
        }
    }
    ss << ")";
    ss << "]";
    ss << "[j";
    ss << "$" << output->tensor->DumpSSA(false, false);
    ss << output->DumpType();
    ss << "(";
    for (size_t i = 0; i < output->offset.size(); ++i) {
        ss << output->offset[i];
        if (i != output->offset.size() - 1) {
            ss << ", ";
        }
    }
    ss << ")";
    ss << "]";
    std::string s = ss.str();
    std::hash<std::string> hasher;
    auto result = hasher(s);
    return result;
}

std::vector<int64_t> SplitReshape::ObtainMapOffset(const LogicalTensorPtr& input, const LogicalTensorPtr& output) const
{
    auto it = mapOffset_.find({input->GetMagic(), output->GetMagic()});
    if (it == mapOffset_.end())
        return {};
    return it->second;
}

Status SplitReshape::CollectReshapeInfo(const Operation& op)
{
    auto input = op.GetIOperands().front();
    auto output = op.GetOOperands().front();
    if (input == nullptr || output == nullptr || output->GetRawTensor() == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Invalid reshape op [%d], at least one of input, output or raw tensor of output is nullptr; "
                          "Please check the input and output of op. %s",
                          op.opmagic, GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    std::vector<SymbolicScalar> dynOutput;
    if (op.GetAttr(OP_ATTR_VALID_SHAPE, dynOutput)) {
        reshapeDynOutput_[output->GetRawTensor()->GetRawMagic()] = dynOutput;
    }
    if (op.HasAttr(OpAttributeKey::groupReshapeNoSplit) && op.GetBoolAttribute(OpAttributeKey::groupReshapeNoSplit)) {
        groupReshapeNoSplitRawMagics_.insert(output->GetRawTensor()->GetRawMagic());
    }
    reshapeSources_[output->GetRawTensor()->GetRawMagic()] = input;
    reshapeOpPtrs_[output->GetMagic()] = &op;
    return SUCCESS;
}

Status SplitReshape::CollectAssembleInfo(const Operation& op,
                                         const std::unordered_set<int64_t>& multiLogicalTensorRawMagics)
{
    auto input = op.GetIOperands().front();
    auto output = op.GetOOperands().front();
    if (input == nullptr || output == nullptr || input->GetRawTensor() == nullptr ||
        output->GetRawTensor() == nullptr) {
        APASS_LOG_ERROR_F(
            Elements::Operation,
            "Invalid assemble op [%d], at least one of input, output, raw tensor of input, raw tensor of output "
            "is nullptr; Please check the input and output of op. %s",
            op.opmagic, GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    const int outputRawMagic = output->GetRawTensor()->GetRawMagic();
    if (multiLogicalTensorRawMagics.count(outputRawMagic) > 0) {
        APASS_LOG_DEBUG_F(Elements::Tensor,
                          "Assemble[%d] output rawMagic[%d] has multiple logicalTensors, "
                          "skip for WAW-only token support.",
                          op.GetOpMagic(), outputRawMagic);
        return SUCCESS;
    }
    assembleOutToInput_[outputRawMagic].emplace_back(input);
    auto assembleOpAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(op.GetOpAttribute());
    if (assembleOpAttr == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Invalid assemble op [%d], assemble op attribute is nullptr. %s",
                          op.opmagic, GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    auto offset = assembleOpAttr->GetToOffset();
    mapOffset_[std::make_pair(input->GetMagic(), output->GetMagic())] = offset;
    mapAssembleOpMagic_[std::make_pair(input->GetMagic(), output->GetMagic())] = op.GetOpMagic();
    assembleOpPtrs_[std::make_pair(input->GetMagic(), output->GetMagic())] = &op;
    return SUCCESS;
}

void SplitReshape::DeduplicateCopySources()
{
    for (auto& [key, copySources] : assembleOutToInput_) {
        (void)key;
        std::sort(copySources.begin(), copySources.end(), TensorPtrComparator());
        copySources.erase(std::unique(copySources.begin(), copySources.end(),
                                      [](const LogicalTensorPtr& lhs, const LogicalTensorPtr& rhs) {
                                          return lhs->GetMagic() == rhs->GetMagic();
                                      }),
                          copySources.end());
    }
}

Status SplitReshape::CollectCopyOut(Function& function)
{
    const auto multiLogicalTensorRawMagics = GraphUtils::GetRawMagicsWithMultipleLogicalTensors(function);
    for (const auto& op : function.Operations(false)) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE && CollectReshapeInfo(op) != SUCCESS) {
            return FAILED;
        }
        if (IsAssembleLike(op.GetOpcode()) && CollectAssembleInfo(op, multiLogicalTensorRawMagics) != SUCCESS) {
            return FAILED;
        }
    }
    DeduplicateCopySources();
    return SUCCESS;
}

// shape1和shape2分别为reshape前后的shape
// 移除shape1和shape2中为1的元素
// 进行的操作为细分对齐：
// example1: [2,4] + [8] -> [2,4]
// example2: [2,4] + [4,2] -> [2,2,2]
// example3: [2,3] + [5] -> FAILED
// 返回对齐后的shape
Status SplitReshape::ShapeAlign(std::vector<int64_t> shape1, std::vector<int64_t> shape2,
                                std::vector<int64_t>& alignedShape)
{
    size_t i1 = 0UL;
    size_t i2 = 0UL;
    int64_t prod1 = 1;
    int64_t prod2 = 1;
    shape1.erase(std::remove(shape1.begin(), shape1.end(), 1), shape1.end());
    shape2.erase(std::remove(shape2.begin(), shape2.end(), 1), shape2.end());
    while (i1 < shape1.size() || i2 < shape2.size()) {
        if (prod1 != 1) {
            if (prod1 % shape2[i2] != 0 && shape2[i2] % prod1 != 0) {
                APASS_LOG_WARN_F(Elements::Tensor,
                                 "Non-segmentable axis, cannot apply splitting for current tiling case.");
                return WARNING;
            }
            if (shape2[i2] > prod1) {
                alignedShape.push_back(prod1);
                prod2 = shape2[i2] / prod1;
                prod1 = 1;
            } else {
                alignedShape.push_back(shape2[i2]);
                prod1 = prod1 / shape2[i2];
            }
            i2++;
            continue;
        }
        if (prod2 != 1) {
            std::swap(i1, i2);
            std::swap(prod1, prod2);
            std::swap(shape1, shape2);
            continue;
        }
        if ((i1 >= shape1.size() || i2 >= shape2.size())) {
            APASS_LOG_ERROR_F(
                Elements::Tensor,
                "Shape1 index i1(%zu) exceeds shape1 size(%zu) or shape2 index i2(%zu) exceeds shape2 size(%zu); "
                "Please check the process of compute alignShape.",
                i1, shape1.size(), i2, shape2.size());
            return FAILED;
        }
        if (shape2[i2] % shape1[i1] != 0 && shape1[i1] % shape2[i2] != 0) {
            APASS_LOG_WARN_F(Elements::Tensor,
                             "Non-segmentable axis, cannot calculate the align of the input[%s] and the output[%s].",
                             GetStr(shape1).c_str(), GetStr(shape2).c_str());
            return WARNING;
        }
        if (shape2[i2] > shape1[i1]) {
            alignedShape.push_back(shape1[i1]);
            prod2 = shape2[i2] / shape1[i1];
        } else {
            alignedShape.push_back(shape2[i2]);
            prod1 = shape1[i1] / shape2[i2];
        }
        i1++;
        i2++;
    }
    return SUCCESS;
}

Status SplitReshape::UpdateShapeOffset(UpdatePara& para, bool& flag, int& currentShape, int& currentOffset)
{
    auto stride = para.stride;
    if (flag && currentShape <= stride) {
        para.OffsetVal = currentOffset / stride;
        para.ShapeVal = 1;
        currentOffset = currentOffset - para.OffsetVal * stride;
        if (currentOffset + currentShape > stride) {
            APASS_LOG_WARN_F(
                Elements::Tensor, "Found tail cond, currentOffset is %ld, currentShape is %ld, stride is %ld.",
                static_cast<long>(currentOffset), static_cast<long>(currentShape), static_cast<long>(stride));
            return WARNING;
        }
        return SUCCESS;
    }
    if (currentOffset % stride != 0) {
        APASS_LOG_WARN_F(Elements::Tensor, "Found tail cond, currentOffset is %ld, stride is %ld.",
                         static_cast<long>(currentOffset), static_cast<long>(stride));
        return WARNING;
    }
    if (currentShape % stride != 0) {
        APASS_LOG_WARN_F(Elements::Tensor, "Found tail cond, currentShape is %ld, stride is %ld.",
                         static_cast<long>(currentShape), static_cast<long>(stride));
        return WARNING;
    }
    para.OffsetVal = currentOffset / stride;
    para.ShapeVal = currentShape / stride;
    currentOffset = 0;
    currentShape = stride;
    flag = false;
    return SUCCESS;
}

Status SplitReshape::ConstructShapeOffset(const ReshapeTilePara& shapePara, size_t& i, size_t j,
                                          std::vector<int64_t>& newOffset, std::vector<int64_t>& newShape)
{
    UpdatePara updatePara;
    auto shape = shapePara.shape;
    auto alignedShape = shapePara.newShape;
    auto tileOffset = shapePara.tileOffset;
    auto tileShape = shapePara.tileShape;
    int stride = shape[j];
    int currentOffset = tileOffset[j];
    int currentShape = tileShape[j];
    bool flag = true; // flag为true代表之前拆分的轴均为1，目前的tile依然占据维度的连续一段，可以不完整。
    while (stride > 1) {
        if (stride % alignedShape[i] != 0) {
            APASS_LOG_WARN_F(Elements::Tensor, "Cannot cut shape, stride is %ld, alignedShape[%zu] is %ld.",
                             static_cast<long>(stride), i, static_cast<long>(alignedShape[i]));
            return WARNING;
        }
        stride /= alignedShape[i];
        updatePara.stride = stride;
        if (UpdateShapeOffset(updatePara, flag, currentShape, currentOffset) == WARNING) {
            APASS_LOG_WARN_F(Elements::Tensor,
                             "Found tail condition, cannot align the shape and the offset for current tiling case.");
            return WARNING;
        }
        newOffset[i] = updatePara.OffsetVal;
        newShape[i] = updatePara.ShapeVal;
        i++;
    }
    return SUCCESS;
}

// 将原始shape的分片信息转换到alignedShape维度上
// 若原始维度shape的某轴为1：需保证对应分片偏移tileOffset为0且分片大小tileShape为1，直接跳过该维度
// 若原始维度shape的某轴非1：
// 用stride记录当前维度的"剩余未拆分长度"，currentOffset和currentShape记录当前维度的分片偏移和大小
// 用flag标记 "当前分片是否仍为连续段"（初始为true，表示可拆分）
//     根据flag和currentShape与stride的大小关系，计算newOffset[i]和newShape[i]，并更新currentOffset、currentShape和flag
// shape = [8], alignedShape = [2, 2, 2], tileOffset = [2], tileShape = [2]
// offset: [0, 1, 0], shape = [1, 1, 2]
// shape = [64, 64], alignedShape = [32, 2, 64], tileOffset = [32, 32], tileShape = [32, 32]
// offset: [16, 0, 32], shape = [16, 2, 32]
Status SplitReshape::RawToAlign(const ReshapeTilePara& shapePara, std::vector<int64_t>& newOffset,
                                std::vector<int64_t>& newShape)
{
    auto shape = shapePara.shape;
    auto alignedShape = shapePara.newShape;
    auto tileOffset = shapePara.tileOffset;
    auto tileShape = shapePara.tileShape;
    newOffset.assign(alignedShape.size(), 0);
    newShape.assign(alignedShape.size(), 0);
    size_t i = 0UL;
    for (size_t j = 0UL; j < shape.size(); j++) {
        if (shape[j] == 1) {
            if (tileOffset[j] != 0 || tileShape[j] != 1) {
                APASS_LOG_ERROR_F(
                    Elements::Tensor,
                    "Shape[%zu] is 1, but tileOffset[%zu] is %ld (should be 0) and tileShape[%zu] is %ld (should be "
                    "1); The tileshape is incorrect.",
                    j, j, static_cast<long>(tileOffset[j]), j, static_cast<long>(tileShape[j]));
                return FAILED;
            }
            continue;
        }
        if (ConstructShapeOffset(shapePara, i, j, newOffset, newShape) != SUCCESS) {
            APASS_LOG_WARN_F(Elements::Tensor,
                             "ConstructShapeOffset failed, cannot apply splitting for current tiling case.");
            return WARNING;
        }
    }
    if (i != alignedShape.size()) {
        APASS_LOG_WARN_F(
            Elements::Tensor,
            "Shape index calculated by construct shape offset (%zu) is different from aligned shape size (%zu), cannot "
            "apply splitting for current tiling case.",
            i, alignedShape.size());
        return WARNING;
    }
    return SUCCESS;
}

// 用于将rawShape的分片信息映射到新的目标形状newRawshape上(本质上是加细的还原)
// 生成对应的newOffset和新形状newShape
// 核心逻辑将原始分片信息 “重组” 到新维度上，特定条件下会清空结果并返回成功。
// example1 : rawShape = [2, 3], newRawshape = [6], tileOffset = [1, 0], tileShape = [1, 3]
//          newOffset = [3], newShape = [3]
// example2 : rawShape = [1], newRawshape = [1], tileOffset = [0], tileShape = [2]
//          newOffset = [0], newShape = [1]
// example3 : rawShape = [32, 2, 64], newRawshape = [64, 64], tileOffset = [16, 0, 32], tileShape = [16, 2, 32]
//          newOffset = [32, 32], newShape = [32, 32]
Status SplitReshape::AlignToRaw(const ReshapeTilePara& shapePara, std::vector<int64_t>& newOffset,
                                std::vector<int64_t>& newShape)
{
    auto rawShape = shapePara.shape;
    auto newRawshape = shapePara.newShape;
    auto tileOffset = shapePara.tileOffset;
    auto tileShape = shapePara.tileShape;
    newOffset.assign(newRawshape.size(), 0);
    newShape.assign(newRawshape.size(), 0);
    size_t i = 0UL;
    for (size_t j = 0UL; j < newRawshape.size(); j++) {
        if (newRawshape[j] == 1) {
            newOffset[j] = 0;
            newShape[j] = 1;
            continue;
        }
        int stride = newRawshape[j];
        bool flag = true; // flag为true表示尚未遇到不为1的tile_shape
        while (stride > 1) {
            if (rawShape[i] == 0) {
                APASS_LOG_ERROR_F(Elements::Tensor,
                                  "Incorrect rawShape, rawShape[%zu] is 0; Please set the correct rawShape.", i);
                return FAILED;
            }
            if (stride % rawShape[i] != 0) {
                APASS_LOG_ERROR_F(Elements::Tensor,
                                  "Incorrect alignment, rawShape[i] %ld should be divisible by %ld stride; Please set "
                                  "the correct alignment.",
                                  static_cast<long>(rawShape[i]), static_cast<long>(stride));
                return FAILED;
            }
            stride /= rawShape[i];
            if (flag && tileShape[i] != 1) {
                newOffset[j] += tileOffset[i] * stride;
                newShape[j] = tileShape[i] * stride;
                flag = false;
            } else if (flag && tileShape[i] == 1) {
                newOffset[j] += tileOffset[i] * stride;
            } else if ((!flag) && (!(tileOffset[i] == 0 && tileShape[i] == rawShape[i]))) {
                newOffset.clear();
                newShape.clear();
                return SUCCESS;
            }
            i++;
        }
        if (flag) {
            newShape[j] = 1;
        }
    }
    return SUCCESS;
}

Status SplitReshape::ObtainCopyOutTile(Function& function, const copyOutTilePara& copyOutTile, LogicalTensors& overlaps,
                                       LogicalTensors& newOverlaps, OverlapStatus& overlapStatus)
{
    (void)function;
    overlapStatus = OverlapStatus::NO_OVER_LAP;
    ReshapeTilePara CopyOutInfo;
    CopyOutInfo.shape = copyOutTile.reshapeSource->GetRawTensor()->GetRawShape();
    CopyOutInfo.newShape = copyOutTile.alignedShape;
    int64_t coveredSize = 0;
    for (const auto& copyOutSource : assembleOutToInput_[copyOutTile.reshapeSource->GetRawTensor()->GetRawMagic()]) {
        AlignResult result;
        int assembleOpMagic = mapAssembleOpMagic_[std::make_pair(copyOutSource->GetMagic(),
                                                                 copyOutTile.reshapeSource->GetMagic())];
        int reshapeOpMagic = copyOutTile.reshapeOpMagic;
        auto it = rawToAlignCache_.find(std::make_pair(assembleOpMagic, reshapeOpMagic));
        if (it != rawToAlignCache_.end()) {
            result = it->second;
        } else {
            // 存在多个tensor assemble成一个tensor再reshape的场景，需要使用assemble op的offset计算
            std::vector<int64_t> copyOutOffset = ObtainMapOffset(copyOutSource, copyOutTile.reshapeSource);
            CopyOutInfo.tileOffset = std::move(copyOutOffset);
            CopyOutInfo.tileShape = copyOutSource->shape;
            std::vector<int64_t> newCopyOutTileShape;
            std::vector<int64_t> newCopyOutTileOffset;
            result.st = RawToAlign(CopyOutInfo, newCopyOutTileOffset, newCopyOutTileShape);
            if (result.st == SUCCESS) {
                result.newCopyOutSource = irBuilder_.CreateTensorVar(copyOutTile.reshapeSource->GetRawTensor(),
                                                                     newCopyOutTileOffset, newCopyOutTileShape,
                                                                     std::vector<SymbolicScalar>{});
            }
            rawToAlignCache_[std::make_pair(assembleOpMagic, reshapeOpMagic)] = result;
        }
        if (result.st == WARNING) {
            APASS_LOG_WARN_F(Elements::Tensor, "Found RawToAlign warning case, copyOutInfo is %s.",
                             GetStr(CopyOutInfo).c_str());
            return WARNING;
        }
        if (result.st == FAILED) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Run RawToAlign failed, copyOutInfo is %s.",
                              GetStr(CopyOutInfo).c_str());
            return FAILED;
        }
        auto status = CalcOverlap(copyOutTile.newInputView, result.newCopyOutSource, true);
        if (status == OverlapStatus::PERFECTLY_MATCH || status == OverlapStatus::BE_COVERED) {
            overlaps.emplace_back(copyOutSource);
            newOverlaps.emplace_back(result.newCopyOutSource);
            overlapStatus = status;
            break;
        }
        if (status == OverlapStatus::COVERED) {
            overlaps.emplace_back(copyOutSource);
            newOverlaps.emplace_back(result.newCopyOutSource);
            coveredSize += CommonUtils::Numel(result.newCopyOutSource->shape);
            overlapStatus = OverlapStatus::COVERED;
        }
    }
    if (overlapStatus == OverlapStatus::COVERED && newOverlaps.size() > 1) {
        const int64_t inputViewSize = CommonUtils::Numel(copyOutTile.newInputView->shape);
        if (inputViewSize == coveredSize) {
            overlapStatus = OverlapStatus::PERFECTLY_MATCH_WITH_ALL;
        } else if (inputViewSize > coveredSize) {
            overlapStatus = OverlapStatus::COVERED_ALL;
        }
    }
    return SUCCESS;
}

Status SplitReshape::ObtainReshapeSource(Function& function, const OpPara& para, LogicalTensorPtr& newReshapeSource)
{
    (void)function;
    auto overlap = para.oldInput;
    auto reshapeSource = para.oldOutput;
    std::vector<int64_t> assembleOffset = ObtainMapOffset(overlap, reshapeSource);
    int reshapeSourceRawMagic = reshapeSource->GetRawTensor()->GetRawMagic();
    if (reshapeRawInputs_.find(reshapeSourceRawMagic) == reshapeRawInputs_.end()) {
        auto reshapeRawInput = irBuilder_.CreateRawTensor(
            overlap->Datatype(), reshapeSource->GetRawTensor()->GetRawShape(), overlap->Format());
        reshapeRawInputs_[reshapeSourceRawMagic] = reshapeRawInput;
    }
    newReshapeSource = irBuilder_.CreateTensorVar(reshapeRawInputs_[reshapeSourceRawMagic], assembleOffset,
                                                  overlap->shape, std::vector<SymbolicScalar>{});
    if (newReshapeSource == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Failed to make a shared ptr for newReshapeSource.");
        return FAILED;
    }
    newReshapeSource->SetMemoryTypeBoth(reshapeSource->GetMemoryTypeOriginal());
    return SUCCESS;
}

Status SplitReshape::ProcessPerfectlyMatch(Function& function, Operation& op, const PerfectlyMatchPara& para)
{
    auto overlap = para.overlap;
    auto input = para.input;
    auto reshapeSource = para.reshapeSource;
    auto reshapeOutput = para.reshapeOutput;
    LogicalTensorPtr newReshapeSource;
    OpPara opPara = {overlap, reshapeSource, nullptr, nullptr, {}};
    std::vector<int64_t> assembleOffset = ObtainMapOffset(overlap, reshapeSource);
    if (ObtainReshapeSource(function, opPara, newReshapeSource) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "ObtainReshapeSource failed. %s", GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(op.GetOpAttribute().get());
    auto isAddReshapeOp = std::make_shared<ReshapeOp>(newReshapeSource, reshapeOutput,
                                                      reshapeOpPtrs_[op.GetIOperands().front()->GetMagic()]);
    if (viewOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Found null view attr for op[%d]. %s", op.opmagic,
                          GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    const auto viewDynOffset = viewOpAttribute->GetFromDynOffset();
    if (UpdateReshapeOutputDynOffset(reshapeOutput, *viewOpAttribute) != SUCCESS) {
        return FAILED;
    }
    auto existOp = ReshapeOperationExist(isAddReshapeOp);
    if (existOp != nullptr) {
        op.ReplaceInput(existOp->output, input);
        viewOpAttribute->SetFromOffset(existOp->output->offset, viewDynOffset);
        GraphUtils::UpdateViewAttr(function, op);
        if (UpdateDynShape(existOp, overlap->GetDynValidShape()) != SUCCESS ||
            GroupReshapeOffset(existOp, existOp->output->offset) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "UpdateDynShape or GroupReshapeOffset failed. %s",
                              GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        return SUCCESS;
    }
    if (AddAssembleOp(overlap->GetMemoryTypeOriginal(), assembleOffset, overlap, newReshapeSource,
                      assembleOpPtrs_[std::make_pair(overlap->GetMagic(), reshapeSource->GetMagic())]) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "AddAssembleOp failed. %s", GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    op.ReplaceInput(reshapeOutput, input);
    viewOpAttribute->SetFromOffset(reshapeOutput->offset, viewDynOffset);
    GraphUtils::UpdateViewAttr(function, op);
    if (UpdateDynShape(isAddReshapeOp, overlap->GetDynValidShape()) != SUCCESS ||
        GroupReshapeOffset(isAddReshapeOp, reshapeOutput->offset) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateDynShape or GroupReshapeOffset failed. %s",
                          GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status SplitReshape::ProcessOnetoOne(Function& function, Operation& op, const CalcOverlapPara& para)
{
    std::vector<int64_t> alignedShape = para.alignedShape;
    LogicalTensorPtr reshapeSource = para.reshapeSource;
    LogicalTensorPtr input = para.input;
    LogicalTensorPtr output = para.output;
    LogicalTensorPtr inputView = para.inputView;
    auto overlap = para.overlaps.front();
    auto newoverlap = para.newOverlaps.front();
    ReshapeTilePara reshapeInfo = {alignedShape, inputView->GetRawTensor()->GetRawShape(), newoverlap->offset,
                                   newoverlap->shape};
    std::vector<int64_t> reshapeTileShape;
    std::vector<int64_t> reshapeTileOffset;
    if (AlignToRaw(reshapeInfo, reshapeTileOffset, reshapeTileShape) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Tensor, "AlignToRaw failed, reshape info is %s.", GetStr(reshapeInfo).c_str());
        return FAILED;
    }
    if (reshapeTileShape != output->shape) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Shape of output (%s) is different from reshapeTileShape (%s).",
                          GetStr(output->shape).c_str(), GetStr(reshapeTileShape).c_str());
        return FAILED;
    }
    if (reshapeRawOutputs_.find(overlap->GetRawTensor()->GetRawMagic()) == reshapeRawOutputs_.end()) {
        auto reshaperawOutput = std::make_shared<RawTensor>(input->Datatype(), inputView->GetRawTensor()->GetRawShape(),
                                                            inputView->Format());
        if (reshaperawOutput == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Failed to make a shared ptr for reshaperawOutput.");
            return FAILED;
        }
        reshapeRawOutputs_[overlap->GetRawTensor()->GetRawMagic()] = reshaperawOutput;
    }
    auto reshapeOutput = irBuilder_.CreateTensorVar(reshapeRawOutputs_[overlap->GetRawTensor()->GetRawMagic()],
                                                    reshapeTileOffset, reshapeTileShape, std::vector<SymbolicScalar>{});
    reshapeOutput->SetMemoryTypeBoth(input->GetMemoryTypeOriginal());
    PerfectlyMatchPara perfectlyMatchPara = {input,         output,        overlap,
                                             reshapeSource, reshapeOutput, para.oriViewDynShape};
    if (ProcessPerfectlyMatch(function, op, perfectlyMatchPara) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "ProcessPerfectlyMatch failed. %s", GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status SplitReshape::ProcessBeCovered(Function& function, Operation& op, const BeCoveredPara& para)
{
    auto input = para.input;
    auto overlap = para.overlap;
    auto reshapeOutput = para.reshapeOutput;
    auto reshapeSource = para.reshapeSource;
    auto newOffset = para.newOffset;
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(op.GetOpAttribute().get());
    if (viewOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Failed to obtain a shared ptr for viewOpAttribute. %s",
                          GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    LogicalTensorPtr newReshapeSource;
    OpPara checkPara = {overlap, reshapeSource, nullptr, nullptr, {}};
    if (ObtainReshapeSource(function, checkPara, newReshapeSource) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "ObtainReshapeSource failed. %s", GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    auto isAddReshapeOp = std::make_shared<ReshapeOp>(newReshapeSource, reshapeOutput,
                                                      reshapeOpPtrs_[op.GetIOperands().front()->GetMagic()]);
    const auto viewDynOffset = viewOpAttribute->GetFromDynOffset();
    if (UpdateReshapeOutputDynOffset(reshapeOutput, *viewOpAttribute) != SUCCESS) {
        return FAILED;
    }
    auto existOp = ReshapeOperationExist(isAddReshapeOp);
    viewOpAttribute->SetFromOffset(newOffset, viewDynOffset);
    GraphUtils::UpdateViewAttr(function, op);
    if (existOp != nullptr) {
        op.ReplaceInput(existOp->output, input);
        if (UpdateDynShape(existOp, overlap->GetDynValidShape()) != SUCCESS ||
            GroupReshapeOffset(existOp, newOffset) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "UpdateDynShape or GroupReshapeOffset failed. %s",
                              GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        return SUCCESS;
    }
    if (AddAssembleOp(overlap->GetMemoryTypeOriginal(), newReshapeSource->offset, overlap, newReshapeSource,
                      assembleOpPtrs_[std::make_pair(overlap->GetMagic(), reshapeSource->GetMagic())]) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "AddAssembleOp failed. %s", GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    op.ReplaceInput(reshapeOutput, input);
    if (UpdateDynShape(isAddReshapeOp, overlap->GetDynValidShape()) != SUCCESS ||
        GroupReshapeOffset(isAddReshapeOp, newOffset) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateDynShape or GroupReshapeOffset failed. %s",
                          GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status SplitReshape::CalcTileInfo(const CalcOverlapPara& para, std::vector<int64_t>& newShape,
                                  std::vector<int64_t>& newOffset, std::vector<int64_t>& reshapeTileShape,
                                  std::vector<int64_t>& reshapeTileOffset)
{
    auto alignedShape = para.alignedShape;
    auto inputView = para.inputView;
    auto output = para.output;
    auto newoverlap = para.newOverlaps.front();
    ReshapeTilePara newInfo = {alignedShape, inputView->GetRawTensor()->GetRawShape(), para.newInputViewTileOffset,
                               para.newInputViewTileShape};
    if (AlignToRaw(newInfo, newOffset, newShape) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Failed to compute the tile of the raw input, newInfo is %s.",
                          GetStr(newInfo).c_str());
        return FAILED;
    }
    if (newShape != output->shape) {
        APASS_LOG_ERROR_F(Elements::Tensor, "The new input shape of view[%s] is not equal to output[%s].",
                          GetStr(newShape).c_str(), GetStr(output->shape).c_str());
        return FAILED;
    }
    ReshapeTilePara reshapeTileInfo = {alignedShape, inputView->GetRawTensor()->GetRawShape(), newoverlap->offset,
                                       newoverlap->shape};
    if (AlignToRaw(reshapeTileInfo, reshapeTileOffset, reshapeTileShape) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Tensor,
                          "Failed to compute the tile of the raw input of inputView, reshapeTileInfo is %s.",
                          GetStr(reshapeTileInfo).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status SplitReshape::ProcessOnetoMulti(Function& function, Operation& op, const CalcOverlapPara& para)
{
    auto input = para.input;
    auto output = para.output;
    auto inputView = para.inputView;
    auto overlap = para.overlaps.front();
    auto newoverlap = para.newOverlaps.front();
    std::vector<int64_t> newShape;
    std::vector<int64_t> newOffset;
    std::vector<int64_t> reshapeTileShape;
    std::vector<int64_t> reshapeTileOffset;
    if (CalcTileInfo(para, newShape, newOffset, reshapeTileShape, reshapeTileOffset)) {
        APASS_LOG_ERROR_F(Elements::Operation, "Process CalcTileInfo failed for view op[%d]. %s", op.GetOpMagic(),
                          GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    if (reshapeTileOffset.size() == 0 && reshapeTileShape.size() == 0) {
        APASS_LOG_WARN_F(Elements::Tensor, "One new assemble overlap tile block cannot cover the input of view op[%d].",
                         op.GetOpMagic());
        return SUCCESS; // 这种情况不会对reshape做切分, 动态shape的处理也跳过
    }
    if (reshapeRawOutputs_.find(overlap->GetRawTensor()->GetRawMagic()) == reshapeRawOutputs_.end()) {
        auto reshaperawOutput = std::make_shared<RawTensor>(input->Datatype(), inputView->GetRawTensor()->GetRawShape(),
                                                            inputView->Format());
        if (reshaperawOutput == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Failed to make a rawtensor ptr for reshapeRawOutput.");
            return FAILED;
        }
        reshapeRawOutputs_[overlap->GetRawTensor()->GetRawMagic()] = reshaperawOutput;
    }
    auto reshapeOutput = irBuilder_.CreateTensorVar(reshapeRawOutputs_[overlap->GetRawTensor()->GetRawMagic()],
                                                    reshapeTileOffset, reshapeTileShape, std::vector<SymbolicScalar>{});
    reshapeOutput->SetMemoryTypeBoth(input->GetMemoryTypeOriginal());
    BeCoveredPara beCoveredPara = {overlap, input, reshapeOutput, para.reshapeSource, newOffset, para.oriViewDynShape};
    if (ProcessBeCovered(function, op, beCoveredPara) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Process ProcessBeCovered failed. %s", GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status SplitReshape::ProcessPerfectlyMatchWithAll(Function& function, Operation& op,
                                                  const PerfectlyMatchWithAllPara& para)
{
    auto input = para.input;
    auto reshapeOutput = para.reshapeOutput;
    auto newReshapeSource = para.newReshapeSource;
    reshapeOutput->SetMemoryTypeBoth(input->GetMemoryTypeOriginal(), true);
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(op.GetOpAttribute().get());
    auto isAddReshapeOp = std::make_shared<ReshapeOp>(newReshapeSource, reshapeOutput,
                                                      reshapeOpPtrs_[op.GetIOperands().front()->GetMagic()]);
    if (viewOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Failed to obtain viewOpAttribute; Please make sure op has attribute. %s",
                          GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    const auto viewDynOffset = viewOpAttribute->GetFromDynOffset();
    if (UpdateReshapeOutputDynOffset(reshapeOutput, *viewOpAttribute) != SUCCESS) {
        return FAILED;
    }
    auto existOp = ReshapeOperationExist(isAddReshapeOp);
    if (existOp != nullptr) {
        if (UpdateDynShape(existOp, para.localInputDynShape) != SUCCESS ||
            GroupReshapeOffset(existOp, existOp->output->offset) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Failed to UpdateDynShape or GroupReshapeOffset for existOp. %s",
                              GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        op.ReplaceInput(existOp->output, input);
        viewOpAttribute->SetFromOffset(existOp->output->offset, viewDynOffset);
        GraphUtils::UpdateViewAttr(function, op);
        return SUCCESS;
    }
    op.ReplaceInput(reshapeOutput, input);
    viewOpAttribute->SetFromOffset(reshapeOutput->offset, viewDynOffset);
    GraphUtils::UpdateViewAttr(function, op);
    if (UpdateDynShape(isAddReshapeOp, para.localInputDynShape) != SUCCESS ||
        GroupReshapeOffset(isAddReshapeOp, reshapeOutput->offset) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Failed to UpdateDynShape or GroupReshapeOffset for isAddReshapeOp. %s",
                          GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status SplitReshape::UpdateForPerfectlyMatchWithAll(Function& function, Operation& op, const CalcOverlapPara& para,
                                                    const ReshapeSourcePara& sourcePara)
{
    LogicalTensors overlaps = para.overlaps;
    LogicalTensorPtr reshapeSource = para.reshapeSource;
    LogicalTensorPtr input = para.input;
    LogicalTensorPtr output = para.output;
    LogicalTensorPtr inputView = para.inputView;
    auto newReshapeSourceTileShape = sourcePara.newReshapeSourceTileShape;
    auto newReshapeSourceTileOffset = sourcePara.newReshapeSourceTileOffset;
    int reshapeSourceRawMagic = reshapeSource->GetRawTensor()->GetRawMagic();
    if (reshapeRawInputs_.find(reshapeSourceRawMagic) == reshapeRawInputs_.end()) {
        auto reshapeRawInput = irBuilder_.CreateRawTensor(
            overlaps.front()->Datatype(), reshapeSource->GetRawTensor()->GetRawShape(), overlaps.front()->Format());
        reshapeRawInputs_[reshapeSourceRawMagic] = reshapeRawInput;
    }
    auto newReshapeSource = irBuilder_.CreateTensorVar(reshapeRawInputs_[reshapeSourceRawMagic],
                                                       newReshapeSourceTileOffset, newReshapeSourceTileShape,
                                                       std::vector<SymbolicScalar>{});
    if (newReshapeSource == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Failed to make a newReshapeSource ptr.");
        return FAILED;
    }
    newReshapeSource->SetMemoryTypeBoth(reshapeSource->GetMemoryTypeOriginal());
    if (reshapeRawOutputs_.find(reshapeSourceRawMagic) == reshapeRawOutputs_.end()) {
        auto reshaperawOutput = irBuilder_.CreateRawTensor(input->Datatype(), inputView->GetRawTensor()->GetRawShape(),
                                                           inputView->Format());
        reshapeRawOutputs_[reshapeSourceRawMagic] = reshaperawOutput;
    }
    auto reshapeOutput = irBuilder_.CreateTensorVar(reshapeRawOutputs_[reshapeSourceRawMagic], inputView->offset,
                                                    inputView->shape, std::vector<SymbolicScalar>{});
    if (reshapeOutput == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Failed to make a reshapeOutput ptr.");
        return FAILED;
    }
    reshapeOutput->UpdateOffset(inputView->GetTensorOffset());
    reshapeOutput->SetMemoryTypeBoth(output->GetMemoryTypeOriginal());
    for (const auto& overlap : overlaps) {
        std::vector<int64_t> overlapOffset = ObtainMapOffset(overlap, reshapeSource);
        if (AddAssembleOp(overlap->GetMemoryTypeOriginal(), overlapOffset, overlap, newReshapeSource,
                          assembleOpPtrs_[std::make_pair(overlap->GetMagic(), reshapeSource->GetMagic())]) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "AddAssembleOp failed. %s", GetFormatBacktrace(op).c_str());
            return FAILED;
        }
    }
    std::vector<SymbolicScalar> localInputDynShape;
    for (const auto& overlap : overlaps) {
        const auto overlapOffset = ObtainMapOffset(overlap, reshapeSource);
        if (GetAssembleDynShape(overlap, newReshapeSource, overlapOffset, localInputDynShape) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "GetAssembleDynShape failed. %s", GetFormatBacktrace(op).c_str());
            return FAILED;
        }
    }
    PerfectlyMatchWithAllPara perfectlyMatchwithAllPara = {
        input, output, overlaps.front(), reshapeOutput, newReshapeSource, para.oriViewDynShape, localInputDynShape};
    if (ProcessPerfectlyMatchWithAll(function, op, perfectlyMatchwithAllPara) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Process ProcessPerfectlyMatchWithAll failed. %s",
                          GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status SplitReshape::AddReshapeRawInputs(const int overlapRawMagic, const LogicalTensorPtr overlap)
{
    if (reshapeRawInputs_.find(overlapRawMagic) == reshapeRawInputs_.end()) {
        auto reshapeRawInput = irBuilder_.CreateRawTensor(overlap->Datatype(), overlap->GetRawTensor()->GetRawShape(),
                                                          overlap->Format());
        reshapeRawInputs_[overlap->GetRawTensor()->GetRawMagic()] = reshapeRawInput;
    }
    return SUCCESS;
}

Status SplitReshape::ProcessMultitoOne(Function& function, Operation& op, const CalcOverlapPara& para)
{
    std::vector<int64_t> alignedShape = para.alignedShape;
    std::vector<int64_t> newInputViewTileOffset = para.newInputViewTileOffset;
    std::vector<int64_t> newInputViewTileShape = para.newInputViewTileShape;
    LogicalTensorPtr reshapeSource = para.reshapeSource;
    std::vector<int64_t> newReshapeSourceTileShape;
    std::vector<int64_t> newReshapeSourceTileOffset;
    ReshapeTilePara newInfo = {alignedShape, reshapeSource->GetRawTensor()->GetRawShape(), newInputViewTileOffset,
                               newInputViewTileShape};
    if (AlignToRaw(newInfo, newReshapeSourceTileOffset, newReshapeSourceTileShape) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Tensor,
                          "Failed to compute the tile of the raw input of reshapeSource, newInfo is %s.",
                          GetStr(newInfo).c_str());
        return FAILED;
    }
    // reshape前的tile无法assemble表达成一个tile时需要先reshape成alignshape然后再assemble成dstshape
    if (newReshapeSourceTileOffset.size() == 0 && newReshapeSourceTileShape.size() == 0) {
        APASS_LOG_WARN_F(Elements::Tensor, "One new assemble overlap tile block cannot cover the input of view op[%d].",
                         op.GetOpMagic());
        return SUCCESS; // 这种情况不会对reshape做切分, 动态shape的处理也跳过
    }
    ReshapeSourcePara sourcePara = {newReshapeSourceTileShape, newReshapeSourceTileOffset};
    if (UpdateForPerfectlyMatchWithAll(function, op, para, sourcePara) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Process ProcessMultitoOne failed. %s", GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status SplitReshape::UpdateReshapeOp(Function& function, Operation& op, const OverlapStatus& status,
                                     const CalcOverlapPara& calcpara)
{
    if (status == OverlapStatus::PERFECTLY_MATCH) {
        if (ProcessOnetoOne(function, op, calcpara) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Process ProcessOnetoOne of view[%d] failed. %s", op.GetOpMagic(),
                              GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        return SUCCESS;
    }
    if (status == OverlapStatus::BE_COVERED) {
        if (ProcessOnetoMulti(function, op, calcpara) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Process ProcessOnetoMulti of view[%d] failed. %s", op.GetOpMagic(),
                              GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        return SUCCESS;
    }
    if (status == OverlapStatus::PERFECTLY_MATCH_WITH_ALL) {
        if (ProcessMultitoOne(function, op, calcpara) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Process ProcessMultitoOne of view[%d] failed. %s", op.GetOpMagic(),
                              GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        return SUCCESS;
    }
    APASS_LOG_WARN_F(Elements::Tensor, "The new input of view[%d] intersects the input of assemble, skip splitreshape.",
                     op.GetOpMagic());
    return SUCCESS;
}

bool SplitReshape::HasAssembleLikeInputProducedByReduceAcc(const LogicalTensorPtr& reshapeSource,
                                                           int& assembleLikeOpMagic) const
{
    if (reshapeSource == nullptr || reshapeSource->GetRawTensor() == nullptr) {
        return false;
    }
    // assembleOutToInput_ 由 CollectAssembleInfo 填充，其门控 CollectCopyOut 已用 IsAssembleLike，
    // 故 OP_ASSEMBLE 与 OP_CONTRACT 均已记录，本函数天然覆盖 reduce_acc-contract-reshape 场景。
    auto it = assembleOutToInput_.find(reshapeSource->GetRawTensor()->GetRawMagic());
    if (it == assembleOutToInput_.end()) {
        return false;
    }
    for (const auto& input : it->second) {
        if (input == nullptr) {
            continue;
        }
        const bool hasReduceAccProducer = std::any_of(
            input->GetProducers().begin(), input->GetProducers().end(),
            [](const auto* producer) { return producer != nullptr && producer->GetOpcode() == Opcode::OP_REDUCE_ACC; });
        if (hasReduceAccProducer) {
            auto assembleIt = assembleOpPtrs_.find({input->GetMagic(), reshapeSource->GetMagic()});
            assembleLikeOpMagic = assembleIt == assembleOpPtrs_.end() ? -1 : assembleIt->second->GetOpMagic();
            return true;
        }
    }
    return false;
}

Status SplitReshape::CheckReshapeSkip(const LogicalTensorPtr& input, const LogicalTensorPtr& output,
                                      CheckOutputParam& checkOutputParam)
{
    if (input->shape == output->shape) {
        APASS_LOG_DEBUG_F(Elements::Tensor,
                          "The input and output of view op have the same shape, no need to split reshape (if "
                          "exist) for each view op.");
        return WARNING;
    }
    const int inputRawMagic = input->GetRawTensor()->GetRawMagic();
    auto reshapeSourceIter = reshapeSources_.find(inputRawMagic);
    if (reshapeSourceIter == reshapeSources_.end()) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "View op has no preceding reshape op.");
        return WARNING;
    }
    if (groupReshapeNoSplitRawMagics_.count(inputRawMagic) > 0) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Skip splitreshape because reshape op has %s flag.",
                          OpAttributeKey::groupReshapeNoSplit.c_str());
        return WARNING;
    }
    checkOutputParam.reshapeSource = reshapeSourceIter->second;
    int assembleLikeOpMagic = -1;
    if (HasAssembleLikeInputProducedByReduceAcc(checkOutputParam.reshapeSource, assembleLikeOpMagic)) {
        APASS_LOG_DEBUG_F(Elements::Tensor,
                          "Skip splitreshape because one assemble-like input is produced by ReduceAcc, op magic is %d.",
                          assembleLikeOpMagic);
        return WARNING;
    }
    size_t assembleLikeProducerCount = 0;
    for (const auto& producer : checkOutputParam.reshapeSource->GetProducers()) {
        if (producer != nullptr && IsAssembleLike(producer->GetOpcode())) {
            ++assembleLikeProducerCount;
        }
    }
    if (assembleLikeProducerCount <= 1) {
        APASS_LOG_DEBUG_F(Elements::Tensor,
                          "Skip splitreshape because reshape source has only one assemble-like producer.");
        return WARNING;
    }
    for (const auto& consumer : input->GetConsumers()) {
        if (consumer == nullptr) {
            continue;
        }
        if (!IsViewLike(consumer->GetOpcode())) {
            APASS_LOG_DEBUG_F(Elements::Tensor,
                              "Skip splitreshape because reshape output has non-view-like consumers.");
            return WARNING;
        }
    }
    const auto& reshapeInputShape = checkOutputParam.reshapeSource->GetRawTensor()->GetRawShape();
    const auto& reshapeOutputShape = input->GetRawTensor()->GetRawShape();
    if (CommonUtils::ContainsNegativeOne(reshapeInputShape) || CommonUtils::ContainsNegativeOne(reshapeOutputShape)) {
        APASS_LOG_DEBUG_F(Elements::Tensor,
                          "Found -1 in reshape input or output shape, input shape is %s, output shape is %s, "
                          "this scenario cannot be handled by split reshape pass.",
                          GetStr(reshapeInputShape).c_str(), GetStr(reshapeOutputShape).c_str());
        return WARNING;
    }
    return SUCCESS;
}

Status SplitReshape::CheckValidOp(const CheckParam& para, CheckOutputParam& checkOutputParam)
{
    auto input = para.input;
    auto inputView = para.inputView;
    if (CheckReshapeSkip(input, para.output, checkOutputParam) == WARNING) {
        return WARNING;
    }
    const int inputRawMagic = input->GetRawTensor()->GetRawMagic();
    if (ShapeAlign(checkOutputParam.reshapeSource->GetRawTensor()->GetRawShape(), input->GetRawTensor()->GetRawShape(),
                   checkOutputParam.alignedShape) == WARNING) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Can not construct align for rawshape.");
        return WARNING;
    }
    Status dynStatus = CheckDynStatus(checkOutputParam.alignedShape,
                                      checkOutputParam.reshapeSource->GetRawTensor()->GetRawShape(),
                                      input->GetRawTensor()->GetRawShape(), reshapeDynOutput_[inputRawMagic]);
    if (dynStatus == WARNING) {
        APASS_LOG_DEBUG_F(
            Elements::Tensor, "Undetermined variable in the changing axis, input is %s, output is %s, dynOutput is %s.",
            GetStr(checkOutputParam.reshapeSource->GetRawTensor()->GetRawShape()).c_str(),
            GetStr(input->GetRawTensor()->GetRawShape()).c_str(), GetStr(reshapeDynOutput_[inputRawMagic]).c_str());
        return WARNING;
    }
    if (dynStatus == FAILED) {
        APASS_LOG_ERROR_F(
            Elements::Tensor, "Illegal dynamic info for the reshape op, input is %s, output is %s, dynOutput is %s.",
            GetStr(checkOutputParam.reshapeSource->GetRawTensor()->GetRawShape()).c_str(),
            GetStr(input->GetRawTensor()->GetRawShape()).c_str(), GetStr(reshapeDynOutput_[inputRawMagic]).c_str());
        return FAILED;
    }
    checkOutputParam.curViewDynShape = GetViewValidShape(reshapeDynOutput_[inputRawMagic], inputView->offset, {},
                                                         inputView->shape);
    ReshapeTilePara reshapeInfo = {inputView->GetRawTensor()->GetRawShape(), checkOutputParam.alignedShape,
                                   inputView->offset, inputView->shape};
    Status alignRet = RawToAlign(reshapeInfo, checkOutputParam.newInputViewTileOffset,
                                 checkOutputParam.newInputViewTileShape);
    if (alignRet == WARNING) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Cannot process the cond, reshapeInfo is %s.", GetStr(reshapeInfo).c_str());
        return WARNING;
    }
    if (alignRet == FAILED) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Process RawToAlign failed, reshapeInfo is %s.",
                          GetStr(reshapeInfo).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status SplitReshape::CheckOp(Function& function, Operation& op)
{
    LogicalTensors overlaps;
    LogicalTensors newOverlaps;
    auto input = op.GetIOperands().front();
    auto output = op.GetOOperands().front();
    if (input == nullptr || output == nullptr || input->GetRawTensor() == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Invalid op [%d], at least one of input, output or raw tensor of input is nullptr; "
                          "Please check the input and output of op. %s",
                          op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(op.GetOpAttribute().get());
    if (viewOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Failed to obtain a view attr for op[%d]. %s", op.opmagic,
                          GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    auto inputView = irBuilder_.CreateTensorVar(input->GetRawTensor(), viewOpAttribute->GetFrom(), output->shape,
                                                std::vector<SymbolicScalar>{});
    CheckOutputParam checkOutputParam;
    CheckParam checkParam = {input, output, inputView};
    auto checkRet = CheckValidOp(checkParam, checkOutputParam);
    if (checkRet == WARNING) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Skip splitreshape for op[%d].", op.GetOpMagic());
        return SUCCESS;
    }
    if (checkRet == FAILED) {
        APASS_LOG_ERROR_F(Elements::Operation, "Failed to CheckValidOp for op[%d]. %s", op.GetOpMagic(),
                          GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    auto newInputView = irBuilder_.CreateTensorVar(inputView->GetRawTensor(), checkOutputParam.newInputViewTileOffset,
                                                   checkOutputParam.newInputViewTileShape,
                                                   std::vector<SymbolicScalar>{});
    auto reshapeOp = *input->GetProducers().begin();
    copyOutTilePara copyOutTile = {checkOutputParam.reshapeSource, reshapeOp->GetOpMagic(), inputView, newInputView,
                                   checkOutputParam.alignedShape};
    OverlapStatus status = OverlapStatus::NO_OVER_LAP;
    Status ret = ObtainCopyOutTile(function, copyOutTile, overlaps, newOverlaps, status);
    if (ret == WARNING) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Obtain CopyOutTile failed, skip splitreshape for [%d].",
                          op.GetOpMagic());
        return SUCCESS;
    }
    if (ret == FAILED) {
        APASS_LOG_ERROR_F(Elements::Operation, "Process ObtainCopyOutTile failed. %s", GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    CalcOverlapPara calcpara = {checkOutputParam.alignedShape,
                                checkOutputParam.reshapeSource,
                                checkOutputParam.newInputViewTileOffset,
                                checkOutputParam.newInputViewTileShape,
                                checkOutputParam.curViewDynShape,
                                overlaps,
                                newOverlaps,
                                input,
                                inputView,
                                output};
    if (UpdateReshapeOp(function, op, status, calcpara) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Process UpdateReshapeOp failed. %s", GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status SplitReshape::CheckCopyIn(Function& function)
{
    for (auto& op : function.Operations(false)) {
        if (!IsViewLike(op.GetOpcode())) {
            continue;
        }
        if (CheckOp(function, op) == FAILED) {
            APASS_LOG_ERROR_F(Elements::Operation, "Run CheckOp failed. %s", GetFormatBacktrace(op).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status SplitReshape::GetAssembleDynShape(const LogicalTensorPtr& input, const LogicalTensorPtr& output,
                                         const std::vector<int64_t>& toOffset,
                                         std::vector<SymbolicScalar>& dynValidShape)
{
    auto iter = reshapeOffset_.find(output);
    if (iter == reshapeOffset_.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find output from reshapeOffset.");
        return FAILED;
    }
    auto upperleftIdx = iter->second;
    auto dynInputShape = input->GetDynValidShape();
    if (dynInputShape.empty()) {
        return SUCCESS;
    }
    if (dynValidShape.empty()) {
        dynValidShape = output->GetDynValidShape();
    }
    if (dynValidShape.empty()) {
        for (size_t i = 0; i < dynInputShape.size(); ++i) {
            dynValidShape.push_back(irBuilder_.CreateConstInt(0));
        }
    }
    for (size_t i = 0U; i < dynValidShape.size(); i++) {
        dynValidShape[i] = std::max(dynValidShape[i],
                                    (dynInputShape[i] + (toOffset[i] - upperleftIdx[i])) * (dynInputShape[i] != 0));
    }
    return SUCCESS;
}

Status SplitReshape::AddAssembleOp(const MemoryType& memoryType, const std::vector<int64_t>& outputOffset,
                                   const LogicalTensorPtr& input, const LogicalTensorPtr& output,
                                   const Operation* originOp)
{
    assembles_.emplace_back(
        AssembleOp{memoryType, outputOffset, input, output, originOp, InferAssembleOpcode(originOp)});
    auto iter = reshapeOffset_.find(output);
    if (iter == reshapeOffset_.end()) {
        reshapeOffset_[output] = outputOffset;
        return SUCCESS;
    }
    auto curReshapeOffset = iter->second;
    std::vector<int64_t> upperleftIdx = outputOffset;
    if (outputOffset.empty() || outputOffset.size() != curReshapeOffset.size()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Incorrect axis, toOffset is %s, curReshapeOffset is %s.",
                          GetStr(outputOffset).c_str(), GetStr(curReshapeOffset).c_str());
        return FAILED;
    }
    for (size_t i = 0; i < outputOffset.size(); ++i) {
        upperleftIdx[i] = std::min(curReshapeOffset[i], outputOffset[i]);
    }
    reshapeOffset_[output] = upperleftIdx;
    return SUCCESS;
}

Status SplitReshape::GetReshapeDynShape(const std::shared_ptr<ReshapeOp>& op,
                                        std::vector<SymbolicScalar>& dynValidShape)
{
    auto iter = viewOffset_.find(op);
    if (iter == viewOffset_.end()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Cannot find reshapeop with sharedptr.");
        return FAILED;
    }
    auto upperleftIdx = iter->second;
    for (const auto& validShape : op->dynValidShapes) {
        if (dynValidShape.empty()) {
            for (size_t i = 0; i < validShape.size(); ++i) {
                dynValidShape.emplace_back(0);
            }
        }
        if (upperleftIdx.empty()) {
            upperleftIdx.assign(validShape.size(), 0);
        }
        if (dynValidShape.size() != validShape.size() || upperleftIdx.size() != validShape.size()) {
            APASS_LOG_ERROR_F(Elements::Tensor,
                              "Incorrect size, dynValidShape is %s, validShape is %s, upperleftIdx is %s.",
                              GetStr(dynValidShape).c_str(), GetStr(validShape).c_str(), GetStr(upperleftIdx).c_str());
            return FAILED;
        }
        for (size_t i = 0; i < validShape.size(); ++i) {
            dynValidShape[i] = std::max(dynValidShape[i], validShape[i] - op->output->GetOffset()[i]);
        }
    }
    return SUCCESS;
}

Status SplitReshape::AddOperation(Function& function)
{
    std::vector<SymbolicScalar> dynValidShape;
    for (const auto& a : assembles_) {
        dynValidShape.clear();
        if (GetAssembleDynShape(a.input, a.output, a.toOffset, dynValidShape) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Get assemble dynamic shape for AddOperation failed.");
            return FAILED;
        }
        auto& newCopyOut = GraphUtils::AddAssembleOperation(function, a, {dynValidShape});
        if (a.originOp != nullptr && !a.originOp->result_token_.empty()) {
            InheritTokenFromOldAssemble(function, *a.originOp, newCopyOut);
        }
        APASS_LOG_INFO_F(Elements::Operation,
                         "ADD %s, magic %d, IOperand tensor magic %d OOperand tensor magic %d, dynValidShape %s.",
                         newCopyOut.GetOpcodeStr().c_str(), newCopyOut.opmagic, a.input->GetMagic(),
                         a.output->GetMagic(), GetStr(a.output->GetDynValidShape()).c_str());
    }
    for (const auto& b : reshapes_) {
        dynValidShape.clear();
        if (GetReshapeDynShape(b.second, dynValidShape) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Get reshape dynamic shape for AddOperation failed.");
            return FAILED;
        }
        auto& newReshape = GraphUtils::AddReshapeOperation(function, b.second->input, b.second->output, *b.second,
                                                           dynValidShape);
        APASS_LOG_INFO_F(
            Elements::Operation,
            "ADD OP_RESHAPE, magic %d, IOperand tensor magic %d OOperand tensor magic %d, dynValidShape %s.",
            newReshape.opmagic, b.second->input->GetMagic(), b.second->output->GetMagic(),
            GetStr(b.second->output->GetDynValidShape()).c_str());
    }
    return SUCCESS;
}

Status SplitReshape::SetMemoryType(Function& function)
{
    for (const auto& op : function.Operations(false)) {
        if (!IsViewLike(op.GetOpcode())) {
            continue;
        }
        for (const auto consumerOp : op.ConsumerOps()) {
            if (consumerOp->GetOpcode() == Opcode::OP_RESHAPE &&
                op.GetOOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_UNKNOWN) {
                op.GetOOperands()[0]->SetMemoryTypeBoth(consumerOp->GetOOperands()[0]->GetMemoryTypeOriginal());
            }
        }
    }
    return SUCCESS;
}

Status SplitReshape::DefaultEnabledPreCheck(Function& function) { return checker_.DoDefaultEnabledPreCheck(function); }

} // namespace npu::tile_fwk
