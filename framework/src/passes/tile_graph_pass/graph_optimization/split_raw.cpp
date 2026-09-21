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
 * \file split_raw.cpp
 * \brief
 */

#include "split_raw.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/graph_utils.h"

#define MODULE_NAME "SplitRawTensor"

namespace npu {
namespace tile_fwk {
std::vector<int64_t> SplitRawTensor::UpdateOffset(std::vector<int64_t>& offset, const std::vector<int64_t>& diff) const
{
    std::vector<int64_t> result = offset;
    for (size_t i = 0; i < offset.size(); i++) {
        if (offset[i] >= diff[i]) {
            result[i] = offset[i] - diff[i];
        }
    }
    return result;
}

std::vector<SymbolicScalar> SplitRawTensor::UpdateDynOffset(std::vector<SymbolicScalar>& offset,
                                                            const std::vector<SymbolicScalar>& diff) const
{
    std::vector<SymbolicScalar> result = offset;
    for (size_t i = 0; i < offset.size(); i++) {
        const auto shouldUpdate = (offset[i] >= diff[i]).Simplify();
        const auto updatedOffset = (offset[i] - diff[i]).Simplify();
        if (shouldUpdate.IsImmediate()) {
            if (shouldUpdate.Concrete() != 0) {
                result[i] = updatedOffset;
            }
        } else {
            result[i] = (offset[i] + (updatedOffset - offset[i]) * shouldUpdate).Simplify();
        }
    }
    return result;
}

std::vector<OpImmediate> SplitRawTensor::UpdateImmediateOffset(std::vector<OpImmediate>& offset,
                                                               const TensorOffset& tensorOffset) const
{
    std::vector<OpImmediate> result = offset;
    std::vector<SymbolicScalar> diff = tensorOffset.GetDynOffset().empty() ?
                                           OpImmediate::ToSpecified(OpImmediate::Specified(tensorOffset.GetOffset())) :
                                           tensorOffset.GetDynOffset();
    if (offset.size() != diff.size()) {
        APASS_LOG_WARN_F(Elements::Operation, "Copy op offset size[%zu] is not equal to tensor offset size[%zu].",
                         offset.size(), diff.size());
        return result;
    }
    std::vector<size_t> specifiedIndex;
    std::vector<SymbolicScalar> specifiedOffset;
    std::vector<SymbolicScalar> specifiedDiff;
    for (size_t i = 0; i < offset.size(); i++) {
        if (!offset[i].IsSpecified()) {
            continue;
        }
        specifiedIndex.emplace_back(i);
        specifiedOffset.emplace_back(offset[i].GetSpecifiedValue());
        specifiedDiff.emplace_back(diff[i]);
    }
    std::vector<SymbolicScalar> updatedOffset = UpdateDynOffset(specifiedOffset, specifiedDiff);
    for (size_t i = 0; i < specifiedIndex.size(); i++) {
        result[specifiedIndex[i]] = OpImmediate::Specified(updatedOffset[i]);
    }
    return result;
}

void SplitRawTensor::UpdateConsumerView(Function& function, const LogicalTensorPtr& logicalTensor) const
{
    /* All the consumer View op's attr offset should be corret */
    /* 1. 更新View相关的属性 */
    TensorOffset tensorOffset = logicalTensor->GetTensorOffset();
    for (const auto& viewOp : logicalTensor->GetConsumers()) {
        if (!IsViewLike(viewOp->GetOpcode())) {
            continue;
        }
        auto& output = viewOp->oOperand[0];
        if (function.IsFromOutCast(output)) {
            APASS_LOG_WARN_F(Elements::Tensor,
                             "View-like op oOperand tensor[%d] is outCast; Please check if it is an external output.",
                             output->GetMagic());
            continue;
        }
        auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(viewOp->GetOpAttribute().get());
        if (viewOpAttribute != nullptr) {
            // VIEW操作的offset要相应被修改, 要减去被拆分LogicalTensor的offset
            auto& fromOffset = viewOpAttribute->GetFromOffset();
            fromOffset = UpdateOffset(fromOffset, tensorOffset.GetOffset());
            if (!viewOpAttribute->GetFromDynOffset().empty()) {
                auto& fromDynOffset = viewOpAttribute->GetFromDynOffset();
                if (!tensorOffset.GetDynOffset().empty()) {
                    fromDynOffset = UpdateDynOffset(fromDynOffset, tensorOffset.GetDynOffset());
                } else {
                    fromDynOffset = UpdateDynOffset(
                        fromDynOffset, OpImmediate::ToSpecified(OpImmediate::Specified(tensorOffset.GetOffset())));
                }
            }
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "Update View-like op needs fromOffset: %d.", viewOp->GetOpMagic());
    }
}

void SplitRawTensor::UpdateProducerAssemble(Function& function, const LogicalTensorPtr& logicalTensor) const
{
    /* 1. 更新Assemble相关的属性 */
    // Assemble1 ->
    //              logicalTensor(UB) -> Reshape -> UB
    // Assemble2 ->
    TensorOffset tensorOffset = logicalTensor->GetTensorOffset();
    for (const auto& assembleOp : logicalTensor->GetProducers()) {
        if (!IsAssembleLike(assembleOp->GetOpcode())) {
            continue;
        }
        auto& input = assembleOp->iOperand[0];
        if (function.IsFromInCast(input)) {
            APASS_LOG_WARN_F(Elements::Operation,
                             "Assemble-like op iOperand tensor[%d] is inCast; Please check if it is an external input.",
                             input->GetMagic());
            continue;
        }
        auto assembleOpAttribute = dynamic_cast<AssembleOpAttribute*>(assembleOp->GetOpAttribute().get());
        if (assembleOpAttribute != nullptr) {
            // Assemble操作的offset要相应被修改, 要减去被拆分LogicalTensor的offset
            auto& toOffset = assembleOpAttribute->GetToOffset();
            toOffset = UpdateOffset(toOffset, tensorOffset.GetOffset());
            if (!assembleOpAttribute->GetToDynOffset().empty()) {
                auto& toDynOffset = assembleOpAttribute->GetToDynOffset();
                if (!tensorOffset.GetDynOffset().empty()) {
                    toDynOffset = UpdateDynOffset(toDynOffset, tensorOffset.GetDynOffset());
                } else {
                    toDynOffset = UpdateDynOffset(
                        toDynOffset, OpImmediate::ToSpecified(OpImmediate::Specified(tensorOffset.GetOffset())));
                }
            }
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "Update Assemble-like op needs toOffset: %d.", assembleOp->GetOpMagic());
    }
}

void SplitRawTensor::UpdateProducerShmemGet(Function& function, const LogicalTensorPtr& logicalTensor) const
{
    TensorOffset tensorOffset = logicalTensor->GetTensorOffset();
    for (const auto& shmemGetOp : logicalTensor->GetProducers()) {
        if (shmemGetOp->GetOpcode() != Opcode::OP_SHMEM_GET) {
            continue;
        }
        if (shmemGetOp->oOperand.empty() || shmemGetOp->oOperand[0] != logicalTensor) {
            continue;
        }
        auto& output = shmemGetOp->oOperand[0];
        if (function.IsFromOutCast(output)) {
            APASS_LOG_WARN_F(Elements::Operation,
                             "OP_SHMEM_GET oOperand tensor[%d] is outCast; Please check if it is an external output.",
                             output->GetMagic());
            continue;
        }
        auto copyOpAttribute = dynamic_cast<CopyOpAttribute*>(shmemGetOp->GetOpAttribute().get());
        if (copyOpAttribute != nullptr && copyOpAttribute->IsCopyOut()) {
            auto& toOffset = copyOpAttribute->GetToOffset();
            toOffset = UpdateImmediateOffset(toOffset, tensorOffset);
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "Update ShmemGet op needs toOffset: %d.", shmemGetOp->GetOpMagic());
    }
}

bool SplitRawTensor::ShouldProcessTensor(Function& function, const LogicalTensorPtr& singleTensor) const
{
    // 检查raw shape 和 shape 是否相等
    if ((singleTensor->GetShape() == singleTensor->tensor->GetRawShape())) {
        return false;
    }
    // 检查是否为InCast或OutCast
    if (function.IsFromOutCast(singleTensor) || function.IsFromInCast(singleTensor)) {
        APASS_LOG_WARN_F(Elements::Tensor,
                         "Tensor[%d] is inCast or outCast; Please check if it is an external input/output",
                         singleTensor->GetMagic());
        return false;
    }
    return true;
}

bool SplitRawTensor::SplitLogicalTensor(Function& function, const LogicalTensorPtr& logicalTensor) const
{
    if (!ShouldProcessTensor(function, logicalTensor)) {
        return false;
    }
    logicalTensor->tensor = std::make_shared<RawTensor>(logicalTensor->tensor->datatype, logicalTensor->GetShape(),
                                                        logicalTensor->Format(), logicalTensor->Symbol());
    APASS_LOG_DEBUG_F(Elements::Operation,
                      "SplitRawTensor::SplitRaw: tensor[%d] updated new raw tensor[%d] with the same raw shape.",
                      logicalTensor->GetMagic(), logicalTensor->GetRawMagic());
    UpdateConsumerView(function, logicalTensor);
    UpdateProducerAssemble(function, logicalTensor);
    UpdateProducerShmemGet(function, logicalTensor);
    const Offset zeroOffset(logicalTensor->GetOffset().size(), 0);
    const std::vector<SymbolicScalar> emptyDynOffset;
    logicalTensor->UpdateOffset(TensorOffset(zeroOffset, emptyDynOffset));
    return true;
}

/*
遍历所有tensor，如果rawTensor的shape大于tensor的shape，
且rawTensor不是InCast和OutCast(当前Incast、OutCast的Symbol不在tensormap里)、且tensor不是ddr，
那么需要将重新创建一个shape和当前tensor相同的rawTensor
*/
void SplitRawTensor::SplitRaw(Function& function) const
{
    // 为了按序访问tensormap, 将tensormap转化为有序map
    TensorSet allTensors = GraphUtils::GetAllTensors(function);
    std::map<int, TensorSet> omap;
    for (const auto& tensor : allTensors) {
        if (tensor && tensor->tensor) {
            omap[tensor->tensor->rawmagic].insert(tensor);
        }
    }
    for (const auto& ele : omap) {
        for (const auto& logicalTensor : ele.second) {
            SplitLogicalTensor(function, logicalTensor);
        }
    }
}

Status SplitRawTensor::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start SplitRaw.");
    SplitRaw(function);
    APASS_LOG_INFO_F(Elements::Function, "===> End SplitRaw.");
    return SUCCESS;
}
Status SplitRawTensor::PostCheck(Function& function) { return checker.DoPostCheck(function); }
} // namespace tile_fwk
} // namespace npu
