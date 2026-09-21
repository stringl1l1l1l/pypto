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
 * \file generate_move_op.cpp
 * \brief
 */

#include "passes/tile_graph_pass/data_path/generate_move_op.h"
#include "interface/configs/config_manager_ng.h"
#include "interface/function/function.h"
#include "interface/tensor/irbuilder.h"
#include "interface/operation/operation.h"
#include "interface/tensor/irbuilder.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "passes/pass_check/generate_move_op_checker.h"
#include "passes/pass_utils/dead_operation_eliminate.h"
#include "passes/pass_log/pass_log.h"
#include "tilefwk/error_code.h"

#define MODULE_NAME "GenerateMoveOp"

namespace npu::tile_fwk {
constexpr int64_t INNER_PAD_VALUE = 32;
constexpr int64_t OUTER_PAD_VALUE = 16;
const Offset ZERO_OFFSET = {0, 0};

int64_t GenerateMoveOp::PadUB(int64_t dim, int64_t padValue)
{
    ASSERT(TensorErr::TENSOR_INVALID_MEMORY_TYPE, padValue > 0);
    return (dim + padValue - 1) / padValue * padValue;
}

bool GenerateMoveOp::IsTransFormatOp(Opcode opcode)
{
    return opcode == Opcode::OP_TRANS_FORMAT_L1 || opcode == Opcode::OP_TRANS_FORMAT_L0C;
}

bool GenerateMoveOp::HasConvConsumer(const Operation& op)
{
    if (op.GetOOperands().empty() || op.GetOOperands().front() == nullptr) {
        return false;
    }
    for (auto* consumer : op.GetOOperands().front()->GetConsumers()) {
        if (consumer != nullptr && consumer->HasAttr(OpAttributeKey::isConv) &&
            consumer->GetBoolAttribute(OpAttributeKey::isConv)) {
            return true;
        }
    }
    return false;
}

bool GenerateMoveOp::HasConvProducer(const Operation& op)
{
    if (op.GetIOperands().empty() || op.GetIOperands().front() == nullptr) {
        return false;
    }
    for (auto* producer : op.GetIOperands().front()->GetProducers()) {
        if (producer != nullptr && producer->HasAttr(OpAttributeKey::isConv) &&
            producer->GetBoolAttribute(OpAttributeKey::isConv)) {
            return true;
        }
    }
    return false;
}

Status GenerateMoveOp::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation, "===> Start GenerateMoveOp");
    Status status = CreateMoveOp(function);
    if (status != SUCCESS) {
        return status;
    }
    APASS_LOG_INFO_F(Elements::Operation, "===> End GenerateMoveOp");
    return SUCCESS;
}

Status GenerateMoveOp::PreCheck(Function& function)
{
    GenerateMoveOpChecker checker;
    return checker.DoPreCheck(function);
}

Status GenerateMoveOp::PostCheck(Function& function)
{
    GenerateMoveOpChecker checker;
    return checker.DoPostCheck(function);
}

bool GenerateMoveOp::HasSpecificConsumer(const Operation& op) const
{
    auto viewResult = op.GetOOperands()[0];
    const auto& consumersCopy = viewResult->GetConsumers();

    for (auto childOp : consumersCopy) {
        if (childOp->GetOpcode() == Opcode::OP_INDEX_OUTCAST || childOp->GetOpcode() == Opcode::OP_RESHAPE) {
            return true;
        }
    }
    return false;
}

Status GenerateMoveOp::A23CreateMoveOpForView(Function& function, Operation& op) const
{
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(op.GetOpAttribute().get());
    bool isGmInput = op.iOperand.front()->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR;
    bool isGmOutput = op.oOperand.front()->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR;
    if (isGmInput) {
        // case1: VIEW转copyIn
        return ProcessGmInput(isGmOutput, op, viewOpAttribute);
    } else if (op.oOperand.front()->GetMemoryTypeOriginal() == MemoryType::MEM_L0A) {
        // case2: VIEW转L0A/L0AT
        return ProcessL0A(op, viewOpAttribute);
    } else if (op.oOperand.front()->GetMemoryTypeOriginal() == MemoryType::MEM_L0B) {
        // case3: VIEW转L0B/L0BT
        return ProcessL0B(op, viewOpAttribute);
    } else {
        // case4: VIEW转其他搬运op
        return ProcessDefault(function, op, viewOpAttribute);
    }
    return SUCCESS;
}

Status GenerateMoveOp::ProcessGmInput(bool& isGmOutput, Operation& op, ViewOpAttribute* viewOpAttribute) const
{
    if (isGmOutput && HasSpecificConsumer(op)) {
        return SUCCESS;
    }
    if ((!isGmOutput)) {
        const auto outputTensor = op.GetOOperands().empty() ? nullptr : op.GetOOperands().front();
        const bool isConv = outputTensor != nullptr && outputTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L1 &&
                            HasConvConsumer(op);
        op.SetOpCode(isConv ? Opcode::OP_L1_COPY_IN_CONV : Opcode::OP_COPY_IN);
        // A5 conv 输入: 合并 consumer OP_TRANS_FORMAT_L1，删除冗余操作
        if (isConv && Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) {
            return CollapseTransFormatL1Consumer(op, viewOpAttribute);
        }
        SetCopyAttr(op, viewOpAttribute);
    }
    return SUCCESS;
}

Status GenerateMoveOp::CollapseTransFormatL1Consumer(Operation& op, ViewOpAttribute* viewOpAttribute) const
{
    // 此处向 consumer 方向合并：将 copy_in 的 output 替换为 TRANS_FORMAT_L1 的 output，
    // 迁移 TRANS_FORMAT_L1 的属性，并标记删除 TRANS_FORMAT_L1。
    auto viewOutput = op.GetOOperands().front();
    const auto& consumers = viewOutput->GetConsumers();
    if (consumers.size() != 1) {
        return SUCCESS;
    }
    auto transFormatOp = *consumers.begin();
    if (transFormatOp == nullptr || transFormatOp->GetOpcode() != Opcode::OP_TRANS_FORMAT_L1 ||
        transFormatOp->GetIOperands().size() != 1 || transFormatOp->GetOOperands().size() != 1 ||
        transFormatOp->GetIOperands().front() != viewOutput || transFormatOp->GetOOperands().front() == nullptr) {
        return SUCCESS;
    }
    auto transFormatOutput = transFormatOp->GetOOperands().front();
    auto transFormatAttrs = transFormatOp->GetAllAttr();
    op.ReplaceOOperand(0, transFormatOutput);
    op.GetAllAttr() = std::move(transFormatAttrs);
    op.SetAttribute(OpAttributeKey::isConv, true);
    if (viewOpAttribute != nullptr) {
        viewOpAttribute->SetToDynValidShape(transFormatOutput->GetDynValidShape());
        viewOpAttribute->SetToType(transFormatOutput->GetMemoryTypeToBe());
    }
    transFormatOp->SetAsDeleted();

    auto copyAttr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(viewOpAttribute->GetFromTensorOffset()), viewOpAttribute->GetTo(),
        OpImmediate::Specified(op.iOperand.front()->shape),
        OpImmediate::Specified(op.iOperand.front()->tensor->GetDynRawShape()),
        OpImmediate::Specified(viewOpAttribute->GetToDynValidShape()));
    op.GetOOperands()[0]->UpdateDynValidShape(viewOpAttribute->GetToDynValidShape());
    op.SetOpAttribute(copyAttr);
    return SUCCESS;
}

Status GenerateMoveOp::ProcessL0A(Operation& op, ViewOpAttribute* viewOpAttribute) const
{
    auto isTrans = (op.HasAttr("op_attr_l1_to_l0_transpose")) ? op.GetBoolAttribute("op_attr_l1_to_l0_transpose") : 0;
    if (isTrans) {
        op.SetOpCode(Opcode::OP_L1_TO_L0_AT);
    } else {
        op.SetOpCode(Opcode::OP_L1_TO_L0A);
    }
    op.SetCoreType(CoreType::AIC);
    SetCopyAttr(op, viewOpAttribute);
    return SUCCESS;
}

Status GenerateMoveOp::ProcessL0B(Operation& op, ViewOpAttribute* viewOpAttribute) const
{
    auto isTrans = (op.HasAttr("op_attr_l1_to_l0_transpose")) ? op.GetBoolAttribute("op_attr_l1_to_l0_transpose") : 0;
    if (isTrans) {
        op.SetOpCode(Opcode::OP_L1_TO_L0_BT);
    } else {
        op.SetOpCode(Opcode::OP_L1_TO_L0B);
    }
    op.SetCoreType(CoreType::AIC);
    SetCopyAttr(op, viewOpAttribute);
    return SUCCESS;
}

Status GenerateMoveOp::ProcessL0AMX(Operation& op, ViewOpAttribute* viewOpAttribute) const
{
    op.SetOpCode(Opcode::OP_L1_TO_L0A_SCALE);
    op.SetCoreType(CoreType::AIC);
    auto input = op.GetIOperands()[0];
    auto prodOp = *input->GetProducers().begin();
    if (prodOp->GetOpcode() == Opcode::OP_COPY_IN && input->GetMemoryTypeOriginal() == MemoryType::MEM_L1) {
        prodOp->SetOpCode(Opcode::OP_L1_COPY_IN_A_SCALE);
        prodOp->SetCoreType(CoreType::AIC);
    }
    SetCopyAttr(op, viewOpAttribute);
    return SUCCESS;
}

Status GenerateMoveOp::ProcessL0BMX(Operation& op, ViewOpAttribute* viewOpAttribute) const
{
    op.SetOpCode(Opcode::OP_L1_TO_L0B_SCALE);
    op.SetCoreType(CoreType::AIC);
    auto input = op.GetIOperands()[0];
    auto prodOp = *input->GetProducers().begin();
    if (prodOp->GetOpcode() == Opcode::OP_COPY_IN && input->GetMemoryTypeOriginal() == MemoryType::MEM_L1) {
        prodOp->SetOpCode(Opcode::OP_L1_COPY_IN_B_SCALE);
        prodOp->SetCoreType(CoreType::AIC);
    }
    SetCopyAttr(op, viewOpAttribute);
    return SUCCESS;
}

Status GenerateMoveOp::ProcessDefault(Function& function, Operation& op, ViewOpAttribute* viewOpAttribute) const
{
    auto from = op.iOperand.front()->GetMemoryTypeOriginal();
    auto to = op.oOperand.front()->GetMemoryTypeOriginal();
    if (from == to) {
        return SUCCESS;
    }
    Status status = SetOpcodeByMemPath(op, from, to);
    if (status != SUCCESS) {
        return status;
    }
    if (op.GetOpcode() == Opcode::OP_UB_COPY_L1) {
        ProcessUB2L1(function, op);
        SetUB2L1CopyAttr(op, op.GetOOperands()[0]->GetShape(),
                         OpImmediate::Specified(viewOpAttribute->GetFromTensorOffset()),
                         OpImmediate::Specified(ZERO_OFFSET), Matrix::CopyMode::EXTRACT);
        return SUCCESS;
    }
    if (op.GetOpcode() == Opcode::OP_L0C_TO_L1) {
        SetL0C2L1CopyAttr(op, op.GetOOperands()[0]->GetShape(),
                          OpImmediate::Specified(viewOpAttribute->GetFromTensorOffset()),
                          OpImmediate::Specified(ZERO_OFFSET), Matrix::CopyMode::EXTRACT);
    } else if (op.GetOpcode() == Opcode::OP_L0C_COPY_UB) {
        SetL0C2UBCopyAttr(op, op.GetOOperands()[0]->GetShape(),
                          OpImmediate::Specified(viewOpAttribute->GetFromTensorOffset()),
                          OpImmediate::Specified(ZERO_OFFSET), Matrix::CopyMode::EXTRACT);
        op.SetAttribute(OpAttributeKey::isCube, true);
    } else {
        SetCopyAttr(op, viewOpAttribute);
    }
    return SUCCESS;
}

Status GenerateMoveOp::A5CreateMoveOpForView(Function& function, Operation& op) const
{
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(op.GetOpAttribute().get());
    bool isGmInput = op.iOperand.front()->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR;
    bool isGmOutput = op.oOperand.front()->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR;
    if (isGmInput) {
        // case1: VIEW转copyIn
        return ProcessGmInput(isGmOutput, op, viewOpAttribute);
    } else {
        auto dstMemType = op.oOperand.front()->GetMemoryTypeOriginal();
        switch (dstMemType) {
            case MemoryType::MEM_L0A:
                return ProcessL0A(op, viewOpAttribute);
            case MemoryType::MEM_L0B:
                return ProcessL0B(op, viewOpAttribute);
            case MemoryType::MEM_L0AMX:
                return ProcessL0AMX(op, viewOpAttribute);
            case MemoryType::MEM_L0BMX:
                return ProcessL0BMX(op, viewOpAttribute);
            default:
                return ProcessDefault(function, op, viewOpAttribute);
        }
    }
    return SUCCESS;
}

void GenerateMoveOp::SetCopyAttr(Operation& op, ViewOpAttribute* viewOpAttribute) const
{
    auto copyAttr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(viewOpAttribute->GetFromTensorOffset()), viewOpAttribute->GetTo(),
        OpImmediate::Specified(op.oOperand.front()->shape),
        OpImmediate::Specified(op.iOperand.front()->tensor->GetDynRawShape()),
        OpImmediate::Specified(viewOpAttribute->GetToDynValidShape()));
    op.GetOOperands()[0]->UpdateDynValidShape(viewOpAttribute->GetToDynValidShape());
    op.SetOpAttribute(copyAttr);
}

void GenerateMoveOp::SetL0C2L1CopyAttr(Operation& op, const Shape& realShape,
                                       const std::vector<OpImmediate>& fromOffset,
                                       const std::vector<OpImmediate>& toOffset, Matrix::CopyMode copyMode) const
{
    IRBuilder builder;
    std::vector<SymbolicScalar> validShape;
    for (auto dim : realShape) {
        SymbolicScalar scal = builder.CreateConstInt(dim);
        validShape.push_back(scal);
    }
    auto copyAttr = std::make_shared<CopyOpAttribute>(
        fromOffset, op.oOperand.front()->GetMemoryTypeOriginal(), OpImmediate::Specified(realShape),
        OpImmediate::Specified(op.iOperand.front()->tensor->GetDynRawShape()), OpImmediate::Specified(validShape));
    copyAttr->SetToOffset(toOffset);
    op.SetOpAttribute(copyAttr);
    op.SetAttr(OpAttributeKey::copyIsNZ, static_cast<int64_t>(1));
    op.SetAttribute(OpAttributeKey::localCopyLocalMode, static_cast<int64_t>(copyMode));
}

void GenerateMoveOp::SetL0C2UBCopyAttr(Operation& op, const Shape& realShape,
                                       const std::vector<OpImmediate>& fromOffset,
                                       const std::vector<OpImmediate>& toOffset, Matrix::CopyMode copyMode) const
{
    IRBuilder builder;
    std::vector<SymbolicScalar> validShape;
    for (auto dim : realShape) {
        SymbolicScalar scal = builder.CreateConstInt(dim);
        validShape.push_back(scal);
    }
    auto copyAttr = std::make_shared<CopyOpAttribute>(
        fromOffset, op.oOperand.front()->GetMemoryTypeOriginal(), OpImmediate::Specified(realShape),
        OpImmediate::Specified(op.iOperand.front()->tensor->GetDynRawShape()), OpImmediate::Specified(validShape));
    copyAttr->SetToOffset(toOffset);
    op.SetOpAttribute(copyAttr);
    op.SetAttribute(OpAttributeKey::isCube, true);
    op.SetAttribute(OpAttributeKey::localCopyLocalMode, static_cast<int64_t>(copyMode));
}

void GenerateMoveOp::SetUB2L1CopyAttr(Operation& op, const Shape& copyShape, const std::vector<OpImmediate>& fromOffset,
                                      const std::vector<OpImmediate>& toOffset, Matrix::CopyMode copyMode) const
{
    IRBuilder builder;
    // 实际搬运的 shape 转换为 validShape
    std::vector<SymbolicScalar> validShape;
    for (auto dim : copyShape) {
        SymbolicScalar scal = builder.CreateConstInt(dim);
        validShape.push_back(scal);
    }
    // 创建 CopyOpAttribute
    auto copyAttr = std::make_shared<CopyOpAttribute>(
        fromOffset,                                                            // fromOffset
        op.oOperand.front()->GetMemoryTypeOriginal(),                          // to (L1)
        OpImmediate::Specified(copyShape),                                     // shape (实际搬运的 shape)
        OpImmediate::Specified(op.iOperand.front()->tensor->GetDynRawShape()), // rawShape (srcValidShape)
        OpImmediate::Specified(validShape)                                     // toDynValidShape (dstValidShape)
    );
    copyAttr->SetToOffset(toOffset);
    op.SetOpAttribute(copyAttr);
    op.SetAttribute(OpAttributeKey::localCopyLocalMode, static_cast<int64_t>(copyMode));
}

Status GenerateMoveOp::SetOpcodeByMemPath(Operation& op, MemoryType from, MemoryType to) const
{
    std::pair<MemoryType, MemoryType> memPathPair = {from, to};
    auto it = platformPathMap.find(memPathPair);
    if (it == platformPathMap.end()) {
        APASS_LOG_ERROR_F(Elements::Operation, "No memory path found from %s to %s for operation %s[%d].",
                          BriefMemoryTypeToString(from).c_str(), BriefMemoryTypeToString(to).c_str(),
                          op.GetOpcodeStr().c_str(), op.GetOpMagic());
        return FAILED;
    }
    auto opcodeFindByPath = it->second;
    op.SetOpCode(opcodeFindByPath);
    return SUCCESS;
}

Status GenerateMoveOp::ProcessConvAssemble(Operation& op, AssembleOpAttribute* assembleOpAttribute) const
{
    // 调用方已校验 attr / input / output 非空，此处仅校验 input/output 数量为 1
    if (op.GetIOperands().size() != 1 || op.GetOOperands().size() != 1) {
        APASS_LOG_ERROR_F(Elements::Operation, "Conv ASSEMBLE[%d] must have exactly one input/output.",
                          op.GetOpMagic());
        return FAILED;
    }
    auto assembleInput = op.iOperand.front();
    const auto& producers = assembleInput->GetProducers();
    if (producers.size() != 1) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Conv ASSEMBLE[%d] input must have exactly one TRANS_FORMAT producer, got %zu.",
                          op.GetOpMagic(), producers.size());
        return FAILED;
    }
    auto transFormatOp = *producers.begin();
    if (transFormatOp == nullptr || !IsTransFormatOp(transFormatOp->GetOpcode()) ||
        transFormatOp->GetIOperands().size() != 1 || transFormatOp->GetOOperands().size() != 1 ||
        transFormatOp->GetIOperands().front() == nullptr || transFormatOp->GetOOperands().front() != assembleInput) {
        APASS_LOG_ERROR_F(Elements::Operation, "Conv ASSEMBLE[%d] input producer must be a single-input TRANS_FORMAT.",
                          op.GetOpMagic());
        return FAILED;
    }

    auto transFormatInput = transFormatOp->GetIOperands().front();
    auto assembleOutput = op.GetOOperands().front();
    if (assembleOutput->GetRawTensor() == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Conv ASSEMBLE[%d] output tensor has no raw tensor.", op.GetOpMagic());
        return FAILED;
    }
    auto transFormatAttrs = transFormatOp->GetAllAttr();
    auto copyAttr = std::make_shared<CopyOpAttribute>(
        transFormatInput->GetMemoryTypeOriginal(), OpImmediate::Specified(assembleOpAttribute->GetToTensorOffset()),
        OpImmediate::Specified(assembleOutput->GetShape()),
        OpImmediate::Specified(assembleOutput->GetRawTensor()->GetDynRawShape()),
        OpImmediate::Specified(transFormatInput->GetDynValidShape()));

    op.SetOpCode(Opcode::OP_L0C_COPY_OUT_CONV);
    op.GetAllAttr() = std::move(transFormatAttrs);
    op.SetAttribute(OpAttributeKey::isConv, true);
    op.SetOpAttribute(copyAttr);
    op.ReplaceIOperand(0, transFormatInput);
    transFormatOp->SetAsDeleted();
    return SUCCESS;
}

bool GenerateMoveOp::TryInsertModeDispatch(Function& function, Operation& op, AssembleOpAttribute* assembleOpAttribute,
                                           MemoryType inputMemtype, MemoryType outputMemtype) const
{
    auto ASSEMBLE_in = op.iOperand.front();
    auto parentOp = *ASSEMBLE_in->GetProducers().begin();
    if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510 && inputMemtype == MemoryType::MEM_L1 &&
        outputMemtype == MemoryType::MEM_DEVICE_DDR && parentOp->GetOpcode() == Opcode::OP_UB_COPY_L1 &&
        parentOp->GetIOperands().size() == 1) {
        auto l1Input = ASSEMBLE_in;
        ASSEMBLE_in = parentOp->GetIOperands().front();
        op.ReplaceIOperand(0, ASSEMBLE_in);
        if (l1Input->GetConsumers().empty()) {
            parentOp->SetAsDeleted();
        }
    }
    if (inputMemtype == MemoryType::MEM_L0C && outputMemtype == MemoryType::MEM_L1) {
        SetOpcodeByMemPath(op, inputMemtype, outputMemtype);
        SetL0C2L1CopyAttr(op, op.GetIOperands()[0]->GetShape(), OpImmediate::Specified(ZERO_OFFSET),
                          OpImmediate::Specified(assembleOpAttribute->GetToTensorOffset()), Matrix::CopyMode::INSERT);
        return true;
    }
    if (inputMemtype == MemoryType::MEM_L0C && outputMemtype == MemoryType::MEM_UB) {
        SetOpcodeByMemPath(op, inputMemtype, outputMemtype);
        SetL0C2UBCopyAttr(op, op.GetIOperands()[0]->GetShape(), OpImmediate::Specified(ZERO_OFFSET),
                          OpImmediate::Specified(assembleOpAttribute->GetToTensorOffset()), Matrix::CopyMode::INSERT);
        return true;
    }
    if (inputMemtype == MemoryType::MEM_UB && outputMemtype == MemoryType::MEM_L1) {
        SetOpcodeByMemPath(op, inputMemtype, outputMemtype);
        ProcessUB2L1(function, op); // 先进行 ND2NZ 转换
        SetUB2L1CopyAttr(op, op.GetIOperands()[0]->GetShape(), OpImmediate::Specified(ZERO_OFFSET),
                         OpImmediate::Specified(assembleOpAttribute->GetToTensorOffset()), Matrix::CopyMode::INSERT);
        return true;
    }
    return false;
}

Status GenerateMoveOp::CreateMoveOpForAssemble(Function& function, Operation& op) const
{
    auto assembleOpAttribute = dynamic_cast<AssembleOpAttribute*>(op.GetOpAttribute().get());
    if (op.GetIOperands().empty() || op.GetOOperands().empty()) {
        return SUCCESS;
    }
    auto assembleInput = op.iOperand.front();
    if (assembleOpAttribute == nullptr || assembleInput == nullptr || op.oOperand.front() == nullptr) {
        return SUCCESS;
    }
    auto inputMemtype = assembleInput->GetMemoryTypeOriginal();
    auto outputMemtype = op.oOperand.front()->GetMemoryTypeOriginal();
    auto parentOp = assembleInput->GetProducers().empty() ? nullptr : *assembleInput->GetProducers().begin();

    // INSERT 模式：L0C→L1 / L0C→UB / UB→L1
    if (TryInsertModeDispatch(function, op, assembleOpAttribute, inputMemtype, outputMemtype)) {
        return SUCCESS;
    }
    assembleInput = op.iOperand.front();
    // 跳过：DDR 输入 / 非 DDR 输出 / 特殊父 op
    if (inputMemtype == MemoryType::MEM_DEVICE_DDR || outputMemtype != MemoryType::MEM_DEVICE_DDR ||
        (parentOp != nullptr && (parentOp->GetOpcode() == Opcode::OP_TRANSPOSE_MOVEOUT ||
                                 parentOp->GetOpcode() == Opcode::OP_INDEX_OUTCAST))) {
        return SUCCESS;
    }
    // DDR 输出路径：Conv 变体（L0C→DDR + Conv producer）或标准 COPY_OUT
    if (inputMemtype == MemoryType::MEM_L0C && HasConvProducer(op)) {
        return ProcessConvAssemble(op, assembleOpAttribute);
    }
    op.SetOpCode(Opcode::OP_COPY_OUT);
    if (assembleOpAttribute->GetFrom() != assembleInput->GetMemoryTypeOriginal()) {
        APASS_LOG_WARN_F(Elements::Operation,
                         "Assemble op from Attr is different from iOperand, opmagic: %d, do force setting.",
                         op.opmagic);
    }
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        assembleInput->GetMemoryTypeOriginal(), OpImmediate::Specified(assembleOpAttribute->GetToTensorOffset()),
        OpImmediate::Specified(op.iOperand.front()->shape),
        OpImmediate::Specified(op.oOperand.front()->tensor->GetDynRawShape()),
        OpImmediate::Specified(op.iOperand.front()->GetDynValidShape())));
    return SUCCESS;
}

Status GenerateMoveOp::CreateMoveOpForConvert(Function& function, Operation& op) const
{
    auto convertOpAttribute = dynamic_cast<ConvertOpAttribute*>(op.GetOpAttribute().get());
    auto [from, to] = convertOpAttribute->GetConvertPath();
    Status status = SetOpcodeByMemPath(op, from, to);
    if (op.GetOpcode() == Opcode::OP_UB_COPY_L1) {
        ProcessUB2L1(function, op);
        SetUB2L1CopyAttr(op, op.GetOOperands()[0]->GetShape(), OpImmediate::Specified(ZERO_OFFSET),
                         OpImmediate::Specified(ZERO_OFFSET), Matrix::CopyMode::MOVE);
    }
    if (op.GetOpcode() == Opcode::OP_L0C_TO_L1) {
        SetL0C2L1CopyAttr(op, op.GetOOperands()[0]->GetShape(), OpImmediate::Specified(ZERO_OFFSET),
                          OpImmediate::Specified(ZERO_OFFSET), Matrix::CopyMode::MOVE);
    }
    if (op.GetOpcode() == Opcode::OP_L0C_COPY_UB) {
        SetL0C2UBCopyAttr(op, op.GetOOperands()[0]->GetShape(), OpImmediate::Specified(ZERO_OFFSET),
                          OpImmediate::Specified(ZERO_OFFSET), Matrix::CopyMode::MOVE);
        op.SetAttribute(OpAttributeKey::isCube, true);
    }
    if (status != SUCCESS) {
        return status;
    }
    auto childOp = *op.oOperand.front()->GetConsumers().begin();
    op.UpdateSubgraphID(childOp->GetSubgraphID());
    return SUCCESS;
}

void GenerateMoveOp::ProcessUB2L1(Function& function, Operation& op) const
{
    op.SetAttribute(OpAttributeKey::isCube, false);
    // GEMV A 保持 ND：跳过 ND2NZ 插入，由 codegen 生成 TCopyUB2L1<mode, isGemv> 走 ND 分支
    int64_t isGemv = 0;
    if (op.GetAttr<int64_t>(OpAttributeKey::isGemv, isGemv) && isGemv != 0) {
        return;
    }
    // UB2L1 pattern 插入的 CONVERT 未继承下游 GEMV 标记：从 L1 侧 consumer（L1_TO_L0A view）继承，
    // 保证 GEMV 场景生成 ND 布局的 TCopyUB2L1ND2ND，避免 L1 NZ 布局与 TExtractL1ToL0ND2ND 冲突。
    for (auto* consumer : op.oOperand.front()->GetConsumers()) {
        if (consumer == nullptr) {
            continue;
        }
        int64_t consumerIsGemv = 0;
        if (consumer->GetAttr<int64_t>(OpAttributeKey::isGemv, consumerIsGemv) && consumerIsGemv != 0) {
            op.SetAttribute(OpAttributeKey::isGemv, static_cast<int64_t>(1));
            return;
        }
    }
    auto inputTensor = op.iOperand.front();
    // 插入UB2L1节点（NZ2NZ)，并设置UBcopyL1的NZ属性
    op.SetAttribute(OP_ATTR_PREFIX + "is_nz", 1);
    if (inputTensor->Format() == TileOpFormat::TILEOP_ND) {
        // 新建一块logcialtensor
        std::shared_ptr<LogicalTensor> ubNdTensor = inputTensor;
        std::shared_ptr<RawTensor> newRawTensor = std::make_shared<RawTensor>(
            ubNdTensor->Datatype(), ubNdTensor->GetShape(), TileOpFormat::TILEOP_NZ);
        std::vector<int64_t> newoffset(inputTensor->GetShape().size(), 0);
        IRBuilder builder;
        std::shared_ptr<LogicalTensor> ubNzTensor = builder.CreateTensorVar(newRawTensor, newoffset, inputTensor->shape,
                                                                            inputTensor->GetDynValidShape());
        ubNzTensor->SetMemoryTypeBoth(MemoryType::MEM_UB);
        // 插入UB2UB节点（ND2NZ)
        auto& ub2ub = builder.CreateTensorOpStmt(function, Opcode::OP_UB_COPY_ND2NZ, {inputTensor}, {ubNzTensor});
        ub2ub.SetSpan(op.GetSpan());
        ub2ub.SetScopeInfo(op.GetScopeInfo());
        ub2ub.UpdateSubgraphID(op.GetSubgraphID());

        // 图重连
        op.iOperand = {ubNzTensor};
        inputTensor->RemoveConsumer(op);
        ubNzTensor->AddConsumer(op);
    }
}

Status GenerateMoveOp::CreateMoveOp(Function& function) const
{
    const Opcode sliceOpcode = config::GetSliceOpcode();
    const Opcode contractOpcode = config::GetContractOpcode();
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE_SSA || op.GetOpcode() == contractOpcode) {
            CreateMoveOpForAssemble(function, op);
            continue;
        }
        if (op.GetOpcode() == sliceOpcode) {
            Status status = ProcessViewOp(function, op);
            if (status != SUCCESS) {
                return status;
            }
            continue;
        }
        switch (op.GetOpcode()) {
            case Opcode::OP_CONVERT: {
                Status createMoveOpForConvert = CreateMoveOpForConvert(function, op);
                if (createMoveOpForConvert != SUCCESS) {
                    return createMoveOpForConvert;
                }
                break;
            }
            case Opcode::OP_DUPLICATE: {
                Status status = ProcessDuplicateOp(op);
                if (status != SUCCESS) {
                    return status;
                }
                break;
            }
            case Opcode::OP_L1_COPY_IN_CONV: {
                Status status = ProcessL1CopyInConv(op);
                if (status != SUCCESS) {
                    return status;
                }
                break;
            }
            case Opcode::OP_L0C_COPY_OUT_CONV: {
                Status status = ProcessL0CCopyOutConv(op);
                if (status != SUCCESS) {
                    return status;
                }
                break;
            }
            default:
                break;
        }
    }
    function.EraseOperations(false, true, SortOperationsMode::LIGHTWEIGHT);
    return SUCCESS;
}

Status GenerateMoveOp::ProcessL1CopyInConv(Operation& op) const
{
    // 1. 获取 L1_COPY_IN_CONV 的 producer view-family op
    auto inputTensor = op.GetIOperands()[0];
    const auto& producers = inputTensor->GetProducers();
    if (producers.empty()) {
        return SUCCESS;
    }

    auto producerOp = *producers.begin();
    if (producerOp->GetOpcode() != config::GetSliceOpcode()) {
        return SUCCESS;
    }

    // 2. 获取 view-family op 的 fromOffset 属性
    auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(producerOp->GetOpAttribute());
    if (viewAttr == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "L1_COPY_IN_CONV op[%d]: %s producer[%d] has null ViewOpAttribute.",
                          op.GetOpMagic(), producerOp->GetOpcodeStr().c_str(), producerOp->GetOpMagic());
        return FAILED;
    }

    // 3. 将 view-family op 的 fromOffset 累加到 L1_COPY_IN_CONV 的 CopyOpAttribute 的 fromOffset
    auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
    if (copyAttr == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "L1_COPY_IN_CONV op[%d]: CopyOpAttribute is null.", op.GetOpMagic());
        return FAILED;
    }
    std::vector<OpImmediate> curFromOffset = copyAttr->GetFromOffset();

    // 如果当前 offset 为空，直接使用 view-family op 的 offset
    if (curFromOffset.empty()) {
        copyAttr->SetFromOffset(OpImmediate::Specified(viewAttr->GetFromTensorOffset()));
    } else {
        // 使用 TensorOffset::Add 进行累加
        std::vector<SymbolicScalar> curFromOffsetScalar = OpImmediate::ToSpecified(curFromOffset);
        std::vector<SymbolicScalar> viewOffsetScalar = OpImmediate::ToSpecified(
            OpImmediate::Specified(TensorOffset(viewAttr->GetFromOffset(), viewAttr->GetFromDynOffset())));

        // 尺寸检查
        if (curFromOffsetScalar.size() == viewOffsetScalar.size()) {
            auto ret = TensorOffset::Add(viewOffsetScalar, curFromOffsetScalar);
            copyAttr->SetFromOffset(OpImmediate::Specified(ret));
        } else {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "L1_COPY_IN_CONV op[%d]: fromOffset size mismatch, cur size=%zu, view size=%zu.",
                              op.GetOpMagic(), curFromOffsetScalar.size(), viewOffsetScalar.size());
            return FAILED;
        }
    }
    auto viewInput = producerOp->GetIOperands().front();
    copyAttr->SetRawShape(OpImmediate::Specified(viewInput->GetRawTensor()->GetDynRawShape()));
    op.SetOpAttribute(copyAttr);

    // 4. 标记删除 view-family op
    op.ReplaceIOperand(0, viewInput);
    producerOp->SetAsDeleted();
    return SUCCESS;
}

Status GenerateMoveOp::ProcessL0CCopyOutConv(Operation& op) const
{
    // 1. 获取 L0C_COPY_OUT_CONV 的 consumer assemble-family op
    auto outputTensor = op.GetOOperands()[0];
    const auto& consumers = outputTensor->GetConsumers();
    if (consumers.empty()) {
        return SUCCESS;
    }

    auto consumerOp = *consumers.begin();
    if (consumerOp->GetOpcode() != config::GetContractOpcode()) {
        return SUCCESS;
    }

    // 2. 获取 assemble-family op 的 toOffset 属性
    auto assembleAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(consumerOp->GetOpAttribute());
    if (assembleAttr == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "L0C_COPY_OUT_CONV op[%d]: %s consumer[%d] has null AssembleOpAttribute.", op.GetOpMagic(),
                          consumerOp->GetOpcodeStr().c_str(), consumerOp->GetOpMagic());
        return FAILED;
    }

    // 3. 将 assemble-family op 的 toOffset 累加到 L0C_COPY_OUT_CONV 的 toOffset
    auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
    if (copyAttr == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "L0C_COPY_OUT_CONV op[%d]: CopyOpAttribute is null.", op.GetOpMagic());
        return FAILED;
    }
    std::vector<OpImmediate> curToOffset = copyAttr->GetToOffset();

    // 如果当前 offset 为空，直接使用 assemble-family op 的 offset
    if (curToOffset.empty()) {
        copyAttr->SetToOffset(
            OpImmediate::Specified(TensorOffset(assembleAttr->GetToOffset(), assembleAttr->GetToDynOffset())));
    } else {
        // 使用 TensorOffset::Add 进行累加
        std::vector<SymbolicScalar> curToOffsetScalar = OpImmediate::ToSpecified(curToOffset);
        std::vector<SymbolicScalar> assembleOffsetScalar = OpImmediate::ToSpecified(
            OpImmediate::Specified(TensorOffset(assembleAttr->GetToOffset(), assembleAttr->GetToDynOffset())));

        // 尺寸检查
        if (curToOffsetScalar.size() == assembleOffsetScalar.size()) {
            auto ret = TensorOffset::Add(assembleOffsetScalar, curToOffsetScalar);
            copyAttr->SetToOffset(OpImmediate::Specified(ret));
        } else {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "L0C_COPY_OUT_CONV op[%d]: toOffset size mismatch, cur size=%zu, assemble size=%zu.",
                              op.GetOpMagic(), curToOffsetScalar.size(), assembleOffsetScalar.size());
            return FAILED;
        }
    }
    auto assembleOutput = consumerOp->GetOOperands().front();
    copyAttr->SetRawShape(OpImmediate::Specified(assembleOutput->GetRawTensor()->GetDynRawShape()));
    op.SetOpAttribute(copyAttr);

    // 4. 标记删除 assemble-family op
    op.ReplaceOOperand(0, assembleOutput);
    consumerOp->SetAsDeleted();
    return SUCCESS;
}

Status GenerateMoveOp::ProcessViewOp(Function& function, Operation& op) const
{
    if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) {
        Status status = A5CreateMoveOpForView(function, op);
        if (status != SUCCESS) {
            return status;
        }
    } else {
        Status status = A23CreateMoveOpForView(function, op);
        if (status != SUCCESS) {
            return status;
        }
    }
    return SUCCESS;
}

Status GenerateMoveOp::ProcessDuplicateOp(Operation& op) const
{
    IRBuilder builder;
    op.SetOpCode(Opcode::OP_COPY_OUT);
    std::vector<OpImmediate> newOffset;
    for (size_t i = 0; i < op.iOperand.front()->shape.size(); i++) {
        newOffset.push_back(OpImmediate::Specified(builder.CreateConstInt(0)));
    }
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        op.iOperand.front()->GetMemoryTypeOriginal(), newOffset, OpImmediate::Specified(op.iOperand.front()->shape),
        OpImmediate::Specified(op.oOperand.front()->tensor->GetDynRawShape())));
    return SUCCESS;
}
} // namespace npu::tile_fwk
