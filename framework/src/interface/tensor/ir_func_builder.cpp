/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ir_func_builder.h"

#include "logical_tensor.h"
#include "raw_tensor.h"
#include "interface/function/function.h"
#include "interface/operation/attribute.h"
#include "interface/operation/operation.h"
#include "interface/operation/opcode.h"
#include "interface/program/program.h"
#include "interface/tensor/tensor_slot.h"
#include "tilefwk/tensor.h"
#include "interface/utils/id_gen.h"
#include "interface/configs/config_manager.h"
#include "interface/configs/config_manager_ng.h"
#include "interface/tensor/irbuilder.h"
#include "interface/tensor/ir_tensor_op_rebuild.h"

#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/type.h"
#include "ir/transforms/base/mutator.h"

using namespace pypto;

namespace npu::tile_fwk {

class StmtTransformer : public ir::IRMutator {
public:
    using IRMutator::VisitExpr_;
    using IRMutator::VisitStmt;
    using IRMutator::VisitStmt_;

    explicit StmtTransformer(RootFunctionBuilder& builder) : builder_(builder) {}

    ir::StmtPtr Apply(ir::StmtPtr stmt, const std::string& loopVarName)
    {
        loopVarName_ = loopVarName;
        return VisitStmt(stmt);
    }

private:
    RootFunctionBuilder& builder_;
    std::string loopVarName_;

    void CollectConsumedTensors(const std::vector<ir::ExprPtr>& values)
    {
        for (auto& value : values) {
            if (!value) {
                continue;
            }
            auto lt = AsLogicalTensor(value);
            if (lt != nullptr) {
                builder_.consumedTensors_.insert(lt);
            }
        }
    }

    ir::StmtPtr VisitStmt_(const ir::SeqStmtsPtr& seq) override
    {
        if (builder_.IsPureTensorOpSeq(seq)) {
            auto newStmts = builder_.CreatePathFuncAndPlaceholder(seq, loopVarName_);
            return std::make_shared<ir::SeqStmts>(std::vector<ir::StmtPtr>{newStmts}, seq->span_);
        }
        auto segments = builder_.SplitIntoTensorOpSegments(seq);
        std::vector<ir::StmtPtr> newStmts;
        for (auto& segment : segments) {
            if (!segment.empty() && segment[0]->GetKind() == ir::ObjectKind::TensorOpStmt) {
                auto segSeq = std::make_shared<ir::SeqStmts>(segment, seq->span_);
                newStmts.push_back(builder_.CreatePathFuncAndPlaceholder(segSeq, loopVarName_));
            } else if (!segment.empty()) {
                newStmts.push_back(VisitStmt(segment[0]));
            }
        }
        return std::make_shared<ir::SeqStmts>(newStmts, seq->span_);
    }

    ir::StmtPtr VisitStmt_(const ir::ForStmtPtr& forStmt) override
    {
        for (auto& iterArg : forStmt->iterArgs_) {
            if (iterArg && iterArg->initValue_) {
                auto lt = AsLogicalTensor(iterArg->initValue_);
                if (lt != nullptr) {
                    builder_.consumedTensors_.insert(lt);
                }
            }
        }
        auto savedLoopVarName = loopVarName_;
        loopVarName_ = IRContext::Get().GetOriginName(forStmt->loopVar_);
        int unrollTimes = forStmt->GetAttr<int>("unroll_times", 1);
        loopVarName_ += "_Unroll" + std::to_string(unrollTimes);
        auto transformedBody = VisitStmt(forStmt->body_);
        loopVarName_ = savedLoopVarName;
        return std::make_shared<ir::ForStmt>(forStmt->loopVar_, forStmt->start_, forStmt->stop_, forStmt->step_,
                                             forStmt->iterArgs_, transformedBody, forStmt->returnVars_, forStmt->span_,
                                             forStmt->attrs_);
    }

    ir::StmtPtr VisitStmt_(const ir::ReturnStmtPtr& op) override
    {
        CollectConsumedTensors(op->value_);
        return op;
    }

    ir::StmtPtr VisitStmt_(const ir::ContinueStmtPtr& op) override
    {
        CollectConsumedTensors(op->value_);
        return op;
    }

    ir::StmtPtr VisitStmt_(const ir::YieldStmtPtr& op) override
    {
        CollectConsumedTensors(op->value_);
        return op;
    }

    ir::StmtPtr VisitStmt_(const ir::IfStmtPtr& ifStmt) override
    {
        auto transformedThen = VisitStmt(ifStmt->thenBody_);
        std::optional<ir::StmtPtr> transformedElse;
        if (ifStmt->elseBody_) {
            transformedElse = VisitStmt(ifStmt->elseBody_.value());
        }
        return std::make_shared<ir::IfStmt>(ifStmt->condition_, transformedThen, transformedElse, ifStmt->returnVars_,
                                            ifStmt->span_);
    }
};

RootFunctionBuilder::RootFunctionBuilder(Function* parentFunc)
    : program_(Program::GetInstance()), parentFunc_(parentFunc)
{}

std::shared_ptr<Function> RootFunctionBuilder::Build(const ir::FunctionPtr& irFunc)
{
    InitDynFunc(irFunc);
    auto stmtsWithCall = TransformBody(irFunc->body_);
    dynFunc_->body_ = ir::SeqStmts::Wrap(stmtsWithCall, irFunc->span_);
    FinalizeDynFunc(irFunc);
    DumpFunctionGraph(dynFunc_.get());
    return dynFunc_;
}

void RootFunctionBuilder::InitDynFunc(const ir::FunctionPtr& irFunc)
{
    auto funcMagicName = irFunc->name_ + "_" + std::to_string(Function::NextMagicNameSuffix(FunctionType::DYNAMIC));
    dynFunc_ = std::make_shared<Function>(program_, funcMagicName, irFunc->name_, parentFunc_);
    dynFunc_->SetFunctionType(FunctionType::DYNAMIC);
    dynFunc_->SetGraphType(GraphType::TENSOR_GRAPH);
    dynFunc_->SetSpan(irFunc->span_);
    dynFunc_->SetDyndevAttribute(std::make_shared<DyndevFunctionAttribute>());
    auto attr = dynFunc_->GetDyndevAttribute();
    auto slotManager = program_.GetTensorSlotManager();

    for (const auto& param : irFunc->params_) {
        auto lt = AsLogicalTensor(param);
        ASSERT(lt) << "RootFunctionBuilder: param is not a LogicalTensor: " << param->name_;
        logicalParams_.push_back(lt);
        attr->startArgsInputLogicalTensorList.push_back(lt);
        auto tensor = slotManager->GetSlotTensor(lt);
        attr->startArgsInputSlotTensorList.emplace_back(tensor);
        attr->startArgsInputTensorList.emplace_back(*tensor);
        slotManager->MarkInput(*tensor);
    }
    program_.SetCurrentDynamicFunction(dynFunc_.get());
}

void RootFunctionBuilder::CollectInOutCastFromReturn()
{
    auto slotManager = program_.GetTensorSlotManager();

    std::unordered_set<LogicalTensorPtr> callInputTensors;
    std::unordered_set<int> callOutputSlots;
    for (auto& op : dynFunc_->Operations(false).DuplicatedOpList()) {
        for (auto& iOperand : op->iOperand) {
            callInputTensors.insert(iOperand);
        }
        for (auto& oOperand : op->oOperand) {
            callOutputSlots.insert(slotManager->GetSlotTensor(oOperand)->Id());
        }
    }

    for (auto& param : logicalParams_) {
        if (callInputTensors.count(param) > 0) {
            dynFunc_->AddOriginIncast(param);
        }
        if (callOutputSlots.count(slotManager->GetSlotTensor(param)->Id()) > 0) {
            dynFunc_->AddOriginOutcast(param);
        }
    }
}

void RootFunctionBuilder::FinalizeDynFunc(const ir::FunctionPtr& irFunc)
{
    dynFunc_->name_ = irFunc->name_;
    dynFunc_->funcType_ = ir::FunctionType::ORCHESTRATION;

    auto dynScope = std::make_shared<TensorSlotScope>(dynFunc_.get());
    dynFunc_->SetSlotScope(dynScope);
    program_.GetTensorSlotManager()->scopeList.push_back(dynScope);

    CollectInOutCastFromReturn();

    for (auto& incast : dynFunc_->GetOriginIncast()) {
        dynFunc_->AppendIncast(incast, 0, 0);
    }
    for (auto& outcast : dynFunc_->GetOriginOutcast()) {
        dynFunc_->AppendOutcast(outcast, 0, 0);
    }

    for (auto& param : logicalParams_) {
        dynFunc_->params_.push_back(std::static_pointer_cast<const ir::Var>(param));
    }
    dynFunc_->ComputeHash();
    program_.SetParamConfig(dynFunc_.get(), ConfigManagerNg::CurrentScope());

    FlushConstructAssembleSlots();
    BuildDynSlotScope();
}

bool RootFunctionBuilder::IsPureTensorOpSeq(const ir::SeqStmtsPtr& seq)
{
    if (!seq || seq->stmts_.empty()) {
        return false;
    }
    for (auto& stmt : seq->stmts_) {
        if (stmt->GetKind() != ir::ObjectKind::TensorOpStmt) {
            return false;
        }
    }
    return true;
}

std::vector<std::vector<ir::StmtPtr>> RootFunctionBuilder::SplitIntoTensorOpSegments(const ir::SeqStmtsPtr& seq)
{
    std::vector<std::vector<ir::StmtPtr>> segments;
    std::vector<ir::StmtPtr> currentRun;
    for (auto& child : seq->stmts_) {
        if (child->GetKind() == ir::ObjectKind::TensorOpStmt) {
            currentRun.push_back(child);
        } else {
            if (!currentRun.empty()) {
                segments.push_back(currentRun);
                currentRun.clear();
            }
            segments.push_back({child});
        }
    }
    if (!currentRun.empty()) {
        segments.push_back(currentRun);
    }
    return segments;
}

void RootFunctionBuilder::RecordDefinedTensors(const ir::TensorOpStmtPtr& tensorOpStmt, Function* pathFunc)
{
    auto sourceOp = std::dynamic_pointer_cast<const Operation>(tensorOpStmt);
    bool isAssembleLike = sourceOp != nullptr &&
                          ((sourceOp->GetOpcode() == Opcode::OP_ASSEMBLE && sourceOp->HasAttr("dassemble")) ||
                           sourceOp->GetOpcode() == Opcode::OP_ASSEMBLE_SSA ||
                           sourceOp->GetOpcode() == Opcode::OP_ATOMIC_RMW);
    for (auto& result : tensorOpStmt->result_) {
        if (!result) {
            continue;
        }
        auto lt = AsLogicalTensor(result);
        if (tensorDefineFunc_.find(lt) == tensorDefineFunc_.end()) {
            tensorDefineFunc_[lt] = pathFunc;
        }
        if (!isAssembleLike && tensorConstructFunc_.find(lt) == tensorConstructFunc_.end()) {
            tensorConstructFunc_[lt] = pathFunc;
        }
    }
}

void RootFunctionBuilder::CollectAssembleConsumed(const std::shared_ptr<const Operation>& sourceOp,
                                                  const LogicalTensors& oOperands, Function* pathFunc)
{
    if (sourceOp == nullptr) {
        return;
    }
    bool isAssembleLike = (sourceOp->GetOpcode() == Opcode::OP_ASSEMBLE && sourceOp->HasAttr("dassemble")) ||
                          sourceOp->GetOpcode() == Opcode::OP_ASSEMBLE_SSA ||
                          sourceOp->GetOpcode() == Opcode::OP_ATOMIC_RMW;
    if (isAssembleLike) {
        for (auto& op : oOperands) {
            auto it = tensorDefineFunc_.find(op);
            if (it == tensorDefineFunc_.end() || it->second != pathFunc) {
                consumedTensors_.insert(op);
            }
        }
    }
}

ir::StmtPtr RootFunctionBuilder::ProcessTensorOp(std::shared_ptr<Function> pathFunc, const ir::StmtPtr& stmt,
                                                 std::unordered_set<std::shared_ptr<LogicalTensor>>& allInputs,
                                                 std::unordered_set<std::shared_ptr<LogicalTensor>>& definedOutputs,
                                                 std::unordered_map<int, std::shared_ptr<RawTensor>>& memIdMap)
{
    auto tensorOpStmt = std::dynamic_pointer_cast<const ir::TensorOpStmt>(stmt);
    if (!tensorOpStmt) {
        return stmt;
    }

    RecordDefinedTensors(tensorOpStmt, pathFunc.get());
    if (tensorOpStmt->opcode_ == "TENSOR_ALLOC") {
        return stmt;
    }

    LogicalTensors iOperands;
    for (auto& arg : tensorOpStmt->args_) {
        if (!arg) {
            continue;
        }
        auto lt = AsLogicalTensor(arg);
        iOperands.push_back(lt);
        allInputs.insert(lt);
        memIdMap.try_emplace(lt->tensor->memoryId, lt->tensor);
    }

    LogicalTensors oOperands;
    for (auto& result : tensorOpStmt->result_) {
        if (!result) {
            continue;
        }
        auto lt = AsLogicalTensor(result);
        oOperands.push_back(lt);
        definedOutputs.insert(lt);
        auto it = memIdMap.find(lt->tensor->memoryId);
        if (it != memIdMap.end()) {
            pathFunc->outIncastLinkMap[lt->tensor] = it->second;
        }
    }

    auto sourceOp = std::dynamic_pointer_cast<const Operation>(tensorOpStmt);
    CollectAssembleConsumed(sourceOp, oOperands, pathFunc.get());

    ir::StmtPtr opStmt = RebuildTensorOpStmt(tensorOpStmt, tensorOpStmt->result_, tensorOpStmt->result_token_,
                                             tensorOpStmt->args_, tensorOpStmt->tokens_, tensorOpStmt->span_,
                                             pathFunc.get());
    for (const auto& token : tensorOpStmt->result_token_) {
        pathFunc->GetVarDependency().AddProducer(token, opStmt);
    }
    for (auto& token : tensorOpStmt->tokens_) {
        pathFunc->GetVarDependency().AddConsumer(token, opStmt);
    }
    return opStmt;
}

void RootFunctionBuilder::ComputeIncast(Function& pathFunc,
                                        const std::unordered_set<std::shared_ptr<LogicalTensor>>& allInputs,
                                        const std::unordered_set<std::shared_ptr<LogicalTensor>>& definedOutputs)
{
    std::vector<std::shared_ptr<LogicalTensor>> sortedInputs(allInputs.begin(), allInputs.end());
    std::sort(sortedInputs.begin(), sortedInputs.end(),
              [](const std::shared_ptr<LogicalTensor>& lhs, const std::shared_ptr<LogicalTensor>& rhs) {
                  return lhs->GetMagic() < rhs->GetMagic();
              });
    std::unordered_set<std::shared_ptr<LogicalTensor>> incastPtrs;
    for (auto& input : sortedInputs) {
        if (definedOutputs.find(input) == definedOutputs.end() && incastPtrs.find(input) == incastPtrs.end()) {
            incastPtrs.insert(input);
            pathFunc.AddOriginIncast(input);
        }
    }
}

void RootFunctionBuilder::ComputeOutcast(Function& pathFunc)
{
    auto isQualified = [this](const LogicalTensorPtr& tensor) {
        return consumedTensors_.count(tensor) > 0 || paramTensors_.count(tensor) > 0 ||
               tensor->IsGetTensorDataOutcast();
    };

    std::unordered_set<int> rawMagics;
    for (auto& op : pathFunc.Operations(false)) {
        for (auto& oOperand : op.GetOOperands()) {
            if (isQualified(oOperand)) {
                rawMagics.insert(oOperand->GetRawMagic());
            }
        }
    }

    for (auto& op : pathFunc.Operations(false)) {
        for (auto& oOperand : op.GetOOperands()) {
            if (rawMagics.count(oOperand->GetRawMagic()) > 0) {
                pathFunc.AddOriginOutcast(oOperand);
            }
        }
    }
}

std::shared_ptr<Function> RootFunctionBuilder::CreateHiddenFunc(const ir::SeqStmtsPtr& seq,
                                                                const std::string& loopVarName)
{
    auto pathFuncId = loopNameCounters_[loopVarName]++;
    std::string pathSuffix = loopVarName.empty() ? "PATH" + std::to_string(pathFuncId) :
                                                   loopVarName + "_PATH" + std::to_string(pathFuncId);
    auto hiddenRawName = dynFunc_->GetRawName() + "_" + pathSuffix + "_hiddenfunc";
    auto hiddenMagicName = hiddenRawName + "_" + std::to_string(IdGen<IdType::FUNCTION>::Inst().NewId());
    auto hiddenFunc = std::make_shared<Function>(program_, hiddenMagicName, hiddenRawName, dynFunc_.get());
    hiddenFunc->SetSpan(seq->span_);
    hiddenFunc->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    hiddenFunc->SetGraphType(GraphType::TENSOR_GRAPH);
    hiddenFunc->SetUnderDynamicFunction(true);
    program_.InsertFuncToFunctionMap(hiddenMagicName, hiddenFunc);

    std::unordered_set<std::shared_ptr<LogicalTensor>> definedOutputs;
    std::unordered_set<std::shared_ptr<LogicalTensor>> allInputs;
    std::unordered_map<int, std::shared_ptr<RawTensor>> memIdMap;
    for (auto& stmt : seq->stmts_) {
        (void)ProcessTensorOp(hiddenFunc, stmt, allInputs, definedOutputs, memIdMap);
    }
    ComputeIncast(*hiddenFunc, allInputs, definedOutputs);
    hiddenFunc->name_ = hiddenMagicName;
    hiddenFunc->funcType_ = ir::FunctionType::IN_CORE;
    return hiddenFunc;
}

void RootFunctionBuilder::BuildPathFuncSlotScope(Function* pathFunc, const std::shared_ptr<TensorSlotScope>& scope,
                                                 const LogicalTensors& originalIncasts,
                                                 const LogicalTensors& originalOutcasts)
{
    auto slotManager = program_.GetTensorSlotManager();

    scope->ioslot.incastSlot.resize(originalIncasts.size());
    for (size_t idx = 0; idx < originalIncasts.size(); idx++) {
        auto tensor = slotManager->GetSlotTensor(originalIncasts[idx]);
        scope->ioslot.incastSlot[idx] = {tensor->Id()};
    }

    std::unordered_map<std::shared_ptr<RawTensor>, LogicalTensorPtr> outcastByRawTensor;
    scope->ioslot.outcastSlot.resize(originalOutcasts.size());
    for (size_t idx = 0; idx < originalOutcasts.size(); idx++) {
        auto tensor = slotManager->GetSlotTensor(originalOutcasts[idx]);
        scope->ioslot.outcastSlot[idx] = {tensor->Id()};
        outcastByRawTensor.emplace(originalOutcasts[idx]->GetRawTensor(), originalOutcasts[idx]);
    }

    for (auto& op : pathFunc->Operations(false)) {
        if ((op.GetOpcode() == Opcode::OP_ASSEMBLE && op.HasAttr("dassemble")) ||
            op.GetOpcode() == Opcode::OP_ASSEMBLE_SSA || op.GetOpcode() == Opcode::OP_ATOMIC_RMW) {
            for (auto& oOperand : op.GetOOperands()) {
                // Internal assembles use root-local storage. Only boundary outcasts need
                // runtime slots, and all SSA versions of an outcast share its slot.
                auto outcast = outcastByRawTensor.find(oOperand->GetRawTensor());
                if (outcast == outcastByRawTensor.end()) {
                    continue;
                }
                auto tensor = slotManager->GetSlotTensor(outcast->second);
                slotManager->TensorWrite(*tensor, SlotProperty::ASSEMBLE_DST);
            }
        }
    }
}

void RootFunctionBuilder::BuildDynSlotScope()
{
    auto attr = dynFunc_->GetDyndevAttribute();
    FE_ASSERT(FeError::INVALID_PTR, attr != nullptr) << "DyndevAttribute is nullptr";

    auto slotManager = program_.GetTensorSlotManager();
    auto& dynScope = dynFunc_->GetSlotScope();

    dynScope->ioslot.incastSlot.resize(dynFunc_->GetIncast().size());
    for (size_t idx = 0; idx < dynFunc_->GetIncast().size(); idx++) {
        auto tensor = slotManager->GetSlotTensor(dynFunc_->GetIncast()[idx]);
        dynScope->ioslot.incastSlot[idx] = {tensor->Id()};
    }
    dynScope->ioslot.outcastSlot.resize(dynFunc_->GetOutcast().size());
    for (size_t idx = 0; idx < dynFunc_->GetOutcast().size(); idx++) {
        auto tensor = slotManager->GetSlotTensor(dynFunc_->GetOutcast()[idx]);
        dynScope->ioslot.outcastSlot[idx] = {tensor->Id()};
    }

    for (size_t idx = 0; idx < logicalParams_.size(); idx++) {
        auto tensor = slotManager->GetSlotTensor(logicalParams_[idx]);
        slotManager->UpdateInputSlot(idx, *tensor);
        attr->startArgsInputTensorList[idx] = *tensor;
        attr->startArgsInputSlotTensorList[idx] = tensor;
    }

    dynFunc_->InferParamDirection();
}

bool RootFunctionBuilder::IsPlaceholderCallStmt(const ir::StmtPtr& stmt)
{
    if (stmt->GetKind() != ir::ObjectKind::TensorOpStmt) {
        return false;
    }
    auto tensorOp = std::static_pointer_cast<const ir::TensorOpStmt>(stmt);
    if (tensorOp->opcode_ != "CALL") {
        return false;
    }
    for (auto& attr : tensorOp->attrs_) {
        if (attr.first == "placeholder_funcname") {
            return true;
        }
    }
    return false;
}

std::string RootFunctionBuilder::GetPlaceholderFuncname(const ir::StmtPtr& stmt)
{
    auto tensorOp = std::static_pointer_cast<const ir::TensorOpStmt>(stmt);
    for (auto& [key, value] : tensorOp->attrs_) {
        if (key == "placeholder_funcname") {
            return std::any_cast<std::string>(value);
        }
    }
    return "";
}

ir::StmtPtr RootFunctionBuilder::CreatePathFuncAndPlaceholder(const ir::SeqStmtsPtr& seq,
                                                              const std::string& loopVarName)
{
    auto pathFunc = CreateHiddenFunc(seq, loopVarName);

    for (auto& incast : pathFunc->GetOriginIncast()) {
        consumedTensors_.insert(incast);
    }

    auto placeholderStmt = std::make_shared<ir::TensorOpStmt>(std::vector<ir::VarPtr>{}, nullptr, "CALL",
                                                              std::vector<ir::ExprPtr>{}, std::vector<ir::VarPtr>{},
                                                              std::vector<std::pair<std::string, std::any>>{
                                                                  {"placeholder_funcname", pathFunc->GetMagicName()},
                                                              },
                                                              seq->span_);

    return placeholderStmt;
}

void RootFunctionBuilder::AddHiddenFuncValueDepend(Function* hiddenFunc)
{
    auto currDynFunc = program_.GetCurrentDynamicFunction();
    FE_ASSERT(FeError::INVALID_PTR, currDynFunc != nullptr) << "CurrentDynamicFunction is nullptr";
    auto dynAttr = currDynFunc->GetDyndevAttribute();
    FE_ASSERT(FeError::INVALID_PTR, dynAttr != nullptr) << "DyndevAttribute is nullptr";
    auto iodescDict = hiddenFunc->GetTensorDataForTensorGraph();
    hiddenFunc->GetTensorDataRefreshIO(iodescDict);
    dynAttr->valueDependDescDict[hiddenFunc] = hiddenFunc->LookupValueDepend();
}

void RootFunctionBuilder::CreateAndFinalizePathFunc(Function* pathFunc, Function* hiddenFunc,
                                                    const LogicalTensors& hiddenInArgs,
                                                    const LogicalTensors& hiddenOutArgs, const ir::StmtPtr& placeholder)
{
    // 1. pathFunc 中创建 OP_CALL 指向 hiddenFunc
    auto& callOp = pathFunc->AddRawOperation(Opcode::OP_CALL, hiddenInArgs, hiddenOutArgs, placeholder->span_);
    callOp.SetOpAttribute(hiddenFunc->CreateCallOpAttribute({}, {}));
    pathFunc->AppendCalleeMagicName(hiddenFunc->GetMagicName());
    callOp.attrs_.emplace_back("callee", hiddenFunc->GetMagicName());

    // 2. pathFunc 的 incast/outcast
    for (auto& incast : hiddenInArgs)
        pathFunc->AddOriginIncast(incast);
    for (auto& outcast : hiddenOutArgs)
        pathFunc->AddOriginOutcast(outcast);

    // 3. pathFunc 的 scope + MakeIncasts/MakeOutcasts
    auto pathScope = std::make_shared<TensorSlotScope>(pathFunc);
    pathFunc->SetSlotScope(pathScope);
    program_.GetTensorSlotManager()->scopeList.push_back(pathScope);

    BuildPathFuncSlotScope(pathFunc, pathScope, pathFunc->GetOriginIncast(), pathFunc->GetOriginOutcast());

    LogicalTensors pathInArgs = pathFunc->MakeIncasts(pathScope);
    LogicalTensors pathOutArgs = pathFunc->MakeOutcasts(pathScope);
    for (size_t idx = 0; idx < pathFunc->GetOutcast().size(); idx++) {
        if (pathScope->partialUpdateOutcastDict.count(pathFunc->GetOutcast()[idx])) {
            pathScope->ioslot.partialUpdateOutcastList.push_back(idx);
        }
    }

    AddHiddenFuncValueDepend(hiddenFunc);

    for (auto& incast : pathFunc->GetIncast())
        pathFunc->params_.push_back(std::static_pointer_cast<const ir::Var>(incast));
    for (auto& outcast : pathFunc->GetOutcast())
        pathFunc->params_.push_back(std::static_pointer_cast<const ir::Var>(outcast));

    // 5. pathFunc body_ 重建
    std::vector<ir::StmtPtr> pathBodyStmts;
    for (auto& op : pathFunc->Operations(false))
        pathBodyStmts.push_back(std::static_pointer_cast<const ir::Stmt>(op.shared_from_this()));
    pathFunc->body_ = std::make_shared<ir::SeqStmts>(std::move(pathBodyStmts), placeholder->span_);
    pathFunc->ComputeHash();
    DumpFunctionGraph(pathFunc);
    program_.GetFunctionCache().Insert(pathFunc->GetFunctionHash(), *pathFunc);
}

std::pair<LogicalTensors, LogicalTensors> RootFunctionBuilder::FinalizeHiddenFunc(Function* hiddenFunc,
                                                                                  const ir::StmtPtr& placeholder)
{
    auto hiddenScope = std::make_shared<TensorSlotScope>(hiddenFunc);
    hiddenFunc->SetSlotScope(hiddenScope);
    program_.GetTensorSlotManager()->scopeList.push_back(hiddenScope);

    BuildPathFuncSlotScope(hiddenFunc, hiddenScope, hiddenFunc->GetOriginIncast(), hiddenFunc->GetOriginOutcast());

    hiddenFunc->CreateTensorDataImportOp();
    auto hiddenInArgs = hiddenFunc->MakeIncasts(hiddenScope);
    auto hiddenOutArgs = hiddenFunc->MakeOutcasts(hiddenScope);
    for (size_t idx = 0; idx < hiddenFunc->GetOutcast().size(); idx++) {
        if (hiddenScope->partialUpdateOutcastDict.count(hiddenFunc->GetOutcast()[idx])) {
            hiddenScope->ioslot.partialUpdateOutcastList.push_back(idx);
        }
    }

    for (auto& incast : hiddenFunc->GetIncast())
        hiddenFunc->params_.push_back(std::static_pointer_cast<const ir::Var>(incast));
    for (auto& outcast : hiddenFunc->GetOutcast())
        hiddenFunc->params_.push_back(std::static_pointer_cast<const ir::Var>(outcast));

    std::vector<ir::StmtPtr> hiddenBodyStmts;
    for (auto& op : hiddenFunc->Operations(false))
        hiddenBodyStmts.push_back(std::static_pointer_cast<const ir::Stmt>(op.shared_from_this()));
    hiddenFunc->body_ = std::make_shared<ir::SeqStmts>(std::move(hiddenBodyStmts), placeholder->span_);
    program_.SetParamConfig(hiddenFunc, ConfigManagerNg::CurrentScope());
    hiddenFunc->ComputeHash();
    program_.GetFunctionCache().Insert(hiddenFunc->GetFunctionHash(), *hiddenFunc);
    DumpFunctionGraph(hiddenFunc);
    return {hiddenInArgs, hiddenOutArgs};
}

ir::StmtPtr RootFunctionBuilder::FinalizePathFunc(const ir::StmtPtr& placeholder)
{
    auto hiddenFuncName = GetPlaceholderFuncname(placeholder);
    auto hiddenFunc = program_.GetFunctionByMagicName(hiddenFuncName);
    FE_ASSERT(FeError::NOT_EXIST, hiddenFunc) << hiddenFuncName << " is not in functionmap!";

    // 1. 创建 pathFunc（在 SetParent 之前）
    auto pathRawName = hiddenFunc->GetRawName().substr(0, hiddenFunc->GetRawName().find("_hiddenfunc"));
    auto pathMagicName = pathRawName + "_" + std::to_string(IdGen<IdType::FUNCTION>::Inst().NewId());
    auto pathFunc = std::make_shared<Function>(program_, pathMagicName, pathRawName, dynFunc_.get());
    pathFunc->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    pathFunc->SetGraphType(GraphType::TENSOR_GRAPH);
    pathFunc->SetUnderDynamicFunction(true);
    program_.InsertFuncToFunctionMap(pathMagicName, pathFunc);
    pathFunc->name_ = pathMagicName;
    pathFunc->funcType_ = ir::FunctionType::IN_CORE;

    // 2. hiddenFunc 设为 hidden，parent 改为 pathFunc
    auto origFunc = program_.GetCurrentFunction();
    program_.SetCurrentFunction(hiddenFunc);
    hiddenFunc->SetHiddenFunction(true);
    hiddenFunc->SetParent(pathFunc.get());

    // 3. hiddenFunc 处理（MakeIncasts 的 Parent() = pathFunc）

    ComputeOutcast(*hiddenFunc);

    /* if two outcasts share the same raw tensor, link them onto one slot. */
    auto slotManager = program_.GetTensorSlotManager();
    std::unordered_map<int, LogicalTensorPtr> outcastByRaw;
    for (auto& outcast : hiddenFunc->GetOriginOutcast()) {
        auto [iter, inserted] = outcastByRaw.emplace(outcast->GetRawMagic(), outcast);
        if (!inserted) {
            slotManager->SetSameSlot(iter->second, outcast);
        }
    }

    auto originalIncasts = hiddenFunc->GetOriginIncast();
    auto originalOutcasts = hiddenFunc->GetOriginOutcast();
    auto hiddenFuncArgs = FinalizeHiddenFunc(hiddenFunc, placeholder);

    // 4. pathFunc 处理（独立函数）
    for (size_t i = 0; i < originalIncasts.size(); i++) {
        slotManager->SetSameSlot(originalIncasts[i], hiddenFuncArgs.first[i]);
    }
    for (size_t i = 0; i < originalOutcasts.size(); i++) {
        slotManager->SetSameSlot(originalOutcasts[i], hiddenFuncArgs.second[i]);
    }
    CreateAndFinalizePathFunc(pathFunc.get(), hiddenFunc, hiddenFuncArgs.first, hiddenFuncArgs.second, placeholder);

    program_.SetCurrentFunction(origFunc);

    // 5. dynFunc 的 CALL op
    auto& callOperation = dynFunc_->AddRawOperation(Opcode::OP_CALL, originalIncasts, originalOutcasts,
                                                    placeholder->span_);
    callOperation.SetOpAttribute(pathFunc->CreateCallOpAttribute({}, {}));
    dynFunc_->AppendCalleeMagicName(pathFunc->GetMagicName());
    callOperation.attrs_.emplace_back("callee", pathFunc->GetMagicName());
    for (auto& input : originalIncasts) {
        input->RemoveConsumer(&callOperation);
    }
    for (auto& output : originalOutcasts) {
        output->RemoveProducer(&callOperation);
    }

    return std::static_pointer_cast<const ir::Stmt>(callOperation.shared_from_this());
}

void RootFunctionBuilder::LinkReturnSlots(const ir::StmtPtr& stmt)
{
    auto seq = std::dynamic_pointer_cast<const ir::SeqStmts>(stmt);
    if (seq == nullptr || seq->stmts_.empty()) {
        return;
    }
    auto returnStmt = std::dynamic_pointer_cast<const ir::ReturnStmt>(seq->stmts_.back());
    if (returnStmt == nullptr) {
        return;
    }
    auto slotManager = program_.GetTensorSlotManager();
    for (size_t i = 0; i < returnStmt->value_.size() && i < logicalParams_.size(); i++) {
        if (!returnStmt->value_[i]) {
            continue;
        }
        auto returnLt = AsLogicalTensor(returnStmt->value_[i]);
        if (returnLt != nullptr) {
            slotManager->SetSameSlot(returnLt, logicalParams_[i]);
        }
    }
}

void RootFunctionBuilder::LinkForStmtSlots(const ir::ForStmt& forStmt)
{
    std::shared_ptr<const ir::ContinueStmt> continueStmt;
    auto seqStmts = std::dynamic_pointer_cast<const ir::SeqStmts>(forStmt.body_);
    if (seqStmts != nullptr && !seqStmts->stmts_.empty()) {
        continueStmt = std::dynamic_pointer_cast<const ir::ContinueStmt>(seqStmts->stmts_.back());
    }
    if (continueStmt != nullptr) {
        FE_ASSERT(FeError::INVALID_VAL, continueStmt->value_.size() == forStmt.iterArgs_.size())
            << "ContinueStmt value count (" << continueStmt->value_.size() << ") must match iterArgs count ("
            << forStmt.iterArgs_.size() << ")";
        FE_ASSERT(FeError::INVALID_VAL, continueStmt->value_.size() == forStmt.returnVars_.size())
            << "ContinueStmt value count (" << continueStmt->value_.size() << ") must match returnVars count ("
            << forStmt.returnVars_.size() << ")";
    }
    auto slotManager = program_.GetTensorSlotManager();
    for (size_t i = 0; i < forStmt.iterArgs_.size(); i++) {
        auto& iterArg = forStmt.iterArgs_[i];
        auto initLt = AsLogicalTensor(iterArg->initValue_);
        auto iterLt = AsLogicalTensor(iterArg->iterVar_);

        LogicalTensorPtr valueLt;
        LogicalTensorPtr returnVarLt;
        if (continueStmt != nullptr) {
            valueLt = AsLogicalTensor(continueStmt->value_[i]);
            returnVarLt = AsLogicalTensor(forStmt.returnVars_[i]);
        }

        // initLt -> iterLt -> valueLt -> returnVarLt
        LogicalTensorPtr prev;
        for (auto& cur : {initLt, iterLt, valueLt, returnVarLt}) {
            if (cur == nullptr) {
                continue;
            }
            if (prev != nullptr) {
                slotManager->SetSameSlot(prev, cur);
            }
            prev = cur;
        }
    }
}

void RootFunctionBuilder::LinkIfStmtSlots(const ir::IfStmt& ifStmt)
{
    auto slotManager = program_.GetTensorSlotManager();
    auto linkYieldSlots = [&](const ir::SeqStmtsPtr& body) {
        if (body == nullptr || body->stmts_.empty()) {
            return;
        }
        auto yieldStmt = std::dynamic_pointer_cast<const ir::YieldStmt>(body->stmts_.back());
        if (yieldStmt == nullptr) {
            return;
        }
        for (size_t i = 0; i < yieldStmt->value_.size() && i < ifStmt.returnVars_.size(); i++) {
            auto yieldLt = AsLogicalTensor(yieldStmt->value_[i]);
            auto returnLt = AsLogicalTensor(ifStmt.returnVars_[i]);
            if (yieldLt != nullptr && returnLt != nullptr) {
                slotManager->SetSameSlot(returnLt, yieldLt);
            }
        }
    };
    linkYieldSlots(ifStmt.thenBody_);
    if (ifStmt.elseBody_) {
        linkYieldSlots(ifStmt.elseBody_.value());
    }
}

void RootFunctionBuilder::LinkControlFlowSlots(const ir::StmtPtr& stmt)
{
    if (!stmt) {
        return;
    }
    switch (stmt->GetKind()) {
        case ir::ObjectKind::SeqStmts: {
            auto seq = std::dynamic_pointer_cast<const ir::SeqStmts>(stmt);
            if (seq) {
                for (auto& child : seq->stmts_) {
                    LinkControlFlowSlots(child);
                }
            }
            break;
        }
        case ir::ObjectKind::ForStmt: {
            auto forStmt = std::static_pointer_cast<const ir::ForStmt>(stmt);
            auto config = forStmt->GetAttr<std::shared_ptr<ConfigScope>>("_config_scope");
            ConfigManagerNg::ScopedRestore scoped(config);
            LinkForStmtSlots(*forStmt);
            LinkControlFlowSlots(forStmt->body_);
            break;
        }
        case ir::ObjectKind::IfStmt: {
            auto ifStmt = std::static_pointer_cast<const ir::IfStmt>(stmt);
            LinkIfStmtSlots(*ifStmt);
            LinkControlFlowSlots(ifStmt->thenBody_);
            if (ifStmt->elseBody_) {
                LinkControlFlowSlots(ifStmt->elseBody_.value());
            }
            break;
        }
        default:
            break;
    }
}

void RootFunctionBuilder::ReplacePlaceholders(ir::StmtPtr stmt)
{
    if (!stmt) {
        return;
    }

    switch (stmt->GetKind()) {
        case ir::ObjectKind::SeqStmts: {
            auto seq = ir::SeqStmts::AsMut(stmt);
            for (auto& child : seq->stmts_) {
                if (IsPlaceholderCallStmt(child)) {
                    child = FinalizePathFunc(child);
                } else {
                    ReplacePlaceholders(child);
                }
            }
            break;
        }
        case ir::ObjectKind::ForStmt: {
            auto forStmt = std::static_pointer_cast<const ir::ForStmt>(stmt);
            auto config = forStmt->GetAttr<std::shared_ptr<ConfigScope>>("_config_scope");
            ConfigManagerNg::ScopedRestore scoped(config);
            ReplacePlaceholders(forStmt->body_);
            break;
        }
        case ir::ObjectKind::IfStmt: {
            auto ifStmt = std::static_pointer_cast<const ir::IfStmt>(stmt);
            ReplacePlaceholders(ifStmt->thenBody_);
            if (ifStmt->elseBody_) {
                ReplacePlaceholders(ifStmt->elseBody_.value());
            }
            break;
        }
        default:
            break;
    }
}

ir::StmtPtr RootFunctionBuilder::TransformBody(ir::StmtPtr stmt)
{
    if (!stmt) {
        return nullptr;
    }

    paramTensors_.clear();
    for (auto& param : logicalParams_) {
        paramTensors_.insert(param);
    }

    consumedTensors_.clear();

    auto irTree = StmtTransformer(*this).Apply(stmt, "");
    LinkControlFlowSlots(irTree);
    LinkReturnSlots(irTree);
    ReplacePlaceholders(irTree);

    return irTree;
}

void RootFunctionBuilder::DumpFunctionGraph(Function* func)
{
    if (!config::GetPassDefaultConfig(KEY_PRINT_GRAPH, false)) {
        return;
    }
    std::string dumpDir = config::LogTensorGraphFolder();
    std::string baseName = func->GetRawName();
    func->DumpJsonFile(dumpDir + "/" + baseName + ".json");
    func->DumpFile(dumpDir + "/" + baseName + ".tifwkgr");
}

void RootFunctionBuilder::FlushConstructAssembleSlots()
{
    auto slotManager = program_.GetTensorSlotManager();

    std::unordered_set<int> paramSlotIds;
    std::unordered_set<int> memIds;
    for (auto& param : logicalParams_) {
        auto tensor = slotManager->GetSlotTensor(param);
        paramSlotIds.insert(tensor->Id());
        memIds.insert(param->tensor->memoryId);
    }

    std::unordered_set<Function*> affectedPathFuncs;
    for (auto& [lt, func] : tensorConstructFunc_) {
        auto tensor = slotManager->GetSlotTensor(lt, false);
        if (tensor == nullptr) {
            continue;
        }
        auto slot = TensorSlot::CreateTensor(*tensor);
        if (slotManager->assembleSlotSet.count(slot) == 0) {
            continue;
        }
        if (paramSlotIds.count(tensor->Id()) > 0) {
            continue;
        }
        if (memIds.count(lt->tensor->memoryId) > 0) {
            continue;
        }
        if (!func->HasParent()) {
            continue;
        }
        auto& pathFunc = func->Parent();
        auto scope = pathFunc.GetSlotScope();
        if (scope) {
            scope->constructAssembleSlotList.push_back(slot.GetId());
            affectedPathFuncs.insert(&pathFunc);
        }
    }
    for (auto* pathFunc : affectedPathFuncs) {
        auto scope = pathFunc->GetSlotScope();
        std::sort(scope->constructAssembleSlotList.begin(), scope->constructAssembleSlotList.end());
    }
}

} // namespace npu::tile_fwk
