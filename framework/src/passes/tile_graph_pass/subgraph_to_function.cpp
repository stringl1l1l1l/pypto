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
 * \file subgraph_to_function.cpp
 * \brief
 */

#include "passes/tile_graph_pass/subgraph_to_function.h"
#include <algorithm>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include "interface/function/function.h"
#include "interface/function/rebuildable_attribute.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_utils/pass_operation_utils.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "passes/pass_utils/parallel_tool.h"
#include "passes/pass_check/subgraph_to_function_checker.h"
#include "passes/pass_utils/graph_utils.h"
#include "passes/pass_utils/token_utils.h"
#include "passes/pass_utils/pass_token_utils.h"
#include "passes/pass_utils/subgraph_utils.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/pass_log/pass_log.h"
#include "passes/tile_graph_pass/graph_partition/graph_partition_token_dependency.h"
#include "tilefwk/error_code.h"

#undef MODULE_NAME
#define MODULE_NAME "SubgraphToFunction"

namespace npu::tile_fwk {
namespace {
const SymbolicScalar& GetParamAddrSymbol()
{
    static const SymbolicScalar kGetParamAddr = []() {
        IRBuilder builder;
        return builder.CreateScalarVar(AddRuntimeCoaPrefix("GET_PARAM_ADDR"));
    }();
    return kGetParamAddr;
}

ir::StmtPtr AsStmtPtr(Operation& op) { return std::static_pointer_cast<const ir::Stmt>(op.shared_from_this()); }

Operation* ToOperation(const ir::StmtPtr& stmt)
{
    if (stmt == nullptr) {
        return nullptr;
    }
    return static_cast<Operation*>(const_cast<ir::Stmt*>(stmt.get()));
}

void AddCallOpTokenDependency(Function& rootFunction, Operation& producerCallOp, Operation& consumerCallOp)
{
    if (&producerCallOp == &consumerCallOp) {
        return;
    }
    if (producerCallOp.result_token_.empty()) {
        producerCallOp.result_token_.push_back(IRBuilder().CreateTokenVar(producerCallOp.GetSpan()));
    }
    const auto& token = producerCallOp.result_token_.front();
    rootFunction.GetVarDependency().AddProducer(token, AsStmtPtr(producerCallOp));
    rootFunction.GetVarDependency().AddConsumer(token, AsStmtPtr(consumerCallOp));
    PassTokenUtils::AddTokenIfAbsent(consumerCallOp, token);
}

} // namespace

void SubgraphToFunction::Init()
{
    subFuncInvokeInfos.clear();
    viewToCopyInMapping_.clear();
    mixSubgraphIds_.clear();
}

Status SubgraphToFunction::RunOnFunction(Function& function)
{
    /* 需要将所有缓存在类成员的信息清零 */
    Init();

    if (FinalizePartitionWithTokenDependency(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Failed to finalize token-aware partitions before subgraph conversion.");
        return FAILED;
    }

    if (TransViewToCopyInBeforeGenSubgraph(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Failed to transfer view into copy in.");
        return FAILED;
    }
    // build in-graph and out-graph at first
    // GetTensorData: Add dependency
    if (GetTensorDataDependencyInsert(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Failed to build the data dependency structure.");
        return FAILED;
    }
    // 只在静态流程中构建图
    if (function.GetFunctionType() == FunctionType::STATIC) {
        // 1. Construct in-graph & out-graph
        if (staticProcessor_.BuildGraph(function) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "Failed to build graph from input function.");
            return FAILED;
        }
        // reconnect in-graph and out-graph by Incast and Outcast
        RecordIncastOutcast(function);
        SetupStaticProcessor();
    } else {
        // reconnect in-graph and out-graph by Incast and Outcast
        RecordIncastOutcast(function);
    }
    // Construct function.subFunctionInvokeMap
    ConstructParamMap(function);
    // Determine the isomorphism of subgraphs and record ProgramInfoMap
    Function::EnableMagicLookupRecord(true, &function);
    IslandToFunction(function);
    Function::EnableMagicLookupRecord(false, &function);
    // GetTensorData: Remove dependency
    if (GetTensorDataDependencyClear(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Failed to clear data dependency.");
        return FAILED;
    }
    if (RecoverCopyInToViewAfterGenSubgraph(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Failed to recover copy in into view.");
        return FAILED;
    }
    return SUCCESS;
}

// Add string name for codegen
std::string SubgraphToFunction::FindSymbolName(std::shared_ptr<LogicalTensor> op, int magic) const
{
    if (magic < 0) {
        return std::to_string(magic);
    }
    if (magic == 0) {
        return "NULL_OPERAND";
    }
    // DDR variable
    MemoryType originalType = op->GetMemoryTypeOriginal();
    if (originalType == MemoryType::MEM_DEVICE_DDR) {
        return "Var$" + std::to_string(magic);
    }

    auto name = "$" + std::to_string(magic);
    return name;
}

void SubgraphToFunction::RecordConnectionWithProducers(RecordInfo recordInfo, SubfuncInvokeInfoTy& iter)
{
    std::vector<int> assembleRawMagic;
    size_t i = recordInfo.i;
    size_t j = recordInfo.j;
    size_t k = recordInfo.k;
    LogicalTensorPtr iOperand = recordInfo.operand;
    Offset offset = recordInfo.offset;
    Shape shape = recordInfo.shape;
    for (auto& producer : iOperand->GetProducers()) {
        auto eSgId = producer->GetSubgraphID();
        std::vector<int>::iterator it = find(assembleRawMagic.begin(), assembleRawMagic.end(), iOperand->GetRawMagic());
        if (it != assembleRawMagic.end()) {
            continue;
        }
        assembleRawMagic.push_back(iOperand->GetRawMagic());
        iter.RecordConnection(eSgId, i, k, iOperand->GetRawMagic() /*placeHolder*/, offset, shape,
                              iOperand->tensor->rawshape, iOperand->Datatype(), iOperand, nLIST[i][j]->opmagic);
    }
}

bool isFromCast(LogicalTensorPtr& operand)
{
    if (!(operand->GetConsumers().empty()) && !(operand->GetProducers().empty())) {
        std::set<int> boundTensorIDs;
        for (auto& outOp : operand->GetConsumers()) {
            boundTensorIDs.insert(outOp->GetSubgraphID());
        }
        for (auto& inOp : operand->GetProducers()) {
            boundTensorIDs.insert(inOp->GetSubgraphID());
        }
        if (boundTensorIDs.size() == 1) {
            return false;
        }
    }
    return true;
}

bool ContainsLogicalTensor(const std::vector<LogicalTensorPtr>& tensors, const LogicalTensorPtr& operand)
{
    if (operand == nullptr) {
        return false;
    }
    const int magic = operand->GetMagic();
    return std::any_of(tensors.begin(), tensors.end(),
                       [magic](const LogicalTensorPtr& tensor) { return tensor && tensor->GetMagic() == magic; });
}

// Function in/outcast by GetMagic, not other versions on the same RawTensor.
bool IsExactFuncCast(const Function& function, const LogicalTensorPtr& operand)
{
    return ContainsLogicalTensor(function.GetOutcast(), operand) ||
           ContainsLogicalTensor(function.GetIncast(), operand);
}

uint32_t GetTensorCoreFlag(const LogicalTensorPtr& tensor)
{
    constexpr uint32_t aicFlag = 1U;
    constexpr uint32_t aivFlag = 2U;
    if (tensor == nullptr) {
        return 0;
    }
    MemoryType memoryType = tensor->GetMemoryTypeOriginal();
    if (memoryType == MemoryType::MEM_L1 || memoryType == MemoryType::MEM_L0A || memoryType == MemoryType::MEM_L0B ||
        memoryType == MemoryType::MEM_L0C) {
        return aicFlag;
    }
    if (memoryType == MemoryType::MEM_UB) {
        return aivFlag;
    }
    return 0;
}

bool IsInternalTensorOfSubgraph(const LogicalTensorPtr& tensor, int subgraphId)
{
    bool hasProducer = false;
    bool hasConsumer = false;
    for (auto& producer : tensor->GetProducers()) {
        hasProducer = hasProducer || (producer != nullptr && producer->GetSubgraphID() == subgraphId);
    }
    for (auto& consumer : tensor->GetConsumers()) {
        hasConsumer = hasConsumer || (consumer != nullptr && consumer->GetSubgraphID() == subgraphId);
    }
    return hasProducer && hasConsumer;
}

std::unordered_set<int> GetMixSubgraphIds(Function& function)
{
    constexpr uint32_t aicFlag = 1U;
    constexpr uint32_t aivFlag = 2U;
    std::unordered_map<int, uint32_t> subgraphCoreFlags;
    std::unordered_set<int> visitedTensors;
    auto allOps = function.Operations(true).DuplicatedOpList();
    for (const auto& op : allOps) {
        if (op == nullptr) {
            continue;
        }
        for (const auto& tensor : op->GetOOperands()) {
            if (tensor == nullptr || visitedTensors.count(tensor->GetMagic()) != 0) {
                continue;
            }
            visitedTensors.insert(tensor->GetMagic());
            uint32_t tensorFlag = GetTensorCoreFlag(tensor);
            if (tensorFlag == 0) {
                continue;
            }
            for (auto& producer : tensor->GetProducers()) {
                if (producer == nullptr || producer->GetSubgraphID() < 0 ||
                    !IsInternalTensorOfSubgraph(tensor, producer->GetSubgraphID())) {
                    continue;
                }
                subgraphCoreFlags[producer->GetSubgraphID()] |= tensorFlag;
            }
        }
    }

    std::unordered_set<int> mixSubgraphIds;
    for (const auto& subgraphCoreFlag : subgraphCoreFlags) {
        uint32_t flags = subgraphCoreFlag.second;
        if ((flags & aicFlag) != 0 && (flags & aivFlag) != 0) {
            mixSubgraphIds.insert(subgraphCoreFlag.first);
        }
    }
    return mixSubgraphIds;
}

bool IsMiddleInCast(LogicalTensorPtr& operand, Operation& op, const std::unordered_set<int>& mixSubgraphIds)
{
    if (operand->GetProducers().size() == 0 || mixSubgraphIds.count(op.GetSubgraphID()) == 0) {
        return false;
    }
    int src = op.GetSubgraphID();
    for (auto& inOp : operand->GetProducers()) {
        if (inOp->GetSubgraphID() != src) {
            return false;
        }
    }
    return true;
}

void SubgraphToFunction::RecordIncastInfo(Function& function, RecordInfo recordInfo, SubfuncInvokeInfoTy& iter)
{
    size_t i = recordInfo.i;
    size_t j = recordInfo.j;
    size_t k = recordInfo.k;
    LogicalTensorPtr iOperand = recordInfo.operand;
    Offset offset = recordInfo.offset;
    Shape shape = recordInfo.shape;
    auto& op = *nLIST[i][j];
    if (!isFromCast(iOperand) || IsMiddleInCast(iOperand, op, mixSubgraphIds_)) {
        APASS_LOG_INFO_F(Elements::Tensor, "Tensor %d has consumer in same subgraph, cannot be incast.",
                         iOperand->GetMagic());
        return;
    }
    // 这里逻辑可能有一些问题，期望是尽可能不要把inplace语义的COPY_OUT的输出变成leaf的incast
    if (op.HasAttribute(OpAttributeKey::inplaceIdx) && !iOperand->GetProducers().empty()) {
        if (!SubgraphUtils::IsBoundary(iOperand)) {
            return;
        }
    }
    if (function.IsFromInCast(iOperand) || function.IsFromOutCast(iOperand)) {
        iter.RecordTensorArg(k, iOperand->GetRawMagic(), offset, shape, iOperand->tensor->rawshape,
                             iOperand->Datatype(), false, iOperand, nLIST[i][j]->opmagic);
        return;
    }
    if (!SubgraphUtils::IsBoundary(iOperand) || iOperand->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
        return;
    }
    auto producers = iOperand->GetProducers();
    if (producers.size() != 0) {
        RecordConnectionWithProducers(recordInfo, iter);
        return;
    }
    if (IsCopyIn(nLIST[i][j]->GetOpcode())) {
        iter.RecordConnection(i, i, k, iOperand->GetRawMagic(), offset, shape, iOperand->tensor->rawshape,
                              iOperand->Datatype(), iOperand, nLIST[i][j]->opmagic);
    }
}

void SubgraphToFunction::RecordEsgIncast(Function& function, size_t i, size_t j, size_t k)
{
    auto& iter = subFuncInvokeInfos[i];
    auto iOperand = nLIST[i][j]->GetInputOperand(k);
    auto offset = iOperand->offset;
    auto shape = iOperand->shape;
    if (IsCopyIn(nLIST[i][j]->GetOpcode())) {
        offset.clear();
        std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(
            nLIST[i][j]->GetOpAttribute());
        std::vector<OpImmediate> opImmList = attr->GetCopyInAttr().first;
        for (auto& opImm : opImmList) {
            offset.push_back(opImm.GetSpecifiedValue().ConcreteValid() ? static_cast<int>(opImm.GetSpecifiedValue()) :
                                                                         -1);
        }
        shape = attr->GetSpecifiedShape(1);
    }
    // 1. IndexOutCast存在外部输入也是子图内的输出和输入的情况
    // 2. 对于Assemble的输入如果是来自于OutCast，那么一定会在某个CopyOut的输出上被记录
    RecordInfo recordInfo = {i, j, k, iOperand, shape, offset};
    RecordIncastInfo(function, recordInfo, iter);
}

void SubgraphToFunction::RecordOutcastInfo(Function& function, RecordInfo recordInfo, SubfuncInvokeInfoTy& iter)
{
    size_t i = recordInfo.i;
    size_t j = recordInfo.j;
    size_t k = recordInfo.k;
    LogicalTensorPtr oOperand = recordInfo.operand;
    Offset offset = recordInfo.offset;
    Shape shape = recordInfo.shape;
    auto& op = *nLIST[i][j];
    if (!isFromCast(oOperand) && !IsExactFuncCast(function, oOperand)) {
        APASS_LOG_INFO_F(Elements::Tensor, "Tensor %d has consumer in same subgraph, cannot be outcast.",
                         oOperand->GetMagic());
        return;
    }
    if (op.HasAttribute(OpAttributeKey::inplaceIdx) &&
        (op.GetOpcode() != Opcode::OP_COPY_OUT && op.GetOpcode() != Opcode::OP_INDEX_PUT &&
         op.GetOpcode() != Opcode::OP_INDEX_ADD)) {
        return;
    }
    if (function.IsFromOutCast(oOperand) || function.IsFromInCast(oOperand)) {
        iter.RecordTensorArg(k, oOperand->GetRawMagic(), offset, shape, oOperand->tensor->rawshape,
                             oOperand->Datatype(), true, oOperand, nLIST[i][j]->opmagic);
        return;
    }
    // boundary outCasts_
    int refCount = 0;
    typename SubfuncInvokeInfoTy::SuccessorIncastInfoTy relatedIncastList;
    if (!SubgraphUtils::IsBoundary(oOperand) || oOperand->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
        return;
    }
    auto consumers = oOperand->GetConsumers();
    for (auto& consumer : consumers) {
        auto eSgId = consumer->GetSubgraphID();
        if (eSgId == static_cast<int>(i)) {
            continue;
        }
        refCount++;
        int connectedTgtOperandIdx = oOperand->magic;
        relatedIncastList.push_back(typename SubfuncInvokeInfoTy::SuccessorIncastRecTy(
            eSgId, connectedTgtOperandIdx, nullptr, consumer->GetOpMagic()));
    }
    iter.RecordOutcast(i, k, refCount, oOperand->GetRawMagic(), relatedIncastList, offset, shape,
                       oOperand->tensor->rawshape, oOperand->Datatype(), oOperand, nLIST[i][j]->opmagic);
    nLIST[i][j]->outcastRefcount = refCount;
}

void SubgraphToFunction::RecordEsgOutcast(Function& function, size_t i, size_t j, size_t k)
{
    auto& iter = subFuncInvokeInfos[i];
    // 4.2 Record oOperand info, global tensor and outCasts_
    auto oOperand = nLIST[i][j]->GetOutputOperand(k);
    auto offset = oOperand->offset;
    auto shape = oOperand->shape;
    if (IsCopyOut(nLIST[i][j]->GetOpcode())) {
        offset.clear();
        std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(
            nLIST[i][j]->GetOpAttribute());
        std::vector<OpImmediate> opImmList = attr->GetCopyOutAttr().second;
        for (auto& opImm : opImmList) {
            offset.push_back(opImm.GetSpecifiedValue().ConcreteValid() ? static_cast<int>(opImm.GetSpecifiedValue()) :
                                                                         -1);
        }
        shape = attr->GetSpecifiedShape(1);
    }
    RecordInfo recordInfo = {i, j, k, oOperand, shape, offset};
    RecordOutcastInfo(function, recordInfo, iter);
}

void SubgraphToFunction::ConstructnList(Function& function)
{
    auto list = function.Operations();
    nLIST.resize(function.GetTotalSubGraphCount());
    for (size_t i = 0; i < list.size(); i++) {
        if (list[i].GetSubgraphID() < 0) {
            continue;
        }
        nLIST[list[i].GetSubgraphID()].push_back(list.operations_[i]);
    }
    while (!nLIST.empty() && nLIST.back().empty()) {
        nLIST.pop_back();
    }
}

void SubgraphToFunction::RecordEsgIncastOutcast(Function& function)
{
    for (int i = 0; i < static_cast<int>(nLIST.size()); i++) {
        for (size_t j = 0; j < nLIST[i].size(); j++) {
            // 4.1 Record iOperand info, global tensor and incast
            for (size_t k = 0; k < nLIST[i][j]->iOperand.size(); k++) {
                RecordEsgIncast(function, i, j, k);
            }
            for (size_t k = 0; k < nLIST[i][j]->GetOOperands().size(); k++) {
                RecordEsgOutcast(function, i, j, k);
            }
        }
    }

    auto scope = function.GetSlotScope();
    if (scope == nullptr) {
        return;
    }
    for (int outcastIndex : scope->ioslot.partialUpdateOutcastList) {
        if (outcastIndex < 0 || static_cast<size_t>(outcastIndex) >= function.GetOutcast().size()) {
            continue;
        }
        const auto& outcast = function.GetOutcast()[outcastIndex];
        auto link = function.outIncastLinkMap.find(outcast->GetRawTensor());
        if (link == function.outIncastLinkMap.end()) {
            continue;
        }
        auto actualIncast = std::find_if(
            function.GetIncast().begin(), function.GetIncast().end(),
            [&link](const auto& incast) { return incast->GetRawTensor() == link->second; });
        if (actualIncast == function.GetIncast().end()) {
            continue;
        }
        for (size_t i = 0; i < nLIST.size(); i++) {
            const auto& tensorArgs = subFuncInvokeInfos[i].GetTensorArgs();
            bool storageAlreadyInput = std::any_of(
                tensorArgs.begin(), tensorArgs.end(), [&outcast, &actualIncast](const auto& arg) {
                    return !arg.isOutputToGM &&
                           (arg.realDDRId == outcast->GetRawMagic() || arg.realDDRId == (*actualIncast)->GetRawMagic());
                });
            if (storageAlreadyInput) {
                continue;
            }
            bool storageComesFromOtherSubgraph = false;
            for (const auto& op : nLIST[i]) {
                for (const auto& input : op->GetIOperands()) {
                    if (input->GetRawMagic() != outcast->GetRawMagic()) {
                        continue;
                    }
                    storageComesFromOtherSubgraph = std::any_of(
                        input->GetProducers().begin(), input->GetProducers().end(),
                        [i](const auto* producer) { return producer->GetSubgraphID() != static_cast<int>(i); });
                    if (storageComesFromOtherSubgraph) {
                        break;
                    }
                }
                if (storageComesFromOtherSubgraph) {
                    break;
                }
            }
            if (storageComesFromOtherSubgraph) {
                continue;
            }
            bool recorded = false;
            for (const auto& op : nLIST[i]) {
                if (!IsCopyIn(op->GetOpcode())) {
                    continue;
                }
                for (size_t k = 0; k < op->GetIOperands().size(); k++) {
                    if (op->GetIOperands()[k]->GetRawMagic() != outcast->GetRawMagic()) {
                        continue;
                    }
                    const auto& incast = *actualIncast;
                    outcast->tensor->actualRawmagic = incast->GetRawMagic();
                    subFuncInvokeInfos[i].RecordTensorArg(k, incast->GetRawMagic(), incast->GetOffset(),
                                                          incast->GetShape(), incast->tensor->rawshape,
                                                          incast->Datatype(), false, incast, op->GetOpMagic());
                    recorded = true;
                    break;
                }
                if (recorded) {
                    break;
                }
            }
        }
    }
}

void SubgraphToFunction::RecordIncastOutcast(Function& function)
{
    // 1. Get function->operations_, construct 2-dimension nLIST（subgraphID, operations)
    ConstructnList(function);
    mixSubgraphIds_ = GetMixSubgraphIds(function);
    // 2. Init InvokeInfo, {ESGID, SubfuncInvokeInfo}, which contains all the invoke info of each subgrahs
    for (size_t i = 0; i < nLIST.size(); i++) {
        subFuncInvokeInfos.push_back(SubfuncInvokeInfoTy());
    }
    // 3. Construct subgraph with label boundary.
    RecordEsgIncastOutcast(function);
    // 4. Incast， Outcast，Construct connection using connetion and outcast
    for (auto& item : subFuncInvokeInfos) {
        item.DoFinishRecord();
    }
}

void SubgraphToFunction::ConstructParamMap(Function& function)
{
    if (function.GetFunctionType() == FunctionType::STATIC) {
        function.topoInfo_ = staticProcessor_.ConstructSubgraphTopologyInfo(function, subFuncInvokeInfos);
    }
    for (size_t i = 0; i < subFuncInvokeInfos.size(); i++) {
        subFuncInvokeInfos[i].ConstructActualInvokeParam(i);
    }
}

void SubgraphToFunction::ProcessInputOperands(Function& rootFunc, Operation& tileOp, SubfuncParam& pSgParamInfo,
                                              int& tParamLoc, int& iParamLoc) const
{
    for (size_t k = 0; k < tileOp.GetIOperands().size(); k++) {
        auto iOperand = tileOp.GetInputOperand(k);
        std::string name = FindSymbolName(iOperand, iOperand->GetRawMagic());
        auto offset = iOperand->offset;
        auto shape = iOperand->shape;
        if (tileOp.HasAttribute(OpAttributeKey::inplaceIdx) && !iOperand->GetProducers().empty()) {
            if (!SubgraphUtils::IsBoundary(iOperand)) {
                continue;
            }
        }
        if (IsCopyIn(tileOp.GetOpcode())) {
            ProcessCopyInOperand(tileOp, offset, shape);
        }
        if (rootFunc.IsFromInCast(iOperand) || rootFunc.IsFromOutCast(iOperand)) {
            // Offsets are a part of each parameter, but we need to keep their original value to
            // keep track of dependencies within a subgraph.
            pSgParamInfo.AppendTensorParam(k, iOperand->GetRawMagic(), shape, offset, name, tParamLoc,
                                           iOperand->tensor->GetSymbol(), iOperand->tensor->GetDataType());
            tileOp.inParamLocation_.push_back(tParamLoc);
            tParamLoc++;
            continue;
        }
        if (!SubgraphUtils::IsBoundary(iOperand) || iOperand->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
            continue;
        }
        pSgParamInfo.AppendIncastParam(k, iOperand->GetRawMagic(), shape, offset, name, iParamLoc,
                                       iOperand->tensor->GetSymbol(), iOperand->tensor->GetDataType());
        tileOp.inParamLocation_.push_back(iParamLoc);
        iParamLoc++;
    }
}

void SubgraphToFunction::ProcessOutputOperands(Function& rootFunc, Operation& tileOp, SubfuncParam& pSgParamInfo,
                                               int& tParamLoc, int& oParamLoc) const
{
    for (size_t k = 0; k < tileOp.GetOOperands().size(); k++) {
        auto oOperand = tileOp.GetOutputOperand(k);
        std::string name = FindSymbolName(oOperand, oOperand->GetRawMagic());
        auto offset = oOperand->offset;
        auto shape = oOperand->shape;
        if (tileOp.HasAttribute(OpAttributeKey::inplaceIdx) &&
            (tileOp.GetOpcode() != Opcode::OP_COPY_OUT && tileOp.GetOpcode() != Opcode::OP_INDEX_PUT) &&
            tileOp.GetOpcode() != Opcode::OP_INDEX_ADD) {
            return;
        }
        if (IsCopyOut(tileOp.GetOpcode())) {
            ProcessCopyOutOperand(tileOp, offset, shape);
        }
        if (rootFunc.IsFromOutCast(oOperand) || rootFunc.IsFromInCast(oOperand)) {
            pSgParamInfo.AppendTensorParam(k, oOperand->GetRawMagic(), shape, offset, name, tParamLoc,
                                           oOperand->tensor->GetSymbol(), oOperand->tensor->GetDataType());
            tileOp.outParamLocation_.push_back(tParamLoc);
            tParamLoc++;
            continue;
        }
        if (!SubgraphUtils::IsBoundary(oOperand) || oOperand->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
            continue;
        }
        pSgParamInfo.AppendOutcastParam(k, oOperand->GetRawMagic(), tileOp.outcastRefcount, shape, offset, name,
                                        oParamLoc, oOperand->tensor->GetSymbol(), oOperand->tensor->GetDataType());
        tileOp.outParamLocation_.push_back(oParamLoc);
        oParamLoc++;
    }
}

void SubgraphToFunction::ProcessCopyInOperand(Operation& tileOp, std::vector<int64_t>& offset,
                                              std::vector<int64_t>& shape) const
{
    std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(tileOp.GetOpAttribute());
    if (!attr) {
        APASS_LOG_WARN_F(
            Elements::Operation,
            "Invalid attribute for copyin operation %d. Please check the input graph if the attribute of the operation "
            "is missing.",
            tileOp.GetOpMagic());
        return;
    }
    std::vector<OpImmediate> opImmList = attr->GetCopyInAttr().first;
    shape = attr->GetSpecifiedShape(kShapePlaceholderForParameterized);
    if (!opImmList.empty() && opImmList[0].IsParameter()) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "CopyinOperand: First operand is parameter, skip offset processing.");
        return;
    }
    offset.clear();
    for (auto& opImm : opImmList) {
        offset.push_back(opImm.GetSpecifiedValue().ConcreteValid() ? static_cast<int>(opImm.GetSpecifiedValue()) : -1);
    }
}

void SubgraphToFunction::ProcessCopyOutOperand(Operation& tileOp, std::vector<int64_t>& offset,
                                               std::vector<int64_t>& shape) const
{
    std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(tileOp.GetOpAttribute());
    if (!attr) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "CopyOutOperand: Invalid op attribute for op %d.", tileOp.GetOpMagic());
        return;
    }
    std::vector<OpImmediate> opImmList = attr->GetCopyOutAttr().second;
    shape = attr->GetSpecifiedShape(kShapePlaceholderForParameterized);
    if (!opImmList.empty() && opImmList[0].IsParameter()) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "CopyOutOperand: First operand is parameter, skip offset processing.");
        return;
    }
    offset.clear();
    for (auto& opImm : opImmList) {
        offset.push_back(opImm.GetSpecifiedValue().ConcreteValid() ? static_cast<int>(opImm.GetSpecifiedValue()) : -1);
    }
}

void SubgraphToFunction::ProcessSymbolOfReshape(Function& function, Operation& op) const
{
    // ddr -> reshape -> ddr -> copyin
    if (op.GetOpcode() == Opcode::OP_RESHAPE) {
        if (function.IsFromInCast(op.GetOOperands().front())) {
            op.GetOOperands().front()->tensor->SetSymbol(op.GetIOperands().front()->tensor->GetSymbol());
        }
    }
    // copyout -> ddr -> reshape -> ddr
    auto nextOp = *(op.GetOOperands().front()->GetConsumers().begin());
    if (nextOp != nullptr && nextOp->GetOpcode() == Opcode::OP_RESHAPE) {
        if (function.IsFromOutCast(op.GetOOperands().front())) {
            op.GetOOperands().front()->tensor->SetSymbol(nextOp->GetOOperands().front()->tensor->GetSymbol());
        }
    }
}

void SubgraphToFunction::SymbolizeEachFunction(Function& rootFunc, std::vector<Function*>& mergedFuncList1,
                                               size_t i) const
{
    int pSgId = i;
    int tParamLoc = 0;
    int iParamLoc = 0 | 0x10000000;
    int oParamLoc = 0 | 0x20000000;
    SubfuncParam pSgParamInfo;
    // do symbolize only for real merged subgraph, others are constant program
    auto& leafFunc = mergedFuncList1[i];
    for (auto& tileOp : leafFunc->Operations()) {
        // symbolic
        ProcessSymbolOfReshape(rootFunc, tileOp);
        ProcessInputOperands(rootFunc, tileOp, pSgParamInfo, tParamLoc, iParamLoc);
        ProcessOutputOperands(rootFunc, tileOp, pSgParamInfo, tParamLoc, oParamLoc);
    }

    pSgParamInfo.Finalize();
    mergedFuncList1[pSgId]->SetParameter(pSgParamInfo);
    mergedFuncList1[pSgId]->SetProgramId(pSgId);
    rootFunc.programs_.insert({pSgId, mergedFuncList1[pSgId]});
}

void SubgraphToFunction::SymbolizeFunction(Function& rootFunc, std::vector<Function*>& mergedFuncList1) const
{
    for (size_t i = 0; i < mergedFuncList1.size(); i++) {
        SymbolizeEachFunction(rootFunc, mergedFuncList1, i);
    }
}

void SubgraphToFunction::InsertParameter(size_t i, Function& leafFunc)
{
    for (auto& in : subFuncInvokeInfos[i].GetIncastTensorParamList()) {
        leafFunc.AppendIncast(in.tensor, in.opMagic, in.operandIdx);
    }
    for (auto& out : subFuncInvokeInfos[i].GetOutcastTensorParamList()) {
        leafFunc.AppendOutcast(out.tensor, out.opMagic, out.operandIdx);
    }
    for (auto& tensor : subFuncInvokeInfos[i].GetTensorParamList()) {
        leafFunc.AddGlobalTensor(tensor.tensor);
        if (tensor.isOutputToGM) {
            leafFunc.AppendOutcast(tensor.tensor, tensor.opMagic, tensor.operandIdx);
            continue;
        }
        leafFunc.AppendIncast(tensor.tensor, tensor.opMagic, tensor.operandIdx);
    }
}

Status SubgraphToFunction::ProcessSubgraph(Function& function, size_t i, size_t& programIdx,
                                           std::vector<Function*>& outputFuncList, Operation*& callOp)
{
    callOp = nullptr;
    auto subgraph = nLIST[i];
    auto leafName = function.GetRawName() + "_leaf" + std::to_string(i);
    APASS_LOG_DEBUG_F(Elements::Graph, "Add leafFunction %s.", leafName.c_str());

    Program::GetInstance().BeginFunction(leafName, FunctionType::STATIC, GraphType::BLOCK_GRAPH);
    auto leafFunc = Program::GetInstance().GetCurrentFunction();
    leafFunc->SetProgramOp(subgraph);
    if (TokenUtils::RebuildTokenDependencies(*leafFunc) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Failed to rebuild token dependencies for leaf %s.", leafName.c_str());
        return FAILED;
    }
    leafFunc->SetLeafFuncAttribute(std::make_shared<LeafFuncAttribute>());
    InsertParameter(i, *leafFunc);

    // In EndFunction to calculate cache hash
    auto result = Program::GetInstance().EndFunction(leafName);
    callOp = std::get<1>(result);
    if (callOp == nullptr) {
        APASS_LOG_ERROR_F(Elements::Graph, "leafname %s, program returned nullptr.", leafName.c_str());
        return FAILED;
    }
    callOp->UpdateSubgraphID(i);
    callOp->SetSubFuncInvokeInfo(subFuncInvokeInfos[i]);

    SetSemanticLabel(subgraph, *callOp);
    SetHashOrderInfoOnCallOp(subgraph, *callOp);

    return ProcessCacheResult(result, i, programIdx, outputFuncList, *callOp);
}

Status SubgraphToFunction::ProcessCacheResult(const std::tuple<Function*, Operation*, bool>& result, size_t i,
                                              size_t& programIdx, std::vector<Function*>& outputFuncList,
                                              Operation& callOp)
{
    const int getValue = 2;
    // 3.1 Hit subgraph
    if (std::get<getValue>(result)) {
        APASS_LOG_DEBUG_F(Elements::Operation, "LeafFunc %zu Hit Current hashValue is %lu.", i,
                          std::get<0>(result)->ComputeHash().GetHash());
        psgToESgMap.insert({std::get<0>(result)->GetProgramId(), i});
        auto callAttr = dynamic_cast<CallOpAttribute*>(callOp.GetOpAttribute().get());
        if (callAttr == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Failed to get CallOpAttribute for operation %zu. %s", i,
                              GetFormatBacktrace(callOp).c_str());
            return FAILED;
        }
        auto cacheValue = Program::GetInstance().TryHitCahce(callAttr->GetCalleeHash());
        if (!cacheValue) {
            APASS_LOG_ERROR_F(Elements::Operation, "Cache miss for callee hash %lu. %s",
                              callAttr->GetCalleeHash().GetHash(), GetFormatBacktrace(callOp).c_str());
            return FAILED;
        }
        callAttr->SetCalleeMagicName(cacheValue->GetFunction()->GetMagicName());
        callAttr->invokeInfo_->UpdateProgramSubgraphId(std::get<0>(result)->GetProgramId());
        return SUCCESS;
    }
    // 3.2 not hit subgraph
    APASS_LOG_DEBUG_F(Elements::Operation, "LeafFunc %zu Not Hit. hashValue is %lu.", i,
                      std::get<0>(result)->ComputeHash().GetHash());
    psgToESgMap.insert({programIdx, i});
    std::get<0>(result)->SetProgramId(programIdx);
    auto callAttr = dynamic_cast<CallOpAttribute*>(callOp.GetOpAttribute().get());
    if (callAttr == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Failed to get CallOpAttribute for operation %zu. %s", i,
                          GetFormatBacktrace(callOp).c_str());
        return FAILED;
    }
    callAttr->invokeInfo_->UpdateProgramSubgraphId(programIdx);
    programIdx++;
    outputFuncList.push_back(std::get<0>(result));
    std::get<0>(result)->UpdateBelongToThis();
    return SUCCESS;
}

void SubgraphToFunction::SetSemanticLabel(const std::vector<std::shared_ptr<Operation>>& subgraph, Operation& callOp)
{
    std::shared_ptr<SemanticLabel> label;
    if (GetConfig("use_max_freq_label", false)) {
        std::unordered_map<std::string, int> freqMap;
        std::unordered_map<std::string, std::shared_ptr<SemanticLabel>> labelMap;
        for (const auto& op : subgraph) {
            auto str = op->GetSemanticLabelStr();
            if (str.empty()) {
                continue;
            }
            freqMap[str]++;
            labelMap[str] = op->GetSemanticLabel();
        }
        int maxCount = 0;
        for (auto& pair : freqMap) {
            if (pair.second > maxCount) {
                maxCount = pair.second;
                label = labelMap[pair.first];
            }
        }
    } else {
        if (subgraph.size() > 0) {
            label = subgraph[0]->GetSemanticLabel();
        }
    }
    callOp.SetSemanticLabel(label);
}

void SubgraphToFunction::CopyHashOrderInfoToCallOp(const std::vector<std::shared_ptr<Operation>>& subgraph,
                                                   Operation& callOp, const std::string& hashOrderKey,
                                                   const std::string& countKey) const
{
    for (const auto& op : subgraph) {
        if (op == nullptr || !op->HasAttribute(hashOrderKey) || !op->HasAttribute(countKey)) {
            continue;
        }
        auto [hashOrder, subgraphCount] = op->GetHashOrderInfo(hashOrderKey, countKey);
        callOp.SetHashOrderInfo(hashOrderKey, countKey, hashOrder, subgraphCount);
        return;
    }
}

void SubgraphToFunction::SetHashOrderInfoOnCallOp(const std::vector<std::shared_ptr<Operation>>& subgraph,
                                                  Operation& callOp) const
{
    CopyHashOrderInfoToCallOp(subgraph, callOp, OpAttributeKey::l1ReuseHashOrder, OpAttributeKey::l1ReuseSubgraphCount);
    CopyHashOrderInfoToCallOp(subgraph, callOp, OpAttributeKey::cubeMergeHashOrder,
                              OpAttributeKey::cubeMergeSubgraphCount);
    CopyHashOrderInfoToCallOp(subgraph, callOp, OpAttributeKey::vecMergeHashOrder,
                              OpAttributeKey::vecMergeSubgraphCount);
}

Status SubgraphToFunction::BuildRootCallOpTokenDependency(
    Function& originFunction, Function& rootFunction,
    const std::unordered_map<const Operation*, Operation*>& originOpToCallOp) const
{
    const auto dependencies = originFunction.GetVarDependency().GetAllDependencies();
    for (const auto& [token, entry] : dependencies) {
        (void)token;
        for (const auto& producerStmt : entry.producers) {
            auto* producerOp = ToOperation(producerStmt);
            auto producerIt = originOpToCallOp.find(producerOp);
            if (producerIt == originOpToCallOp.end()) {
                continue;
            }
            for (const auto& consumerStmt : entry.consumers) {
                auto* consumerOp = ToOperation(consumerStmt);
                auto consumerIt = originOpToCallOp.find(consumerOp);
                if (consumerIt == originOpToCallOp.end()) {
                    continue;
                }
                AddCallOpTokenDependency(rootFunction, *producerIt->second, *consumerIt->second);
            }
        }
    }
    return SUCCESS;
}

Status SubgraphToFunction::IslandToFunction(Function& function)
{
    // 1. Create root function
    auto rootName = Function::CreateRootRawName(function.GetRawName());
    Program::GetInstance().BeginFunction(rootName, function.GetFunctionType(), GraphType::EXECUTE_GRAPH);
    auto rootFunc = Program::GetInstance().GetCurrentFunction();
    if (rootFunc == nullptr) {
        APASS_LOG_ERROR_F(Elements::Function, "Failed to create root function.");
        return FAILED;
    }
    InitializeRootFunction(function, *rootFunc);

    // 2. Call HashInterface to compute hash value to determine isomorphism of each subgraph.
    size_t programIdx = 0;
    std::unordered_map<const Operation*, Operation*> originOpToCallOp;
    for (size_t i = 0; i < nLIST.size(); i++) {
        Operation* callOp = nullptr;
        Status status = ProcessSubgraph(function, i, programIdx, mergedFuncList, callOp);
        if (status != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Graph, "Failed to process subgraph %zu.", i);
            return status;
        }
        for (const auto& op : nLIST[i]) {
            if (op != nullptr) {
                originOpToCallOp[op.get()] = callOp;
            }
        }
    }
    if (BuildRootCallOpTokenDependency(function, *rootFunc, originOpToCallOp) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Failed to build root callop token dependency.");
        return FAILED;
    }

    // 3. Finalize root function
    auto rootEndResult = Program::GetInstance().EndFunction(rootName, false);
    auto resultFunc = std::get<0>(rootEndResult);
    if (resultFunc != rootFunc) {
        APASS_LOG_ERROR_F(Elements::Function, "Root function mismatch after finalization.");
        return FAILED;
    }
    if (function.GetFunctionType() == FunctionType::STATIC) {
        rootFunc->topoInfo_ = function.topoInfo_;
    }
    function.rootFunc_ = rootFunc;

    // 4. Add graphType of ESGInvokeInfoMap
    if (function.GetFunctionType() == FunctionType::STATIC) {
        Status readyStateStatus = staticProcessor_.HandleReadyStates(rootFunc);
        if (readyStateStatus != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Graph, "Failed to handle ready states.");
            return readyStateStatus;
        }
    }
    // Root calls are emitted in a valid historical order. Re-sort only when token-only dependencies require it.
    if (!rootFunc->GetVarDependency().GetAllDependencies().empty()) {
        rootFunc->SortOperations(SortOperationsMode::LIGHTWEIGHT_STABLE);
    }
    // 5. symbolize esg to program subgraph for both static and dynamic paths
    SymbolizeFunction(*rootFunc, mergedFuncList);

    auto graphNum = nLIST.size();
    APASS_LOG_INFO_F(Elements::Operation, "Compressed Graph %zu, total_Graph %zu.", programIdx, graphNum);
    return SUCCESS;
}

void SubgraphToFunction::InitializeRootFunction(Function& function, Function& rootFunc)
{
    rootFunc.SetParent(nullptr);
    if (function.IsFunctionTypeAndGraphType(FunctionType::DYNAMIC_LOOP_PATH,
                                            {GraphType::TENSOR_GRAPH, GraphType::TILE_GRAPH})) {
        rootFunc.SetDynloopAttribute(function.GetDynloopAttribute());
    }

    auto marked = RebuildableAttributeManager::GetInstance().GetAttr<RebuildableMultiIterNoOverlap>(&function)->Get();
    if (!marked.empty()) {
        RebuildableAttributeManager::GetInstance().ResetAttr<RebuildableMultiIterNoOverlap>(&rootFunc, &marked);
    }

    for (auto& tensor : function.outCasts_) {
        auto newOutcast = tensor->Clone(rootFunc);
        rootFunc.outCasts_.push_back(newOutcast);
        // update outcast
        auto it = function.outIncastLinkMap.find(tensor->tensor);
        if (it != function.outIncastLinkMap.end()) {
            rootFunc.outIncastLinkMap[newOutcast->tensor] = it->second;
        }
    }

    for (auto& tensor : function.inCasts_) {
        auto newIncast = tensor->Clone(rootFunc);
        rootFunc.inCasts_.push_back(newIncast);
        // update rootFunc incast
        for (auto it : rootFunc.outIncastLinkMap) {
            if (it.second == tensor->tensor) {
                rootFunc.outIncastLinkMap[it.first] = newIncast->tensor;
            }
        }
    }
    TensorSet allTensors = GraphUtils::GetAllTensors(rootFunc);
    APASS_LOG_DEBUG_F(Elements::Operation, "Root function tensor map size is %zu.", allTensors.size());
}

struct GetTensorDataOutcastDesc {
    std::unordered_map<Opcode, std::vector<Operation*>> opListDict;
    Operation* mark;
    Operation* copyout;
    std::shared_ptr<LogicalTensor> outcast;
};
static std::unordered_map<int, GetTensorDataOutcastDesc> GetTensorDataBuildOutcastDescDict(Function& function)
{
    auto operationViewer = function.Operations(false);
    std::unordered_map<int, GetTensorDataOutcastDesc> getTensorDataOutcastDescDict;
    for (size_t i = 0; i < operationViewer.size(); i++) {
        auto& op = operationViewer[i];
        int index = GetTensorDataGetIndex(&op);
        if (index != -1) {
            getTensorDataOutcastDescDict[index].opListDict[op.GetOpcode()].push_back(&op);
        }
    }
    for (auto& [index, desc] : getTensorDataOutcastDescDict) {
        (void)index;
        ASSERT(OperationErr::OP_SPECIAL_CONSTRAINT, desc.opListDict[Opcode::OP_ADDS].size() == 1)
            << "Expect the size is 1 for opListDict, but we get " << desc.opListDict[Opcode::OP_ADDS].size()
            << "OP_ADDS";
        auto mark = desc.opListDict[Opcode::OP_ADDS][0];

        std::shared_ptr<LogicalTensor> addsOpOut = mark->GetOOperands()[0];
        auto copyout = *addsOpOut->GetConsumers().begin();
        ASSERT(OperationErr::OP_SPECIAL_CONSTRAINT, copyout->GetOpcode() == Opcode::OP_COPY_OUT)
            << "Expect Opcode OP_COPY_OUT, but we get " << copyout->GetOpcodeStr() << " at operation["
            << copyout->GetOpMagic() << "].";
        ;

        auto outcast = copyout->GetOOperands()[0];

        desc.mark = mark;
        desc.copyout = copyout;
        desc.outcast = outcast;
    }
    return getTensorDataOutcastDescDict;
}

struct GetTensorDataUsageDesc {
    Operation* refOp;
    std::map<int, std::vector<RawSymbolicScalarPtr>> usageDict;
    MemoryType subgraphMemoryType;
    int subgraphID;

    GetTensorDataUsageDesc(Operation* refOp_, const std::map<int, std::vector<RawSymbolicScalarPtr>>& usageDict_,
                           MemoryType subgraphMemoryType_, int subgraphID_)
        : refOp(refOp_), usageDict(usageDict_), subgraphMemoryType(subgraphMemoryType_), subgraphID(subgraphID_)
    {}
};

std::shared_ptr<LogicalTensor> GetTensorDataSubgraphTensor(Operation& refOp)
{
    std::shared_ptr<LogicalTensor> subgraphTensor;
    switch (refOp.GetOpcode()) {
        case Opcode::OP_COPY_IN:
            subgraphTensor = refOp.GetOOperands()[0];
            break;
        case Opcode::OP_COPY_OUT:
            subgraphTensor = refOp.GetIOperands()[0];
            break;
        case Opcode::OP_VEC_DUP:
            subgraphTensor = refOp.GetOOperands()[0];
            break;
        case Opcode::OP_BIND_TENSOR:
            subgraphTensor = refOp.GetOOperands()[0];
            break;
        case Opcode::OP_SHMEM_LOAD:
            subgraphTensor = refOp.GetOOperands()[0];
            break;
        case Opcode::OP_SHMEM_STORE:
            subgraphTensor = refOp.GetIOperands()[1];
            break;
        case Opcode::OP_SHMEM_GET:
            subgraphTensor = refOp.GetIOperands()[1];
            break;
        case Opcode::OP_SHMEM_PUT:
            subgraphTensor = refOp.GetIOperands()[2];
            break;
        case Opcode::OP_VIEW:
            subgraphTensor = refOp.GetOOperands()[0];
            break;
        case Opcode::OP_L1_RESHAPE_COPY_IN:
            subgraphTensor = refOp.GetOOperands()[0];
            break;
        case Opcode::OP_L0C_RESHAPE_COPY_OUT:
            subgraphTensor = refOp.GetIOperands()[0];
            break;
        case Opcode::OP_RESHAPE_COPY_IN:
            subgraphTensor = refOp.GetOOperands()[0];
            break;
        case Opcode::OP_RESHAPE_COPY_OUT:
            subgraphTensor = refOp.GetIOperands()[0];
            break;
        default:
            break;
    }
    return subgraphTensor;
}

static std::vector<GetTensorDataUsageDesc> GetTensorDataBuildUsageDesc(Function& function)
{
    auto operationViewer = function.Operations(false);
    std::vector<GetTensorDataUsageDesc> getTensorDataUsageDescList;
    for (size_t i = 0; i < operationViewer.size(); i++) {
        auto& refOp = operationViewer[i];
        std::vector<std::reference_wrapper<SymbolicScalar>> dynScalarList = refOp.GetDynamicAttributeList();
        if (dynScalarList.size() == 0) {
            continue;
        }
        std::map<int, std::vector<RawSymbolicScalarPtr>> usageDict = GetTensorDataDict(dynScalarList);
        if (usageDict.size() == 0) {
            continue;
        }
        // subgraphTensor should be the same subgraph to the copyin.
        std::shared_ptr<LogicalTensor> subgraphTensor = GetTensorDataSubgraphTensor(refOp);
        ASSERT(TensorErr::TENSOR_NULL_POINTER, subgraphTensor != nullptr)
            << "Expect operation[" << refOp.GetOpMagic()
            << "] has valid IOperand/OOperand, but we get nullptr. Please check the operation.";
        MemoryType subgraphMemoryType = subgraphTensor->GetMemoryTypeToBe();
        int subgraphID = CommonUtils::GetTensorSubgraphID(subgraphTensor);
        getTensorDataUsageDescList.emplace_back(&refOp, usageDict, subgraphMemoryType, subgraphID);
    }
    return getTensorDataUsageDescList;
}

Status SubgraphToFunction::GetTensorDataDependencyInsert(Function& function)
{
    IRBuilder builder;
    std::unordered_map<int, GetTensorDataOutcastDesc> getTensorDataOutcastDescDict = GetTensorDataBuildOutcastDescDict(
        function);
    std::vector<GetTensorDataUsageDesc> getTensorDataUsageDescList = GetTensorDataBuildUsageDesc(function);

    for (auto& [refOp, usageDict, subgraphMemoryType, subgraphID] : getTensorDataUsageDescList) {
        for (auto& [index, callList] : usageDict) {
            std::shared_ptr<LogicalTensor> copyInTensor;
            if (callList.size() == 0) {
                APASS_LOG_ERROR_F(
                    Elements::Function,
                    "Call list is empty in function %s. Please check whether the input graph is complete.",
                    function.GetRawName().c_str());
                return FAILED;
            }
            // For the same index, only one copyin is necessary.
            auto getTensorDataIOType = callList[0]
                                           ->GetExpressionOperandList()[GET_TENSOR_DATA_OPERAND_INDEX_IOTYPE]
                                           ->GetImmediateValue();
            auto getTensorDataIOTypeIndex = callList[0]
                                                ->GetExpressionOperandList()[GET_TENSOR_DATA_OPERAND_INDEX_IOTYPE_INDEX]
                                                ->GetImmediateValue();

            std::shared_ptr<LogicalTensor> copyInSourceTensor;
            std::shared_ptr<CopyOpAttribute> copyInAttr;
            if (getTensorDataIOType == GET_TENSOR_DATA_OPERAND_IOTYPE_INCAST) {
                copyInSourceTensor = function.GetIncast()[getTensorDataIOTypeIndex];
                copyInTensor = builder.CreateTensorVar(copyInSourceTensor->Datatype(), copyInSourceTensor->GetShape(),
                                                       std::vector<SymbolicScalar>{}, copyInSourceTensor->Format());
                GraphUtils::CopyDynStatus(copyInTensor, copyInSourceTensor);
                std::vector<OpImmediate> copyInOffset(
                    OpImmediate::Specified(std::vector<int64_t>(copyInTensor->GetShape().size(), 0)));
                std::vector<OpImmediate> copyInShape(OpImmediate::Specified(copyInTensor->GetShape()));
                std::vector<OpImmediate> copyInRawShape(OpImmediate::Specified(copyInTensor->GetShape()));
                copyInAttr = std::make_shared<CopyOpAttribute>(copyInOffset, MemoryType::MEM_UB, copyInShape,
                                                               copyInRawShape);
            } else if (getTensorDataIOType == GET_TENSOR_DATA_OPERAND_IOTYPE_OUTCAST) {
                if (!getTensorDataOutcastDescDict.count(index)) {
                    APASS_LOG_ERROR_F(
                        Elements::Function,
                        "Index %d is not found in function %s. Please check whether the input graph is complete.",
                        index, function.GetRawName().c_str());
                    return FAILED;
                }
                auto& outcastDesc = getTensorDataOutcastDescDict[index];
                auto outcastAttr = std::static_pointer_cast<CopyOpAttribute>(outcastDesc.copyout->GetOpAttribute());
                copyInSourceTensor = outcastDesc.outcast;
                copyInTensor = builder.CreateTensorVar(copyInSourceTensor->Datatype(), copyInSourceTensor->GetShape(),
                                                       std::vector<SymbolicScalar>{}, copyInSourceTensor->Format());
                GraphUtils::CopyDynStatus(copyInTensor, copyInSourceTensor);
                copyInAttr = std::make_shared<CopyOpAttribute>(outcastAttr->GetToOffset(), MemoryType::MEM_UB,
                                                               outcastAttr->GetShape(), outcastAttr->GetRawShape());
            } else {
                // Impossible
                APASS_LOG_ERROR_F(
                    Elements::Function,
                    "The operation is neither MOVE_IN nor MOVE_OUT in function %s. Please check whether the input "
                    "graph is valid.",
                    function.GetRawName().c_str());
                return FAILED;
            }

            copyInTensor->SetMemoryTypeBoth(subgraphMemoryType);
            auto& copyInOp = PassOperationUtils::AddOperation(
                function, Opcode::OP_COPY_IN, {copyInSourceTensor}, {copyInTensor},
                [&copyInAttr](Operation& op) { op.SetOpAttribute(copyInAttr); });
            copyInOp.UpdateSubgraphID(subgraphID);
            SetEmuOpcode(&copyInOp, EMUOP_TENSOR_GETDATA_DEPEND);
            GetTensorDataSetIndex(&copyInOp, index);

            refOp->GetIOperands().push_back(copyInTensor);
            copyInTensor->AddConsumer(refOp);
        }
    }
    return SUCCESS;
}

Status SubgraphToFunction::GetTensorDataDependencyClear(Function& function)
{
    auto root = function.GetRootFunction();

    IRBuilder builder;
    for (const auto& [psgId, leaf] : root->programs_) {
        (void)psgId;
        auto iodescDict = leaf->GetTensorDataForLeafGraph();

        for (auto& op : leaf->Operations(false)) {
            if (!CheckEmuOpcode(&op, EMUOP_TENSOR_GETDATA_DEPEND)) {
                continue;
            }
            auto& copyInOp = op;
            copyInOp.SetAsDeleted();
            int tensorIndex = GetTensorDataGetIndex(&op);
            int addrCoaIndex = GetTensorDataGetCoaIndex(&op);
            if (tensorIndex == -1) {
                APASS_LOG_ERROR_F(Elements::Operation,
                                  "Attribute op_emuop_GetTensorData_index is not found for operation[%d]. %s",
                                  op.GetOpMagic(), GetFormatBacktrace(copyInOp).c_str());
                return FAILED;
            }
            if (addrCoaIndex == -1) {
                APASS_LOG_ERROR_F(Elements::Operation,
                                  "Attribute op_emuop_GetTensorData_coaIndex is not found for operation[%d]. %s",
                                  op.GetOpMagic(), GetFormatBacktrace(copyInOp).c_str());
                return FAILED;
            }
            auto incastIndex = leaf->GetIncastIndex(copyInOp.GetIOperands()[0]);
            auto desc = GetTensorDataIODesc(
                GET_TENSOR_DATA_OPERAND_IOTYPE_INCAST, incastIndex,
                GetParamAddrSymbol()(builder.CreateConstInt(-1), builder.CreateConstInt(addrCoaIndex)));
            iodescDict[tensorIndex] = desc;
        }
        leaf->GetTensorDataRefreshIO(iodescDict);
        leaf->EraseOperations(true, true);
    }

    return SUCCESS;
}

void SubgraphToFunction::DoHealthCheckAfter(Function& function, const std::string& folderPath)
{
    // 使用GetDumpFilePrefix生成前缀
    std::string prefix = GetDumpFilePrefix(function);

    // 构建完整路径：前缀 + 固定后缀
    std::string reportPath = folderPath + "/" + prefix + "_ExecuteGraph_Health_Report.json";

    // 生成并导出报告
    GenerateAndExportCombinedReport(function, psgToESgMap, nLIST, reportPath);
}

void SubgraphToFunction::GenerateAndExportCombinedReport(Function& func,
                                                         const std::multimap<int, int>& psgToESgMapParam,
                                                         const std::vector<std::vector<OperationPtr>>& subgraphGroups,
                                                         const std::string& filename)
{
    json report;
    ExecutionGraphStatistic execAnalyzer;
    report["execution_graph_analysis"] = execAnalyzer.AnalyzeExecutionGraph(func, psgToESgMapParam, subgraphGroups);

    // 写入文件
    std::ofstream outfile(filename);
    constexpr int JSON_INDENTATION_SPACES = 4;
    outfile << report.dump(JSON_INDENTATION_SPACES); // 4空格缩进
    outfile.close();
}

Status SubgraphToFunction::TransViewToCopyInBeforeGenSubgraph(Function& function)
{
    for (auto& op : function.Operations(false)) {
        if (op.GetOpcode() != Opcode::OP_VIEW) {
            continue;
        }
        if (!op.HasAttribute(OpAttributeKey::inplaceIdx)) {
            continue;
        }
        if (op.GetOOperands().size() != 1) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Operation[%d] is OP_VIEW. We Expect it has one OOperand but get %zu instead. %s",
                              op.GetOpMagic(), op.GetOOperands().size(), GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        auto oOperand = op.GetOutputOperand(0);
        bool canTrans = true;
        for (const auto& consumer : oOperand->GetConsumers()) {
            if (OpcodeManager::Inst().IsSharedMemory(consumer->GetOpcode())) {
                canTrans = false;
                break;
            }
            if (OpcodeManager::Inst().IsCopyIn(consumer->GetOpcode())) {
                return SUCCESS;
            }
        }
        if (!canTrans) {
            continue;
        }
        op.SetOpCode(Opcode::OP_COPY_IN);
        auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(op.GetOpAttribute().get());
        if (viewOpAttribute == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "OP Attribute is not found at Operation[%d]. %s", op.GetOpMagic(),
                              GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        viewToCopyInMapping_.emplace(&op, op.GetOpAttribute());
        op.SetOpAttribute(std::make_shared<CopyOpAttribute>(
            OpImmediate::Specified(viewOpAttribute->GetFromTensorOffset()), viewOpAttribute->GetTo(),
            OpImmediate::Specified(op.oOperand.front()->shape),
            OpImmediate::Specified(op.iOperand.front()->tensor->GetDynRawShape()),
            OpImmediate::Specified(viewOpAttribute->GetToDynValidShape())));
    }
    return SUCCESS;
}

Status SubgraphToFunction::RecoverCopyInToViewAfterGenSubgraph(Function& function)
{
    for (auto& program : function.rootFunc_->programs_) {
        for (auto& op : program.second->Operations(false)) {
            if (op.HasAttribute(OpAttributeKey::inplaceIdx) && op.GetOpcode() == Opcode::OP_COPY_IN) {
                if (viewToCopyInMapping_.count(&op) <= 0) {
                    APASS_LOG_ERROR_F(
                        Elements::Operation,
                        "Operation[%d] is not found after SubgrahToFunction. It exists before the pass. %s",
                        op.GetOpMagic(), GetFormatBacktrace(op).c_str());
                    return FAILED;
                }
                op.SetOpCode(Opcode::OP_VIEW);
                op.SetOpAttribute(viewToCopyInMapping_.at(&op));
            }
        }
    }

    return SUCCESS;
}

Status SubgraphToFunction::PreCheck(Function& function)
{
    SubGraphToFuncChecker checker;
    return checker.DoPreCheck(function);
}

Status SubgraphToFunction::PostCheck(Function& function)
{
    SubGraphToFuncChecker checker;
    if (function.GetFunctionType() == FunctionType::STATIC) {
        checker.SetInOutGraph(staticProcessor_.inGraph, staticProcessor_.outGraph);
        checker.SetColorGraph(staticProcessor_.colorInGraph, staticProcessor_.colorOutGraph);
    }
    return checker.DoPostCheck(function);
}

} // namespace npu::tile_fwk
