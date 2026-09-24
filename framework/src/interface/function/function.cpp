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
 * \file function.cpp
 * \brief
 */

#include "interface/function/function.h"
#include <queue>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include "interface/inner/pre_def.h"
#include "interface/cache/hash.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation.h"
#include "interface/tensor/tensor_offset.h"
#include "interface/utils/id_gen.h"
#include "tilefwk/data_type.h"
#include "tilefwk/symbolic_scalar.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/operation/attribute.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation_impl.h"
#include "interface/utils/serialization.h"
#include "interface/interpreter/flow_verifier.h"
#include "passes/pass_utils/subgraph_utils.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/pass_utils/graph_utils.h"
#include "interface/tensor/irbuilder.h"
#include "passes/tensor_graph_pass/expand_function.h"
#include "ir/transforms/printer.h"
using namespace npu::tile_fwk;

namespace {
const std::string PREFIX = "  ";
const int SPACE_NUM_THREE = 3;
const int LAST_TWO = -2;

const std::set<Opcode> SPECIAL_OPCODE_SET = {Opcode::OP_INDEX_OUTCAST, Opcode::OP_VIEW,     Opcode::OP_SLICE,
                                             Opcode::OP_ASSEMBLE,      Opcode::OP_CONTRACT, Opcode::OP_CALL,
                                             Opcode::OP_CONVERT,       Opcode::OP_COPY_IN,  Opcode::OP_COPY_OUT};
struct ViewKey {
    ViewKey(const int magic, const std::vector<int64_t>& newShape, const std::vector<int64_t>& newOffset,
            const std::vector<SymbolicScalar>& tmpDynOffset)
        : rawMagic(magic), shape(newShape), offset(newOffset), dynOffset(tmpDynOffset)
    {}

    bool operator<(const ViewKey& x) const
    {
        if (shape != x.shape) {
            return shape < x.shape;
        } else if (offset != x.offset) {
            return offset < x.offset;
        }
        if (dynOffset.size() != x.dynOffset.size()) {
            return dynOffset.size() < x.dynOffset.size();
        }
        for (size_t i = 0; i < dynOffset.size(); i++) {
            if (dynOffset[i].Raw() != x.dynOffset[i].Raw()) {
                return dynOffset[i].Raw() < x.dynOffset[i].Raw();
            }
        }

        return rawMagic < x.rawMagic;
    }

    int rawMagic;
    Shape shape;
    Offset offset;
    std::vector<SymbolicScalar> dynOffset;
};

struct TensorDependency {
    std::vector<int> producers;
    int pendingConsumerCount{0};
    bool hasSelfConsumer{false};
};

struct SortContext {
    std::unordered_map<const Operation*, int> opToIndex;
    std::unordered_map<const LogicalTensor*, size_t> tensorToDep;
    std::vector<TensorDependency> tensorDeps;
    std::vector<std::vector<size_t>> tensorDepsByConsumer;
    std::vector<std::vector<int>> tokenProducersByConsumer;
    std::vector<int> outDegree;
};

class LightweightOperationSorter {
public:
    LightweightOperationSorter(const Function& function, const std::vector<std::shared_ptr<Operation>>& operations,
                               const LogicalTensors& inCasts, const LogicalTensors& outCasts,
                               bool preserveOriginalOrder)
        : function_(function),
          operations_(operations),
          inCasts_(inCasts),
          outCasts_(outCasts),
          preserveOriginalOrder_(preserveOriginalOrder)
    {}

    std::vector<std::shared_ptr<Operation>> Sort()
    {
        // Cache each tensor's producers as a dependency barrier, then run Kahn's algorithm from consumers to
        // producers. A tensor barrier is released once all pending consumers are emitted, avoiding repeated
        // producer-list scans during lightweight sorting.
        BuildOpIndex();
        BuildTensorDeps();
        BuildTokenDeps();
        BuildOutDegree();
        auto sortedOperations = RunTopologicalSort();
        CheckSortedOperations(sortedOperations);
        std::reverse(sortedOperations.begin(), sortedOperations.end());
        return sortedOperations;
    }

private:
    void BuildOpIndex()
    {
        context_.opToIndex.reserve(operations_.size());
        for (size_t idx = 0; idx < operations_.size(); idx++) {
            auto op = operations_[idx].get();
            auto insertResult = context_.opToIndex.emplace(op, static_cast<int>(idx));
            FE_ASSERT(FeError::IS_EXIST, insertResult.second) << "Duplicate operation found: " << op->Dump();
        }
    }

    size_t GetOrCreateTensorDepIndex(const LogicalTensorPtr& tensor)
    {
        auto iter = context_.tensorToDep.find(tensor.get());
        if (iter != context_.tensorToDep.end()) {
            return iter->second;
        }
        size_t depIndex = context_.tensorDeps.size();
        context_.tensorToDep.emplace(tensor.get(), depIndex);
        TensorDependency dep;
        dep.producers.reserve(tensor->GetProducers().size());
        for (const auto& producer : tensor->GetProducers()) {
            if (producer->BelongTo() != &function_) {
                continue;
            }
            auto producerIter = context_.opToIndex.find(producer);
            FE_ASSERT(FeError::NOT_EXIST, producerIter != context_.opToIndex.end())
                << "Producer not found in opToIndex: " << producer->Dump();
            dep.producers.emplace_back(producerIter->second);
        }
        context_.tensorDeps.emplace_back(std::move(dep));
        return depIndex;
    }

    void RecordTensorDep(int consumerIdx, const LogicalTensorPtr& tensor)
    {
        size_t depIndex = GetOrCreateTensorDepIndex(tensor);
        auto& dep = context_.tensorDeps[depIndex];
        if (dep.producers.empty()) {
            return;
        }
        if (std::find(dep.producers.begin(), dep.producers.end(), consumerIdx) != dep.producers.end()) {
            dep.hasSelfConsumer = true;
        }
        dep.pendingConsumerCount++;
        context_.tensorDepsByConsumer[consumerIdx].emplace_back(depIndex);
    }

    void BuildTensorDeps()
    {
        context_.tensorToDep.reserve(operations_.size());
        context_.tensorDepsByConsumer.resize(operations_.size());
        for (size_t idx = 0; idx < operations_.size(); idx++) {
            const auto& op = operations_[idx];
            for (const auto& iop : op->iOperand) {
                RecordTensorDep(static_cast<int>(idx), iop);
            }
            for (const auto& dop : op->dependOperand) {
                RecordTensorDep(static_cast<int>(idx), dop);
            }
            if (!op->IsCall()) {
                for (auto [type, index] : GetTensorDataUsage(op->GetDynamicAttributeList())) {
                    if (type == GET_TENSOR_DATA_OPERAND_IOTYPE_INCAST) {
                        RecordTensorDep(static_cast<int>(idx), inCasts_[index]);
                    } else if (type == GET_TENSOR_DATA_OPERAND_IOTYPE_OUTCAST) {
                        RecordTensorDep(static_cast<int>(idx), outCasts_[index]);
                    }
                }
            }
        }
    }

    void BuildTokenDeps()
    {
        context_.tokenProducersByConsumer.resize(operations_.size());
        for (size_t idx = 0; idx < operations_.size(); idx++) {
            auto& producerIndexes = context_.tokenProducersByConsumer[idx];
            for (auto* producer : operations_[idx]->ProducerOpsByToken()) {
                if (producer == operations_[idx].get()) {
                    continue;
                }
                auto producerIter = context_.opToIndex.find(producer);
                if (producerIter != context_.opToIndex.end()) {
                    producerIndexes.emplace_back(producerIter->second);
                }
            }
            std::sort(producerIndexes.begin(), producerIndexes.end());
            producerIndexes.erase(std::unique(producerIndexes.begin(), producerIndexes.end()), producerIndexes.end());
        }
    }

    void BuildOutDegree()
    {
        context_.outDegree.assign(operations_.size(), 0);
        for (auto& dep : context_.tensorDeps) {
            if (dep.producers.empty() || dep.pendingConsumerCount == 0 || dep.hasSelfConsumer) {
                continue;
            }
            for (int producerIdx : dep.producers) {
                context_.outDegree[producerIdx]++;
            }
        }
        for (size_t consumerIdx = 0; consumerIdx < context_.tensorDepsByConsumer.size(); ++consumerIdx) {
            for (size_t depIndex : context_.tensorDepsByConsumer[consumerIdx]) {
                const auto& dep = context_.tensorDeps[depIndex];
                if (!dep.hasSelfConsumer) {
                    continue;
                }
                for (int producerIdx : dep.producers) {
                    if (producerIdx != static_cast<int>(consumerIdx)) {
                        context_.outDegree[producerIdx]++;
                    }
                }
            }
        }
        for (const auto& producerIndexes : context_.tokenProducersByConsumer) {
            for (int producerIdx : producerIndexes) {
                context_.outDegree[producerIdx]++;
            }
        }
    }

    std::vector<std::shared_ptr<Operation>> RunTopologicalSort()
    {
        std::vector<int> readyList;
        for (size_t idx = 0; idx < operations_.size(); idx++) {
            if (context_.outDegree[idx] == 0) {
                readyList.push_back(static_cast<int>(idx));
            }
        }
        if (!preserveOriginalOrder_) {
            // Sort initial ready ops by opmagic to ensure deterministic topological order.
            std::sort(readyList.begin(), readyList.end(),
                      [this](int a, int b) { return operations_[a]->GetOpMagic() < operations_[b]->GetOpMagic(); });
        }
        for (int idx : readyList) {
            PushReadyOp(idx);
        }
        auto releasePrevOp = [this](int prevOpIndex) {
            if (--context_.outDegree[prevOpIndex] == 0) {
                PushReadyOp(prevOpIndex);
            }
        };
        auto releaseTensorDep = [this, &releasePrevOp](size_t depIndex, int consumerIdx) {
            auto& dep = context_.tensorDeps[depIndex];
            if (dep.hasSelfConsumer) {
                for (int producerIdx : dep.producers) {
                    if (producerIdx != consumerIdx) {
                        releasePrevOp(producerIdx);
                    }
                }
                return;
            }
            if (--dep.pendingConsumerCount == 0) {
                for (int producerIdx : dep.producers) {
                    releasePrevOp(producerIdx);
                }
            }
        };
        std::vector<std::shared_ptr<Operation>> sortedOperations;
        sortedOperations.reserve(operations_.size());
        while (HasReadyOp()) {
            int currIndex = PopReadyOp();
            sortedOperations.emplace_back(operations_[currIndex]);
            for (size_t depIndex : context_.tensorDepsByConsumer[currIndex]) {
                releaseTensorDep(depIndex, currIndex);
            }
            for (int producerIdx : context_.tokenProducersByConsumer[currIndex]) {
                releasePrevOp(producerIdx);
            }
        }
        return sortedOperations;
    }

    void PushReadyOp(int opIndex)
    {
        if (preserveOriginalOrder_) {
            stableReadyOps_.push(opIndex);
        } else {
            readyOps_.push(opIndex);
        }
    }

    [[nodiscard]] bool HasReadyOp() const
    {
        return preserveOriginalOrder_ ? !stableReadyOps_.empty() : !readyOps_.empty();
    }

    int PopReadyOp()
    {
        if (preserveOriginalOrder_) {
            int opIndex = stableReadyOps_.top();
            stableReadyOps_.pop();
            return opIndex;
        }
        int opIndex = readyOps_.front();
        readyOps_.pop();
        return opIndex;
    }

    void CheckSortedOperations(const std::vector<std::shared_ptr<Operation>>& sortedOperations) const
    {
        for (auto& op : operations_) {
            const int opIndex = context_.opToIndex.at(op.get());
            FE_ASSERT(context_.outDegree[opIndex] == 0) << "cycle detected: " << op->Dump();
        }
        FE_ASSERT(operations_.size() == sortedOperations.size())
            << "Sorted operations size mismatch: " << sortedOperations.size() << " and original size "
            << operations_.size();
    }

    const Function& function_;
    const std::vector<std::shared_ptr<Operation>>& operations_;
    const LogicalTensors& inCasts_;
    const LogicalTensors& outCasts_;
    bool preserveOriginalOrder_;
    std::queue<int> readyOps_;
    std::priority_queue<int> stableReadyOps_;
    SortContext context_;
};

} // namespace

std::vector<Operation*> OperationsViewer::DuplicatedOpList() const
{
    std::vector<Operation*> opList;
    opList.reserve(operations_.size());
    for (auto op : operations_) {
        opList.emplace_back(op.get());
    }
    return opList;
}

std::string DynloopFunctionPathNode::Dump() const
{
    int indent = 2;
    std::ostringstream oss;
    std::function<void(const DynloopFunctionPathNode*, int)> dump =
        [&oss, &indent, &dump](const DynloopFunctionPathNode* node, int level) {
            if (!node->cond.IsValid()) {
                oss << std::setw(level * indent) << ' ' << node->root->GetRawName() << "("
                    << node->root->GetFunctionHash() << ")\n";
            } else {
                oss << std::setw(level * indent) << ' ' << node->cond.Dump() << "\n";
                if (node->branchNodeList[0] != nullptr) {
                    dump(node->branchNodeList[0].get(), level + 1);
                }
                if (node->branchNodeList[1] != nullptr) {
                    dump(node->branchNodeList[1].get(), level + 1);
                }
            }
        };
    dump(this, 0);
    return oss.str();
}

std::shared_ptr<DynloopFunctionPathNode> DynloopFunctionAttribute::BuildPathNode()
{
    std::shared_ptr<DynloopFunctionPathNode> root = std::make_shared<DynloopFunctionPathNode>();
    if (pathList.size() == 1) {
        root->root = pathList[0].GetRoot();
    } else {
        for (size_t i = 0; i < pathList.size(); i++) {
            auto node = root;
            for (size_t j = 0; j < pathList[i].pathCondList.size(); j++) {
                auto& pathCond = pathList[i].pathCondList[j];
                if (!node->cond.IsValid()) {
                    node->cond = pathCond.GetCond();
                }
                if (node->branchNodeList[pathCond.IsSat()] == nullptr) {
                    node->branchNodeList[pathCond.IsSat()] = std::make_shared<DynloopFunctionPathNode>();
                }
                node = node->branchNodeList[pathCond.IsSat()];
            }
            node->root = pathList[i].GetRoot();
        }
    }
    return root;
}

std::string DynloopFunctionAttribute::DumpBranch() const
{
    std::ostringstream oss;
    for (size_t i = 0; i < pathList.size(); i++) {
        auto& path = pathList[i];
        oss << "Branch-" << i << ": "
            << "\n";
        for (size_t j = 0; j < path.pathCondList.size(); j++) {
            auto& cond = path.pathCondList[j];
            oss << "  " << cond.GetFile() << ":" << cond.GetLine() << "] " << cond.GetCond().Dump() << ":"
                << cond.IsSat() << "\n";
        }
    }
    oss << "current:" << currIndex << "\n";
    for (size_t i = 0; i < currPathCond.size(); i++) {
        oss << "  " << currPathCond[i].GetFile() << ":" << currPathCond[i].GetLine() << "] "
            << currPathCond[i].GetCond().Dump() << ":" << currPathCond[i].IsSat() << "\n";
    }
    return oss.str();
}

std::vector<DynloopFunctionPathCondition> DynloopFunctionAttribute::GenCondWithBeginEnd(
    const std::vector<DynloopFunctionPathCondition>& conds) const
{
    std::vector<DynloopFunctionPathCondition> resultPathCond = conds;
    for (auto& cond : resultPathCond) {
        if (!cond.cond_.IsExpression()) {
            continue;
        }
        auto expr = std::static_pointer_cast<RawSymbolicExpression>(cond.cond_.Raw());
        if (expr->IsLoopEndCall()) {
            std::vector<RawSymbolicScalarPtr> operandList{
                expr->OperandList()[0], expr->OperandList()[1],
                RawSymbolicExpression::CreateBopSub(expr->OperandList()[2], originalRange.Step().Raw())};
            auto newExpr = std::make_shared<RawSymbolicExpression>(SymbolicOpcode::T_MOP_CALL, operandList);
            cond.cond_ = SymbolicScalar(newExpr);
        }
    }
    return resultPathCond;
}

bool DynloopFunctionAttribute::IterationEnd(int unroll, Function* pathFunc, Operation* operation)
{
    auto resultPathCond = GenCondWithBeginEnd(currPathCond);
    unrollTimes = unroll;
    pathList.emplace_back(pathFunc, resultPathCond, operation);

    bool finished = true;
    for (size_t idx = 0; idx < currPathCond.size(); idx++) {
        if (!currPathCond[idx].IsSat() && !currPathCond[idx].isConst_) {
            const auto& cond = currPathCond[idx].cond_;
            if (IsLoopBeginOrEndExpr(cond)) {
                if (!cond.IsLoopBegin() && !cond.IsLoopEnd()) {
                    continue;
                }
                if (std::static_pointer_cast<RawSymbolicExpression>(cond.Raw())->IsLoopBeginCall() &&
                    !cond.IsLoopBegin()) {
                    continue;
                }
                if (std::static_pointer_cast<RawSymbolicExpression>(cond.Raw())->IsLoopEndCall() && !cond.IsLoopEnd()) {
                    continue;
                }
            }
            finished = false;
            break;
        }
    }
    return finished;
}

bool DynloopFunctionAttribute::GuessCondResult(const SymbolicScalar& cond, bool& result)
{
    if (cond.ConcreteValid()) {
        result = cond.Concrete();
        return true;
    }
    auto condstr = cond.Dump();
    for (auto& pcond : currPathCond) {
        if (condstr == pcond.GetCond().Dump()) {
            result = pcond.IsSat();
            return true;
        }
    }
    return false;
}

bool DynloopFunctionAttribute::AppendCond(const SymbolicScalar& cond, const std::string& file, int line)
{
    bool result = false;
    if (currIndex < currPathCond.size()) {
        result = currPathCond[currIndex].IsSat();
    } else {
        bool isConst = GuessCondResult(cond, result);
        currPathCond.emplace_back(result, isConst, cond, file, line);
    }
    currIndex++;
    return result;
}

void DynloopFunctionAttribute::CreateCurrCond()
{
    if (pathList.size() == 0) {
        currPathCond.clear();
        currIndex = 0;
        return;
    }
    for (size_t idx = currPathCond.size() - 1; idx != static_cast<size_t>(-1); idx--) {
        if ((!currPathCond[idx].IsSat()) && (!currPathCond[idx].isConst_)) {
            const auto& cond = currPathCond[idx].cond_;
            if (cond.IsExpression()) {
                auto expr = std::static_pointer_cast<RawSymbolicExpression>(cond.Raw());
                if (expr->IsLoopBeginCall() && !cond.IsLoopBegin()) {
                    continue;
                }
                if (expr->IsLoopEndCall() && !cond.IsLoopEnd()) {
                    continue;
                }
            }
            currPathCond[idx].IsSat() = true;
            currPathCond.erase(currPathCond.begin() + idx + 1, currPathCond.end());
            break;
        }
    }
    currIndex = 0;
}

int Function::NextMagicNameSuffix(FunctionType funcType)
{
    if (funcType == FunctionType::DYNAMIC) {
        return IdGen<IdType::FUNCTION_MAGIC_NAME>::Inst().NewId();
    }
    return IdGen<IdType::FUNCTION>::Inst().CurId();
}

Function::Function(const Program& belongTo, const std::string& funcMagicName, const std::string& funcRawName,
                   Function* parentFunc)
    : ir::Function(ir::Span::Unknown()),
      funcMagicName_(funcMagicName),
      funcRawName_(funcRawName),
      tensorMap_(*this),
      belongTo_(belongTo)
{
    parent_ = parentFunc;
    functionMagic_ = IdGen<IdType::FUNCTION>::Inst().NewId();

    opSeed_ = FUNCTION_MAX_INCASTS;
}

OperationsViewer Function::Operations(bool sorted, SortOperationsMode mode)
{
    if (!sorted_ && sorted) {
        sorted_ = true;
        SortOperations(mode);
    }
    return OperationsViewer(operations_, opPosition_);
}

bool Function::IsCube() const
{
    for (const auto& oper : OperationsViewer(operations_, opPosition_)) {
        if ((oper.HasAttr(OpAttributeKey::isCube) && oper.GetBoolAttribute(OpAttributeKey::isCube)) ||
            oper.GetOpcode() == Opcode::OP_L1_COPY_IN_CONV) {
            return true;
        }
    }
    return false;
}

std::string Function::GetOriginalRawName() const
{
    const std::string& OriginalRawName = funcRawName_;
    size_t prefixLen = FUNCTION_PREFIX.length();
    if (OriginalRawName.substr(0, prefixLen) == FUNCTION_PREFIX) {
        return OriginalRawName.substr(prefixLen);
    }
    return OriginalRawName;
}

OperationsViewer Function::OperationsAfterOOO() { return OperationsViewer(operationsAfterOOO_, opPositionAfterOOO_); }

void Function::RecordOOOSeq()
{
    operationsAfterOOO_ = operations_;
    opPositionAfterOOO_ = opPosition_;
}

std::vector<OperationPtr>& Function::GetProgramOp()
{
    FE_ASSERT(FeError::INVALID_TYPE, graphType_ == GraphType::BLOCK_GRAPH)
        << "Function::GetProgramOp called. Current graph type: " << static_cast<int>(graphType_);
    return operations_;
}

void Function::SetProgramOp(const std::vector<OperationPtr>& operations)
{
    FE_ASSERT(FeError::INVALID_TYPE, graphType_ == GraphType::BLOCK_GRAPH)
        << "Function::SetProgramOp called. Current graph type: " << static_cast<int>(graphType_);
    operations_ = operations;

    RefreshOpPosition();
    sorted_ = true;
}

void Function::UpdateBelongToThis()
{
    FE_ASSERT(FeError::INVALID_TYPE, graphType_ == GraphType::BLOCK_GRAPH)
        << "Function::UpdateBelongToThis called. Current graph type: " << static_cast<int>(graphType_);
    for (auto& ele : operations_) {
        ele->function_ = this;
    }
}

const SubfuncInvokeInfoTy& Function::GetSubFuncInvokeInfo(const size_t i) const
{
    auto callAttr = std::dynamic_pointer_cast<CallOpAttribute>(operations_[i]->GetOpAttribute());
    FE_ASSERT(FeError::INVALID_PTR, callAttr != nullptr)
        << "Operation at index " << i << " must have a CallOpAttribute";
    return *(callAttr->invokeInfo_);
}

bool Function::HasCallOperation()
{
    for (const auto& op : Operations()) {
        if (op.GetOpcode() == Opcode::OP_CALL) {
            return true;
        }
    }
    return false;
}

void Function::CreateLeafInAndOutCast(const LogicalTensorPtr& inOrOut, LogicalTensors& inOrOutList) const
{
    inOrOutList.emplace_back(inOrOut->Clone(*parent_));
}

static int GetTensorDataLookupOutcast(Function* func, Operation* import)
{
    auto importTensor = import->GetIOperands()[0];
    auto consumerSet = importTensor->GetConsumers();
    if (consumerSet.size() != 2) {
        return INVALID_IOINDEX;
    }
    for (auto consumer : consumerSet) {
        if (consumer != import) {
            auto outcast = consumer->GetOOperands()[0];
            auto outcastIndex = func->GetOutcastIndex(outcast);
            return outcastIndex;
        }
    }
    return INVALID_IOINDEX;
}

static int GetTensorDataLookupIncast(Function* func, Operation* import)
{
    auto importTensor = import->GetIOperands()[0];
    auto producerSet = importTensor->GetProducers();
    if (producerSet.size() != 1) {
        return INVALID_IOINDEX;
    }
    auto producer = *producerSet.begin();
    auto incast = producer->GetIOperands()[0];
    auto incastIndex = func->GetIncastIndex(incast);
    return incastIndex;
}

GetTensorDataIODescDict Function::GetTensorDataForTensorGraph()
{
    GetTensorDataIODescDict iodescDict;
    auto currDynFunc = Program::GetInstance().GetCurrentDynamicFunction();
    if (currDynFunc == nullptr) {
        return iodescDict;
    }
    auto currDynAttr = currDynFunc->GetDyndevAttribute();
    for (auto& op : Operations(false)) {
        if (!CheckEmuOpcode(&op, EMUOP_TENSOR_GETDATA_IMPORT)) {
            continue;
        }
        int index = GetTensorDataGetIndex(&op);
        FE_ASSERT(FeError::INVALID_VAL, index != -1) << "Invalid import index";
        int outcastIndex = GetTensorDataLookupOutcast(this, &op);
        if (outcastIndex != INVALID_IOINDEX) {
            iodescDict[index] = GetTensorDataIODesc(GET_TENSOR_DATA_OPERAND_IOTYPE_OUTCAST, outcastIndex, 0);
        } else {
            int incastIndex = GetTensorDataLookupIncast(this, &op);
            FE_ASSERT(FeError::INVALID_VAL, incastIndex != INVALID_IOINDEX)
                << "Both outcast and incast indices are invalid";
            iodescDict[index] = GetTensorDataIODesc(GET_TENSOR_DATA_OPERAND_IOTYPE_INCAST, incastIndex, 0);
        }
    }
    return iodescDict;
}

GetTensorDataIODescDict Function::GetTensorDataForLeafGraph()
{
    GetTensorDataIODescDict iodescDict;
    for (auto& op : Operations(false)) {
        if (!CheckEmuOpcode(&op, EMUOP_TENSOR_GETDATA_IMPORT)) {
            continue;
        }
        int getTensorDataIndex = GetTensorDataGetIndex(&op);
        FE_ASSERT(FeError::INVALID_VAL, getTensorDataIndex != -1) << "Failed to get tensor data index for operation";
        auto tensor = op.GetIOperands()[0];
        auto incastIndex = GetIncastIndex(tensor);
        if (incastIndex != INVALID_IOINDEX) {
            iodescDict[getTensorDataIndex] = GetTensorDataIODesc(GET_TENSOR_DATA_OPERAND_IOTYPE_INCAST, incastIndex, 0);
        }
    }
    return iodescDict;
}

void Function::GetTensorDataRefreshIO(const GetTensorDataIODescDict& iodescDict)
{
    for (auto& op : Operations(false)) {
        std::vector<std::reference_wrapper<SymbolicScalar>> dynamicAttributeList = op.GetDynamicAttributeList();
        for (auto& attr : dynamicAttributeList) {
            attr.get() = GetTensorDataFillIO(iodescDict, attr.get());
        }
    }
    std::set<std::shared_ptr<LogicalTensor>> tensorSet;
    for (auto& incast : inCasts_) {
        tensorSet.insert(incast);
    }
    for (auto& outcast : outCasts_) {
        tensorSet.insert(outcast);
    }
    for (auto& op : Operations(false)) {
        for (auto& iOperand : op.GetIOperands()) {
            tensorSet.insert(iOperand);
        }
        for (auto& oOperand : op.GetOOperands()) {
            tensorSet.insert(oOperand);
        }
    }
    for (auto& tensor : tensorSet) {
        for (auto& shape : tensor->GetDynValidShape()) {
            shape = GetTensorDataFillIO(iodescDict, shape);
        }
        if (tensor->GetDynOffset().size() != 0) {
            std::vector<SymbolicScalar> refreshedDynOffset;
            for (auto& offset : tensor->GetDynOffset()) {
                refreshedDynOffset.push_back(GetTensorDataFillIO(iodescDict, offset));
            }
            tensor->UpdateOffset(TensorOffset(tensor->GetOffset(), refreshedDynOffset));
        }
    }
}

void Function::BeginFunction(const std::vector<std::reference_wrapper<const Tensor>>& explicitOpArgs)
{
    auto slotManager = Program::GetInstance().GetTensorSlotManager();
    for (auto& arg : explicitOpArgs) {
        explicitArgSlots_.push_back(TensorSlot::CreateTensor(arg));
    }
}

static bool IsAssembleOutcast(Function& func, size_t outcastIdx)
{
    auto& outcasts = func.GetOutcast();
    auto slots = func.GetOutCastSlot(outcasts[outcastIdx]);
    auto slotMngr = Program::GetInstance().GetTensorSlotManager();
    for (const auto& slot : slotMngr->assembleSlotSet) {
        auto it = slotMngr->slotIndexDict.find(slot);
        if (it == slotMngr->slotIndexDict.end()) {
            continue;
        }
        std::vector<int> assembleSlot = {it->second};
        // output use as assemble later, it's not removable
        if (TensorSlotManager::HasSameSlot(slots, assembleSlot)) {
            return true;
        }
    }
    return false;
}

bool HasCalleeConsumer(Function& func, Function& calleeFunc, size_t outcastIdx)
{
    auto outcast = calleeFunc.GetOutcast()[outcastIdx];
    auto outcastSlots = calleeFunc.GetOutCastSlot(outcast);
    bool isAssemble = IsAssembleOutcast(calleeFunc, outcastIdx);

    for (auto otherCallee : func.GetCalleeFunctionList()) {
        if (otherCallee == &calleeFunc) {
            continue;
        }
        for (auto& incast : otherCallee->GetIncast()) {
            auto incastSlots = otherCallee->GetInCastSlot(incast);
            if (TensorSlotManager::HasSameSlot(incastSlots, outcastSlots)) {
                return true;
            }
        }
        if (isAssemble) {
            for (auto& out : otherCallee->GetOutcast()) {
                auto outSlots = otherCallee->GetOutCastSlot(out);
                if (TensorSlotManager::HasSameSlot(outSlots, outcastSlots)) {
                    return true;
                }
            }
        }
    }

    return false;
}

static bool IsRemovableOutcast(Function& func, size_t outcastIdx)
{
    auto& outcasts = func.GetOutcast();
    auto slots = func.GetOutCastSlot(outcasts[outcastIdx]);
    for (auto& slot : slots) {
        // outcast create before the function, it's not removable
        if (func.GetSlotScope()->IsAliveSlot(slot)) {
            return false;
        }
    }

    return true;
}

void CalleeSlotNoConsumer(Function& calleeFunc, Function& func, const std::map<size_t, size_t>& outcasts,
                          std::map<size_t, size_t>& outcastIdx2parent)
{
    for (size_t calleeOutcastIdx = 0; calleeOutcastIdx < calleeFunc.GetOutcast().size(); calleeOutcastIdx++) {
        auto caleeOutcast = calleeFunc.GetOutcast()[calleeOutcastIdx];
        auto incastSlots = calleeFunc.GetOutCastSlot(caleeOutcast);
        for (const auto& [outcastIdx, val] : outcasts) {
            (void)val;
            auto outcast = func.GetOutcast()[outcastIdx];
            auto outcastSlots = func.GetOutCastSlot(outcast);
            if (TensorSlotManager::HasSameSlot(incastSlots, outcastSlots) &&
                !HasCalleeConsumer(func, calleeFunc, calleeOutcastIdx)) {
                outcastIdx2parent[calleeOutcastIdx] = outcastIdx;
                break;
            }
        }
    }
}

void Function::EraseCallOpOpnd(const FunctionHash& calleeHash, size_t index)
{
    for (auto callop : GetCallopList()) {
        auto callopAttr = std::static_pointer_cast<CallOpAttribute>(callop->GetOpAttribute());
        FE_ASSERT(FeError::INVALID_PTR, callopAttr != nullptr) << "Processing CallOp:" << callop->Dump();
        if (callopAttr->GetCalleeHash() != calleeHash) {
            continue;
        }
        FE_ASSERT(FeError::INVALID_VAL, index < callop->oOperand.size())
            << "Index " << index << " out of bounds for oOperand size " << callop->oOperand.size();
        FE_ASSERT(callop->GetOOpAttr().empty()) << "oOpAttrOffset is not empty for CallOp:" << callop->Dump();
        FE_ASSERT(callopAttr->GetArgList().empty()) << "ArgList is not empty for CallOp:" << callop->Dump();
        FE_ASSERT(callopAttr->GetOutCastIndexToExpr().empty())
            << "OutCastIndexToExpr is not empty for CallOp:" << callop->Dump();
        for (auto& consumer : callop->oOperand[index]->GetConsumers()) {
            if (IsAssembleLike(consumer->GetOpcode())) {
                consumer->SetAsDeleted();
            }
        }
        callop->oOperand.erase(callop->oOperand.begin() + index);
    }
    EraseOperations(true, false);
}

void Function::CheckAndUpdateGetTensorData(size_t currOutcastIdx, size_t newOutcastIdx)
{
    for (auto& op : Operations(false)) {
        if (!op.IsCall()) {
            for (auto& attr : op.GetDynamicAttributeList()) {
                attr.get() = UpdateGetTensorDataIOIndex(currOutcastIdx, newOutcastIdx, attr.get());
            }
        }
    }
}

void Function::CleanRedundantOutcast(std::map<Function*, std::set<size_t>>& removeRecord,
                                     std::map<Function*, std::set<size_t>>& getTensorDataRecord)
{
    for (auto& [func, removeList] : removeRecord) {
        for (auto it = removeList.rbegin(); it != removeList.rend(); ++it) {
            auto outCastIdx = *it;
            func->Parent().EraseCallOpOpnd(func->GetFunctionHash(), outCastIdx);
            func->RemoveOutcast(outCastIdx);
        }
        if (getTensorDataRecord.count(func) <= 0) {
            continue;
        }
        auto& tensorDataList = getTensorDataRecord[func];
        for (auto currOutcastIdx : tensorDataList) {
            auto it = std::lower_bound(removeList.begin(), removeList.end(), currOutcastIdx);
            size_t newOutcastIdx = currOutcastIdx - std::distance(removeList.begin(), it);
            if (currOutcastIdx != newOutcastIdx) {
                func->CheckAndUpdateGetTensorData(currOutcastIdx, newOutcastIdx);
            }
        }
    }
}

void RedundantOutCastCheck(std::map<Function*, std::set<size_t>>& removeRecord,
                           std::map<Function*, std::set<size_t>>& getTensorDataRecord, Function* func,
                           std::map<size_t, size_t>& outcasts)
{
    for (auto calleeFunc : func->GetCalleeFunctionList()) {
        FE_ASSERT(FeError::INVALID_PTR, calleeFunc != nullptr) << func->GetMagicName() << " has nullptr calleeFunc";
        std::map<size_t, size_t> outcastIdx2parent; // key: callee outcastIdx, value: caller outcastIdx
        CalleeSlotNoConsumer(*calleeFunc, *func, outcasts, outcastIdx2parent);
        if (!outcastIdx2parent.empty()) {
            RedundantOutCastCheck(removeRecord, getTensorDataRecord, calleeFunc, outcastIdx2parent);
        }
        auto& calleeOutCasts = calleeFunc->GetOutcast();
        for (auto& [outCastIdx, val] : outcastIdx2parent) {
            (void)val;
            FE_ASSERT(FeError::INVALID_VAL, calleeOutCasts[outCastIdx].get() != nullptr)
                << "Outcast at index " << outCastIdx << " should not be null";
            if (calleeOutCasts[outCastIdx]->IsGetTensorDataOutcast()) {
                getTensorDataRecord[calleeFunc].insert(outCastIdx);
                FE_ASSERT(FeError::INVALID_VAL, outcastIdx2parent.count(outCastIdx) > 0)
                    << "Outcast index " << outCastIdx << " should be in outcastIdx2parent";
                getTensorDataRecord[func].insert(outcastIdx2parent[outCastIdx]);
            } else if (getTensorDataRecord[calleeFunc].count(outCastIdx) > 0) {
                FE_ASSERT(FeError::INVALID_VAL, outcastIdx2parent.count(outCastIdx) > 0)
                    << "Outcast index " << outCastIdx << " should be in outcastIdx2parent";
                getTensorDataRecord[func].insert(outcastIdx2parent[outCastIdx]);
            } else if (IsRemovableOutcast(*calleeFunc, outCastIdx)) {
                removeRecord[calleeFunc].insert(outCastIdx);
            }
        }
    }
}

void Function::CleanRedundantOutCast()
{
    auto slotMngr = Program::GetInstance().GetTensorSlotManager();
    std::vector<int> outputSlots;
    for (const auto& slot : slotMngr->outputSlotList) {
        outputSlots.push_back(slotMngr->slotIndexDict[slot]);
    }

    // support flatten input output
    if (slotMngr->outputSlotList.size() == 0) {
        for (const auto& slot : slotMngr->inputSlotList) {
            outputSlots.push_back(slotMngr->slotIndexDict[slot]);
        }
    }
    std::map<Function*, std::set<size_t>> removeRecord;
    std::map<Function*, std::set<size_t>> getTensorDataRecord;
    auto calleeLists = GetCalleeFunctionList();
    auto& calleeOutCasts = GetOutcast();
    std::map<size_t, size_t> outputMap;
    for (size_t outCastIdx = 0; outCastIdx < calleeOutCasts.size(); outCastIdx++) {
        auto outcastSlots = GetOutCastSlot(calleeOutCasts[outCastIdx]);
        if ((!TensorSlotManager::HasSameSlot(outputSlots, outcastSlots)) &&
            !HasCalleeConsumer(Parent(), *this, outCastIdx)) {
            outputMap[outCastIdx] = 0;
        }
    }
    if (!outputMap.empty()) {
        RedundantOutCastCheck(removeRecord, getTensorDataRecord, this, outputMap);
    }
    for (auto& [outCastIdx, val] : outputMap) {
        (void)val;
        if (getTensorDataRecord[this].count(outCastIdx) > 0) {
            FE_ASSERT(FeError::NOT_EXIST, outputMap.count(outCastIdx) > 0)
                << "outputMap does not contain outCastIdx " << outCastIdx;
            getTensorDataRecord[parent_].insert(outputMap[outCastIdx]);
        } else {
            if (IsRemovableOutcast(*this, outCastIdx)) {
                removeRecord[this].insert(outCastIdx);
            }
        }
    }
    CleanRedundantOutcast(removeRecord, getTensorDataRecord);
}

void Function::InferParamDirection()
{
    FE_ASSERT(FeError::INVALID_PTR, dyndevAttr_ != nullptr) << "dyndevAttr_ is empty";
    FE_ASSERT(FeError::INVALID_PTR, slotScope_ != nullptr) << "slotscope is empty";

    std::map<int, int> slotAttr;
    for (size_t i = 0; i < slotScope_->ioslot.outcastSlot.size(); i++) {
        for (auto k : slotScope_->ioslot.outcastSlot[i]) {
            slotAttr[k] |= (int)ParamDirection::OUT;
        }
    }
    for (size_t i = 0; i < slotScope_->ioslot.incastSlot.size(); i++) {
        for (auto k : slotScope_->ioslot.incastSlot[i]) {
            slotAttr[k] |= (int)ParamDirection::IN;
        }
    }

    auto& startArgs = dyndevAttr_->startArgsInputTensorList;
    auto& directionList = dyndevAttr_->startArgsDirectionList;
    for (auto& t : startArgs) {
        auto attr = slotAttr[t.get().Id()];
        // if attr is 0, it maybe not used or used by getinputdata only, treat it as input
        auto direction = attr ? (ParamDirection)attr : ParamDirection::IN;
        directionList.push_back(direction);
    }
}

void Function::FillOriginInOutCast(std::vector<Operation*>& operationList)
{
    std::unordered_set<LogicalTensorPtr> visited;
    auto addOrigin = [](auto& t, auto& list) {
        for (auto& ele : list) {
            if (ele == t) {
                return;
            }
        }
        list.push_back(t);
    };
    for (auto& op : operationList) {
        for (auto& iOperand : op->iOperand) {
            if (visited.count(iOperand) != 0) {
                continue;
            }
            visited.insert(iOperand);
            if (op->IsCall() || &iOperand->BelongFunction() != this) {
                addOrigin(iOperand, originInCasts_);
            }
        }
        for (auto& oOperand : op->oOperand) {
            if (op->IsCall() || oOperand->tensor->GetRefCount()) {
                addOrigin(oOperand, originOutCasts_);
            }
            visited.insert(oOperand);
        }
    }
}

FunctionCallArgs Function::EndFunction(const std::shared_ptr<TensorSlotScope>& scope)
{
    // Deduce Incast and Outcast here, need by TENSOR_GRAPH & STATIC_TILE_GRAPH
    std::vector<Operation*> operationList = Operations(false).DuplicatedOpList();
    if (IsGraphType(GraphType::TENSOR_GRAPH) ||
        IsFunctionTypeAndGraphType(FunctionType::STATIC, GraphType::TILE_GRAPH)) {
        FillOriginInOutCast(operationList);
    }

    LogicalTensors inArgumentList, outArgumentList;
    if (IsGraphType(GraphType::TENSOR_GRAPH) ||
        IsFunctionTypeAndGraphType(FunctionType::STATIC, GraphType::TILE_GRAPH)) {
        SetCallOpSlot();
        CreateTensorDataImportOp();
        inArgumentList = MakeIncasts(scope);
        outArgumentList = MakeOutcasts(scope);
        auto iodescDict = GetTensorDataForTensorGraph();
        GetTensorDataRefreshIO(iodescDict);
        sorted_ = true;
        if (Program::GetInstance().GetCurrentDynamicFunction()) {
            DyndevFunctionAttribute::ValueDependDesc desc = LookupValueDepend();
            auto currDynFuncAttr = Program::GetInstance().GetCurrentDynamicFunction()->GetDyndevAttribute();
            if (currDynFuncAttr != nullptr) {
                currDynFuncAttr->valueDependDescDict[this] = desc;
            }
        }
    } else if (graphType_ == GraphType::EXECUTE_GRAPH) {
    } else if (graphType_ == GraphType::BLOCK_GRAPH) {
        for (auto& out : outCasts_) {
            CreateLeafInAndOutCast(out, outArgumentList);
        }
        for (auto& in : inCasts_) {
            /* actualRawmagic存在的场景下，前序 LogicalTensor 应该已经创建好，直接获取 */
            CreateLeafInAndOutCast(in, inArgumentList);
        }
        for (const auto& op : operations_) {
            opSeed_ = std::max(opSeed_, op->GetOpMagic() + 1);
        }
    } else {
        FE_ASSERT(FeError::INVALID_TYPE, false) << "Not support connecting other type of function currently";
    }
    std::vector<OperandAttribute> iOpAttr;
    std::vector<OperandAttribute> oOpAttr;
    std::map<int, SymbolicScalar> outIndexToExpr;
    std::vector<std::vector<SymbolicScalar>> argList;
    if (graphType_ == GraphType::BLOCK_GRAPH) {
        argList = NormalizeCoa(iOpAttr, oOpAttr);
        GetOutcastSymbolicExpr(outIndexToExpr);
    }
    BuildTensorMap();
    ComputeHash();
    return {std::move(inArgumentList), std::move(outArgumentList), std::move(iOpAttr),
            std::move(oOpAttr),        std::move(outIndexToExpr),  std::move(argList)};
}

void Function::AddWhenNotExistOrAssert(const std::shared_ptr<LogicalTensor>& tensor,
                                       std::map<int, int>& magicToRawMagic,
                                       std::map<int, std::shared_ptr<LogicalTensor>>& magicToLogicalTensor)
{
    if (auto it = magicToRawMagic.find(tensor->magic); it != magicToRawMagic.end()) {
        if (it->second != tensor->tensor->GetRawMagic()) {
            FE_LOGI("Diff Magic Same RawMagic: %d %s %d %s", it->second,
                    magicToLogicalTensor[tensor->magic]->Dump().c_str(), tensor->tensor->GetRawMagic(),
                    tensor->Dump().c_str());
        }
    }
    magicToRawMagic[tensor->magic] = tensor->tensor->GetRawMagic();
    magicToLogicalTensor[tensor->magic] = tensor;
}

// Tensor magic should be only in the function, so same magic have same rawmagic
void Function::TensorMagicCheck() const
{
    std::map<int, int> magicToRawMagic;
    std::map<int, std::shared_ptr<LogicalTensor>> magicToLogicalTensor;
    for (const auto& op : operations_) {
        std::map<int, int> subGraphIDCount;
        // Count subGraphID occurrences in iOperand
        for (const auto& tensor : op->iOperand) {
            AddWhenNotExistOrAssert(tensor, magicToRawMagic, magicToLogicalTensor);
        }

        // Count subGraphID occurrences in oOperand
        for (const auto& tensor : op->oOperand) {
            AddWhenNotExistOrAssert(tensor, magicToRawMagic, magicToLogicalTensor);
        }
    }
}

void Function::OperationLoopCheck(const std::string& errorMsg)
{
    std::map<LogicalTensor*, std::vector<Operation*>> producers;
    std::map<LogicalTensor*, std::vector<Operation*>> consumers;

    for (auto&& op : operations_) {
        for (auto&& iop : op->GetIOperands()) {
            consumers[iop.get()].emplace_back(op.get());
        }
        for (auto&& oop : op->GetOOperands()) {
            producers[oop.get()].emplace_back(op.get());
        }
    }

    enum class DfsState {
        TODO = 0,
        IN_STACK,
        DONE,
    };

    std::map<int, DfsState> states;
    for (auto&& op : operations_) {
        int dupOpMagic = -1;
        auto cycleDetection = [&states, &dupOpMagic, &consumers](Operation* curr, auto self) -> bool {
            int magic = curr->GetOpMagic();
            if (states[magic] == DfsState::DONE) {
                return false;
            }

            if (states[magic] == DfsState::IN_STACK) {
                dupOpMagic = magic;
                FE_LOGE(InternalError::FE_INNER_ERROR, "[OperationLoopCheck] Cycle detected: ");
                FE_LOGE(InternalError::FE_INNER_ERROR, "[OperationLoopCheck]     Operation: %s", curr->Dump().c_str());
                return true;
            }

            states[magic] = DfsState::IN_STACK;

            for (auto&& oop : curr->GetOOperands()) {
                for (auto* consumer : consumers[oop.get()]) {
                    if (self(consumer, self)) {
                        if (dupOpMagic != -1) {
                            FE_LOGE(InternalError::FE_INNER_ERROR, "[OperationLoopCheck]     Tensor:    %s",
                                    oop->Dump().c_str());
                            FE_LOGE(InternalError::FE_INNER_ERROR, "[OperationLoopCheck]     Operation: %s",
                                    curr->Dump().c_str());
                            if (magic == dupOpMagic) {
                                dupOpMagic = -1; // stop dumpping
                            }
                        }
                        return true;
                    }
                }
            }

            states[magic] = DfsState::DONE;

            return false;
        };

        FE_ASSERT(!cycleDetection(op.get(), cycleDetection)) << errorMsg;
    }
}

bool Function::OperationLoopCheck()
{
    std::unordered_map<Operation*, int> inLinkNum;
    std::unordered_set<Operation*> visitedOp;
    std::vector<Operation*> visitStack;
    for (std::shared_ptr<Operation> op : operations_) {
        inLinkNum[op.get()] = op->ProducerOps().size();
        if (inLinkNum[op.get()] == 0) {
            visitStack.push_back(op.get());
        }
    }
    while (!visitStack.empty()) {
        Operation* currOp = visitStack.back();
        visitStack.pop_back();
        visitedOp.insert(currOp);
        for (Operation* nextOp : currOp->ConsumerOps()) {
            inLinkNum[nextOp] -= 1;
            if (inLinkNum[nextOp] == 0) {
                visitStack.push_back(nextOp);
            }
            if (inLinkNum[nextOp] < 0) {
                FE_LOGE(InternalError::FE_INNER_ERROR, "[OperationLoopCheck]     Operation:%s", nextOp->Dump().c_str());
                return false;
            }
        }
    }
    if (visitedOp.size() != operations_.size()) {
        FE_LOGE(InternalError::FE_INNER_ERROR, "[OperationLoopCheck]     Loop Detected.");
        return false;
    }
    return true;
}

void Function::GetAnIslandIncastsOutcasts(const std::map<int, int>& opToSubgraph, const int subgraphID,
                                          const std::vector<Operation*>& operations,
                                          std::vector<std::shared_ptr<LogicalTensor>>& iOperands,
                                          std::vector<std::shared_ptr<LogicalTensor>>& oOperands) const
{
    std::set<std::shared_ptr<LogicalTensor>> allLogicalTensors;
    std::set<std::shared_ptr<LogicalTensor>> notOutcasts;
    std::set<std::shared_ptr<LogicalTensor>> notIncasts;
    for (const auto& opPtr : operations) {
        const auto& op = *opPtr;
        for (auto&& operand : op.GetIOperands()) {
            allLogicalTensors.insert(operand);
            bool usedbyotherfunction = false;
            for (auto& consumer : operand->GetConsumers()) {
                auto magic = consumer->GetOpMagic();
                if (consumer->GetOpcode() == Opcode::OP_CALL) {
                    continue;
                }
                FE_ASSERT(FeError::NOT_EXIST, opToSubgraph.find(magic) != opToSubgraph.end())
                    << "Consumer magic " << magic << " not found in opToSubgraph. "
                    << "\n"
                    << "Operation: " << op.Dump();

                if (opToSubgraph.at(magic) != subgraphID) {
                    usedbyotherfunction = true;
                    break;
                }
            }
            if (!usedbyotherfunction) {
                notOutcasts.insert(operand);
            }
        }

        for (auto&& operand : op.GetOOperands()) {
            allLogicalTensors.insert(operand);
            notIncasts.insert(operand);
        }
    }
    std::set_difference(allLogicalTensors.begin(), allLogicalTensors.end(), notIncasts.begin(), notIncasts.end(),
                        std::inserter(iOperands, iOperands.begin()));
    std::set<std::shared_ptr<LogicalTensor>> tryOOperands;
    std::set_difference(allLogicalTensors.begin(), allLogicalTensors.end(), notOutcasts.begin(), notOutcasts.end(),
                        std::inserter(tryOOperands, tryOOperands.begin()));
    std::set_difference(tryOOperands.begin(), tryOOperands.end(), iOperands.begin(), iOperands.end(),
                        std::inserter(oOperands, oOperands.begin()));

    std::sort(iOperands.begin(), iOperands.end(), TensorPtrComparator());
    std::sort(oOperands.begin(), oOperands.end(), TensorPtrComparator());
}

auto Function::AnnotateOperation()
{
    std::map<int, std::vector<Operation*>> subgraphs;
    std::map<int, int> opToSubgraph;
    for (auto&& op : Operations()) {
        // same op magic shall only appear once
        FE_ASSERT(FeError::IS_EXIST, opToSubgraph.find(op.GetOpMagic()) == opToSubgraph.end())
            << "Same op magic shall only appear once."
            << "\n"
            << "Duplicate OpMagic found: " << op.GetOpMagic() << "\n"
            << "Operation: " << op.Dump();
        if (op.GetSubgraphID() < 0) {
            FE_LOGD("Op magic: %d less than 0 graph: %d", op.GetOpMagic(), op.GetSubgraphID());
            continue;
        }
        subgraphs[op.GetSubgraphID()].emplace_back(&op);
        opToSubgraph[op.GetOpMagic()] = op.GetSubgraphID();
        FE_LOGD("Operation: %d Belong To subgraph: %d", op.GetOpMagic(), op.GetSubgraphID());
    }

    for (const auto& pair : subgraphs) {
        FE_LOGD("Subgraph ID: %d", pair.first);
        for (const auto& op : pair.second) {
            FE_LOGD("Operation: %s", op->Dump().c_str());
        }
    }
    return std::make_pair(std::move(subgraphs), std::move(opToSubgraph));
}

std::unordered_set<int> Function::LoopCheck()
{
    if (totalSubGraphCount_ == 0) {
        return {};
    }
    FE_LOGI("LoopCheck begin.");

    auto [subgraphs, opToSubgraph] = AnnotateOperation();
    std::map<LogicalTensor*, std::vector<int>> producers;
    std::map<LogicalTensor*, std::vector<int>> consumers;

    std::map<int, std::vector<std::shared_ptr<LogicalTensor>>> iOperands;
    std::map<int, std::vector<std::shared_ptr<LogicalTensor>>> oOperands;

    for (auto&& [subgraphID, operations] : subgraphs) {
        if (subgraphID == NOT_IN_SUBGRAPH) {
            continue;
        }

        GetAnIslandIncastsOutcasts(opToSubgraph, subgraphID, operations, iOperands[subgraphID], oOperands[subgraphID]);

        for (auto&& iop : iOperands[subgraphID]) {
            consumers[iop.get()].push_back(subgraphID);
        }
        for (auto&& oop : oOperands[subgraphID]) {
            producers[oop.get()].push_back(subgraphID);
        }
    }

    enum class DfsState {
        TODO = 0,
        IN_STACK,
        DONE,
    };

    std::map<int, DfsState> states;
    std::unordered_set<int> subGraphInCycle;
    for (auto&& [subgraphID, operations] : subgraphs) {
        (void)operations;
        if (subgraphID == NOT_IN_SUBGRAPH) {
            continue;
        }

        int duplicatedSubgraphID = -2;
        auto cycleDetection = [&states, &duplicatedSubgraphID, &oOperands, &consumers, &subGraphInCycle](
                                  int currSubgraph, auto self) -> bool {
            if (states[currSubgraph] == DfsState::DONE) {
                return false;
            }

            if (states[currSubgraph] == DfsState::IN_STACK) {
                duplicatedSubgraphID = currSubgraph;
                FE_LOGE(InternalError::FE_INNER_ERROR, "[Cycle Detection] Cycle detected: ");
                FE_LOGE(InternalError::FE_INNER_ERROR, "[Cycle Detection]     subgraph id: %d", currSubgraph);
                subGraphInCycle.emplace(currSubgraph);
                return true;
            }

            states[currSubgraph] = DfsState::IN_STACK;

            for (auto&& oop : oOperands[currSubgraph]) {
                for (int consumer : consumers[oop.get()]) {
                    if (self(consumer, self)) {
                        if (duplicatedSubgraphID != -2) {
                            FE_LOGE(InternalError::FE_INNER_ERROR, "[Cycle Detection]     tensor:      %s",
                                    oop->Dump().c_str());
                            FE_LOGE(InternalError::FE_INNER_ERROR, "[producer]=");
                            for (const auto& producer : oop->GetProducers()) {
                                FE_LOGE(InternalError::FE_INNER_ERROR, "%d", producer->GetOpMagic());
                            }
                            FE_LOGE(InternalError::FE_INNER_ERROR, "[Cycle Detection]     subgraph id: %d",
                                    currSubgraph);
                            subGraphInCycle.emplace(currSubgraph);
                            if (currSubgraph == duplicatedSubgraphID) {
                                duplicatedSubgraphID = -2; // stop dumpping
                            }
                        }
                        return true;
                    }
                }
            }

            states[currSubgraph] = DfsState::DONE;
            return false;
        };
        if (cycleDetection(subgraphID, cycleDetection)) {
            return subGraphInCycle;
        }
    }
    return std::unordered_set<int>{};
}

std::vector<std::shared_ptr<Operation>> Function::GetSortedOperations() const
{
    std::unordered_map<const Operation*, int> opToIndex;
    std::unordered_map<const Operation*, std::set<std::pair<int, int>>> usageDict;

    for (size_t idx = 0; idx < operations_.size(); idx++) {
        auto op = operations_[idx].get();
        FE_ASSERT(FeError::IS_EXIST, opToIndex.count(op) == 0) << "Duplicate operation found: " << op->Dump();
        opToIndex.emplace(op, idx);
        if (!op->IsCall()) {
            auto attrList = op->GetDynamicAttributeList();
            usageDict[op] = GetTensorDataUsage(attrList);
        }
    }
    std::vector<int> outDegree(operations_.size(), 0);
    std::vector<int> prevOperation(operations_.size(), -1);

    auto addProd = [&](auto operation, auto ioperand) {
        for (const auto& prod : ioperand->GetProducers()) {
            if (prod->BelongTo() != this || prod == operation) {
                continue;
            }
            FE_ASSERT(FeError::NOT_EXIST, opToIndex.count(prod) != 0)
                << "Producer not found in opToIndex: " << prod->Dump();
            outDegree[opToIndex[prod]]++;
        }
    };

    for (auto& op : operations_) {
        for (auto& iop : op->iOperand) {
            addProd(op.get(), iop);
        }
        for (auto& dop : op->dependOperand) {
            addProd(op.get(), dop);
        }
        for (auto [type, index] : usageDict[op.get()]) {
            if (type == GET_TENSOR_DATA_OPERAND_IOTYPE_INCAST) {
                addProd(op.get(), inCasts_[index]);
            } else if (type == GET_TENSOR_DATA_OPERAND_IOTYPE_OUTCAST) {
                addProd(op.get(), outCasts_[index]);
            }
        }
        for (auto* producer : op->ProducerOpsByToken()) {
            if (producer == op.get() || producer->BelongTo() != this) {
                continue;
            }
            auto producerIter = opToIndex.find(producer);
            FE_ASSERT(FeError::NOT_EXIST, producerIter != opToIndex.end())
                << "Token producer not found in opToIndex: " << producer->Dump();
            outDegree[producerIter->second]++;
        }
    }
    for (auto& opGroup : operationGroups_) {
        for (size_t idx = 1; idx < opGroup.size(); idx++) {
            prevOperation[opToIndex[opGroup[idx]]] = opToIndex[opGroup[idx - 1]];
            outDegree[opToIndex[opGroup[idx - 1]]]++;
        }
    }
    std::queue<int> q;
    for (size_t idx = 0; idx < operations_.size(); idx++) {
        if (outDegree[idx] == 0) {
            q.emplace(idx);
        }
    }

    auto visit = [&](auto operation, auto ioperand) {
        for (const auto& producer : ioperand->GetProducers()) {
            if (producer->BelongTo() != this || producer == operation) {
                continue;
            }
            auto nxtOpIndex = opToIndex[producer];
            if (--outDegree[nxtOpIndex] == 0) {
                q.emplace(nxtOpIndex);
            }
        }
    };

    std::vector<std::shared_ptr<Operation>> sortedOperations;
    while (!q.empty()) {
        const auto& op = operations_[q.front()];
        q.pop();
        sortedOperations.emplace_back(op);
        int prevOpIndex = prevOperation[opToIndex[op.get()]];
        if (prevOpIndex >= 0) {
            if (--outDegree[prevOpIndex] == 0) {
                q.emplace(prevOpIndex);
            }
        }
        for (auto& iop : op->iOperand) {
            visit(op.get(), iop);
        }
        for (auto& dop : op->dependOperand) {
            visit(op.get(), dop);
        }
        for (auto [type, index] : usageDict[op.get()]) {
            if (type == GET_TENSOR_DATA_OPERAND_IOTYPE_INCAST) {
                visit(op.get(), inCasts_[index]);
            } else if (type == GET_TENSOR_DATA_OPERAND_IOTYPE_OUTCAST) {
                visit(op.get(), outCasts_[index]);
            }
        }
        for (auto* producer : op->ProducerOpsByToken()) {
            if (producer == op.get() || producer->BelongTo() != this) {
                continue;
            }
            auto producerIter = opToIndex.find(producer);
            FE_ASSERT(FeError::NOT_EXIST, producerIter != opToIndex.end())
                << "Token producer not found in opToIndex: " << producer->Dump();
            if (--outDegree[producerIter->second] == 0) {
                q.emplace(producerIter->second);
            }
        }
    }
    for (auto& op : operations_) {
        FE_ASSERT(FeError::OP_DEPENDENCY_CYCLE, outDegree[opToIndex[op.get()]] == 0)
            << "cycle detected: " << op->Dump();
    }
    FE_ASSERT(operations_.size() == sortedOperations.size())
        << "Sorted operations size mismatch: " << sortedOperations.size() << " and original size "
        << operations_.size();
    std::reverse(sortedOperations.begin(), sortedOperations.end());
    return sortedOperations;
}

std::vector<std::shared_ptr<Operation>> Function::GetLightweightSortedOperations(bool preserveOriginalOrder) const
{
    FE_ASSERT(operationGroups_.empty()) << "Lightweight sort does not support operationGroups_.";
    LightweightOperationSorter sorter(*this, operations_, inCasts_, outCasts_, preserveOriginalOrder);
    return sorter.Sort();
}

void Function::SortOperations(SortOperationsMode mode)
{
    std::vector<std::shared_ptr<Operation>> sortedOperations;
    switch (mode) {
        case SortOperationsMode::GENERAL:
            sortedOperations = GetSortedOperations();
            break;
        case SortOperationsMode::LIGHTWEIGHT:
            sortedOperations = GetLightweightSortedOperations();
            break;
        case SortOperationsMode::LIGHTWEIGHT_STABLE:
            sortedOperations = GetLightweightSortedOperations(true);
            break;
        default:
            FE_ASSERT(FeError::INVALID_VAL, false) << "Invalid sort operations mode.";
    }
    operations_ = sortedOperations;
    RefreshOpPosition();
    sorted_ = true;
}

void Function::ScheduleBy(const std::vector<Operation*>& newList, bool needRefresh)
{
    if (needRefresh) {
        RefreshOpPosition();
    }
    FE_ASSERT(newList.size() == operations_.size())
        << "Size mismatch: newList size = " << newList.size() << ", operations_ size = " << operations_.size();
    std::vector<std::shared_ptr<Operation>> newOperations;
    for (auto op : newList) {
        FE_ASSERT(FeError::NOT_EXIST, opPosition_.count(op) > 0) << "Operation not found in opPosition_:" << op->Dump();
        newOperations.emplace_back(operations_[opPosition_.at(op)]);
    }
    operations_ = newOperations;
    RefreshOpPosition();

    sorted_ = true;
}

void Function::AddOperationGroup(std::vector<Operation*> operationGroup)
{
    size_t groupID = operationGroups_.size();
    for (const auto& operation : operationGroup) {
        FE_ASSERT(FeError::IS_EXIST, operation->GroupID() == NON_GROUP)
            << "Operation already in a group:" << operation->Dump();
        operation->SetGroupID(groupID);
    }
    operationGroups_.emplace_back(std::move(operationGroup));
    sorted_ = false;
}

void Function::ClearOperationGroups()
{
    for (auto& opGroup : operationGroups_) {
        for (auto& op : opGroup) {
            op->SetGroupID(NON_GROUP);
        }
    }
    operationGroups_.clear();
}

void Function::CheckGroupValid() const
{
    std::unordered_set<const Operation*> inGroupOp;
    for (size_t i = 0; i < operationGroups_.size(); i++) {
        for (auto& operation : operationGroups_[i]) {
            FE_ASSERT(operation->GroupID() == i) << "Operation GroupID mismatch:\n"
                                                 << "Expected: " << i << ", Actual: " << operation->GroupID() << "\n"
                                                 << "Operation:" << operation->Dump();
            FE_ASSERT(FeError::IS_EXIST, inGroupOp.count(operation) == 0)
                << "Duplicate operation in group:" << operation->Dump();
            inGroupOp.emplace(operation);
        }
    }
    for (const auto& operation : operations_) {
        FE_ASSERT(FeError::IS_EXIST, inGroupOp.count(operation.get()) == (operation->GroupID() != NON_GROUP))
            << "Operation group membership mismatch:\n"
            << "Operation: " << operation->Dump() << "\n"
            << "GroupID: " << operation->GroupID();
    }
}

void Function::RefreshOpPosition()
{
    opPosition_.clear();
    for (size_t idx = 0; idx < operations_.size(); ++idx) {
        FE_ASSERT(FeError::NOT_EXIST, opPosition_.count(operations_[idx].get()) == 0)
            << "Duplicate operation found in opPosition_:\n"
            << operations_[idx]->Dump();
        opPosition_.emplace(operations_[idx].get(), idx);
    }
}

void Function::RefreshVarDependency()
{
    varDependency_.Clear();
    for (const auto& operation : operations_) {
        if (operation->IsDeleted()) {
            continue;
        }
        auto stmt = std::static_pointer_cast<const ir::Stmt>(operation);
        for (const auto& token : operation->result_token_) {
            varDependency_.AddProducer(token, stmt);
        }
        for (const auto& token : operation->tokens_) {
            varDependency_.AddConsumer(token, stmt);
        }
    }
}

bool Function::enableMagicLookupRecord_{false};
std::map<std::pair<int, int>, std::set<Operation*, LogicalTensor::CompareOp>> Function::tensorAndSubgraphToProducer_;

void Function::ProducerMagicLookup(const Function* function, const LogicalTensorPtr& tensor,
                                   const std::set<Operation*, LogicalTensor::CompareOp>& producers,
                                   const int subGraphId, int& index, std::unordered_map<int, int>& magic2index,
                                   std::stringstream& ss)
{
    for (auto& op : producers) {
        if (subGraphId != INT32_MIN && op->GetSubgraphID() != subGraphId) {
            continue;
        }
        ss << " " << op->GetOpcodeStr(true);
        for (size_t idx = 0; idx < op->GetOOperands().size(); idx++) {
            if (op->GetOutputOperand(idx) == tensor) {
                ss << "oAttrOffset " << idx << " " << op->GetOOpAttrOffset(idx) << " ";
            }
        }
        for (size_t idx = 0; idx < op->GetIOperands().size(); idx++) {
            ss << "iAttrOffset " << idx << " " << op->GetIOpAttrOffset(idx) << " ";
        }
        if (function->GetFunctionType() == FunctionType::STATIC) {
            if (OpcodeManager::Inst().IsBoundaryIn(op->GetOpcode())) {
                for (size_t idx = 1; idx < op->iOperand[0]->tensor->rawshape.size(); idx++) {
                    ss << op->iOperand[0]->tensor->rawshape[idx] << " ";
                }
            }
            if (OpcodeManager::Inst().IsBoundaryOut(op->GetOpcode())) {
                for (size_t idx = 1; idx < op->oOperand[0]->tensor->rawshape.size(); idx++) {
                    ss << op->oOperand[0]->tensor->rawshape[idx] << " ";
                }
            }
        }
        for (const auto& attr : OpcodeManager::Inst().GetAttrs(op->GetOpcode())) {
            ss << " attr: [" << attr << " : " << op->DumpAttr(attr) << "]";
        }
        if (function->GetGraphType() != GraphType::BLOCK_GRAPH) {
            ss << op->GetTileShape().ToString();
        }
        if (op->GetOpAttribute() != nullptr) {
            if (IsAssembleLike(op->GetOpcode())) {
                if (!SubgraphUtils::IsBoundary(op->oOperand[0])) {
                    ss << " " << op->GetOpAttribute()->Dump();
                }
            } else if (function->GetGraphType() == GraphType::BLOCK_GRAPH) {
                ss << " " << op->GetOpAttribute()->Dump();
            } else if ((!IsCopyIn(op->GetOpcode()) && !IsCopyOut(op->GetOpcode())) ||
                       function->GetGraphType() != GraphType::BLOCK_GRAPH) {
                ss << " " << op->GetOpAttribute()->Dump();
            }
        }
        MagicLookup(function, op->iOperand, subGraphId, index, magic2index, ss);
    }
}

void Function::MagicLookup(const Function* function, const std::vector<LogicalTensorPtr>& operand, const int subGraphId,
                           int& index, std::unordered_map<int, int>& magic2index, std::stringstream& ss)
{
    for (auto& t : operand) {
        if (magic2index.count(t->GetMagic()) && (function->inCastsSet_.count(t) == 0) &&
            t->GetProducers().size() != 0) {
            continue;
        }
        magic2index[t->GetMagic()] = index++;
        ss << "("
           << " " << static_cast<int>(t->tensor->datatype) << " ";
        // Add shape information
        for (const auto& dim : t->shape) {
            ss << dim << " ";
        }
        if (function->IsFunctionType(FunctionType::STATIC)) {
            for (const auto& dim : t->oriShape) {
                ss << dim << " ";
            }
        }
        if (t->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
            for (const auto& dim : t->offset) {
                ss << dim << " ";
            }
        }
        if (!enableMagicLookupRecord_) {
            ProducerMagicLookup(function, t, t->GetProducers(), subGraphId, index, magic2index, ss);
        } else if (tensorAndSubgraphToProducer_.count({t->GetMagic(), subGraphId}) > 0) {
            ProducerMagicLookup(function, t, tensorAndSubgraphToProducer_[{t->GetMagic(), subGraphId}], subGraphId,
                                index, magic2index, ss);
        }
        ss << ")";
    }
}

unsigned long Function::ComputeHashOrderless() const
{
    std::stringstream ss;
    ss << std::to_string(static_cast<int>(functionType_)) << " ";
    ss << std::to_string(static_cast<int>(graphType_)) << " ";
    if (!IsGraphType({GraphType::BLOCK_GRAPH, GraphType::LEAF_VF_GRAPH}) &&
        !IsFunctionTypeAndGraphType(FunctionType::STATIC, GraphType::TENSOR_GRAPH)) {
        ss << GetMagicName() << " ";
    }

    // Build using Polish Notation
    int index = 0;
    std::unordered_map<int, int> magic2index;
    // 只有leaf graph需要判断边界
    if (graphType_ == GraphType::BLOCK_GRAPH) {
        if (operations_.size()) {
            MagicLookup(this, GetOutcast(), operations_[operations_.size() - 1]->GetSubgraphID(), index, magic2index,
                        ss);
        }
    } else {
        MagicLookup(this, GetOutcast(), INT32_MIN, index, magic2index, ss);
    }

    // 补充一些没有输出的Op的hash
    for (size_t i = 0; i < operations_.size(); i++) {
        if (operations_[i]->oOperand.empty()) {
            ss << " " << operations_[i]->GetOpcodeStr(true);
            for (const auto& attr : OpcodeManager::Inst().GetAttrs(operations_[i]->GetOpcode())) {
                ss << " attr: [" << attr << " : " << operations_[i]->DumpAttr(attr) << "]";
            }
            ss << operations_[i]->GetTileShape().ToString();
            MagicLookup(this, operations_[i]->GetIOperands(), operations_[0]->GetSubgraphID(), index, magic2index, ss);
        }
    }

    for (auto& i : inCasts_) {
        ss << "(i" << magic2index[i->GetMagic()] << ")";
        bool isGlobal = (globalTensors_.count(i) != 0);
        if (isGlobal) {
            ss << "(Global)";
        }
    }
    for (auto& o : outCasts_) {
        ss << "(o" << magic2index[o->GetMagic()] << ")";
        bool isGlobal = (globalTensors_.count(o) != 0);
        if (isGlobal) {
            ss << "(Global)";
        }
    }

    // fill symbol and loop range attr of dyndev tensor graph for dynamic binary reuse
    if (functionType_ == FunctionType::DYNAMIC_LOOP && dynloopAttr_ != nullptr) {
        ss << "symbol name:[" << dynloopAttr_->iterSymbolName << "]";
        ss << "loop range:[" << dynloopAttr_->loopRange.Dump() << "]";
    }
    // temporary avoidance, switch SUPPORT_DYNAMIC_ALIGNED has an unexpected effect on dynamic binary reuse
    if (functionType_ == FunctionType::DYNAMIC) {
        ss << "dynamic unaligned:" << config::GetCodeGenOption<bool>(SUPPORT_DYNAMIC_ALIGNED);
        if (body_ != nullptr) {
            ss << "body:" << ir::PythonPrint(std::static_pointer_cast<const ir::IRNode>(body_));
        }
    }
    if (leafFuncAttr_ != nullptr) {
        // mixId标识同一次Mix拆出来的leafFunction组
        if (leafFuncAttr_->mixId != LeafFuncAttribute::INVALID_MIX_ID) {
            ss << " MIX_ID:" << leafFuncAttr_->mixId;
        }
        if (leafFuncAttr_->aivCore != AIVCore::UNSPECIFIED) {
            ss << " AIV_CORE:" << static_cast<int>(leafFuncAttr_->aivCore);
        }
    }
    std::hash<std::string> hasher;
    auto result = hasher(ss.str());
    FE_LOGD("Hash for function %d %s is %s hash value is %lu\n", functionMagic_, GetMagicName().c_str(),
            ss.str().c_str(), result);
    return result;
}

void Function::EraseOperations(bool eraseRelatedTensor, bool sorted, SortOperationsMode mode)
{
    std::unordered_set<LogicalTensorPtr> inOutCastSet(inCasts_.begin(), inCasts_.end());
    inOutCastSet.insert(outCasts_.begin(), outCasts_.end());
    std::vector<std::shared_ptr<Operation>> operations;
    std::unordered_set<std::shared_ptr<LogicalTensor>> removeCandidiateTensor;
    std::unordered_set<std::shared_ptr<LogicalTensor>> removeProducerTensor;
    for (auto& op : operations_) {
        if (!op->IsDeleted()) {
            operations.emplace_back(op);
            continue;
        }
        FE_ASSERT(op->IsDeleted()) << "Operation not marked as deleted:" << op->Dump();
        for (auto& input : op->GetIOperands()) {
            input->RemoveConsumer(op.get());
            removeCandidiateTensor.insert(input);
        }

        for (auto& output : op->GetOOperands()) {
            output->RemoveProducer(op.get());
            removeCandidiateTensor.insert(output);
            removeProducerTensor.insert(output);
        }

        for (auto& depend : op->GetDependOperands()) {
            depend->RemoveDependOp(op.get());
            removeCandidiateTensor.insert(depend);
        }
    }
    operations_ = operations;

    if (eraseRelatedTensor) {
        EraseRelatedTensors(removeCandidiateTensor, removeProducerTensor, inOutCastSet);
    }

    RefreshVarDependency();

    if (sorted) {
        SortOperations(mode);
    }
}

void Function::EraseRelatedTensors(const std::unordered_set<std::shared_ptr<LogicalTensor>>& removeCandidateTensor,
                                   const std::unordered_set<std::shared_ptr<LogicalTensor>>& removeProducerTensor,
                                   const std::unordered_set<LogicalTensorPtr>& inOutCastSet)
{
    for (auto tensorPtr : removeCandidateTensor) {
        if (inOutCastSet.count(tensorPtr) != 0) {
            continue;
        }
        if (tensorPtr->GetProducers().empty() && tensorPtr->GetConsumers().empty()) {
            GetTensorMap().Erase(tensorPtr);
        } else if (removeProducerTensor.count(tensorPtr) > 0 && tensorPtr->GetProducers().empty()) {
            GetTensorMap().Erase(tensorPtr);
            for (auto& consumer : tensorPtr->GetConsumers()) {
                if (consumer->BelongTo() == this) {
                    consumer->EraseInput(tensorPtr);
                }
            }
        }
    }
}

void Function::EraseOperations(const OperationDeleter& deleter)
{
    if (!sorted_) {
        SortOperations();
    }
    for (auto& op : operations_) {
        if (deleter(op, *this)) {
            op->SetAsDeleted();
        }
    }

    EraseOperations();
}

FunctionHash Function::ComputeHash()
{
    if (functionHash_.GetHash() != 0 &&
        (functionType_ != FunctionType::DYNAMIC_LOOP && functionType_ != FunctionType::DYNAMIC)) {
        /* 动态类型的graph里面的op和tensor会随着循环的展开而变化，每次都需要刷新 */
        return functionHash_;
    }
    for (auto& ele : inCasts_) {
        inCastsSet_.emplace(ele);
    }
    functionHash_ = ComputeHashOrderless();
    return functionHash_;
}

void Function::BuildTensorMap()
{
    tensorMap_.Reset();
    for (auto& ele : inCasts_) {
        tensorMap_.Insert(ele);
    }
    for (auto& ele : operations_) {
        for (auto& oOp : ele->GetOOperands()) {
            if (!tensorMap_.GetTensorByMagic(oOp->magic)) {
                tensorMap_.Insert(oOp);
            }
        }
    }
}

Operation& Function::AddOperation(const std::string& opName, LogicalTensors iOperands, const LogicalTensors& oOperands)
{
    return AddOperation(FindOpcode(opName), iOperands, oOperands);
}

namespace {

LogicalTensorPtr GetSliceParent(const LogicalTensorPtr& tensor)
{
    LogicalTensorPtr parent = nullptr;
    if (tensor == nullptr || !tensor->GetAttr("SLICE_PARENT", parent)) {
        return nullptr;
    }
    return parent;
}

void ResetSliceView(const LogicalTensorPtr& tensor)
{
    int dim = tensor->shape.size();
    tensor->offset = std::vector<int64_t>(dim, 0);
    tensor->dynOffset_ = std::vector<SymbolicScalar>(dim, SymbolicScalar(0));
    tensor->RemoveAttr("SLICE_PARENT");
    tensor->tensor = std::make_shared<RawTensor>(tensor->tensor->datatype, tensor->shape, tensor->tensor->format,
                                                 tensor->tensor->symbol);
}

ir::VarPtr GetPairedNormalToken(const ir::VarPtr& token)
{
    if (token == nullptr) {
        return nullptr;
    }
    return ExpandFunction::GetNormalToken(token);
}

bool SameLogicalTensorPtr(const LogicalTensorPtr& lhs, const LogicalTensorPtr& rhs)
{
    if (lhs == nullptr || rhs == nullptr) {
        return false;
    }
    return lhs.get() == rhs.get() || lhs->GetMagic() == rhs->GetMagic();
}

bool IsCurrentTileOpOperand(const LogicalTensorPtr& tensor, const LogicalTensors& currentOperands)
{
    if (tensor == nullptr) {
        return false;
    }
    auto parent = GetSliceParent(tensor);
    for (const auto& current : currentOperands) {
        if (SameLogicalTensorPtr(tensor, current) || SameLogicalTensorPtr(parent, current)) {
            return true;
        }
    }
    return false;
}

bool HasCurrentTileOpOperand(const LogicalTensors& tensors, const LogicalTensors& currentOperands)
{
    for (const auto& tensor : tensors) {
        if (IsCurrentTileOpOperand(tensor, currentOperands)) {
            return true;
        }
    }
    return false;
}

bool AttachReadToMatchingInputParents(const LogicalTensors& iOperands, const LogicalTensors& currentInputs,
                                      const ir::VarPtr& token, const ir::VarPtr& normal,
                                      std::unordered_map<LogicalTensor*, std::vector<ir::VarPtr>>& target)
{
    bool matched = false;
    for (const auto& iOp : iOperands) {
        if (!IsCurrentTileOpOperand(iOp, currentInputs)) {
            continue;
        }
        auto parent = GetSliceParent(iOp);
        if (parent == nullptr || parent->GetReadToken() != token) {
            continue;
        }
        target[iOp.get()].push_back(normal);
        matched = true;
    }
    return matched;
}

bool AttachToOutputParents(const LogicalTensors& oOperands, const LogicalTensors& currentOutputs,
                           const ir::VarPtr& normal,
                           std::unordered_map<LogicalTensor*, std::vector<ir::VarPtr>>& target)
{
    bool hasParent = false;
    for (const auto& oOp : oOperands) {
        if (!IsCurrentTileOpOperand(oOp, currentOutputs)) {
            continue;
        }
        if (GetSliceParent(oOp) == nullptr) {
            continue;
        }
        target[oOp.get()].push_back(normal);
        hasParent = true;
    }
    return hasParent;
}

struct ExpandTokenPlan {
    std::unordered_map<LogicalTensor*, std::vector<ir::VarPtr>> sliceResultTokens;
    std::unordered_map<LogicalTensor*, std::vector<ir::VarPtr>> contractResultTokens;
    std::unordered_map<LogicalTensor*, std::vector<ir::VarPtr>> contractInputTokens;
    std::vector<ir::VarPtr> opResultTokens;
    std::vector<ir::VarPtr> opInputTokens;
};

void CollectResultTokens(const Operation& tileOp, const LogicalTensors& iOperands, const LogicalTensors& oOperands,
                         ExpandTokenPlan& plan)
{
    const auto& currentInputs = tileOp.GetIOperands();
    const auto& currentOutputs = tileOp.GetOOperands();
    for (const auto& token : tileOp.result_token_) {
        auto tokenType = token ? std::dynamic_pointer_cast<const ir::TokenType>(token->GetType()) : nullptr;
        auto normal = GetPairedNormalToken(token);
        if (tokenType == nullptr || normal == nullptr) {
            continue;
        }
        if (tokenType->kind_ == ir::TokenKind::READ) {
            if (!AttachReadToMatchingInputParents(iOperands, currentInputs, token, normal, plan.sliceResultTokens) &&
                HasCurrentTileOpOperand(oOperands, currentOutputs)) {
                plan.opResultTokens.push_back(normal);
            }
        } else if (tokenType->kind_ == ir::TokenKind::WRITE) {
            if (!AttachToOutputParents(oOperands, currentOutputs, normal, plan.contractResultTokens) &&
                HasCurrentTileOpOperand(oOperands, currentOutputs)) {
                plan.opResultTokens.push_back(normal);
            }
        }
    }
}

void CollectInputTokens(const Operation& tileOp, const LogicalTensors& iOperands, const LogicalTensors& oOperands,
                        ExpandTokenPlan& plan)
{
    const auto& currentInputs = tileOp.GetIOperands();
    const auto& currentOutputs = tileOp.GetOOperands();
    for (const auto& token : tileOp.tokens_) {
        auto normal = GetPairedNormalToken(token);
        if (normal == nullptr) {
            continue;
        }
        if (!AttachToOutputParents(oOperands, currentOutputs, normal, plan.contractInputTokens) &&
            HasCurrentTileOpOperand(iOperands, currentInputs)) {
            plan.opInputTokens.push_back(normal);
        }
    }
}

ExpandTokenPlan CollectExpandTokenPlan(Operation* currentTileOp, const LogicalTensors& iOperands,
                                       const LogicalTensors& oOperands)
{
    ExpandTokenPlan plan;
    if (currentTileOp == nullptr) {
        return plan;
    }
    CollectResultTokens(*currentTileOp, iOperands, oOperands, plan);
    CollectInputTokens(*currentTileOp, iOperands, oOperands, plan);
    return plan;
}

void AssignMappedTokens(std::vector<ir::VarPtr>& dst,
                        const std::unordered_map<LogicalTensor*, std::vector<ir::VarPtr>>& src, LogicalTensor* key)
{
    auto it = src.find(key);
    if (it != src.end()) {
        dst = it->second;
    }
}

} // namespace

Operation& Function::AddOperation(const Opcode opCode, LogicalTensors iOperands, const LogicalTensors& oOperands)
{
    auto plan = CollectExpandTokenPlan(ExpandFunction::GetCurrentTileOp(), iOperands, oOperands);
    CheckTensorDynamicShape(iOperands, opCode);

    const Opcode sliceOpcode = config::GetSliceOpcode();
    const Opcode contractOpcode = config::GetContractOpcode();
    for (auto& iOp : iOperands) {
        auto parent = GetSliceParent(iOp);
        if (parent == nullptr) {
            continue;
        }
        auto& sliceOp = AddRawOperation(sliceOpcode, {parent}, {iOp});
        sliceOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(iOp->offset, iOp->dynOffset_, iOp->dynValidShape_));
        AssignMappedTokens(sliceOp.result_token_, plan.sliceResultTokens, iOp.get());
        ResetSliceView(iOp);
    }

    auto& ret = AddRawOperation(opCode, iOperands, oOperands);
    if (!plan.opResultTokens.empty()) {
        ret.result_token_ = std::move(plan.opResultTokens);
    }
    if (!plan.opInputTokens.empty()) {
        ret.tokens_ = std::move(plan.opInputTokens);
    }

    for (auto& oOp : oOperands) {
        auto parent = GetSliceParent(oOp);
        if (parent == nullptr) {
            continue;
        }
        auto& contractOp = AddRawOperation(contractOpcode, {oOp}, {parent});
        contractOp.SetAssembleOpAttribute(oOp->offset, oOp->dynOffset_);
        AssignMappedTokens(contractOp.result_token_, plan.contractResultTokens, oOp.get());
        AssignMappedTokens(contractOp.tokens_, plan.contractInputTokens, oOp.get());
        ResetSliceView(oOp);
    }
    return ret;
}

void Function::CreateTensorDataImportOp()
{
    IRBuilder builder;

    std::unordered_set<int> visited;
    auto n = operations_.size();
    for (size_t i = 0; i < n; i++) {
        auto& op = operations_[i];
        auto dynAttrList = op->GetDynamicAttributeList();
        auto dict = GetTensorDataDict(dynAttrList);
        for (auto& [index, callList] : dict) {
            (void)callList;
            if (visited.count(index)) {
                continue;
            }
            auto assemble = builder.GetTensorDataDesc(index);
            std::vector<int64_t> importShape(assemble->GetShape().size(), 1);
            std::vector<int64_t> importOffset(assemble->GetShape().size(), 0);
            auto import = View(*assemble, importShape, importOffset);
            auto importOp = *import.GetStorage()->GetProducers().begin();
            SetEmuOpcode(importOp, EMUOP_TENSOR_GETDATA_IMPORT);
            GetTensorDataSetIndex(importOp, index);
            visited.insert(index);
        }
    }
}

Operation& Function::AddRawOperation(const Opcode opCode, const LogicalTensors& iOperands,
                                     const LogicalTensors& oOperands, ir::Span span)
{
    if (IsFunctionTypeAndGraphType(FunctionType::STATIC, {GraphType::EXECUTE_GRAPH, GraphType::BLOCK_GRAPH})) {
        sorted_ = true;
    } else {
        sorted_ = functionType_ == FunctionType::DYNAMIC;
    }
    auto& op = operations_.emplace_back(std::make_shared<Operation>(*this, opCode, iOperands, oOperands));
    opPosition_.emplace(op.get(), operations_.size() - 1);
    auto scopeConfig = config::GetPassOption<std::vector<int64_t>>(SG_SET_SCOPE);
    if (scopeConfig.size() == 0x3) {
        operations_.back()->SetScopeInfo(Operation::ScopeInfo::FromConfig(scopeConfig));
    } else if (scopeConfig.size() == 1) {
        operations_.back()->SetScopeId(static_cast<int>(scopeConfig[0]));
    } else {
        operations_.back()->SetScopeId(-1);
    }

    operations_.back()->SetAtomicScopeId(config::GetAtomicScopeId());

    if (!span.IsUnknown()) {
        operations_.back()->SetSpan(span);
    }
    return *operations_.back();
}

void Function::AddOriginIncast(const std::shared_ptr<LogicalTensor>& tensor)
{
    for (auto& ele : originInCasts_) {
        if (ele == tensor) {
            return;
        }
    }
    originInCasts_.push_back(tensor);
}

void Function::AddOriginOutcast(const std::shared_ptr<LogicalTensor>& tensor)
{
    for (auto& ele : originOutCasts_) {
        if (ele == tensor) {
            return;
        }
    }
    originOutCasts_.push_back(tensor);
}

void Function::SetSameMemId(const LogicalTensorPtr& operand, const LogicalTensorPtr& dst)
{
    FE_ASSERT(FeError::INVALID_TYPE, operand->Datatype() == dst->Datatype()) << "Check Dtype failed!";

    auto dstRaw = dst->GetRawTensor();
    auto operandRaw = operand->GetRawTensor();
    dstRaw->memoryId = operandRaw->memoryId;
    outIncastLinkMap[dstRaw] = operandRaw;
}

std::vector<Operation*> Function::GetAllInputOperations(const Operation& op) const
{
    std::vector<Operation*> retOps;
    if (op.BelongTo() != this) {
        return retOps;
    }
    for (const LogicalTensorPtr& inTensor : op.GetIOperands()) {
        if (inTensor == nullptr) {
            continue;
        }
        for (auto& producer : inTensor->GetProducers()) {
            retOps.push_back(producer);
        }
    }
    return retOps;
}

std::vector<Operation*> Function::GetAllOutputOperations(const Operation& op) const
{
    std::vector<Operation*> retOps;
    if (op.BelongTo() != this) {
        return retOps;
    }
    for (const LogicalTensorPtr& outTensor : op.GetOOperands()) {
        if (outTensor == nullptr) {
            continue;
        }
        for (auto& consumer : outTensor->GetConsumers()) {
            retOps.push_back(consumer);
        }
    }
    return retOps;
}

std::vector<Operation*> Function::GetCallopList() const
{
    std::vector<Operation*> callopList;
    for (auto& op : operations_) {
        if (op->GetOpcode() != Opcode::OP_CALL) {
            continue;
        }
        callopList.push_back(op.get());
    }
    return callopList;
}

std::vector<std::shared_ptr<CallOpAttribute>> Function::GetCallopAttrList() const
{
    std::vector<Operation*> callopList = GetCallopList();
    std::vector<std::shared_ptr<CallOpAttribute>> callopAttrList;
    for (auto callop : callopList) {
        auto callopAttr = std::static_pointer_cast<CallOpAttribute>(callop->GetOpAttribute());
        callopAttrList.push_back(callopAttr);
    }
    return callopAttrList;
}

std::vector<Function*> Function::GetCalleeFunctionList() const
{
    std::vector<Operation*> callopList = GetCallopList();
    std::vector<Function*> calleeFuncList;
    for (auto callop : callopList) {
        auto callopAttr = std::static_pointer_cast<CallOpAttribute>(callop->GetOpAttribute());
        auto calleeFunc = Program::GetInstance().GetFunctionByMagicName(callopAttr->GetCalleeMagicName());
        FE_ASSERT(FeError::NOT_EXIST, calleeFunc) << callopAttr->GetCalleeMagicName() << " is not in functionmap!";
        calleeFuncList.push_back(calleeFunc);
    }
    return calleeFuncList;
}

void Function::SubstituteIn(std::shared_ptr<LogicalTensor> oldTensor, std::shared_ptr<LogicalTensor> newTensor)
{
    for (auto& operation : operations_) {
        auto& cur = *operation;
        for (size_t i = 0; i < cur.GetInputOperandSize(); i++) {
            if (cur.GetInputOperand(i) == oldTensor) {
                cur.ReplaceIOperand(i, newTensor);
            }
        }
    }
}

void Function::SubstituteOut(std::shared_ptr<LogicalTensor> oldTensor, std::shared_ptr<LogicalTensor> newTensor)
{
    for (auto& operation : operations_) {
        auto& cur = *operation;
        for (size_t i = 0; i < cur.GetOOperands().size(); i++) {
            if (cur.GetOutputOperand(i) == oldTensor) {
                cur.ReplaceOOperand(i, newTensor);
            }
        }
    }
}

void Function::Substitute(std::shared_ptr<LogicalTensor> oldTensor, std::shared_ptr<LogicalTensor> newTensor)
{
    SubstituteIn(oldTensor, newTensor);
    SubstituteOut(oldTensor, newTensor);
}

void Function::RemoveOriginIncastConsumer(const std::shared_ptr<LogicalTensor>& originIncast) const
{
    for (const auto& producer : originIncast->GetProducers()) {
        for (auto& oOperandForProducerOp : producer->oOperand) {
            auto& consumers = oOperandForProducerOp->GetConsumers();
            for (auto it = consumers.begin(); it != consumers.end();) {
                if ((*it)->BelongTo() == this) {
                    it = consumers.erase(it);
                } else {
                    it++;
                }
            }
        }
    }
}

void Function::UpdateLinkMap(const std::shared_ptr<LogicalTensor>& oriLogicalTensor,
                             const std::shared_ptr<LogicalTensor>& newLogicalTensor, const bool isOutCast)
{
    if (isOutCast) {
        //  update outcast
        auto it = outIncastLinkMap.find(oriLogicalTensor->tensor);
        if (it != outIncastLinkMap.end()) {
            outIncastLinkMap[newLogicalTensor->tensor] = it->second;
            newLogicalTensor->tensor->memoryId = it->second->memoryId;
            FE_LOGD("UpdateLinkMap memoryId to %d  \n", it->second->memoryId);
            outIncastLinkMap.erase(it);
        }
    } else {
        //  update incast
        for (auto& ele : outIncastLinkMap) {
            if (ele.second == oriLogicalTensor->tensor) {
                ele.second = newLogicalTensor->tensor;
            }
        }
    }
}

std::shared_ptr<LogicalTensor> Function::CreateIncastTensor(const std::shared_ptr<LogicalTensor>& inArgument)
{
    auto idx = inCasts_.size();
    auto newSymbol = inArgument->tensor->GetSymbol();
    if (newSymbol == "") {
        newSymbol = "INCAST_SYMBOL" + std::to_string(idx);
    }
    auto incastSymbol = std::make_shared<LogicalTensor>(*this, inArgument->tensor->datatype, inArgument->shape,
                                                        inArgument->tensor->GetDynRawShape(), inArgument->Format(),
                                                        newSymbol);
    incastSymbol->tensor->UpdateDynRawShape(inArgument->tensor->GetDynRawShape());
    inCasts_.push_back(incastSymbol);

    UpdateLinkMap(inArgument, incastSymbol);
    return incastSymbol;
}

void Function::CreateFromIncast(const std::shared_ptr<LogicalTensor>& symbol,
                                const std::shared_ptr<LogicalTensor>& newIncast,
                                const std::shared_ptr<LogicalTensor>& origin)
{
    ir::Span::SetCurrent(ir::Span(__FILE__, __LINE__, 0));
    auto& incastOp = AddOperation(Opcode::OP_VIEW, {symbol}, {newIncast});
    ir::Span::ClearCurrent();
    incastOp.SetAttr(OpAttributeKey::isGlobalInput, true);

    auto dynOffset = origin->GetDynOffset();
    if (dynOffset.empty()) {
        dynOffset = SymbolicScalar::FromConcrete(origin->GetOffset());
    }

    auto validShape = symbol->GetDynValidShape();
    ASSERT(!validShape.empty());

    incastOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(origin->GetOffset(), dynOffset, validShape));
    newIncast->UpdateDynValidShape(validShape);
    newIncast->GetRawTensor()->UpdateDynRawShape(symbol->GetDynValidShape());
    newIncast->CopyMemoryType(origin);
    auto readToken = origin->GetReadToken();
    auto writeToken = origin->GetWriteToken();
    newIncast->SetReadToken(readToken);
    newIncast->SetWriteToken(writeToken);
}

void Function::MoveNewlyInsertedOpsToFront(size_t operationCountBefore)
{
    if (operations_.size() <= operationCountBefore) {
        return;
    }
    // CreateFromIncast 在尾部追加 VIEW op；将其移到最前面，保证 incast view 位于程序开头，前端无需额外排序。
    std::vector<std::shared_ptr<Operation>> newlyInsertedOps;
    newlyInsertedOps.reserve(operations_.size() - operationCountBefore);
    for (size_t i = operationCountBefore; i < operations_.size(); ++i) {
        newlyInsertedOps.push_back(operations_[i]);
    }
    operations_.resize(operationCountBefore);
    operations_.insert(operations_.begin(), newlyInsertedOps.begin(), newlyInsertedOps.end());
    RefreshOpPosition();
}

LogicalTensors Function::MakeIncasts(const std::shared_ptr<TensorSlotScope>& scope)
{
    LogicalTensors inArgumentList;
    size_t operationCountBefore = operations_.size();

    for (size_t idx = 0; idx < originInCasts_.size(); idx++) {
        auto origin = originInCasts_[idx];
        int rank = origin->shape.size();
        std::vector<int64_t> zeroOffset(rank, 0);
        auto inArgument = std::make_shared<LogicalTensor>(Parent(), origin->tensor, zeroOffset, origin->shape);
        inArgumentList.push_back(inArgument);

        auto incastSymbol = CreateIncastTensor(inArgument);

        if (scope) {
            scope->incastToInArgumentDict[incastSymbol] = inArgument;
        }
        if (slotScope_ && idx < slotScope_->oriIncastReadSlotSet.size()) {
            slotScope_->incastReadSlotSet.push_back(slotScope_->oriIncastReadSlotSet[idx]);
            slotScope_->ioslot.incastSlot.push_back(slotScope_->originalIocastsSlot.incastSlot[idx]);
        }

        auto newIncast = std::make_shared<LogicalTensor>(*this, origin->tensor->datatype, origin->shape,
                                                         origin->Format(), "INCAST_LOCAL_BUF" + std::to_string(idx));

        CreateFromIncast(incastSymbol, newIncast, origin);
        Substitute(origin, newIncast);
        originInCasts_[idx] = newIncast;
        RemoveOriginIncastConsumer(origin);
    }

    MoveNewlyInsertedOpsToFront(operationCountBefore);

    return inArgumentList;
}

std::shared_ptr<LogicalTensor> Function::CreateOutcastTensor(const std::shared_ptr<LogicalTensor>& outArgument,
                                                             const std::shared_ptr<RawTensor>& shareRaw,
                                                             const std::string& name)
{
    auto idx = outCasts_.size();
    auto newSymbol = name.empty() ? outArgument->tensor->GetSymbol() : name;
    if (newSymbol == "") {
        newSymbol = "OUTCAST_SYMBOL" + std::to_string(idx);
    }
    auto outSymbol = std::make_shared<LogicalTensor>(*this, outArgument->tensor->datatype, outArgument->shape,
                                                     outArgument->tensor->GetDynRawShape(), outArgument->Format(),
                                                     newSymbol);
    if (!name.empty()) {
        outSymbol->name_ = name;
    }
    if (shareRaw != nullptr) {
        /* Later versions of one raw tensor reuse the first version's outcast buffer. */
        outSymbol->tensor = shareRaw;
    } else {
        outSymbol->tensor->UpdateDynRawShape(outArgument->tensor->GetDynRawShape());
    }

    outCasts_.push_back(outSymbol);

    UpdateLinkMap(outArgument, outSymbol, true);
    return outSymbol;
}

void Function::CreateFromOutcast(const LogicalTensorPtr& symbol, const LogicalTensorPtr& newOutcast,
                                 const LogicalTensorPtr& origin)
{
    ir::Span::SetCurrent(ir::Span(__FILE__, __LINE__, 0));
    auto& op = AddOperation(Opcode::OP_ASSEMBLE, {newOutcast}, {symbol});
    ir::Span::ClearCurrent();

    auto dynOffset = origin->GetDynOffset();
    if (dynOffset.empty()) {
        dynOffset = SymbolicScalar::FromConcrete(origin->GetOffset());
    }

    auto validShape = origin->GetDynValidShape();
    if (validShape.empty()) {
        validShape = symbol->GetDynValidShape();
    }
    ASSERT(!validShape.empty());

    op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(origin->GetOffset(), dynOffset));
    newOutcast->UpdateDynValidShape(validShape);
    newOutcast->GetRawTensor()->UpdateDynRawShape(symbol->GetDynValidShape());
}

namespace {

// Result of scanning all producers of an original outcast tensor in MakeOutcasts.
// The two flags are independent and may both be true (e.g. full write then assemble).
struct OutcastProducerTraits {
    bool isAssembleOut{false};
    bool needRuntimeAlloc{false};
};

bool IsAssembleLikeProducer(const Operation* op)
{
    return (op->GetOpcode() == Opcode::OP_ASSEMBLE && op->HasAttribute("dassemble")) ||
           op->GetOpcode() == Opcode::OP_ASSEMBLE_SSA || op->GetOpcode() == Opcode::OP_ATOMIC_RMW;
}

// Inspects @p origin's producer set once and fills OutcastProducerTraits:
//   isAssembleOut: at least one assemble-like producer uses the partial-update rewrite path.
//   needRuntimeAlloc: at least one non-assemble producer needs RUNTIME_SlotMarkNeedAlloc,
//   except reshape(inplace=True) outcasts that already reuse the linked incast address.
OutcastProducerTraits ClassifyOutcastByProducers(const LogicalTensor& origin, bool isLinkedInplaceOutcast)
{
    OutcastProducerTraits traits;
    for (Operation* op : origin.GetProducers()) {
        // Per-producer classification is exclusive, but traits are accumulated across all producers,
        // so the final result may have both flags set (mixed assemble + compute producers).
        if (IsAssembleLikeProducer(op)) {
            traits.isAssembleOut = true;
        } else {
            if (!isLinkedInplaceOutcast) {
                traits.needRuntimeAlloc = true;
            }
        }
    }
    return traits;
}

} // namespace

void Function::RebindAssembleVersionsToOutcast(const LogicalTensorPtr& originOutcast,
                                               const LogicalTensorPtr& newOutcast)
{
    auto originRawTensor = originOutcast->GetRawTensor();
    auto outcastRawTensor = newOutcast->GetRawTensor();
    for (auto& operation : operations_) {
        for (auto& output : operation->GetOOperands()) {
            if (output->GetRawTensor() == originRawTensor) {
                output->tensor = outcastRawTensor;
            }
        }
    }
}

LogicalTensors Function::MakeOutcasts(const std::shared_ptr<TensorSlotScope>& scope)
{
    LogicalTensors outArgumentList;

    /* Multiple assemble versions of one raw tensor all write the same slot.
     * Snapshot the raw tensors first (the assemble-outcast rewrite below
     * rebinds them), so the versions can be grouped: every version of a group
     * shares the first version's outcast buffer and keeps its own version
     * name, i.e. one runtime slot with one outcast entry per assemble write. */
    std::vector<std::shared_ptr<RawTensor>> originRaws;
    std::unordered_map<int, std::vector<size_t>> rawMagicToIndices;
    originRaws.reserve(originOutCasts_.size());
    for (size_t idx = 0; idx < originOutCasts_.size(); idx++) {
        originRaws.push_back(originOutCasts_[idx]->tensor);
        rawMagicToIndices[originRaws[idx]->rawmagic].push_back(idx);
    }
    std::unordered_map<int, std::shared_ptr<RawTensor>> rawMagicToOutSymbolRaw;

    for (size_t idx = 0; idx < originOutCasts_.size(); idx++) {
        auto origin = originOutCasts_[idx];
        int rank = origin->shape.size();
        std::vector<int64_t> zeroOffset(rank, 0);
        auto outArgument = std::make_shared<LogicalTensor>(Parent(), originRaws[idx], zeroOffset, origin->shape,
                                                           originRaws[idx]->GetDynRawShape());
        bool versionedGroup = rawMagicToIndices[originRaws[idx]->rawmagic].size() > 1;
        if (versionedGroup) {
            outArgument->name_ = origin->name_;
        }
        Parent().tensorMap_.Insert(outArgument);
        outArgumentList.push_back(outArgument);

        bool isLinkedInplaceOutcast = outIncastLinkMap.count(origin->GetRawTensor()) != 0;
        std::shared_ptr<RawTensor> shareRaw;
        auto [groupIter, firstInGroup] = rawMagicToOutSymbolRaw.emplace(originRaws[idx]->rawmagic, nullptr);
        if (!firstInGroup) {
            shareRaw = groupIter->second;
        }
        auto outSymbol = CreateOutcastTensor(outArgument, shareRaw, versionedGroup ? origin->name_ : "");
        if (firstInGroup) {
            groupIter->second = outSymbol->tensor;
        }
        /* The outcast param is the same runtime buffer as the version it
         * exports: share the slot so it resolves to the version's slot id. */
        Program::GetInstance().GetTensorSlotManager()->SetSameSlot(origin, outSymbol);
        if (scope) {
            scope->outcastToOutArgumentDict[outSymbol] = outArgument;
        }
        if (slotScope_ && idx < slotScope_->oriOutcastWriteSlotSet.size()) {
            slotScope_->outcastWriteSlotSet.push_back(slotScope_->oriOutcastWriteSlotSet[idx]);
            slotScope_->ioslot.outcastSlot.push_back(slotScope_->originalIocastsSlot.outcastSlot[idx]);
        }
        const OutcastProducerTraits traits = ClassifyOutcastByProducers(*origin, isLinkedInplaceOutcast);
        SetOutcastNeedAlloc(outCasts_.back(), traits.needRuntimeAlloc);
        if (traits.isAssembleOut) {
            Substitute(origin, outSymbol);
            RebindAssembleVersionsToOutcast(origin, outSymbol);
            originOutCasts_[idx] = outSymbol;
            if (scope) {
                scope->partialUpdateOutcastDict[outSymbol] = true;
            }
        } else {
            auto newOutcast = std::make_shared<LogicalTensor>(*this, origin->tensor->datatype, origin->shape,
                                                              origin->Format(),
                                                              "OUTCAST_LOCAL_BUF" + std::to_string(idx));
            CreateFromOutcast(outSymbol, newOutcast, origin);
            Substitute(origin, newOutcast);
            originOutCasts_[idx] = newOutcast;
        }
    }

    return outArgumentList;
}

bool Function::IsFlattening() const
{
    return IsFunctionTypeAndGraphType(FunctionType::STATIC, {GraphType::TENSOR_GRAPH, GraphType::TILE_GRAPH});
}

FunctionType Function::GetFunctionType() const { return functionType_; }

void Function::SetFunctionType(FunctionType type) { functionType_ = type; }

std::string Function::GetFunctionTypeStr() const { return GetFunctionTypeNameDict().Find(functionType_); }

GraphType Function::GetGraphType() const { return graphType_; }

void Function::SetGraphType(GraphType type) { graphType_ = type; }

bool Function::IsFunctionType(FunctionType type) const { return functionType_ == type; }

bool Function::IsFunctionType(std::set<FunctionType> types) const { return types.count(functionType_) != 0; }

bool Function::IsGraphType(GraphType type) const { return graphType_ == type; }

bool Function::IsGraphType(std::set<GraphType> types) const { return types.count(graphType_) != 0; }

bool Function::IsFunctionTypeAndGraphType(FunctionType funcType, GraphType graphType) const
{
    return functionType_ == funcType && graphType_ == graphType;
}

bool Function::IsFunctionTypeAndGraphType(FunctionType funcType, std::set<GraphType> graphTypes) const
{
    return IsFunctionType(funcType) && IsGraphType(graphTypes);
}

bool Function::IsFunctionTypeAndGraphType(std::set<FunctionType> funcTypes, GraphType graphType) const
{
    return IsFunctionType(funcTypes) && IsGraphType(graphType);
}

bool Function::IsFunctionTypeAndGraphType(std::set<FunctionType> funcTypes, std::set<GraphType> graphTypes) const
{
    return IsFunctionType(funcTypes) && IsGraphType(graphTypes);
}

void Function::DumpJsonFile(std::string fileName)
{
    auto filePath = config::LogTopFolder() + "/" + funcRawName_ + ".json";
    if (!fileName.empty()) {
        filePath = fileName;
    }
    std::ofstream file(filePath);
    CHECK(ExternalError::BAD_FD, file.is_open()) << "Failed to open file: " << filePath;
    Json progDump;
    progDump["version"] = T_VERSION;
    progDump["functions"].push_back(DumpJson());
    progDump["entryhash"] = this->GetFunctionHash().Data();
    file << progDump.dump(1) << std::endl;
    file.close();
}

struct RawTensorCompare {
    bool operator()(const std::shared_ptr<RawTensor>& a, const std::shared_ptr<RawTensor>& b) const
    {
        return a->rawmagic < b->rawmagic;
    }
};

struct TensorCompare {
    bool operator()(const std::shared_ptr<LogicalTensor>& a, const std::shared_ptr<LogicalTensor>& b) const
    {
        return a->magic < b->magic;
    }
};

Json Function::DumpJson(bool useTable)
{
    Json funcJson;
    funcJson[T_FIELD_KIND] = static_cast<int>(Kind::T_KIND_FUNCTION);
    funcJson["rawname"] = funcRawName_;
    funcJson["funcmagic"] = GetFuncMagic();
    if (parent_ != nullptr) {
        funcJson["parent_funcmagic"] = parent_->GetFuncMagic();
    }
    funcJson["functype"] = functionType_;
    funcJson["graphtype"] = graphType_;
    funcJson["func_magicname"] = funcMagicName_;
    funcJson["_opseed"] = opSeed_;
    funcJson["_rawid"] = IdGen<IdType::RAW_TENSOR>::Inst().CurId();
    funcJson["_funcid"] = IdGen<IdType::FUNCTION>::Inst().CurId();
    funcJson["_sg_pg_lowerbound"] = paramConfigs_.sgPgLowerBound;
    funcJson["_sg_parallel_num"] = paramConfigs_.sgParallelNum;
    funcJson["_sg_partition_algorithm"] = paramConfigs_.sgPartitionAlgorithm;
    funcJson["_sg_mg_copyin_upper_bound"] = paramConfigs_.sgMgCopyInUpperBound;
    funcJson["_mg_vec_parallel_lb"] = paramConfigs_.mgVecParallelLb;
    funcJson["_total_subgraph_count"] = totalSubGraphCount_;
    funcJson["_auto_mix_partition"] = paramConfigs_.autoMixPartition;
    funcJson["_ooo_preschedule_method"] = paramConfigs_.OoOPreScheduleMethod;
    if (!span_.IsUnknown()) {
        funcJson["file"] = span_.Filename();
        funcJson["line"] = span_.BeginLine();
    }

    if (useTable) {
        std::vector<std::pair<int, std::vector<int>>> incasts;
        std::vector<std::pair<int, std::vector<int>>> outcasts;
        size_t inSize = inCasts_.size();
        for (size_t i = 0; i < inSize; i++) {
            std::pair<int, std::vector<int>> incast;
            incast.first = inCasts_[i]->GetMagic();
            if (slotScope_ != nullptr && i < slotScope_->ioslot.incastSlot.size()) {
                incast.second = slotScope_->ioslot.incastSlot[i];
            } else {
                incast.second = std::vector<int>();
            }
            incasts.push_back(incast);
        }
        size_t outSize = outCasts_.size();
        for (size_t i = 0; i < outSize; i++) {
            std::pair<int, std::vector<int>> outcast;
            outcast.first = outCasts_[i]->GetMagic();
            if (slotScope_ != nullptr && i < slotScope_->ioslot.outcastSlot.size()) {
                outcast.second = slotScope_->ioslot.outcastSlot[i];
            } else {
                std::vector<int> emptyOutcast;
                outcast.second = emptyOutcast;
            }
            outcasts.push_back(outcast);
        }

        funcJson["incasts"] = incasts;
        funcJson["outcasts"] = outcasts;
    } else {
        Json incasts = Json::array();
        Json outcasts = Json::array();
        Json globalTensors = Json::array();
        for (auto& i : inCasts_) {
            incasts.push_back(i->DumpJson(*this, true));
        }
        for (auto& o : outCasts_) {
            outcasts.push_back(o->DumpJson(*this, true));
        }
        funcJson["incasts"] = incasts;
        funcJson["outcasts"] = outcasts;
    }

    std::set<int> globalTensorSet;
    for (auto& t : globalTensors_) {
        globalTensorSet.emplace(t->GetMagic());
    }
    std::vector<int> globalTensorVec;
    for (auto& tMagic : globalTensorSet) {
        globalTensorVec.emplace_back(tMagic);
    }
    funcJson["global_tensors"] = globalTensorVec;
    funcJson["static"]["global_tensors"] = funcJson["global_tensors"];

    Json operations = Json::array();
    if (useTable) {
        for (const auto& op : operations_) {
            operations.push_back(op->DumpJson(false));
        }
    } else {
        for (const auto& op : operations_) {
            operations.push_back(op->DumpJson(true));
        }
    }
    funcJson["operations"] = operations;
    funcJson["hash"] = functionHash_.Data();

    if (leafFuncAttr_ != nullptr && leafFuncAttr_->coreType != CoreType::INVALID) {
        funcJson["leaf_func_attr"]["coretype"] = leafFuncAttr_->coreType;
    }

    if (rootFunc_ != nullptr) {
        funcJson["root_func_magic"] = rootFunc_->GetFuncMagic();
    }
    if (!programs_.empty()) {
        Json programsJson;
        for (auto& ele : programs_) {
            programsJson[ele.first] = ele.second->GetFuncMagic();
        }
        funcJson["programs"] = programsJson;
        funcJson["topo"] = topoInfo_.DumpJson();
        funcJson["static"]["topo"] = funcJson["topo"];
    }
    if (graphType_ == GraphType::BLOCK_GRAPH) {
        funcJson["subfunc_param"] = parameter_.ToJson();
        funcJson["static"]["subfunc_param"] = funcJson["subfunc_param"];
    }

    auto aicIt = readySubGraphIds_.find(CoreType::AIC);
    if (aicIt != readySubGraphIds_.end() && !aicIt->second.empty()) {
        funcJson["aic_ready_subgraph_ids"] = aicIt->second;
        funcJson["static"]["aic_ready_subgraph_ids"] = funcJson["aic_ready_subgraph_ids"];
    }

    auto aivIt = readySubGraphIds_.find(CoreType::AIV);
    if (aivIt != readySubGraphIds_.end() && !aivIt->second.empty()) {
        funcJson["aiv_ready_subgraph_ids"] = aivIt->second;
        funcJson["static"]["aiv_ready_subgraph_ids"] = funcJson["aiv_ready_subgraph_ids"];
    }

    auto aicpuIt = readySubGraphIds_.find(CoreType::AICPU);
    if (aicpuIt != readySubGraphIds_.end() && !aicpuIt->second.empty()) {
        funcJson["aicpu_ready_subgraph_ids"] = aicpuIt->second;
        funcJson["static"]["aicpu_ready_subgraph_ids"] = funcJson["aicpu_ready_subgraph_ids"];
    }

    if (useTable) {
        std::set<std::shared_ptr<RawTensor>, RawTensorCompare> rawTensorSet;
        std::set<std::shared_ptr<LogicalTensor>, TensorCompare> tensorSet;
        for (const auto& incast : inCasts_) {
            tensorSet.insert(incast);
            rawTensorSet.insert(incast->tensor);
        }
        for (const auto& outcast : outCasts_) {
            tensorSet.insert(outcast);
            rawTensorSet.insert(outcast->tensor);
        }
        for (const auto& op : operations_) {
            for (auto& i : op->GetIOperands()) {
                tensorSet.insert(i);
                rawTensorSet.insert(i->tensor);
            }
            for (auto& o : op->GetOOperands()) {
                tensorSet.insert(o);
                rawTensorSet.insert(o->tensor);
            }
        }
        std::vector<std::shared_ptr<RawTensor>> rawTensorList(rawTensorSet.begin(), rawTensorSet.end());
        std::vector<std::shared_ptr<LogicalTensor>> tensorList(tensorSet.begin(), tensorSet.end());
        std::sort(rawTensorList.begin(), rawTensorList.end(),
                  [](auto l, auto r) { return l->GetRawMagic() < r->GetRawMagic(); });
        std::sort(tensorList.begin(), tensorList.end(), [](auto l, auto r) { return l->GetMagic() < r->GetMagic(); });

        Json rawtensors = Json::array();
        Json tensors = Json::array();
        for (auto& rawTensor : rawTensorList) {
            rawtensors.push_back(rawTensor->DumpJson());
        }
        for (auto& tensor : tensorList) {
            tensors.push_back(tensor->DumpJson(*this, false));
        }
        funcJson["rawtensors"] = rawtensors;
        funcJson["tensors"] = tensors;
    }
    if (functionType_ == FunctionType::DYNAMIC_LOOP) {
        auto loopAttr = GetDynloopAttribute();
        if (loopAttr != nullptr) {
            // itername
            std::string itername = loopAttr->iterSymbolName;
            funcJson["dynamic"]["itername"] = itername;

            // begin
            SymbolicScalar begin = loopAttr->Begin();
            auto jbegin = ToJson(begin);
            if (jbegin.size() > 0) {
                funcJson["dynamic"]["begin"] = jbegin;
            }

            // end
            SymbolicScalar end = loopAttr->End();
            auto jend = ToJson(end);
            if (jend.size() > 0) {
                funcJson["dynamic"]["end"] = jend;
            }

            // step
            SymbolicScalar step = loopAttr->Step();
            auto jstep = ToJson(step);
            if (jstep.size() > 0) {
                funcJson["dynamic"]["step"] = jstep;
            }

            // originalBegin
            SymbolicScalar originalBegin = loopAttr->originalRange.Begin();
            auto jOriBegin = ToJson(originalBegin);
            if (jOriBegin.size() > 0) {
                funcJson["dynamic"]["originalBegin"] = jOriBegin;
            }

            // originalEnd
            SymbolicScalar originalEnd = loopAttr->originalRange.End();
            auto jOriEnd = ToJson(originalEnd);
            if (jOriEnd.size() > 0) {
                funcJson["dynamic"]["originalEnd"] = jOriEnd;
            }

            // unrollTimes
            int unrollTimes = loopAttr->unrollTimes;
            funcJson["dynamic"]["unrollTimes"] = unrollTimes;

            // pathList
            Json loopFuncPathList = Json::array();
            for (auto& path : loopAttr->pathList) {
                Json pathJson = Json::array();
                auto opmagic = path.callop->GetOpMagic();
                for (auto& pathCond : path.pathCondList) {
                    Json pathCondJson = Json::array();
                    SymbolicScalar cond = pathCond.GetCond();
                    auto jcond = ToJson(cond);
                    pathCondJson.push_back(jcond);
                    pathCondJson.push_back(pathCond.IsSat());
                    pathJson.push_back(pathCondJson);
                }
                if (pathJson.size() > 0) {
                    loopFuncPathList.push_back({opmagic, pathJson});
                }
            }
            if (loopFuncPathList.size() > 0) {
                funcJson["dynamic"]["paths"] = loopFuncPathList;
            }
        }
    }
    return funcJson;
}

void Function::LoadTensorJson(const std::shared_ptr<Function>& func, const Json& tensorJson,
                              const std::unordered_map<int, std::shared_ptr<RawTensor>>& rawTensorDict,
                              std::unordered_map<int, std::shared_ptr<LogicalTensor>>& tensorDict)
{
    if (tensorJson.count("tensors") != 0) {
        for (auto& tensorDump : tensorJson["tensors"]) {
            std::shared_ptr<LogicalTensor> tensor = LogicalTensor::LoadJson(*func, rawTensorDict, tensorDump);
            tensorDict[tensor->GetMagic()] = tensor;
        }
    }
    for (auto& iDump : tensorJson["incasts"]) {
        if (!iDump[0].is_number()) {
            std::shared_ptr<LogicalTensor> tensor = LogicalTensor::LoadJson(*func, rawTensorDict, iDump);
            tensorDict[tensor->GetMagic()] = tensor;
        }
    }
    for (auto& oDump : tensorJson["outcasts"]) {
        if (!oDump[0].is_number()) {
            std::shared_ptr<LogicalTensor> tensor = LogicalTensor::LoadJson(*func, rawTensorDict, oDump);
            tensorDict[tensor->GetMagic()] = tensor;
        }
    }

    for (auto& tDump : tensorJson["global_tensors"]) {
        int magic = tDump.get<int>();
        auto& t = tensorDict[magic];
        func->globalTensors_.emplace(t);
    }

    for (auto& iDump : tensorJson["incasts"]) {
        int magic = iDump[0].is_number() ? iDump[0].get<int>() : iDump["magic"].get<int>();
        auto& in = tensorDict[magic];
        func->inCasts_.push_back(in);
        func->GetTensorMap().Insert(in);
    }
    for (auto& oDump : tensorJson["outcasts"]) {
        int magic = oDump[0].is_number() ? oDump[0].get<int>() : oDump["magic"].get<int>();
        func->outCasts_.push_back(tensorDict[magic]);
    }
    /* 补充对于临时workspace的tensor添加 */
    for (auto& ele : tensorDict) {
        func->GetTensorMap().Insert(ele.second, false);
    }

    for (auto& opDump : tensorJson["operations"]) {
        auto op = Operation::LoadJson(*func, tensorDict, opDump);
        func->operations_.push_back(op);
    }
}

std::shared_ptr<Function> Function::LoadJson(Program& belongTo, const Json& funcJson)
{
    FE_ASSERT(FeError::INVALID_VAL, funcJson[T_FIELD_KIND].get<int>() == static_cast<int>(Kind::T_KIND_FUNCTION))
        << "Invalid function kind in JSON";
    int funcmagic = funcJson["funcmagic"].get<int>();
    std::string rawname = funcJson["rawname"].get<std::string>();
    std::shared_ptr<Function> func = std::make_shared<Function>(belongTo, rawname + "_" + std::to_string(funcmagic),
                                                                rawname, nullptr);
    func->funcMagicName_ = funcJson["func_magicname"];
    func->functionMagic_ = funcmagic;
    func->functionType_ = static_cast<FunctionType>(funcJson["functype"].get<int>());
    func->graphType_ = static_cast<GraphType>(funcJson["graphtype"].get<int>());
    func->sorted_ = true;
    std::unordered_map<int, std::shared_ptr<RawTensor>> rawTensorDict;
    if (funcJson.count("rawtensors") != 0) {
        for (auto& rawTensorDump : funcJson["rawtensors"]) {
            std::shared_ptr<RawTensor> rawTensor = RawTensor::LoadJson(rawTensorDump);
            rawTensorDict[rawTensor->rawmagic] = rawTensor;
        }
    }
    for (auto& iDump : funcJson["incasts"]) {
        if (!iDump[0].is_number() && !iDump[T_FIELD_RAWTENSOR].is_number()) {
            std::shared_ptr<RawTensor> rawTensor = RawTensor::LoadJson(iDump[T_FIELD_RAWTENSOR]);
            rawTensorDict[rawTensor->rawmagic] = rawTensor;
        }
    }
    for (auto& oDump : funcJson["outcasts"]) {
        if (!oDump[0].is_number() && !oDump[T_FIELD_RAWTENSOR].is_number()) {
            std::shared_ptr<RawTensor> rawTensor = RawTensor::LoadJson(oDump[T_FIELD_RAWTENSOR]);
            rawTensorDict[rawTensor->rawmagic] = rawTensor;
        }
    }

    std::unordered_map<int, std::shared_ptr<LogicalTensor>> tensorDict;
    LoadTensorJson(func, funcJson, rawTensorDict, tensorDict);
    func->opSeed_ = funcJson["_opseed"].get<int>();
    IdGen<IdType::RAW_TENSOR>::Inst().SetId(funcJson["_rawid"].get<int>());
    int funcid = funcJson["_funcid"].get<int>();
    IdGen<IdType::FUNCTION>::Inst().SetId(funcid);
    func->paramConfigs_.sgPgLowerBound = funcJson["_sg_pg_lowerbound"].get<int>();
    func->paramConfigs_.sgParallelNum = funcJson["_sg_parallel_num"].get<int>();
    func->paramConfigs_.sgPartitionAlgorithm = funcJson["_sg_partition_algorithm"].get<std::string>();
    func->paramConfigs_.sgMgCopyInUpperBound = funcJson["_sg_mg_copyin_upper_bound"].get<int>();
    func->paramConfigs_.mgVecParallelLb = funcJson["_mg_vec_parallel_lb"].get<int>();
    func->paramConfigs_.autoMixPartition = funcJson["_auto_mix_partition"].get<int>();
    auto subGraphCount = funcJson["_total_subgraph_count"].get<size_t>();
    func->SetTotalSubGraphCount(subGraphCount);

    std::vector<std::vector<int>> incastSlot;
    for (auto& iDump : funcJson["incasts"]) {
        std::vector<int> iSlot;
        if (iDump[0].is_number()) {
            for (auto& slot : iDump[1]) {
                iSlot.push_back(slot.get<int>());
            }
            incastSlot.push_back(iSlot);
        }
    }
    std::vector<std::vector<int>> outcastSlot;
    for (auto& oDump : funcJson["outcasts"]) {
        std::vector<int> oSlot;
        if (oDump[0].is_number()) {
            for (auto& slot : oDump[1]) {
                oSlot.push_back(slot.get<int>());
            }
            outcastSlot.push_back(oSlot);
        }
    }
    IncastOutcastSlot ioSlot;
    ioSlot.incastSlot = incastSlot;
    ioSlot.outcastSlot = outcastSlot;
    std::shared_ptr<TensorSlotScope> tensorSlotScope = std::make_shared<TensorSlotScope>(func.get());
    tensorSlotScope->ioslot = ioSlot;
    func->slotScope_ = tensorSlotScope;
    func->ComputeHashOrderless();
    func->functionHash_ = std::stoull(funcJson["hash"].get<std::string>());

    if (func->GetGraphType() == GraphType::BLOCK_GRAPH && func->GetLeafFuncAttribute() == nullptr) {
        std::shared_ptr<LeafFuncAttribute> attr = std::make_shared<LeafFuncAttribute>();
        func->SetLeafFuncAttribute(attr);
    }
    if (funcJson.count("leaf_func_attr") != 0 && funcJson["leaf_func_attr"].count("coretype") != 0) {
        std::shared_ptr<LeafFuncAttribute> attr = func->GetLeafFuncAttribute();
        attr->coreType = static_cast<CoreType>(funcJson["leaf_func_attr"]["coretype"].get<int>());
    }

    if (funcJson.count("root_func_magic") != 0) {
        func->rootFunc_ = belongTo.GetFunctionByMagic(funcJson["root_func_magic"].get<int>()).get();
    }
    if (funcJson.count("programs") != 0) {
        uint64_t index = 0;
        for (auto& programMagic : funcJson["programs"]) {
            func->programs_.emplace(
                std::make_pair(index++, belongTo.GetFunctionByMagic(programMagic.get<int>()).get()));
        }
    }

    if (funcJson.count("topo") != 0) {
        func->topoInfo_.LoadJson(funcJson["topo"]);
    }

    if (funcJson.count("subfunc_param") != 0) {
        func->parameter_.FromJson(funcJson["subfunc_param"]);
    }

    if (funcJson.count("aic_ready_subgraph_ids") != 0) {
        func->SetReadySubGraphIds(CoreType::AIC, funcJson["aic_ready_subgraph_ids"].get<std::vector<int>>());
    }
    if (funcJson.count("aiv_ready_subgraph_ids") != 0) {
        func->SetReadySubGraphIds(CoreType::AIV, funcJson["aiv_ready_subgraph_ids"].get<std::vector<int>>());
    }

    if (funcJson.count("aicpu_ready_subgraph_ids") != 0) {
        func->SetReadySubGraphIds(CoreType::AICPU, funcJson["aicpu_ready_subgraph_ids"].get<std::vector<int>>());
    }

    if (funcJson.count("dynamic") != 0) {
        auto iterName = funcJson["dynamic"]["itername"];
        auto beginJson = funcJson["dynamic"]["begin"];
        SymbolicScalar begin = LoadSymbolicScalar(beginJson);
        auto endJson = funcJson["dynamic"]["end"];
        SymbolicScalar end = LoadSymbolicScalar(endJson);
        auto stepJson = funcJson["dynamic"]["step"];
        SymbolicScalar step = LoadSymbolicScalar(stepJson);
        LoopRange range(begin, end, step);
        auto originalBeginJson = funcJson["dynamic"]["originalBegin"];
        SymbolicScalar originalBegin = LoadSymbolicScalar(originalBeginJson);
        auto originalEndJson = funcJson["dynamic"]["originalEnd"];
        SymbolicScalar originalEnd = LoadSymbolicScalar(originalEndJson);
        LoopRange originalRange(originalBegin, originalEnd);
        auto attr = std::make_shared<DynloopFunctionAttribute>(iterName, range, originalRange);
        attr->unrollTimes = funcJson["dynamic"]["unrollTimes"];
        auto dynFuncDump = funcJson["dynamic"];
        if (dynFuncDump.count("paths") != 0) {
            std::vector<DynloopFunctionPath> pathList;
            auto pathsJson = funcJson["dynamic"]["paths"];
            for (auto& pathJson : pathsJson) {
                Function* root = func.get();
                std::vector<DynloopFunctionPathCondition> pathCondList;
                auto callOpMagic = pathJson[0];
                Operation* callop = nullptr;
                for (auto& op : func->Operations().DuplicatedOpList()) {
                    if (op->GetOpMagic() == callOpMagic) {
                        callop = op;
                        break;
                    }
                }
                for (auto& pathCondJson : pathJson[1]) {
                    bool isSat = static_cast<bool>(pathCondJson[1]);
                    SymbolicScalar cond = LoadSymbolicScalar(pathCondJson[0]);
                    DynloopFunctionPathCondition pathCond;
                    pathCond.isSat_ = isSat;
                    pathCond.cond_ = cond;
                    pathCondList.push_back(pathCond);
                }
                DynloopFunctionPath path(root, pathCondList, callop);
                pathList.push_back(path);
            }
            attr->pathList = pathList;
        }
        func->SetDynloopAttribute(attr);
    }
    func->RefreshOpPosition();
    return func;
}

static const SymbolicScalar RUNTIME_COA_GetOffset = AddRuntimeCoaPrefix("GET_PARAM_OFFSET");
static const SymbolicScalar RUNTIME_COA_GetValidShape = AddRuntimeCoaPrefix("GET_PARAM_VALID_SHAPE");
static const SymbolicScalar RUNTIME_COA_GetRawShape = AddRuntimeCoaPrefix("GET_PARAM_RAW_SHAPE");
static const SymbolicScalar RUNTIME_COA_GetParam = AddRuntimeCoaPrefix("GET_PARAM");

static int64_t MakeTensorIndex(int64_t magic)
{
    return magic | RAW_TENSOR_INDEX_BIT_MASK;
} // Create tensor index by setting bit 62 of magic number

static void MaybeNormalizeValue(const SymbolicScalar& coaFunc, std::vector<SymbolicScalar>& operandCoaList,
                                int operandCoaIndex, std::vector<OpImmediate>& opImmList, int coaIndex,
                                bool valueToIndex)
{
    for (size_t dimIndex = 0; dimIndex < opImmList.size(); dimIndex++) {
        auto& opImm = opImmList[dimIndex];
        SymbolicScalar scalar = opImm.GetSpecifiedValue();
        auto getTensorDataDict = GetTensorDataDict(scalar);
        if (getTensorDataDict.size() == 0) {
            OpImmediate::NormalizeValue(operandCoaList[operandCoaIndex + dimIndex], opImm,
                                        coaFunc(opImmList.size(), coaIndex, dimIndex), valueToIndex);
        }
    }
};

static void MaybeNormalizeValue(std::vector<SymbolicScalar>& valueCoa, SymbolicScalar& value, int coaIndex,
                                bool valueToIndex)
{
    auto getTensorDataDict = GetTensorDataDict(value);
    if (getTensorDataDict.size() == 0) {
        valueCoa.push_back(value);
        if (valueToIndex) {
            value = RUNTIME_COA_GetParam(coaIndex);
        }
    }
}

static std::vector<SymbolicScalar> MaybeNormalizeValue(std::vector<OpImmediate>& opImmList, int coaIndex,
                                                       bool valueToIndex)
{
    std::vector<SymbolicScalar> coaList;
    auto values = OpImmediate::ToSpecified(opImmList);
    for (auto& value : values) {
        MaybeNormalizeValue(coaList, value, coaIndex, valueToIndex);
        coaIndex += 1;
    }
    opImmList = OpImmediate::Specified(values);
    return coaList;
}

static std::vector<SymbolicScalar> NormalizeCopyIn(Operation* op, int coaIndexBase, bool valueToIndex)
{
    auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
    int dim = copyAttr->GetShape().size();
    int operandCoaIndex = COA_INDEX_DIM_BASE;
    int coaIndex = coaIndexBase + COA_INDEX_DIM_BASE;
    int dimCount = dim * 0x3 + copyAttr->GetToDynValidShape().size(); // toValidShape may has different dim in conv
    std::vector<SymbolicScalar> operandCoaList(COA_INDEX_DIM_BASE + dimCount, 0);

    operandCoaList[0] = MakeTensorIndex(op->GetIOperands()[0]->GetRawMagic());

    auto opImmList = copyAttr->GetFromOffset();
    MaybeNormalizeValue(RUNTIME_COA_GetOffset, operandCoaList, operandCoaIndex, opImmList, coaIndexBase, valueToIndex);
    copyAttr->SetFromOffset(opImmList);
    operandCoaIndex += dim;
    coaIndex += dim;

    // shape to normal
    opImmList = copyAttr->GetShape();
    OpImmediate::NormalizeValue(operandCoaList, operandCoaIndex, opImmList, coaIndex, valueToIndex);
    operandCoaIndex += dim;
    coaIndex += dim;

    opImmList = copyAttr->GetRawShape();
    MaybeNormalizeValue(RUNTIME_COA_GetRawShape, operandCoaList, operandCoaIndex, opImmList, coaIndexBase,
                        valueToIndex);
    copyAttr->SetRawShape(opImmList);
    operandCoaIndex += dim;
    coaIndex += dim;

    opImmList = copyAttr->GetToDynValidShape();
    if (op->GetOpcode() == Opcode::OP_L1_COPY_IN_CONV) {
        auto valueCoaList = MaybeNormalizeValue(opImmList, coaIndex, false);
        coaIndex += valueCoaList.size();
        operandCoaList.erase(operandCoaList.end() - valueCoaList.size(), operandCoaList.end());
        operandCoaList.insert(operandCoaList.end(), valueCoaList.begin(), valueCoaList.end());
    } else {
        MaybeNormalizeValue(RUNTIME_COA_GetValidShape, operandCoaList, operandCoaIndex, opImmList, coaIndexBase,
                            valueToIndex);
    }
    copyAttr->SetToDynValidShape(opImmList);

    return operandCoaList;
}

static std::vector<SymbolicScalar> NormalizeCopyOut(Operation* op, int coaIndexBase, bool valueToIndex)
{
    auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
    int dim = copyAttr->GetShape().size();
    int operandCoaIndex = COA_INDEX_DIM_BASE;
    int coaIndex = coaIndexBase + COA_INDEX_DIM_BASE;
    int dimCount = dim * 0x3 + copyAttr->GetFromDynValidShape().size();
    std::vector<SymbolicScalar> operandCoaList(COA_INDEX_DIM_BASE + dimCount, 0);

    operandCoaList[0] = MakeTensorIndex(op->GetOOperands()[0]->GetRawMagic());

    auto opImmList = copyAttr->GetToOffset();
    MaybeNormalizeValue(RUNTIME_COA_GetOffset, operandCoaList, operandCoaIndex, opImmList, coaIndexBase, valueToIndex);
    copyAttr->SetToOffset(opImmList);
    operandCoaIndex += dim;
    coaIndex += dim;

    // shape to normals
    opImmList = copyAttr->GetShape();
    OpImmediate::NormalizeValue(operandCoaList, operandCoaIndex, opImmList, coaIndex, valueToIndex);
    operandCoaIndex += dim;
    coaIndex += dim;

    opImmList = copyAttr->GetRawShape();
    MaybeNormalizeValue(RUNTIME_COA_GetRawShape, operandCoaList, operandCoaIndex, opImmList, coaIndexBase,
                        valueToIndex);
    copyAttr->SetRawShape(opImmList);
    operandCoaIndex += dim;
    coaIndex += dim;

    opImmList = copyAttr->GetFromDynValidShape();
    if (op->GetOpcode() == Opcode::OP_L0C_COPY_OUT_CONV) {
        auto valueCoaList = MaybeNormalizeValue(opImmList, coaIndex, false);
        coaIndex += valueCoaList.size();
        operandCoaList.erase(operandCoaList.end() - valueCoaList.size(), operandCoaList.end());
        operandCoaList.insert(operandCoaList.end(), valueCoaList.begin(), valueCoaList.end());
    } else {
        MaybeNormalizeValue(RUNTIME_COA_GetValidShape, operandCoaList, operandCoaIndex, opImmList, coaIndexBase,
                            valueToIndex);
    }
    copyAttr->SetFromDynValidShape(opImmList);

    return operandCoaList;
}

static std::vector<SymbolicScalar> NormalizeTensor(LogicalTensorPtr operand, int coaIndexBase, bool valueToIndex,
                                                   bool isNop = false)
{
    auto offset = OpImmediate::Specified(operand->GetOffset());
    auto dynOffset = OpImmediate::Specified(operand->GetDynOffset());
    auto shape = OpImmediate::Specified(operand->GetShape());
    auto rawshape = OpImmediate::Specified(operand->GetRawTensor()->GetRawShape());
    auto dynRawshape = OpImmediate::Specified(operand->GetRawTensor()->GetDynRawShape());
    auto dynValidShape = OpImmediate::Specified(operand->GetDynValidShape());
    if (isNop) {
        offset = OpImmediate::Specified(Offset(operand->GetShape().size()));
        dynOffset = OpImmediate::Specified(Offset(operand->GetShape().size()));
        shape = OpImmediate::Specified(Shape(operand->GetShape().size()));
        dynValidShape = OpImmediate::Specified(Shape(operand->GetShape().size()));
    }

    int dim = shape.size();
    int operandCoaIndex = COA_INDEX_DIM_BASE;
    int coaIndex = coaIndexBase + COA_INDEX_DIM_BASE;
    std::vector<SymbolicScalar> operandCoaList(COA_INDEX_DIM_BASE + dim * COA_INDEX_TYPE_COUNT, 0);

    if (operand->GetMemoryTypeToBe() == MemoryType::MEM_DEVICE_DDR) {
        operandCoaList[0] = MakeTensorIndex(operand->GetRawMagic());
    }

    if (!dynOffset.empty()) {
        MaybeNormalizeValue(RUNTIME_COA_GetOffset, operandCoaList, operandCoaIndex, dynOffset, coaIndexBase,
                            valueToIndex);
        operand->UpdateOffset(TensorOffset{operand->GetOffset(), OpImmediate::ToSpecified(dynOffset)});
    } else {
        OpImmediate::NormalizeValue(operandCoaList, operandCoaIndex, offset, coaIndex, false);
    }
    operandCoaIndex += dim;
    coaIndex += dim;

    OpImmediate::NormalizeValue(operandCoaList, operandCoaIndex, shape, coaIndex, false);
    operandCoaIndex += dim;
    coaIndex += dim;

    if (!dynRawshape.empty()) {
        OpImmediate::NormalizeValue(operandCoaList, operandCoaIndex, dynRawshape, coaIndex, valueToIndex);
    } else {
        OpImmediate::NormalizeValue(operandCoaList, operandCoaIndex, rawshape, coaIndex, false);
    }
    operandCoaIndex += dim;
    coaIndex += dim;

    if (!dynValidShape.empty()) {
        MaybeNormalizeValue(RUNTIME_COA_GetValidShape, operandCoaList, operandCoaIndex, dynValidShape, coaIndexBase,
                            valueToIndex);
        operand->UpdateDynValidShape(OpImmediate::ToSpecified(dynValidShape));
    } else {
        OpImmediate::NormalizeValue(operandCoaList, operandCoaIndex, shape, coaIndex, false);
    }
    operandCoaIndex += dim;
    coaIndex += dim;

    return operandCoaList;
}

void Function::GetOutcastSymbolicExpr(std::map<int, SymbolicScalar>& tabel)
{
    for (size_t idx = 0; idx < outCasts_.size(); idx++) {
        auto op = *outCasts_[idx]->GetProducers().begin();
        if (op->GetOpcode() == Opcode::OP_BIND_TENSOR) {
            if (op->HasAttr(OpAttributeKey::bindTensor) && (op->GetOOperands().size() == 1UL)) {
                tabel[idx] = op->GetSymbolicScalarAttribute(OpAttributeKey::bindTensor);
            }
        }
    }
}

std::vector<std::vector<SymbolicScalar>> Function::NormalizeCoa(std::vector<OperandAttribute>& iOpAttr,
                                                                std::vector<OperandAttribute>& oOpAttr)
{
    std::unordered_map<int, Operation*> opmagicToOp;
    std::unordered_map<LogicalTensorPtr, int> normTensors;

    opmagicToOp.reserve(operations_.size());
    for (auto& op : operations_) {
        opmagicToOp[op->GetOpMagic()] = op.get();
    }

    int coaIndex = COA_INDEX_BASE;
    std::vector<std::vector<SymbolicScalar>> coaLists;
    coaLists.reserve(incastPosition.size() + outcastPosition.size());
    NormalizeCoaForInCasts(iOpAttr, coaLists, coaIndex, normTensors, opmagicToOp);
    NormalizeCoaForOutCasts(oOpAttr, coaLists, coaIndex, normTensors, opmagicToOp);
    NormalizeCoaForNormalOperands(coaLists, coaIndex, normTensors);
    NormalizeCoaForSpecialInfo(coaLists, coaIndex);

    return coaLists;
}

static bool IsInplaceIncast(Operation* op, std::vector<Operation*>& copyInList)
{
    if (!IsViewLike(op->GetOpcode()) && op->GetOpcode() != Opcode::OP_RESHAPE) {
        return false;
    }
    LogicalTensorPtr data = op->GetOOperands()[0];
    copyInList.clear();
    for (auto oop : data->GetConsumers()) {
        if (OpcodeManager::Inst().IsSharedMemory(oop->GetOpcode())) {
            return false;
        }
        if (IsCopyIn(oop->GetOpcode())) {
            copyInList.push_back(oop);
        }
    }
    return true;
}

// 出口侧链式检测：outcast 登记在 RESHAPE/view-like 上，其输入 tensor 的全部 producer 都是 COPY_OUT。
// 与入口侧 IsInplaceIncast 对称，用于把 COPY_OUT→RESHAPE 出口链上的折叠 offset 纳入 COA 参数化。
static bool IsOutcastCopyOutChain(Operation* op, std::vector<Operation*>& copyOutList)
{
    if (!IsViewLike(op->GetOpcode()) && op->GetOpcode() != Opcode::OP_RESHAPE) {
        return false;
    }
    LogicalTensorPtr data = op->GetIOperands().front();
    copyOutList.clear();
    for (auto prod : data->GetProducers()) {
        if (!IsCopyOut(prod->GetOpcode())) {
            return false;
        }
        copyOutList.push_back(prod);
    }
    return !copyOutList.empty();
}

void Function::NormalizeCoaForInCasts(std::vector<OperandAttribute>& iOpAttr,
                                      std::vector<std::vector<SymbolicScalar>>& coaLists, int& coaIndex,
                                      std::unordered_map<LogicalTensorPtr, int>& normTensors,
                                      const std::unordered_map<int, Operation*>& opmagicToOp)
{
    bool valueToIndex = parent_->GetFunctionType() == FunctionType::DYNAMIC_LOOP_PATH;
    iOpAttr.reserve(incastPosition.size());
    for (auto [opmagic, k] : incastPosition) {
        auto op = opmagicToOp.at(opmagic);
        if (op->GetIOpAttrOffset(k) != -1) {
            continue;
        }
        std::vector<SymbolicScalar> operandCoaList;
        if (IsCopyIn(op->GetOpcode()) && k == 0) {
            operandCoaList = NormalizeCopyIn(op, coaIndex, valueToIndex);
            if (CheckEmuOpcode(op, EMUOP_TENSOR_GETDATA_DEPEND)) {
                GetTensorDataSetCoaIndex(op, coaIndex);
            }
        } else {
            std::vector<Operation*> copyInList;
            auto& consOp = *(op->GetOOperands()[0])->GetConsumers().begin();
            if (!this->IsFromOutCast(op->GetOOperands().front()) && IsCopyIn(consOp->GetOpcode()) &&
                IsInplaceIncast(op, copyInList)) {
                bool isReshape = op->GetOpcode() == Opcode::OP_RESHAPE;
                for (auto copyIn : copyInList) {
                    operandCoaList = NormalizeCopyIn(copyIn, coaIndex, valueToIndex);
                    copyIn->SetIOpAtt(isReshape ? 0 : k, coaIndex);
                    if (!isReshape) {
                        iOpAttr.emplace_back(coaIndex);
                    }
                    coaIndex += operandCoaList.size();
                    coaLists.emplace_back(std::move(operandCoaList));
                }
                if (isReshape) {
                    operandCoaList = NormalizeTensor(op->GetInputOperand(k), coaIndex, false);
                    op->SetIOpAtt(k, coaIndex);
                    iOpAttr.emplace_back(coaIndex);
                    coaIndex += operandCoaList.size();
                    coaLists.emplace_back(std::move(operandCoaList));
                }
                continue;
            }
            auto iOperand = op->GetInputOperand(k);
            auto it = normTensors.find(iOperand);
            if (it != normTensors.end()) {
                op->SetIOpAtt(k, it->second);
                iOpAttr.emplace_back(it->second);
                continue;
            }
            operandCoaList = NormalizeTensor(iOperand, coaIndex, false, op->GetOpcode() == Opcode::OP_NOP);
            normTensors.emplace(iOperand, coaIndex);
        }
        op->SetIOpAtt(k, coaIndex);
        iOpAttr.emplace_back(coaIndex);
        coaIndex += operandCoaList.size();
        coaLists.emplace_back(std::move(operandCoaList));
    }
}

void Function::NormalizeCoaForOutCasts(std::vector<OperandAttribute>& oOpAttr,
                                       std::vector<std::vector<SymbolicScalar>>& coaLists, int& coaIndex,
                                       std::unordered_map<LogicalTensorPtr, int>& normTensors,
                                       const std::unordered_map<int, Operation*>& opmagicToOp)
{
    bool valueToIndex = parent_->GetFunctionType() == FunctionType::DYNAMIC_LOOP_PATH;
    oOpAttr.reserve(outcastPosition.size());
    for (auto [opmagic, k] : outcastPosition) {
        auto op = opmagicToOp.at(opmagic);
        if (op->GetOOpAttrOffset(k) != -1) {
            continue;
        }
        bool isAtomic = op->GetOOperands()[0]->HasAttr(OpAttributeKey::writeConflict);
        std::vector<SymbolicScalar> operandCoaList;
        std::vector<Operation*> copyOutList;
        if (IsCopyOut(op->GetOpcode()) && k == 0) {
            operandCoaList = NormalizeCopyOut(op, coaIndex, valueToIndex);
        } else if (op->GetIOperands().size() > 0 && !this->IsFromInCast(op->GetIOperands().front()) &&
                   IsOutcastCopyOutChain(op, copyOutList)) {
            bool isReshape = op->GetOpcode() == Opcode::OP_RESHAPE;
            for (auto copyOut : copyOutList) {
                if (copyOut->GetOOpAttrOffset(0) != -1) {
                    continue;
                }
                operandCoaList = NormalizeCopyOut(copyOut, coaIndex, valueToIndex);
                copyOut->SetOOpAtt(isReshape ? 0 : k, coaIndex, false);
                if (!isReshape) {
                    oOpAttr.emplace_back(coaIndex, isAtomic);
                }
                coaIndex += operandCoaList.size();
                coaLists.emplace_back(std::move(operandCoaList));
            }
            if (isReshape) {
                operandCoaList = NormalizeTensor(op->GetOutputOperand(k), coaIndex, false);
                op->SetOOpAtt(k, coaIndex, isAtomic);
                oOpAttr.emplace_back(coaIndex, isAtomic);
                coaIndex += operandCoaList.size();
                coaLists.emplace_back(std::move(operandCoaList));
            }
            continue;
        } else {
            auto oOperand = op->GetOutputOperand(k);
            auto it = normTensors.find(oOperand);
            if (it != normTensors.end()) {
                op->SetOOpAtt(k, it->second, isAtomic);
                oOpAttr.emplace_back(it->second, isAtomic);
                continue;
            }
            operandCoaList = NormalizeTensor(oOperand, coaIndex, false);
            normTensors.emplace(oOperand, coaIndex);
        }
        op->SetOOpAtt(k, coaIndex, isAtomic);
        oOpAttr.emplace_back(coaIndex, isAtomic);
        coaIndex += operandCoaList.size();
        coaLists.emplace_back(std::move(operandCoaList));
    }
}

void Function::NormalizeCoaForNormalOperands(std::vector<std::vector<SymbolicScalar>>& coaLists, int& coaIndex,
                                             std::unordered_map<LogicalTensorPtr, int>& normTensors)
{
    std::unordered_set<LogicalTensorPtr> inOutCasts;
    inOutCasts.insert(inCasts_.begin(), inCasts_.end());
    inOutCasts.insert(outCasts_.begin(), outCasts_.end());
    std::unordered_set<int> outcastRawMagics;
    for (const auto& outcast : outCasts_) {
        if (outcast != nullptr) {
            outcastRawMagics.insert(outcast->GetRawMagic());
        }
    }

    bool valueToIndex = parent_->GetFunctionType() == FunctionType::DYNAMIC_LOOP_PATH;
    for (auto& op : operations_) {
        if (op->GetOpcode() == Opcode::OP_NOP) {
            continue;
        }
        if ((IsCopyIn(op->GetOpcode()) || OpcodeManager::Inst().IsCopyFromL1(op->GetOpcode())) &&
            !op->GetIOperands().empty() && op->GetIOpAttrOffset(0) == -1 && op->GetOpAttribute() != nullptr) {
            auto operandCoaList = NormalizeCopyIn(op.get(), coaIndex, valueToIndex);
            // Inner copy attr still goes into coaLists; only outcast-raw copies may occupy IOpAtt,
            // which InferParamIndex treats as tensor validshape COA.
            if (outcastRawMagics.count(op->GetInputOperand(0)->GetRawMagic()) > 0) {
                op->SetIOpAtt(0, coaIndex);
            }
            coaIndex += operandCoaList.size();
            coaLists.emplace_back(std::move(operandCoaList));
        }
        if (IsCopyOut(op->GetOpcode()) && !op->GetOOperands().empty() && op->GetOOpAttrOffset(0) == -1 &&
            op->GetOpAttribute() != nullptr) {
            auto operandCoaList = NormalizeCopyOut(op.get(), coaIndex, valueToIndex);
            if (outcastRawMagics.count(op->GetOutputOperand(0)->GetRawMagic()) > 0) {
                op->SetOOpAtt(0, coaIndex, false);
            }
            coaIndex += operandCoaList.size();
            coaLists.emplace_back(std::move(operandCoaList));
        }
        for (size_t i = 0; i < op->GetInputOperandSize(); i++) {
            auto iOperand = op->GetInputOperand(i);
            if (op->GetIOpAttrOffset(i) != -1) {
                continue;
            }
            if (inOutCasts.count(iOperand) > 0) {
                continue;
            }
            auto it = normTensors.find(iOperand);
            if (it != normTensors.end()) {
                op->SetIOpAtt(i, it->second);
                continue;
            }
            if (!iOperand->GetDynOffset().empty() || !iOperand->GetDynValidShape().empty()) {
                auto operandCoaList = NormalizeTensor(iOperand, coaIndex, valueToIndex);
                normTensors.emplace(iOperand, coaIndex);
                coaIndex += operandCoaList.size();
                coaLists.emplace_back(std::move(operandCoaList));
            }
        }
        for (size_t i = 0; i < op->GetOutputOperandSize(); i++) {
            auto oOperand = op->GetOutputOperand(i);
            if (op->GetOOpAttrOffset(i) != -1) {
                continue;
            }
            if (inOutCasts.count(oOperand) > 0) {
                continue;
            }
            // 对于Copy类多输出Op，由于InferParamIndex对每个输出都需要通过专属的coaIndex传递Validshape给codegen，故需要给此类op所有输出都进行normalize
            // 如 Opcode::OP_SHMEM_GET
            if (oOperand->GetConsumers().empty() && !OpcodeManager::Inst().IsCopyInOrOut(op->GetOpcode())) {
                continue;
            }
            auto it = normTensors.find(oOperand);
            if (it != normTensors.end()) {
                op->SetOOpAtt(i, it->second, false);
                continue;
            }
            if (!oOperand->GetDynOffset().empty() || !oOperand->GetDynValidShape().empty()) {
                auto operandCoaList = NormalizeTensor(oOperand, coaIndex, valueToIndex);
                normTensors.emplace(oOperand, coaIndex);
                op->SetOOpAtt(i, coaIndex, false);
                coaIndex += operandCoaList.size();
                coaLists.emplace_back(std::move(operandCoaList));
            }
        }
    }
}

void Function::NormalizeCoaForSpecialInfo(std::vector<std::vector<SymbolicScalar>>& coaLists, int& coaIndex)
{
    bool valueToIndex = parent_->GetFunctionType() == FunctionType::DYNAMIC_LOOP_PATH;
    for (auto& op : operations_) {
        if (op->GetOpcode() == Opcode::OP_VEC_DUP || op->GetOpcode() == Opcode::OP_RANGE ||
            op->GetOpcode() == Opcode::OP_TRIUL || op->GetOpcode() == Opcode::OP_UNIFORM) {
            if (op->HasAttr(OpAttributeKey::dynScalar)) {
                SymbolicScalar dynScalar = op->GetSymbolicScalarAttribute(OpAttributeKey::dynScalar);
                std::vector<SymbolicScalar> valueCoaList;
                MaybeNormalizeValue(valueCoaList, dynScalar, coaIndex, valueToIndex);
                op->SetAttribute(OpAttributeKey::dynScalar, dynScalar);
                coaLists.emplace_back(valueCoaList);
                coaIndex += 1;
            }
        } else if (op->GetOpcode() == Opcode::OP_BIND_TENSOR) {
            if (op->HasAttr(OpAttributeKey::bindTensor) && (op->GetOOperands().size() == 1UL)) {
                SymbolicScalar bindTensor = op->GetSymbolicScalarAttribute(OpAttributeKey::bindTensor);
                std::vector<SymbolicScalar> valueCoaList;
                MaybeNormalizeValue(valueCoaList, bindTensor, coaIndex, valueToIndex);
                coaLists.emplace_back(valueCoaList);
                coaIndex += 1;
            }
        } else if (OpcodeManager::Inst().IsTransData(op->GetOpcode())) {
            if (op->HasAttr(OpAttributeKey::transDataOffset)) {
                std::vector<SymbolicScalar> offsets;
                op->GetAttr(OpAttributeKey::transDataOffset, offsets);
                for (auto& offset : offsets) {
                    std::vector<SymbolicScalar> valueCoaList;
                    MaybeNormalizeValue(valueCoaList, offset, coaIndex, valueToIndex);
                    coaLists.emplace_back(valueCoaList);
                    coaIndex += 1;
                }
                op->SetAttribute(OpAttributeKey::transDataOffset, offsets);
            }
        } else if (OpcodeManager::Inst().IsReshapeCopyIn(op->GetOpcode())) {
            // reshape copy need both to and from validshape normalized, from is done during NormalizeCopyIn
            auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
            auto validshape = copyAttr->GetFromDynValidShape();
            auto coaList = MaybeNormalizeValue(validshape, coaIndex, valueToIndex);
            copyAttr->SetFromDynValidShape(validshape);
            coaIndex += coaList.size();
            coaLists.emplace_back(std::move(coaList));
        } else if (OpcodeManager::Inst().IsReshapeCopyOut(op->GetOpcode())) {
            // reshape copy need both to and from validshape normalized, from is done during NormalizeCopyIn
            auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
            auto validshape = copyAttr->GetToDynValidShape();
            auto coaList = MaybeNormalizeValue(validshape, coaIndex, valueToIndex);
            copyAttr->SetToDynValidShape(validshape);
            coaIndex += coaList.size();
            coaLists.emplace_back(std::move(coaList));
        }
    }
}

void Function::DumpTopoFile(const std::string& fileName) const
{
    Json totalTopoJson;
    for (const auto& topo : topoInfo_.GetTopology()) {
        Json sJson;
        sJson["taskId"] = topo.esgId;
        sJson["successors"] = Json::array();
        for (const auto& successor : topo.outGraph) {
            sJson["successors"].push_back(successor);
        }
        int id = operations_[topo.esgId]->GetProgramId();
        if (static_cast<size_t>(id) >= calleeMagicNameList_.size()) {
            continue;
        }
        sJson["funcName"] = calleeMagicNameList_[id];
        sJson["semanticLabel"] = operations_[topo.esgId]->GetSemanticLabelStr();
        totalTopoJson.push_back(sJson);
    }
    std::ofstream ofs(fileName);
    ofs << totalTopoJson.dump(1) << std::endl;
    ofs.close();
}

std::string Function::DumpSSATitle() const
{
    std::stringstream ss;
    ss << GetMagicName() << "[" << functionMagic_ << "]"
       << " " << GetFunctionHash() << " " << GetFunctionTypeNameDict().Find(GetFunctionType()) << " "
       << GetGraphTypeNameDict().Find(GetGraphType());
    return ss.str();
}

std::string Function::DumpSSARawTensor(int indent) const
{
    std::string prefix(indent, ' ');
    std::unordered_set<int> dumpedRawTensor;
    auto dumped = [&dumpedRawTensor](const std::shared_ptr<LogicalTensor>& tensor) -> bool {
        if (dumpedRawTensor.count(tensor->GetRawTensor()->GetRawMagic()) == 0) {
            dumpedRawTensor.insert(tensor->GetRawTensor()->GetRawMagic());
            return false;
        } else {
            return true;
        }
    };
    std::stringstream ss;
    int rawIndex = 0;
    for (size_t i = 0; i < inCasts_.size(); ++i) {
        if (!dumped(inCasts_[i])) {
            ss << prefix << "RAWTENSOR[" << std::setw(SPACE_NUM_THREE) << std::setfill(' ') << rawIndex++ << "] "
               << inCasts_[i]->GetRawTensor()->DumpSSA() << "\n";
        }
    }
    for (size_t i = 0; i < outCasts_.size(); ++i) {
        if (!dumped(outCasts_[i])) {
            ss << prefix << "RAWTENSOR[" << std::setw(SPACE_NUM_THREE) << std::setfill(' ') << rawIndex++ << "] "
               << outCasts_[i]->GetRawTensor()->DumpSSA() << "\n";
        }
    }
    for (size_t i = 0; i < operations_.size(); ++i) {
        for (auto& input : operations_[i]->GetIOperands()) {
            if (!dumped(input)) {
                ss << prefix << "RAWTENSOR[" << std::setw(SPACE_NUM_THREE) << std::setfill(' ') << rawIndex++ << "] "
                   << input->GetRawTensor()->DumpSSA() << "\n";
            }
        }
        for (auto& output : operations_[i]->GetOOperands()) {
            if (!dumped(output)) {
                ss << prefix << "RAWTENSOR[" << std::setw(SPACE_NUM_THREE) << std::setfill(' ') << rawIndex++ << "] "
                   << output->GetRawTensor()->DumpSSA() << "\n";
            }
        }
    }
    return ss.str();
}
std::string Function::DumpSSAIncast(int indent) const
{
    std::string prefix(indent, ' ');
    std::stringstream ss;
    for (size_t i = 0; i < inCasts_.size(); ++i) {
        ss << prefix << "INCAST[" << std::setw(SPACE_NUM_THREE) << std::setfill(' ') << i << "]  "
           << inCasts_[i]->DumpSSA(false, false, true);
        if (slotScope_ && i < slotScope_->ioslot.incastSlot.size()) {
            auto& incastSlotList = slotScope_->ioslot.incastSlot[i];
            ss << " fromSlot[";
            for (size_t k = 0; k < incastSlotList.size(); k++) {
                if (k != 0) {
                    ss << ", ";
                }
                ss << incastSlotList[k];
            }
            ss << "]";
        }
        ss << "\n";
    }
    return ss.str();
}
std::string Function::DumpSSAOutcast(int indent) const
{
    std::string prefix(indent, ' ');
    std::stringstream ss;
    for (size_t i = 0; i < outCasts_.size(); ++i) {
        ss << prefix << "OUTCAST[" << std::setw(SPACE_NUM_THREE) << std::setfill(' ') << i << "]  "
           << outCasts_[i]->DumpSSA(false, false, true);
        if (slotScope_ && i < slotScope_->ioslot.outcastSlot.size()) {
            auto& outcastSlotList = slotScope_->ioslot.outcastSlot[i];
            ss << " toSlot[";
            for (size_t k = 0; k < outcastSlotList.size(); k++) {
                if (k != 0) {
                    ss << ", ";
                }
                ss << outcastSlotList[k];
            }
            ss << "]";
        }
        ss << "\n";
    }
    return ss.str();
}

std::string Function::DumpSSAAttribute(int indent) const
{
    std::string prefix(indent, ' ');
    std::stringstream ss;
    if (IsDynloop()) {
        auto attr = GetDynloopAttribute();
        ss << prefix << "LOOP SYMBOL " << attr->iterSymbolName << "\n";
        ss << prefix << "LOOP BEGIN  " << attr->Begin().Dump() << "\n";
        ss << prefix << "LOOP END    " << attr->End().Dump() << "\n";
        ss << prefix << "LOOP STEP   " << attr->Step().Dump() << "\n";
    }
    return ss.str();
}

constexpr int INDENT_TWO = 2;

std::string Function::DumpSSA() const
{
    std::stringstream ss;
    ss << "\n-------------\n";
    ss << "Function " << DumpSSATitle() << " {\n";
    ss << DumpSSARawTensor(INDENT_TWO) << "\n";
    ss << DumpSSAIncast(INDENT_TWO) << "\n";
    ss << DumpSSAOutcast(INDENT_TWO) << "\n";
    ss << DumpSSAAttribute(INDENT_TWO) << "\n";
    for (size_t i = 0; i < operations_.size(); ++i) {
        auto op = operations_[i];
        ss << op->DumpSSA(PREFIX); // Operation dump
    }
    ss << "}\n";
    return ss.str();
}

std::string Function::Dump() const { return DumpSSA(); }

void Function::DumpFile(const std::string& filePath) const
{
    std::ofstream fout(filePath);
    CHECK(ExternalError::BAD_FD, fout.is_open()) << "Failed to open file: " << filePath;
    fout << Dump();
    fout.close();
}

void Function::UpdateOperandBeforeRemoveOp(Operation& op, const bool keepOutTensor)
{
    // relink, replace input of following op with the input of current op
    if (!op.GetIOperands().empty() && !op.GetOOperands().empty()) {
        LogicalTensorPtr inputTensor = op.GetIOperands().at(0);
        LogicalTensorPtr outputTensor = op.GetOOperands().at(0);
        bool isOutCast = std::find(outCasts_.begin(), outCasts_.end(), outputTensor) != outCasts_.end();
        if (isOutCast || keepOutTensor) {
            outputTensor->RemoveProducer(op);
            for (auto& producer : inputTensor->GetProducers()) {
                outputTensor->AddProducer(*producer);
                if (!inputTensor->GetDynValidShape().empty()) {
                    outputTensor->UpdateDynValidShape(inputTensor->GetDynValidShape());
                }
                producer->ReplaceOutputOperand(inputTensor, outputTensor);
            }
            inputTensor->GetProducers().clear();
            inputTensor->RemoveConsumer(op);
            for (auto& consumer : inputTensor->GetConsumers()) {
                outputTensor->AddConsumer(*consumer);
                consumer->ReplaceInputOperand(inputTensor, outputTensor);
            }
            inputTensor->GetConsumers().clear();
        } else {
            inputTensor->RemoveConsumer(op);
            for (const auto& consumer : outputTensor->GetConsumers()) {
                inputTensor->AddConsumer(consumer);
                consumer->ReplaceInputOperand(outputTensor, inputTensor);
            }
            outputTensor->GetConsumers().clear();
        }
    }
}

/**
 * @brief handle input and output control edges
 * all input ctrl edges shall be moved to output ops of current op
 * all output ctrl edges shall be moved to input ops of current op
 * @param op
 */
void Function::HandleControlOps(Operation& op, std::vector<Operation*>& toRemoveOps) const
{
    const auto& inputCtrlOpSet = op.GetInCtrlOperations();
    if (!inputCtrlOpSet.empty()) {
        auto outputOps = GetAllOutputOperations(op);
        for (auto peerCtrlOp : inputCtrlOpSet) {
            if (peerCtrlOp == nullptr) {
                continue;
            }
            if (peerCtrlOp->OnlyHasCtrlEdgeToOp(op)) {
                op.RemoveInCtrlOperation(*peerCtrlOp);
                toRemoveOps.push_back(peerCtrlOp);
            } else {
                for (auto& outputOp : outputOps) {
                    outputOp->AddInCtrlOperation(*peerCtrlOp);
                }
            }
        }
        op.ClearInCtrlOperations();
    }
    const auto& outputCtrlOpSet = op.GetOutCtrlOperations();
    if (!outputCtrlOpSet.empty()) {
        auto inputOps = GetAllInputOperations(op);
        for (auto& inputOp : inputOps) {
            for (auto outCtrlOp : outputCtrlOpSet) {
                inputOp->AddOutCtrlOperation(*outCtrlOp);
            }
        }
        op.ClearOutCtrlOperations();
    }
}

Operation* Function::GetOpByOpMagic(const int opMagic) const
{
    for (auto op : operations_) {
        if (op->GetOpMagic() == opMagic) {
            return op.get();
        }
    }
    return nullptr;
}

bool Function::TensorReuse(const LogicalTensorPtr& dstTensor, const LogicalTensorPtr& srcTensor)
{
    if (dstTensor == nullptr || srcTensor == nullptr) {
        return false;
    }
    if (dstTensor->Datatype() != srcTensor->Datatype() ||
        dstTensor->tensor->GetRawShapeSize() != srcTensor->tensor->GetRawShapeSize()) {
        FE_LOGI("Data type or raw shape size of src and dst tensor is not same.");
        return false;
    }

    if (dstTensor->tensor->rawshape == srcTensor->tensor->rawshape) {
        dstTensor->tensor = srcTensor->tensor;
    } else {
        dstTensor->tensor->actualRawmagic = srcTensor->tensor->actualRawmagic == -1 ? srcTensor->tensor->rawmagic :
                                                                                      srcTensor->tensor->actualRawmagic;
    }
    return true;
}

bool Function::IsFromInCast(const std::shared_ptr<LogicalTensor>& tensor)
{
    for (auto& t : inCasts_) {
        if (t->GetRawMagic() == tensor->GetRawMagic()) {
            return true;
        }
    }
    return false;
}

bool Function::IsFromOutCast(const std::shared_ptr<LogicalTensor>& tensor)
{
    for (auto& t : outCasts_) {
        if (t->GetRawMagic() == tensor->GetRawMagic()) {
            return true;
        }
    }
    return false;
}

bool Function::IsFromDummyOutCast(int rawMagic)
{
    for (auto& t : outCasts_) {
        if (t->tensor->rawmagic == rawMagic) {
            return true;
        }
    }
    return false;
}

int Function::GetIncastIndex(std::shared_ptr<LogicalTensor>& tensor) const
{
    for (size_t idx = 0; idx < inCasts_.size(); idx++) {
        if (inCasts_[idx] == tensor) {
            return (int)idx;
        }
    }
    return INVALID_IOINDEX;
}

int Function::GetOutcastIndex(std::shared_ptr<LogicalTensor>& tensor) const
{
    for (size_t idx = 0; idx < outCasts_.size(); idx++) {
        if (outCasts_[idx] == tensor) {
            return (int)idx;
        }
    }
    return INVALID_IOINDEX;
}

TensorGraphInfo Function::GetGraphInfo()
{
    std::vector<LogicalTensors> callopInCasts, callopOutCasts;
    std::set<std::shared_ptr<Operation>> viewOpSet, assembleOpSet;
    std::set<std::shared_ptr<LogicalTensor>> iOperandSet, oOperandSet;
    std::vector<std::shared_ptr<Operation>> operations;
    for (auto& op : operations_) {
        if (IsViewLike(op->GetOpcode())) {
            viewOpSet.emplace(op);
            continue;
        }
        if (IsAssembleLike(op->GetOpcode())) {
            assembleOpSet.emplace(op);
            continue;
        }
        FE_ASSERT(FeError::INVALID_VAL, op->GetOpcode() == Opcode::OP_CALL)
            << "Invalid operation code: " << static_cast<int>(op->GetOpcode()) << "\n"
            << "Operation: " << op->Dump();
        operations.emplace_back(op);
        LogicalTensors incasts;
        LogicalTensors outcasts;
        for (auto& iOperand : op->GetIOperands()) {
            const auto& producers = iOperand->GetProducers();
            if (config::EnableSlice() && producers.empty()) {
                continue;
            }
            auto& viewOp = *producers.begin();
            auto& incast = viewOp->GetIOperands()[0];

            iOperandSet.emplace(iOperand);
            incasts.push_back(incast);
        }
        for (auto& oOperand : op->GetOOperands()) {
            const auto& consumers = oOperand->GetConsumers();
            if (config::EnableSlice() && consumers.empty()) {
                continue;
            }
            auto& assembleOp = *consumers.begin();
            auto& outcast = assembleOp->GetOOperands()[0];

            oOperandSet.emplace(oOperand);
            outcasts.push_back(outcast);
        }
        callopInCasts.emplace_back(incasts);
        callopOutCasts.emplace_back(outcasts);
    }
    operations_ = operations;
    return std::make_tuple(std::move(callopInCasts), std::move(callopOutCasts), std::move(viewOpSet),
                           std::move(assembleOpSet), std::move(iOperandSet), std::move(oOperandSet));
}

void Function::ClearUselessLink(TensorGraphInfo& graphInfo)
{
    auto& callopInCasts = std::get<0>(graphInfo);
    auto& callopOutCasts = std::get<1>(graphInfo);
    auto& viewOpSet = std::get<2>(graphInfo);
    auto& assembleOpSet = std::get<3>(graphInfo);
    auto& iOperandSet = std::get<4>(graphInfo);
    auto& oOperandSet = std::get<5>(graphInfo);

    for (auto iOperand : iOperandSet) {
        iOperand->GetProducers().clear();
        iOperand->GetConsumers().clear();
    };
    for (auto oOperand : oOperandSet) {
        oOperand->GetProducers().clear();
        oOperand->GetConsumers().clear();
    };

    for (auto viewOp : viewOpSet) {
        viewOp->GetIOperands().clear();
        viewOp->GetOOperands().clear();
    };

    for (auto assembleOp : assembleOpSet) {
        assembleOp->GetIOperands().clear();
        assembleOp->GetOOperands().clear();
    };

    for (auto incasts : callopInCasts) {
        for (auto incast : incasts) {
            incast->GetConsumers().clear();
        }
    };

    for (auto outcasts : callopOutCasts) {
        for (auto outcast : outcasts) {
            outcast->GetProducers().clear();
        }
    };

    for (auto operation : operations_) {
        operation->GetIOperands().clear();
        operation->GetOOperands().clear();
    };
}

void Function::LinkIoWithCallOp(std::vector<LogicalTensors>& callopInCasts, std::vector<LogicalTensors>& callopOutCasts)
{
    for (size_t idx = 0; idx < operations_.size(); ++idx) {
        auto& incasts = callopInCasts[idx];
        for (auto incast : incasts) {
            incast->AddConsumer(*operations_[idx]);
            operations_[idx]->iOperand.emplace_back(incast);
        }
    }

    for (size_t idx = 0; idx < operations_.size(); ++idx) {
        auto& outcasts = callopOutCasts[idx];
        for (auto outcast : outcasts) {
            outcast->AddProducer(*operations_[idx]);
            operations_[idx]->oOperand.emplace_back(outcast);
        }
    }
}

void Function::RemoveCallOpViewAssemble()
{
    auto graphInfo = GetGraphInfo();
    ClearUselessLink(graphInfo);
    LinkIoWithCallOp(std::get<0>(graphInfo), std::get<1>(graphInfo));
}

void Function::UpdateOriIocastSlot(const std::shared_ptr<TensorSlotScope> scope)
{
    FE_ASSERT(FeError::INVALID_PTR, slotScope_ != nullptr) << "slotScope_ is null";
    auto& incastDst = slotScope_->oriIncastReadSlotSet;
    incastDst.insert(incastDst.end(), scope->incastReadSlotSet.begin(), scope->incastReadSlotSet.end());

    auto& outcastDst = slotScope_->oriOutcastWriteSlotSet;
    outcastDst.insert(outcastDst.end(), scope->outcastWriteSlotSet.begin(), scope->outcastWriteSlotSet.end());

    auto& iSlot = slotScope_->originalIocastsSlot.incastSlot;
    auto& oSlot = slotScope_->originalIocastsSlot.outcastSlot;

    iSlot.insert(iSlot.end(), scope->ioslot.incastSlot.begin(), scope->ioslot.incastSlot.end());
    oSlot.insert(oSlot.end(), scope->ioslot.outcastSlot.begin(), scope->ioslot.outcastSlot.end());
}

void Function::SetCallOpSlot()
{
    // op all OP_CALL
    bool isAllCallOp = std::all_of(operations_.begin(), operations_.end(),
                                   [](const auto& op) { return op->GetOpcode() == Opcode::OP_CALL; });
    if (!isAllCallOp) {
        return;
    }
    std::vector<Function*> calleeList = GetCalleeFunctionList();
    for (auto callee : calleeList) {
        if (callee == nullptr) {
            continue;
        }
        const std::shared_ptr<TensorSlotScope> calleeScope = callee->GetSlotScope();
        // callee incast -> call op iOperand, callee outcast -> call op oOperand
        UpdateOriIocastSlot(calleeScope);
    }
    return;
}

std::vector<int> Function::GetInCastSlot(const std::shared_ptr<LogicalTensor>& incast)
{
    std::vector<int> ret;
    for (size_t idx = 0; idx < inCasts_.size(); ++idx) {
        if (inCasts_[idx] == incast) {
            auto& scope = GetSlotScope();
            FE_ASSERT(FeError::INVALID_PTR, scope != nullptr) << "SlotScope is null";
            ret = scope->ioslot.incastSlot[idx];
        }
    }
    return ret;
}

std::vector<int> Function::GetOutCastSlot(const std::shared_ptr<LogicalTensor>& outcast)
{
    std::vector<int> ret;
    for (size_t idx = 0; idx < outCasts_.size(); ++idx) {
        if (outCasts_[idx] == outcast) {
            auto& scope = GetSlotScope();
            FE_ASSERT(FeError::INVALID_PTR, scope != nullptr) << "SlotScope is null";
            ret = scope->ioslot.outcastSlot[idx];
        }
    }
    return ret;
}

void Function::ResetOperations()
{
    operations_.clear();
    opPosition_.clear();
    sorted_ = false;
    tensorMap_.Reset();
    for (auto& t : inCasts_) {
        tensorMap_.Insert(t);
        t->GetConsumers().clear();
    }
    for (auto& t : outCasts_) {
        t->GetProducers().clear();
    }
}

std::set<Operation*, LogicalTensor::CompareOp> Function::FindConsumers(const Operation& op) const
{
    std::set<Operation*, LogicalTensor::CompareOp> consumers;
    for (const auto& output : op.oOperand) {
        for (auto& consumer : output->GetConsumers()) {
            if (consumer->BelongTo() == this) {
                consumers.emplace(consumer);
            }
        }
    }
    return consumers;
}

std::set<Operation*, LogicalTensor::CompareOp> Function::FindProducers(const Operation& op) const
{
    std::set<Operation*, LogicalTensor::CompareOp> producers;
    for (const auto& input : op.iOperand) {
        for (auto& producer : input->GetProducers()) {
            if (producer->BelongTo() == this) {
                producers.emplace(producer);
            }
        }
    }
    return producers;
}

void Function::OpValidCheck(Operation& op) const
{
    std::unordered_set<const Operation*> opMap;
    std::unordered_set<std::shared_ptr<LogicalTensor>> incasts(GetIncast().begin(), GetIncast().end());
    if (SPECIAL_OPCODE_SET.count(op.GetOpcode()) != 0) {
        if (IsViewLike(op.GetOpcode())) {
            FE_ASSERT(FeError::OUT_OF_RANGE, op.GetIOperands().size() == 1)
                << "View-like op expects 1 input operand, but got " << op.GetIOperands().size();
            FE_ASSERT(FeError::OUT_OF_RANGE, op.GetOOperands().size() <= 1)
                << "View-like op expects at most 1 output operand, but got " << op.GetOOperands().size();
            auto opAttr = std::dynamic_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
            FE_ASSERT(FeError::INVALID_PTR, opAttr != nullptr)
                << "View-like op should have a ViewOpAttribute, but it is null";
            ASSERT(FeError::INVALID_VAL, op.GetIOperands()[0]->GetOffset().size() == opAttr->GetFromOffset().size())
                << "View-like op input operand offset size does not match attribute from offset size";
            if (!op.GetOOperands().empty()) {
                ASSERT(FeError::INVALID_VAL, op.GetOOperands()[0]->GetOffset().size() == opAttr->GetFromOffset().size())
                    << "View-like op output operand offset size does not match attribute from offset size";
            }
        }
        if (IsAssembleLike(op.GetOpcode())) {
            FE_ASSERT(FeError::OUT_OF_RANGE, op.GetIOperands().size() == 1)
                << "Assemble-like op should have exactly 1 input operand, but has " << op.GetIOperands().size();
            FE_ASSERT(FeError::OUT_OF_RANGE, op.GetOOperands().size() <= 1)
                << "Assemble-like op should have at most 1 output operand, but has " << op.GetOOperands().size();
            auto opAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(op.GetOpAttribute());
            FE_ASSERT(FeError::INVALID_PTR, opAttr != nullptr)
                << "Assemble-like op should have an AssembleOpAttribute, but it is null";
            if (!op.GetIOperands().empty()) {
                ASSERT(FeError::INVALID_VAL, op.GetIOperands()[0]->GetOffset().size() == opAttr->GetToOffset().size())
                    << "Assemble-like op input operand offset size does not match attribute to offset size";
            }
            ASSERT(FeError::INVALID_VAL, op.GetOOperands()[0]->GetOffset().size() == opAttr->GetToOffset().size())
                << "Assemble-like op output operand offset size does not match attribute to offset size";
        }
    } else {
        FE_ASSERT(FeError::INVALID_PTR, op.GetOpAttribute() == nullptr)
            << "Non-special operation should not have an operation attribute";
    }

    FE_ASSERT(FeError::OUT_OF_RANGE, op.GetOpMagic() >= 0 && op.GetOpMagic() < opSeed_)
        << "Operation magic number is out of bounds: " << op.GetOpMagic() << ", function opSeed_ is: " << opSeed_;
    if (!op.IsCall()) { // call 允许多输出，其余操作目前不允许
        FE_ASSERT(FeError::OUT_OF_RANGE, op.GetOOperands().size() <= 1)
            << "Non-call operation should have at most 1 output operand, but has " << op.GetOOperands().size();
    }
    for (auto& oOperand : op.GetOOperands()) {
        FE_ASSERT(FeError::INVALID_VAL, oOperand->GetShape().size() == oOperand->GetOffset().size())
            << "Output operand shape size does not match offset size";
        FE_ASSERT(&oOperand->BelongFunction() == this) << "Output operand does not belong to current function";
        auto tmp = GraphUtils::GetTensorByMagic(const_cast<Function&>(*this), oOperand->magic);
        FE_ASSERT(FeError::NOT_EXIST, tmp == oOperand) << "Tensor map does not match output operand";
        FE_ASSERT(oOperand->HasProducer(op))
            << "Output operand does not have current operation as a producer, opmagic: " << op.GetOpMagic()
            << ", operand magic: " << oOperand->magic;
    }
    for (auto& iOperand : op.GetIOperands()) {
        FE_ASSERT(FeError::INVALID_VAL, iOperand->GetShape().size() == iOperand->GetOffset().size())
            << "Input operand shape size does not match offset size";
        FE_ASSERT(&iOperand->BelongFunction() == this) << "Input operand does not belong to current function";
        if (!iOperand->GetProducers().empty() || incasts.count(iOperand) != 0) {
            auto tmp = GraphUtils::GetTensorByMagic(const_cast<Function&>(*this), iOperand->magic);
            FE_ASSERT(FeError::NOT_EXIST, tmp == iOperand) << "Tensor map does not match input operand";
        }

        FE_ASSERT(iOperand->HasConsumer(op)) << "Input operand does not have current operation as a consumer";
        for (const auto& producer : iOperand->GetProducers()) {
            FE_ASSERT(producer->BelongTo() == this) << "Producer does not belong to current function";
            FE_ASSERT(FeError::OUT_OF_RANGE, producer->GetOpMagic() >= 0 && producer->GetOpMagic() < opSeed_)
                << "Producer magic number is out of bounds: " << producer->GetOpMagic()
                << ", function opSeed_ is: " << opSeed_ << ", producer in tensor(" << iOperand->magic << ","
                << iOperand->tensor->rawmagic << ")";
            if (producer->IsDeleted()) {
                continue;
            }
            FE_ASSERT(FeError::NOT_EXIST, opMap.find(producer) != opMap.end()) << "Producer not found in operation map";
        }
    }

    FE_ASSERT(FeError::IS_EXIST, opMap.count(&op) == 0) << "Operation is already in the operation map";
    opMap.emplace(&op);
}

DyndevFunctionAttribute::ValueDependDesc Function::LookupValueDepend()
{
    struct ValueDependSearcher {
        static void Search(DyndevFunctionAttribute::ValueDependDesc& desc, const SymbolicScalar& attr)
        {
            std::vector<RawSymbolicScalarPtr> callList = LookupExpressionByOpcode(attr.Raw(),
                                                                                  SymbolicOpcode::T_MOP_CALL);
            for (auto& call : callList) {
                auto caller = call->GetExpressionOperandList()[0];
                if (!caller->IsSymbol()) {
                    continue;
                }
                std::string name = caller->GetSymbolName();
                if (CallIsGetInputData(name)) {
                    desc.getInputDataCount++;
                } else if (CallIsGetTensorData(name)) {
                    desc.getTensorDataCount++;
                }
            }
        }
    };

    DyndevFunctionAttribute::ValueDependDesc desc;
    if (GetFunctionType() == FunctionType::DYNAMIC_LOOP) {
        auto loopAttr = GetDynloopAttribute();
        ValueDependSearcher::Search(desc, loopAttr->Begin());
        ValueDependSearcher::Search(desc, loopAttr->End());
        ValueDependSearcher::Search(desc, loopAttr->Step());
        for (auto& path : loopAttr->GetPathList()) {
            for (auto& cond : path.GetPathCondList()) {
                ValueDependSearcher::Search(desc, cond.GetCond());
            }
        }

    } else {
        for (auto& op : Operations(false)) {
            std::vector<std::reference_wrapper<SymbolicScalar>> attrList = op.GetDynamicAttributeList();
            for (auto& attr : attrList) {
                ValueDependSearcher::Search(desc, attr.get());
            }
        }
    }
    return desc;
}

void Function::ValidCheck() const
{
    int opMagic = -1000000;
    for (auto& op : const_cast<Function&>(*this).Operations()) {
        opMagic = std::max(opMagic, op.GetOpMagic());
    }
    FE_ASSERT(FeError::OUT_OF_RANGE, opMagic + 1 <= opSeed_)
        << "Invalid opMagic range: max opMagic is " << opMagic << ", function opSeed_ is: " << opSeed_;

    TensorMagicCheck();

    std::unordered_map<std::shared_ptr<LogicalTensor>, std::vector<Operation*>> used;
    for (auto& op : const_cast<Function&>(*this).Operations()) {
        if (op.IsDeleted()) {
            continue;
        }
        for (const auto& operand : op.GetOOperands()) {
            if (used.count(operand) > 0) {
                for (auto innerOp : used.at(operand)) {
                    FE_ASSERT(FeError::IS_EXIST, innerOp->ComputeHash() != op.ComputeHash())
                        << "Duplicate operation detected with same hash: " << op.ComputeHash();
                }
            }
            used[operand].emplace_back(&op);
        }
    }

    for (auto& op : const_cast<Function&>(*this).Operations()) {
        if (op.IsDeleted()) {
            continue;
        }
        OpValidCheck(op);
    }
}

std::shared_ptr<OpAttribute> Function::CreateCallOpAttribute(const std::vector<std::vector<SymbolicScalar>>& argList,
                                                             const std::map<int, SymbolicScalar>& outIndexToExpr)
{
    FunctionHash hash;
    if (rootFunc_ != nullptr) {
        /* has rootFunc, then current function is cutted */
        hash = rootFunc_->ComputeHash();
    } else {
        hash = ComputeHash();
    }
    auto opAttribute = std::make_shared<CallOpAttribute>(hash, argList, GetMagicName(), outIndexToExpr);
    return opAttribute;
}

using SameSlotSetIndex = std::map<std::vector<int>, std::vector<int>>;

template <typename T>
void RemoveDupIndices(std::vector<T>& data, std::vector<int> indexList)
{
    std::sort(indexList.begin(), indexList.end(), std::greater<int>());
    for (int index : indexList) {
        if (index >= 0 && index < static_cast<int>(data.size())) {
            data.erase(data.begin() + index);
        }
    }
}

SameSlotSetIndex ClassifyIocasts(const std::vector<std::vector<int>>& vec)
{
    SameSlotSetIndex classification;
    for (size_t idx = 0; idx < vec.size(); ++idx) {
        classification[vec[idx]].push_back(idx);
    }

    // if value vector size < 2，delete this pair
    for (auto it = classification.begin(); it != classification.end();) {
        if (it->second.size() == 1) {
            it = classification.erase(it);
        } else {
            ++it;
        }
    }
    return classification;
}

void Function::DoMergeFunctionDupIncast()
{
    auto sameSlotSetIndex = ClassifyIocasts(GetSlotScope()->ioslot.incastSlot);
    std::vector<int> removeIdx;
    for (auto& pair : sameSlotSetIndex) {
        auto& slotSetIndex = pair.second;
        FE_ASSERT(FeError::INVALID_VAL, slotSetIndex.size() > 1) << "Slot set index should have more than one element";
        removeIdx.insert(removeIdx.end(), slotSetIndex.begin() + 1, slotSetIndex.end());
        auto oriIncast = inCasts_[slotSetIndex[0]];
        auto newIncast = std::make_shared<LogicalTensor>(*this, oriIncast->tensor->datatype, oriIncast->shape,
                                                         oriIncast->tensor->GetDynRawShape(), oriIncast->Format(),
                                                         oriIncast->tensor->GetSymbol());
        newIncast->tensor->UpdateDynRawShape(oriIncast->tensor->GetDynRawShape());
        for (auto incastIdx : slotSetIndex) {
            FE_ASSERT(FeError::NOT_EXIST, inCasts_[incastIdx]->GetConsumers().size() > 0)
                << "Incast at index " << incastIdx << " should have at least one consumer";
            auto op = *inCasts_[incastIdx]->GetConsumers().begin();
            op->ReplaceIOperand(0, newIncast);
            tensorMap_.Insert(newIncast);
        }
        inCasts_[slotSetIndex[0]] = newIncast;
    }
    RemoveDupIndices(inCasts_, removeIdx);
    RemoveDupIndices(GetSlotScope()->incastReadSlotSet, removeIdx);
    RemoveDupIndices(GetSlotScope()->ioslot.incastSlot, removeIdx);
}

void Function::DoMergeFunctionDupOutcast()
{
    auto sameSlotSetIndex = ClassifyIocasts(GetSlotScope()->ioslot.outcastSlot);
    std::vector<int> removeIdx;
    for (auto& pair : sameSlotSetIndex) {
        auto& slotSetIndex = pair.second;
        FE_ASSERT(FeError::INVALID_VAL, slotSetIndex.size() > 1) << "Slot set index should have more than one element";
        removeIdx.insert(removeIdx.end(), slotSetIndex.begin() + 1, slotSetIndex.end());
        auto oriOutcast = outCasts_[slotSetIndex[0]];
        auto newOutcast = std::make_shared<LogicalTensor>(*this, oriOutcast->tensor->datatype, oriOutcast->shape,
                                                          oriOutcast->tensor->GetDynRawShape(), oriOutcast->Format(),
                                                          oriOutcast->tensor->GetSymbol());
        newOutcast->tensor->UpdateDynRawShape(oriOutcast->tensor->GetDynRawShape());
        for (auto incastIdx : slotSetIndex) {
            FE_ASSERT(FeError::NOT_EXIST, outCasts_[incastIdx]->GetProducers().size() > 0)
                << "Outcast at index " << incastIdx << " should have at least one producer";
            auto& op = *outCasts_[incastIdx]->GetProducers().begin();
            op->ReplaceOOperand(0, newOutcast);
            tensorMap_.Insert(newOutcast);
        }
        outCasts_[slotSetIndex[0]] = newOutcast;
    }
    RemoveDupIndices(outCasts_, removeIdx);
    RemoveDupIndices(GetSlotScope()->outcastWriteSlotSet, removeIdx);
    RemoveDupIndices(GetSlotScope()->ioslot.outcastSlot, removeIdx);
}

void Function::MergeFunctionDupIocast()
{
    DoMergeFunctionDupIncast();
    DoMergeFunctionDupOutcast();
}

bool Function::InsertLoopIdxNameList(const std::string& idxName)
{
    if (parent_ == nullptr) {
        loopIdxNameList_.insert(idxName);
        return true;
    }

    auto realParent = parent_->parent_;
    FE_ASSERT(FeError::INVALID_VAL, realParent != nullptr)
        << "Current function name: " << GetRawName() << " has no parent function, frontend kernel is not JIT decorated";
    if (realParent->GetFunctionType() == FunctionType::DYNAMIC_LOOP &&
        realParent->LoopIdxNameList().find(idxName) != realParent->LoopIdxNameList().end()) {
        return false;
    }

    loopIdxNameList_.insert(idxName);
    for (const auto& it : realParent->LoopIdxNameList()) {
        loopIdxNameList_.insert(it);
    }
    return true;
}

DefineProg::DefineProg(const std::string& name) : isRecording_(true) { Program::GetInstance().SetName(name); }
DefineProg::~DefineProg()
{
    if (isRecording_) {
        FE_LOGI("prog.end: name=%s", Program::GetInstance().Name().c_str());
    }
}
