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
 * \file function.h
 * \brief
 */
/* for flow verify tool */

#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <functional>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "interface/interpreter/interpreter_log.h"
#include "interface/interpreter/thread_pool.h"
#include "interface/tensor/tensor_slot.h"
#include "interface/interpreter/operation.h"
#include "interface/tensor/symbolic_scalar_evaluate.h"
#include "calc.h"
#include "tilefwk/error_code.h"
#include "communication.h"
#include "tilefwk/comm_group_recorder.h"
#include "interface/operation/distributed/distributed_common.h"
#include "ir/kind_traits.h"

namespace npu::tile_fwk {

enum class VerifyType { INVALID, TENSOR_GRAPH, PASS, EXECUTE_GRAPH };

struct PairHash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const
    {
        auto hash1 = std::hash<T1>{}(p.first);
        auto hash2 = std::hash<T2>{}(p.second);
        return hash1 ^ (hash2 << 1);
    }
};

struct FunctionIODataPair {
    std::vector<std::shared_ptr<LogicalTensorData>> incastDataViewList;
    std::vector<std::shared_ptr<LogicalTensorData>> outcastDataViewList;

    FunctionIODataPair() {}
    FunctionIODataPair(std::vector<std::shared_ptr<LogicalTensorData>> incastDataViewList_,
                       std::vector<std::shared_ptr<LogicalTensorData>> outcastDataViewList_)
        : incastDataViewList(incastDataViewList_), outcastDataViewList(outcastDataViewList_)
    {}

    std::vector<std::shared_ptr<LogicalTensorData>>& GetIncastDataViewList() { return incastDataViewList; }
    std::vector<std::shared_ptr<LogicalTensorData>>& GetOutcastDataViewList() { return outcastDataViewList; }

    // One tensor might occur in both incast and outcast simultaneously for multiple times, so we must copy
    // simultaneously
    static void CopyWithLinkRelationship(FunctionIODataPair& dst, const FunctionIODataPair& src)
    {
        std::unordered_map<std::shared_ptr<StorageData>, std::shared_ptr<StorageData>> storageDeepCopyDict;
        dst.incastDataViewList.resize(src.incastDataViewList.size());
        dst.outcastDataViewList.resize(src.outcastDataViewList.size());

        for (size_t k = 0; k < src.incastDataViewList.size(); k++) {
            auto& srcDataView = src.incastDataViewList[k];
            auto srcRaw = srcDataView->GetData();
            auto storage = srcRaw->GetRawData();
            if (storageDeepCopyDict.count(storage) == 0) {
                storageDeepCopyDict[storage] = std::make_shared<StorageData>(*storage);
            }
            auto dstRaw = std::make_shared<RawTensorData>(storageDeepCopyDict[storage], srcRaw->GetDataType(),
                                                          srcRaw->GetShape());
            if (srcRaw->IsShmTensor()) {
                dstRaw->SetShmOffset(srcRaw->GetShmOffset());
                dstRaw->SetAsShmTensor();
            }
            dst.incastDataViewList[k] = std::make_shared<LogicalTensorData>(
                dstRaw, srcDataView->GetShape(), srcDataView->GetValidShape(), srcDataView->GetOffset());
            dst.incastDataViewList[k]->SetIsSpilled(srcDataView->GetIsSpilled());
        }
        for (size_t k = 0; k < src.outcastDataViewList.size(); k++) {
            auto& srcDataView = src.outcastDataViewList[k];
            auto srcRaw = srcDataView->GetData();
            auto storage = srcRaw->GetRawData();
            if (storageDeepCopyDict.count(storage) == 0) {
                storageDeepCopyDict[storage] = std::make_shared<StorageData>(*storage);
            }
            auto dstRaw = std::make_shared<RawTensorData>(storageDeepCopyDict[storage], srcRaw->GetDataType(),
                                                          srcRaw->GetShape());
            if (srcRaw->IsShmTensor()) {
                dstRaw->SetShmOffset(srcRaw->GetShmOffset());
                dstRaw->SetAsShmTensor();
            }
            dst.outcastDataViewList[k] = std::make_shared<LogicalTensorData>(
                dstRaw, srcDataView->GetShape(), srcDataView->GetValidShape(), srcDataView->GetOffset());
            dst.outcastDataViewList[k]->SetIsSpilled(srcDataView->GetIsSpilled());
        }
        for (size_t k = 0; k < dst.incastDataViewList.size(); k++) {
            ASSERT(ControlFlowScene::FUNC_IO_DATAVIEW_NULL, dst.incastDataViewList[k] != nullptr);
        }
        for (size_t k = 0; k < dst.outcastDataViewList.size(); k++) {
            ASSERT(ControlFlowScene::FUNC_IO_DATAVIEW_NULL, dst.outcastDataViewList[k] != nullptr);
        }
    }
};

struct FunctionFrame {
    const Function* func;
    const Operation* callop;
    const std::shared_ptr<CallOpAttribute> callopAttr;
    std::shared_ptr<FunctionIODataPair> inoutDataPair;
    std::unordered_map<int, std::shared_ptr<StorageData>> rawTensorDataDict;
    std::unordered_map<std::shared_ptr<LogicalTensor>, std::shared_ptr<RawTensor>> spillRawTensorDict;
    std::unordered_map<std::shared_ptr<LogicalTensor>, std::shared_ptr<LogicalTensorData>> tensorDataViewDict;
    std::unordered_map<std::shared_ptr<LogicalTensor>, std::string> tensorDataBinDict;
    std::unordered_map<std::shared_ptr<LogicalTensorData>, std::shared_ptr<LogicalTensor>>
        callopDataViewTensorDict; // Record the relationship between the callop data view and the tensor
    int frameIndex;
    std::set<int> indexOutcastOpIndices;
    std::set<int> indexAddOpIndices;
    std::vector<std::set<LogicalTensorPtr>> inplaceTensorSetList;
    int funcIndex;
    size_t funcHash;
    std::string funcType;
    std::string funcGraphType;
    int rootFuncIndex{-1};
    size_t rootFuncHash;
    std::string rootFuncType;
    std::string rootFuncGraphType;
    int passIndex{-1};
    VerifyType verifyType{VerifyType::INVALID};

    Operation* currentOperation;

    /// Mix-split calls executed in a forward-collected same-wrapId parallel batch; skipped on linear re-walk.
    std::unordered_set<Operation*> executedParallelMixSplitCallOps;

    const std::unordered_map<std::shared_ptr<LogicalTensor>, std::shared_ptr<LogicalTensorData>>&
    GetTensorDataViewDict() const
    {
        return tensorDataViewDict;
    }

    int GetFrameIndex() const { return frameIndex; }

    void InitInplaceDataViewList()
    {
        inplaceTensorSetList.clear();
        if (func == nullptr || (indexOutcastOpIndices.empty() && indexAddOpIndices.empty())) {
            return;
        }

        auto ops = const_cast<Function*>(func)->Operations(false);
        // 建立 Operation 指针到其在 Operations(false) 中下标的映射
        std::unordered_map<Operation*, int> opIndexMap;
        opIndexMap.reserve(ops.size());
        for (int i = 0; i < static_cast<int>(ops.size()); ++i) {
            opIndexMap[&ops[static_cast<size_t>(i)]] = i;
        }

        ProcessInplaceOpIndices(ops, opIndexMap, indexOutcastOpIndices, 0x2);
        ProcessInplaceOpIndices(ops, opIndexMap, indexAddOpIndices, 0);
    }

    void ProcessInplaceOpIndices(OperationsViewer& ops, const std::unordered_map<Operation*, int>& opIndexMap,
                                 std::set<int>& opIndices, size_t startTensorIdx)
    {
        std::set<int> indices = opIndices;
        for (auto index : indices) {
            if (opIndices.count(index) == 0) {
                continue;
            }
            if (index < 0 || static_cast<size_t>(index) >= ops.size()) {
                continue;
            }

            Operation& op = ops[static_cast<size_t>(index)];
            auto iOps = op.GetIOperands();
            auto oOps = op.GetOOperands();
            ASSERT(ControlFlowScene::INVALID_INPLACE_CHAIN, iOps.size() > 0x2);
            ASSERT(ControlFlowScene::INVALID_INPLACE_CHAIN, !oOps.empty());

            LogicalTensorPtr startTensor = iOps[startTensorIdx];
            LogicalTensorPtr endTensor = oOps[0];
            ASSERT(ControlFlowScene::INVALID_INPLACE_CHAIN, startTensor != nullptr);
            ASSERT(ControlFlowScene::INVALID_INPLACE_CHAIN, endTensor != nullptr);

            std::set<LogicalTensorPtr> tensorGroup;
            std::unordered_set<LogicalTensorPtr> visitedTensor;
            std::unordered_set<Operation*> visitedOp;
            bool chainValid = true;

            TraverseBackward(startTensor, tensorGroup, visitedTensor, visitedOp, chainValid, opIndexMap);
            if (!chainValid) {
                continue;
            }
            TraverseForward(endTensor, tensorGroup, visitedTensor, visitedOp, chainValid, opIndexMap);
            if (!chainValid) {
                continue;
            }

            inplaceTensorSetList.emplace_back(std::move(tensorGroup));
        }
    }

    FunctionFrame(const Function* func_, const Operation* callop_, const std::shared_ptr<CallOpAttribute>& callopAttr_,
                  std::shared_ptr<FunctionIODataPair> inoutDataPair_, int frameIndex_)
        : func(func_), callop(callop_), callopAttr(callopAttr_), inoutDataPair(inoutDataPair_), frameIndex(frameIndex_)
    {
        if (func != nullptr) {
            int idx = 0;
            auto ops = const_cast<Function*>(func)->Operations(false);
            for (auto& op : ops) {
                if (op.GetOpcode() == Opcode::OP_INDEX_OUTCAST) {
                    indexOutcastOpIndices.insert(idx);
                }
                if (op.GetOpcode() == Opcode::OP_INDEX_ADD) {
                    indexAddOpIndices.insert(idx);
                }
                ++idx;
            }
        }
        InitInplaceDataViewList();
        if (inoutDataPair != nullptr) {
            ASSERT(ControlFlowScene::FUNC_INCAST_COUNT_MISMATCH,
                   func->GetIncast().size() == inoutDataPair->incastDataViewList.size());
            for (size_t i = 0; i < inoutDataPair->incastDataViewList.size(); i++) {
                AddDataView(func->GetIncast()[i], inoutDataPair->incastDataViewList[i]);
            }
            ASSERT(ControlFlowScene::FUNC_OUTCAST_COUNT_MISMATCH,
                   func->GetOutcast().size() == inoutDataPair->outcastDataViewList.size());
            for (size_t i = 0; i < inoutDataPair->outcastDataViewList.size(); i++) {
                AddDataView(func->GetOutcast()[i], inoutDataPair->outcastDataViewList[i]);
            }
            DoAddCallopInOutDataView();
        }
    }

    void UpdateCurrentOperation(Operation* op) { currentOperation = op; }

    std::shared_ptr<LogicalTensorData> GetDataView(const std::shared_ptr<LogicalTensor>& tensor)
    {
        if (!tensorDataViewDict.count(tensor)) {
            return nullptr;
        }
        auto view = tensorDataViewDict[tensor];
        return view;
    }

    std::vector<std::shared_ptr<LogicalTensorData>> GetDataViewList(
        const std::vector<std::shared_ptr<LogicalTensor>>& tensorList)
    {
        std::vector<std::shared_ptr<LogicalTensorData>> viewList(tensorList.size());
        for (size_t i = 0; i < tensorList.size(); i++) {
            viewList[i] = GetDataView(tensorList[i]);
        }
        return viewList;
    }

    void AddDataView(const std::shared_ptr<LogicalTensor>& tensor, const std::shared_ptr<LogicalTensorData>& dataView)
    {
        if (tensorDataViewDict.count(tensor)) {
            ASSERT(ControlFlowScene::FUNC_TENSOR_DATAVIEW_MISMATCH, tensorDataViewDict[tensor] == dataView);
        } else {
            DoAddTensorDataView(tensor, dataView);
            DoAddRawTensorDataView(tensor->GetRawTensor(), dataView->GetData()->GetRawData());
        }
    }
    void AddDataViewList(const std::vector<std::shared_ptr<LogicalTensor>>& tensorList,
                         const std::vector<std::shared_ptr<LogicalTensorData>>& dataViewList)
    {
        ASSERT(ControlFlowScene::FUNC_TENSOR_DATAVIEW_LIST_SIZE_MISMATCH, tensorList.size() == dataViewList.size());
        for (size_t i = 0; i < tensorList.size(); i++) {
            AddDataView(tensorList[i], dataViewList[i]);
        }
    }

    std::shared_ptr<LogicalTensorData> AllocateDataView(const std::shared_ptr<LogicalTensor>& tensor,
                                                        const std::vector<int64_t>& offset,
                                                        const std::vector<int64_t>& validShape,
                                                        const std::vector<int64_t>& rawShape, DataType dtype,
                                                        const std::shared_ptr<LogicalTensor>& inplaceTensor = nullptr)
    {
        if (tensorDataViewDict.count(tensor)) {
            const auto& existingView = tensorDataViewDict[tensor];
            if (validShape.empty() || existingView->GetValidShape() == validShape || callop != nullptr) {
                return existingView;
            }
            size_t requiredSize = RawTensorData::CalcRequiredSize(dtype, rawShape);
            ASSERT(ControlFlowScene::FUNC_RAW_TENSOR_SIZE_MISMATCH, existingView->GetData()->size() == requiredSize)
                << "RawTensor size mismatch when updating: existing size=" << existingView->GetData()->size()
                << ", required size=" << requiredSize << ", rawMagic=" << tensor->tensor->GetRawMagic()
                << ", tensorMagic=" << tensor->magic;
            auto rawData = std::make_shared<RawTensorData>(existingView->GetData()->GetRawData(), dtype, rawShape);
            if (existingView->GetData()->IsShmTensor()) {
                rawData->SetShmOffset(existingView->GetData()->GetShmOffset());
                rawData->SetAsShmTensor();
            }
            auto view = std::make_shared<LogicalTensorData>(rawData, tensor->GetShape(), validShape, offset);
            view->SetIsSpilled(existingView->GetIsSpilled());
            tensorDataViewDict[tensor] = view;
            return view;
        }

        auto raw = inplaceTensor ? inplaceTensor->GetRawTensor() : tensor->GetRawTensor();
        ASSERT(ControlFlowScene::FUNC_RAW_TENSOR_NULL, raw != nullptr) << "raw is nullptr.";
        bool isSpilled = false;

        const std::string spillRawMagic = "1056964608";
        std::string rawMagic = std::to_string(raw->GetRawMagic());
        if (rawMagic.find(spillRawMagic) != std::string::npos) {
            if (spillRawTensorDict.count(tensor)) {
                raw = spillRawTensorDict[tensor];
            } else {
                raw = std::make_shared<RawTensor>(dtype, rawShape);
                DoAddSpillRawTensor(tensor, raw);
            }
            isSpilled = true;
        }
        std::shared_ptr<RawTensorData> rawData;
        if (rawTensorDataDict.count(raw->GetRawMagic())) {
            auto existingRawData = rawTensorDataDict[raw->GetRawMagic()];
            // Validate size consistency when sharing RawTensor
            size_t requiredSize = RawTensorData::CalcRequiredSize(dtype, rawShape);
            ASSERT(ControlFlowScene::FUNC_RAW_TENSOR_SIZE_MISMATCH, existingRawData->size() >= requiredSize)
                << "RawTensor size mismatch when sharing: rawMagic=" << raw->GetRawMagic()
                << ", existing size=" << existingRawData->size() << ", required size=" << requiredSize
                << ", rawMagic=" << tensor->tensor->GetRawMagic() << ", tensorMagic=" << tensor->magic;
            rawData = std::make_shared<RawTensorData>(existingRawData, dtype, rawShape);
        } else {
            ASSERT(ControlFlowScene::FUNC_INPLACE_ALLOC_CONFLICT, inplaceTensor == nullptr);
            rawData = std::make_shared<RawTensorData>(dtype, rawShape);
            rawData->resize(rawData->GetDataSize());
            for (auto& [lt, ltd] : tensorDataViewDict) {
                if (lt->GetRawTensor() == tensor->GetRawTensor() && ltd->IsShmTensor()) {
                    rawData->SetShmOffset(ltd->GetData()->GetShmOffset());
                    rawData->SetAsShmTensor();
                }
            }
        }
        DoAddRawTensorDataView(tensor->GetRawTensor(), rawData->GetRawData());
        std::shared_ptr<LogicalTensorData> view = std::make_shared<LogicalTensorData>(rawData, tensor->GetShape(),
                                                                                      validShape, offset);
        view->SetIsSpilled(isSpilled);
        DoAddTensorDataView(tensor, view);
        return view;
    }

private:
    bool IsAllowedInplaceChainOpcode(Opcode opcode) const
    {
        return opcode == Opcode::OP_INDEX_OUTCAST || IsViewLike(opcode) || opcode == Opcode::OP_RESHAPE ||
               IsAssembleLike(opcode) || opcode == Opcode::OP_PRINT || opcode == Opcode::OP_INDEX_ADD;
    }

    void TryEraseOpIndex(Operation* op, const std::unordered_map<Operation*, int>& opIndexMap)
    {
        auto it = opIndexMap.find(op);
        if (it == opIndexMap.end()) {
            return;
        }
        auto opcode = op->GetOpcode();
        if (opcode == Opcode::OP_INDEX_ADD) {
            indexAddOpIndices.erase(it->second);
        } else if (opcode == Opcode::OP_INDEX_OUTCAST) {
            indexOutcastOpIndices.erase(it->second);
        }
    }

    void TraverseBackward(LogicalTensorPtr t, std::set<LogicalTensorPtr>& tensorGroup,
                          std::unordered_set<LogicalTensorPtr>& visitedTensor,
                          std::unordered_set<Operation*>& visitedOp, bool& chainValid,
                          const std::unordered_map<Operation*, int>& opIndexMap)
    {
        if (!chainValid || t == nullptr) {
            return;
        }
        if (visitedTensor.insert(t).second) {
            tensorGroup.insert(t);
        }
        for (auto producer : t->GetProducers()) {
            if (producer == nullptr) {
                continue;
            }
            if (!IsAllowedInplaceChainOpcode(producer->GetOpcode())) {
                chainValid = false;
                return;
            }
            if (visitedOp.insert(producer).second) {
                TryEraseOpIndex(producer, opIndexMap);
                auto& producerInputs = producer->GetIOperands();
                auto opcode = producer->GetOpcode();
                int idx = (opcode == Opcode::OP_INDEX_OUTCAST) ? 2 : 0;
                if (producerInputs.size() > static_cast<size_t>(idx) && producerInputs[idx] != nullptr) {
                    TraverseBackward(producerInputs[idx], tensorGroup, visitedTensor, visitedOp, chainValid,
                                     opIndexMap);
                }
            }
        }
    }

    void TraverseForward(LogicalTensorPtr t, std::set<LogicalTensorPtr>& tensorGroup,
                         std::unordered_set<LogicalTensorPtr>& visitedTensor, std::unordered_set<Operation*>& visitedOp,
                         bool& chainValid, const std::unordered_map<Operation*, int>& opIndexMap)
    {
        if (!chainValid || t == nullptr) {
            return;
        }
        if (visitedTensor.insert(t).second) {
            tensorGroup.insert(t);
        }
        for (auto consumerOp : t->GetConsumers()) {
            if (consumerOp == nullptr) {
                continue;
            }
            if (!IsAllowedInplaceChainOpcode(consumerOp->GetOpcode())) {
                chainValid = false;
                return;
            }
            if (visitedOp.insert(consumerOp).second) {
                TryEraseOpIndex(consumerOp, opIndexMap);
                auto& consumerOutputs = consumerOp->GetOOperands();
                if (!consumerOutputs.empty() && consumerOutputs[0] != nullptr) {
                    TraverseForward(consumerOutputs[0], tensorGroup, visitedTensor, visitedOp, chainValid, opIndexMap);
                }
            }
        }
    }

    void DoAddTensorDataView(const std::shared_ptr<LogicalTensor>& tensor,
                             const std::shared_ptr<LogicalTensorData>& dataView)
    {
        ASSERT(ControlFlowScene::FUNC_TENSOR_DATAVIEW_DUP, !tensorDataViewDict.count(tensor));
        tensorDataViewDict[tensor] = dataView;
    }
    void DoAddRawTensorDataView(const std::shared_ptr<RawTensor>& rawTensor, const std::shared_ptr<StorageData>& data)
    {
        rawTensorDataDict[rawTensor->GetRawMagic()] = data;
    }
    void DoAddSpillRawTensor(const std::shared_ptr<LogicalTensor>& tensor, const std::shared_ptr<RawTensor>& rawtensor)
    {
        ASSERT(ControlFlowScene::FUNC_SPILL_RAW_TENSOR_DUP, !spillRawTensorDict.count(tensor));
        spillRawTensorDict[tensor] = rawtensor;
    }
    void DoAddCallopInOutDataView()
    {
        if (callop == nullptr) {
            return;
        }
        for (size_t i = 0; i < inoutDataPair->incastDataViewList.size(); i++) {
            callopDataViewTensorDict[inoutDataPair->incastDataViewList[i]] = callop->GetIOperands()[i];
        }
        for (size_t i = 0; i < inoutDataPair->outcastDataViewList.size(); i++) {
            callopDataViewTensorDict[inoutDataPair->outcastDataViewList[i]] = callop->GetOOperands()[i];
        }
    }
};

struct FunctionCaptureExecution {
    Function* func;
    std::shared_ptr<FunctionIODataPair> baseline;
    std::unordered_map<std::string, ScalarImmediateType> symbolDict;
    std::unordered_map<std::string, ScalarImmediateType> loopSymbolDict;

    std::vector<std::shared_ptr<FunctionFrame>> frameList;

    std::shared_ptr<FunctionIODataPair> golden;

    FunctionCaptureExecution(Function* func_ = nullptr) : func(func_)
    {
        baseline = std::make_shared<FunctionIODataPair>();
        golden = std::make_shared<FunctionIODataPair>();
    }

    const std::vector<std::shared_ptr<FunctionFrame>>& GetFrameList() const { return frameList; }

    void CaptureFrom(const std::shared_ptr<FunctionIODataPair>& b,
                     const std::unordered_map<std::string, ScalarImmediateType>& s)
    {
        FunctionIODataPair::CopyWithLinkRelationship(*baseline, *b);
        symbolDict = s;
    }

    void CaptureSymbolDictFrom(const std::unordered_map<std::string, ScalarImmediateType>& s) { symbolDict = s; }

    void CaptureGoldenFrom(const std::shared_ptr<FunctionIODataPair>& g)
    {
        FunctionIODataPair::CopyWithLinkRelationship(*golden, *g);
    }

    std::unordered_map<std::string, ScalarImmediateType> CaptureTo(std::shared_ptr<FunctionIODataPair>& c) const
    {
        FunctionIODataPair::CopyWithLinkRelationship(*c, *baseline);
        return symbolDict;
    }
};

struct FunctionControlFlowExecution {
    std::unordered_map<Function*, std::vector<std::shared_ptr<FunctionCaptureExecution>>> executionListDict;
};

constexpr int EXEC_DUMP_LEVEL_OPERATION = 1;
constexpr int EXEC_DUMP_LEVEL_TENSOR = 2;
const std::unordered_set<Opcode> MIX_PATH_OPS = {Opcode::OP_UB_COPY_L1, Opcode::OP_L0C_COPY_UB, Opcode::OP_COPY_OUT,
                                                 Opcode::OP_RESHAPE};

enum class OpInfoCsvHeader {
    num = 0,
    passName,
    pathFuncMagicName,
    pathFuncMagic,
    pathFuncHash,
    loopInfo,
    rootFuncType,
    rootFuncGraphType,
    rootFuncID,
    rootFuncHash,
    funcType,
    funcGraphType,
    funcID,
    funcHash,
    rawTensorMagic,
    outputRawShape,
    outputDtype,
    outputFormat,
    outputSymbol,
    tensorMagic,
    tensorOffset,
    outputShape,
    outputValidShape,
    outputDynValidShape,
    callopMagic,
    callopRawMagic,
    opMagic,
    opCode,
    attrOffset,
    attrAtomic,
    ioflag,
    timeStamp,
    outputTensor,
    inputTensors,
    inputValidShape,
    inputRawMagic,
    COL_COUNT
};

enum class ProgrameInfoCsvHeader {
    num = 0,
    goldenPassName,
    passName,
    pathFuncMagicName,
    pathFuncMagic,
    pathFuncHash,
    loopInfo,
    ioflag,
    goldenRawMagic,
    outputRawMagic,
    outputRawShape,
    outputDtype,
    outputFormat,
    outputSymbol,
    outputShape,
    outputValidShape,
    aTimeStamp,
    goldenTensor,
    bTimeStamp,
    outputTensor,
    verifyResult,
    rtolAndAtol,
    failCnt,
    totalCnt,
    mre,
    mreTop8,
    mreTop1Permil,
    mae,
    maeTop8,
    maeTop1Permil,
    aMax,
    aMin,
    aAvg,
    aAavg,
    aZero,
    aInfnan,
    bMax,
    bMin,
    bAvg,
    bAavg,
    bZero,
    bInfnan,
    COL_COUNT
};

constexpr int32_t toIndex(OpInfoCsvHeader e) noexcept { return static_cast<int32_t>(e); }

constexpr int32_t toIndex(ProgrameInfoCsvHeader e) noexcept { return static_cast<int32_t>(e); }

struct FunctionInterpreter {
    FunctionInterpreter();

    ~FunctionInterpreter()
    {
        if (execOpResultFile != nullptr) {
            fclose(execOpResultFile);
        }
        if (execProgrameResultFile != nullptr) {
            fclose(execProgrameResultFile);
        }
        if (execDumpErrorFile != nullptr) {
            fclose(execDumpErrorFile);
        }
    }

    Function* entry_;
    std::unordered_map<FunctionHash, Function*> calleeHashDict;
    std::unordered_set<int> outputSlotSet_;
    std::shared_ptr<InterpreterSyncSimulationState> interpreterSyncSimulation_;
    util::ThreadPool interpreterThreadPool_;
    mutable std::mutex threadOpInterpMutex_;
    mutable std::unordered_map<std::thread::id, std::shared_ptr<OperationInterpreter>> perThreadOperationInterpreter_;
    std::vector<std::shared_ptr<LogicalTensorData>> interpreterEntryInputViews_;
    std::unordered_map<std::string, ScalarImmediateType> interpreterBootstrapSymbolDict_;
    std::unordered_map<int, std::shared_ptr<StorageData>> slotDataViewDict_;
    std::vector<std::shared_ptr<FunctionFrame>>* captureFrameList{nullptr};
    std::unordered_map<std::string, ScalarImmediateType> loopSymbolDict;
    std::unordered_map<std::pair<std::shared_ptr<LogicalTensor>, int32_t>, std::shared_ptr<LogicalTensorData>, PairHash>
        mixGlobalTensorDict;

    int execDumpLevel{0};

    std::string execDumpDir;
    std::string dumpPath;
    FILE* execDumpFile{nullptr};
    FILE* execOpResultFile{nullptr};
    FILE* execProgrameResultFile{nullptr};
    FILE* execDumpStyleFile{nullptr};
    FILE* execDumpErrorFile{nullptr};
    std::string execDumpFuncKey;
    std::string execDumpPassName;
    std::string execDumpFunPath;
    int pathFuncMagic;
    size_t pathFuncHash;
    std::vector<ElementDump> execDumpElementList;
    std::vector<std::shared_ptr<FunctionFrame>> execDumpStack;
    std::atomic<int> frameCount{0};
    int opInfoRowNum{0};
    int ProgrameRowNum{0};

    std::map<std::string, uint64_t> opUsage;
    uint64_t dumpTensorUsage{0};
    uint64_t dumpOperationUsage{0};
    uint64_t totalTimeUsage{0};

    VerifyType verifyType{VerifyType::INVALID};
    int captureIndex{0};
    int passIndex{-1};
    std::mutex captureFrameListMutex_;
    std::mutex mixGlobalTensorMutex_;
    std::condition_variable mixGlobalTensorCv_;
    static constexpr int64_t MIX_GLOBAL_TENSOR_WAIT_TIMEOUT_MS = 60000; // 60s timeout to detect dead waits
    bool mixMultiThreadEnabled_{false};
    std::mutex dumpStateMutex_;

    bool CheckWaitUntilReady(Operation* op, LogicalTensorDataPtr shmData)
    {
        Distributed::ShmemWaitUntilAttr attr;
        op->GetAttr(OpAttributeKey::distOpAttr, attr);
        std::shared_ptr<SimulationCommContext> context = SimulationCommManager::Instance().GetCommContext(attr.group);
        int srcRank = context->GetRank();
        size_t slotSize = shmData->GetSize() * BytesOf(shmData->GetDataType());
        uint64_t offset = shmData->GetShmStorageOffset();
        return context->CheckWaitCondition(srcRank, attr.expectedSum, slotSize, offset);
    }

    std::unordered_map<Operation*, std::vector<Operation*>> ConstructOpConsumers(
        OperationsViewer operations, std::unordered_map<Operation*, int>& inDegree)
    {
        std::unordered_map<Operation*, std::vector<Operation*>> consumers;
        std::unordered_set<Operation*> opSet;
        for (auto& op : operations) {
            opSet.insert(&op);
        }
        for (auto& op : operations) {
            for (auto& iTensor : op.GetIOperands()) {
                for (auto* producer : iTensor->GetProducers()) {
                    if (opSet.count(producer)) {
                        inDegree[&op]++;
                        consumers[producer].push_back(&op);
                    }
                }
            }
        }
        for (auto& op : operations) {
            for (auto& dTensor : op.GetDependOperands()) {
                for (auto* producer : dTensor->GetProducers()) {
                    if (opSet.count(producer)) {
                        inDegree[&op]++;
                        consumers[producer].push_back(&op);
                    }
                }
            }
        }
        return consumers;
    }

    OperationInterpreter& GetOperationInterpreterForThisThread() const
    {
        const auto tid = std::this_thread::get_id();
        std::lock_guard<std::mutex> lk(threadOpInterpMutex_);
        auto it = perThreadOperationInterpreter_.find(tid);
        if (it != perThreadOperationInterpreter_.end()) {
            return *it->second;
        }
        auto op = std::make_shared<OperationInterpreter>(interpreterSyncSimulation_);
        op->evaluateSymbol->InitInputDataViewList(interpreterEntryInputViews_);
        if (!interpreterBootstrapSymbolDict_.empty()) {
            op->evaluateSymbol->SetSymbolDict(interpreterBootstrapSymbolDict_);
        }
        auto ins = perThreadOperationInterpreter_.emplace(tid, std::move(op));
        return *ins.first->second;
    }

    std::vector<std::shared_ptr<LogicalTensorData>> GetInputDataViewList()
    {
        return GetOperationInterpreterForThisThread().evaluateSymbol->GetInputDataViewList();
    }
    void UpdateInputDataViewList(size_t index, const std::shared_ptr<LogicalTensorData>& inputDataView)
    {
        interpreterEntryInputViews_[index] = inputDataView;
        std::lock_guard<std::mutex> lk(threadOpInterpMutex_);
        for (auto& entry : perThreadOperationInterpreter_) {
            entry.second->evaluateSymbol->UpdateInputDataViewList(index, inputDataView);
        }
    }
    void InitInputDataViewList(const std::vector<std::shared_ptr<LogicalTensorData>>& inputDataViewList)
    {
        interpreterEntryInputViews_ = inputDataViewList;
        interpreterBootstrapSymbolDict_.clear();
        std::lock_guard<std::mutex> lk(threadOpInterpMutex_);
        perThreadOperationInterpreter_.clear();
        auto op = std::make_shared<OperationInterpreter>(interpreterSyncSimulation_);
        op->evaluateSymbol->InitInputDataViewList(inputDataViewList);
        perThreadOperationInterpreter_[std::this_thread::get_id()] = op;
    }
    void UpdateIODataPair(std::shared_ptr<FunctionIODataPair>& inoutDataPair)
    {
        GetOperationInterpreterForThisThread().evaluateSymbol->UpdateIODataPair(inoutDataPair);
    }
    std::unordered_map<std::string, ScalarImmediateType> GetSymbolDict() const
    {
        return GetOperationInterpreterForThisThread().evaluateSymbol->GetSymbolDict();
    }
    void UpdateSymbolDict(const std::string key, const ScalarImmediateType value)
    {
        GetOperationInterpreterForThisThread().evaluateSymbol->UpdateSymbolDict(key, value);
    }
    void SetSymbolDict(const std::unordered_map<std::string, ScalarImmediateType>& symbolDict)
    {
        interpreterBootstrapSymbolDict_ = symbolDict;
        std::lock_guard<std::mutex> lk(threadOpInterpMutex_);
        for (auto& entry : perThreadOperationInterpreter_) {
            entry.second->evaluateSymbol->SetSymbolDict(symbolDict);
        }
    }

    ScalarImmediateType EvaluateSymbolicScalar(const SymbolicScalar& ss)
    {
        return GetOperationInterpreterForThisThread().EvaluateSymbolicScalar(ss);
    }

    static SymbolicScalar ExprToSymbolicScalar(const ir::ExprPtr& expr)
    {
        if (auto imm = std::dynamic_pointer_cast<const RawSymbolicImmediate>(expr)) {
            return SymbolicScalar(std::const_pointer_cast<RawSymbolicImmediate>(imm));
        }
        if (auto sym = std::dynamic_pointer_cast<const RawSymbolicSymbol>(expr)) {
            return SymbolicScalar(std::const_pointer_cast<RawSymbolicSymbol>(sym));
        }
        if (auto rawExpr = std::dynamic_pointer_cast<const RawSymbolicExpression>(expr)) {
            return SymbolicScalar(std::const_pointer_cast<RawSymbolicExpression>(rawExpr));
        }
        if (auto constInt = std::dynamic_pointer_cast<const ir::ConstInt>(expr)) {
            return SymbolicScalar(constInt->value_);
        }
        ASSERT(false) << "ExprToSymbolicScalar: unsupported Expr type";
        return SymbolicScalar();
    }
    std::vector<int64_t> EvaluateOffset(const std::vector<int64_t>& offset,
                                        const std::vector<SymbolicScalar>& dynOffset,
                                        const std::vector<SymbolicScalar>& linearArgList = {})
    {
        return GetOperationInterpreterForThisThread().EvaluateOffset(offset, dynOffset, linearArgList);
    }
    std::vector<int64_t> EvaluateValidShape(const std::vector<SymbolicScalar>& dynValidShape,
                                            const std::vector<SymbolicScalar>& linearArgList = {})
    {
        return GetOperationInterpreterForThisThread().EvaluateValidShape(dynValidShape, linearArgList);
    }
    void EvaluateDynParam(const std::map<std::string, DynParamInfo>& dynParamTable,
                          const std::vector<SymbolicScalar>& linearArgList)
    {
        GetOperationInterpreterForThisThread().evaluateSymbol->EvaluateDynParam(dynParamTable, linearArgList);
    }

    size_t GetFrameSize() const { return execDumpStack.size(); }
    std::string GetFrameIndex(const std::shared_ptr<FunctionFrame>& frame) const
    {
        if (frame == nullptr) {
            return "null";
        } else {
            return std::to_string(frame->GetFrameIndex());
        }
    }
    std::shared_ptr<FunctionFrame> GetFrameCurr() const
    {
        if (execDumpStack.size() == 0) {
            return nullptr;
        } else {
            return execDumpStack.back();
        }
    }
    std::string GetFrameCurrIndex() const { return GetFrameIndex(GetFrameCurr()); }

    LogicalTensorDataPtr FormatNZ2ND(LogicalTensorDataPtr& view)
    {
        auto out = LogicalTensorData::CreateEmpty(view->GetDataType(), view->GetShape(), view->GetValidShape(),
                                                  view->GetData()->GetShape());
        calc::FormatNZ2ND(out, view);
        return out;
    }

    LogicalTensorDataPtr FormatND2NZ(LogicalTensorDataPtr& view)
    {
        auto out = LogicalTensorData::CreateEmpty(view->GetDataType(), view->GetShape(), view->GetValidShape(),
                                                  view->GetData()->GetShape());
        calc::FormatND2NZ(out, view);
        return out;
    }

    void Initialize(Function* entry, const std::vector<std::shared_ptr<LogicalTensorData>>& inputDataViewList)
    {
        entry_ = entry;
        InitInputDataViewList(inputDataViewList);
    }

    Function* GetEntry() const { return entry_; }

    Function* GetCallee(const Operation* callop)
    {
        auto calleeHash = callop->GetCalleeHash();
        ASSERT(ControlFlowScene::INVALID_CALLEE_MAPPING, calleeHashDict.count(calleeHash));
        Function* callee = calleeHashDict.find(calleeHash)->second;
        return callee;
    }

    void UpdateHashDict(const std::unordered_map<FunctionHash, Function*>& hashDict)
    {
        for (auto& [hash, callee] : hashDict) {
            if (calleeHashDict.count(hash)) {
                ASSERT(ControlFlowScene::INVALID_CALLEE_MAPPING, calleeHashDict.find(hash)->second == callee);
            } else {
                calleeHashDict[hash] = callee;
            }
        }
    }

    std::shared_ptr<LogicalTensorData> AllocateDataView(FunctionFrame& frame,
                                                        const std::shared_ptr<LogicalTensor>& tensor, DataType dtype,
                                                        const std::shared_ptr<LogicalTensor>& inplaceTensor = nullptr)
    {
        std::vector<SymbolicScalar> linearArgList;
        if (frame.callopAttr != nullptr) {
            linearArgList = frame.callopAttr->GetLinearArgList();
        }
        std::vector<int64_t> offset = EvaluateOffset(tensor->GetOffset(), tensor->GetDynOffset(), linearArgList);
        auto validShape = EvaluateValidShape(tensor->GetDynValidShape(), linearArgList);
        auto rawShape = EvaluateValidShape(tensor->GetRawTensor()->GetDynRawShape());
        ASSERT(ControlFlowScene::INVALID_TENSOR_SHAPE,
               std::all_of(rawShape.begin(), rawShape.end(), [](int64_t dim) { return dim >= 0; }))
            << "AllocateDataView: shape dimension must be non-negative" << rawShape
            << ", rawMagic=" << tensor->tensor->GetRawMagic() << ", tensorMagic=" << tensor->magic;
        auto ret = frame.AllocateDataView(tensor, offset, validShape, rawShape, dtype, inplaceTensor);
        return ret;
    }

    std::shared_ptr<LogicalTensorData> AllocateDataView(FunctionFrame& frame,
                                                        const std::shared_ptr<LogicalTensor>& tensor,
                                                        const std::shared_ptr<LogicalTensor>& inplaceTensor = nullptr)
    {
        return AllocateDataView(frame, tensor, tensor->GetRawTensor()->GetDataType(), inplaceTensor);
    }

    void ExecuteOpCallLeaf(ExecuteOperationContext* ctx)
    {
        Function* callee = GetCallee(ctx->op);
        auto inoutDataPair = std::make_shared<FunctionIODataPair>(*ctx->ioperandDataViewList,
                                                                  *ctx->ooperandInplaceDataViewList);
        ExecuteFunctionFrame(callee, ctx->op, inoutDataPair);
    }

    int32_t GetCallOpWrapId(const Operation* op) const
    {
        if (op == nullptr || op->GetOpcode() != Opcode::OP_CALL) {
            return -1;
        }
        auto callopAttr = std::dynamic_pointer_cast<CallOpAttribute>(op->GetOpAttribute());
        if (callopAttr == nullptr) {
            return -1;
        }
        return callopAttr->wrapId;
    }

    int32_t GetCallOpMixId(const Operation* op)
    {
        if (op == nullptr || op->GetOpcode() != Opcode::OP_CALL) {
            return LeafFuncAttribute::INVALID_MIX_ID;
        }
        Function* callee = GetCallee(op);
        if (callee == nullptr || callee->GetLeafFuncAttribute() == nullptr) {
            return LeafFuncAttribute::INVALID_MIX_ID;
        }
        return callee->GetLeafFuncAttribute()->mixId;
    }

    bool IsMixSplitCallOp(const Operation* op)
    {
        return GetCallOpWrapId(op) != -1 && GetCallOpMixId(op) != LeafFuncAttribute::INVALID_MIX_ID;
    }

    struct MixSplitCallTask {
        FunctionInterpreter* interpreter{nullptr};
        Function* callee{nullptr};
        Operation* callop{nullptr};
        std::shared_ptr<FunctionIODataPair> inoutDataPair{nullptr};
        static void Entry(void* ctx)
        {
            auto* task = static_cast<MixSplitCallTask*>(ctx);
            if (task == nullptr || task->interpreter == nullptr || task->callee == nullptr || task->callop == nullptr ||
                task->inoutDataPair == nullptr) {
                return;
            }
            task->interpreter->ExecuteFunctionFrame(task->callee, task->callop, task->inoutDataPair);
        }
    };

    std::shared_ptr<FunctionIODataPair> BuildCallInOutDataPair(FunctionFrame& frame, Operation* callop)
    {
        auto iOpDataList = frame.GetDataViewList(callop->GetIOperands());
        for (size_t index = 0; index < iOpDataList.size(); index++) {
            if (iOpDataList[index] == nullptr) {
                auto iop = callop->GetIOperands()[index];
                if (mixMultiThreadEnabled_) {
                    VERIFY_LOGI("BuildCallInOutDataPair: iop %zu is null, try to find in mixGlobalTensorDict.", index);
                    iOpDataList[index] = WaitAndGetMixGlobalTensorDataView(frame, iop, callop);
                    ASSERT(ControlFlowScene::FUNC_IO_DATAVIEW_NULL, iOpDataList[index] != nullptr)
                        << "BuildCallInOutDataPair: input data view not found in mixGlobalTensorDict, callop magic="
                        << callop->GetOpMagic() << ", operandIdx=" << index << ", tensorMagic=" << iop->GetMagic();
                    continue;
                }
                ASSERT(ControlFlowScene::FUNC_IO_DATAVIEW_NULL, false)
                    << "BuildCallInOutDataPair: input data view not found for callop magic=" << callop->GetOpMagic()
                    << ", operandIdx=" << index << ", tensorMagic=" << iop->GetMagic()
                    << " (mix multi-thread is disabled)";
            }
        }

        std::vector<std::shared_ptr<LogicalTensorData>> oOpDataList;
        for (size_t i = 0; i < callop->GetOOperands().size(); i++) {
            auto oop = callop->GetOOperands()[i];
            if (auto index = callop->GetInplaceIndex(i); index != -1) {
                ExecuteInplaceOperation(frame, *callop, i, iOpDataList, oOpDataList);
            } else {
                oOpDataList.push_back(AllocateDataView(frame, oop));
            }
        }
        return std::make_shared<FunctionIODataPair>(iOpDataList, oOpDataList);
    }

    void ExecuteMixSplitCallOpGroupParallel(FunctionFrame& frame, const std::vector<Operation*>& groupedCallOps)
    {
        constexpr size_t kMixSplitParallelLimit = 3;
        ASSERT(ControlFlowScene::MIX_SPLIT_PARALLEL_LIMIT_EXCEEDED, groupedCallOps.size() <= kMixSplitParallelLimit)
            << "MixSplit grouped callops exceeds parallel limit, groupedSize=" << groupedCallOps.size()
            << ", limit=" << kMixSplitParallelLimit;
        struct MixMultiThreadGuard {
            FunctionInterpreter* interpreter;
            explicit MixMultiThreadGuard(FunctionInterpreter* self) : interpreter(self)
            {
                interpreter->mixMultiThreadEnabled_ = true;
            }
            ~MixMultiThreadGuard() { interpreter->mixMultiThreadEnabled_ = false; }
        } mixMultiThreadGuard(this);
        std::vector<std::shared_ptr<MixSplitCallTask>> taskList;
        taskList.reserve(groupedCallOps.size());
        for (auto* groupedCallOp : groupedCallOps) {
            auto task = std::make_shared<MixSplitCallTask>();
            task->interpreter = this;
            task->callee = GetCallee(groupedCallOp);
            task->callop = groupedCallOp;
            task->inoutDataPair = BuildCallInOutDataPair(frame, groupedCallOp);
            taskList.push_back(task);
        }

        auto& pool = interpreterThreadPool_;
        for (size_t i = 0; i < taskList.size(); i++) {
            pool.SubmitTask(taskList[i].get(), MixSplitCallTask::Entry);
        }
        pool.NotifyAll();
        pool.WaitForAll();
    }

    std::vector<uint64_t> UnBind(SymbolicScalar attr)
    {
        std::shared_ptr<RawSymbolicExpression> expr = std::static_pointer_cast<RawSymbolicExpression>(attr.Raw());
        ASSERT(expr->Opcode() == SymbolicOpcode::T_MOP_CALL);
        std::vector<uint64_t> parameters;
        for (size_t i = 1; i < expr->OperandList().size(); i++) {
            ScalarImmediateType value = EvaluateSymbolicScalar(SymbolicScalar(expr->OperandList()[i]));
            parameters.emplace_back(value);
        }
        return parameters;
    }

    void ExecuteBindTensor(FunctionFrame& frame, Operation& op,
                           const std::vector<std::shared_ptr<LogicalTensorData>>& iOpDataList,
                           std::vector<std::shared_ptr<LogicalTensorData>>& oOpDataList)
    {
        (void)iOpDataList;
        (void)oOpDataList;
        ASSERT(op.GetIOperands().size() == 0);
        ASSERT(op.GetOOperands().size() == 1);
        SymbolicScalar attr = op.GetSymbolicScalarAttribute(OpAttributeKey::bindTensor);
        std::vector<uint64_t> parameters = UnBind(attr);
        uint64_t groupIndex = parameters[0];
        uint64_t memType = parameters[1];
        const auto& groupNames = Distributed::CommGroupRecorder::GetInstance().Output();
        ASSERT(groupIndex < static_cast<uint64_t>(groupNames.size()));
        const std::string& groupName = groupNames[groupIndex];
        LogicalTensorDataPtr out;
        RawTensorDataPtr tmp;
        auto outOp = op.GetOOperands()[0];
        if (frame.GetDataView(outOp) != nullptr) {
            out = frame.GetDataView(outOp);
            oOpDataList.emplace_back(out);
            return;
        }
        if (memType == 0) {
            tmp = SimulationCommManager::Instance().Alloc(groupName, outOp->Datatype(), outOp->GetShape());
            out = LogicalTensorData::Create(*tmp);
        }
        if (memType == 1) {
            tmp = SimulationCommManager::Instance().AllocSignal(groupName, outOp->Datatype(), outOp->GetShape());
            out = LogicalTensorData::Create(*tmp);
        }
        ASSERT(ExecuteOperationScene::RUNTIME_EXCEPTION, out != nullptr);
        frame.AddDataView(outOp, out);
        oOpDataList.emplace_back(out);
    }

    void ExecuteInplaceOperation(FunctionFrame& frame, Operation& op, int oOperandIdx,
                                 const std::vector<std::shared_ptr<LogicalTensorData>>& iOpDataList,
                                 std::vector<std::shared_ptr<LogicalTensorData>>& oOpDataList)
    {
        auto oop = op.GetOOperands()[oOperandIdx];
        auto index = op.GetInplaceIndex(oOperandIdx);
        ASSERT(ControlFlowScene::INVALID_INPLACE_CHAIN, index != -1);
        auto iop = op.GetInputOperand(index);
        ASSERT(ControlFlowScene::INVALID_INPLACE_CHAIN, iOpDataList[index] != nullptr);
        if (IsViewLike(op.GetOpcode())) {
            if (iOpDataList[0]->IsShmTensor()) {
                auto ret = AllocateDataView(frame, oop);
                ret->GetData()->SetAsShmTensor();
                ret->GetData()->SetShmOffset(iOpDataList[index]->GetData()->GetShmOffset());
                oOpDataList.emplace_back(ret);
                return;
            }
            auto ret = AllocateDataView(frame, oop);
            oOpDataList.emplace_back(ret);
        } else {
            oOpDataList.emplace_back(AllocateDataView(frame, oop, iop));
        }
    }

    bool isConsumerAccMatmul(Operation* op)
    {
        for (auto cons : op->ConsumerOps()) {
            if (cons->GetOpcode() == Opcode::OP_A_MULACC_B || cons->GetOpcode() == Opcode::OP_A_MULACC_BT) {
                return true;
            }
        }
        return false;
    }

    bool isOutCast(const std::vector<std::shared_ptr<LogicalTensor>>& outCasts,
                   const std::shared_ptr<LogicalTensor>& oop)
    {
        auto it = std::find(outCasts.begin(), outCasts.end(), oop);
        if (it == outCasts.end()) {
            return false;
        } else {
            return true;
        }
    }

    /// Mix-split leaf output: reuse mixGlobalTensorDict entry for (oop, wrapId) if present; else allocate under
    /// mixGlobalTensorMutex_ and publish. Caller must only use this after input views are ready (AllocateDataView
    /// must not recurse into WaitAndGetMixGlobalTensorDataView while the mutex is held).
    std::shared_ptr<LogicalTensorData> AllocateOrReuseMixGlobalOutputDataView(FunctionFrame& frame,
                                                                              const std::shared_ptr<LogicalTensor>& oop,
                                                                              int32_t wrapId)
    {
        const std::pair<std::shared_ptr<LogicalTensor>, int32_t> mixKey{oop, wrapId};
        bool reusedFromMixDict = false;
        std::shared_ptr<LogicalTensorData> ret;
        {
            std::lock_guard<std::mutex> mixTensorGuard(mixGlobalTensorMutex_);
            auto mixIt = mixGlobalTensorDict.find(mixKey);
            if (mixIt != mixGlobalTensorDict.end() && mixIt->second != nullptr) {
                ret = mixIt->second;
                reusedFromMixDict = true;
            } else {
                ret = AllocateDataView(frame, oop);
                mixGlobalTensorDict[mixKey] = ret;
                mixGlobalTensorCv_.notify_all();
            }
        }
        if (reusedFromMixDict) {
            frame.AddDataView(oop, ret);
        }
        return ret;
    }

    void ExecuteOperation(FunctionFrame& frame, Operation* op)
    {
        auto iOpDataList = frame.GetDataViewList(op->GetIOperands());
        for (size_t index = 0; index < iOpDataList.size(); index++) {
            if (iOpDataList[index] == nullptr) {
                auto iop = op->GetIOperands()[index];
                if (op->GetOpcode() == Opcode::OP_SHMEM_PUT || op->GetOpcode() == Opcode::OP_SHMEM_GET ||
                    op->GetOpcode() == Opcode::OP_NOP) {
                    iOpDataList[index] = AllocateDataView(frame, iop);
                    continue;
                }
                if (frame.callop != nullptr && IsMixSplitCallOp(frame.callop)) {
                    INTERPRETER_LOGI("ExecuteOperation: iop %zu is null, try to find in mixGlobalTensorDict.", index);
                    iOpDataList[index] = WaitAndGetMixGlobalTensorDataView(frame, iop);
                    if (iOpDataList[index] != nullptr) {
                        continue;
                    }
                }
                ASSERT(ControlFlowScene::INVALID_INPLACE_CHAIN, op->GetOpcode() == Opcode::OP_CALL);
                iOpDataList[index] = AllocateDataView(frame, iop);
            }
        }
        std::vector<std::shared_ptr<LogicalTensorData>> oOpDataList;
        for (size_t i = 0; i < op->GetOOperands().size(); i++) {
            auto oop = op->GetOOperands()[i];
            if (op->GetOpcode() == Opcode::OP_INDEX_ADD && i > 0) {
                continue;
            }
            // Mix-split cross-component outputs must be published to mixGlobalTensorDict before the inplace
            // handling: pipe/copy ops (UB_COPY_L1 / L0C_COPY_UB / COPY_OUT) are never inplace, while view-like
            // inplace ops (e.g. RESHAPE with op_attr_isInplace=true) would otherwise be intercepted below and
            // their outputs would never be registered for sibling mix-split leaves.
            if (frame.callop != nullptr && IsMixSplitCallOp(frame.callop) && MIX_PATH_OPS.count(op->GetOpcode()) > 0) {
                auto callopAttr = std::static_pointer_cast<CallOpAttribute>(frame.callop->GetOpAttribute());
                oOpDataList.push_back(AllocateOrReuseMixGlobalOutputDataView(frame, oop, callopAttr->wrapId));
            } else if (auto index = op->GetInplaceIndex(i); index != -1) {
                ExecuteInplaceOperation(frame, *op, i, iOpDataList, oOpDataList);
            } else if (op->GetOpcode() == Opcode::OP_BIND_TENSOR) {
                ExecuteBindTensor(frame, *op, iOpDataList, oOpDataList);
            } else {
                if (isConsumerAccMatmul(op)) {
                    auto dtype = oop->GetRawTensor()->GetDataType();
                    // mm output dtype promotion
                    if ((dtype == DataType::DT_FP16 || dtype == DataType::DT_BF16)) {
                        dtype = DataType::DT_FP32;
                    }
                    oOpDataList.push_back(AllocateDataView(frame, oop, dtype));
                } else {
                    oOpDataList.push_back(AllocateDataView(frame, oop));
                }
            }
        }
        ExecuteOperationContext ctx = {&frame, {}, op, &iOpDataList, {}, &oOpDataList};

        if (op->GetOpcode() == Opcode::OP_CALL) {
            ExecuteOpCallLeaf(&ctx);
        } else {
            TimeStamp ts;
            GetOperationInterpreterForThisThread().ExecuteOperation(&ctx);
            {
                std::lock_guard<std::mutex> dumpGuard(dumpStateMutex_);
                opUsage[op->GetOpcodeStr()] += ts.Duration();
                auto* ooperandDumpList = ctx.ooperandInplaceDataViewList ? ctx.ooperandInplaceDataViewList :
                                                                           ctx.ooperandDataViewList;
                DumpOperationTensor(ctx.op, ctx.frame, ooperandDumpList, ctx.ioperandDataViewList);
                dumpOperationUsage += ts.Duration();
            }
        }
    }

    std::shared_ptr<LogicalTensorData> WaitAndGetMixGlobalTensorDataView(FunctionFrame& frame,
                                                                         const std::shared_ptr<LogicalTensor>& iop,
                                                                         Operation* mixCallop = nullptr)
    {
        Operation* callop = mixCallop != nullptr ? mixCallop : const_cast<Operation*>(frame.callop);
        ASSERT(ControlFlowScene::INVALID_INPLACE_CHAIN, callop != nullptr);
        auto callopAttr = std::static_pointer_cast<CallOpAttribute>(callop->GetOpAttribute());
        const std::pair<std::shared_ptr<LogicalTensor>, int32_t> mixKey{iop, callopAttr->wrapId};
        std::unique_lock<std::mutex> mixTensorGuard(mixGlobalTensorMutex_);
        const auto waitDeadline = std::chrono::steady_clock::now() +
                                  std::chrono::milliseconds(MIX_GLOBAL_TENSOR_WAIT_TIMEOUT_MS);
        const bool waitOk = mixGlobalTensorCv_.wait_until(mixTensorGuard, waitDeadline, [this, mixKey] {
            auto it = mixGlobalTensorDict.find(mixKey);
            return it != mixGlobalTensorDict.end() && it->second != nullptr;
        });
        ASSERT(ControlFlowScene::MIX_GLOBAL_TENSOR_WAIT_TIMEOUT, waitOk)
            << "Timeout while waiting mixGlobalTensorDict in multithread execution, wrapId=" << callopAttr->wrapId
            << ", timeoutMs=" << MIX_GLOBAL_TENSOR_WAIT_TIMEOUT_MS;
        std::shared_ptr<LogicalTensorData> result = nullptr;
        if (waitOk) {
            auto it = mixGlobalTensorDict.find(mixKey);
            if (it != mixGlobalTensorDict.end() && it->second != nullptr) {
                result = it->second;
                frame.AddDataView(iop, result);
            }
        }
        return result;
    }

    void ExecuteHandleFunctionBegin(Function* func, std::shared_ptr<FunctionFrame> frame)
    {
        std::lock_guard<std::mutex> dumpGuard(dumpStateMutex_);
        TimeStamp ts;
        execDumpStack.push_back(frame);
        DumpFunctionHead(func);
        if (frame->inoutDataPair != nullptr) {
            for (size_t k = 0; k < func->GetIncast().size(); k++) {
                auto rawMagic = func->GetIncast()[k]->GetRawTensor()->GetRawMagic();
                std::string fileName = "tensor_Incast_" + std::to_string(rawMagic) + ".data";
                DumpTensorBinary(frame->inoutDataPair->incastDataViewList[k], fileName);
                frame->tensorDataBinDict[func->GetIncast()[k]] = fileName;
            }
        }
        dumpTensorUsage += ts.Duration();
    }
    void ExecuteHandleFunctionEnd()
    {
        std::lock_guard<std::mutex> dumpGuard(dumpStateMutex_);
        execDumpStack.pop_back();
    }
    void ExecuteHandleOperationBegin(Operation* op)
    {
        std::lock_guard<std::mutex> dumpGuard(dumpStateMutex_);
        execDumpStack.back()->UpdateCurrentOperation(op);
        TimeStamp ts;
        DumpOperation(op);
        dumpOperationUsage += ts.Duration();
    }
    void ExecuteHandleOperationEnd() {}

    bool TryExecuteMixSplitCallOps(FunctionFrame& frame, const OperationsViewer& operations, size_t& opIdx,
                                   Operation& op)
    {
        if (!IsMixSplitCallOp(&op)) {
            return false;
        }
        const int32_t wrapId = GetCallOpWrapId(&op);
        std::vector<Operation*> groupedCallOps;
        for (size_t i = opIdx; i < operations.size(); ++i) {
            auto& cand = operations.at(i);
            if (IsMixSplitCallOp(&cand) && GetCallOpWrapId(&cand) == wrapId) {
                groupedCallOps.push_back(&cand);
            }
        }
        if (groupedCallOps.size() > 1) {
            constexpr size_t kMixSplitParallelLimit = 3;
            for (size_t batchStart = 0; batchStart < groupedCallOps.size(); batchStart += kMixSplitParallelLimit) {
                const size_t batchEnd = std::min(batchStart + kMixSplitParallelLimit, groupedCallOps.size());
                std::vector<Operation*> batch;
                batch.reserve(batchEnd - batchStart);
                for (size_t j = batchStart; j < batchEnd; ++j) {
                    batch.push_back(groupedCallOps[j]);
                }
                ExecuteMixSplitCallOpGroupParallel(frame, batch);
            }
            for (auto* executedOp : groupedCallOps) {
                frame.executedParallelMixSplitCallOps.insert(executedOp);
            }
        } else {
            ExecuteHandleOperationBegin(&op);
            ExecuteOperation(frame, &op);
            ExecuteHandleOperationEnd();
        }
        opIdx = opIdx + 1;
        return true;
    }

    void ExecuteHasWaitUntilFrame(Function* func, std::shared_ptr<FunctionFrame> frame)
    {
        auto operations = func->Operations();
        std::queue<Operation*> queue;
        std::unordered_map<Operation*, int> inDegree;
        for (auto& operation : operations) {
            queue.push(&operation);
            inDegree[&operation] = 0;
        }
        std::unordered_map<Operation*, std::vector<Operation*>> consumers = ConstructOpConsumers(operations, inDegree);
        while (!queue.empty()) {
            auto op = queue.front();
            queue.pop();
            if (inDegree[op] != 0) {
                queue.push(op);
                continue;
            }
            if (op->GetOpcode() == Opcode::OP_SHMEM_WAIT_UNTIL) {
                auto iopList = frame->GetDataViewList(op->GetIOperands());
                LogicalTensorDataPtr shmData = iopList[1];
                if (CheckWaitUntilReady(op, shmData)) {
                    ExecuteHandleOperationBegin(op);
                    ExecuteOperation(*frame, op);
                    ExecuteHandleOperationEnd();
                    for (auto* consumer : consumers[op]) {
                        inDegree[consumer]--;
                    }
                } else {
                    constexpr int64_t retrySleepMicroseconds = 10;
                    std::this_thread::sleep_for(std::chrono::microseconds(retrySleepMicroseconds));
                    queue.push(op);
                }
            } else {
                ExecuteHandleOperationBegin(op);
                ExecuteOperation(*frame, op);
                ExecuteHandleOperationEnd();
                for (auto* consumer : consumers[op]) {
                    inDegree[consumer]--;
                }
            }
        }
    }

    std::shared_ptr<FunctionFrame> ExecuteFunctionFrame(Function* func, Operation* callop,
                                                        std::shared_ptr<FunctionIODataPair>& inoutDataPair)
    {
        std::shared_ptr<CallOpAttribute> callopAttr;
        std::vector<SymbolicScalar> linearArgList;
        if (callop != nullptr) {
            callopAttr = std::static_pointer_cast<CallOpAttribute>(callop->GetOpAttribute());
            linearArgList = callopAttr->GetLinearArgList();
        }
        std::shared_ptr<FunctionFrame> frame = std::make_shared<FunctionFrame>(func, callop, callopAttr, inoutDataPair,
                                                                               frameCount.fetch_add(1));
        if (callop == nullptr) {
            interpreterSyncSimulation_->Reset();
        }
        if (captureFrameList != nullptr) {
            std::lock_guard<std::mutex> captureGuard(captureFrameListMutex_);
            captureFrameList->push_back(frame);
        }
        frame->funcIndex = func->GetFuncMagic();
        frame->funcHash = func->GetFunctionHash().GetHash();
        frame->funcType = func->GetFunctionTypeStr();
        frame->funcGraphType = GetGraphTypeNameDict().Find(func->GetGraphType());
        frame->passIndex = passIndex;
        frame->verifyType = verifyType;
        if (func->HasParent()) {
            frame->rootFuncIndex = func->Parent().GetFuncMagic();
            frame->rootFuncHash = func->Parent().GetFunctionHash().GetHash();
            frame->rootFuncType = func->Parent().GetFunctionTypeStr();
            frame->rootFuncGraphType = GetGraphTypeNameDict().Find(func->Parent().GetGraphType());
        }

        UpdateIODataPair(inoutDataPair);
        auto dynParamTable = func->GetDynParamTable();
        EvaluateDynParam(dynParamTable, linearArgList);

        ExecuteHandleFunctionBegin(func, frame);
        if (verifyType == VerifyType::TENSOR_GRAPH && func->GetCallopList().empty()) {
            func->SortOperations();
        }
        auto operations = func->Operations();

        bool hasWaitUntil = false;
        for (auto& op : operations) {
            if (op.GetOpcode() == Opcode::OP_SHMEM_WAIT_UNTIL) {
                hasWaitUntil = true;
                break;
            }
        }
        if (hasWaitUntil) {
            ExecuteHasWaitUntilFrame(func, frame);
        } else {
            for (size_t opIdx = 0; opIdx < operations.size();) {
                auto& op = operations.at(opIdx);
                if (op.GetOpcode() == Opcode::OP_PRINT && verifyType != VerifyType::TENSOR_GRAPH) {
                    opIdx++;
                    continue;
                }

                if (frame->executedParallelMixSplitCallOps.count(&op) != 0U) {
                    opIdx++;
                    continue;
                }

                if (TryExecuteMixSplitCallOps(*frame, operations, opIdx, op)) {
                    continue;
                }

                ExecuteHandleOperationBegin(&op);
                ExecuteOperation(*frame, &op);
                ExecuteHandleOperationEnd();
                opIdx++;
            }
        }

        ExecuteHandleFunctionEnd();

        CopyInplaceOutcastToIncast(func, frame);

        EraseTensorDataView(func, *frame);
        return frame;
    }

    void CopyInplaceOutcastToIncast(Function* func, const std::shared_ptr<FunctionFrame>& frame)
    {
        // After function execution, if inplace chains exist, copy data from outcast back to incast.
        if (frame == nullptr || frame->inplaceTensorSetList.empty()) {
            return;
        }

        auto& incastList = func->GetIncast();
        auto& outcastList = func->GetOutcast();

        for (const auto& tensorGroup : frame->inplaceTensorSetList) {
            if (tensorGroup.size() < 0x2) {
                continue;
            }

            // Find incast tensor in this group.
            LogicalTensorPtr incastTensor = nullptr;
            bool hasIncastFromFunc = false;
            for (const auto& t : tensorGroup) {
                if (std::find(incastList.begin(), incastList.end(), t) != incastList.end()) {
                    incastTensor = t;
                    hasIncastFromFunc = true;
                    break;
                }
            }

            // Find outcast tensor in this group.
            LogicalTensorPtr outcastTensor = nullptr;
            bool hasOutcastFromFunc = false;
            for (const auto& t : tensorGroup) {
                if (std::find(outcastList.begin(), outcastList.end(), t) != outcastList.end()) {
                    outcastTensor = t;
                    hasOutcastFromFunc = true;
                    break;
                }
            }

            // If this group has neither incast nor outcast belonging to current function IO,
            // it indicates that inplaceTensorSetList is inconsistent with function IO spec.
            if (!hasIncastFromFunc || !hasOutcastFromFunc) {
                continue;
            }

            auto incastView = frame->GetDataView(incastTensor);
            auto outcastView = frame->GetDataView(outcastTensor);
            if (incastView == nullptr || outcastView == nullptr) {
                continue;
            }

            auto incData = incastView->GetData();
            auto outData = outcastView->GetData();
            if (incData == nullptr || outData == nullptr || incData.get() == outData.get()) {
                continue;
            }

            ASSERT(ExecuteOperationScene::INVALID_TENSOR_DTYPE, incData->GetDataType() == outData->GetDataType());
            ASSERT(ExecuteOperationScene::INVALID_TENSOR_SIZE, incData->GetDataSize() == outData->GetDataSize());
            ASSERT(ExecuteOperationScene::INVALID_TENSOR_SIZE, incData->size() == outData->size());

            std::copy(outData->data(), outData->data() + outData->size(), incData->data());
        }
    }

    void EraseTensorDataView(Function* func, FunctionFrame& frame)
    {
        for (auto it = frame.tensorDataViewDict.begin(); it != frame.tensorDataViewDict.end();) {
            auto incast = std::find(func->GetIncast().begin(), func->GetIncast().end(), it->first);
            if (incast != func->GetIncast().end()) {
                it++;
                continue;
            }
            auto outcast = std::find(func->GetOutcast().begin(), func->GetOutcast().end(), it->first);
            if (outcast != func->GetOutcast().end()) {
                it++;
                continue;
            }
            it = frame.tensorDataViewDict.erase(it);
        }
    }

    std::vector<std::shared_ptr<FunctionFrame>> ExecuteFunctionCapture(
        Function* func, Operation* callop, std::shared_ptr<FunctionIODataPair>& inoutDataPair)
    {
        std::vector<std::shared_ptr<FunctionFrame>> frameList;
        captureFrameList = &frameList;
        ExecuteFunctionFrame(func, callop, inoutDataPair);
        return frameList;
    }

    void ExecuteFunctionDynamic(Function* func, FunctionControlFlowExecution& controlFlowExecution)
    {
        std::shared_ptr<FunctionFrame> frame = std::make_shared<FunctionFrame>(func, nullptr, nullptr, nullptr,
                                                                               frameCount++);
        ExecuteHandleFunctionBegin(func, frame);

        std::vector<Operation*> callopList = func->GetCallopList();
        for (auto callop : callopList) {
            Function* callee = GetCallee(callop);

            ExecuteHandleOperationBegin(callop);
            ExecuteControlFlow(callee, controlFlowExecution);
            ExecuteHandleOperationEnd();
        }

        ExecuteHandleFunctionEnd();
    }

    Operation* ExecuteFunctionLoopLookupSat(const std::shared_ptr<DynloopFunctionAttribute>& controlFlowExecution)
    {
        for (auto& path : controlFlowExecution->pathList) {
            bool sat = true;
            for (auto cond : path.pathCondList) {
                if (static_cast<bool>(EvaluateSymbolicScalar(cond.GetCond())) != cond.IsSat()) {
                    sat = false;
                    break;
                }
            }
            if (!sat) {
                continue;
            }
            return path.callop;
        }
        return nullptr;
    }

    void ExecuteFunctionLoop(Function* func, FunctionControlFlowExecution& controlFlowExecution)
    {
        std::shared_ptr<FunctionFrame> frame = std::make_shared<FunctionFrame>(func, nullptr, nullptr, nullptr,
                                                                               frameCount++);
        ExecuteHandleFunctionBegin(func, frame);

        auto loop = func->GetDynloopAttribute();
        ScalarImmediateType begin = EvaluateSymbolicScalar(loop->Begin());
        ScalarImmediateType end = EvaluateSymbolicScalar(loop->End());
        ScalarImmediateType step = EvaluateSymbolicScalar(loop->Step());
        if (begin == end) {
            INTERPRETER_EVENT("Function %s skip execute due to idx range = 0", func->GetMagicName().c_str());
        }
        for (ScalarImmediateType idx = begin; idx < end; idx += step) {
            UpdateSymbolDict(loop->IterSymbolName(), idx);
            loopSymbolDict[loop->IterSymbolName()] = idx;
            Operation* callop = ExecuteFunctionLoopLookupSat(loop);
            if (callop == nullptr) {
                continue;
            }
            Function* callee = GetCallee(callop);

            ExecuteHandleOperationBegin(callop);
            ExecuteControlFlow(callee, controlFlowExecution);
            ExecuteHandleOperationEnd();
        }

        ExecuteHandleFunctionEnd();
    }

    void ExecuteStmtBody(Function* func, const ir::StmtPtr& stmt, FunctionControlFlowExecution& controlFlowExecution)
    {
        if (!stmt) {
            return;
        }
        if (auto seq = ir::As<ir::SeqStmts>(stmt)) {
            for (auto& child : seq->stmts_) {
                ExecuteStmtBody(func, child, controlFlowExecution);
            }
        } else if (auto forStmt = ir::As<ir::ForStmt>(stmt)) {
            ScalarImmediateType begin = EvaluateSymbolicScalar(ExprToSymbolicScalar(forStmt->start_));
            ScalarImmediateType end = EvaluateSymbolicScalar(ExprToSymbolicScalar(forStmt->stop_));
            ScalarImmediateType step = EvaluateSymbolicScalar(ExprToSymbolicScalar(forStmt->step_));
            for (ScalarImmediateType idx = begin; idx < end; idx += step) {
                UpdateSymbolDict(forStmt->loopVar_->name_, idx);
                loopSymbolDict[forStmt->loopVar_->name_] = idx;
                ExecuteStmtBody(func, forStmt->body_, controlFlowExecution);
            }
        } else if (auto ifStmt = ir::As<ir::IfStmt>(stmt)) {
            auto condVal = EvaluateSymbolicScalar(ExprToSymbolicScalar(ifStmt->condition_));
            if (static_cast<bool>(condVal)) {
                ExecuteStmtBody(func, ifStmt->thenBody_, controlFlowExecution);
            } else if (ifStmt->elseBody_) {
                ExecuteStmtBody(func, ifStmt->elseBody_.value(), controlFlowExecution);
            }
        } else if (auto tensorOp = ir::AsMut<ir::TensorOpStmt>(stmt)) {
            auto op = std::dynamic_pointer_cast<Operation>(tensorOp);
            ASSERT(op != nullptr) << "ExecuteStmtBody: TensorOpStmt is not an Operation";
            Function* callee = GetCallee(op.get());
            ExecuteHandleOperationBegin(op.get());
            ExecuteControlFlow(callee, controlFlowExecution);
            ExecuteHandleOperationEnd();
        } else {
            INTERPRETER_LOGW("ExecuteStmtBody: unhandled stmt kind: %d", static_cast<int>(stmt->GetKind()));
        }
    }

    void ExecuteControlFlow(Function* func, FunctionControlFlowExecution& controlFlowExecution)
    {
        if (func->body_ && func->GetFunctionType() == FunctionType::DYNAMIC) {
            std::shared_ptr<FunctionFrame> frame = std::make_shared<FunctionFrame>(func, nullptr, nullptr, nullptr,
                                                                                   frameCount++);
            ExecuteHandleFunctionBegin(func, frame);
            ExecuteStmtBody(func, func->body_, controlFlowExecution);
            ExecuteHandleFunctionEnd();
            return;
        }

        if (func->GetFunctionType() != FunctionType::DYNAMIC &&
            func->GetFunctionType() != FunctionType::DYNAMIC_LOOP_PATH) {
            func->SortOperations();
        }
        auto funcType = func->GetFunctionType();
        if (funcType == FunctionType::DYNAMIC) {
            ExecuteFunctionDynamic(func, controlFlowExecution);
        } else if (funcType == FunctionType::DYNAMIC_LOOP) {
            ExecuteFunctionLoop(func, controlFlowExecution);
        } else if (func->IsGraphType(GraphType::TENSOR_GRAPH)) {
            std::vector<Operation*> callopList = func->GetCallopList();
            if (callopList.size() != 0) {
                ExecuteFunctionDynamic(func, controlFlowExecution);
            } else {
                execDumpFunPath = func->GetMagicName();
                pathFuncMagic = func->GetFuncMagic();
                pathFuncHash = func->GetFunctionHash().GetHash();
                auto& incastSlot = func->GetSlotScope()->ioslot.incastSlot;
                auto& outcastSlot = func->GetSlotScope()->ioslot.outcastSlot;
                auto& partialSlot = func->GetSlotScope()->ioslot.partialUpdateOutcastList;

                auto getOutputSlot = [this](const std::vector<int>& slotList) {
                    for (auto& slot : slotList) {
                        if (this->outputSlotSet_.count(slot)) {
                            return slot;
                        }
                    }
                    return -1;
                };

                auto inoutDataPair = std::make_shared<FunctionIODataPair>();

                ASSERT(ControlFlowScene::FUNC_SLOT_IO_COUNT_MISMATCH, func->GetIncast().size() == incastSlot.size());
                for (size_t i = 0; i < func->GetIncast().size(); i++) {
                    int slot = incastSlot[i][0];
                    ASSERT(ControlFlowScene::FUNC_SLOT_MISSING, slotDataViewDict_.count(slot))
                        << "missing slot=" << slot;
                    auto tensor = func->GetIncast()[i];
                    auto offset = EvaluateOffset(tensor->GetOffset(), tensor->GetDynOffset());
                    auto validShape = EvaluateValidShape(tensor->GetDynValidShape());
                    auto rawShape = EvaluateValidShape(tensor->GetRawTensor()->GetDynRawShape());
                    auto rawData = std::make_shared<RawTensorData>(slotDataViewDict_[slot], tensor->Datatype(),
                                                                   rawShape);
                    auto incastDataView = std::make_shared<LogicalTensorData>(rawData, tensor->GetShape(), validShape,
                                                                              offset);
                    inoutDataPair->incastDataViewList.push_back(incastDataView);
                }

                ASSERT(ControlFlowScene::FUNC_SLOT_IO_COUNT_MISMATCH, func->GetOutcast().size() == outcastSlot.size());
                for (size_t i = 0; i < func->GetOutcast().size(); i++) {
                    int outputSlot = getOutputSlot(outcastSlot[i]);
                    bool isPartialSlot = std::find(partialSlot.begin(), partialSlot.end(), i) != partialSlot.end();
                    std::shared_ptr<LogicalTensorData> outcastView;
                    auto outcast = func->GetOutcast()[i];
                    auto offset = EvaluateOffset(outcast->GetOffset(), outcast->GetDynOffset());
                    auto validShape = EvaluateValidShape(outcast->GetDynValidShape());
                    auto rawShape = EvaluateValidShape(outcast->GetRawTensor()->GetDynRawShape());
                    if (outputSlot != -1) {
                        auto expectedSize = RawTensorData::CalcRequiredSize(outcast->Datatype(), rawShape);
                        ASSERT(ControlFlowScene::FUNC_RAW_TENSOR_SIZE_MISMATCH,
                               slotDataViewDict_[outputSlot]->size() == expectedSize)
                            << "Slot buffer size mismatch: funcMagicName=" << func->GetMagicName()
                            << ", outcastIndex=" << i << ", slot=" << outputSlot
                            << ", slotBufferSize=" << slotDataViewDict_[outputSlot]->size()
                            << ", expectedSize=" << expectedSize
                            << ", dtype=" << DataType2String(outcast->Datatype(), true) << ", rawShape=" << rawShape
                            << ", rawMagic=" << outcast->GetRawTensor()->GetRawMagic();
                        auto rawData = std::make_shared<RawTensorData>(slotDataViewDict_[outputSlot],
                                                                       outcast->Datatype(), rawShape);
                        outcastView = std::make_shared<LogicalTensorData>(rawData, outcast->GetShape(), validShape,
                                                                          offset);
                    } else if (isPartialSlot && slotDataViewDict_[outcastSlot[i][0]]) {
                        auto expectedSize = RawTensorData::CalcRequiredSize(outcast->Datatype(), rawShape);
                        ASSERT(ControlFlowScene::FUNC_RAW_TENSOR_SIZE_MISMATCH,
                               slotDataViewDict_[outcastSlot[i][0]]->size() == expectedSize)
                            << "Slot buffer size mismatch: funcMagicName=" << func->GetMagicName()
                            << ", outcastIndex=" << i << ", slot=" << outcastSlot[i][0]
                            << ", slotBufferSize=" << slotDataViewDict_[outcastSlot[i][0]]->size()
                            << ", expectedSize=" << expectedSize
                            << ", dtype=" << DataType2String(outcast->Datatype(), true) << ", rawShape=" << rawShape
                            << ", rawMagic=" << outcast->GetRawTensor()->GetRawMagic();
                        auto rawData = std::make_shared<RawTensorData>(slotDataViewDict_[outcastSlot[i][0]],
                                                                       outcast->Datatype(), rawShape);
                        outcastView = std::make_shared<LogicalTensorData>(rawData, outcast->GetShape(), validShape,
                                                                          offset);
                    } else {
                        outcastView = LogicalTensorData::CreateEmpty(outcast->Datatype(), outcast->GetShape(),
                                                                     validShape, rawShape);
                    }
                    for (auto& s : outcastSlot[i]) {
                        slotDataViewDict_[s] = outcastView->GetData()->GetRawData();
                    }
                    inoutDataPair->outcastDataViewList.push_back(outcastView);
                }

                auto capture = std::make_shared<FunctionCaptureExecution>(func);

                capture->CaptureFrom(inoutDataPair, GetSymbolDict());
                capture->loopSymbolDict = loopSymbolDict;
                capture->frameList = ExecuteFunctionCapture(func, nullptr, inoutDataPair);
                capture->CaptureGoldenFrom(inoutDataPair);
                controlFlowExecution.executionListDict[func].emplace_back(capture);
            }
        } else {
            ASSERT(ControlFlowScene::FUNC_UNKNOWN_IO_TYPE, false);
        }
    }

    std::string DumpSymbolDict() const
    {
        std::ostringstream oss;
        for (auto& [name, value] : GetSymbolDict()) {
            oss << name << " = " << value << "\n";
        }
        return oss.str();
    }
    void DumpOperation(Operation* op);
    void DumpOperationTensor(Operation* op, FunctionFrame* frame,
                             const std::vector<std::shared_ptr<LogicalTensorData>>* ooperandDataViewList,
                             const std::vector<std::shared_ptr<LogicalTensorData>>* ioperandDataViewList);

private:
    void FillOperationBasicInfo(Operation* op, FunctionFrame* frame, std::vector<std::string>& opInfo);
    void FillOperationOffsetInfo(Operation* op, FunctionFrame* frame, const std::vector<SymbolicScalar>& linearArgList,
                                 std::vector<std::string>& opInfo);
    void FillOperationInputInfo(Operation* op, FunctionFrame* frame,
                                const std::vector<std::shared_ptr<LogicalTensorData>>* ioperandDataViewList,
                                std::vector<std::string>& opInfo);
    void FillOperationOutputInfo(Operation* op, FunctionFrame* frame,
                                 const std::vector<std::shared_ptr<LogicalTensorData>>* ooperandDataViewList,
                                 const std::vector<SymbolicScalar>& linearArgList, int indent,
                                 std::vector<std::string>& opInfo);

public:
    void DumpTensorBinary(const std::shared_ptr<LogicalTensor>& tensor,
                          const std::shared_ptr<LogicalTensorData>& dataView);
    void DumpTensorBinary(const std::shared_ptr<LogicalTensorData>& dataView, std::string dumpTensorFileName,
                          bool isRaw = false);
    void DumpBinary(std::vector<int64_t>& shape, std::vector<int64_t>& stride, std::vector<int64_t>& offset,
                    FILE* fdata, uint8_t* data, size_t dtypeSize);
    std::shared_ptr<LogicalTensorData> LoadTensorBinary(const std::shared_ptr<LogicalTensor>& tensor,
                                                        const std::string dumpTensorFileName);
    void DumpFunctionHead(Function* func);
    void DumpBegin();
    void DumpEnd();
    void DumpPassTensorDiff(const std::shared_ptr<FunctionCaptureExecution>& captureExecution,
                            const std::shared_ptr<FunctionCaptureExecution>& captureGolden);
    std::string GetDumpFilePath(const std::string& lv0, const std::string& lv1, const std::string& filename);

    std::string GetDumpFrameDirName() const
    {
        std::string dirName = "frame_" + GetFrameCurrIndex();
        return dirName;
    }
    std::string GetDumpOperationTensorFileName(Operation* op) const
    {
        std::string fileName = "frame_" + GetFrameCurrIndex() + "_operation_" + std::to_string(op->GetOpMagic()) +
                               ".html";
        return fileName;
    }
    std::string GetDumpTensorFileName(const std::shared_ptr<LogicalTensor>& tensor) const
    {
        std::string fileName = "frame_" + GetFrameCurrIndex() + "_tensor_" + std::to_string(tensor->GetMagic()) +
                               ".data";
        return fileName;
    }
    std::string GetDumpTensorFileName(const std::shared_ptr<LogicalTensor>& tensor, Operation* op,
                                      FunctionFrame* frame) const
    {
        auto callopMagic = (frame->callop != nullptr) ? std::to_string(frame->callop->GetOpMagic()) + "~" : "~";
        struct timeval tv;
        gettimeofday(&tv, nullptr);
        auto ts = tv.tv_sec * 1000000 + tv.tv_usec; // 1000000 is us per sec

        std::string fileName = std::to_string(frame->rootFuncIndex) + "~" + callopMagic + GetLoopSymbolString(false) +
                               "~" + std::to_string(frame->funcIndex) + "~" + std::to_string(op->GetOpMagic()) + "~" +
                               op->GetOpcodeStr() + "~" + std::to_string(tensor->GetRawTensor()->GetRawMagic()) + "~" +
                               std::to_string(tensor->GetMagic()) + "~" + std::to_string(ts) + ".data";
        return fileName;
    }
    std::string GetLoopSymbolString(bool withName = true) const
    {
        std::ostringstream loop;
        size_t loopCount = loopSymbolDict.size();
        size_t count = 0;
        for (auto& [name, value] : loopSymbolDict) {
            if (withName) {
                loop << name << "=" << value;
            } else {
                loop << value;
            }
            if (++count < loopCount) {
                loop << "@";
            }
        }
        return loop.str();
    }
    std::string GetDumpTensorId(const std::shared_ptr<FunctionFrame>& frame,
                                const std::shared_ptr<LogicalTensor>& tensor) const
    {
        std::string index = GetFrameIndex(frame);
        std::string magic = tensor != nullptr ? std::to_string(tensor->GetMagic()) : "null";
        std::string tensorId = "tensor_" + index + "_" + magic;
        return tensorId;
    }
    std::string GetDumpTensorId(const std::shared_ptr<FunctionFrame>& frame, Operation* op) const
    {
        std::shared_ptr<LogicalTensor> tensor = op->GetOOperands().size() != 0 ? op->GetOOperands()[0] : nullptr;
        return GetDumpTensorId(frame, tensor);
    }
    std::string GetDumpOperationId(const std::shared_ptr<FunctionFrame>& frame, Operation* op) const
    {
        std::string index = GetFrameIndex(frame);
        std::string magic = op != nullptr ? std::to_string(op->GetOpMagic()) : "null";
        std::string tensorId = "operation_" + index + "_" + magic;
        return tensorId;
    }

    void DumpSetLevelOperation() { execDumpLevel = EXEC_DUMP_LEVEL_OPERATION; }
    void DumpSetLevelTensor() { execDumpLevel = EXEC_DUMP_LEVEL_TENSOR; }

    void DumpReset()
    {
        std::lock_guard<std::mutex> dumpGuard(dumpStateMutex_);
        execDumpLevel = 0;
        opUsage.clear();
        totalTimeUsage = 0;
        dumpTensorUsage = 0;
        dumpOperationUsage = 0;
    }

    static std::string ShapeToString(const std::vector<int64_t>& shape)
    {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) {
                oss << ",";
            }
            oss << std::to_string(shape[i]);
        }
        oss << "]";
        return oss.str();
    }

    template <typename Type>
    static std::string ToStrWithPrecision(const Type& value)
    {
        std::ostringstream oss;
        constexpr auto max_precision{std::numeric_limits<float>::digits10 + 1};
        oss << std::setprecision(max_precision);
        oss << value;
        return oss.str();
    }

    void WriteCsvRow(std::vector<std::string>& row, int& rowNum, FILE* file)
    {
        if (file == nullptr) {
            INTERPRETER_LOGE(OpDumpScene::DUMP_OPEN_FILE_FAILED, "File is nullptr.");
            return;
        }
        if (rowNum > 0) {
            row[0] = std::to_string(rowNum);
        }
        rowNum += 1;
        std::string textLine = row[0];
        for (size_t i = 1; i < row.size(); ++i) {
            if (row[i].find(',') != std::string::npos) {
                textLine += ",\"" + row[i] + "\"";
            } else {
                textLine += "," + row[i];
            }
        }
        (void)fprintf(file, "%s\n", textLine.c_str());
    }

    std::string DumpStatistics() const
    {
        auto& nonConstSelf = const_cast<FunctionInterpreter&>(*this);
        std::lock_guard<std::mutex> dumpGuard(nonConstSelf.dumpStateMutex_);
        std::stringstream ss;
        const int labelWidth = 24;
        uint64_t totalOpUsage = 0;
        for (auto& [opcode, time] : opUsage) {
            if (time) {
                totalOpUsage += time;
                ss << std::left << std::setw(labelWidth) << opcode << ": " << time << "\n";
            }
        }
        ss << std::left << std::setw(labelWidth) << "TotalTimeUsage"
           << ": " << totalTimeUsage << "\n";
        ss << std::left << std::setw(labelWidth) << "TotalOpUsage"
           << ": " << totalOpUsage << "\n";
        if (dumpTensorUsage) {
            ss << std::left << std::setw(labelWidth) << "TotalDumpTensorUsage:"
               << ": " << dumpTensorUsage << "\n";
        }
        if (dumpOperationUsage) {
            ss << std::left << std::setw(labelWidth) << "TotalDumpOperationUsage"
               << ": " << dumpOperationUsage << "\n";
        }
        return ss.str();
    }

    std::shared_ptr<FunctionCaptureExecution> ExecuteUnit(Function* func,
                                                          const std::shared_ptr<FunctionCaptureExecution>& capture)
    {
        auto unitCapture = std::make_shared<FunctionCaptureExecution>(func);
        auto symbolDict = capture->CaptureTo(unitCapture->baseline);
        SetSymbolDict(symbolDict);
        loopSymbolDict = capture->loopSymbolDict;

        Function* target = func;
        if (func->GetRootFunction()) {
            target = func->GetRootFunction();
        }
        unitCapture->CaptureGoldenFrom(unitCapture->baseline);
        unitCapture->CaptureSymbolDictFrom(capture->symbolDict);
        unitCapture->frameList = ExecuteFunctionCapture(target, nullptr, unitCapture->golden);

        return unitCapture;
    }

    std::shared_ptr<FunctionControlFlowExecution> RunForControlFlow(
        const std::string& funcKey, const std::unordered_map<int, TileOpFormat>& slotTileOpFormatDict,
        const std::unordered_map<int, std::shared_ptr<StorageData>>& slotDataViewDict,
        const std::unordered_map<int, std::shared_ptr<LogicalTensorData>>& slotLogicalDataViewDict,
        const std::unordered_set<int>& outputSlotSet,
        const std::unordered_map<std::string, ScalarImmediateType>& controlFlowSymbolDict)
    {
        execDumpFuncKey = funcKey;
        std::shared_ptr<FunctionControlFlowExecution> execution = std::make_shared<FunctionControlFlowExecution>();

        SetSymbolDict(controlFlowSymbolDict);
        auto findInputIndex = [this](std::shared_ptr<LogicalTensorData>& inputDataView) -> int {
            auto inputList = this->GetInputDataViewList();
            for (size_t k = 0; k < inputList.size(); k++) {
                if (inputList[k] == inputDataView) {
                    return static_cast<int>(k);
                }
            }
            return -1;
        };
        slotDataViewDict_ = slotDataViewDict;
        auto slotLogicalDataViewLocal = slotLogicalDataViewDict;
        outputSlotSet_ = outputSlotSet;
        for (auto& [slot, tileOpFormat] : slotTileOpFormatDict) {
            if (tileOpFormat == TileOpFormat::TILEOP_NZ) {
                ASSERT(ControlFlowScene::FUNC_SLOT_MISSING, slotLogicalDataViewLocal.count(slot));
                auto dataView = slotLogicalDataViewLocal[slot];
                auto inputIndex = findInputIndex(dataView);
                auto nzInputDataView = FormatNZ2ND(dataView);
                slotLogicalDataViewLocal[slot] = nzInputDataView;
                slotDataViewDict_[slot] = nzInputDataView->GetData()->GetRawData();
                UpdateInputDataViewList(inputIndex, nzInputDataView);
            }
        }

        DumpBegin();
        TimeStamp ts;
        ExecuteControlFlow(entry_, *execution);
        for (auto& slot : outputSlotSet_) {
            if (slotTileOpFormatDict.count(slot) && slotTileOpFormatDict.at(slot) == TileOpFormat::TILEOP_NZ) {
                auto dataView = slotLogicalDataViewDict.at(slot);
                slotDataViewDict_[slot] = FormatND2NZ(dataView)->GetData()->GetRawData();
            }
        }

        totalTimeUsage += ts.Duration();
        DumpEnd();
        return execution;
    }

    std::shared_ptr<FunctionCaptureExecution> RunForPass(std::string& funcKey, Function* func,
                                                         const std::shared_ptr<FunctionCaptureExecution>& capture)
    {
        execDumpFuncKey = funcKey;

        DumpBegin();
        TimeStamp ts;
        mixGlobalTensorDict.clear();
        std::shared_ptr<FunctionCaptureExecution> unitCapture = ExecuteUnit(func, capture);
        DumpEnd();
        TimeStamp ts1;
        DumpPassTensorDiff(unitCapture, capture);
        dumpTensorUsage += ts1.Duration();
        totalTimeUsage += ts.Duration();
        return unitCapture;
    }

    std::shared_ptr<FunctionCaptureExecution> RunForExecuteGraph(
        const std::string& funcKey, Function* func, const std::shared_ptr<FunctionCaptureExecution>& capture)
    {
        execDumpFuncKey = funcKey;

        DumpBegin();
        TimeStamp ts;
        std::shared_ptr<FunctionCaptureExecution> unitCapture = ExecuteUnit(func, capture);
        totalTimeUsage += ts.Duration();
        DumpEnd();
        return unitCapture;
    }
};

} // namespace npu::tile_fwk
