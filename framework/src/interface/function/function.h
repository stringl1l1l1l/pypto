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

#pragma once

#include <algorithm>
#include <cstddef>
#include <set>
#include <string>
#include <memory>
#include <stack>
#include <unordered_map>
#include <unordered_set>

#include "tilefwk/error.h"
#include "tilefwk/tilefwk.h"
#include "interface/operation/operation.h"
#include "interface/inner/pre_def.h"
#include "interface/tensor/symbolic_scalar.h"
#include "tilefwk/tensor.h"
#include "interface/tensor/tensormap.h"
#include "interface/tensor/tensor_slot.h"
#include "interface/cache/hash.h"
#include "passes/pass_utils/subfunc_utils.h"

#include "ir/function.h"

using namespace pypto;

namespace npu::tile_fwk {
constexpr int FUNCTION_MAX_INCASTS = 10000;
constexpr int FUNCTION_MAX_CV_CORES = 10000;
constexpr int64_t RAW_TENSOR_INDEX_BIT_MASK = 1L << 62;

inline const BiMap<FunctionType>& GetFunctionTypeNameDict()
{
    static BiMap<FunctionType> dict{{{FunctionType::INVALID, "INVALID"},
                                     {FunctionType::EAGER, "EAGER"},
                                     {FunctionType::STATIC, "STATIC"},
                                     {FunctionType::DYNAMIC, "DYNAMIC"},
                                     {FunctionType::DYNAMIC_LOOP, "DYNAMIC_LOOP"},
                                     {FunctionType::DYNAMIC_LOOP_PATH, "DYNAMIC_LOOP_PATH"}}};
    return dict;
}

enum class GraphType { TENSOR_GRAPH, TILE_GRAPH, EXECUTE_GRAPH, BLOCK_GRAPH, LEAF_VF_GRAPH, INVALID };

inline const BiMap<GraphType>& GetGraphTypeNameDict()
{
    static BiMap<GraphType> dict{{{GraphType::TENSOR_GRAPH, "TENSOR_GRAPH"},
                                  {GraphType::TILE_GRAPH, "TILE_GRAPH"},
                                  {GraphType::EXECUTE_GRAPH, "EXECUTE_GRAPH"},
                                  {GraphType::BLOCK_GRAPH, "BLOCK_GRAPH"},
                                  {GraphType::LEAF_VF_GRAPH, "LEAF_VF_GRAPH"},
                                  {GraphType::INVALID, "INVALID"}}};
    return dict;
}

enum class EndFuncReturnParam { INPUT = 0, OUTPUT, ARGS };

enum class SortOperationsMode {
    GENERAL,
    LIGHTWEIGHT,
    LIGHTWEIGHT_STABLE,
};

enum class MixResourceType {
    UNKNOWN = 0,
    ONE_CUBE_ONE_VECTOR = 1, // 1C1V
    ONE_CUBE_TWO_VECTOR = 2  // 1C2V
};

struct FunctionCallArgs {
    LogicalTensors iOperands;
    LogicalTensors oOperands;
    std::vector<OperandAttribute> iOpAttr;
    std::vector<OperandAttribute> oOpAttr;
    std::map<int, SymbolicScalar> outIndexToExpr;
    std::vector<std::vector<SymbolicScalar>> argList;
};

using OperationDeleter = std::function<bool(std::shared_ptr<Operation>&, Function&)>;

using TensorGraphInfo = std::tuple<std::vector<LogicalTensors>, std::vector<LogicalTensors>,
                                   std::set<std::shared_ptr<Operation>>, std::set<std::shared_ptr<Operation>>,
                                   std::set<std::shared_ptr<LogicalTensor>>, std::set<std::shared_ptr<LogicalTensor>>>;

class OperationsViewer {
    friend class SubgraphToFunction;
    friend class ExpandFunction;
    friend class VFFusionPass;

public:
    class IteratorDelimiter {};
    class Iterator {
    public:
        explicit Iterator(const std::vector<std::shared_ptr<Operation>>& operations) : operations_(operations) {}

        void operator++()
        {
            FE_ASSERT(cur_ <= operations_.size())
                << "operator(++) out of its size. cur_: " << cur_ << ", operations_.size(): " << operations_.size();
            cur_++;
        }

        void operator++(int) { ++(*this); }

        [[nodiscard]] Operation& operator*() const { return *operations_[cur_]; }
        [[nodiscard]] Operation* operator->() const { return operations_[cur_].get(); }

        [[nodiscard]] bool operator==(const IteratorDelimiter&) const { return operations_.size() == cur_; }
        [[nodiscard]] bool operator!=(const IteratorDelimiter&) const { return operations_.size() != cur_; }

    private:
        size_t cur_{0};
        const std::vector<std::shared_ptr<Operation>>& operations_;
    };

public:
    OperationsViewer(const std::vector<std::shared_ptr<Operation>>& operations,
                     const std::unordered_map<const Operation*, int>& opPosition)
        : operations_(operations), opPosition_(opPosition)
    {}
    [[nodiscard]] auto size() const { return operations_.size(); }
    [[nodiscard]] auto begin() const { return Iterator(operations_); }
    [[nodiscard]] static auto end() { return IteratorDelimiter{}; }
    [[nodiscard]] Operation& at(size_t index) const { return *operations_[index]; }
    [[nodiscard]] Operation& back() const { return *operations_.back(); }
    [[nodiscard]] Operation& operator[](const size_t index) const { return *operations_[index]; }
    [[nodiscard]] std::vector<Operation*> DuplicatedOpList() const;

    [[nodiscard]] bool Contains(const Operation& op) const { return opPosition_.count(&op) > 0; }
    [[nodiscard]] int GetOpPosition(const Operation& op) const
    {
        auto it = opPosition_.find(&op);
        if (it == opPosition_.end()) {
            FE_ASSERT(FeError::NOT_EXIST, false) << "Magic[" << op.opmagic << "] Op has not been found in opPosition.";
            return 0;
        }
        return it->second;
    }

    [[nodiscard]] std::pair<int, bool> FindOpPosition(const Operation& op) const
    {
        auto it = opPosition_.find(&op);
        if (it == opPosition_.end()) {
            return {0, false};
        }
        FE_ASSERT(operations_[it->second].get() == &op)
            << "operations_[it->second].get(): 0x" << reinterpret_cast<uintptr_t>(operations_[it->second].get())
            << "&op: " << reinterpret_cast<uintptr_t>(&op);
        return {it->second, true};
    }
    [[nodiscard]] bool IsEmpty() const { return operations_.empty(); }

private:
    const std::vector<std::shared_ptr<Operation>>& operations_;
    const std::unordered_map<const Operation*, int>& opPosition_;
};

struct LeafFuncAttribute {
    static constexpr int32_t INVALID_MIX_ID = -1;
    std::string kernelName;          // 异构子图kernel函数名
    std::string kernelNameMainBlock; // 异构子图kernel函数名(运行时选择主尾块场景中的主块)
    std::string binPath;             // 异构子图二进制文件路径
    std::string binPathMainBlock;    // 异构子图二进制文件路径(运行时选择主尾块场景中的主块)
    std::string kernelDeclare;       // 异构子图代码的kernel声明，用于后续整体调用
    std::string kernelDeclareMainBlock; // 异构子图代码的kernel声明，用于后续整体调用(运行时选择主尾块场景中的主块)
    CoreType coreType{CoreType::INVALID};
    AIVCore aivCore{AIVCore::UNSPECIFIED}; // 表示Mix子图切完的vector子图放在AIV0核还是AIV1核，0=AIV0, 1=AIV1, -1=未指定
    int32_t mixId{INVALID_MIX_ID}; // 表示哪些切完的leafFunction是从一个Mix子图切出来的
    MixResourceType mixResourceType{MixResourceType::UNKNOWN}; // mix任务资源诉求是1c2v还是1c1v
    std::vector<int32_t> aicpuLeafCode;
    std::vector<int> outcastCopyOutResolveCounterList;
    int copyOutResolveSize{0};
};

struct DynloopFunctionPathCondition {
    bool isSat_;
    bool isConst_;
    SymbolicScalar cond_;
    std::string file_;
    int line_;

    const bool& IsSat() const { return isSat_; }
    bool& IsSat() { return isSat_; }

    const SymbolicScalar& GetCond() const { return cond_; }
    const std::string GetFile() const { return file_; }
    int GetLine() const { return line_; }

    DynloopFunctionPathCondition() {}
    DynloopFunctionPathCondition(bool isSat, bool isConst, const SymbolicScalar& cond, const std::string& file,
                                 int line)
        : isSat_(isSat), isConst_(isConst), cond_(cond), file_(file), line_(line)
    {}
};

struct DynloopFunctionPath {
    Function* root;
    std::vector<DynloopFunctionPathCondition> pathCondList;
    Operation* callop;

    DynloopFunctionPath(Function* pathRoot, const std::vector<DynloopFunctionPathCondition>& pathConds,
                        Operation* operation)
        : root(pathRoot), pathCondList(pathConds), callop(operation)
    {}

    const std::vector<DynloopFunctionPathCondition>& GetPathCondList() const { return pathCondList; }
    Function* GetRoot() const { return root; }
};

struct DynloopFunctionPathNode {
    SymbolicScalar cond;
    std::shared_ptr<DynloopFunctionPathNode> branchNodeList[2] = {nullptr, nullptr};

    Function* root{nullptr};

    DynloopFunctionPathNode() = default;
    explicit DynloopFunctionPathNode(Function* pathRoot) : root(pathRoot) {}

    std::string Dump() const;
};

struct DynloopFunctionAttribute {
    std::string iterSymbolName;
    LoopRange loopRange;
    LoopRange originalRange;

    bool submitBeforeLoop;
    bool parallel;
    int unrollTimes{1};
    std::vector<DynloopFunctionPath> pathList;

    std::vector<DynloopFunctionPathCondition> currPathCond;
    std::vector<Operation*> underDynLoopCallOpGroup_;
    size_t currIndex{0};

    DynloopFunctionAttribute(const std::string& symbolName, const LoopRange& range, const LoopRange& originRange,
                             bool submit = false, bool parallelMode = false)
        : iterSymbolName(symbolName),
          loopRange(range),
          originalRange(originRange),
          submitBeforeLoop(submit),
          parallel(parallelMode)
    {}

    const std::string& IterSymbolName() { return iterSymbolName; }
    const SymbolicScalar& Begin() { return loopRange.Begin(); }
    const SymbolicScalar& End() { return loopRange.End(); }
    const SymbolicScalar& Step() { return loopRange.Step(); }
    const std::vector<DynloopFunctionPath>& GetPathList() const { return pathList; }

    std::shared_ptr<DynloopFunctionPathNode> BuildPathNode();
    std::string DumpBranch() const;
    void IterationBegin() { CreateCurrCond(); }

    static bool IsLoopBeginCall(const SymbolicScalar& symbol)
    {
        if (!symbol.IsExpression()) {
            return false;
        }
        auto expr = std::static_pointer_cast<RawSymbolicExpression>(symbol.Raw());
        if (expr->Opcode() == SymbolicOpcode::T_MOP_CALL) {
            auto raw = expr->OperandList()[0];
            auto rawSymbol = std::dynamic_pointer_cast<RawSymbolicSymbol>(raw);
            auto callee = rawSymbol->Name();
            return callee == AddRuntimePrefix(SymbolHandler::GetNameByHandlerId(SymbolHandlerId::IsLoopBegin));
        }
        return false;
    }

    static bool IsLoopEndCall(const SymbolicScalar& symbol)
    {
        if (!symbol.IsExpression()) {
            return false;
        }
        auto expr = std::static_pointer_cast<RawSymbolicExpression>(symbol.Raw());
        if (expr->Opcode() == SymbolicOpcode::T_MOP_CALL) {
            auto raw = expr->OperandList()[0];
            auto rawSymbol = std::dynamic_pointer_cast<RawSymbolicSymbol>(raw);
            auto callee = rawSymbol->Name();
            return callee == AddRuntimePrefix(SymbolHandler::GetNameByHandlerId(SymbolHandlerId::IsLoopEnd));
        }
        return false;
    }

    static bool IsLoopBeginOrEndExpr(const SymbolicScalar& symbol)
    {
        if (!symbol.IsExpression()) {
            return false;
        }
        auto expr = std::static_pointer_cast<RawSymbolicExpression>(symbol.Raw());
        return expr->IsLoopBeginCall() || expr->IsLoopEndCall();
    }
    std::vector<DynloopFunctionPathCondition> GenCondWithBeginEnd(
        const std::vector<DynloopFunctionPathCondition>& conds) const;
    bool IterationEnd(int unroll, Function* pathFunc, Operation* operation);
    bool AppendCond(const SymbolicScalar& cond, const std::string& file, int line);
    bool GuessCondResult(const SymbolicScalar& cond, bool& result);

private:
    void CreateCurrCond();
};

std::vector<uint8_t> LoadBinData(const std::string& binPath);

struct CceCodeInfo {
    uint32_t coreType;
    uint32_t psgId;
    uint64_t funcHash;
    std::vector<int32_t> aicpuLeafCode;
    int32_t wrapVecId{-1};
    uint32_t mixResourceType{0};
};

struct OriArgInfo {
    uint64_t addr;
    uint64_t size;
    bool needPrefetch;

    bool operator==(const OriArgInfo& other) const
    {
        return addr == other.addr && size == other.size && needPrefetch == other.needPrefetch;
    }

    std::string Dump()
    {
        std::ostringstream oss;
        oss << "addr: " << addr << ", size: " << size << ", needPrefetch: " << (needPrefetch ? "true" : "false");
        return oss.str();
    }
};

struct L2Info {
    uint64_t tensorSize;
    uint64_t tensorIdx;
    L2Info(uint64_t size, uint64_t idx) : tensorSize(size), tensorIdx(idx) {}
};

enum class ParamDirection {
    NONE = 0,
    IN = 1,
    OUT = 2,
    INOUT = IN | OUT,
};

struct DyndevFunctionAttribute {
    std::vector<std::reference_wrapper<const Tensor>> startArgsInputTensorList;
    std::vector<std::reference_wrapper<const Tensor>> startArgsOutputTensorList;
    // Holds shared ownership of the input slot tensors so they outlive the references
    // in startArgsInputTensorList; slot tensors may be migrated by SetSameSlot and
    // would otherwise be destructed before the function attribute is consumed.
    std::vector<std::shared_ptr<Tensor>> startArgsInputSlotTensorList;

    std::vector<std::shared_ptr<LogicalTensor>> startArgsInputLogicalTensorList;
    std::vector<std::shared_ptr<LogicalTensor>> startArgsOutputLogicalTensorList;

    std::vector<ParamDirection> startArgsDirectionList;

    struct ValueDependDesc {
        uint64_t getInputDataCount{0};
        uint64_t getTensorDataCount{0};
    };
    std::unordered_map<Function*, ValueDependDesc> valueDependDescDict;

    std::vector<std::pair<std::string, std::string>> readyOnHostCpuPairs;
    std::vector<size_t> valueDependInputIndices;

    struct FunctionGroup {
        /* loop */
        OrderedSet<Function*> loopList;
        /* loop path */
        OrderedSet<Function*> loopPathList;
        std::unordered_map<Function*, OrderedSet<RawSymbolicScalarPtr>> loopPathCondList;
        /* devRoot */
        OrderedSet<Function*> devRootList;
        /* devLeaf */
        OrderedSet<Function*> devLeafList;
        std::unordered_map<Function*, OrderedSet<Operation*>> devLeafOpList;
    } funcGroup;

    SymbolicSymbolTable symbolTable;

    std::unordered_map<std::string, ScalarImmediateType> inputSymbolDict;

    struct ExpressionTableDictGroup {
        std::unordered_map<Function*, SymbolicExpressionTable> loopBesDict;
        std::unordered_map<Function*, std::unordered_map<RawSymbolicScalarPtr, SymbolicExpressionTable>>
            loopPathCondDict;
        std::unordered_map<Function*, SymbolicExpressionTable> devRootCoaDict;
        std::unordered_map<Function*, std::unordered_map<Operation*, SymbolicExpressionTable>> devLeafOpDict;
    } exprTableDictGroup;

    /*
     *  AOT code for expression table:
     *      signature: uint64_t(*)(uint64_t *symbolTable)
     *      input:
     *          uint64_t *symbolTable
     *      output:
     *          uint64_t, expression result
     */
    std::vector<std::vector<uint8_t>> expressionTableBinaryList;

    IncastOutcastLink inoutLink;

    std::unordered_map<Function*, Function*> rootTileDict;                           // root -> tile
    std::unordered_map<Function*, int> rootFuncKeyDict;                              // root -> funcKey
    std::unordered_map<int, std::unordered_map<Function*, int>> slotRootIncastDict;  // slotIndex -> root -> incastIndex
    std::unordered_map<int, std::unordered_map<Function*, int>> slotRootOutcastDict; // sloIndex -> root -> utcastIndex

    OrderedSet<Function*> cceLeafList;
    std::vector<std::vector<uint8_t>> devEncodeList;
    std::vector<CceCodeInfo> cceCodeInfo;
    std::vector<L2Info> l2InfoList;
    std::vector<uint8_t> disableL2List;
    /*
     *  AOT code for control flow graph binary code:
     *      signature: uint64_t(*)(int64_t *symbolTable, void (*call)(void *ctx, uint64_t rootKey), void *ctx)
     *      input:
     *          int64_t *symbolTable
     *          void (*call)(void *ctx, uint64_t rootKey)
     *          void *ctx
     *      output:
     *          0
     */
    std::vector<uint8_t> hostControlFlowBinary; // using host system gcc
    std::vector<uint8_t> devControlFlowBinary;  // using CANN gcc (in order to work on x86 host)

    std::vector<int> startArgsInputSymbolIndexList;

    std::vector<std::string> commGroupNames;

    SymbolicScalar maxDynamicAssembleOutcastMem;
    SymbolicScalar maxDynamicCellMatchTableMem;
    struct DynamicCellMatchLaunchMeta {
        int slotIndex{-1};
        uint64_t descOffset{0};
        std::vector<int> cellShape;
        std::vector<std::vector<SymbolicScalar>> candidateRawDims;
    };
    std::vector<DynamicCellMatchLaunchMeta> dynamicCellMatchLaunchMetaList;

    // Runtime slot ids that emit RUNTIME_SlotMarkNeedAlloc; prepared before BuildControlFlow for pool sizing.
    std::unordered_set<int> constructAssembleNeedAllocRuntimeSlots;

    std::vector<uint8_t> devProgBinary;

    std::vector<uint8_t> kernelBinary;

    // for costmodel
    std::map<int, uint64_t> devLeafIndex2Hash;
};

enum class DynParamInfoType { VALID_SHAPE, OFFSET, END };

struct DynParamInfo {
    int dimSize;
    int tensorIndex;
    int tensorBaseAddrCoaIndex;
    DynParamInfoType type;
    int dimIndex;
    SymbolicScalar dim;
    bool isBaseParam{false};
    std::string replacedSymbol;
};
// cubeL1ReuseSetting packs the L1-reuse matrix-side preference into the merge-count value as
// side * L1_REUSE_SIDE_BASE + count (side 0=auto / 1=left(L0A) / 2=right(L0B)). The frontend
// encodes it; L1CopyInReuseMerge decodes it. Must stay in sync with python config.py.
constexpr int64_t L1_REUSE_SIDE_BASE = 1000000;

struct ParamConfigs {
    bool dynamicAlignedOps;
    int sgPgLowerBound{1};
    int sgParallelNum{1};
    int sgMgCopyInUpperBound{2 * 1024 * 1024};
    std::string sgPartitionAlgorithm{"Iso"};
    uint8_t machineConfig_{0}; // machine config
    std::map<int64_t, int64_t> cubeL1ReuseSetting;
    std::map<std::string, int64_t> cubeL1ReuseSettingByFunc;
    std::map<int64_t, int64_t> cubeNBufferSetting;
    std::map<std::string, int64_t> cubeNBufferSettingByFunc;
    std::string OoOPreScheduleMethod{"PriorDFS"};
    int mgVecParallelLb{48};
    std::map<int64_t, int64_t> vecNBufferSetting;
    std::map<std::string, int64_t> vecNBufferSettingByFunc;
    std::map<std::string, int64_t> cubeL1ReuseSettingByLabel;
    std::map<std::string, int64_t> cubeNBufferSettingByLabel;
    std::map<std::string, int64_t> vecNBufferSettingByLabel;
    int copyOutResolveCoalescing{0};
    bool combineAxis{false};
    int autoMixPartition{0};
    std::string oooSchedMode;
    std::string oooSortModeAic;
    std::string oooSortModeAiv;
};

struct FunctionParamInfo {
    const Tensor* key;           // 显式指定参数时所指定参数的Tensor信息
    LogicalTensorPtr beginValue; // Begin Function时Tensor指向的 LogicalTensor
    LogicalTensorPtr endValue;   // End Function时Tensor指向的 LogicalTensor
};

class VarDependency {
public:
    struct Entry {
        std::unordered_set<ir::StmtPtr> producers;
        std::unordered_set<ir::StmtPtr> consumers;
    };

    void AddProducer(ir::VarPtr var, ir::StmtPtr producer) { dependencies_[var].producers.insert(producer); }
    void AddConsumer(ir::VarPtr var, ir::StmtPtr consumer) { dependencies_[var].consumers.insert(consumer); }

    void RemoveProducer(ir::VarPtr var, ir::StmtPtr producer)
    {
        auto it = dependencies_.find(var);
        if (it != dependencies_.end()) {
            it->second.producers.erase(producer);
        }
    }
    void RemoveConsumer(ir::VarPtr var, ir::StmtPtr consumer)
    {
        auto it = dependencies_.find(var);
        if (it != dependencies_.end()) {
            it->second.consumers.erase(consumer);
        }
    }
    void RemoveVar(ir::VarPtr var) { dependencies_.erase(var); }
    void Clear() { dependencies_.clear(); }

    const std::unordered_set<ir::StmtPtr>& GetProducers(ir::VarPtr var) const
    {
        static const std::unordered_set<ir::StmtPtr> empty;
        auto it = dependencies_.find(var);
        return it != dependencies_.end() ? it->second.producers : empty;
    }
    const std::unordered_set<ir::StmtPtr>& GetConsumers(ir::VarPtr var) const
    {
        static const std::unordered_set<ir::StmtPtr> empty;
        auto it = dependencies_.find(var);
        return it != dependencies_.end() ? it->second.consumers : empty;
    }
    bool HasDependency(ir::VarPtr var) const { return dependencies_.find(var) != dependencies_.end(); }
    bool HasProducer(ir::VarPtr var, ir::StmtPtr producer) const
    {
        auto it = dependencies_.find(var);
        return it != dependencies_.end() && it->second.producers.count(producer) > 0;
    }
    bool HasConsumer(ir::VarPtr var, ir::StmtPtr consumer) const
    {
        auto it = dependencies_.find(var);
        return it != dependencies_.end() && it->second.consumers.count(consumer) > 0;
    }
    size_t Size() const { return dependencies_.size(); }
    bool Empty() const { return dependencies_.empty(); }

    const std::unordered_map<ir::VarPtr, Entry>& GetAllDependencies() const { return dependencies_; }

private:
    std::unordered_map<ir::VarPtr, Entry> dependencies_;
};

#ifndef INVALID_IOINDEX
#define INVALID_IOINDEX (-1)
#endif

class Function : public ir::Function {
public:
    friend class ExpandFunction;
    friend class VFFusionPass;
    friend class TokenUtils;

    std::vector<std::shared_ptr<LogicalTensor>> inCasts_;  // Input tensors
    std::vector<std::shared_ptr<LogicalTensor>> outCasts_; // Output tensors
    ir::SeqStmtsPtr originalBody_;

    int opSeed_{FUNCTION_MAX_INCASTS};
    SubfuncTopologyInfoTy topoInfo_;         // root function持有，对应1.0的SubgraphTopologyInfoTy
    std::map<uint64_t, Function*> programs_; // root function持有，所有异构的leaf function
    Function* rootFunc_ = nullptr;           // TileGraph和RootGraph都需要保留，且需要映射关系
    ParamConfigs paramConfigs_;
    // vf融合适配需要pass间传递的参数
    std::unordered_map<PipeType, int> pipeEndTime; // function中每个pipe执行结束的时间
    std::unordered_map<Operation*, Operation*> setOpMap;
    std::unordered_map<Operation*, Operation*> waitOpMap;
    std::vector<Operation*> oriOpList;

    Function(const Program& belongTo, const std::string& funcMagicName, const std::string& funcRawName,
             Function* parentFunc);

    // DYNAMIC magic names use a process-lifetime suffix (not reset by Program::Reset).
    // Other types still use FUNCTION CurId so the suffix matches functionMagic_.
    static int NextMagicNameSuffix(FunctionType funcType);

    Function(const Function& other) = delete;
    Function(Function&& other) = delete;
    Function& operator=(const Function& other) = delete;
    Function& operator=(Function&& other) = delete;

    bool IsCompiledFunction() const
    {
        return IsFunctionTypeAndGraphType(FunctionType::STATIC, {GraphType::EXECUTE_GRAPH, GraphType::BLOCK_GRAPH});
    }
    std::unordered_set<int> LoopCheck();
    FunctionHash ComputeHash();
    void BuildTensorMap();
    std::vector<std::shared_ptr<Operation>> GetSortedOperations() const;
    OperationsViewer Operations(bool sorted = true, SortOperationsMode mode = SortOperationsMode::GENERAL);
    OperationsViewer OperationsAfterOOO();
    void RecordOOOSeq();
    // 这个LeafOperations写法破坏封装性，但是是针对LeafFunction特有的，后续在Function按类拆分的时候会将其只放到LeafFunction中
    std::vector<OperationPtr>& GetProgramOp();
    void SetProgramOp(const std::vector<OperationPtr>& operations);
    void SortOperations(SortOperationsMode mode = SortOperationsMode::GENERAL);
    void ScheduleBy(const std::vector<Operation*>& newList, bool needRefresh = false);
    void EraseOperations(bool eraseRelatedTensor = true, bool sorted = true,
                         SortOperationsMode mode = SortOperationsMode::GENERAL);
    void EraseOperations(const OperationDeleter& deleter);
    void AddGlobalTensor(std::shared_ptr<LogicalTensor> tensor) { globalTensors_.emplace(tensor); };
    void AddOperationGroup(std::vector<Operation*> operationGroup);
    const auto& GetGroupByID(const size_t groupID) const
    {
        FE_ASSERT(groupID < operationGroups_.size())
            << "groupID: " << groupID << ", operationGroups_.size(): " << operationGroups_.size();
        return operationGroups_[groupID];
    }
    void ClearOperationGroups();
    void CheckGroupValid() const;

    void CreateLeafInAndOutCast(const LogicalTensorPtr& inOrOut, LogicalTensors& inOrOutList) const;
    bool IsFromInCast(const std::shared_ptr<LogicalTensor>& tensor);
    bool IsFromOutCast(const std::shared_ptr<LogicalTensor>& tensor);
    bool IsFromDummyOutCast(int rawMagic);
    int GetIncastIndex(std::shared_ptr<LogicalTensor>& tensor) const;
    int GetOutcastIndex(std::shared_ptr<LogicalTensor>& tensor) const;
    void MergeFunctionDupIocast();
    void RemoveCallOpViewAssemble();
    void ResetOperations();

    Operation& AddOperation(const std::string& opName, LogicalTensors iOperands, const LogicalTensors& oOperands);
    Operation& AddOperation(const Opcode opCode, LogicalTensors iOperands, const LogicalTensors& oOperands);
    Operation& AddRawOperation(const Opcode opCode, const LogicalTensors& iOperands, const LogicalTensors& oOperands,
                               ir::Span span = ir::Span::Unknown());

    std::map<std::shared_ptr<RawTensor>, std::shared_ptr<RawTensor>> outIncastLinkMap; // 记录outcast 共享地址的 incast
    void SetSameMemId(const LogicalTensorPtr& operand, const LogicalTensorPtr& dst);
    void UpdateLinkMap(const std::shared_ptr<LogicalTensor>& oriLogicalTensor,
                       const std::shared_ptr<LogicalTensor>& newLogicalTensor, const bool isOutCast = false);

    std::vector<Operation*> GetAllInputOperations(const Operation& op) const;
    std::vector<Operation*> GetAllOutputOperations(const Operation& op) const;

    std::vector<Operation*> GetCallopList() const;
    std::vector<std::shared_ptr<CallOpAttribute>> GetCallopAttrList() const;
    std::vector<Function*> GetCalleeFunctionList() const;

    bool IsCube() const;

    Function* GetRootFunction() const { return rootFunc_; }

    void Substitute(std::shared_ptr<LogicalTensor> oldTensor, std::shared_ptr<LogicalTensor> newTensor);
    void SubstituteIn(std::shared_ptr<LogicalTensor> oldTensor, std::shared_ptr<LogicalTensor> newTensor);
    void SubstituteOut(std::shared_ptr<LogicalTensor> oldTensor, std::shared_ptr<LogicalTensor> newTensor);

    void DumpJsonFile(std::string fileName = "");
    Json DumpJson(bool useTable = true);
    static std::shared_ptr<Function> LoadJson(Program& belongTo, const Json& funcJson);

    std::vector<std::vector<SymbolicScalar>> NormalizeCoa(std::vector<OperandAttribute>& iOpAttr,
                                                          std::vector<OperandAttribute>& oOpAttr);
    void NormalizeCoaForInCasts(std::vector<OperandAttribute>& iOpAttr,
                                std::vector<std::vector<SymbolicScalar>>& coaLists, int& coaIndex,
                                std::unordered_map<LogicalTensorPtr, int>& normTensors,
                                const std::unordered_map<int, Operation*>& opmagicToOp);
    void NormalizeCoaForOutCasts(std::vector<OperandAttribute>& oOpAttr,
                                 std::vector<std::vector<SymbolicScalar>>& coaLists, int& coaIndex,
                                 std::unordered_map<LogicalTensorPtr, int>& normTensors,
                                 const std::unordered_map<int, Operation*>& opmagicToOp);
    void NormalizeCoaForNormalOperands(std::vector<std::vector<SymbolicScalar>>& coaLists, int& coaIndex,
                                       std::unordered_map<LogicalTensorPtr, int>& normTensors);
    void NormalizeCoaForSpecialInfo(std::vector<std::vector<SymbolicScalar>>& coaLists, int& coaIndex);
    void GetOutcastSymbolicExpr(std::map<int, SymbolicScalar>& tabel);

    void DumpTopoFile(const std::string& fileName) const;
    std::string DumpSSA() const;
    std::string Dump() const; // Serialize brief format
    void DumpFile(const std::string& filePath) const;

    LogicalTensors MakeIncasts(const std::shared_ptr<TensorSlotScope>& scope);
    LogicalTensors MakeOutcasts(const std::shared_ptr<TensorSlotScope>& scope);

    void TensorMagicCheck() const;
    void OperationLoopCheck(const std::string& errorMsg);
    bool OperationLoopCheck();
    void ValidCheck() const;

    DyndevFunctionAttribute::ValueDependDesc LookupValueDepend();

    std::shared_ptr<OpAttribute> CreateCallOpAttribute(const std::vector<std::vector<SymbolicScalar>>& argList,
                                                       const std::map<int, SymbolicScalar>& outIndexToExpr);

    bool IsEager() const { return functionType_ == FunctionType::EAGER; }
    bool IsStatic() const { return functionType_ == FunctionType::STATIC; }
    bool IsExplicit() const { return explicitArgSlots_.empty(); }
    size_t GetOperationSize() const { return operations_.size(); }
    const std::string& GetMagicName() const { return funcMagicName_; }
    std::string GetMagicNameWithHash() const { return GetMagicName() + "_" + GetFunctionHash(); }
    const std::string& GetRawName() const { return funcRawName_; }
    std::string GetOriginalRawName() const;
    void AppendCalleeMagicName(const std::string& name) { calleeMagicNameList_.push_back(name); }
    const std::vector<std::string>& GetCalleeMagicNameList() const { return calleeMagicNameList_; }
    int GetFuncMagic() const { return functionMagic_; }
    const TensorMap& GetTensorMap() const { return tensorMap_; }
    TensorMap& GetTensorMap() { return tensorMap_; }
    int64_t GetStackWorkespaceSize() const { return stackWorkespaceSize_; }
    void SetStackWorkespaceSize(int64_t size) { stackWorkespaceSize_ = size; }

    size_t GetTotalSubGraphCount() const { return totalSubGraphCount_; }

    void SetTotalSubGraphCount(const size_t totalSubGraphCount) { totalSubGraphCount_ = totalSubGraphCount; }

    const std::vector<std::shared_ptr<LogicalTensor>>& GetOriginIncast() const { return originInCasts_; }
    const std::vector<std::shared_ptr<LogicalTensor>>& GetOriginOutcast() const { return originOutCasts_; }
    void AddOriginIncast(const std::shared_ptr<LogicalTensor>& tensor);
    void AddOriginOutcast(const std::shared_ptr<LogicalTensor>& tensor);
    const std::vector<std::shared_ptr<LogicalTensor>>& GetIncast() const { return inCasts_; }
    const std::vector<std::shared_ptr<LogicalTensor>>& GetOutcast() const { return outCasts_; }
    FunctionHash GetFunctionHash() const { return functionHash_; }

    bool HasParent() const { return parent_ != nullptr; }
    auto& Parent() { return *parent_; }
    const Function& Parent() const { return *parent_; }
    void SetParent(Function* parent) { parent_ = parent; }

    const Program& BelongTo() const { return belongTo_; }
    void UpdateBelongToThis();
    bool IsFlattening() const;

    FunctionType GetFunctionType() const;
    void SetFunctionType(FunctionType type);
    std::string GetFunctionTypeStr() const;

    GraphType GetGraphType() const;
    void SetGraphType(GraphType type);

    bool IsFunctionType(FunctionType type) const;
    bool IsFunctionType(std::set<FunctionType> types) const;
    bool IsGraphType(GraphType type) const;
    bool IsGraphType(std::set<GraphType> types) const;
    bool IsFunctionTypeAndGraphType(FunctionType funcType, GraphType graphType) const;
    bool IsFunctionTypeAndGraphType(FunctionType funcType, std::set<GraphType> graphTypes) const;
    bool IsFunctionTypeAndGraphType(std::set<FunctionType> funcTypes, GraphType graphType) const;
    bool IsFunctionTypeAndGraphType(std::set<FunctionType> funcTypes, std::set<GraphType> graphTypes) const;

    static std::string CreateRootRawName(const std::string& funcRawName) { return funcRawName + "_root"; }
    static std::string CreateLeafRawName(const std::string& funcRawName, int subgraphId)
    {
        return funcRawName + "_leaf_" + std::to_string(subgraphId);
    }

    void BeginFunction(const std::vector<std::reference_wrapper<const Tensor>>& explicitOpArgs);
    FunctionCallArgs EndFunction(const std::shared_ptr<TensorSlotScope>& scope);
    /* -------------------------常用改图接口------------------------------ */
    Operation* GetOpByOpMagic(const int opMagic) const;

    static bool TensorReuse(const LogicalTensorPtr& dstTensor, const LogicalTensorPtr& srcTensor);
    std::set<Operation*, LogicalTensor::CompareOp> FindConsumers(const Operation& op) const;
    std::set<Operation*, LogicalTensor::CompareOp> FindProducers(const Operation& op) const;
    const SubfuncInvokeInfoTy& GetSubFuncInvokeInfo(const size_t i) const;
    void GetAnIslandIncastsOutcasts(const std::map<int, int>& opToSubgraph, const int subgraphID,
                                    const std::vector<Operation*>& operations,
                                    std::vector<std::shared_ptr<LogicalTensor>>& iOperands,
                                    std::vector<std::shared_ptr<LogicalTensor>>& oOperands) const;

    void SetDynloopAttribute(const std::shared_ptr<DynloopFunctionAttribute>& attr) { dynloopAttr_ = attr; }
    const std::shared_ptr<DynloopFunctionAttribute>& GetDynloopAttribute() const { return dynloopAttr_; }
    std::shared_ptr<DynloopFunctionAttribute>& GetDynloopAttribute() { return dynloopAttr_; }

    void SetDyndevAttribute(const std::shared_ptr<DyndevFunctionAttribute>& attr) { dyndevAttr_ = attr; }
    const std::shared_ptr<DyndevFunctionAttribute>& GetDyndevAttribute() const { return dyndevAttr_; }
    std::shared_ptr<DyndevFunctionAttribute>& GetDyndevAttribute() { return dyndevAttr_; }

    void SetLeafFuncAttribute(const std::shared_ptr<LeafFuncAttribute>& attr) { leafFuncAttr_ = attr; }
    const std::shared_ptr<LeafFuncAttribute>& GetLeafFuncAttribute() const { return leafFuncAttr_; }
    std::shared_ptr<LeafFuncAttribute>& GetLeafFuncAttribute() { return leafFuncAttr_; }

    void SetSlotScope(const std::shared_ptr<TensorSlotScope>& slotScope) { slotScope_ = slotScope; }
    const std::shared_ptr<TensorSlotScope>& GetSlotScope() const { return slotScope_; }
    std::shared_ptr<TensorSlotScope>& GetSlotScope() { return slotScope_; }
    std::vector<int> GetInCastSlot(const std::shared_ptr<LogicalTensor>& incast);
    std::vector<int> GetOutCastSlot(const std::shared_ptr<LogicalTensor>& outcast);

    bool HasCallOperation();
    bool IsDynloop() const { return dynloopAttr_ != nullptr; }
    bool IsDyndev() const { return dyndevAttr_ != nullptr; }

    void HandleControlOps(Operation& op, std::vector<Operation*>& toRemoveOps) const;
    void UpdateOperandBeforeRemoveOp(Operation& op, const bool keepOutTensor = false);
    std::pair<bool, Opcode> IsAicpuSubFunction() const
    {
        Opcode code = Opcode::OP_UNKNOWN;
        for (size_t i = 0UL; i < operations_.size(); i++) {
            if ((operations_[i]->GetOpcode() != Opcode::OP_VIEW) && (operations_[i]->GetOpcode() != Opcode::OP_SLICE) &&
                (operations_[i]->GetOpcode() != Opcode::OP_ASSEMBLE) &&
                (operations_[i]->GetOpcode() != Opcode::OP_CONTRACT) &&
                (operations_[i]->GetCoreType() != CoreType::AICPU)) {
                return std::make_pair(false, Opcode::OP_UNKNOWN);
            } else if (operations_[i]->GetCoreType() == CoreType::AICPU) {
                code = operations_[i]->GetOpcode();
            }
        }
        return std::make_pair(true, code);
    }

    bool IsDummyFunction() const
    {
        if (GetLeafFuncAttribute()->mixId != LeafFuncAttribute::INVALID_MIX_ID) {
            return false;
        }
        return std::all_of(operations_.begin(), operations_.end(), [](auto& op) {
            Opcode opcode = op->GetOpcode();
            // 扩展支持的算子类型：RESHAPE、VIEW、ASSEMBLE
            return opcode == Opcode::OP_RESHAPE || IsViewLike(opcode) || IsAssembleLike(opcode) ||
                   opcode == Opcode::OP_BIND_TENSOR;
        });
    }

    const std::map<std::string, DynParamInfo>& GetDynParamTable() const { return dynParamTable_; }
    void InsertDynParam(std::string dim, DynParamInfo& info) { dynParamTable_.emplace(dim, info); }

    DynParamInfo& GetMutableDynParam(std::string dim) { return dynParamTable_[dim]; }

    bool IsUnderDynamicFunction() const { return isUnderDynamicFunction_; }
    void SetUnderDynamicFunction(bool underDynamicFunciton) { isUnderDynamicFunction_ = underDynamicFunciton; }

    bool expandFunctionAccelerate{false};

    void AddLoopCallToOrderGroup(Operation* callOp) { loopCallOrderGroup_.push_back(callOp); }

    void ApplyLoopCallOrderGroup()
    {
        if (!loopCallOrderGroup_.empty()) {
            AddOperationGroup(loopCallOrderGroup_);
        }
    }

    void AppendIncast(LogicalTensorPtr tensor, int opmagic, int k)
    {
        incastPosition.emplace_back(opmagic, k);
        inCasts_.emplace_back(tensor);
    }

    void AppendOutcast(LogicalTensorPtr tensor, int opmagic, int k)
    {
        outcastPosition.emplace_back(opmagic, k);
        outCasts_.emplace_back(tensor);
    }

    void RemoveOutcast(int idx)
    {
        if (idx < 0) {
            return;
        }
        if (idx < static_cast<int>(outcastPosition.size())) {
            outcastPosition.erase(outcastPosition.begin() + idx);
        }
        if (idx < static_cast<int>(outCasts_.size())) {
            outCasts_.erase(outCasts_.begin() + idx);
        }
        if (slotScope_ != nullptr) {
            auto& outcastSlot = slotScope_->ioslot.outcastSlot;
            if (idx < static_cast<int>(outcastSlot.size())) {
                outcastSlot.erase(outcastSlot.begin() + idx);
            }

            // rebuild partial outcast list
            auto& partialList = slotScope_->ioslot.partialUpdateOutcastList;
            auto& partialDict = slotScope_->partialUpdateOutcastDict;
            partialList.clear();
            for (size_t i = 0; i < outCasts_.size(); i++) {
                if (partialDict.find(outCasts_[i]) != partialDict.end()) {
                    partialList.push_back(i);
                }
            }
        }
    }

    const SubfuncParam& GetParameter() const { return parameter_; }
    SubfuncParam& GetParameter() { return parameter_; }
    void SetParameter(const SubfuncParam& parameter) { parameter_ = parameter; }

    int GetProgramId() const { return programId_; }
    void SetProgramId(int programId) { programId_ = programId; }

    void SetReadySubGraphIds(CoreType coreType, const std::vector<int>& readySubGraphIds)
    {
        readySubGraphIds_[coreType] = readySubGraphIds;
    }
    void EmplaceReadySubGraphIds(CoreType coreType, int readySubGraphId)
    {
        readySubGraphIds_[coreType].emplace_back(readySubGraphId);
    }
    void ReplaceReadySubGraphIds(CoreType coreType, int oldIdx, int newId)
    {
        readySubGraphIds_[coreType][oldIdx] = newId;
    }
    size_t GetReadySubGraphCount(CoreType coreType) const
    {
        auto it = readySubGraphIds_.find(coreType);
        if (it == readySubGraphIds_.end()) {
            return 0; // 返回 0 而不是抛出异常
        }
        return it->second.size();
    }
    int GetReadySubGraphId(CoreType coreType, int index) const
    {
        auto it = readySubGraphIds_.find(coreType);
        if (it == readySubGraphIds_.end()) {
            // 如果 coreType 不存在，返回一个默认值或者抛出异常
            // 这里选择抛出异常，因为调用者应该先检查 count 是否为 0
            throw std::out_of_range("CoreType not found in readySubGraphIds_");
        }
        if (index >= static_cast<int>(it->second.size())) {
            throw std::out_of_range("Index out of range in readySubGraphIds_");
        }
        return it->second[index];
    }
    int GetAllReadySubGraphCount() const
    {
        int size = 0;
        for (auto& ele : readySubGraphIds_) {
            size += ele.second.size();
        }
        return size;
    }

    static void EnableMagicLookupRecord(bool enable, Function* function)
    {
        enableMagicLookupRecord_ = enable;
        if (!enable) {
            tensorAndSubgraphToProducer_.clear();
            return;
        }
        for (Operation& op : function->Operations()) {
            int subgraphId = op.GetSubgraphID();
            for (std::shared_ptr<LogicalTensor> tensor : op.GetOOperands()) {
                std::pair<int, int> tensorAndSubgraph{tensor->GetMagic(), subgraphId};
                tensorAndSubgraphToProducer_[tensorAndSubgraph].insert(&op);
            }
        }
    }
    GetTensorDataIODescDict GetTensorDataForTensorGraph();
    GetTensorDataIODescDict GetTensorDataForLeafGraph();
    void GetTensorDataRefreshIO(const GetTensorDataIODescDict& descDict);
    void CreateTensorDataImportOp();

    void SetSpan(const ir::Span& span) { span_ = span; }

    ir::Span& GetSpan() { return span_; }
    void CleanRedundantOutCast();

    void InferParamDirection();

    void SetHiddenFunction(bool hiddenFunction) { hiddenFunction_ = hiddenFunction; }
    bool IsHiddenFunction() const { return hiddenFunction_; }

    const std::unordered_set<std::string>& LoopIdxNameList() { return loopIdxNameList_; }
    bool InsertLoopIdxNameList(const std::string& idxName);

    void SetMaxCVCoreUsage(std::pair<uint32_t, uint32_t> maxCVCores) { maxCVCoreUsage_ = maxCVCores; }
    std::pair<uint32_t, uint32_t> GetMaxCVCoreUsage() { return maxCVCoreUsage_; }

    VarDependency& GetVarDependency() { return varDependency_; }
    const VarDependency& GetVarDependency() const { return varDependency_; }
    // Rebuild the function-local token index from live operations' token lists.
    void RefreshVarDependency();

    void SetOutcastNeedAlloc(const std::shared_ptr<LogicalTensor>& outcast, bool needAlloc)
    {
        outcastNeedAllocMap_[outcast] = needAlloc;
    }
    bool IsOutcastNeedAlloc(const std::shared_ptr<LogicalTensor>& outcast) const
    {
        auto it = outcastNeedAllocMap_.find(outcast);
        return it != outcastNeedAllocMap_.end() && it->second;
    }

private:
    int functionMagic_{-1};
    std::string funcMagicName_; // Function name
    std::string funcRawName_;   // raw name
    bool sorted_{false};
    size_t totalAicSubGraphCount_ = 0;
    size_t totalAivSubGraphCount_ = 0;
    size_t totalSubGraphCount_ = 0;
    int64_t stackWorkespaceSize_ = 0;
    std::pair<uint32_t, uint32_t> maxCVCoreUsage_{FUNCTION_MAX_CV_CORES, FUNCTION_MAX_CV_CORES};
    FunctionHash functionHash_{0};
    std::vector<std::string> calleeMagicNameList_;
    std::unordered_set<std::string> loopIdxNameList_;
    bool isUnderDynamicFunction_{false};

    std::vector<std::shared_ptr<LogicalTensor>> originInCasts_;
    std::unordered_set<std::shared_ptr<LogicalTensor>> inCastsSet_; // Input tensors set
    std::vector<std::pair<int, int>> incastPosition;

    std::vector<std::shared_ptr<LogicalTensor>> originOutCasts_;
    std::map<int, int> opmagicToOutcastIdx_;
    std::vector<std::pair<int, int>> outcastPosition;

    TensorMap tensorMap_;                                              // TensorMap to register tensors
    std::unordered_set<std::shared_ptr<LogicalTensor>> globalTensors_; // global tensors

    // -----------------------子图信息------------------------
    SubfuncParam parameter_; // 每一个异构子图的形参
    int programId_;          // 异构子图的id

    // we use int instead of int64 to reduce memory usage and cache miss on aicpu
    std::map<CoreType, std::vector<int>> readySubGraphIds_;

    std::vector<std::vector<Operation*>> operationGroups_;
    std::vector<std::shared_ptr<Operation>>
        operations_; // operation的获取必须要使用Operations函数，来获取到符合拓扑序的List
    std::unordered_map<const Operation*, int> opPosition_; // position of operation in Operation.operations_
    std::vector<std::shared_ptr<Operation>> operationsAfterOOO_;
    std::unordered_map<const Operation*, int> opPositionAfterOOO_; // position of operation sequence after OOO schedule
    const Program& belongTo_;
    Function* parent_{nullptr};
    FunctionType functionType_{FunctionType::INVALID};
    GraphType graphType_{GraphType::INVALID};
    std::vector<TensorSlot> explicitArgSlots_;

    std::map<std::string, DynParamInfo> dynParamTable_;

    std::shared_ptr<DynloopFunctionAttribute> dynloopAttr_;
    std::shared_ptr<DyndevFunctionAttribute> dyndevAttr_;
    std::shared_ptr<LeafFuncAttribute> leafFuncAttr_;
    std::shared_ptr<TensorSlotScope> slotScope_;

    std::vector<Operation*> loopCallOrderGroup_;

    static bool enableMagicLookupRecord_;
    static std::map<std::pair<int, int>, std::set<Operation*, LogicalTensor::CompareOp>> tensorAndSubgraphToProducer_;
    std::shared_ptr<Tensor> getTensorDataOutcast_;
    bool hiddenFunction_{false};
    VarDependency varDependency_;
    std::unordered_map<LogicalTensorPtr, bool> outcastNeedAllocMap_;

private:
    std::vector<std::shared_ptr<Operation>> GetLightweightSortedOperations(bool preserveOriginalOrder = false) const;

    unsigned long ComputeHashOrderless() const;
    void OpValidCheck(Operation& op) const;
    void RemoveOriginIncastConsumer(const std::shared_ptr<LogicalTensor>& originIncast) const;
    std::shared_ptr<LogicalTensor> CreateIncastTensor(const std::shared_ptr<LogicalTensor>& inArgument);
    void CreateFromIncast(const LogicalTensorPtr& symbol, const LogicalTensorPtr& newIncast,
                          const LogicalTensorPtr& originIncast);
    // 将 operationCountBefore 之后新插入的 op 移到 operations_ 最前面。
    // MakeIncasts 中 CreateFromIncast 会在尾部追加 VIEW op，提前到程序开头后前端无需再排序。
    void MoveNewlyInsertedOpsToFront(size_t operationCountBefore);
    std::shared_ptr<LogicalTensor> CreateOutcastTensor(const std::shared_ptr<LogicalTensor>& outArgument,
                                                       const std::shared_ptr<RawTensor>& shareRaw = nullptr,
                                                       const std::string& name = "");
    void CreateFromOutcast(const LogicalTensorPtr& symbol, const LogicalTensorPtr& newOutcast,
                           const LogicalTensorPtr& originOutcast);
    void RebindAssembleVersionsToOutcast(const LogicalTensorPtr& originOutcast, const LogicalTensorPtr& newOutcast);

    static void AddWhenNotExistOrAssert(const std::shared_ptr<LogicalTensor>& tensor,
                                        std::map<int, int>& magicToRawMagic,
                                        std::map<int, std::shared_ptr<LogicalTensor>>& magicToLogicalTensor);
    static void MagicLookup(const Function* function, const std::vector<LogicalTensorPtr>& operand,
                            const int subGraphId, int& index, std::unordered_map<int, int>& magic2index,
                            std::stringstream& ss);
    static void ProducerMagicLookup(const Function* function, const LogicalTensorPtr& tensor,
                                    const std::set<Operation*, LogicalTensor::CompareOp>& producers,
                                    const int subGraphId, int& index, std::unordered_map<int, int>& magic2index,
                                    std::stringstream& ss);
    static void LoadTensorJson(const std::shared_ptr<Function>& func, const Json& tensorJson,
                               const std::unordered_map<int, std::shared_ptr<RawTensor>>& rawTensorDict,
                               std::unordered_map<int, std::shared_ptr<LogicalTensor>>& tensorDict);

    std::string DumpSSATitle() const;
    std::string DumpSSARawTensor(int indent = 2) const;
    std::string DumpSSAIncast(int indent = 2) const;
    std::string DumpSSAOutcast(int indent = 2) const;
    std::string DumpSSAAttribute(int indent = 2) const;
    friend class FunctionInterpreter;
    friend class RootFunctionBuilder;

    void RefreshOpPosition();
    auto AnnotateOperation();
    void EraseRelatedTensors(const std::unordered_set<std::shared_ptr<LogicalTensor>>& removeCandidateTensor,
                             const std::unordered_set<std::shared_ptr<LogicalTensor>>& removeProducerTensor,
                             const std::unordered_set<LogicalTensorPtr>& inOutCastSet);

    void FillOriginInOutCast(std::vector<Operation*>& operationList);
    void SetCallOpSlot();
    void UpdateOriIocastSlot(const std::shared_ptr<TensorSlotScope> scope);
    void DoMergeFunctionDupIncast();
    void DoMergeFunctionDupOutcast();
    TensorGraphInfo GetGraphInfo();
    void ClearUselessLink(TensorGraphInfo& graphInfo);
    void LinkIoWithCallOp(std::vector<LogicalTensors>& callopInCasts, std::vector<LogicalTensors>& callopOutCasts);
    void EraseCallOpOpnd(const FunctionHash& calleeHash, size_t index);
    void CheckAndUpdateGetTensorData(size_t currOutcastIdx, size_t newOutcastIdx);
    void CleanRedundantOutcast(std::map<Function*, std::set<size_t>>& removeRecord,
                               std::map<Function*, std::set<size_t>>& getTensorDataRecord);
};
} // namespace npu::tile_fwk
