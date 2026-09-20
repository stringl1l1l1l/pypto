/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_cell_match_dynamic.cpp
 * \brief UT for machine/runtime/launcher/cell_match_dynamic.cpp and .h
 */

#include <gtest/gtest.h>
#include <cstring>
#include <ios>
#include <memory>
#include <vector>

#define private public
#define protected public

#include "machine/runtime/launcher/cell_match_dynamic.h"
#include "machine/runtime/launcher/device_launcher_binding.h"
#include "machine/runtime/launcher/aicore_model_launcher.h"
#include "machine/device/dynamic/context/device_stitch_context.h"
#include "machine/device/dynamic/context/device_slot_context.h"
#include "machine/utils/dynamic/dev_encode_function_dupped_data.h"
#include "machine/utils/dynamic/dev_encode_function_stitch.h"
#include "machine/utils/dynamic/dev_encode_program.h"
#include "machine/utils/dynamic/dev_cell_match_mem_layout.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "interface/configs/config_manager_ng.h"
#include "interface/function/rebuildable_attribute.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/data_type.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

TEST(CellMatchDynamicFuncTest, NullAndEmptyInputs_NoCrash)
{
    std::vector<DevDynamicCellMatchStridePatch> patches;
    PatchHostDynamicCellMatchTableDesc(nullptr, patches);

    uint8_t buf[1024] = {0};
    auto* prog = reinterpret_cast<DevAscendProgram*>(buf);
    PatchHostDynamicCellMatchTableDesc(prog, patches);

    WriteDynamicCellMatchStridePatchesToLaunchArgs(nullptr, patches);

    DyndevFunctionAttribute dynAttr;
    ValidateDynamicCellMatchTableMemBudget(dynAttr, nullptr);

    std::unordered_map<std::string, ScalarImmediateType> symbolDict;
    std::vector<DeviceTensorData> inputs;
    std::vector<DeviceTensorData> outputs;
    Evaluator eval(symbolDict, &inputs, &outputs);

    RefillDynamicMemBudgets(nullptr, dynAttr, eval);

    auto evalPatches = PrepareDynamicCellMatchDescPatches(dynAttr, eval);
    EXPECT_TRUE(evalPatches.empty());

    auto launchPatches = PrepareHostDynamicCellMatchForLaunch(dynAttr, eval, nullptr);
    EXPECT_TRUE(launchPatches.empty());
}

TEST(CellMatchDynamicFuncTest, ResetRuntimeDynamicCellMatchPoolHost_HostPath)
{
    uint64_t table[4] = {0};
    ResetRuntimeDynamicCellMatchPoolHost(reinterpret_cast<uint64_t>(table), sizeof(table), false);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(table[i], AICORE_TASK_INIT);
    }
}

TEST(CellMatchDynamicFuncTest, ResetRuntimeDynamicCellMatchPoolHost_ZeroCapacity)
{
    uint64_t table[4] = {0};
    ResetRuntimeDynamicCellMatchPoolHost(reinterpret_cast<uint64_t>(table), 0, false);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(table[i], 0ULL);
    }
}

TEST(CellMatchDynamicFuncTest, ValidateDynamicCellMatchTableMemBudget_WithValidProg)
{
    uint8_t buf[2048] = {0};
    auto* prog = reinterpret_cast<DevAscendProgram*>(buf);
    prog->memBudget.metadata.dynamicCellMatchSlotNum = 1;

    DyndevFunctionAttribute dynAttr;
    DyndevFunctionAttribute::DynamicCellMatchLaunchMeta launchMeta;
    launchMeta.slotIndex = 0;
    launchMeta.descOffset = 1024;
    dynAttr.dynamicCellMatchLaunchMetaList.push_back(launchMeta);

    auto* desc = reinterpret_cast<DevCellMatchTableDesc*>(buf + 1024);
    desc->stride.dimStride[0] = 100;

    ValidateDynamicCellMatchTableMemBudget(dynAttr, prog);
}

TEST(CellMatchDynamicFuncTest, RefillDynamicMemBudgets_WithValidProg)
{
    uint8_t buf[2048] = {0};
    auto* prog = reinterpret_cast<DevAscendProgram*>(buf);
    prog->memBudget.metadata.dynamicCellMatchSlotNum = 10;

    DyndevFunctionAttribute dynAttr;
    dynAttr.maxDynamicAssembleOutcastMem = SymbolicScalar(100);
    dynAttr.maxDynamicCellMatchTableMem = SymbolicScalar(200);

    std::unordered_map<std::string, ScalarImmediateType> symbolDict;
    std::vector<DeviceTensorData> inputs;
    std::vector<DeviceTensorData> outputs;
    Evaluator eval(symbolDict, &inputs, &outputs);

    RefillDynamicMemBudgets(prog, dynAttr, eval);

    EXPECT_EQ(prog->memBudget.tensor.maxDynamicAssembleOutcastMem, 100ULL);
    EXPECT_EQ(prog->memBudget.metadata.maxDynamicCellMatchTableMem, 200ULL);
    EXPECT_EQ(prog->memBudget.metadata.dynamicCellMatch, 2000ULL);
}

TEST(CellMatchDynamicFuncTest, PrepareHostDynamicCellMatchForLaunch_WithValidProg)
{
    uint8_t buf[2048] = {0};
    auto* prog = reinterpret_cast<DevAscendProgram*>(buf);
    prog->memBudget.metadata.dynamicCellMatchSlotNum = 1;

    DyndevFunctionAttribute dynAttr;
    dynAttr.maxDynamicCellMatchTableMem = SymbolicScalar(100);

    std::unordered_map<std::string, ScalarImmediateType> symbolDict;
    std::vector<DeviceTensorData> inputs;
    std::vector<DeviceTensorData> outputs;
    Evaluator eval(symbolDict, &inputs, &outputs);

    auto patches = PrepareHostDynamicCellMatchForLaunch(dynAttr, eval, prog);
    EXPECT_TRUE(patches.empty());
}

TEST(CellMatchDynamicFuncTest, PrepareDynamicCellMatchDescPatches_WithValidLaunchMeta)
{
    DyndevFunctionAttribute dynAttr;
    DyndevFunctionAttribute::DynamicCellMatchLaunchMeta launchMeta;
    launchMeta.slotIndex = 0;
    launchMeta.descOffset = 1024;
    launchMeta.cellShape = {32, 32};

    std::vector<SymbolicScalar> candidate1 = {SymbolicScalar(64), SymbolicScalar(64)};
    std::vector<std::vector<SymbolicScalar>> candidates;
    candidates.push_back(candidate1);
    launchMeta.candidateRawDims = candidates;
    dynAttr.dynamicCellMatchLaunchMetaList.push_back(launchMeta);

    std::unordered_map<std::string, ScalarImmediateType> symbolDict;
    std::vector<DeviceTensorData> inputs;
    std::vector<DeviceTensorData> outputs;
    Evaluator eval(symbolDict, &inputs, &outputs);

    auto patches = PrepareDynamicCellMatchDescPatches(dynAttr, eval);
    EXPECT_EQ(patches.size(), 1u);
    EXPECT_EQ(patches[0].descOffset, 1024u);
}

TEST(CellMatchDynamicFuncTest, PrepareDynamicCellMatchDescPatches_WithInconsistentStrides)
{
    DyndevFunctionAttribute dynAttr;
    DyndevFunctionAttribute::DynamicCellMatchLaunchMeta launchMeta;
    launchMeta.slotIndex = 0;
    launchMeta.descOffset = 1024;
    launchMeta.cellShape = {32, 32};

    std::vector<SymbolicScalar> candidate1 = {SymbolicScalar(64), SymbolicScalar(64)};
    std::vector<SymbolicScalar> candidate2 = {SymbolicScalar(128), SymbolicScalar(64)};
    std::vector<std::vector<SymbolicScalar>> candidates;
    candidates.push_back(candidate1);
    candidates.push_back(candidate2);
    launchMeta.candidateRawDims = candidates;
    dynAttr.dynamicCellMatchLaunchMetaList.push_back(launchMeta);

    std::unordered_map<std::string, ScalarImmediateType> symbolDict;
    std::vector<DeviceTensorData> inputs;
    std::vector<DeviceTensorData> outputs;
    Evaluator eval(symbolDict, &inputs, &outputs);

    EXPECT_ANY_THROW(PrepareDynamicCellMatchDescPatches(dynAttr, eval));
}

TEST(CellMatchDynamicFuncTest, PatchRuntimeDynamicCellMatchMeta_AllBranches)
{
    AicoreModelMemoryUtils memUtils;
    DeviceKernelArgs nullKArgs{};
    PatchRuntimeDynamicCellMatchMeta(memUtils, nullptr, nullKArgs);

    uint8_t zeroBuf[sizeof(DevAscendProgram)] = {0};
    auto* zeroProg = reinterpret_cast<DevAscendProgram*>(zeroBuf);
    DeviceKernelArgs zeroKArgs{};
    zeroProg->memBudget.metadata.dynamicCellMatch = 0;
    PatchRuntimeDynamicCellMatchMeta(memUtils, zeroProg, zeroKArgs);
    EXPECT_EQ(zeroProg->devArgs.dynamicCellMatchAddr, 0ULL);
    EXPECT_EQ(zeroKArgs.runtimeDynamicCellMatchAddr, 0ULL);

    uint8_t nonZeroBuf[sizeof(DevAscendProgram)] = {0};
    auto* nonZeroProg = reinterpret_cast<DevAscendProgram*>(nonZeroBuf);
    DeviceKernelArgs nonZeroKArgs{};
    nonZeroProg->memBudget.metadata.dynamicCellMatch = 1024;
    PatchRuntimeDynamicCellMatchMeta(memUtils, nonZeroProg, nonZeroKArgs);
    EXPECT_NE(nonZeroProg->devArgs.dynamicCellMatchAddr, 0ULL);
    EXPECT_EQ(nonZeroProg->devArgs.dynamicCellMatchCapacity, 1024ULL);
    EXPECT_EQ(nonZeroKArgs.runtimeDynamicCellMatchAddr, nonZeroProg->devArgs.dynamicCellMatchAddr);
    EXPECT_EQ(nonZeroKArgs.runtimeDynamicCellMatchCapacity, 1024ULL);
}

namespace {

struct MiniDup {
    static constexpr size_t kBuf = 4096;
    std::unique_ptr<uint8_t[]> funcBuf;
    std::unique_ptr<uint8_t[]> dupBuf;
    DevAscendFunction* func{nullptr};
    DevAscendFunctionDuppedData* dupData{nullptr};
    DevAscendFunctionDupped dup;
    size_t cursor{0};

    void Align(size_t a) { cursor = (cursor + a - 1) & ~(a - 1); }

    template <typename T>
    T* Alloc(size_t n = 1)
    {
        Align(alignof(T));
        auto* p = reinterpret_cast<T*>(funcBuf.get() + cursor);
        cursor += sizeof(T) * n;
        EXPECT_LT(cursor, kBuf);
        return p;
    }

    void Build(bool withOutcast)
    {
        funcBuf = std::make_unique<uint8_t[]>(kBuf);
        memset(funcBuf.get(), 0, kBuf);
        cursor = sizeof(DevAscendFunction);
        func = reinterpret_cast<DevAscendFunction*>(funcBuf.get());
        func->funcKey = 1;

        func->operationList_.AssignOffsetSize(cursor, 1);
        auto* op = Alloc<DevAscendOperation>();
        new (op) DevAscendOperation();
        op->stitchIndex = 0;
        op->depGraphPredCount = 0;
        op->depGraphSuccList.AssignOffsetSize(0, 0);

        if (withOutcast) {
            Align(alignof(DevAscendFunctionOutcast));
            func->outcastList.AssignOffsetSize(cursor, 1);
            auto* oc = Alloc<DevAscendFunctionOutcast>();
            new (oc) DevAscendFunctionOutcast();
            oc->stitchPolicyFullCoverProducerHubOpIdx = -1;
            oc->producerConsumerList.AssignOffsetSize(0, 0);
            oc->stitchPolicyFullCoverProducerList.AssignOffsetSize(0, 0);
            oc->toSlotList.AssignOffsetSize(0, 0);
            oc->cellMatchRuntimeFullUpdateTable.AssignOffsetSize(0, 0);
        }

        dupBuf = std::make_unique<uint8_t[]>(kBuf);
        memset(dupBuf.get(), 0, kBuf);
        dupData = reinterpret_cast<DevAscendFunctionDuppedData*>(dupBuf.get());
        uint8_t* p = dupBuf.get() + sizeof(DevAscendFunctionDuppedData);
        dupData->source_ = func;
        dupData->operationList_.size = 1;
        dupData->operationList_.predCountBase = static_cast<uint32_t>(p - dupBuf.get());
        *reinterpret_cast<predcount_t*>(p) = 0;
        p += sizeof(predcount_t);
        dupData->operationList_.stitchBase = static_cast<uint32_t>(p - dupBuf.get());
        dupData->operationList_.stitchCount = 1;
        new (p) DevAscendFunctionDuppedStitchList();
        p += sizeof(DevAscendFunctionDuppedStitchList);
        dupData->expressionList_.base = static_cast<uint32_t>(p - dupBuf.get());
        dupData->expressionList_.size = 1;
        *reinterpret_cast<uint64_t*>(p) = 0;

        WsAllocation a;
        a.ptr = reinterpret_cast<uint64_t>(dupData);
        dup = DevAscendFunctionDupped(a);
    }
};

void BindStitchedList(DeviceStitchContext& ctx, DevAscendFunctionDupped* storage, uint32_t n)
{
    ctx.stitchedList_.dataAllocation_.ptr = reinterpret_cast<uint64_t>(storage);
    ctx.stitchedList_.size_ = n;
    ctx.stitchedList_.capacity_ = n;
}

void UnbindStitchedList(DeviceStitchContext& ctx)
{
    ctx.stitchedList_.dataAllocation_.Invalidate();
    ctx.stitchedList_.size_ = 0;
    ctx.stitchedList_.capacity_ = 0;
}

} // namespace

// FullCoverUpdateStitch → FullCoverPartialProdToFullConsStitch early return (empty lists).
TEST(FullCoverUpdateStitchTest, EmptyLists_EarlyReturn)
{
    MiniDup prev;
    MiniDup next;
    prev.Build(true);
    next.Build(false);

    DevAscendFunctionDupped storage[1] = {prev.dup};
    DeviceStitchContext stitchCtx;
    BindStitchedList(stitchCtx, storage, 1);

    DeviceExecuteSlot slot{};
    slot.stitchDupIdx = 0;
    slot.stitchOutcastIdx = 0;

    DevAscendFunctionIncast incast{};
    incast.consumerList.AssignOffsetSize(0, 0);
    incast.stitchPolicyFullCoverConsumerList.AssignOffsetSize(0, 0);
    incast.stitchPolicyFullCoverConsumerAllOpIdxList.AssignOffsetSize(0, 0);

    EXPECT_EQ(stitchCtx.FullCoverUpdateStitch(next.dup, 0, 1, slot, 0, incast), 0u);
    UnbindStitchedList(stitchCtx);
}

// FullCoverPartialProdToFullConsStitch loop: READ producer is skipped (no HandleOneStitch).
TEST(FullCoverUpdateStitchTest, ReadProducerSkipped)
{
    MiniDup prev;
    MiniDup next;
    prev.Build(true);
    next.Build(false);

    auto& outcast = prev.func->GetOutcast(0);
    auto* prod = prev.Alloc<DevAscendFunctionCallOperandUse>();
    new (prod) DevAscendFunctionCallOperandUse(0, -1, CellMatchOpType::READ);
    outcast.producerConsumerList.AssignOffsetSize(reinterpret_cast<uint8_t*>(prod) - prev.funcBuf.get(), 1);

    auto* cons = next.Alloc<DevAscendFunctionCallOperandUse>();
    new (cons) DevAscendFunctionCallOperandUse(0, -1, CellMatchOpType::READ);

    DevAscendFunctionIncast incast{};
    incast.stitchPolicyFullCoverConsumerList.AssignOffsetSize(reinterpret_cast<uint8_t*>(cons) - next.funcBuf.get(), 1);
    incast.consumerList.AssignOffsetSize(0, 0);
    incast.stitchPolicyFullCoverConsumerAllOpIdxList.AssignOffsetSize(0, 0);

    DevAscendFunctionDupped storage[1] = {prev.dup};
    DeviceStitchContext stitchCtx;
    BindStitchedList(stitchCtx, storage, 1);

    DeviceExecuteSlot slot{};
    slot.stitchDupIdx = 0;
    slot.stitchOutcastIdx = 0;

    EXPECT_EQ(stitchCtx.FullCoverUpdateStitch(next.dup, 0, 1, slot, 0, incast), 0u);
    UnbindStitchedList(stitchCtx);
}

// Covers UpdateSlotsForIncastStitch dual Fill (consumerList + stitchPolicyFullCoverConsumerList).
TEST(UpdateSlotsIncastFillTest, DualConsumerLists)
{
    MiniDup root;
    root.Build(false);

    root.Align(alignof(DevAscendFunctionIncast));
    root.func->incastList.AssignOffsetSize(root.cursor, 1);
    auto* ic = root.Alloc<DevAscendFunctionIncast>();
    new (ic) DevAscendFunctionIncast();

    auto* slotIdx = root.Alloc<int>();
    *slotIdx = 0;
    ic->fromSlotList.AssignOffsetSize(reinterpret_cast<uint8_t*>(slotIdx) - root.funcBuf.get(), 1);

    auto* partialConsumer = root.Alloc<DevAscendFunctionCallOperandUse>();
    new (partialConsumer) DevAscendFunctionCallOperandUse(0, -1, CellMatchOpType::READ);
    ic->consumerList.AssignOffsetSize(reinterpret_cast<uint8_t*>(partialConsumer) - root.funcBuf.get(), 1);

    auto* full = root.Alloc<DevAscendFunctionCallOperandUse>();
    new (full) DevAscendFunctionCallOperandUse(0, -1, CellMatchOpType::READ);
    ic->stitchPolicyFullCoverConsumerList.AssignOffsetSize(reinterpret_cast<uint8_t*>(full) - root.funcBuf.get(), 1);
    ic->stitchPolicyFullCoverConsumerAllOpIdxList.AssignOffsetSize(0, 0);

    DevAscendProgramUpdate partial{};
    partial.slotIndex = 0;
    partial.cellMatchTableDesc.SetCacheOpMaxCount({1, 0, 1});
    uint64_t table[4] = {0};
    partial.cellMatchRuntimePartialUpdateTable = DevRelocVector<uint64_t>(4, table);

    DeviceExecuteSlot slots[1]{};
    slots[0].isPartialUpdateStitch = true;
    partial.stitchCtrlBitMask = STITCH_CTRL_RAW;
    slots[0].partialUpdate = &partial;

    DeviceSlotContext slotCtx;
    slotCtx.slotList_.dataAllocation_.ptr = reinterpret_cast<uint64_t>(slots);
    slotCtx.slotList_.size_ = 1;
    slotCtx.slotList_.capacity_ = 1;
    slotCtx.workspace_ = nullptr;

    EXPECT_EQ(slotCtx.UpdateSlots(root.dup, 0, 0, 0), 0u);

    slotCtx.slotList_.dataAllocation_.Invalidate();
    slotCtx.slotList_.size_ = 0;
    slotCtx.slotList_.capacity_ = 0;
}

namespace {

constexpr int kAssembleMaskTile = 32;
constexpr int kAssembleMaskIters = 4;

void SetupAssembleBitmaskEncodeTest()
{
    Program::GetInstance().Reset();
    config::Reset();
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    config::SetRuntimeOption(STITCH_FUNCTION_MAX_NUM, 64);
    TileShape::Current().SetVecTile(kAssembleMaskTile, kAssembleMaskTile);
    TileShape::Current().SetCubeTile({kAssembleMaskTile, kAssembleMaskTile}, {kAssembleMaskTile, kAssembleMaskTile},
                                     {kAssembleMaskTile, kAssembleMaskTile});
}

DevAscendProgram* EncodeAndGetDevProg()
{
    Function* func = Program::GetInstance().GetLastFunction();
    EXPECT_NE(func, nullptr);
    if (func == nullptr) {
        return nullptr;
    }
    auto dynAttr = func->GetDyndevAttribute();
    EXPECT_NE(dynAttr, nullptr);
    if (dynAttr == nullptr) {
        return nullptr;
    }
    EXPECT_FALSE(dynAttr->devProgBinary.empty());
    if (dynAttr->devProgBinary.empty()) {
        return nullptr;
    }
    auto* devProg = reinterpret_cast<DevAscendProgram*>(dynAttr->devProgBinary.data());
    EXPECT_NE(devProg, nullptr);
    if (devProg == nullptr) {
        return nullptr;
    }
    // After encode, RelocProgram(this, 0) stores relative offsets; restore before At()/list access.
    devProg->RelocProgram(0, reinterpret_cast<uint64_t>(devProg));
    return devProg;
}

const DevAscendProgramUpdate* FindPartialUpdateBySlot(DevAscendProgram* devProg, int slotIndex)
{
    if (devProg == nullptr || slotIndex < 0) {
        return nullptr;
    }
    if (static_cast<size_t>(slotIndex) >= devProg->updateList.size()) {
        return nullptr;
    }
    const auto& partial = devProg->At(devProg->updateList, slotIndex);
    if (partial.slotIndex < 0) {
        return nullptr;
    }
    return &partial;
}

StitchCtrlBitMask GetOutAssembleSlotBitmask(DevAscendProgram* devProg)
{
    if (devProg == nullptr) {
        return STITCH_CTRL_NONE;
    }
    const std::vector<int> outputSlots = devProg->GetOutputTensorSlotIndexList();
    EXPECT_FALSE(outputSlots.empty());
    if (outputSlots.empty()) {
        return STITCH_CTRL_NONE;
    }
    const auto* partial = FindPartialUpdateBySlot(devProg, outputSlots[0]);
    EXPECT_NE(partial, nullptr) << "out should be a partial-update slot, slot=" << outputSlots[0];
    if (partial == nullptr) {
        return STITCH_CTRL_NONE;
    }
    return partial->stitchCtrlBitMask;
}

} // namespace

// WAW: two loops Assemble same out; both Assemble(..., true) → NONE
TEST(StitchCtrlBitMaskEncodeTest, Waw_ParallelTrue_None)
{
    SetupAssembleBitmaskEncodeTest();

    Tensor t0(DT_FP32, {kAssembleMaskTile, kAssembleMaskTile}, "t0");
    Tensor t1(DT_FP32, {kAssembleMaskTile, kAssembleMaskTile}, "t1");
    Tensor out(DT_FP32, {kAssembleMaskIters * kAssembleMaskTile, kAssembleMaskTile}, "out");
    FUNCTION("assemble_waw_parallel_true", {t0, t1}, {out})
    {
        LOOP("loop_waw_0", FunctionType::DYNAMIC_LOOP, i, LoopRange(kAssembleMaskIters))
        {
            auto temp = Add(t0, t1);
            Assemble(temp, {i * kAssembleMaskTile, 0}, out, true);
        }
        LOOP("loop_waw_1", FunctionType::DYNAMIC_LOOP, j, LoopRange(kAssembleMaskIters))
        {
            auto temp = Add(t0, t1);
            Assemble(temp, {j * kAssembleMaskTile, 0}, out, true);
        }
    }

    DevAscendProgram* devProg = EncodeAndGetDevProg();
    ASSERT_NE(devProg, nullptr);
    EXPECT_EQ(GetOutAssembleSlotBitmask(devProg), STITCH_CTRL_NONE);
}

// WAW: two loops Assemble same out; both Assemble unmarked → WAW
TEST(StitchCtrlBitMaskEncodeTest, Waw_ParallelFalse_HasWaw)
{
    SetupAssembleBitmaskEncodeTest();

    Tensor t0(DT_FP32, {kAssembleMaskTile, kAssembleMaskTile}, "t0");
    Tensor t1(DT_FP32, {kAssembleMaskTile, kAssembleMaskTile}, "t1");
    Tensor out(DT_FP32, {kAssembleMaskIters * kAssembleMaskTile, kAssembleMaskTile}, "out");
    FUNCTION("assemble_waw_unmarked", {t0, t1}, {out})
    {
        LOOP("loop_waw_0", FunctionType::DYNAMIC_LOOP, i, LoopRange(kAssembleMaskIters))
        {
            auto temp = Add(t0, t1);
            Assemble(temp, {i * kAssembleMaskTile, 0}, out);
        }
        LOOP("loop_waw_1", FunctionType::DYNAMIC_LOOP, j, LoopRange(kAssembleMaskIters))
        {
            auto temp = Add(t0, t1);
            Assemble(temp, {j * kAssembleMaskTile, 0}, out);
        }
    }

    DevAscendProgram* devProg = EncodeAndGetDevProg();
    ASSERT_NE(devProg, nullptr);
    EXPECT_NE(GetOutAssembleSlotBitmask(devProg) & STITCH_CTRL_WAW, 0u);
}

// Encode-side: FUNCTION path does not run InferMultiIterOverlap.
// Hand-Mark simulates PASS → stitchCtrlBitMask must drop WAW.

TEST(StitchCtrlBitMaskEncodeTest, MultiIterDisjoint_RootKeepsNoOverlapMark)
{
    SetupAssembleBitmaskEncodeTest();

    Tensor t0(DT_FP32, {kAssembleMaskTile, kAssembleMaskTile}, "t0");
    Tensor t1(DT_FP32, {kAssembleMaskTile, kAssembleMaskTile}, "t1");
    Tensor out(DT_FP32, {kAssembleMaskIters * kAssembleMaskTile, kAssembleMaskTile}, "out");
    FUNCTION("assemble_multi_iter_disjoint", {t0, t1}, {out})
    {
        LOOP("loop_disjoint", FunctionType::DYNAMIC_LOOP, i, LoopRange(kAssembleMaskIters))
        {
            auto temp = Add(t0, t1);
            Assemble(temp, {i * kAssembleMaskTile, 0}, out);
        }
        for (const auto& [name, func] : Program::GetInstance().GetFunctionMap()) {
            (void)name;
            for (const auto& oc : func->GetOutcast()) {
                RebuildableAttributeManager::GetInstance()
                    .GetAttr<RebuildableMultiIterNoOverlap>(func.get())
                    ->Mark(oc->GetRawMagic());
            }
        }
    }

    auto dynAttr = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    EXPECT_EQ(GetOutAssembleSlotBitmask(EncodeAndGetDevProg()) & STITCH_CTRL_WAW, 0u);

    bool anyRootMarked = false;
    for (Function* root : dynAttr->funcGroup.devRootList) {
        auto* attr = RebuildableAttributeManager::GetInstance().GetAttr<RebuildableMultiIterNoOverlap>(root);
        for (const auto& oc : root->GetOutcast()) {
            if (attr->Has(oc->GetRawMagic())) {
                anyRootMarked = true;
                break;
            }
        }
        if (anyRootMarked) {
            break;
        }
    }
    EXPECT_TRUE(anyRootMarked);
}

// WAR: Assemble tmp→mid, then View(mid)→Assemble→out → stitchCtrlBitMask has WAR|RAW
TEST(StitchCtrlBitMaskEncodeTest, War_ParallelFalse_HasWarRaw)
{
    SetupAssembleBitmaskEncodeTest();

    Tensor t0(DT_FP32, {kAssembleMaskTile, kAssembleMaskTile}, "t0");
    Tensor t1(DT_FP32, {kAssembleMaskTile, kAssembleMaskTile}, "t1");
    Tensor out(DT_FP32, {kAssembleMaskIters * kAssembleMaskTile, kAssembleMaskTile}, "out");
    FUNCTION("assemble_war_tmp_mid_out", {t0, t1}, {out})
    {
        Tensor mid(DT_FP32, {kAssembleMaskIters * kAssembleMaskTile, kAssembleMaskTile}, "mid");
        LOOP("loop_tmp_to_mid", FunctionType::DYNAMIC_LOOP, i, LoopRange(kAssembleMaskIters))
        {
            auto tmp = Add(t0, t1);
            Assemble(tmp, {i * kAssembleMaskTile, 0}, mid);
        }
        LOOP("loop_mid_to_out", FunctionType::DYNAMIC_LOOP, j, LoopRange(kAssembleMaskIters))
        {
            auto midTile = View(mid, {kAssembleMaskTile, kAssembleMaskTile}, {j * kAssembleMaskTile, 0});
            Assemble(midTile, {j * kAssembleMaskTile, 0}, out);
        }
    }

    DevAscendProgram* devProg = EncodeAndGetDevProg();
    ASSERT_NE(devProg, nullptr);

    bool sawWarRaw = false;
    StitchCtrlBitMask midMask = STITCH_CTRL_NONE;
    for (size_t i = 0; i < devProg->updateList.size(); ++i) {
        const auto& partial = devProg->At(devProg->updateList, i);
        if (partial.slotIndex < 0) {
            continue;
        }
        const StitchCtrlBitMask mask = partial.stitchCtrlBitMask;
        if ((mask & STITCH_CTRL_WAR) != 0 && (mask & STITCH_CTRL_RAW) != 0) {
            sawWarRaw = true;
            midMask = mask;
            break;
        }
    }
    EXPECT_TRUE(sawWarRaw) << "WAR (tmp→mid then mid→out) mid slot should have WAR|RAW, lastSeen=0x" << std::hex
                           << static_cast<unsigned>(midMask);
}

TEST(PrepareRuntimeDynamicPartialUpdateTableTest, ZeroStrideAndZeroCapacityCoverAssertBranches)
{
    MiniDup root;
    root.Build(true);

    root.Align(alignof(DevAscendFunctionOutcast));
    auto* oc = reinterpret_cast<DevAscendFunctionOutcast*>(root.funcBuf.get() + root.cursor);
    auto* slotIdxVal = root.Alloc<int>();
    *slotIdxVal = 0;
    oc->toSlotList.AssignOffsetSize(reinterpret_cast<uint8_t*>(slotIdxVal) - root.funcBuf.get(), 1);
    oc->producerConsumerList.AssignOffsetSize(0, 0);
    oc->stitchPolicyFullCoverProducerList.AssignOffsetSize(0, 0);
    oc->stitchPolicyFullCoverProducerHubOpIdx = -1;
    oc->cellMatchRuntimeFullUpdateTable.AssignOffsetSize(0, 0);
    root.func->outcastList.AssignOffsetSize(root.cursor, 1);

    uint64_t table[4] = {0};
    DevAscendProgramUpdate partial{};
    partial.slotIndex = 0;
    partial.isPartial = true;
    partial.cellMatchTableDesc.SetCellShape({4});
    partial.cellMatchTableDesc.SetStrideShape({0});
    partial.cellMatchTableDesc.cellUint64Size = 2;
    partial.cellMatchRuntimePartialUpdateTable = DevRelocVector<uint64_t>(4, table);
    partial.stitchCtrlBitMask = STITCH_CTRL_WAW;

    DeviceExecuteSlot slots[1]{};
    slots[0].isPartialUpdateStitch = true;
    slots[0].stitchCtrlBitMask = STITCH_CTRL_WAW;
    slots[0].partialUpdate = &partial;
    slots[0].stitchDupIdx = 0;
    slots[0].stitchOutcastIdx = 0;

    uint8_t progBuf[sizeof(DevAscendProgram)] = {0};
    auto* prog = reinterpret_cast<DevAscendProgram*>(progBuf);
    prog->memBudget.metadata.maxDynamicCellMatchTableMem = 0;
    DeviceWorkspaceAllocator ws(prog);

    DeviceSlotContext slotCtx;
    slotCtx.slotList_.dataAllocation_.ptr = reinterpret_cast<uint64_t>(slots);
    slotCtx.slotList_.size_ = 1;
    slotCtx.slotList_.capacity_ = 1;
    slotCtx.workspace_ = &ws;

    try {
        (void)slotCtx.UpdateSlots(root.dup, 0, 0, 0);
    } catch (const std::exception&) {
    }

    slotCtx.slotList_.dataAllocation_.Invalidate();
    slotCtx.slotList_.size_ = 0;
    slotCtx.slotList_.capacity_ = 0;
}
