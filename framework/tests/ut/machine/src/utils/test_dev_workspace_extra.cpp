/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <gtest/gtest.h>
#include <cstring>
#include <memory>
#include <vector>

#include "interface/machine/device/tilefwk/aikernel_device_task.h"

#define private public
#include "machine/utils/dynamic/dev_workspace.h"
#include "machine/utils/dynamic/dev_encode_workspace.h"
#undef private

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class DevWorkspaceTest : public testing::Test {};

TEST_F(DevWorkspaceTest, Constructor_Default)
{
    DeviceWorkspaceAllocator allocator;
    EXPECT_EQ(allocator.devProg_, nullptr);
}

TEST_F(DevWorkspaceTest, Constructor_WithDevProg)
{
    DevAscendProgram prog{};
    DeviceWorkspaceAllocator allocator(&prog);
    EXPECT_EQ(allocator.devProg_, &prog);
}

TEST_F(DevWorkspaceTest, SwitchWParallelWorkSpace_SetsId)
{
    DeviceWorkspaceAllocator allocator;
    allocator.SwitchWParallelWorkSpace(5);
    EXPECT_EQ(allocator.curParallelWsId, 5u);
}

TEST_F(DevWorkspaceTest, StackWorkspaceAddr_ReturnsBase)
{
    DeviceWorkspaceAllocator allocator;
    allocator.stackWorkspaceBase_ = 0x12345678;
    EXPECT_EQ(allocator.StackWorkspaceAddr(), 0x12345678u);
}

TEST_F(DevWorkspaceTest, StandardStackWorkspacePerCore_ReturnsValue)
{
    DeviceWorkspaceAllocator allocator;
    allocator.standardStackWorkspacePerCore_ = 4096;
    EXPECT_EQ(allocator.StandardStackWorkspacePerCore(), 4096u);
}

TEST_F(DevWorkspaceTest, StitchCacheAddr_ReturnsPointer)
{
    DeviceWorkspaceAllocator allocator;
    allocator.stitchCacheAddr_ = 0xABCD0000;
    EXPECT_EQ(allocator.StitchCacheAddr(), reinterpret_cast<uint64_t*>(0xABCD0000));
}

TEST_F(DevWorkspaceTest, RootFuncMaxCallOpsize_ReturnsValue)
{
    DeviceWorkspaceAllocator allocator;
    allocator.rootFuncMaxCallOpsize_ = 100;
    EXPECT_EQ(allocator.RootFuncMaxCallOpsize(), 100u);
}

TEST_F(DevWorkspaceTest, StitchCacheEpoch_ReturnsLow16Bits)
{
    DevAscendProgram prog{};
    prog.stitchCacheEpoch_ = 0x12345678;
    DeviceWorkspaceAllocator allocator(&prog);
    EXPECT_EQ(allocator.StitchCacheEpoch(), 0x5678u);
}

TEST_F(DevWorkspaceTest, AdvanceStitchCacheEpoch_IncrementsEpoch)
{
    DevAscendProgram prog{};
    prog.stitchCacheEpoch_ = 100;
    prog.memBudget.metadata.stitchCacheSize = 0;
    DeviceWorkspaceAllocator allocator(&prog);
    allocator.stitchCacheAddr_ = 0;
    allocator.AdvanceStitchCacheEpoch();
    EXPECT_EQ(prog.stitchCacheEpoch_, 101u);
}

TEST_F(DevWorkspaceTest, AdvanceStitchCacheEpoch_SkipsZeroLowBits)
{
    DevAscendProgram prog{};
    prog.stitchCacheEpoch_ = 0xFFFF;
    prog.memBudget.metadata.stitchCacheSize = 0;
    DeviceWorkspaceAllocator allocator(&prog);
    allocator.stitchCacheAddr_ = 0;
    allocator.AdvanceStitchCacheEpoch();
    EXPECT_EQ(prog.stitchCacheEpoch_, 0x10001u);
}

TEST_F(DevWorkspaceTest, TENSOR_ADDR_ALIGNMENT_CorrectValue) { EXPECT_EQ(TENSOR_ADDR_ALIGNMENT, 512); }

TEST_F(DevWorkspaceTest, SUBMMIT_TASK_QUE_SIZE_CorrectValue) { EXPECT_EQ(SUBMMIT_TASK_QUE_SIZE, 512u); }

TEST_F(DevWorkspaceTest, ALLOC_NUM_ONE_SLAB_CorrectValue) { EXPECT_EQ(ALLOC_NUM_ONE_SLAB, 4); }

TEST_F(DevWorkspaceTest, CalculateVectorCapacity_Zero)
{
    DeviceWorkspaceAllocator allocator;
    EXPECT_EQ(allocator.CalculateVectorCapacity(0), 0u);
}

TEST_F(DevWorkspaceTest, CalculateVectorCapacity_SmallValues)
{
    DeviceWorkspaceAllocator allocator;
    EXPECT_EQ(allocator.CalculateVectorCapacity(1), 8u);
    EXPECT_EQ(allocator.CalculateVectorCapacity(5), 8u);
    EXPECT_EQ(allocator.CalculateVectorCapacity(8), 8u);
}

TEST_F(DevWorkspaceTest, CalculateVectorCapacity_PowerOfTwo)
{
    DeviceWorkspaceAllocator allocator;
    EXPECT_EQ(allocator.CalculateVectorCapacity(9), 16u);
    EXPECT_EQ(allocator.CalculateVectorCapacity(16), 16u);
    EXPECT_EQ(allocator.CalculateVectorCapacity(17), 32u);
    EXPECT_EQ(allocator.CalculateVectorCapacity(64), 64u);
    EXPECT_EQ(allocator.CalculateVectorCapacity(65), 128u);
}

TEST_F(DevWorkspaceTest, CalculateVectorCapacity_LargeValues)
{
    DeviceWorkspaceAllocator allocator;
    EXPECT_EQ(allocator.CalculateVectorCapacity(1000), 1024u);
    EXPECT_EQ(allocator.CalculateVectorCapacity(1024), 1024u);
    EXPECT_EQ(allocator.CalculateVectorCapacity(1025), 2048u);
}

TEST_F(DevWorkspaceTest, DeviceTaskMemTryRecycle_EmptyQueue)
{
    DeviceWorkspaceAllocator allocator;
    bool result = allocator.DeviceTaskMemTryRecycle();
    EXPECT_FALSE(result);
}

namespace {
void SetupDrcoTaskForRecycle(DeviceWorkspaceAllocator& allocator, DynDeviceTask& dyntask,
                             npu::tile_fwk::DrcoRootFuncList& rootFuncList, uint32_t totalTaskCount,
                             uint32_t devTaskFinished)
{
    dyntask.taskStageAllocMem.generalMetadataStageMem = {};
    dyntask.taskStageAllocMem.stitchStageMem = {};
    dyntask.dynFuncDataList = reinterpret_cast<npu::tile_fwk::DynFuncHeader*>(0x1);
    rootFuncList.totalTaskCount = totalTaskCount;
    rootFuncList.devTaskFinished = devTaskFinished;
    dyntask.drcoRootFuncList = &rootFuncList;
    allocator.submmitTaskQueue_.Enqueue(&dyntask);
}
} // namespace

TEST_F(DevWorkspaceTest, DeviceTaskMemTryRecycle_DevTaskFinished_SetsCanFree)
{
    DeviceWorkspaceAllocator allocator;
    auto dyntask = std::make_unique<DynDeviceTask>(allocator);
    npu::tile_fwk::DrcoRootFuncList rootFuncList{};
    SetupDrcoTaskForRecycle(allocator, *dyntask, rootFuncList, 16, 1);

    EXPECT_TRUE(allocator.DeviceTaskMemTryRecycle());
    EXPECT_TRUE(dyntask->taskStageAllocMem.canFree.load());
}

TEST_F(DevWorkspaceTest, DeviceTaskMemTryRecycle_TotalTaskCountZero_SetsCanFree)
{
    DeviceWorkspaceAllocator allocator;
    auto dyntask = std::make_unique<DynDeviceTask>(allocator);
    npu::tile_fwk::DrcoRootFuncList rootFuncList{};
    SetupDrcoTaskForRecycle(allocator, *dyntask, rootFuncList, 0, 0);

    EXPECT_TRUE(allocator.DeviceTaskMemTryRecycle());
    EXPECT_TRUE(dyntask->taskStageAllocMem.canFree.load());
}

TEST_F(DevWorkspaceTest, DeviceTaskMemTryRecycle_NotFinished_KeepsCanFreeFalse)
{
    DeviceWorkspaceAllocator allocator;
    auto dyntask = std::make_unique<DynDeviceTask>(allocator);
    npu::tile_fwk::DrcoRootFuncList rootFuncList{};
    SetupDrcoTaskForRecycle(allocator, *dyntask, rootFuncList, 16, 0);

    EXPECT_FALSE(allocator.DeviceTaskMemTryRecycle());
    EXPECT_FALSE(dyntask->taskStageAllocMem.canFree.load());
}

TEST_F(DevWorkspaceTest, DynDevTaskSlabMemObjSize_ReturnsSizeof)
{
    DeviceWorkspaceAllocator allocator;
    EXPECT_EQ(allocator.DynDevTaskSlabMemObjSize(), sizeof(struct DynDeviceTask));
}

TEST_F(DevWorkspaceTest, DuppedStitchSlabMemObjSize_ReturnsSizeof)
{
    DeviceWorkspaceAllocator allocator;
    EXPECT_EQ(allocator.DuppedStitchSlabMemObjSize(), sizeof(DevAscendFunctionDuppedStitch));
}

TEST_F(DevWorkspaceTest, ShmemWaitUntilCacheSlabMemObjSize_NoAicpuTask)
{
    DevAscendProgram prog{};
    prog.devArgs.hasAicpuTask = false;
    DeviceWorkspaceAllocator allocator(&prog);
    EXPECT_EQ(allocator.ShmemWaitUntilCacheSlabMemObjSize(), 0u);
}

TEST_F(DevWorkspaceTest, ShmemWaitUntilCacheSlabMemObjSize_WithAicpuTask)
{
    DevAscendProgram prog{};
    prog.devArgs.hasAicpuTask = true;
    DeviceWorkspaceAllocator allocator(&prog);
    EXPECT_GT(allocator.ShmemWaitUntilCacheSlabMemObjSize(), 0u);
}

TEST_F(DevWorkspaceTest, WrapQueSlabMemObjSize_NonDAV3510)
{
    DevAscendProgram prog{};
    prog.devArgs.archInfo = ArchInfo::DAV_2201;
    prog.stitchFunctionsize = 10;
    DeviceWorkspaceAllocator allocator(&prog);
    EXPECT_EQ(allocator.WrapQueSlabMemObjSize(), 0u);
}

TEST_F(DevWorkspaceTest, DieReadyQueSlabMemObjSize_NonDAV3510)
{
    DevAscendProgram prog{};
    prog.devArgs.archInfo = ArchInfo::DAV_2201;
    prog.stitchFunctionsize = 10;
    DeviceWorkspaceAllocator allocator(&prog);
    EXPECT_EQ(allocator.DieReadyQueSlabMemObjSize(), 0u);
}

TEST_F(DevWorkspaceTest, ReadyQueSlabMemObjSize_ReturnsValue)
{
    DevAscendProgram prog{};
    prog.stitchFunctionsize = 10;
    DeviceWorkspaceAllocator allocator(&prog);
    uint32_t size = allocator.ReadyQueSlabMemObjSize();
    EXPECT_GT(size, 0u);
}

TEST_F(DevWorkspaceTest, DynFuncDataSlabMemObjSize_ReturnsValue)
{
    DeviceWorkspaceAllocator allocator;
    uint32_t size = allocator.DynFuncDataSlabMemObjSize();
    EXPECT_GT(size, 0u);
}

TEST_F(DevWorkspaceTest, VecStitchListSLabMemObjSize_ReturnsValue)
{
    DeviceWorkspaceAllocator allocator;
    uint32_t size = allocator.VecStitchListSLabMemObjSize();
    EXPECT_GT(size, 0u);
}

TEST_F(DevWorkspaceTest, RecycleDevFuncWorkspace_NoOpWithoutInit)
{
    DeviceWorkspaceAllocator allocator;
    allocator.curParallelWsId = 0;
    allocator.RecycleDevFuncWorkspace();
    SUCCEED();
}

TEST_F(DevWorkspaceTest, VerifyStitchedListMemory_EmptyList)
{
    DeviceWorkspaceAllocator allocator;
    DevStartArgs args{};
    allocator.VerifyStitchedListMemory(args, nullptr, 0);
    SUCCEED();
}

TEST_F(DevWorkspaceTest, DumpMemoryUsage_NoOpWithoutInit)
{
    DeviceWorkspaceAllocator allocator;
    allocator.DumpMemoryUsage("test_hint");
    SUCCEED();
}

TEST_F(DevWorkspaceTest, ResetAicpuMemCounter_NoOpWithoutInit)
{
    DeviceWorkspaceAllocator allocator;
    allocator.ResetAicpuMemCounter();
    SUCCEED();
}

TEST_F(DevWorkspaceTest, RewindMemoryDumper_NoOpWithoutInit)
{
    DeviceWorkspaceAllocator allocator;
    allocator.RewindMemoryDumper();
    SUCCEED();
}

TEST_F(DevWorkspaceTest, MarkAsNewStitchWindow_NoOpWithoutInit)
{
    DeviceWorkspaceAllocator allocator;
    allocator.MarkAsNewStitchWindow();
    SUCCEED();
}

TEST_F(DevWorkspaceTest, CalcSlabMemObjmaxSize_ReturnsValue)
{
    DevAscendProgram prog{};
    prog.stitchFunctionsize = 10;
    DeviceWorkspaceAllocator allocator(&prog);
    uint32_t size = allocator.CalcSlabMemObjmaxSize();
    EXPECT_GT(size, 0u);
}

TEST_F(DevWorkspaceTest, CalcStitchSlabMemObjmaxSize_ReturnsValue)
{
    DevAscendProgram prog{};
    prog.stitchFunctionsize = 10;
    DeviceWorkspaceAllocator allocator(&prog);
    uint32_t slabCapacity[16] = {0};
    uint32_t size = allocator.CalcStitchSlabMemObjmaxSize(slabCapacity);
    EXPECT_GT(size, 0u);
}

TEST_F(DevWorkspaceTest, CalcStitchWorkspace_AicpuResolve_OnlySharedTypes)
{
    DevAscendProgram prog{};
    prog.stitchFunctionsize = 100;
    uint64_t size = CalcStitchWorkspace(prog, false);
    EXPECT_GT(size, 0u);
}

TEST_F(DevWorkspaceTest, CalcStitchWorkspace_AicoreResolve_LargerThanAicpu)
{
    DevAscendProgram prog{};
    prog.stitchFunctionsize = 100;
    uint64_t sizeAicpu = CalcStitchWorkspace(prog, false);
    uint64_t sizeAicore = CalcStitchWorkspace(prog, true);
    // DRCO 场景额外预留 PER_CORE/LOCAL_READY_QUE/LOCAL_READY_MATRIX/PRED_COUNT 池内存
    EXPECT_GT(sizeAicore, sizeAicpu);
}

TEST_F(DevWorkspaceTest, CalculateSlabCapacityPerType_NullSlabCapacity)
{
    DevAscendProgram prog{};
    DeviceWorkspaceAllocator allocator(&prog);
    allocator.CalculateSlabCapacityPerType(4096, nullptr, 5);
    SUCCEED();
}

TEST_F(DevWorkspaceTest, CalculateSlabCapacityPerType_ExceedsMaxTypes)
{
    DevAscendProgram prog{};
    DeviceWorkspaceAllocator allocator(&prog);
    uint32_t slabCapacity[32] = {0};
    allocator.CalculateSlabCapacityPerType(4096, slabCapacity, 999);
    SUCCEED();
}

TEST_F(DevWorkspaceTest, DevFunctionDuppedSlabMemObjSize_ZeroMax)
{
    DevAscendProgram prog{};
    DeviceWorkspaceAllocator allocator(&prog);
    allocator.maxDevFuncDuppedSize_ = 0;
    uint32_t size = allocator.DevFunctionDuppedSlabMemObjSize();
    EXPECT_EQ(size, 0u);
}

TEST_F(DevWorkspaceTest, CalcMetadataItemPoolMemSize_ReturnsValue)
{
    DevAscendProgram prog{};
    prog.memBudget.tensor.runtimeOutcastPoolSize = 100;
    DeviceWorkspaceAllocator allocator(&prog);
    uint64_t size = allocator.CalcMetadataItemPoolMemSize(&prog);
    EXPECT_GT(size, 0u);
}

TEST_F(DevWorkspaceTest, CalcMetadataVectorMemSize_ReturnsValue)
{
    DevAscendProgram prog{};
    prog.symbolTable.size_ = 10;
    prog.slotSize = 20;
    prog.memBudget.tensor.devTaskBoundaryOutcastNum = 5;
    prog.memBudget.tensor.devTaskInnerTemporalOutcastNum = 5;
    DeviceWorkspaceAllocator allocator(&prog);
    uint64_t size = allocator.CalcMetadataVectorMemSize(&prog);
    EXPECT_GT(size, 0u);
}

TEST_F(DevWorkspaceTest, CalcMetadataSlotAllocatorMemSize_ReturnsValue)
{
    DevAscendProgram prog{};
    prog.memBudget.tensor.devTaskBoundaryOutcastNum = 10;
    prog.memBudget.tensor.devTaskInnerTemporalOutcastNum = 10;
    prog.memBudget.metadata.dynamicCellMatchSlotNum = 5;
    DeviceWorkspaceAllocator allocator(&prog);
    uint64_t size = allocator.CalcMetadataSlotAllocatorMemSize(&prog);
    EXPECT_GT(size, 0u);
}

TEST_F(DevWorkspaceTest, CalcAicpuMetaSlabAlloctorSlabPageSize_ReturnsValue)
{
    DevAscendProgram prog{};
    prog.stitchFunctionsize = 10;
    prog.memBudget.metadata.generalSlabSize = 4096;
    DeviceWorkspaceAllocator allocator(&prog);
    uint32_t pageSize = allocator.CalcAicpuMetaSlabAlloctorSlabPageSize(1024 * 1024);
    EXPECT_GT(pageSize, 0u);
}
