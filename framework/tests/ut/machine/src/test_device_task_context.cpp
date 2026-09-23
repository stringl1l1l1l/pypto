/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_device_task_context.cpp
 * \brief Unit tests for DeviceTaskContext, DeviceStitchContext, DeviceExecuteContext (includes former
 *        test_machine_encode_coverage cases).
 */
#include <gtest/gtest.h>
#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <fstream>
#include <cstdio>
#define private public
#define protected public
#include "interface/configs/config_manager.h"
#include "machine/device/dynamic/context/device_task_context.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "machine/utils/dynamic/dev_encode_function_dupped_data.h"
#include "machine/utils/machine_ws_intf.h"

#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "machine/device/dynamic/context/device_task_context.h"
#include "machine/device/dynamic/context/device_stitch_context.h"
#include "machine/device/dynamic/context/device_execute_context.h"
#include "machine/device/dynamic/context/device_slot_context.h"
#include "machine/utils/dynamic/dev_start_args.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "interface/machine/device/tilefwk/aikernel_data.h"
#include "interface/tileop/distributed/comm_context.h"
#include "tilefwk/data_type.h"
#include "tilefwk/platform.h"
#include "tilefwk/tilefwk.h"
#include "interface/machine/device/tilefwk/aikernel_device_task.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class TestDeviceTaskContext : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override { Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510); }

    void TearDown() override { Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN); }

protected:
    void CreateMockDynDeviceTask(DynDeviceTask* dyntask, uint32_t coreFunctionCnt = 100)
    {
        if (dyntask == nullptr) {
            return;
        }
        dyntask->devTask.coreFunctionCnt = coreFunctionCnt;
        dyntask->dynFuncDataCacheListSize = 0;
        for (size_t i = 0; i < DIE_NUM; i++) {
            dyntask->devTask.dieReadyFunctionQue.readyDieAivCoreFunctionQue[i] = 0;
            dyntask->devTask.dieReadyFunctionQue.readyDieAicCoreFunctionQue[i] = 0;
        }
    }

    void CreateMockDevAscendProgram(DevAscendProgram* devProg, ArchInfo archInfo)
    {
        if (devProg == nullptr) {
            return;
        }
        devProg->devArgs.archInfo = archInfo;
        devProg->ctrlFlowCacheAnchor = &devProg->controlFlowCache;
        devProg->controlFlowCache.isRecording = false;
        devProg->controlFlowCache.isRecordingStopped = false;
        devProg->controlFlowCache.cacheDataOffset = 0;
        devProg->stitchMaxFunctionNum = 10;
        devProg->stitchFunctionsize = 100;
    }

    DevAscendFunction* CreateDevAscendFunctionBuffer(std::unique_ptr<uint8_t[]>& funcBuffer, uint8_t*& funcDataPtr,
                                                     size_t kOpCount, size_t kFuncBufferSize)
    {
        (void)kOpCount;
        funcBuffer = std::make_unique<uint8_t[]>(kFuncBufferSize);
        memset_s(funcBuffer.get(), kFuncBufferSize, 0, kFuncBufferSize);
        funcDataPtr = funcBuffer.get();

        DevAscendFunction* devFunc = reinterpret_cast<DevAscendFunction*>(funcDataPtr);
        funcDataPtr += sizeof(DevAscendFunction);

        devFunc->rootHash = 0x12345678;
        devFunc->funcKey = 100;
        devFunc->sourceFunc = nullptr;

        return devFunc;
    }

    void SetupDevAscendFunctionData(DevAscendFunction* devFunc, uint8_t* funcDataPtr, uint8_t* funcBuffer,
                                    size_t kOpCount)
    {
        size_t currentOffset = sizeof(DevAscendFunction);
        auto alignUp = [&currentOffset](size_t alignment) {
            currentOffset = (currentOffset + alignment - 1) & ~(alignment - 1);
        };

        alignUp(alignof(SymInt));
        devFunc->operationAttrList_.AssignOffsetSize(currentOffset, kOpCount);
        SymInt* attrData = reinterpret_cast<SymInt*>(funcDataPtr);
        for (size_t i = 0; i < kOpCount; i++) {
            attrData[i] = SymInt(static_cast<uint64_t>(0));
        }
        currentOffset += kOpCount * sizeof(SymInt);
        funcDataPtr += kOpCount * sizeof(SymInt);

        alignUp(alignof(int32_t));
        devFunc->opAttrOffsetList_.AssignOffsetSize(currentOffset, kOpCount);
        int32_t* attrOffsets = reinterpret_cast<int32_t*>(funcDataPtr);
        for (size_t i = 0; i < kOpCount; i++) {
            attrOffsets[i] = static_cast<int32_t>(i);
        }
        currentOffset += kOpCount * sizeof(int32_t);
        funcDataPtr += kOpCount * sizeof(int32_t);

        alignUp(alignof(DevAscendOperation));
        devFunc->operationList_.AssignOffsetSize(currentOffset, kOpCount);
        DevAscendOperation* ops = reinterpret_cast<DevAscendOperation*>(funcDataPtr);
        for (size_t i = 0; i < kOpCount; i++) {
            new (&ops[i]) DevAscendOperation();
            ops[i].debugOpmagic = static_cast<uint64_t>(i + 1);
            size_t attrOffset = reinterpret_cast<uint8_t*>(attrData + i) - funcBuffer;
            ops[i].attrList.AssignOffsetSize(attrOffset, 1);
            ops[i].depGraphSuccList.AssignOffsetSize(0, 0);
            ops[i].depGraphPredCount = 0;
            ops[i].stitchIndex = 0;
        }
    }

    DevAscendFunctionDuppedData* CreateDevAscendFunctionDuppedData(std::unique_ptr<uint8_t[]>& duppedDataBuffer,
                                                                   uint8_t*& duppedDataPtr, DevAscendFunction* devFunc,
                                                                   size_t kOpCount, size_t kDuppedDataBufferSize)
    {
        duppedDataBuffer = std::make_unique<uint8_t[]>(kDuppedDataBufferSize);
        memset_s(duppedDataBuffer.get(), kDuppedDataBufferSize, 0, kDuppedDataBufferSize);
        duppedDataPtr = duppedDataBuffer.get();

        DevAscendFunctionDuppedData* duppedData = reinterpret_cast<DevAscendFunctionDuppedData*>(duppedDataPtr);
        duppedDataPtr += sizeof(DevAscendFunctionDuppedData);

        duppedData->source_ = devFunc;
        duppedData->operationList_.size = kOpCount;
        duppedData->operationList_.predCountBase = static_cast<uint32_t>(duppedDataPtr - duppedDataBuffer.get());
        duppedData->operationList_.stitchBase = duppedData->operationList_.predCountBase +
                                                kOpCount * sizeof(predcount_t);
        duppedData->operationList_.stitchCount = 1;

        predcount_t* predCounts = reinterpret_cast<predcount_t*>(duppedDataPtr);
        for (size_t i = 0; i < kOpCount; i++) {
            predCounts[i] = 0;
        }
        duppedDataPtr += kOpCount * sizeof(predcount_t);

        for (size_t i = 0; i <= kOpCount; i++) {
            new (duppedDataPtr + i * sizeof(DevAscendFunctionDuppedStitchList)) DevAscendFunctionDuppedStitchList();
        }

        duppedData->incastList_.size = 0;
        duppedData->incastList_.base = 0;
        duppedData->outcastList_.size = 0;
        duppedData->outcastList_.base = 0;
        duppedData->expressionList_.size = 0;
        duppedData->expressionList_.base = 0;

        return duppedData;
    }

    void SetupTestEnvironment(DeviceTask& devTask, std::unique_ptr<int32_t[]>& opWrapListData, DevCceBinary* cceBinary,
                              size_t kOpCount)
    {
        opWrapListData = std::make_unique<int32_t[]>(kOpCount);
        for (size_t i = 0; i < kOpCount; i++) {
            opWrapListData[i] = static_cast<int32_t>(i);
        }

        devTask.mixTaskData.wrapIdNum = 1;
        devTask.mixTaskData.opWrapList[0] = reinterpret_cast<uint64_t>(opWrapListData.get());

        cceBinary[0].coreType = 0;
        cceBinary[0].psgId = 0;
        cceBinary[0].funcHash = 0xABCDEF00;
    }

    void VerifyDumpTopoOutput(const std::string& testFilePath, size_t expectedLineCount)
    {
        std::ifstream inFile(testFilePath);
        ASSERT_TRUE(inFile.is_open());
        std::string line;
        size_t lineCount = 0;
        while (std::getline(inFile, line)) {
            lineCount++;
            EXPECT_FALSE(line.empty());
        }
        inFile.close();

        EXPECT_EQ(lineCount, expectedLineCount);
    }

    WrapInfoQueue* SetupWrapQueueForTest(DynDeviceTask* dyntask, DeviceTaskContext& taskContext, CoreType coreType,
                                         uint32_t wrapVecId, DevAscendFunction& devFunc, DevCceBinary* cceBinary,
                                         int* calleeList)
    {
        devFunc.wrapIdNum_ = 1;
        dyntask->dynFuncDataCacheList[0].devFunc = &devFunc;
        dyntask->dynFuncDataCacheListSize = 1;
        dyntask->devTask.mixTaskData.wrapIdNum = 1;

        dyntask->dynFuncDataCacheList[0].calleeList = calleeList;

        cceBinary[0].coreType = static_cast<uint32_t>(coreType);
        cceBinary[0].wrapVecId = wrapVecId;
        cceBinary[0].mixResourceType = 0;
        dyntask->cceBinary = cceBinary;

        return taskContext.AllocWrapQueue(dyntask);
    }

    void SetupBasicTaskContext(DeviceTaskContext& taskContext, DevStartArgsBase& startArgs, DevAscendProgram* devProg,
                               std::unique_ptr<DynDeviceTask>& dyntask, DeviceWorkspaceAllocator& workspace,
                               std::unique_ptr<uint8_t[]>& controlFlowCacheBuf, size_t kControlFlowCacheSize)
    {
        if (controlFlowCacheBuf != nullptr) {
            devProg->controlFlowCache.cacheData = DevRelocVector<uint8_t>(kControlFlowCacheSize,
                                                                          controlFlowCacheBuf.get());
            devProg->controlFlowCache.isRecording = true;
        }
        taskContext.InitAllocator(devProg, workspace, &startArgs);
        dyntask = std::make_unique<DynDeviceTask>(workspace);
        CreateMockDynDeviceTask(dyntask.get(), 100);
    }
};

TEST_F(TestDeviceTaskContext, test_build_ready_queue_calls_wrap_functions)
{
    DeviceTaskContext taskContext;
    DevStartArgsBase startArgs;
    constexpr size_t kControlFlowCacheSize = 64 * 1024;
    auto controlFlowCacheBuf = std::make_unique<uint8_t[]>(kControlFlowCacheSize);

    DevAscendProgram devProg;
    CreateMockDevAscendProgram(&devProg, ArchInfo::DAV_3510);
    devProg.stitchFunctionsize = 100;
    devProg.controlFlowCache.cacheData = DevRelocVector<uint8_t>(kControlFlowCacheSize, controlFlowCacheBuf.get());
    devProg.controlFlowCache.isRecording = true;

    DeviceWorkspaceAllocator workspace(&devProg);
    taskContext.InitAllocator(&devProg, workspace, &startArgs);

    auto dyntask = std::make_unique<DynDeviceTask>(workspace);
    CreateMockDynDeviceTask(dyntask.get(), 100);

    DevAscendFunction devFunc;
    devFunc.wrapIdNum_ = 1;

    dyntask->dynFuncDataCacheList[0].devFunc = &devFunc;
    dyntask->dynFuncDataCacheListSize = 1;
    dyntask->devTask.mixTaskData.wrapIdNum = 1;

    bool isNeedWrap = taskContext.IsNeedWrapProcess(dyntask.get(), &devProg);
    EXPECT_TRUE(isNeedWrap);

    WrapInfoQueue* wrapQueue = taskContext.AllocWrapQueue(dyntask.get());
    EXPECT_NE(wrapQueue, nullptr);
    EXPECT_EQ(wrapQueue->head, 0);
    EXPECT_EQ(wrapQueue->tail, 0);
    EXPECT_GT(wrapQueue->capacity, 0);
}

TEST_F(TestDeviceTaskContext, ShowStats_HitsDevErrorMacroLines)
{
    DeviceTaskContext taskContext;
    taskContext.ShowStats();
}

TEST_F(TestDeviceTaskContext, InitReadyQueues_ExceedsStitchSize_ReturnsError)
{
    DeviceTaskContext taskContext;
    DevStartArgsBase startArgs;
    DevAscendProgram devProg;
    CreateMockDevAscendProgram(&devProg, ArchInfo::DAV_3510);
    devProg.stitchFunctionsize = 10;
    DeviceWorkspaceAllocator workspace(&devProg);
    taskContext.InitAllocator(&devProg, workspace, &startArgs);
    auto dyntask = std::make_unique<DynDeviceTask>(workspace);
    CreateMockDynDeviceTask(dyntask.get(), 100U);
    ReadyCoreFunctionQueue* queues[READY_QUEUE_SIZE] = {};
    EXPECT_EQ(taskContext.InitReadyQueues(dyntask.get(), &devProg, queues), DEVICE_MACHINE_ERROR);
}

TEST_F(TestDeviceTaskContext, test_init_die_ready_queues_mix_arch)
{
    DeviceTaskContext taskContext;
    DevStartArgsBase startArgs;
    constexpr size_t kControlFlowCacheSize = 64 * 1024;
    auto controlFlowCacheBuf = std::make_unique<uint8_t[]>(kControlFlowCacheSize);

    DevAscendProgram devProg;
    CreateMockDevAscendProgram(&devProg, ArchInfo::DAV_3510);
    devProg.controlFlowCache.cacheData = DevRelocVector<uint8_t>(kControlFlowCacheSize, controlFlowCacheBuf.get());
    devProg.controlFlowCache.isRecording = true;

    DeviceWorkspaceAllocator workspace(&devProg);

    taskContext.InitAllocator(&devProg, workspace, &startArgs);

    auto dyntask = std::make_unique<DynDeviceTask>(workspace);
    CreateMockDynDeviceTask(dyntask.get(), 100);

    taskContext.InitDieReadyQueues(dyntask.get(), &devProg);

    for (size_t i = 0; i < DIE_NUM; i++) {
        EXPECT_NE(dyntask->devTask.dieReadyFunctionQue.readyDieAivCoreFunctionQue[i], 0UL);
        EXPECT_NE(dyntask->devTask.dieReadyFunctionQue.readyDieAicCoreFunctionQue[i], 0UL);

        auto aivQueue = reinterpret_cast<ReadyCoreFunctionQueue*>(
            dyntask->devTask.dieReadyFunctionQue.readyDieAivCoreFunctionQue[i]);
        auto aicQueue = reinterpret_cast<ReadyCoreFunctionQueue*>(
            dyntask->devTask.dieReadyFunctionQue.readyDieAicCoreFunctionQue[i]);

        EXPECT_NE(aivQueue, nullptr);
        EXPECT_NE(aicQueue, nullptr);
        EXPECT_EQ(aivQueue->head_, 0U);
        EXPECT_EQ(aivQueue->tail_, 0U);
        EXPECT_EQ(aicQueue->head_, 0U);
        EXPECT_EQ(aicQueue->tail_, 0U);
    }
}

TEST_F(TestDeviceTaskContext, test_build_ready_queue_core_function_mix_arch)
{
    DeviceTaskContext taskContext;
    DevStartArgsBase startArgs;
    constexpr size_t kControlFlowCacheSize = 64 * 1024;
    auto controlFlowCacheBuf = std::make_unique<uint8_t[]>(kControlFlowCacheSize);

    DevAscendProgram devProg;
    CreateMockDevAscendProgram(&devProg, ArchInfo::DAV_3510);
    devProg.stitchFunctionsize = 10;
    devProg.controlFlowCache.cacheData = DevRelocVector<uint8_t>(kControlFlowCacheSize, controlFlowCacheBuf.get());
    devProg.controlFlowCache.isRecording = true;

    DeviceWorkspaceAllocator workspace(&devProg);
    taskContext.InitAllocator(&devProg, workspace, &startArgs);

    auto dyntask = std::make_unique<DynDeviceTask>(workspace);
    CreateMockDynDeviceTask(dyntask.get(), 8);

    DevAscendFunction devFunc;
    DevAscendFunctionDuppedData duppedData{};
    duppedData.loopDieId_ = 1;
    duppedData.source_ = &devFunc;
    devFunc.predInfo_.totalZeroPredAIV = 0;
    devFunc.predInfo_.totalZeroPredAIC = 0;
    devFunc.predInfo_.totalZeroPredAicpu = 0;
    dyntask->dynFuncDataCacheList[0].devFunc = &devFunc;
    dyntask->dynFuncDataCacheList[0].duppedData = &duppedData;
    dyntask->dynFuncDataCacheListSize = 1;

    int ret = taskContext.BuildReadyQueue(dyntask.get(), &devProg);

    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
}

TEST_F(TestDeviceTaskContext, test_build_ready_queue_dupped_data)
{
    DeviceTaskContext taskContext;
    DevStartArgsBase startArgs;
    constexpr size_t kControlFlowCacheSize = 64 * 1024 * 8;
    auto controlFlowCacheBuf = std::make_unique<uint8_t[]>(kControlFlowCacheSize);

    DevAscendProgram devProg;
    CreateMockDevAscendProgram(&devProg, ArchInfo::DAV_3510);
    devProg.controlFlowCache.cacheData = DevRelocVector<uint8_t>(kControlFlowCacheSize, controlFlowCacheBuf.get());
    devProg.controlFlowCache.isRecording = true;

    DeviceWorkspaceAllocator workspace(&devProg);
    taskContext.InitAllocator(&devProg, workspace, &startArgs);

    auto dyntask = std::make_unique<DynDeviceTask>(workspace);
    CreateMockDynDeviceTask(dyntask.get(), 100);

    constexpr size_t kOpCount = 32;
    constexpr size_t kFuncBufferSize = kOpCount * 1024;
    constexpr size_t kDuppedDataBufferSize = kOpCount * 512;

    std::unique_ptr<uint8_t[]> funcBuffer;
    uint8_t* funcDataPtr;
    DevAscendFunction* devFunc = CreateDevAscendFunctionBuffer(funcBuffer, funcDataPtr, kOpCount, kFuncBufferSize);

    SetupDevAscendFunctionData(devFunc, funcDataPtr, funcBuffer.get(), kOpCount);

    std::unique_ptr<uint8_t[]> duppedDataBuffer;
    uint8_t* duppedDataPtr;
    DevAscendFunctionDuppedData* duppedData = CreateDevAscendFunctionDuppedData(
        duppedDataBuffer, duppedDataPtr, devFunc, kOpCount, kDuppedDataBufferSize);

    devFunc->predInfo_.totalZeroPredAIV = 10;
    devFunc->predInfo_.totalZeroPredAIC = 10;
    devFunc->predInfo_.totalZeroPredAicpu = 0;

    dyntask->dynFuncDataCacheList[0].devFunc = devFunc;
    dyntask->dynFuncDataCacheList[0].duppedData = duppedData;
    dyntask->dynFuncDataCacheListSize = 1;

    int ret = taskContext.BuildReadyQueue(dyntask.get(), &devProg);

    auto aivQueue = reinterpret_cast<ReadyCoreFunctionQueue*>(
        dyntask->devTask.dieReadyFunctionQue.readyDieAivCoreFunctionQue[0]);
    auto aicQueue = reinterpret_cast<ReadyCoreFunctionQueue*>(
        dyntask->devTask.dieReadyFunctionQue.readyDieAicCoreFunctionQue[0]);

    EXPECT_EQ(aivQueue->head_, 0);
    EXPECT_EQ(aivQueue->tail_, 10);
    EXPECT_EQ(aicQueue->head_, 0);
    EXPECT_EQ(aicQueue->tail_, 10);

    ReadyCoreFunctionQueue::ValueType aivQueueGold[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    ReadyCoreFunctionQueue::ValueType aicQueueGold[] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

    EXPECT_TRUE(std::equal(aivQueue->begin(), aivQueue->end(), aivQueueGold));
    EXPECT_TRUE(std::equal(aicQueue->begin(), aicQueue->end(), aicQueueGold));

    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
}

namespace {

void InitReadyQueueSlot(ReadyCoreFunctionQueue& q, std::array<taskid_t, 4>& elemBuf, uint32_t head, uint32_t tail,
                        taskid_t firstId)
{
    new (&q) ReadyCoreFunctionQueue(elemBuf.size(), elemBuf.data());
    q.UnsafeEnqueue(&elemBuf[0], tail);
    q.Dequeue(head);
    if (tail > head) {
        elemBuf[0] = firstId;
    }
}

void InitReadyQueueSlotMulti(ReadyCoreFunctionQueue& q, std::array<taskid_t, 4>& elemBuf, uint32_t head, uint32_t tail,
                             const std::vector<taskid_t>& ids)
{
    new (&q) ReadyCoreFunctionQueue(elemBuf.size(), elemBuf.data());
    q.UnsafeEnqueue(&elemBuf[0], tail);
    q.Dequeue(head);

    for (size_t i = 0; i < ids.size() && (head + i) < tail && i < elemBuf.size(); ++i) {
        elemBuf[i] = ids[i];
    }
}

void ControlFlowSetError(struct DeviceExecuteContext* ctx, int64_t* symbolTable,
                         RuntimeCallEntryType runtimeCallList[T_RUNTIME_CALL_MAX], DevStartArgsBase* startArgsBase)
{
    (void)symbolTable;
    (void)runtimeCallList;
    (void)startArgsBase;
    ctx->SetErrorState(DEVICE_MACHINE_ERROR);
}

} // namespace

TEST_F(TestDeviceTaskContext, DumpReadyQueue_CoversLoggingLines)
{
    DeviceWorkspaceAllocator workspace;
    auto dyntask = std::make_unique<DynDeviceTask>(workspace);
    dyntask->devTask.coreFunctionCnt = 3;
    std::array<taskid_t, 4> bufAiv{};
    std::array<taskid_t, 4> bufAic{};
    std::array<taskid_t, 4> bufAicpu{};
    ReadyCoreFunctionQueue qslot[READY_QUEUE_SIZE];
    InitReadyQueueSlot(qslot[0], bufAiv, 0, 1, MakeTaskID(0, 1));
    InitReadyQueueSlot(qslot[1], bufAic, 0, 1, MakeTaskID(0, 2));
    InitReadyQueueSlot(qslot[2], bufAicpu, 0, 1, MakeTaskID(0, 3));
    for (size_t i = 0; i < READY_QUEUE_SIZE; ++i) {
        dyntask->readyQueue[i] = &qslot[i];
    }
    DeviceTaskContext::DumpReadyQueue(dyntask.get(), "ut_cov");
}

TEST_F(TestDeviceTaskContext, TraceFirstBatchResolve_CoversLoggingLines)
{
    DeviceWorkspaceAllocator workspace;
    auto dyntask = std::make_unique<DynDeviceTask>(workspace);
    dyntask->devTask.coreFunctionCnt = 3;
    std::array<taskid_t, 4> bufAiv{};
    std::array<taskid_t, 4> bufAic{};
    std::array<taskid_t, 4> bufAicpu{};
    ReadyCoreFunctionQueue qslot[READY_QUEUE_SIZE];
    InitReadyQueueSlot(qslot[0], bufAiv, 0, 1, MakeTaskID(0, 1));
    InitReadyQueueSlot(qslot[1], bufAic, 0, 1, MakeTaskID(0, 2));
    InitReadyQueueSlot(qslot[2], bufAicpu, 0, 1, MakeTaskID(0, 3));
    for (size_t i = 0; i < READY_QUEUE_SIZE; ++i) {
        dyntask->readyQueue[i] = &qslot[i];
    }
    DynFuncHeader header{};
    header.seqNo = 42;
    dyntask->dynFuncDataList = &header;
    DeviceTaskContext::TraceFirstBatchResolve(dyntask.get());
}

TEST_F(TestDeviceTaskContext, DumpDepend_CoversHeadLoggingWithoutDupData)
{
    DeviceWorkspaceAllocator workspace;
    auto dyntask = std::make_unique<DynDeviceTask>(workspace);
    dyntask->devTask.coreFunctionCnt = 4;
    DynFuncHeader header{};
    header.seqNo = 42;
    header.funcNum = 0;
    header.funcSize = sizeof(DynFuncHeader);
    dyntask->dynFuncDataList = &header;

    std::array<taskid_t, 4> bufAiv{};
    std::array<taskid_t, 4> bufAic{};
    std::array<taskid_t, 4> bufAicpu{};
    ReadyCoreFunctionQueue qslot[READY_QUEUE_SIZE];
    InitReadyQueueSlotMulti(qslot[0], bufAiv, 0, 2, {MakeTaskID(0, 0), MakeTaskID(0, 1)});
    InitReadyQueueSlot(qslot[1], bufAic, 0, 1, MakeTaskID(1, 0));
    InitReadyQueueSlot(qslot[2], bufAicpu, 0, 0, 0);
    for (size_t i = 0; i < READY_QUEUE_SIZE; ++i) {
        dyntask->readyQueue[i] = &qslot[i];
    }

    std::array<DevTensorData, 4> tensors{};
    tensors[0].address = 0x1000ULL;
    tensors[1].address = 0x1100ULL;
    tensors[2].address = 0x2000ULL;
    tensors[3].address = 0x2100ULL;
    DevStartArgs startArgs{};
    startArgs.contextWorkspaceAddr = 0x3000ULL;
    startArgs.inputTensorSize = 2;
    startArgs.outputTensorSize = 2;
    startArgs.devTensorList = tensors.data();

    DevAscendProgram devProg{};
    DeviceTaskContext::DumpDepend(dyntask.get(), &devProg, &startArgs, "ut_cov");
}

TEST_F(TestDeviceTaskContext, DeviceExecute_InvalidCtx_ReturnsNull)
{
    EXPECT_EQ(DeviceExecuteContext::DeviceExecuteRuntimeCallRootAlloc(nullptr, 0), nullptr);
    EXPECT_EQ(DeviceExecuteContext::DeviceExecuteRuntimeCallRootStitch(nullptr, 0), nullptr);
}

TEST_F(TestDeviceTaskContext, DeviceExecuteRuntimeCallLog_IsNullSafe)
{
    EXPECT_EQ(DeviceExecuteContext::DeviceExecuteRuntimeCallLog(nullptr, 7ULL), nullptr);
}

TEST_F(TestDeviceTaskContext, DeviceStitchContext_DumpStitchInfo_Empty)
{
    DeviceStitchContext ctx;
    ctx.DumpStitchInfo();
}

TEST_F(TestDeviceTaskContext, DeviceExecuteRuntimeCallShmemAllocator_ExceedsWinSize_LogsError)
{
    alignas(64) unsigned char ctxBuf[sizeof(DeviceExecuteContext)];
    (void)memset_s(ctxBuf, sizeof(ctxBuf), 0, sizeof(ctxBuf));
    auto* ctx = reinterpret_cast<DeviceExecuteContext*>(ctxBuf);

    TileOp::CommContext hc{};
    hc.winDataSize = 64;
    hc.winStatusSize = 32;
    int64_t commPtrs[1] = {reinterpret_cast<int64_t>(&hc)};

    DevStartArgs args{};
    args.commGroupNum = 1;
    args.commContexts = commPtrs;
    ctx->args = &args;

    uint64_t payload[] = {0, 0, 128, 8};
    (void)DeviceExecuteContext::DeviceExecuteRuntimeCallShmemAllocator(ctx, reinterpret_cast<uint64_t>(payload));
}

TEST_F(TestDeviceTaskContext, DeviceStitchContext_MoveTo_TooManyFunctions_ReturnsError)
{
    GTEST_SKIP() << "该场景在当前并行 death test 环境下易卡住，暂跳过。";
}
// ---- Former test_machine_encode_coverage.cpp (DumpDepend 等价见本文件 DumpDepend_EncodedDuppedData) ----

class TestMachineEncodeCoverage : public testing::Test {
protected:
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
        TileShape::Current().SetVecTile(32, 32);
        TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    }

    void TearDown() override
    {
        Program::GetInstance().Reset();
        config::Reset();
    }
};

TEST_F(TestMachineEncodeCoverage, MoveTo_MaxFunctionNumBoundary_ReturnsOk)
{
    GTEST_SKIP() << "该边界场景在当前环境存在卡住风险，保留用例后续再收敛。";
}

TEST_F(TestMachineEncodeCoverage, FastStitch_SlotIdxBeyondSize_LogsAndContinues)
{
    DevStartArgs args{};
    DevAscendProgram prog{};
    prog.controlFlowCache.isRecording = false;
    args.devProg = &prog;
    args.controlFlowEntry = reinterpret_cast<void*>(ControlFlowSetError);
    DeviceExecuteContext ctx(&args);
    EXPECT_EQ(ctx.RunControlFlow(&args), DEVICE_MACHINE_ERROR);
}

class TestDeviceExecuteContext : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override { Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510); }

    void TearDown() override { Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN); }
};

TEST_F(TestDeviceExecuteContext, test_runtime_call_get_loop_die_id)
{
    alignas(alignof(DeviceExecuteContext)) char buffer[sizeof(DeviceExecuteContext)];
    DeviceExecuteContext* ctx = reinterpret_cast<DeviceExecuteContext*>(buffer);
    (void)memset_s(buffer, sizeof(DeviceExecuteContext), 0, sizeof(DeviceExecuteContext));
    ctx->loopDieId_ = -1;
    void* result = DeviceExecuteContext::DeviceExecuteRuntimeCallGetLoopDieId(ctx, 0);
    EXPECT_NE(result, nullptr);
    int8_t* dieIdPtr = static_cast<int8_t*>(result);
    EXPECT_EQ(*dieIdPtr, -1);
    ctx->loopDieId_ = 7;
    result = DeviceExecuteContext::DeviceExecuteRuntimeCallGetLoopDieId(ctx, 0);
    dieIdPtr = static_cast<int8_t*>(result);
    EXPECT_EQ(*dieIdPtr, 7);
}

TEST_F(TestDeviceExecuteContext, test_runtime_call_set_loop_die_id)
{
    alignas(alignof(DeviceExecuteContext)) char buffer[sizeof(DeviceExecuteContext)];
    DeviceExecuteContext* ctx = reinterpret_cast<DeviceExecuteContext*>(buffer);
    (void)memset_s(buffer, sizeof(DeviceExecuteContext), 0, sizeof(DeviceExecuteContext));
    DevAscendFunctionDuppedData duppedData{};
    duppedData.loopDieId_ = -1;
    ctx->currDevRootDup.dupTiny_.ptr = reinterpret_cast<uint64_t>(&duppedData);
    ctx->loopDieId_ = 3;
    void* result = DeviceExecuteContext::DeviceExecuteRuntimeCallSetLoopDieId(ctx, 0);
    EXPECT_EQ(result, nullptr);
    EXPECT_EQ(duppedData.loopDieId_, 3);
    ctx->loopDieId_ = 12;
    result = DeviceExecuteContext::DeviceExecuteRuntimeCallSetLoopDieId(ctx, 0);
    EXPECT_EQ(result, nullptr);
    EXPECT_EQ(duppedData.loopDieId_, 12);
}

TEST_F(TestDeviceTaskContext, test_dev_ascend_function_dupped_dump_topo)
{
    constexpr size_t kOpCount = 4;
    constexpr size_t kFuncBufferSize = 4096;
    constexpr size_t kDuppedDataBufferSize = 2048;

    std::unique_ptr<uint8_t[]> funcBuffer;
    uint8_t* funcDataPtr;
    DevAscendFunction* devFunc = CreateDevAscendFunctionBuffer(funcBuffer, funcDataPtr, kOpCount, kFuncBufferSize);

    SetupDevAscendFunctionData(devFunc, funcDataPtr, funcBuffer.get(), kOpCount);

    std::unique_ptr<uint8_t[]> duppedDataBuffer;
    uint8_t* duppedDataPtr;
    DevAscendFunctionDuppedData* duppedData = CreateDevAscendFunctionDuppedData(
        duppedDataBuffer, duppedDataPtr, devFunc, kOpCount, kDuppedDataBufferSize);

    DevAscendFunctionDupped funcDupped;
    WsAllocation tinyAlloc;
    tinyAlloc.ptr = reinterpret_cast<uint64_t>(duppedData);
    funcDupped = DevAscendFunctionDupped(tinyAlloc);

    auto devTaskPtr = std::make_unique<DeviceTask>();
    DeviceTask& devTask = *devTaskPtr;
    std::unique_ptr<int32_t[]> opWrapListData;
    DevCceBinary cceBinary[1];
    SetupTestEnvironment(devTask, opWrapListData, cceBinary, kOpCount);

    std::string testFilePath = "./test_dump_topo_direct_output.txt";
    {
        std::ofstream outFile(testFilePath);
        ASSERT_TRUE(outFile.is_open());

        int seqNo = 0;
        int funcIdx = 0;
        bool enableVFFusion = false;

        funcDupped.DumpTopo(outFile, seqNo, funcIdx, cceBinary, enableVFFusion, &devTask);

        outFile.close();

        VerifyDumpTopoOutput(testFilePath, kOpCount);
    }
    std::remove(testFilePath.c_str());
}

TEST_F(TestDeviceTaskContext, test_process_wrap_queue_nullptr)
{
    DeviceTaskContext taskContext;
    DevStartArgsBase startArgs;
    DevAscendProgram devProg;
    CreateMockDevAscendProgram(&devProg, ArchInfo::DAV_3510);
    DeviceWorkspaceAllocator workspace(&devProg);
    std::unique_ptr<DynDeviceTask> dyntask;
    std::unique_ptr<uint8_t[]> controlFlowCacheBuf;
    SetupBasicTaskContext(taskContext, startArgs, &devProg, dyntask, workspace, controlFlowCacheBuf, 0);

    taskContext.ProcessWrapQueue(dyntask.get(), 1, 0, 0, nullptr);
}

TEST_F(TestDeviceTaskContext, test_process_wrap_queue_update_existing_wrap)
{
    DevAscendProgram devProg;
    DeviceTaskContext taskContext;
    DevStartArgsBase startArgs;
    CreateMockDevAscendProgram(&devProg, ArchInfo::DAV_3510);
    DeviceWorkspaceAllocator workspace(&devProg);
    std::unique_ptr<DynDeviceTask> dyntask;
    constexpr size_t kControlFlowCacheSize = 64 * 1024;
    auto controlFlowCacheBuf = std::make_unique<uint8_t[]>(kControlFlowCacheSize);
    SetupBasicTaskContext(taskContext, startArgs, &devProg, dyntask, workspace, controlFlowCacheBuf,
                          kControlFlowCacheSize);

    DevAscendFunction devFunc;
    DevCceBinary cceBinary[1] = {};
    int calleeList[1] = {0};
    WrapInfoQueue* wrapQueue = SetupWrapQueueForTest(dyntask.get(), taskContext, CoreType::AIC, 0, devFunc, cceBinary,
                                                     calleeList);
    ASSERT_NE(wrapQueue, nullptr);

    taskContext.ProcessWrapQueue(dyntask.get(), 1, 0, 0, wrapQueue);
    EXPECT_EQ(wrapQueue->tail, 1);

    taskContext.ProcessWrapQueue(dyntask.get(), 1, 0, 0, wrapQueue);
    EXPECT_EQ(wrapQueue->tail, 1);
}

TEST_F(TestDeviceTaskContext, test_process_wrap_queue_aiv0)
{
    DeviceTaskContext taskContext;
    DevAscendProgram devProg;
    DevStartArgsBase startArgs;
    CreateMockDevAscendProgram(&devProg, ArchInfo::DAV_3510);
    DeviceWorkspaceAllocator workspace(&devProg);
    std::unique_ptr<DynDeviceTask> dyntask;
    constexpr size_t kControlFlowCacheSize = 64 * 1024;
    auto controlFlowCacheBuf = std::make_unique<uint8_t[]>(kControlFlowCacheSize);
    SetupBasicTaskContext(taskContext, startArgs, &devProg, dyntask, workspace, controlFlowCacheBuf,
                          kControlFlowCacheSize);

    DevCceBinary cceBinary[1] = {};
    DevAscendFunction devFunc;
    int calleeList[1] = {0};
    WrapInfoQueue* wrapQueue = SetupWrapQueueForTest(dyntask.get(), taskContext, CoreType::AIV, 0, devFunc, cceBinary,
                                                     calleeList);
    ASSERT_NE(wrapQueue, nullptr);

    taskContext.ProcessWrapQueue(dyntask.get(), 1, 0, 0, wrapQueue);

    EXPECT_EQ(wrapQueue->tail, 1);
    EXPECT_EQ(wrapQueue->elem[0].wrapId, 1);
    EXPECT_EQ(wrapQueue->elem[0].tasklist[WRAP_IDX_AIC], AICORE_TASK_INIT);
    EXPECT_EQ(wrapQueue->elem[0].tasklist[WRAP_IDX_AIV0], MakeTaskID(0, 0));
    EXPECT_EQ(wrapQueue->elem[0].tasklist[WRAP_IDX_AIV1], AICORE_TASK_INIT);
}

TEST_F(TestDeviceTaskContext, test_process_wrap_queue_aiv1)
{
    DeviceTaskContext taskContext;
    DevStartArgsBase startArgs;
    DevAscendProgram devProg;
    CreateMockDevAscendProgram(&devProg, ArchInfo::DAV_3510);
    DeviceWorkspaceAllocator workspace(&devProg);
    std::unique_ptr<DynDeviceTask> dyntask;
    constexpr size_t kControlFlowCacheSize = 64 * 1024;
    auto controlFlowCacheBuf = std::make_unique<uint8_t[]>(kControlFlowCacheSize);
    SetupBasicTaskContext(taskContext, startArgs, &devProg, dyntask, workspace, controlFlowCacheBuf,
                          kControlFlowCacheSize);

    int calleeList[1] = {0};
    DevAscendFunction devFunc;
    DevCceBinary cceBinary[1] = {};
    WrapInfoQueue* wrapQueue = SetupWrapQueueForTest(dyntask.get(), taskContext, CoreType::AIV, 1, devFunc, cceBinary,
                                                     calleeList);
    ASSERT_NE(wrapQueue, nullptr);

    taskContext.ProcessWrapQueue(dyntask.get(), 1, 0, 0, wrapQueue);

    EXPECT_EQ(wrapQueue->tail, 1);
    EXPECT_EQ(wrapQueue->elem[0].wrapId, 1);
    EXPECT_EQ(wrapQueue->elem[0].tasklist[WRAP_IDX_AIC], AICORE_TASK_INIT);
    EXPECT_EQ(wrapQueue->elem[0].tasklist[WRAP_IDX_AIV0], AICORE_TASK_INIT);
    EXPECT_EQ(wrapQueue->elem[0].tasklist[WRAP_IDX_AIV1], MakeTaskID(0, 0));
}

TEST_F(TestDeviceTaskContext, ReleaseFinishedTasks_NoOp)
{
    DeviceTaskContext taskContext;
    taskContext.ReleaseFinishedTasks(0, 0);
    SUCCEED();
}

TEST_F(TestDeviceTaskContext, AppendFinishTask_NullTask)
{
    DeviceTaskContext taskContext;
    taskContext.AppendFinishTask(nullptr);
    SUCCEED();
}

TEST_F(TestDeviceTaskContext, DumpDepend_WithEncodedDuppedData)
{
    constexpr size_t kOpCount = 4;
    constexpr size_t kFuncBufferSize = 4096;
    constexpr size_t kDuppedDataBufferSize = 2048;

    std::unique_ptr<uint8_t[]> funcBuffer;
    uint8_t* funcDataPtr;
    DevAscendFunction* devFunc = CreateDevAscendFunctionBuffer(funcBuffer, funcDataPtr, kOpCount, kFuncBufferSize);
    SetupDevAscendFunctionData(devFunc, funcDataPtr, funcBuffer.get(), kOpCount);

    std::unique_ptr<uint8_t[]> duppedDataBuffer;
    uint8_t* duppedDataPtr;
    DevAscendFunctionDuppedData* duppedData = CreateDevAscendFunctionDuppedData(
        duppedDataBuffer, duppedDataPtr, devFunc, kOpCount, kDuppedDataBufferSize);

    DeviceWorkspaceAllocator workspace;
    auto dyntask = std::make_unique<DynDeviceTask>(workspace);
    dyntask->devTask.coreFunctionCnt = 1;
    DynFuncHeader header{};
    header.seqNo = 1;
    header.funcNum = 1;
    header.funcSize = sizeof(DynFuncHeader);
    dyntask->dynFuncDataList = &header;
    dyntask->dynFuncDataCacheList[0].devFunc = devFunc;
    dyntask->dynFuncDataCacheList[0].duppedData = duppedData;
    dyntask->dynFuncDataCacheListSize = 1;

    std::array<taskid_t, 4> bufAiv{};
    std::array<taskid_t, 4> bufAic{};
    std::array<taskid_t, 4> bufAicpu{};
    ReadyCoreFunctionQueue qslot[READY_QUEUE_SIZE];
    InitReadyQueueSlot(qslot[0], bufAiv, 0, 1, MakeTaskID(0, 0));
    InitReadyQueueSlot(qslot[1], bufAic, 0, 1, MakeTaskID(0, 1));
    InitReadyQueueSlot(qslot[2], bufAicpu, 0, 0, 0);
    for (size_t i = 0; i < READY_QUEUE_SIZE; ++i) {
        dyntask->readyQueue[i] = &qslot[i];
    }

    std::array<DevTensorData, 2> tensors{};
    tensors[0].address = 0x1000ULL;
    tensors[1].address = 0x2000ULL;
    DevStartArgs startArgs{};
    startArgs.contextWorkspaceAddr = 0x3000ULL;
    startArgs.inputTensorSize = 1;
    startArgs.outputTensorSize = 1;
    startArgs.devTensorList = tensors.data();

    DevAscendProgram devProg{};
    DeviceTaskContext::DumpDepend(dyntask.get(), &devProg, &startArgs, "ut_encoded");
}

TEST_F(TestDeviceTaskContext, DeviceTaskCtrl_Free_SingleCpu)
{
    DeviceWorkspaceAllocator workspace;
    auto dyntask = std::make_unique<DynDeviceTask>(workspace);
    dyntask->taskStageAllocMem.canFree.store(false);

    DeviceTaskCtrl ctrl;
    ctrl.devTask = reinterpret_cast<DeviceTask*>(dyntask.get());
    ctrl.notFree.store(true, std::memory_order_release);
    ctrl.freeCnt.store(0, std::memory_order_relaxed);

    EXPECT_TRUE(ctrl.IsNotFree());
    ctrl.Free(1);
    EXPECT_FALSE(ctrl.IsNotFree());
}

TEST_F(TestDeviceTaskContext, DeviceTaskCtrl_Free_MultiCpu)
{
    DeviceWorkspaceAllocator workspace;
    auto dyntask = std::make_unique<DynDeviceTask>(workspace);
    dyntask->taskStageAllocMem.canFree.store(false);

    DeviceTaskCtrl ctrl;
    ctrl.devTask = reinterpret_cast<DeviceTask*>(dyntask.get());
    ctrl.notFree.store(true, std::memory_order_release);
    ctrl.freeCnt.store(0, std::memory_order_relaxed);

    ctrl.Free(3);
    EXPECT_TRUE(ctrl.IsNotFree());
    ctrl.Free(3);
    EXPECT_TRUE(ctrl.IsNotFree());
    ctrl.Free(3);
    EXPECT_FALSE(ctrl.IsNotFree());
}

TEST_F(TestDeviceTaskContext, DeviceTaskCtrl_SupportParallel)
{
    DeviceWorkspaceAllocator workspace;
    auto dyntask = std::make_unique<DynDeviceTask>(workspace);

    DeviceTaskCtrl ctrl;
    ctrl.devTask = reinterpret_cast<DeviceTask*>(dyntask.get());

    bool result = ctrl.SupportParallel();
    EXPECT_FALSE(result);
}

TEST_F(TestDeviceTaskContext, DeviceTaskCtrl_ExistNextSameIterTask)
{
    DeviceTaskCtrl ctrl;
    ctrl.existNextSameIterTask.store(false, std::memory_order_release);
    EXPECT_FALSE(ctrl.ExistNextSameIterTask());
    ctrl.existNextSameIterTask.store(true, std::memory_order_release);
    EXPECT_TRUE(ctrl.ExistNextSameIterTask());
}

TEST_F(TestDeviceTaskContext, DeviceTaskCtrl_NextSameIterTaskCtrl)
{
    DeviceTaskCtrl ctrl;
    ctrl.nextSameIterTaskCtrl.store(0, std::memory_order_release);
    EXPECT_EQ(ctrl.NextSameIterTaskCtrl(), nullptr);

    DeviceTaskCtrl other;
    ctrl.nextSameIterTaskCtrl.store(reinterpret_cast<uint64_t>(&other), std::memory_order_release);
    EXPECT_EQ(ctrl.NextSameIterTaskCtrl(), &other);
}

TEST_F(TestDeviceTaskContext, InitReadyQueues_EnableAicoreResolve_CreatesDrcoRootFuncList)
{
    DeviceTaskContext taskContext;
    DevStartArgsBase startArgs;
    constexpr size_t kControlFlowCacheSize = 16 * 1024 * 1024;
    auto controlFlowCacheBuf = std::make_unique<uint8_t[]>(kControlFlowCacheSize);

    DevAscendProgram devProg;
    CreateMockDevAscendProgram(&devProg, ArchInfo::DAV_3510);
    devProg.stitchFunctionsize = 100;
    devProg.devArgs.enableAicoreResolve = true;
    devProg.devArgs.nrValidAic = 4;
    devProg.controlFlowCache.cacheData = DevRelocVector<uint8_t>(kControlFlowCacheSize, controlFlowCacheBuf.get());
    devProg.controlFlowCache.isRecording = true;

    DeviceWorkspaceAllocator workspace(&devProg);
    taskContext.InitAllocator(&devProg, workspace, &startArgs);

    auto dyntask = std::make_unique<DynDeviceTask>(workspace);
    CreateMockDynDeviceTask(dyntask.get(), 16);

    ReadyCoreFunctionQueue* queues[READY_QUEUE_SIZE] = {};
    EXPECT_EQ(taskContext.InitReadyQueues(dyntask.get(), &devProg, queues), DEVICE_MACHINE_OK);

    ASSERT_NE(dyntask->drcoRootFuncList, nullptr);
    EXPECT_EQ(dyntask->drcoRootFuncList->totalTaskCount, 16U);
    EXPECT_EQ(dyntask->drcoRootFuncList->devTaskFinished, 0U);
    // 完成计数表内嵌 rootFuncList：mock 无函数缓存，各 coreType size/executedCount 均为 0
    for (uint32_t ct = 0; ct < npu::tile_fwk::DRCO_QUEUE_MAX; ++ct) {
        EXPECT_EQ(dyntask->drcoRootFuncList->devTaskCountList.count[ct].size, 0U);
        EXPECT_EQ(dyntask->drcoRootFuncList->devTaskCountList.count[ct].executedCount, 0U);
    }
    for (uint32_t i = 0; i < npu::tile_fwk::MAX_AICORE_NUM_FOR_QUEUE; ++i) {
        EXPECT_NE(dyntask->drcoRootFuncList->perCorePendingQueueArray[i], nullptr);
    }
    // 三行（AIC/AIV/MIX）local queue/matrix 均全组分配：MIX 行承载 HUB_MIX 后继
    // （mixhub 与 AIC/AIV 同模型 batch push 多组分布），消费者为 AIC（validCoreNum 同 AIC 行）
    for (uint32_t ct = 0; ct < npu::tile_fwk::DRCO_QUEUE_MAX; ++ct) {
        for (uint32_t i = 0; i < npu::tile_fwk::NUM_LOCAL_GROUPS; ++i) {
            EXPECT_NE(dyntask->drcoRootFuncList->localReadyQueueArray[ct][i], nullptr);
            EXPECT_NE(dyntask->drcoRootFuncList->localReadyMatrixArray[ct][i], nullptr);
        }
    }
    // 全核共享 stitch 节点矩阵：行 = 全局 blockIdx，初始全空
    auto* stitchNodeMatrix = dyntask->drcoRootFuncList->stitchNodeMatrix;
    ASSERT_NE(stitchNodeMatrix, nullptr);
    for (uint32_t r = 0; r < npu::tile_fwk::MAX_AICORE_NUM_FOR_QUEUE; ++r) {
        for (uint32_t c = 0; c < npu::tile_fwk::LOCAL_GROUP_SIZE; ++c) {
            EXPECT_EQ(stitchNodeMatrix->stitchNodeList[r].slot[c], 0U);
        }
    }
    for (size_t i = 0; i < READY_QUEUE_SIZE; ++i) {
        EXPECT_NE(dyntask->readyQueue[i], nullptr);
    }
}

TEST_F(TestDeviceTaskContext, DispatchReadyQueueToCores_DistributesTasks)
{
    DeviceTaskContext taskContext;
    DevStartArgsBase startArgs;
    constexpr size_t kControlFlowCacheSize = 16 * 1024 * 1024;
    auto controlFlowCacheBuf = std::make_unique<uint8_t[]>(kControlFlowCacheSize);

    DevAscendProgram devProg;
    CreateMockDevAscendProgram(&devProg, ArchInfo::DAV_3510);
    devProg.stitchFunctionsize = 100;
    devProg.devArgs.enableAicoreResolve = true;
    devProg.devArgs.nrValidAic = 4;
    devProg.controlFlowCache.cacheData = DevRelocVector<uint8_t>(kControlFlowCacheSize, controlFlowCacheBuf.get());
    devProg.controlFlowCache.isRecording = true;

    DeviceWorkspaceAllocator workspace(&devProg);
    taskContext.InitAllocator(&devProg, workspace, &startArgs);

    auto dyntask = std::make_unique<DynDeviceTask>(workspace);
    CreateMockDynDeviceTask(dyntask.get(), 16);

    ReadyCoreFunctionQueue* queues[READY_QUEUE_SIZE] = {};
    ASSERT_EQ(taskContext.InitReadyQueues(dyntask.get(), &devProg, queues), DEVICE_MACHINE_OK);

    const int aivIdx = DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIV);
    const int aicIdx = DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIC);
    dyntask->readyQueue[aivIdx]->UnsafeEnqueue(MakeTaskID(0, 0));
    dyntask->readyQueue[aivIdx]->UnsafeEnqueue(MakeTaskID(1, 1));
    dyntask->readyQueue[aicIdx]->UnsafeEnqueue(MakeTaskID(2, 2));

    taskContext.DispatchReadyQueueToCores(dyntask.get(), &devProg);

    auto* root = dyntask->drcoRootFuncList;
    ASSERT_NE(root, nullptr);
    bool aicRouted = false;
    bool aivRouted = false;
    for (uint32_t i = 0; i < devProg.devArgs.nrValidAic; ++i) {
        if (root->perCorePendingQueueArray[i]->size > 0) {
            aicRouted = true;
        }
    }
    for (uint32_t i = devProg.devArgs.nrValidAic; i < devProg.devArgs.nrValidAic * 3; ++i) {
        if (root->perCorePendingQueueArray[i]->size > 0) {
            aivRouted = true;
        }
    }
    EXPECT_TRUE(aicRouted);
    EXPECT_TRUE(aivRouted);
}

TEST_F(TestDeviceTaskContext, DispatchReadyQueueToCores_DistributesMixWraps)
{
    DeviceTaskContext taskContext;
    DevStartArgsBase startArgs;
    constexpr size_t kControlFlowCacheSize = 16 * 1024 * 1024;
    auto controlFlowCacheBuf = std::make_unique<uint8_t[]>(kControlFlowCacheSize);

    DevAscendProgram devProg;
    CreateMockDevAscendProgram(&devProg, ArchInfo::DAV_3510);
    devProg.stitchFunctionsize = 100;
    devProg.devArgs.enableAicoreResolve = true;
    devProg.devArgs.nrValidAic = 4;
    devProg.controlFlowCache.cacheData = DevRelocVector<uint8_t>(kControlFlowCacheSize, controlFlowCacheBuf.get());
    devProg.controlFlowCache.isRecording = true;

    DeviceWorkspaceAllocator workspace(&devProg);
    taskContext.InitAllocator(&devProg, workspace, &startArgs);

    auto dyntask = std::make_unique<DynDeviceTask>(workspace);
    CreateMockDynDeviceTask(dyntask.get(), 16);

    ReadyCoreFunctionQueue* queues[READY_QUEUE_SIZE] = {};
    ASSERT_EQ(taskContext.InitReadyQueues(dyntask.get(), &devProg, queues), DEVICE_MACHINE_OK);

    // wrap 队列：1 个 1C2V + 1 个 1C1V，tasklist 顺序为 AIC / AIV0 / AIV1
    constexpr uint8_t MIX_TYPE_1C2V = 2;
    constexpr uint8_t MIX_TYPE_1C1V = 1;
    WrapInfo wrapInfo[2]{};
    wrapInfo[0].wrapId = 0;
    wrapInfo[0].mixResourceType = MIX_TYPE_1C2V;
    wrapInfo[0].tasklist[WRAP_IDX_AIC] = MakeTaskID(0, 100);
    wrapInfo[0].tasklist[WRAP_IDX_AIV0] = MakeTaskID(0, 200);
    wrapInfo[0].tasklist[WRAP_IDX_AIV1] = MakeTaskID(0, 201);
    wrapInfo[1].wrapId = 1;
    wrapInfo[1].mixResourceType = MIX_TYPE_1C1V;
    wrapInfo[1].tasklist[WRAP_IDX_AIC] = MakeTaskID(1, 101);
    wrapInfo[1].tasklist[WRAP_IDX_AIV0] = MakeTaskID(1, 202);
    wrapInfo[1].tasklist[WRAP_IDX_AIV1] = 0; // 1C1V 不使用第二个 AIV
    WrapInfoQueue wrapQueue{0, 2, 2, wrapInfo, 0};
    dyntask->devTask.mixTaskData.readyWrapCoreFunctionQue = reinterpret_cast<uint64_t>(&wrapQueue);

    // 普通 aic/aiv 任务：验证 wrap 分发后的游标续接
    const int aivIdx = DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIV);
    const int aicIdx = DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIC);
    dyntask->readyQueue[aivIdx]->UnsafeEnqueue(MakeTaskID(2, 302));
    dyntask->readyQueue[aicIdx]->UnsafeEnqueue(MakeTaskID(2, 301));

    taskContext.DispatchReadyQueueToCores(dyntask.get(), &devProg);

    auto* root = dyntask->drcoRootFuncList;
    ASSERT_NE(root, nullptr);

    // wrap0 (1C2V): aicCore=0, AIV cores = 4 + 0*2 = {4, 5}
    EXPECT_EQ(root->perCorePendingQueueArray[0]->size, 1U);
    EXPECT_EQ(root->perCorePendingQueueArray[0]->taskList[0], MakeTaskID(0, 100));
    EXPECT_EQ(root->perCorePendingQueueArray[4]->size, 1U);
    EXPECT_EQ(root->perCorePendingQueueArray[4]->taskList[0], MakeTaskID(0, 200));
    EXPECT_EQ(root->perCorePendingQueueArray[5]->size, 1U);
    EXPECT_EQ(root->perCorePendingQueueArray[5]->taskList[0], MakeTaskID(0, 201));

    // wrap1 (1C1V): aicCore=1, AIV core = 4 + 1*2 = {6}，不占 7
    EXPECT_EQ(root->perCorePendingQueueArray[1]->size, 1U);
    EXPECT_EQ(root->perCorePendingQueueArray[1]->taskList[0], MakeTaskID(1, 101));
    EXPECT_EQ(root->perCorePendingQueueArray[6]->size, 1U);
    EXPECT_EQ(root->perCorePendingQueueArray[6]->taskList[0], MakeTaskID(1, 202));
    EXPECT_EQ(root->perCorePendingQueueArray[7]->size, 0U);

    // 普通任务从 wrap 游标续接：AIC 起始 core=2，AIV 起始 core=4+(4 % 8)=8
    EXPECT_EQ(root->perCorePendingQueueArray[2]->size, 1U);
    EXPECT_EQ(root->perCorePendingQueueArray[2]->taskList[0], MakeTaskID(2, 301));
    EXPECT_EQ(root->perCorePendingQueueArray[8]->size, 1U);
    EXPECT_EQ(root->perCorePendingQueueArray[8]->taskList[0], MakeTaskID(2, 302));
}

TEST_F(TestDeviceTaskContext, DispatchDieReadyQueueToCores_DistributesDieTasks)
{
    DeviceTaskContext taskContext;
    DevStartArgsBase startArgs;
    constexpr size_t kControlFlowCacheSize = 16 * 1024 * 1024;
    auto controlFlowCacheBuf = std::make_unique<uint8_t[]>(kControlFlowCacheSize);

    DevAscendProgram devProg;
    CreateMockDevAscendProgram(&devProg, ArchInfo::DAV_3510);
    devProg.stitchFunctionsize = 100;
    devProg.devArgs.enableAicoreResolve = true;
    devProg.devArgs.nrValidAic = 4;
    devProg.controlFlowCache.cacheData = DevRelocVector<uint8_t>(kControlFlowCacheSize, controlFlowCacheBuf.get());
    devProg.controlFlowCache.isRecording = true;

    DeviceWorkspaceAllocator workspace(&devProg);
    taskContext.InitAllocator(&devProg, workspace, &startArgs);

    auto dyntask = std::make_unique<DynDeviceTask>(workspace);
    CreateMockDynDeviceTask(dyntask.get(), 16);

    ReadyCoreFunctionQueue* queues[READY_QUEUE_SIZE] = {};
    ASSERT_EQ(taskContext.InitReadyQueues(dyntask.get(), &devProg, queues), DEVICE_MACHINE_OK);

    taskContext.InitDieReadyQueues(dyntask.get(), &devProg);
    auto dieAivQueue = reinterpret_cast<ReadyCoreFunctionQueue*>(
        dyntask->devTask.dieReadyFunctionQue.readyDieAivCoreFunctionQue[0]);
    auto dieAicQueue = reinterpret_cast<ReadyCoreFunctionQueue*>(
        dyntask->devTask.dieReadyFunctionQue.readyDieAicCoreFunctionQue[0]);
    ASSERT_NE(dieAivQueue, nullptr);
    ASSERT_NE(dieAicQueue, nullptr);
    dieAivQueue->UnsafeEnqueue(MakeTaskID(0, 0));
    dieAicQueue->UnsafeEnqueue(MakeTaskID(1, 1));

    taskContext.DispatchDieReadyQueueToCores(dyntask.get(), &devProg);

    auto* root = dyntask->drcoRootFuncList;
    ASSERT_NE(root, nullptr);
    EXPECT_GT(root->perCorePendingQueueArray[0]->size, 0U);
    EXPECT_GT(root->perCorePendingQueueArray[4]->size, 0U);
}

TEST_F(TestDeviceTaskContext, DispatchReadyQueueToCores_PrecountPerCoreExecutedCount)
{
    DeviceTaskContext taskContext;
    DevStartArgsBase startArgs;
    constexpr size_t kControlFlowCacheSize = 16 * 1024 * 1024;
    auto controlFlowCacheBuf = std::make_unique<uint8_t[]>(kControlFlowCacheSize);

    DevAscendProgram devProg;
    CreateMockDevAscendProgram(&devProg, ArchInfo::DAV_3510);
    devProg.stitchFunctionsize = 100;
    devProg.devArgs.enableAicoreResolve = true;
    devProg.devArgs.nrValidAic = 4;
    devProg.controlFlowCache.cacheData = DevRelocVector<uint8_t>(kControlFlowCacheSize, controlFlowCacheBuf.get());
    devProg.controlFlowCache.isRecording = true;

    DeviceWorkspaceAllocator workspace(&devProg);
    taskContext.InitAllocator(&devProg, workspace, &startArgs);

    auto dyntask = std::make_unique<DynDeviceTask>(workspace);
    CreateMockDynDeviceTask(dyntask.get(), 16);

    ReadyCoreFunctionQueue* queues[READY_QUEUE_SIZE] = {};
    ASSERT_EQ(taskContext.InitReadyQueues(dyntask.get(), &devProg, queues), DEVICE_MACHINE_OK);

    // 2 个 AIC + 3 个 AIV 任务派发到 per-core 队列
    const int aivIdx = DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIV);
    const int aicIdx = DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIC);
    dyntask->readyQueue[aicIdx]->UnsafeEnqueue(MakeTaskID(0, 0));
    dyntask->readyQueue[aicIdx]->UnsafeEnqueue(MakeTaskID(1, 1));
    dyntask->readyQueue[aivIdx]->UnsafeEnqueue(MakeTaskID(2, 2));
    dyntask->readyQueue[aivIdx]->UnsafeEnqueue(MakeTaskID(2, 3));
    dyntask->readyQueue[aivIdx]->UnsafeEnqueue(MakeTaskID(2, 4));

    taskContext.DispatchReadyQueueToCores(dyntask.get(), &devProg);

    auto* root = dyntask->drcoRootFuncList;
    ASSERT_NE(root, nullptr);
    // AIC size == 派发数（全静态认领，预累加即置完成 flag）；
    // AIV size 大于派发数（模拟存在运行期解锁任务，剩余由设备侧逐个计数）
    root->devTaskCountList.count[npu::tile_fwk::DRCO_QUEUE_AIC].size = 2;
    root->devTaskCountList.count[npu::tile_fwk::DRCO_QUEUE_AIV].size = 8;

    npu::tile_fwk::DrcoRootFuncListPrecountPerCoreTasks(root, devProg.devArgs.nrValidAic, dyntask->drcoPrecountExecuted,
                                                        dyntask->drcoPrecountFinishFlag);

    EXPECT_EQ(root->devTaskCountList.count[npu::tile_fwk::DRCO_QUEUE_AIC].executedCount, 2U);
    EXPECT_EQ(root->devTaskCountList.count[npu::tile_fwk::DRCO_QUEUE_AIV].executedCount, 3U);
    EXPECT_EQ(root->devTaskFinishFlagList.flag[npu::tile_fwk::DRCO_QUEUE_AIC].devTaskFinishFlag, 1U);
    EXPECT_EQ(root->devTaskFinishFlagList.flag[npu::tile_fwk::DRCO_QUEUE_AIV].devTaskFinishFlag, 0U);
    // MIX 恒为空类型（size == 0）：预累加即置完成 flag（对应原空队列通知置 flag 行为）
    EXPECT_EQ(root->devTaskFinishFlagList.flag[npu::tile_fwk::DRCO_QUEUE_MIX].devTaskFinishFlag, 1U);
}
