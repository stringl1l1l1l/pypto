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
 * \file test_dev_encode_program_ctrlflow_cache.cpp
 * \brief Unit tests for the aicore-resolve (drco) branches in DevControlFlowCache.
 */
#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include <memory>
#include <vector>

#define private public
#define protected public
#include "machine/device/dynamic/context/device_task_context.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "machine/utils/dynamic/device_task.h"
#include "machine/utils/dynamic/dev_encode_program_ctrlflow_cache.h"
#include "machine/utils/dynamic/dev_encode_function_dupped_data.h"
#include "machine/utils/dynamic/dev_start_args.h"
#include "machine/utils/queues.h"
#include "interface/machine/device/tilefwk/aikernel_device_task.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/tilefwk.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

namespace {

constexpr size_t kCtrlCacheSize = 512 * 1024;

// A host-side DrcoRootFuncList that mimics what InitDrcoRootFuncList builds:
// all per-core/local queues are backed by real memory, except the last entry of
// each array which stays null to exercise the null-continue branches.
struct DrcoQueueFixture {
    DrcoRootFuncList root{};
    std::vector<uint8_t> perCoreStorage;
    std::vector<uint8_t> localStorage;

    void Build(uint32_t coreFunctionCnt = 8)
    {
        const size_t perCoreBytes = sizeof(PerCorePendingQueue) + 16 * sizeof(LeafTaskId);
        const size_t localBytes = sizeof(DrcoLocalReadyQueue) + 16 * sizeof(LeafTaskId);
        constexpr size_t localAlign = alignof(DrcoLocalReadyQueue);
        perCoreStorage.assign(MAX_AICORE_NUM_FOR_QUEUE * perCoreBytes, 0);
        localStorage.assign(DRCO_QUEUE_MAX * NUM_LOCAL_GROUPS * localBytes + localAlign, 0);
        uintptr_t localBase = reinterpret_cast<uintptr_t>(localStorage.data());
        uint8_t* localAlignedBase = reinterpret_cast<uint8_t*>((localBase + localAlign - 1) &
                                                               ~static_cast<uintptr_t>(localAlign - 1));
        for (uint32_t i = 0; i < MAX_AICORE_NUM_FOR_QUEUE; ++i) {
            if (i == MAX_AICORE_NUM_FOR_QUEUE - 1) {
                root.perCorePendingQueueArray[i] = nullptr;
                continue;
            }
            root.perCorePendingQueueArray[i] = reinterpret_cast<PerCorePendingQueue*>(perCoreStorage.data() +
                                                                                      i * perCoreBytes);
        }
        for (uint32_t ct = 0; ct < DRCO_QUEUE_MAX; ++ct) {
            for (uint32_t i = 0; i < NUM_LOCAL_GROUPS; ++i) {
                if (ct == DRCO_QUEUE_MAX - 1 && i == NUM_LOCAL_GROUPS - 1) {
                    root.localReadyQueueArray[ct][i] = nullptr;
                    continue;
                }
                root.localReadyQueueArray[ct][i] = reinterpret_cast<DrcoLocalReadyQueue*>(
                    localAlignedBase + (ct * NUM_LOCAL_GROUPS + i) * localBytes);
            }
        }
        root.totalTaskCount = coreFunctionCnt;
        root.devTaskFinished = 0;
    }
};

// Sets up the three ready queues (AIV/AIC/AICPU) on a DynDeviceTask.
void SetupReadyQueues(DynDeviceTask* dyntask, std::array<std::array<uint32_t, 16>, READY_QUEUE_SIZE>& elemBuf,
                      std::array<std::unique_ptr<ReadyCoreFunctionQueue>, READY_QUEUE_SIZE>& queue)
{
    for (size_t i = 0; i < READY_QUEUE_SIZE; ++i) {
        queue[i] = std::make_unique<ReadyCoreFunctionQueue>(16, elemBuf[i].data());
        dyntask->readyQueue[i] = queue[i].get();
    }
}

// Builds a DynFuncHeader with funcNum entries followed by the DynFuncData array.
void SetupDynFuncHeader(DynDeviceTask* dyntask,
                        std::array<uint8_t, sizeof(DynFuncHeader) + 8 * sizeof(DynFuncData)>& hdrBuf,
                        uint32_t funcNum = 1)
{
    auto* header = reinterpret_cast<DynFuncHeader*>(hdrBuf.data());
    header->seqNo = 0;
    header->funcNum = funcNum;
    header->funcSize = static_cast<uint32_t>(sizeof(DynFuncHeader) + funcNum * sizeof(DynFuncData));
    dyntask->dynFuncDataList = header;
}

// Builds a minimal DevAscendFunctionDuppedData with one leaf op and no stitches.
DevAscendFunctionDuppedData* SetupDuppedData(std::array<uint8_t, 1024>& dupBuf, uint32_t opCount = 1)
{
    auto* duppedData = reinterpret_cast<DevAscendFunctionDuppedData*>(dupBuf.data());
    duppedData->source_ = nullptr;
    duppedData->operationList_.size = opCount;
    duppedData->operationList_.predCountBase = static_cast<uint32_t>(sizeof(DevAscendFunctionDuppedData));
    duppedData->operationList_.stitchCount = 0;
    return duppedData;
}

void SetupCtrlCache(DevControlFlowCache& ctrl, std::vector<uint8_t>& cacheBuf, size_t cacheSize = kCtrlCacheSize)
{
    cacheBuf.assign(cacheSize, 0);
    ctrl.cacheData = DevRelocVector<uint8_t>(static_cast<int>(cacheSize), cacheBuf.data());
    ctrl.cacheDataOffset = 0;
}

// Packs {kind, value} into the 64-bit cache-form word stored in rawTensorAddrBackup arrays.
uint64_t CacheFormWord(AddressCacheKind kind, uint64_t value)
{
    AddressDescriptor desc = AddressDescriptor::MakeCache(kind, value);
    uint64_t word = 0;
    std::memcpy(&word, &desc, sizeof(word));
    return word;
}

// Builds a DevAscendFunctionDuppedData whose incast/outcast AddressDescriptor arrays
// live at data_[0..]: incast at base 0, outcast right behind the incast entries.
DevAscendFunctionDuppedData* SetupIncastOutcastDup(std::array<uint8_t, 1024>& dupBuf, uint32_t incastNum,
                                                   uint32_t outcastNum)
{
    auto* duppedData = reinterpret_cast<DevAscendFunctionDuppedData*>(dupBuf.data());
    duppedData->source_ = nullptr;
    duppedData->incastList_.size = incastNum;
    duppedData->incastList_.base = 0;
    duppedData->outcastList_.size = outcastNum;
    duppedData->outcastList_.base = incastNum * static_cast<uint32_t>(sizeof(AddressDescriptor));
    return duppedData;
}

// Single-task/single-dup fixture for the incast/outcast reloc table: dup0 carries
// 2 incast (Input#1, Workspace@kWsOffset) and 2 outcast (Output#0, Comm) descriptors,
// so the built table holds 4 batches in kind order Workspace/Input/Output/Comm.
// DevControlFlowCache is placement-new'ed at the blob start because the reloc table
// stores every pointer as an offset from `this` (same layout as production).
struct RelocTableFixture {
    static constexpr uint64_t kDstWorkspace = 0x100000;
    static constexpr uint64_t kWsOffset = 0x2000;
    static constexpr uint64_t kCommAddr = (1ULL << 58) | 0x1234;
    static constexpr uint64_t kInputAddr = 0xAAA1000;
    static constexpr uint64_t kOutputAddr = 0xAAA2000;

    DeviceWorkspaceAllocator workspace;
    std::unique_ptr<DynDeviceTask> dyntask{std::make_unique<DynDeviceTask>(workspace)};
    DevAscendFunctionDuppedData* duppedData{nullptr};
    DynFuncHeader* header{nullptr};
    uint64_t* live{nullptr};
    uint64_t* backup{nullptr};
    DevControlFlowCache* ctrl{nullptr};
    DeviceTaskCache entry{};
    std::array<DevTensorData, 3> tensors{};
    DevStartArgs args{};

    RelocTableFixture()
    {
        blob.assign(512 * 1024, 0);
        ctrl = new (blob.data()) DevControlFlowCache();
        ctrl->cacheData = DevRelocVector<uint8_t>(static_cast<int>(blob.size()), blob.data());
        ctrl->cacheDataOffset = (sizeof(DevControlFlowCache) + CFGCACHE_ALIGN - 1) / CFGCACHE_ALIGN * CFGCACHE_ALIGN;

        header = static_cast<DynFuncHeader*>(ctrl->AllocateCache(sizeof(DynFuncHeader) + sizeof(DynFuncData)));
        header->seqNo = 0;
        header->funcNum = 1;
        header->funcSize = static_cast<uint32_t>(sizeof(DynFuncHeader) + sizeof(DynFuncData));
        live = static_cast<uint64_t*>(ctrl->AllocateCache(4 * sizeof(uint64_t)));
        backup = static_cast<uint64_t*>(ctrl->AllocateCache(4 * sizeof(uint64_t)));
        header->At(0).rawTensorAddr = live;

        duppedData = SetupIncastOutcastDup(dupBuf, 2, 2);
        dyntask->dynFuncDataList = header;
        dyntask->dynFuncDataCacheList[0].duppedData = duppedData;
        dyntask->dynFuncDataBackupList[0].rawTensorAddrBackup = backup;
        backup[0] = CacheFormWord(AddressCacheKind::Input, 1);
        backup[1] = CacheFormWord(AddressCacheKind::Workspace, kWsOffset);
        backup[2] = CacheFormWord(AddressCacheKind::Output, 0);
        backup[3] = CacheFormWord(AddressCacheKind::Communication, kCommAddr);

        entry.dynTaskBase = dyntask.get();
        ctrl->deviceTaskCacheList = DevRelocVector<DeviceTaskCache>(1, &entry);
        ctrl->deviceTaskCount = 1;

        tensors[1].address = kInputAddr;
        tensors[2].address = kOutputAddr;
        args.devTensorList = tensors.data();
        args.inputTensorSize = 2;
        args.outputTensorSize = 1;
        args.contextWorkspaceAddr = kDstWorkspace;
    }

private:
    std::vector<uint8_t> blob;
    std::array<uint8_t, 1024> dupBuf{};
};

} // namespace

TEST(CtrlFlowCacheDrcoUt, PredCountDataRestore_CoversDrcoPredCount)
{
    DeviceWorkspaceAllocator workspace;
    auto dyntask = std::make_unique<DynDeviceTask>(workspace);

    std::array<uint8_t, sizeof(DynFuncHeader) + 8 * sizeof(DynFuncData)> hdrBuf{};
    SetupDynFuncHeader(dyntask.get(), hdrBuf, 1);

    std::array<uint8_t, 1024> dupBuf{};
    DynFuncDataCache& cache = dyntask->dynFuncDataCacheList[0];
    cache.duppedData = SetupDuppedData(dupBuf, 1);
    cache.predCountPingPong[PRED_COUNT_PING] = nullptr;
    cache.predCountPingPong[PRED_COUNT_PONG] = nullptr;
    cache.predCount = nullptr;
    cache.calleeList = nullptr;
    cache.devFunc = nullptr;

    dyntask->dynFuncDataBackupList[0] = DynFuncDataBackup{};
    std::array<uint8_t, 64> predCountBackup{};
    dyntask->dynFuncDataBackupList[0].predCountBackup = reinterpret_cast<predcount_t*>(predCountBackup.data());

    std::array<uint8_t, 64> predCountDest{};
    dyntask->dynFuncDataList->At(0).drcoRootFuncData.predCount = reinterpret_cast<int32_t*>(predCountDest.data());

    std::vector<uint8_t> cacheBuf;
    DevControlFlowCache ctrl;
    SetupCtrlCache(ctrl, cacheBuf);

    ctrl.PredCountPingPongSwap(dyntask.get());
    ctrl.DrcoPredCountDataRestore(dyntask.get());
    ctrl.BitmapDataRestoreTask(dyntask.get());
    SUCCEED();
}

TEST(CtrlFlowCacheDrcoUt, PredCountPingPongRoundTrip)
{
    DeviceWorkspaceAllocator workspace;
    auto dyntask = std::make_unique<DynDeviceTask>(workspace);

    std::array<uint8_t, sizeof(DynFuncHeader) + 8 * sizeof(DynFuncData)> hdrBuf{};
    SetupDynFuncHeader(dyntask.get(), hdrBuf, 2);

    std::array<std::array<uint8_t, 1024>, 2> dupBuf{};
    std::array<DevAscendFunctionDuppedData*, 2> dupped{};
    /* 16KB on the stack exceeds the -Wframe-larger-than budget; scratch does not need stack storage. */
    static std::array<std::array<uint8_t, 8192>, 2> devFuncBuf{};
    const uint32_t opCounts[2] = {3, 4};
    const predcount_t initVals[2][4] = {{3, 1, 4, 0}, {5, 9, 2, 6}};
    for (int i = 0; i < 2; ++i) {
        dupped[i] = SetupDuppedData(dupBuf[i], opCounts[i]);
        DynFuncDataCache& cache = dyntask->dynFuncDataCacheList[i];
        cache.duppedData = dupped[i];
        cache.predCountPingPong[PRED_COUNT_PING] = dupped[i]->GetOperationPredCountPingPong(PRED_COUNT_PING);
        cache.predCountPingPong[PRED_COUNT_PONG] = dupped[i]->GetOperationPredCountPingPong(PRED_COUNT_PONG);
        cache.predCount = cache.predCountPingPong[PRED_COUNT_PING];
        cache.devFunc = reinterpret_cast<DevAscendFunction*>(devFuncBuf[i].data());
        cache.calleeList = nullptr;
        dyntask->dynFuncDataBackupList[i] = DynFuncDataBackup{};
        for (uint32_t j = 0; j < opCounts[i]; ++j) {
            dupped[i]->GetOperationCurrPredCount(j) = initVals[i][j];
        }
    }

    std::vector<uint8_t> cacheBuf;
    DevControlFlowCache ctrl;
    SetupCtrlCache(ctrl, cacheBuf);

    ctrl.PredCountDataBackup(dyntask.get());

    // Record: both slices carry the initial snapshot; predCount aliases slice 0.
    for (int i = 0; i < 2; ++i) {
        for (uint32_t j = 0; j < opCounts[i]; ++j) {
            EXPECT_EQ(dyntask->dynFuncDataCacheList[i].predCountPingPong[PRED_COUNT_PING][j], initVals[i][j]);
            EXPECT_EQ(dyntask->dynFuncDataCacheList[i].predCountPingPong[PRED_COUNT_PONG][j], initVals[i][j]);
        }
    }

    // Simulate back-to-back launches with the shadow-restore protocol: pre-push only swaps the
    // slice pointers onto the clean copy (no memcpy - it was refreshed by the previous launch's
    // shadow), sche dirties it during execution, and the post-push shadow refreshes the OTHER
    // slice for the next launch. Every launch must consume initial values.
    auto runLaunch = [&]() {
        ctrl.PredCountPingPongSwap(dyntask.get());
        for (int i = 0; i < 2; ++i) {
            EXPECT_EQ(dyntask->dynFuncDataCacheList[i].predCount,
                      dyntask->dynFuncDataCacheList[i].predCountPingPong[PRED_COUNT_PING]);
            for (uint32_t j = 0; j < opCounts[i]; ++j) {
                EXPECT_EQ(dyntask->dynFuncDataCacheList[i].predCount[j], initVals[i][j]);
            }
        }
        for (int i = 0; i < 2; ++i) {
            for (uint32_t j = 0; j < opCounts[i]; ++j) {
                dyntask->dynFuncDataCacheList[i].predCount[j] -= 1;
            }
        }
        ctrl.PredCountPingPongRestore(dyntask.get());
        for (int i = 0; i < 2; ++i) {
            for (uint32_t j = 0; j < opCounts[i]; ++j) {
                EXPECT_EQ(dyntask->dynFuncDataCacheList[i].predCountPingPong[PRED_COUNT_PONG][j], initVals[i][j]);
            }
        }
    };

    runLaunch(); // consumes slice 1
    runLaunch(); // consumes slice 0 (cleans record-time dirt)
    runLaunch(); // consumes slice 1 again (previous dirt cleaned)
    runLaunch(); // consumes slice 0 again

    // The live pointers must never fall back to scattered state: they always alias slice[0].
    for (int i = 0; i < 2; ++i) {
        EXPECT_EQ(dyntask->dynFuncDataCacheList[i].predCount,
                  dyntask->dynFuncDataCacheList[i].predCountPingPong[PRED_COUNT_PING]);
    }
}

TEST(CtrlFlowCacheDrcoUt, ReadyQueuePingPongRoundTrip)
{
    DeviceWorkspaceAllocator workspace;
    auto dyntask = std::make_unique<DynDeviceTask>(workspace);
    dyntask->devTask.coreFunctionCnt = 8;

    std::array<std::array<uint32_t, 16>, READY_QUEUE_SIZE> elemBuf{};
    std::array<std::unique_ptr<ReadyCoreFunctionQueue>, READY_QUEUE_SIZE> queue{};
    SetupReadyQueues(dyntask.get(), elemBuf, queue);

    const int aivIdx = DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIV);
    const int aicIdx = DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIC);
    std::array<std::array<uint32_t, 16>, READY_QUEUE_SIZE> initial{};
    std::array<uint32_t, READY_QUEUE_SIZE> initSize{};
    initSize[aivIdx] = 2;
    initSize[aicIdx] = 1;
    initial[aivIdx][0] = MakeTaskID(0, 1);
    initial[aivIdx][1] = MakeTaskID(0, 2);
    initial[aicIdx][0] = MakeTaskID(1, 1);
    for (size_t i = 0; i < READY_QUEUE_SIZE; ++i) {
        for (uint32_t j = 0; j < initSize[i]; ++j) {
            queue[i]->UnsafeEnqueue(initial[i][j]);
        }
    }

    std::vector<uint8_t> cacheBuf;
    DevControlFlowCache ctrl;
    SetupCtrlCache(ctrl, cacheBuf);

    ctrl.ReadyQueueDataBackup(dyntask.get());
    ASSERT_NE(dyntask->readyQueueBackup, nullptr);

    // Record state: pingElem = the buffers the record launch will dirty; pongElem = clean
    // copies initialized from the snapshot for the first cached launch.
    std::array<uint32_t*, READY_QUEUE_SIZE> snapData{};
    for (size_t i = 0; i < READY_QUEUE_SIZE; ++i) {
        EXPECT_EQ(dyntask->readyQueueBackup->pingElem[i], elemBuf[i].data());
        EXPECT_NE(dyntask->readyQueueBackup->pongElem[i], elemBuf[i].data());
        snapData[i] = dyntask->readyQueueBackup->queueList[i].Data();
        for (uint32_t j = 0; j < initSize[i]; ++j) {
            EXPECT_EQ(dyntask->readyQueueBackup->pongElem[i][j], initial[i][j]);
        }
    }

    // Simulate back-to-back launches with the shadow-restore protocol: restore rebinds elem_ to
    // the clean buffer (no memcpy), sche consumes from the head and LIFO-tail re-appends
    // (dirtying the buffer), the post-push shadow refreshes the OTHER buffer. Every launch must
    // consume the initial snapshot; the AICPU queue stays empty (Size 0) to cover that path.
    std::array<uint32_t*, READY_QUEUE_SIZE> lastData{};
    for (size_t i = 0; i < READY_QUEUE_SIZE; ++i) {
        lastData[i] = queue[i]->Data();
    }
    auto runLaunch = [&]() {
        ctrl.ReadyQueueDataPingPongSwap(dyntask.get(), 4);
        for (size_t i = 0; i < READY_QUEUE_SIZE; ++i) {
            EXPECT_NE(queue[i]->Data(), lastData[i]);
            lastData[i] = queue[i]->Data();

            ASSERT_EQ(queue[i]->Size(), initSize[i]);
            uint32_t k = 0;
            for (const uint32_t* it = queue[i]->begin(); it != queue[i]->end(); ++it, ++k) {
                EXPECT_EQ(*it, initial[i][k]);
            }
        }
        // sche-side dirtying: head dequeue plus LIFO tail pull and re-append.
        for (size_t i = 0; i < READY_QUEUE_SIZE; ++i) {
            if (initSize[i] == 0) {
                continue;
            }
            uint32_t out[4];
            (void)queue[i]->Dequeue(1);
            (void)queue[i]->DequeueTail(1, out);
            queue[i]->UnsafeEnqueue(MakeTaskID(7, 7));
        }
        ctrl.ReadyQueueDataPingPongRestore(dyntask.get());
        // The buffer the NEXT launch consumes must be clean again.
        for (size_t i = 0; i < READY_QUEUE_SIZE; ++i) {
            const uint32_t* inactive = (queue[i]->Data() == dyntask->readyQueueBackup->pingElem[i]) ?
                                           dyntask->readyQueueBackup->pongElem[i] :
                                           dyntask->readyQueueBackup->pingElem[i];
            for (uint32_t j = 0; j < initSize[i]; ++j) {
                EXPECT_EQ(inactive[j], initial[i][j]);
            }
        }
    };

    runLaunch(); // consumes pongElem (initialized at backup)
    runLaunch(); // consumes pingElem (cleaned by launch 1's shadow)
    runLaunch(); // consumes pongElem again (previous dirt cleaned)
    runLaunch(); // consumes pingElem again

    // The snapshot buffers stay immutable across all rounds.
    for (size_t i = 0; i < READY_QUEUE_SIZE; ++i) {
        EXPECT_EQ(dyntask->readyQueueBackup->queueList[i].Data(), snapData[i]);
        for (uint32_t j = 0; j < initSize[i]; ++j) {
            EXPECT_EQ(snapData[i][j], initial[i][j]);
        }
    }
}

TEST(CtrlFlowCacheDrcoUt, ReadyQueueDataBackupRestore_WithDrco)
{
    DeviceWorkspaceAllocator workspace;
    auto dyntask = std::make_unique<DynDeviceTask>(workspace);
    dyntask->devTask.coreFunctionCnt = 8;

    DrcoQueueFixture drco;
    drco.Build();
    // mark one per-core queue non-empty so the backup memcpy path uses real size
    drco.root.perCorePendingQueueArray[3]->size = 1;
    drco.root.perCorePendingQueueArray[3]->taskList[0] = MakeTaskID(2, 0);
    dyntask->drcoRootFuncList = &drco.root;

    std::array<std::array<uint32_t, 16>, READY_QUEUE_SIZE> elemBuf{};
    std::array<std::unique_ptr<ReadyCoreFunctionQueue>, READY_QUEUE_SIZE> queue{};
    SetupReadyQueues(dyntask.get(), elemBuf, queue);

    const int aivIdx = DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIV);
    const int aicIdx = DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIC);
    queue[aivIdx]->UnsafeEnqueue(MakeTaskID(0, 1));
    queue[aivIdx]->UnsafeEnqueue(MakeTaskID(0, 2));
    queue[aicIdx]->UnsafeEnqueue(MakeTaskID(1, 1));

    std::vector<uint8_t> cacheBuf;
    DevControlFlowCache ctrl;
    SetupCtrlCache(ctrl, cacheBuf);

    ctrl.ReadyQueueDataBackup(dyntask.get());
    ASSERT_NE(dyntask->readyQueueBackup, nullptr);

    ctrl.ReadyQueueDataPingPongSwap(dyntask.get(), 4);
    SUCCEED();

    EXPECT_EQ(dyntask->devTask.coreFunctionCnt, 8U);
    EXPECT_EQ(drco.root.perCorePendingQueueArray[0]->size, 1U);
    uint32_t aivRouted = 0;
    for (uint32_t i = 4; i < 4 + 8; ++i) {
        aivRouted += drco.root.perCorePendingQueueArray[i]->size;
    }
    EXPECT_EQ(aivRouted, 2U);
    EXPECT_EQ(drco.root.devTaskFinished, 0U);
}

TEST(CtrlFlowCacheDrcoUt, DieReadyQueueDataBackupRestore_WithDrco)
{
    DeviceWorkspaceAllocator workspace;
    auto dyntask = std::make_unique<DynDeviceTask>(workspace);
    dyntask->devTask.coreFunctionCnt = 8;

    DrcoQueueFixture drco;
    drco.Build();
    dyntask->drcoRootFuncList = &drco.root;

    std::array<std::array<uint32_t, 16>, DIE_NUM> bufAiv{};
    std::array<std::array<uint32_t, 16>, DIE_NUM> bufAic{};
    std::array<std::unique_ptr<ReadyCoreFunctionQueue>, DIE_NUM> qAiv;
    std::array<std::unique_ptr<ReadyCoreFunctionQueue>, DIE_NUM> qAic;
    for (uint32_t i = 0; i < DIE_NUM; ++i) {
        qAiv[i] = std::make_unique<ReadyCoreFunctionQueue>(16, bufAiv[i].data());
        qAic[i] = std::make_unique<ReadyCoreFunctionQueue>(16, bufAic[i].data());
        dyntask->devTask.dieReadyFunctionQue.readyDieAivCoreFunctionQue[i] = reinterpret_cast<uint64_t>(qAiv[i].get());
        dyntask->devTask.dieReadyFunctionQue.readyDieAicCoreFunctionQue[i] = reinterpret_cast<uint64_t>(qAic[i].get());
        qAiv[i]->UnsafeEnqueue(MakeTaskID(i, 1));
        qAic[i]->UnsafeEnqueue(MakeTaskID(i, 2));
    }

    std::vector<uint8_t> cacheBuf;
    DevControlFlowCache ctrl;
    SetupCtrlCache(ctrl, cacheBuf);

    ctrl.DieReadyQueueDataBackup(dyntask.get());
    ASSERT_NE(dyntask->dieReadyQueueBackup, nullptr);

    ctrl.DieReadyQueuePingPongSwap(dyntask.get(), 4);
    SUCCEED();

    EXPECT_EQ(drco.root.perCorePendingQueueArray[0]->size, 1U);
    EXPECT_EQ(drco.root.perCorePendingQueueArray[2]->size, 1U);
    EXPECT_EQ(drco.root.perCorePendingQueueArray[4]->size, 1U);
    EXPECT_EQ(drco.root.perCorePendingQueueArray[8]->size, 1U);
}

TEST(CtrlFlowCacheDrcoUt, TaskAddrRelocProgramAndCtrlCache_WithDrco)
{
    DeviceWorkspaceAllocator workspace;
    auto dyntask = std::make_unique<DynDeviceTask>(workspace);
    dyntask->devTask.coreFunctionCnt = 8;

    DrcoQueueFixture drco;
    drco.Build();
    dyntask->drcoRootFuncList = &drco.root;

    std::array<std::array<uint32_t, 16>, READY_QUEUE_SIZE> elemBuf{};
    std::array<std::unique_ptr<ReadyCoreFunctionQueue>, READY_QUEUE_SIZE> queue{};
    SetupReadyQueues(dyntask.get(), elemBuf, queue);

    std::array<uint8_t, sizeof(DynFuncHeader) + 8 * sizeof(DynFuncData)> hdrBuf{};
    SetupDynFuncHeader(dyntask.get(), hdrBuf, 1);

    std::array<uint8_t, 1024> dupBuf{};
    DynFuncDataCache& cache = dyntask->dynFuncDataCacheList[0];
    cache.duppedData = SetupDuppedData(dupBuf, 1);
    cache.predCount = nullptr;
    cache.calleeList = nullptr;
    cache.devFunc = nullptr;
    dyntask->dynFuncDataBackupList[0] = DynFuncDataBackup{};

    std::vector<uint8_t> cacheBuf;
    DevControlFlowCache ctrl;
    SetupCtrlCache(ctrl, cacheBuf);

    ctrl.ReadyQueueDataBackup(dyntask.get());
    ASSERT_NE(dyntask->readyQueueBackup, nullptr);

    DeviceTaskCache entry;
    entry.dynTaskBase = dyntask.get();
    ctrl.deviceTaskCacheList = DevRelocVector<DeviceTaskCache>(1, &entry);
    ctrl.deviceTaskCount = 1;

    ctrl.TaskAddrRelocProgramAndCtrlCache(0, 0, 0, 0);
    SUCCEED();
}

TEST(CtrlFlowCacheDrcoUt, DrcoReadyQueueDataRestore_WithMixWraps)
{
    DeviceWorkspaceAllocator workspace;
    auto dyntask = std::make_unique<DynDeviceTask>(workspace);
    dyntask->devTask.coreFunctionCnt = 8;

    DrcoQueueFixture drco;
    drco.Build();
    dyntask->drcoRootFuncList = &drco.root;

    // Dirty the MIX shared queue (localReadyQueueArray[DRCO_QUEUE_MIX] group 0) and counters
    // so the DRCO reset loops are exercised.
    drco.root.localReadyQueueArray[0][0]->head = 1;
    drco.root.localReadyQueueArray[0][0]->tail = 1;
    drco.root.localReadyQueueArray[DRCO_QUEUE_MIX][0]->head = 1;
    drco.root.localReadyQueueArray[DRCO_QUEUE_MIX][0]->tail = 1;
    for (uint32_t ct = 0; ct < DRCO_QUEUE_MAX; ++ct) {
        drco.root.devTaskCountList.count[ct].executedCount = 3;
    }

    // Set up the ready queues (AIV/AIC/AICPU) on the DynDeviceTask.
    std::array<std::array<uint32_t, 16>, READY_QUEUE_SIZE> elemBuf{};
    std::array<std::unique_ptr<ReadyCoreFunctionQueue>, READY_QUEUE_SIZE> queue{};
    SetupReadyQueues(dyntask.get(), elemBuf, queue);

    // ctrl 实例，调用 DrcoReadyQueueDataRestore 不需要 AllocateCache/SetupCtrlCache
    DevControlFlowCache ctrl;

    // Wrap 队列：1 个 1C2V + 1 个 1C1V，tasklist 顺序为 AIC / AIV0 / AIV1
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
    queue[aivIdx]->UnsafeEnqueue(MakeTaskID(2, 302));
    queue[aicIdx]->UnsafeEnqueue(MakeTaskID(2, 301));

    ctrl.DrcoReadyQueueDataRestore(dyntask.get(), 4);

    auto* root = &drco.root;

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

    // Local ready queues (incl. the MIX shared queue at [DRCO_QUEUE_MIX][0]) reset to empty;
    // executed counters cleared (size untouched).
    EXPECT_EQ(drco.root.localReadyQueueArray[0][0]->head, 0U);
    EXPECT_EQ(drco.root.localReadyQueueArray[0][0]->tail, 0U);
    EXPECT_EQ(drco.root.localReadyQueueArray[DRCO_QUEUE_MIX][0]->head, 0U);
    EXPECT_EQ(drco.root.localReadyQueueArray[DRCO_QUEUE_MIX][0]->tail, 0U);
    for (uint32_t ct = 0; ct < DRCO_QUEUE_MAX; ++ct) {
        EXPECT_EQ(drco.root.devTaskCountList.count[ct].executedCount, 0U);
    }

    // 协议回环：构建期 precount（顺带备份进 devTask）→ 模拟设备污染 → 重放侧恢复。
    // AIC = 3 / AIV = 4；size 5 / 4（AIV 恰等于派发数置 flag，AIC 不置）
    drco.root.devTaskCountList.count[DRCO_QUEUE_AIC].size = 5;
    drco.root.devTaskCountList.count[DRCO_QUEUE_AIV].size = 4;
    DrcoRootFuncListPrecountPerCoreTasks(root, 4, dyntask->drcoPrecountExecuted, dyntask->drcoPrecountFinishFlag);
    EXPECT_EQ(dyntask->drcoPrecountExecuted[DRCO_QUEUE_AIC], 3U);
    EXPECT_EQ(dyntask->drcoPrecountExecuted[DRCO_QUEUE_AIV], 4U);
    EXPECT_EQ(dyntask->drcoPrecountFinishFlag[DRCO_QUEUE_AIC], 0U);
    EXPECT_EQ(dyntask->drcoPrecountFinishFlag[DRCO_QUEUE_AIV], 1U);

    drco.root.devTaskCountList.count[DRCO_QUEUE_AIC].executedCount = 100;
    drco.root.devTaskCountList.count[DRCO_QUEUE_AIV].executedCount = 100;
    drco.root.devTaskFinishFlagList.flag[DRCO_QUEUE_AIC].devTaskFinishFlag = 1;
    drco.root.devTaskFinishFlagList.flag[DRCO_QUEUE_AIV].devTaskFinishFlag = 0;
    DrcoRootFuncListRestorePrecount(root, dyntask->drcoPrecountExecuted, dyntask->drcoPrecountFinishFlag);
    EXPECT_EQ(drco.root.devTaskCountList.count[DRCO_QUEUE_AIC].executedCount, 3U);
    EXPECT_EQ(drco.root.devTaskCountList.count[DRCO_QUEUE_AIV].executedCount, 4U);
    EXPECT_EQ(drco.root.devTaskFinishFlagList.flag[DRCO_QUEUE_AIC].devTaskFinishFlag, 0U);
    EXPECT_EQ(drco.root.devTaskFinishFlagList.flag[DRCO_QUEUE_AIV].devTaskFinishFlag, 1U);
}

TEST(CtrlFlowCacheRelocUt, BuildIncastOutcastRelocTable_EmitsKindOrderedBatches)
{
    RelocTableFixture f;
    f.ctrl->BuildIncastOutcastRelocTable();

    ASSERT_EQ(f.ctrl->taskRelocTableCount, 1U);
    ASSERT_NE(f.ctrl->taskRelocTableOffset, 0U);

    const uint8_t* base = f.blob.data();
    auto* table = reinterpret_cast<const DevControlFlowCache::TaskRelocEntry*>(base + f.ctrl->taskRelocTableOffset);
    EXPECT_EQ(table[0].batchCount, 4U);
    EXPECT_EQ(table[0].dynFuncHeaderOffset, static_cast<uint64_t>(reinterpret_cast<const uint8_t*>(f.header) - base));

    // Batches must be dup-major and kind-ascending, each carrying the base offsets of
    // the dup's backup (src) and live (dst) arrays and a compact index segment.
    auto* batches = reinterpret_cast<const DevControlFlowCache::IncastOutcastRelocBatch*>(base +
                                                                                          table[0].batchListOffset);
    const AddressCacheKind kExpectedKinds[4] = {AddressCacheKind::Workspace, AddressCacheKind::Input,
                                                AddressCacheKind::Output, AddressCacheKind::Communication};
    const uint16_t kExpectedIdx[4] = {1, 0, 2, 3};
    for (uint32_t j = 0; j < table[0].batchCount; ++j) {
        EXPECT_EQ(batches[j].kind, kExpectedKinds[j]);
        EXPECT_EQ(batches[j].descCount, 1U);
        EXPECT_EQ(batches[j].srcBaseOffset, static_cast<uint64_t>(reinterpret_cast<const uint8_t*>(f.backup) - base));
        EXPECT_EQ(batches[j].dstBaseOffset, static_cast<uint64_t>(reinterpret_cast<const uint8_t*>(f.live) - base));
        auto* idx = reinterpret_cast<const uint16_t*>(base + batches[j].idxListOffset);
        EXPECT_EQ(idx[0], kExpectedIdx[j]);
    }
}

TEST(CtrlFlowCacheRelocUt, RelocIncastOutcastTask_TablePath_ResolvesAllKinds)
{
    RelocTableFixture f;
    f.ctrl->BuildIncastOutcastRelocTable();

    f.ctrl->RelocIncastOutcastTask(0, 0, RelocTableFixture::kDstWorkspace, &f.args);

    // live layout: [0]=Input#1, [1]=Workspace offset, [2]=Output#0, [3]=Comm.
    EXPECT_EQ(f.live[0], RelocTableFixture::kInputAddr);
    EXPECT_EQ(f.live[1], RelocTableFixture::kDstWorkspace + RelocTableFixture::kWsOffset);
    EXPECT_EQ(f.live[2], RelocTableFixture::kOutputAddr);
    EXPECT_EQ(f.live[3], RelocTableFixture::kCommAddr);
    EXPECT_EQ(f.dyntask->dynFuncDataList->startArgs, &f.args);
}

TEST(CtrlFlowCacheRelocUt, RelocIncastOutcastTask_NoTable_FallsBackToStructural)
{
    DeviceWorkspaceAllocator workspace;
    auto dyntask = std::make_unique<DynDeviceTask>(workspace);

    std::array<uint8_t, sizeof(DynFuncHeader) + 8 * sizeof(DynFuncData)> hdrBuf{};
    SetupDynFuncHeader(dyntask.get(), hdrBuf, 1);

    std::array<uint8_t, 1024> dupBuf{};
    auto* duppedData = SetupIncastOutcastDup(dupBuf, 2, 2);
    dyntask->dynFuncDataCacheList[0].duppedData = duppedData;

    std::array<uint64_t, 4> backup{};
    backup[0] = CacheFormWord(AddressCacheKind::Input, 1);
    backup[1] = CacheFormWord(AddressCacheKind::Workspace, RelocTableFixture::kWsOffset);
    backup[2] = CacheFormWord(AddressCacheKind::Output, 0);
    backup[3] = CacheFormWord(AddressCacheKind::Communication, RelocTableFixture::kCommAddr);
    dyntask->dynFuncDataBackupList[0].rawTensorAddrBackup = backup.data();

    // No BuildIncastOutcastRelocTable: taskRelocTableCount(0) != deviceTaskCount(1)
    // must route the call into RelocIncastOutcastTaskStructural, which resolves the
    // descriptors into the dupped data's own incast/outcast arrays.
    DevControlFlowCache ctrl;
    DeviceTaskCache entry;
    entry.dynTaskBase = dyntask.get();
    ctrl.deviceTaskCacheList = DevRelocVector<DeviceTaskCache>(1, &entry);
    ctrl.deviceTaskCount = 1;

    std::array<DevTensorData, 3> tensors{};
    tensors[1].address = RelocTableFixture::kInputAddr;
    tensors[2].address = RelocTableFixture::kOutputAddr;
    DevStartArgs args{};
    args.devTensorList = tensors.data();
    args.inputTensorSize = 2;
    args.outputTensorSize = 1;
    args.contextWorkspaceAddr = RelocTableFixture::kDstWorkspace;

    ctrl.RelocIncastOutcastTask(0, 0, RelocTableFixture::kDstWorkspace, &args);

    EXPECT_EQ(duppedData->GetIncastAddress(0).GetAddressValue(), RelocTableFixture::kInputAddr);
    EXPECT_EQ(duppedData->GetIncastAddress(1).GetAddressValue(),
              RelocTableFixture::kDstWorkspace + RelocTableFixture::kWsOffset);
    EXPECT_EQ(duppedData->GetOutcastAddress(0).GetAddressValue(), RelocTableFixture::kOutputAddr);
    EXPECT_EQ(duppedData->GetOutcastAddress(1).GetAddressValue(), RelocTableFixture::kCommAddr);
    EXPECT_EQ(dyntask->dynFuncDataList->startArgs, &args);
}
