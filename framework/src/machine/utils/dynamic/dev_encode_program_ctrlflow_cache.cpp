/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dev_encode_program_ctrlflow_cache.cpp
 * \brief
 */

#include "machine/utils/dynamic/dev_encode_program_ctrlflow_cache.h"
#include "machine/utils/machine_ws_intf.h"

namespace npu::tile_fwk::dynamic {

namespace {
template <typename Func>
bool ForEachIncastOutcastAddr(DynDeviceTaskBase* base, Func func)
{
    DynFuncHeader* dynFuncDataList = base->GetDynFuncDataList();
    DynFuncDataCache* dynFuncDataCacheList = base->dynFuncDataCacheList;
    DynFuncDataBackup* dynFuncDataBackupList = base->dynFuncDataBackupList;
    for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); ++dupIndex) {
        DynFuncData* dynData = &dynFuncDataList->At(dupIndex);
        DynFuncDataCache* dynDataCache = &dynFuncDataCacheList->At(dupIndex);
        DynFuncDataBackup* dynDataBackup = &dynFuncDataBackupList->At(dupIndex);
        DevAscendFunctionDuppedData* duppedData = dynDataCache->duppedData;
        size_t backupSize = sizeof(uint64_t) * (duppedData->GetIncastSize() + duppedData->GetOutcastSize());

        if (!func(dynData, dynDataBackup, backupSize)) {
            return false;
        }
    }
    return true;
}

template <AddressCacheKind Kind>
void RelocLoop(uint32_t descCount, const uint16_t* idxList, uint64_t* dst, const uint64_t* src, uint64_t wsShift,
               DevStartArgsBase* devStartArgs)
{
    for (uint32_t k = 0; k < descCount; k++) {
        uint32_t i = idxList[k];
        uint64_t srcVal = src[i];
        if constexpr (Kind == AddressCacheKind::Workspace) {
            dst[i] = srcVal + wsShift;
        } else if constexpr (Kind == AddressCacheKind::Input) {
            dst[i] = devStartArgs->GetInputTensor(static_cast<int>(srcVal & AddressDescriptor::kCacheValueMask))
                         .address;
        } else if constexpr (Kind == AddressCacheKind::Output) {
            dst[i] = devStartArgs->GetOutputTensor(static_cast<int>(srcVal & AddressDescriptor::kCacheValueMask))
                         .address;
        } else if constexpr (Kind == AddressCacheKind::Communication) {
            dst[i] = srcVal & AddressDescriptor::kCacheValueMask;
        } else {
            DEV_ERROR(ProgEncodeErr::CACHE_RELOC_KIND_INVALID,
                      "#ctrl.task.pre.cache.reloc: [RelocDescFromCache] Invalid kind: %u\n",
                      static_cast<uint32_t>(Kind));
        }
    }
}
} // namespace

void DevControlFlowCache::MatchInputOutputDump(DevStartArgsBase* startArgs) const
{
    DEV_VERBOSE_DEBUG("matchio cache input size: %d", (int)inputTensorDataList.size());
    for (size_t k = 0; k < inputTensorDataList.size(); k++) {
        DEV_VERBOSE_DEBUG("matchio cache input %d: %s", (int)k, DumpShape(inputTensorDataList[k].shape).c_str());
    }

    DEV_VERBOSE_DEBUG("matchio cache output size: %d", (int)outputTensorDataList.size());
    for (size_t k = 0; k < outputTensorDataList.size(); k++) {
        DEV_VERBOSE_DEBUG("matchio cache output %d: %s", (int)k, DumpShape(outputTensorDataList[k].shape).c_str());
    }

    DEV_VERBOSE_DEBUG("matchio real input size: %d", (int)startArgs->inputTensorSize);
    for (size_t k = 0; k < startArgs->inputTensorSize; k++) {
        DEV_VERBOSE_DEBUG("matchio real input %d: %s", (int)k, DumpShape(startArgs->GetInputTensor(k).shape).c_str());
    }

    DEV_VERBOSE_DEBUG("matchio real output size: %d", (int)startArgs->outputTensorSize);
    for (size_t k = 0; k < startArgs->outputTensorSize; k++) {
        DEV_VERBOSE_DEBUG("matchio real output %d: %s", (int)k, DumpShape(startArgs->GetOutputTensor(k).shape).c_str());
    }
}

void DevControlFlowCache::PredCountDataBackup(DynDeviceTaskBase* base)
{
    DynFuncHeader* dynFuncDataList = base->GetDynFuncDataList();
    DynFuncDataCache* dynFuncDataCacheList = base->dynFuncDataCacheList;
    DynFuncDataBackup* dynFuncDataBackupList = base->dynFuncDataBackupList;
    for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); ++dupIndex) {
        DynFuncDataCache* dynDataCache = &dynFuncDataCacheList->At(dupIndex);
        DynFuncDataBackup* dynDataBackup = &dynFuncDataBackupList->At(dupIndex);
        DevAscendFunctionDuppedData* duppedData = dynDataCache->duppedData;
        size_t backupSize = sizeof(predcount_t) * duppedData->GetOperationSize();

        predcount_t* predCountBackup = reinterpret_cast<predcount_t*>(AllocateCache(backupSize));
        if (predCountBackup == nullptr) {
            return;
        }
        dynDataBackup->predCountBackup = predCountBackup;
        DevMemcpyS(dynDataBackup->predCountBackup, backupSize, &duppedData->GetOperationCurrPredCount(0), backupSize);
        DevMemcpyS(duppedData->GetOperationPredCountPingPong(PRED_COUNT_PONG), backupSize,
                   dynDataBackup->predCountBackup, backupSize);

        BitmapDataBackup(dynDataCache, dynDataBackup);
    }
}

void DevControlFlowCache::PredCountPingPongSwap(DynDeviceTaskBase* base)
{
    DynFuncHeader* dynFuncDataList = base->GetDynFuncDataList();
    DynFuncDataCache* dynFuncDataCacheList = base->dynFuncDataCacheList;
    for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); ++dupIndex) {
        DynFuncDataCache& dynDataCache = dynFuncDataCacheList->At(dupIndex);
        std::swap(dynDataCache.predCountPingPong[PRED_COUNT_PING], dynDataCache.predCountPingPong[PRED_COUNT_PONG]);
        dynDataCache.predCount = dynDataCache.predCountPingPong[PRED_COUNT_PING];
    }
}

void DevControlFlowCache::DrcoPredCountDataRestore(DynDeviceTaskBase* base)
{
    DynFuncHeader* dynFuncDataList = base->GetDynFuncDataList();
    DynFuncDataBackup* dynFuncDataBackupList = base->dynFuncDataBackupList;
    for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); ++dupIndex) {
        DynFuncData* dynData = &dynFuncDataList->At(dupIndex);
        if (dynData->drcoRootFuncData.predCount == nullptr) {
            continue;
        }
        int32_t* drcoPredCount = dynData->drcoRootFuncData.predCount;
        predcount_t* predCountBackup = dynFuncDataBackupList->At(dupIndex).predCountBackup;
        uint32_t opSize = base->dynFuncDataCacheList->At(dupIndex).duppedData->GetOperationSize();
        for (uint32_t i = 0; i < opSize; ++i) {
            drcoPredCount[i] = static_cast<int32_t>(predCountBackup[i]);
        }
    }
}

void DevControlFlowCache::PredCountPingPongRestore(DynDeviceTaskBase* base)
{
    DynFuncHeader* dynFuncDataList = base->GetDynFuncDataList();
    DynFuncDataCache* dynFuncDataCacheList = base->dynFuncDataCacheList;
    DynFuncDataBackup* dynFuncDataBackupList = base->dynFuncDataBackupList;
    for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); ++dupIndex) {
        DynFuncDataCache* dynDataCache = &dynFuncDataCacheList->At(dupIndex);
        DynFuncDataBackup* dynDataBackup = &dynFuncDataBackupList->At(dupIndex);
        DevAscendFunctionDuppedData* duppedData = dynDataCache->duppedData;
        size_t backupSize = sizeof(predcount_t) * duppedData->GetOperationSize();
        DevMemcpyS(dynDataCache->predCountPingPong[PRED_COUNT_PONG], backupSize, dynDataBackup->predCountBackup,
                   backupSize);
    }
}

void DevControlFlowCache::BitmapDataRestoreTask(DynDeviceTaskBase* base)
{
    DynFuncHeader* dynFuncDataList = base->GetDynFuncDataList();
    DynFuncDataCache* dynFuncDataCacheList = base->dynFuncDataCacheList;
    DynFuncDataBackup* dynFuncDataBackupList = base->dynFuncDataBackupList;
    for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); ++dupIndex) {
        BitmapDataRestore(&dynFuncDataCacheList->At(dupIndex), &dynFuncDataBackupList->At(dupIndex));
    }
}

void DevControlFlowCache::ReadyQueueDataBackup(DynDeviceTaskBase* base)
{
    ReadyQueueCache* readyQueueBackup = reinterpret_cast<ReadyQueueCache*>(AllocateCache(sizeof(ReadyQueueCache)));
    if (readyQueueBackup == nullptr) {
        return;
    }
    readyQueueBackup->coreFunctionCnt = base->devTask.coreFunctionCnt;
    uint32_t readyTaskNum = 0;
    for (size_t i = 0; i < READY_QUEUE_SIZE; i++) {
        size_t backupSize = sizeof(uint32_t) * base->readyQueue[i]->Capacity();
        uint32_t* readyQueueBackupElem = reinterpret_cast<uint32_t*>(AllocateCache(backupSize));
        if (readyQueueBackupElem == nullptr) {
            return;
        }
        uint32_t* readyQueueShadowElem = reinterpret_cast<uint32_t*>(AllocateCache(backupSize));
        if (readyQueueShadowElem == nullptr) {
            return;
        }

        new (&readyQueueBackup->queueList[i])
            ReadyCoreFunctionQueueUnsafe(base->readyQueue[i]->Capacity(), readyQueueBackupElem);
        readyQueueBackup->queueList[i] = *base->readyQueue[i];
        readyTaskNum += base->readyQueue[i]->UnsafeSize();

        readyQueueBackup->pingElem[i] = base->readyQueue[i]->Data();
        readyQueueBackup->pongElem[i] = readyQueueShadowElem;
        DevMemcpyS(readyQueueShadowElem, backupSize, readyQueueBackup->queueList[i].Data(),
                   sizeof(uint32_t) * readyQueueBackup->queueList[i].Size());
    }
    readyQueueBackup->readyTaskNum = readyTaskNum;

    if (base->drcoRootFuncList != nullptr) {
        for (size_t i = 0; i < npu::tile_fwk::MAX_AICORE_NUM_FOR_QUEUE; i++) {
            auto* src = base->drcoRootFuncList->perCorePendingQueueArray[i];
            if (src == nullptr) {
                continue;
            }
            size_t backupSize = sizeof(npu::tile_fwk::PerCorePendingQueue) +
                                sizeof(npu::tile_fwk::LeafTaskId) * src->size;
            auto* dst = reinterpret_cast<npu::tile_fwk::PerCorePendingQueue*>(AllocateCache(backupSize));
            if (dst == nullptr) {
                continue;
            }
            memcpy_s(dst, backupSize, src, backupSize);
            readyQueueBackup->perCorePendingQueueList[i] = dst;
        }
    }

    base->readyQueueBackup = readyQueueBackup;
}

void DevControlFlowCache::DrcoReadyQueueDataRestore(DynDeviceTaskBase* base, uint32_t nrValidAic)
{
    auto* perCorePendingQueueArray = base->drcoRootFuncList->perCorePendingQueueArray;
    for (size_t i = 0; i < npu::tile_fwk::MAX_AICORE_NUM_FOR_QUEUE; i++) {
        auto* dst = perCorePendingQueueArray[i];
        if (dst == nullptr) {
            continue;
        }
        dst->head = 0;
        dst->tail = 0;
        dst->size = 0;
    }

    const uint32_t nrAivCores = nrValidAic * 2;
    ReadyCoreFunctionQueue* aivQueue = base->readyQueue[0];
    ReadyCoreFunctionQueue* aicQueue = base->readyQueue[1];
    uint32_t wrapCoreIdx = 0;

    // Restore wrap (mix) tasks: AIC task to core w, AIV tasks to paired vector cores.
    // Mirrors DispatchReadyQueueToCores so mix-task dispatch is not lost on cache replay.
    WrapInfoQueue* wrapQueue = reinterpret_cast<WrapInfoQueue*>(base->devTask.mixTaskData.readyWrapCoreFunctionQue);
    if (wrapQueue != nullptr && nrValidAic > 0) {
        constexpr uint8_t MIX_TYPE_1C2V = 2;
        constexpr uint8_t MIX_TYPE_1C1V = 1;
        for (uint8_t targetType : {MIX_TYPE_1C2V, MIX_TYPE_1C1V}) {
            for (uint32_t idx = wrapQueue->head; idx < wrapQueue->tail; idx++) {
                WrapInfo& info = wrapQueue->elem[idx];
                if (info.mixResourceType != targetType) {
                    continue;
                }
                uint32_t aicCore = wrapCoreIdx++ % nrValidAic;
                uint32_t aiv0Core = nrValidAic + aicCore * 2;
                perCorePendingQueueArray[aicCore]->UnsafeEnqueue(info.tasklist[WRAP_IDX_AIC]);
                perCorePendingQueueArray[aiv0Core]->UnsafeEnqueue(info.tasklist[WRAP_IDX_AIV0]);
                if (targetType == MIX_TYPE_1C2V) {
                    perCorePendingQueueArray[aiv0Core + 1]->UnsafeEnqueue(info.tasklist[WRAP_IDX_AIV1]);
                }
            }
        }
    }

    uint32_t aicIdx = wrapCoreIdx;
    for (const auto* it = aicQueue->begin(); it != aicQueue->end(); ++it) {
        perCorePendingQueueArray[aicIdx++ % nrValidAic]->UnsafeEnqueue(*it);
    }
    uint32_t aivIdx = wrapCoreIdx * 2;
    for (const auto* it = aivQueue->begin(); it != aivQueue->end(); ++it) {
        perCorePendingQueueArray[nrValidAic + (aivIdx++ % nrAivCores)]->UnsafeEnqueue(*it);
    }

    __sync_synchronize();
    for (uint32_t ct = 0; ct < npu::tile_fwk::DRCO_QUEUE_MAX; ct++) {
        base->drcoRootFuncList->devTaskCountList.count[ct].executedCount = 0;
    }
    // 就绪队列复位按各队列自身容量清理（MIX 行容量 = coreFunctionCnt + HUB_MIX op 数，
    // 大于 coreFunctionCnt）；三行（AIC/AIV/MIX）均全组分配
    for (uint32_t ct = 0; ct < npu::tile_fwk::DRCO_QUEUE_MAX; ct++) {
        for (uint32_t i = 0; i < npu::tile_fwk::NUM_LOCAL_GROUPS; i++) {
            auto* dst = base->drcoRootFuncList->localReadyQueueArray[ct][i];
            if (dst == nullptr) {
                continue;
            }
            dst->head = 0;
            dst->tail = 0;
            uint32_t taskListSize = dst->size * sizeof(npu::tile_fwk::LeafTaskId);
            (void)memset_s(reinterpret_cast<uint8_t*>(dst) + sizeof(DrcoLocalReadyQueue), taskListSize, 0,
                           taskListSize);
        }
    }
    for (uint32_t ct = 0; ct < npu::tile_fwk::DRCO_QUEUE_MAX; ct++) {
        for (uint32_t i = 0; i < npu::tile_fwk::NUM_LOCAL_GROUPS; i++) {
            auto* matrix = base->drcoRootFuncList->localReadyMatrixArray[ct][i];
            if (matrix == nullptr) {
                continue;
            }
            (void)memset_s(matrix->taskList, sizeof(matrix->taskList), 0, sizeof(matrix->taskList));
        }
    }
    // 槽内为 stitch 节点指针，复位为 nullptr，避免 cache 重放后残留悬空指针
    auto* stitchNodeMatrix = base->drcoRootFuncList->stitchNodeMatrix;
    if (stitchNodeMatrix != nullptr) {
        (void)memset_s(stitchNodeMatrix->stitchNodeList, sizeof(stitchNodeMatrix->stitchNodeList), 0,
                       sizeof(stitchNodeMatrix->stitchNodeList));
    }
    // 兜底 hub 矩阵槽内存 taskId，复位为 0，避免 cache 重放后残留任务
    auto* hubTaskMatrix = base->drcoRootFuncList->hubTaskMatrix;
    if (hubTaskMatrix != nullptr) {
        (void)memset_s(hubTaskMatrix->hubTaskList, sizeof(hubTaskMatrix->hubTaskList), 0,
                       sizeof(hubTaskMatrix->hubTaskList));
    }
    base->drcoRootFuncList->devTaskFinished = 0;
    auto* finishFlagList = &base->drcoRootFuncList->devTaskFinishFlagList;
    (void)memset_s(reinterpret_cast<uint8_t*>(finishFlagList), sizeof(*finishFlagList), 0, sizeof(*finishFlagList));
}

void DevControlFlowCache::ReadyQueueDataPingPongRestore(DynDeviceTaskBase* base)
{
    ReadyQueueCache* readyQueueBackup = base->readyQueueBackup;
    for (size_t i = 0; i < READY_QUEUE_SIZE; i++) {
        /* elem_ targets the buffer this launch just dirtied; refresh the other one. */
        uint32_t* inactiveElem = (base->readyQueue[i]->Data() == readyQueueBackup->pingElem[i]) ?
                                     readyQueueBackup->pongElem[i] :
                                     readyQueueBackup->pingElem[i];
        size_t capBytes = sizeof(uint32_t) * readyQueueBackup->queueList[i].Capacity();
        size_t copyBytes = sizeof(uint32_t) * readyQueueBackup->queueList[i].Size();
        DevMemcpyS(inactiveElem, capBytes, readyQueueBackup->queueList[i].Data(), copyBytes);
    }
}

void DevControlFlowCache::ReadyQueueDataPingPongSwap(DynDeviceTaskBase* base, uint32_t nrValidAic)
{
    ReadyQueueCache* readyQueueBackup = base->readyQueueBackup;
    base->devTask.coreFunctionCnt = readyQueueBackup->coreFunctionCnt;
    for (size_t i = 0; i < READY_QUEUE_SIZE; i++) {
        uint32_t* inactiveElem = (base->readyQueue[i]->Data() == readyQueueBackup->pingElem[i]) ?
                                     readyQueueBackup->pongElem[i] :
                                     readyQueueBackup->pingElem[i];
        base->readyQueue[i]->FillFrom(readyQueueBackup->queueList[i], inactiveElem);
    }

    // for aicore-resolve
    if (base->drcoRootFuncList != nullptr) {
        DrcoReadyQueueDataRestore(base, nrValidAic);
    }
}

void DevControlFlowCache::DieReadyQueueDataBackup(DynDeviceTaskBase* base)
{
    DieReadyQueueCache* dieReadyQueueBackup = reinterpret_cast<DieReadyQueueCache*>(
        AllocateCache(sizeof(DieReadyQueueCache)));
    if (dieReadyQueueBackup == nullptr) {
        return;
    }
    dieReadyQueueBackup->coreFunctionCnt = base->devTask.coreFunctionCnt;
    uint32_t readyTaskNum = 0;
    for (size_t i = 0; i < DIE_READY_QUEUE_SIZE * DIE_NUM; i++) {
        ReadyCoreFunctionQueue* dieReadyQueue = reinterpret_cast<ReadyCoreFunctionQueue*>(
            (i < DIE_NUM) ? base->devTask.dieReadyFunctionQue.readyDieAivCoreFunctionQue[i] :
                            base->devTask.dieReadyFunctionQue.readyDieAicCoreFunctionQue[i - DIE_NUM]);
        if (dieReadyQueue == nullptr) {
            new (&dieReadyQueueBackup->queueList[i]) ReadyCoreFunctionQueueUnsafe(0, nullptr);
            dieReadyQueueBackup->pingElem[i] = nullptr;
            dieReadyQueueBackup->pongElem[i] = nullptr;
            continue;
        }
        size_t backupSize = sizeof(uint32_t) * dieReadyQueue->Capacity();
        uint32_t* dieReadyQueueBackupElem = reinterpret_cast<uint32_t*>(AllocateCache(backupSize));
        if (dieReadyQueueBackupElem == nullptr) {
            return;
        }
        uint32_t* dieReadyQueueShadowElem = reinterpret_cast<uint32_t*>(AllocateCache(backupSize));
        if (dieReadyQueueShadowElem == nullptr) {
            return;
        }

        new (&dieReadyQueueBackup->queueList[i])
            ReadyCoreFunctionQueueUnsafe(dieReadyQueue->Capacity(), dieReadyQueueBackupElem);
        dieReadyQueueBackup->queueList[i] = *dieReadyQueue;
        readyTaskNum += dieReadyQueue->UnsafeSize();

        /* Ping-pong, same protocol as ReadyQueueDataBackup. */
        dieReadyQueueBackup->pingElem[i] = dieReadyQueue->Data();
        dieReadyQueueBackup->pongElem[i] = dieReadyQueueShadowElem;
        DevMemcpyS(dieReadyQueueShadowElem, backupSize, dieReadyQueueBackup->queueList[i].Data(),
                   sizeof(uint32_t) * dieReadyQueueBackup->queueList[i].Size());
    }
    dieReadyQueueBackup->readyTaskNum = readyTaskNum;
    base->dieReadyQueueBackup = dieReadyQueueBackup;
}

void DevControlFlowCache::DieReadyQueuePingPongRestore(DynDeviceTaskBase* base)
{
    DieReadyQueueCache* dieReadyQueueBackup = base->dieReadyQueueBackup;
    if (dieReadyQueueBackup == nullptr) {
        return;
    }
    for (size_t i = 0; i < DIE_READY_QUEUE_SIZE * DIE_NUM; i++) {
        ReadyCoreFunctionQueue* dieReadyQueue = reinterpret_cast<ReadyCoreFunctionQueue*>(
            (i < DIE_NUM) ? base->devTask.dieReadyFunctionQue.readyDieAivCoreFunctionQue[i] :
                            base->devTask.dieReadyFunctionQue.readyDieAicCoreFunctionQue[i - DIE_NUM]);
        if (dieReadyQueue == nullptr) {
            continue;
        }
        /* elem_ targets the buffer this launch just dirtied; refresh the other one. */
        uint32_t* inactiveElem = (dieReadyQueue->Data() == dieReadyQueueBackup->pingElem[i]) ?
                                     dieReadyQueueBackup->pongElem[i] :
                                     dieReadyQueueBackup->pingElem[i];
        size_t capBytes = sizeof(uint32_t) * dieReadyQueueBackup->queueList[i].Capacity();
        size_t copyBytes = sizeof(uint32_t) * dieReadyQueueBackup->queueList[i].Size();
        DevMemcpyS(inactiveElem, capBytes, dieReadyQueueBackup->queueList[i].Data(), copyBytes);
    }
}

void DevControlFlowCache::DieReadyQueuePingPongSwap(DynDeviceTaskBase* base, uint32_t nrValidAic)
{
    DieReadyQueueCache* dieReadyQueueBackup = base->dieReadyQueueBackup;
    if (dieReadyQueueBackup == nullptr) {
        return;
    }
    base->devTask.coreFunctionCnt = dieReadyQueueBackup->coreFunctionCnt;
    for (size_t i = 0; i < DIE_READY_QUEUE_SIZE * DIE_NUM; i++) {
        ReadyCoreFunctionQueue* dieReadyQueue = reinterpret_cast<ReadyCoreFunctionQueue*>(
            (i < DIE_NUM) ? base->devTask.dieReadyFunctionQue.readyDieAivCoreFunctionQue[i] :
                            base->devTask.dieReadyFunctionQue.readyDieAicCoreFunctionQue[i - DIE_NUM]);
        if (dieReadyQueue == nullptr) {
            continue;
        }
        /* Ping-pong: rebind to the buffer the previous launch's shadow restore refreshed. */
        uint32_t* inactiveElem = (dieReadyQueue->Data() == dieReadyQueueBackup->pingElem[i]) ?
                                     dieReadyQueueBackup->pongElem[i] :
                                     dieReadyQueueBackup->pingElem[i];
        dieReadyQueue->FillFrom(dieReadyQueueBackup->queueList[i], inactiveElem);
    }

    if (base->drcoRootFuncList == nullptr) {
        return;
    }
    // for aicore-resolve
    uint32_t nrAivCores = 2 * nrValidAic;
    uint32_t halfAic = nrValidAic / 2;
    uint32_t halfAiv = nrAivCores / 2;
    // perCorePendingQueue: distribute die tasks, die0 to first-half cores, die1 to second-half cores
    for (uint32_t dieId = 0; dieId < DIE_NUM; ++dieId) {
        uint32_t aicCoreBase = dieId * halfAic;
        uint32_t aicCoreCnt = (dieId == 0) ? halfAic : (nrValidAic - halfAic);
        ReadyCoreFunctionQueue* dieAicQue = reinterpret_cast<ReadyCoreFunctionQueue*>(
            base->devTask.dieReadyFunctionQue.readyDieAicCoreFunctionQue[dieId]);
        if (dieAicQue != nullptr) {
            uint32_t dieAicIdx = 0;
            for (const auto* it = dieAicQue->begin(); it != dieAicQue->end(); ++it) {
                uint32_t coreIdx = aicCoreBase + (dieAicIdx % aicCoreCnt);
                base->drcoRootFuncList->perCorePendingQueueArray[coreIdx]->UnsafeEnqueue(*it);
                dieAicIdx++;
            }
        }

        uint32_t aivCoreBase = nrValidAic + dieId * halfAiv;
        uint32_t aivCoreCnt = (dieId == 0) ? halfAiv : (nrAivCores - halfAiv);
        ReadyCoreFunctionQueue* dieAivQue = reinterpret_cast<ReadyCoreFunctionQueue*>(
            base->devTask.dieReadyFunctionQue.readyDieAivCoreFunctionQue[dieId]);
        if (dieAivQue != nullptr) {
            uint32_t dieAivIdx = 0;
            for (const auto* it = dieAivQue->begin(); it != dieAivQue->end(); ++it) {
                uint32_t coreIdx = aivCoreBase + (dieAivIdx % aivCoreCnt);
                base->drcoRootFuncList->perCorePendingQueueArray[coreIdx]->UnsafeEnqueue(*it);
                dieAivIdx++;
            }
        }
    }
}

void DevControlFlowCache::MixTaskDataBackup(DynDeviceTaskBase* base)
{
    if (base->devTask.mixTaskData.wrapIdNum == 0) {
        return;
    }
    MixTaskDataCache* mixTaskDataBackup = reinterpret_cast<MixTaskDataCache*>(AllocateCache(sizeof(MixTaskDataCache)));
    if (mixTaskDataBackup == nullptr) {
        return;
    }
    mixTaskDataBackup->wrapIdNum = base->devTask.mixTaskData.wrapIdNum;
    WrapInfoQueue* wrapInfoQueue = reinterpret_cast<WrapInfoQueue*>(base->devTask.mixTaskData.readyWrapCoreFunctionQue);
    size_t wrapInfoBackupSize = sizeof(WrapInfo) * wrapInfoQueue->capacity;
    WrapInfo* wrapQueueBackupElem = reinterpret_cast<WrapInfo*>(AllocateCache(wrapInfoBackupSize));
    if (wrapQueueBackupElem == nullptr) {
        return;
    }
    mixTaskDataBackup->queue.head = wrapInfoQueue->head;
    mixTaskDataBackup->queue.tail = wrapInfoQueue->tail;
    mixTaskDataBackup->queue.capacity = wrapInfoQueue->capacity;
    mixTaskDataBackup->queue.elem = wrapQueueBackupElem;
    DevMemcpyS(mixTaskDataBackup->queue.elem, wrapInfoBackupSize, wrapInfoQueue->elem, wrapInfoBackupSize);

    constexpr size_t arrSize = sizeof(uint64_t) * MAX_STITCH_FUNC_NUM;
    DevMemcpyS(mixTaskDataBackup->opWrapList, arrSize, base->devTask.mixTaskData.opWrapList, arrSize);

    base->mixTaskDataBackup = mixTaskDataBackup;
}

void DevControlFlowCache::MixTaskDataRestore(DynDeviceTaskBase* base)
{
    if (base->mixTaskDataBackup == nullptr) {
        return;
    }
    MixTaskDataCache* mixTaskDataBackup = base->mixTaskDataBackup;
    base->devTask.mixTaskData.wrapIdNum = mixTaskDataBackup->wrapIdNum;

    WrapInfoQueue* wrapInfoQueue = reinterpret_cast<WrapInfoQueue*>(base->devTask.mixTaskData.readyWrapCoreFunctionQue);
    wrapInfoQueue->head = mixTaskDataBackup->queue.head;
    wrapInfoQueue->tail = mixTaskDataBackup->queue.tail;
    wrapInfoQueue->capacity = mixTaskDataBackup->queue.capacity;

    size_t wrapInfoBackupSize = sizeof(WrapInfo) * wrapInfoQueue->capacity;
    DevMemcpyS(wrapInfoQueue->elem, wrapInfoBackupSize, mixTaskDataBackup->queue.elem, wrapInfoBackupSize);

    constexpr size_t arrSize = sizeof(uint64_t) * MAX_STITCH_FUNC_NUM;
    DevMemcpyS(base->devTask.mixTaskData.opWrapList, arrSize, mixTaskDataBackup->opWrapList, arrSize);
}

void DevControlFlowCache::RelocBuildInputOutputDesc(
    std::unordered_map<uint64_t, AddressDescriptor>& cacheInputOutputDict, DevStartArgsBase* devStartArgs)
{
    for (uint64_t i = 0; i < devStartArgs->inputTensorSize; i++) {
        uint64_t addr = devStartArgs->GetInputTensor(i).address;
        cacheInputOutputDict[addr] = AddressDescriptor::MakeCache(AddressCacheKind::Input, i);
    }
    for (uint64_t i = 0; i < devStartArgs->outputTensorSize; i++) {
        uint64_t addr = devStartArgs->GetOutputTensor(i).address;
        cacheInputOutputDict[addr] = AddressDescriptor::MakeCache(AddressCacheKind::Output, i);
    }
}

void DevControlFlowCache::RelocBuildInputOutputDesc(
    std::unordered_map<uint64_t, AddressDescriptor>& cacheInputOutputDict,
    DevRelocVector<DevTensorData> inputTensorDataList, DevRelocVector<DevTensorData> outputTensorDataList)
{
    for (uint64_t i = 0; i < inputTensorDataList.size(); i++) {
        uint64_t addr = inputTensorDataList[i].address;
        cacheInputOutputDict[addr] = AddressDescriptor::MakeCache(AddressCacheKind::Input, i);
    }
    for (uint64_t i = 0; i < outputTensorDataList.size(); i++) {
        uint64_t addr = outputTensorDataList[i].address;
        cacheInputOutputDict[addr] = AddressDescriptor::MakeCache(AddressCacheKind::Output, i);
    }
}

void DevControlFlowCache::RelocDescToCache(AddressDescriptor& desc, const RelocRange& relocWorkspace,
                                           std::unordered_map<uint64_t, AddressDescriptor>& cacheInputOutputDict)
{
    AddressDescriptor resultDesc;
    uint64_t addr = desc.GetAddressValue();
    if (cacheInputOutputDict.count(addr)) {
        resultDesc = cacheInputOutputDict[addr];
    } else if (addr & (1UL << 58)) {
        resultDesc = AddressDescriptor::MakeCache(AddressCacheKind::Communication, addr);
    } else {
        relocWorkspace.Reloc(addr);
        resultDesc = AddressDescriptor::MakeCache(AddressCacheKind::Workspace, addr);
    }
    desc = resultDesc;
}

void DevControlFlowCache::RelocDescFromCache(const AddressDescriptor& desc, AddressDescriptor& result,
                                             const RelocRange& relocWorkspace, DevStartArgsBase* devStartArgs)
{
    uint64_t resultAddr = 0;
    switch (desc.GetCacheKind()) {
        case AddressCacheKind::Workspace:
            resultAddr = desc.cacheValue;
            relocWorkspace.Reloc(resultAddr);
            break;
        case AddressCacheKind::Input:
            resultAddr = devStartArgs->GetInputTensor(desc.cacheValue).address;
            break;
        case AddressCacheKind::Output:
            resultAddr = devStartArgs->GetOutputTensor(desc.cacheValue).address;
            break;
        case AddressCacheKind::Communication:
            resultAddr = desc.cacheValue;
            break;
        default:
            DEV_ERROR(ProgEncodeErr::CACHE_RELOC_KIND_INVALID,
                      "#ctrl.task.pre.cache.reloc: [RelocDescFromCache] Invalid kind: %lu\n",
                      (unsigned long)desc.cacheKind);
            break;
    }
    result = AddressDescriptor::MakeFromAddress(resultAddr);
}

void DevControlFlowCache::IncastOutcastAddrBackup(DynDeviceTaskBase* base)
{
    ForEachIncastOutcastAddr(base, [this](DynFuncData* dynData, DynFuncDataBackup* dynDataBackup, size_t backupSize) {
        uint64_t* rawTensorAddrBackup = reinterpret_cast<uint64_t*>(AllocateCache(backupSize));
        if (rawTensorAddrBackup == nullptr) {
            return false;
        }
        dynDataBackup->rawTensorAddrBackup = rawTensorAddrBackup;
        DevMemcpyS(dynDataBackup->rawTensorAddrBackup, backupSize, dynData->rawTensorAddr, backupSize);
        return true;
    });
}

void DevControlFlowCache::TaskAddrBackupWorkspace(DynDeviceTaskBase* base)
{
    DynFuncHeader* dynFuncDataList = base->GetDynFuncDataList();
    DynFuncDataCache* dynFuncDataCacheList = base->dynFuncDataCacheList;
    DynFuncDataBackup* dynFuncDataBackupList = base->dynFuncDataBackupList;
    for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); ++dupIndex) {
        DynFuncData* dynData = &dynFuncDataList->At(dupIndex);
        DynFuncDataCache* dynDataCache = &dynFuncDataCacheList->At(dupIndex);
        DynFuncDataBackup* dynDataBackup = &dynFuncDataBackupList->At(dupIndex);
        DevAscendFunctionDuppedData* duppedData = dynDataCache->duppedData;

        dynDataBackup->workspaceAddressBackup.runtimeWorkspace = duppedData->runtimeWorkspace_;
        dynDataBackup->workspaceAddressBackup.runtimeOutcastWorkspace = duppedData->runtimeOutcastWorkspace_;
        dynDataBackup->workspaceAddressBackup.workspaceAddr = dynData->workspaceAddr;
        dynDataBackup->workspaceAddressBackup.stackWorkspaceAddr = dynFuncDataList->stackWorkSpaceAddr;
    }
}

void DevControlFlowCache::TaskAddrRestoreWorkspace(DynDeviceTaskBase* base)
{
    DynFuncHeader* dynFuncDataList = base->GetDynFuncDataList();
    DynFuncDataCache* dynFuncDataCacheList = base->dynFuncDataCacheList;
    DynFuncDataBackup* dynFuncDataBackupList = base->dynFuncDataBackupList;
    for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); ++dupIndex) {
        DynFuncData* dynData = &dynFuncDataList->At(dupIndex);
        DynFuncDataCache* dynDataCache = &dynFuncDataCacheList->At(dupIndex);
        DynFuncDataBackup* dynDataBackup = &dynFuncDataBackupList->At(dupIndex);
        DevAscendFunctionDuppedData* duppedData = dynDataCache->duppedData;

        duppedData->runtimeWorkspace_ = dynDataBackup->workspaceAddressBackup.runtimeWorkspace;
        duppedData->runtimeOutcastWorkspace_ = dynDataBackup->workspaceAddressBackup.runtimeOutcastWorkspace;
        dynData->workspaceAddr = dynDataBackup->workspaceAddressBackup.workspaceAddr;
    }
    dynFuncDataList->stackWorkSpaceAddr = dynFuncDataBackupList[0].workspaceAddressBackup.stackWorkspaceAddr;
}

void DevControlFlowCache::TaskAddrRestoreWorkspace()
{
    for (size_t i = 0; i < deviceTaskCount; i++) {
        DynDeviceTaskBase* dynTaskBase = deviceTaskCacheList[i].dynTaskBase;
        TaskAddrRestoreWorkspace(dynTaskBase);
    }
}

void DevControlFlowCache::TaskAddrRelocWorkspace(uint64_t srcWorkspace, uint64_t dstWorkspace,
                                                 DevStartArgsBase* devStartArgs)
{
    RelocRange relocWorkspace(srcWorkspace, dstWorkspace);
    for (uint64_t deviceIndex = 0; deviceIndex < deviceTaskCount; deviceIndex++) {
        DynDeviceTaskBase* dynTaskBase = deviceTaskCacheList[deviceIndex].dynTaskBase;

        DynFuncHeader* dynFuncDataList = dynTaskBase->dynFuncDataList;
        DynFuncDataCache* dynFuncDataCacheList = dynTaskBase->dynFuncDataCacheList;
        DynFuncDataBackup* dynFuncDataBackupList = dynTaskBase->dynFuncDataBackupList;
        for (uint32_t dupIndex = 0; dupIndex < dynFuncDataList->funcNum; dupIndex++) {
            DynFuncData* dynData = &dynFuncDataList->At(dupIndex);
            DynFuncDataCache* dynDataCache = &dynFuncDataCacheList->At(dupIndex);
            DynFuncDataBackup* dynDataBackup = &dynFuncDataBackupList->At(dupIndex);
            DevAscendFunctionDuppedData* duppedData = dynDataCache->duppedData;

            if (devStartArgs == nullptr) {
                // Host: addr uses backup

                // Reloc Dupped
                relocWorkspace.RelocNullable(dynDataBackup->workspaceAddressBackup.runtimeWorkspace);
                relocWorkspace.RelocNullable(dynDataBackup->workspaceAddressBackup.runtimeOutcastWorkspace);

                // Reloc DynFuncData
                relocWorkspace.Reloc(dynDataBackup->workspaceAddressBackup.workspaceAddr);
            } else {
                // Device: addr uses actual

                // Reloc Dupped
                relocWorkspace.RelocNullable(duppedData->runtimeWorkspace_);
                relocWorkspace.RelocNullable(duppedData->runtimeOutcastWorkspace_);

                // Reloc DynFuncData
                relocWorkspace.Reloc(dynData->workspaceAddr);
            }
        }
        // Reloc header-level fields (shared by all funcs)
        if (devStartArgs == nullptr) {
            relocWorkspace.Reloc(dynFuncDataBackupList[0].workspaceAddressBackup.stackWorkspaceAddr);
            dynFuncDataList->stackWorkSpaceAddr = dynFuncDataBackupList[0].workspaceAddressBackup.stackWorkspaceAddr;
        } else {
            relocWorkspace.Reloc(dynFuncDataList->stackWorkSpaceAddr);
        }
    }
}

void DevControlFlowCache::IncastOutcastAddrReloc(uint64_t srcWorkspace, uint64_t dstWorkspace,
                                                 DevStartArgsBase* devStartArgs)
{
    RelocRange relocWorkspace(srcWorkspace, dstWorkspace);
    /* empty constructor's overhead should be negligible */
    std::unordered_map<uint64_t, AddressDescriptor> cacheInputOutputDict;
    if (devStartArgs == nullptr) {
        /* only run on host */
        RelocBuildInputOutputDesc(cacheInputOutputDict, inputTensorDataList, outputTensorDataList);
    }
    for (uint64_t deviceIndex = 0; deviceIndex < deviceTaskCount; deviceIndex++) {
        DynDeviceTaskBase* dynTaskBase = deviceTaskCacheList[deviceIndex].dynTaskBase;
        DynFuncHeader* dynFuncDataList = dynTaskBase->dynFuncDataList;
        DynFuncDataCache* dynFuncDataCacheList = dynTaskBase->dynFuncDataCacheList;
        DynFuncDataBackup* dynFuncDataBackupList = dynTaskBase->dynFuncDataBackupList;
        for (uint32_t dupIndex = 0; dupIndex < dynFuncDataList->funcNum; dupIndex++) {
            DynFuncDataCache* dynDataCache = &dynFuncDataCacheList->At(dupIndex);
            DynFuncDataBackup* dynDataBackup = &dynFuncDataBackupList->At(dupIndex);

            DevAscendFunctionDuppedData* duppedData = dynDataCache->duppedData;
            if (devStartArgs == nullptr) {
                // Host: addr uses backup
                for (uint64_t i = 0; i < duppedData->GetIncastSize(); i++) {
                    AddressDescriptor* addr = reinterpret_cast<AddressDescriptor*>(dynDataBackup->rawTensorAddrBackup +
                                                                                   i);
                    RelocDescToCache(*addr, relocWorkspace, cacheInputOutputDict);
                }
                for (uint64_t i = 0; i < duppedData->GetOutcastSize(); i++) {
                    AddressDescriptor* addr = reinterpret_cast<AddressDescriptor*>(dynDataBackup->rawTensorAddrBackup +
                                                                                   duppedData->GetIncastSize() + i);
                    RelocDescToCache(*addr, relocWorkspace, cacheInputOutputDict);
                }
            }
        }
        dynFuncDataList->startArgs = devStartArgs;
    }
}

namespace {

uint32_t RelocTableCountDupKinds(const AddressDescriptor* backup, uint32_t descNum, uint32_t* present,
                                 std::vector<uint8_t>& kinds)
{
    for (size_t k = 0; k < static_cast<size_t>(AddressCacheKind::MaxKind); k++) {
        present[k] = 0;
    }
    for (uint32_t i = 0; i < descNum; i++) {
        uint32_t rawKind = backup[i].cacheKind;
        DEV_ASSERT(ProgEncodeErr::CTRL_CACHE_RELOC_BUILD_FAILED,
                   rawKind < static_cast<uint32_t>(AddressCacheKind::MaxKind));
        present[rawKind]++;
        kinds.push_back(static_cast<uint8_t>(rawKind));
    }
    uint32_t kindsPresent = 0;
    for (size_t k = 0; k < static_cast<size_t>(AddressCacheKind::MaxKind); k++) {
        kindsPresent += (present[k] > 0) ? 1U : 0U;
    }
    return kindsPresent;
}

/* Sizes of the per-task reloc block: one batch per (dup, kind) with at least one
 * descriptor, plus one uint16 index per descriptor. */
struct RelocTableTaskBatchSizes {
    uint32_t batchCount;
    uint32_t totalIdx;
};

/* Per-dup data captured by the scan pass, so the fill pass never re-reads
 * backup descriptors. */
struct RelocTableDupRelocInfo {
    uint32_t descNum;
    uint32_t present[static_cast<size_t>(AddressCacheKind::MaxKind)];
    uint64_t srcBase;
    uint64_t dstBase;
};

/* Pass 1 of the reloc-table build: measure the per-task batch block, capturing
 * the per-dup census and base offsets along the way. */
RelocTableTaskBatchSizes RelocTableScanTaskBatches(const DevControlFlowCache& cache, DynDeviceTaskBase* dynTaskBase,
                                                   std::vector<RelocTableDupRelocInfo>& infos,
                                                   std::vector<uint8_t>& kinds)
{
    const uint8_t* cacheBase = reinterpret_cast<const uint8_t*>(&cache);
    DynFuncHeader* dynFuncDataList = dynTaskBase->dynFuncDataList;
    DynFuncDataCache* dynFuncDataCacheList = dynTaskBase->dynFuncDataCacheList;
    DynFuncDataBackup* dynFuncDataBackupList = dynTaskBase->dynFuncDataBackupList;
    infos.clear();
    kinds.clear();
    RelocTableTaskBatchSizes sizes{0, 0};
    for (uint32_t dupIndex = 0; dupIndex < dynFuncDataList->funcNum; dupIndex++) {
        DevAscendFunctionDuppedData* duppedData = dynFuncDataCacheList->At(dupIndex).duppedData;
        uint32_t descNum = duppedData->GetIncastSize() + duppedData->GetOutcastSize();
        DEV_ASSERT(ProgEncodeErr::CTRL_CACHE_RELOC_BUILD_FAILED, descNum <= UINT16_MAX);
        uint64_t* rawTensorAddrBackup = dynFuncDataBackupList->At(dupIndex).rawTensorAddrBackup;
        const AddressDescriptor* backup = reinterpret_cast<const AddressDescriptor*>(rawTensorAddrBackup);
        RelocTableDupRelocInfo info{};
        info.descNum = descNum;
        sizes.batchCount += RelocTableCountDupKinds(backup, descNum, info.present, kinds);
        info.srcBase = reinterpret_cast<uint8_t*>(rawTensorAddrBackup) - cacheBase;
        info.dstBase = reinterpret_cast<uint8_t*>(dynFuncDataList->At(dupIndex).rawTensorAddr) - cacheBase;
        infos.push_back(info);
        sizes.totalIdx += descNum;
    }
    return sizes;
}

/* Pass 2 of the reloc-table build: fill the allocated per-task block with
 * batch records and their descriptor-index segments, reading only the captured per-dup info. */
void RelocTableEmitTaskRelocBatches(const DevControlFlowCache& cache, const std::vector<RelocTableDupRelocInfo>& infos,
                                    const std::vector<uint8_t>& kinds,
                                    DevControlFlowCache::IncastOutcastRelocBatch* batches, uint16_t* idxList)
{
    const uint8_t* cacheBase = reinterpret_cast<const uint8_t*>(&cache);
    uint32_t batchWritePos = 0;
    uint32_t idxWritePos = 0;
    uint32_t kindReadPos = 0;
    for (const RelocTableDupRelocInfo& info : infos) {
        for (uint32_t kind = 0; kind < static_cast<uint32_t>(AddressCacheKind::MaxKind); kind++) {
            if (info.present[kind] == 0) {
                continue;
            }
            batches[batchWritePos] = {
                info.srcBase, info.dstBase,
                static_cast<uint64_t>(reinterpret_cast<uint8_t*>(idxList + idxWritePos) - cacheBase),
                info.present[kind], static_cast<AddressCacheKind>(kind)};
            batchWritePos++;
            for (uint32_t i = 0; i < info.descNum; i++) {
                if (kinds[kindReadPos + i] == kind) {
                    idxList[idxWritePos++] = static_cast<uint16_t>(i);
                }
            }
        }
        kindReadPos += info.descNum;
    }
}

} // namespace

/* Reloc table memory layout
 * [TaskRelocEntry x deviceTaskCount]
 * [IncastOutcastRelocBatch x batchCount | uint16 x totalIdx]
 *  batch (dup-major kind-ascending, only (dup, kind) with descriptors):
 *  +---------------+---------------+---------------+-----------+------+
 *  | srcBaseOffset | dstBaseOffset | idxListOffset | descCount | kind |
 *  +---------------+---------------+---------------+-----------+------+
 *  uint16 idx (ascending within each batch, appended in batch order):
 *  +---------------------------+---------------------------+-----+
 *  | batch0: i0 i1 .. iN-1     | batch1: i0 i1 .. iM-1     | ... |
 *  +---------------------------+---------------------------+-----+
 */
void DevControlFlowCache::BuildIncastOutcastRelocTable()
{
    taskRelocTableOffset = 0;
    taskRelocTableCount = 0;
    if (deviceTaskCount == 0) {
        return;
    }
    void* indexBlock = AllocateCache(deviceTaskCount * sizeof(TaskRelocEntry));
    if (indexBlock == nullptr) {
        return;
    }
    taskRelocTableOffset = reinterpret_cast<uint8_t*>(indexBlock) - reinterpret_cast<uint8_t*>(this);
    TaskRelocEntry* taskEntries = static_cast<TaskRelocEntry*>(indexBlock);
    std::vector<RelocTableDupRelocInfo> infos;
    std::vector<uint8_t> kinds;
    for (size_t index = 0; index < deviceTaskCount; index++) {
        DynDeviceTaskBase* dynTaskBase = deviceTaskCacheList[index].dynTaskBase;
        const RelocTableTaskBatchSizes sizes = RelocTableScanTaskBatches(*this, dynTaskBase, infos, kinds);
        void* block = AllocateCache(sizes.batchCount * sizeof(IncastOutcastRelocBatch) +
                                    sizes.totalIdx * sizeof(uint16_t));
        if (block == nullptr) {
            taskRelocTableOffset = 0;
            taskRelocTableCount = 0;
            return;
        }
        taskEntries[index] = {
            static_cast<uint64_t>(reinterpret_cast<uint8_t*>(block) - reinterpret_cast<uint8_t*>(this)),
            static_cast<uint64_t>(reinterpret_cast<uint8_t*>(dynTaskBase->dynFuncDataList) -
                                  reinterpret_cast<uint8_t*>(this)),
            sizes.batchCount};
        RelocTableEmitTaskRelocBatches(
            *this, infos, kinds, static_cast<IncastOutcastRelocBatch*>(block),
            reinterpret_cast<uint16_t*>(static_cast<IncastOutcastRelocBatch*>(block) + sizes.batchCount));
    }
    taskRelocTableCount = static_cast<uint32_t>(deviceTaskCount);
}

void DevControlFlowCache::RelocIncastOutcastTaskStructural(uint32_t taskIndex, uint64_t srcWorkspace,
                                                           uint64_t dstWorkspace, DevStartArgsBase* devStartArgs)
{
    RelocRange relocWorkspace(srcWorkspace, dstWorkspace);
    DynDeviceTaskBase* dynTaskBase = deviceTaskCacheList[taskIndex].dynTaskBase;
    DynFuncHeader* dynFuncDataList = dynTaskBase->dynFuncDataList;
    DynFuncDataCache* dynFuncDataCacheList = dynTaskBase->dynFuncDataCacheList;
    DynFuncDataBackup* dynFuncDataBackupList = dynTaskBase->dynFuncDataBackupList;
    for (uint32_t dupIndex = 0; dupIndex < dynFuncDataList->funcNum; dupIndex++) {
        DevAscendFunctionDuppedData* duppedData = dynFuncDataCacheList->At(dupIndex).duppedData;
        const AddressDescriptor* backupAddr = reinterpret_cast<const AddressDescriptor*>(
            dynFuncDataBackupList->At(dupIndex).rawTensorAddrBackup);
        for (uint64_t i = 0; i < duppedData->GetIncastSize(); i++) {
            RelocDescFromCache(backupAddr[i], duppedData->GetIncastAddress(i), relocWorkspace, devStartArgs);
        }
        for (uint64_t i = 0; i < duppedData->GetOutcastSize(); i++) {
            RelocDescFromCache(backupAddr[duppedData->GetIncastSize() + i], duppedData->GetOutcastAddress(i),
                               relocWorkspace, devStartArgs);
        }
    }
    dynFuncDataList->startArgs = devStartArgs;
}

void DevControlFlowCache::RelocIncastOutcastTask(uint32_t taskIndex, uint64_t srcWorkspace, uint64_t dstWorkspace,
                                                 DevStartArgsBase* devStartArgs)
{
    // Cache memory insufficient, taskRelocTabl construction failed
    if (taskRelocTableCount != deviceTaskCount) {
        RelocIncastOutcastTaskStructural(taskIndex, srcWorkspace, dstWorkspace, devStartArgs);
        return;
    }
    uint8_t* cacheBase = reinterpret_cast<uint8_t*>(this);
    TaskRelocEntry& entry = reinterpret_cast<TaskRelocEntry*>(cacheBase + taskRelocTableOffset)[taskIndex];
    reinterpret_cast<DynFuncHeader*>(cacheBase + entry.dynFuncHeaderOffset)->startArgs = devStartArgs;
    const IncastOutcastRelocBatch* batches = reinterpret_cast<const IncastOutcastRelocBatch*>(cacheBase +
                                                                                              entry.batchListOffset);
    const uint64_t wsShift = dstWorkspace - srcWorkspace;
    for (uint32_t j = 0; j < entry.batchCount; j++) {
        const IncastOutcastRelocBatch& batch = batches[j];
        const uint16_t* idxList = reinterpret_cast<const uint16_t*>(cacheBase + batch.idxListOffset);
        const uint64_t* src = reinterpret_cast<const uint64_t*>(cacheBase + batch.srcBaseOffset);
        uint64_t* dst = reinterpret_cast<uint64_t*>(cacheBase + batch.dstBaseOffset);
        switch (batch.kind) {
            case AddressCacheKind::Workspace:
                RelocLoop<AddressCacheKind::Workspace>(batch.descCount, idxList, dst, src, wsShift, devStartArgs);
                break;
            case AddressCacheKind::Input:
                RelocLoop<AddressCacheKind::Input>(batch.descCount, idxList, dst, src, wsShift, devStartArgs);
                break;
            case AddressCacheKind::Output:
                RelocLoop<AddressCacheKind::Output>(batch.descCount, idxList, dst, src, wsShift, devStartArgs);
                break;
            case AddressCacheKind::Communication:
                RelocLoop<AddressCacheKind::Communication>(batch.descCount, idxList, dst, src, wsShift, devStartArgs);
                break;
            default:
                DEV_ERROR(ProgEncodeErr::CACHE_RELOC_KIND_INVALID,
                          "#ctrl.task.pre.cache.reloc: [RelocDescFromCache] Invalid kind: %u\n",
                          static_cast<uint32_t>(batch.kind));
                break;
        }
    }
}

void DevControlFlowCache::RuntimeAddrBackup(DeviceExecuteSlot* runtimeSlotList,
                                            ItemPool<RuntimeOutcastTensor>* runtimeOutcastTensorPool, uint64_t slotSize,
                                            uint64_t runtimeOutcastTensorSize, TensorAllocator* allocator,
                                            uint32_t parallelism)
{
    uint64_t slotDataSize = sizeof(DeviceExecuteSlot) * slotSize;
    uint64_t runtimeOutcastPoolDataSize = sizeof(ItemPool<RuntimeOutcastTensor>::ItemBlock) * runtimeOutcastTensorSize;
    DevMemcpyS(runtimeBackup.slotContext.slotList.Data(), slotDataSize, runtimeSlotList, slotDataSize);
    auto itemBlockBase = reinterpret_cast<ItemPool<RuntimeOutcastTensor>::ItemBlock*>(&runtimeOutcastTensorPool->At(0));
    DevMemcpyS(runtimeBackup.workspace.runtimeOutcastTensorPool.Data(), runtimeOutcastPoolDataSize, itemBlockBase,
               runtimeOutcastPoolDataSize);
    runtimeBackup.workspace.itemPoolMeta = runtimeOutcastTensorPool->GetMetaData();
    struct Backup {
        static void BackupBlockHeader(WsSlotAllocator::BlockHeader*& ptr, WsSlotAllocator::BlockHeader* base)
        {
            if (ptr == nullptr) {
                ptr = reinterpret_cast<WsSlotAllocator::BlockHeader*>(WsSlotAllocator::kNullBlockHeaderIndex);
                return;
            }
            ptr = reinterpret_cast<WsSlotAllocator::BlockHeader*>(static_cast<uintptr_t>(ptr - base));
        }
        static void BackupPool(WsSlotAllocator& backupAllocator, WsSlotAllocator& runtimePool,
                               WsSlotAllocator::BlockHeader* checkpointList, uint64_t checkpointOffset)
        {
            backupAllocator = runtimePool;
            uint64_t readyCount = runtimePool.initReadyCount_;
            uint64_t backupBytes = sizeof(WsSlotAllocator::BlockHeader) * readyCount;
            WsSlotAllocator::BlockHeader* runtimeBase = runtimePool.GetBlockHeaderBase();
            if (backupBytes > 0) {
                DevMemcpyS(checkpointList + checkpointOffset, backupBytes, runtimeBase, backupBytes);
            }
            BackupBlockHeader(backupAllocator.freeListHeader_, runtimeBase);
            BackupBlockHeader(backupAllocator.notInUseHeaders_, runtimeBase);
            for (uint64_t k = 0; k < readyCount; k++) {
                BackupBlockHeader(checkpointList[checkpointOffset + k].listNext, runtimeBase);
            }
        }
    };

    for (uint32_t i = 0; i < parallelism; i++) {
        runtimeBackup.workspace.tensorAllocators[i].rootInner = allocator[i].rootInner;
        runtimeBackup.workspace.tensorAllocators[i].devTaskInnerExclusiveOutcasts = allocator[i]
                                                                                        .devTaskInnerExclusiveOutcasts;

        auto& checkpointList = runtimeBackup.workspace.tensorAllocators[i].slottedOutcastsBlockList;
        uint64_t boundaryOffset = 0;
        uint64_t innerTemporalOffset = allocator[i].devTaskBoundaryOutcasts.slotNum_;
        Backup::BackupPool(runtimeBackup.workspace.tensorAllocators[i].devTaskBoundaryOutcasts,
                           allocator[i].devTaskBoundaryOutcasts, checkpointList.Data(), boundaryOffset);
        Backup::BackupPool(runtimeBackup.workspace.tensorAllocators[i].devTaskInnerTemporalOutcasts,
                           allocator[i].devTaskInnerTemporalOutcasts, checkpointList.Data(), innerTemporalOffset);
    }
}

void DevControlFlowCache::RuntimeAddrRestore(DeviceExecuteSlot* runtimeSlotList,
                                             ItemPool<RuntimeOutcastTensor>* runtimeOutcastTensorPool,
                                             uint64_t slotSize, uint64_t runtimeOutcastTensorSize,
                                             TensorAllocator* allocator, uint32_t parallelism)
{
    uint64_t slotDataSize = sizeof(DeviceExecuteSlot) * slotSize;
    uint64_t runtimeOutcastPoolDataSize = sizeof(ItemPool<RuntimeOutcastTensor>::ItemBlock) * runtimeOutcastTensorSize;
    DevMemcpyS(runtimeSlotList, slotDataSize, runtimeBackup.slotContext.slotList.Data(), slotDataSize);
    auto itemBlockBase = reinterpret_cast<ItemPool<RuntimeOutcastTensor>::ItemBlock*>(&runtimeOutcastTensorPool->At(0));
    DevMemcpyS(itemBlockBase, runtimeOutcastPoolDataSize, runtimeBackup.workspace.runtimeOutcastTensorPool.Data(),
               runtimeOutcastPoolDataSize);
    runtimeOutcastTensorPool->RestoreMetaData(runtimeBackup.workspace.itemPoolMeta);
    struct Restore {
        static void RestoreBlockHeader(WsSlotAllocator::BlockHeader*& ptr, WsSlotAllocator::BlockHeader* base,
                                       WsSlotAllocator::BlockHeader* index)
        {
            if (index == reinterpret_cast<WsSlotAllocator::BlockHeader*>(WsSlotAllocator::kNullBlockHeaderIndex)) {
                ptr = nullptr;
                return;
            }
            ptr = base + (uintptr_t)index;
        }
        static void RestoreSeqAllocator(SeqWsAllocator& dst, SeqWsAllocator& src)
        {
            dst.allocated_ = src.allocated_;
            dst.resetTimes_ = src.resetTimes_;
        }
        static void RestorePool(WsSlotAllocator& runtimePool, WsSlotAllocator& backupAllocator,
                                WsSlotAllocator::BlockHeader* checkpointList, uint64_t checkpointOffset)
        {
            runtimePool.availableSlots_ = backupAllocator.availableSlots_;
            runtimePool.initReadyCount_ = backupAllocator.initReadyCount_;
            WsSlotAllocator::BlockHeader* runtimeBase = runtimePool.GetBlockHeaderBase();
            RestoreBlockHeader(runtimePool.freeListHeader_, runtimeBase, backupAllocator.freeListHeader_);
            RestoreBlockHeader(runtimePool.notInUseHeaders_, runtimeBase, backupAllocator.notInUseHeaders_);
            for (uint64_t k = 0; k < runtimePool.initReadyCount_; k++) {
                RestoreBlockHeader(runtimeBase[k].listNext, runtimeBase, checkpointList[checkpointOffset + k].listNext);
                runtimeBase[k].ptr = checkpointList[checkpointOffset + k].ptr;
            }
        }
    };

    for (uint32_t i = 0; i < parallelism; i++) {
        Restore::RestoreSeqAllocator(allocator[i].rootInner, runtimeBackup.workspace.tensorAllocators[i].rootInner);
        Restore::RestoreSeqAllocator(allocator[i].devTaskInnerExclusiveOutcasts,
                                     runtimeBackup.workspace.tensorAllocators[i].devTaskInnerExclusiveOutcasts);

        auto& checkpointList = runtimeBackup.workspace.tensorAllocators[i].slottedOutcastsBlockList;
        uint64_t innerTemporalOffset = allocator[i].devTaskBoundaryOutcasts.slotNum_;
        Restore::RestorePool(allocator[i].devTaskBoundaryOutcasts,
                             runtimeBackup.workspace.tensorAllocators[i].devTaskBoundaryOutcasts, checkpointList.Data(),
                             0);
        Restore::RestorePool(allocator[i].devTaskInnerTemporalOutcasts,
                             runtimeBackup.workspace.tensorAllocators[i].devTaskInnerTemporalOutcasts,
                             checkpointList.Data(), innerTemporalOffset);
    }
}

void DevControlFlowCache::RuntimeAddrRelocProgram(uint64_t srcProgram, uint64_t dstProgram)
{
    RelocRange relocProgram(srcProgram, dstProgram);
    {
        auto& slotList = runtimeBackup.slotContext.slotList;
        DeviceExecuteSlot* base = slotList.Data();
        uint64_t size = slotList.size();
        for (uint64_t k = 0; k < size; k++) {
            relocProgram.RelocNullable(base[k].partialUpdate);
        }
    }
}

void DevControlFlowCache::RuntimeAddrRelocWorkspace(uint64_t srcWorkspace, uint64_t dstWorkspace,
                                                    DevStartArgsBase* devStartArgs, DeviceExecuteSlot* runtimeSlotList,
                                                    ItemPool<RuntimeOutcastTensor>::ItemBlock* runtimeOutcastTensorPool,
                                                    uint32_t parallelism, TensorAllocator* allocator)
{
    RelocRange relocWorkspace(srcWorkspace, dstWorkspace);
    /* empty constructor's overhead should be negligible */
    std::unordered_map<uint64_t, AddressDescriptor> cacheInputOutputDict;
    if (devStartArgs == nullptr) {
        RelocBuildInputOutputDesc(cacheInputOutputDict, inputTensorDataList, outputTensorDataList);
    }
    {
        for (uint32_t i = 0; i < parallelism; i++) {
            if (devStartArgs == nullptr) {
                // Host: reloc backup only (absolute addr -> workspace offset for CF cache storage).
                auto& slottedOutcastsBlockList = runtimeBackup.workspace.tensorAllocators[i].slottedOutcastsBlockList;
                WsSlotAllocator::BlockHeader* base = slottedOutcastsBlockList.Data();
                uint64_t boundaryReady = runtimeBackup.workspace.tensorAllocators[i]
                                             .devTaskBoundaryOutcasts.initReadyCount_;
                uint64_t innerReady = runtimeBackup.workspace.tensorAllocators[i]
                                          .devTaskInnerTemporalOutcasts.initReadyCount_;
                uint64_t innerOffset = runtimeBackup.workspace.tensorAllocators[i].devTaskBoundaryOutcasts.slotNum_;
                for (uint64_t k = 0; k < boundaryReady; k++) {
                    relocWorkspace.RelocNullable(base[k].ptr);
                }
                for (uint64_t k = 0; k < innerReady; k++) {
                    relocWorkspace.RelocNullable(base[innerOffset + k].ptr);
                }
            } else {
                // Device: Restore has copied backup offsets into the live freelist; reloc live only
                // so backup stays in offset form for subsequent cache hits. Use Reloc (not
                // RelocNullable): in the offset domain 0 is a valid address (workspace start),
                // skipping it leaves a null slot that Allocate hands out verbatim.
                WsSlotAllocator::BlockHeader* liveBoundaryBase = allocator[i]
                                                                     .devTaskBoundaryOutcasts.GetBlockHeaderBase();
                uint64_t liveBoundaryReady = allocator[i].devTaskBoundaryOutcasts.initReadyCount_;
                for (uint64_t k = 0; k < liveBoundaryReady; k++) {
                    relocWorkspace.Reloc(liveBoundaryBase[k].ptr);
                }
                WsSlotAllocator::BlockHeader* liveInnerBase = allocator[i]
                                                                  .devTaskInnerTemporalOutcasts.GetBlockHeaderBase();
                uint64_t liveInnerReady = allocator[i].devTaskInnerTemporalOutcasts.initReadyCount_;
                for (uint64_t k = 0; k < liveInnerReady; k++) {
                    relocWorkspace.Reloc(liveInnerBase[k].ptr);
                }
            }
        }
    }
    {
        auto& slotList = runtimeBackup.slotContext.slotList;
        DeviceExecuteSlot* base = slotList.Data();
        ItemPool<RuntimeOutcastTensor>::ItemBlock* backupRtOutcastPool = runtimeBackup.workspace
                                                                             .runtimeOutcastTensorPool.Data();
        uint64_t size = slotList.size();
        for (uint64_t k = 0; k < size; k++) {
            static_assert(sizeof(AddressDescriptor) == sizeof(uintdevptr_t),
                          "Please review the following logics when the condition does not hold anymore.");
            if (devStartArgs == nullptr) {
                // Host: addr uses backup
                if (base[k].rtOutcastIter == ITEM_POOL_INVALID_INDEX) {
                    continue;
                }

                auto& rtOutcast = backupRtOutcastPool[base[k].rtOutcastIter].Item();
                if (rtOutcast.isCache) {
                    continue;
                } // To avoid duplicate reloc
                rtOutcast.isCache = true;

                uintdevptr_t addr = rtOutcast.Addr();
                AddressDescriptor* desc = reinterpret_cast<AddressDescriptor*>(&rtOutcast.Addr());
                *desc = AddressDescriptor::MakeFromAddress(addr);
                RelocDescToCache(*desc, relocWorkspace, cacheInputOutputDict);
            } else {
                // Device: addr uses actual
                if (runtimeSlotList[k].rtOutcastIter == ITEM_POOL_INVALID_INDEX) {
                    continue;
                }

                auto& rtOutcast = runtimeOutcastTensorPool[runtimeSlotList[k].rtOutcastIter].Item();
                if (!rtOutcast.isCache) {
                    continue;
                } // To avoid duplicate reloc
                rtOutcast.isCache = false;

                AddressDescriptor* desc = reinterpret_cast<AddressDescriptor*>(&rtOutcast.allocation.ptr);
                RelocDescFromCache(*desc, *desc, relocWorkspace, devStartArgs);
                rtOutcast.Addr() = desc->GetAddressValue();
            }
        }
    }
}

void DevControlFlowCache::MixTaskDataReloc(RelocRange& relocCtrlCache, RelocRange& relocProgram,
                                           DynDeviceTaskBase* dynTaskBase, DynFuncHeader* dynFuncDataList)
{
    if (dynTaskBase->devTask.mixTaskData.wrapIdNum == 0) {
        return;
    }

    WrapInfoQueue* tmpWrapInfoQueue = reinterpret_cast<WrapInfoQueue*>(
        dynTaskBase->devTask.mixTaskData.readyWrapCoreFunctionQue);
    WrapInfoQueue*& wrapInfoQueueRef = tmpWrapInfoQueue;
    WrapInfoQueue* wrapInfoQueue = RelocControlFlowCachePointer(wrapInfoQueueRef, relocCtrlCache);
    relocCtrlCache.Reloc(dynTaskBase->devTask.mixTaskData.readyWrapCoreFunctionQue);

    WrapInfo*& wrapInfoElemRef = wrapInfoQueue->elem;
    WrapInfo* wrapInfoElem = RelocControlFlowCachePointer(wrapInfoElemRef, relocCtrlCache);
    (void)wrapInfoElem;

    MixTaskDataCache*& mixTaskDataBackupRef = dynTaskBase->mixTaskDataBackup;
    MixTaskDataCache* mixTaskDataBackup = RelocControlFlowCachePointer(mixTaskDataBackupRef, relocCtrlCache);
    WrapInfo*& wrapInfoBackupElemRef = mixTaskDataBackup->queue.elem;
    WrapInfo* wrapInfoBackupElem = RelocControlFlowCachePointer(wrapInfoBackupElemRef, relocCtrlCache);
    (void)wrapInfoBackupElem;

    for (uint32_t dupIndex = 0; dupIndex < dynFuncDataList->funcNum; dupIndex++) {
        relocProgram.Reloc(dynTaskBase->devTask.mixTaskData.opWrapList[dupIndex]);
        relocProgram.Reloc(mixTaskDataBackup->opWrapList[dupIndex]);
    }
}

void DevControlFlowCache::DieReadyQueueReloc(RelocRange& relocCtrlCache, DynDeviceTaskBase* dynTaskBase)
{
    auto processQueue = [&](uint64_t& queuePtrValue) {
        if (queuePtrValue == 0) {
            return;
        }
        ReadyCoreFunctionQueue* queueRef = reinterpret_cast<ReadyCoreFunctionQueue*>(queuePtrValue);
        ReadyCoreFunctionQueue* queue = RelocControlFlowCachePointer(queueRef, relocCtrlCache);
        relocCtrlCache.Reloc(queuePtrValue);
        if (queue != nullptr) {
            queue->Reloc(relocCtrlCache);
        }
    };
    for (size_t i = 0; i < DIE_NUM; i++) {
        processQueue(dynTaskBase->devTask.dieReadyFunctionQue.readyDieAivCoreFunctionQue[i]);
        processQueue(dynTaskBase->devTask.dieReadyFunctionQue.readyDieAicCoreFunctionQue[i]);
    }

    DieReadyQueueCache*& dieReadyQueueBackupRef = dynTaskBase->dieReadyQueueBackup;
    DieReadyQueueCache* dieReadyQueueBackup = RelocControlFlowCachePointer(dieReadyQueueBackupRef, relocCtrlCache);
    if (dieReadyQueueBackup != nullptr) {
        for (size_t i = 0; i < DIE_READY_QUEUE_SIZE * DIE_NUM; i++) {
            dieReadyQueueBackup->queueList[i].Reloc(relocCtrlCache);
        }
        for (size_t i = 0; i < DIE_READY_QUEUE_SIZE * DIE_NUM; i++) {
            relocCtrlCache.RelocNullable(dieReadyQueueBackup->pingElem[i]);
            relocCtrlCache.RelocNullable(dieReadyQueueBackup->pongElem[i]);
        }
    }
}

void DevControlFlowCache::RelocDuppedDataAndDynFuncData(RelocRange& relocProgram, RelocRange& relocCtrlCache,
                                                        DevAscendFunctionDuppedData* duppedData, DynFuncData* dynData,
                                                        DynFuncDataCache* dynDataCache,
                                                        DynFuncDataBackup* dynDataBackup)
{
    // Reloc Dupped
    relocProgram.Reloc(duppedData->source_);

    // Reloc DynFuncData
    relocProgram.Reloc(dynData->opAttrs);
    relocProgram.Reloc(dynData->opAtrrOffsets);
    relocProgram.Reloc(dynData->rawTensorDesc);

    relocCtrlCache.Reloc(dynData->exprTbl);
    relocCtrlCache.Reloc(dynData->rawTensorAddr);

    relocProgram.Reloc(dynDataCache->devFunc);
    relocProgram.Reloc(dynDataCache->calleeList);
    relocProgram.RelocNullable(dynData->cceBinaryIndexList);

    relocCtrlCache.Reloc(dynDataCache->predCount);
    relocCtrlCache.RelocNullable(dynDataCache->predCountPingPong[PRED_COUNT_PING]);
    relocCtrlCache.RelocNullable(dynDataCache->predCountPingPong[PRED_COUNT_PONG]);
    relocCtrlCache.RelocNullable(dynDataBackup->predCountBackup);
    relocCtrlCache.RelocNullable(dynDataBackup->rawTensorAddrBackup);
    relocCtrlCache.RelocNullable(dynDataBackup->deadEndHubBitmapBackup);
    relocCtrlCache.RelocNullable(dynDataBackup->tailTaskBitmapBackup);

    if (dynData->drcoRootFuncData.predCount != nullptr) {
        relocCtrlCache.Reloc(dynData->drcoRootFuncData.predCount);
        // succStaticList points at the encode-time program-resident table (DrcoEncodedSuccList),
        // nullable when the function has no static successors.
        relocProgram.RelocNullable(dynData->drcoRootFuncData.succStaticList);
        relocCtrlCache.Reloc(dynData->drcoRootFuncData.succStitchList);
        relocProgram.Reloc(dynData->drcoRootFuncData.succInfoList);
    }
}

/* Host-to-cache: devStartArgs should be nullptr. Cache-to-Device: devStartArgs should be filled */
void DevControlFlowCache::TaskAddrRelocProgramAndCtrlCache(uint64_t srcProgram, uint64_t srcCtrlCache,
                                                           uint64_t dstProgram, uint64_t dstCtrlCache)
{
    RelocRange relocCtrlCache(srcCtrlCache, dstCtrlCache);
    RelocRange relocProgram(srcProgram, dstProgram);
    for (uint64_t deviceIndex = 0; deviceIndex < deviceTaskCount; deviceIndex++) {
        /* When cached, the pointer is always legal */
        DynDeviceTaskBase*& dynTaskBaseRef = deviceTaskCacheList[deviceIndex].dynTaskBase;
        DynDeviceTaskBase* dynTaskBase = RelocControlFlowCachePointer(dynTaskBaseRef, relocCtrlCache);
        relocCtrlCache.Reloc(dynTaskBase->devTask.readyAivCoreFunctionQue);
        relocCtrlCache.Reloc(dynTaskBase->devTask.readyAicCoreFunctionQue);
        relocCtrlCache.Reloc(dynTaskBase->devTask.readyAicpuFunctionQue);

        for (size_t i = 0; i < READY_QUEUE_SIZE; i++) {
            ReadyCoreFunctionQueue*& readyQueueRef = dynTaskBase->readyQueue[i];
            ReadyCoreFunctionQueue* readyQueue = RelocControlFlowCachePointer(readyQueueRef, relocCtrlCache);
            readyQueue->Reloc(relocCtrlCache);
        }
        relocProgram.Reloc(dynTaskBase->cceBinary);
        relocProgram.Reloc(dynTaskBase->aicpuLeafBinary);

        ReadyQueueCache*& readyQueueBackupRef = dynTaskBase->readyQueueBackup;
        ReadyQueueCache* readyQueueBackup = RelocControlFlowCachePointer(readyQueueBackupRef, relocCtrlCache);
        for (size_t i = 0; i < READY_QUEUE_SIZE; i++) {
            readyQueueBackup->queueList[i].Reloc(relocCtrlCache);
        }
        for (size_t i = 0; i < READY_QUEUE_SIZE; i++) {
            relocCtrlCache.RelocNullable(readyQueueBackup->pingElem[i]);
            relocCtrlCache.RelocNullable(readyQueueBackup->pongElem[i]);
        }

        for (size_t i = 0; i < npu::tile_fwk::MAX_AICORE_NUM_FOR_QUEUE; i++) {
            if (readyQueueBackup->perCorePendingQueueList[i] != nullptr) {
                RelocControlFlowCachePointer(readyQueueBackup->perCorePendingQueueList[i], relocCtrlCache);
            }
        }

        DynFuncHeader*& dynFuncDataListRef = dynTaskBase->dynFuncDataList;
        DynFuncHeader* dynFuncDataList = RelocControlFlowCachePointer(dynFuncDataListRef, relocCtrlCache);
        relocProgram.Reloc(dynFuncDataList->cceBinary);

        RelocDrcoRootFuncList(relocCtrlCache, dynTaskBase);

        DynFuncDataCache* dynFuncDataCacheList = dynTaskBase->dynFuncDataCacheList;
        DynFuncDataBackup* dynFuncDataBackupList = dynTaskBase->dynFuncDataBackupList;
        MixTaskDataReloc(relocCtrlCache, relocProgram, dynTaskBase, dynFuncDataList);
        DieReadyQueueReloc(relocCtrlCache, dynTaskBase);
        void*& shmemWaitUntilCacheBackupRef = dynTaskBase->shmemWaitUntilCacheBackup;
        RelocControlFlowCachePointer(shmemWaitUntilCacheBackupRef, relocCtrlCache);
        for (uint32_t dupIndex = 0; dupIndex < dynFuncDataList->funcNum; dupIndex++) {
            DynFuncData* dynData = &dynFuncDataList->At(dupIndex);
            DynFuncDataCache* dynDataCache = &dynFuncDataCacheList->At(dupIndex);
            DynFuncDataBackup* dynDataBackup = &dynFuncDataBackupList->At(dupIndex);

            DevAscendFunctionDuppedData*& duppedDataRef = dynDataCache->duppedData;
            DevAscendFunctionDuppedData* duppedData = RelocControlFlowCachePointer(duppedDataRef, relocCtrlCache);

            // Reloc Stitch
            for (uint32_t i = 0; i < duppedData->GetStitchSize(); i++) {
                DevAscendFunctionDuppedStitchList& stitchList = duppedData->GetStitch(i);
                DevAscendFunctionDuppedStitch*& stitchRef = stitchList.Head();
                for (DevAscendFunctionDuppedStitch** nodePtr = &stitchRef; *nodePtr != nullptr;) {
                    DevAscendFunctionDuppedStitch* node = RelocControlFlowCachePointer(*nodePtr, relocCtrlCache);
                    nodePtr = &node->NextRaw();
                }
            }
            RelocDuppedDataAndDynFuncData(relocProgram, relocCtrlCache, duppedData, dynData, dynDataCache,
                                          dynDataBackup);
        }
    }
}

void DevControlFlowCache::RelocDrcoRootFuncList(RelocRange& relocCtrlCache, DynDeviceTaskBase* dynTaskBase)
{
    npu::tile_fwk::DrcoRootFuncList*& drcoRootFuncListRef = dynTaskBase->drcoRootFuncList;
    npu::tile_fwk::DrcoRootFuncList* drcoRootFuncList = nullptr;
    if (drcoRootFuncListRef != nullptr) {
        drcoRootFuncList = RelocControlFlowCachePointer(drcoRootFuncListRef, relocCtrlCache);
    }
    if (drcoRootFuncList != nullptr) {
        for (size_t i = 0; i < npu::tile_fwk::MAX_AICORE_NUM_FOR_QUEUE; i++) {
            relocCtrlCache.Reloc(drcoRootFuncList->perCorePendingQueueArray[i]);
        }

        for (uint32_t ct = 0; ct < npu::tile_fwk::DRCO_QUEUE_MAX; ct++) {
            for (uint32_t i = 0; i < npu::tile_fwk::NUM_LOCAL_GROUPS; i++) {
                relocCtrlCache.Reloc(drcoRootFuncList->localReadyQueueArray[ct][i]);
            }
        }
        for (uint32_t ct = 0; ct < npu::tile_fwk::DRCO_QUEUE_MAX; ct++) {
            for (uint32_t i = 0; i < npu::tile_fwk::NUM_LOCAL_GROUPS; i++) {
                relocCtrlCache.Reloc(drcoRootFuncList->localReadyMatrixArray[ct][i]);
            }
        }
        relocCtrlCache.Reloc(drcoRootFuncList->stitchNodeMatrix);
        // 基址与 stitch 节点指针同域，走同一平移；slot 内偏移相对基址，平移不变
        relocCtrlCache.Reloc(drcoRootFuncList->stitchNodeBase);
        relocCtrlCache.Reloc(drcoRootFuncList->hubTaskMatrix);
    }
}

void DevControlFlowCache::RelocMetaCache(uint64_t srcCache, uint64_t dstCache)
{
    intptr_t shift = static_cast<int64_t>(dstCache) - static_cast<int64_t>(srcCache);
    void* offset = data;
    RelocOffset(shift, offset, inputTensorDataList);
    RelocOffset(shift, offset, outputTensorDataList);
    for (uint32_t i = 0; i < SCH_DEVTASK_MAX_PARALLELISM; i++) {
        RelocOffset(shift, offset, runtimeBackup.workspace.tensorAllocators[i].slottedOutcastsBlockList);
    }
    RelocOffset(shift, offset, runtimeBackup.slotContext.slotList);
    RelocOffset(shift, offset, runtimeBackup.workspace.runtimeOutcastTensorPool);
    RelocOffset(shift, offset, deviceTaskCacheList);
    RelocOffset(shift, offset, cacheData);
}

} // namespace npu::tile_fwk::dynamic
