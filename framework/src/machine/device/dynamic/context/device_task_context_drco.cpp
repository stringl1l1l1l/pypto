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
 * \file device_task_context_drco.cpp
 * \brief AICore dependency resolve (DRCO) helpers of DeviceTaskContext, gated by enableAicoreResolve.
 */

#include "machine/device/dynamic/context/device_task_context.h"

namespace npu::tile_fwk::dynamic {

void DeviceTaskContext::InitDrcoRootFuncList(DynDeviceTask* dyntask)
{
    auto* rootFuncList = reinterpret_cast<npu::tile_fwk::DrcoRootFuncList*>(
        ControlFlowAllocateSlab(
            devProg_, sizeof(npu::tile_fwk::DrcoRootFuncList),
            workspace_->SlabAlloc(sizeof(npu::tile_fwk::DrcoRootFuncList), WsAicpuSlabMemType::DYN_FUNC_DATA))
            .ptr);
    dyntask->drcoRootFuncList = rootFuncList;
    uint32_t queueCapacity = dyntask->devTask.coreFunctionCnt;
    uint32_t aicTaskCnt = 0;
    uint32_t aivTaskCnt = 0;
    for (uint64_t funcId = 0; funcId < dyntask->dynFuncDataCacheListSize; ++funcId) {
        auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;
        for (size_t opIndex = 0; opIndex < dyntask->dynFuncDataCacheList[funcId].devFunc->GetOperationSize();
             ++opIndex) {
            auto coreType = dyntask->cceBinary[callList[opIndex]].coreType;
            if (coreType == static_cast<int>(CoreType::AIC)) {
                aicTaskCnt++;
            } else if (coreType == static_cast<int>(CoreType::AIV)) {
                aivTaskCnt++;
            }
        }
    }
    // 完成计数表内嵌 rootFuncList（承接原全局队列的 executedCount/size），MIX 恒为 0
    new (&rootFuncList->devTaskCountList) npu::tile_fwk::DrcoDevTaskCountList();
    rootFuncList->devTaskCountList.count[npu::tile_fwk::DRCO_QUEUE_AIC].size = aicTaskCnt;
    rootFuncList->devTaskCountList.count[npu::tile_fwk::DRCO_QUEUE_AIV].size = aivTaskCnt;
    uint32_t perCoreSize = sizeof(npu::tile_fwk::PerCorePendingQueue) +
                           queueCapacity * sizeof(npu::tile_fwk::LeafTaskId);
    for (uint32_t i = 0; i < npu::tile_fwk::MAX_AICORE_NUM_FOR_QUEUE; i++) {
        auto* perCoreQueue = workspace_->AllocatePerCorePendingQueue(perCoreSize);
        new (perCoreQueue) npu::tile_fwk::PerCorePendingQueue();
        rootFuncList->perCorePendingQueueArray[i] = perCoreQueue;
    }
    uint32_t nrValidAic = devProg_->devArgs.nrValidAic;
    constexpr uint32_t groupSize = npu::tile_fwk::LOCAL_GROUP_SIZE;
    uint32_t localSize = sizeof(npu::tile_fwk::DrcoLocalReadyQueue);
    localSize += queueCapacity * sizeof(npu::tile_fwk::LeafTaskId);
    for (uint32_t ct = 0; ct < npu::tile_fwk::DRCO_QUEUE_MAX; ct++) {
        for (uint32_t i = 0; i < npu::tile_fwk::NUM_LOCAL_GROUPS; i++) {
            auto* localQueue = workspace_->AllocateDrcoLocalReadyQueue(localSize);
            new (localQueue) npu::tile_fwk::DrcoLocalReadyQueue(queueCapacity);
            (void)memset_s(reinterpret_cast<uint8_t*>(localQueue) + sizeof(DrcoLocalReadyQueue),
                           queueCapacity * sizeof(LeafTaskId), 0, queueCapacity * sizeof(LeafTaskId));
            rootFuncList->localReadyQueueArray[ct][i] = localQueue;
        }
    }
    // LocalMatrix 分配：数组按类型内本地编号索引（AIC 本地编号 = blockIdx ∈ [0, nrValidAic)，
    // AIV 本地编号 = blockIdx - nrValidAic ∈ [0, 2*nrValidAic)），组内本地编号 [i*N, (i+1)*N)
    // 连续映射到列 [0, N)；validCoreNum = 该组内该类型核数，尾组不足 N 时为剩余数，
    // 天然构成列 [0, validCoreNum) 前缀（无 MIX 边界组问题）
    auto matrixValidCoreNum = [groupSize](uint32_t groupIdx, uint32_t coreCount) -> uint32_t {
        uint32_t valid = coreCount > groupIdx * groupSize ? coreCount - groupIdx * groupSize : 0;
        return valid > groupSize ? groupSize : valid;
    };
    for (uint32_t ct = 0; ct < npu::tile_fwk::DRCO_QUEUE_MAX; ct++) {
        for (uint32_t i = 0; i < npu::tile_fwk::NUM_LOCAL_GROUPS; i++) {
            uint32_t validCoreNum = 0;
            if (ct == npu::tile_fwk::DRCO_QUEUE_AIC || ct == npu::tile_fwk::DRCO_QUEUE_MIX) {
                validCoreNum = matrixValidCoreNum(i, nrValidAic);
            } else if (ct == npu::tile_fwk::DRCO_QUEUE_AIV) {
                validCoreNum = matrixValidCoreNum(i, nrValidAic * 2);
            }
            auto* localMatrix = workspace_->AllocateDrcoLocalReadyMatrix(sizeof(npu::tile_fwk::DrcoLocalReadyMatrix));
            new (localMatrix) npu::tile_fwk::DrcoLocalReadyMatrix(validCoreNum);
            rootFuncList->localReadyMatrixArray[ct][i] = localMatrix;
        }
    }
    // 全局共享 stitch 节点矩阵（单实例）：行 = 全局 blockIdx（AIC [0,nrValidAic) + AIV
    // [nrValidAic,3*nrValidAic)），任何核可 push 任意行、每核只 pop 自己行；
    // slot 语义为节点相对 stitch pool 基址的 u32 偏移（0 = 空闲），基址随 rootFuncList
    // 透传给 aicore 侧
    auto* stitchNodeMatrix = workspace_->AllocateDrcoStitchNodeMatrix(
        sizeof(npu::tile_fwk::DrcoGlobalStitchNodeMatrix));
    new (stitchNodeMatrix) npu::tile_fwk::DrcoGlobalStitchNodeMatrix();
    rootFuncList->stitchNodeMatrix = stitchNodeMatrix;
    rootFuncList->stitchNodeBase = workspace_->GetStitchPoolBase();
    rootFuncList->totalTaskCount = dyntask->devTask.coreFunctionCnt;
    rootFuncList->devTaskFinished = 0;
    new (&rootFuncList->devTaskFinishFlagList) npu::tile_fwk::DrcoDevTaskFinishFlagList();
}

void DeviceTaskContext::DispatchReadyQueueToCores(DynDeviceTask* dyntask, DevAscendProgram* devProg)
{
    ReadyCoreFunctionQueue* aivQueue = dyntask->readyQueue[DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIV)];
    ReadyCoreFunctionQueue* aicQueue = dyntask->readyQueue[DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIC)];

    uint32_t nrValidAic = devProg->devArgs.nrValidAic;
    uint32_t nrAivCores = 2 * nrValidAic;

    // perCorePendingQueue: distribute wrap (mix) tasks, AIC task to core w, AIV tasks to paired vector cores
    WrapInfoQueue* wrapQueue = reinterpret_cast<WrapInfoQueue*>(dyntask->devTask.mixTaskData.readyWrapCoreFunctionQue);
    auto* perCorePendingQueueArray = dyntask->drcoRootFuncList->perCorePendingQueueArray;
    uint32_t wrapCoreIdx = 0;
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

    // perCorePendingQueue: distribute aic tasks to aicore cores (0..nrValidAic-1), continue from wrap dispatch
    uint32_t aicIdx = wrapCoreIdx;
    for (const auto* it = aicQueue->begin(); it != aicQueue->end(); ++it) {
        uint32_t coreIdx = aicIdx % nrValidAic;
        perCorePendingQueueArray[coreIdx]->UnsafeEnqueue(*it);
        aicIdx++;
    }

    // perCorePendingQueue: distribute aiv tasks to aiv cores (nrValidAic..3*nrValidAic-1)
    uint32_t aivIdx = wrapCoreIdx * 2;
    for (const auto* it = aivQueue->begin(); it != aivQueue->end(); ++it) {
        uint32_t coreIdx = nrValidAic + (aivIdx % nrAivCores);
        perCorePendingQueueArray[coreIdx]->UnsafeEnqueue(*it);
        aivIdx++;
    }
}

void DeviceTaskContext::DispatchDieReadyQueueToCores(DynDeviceTask* dyntask, DevAscendProgram* devProg)
{
    if (!IsMultiDie(devProg)) {
        return;
    }
    uint32_t nrValidAic = devProg->devArgs.nrValidAic;
    uint32_t nrAivCores = 2 * nrValidAic;
    uint32_t halfAic = nrValidAic / 2;
    uint32_t halfAiv = nrAivCores / 2;
    // perCorePendingQueue: distribute die tasks, die0 to first-half cores, die1 to second-half cores
    for (uint32_t dieId = 0; dieId < DIE_NUM; ++dieId) {
        uint32_t aicCoreBase = dieId * halfAic;
        uint32_t aicCoreCnt = (dieId == 0) ? halfAic : (nrValidAic - halfAic);
        ReadyCoreFunctionQueue* dieAicQue = reinterpret_cast<ReadyCoreFunctionQueue*>(
            dyntask->devTask.dieReadyFunctionQue.readyDieAicCoreFunctionQue[dieId]);
        uint32_t dieAicIdx = 0;
        for (const auto* it = dieAicQue->begin(); it != dieAicQue->end(); ++it) {
            uint32_t coreIdx = aicCoreBase + (dieAicIdx % aicCoreCnt);
            dyntask->drcoRootFuncList->perCorePendingQueueArray[coreIdx]->UnsafeEnqueue(*it);
            dieAicIdx++;
        }

        uint32_t aivCoreBase = nrValidAic + dieId * halfAiv;
        uint32_t aivCoreCnt = (dieId == 0) ? halfAiv : (nrAivCores - halfAiv);
        ReadyCoreFunctionQueue* dieAivQue = reinterpret_cast<ReadyCoreFunctionQueue*>(
            dyntask->devTask.dieReadyFunctionQue.readyDieAivCoreFunctionQue[dieId]);
        uint32_t dieAivIdx = 0;
        for (const auto* it = dieAivQue->begin(); it != dieAivQue->end(); ++it) {
            uint32_t coreIdx = aivCoreBase + (dieAivIdx % aivCoreCnt);
            dyntask->drcoRootFuncList->perCorePendingQueueArray[coreIdx]->UnsafeEnqueue(*it);
            dieAivIdx++;
        }
    }
}

void DeviceTaskContext::BuildDrcoRootFuncData(DynFuncData* dyndata, DevAscendFunctionDupped& stitchedFunc)
{
    DevAscendFunction* source = stitchedFunc.GetSource();
    uint32_t opSize = stitchedFunc.GetOperationSize();
    // AICore drco uses int32_t predCount because aicore atomicAdd only supports 32bit/64bit operands,
    // while the aicpu-side dupped data predCount stays uint16_t (sizeof(predcount_t)).
    uint32_t drcoPredCountSize = opSize * sizeof(int32_t);
    WsAllocation predAlloc = ControlFlowAllocateSlab(
        devProg_, drcoPredCountSize, workspace_->SlabAlloc(drcoPredCountSize, WsAicpuSlabMemType::PRED_COUNT));
    int32_t* aicorePredCount = predAlloc.As<int32_t>();
    // DRCO pair resolve does u64 atomicAdd on &predCount[even opIdx] (aikernel_data.h). The slab
    // allocator's 64B granularity keeps the base 64B aligned so even slots stay 8B aligned; an
    // unaligned base would fault the hardware atomic, so guard the implicit contract here.
    DEV_ASSERT_MSG(ProgEncodeErr::DYNFUNC_DATA_ALIGNMENT_ERROR, (predAlloc.ptr % 8ULL) == 0,
                   "#drco.predcount.align: base=%llu is not 8B aligned",
                   static_cast<unsigned long long>(predAlloc.ptr));
    for (uint32_t i = 0; i < opSize; ++i) {
        aicorePredCount[i] = static_cast<int32_t>(stitchedFunc.GetOperationCurrPredCount(i));
    }
    auto* rootFuncData = &dyndata->drcoRootFuncData;
    rootFuncData->predCount = aicorePredCount;
    // DRCO consumes the encode-time coreType-encoded successor table (program memory, zero copy).
    // The source operationSuccList_ stays untouched (shared with DRCU/aicpu resolve).
    rootFuncData->succStaticList = source->GetDrcoEncodedSuccAddr();
    // Stitch successors are encoded at build time (MakeDrcoStitchTaskId in device_stitch_context),
    // the shared linked nodes are consumed directly, no flattened copy here.
    rootFuncData->succStitchList = &stitchedFunc.DupDataForDynFuncData()->GetStitch(0).Head();
    rootFuncData->succInfoList = &source->GetOperationSuccInfo(0);
    dyndata->cceBinaryIndexList = source->GetCalleeIndexAddr();
}

} // namespace npu::tile_fwk::dynamic
