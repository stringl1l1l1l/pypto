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
 * \file aikernel_device_task.h
 * \brief
 */

#ifndef AIKERNEL_DEVICE_TASK_H
#define AIKERNEL_DEVICE_TASK_H

#include <atomic>
#include "tilefwk/aikernel_tensor.h"
#include "tilefwk/aikernel_drco_leaf_task_ready_queue.h"
#include "tilefwk/aikernel_root_function.h"

namespace npu::tile_fwk {

struct DevStartArgsBase;

struct DrcoRootFuncData {
    __gm__ int32_t* predCount;
    __gm__ int32_t* succStaticList;
    __gm__ DevAscendFunctionDuppedStitchNode** succStitchList;
    __gm__ DevAscendFunctionOperationSuccInfo* succInfoList;
};

struct DynFuncData {
    uint64_t exprNum;              // static
    __gm__ uint64_t* opAttrs;      // static
    __gm__ int32_t* opAtrrOffsets; // static
    __gm__ uint64_t* exprTbl;      // dyn
    __gm__ DevRawTensorDesc* rawTensorDesc;
    __gm__ uint64_t* rawTensorAddr;
    uint64_t opAttrSize;
    uint64_t rawTensorDescSize;
    uint64_t rawTensorAddrSize;
    uint64_t workspaceAddr;
    __gm__ int* cceBinaryIndexList;
    DrcoRootFuncData drcoRootFuncData;
};

struct DynFuncBin {
    uint32_t coreType;
    uint32_t psgId;
    uint64_t funcHash;
    int32_t wrapVecId{-1};
    uint8_t mixResourceType{0};
};

constexpr uint32_t MAX_AICORE_NUM_FOR_QUEUE = 108;
constexpr uint32_t NUM_LOCAL_GROUPS = (MAX_AICORE_NUM_FOR_QUEUE + LOCAL_GROUP_SIZE - 1) / LOCAL_GROUP_SIZE;
constexpr uint32_t NUM_CORE_TYPES = 2;

struct DynFuncHeader {
    uint64_t seqNo;
    uint32_t funcNum;
    uint32_t funcSize;
    __gm__ DynFuncBin* cceBinary;
    uint64_t stackWorkSpaceAddr;
    uint64_t stackWorkSpaceSize;
    __gm__ DevStartArgsBase* startArgs;

    INLINE uint64_t GetIndex() { return seqNo; }
    INLINE uint32_t Size() { return funcNum; }
    INLINE DynFuncData& At(int index) { return (reinterpret_cast<DynFuncData*>(this + 1))[index]; }
};

// 取值与 CoreType::AIV/AIC/HUB_MIX 数值对齐（aikernel_data.h 中有 static_assert 强校验）：
// 设备侧解码 taskId 的 coreType 后可直接用作队列下标，无需映射
constexpr uint32_t DRCO_QUEUE_AIV = 0; // 与 CoreType::AIV 一致
constexpr uint32_t DRCO_QUEUE_AIC = 1; // 与 CoreType::AIC 一致
constexpr uint32_t DRCO_QUEUE_MIX = 2; // 与 CoreType::HUB_MIX 一致（MIX 行承载 HUB_MIX 后继）
constexpr uint32_t DRCO_QUEUE_MAX = 3;

// 全局 stitch 节点矩阵：按核类型各一个（DRCO_QUEUE_MAX 个），每个合并对应类型的
// （× group）矩阵为单一二维数组，行按核类型内本地编号索引（AIC = blockIdx，
// AIV = blockIdx - nrValidAic，行数上限 MAX_AICORE_NUM_FOR_QUEUE），列 = LOCAL_GROUP_SIZE；
// push 只写本类型核的行，pop 只读自己类型内编号对应的行
// slot 语义为节点相对 stitch pool 基址的 u32 偏移（0 = 空闲；首对象不在偏移 0，
// 由 SlabWsAllocator::FirstObjBaseOffset 保证），基址见 DrcoRootFuncList::stitchNodeBase
struct DrcoGlobalStitchNodeMatrix {
    enum {
        COL_SIZE = 1,
    };
    // 每行独占一个 64B cache line：消除跨行 false sharing（push 写相邻行不再 invalidate
    // 本行 probe）。代价：108 行 × 64B = 6.75 KB/矩阵（×3 ≈ 20 KB），可接受
    struct alignas(64) Row {
        uint32_t slot[COL_SIZE];
        uint8_t pad[64 - sizeof(uint32_t) * COL_SIZE];
    };
    Row stitchNodeList[MAX_AICORE_NUM_FOR_QUEUE];
#ifdef __TILE_FWK_HOST__
    DrcoGlobalStitchNodeMatrix()
    {
        for (uint32_t i = 0; i < MAX_AICORE_NUM_FOR_QUEUE; i++) {
            for (uint32_t j = 0; j < COL_SIZE; j++) {
                stitchNodeList[i].slot[j] = 0;
            }
        }
    }
#endif
};

struct DrcoDevTaskFinishFlagList {
    // 每类型共享一个完成标志（原每组一个）：本 coreType 全部 leaf task 计数到 size 后置 1，
    // 本类型全部核经 DrcoGmLoad（dcci 失效读）轮询同一 flag；广播只写单个 cacheline
    struct TypeFlag {
        uint32_t devTaskFinishFlag;
        uint8_t pad[64 - sizeof(uint32_t)]; // 独占 cacheline
    };
    TypeFlag flag[DRCO_QUEUE_MAX];
#ifdef __TILE_FWK_HOST__
    DrcoDevTaskFinishFlagList()
    {
        for (uint32_t ct = 0; ct < DRCO_QUEUE_MAX; ct++) {
            flag[ct].devTaskFinishFlag = 0;
        }
    }
#endif
};

// leaf task 完成计数表：承接原 DrcoGlobalReadyQueue 的 executedCount/size 计数职责，
// 内嵌 DrcoRootFuncList（不再独立分配）；每 coreType 一项、独占 cacheline，
// 仅 executedCount 加到 size 的最后一次累加方广播 devTaskFinishFlagList
struct DrcoDevTaskCountList {
    struct CoreTypeCount {
        uint32_t size;          // 本 coreType leaf task 总数，host 初始化后只读
        uint32_t executedCount; // 已执行 leaf 计数（原子加）
        uint8_t pad[64 - 2 * sizeof(uint32_t)];
    };
    CoreTypeCount count[DRCO_QUEUE_MAX];
#ifdef __TILE_FWK_HOST__
    DrcoDevTaskCountList()
    {
        for (uint32_t ct = 0; ct < DRCO_QUEUE_MAX; ct++) {
            count[ct].size = 0;
            count[ct].executedCount = 0;
        }
    }
#endif
};

struct DrcoRootFuncList {
    __gm__ PerCorePendingQueue* perCorePendingQueueArray[MAX_AICORE_NUM_FOR_QUEUE];
    __gm__ DrcoLocalReadyQueue* localReadyQueueArray[DRCO_QUEUE_MAX][NUM_LOCAL_GROUPS];
    __gm__ DrcoLocalReadyMatrix* localReadyMatrixArray[DRCO_QUEUE_MAX][NUM_LOCAL_GROUPS];
    // 全核共享 stitch 节点矩阵：行 = 全局 blockIdx（AIC [0,nrValidAic) + AIV [nrValidAic,3*nrValidAic)），
    // 任何核可 push 任意行、每核只 pop 自己行——AIC/AIV 空闲侧可帮忙展开忙侧的 defer 节点
    // （节点展开类型无关：盲减 + 按 succCoreType 路由），消除分池时"一边排队一边空转"
    __gm__ DrcoGlobalStitchNodeMatrix* stitchNodeMatrix;
    // stitch pool 基址（设备地址）：stitchNodeList 槽位偏移以此为原点还原节点地址，
    // 写入于 InitDrcoRootFuncList，cache 激活时随矩阵指针一同 reloc（偏移本身平移不变）
    uint64_t stitchNodeBase;

    alignas(64) uint32_t totalTaskCount;
    alignas(64) uint32_t devTaskFinished;
    alignas(64) DrcoDevTaskCountList devTaskCountList;
    alignas(64) DrcoDevTaskFinishFlagList devTaskFinishFlagList;
    alignas(64) uint8_t pad[64];
};

constexpr int32_t DEVICE_TASK_QUEUE_SIZE = 64;
struct DrcoDeviceTask {
    __gm__ DynFuncHeader* dynFuncDataList;
    __gm__ DrcoRootFuncList* drcoRootFuncList;
};
struct DrcoDeviceTaskReadyQueue {
    uint32_t head;
    uint32_t tail;
    DrcoDeviceTask dynFuncDataListList[DEVICE_TASK_QUEUE_SIZE];
#ifdef __TILE_FWK_HOST__
    void Reset()
    {
        head = 0;
        tail = 0;
    }
    bool TryAppend(DynFuncHeader* dynFuncDataList, DrcoRootFuncList* rootFuncList)
    {
        if (tail - __atomic_load_n(&head, __ATOMIC_ACQUIRE) == DEVICE_TASK_QUEUE_SIZE) {
            return false;
        }
        uint32_t idx = tail % DEVICE_TASK_QUEUE_SIZE;
        dynFuncDataListList[idx].dynFuncDataList = dynFuncDataList;
        dynFuncDataListList[idx].drcoRootFuncList = rootFuncList;
        __sync_synchronize();
        __atomic_store_n(&tail, tail + 1, __ATOMIC_RELEASE);
        return true;
    }
#endif
};

} // namespace npu::tile_fwk

#endif
