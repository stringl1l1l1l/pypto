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
 * \file aicore_state_drco.h
 * \brief DRCO (Dependency Resolving by aiCOre) scheduling flow for aicore_state.h.
 */

#ifndef AICORE_ENTRY_DRCO_H
#define AICORE_ENTRY_DRCO_H

// ASC printf 走上游 gating 头（opt-in：PYPTO_ENABLE_AICORE_PRINT=true)
#include "tilefwk/aicore_asc_printf.h"

#ifdef __DAV_C310__

#if defined(__MIX__)
#define IS_MIX 1
#else
#define IS_MIX 0
#endif

#if defined(__AIV__)
#define IS_AIV 1
#else
#define IS_AIV 0
#endif

#if defined(__AIC__)
#define IS_AIC 1
#else
#define IS_AIC 0
#endif

namespace npu::tile_fwk {

#if defined(__AIV__)
constexpr uint32_t DRCO_CORE_TYPE = static_cast<uint32_t>(CoreType::AIV);
#elif defined(__AIC__)
constexpr uint32_t DRCO_CORE_TYPE = static_cast<uint32_t>(CoreType::AIC);
#endif

} // namespace npu::tile_fwk

using npu::tile_fwk::DRCO_CORE_TYPE;
using npu::tile_fwk::LOCAL_GROUP_SIZE;

using npu::tile_fwk::FuncID;
using npu::tile_fwk::TaskID;

#define DRCO_DCCI_SINGLE_CACHE_LINE(ptr) dcci((__gm__ uint8_t*)ptr, SINGLE_CACHE_LINE, CACHELINE_OUT)
#define DRCO_DCCI_ENTIRE_DATA_CACHE() dcci((__gm__ void*)0, ENTIRE_DATA_CACHE, CACHELINE_OUT)

#define DRCO_BUSY_BACKOFF_CYC 500
// 共享 stitch 矩阵的全核行域倍率：AIC [0,nrValidAic) + AIV [nrValidAic, 3*nrValidAic)，
// 即全核数 = AIC 1 倍 + AIV 2 倍；矩阵行 = 全局 blockIdx，行数上限钳到 MAX_AICORE_NUM_FOR_QUEUE
constexpr uint32_t DRCO_ALL_CORES_PER_AIC = 3;

#if ENABLE_AICORE_PRINT
#define DRCO_LOG(ctx, fmt, ...) AICORE_LOGE((ctx)->logger.Context(), fmt, ##__VA_ARGS__)
#else
#define DRCO_LOG(ctx, fmt, ...) \
    do {                        \
    } while (0)
#endif

INLINE void DrcoBusyBackOff()
{
    uint64_t t0 = get_sys_cnt();
    while (get_sys_cnt() - t0 < DRCO_BUSY_BACKOFF_CYC) {
    }
}

#define DRCO_LEADER_TIMEOUT_CHECK(t0, loopCount, timelen, lastStatus) \
    ++loopCount;                                                      \
    if ((loopCount % 1000 == 0)) {                                    \
        uint64_t elapsed = get_sys_cnt() - t0;                        \
        if (!warningSet && elapsed > AICORE_WARNING_CYCLES) {         \
            SetWarningStatus(state->args, lastStatus);                \
            warningSet = true;                                        \
        }                                                             \
        if (elapsed > (timelen)) {                                    \
            SetLastWordStatus(state->args, lastStatus);               \
            SyncAllMix();                                             \
            Trap();                                                   \
            return nullptr;                                           \
        }                                                             \
    }

constexpr uint16_t SYNC_MODE_SHIFT_VALUE = 4;
constexpr uint16_t SYNC_FLAG_SHIFT_VALUE = 8;
constexpr uint32_t BATCH_PUSH_BUF_SIZE = 6;
constexpr uint32_t HUB_STACK_SIZE = 64;

__aicore__ inline uint16_t GetffstMsg(uint16_t mode, uint16_t flagId)
{
    return (0x1 + ((mode & 0x3) << SYNC_MODE_SHIFT_VALUE) + ((flagId & 0xf) << SYNC_FLAG_SHIFT_VALUE));
}

constexpr uint16_t SYNC_AIC_FLAG = 11;
constexpr uint16_t SYNC_AIV_FLAG = 12;
constexpr uint16_t SYNC_AIC_AIV_FLAG = 13;
constexpr uint16_t SYNC_FLAG_ID_MAX = 16;

__aicore__ inline void SyncAllMix()
{
#if IS_AICORE
    pipe_barrier(PIPE_ALL);
#if defined(__DAV_CUBE__)
    wait_intra_block(PIPE_S, SYNC_AIV_FLAG);
    wait_intra_block(PIPE_S, SYNC_AIV_FLAG + SYNC_FLAG_ID_MAX);
    ffts_cross_core_sync(PIPE_FIX, GetffstMsg(0, SYNC_AIC_FLAG));
    wait_flag_dev(PIPE_S, SYNC_AIC_FLAG);
    set_intra_block(PIPE_S, SYNC_AIC_AIV_FLAG);
    set_intra_block(PIPE_S, SYNC_AIC_AIV_FLAG + SYNC_FLAG_ID_MAX);
#elif defined(__DAV_VEC__)
    set_intra_block(PIPE_MTE3, SYNC_AIV_FLAG);
    wait_intra_block(PIPE_S, SYNC_AIC_AIV_FLAG);
#endif
    pipe_barrier(PIPE_ALL);
#endif
}

template <typename T>
INLINE T DrcoGmLoad(__gm__ T* ptr)
{
    DRCO_DCCI_SINGLE_CACHE_LINE(ptr);
    return *ptr;
}

template <typename T>
INLINE void DrcoGmStore(__gm__ T* ptr, T value)
{
    *ptr = value;
    DRCO_DCCI_SINGLE_CACHE_LINE(ptr);
}

template <typename T>
INLINE T DrcoGmLoadArray(__gm__ T* ptr, uint32_t idx)
{
    DRCO_DCCI_SINGLE_CACHE_LINE(&ptr[idx]);
    return ptr[idx];
}

template <typename T>
INLINE void DrcoGmStoreArray(__gm__ T* ptr, uint32_t idx, T value)
{
    ptr[idx] = value;
    DRCO_DCCI_SINGLE_CACHE_LINE(&ptr[idx]);
}

INLINE uint32_t DrcoAtomicLoad(__gm__ uint32_t* ptr) { return static_cast<uint32_t>(atomicCAS(ptr, 0, 0)); }

INLINE uint32_t DrcoAtomicAddToS32(__gm__ int32_t* ptr, int32_t value)
{
    return static_cast<uint32_t>(atomicAdd(ptr, value));
}

INLINE uint32_t DrcoAtomicAddToU32(__gm__ uint32_t* ptr, uint32_t value)
{
    return static_cast<uint32_t>(atomicAdd(ptr, value));
}

INLINE uint32_t DrcoAtomicCasToU32(__gm__ uint32_t* ptr, uint32_t compare, uint32_t value)
{
    return static_cast<uint32_t>(atomicCAS(ptr, compare, value));
}

INLINE uint32_t DrcoAtomicExchToU32(__gm__ uint32_t* ptr, uint32_t value)
{
    return static_cast<uint32_t>(atomicExch(ptr, value));
}

using DrcoDeviceTask = npu::tile_fwk::DrcoDeviceTask;
using DrcoDeviceTaskReadyQueue = npu::tile_fwk::DrcoDeviceTaskReadyQueue;
using DrcoLocalReadyQueue = npu::tile_fwk::DrcoLocalReadyQueue;
using DrcoLocalReadyMatrix = npu::tile_fwk::DrcoLocalReadyMatrix;
using DrcoGlobalStitchNodeMatrix = npu::tile_fwk::DrcoGlobalStitchNodeMatrix;

// 不确定bisheng在处理小于 64bit 的结构体的时候，是否能够完全保存在寄存器中，先用 uint64_t + bit 计算的方式处理
typedef uint64_t BlockDesc;

template <uint64_t lo, uint64_t hi>
INLINE uint64_t BlockDescInsert(BlockDesc data)
{
    return (data & (((uint64_t)1 << (hi - lo)) - 1)) << lo;
}
template <uint64_t lo, uint64_t hi>
INLINE uint64_t BlockDescExtract(BlockDesc data)
{
    return (data >> lo) & (((uint64_t)1 << (hi - lo)) - 1);
}
INLINE uint64_t BlockDescExtract(BlockDesc data, uint64_t lo, uint64_t hi)
{
    return (data >> lo) & (((uint64_t)1 << (hi - lo)) - 1);
}

INLINE BlockDesc BlockDescCreate(uint64_t blockIdx, uint64_t typedBlockIdx, uint64_t validAicNum, uint64_t validAivNum)
{
    return BlockDescInsert<24, 32>(blockIdx) | BlockDescInsert<16, 24>(typedBlockIdx) |
           BlockDescInsert<8, 16>(validAivNum) | BlockDescInsert<0, 8>(validAicNum);
}
INLINE uint32_t BlockDescBlockIdx(BlockDesc desc) { return BlockDescExtract<24, 32>(desc); }
INLINE uint32_t BlockDescTypedBlockIdx(BlockDesc desc) { return BlockDescExtract<16, 24>(desc); }
INLINE uint32_t BlockDescValidCoreNum(BlockDesc desc, bool isAiv)
{
    return BlockDescExtract(desc, 0 + isAiv * 8, 8 + isAiv * 8);
}

#define ENABLE_AICORE_TRACE 0

struct DrcoEntryState {
    BlockDesc blockDesc;
    __gm__ KernelArgs* args;
    __gm__ Metrics* metric;
    ExecuteContext ctx;
    __gm__ npu::tile_fwk::RuntimeDataRingBufferHeadData* runtimeDataRingBufferHeadData;
    __gm__ npu::tile_fwk::DevStartArgsBase* base;
    __gm__ DrcoDeviceTaskReadyQueue* deviceTaskReadyQueue;
    uint8_t lastMixResourceType;

    uint32_t readyMatrixPushGroupIndex;
    uint32_t readyMatrixPushColIdx;
    uint32_t readyMatrixPopRowIndex;
    bool isExectedLeafTask{false};

#if ENABLE_AICORE_TRACE
    struct TraceEventStatistic {
        uint32_t taskIndex;
        struct TraceEvent {
            uint64_t timestamp;
            uint32_t taskId;

#define EVENT_SIZE 0x400
#define EVENT(hi, lo) (((hi) << 16) | ((lo) & 0xffff))
#define EVENT_LEAF_START() EVENT(1, 0x1001)
#define EVENT_LEAF_END() EVENT(2, 0x1002)
#define EVENT_RESOLVE() EVENT(3, 0)
#define EVENT_PUSH_LOCAL(group) EVENT(4, group)
#define EVENT_PUSH_GLOBAL() EVENT(5, 0)
#define EVENT_SUCC_STATIC(size) EVENT(6, size)
#define EVENT_SUCC_DYNAMIC(size) EVENT(7, size)
#define EVENT_STITCH_NODE(size) EVENT(8, size)
#define EVENT_FETCH_START() EVENT(9, 0x1000)
#define EVENT_FETCH_END() EVENT(9, 0x2000)
            uint32_t eventCode;
        } traceEventList[EVENT_SIZE];
    } traceEventStatistic;
#endif
};
#define FUNCID_TASKID(id) FuncID(id), TaskID(id)

#if ENABLE_AICORE_TRACE
INLINE static void _TraceInit(DrcoEntryState* state) { state->traceEventStatistic.taskIndex = 0; }

INLINE static void _TraceEvent(DrcoEntryState* state, uint32_t taskId, uint32_t eventCode)
{
    if (state->traceEventStatistic.taskIndex < EVENT_SIZE) {
        auto* code = &state->traceEventStatistic.traceEventList[state->traceEventStatistic.taskIndex++];
        code->timestamp = get_sys_cnt();
        code->taskId = taskId;
        code->eventCode = eventCode;
    }
}

INLINE static void _TracePrint(DrcoEntryState* state)
{
    auto traceEventStatistic = &state->traceEventStatistic;
    PYPTO_AICORE_PRINTF("total=%d", traceEventStatistic->taskIndex);
    for (uint32_t i = 0; i < traceEventStatistic->taskIndex; i++) {
        auto event = &traceEventStatistic->traceEventList[i];
        PYPTO_AICORE_PRINTF("timestamp=%llu, code=%x taskId=%d:%d duration=%llu", (unsigned long long)event->timestamp,
                            event->eventCode, FUNCID_TASKID(event->taskId),
                            i == 0 ? 0 : (event->timestamp - traceEventStatistic->traceEventList[i - 1].timestamp));
    }
}
#define TraceInit(state) _TraceInit(state)
#define TraceEvent(state, taskId, eventCode) _TraceEvent(state, taskId, eventCode)
#define TracePrint(state) _TracePrint(state)
#else
#define TraceInit(state)
#define TraceEvent(state, taskId, eventCode)
#define TracePrint(state)
#endif

INLINE __gm__ DrcoDeviceTask* GetCurrentDeviceTask(__gm__ DrcoDeviceTaskReadyQueue* queue)
{
    uint32_t head = DrcoGmLoad(&queue->head);
    uint32_t tail = DrcoGmLoad(&queue->tail);
    if (head < tail) {
        __gm__ DrcoDeviceTask* elem = &queue->dynFuncDataListList[head % npu::tile_fwk::DEVICE_TASK_QUEUE_SIZE];
        if (DrcoGmLoad(&elem->dynFuncDataList) != nullptr) {
            return elem;
        }
    }
    return nullptr;
}

template <typename T, T SENTINEL, T DEC>
INLINE T DrcoAtomicResolveDependOnce(__gm__ T* ptr)
{
    if (*ptr == SENTINEL) {
        return SENTINEL;
    }
    return static_cast<T>(atomicAdd(ptr, DEC));
}

// ==================== LocalReadyMatrix 原语：push 遍历一行 / pop 遍历一列 ====================

INLINE uint32_t DrcoLocalReadyMatrixGetRowIdx(uint32_t localIdx) { return localIdx % npu::tile_fwk::LOCAL_GROUP_SIZE; }

// 遍历 rowIdx 行的有效列 [0, validCoreNum)，依次 CAS 写入多个任务，返回实际写入个数；
// 每个 CAS 成功的任务记录 EVENT_PUSH_LOCAL(groupIdx) 事件（打点写入的 group）
// 写入流程 4：多余的任务依次写到后续核的矩阵，具体行由 coreIdx % N 决定；只写有效列，无主列不写
INLINE uint32_t DrcoLocalReadyMatrixPushBatch(DrcoEntryState* state, __gm__ DrcoLocalReadyMatrix* matrix,
                                              uint32_t groupIdx, uint32_t rowIdx, uint32_t* readyTaskList, uint32_t n,
                                              bool* rowExhausted)
{
    uint32_t validCoreNum = matrix->validCoreNum;
    uint32_t pushed = 0;
    uint32_t col = state->readyMatrixPushColIdx % validCoreNum;
    for (; col < validCoreNum && pushed < n; col++) {
        uint32_t prev = DrcoAtomicCasToU32(&matrix->taskList[rowIdx][col], 0, DRCO_ENCODE_TASK(readyTaskList[pushed]));
        if (prev == 0) {
            TraceEvent(state, readyTaskList[pushed], EVENT_PUSH_LOCAL(groupIdx));
            pushed++;
        }
    }
    state->readyMatrixPushColIdx = col;
    *rowExhausted = (col >= validCoreNum);
    return pushed;
}

// 遍历 colIdx 整列，把所有待执行任务依次 pop 出来，CAS(task -> 0) 抢占成功即独占该任务；
// 从 state->matrixRowIndex（核本地游标，前一次 pop 结束处）起环形扫描，摊平重复扫描开销。
// 每核只 pop 自己固定的一列（colIdx = 本地编号 % N），单游标即该列游标；
// maxCount 限制本次最多取出的个数；游标保存为最后扫描行的下一行，未扫到的行下次优先
INLINE uint32_t DrcoLocalReadyMatrixPopColTasks(DrcoEntryState* state, __gm__ DrcoLocalReadyMatrix* matrix,
                                                uint32_t colIdx, uint32_t* outTaskList, uint32_t maxCount)
{
    constexpr uint32_t rowCnt = npu::tile_fwk::LOCAL_GROUP_SIZE;
    uint32_t count = 0;
    uint32_t start = state->readyMatrixPopRowIndex;
    uint32_t next = start;
    for (uint32_t i = 0; i < rowCnt && count < maxCount; i++) {
        uint32_t row = (start + i) % rowCnt;
        next = (row + 1) % rowCnt;
        uint32_t taskId = DrcoAtomicCasToU32(&matrix->taskList[row][colIdx], 0, 0);
        if (taskId == 0) {
            continue;
        }
        if (DrcoAtomicCasToU32(&matrix->taskList[row][colIdx], taskId, 0) == taskId) {
            outTaskList[count++] = DRCO_DECODE_TASK(taskId);
        }
    }
    state->readyMatrixPopRowIndex = next;
    return count;
}

// ==================== LocalReadyQueue 原语 ====================

INLINE int DrcoLocalReadyQueueTryPushTask(__gm__ DrcoLocalReadyQueue* queue, uint32_t readyTask)
{
    __gm__ uint32_t* tailPtr = &queue->tail;
    uint32_t tail = DrcoAtomicLoad(tailPtr);
    if (tail >= DrcoGmLoad(&queue->size)) {
        return -1;
    }
    uint32_t tailPrev = DrcoAtomicCasToU32(&queue->tail, tail, tail + 1);
    if (tailPrev == tail) {
        DrcoAtomicExchToU32(&queue->taskList[tailPrev], DRCO_ENCODE_TASK(readyTask));
        return 0;
    }
    return 1;
}

INLINE bool DrcoLocalReadyQueuePushTask(__gm__ DrcoLocalReadyQueue* queue, uint32_t readyTask)
{
    int result = DrcoLocalReadyQueueTryPushTask(queue, readyTask);
    while (result == 1) {
        result = DrcoLocalReadyQueueTryPushTask(queue, readyTask);
    }
    return result == 0;
}

INLINE int DrcoLocalReadyQueueTryBatchPushTask(__gm__ DrcoLocalReadyQueue* queue, uint32_t* readyTaskList, uint32_t n)
{
    __gm__ uint32_t* tailPtr = &queue->tail;
    uint32_t tail = DrcoAtomicLoad(tailPtr);
    if (tail >= DrcoGmLoad(&queue->size)) {
        return -1;
    }
    uint32_t tailPrev = DrcoAtomicCasToU32(&queue->tail, tail, tail + n);
    if (tailPrev == tail) {
        for (uint32_t i = 0; i < n; i++) {
            DrcoAtomicExchToU32(&queue->taskList[tail + i], DRCO_ENCODE_TASK(readyTaskList[i]));
        }
        return 0;
    }
    return 1;
}

INLINE void DrcoLocalReadyQueuePushBatch(__gm__ DrcoLocalReadyQueue* queue, uint32_t* readyTaskList, uint32_t n)
{
    int result = DrcoLocalReadyQueueTryBatchPushTask(queue, readyTaskList, n);
    while (result == 1) {
        result = DrcoLocalReadyQueueTryBatchPushTask(queue, readyTaskList, n);
    }
}

INLINE uint32_t DrcoLocalReadyQueueGetFirstTask(__gm__ DrcoLocalReadyQueue* queue)
{
    uint32_t head = DrcoAtomicLoad(&queue->head);
    uint32_t tail = DrcoAtomicLoad(&queue->tail);
    if (head >= tail) {
        return static_cast<uint32_t>(AICORE_TASK_NO_INCOME);
    }
    uint32_t headPrev = DrcoAtomicCasToU32(&queue->head, head, head + 1);
    if (headPrev != head) {
        return static_cast<uint32_t>(AICORE_TASK_FETCH_CONFLICT);
    }
    uint32_t taskId = DrcoAtomicLoad(&queue->taskList[headPrev]);
    while (taskId == 0) {
        DrcoBusyBackOff();
        taskId = DrcoAtomicLoad(&queue->taskList[headPrev]);
    }
    return DRCO_DECODE_TASK(taskId);
}

// 类型内本地编号：blockIdx < nrValidAic（AIC 核）为 blockIdx，否则（AIV 核）为 blockIdx - nrValidAic。
// LocalQueue/LocalMatrix 统一按此编号索引：group = 本地编号 / N，行/列 = 本地编号 % N
INLINE uint32_t DrcoGetCoreTypedIdx(uint32_t blockIdx, uint32_t nrValidAic)
{
    return blockIdx < nrValidAic ? blockIdx : blockIdx - nrValidAic;
}

// ==================== RootFuncList 原语 ====================

// leaf task 计数并广播完成：累加本 coreType 计数表 executedCount
// 仅加到 size 的最后一次累加方置 1 本 coreType 的 devTaskFinishFlag
// 本类型全部核经 DrcoGmLoad（含 dcci 失效）轮询同一标志
INLINE void DrcoNotifyTaskExecutedAdd([[maybe_unused]] DrcoEntryState* state,
                                      __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList, uint32_t addCount)
{
    __gm__ npu::tile_fwk::DrcoDevTaskCountList::CoreTypeCount* cnt = &rootFuncList->devTaskCountList
                                                                          .count[DRCO_CORE_TYPE];
    if (DrcoAtomicAddToU32(&cnt->executedCount, addCount) + addCount < DrcoGmLoad(&cnt->size)) {
        return;
    }
    DrcoGmStore(&rootFuncList->devTaskFinishFlagList.flag[DRCO_CORE_TYPE].devTaskFinishFlag, uint32_t(1));
}

INLINE __gm__ DrcoLocalReadyQueue* DrcoRootFuncListGetLocalReadyQueue(
    __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList, uint32_t readyQueueCoreType, uint32_t groupIdx)
{
    if (readyQueueCoreType < npu::tile_fwk::DRCO_QUEUE_MAX && groupIdx < npu::tile_fwk::NUM_LOCAL_GROUPS) {
        return rootFuncList->localReadyQueueArray[readyQueueCoreType][groupIdx];
    }
    return nullptr;
}

INLINE __gm__ DrcoLocalReadyMatrix* DrcoRootFuncListGetLocalReadyMatrix(
    __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList, uint32_t readyMatrixCoreType, uint32_t groupIdx)
{
    if (readyMatrixCoreType < npu::tile_fwk::DRCO_QUEUE_MAX && groupIdx < npu::tile_fwk::NUM_LOCAL_GROUPS) {
        return rootFuncList->localReadyMatrixArray[readyMatrixCoreType][groupIdx];
    }
    return nullptr;
}

// 全核共享 stitch 矩阵（行 = 全局 blockIdx）：AIC/AIV 任意核可 push/pop，
// 展开动作类型无关，空闲侧核帮忙消化忙侧 defer 链
INLINE __gm__ DrcoGlobalStitchNodeMatrix* DrcoRootFuncListGetStitchNodeMatrix(
    __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList)
{
    return rootFuncList->stitchNodeMatrix;
}

// 类型内本地编号模型（AIC: blockIdx ∈ [0, aicCoreNum)，AIV: blockIdx - aicCoreNum ∈ [0, 2*aicCoreNum)）
// 下各队列行实际使用的 local ready queue/matrix 组区间为 [groupBeg, groupEnd)：两类均自组 0 起
// （AIC/MIX 组数 ceil(aicCoreNum/N)，AIV 组数 ceil(2*aicCoreNum/N)），区间外的组既无 push 亦无 poll，
// 其消费者为 AIC 核，组区间与 AIC 行一致。
// 仅在 InitDrcoEntry 调用一次，结果缓存在 ExecuteContext::drcoGroupBeg/End，热路径直接查表
INLINE void GetDrcoActiveGroupRange(uint32_t queueType, uint32_t aicCoreNum, uint32_t& groupBeg, uint32_t& groupEnd)
{
    if (queueType == npu::tile_fwk::DRCO_QUEUE_AIV) {
        groupBeg = 0;
        groupEnd = (aicCoreNum * 2 + npu::tile_fwk::LOCAL_GROUP_SIZE - 1) / npu::tile_fwk::LOCAL_GROUP_SIZE;
    } else if (queueType == npu::tile_fwk::DRCO_QUEUE_AIC || queueType == npu::tile_fwk::DRCO_QUEUE_MIX) {
        groupBeg = 0;
        groupEnd = (aicCoreNum + npu::tile_fwk::LOCAL_GROUP_SIZE - 1) / npu::tile_fwk::LOCAL_GROUP_SIZE;
    } else {
        groupBeg = 0;
        groupEnd = npu::tile_fwk::NUM_LOCAL_GROUPS;
    }
    if (groupEnd > npu::tile_fwk::NUM_LOCAL_GROUPS) {
        groupEnd = npu::tile_fwk::NUM_LOCAL_GROUPS;
    }
}

INLINE __gm__ DrcoLocalReadyQueue* DrcoDynFuncDataListGetIdleQueue(__gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                                                   uint32_t succCoreType, uint32_t groupCount,
                                                                   uint32_t startOff)
{
    for (uint32_t i = 0; i < groupCount; i++) {
        uint32_t groupIdx = (startOff + i) % groupCount;
        __gm__ DrcoLocalReadyQueue* queue = DrcoRootFuncListGetLocalReadyQueue(rootFuncList, succCoreType, groupIdx);
        if (queue == nullptr) {
            continue;
        }
        uint32_t head = DrcoAtomicLoad(&queue->head);
        uint32_t tail = DrcoAtomicLoad(&queue->tail);
        if (head >= tail) {
            return queue;
        }
    }
    return nullptr;
}

INLINE uint32_t DrcoDynFuncDataListPushMatrixBatch(DrcoEntryState* state,
                                                   __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                                   uint32_t succCoreType, uint32_t groupCount, uint32_t rowIdx,
                                                   uint32_t* succTaskIdList, uint32_t succTaskIdListSize)
{
    uint32_t pushed = 0;
    uint32_t groupIndex = state->readyMatrixPushGroupIndex;
    uint32_t nextGroupIndex = groupIndex;
    for (uint32_t i = 0; i < groupCount && pushed < succTaskIdListSize; i++) {
        uint32_t groupIdx = (groupIndex + i) % groupCount;
        __gm__ DrcoLocalReadyMatrix* matrix = DrcoRootFuncListGetLocalReadyMatrix(rootFuncList, succCoreType, groupIdx);
        if (matrix == nullptr) {
            continue;
        }
        bool rowExhausted = false;
        pushed += DrcoLocalReadyMatrixPushBatch(state, matrix, groupIdx, rowIdx, &succTaskIdList[pushed],
                                                succTaskIdListSize - pushed, &rowExhausted);
        nextGroupIndex = rowExhausted ? (groupIdx + 1) % groupCount : groupIdx;
    }
    state->readyMatrixPushGroupIndex = nextGroupIndex;
    return pushed;
}

INLINE uint32_t DrcoDynFuncDataListPushBatchLocalMatrix(DrcoEntryState* state,
                                                        __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                                        uint32_t* succTaskIdList, uint32_t succTaskIdListSize,
                                                        uint32_t succCoreType, uint32_t groupCount,
                                                        uint32_t typedBlockIdx)
{
    uint32_t rowIdx = DrcoLocalReadyMatrixGetRowIdx(typedBlockIdx);

    uint32_t pushed = 0;
    for (int i = 0; i < LOCAL_GROUP_SIZE && pushed < succTaskIdListSize; i++) {
        uint32_t oncePushed = DrcoDynFuncDataListPushMatrixBatch(state, rootFuncList, succCoreType, groupCount,
                                                                 (rowIdx + i) % npu::tile_fwk::LOCAL_GROUP_SIZE,
                                                                 succTaskIdList + pushed, succTaskIdListSize - pushed);
        pushed += oncePushed;
    }
    return pushed;
}

INLINE void DrcoDynFuncDataListPushBatchLocalQueue(DrcoEntryState* state,
                                                   __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                                   uint32_t* succTaskIdList, uint32_t succTaskIdListSize,
                                                   uint32_t succCoreType, uint32_t groupCount, uint32_t typedBlockIdx)
{
    // 剩余任务溢出到本类型 local ready queue：
    // 份额递减分配 ceil(size/2)→当前队列、ceil(size/4)→+1 队列、ceil(size/8)→+2 …，
    // 当前队列整批放不下则环形换下一个 group；递减份额取尽后，最后不足一批的剩余全部放进当前队列
    uint32_t groupStart = typedBlockIdx / npu::tile_fwk::LOCAL_GROUP_SIZE;
    uint32_t groupIdx = groupStart;
    uint32_t batchTaskIdx = 0;
    uint32_t rem = succTaskIdListSize;
    uint32_t shift = 1;
    while (rem > 0) {
        uint32_t batch = (succTaskIdListSize + (1u << shift) - 1) >> shift;
        if (batch > rem) {
            batch = rem;
        }
        bool batchPushed = false;
        for (uint32_t g = 0; g < groupCount; g++) {
            uint32_t queueIdx = (groupIdx + g) % groupCount;
            __gm__ DrcoLocalReadyQueue* localQueue = DrcoRootFuncListGetLocalReadyQueue(rootFuncList, succCoreType,
                                                                                        queueIdx);
            // 原子预留 tail..tail+batch-1；容量不足或被其他核抢满则换下一个队列
            while (true) {
                uint32_t tail = DrcoAtomicLoad(&localQueue->tail);
                if (tail + batch > DrcoGmLoad(&localQueue->size)) {
                    break;
                }
                uint32_t tailPrev = DrcoAtomicCasToU32(&localQueue->tail, tail, tail + batch);
                if (tailPrev == tail) {
                    for (uint32_t i = 0; i < batch; i++) {
                        uint32_t succTaskId = succTaskIdList[batchTaskIdx + i];
                        DrcoAtomicExchToU32(&localQueue->taskList[tail + i], DRCO_ENCODE_TASK(succTaskId));
                        TraceEvent(state, succTaskId, EVENT_PUSH_LOCAL(queueIdx));
                    }
                    batchTaskIdx += batch;
                    rem -= batch;
                    groupIdx = (queueIdx + 1) % groupCount;
                    batchPushed = true;
                    break;
                }
            }
            if (batchPushed) {
                break;
            }
        }
        if (!batchPushed) {
            break;
        }
        shift++;
    }
}

INLINE void DrcoDynFuncDataListPushBatch(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                         uint32_t* succTaskIdList, uint32_t succTaskIdListSize, uint32_t succCoreType)
{
    BlockDesc blockDesc = state->blockDesc;
    uint32_t groupCount = state->ctx.drcoGroupEnd[succCoreType] - state->ctx.drcoGroupBeg[succCoreType];
    uint32_t typedBlockIdx = BlockDescTypedBlockIdx(blockDesc);

    uint32_t pushed = DrcoDynFuncDataListPushBatchLocalMatrix(state, rootFuncList, succTaskIdList, succTaskIdListSize,
                                                              succCoreType, groupCount, typedBlockIdx);
    succTaskIdListSize -= pushed;

    if (succTaskIdListSize > 0) {
        DrcoDynFuncDataListPushBatchLocalQueue(state, rootFuncList, succTaskIdList + pushed, succTaskIdListSize,
                                               succCoreType, groupCount, typedBlockIdx);
    }
}

INLINE void DrcoFlushBatchTasks(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                uint32_t succTaskIdListCoreList[][BATCH_PUSH_BUF_SIZE],
                                uint32_t succTaskIdListSizeCoreList[])
{
    for (uint32_t coreType = 0; coreType < npu::tile_fwk::DRCO_QUEUE_MAX; coreType++) {
        if (succTaskIdListSizeCoreList[coreType] > 0) {
            DrcoDynFuncDataListPushBatch(state, rootFuncList, succTaskIdListCoreList[coreType],
                                         succTaskIdListSizeCoreList[coreType], coreType);
            succTaskIdListSizeCoreList[coreType] = 0;
        }
    }
}

INLINE void DrcoResolveDependOnce(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                  uint32_t succTaskId, uint32_t hubStack[], int32_t& hubStackTop,
                                  uint32_t succTaskIdListCoreList[][BATCH_PUSH_BUF_SIZE],
                                  uint32_t succTaskIdListSizeCoreList[])
{
    TraceEvent(state, succTaskId, EVENT_RESOLVE());
    // taskId 解码出的 coreType（AIV/AIC/HUB_MIX）与 DRCO_QUEUE_* 取值一一对应，直接作队列下标
    uint32_t succQueueType = npu::tile_fwk::DrcoTaskCoreTypeOf(succTaskId);
    if (succQueueType < npu::tile_fwk::DRCO_QUEUE_MAX) {
        succTaskIdListCoreList[succQueueType][succTaskIdListSizeCoreList[succQueueType]++] = succTaskId;
        if (succTaskIdListSizeCoreList[succQueueType] >= BATCH_PUSH_BUF_SIZE) {
            DrcoFlushBatchTasks(state, rootFuncList, succTaskIdListCoreList, succTaskIdListSizeCoreList);
        }
    } else {
        RecordMetricHubStatistic(&state->ctx, succTaskId, npu::tile_fwk::FuncID(succTaskId));
        if (hubStackTop + 1 < HUB_STACK_SIZE) {
            hubStack[++hubStackTop] = succTaskId;
        } else {
            // hub 链超深溢出：改投本类型 local matrix/queue，被任意同类型核取走后由
            // IsHubTask 分支就地解依赖（不执行不计数）
            uint32_t overflowTaskId = succTaskId;
            DrcoDynFuncDataListPushBatch(state, rootFuncList, &overflowTaskId, 1, DRCO_CORE_TYPE);
        }
    }
}

// stitch 节点 defer 消费侧专用：stitch 后继不出现 hub，无需 hubStack 分支，
// 按核类型 batch push 即可（类型越界走本类型 local 队列兜底，防数组越界）
INLINE void DrcoResolveDependOnceCore(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                      uint32_t succTaskId, uint32_t succTaskIdListCoreList[][BATCH_PUSH_BUF_SIZE],
                                      uint32_t succTaskIdListSizeCoreList[])
{
    TraceEvent(state, succTaskId, EVENT_RESOLVE());
    // 同 DrcoResolveDependOnce：解码出的 coreType 即队列下标（HUB_MIX 对应 MIX 行）；
    // stitch 后继按约定不出现 hub，万一出现 HUB_MIX 也走 mixhub 派发路径而非被当 leaf 执行
    uint32_t succQueueType = npu::tile_fwk::DrcoTaskCoreTypeOf(succTaskId);
    if (succQueueType < npu::tile_fwk::DRCO_QUEUE_MAX) {
        succTaskIdListCoreList[succQueueType][succTaskIdListSizeCoreList[succQueueType]++] = succTaskId;
        if (succTaskIdListSizeCoreList[succQueueType] >= BATCH_PUSH_BUF_SIZE) {
            DrcoFlushBatchTasks(state, rootFuncList, succTaskIdListCoreList, succTaskIdListSizeCoreList);
        }
    } else {
        // 类型越界兜底：同 hub 溢出路径改投本类型 local matrix/queue
        uint32_t fallbackTaskId = succTaskId;
        DrcoDynFuncDataListPushBatch(state, rootFuncList, &fallbackTaskId, 1, DRCO_CORE_TYPE);
    }
}

// 就地遍历单个 stitch 节点解依赖：DrcoResolveDepend 首节点路径与矩阵 push 失败兜底共用，
// 走完整 DrcoResolveDependOnce 路由（含 hub 处理）
INLINE void DrcoResolveStitchNodeTasks(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                       __gm__ npu::tile_fwk::DevAscendFunctionDuppedStitchNode* node,
                                       uint32_t curTaskId, uint32_t hubStack[], int32_t& hubStackTop,
                                       uint32_t succTaskIdListCoreList[][BATCH_PUSH_BUF_SIZE],
                                       uint32_t succTaskIdListSizeCoreList[])
{
    TraceEvent(state, curTaskId, EVENT_SUCC_DYNAMIC(node->nodeSize));
    for (uint32_t i = 0; i < node->nodeSize; i++) {
        uint32_t succTaskId = node->nodeTaskList[i]; // already coreType-encoded at build time
        uint32_t succFuncId = npu::tile_fwk::FuncID(succTaskId);
        uint32_t succOpIdx = npu::tile_fwk::TaskID(succTaskId);
        auto* succFuncData = &state->ctx.cachedDevTaskCurr->funcDataList[succFuncId];
        __gm__ npu::tile_fwk::DrcoRootFuncData* succRootFuncData = &succFuncData->drcoRootFuncData;
        int32_t old = DrcoAtomicResolveDependOnce<int32_t, 1, -1>(&succRootFuncData->predCount[succOpIdx]);
        if (old == 1) {
            DrcoResolveDependOnce(state, rootFuncList, succTaskId, hubStack, hubStackTop, succTaskIdListCoreList,
                                  succTaskIdListSizeCoreList);
        }
    }
}

// Fire a claimed successor: decode the drco-encoded entry into a taskId and route it
// through the full DrcoResolveDependOnce path (incl. hub handling).
INLINE void DrcoFireEncodedSucc(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                uint32_t funcIdx, uint32_t succEncoded, uint32_t hubStack[], int32_t& hubStackTop,
                                uint32_t succTaskIdListCoreList[][BATCH_PUSH_BUF_SIZE],
                                uint32_t succTaskIdListSizeCoreList[])
{
    uint32_t succOpIdx = succEncoded & TASKID_TASK_MASK;
    uint32_t succCoreType = (succEncoded >> npu::tile_fwk::TASKID_DRCO_CT_SHIFT) & npu::tile_fwk::TASKID_DRCO_CT_MASK;
    uint32_t succTaskId = npu::tile_fwk::MakeDrcoTaskId(funcIdx, succOpIdx, succCoreType);
    DrcoResolveDependOnce(state, rootFuncList, succTaskId, hubStack, hubStackTop, succTaskIdListCoreList,
                          succTaskIdListSizeCoreList);
}

// 尝试把单个 stitch 节点 push 入矩阵：从本核行（全局 blockIdx 对应行）的 start 偏移
// 起逐行 CAS 抢占空槽，成功返回 true 并推进 start（保证同链节点行分散、减少竞争）；
// 失败返回 false，由调用方就地解依赖兜底。
// slot 存节点相对 stitch pool 基址的 u32 偏移（0 = 空闲），基址见 DrcoRootFuncList::stitchNodeBase
INLINE static bool DrcoStitchNodeMatrixTryPush(DrcoEntryState* state,
                                               __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                               __gm__ DrcoGlobalStitchNodeMatrix* matrix,
                                               __gm__ npu::tile_fwk::DevAscendFunctionDuppedStitchNode* node,
                                               uint32_t& start)
{
    BlockDesc blockDesc = state->blockDesc;
    // 行 = 全局 blockIdx（AIC [0,nrValidAic) + AIV [nrValidAic,3*nrValidAic)）：
    // 全核共享池，节点落在任意空行即被该行主人 pop 展开（类型无关）
    uint32_t rowCnt = state->ctx.aicCoreNum * DRCO_ALL_CORES_PER_AIC;
    uint32_t coreTypeIdx = BlockDescBlockIdx(blockDesc);
    uint32_t colIdx = coreTypeIdx % npu::tile_fwk::DrcoGlobalStitchNodeMatrix::COL_SIZE;
    uint64_t stitchNodeBase = rootFuncList->stitchNodeBase;
    for (uint32_t i = 0; i < rowCnt; i++) {
        uint32_t rowIdx = (coreTypeIdx + start + i) % rowCnt;
        __gm__ uint32_t* slot = &matrix->stitchNodeList[rowIdx].slot[colIdx];
        uint32_t prev = DrcoAtomicCasToU32(slot, 0,
                                           static_cast<uint32_t>(reinterpret_cast<uint64_t>(node) - stitchNodeBase));
        if (prev == 0) {
            start = start + i + 1;
            return true;
        }
    }
    return false;
}

// 就地解单个 stitch 节点依赖：走 DrcoResolveDependOnceCore 路由（无 hub 级联，batch push），
// 任务数从 nodeSize 字段直读；消费核 pop 与生产核 push 失败兜底共用
INLINE void DrcoStitchNodeTasksResolveCore(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                           __gm__ npu::tile_fwk::DevAscendFunctionDuppedStitchNode* node,
                                           uint32_t succTaskIdListCoreList[][BATCH_PUSH_BUF_SIZE],
                                           uint32_t succTaskIdListSizeCoreList[])
{
    for (uint32_t i = 0; i < node->nodeSize; i++) {
        uint32_t succTaskId = node->nodeTaskList[i]; // already coreType-encoded at build time
        uint32_t succFuncId = npu::tile_fwk::FuncID(succTaskId);
        uint32_t succOpIdx = npu::tile_fwk::TaskID(succTaskId);
        auto* succFuncData = &state->ctx.cachedDevTaskCurr->funcDataList[succFuncId];
        __gm__ npu::tile_fwk::DrcoRootFuncData* succRootFuncData = &succFuncData->drcoRootFuncData;
        int32_t old = DrcoAtomicResolveDependOnce<int32_t, 1, -1>(&succRootFuncData->predCount[succOpIdx]);
        if (old == 1) {
            DrcoResolveDependOnceCore(state, rootFuncList, succTaskId, succTaskIdListCoreList,
                                      succTaskIdListSizeCoreList);
        }
    }
}

INLINE void DrcoResolveDepend(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                              uint32_t* taskIdList, uint32_t taskCount = 1)
{
    uint32_t succTaskIdListCoreList[npu::tile_fwk::DRCO_QUEUE_MAX][BATCH_PUSH_BUF_SIZE];
    uint32_t succTaskIdListSizeCoreList[npu::tile_fwk::DRCO_QUEUE_MAX] = {0};

    uint32_t hubStack[HUB_STACK_SIZE];
    int32_t hubStackTop = -1;
    for (uint32_t i = 0; i < taskCount; i++) {
        hubStack[++hubStackTop] = taskIdList[i];
    }
    while (hubStackTop >= 0) {
        uint32_t curTaskId = hubStack[hubStackTop--];
        uint32_t funcIdx = npu::tile_fwk::FuncID(curTaskId);
        uint32_t operIdx = npu::tile_fwk::TaskID(curTaskId);

        auto funcData = &state->ctx.cachedDevTaskCurr->funcDataList[funcIdx];
        __gm__ npu::tile_fwk::DrcoRootFuncData* rootFuncData = &funcData->drcoRootFuncData;
        __gm__ npu::tile_fwk::DevAscendFunctionOperationSuccInfo* succInfoList = rootFuncData->succInfoList;
        __gm__ int32_t* succStaticList = rootFuncData->succStaticList;
        __gm__ int32_t* predCount = rootFuncData->predCount;
        volatile __gm__ npu::tile_fwk::DevAscendFunctionOperationSuccInfo* succInfo = &succInfoList[operIdx];

        uint16_t staticIndex = succInfo->staticIndex;
        uint16_t staticSize = succInfo->staticSize;
        TraceEvent(state, curTaskId, EVENT_SUCC_STATIC(staticSize));
        uint16_t i = staticIndex;
        const uint16_t staticEnd = staticIndex + staticSize;
        while (i < staticEnd) {
            uint32_t succEncoded = succStaticList[i];
            uint32_t succOpIdx = succEncoded & TASKID_TASK_MASK;
            if ((succEncoded & npu::tile_fwk::DRCO_SUCC_PAIR_BIT) != 0) {
                // Pair entry: even-aligned (2k, 2k+1) successors share one u64 predCount slot,
                // decremented together by one atomicAdd. Exactly-once edge resolution keeps each half
                // >= 1 until its final decrement, so the packed subtract never borrows across halves;
                // fire each half whose old count was 1 (the unique zero-crossing resolver).
                uint32_t nextEncoded = succStaticList[i + 1];
                __gm__ uint64_t* slot = reinterpret_cast<__gm__ uint64_t*>(&predCount[succOpIdx]);
                uint64_t old = DrcoAtomicResolveDependOnce<uint64_t, npu::tile_fwk::DRCO_SUCC_PAIR_BOTH_ONE,
                                                           npu::tile_fwk::DRCO_SUCC_PAIR_DEC>(slot);
                if (static_cast<uint32_t>(old) == 1u) {
                    DrcoFireEncodedSucc(state, rootFuncList, funcIdx, succEncoded, hubStack, hubStackTop,
                                        succTaskIdListCoreList, succTaskIdListSizeCoreList);
                }
                if (static_cast<uint32_t>(old >> 32) == 1u) {
                    DrcoFireEncodedSucc(state, rootFuncList, funcIdx, nextEncoded, hubStack, hubStackTop,
                                        succTaskIdListCoreList, succTaskIdListSizeCoreList);
                }
                i += 2;
            } else {
                int32_t old = DrcoAtomicResolveDependOnce<int32_t, 1, -1>(&predCount[succOpIdx]);
                if (old == 1) {
                    DrcoFireEncodedSucc(state, rootFuncList, funcIdx, succEncoded, hubStack, hubStackTop,
                                        succTaskIdListCoreList, succTaskIdListSizeCoreList);
                }
                i += 1;
            }
        }

        uint32_t stitchIndex = succInfo->stitchIndex;
        if (stitchIndex != 0) {
            __gm__ npu::tile_fwk::DevAscendFunctionDuppedStitchNode** succStitchList = rootFuncData->succStitchList;
            // 先保存首节点，其余节点一次性 push 到本类型 localStitchNodeMatrix 自己的行，由消费核
            // fetch 时 pop 解依赖（摊平长链遍历开销）；push 结束后再统一就地解依赖：首节点 +
            // 首个 push 失败起的剩余链尾（矩阵写不进则整段链尾就地兜底，保证前向推进）
            __gm__ npu::tile_fwk::DevAscendFunctionDuppedStitchNode* firstNode = succStitchList[stitchIndex];
            __gm__ npu::tile_fwk::DevAscendFunctionDuppedStitchNode* fallbackNode = nullptr;
            __gm__ DrcoGlobalStitchNodeMatrix* matrix = DrcoRootFuncListGetStitchNodeMatrix(rootFuncList);

            if (firstNode != nullptr) {
                uint32_t start = 0;

                for (__gm__ npu::tile_fwk::DevAscendFunctionDuppedStitchNode* node = firstNode->nodeNext;
                     node != nullptr && fallbackNode == nullptr; node = node->nodeNext) {
                    if (DrcoStitchNodeMatrixTryPush(state, rootFuncList, matrix, node, start)) {
                        TraceEvent(state, curTaskId, EVENT_STITCH_NODE(node->nodeSize));
                    } else {
                        fallbackNode = node;
                    }
                }

                DrcoResolveStitchNodeTasks(state, rootFuncList, firstNode, curTaskId, hubStack, hubStackTop,
                                           succTaskIdListCoreList, succTaskIdListSizeCoreList);
            }
            for (__gm__ npu::tile_fwk::DevAscendFunctionDuppedStitchNode* node = fallbackNode; node != nullptr;
                 node = node->nodeNext) {
                DrcoResolveStitchNodeTasks(state, rootFuncList, node, curTaskId, hubStack, hubStackTop,
                                           succTaskIdListCoreList, succTaskIdListSizeCoreList);
            }
        }
    }
    DrcoFlushBatchTasks(state, rootFuncList, succTaskIdListCoreList, succTaskIdListSizeCoreList);
}

__aicore__ INLINE static bool MixTaskPush(DrcoEntryState* state, uint32_t succTaskId,
                                          npu::tile_fwk::MixHubC2VReadyQueue* buf)
{
#if defined(__AIV__)
    (void)state;
    return false;
#else
    if (npu::tile_fwk::MixHubC2VReadyQueue::Push(buf, succTaskId)) {
        DRCO_LOG(&state->ctx, "MIX C2V push task=%d", (int)succTaskId);
        return true;
    }
    return false;
#endif
}

__aicore__ INLINE static uint32_t ResolveHubMixTask(DrcoEntryState* state, uint32_t hubMixTaskId,
                                                    npu::tile_fwk::MixHubC2VReadyQueue* body)
{
    uint32_t funcIdx = npu::tile_fwk::FuncID(hubMixTaskId);
    uint32_t opIdx = npu::tile_fwk::TaskID(hubMixTaskId);
    auto funcData = &state->ctx.cachedDevTasks[state->ctx.curLeafTaskParallelIdx].funcDataList[funcIdx];
    __gm__ npu::tile_fwk::DevAscendFunctionOperationSuccInfo* mixSuccInfoList = funcData->drcoRootFuncData.succInfoList;
    __gm__ int32_t* mixSuccStaticList = funcData->drcoRootFuncData.succStaticList;
    __gm__ int* mixCceBinaryIndexList = funcData->cceBinaryIndexList;
    uint32_t aicTaskId = static_cast<uint32_t>(AICORE_TASK_INIT);

    uint16_t mixStaticIndex = mixSuccInfoList[opIdx].staticIndex;
    uint16_t mixStaticSize = mixSuccInfoList[opIdx].staticSize;
    for (uint16_t j = mixStaticIndex; j < mixStaticIndex + mixStaticSize; j++) {
        uint32_t mixSuccEncoded = mixSuccStaticList[j];
        uint32_t mixSuccOpIdx = mixSuccEncoded & TASKID_TASK_MASK;
        uint32_t mixSuccCoreType = (mixSuccEncoded >> npu::tile_fwk::TASKID_DRCO_CT_SHIFT) &
                                   npu::tile_fwk::TASKID_DRCO_CT_MASK;
        uint32_t mixSuccTaskId = npu::tile_fwk::MakeDrcoTaskId(funcIdx, mixSuccOpIdx, mixSuccCoreType);
        DRCO_LOG(&state->ctx, "MIX resolve taskId=%u", mixSuccTaskId);
        if (mixSuccCoreType == static_cast<uint32_t>(npu::tile_fwk::CoreType::AIC)) {
            aicTaskId = mixSuccTaskId;
        } else {
            // 正常路径下环不会满（每次派发每环至多压 1 个、配对 AIV 在 fetch 循环持续 Pop）；
            // 万一暂满则自旋等待：环内有未执行任务时 AIV finish flag 必未置位、配对 AIV 不会退出
            // fetch 循环，等待必有进展；超时 Trap 兜底，杜绝静默丢任务导致的挂死
            npu::tile_fwk::MixHubC2VReadyQueue* dstAddr = body +
                                                          state->ctx.cachedDevTasks[state->ctx.curLeafTaskParallelIdx]
                                                              .cceBinary[mixCceBinaryIndexList[mixSuccOpIdx]]
                                                              .wrapVecId;
            uint64_t pushStart = get_sys_cnt();
            while (!MixTaskPush(state, mixSuccTaskId, dstAddr)) {
                if (get_sys_cnt() - pushStart > AICORE_LEAF_TASK_RUN_TIMEOUT) {
                    Trap();
                }
                DrcoBusyBackOff();
            }
        }
    }
    return aicTaskId;
}

INLINE __gm__ DrcoLocalReadyQueue* TryGetOtherLocalQue(__gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                                       uint32_t coreType, uint32_t groupIdx)
{
    for (uint32_t g = 1; g < npu::tile_fwk::NUM_LOCAL_GROUPS; g++) {
        uint32_t victim = (groupIdx + g) % npu::tile_fwk::NUM_LOCAL_GROUPS;
        __gm__ DrcoLocalReadyQueue* que = DrcoRootFuncListGetLocalReadyQueue(rootFuncList, coreType, victim);
        if (que == nullptr) {
            continue;
        }
        uint32_t head = DrcoAtomicLoad(&que->head);
        uint32_t tail = DrcoAtomicLoad(&que->tail);
        if (head < tail) {
            return que;
        }
    }
    return nullptr;
}

// LocalMatrix 版 TryGetOtherLocalQue：找其他 group 中自己列还有任务的矩阵（窃取）
INLINE __gm__ DrcoLocalReadyMatrix* TryGetOtherLocalMatrix(__gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                                           uint32_t coreType, uint32_t groupIdx, uint32_t colIdx)
{
    for (uint32_t g = 1; g < npu::tile_fwk::NUM_LOCAL_GROUPS; g++) {
        uint32_t victim = (groupIdx + g) % npu::tile_fwk::NUM_LOCAL_GROUPS;
        __gm__ DrcoLocalReadyMatrix* matrix = DrcoRootFuncListGetLocalReadyMatrix(rootFuncList, coreType, victim);
        if (matrix == nullptr) {
            continue;
        }
        for (uint32_t row = 0; row < npu::tile_fwk::LOCAL_GROUP_SIZE; row++) {
            if (DrcoAtomicLoad(&matrix->taskList[row][colIdx]) != 0) {
                return matrix;
            }
        }
    }
    return nullptr;
}

// 消费本核行上的 defer stitch 节点：与 PopColTasks 相同的两阶段 CAS 抢占
// （CAS(0,0) 原子读 → CAS(offset,0) 独占），节点内任务就地解依赖后 batch push；
// 节点内容 host 侧构建后不可变，slot 存相对 stitch pool 基址的 u32 偏移，CAS 发布偏移即可，
// 经 DrcoRootFuncList::stitchNodeBase 还原节点地址；只 pop 自己类型内编号对应的行
INLINE void DrcoStitchNodeMatrixPopResolve(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                           __gm__ DrcoGlobalStitchNodeMatrix* matrix, uint32_t rowIdx)
{
    uint32_t succTaskIdListCoreList[npu::tile_fwk::DRCO_QUEUE_MAX][BATCH_PUSH_BUF_SIZE];
    uint32_t succTaskIdListSizeCoreList[npu::tile_fwk::DRCO_QUEUE_MAX] = {0};
    for (uint32_t col = 0; col < DrcoGlobalStitchNodeMatrix::COL_SIZE; col++) {
        __gm__ uint32_t* slot = &matrix->stitchNodeList[rowIdx].slot[col];
        uint32_t nodeOffset = DrcoAtomicCasToU32(slot, 0, 0);
        if (nodeOffset == 0) {
            continue;
        }
        if (DrcoAtomicCasToU32(slot, nodeOffset, 0) != nodeOffset) {
            continue;
        }
        __gm__ npu::tile_fwk::DevAscendFunctionDuppedStitchNode*
            node = (__gm__ npu::tile_fwk::DevAscendFunctionDuppedStitchNode*)(rootFuncList->stitchNodeBase +
                                                                              nodeOffset);
        DrcoStitchNodeTasksResolveCore(state, rootFuncList, node, succTaskIdListCoreList, succTaskIdListSizeCoreList);
    }
    DrcoFlushBatchTasks(state, rootFuncList, succTaskIdListCoreList, succTaskIdListSizeCoreList);
}

INLINE void DrcoDynFuncDataListFetchResolveStitchNodeMatrix(DrcoEntryState* state,
                                                            __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                                            uint32_t blockIdx)
{
    __gm__ DrcoGlobalStitchNodeMatrix* stitchNodeMatrix = DrcoRootFuncListGetStitchNodeMatrix(rootFuncList);
    DrcoStitchNodeMatrixPopResolve(state, rootFuncList, stitchNodeMatrix, blockIdx);
}

INLINE bool DrcoDynFuncDataListFetchTaskMixHubC2VReadyQueue(DrcoEntryState* state,
                                                            __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                                            uint32_t& resultCoreType,
                                                            uint32_t resultTaskIdList[LOCAL_GROUP_SIZE],
                                                            uint32_t& resultTaskIdCount, uint32_t blockIdx)
{
    if (IS_AIV) {
        __gm__ npu::tile_fwk::PerCorePendingQueue* perCoreQueue = rootFuncList->perCorePendingQueueArray[blockIdx];
        uint32_t bodyTaskId = 0;
        if (npu::tile_fwk::MixHubC2VReadyQueue::Pop(perCoreQueue->mixHubC2VReadyQueue, bodyTaskId)) {
            resultTaskIdList[0] = bodyTaskId;
            resultTaskIdCount = 1;
            resultCoreType = npu::tile_fwk::DRCO_QUEUE_AIV;
            return true;
        }
    }
    return false;
}

INLINE bool DrcoDynFuncDataListFetchTaskLocalReadyMatrix(DrcoEntryState* state,
                                                         __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                                         uint32_t& resultCoreType,
                                                         uint32_t resultTaskIdList[LOCAL_GROUP_SIZE],
                                                         uint32_t& resultTaskIdCount,
                                                         __gm__ DrcoLocalReadyMatrix* localReadyMatrix, uint32_t colIdx)
{
    uint32_t count = DrcoLocalReadyMatrixPopColTasks(state, localReadyMatrix, colIdx, resultTaskIdList,
                                                     npu::tile_fwk::LOCAL_GROUP_SIZE);
    if (count > 0) {
        resultTaskIdCount = count;
        resultCoreType = DRCO_CORE_TYPE;
        return true;
    }
    return false;
}

// queueCoreType 参数化：本类型行（AIC/AIV）与 MIX 行共用同一取数逻辑，命中时 resultCoreType = queueCoreType；
// 仅取本组队列，不跨组窃取（push 始于推送核本组、活跃区间内每组均有本组消费者，本组队列必被消费）
INLINE bool DrcoDynFuncDataListFetchTaskLocalReadyQueue([[maybe_unused]] DrcoEntryState* state,
                                                        __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                                        uint32_t queueCoreType, uint32_t& resultCoreType,
                                                        uint32_t resultTaskIdList[LOCAL_GROUP_SIZE],
                                                        uint32_t& resultTaskIdCount, uint32_t groupIdx)
{
    __gm__ DrcoLocalReadyQueue* localReadyQueue = DrcoRootFuncListGetLocalReadyQueue(rootFuncList, queueCoreType,
                                                                                     groupIdx);
    DRCO_DCCI_SINGLE_CACHE_LINE(localReadyQueue);
    uint32_t taskId = static_cast<uint32_t>(AICORE_TASK_FETCH_CONFLICT);
    while (taskId == static_cast<uint32_t>(AICORE_TASK_FETCH_CONFLICT)) {
        taskId = DrcoLocalReadyQueueGetFirstTask(localReadyQueue);
    }
    if (taskId != static_cast<uint32_t>(AICORE_TASK_NO_INCOME)) {
        resultTaskIdList[0] = taskId;
        resultTaskIdCount = 1;
        resultCoreType = queueCoreType;
        return true;
    }

    return false;
}

// mixhub 与 AIC/AIV 同模型消费：从 MIX 行取任务（组区间/列号均按 AIC 核的类型内编号，消费者
// 为 AIC），先 pop 本组 matrix 自己的列，再取本组 MIX queue；取出后不执行
// leaf，仅由 ExecDrcoReadyQueueTaskOnce 经 ResolveHubMixTask 解依赖派发（AIC 后继就地执行、
// AIV 后继 C2V 投递配对 AIV）
INLINE bool DrcoDynFuncDataListFetchTaskMixHub(DrcoEntryState* state,
                                               __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                               uint32_t& resultCoreType, uint32_t resultTaskIdList[LOCAL_GROUP_SIZE],
                                               uint32_t& resultTaskIdCount, uint32_t groupIdx, uint32_t colIdx)
{
    if (IS_AIC) {
        __gm__ DrcoLocalReadyMatrix* mixMatrix = DrcoRootFuncListGetLocalReadyMatrix(
            rootFuncList, npu::tile_fwk::DRCO_QUEUE_MIX, groupIdx);
        if (mixMatrix != nullptr) {
            uint32_t count = DrcoLocalReadyMatrixPopColTasks(state, mixMatrix, colIdx, resultTaskIdList,
                                                             npu::tile_fwk::LOCAL_GROUP_SIZE);
            if (count > 0) {
                resultTaskIdCount = count;
                resultCoreType = npu::tile_fwk::DRCO_QUEUE_MIX;
                return true;
            }
        }
        return DrcoDynFuncDataListFetchTaskLocalReadyQueue(state, rootFuncList, npu::tile_fwk::DRCO_QUEUE_MIX,
                                                           resultCoreType, resultTaskIdList, resultTaskIdCount,
                                                           groupIdx);
    }
    return false;
}

INLINE uint32_t DrcoDynFuncDataListFetchTask(DrcoEntryState* state,
                                             __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                             uint32_t& resultCoreType, uint32_t resultTaskIdList[LOCAL_GROUP_SIZE])
{
    TraceEvent(state, 0xffffffff, EVENT_FETCH_START());
    BlockDesc blockDesc = state->blockDesc;
    uint32_t blockIdx = BlockDescBlockIdx(blockDesc);
    uint32_t typedBlockIdx = BlockDescTypedBlockIdx(blockDesc);
    uint32_t groupIdx = typedBlockIdx / npu::tile_fwk::LOCAL_GROUP_SIZE;

    __gm__ DrcoLocalReadyMatrix* localReadyMatrix = DrcoRootFuncListGetLocalReadyMatrix(rootFuncList, DRCO_CORE_TYPE,
                                                                                        groupIdx);
    uint32_t colIdx = DrcoLocalReadyMatrixGetRowIdx(typedBlockIdx);
    // finishFlag 每类型共享一个：本类型全部核经 DrcoGmLoad（含 dcci 失效读）轮询同一 flag
    __gm__ uint32_t* devTaskFinishFlag = &rootFuncList->devTaskFinishFlagList.flag[DRCO_CORE_TYPE].devTaskFinishFlag;
    // 对方类型完成 flag。
    // 全局退出 = 两类型 flag 均置位；本类型先完成时进入帮忙模式（只扫共享 stitch 矩阵，
    // 跳过本类型 leaf 矩阵/队列 pop——必然空，省 CAS 竞争），避免对侧链节点落在已退出核的行无人 pop
    constexpr uint32_t DRCO_OTHER_QUEUE_TYPE = (DRCO_CORE_TYPE == static_cast<uint32_t>(npu::tile_fwk::CoreType::AIC)) ?
                                                   npu::tile_fwk::DRCO_QUEUE_AIV :
                                                   npu::tile_fwk::DRCO_QUEUE_AIC;
    __gm__ uint32_t* otherTaskFinishFlag = &rootFuncList->devTaskFinishFlagList.flag[DRCO_OTHER_QUEUE_TYPE]
                                                .devTaskFinishFlag;
    uint32_t resultTaskIdCount = 0;

    uint64_t t0 = get_sys_cnt();
    while (true) {
        // 全局退出 = 本类型 + 对方类型 flag 均置位（leaf 计数含 mix 任务，两类型齐全即全部完成）；
        // 单侧完成不退出——继续循环扫共享 stitch 矩阵帮忙对侧，防对侧链节点落在己方行无人 pop
        if (DrcoGmLoad(devTaskFinishFlag) != 0 && DrcoGmLoad(otherTaskFinishFlag) != 0) {
            resultTaskIdCount = static_cast<uint32_t>(AICORE_TASK_ALL_FINISH);
            break;
        }

        if (get_sys_cnt() - t0 > AICORE_LEAF_TASK_RUN_TIMEOUT) {
            Trap();
        }

        DrcoDynFuncDataListFetchResolveStitchNodeMatrix(state, rootFuncList, blockIdx);

        // 本类型 leaf 全部认领/执行完（flag 置位）后跳过本类型 leaf 矩阵/队列/mixhub 消费
        // （必然空，省 CAS 竞争——mixhub 任务的写入方即本类型核，全部完成后不再有新任务），
        // 仅保留上方共享 stitch 矩阵扫描帮忙对侧展开
        if (DrcoGmLoad(devTaskFinishFlag) == 0) {
            if (DrcoDynFuncDataListFetchTaskMixHubC2VReadyQueue(state, rootFuncList, resultCoreType, resultTaskIdList,
                                                                resultTaskIdCount, blockIdx)) {
                break;
            }
            if (DrcoDynFuncDataListFetchTaskMixHub(state, rootFuncList, resultCoreType, resultTaskIdList,
                                                   resultTaskIdCount, groupIdx, colIdx)) {
                break;
            }
            if (DrcoDynFuncDataListFetchTaskLocalReadyMatrix(state, rootFuncList, resultCoreType, resultTaskIdList,
                                                             resultTaskIdCount, localReadyMatrix, colIdx)) {
                break;
            }
            if (DrcoDynFuncDataListFetchTaskLocalReadyQueue(state, rootFuncList, DRCO_CORE_TYPE, resultCoreType,
                                                            resultTaskIdList, resultTaskIdCount, groupIdx)) {
                break;
            }
        } // 本类型未完成（帮忙模式跳过 leaf 消费）
    }
    TraceEvent(state, 0xffffffff, EVENT_FETCH_END());
    return resultTaskIdCount;
}

INLINE void InitDrcoEntry(DrcoEntryState* state, int64_t cfgdata)
{
    auto devArgs = (DeviceArgs*)cfgdata;
    uint64_t start = get_sys_cnt();
#if defined(__AIV__) && defined(__MIX__)
    uint32_t blockIdx = get_block_idx() * get_subblockdim() + get_subblockid() + get_block_num();
#else
    uint32_t blockIdx = get_block_idx();
#endif
    uint32_t typedBlockIdx = DrcoGetCoreTypedIdx(blockIdx, devArgs->nrValidAic);
    state->blockDesc = BlockDescCreate(blockIdx, typedBlockIdx, devArgs->nrValidAic, devArgs->nrValidAic * 2);
    state->args = (__gm__ KernelArgs*)(devArgs->sharedBuffer +
                                       BlockDescBlockIdx(state->blockDesc) * SHARED_BUFFER_SIZE);
    __gm__ Metrics* metric = (__gm__ Metrics*)(state->args->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);
    state->metric = metric;
    state->ctx.args = state->args;
    state->ctx.blockIdx = BlockDescBlockIdx(state->blockDesc);
    __gm__ DevDfxArgs* devDfxAddr = (__gm__ DevDfxArgs*)devArgs->devDfxArgAddr;
    state->ctx.aicoreDevTaskMetric.devTaskMetricEnable = devDfxAddr->isOpenPerfTrace != 0;
    state->ctx.profLevel = devDfxAddr->profLevel;
    state->ctx.aicCoreNum = devArgs->nrValidAic;
    // 各 coreType 活跃组区间是本 kernel 生命周期内的不变量，入口一次算好供全部热路径查表
    for (uint32_t ct = 0; ct < npu::tile_fwk::DRCO_QUEUE_MAX; ct++) {
        GetDrcoActiveGroupRange(ct, state->ctx.aicCoreNum, state->ctx.drcoGroupBeg[ct], state->ctx.drcoGroupEnd[ct]);
    }

    uint8_t aicoreLogLevel = static_cast<uint8_t>(AicoreLogLevel::NONE);
#if ENABLE_AICORE_PRINT
    if (devDfxAddr->logLevel >= 0) {
        aicoreLogLevel = static_cast<uint8_t>(devDfxAddr->logLevel);
    }
#endif
    if (state->ctx.aicoreDevTaskMetric.devTaskMetricEnable && metric->turnNum < MAX_ROUND_NUM) {
        uint64_t round = metric->turnNum;
        state->ctx.aicoreDevTaskMetric.devTaskMetric = &(metric->aicoreDevTaskInfo[round]);
        PerfTraceRecord(INVALID_DEV_TASK_ID, state->ctx.aicoreDevTaskMetric.devTaskMetric, PERF_TRACE_CORE_BEGIN,
                        start);
    }

    set_mask_norm();
    state->lastMixResourceType = static_cast<uint8_t>(MixResourceType::MIX_UNKNOWN);

    PerfTraceRecord(INVALID_DEV_TASK_ID, state->ctx.aicoreDevTaskMetric.devTaskMetric, PERF_TRACE_CORE_INIT);

    InitCtx(&state->ctx, metric, nullptr, aicoreLogLevel);

    state->runtimeDataRingBufferHeadData = (__gm__ npu::tile_fwk::RuntimeDataRingBufferHeadData*)
                                               devArgs->runtimeDataRingBufferAddr;
    DRCO_DCCI_SINGLE_CACHE_LINE(state->runtimeDataRingBufferHeadData);
    state->base = npu::tile_fwk::RuntimeDataRingBufferHeadData::GetRuntimeDataCurrent(
        state->runtimeDataRingBufferHeadData);
    DRCO_DCCI_SINGLE_CACHE_LINE(state->base);
    state->deviceTaskReadyQueue = DrcoGmLoad(&state->base->drcoDeviceTaskReadyQueue);
}

INLINE __gm__ DrcoDeviceTask* GetDrcoDeviceTask(DrcoEntryState* state, bool& isFirstTask, uint64_t t0,
                                                uint64_t& loopCount, bool& warningSet)
{
    if (BlockDescBlockIdx(state->blockDesc) == 0) {
        if (!isFirstTask) {
            uint32_t oldHead = DrcoAtomicLoad(&state->deviceTaskReadyQueue->head);
            DrcoAtomicCasToU32(&state->deviceTaskReadyQueue->head, oldHead, oldHead + 1);
        }
        isFirstTask = false;
        __gm__ DrcoDeviceTask* deviceTask = nullptr;
        while (true) {
            DRCO_LEADER_TIMEOUT_CHECK(t0, loopCount, AICORE_LEAF_TASK_RUN_TIMEOUT, STAGE_RUN_LEAFTASK_TIMEOUT);
            if (state->deviceTaskReadyQueue == nullptr) {
                DRCO_DCCI_SINGLE_CACHE_LINE(state->runtimeDataRingBufferHeadData);
                state->base = npu::tile_fwk::RuntimeDataRingBufferHeadData::GetRuntimeDataCurrent(
                    state->runtimeDataRingBufferHeadData);
                DRCO_DCCI_SINGLE_CACHE_LINE(state->base);
                state->deviceTaskReadyQueue = DrcoGmLoad(&state->base->drcoDeviceTaskReadyQueue);
                if (state->deviceTaskReadyQueue == nullptr) {
                    DrcoBusyBackOff();
                    continue;
                }
            }
            deviceTask = GetCurrentDeviceTask(state->deviceTaskReadyQueue);
            if (deviceTask == nullptr) {
                uint32_t qHead = DrcoAtomicLoad(&state->deviceTaskReadyQueue->head);
                uint32_t qTail = DrcoGmLoad(&state->deviceTaskReadyQueue->tail);
                if (qHead < qTail) {
                    __gm__ DrcoDeviceTask*
                        elem = &state->deviceTaskReadyQueue
                                    ->dynFuncDataListList[qHead % npu::tile_fwk::DEVICE_TASK_QUEUE_SIZE];
                    if (DrcoGmLoad(&elem->dynFuncDataList) == nullptr) {
                        break;
                    }
                }
                DrcoBusyBackOff();
                continue;
            }
            break;
        }
        SyncAllMix();
        return deviceTask;
    }

    SyncAllMix();
    if (state->deviceTaskReadyQueue == nullptr) {
        DRCO_DCCI_SINGLE_CACHE_LINE(state->runtimeDataRingBufferHeadData);
        state->base = npu::tile_fwk::RuntimeDataRingBufferHeadData::GetRuntimeDataCurrent(
            state->runtimeDataRingBufferHeadData);
        DRCO_DCCI_SINGLE_CACHE_LINE(state->base);
        state->deviceTaskReadyQueue = DrcoGmLoad(&state->base->drcoDeviceTaskReadyQueue);
    }
    __gm__ DrcoDeviceTask* deviceTask = GetCurrentDeviceTask(state->deviceTaskReadyQueue);
    return deviceTask;
}

INLINE void ExecLeafFunction(DrcoEntryState* state, uint32_t taskId)
{
    TraceEvent(state, taskId, EVENT_LEAF_START());

    ExecCoreFunctionKernel(&state->ctx, taskId, state->lastMixResourceType);

    TraceEvent(state, taskId, EVENT_LEAF_END());
}

INLINE void PerfDevTaskFirstLeafTask(DrcoEntryState* state)
{
    if (!state->isExectedLeafTask) {
        PerfTraceRecord(state->ctx.SeqNo(), state->ctx.aicoreDevTaskMetric.devTaskMetric,
                        PERF_TRACE_CORE_DEV_TASK_WAIT_RCV_FIRST_LEAF_TASK);
        state->isExectedLeafTask = true;
    }
}

INLINE void ExecDrcoPerCoreTasks(DrcoEntryState* state, __gm__ npu::tile_fwk::PerCorePendingQueue* perCoreQueue,
                                 __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList)
{
    uint32_t perCoreQueueSize = DrcoGmLoad(&perCoreQueue->size);
    uint32_t perCoreQueueHead = DrcoGmLoad(&perCoreQueue->head);
    DrcoNotifyTaskExecutedAdd(state, rootFuncList, perCoreQueueSize);
    while (perCoreQueueSize > perCoreQueueHead) {
        uint32_t taskId = DrcoGmLoadArray(perCoreQueue->taskList, perCoreQueueHead);
        PerfDevTaskFirstLeafTask(state);
        ExecLeafFunction(state, taskId);
        DrcoResolveDepend(state, rootFuncList, &taskId);
        perCoreQueueHead++;
    }
}

// 兜底专用判断：hub 栈溢出/stitch 类型越界改投本类型 local 队列的 HUB 任务，被取出后就地
// 解依赖（不执行不计数）；HUB_MIX 不会经兜底入队——其恒被路由到 MIX 行，由
// ExecDrcoReadyQueueTaskOnce 的 DRCO_QUEUE_MIX 分支（ResolveHubMixTask）消费，不经过此处
INLINE bool IsHubTask(DrcoEntryState* state, uint32_t taskId)
{
    (void)state;
    return npu::tile_fwk::DrcoTaskCoreTypeOf(taskId) == static_cast<uint32_t>(npu::tile_fwk::CoreType::HUB);
}

// 返回本任务是否执行了 leaf（FIN 标记 / hub 任务只解依赖不计数）。
// executedCount 前移到执行前（notify before run）：非 hub_mix/hub 的可执行 leaf 在执行前
// 逐个计数——计数到 size 即本类型所有 leaf 已被认领，末批执行期间其它核可并行退出
// fetch 自旋，消除整批执行完才广播的尾部空隙
INLINE bool ExecDrcoReadyQueueTaskOnce(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                       [[maybe_unused]] __gm__ npu::tile_fwk::PerCorePendingQueue* perCoreQueue,
                                       uint32_t taskId, uint32_t outCoreType, uint32_t& executedTaskId)
{
#if defined(__MIX__) && defined(__AIC__)
    if (outCoreType == npu::tile_fwk::DRCO_QUEUE_MIX) {
        // mixhub 任务（来自 MIX 行 matrix/queue）不执行 leaf，仅解依赖派发
        uint32_t aicTaskId = ResolveHubMixTask(state, taskId, perCoreQueue->mixHubC2VReadyQueue);
        if (aicTaskId == static_cast<uint32_t>(AICORE_TASK_INIT)) {
            // 无 AIC 后继：V 后继已在 ResolveHubMixTask 内经 C2V 派发完毕，hub_mix 节点
            // 自身非可执行 leaf（不计入 executedCount），无事可做
            return false;
        }
        DRCO_LOG(&state->ctx, "MIX exec=%u", aicTaskId);
        taskId = aicTaskId;
    }
#endif
    if ((taskId & AICORE_FIN_MASK) != 0) {
        return false;
    }
    if (IsHubTask(state, taskId)) {
        DrcoResolveDepend(state, rootFuncList, &taskId);
        return false;
    }
    DrcoNotifyTaskExecutedAdd(state, rootFuncList, 1);
    ExecLeafFunction(state, taskId);
    executedTaskId = taskId;
    return true;
}

INLINE void ExecDrcoReadyQueueTasks(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                    __gm__ npu::tile_fwk::PerCorePendingQueue* perCoreQueue)
{
    uint32_t taskIdList[LOCAL_GROUP_SIZE];

    uint32_t outCoreType = 0;
    uint32_t taskCount = DrcoDynFuncDataListFetchTask(state, rootFuncList, outCoreType, taskIdList);
    while (taskCount != static_cast<uint32_t>(AICORE_TASK_ALL_FINISH)) {
        PerfDevTaskFirstLeafTask(state);
        uint32_t executedTaskIdList[LOCAL_GROUP_SIZE];
        uint32_t resolveCount = 0;
        for (uint32_t i = 0; i < taskCount; i++) {
            if (ExecDrcoReadyQueueTaskOnce(state, rootFuncList, perCoreQueue, taskIdList[i], outCoreType,
                                           executedTaskIdList[resolveCount])) {
                resolveCount++;
            }
        }
        if (resolveCount > 0) {
            DrcoResolveDepend(state, rootFuncList, executedTaskIdList, resolveCount);
        }
        taskCount = DrcoDynFuncDataListFetchTask(state, rootFuncList, outCoreType, taskIdList);
    }
}

INLINE void KernelEntryDrco(int64_t ffts_addr, int64_t inputs, int64_t outputs, int64_t workspace, int64_t tilingdata,
                            int64_t cfgdata)
{
    UNUSED(ffts_addr);
    UNUSED(inputs);
    UNUSED(outputs);
    UNUSED(workspace);
    UNUSED(tilingdata);

    DrcoEntryState state = {};
    InitDrcoEntry(&state, cfgdata);
    TraceInit(&state);

    bool isFirstTask = true;
    AICORE_TIMEOUT_CHECK_BEGIN(t0, loop_count);
    while (true) {
        __gm__ DrcoDeviceTask* deviceTask = GetDrcoDeviceTask(&state, isFirstTask, t0, loop_count, warningSet);
        if (deviceTask == nullptr) {
            break;
        }

        UpdateCacheDevTask(&state.ctx, state.ctx.curLeafTaskParallelIdx, (int64_t)deviceTask->dynFuncDataList);
        __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList = deviceTask->drcoRootFuncList;
        state.ctx.lastTaskFinishCycle = 0;
        state.isExectedLeafTask = false;

        __gm__ npu::tile_fwk::PerCorePendingQueue* perCoreQueue = DrcoGmLoad(
            &rootFuncList->perCorePendingQueueArray[BlockDescBlockIdx(state.blockDesc)]);
        DRCO_DCCI_SINGLE_CACHE_LINE(perCoreQueue);
#if defined(__MIX__)
#if defined(__AIV__)
        perCoreQueue->mixHubC2VReadyQueue = reinterpret_cast<npu::tile_fwk::MixHubC2VReadyQueue*>(
            get_subblockid() == 1 ? sizeof(npu::tile_fwk::MixHubC2VReadyQueue) : 0);
        wait_intra_block(PIPE_S, EVENT_ID14);
#else
        perCoreQueue->mixHubC2VReadyQueue = reinterpret_cast<npu::tile_fwk::MixHubC2VReadyQueue*>(0);
        npu::tile_fwk::MixHubC2VReadyQueue::Init(perCoreQueue->mixHubC2VReadyQueue);
        npu::tile_fwk::MixHubC2VReadyQueue::Init(perCoreQueue->mixHubC2VReadyQueue + 1);
        set_intra_block(PIPE_S, EVENT_ID14);                      // 作用于Vec0
        set_intra_block(PIPE_S, EVENT_ID14 + EVENT_NUMS_PER_AIV); // 作用与Vec1
#endif
#endif
        if (DrcoGmLoad(&rootFuncList->totalTaskCount) == 0) {
            uint32_t oldHead = DrcoAtomicLoad(&state.deviceTaskReadyQueue->head);
            DrcoAtomicCasToU32(&state.deviceTaskReadyQueue->head, oldHead, oldHead + 1);
            continue;
        }

        state.readyMatrixPopRowIndex = 0;
        state.readyMatrixPushGroupIndex = BlockDescTypedBlockIdx(state.blockDesc) / npu::tile_fwk::LOCAL_GROUP_SIZE;
        state.readyMatrixPushColIdx = BlockDescTypedBlockIdx(state.blockDesc) % npu::tile_fwk::LOCAL_GROUP_SIZE;

        ExecDrcoPerCoreTasks(&state, perCoreQueue, rootFuncList);
        ExecDrcoReadyQueueTasks(&state, rootFuncList, perCoreQueue);

        SyncAllMix();
        if (BlockDescBlockIdx(state.blockDesc) == 0) {
            DrcoGmStore(&rootFuncList->devTaskFinished, (uint32_t)1);
        }
        DfxProcWhenDevTaskStop(&state.ctx, state.args, state.metric, state.isExectedLeafTask);
    }
    if (BlockDescBlockIdx(state.blockDesc) == 0) {
        state.deviceTaskReadyQueue->head = 0;
        state.deviceTaskReadyQueue->tail = 0;
        uint64_t finished = DrcoGmLoad(&state.runtimeDataRingBufferHeadData->indexFinished.value) + 1;
        DrcoGmStore(&state.runtimeDataRingBufferHeadData->indexFinished.value, finished);
    }

    state.args->taskEntry.reserved[0] = state.ctx.profLevel;
    DfxProcWhenCoreExit(&state.ctx, state.args, state.metric);

    TracePrint(&state);
    return;
}

#endif //__DAV_C310__
#endif
