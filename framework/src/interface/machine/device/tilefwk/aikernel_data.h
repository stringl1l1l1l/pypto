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
 * \file aikernel_data.h
 * \brief
 */

#ifndef AIKERNEL_DATA_H
#define AIKERNEL_DATA_H
#include <atomic>
#include "tilefwk/aikernel_define.h"
#include "tilefwk/aikernel_tensor.h"
#include "tilefwk/aikernel_device_task.h"
#include "tilefwk/aikernel_runtime_data_ring_buffer.h"

struct LogContext;

namespace npu::tile_fwk {

constexpr uint32_t HCCL_GROUP_NUM = 2;

// AIV/AIC/HUB_MIX 取值与 DRCO_QUEUE_AIV/AIC/MIX 一一对应：DRCO 设备侧把 taskId 中解码出的 coreType
// 直接用作就绪队列下标；调整取值须与 DRCO_QUEUE_* 联动（下方 static_assert 强校验）
enum class CoreType { AIV = 0, AIC = 1, HUB_MIX = 2, AICPU = 3, HUB = 4, GMATOMIC = 5, MIX = 6, INVALID = 20 };
static_assert(static_cast<uint32_t>(CoreType::AIV) == DRCO_QUEUE_AIV, "CoreType::AIV must align with DRCO_QUEUE_AIV");
static_assert(static_cast<uint32_t>(CoreType::AIC) == DRCO_QUEUE_AIC, "CoreType::AIC must align with DRCO_QUEUE_AIC");
static_assert(static_cast<uint32_t>(CoreType::HUB_MIX) == DRCO_QUEUE_MIX,
              "CoreType::HUB_MIX must align with DRCO_QUEUE_MIX");

struct CoreFuncParam {
    __gm__ npu::tile_fwk::DynFuncData* funcData;
    __gm__ npu::tile_fwk::DynFuncHeader* funcHeader;
    __gm__ uint64_t* opAttrs;
    __gm__ uint64_t* exprTbl;
    uint32_t taskId;
    LogContext* ctx;
};

/*
    |--------16bit-------------|----16bit----|----1bit----|-----1bit------|------1bit-----|-----3bit--------|---10bit---|---16bit--|
    |-parallel ctx modifyflag--|--devtaskid--|----rspflag-|--pingpongflag-|---dcci flag---|--prallel index--|--func
   id--|--opindex-|
*/
#define TASKID_TASK_BITS 16
#define TASKID_TASK_MASK ((1 << TASKID_TASK_BITS) - 1)

#define TASKID_FUNC_BITS 10
#define TASKID_FUNC_MASK ((1 << TASKID_FUNC_BITS) - 1)

#define TASKID_PARALLEL_INDEX_BITS 3
#define TASKID_PARALLEL_INDEX_MASK ((1 << TASKID_PARALLEL_INDEX_BITS) - 1)

#define TASKID_DEVTASK_DCCI_BITS 1
#define TASKID_DEVTASK_DCCI_MASK ((1 << TASKID_DEVTASK_DCCI_BITS) - 1)

/* DRCO-only: CoreType enum encoded into taskId bit31-29 (rspflag/pingpong/dcci flags, all consumed
 * only by the AICPU-dispatch path, which never runs under DRCO). Host encodes at DRCO successor-table
 * / stitch; device decodes to skip cceBinaryIndexList + cceBinary GM dereference. Per-core dispatch
 * queues stay unencoded: the queue slot already fixes the core type and ExecDrcoPerCoreTasks reads
 * only FuncID/TaskID. Bit28 (parallel-index MSB) left intact. */
constexpr uint32_t TASKID_DRCO_CT_SHIFT = TASKID_TASK_BITS + TASKID_FUNC_BITS + TASKID_PARALLEL_INDEX_BITS;
constexpr uint32_t TASKID_DRCO_CT_MASK = 0x7;

INLINE bool IsValidDrcoCoreType(uint32_t coreType)
{
    return coreType == static_cast<uint32_t>(CoreType::AIV) || coreType == static_cast<uint32_t>(CoreType::AIC) ||
           coreType == static_cast<uint32_t>(CoreType::HUB) || coreType == static_cast<uint32_t>(CoreType::HUB_MIX);
}

INLINE uint32_t EncodeDrcoCoreType(uint32_t taskId, uint32_t coreType)
{
    return taskId | (coreType << TASKID_DRCO_CT_SHIFT);
}

/* DRCO-only: successor-table pairing. Entry bits 16-28 are free in the drco-encoded succ table
 * (entries are per-func opIdx, not full taskIds). Bit16 on an entry marks "this entry and the next
 * one are an even-aligned (2k, 2k+1) opIdx pair" (set at encode after a per-segment sort), so the
 * device resolve can decrement both packed predCount halves with one u64 atomicAdd. The even
 * alignment keeps the u64 slot naturally aligned: an odd start would both fault the hardware
 * atomic and overlap neighbouring pair domains without atomicity. Runtime pairing degrades
 * gracefully: entries without the bit take the original 32-bit path. */
constexpr uint32_t DRCO_SUCC_PAIR_BIT = 1u << 16;
/* u64 addend = -(1 | 1<<32): subtracts 1 from each 32-bit half of a packed predCount pair slot.
 * Valid only while neither half underflows (each pred edge resolves exactly once). */
constexpr uint64_t DRCO_SUCC_PAIR_DEC = 0xFFFFFFFEFFFFFFFFull;
/* Packed slot value (low | high << 32) where both halves are 1: each op's last unresolved edge is
 * the pair's own half-edge, so the resolver can fire both without the atomic. */
constexpr uint64_t DRCO_SUCC_PAIR_BOTH_ONE = 0x0000000100000001ull;

#define TASKID_SHIFT32 32
#define TASKID_FROM_CTRL_TOPO_MASK ((1 << (TASKID_TASK_BITS + TASKID_FUNC_BITS)) - 1)

const uint32_t SCH_DEVTASK_MAX_PARALLELISM = (1 << TASKID_PARALLEL_INDEX_BITS);

INLINE uint32_t FuncID(uint32_t taskId) { return (taskId >> TASKID_TASK_BITS) & TASKID_FUNC_MASK; }

INLINE uint32_t TaskID(uint32_t taskId) { return taskId & TASKID_TASK_MASK; }

INLINE uint32_t MakeTaskID(uint32_t rootId, uint32_t leafId) { return (rootId << TASKID_TASK_BITS) | leafId; }

INLINE uint32_t ParallelIndex(uint32_t taskId)
{
    return (taskId >> (TASKID_TASK_BITS + TASKID_FUNC_BITS)) & TASKID_PARALLEL_INDEX_MASK;
}

INLINE uint32_t DevTaskDcciFlag(uint32_t taskId)
{
    return (taskId >> (TASKID_TASK_BITS + TASKID_FUNC_BITS + TASKID_PARALLEL_INDEX_BITS)) & TASKID_DEVTASK_DCCI_MASK;
}

INLINE uint32_t MakeDrcoTaskId(uint32_t funcIdx, uint32_t opIdx, uint32_t coreType)
{
    return EncodeDrcoCoreType(MakeTaskID(funcIdx, opIdx), coreType);
}

INLINE uint32_t DrcoTaskCoreTypeOf(uint32_t taskId) { return (taskId >> TASKID_DRCO_CT_SHIFT) & TASKID_DRCO_CT_MASK; }

#define REG_VAL_DEVTASK_ID_BITS 24
#define REG_VAL_DEVTASK_ID_MASK ((1 << REG_VAL_DEVTASK_ID_BITS) - 1)

#define REG_VAL_PARALLEL_DEVTASK_CTX_MODIFYFLAG_BITS 8
#define REG_VAL_PARALLEL_DEVTASK_CTX_MODIFYFLAG_MASK ((1 << REG_VAL_PARALLEL_DEVTASK_CTX_MODIFYFLAG_BITS) - 1)

INLINE uint32_t DevTaskId(uint64_t highRegValue) { return highRegValue & REG_VAL_DEVTASK_ID_MASK; }

INLINE uint32_t ParallelDevTaskModifyFlag(uint64_t highRegValue)
{
    return (highRegValue >> REG_VAL_DEVTASK_ID_BITS) & REG_VAL_PARALLEL_DEVTASK_CTX_MODIFYFLAG_MASK;
}

} // namespace npu::tile_fwk

#endif
