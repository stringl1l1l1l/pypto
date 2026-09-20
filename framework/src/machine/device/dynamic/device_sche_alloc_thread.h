/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <algorithm>
#include "device_common.h"
#include "device_utils.h"
#include "machine/utils/device_log.h"

namespace npu::tile_fwk::dynamic {

constexpr int DAV3510_CPUS_PER_CLUSTER = 2;
constexpr int DAV3510_START_CPU_ID = 3;
constexpr int MAX_CPU_MASK_NUM = 16;
constexpr int CPU_ID_LOW_BOUND_DIE0 = 4;
constexpr int CPU_ID_HIGH_BOUND_DIE0 = 7;
constexpr int CPU_ID_LOW_BOUND_DIE1 = 12;
constexpr int CPU_ID_HIGH_BOUND_DIE1 = 15;
constexpr int DAV2201_DUAL_SCHE_NUM = 2;
constexpr int DAV2201_SINGLE_SCHE_NUM = 1;

constexpr uint64_t CPU_MASK_DIE0 = ((1ULL << (CPU_ID_HIGH_BOUND_DIE0 - CPU_ID_LOW_BOUND_DIE0 + 1)) - 1)
                                   << CPU_ID_LOW_BOUND_DIE0;
constexpr uint64_t CPU_MASK_DIE1 = ((1ULL << (CPU_ID_HIGH_BOUND_DIE1 - CPU_ID_LOW_BOUND_DIE1 + 1)) - 1)
                                   << CPU_ID_LOW_BOUND_DIE1;
constexpr uint64_t CLUSTER_CPU_MASK = CPU_MASK_DIE0 | CPU_MASK_DIE1;

enum ArbitrationLevel : int {
    ARBIT_ARBITRATING = -2,        // 仲裁线程已选出，仲裁结果尚未发布
    ARBIT_FAILED = -1,             // 资源完全不足，分配失败
    ARBIT_UNSET = 0,               // 仲裁尚未发布
    ARBIT_A5_SAME_DIE_CLUSTER = 1, // 同 die 同 cluster
    ARBIT_A5_CROSS_DIE = 2,        // A5 跨 die / 动态降级（实际 sche 数看 arbitrationCpumask）
    ARBIT_A5_SAME_DIE = 3,         // A5 同 die 分配
    ARBIT_A2A3_SAME_CLUSTER = 4,
    ARBIT_A2A3_CROSS_CLUSTER = 5, // A2 A3 跨 cluster
    ARBIT_A2A3_DUAL_SCHE = 6,     // A2 A3 两个 sche 线程
    ARBIT_A2A3_SINGLE_SCHE = 7,   // A2 A3 一个 sche 线程
};

struct DieMaskInfo {
    int die0MaxCpuid;      // die0 的最大 CPU ID
    uint64_t die0Boundary; // die0 的边界 mask：(1ULL << (die0MaxCpuid+1)) - 1
    uint64_t die0Mask;     // die0 的可用 CPU mask：cpumask & die0Boundary
    uint64_t die1Mask;     // die1 的可用 CPU mask：cpumask & ~die0Boundary
    int die0Cnt;           // die0 可用 CPU 数量
    int die1Cnt;           // die1 可用 CPU 数量

    DieMaskInfo(int die0Max, uint64_t mask)
    {
        die0MaxCpuid = die0Max;
        die0Boundary = (1ULL << (die0MaxCpuid + 1)) - 1;
        die0Mask = mask & die0Boundary;
        die1Mask = mask & ~die0Boundary;
        die0Cnt = __builtin_popcount(die0Mask);
        die1Cnt = __builtin_popcount(die1Mask);
    }

    int TotalCnt() const { return die0Cnt + die1Cnt; }
};

static inline int GetSameClusterCpuCnt(std::atomic<uint64_t>& cpumask)
{
    uint64_t mask = cpumask.load(std::memory_order_acquire);
    return __builtin_popcountll(mask & CLUSTER_CPU_MASK);
}

inline uint64_t GetArbitTimeOutVal(ArchInfo archInfo)
{
    uint64_t arbitTimeout;
    if (archInfo == ArchInfo::DAV_3510) {
        arbitTimeout = TIMEOUT_A5_50US;
        DEV_IF_DEBUG { arbitTimeout = TIMEOUT_A5_5SEC; }
        DEV_IF_INFO { arbitTimeout = TIMEOUT_A5_5SEC; }
    } else {
        arbitTimeout = TIMEOUT_A2A3_50US;
        DEV_IF_INFO { arbitTimeout = TIMEOUT_A2A3_1MS; }
        DEV_IF_DEBUG { arbitTimeout = TIMEOUT_A2A3_2MS; }
    }
    return arbitTimeout;
}

inline int WaitForCpuMaskReadyForArbitration(ArchInfo archInfo, int targetVal, std::atomic<uint64_t>& mask,
                                             std::atomic<uint64_t>* snapshot = nullptr, bool osThrottleFallback = false)
{
    uint64_t arbitTimeout = osThrottleFallback ? GetOsThrottleSafeWaitTimeout(archInfo) : GetArbitTimeOutVal(archInfo);
    TIMEOUT_CHECK_INIT(archInfo, arbitTimeout);

    uint32_t cpumask = static_cast<uint32_t>(mask.load(std::memory_order_acquire));
    while (__builtin_popcount(cpumask) < targetVal) {
        cpumask = static_cast<uint32_t>(mask.load(std::memory_order_acquire));
        __PYPTO_TIMEOUT_CHECK_WARN_EXIT(return DEVICE_MACHINE_ERROR, "#cur alloc %d cpu failed.", targetVal);
    }
    if (snapshot != nullptr) {
        snapshot->store(cpumask, std::memory_order_release);
    }
    return DEVICE_MACHINE_OK;
}

inline int ComputeArbitrationLevel(DeviceArgs* devArgs, std::atomic<uint64_t>& cpumask,
                                   std::atomic<uint64_t>& arbitrationCpumask)
{
    int die0CpuNum = static_cast<int>(devArgs->die0MaxCpuid) - DAV3510_START_CPU_ID + 1;
    int die0MaxCpuNum = ((die0CpuNum + DAV3510_CPUS_PER_CLUSTER - 1) / DAV3510_CPUS_PER_CLUSTER) *
                        DAV3510_CPUS_PER_CLUSTER;
    int nrAicpu = static_cast<int>(devArgs->nrAicpu);

    int ret = WaitForCpuMaskReadyForArbitration(devArgs->archInfo, nrAicpu, cpumask, &arbitrationCpumask);
    if (ret == DEVICE_MACHINE_OK) {
        return ARBIT_A5_SAME_DIE_CLUSTER;
    }

    // 允许部分 die1 的 CPU 未就绪，但 die0 至少有足够的 cluster
    ret = WaitForCpuMaskReadyForArbitration(devArgs->archInfo, die0MaxCpuNum + DAV3510_CPUS_PER_CLUSTER, cpumask,
                                            &arbitrationCpumask);
    if (ret == DEVICE_MACHINE_OK) {
        return ARBIT_A5_SAME_DIE;
    }

    // 从 scheCpuNum 到 1 依次降级：短超时探测，到位几个就用几个；实际 sche 数由 arbitrationCpumask 快照决定
    for (int arbitScheNum = static_cast<int>(devArgs->scheCpuNum); arbitScheNum >= 1; arbitScheNum--) {
        ret = WaitForCpuMaskReadyForArbitration(devArgs->archInfo, arbitScheNum, cpumask, &arbitrationCpumask);
        if (ret == DEVICE_MACHINE_OK) {
            return ARBIT_A5_CROSS_DIE;
        }
    }
    DEV_ERROR(DevCommonErr::PARAM_INVALID,
              "#sche.arbitration.failed: All arbitration levels failed. scheCpuNum=%u, currentCpuNum=%d, cpumask=%lx",
              devArgs->scheCpuNum, __builtin_popcount(static_cast<uint32_t>(cpumask.load(std::memory_order_acquire))),
              cpumask.load(std::memory_order_acquire));
    return ARBIT_FAILED;
}

inline int WaitForArbitrationLevel(ArchInfo archInfo, std::atomic<int>& globalArbitrationLevel)
{
    uint64_t arbitTimeout = GetOsThrottleSafeWaitTimeout(archInfo);
    TIMEOUT_CHECK_INIT(archInfo, arbitTimeout);

    int level = globalArbitrationLevel.load(std::memory_order_acquire);
    while (level == ARBIT_UNSET || level == ARBIT_ARBITRATING) {
        __PYPTO_TIMEOUT_CHECK_WARN_EXIT(return ARBIT_FAILED, "#cur alloc cpu arbitration wait failed.");
        level = globalArbitrationLevel.load(std::memory_order_acquire);
    }

    return level;
}

/**
 * 单线程仲裁：第一个到达的线程执行一次三级递进判定并发布全局级别，
 * 其余线程轻量等待后读取同一结果。
 */
inline int PerformArbitrationDav3510(DeviceArgs* devArgs, std::atomic<uint64_t>& cpumask,
                                     std::atomic<int>& globalArbitrationLevel,
                                     std::atomic<uint64_t>& arbitrationCpumask)
{
    int expected = ARBIT_UNSET;
    if (globalArbitrationLevel.compare_exchange_strong(expected, ARBIT_ARBITRATING, std::memory_order_acq_rel,
                                                       std::memory_order_acquire)) {
        int level = ComputeArbitrationLevel(devArgs, cpumask, arbitrationCpumask);
        globalArbitrationLevel.store(level, std::memory_order_release);
        return level;
    }

    return WaitForArbitrationLevel(devArgs->archInfo, globalArbitrationLevel);
}

inline uint64_t SelectCpusForCluster(uint64_t dieMask, int needNum)
{
    if (needNum <= 0 || dieMask == 0) {
        return 0;
    }

    uint64_t selected = 0;
    int count = 0;

    // 第一阶段：优先选择相邻 CPU 对（cluster）
    // 遍历所有 CPU，查找 pairMask = 0b11 << cpu 的模式
    for (int cpu = 0; cpu < MAX_CPU_MASK_NUM && count < needNum - 1; cpu++) {
        // 3 ：连续 2 位为 1（例如 cpu=3 → pairMask=0b11000 = 3<<3）
        uint64_t pairMask = (3ULL << cpu);
        // 检查 pairMask 是否在 dieMask 中完整出现（cpu 和 cpu+1 都可用）
        if ((dieMask & pairMask) == pairMask) {
            // 将整个 pair 加入 selected
            selected |= pairMask;
            count += DAV3510_CPUS_PER_CLUSTER; // 增加 2 个 CPU
            cpu++;                             // 跳过 cpu+1，避免重复处理
        }
    }

    // 第二阶段：补充单个 CPU（当第一阶段未达到 needNum）
    // 选择剩余的单个可用 CPU（未形成 pair 的）
    for (int cpu = 0; cpu < MAX_CPU_MASK_NUM && count < needNum; cpu++) {
        // 检查：CPU 在 dieMask 中可用，且未被第一阶段选中
        if ((dieMask & (1ULL << cpu)) && !(selected & (1ULL << cpu))) {
            selected |= (1ULL << cpu);
            count++;
        }
    }

    return selected;
}

inline void CalculateDieScheNum(DeviceArgs* devArgs, const DieMaskInfo& info, int& die0ScheNum, int& die1ScheNum)
{
    die0ScheNum = 0;
    die1ScheNum = 0;
    int totalCnt = info.die0Cnt + info.die1Cnt;

    if (totalCnt > 0) {
        // 计算原始比例：按可用 CPU 数量分配
        int rawDie0ScheNum = static_cast<int>(devArgs->scheCpuNum) * info.die0Cnt / totalCnt;

        // 向上取整到 cluster 倍数：确保分配完整 cluster
        // 例如 rawDie0ScheNum=3 → (3+2-1)/2*2 = 4
        if (rawDie0ScheNum % DAV3510_CPUS_PER_CLUSTER != 0) {
            die0ScheNum = ((rawDie0ScheNum + DAV3510_CPUS_PER_CLUSTER - 1) / DAV3510_CPUS_PER_CLUSTER) *
                          DAV3510_CPUS_PER_CLUSTER;
        } else {
            die0ScheNum = rawDie0ScheNum;
        }

        // 限制不超过可用数量和请求数量（防止分配超出实际可用）
        die0ScheNum = std::min(die0ScheNum, info.die0Cnt);
        die0ScheNum = std::min(die0ScheNum, static_cast<int>(devArgs->scheCpuNum));

        // die1 的分配数量：剩余部分
        die1ScheNum = static_cast<int>(devArgs->scheCpuNum) - die0ScheNum;
    }
}

inline int CalculateThreadIdx(int cpu, int die0ScheNum, uint64_t die0Selected, uint64_t die1Selected,
                              const DieMaskInfo& info)
{
    int threadIdx = 0;

    if (cpu <= info.die0MaxCpuid) {
        // die0：统计当前 CPU 在 die0Selected 中的位序
        // mask：从 bit 0 到 cpu 的 mask（包含 cpu）
        uint64_t mask = (1ULL << (cpu + 1)) - 1;
        // tmp：die0Selected 中 cpu 及之前的位
        uint64_t tmp = die0Selected & mask;
        // popcount：统计 tmp 中 1 的数量，得到 threadIdx
        threadIdx = __builtin_popcount(tmp);
    } else {
        // die1：die0ScheNum + 在 die1Selected 中的位序
        uint64_t mask = (1ULL << (cpu + 1)) - 1;
        uint64_t tmp = die1Selected & mask;
        int threadIdxInDie = __builtin_popcount(tmp);
        threadIdx = die0ScheNum + threadIdxInDie;
    }

    return threadIdx;
}

inline bool CheckThreadIdxUnique(int cpu, int curThreadIdx, std::atomic<uint64_t>& threadIdxBitmap, int level,
                                 uint64_t arbitrationCpumask, int die0ScheNum, uint64_t die0Selected,
                                 uint64_t die1Selected)
{
    uint64_t prevBitmap = threadIdxBitmap.fetch_or(1ULL << curThreadIdx, std::memory_order_acq_rel);
    if (prevBitmap & (1ULL << curThreadIdx)) {
        // AICPU blockDim 分波串行时，后波可能复用前波物理核并算出相同 threadIdx。
        // 前波已跑完对应 sche，后波应丢弃（threadIdx=-1），仍参与 SyncSchExit 凑齐 nrAicpu。
        DEV_WARN("#sche.thread.duplicate: Duplicate threadIdx, drop as excess block. "
                 "cpu=%d, threadIdx=%d, bitmap=0x%lx, level=%d, arbitrationCpumask=0x%lx, "
                 "die0ScheNum=%d, die0Selected=0x%lx, die1Selected=0x%lx",
                 cpu, curThreadIdx, prevBitmap, level, arbitrationCpumask, die0ScheNum, die0Selected, die1Selected);
        return false;
    }
    return true;
}

/**
 * DAV3510 线程分配主流程（核心实现）
 *
 * 流程步骤：
 * 1. 非设备模式或 die0MaxCpuid=0：直接递增 threadIdx（无仲裁）
 * 2. 更新 cpumask：标记当前 CPU 就绪
 * 3. 单线程仲裁：第一个到达的线程计算全局级别，其余线程等待并读取
 * 4. ARBIT_FAILED：报错
 * 5. CROSS_DIE：sche 数 = min(popcount(arbitrationCpumask), scheCpuNum)，多余线程 threadIdx=-1
 * 6. SAME_DIE_CLUSTER/SAME_DIE：按拓扑选 CPU 并算 threadIdx；未选中则 threadIdx=-1
 * 7. threadIdx 已被占用（分波复用物理核）：threadIdx=-1，不报错，仍走 SyncSchExit
 */
inline int AllocThreadIdxForDav3510Impl(DeviceArgs* devArgs, int cpu, int& curThreadIdx, std::atomic<int>& threadIdx,
                                        std::atomic<uint64_t>& cpumask, int& arbitratedScheNum,
                                        std::atomic<int>& globalArbitrationLevel,
                                        std::atomic<uint64_t>& arbitrationCpumask,
                                        std::atomic<uint64_t>& threadIdxBitmap)
{
#ifndef __DEVICE__
    curThreadIdx = ++threadIdx;
    arbitratedScheNum = static_cast<int>(devArgs->scheCpuNum);
    return DEVICE_MACHINE_OK;
#endif
    if (devArgs->die0MaxCpuid == 0) {
        curThreadIdx = ++threadIdx;
        arbitratedScheNum = static_cast<int>(devArgs->scheCpuNum);
        return DEVICE_MACHINE_OK;
    }

    cpumask.fetch_or(1ULL << cpu, std::memory_order_release);
    int level = PerformArbitrationDav3510(devArgs, cpumask, globalArbitrationLevel, arbitrationCpumask);
    if (level == ARBIT_FAILED) {
        DEV_ERROR(ThreadErr::THREAD_CPU_ALLOC_FAILED,
                  "#sche.thread.arbitration: Currently aicpus resources are insufficient. "
                  "Please use the launchSchedAicpuNum frontend interface to reduce the number of aicpus being used.");
        return DEVICE_MACHINE_ERROR;
    }

    const int scheCpuNum = static_cast<int>(devArgs->scheCpuNum);
    if (level == ARBIT_A5_CROSS_DIE) {
        // level release 之后读快照，popcount 即本轮实际可用 sche 数
        int scheNum = __builtin_popcount(static_cast<uint32_t>(arbitrationCpumask.load(std::memory_order_acquire)));
        scheNum = std::min(scheNum, scheCpuNum);
        arbitratedScheNum = scheNum;
        curThreadIdx = ++threadIdx;
        if (curThreadIdx > scheNum) {
            DEV_INFO("Arbitration drop excess sche thread: arbitratedScheNum=%d, physicalCpu=%d, threadIdx=%d", scheNum,
                     cpu, curThreadIdx);
            curThreadIdx = -1;
            return DEVICE_MACHINE_OK;
        }
        DEV_INFO("Arbitration succeeded. level=CROSS_DIE, arbitratedScheNum=%d, physicalCpu=%d, threadIdx=%d", scheNum,
                 cpu, curThreadIdx);
        return DEVICE_MACHINE_OK;
    }

    arbitratedScheNum = scheCpuNum;
    DieMaskInfo info(static_cast<int>(devArgs->die0MaxCpuid), arbitrationCpumask.load(std::memory_order_acquire));

    int die0ScheNum = 0;
    int die1ScheNum = 0;
    CalculateDieScheNum(devArgs, info, die0ScheNum, die1ScheNum);

    uint64_t die0Selected = SelectCpusForCluster(info.die0Mask, die0ScheNum);
    uint64_t die1Selected = SelectCpusForCluster(info.die1Mask, die1ScheNum);
    uint64_t selectedMask = die0Selected | die1Selected;
    int actualDie0Count = __builtin_popcount(die0Selected);
    int actualDie1Count = __builtin_popcount(die1Selected);
    if (actualDie0Count < die0ScheNum || actualDie1Count < die1ScheNum) {
        DEV_WARN("#sche.thread.alloc: Insufficient CPUs: die0 requested=%d actual=%d, "
                 "die1 requested=%d actual=%d",
                 die0ScheNum, actualDie0Count, die1ScheNum, actualDie1Count);
    }

    if (!(selectedMask & (1ULL << cpu))) {
        curThreadIdx = -1;
        return DEVICE_MACHINE_OK;
    }

    curThreadIdx = CalculateThreadIdx(cpu, die0ScheNum, die0Selected, die1Selected, info);
    if (!CheckThreadIdxUnique(cpu, curThreadIdx, threadIdxBitmap, level, info.die0Mask | info.die1Mask, die0ScheNum,
                              die0Selected, die1Selected)) {
        // 分波后波 / 重复占用：丢弃本 block，避免重复跑同一 threadIdx
        curThreadIdx = -1;
        return DEVICE_MACHINE_OK;
    }

    threadIdx.store(curThreadIdx, std::memory_order_release);
    DEV_INFO("Thread alloc success: level=%d, arbitratedScheNum=%d, physicalCpu=%d, threadIdx=%d.", level, scheCpuNum,
             cpu, curThreadIdx);
    return DEVICE_MACHINE_OK;
}

static inline int WaitForScheCpuInSameCluster(ArchInfo archInfo, int scheCpuNum, std::atomic<uint64_t>& cpumask)
{
    uint64_t arbitTimeout = GetArbitTimeOutVal(archInfo);
    TIMEOUT_CHECK_INIT(archInfo, arbitTimeout);
    while (GetSameClusterCpuCnt(cpumask) < scheCpuNum) {
        __PYPTO_TIMEOUT_CHECK_WARN_EXIT(return DEVICE_MACHINE_ERROR, "#cur alloc cpu in same cluster failed.");
    }
    return DEVICE_MACHINE_OK;
}

static inline int CalculateArbitLevelFromArbitScheNum(int scheCpuNum, int arbitScheNum, std::atomic<uint64_t>& cpumask)
{
    if (scheCpuNum == arbitScheNum) {
        if (GetSameClusterCpuCnt(cpumask) == scheCpuNum) {
            return ARBIT_A2A3_SAME_CLUSTER;
        } else {
            return ARBIT_A2A3_CROSS_CLUSTER;
        }
    } else if (arbitScheNum == DAV2201_DUAL_SCHE_NUM) {
        return ARBIT_A2A3_DUAL_SCHE;
    } else if (arbitScheNum == DAV2201_SINGLE_SCHE_NUM) {
        return ARBIT_A2A3_SINGLE_SCHE;
    }
    return ARBIT_FAILED;
}

static inline bool IsSameCpuCluster(const int cpu)
{
    if ((cpu >= CPU_ID_LOW_BOUND_DIE0 && cpu <= CPU_ID_HIGH_BOUND_DIE0) ||
        (cpu >= CPU_ID_LOW_BOUND_DIE1 && cpu <= CPU_ID_HIGH_BOUND_DIE1)) {
        return true;
    }
    return false;
}

static inline void AllocThreadIdByArbitrationLevel(int level, int& curThreadIdx, int isSameClusterThread,
                                                   std::atomic<int>& threadIdx)
{
    if (level == ARBIT_A2A3_SAME_CLUSTER) {
        if (!isSameClusterThread) {
            curThreadIdx = -1;
        } else {
            curThreadIdx = ++threadIdx;
        }
    } else {
        curThreadIdx = ++threadIdx;
    }
}

static inline int GetScheNumByArbitrationLevel(int scheCpuNum, int level)
{
    if (level == ARBIT_A2A3_SAME_CLUSTER || level == ARBIT_A2A3_CROSS_CLUSTER) {
        return scheCpuNum;
    } else if (level == ARBIT_A2A3_DUAL_SCHE) {
        return DAV2201_DUAL_SCHE_NUM;
    } else if (level == ARBIT_A2A3_SINGLE_SCHE) {
        return DAV2201_SINGLE_SCHE_NUM;
    } else {
        return 0;
    }
}

static inline int ComputeArbitrationLevelDav2201(const DeviceArgs* devArgs, std::atomic<uint64_t>& cpumask)
{
    // Launch Sched Same Cluster 模式, 有 scheCpuNum 在同一个 cluster 的线程就退出
    if (devArgs->launchSchedSameCluster) {
        int ret = WaitForScheCpuInSameCluster(devArgs->archInfo, static_cast<int>(devArgs->scheCpuNum), cpumask);
        if (ret == DEVICE_MACHINE_OK) {
            return ARBIT_A2A3_SAME_CLUSTER;
        }
    }
    // 从 devArgs->scheCpuNum 到 1 sche 依次降级；当前线程已登记 cpumask，降至 1 时必成功，无需 100ms
    for (int arbitScheNum = devArgs->scheCpuNum; arbitScheNum >= 1; arbitScheNum--) {
        int ret = WaitForCpuMaskReadyForArbitration(devArgs->archInfo, arbitScheNum, cpumask);
        if (ret == DEVICE_MACHINE_OK) {
            return CalculateArbitLevelFromArbitScheNum(static_cast<int>(devArgs->scheCpuNum), arbitScheNum, cpumask);
        }
    }
    return ARBIT_FAILED;
}

static inline int AllocThreadIdxForDav2201Impl(const DeviceArgs* devArgs, const int cpu, int& curThreadIdx,
                                               std::atomic<int>& threadIdx, std::atomic<uint64_t>& cpumask,
                                               int& arbitratedScheNum, std::atomic<int>& globalArbitrationLevel)
{
#ifndef __DEVICE__
    curThreadIdx = ++threadIdx;
    return DEVICE_MACHINE_OK;
#endif
    int ret = DEVICE_MACHINE_OK;
    // 判断线程是否在 Cluster1 上
    bool isSameClusterThread = IsSameCpuCluster(cpu);

    // 执行仲裁，level 通过单次 CAS 原子提交
    cpumask.fetch_or(1ULL << cpu, std::memory_order_release);
    int expected = ARBIT_UNSET;
    int level;
    if (globalArbitrationLevel.compare_exchange_strong(expected, ARBIT_ARBITRATING, std::memory_order_acq_rel,
                                                       std::memory_order_acquire)) {
        level = ComputeArbitrationLevelDav2201(devArgs, cpumask);
        globalArbitrationLevel.store(level, std::memory_order_release);
    } else {
        level = WaitForArbitrationLevel(devArgs->archInfo, globalArbitrationLevel);
    }

    if (level == ARBIT_FAILED) {
        DEV_ERROR(ThreadErr::THREAD_CPU_ALLOC_FAILED,
                  "#sche.thread.arbitration: Currently aicpus resources are insufficient. "
                  "Please use the launchSchedAicpuNum frontend interface to reduce the number of aicpus being used.");
        return DEVICE_MACHINE_ERROR;
    }
    int scheNum = GetScheNumByArbitrationLevel(static_cast<int>(devArgs->scheCpuNum), level);
    arbitratedScheNum = scheNum;
    AllocThreadIdByArbitrationLevel(level, curThreadIdx, isSameClusterThread, threadIdx);
    DEV_INFO("Arbitration succeeded. Arbitration level: %d, arbitrated sched num: %d", level, arbitratedScheNum);
    if (curThreadIdx == -1 || curThreadIdx > scheNum) {
        curThreadIdx = -1;
        return DEVICE_MACHINE_OK;
    }
    return ret;
}

} // namespace npu::tile_fwk::dynamic
