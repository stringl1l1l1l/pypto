/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <acl/acl_prof.h>
#include <acl/acl_rt.h>
#include <profiling/aprof_pub.h>
#include <cstdlib>
#include <cstring>
#include <sys/syscall.h>
#include <unistd.h>

// Process-wide collection state, registered when the PyPTO backend loads.
// Unlike the compiler-linked ASC flag, this also covers profiling sessions
// that start before this JIT library is compiled or loaded.
extern "C" bool GetPyptoProfStatus() __attribute__((weak));

namespace pypto {

inline bool GetProfStatus()
{
    // Offline error reproduction loads the dumped launcher without importing
    // PyPTO. Without the backend's collection state, leave profiling disabled.
    return GetPyptoProfStatus != nullptr && GetPyptoProfStatus();
}

inline uint64_t BeginCaptureTensorReport(bool profiling)
{
    if (!profiling) {
        return 0;
    }
    // Report for every launch, captured or eager: the toolchain launch stub no
    // longer reports node basics and the range (Tx) pipeline no longer persists
    // eager tensor data, so the exporter needs the full record set from us.
    return MsprofSysCycleTime();
}

// Runtime's cached op-info wire header, followed by tensorNum MsrofTensorData
// records. Keep in sync with CacheTaskInfo in machine/runtime/runner/host_prof.h
// and CANN's CacheOpInfoBasic. Use public CANN types here: JIT headers cannot
// depend on the framework's private adapter or machine headers.
struct ProfCacheTaskInfo {
    uint32_t taskType;
    uint32_t numBlocks;
    uint64_t nodeId;
    uint64_t opType;
    uint64_t attrId;
    uint64_t reserve;
    uint32_t opFlag;
    uint32_t tensorNum;
};
static_assert(sizeof(ProfCacheTaskInfo) == 48, "Unexpected cached op-info layout");
static_assert(sizeof(MsrofTensorData) == 44, "Unexpected cached tensor layout");

inline MsrofTensorData MakeProfTensorData(const aclprofTensor& source)
{
    MsrofTensorData dest{};
    dest.tensorType = source.type;
    dest.format = source.format;
    dest.dataType = source.dataType;
    for (uint32_t dim = 0; dim < source.shapeDim && dim < MSPROF_GE_TENSOR_DATA_SHAPE_LEN; ++dim) {
        dest.shape[dim] = source.shape[dim];
    }
    return dest;
}

inline void CacheCaptureTensorInfo(const aclprofTensorInfo& info)
{
    // aclrtCacheLastTaskOpInfo accepts at most 64 KiB, copies the bytes, and
    // attaches them to this thread's last task. Call immediately after launch.
    constexpr size_t maxInfoSize = 64 * 1024;
    if (info.tensorNum > (maxInfoSize - sizeof(ProfCacheTaskInfo)) / sizeof(MsrofTensorData)) {
        return;
    }
    const size_t size = sizeof(ProfCacheTaskInfo) + info.tensorNum * sizeof(MsrofTensorData);
    auto* buffer = static_cast<unsigned char*>(std::malloc(size));
    if (buffer == nullptr) {
        return;
    }
    const ProfCacheTaskInfo task{info.kernelType, info.blockNums, info.opNameId, info.opTypeId, 0, 0, 0,
                                 info.tensorNum};
    std::memcpy(buffer, &task, sizeof(task));
    for (uint32_t index = 0; index < info.tensorNum; ++index) {
        const auto tensor = MakeProfTensorData(info.tensors[index]);
        std::memcpy(buffer + sizeof(task) + index * sizeof(tensor), &tensor, sizeof(tensor));
    }
    (void)aclrtCacheLastTaskOpInfo(buffer, size);
    std::free(buffer);
}

inline void ReportCaptureTensorInfo(const aclprofTensorInfo& info, uint64_t begin, bool capture)
{
    // Capture may precede collection (for example a profiler schedule's skipped
    // steps). Replay does not re-enter this launcher: Runtime must retain shapes
    // on the captured task and report them when that graph is profiled later.
    if (capture) {
        CacheCaptureTensorInfo(info);
    }
    if (begin == 0) {
        return;
    }
    const uint64_t end = MsprofSysCycleTime();
    const uint64_t timestamp = end;
    const auto threadId = static_cast<uint32_t>(syscall(SYS_gettid));

    // The exporter anchors eager kernel rows on an aging node-basic record and
    // only then attaches the additional tensor information, as the aclnn op
    // framework reports both with one shared timestamp. Emit the anchor first:
    // newer toolkits no longer report it from the launch stub.
    MsprofCompactInfo anchor{};
    anchor.level = MSPROF_REPORT_NODE_LEVEL;
    anchor.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
    anchor.threadId = threadId;
    anchor.timeStamp = timestamp;
    anchor.dataLen = sizeof(MsprofNodeBasicInfo);
    auto& node = anchor.data.nodeBasicInfo;
    node.opName = info.opNameId;
    node.taskType = info.kernelType;
    node.opType = info.opTypeId;
    node.blockDim = info.blockNums & 0xffffU;
    node.opFlag = 0;
    // Metadata collection must never prevent or fail a kernel launch.
    (void)MsprofReportCompactInfo(true, &anchor, sizeof(anchor));

    // The launch api event (node level, launch type, itemId = opName) spans the
    // op window and is what the exporter joins the anchor and tensor records
    // against; the aclnn op framework reports it for every eager op.
    MsprofApi launchEvent{};
    launchEvent.level = MSPROF_REPORT_NODE_LEVEL;
    launchEvent.type = MSPROF_REPORT_NODE_LAUNCH_TYPE;
    launchEvent.threadId = threadId;
    launchEvent.beginTime = begin;
    launchEvent.endTime = end;
    launchEvent.itemId = info.opNameId;
    (void)MsprofReportApi(true, &launchEvent);

    // Live node tensors cover eager execution and collection during capture.
    // They supplement, rather than replace, Runtime's cached replay metadata.
    for (uint32_t offset = 0; offset < info.tensorNum; offset += MSPROF_GE_TENSOR_DATA_NUM) {
        MsprofAdditionalInfo additional{};
        additional.level = MSPROF_REPORT_NODE_LEVEL;
        additional.type = MSPROF_REPORT_NODE_TENSOR_INFO_TYPE;
        additional.threadId = threadId;
        additional.timeStamp = timestamp;
        additional.dataLen = sizeof(MsprofTensorInfo);
        auto& tensors = *reinterpret_cast<MsprofTensorInfo*>(additional.data);
        tensors.opName = info.opNameId;
        const uint32_t remaining = info.tensorNum - offset;
        tensors.tensorNum = remaining < MSPROF_GE_TENSOR_DATA_NUM ? remaining : MSPROF_GE_TENSOR_DATA_NUM;
        for (uint32_t index = 0; index < tensors.tensorNum; ++index) {
            tensors.tensorData[index] = MakeProfTensorData(info.tensors[offset + index]);
        }
        // Metadata collection must never prevent or fail a kernel launch.
        (void)MsprofReportAdditionalInfo(true, &additional, sizeof(additional));
    }
}

} // namespace pypto
