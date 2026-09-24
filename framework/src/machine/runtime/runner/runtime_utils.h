/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file runtime_utils.h
 * \brief
 */

#pragma once

#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error.h"
#include "tilefwk/error_code.h"
#include "tilefwk/platform.h"
#include "adapter/api/runtime_api.h"
#include "adapter/api/runtime_capture_context.h"
#include "adapter/api/acl_api.h"
#include "machine/runtime/context/stream_context.h"
#include "interface/utils/common.h"
#include "securec.h"

#define MemcpyS(dest, destMax, src, count) \
    npu::tile_fwk::MemcpySWithCheck((dest), (destMax), (src), (count), __func__, __FILE__, __LINE__)
#define RuntimeMemcpy(dst, destMax, src, cnt, kind) \
    npu::tile_fwk::RuntimeMemcpyWithCheck((dst), (destMax), (src), (cnt), (kind), __func__, __FILE__, __LINE__)
#define RuntimeMemcpyAsync(dst, destMax, src, cnt, kind, stm)                                                     \
    npu::tile_fwk::RuntimeMemcpyAsyncWithCheck((dst), (destMax), (src), (cnt), (kind), (stm), __func__, __FILE__, \
                                               __LINE__)

namespace npu::tile_fwk {
inline uint32_t GetAicoreLocalMemorySize(NPUArch arch, bool requiresSimt)
{
    return arch == NPUArch::DAV_3510 && requiresSimt ? A5_SIMT_DYNAMIC_UB_SIZE : 0U;
}

inline void MemcpySWithCheck(void* dest, size_t destMax, const void* src, size_t count, const char* func,
                             const char* file, int line)
{
    const errno_t ret = memcpy_s(dest, destMax, src, count);
    if (ret != EOK) {
        MACHINE_LOGE(DevCommonErr::MEMCPY_FAILED,
                     "memcpy_s failed: func=%s, file=%s:%d, ret=%d, dest=%p, destMax=%zu, src=%p, count=%zu", func,
                     file, line, ret, dest, destMax, src, count);
        MACHINE_ASSERT(false) << "memcpy_s failed, ret=" << ret;
    }
}

inline void CheckCaptureRelaxedBeforeMemcpy(const char* api, const char* func, const char* file, int line)
{
    if (!RuntimeCaptureContext::IsCaptureMode()) {
        return;
    }
    AclMdlRICaptureMode currentMode = AclMdlRICaptureMode::GLOBAL;
    const bool queryOk = RuntimeCaptureContext::QueryThreadCaptureMode(currentMode);
    if (!queryOk) {
        MACHINE_LOGE(RtErr::RT_MEMCPY_FAILED, "cannot query ACL capture thread mode before %s, func=%s, file=%s:%d",
                     api, func, file, line);
        MACHINE_ASSERT(false) << api << ": cannot query ACL capture thread mode";
    }
    if (currentMode != AclMdlRICaptureMode::RELAXED) {
        MACHINE_LOGE(RtErr::RT_MEMCPY_FAILED,
                     "%s requires RELAXED capture mode on this thread, "
                     "func=%s, file=%s:%d, currentMode=%d, required=RELAXED(%d), "
                     "fix: switch to RELAXED before %s "
                     "(AclMdlRICaptureThreadExchangeMode or AclModeGuard)",
                     api, func, file, line, static_cast<int>(currentMode),
                     static_cast<int>(AclMdlRICaptureMode::RELAXED), api);
        MACHINE_ASSERT(false) << api << " requires RELAXED capture mode";
    }
}

inline void RuntimeMemcpyWithCheck(void* dst, uint64_t destMax, const void* src, uint64_t cnt, RtMemcpyKind kind,
                                   const char* func, const char* file, int line)
{
    CheckCaptureRelaxedBeforeMemcpy("RuntimeMemcpy", func, file, line);
    const RtError ret = RuntimeMemcpyDirect(dst, destMax, src, cnt, kind);
    if (ret != RT_SUCCESS) {
        MACHINE_LOGE(RtErr::RT_MEMCPY_FAILED,
                     "RuntimeMemcpy failed: func=%s, file=%s:%d, ret=%d, kind=%d, size=%lu bytes, dst=%p, src=%p", func,
                     file, line, static_cast<int>(ret), static_cast<int>(kind), cnt, dst, src);
        MACHINE_ASSERT(false) << "RuntimeMemcpy failed, ret=" << static_cast<int>(ret);
    }
}

inline void RuntimeMemcpyAsyncWithCheck(void* dst, uint64_t destMax, const void* src, uint64_t cnt, RtMemcpyKind kind,
                                        RtStream stm, const char* func, const char* file, int line)
{
    CheckCaptureRelaxedBeforeMemcpy("RuntimeMemcpyAsync", func, file, line);
    const RtError ret = RuntimeMemcpyDirectAsync(dst, destMax, src, cnt, kind, stm);
    if (ret != RT_SUCCESS) {
        MACHINE_LOGE(RtErr::RT_MEMCPY_FAILED,
                     "RuntimeMemcpyAsync failed: func=%s, file=%s:%d, ret=%d, kind=%d, size=%lu bytes, dst=%p, src=%p",
                     func, file, line, static_cast<int>(ret), static_cast<int>(kind), cnt, dst, src);
        MACHINE_ASSERT(false) << "RuntimeMemcpyAsync failed, ret=" << static_cast<int>(ret);
    }
}

inline void CheckDeviceId()
{
    int32_t devId = 0;
    int32_t getDeviceResult = RuntimeGetDevice(&devId);
    if (getDeviceResult != RT_SUCCESS) {
        MACHINE_LOGE(RtErr::RT_DEVICE_FAILED, "Failed to get device id, please check if device id is set");
        return;
    }
    MACHINE_LOGI("Current device is %d.", devId);
}

inline int32_t GetUserDeviceId()
{
    int32_t userDeviceId = 0;
    RuntimeGetDevice(&userDeviceId);
    return userDeviceId;
}

inline int32_t GetLogDeviceId()
{
    int32_t logicDeviceId = 0;
    int32_t userDeviceId = GetUserDeviceId();
    ASSERT(RtErr::RT_DEVICE_FAILED, RuntimeGetLogicDevIdByUserDevId(userDeviceId, &logicDeviceId) == RT_SUCCESS)
        << "Trans usrDeviceId: " << userDeviceId << " to logDevId not success";
    MACHINE_LOGD("Current userDeviceId=%d, logicDeviceId=%d.", userDeviceId, logicDeviceId);
    return logicDeviceId;
}

inline uint64_t GetRuntimeL2Offset()
{
    uint64_t offset = 0;
    int32_t userDeviceId = GetUserDeviceId();
    RuntimeGetL2CacheOffset(userDeviceId, &offset);
    MACHINE_LOGD("L2 cache offset of device[%d] is %lu bytes", userDeviceId, offset);
    return offset;
}

inline uint64_t AlignSize(const uint64_t bytes, const uint32_t aligns = 512U)
{
    const uint64_t alignSize = (aligns == 0U) ? sizeof(uintptr_t) : aligns;
    return (((bytes + alignSize) - 1U) / alignSize) * alignSize;
}

inline void* DevMallocWithAlignSize(const uint64_t size, const RtMemType memType)
{
    uint8_t* devPtr = nullptr;
    if (RuntimeMalloc(reinterpret_cast<void**>(&devPtr), AlignSize(size), memType, PYPTO) != 0) {
        MACHINE_LOGW("Fail to malloc dev memory with size[%lu] bytes and mem type[%u].", size, memType);
        return nullptr;
    }
    if (RuntimeMemset(devPtr, AlignSize(size), 0, AlignSize(size)) != 0) {
        MACHINE_LOGW("Fail to memset of size[%lu] bytes.", size);
        RuntimeFree(devPtr);
        return nullptr;
    }
    return devPtr;
}

inline void CopyFromTensor(uint8_t* hostDstAddr, const uint8_t* devSrcAddr, const uint64_t size)
{
#ifdef RUN_WITH_ASCEND_CAMODEL
    RuntimeMemcpy(hostDstAddr, size, devSrcAddr, size, RtMemcpyKind::DEVICE_TO_HOST);
#else
    RuntimeMemcpyAsync(hostDstAddr, size, devSrcAddr, size, RtMemcpyKind::DEVICE_TO_HOST,
                       GetStreamContext().GetAiCoreStream());
    RuntimeStreamSynchronize(GetStreamContext().GetAiCoreStream());
#endif
}

inline void ExchangeCaptureModeRelax()
{
    AclMdlRICaptureMode mode = AclMdlRICaptureMode::RELAXED;
    // aclgraph does not support rtmemcpy / rtmemset, set to relaxed mode
    AclMdlRICaptureThreadExchangeMode(&mode);
}

inline void ExchangeCaptureModeGlobal()
{
    AclMdlRICaptureMode mode = AclMdlRICaptureMode::GLOBAL;
    AclMdlRICaptureThreadExchangeMode(&mode);
}

inline void* RegisterKernelBin(const std::vector<uint8_t>& kernelBinary)
{
    void* hdl = nullptr;
    if (kernelBinary.empty()) {
        MACHINE_LOGE(HostLauncherErr::REGISTER_KERNEL_FAILED, "Kernel binary is empty.");
        return hdl;
    }
    RtDevBinary binary = {
        .magic = RT_DEV_BINARY_MAGIC_ELF,
        .version = 0,
        .data = kernelBinary.data(),
        .length = kernelBinary.size(),
    };
    if (RuntimeRegisterAllKernel(&binary, &hdl) != RT_SUCCESS) {
        MACHINE_LOGE(HostLauncherErr::REGISTER_KERNEL_FAILED, "Failed to register kernel bin");
    }
    MACHINE_LOGD("Kernel binary has been registered successfully, size is [%zu] bytes.", kernelBinary.size());
    return hdl;
}

inline bool GetStreamCaptureInfo(RtStream aicoreStream, AclMdlRI& rtModel, bool& isCapture)
{
    AclMdlRICaptureStatus captureStatus = AclMdlRICaptureStatus::NONE;
    AclError ret = AclMdlRICaptureGetInfo(aicoreStream, &captureStatus, &rtModel);
    if (ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        MACHINE_LOGW("Stream capture is not supported");
        return true;
    }
    if (ret != ACLRT_SUCCESS) {
        MACHINE_LOGE(RtErr::RT_CAPTURE_FAILED, "AclMdlRICaptureGetInfo failed, return[%d]", ret);
        return false;
    }
    isCapture = captureStatus == AclMdlRICaptureStatus::ACTIVE;
    MACHINE_LOGI("Capture status [%d], capture mode[%d]", static_cast<int32_t>(captureStatus), isCapture);
    return true;
}

int GetCfgBlockdim();

uint32_t GetProcessId();
} // namespace npu::tile_fwk
