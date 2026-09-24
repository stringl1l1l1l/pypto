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
 * \file aicore_model_launcher.h
 * \brief DebugMode AicoreModel host simulation launcher
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdint.h>
#include <string>
#include <vector>
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "machine/runtime/launcher/device_launcher_binding.h"
#include "machine/runtime/runner/runtime_utils.h"

namespace npu::tile_fwk::dynamic {

struct AicoreModelMemoryUtils {
    AicoreModelMemoryUtils() {}
    ~AicoreModelMemoryUtils() = default;
    static bool IsDevice() { return false; }
    // 默认 alignof(max_align_t)，与原先裸 malloc 的保证完全等价；需要更强对齐时显式传参
    uint8_t* AllocDev(size_t size, uint8_t** cachedDevAddrHolder, size_t align = alignof(std::max_align_t))
    {
        (void)cachedDevAddrHolder;
        if (size == 0 || size > 0x500000000) {
            MACHINE_LOGE(DevCommonErr::PARAM_INVALID, "AllocDev failed: size=%zu bytes", size);
            return nullptr;
        }
        uint8_t* rawPtr = (uint8_t*)aligned_alloc(align, (size + align - 1) / align * align);
        if (rawPtr == nullptr) {
            return nullptr;
        }
        std::shared_ptr<uint8_t> ptr(rawPtr, free);
        HostSimAllocatePtrs_.push_back(ptr);
        return rawPtr;
    }

    uint8_t* AllocZero(uint64_t size, uint8_t** cachedDevAddrHolder, size_t align = alignof(std::max_align_t))
    {
        (void)cachedDevAddrHolder;
        uint8_t* devPtr = AllocDev(size, nullptr, align);
        if (devPtr == nullptr) {
            return nullptr;
        }
        memset_s(devPtr, size, 0, size);
        return devPtr;
    }

    void CopyFromDev(uint8_t* data, uint8_t* devPtr, uint64_t size) { MemcpyS(data, size, devPtr, size); }

    template <typename T>
    T* CopyToDev(std::vector<T> data, uint8_t** cachedDevAddrHolder)
    {
        (void)cachedDevAddrHolder;
        return (T*)CopyToDev((uint8_t*)data.data(), data.size() * sizeof(T), nullptr);
    }

    uint8_t* CopyToDev(uint8_t* data, uint64_t size, uint8_t** cachedDevAddrHolder)
    {
        uint8_t* devPtr = AllocDev(size, cachedDevAddrHolder);
        if (devPtr != nullptr) {
            MemcpyS(devPtr, size, data, size);
        }
        return devPtr;
    }

    void CopyFromDev(RawTensorData& t) { CopyFromDev(t.data(), t.GetDevPtr(), t.size()); }

    uint8_t* CopyToDev(RawTensorData& data)
    {
        if (data.GetDevPtr() == nullptr) {
            auto devAddr = CopyToDev((uint8_t*)data.data(), data.size(), nullptr);
            data.SetDevPtr(devAddr);
        }
        return data.GetDevPtr();
    }

    uint64_t GetL2Offset() { return 0; }

private:
    std::vector<std::shared_ptr<uint8_t>> HostSimAllocatePtrs_;
};

class AicoreModelLauncher {
public:
    static int AicoreModelLaunchOnceWithHostTensorData(Function* function,
                                                       const std::vector<DeviceTensorData>& inputList,
                                                       const std::vector<DeviceTensorData>& outputList,
                                                       DevControlFlowCache* ctrlCache, AicoreModelMemoryUtils& memUtils,
                                                       const DeviceLauncherConfig& config = DeviceLauncherConfig());
    static int AicoreModelRunOnce(Function* function, DevControlFlowCache* inputCtrlCache,
                                  const DeviceLauncherConfig& config = DeviceLauncherConfig());
    static int AicoreModelLaunchDeviceTensorData(Function* function, const std::vector<DeviceTensorData>& inDevList,
                                                 const std::vector<DeviceTensorData>& outDevList,
                                                 const DeviceLauncherConfig& config = DeviceLauncherConfig(),
                                                 DevControlFlowCache* ctrlCache = nullptr);
};
} // namespace npu::tile_fwk::dynamic
