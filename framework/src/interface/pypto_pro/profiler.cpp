/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>

#ifdef BUILD_WITH_CANN
#include <cstdint>
#include <cstring>

#include "profiling/aprof_pub.h"
#endif

namespace {

// Keep the state in the process-wide backend, which Python loads before JIT
// libraries. CANN replays an active session asynchronously to late registrants,
// so registering in a JIT library can race its first launch.
std::atomic<bool> gProfLive{false};

#ifdef BUILD_WITH_CANN
int32_t ProfCommand(uint32_t type, void* data, uint32_t length)
{
    // ProfCtrlType::PROF_CTRL_SWITCH carries MsprofCommandHandle; a zero
    // profSwitch stops collection. Ignore reporter/step-info notifications.
    if (type != PROF_CTRL_SWITCH || data == nullptr || length < sizeof(uint64_t)) {
        return 0;
    }
    uint64_t profSwitch = 0;
    std::memcpy(&profSwitch, data, sizeof(profSwitch));
    gProfLive.store(profSwitch != 0, std::memory_order_relaxed);
    return 0;
}

// Use CANN's public API: the interface library must not include adapter-private
// headers. Register at backend load time, before profiling or JIT compilation.
constexpr uint32_t pyptoModuleId = 0x46;
const int32_t gProfRegistration = MsprofRegisterCallback(pyptoModuleId, &ProfCommand);
#endif

} // namespace

// Native ABI used by generated launchers. libtile_fwk_interface is loaded with
// RTLD_GLOBAL by pypto._loader, so every JIT library queries this same state.
extern "C" bool GetPyptoProfStatus() { return gProfLive.load(std::memory_order_relaxed); }
