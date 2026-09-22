/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INTERFACE_UTILS_ERROR_H_
#define INTERFACE_UTILS_ERROR_H_

#include <stdlib.h>
#include <signal.h>
#include <exception>
#include <execinfo.h>
#include <iostream>

#include "tilefwk/error.h"
#include "tilefwk/error_code.h"
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_manager.h"

namespace npu::tile_fwk {
struct TerminateHandler {
    TerminateHandler();

    static void SigAction(int signo)
    {
        (void)signo;
        auto backtrace = GetBacktrace(0x2, 0x06)->Get();
        auto msg = "ops !!!";
        if (signo == SIGSEGV) {
            msg = "segment fault !!!";
        } else if (signo == SIGFPE) {
            msg = "floating point exception !!!";
        }
        PYPTO_LOGE_FULL("%s\n%s", msg, backtrace.c_str());
        ErrorManager::Instance().OutputErrorMessage();
        std::cerr << msg << "\n" << backtrace << std::endl;
        (void)fflush(nullptr);
        _Exit(1);
    }

    ~TerminateHandler()
    {
        sigaction(SIGSEGV, &ori[0], nullptr);
        sigaction(SIGFPE, &ori[1], nullptr);
    }

    struct sigaction ori[2];
};
} // namespace npu::tile_fwk
#endif
