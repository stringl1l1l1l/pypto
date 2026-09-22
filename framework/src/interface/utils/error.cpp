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
 * \file error.cpp
 * \brief
 */

#include <cstring>
#include <sstream>
#include <functional>
#include <cxxabi.h>
#include <securec.h>
#include <algorithm>

#include "error.h"
#include "utils/host_log/log_time.h"
#include "interface/utils/string_utils.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk {

class BacktraceImpl : public LazyValue<std::string> {
public:
    __always_inline BacktraceImpl(size_t skipFrames, size_t maxFrames) : callStack_(maxFrames, 0)
    {
        skipFrames += 1;
        auto nrFrames = static_cast<size_t>(::backtrace(callStack_.data(), static_cast<int>(callStack_.size())));
        skipFrames = std::min(skipFrames, nrFrames);
        callStack_.erase(callStack_.begin(), callStack_.begin() + static_cast<ssize_t>(skipFrames));
        callStack_.resize(nrFrames - skipFrames);
    }

    void ParseFrame(std::stringstream& ss, char* line, bool& isPyptoFrame) const
    {
        auto funcName = strchr(line, '(');
        auto funcOffset = strchr(line, '+');
        auto libname = strrchr(line, '/');
        if (funcName == nullptr || funcOffset == nullptr) {
            ss << line << '\n';
            return;
        }

        *funcName++ = '\0';
        *funcOffset++ = '\0';
        libname = (libname == nullptr) ? line : libname + 1;
        if (!strncmp(libname, "pypto_impl", strlen("pypto_impl"))) {
            isPyptoFrame = true;
        } else if (isPyptoFrame) {
            // python frames after pypto frame, skip it
            return;
        }

        int status = 0;
        std::unique_ptr<char, std::function<void(char*)>> demangled(
            abi::__cxa_demangle(funcName, nullptr, nullptr, &status),
            /* deleter */ free);
        if (status == 0)
            funcName = demangled.get();
        ss << libname << '(' << funcName << '+' << funcOffset << '\n';
    }

    const std::string& Get() const
    {
        return symbols_.Ensure([this]() -> std::string {
            auto strings = backtrace_symbols(callStack_.data(), callStack_.size());
            if (strings == nullptr) {
                return "Backtrace Failed";
            }
            std::stringstream ss;
            bool isPyptoFrame = false;
            for (size_t i = 0; i < callStack_.size(); i++) {
                ParseFrame(ss, strings[i], isPyptoFrame);
            }
            free(strings);
            return ss.str();
        });
    }

private:
    mutable LazyShared<std::string> symbols_;
    std::vector<void*> callStack_;
};

Backtrace GetBacktrace(size_t skipFrames, size_t maxFrames)
{
    return std::make_shared<BacktraceImpl>(BacktraceImpl{skipFrames, maxFrames});
}

std::string Error::DiagnosticWithBacktrace() const
{
    std::stringstream ss;
    ss << msg_ << ", func " << func_ << ", file " << StringUtils::BaseName(file_) << ", line " << line_ << "\n";
    if (backtrace_) {
        ss << backtrace_->Get();
    }
    return ss.str();
}

int Error::operator=(ErrorMessage& msg)
{
    msg_ = msg.Message();
    what_.Reset();
    if (std::uncaught_exceptions() == 0) {
        PYPTO_LOGE_FULL("%s", DiagnosticWithBacktrace().c_str());
        (void)ErrorManager::Instance().GetFirstErrorMessage(msg_);
        throw *this;
    }
    return 0;
}

const char* Error::what() const noexcept
{
    return what_
        .Ensure([this]() -> std::string {
            std::stringstream ss;
            ss << msg_;
            return ss.str();
        })
        .c_str();
}

TerminateHandler::TerminateHandler()
{
    InitializeLogTime();
    struct sigaction sa;
    sa.sa_handler = TerminateHandler::SigAction;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;

    sigaction(SIGSEGV, &sa, &ori[0]);
    sigaction(SIGFPE, &sa, &ori[1]);

    std::set_terminate([] {
        try {
            auto eptr = std::current_exception();
            if (eptr) {
                std::rethrow_exception(eptr);
            }
        } catch (const std::exception& e) {
            PYPTO_LOGE_FULL("Caught exception: %s", e.what());
            ErrorManager::Instance().OutputErrorMessage();
            std::cerr << "Caught exception: '" << e.what() << "'\n";
        }
        (void)fflush(nullptr);
        _Exit(1);
    });
}

static struct TerminateHandler terminateHandler;
} // namespace npu::tile_fwk
