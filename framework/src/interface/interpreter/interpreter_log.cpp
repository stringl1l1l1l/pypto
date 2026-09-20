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
 * \file interpreter_log.cpp
 * \brief Interpreter logging helpers and macros implementation.
 */

#include "interface/interpreter/interpreter_log.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <mutex>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>

#include "securec.h"

namespace npu::tile_fwk::interpreter {
namespace {
constexpr const char* kEnvGlobalLogLevel = "ASCEND_GLOBAL_LOG_LEVEL";
constexpr const char* kEnvModuleLogLevel = "ASCEND_MODULE_LOG_LEVEL";
constexpr const char* kModulePrefix = "PYPTO=";
constexpr int kDefaultGlobalLogThreshold = static_cast<int>(LogLevel::kError);

bool IsEnvEnabled(const char* name)
{
    const char* value = std::getenv(name);
    return value != nullptr && std::strcmp(value, "1") == 0;
}

int ParseEnvIntOrDefault(const char* value, int defaultValue)
{
    if (value == nullptr || value[0] == '\0') {
        return defaultValue;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || (end != nullptr && *end != '\0')) {
        return defaultValue;
    }
    if (parsed < 0 || parsed > 0x5) {
        return defaultValue;
    }
    return static_cast<int>(parsed);
}

int ParseModuleLogLevel(const char* envValue)
{
    if (envValue == nullptr || envValue[0] == '\0') {
        return -1;
    }
    const char* moduleLevel = std::strstr(envValue, kModulePrefix);
    if (moduleLevel == nullptr) {
        return -1;
    }
    return ParseEnvIntOrDefault(moduleLevel + std::strlen(kModulePrefix), -1);
}

int GetGlobalLogThreshold()
{
    int threshold = kDefaultGlobalLogThreshold;
    const char* globalLevel = std::getenv(kEnvGlobalLogLevel);
    if (globalLevel != nullptr && globalLevel[0] != '\0') {
        const int moduleInGlobal = ParseModuleLogLevel(globalLevel);
        if (moduleInGlobal >= 0) {
            threshold = moduleInGlobal;
        } else {
            threshold = ParseEnvIntOrDefault(globalLevel, kDefaultGlobalLogThreshold);
        }
    }
    const int moduleLevel = ParseModuleLogLevel(std::getenv(kEnvModuleLogLevel));
    if (moduleLevel >= 0) {
        threshold = moduleLevel;
    }
    return threshold;
}

int ToCanonicalLevel(LogLevel level)
{
    const int canonical = static_cast<int>(level);
    if (canonical > static_cast<int>(LogLevel::kEvent)) {
        return kDefaultGlobalLogThreshold;
    }
    return canonical;
}

struct LogContext {
    std::mutex mutex;
    std::string logFilePath = "output/interpreter.log";
    FILE* logFile = nullptr;
    bool printToStdout = false;

    LogContext() : printToStdout(IsEnvEnabled("ASCEND_SLOG_PRINT_TO_STDOUT")) {}

    ~LogContext()
    {
        if (logFile != nullptr) {
            fclose(logFile);
            logFile = nullptr;
        }
    }
};

LogContext& GetLogContext()
{
    static LogContext context;
    return context;
}

bool ShouldWriteLevel(LogLevel level) { return ToCanonicalLevel(level) >= GetGlobalLogThreshold(); }

const char* LevelToString(LogLevel level)
{
    switch (level) {
        case LogLevel::kDebug:
            return "DEBUG";
        case LogLevel::kInfo:
            return "INFO";
        case LogLevel::kWarn:
            return "WARN";
        case LogLevel::kEvent:
            return "EVENT";
        case LogLevel::kError:
            return "ERROR";
        default:
            return "UNKNOWN";
    }
}

// 单行日志上限，避免格式串异常时无限扩容；需要更大时可改环境或调大该常量。
constexpr size_t kLogLineMaxBytes = 16U * 1024U * 1024U;

/// 栈缓冲 + 堆扩容缓冲 + 成功时最终 C 字符串指针（指向 stack 或 heap 内部）。
struct LogFormatScratch {
    char* stackBuf = nullptr;
    size_t stackBufSize = 0;
    std::string* heapBuf = nullptr;
    const char* msg = nullptr;
};

/// 将 fmt/args 格式化为以 '\\0' 结尾的连续 C 字符串；成功时 scratch.msg 指向其内部缓冲。
bool FormatLogMessage(const char* fmt, va_list args, LogFormatScratch& scratch)
{
    char* const stackBuf = scratch.stackBuf;
    const size_t stackBufSize = scratch.stackBufSize;
    std::string& heapBuf = *scratch.heapBuf;
    const size_t kInitialBufSize = stackBufSize;
    va_list argsCopy;
    va_copy(argsCopy, args);
    int msgLength = vsnprintf_s(stackBuf, stackBufSize, stackBufSize - 1, fmt, argsCopy);
    va_end(argsCopy);

    if (msgLength >= 0 && static_cast<size_t>(msgLength) < kInitialBufSize) {
        scratch.msg = stackBuf;
        return true;
    }
    if (msgLength >= static_cast<int>(kInitialBufSize)) {
        heapBuf.resize(static_cast<size_t>(msgLength) + 1U);
        va_copy(argsCopy, args);
        msgLength = vsnprintf_s(heapBuf.data(), heapBuf.size(), heapBuf.size() - 1, fmt, argsCopy);
        va_end(argsCopy);
        if (msgLength < 0) {
            return false;
        }
        scratch.msg = heapBuf.c_str();
        return true;
    }
    // libboundscheck 的 vsnprintf_s 在截断时返回 -1，需扩大缓冲后重试直至完整写入。
    size_t cap = std::min(kLogLineMaxBytes, kInitialBufSize * 2U);
    constexpr size_t kMaxHeapFormatAttempts = 64U;
    bool formattedOk = false;
    for (size_t attempt = 0; attempt < kMaxHeapFormatAttempts; ++attempt) {
        heapBuf.resize(cap);
        va_copy(argsCopy, args);
        msgLength = vsnprintf_s(heapBuf.data(), heapBuf.size(), heapBuf.size() - 1, fmt, argsCopy);
        va_end(argsCopy);
        if (msgLength >= 0) {
            scratch.msg = heapBuf.c_str();
            formattedOk = true;
            break;
        }
        if (cap >= kLogLineMaxBytes) {
            return false;
        }
        size_t nextCap = std::min(kLogLineMaxBytes, cap * 2U);
        if (nextCap == cap) {
            return false;
        }
        cap = nextCap;
    }
    return formattedOk;
}

void WritePrefixedLine(FILE* file, LogLevel level, const char* msgPtr)
{
    std::time_t now = std::time(nullptr);
    std::tm localTm{};
    (void)localtime_r(&now, &localTm);
    char timeBuf[32] = {0};
    (void)std::strftime(timeBuf, sizeof(timeBuf), "%Y-%m-%d %H:%M:%S", &localTm);
    const uint64_t threadId = static_cast<uint64_t>(syscall(SYS_gettid));
    const char* levelStr = LevelToString(level);
    fprintf(file, "[%s][%s][tid:%" PRIu64 "] %s\n", timeBuf, levelStr, threadId, msgPtr);
}

__attribute__((format(printf, 3, 0))) void WriteLineImpl(FILE* file, LogLevel level, const char* fmt, va_list args)
{
    constexpr size_t kStackBufSize = 1024U;
    char stackBuf[kStackBufSize];
    std::string heapBuf;
    LogFormatScratch scratch{stackBuf, kStackBufSize, &heapBuf};
    if (!FormatLogMessage(fmt, args, scratch)) {
        return;
    }
    if (file != nullptr) {
        WritePrefixedLine(file, level, scratch.msg);
        fflush(file);
        return;
    }
    auto& context = GetLogContext();
    std::lock_guard<std::mutex> lock(context.mutex);
    if (context.logFile == nullptr) {
        context.logFile = fopen(context.logFilePath.c_str(), "a");
        if (context.logFile == nullptr) {
            return;
        }
    }
    WritePrefixedLine(context.logFile, level, scratch.msg);
    fflush(context.logFile);
    if (context.printToStdout) {
        WritePrefixedLine(stdout, level, scratch.msg);
    }
}
} // namespace

const std::string& LogFilePath() { return GetLogContext().logFilePath; }

void SetLogFilePath(const std::string& path)
{
    auto& context = GetLogContext();
    std::lock_guard<std::mutex> lock(context.mutex);
    if (path.empty() || context.logFilePath == path) {
        return;
    }
    if (context.logFile != nullptr) {
        fclose(context.logFile);
        context.logFile = nullptr;
    }
    context.logFilePath = path;
}

void Log(LogLevel level, const char* fmt, ...)
{
    if (!ShouldWriteLevel(level)) {
        return;
    }

    va_list args;
    va_start(args, fmt);
    WriteLineImpl(nullptr, level, fmt, args);
    va_end(args);
}

void LogToFile(FILE* file, LogLevel level, const char* fmt, ...)
{
    if (file == nullptr || !ShouldWriteLevel(level)) {
        return;
    }

    va_list args;
    va_start(args, fmt);
    WriteLineImpl(file, level, fmt, args);
    va_end(args);
}
} // namespace npu::tile_fwk::interpreter
