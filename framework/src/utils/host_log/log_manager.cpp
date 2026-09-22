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
 * \file log_manager.cpp
 * \brief
 */

#include "host_log/log_manager.h"

#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <sys/syscall.h>
#include <array>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include "securec.h"
#include "file_utils.h"
#include "host_log/log_time.h"

namespace npu::tile_fwk {
namespace {
constexpr const char* kEnvGlobalLogLevel = "ASCEND_GLOBAL_LOG_LEVEL";
constexpr const char* kEnvModuleLogLevel = "ASCEND_MODULE_LOG_LEVEL";
constexpr const char* kEnvGlobalLogEvent = "ASCEND_GLOBAL_EVENT_ENABLE";
constexpr const char* kEnvPrintToStdout = "ASCEND_SLOG_PRINT_TO_STDOUT";
constexpr const char* kEnvHostLogFileNum = "ASCEND_HOST_LOG_FILE_NUM";
constexpr const char* kEnvProcessLogPath = "ASCEND_PROCESS_LOG_PATH";
constexpr const char* kEnvWorkPath = "ASCEND_WORK_PATH";
constexpr const char* kModuleName = "PYPTO";
constexpr const char* kModulePrefix = "PYPTO=";
constexpr const char* kHostLogFilePrefix = "pypto-log-";
constexpr const char* kDevLogFilePrefix = "pypto-simulation-";
constexpr int64_t kMaxLogFileSize = 20 * 1024 * 1024; // 10MB

const std::string kLogLevelNoneStr = "[NONE] ";
const std::array<std::string, static_cast<size_t>(LogLevel::NONE)> kLogLevelStrArray = {"[DEBUG]", "[INFO ]", "[WARN ]",
                                                                                        "[ERROR]", "[EVENT]"};

uint64_t GetTid()
{
    thread_local uint64_t tid = static_cast<uint64_t>(syscall(__NR_gettid));
    return tid;
}

static bool StartWith(const std::string& s, const std::string& prefix)
{
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

int64_t GetPid() { return getpid(); }

const std::string& GetLogLevelStr(const LogLevel logLevel)
{
    return (logLevel >= LogLevel::DEBUG && logLevel < LogLevel::NONE) ?
               kLogLevelStrArray[static_cast<size_t>(logLevel)] :
               kLogLevelNoneStr;
}

bool GetEnvStr(const char* envName, std::string& envValue)
{
    const size_t envValueMaxLen = 1024UL;
    const char* envTemp = std::getenv(envName);
    if (envTemp == nullptr || strnlen(envTemp, envValueMaxLen) >= envValueMaxLen) {
        return false;
    }
    envValue = envTemp;
    return true;
}

int ParseStrToInt(const std::string& str)
{
    try {
        return std::stoi(str);
    } catch (const std::invalid_argument& ia) {
        std::cerr << "Invalid argument: " << ia.what() << '\n';
    } catch (const std::out_of_range& oor) {
        std::cerr << "Out of Range error: " << oor.what() << '\n';
    }
    return -1;
}

int GetModLogLevel(const std::string& str)
{
    if (str.empty()) {
        return -1;
    }
    size_t posLeft = str.find(kModulePrefix);
    if (posLeft == std::string::npos) {
        return -1;
    }
    size_t posRight = str.find(kModulePrefix, posLeft);
    if (posRight == std::string::npos) {
        return ParseStrToInt(str.substr(posLeft + strlen(kModulePrefix)));
    }
    return ParseStrToInt(str.substr(posLeft + strlen(kModulePrefix), posRight - posLeft - strlen(kModulePrefix)));
}

void RemoveRedundantLogFiles(const size_t maxLogFileNum, std::queue<std::string>& logFilesQueue)
{
    if (maxLogFileNum == 0) {
        return;
    }
    while (logFilesQueue.size() > maxLogFileNum) {
        DeleteFile(logFilesQueue.front().c_str());
        logFilesQueue.pop();
    }
}
} // namespace
LogManager& LogManager::Instance()
{
    static LogManager instance;
    return instance;
}

// take care to use file api in file_utils.h, ensue it does not use PYPTO_LOGX
LogManager::LogManager()
{
    std::string envGlobalLogLevel;
    if (GetEnvStr(kEnvGlobalLogLevel, envGlobalLogLevel)) {
        SetLogLevel(static_cast<LogLevel>(ParseStrToInt(envGlobalLogLevel)));
    }

    std::string envModuleLogLevel;
    if (GetEnvStr(kEnvModuleLogLevel, envModuleLogLevel)) {
        SetLogLevel(static_cast<LogLevel>(GetModLogLevel(envModuleLogLevel)));
    }

    std::string envGlobalEvent;
    if (GetEnvStr(kEnvGlobalLogEvent, envGlobalEvent)) {
        enableEvent_ = ParseStrToInt(envGlobalEvent) != 0;
    }

    std::string envPrintToStdout;
    if (GetEnvStr(kEnvPrintToStdout, envPrintToStdout)) {
        enableStdOut_ = ParseStrToInt(envPrintToStdout) != 0;
    }

    if (enableStdOut_) {
        return;
    }

    std::string envHostLogFileNum;
    if (GetEnvStr(kEnvHostLogFileNum, envHostLogFileNum)) {
        int maxLogFileNum = ParseStrToInt(envHostLogFileNum);
        maxLogFileNum_ = maxLogFileNum > 0 ? static_cast<size_t>(maxLogFileNum) : MAX_LOG_FILES_NUM;
    }

    std::string envLogDirPath;
    if (!GetEnvStr(kEnvProcessLogPath, envLogDirPath)) {
        if (!GetEnvStr(kEnvWorkPath, envLogDirPath)) {
            if (!GetEnvStr("HOME", envLogDirPath)) {
                envLogDirPath = ".";
            }
            envLogDirPath += "/ascend/log";
        } else {
            envLogDirPath += "/log";
        }
    }

    auto getLogFiles = [](const std::string& path, const std::string& prefix, const std::string& ext) {
        std::queue<std::string> files;
        for (auto file : GetFiles(path, ext)) {
            if (StartWith(file, prefix)) {
                files.push(RealPath(path + "/" + file));
            }
        }
        return files;
    };

    hostLogDir_ = envLogDirPath + "/debug/plog";
    if (CreateDir(hostLogDir_, true)) {
        hostLogFiles_ = getLogFiles(hostLogDir_, kHostLogFilePrefix, "log");
        RemoveRedundantLogFiles(maxLogFileNum_, hostLogFiles_);
    }

    deviceLogDir_ = envLogDirPath + "/debug/device-" + std::to_string(attr_.deviceId);
    if (CreateDir(deviceLogDir_, true)) {
        devLogFiles_ = getLogFiles(deviceLogDir_, kDevLogFilePrefix, "log");
        RemoveRedundantLogFiles(maxLogFileNum_, devLogFiles_);
    }
}

LogManager::~LogManager()
{
    level_ = LogLevel::ERROR;
    enableStdOut_ = true;
    maxLogFileNum_ = MAX_LOG_FILES_NUM;
    if (hostFileStream_.is_open()) {
        hostFileStream_.flush();
        hostFileStream_.close();
    }
    if (devFileStream_.is_open()) {
        devFileStream_.flush();
        devFileStream_.close();
    }
    std::queue<std::string> tmp_host_files;
    hostLogFiles_.swap(tmp_host_files);
    std::queue<std::string> tmp_dev_files;
    devLogFiles_.swap(tmp_dev_files);
}

void LogManager::SetLogLevel(const LogLevel logLevel)
{
    if (logLevel >= LogLevel::DEBUG && logLevel < LogLevel::NONE) {
        level_ = logLevel;
    }
}

bool LogManager::CheckLevel(const LogLevel logLevel) const
{
    if (logLevel == LogLevel::EVENT) {
        return enableEvent_;
    }
    if (logLevel >= LogLevel::DEBUG && logLevel < LogLevel::NONE) {
        return logLevel >= level_;
    }
    return false;
}

void LogManager::Record(const LogLevel logLevel, const char* fmt, va_list list)
{
    LogMsg logMsg{};
    ConstructMessage(logLevel, fmt, list, logMsg);
    WriteMessage(logMsg);
}

void LogManager::ConstructMessage(const LogLevel logLevel, const char* fmt, va_list list, LogMsg& logMsg)
{
    ConstructMsgHeader(logLevel, logMsg);
    int ret = vsnprintf_truncated_s(logMsg.msg + logMsg.length, MAX_MSG_LENGTH - logMsg.length, fmt, list);
    if (ret < 0) {
        std::cerr << "Construct message failed: " << ret << std::endl;
        return;
    }
    logMsg.length += static_cast<size_t>(ret);

    ConstructMsgTail(logMsg);
}

void LogManager::ConstructMsgHeader(const LogLevel logLevel, LogMsg& logMsg)
{
    int ret = snprintf_s(logMsg.msg, MAX_MSG_LENGTH, MAX_MSG_LENGTH - 1, "%s %s(%lu):%s ",
                         GetLogLevelStr(logLevel).c_str(), kModuleName, GetTid(), GetCurrentTime().c_str());
    if (ret < 0) {
        std::cerr << "Construct log msg header failed: " << ret << std::endl;
        return;
    }
    logMsg.length = static_cast<size_t>(ret);
}

void LogManager::ConstructMsgTail(LogMsg& logMsg)
{
    if (logMsg.msg[logMsg.length - 1] != '\n') {
        if (logMsg.length < MAX_MSG_LENGTH) {
            logMsg.msg[logMsg.length] = '\n';
            logMsg.length++;
        } else {
            logMsg.msg[MAX_MSG_LENGTH - 1] = '\n';
        }
    }
}

void LogManager::WriteMessage(const LogMsg& logMsg)
{
    const std::lock_guard<std::mutex> lockGuard(writeMutex_);
    if (enableStdOut_) {
        WriteToStdOut(logMsg);
    } else {
        WriteToFile(logMsg);
    }
}

void LogManager::WriteToStdOut(const LogMsg& logMsg)
{
    int fd = fileno(stdout);
    if (fd <= 0) {
        std::cerr << "Cannot get fileno of stdout" << std::endl;
    }
    int ret = write(fd, logMsg.msg, logMsg.length);
    if (ret < 0) {
        std::cerr << "Cannot write to stdout: " << ret << std::endl;
    }
}

void LogManager::WriteToFile(const LogMsg& logMsg)
{
    std::ofstream& currentFileStream = GetCurrentFileStream();
    if (!currentFileStream.is_open()) {
        // init log file stream
        CreateAndOpenNewLogFile();
    }
    // write log into file
    if (!currentFileStream.is_open()) {
        std::cerr << "Failed to open file: " << GetLogFilesQueue().back() << std::endl;
        return;
    }
    currentFileStream << logMsg.msg;
    currentFileStream.flush();
    // check log
    CheckAndCloseLogFile(currentFileStream);
}

void LogManager::CreateAndOpenNewLogFile()
{
    std::ostringstream oss;
    const std::string& logFilePrefix = attr_.isDevice ? kDevLogFilePrefix : kHostLogFilePrefix;
    oss << GetLogDir() << "/" << logFilePrefix << GetPid() << "_" << GetCurrentTimeStr() << ".log";
    std::string newLogFileName = oss.str();
    GetCurrentFileStream().open(newLogFileName);
    AddNewLogFile(newLogFileName);
}

void LogManager::AddNewLogFile(const std::string& newLogFileName)
{
    std::queue<std::string>& logFilesQueue = GetLogFilesQueue();
    logFilesQueue.push(newLogFileName);
    while (logFilesQueue.size() > maxLogFileNum_) {
        DeleteFile(logFilesQueue.front().c_str()); // remove file
        logFilesQueue.pop();
    }
}

void LogManager::CheckAndCloseLogFile(std::ofstream& currentFileStream)
{
    std::streamsize fileSize = currentFileStream.tellp();
    if (fileSize < kMaxLogFileSize) {
        return;
    }
    currentFileStream.close();
}
} // namespace npu::tile_fwk
