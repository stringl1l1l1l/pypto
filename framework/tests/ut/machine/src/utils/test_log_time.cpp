/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <atomic>
#include <ctime>
#include <regex>
#include <thread>
#include <vector>
#include "utils/host_log/log_time.h"

namespace npu::tile_fwk {
namespace {
std::string FormatReferenceTime(const time_t seconds)
{
    std::tm localTime{};
    if (localtime_r(&seconds, &localTime) == nullptr) {
        return {};
    }
    char buffer[32]{};
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &localTime);
    return buffer;
}
std::string FormatReferenceFilenameTime(const time_t seconds)
{
    std::tm localTime{};
    if (localtime_r(&seconds, &localTime) == nullptr) {
        return {};
    }
    char buffer[32]{};
    std::strftime(buffer, sizeof(buffer), "%Y%m%d%H%M%S", &localTime);
    return buffer;
}
void CheckConcurrentTimestamps(std::atomic<bool>* valid)
{
    for (int j = 0; j < 1000; ++j) {
        const std::string normal = GetCurrentTime();
        const std::string compact = GetCurrentTimeStr();
        if (normal.size() != 23 || compact.size() != 17 ||
            compact.find_first_not_of("0123456789") != std::string::npos) {
            *valid = false;
        }
    }
}
} // namespace

TEST(LogTimeTest, PreservesLocalTimestampFormat)
{
    const time_t before = std::time(nullptr);
    const std::string actual = GetCurrentTime();
    const time_t after = std::time(nullptr);
    ASSERT_TRUE(std::regex_match(actual, std::regex(R"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})")));
    const std::string date = actual.substr(0, 19);
    EXPECT_TRUE(date == FormatReferenceTime(before) || date == FormatReferenceTime(after));
}

TEST(LogTimeTest, PreservesFilenameTimestampFormat)
{
    const time_t before = std::time(nullptr);
    const std::string actual = GetCurrentTimeStr();
    const time_t after = std::time(nullptr);
    ASSERT_TRUE(std::regex_match(actual, std::regex(R"(\d{17})")));
    const std::string date = actual.substr(0, 14);
    EXPECT_TRUE(date == FormatReferenceFilenameTime(before) || date == FormatReferenceFilenameTime(after));
}

TEST(LogTimeTest, ConcurrentTimestampGeneration)
{
    std::atomic<bool> valid{true};
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i) {
        threads.emplace_back(CheckConcurrentTimestamps, &valid);
    }
    for (auto& thread : threads) {
        thread.join();
    }
    EXPECT_TRUE(valid.load());
}
} // namespace npu::tile_fwk
