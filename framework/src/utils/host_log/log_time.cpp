/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "host_log/log_time.h"

#include <algorithm>
#include <cstdint>
#include <ctime>
#include <mutex>
#include <sys/time.h>
#include "securec.h"

namespace npu::tile_fwk {
namespace {
// Constant-initialized so explicit startup is safe across translation units.
std::once_flag g_logTimeInit;
time_t g_timeZone = 0;
int32_t g_timeDst = 0;

int32_t IsLeapYear(const int32_t year)
{
    // years can be divisible by 4, but not by 100, or years can be divisible by 400
    return ((((year % 4) == 0) && ((year % 100) != 0)) || ((year % 400) == 0)) ? 1 : 0;
}

/**
 * @brief CalLocalTime: calculate local time
 * @param [in/out]timeInfo: local time struct
 * @param [in]sec: seconds from 1970/1/1
 * @param [in]tzone: current time zone
 * @param [in]dst: daylight time
 * @return: void
 */
void CalLocalTime(struct tm* timeInfo, const time_t sec, const time_t tzone, const int32_t dst)
{
    const time_t oneMin = 60;    // 1m: 60s
    const time_t oneHour = 3600; // 1h: 3600s
    const time_t oneDay = 86400; // 24h: 86400s
    const time_t oneYear = 365;  // 365 days

    time_t realSec = sec - tzone;      // Adjust for timezone
    realSec += oneHour * dst;          // Adjust for daylight time
    time_t days = realSec / oneDay;    // Days passed since epoch
    time_t seconds = realSec % oneDay; // Remaining seconds

    timeInfo->tm_isdst = dst;
    timeInfo->tm_hour = (int32_t)(seconds / oneHour);
    timeInfo->tm_min = (int32_t)((seconds % oneHour) / oneMin);
    timeInfo->tm_sec = (int32_t)((seconds % oneHour) % oneMin);

    // 1/1/1970 was a Thursday, that is, day 4 from the POV of the tm structure * where sunday = 0,
    // so to calculate the day of the week we have to add 4 * and take the modulo by 7.
    timeInfo->tm_wday = (int32_t)((days + 4) % 7); // start from Thursday
    // Calculate the current year
    timeInfo->tm_year = 1970; // start from 1970
    while (1) {
        // Leap years have one day more
        time_t yearDays = oneYear + (time_t)IsLeapYear(timeInfo->tm_year);
        if (yearDays > days) {
            break;
        }
        days -= yearDays;
        timeInfo->tm_year++;
    }
    timeInfo->tm_yday = (int32_t)days; // Number of day of the current year

    int32_t mDays[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}; // days of month: 31, 30, 28/29
    mDays[1] += IsLeapYear(timeInfo->tm_year);                            // leap year

    timeInfo->tm_mon = 0;
    while (days >= mDays[timeInfo->tm_mon]) {
        days -= mDays[timeInfo->tm_mon];
        timeInfo->tm_mon++;
    }

    timeInfo->tm_mon++;                    // Add 1 since our 'month' is zero-based
    timeInfo->tm_mday = (int32_t)days + 1; // Add 1 since our 'days' is zero-based
}

void InitializeTimeZone()
{
    tzset();
    timeval now{};
    std::tm localTime{};
    if (gettimeofday(&now, nullptr) == 0 && localtime_r(&now.tv_sec, &localTime) != nullptr) {
        g_timeZone = timezone;
        g_timeDst = localTime.tm_isdst > 0 ? 1 : 0;
    }
    // Keep the zero-initialized UTC fallback on failure; do not log recursively.
}

bool IsTimeSeparator(const char ch) { return ch == '-' || ch == ' ' || ch == ':' || ch == '.'; }
} // namespace

void InitializeLogTime()
{
    // The once guard is used only by startup, never by timestamp generation.
    std::call_once(g_logTimeInit, InitializeTimeZone);
}

namespace {
struct LogTimeInitializer {
    LogTimeInitializer() { InitializeLogTime(); }
};

// Initialize log time when the utils library is loaded, even if no
// TerminateHandler is constructed.
LogTimeInitializer g_logTimeInitializer;
} // namespace

std::string GetCurrentTime()
{
    timeval now{};
    if (gettimeofday(&now, nullptr) != 0) {
        return {};
    }
    std::tm localTime{};
    CalLocalTime(&localTime, now.tv_sec, g_timeZone, g_timeDst);
    char timeStr[24]{};
    const int ret = snprintf_s(timeStr, sizeof(timeStr), sizeof(timeStr) - 1, "%04d-%02d-%02d %02d:%02d:%02d.%03ld",
                               localTime.tm_year, localTime.tm_mon, localTime.tm_mday, localTime.tm_hour,
                               localTime.tm_min, localTime.tm_sec, static_cast<long>(now.tv_usec / 1000));
    return ret < 0 ? std::string{} : std::string(timeStr);
}

std::string GetCurrentTimeStr()
{
    std::string timeStr = GetCurrentTime();
    timeStr.erase(std::remove_if(timeStr.begin(), timeStr.end(), IsTimeSeparator), timeStr.end());
    return timeStr;
}
} // namespace npu::tile_fwk
