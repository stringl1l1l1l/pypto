/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cstring>

#define private public
#define protected public
#include "machine/device/dynamic/aicore_manager.h"

using namespace npu::tile_fwk::dynamic;

namespace {
struct HalTestEnv {
    AicoreHAL hal;
    DeviceArgs devArgs{};
    AiCoreProf* prof{nullptr};
    uint8_t* sharedBuf{nullptr};
    int64_t* regAddrsArr{nullptr};
    uint32_t* regMem{nullptr};
    static constexpr int NUM_CORES = 2;

    HalTestEnv()
    {
        const size_t rawSize = SHARED_BUFFER_SIZE * NUM_CORES;
        const size_t sharedSize = ((rawSize + PAGE_SIZE - 1) / PAGE_SIZE) * PAGE_SIZE;
        sharedBuf = reinterpret_cast<uint8_t*>(aligned_alloc(PAGE_SIZE, sharedSize));
        memset(sharedBuf, 0, sharedSize);

        regMem = new uint32_t[256]();
        regAddrsArr = new int64_t[NUM_CORES]();
        for (int i = 0; i < NUM_CORES; i++) {
            regAddrsArr[i] = reinterpret_cast<int64_t>(&regMem[i * 64]);
        }

        devArgs.sharedBuffer = reinterpret_cast<int64_t>(sharedBuf);
        devArgs.coreRegAddr = reinterpret_cast<int64_t>(regAddrsArr);
        devArgs.nrAic = NUM_CORES;
        devArgs.nrAiv = 0;
        devArgs.archInfo = ArchInfo::DAV_2201;
        devArgs.enableEslModel = false;

        hal.Init(&devArgs, prof);
    }

    ~HalTestEnv()
    {
        if (sharedBuf)
            free(sharedBuf);
        delete[] regMem;
        delete[] regAddrsArr;
    }

    volatile KernelArgs* GetArgs(int coreIdx)
    {
        return reinterpret_cast<volatile KernelArgs*>(sharedBuf + coreIdx * SHARED_BUFFER_SIZE);
    }
};
} // namespace

TEST(AicoreHalTest, GetRegAddrsAndRegNum)
{
    HalTestEnv env;
    EXPECT_EQ(env.hal.GetRegAddrs(), env.regAddrsArr);
    EXPECT_EQ(env.hal.GetregNum(), static_cast<uint32_t>(HalTestEnv::NUM_CORES));
}

TEST(AicoreHalTest, ReadWriteReg32)
{
    HalTestEnv env;
    int coreIdx = 0;
    env.hal.GetPhyIdByBlockId(coreIdx) = 0;

    env.hal.WriteReg32(coreIdx, 0x10, 0xABCD1234);
    EXPECT_EQ(env.hal.ReadReg32(coreIdx, 0x10), 0xABCD1234u);

    env.hal.WriteReg32(coreIdx, 0x20, 0x5678);
    EXPECT_EQ(env.hal.ReadReg32(coreIdx, 0x20), 0x5678u);
}

TEST(AicoreHalTest, ReadReg32_InvalidPhyId)
{
    HalTestEnv env;
    int coreIdx = 0;
    env.hal.GetPhyIdByBlockId(coreIdx) = -1;
    EXPECT_EQ(env.hal.ReadReg32(coreIdx, 0), 0u);
}

TEST(AicoreHalTest, WriteReg32_InvalidPhyId)
{
    HalTestEnv env;
    int coreIdx = 0;
    env.hal.GetPhyIdByBlockId(coreIdx) = -1;
    env.hal.WriteReg32(coreIdx, 0, 0x1234);
}

TEST(AicoreHalTest, ReadPathReg_FastPathDisabled)
{
    HalTestEnv env;
    env.hal.isNeedWriteRegForFastPath_ = false;
    EXPECT_EQ(env.hal.ReadPathReg(0), 0u);
}

TEST(AicoreHalTest, ReadPathReg_FastPathEnabled)
{
    HalTestEnv env;
    int coreIdx = 0;
    env.hal.GetPhyIdByBlockId(coreIdx) = 0;
    env.hal.isNeedWriteRegForFastPath_ = true;

    uint32_t offset = REG_SPR_FAST_PATH_ENABLE / sizeof(uint32_t);
    env.regMem[offset] = static_cast<uint32_t>(REG_SPR_FAST_PATH_OPEN);
    EXPECT_EQ(env.hal.ReadPathReg(coreIdx), static_cast<uint32_t>(REG_SPR_FAST_PATH_OPEN));
}

TEST(AicoreHalTest, NeedsFastPathRegClose)
{
    HalTestEnv env;
    EXPECT_TRUE(env.hal.NeedsFastPathRegClose());
    env.hal.isNeedWriteRegForFastPath_ = false;
    EXPECT_FALSE(env.hal.NeedsFastPathRegClose());
}

TEST(AicoreHalTest, GetAicoreStatusAndLastWord)
{
    HalTestEnv env;
    int coreIdx = 0;
    volatile KernelArgs* args = env.GetArgs(coreIdx);
    args->dfxBuffer[2] = 0xDEADBEEF;
    args->dfxBuffer[3] = 0xCAFEBABE;

    EXPECT_EQ(env.hal.GetAicoreStatus(coreIdx), 0xDEADBEEFu);
    EXPECT_EQ(env.hal.GetAicoreStatusLastWord(coreIdx), 0xCAFEBABEu);
}

TEST(AicoreHalTest, GetHostSimPhyId)
{
    HalTestEnv env;
    env.hal.GetPhyIdByBlockId(0) = 5;
    EXPECT_EQ(env.hal.GetHostSimPhyId(0), 5);

    env.hal.GetPhyIdByBlockId(1) = -1;
    EXPECT_EQ(env.hal.GetHostSimPhyId(1), 1);
}

TEST(AicoreHalTest, IsHostSimMode_CostModel)
{
    HalTestEnv env;
    EXPECT_FALSE(env.hal.IsHostSimMode());
}

TEST(AicoreHalTest, IsSpecialTask)
{
    HalTestEnv env;
    EXPECT_TRUE(env.hal.IsSpecialTask(AICORE_TASK_INIT));
    EXPECT_TRUE(env.hal.IsSpecialTask(AICORE_TASK_STOP));
    EXPECT_TRUE(env.hal.IsSpecialTask(AICORE_FUNC_STOP));
    EXPECT_FALSE(env.hal.IsSpecialTask(42));
}

TEST(AicoreHalTest, TryHandShakeByGm_CostModelReturnsFalse)
{
    HalTestEnv env;
    int coreIdx = 0;
    EXPECT_FALSE(env.hal.TryHandShakeByGm(coreIdx, 0));
}

TEST(AicoreHalTest, SendWaveGoodbye_CostModel)
{
    HalTestEnv env;
    int coreIdx = 0;
    env.hal.args_[coreIdx] = const_cast<KernelArgs*>(env.GetArgs(coreIdx));
    env.hal.SendWaveGoodbye(coreIdx);
}

TEST(AicoreHalTest, InitKernelArgs_CostModel)
{
    HalTestEnv env;
    int coreIdx = 0;
    env.hal.args_[coreIdx] = nullptr;
    env.hal.InitKernelArgs(coreIdx, 0);
}

TEST(AicoreHalTest, ParallelDevTaskCtxVersion_GetSet)
{
    HalTestEnv env;
    int coreIdx = 0;
    env.hal.args_[coreIdx] = const_cast<KernelArgs*>(env.GetArgs(coreIdx));

    env.hal.SetParallelDevTaskCtxVersion(coreIdx, 42);
    EXPECT_EQ(env.hal.ParallelDevTaskCtxVersion(coreIdx), 42u);
    env.hal.SetParallelDevTaskCtxVersion(coreIdx, 100);
    EXPECT_EQ(env.hal.ParallelDevTaskCtxVersion(coreIdx), 100u);
}

TEST(AicoreHalTest, GetSetParallelDevTask)
{
    HalTestEnv env;
    int coreIdx = 0;
    env.hal.args_[coreIdx] = const_cast<KernelArgs*>(env.GetArgs(coreIdx));

    volatile ParallelDevTask* pdt = env.hal.GetParallelDevTask(coreIdx);
    ASSERT_NE(pdt, nullptr);

    env.hal.SetParallelDevTask(pdt, 0, 0x1234, 0x5678);
    env.hal.SetParallelDevTaskSize(pdt, 1, 3);
}

TEST(AicoreHalTest, ResetCoreStopSlot_CostModel)
{
    HalTestEnv env;
    int coreIdx = 0;
    env.hal.args_[coreIdx] = const_cast<KernelArgs*>(env.GetArgs(coreIdx));
    env.hal.ResetCoreStopSlot(coreIdx);
}

TEST(AicoreHalTest, InitCostModelDevTaskData)
{
    HalTestEnv env;
    env.hal.InitCostModelDevTaskData(0, 0xABCD);
}

TEST(AicoreHalTest, GetSharedBuffer)
{
    HalTestEnv env;
    EXPECT_EQ(env.hal.GetSharedBuffer(), reinterpret_cast<int64_t>(env.sharedBuf));
}

TEST(AicoreHalTest, SetMngCoreBlockId)
{
    HalTestEnv env;
    env.hal.SetMngCoreBlockId(0, 2, 2, 4);
    EXPECT_EQ(env.hal.aicStart_, 0);
    EXPECT_EQ(env.hal.aicEnd_, 2);
    EXPECT_EQ(env.hal.aivStart_, 2);
    EXPECT_EQ(env.hal.aivEnd_, 4);
}

TEST(AicoreHalTest, GetRegSprDataMainBase_Default)
{
    HalTestEnv env;
    EXPECT_EQ(env.hal.GetRegSprDataMainBase(), DAV_2201::REG_SPR_DATA_MAIN_BASE);
}

TEST(AicoreHalTest, CostModelSendAndGetTask)
{
    HalTestEnv env;
    int coreIdx = 0;

    env.hal.SetTaskTimeCost([](uint64_t, uint64_t, uint64_t) -> uint64_t { return 100; });

    env.hal.CostModelSendTask(coreIdx, 1);
    env.hal.CostModelSendTask(coreIdx, 2);

    uint64_t result = env.hal.CostModelGetTask(coreIdx);
    EXPECT_NE(result, AICORE_FUNC_STOP | AICORE_FIN_MASK);
}

TEST(AicoreHalTest, CostModelGetTask_Empty)
{
    HalTestEnv env;
    uint64_t result = env.hal.CostModelGetTask(0);
    EXPECT_EQ(result, AICORE_FUNC_STOP | AICORE_FIN_MASK);
}
