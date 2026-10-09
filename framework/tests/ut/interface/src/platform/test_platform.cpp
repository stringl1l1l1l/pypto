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
 * \file test_config_manager.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "utils/file_utils.h"
#define private public
#include "platform/parser/platform_parser.h"
#include "platform/parser/internal_parser.h"

using namespace npu::tile_fwk;
const size_t Num2 = 2UL;
const std::string archInfo = "ArchInfo";
const std::string version = "version";
const std::string socInfo = "SoCInfo";
const std::string aiCoreSpec = "AICoreSpec";
const std::string shortSocVer = "Short_SoC_version";
const std::string aiCoreCnt = "ai_core_cnt";
const std::string cubeCoreCnt = "cube_core_cnt";
const std::string vectorCoreCnt = "vector_core_cnt";
const std::string aiCpuCnt = "ai_cpu_cnt";
const std::string l0aSize = "l0_a_size";
const std::string l0bSize = "l0_b_size";
const std::string l0cSize = "l0_c_size";
const std::string l1Size = "l1_size";
const std::string ubSize = "ub_size";
const std::string aic = "AIC";
const std::string aiv = "AIV";
const std::string
    INI_PATH = "/../../../framework/tests/ut/machine/stubs/compiler/data/platform_config/Ascend910_9572.ini";

class TestPlatform : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestPlatform, TestParser)
{
    const size_t expectAICoreCnt = 28UL;
    const size_t expectCubeCoreCnt = 28UL;
    const size_t expectVectorCoreCnt = 56UL;
    const size_t expectAICpuCnt = 6UL;
    const size_t expectl0aSize = 65536UL;
    const size_t expectl0bSize = 65536UL;
    const size_t expectl0cSize = 262144UL;
    const size_t expectl1Size = 524288UL;
    const size_t expectubSize = 253952UL;

    std::unique_ptr<INIParser> parser = std::make_unique<INIParser>();
    std::string iniPath = RealPath(GetPyptoLibPath() + INI_PATH);
    EXPECT_TRUE(parser->Initialize(iniPath));

    std::string socVersion;
    EXPECT_TRUE(parser->GetStringVal(version, shortSocVer, socVersion));
    EXPECT_EQ(socVersion, "Ascend910_95");

    std::unordered_map<std::string, std::string> ccecVersion;
    EXPECT_TRUE(parser->GetCCECVersion(ccecVersion));
    EXPECT_NE(ccecVersion.find("AIC"), ccecVersion.end());
    EXPECT_EQ(ccecVersion["AIC"], "dav-c310");
    EXPECT_NE(ccecVersion.find("AIV"), ccecVersion.end());
    EXPECT_EQ(ccecVersion["AIV"], "dav-c310");

    size_t coreNum;
    EXPECT_TRUE(parser->GetSizeVal(socInfo, aiCoreCnt, coreNum));
    EXPECT_EQ(coreNum, expectAICoreCnt);
    EXPECT_TRUE(parser->GetSizeVal(socInfo, cubeCoreCnt, coreNum));
    EXPECT_EQ(coreNum, expectCubeCoreCnt);
    EXPECT_TRUE(parser->GetSizeVal(socInfo, vectorCoreCnt, coreNum));
    EXPECT_EQ(coreNum, expectVectorCoreCnt);
    EXPECT_TRUE(parser->GetSizeVal(socInfo, aiCpuCnt, coreNum));
    EXPECT_EQ(coreNum, expectAICpuCnt);

    size_t memoryLimit;
    EXPECT_TRUE(parser->GetSizeVal(aiCoreSpec, l0aSize, memoryLimit));
    EXPECT_EQ(memoryLimit, expectl0aSize);
    EXPECT_TRUE(parser->GetSizeVal(aiCoreSpec, l0bSize, memoryLimit));
    EXPECT_EQ(memoryLimit, expectl0bSize);
    EXPECT_TRUE(parser->GetSizeVal(aiCoreSpec, l0cSize, memoryLimit));
    EXPECT_EQ(memoryLimit, expectl0cSize);
    EXPECT_TRUE(parser->GetSizeVal(aiCoreSpec, l1Size, memoryLimit));
    EXPECT_EQ(memoryLimit, expectl1Size);
    EXPECT_TRUE(parser->GetSizeVal(aiCoreSpec, ubSize, memoryLimit));
    EXPECT_EQ(memoryLimit, expectubSize);
}

TEST_F(TestPlatform, AbnormalTest)
{
    std::unique_ptr<INIParser> parser = std::make_unique<INIParser>();
    EXPECT_FALSE(parser->Initialize(""));

    std::unordered_map<std::string, std::string> ccecVersion;
    EXPECT_FALSE(parser->GetCCECVersion(ccecVersion));

    std::string iniPath = RealPath(GetPyptoLibPath() + INI_PATH);
    EXPECT_TRUE(parser->Initialize(iniPath));

    std::string test;
    EXPECT_FALSE(parser->GetStringVal("none", "", test));
    EXPECT_FALSE(parser->GetStringVal(version, "none_other", test));

    size_t testSize;
    EXPECT_FALSE(parser->GetSizeVal("none", "", testSize));

    std::vector<std::pair<MemoryType, MemoryType>> dataPath;
    InternalParser internalParser1 = InternalParser("");
    EXPECT_TRUE(internalParser1.LoadInternalInfo());
    EXPECT_FALSE(internalParser1.GetDataPath(dataPath));

    InternalParser internalParser2 = InternalParser("2201");
    EXPECT_TRUE(internalParser2.LoadInternalInfo());
    EXPECT_TRUE(internalParser2.GetDataPath(dataPath));
}

TEST_F(TestPlatform, A5Stub)
{
    std::vector<std::pair<MemoryType, MemoryType>> dataPath;
    InternalParser parser = InternalParser("3510");
    EXPECT_TRUE(parser.LoadInternalInfo());
    EXPECT_TRUE(parser.GetDataPath(dataPath));
    Platform::Instance().GetDie().SetMemoryPath(dataPath);

    std::vector<MemoryType> path;
    Platform::Instance().GetDie().FindNearestPath(MemoryType::MEM_L0C, MemoryType::MEM_UB, path);
    EXPECT_EQ(path.size(), Num2);
    path.clear();
    Platform::Instance().GetDie().FindNearestPath(MemoryType::MEM_L1, MemoryType::MEM_UB, path);
    EXPECT_EQ(path.size(), Num2);
    path.clear();
    Platform::Instance().GetDie().FindNearestPath(MemoryType::MEM_UB, MemoryType::MEM_L1, path);
    EXPECT_EQ(path.size(), Num2);
    path.clear();
}

TEST(PlatformArchTest, GetMemoryLimitForArchRejectsUnconfiguredArchAndSpace)
{
    EXPECT_EQ(GetMemoryLimitForArch(NPUArch::DAV_2201, kMemorySpaceVec), 0u);
    EXPECT_EQ(GetMemoryLimitForArch(NPUArch::DAV_3510, "UnmappedSpace"), 0u);
}
