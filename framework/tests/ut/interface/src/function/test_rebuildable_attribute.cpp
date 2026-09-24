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
 * \file test_rebuildable_attribute.cpp
 * \brief
 */

#include <memory>

#include "gtest/gtest.h"
#include "interface/function/function.h"
#include "interface/function/rebuildable_attribute.h"
#include "interface/program/program.h"

using namespace npu::tile_fwk;

class RebuildableAttributeTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override { Program::GetInstance().Reset(); }

    void TearDown() override {}
};

struct RebuildableNumber : RebuildableAttributeBase {
    virtual void Reset(void* data) override { number = *static_cast<int*>(data); }
    virtual std::string DumpValue() const override { return std::to_string(number); }
    virtual const char* Name() const override { return "Number"; }
    int number;
};

RBUILDABLE_ATTRIBUTE_REGISTER(RebuildableNumber);

TEST_F(RebuildableAttributeTest, TestAttribute)
{
    auto func = std::make_unique<Function>(Program::GetInstance(), "rebuildable_attr_ut", "rebuildable_attr_ut",
                                           nullptr);
    int data = 20;
    auto& mgr = RebuildableAttributeManager::GetInstance();
    mgr.ResetAttr<RebuildableNumber>(func.get(), &data);
    EXPECT_EQ(20, mgr.GetAttr<RebuildableNumber>(func.get())->number);
}

TEST_F(RebuildableAttributeTest, RequiresSimtDefaultsToFalseAndCanBeReset)
{
    auto func = std::make_unique<Function>(Program::GetInstance(), "rebuildable_simt_attr_ut",
                                           "rebuildable_simt_attr_ut", nullptr);
    auto& mgr = RebuildableAttributeManager::GetInstance();
    EXPECT_FALSE(mgr.GetAttr<RebuildableRequiresSimt>(func.get())->Get());

    bool requiresSimt = true;
    mgr.ResetAttr<RebuildableRequiresSimt>(func.get(), &requiresSimt);
    EXPECT_TRUE(mgr.GetAttr<RebuildableRequiresSimt>(func.get())->Get());
}
