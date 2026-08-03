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
 * \file test_sync_ops.cpp
 * \brief Coverage tests for sync_ops/sync.cpp — system.dcci type deduction
 */

#include "gtest/gtest.h"

#include <any>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "core/error.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/op_attr_types.h"
#include "ir/op_registry.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"
#include "test_op_helpers.h"
#include "tilefwk/error.h"

namespace pypto {
namespace ir {

using namespace test_helpers;

// ============================================================================
// system.mutex_lock_dyn / system.mutex_unlock_dyn
// ============================================================================

TEST(SyncOpsMutexTest, DynamicMutexAcceptsMutexIdOwnerIndices)
{
    auto& reg = OpRegistry::GetInstance();
    auto id0 = MakeScalarVar("id0", DataType::INDEX);
    auto id1 = MakeScalarVar("id1", DataType::INDEX);
    const std::vector<int> mutex_id_owner_indices = {0, 0};
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"pipe", 5},
        {"mode", 0},
        {"mutex_ids", std::vector<int>{0, 1, 2, 3}},
        {"mutex_id_owner_indices", mutex_id_owner_indices},
    };

    for (const char* op_name : {"system.mutex_lock_dyn", "system.mutex_unlock_dyn"}) {
        auto call = reg.Create(op_name, {id0, id1}, kwargs, Sp());
        ASSERT_NE(call, nullptr);
        EXPECT_NE(As<UnknownType>(call->GetType()), nullptr);
        EXPECT_EQ(call->GetKwarg<std::vector<int>>("mutex_id_owner_indices"), mutex_id_owner_indices);
    }
}

// ============================================================================
// system.dcci
// ============================================================================

class SyncOpsDcciTest : public testing::Test {};

TEST_F(SyncOpsDcciTest, Dcci_TensorTarget_TupleOffset_ReturnsUnknown)
{
    auto& reg = OpRegistry::GetInstance();
    auto tensor = MakeTensorVar("gm", {32, 64}, DataType::FP16);
    auto offset = MakeOffsetsTuple({0, 16});
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"cache_line", static_cast<int>(CacheLine::SINGLE_CACHE_LINE)}, {"dst", static_cast<int>(DcciDst::AUTO)}};
    auto call = reg.Create("system.dcci", {tensor, offset}, kwargs, Sp());
    ASSERT_NE(call, nullptr);
    EXPECT_NE(As<UnknownType>(call->GetType()), nullptr);
}

TEST_F(SyncOpsDcciTest, Dcci_TileTarget_ScalarOffset_ReturnsUnknown)
{
    auto& reg = OpRegistry::GetInstance();
    auto tile = MakeTileVar("ub", {16, 32}, DataType::FP16);
    auto offset = MakeScalarVar("off", DataType::INDEX);
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"cache_line", static_cast<int>(CacheLine::SINGLE_CACHE_LINE)},
        {"dst", static_cast<int>(DcciDst::CACHELINE_UB)}};
    auto call = reg.Create("system.dcci", {tile, offset}, kwargs, Sp());
    ASSERT_NE(call, nullptr);
    EXPECT_NE(As<UnknownType>(call->GetType()), nullptr);
}

TEST_F(SyncOpsDcciTest, Dcci_TileTarget_TupleOffset_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    auto tile = MakeTileVar("ub", {16, 32}, DataType::FP16);
    auto offset = MakeOffsetsTuple({0, 16});
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"cache_line", static_cast<int>(CacheLine::SINGLE_CACHE_LINE)}, {"dst", static_cast<int>(DcciDst::AUTO)}};
    EXPECT_THROW((void)reg.Create("system.dcci", {tile, offset}, kwargs, Sp()), npu::tile_fwk::Error);
}

TEST_F(SyncOpsDcciTest, Dcci_NonTensorNonTileTarget_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    auto scalar = MakeScalarVar("x", DataType::INT32);
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"cache_line", static_cast<int>(CacheLine::SINGLE_CACHE_LINE)}, {"dst", static_cast<int>(DcciDst::AUTO)}};
    EXPECT_THROW((void)reg.Create("system.dcci", {scalar}, kwargs, Sp()), npu::tile_fwk::Error);
}

TEST_F(SyncOpsDcciTest, Dcci_TensorTarget_NonIntOffset_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    auto tensor = MakeTensorVar("gm", {64, 128}, DataType::FP16);
    auto bad_offset = MakeScalarVar("off", DataType::FP32);
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"cache_line", static_cast<int>(CacheLine::SINGLE_CACHE_LINE)}, {"dst", static_cast<int>(DcciDst::AUTO)}};
    EXPECT_THROW((void)reg.Create("system.dcci", {tensor, bad_offset}, kwargs, Sp()), npu::tile_fwk::Error);
}

} // namespace ir
} // namespace pypto
