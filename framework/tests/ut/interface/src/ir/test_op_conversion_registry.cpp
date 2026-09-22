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
 * \file test_op_conversion_registry.cpp
 * \brief Coverage tests for OpConversionRegistry (op_conversion_registry.cpp)
 */

#include "gtest/gtest.h"

#include <any>
#include <string>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/scalar_expr.h"
#include "ir/span.h"
#include "ir/transforms/op_conversion_registry.h"

namespace pypto {
namespace ir {

static Span Sp() { return Span("test", 1, 1); }

class OpConversionRegistryTest : public testing::Test {
protected:
    OpConversionRegistry& Registry() { return OpConversionRegistry::GetInstance(); }

    ExprPtr Int(int64_t value) { return std::make_shared<ConstInt>(value, DataType::INT32, Sp()); }
};

TEST_F(OpConversionRegistryTest, TestLookupUnknownReturnsNull)
{
    const std::string unknown = "nonexistent.op";
    EXPECT_FALSE(Registry().HasConversion(unknown));
    EXPECT_EQ(Registry().Lookup(unknown), nullptr);
}

TEST_F(OpConversionRegistryTest, TestRegisterCustomUsesLatestConverter)
{
    int call_count = 0;
    Registry().RegisterCustom(
        "test.custom_op",
        [&call_count, this](const std::vector<ExprPtr>&, const std::vector<std::pair<std::string, std::any>>&,
                            const Span&) -> ConversionResult {
            ++call_count;
            return ConversionResult{Int(1)};
        });
    Registry().RegisterCustom(
        "test.custom_op",
        [&call_count, this](const std::vector<ExprPtr>&, const std::vector<std::pair<std::string, std::any>>&,
                            const Span&) -> ConversionResult {
            ++call_count;
            return ConversionResult{Int(2)};
        });

    auto* func = Registry().Lookup("test.custom_op");
    ASSERT_NE(func, nullptr);
    auto result = (*func)({}, {}, Sp());
    auto value = std::dynamic_pointer_cast<const ConstInt>(result.result);
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(value->value_, 2);
    EXPECT_EQ(call_count, 1);
}

} // namespace ir
} // namespace pypto
