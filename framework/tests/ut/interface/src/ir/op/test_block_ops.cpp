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
 * \file test_block_ops_type_deduction.cpp
 * \brief Coverage tests for block_ops type deduction via OpRegistry::Create
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
// memory.cpp: get_block_idx, get_block_num, get_subblock_idx
// ============================================================================

class BlockOpsMemoryTest : public testing::Test {};

TEST_F(BlockOpsMemoryTest, GetBlockIdx_NoArgs_ReturnsInt64Scalar)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("get_block_idx", {}, Sp());
    ASSERT_NE(call, nullptr);
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::INT64);
}

TEST_F(BlockOpsMemoryTest, GetBlockNum_NoArgs_ReturnsInt64Scalar)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("get_block_num", {}, Sp());
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::INT64);
}

TEST_F(BlockOpsMemoryTest, GetSubblockIdx_NoArgs_ReturnsInt64Scalar)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("get_subblock_idx", {}, Sp());
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::INT64);
}

TEST_F(BlockOpsMemoryTest, GetSubblockNum_NoArgs_ReturnsInt64Scalar)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("get_subblock_num", {}, Sp());
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::INT64);
}

TEST_F(BlockOpsMemoryTest, GetBlockIdx_WithArgs_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("get_block_idx", {MakeScalarVar("x", DataType::INT32)}, Sp()), npu::tile_fwk::Error);
}

TEST_F(BlockOpsMemoryTest, GetBlockNum_WithArgs_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("get_block_num", {MakeScalarVar("x", DataType::INT32)}, Sp()), npu::tile_fwk::Error);
}

TEST_F(BlockOpsMemoryTest, GetSubblockNum_WithArgs_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("get_subblock_num", {MakeScalarVar("x", DataType::INT32)}, Sp()),
                 npu::tile_fwk::Error);
}

// ============================================================================
// memory.cpp: block.make_tile
// ============================================================================

TEST_F(BlockOpsMemoryTest, MakeTile_BasicShape_ReturnsTileType)
{
    auto& reg = OpRegistry::GetInstance();
    auto dim1 = std::make_shared<ConstInt>(16, DataType::INT64, Sp());
    auto dim2 = std::make_shared<ConstInt>(32, DataType::INT64, Sp());
    auto shape_tuple = std::make_shared<MakeTuple>(std::vector<ExprPtr>{dim1, dim2}, Sp());
    auto valid_tuple = std::make_shared<MakeTuple>(std::vector<ExprPtr>{dim1, dim2}, Sp());

    std::vector<std::pair<std::string, std::any>> kwargs = {{"dtype", DataType::FP16}};
    auto call = reg.Create("block.make_tile", {shape_tuple, valid_tuple}, kwargs, Sp());
    ASSERT_NE(call, nullptr);
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
    EXPECT_EQ(rt->shape_.size(), 2u);
}

TEST_F(BlockOpsMemoryTest, MakeTile_WithMemRef_ReturnsTileWithMemRef)
{
    auto& reg = OpRegistry::GetInstance();
    auto dim1 = std::make_shared<ConstInt>(8, DataType::INT64, Sp());
    auto dim2 = std::make_shared<ConstInt>(8, DataType::INT64, Sp());
    auto shape_tuple = std::make_shared<MakeTuple>(std::vector<ExprPtr>{dim1, dim2}, Sp());
    auto valid_tuple = std::make_shared<MakeTuple>(std::vector<ExprPtr>{dim1, dim2}, Sp());

    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"dtype", DataType::INT32},
        {"memref_addr", int(0)},
        {"memref_size", int(256)},
        {"memref_id", int(1)},
    };
    auto call = reg.Create("block.make_tile", {shape_tuple, valid_tuple}, kwargs, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_TRUE(rt->memref_.has_value());
}

TEST_F(BlockOpsMemoryTest, MakeTile_EmptyShape_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    auto shape_tuple = std::make_shared<MakeTuple>(std::vector<ExprPtr>{}, Sp());
    auto valid_tuple = std::make_shared<MakeTuple>(std::vector<ExprPtr>{}, Sp());
    std::vector<std::pair<std::string, std::any>> kwargs = {{"dtype", DataType::FP16}};

    EXPECT_THROW((void)reg.Create("block.make_tile", {shape_tuple, valid_tuple}, kwargs, Sp()), npu::tile_fwk::Error);
}

TEST_F(BlockOpsMemoryTest, MakeTile_RankOneShape_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    auto dim = std::make_shared<ConstInt>(8, DataType::INT64, Sp());
    auto shape_tuple = std::make_shared<MakeTuple>(std::vector<ExprPtr>{dim}, Sp());
    auto valid_tuple = std::make_shared<MakeTuple>(std::vector<ExprPtr>{dim}, Sp());
    std::vector<std::pair<std::string, std::any>> kwargs = {{"dtype", DataType::FP16}};

    EXPECT_THROW((void)reg.Create("block.make_tile", {shape_tuple, valid_tuple}, kwargs, Sp()), npu::tile_fwk::Error);
}

TEST_F(BlockOpsMemoryTest, MakeTile_RankThreeShape_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    auto dim1 = std::make_shared<ConstInt>(8, DataType::INT64, Sp());
    auto dim2 = std::make_shared<ConstInt>(8, DataType::INT64, Sp());
    auto dim3 = std::make_shared<ConstInt>(8, DataType::INT64, Sp());
    auto shape_tuple = std::make_shared<MakeTuple>(std::vector<ExprPtr>{dim1, dim2, dim3}, Sp());
    auto valid_tuple = std::make_shared<MakeTuple>(std::vector<ExprPtr>{dim1, dim2, dim3}, Sp());
    std::vector<std::pair<std::string, std::any>> kwargs = {{"dtype", DataType::FP16}};

    EXPECT_THROW((void)reg.Create("block.make_tile", {shape_tuple, valid_tuple}, kwargs, Sp()), npu::tile_fwk::Error);
}

TEST_F(BlockOpsMemoryTest, MakeTile_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"dtype", DataType::FP32}};
    EXPECT_THROW((void)reg.Create("block.make_tile", {}, kwargs, Sp()), npu::tile_fwk::Error);
}

TEST_F(BlockOpsMemoryTest, MakeTile_WithLayoutKwargs_ReturnsTileType)
{
    auto& reg = OpRegistry::GetInstance();
    auto dim1 = std::make_shared<ConstInt>(int64_t(16), DataType::INT64, Sp());
    auto dim2 = std::make_shared<ConstInt>(int64_t(32), DataType::INT64, Sp());
    auto shape_tuple = std::make_shared<MakeTuple>(std::vector<ExprPtr>{dim1, dim2}, Sp());
    auto valid_tuple = std::make_shared<MakeTuple>(std::vector<ExprPtr>{dim1, dim2}, Sp());

    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"dtype", DataType::FP16}, {"blayout", int(1)}, {"slayout", int(0)},
        {"fractal", int(256)},     {"pad", int(1)},     {"compact", int(0)},
    };
    auto call = reg.Create("block.make_tile", {shape_tuple, valid_tuple}, kwargs, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

// ============================================================================
// memory.cpp: block.tile_valid_shape
// ============================================================================
TEST_F(BlockOpsMemoryTest, TileValidShape_ReturnsInt64)
{
    auto& reg = OpRegistry::GetInstance();
    auto tile = MakeTileVar("tile", {16, 32}, DataType::FP16);
    auto call = reg.Create("block.tile_valid_shape", {tile}, std::vector<std::pair<std::string, std::any>>{{"axis", 1}},
                           Sp());
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::INT64);
}

TEST_F(BlockOpsMemoryTest, TileValidShape_RejectsInvalidAxis)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("block.tile_valid_shape", {MakeTileVar("tile", {16, 32}, DataType::FP16)},
                                  std::vector<std::pair<std::string, std::any>>{{"axis", 2}}, Sp()),
                 npu::tile_fwk::Error);
}

TEST_F(BlockOpsMemoryTest, TileValidShape_RejectsNonTile)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("block.tile_valid_shape", {MakeScalarVar("value", DataType::INT32)},
                                  std::vector<std::pair<std::string, std::any>>{{"axis", 0}}, Sp()),
                 npu::tile_fwk::Error);
}

// ============================================================================
// memory.cpp: block.getval, block.setval
// ============================================================================

TEST_F(BlockOpsMemoryTest, GetVal_TileAndIndex_ReturnsScalarOfTileDtype)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.getval",
                           {MakeTileVar("t", {64}, DataType::FP16), MakeScalarVar("i", DataType::INT32)}, Sp());
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsMemoryTest, GetVal_IntegerContainer_ReturnsInt64UnlessUint64)
{
    auto& reg = OpRegistry::GetInstance();
    const std::vector<std::pair<DataType, DataType>> cases = {
        {DataType::INT8, DataType::INT64},   {DataType::INT16, DataType::INT64},   {DataType::INT32, DataType::INT64},
        {DataType::INT64, DataType::INT64},  {DataType::UINT8, DataType::INT64},   {DataType::UINT16, DataType::INT64},
        {DataType::UINT32, DataType::INT64}, {DataType::UINT64, DataType::UINT64},
    };
    for (const auto& [source_dtype, expected_dtype] : cases) {
        SCOPED_TRACE(source_dtype.ToString());
        auto offset = MakeScalarVar("i", DataType::INT64);
        auto tile_call = reg.Create("block.getval", {MakeTileVar("tile", {64}, source_dtype), offset}, Sp());
        auto tensor_call = reg.Create("block.getval", {MakeTensorVar("tensor", {64}, source_dtype), offset}, Sp());

        EXPECT_EQ(As<ScalarType>(tile_call->GetType())->dtype_, expected_dtype);
        EXPECT_EQ(As<ScalarType>(tensor_call->GetType())->dtype_, expected_dtype);
    }
}

TEST_F(BlockOpsMemoryTest, GetVal_SimtPreservesContainerDtype)
{
    auto& reg = OpRegistry::GetInstance();
    const std::vector<DataType> cases = {
        DataType::INT8,   DataType::INT16,  DataType::INT32, DataType::INT64, DataType::UINT8, DataType::UINT16,
        DataType::UINT32, DataType::UINT64, DataType::FP16,  DataType::BF16,  DataType::FP32,  DataType::BOOL,
    };
    for (const auto& source_dtype : cases) {
        SCOPED_TRACE(source_dtype.ToString());
        auto offset = MakeScalarVar("i", DataType::INT64);
        const std::vector<std::pair<std::string, std::any>> kwargs = {{"preserve_dtype", true}};
        auto tile_call = reg.Create("block.getval", {MakeTileVar("tile", {64}, source_dtype), offset}, kwargs, Sp());
        auto tensor_call = reg.Create("block.getval", {MakeTensorVar("tensor", {64}, source_dtype), offset}, kwargs,
                                      Sp());

        EXPECT_EQ(As<ScalarType>(tile_call->GetType())->dtype_, source_dtype);
        EXPECT_EQ(As<ScalarType>(tensor_call->GetType())->dtype_, source_dtype);
    }
}

TEST_F(BlockOpsMemoryTest, GetVal_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("block.getval", {MakeTileVar("t", {64}, DataType::FP32)}, Sp()),
                 npu::tile_fwk::Error);
}

TEST_F(BlockOpsMemoryTest, GetVal_NonTileFirst_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("block.getval",
                                  {MakeScalarVar("a", DataType::FP32), MakeScalarVar("b", DataType::INT32)}, Sp()),
                 npu::tile_fwk::Error);
}

TEST_F(BlockOpsMemoryTest, GetVal_NonIntIndex_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("block.getval",
                                  {MakeTileVar("t", {64}, DataType::FP16), MakeScalarVar("i", DataType::FP32)}, Sp()),
                 npu::tile_fwk::Error);
}

TEST_F(BlockOpsMemoryTest, SetVal_TileIndexValue_ReturnsTileType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.setval",
                           {MakeTileVar("t", {128}, DataType::FP32), MakeScalarVar("i", DataType::INT64),
                            MakeScalarVar("v", DataType::FP32)},
                           Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

TEST_F(BlockOpsMemoryTest, SetVal_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("block.setval",
                                  {MakeTileVar("t", {128}, DataType::FP32), MakeScalarVar("i", DataType::INT64)}, Sp()),
                 npu::tile_fwk::Error);
}

TEST_F(BlockOpsMemoryTest, SetVal_NonTileFirst_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("block.setval",
                                  {MakeScalarVar("s", DataType::FP32), MakeScalarVar("i", DataType::INT32),
                                   MakeScalarVar("v", DataType::FP32)},
                                  Sp()),
                 npu::tile_fwk::Error);
}

TEST_F(BlockOpsMemoryTest, SetVal_NonIntIndex_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("block.setval",
                                  {MakeTileVar("t", {128}, DataType::FP32), MakeScalarVar("i", DataType::FP32),
                                   MakeScalarVar("v", DataType::FP32)},
                                  Sp()),
                 npu::tile_fwk::Error);
}

TEST_F(BlockOpsMemoryTest, SetVal_NonScalarValue_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("block.setval",
                                  {MakeTileVar("t", {128}, DataType::FP32), MakeScalarVar("i", DataType::INT64),
                                   MakeTileVar("v", {128}, DataType::FP32)},
                                  Sp()),
                 npu::tile_fwk::Error);
}

// ============================================================================
// memory.cpp: block.subview
// ============================================================================

TEST_F(BlockOpsMemoryTest, Subview_TilePreservesPhysicalTypeAndMemRef)
{
    auto& reg = OpRegistry::GetInstance();
    auto tile = MakeTileVarWithMemRef("tile", {16, 32}, DataType::FP16, MemorySpace::Vec, 0x100, 1024);
    auto source_type = As<TileType>(tile->GetType());

    auto call = reg.Create("block.subview", {tile, MakeScalarVar("offset", DataType::INT64), MakeIntTuple({8, 12})},
                           Sp());

    auto result_type = As<TileType>(call->GetType());
    ASSERT_NE(result_type, nullptr);
    EXPECT_EQ(result_type.get(), source_type.get());
    EXPECT_EQ(result_type->dtype_, source_type->dtype_);
    EXPECT_EQ(result_type->shape_, source_type->shape_);
    EXPECT_EQ(result_type->memref_, source_type->memref_);
}

TEST_F(BlockOpsMemoryTest, Subview_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create(
                     "block.subview",
                     {MakeTileVar("tile", {16, 32}, DataType::FP16), MakeScalarVar("offset", DataType::INT64)}, Sp()),
                 npu::tile_fwk::Error);
}

TEST_F(BlockOpsMemoryTest, Subview_NonTileContainer_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("block.subview",
                                  {MakeScalarVar("container", DataType::FP16), MakeScalarVar("offset", DataType::INT64),
                                   MakeIntTuple({8, 12})},
                                  Sp()),
                 npu::tile_fwk::Error);
}

// ============================================================================
// struct_ops.cpp: struct.create, struct.set
// ============================================================================

class BlockOpsStructTest : public testing::Test {};

TEST_F(BlockOpsStructTest, StructCreate_ReturnsTupleType)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"name", std::string("BufferState")},
        {"fields", std::vector<std::string>{"cursor", "limit"}},
    };
    auto call = reg.Create("struct.create",
                           {MakeScalarVar("cursor", DataType::INT64), MakeScalarVar("limit", DataType::INT32)}, kwargs,
                           Sp());
    auto rt = As<TupleType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    ASSERT_EQ(rt->types_.size(), 2u);
    auto cursor_type = As<ScalarType>(rt->types_[0]);
    ASSERT_NE(cursor_type, nullptr);
    EXPECT_EQ(cursor_type->dtype_, DataType::INT64);
    auto limit_type = As<ScalarType>(rt->types_[1]);
    ASSERT_NE(limit_type, nullptr);
    EXPECT_EQ(limit_type->dtype_, DataType::INT32);
}

TEST_F(BlockOpsStructTest, StructCreate_FieldCountMismatch_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"name", std::string("BufferState")},
        {"fields", std::vector<std::string>{"cursor", "limit"}},
    };
    EXPECT_THROW((void)reg.Create("struct.create", {MakeScalarVar("cursor", DataType::INT64)}, kwargs, Sp()),
                 npu::tile_fwk::Error);
}

TEST_F(BlockOpsStructTest, StructSet_ReturnsTupleType)
{
    auto& reg = OpRegistry::GetInstance();
    auto struct_type = std::make_shared<TupleType>(std::vector<TypePtr>{
        std::make_shared<ScalarType>(DataType::INT64),
        std::make_shared<ScalarType>(DataType::INT32),
    });
    auto base = std::make_shared<Var>("state", struct_type, Sp());
    std::vector<std::pair<std::string, std::any>> kwargs = {{"field", std::string("cursor")}};
    auto call = reg.Create("struct.set", {base, MakeScalarVar("cursor", DataType::INT64)}, kwargs, Sp());
    auto rt = As<TupleType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->types_.size(), 2u);
}

TEST_F(BlockOpsStructTest, StructSet_MissingFieldKwarg_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    auto struct_type = std::make_shared<TupleType>(std::vector<TypePtr>{
        std::make_shared<ScalarType>(DataType::INT64),
    });
    auto base = std::make_shared<Var>("state", struct_type, Sp());
    EXPECT_THROW((void)reg.Create("struct.set", {base, MakeScalarVar("cursor", DataType::INT64)}, Sp()),
                 npu::tile_fwk::Error);
}

TEST_F(BlockOpsStructTest, StructSet_NonTupleBase_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"field", std::string("cursor")}};
    EXPECT_THROW((void)reg.Create("struct.set",
                                  {MakeScalarVar("state", DataType::INT64), MakeScalarVar("cursor", DataType::INT64)},
                                  kwargs, Sp()),
                 npu::tile_fwk::Error);
}

TEST_F(BlockOpsStructTest, StructSet_WithIndex_ReturnsTupleType)
{
    auto& reg = OpRegistry::GetInstance();
    auto array_type = std::make_shared<TupleType>(std::vector<TypePtr>{
        std::make_shared<ScalarType>(DataType::INT64),
        std::make_shared<ScalarType>(DataType::INT64),
        std::make_shared<ScalarType>(DataType::INT64),
        std::make_shared<ScalarType>(DataType::INT64),
    });
    auto struct_type = std::make_shared<TupleType>(std::vector<TypePtr>{
        std::make_shared<ScalarType>(DataType::INT64),
        array_type,
    });
    auto base = std::make_shared<Var>("state", struct_type, Sp());
    auto index = std::make_shared<Var>("idx", std::make_shared<ScalarType>(DataType::INDEX), Sp());
    std::vector<std::pair<std::string, std::any>> kwargs = {{"field", std::string("arr")}};
    auto call = reg.Create("struct.set", {base, index, MakeScalarVar("val", DataType::INT64)}, kwargs, Sp());
    auto rt = As<TupleType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->types_.size(), 2u);
}

TEST_F(BlockOpsStructTest, StructSet_WithIndex_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    auto struct_type = std::make_shared<TupleType>(std::vector<TypePtr>{
        std::make_shared<ScalarType>(DataType::INT64),
    });
    auto base = std::make_shared<Var>("state", struct_type, Sp());
    std::vector<std::pair<std::string, std::any>> kwargs = {{"field", std::string("f")}};
    EXPECT_THROW((void)reg.Create("struct.set",
                                  {base, MakeScalarVar("a", DataType::INT64), MakeScalarVar("b", DataType::INT64),
                                   MakeScalarVar("c", DataType::INT64)},
                                  kwargs, Sp()),
                 npu::tile_fwk::Error);
}

TEST_F(BlockOpsStructTest, StructSet_WithIndex_NullIndex_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    auto array_type = std::make_shared<TupleType>(std::vector<TypePtr>{
        std::make_shared<ScalarType>(DataType::INT64),
        std::make_shared<ScalarType>(DataType::INT64),
    });
    auto struct_type = std::make_shared<TupleType>(std::vector<TypePtr>{
        std::make_shared<ScalarType>(DataType::INT64),
        array_type,
    });
    auto base = std::make_shared<Var>("state", struct_type, Sp());
    std::vector<std::pair<std::string, std::any>> kwargs = {{"field", std::string("arr")}};
    EXPECT_THROW((void)reg.Create("struct.set", {base, nullptr, MakeScalarVar("val", DataType::INT64)}, kwargs, Sp()),
                 npu::tile_fwk::Error);
}

// ============================================================================
// out_elementwise.cpp: Tile x Tile binary ops
// ============================================================================

class BlockOpsOutElemwiseTest : public testing::Test {};

TEST_F(BlockOpsOutElemwiseTest, BlockAdd_ThreeTiles_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto lhs = MakeTileVar("lhs", {16, 32}, DataType::FP16);
    auto rhs = MakeTileVar("rhs", {16, 32}, DataType::FP16);
    auto out = MakeTileVar("out", {16, 32}, DataType::FP16);
    auto call = reg.Create("block.add", {out, lhs, rhs}, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutElemwiseTest, BlockAdd_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create(
                     "block.add",
                     {MakeTileVar("l", {16, 32}, DataType::FP16), MakeTileVar("r", {16, 32}, DataType::FP16)}, Sp()),
                 npu::tile_fwk::Error);
}

TEST_F(BlockOpsOutElemwiseTest, BlockAdd_NonTileArg_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("block.add",
                                  {MakeScalarVar("s", DataType::FP32), MakeTileVar("r", {16, 32}, DataType::FP32),
                                   MakeTileVar("o", {16, 32}, DataType::FP32)},
                                  Sp()),
                 npu::tile_fwk::Error);
}

// Representative coverage for REGISTER_BLOCK_OUT_BINARY_TILE macro-registered ops.
// All share the same DeduceBlockOutBinaryTile logic: validate 3 tile args, return out's type.

TEST_F(BlockOpsOutElemwiseTest, BlockSub_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto out = MakeTileVar("out", {16, 32}, DataType::FP32);
    auto lhs = MakeTileVar("lhs", {16, 32}, DataType::FP16);
    auto rhs = MakeTileVar("rhs", {16, 32}, DataType::FP16);
    auto call = reg.Create("block.sub", {out, lhs, rhs}, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

TEST_F(BlockOpsOutElemwiseTest, BlockBitwiseAnd_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto lhs = MakeTileVar("lhs", {16, 32}, DataType::INT32);
    auto rhs = MakeTileVar("rhs", {16, 32}, DataType::INT32);
    auto out = MakeTileVar("out", {16, 32}, DataType::INT32);
    auto call = reg.Create("block.and", {out, lhs, rhs}, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::INT32);
}

TEST_F(BlockOpsOutElemwiseTest, BlockAddRelu_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto lhs = MakeTileVar("lhs", {16, 32}, DataType::FP16);
    auto rhs = MakeTileVar("rhs", {16, 32}, DataType::FP16);
    auto out = MakeTileVar("out", {16, 32}, DataType::FP16);
    auto call = reg.Create("block.add_relu", {out, lhs, rhs}, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutElemwiseTest, BlockFusedMulAdd_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto lhs = MakeTileVar("lhs", {16, 32}, DataType::FP16);
    auto rhs = MakeTileVar("rhs", {16, 32}, DataType::FP16);
    auto out = MakeTileVar("out", {16, 32}, DataType::FP16);
    auto call = reg.Create("block.fused_mul_add", {out, lhs, rhs}, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutElemwiseTest, BlockRemainingBinaryTileOps_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto lhs = MakeTileVar("lhs", {16, 32}, DataType::FP16);
    auto rhs = MakeTileVar("rhs", {16, 32}, DataType::FP16);
    auto out = MakeTileVar("out", {16, 32}, DataType::FP16);

    for (const auto& op_name :
         {"block.mul", "block.div", "block.rem", "block.maximum", "block.minimum", "block.or", "block.shl", "block.shr",
          "block.sub_relu", "block.mul_add_dst", "block.fused_mul_add_relu", "block.partadd", "block.partmax",
          "block.partmin", "block.partmul"}) {
        auto call = reg.Create(op_name, {out, lhs, rhs}, Sp());
        auto rt = As<TileType>(call->GetType());
        ASSERT_NE(rt, nullptr) << op_name;
        EXPECT_EQ(rt->dtype_, DataType::FP16) << op_name;
    }
}

// block.add_relu_cast / sub_relu_cast / mul_cast: same deduction as binary tile ops
TEST_F(BlockOpsOutElemwiseTest, BlockAddReluCast_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto out = MakeTileVar("out", {16, 32}, DataType::FP16);
    auto lhs = MakeTileVar("lhs", {16, 32}, DataType::FP32);
    auto rhs = MakeTileVar("rhs", {16, 32}, DataType::FP32);
    std::vector<std::pair<std::string, std::any>> kwargs = {{"target_type", DataType::FP16},
                                                            {"mode", static_cast<int>(RoundMode::CAST_ROUND)}};
    auto call = reg.Create("block.add_relu_cast", {out, lhs, rhs}, kwargs, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutElemwiseTest, BlockSubReluCast_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto out = MakeTileVar("out", {16, 32}, DataType::FP16);
    auto lhs = MakeTileVar("lhs", {16, 32}, DataType::FP32);
    auto rhs = MakeTileVar("rhs", {16, 32}, DataType::FP32);
    std::vector<std::pair<std::string, std::any>> kwargs = {{"target_type", DataType::FP16},
                                                            {"mode", static_cast<int>(RoundMode::CAST_ROUND)}};
    auto call = reg.Create("block.sub_relu_cast", {out, lhs, rhs}, kwargs, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutElemwiseTest, BlockMulCast_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto out = MakeTileVar("out", {16, 32}, DataType::FP16);
    auto lhs = MakeTileVar("lhs", {16, 32}, DataType::FP32);
    auto rhs = MakeTileVar("rhs", {16, 32}, DataType::FP32);
    std::vector<std::pair<std::string, std::any>> kwargs = {{"target_type", DataType::FP16},
                                                            {"mode", static_cast<int>(RoundMode::CAST_ROUND)}};
    auto call = reg.Create("block.mul_cast", {out, lhs, rhs}, kwargs, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

// ============================================================================
// out_elementwise.cpp: Tile x Scalar binary ops
// ============================================================================

TEST_F(BlockOpsOutElemwiseTest, BlockAdds_TileScalarOut_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.adds",
                           {MakeTileVar("o", {16, 16}, DataType::FP16), MakeTileVar("t", {16, 16}, DataType::FP16),
                            MakeScalarVar("s", DataType::FP16)},
                           Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutElemwiseTest, BlockAdds_NonScalarThird_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("block.adds",
                         {MakeTileVar("o", {16, 16}, DataType::FP16), MakeTileVar("t", {16, 16}, DataType::FP16),
                          MakeTileVar("r", {16, 16}, DataType::FP16)},
                         Sp()),
        npu::tile_fwk::Error);
}

TEST_F(BlockOpsOutElemwiseTest, BlockRemainingBinaryScalarOps_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto out = MakeTileVar("o", {16, 16}, DataType::FP16);
    auto tile = MakeTileVar("t", {16, 16}, DataType::FP16);
    auto scalar = MakeScalarVar("s", DataType::FP16);

    for (const auto& op_name :
         {"block.subs", "block.muls", "block.divs", "block.rems", "block.maxs", "block.mins", "block.axpy"}) {
        auto call = reg.Create(op_name, {out, tile, scalar}, Sp());
        auto rt = As<TileType>(call->GetType());
        ASSERT_NE(rt, nullptr) << op_name;
        EXPECT_EQ(rt->dtype_, DataType::FP16) << op_name;
    }
}

TEST_F(BlockOpsOutElemwiseTest, BlockRemainingBinaryScalarIntOps_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto out = MakeTileVar("o", {16, 16}, DataType::INT32);
    auto tile = MakeTileVar("t", {16, 16}, DataType::INT32);
    auto scalar = MakeScalarVar("s", DataType::INT32);

    for (const auto& op_name : {"block.ands", "block.ors", "block.shls", "block.shrs"}) {
        auto call = reg.Create(op_name, {out, tile, scalar}, Sp());
        auto rt = As<TileType>(call->GetType());
        ASSERT_NE(rt, nullptr) << op_name;
        EXPECT_EQ(rt->dtype_, DataType::INT32) << op_name;
    }
}

// ============================================================================
// out_elementwise.cpp: Unary ops
// ============================================================================

TEST_F(BlockOpsOutElemwiseTest, BlockNeg_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.neg", {MakeTileVar("o", {16, 16}, DataType::FP16), MakeTileVar("s", {16, 16}, DataType::FP16)}, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutElemwiseTest, BlockNeg_NonTileSrc_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("block.neg", {MakeTileVar("o", {16, 16}, DataType::FP16), MakeScalarVar("s", DataType::FP16)},
                         Sp()),
        npu::tile_fwk::Error);
}

TEST_F(BlockOpsOutElemwiseTest, BlockRemainingUnaryOps_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto src = MakeTileVar("s", {16, 16}, DataType::FP16);
    auto out = MakeTileVar("o", {16, 16}, DataType::FP16);

    for (const auto& op_name :
         {"block.exp", "block.sqrt", "block.rsqrt", "block.recip", "block.log", "block.abs", "block.relu"}) {
        auto call = reg.Create(op_name, {out, src}, Sp());
        auto rt = As<TileType>(call->GetType());
        ASSERT_NE(rt, nullptr) << op_name;
        EXPECT_EQ(rt->dtype_, DataType::FP16) << op_name;
    }
}

TEST_F(BlockOpsOutElemwiseTest, BlockNot_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto src = MakeTileVar("s", {16, 16}, DataType::BOOL);
    auto out = MakeTileVar("o", {16, 16}, DataType::BOOL);
    auto call = reg.Create("block.not", {out, src}, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::BOOL);
}

TEST_F(BlockOpsOutElemwiseTest, BlockCast_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"target_type", DataType::FP32},
                                                            {"mode", static_cast<int>(RoundMode::CAST_NONE)}};
    auto call = reg.Create("block.cast",
                           {MakeTileVar("o", {16, 16}, DataType::FP32), MakeTileVar("s", {16, 16}, DataType::FP16)},
                           kwargs, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

TEST_F(BlockOpsOutElemwiseTest, BlockUnary_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("block.neg", {MakeTileVar("s", {16, 16}, DataType::FP16)}, Sp()),
                 npu::tile_fwk::Error);
}

// ============================================================================
// out_elementwise.cpp: Ternary / multi-input ops
// ============================================================================

TEST_F(BlockOpsOutElemwiseTest, BlockXor_FourTiles_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.xor",
                           {MakeTileVar("o", {16, 16}, DataType::INT32), MakeTileVar("l", {16, 16}, DataType::INT32),
                            MakeTileVar("r", {16, 16}, DataType::INT32), MakeTileVar("tmp", {16, 16}, DataType::INT32)},
                           Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutElemwiseTest, BlockXor_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("block.xor",
                         {MakeTileVar("l", {16, 16}, DataType::INT32), MakeTileVar("r", {16, 16}, DataType::INT32),
                          MakeTileVar("o", {16, 16}, DataType::INT32)},
                         Sp()),
        npu::tile_fwk::Error);
}

TEST_F(BlockOpsOutElemwiseTest, BlockXors_FourArgs_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.xors",
                           {MakeTileVar("o", {16, 16}, DataType::INT32), MakeTileVar("lhs", {16, 16}, DataType::INT32),
                            MakeScalarVar("s", DataType::INT32), MakeTileVar("tmp", {16, 16}, DataType::INT32)},
                           Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::INT32);
}

TEST_F(BlockOpsOutElemwiseTest, BlockSels_FiveArgs_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.sels",
                           {MakeTileVar("o", {16, 16}, DataType::FP16), MakeTileVar("mask", {16, 16}, DataType::FP16),
                            MakeTileVar("src", {16, 16}, DataType::FP16), MakeTileVar("tmp", {16, 16}, DataType::FP16),
                            MakeScalarVar("s", DataType::FP16)},
                           Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutElemwiseTest, BlockSel_FiveTiles_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.sel",
                           {MakeTileVar("o", {16, 16}, DataType::FP16), MakeTileVar("m", {16, 16}, DataType::FP16),
                            MakeTileVar("l", {16, 16}, DataType::FP16), MakeTileVar("r", {16, 16}, DataType::FP16),
                            MakeTileVar("tmp", {16, 16}, DataType::FP16)},
                           Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutElemwiseTest, BlockSel_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("block.sel",
                         {MakeTileVar("l", {16, 16}, DataType::FP16), MakeTileVar("m", {16, 16}, DataType::FP16),
                          MakeTileVar("r", {16, 16}, DataType::FP16), MakeTileVar("o", {16, 16}, DataType::FP16)},
                         Sp()),
        npu::tile_fwk::Error);
}

// ============================================================================
// out_elementwise.cpp: Comparison ops
// ============================================================================

TEST_F(BlockOpsOutElemwiseTest, BlockCmp_ThreeTiles_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"cmp_mode", int(0)}};
    auto call = reg.Create("block.cmp",
                           {MakeTileVar("o", {16, 16}, DataType::FP16), MakeTileVar("l", {16, 16}, DataType::FP16),
                            MakeTileVar("r", {16, 16}, DataType::FP16)},
                           kwargs, Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutElemwiseTest, BlockCmps_TileScalarOut_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"cmp_mode", int(1)}};
    auto call = reg.Create("block.cmps",
                           {MakeTileVar("o", {16, 16}, DataType::FP16), MakeTileVar("t", {16, 16}, DataType::FP16),
                            MakeScalarVar("s", DataType::FP16)},
                           kwargs, Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

// ============================================================================
// out_elementwise.cpp: Scalar broadcast, layout
// ============================================================================

TEST_F(BlockOpsOutElemwiseTest, BlockExpands_ScalarAndOut_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.expands",
                           {MakeTileVar("o", {16, 32}, DataType::FP16), MakeScalarVar("s", DataType::FP16)}, Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutElemwiseTest, BlockFillIndex_ScalarAndOut_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.fill_index",
                           {MakeTileVar("o", {1, 128}, DataType::INT32), MakeScalarVar("start", DataType::INT32)},
                           Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::INT32);
}

TEST_F(BlockOpsOutElemwiseTest, BlockReshape_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto shape = MakeOffsetsTuple({32, 16});
    auto call = reg.Create(
        "block.reshape",
        {MakeTileVar("o", {32, 16}, DataType::FP16), MakeTileVar("s", {16, 32}, DataType::FP16), shape}, Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutElemwiseTest, BlockTranspose_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"axis1", int(0)}, {"axis2", int(1)}};
    auto call = reg.Create("block.transpose",
                           {MakeTileVar("o", {32, 16}, DataType::FP16), MakeTileVar("s", {16, 32}, DataType::FP16)},
                           kwargs, Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

// ============================================================================
// out_elementwise.cpp: Quantize / dequantize
// ============================================================================

TEST_F(BlockOpsOutElemwiseTest, BlockQuant_3Args_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"mode", static_cast<int>(QuantMode::SYM)}};
    auto call = reg.Create("block.quant",
                           {MakeTileVar("out", {16, 32}, DataType::INT8), MakeTileVar("src", {16, 32}, DataType::FP32),
                            MakeTileVar("scale", {16, 1}, DataType::FP32)},
                           kwargs, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::INT8);
}

TEST_F(BlockOpsOutElemwiseTest, BlockDequant_4Args_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.dequant",
        {MakeTileVar("out", {16, 32}, DataType::FP32), MakeTileVar("src", {16, 32}, DataType::INT8),
         MakeTileVar("scale", {16, 1}, DataType::FP32), MakeTileVar("offset", {16, 1}, DataType::FP32)},
        Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

// ============================================================================
// out_matmul.cpp: block.matmul, block.matmul_acc, block.matmul_bias, block.gemv, block.gemv_acc, block.gemv_bias
// ============================================================================

class BlockOpsOutMatmulTest : public testing::Test {};

TEST_F(BlockOpsOutMatmulTest, BlockMatmul_LhsRhsOut_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.matmul",
                           {MakeTileVar("o", {16, 16}, DataType::FP32), MakeTileVar("l", {16, 32}, DataType::FP16),
                            MakeTileVar("r", {32, 16}, DataType::FP16)},
                           Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

TEST_F(BlockOpsOutMatmulTest, BlockMatmulAcc_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.matmul_acc",
                           {MakeTileVar("o", {16, 16}, DataType::FP32), MakeTileVar("acc", {16, 16}, DataType::FP32),
                            MakeTileVar("l", {16, 32}, DataType::FP16), MakeTileVar("r", {32, 16}, DataType::FP16)},
                           Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutMatmulTest, BlockMatmulMx_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.matmul_mx",
        {MakeTileVar("o", {64, 64}, DataType::FP32), MakeTileVar("l", {64, 64}, DataType::FP8E4M3FN),
         MakeTileVar("r", {64, 64}, DataType::FP8E5M2), MakeTileVar("sa", {64, 2}, DataType::FP8E8M0),
         MakeTileVar("sb", {2, 64}, DataType::FP8E8M0)},
        Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

TEST_F(BlockOpsOutMatmulTest, BlockMatmulMxAcc_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.matmul_mx_acc",
        {MakeTileVar("o", {64, 64}, DataType::FP32), MakeTileVar("acc", {64, 64}, DataType::FP32),
         MakeTileVar("l", {64, 64}, DataType::FP4E2M1), MakeTileVar("r", {64, 64}, DataType::FP4E1M2),
         MakeTileVar("sa", {64, 2}, DataType::FP8E8M0), MakeTileVar("sb", {2, 64}, DataType::FP8E8M0)},
        Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

TEST_F(BlockOpsOutMatmulTest, BlockGemv_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.gemv",
                           {MakeTileVar("o", {1, 16}, DataType::FP32), MakeTileVar("l", {1, 32}, DataType::FP16),
                            MakeTileVar("r", {32, 16}, DataType::FP16)},
                           Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutMatmulTest, BlockMatmulBias_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.matmul_bias",
                           {MakeTileVar("o", {16, 16}, DataType::FP32), MakeTileVar("l", {16, 32}, DataType::FP16),
                            MakeTileVar("r", {32, 16}, DataType::FP16), MakeTileVar("bias", {1, 16}, DataType::FP32)},
                           Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

TEST_F(BlockOpsOutMatmulTest, BlockGemvAcc_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.gemv_acc",
                           {MakeTileVar("o", {1, 16}, DataType::FP32), MakeTileVar("acc", {1, 16}, DataType::FP32),
                            MakeTileVar("l", {1, 32}, DataType::FP16), MakeTileVar("r", {32, 16}, DataType::FP16)},
                           Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

TEST_F(BlockOpsOutMatmulTest, BlockGemvBias_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.gemv_bias",
                           {MakeTileVar("o", {1, 16}, DataType::FP32), MakeTileVar("l", {1, 32}, DataType::FP16),
                            MakeTileVar("r", {32, 16}, DataType::FP16), MakeTileVar("bias", {1, 16}, DataType::FP32)},
                           Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

TEST_F(BlockOpsOutMatmulTest, BlockMatmul_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create(
                     "block.matmul",
                     {MakeTileVar("l", {16, 32}, DataType::FP16), MakeTileVar("r", {32, 16}, DataType::FP16)}, Sp()),
                 npu::tile_fwk::Error);
}

// ============================================================================
// out_memory.cpp: block.load, block.store
// ============================================================================

class BlockOpsOutMemoryTest : public testing::Test {};

TEST_F(BlockOpsOutMemoryTest, BlockLoad_TensorOffsetsOut_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto tensor = MakeTensorVar("tensor", {64, 128}, DataType::FP16);
    auto offsets = MakeOffsetsTuple({0, 0});
    auto out = MakeTileVar("out", {64, 128}, DataType::FP16);
    auto call = reg.Create("block.load", {out, tensor, offsets}, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutMemoryTest, BlockLoad_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("block.load", {MakeTensorVar("t", {64, 128}, DataType::FP16), MakeOffsetsTuple({0, 0})}, Sp()),
        npu::tile_fwk::Error);
}

TEST_F(BlockOpsOutMemoryTest, BlockLoad_NonTileFirst_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("block.load",
                                  {MakeScalarVar("s", DataType::FP32), MakeOffsetsTuple({0, 0}),
                                   MakeTileVar("o", {16, 32}, DataType::FP32)},
                                  Sp()),
                 npu::tile_fwk::Error);
}

TEST_F(BlockOpsOutMemoryTest, BlockStore_TileOffsetsTensor_ReturnsTensorType)
{
    auto& reg = OpRegistry::GetInstance();
    auto tile = MakeTileVar("tile", {16, 32}, DataType::FP32);
    auto offsets = MakeOffsetsTuple({0, 0});
    auto tensor = MakeTensorVar("tensor", {16, 32}, DataType::FP32);
    auto call = reg.Create("block.store", {tensor, tile, offsets}, Sp());
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

TEST_F(BlockOpsOutMemoryTest, BlockStore_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("block.store", {MakeTileVar("t", {16, 32}, DataType::FP32), MakeOffsetsTuple({0, 0})}, Sp()),
        npu::tile_fwk::Error);
}

TEST_F(BlockOpsOutMemoryTest, BlockStore_NonTensorFirst_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("block.store",
                                  {MakeScalarVar("s", DataType::FP32), MakeOffsetsTuple({0, 0}),
                                   MakeTensorVar("t", {16, 32}, DataType::FP32)},
                                  Sp()),
                 npu::tile_fwk::Error);
}

TEST_F(BlockOpsOutMemoryTest, BlockStore_ScalingTile_ReturnsTensorType)
{
    auto& reg = OpRegistry::GetInstance();
    auto tile = MakeTileVar("tile", {16, 32}, DataType::FP32);
    auto fp_tile = MakeTileVar("fp", {16, 32}, DataType::FP32);
    auto offsets = MakeOffsetsTuple({0, 0});
    auto tensor = MakeTensorVar("tensor", {16, 32}, DataType::FP32);
    auto call = reg.Create("block.store", {tensor, tile, offsets, fp_tile}, Sp());
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
}

TEST_F(BlockOpsOutMemoryTest, BlockStore_InvalidScaleType_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("block.store",
                         {MakeTensorVar("out", {16, 32}, DataType::FP32), MakeTileVar("tile", {16, 32}, DataType::FP32),
                          MakeOffsetsTuple({0, 0}), MakeTensorVar("scale", {16, 32}, DataType::FP32)},
                         Sp()),
        npu::tile_fwk::Error);
}

// ============================================================================
// out_memory.cpp: block.move, block.ub_copy
// ============================================================================

TEST_F(BlockOpsOutMemoryTest, BlockMove_OutSrc_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.move", {MakeTileVar("o", {16, 32}, DataType::FP16), MakeTileVar("s", {16, 32}, DataType::FP16)}, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutMemoryTest, BlockMove_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("block.move", {MakeTileVar("s", {16, 32}, DataType::FP16)}, Sp()),
                 npu::tile_fwk::Error);
}

TEST_F(BlockOpsOutMemoryTest, BlockMove_WithScalingTile_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.move",
                           {MakeTileVar("o", {16, 32}, DataType::FP16), MakeTileVar("s", {16, 32}, DataType::FP32),
                            MakeTileVar("fp", {16, 32}, DataType::FP32)},
                           Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutMemoryTest, BlockMove_WithOffsetAndScalingTile_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.move",
                           {MakeTileVar("o", {16, 32}, DataType::FP16), MakeTileVar("s", {32, 64}, DataType::FP32),
                            MakeOffsetsTuple({0, 16}), MakeTileVar("fp", {1, 32}, DataType::INT64)},
                           Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutMemoryTest, BlockMove_InvalidOptionalArgument_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("block.move",
                         {MakeTileVar("o", {16, 32}, DataType::FP16), MakeTileVar("s", {32, 64}, DataType::FP32),
                          MakeTensorVar("invalid", {1, 32}, DataType::INT64)},
                         Sp()),
        npu::tile_fwk::Error);
}

TEST_F(BlockOpsOutMemoryTest, BlockUbCopy_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.ub_copy",
                           {MakeTileVar("o", {16, 16}, DataType::FP16), MakeTileVar("s", {16, 16}, DataType::FP16)},
                           Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

// ============================================================================
// out_memory.cpp: block.insert
// ============================================================================

TEST_F(BlockOpsOutMemoryTest, BlockInsert_4Args_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.insert",
                           {MakeTileVar("o", {64, 64}, DataType::FP16), MakeTileVar("src", {16, 16}, DataType::FP16),
                            MakeScalarVar("row", DataType::INT32), MakeScalarVar("col", DataType::INT32)},
                           Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
}

TEST_F(BlockOpsOutMemoryTest, BlockInsert_WithScale_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.insert",
                           {MakeTileVar("o", {64, 64}, DataType::INT8), MakeTileVar("src", {16, 16}, DataType::FP32),
                            MakeScalarVar("row", DataType::INT32), MakeScalarVar("col", DataType::INT32),
                            MakeScalarVar("scale", DataType::INT64)},
                           Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutMemoryTest, BlockInsert_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("block.insert",
                         {MakeTileVar("s", {16, 16}, DataType::FP16), MakeScalarVar("r", DataType::INT32)}, Sp()),
        npu::tile_fwk::Error);
}

// ============================================================================
// out_memory.cpp: block.full, block.ssbuf_store, block.ssbuf_load
// ============================================================================

TEST_F(BlockOpsOutMemoryTest, BlockFull_OutScalar_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.full",
                           {MakeTileVar("o", {16, 32}, DataType::FP16), MakeScalarVar("v", DataType::FP16)}, Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutMemoryTest, BlockSsbufStore_ReturnsNone)
{
    auto& reg = OpRegistry::GetInstance();
    auto struct_type = std::make_shared<TupleType>(std::vector<TypePtr>{std::make_shared<ScalarType>(DataType::INT64)});
    auto struct_var = std::make_shared<Var>("state", struct_type, Sp());
    auto call = reg.Create("block.ssbuf_store", {struct_var, MakeScalarVar("off", DataType::INT32)}, Sp());
    EXPECT_NE(As<NoneType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutMemoryTest, BlockSsbufLoad_ReturnsNone)
{
    auto& reg = OpRegistry::GetInstance();
    auto struct_type = std::make_shared<TupleType>(std::vector<TypePtr>{std::make_shared<ScalarType>(DataType::INT64)});
    auto struct_var = std::make_shared<Var>("state", struct_type, Sp());
    auto call = reg.Create("block.ssbuf_load", {struct_var, MakeScalarVar("off", DataType::INT32)}, Sp());
    EXPECT_NE(As<NoneType>(call->GetType()), nullptr);
}

// ============================================================================
// out_memory.cpp: block.set_validshape
// ============================================================================

TEST_F(BlockOpsOutMemoryTest, BlockSetValidshape_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.set_validshape",
                           {MakeTileVar("t", {16, 32}, DataType::FP16), MakeScalarVar("row", DataType::INT32),
                            MakeScalarVar("col", DataType::INT32)},
                           Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

// ============================================================================
// out_memory.cpp: block.init_output
// ============================================================================

TEST_F(BlockOpsOutMemoryTest, BlockInitOutput_ReturnsNone)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create(
        "block.init_output",
        {MakeTensorVar("tensor", {64, 128}, DataType::FP32), MakeScalarVar("offset", DataType::INT64),
         MakeScalarVar("size", DataType::INT64), MakeScalarVar("value", DataType::FP32)},
        Sp());
    EXPECT_NE(As<NoneType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutMemoryTest, BlockInitOutput_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("block.init_output",
                                  {MakeTensorVar("t", {64, 128}, DataType::FP32),
                                   MakeScalarVar("offset", DataType::INT64), MakeScalarVar("size", DataType::INT64)},
                                  Sp()),
                 npu::tile_fwk::Error);
}

TEST_F(BlockOpsOutMemoryTest, BlockInitOutput_NonTensorFirst_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("block.init_output",
                                  {MakeScalarVar("s", DataType::FP32), MakeScalarVar("offset", DataType::INT64),
                                   MakeScalarVar("size", DataType::INT64), MakeScalarVar("value", DataType::FP32)},
                                  Sp()),
                 npu::tile_fwk::Error);
}

// ============================================================================
// out_memory.cpp: block.fillpad, block.fillpad_expand
// ============================================================================

TEST_F(BlockOpsOutMemoryTest, BlockFillpad_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto src = MakeTileVarWithHwInfo("src", {16, 32}, DataType::FP16, TilePad::zero);
    auto out = MakeTileVarWithHwInfo("out", {16, 32}, DataType::FP16, TilePad::zero);
    auto call = reg.Create("block.fillpad", {out, src}, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
}

TEST_F(BlockOpsOutMemoryTest, BlockFillpad_NoHardwareInfo_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    auto src = MakeTileVar("src", {16, 32}, DataType::FP16);
    auto out = MakeTileVar("out", {16, 32}, DataType::FP16);
    EXPECT_THROW((void)reg.Create("block.fillpad", {out, src}, Sp()), npu::tile_fwk::Error);
}

TEST_F(BlockOpsOutMemoryTest, BlockFillpad_PadNull_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    auto src = MakeTileVar("src", {16, 32}, DataType::FP16);
    auto out = MakeTileVarWithHwInfo("out", {16, 32}, DataType::FP16, TilePad::null);
    EXPECT_THROW((void)reg.Create("block.fillpad", {out, src}, Sp()), npu::tile_fwk::Error);
}

TEST_F(BlockOpsOutMemoryTest, BlockFillpadExpand_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto src = MakeTileVarWithHwInfo("src", {16, 32}, DataType::FP16, TilePad::zero);
    auto out = MakeTileVarWithHwInfo("out", {32, 64}, DataType::FP16, TilePad::zero);
    auto call = reg.Create("block.fillpad_expand", {out, src}, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
}

TEST_F(BlockOpsOutMemoryTest, BlockFillpadExpand_SrcLargerThanOut_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    auto src = MakeTileVarWithHwInfo("src", {32, 64}, DataType::FP16, TilePad::zero);
    auto out = MakeTileVarWithHwInfo("out", {16, 32}, DataType::FP16, TilePad::zero);
    EXPECT_THROW((void)reg.Create("block.fillpad_expand", {out, src}, Sp()), npu::tile_fwk::Error);
}

TEST_F(BlockOpsOutMemoryTest, BlockFillpadInplace_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto src = MakeTileVarWithMemRef("src", {16, 32}, DataType::FP16, MemorySpace::Vec, 0, 1024);
    auto addr_expr = std::make_shared<ConstInt>(int64_t(0), DataType::INDEX, Sp());
    auto memref = std::make_shared<MemRef>(MemorySpace::Vec, addr_expr, uint64_t(1024));
    HardwareInfo hw(TileLayout::row_major, TileLayout::none_box, 512, TilePad::zero, CompactMode::null);
    auto out_type = std::make_shared<TileType>(std::vector<int64_t>{16, 32}, DataType::FP16,
                                               std::optional<MemRefPtr>(memref), std::nullopt,
                                               std::optional<HardwareInfo>(hw));
    auto out = std::make_shared<Var>("out", out_type, Sp());
    auto call = reg.Create("block.fillpad_inplace", {out, src}, Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
}

TEST_F(BlockOpsOutMemoryTest, BlockFillpadInplace_NoMemRef_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    auto src = MakeTileVarWithHwInfo("src", {16, 32}, DataType::FP16, TilePad::zero);
    auto out = MakeTileVarWithHwInfo("out", {16, 32}, DataType::FP16, TilePad::zero);
    EXPECT_THROW((void)reg.Create("block.fillpad_inplace", {out, src}, Sp()), npu::tile_fwk::Error);
}

// ============================================================================
// out_reduction.cpp: reductions and broadcasts
// ============================================================================

class BlockOpsOutReductionTest : public testing::Test {};

TEST_F(BlockOpsOutReductionTest, BlockRowSum_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.row_sum",
                           {MakeTileVar("o", {16, 1}, DataType::FP16), MakeTileVar("t", {16, 32}, DataType::FP16),
                            MakeTileVar("tmp", {16, 1}, DataType::FP16)},
                           Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutReductionTest, BlockRowMax_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.row_max",
                           {MakeTileVar("o", {16, 1}, DataType::FP16), MakeTileVar("t", {16, 32}, DataType::FP16),
                            MakeTileVar("tmp", {16, 1}, DataType::FP16)},
                           Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutReductionTest, BlockRowMin_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.row_min",
                           {MakeTileVar("o", {16, 1}, DataType::FP16), MakeTileVar("t", {16, 32}, DataType::FP16),
                            MakeTileVar("tmp", {16, 1}, DataType::FP16)},
                           Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutReductionTest, BlockColMax_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.col_max",
                           {MakeTileVar("o", {1, 32}, DataType::FP16), MakeTileVar("t", {16, 32}, DataType::FP16),
                            MakeTileVar("tmp", {1, 32}, DataType::FP16)},
                           Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockRowArgmax_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.row_argmax",
                           {MakeTileVar("o", {16, 1}, DataType::INT32), MakeTileVar("t", {16, 32}, DataType::FP16),
                            MakeTileVar("tmp", {16, 1}, DataType::INT32)},
                           Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockRowExpand_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.row_expand",
                           {MakeTileVar("o", {16, 32}, DataType::FP16), MakeTileVar("s", {16, 1}, DataType::FP16)},
                           Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockRowExpandAdd_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.row_expand_add",
                           {MakeTileVar("o", {16, 32}, DataType::FP16), MakeTileVar("t", {16, 32}, DataType::FP16),
                            MakeTileVar("rv", {16, 1}, DataType::FP16)},
                           Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockColExpandAdd_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.col_expand_add",
                           {MakeTileVar("o", {16, 32}, DataType::FP16), MakeTileVar("t", {16, 32}, DataType::FP16),
                            MakeTileVar("cv", {1, 32}, DataType::FP16)},
                           Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockColSum_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.col_sum",
                           {MakeTileVar("o", {1, 32}, DataType::FP16), MakeTileVar("t", {16, 32}, DataType::FP16),
                            MakeTileVar("tmp", {1, 32}, DataType::FP16)},
                           Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutReductionTest, BlockColMin_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.col_min",
                           {MakeTileVar("o", {1, 32}, DataType::FP16), MakeTileVar("t", {16, 32}, DataType::FP16),
                            MakeTileVar("tmp", {1, 32}, DataType::FP16)},
                           Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutReductionTest, BlockRowProd_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.row_prod",
                           {MakeTileVar("o", {16, 1}, DataType::FP16), MakeTileVar("t", {16, 32}, DataType::FP16),
                            MakeTileVar("tmp", {16, 1}, DataType::FP16)},
                           Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutReductionTest, BlockColProd_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.col_prod",
                           {MakeTileVar("o", {1, 32}, DataType::FP16), MakeTileVar("t", {16, 32}, DataType::FP16),
                            MakeTileVar("tmp", {1, 32}, DataType::FP16)},
                           Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutReductionTest, BlockRowReduce_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.row_reduce",
                           {MakeTileVar("o", {16, 1}, DataType::FP16), MakeTileVar("t", {16, 32}, DataType::FP16),
                            MakeTileVar("tmp", {16, 1}, DataType::FP16)},
                           Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutReductionTest, BlockColReduce_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.col_reduce",
                           {MakeTileVar("o", {1, 32}, DataType::FP16), MakeTileVar("t", {16, 32}, DataType::FP16),
                            MakeTileVar("tmp", {1, 32}, DataType::FP16)},
                           Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutReductionTest, BlockRowArgmin_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.row_argmin",
                           {MakeTileVar("o", {16, 1}, DataType::INT32), MakeTileVar("t", {16, 32}, DataType::FP16),
                            MakeTileVar("tmp", {16, 1}, DataType::INT32)},
                           Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::INT32);
}

TEST_F(BlockOpsOutReductionTest, BlockColArgmax_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.col_argmax",
                           {MakeTileVar("o", {1, 32}, DataType::INT32), MakeTileVar("t", {16, 32}, DataType::FP16),
                            MakeTileVar("tmp", {1, 32}, DataType::INT32)},
                           Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::INT32);
}

TEST_F(BlockOpsOutReductionTest, BlockColArgmin_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.col_argmin",
                           {MakeTileVar("o", {1, 32}, DataType::INT32), MakeTileVar("t", {16, 32}, DataType::FP16),
                            MakeTileVar("tmp", {1, 32}, DataType::INT32)},
                           Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::INT32);
}

TEST_F(BlockOpsOutReductionTest, BlockColExpand_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.col_expand",
                           {MakeTileVar("o", {16, 32}, DataType::FP16), MakeTileVar("cv", {1, 32}, DataType::FP16)},
                           Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsOutReductionTest, BlockRowExpandSub_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.row_expand_sub",
                           {MakeTileVar("o", {16, 32}, DataType::FP16), MakeTileVar("t", {16, 32}, DataType::FP16),
                            MakeTileVar("rv", {16, 1}, DataType::FP16)},
                           Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockRowExpandMul_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.row_expand_mul",
                           {MakeTileVar("o", {16, 32}, DataType::FP16), MakeTileVar("t", {16, 32}, DataType::FP16),
                            MakeTileVar("rv", {16, 1}, DataType::FP16)},
                           Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockRowExpandExpdif_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.row_expand_expdif",
                           {MakeTileVar("o", {16, 32}, DataType::FP16), MakeTileVar("t", {16, 32}, DataType::FP16),
                            MakeTileVar("rv", {16, 1}, DataType::FP16)},
                           Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockRowExpandBinop_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.row_expand_binop",
                           {MakeTileVar("o", {16, 32}, DataType::FP16), MakeTileVar("t", {16, 32}, DataType::FP16),
                            MakeTileVar("rv", {16, 1}, DataType::FP16)},
                           Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockColExpandMul_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.col_expand_mul",
                           {MakeTileVar("o", {16, 32}, DataType::FP16), MakeTileVar("t", {16, 32}, DataType::FP16),
                            MakeTileVar("cv", {1, 32}, DataType::FP16)},
                           Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockColExpandExpdif_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.col_expand_expdif",
                           {MakeTileVar("o", {16, 32}, DataType::FP16), MakeTileVar("t", {16, 32}, DataType::FP16),
                            MakeTileVar("cv", {1, 32}, DataType::FP16)},
                           Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockColExpandBinop_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.col_expand_binop",
                           {MakeTileVar("o", {16, 32}, DataType::FP16), MakeTileVar("t", {16, 32}, DataType::FP16),
                            MakeTileVar("cv", {1, 32}, DataType::FP16)},
                           Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockRowExpandDiv_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.row_expand_div",
                           {MakeTileVar("o", {16, 32}, DataType::FP16), MakeTileVar("t", {16, 32}, DataType::FP16),
                            MakeTileVar("rv", {16, 1}, DataType::FP16)},
                           Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockRowExpandMax_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.row_expand_max",
                           {MakeTileVar("o", {16, 32}, DataType::FP16), MakeTileVar("t", {16, 32}, DataType::FP16),
                            MakeTileVar("rv", {16, 1}, DataType::FP16)},
                           Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockRowExpandMin_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.row_expand_min",
                           {MakeTileVar("o", {16, 32}, DataType::FP16), MakeTileVar("t", {16, 32}, DataType::FP16),
                            MakeTileVar("rv", {16, 1}, DataType::FP16)},
                           Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockColExpandDiv_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.col_expand_div",
                           {MakeTileVar("o", {16, 32}, DataType::FP16), MakeTileVar("t", {16, 32}, DataType::FP16),
                            MakeTileVar("cv", {1, 32}, DataType::FP16)},
                           Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockColExpandSub_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.col_expand_sub",
                           {MakeTileVar("o", {16, 32}, DataType::FP16), MakeTileVar("t", {16, 32}, DataType::FP16),
                            MakeTileVar("cv", {1, 32}, DataType::FP16)},
                           Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockColExpandMax_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.col_expand_max",
                           {MakeTileVar("o", {16, 32}, DataType::FP16), MakeTileVar("t", {16, 32}, DataType::FP16),
                            MakeTileVar("cv", {1, 32}, DataType::FP16)},
                           Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutReductionTest, BlockColExpandMin_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.col_expand_min",
                           {MakeTileVar("o", {16, 32}, DataType::FP16), MakeTileVar("t", {16, 32}, DataType::FP16),
                            MakeTileVar("cv", {1, 32}, DataType::FP16)},
                           Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

// ============================================================================
// sort.cpp: block.sort32, block.mrgsort, block.mrgsort2, block.histogram
// ============================================================================

class BlockOpsSortTest : public testing::Test {};

TEST_F(BlockOpsSortTest, BlockSort32_3Args_ReturnsDstType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.sort32",
                           {MakeTileVar("dst", {32}, DataType::FP16), MakeTileVar("src", {32}, DataType::FP16),
                            MakeTileVar("idx", {32}, DataType::UINT32)},
                           Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsSortTest, BlockSort32_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("block.sort32", {MakeTileVar("src", {32}, DataType::FP16)}, Sp()),
                 npu::tile_fwk::Error);
}

TEST_F(BlockOpsSortTest, BlockMrgsort_SrcDst_ReturnsDstType)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"block_len", int(32)}};
    auto call = reg.Create("block.mrgsort",
                           {MakeTileVar("dst", {64}, DataType::FP16), MakeTileVar("src", {64}, DataType::FP16)}, kwargs,
                           Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsSortTest, BlockMrgsort_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("block.mrgsort", {MakeTileVar("src", {64}, DataType::FP16)}, Sp()),
                 npu::tile_fwk::Error);
}

TEST_F(BlockOpsSortTest, BlockMrgsort2_4Args_ReturnsDstType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.mrgsort2",
                           {MakeTileVar("dst", {64}, DataType::FP16), MakeTileVar("src0", {64}, DataType::FP16),
                            MakeTileVar("tmp", {64}, DataType::FP16), MakeTileVar("src1", {64}, DataType::FP16)},
                           Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(BlockOpsSortTest, BlockMrgsort2_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("block.mrgsort2",
                                  {MakeTileVar("s0", {64}, DataType::FP16), MakeTileVar("dst", {64}, DataType::FP16),
                                   MakeTileVar("tmp", {64}, DataType::FP16)},
                                  Sp()),
                 npu::tile_fwk::Error);
}

TEST_F(BlockOpsSortTest, BlockHistogram_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.histogram",
                           {MakeTileVar("dst", {1, 256}, DataType::UINT32), MakeTileVar("src", {32}, DataType::UINT16),
                            MakeTileVar("idx", {32}, DataType::UINT8)},
                           Sp());
    auto rt = As<TileType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::UINT32);
}

TEST_F(BlockOpsSortTest, BlockHistogram_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("block.histogram",
                         {MakeTileVar("src", {32}, DataType::UINT16), MakeTileVar("idx", {32}, DataType::UINT8)}, Sp()),
        npu::tile_fwk::Error);
}

// ============================================================================
// out_elementwise.cpp: block.gather (index form and compare form)
// ============================================================================

class BlockOpsGatherTest : public testing::Test {};

TEST_F(BlockOpsGatherTest, BlockGather_IndexForm3Args_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.gather",
                           {MakeTileVar("out", {64}, DataType::FP16), MakeTileVar("src", {64}, DataType::FP16),
                            MakeTileVar("indices", {64}, DataType::INT32)},
                           Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsGatherTest, BlockGather_CompareForm5Args_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"cmp_mode", int(0)}, {"offset", int(0)}};
    auto call = reg.Create("block.gather",
                           {MakeTileVar("out", {64}, DataType::FP16), MakeTileVar("src", {64}, DataType::FP16),
                            MakeTileVar("kval", {64}, DataType::FP16), MakeTileVar("cdst", {64}, DataType::FP16),
                            MakeTileVar("tmp", {64}, DataType::FP16)},
                           kwargs, Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsGatherTest, BlockGatherb_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.gatherb",
                           {MakeTileVar("out", {64}, DataType::FP16), MakeTileVar("src", {64}, DataType::FP16),
                            MakeTileVar("offsets", {64}, DataType::INT32)},
                           Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsGatherTest, BlockGathermask_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"pattern_mode", int(1)}};
    auto call = reg.Create("block.gathermask",
                           {MakeTileVar("out", {64}, DataType::FP16), MakeTileVar("src", {64}, DataType::FP16)}, kwargs,
                           Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

TEST_F(BlockOpsOutElemwiseTest, BlockGather_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create(
                     "block.gather",
                     {MakeTileVar("src", {64}, DataType::FP16), MakeTileVar("indices", {64}, DataType::INT32)}, Sp()),
                 std::runtime_error);
}

TEST_F(BlockOpsGatherTest, BlockScatter_ReturnsOutType)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("block.scatter",
                           {MakeTileVar("dst", {16, 32}, DataType::FP16), MakeTileVar("src", {16, 32}, DataType::FP16),
                            MakeTileVar("indices", {16, 32}, DataType::INT32)},
                           Sp());
    EXPECT_NE(As<TileType>(call->GetType()), nullptr);
}

// ============================================================================
// OpRegistry::GetEntry and GetOp
// ============================================================================

class OpRegistryQueryTest : public testing::Test {};

TEST_F(OpRegistryQueryTest, GetEntry_RegisteredOp_ReturnsEntry)
{
    auto& reg = OpRegistry::GetInstance();
    const auto& entry = reg.GetEntry("get_block_idx");
    EXPECT_EQ(entry.GetName(), "get_block_idx");
    EXPECT_NE(entry.GetOp(), nullptr);
}

TEST_F(OpRegistryQueryTest, GetEntry_UnregisteredOp_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.GetEntry("nonexistent_op_xyz"), npu::tile_fwk::Error);
}

TEST_F(OpRegistryQueryTest, GetOp_RegisteredOp_ReturnsOp)
{
    auto& reg = OpRegistry::GetInstance();
    auto op = reg.GetOp("block.add");
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->name_, "block.add");
}

TEST_F(OpRegistryQueryTest, GetOp_UnregisteredOp_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.GetOp("nonexistent_op_abc"), npu::tile_fwk::Error);
}

// ============================================================================
// memory.cpp: set_saturation_flag, get_saturation_flag
// ============================================================================

class BlockOpsSaturationFlagTest : public testing::Test {};

TEST_F(BlockOpsSaturationFlagTest, SetSaturationFlag_ReturnsUnknownType)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"mode", static_cast<int>(ir::SaturationFlagMode::CAST)},
                                                            {"enable", true}};
    auto call = reg.Create("set_saturation_flag", {}, kwargs, Sp());
    ASSERT_NE(call, nullptr);
    auto rt = call->GetType();
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->TypeName(), "UnknownType");
}

TEST_F(BlockOpsSaturationFlagTest, GetSaturationFlag_ReturnsBoolType)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"mode", static_cast<int>(ir::SaturationFlagMode::CAST)}};
    auto call = reg.Create("get_saturation_flag", {}, kwargs, Sp());
    ASSERT_NE(call, nullptr);
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::BOOL);
}

// ============================================================================
// memory.cpp: set_ctrl_spr, get_ctrl_spr, reset_ctrl_spr
// ============================================================================

class BlockOpsCtrlSprTest : public testing::Test {};

TEST_F(BlockOpsCtrlSprTest, SetCtrlSpr_ReturnsUnknownType)
{
    auto& reg = OpRegistry::GetInstance();
    auto startBit = std::make_shared<ConstInt>(59, DataType::INT64, Sp());
    auto endBit = std::make_shared<ConstInt>(59, DataType::INT64, Sp());
    auto value = std::make_shared<ConstInt>(0, DataType::INT64, Sp());
    auto call = reg.Create("set_ctrl_spr", {startBit, endBit, value}, Sp());
    ASSERT_NE(call, nullptr);
    auto rt = call->GetType();
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->TypeName(), "UnknownType");
}

TEST_F(BlockOpsCtrlSprTest, SetCtrlSpr_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    auto startBit = std::make_shared<ConstInt>(59, DataType::INT64, Sp());
    EXPECT_THROW((void)reg.Create("set_ctrl_spr", {startBit}, Sp()), npu::tile_fwk::Error);
}

TEST_F(BlockOpsCtrlSprTest, GetCtrlSpr_ReturnsInt64Type)
{
    auto& reg = OpRegistry::GetInstance();
    auto startBit = std::make_shared<ConstInt>(48, DataType::INT64, Sp());
    auto endBit = std::make_shared<ConstInt>(48, DataType::INT64, Sp());
    auto call = reg.Create("get_ctrl_spr", {startBit, endBit}, Sp());
    ASSERT_NE(call, nullptr);
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::INT64);
}

TEST_F(BlockOpsCtrlSprTest, GetCtrlSpr_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("get_ctrl_spr", {}, Sp()), npu::tile_fwk::Error);
}

TEST_F(BlockOpsCtrlSprTest, ResetCtrlSpr_ReturnsUnknownType)
{
    auto& reg = OpRegistry::GetInstance();
    auto startBit = std::make_shared<ConstInt>(59, DataType::INT64, Sp());
    auto endBit = std::make_shared<ConstInt>(59, DataType::INT64, Sp());
    auto call = reg.Create("reset_ctrl_spr", {startBit, endBit}, Sp());
    ASSERT_NE(call, nullptr);
    auto rt = call->GetType();
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->TypeName(), "UnknownType");
}

TEST_F(BlockOpsCtrlSprTest, ResetCtrlSpr_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("reset_ctrl_spr", {}, Sp()), npu::tile_fwk::Error);
}

} // namespace ir
} // namespace pypto
