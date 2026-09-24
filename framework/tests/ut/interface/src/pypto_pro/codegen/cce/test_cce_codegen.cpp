/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include <any>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "codegen/cce/cce_codegen.h"
#include "ir/debug_info.h"
#include "ir/function.h"
#include "ir/program.h"
#include "ir/stmt.h"

namespace pypto {
namespace codegen {
namespace {

ir::ExprPtr MakeConstInt(int64_t value)
{
    return std::make_shared<const ir::ConstInt>(value, ir::DataType::INT64, ir::Span::Unknown());
}

ir::VarPtr MakeVar(const std::string& name, const ir::TypePtr& type)
{
    return std::make_shared<const ir::Var>(name, type, ir::Span::Unknown());
}

ir::VarPtr MakeTensorVar(const std::string& name, const std::vector<int64_t>& shape, ir::DataType dtype)
{
    auto ptr = MakeVar(name + "_base", std::make_shared<const ir::PtrType>(dtype));
    ir::TensorView view({}, ir::TensorLayout::ND, ptr);
    auto tensor_type = std::make_shared<const ir::TensorType>(shape, dtype, std::optional<ir::MemRefPtr>(std::nullopt),
                                                              std::optional<ir::TensorView>(view));
    return MakeVar(name, tensor_type);
}

ir::ProgramPtr MakeProgram(const ir::StmtPtr& body, const std::vector<ir::VarPtr>& params = {},
                           ir::IRDebugInfoPtr debug_info = nullptr)
{
    if (debug_info == nullptr) {
        debug_info = std::make_shared<ir::IRDebugInfo>();
    }
    auto function = std::make_shared<const ir::Function>("kernel", params, std::vector<ir::TypePtr>{}, body,
                                                         ir::Span::Unknown(), ir::FunctionType::IN_CORE, true);
    return std::make_shared<const ir::Program>(std::vector<ir::FunctionPtr>{function}, "test_program",
                                               ir::Span::Unknown(), std::move(debug_info));
}

size_t CountOccurrences(const std::string& text, const std::string& needle)
{
    size_t count = 0;
    size_t pos = 0;
    while ((pos = text.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}

} // namespace

TEST(CCECodegenHeaderTest, CoversHeaderOnlyStateAccessors)
{
    CCECodegen codegen(ir::SectionKind::Vector);

    EXPECT_TRUE(codegen.GetCurrentResultTarget().empty());
    EXPECT_EQ(codegen.GetArch(), npu::tile_fwk::NPUArch::DAV_2201);
    EXPECT_EQ(codegen.GetTarget(), ir::SectionKind::Vector);
    EXPECT_FALSE(codegen.IsInVFSection());
    EXPECT_EQ(codegen.GetTileAddress("unknown_tile"), "0x0");
    EXPECT_TRUE(codegen.GetTilingHeaders().empty());
    EXPECT_EQ(codegen.GetTypeConverter().ConvertEventId(3), "EVENT_ID3");

    codegen.RegisterRegTensorVar("reg_tensor");
    EXPECT_TRUE(codegen.IsRegTensorVar("reg_tensor"));
    EXPECT_FALSE(codegen.IsRegTensorVar("other_reg_tensor"));
    codegen.HoistRegTensorDecl("RegTensor<float> reg_tensor;");

    codegen.RegisterMaskRegVar("mask_reg");
    EXPECT_TRUE(codegen.IsMaskRegVar("mask_reg"));
    EXPECT_FALSE(codegen.IsMaskRegVar("other_mask_reg"));

    codegen.RegisterAddrRegVar("addr_reg");
    EXPECT_TRUE(codegen.IsAddrRegVar("addr_reg"));
    EXPECT_FALSE(codegen.IsAddrRegVar("other_addr_reg"));

    EXPECT_EQ(codegen.GetTileOffsetCounter(), 0);
    EXPECT_EQ(codegen.GetTileOffsetCounter(), 1);

    EXPECT_FALSE(codegen.HasTileAddress("tile_0"));
    codegen.SetTileAddress("tile_0", "0x100");
    EXPECT_TRUE(codegen.HasTileAddress("tile_0"));
    EXPECT_EQ(codegen.GetTileAddress("tile_0"), "0x100");
}

TEST(CCECodegenHeaderTest, GeneratesStandaloneTilingHeader)
{
    auto index_type = std::make_shared<const ir::ScalarType>(ir::DataType::INDEX);
    auto float_type = std::make_shared<const ir::ScalarType>(ir::DataType::FP32);
    auto bool_type = std::make_shared<const ir::ScalarType>(ir::DataType::BOOL);
    auto offsets_type = std::make_shared<const ir::TupleType>(std::vector<ir::TypePtr>(4, index_type));

    auto header = CCECodegen::GenerateTilingHeader("TestTiling", {"rows", "scale", "enabled", "offsets"},
                                                   {index_type, float_type, bool_type, offsets_type});

    EXPECT_NE(header.find("#pragma once"), std::string::npos);
    EXPECT_NE(header.find("class TestTiling"), std::string::npos);
    EXPECT_NE(header.find("int64_t rows;"), std::string::npos);
    EXPECT_NE(header.find("float scale;"), std::string::npos);
    EXPECT_NE(header.find("bool enabled;"), std::string::npos);
    EXPECT_NE(header.find("int64_t offsets[4];"), std::string::npos);
    EXPECT_THROW((void)CCECodegen::GenerateTilingHeader("BadTiling", {"rows"}, {}), npu::tile_fwk::Error);
}

TEST(CCECodegenHeaderTest, RejectsNonCubeOrVectorTarget)
{
    EXPECT_THROW((void)CCECodegen(ir::SectionKind::VF), npu::tile_fwk::Error);
}

TEST(CCECodegenTest, EmitsTargetSpecificGuardsNamesAndVectorSetup)
{
    auto body = std::make_shared<const ir::ReturnStmt>(ir::Span::Unknown());

    CCECodegen cube_codegen(ir::SectionKind::Cube);
    auto cube = cube_codegen.GenerateSingle(MakeProgram(body), "a3");
    EXPECT_NE(cube.find("#if defined(__DAV_CUBE__)"), std::string::npos);
    EXPECT_NE(cube.find("kernel_impl_cube("), std::string::npos);
    EXPECT_EQ(cube.find("set_mask_norm();"), std::string::npos);
    EXPECT_EQ(cube.find("set_vector_mask(-1, -1);"), std::string::npos);

    CCECodegen vector_codegen(ir::SectionKind::Vector);
    auto vector = vector_codegen.GenerateSingle(MakeProgram(body), "a3");
    EXPECT_NE(vector.find("#if defined(__DAV_VEC__)"), std::string::npos);
    EXPECT_NE(vector.find("kernel_impl_vector("), std::string::npos);
    EXPECT_NE(vector.find("set_mask_norm();"), std::string::npos);
    EXPECT_NE(vector.find("set_vector_mask(-1, -1);"), std::string::npos);
}

TEST(CCECodegenTest, EmitsInt64DynamicTensorDimensions)
{
    auto m = MakeVar("__pypto_dyn_x_0", std::make_shared<const ir::ScalarType>(ir::DataType::INT64));
    auto n = MakeVar("__pypto_dyn_x_1", std::make_shared<const ir::ScalarType>(ir::DataType::INT64));
    auto ptr = MakeVar("x_base", std::make_shared<const ir::PtrType>(ir::DataType::FP32));
    ir::TensorView view({}, ir::TensorLayout::ND, ptr);
    auto tensor_type = std::make_shared<const ir::TensorType>(std::vector<ir::ExprPtr>{m, n}, ir::DataType::FP32,
                                                              std::optional<ir::MemRefPtr>(std::nullopt),
                                                              std::optional<ir::TensorView>(view));
    auto x = MakeVar("x", tensor_type);
    auto body = std::make_shared<const ir::ReturnStmt>(ir::Span::Unknown());

    CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body, {x}), "a5");
    EXPECT_NE(generated.find("int64_t __pypto_dyn_x_0"), std::string::npos);
    EXPECT_NE(generated.find("int64_t __pypto_dyn_x_1"), std::string::npos);
    EXPECT_EQ(generated.find("int32_t __pypto_dyn_x_0"), std::string::npos);
}

TEST(CCECodegenTest, EmitsTargetSpecificTilingStructCopy)
{
    std::vector<ir::TypePtr> field_types = {
        std::make_shared<const ir::ScalarType>(ir::DataType::INT32),
        std::make_shared<const ir::ScalarType>(ir::DataType::INT64),
    };
    auto tuple_type = std::make_shared<const ir::TupleType>(field_types);
    auto tiling = MakeVar("tiling", tuple_type);
    auto debug_info = std::make_shared<ir::IRDebugInfo>();
    debug_info->RegisterTupleTypeInfo(tuple_type,
                                      {ir::TupleTypeKind::STRUCT, std::string("TestTiling"), {"rows", "cols"}});
    auto body = std::make_shared<const ir::ReturnStmt>(ir::Span::Unknown());
    auto program = MakeProgram(body, {tiling}, debug_info);

    CCECodegen cube_codegen(ir::SectionKind::Cube);
    auto cube = cube_codegen.GenerateSingle(program, "a5");
    EXPECT_NE(cube.find("copy_data_align64((uint8_t*)&tiling, (__gm__ uint8_t *)tiling_ptr"), std::string::npos);
    EXPECT_EQ(cube.find("copy_gm_to_ubuf_align_v2"), std::string::npos);

    CCECodegen vector_codegen(ir::SectionKind::Vector);
    auto vector = vector_codegen.GenerateSingle(program, "a5");
    EXPECT_NE(vector.find("tiling_in_ub"), std::string::npos);
    EXPECT_NE(vector.find("copy_gm_to_ubuf_align_v2"), std::string::npos);
    EXPECT_NE(vector.find("copy_data_align64((uint8_t*)&tiling, (__ubuf__ uint8_t *)tiling_in_ub"), std::string::npos);
}

TEST(CCECodegenTest, RejectsUnprojectedOrWrongTargetSections)
{
    auto body = std::make_shared<const ir::ReturnStmt>(ir::Span::Unknown());
    auto cube_section = std::make_shared<const ir::SectionStmt>(ir::SectionKind::Cube, body, ir::Span::Unknown());
    CCECodegen vector_codegen(ir::SectionKind::Vector);
    EXPECT_THROW((void)vector_codegen.GenerateSingle(MakeProgram(cube_section), "a5"), pypto::ir::InternalError);

    auto vf_section = std::make_shared<const ir::SectionStmt>(ir::SectionKind::VF, body, ir::Span::Unknown());
    CCECodegen cube_codegen(ir::SectionKind::Cube);
    EXPECT_THROW((void)cube_codegen.GenerateSingle(MakeProgram(vf_section), "a5"), pypto::ir::InternalError);
}

TEST(CCECodegenTest, RejectsProgramWithoutDebugInfo)
{
    auto body = std::make_shared<const ir::ReturnStmt>(ir::Span::Unknown());
    auto function = std::make_shared<const ir::Function>("kernel", std::vector<ir::VarPtr>{},
                                                         std::vector<ir::TypePtr>{}, body, ir::Span::Unknown(),
                                                         ir::FunctionType::IN_CORE, true);
    auto program = std::make_shared<const ir::Program>(std::vector<ir::FunctionPtr>{function}, "test_program",
                                                       ir::Span::Unknown());

    CCECodegen codegen(ir::SectionKind::Vector);
    EXPECT_THROW((void)codegen.GenerateSingle(program, "a5"), pypto::ir::InternalError);
}

TEST(CCECodegenHeaderTest, CoversPointerAndBasicCodegenHelpers)
{
    CCECodegen codegen(ir::SectionKind::Vector);

    EXPECT_EQ(codegen.GetTypeString(ir::DataType::FP32), "float");
    EXPECT_EQ(codegen.GetConstIntValue(MakeConstInt(7)), 7);

    auto scalar_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT32);
    auto var = MakeVar("scalar_value", scalar_type);
    EXPECT_EQ(codegen.GetVarName(var), "scalar_value");

    EXPECT_FALSE(codegen.HasPointer("tensor"));
    codegen.RegisterPointer("tensor", "tensor_ptr");
    EXPECT_TRUE(codegen.HasPointer("tensor"));
    EXPECT_EQ(codegen.GetPointer("tensor"), "tensor_ptr");
}

TEST(CCECodegenHeaderTest, CoversTileAndTensorDefDefaults)
{
    auto tile_type = std::make_shared<const ir::TileType>(std::vector<int64_t>{16, 16}, ir::DataType::FP16,
                                                          std::optional<ir::MemRefPtr>(std::nullopt),
                                                          std::optional<ir::TileView>(std::nullopt));
    auto tile_var = MakeVar("tile", tile_type);

    TileDef tile_def;
    tile_def.var = tile_var;
    tile_def.tile_type = tile_type;

    EXPECT_EQ(tile_def.var, tile_var);
    EXPECT_EQ(tile_def.tile_type, tile_type);

    auto tensor_type = std::make_shared<const ir::TensorType>(std::vector<int64_t>{32, 32}, ir::DataType::FP16,
                                                              std::optional<ir::MemRefPtr>(std::nullopt));
    auto tensor_var = MakeVar("tensor", tensor_type);

    TensorDef tensor_def;
    tensor_def.var = tensor_var;
    tensor_def.access_shape = {MakeConstInt(16), MakeConstInt(16)};

    EXPECT_EQ(tensor_def.var, tensor_var);
    EXPECT_EQ(tensor_def.access_shape.size(), 2);
    EXPECT_FALSE(tensor_def.tile_dims.has_value());
    EXPECT_FALSE(tensor_def.is_transpose);
}

TEST(CCECodegenTest, KeepsSeparateVFTilePointersForPostUpdate)
{
    CCECodegen codegen(ir::SectionKind::Vector);
    auto tile_type = std::make_shared<const ir::TileType>(std::vector<int64_t>{16, 16}, ir::DataType::FP16,
                                                          std::optional<ir::MemRefPtr>(std::nullopt),
                                                          std::optional<ir::TileView>(std::nullopt));
    auto tile = MakeVar("tile", tile_type);

    std::string base_ptr = codegen.GetOrCreateVFTilePtr(tile);
    EXPECT_EQ(codegen.GetOrCreateVFTilePtr(tile), base_ptr);
    EXPECT_NE(codegen.GetOrCreateVFTilePtr(tile, true), base_ptr);
}

TEST(CCECodegenTest, GeneratesNativeLoopJumpsAndReturn)
{
    auto scalar_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT64);
    auto bool_type = std::make_shared<const ir::ScalarType>(ir::DataType::BOOL);
    auto loop_var = MakeVar("i", scalar_type);
    auto condition = MakeVar("condition", bool_type);

    auto for_loop = std::make_shared<const ir::ForStmt>(
        loop_var, MakeConstInt(0), MakeConstInt(1), MakeConstInt(1), std::vector<ir::IterArgPtr>{},
        std::make_shared<const ir::ContinueStmt>(ir::Span::Unknown()), std::vector<ir::VarPtr>{}, ir::Span::Unknown());
    auto while_loop = std::make_shared<const ir::WhileStmt>(condition, std::vector<ir::IterArgPtr>{},
                                                            std::make_shared<const ir::BreakStmt>(ir::Span::Unknown()),
                                                            std::vector<ir::VarPtr>{}, ir::Span::Unknown());
    auto body = std::make_shared<const ir::SeqStmts>(
        std::vector<ir::StmtPtr>{for_loop, while_loop, std::make_shared<const ir::ReturnStmt>(ir::Span::Unknown())},
        ir::Span::Unknown());

    CCECodegen codegen(ir::SectionKind::Vector);
    std::string generated = codegen.GenerateSingle(MakeProgram(body, {condition}), "a5");

    EXPECT_NE(generated.find("for (int64_t i"), std::string::npos);
    EXPECT_NE(generated.find("continue;"), std::string::npos);
    EXPECT_NE(generated.find("while (condition"), std::string::npos);
    EXPECT_NE(generated.find("break;"), std::string::npos);
    EXPECT_NE(generated.find("return;"), std::string::npos);
}

TEST(CCECodegenTest, WritesBackLoopCarriedValueBeforeContinue)
{
    auto scalar_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT64);
    auto loop_var = MakeVar("i", scalar_type);
    auto iter_arg = std::make_shared<const ir::IterArg>("acc", scalar_type, MakeConstInt(0), ir::Span::Unknown());
    auto return_var = MakeVar("acc_out", scalar_type);
    auto updated = MakeVar("acc_updated", scalar_type);
    auto update = std::make_shared<const ir::AssignStmt>(updated, MakeConstInt(7), ir::Span::Unknown());
    auto loop_body = std::make_shared<const ir::SeqStmts>(
        std::vector<ir::StmtPtr>{
            update, std::make_shared<const ir::ContinueStmt>(std::vector<ir::ExprPtr>{updated}, ir::Span::Unknown())},
        ir::Span::Unknown());
    auto for_loop = std::make_shared<const ir::ForStmt>(loop_var, MakeConstInt(0), MakeConstInt(2), MakeConstInt(1),
                                                        std::vector<ir::IterArgPtr>{iter_arg}, loop_body,
                                                        std::vector<ir::VarPtr>{return_var}, ir::Span::Unknown());

    CCECodegen codegen(ir::SectionKind::Vector);
    std::string generated = codegen.GenerateSingle(MakeProgram(for_loop), "a5");

    EXPECT_NE(generated.find("acc = acc_updated;"), std::string::npos);
    EXPECT_NE(generated.find("continue;"), std::string::npos);
}

TEST(CCECodegenTest, SnapshotsCyclicCarriedValuesBeforeForJump)
{
    auto scalar_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT64);
    for (bool is_break : {false, true}) {
        SCOPED_TRACE(is_break ? "break" : "continue");
        auto left = std::make_shared<const ir::IterArg>("left", scalar_type, MakeConstInt(1), ir::Span::Unknown());
        auto right = std::make_shared<const ir::IterArg>("right", scalar_type, MakeConstInt(2), ir::Span::Unknown());
        auto alias = MakeVar("old_left", scalar_type);
        auto copy = std::make_shared<const ir::AssignStmt>(alias, left->iterVar_, ir::Span::Unknown());
        std::vector<ir::ExprPtr> values{right->iterVar_, alias};
        ir::StmtPtr jump;
        if (is_break) {
            jump = std::make_shared<const ir::BreakStmt>(values, ir::Span::Unknown());
        } else {
            jump = std::make_shared<const ir::ContinueStmt>(values, ir::Span::Unknown());
        }
        auto body = std::make_shared<const ir::SeqStmts>(std::vector<ir::StmtPtr>{copy, jump}, ir::Span::Unknown());
        auto loop = std::make_shared<const ir::ForStmt>(
            MakeVar("i", scalar_type), MakeConstInt(0), MakeConstInt(3), MakeConstInt(1),
            std::vector<ir::IterArgPtr>{left, right}, body,
            std::vector<ir::VarPtr>{MakeVar("left_out", scalar_type), MakeVar("right_out", scalar_type)},
            ir::Span::Unknown());

        CCECodegen codegen(ir::SectionKind::Vector);
        auto generated = codegen.GenerateSingle(MakeProgram(loop), "a5");
        auto save_left = generated.find("int64_t left__next = right;");
        auto save_right = generated.find("int64_t right__next = left;");
        auto write_left = generated.find("left = left__next;");
        auto write_right = generated.find("right = right__next;");
        ASSERT_NE(save_left, std::string::npos);
        ASSERT_NE(save_right, std::string::npos);
        ASSERT_NE(write_left, std::string::npos);
        ASSERT_NE(write_right, std::string::npos);
        EXPECT_LT(save_left, write_left);
        EXPECT_LT(save_right, write_left);
        EXPECT_LT(write_right, generated.find(is_break ? "break;" : "continue;"));
    }
}

TEST(CCECodegenTest, SnapshotsWhileCarriedExpressionsBeforeWritingSlots)
{
    auto scalar_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT64);
    auto left = std::make_shared<const ir::IterArg>("left", scalar_type, MakeConstInt(1), ir::Span::Unknown());
    auto right = std::make_shared<const ir::IterArg>("right", scalar_type, MakeConstInt(2), ir::Span::Unknown());
    auto sum = std::make_shared<const ir::Add>(left->iterVar_, right->iterVar_, ir::DataType::INT64,
                                               ir::Span::Unknown());
    auto jump = std::make_shared<const ir::ContinueStmt>(std::vector<ir::ExprPtr>{right->iterVar_, sum},
                                                         ir::Span::Unknown());
    auto loop = std::make_shared<const ir::WhileStmt>(
        std::make_shared<const ir::ConstBool>(true, ir::Span::Unknown()), std::vector<ir::IterArgPtr>{left, right},
        jump, std::vector<ir::VarPtr>{MakeVar("left_out", scalar_type), MakeVar("right_out", scalar_type)},
        ir::Span::Unknown());

    CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(loop), "a5");
    auto save = generated.find("int64_t right__next = (left + right);");
    auto write = generated.find("left = left__next;");
    ASSERT_NE(save, std::string::npos);
    ASSERT_NE(write, std::string::npos);
    EXPECT_LT(save, write);
}

TEST(CCECodegenTest, SnapshotsCarriedTupleStorageBeforeWritingSlots)
{
    auto left_value = std::make_shared<const ir::MakeTuple>(std::vector<ir::ExprPtr>{MakeConstInt(1), MakeConstInt(2)},
                                                            ir::Span::Unknown());
    auto right_value = std::make_shared<const ir::MakeTuple>(std::vector<ir::ExprPtr>{MakeConstInt(3), MakeConstInt(4)},
                                                             ir::Span::Unknown());
    auto type = left_value->GetType();
    auto left_init = MakeVar("left_init", type);
    auto right_init = MakeVar("right_init", type);
    auto left = std::make_shared<const ir::IterArg>("left", type, left_init, ir::Span::Unknown());
    auto right = std::make_shared<const ir::IterArg>("right", type, right_init, ir::Span::Unknown());
    auto jump = std::make_shared<const ir::ContinueStmt>(std::vector<ir::ExprPtr>{right->iterVar_, left->iterVar_},
                                                         ir::Span::Unknown());
    auto loop = std::make_shared<const ir::WhileStmt>(
        std::make_shared<const ir::ConstBool>(true, ir::Span::Unknown()), std::vector<ir::IterArgPtr>{left, right},
        jump, std::vector<ir::VarPtr>{MakeVar("left_out", type), MakeVar("right_out", type)}, ir::Span::Unknown());
    auto body = std::make_shared<const ir::SeqStmts>(
        std::vector<ir::StmtPtr>{std::make_shared<const ir::AssignStmt>(left_init, left_value, ir::Span::Unknown()),
                                 std::make_shared<const ir::AssignStmt>(right_init, right_value, ir::Span::Unknown()),
                                 loop},
        ir::Span::Unknown());

    CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a5");
    auto save = generated.find("right__next[1] = left[1];");
    auto write = generated.find("left[0] = left__next[0];");
    ASSERT_NE(save, std::string::npos);
    ASSERT_NE(write, std::string::npos);
    EXPECT_LT(save, write);
    EXPECT_NE(generated.find("right[1] = right__next[1];"), std::string::npos);
}

TEST(CCECodegenTest, SnapshotsSingleAggregateBeforePermutingItsLeaves)
{
    auto initial = std::make_shared<const ir::MakeTuple>(std::vector<ir::ExprPtr>{MakeConstInt(1), MakeConstInt(2)},
                                                         ir::Span::Unknown());
    auto type = ir::As<ir::TupleType>(initial->GetType());
    auto init = MakeVar("init", type);
    auto state = std::make_shared<const ir::IterArg>("state", type, init, ir::Span::Unknown());
    auto first = std::make_shared<const ir::GetItemExpr>(state->iterVar_, MakeConstInt(0), ir::Span::Unknown());
    auto second = std::make_shared<const ir::GetItemExpr>(state->iterVar_, MakeConstInt(1), ir::Span::Unknown());
    auto swapped = std::make_shared<const ir::MakeTuple>(std::vector<ir::ExprPtr>{second, first}, ir::Span::Unknown());
    auto next = MakeVar("next", type);
    auto jump = std::make_shared<const ir::ContinueStmt>(std::vector<ir::ExprPtr>{next}, ir::Span::Unknown());
    auto loop_body = std::make_shared<const ir::SeqStmts>(
        std::vector<ir::StmtPtr>{std::make_shared<const ir::AssignStmt>(next, swapped, ir::Span::Unknown()), jump},
        ir::Span::Unknown());
    auto loop = std::make_shared<const ir::WhileStmt>(
        std::make_shared<const ir::ConstBool>(true, ir::Span::Unknown()), std::vector<ir::IterArgPtr>{state}, loop_body,
        std::vector<ir::VarPtr>{MakeVar("result", type)}, ir::Span::Unknown());
    auto body = std::make_shared<const ir::SeqStmts>(
        std::vector<ir::StmtPtr>{std::make_shared<const ir::AssignStmt>(init, initial, ir::Span::Unknown()), loop},
        ir::Span::Unknown());
    auto debug_info = std::make_shared<ir::IRDebugInfo>();
    debug_info->RegisterTupleTypeInfo(type, {ir::TupleTypeKind::NAMED_TUPLE, std::nullopt, {"first", "second"}});

    CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body, {}, debug_info), "a5");
    auto save = generated.find("int64_t state__next__item_1 = state__item_0;");
    auto write = generated.find("state__item_0 = state__next__item_0;");
    ASSERT_NE(save, std::string::npos);
    ASSERT_NE(write, std::string::npos);
    EXPECT_LT(save, write);
    EXPECT_NE(generated.find("state__item_1 = state__next__item_1;"), std::string::npos);
}

TEST(CCECodegenTest, PreservesSingleIterationLoopForAddrReg)
{
    auto scalar_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT64);
    auto loop_var = MakeVar("i", scalar_type);
    auto addr_reg = MakeVar("addr", scalar_type);
    auto create_addr = std::make_shared<const ir::Call>("vf.create_addr_reg", std::vector<ir::ExprPtr>{MakeConstInt(1)},
                                                        scalar_type, ir::Span::Unknown());
    auto assign = std::make_shared<const ir::AssignStmt>(addr_reg, create_addr, ir::Span::Unknown());
    auto for_loop = std::make_shared<const ir::ForStmt>(loop_var, MakeConstInt(0), MakeConstInt(1), MakeConstInt(1),
                                                        std::vector<ir::IterArgPtr>{}, assign,
                                                        std::vector<ir::VarPtr>{}, ir::Span::Unknown());

    CCECodegen codegen(ir::SectionKind::Vector);
    std::string generated = codegen.GenerateSingle(MakeProgram(for_loop), "a5");

    EXPECT_NE(generated.find("for (int64_t i"), std::string::npos);
    EXPECT_NE(generated.find("AddrReg addr"), std::string::npos);
    EXPECT_NE(generated.find("vag_b32(1)"), std::string::npos);
}

TEST(CCECodegenTest, UsesOneBackingArrayForDynamicAndStaticTupleReads)
{
    auto scalar_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT64);
    auto index = MakeVar("index", scalar_type);
    auto values = std::make_shared<const ir::MakeTuple>(std::vector<ir::ExprPtr>{MakeConstInt(11), MakeConstInt(22)},
                                                        ir::Span::Unknown());
    auto tuple_var = MakeVar("values", values->GetType());
    auto dynamic_value = MakeVar("dynamic_value", scalar_type);
    auto static_value = MakeVar("static_value", scalar_type);

    auto tuple_assign = std::make_shared<const ir::AssignStmt>(tuple_var, values, ir::Span::Unknown());
    auto dynamic_read = std::make_shared<const ir::AssignStmt>(
        dynamic_value, std::make_shared<const ir::GetItemExpr>(tuple_var, index, ir::Span::Unknown()),
        ir::Span::Unknown());
    auto static_read = std::make_shared<const ir::AssignStmt>(
        static_value, std::make_shared<const ir::GetItemExpr>(tuple_var, MakeConstInt(0), ir::Span::Unknown()),
        ir::Span::Unknown());
    auto body = std::make_shared<const ir::SeqStmts>(std::vector<ir::StmtPtr>{tuple_assign, dynamic_read, static_read},
                                                     ir::Span::Unknown());

    CCECodegen codegen(ir::SectionKind::Vector);
    std::string generated = codegen.GenerateSingle(MakeProgram(body, {index}), "a5");

    EXPECT_EQ(generated.find("const int64_t values"), std::string::npos);
    EXPECT_NE(generated.find("int64_t values"), std::string::npos);
    EXPECT_NE(generated.find("static_cast<int64_t>(11)"), std::string::npos);
    EXPECT_NE(generated.find("[index"), std::string::npos);
    EXPECT_NE(generated.find("[0]"), std::string::npos);
}

TEST(CCECodegenTest, SharesBackingArrayAcrossTupleAliases)
{
    auto scalar_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT64);
    auto index = MakeVar("index", scalar_type);
    auto values = std::make_shared<const ir::MakeTuple>(std::vector<ir::ExprPtr>{MakeConstInt(11), MakeConstInt(22)},
                                                        ir::Span::Unknown());
    auto first = MakeVar("first", values->GetType());
    auto second = MakeVar("second", values->GetType());
    auto selected = MakeVar("selected", scalar_type);

    auto first_assign = std::make_shared<const ir::AssignStmt>(first, values, ir::Span::Unknown());
    auto second_assign = std::make_shared<const ir::AssignStmt>(second, first, ir::Span::Unknown());
    auto read = std::make_shared<const ir::AssignStmt>(
        selected, std::make_shared<const ir::GetItemExpr>(second, index, ir::Span::Unknown()), ir::Span::Unknown());
    auto body = std::make_shared<const ir::SeqStmts>(std::vector<ir::StmtPtr>{first_assign, second_assign, read},
                                                     ir::Span::Unknown());

    CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body, {index}), "a5");

    EXPECT_EQ(generated.find("const int64_t first"), std::string::npos);
    EXPECT_EQ(CountOccurrences(generated, "int64_t first"), 1);
    EXPECT_EQ(generated.find("int64_t second"), std::string::npos);
    EXPECT_NE(generated.find("first"), std::string::npos);
    EXPECT_NE(generated.find("[index"), std::string::npos);
}

TEST(CCECodegenTest, ClearsTupleBackingArraysBetweenGenerations)
{
    auto scalar_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT64);
    auto index = MakeVar("index", scalar_type);
    auto values = std::make_shared<const ir::MakeTuple>(std::vector<ir::ExprPtr>{MakeConstInt(1), MakeConstInt(2)},
                                                        ir::Span::Unknown());
    auto tuple_var = MakeVar("values", values->GetType());
    auto selected = MakeVar("selected", scalar_type);
    auto tuple_assign = std::make_shared<const ir::AssignStmt>(tuple_var, values, ir::Span::Unknown());
    auto read = std::make_shared<const ir::AssignStmt>(
        selected, std::make_shared<const ir::GetItemExpr>(tuple_var, index, ir::Span::Unknown()), ir::Span::Unknown());
    auto body = std::make_shared<const ir::SeqStmts>(std::vector<ir::StmtPtr>{tuple_assign, read}, ir::Span::Unknown());
    auto program = MakeProgram(body, {index});

    CCECodegen codegen(ir::SectionKind::Vector);
    auto first = codegen.GenerateSingle(program, "a5");
    auto second = codegen.GenerateSingle(program, "a5");

    EXPECT_EQ(first.find("const int64_t values"), std::string::npos);
    EXPECT_EQ(second.find("const int64_t values"), std::string::npos);
    EXPECT_NE(first.find("int64_t values"), std::string::npos);
    EXPECT_NE(second.find("int64_t values"), std::string::npos);
}

TEST(CCECodegenTest, ClearsTileTypeStateBetweenGenerations)
{
    auto tile_type = std::make_shared<const ir::TileType>(std::vector<int64_t>{16, 16}, ir::DataType::FP16,
                                                          std::optional<ir::MemRefPtr>(std::nullopt),
                                                          std::optional<ir::TileView>(std::nullopt));
    auto tile = MakeVar("tile", tile_type);
    auto make_tile = std::make_shared<const ir::Call>("block.make_tile", std::vector<ir::ExprPtr>{}, tile_type,
                                                      ir::Span::Unknown());
    auto body = std::make_shared<const ir::AssignStmt>(tile, make_tile, ir::Span::Unknown());
    auto program = MakeProgram(body);

    CCECodegen codegen(ir::SectionKind::Vector);
    auto first = codegen.GenerateSingle(program, "a5");
    auto second = codegen.GenerateSingle(program, "a5");

    constexpr const char* tile_type_declaration = "using tile_Type = ";
    EXPECT_NE(first.find(tile_type_declaration), std::string::npos);
    EXPECT_NE(second.find(tile_type_declaration), std::string::npos);
}

TEST(CCECodegenTest, MaterializesHomogeneousTileTuple)
{
    auto index_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT64);
    auto index = MakeVar("index", index_type);
    auto tile_type = std::make_shared<const ir::TileType>(std::vector<int64_t>{16, 16}, ir::DataType::FP16,
                                                          std::optional<ir::MemRefPtr>(std::nullopt),
                                                          std::optional<ir::TileView>(std::nullopt));
    auto tile0 = MakeVar("tile0", tile_type);
    auto tile1 = MakeVar("tile1", tile_type);
    auto make_tile0 = std::make_shared<const ir::Call>("block.make_tile", std::vector<ir::ExprPtr>{}, tile_type,
                                                       ir::Span::Unknown());
    auto make_tile1 = std::make_shared<const ir::Call>("block.make_tile", std::vector<ir::ExprPtr>{}, tile_type,
                                                       ir::Span::Unknown());
    auto values = std::make_shared<const ir::MakeTuple>(std::vector<ir::ExprPtr>{tile0, tile1}, ir::Span::Unknown());
    auto tiles = MakeVar("tiles", values->GetType());
    auto selected = MakeVar("selected", tile_type);

    auto body = std::make_shared<const ir::SeqStmts>(
        std::vector<ir::StmtPtr>{
            std::make_shared<const ir::AssignStmt>(tile0, make_tile0, ir::Span::Unknown()),
            std::make_shared<const ir::AssignStmt>(tile1, make_tile1, ir::Span::Unknown()),
            std::make_shared<const ir::AssignStmt>(tiles, values, ir::Span::Unknown()),
            std::make_shared<const ir::AssignStmt>(
                selected, std::make_shared<const ir::GetItemExpr>(tiles, index, ir::Span::Unknown()),
                ir::Span::Unknown()),
        },
        ir::Span::Unknown());

    CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body, {index}), "a5");

    EXPECT_NE(generated.find("tiles"), std::string::npos);
    EXPECT_NE(generated.find("[] = {tile0"), std::string::npos);
}

TEST(CCECodegenTest, HandlesStaticTupleGetItemWithoutBackingArray)
{
    auto tuple = std::make_shared<const ir::MakeTuple>(std::vector<ir::ExprPtr>{MakeConstInt(11), MakeConstInt(22)},
                                                       ir::Span::Unknown());
    CCECodegen codegen(ir::SectionKind::Vector);

    auto item = std::make_shared<const ir::GetItemExpr>(tuple, MakeConstInt(1), ir::Span::Unknown());
    EXPECT_EQ(codegen.GetExprAsCode(item), "22");
}

TEST(CCECodegenTest, EmitsPythonSemanticsForFloorDivAndMod)
{
    auto scalar_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT64);
    auto dividend = MakeVar("dividend", scalar_type);
    auto divisor = MakeVar("divisor", scalar_type);
    auto floor_div = std::make_shared<const ir::FloorDiv>(dividend, divisor, ir::DataType::INT64, ir::Span::Unknown());
    auto floor_mod = std::make_shared<const ir::FloorMod>(dividend, divisor, ir::DataType::INT64, ir::Span::Unknown());
    CCECodegen codegen(ir::SectionKind::Vector);

    const std::string div_code = codegen.GetExprAsCode(floor_div);
    EXPECT_NE(div_code.find("__pypto_quot = __pypto_lhs / __pypto_rhs"), std::string::npos);
    EXPECT_NE(div_code.find("((__pypto_lhs < 0) != (__pypto_rhs < 0)) &&"), std::string::npos);
    EXPECT_NE(div_code.find("(__pypto_lhs % __pypto_rhs != 0)"), std::string::npos);
    EXPECT_EQ(div_code.find("__pypto_rem ="), std::string::npos);
    EXPECT_EQ(CountOccurrences(div_code, "(dividend)"), 1U);
    EXPECT_EQ(CountOccurrences(div_code, "(divisor)"), 1U);

    const std::string mod_code = codegen.GetExprAsCode(floor_mod);
    EXPECT_NE(mod_code.find("__pypto_rem +"), std::string::npos);
    EXPECT_NE(mod_code.find("(__pypto_rem < 0) != (__pypto_rhs < 0)"), std::string::npos);
    EXPECT_EQ(CountOccurrences(mod_code, "(dividend)"), 1U);
    EXPECT_EQ(CountOccurrences(mod_code, "(divisor)"), 1U);

    auto float_type = std::make_shared<const ir::ScalarType>(ir::DataType::FP32);
    auto float_dividend = MakeVar("float_dividend", float_type);
    auto float_divisor = MakeVar("float_divisor", float_type);
    auto float_floor_div = std::make_shared<const ir::FloorDiv>(float_dividend, float_divisor, ir::DataType::FP32,
                                                                ir::Span::Unknown());
    auto float_floor_mod = std::make_shared<const ir::FloorMod>(float_dividend, float_divisor, ir::DataType::FP32,
                                                                ir::Span::Unknown());

    const std::string float_div_code = codegen.GetExprAsCode(float_floor_div);
    EXPECT_NE(float_div_code.find("__pypto_div = (__pypto_lhs - __pypto_mod) / __pypto_rhs"), std::string::npos);
    EXPECT_NE(float_div_code.find("__pypto_floor_result = __pypto_floor_input"), std::string::npos);
    EXPECT_NE(float_div_code.find("__pypto_floor_exp = __pypto_floor_abs >> 23"), std::string::npos);
    EXPECT_NE(float_div_code.find("__pypto_floor_frac_mask = (1u << (150u - __pypto_floor_exp)) - 1u"),
              std::string::npos);
    EXPECT_EQ(float_div_code.find("__pypto_floor_exp == 0xffu"), std::string::npos);
    EXPECT_EQ(float_div_code.find("__pypto_div_mod"), std::string::npos);
    EXPECT_EQ(CountOccurrences(float_div_code, "while (__ex > __ey)"), 1U);
    EXPECT_EQ(float_div_code.find("__floorf("), std::string::npos);
    EXPECT_NE(float_div_code.find("__pypto_div -= 1.0f"), std::string::npos);
    EXPECT_EQ(float_div_code.find("__pypto_mod += __pypto_rhs"), std::string::npos);
    EXPECT_EQ(CountOccurrences(float_div_code, "(float_dividend)"), 1U);
    EXPECT_EQ(CountOccurrences(float_div_code, "(float_divisor)"), 1U);

    const std::string float_mod_code = codegen.GetExprAsCode(float_floor_mod);
    EXPECT_NE(float_mod_code.find("__pypto_mod += __pypto_rhs"), std::string::npos);
    EXPECT_NE(float_mod_code.find("reinterpret_cast<float&>(__result)"), std::string::npos);
    EXPECT_EQ(CountOccurrences(float_mod_code, "(float_dividend)"), 1U);
    EXPECT_EQ(CountOccurrences(float_mod_code, "(float_divisor)"), 1U);
}

TEST(CCECodegenTest, KeepsNativeFloorDivAndModForUnsignedIntegers)
{
    auto scalar_type = std::make_shared<const ir::ScalarType>(ir::DataType::UINT64);
    auto dividend = MakeVar("dividend", scalar_type);
    auto divisor = MakeVar("divisor", scalar_type);
    auto floor_div = std::make_shared<const ir::FloorDiv>(dividend, divisor, ir::DataType::UINT64, ir::Span::Unknown());
    auto floor_mod = std::make_shared<const ir::FloorMod>(dividend, divisor, ir::DataType::UINT64, ir::Span::Unknown());
    CCECodegen codegen(ir::SectionKind::Vector);

    EXPECT_EQ(codegen.GetExprAsCode(floor_div), "(dividend / divisor)");
    EXPECT_EQ(codegen.GetExprAsCode(floor_mod), "(dividend % divisor)");
}

TEST(CCECodegenTest, EmitsArrayAccessForUnmaterializedTupleVar)
{
    auto scalar_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT64);
    auto tuple_type = std::make_shared<const ir::TupleType>(
        std::vector<ir::TypePtr>{scalar_type, scalar_type, scalar_type});
    auto tuple_var = MakeVar("values", tuple_type);
    auto item = std::make_shared<const ir::GetItemExpr>(tuple_var, MakeConstInt(1), ir::Span::Unknown());

    auto body = std::make_shared<const ir::ReturnStmt>(ir::Span::Unknown());
    auto program = MakeProgram(body);
    CCECodegen codegen(ir::SectionKind::Vector);
    (void)codegen.GenerateSingle(program, "a5");
    EXPECT_EQ(codegen.GetExprAsCode(item), "values[1]");
}

TEST(CCECodegenTest, EmitsFieldAccessForUnmaterializedStructVar)
{
    auto scalar_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT64);
    auto tuple_type = std::make_shared<const ir::TupleType>(std::vector<ir::TypePtr>{scalar_type, scalar_type});
    auto tuple_var = MakeVar("config", tuple_type);
    auto item = std::make_shared<const ir::GetItemExpr>(tuple_var, MakeConstInt(1), ir::Span::Unknown());
    auto debug_info = std::make_shared<ir::IRDebugInfo>();
    debug_info->RegisterTupleTypeInfo(tuple_type, {ir::TupleTypeKind::STRUCT, std::string("Config"), {"rows", "cols"}});

    auto body = std::make_shared<const ir::ReturnStmt>(ir::Span::Unknown());
    auto program = MakeProgram(body, {}, debug_info);
    CCECodegen codegen(ir::SectionKind::Vector);
    (void)codegen.GenerateSingle(program, "a5");
    EXPECT_EQ(codegen.GetExprAsCode(item), "config.cols");
}

TEST(CCECodegenTest, RejectsDynamicTupleWithoutBackingArray)
{
    auto scalar_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT64);
    auto index = MakeVar("index", scalar_type);
    auto tuple = std::make_shared<const ir::MakeTuple>(std::vector<ir::ExprPtr>{MakeConstInt(1), MakeConstInt(2)},
                                                       ir::Span::Unknown());
    auto unowned = std::make_shared<const ir::GetItemExpr>(tuple, index, ir::Span::Unknown());

    CCECodegen codegen(ir::SectionKind::Vector);
    EXPECT_THROW((void)codegen.GetExprAsCode(unowned), std::exception);
}

TEST(CCECodegenTest, EmitsUnusedIfPhiWithoutDce)
{
    auto bool_type = std::make_shared<const ir::ScalarType>(ir::DataType::BOOL);
    auto scalar_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT64);
    auto condition = MakeVar("condition", bool_type);
    auto unused_phi = MakeVar("unused_phi", scalar_type);
    auto then_yield = std::make_shared<const ir::YieldStmt>(std::vector<ir::ExprPtr>{MakeConstInt(1)},
                                                            ir::Span::Unknown());
    auto else_yield = std::make_shared<const ir::YieldStmt>(std::vector<ir::ExprPtr>{MakeConstInt(2)},
                                                            ir::Span::Unknown());
    auto if_stmt = std::make_shared<const ir::IfStmt>(condition, then_yield, std::optional<ir::StmtPtr>(else_yield),
                                                      std::vector<ir::VarPtr>{unused_phi}, ir::Span::Unknown());

    CCECodegen codegen(ir::SectionKind::Vector);
    std::string generated = codegen.GenerateSingle(MakeProgram(if_stmt, {condition}), "a5");

    EXPECT_NE(generated.find("if (condition"), std::string::npos);
    EXPECT_NE(generated.find("unused_phi"), std::string::npos);
    EXPECT_NE(generated.find("} else {"), std::string::npos);
}

TEST(CCECodegenTest, MergesArrayTuplePhiThroughOneBackingArray)
{
    auto bool_type = std::make_shared<const ir::ScalarType>(ir::DataType::BOOL);
    auto scalar_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT64);
    auto condition = MakeVar("condition", bool_type);
    auto index = MakeVar("index", scalar_type);
    auto left_value = std::make_shared<const ir::MakeTuple>(std::vector<ir::ExprPtr>{MakeConstInt(1), MakeConstInt(2)},
                                                            ir::Span::Unknown());
    auto right_value = std::make_shared<const ir::MakeTuple>(std::vector<ir::ExprPtr>{MakeConstInt(3), MakeConstInt(4)},
                                                             ir::Span::Unknown());
    auto left = MakeVar("left", left_value->GetType());
    auto right = MakeVar("right", right_value->GetType());
    auto selected = MakeVar("selected", left_value->GetType());
    auto picked = MakeVar("picked", scalar_type);
    auto then_yield = std::make_shared<const ir::YieldStmt>(std::vector<ir::ExprPtr>{left}, ir::Span::Unknown());
    auto else_yield = std::make_shared<const ir::YieldStmt>(std::vector<ir::ExprPtr>{right}, ir::Span::Unknown());
    auto if_stmt = std::make_shared<const ir::IfStmt>(condition, then_yield, std::optional<ir::StmtPtr>(else_yield),
                                                      std::vector<ir::VarPtr>{selected}, ir::Span::Unknown());
    auto read_selected = std::make_shared<const ir::AssignStmt>(
        picked, std::make_shared<const ir::GetItemExpr>(selected, index, ir::Span::Unknown()), ir::Span::Unknown());
    auto body = std::make_shared<const ir::SeqStmts>(
        std::vector<ir::StmtPtr>{std::make_shared<const ir::AssignStmt>(left, left_value, ir::Span::Unknown()),
                                 std::make_shared<const ir::AssignStmt>(right, right_value, ir::Span::Unknown()),
                                 if_stmt, read_selected},
        ir::Span::Unknown());

    CCECodegen codegen(ir::SectionKind::Vector);
    std::string generated = codegen.GenerateSingle(MakeProgram(body, {condition, index}), "a5");

    EXPECT_EQ(CountOccurrences(generated, "int64_t selected[2];"), 1);
    EXPECT_NE(generated.find("selected[0] = left[0];"), std::string::npos);
    EXPECT_NE(generated.find("selected[1] = left[1];"), std::string::npos);
    EXPECT_NE(generated.find("selected[0] = right[0];"), std::string::npos);
    EXPECT_NE(generated.find("selected[1] = right[1];"), std::string::npos);
    EXPECT_NE(generated.find("selected[index]"), std::string::npos);
    EXPECT_EQ(generated.find("selected__item_0"), std::string::npos);
}

TEST(CCECodegenTest, DoesNotMaterializeHomogeneousTupleOfTuples)
{
    auto inner_value = std::make_shared<const ir::MakeTuple>(std::vector<ir::ExprPtr>{MakeConstInt(1), MakeConstInt(2)},
                                                             ir::Span::Unknown());
    auto outer_value = std::make_shared<const ir::MakeTuple>(std::vector<ir::ExprPtr>{inner_value, inner_value},
                                                             ir::Span::Unknown());
    auto nested = MakeVar("nested", outer_value->GetType());
    auto body = std::make_shared<const ir::SeqStmts>(
        std::vector<ir::StmtPtr>{std::make_shared<const ir::AssignStmt>(nested, outer_value, ir::Span::Unknown())},
        ir::Span::Unknown());

    CCECodegen codegen(ir::SectionKind::Vector);
    std::string generated = codegen.GenerateSingle(MakeProgram(body), "a5");

    EXPECT_EQ(generated.find("nested[]"), std::string::npos);
    EXPECT_EQ(generated.find("nested[2]"), std::string::npos);
}

TEST(CCECodegenTest, FlattensAggregateTuplePhiIntoLeafSlots)
{
    auto bool_type = std::make_shared<const ir::ScalarType>(ir::DataType::BOOL);
    auto scalar_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT64);
    auto condition = MakeVar("condition", bool_type);
    auto aggregate_type = std::make_shared<const ir::TupleType>(std::vector<ir::TypePtr>{scalar_type, bool_type});
    auto left_value = std::make_shared<const ir::MakeTuple>(
        std::vector<ir::ExprPtr>{MakeConstInt(1), std::make_shared<const ir::ConstBool>(true, ir::Span::Unknown())},
        ir::Span::Unknown());
    auto right_value = std::make_shared<const ir::MakeTuple>(
        std::vector<ir::ExprPtr>{MakeConstInt(3), std::make_shared<const ir::ConstBool>(false, ir::Span::Unknown())},
        ir::Span::Unknown());
    auto left = MakeVar("left", aggregate_type);
    auto right = MakeVar("right", aggregate_type);
    auto selected = MakeVar("selected", aggregate_type);
    auto picked = MakeVar("picked", bool_type);
    auto then_yield = std::make_shared<const ir::YieldStmt>(std::vector<ir::ExprPtr>{left}, ir::Span::Unknown());
    auto else_yield = std::make_shared<const ir::YieldStmt>(std::vector<ir::ExprPtr>{right}, ir::Span::Unknown());
    auto if_stmt = std::make_shared<const ir::IfStmt>(condition, then_yield, std::optional<ir::StmtPtr>(else_yield),
                                                      std::vector<ir::VarPtr>{selected}, ir::Span::Unknown());
    auto read_selected = std::make_shared<const ir::AssignStmt>(
        picked, std::make_shared<const ir::GetItemExpr>(selected, MakeConstInt(1), ir::Span::Unknown()),
        ir::Span::Unknown());
    auto body = std::make_shared<const ir::SeqStmts>(
        std::vector<ir::StmtPtr>{std::make_shared<const ir::AssignStmt>(left, left_value, ir::Span::Unknown()),
                                 std::make_shared<const ir::AssignStmt>(right, right_value, ir::Span::Unknown()),
                                 if_stmt, read_selected},
        ir::Span::Unknown());
    auto debug_info = std::make_shared<ir::IRDebugInfo>();
    debug_info->RegisterTupleTypeInfo(aggregate_type,
                                      {ir::TupleTypeKind::NAMED_TUPLE, std::nullopt, {"first", "second"}});

    CCECodegen codegen(ir::SectionKind::Vector);
    std::string generated = codegen.GenerateSingle(MakeProgram(body, {condition}, debug_info), "a5");

    EXPECT_EQ(generated.find("selected[2]"), std::string::npos);
    EXPECT_NE(generated.find("int64_t selected__item_0;"), std::string::npos);
    EXPECT_NE(generated.find("bool selected__item_1;"), std::string::npos);
    EXPECT_NE(generated.find("selected__item_0 = 1;"), std::string::npos);
    EXPECT_NE(generated.find("selected__item_1 = false;"), std::string::npos);
}

TEST(CCECodegenTest, WritesBackWhileCarriedValueBeforeBreak)
{
    auto bool_type = std::make_shared<const ir::ScalarType>(ir::DataType::BOOL);
    auto scalar_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT64);
    auto condition = MakeVar("condition", bool_type);
    auto iter_arg = std::make_shared<const ir::IterArg>("acc", scalar_type, MakeConstInt(0), ir::Span::Unknown());
    auto return_var = MakeVar("acc_out", scalar_type);
    auto updated = MakeVar("acc_updated", scalar_type);
    auto update = std::make_shared<const ir::AssignStmt>(updated, MakeConstInt(9), ir::Span::Unknown());
    auto break_stmt = std::make_shared<const ir::BreakStmt>(std::vector<ir::ExprPtr>{updated}, ir::Span::Unknown());
    auto loop_body = std::make_shared<const ir::SeqStmts>(std::vector<ir::StmtPtr>{update, break_stmt},
                                                          ir::Span::Unknown());
    auto while_loop = std::make_shared<const ir::WhileStmt>(condition, std::vector<ir::IterArgPtr>{iter_arg}, loop_body,
                                                            std::vector<ir::VarPtr>{return_var}, ir::Span::Unknown());

    CCECodegen codegen(ir::SectionKind::Vector);
    std::string generated = codegen.GenerateSingle(MakeProgram(while_loop, {condition}), "a5");

    EXPECT_NE(generated.find("int64_t acc = 0;"), std::string::npos);
    EXPECT_NE(generated.find("while (condition"), std::string::npos);
    size_t writeback = generated.find("acc_out = acc_updated;");
    size_t jump = generated.find("break;");
    ASSERT_NE(writeback, std::string::npos);
    ASSERT_NE(jump, std::string::npos);
    EXPECT_LT(writeback, jump);
}

TEST(CCECodegenTest, GeneratesTensorDescriptorAndLoadFromAccessShape)
{
    auto input = MakeTensorVar("input", {64, 128}, ir::DataType::FP16);
    auto tile_type = std::make_shared<const ir::TileType>(std::vector<int64_t>{16, 32}, ir::DataType::FP16,
                                                          std::optional<ir::MemRefPtr>(std::nullopt),
                                                          std::optional<ir::TileView>(std::nullopt));
    auto tile = MakeVar("tile", tile_type);
    auto make_tile = std::make_shared<const ir::Call>("block.make_tile", std::vector<ir::ExprPtr>{}, tile_type,
                                                      ir::Span::Unknown());
    auto tile_assign = std::make_shared<const ir::AssignStmt>(tile, make_tile, ir::Span::Unknown());
    auto offsets = std::make_shared<const ir::MakeTuple>(std::vector<ir::ExprPtr>{MakeConstInt(0), MakeConstInt(0)},
                                                         ir::Span::Unknown());
    auto load = std::make_shared<const ir::Call>("block.load", std::vector<ir::ExprPtr>{tile, input, offsets},
                                                 tile_type, ir::Span::Unknown());
    auto load_stmt = std::make_shared<const ir::EvalStmt>(load, ir::Span::Unknown());
    auto body = std::make_shared<const ir::SeqStmts>(std::vector<ir::StmtPtr>{tile_assign, load_stmt},
                                                     ir::Span::Unknown());

    CCECodegen codegen(ir::SectionKind::Vector);
    std::string generated = codegen.GenerateSingle(MakeProgram(body, {input}), "a5");

    // The declaration's dims are DYNAMIC and each access resizes them in place, so the access
    // shape shows up in the load's SetShape rather than in the hoisted type.
    EXPECT_NE(generated.find("using inputShapeDim5 = pto::TileShape2D<half, pto::DYNAMIC, pto::DYNAMIC,"),
              std::string::npos);
    EXPECT_NE(generated.find("inputStrideDim5, Layout::ND>;"), std::string::npos);
    EXPECT_NE(generated.find("using inputStrideDim5 = pto::Stride<-1, -1, -1, -1, -1>;"), std::string::npos);
    EXPECT_NE(generated.find("inputType input(input_ptr"), std::string::npos);
    EXPECT_NE(generated.find("inputStrideDim5(1, 1, 1, 128, 1)"), std::string::npos);
    EXPECT_NE(generated.find("input.SetShape<pto::GlobalTensorDim::DIM_3, pto::GlobalTensorDim::DIM_4>"),
              std::string::npos);
    EXPECT_NE(generated.find("TLOAD(tile"), std::string::npos);
    EXPECT_NE(generated.find(", input);"), std::string::npos);
}

TEST(CCECodegenTest, ResolvesTensorAliasAndTransposeLayout)
{
    auto input = MakeTensorVar("input", {64, 128}, ir::DataType::FP16);
    auto alias = MakeVar("alias", input->GetType());
    auto tile_type = std::make_shared<const ir::TileType>(std::vector<int64_t>{16, 32}, ir::DataType::FP16,
                                                          std::optional<ir::MemRefPtr>(std::nullopt),
                                                          std::optional<ir::TileView>(std::nullopt));
    auto tile = MakeVar("tile", tile_type);
    auto make_tile = std::make_shared<const ir::Call>("block.make_tile", std::vector<ir::ExprPtr>{}, tile_type,
                                                      ir::Span::Unknown());
    auto alias_assign = std::make_shared<const ir::AssignStmt>(alias, input, ir::Span::Unknown());
    auto tile_assign = std::make_shared<const ir::AssignStmt>(tile, make_tile, ir::Span::Unknown());
    auto offsets = std::make_shared<const ir::MakeTuple>(std::vector<ir::ExprPtr>{MakeConstInt(0), MakeConstInt(0)},
                                                         ir::Span::Unknown());
    std::vector<std::pair<std::string, std::any>> kwargs{{"is_transpose", true}};
    auto load = std::make_shared<const ir::Call>("block.load", std::vector<ir::ExprPtr>{tile, alias, offsets}, kwargs,
                                                 tile_type, ir::Span::Unknown());
    auto load_stmt = std::make_shared<const ir::EvalStmt>(load, ir::Span::Unknown());
    auto body = std::make_shared<const ir::SeqStmts>(std::vector<ir::StmtPtr>{alias_assign, tile_assign, load_stmt},
                                                     ir::Span::Unknown());

    CCECodegen codegen(ir::SectionKind::Vector);
    std::string generated = codegen.GenerateSingle(MakeProgram(body, {input}), "a5");

    EXPECT_NE(generated.find("inputStrideDim5, Layout::DN"), std::string::npos);
    EXPECT_NE(generated.find("inputStrideDim5(1, 1, 1, 1, 128)"), std::string::npos);
    EXPECT_EQ(generated.find("aliasShapeDim5"), std::string::npos);
    EXPECT_NE(generated.find("TLOAD(tile"), std::string::npos);
    EXPECT_NE(generated.find(", input);"), std::string::npos);
}

TEST(CCECodegenTest, HoistsAutoDeclaredVFDestinations)
{
    auto fp32_type = std::make_shared<const ir::ScalarType>(ir::DataType::FP32);
    auto int32_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT32);
    auto dst0 = MakeVar("dst0", fp32_type);
    auto dst1 = MakeVar("dst1", fp32_type);
    auto src0 = MakeVar("src0", fp32_type);
    auto src1 = MakeVar("src1", fp32_type);
    auto mask = MakeVar("mask", int32_type);
    auto add = std::make_shared<const ir::Call>("vf.add", std::vector<ir::ExprPtr>{dst0, src0, src1, mask}, fp32_type,
                                                ir::Span::Unknown());
    auto interleave = std::make_shared<const ir::Call>(
        "vf.interleave", std::vector<ir::ExprPtr>{dst0, dst1, src0, src1}, fp32_type, ir::Span::Unknown());
    auto vf_body = std::make_shared<const ir::SeqStmts>(
        std::vector<ir::StmtPtr>{std::make_shared<const ir::EvalStmt>(add, ir::Span::Unknown()),
                                 std::make_shared<const ir::EvalStmt>(interleave, ir::Span::Unknown())},
        ir::Span::Unknown());
    auto vf_section = std::make_shared<const ir::SectionStmt>(ir::SectionKind::VF, vf_body, ir::Span::Unknown());

    CCECodegen codegen(ir::SectionKind::Vector);
    std::string generated = codegen.GenerateSingle(MakeProgram(vf_section, {src0, src1, mask}), "a5");

    EXPECT_EQ(CountOccurrences(generated, "RegTensor<float> "), 2);
    size_t first_decl = generated.find("RegTensor<float>");
    size_t add_pos = generated.find("vadd(");
    size_t interleave_pos = generated.find("vintlv(");
    ASSERT_NE(first_decl, std::string::npos);
    ASSERT_NE(add_pos, std::string::npos);
    ASSERT_NE(interleave_pos, std::string::npos);
    EXPECT_LT(first_decl, add_pos);
    EXPECT_LT(first_decl, interleave_pos);
}

TEST(CCECodegenTest, HoistsDynamicVFLoopBoundAsUint16)
{
    auto scalar_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT64);
    auto loop_var = MakeVar("i", scalar_type);
    auto limit = MakeVar("limit", scalar_type);
    auto loop_body = std::make_shared<const ir::ContinueStmt>(ir::Span::Unknown());
    auto for_loop = std::make_shared<const ir::ForStmt>(loop_var, MakeConstInt(0), limit, MakeConstInt(1),
                                                        std::vector<ir::IterArgPtr>{}, loop_body,
                                                        std::vector<ir::VarPtr>{}, ir::Span::Unknown());
    auto vf_section = std::make_shared<const ir::SectionStmt>(ir::SectionKind::VF, for_loop, ir::Span::Unknown());

    CCECodegen codegen(ir::SectionKind::Vector);
    std::string generated = codegen.GenerateSingle(MakeProgram(vf_section, {limit}), "a5");

    EXPECT_NE(generated.find("const uint16_t i_ub = (uint16_t)(limit)"), std::string::npos);
    EXPECT_NE(generated.find("for (uint16_t i = 0; i < i_ub; i += 1)"), std::string::npos);
}

TEST(CCECodegenTest, EmitsKernelTileValidShapeGetters)
{
    auto tile_type = std::make_shared<const ir::TileType>(std::vector<int64_t>{16, 32}, ir::DataType::FP16,
                                                          std::optional<ir::MemRefPtr>(std::nullopt),
                                                          std::optional<ir::TileView>(std::nullopt));
    auto int64_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT64);
    auto tile = MakeVar("tile", tile_type);
    auto make_tile = std::make_shared<const ir::Call>("block.make_tile", std::vector<ir::ExprPtr>{}, tile_type,
                                                      ir::Span::Unknown());
    auto valid_shape = [&](int axis) {
        return std::make_shared<const ir::Call>("block.tile_valid_shape", std::vector<ir::ExprPtr>{tile},
                                                std::vector<std::pair<std::string, std::any>>{{"axis", axis}},
                                                int64_type, ir::Span::Unknown());
    };
    auto body = std::make_shared<const ir::SeqStmts>(
        std::vector<ir::StmtPtr>{
            std::make_shared<const ir::AssignStmt>(tile, make_tile, ir::Span::Unknown()),
            std::make_shared<const ir::AssignStmt>(MakeVar("rows", int64_type), valid_shape(0), ir::Span::Unknown()),
            std::make_shared<const ir::AssignStmt>(MakeVar("cols", int64_type), valid_shape(1), ir::Span::Unknown()),
        },
        ir::Span::Unknown());
    CCECodegen codegen(ir::SectionKind::Vector);
    std::string generated = codegen.GenerateSingle(MakeProgram(body), "a5");
    EXPECT_NE(generated.find("(int64_t)(tile.GetValidRow())"), std::string::npos);
    EXPECT_NE(generated.find("(int64_t)(tile.GetValidCol())"), std::string::npos);
}

TEST(CCECodegenTest, ComputesTensorOffsetAndRejectsRankMismatch)
{
    auto tensor_type = std::make_shared<const ir::TensorType>(std::vector<int64_t>{2, 3, 4}, ir::DataType::FP16,
                                                              std::optional<ir::MemRefPtr>(std::nullopt));
    auto offsets = std::make_shared<const ir::MakeTuple>(
        std::vector<ir::ExprPtr>{MakeConstInt(1), MakeConstInt(2), MakeConstInt(3)}, ir::Span::Unknown());
    auto short_offsets = std::make_shared<const ir::MakeTuple>(
        std::vector<ir::ExprPtr>{MakeConstInt(1), MakeConstInt(2)}, ir::Span::Unknown());

    CCECodegen codegen(ir::SectionKind::Vector);

    EXPECT_EQ(codegen.ComputeTensorOffset(tensor_type, offsets), "(1 * (3 * 4) + 2 * (4) + 3)");
    EXPECT_THROW((void)codegen.ComputeTensorOffset(tensor_type, short_offsets), std::exception);
}

TEST(CCECodegenTest, ComputesNzTensorOffsetsFor2DAndHighDimensionalShapes)
{
    auto make_nz_type = [](const std::vector<int64_t>& shape, ir::DataType dtype = ir::DataType::FP16) {
        auto ptr = MakeVar("nz_base", std::make_shared<const ir::PtrType>(dtype));
        ir::TensorView view({}, ir::TensorLayout::NZ, ptr);
        return std::make_shared<const ir::TensorType>(shape, dtype, std::optional<ir::MemRefPtr>(std::nullopt),
                                                      std::optional<ir::TensorView>(view));
    };

    auto offsets_2d = std::make_shared<const ir::MakeTuple>(
        std::vector<ir::ExprPtr>{MakeConstInt(16), MakeConstInt(32)}, ir::Span::Unknown());
    auto offsets_4d = std::make_shared<const ir::MakeTuple>(
        std::vector<ir::ExprPtr>{MakeConstInt(1), MakeConstInt(2), MakeConstInt(16), MakeConstInt(32)},
        ir::Span::Unknown());
    auto offsets_4d_fp4 = std::make_shared<const ir::MakeTuple>(
        std::vector<ir::ExprPtr>{MakeConstInt(1), MakeConstInt(2), MakeConstInt(16), MakeConstInt(64)},
        ir::Span::Unknown());

    CCECodegen codegen(ir::SectionKind::Vector);

    EXPECT_EQ(codegen.ComputeTensorOffset(make_nz_type({64, 64}), offsets_2d), "(32 * 64 + 16 * 16)");
    EXPECT_EQ(codegen.ComputeTensorOffset(make_nz_type({2, 3, 64, 64}), offsets_4d),
              "(1 * 3 * 64 * 64 + 2 * 64 * 64 + 32 * 64 + 16 * 16)");
    EXPECT_EQ(codegen.ComputeTensorOffset(make_nz_type({64, 64}, ir::DataType::FP8E4M3FN), offsets_2d),
              "(32 * 64 + 16 * 32)");
    EXPECT_EQ(codegen.ComputeTensorOffset(make_nz_type({64, 128}, ir::DataType::FP4E2M1), offsets_2d),
              "((32 * 64 + 16 * 64) / 2)");
    EXPECT_EQ(codegen.ComputeTensorOffset(make_nz_type({2, 3, 64, 128}, ir::DataType::FP4E2M1), offsets_4d_fp4),
              "((1 * 3 * 64 * 128 + 2 * 64 * 128 + 64 * 64 + 16 * 64) / 2)");
}

TEST(CCECodegenTest, AddsExpressionSpanToCodegenErrors)
{
    ir::Span call_span("kernel.py", 42, 7);
    auto scalar_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT64);
    auto unknown_call = std::make_shared<const ir::Call>("unknown.codegen.op", std::vector<ir::ExprPtr>{}, scalar_type,
                                                         call_span);
    auto body = std::make_shared<const ir::EvalStmt>(unknown_call, call_span);

    CCECodegen codegen(ir::SectionKind::Vector);
    try {
        (void)codegen.GenerateSingle(MakeProgram(body), "a5");
        FAIL() << "Expected an unknown backend operation to fail code generation";
    } catch (const std::exception& error) {
        std::string message = error.what();
        EXPECT_NE(message.find("Unknown call 'unknown.codegen.op'"), std::string::npos);
        EXPECT_NE(message.find("kernel.py"), std::string::npos);
        EXPECT_NE(message.find("42"), std::string::npos);
    }
}

} // namespace codegen
} // namespace pypto
