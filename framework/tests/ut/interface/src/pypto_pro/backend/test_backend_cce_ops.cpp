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
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "backend/backend_cce.h"
#include "codegen/cce/cce_codegen.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/memref.h"
#include "ir/op_attr_types.h"
#include "ir/pipe.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/type.h"

namespace pypto {
namespace backend {
namespace {

using Kwargs = std::vector<std::pair<std::string, std::any>>;

class TestableCCECodegen : public codegen::CCECodegen {
public:
    using codegen::CCECodegen::CCECodegen;
    std::string GetEmittedCode() const { return emitter_.GetCode(); }
    void SetCurrentResultTarget(std::string target) { current_target_var_ = std::move(target); }
    void SetInVFSection(bool value) { in_vf_section_ = value; }
    void RegisterVarName(const ir::VarPtr& var, const std::string& name) { context_.RegisterVar(var, name); }
};

ir::ExprPtr MakeConstInt(int64_t value, ir::DataType dtype = ir::DataType::INT64)
{
    return std::make_shared<const ir::ConstInt>(value, dtype, ir::Span::Unknown());
}

ir::VarPtr MakeVar(const std::string& name, const ir::TypePtr& type)
{
    return std::make_shared<const ir::Var>(name, type, ir::Span::Unknown());
}

ir::MakeTuplePtr MakeTuple(std::vector<ir::ExprPtr> elements)
{
    return std::make_shared<const ir::MakeTuple>(std::move(elements), ir::Span::Unknown());
}

ir::VarPtr MakeTensorVar(const std::string& name, const std::vector<int64_t>& shape, ir::DataType dtype,
                         ir::TensorLayout layout = ir::TensorLayout::ND, std::vector<ir::ExprPtr> strides = {})
{
    auto ptr = MakeVar(name + "_base", std::make_shared<const ir::PtrType>(dtype));
    ir::TensorView view(std::move(strides), layout, ptr);
    auto tensor_type = std::make_shared<const ir::TensorType>(shape, dtype, std::optional<ir::MemRefPtr>(std::nullopt),
                                                              std::optional<ir::TensorView>(view));
    return MakeVar(name, tensor_type);
}

ir::VarPtr MakeDynamicTensorVar(const std::string& name, std::vector<ir::ExprPtr> shape, ir::DataType dtype,
                                ir::TensorLayout layout = ir::TensorLayout::ND, std::vector<ir::ExprPtr> strides = {})
{
    auto ptr = MakeVar(name + "_base", std::make_shared<const ir::PtrType>(dtype));
    ir::TensorView view(std::move(strides), layout, ptr);
    auto tensor_type = std::make_shared<const ir::TensorType>(
        std::move(shape), dtype, std::optional<ir::MemRefPtr>(std::nullopt), std::optional<ir::TensorView>(view));
    return MakeVar(name, tensor_type);
}

ir::ProgramPtr MakeProgram(const ir::StmtPtr& body, const std::vector<ir::VarPtr>& params = {})
{
    auto function = std::make_shared<const ir::Function>("kernel", params, std::vector<ir::TypePtr>{}, body,
                                                         ir::Span::Unknown(), ir::FunctionType::IN_CORE, true);
    return std::make_shared<const ir::Program>(std::vector<ir::FunctionPtr>{function}, "test_program",
                                               ir::Span::Unknown());
}

ir::TileTypePtr MakeTileType(const std::vector<int64_t>& shape, ir::DataType dtype,
                             std::optional<ir::MemorySpace> space = std::nullopt)
{
    std::optional<ir::MemRefPtr> memref = std::nullopt;
    if (space.has_value()) {
        memref = std::make_shared<const ir::MemRef>(space.value(), MakeConstInt(0), 1024, 0, ir::Span::Unknown());
    }
    return std::make_shared<const ir::TileType>(shape, dtype, memref, std::optional<ir::TileView>(std::nullopt),
                                                std::optional<ir::HardwareInfo>(ir::HardwareInfo{}));
}

std::string RunCodegen(const std::string& op_name, const ir::CallPtr& call, TestableCCECodegen& codegen)
{
    for (const auto& arg : call->args_) {
        auto var = ir::As<ir::Var>(arg);
        if (var != nullptr && ir::As<ir::TensorType>(var->GetType()) != nullptr && !codegen.HasPointer(var->name_)) {
            codegen.RegisterPointer(var->name_, var->name_ + ".data()");
        }
    }
    const auto* info = BackendCCE::Instance().GetOpInfo(op_name);
    EXPECT_NE(info, nullptr);
    if (info == nullptr) {
        return "";
    }
    info->codegen_func(call, codegen);
    return codegen.GetEmittedCode();
}

std::string RunCodegen(const std::string& op_name, const ir::CallPtr& call)
{
    TestableCCECodegen codegen(ir::SectionKind::Vector);
    return RunCodegen(op_name, call, codegen);
}

} // namespace

// ============================================================================
// Return-value ops (tested via AssignStmt)
// ============================================================================

TEST(BackendCceOpsTest, GetBlockIdx)
{
    auto int_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT32);
    auto result = MakeVar("idx", int_type);
    auto call = std::make_shared<const ir::Call>("get_block_idx", std::vector<ir::ExprPtr>{}, int_type,
                                                 ir::Span::Unknown());
    auto body = std::make_shared<const ir::AssignStmt>(result, call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Cube);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a3");
    EXPECT_NE(generated.find("(int32_t)(get_block_idx())"), std::string::npos);
}

TEST(BackendCceOpsTest, GetBlockIdxVectorUsesGlobalAivIndex)
{
    auto int_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT32);
    auto call = std::make_shared<const ir::Call>("get_block_idx", std::vector<ir::ExprPtr>{}, int_type,
                                                 ir::Span::Unknown());
    const auto* info = BackendCCE::Instance().GetOpInfo("get_block_idx");
    ASSERT_NE(info, nullptr);

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    EXPECT_EQ(info->codegen_func(call, codegen), "(int32_t)(get_block_idx() * get_subblockdim() + get_subblockid())");
}

TEST(BackendCceOpsTest, GetBlockNum)
{
    auto int_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT32);
    auto result = MakeVar("num", int_type);
    auto call = std::make_shared<const ir::Call>("get_block_num", std::vector<ir::ExprPtr>{}, int_type,
                                                 ir::Span::Unknown());
    auto body = std::make_shared<const ir::AssignStmt>(result, call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a3");
    EXPECT_NE(generated.find("(int32_t)(get_block_num())"), std::string::npos);
}

TEST(BackendCceOpsTest, GetSpr)
{
    auto int_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT64);
    auto call = std::make_shared<const ir::Call>("get_spr", std::vector<ir::ExprPtr>{}, int_type, ir::Span::Unknown());
    const auto* info = BackendCCE::Instance().GetOpInfo("get_spr");
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->pipe, ir::PipeType::V);

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    EXPECT_EQ(info->codegen_func(call, codegen), "get_ar()");
}

TEST(BackendCceOpsTest, GetSubBlockIdx)
{
    auto int_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT32);
    auto result = MakeVar("sub", int_type);
    auto call = std::make_shared<const ir::Call>("get_subblock_idx", std::vector<ir::ExprPtr>{}, int_type,
                                                 ir::Span::Unknown());
    auto body = std::make_shared<const ir::AssignStmt>(result, call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a3");
    EXPECT_NE(generated.find("(int32_t)(get_subblockid())"), std::string::npos);
}

TEST(BackendCceOpsTest, GetSubBlockNumIsTargetSpecific)
{
    auto int_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT32);
    auto call = std::make_shared<const ir::Call>("get_subblock_num", std::vector<ir::ExprPtr>{}, int_type,
                                                 ir::Span::Unknown());
    const auto* info = BackendCCE::Instance().GetOpInfo("get_subblock_num");
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->pipe, ir::PipeType::V);

    codegen::CCECodegen cube_codegen(ir::SectionKind::Cube);
    EXPECT_EQ(info->codegen_func(call, cube_codegen), "(int32_t)(1)");

    codegen::CCECodegen vector_codegen(ir::SectionKind::Vector);
    EXPECT_EQ(info->codegen_func(call, vector_codegen), "(int32_t)(get_subblockdim())");
}

TEST(BackendCceOpsTest, DebugTrap)
{
    auto call = std::make_shared<const ir::Call>("debug.trap", std::vector<ir::ExprPtr>{}, ir::Span::Unknown());
    const auto* info = BackendCCE::Instance().GetOpInfo("debug.trap");
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->pipe, ir::PipeType::S);

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    EXPECT_EQ(info->codegen_func(call, codegen), "return");
}

// ============================================================================
// Barrier ops (tested via EvalStmt)
// ============================================================================

TEST(BackendCceOpsTest, BarM)
{
    auto call = std::make_shared<const ir::Call>("system.bar_m", std::vector<ir::ExprPtr>{}, ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a3");
    EXPECT_NE(generated.find("pipe_barrier(PIPE_M);"), std::string::npos);
}

TEST(BackendCceOpsTest, AdditionalPipelineBarriers)
{
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"system.bar_mte1", "pipe_barrier(PIPE_MTE1);"},
        {"system.bar_mte2", "pipe_barrier(PIPE_MTE2);"},
        {"system.bar_mte3", "pipe_barrier(PIPE_MTE3);"},
        {"system.bar_fix", "pipe_barrier(PIPE_FIX);"},
    };

    for (const auto& [op_name, expected] : cases) {
        auto call = std::make_shared<const ir::Call>(op_name, std::vector<ir::ExprPtr>{}, ir::Span::Unknown());
        const auto* info = BackendCCE::Instance().GetOpInfo(op_name);
        ASSERT_NE(info, nullptr);
        EXPECT_EQ(info->pipe, ir::PipeType::S);

        TestableCCECodegen codegen(ir::SectionKind::Vector);
        EXPECT_EQ(info->codegen_func(call, codegen), "");
        EXPECT_NE(codegen.GetEmittedCode().find(expected), std::string::npos);
    }
}

TEST(BackendCceOpsTest, BarAll)
{
    auto call = std::make_shared<const ir::Call>("system.bar_all", std::vector<ir::ExprPtr>{}, ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a3");
    EXPECT_NE(generated.find("pipe_barrier(PIPE_ALL);"), std::string::npos);
}

// ============================================================================
// Mask ops (tested via EvalStmt)
// ============================================================================

TEST(BackendCceOpsTest, SetMaskCount)
{
    auto call = std::make_shared<const ir::Call>("system.set_mask_count", std::vector<ir::ExprPtr>{},
                                                 ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a3");
    EXPECT_NE(generated.find("set_mask_count();"), std::string::npos);
}

TEST(BackendCceOpsTest, SetMaskNorm)
{
    auto call = std::make_shared<const ir::Call>("system.set_mask_norm", std::vector<ir::ExprPtr>{},
                                                 ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a3");
    EXPECT_NE(generated.find("set_mask_norm();"), std::string::npos);
}

TEST(BackendCceOpsTest, ResetMask)
{
    auto call = std::make_shared<const ir::Call>("system.reset_mask", std::vector<ir::ExprPtr>{}, ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a3");
    EXPECT_NE(generated.find("set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));"),
              std::string::npos);
}

TEST(BackendCceOpsTest, SetVecMask)
{
    auto call = std::make_shared<const ir::Call>(
        "system.set_vec_mask", std::vector<ir::ExprPtr>{MakeConstInt(255), MakeConstInt(128)}, ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a3");
    EXPECT_NE(generated.find("set_vector_mask(255, 128);"), std::string::npos);
}

// ============================================================================
// Sync ops with kwargs (tested via EvalStmt)
// ============================================================================

TEST(BackendCceOpsTest, SyncSrc)
{
    Kwargs kwargs = {{"set_pipe", 4}, {"wait_pipe", 5}, {"event_id", 3}};
    auto call = std::make_shared<const ir::Call>("system.sync_src", std::vector<ir::ExprPtr>{}, kwargs,
                                                 ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a3");
    EXPECT_NE(generated.find("set_flag(PIPE_V, PIPE_S, EVENT_ID3);"), std::string::npos);
}

TEST(BackendCceOpsTest, SyncDst)
{
    Kwargs kwargs = {{"set_pipe", 4}, {"wait_pipe", 5}, {"event_id", 3}};
    auto call = std::make_shared<const ir::Call>("system.sync_dst", std::vector<ir::ExprPtr>{}, kwargs,
                                                 ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a3");
    EXPECT_NE(generated.find("wait_flag(PIPE_V, PIPE_S, EVENT_ID3);"), std::string::npos);
}

TEST(BackendCceOpsTest, SyncSupportsEveryPipeKind)
{
    const std::vector<std::pair<ir::PipeType, std::string>> pipes = {{ir::PipeType::MTE1, "PIPE_MTE1"},
                                                                     {ir::PipeType::MTE2, "PIPE_MTE2"},
                                                                     {ir::PipeType::MTE3, "PIPE_MTE3"},
                                                                     {ir::PipeType::FIX, "PIPE_FIX"},
                                                                     {ir::PipeType::ALL, "PIPE_ALL"}};

    for (const auto& [pipe, pipe_name] : pipes) {
        Kwargs kwargs = {
            {"set_pipe", static_cast<int>(pipe)}, {"wait_pipe", static_cast<int>(ir::PipeType::S)}, {"event_id", 1}};
        auto call = std::make_shared<const ir::Call>("system.sync_src", std::vector<ir::ExprPtr>{}, kwargs,
                                                     ir::Span::Unknown());
        EXPECT_NE(RunCodegen("system.sync_src", call).find("set_flag(" + pipe_name + ", PIPE_S, EVENT_ID1);"),
                  std::string::npos);
    }
}

// ============================================================================
// Dynamic sync ops with args + kwargs (tested via EvalStmt)
// ============================================================================

TEST(BackendCceOpsTest, SyncSrcDyn)
{
    Kwargs kwargs = {{"set_pipe", 3}, {"wait_pipe", 4}};
    auto call = std::make_shared<const ir::Call>("system.sync_src_dyn", std::vector<ir::ExprPtr>{MakeConstInt(5)},
                                                 kwargs, ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a3");
    EXPECT_NE(generated.find("set_flag(PIPE_M, PIPE_V, (event_t)5);"), std::string::npos);
}

TEST(BackendCceOpsTest, SyncDstDyn)
{
    Kwargs kwargs = {{"set_pipe", 3}, {"wait_pipe", 4}};
    auto call = std::make_shared<const ir::Call>("system.sync_dst_dyn", std::vector<ir::ExprPtr>{MakeConstInt(5)},
                                                 kwargs, ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a3");
    EXPECT_NE(generated.find("wait_flag(PIPE_M, PIPE_V, (event_t)5);"), std::string::npos);
}

// ============================================================================
// Cross-core sync ops - non-A5 path (tested via EvalStmt)
// ============================================================================

TEST(BackendCceOpsTest, SetCrossCoreA3)
{
    Kwargs kwargs = {{"pipe", 4}, {"sync_mode", 0}, {"event_id", 2}};
    auto call = std::make_shared<const ir::Call>("system.set_cross_core", std::vector<ir::ExprPtr>{}, kwargs,
                                                 ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a3");
    EXPECT_NE(generated.find("ffts_cross_core_sync(PIPE_V, getFFTSMsg(0, 2));"), std::string::npos);
}

TEST(BackendCceOpsTest, WaitCrossCoreA3)
{
    Kwargs kwargs = {{"pipe", 4}, {"sync_mode", 0}, {"event_id", 2}};
    auto call = std::make_shared<const ir::Call>("system.wait_cross_core", std::vector<ir::ExprPtr>{}, kwargs,
                                                 ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a3");
    EXPECT_NE(generated.find("wait_flag_dev(2);"), std::string::npos);
}

TEST(BackendCceOpsTest, SetCrossCoreDynA3)
{
    Kwargs kwargs = {{"pipe", 4}, {"sync_mode", 0}};
    auto call = std::make_shared<const ir::Call>("system.set_cross_core_dyn", std::vector<ir::ExprPtr>{MakeConstInt(7)},
                                                 kwargs, ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a3");
    EXPECT_NE(generated.find("ffts_cross_core_sync(PIPE_V, getFFTSMsg(0, 7));"), std::string::npos);
}

TEST(BackendCceOpsTest, WaitCrossCoreDynA3)
{
    Kwargs kwargs = {{"pipe", 4}, {"sync_mode", 0}};
    auto call = std::make_shared<const ir::Call>(
        "system.wait_cross_core_dyn", std::vector<ir::ExprPtr>{MakeConstInt(7)}, kwargs, ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a3");
    EXPECT_NE(generated.find("wait_flag_dev(7);"), std::string::npos);
}

// ============================================================================
// Cross-core sync ops - A5 INTER_BLOCK path
// ============================================================================

TEST(BackendCceOpsTest, SetCrossCoreA5InterBlock)
{
    Kwargs kwargs = {{"pipe", 4}, {"sync_mode", 0}, {"event_id", 2}};
    auto call = std::make_shared<const ir::Call>("system.set_cross_core", std::vector<ir::ExprPtr>{}, kwargs,
                                                 ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a5");
    EXPECT_NE(generated.find("ffts_cross_core_sync(PIPE_V, getFFTSMsg(0, 2));"), std::string::npos);
}

TEST(BackendCceOpsTest, WaitCrossCoreA5InterBlock)
{
    Kwargs kwargs = {{"pipe", 4}, {"sync_mode", 0}, {"event_id", 2}};
    auto call = std::make_shared<const ir::Call>("system.wait_cross_core", std::vector<ir::ExprPtr>{}, kwargs,
                                                 ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a5");
    EXPECT_NE(generated.find("wait_flag_dev(PIPE_V, 2);"), std::string::npos);
}

// ============================================================================
// Cross-core sync ops - A5 INTRA_BLOCK path (non-cube section)
// ============================================================================

TEST(BackendCceOpsTest, SetCrossCoreA5IntraBlock)
{
    Kwargs kwargs = {{"pipe", 4}, {"sync_mode", 2}, {"event_id", 3}};
    auto call = std::make_shared<const ir::Call>("system.set_cross_core", std::vector<ir::ExprPtr>{}, kwargs,
                                                 ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a5");
    EXPECT_NE(generated.find("set_intra_block(PIPE_V, 3);"), std::string::npos);
}

TEST(BackendCceOpsTest, WaitCrossCoreA5IntraBlock)
{
    Kwargs kwargs = {{"pipe", 4}, {"sync_mode", 2}, {"event_id", 3}};
    auto call = std::make_shared<const ir::Call>("system.wait_cross_core", std::vector<ir::ExprPtr>{}, kwargs,
                                                 ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a5");
    EXPECT_NE(generated.find("wait_intra_block(PIPE_V, 3);"), std::string::npos);
}

TEST(BackendCceOpsTest, SetCrossCoreDynA5IntraBlock)
{
    Kwargs kwargs = {{"pipe", 4}, {"sync_mode", 2}};
    auto call = std::make_shared<const ir::Call>("system.set_cross_core_dyn", std::vector<ir::ExprPtr>{MakeConstInt(8)},
                                                 kwargs, ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a5");
    EXPECT_NE(generated.find("set_intra_block(PIPE_V, 8);"), std::string::npos);
}

TEST(BackendCceOpsTest, WaitCrossCoreDynA5IntraBlock)
{
    Kwargs kwargs = {{"pipe", 4}, {"sync_mode", 2}};
    auto call = std::make_shared<const ir::Call>(
        "system.wait_cross_core_dyn", std::vector<ir::ExprPtr>{MakeConstInt(8)}, kwargs, ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a5");
    EXPECT_NE(generated.find("wait_intra_block(PIPE_V, 8);"), std::string::npos);
}

TEST(BackendCceOpsTest, CubeCrossCoreA5IntraBlockSignalsBothVectorSubcores)
{
    Kwargs static_kwargs = {{"pipe", 4}, {"sync_mode", 2}, {"event_id", 3}};
    auto set_call = std::make_shared<const ir::Call>("system.set_cross_core", std::vector<ir::ExprPtr>{}, static_kwargs,
                                                     ir::Span::Unknown());
    auto wait_call = std::make_shared<const ir::Call>("system.wait_cross_core", std::vector<ir::ExprPtr>{},
                                                      static_kwargs, ir::Span::Unknown());
    auto body = std::make_shared<const ir::SeqStmts>(
        std::vector<ir::StmtPtr>{std::make_shared<const ir::EvalStmt>(set_call, ir::Span::Unknown()),
                                 std::make_shared<const ir::EvalStmt>(wait_call, ir::Span::Unknown())},
        ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Cube);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a5");
    EXPECT_NE(generated.find("set_intra_block(PIPE_V, 3);"), std::string::npos);
    EXPECT_NE(generated.find("set_intra_block(PIPE_V, 19);"), std::string::npos);
    EXPECT_NE(generated.find("wait_intra_block(PIPE_V, 3);"), std::string::npos);
    EXPECT_NE(generated.find("wait_intra_block(PIPE_V, 19);"), std::string::npos);
}

TEST(BackendCceOpsTest, CubeCrossCoreDynamicA5IntraBlockSignalsBothVectorSubcores)
{
    Kwargs kwargs = {{"pipe", 4}, {"sync_mode", 2}};
    auto event_id = MakeVar("event_id", std::make_shared<const ir::ScalarType>(ir::DataType::INT32));
    auto set_call = std::make_shared<const ir::Call>("system.set_cross_core_dyn", std::vector<ir::ExprPtr>{event_id},
                                                     kwargs, ir::Span::Unknown());
    auto wait_call = std::make_shared<const ir::Call>("system.wait_cross_core_dyn", std::vector<ir::ExprPtr>{event_id},
                                                      kwargs, ir::Span::Unknown());
    auto body = std::make_shared<const ir::SeqStmts>(
        std::vector<ir::StmtPtr>{std::make_shared<const ir::EvalStmt>(set_call, ir::Span::Unknown()),
                                 std::make_shared<const ir::EvalStmt>(wait_call, ir::Span::Unknown())},
        ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Cube);
    auto generated = codegen.GenerateSingle(MakeProgram(body, {event_id}), "a5");
    EXPECT_NE(generated.find("set_intra_block(PIPE_V, event_id_0);"), std::string::npos);
    EXPECT_NE(generated.find("set_intra_block(PIPE_V, event_id_0 + 16);"), std::string::npos);
    EXPECT_NE(generated.find("wait_intra_block(PIPE_V, event_id_0);"), std::string::npos);
    EXPECT_NE(generated.find("wait_intra_block(PIPE_V, event_id_0 + 16);"), std::string::npos);
}

// ============================================================================
// Mutex ops (tested via EvalStmt)
// ============================================================================

TEST(BackendCceOpsTest, MutexLock)
{
    Kwargs kwargs = {{"pipe", 5}, {"mutex_id", 1}, {"mode", 0}};
    auto call = std::make_shared<const ir::Call>("system.mutex_lock", std::vector<ir::ExprPtr>{}, kwargs,
                                                 ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a3");
    EXPECT_NE(generated.find("get_buf(PIPE_S, 1, 0);"), std::string::npos);
}

TEST(BackendCceOpsTest, MutexUnlock)
{
    Kwargs kwargs = {{"pipe", 5}, {"mutex_id", 1}, {"mode", 0}};
    auto call = std::make_shared<const ir::Call>("system.mutex_unlock", std::vector<ir::ExprPtr>{}, kwargs,
                                                 ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a3");
    EXPECT_NE(generated.find("rls_buf(PIPE_S, 1, 0);"), std::string::npos);
}

TEST(BackendCceOpsTest, MutexLockDyn)
{
    Kwargs kwargs = {{"pipe", 5}, {"mode", 0}};
    auto call = std::make_shared<const ir::Call>("system.mutex_lock_dyn", std::vector<ir::ExprPtr>{MakeConstInt(2)},
                                                 kwargs, ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a3");
    EXPECT_NE(generated.find("get_buf(PIPE_S, 2, 0);"), std::string::npos);
}

TEST(BackendCceOpsTest, MutexUnlockDyn)
{
    Kwargs kwargs = {{"pipe", 5}, {"mode", 0}};
    auto call = std::make_shared<const ir::Call>("system.mutex_unlock_dyn", std::vector<ir::ExprPtr>{MakeConstInt(2)},
                                                 kwargs, ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a3");
    EXPECT_NE(generated.find("rls_buf(PIPE_S, 2, 0);"), std::string::npos);
}

TEST(BackendCceOpsTest, MutexDynDedupUnlocksFirstOccurrencesInInputOrder)
{
    Kwargs kwargs = {{"pipe", 5}, {"mode", 0}};
    auto id0 = MakeVar("id0", std::make_shared<const ir::ScalarType>(ir::DataType::INT32));
    auto id1 = MakeVar("id1", std::make_shared<const ir::ScalarType>(ir::DataType::INT32));
    std::vector<ir::ExprPtr> ids = {id0, id1, id0};
    auto lock = std::make_shared<const ir::Call>("system.mutex_lock_dyn", ids, kwargs, ir::Span::Unknown());
    auto unlock = std::make_shared<const ir::Call>("system.mutex_unlock_dyn", ids, kwargs, ir::Span::Unknown());
    auto body = std::make_shared<const ir::SeqStmts>(
        std::vector<ir::StmtPtr>{std::make_shared<const ir::EvalStmt>(lock, ir::Span::Unknown()),
                                 std::make_shared<const ir::EvalStmt>(unlock, ir::Span::Unknown())},
        ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body, {id0, id1}), "a5");
    auto release_id0 = generated.find("rls_buf(PIPE_S, id0_0, 0);");
    auto release_id1 = generated.find("rls_buf(PIPE_S, id1_0, 0);", release_id0);
    auto duplicate_guard = generated.find("if ((id0_0 != id0_0) && (id0_0 != id1_0)) {", release_id1);
    auto duplicate_release = generated.find("rls_buf(PIPE_S, id0_0, 0);", duplicate_guard);
    EXPECT_NE(release_id0, std::string::npos);
    EXPECT_NE(release_id1, std::string::npos);
    EXPECT_NE(duplicate_guard, std::string::npos);
    EXPECT_NE(duplicate_release, std::string::npos);
    EXPECT_LT(release_id0, release_id1);
    EXPECT_LT(release_id1, duplicate_guard);
    EXPECT_LT(duplicate_guard, duplicate_release);
}

TEST(BackendCceOpsTest, MutexDynSkipsDedupWithinOneTile)
{
    Kwargs kwargs = {{"pipe", 5}, {"mode", 0}, {"mutex_id_owner_indices", std::vector<int>{0, 0, 1}}};
    auto output0 = MakeVar("output0", std::make_shared<const ir::ScalarType>(ir::DataType::INT32));
    auto output1 = MakeVar("output1", std::make_shared<const ir::ScalarType>(ir::DataType::INT32));
    auto source = MakeVar("source", std::make_shared<const ir::ScalarType>(ir::DataType::INT32));
    std::vector<ir::ExprPtr> ids = {output0, output1, source};
    auto lock = std::make_shared<const ir::Call>("system.mutex_lock_dyn", ids, kwargs, ir::Span::Unknown());
    auto unlock = std::make_shared<const ir::Call>("system.mutex_unlock_dyn", ids, kwargs, ir::Span::Unknown());
    auto body = std::make_shared<const ir::SeqStmts>(
        std::vector<ir::StmtPtr>{std::make_shared<const ir::EvalStmt>(lock, ir::Span::Unknown()),
                                 std::make_shared<const ir::EvalStmt>(unlock, ir::Span::Unknown())},
        ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body, {output0, output1, source}), "a5");
    const std::string same_tile_guard = "if ((output1_0 != output0_0)) {";
    const std::string cross_tile_guard = "if ((source_0 != output0_0) && (source_0 != output1_0)) {";
    auto acquire_output0 = generated.find("get_buf(PIPE_S, output0_0, 0);");
    auto acquire_output1 = generated.find("get_buf(PIPE_S, output1_0, 0);", acquire_output0);
    auto acquire_source_guard = generated.find(cross_tile_guard, acquire_output1);
    auto release_output0 = generated.find("rls_buf(PIPE_S, output0_0, 0);", acquire_source_guard);
    auto release_output1 = generated.find("rls_buf(PIPE_S, output1_0, 0);", release_output0);
    auto release_source_guard = generated.find(cross_tile_guard, release_output1);
    EXPECT_EQ(generated.find(same_tile_guard), std::string::npos);
    EXPECT_NE(acquire_output0, std::string::npos);
    EXPECT_NE(acquire_output1, std::string::npos);
    EXPECT_NE(acquire_source_guard, std::string::npos);
    EXPECT_NE(release_output0, std::string::npos);
    EXPECT_NE(release_output1, std::string::npos);
    EXPECT_NE(release_source_guard, std::string::npos);
    EXPECT_LT(acquire_output0, acquire_output1);
    EXPECT_LT(acquire_output1, acquire_source_guard);
    EXPECT_LT(release_output0, release_output1);
    EXPECT_LT(release_output1, release_source_guard);
}

TEST(BackendCceOpsTest, ManualVPipeMutexIsNotSkippedOnA5)
{
    Kwargs kwargs = {{"pipe", 4}, {"mutex_id", 7}, {"mode", 0}, {"auto_mutex", false}};
    auto call = std::make_shared<const ir::Call>("system.mutex_lock", std::vector<ir::ExprPtr>{}, kwargs,
                                                 ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a5");
    EXPECT_NE(generated.find("get_buf(PIPE_V, 7, 0);"), std::string::npos);
}

TEST(BackendCceOpsTest, MissingAutoMutexKwargDefaultsToManualOnA5)
{
    Kwargs kwargs = {{"pipe", 4}, {"mutex_id", 7}, {"mode", 0}};
    auto call = std::make_shared<const ir::Call>("system.mutex_lock", std::vector<ir::ExprPtr>{}, kwargs,
                                                 ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a5");
    EXPECT_NE(generated.find("get_buf(PIPE_V, 7, 0);"), std::string::npos);
}

TEST(BackendCceOpsTest, AutoVPipeMutexIsSkippedWhenOnlyUsedByVPipe)
{
    Kwargs kwargs = {{"pipe", 4},
                     {"mutex_id", 7},
                     {"mode", 0},
                     {"mutex_ids", std::vector<int>{7}},
                     {"auto_mutex", true}};
    auto call = std::make_shared<const ir::Call>("system.mutex_lock", std::vector<ir::ExprPtr>{}, kwargs,
                                                 ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a5");
    EXPECT_EQ(generated.find("get_buf(PIPE_V, 7, 0);"), std::string::npos);
}

TEST(BackendCceOpsTest, AutoStaticVPipeMutexUsesTileGroupIds)
{
    Kwargs v_kwargs = {{"pipe", 4},
                       {"mutex_id", 6},
                       {"mode", 0},
                       {"mutex_ids", std::vector<int>{6, 7}},
                       {"auto_mutex", true}};
    Kwargs mte2_kwargs = {{"pipe", 1},
                          {"mutex_id", 7},
                          {"mode", 0},
                          {"mutex_ids", std::vector<int>{6, 7}},
                          {"auto_mutex", true}};
    auto v_call = std::make_shared<const ir::Call>("system.mutex_lock", std::vector<ir::ExprPtr>{}, v_kwargs,
                                                   ir::Span::Unknown());
    auto mte2_call = std::make_shared<const ir::Call>("system.mutex_lock", std::vector<ir::ExprPtr>{}, mte2_kwargs,
                                                      ir::Span::Unknown());
    auto body = std::make_shared<const ir::SeqStmts>(
        std::vector<ir::StmtPtr>{std::make_shared<const ir::EvalStmt>(v_call, ir::Span::Unknown()),
                                 std::make_shared<const ir::EvalStmt>(mte2_call, ir::Span::Unknown())},
        ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a5");
    EXPECT_NE(generated.find("get_buf(PIPE_V, 6, 0);"), std::string::npos);
    EXPECT_NE(generated.find("get_buf(PIPE_MTE2, 7, 0);"), std::string::npos);
}

TEST(BackendCceOpsTest, AutoDynamicVPipeMutexUsesInternalCandidateIds)
{
    auto dynamic_id = MakeVar("dynamic_id", std::make_shared<const ir::ScalarType>(ir::DataType::INT32));
    Kwargs dyn_kwargs = {{"pipe", 4},
                         {"mode", 0},
                         {"mutex_ids", std::vector<int>{6, 7}},
                         {"auto_mutex", true}};
    Kwargs mte2_kwargs = {{"pipe", 1},
                          {"mutex_id", 7},
                          {"mode", 0},
                          {"mutex_ids", std::vector<int>{6, 7}},
                          {"auto_mutex", true}};
    auto dyn_call = std::make_shared<const ir::Call>("system.mutex_lock_dyn", std::vector<ir::ExprPtr>{dynamic_id},
                                                     dyn_kwargs, ir::Span::Unknown());
    auto mte2_call = std::make_shared<const ir::Call>("system.mutex_lock", std::vector<ir::ExprPtr>{}, mte2_kwargs,
                                                      ir::Span::Unknown());
    auto body = std::make_shared<const ir::SeqStmts>(
        std::vector<ir::StmtPtr>{std::make_shared<const ir::EvalStmt>(dyn_call, ir::Span::Unknown()),
                                 std::make_shared<const ir::EvalStmt>(mte2_call, ir::Span::Unknown())},
        ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body, {dynamic_id}), "a5");
    EXPECT_NE(generated.find("get_buf(PIPE_V, dynamic_id_0, 0);"), std::string::npos);
}

TEST(BackendCceOpsTest, AutoDynamicVPipeMutexWithoutCandidatesIsKept)
{
    auto dynamic_id = MakeVar("dynamic_id", std::make_shared<const ir::ScalarType>(ir::DataType::INT32));
    Kwargs kwargs = {{"pipe", 4}, {"mode", 0}, {"auto_mutex", true}};
    auto call = std::make_shared<const ir::Call>("system.mutex_lock_dyn", std::vector<ir::ExprPtr>{dynamic_id},
                                                 kwargs, ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body, {dynamic_id}), "a5");
    EXPECT_NE(generated.find("get_buf(PIPE_V, dynamic_id_0, 0);"), std::string::npos);
}

TEST(BackendCceOpsTest, DynamicMutexDedupIfGuardsRemainUnchanged)
{
    auto first_id = MakeVar("first_id", std::make_shared<const ir::ScalarType>(ir::DataType::INT32));
    auto second_id = MakeVar("second_id", std::make_shared<const ir::ScalarType>(ir::DataType::INT32));
    Kwargs kwargs = {{"pipe", 1},
                     {"mode", 0},
                     {"mutex_ids", std::vector<int>{6, 7}},
                     {"auto_mutex", true}};
    auto call = std::make_shared<const ir::Call>(
        "system.mutex_lock_dyn", std::vector<ir::ExprPtr>{first_id, second_id}, kwargs, ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body, {first_id, second_id}), "a5");
    EXPECT_NE(generated.find("get_buf(PIPE_MTE2, first_id_0, 0);"), std::string::npos);
    EXPECT_NE(generated.find("if ((second_id_0 != first_id_0))"), std::string::npos);
    EXPECT_NE(generated.find("get_buf(PIPE_MTE2, second_id_0, 0);"), std::string::npos);
}

// ============================================================================
// A5-specific: set_mm_layout_transform
// ============================================================================

TEST(BackendCceOpsTest, SetMmLayoutTransformEnabled)
{
    Kwargs kwargs = {{"enabled", 1}};
    auto call = std::make_shared<const ir::Call>("system.set_mm_layout_transform", std::vector<ir::ExprPtr>{}, kwargs,
                                                 ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a5");
    EXPECT_NE(generated.find("set_ctrl(sbitset1(get_ctrl(), 51));"), std::string::npos);
}

TEST(BackendCceOpsTest, SetMmLayoutTransformDisabled)
{
    Kwargs kwargs = {{"enabled", 0}};
    auto call = std::make_shared<const ir::Call>("system.set_mm_layout_transform", std::vector<ir::ExprPtr>{}, kwargs,
                                                 ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a5");
    EXPECT_NE(generated.find("set_ctrl(sbitset0(get_ctrl(), 51));"), std::string::npos);
}

// ============================================================================
// Ptr ops (tested via AssignStmt)
// ============================================================================

TEST(BackendCceOpsTest, PtrAddPtr)
{
    auto ptr_type = std::make_shared<const ir::PtrType>(ir::DataType::FP32);
    auto ptr_var = MakeVar("base", ptr_type);
    auto result = MakeVar("offset_ptr", ptr_type);
    auto call = std::make_shared<const ir::Call>("ptr.addptr", std::vector<ir::ExprPtr>{ptr_var, MakeConstInt(10)},
                                                 ptr_type, ir::Span::Unknown());
    auto body = std::make_shared<const ir::AssignStmt>(result, call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body, {ptr_var}), "a3");
    EXPECT_NE(generated.find("+ 10)"), std::string::npos);
}

TEST(BackendCceOpsTest, PtrMakePtr)
{
    auto ptr_type = std::make_shared<const ir::PtrType>(ir::DataType::FP32);
    auto ptr_var = MakeVar("src", ptr_type);
    auto result = MakeVar("cast", ptr_type);
    auto call = std::make_shared<const ir::Call>("ptr.make_ptr", std::vector<ir::ExprPtr>{ptr_var}, ptr_type,
                                                 ir::Span::Unknown());
    auto body = std::make_shared<const ir::AssignStmt>(result, call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body, {ptr_var}), "a3");
    EXPECT_NE(generated.find("__gm__ float*"), std::string::npos);
}

TEST(BackendCceOpsTest, PtrMakePtrExtractsRegisteredTensorPointer)
{
    auto tensor = MakeTensorVar("src", {16}, ir::DataType::FP32);
    auto result_type = std::make_shared<const ir::PtrType>(ir::DataType::FP16);
    auto call = std::make_shared<const ir::Call>("ptr.make_ptr", std::vector<ir::ExprPtr>{tensor}, result_type,
                                                 ir::Span::Unknown());
    const auto* info = BackendCCE::Instance().GetOpInfo("ptr.make_ptr");
    ASSERT_NE(info, nullptr);

    TestableCCECodegen codegen(ir::SectionKind::Vector);
    codegen.RegisterVarName(tensor, "src");
    codegen.RegisterPointer("src", "src_raw_ptr");
    EXPECT_EQ(info->codegen_func(call, codegen), "((__gm__ half*)(src_raw_ptr))");
}

TEST(BackendCceOpsTest, PtrMakePtrFallsBackToTensorData)
{
    auto tensor = MakeTensorVar("src", {16}, ir::DataType::FP32);
    auto result_type = std::make_shared<const ir::PtrType>(ir::DataType::FP16);
    auto call = std::make_shared<const ir::Call>("ptr.make_ptr", std::vector<ir::ExprPtr>{tensor}, result_type,
                                                 ir::Span::Unknown());
    const auto* info = BackendCCE::Instance().GetOpInfo("ptr.make_ptr");
    ASSERT_NE(info, nullptr);

    TestableCCECodegen codegen(ir::SectionKind::Vector);
    codegen.RegisterVarName(tensor, "src");
    EXPECT_EQ(info->codegen_func(call, codegen), "((__gm__ half*)(src.data()))");
}

TEST(BackendCceOpsTest, PtrMakePtrRejectsInvalidSourceAndResultTypes)
{
    auto scalar_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT32);
    auto ptr_type = std::make_shared<const ir::PtrType>(ir::DataType::FP32);
    auto scalar = MakeVar("scalar", scalar_type);
    auto ptr = MakeVar("ptr", ptr_type);
    const auto* info = BackendCCE::Instance().GetOpInfo("ptr.make_ptr");
    ASSERT_NE(info, nullptr);

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto invalid_source = std::make_shared<const ir::Call>("ptr.make_ptr", std::vector<ir::ExprPtr>{scalar}, ptr_type,
                                                           ir::Span::Unknown());
    EXPECT_THROW((void)info->codegen_func(invalid_source, codegen), npu::tile_fwk::Error);

    auto invalid_result = std::make_shared<const ir::Call>("ptr.make_ptr", std::vector<ir::ExprPtr>{ptr}, scalar_type,
                                                           ir::Span::Unknown());
    EXPECT_THROW((void)info->codegen_func(invalid_result, codegen), npu::tile_fwk::Error);
}

// ============================================================================
// Block make_tile / getval / setval (tile)
// ============================================================================

TEST(BackendCceOpsTest, BlockMakeTileAndSetValTile)
{
    auto tile_type = MakeTileType({16, 16}, ir::DataType::FP16);
    auto tile = MakeVar("tile", tile_type);
    auto make_tile = std::make_shared<const ir::Call>("block.make_tile", std::vector<ir::ExprPtr>{}, tile_type,
                                                      ir::Span::Unknown());
    auto tile_assign = std::make_shared<const ir::AssignStmt>(tile, make_tile, ir::Span::Unknown());

    auto setval = std::make_shared<const ir::Call>(
        "block.setval", std::vector<ir::ExprPtr>{tile, MakeConstInt(5), MakeConstInt(42)}, ir::Span::Unknown());
    auto setval_stmt = std::make_shared<const ir::EvalStmt>(setval, ir::Span::Unknown());

    auto body = std::make_shared<const ir::SeqStmts>(std::vector<ir::StmtPtr>{tile_assign, setval_stmt},
                                                     ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a3");
    EXPECT_NE(generated.find(".SetValue(5, 42);"), std::string::npos);
}

TEST(BackendCceOpsTest, BlockGetValTile)
{
    auto tile_type = MakeTileType({16, 16}, ir::DataType::FP16);
    auto tile = MakeVar("tile", tile_type);
    auto make_tile = std::make_shared<const ir::Call>("block.make_tile", std::vector<ir::ExprPtr>{}, tile_type,
                                                      ir::Span::Unknown());
    auto tile_assign = std::make_shared<const ir::AssignStmt>(tile, make_tile, ir::Span::Unknown());

    auto int_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT32);
    auto result = MakeVar("val", int_type);
    auto getval = std::make_shared<const ir::Call>("block.getval", std::vector<ir::ExprPtr>{tile, MakeConstInt(3)},
                                                   int_type, ir::Span::Unknown());
    auto getval_assign = std::make_shared<const ir::AssignStmt>(result, getval, ir::Span::Unknown());

    auto body = std::make_shared<const ir::SeqStmts>(std::vector<ir::StmtPtr>{tile_assign, getval_assign},
                                                     ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a3");
    EXPECT_NE(generated.find(".GetValue(3)"), std::string::npos);
}

// ============================================================================
// Block getval / setval (tensor)
// ============================================================================

TEST(BackendCceOpsTest, BlockGetValTensor)
{
    auto tensor = MakeTensorVar("data", {64}, ir::DataType::FP32);
    auto int_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT32);
    auto result = MakeVar("val", int_type);
    auto call = std::make_shared<const ir::Call>("block.getval", std::vector<ir::ExprPtr>{tensor, MakeConstInt(5)},
                                                 int_type, ir::Span::Unknown());
    auto body = std::make_shared<const ir::AssignStmt>(result, call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body, {tensor}), "a3");
    EXPECT_NE(generated.find("*((__gm__ float*)"), std::string::npos);
    EXPECT_NE(generated.find("+ 5)"), std::string::npos);
}

TEST(BackendCceOpsTest, BlockSetValTensor)
{
    auto tensor = MakeTensorVar("data", {64}, ir::DataType::FP32);
    auto call = std::make_shared<const ir::Call>(
        "block.setval", std::vector<ir::ExprPtr>{tensor, MakeConstInt(5), MakeConstInt(42)}, ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body, {tensor}), "a3");
    EXPECT_NE(generated.find("*((__gm__ float*)"), std::string::npos);
    EXPECT_NE(generated.find("+ 5) = 42;"), std::string::npos);
}

// ============================================================================
// block.subview
// ============================================================================

TEST(BackendCceOpsTest, BlockSubviewUsesRegisteredAddressAndResultTarget)
{
    auto tile = MakeVar("tile", MakeTileType({16, 32}, ir::DataType::FP16, ir::MemorySpace::Vec));
    auto call = std::make_shared<const ir::Call>(
        "block.subview",
        std::vector<ir::ExprPtr>{tile, MakeTuple({MakeConstInt(3), MakeConstInt(0)}),
                                 MakeTuple({MakeConstInt(8), MakeConstInt(12)})},
        tile->GetType(), ir::Span::Unknown());
    auto* info = BackendCCE::Instance().GetOpInfo("block.subview");
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->pipe, ir::PipeType::MTE2);

    TestableCCECodegen codegen(ir::SectionKind::Vector);
    codegen.SetTileAddress("tile", "0x200");
    codegen.SetCurrentResultTarget("view");
    EXPECT_EQ(info->codegen_func(call, codegen), "view");

    auto generated = codegen.GetEmittedCode();
    EXPECT_NE(generated.find("TASSIGN(view, 0x200 + ((3)*32+(0))*2)"), std::string::npos);
    EXPECT_NE(generated.find("view.SetValidShape(8, 12)"), std::string::npos);
    EXPECT_TRUE(codegen.HasTileAddress("view"));
    EXPECT_EQ(codegen.GetTileAddress("view"), "0x200 + ((3)*32+(0))*2");
}

TEST(BackendCceOpsTest, BlockSubviewUsesDynamicTileGroupAddressAndGeneratedName)
{
    auto tile = MakeVar("tile", MakeTileType({16, 32}, ir::DataType::FP16, ir::MemorySpace::Vec));
    auto call = std::make_shared<const ir::Call>(
        "block.subview",
        std::vector<ir::ExprPtr>{tile, MakeTuple({MakeConstInt(5), MakeConstInt(0)}),
                                 MakeTuple({MakeConstInt(4), MakeConstInt(7)})},
        tile->GetType(), ir::Span::Unknown());
    auto* info = BackendCCE::Instance().GetOpInfo("block.subview");
    ASSERT_NE(info, nullptr);

    TestableCCECodegen codegen(ir::SectionKind::Vector);
    codegen.RegisterVarName(tile, "tiles[index]");
    auto result = info->codegen_func(call, codegen);

    EXPECT_EQ(result, "tiles_index__view_0");
    auto generated = codegen.GetEmittedCode();
    EXPECT_NE(generated.find("TASSIGN(tiles_index__view_0, (uint64_t)tiles[index].data() + ((5)*32+(0))*2)"),
              std::string::npos);
    EXPECT_NE(generated.find("tiles_index__view_0.SetValidShape(4, 7)"), std::string::npos);
}

TEST(BackendCceOpsTest, BlockSubviewFallsBackToMemRefAddress)
{
    auto addr = MakeConstInt(0x180, ir::DataType::INDEX);
    auto memref = std::make_shared<const ir::MemRef>(ir::MemorySpace::Vec, addr, 1024, 0, ir::Span::Unknown());
    auto tile_type = std::make_shared<const ir::TileType>(
        std::vector<int64_t>{8, 16}, ir::DataType::FP32, std::optional<ir::MemRefPtr>(memref),
        std::optional<ir::TileView>(std::nullopt), std::optional<ir::HardwareInfo>(ir::HardwareInfo{}));
    auto tile = MakeVar("tile", tile_type);
    auto call = std::make_shared<const ir::Call>(
        "block.subview",
        std::vector<ir::ExprPtr>{tile, MakeTuple({MakeConstInt(2), MakeConstInt(0)}),
                                 MakeTuple({MakeConstInt(3), MakeConstInt(5)})},
        tile_type, ir::Span::Unknown());
    auto* info = BackendCCE::Instance().GetOpInfo("block.subview");
    ASSERT_NE(info, nullptr);

    TestableCCECodegen codegen(ir::SectionKind::Vector);
    codegen.SetCurrentResultTarget("memref_view");
    EXPECT_EQ(info->codegen_func(call, codegen), "memref_view");
    EXPECT_NE(codegen.GetEmittedCode().find("TASSIGN(memref_view, 0x180 + ((2)*16+(0))*4)"), std::string::npos);
}

TEST(BackendCceOpsTest, BlockSubviewInVFSectionReturnsPointerArithmetic)
{
    auto tile = MakeVar("tile", MakeTileType({16, 32}, ir::DataType::FP16));
    auto call = std::make_shared<const ir::Call>(
        "block.subview",
        std::vector<ir::ExprPtr>{tile, MakeTuple({MakeConstInt(7), MakeConstInt(0)}),
                                 MakeTuple({MakeConstInt(8), MakeConstInt(8)})},
        tile->GetType(), ir::Span::Unknown());
    auto* info = BackendCCE::Instance().GetOpInfo("block.subview");
    ASSERT_NE(info, nullptr);

    TestableCCECodegen codegen(ir::SectionKind::Vector);
    codegen.SetInVFSection(true);
    EXPECT_EQ(info->codegen_func(call, codegen), "(vf_tile_ptr_0 + (7)*32+(0))");
    EXPECT_TRUE(codegen.GetEmittedCode().empty());
}

TEST(BackendCceOpsTest, BlockSubviewUsesDnElementOffset)
{
    ir::HardwareInfo hardware_info(ir::TileLayout::col_major);
    auto tile_type = std::make_shared<const ir::TileType>(
        std::vector<int64_t>{16, 32}, ir::DataType::FP16, std::optional<ir::MemRefPtr>(std::nullopt),
        std::optional<ir::TileView>(std::nullopt), std::optional<ir::HardwareInfo>(hardware_info));
    auto tile = MakeVar("tile", tile_type);
    auto call = std::make_shared<const ir::Call>(
        "block.subview",
        std::vector<ir::ExprPtr>{tile, MakeTuple({MakeConstInt(3), MakeConstInt(5)}),
                                 MakeTuple({MakeConstInt(8), MakeConstInt(12)})},
        tile_type, ir::Span::Unknown());
    const auto* info = BackendCCE::Instance().GetOpInfo("block.subview");
    ASSERT_NE(info, nullptr);

    TestableCCECodegen codegen(ir::SectionKind::Vector);
    codegen.SetTileAddress("tile", "0x200");
    codegen.SetCurrentResultTarget("view");
    EXPECT_EQ(info->codegen_func(call, codegen), "view");
    EXPECT_NE(codegen.GetEmittedCode().find("TASSIGN(view, 0x200 + ((5)*16+(3))*2)"), std::string::npos);
}

TEST(BackendCceOpsTest, BlockSubviewInVFSectionSupportsScalarOffset)
{
    auto tile = MakeVar("tile", MakeTileType({16, 32}, ir::DataType::FP16));
    auto call = std::make_shared<const ir::Call>(
        "block.subview", std::vector<ir::ExprPtr>{tile, MakeConstInt(7), MakeTuple({MakeConstInt(8), MakeConstInt(8)})},
        tile->GetType(), ir::Span::Unknown());
    const auto* info = BackendCCE::Instance().GetOpInfo("block.subview");
    ASSERT_NE(info, nullptr);

    TestableCCECodegen codegen(ir::SectionKind::Vector);
    codegen.SetInVFSection(true);
    EXPECT_EQ(info->codegen_func(call, codegen), "(vf_tile_ptr_0 + (7))");
}

// ============================================================================
// Sync all (HARD mode)
// ============================================================================

TEST(BackendCceOpsTest, SyncAllHardMix)
{
    auto empty_tuple = std::make_shared<const ir::MakeTuple>(std::vector<ir::ExprPtr>{}, ir::Span::Unknown());
    Kwargs kwargs = {{"mode", 0}};
    auto call = std::make_shared<const ir::Call>("system.sync_all", std::vector<ir::ExprPtr>{empty_tuple}, kwargs,
                                                 ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a3");
    EXPECT_NE(generated.find("SYNCALL<SyncCoreType::Mix>();"), std::string::npos);
}

TEST(BackendCceOpsTest, SyncAllHardAIVOnly)
{
    auto empty_tuple = std::make_shared<const ir::MakeTuple>(std::vector<ir::ExprPtr>{}, ir::Span::Unknown());
    Kwargs kwargs = {{"mode", 0}, {"core_type", 0}};
    auto call = std::make_shared<const ir::Call>("system.sync_all", std::vector<ir::ExprPtr>{empty_tuple}, kwargs,
                                                 ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a3");
    EXPECT_NE(generated.find("SYNCALL<SyncCoreType::AIVOnly>();"), std::string::npos);
}

TEST(BackendCceOpsTest, SyncAllSoftUsesCoreSpecificWorkspaces)
{
    auto gm = MakeTensorVar("gm", {64}, ir::DataType::UINT8);
    auto ub = MakeVar("ub", MakeTileType({16, 16}, ir::DataType::FP16, ir::MemorySpace::Vec));
    auto l1 = MakeVar("l1", MakeTileType({16, 16}, ir::DataType::FP16, ir::MemorySpace::Mat));
    auto used_cores = MakeVar("used_cores", std::make_shared<const ir::ScalarType>(ir::DataType::INT32));

    auto make_sync_all = [&](ir::SyncCoreType core_type, std::vector<ir::ExprPtr> workspaces) {
        Kwargs kwargs = {{"mode", static_cast<int>(ir::SyncAllMode::SOFT)}, {"core_type", static_cast<int>(core_type)}};
        return std::make_shared<const ir::Call>(
            "system.sync_all", std::vector<ir::ExprPtr>{MakeTuple(std::move(workspaces))}, kwargs, ir::Span::Unknown());
    };

    auto aiv_code = RunCodegen("system.sync_all", make_sync_all(ir::SyncCoreType::AIV_ONLY, {gm, ub, used_cores}));
    EXPECT_NE(aiv_code.find("SYNCALL<SyncAllMode::Soft, SyncCoreType::AIVOnly>(gm, ub, used_cores);"),
              std::string::npos);

    auto aic_code = RunCodegen("system.sync_all", make_sync_all(ir::SyncCoreType::AIC_ONLY, {gm, l1, used_cores}));
    EXPECT_NE(aic_code.find("SYNCALL<SyncAllMode::Soft, SyncCoreType::AICOnly>(gm, l1, used_cores);"),
              std::string::npos);

    auto mix_code = RunCodegen("system.sync_all", make_sync_all(ir::SyncCoreType::MIX, {gm, ub, l1, used_cores}));
    EXPECT_NE(mix_code.find("SYNCALL<SyncAllMode::Soft, SyncCoreType::Mix>(gm, ub, l1, used_cores);"),
              std::string::npos);
}

// ============================================================================
// DCCI (tensor)
// ============================================================================

TEST(BackendCceOpsTest, DcciTensorDefaultKwargs)
{
    auto tensor = MakeTensorVar("data", {64}, ir::DataType::FP32);
    auto call = std::make_shared<const ir::Call>("system.dcci", std::vector<ir::ExprPtr>{tensor}, ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body, {tensor}), "a3");
    EXPECT_NE(generated.find("dcci("), std::string::npos);
    EXPECT_NE(generated.find("ENTIRE_DATA_CACHE"), std::string::npos);
    EXPECT_NE(generated.find("CACHELINE_OUT"), std::string::npos);
}

TEST(BackendCceOpsTest, DcciTensorWithOffsetAndKwargs)
{
    auto tensor = MakeTensorVar("data", {64}, ir::DataType::FP32);
    Kwargs kwargs = {{"cache_line", 0}, {"dst", 2}};
    auto call = std::make_shared<const ir::Call>("system.dcci", std::vector<ir::ExprPtr>{tensor, MakeConstInt(16)},
                                                 kwargs, ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body, {tensor}), "a3");
    EXPECT_NE(generated.find("dcci("), std::string::npos);
    EXPECT_NE(generated.find("SINGLE_CACHE_LINE"), std::string::npos);
    EXPECT_NE(generated.find("CACHELINE_UB"), std::string::npos);
}

TEST(BackendCceOpsTest, DcciTensorWithMultidimensionalOffset)
{
    auto tensor = MakeTensorVar("data", {8, 16}, ir::DataType::FP32);
    auto call = std::make_shared<const ir::Call>(
        "system.dcci", std::vector<ir::ExprPtr>{tensor, MakeTuple({MakeConstInt(2), MakeConstInt(3)})},
        ir::Span::Unknown());

    auto generated = RunCodegen("system.dcci", call);
    EXPECT_NE(generated.find("2 * (16) + 3"), std::string::npos);
    EXPECT_NE(generated.find("CACHELINE_OUT"), std::string::npos);
}

// ============================================================================
// DCCI (tile)
// ============================================================================

TEST(BackendCceOpsTest, DcciTileDefaultKwargs)
{
    auto tile_type = MakeTileType({16, 16}, ir::DataType::FP16, ir::MemorySpace::Vec);
    auto tile = MakeVar("tile", tile_type);
    auto make_tile = std::make_shared<const ir::Call>("block.make_tile", std::vector<ir::ExprPtr>{}, tile_type,
                                                      ir::Span::Unknown());
    auto tile_assign = std::make_shared<const ir::AssignStmt>(tile, make_tile, ir::Span::Unknown());

    auto dcci = std::make_shared<const ir::Call>("system.dcci", std::vector<ir::ExprPtr>{tile}, ir::Span::Unknown());
    auto dcci_stmt = std::make_shared<const ir::EvalStmt>(dcci, ir::Span::Unknown());

    auto body = std::make_shared<const ir::SeqStmts>(std::vector<ir::StmtPtr>{tile_assign, dcci_stmt},
                                                     ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a3");
    EXPECT_NE(generated.find("dcci("), std::string::npos);
    EXPECT_NE(generated.find("__ubuf__ void*"), std::string::npos);
    EXPECT_NE(generated.find("ENTIRE_DATA_CACHE"), std::string::npos);
    EXPECT_NE(generated.find("CACHELINE_UB"), std::string::npos);
}

// ============================================================================
// Debug ops
// ============================================================================

TEST(BackendCceOpsTest, DebugDumpTensorStaticNdWindow)
{
    auto tensor = MakeTensorVar("data", {8, 16}, ir::DataType::FP32);
    Kwargs kwargs = {{"show_location", true}};
    auto call = std::make_shared<const ir::Call>(
        "debug.dump_tensor",
        std::vector<ir::ExprPtr>{tensor, MakeTuple({MakeConstInt(1), MakeConstInt(2)}),
                                 MakeTuple({MakeConstInt(4), MakeConstInt(5)})},
        kwargs, ir::Span("kernel.py", 7, 4));

    auto generated = RunCodegen("debug.dump_tensor", call);
    EXPECT_NE(generated.find("kernel.py"), std::string::npos);
    EXPECT_NE(generated.find("dump_tensor"), std::string::npos);
    EXPECT_NE(generated.find("pto::Shape<1, 1, 1, 4, 5>"), std::string::npos);
    EXPECT_NE(generated.find("pto::Stride<1, 1, 1, 16, 1>"), std::string::npos);
    EXPECT_NE(generated.find("Layout::ND"), std::string::npos);
    EXPECT_NE(generated.find("TPRINT(__debug_dump_tensor_view_"), std::string::npos);
}

TEST(BackendCceOpsTest, DebugDumpTensorDynamicFullWindowUsesRuntimeView)
{
    auto index_type = std::make_shared<const ir::ScalarType>(ir::DataType::INDEX);
    auto rows = MakeVar("rows", index_type);
    auto cols = MakeVar("cols", index_type);
    auto tensor = MakeDynamicTensorVar("data", std::vector<ir::ExprPtr>{rows, cols}, ir::DataType::FP16,
                                       ir::TensorLayout::ND);
    auto call = std::make_shared<const ir::Call>(
        "debug.dump_tensor",
        std::vector<ir::ExprPtr>{tensor, MakeTuple({MakeConstInt(0), MakeConstInt(0)}), MakeTuple({rows, cols})},
        ir::Span::Unknown());

    auto generated = RunCodegen("debug.dump_tensor", call);
    EXPECT_NE(generated.find("pto::Shape<1, 1, 1, -1, -1>"), std::string::npos);
    EXPECT_NE(generated.find("data.GetShape(GlobalTensorDim::DIM_3)"), std::string::npos);
    EXPECT_NE(generated.find("data.GetShape(GlobalTensorDim::DIM_4)"), std::string::npos);
    EXPECT_NE(generated.find("pto::Stride<1, 1, 1, -1, -1>"), std::string::npos);
    EXPECT_NE(generated.find("data.GetStride(GlobalTensorDim::DIM_3)"), std::string::npos);
    EXPECT_NE(generated.find("data.GetStride(GlobalTensorDim::DIM_4)"), std::string::npos);
}

TEST(BackendCceOpsTest, DebugDumpTensorUsesExplicitDynamicStride)
{
    auto index_type = std::make_shared<const ir::ScalarType>(ir::DataType::INDEX);
    auto row_stride = MakeVar("row_stride", index_type);
    auto tensor = MakeTensorVar("strided", {8, 16}, ir::DataType::FP16, ir::TensorLayout::ND,
                                {row_stride, MakeConstInt(1)});
    auto call = std::make_shared<const ir::Call>(
        "debug.dump_tensor",
        std::vector<ir::ExprPtr>{tensor, MakeTuple({MakeConstInt(0), MakeConstInt(0)}),
                                 MakeTuple({MakeConstInt(4), MakeConstInt(8)})},
        ir::Span::Unknown());

    auto generated = RunCodegen("debug.dump_tensor", call);
    EXPECT_NE(generated.find("pto::Stride<1, 1, 1, -1, 1>"), std::string::npos);
    EXPECT_NE(generated.find("__debug_dump_tensor_stride_"), std::string::npos);
    EXPECT_NE(generated.find("(row_stride)"), std::string::npos);
}

TEST(BackendCceOpsTest, DebugDumpTensorSelectsDnLayout)
{
    auto tensor = MakeTensorVar("column", {32, 1}, ir::DataType::FP16);
    auto call = std::make_shared<const ir::Call>(
        "debug.dump_tensor",
        std::vector<ir::ExprPtr>{tensor, MakeTuple({MakeConstInt(0), MakeConstInt(0)}),
                                 MakeTuple({MakeConstInt(32), MakeConstInt(1)})},
        ir::Span::Unknown());

    auto generated = RunCodegen("debug.dump_tensor", call);
    EXPECT_NE(generated.find("Layout::DN"), std::string::npos);
}

TEST(BackendCceOpsTest, DebugDumpTensorNzStaticAlignedWindow)
{
    auto tensor = MakeTensorVar("nz_data", {32, 64}, ir::DataType::FP16, ir::TensorLayout::NZ);
    auto call = std::make_shared<const ir::Call>(
        "debug.dump_tensor",
        std::vector<ir::ExprPtr>{tensor, MakeTuple({MakeConstInt(16), MakeConstInt(16)}),
                                 MakeTuple({MakeConstInt(16), MakeConstInt(32)})},
        ir::Span::Unknown());

    auto generated = RunCodegen("debug.dump_tensor", call);
    EXPECT_NE(generated.find("pto::Shape<1, 2, 1, 16, 16>"), std::string::npos);
    EXPECT_NE(generated.find("pto::Stride<2048, 512, 256, 16, 1>"), std::string::npos);
    EXPECT_NE(generated.find("Layout::NZ"), std::string::npos);
    EXPECT_NE(generated.find("nz_data.data() + 768"), std::string::npos);
}

TEST(BackendCceOpsTest, DebugDumpTensorNzMisalignedWindowUsesDynamicView)
{
    auto tensor = MakeTensorVar("nz_data", {32, 64}, ir::DataType::FP16, ir::TensorLayout::NZ);
    auto call = std::make_shared<const ir::Call>(
        "debug.dump_tensor",
        std::vector<ir::ExprPtr>{tensor, MakeTuple({MakeConstInt(1), MakeConstInt(0)}),
                                 MakeTuple({MakeConstInt(16), MakeConstInt(16)})},
        ir::Span::Unknown());

    auto generated = RunCodegen("debug.dump_tensor", call);
    EXPECT_NE(generated.find("pto::TileShape2D<"), std::string::npos);
    EXPECT_NE(generated.find("pto::DYNAMIC, pto::DYNAMIC, Layout::NZ"), std::string::npos);
    EXPECT_NE(generated.find("(0 * 32 + 1 * 16)"), std::string::npos);
    EXPECT_NE(generated.find("TPRINT(__debug_dump_tensor_view_"), std::string::npos);
}

TEST(BackendCceOpsTest, DebugDumpTileFullAndWindow)
{
    auto tile = MakeVar("tile", MakeTileType({16, 32}, ir::DataType::FP16));
    auto full_call = std::make_shared<const ir::Call>("debug.dump_tile", std::vector<ir::ExprPtr>{tile},
                                                      ir::Span::Unknown());
    EXPECT_NE(RunCodegen("debug.dump_tile", full_call).find("TPRINT(tile);"), std::string::npos);

    auto window_call = std::make_shared<const ir::Call>(
        "debug.dump_tile",
        std::vector<ir::ExprPtr>{tile, MakeTuple({MakeConstInt(2), MakeConstInt(3)}),
                                 MakeTuple({MakeConstInt(8), MakeConstInt(12)})},
        ir::Span::Unknown());
    auto generated = RunCodegen("debug.dump_tile", window_call);
    EXPECT_NE(generated.find("[TPRINT Tile Window]"), std::string::npos);
    EXPECT_NE(generated.find("tile.GetValidRow() - (2)"), std::string::npos);
    EXPECT_NE(generated.find("pto::GetTileOffset<std::remove_reference_t<decltype(tile)>>"), std::string::npos);
    EXPECT_NE(generated.find("__pypto_dtype_name"), std::string::npos);
    EXPECT_NE(generated.find("__pypto_print_val"), std::string::npos);
}

TEST(BackendCceOpsTest, DebugDumpAccTileUsesWorkspace)
{
    auto tile = MakeVar("acc", MakeTileType({16, 32}, ir::DataType::FP16, ir::MemorySpace::Mat));
    auto workspace = MakeTensorVar("workspace", {512}, ir::DataType::FP16);
    auto call = std::make_shared<const ir::Call>(
        "debug.dump_tile",
        std::vector<ir::ExprPtr>{tile, MakeTuple({MakeConstInt(1), MakeConstInt(2)}),
                                 MakeTuple({MakeConstInt(8), MakeConstInt(16)}), workspace},
        ir::Span::Unknown());

    TestableCCECodegen codegen(ir::SectionKind::Vector);
    codegen.RegisterPointer("workspace", "workspace_ptr");
    auto generated = RunCodegen("debug.dump_tile", call, codegen);
    EXPECT_NE(generated.find("copy_matrix_cc_to_gm"), std::string::npos);
    EXPECT_NE(generated.find("reinterpret_cast<__gm__"), std::string::npos);
    EXPECT_NE(generated.find("workspace_ptr"), std::string::npos);
    EXPECT_NE(generated.find("[TPRINT Acc Tile Window]"), std::string::npos);
    EXPECT_NE(generated.find("__pypto_dtype_name"), std::string::npos);
    EXPECT_NE(generated.find("__pypto_print_val"), std::string::npos);
}

TEST(BackendCceOpsTest, DebugPrintfExpandsUnsigned64BitValues)
{
    auto u64_type = std::make_shared<const ir::ScalarType>(ir::DataType::UINT64);
    auto decimal = MakeVar("decimal", u64_type);
    auto hexadecimal = MakeVar("hexadecimal", u64_type);
    Kwargs kwargs = {{"format", std::string("decimal=%u hex=%#x\n")}};
    auto call = std::make_shared<const ir::Call>("debug.printf", std::vector<ir::ExprPtr>{decimal, hexadecimal}, kwargs,
                                                 ir::Span::Unknown());

    auto generated = RunCodegen("debug.printf", call);
    EXPECT_NE(generated.find("__pypto_printf_u64_rest_"), std::string::npos);
    EXPECT_NE(generated.find("%u%09u%09u"), std::string::npos);
    EXPECT_NE(generated.find(">> 32"), std::string::npos);
    EXPECT_NE(generated.find("0x%x%08x"), std::string::npos);
}

TEST(BackendCceOpsTest, DebugPrintfRewritesPromotedScalarTypes)
{
    auto signed_value = MakeVar("signed_value", std::make_shared<const ir::ScalarType>(ir::DataType::INT64));
    auto small = MakeVar("small", std::make_shared<const ir::ScalarType>(ir::DataType::INT8));
    auto unsigned_value = MakeVar("unsigned_value", std::make_shared<const ir::ScalarType>(ir::DataType::UINT16));
    auto flag = MakeVar("flag", std::make_shared<const ir::ScalarType>(ir::DataType::BOOL));
    auto ptr = MakeVar("ptr", std::make_shared<const ir::PtrType>(ir::DataType::FP32));
    auto fp = MakeVar("fp", std::make_shared<const ir::ScalarType>(ir::DataType::FP32));
    Kwargs kwargs = {{"format", std::string("signed=%d small=%d unsigned=%u flag=%u ptr=%p float=%f\n")}};
    auto call = std::make_shared<const ir::Call>(
        "debug.printf", std::vector<ir::ExprPtr>{signed_value, small, unsigned_value, flag, ptr, fp}, kwargs,
        ir::Span::Unknown());

    auto generated = RunCodegen("debug.printf", call);
    EXPECT_NE(generated.find("signed=%lld small=%d unsigned=%u flag=%u ptr=%lld float=%f"), std::string::npos);
    EXPECT_NE(generated.find("static_cast<long long>(signed_value)"), std::string::npos);
    EXPECT_NE(generated.find("static_cast<int>(small)"), std::string::npos);
    EXPECT_NE(generated.find("static_cast<unsigned int>(unsigned_value)"), std::string::npos);
    EXPECT_NE(generated.find("static_cast<unsigned int>(flag)"), std::string::npos);
    EXPECT_NE(generated.find("static_cast<long long>((uint64_t)ptr)"), std::string::npos);
}

TEST(BackendCceOpsTest, DebugPrintf)
{
    Kwargs kwargs = {{"format", std::string("value=%d\n")}};
    auto call = std::make_shared<const ir::Call>("debug.printf", std::vector<ir::ExprPtr>{MakeConstInt(42)}, kwargs,
                                                 ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a3");
    EXPECT_NE(generated.find("cce::printf"), std::string::npos);
}

TEST(BackendCceOpsTest, DebugAssert)
{
    Kwargs kwargs = {{"condition_text", std::string("x > 0")}, {"format", std::string("")}, {"show_location", true}};
    auto call = std::make_shared<const ir::Call>("debug.assert", std::vector<ir::ExprPtr>{MakeConstInt(1)}, kwargs,
                                                 ir::Span("kernel.py", 12, 3));
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a3");
    EXPECT_NE(generated.find("if (!("), std::string::npos);
    EXPECT_NE(generated.find("kernel.py"), std::string::npos);
    EXPECT_NE(generated.find("Assertion failed: x > 0"), std::string::npos);
}

TEST(BackendCceOpsTest, DebugAssertWithFormatAndArgs)
{
    Kwargs kwargs = {{"condition_text", std::string("x > 0")}, {"format", std::string("x=%d\n")}};
    auto call = std::make_shared<const ir::Call>(
        "debug.assert", std::vector<ir::ExprPtr>{MakeConstInt(1), MakeConstInt(99)}, kwargs, ir::Span::Unknown());
    auto body = std::make_shared<const ir::EvalStmt>(call, ir::Span::Unknown());

    codegen::CCECodegen codegen(ir::SectionKind::Vector);
    auto generated = codegen.GenerateSingle(MakeProgram(body), "a3");
    EXPECT_NE(generated.find("if (!("), std::string::npos);
    EXPECT_NE(generated.find("cce::printf"), std::string::npos);
}

} // namespace backend
} // namespace pypto
