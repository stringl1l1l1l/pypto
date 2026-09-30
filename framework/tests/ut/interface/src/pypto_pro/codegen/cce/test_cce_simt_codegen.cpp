/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_cce_simt_codegen.cpp
 * \brief End-to-end CCE codegen tests for SIMT functions, callees, context operations, and launch.
 */

#include "gtest/gtest.h"

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "codegen/cce/cce_codegen.h"
#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/memref.h"
#include "ir/op_attr_types.h"
#include "ir/op_registry.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/type.h"
#include "tilefwk/error.h"

namespace pypto {
namespace codegen {
namespace {

using Kwargs = std::vector<std::pair<std::string, std::any>>;

ir::Span Sp() { return ir::Span::Unknown(); }

ir::ExprPtr ConstInt(int64_t value) { return std::make_shared<const ir::ConstInt>(value, ir::DataType::INT64, Sp()); }

ir::TypePtr Scalar(ir::DataType dtype) { return std::make_shared<const ir::ScalarType>(dtype); }

ir::VarPtr Var(const std::string& name, const ir::TypePtr& type)
{
    return std::make_shared<const ir::Var>(name, type, Sp());
}

ir::TileTypePtr MakeSimtTileType(ir::DataType dtype = ir::DataType::FP32, int64_t address = 0)
{
    auto memref = std::make_shared<const ir::MemRef>(ir::MemorySpace::Vec, ConstInt(address), 1024);
    return std::make_shared<const ir::TileType>(
        std::vector<int64_t>{1, 256}, dtype, std::optional<ir::MemRefPtr>(memref),
        std::optional<ir::TileView>(std::nullopt), std::optional<ir::HardwareInfo>(ir::HardwareInfo{}));
}

ir::TensorTypePtr MakeSimtTensorType(const std::string& name, ir::DataType dtype = ir::DataType::FP32)
{
    auto ptr = Var(name + "_base", std::make_shared<const ir::PtrType>(dtype));
    ir::TensorView view({}, ir::TensorLayout::ND, ptr);
    return std::make_shared<const ir::TensorType>(std::vector<int64_t>{256}, dtype,
                                                  std::optional<ir::MemRefPtr>(std::nullopt),
                                                  std::optional<ir::TensorView>(view));
}

ir::CallPtr Call(const std::string& name, std::vector<ir::ExprPtr> args, const ir::TypePtr& type)
{
    return std::make_shared<const ir::Call>(name, std::move(args), type, Sp());
}

ir::CallPtr Call(const std::string& name, std::vector<ir::ExprPtr> args, Kwargs kwargs, const ir::TypePtr& type)
{
    return std::make_shared<const ir::Call>(name, std::move(args), std::move(kwargs), type, Sp());
}

ir::CallPtr RegisteredCall(const std::string& name, const std::vector<ir::ExprPtr>& args, const Kwargs& kwargs = {})
{
    return ir::OpRegistry::GetInstance().Create(name, args, kwargs, Sp());
}

ir::StmtPtr Eval(const ir::ExprPtr& expr) { return std::make_shared<const ir::EvalStmt>(expr, Sp()); }

ir::StmtPtr Assign(const ir::VarPtr& var, const ir::ExprPtr& value)
{
    return std::make_shared<const ir::AssignStmt>(var, value, Sp());
}

ir::StmtPtr Seq(std::vector<ir::StmtPtr> stmts) { return std::make_shared<const ir::SeqStmts>(std::move(stmts), Sp()); }

void AppendRegisteredResult(std::vector<ir::StmtPtr>& stmts, std::size_t& result_index, const std::string& name,
                            const std::vector<ir::ExprPtr>& args, const Kwargs& kwargs = {})
{
    auto call = RegisteredCall(name, args, kwargs);
    auto result = Var("result_" + std::to_string(result_index++), call->GetType());
    stmts.push_back(Assign(result, call));
}

std::string FindLineContaining(const std::string& text, const std::string& needle)
{
    auto pos = text.find(needle);
    if (pos == std::string::npos) {
        return "";
    }
    auto line_begin = text.rfind('\n', pos);
    line_begin = line_begin == std::string::npos ? 0 : line_begin + 1;
    auto line_end = text.find('\n', pos);
    return text.substr(line_begin, line_end - line_begin);
}

std::size_t CountOccurrences(const std::string& text, const std::string& needle)
{
    std::size_t count = 0;
    std::size_t pos = 0;
    while ((pos = text.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}

ir::FunctionPtr MakeScalarAddCallee(const ir::TypePtr& fp32)
{
    auto value = Var("value", fp32);
    auto delta = Var("delta", fp32);
    auto sum = std::make_shared<const ir::Add>(value, delta, ir::DataType::FP32, Sp());
    auto body = std::make_shared<const ir::ReturnStmt>(std::vector<ir::ExprPtr>{sum}, Sp());
    return std::make_shared<const ir::Function>("add_scalar", std::vector<ir::VarPtr>{value, delta},
                                                std::vector<ir::TypePtr>{fp32}, body, Sp(),
                                                ir::FunctionType::SIMT_CALLEE);
}

ir::FunctionPtr MakeStoreCallee(const ir::TileTypePtr& tile_type, const ir::TypePtr& u32, const ir::TypePtr& fp32)
{
    auto tile = Var("tile", tile_type);
    auto index = Var("index", u32);
    auto value = Var("value", fp32);
    auto store = Call("block.setval", {tile, index, value}, ir::GetUnknownType());
    auto body = Seq({Eval(store), std::make_shared<const ir::ReturnStmt>(Sp())});
    return std::make_shared<const ir::Function>("store_scalar", std::vector<ir::VarPtr>{tile, index, value},
                                                std::vector<ir::TypePtr>{}, body, Sp(), ir::FunctionType::SIMT_CALLEE);
}

ir::FunctionPtr MakeApplyCallee(const ir::TileTypePtr& tile_type, const ir::TypePtr& u32, const ir::TypePtr& fp32)
{
    auto tile = Var("tile", tile_type);
    auto index = Var("index", u32);
    auto delta = Var("delta", fp32);
    auto value = Var("value", fp32);
    auto updated = Var("updated", fp32);
    auto load = Call("block.getval", {tile, index}, fp32);
    auto add = Call("add_scalar", {value, delta}, fp32);
    auto store = Call("store_scalar", {tile, index, updated}, ir::GetUnknownType());
    auto body = Seq({Assign(value, load), Assign(updated, add), Eval(store),
                     std::make_shared<const ir::ReturnStmt>(std::vector<ir::ExprPtr>{updated}, Sp())});
    return std::make_shared<const ir::Function>("apply_scalar", std::vector<ir::VarPtr>{tile, index, delta},
                                                std::vector<ir::TypePtr>{fp32}, body, Sp(),
                                                ir::FunctionType::SIMT_CALLEE);
}

ir::FunctionPtr MakeSimtEntry(const ir::TileTypePtr& tile_type, const ir::TensorTypePtr& tensor_type,
                              const ir::TypePtr& u32, const ir::TypePtr& i32, const ir::TypePtr& fp32)
{
    auto tile = Var("tile", tile_type);
    auto tensor = Var("tensor", tensor_type);
    auto count = Var("count", u32);
    auto delta = Var("delta", fp32);
    auto thread_x = Var("thread_x", u32);
    auto block_y = Var("block_y", u32);
    auto block_z = Var("block_z", u32);
    auto grid_x = Var("grid_x", u32);
    auto tid = Var("tid", u32);
    auto warp = Var("warp", i32);
    auto rows = Var("rows", u32);
    auto cols = Var("cols", u32);
    auto updated = Var("updated", fp32);

    auto context_call = [&](const std::string& name, int axis) { return Call(name, {}, Kwargs{{"axis", axis}}, u32); };
    auto body = Seq({
        Assign(thread_x, context_call("simt.thread_idx", 0)),
        Assign(block_y, context_call("simt.block_dim", 1)),
        Assign(block_z, context_call("simt.block_idx", 2)),
        Assign(grid_x, context_call("simt.grid_dim", 0)),
        Assign(tid, Call("simt.linear_thread_idx", {}, u32)),
        Assign(warp, Call("simt.warp_size", {}, i32)),
        Assign(rows, Call("block.tile_valid_shape", {tile}, Kwargs{{"axis", 0}}, u32)),
        Assign(cols, Call("block.tile_valid_shape", {tile}, Kwargs{{"axis", 1}}, u32)),
        Assign(updated, Call("apply_scalar", {tile, tid, delta}, fp32)),
        Eval(Call("block.setval", {tensor, tid, updated}, ir::GetUnknownType())),
    });
    return std::make_shared<const ir::Function>(
        "simt_entry", std::vector<ir::VarPtr>{tile, tensor, count, delta}, std::vector<ir::TypePtr>{}, body, Sp(),
        ir::FunctionType::SIMT_VF, false, std::vector<std::pair<std::string, std::any>>{{ir::kMaxThreadsAttr, 256}});
}

ir::ProgramPtr MakeSimtProgram()
{
    auto u32 = Scalar(ir::DataType::UINT32);
    auto i32 = Scalar(ir::DataType::INT32);
    auto fp32 = Scalar(ir::DataType::FP32);
    auto tile_type = MakeSimtTileType();
    auto tensor_type = MakeSimtTensorType("tensor");

    auto tensor = Var("tensor", tensor_type);
    auto count = Var("count", u32);
    auto delta = Var("delta", fp32);
    auto tile = Var("tile", tile_type);
    auto make_tile = Call("block.make_tile", {}, tile_type);
    auto launch = Call("simt.launch", {ConstInt(8), ConstInt(4), ConstInt(8), tile, tensor, count, delta},
                       Kwargs{{"callee", std::string("simt_entry")}, {"max_threads", 256}}, ir::GetUnknownType());
    auto kernel_body = Seq({Assign(tile, make_tile), Eval(launch)});
    auto kernel = std::make_shared<const ir::Function>("kernel", std::vector<ir::VarPtr>{tensor, count, delta},
                                                       std::vector<ir::TypePtr>{}, kernel_body, Sp(),
                                                       ir::FunctionType::IN_CORE, true);

    std::vector<ir::FunctionPtr> functions = {MakeScalarAddCallee(fp32), MakeStoreCallee(tile_type, u32, fp32),
                                              MakeApplyCallee(tile_type, u32, fp32),
                                              MakeSimtEntry(tile_type, tensor_type, u32, i32, fp32), kernel};
    return std::make_shared<const ir::Program>(std::move(functions), "simt_program", Sp(),
                                               std::make_shared<ir::IRDebugInfo>());
}

ir::ProgramPtr MakeSimtCalleeNameCollisionProgram()
{
    auto i32 = Scalar(ir::DataType::INT32);
    auto callee_body = std::make_shared<const ir::ReturnStmt>(Sp());
    auto callee = std::make_shared<const ir::Function>("get_subblock_idx", std::vector<ir::VarPtr>{},
                                                       std::vector<ir::TypePtr>{}, callee_body, Sp(),
                                                       ir::FunctionType::SIMT_CALLEE);

    auto simt_body = Eval(Call("get_subblock_idx", {}, ir::GetUnknownType()));
    auto simt_entry = std::make_shared<const ir::Function>(
        "collision_entry", std::vector<ir::VarPtr>{}, std::vector<ir::TypePtr>{}, simt_body, Sp(),
        ir::FunctionType::SIMT_VF, false, std::vector<std::pair<std::string, std::any>>{{ir::kMaxThreadsAttr, 32}});

    auto subblock_idx = Var("subblock_idx", i32);
    auto backend_call = Call("get_subblock_idx", {}, i32);
    auto launch = Call("simt.launch", {ConstInt(32), ConstInt(1), ConstInt(1)},
                       Kwargs{{"callee", std::string("collision_entry")}, {"max_threads", 32}}, ir::GetUnknownType());
    auto kernel_body = Seq({Assign(subblock_idx, backend_call), Eval(backend_call), Eval(launch)});
    auto kernel = std::make_shared<const ir::Function>("kernel", std::vector<ir::VarPtr>{}, std::vector<ir::TypePtr>{},
                                                       kernel_body, Sp(), ir::FunctionType::IN_CORE, true);

    return std::make_shared<const ir::Program>(std::vector<ir::FunctionPtr>{callee, simt_entry, kernel},
                                               "simt_callee_name_collision", Sp(), std::make_shared<ir::IRDebugInfo>());
}

ir::ProgramPtr MakeSimtScalarOpsProgram()
{
    auto fp16 = Scalar(ir::DataType::FP16);
    auto bf16 = Scalar(ir::DataType::BF16);
    auto fp32 = Scalar(ir::DataType::FP32);
    auto i32 = Scalar(ir::DataType::INT32);
    auto u32 = Scalar(ir::DataType::UINT32);
    auto i64 = Scalar(ir::DataType::INT64);
    auto u64 = Scalar(ir::DataType::UINT64);

    auto fp16_value = Var("fp16_value", fp16);
    auto bf16_value = Var("bf16_value", bf16);
    auto fp32_value = Var("fp32_value", fp32);
    auto i32_value = Var("i32_value", i32);
    auto u32_value = Var("u32_value", u32);
    auto i64_value = Var("i64_value", i64);
    auto u64_value = Var("u64_value", u64);

    std::vector<ir::StmtPtr> simt_stmts;
    for (const auto& op_name : {"simt.syncthreads", "simt.threadfence_block", "simt.threadfence"}) {
        simt_stmts.push_back(Eval(RegisteredCall(op_name, {})));
    }
    std::size_t result_index = 0;
    const std::vector<std::string> dtype_specific_unary_ops = {
        "simt.abs",  "simt.sqrt", "simt.rsqrt", "simt.exp",  "simt.exp2",  "simt.log",
        "simt.log2", "simt.sin",  "simt.cos",   "simt.tanh", "simt.round", "simt.trunc",
    };
    for (const auto& op_name : dtype_specific_unary_ops) {
        AppendRegisteredResult(simt_stmts, result_index, op_name, {fp32_value});
        AppendRegisteredResult(simt_stmts, result_index, op_name, {fp16_value});
        AppendRegisteredResult(simt_stmts, result_index, op_name, {bf16_value});
    }
    for (const auto& op_name : {"simt.rint", "simt.floor", "simt.ceil", "simt.isnan", "simt.isinf"}) {
        AppendRegisteredResult(simt_stmts, result_index, op_name, {fp32_value});
    }
    AppendRegisteredResult(simt_stmts, result_index, "simt.log1p", {fp32_value});
    AppendRegisteredResult(simt_stmts, result_index, "simt.abs", {i64_value});
    AppendRegisteredResult(simt_stmts, result_index, "simt.isfinite", {fp16_value});
    AppendRegisteredResult(simt_stmts, result_index, "simt.isfinite", {fp32_value});
    AppendRegisteredResult(simt_stmts, result_index, "simt.popcount", {u32_value});
    AppendRegisteredResult(simt_stmts, result_index, "simt.popcount", {u64_value});
    AppendRegisteredResult(simt_stmts, result_index, "simt.mul_hi", {i32_value, i32_value});
    AppendRegisteredResult(simt_stmts, result_index, "simt.mul_hi", {u32_value, u32_value});
    AppendRegisteredResult(simt_stmts, result_index, "simt.mul_hi", {i64_value, i64_value});
    AppendRegisteredResult(simt_stmts, result_index, "simt.mul_hi", {u64_value, u64_value});
    AppendRegisteredResult(simt_stmts, result_index, "simt.fmod", {fp32_value, fp32_value});

    for (const auto& op_name : {"simt.min", "simt.max"}) {
        AppendRegisteredResult(simt_stmts, result_index, op_name, {fp32_value, fp32_value});
        AppendRegisteredResult(simt_stmts, result_index, op_name, {fp16_value, fp16_value});
        AppendRegisteredResult(simt_stmts, result_index, op_name, {bf16_value, bf16_value});
        AppendRegisteredResult(simt_stmts, result_index, op_name, {i64_value, i64_value});
    }
    AppendRegisteredResult(simt_stmts, result_index, "simt.fma", {fp32_value, fp32_value, fp32_value});

    auto append_cast = [&](const ir::ExprPtr& value, ir::DataType target_dtype, ir::RoundMode mode) {
        AppendRegisteredResult(simt_stmts, result_index, "simt.cast", {value},
                               Kwargs{{"target_type", target_dtype}, {"mode", static_cast<int>(mode)}});
    };
    const std::vector<ir::RoundMode> fp16_round_modes = {
        ir::RoundMode::CAST_NONE, ir::RoundMode::CAST_RINT,  ir::RoundMode::CAST_ROUND, ir::RoundMode::CAST_FLOOR,
        ir::RoundMode::CAST_CEIL, ir::RoundMode::CAST_TRUNC, ir::RoundMode::CAST_ODD,
    };
    for (const auto mode : fp16_round_modes) {
        append_cast(fp32_value, ir::DataType::FP16, mode);
    }
    append_cast(fp32_value, ir::DataType::BF16, ir::RoundMode::CAST_ROUND);
    append_cast(fp16_value, ir::DataType::FP32, ir::RoundMode::CAST_NONE);
    append_cast(bf16_value, ir::DataType::FP32, ir::RoundMode::CAST_NONE);
    append_cast(fp32_value, ir::DataType::INT32, ir::RoundMode::CAST_RINT);
    append_cast(fp32_value, ir::DataType::UINT32, ir::RoundMode::CAST_ROUND);
    append_cast(fp32_value, ir::DataType::INT64, ir::RoundMode::CAST_FLOOR);
    append_cast(fp32_value, ir::DataType::UINT64, ir::RoundMode::CAST_CEIL);
    append_cast(i32_value, ir::DataType::FP32, ir::RoundMode::CAST_TRUNC);
    append_cast(i64_value, ir::DataType::INT32, ir::RoundMode::CAST_NONE);
    append_cast(fp32_value, ir::DataType::FP32, ir::RoundMode::CAST_NONE);

    std::vector<ir::VarPtr> simt_args = {fp16_value, bf16_value, fp32_value, i32_value,
                                         u32_value,  i64_value,  u64_value};
    auto simt_function = std::make_shared<const ir::Function>(
        "scalar_ops", simt_args, std::vector<ir::TypePtr>{}, Seq(std::move(simt_stmts)), Sp(),
        ir::FunctionType::SIMT_VF, false, std::vector<std::pair<std::string, std::any>>{{ir::kMaxThreadsAttr, 1}});

    auto kernel_fp16 = Var("fp16_value", fp16);
    auto kernel_bf16 = Var("bf16_value", bf16);
    auto kernel_fp32 = Var("fp32_value", fp32);
    auto kernel_i32 = Var("i32_value", i32);
    auto kernel_u32 = Var("u32_value", u32);
    auto kernel_i64 = Var("i64_value", i64);
    auto kernel_u64 = Var("u64_value", u64);
    std::vector<ir::VarPtr> kernel_args = {kernel_fp16, kernel_bf16, kernel_fp32, kernel_i32,
                                           kernel_u32,  kernel_i64,  kernel_u64};
    auto launch = RegisteredCall("simt.launch",
                                 {ConstInt(1), ConstInt(1), ConstInt(1), kernel_fp16, kernel_bf16, kernel_fp32,
                                  kernel_i32, kernel_u32, kernel_i64, kernel_u64},
                                 Kwargs{{"callee", std::string("scalar_ops")}, {"max_threads", 1}});
    auto kernel = std::make_shared<const ir::Function>("kernel", kernel_args, std::vector<ir::TypePtr>{}, Eval(launch),
                                                       Sp(), ir::FunctionType::IN_CORE, true);

    return std::make_shared<const ir::Program>(std::vector<ir::FunctionPtr>{simt_function, kernel}, "simt_scalar_ops",
                                               Sp(), std::make_shared<ir::IRDebugInfo>());
}

ir::ProgramPtr MakeSimtAtomicOpsProgram()
{
    auto i32 = Scalar(ir::DataType::INT32);
    auto u32 = Scalar(ir::DataType::UINT32);
    auto u64 = Scalar(ir::DataType::UINT64);
    auto fp16 = Scalar(ir::DataType::FP16);
    auto bf16 = Scalar(ir::DataType::BF16);
    auto i32_tile_type = MakeSimtTileType(ir::DataType::INT32, 0);
    auto fp16_tile_type = MakeSimtTileType(ir::DataType::FP16, 1024);
    auto u64_tensor_type = MakeSimtTensorType("u64_tensor", ir::DataType::UINT64);
    auto bf16_tensor_type = MakeSimtTensorType("bf16_tensor", ir::DataType::BF16);

    auto i32_tile = Var("i32_tile", i32_tile_type);
    auto fp16_tile = Var("fp16_tile", fp16_tile_type);
    auto u64_tensor = Var("u64_tensor", u64_tensor_type);
    auto bf16_tensor = Var("bf16_tensor", bf16_tensor_type);
    auto offset = Var("offset", u32);
    auto i32_value = Var("i32_value", i32);
    auto u64_value = Var("u64_value", u64);
    auto fp16_value = Var("fp16_value", fp16);
    auto bf16_value = Var("bf16_value", bf16);

    std::vector<ir::StmtPtr> simt_stmts;
    auto append_atomic = [&](const std::string& name, const std::vector<ir::ExprPtr>& args) {
        simt_stmts.push_back(Eval(RegisteredCall(name, args)));
    };
    for (const auto& op_name : {"simt.atomic_add", "simt.atomic_sub", "simt.atomic_exch", "simt.atomic_max",
                                "simt.atomic_min", "simt.atomic_and", "simt.atomic_or", "simt.atomic_xor"}) {
        append_atomic(op_name, {i32_tile, offset, i32_value});
    }
    append_atomic("simt.atomic_cas", {i32_tile, offset, i32_value, i32_value});
    append_atomic("simt.atomic_inc", {u64_tensor, offset, u64_value});
    append_atomic("simt.atomic_dec", {u64_tensor, offset, u64_value});
    for (const auto& op_name : {"simt.atomic_add", "simt.atomic_max", "simt.atomic_min"}) {
        append_atomic(op_name, {fp16_tile, offset, fp16_value});
        append_atomic(op_name, {bf16_tensor, offset, bf16_value});
    }

    std::vector<ir::VarPtr> simt_args = {i32_tile,  fp16_tile, u64_tensor, bf16_tensor, offset,
                                         i32_value, u64_value, fp16_value, bf16_value};
    auto simt_function = std::make_shared<const ir::Function>(
        "atomic_ops", simt_args, std::vector<ir::TypePtr>{}, Seq(std::move(simt_stmts)), Sp(),
        ir::FunctionType::SIMT_VF, false, std::vector<std::pair<std::string, std::any>>{{ir::kMaxThreadsAttr, 32}});

    auto kernel_u64_tensor = Var("u64_tensor", u64_tensor_type);
    auto kernel_bf16_tensor = Var("bf16_tensor", bf16_tensor_type);
    auto kernel_offset = Var("offset", u32);
    auto kernel_i32_value = Var("i32_value", i32);
    auto kernel_u64_value = Var("u64_value", u64);
    auto kernel_fp16_value = Var("fp16_value", fp16);
    auto kernel_bf16_value = Var("bf16_value", bf16);
    auto kernel_i32_tile = Var("i32_tile", i32_tile_type);
    auto kernel_fp16_tile = Var("fp16_tile", fp16_tile_type);
    auto make_i32_tile = Call("block.make_tile", {}, i32_tile_type);
    auto make_fp16_tile = Call("block.make_tile", {}, fp16_tile_type);
    auto launch = RegisteredCall(
        "simt.launch",
        {ConstInt(32), ConstInt(1), ConstInt(1), kernel_i32_tile, kernel_fp16_tile, kernel_u64_tensor,
         kernel_bf16_tensor, kernel_offset, kernel_i32_value, kernel_u64_value, kernel_fp16_value, kernel_bf16_value},
        Kwargs{{"callee", std::string("atomic_ops")}, {"max_threads", 32}});
    auto kernel_body = Seq(
        {Assign(kernel_i32_tile, make_i32_tile), Assign(kernel_fp16_tile, make_fp16_tile), Eval(launch)});
    std::vector<ir::VarPtr> kernel_args = {kernel_u64_tensor, kernel_bf16_tensor, kernel_offset,    kernel_i32_value,
                                           kernel_u64_value,  kernel_fp16_value,  kernel_bf16_value};
    auto kernel = std::make_shared<const ir::Function>("kernel", kernel_args, std::vector<ir::TypePtr>{}, kernel_body,
                                                       Sp(), ir::FunctionType::IN_CORE, true);

    return std::make_shared<const ir::Program>(std::vector<ir::FunctionPtr>{simt_function, kernel}, "simt_atomic_ops",
                                               Sp(), std::make_shared<ir::IRDebugInfo>());
}

} // namespace

TEST(CCESimtCodegenTest, GeneratesContextCalleesMemoryAccessAndLaunch)
{
    CCECodegen codegen(ir::SectionKind::Vector);
    std::string generated = codegen.GenerateSingle(MakeSimtProgram(), npu::tile_fwk::NPUArch::DAV_3510);

    auto add_pos = generated.find("__simt_callee__ inline float add_scalar(");
    auto store_pos = generated.find("__simt_callee__ inline void store_scalar(");
    auto apply_pos = generated.find("__simt_callee__ inline float apply_scalar(");
    auto entry_pos = generated.find("__simt_vf__ __launch_bounds__(256) inline void simt_entry(");
    ASSERT_NE(add_pos, std::string::npos);
    ASSERT_NE(store_pos, std::string::npos);
    ASSERT_NE(apply_pos, std::string::npos);
    ASSERT_NE(entry_pos, std::string::npos);
    EXPECT_LT(add_pos, apply_pos);
    EXPECT_LT(store_pos, apply_pos);
    EXPECT_LT(apply_pos, entry_pos);

    auto store_signature = FindLineContaining(generated, "__simt_callee__ inline void store_scalar(");
    auto apply_signature = FindLineContaining(generated, "__simt_callee__ inline float apply_scalar(");
    auto entry_signature = FindLineContaining(generated, "__simt_vf__ __launch_bounds__(256) inline void simt_entry(");
    ASSERT_FALSE(store_signature.empty());
    ASSERT_FALSE(apply_signature.empty());
    ASSERT_FALSE(entry_signature.empty());
    EXPECT_NE(store_signature.find("__ubuf__ float*"), std::string::npos);
    EXPECT_NE(store_signature.find("__valid_row"), std::string::npos);
    EXPECT_NE(store_signature.find("__valid_col"), std::string::npos);
    EXPECT_NE(apply_signature.find("__ubuf__ float*"), std::string::npos);
    EXPECT_NE(entry_signature.find("__gm__ float*"), std::string::npos);

    EXPECT_NE(generated.find("threadIdx.x"), std::string::npos);
    EXPECT_NE(generated.find("blockDim.y"), std::string::npos);
    EXPECT_NE(generated.find("blockIdx.z"), std::string::npos);
    EXPECT_NE(generated.find("gridDim.x"), std::string::npos);
    EXPECT_NE(generated.find("threadIdx.y * blockDim.x"), std::string::npos);
    EXPECT_NE(generated.find("warpSize"), std::string::npos);

    auto store_function = generated.substr(store_pos, apply_pos - store_pos);
    auto apply_function = generated.substr(apply_pos, entry_pos - apply_pos);
    auto entry_function_end = generated.find("__aicore__ inline void kernel_impl_vector(", entry_pos);
    ASSERT_NE(entry_function_end, std::string::npos);
    auto entry_function = generated.substr(entry_pos, entry_function_end - entry_pos);
    EXPECT_NE(store_function.find("] ="), std::string::npos);
    EXPECT_EQ(store_function.find(".SetValue("), std::string::npos);
    EXPECT_NE(apply_function.find("["), std::string::npos);
    EXPECT_EQ(apply_function.find(".GetValue("), std::string::npos);
    EXPECT_NE(apply_function.find("store_scalar("), std::string::npos);
    EXPECT_NE(entry_function.find("apply_scalar("), std::string::npos);
    EXPECT_NE(entry_function.find("*((__gm__ float*)"), std::string::npos);

    auto launch_line = FindLineContaining(generated, "cce::async_invoke<simt_entry>");
    ASSERT_FALSE(launch_line.empty());
    EXPECT_NE(launch_line.find("cce::dim3{8, 4, 8}"), std::string::npos);
    EXPECT_EQ(CountOccurrences(launch_line, "(__ubuf__ float*)"), 1u);
    EXPECT_EQ(CountOccurrences(launch_line, ".data()"), 1u);
    EXPECT_EQ(CountOccurrences(launch_line, ".GetValidRow()"), 1u);
    EXPECT_EQ(CountOccurrences(launch_line, ".GetValidCol()"), 1u);
    EXPECT_EQ(CountOccurrences(launch_line, "(__gm__ float*)"), 1u);
    EXPECT_EQ(generated.find("pipe_barrier(PIPE_ALL);"), std::string::npos);
    EXPECT_NE(generated.find("__aicore__ inline void kernel_impl_vector("), std::string::npos);
}

TEST(CCESimtCodegenTest, RejectsUnsupportedTargetAndArchitecture)
{
    CCECodegen cube_codegen(ir::SectionKind::Cube);
    EXPECT_THROW((void)cube_codegen.GenerateSingle(MakeSimtProgram(), npu::tile_fwk::NPUArch::DAV_3510),
                 npu::tile_fwk::Error);

    CCECodegen vector_codegen(ir::SectionKind::Vector);
    EXPECT_THROW((void)vector_codegen.GenerateSingle(MakeSimtProgram(), npu::tile_fwk::NPUArch::DAV_2201),
                 npu::tile_fwk::Error);
}

TEST(CCESimtCodegenTest, DoesNotInterceptOrdinaryKernelCallsWithMatchingCalleeName)
{
    CCECodegen codegen(ir::SectionKind::Vector);
    std::string generated = codegen.GenerateSingle(MakeSimtCalleeNameCollisionProgram(),
                                                   npu::tile_fwk::NPUArch::DAV_3510);

    auto entry_pos = generated.find("__simt_vf__ __launch_bounds__(32) inline void collision_entry(");
    auto kernel_pos = generated.find("__aicore__ inline void kernel_impl_vector(");
    ASSERT_NE(entry_pos, std::string::npos);
    ASSERT_NE(kernel_pos, std::string::npos);

    auto entry_function = generated.substr(entry_pos, kernel_pos - entry_pos);
    auto kernel_function = generated.substr(kernel_pos);
    EXPECT_EQ(CountOccurrences(entry_function, "get_subblock_idx();"), 1u);
    EXPECT_EQ(CountOccurrences(kernel_function, "(int64_t)(get_subblockid())"), 1u);
}

TEST(CCESimtCodegenTest, GeneratesRegisteredSynchronizationScalarCastAndMathOperations)
{
    CCECodegen codegen(ir::SectionKind::Vector);
    std::string generated = codegen.GenerateSingle(MakeSimtScalarOpsProgram(), npu::tile_fwk::NPUArch::DAV_3510);

    const std::vector<std::string> scalar_intrinsics = {
        "__sync_workitems();",
        "__threadfence_block();",
        "__threadfence();",
        "__fabsf(",
        "__sqrtf(",
        "1.0f / __sqrtf(",
        "__expf(",
        "0.6931471805599453f",
        "__logf(",
        "__logf(2.0f)",
        "__logf(1.0f +",
        "__rintf(",
        "__roundf(",
        "__floorf(",
        "__ceilf(",
        "__isnan(",
        "__isinf(",
        "__isfinite(",
        "__popc((unsigned int)(",
        "__popc((unsigned long long)(",
        "__mulhi(",
        "__umulhi(",
        "__mul64hi(",
        "__umul64hi(",
        "__fma(",
        "__hmin_nan(",
        "__hmax_nan(",
        "__min(",
        "__max(",
        "min((int64_t)",
        "max((int64_t)",
        "(half)1.0 /",
        "(bfloat16_t)0",
        "__fabsf(__t)",
        "if (__q & 1) { __s = __c; }",
        "if (__q & 1) { __c = -__s; }",
        "uint32_t __sign = __ux & 0x80000000u;",
        "if (__ay == 0 || __ax >= 0x7f800000u || __ay > 0x7f800000u)",
        "while (__ex > __ey)",
        "reinterpret_cast<float&>(__result);",
    };
    for (const auto& intrinsic : scalar_intrinsics) {
        EXPECT_NE(generated.find(intrinsic), std::string::npos) << intrinsic;
    }

    const std::vector<std::string> cast_intrinsics = {
        "__cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(",
        "__cvt_half<ROUND::A, RoundingSaturation::RS_DISABLE_VALUE>(",
        "__cvt_half<ROUND::F, RoundingSaturation::RS_DISABLE_VALUE>(",
        "__cvt_half<ROUND::C, RoundingSaturation::RS_DISABLE_VALUE>(",
        "__cvt_half<ROUND::Z, RoundingSaturation::RS_DISABLE_VALUE>(",
        "__cvt_half<ROUND::O, RoundingSaturation::RS_DISABLE_VALUE>(",
        "__cvt_bfloat16_t<ROUND::A, RoundingSaturation::RS_DISABLE_VALUE>(",
        "__cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(",
        "__cvt_float<ROUND::Z, RoundingSaturation::RS_DISABLE_VALUE>(",
        "__cvt_int32_t<ROUND::R, RoundingSaturation::RS_ENABLE_VALUE>(",
        "__cvt_uint32_t<ROUND::A, RoundingSaturation::RS_ENABLE_VALUE>(",
        "__cvt_int64_t<ROUND::F, RoundingSaturation::RS_ENABLE_VALUE>(",
        "__cvt_uint64_t<ROUND::C, RoundingSaturation::RS_ENABLE_VALUE>(",
        "((int32_t)",
    };
    for (const auto& intrinsic : cast_intrinsics) {
        EXPECT_NE(generated.find(intrinsic), std::string::npos) << intrinsic;
    }
    EXPECT_EQ(generated.find("simt_api/"), std::string::npos);
}

TEST(CCESimtCodegenTest, GeneratesRegisteredAtomicOperationsForTileAndTensor)
{
    CCECodegen codegen(ir::SectionKind::Vector);
    std::string generated = codegen.GenerateSingle(MakeSimtAtomicOpsProgram(), npu::tile_fwk::NPUArch::DAV_3510);

    for (const auto& intrinsic : {"atomicAdd(", "atomicSub(", "atomicExch(", "atomicMax(", "atomicMin(", "atomicInc(",
                                  "atomicDec(", "atomicCAS(", "atomicAnd(", "atomicOr(", "atomicXOr("}) {
        EXPECT_NE(generated.find(intrinsic), std::string::npos) << intrinsic;
    }
    auto tile_line = FindLineContaining(generated, "atomicCAS(");
    auto tensor_line = FindLineContaining(generated, "atomicInc(");
    ASSERT_FALSE(tile_line.empty());
    ASSERT_FALSE(tensor_line.empty());
    EXPECT_NE(tile_line.find(" + ("), std::string::npos) << tile_line;
    EXPECT_EQ(CountOccurrences(tile_line, ", "), 2u) << tile_line;
    EXPECT_NE(tensor_line.find(" + ("), std::string::npos) << tensor_line;
    EXPECT_EQ(CountOccurrences(tensor_line, ", "), 1u) << tensor_line;
    EXPECT_EQ(CountOccurrences(generated, "atomicAdd("), 3u);
    EXPECT_EQ(CountOccurrences(generated, "atomicMax("), 3u);
    EXPECT_EQ(CountOccurrences(generated, "atomicMin("), 3u);
}

TEST(CCESimtCodegenTest, ValidatesRegisteredSynchronizationScalarAndAtomicContracts)
{
    auto fp16 = Scalar(ir::DataType::FP16);
    auto bf16 = Scalar(ir::DataType::BF16);
    auto fp32 = Scalar(ir::DataType::FP32);
    auto i32 = Scalar(ir::DataType::INT32);
    auto u32 = Scalar(ir::DataType::UINT32);
    auto fp16_value = Var("fp16_value", fp16);
    auto bf16_value = Var("bf16_value", bf16);
    auto fp32_value = Var("fp32_value", fp32);
    auto i32_value = Var("i32_value", i32);
    auto u32_value = Var("u32_value", u32);
    auto u64_value = Var("u64_value", Scalar(ir::DataType::UINT64));
    auto offset = Var("offset", u32);
    auto bool_offset = Var("bool_offset", Scalar(ir::DataType::BOOL));
    auto fp16_tile = Var("fp16_tile", MakeSimtTileType(ir::DataType::FP16));
    auto fp32_tile = Var("fp32_tile", MakeSimtTileType(ir::DataType::FP32));
    auto i32_tile = Var("i32_tile", MakeSimtTileType(ir::DataType::INT32));

    auto half_atomic = RegisteredCall("simt.atomic_add", {fp16_tile, offset, fp16_value});
    EXPECT_EQ(half_atomic->GetType(), ir::GetNoneType());
    auto isnan = RegisteredCall("simt.isnan", {fp32_value});
    auto isnan_type = ir::As<ir::ScalarType>(isnan->GetType());
    ASSERT_NE(isnan_type, nullptr);
    EXPECT_EQ(isnan_type->dtype_, ir::DataType::BOOL);

    for (const auto& op_name : {"simt.syncthreads", "simt.threadfence_block", "simt.threadfence"}) {
        EXPECT_EQ(RegisteredCall(op_name, {})->GetType(), ir::GetNoneType());
    }
    auto isfinite_fp16_type = ir::As<ir::ScalarType>(RegisteredCall("simt.isfinite", {fp16_value})->GetType());
    auto isfinite_fp32_type = ir::As<ir::ScalarType>(RegisteredCall("simt.isfinite", {fp32_value})->GetType());
    auto popcount_type = ir::As<ir::ScalarType>(RegisteredCall("simt.popcount", {u64_value})->GetType());
    auto mul_hi_type = ir::As<ir::ScalarType>(RegisteredCall("simt.mul_hi", {u32_value, u32_value})->GetType());
    auto fmod_type = ir::As<ir::ScalarType>(RegisteredCall("simt.fmod", {fp32_value, fp32_value})->GetType());
    ASSERT_NE(isfinite_fp16_type, nullptr);
    ASSERT_NE(isfinite_fp32_type, nullptr);
    ASSERT_NE(popcount_type, nullptr);
    ASSERT_NE(mul_hi_type, nullptr);
    ASSERT_NE(fmod_type, nullptr);
    EXPECT_EQ(isfinite_fp16_type->dtype_, ir::DataType::BOOL);
    EXPECT_EQ(isfinite_fp32_type->dtype_, ir::DataType::BOOL);
    EXPECT_EQ(popcount_type->dtype_, ir::DataType::INT32);
    EXPECT_EQ(mul_hi_type->dtype_, ir::DataType::UINT32);
    EXPECT_EQ(fmod_type->dtype_, ir::DataType::FP32);

    EXPECT_THROW((void)RegisteredCall("simt.sqrt", {i32_value}), npu::tile_fwk::Error);
    EXPECT_THROW((void)RegisteredCall("simt.min", {fp32_value, i32_value}), npu::tile_fwk::Error);
    EXPECT_THROW((void)RegisteredCall("simt.isfinite", {bf16_value}), npu::tile_fwk::Error);
    EXPECT_THROW((void)RegisteredCall("simt.popcount", {i32_value}), npu::tile_fwk::Error);
    EXPECT_THROW((void)RegisteredCall("simt.mul_hi", {i32_value, u32_value}), npu::tile_fwk::Error);
    EXPECT_THROW((void)RegisteredCall("simt.fmod", {i32_value, i32_value}), npu::tile_fwk::Error);
    EXPECT_THROW((void)RegisteredCall("simt.atomic_or", {fp32_tile, offset, fp32_value}), npu::tile_fwk::Error);
    EXPECT_THROW((void)RegisteredCall("simt.atomic_inc", {i32_tile, offset, i32_value}), npu::tile_fwk::Error);
    EXPECT_THROW((void)RegisteredCall("simt.atomic_add", {i32_tile, bool_offset, i32_value}), npu::tile_fwk::Error);
}

} // namespace codegen
} // namespace pypto
