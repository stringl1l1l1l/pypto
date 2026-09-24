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
 * @file block_ops/out_matmul.cpp
 * \brief Block matrix multiplication operations with explicit output tiles.
 *
 * All operations receive a pre-allocated output tile as the first argument.
 *   block.matmul       (out, lhs, rhs)
 *   block.matmul_acc   (out, acc, lhs, rhs)
 *   block.matmul_bias  (out, lhs, rhs, bias)
 *   block.matmul_mx       (out, lhs, rhs, scale_a, scale_b)
 *   block.matmul_mx_acc   (out, acc, lhs, rhs, scale_a, scale_b)
 *   block.gemv         (out, lhs, rhs)
 *   block.gemv_acc     (out, acc, lhs, rhs)
 *   block.gemv_bias    (out, lhs, rhs, bias)
 */

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/logging.h"
#include "ir/kind_traits.h"
#include "ir/memory_space.h"
#include "ir/op_registry.h"
#include "ir/type.h"
#include "ir/type_inference.h"
#include "ir/op/op_common.h"

namespace pypto {
namespace ir {

// The cube datapath pins every matmul tile to a dedicated buffer (TMATMUL /
// TGEMV tile-position constraints + TMatmul.hpp static_asserts): lhs=Left
// (L0A), rhs=Right (L0B), out/acc=Acc (L0C), bias=Bias, scales=ScaleLeft /
// ScaleRight. Each tile argument below is validated one by one via
// CheckTileArg with its own space list.

// ---------------------------------------------------------------------------
// Op registration
// ---------------------------------------------------------------------------

// block.matmul: (out, lhs, rhs) -> out's type
REGISTER_OP("block.matmul")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output matrix multiplication: out = lhs @ rhs")
    .add_argument("out", "Pre-allocated output tile [M,N] (TileType)")
    .add_argument("lhs", "Left matrix tile [M,K] (TileType)")
    .add_argument("rhs", "Right matrix tile [K,N] (TileType)")
    .set_attr<int>("phase")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x3)
            << "The operator block.matmul requires 3 arguments (out, lhs, rhs)";
        CheckTileArg(args, 0, "block.matmul", {MemorySpace::Acc});
        CheckTileArg(args, 1, "block.matmul", {MemorySpace::Left});
        CheckTileArg(args, 2, "block.matmul", {MemorySpace::Right});
        return DeduceBlockOutTileType(args, kwargs, "block.matmul", 3);
    });

// block.matmul_acc: (out, acc, lhs, rhs) -> out's type
REGISTER_OP("block.matmul_acc")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output matmul with accumulation: out = acc + lhs @ rhs")
    .add_argument("out", "Pre-allocated output tile [M,N] (TileType)")
    .add_argument("acc", "Accumulator tile [M,N] (TileType)")
    .add_argument("lhs", "Left matrix tile [M,K] (TileType)")
    .add_argument("rhs", "Right matrix tile [K,N] (TileType)")
    .set_attr<int>("phase")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x4)
            << "The operator block.matmul_acc requires 4 arguments (out, acc, lhs, rhs)";
        CheckTileArg(args, 0, "block.matmul_acc", {MemorySpace::Acc});
        CheckTileArg(args, 1, "block.matmul_acc", {MemorySpace::Acc});
        CheckTileArg(args, 2, "block.matmul_acc", {MemorySpace::Left});
        CheckTileArg(args, 3, "block.matmul_acc", {MemorySpace::Right});
        return DeduceBlockOutTileType(args, kwargs, "block.matmul_acc", 4);
    });

// block.matmul_bias: (out, lhs, rhs, bias) -> out's type
REGISTER_OP("block.matmul_bias")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output matmul with bias: out = lhs @ rhs + bias")
    .add_argument("out", "Pre-allocated output tile [M,N] (TileType)")
    .add_argument("lhs", "Left matrix tile [M,K] (TileType)")
    .add_argument("rhs", "Right matrix tile [K,N] (TileType)")
    .add_argument("bias", "Bias tile [1,N] (TileType)")
    .set_attr<int>("phase")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x4)
            << "The operator block.matmul_bias requires 4 arguments (out, lhs, rhs, bias)";
        CheckTileArg(args, 0, "block.matmul_bias", {MemorySpace::Acc});
        CheckTileArg(args, 1, "block.matmul_bias", {MemorySpace::Left});
        CheckTileArg(args, 2, "block.matmul_bias", {MemorySpace::Right});
        CheckTileArg(args, 3, "block.matmul_bias", {MemorySpace::Bias});
        return DeduceBlockOutTileType(args, kwargs, "block.matmul_bias", 4);
    });

// block.matmul_mx: (out, lhs, rhs, scale_a, scale_b) -> out's type
REGISTER_OP("block.matmul_mx")
    .set_op_category("BlockOp")
    .set_description("Block MX matmul: out = lhs @ rhs (with per-group E8M0 scale)")
    .add_argument("out", "Pre-allocated output tile [M,N] in Acc (TileType)")
    .add_argument("lhs", "Left matrix tile [M,K] (TileType, FP8/FP4)")
    .add_argument("rhs", "Right matrix tile [K,N] (TileType, FP8/FP4)")
    .add_argument("scale_a", "Left scale tile in ScaleLeft (TileType, E8M0)")
    .add_argument("scale_b", "Right scale tile in ScaleRight (TileType, E8M0)")
    .set_attr<int>("phase")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x5)
            << "The operator block.matmul_mx requires 5 arguments (out, lhs, rhs, scale_a, scale_b)";
        CheckTileArg(args, 0, "block.matmul_mx", {MemorySpace::Acc});
        CheckTileArg(args, 1, "block.matmul_mx", {MemorySpace::Left});
        CheckTileArg(args, 2, "block.matmul_mx", {MemorySpace::Right});
        CheckTileArg(args, 3, "block.matmul_mx", {MemorySpace::ScaleLeft});
        CheckTileArg(args, 4, "block.matmul_mx", {MemorySpace::ScaleRight});
        return DeduceBlockOutTileType(args, kwargs, "block.matmul_mx", 5);
    });

// block.matmul_mx_acc: (out, acc, lhs, rhs, scale_a, scale_b) -> out's type
REGISTER_OP("block.matmul_mx_acc")
    .set_op_category("BlockOp")
    .set_description("Block MX matmul with accumulation: out = acc + lhs @ rhs")
    .add_argument("out", "Pre-allocated output tile [M,N] in Acc (TileType)")
    .add_argument("acc", "Accumulator tile [M,N] (TileType)")
    .add_argument("lhs", "Left matrix tile [M,K] (TileType, FP8/FP4)")
    .add_argument("rhs", "Right matrix tile [K,N] (TileType, FP8/FP4)")
    .add_argument("scale_a", "Left scale tile in ScaleLeft (TileType, E8M0)")
    .add_argument("scale_b", "Right scale tile in ScaleRight (TileType, E8M0)")
    .set_attr<int>("phase")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x6)
            << "The operator block.matmul_mx_acc requires 6 arguments (out, acc, lhs, rhs, scale_a, scale_b)";
        CheckTileArg(args, 0, "block.matmul_mx_acc", {MemorySpace::Acc});
        CheckTileArg(args, 1, "block.matmul_mx_acc", {MemorySpace::Acc});
        CheckTileArg(args, 2, "block.matmul_mx_acc", {MemorySpace::Left});
        CheckTileArg(args, 3, "block.matmul_mx_acc", {MemorySpace::Right});
        CheckTileArg(args, 4, "block.matmul_mx_acc", {MemorySpace::ScaleLeft});
        CheckTileArg(args, 5, "block.matmul_mx_acc", {MemorySpace::ScaleRight});
        return DeduceBlockOutTileType(args, kwargs, "block.matmul_mx_acc", 6);
    });

// block.gemv: (out, lhs, rhs) -> out's type
REGISTER_OP("block.gemv")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output GEMV: out[1,N] = lhs[1,K] @ rhs[K,N]")
    .add_argument("out", "Pre-allocated output tile [1,N] (TileType)")
    .add_argument("lhs", "Row vector tile [1,K] (TileType)")
    .add_argument("rhs", "Matrix tile [K,N] (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x3)
            << "The operator block.gemv requires 3 arguments (out, lhs, rhs)";
        CheckTileArg(args, 0, "block.gemv", {MemorySpace::Acc});
        CheckTileArg(args, 1, "block.gemv", {MemorySpace::Left});
        CheckTileArg(args, 2, "block.gemv", {MemorySpace::Right});
        return DeduceBlockOutTileType(args, kwargs, "block.gemv", 3);
    });

// block.gemv_acc: (out, acc, lhs, rhs) -> out's type
REGISTER_OP("block.gemv_acc")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output GEMV with accumulation: out += lhs @ rhs")
    .add_argument("out", "Pre-allocated output tile [1,N] (TileType)")
    .add_argument("acc", "Accumulator tile [1,N] (TileType)")
    .add_argument("lhs", "Row vector tile [1,K] (TileType)")
    .add_argument("rhs", "Matrix tile [K,N] (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x4)
            << "The operator block.gemv_acc requires 4 arguments (out, acc, lhs, rhs)";
        CheckTileArg(args, 0, "block.gemv_acc", {MemorySpace::Acc});
        CheckTileArg(args, 1, "block.gemv_acc", {MemorySpace::Acc});
        CheckTileArg(args, 2, "block.gemv_acc", {MemorySpace::Left});
        CheckTileArg(args, 3, "block.gemv_acc", {MemorySpace::Right});
        return DeduceBlockOutTileType(args, kwargs, "block.gemv_acc", 4);
    });

// block.gemv_bias: (out, lhs, rhs, bias) -> out's type
REGISTER_OP("block.gemv_bias")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output GEMV with bias: out = lhs @ rhs + bias")
    .add_argument("out", "Pre-allocated output tile [1,N] (TileType)")
    .add_argument("lhs", "Row vector tile [1,K] (TileType)")
    .add_argument("rhs", "Matrix tile [K,N] (TileType)")
    .add_argument("bias", "Bias tile [1,N] (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x4)
            << "The operator block.gemv_bias requires 4 arguments (out, lhs, rhs, bias)";
        CheckTileArg(args, 0, "block.gemv_bias", {MemorySpace::Acc});
        CheckTileArg(args, 1, "block.gemv_bias", {MemorySpace::Left});
        CheckTileArg(args, 2, "block.gemv_bias", {MemorySpace::Right});
        CheckTileArg(args, 3, "block.gemv_bias", {MemorySpace::Bias});
        return DeduceBlockOutTileType(args, kwargs, "block.gemv_bias", 4);
    });

} // namespace ir
} // namespace pypto
