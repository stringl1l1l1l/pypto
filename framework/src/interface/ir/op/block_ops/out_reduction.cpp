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
 * @file block_ops/out_reduction.cpp
 * \brief Block reduction and broadcast operations with explicit output tiles.
 *
 * Reduction ops: row_sum, row_max, row_min  (out, tile, tmp)
 * Broadcast ops: row_expand, col_expand, row_expand_*, col_expand_*
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

// The reduction families execute on the vector pipe over UB tiles: the ISA
// pins dst/src/tmp to TileType::Vec (the TROWMAX/TROWMIN/TCOLMAX/TCOLSUM
// implementation checks state that dst and src must both be TileType::Vec),
// so every tile argument below is validated one by one with CheckTileArg and
// a Vec space list.

// ---------------------------------------------------------------------------
// Reduction operations: (out, tile, tmp) -> out's type
// ---------------------------------------------------------------------------

REGISTER_OP("block.row_sum")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output row-wise sum reduction: out[i,0] = sum_j(tile[i,j])")
    .add_argument("out", "Pre-allocated output row vector tile (TileType)")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("tmp", "Scratch tile required by hardware (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x3)
            << "The operator block.row_sum requires 3 arguments (out, tile, tmp)";
        CheckTileArg(args, 0, "block.row_sum", {MemorySpace::Vec});
        CheckTileArg(args, 1, "block.row_sum", {MemorySpace::Vec});
        CheckTileArg(args, 2, "block.row_sum", {MemorySpace::Vec});
        return DeduceBlockOutTileType(args, kwargs, "block.row_sum", 3);
    });

REGISTER_OP("block.row_max")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output row-wise max reduction: out[i,0] = max_j(tile[i,j])")
    .add_argument("out", "Pre-allocated output row vector tile (TileType)")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("tmp", "Scratch tile required by hardware (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x3)
            << "The operator block.row_max requires 3 arguments (out, tile, tmp)";
        CheckTileArg(args, 0, "block.row_max", {MemorySpace::Vec});
        CheckTileArg(args, 1, "block.row_max", {MemorySpace::Vec});
        CheckTileArg(args, 2, "block.row_max", {MemorySpace::Vec});
        return DeduceBlockOutTileType(args, kwargs, "block.row_max", 3);
    });

REGISTER_OP("block.row_min")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output row-wise min reduction: out[i,0] = min_j(tile[i,j])")
    .add_argument("out", "Pre-allocated output row vector tile (TileType)")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("tmp", "Scratch tile required by hardware (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x3)
            << "The operator block.row_min requires 3 arguments (out, tile, tmp)";
        CheckTileArg(args, 0, "block.row_min", {MemorySpace::Vec});
        CheckTileArg(args, 1, "block.row_min", {MemorySpace::Vec});
        CheckTileArg(args, 2, "block.row_min", {MemorySpace::Vec});
        return DeduceBlockOutTileType(args, kwargs, "block.row_min", 3);
    });

REGISTER_OP("block.col_max")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output col-wise max reduction: out[0,j] = max_i(tile[i,j])")
    .add_argument("out", "Pre-allocated output col vector tile [1,N] (TileType)")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("tmp", "Scratch tile required by hardware (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x3)
            << "The operator block.col_max requires 3 arguments (out, tile, tmp)";
        CheckTileArg(args, 0, "block.col_max", {MemorySpace::Vec});
        CheckTileArg(args, 1, "block.col_max", {MemorySpace::Vec});
        CheckTileArg(args, 2, "block.col_max", {MemorySpace::Vec});
        return DeduceBlockOutTileType(args, kwargs, "block.col_max", 3);
    });

REGISTER_OP("block.col_sum")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output col-wise sum reduction: out[0,j] = sum_i(tile[i,j])")
    .add_argument("out", "Pre-allocated output col vector tile [1,N] (TileType)")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("tmp", "Scratch tile required by hardware (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x3)
            << "The operator block.col_sum requires 3 arguments (out, tile, tmp)";
        CheckTileArg(args, 0, "block.col_sum", {MemorySpace::Vec});
        CheckTileArg(args, 1, "block.col_sum", {MemorySpace::Vec});
        CheckTileArg(args, 2, "block.col_sum", {MemorySpace::Vec});
        return DeduceBlockOutTileType(args, kwargs, "block.col_sum", 3);
    });

REGISTER_OP("block.col_min")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output col-wise min reduction: out[0,j] = min_i(tile[i,j])")
    .add_argument("out", "Pre-allocated output col vector tile [1,N] (TileType)")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("tmp", "Scratch tile required by hardware (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x3)
            << "The operator block.col_min requires 3 arguments (out, tile, tmp)";
        CheckTileArg(args, 0, "block.col_min", {MemorySpace::Vec});
        CheckTileArg(args, 1, "block.col_min", {MemorySpace::Vec});
        CheckTileArg(args, 2, "block.col_min", {MemorySpace::Vec});
        return DeduceBlockOutTileType(args, kwargs, "block.col_min", 3);
    });

REGISTER_OP("block.row_prod")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output row-wise product reduction: out[i,0] = prod_j(tile[i,j])")
    .add_argument("out", "Pre-allocated output row vector tile (TileType)")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("tmp", "Scratch tile required by hardware (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x3)
            << "The operator block.row_prod requires 3 arguments (out, tile, tmp)";
        CheckTileArg(args, 0, "block.row_prod", {MemorySpace::Vec});
        CheckTileArg(args, 1, "block.row_prod", {MemorySpace::Vec});
        CheckTileArg(args, 2, "block.row_prod", {MemorySpace::Vec});
        return DeduceBlockOutTileType(args, kwargs, "block.row_prod", 3);
    });

REGISTER_OP("block.col_prod")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output col-wise product reduction: out[0,j] = prod_i(tile[i,j])")
    .add_argument("out", "Pre-allocated output col vector tile [1,N] (TileType)")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("tmp", "Scratch tile required by hardware (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x3)
            << "The operator block.col_prod requires 3 arguments (out, tile, tmp)";
        CheckTileArg(args, 0, "block.col_prod", {MemorySpace::Vec});
        CheckTileArg(args, 1, "block.col_prod", {MemorySpace::Vec});
        CheckTileArg(args, 2, "block.col_prod", {MemorySpace::Vec});
        return DeduceBlockOutTileType(args, kwargs, "block.col_prod", 3);
    });

REGISTER_OP("block.row_reduce")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output generic row-wise reduction with op_type param")
    .add_argument("out", "Pre-allocated output row vector tile (TileType)")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("tmp", "Scratch tile required by hardware (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutTileType(args, kwargs, "block.row_reduce", 3);
    });

REGISTER_OP("block.col_reduce")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output generic col-wise reduction with op_type param")
    .add_argument("out", "Pre-allocated output col vector tile [1,N] (TileType)")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("tmp", "Scratch tile required by hardware (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutTileType(args, kwargs, "block.col_reduce", 3);
    });

// ---------------------------------------------------------------------------
// Reduce-with-index operations (argmax / argmin)
// ---------------------------------------------------------------------------

#define REGISTER_BLOCK_REDUCE_IDX(direction, op_suffix, description)                                      \
    REGISTER_OP("block." #direction "_" #op_suffix)                                                       \
        .set_op_category("BlockOp")                                                                       \
        .set_description("Block explicit-output " #direction "-wise " description)                        \
        .add_argument("out", "Pre-allocated output index tile (TileType)")                                \
        .add_argument("tile", "Input tile (TileType)")                                                    \
        .add_argument("tmp", "Scratch tile required by hardware (TileType)")                              \
        .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,                              \
                          [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) { \
            return DeduceBlockOutTileType(args, kwargs, "block." #direction "_" #op_suffix, 3);           \
        })

REGISTER_BLOCK_REDUCE_IDX(row, argmax, "argmax: out[i,0] = argmax_j(tile[i,j])");
REGISTER_BLOCK_REDUCE_IDX(row, argmin, "argmin: out[i,0] = argmin_j(tile[i,j])");
REGISTER_BLOCK_REDUCE_IDX(col, argmax, "argmax: out[0,j] = argmax_i(tile[i,j])");
REGISTER_BLOCK_REDUCE_IDX(col, argmin, "argmin: out[0,j] = argmin_i(tile[i,j])");

#undef REGISTER_BLOCK_REDUCE_IDX

// ---------------------------------------------------------------------------
// Broadcast / expansion operations
// ---------------------------------------------------------------------------

// row_expand (out, src): unary broadcast.
REGISTER_OP("block.row_expand")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output row broadcast: out[i,j] = src[i,0] for all j")
    .add_argument("out", "Pre-allocated output tile [M,N] (TileType)")
    .add_argument("src", "Source tile [M,1] (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutTileType(args, kwargs, "block.row_expand", 2);
    });

// col_expand (out, col_vec): unary broadcast.
REGISTER_OP("block.col_expand")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output column broadcast: out[i,j] = col_vec[0,j] for all i")
    .add_argument("out", "Pre-allocated output tile [M,N] (TileType)")
    .add_argument("col_vec", "Source column vector [1,N] (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutTileType(args, kwargs, "block.col_expand", 2);
    });

// row_expand_* (out, tile, row_vec): binary broadcast-arithmetic.
#define REGISTER_BLOCK_OUT_ROW_EXPAND(op_suffix, description)                                             \
    REGISTER_OP("block.row_expand_" #op_suffix)                                                           \
        .set_op_category("BlockOp")                                                                       \
        .set_description("Block explicit-output row broadcast " description)                              \
        .add_argument("out", "Pre-allocated output tile [M,N] (TileType)")                                \
        .add_argument("tile", "Input tile [M,N] (TileType)")                                              \
        .add_argument("row_vec", "Row vector [M,1] (TileType)")                                           \
        .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,                              \
                          [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) { \
            return DeduceBlockOutTileType(args, kwargs, "block.row_expand_" #op_suffix, 3);               \
        })

REGISTER_BLOCK_OUT_ROW_EXPAND(add, "add: out = tile + broadcast(row_vec)");
REGISTER_BLOCK_OUT_ROW_EXPAND(sub, "sub: out = tile - broadcast(row_vec)");
REGISTER_BLOCK_OUT_ROW_EXPAND(mul, "mul: out = tile * broadcast(row_vec)");
REGISTER_BLOCK_OUT_ROW_EXPAND(div, "div: out = tile / broadcast(row_vec)");
REGISTER_BLOCK_OUT_ROW_EXPAND(max, "max: out = max(tile, broadcast(row_vec))");
REGISTER_BLOCK_OUT_ROW_EXPAND(min, "min: out = min(tile, broadcast(row_vec))");
REGISTER_BLOCK_OUT_ROW_EXPAND(expdif, "expdif: out = exp(tile - broadcast(row_vec))");

#undef REGISTER_BLOCK_OUT_ROW_EXPAND

// row_expand_binop (out, tile, row_vec) + op_type kwarg: generic binary broadcast.
REGISTER_OP("block.row_expand_binop")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output row broadcast with parameterized binary op")
    .add_argument("out", "Pre-allocated output tile [M,N] (TileType)")
    .add_argument("tile", "Input tile [M,N] (TileType)")
    .add_argument("row_vec", "Row vector [M,1] (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutTileType(args, kwargs, "block.row_expand_binop", 3);
    });

// col_expand_* (out, tile, col_vec): binary broadcast-arithmetic.
#define REGISTER_BLOCK_OUT_COL_EXPAND(op_suffix, description)                                             \
    REGISTER_OP("block.col_expand_" #op_suffix)                                                           \
        .set_op_category("BlockOp")                                                                       \
        .set_description("Block explicit-output column broadcast " description)                           \
        .add_argument("out", "Pre-allocated output tile [M,N] (TileType)")                                \
        .add_argument("tile", "Input tile [M,N] (TileType)")                                              \
        .add_argument("col_vec", "Column vector [1,N] (TileType)")                                        \
        .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,                              \
                          [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) { \
            return DeduceBlockOutTileType(args, kwargs, "block.col_expand_" #op_suffix, 3);               \
        })

REGISTER_BLOCK_OUT_COL_EXPAND(add, "add: out = tile + broadcast(col_vec)");
REGISTER_BLOCK_OUT_COL_EXPAND(mul, "mul: out = tile * broadcast(col_vec)");
REGISTER_BLOCK_OUT_COL_EXPAND(div, "div: out = tile / broadcast(col_vec)");
REGISTER_BLOCK_OUT_COL_EXPAND(sub, "sub: out = tile - broadcast(col_vec)");
REGISTER_BLOCK_OUT_COL_EXPAND(max, "max: out = max(tile, broadcast(col_vec))");
REGISTER_BLOCK_OUT_COL_EXPAND(min, "min: out = min(tile, broadcast(col_vec))");
REGISTER_BLOCK_OUT_COL_EXPAND(expdif, "expdif: out = exp(tile - broadcast(col_vec))");

#undef REGISTER_BLOCK_OUT_COL_EXPAND

// col_expand_binop (out, tile, col_vec) + op_type kwarg: generic binary broadcast.
REGISTER_OP("block.col_expand_binop")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output column broadcast with parameterized binary op")
    .add_argument("out", "Pre-allocated output tile [M,N] (TileType)")
    .add_argument("tile", "Input tile [M,N] (TileType)")
    .add_argument("col_vec", "Column vector [1,N] (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutTileType(args, kwargs, "block.col_expand_binop", 3);
    });

#undef REGISTER_BLOCK_OUT_COL_EXPAND

} // namespace ir
} // namespace pypto
