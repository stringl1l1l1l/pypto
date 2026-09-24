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
 * @file block_ops/out_elementwise.cpp
 * \brief Block element-wise operations with explicit output tiles.
 *
 * All ops accept a pre-allocated output tile as the last argument and return
 * that tile's type.  This file covers:
 *   - Tile x Tile binary: add, sub, mul, div, rem, maximum, minimum, and, or, shl, shr
 *   - Tile x Scalar binary: adds, subs, muls, divs, rems, ands, ors, shls, shrs, maxs, mins, lrelu
 *   - Unary: neg, exp, sqrt, rsqrt, recip, log, abs, relu, not, cast
 *   - Ternary: xor/xors (with tmp), prelu (with tmp)
 *   - Quaternary: sel (mask,lhs,rhs,tmp), sels (mask,src,tmp,scalar)
 *   - Comparison: cmp, cmps
 *   - Scalar-to-tile: expands
 *   - Layout: reshape, transpose
 */

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/logging.h"
#include "pypto_pro/error.h"
#include "ir/kind_traits.h"
#include "ir/memory_space.h"
#include "ir/op_registry.h"
#include "ir/type.h"
#include "ir/type_inference.h"
#include "ir/op/op_common.h"

namespace pypto {
namespace ir {
using npu::tile_fwk::ExternalError;

// ---------------------------------------------------------------------------
// Shared type deduction helpers
// ---------------------------------------------------------------------------

// Validate that args[idx] is ScalarType.
static void CheckScalarArg([[maybe_unused]] const std::vector<ExprPtr>& args, size_t idx, const std::string& op_name)
{
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, As<ScalarType>(args[idx]->GetType()))
        << "The operator " << op_name << " requires argument " << idx << " to be ScalarType, but got "
        << args[idx]->GetType()->TypeName();
}

// The elementwise families execute on the vector pipe over UB tiles: the ISA
// pins their tiles to TileType::Vec (implementation checks of TADDS/TABS/
// TRELU/TMAX/TCMP/TAXPY/TTRANS; no elementwise compute op accepts Mat), so
// every tile argument below is validated one by one with CheckTileArg and a
// Vec space list. TEXPANDS is the documented exception and passes a Vec-or-Mat
// space list instead.
static TypePtr DeduceBlockOutBinaryTile([[maybe_unused]] const std::vector<ExprPtr>& args,
                                        [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs,
                                        const std::string& op_name)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x3)
        << op_name << " requires 3 arguments (out, lhs, rhs)";
    CheckTileArg(args, 0, op_name, {MemorySpace::Vec});
    CheckTileArg(args, 1, op_name, {MemorySpace::Vec}); // NOLINT: lhs index
    CheckTileArg(args, 2, op_name, {MemorySpace::Vec}); // NOLINT: rhs index
    return DeduceBlockOutTileType(args, kwargs, op_name, 0x3);
}

// Type deduction for (out:TileType, TileType, ScalarType) -> out.
static TypePtr DeduceBlockOutBinaryScalar([[maybe_unused]] const std::vector<ExprPtr>& args,
                                          [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs,
                                          const std::string& op_name)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x3)
        << op_name << " requires 3 arguments (out, tile, scalar)";
    CheckTileArg(args, 0, op_name, {MemorySpace::Vec});
    CheckTileArg(args, 1, op_name, {MemorySpace::Vec}); // NOLINT: lhs index
    CheckScalarArg(args, 2, op_name);                   // NOLINT: rhs index
    return DeduceBlockOutTileType(args, kwargs, op_name, 0x3);
}

// Type deduction for (out:TileType, TileType) -> out  (unary).
static TypePtr DeduceBlockOutUnary([[maybe_unused]] const std::vector<ExprPtr>& args,
                                   [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs,
                                   const std::string& op_name)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x2) << op_name << " requires 2 arguments (out, src)";
    CheckTileArg(args, 0, op_name, {MemorySpace::Vec});
    CheckTileArg(args, 1, op_name, {MemorySpace::Vec});
    return DeduceBlockOutTileType(args, kwargs, op_name, 0x2);
}

// ---------------------------------------------------------------------------
// Tile x Tile binary operations
// ---------------------------------------------------------------------------

#define REGISTER_BLOCK_OUT_BINARY_TILE(name)                                                              \
    REGISTER_OP("block." #name)                                                                           \
        .set_op_category("BlockOp")                                                                       \
        .set_description("Block explicit-output element-wise " #name ": out = lhs " #name " rhs")         \
        .add_argument("out", "Pre-allocated output tile (TileType)")                                      \
        .add_argument("lhs", "Left tile (TileType)")                                                      \
        .add_argument("rhs", "Right tile (TileType)")                                                     \
        .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,                              \
                          [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) { \
            return DeduceBlockOutBinaryTile(args, kwargs, "block." #name);                                \
        })

REGISTER_BLOCK_OUT_BINARY_TILE(add);
REGISTER_BLOCK_OUT_BINARY_TILE(sub);
REGISTER_BLOCK_OUT_BINARY_TILE(mul);
REGISTER_BLOCK_OUT_BINARY_TILE(div);
REGISTER_BLOCK_OUT_BINARY_TILE(rem);
REGISTER_BLOCK_OUT_BINARY_TILE(maximum);
REGISTER_BLOCK_OUT_BINARY_TILE(minimum);

// Bitwise tile-tile ops (integer only; validated at the Python layer).
REGISTER_BLOCK_OUT_BINARY_TILE(and);
REGISTER_BLOCK_OUT_BINARY_TILE(or);
REGISTER_BLOCK_OUT_BINARY_TILE(shl);
REGISTER_BLOCK_OUT_BINARY_TILE(shr);
REGISTER_BLOCK_OUT_BINARY_TILE(add_relu);
REGISTER_BLOCK_OUT_BINARY_TILE(sub_relu);
REGISTER_BLOCK_OUT_BINARY_TILE(mul_add_dst);
REGISTER_BLOCK_OUT_BINARY_TILE(fused_mul_add);
REGISTER_BLOCK_OUT_BINARY_TILE(fused_mul_add_relu);
REGISTER_BLOCK_OUT_BINARY_TILE(partadd);
REGISTER_BLOCK_OUT_BINARY_TILE(partmax);
REGISTER_BLOCK_OUT_BINARY_TILE(partmin);
REGISTER_BLOCK_OUT_BINARY_TILE(partmul);

#undef REGISTER_BLOCK_OUT_BINARY_TILE

// block.add_relu_cast: (out, lhs_tile, rhs_tile) -> out's type; carries target_type and mode attrs.
REGISTER_OP("block.add_relu_cast")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output add-relu-cast: out = cast(max(0, lhs + rhs), target_dtype, rounding_mode)")
    .add_argument("out", "Pre-allocated output tile with target dtype (TileType)")
    .add_argument("lhs", "Left tile (TileType)")
    .add_argument("rhs", "Right tile (TileType)")
    .set_attr<DataType>("target_type")
    .set_attr<int>("mode")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutBinaryTile(args, kwargs, "block.add_relu_cast");
    });

// block.sub_relu_cast: (out, lhs_tile, rhs_tile) -> out's type; carries target_type and mode attrs.
REGISTER_OP("block.sub_relu_cast")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output sub-relu-cast: out = cast(max(0, lhs - rhs), target_dtype, rounding_mode)")
    .add_argument("out", "Pre-allocated output tile with target dtype (TileType)")
    .add_argument("lhs", "Left tile (TileType)")
    .add_argument("rhs", "Right tile (TileType)")
    .set_attr<DataType>("target_type")
    .set_attr<int>("mode")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutBinaryTile(args, kwargs, "block.sub_relu_cast");
    });

// block.mul_cast: (out, lhs_tile, rhs_tile) -> out's type; carries target_type and mode attrs.
REGISTER_OP("block.mul_cast")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output mul-cast: out = cast(lhs * rhs, target_dtype_dtype, rounding_mode)")
    .add_argument("out", "Pre-allocated output tile with target dtype (TileType)")
    .add_argument("lhs", "Left tile (TileType)")
    .add_argument("rhs", "Right tile (TileType)")
    .set_attr<DataType>("target_type")
    .set_attr<int>("mode")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutBinaryTile(args, kwargs, "block.mul_cast");
    });

// ---------------------------------------------------------------------------
// Tile x Scalar binary operations
// ---------------------------------------------------------------------------

#define REGISTER_BLOCK_OUT_BINARY_SCALAR(name)                                                            \
    REGISTER_OP("block." #name)                                                                           \
        .set_op_category("BlockOp")                                                                       \
        .set_description("Block explicit-output tile-scalar " #name ": out = tile " #name " scalar")      \
        .add_argument("out", "Pre-allocated output tile (TileType)")                                      \
        .add_argument("tile", "Input tile (TileType)")                                                    \
        .add_argument("scalar", "Scalar operand (ScalarType)")                                            \
        .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,                              \
                          [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) { \
            return DeduceBlockOutBinaryScalar(args, kwargs, "block." #name);                              \
        })

REGISTER_BLOCK_OUT_BINARY_SCALAR(adds);
REGISTER_BLOCK_OUT_BINARY_SCALAR(subs);
REGISTER_BLOCK_OUT_BINARY_SCALAR(muls);
REGISTER_BLOCK_OUT_BINARY_SCALAR(divs);
REGISTER_BLOCK_OUT_BINARY_SCALAR(rems);
REGISTER_BLOCK_OUT_BINARY_SCALAR(ands);
REGISTER_BLOCK_OUT_BINARY_SCALAR(ors);
REGISTER_BLOCK_OUT_BINARY_SCALAR(shls);
REGISTER_BLOCK_OUT_BINARY_SCALAR(shrs);
REGISTER_BLOCK_OUT_BINARY_SCALAR(maxs);
REGISTER_BLOCK_OUT_BINARY_SCALAR(mins);
REGISTER_BLOCK_OUT_BINARY_SCALAR(axpy);

#undef REGISTER_BLOCK_OUT_BINARY_SCALAR

// block.gather: Two forms:
//   Index form: (out, src_tile, indices_tile) or (out, src_tile, indices_tile, tmp) -> out's type
//   Compare form: (out, src_tile, k_value_tile, cdst_tile, tmp_tile) + kwargs(cmp_mode, offset) -> out's type
REGISTER_OP("block.gather")
    .set_op_category("BlockOp")
    .set_description(
        "Block explicit-output gather: index form (out[i] = src[indices[i]]) or compare form (indices where src > kth "
        "or src == kth)")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .add_argument("src", "Source tile (TileType)")
    .add_argument("indices_or_k_value", "Index tile (index form) or k-value tile (compare form) (TileType)")
    .add_argument("tmp_or_cdst", "Optional tmp tile (index form) or cdst tile (compare form) (TileType)")
    .add_argument("tmp", "Optional tmp tile for compare form (TileType)")
    .set_attr<int>("cmp_mode")
    .set_attr<int>("offset")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        // Index form: 3 or 4 args (out, src, indices) or (out, src, indices, tmp)
        if (args.size() == 3 || args.size() == 4) {
            bool has_cmp_mode = false;
            for (const auto& [key, val] : kwargs) {
                (void)val;
                if (key == "cmp_mode")
                    has_cmp_mode = true;
            }
            if (!has_cmp_mode) {
                return DeduceBlockOutTileType(args, kwargs, "block.gather", args.size());
            }
        }
        // Compare form: 5 args (out, src, k_value, cdst, tmp) + cmp_mode + offset
        if (args.size() == 5) {
            return DeduceBlockOutTileType(args, kwargs, "block.gather", 5);
        }
        PRO_IR_THROW(std::runtime_error, ExternalError::INVALID_ARGUMENT)
            << "block.gather: expected index form (3-4 args) or compare form (5 args + cmp_mode)";
    });

// block.gatherb: (out, src_tile, offsets_tile) -> out's type
REGISTER_OP("block.gatherb")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output gatherb: out[i] = src[byte_offsets[i]]")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .add_argument("src", "Source tile (TileType)")
    .add_argument("offsets", "Byte offset tile (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutTileType(args, kwargs, "block.gatherb", 3);
    });

// block.gathermask: (out, src_tile) + kwargs(pattern_mode) -> out's type
// Gathers elements where the corresponding bit in the pattern is 1.
REGISTER_OP("block.gathermask")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output gathermask: gather elements by built-in mask pattern (1-7)")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .add_argument("src", "Source tile (TileType, b16 or b32)")
    .set_attr<int>("pattern_mode")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutTileType(args, kwargs, "block.gathermask", 2);
    });

// block.scatter: (dst, src_tile, indices_tile) -> dst's type
// Semantics: dst[indices[i,j], j] = src[i, j]
REGISTER_OP("block.scatter")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output scatter: dst[indices[i,j], j] = src[i, j]")
    .add_argument("dst", "Pre-allocated destination tile (TileType)")
    .add_argument("src", "Source tile (TileType)")
    .add_argument("indices", "Index tile (TileType, INT32)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutTileType(args, kwargs, "block.scatter", 3);
    });

// ---------------------------------------------------------------------------
// Unary operations
// ---------------------------------------------------------------------------

#define REGISTER_BLOCK_OUT_UNARY(name)                                                                    \
    REGISTER_OP("block." #name)                                                                           \
        .set_op_category("BlockOp")                                                                       \
        .set_description("Block explicit-output unary " #name ": out = " #name "(src)")                   \
        .add_argument("out", "Pre-allocated output tile (TileType)")                                      \
        .add_argument("src", "Input tile (TileType)")                                                     \
        .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,                              \
                          [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) { \
            return DeduceBlockOutUnary(args, kwargs, "block." #name);                                     \
        })

REGISTER_BLOCK_OUT_UNARY(neg);
REGISTER_BLOCK_OUT_UNARY(exp);
REGISTER_BLOCK_OUT_UNARY(sqrt);
REGISTER_BLOCK_OUT_UNARY(rsqrt);
REGISTER_BLOCK_OUT_UNARY(recip);
REGISTER_BLOCK_OUT_UNARY(log);
REGISTER_BLOCK_OUT_UNARY(abs);
REGISTER_BLOCK_OUT_UNARY(relu);
REGISTER_BLOCK_OUT_UNARY(not );

#undef REGISTER_BLOCK_OUT_UNARY

// block.cast: (out, src_tile) -> out's type; carries target_type and mode attrs.
REGISTER_OP("block.cast")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output type-cast: out = cast(src, target_dtype, rounding_mode)")
    .add_argument("out", "Pre-allocated output tile with target dtype (TileType)")
    .add_argument("src", "Input tile (TileType)")
    .set_attr<DataType>("target_type")
    .set_attr<int>("mode")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutUnary(args, kwargs, "block.cast");
    });

// ---------------------------------------------------------------------------
// Ternary / multi-input operations
// ---------------------------------------------------------------------------

// XOR with tmp buffer (out, tile, tile, tmp): 4 args.
REGISTER_OP("block.xor")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output bitwise XOR: out = lhs ^ rhs (integer tiles; tmp is scratch buffer)")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .add_argument("lhs", "Left tile (TileType)")
    .add_argument("rhs", "Right tile (TileType)")
    .add_argument("tmp", "Scratch tile required by hardware (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x4)
            << "The operator block.xor requires 4 arguments (out, lhs, rhs, tmp)";
        CheckTileArg(args, 0, "block.xor", {MemorySpace::Vec});
        CheckTileArg(args, 1, "block.xor", {MemorySpace::Vec});
        CheckTileArg(args, 2, "block.xor", {MemorySpace::Vec});
        CheckTileArg(args, 3, "block.xor", {MemorySpace::Vec});
        return DeduceBlockOutTileType(args, kwargs, "block.xor", 4);
    });

// XOR-scalar with tmp buffer (out, tile, scalar, tmp): 4 args.
REGISTER_OP("block.xors")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output bitwise XOR with scalar: out = lhs ^ scalar (integer tiles)")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .add_argument("lhs", "Input tile (TileType)")
    .add_argument("scalar", "Scalar operand (ScalarType)")
    .add_argument("tmp", "Scratch tile required by hardware (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x4)
            << "The operator block.xors requires 4 arguments (out, lhs, scalar, tmp)";
        CheckTileArg(args, 0, "block.xors", {MemorySpace::Vec});
        CheckTileArg(args, 1, "block.xors", {MemorySpace::Vec});
        CheckTileArg(args, 3, "block.xors", {MemorySpace::Vec});
        return DeduceBlockOutTileType(args, kwargs, "block.xors", 4);
    });

// Selection (out, mask, lhs, rhs, tmp): 5 args.
REGISTER_OP("block.sel")
    .set_op_category("BlockOp")
    .set_description(
        "Block explicit-output per-element selection: out[i]=lhs[i] if mask[i] else rhs[i]; tmp is scratch buffer")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .add_argument("mask", "Predicate mask tile (TileType)")
    .add_argument("lhs", "True-branch tile (TileType)")
    .add_argument("rhs", "False-branch tile (TileType)")
    .add_argument("tmp", "Scratch tile required by hardware (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x5)
            << "The operator block.sel requires 5 arguments (out, mask, lhs, rhs, tmp)";
        CheckTileArg(args, 0, "block.sel", {MemorySpace::Vec});
        CheckTileArg(args, 1, "block.sel", {MemorySpace::Vec});
        CheckTileArg(args, 2, "block.sel", {MemorySpace::Vec});
        CheckTileArg(args, 3, "block.sel", {MemorySpace::Vec});
        CheckTileArg(args, 4, "block.sel", {MemorySpace::Vec});
        return DeduceBlockOutTileType(args, kwargs, "block.sel", 5);
    });

// sels (out, mask, src, tmp, scalar): 5 args.
// pto-isa TSELS: out[i] = src[i] if mask_bit[i] else scalar.
REGISTER_OP("block.sels")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output mask selection between a source tile and a scalar: "
                     "out[i]=src[i] if mask[i] else scalar; tmp is scratch buffer")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .add_argument("mask", "Predicate mask tile (TileType)")
    .add_argument("src", "Source tile selected where mask bit is 1 (TileType)")
    .add_argument("tmp", "Scratch tile required by hardware (TileType)")
    .add_argument("scalar", "Scalar value selected where mask bit is 0 (ScalarType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x5)
            << "The operator block.sels requires 5 arguments (out, mask, src, tmp, scalar)";
        CheckTileArg(args, 0, "block.sels", {MemorySpace::Vec});
        CheckTileArg(args, 1, "block.sels", {MemorySpace::Vec});
        CheckTileArg(args, 2, "block.sels", {MemorySpace::Vec});
        CheckTileArg(args, 3, "block.sels", {MemorySpace::Vec});
        return DeduceBlockOutTileType(args, kwargs, "block.sels", 5);
    });

// ---------------------------------------------------------------------------
// Comparison operations
// ---------------------------------------------------------------------------

REGISTER_OP("block.cmp")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output element-wise tile comparison: out = (lhs cmp_op rhs)")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .add_argument("lhs", "Left tile (TileType)")
    .add_argument("rhs", "Right tile (TileType)")
    .set_attr<int>("cmp_mode")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutBinaryTile(args, kwargs, "block.cmp");
    });

REGISTER_OP("block.cmps")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output element-wise tile-scalar comparison: out = (tile cmp_op scalar)")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("scalar", "Scalar comparand (ScalarType)")
    .set_attr<int>("cmp_mode")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutBinaryScalar(args, kwargs, "block.cmps");
    });

// ---------------------------------------------------------------------------
// Scalar-to-tile broadcast
// ---------------------------------------------------------------------------

// TEXPANDS is the documented elementwise exception: the ISA accepts both Vec
// (UB) and Mat (L1) destinations on A2/A3 and 950 (TExpandS.hpp
// static_asserts), so it passes a Vec-or-Mat space list to CheckTileArg
// instead of the shared Vec-only one.
static TypePtr DeduceBlockOutExpands([[maybe_unused]] const std::vector<ExprPtr>& args,
                                     [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs,
                                     const std::string& op_name)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x2)
        << op_name << " requires 2 arguments (out, scalar)";
    CheckTileArg(args, 0, op_name, {MemorySpace::Vec, MemorySpace::Mat});
    return DeduceBlockOutTileType(args, kwargs, op_name, 0x2);
}

REGISTER_OP("block.expands")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output scalar broadcast: fill out tile with scalar value (out[i,j] = scalar)")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .add_argument("scalar", "Fill value (ScalarType or constant)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutExpands(args, kwargs, "block.expands");
    });

REGISTER_OP("block.fill_index")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output consecutive index fill: out[j] = start + j (calls TCI)")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .add_argument("start", "Starting index value (ScalarType or constant)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutTileType(args, kwargs, "block.fill_index", 2);
    });

// ---------------------------------------------------------------------------
// Layout operations
// ---------------------------------------------------------------------------

// block.reshape: (out, src_tile, shape_tuple) -> out's type.
REGISTER_OP("block.reshape")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output reshape: reinterpret src tile layout into out's shape")
    .add_argument("out", "Pre-allocated output tile with target shape (TileType)")
    .add_argument("src", "Source tile (TileType)")
    .add_argument("shape", "New shape dimensions (MakeTuple)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutTileType(args, kwargs, "block.reshape", 3);
    });

// block.transpose: (out, src_tile) -> out's type; axis attrs.
REGISTER_OP("block.transpose")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output transpose: swap two axes of src tile into out")
    .add_argument("out", "Pre-allocated output tile with transposed shape (TileType)")
    .add_argument("src", "Source tile (TileType)")
    .set_attr<int>("axis1")
    .set_attr<int>("axis2")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutUnary(args, kwargs, "block.transpose");
    });

// ---------------------------------------------------------------------------
// Quantize / Dequantize operations
// ---------------------------------------------------------------------------

// block.quant: (out, src, scale [, offset]) -> out's type; mode kwarg ("sym" or "asym").
REGISTER_OP("block.quant")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output quantize: TQuant (FP32 -> INT8/UINT8)")
    .add_argument("out", "Pre-allocated output tile (TileType, INT8 or UINT8)")
    .add_argument("src", "Source tile (TileType, FP32)")
    .add_argument("scale", "Per-row scale tile (TileType, FP32)")
    .set_attr<int>("mode")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 3 || args.size() == 4)
            << "block.quant requires 3 args (sym) or 4 args (asym), got " << args.size();
        auto out_type = As<TileType>(args.front()->GetType());
        PRO_IR_CHECK(ExternalError::INVALID_TYPE, out_type) << "block.quant: first argument (out) must be TileType";
        return out_type;
    });

// block.dequant: (out, src, scale, offset) -> out's type.
REGISTER_OP("block.dequant")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output dequantize: TDequant (INT8/INT16 -> FP32)")
    .add_argument("out", "Pre-allocated output tile (TileType, FP32)")
    .add_argument("src", "Source tile (TileType, INT8 or INT16)")
    .add_argument("scale", "Per-row scale tile (TileType, FP32)")
    .add_argument("offset", "Zero-point offset tile (TileType, FP32)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutTileType(args, kwargs, "block.dequant", 4);
    });

} // namespace ir
} // namespace pypto
