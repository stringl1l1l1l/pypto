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
 * @file block_ops/out_memory.cpp
 * \brief Block memory operations with explicit output tiles: load, store, store_fp, move, ub_copy, full, fillpad,
 * fillpad_inplace, fillpad_expand.
 *
 * Each block explicit-output op receives the pre-allocated output tile as its first argument
 * and returns that tile's type rather than creating a fresh SSA result type.
 * This mirrors the hardware semantics where the programmer explicitly manages
 * tile buffers.
 */

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/logging.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/op_registry.h"
#include "ir/transforms/structural_comparison.h"
#include "ir/type.h"
#include "ir/type_inference.h"

namespace pypto {
namespace ir {

// ---------------------------------------------------------------------------
// Common helpers
// ---------------------------------------------------------------------------

static TypePtr DeduceBlockOutFillPadType([[maybe_unused]] const std::vector<ExprPtr>& args,
                                         [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs,
                                         const std::string& op_name, bool allow_expand,
                                         bool require_shared_backing_storage = false)
{
    auto out_type = As<TileType>(DeduceBlockOutTileType(args, kwargs, op_name, 2));
    CHECK(out_type) << op_name << ": out must be TileType";
    auto src_type = As<TileType>(args[1]->GetType());
    CHECK(src_type) << op_name << ": src must be TileType";
    CHECK(out_type->hardwareInfo_.has_value()) << op_name << ": out tile must carry hardware_info metadata";

    HardwareInfo hw = out_type->hardwareInfo_.value();
    int pad_value = static_cast<int>(hw.pad);
    CHECK(pad_value >= static_cast<int>(TilePad::null) && pad_value <= static_cast<int>(TilePad::min))
        << op_name << ": out.hardware_info.pad must be one of TilePad.null/zero/max/min";
    CHECK(pad_value != static_cast<int>(TilePad::null))
        << op_name << ": out.hardware_info.pad must not be TilePad.null";

    if (require_shared_backing_storage) {
        CHECK(src_type->memref_.has_value() && out_type->memref_.has_value())
            << op_name << ": src and out must share backing storage";
        auto src_memref = src_type->memref_.value();
        auto out_memref = out_type->memref_.value();
        CHECK(src_memref->memorySpace_ == out_memref->memorySpace_ &&
              structural_equal(src_memref->addr_, out_memref->addr_))
            << op_name << ": src and out must share backing storage";

        CHECK(src_type->shape_.size() == 0x2 && out_type->shape_.size() == 0x2)
            << op_name << ": src/out tile shapes must be rank-2";
        auto src_rows = As<ConstInt>(src_type->shape_[0]);
        auto src_cols = As<ConstInt>(src_type->shape_[1]);
        auto out_rows = As<ConstInt>(out_type->shape_[0]);
        auto out_cols = As<ConstInt>(out_type->shape_[1]);
        CHECK(src_rows && src_cols && out_rows && out_cols) << op_name << ": src/out tile shapes must be static";
        CHECK(out_rows->value_ == src_rows->value_ && out_cols->value_ == src_cols->value_)
            << op_name << ": src and out tile rows/cols must match";
    }

    if (allow_expand) {
        CHECK(src_type->shape_.size() == 0x2 && out_type->shape_.size() == 0x2)
            << op_name << ": src/out tile shapes must be rank-2";

        auto src_rows = As<ConstInt>(src_type->shape_[0]);
        auto src_cols = As<ConstInt>(src_type->shape_[1]);
        auto out_rows = As<ConstInt>(out_type->shape_[0]);
        auto out_cols = As<ConstInt>(out_type->shape_[1]);
        CHECK(src_rows && src_cols && out_rows && out_cols) << op_name << ": src/out tile shapes must be static";
        CHECK(out_rows->value_ >= src_rows->value_ && out_cols->value_ >= src_cols->value_)
            << op_name << ": out tile rows/cols must be >= src tile rows/cols";
    }
    return out_type;
}

// ---------------------------------------------------------------------------
// Op registration
// ---------------------------------------------------------------------------

// block.load: (out, tensor, offsets) -> TileType (out's type)
REGISTER_OP("block.load")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output load: copy data from a global tensor into a pre-allocated tile. "
                     "The output tile (first arg) defines the destination buffer; its type is returned. "
                     "Partition view sizes are derived from the tile type's shape/valid_shape.")
    .add_argument("out", "Pre-allocated destination tile (TileType)")
    .add_argument("tensor", "Source tensor (TensorType)")
    .add_argument("offsets", "Offset tuple per dimension (MakeTuple)")
    .set_attr<bool>("is_transpose")
    .set_attr<std::vector<int>>("tile_dims")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        CHECK(args.size() == 3) << "block.load requires 3 arguments, got " << args.size();
        CHECK(As<TileType>(args[0]->GetType())) << "block.load: arg 0 must be TileType";
        CHECK(As<TensorType>(args[1]->GetType())) << "block.load: arg 1 must be TensorType";
        auto offsets = As<MakeTuple>(args[2]);
        CHECK(offsets) << "block.load: arg 2 must be MakeTuple (offsets)";
        return args[0]->GetType();
    });

// block.store: (output_tensor, tile, offsets, [pre_quant_scalar]) -> TensorType
REGISTER_OP("block.store")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output store: copy data from a pre-allocated tile to a global tensor.")
    .add_argument("output_tensor", "Destination tensor (TensorType)")
    .add_argument("tile", "Source tile (TileType)")
    .add_argument("offsets", "Offset tuple per dimension (MakeTuple)")
    .add_argument("pre_quant_scalar", "Optional fixpipe quant scale (ScalarType, low-32-bit float bits)")
    .set_attr<std::vector<int>>("tile_dims")
    .set_attr<int>("relu_pre_mode")
    .set_attr<int>("atomic")
    .set_attr<int>("phase")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        CHECK(args.size() == 3 || args.size() == 4) << "block.store requires 3 or 4 arguments, got " << args.size();
        auto out_type = As<TensorType>(args[0]->GetType());
        CHECK(out_type) << "block.store: arg 0 must be TensorType";
        CHECK(As<TileType>(args[1]->GetType())) << "block.store: arg 1 must be TileType";
        auto offsets = As<MakeTuple>(args[2]);
        CHECK(offsets) << "block.store: arg 2 must be MakeTuple (offsets)";
        if (args.size() == 4) {
            CHECK(As<ScalarType>(args[3]->GetType())) << "block.store: arg 3 (pre_quant_scalar) must be ScalarType";
        }
        return out_type;
    });

// block.store_fp: (output_tensor, tile, fp_tile, offsets) -> TensorType
REGISTER_OP("block.store_fp")
    .set_op_category("BlockOp")
    .set_description(
        "Block explicit-output floating-point store: copy data from a pre-allocated Acc tile to a global tensor "
        "using an auxiliary fp tile.")
    .add_argument("output_tensor", "Destination tensor (TensorType)")
    .add_argument("tile", "Source tile (TileType, Acc memory)")
    .add_argument("fp_tile", "Floating-point parameter tile (TileType)")
    .add_argument("offsets", "Offset tuple per dimension (MakeTuple)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        CHECK(args.size() == 4) << "block.store_fp requires 4 arguments, got " << args.size();
        auto out_type = As<TensorType>(args[0]->GetType());
        CHECK(out_type) << "block.store_fp: arg 0 must be TensorType";
        CHECK(As<TileType>(args[1]->GetType())) << "block.store_fp: arg 1 must be TileType";
        CHECK(As<TileType>(args[2]->GetType())) << "block.store_fp: arg 2 must be TileType";
        auto offsets = As<MakeTuple>(args[3]);
        CHECK(offsets) << "block.store_fp: arg 3 must be MakeTuple (offsets)";
        return out_type;
    });

// block.move: (out, src_tile, [offset]) -> TileType (out's type)
REGISTER_OP("block.move")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output move: transfer a tile between memory levels into a pre-allocated buffer. "
                     "The TMOV variant is determined by the output tile's memory space.")
    .add_argument("out", "Pre-allocated destination tile (TileType)")
    .add_argument("src", "Source tile (TileType)")
    .add_argument("offset", "Optional 2D offset [offset_m, offset_k] for sub-tile extraction (TupleType)")
    .add_argument("pre_quant_scalar", "Optional fixpipe quant scale (ScalarType, low-32-bit float bits)")
    .set_attr<int>("acc_to_vec_mode")
    .set_attr<int>("relu_pre_mode")
    .set_attr<int>("phase")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        CHECK(args.size() >= 2 && args.size() <= 4)
            << "The operator block.move requires 2 to 4 arguments, but got " << args.size();
        auto out_type = As<TileType>(args[0]->GetType());
        CHECK(out_type) << "block.move: first argument (out) must be TileType";
        // Optional trailing operands (args[2], args[3]) are distinguished by TYPE, not position:
        // a TupleType is the 2D sub-tile offset; a ScalarType is the pre_quant_scalar quant scale.
        for (size_t i = 2; i < args.size(); ++i) {
            CHECK(As<TupleType>(args[i]->GetType()) || As<ScalarType>(args[i]->GetType()))
                << "block.move: optional arg " << i << " must be TupleType (offset) or ScalarType (pre_quant_scalar)";
        }
        return out_type;
    });

// block.move_fp: (out, src_tile, fp_tile) -> TileType (out's type)
REGISTER_OP("block.move_fp")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output move with scaling tile: convert an Acc tile into a pre-allocated Vec tile.")
    .add_argument("out", "Pre-allocated destination tile (TileType, Vec memory)")
    .add_argument("src", "Source tile (TileType, Acc memory)")
    .add_argument("fp_tile", "Floating-point parameter tile (TileType, Scaling memory)")
    .set_attr<int>("acc_to_vec_mode")
    .set_attr<int>("relu_pre_mode")
    .set_attr<int>("phase")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        CHECK(args.size() == 3) << "The operator block.move_fp requires 3 arguments, but got " << args.size();
        auto out_type = As<TileType>(args[0]->GetType());
        CHECK(out_type) << "block.move_fp: first argument (out) must be TileType";
        CHECK(As<TileType>(args[1]->GetType())) << "block.move_fp: arg 1 must be TileType";
        CHECK(As<TileType>(args[2]->GetType())) << "block.move_fp: arg 2 must be TileType";
        return out_type;
    });

// block.insert: (out, src, row, col) -> TileType
REGISTER_OP("block.insert")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output insert: insert source sub-tile into destination tile at a 2-D offset. "
                     "Corresponds to pto-isa TINSERT instruction for UB→L1 transfer.")
    .add_argument("out", "Destination tile (TileType, Mat memory)")
    .add_argument("src", "Source sub-tile (TileType, Vec memory)")
    .add_argument("row", "Row offset where insertion begins")
    .add_argument("col", "Column offset where insertion begins")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        CHECK(args.size() == 4) << "The operator block.insert requires 4 arguments, but got " << args.size();
        auto out_type = As<TileType>(args[0]->GetType());
        CHECK(out_type) << "block.insert: first argument (out) must be TileType";
        return out_type;
    });

// block.ub_copy: (out, src_tile) -> TileType (out's type)
REGISTER_OP("block.ub_copy")
    .set_op_category("BlockOp")
    .set_description(
        "Block explicit-output UB-to-UB copy: copy a tile within unified buffer into a pre-allocated buffer.")
    .add_argument("out", "Pre-allocated destination UB tile (TileType)")
    .add_argument("src", "Source UB tile (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutTileType(args, kwargs, "block.ub_copy", 2);
    });

// block.ssbuf_store: (struct_var, offset) -> void
// Copies raw bytes of a named C++ struct into SSBUF at byte address offset.
// args[0] is the named-tuple IR Var bound to the struct value; args[1] is the offset.
REGISTER_OP("block.ssbuf_store")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output SSBUF store: copy a named struct into SSBUF at a byte address offset.")
    .add_argument("struct_var", "Named-struct IR Var to copy from")
    .add_argument("offset", "SSBUF byte address offset")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        CHECK(args.size() == 2) << "The operator block.ssbuf_store requires 2 arguments (struct_var, offset)";
        return std::make_shared<ScalarType>(DataType::INDEX);
    });

// block.ssbuf_load: (struct_var, offset) -> void
// Copies raw bytes from SSBUF at byte address offset into a named C++ struct.
REGISTER_OP("block.ssbuf_load")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output SSBUF load: copy SSBUF bytes at a byte address offset into a named struct.")
    .add_argument("struct_var", "Named-struct IR Var to copy into")
    .add_argument("offset", "SSBUF byte address offset")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        CHECK(args.size() == 2) << "The operator block.ssbuf_load requires 2 arguments (struct_var, offset)";
        return std::make_shared<ScalarType>(DataType::INDEX);
    });

// block.full: (out, scalar) -> TileType (out's type)
// Fills the pre-allocated tile with a scalar value. Shape comes from out's TileType.
REGISTER_OP("block.full")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output fill: broadcast a scalar value across a pre-allocated tile (out = scalar).")
    .add_argument("out", "Pre-allocated tile to fill (TileType)")
    .add_argument("scalar", "Fill value (ScalarType or constant)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutTileType(args, kwargs, "block.full", 2);
    });

// block.fillpad: (out, src_tile) -> TileType (out's type)
REGISTER_OP("block.fillpad")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output fill-with-padding: copy src tile into out and pad remaining elements.")
    .add_argument("out", "Pre-allocated destination tile (TileType)")
    .add_argument("src", "Source tile (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutFillPadType(args, kwargs, "block.fillpad", false);
    });

// block.fillpad_inplace: (out, src_tile) -> TileType (out's type)
REGISTER_OP("block.fillpad_inplace")
    .set_op_category("BlockOp")
    .set_description(
        "Block explicit-output inplace fill-with-padding: src and out share backing storage while preserving distinct "
        "metadata.")
    .add_argument("out", "Pre-allocated destination tile (TileType)")
    .add_argument("src", "Source tile (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutFillPadType(args, kwargs, "block.fillpad_inplace", false, true);
    });

// block.fillpad_expand: (out, src_tile) -> TileType (out's type)
REGISTER_OP("block.fillpad_expand")
    .set_op_category("BlockOp")
    .set_description(
        "Block explicit-output fill-with-padding: copy src tile into a larger out tile and pad remaining elements.")
    .add_argument("out", "Pre-allocated destination tile (TileType)")
    .add_argument("src", "Source tile (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutFillPadType(args, kwargs, "block.fillpad_expand", true);
    });

// block.set_validshape: (tile, row, col) -> TileType (tile's type)
REGISTER_OP("block.set_validshape")
    .set_op_category("BlockOp")
    .set_description("Update valid-shape metadata on a dynamic tile in place. "
                     "Emits a pto.set_validshape instruction to set the runtime valid row/col.")
    .add_argument("tile", "Dynamic tile buffer to update (TileType)")
    .add_argument("row", "Runtime valid row count (ScalarType or constant)")
    .add_argument("col", "Runtime valid column count (ScalarType or constant)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockOutTileType(args, kwargs, "block.set_validshape", 3);
    });

} // namespace ir
} // namespace pypto
