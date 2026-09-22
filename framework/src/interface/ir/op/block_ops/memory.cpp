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
 * @file memory.cpp
 * \brief Memory block operations (get_block_idx, load, store)
 *
 * This file implements memory operations for block-level programming.
 * These operations handle data movement between tensors and unified buffers (tiles).
 */

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "core/error.h"
#include "core/logging.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/op_registry.h"
#include "ir/scalar_expr.h"
#include "ir/span.h"
#include "ir/type.h"
#include "ir/type_inference.h"
#include "pypto_pro/error.h"

namespace pypto {
namespace ir {
using npu::tile_fwk::ExternalError;

namespace {

DataType GetValResultDtype(const DataType& element_dtype, bool preserve_dtype)
{
    if (preserve_dtype || !element_dtype.IsInt()) {
        return element_dtype;
    }
    return element_dtype == DataType::UINT64 ? DataType::UINT64 : DataType::INT64;
}

} // namespace

TypePtr DeduceBlockGetBlockIdxType([[maybe_unused]] const std::vector<ExprPtr>& args,
                                   [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs,
                                   const std::string& op_name)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0)
        << "The operator " << op_name << " requires no arguments, but got " << args.size();

    return std::make_shared<ScalarType>(DataType::INT64);
}

TypePtr DeduceBlockCreateTileType([[maybe_unused]] const std::vector<ExprPtr>& args,
                                  [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs,
                                  const std::string& op_name)
{
    // make_tile signature: (shape)
    // TileType requires static compile-time constant shapes
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x2)
        << "The operator " << op_name << " requires exactly 2 arguments, but got " << args.size();

    // Extract dtype attribute
    DataType dtype = GetOpKwarg<DataType>(kwargs, "dtype");

    // First argument must be MakeTuple with static ConstInt elements
    auto shape_tuple = As<MakeTuple>(args[0]);
    PRO_IR_CHECK(ExternalError::DYNAMIC_SHAPE_COMPUTE_UNSUPPORTED, shape_tuple)
        << "The operator " << op_name
        << " requires first argument to be a MakeTuple expression with static shape values, but got "
        << args[0]->TypeName();
    PRO_IR_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, shape_tuple->elements_.size() == 0x2)
        << "TileType only supports rank-2 shape; "
        << "for a 1-D tensor with shape [N], use TileType shape [N, 1] or [1, N], "
        << "and keep the same convention for all tile shapes of that tensor";

    // Validate all elements are ConstInt (static compile-time constants)
    std::vector<ExprPtr> tile_shape;
    tile_shape.reserve(shape_tuple->elements_.size());

    for (size_t i = 0; i < shape_tuple->elements_.size(); ++i) {
        auto const_int = As<ConstInt>(shape_tuple->elements_[i]);
        PRO_IR_CHECK(ExternalError::DYNAMIC_SHAPE_COMPUTE_UNSUPPORTED, const_int)
            << "The operator " << op_name << " shape element " << i
            << " must be a compile-time constant (ConstInt), but got " << shape_tuple->elements_[i]->TypeName();
        PRO_IR_CHECK(ExternalError::INVALID_SHAPE, const_int->value_ > 0)
            << "The operator " << op_name << " shape element " << i << " must be positive, got " << const_int->value_;
        tile_shape.push_back(shape_tuple->elements_[i]);
    }

    TileView tile_view;

    auto valid_shape_tuple = As<MakeTuple>(args[1]);
    if (valid_shape_tuple)
        tile_view.validShape = valid_shape_tuple->elements_;

    HardwareInfo hw_info;

    int blayout = GetOpKwarg<int>(kwargs, "blayout", -1);
    if (blayout >= 0) {
        hw_info.blayout = static_cast<TileLayout>(blayout);
    }

    int slayout = GetOpKwarg<int>(kwargs, "slayout", -1);
    if (slayout >= 0) {
        hw_info.slayout = static_cast<TileLayout>(slayout);
    }

    int fractal = GetOpKwarg<int>(kwargs, "fractal", -1);
    if (fractal >= 0) {
        hw_info.fractal = static_cast<uint64_t>(fractal);
    }

    int pad = GetOpKwarg<int>(kwargs, "pad", -1);
    if (pad >= 0) {
        hw_info.pad = static_cast<TilePad>(pad);
    }

    int compact = GetOpKwarg<int>(kwargs, "compact", -1);
    if (compact >= 0) {
        hw_info.compact = static_cast<CompactMode>(compact);
    }
    // If explicit memref kwargs are provided (addr + size + id), attach a MemRef to the TileType.
    // This allows the PTO codegen to emit pto.alloc_tile with base_addr directly from the IR,
    // without requiring the init_memref pass.
    MemorySpace target_memory = GetOpKwarg<MemorySpace>(kwargs, "target_memory",
                                                        std::optional<MemorySpace>(MemorySpace::Vec));

    bool has_memref = false;
    for (const auto& kwarg : kwargs) {
        if (kwarg.first == "memref_id") {
            has_memref = true;
            break;
        }
    }
    if (has_memref) {
        int64_t addr_val = GetOpKwarg<int>(kwargs, "memref_addr");
        int64_t size_val = GetOpKwarg<int>(kwargs, "memref_size");
        uint64_t id_val = static_cast<uint64_t>(GetOpKwarg<int>(kwargs, "memref_id"));
        auto addr_expr = std::make_shared<ConstInt>(addr_val, DataType::INT64, Span::Unknown());
        MemRefPtr memref = std::make_shared<MemRef>(target_memory, addr_expr, static_cast<uint64_t>(size_val), id_val);
        return std::make_shared<TileType>(tile_shape, dtype, std::optional<MemRefPtr>(memref), tile_view, hw_info);
    }

    return std::make_shared<TileType>(tile_shape, dtype, std::nullopt, tile_view, hw_info);
}

TypePtr DeduceGetValType([[maybe_unused]] const std::vector<ExprPtr>& args,
                         [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x2)
        << "getval requires exactly 2 arguments, but got " << args.size();

    auto first_type = args[0]->GetType();
    auto offset_type = As<ScalarType>(args[1]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, offset_type)
        << "getval requires offset to be ScalarType, but got " << args[1]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, offset_type->dtype_.IsInt())
        << "getval offset must have integer dtype, but got " << offset_type->dtype_.ToString();

    // SIMT element reads preserve their storage dtype; SIMD scalar reads retain integer widening.
    const bool preserve_dtype = GetOpKwarg<bool>(kwargs, "preserve_dtype", false);
    if (auto tile_type = As<TileType>(first_type)) {
        return std::make_shared<ScalarType>(GetValResultDtype(tile_type->dtype_, preserve_dtype));
    }
    auto tensor_type = As<TensorType>(first_type);
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, tensor_type)
        << "getval requires first argument to be TileType or TensorType, but got " << first_type->TypeName();
    return std::make_shared<ScalarType>(GetValResultDtype(tensor_type->dtype_, preserve_dtype));
}

TypePtr DeduceTileValidShapeType(const std::vector<ExprPtr>& args,
                                 const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 1 && As<TileType>(args[0]->GetType()))
        << "block.tile_valid_shape requires one Tile argument";
    int axis = GetOpKwarg<int>(kwargs, "axis", 0);
    PRO_IR_CHECK(ExternalError::INVALID_SHAPE, axis >= 0 && axis <= 1)
        << "block.tile_valid_shape axis must be in [0, 1], got axis=" << axis;
    return std::make_shared<ScalarType>(DataType::INT64);
}

TypePtr DeduceSetValType([[maybe_unused]] const std::vector<ExprPtr>& args,
                         [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x3)
        << "setval requires exactly 3 arguments, but got " << args.size();

    auto first_type = args[0]->GetType();
    auto offset_type = As<ScalarType>(args[1]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, offset_type)
        << "setval requires offset to be ScalarType, but got " << args[1]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, offset_type->dtype_.IsInt())
        << "setval offset must have integer dtype, but got " << offset_type->dtype_.ToString();
    auto value_type = As<ScalarType>(args[2]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, value_type)
        << "setval requires value to be ScalarType, but got " << args[2]->GetType()->TypeName();

    if (auto tile_type = As<TileType>(first_type)) {
        return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, tile_type->memref_);
    }
    auto tensor_type = As<TensorType>(first_type);
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, tensor_type)
        << "setval requires first argument to be TileType or TensorType, but got " << first_type->TypeName();
    return std::make_shared<TensorType>(tensor_type->shape_, tensor_type->dtype_);
}

// ============================================================================
// Registration Function for Block Memory Operations
// ============================================================================

REGISTER_OP("get_block_idx")
    .set_op_category("LanguageOp")
    .set_description("Get the current block index")
    .no_argument()
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockGetBlockIdxType(args, kwargs, "get_block_idx");
    });

REGISTER_OP("get_block_num")
    .set_op_category("LanguageOp")
    .set_description("Get the current block number")
    .no_argument()
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockGetBlockIdxType(args, kwargs, "get_block_num");
    });

REGISTER_OP("get_subblock_idx")
    .set_op_category("LanguageOp")
    .set_description("Get the current subblock index")
    .no_argument()
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockGetBlockIdxType(args, kwargs, "get_subblock_idx");
    });

REGISTER_OP("get_subblock_num")
    .set_op_category("LanguageOp")
    .set_description("Get the subblock count per AI Core (task ration)")
    .no_argument()
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockGetBlockIdxType(args, kwargs, "get_subblock_num");
    });

REGISTER_OP("get_spr")
    .set_op_category("LanguageOp")
    .set_description("Read special purpose register value (get_ar instruction). "
                     "Currently only AR register is supported.")
    .no_argument()
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0) << "get_spr requires no arguments";
        return std::make_shared<ScalarType>(DataType::INT64);
    });

REGISTER_OP("set_saturation_flag")
    .set_op_category("LanguageOp")
    .set_description("Set saturation flag in CTRL special purpose register.")
    .no_argument()
    .set_attr<int>("mode")
    .set_attr<bool>("enable")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return GetUnknownType();
    });

REGISTER_OP("get_saturation_flag")
    .set_op_category("LanguageOp")
    .set_description("Read saturation flag from CTRL special purpose register.")
    .no_argument()
    .set_attr<int>("mode")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return std::make_shared<ScalarType>(DataType::BOOL);
    });

REGISTER_OP("set_ctrl_spr")
    .set_op_category("LanguageOp")
    .set_description("Set a bit range in the CTRL special purpose register.")
    .add_argument("start_bit", "Start bit index (0-63)")
    .add_argument("end_bit", "End bit index (0-63)")
    .add_argument("value", "Value to write into the bit range")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 3)
            << "set_ctrl_spr requires 3 arguments (start_bit, end_bit, value), but got " << args.size();
        return GetUnknownType();
    });

REGISTER_OP("get_ctrl_spr")
    .set_op_category("LanguageOp")
    .set_description("Read a bit range from the CTRL special purpose register.")
    .add_argument("start_bit", "Start bit index (0-63)")
    .add_argument("end_bit", "End bit index (0-63)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 2)
            << "get_ctrl_spr requires 2 arguments (start_bit, end_bit), but got " << args.size();
        return std::make_shared<ScalarType>(DataType::INT64);
    });

REGISTER_OP("reset_ctrl_spr")
    .set_op_category("LanguageOp")
    .set_description("Reset a bit range in the CTRL register to default values.")
    .add_argument("start_bit", "Start bit index (0-63)")
    .add_argument("end_bit", "End bit index (0-63)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 2)
            << "reset_ctrl_spr requires 2 arguments (start_bit, end_bit), but got " << args.size();
        return GetUnknownType();
    });

REGISTER_OP("block.make_tile")
    .set_op_category("BlockOp")
    .set_description("Create a tile")
    .add_argument("shape", "Shape dimensions (TupleType of ScalarType(INT64))")
    .add_argument("valid_shape", "Valid shape dimensions (optional, TupleType)")
    .set_attr<DataType>("dtype")
    .set_attr<MemorySpace>("target_memory")
    .set_attr<int>("memref_addr")
    .set_attr<int>("memref_size")
    .set_attr<int>("memref_id")
    .set_attr<int>("blayout")
    .set_attr<int>("slayout")
    .set_attr<int>("fractal")
    .set_attr<int>("pad")
    .set_attr<int>("compact")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockCreateTileType(args, kwargs, "block.make_tile");
    });

REGISTER_OP("block.getval")
    .set_op_category("BlockOp")
    .set_description("Read a scalar value from a tile or tensor at offset")
    .add_argument("container", "Input tile (TileType) or tensor (TensorType)")
    .add_argument("offset", "Element offset (ScalarType with integer dtype)")
    .set_attr<bool>("preserve_dtype")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceGetValType(args, kwargs);
    });

REGISTER_OP("block.tile_valid_shape")
    .set_op_category("BlockOp")
    .set_description("Read one runtime valid-shape dimension of a Tile")
    .add_argument("tile", "Input tile")
    .set_attr<int>("axis")
    .f_deduce_type(DeduceTileValidShapeType);

REGISTER_OP("block.setval")
    .set_op_category("BlockOp")
    .set_description("Write a scalar value to a tile or tensor at offset")
    .add_argument("container", "Input tile (TileType) or tensor (TensorType)")
    .add_argument("offset", "Element offset (ScalarType with integer dtype)")
    .add_argument("value", "Scalar value to write (ScalarType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSetValType(args, kwargs);
    });

// ============================================================================
// block.subview — Tile/Tensor sub-view with offset and new shape
// ============================================================================

TypePtr DeduceSubViewType([[maybe_unused]] const std::vector<ExprPtr>& args,
                          [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    // args: [container, offset, valid_shape]
    // Type deduction uses the container's original shape (preserving row_stride).
    // The valid_shape argument is consumed by codegen to auto-emit SetValidShape.
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x3)
        << "block.subview requires exactly 3 arguments (container, offset, valid_shape), but got " << args.size();
    auto container_type = args[0]->GetType();

    if (auto tile_type = As<TileType>(container_type)) {
        return tile_type;
    }

    PRO_IR_CHECK(ExternalError::INVALID_TYPE, false)
        << "block.subview requires first argument to be TileType, but got " << container_type->TypeName();
    return nullptr;
}

REGISTER_OP("block.subview")
    .set_op_category("BlockOp")
    .set_description("Create a sub-view of a tile with offset and valid_shape")
    .add_argument("container", "Input tile (TileType)")
    .add_argument("offset", "Linear element offset (ScalarType with integer dtype)")
    .add_argument("valid_shape", "Sub-window dimensions (MakeTuple)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSubViewType(args, kwargs);
    });

} // namespace ir
} // namespace pypto
