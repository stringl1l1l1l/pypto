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
 * @file op/op_common.h
 * \brief Shared TileType argument validation for block explicit-output operators.
 *
 * The block explicit-output op families (elementwise, reduction, matmul)
 * validate their tile arguments one by one via CheckTileArg, passing each
 * argument's allowed memory spaces as a list.
 */

#ifndef PYPTO_IR_OP_OP_COMMON_H_
#define PYPTO_IR_OP_OP_COMMON_H_

#include <algorithm>
#include <initializer_list>
#include <optional>
#include <string>
#include <vector>

#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/memory_space.h"
#include "ir/type.h"
#include "pypto_pro/error.h"

namespace pypto {
namespace ir {

// Report the memory space of a tile, or nullopt when the tile carries no
// memref yet (nothing to check in that case).
inline std::optional<MemorySpace> GetTileMemSpace(const TileTypePtr& tile_type)
{
    if (tile_type->memref_ == std::nullopt) {
        return std::nullopt;
    }
    return (*tile_type->memref_)->memorySpace_;
}

// Format the allowed memory spaces of CheckTileArg for diagnostics, e.g. "Vec or Mat".
inline std::string MemorySpacesToString(std::initializer_list<MemorySpace> spaces)
{
    std::string result;
    for (MemorySpace space : spaces) {
        if (!result.empty()) {
            result += " or ";
        }
        result += MemorySpaceToString(space);
    }
    return result;
}

/**
 * \brief Validate that one operator argument is a TileType whose memory space is allowed
 *
 * Tile arguments are validated one by one: ops with several tile arguments
 * call this once per index, passing the allowed memory spaces of that
 * argument as a list (vec-pipe ops pass {Vec}, cube ops pass {Acc}/{Left}/
 * {Right}/..., and ops that accept several destinations list them all, e.g.
 * {Vec, Mat}). When the tile already carries a memref, its memory space must
 * be one of the allowed spaces; tiles without a memref are accepted (nothing
 * to check yet).
 *
 * \param args Operator argument expressions
 * \param idx Index of the argument to validate
 * \param op_name Operator name for diagnostics
 * \param spaces Allowed memory spaces for the tile
 */
inline void CheckTileArg(const std::vector<ExprPtr>& args, size_t idx, const std::string& op_name,
                         std::initializer_list<MemorySpace> spaces)
{
    using npu::tile_fwk::ExternalError; // Block-scoped: short enum names without leaking to includers
    auto tile_type = As<TileType>(args[idx]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, tile_type)
        << "The operator " << op_name << " requires argument " << idx << " to be TileType, but got "
        << args[idx]->GetType()->TypeName();
    auto space = GetTileMemSpace(tile_type);
    PRO_IR_CHECK(ExternalError::INVALID_OPERATION,
                 space == std::nullopt || std::find(spaces.begin(), spaces.end(), *space) != spaces.end())
        << "The operator " << op_name << " requires argument " << idx << " to be a " << MemorySpacesToString(spaces)
        << " tile, but got " << MemorySpaceToString(space.value_or(MemorySpace::DDR));
}

} // namespace ir
} // namespace pypto

#endif // PYPTO_IR_OP_OP_COMMON_H_
