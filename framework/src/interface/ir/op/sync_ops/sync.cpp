/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <any>
#include <string>
#include <utility>
#include <vector>

#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/op_registry.h"
#include "ir/type.h"
#include "pypto_pro/error.h"

namespace pypto {
namespace ir {
using npu::tile_fwk::ExternalError;

namespace {

// Helper to deduce UnknownType (for ops with no return value)
TypePtr DeduceUnknownType([[maybe_unused]] const std::vector<ExprPtr>& args,
                          [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    (void)args;
    (void)kwargs;
    return GetUnknownType();
}

bool IsIntScalar(const ExprPtr& expr)
{
    auto scalar_type = As<ScalarType>(expr->GetType());
    return scalar_type && (scalar_type->dtype_.IsInt() || scalar_type->dtype_ == DataType::INDEX);
}

bool IsDcciTensorOffset(const ExprPtr& offset)
{
    if (IsIntScalar(offset)) {
        return true;
    }
    auto tuple = As<MakeTuple>(offset);
    if (!tuple) {
        return false;
    }
    for (const auto& element : tuple->elements_) {
        if (!IsIntScalar(element)) {
            return false;
        }
    }
    return true;
}

} // namespace

// ============================================================================
// Registration Function for Sync Operations
// ============================================================================

// Register system.sync_src_dyn (Set Flag)
REGISTER_OP("system.sync_src_dyn")
    .set_description("Send a synchronization signal with an event_id expression (Set Flag)")
    .set_op_category("SyncOp")
    .add_argument("event_id", "Event ID expression (integer ScalarType)")
    .set_attr<int>("set_pipe")
    .set_attr<int>("wait_pipe")
    .f_deduce_type(DeduceUnknownType);

// Register system.sync_dst_dyn (Wait Flag)
REGISTER_OP("system.sync_dst_dyn")
    .set_description("Wait for a synchronization signal with an event_id expression (Wait Flag)")
    .set_op_category("SyncOp")
    .add_argument("event_id", "Event ID expression (integer ScalarType)")
    .set_attr<int>("set_pipe")
    .set_attr<int>("wait_pipe")
    .f_deduce_type(DeduceUnknownType);

// Register system.bar_m (Matrix Barrier)
// Attributes: None
REGISTER_OP("system.bar_m")
    .set_description("Matrix unit barrier")
    .set_op_category("SyncOp")
    .no_argument()
    .f_deduce_type(DeduceUnknownType);

// Register system.bar_mte1 (MTE1 Barrier)
// Attributes: None
REGISTER_OP("system.bar_mte1")
    .set_description("MTE1 pipeline barrier")
    .set_op_category("SyncOp")
    .no_argument()
    .f_deduce_type(DeduceUnknownType);

// Register system.bar_mte2 (MTE2 Barrier)
// Attributes: None
REGISTER_OP("system.bar_mte2")
    .set_description("MTE2 pipeline barrier")
    .set_op_category("SyncOp")
    .no_argument()
    .f_deduce_type(DeduceUnknownType);

// Register system.bar_mte3 (MTE3 Barrier)
// Attributes: None
REGISTER_OP("system.bar_mte3")
    .set_description("MTE3 pipeline barrier")
    .set_op_category("SyncOp")
    .no_argument()
    .f_deduce_type(DeduceUnknownType);

// Register system.bar_fix (FIX Barrier)
// Attributes: None
REGISTER_OP("system.bar_fix")
    .set_description("FIX pipeline barrier")
    .set_op_category("SyncOp")
    .no_argument()
    .f_deduce_type(DeduceUnknownType);

// Register system.bar_all (Global Barrier)
// Attributes: None
REGISTER_OP("system.bar_all")
    .set_description("Global barrier synchronization")
    .set_op_category("SyncOp")
    .no_argument()
    .f_deduce_type(DeduceUnknownType);

// Register system.set_cross_core (Set Cross Core Flag)
// Attributes: pipe, event_id
REGISTER_OP("system.set_cross_core")
    .set_description("Set for a synchronization signal (Cross core)")
    .set_op_category("SyncOp")
    .no_argument()
    .set_attr<int>("pipe")
    .set_attr<int>("event_id")
    .set_attr<int>("sync_mode")
    .f_deduce_type(DeduceUnknownType);

// Register system.wait_cross_core (Wait Cross Core Flag)
REGISTER_OP("system.wait_cross_core")
    .set_description("Wait for a synchronization signal (Cross core)")
    .set_op_category("SyncOp")
    .no_argument()
    .set_attr<int>("pipe")
    .set_attr<int>("event_id")
    .set_attr<int>("sync_mode")
    .f_deduce_type(DeduceUnknownType);

// Register system.set_cross_core_dyn (Set Cross Core Flag, dynamic event_id)
REGISTER_OP("system.set_cross_core_dyn")
    .set_description("Cross-core sync set with dynamic event_id")
    .set_op_category("SyncOp")
    .add_argument("event_id", "Dynamic event ID (ScalarType with integer dtype)")
    .set_attr<int>("pipe")
    .set_attr<int>("sync_mode")
    .f_deduce_type(DeduceUnknownType);

// Register system.wait_cross_core_dyn (Wait Cross Core Flag, dynamic event_id)
REGISTER_OP("system.wait_cross_core_dyn")
    .set_description("Cross-core sync wait with dynamic event_id")
    .set_op_category("SyncOp")
    .add_argument("event_id", "Dynamic event ID (ScalarType with integer dtype)")
    .set_attr<int>("pipe")
    .set_attr<int>("sync_mode")
    .f_deduce_type(DeduceUnknownType);

// Register system.sync_all (Global Core Synchronization)
// Delegates to pto-isa SYNCALL<SyncCoreType>() / pto.syncall MLIR op.
REGISTER_OP("system.sync_all")
    .set_description("Global core synchronization")
    .set_op_category("SyncOp")
    .no_argument()
    .set_attr<int>("core_type")
    .f_deduce_type(DeduceUnknownType);

// Register system.dcci (Data Cache Clean and Invalid)
// Arguments:
//   - target: TensorType for GM or TileType for UB
//   - offset: optional. Tensor target uses MakeTuple offsets or scalar element offset;
//     Tile target uses scalar element offset.
// Attributes:
//   - cache_line: "SINGLE_CACHE_LINE" or "ENTIRE_DATA_CACHE"
//   - dst: "auto", "CACHELINE_OUT", "CACHELINE_UB", "CACHELINE_ALL", or "CACHELINE_ATOMIC"
REGISTER_OP("system.dcci")
    .set_description("Data Cache Clean and Invalid for GM tensor or UB tile")
    .set_op_category("SyncOp")
    .add_argument("target", "GM tensor (TensorType) or UB tile (TileType)")
    .add_argument("offset", "Optional tensor offsets or tile element offset")
    .set_attr<int>("cache_line")
    .set_attr<int>("dst")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        (void)kwargs;
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 1 || args.size() == 2)
            << "system.dcci requires 1 or 2 arguments, got " << args.size();
        auto target_type = args[0]->GetType();
        PRO_IR_CHECK(ExternalError::INVALID_TYPE, As<TensorType>(target_type) || As<TileType>(target_type))
            << "system.dcci: target must be TensorType or TileType, but got " << target_type->TypeName();
        if (As<TensorType>(target_type) && args.size() == 2) {
            PRO_IR_CHECK(ExternalError::INVALID_TYPE, IsDcciTensorOffset(args[1]))
                << "system.dcci: tensor target offset must be a per-dimension list/tuple "
                << "or a scalar integer element offset";
        }
        if (As<TileType>(target_type) && args.size() == 2) {
            PRO_IR_CHECK(ExternalError::INVALID_TYPE, IsIntScalar(args[1]))
                << "system.dcci: tile target offset must be a scalar integer element offset (int or index Expr).\n"
                << " Example: dcci(tile, 0) - cache invalidation starting at element offset 0\n"
                << " Note: Use list/tuple offset only for tensor targets";
        }
        return GetUnknownType();
    });

// ============================================================================
// Mutex (Buffer-ID Token) Ops — A5 only
// ----------------------------------------------------------------------------
// Acquire/release a Mutex buffer-id token for a given pipe. Used as an
// alternative to event-id based sync_src/sync_dst pairs on A5. Lowered to
// pto.get_buf / pto.rls_buf in the PTO backend codegen.
// Semantics:
//   - Lock<pipe>(id): blocks current pipe instruction queue until the token
//     `id` is released.
//   - Unlock<pipe>(id): releases the token after prior instructions on the
//     given pipe drain.
// Valid mutex_id range: 0-31 (per Ascend C Mutex ISASI spec).
// ============================================================================

// Register system.mutex_lock_dyn (Mutex::Lock, A5)
REGISTER_OP("system.mutex_lock_dyn")
    .set_description("Acquire Mutex buffer-id token with a mutex_id expression (A5)")
    .set_op_category("SyncOp")
    .add_argument("mutex_id", "Mutex ID expression (integer ScalarType)")
    .set_attr<int>("pipe")
    .set_attr<std::vector<int>>("mutex_ids")
    .set_attr<std::vector<int>>("mutex_id_owner_indices")
    .f_deduce_type(DeduceUnknownType);

// Register system.mutex_unlock_dyn (Mutex::Unlock, A5)
REGISTER_OP("system.mutex_unlock_dyn")
    .set_description("Release Mutex buffer-id token with a mutex_id expression (A5)")
    .set_op_category("SyncOp")
    .add_argument("mutex_id", "Mutex ID expression (integer ScalarType)")
    .set_attr<int>("pipe")
    .set_attr<std::vector<int>>("mutex_ids")
    .set_attr<std::vector<int>>("mutex_id_owner_indices")
    .f_deduce_type(DeduceUnknownType);

// ============================================================================
// Mask Control Ops — CCE-mode only
// ----------------------------------------------------------------------------
// These ops directly map to CCE intrinsics for controlling the vector mask
// register. They are NOT lowered through PTO/ISA.
//   - set_mask_count: switch mask register to count mode
//   - set_mask_norm:  switch mask register back to normal (bit) mode
//   - set_vec_mask:   set mask register to (mask_high, mask_low)
//   - reset_mask:     reset mask register to all-ones
// ============================================================================

REGISTER_OP("system.set_mask_count")
    .set_description("Switch vector mask register to count mode")
    .set_op_category("SyncOp")
    .no_argument()
    .f_deduce_type(DeduceUnknownType);

REGISTER_OP("system.set_mask_norm")
    .set_description("Switch vector mask register back to normal (bit) mode")
    .set_op_category("SyncOp")
    .no_argument()
    .f_deduce_type(DeduceUnknownType);

REGISTER_OP("system.set_vec_mask")
    .set_description("Set vector mask register to (mask_high, mask_low)")
    .set_op_category("SyncOp")
    .add_argument("mask_high", "Upper 64-bit mask value")
    .add_argument("mask_low", "Lower 64-bit mask value")
    .f_deduce_type(DeduceUnknownType);

REGISTER_OP("system.reset_mask")
    .set_description("Reset vector mask register to all-ones")
    .set_op_category("SyncOp")
    .no_argument()
    .f_deduce_type(DeduceUnknownType);

// ============================================================================
// Matmul Layout Transform Op — CCE-mode only (A5)
// ----------------------------------------------------------------------------
// Controls the fixpipe L0C drain direction. When enabled, fixpipe drains L0C
// in N-direction (column-first) instead of M-direction (row-first), allowing
// cube and fixpipe to access L0C along orthogonal axes within the same slot.
// This eliminates RAW hazards and enables single-buffer L0C with K-accumulation.
// Maps to AscendC SDK: SetMMLayoutTransform(bool).
// ============================================================================

REGISTER_OP("system.set_mm_layout_transform")
    .set_description("Set matmul layout transform for fixpipe N-direction drain (A5)")
    .set_op_category("SyncOp")
    .no_argument()
    .set_attr<int>("enabled")
    .f_deduce_type(DeduceUnknownType);

} // namespace ir
} // namespace pypto
