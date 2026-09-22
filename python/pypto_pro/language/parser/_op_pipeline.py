# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Block-op pipeline metadata used by parser features that need pipe IDs."""

from __future__ import annotations

from pypto.pypto_impl.backend import BackendCCE
from pypto.pypto_impl.ir import MemorySpace, PipeType

_BLOCK_OP_ALIASES: dict[str, str] = {
    "and_": "and",
    "eq": "cmp",
    "ne": "cmp",
    "lt": "cmp",
    "le": "cmp",
    "gt": "cmp",
    "ge": "cmp",
    "select": "sel",
    "load_tile": "load",
    "not_": "not",
    "or_": "or",
    "sum": "row_sum",
    "argmax": "row_argmax",
    "argmin": "row_argmin",
    "expand_max": "row_expand_max",
    "expand_min": "row_expand_min",
    "expand_mul": "row_expand_mul",
    "expand_sub": "row_expand_sub",
    "expand_div": "row_expand_div",
}


def get_op_pipe(op_name: str) -> PipeType | None:
    """Return the pipe registered by the CCE backend for a block op."""
    backend_op_name = _BLOCK_OP_ALIASES.get(op_name, op_name)
    return BackendCCE.instance().get_op_pipe(f"block.{backend_op_name}")


# ----------------------------------------------------------------------------
# Descriptor-only (metadata) ops
# ----------------------------------------------------------------------------
# These ops only rewrite a tile's *descriptor* (e.g. its valid (row, col)
# window); they never read or write the tile's buffer *data*.  Accesses to
# their tiles therefore cannot race, so they need neither a buffer mutex
# (auto_mutex) nor cross-core / cross-pipe synchronisation.  This mirrors the
# tile-graph sync pass, which likewise excludes the metadata opcodes
# OP_VIEW / OP_VIEW_TYPE / OP_RESHAPE / OP_ASSEMBLE from synchronisation.
_DESCRIPTOR_ONLY_OPS: frozenset[str] = frozenset(
    {
        "set_validshape",
    }
)


def op_accesses_buffer(op_name: str) -> bool:
    """Whether a block op reads or writes its tiles' buffer data.

    False for descriptor-only ops (they only touch tile metadata), which is the
    signal that auto_mutex / sync insertion can skip them.
    """
    return _BLOCK_OP_ALIASES.get(op_name, op_name) not in _DESCRIPTOR_ONLY_OPS


def get_move_pipe(
    src_memory: MemorySpace | None,
    target_memory: MemorySpace | None,
) -> PipeType:
    """Determine the hardware pipe for a block ``move`` operation."""
    if src_memory == MemorySpace.Acc and target_memory in (MemorySpace.Vec, MemorySpace.Mat):
        return PipeType.FIX
    if src_memory == MemorySpace.Mat:
        if target_memory in (MemorySpace.Left, MemorySpace.Right, MemorySpace.Bias, MemorySpace.ScaleLeft,
                             MemorySpace.ScaleRight):
            return PipeType.MTE1
        if target_memory == MemorySpace.Vec:
            return PipeType.V
        return PipeType.FIX
    if src_memory == MemorySpace.Vec and target_memory == MemorySpace.Mat:
        return PipeType.MTE3
    return PipeType.V


def get_store_pipe(src_memory: MemorySpace | None) -> PipeType:
    """Determine the hardware pipe for a block ``store`` / ``store_tile`` operation."""
    if src_memory == MemorySpace.Acc:
        return PipeType.FIX
    return PipeType.MTE3


# ----------------------------------------------------------------------------
# Per-op tile-argument role table (for cross-core sync direction inference)
# ----------------------------------------------------------------------------
#
# Each entry is a list of role strings, one per Python-call positional arg:
#   "W"   - written  (producer of cross-core data)
#   "R"   - read     (consumer of cross-core data)
#   "RW"  - both read and written (e.g. matmul_acc accumulator)
#   None  - not a tile arg (skip cross-core sync check at this position)
#
# Used by the pipeline cross-core scanner to determine for each tile argument
# the access role (W/R), which drives auto cross-core wait/set generation.
#
# Note: cross-core sync MUST distinguish read vs write because the sync
# direction depends on it.

# ----------------------------------------------------------------------------
# Per-op tile-argument role table for vector-function (vf.*) ops
# ----------------------------------------------------------------------------
#
# Same shape and meaning as _BLOCK_OP_TILE_ROLES below, for the vf.* ops that
# touch a UB tile.  Every other vf op works purely on registers and is absent.
#
# A table rather than a rule, because no rule holds: the pointer is not always
# the first argument (the unaligned loads take their alignment-state register
# there), and the name does not say the direction either --- `gather` reads a
# tile, `scatter` and `squeeze_store_unalign` write one, and neither starts
# with "load" or "store".
#
# Adding a vf op that reads or writes a UB tile means adding it here too;
# the pipeline scan reports an op it cannot find when one reaches a buffer
# that cross-core sync tracks.

_VF_OP_TILE_ROLES: dict[str, list] = {
    # ===== Aligned =====
    "load_align": ["R"],  # load_align(src, offset, ...) -> reg
    "store_align": ["W", None],  # store_align(dst, src, ...)
    # ===== Unified (AscendC-style Load/Store) =====
    "load": ["R"],  # load(src_ptr, stride, ...) -> reg
    "store": ["W", None, None],  # store(dst_ptr, src, count)
    # ===== Unaligned: arg 0 is the alignment-state register, not the tile =====
    "load_unalign_pre": [None, "R"],  # load_unalign_pre(ureg, src_ptr)
    "load_unalign": [None, "R", None],  # load_unalign(ureg, src_ptr, stride)
    "store_unalign": ["W", None, None, None],  # (dst_ptr, src, align_reg, stride)
    "store_unalign_post": ["W", None, None],  # (dst_ptr, align_reg, stride)
    "squeeze_store_unalign": ["W", None, None],  # (dst_ptr, src, align_reg)
    "squeeze_store_unalign_post": ["W", None],  # (dst_ptr, align_reg)
    # ===== Indexed =====
    "gather": ["R", None, None],  # gather(src, indices, mask) -- src is a UB tile
    "scatter": ["W", None, None, None],  # scatter(base_ptr, src, index, mask)
}


_BLOCK_OP_TILE_ROLES: dict[str, list] = {
    # ===== Scalar element access (S pipe) =====
    # Reached both as pl.getval(t, i) / pl.setval(t, i, v) and as the subscript spellings
    # t[i] / t[i] = v that the parser lowers to them. A scalar read of a tile another core
    # wrote still has to wait for that write: measured on device, without the handshake the
    # read returns the buffer's previous contents, and wait_cross_core(pipe=PipeType.S)
    # fixes it.
    "getval": ["R", None],  # getval(container, offset) -> scalar
    "setval": ["W", None, None],  # setval(container, offset, value)
    # ===== Data movement =====
    "load": ["W"],  # load(out, tensor, offsets, ...)
    "load_tile": ["W"],
    "store": ["W", "R"],  # store(out, tile, offsets, ...)
    "store_tile": ["W", "R"],
    "move": ["W", "R", "R", "R"],  # move(out, src, [offset], [scale])
    "insert": ["W", "R", None, None, "R"],  # insert(out, src, row, col, [scale])
    # ===== Matmul =====
    "matmul": ["W", "R", "R"],  # matmul(out, lhs, rhs)
    "matmul_acc": ["W", "RW", "R", "R"],  # matmul_acc(out, acc, lhs, rhs)
    "matmul_bias": ["W", "R", "R", "R"],  # matmul_bias(out, lhs, rhs, bias)
    "matmul_mx": ["W", "R", "R", "R", "R"],  # matmul_mx(out, lhs, rhs, scale_a, scale_b)
    "matmul_mx_acc": ["W", "RW", "R", "R", "R", "R"],  # matmul_mx_acc(out, acc, lhs, rhs, scale_a, scale_b)
    "gemv": ["W", "R", "R"],
    "gemv_acc": ["W", "RW", "R", "R"],
    "gemv_bias": ["W", "R", "R", "R"],
    # ===== Binary element-wise =====
    "add": ["W", "R", "R"],
    "sub": ["W", "R", "R"],
    "mul": ["W", "R", "R"],
    "div": ["W", "R", "R"],
    "rem": ["W", "R", "R"],
    "maximum": ["W", "R", "R"],
    "minimum": ["W", "R", "R"],
    "and_": ["W", "R", "R"],
    "or_": ["W", "R", "R"],
    "xor": ["W", "R", "R", None],
    "shl": ["W", "R", "R"],
    "shr": ["W", "R", "R"],
    # ===== Part ops =====
    "partadd": ["W", "R", "R"],
    "partmax": ["W", "R", "R"],
    "partmin": ["W", "R", "R"],
    "partmul": ["W", "R", "R"],
    "lrelu": ["W", "R", "R"],
    "prelu": ["W", "R", "R", None],
    # ===== Tile-scalar (rhs is scalar, not tile) =====
    "adds": ["W", "R", None],
    "subs": ["W", "R", None],
    "muls": ["W", "R", None],
    "divs": ["W", "R", None],
    "rems": ["W", "R", None],
    "ands": ["W", "R", None],
    "ors": ["W", "R", None],
    "xors": ["W", "R", None, None],
    "shls": ["W", "R", None],
    "shrs": ["W", "R", None],
    "maxs": ["W", "R", None],
    "mins": ["W", "R", None],
    # ===== Unary =====
    "neg": ["W", "R"],
    "exp": ["W", "R"],
    "log": ["W", "R"],
    "sqrt": ["W", "R"],
    "rsqrt": ["W", "R"],
    "recip": ["W", "R"],
    "abs": ["W", "R"],
    "relu": ["W", "R"],
    "not_": ["W", "R"],
    # ===== Type cast =====
    "cast": ["W", "R"],  # cast(out, src, *, mode)
    # ===== Select / compare =====
    "select": ["W", "R", "R", "R", None],
    "eq": ["W", "R", "R"],
    "ne": ["W", "R", "R"],
    "lt": ["W", "R", "R"],
    "le": ["W", "R", "R"],
    "gt": ["W", "R", "R"],
    "ge": ["W", "R", "R"],
    # ===== Reduce (unified: dim=0 → row, dim=1 → col) =====
    "sum": ["W", "R", None],
    "argmax": ["W", "R", None],
    "argmin": ["W", "R", None],
    "row_prod": ["W", "R", None],
    "col_prod": ["W", "R", None],
    "row_expand": ["W", "R"],
    "col_expand": ["W", "R"],
    "row_expand_add": ["W", "R", "R"],
    "expand_sub": ["W", "R", "R"],
    "expand_mul": ["W", "R", "R"],
    "expand_div": ["W", "R", "R"],
    "expand_max": ["W", "R", "R"],
    "expand_min": ["W", "R", "R"],
    "row_expand_binop": ["W", "R", "R"],
    "col_expand_binop": ["W", "R", "R"],
    "expands": ["W", None],
    # ===== Layout =====
    "reshape": ["W", "R", None],
    "transpose": ["W", "R"],
    "ub_copy": ["W", "R"],
    # ===== Fill =====
    "full": ["W", None],
    "fillpad": ["W", "R"],
    "fillpad_expand": ["W", "R"],
    "fillpad_inplace": ["RW", "R"],
}


def get_op_tile_roles(op_name: str) -> list | None:
    """Return the per-arg tile role list for the given op, or None if unknown.

    Returns:
        List of role strings ("W"/"R"/"RW") or None for non-tile arg positions.
        Returns None if op_name is not in the table (caller should skip cross-core
        sync emission for unknown ops to be safe).
    """
    return _BLOCK_OP_TILE_ROLES.get(op_name)
