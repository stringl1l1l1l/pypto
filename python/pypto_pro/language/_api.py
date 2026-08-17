# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Python API declarations for the PyPTO Pro DSL (``pl.xxx``).

These declarations exist so that:
- IDE "Go to Definition" works for every ``pl.xxx`` call
- Python catches typos at import time (``pl.tensr`` raises ``AttributeError``)
- Type checkers can validate argument types
- Docstrings document the user-facing calling convention

None of these functions are meant to be called at runtime.  Inside a PyPTO
kernel the AST parser intercepts every ``pl.xxx`` call before Python executes
it.  Outside a kernel, calling a declaration raises ``RuntimeError``.

When adding a new op to the parser registry / block-default handler, add a
matching declaration here and re-export it from the package ``__init__.py``.
"""

from __future__ import annotations

import builtins
from contextlib import contextmanager
import functools
import inspect
from typing import Any, List, Optional, Union

from pypto.ir import (
    AccPhase,
    AccToVecMode,
    AtomicType,
    QuantMode,
    ReluPreMode,
    RoundMode,
    SaturationFlagMode,
    STPhase,
)
from pypto_pro.ir.op.block_ops import FillPadMode

from . import TensorLayout

# ---------------------------------------------------------------------------
# User-facing type aliases (NOT IR types)
# ---------------------------------------------------------------------------
Tile = Any
TileGroup = Any
Tensor = Any
Scalar = Any
DType = Any
Offset = Union[List[int], int]

# ---------------------------------------------------------------------------
# API declaration decorator
# ---------------------------------------------------------------------------

_API_MSG = "This function is a DSL API declaration and must be used inside a PyPTO kernel"


def _api_decl(func):
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        sig.bind(*args, **kwargs)
        raise RuntimeError(_API_MSG)

    wrapper.__wrapped__ = func
    return wrapper


# ===================================================================
# Section A: Data-movement block ops
# ===================================================================


@_api_decl
def load(dst_tile: Tile, src_tensor: Tensor, offsets: Offset, *, order: Optional[List[int]] = None) -> None:
    """Load data from GM Tensor into on-chip Tile by absolute element coordinates.

    Args:
        dst_tile: Destination Tile (L1 or UB only)
        src_tensor: Source Tensor (global memory, from kernel parameter)
        offsets: Element-level offset per axis, e.g. ``[row, col]`` or ``[b, n, sq, sk]``
        order: Optional, mapping of Tile dimensions to Tensor axes.
            Each element is an absolute axis index of the Tensor.
            Ascending order (e.g. ``[0, 1]``) loads without transposition;
            descending order (e.g. ``[1, 0]``) loads with transposition (DN layout).
            Default: last N axes ascending (N = Tile ndim), i.e. no transposition.
    """


@_api_decl
def load_tile(dst_tile: Tile, src_tensor: Tensor, tile_offsets: Offset, *, order: Optional[List[int]] = None) -> None:
    """Load data from GM Tensor into on-chip Tile by tile-block index.

    Offsets are in tile-block units, internally multiplied by tile shape.
    E.g. with tile shape ``[64, 128]``, ``tile_offsets=[2, 2]`` equals
    ``load`` with absolute offsets ``[128, 256]``.

    Args:
        dst_tile: Destination Tile (L1 or UB only)
        src_tensor: Source Tensor (global memory)
        tile_offsets: Tile-block index, e.g. ``[tile_row, tile_col]``
        order: Optional, mapping of Tile dimensions to Tensor axes.
            Each element is an absolute axis index of the Tensor.
            Ascending order (e.g. ``[0, 1]``) loads without transposition;
            descending order (e.g. ``[1, 0]``) loads with transposition (DN layout).
            Default: last N axes ascending (N = Tile ndim), i.e. no transposition.
    """


@_api_decl
def store(
    dst_tensor: Tensor,
    src_tile: Tile,
    offsets: Offset,
    *,
    relu_pre_mode: Optional[ReluPreMode] = None,
    scale: Optional[Union[float, Scalar, Tile]] = None,
    order: Optional[List[int]] = None,
    atomic: AtomicType = AtomicType.AtomicNone,
    phase: Optional[STPhase] = None,
) -> None:
    """Store on-chip Tile back to GM Tensor by absolute element coordinates.

    Args:
        dst_tensor: Destination Tensor (global memory)
        src_tile: Source Tile (on-chip buffer)
        offsets: Element-level offset per axis, e.g. ``[row, col]`` or ``[b, n, sq, sk]``
        relu_pre_mode: Optional ReLU fusion — ``pl.ReluPreMode.NormalRelu``
        scale: Optional fixpipe quantization scale.
            ``float`` or runtime scalar → per-tensor quantization (deqScalar path);
            a runtime **FP32** scalar is auto-reinterpreted as its IEEE-754 bit
            pattern (codegen bitcast), so pass the raw float value; a runtime
            **INT32/INT64** scalar must carry the pre-encoded float32 bit pattern
            (``struct.pack("!f", scale)``). Other runtime scalar dtypes
            (FP16/BF16/unsigned/narrower ints) are rejected at parse time.
            ``Tile`` (INT64, MemorySpace.Scaling, shape ``[1, N]``) → per-channel quantization with a
            **user-prepared deqTensor tile**; the framework reuses it directly (no auto-allocation,
            no sync insertion) — the user owns the data flow and must ensure the tile is ready
            (load → move → sync MTE1→FIX) before the store.
            Hardware requires ``[1, N]`` (row == 1, per-column), ``N % 16 == 0`` and ``N <= 512`` —
            ``[N, 1]`` per-row scaling is NOT supported.
            ``None`` → no fixpipe quantization.
            Quantization direction (quantize vs dequantize) is determined by ``dst_tensor`` dtype:
            INT8 → quantize; FP16 → dequantize (from INT32 L0C). Unsupported scale combos —
            UINT8 output, and FP32/INT32 → BF16 output — are rejected at parse time
            (hardware fixpipe has no unsigned requantization and only dequantizes INT32→FP16).
        order: Optional, which axes of the Tensor the Tile dimensions map to.
            When the Tensor has more dimensions than the Tile, this specifies the mapping.
            E.g. ``order=[0, 2]`` means Tile dim 0 → Tensor axis 0, Tile dim 1 → Tensor axis 2.
            Default: last N axes of the Tensor (N = Tile ndim)
        atomic: Atomic write mode — ``pl.AtomicType.AtomicNone`` (overwrite) or
            ``pl.AtomicType.AtomicAdd`` (atomic accumulate)
        phase: Fixpipe drain phase — ``pl.STPhase.Partial`` or ``pl.STPhase.Final``
    """


@_api_decl
def store_tile(
    dst_tensor: Tensor,
    src_tile: Tile,
    tile_offsets: Offset,
    *,
    relu_pre_mode: Optional[ReluPreMode] = None,
    scale: Optional[Union[float, Scalar, Tile]] = None,
    order: Optional[List[int]] = None,
    atomic: AtomicType = AtomicType.AtomicNone,
    phase: Optional[STPhase] = None,
) -> None:
    """Store on-chip Tile back to GM Tensor by tile-block index.

    Args:
        dst_tensor: Destination Tensor (global memory)
        src_tile: Source Tile (on-chip buffer)
        tile_offsets: Tile-block index, e.g. ``[tile_row, tile_col]``;
            internally multiplied by tile shape to get element offsets
        relu_pre_mode: Optional ReLU fusion — ``pl.ReluPreMode.NormalRelu``
        scale: Optional fixpipe quantization scale.
            ``float`` or runtime scalar → per-tensor quantization (deqScalar path);
            a runtime **FP32** scalar is auto-reinterpreted as its IEEE-754 bit
            pattern (codegen bitcast), so pass the raw float value; a runtime
            **INT32/INT64** scalar must carry the pre-encoded float32 bit pattern
            (``struct.pack("!f", scale)``). Other runtime scalar dtypes
            (FP16/BF16/unsigned/narrower ints) are rejected at parse time.
            ``Tile`` (INT64, MemorySpace.Scaling, shape ``[1, N]``) → per-channel quantization with a
            **user-prepared deqTensor tile**; the framework reuses it directly (no auto-allocation,
            no sync insertion) — the user owns the data flow and must ensure the tile is ready
            (load → move → sync MTE1→FIX) before the store.
            Hardware requires ``[1, N]`` (row == 1, per-column), ``N % 16 == 0`` and ``N <= 512`` —
            ``[N, 1]`` per-row scaling is NOT supported.
            ``None`` → no fixpipe quantization.
        order: Optional, which axes of the Tensor the Tile dimensions map to.
            When the Tensor has more dimensions than the Tile, this specifies the mapping.
            E.g. ``order=[0, 2]`` means Tile dim 0 → Tensor axis 0, Tile dim 1 → Tensor axis 2.
            Default: last N axes of the Tensor (N = Tile ndim)
        atomic: Atomic write mode — ``pl.AtomicType.AtomicNone`` (overwrite) or
            ``pl.AtomicType.AtomicAdd`` (atomic accumulate)
        phase: Fixpipe drain phase — ``pl.STPhase.Partial`` or ``pl.STPhase.Final``
    """


@_api_decl
def move(
    dst_tile: Tile,
    src_tile: Tile,
    offset: Optional[Offset] = None,
    *,
    acc_to_vec_mode: Optional[AccToVecMode] = None,
    relu_pre_mode: Optional[ReluPreMode] = None,
    scale: Optional[Union[float, Scalar, Tile]] = None,
    phase: Optional[STPhase] = None,
) -> None:
    """Move data between on-chip Tiles (tile↔tile, no GM access).

    Supported memory-space paths:

    ============ ============ ========
    src          dst          pipe
    ============ ============ ========
    Acc (L0C)    Vec (UB)     fix
    Mat (L1)     Left (L0A)   mte1
    Mat (L1)     Right (L0B)  mte1
    Mat (L1)     Vec (UB)     v
    Vec (UB)     Mat (L1)     mte3
    others       —            v
    ============ ============ ========

    Supported fusion (side operations):

    - ``acc_to_vec_mode``: Acc→Vec conversion mode (single/dual split M/N)
    - ``relu_pre_mode``: ReLU activation before destination
    - ``scale``: fixpipe quantization scale (per-tensor or per-channel)

    Args:
        dst_tile: Destination Tile
        src_tile: Source Tile
        offset: Optional ``[offset_m, offset_k]`` to extract a sub-block from a wider source Tile
        acc_to_vec_mode: Acc→Vec mode — ``pl.AccToVecMode.SingleModeVec0``, ``pl.AccToVecMode.SingleModeVec1``,
            ``pl.AccToVecMode.DualModeSplitM``, ``pl.AccToVecMode.DualModeSplitN``;
            only meaningful when src is Acc and dst is Vec
        relu_pre_mode: Optional ReLU fusion — ``pl.ReluPreMode.NormalRelu``
        scale: Optional fixpipe quantization scale.
            ``float`` or runtime scalar → per-tensor quantization (deqScalar path);
            a runtime **FP32** scalar is auto-reinterpreted as its IEEE-754 bit
            pattern (codegen bitcast), so pass the raw float value; a runtime
            **INT32/INT64** scalar must carry the pre-encoded float32 bit pattern
            (``struct.pack("!f", scale)``). Other runtime scalar dtypes
            (FP16/BF16/unsigned/narrower ints) are rejected at parse time.
            ``Tile`` (INT64, MemorySpace.Scaling, shape ``[1, N]``) → per-channel quantization with a
            **user-prepared deqTensor tile**; the framework reuses it directly (no auto-allocation,
            no sync insertion) — the user owns the data flow and must ensure the tile is ready
            (load → move → sync MTE1→FIX) before the move.
            Hardware requires ``[1, N]`` (row == 1, per-column), ``N % 16 == 0`` and ``N <= 512`` —
            ``[N, 1]`` per-row scaling is NOT supported.
            ``None`` → no fixpipe quantization.
        phase: Optional — ``pl.STPhase.Partial`` or ``pl.STPhase.Final``; only for Acc->Vec path;
            enables hardware unit_flag handshake with matmul producer; cannot be combined with ``offset``
    """


@_api_decl
def insert(dst_tile: Tile, src_tile: Tile, offset: List[int]) -> None:
    """Insert a small Tile into a larger Tile at the given 2-D offset (TINSERT).

    Args:
        dst_tile: Destination (larger) Tile
        src_tile: Source (smaller) Tile
        offset: Insertion coordinates ``[row, col]`` in the destination
    """


@_api_decl
def ssbuf_load(struct_var: Any, offset: int) -> None:
    """Load data from SuperScalar Buffer (SSBUF) into a struct variable.

    Args:
        struct_var: Struct variable (created via ``pl.struct``)
        offset: SSBUF offset
    """


@_api_decl
def ssbuf_store(struct_var: Any, offset: int) -> None:
    """Write a struct variable to SuperScalar Buffer (SSBUF).

    Args:
        struct_var: Struct variable (created via ``pl.struct``)
        offset: SSBUF offset
    """


# ===================================================================
# Section B: Compute block ops
# ===================================================================

# --- B1. Binary element-wise (out, lhs, rhs) ---
#
# Constraints (apply to all ops in this section):
# - No broadcast: ``out``, ``lhs``, ``rhs`` (when Tile) must have identical shape.
# - No implicit type promotion: all operands must have the same dtype.
# - Supported dtypes: FP16, FP32, BF16 (op-dependent; FP8/FP4 are storage-only
#   — use vf.astype to convert to/from FP32/BF16/FP16 for computation).


@_api_decl
def add(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None:
    """Element-wise addition: ``out = lhs + rhs``

    Supports both tile-tile and tile-scalar operations:
        - ``add(out, tile_a, tile_b)`` -> tile-tile
        - ``add(out, tile_a, scalar)`` -> tile-scalar
    """


@_api_decl
def sub(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None:
    """Element-wise subtraction: ``out = lhs - rhs``

    Supports both tile-tile and tile-scalar operations:
        - ``sub(out, tile_a, tile_b)`` -> tile-tile
        - ``sub(out, tile_a, scalar)`` -> tile-scalar
    """


@_api_decl
def mul(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None:
    """Element-wise multiplication: ``out = lhs * rhs``

    Supports both tile-tile and tile-scalar operations:
        - ``mul(out, tile_a, tile_b)`` -> tile-tile
        - ``mul(out, tile_a, scalar)`` -> tile-scalar
    """


@_api_decl
def div(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None:
    """Element-wise division: ``out = lhs / rhs``

    Supports both tile-tile and tile-scalar operations:
        - ``div(out, tile_a, tile_b)`` -> tile-tile
        - ``div(out, tile_a, scalar)`` -> tile-scalar
    """


# --- B2. Bitwise element-wise (out, lhs, rhs) ---


@_api_decl
def and_(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None:
    """Element-wise bitwise AND: ``out = lhs & rhs``

    Supports both tile-tile and tile-scalar operations:
        - ``and_(out, tile_a, tile_b)`` -> tile-tile
        - ``and_(out, tile_a, scalar)`` -> tile-scalar
    """


@_api_decl
def shl(out: Tile, lhs: Tile, rhs: Tile) -> None:
    """Element-wise left shift with a Tile shift operand."""


@_api_decl
def shr(out: Tile, lhs: Tile, rhs: Tile) -> None:
    """Element-wise right shift with a Tile shift operand."""


@_api_decl
def xor(out: Tile, lhs: Tile, rhs: Tile, tmp: Tile) -> None:
    """Element-wise bitwise XOR: ``out = lhs ^ rhs``

    Args:
        out: Destination Tile
        lhs: Left-hand Tile
        rhs: Right-hand Tile
        tmp: Workspace Tile
    """


@_api_decl
def expands(out: Tile, scalar: Scalar) -> None:
    """Fill Tile with a scalar (splat): ``out[i] = scalar``"""


# --- B3. Unary element-wise (out, src) ---


@_api_decl
def neg(out: Tile, src: Tile) -> None:
    """Element-wise negate: ``out = -src``"""


@_api_decl
def abs(out: Tile, src: Tile) -> None:
    """Element-wise absolute value: ``out = |src|``"""


@_api_decl
def exp(out: Tile, src: Tile) -> None:
    """Element-wise exponential: ``out = exp(src)``"""


@_api_decl
def log(out: Tile, src: Tile) -> None:
    """Element-wise natural log: ``out = log(src)``"""


@_api_decl
def sqrt(out: Tile, src: Tile) -> None:
    """Element-wise square root: ``out = sqrt(src)``"""


@_api_decl
def rsqrt(out: Tile, src: Tile) -> None:
    """Element-wise reciprocal square root: ``out = 1/sqrt(src)``"""


@_api_decl
def recip(out: Tile, src: Tile) -> None:
    """Element-wise reciprocal: ``out = 1/src``"""


@_api_decl
def relu(out: Tile, src: Tile) -> None:
    """Element-wise ReLU: ``out = max(0, src)``"""


@_api_decl
def fillpad(out: Tile, src: Tile, *, mode: FillPadMode = FillPadMode.NORMAL) -> None:
    """Fill padding region of a Tile.

    Args:
        out: Destination Tile.
        src: Source Tile.
        mode: Fill mode — ``pl.FillPadMode.NORMAL`` (default, dst and src same shape),
            ``pl.FillPadMode.EXPAND`` (dst larger than src, expand fill),
            ``pl.FillPadMode.INPLACE`` (dst and src share the same address).
    """


# --- B4. Type conversion ---


@_api_decl
def cast(out: Tile, src: Tile, *, mode: RoundMode = RoundMode.CAST_ROUND) -> None:
    """Cast Tile to a different data type.

    The target dtype is inferred from ``out`` tile's dtype.

    Args:
        out: Destination Tile (determines target dtype)
        src: Source Tile
        mode: Rounding mode — ``pl.RoundMode.CAST_NONE``, ``pl.RoundMode.CAST_RINT``, ``pl.RoundMode.CAST_ROUND``,
            ``pl.RoundMode.CAST_FLOOR``, ``pl.RoundMode.CAST_CEIL``,
            ``pl.RoundMode.CAST_TRUNC``, ``pl.RoundMode.CAST_ODD``
    """


@_api_decl
def add_relu_cast(
    out: Tile, lhs: Tile, rhs: Tile, *, target_type: DType, mode: RoundMode = RoundMode.CAST_ROUND
) -> None:
    """Fused add + ReLU + cast: ``out = cast(relu(lhs + rhs), target_type)``"""


@_api_decl
def sub_relu_cast(
    out: Tile, lhs: Tile, rhs: Tile, *, target_type: DType, mode: RoundMode = RoundMode.CAST_ROUND
) -> None:
    """Fused sub + ReLU + cast: ``out = cast(relu(lhs - rhs), target_type)``"""


@_api_decl
def mul_cast(out: Tile, lhs: Tile, rhs: Tile, *, target_type: DType, mode: RoundMode = RoundMode.CAST_ROUND) -> None:
    """Fused mul + cast: ``out = cast(lhs * rhs, target_type)``"""


# --- B5. Compare / select ---


@_api_decl
def eq(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None:
    """Element-wise equal: ``out = (lhs == rhs)``

    Supports both tile-tile and tile-scalar operations:
        - ``eq(out, tile_a, tile_b)`` -> tile-tile
        - ``eq(out, tile_a, scalar)`` -> tile-scalar
    """


@_api_decl
def ne(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None:
    """Element-wise not equal: ``out = (lhs != rhs)``

    Supports both tile-tile and tile-scalar operations:
        - ``ne(out, tile_a, tile_b)`` -> tile-tile
        - ``ne(out, tile_a, scalar)`` -> tile-scalar
    """


@_api_decl
def lt(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None:
    """Element-wise less than: ``out = (lhs < rhs)``

    Supports both tile-tile and tile-scalar operations:
        - ``lt(out, tile_a, tile_b)`` -> tile-tile
        - ``lt(out, tile_a, scalar)`` -> tile-scalar
    """


@_api_decl
def le(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None:
    """Element-wise less or equal: ``out = (lhs <= rhs)``

    Supports both tile-tile and tile-scalar operations:
        - ``le(out, tile_a, tile_b)`` -> tile-tile
        - ``le(out, tile_a, scalar)`` -> tile-scalar
    """


@_api_decl
def gt(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None:
    """Element-wise greater than: ``out = (lhs > rhs)``

    Supports both tile-tile and tile-scalar operations:
        - ``gt(out, tile_a, tile_b)`` -> tile-tile
        - ``gt(out, tile_a, scalar)`` -> tile-scalar
    """


@_api_decl
def ge(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None:
    """Element-wise greater or equal: ``out = (lhs >= rhs)``

    Supports both tile-tile and tile-scalar operations:
        - ``ge(out, tile_a, tile_b)`` -> tile-tile
        - ``ge(out, tile_a, scalar)`` -> tile-scalar
    """


@_api_decl
def select(out: Tile, mask: Tile, lhs: Tile, rhs: Union[Tile, Scalar], tmp: Tile) -> None:
    """Select by mask: ``out[i] = lhs[i] if mask[i] else rhs[i]``

    Supports both tile-tile and tile-scalar operations:
        - ``select(out, mask, tile_a, tile_b, tmp)`` -> tile-tile
        - ``select(out, mask, tile_a, scalar, tmp)`` -> tile-scalar

    Args:
        out: Destination Tile
        mask: Mask Tile (from ``eq``/``gt``/...)
        lhs: Tile selected when mask is true
        rhs: Tile or Scalar selected when mask is false
        tmp: Workspace Tile
    """


# --- B6. Fused ops ---


@_api_decl
def add_relu(out: Tile, lhs: Tile, rhs: Tile) -> None:
    """Fused add + ReLU: ``out = relu(lhs + rhs)``"""


@_api_decl
def sub_relu(out: Tile, lhs: Tile, rhs: Tile) -> None:
    """Fused sub + ReLU: ``out = relu(lhs - rhs)``"""


@_api_decl
def addc(out: Tile, a: Tile, b: Tile, c: Tile) -> None:
    """Three-operand add: ``out = a + b + c``"""


@_api_decl
def mul_add_dst(out: Tile, lhs: Tile, rhs: Tile) -> None:
    """Fused multiply-add into out: ``out = lhs * rhs + out``"""


@_api_decl
def fused_mul_add(out: Tile, lhs: Tile, rhs: Tile) -> None:
    """Fused multiply-add in-place: ``out = lhs * out + rhs``"""


@_api_decl
def fused_mul_add_relu(out: Tile, a: Tile, b: Tile) -> None:
    """Fused multiply-add + ReLU in-place: ``out = relu(out * a + b)``"""


@_api_decl
def axpy(out: Tile, src: Tile, alpha: Scalar) -> None:
    """AXPY: ``out[i] = alpha * src[i] + out[i]``

    Args:
        out: Destination Tile (also accumulates)
        src: Source Tile
        alpha: Scalar multiplier
    """


@_api_decl
def partadd(out: Tile, src0: Tile, src1: Tile) -> None:
    """Partial addition: ``out = src0 + src1`` (only src1's valid region)."""


@_api_decl
def partmax(out: Tile, src0: Tile, src1: Tile) -> None:
    """Partial maximum over the valid regions of two Tiles."""


@_api_decl
def partmin(out: Tile, src0: Tile, src1: Tile) -> None:
    """Partial minimum over the valid regions of two Tiles."""


@_api_decl
def partmul(out: Tile, src0: Tile, src1: Tile) -> None:
    """Partial multiplication over the valid regions of two Tiles."""


# --- B7. Matrix ops ---


@_api_decl
def matmul(
    dst_tile: Tile, lhs_tile: Tile, rhs_tile: Tile, bias_tile: Tile = None, *, phase: Optional[AccPhase] = None
) -> None:
    """Matrix multiply: ``dst = lhs @ rhs`` (L0A × L0B → L0C).

    When ``bias_tile`` is provided, performs fused matmul + bias:
    ``dst = lhs @ rhs + bias`` (bias row-broadcast [1,N], added in fixpipe).

    Args:
        dst_tile: Accumulator Tile (L0C, output)
        lhs_tile: Left matrix Tile (L0A)
        rhs_tile: Right matrix Tile (L0B)
        bias_tile: Optional — Bias Tile (L0B Bias area, shape [1, N], row-broadcast)
        phase: Optional — ``pl.AccPhase.Partial`` or ``pl.AccPhase.Final``
    """


@_api_decl
def matmul_acc(
    dst_tile: Tile, acc_tile: Tile, lhs_tile: Tile, rhs_tile: Tile, *, phase: Optional[AccPhase] = None
) -> None:
    """Accumulating matrix multiply: ``dst = acc + lhs @ rhs`` (K-dim block accumulation).

    Args:
        dst_tile: Destination Tile (L0C, output)
        acc_tile: Accumulator Tile (existing value to add to)
        lhs_tile: Left matrix Tile (L0A)
        rhs_tile: Right matrix Tile (L0B)
        phase: Optional — ``pl.AccPhase.Partial`` or ``pl.AccPhase.Final``
    """


@_api_decl
def matmul_mx(
    dst_tile: Tile, lhs_tile: Tile, rhs_tile: Tile, scale_a: Tile, scale_b: Tile,
    *, phase: Optional[AccPhase] = None
) -> None:
    """MX matmul with per-group E8M0 scale: ``dst = lhs @ rhs``

    Left/Right tiles use FP8/FP4 dtype; scale tiles use E8M0 in ScaleLeft/ScaleRight.
    Hardware mad_mx reads scale implicitly via SFractal layout.

    Args:
        dst_tile: Accumulator Tile (L0C, output, FP32)
        lhs_tile: Left matrix Tile (L0A, FP8/FP4)
        rhs_tile: Right matrix Tile (L0B, FP8/FP4)
        scale_a: Left scale Tile (ScaleLeft, E8M0)
        scale_b: Right scale Tile (ScaleRight, E8M0)
        phase: Optional — ``pl.AccPhase.Partial`` or ``pl.AccPhase.Final``
    """


@_api_decl
def matmul_mx_acc(
    dst_tile: Tile, acc_tile: Tile, lhs_tile: Tile, rhs_tile: Tile, scale_a: Tile, scale_b: Tile,
    *, phase: Optional[AccPhase] = None
) -> None:
    """MX matmul with accumulation: ``dst = acc + lhs @ rhs``

    Args:
        dst_tile: Destination Tile (L0C, output, FP32)
        acc_tile: Accumulator Tile (existing value to add to)
        lhs_tile: Left matrix Tile (L0A, FP8/FP4)
        rhs_tile: Right matrix Tile (L0B, FP8/FP4)
        scale_a: Left scale Tile (ScaleLeft, E8M0)
        scale_b: Right scale Tile (ScaleRight, E8M0)
        phase: Optional — ``pl.AccPhase.Partial`` or ``pl.AccPhase.Final``
    """


@_api_decl
def transpose(out: Tile, src: Tile) -> None:
    """Transpose Tile by swapping the last two dimensions.

    Args:
        out: Destination Tile
        src: Source Tile
    """


# --- B8. Reductions / expands ---
#
# Unified reduce / expand interfaces.  ``dim=0`` reduces along the last axis
# (row-wise), ``dim=1`` reduces along the first axis (column-wise).
#
# tmp Tile constraints (apply to all reduce ops with ``tmp`` parameter):
# - dtype: must match ``src`` Tile dtype
# - shape: must match ``src`` Tile shape (1:1 workspace, not reduced size)
# - memory: must be in UB (Vec) memory space


@_api_decl
def sum(out: Tile, src: Tile, tmp: Tile, *, dim: int = 0) -> None:
    """Sum reduction along the specified dimension.

    Args:
        out: Destination Tile
        src: Source Tile
        tmp: Workspace Tile (required by hardware)
        dim: Reduction dimension — 0=row (last axis), 1=column (first axis)

    Example::

        tmp_type = pl.TileType(shape=src.type.shape, dtype=pl.DT_FP32,
                               target_memory=pl.MemorySpace.Vec)
        tmp = pl.make_tile(tmp_type, addr=0x1000)
        pl.sum(out, src, tmp, dim=0)   # row-wise sum, out shape = [src.shape[0], 1]
    """


@_api_decl
def argmax(out: Tile, src: Tile, tmp: Tile, *, dim: int = 0) -> None:
    """Argmax reduction along the specified dimension.

    Args:
        out: Destination Tile
        src: Source Tile
        tmp: Workspace Tile (required by hardware)
        dim: Reduction dimension — 0=row (last axis), 1=column (first axis)
    """


@_api_decl
def argmin(out: Tile, src: Tile, tmp: Tile, *, dim: int = 0) -> None:
    """Argmin reduction along the specified dimension.

    Args:
        out: Destination Tile
        src: Source Tile
        tmp: Workspace Tile (required by hardware)
        dim: Reduction dimension — 0=row (last axis), 1=column (first axis)
    """


@_api_decl
def expand_max(out: Tile, src: Tile, scalar: Tile, *, dim: int = 0) -> None:
    """Max expand (broadcast reduction result back to full shape).

    Args:
        out: Destination Tile
        src: Source Tile (reduction result)
        scalar: Scalar Tile for the expand operation
        dim: Expand dimension — 0=row, 1=column
    """


@_api_decl
def expand_min(out: Tile, src: Tile, scalar: Tile, *, dim: int = 0) -> None:
    """Min expand.

    Args:
        out: Destination Tile
        src: Source Tile (reduction result)
        scalar: Scalar Tile for the expand operation
        dim: Expand dimension — 0=row, 1=column
    """


@_api_decl
def expand_mul(out: Tile, src: Tile, scalar: Tile, *, dim: int = 0) -> None:
    """Multiply expand.

    Args:
        out: Destination Tile
        src: Source Tile (reduction result)
        scalar: Scalar Tile for the expand operation
        dim: Expand dimension — 0=row, 1=column
    """


@_api_decl
def expand_sub(out: Tile, src: Tile, scalar: Tile, *, dim: int = 0) -> None:
    """Subtract expand.

    Args:
        out: Destination Tile
        src: Source Tile (reduction result)
        scalar: Scalar Tile for the expand operation
        dim: Expand dimension — 0=row, 1=column
    """


@_api_decl
def expand_div(out: Tile, src: Tile, scalar: Tile, *, dim: int = 0) -> None:
    """Divide expand.

    Args:
        out: Destination Tile
        src: Source Tile (reduction result)
        scalar: Scalar Tile for the expand operation
        dim: Expand dimension — 0=row, 1=column
    """


# --- B9. Gather / scatter / sort ---


@_api_decl
def gather(out: Tile, src: Tile, idx: Tile, tmp: Tile, *, cmp_mode: int = 0, offset: int = 0) -> None:
    """Gather elements by index.

    Args:
        out: Destination Tile
        src: Source Tile
        idx: Index Tile
        tmp: Workspace Tile
        cmp_mode: Comparison mode (default 0)
        offset: Index offset
    """


@_api_decl
def gatherb(out: Tile, src: Tile, offsets: Tile) -> None:
    """Gather elements by 32-byte block byte offset.

    Args:
        out: Destination Tile
        src: Source Tile
        offsets: Byte offset Tile
    """


@_api_decl
def gathermask(out: Tile, src: Tile, *, pattern_mode: int) -> None:
    """Extract columns by bit-pattern mask.

    Args:
        out: Destination Tile
        src: Source Tile
        pattern_mode: Bit-pattern extraction mode
    """


@_api_decl
def scatter(out: Tile, src: Tile, idx: Tile) -> None:
    """Scatter elements by index.

    Args:
        out: Destination Tile
        src: Source Tile
        idx: Index Tile
    """


@_api_decl
def mrgsort(dst: Tile, src: Tile, *, block_len: int) -> None:
    """Merge sort.

    Args:
        dst: Sorted output Tile
        src: Source Tile
        block_len: Sort block length
    """


@_api_decl
def mrgsort2(src0: Tile, src1: Tile, dst: Tile, tmp: Tile, *args, exhausted: bool = False) -> None:
    """Two-way (or multi-way) merge sort.

    Args:
        src0: First source Tile (val-idx pairs)
        src1: Second source Tile (val-idx pairs)
        dst: Destination Tile
        tmp: Workspace Tile
        *args: Optional additional source Tiles for multi-way merge
        exhausted: Whether any source is already exhausted
    """


@_api_decl
def sort32(dst: Tile, src: Tile, idx: Tile, tmp: Optional[Tile] = None) -> None:
    """Sort 32 elements with index tracking.

    Args:
        dst: Destination Tile for sorted values
        src: Source Tile (32 elements)
        idx: Index Tile for tracking original positions
        tmp: Optional workspace Tile (for tail-block handling)
    """


@_api_decl
def histogram(dst: Tile, src: Tile, idx: Tile, *, is_msb: bool) -> None:
    """Histogram accumulation for radix sort preprocessing.

    Counts byte-value frequencies in *src* and writes per-row bin counts to *dst*.

    Args:
        dst: Destination Tile (``pl.DT_UINT32``, cols >= 256)
        src: Source Tile (``pl.DT_UINT16``)
        idx: Index Tile (``pl.DT_UINT8``, DN layout); used for filtering when ``is_msb=False``
        is_msb: ``True`` counts high byte (bits 15-8); ``False`` counts low byte (bits 7-0)
            filtered by rows where the high byte matches ``idx``
    """


# --- B10. Quantization / index / misc ---


@_api_decl
def quant(out: Tile, src: Tile, scale: Tile, *, mode: QuantMode = QuantMode.SYM, offset: Optional[Tile] = None) -> None:
    """Quantize Tile (high-precision → low-precision integer).

    Args:
        out: Destination Tile (quantized output)
        src: Source Tile (float input)
        scale: Scale Tile
        mode: ``pl.QuantMode.SYM`` (symmetric) or ``pl.QuantMode.ASYM`` (asymmetric, requires *offset*)
        offset: Offset Tile (required for ``mode=pl.QuantMode.ASYM``)
    """


@_api_decl
def dequant(out: Tile, src: Tile, scale: Tile, offset: Tile) -> None:
    """Dequantize Tile (low-precision integer → high-precision).

    Args:
        out: Destination Tile (float output)
        src: Source Tile (quantized input)
        scale: Scale Tile
        offset: Offset Tile
    """


@_api_decl
def getval(container: "Tile | Tensor", offset: int) -> Scalar:
    """Read a scalar value from a Tile or Tensor at the given linear offset."""


@_api_decl
def setval(container: "Tile | Tensor", offset: int, value: Scalar) -> None:
    """Write a scalar value into a Tile or Tensor at the given linear offset."""


@_api_decl
def set_validshape(tile: "Tile | TileGroup", shape: List[int]) -> None:
    """Set the valid shape of a Tile or tile_group (for partial-tile / tail-block operations).

    When a tile_group is passed, valid_shape is set on all tiles in the group.
    """

@_api_decl
def reinterpret(tile: "Tile | TileGroup", *, dtype: DType = None, shape: List[int] = None,
                layout: Optional[TensorLayout] = None) -> "Tile | TileGroup":
    """Reinterpret a Tile or tile_group's dtype/shape/layout metadata without data movement.

    Returns a new handle over the same on-chip buffer: the new tile is
    re-allocated at the ORIGINAL address and size (same-buffer semantics hold),
    with memory space, fractal, pad, compact and mutex mapping inherited; only
    the given properties are overridden. valid_shape is NOT inherited (the new
    handle starts with a dynamic valid shape, equivalent to -1; set it with
    ``pl.set_validshape``). The original handle is unchanged.

    ``reinterpret`` is a compile-time view declaration, not a conversion — no
    load/move/store/cast instruction is generated. The caller must guarantee
    that the physical data actually matches the new declaration (e.g. declaring
    ``layout=pl.ZN`` on data that really is ZN-arranged).

    Constraints:
        - at least one of ``dtype`` / ``shape`` / ``layout`` must be given
        - when ``dtype`` changes, ``shape`` must be given as well (the element
          count is re-stated explicitly)
        - ``shape`` must be compile-time constants; use ``pl.set_validshape``
          for runtime windows
        - the new footprint (shape x dtype bytes) must not exceed the original
          buffer size, and the base address must be aligned to the new element
          size (both checked at parse time)

    Example::

        t2 = pl.reinterpret(tile_a, shape=[64, 64], dtype=pl.DT_BF16)  # same buffer as BF16 (dtype change needs shape)
        g2 = pl.reinterpret(group_a, shape=[128, 32])        # whole group, mutex kept
    """


@_api_decl
def set_mask_count() -> None:
    """Switch mask to counting mode."""


@_api_decl
def set_mask_norm() -> None:
    """Switch mask to per-bit normalization mode."""


@_api_decl
def set_vec_mask(mask_high: int, mask_low: int) -> None:
    """Explicitly set the 128-bit vector mask from two 64-bit integers."""


@_api_decl
def reset_mask() -> None:
    """Reset the mask to all-ones (no masking)."""


@_api_decl
def fill_index(out: Tile, start: Scalar) -> None:
    """Fill target tile with sequential indices starting from *start*."""


# ===================================================================
# Section E: Debug ops
# ===================================================================


@_api_decl
def pto_assert(condition: bool, format_str: Optional[str] = None, *args, loc: bool = False) -> None:
    """Runtime assert: abort if condition is false, optionally print error message.

    Args:
        condition: Scalar boolean condition (dtype must be BOOL)
        format_str: Optional compile-time constant format string (printf-style)
        *args: Scalar values to print in the format string
        loc: Show source location in the error message
    """


@_api_decl
def printf(format_str: str, *args, loc: bool = False) -> None:
    """Print scalar values using a compile-time format string (printf-style).

    Supported conversions: ``%d``/``%i`` (signed int), ``%u`` (unsigned int),
    ``%x`` (hex), ``%f`` (float, FP32 only).

    Args:
        format_str: Compile-time constant format string
        *args: Scalar values (int, float, or ``pl.Scalar``)
        loc: Show source location
    """


@_api_decl
def dump_data(
    data: Union[Tensor, Tile],
    offsets: Optional[List[int]] = None,
    shapes: Optional[List[int]] = None,
    *,
    workspace: Optional[Tensor] = None,
    loc: bool = False,
    flag: Optional[str] = None,
) -> None:
    """Print Tensor or Tile contents for debugging.

    Automatically dispatches based on input type:
    - TensorType (GM Tensor): prints GM tensor data
    - TileType (on-chip Tile): prints tile data (Acc tiles require workspace)

    Args:
        data: Tensor or Tile to dump
        offsets: Optional window start offsets (per dimension); must pair with *shapes*
        shapes: Optional window shape (per dimension); must pair with *offsets*
        workspace: GM Tensor used as temporary space (only valid for Acc Tile inputs,
                   size >= tile_numel * sizeof(element_type))
        loc: Show source location
        flag: Optional compile-time string label printed as a standalone marker line
              before the dump output, to distinguish multiple dump sites
    """


@_api_decl
def trap() -> None:
    """Insert a trap instruction to unconditionally abort execution."""


# ===================================================================
# Section F: Scalar ops
# ===================================================================


@_api_decl
def min(lhs: Scalar, rhs: Scalar) -> Scalar:
    """Return the minimum of two scalars.

    Scalar-only operation for loop-bound calculations etc.
    For tile element-wise minimum, use ``pl.minimum``.

    Args:
        lhs: Left operand (scalar)
        rhs: Right operand (scalar)

    Returns:
        Scalar result.
    """


@_api_decl
def max(lhs: Scalar, rhs: Scalar) -> Scalar:
    """Return the maximum of two scalars.

    Scalar-only operation for loop-bound calculations etc.
    For tile element-wise maximum, use ``pl.maximum``.

    Args:
        lhs: Left operand (scalar)
        rhs: Right operand (scalar)

    Returns:
        Scalar result.
    """


# ===================================================================
# Section G: Tile min/max (element-wise + reduce overload)
#
# ``maximum`` / ``minimum`` are overloaded:
#   - Element-wise (no ``dim``): ``out = max(lhs, rhs)``
#       - ``maximum(out, tile_a, tile_b)``  -> tile-tile
#       - ``maximum(out, tile_a, scalar)``  -> tile-scalar
#   - Reduce (with ``dim``): reduction along the specified dimension
#       - ``maximum(out, src, tmp, dim=0)`` -> row-wise max (last axis)
#       - ``maximum(out, src, tmp, dim=1)`` -> column-wise max (first axis)
# ===================================================================


@_api_decl
def minimum(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar], *, dim: Optional[int] = None) -> None:
    """Element-wise minimum or dimension-wise min reduction.

    Without ``dim`` (element-wise): ``out = min(lhs, rhs)``
        - ``minimum(out, tile_a, tile_b)`` -> tile-tile
        - ``minimum(out, tile_a, scalar)`` -> tile-scalar

    With ``dim`` (reduce): min reduction along the specified dimension.
        - ``minimum(out, src, tmp, dim=0)`` -> row-wise min (last axis)
        - ``minimum(out, src, tmp, dim=1)`` -> column-wise min (first axis)

    Args:
        out: Destination Tile
        lhs: Source Tile (or left operand in element-wise mode)
        rhs: Right operand — Tile/Scalar in element-wise mode, or workspace
             Tile in reduce mode
        dim: Reduction dimension — None=element-wise, 0=row, 1=column
    """


@_api_decl
def maximum(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar], *, dim: Optional[int] = None) -> None:
    """Element-wise maximum or dimension-wise max reduction.

    Without ``dim`` (element-wise): ``out = max(lhs, rhs)``
        - ``maximum(out, tile_a, tile_b)`` -> tile-tile
        - ``maximum(out, tile_a, scalar)`` -> tile-scalar

    With ``dim`` (reduce): max reduction along the specified dimension.
        - ``maximum(out, src, tmp, dim=0)`` -> row-wise max (last axis)
        - ``maximum(out, src, tmp, dim=1)`` -> column-wise max (first axis)

    Args:
        out: Destination Tile
        lhs: Source Tile (or left operand in element-wise mode)
        rhs: Right operand — Tile/Scalar in element-wise mode, or workspace
             Tile in reduce mode
        dim: Reduction dimension — None=element-wise, 0=row, 1=column
    """


@_api_decl
def const(value: Union[int, float], dtype: DType) -> Scalar:
    """Create a typed compile-time constant.

    Args:
        value: Numeric literal (int or float)
        dtype: Data type, e.g. ``pl.DT_INT32``, ``pl.DT_FP16``
    """


# ===================================================================
# Section G: Control flow
# ===================================================================


def range(start: int, stop: Optional[int] = None, step: int = 1):
    """Loop iterator for ``for`` loops.

    Usage::

        for i in pl.range(10):          # 0..9
        for i in pl.range(0, 10, 2):    # 0, 2, 4, 6, 8
    """
    if stop is None:
        start, stop = 0, start
    return builtins.range(start, stop, step)


@contextmanager
def section_vector():
    """Context manager for a Vector-pipe section scope."""
    yield


@contextmanager
def section_cube():
    """Context manager for a Cube-pipe section scope."""
    yield


# ===================================================================
# Section H: Utility / system-level ops
# ===================================================================


@_api_decl
def get_block_idx() -> int:
    """Get the current block (AI Core) index."""


@_api_decl
def get_subblock_idx() -> int:
    """Get the current sub-block index (0 or 1)."""


@_api_decl
def get_block_num() -> int:
    """Get the total number of blocks."""


@_api_decl
def get_subblock_num() -> int:
    """Get the sub-block count per AI Core (task ration).

    Returns 1 on AIC binaries, get_subblockdim() on AIV binaries.
    Matches AscendC GetTaskRation().
    """


@_api_decl
def get_spr() -> int:
    """Read a special purpose register value (get_ar instruction).

    Currently only the AR register is supported. The AR register stores
    the total byte count of valid elements produced by Squeeze.

    ``get_ar()`` is an ``__aicore__`` instruction and cannot be used inside
    ``@pl.vector_function``. Call this in the ``@pl.jit`` kernel body.

    Returns:
        int64_t scalar value from the SPR
    """


@_api_decl
def set_saturation_flag(mode: SaturationFlagMode, enable: bool) -> None:
    """Set the saturation flag in the CTRL special purpose register.

    Controls the global saturation mode for Cast (vcvt) and other
    vector compute instructions. Must be called outside
    ``@pl.vector_function`` (in the ``@pl.jit`` kernel body),
    before the ``pl.section_vector()`` block that uses ``vf.astype``
    with the corresponding saturation mode.

    Args:
        mode: Saturation mode category, one of:

            - ``pl.SaturationFlagMode.FLOAT`` — float compute/convert (CTRL bit 48)
            - ``pl.SaturationFlagMode.FLOAT8`` — float8 compute (CTRL bit 50)
            - ``pl.SaturationFlagMode.INT`` — int compute (CTRL bit 53)
            - ``pl.SaturationFlagMode.CAST`` — float→int / int→int convert (CTRL bit 59)

        enable: ``True`` to enable saturation (clamp to target type's
            min/max), ``False`` to disable (truncate).
    """


@_api_decl
def get_saturation_flag(mode: SaturationFlagMode) -> bool:
    """Read the saturation flag from the CTRL special purpose register.

    Returns the current saturation state for the given mode category.
    Must be called outside ``@pl.vector_function``.

    Args:
        mode: Saturation mode category (same as ``set_saturation_flag``).

    Returns:
        ``True`` if saturation is enabled, ``False`` otherwise.
    """


@_api_decl
def set_ctrl_spr(start_bit: int, end_bit: int, value: int) -> None:
    """Set a bit range in the CTRL special purpose register.

    Writes ``value`` into the CTRL register bits ``[start_bit, end_bit]``,
    preserving all other bits. This is the low-level counterpart to
    ``set_saturation_flag`` — use it when direct CTRL bit manipulation
    is needed (e.g., setting the global override bit CTRL[60]).

    Must be called outside ``@pl.vector_function``.

    Args:
        start_bit: Start bit index (0-63), compile-time constant.
        end_bit: End bit index (0-63), compile-time constant.
            Writable bits on A5: 6-10, 45, 48, 50, 53, 59, 60.
        value: Value to write into the bit range.
    """


@_api_decl
def get_ctrl_spr(start_bit: int, end_bit: int) -> int:
    """Read a bit range from the CTRL special purpose register.

    Returns the value of CTRL register bits ``[start_bit, end_bit]``.
    Must be called outside ``@pl.vector_function``.

    Args:
        start_bit: Start bit index (0-63), compile-time constant.
        end_bit: End bit index (0-63), compile-time constant.

    Returns:
        int64_t value of the extracted bit range.
    """


@_api_decl
def reset_ctrl_spr(start_bit: int, end_bit: int) -> None:
    """Reset a bit range in the CTRL register to default values.

    Restores CTRL register bits ``[start_bit, end_bit]`` to their
    hardware default (CTRL default = 0x1000000000000008).
    Must be called outside ``@pl.vector_function``.

    Args:
        start_bit: Start bit index (0-63), compile-time constant.
        end_bit: End bit index (0-63), compile-time constant.
            Writable bits on A5: 6-10, 45, 48, 50, 53, 59, 60.
    """


@_api_decl
def make_tile(
    tile_type: Any,
    *,
    addr: int,
    size: Optional[int] = None,
) -> Tile:
    """Place one Tile at a fixed address in the memory space its type names.

    The Tile's shape, dtype, memory space and layout all come from ``tile_type``;
    this call only binds it to an address range. For several Tiles of one type
    rotating through a ping-pong buffer, use ``pl.make_tile_group``.

    ``tile_type`` is the only positional argument; everything else is a keyword.
    ``addr`` and ``size`` describe where the Tile sits rather than what it holds,
    and two bare integers in a row read the same whichever order they are in — so
    they are named at the call site, as ``pl.make_tile_group(type=, addrs=)`` is.

    Args:
        tile_type: ``pl.TileType`` descriptor — the only accepted spelling of the
            Tile's shape/dtype/memory space/layout
        addr: Required byte offset within the memory space. Fixed at parse time,
            and aligned to that space (32B Vec/Mat, 512B Left/Right, 64B Acc)
        size: Optional byte span to reserve; defaults to the ``tile_type``
            footprint (elements x dtype bytes). Pass it only to reserve more, as
            an NZ/ZN tile rounded up to whole fractals needs
    Example::

        tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        tile_a = pl.make_tile(tt, addr=0x0000)           # size derived: 64 * 128 * 2
        tile_b = pl.make_tile(tt, addr=0x4000, size=16384)
    """


@_api_decl
def make_tile_group(*, type: Any, addrs: int | list,
    mutex_ids: list[int | list[int] | tuple[int, ...]] | tuple | None = None,
    depth: int | None = None,
) -> Any:
    """Create a rotating Tile group with optional mutex metadata.

    The handle supports ``group.next()``, ``group.current()``,
    ``group.previous()``, and ``group[i]``. When ``mutex_ids`` is provided,
    ``auto_mutex`` synchronizes accesses automatically.

    Args:
        type: ``pl.TileType`` descriptor
        addrs: Base address for contiguous Tiles, or one address per Tile
        mutex_ids: Optional mutex IDs for synchronization. Each Tile may use
            one int or a non-empty list/tuple. Every Tile must use the same
            mutex ID count, with no duplicates for one Tile
        depth: Number of Tiles. Required when ``mutex_ids`` is None or empty;
            otherwise inferred from ``len(mutex_ids)``
    """
