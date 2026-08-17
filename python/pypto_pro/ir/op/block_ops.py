# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""IR construction helpers for block ops.

These functions accept IR ``Expr`` objects and return ``Call`` expressions.
Parse handlers registered via ``@op_impl`` dispatch through ``_OP_REGISTRY``
in ``call_parser.parse_op_call``.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass, fields
import enum
import os
import struct
from typing import Any, Optional

from pypto.pypto_impl import ir as _ir_core
from pypto.pypto_impl.ir import (
    AccPhase,
    AccToVecMode,
    AtomicType,
    CmpMode,
    ConstInt,
    DataType,
    Expr,
    MemorySpace,
    QuantMode,
    ReluPreMode,
    RoundMode,
    Span,
    STPhase,
    TensorLayout,
    TilePad,
)
from pypto.pypto_impl.ir import TileType as _IRTileType  # IR-level TileType (C++ binding);
from pypto_pro.ir._utils import _is_int, _normalize_expr, _to_make_tuple

# NOTE: a DSL-descriptor dataclass named ``TileType`` is defined later in this
# module and shadows this import, so use ``_IRTileType`` for isinstance checks.
from ._op_registry import OpSpec, op_impl, register_table


def _span() -> Span:
    return Span.unknown()


_BLOCK_OP_NAMESPACE = "block"


def block_ir_op(op_name: str) -> str:
    """Return the IR name for the explicit-output block DSL ops."""
    return f"{_BLOCK_OP_NAMESPACE}.{op_name}"


class FillPadMode(enum.Enum):
    """Fill-pad operating mode (lowered to different IR op names)."""

    NORMAL = 0
    EXPAND = 1
    INPLACE = 2


def _compute_absolute_offsets(
    tile_offsets: _ir_core.MakeTuple,
    tile_shape: list,
    tile_dims: list[int] | None,
    span: Span,
    *,
    access_size: Any | None = None,
) -> _ir_core.MakeTuple:
    """Convert tile-relative offsets to absolute tensor offsets."""
    offsets = []
    for i, tile_offset in enumerate(tile_offsets.elements):
        if access_size is not None:
            shape = access_size
        elif tile_dims is not None and i in tile_dims:
            shape = tile_shape[tile_dims.index(i)]
        else:
            offsets.append(tile_offset)
            continue

        if isinstance(tile_offset, ConstInt) and isinstance(shape, ConstInt):
            offset = ConstInt(tile_offset.value * shape.value, DataType.INT64, span)
        else:
            offset = _ir_core.Mul(tile_offset, shape, DataType.INT64, span)
        offsets.append(offset)
    return _ir_core.MakeTuple(offsets, span)


def _const_int_attr(value: int | Expr, name: str) -> int:
    if isinstance(value, int):
        return value
    const_value = getattr(value, "value", None)
    if isinstance(const_value, int):
        return const_value
    raise TypeError(f"block op requires constant integer {name}")


def _validate_offset_bounds(
    op_name: str,
    src_shape: Sequence[Expr],
    offsets: Sequence[int | Expr],
) -> None:
    """Validate that offset does not exceed src_shape bounds at compile time.

    Only checks dimensions where both offset and src_shape are compile-time constants.
    Note: offset + access_shape exceeding src_shape is allowed (valid_shape handles tail blocks).
    """
    for i, (off, src) in enumerate(zip(offsets, src_shape)):
        off_val = off if isinstance(off, int) else getattr(off, "value", None)
        src_val = getattr(src, "value", None)
        if isinstance(off_val, int) and isinstance(src_val, int):
            if off_val >= src_val:
                raise ValueError(f"{op_name}: offset[{i}] ({off_val}) exceeds source shape[{i}] ({src_val})")


def _validate_nz_transfer_axes(
    tensor_type: _ir_core.TensorType,
    tile_dims: list[int] | None,
    op_name: str,
) -> None:
    """Require a GM NZ transfer's 2D Tile to map to the Tensor's final M/N axes."""
    tensor_view = getattr(tensor_type, "tensor_view", None)
    if tensor_view is None or tensor_view.layout != TensorLayout.NZ:
        return

    tensor_ndim = len(tensor_type.shape)
    expected_dims = list(range(tensor_ndim - 2, tensor_ndim))
    if tile_dims != expected_dims:
        raise ValueError(f"{op_name}: NZ transfer only supports the last two tensor axes")


def _ir_binary_cast(
    op_name: str,
    out: Expr,
    lhs: Expr,
    rhs: Expr,
    target_type: int | DataType,
    *,
    span: Span | None = None,
    mode: RoundMode = RoundMode.CAST_ROUND,
) -> Expr:
    return _ir_core.create_op_call(
        block_ir_op(op_name),
        [out, lhs, rhs],
        {"target_type": target_type, "mode": mode},
        span or _span(),
    )


def _ir_load(
    out: Expr,
    tensor: Expr,
    offsets: Sequence[int | Expr] | _ir_core.MakeTuple,
    *,
    span: Span | None = None,
    order: list[int] | None = None,
) -> Expr:
    op_name = "load"
    actual_span = span or _span()
    offsets_tuple = _to_make_tuple(offsets, actual_span)

    tensor_ndim = len(tensor.type.shape)
    tile_ndim = len(out.type.shape)
    tile_dims, is_transpose = _resolve_order(order, tensor_ndim, tile_ndim, op_name)

    tensor_ndim, tile_shape, tile_dims, _ = _validate_load_operands(
        out, tensor, offsets_tuple, tile_dims, op_name
    )
    _validate_nz_transfer_axes(tensor.type, tile_dims, op_name)

    kwargs: dict[str, Any] = {}
    if is_transpose:
        kwargs["is_transpose"] = is_transpose
    # Specialized lowerings may use a different fallback, so preserve every
    # explicitly supplied order, including the ordinary last-two-axes order.
    if order is not None and tile_dims is not None:
        kwargs["tile_dims"] = tile_dims
    if isinstance(offsets, _ir_core.MakeTuple):
        _validate_offset_bounds("load", tensor.type.shape, offsets.elements)
    else:
        _validate_offset_bounds("load", tensor.type.shape, offsets)
    return _ir_core.create_op_call(block_ir_op(op_name), [out, tensor, offsets_tuple], kwargs, actual_span)


def _ir_load_tile(
    out: Expr,
    tensor: Expr,
    tile_offsets: Sequence[int | Expr] | _ir_core.MakeTuple,
    *,
    span: Span | None = None,
    order: list[int] | None = None,
) -> Expr:
    op_name = "load_tile"
    actual_span = span or _span()
    offsets_tuple = _to_make_tuple(tile_offsets, actual_span)

    tensor_ndim = len(tensor.type.shape)
    tile_ndim = len(out.type.shape)
    tile_dims, is_transpose = _resolve_order(order, tensor_ndim, tile_ndim, op_name)

    tensor_ndim, tile_shape, tile_dims, access_size = _validate_load_operands(
        out,
        tensor,
        offsets_tuple,
        tile_dims,
        op_name,
        use_tile_absolute=True,
    )
    _validate_nz_transfer_axes(tensor.type, tile_dims, op_name)

    abs_offsets = _compute_absolute_offsets(
        offsets_tuple,
        tile_shape,
        tile_dims,
        actual_span,
        access_size=access_size,
    )
    # Validate offset bounds at Python frontend level
    _validate_offset_bounds("load_tile", tensor.type.shape, abs_offsets.elements)
    kwargs: dict[str, Any] = {}
    if is_transpose:
        kwargs["is_transpose"] = is_transpose
    if order is not None and tile_dims is not None:
        kwargs["tile_dims"] = tile_dims
    return _ir_core.create_op_call(block_ir_op("load"), [out, tensor, abs_offsets], kwargs, actual_span)


def _encode_deq_scalar(scale: float) -> int:
    scale_bits = struct.unpack("!I", struct.pack("!f", scale))[0]
    return scale_bits


def _resolve_scale_param(
    scale: Any,
    span: Span,
) -> tuple[Optional["int | Expr"], Optional[Expr]]:
    if scale is None:
        return None, None
    if isinstance(scale, (int, float)):
        encoded = _encode_deq_scalar(float(scale))
        return encoded, None
    if isinstance(scale, Expr):
        scale_type = getattr(scale, "type", None)
        if isinstance(scale_type, _ir_core.TileType):
            # User-prepared Scaling tile (per-channel): already validated by
            # _auto_alloc_scaling_tile_hook before the builder runs; resolve it
            # here to the store_fp/move_fp operand.
            return None, scale
        if not isinstance(scale_type, _ir_core.ScalarType):
            raise TypeError(
                f"scale Expr must be a runtime scalar, got {type(scale_type).__name__}; "
                f"per-channel quantization requires a user-prepared Scaling Tile"
            )
        # Only FP32 (auto bitcast in codegen) and INT32/INT64 (user passes the
        # pre-encoded float32 bit pattern) are supported as runtime scalars:
        # FP16/BF16 would be numerically converted to uint64 (wrong bits, and
        # fp_to_uint crashes the bisheng backend), and narrower/unsigned ints
        # risk sign-extension or truncation of the bit pattern.
        if scale_type.dtype not in (DataType.FP32, DataType.INT32, DataType.INT64):
            raise TypeError(
                f"scale runtime scalar dtype {scale_type.dtype} is not supported — pass "
                f"an FP32 scalar (auto-reinterpreted as its IEEE-754 bit pattern) or an "
                f"INT32/INT64 scalar carrying the pre-encoded float32 bit pattern "
                f"(struct.pack(\"!f\", scale))"
            )
        return scale, None
    raise TypeError(
        f"scale must be float, int, Expr, or Tile, got {type(scale).__name__}"
    )


def _ir_store(
    out: Expr,
    tile: Expr,
    offsets: Sequence[int | Expr] | _ir_core.MakeTuple,
    *,
    span: Span | None = None,
    relu_pre_mode: ReluPreMode | None = None,
    scale: Any = None,
    order: list[int] | None = None,
    atomic: AtomicType = AtomicType.AtomicNone,
    phase: STPhase | None = None,
) -> Expr:
    op_name = "store"
    actual_span = span or _span()
    offsets_tuple = _to_make_tuple(offsets, actual_span)

    if not isinstance(out.type, _ir_core.TensorType):
        raise ValueError(f"{op_name}: dst must be a Tensor, got {type(out.type).__name__}")
    _validate_dtype(getattr(out.type, "dtype", None), "dst tensor", op_name)

    if not isinstance(tile.type, _ir_core.TileType):
        raise ValueError(f"{op_name}: src must be a Tile, got {type(tile.type).__name__}")
    _src_mem = getattr(getattr(tile.type, "memref", None), "memory_space", None)
    if _src_mem is not None and _src_mem not in (_ir_core.MemorySpace.Vec, _ir_core.MemorySpace.Acc):
        raise ValueError(f"{op_name}: src tile must be in Vec (UB) or Acc (L0C) memory, got {_src_mem.name}")
    _validate_dtype(getattr(tile.type, "dtype", None), "src tile", op_name)

    if phase is not None and not isinstance(phase, STPhase):
        raise ValueError(f"{op_name}: invalid phase value {phase!r}, expected STPhase")
    if not isinstance(atomic, AtomicType):
        raise ValueError(f"{op_name}: invalid atomic value {atomic!r}, expected AtomicType")

    tensor_ndim = len(out.type.shape)
    tile_shape = list(tile.type.shape)
    tile_dims, access_size = _validate_tile_dims(order, tensor_ndim, tile_shape, op_name)
    tile_ndim = len(tile_shape)
    if order is not None and order != sorted(order):
        raise ValueError(f"{op_name}: order must be ascending, got {order}")
    _validate_nz_transfer_axes(out.type, tile_dims, op_name)
    _validate_offsets(
        offsets_tuple,
        tile_dims,
        tile_shape,
        out.type.shape,
        op_name,
        access_size=access_size,
    )

    _validate_offset_bounds("store", out.type.shape, offsets_tuple.elements)

    pre_quant_scalar, fp_tile = _resolve_scale_param(scale, actual_span)

    _check_scale_dst_supported(
        getattr(tile.type, "dtype", None),
        getattr(out.type, "dtype", None),
        pre_quant_scalar is not None or fp_tile is not None,
        op_name,
    )
    if fp_tile is not None and relu_pre_mode is not None:
        raise ValueError("scale (per-channel) cannot be used together with relu_pre_mode")
    if fp_tile is not None and phase is not None:
        raise ValueError("scale (per-channel) cannot be combined with phase")
    if fp_tile is not None:
        return _ir_core.create_op_call(block_ir_op("store_fp"), [out, tile, fp_tile, offsets_tuple], {}, actual_span)

    kwargs = _build_store_kwargs(
        relu_pre_mode=relu_pre_mode,
        tile_dims=tile_dims,
        tensor_ndim=tensor_ndim,
        tile_ndim=tile_ndim,
        atomic=atomic,
        phase=phase,
    )
    operands: list[Expr] = [out, tile, offsets_tuple]
    if pre_quant_scalar is not None:
        pre_quant_operand = (
            ConstInt(pre_quant_scalar, DataType.UINT64, actual_span)
            if isinstance(pre_quant_scalar, int)
            else pre_quant_scalar
        )
        operands.append(pre_quant_operand)
    return _ir_core.create_op_call(block_ir_op(op_name), operands, kwargs, actual_span)


def _ir_store_fp(
    out: Expr,
    tile: Expr,
    fp_tile: Expr,
    offsets: Sequence[int | Expr] | _ir_core.MakeTuple,
    *,
    span: Span | None = None,
) -> Expr:
    actual_span = span or _span()
    offsets_tuple = _to_make_tuple(offsets, actual_span)
    # Validate offset bounds at Python frontend level
    _validate_offset_bounds("store_fp", out.type.shape, offsets_tuple.elements)
    return _ir_core.create_op_call(
        block_ir_op("store_fp"),
        [out, tile, fp_tile, offsets_tuple],
        {},
        actual_span,
    )


def _ir_store_tile(
    out: Expr,
    tile: Expr,
    tile_offsets: Sequence[int | Expr] | _ir_core.MakeTuple,
    *,
    span: Span | None = None,
    relu_pre_mode: ReluPreMode | None = None,
    scale: Any = None,
    order: list[int] | None = None,
    atomic: AtomicType = AtomicType.AtomicNone,
    phase: STPhase | None = None,
) -> Expr:
    op_name = "store_tile"
    actual_span = span or _span()
    offsets_tuple = _to_make_tuple(tile_offsets, actual_span)

    if not isinstance(out.type, _ir_core.TensorType):
        raise ValueError(f"{op_name}: dst must be a Tensor, got {type(out.type).__name__}")
    _validate_dtype(getattr(out.type, "dtype", None), "dst tensor", op_name)

    if not isinstance(tile.type, _ir_core.TileType):
        raise ValueError(f"{op_name}: src must be a Tile, got {type(tile.type).__name__}")
    _src_mem = getattr(getattr(tile.type, "memref", None), "memory_space", None)
    if _src_mem is not None and _src_mem not in (_ir_core.MemorySpace.Vec, _ir_core.MemorySpace.Acc):
        raise ValueError(f"{op_name}: src tile must be in Vec (UB) or Acc (L0C) memory, got {_src_mem.name}")
    _validate_dtype(getattr(tile.type, "dtype", None), "src tile", op_name)

    tensor_ndim = len(out.type.shape)
    tile_shape = list(tile.type.shape)
    tile_dims, access_size = _validate_tile_dims(order, tensor_ndim, tile_shape, op_name)
    tile_ndim = len(tile_shape)
    if order is not None and order != sorted(order):
        raise ValueError(f"{op_name}: order must be ascending, got {order}")
    _validate_nz_transfer_axes(out.type, tile_dims, op_name)
    _validate_offsets(
        offsets_tuple,
        tile_dims,
        tile_shape,
        out.type.shape,
        op_name,
        use_tile_absolute=True,
        access_size=access_size,
    )

    abs_offsets = _compute_absolute_offsets(
        offsets_tuple,
        tile_shape,
        tile_dims,
        actual_span,
        access_size=access_size,
    )
    _validate_offset_bounds("store_tile", out.type.shape, abs_offsets.elements)

    pre_quant_scalar, fp_tile = _resolve_scale_param(scale, actual_span)

    _check_scale_dst_supported(
        getattr(tile.type, "dtype", None),
        getattr(out.type, "dtype", None),
        pre_quant_scalar is not None or fp_tile is not None,
        op_name,
    )
    if fp_tile is not None and relu_pre_mode is not None:
        raise ValueError("scale (per-channel) cannot be used together with relu_pre_mode")
    if fp_tile is not None and phase is not None:
        raise ValueError("scale (per-channel) cannot be combined with phase")
    if fp_tile is not None:
        return _ir_core.create_op_call(block_ir_op("store_fp"), [out, tile, fp_tile, abs_offsets], {}, actual_span)
    kwargs = _build_store_kwargs(
        relu_pre_mode=relu_pre_mode,
        tile_dims=tile_dims,
        tensor_ndim=tensor_ndim,
        tile_ndim=tile_ndim,
        atomic=atomic,
        phase=phase,
    )
    operands: list[Expr] = [out, tile, abs_offsets]
    if pre_quant_scalar is not None:
        pre_quant_operand = (
            ConstInt(pre_quant_scalar, DataType.UINT64, actual_span)
            if isinstance(pre_quant_scalar, int)
            else pre_quant_scalar
        )
        operands.append(pre_quant_operand)
    return _ir_core.create_op_call(block_ir_op("store"), operands, kwargs, actual_span)


def _tile_shape_ints(tile_type: "_IRTileType") -> list[int] | None:
    """Return the compile-time integer shape of a TileType, or None if any
    dimension is not a static constant (skip check — never over-rejects)."""
    shape: list[int] = []
    for dim in tile_type.shape:
        if isinstance(dim, ConstInt):
            shape.append(int(dim.value))
        elif isinstance(dim, int):
            shape.append(int(dim))
        else:
            return None  # symbolic / dynamic dim — cannot check statically
    return shape


def _check_move_shape_compat(
    out: Expr,
    src: Expr,
    offset: Expr | Sequence[Any] | None,
    acc_to_vec_mode: AccToVecMode | None,
    actual_span: Span,
) -> None:
    """Validate src/dst tile shape compatibility for ``block.move``.

    Rules (from hardware semantics; see test_matmul_perf_asw_4k_dn_move_offset
    comment: "TEXTRACT allows src wide / dst narrow, TMOV requires shape equality"):

    - ``acc_to_vec_mode`` set (Acc→Vec split move): shape governed by split mode,
      not plain equality. Skip (safe — declared shapes may not reflect valid_shape).
    - ``offset`` given (TEXTRACT sub-block extraction): ``dst <= src`` per-dim.
    - No ``offset`` (plain TMOV): ``dst == src``, or — for 2D tiles — a transpose
      ``dst == src[::-1]`` (i.e. ``[M,N] → [N,M]``). Mat→Left / Mat→Right moves
      realize the transpose via fractal conversion (see
      test_insert_zn_transpose_left/right.py and
      test_fa_perf_tkv_preload_nbuf.py:221 where Right[TKV,TD] ← Mat[TD,TKV]).

    NOTE: ``offset[i] + dst[i] <= src[i]`` is deliberately NOT checked here.
    ``set_validshape`` can narrow runtime shapes below declared shapes, so
    offset-bounds on declared shapes would false-reject legal cases (e.g.
    test_quant_lightning_indexer_vf.py:606). The ``_validate_offset_bounds``
    helper above guards the no-acc_to_vec_mode path separately.
    """
    out_type = out.type
    src_type = src.type
    if not isinstance(out_type, _IRTileType) or not isinstance(src_type, _IRTileType):
        raise TypeError(
            f"pl.move: both dst and src must be Tiles, got dst={type(out_type).__name__}, src={type(src_type).__name__}"
        )
    dst_shape = _tile_shape_ints(out_type)
    src_shape = _tile_shape_ints(src_type)
    if dst_shape is None or src_shape is None:
        return  # symbolic shape — cannot verify statically
    if len(dst_shape) != len(src_shape):
        raise ValueError(
            f"pl.move: dst tile rank {len(dst_shape)} != src tile rank {len(src_shape)} "
            f"(dst shape={dst_shape}, src shape={src_shape})."
        )
    if acc_to_vec_mode is not None:
        return  # Acc→Vec split mode: shape relation determined by mode
    if offset is not None:
        for axis, (d, s) in enumerate(zip(dst_shape, src_shape)):
            if d > s:
                raise ValueError(
                    f"pl.move: with offset, dst dim {axis} ({d}) exceeds src dim ({s}) "
                    f"— dst must be a sub-rectangle of src "
                    f"(dst shape={dst_shape}, src shape={src_shape})."
                )
    else:
        # Plain TMOV: shapes must match, OR — for 2D tiles — be a transpose
        # ([M,N] → [N,M]). Mat→Left / Mat→Right moves realize the transpose via
        # fractal conversion; see test_fa_perf_tkv_preload_nbuf.py:221
        # (Right[TKV,TD] ← Mat[TD,TKV]) and test_insert_zn_transpose_*.py.
        if dst_shape != src_shape:
            is_2d_transpose = len(dst_shape) == 2 and dst_shape == src_shape[::-1]
            if not is_2d_transpose:
                raise ValueError(
                    f"pl.move: dst tile shape {dst_shape} != src tile shape {src_shape} "
                    f"— without offset, move requires equal shapes "
                    f"or a 2D transpose [M,N]->[N,M] "
                    f"(use offset= for sub-block extraction)."
                )


def _maybe_dispatch_to_insert(
    out: Expr,
    src: Expr,
    offset: Expr | Sequence[Any] | None,
    acc_to_vec_mode: AccToVecMode | None,
    relu_pre_mode: ReluPreMode | None,
    scale: Any,
) -> bool:
    """Whether ``move`` should dispatch to ``insert`` (TINSERT).

    Rule A: offset given, dst is a super-block of src (fractal-transpose
    pairs compare via dst[::-1]).
    Rule B: equal-shape fractal transpose on Vec->Mat (TMOV can't convert layout).
    Skipped when move-only kwargs are present.
    """
    if acc_to_vec_mode is not None or relu_pre_mode is not None or scale is not None:
        return False

    out_type = out.type
    src_type = src.type
    if not isinstance(out_type, _IRTileType) or not isinstance(src_type, _IRTileType):
        return False
    dst_shape = _tile_shape_ints(out_type)
    src_shape = _tile_shape_ints(src_type)
    if dst_shape is None or src_shape is None:
        return False
    if len(dst_shape) != len(src_shape):
        return False

    dst_layout = _tile_layout(out_type)
    src_layout = _tile_layout(src_type)
    layout_transpose = (
        dst_layout is not None
        and src_layout is not None
        and _is_fractal_transpose(dst_layout, src_layout)
    )
    dst_cmp = dst_shape[::-1] if layout_transpose else dst_shape

    is_shape_insert = (
        offset is not None
        and all(d >= s for d, s in zip(dst_cmp, src_shape))
        and any(d > s for d, s in zip(dst_cmp, src_shape))
    )

    out_mem = getattr(getattr(out_type, "memref", None), "memory_space_", None)
    src_mem = getattr(getattr(src_type, "memref", None), "memory_space_", None)
    is_equal_transpose_vec_mat = (
        layout_transpose
        and len(dst_shape) == 2
        and dst_shape == src_shape[::-1]
        and src_mem == MemorySpace.Vec
        and out_mem == MemorySpace.Mat
    )

    return is_shape_insert or is_equal_transpose_vec_mat


def _ir_move(
    out: Expr,
    src: Expr,
    offset: Expr | Sequence[Any] | None = None,
    *,
    span: Span | None = None,
    acc_to_vec_mode: AccToVecMode | None = None,
    relu_pre_mode: ReluPreMode | None = None,
    scale: Any = None,
    phase: STPhase | None = None,
) -> Expr:
    actual_span = span or _span()

    if _maybe_dispatch_to_insert(out, src, offset, acc_to_vec_mode, relu_pre_mode, scale):
        actual_offset = offset if offset is not None else [0, 0]
        return _ir_insert(out, src, actual_offset, span=actual_span)

    if not isinstance(out.type, _ir_core.TileType):
        raise ValueError(f"move: dst must be a Tile, got {type(out.type).__name__}")
    if not isinstance(src.type, _ir_core.TileType):
        raise ValueError(f"move: src must be a Tile, got {type(src.type).__name__}")
    _dst_mem = getattr(getattr(out.type, "memref", None), "memory_space_", None)
    _src_mem = getattr(getattr(src.type, "memref", None), "memory_space_", None)
    _supported_move_paths = {
        (MemorySpace.Mat, MemorySpace.Left),
        (MemorySpace.Mat, MemorySpace.Right),
        (MemorySpace.Acc, MemorySpace.Vec),
        (MemorySpace.Vec, MemorySpace.Vec),
        (MemorySpace.Mat, MemorySpace.Scaling),
        (MemorySpace.Vec, MemorySpace.Mat),
        (MemorySpace.Mat, MemorySpace.Bias),
        (MemorySpace.Mat, MemorySpace.ScaleLeft),
        (MemorySpace.Mat, MemorySpace.ScaleRight),
    }
    if _src_mem is not None and _dst_mem is not None and (_src_mem, _dst_mem) not in _supported_move_paths:
        raise ValueError(
            f"move: unsupported data path src({_src_mem.name})->dst({_dst_mem.name}), "
            f"supported paths: Mat->Left, Mat->Right, Mat->Scaling, Mat->Bias, Mat->ScaleLeft, "
            f"Mat->ScaleRight, Acc->Vec, Vec->Vec, Vec->Mat"
        )
    if phase is not None and not isinstance(phase, STPhase):
        raise ValueError(f"move: invalid phase value {phase!r}, expected STPhase")
    if phase is not None and (_src_mem, _dst_mem) != (MemorySpace.Acc, MemorySpace.Vec):
        raise ValueError("move: phase is only supported for Acc->Vec path")
    # Validate src/dst tile shape compatibility (issue #99: transpose-style mismatch)
    _check_move_shape_compat(out, src, offset, acc_to_vec_mode, actual_span)
    if offset is not None:
        if isinstance(offset, _ir_core.MakeTuple):
            _validate_offset_bounds("move", src.type.shape, offset.elements)
        elif isinstance(offset, (list, tuple)):
            _validate_offset_bounds("move", src.type.shape, offset)

    pre_quant_scalar, fp_tile = _resolve_scale_param(scale, actual_span)

    _check_scale_dst_supported(
        getattr(src.type, "dtype", None),
        getattr(out.type, "dtype", None),
        pre_quant_scalar is not None or fp_tile is not None,
        "move",
    )
    if fp_tile is not None and acc_to_vec_mode in {AccToVecMode.DualModeSplitM, AccToVecMode.DualModeSplitN}:
        raise ValueError("scale (per-channel) only supports single-mode acc_to_vec_mode")
    if phase is not None and offset is not None:
        raise ValueError("move: phase cannot be combined with offset (TEXTRACT path does not support unit_flag)")
    kwargs: dict[str, Any] = {}
    if acc_to_vec_mode is not None:
        kwargs["acc_to_vec_mode"] = acc_to_vec_mode
    if relu_pre_mode is not None:
        kwargs["relu_pre_mode"] = relu_pre_mode
    if phase is not None:
        kwargs["phase"] = phase
    if fp_tile is not None:
        return _ir_core.create_op_call(block_ir_op("move_fp"), [out, src, fp_tile], kwargs, actual_span)
    args = [out, src]
    if offset is not None:
        args.append(_to_make_tuple(offset, actual_span))
    if pre_quant_scalar is not None:
        pre_quant_operand = (
            ConstInt(pre_quant_scalar, DataType.UINT64, actual_span)
            if isinstance(pre_quant_scalar, int)
            else pre_quant_scalar
        )
        args.append(pre_quant_operand)
    return _ir_core.create_op_call(block_ir_op("move"), args, kwargs, actual_span)


def _normalize_2d_sequence(value: Any, parameter: str, span: Span) -> tuple[Expr, Expr]:
    sequence = _to_make_tuple(value, span)
    if len(sequence.elements) != 2:
        raise ValueError(f"{parameter} must contain exactly 2 elements")
    return sequence.elements[0], sequence.elements[1]


def _ir_insert(
    out: Expr,
    src: Expr,
    offset: Sequence[int | Expr] | _ir_core.MakeTuple,
    *,
    span: Span | None = None,
) -> Expr:
    actual_span = span or _span()
    row, col = _normalize_2d_sequence(offset, "offset", actual_span)
    # Validate offset bounds at Python frontend level
    _validate_offset_bounds("insert", out.type.shape, [row, col])
    return _ir_core.create_op_call(block_ir_op("insert"), [out, src, row, col], {}, actual_span)


def _ir_sel(out: Expr, mask: Expr, lhs: Expr, rhs: Expr, tmp: Expr, *, span: Span | None = None) -> Expr:
    _check_dtype("sel", getattr(out.type, "dtype", None), _SEL_DTYPES)
    return _ir_core.create_op_call(block_ir_op("sel"), [out, mask, lhs, rhs, tmp], {}, span or _span())


def _ir_sels(out: Expr, mask: Expr, src: Expr, tmp: Expr, scalar: Expr, *, span: Span | None = None) -> Expr:
    _check_dtype("sels", getattr(out.type, "dtype", None), _SEL_DTYPES)
    actual_span = span or _span()
    scalar_expr = scalar if isinstance(scalar, Expr) else _normalize_expr(scalar, actual_span)
    return _ir_core.create_op_call(block_ir_op("sels"), [out, mask, src, tmp, scalar_expr], {}, actual_span)


def _ir_neg(out: Expr, src: Expr, *, span: Span | None = None) -> Expr:
    dt = getattr(out.type, "dtype", None)
    _check_dtype("neg", dt, _NEG_DTYPES)
    _check_dtype_match("neg", dt, getattr(src.type, "dtype", None))
    return _ir_core.create_op_call(block_ir_op("neg"), [out, src], {}, span or _span())


def _ir_abs(out: Expr, src: Expr, *, span: Span | None = None) -> Expr:
    dt = getattr(out.type, "dtype", None)
    _check_dtype("abs", dt, _ABS_DTYPES)
    _check_dtype_match("abs", dt, getattr(src.type, "dtype", None))
    return _ir_core.create_op_call(block_ir_op("abs"), [out, src], {}, span or _span())


def _ir_relu(out: Expr, src: Expr, *, span: Span | None = None) -> Expr:
    dt = getattr(out.type, "dtype", None)
    _check_dtype("relu", dt, _RELU_DTYPES)
    _check_dtype_match("relu", dt, getattr(src.type, "dtype", None))
    return _ir_core.create_op_call(block_ir_op("relu"), [out, src], {}, span or _span())


def _ir_axpy(out: Expr, src: Expr, scalar: Expr, *, span: Span | None = None) -> Expr:
    dt = getattr(out.type, "dtype", None)
    src_dt = getattr(src.type, "dtype", None)
    _check_dtype("axpy", dt, _AXPY_DTYPES)
    _check_dtype("axpy", src_dt, _AXPY_DTYPES)
    if dt != src_dt:
        if not (dt == DataType.FP32 and src_dt == DataType.FP16):
            raise ValueError(
                f"axpy: dtype mismatch between out ({dt}) and src ({src_dt}). "
                f"Supported: same type, or src=FP16 + out=FP32."
            )
    return _ir_core.create_op_call(block_ir_op("axpy"), [out, src, scalar], {}, span or _span())


def _ir_add_relu(out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None) -> Expr:
    dt = getattr(out.type, "dtype", None)
    _check_dtype("add_relu", dt, _RELU_DTYPES)
    _check_dtype_match("add_relu", dt, getattr(lhs.type, "dtype", None), getattr(rhs.type, "dtype", None))
    return _ir_core.create_op_call(block_ir_op("add_relu"), [out, lhs, rhs], {}, span or _span())


def _ir_xor(out: Expr, lhs: Expr, rhs: Expr, tmp: Expr, *, span: Span | None = None) -> Expr:
    dt = getattr(out.type, "dtype", None)
    _check_dtype("xor", dt, _BITWISE_DTYPES)
    _check_dtype_match("xor", dt, getattr(lhs.type, "dtype", None), getattr(rhs.type, "dtype", None))
    return _ir_core.create_op_call(block_ir_op("xor"), [out, lhs, rhs, tmp], {}, span or _span())


def _ir_expands(out: Expr, scalar: Expr, *, span: Span | None = None) -> Expr:
    from pypto_pro.language.parser.diagnostics import check_const_expr_fits_dtype

    out_dtype = getattr(out.type, "dtype", None)
    _check_dtype("expands", out_dtype, _EXPANDS_DTYPES)
    # The splat value lands in the out tile, so it has to be representable there. Without this the
    # scalar was only bounded by the IR storage band, which is far wider than a narrow tile dtype.
    check_const_expr_fits_dtype(scalar, out_dtype, span=span, api="pl.expands")
    return _ir_core.create_op_call(block_ir_op("expands"), [out, scalar], {}, span or _span())


def _ir_row_sum(out: Expr, src: Expr, tmp: Expr, *, span: Span | None = None) -> Expr:
    dt = getattr(src.type, "dtype", None)
    _check_dtype("row_sum", dt, _SUM_DTYPES)
    _check_dtype_match("row_sum", getattr(out.type, "dtype", None), dt)
    return _ir_core.create_op_call(block_ir_op("row_sum"), [out, src, tmp], {}, span or _span())


def _ir_col_sum(out: Expr, src: Expr, tmp: Expr, *, span: Span | None = None) -> Expr:
    dt = getattr(src.type, "dtype", None)
    _check_dtype("col_sum", dt, _SUM_DTYPES)
    _check_dtype_match("col_sum", getattr(out.type, "dtype", None), dt)
    return _ir_core.create_op_call(block_ir_op("col_sum"), [out, src, tmp], {}, span or _span())


def _validate_validshape_bounds(
    tile: Expr,
    valid_shape: Sequence[int | Expr] | _ir_core.MakeTuple,
    span: Span | None = None,
) -> None:
    """Validate that valid_shape dimensions are positive and do not exceed the tile shape.

    Checks are performed only when both the valid_shape value and the corresponding
    tile shape dimension are compile-time integer constants. Symbolic shapes are
    deferred to runtime/hardware validation.
    """
    if isinstance(valid_shape, _ir_core.MakeTuple):
        return  # already normalized — caller should validate before normalization

    if not isinstance(valid_shape, (list, tuple)):
        return

    if len(valid_shape) != 2:
        return

    # Reuse _tile_shape_ints to extract compile-time tile shape
    tile_type = getattr(tile, "type", None)
    tile_shape: list[int] | None = None
    if isinstance(tile_type, _IRTileType):
        tile_shape = _tile_shape_ints(tile_type)

    dim_names = ("row", "col")
    span_info = f" at {span}" if span else ""

    for i, vs in enumerate(valid_shape):
        dim_name = dim_names[i]

        if isinstance(vs, int):
            # Rule 1: valid_shape dimensions must be positive
            if vs <= 0:
                raise ValueError(
                    f"set_validshape {dim_name}={vs} must be positive "
                    f"(got {vs}){span_info}. "
                    f"Valid shape dimensions must be >= 1."
                )

            # Rule 2: valid_shape must not exceed tile shape
            if tile_shape is not None and i < len(tile_shape):
                ts_val = tile_shape[i]
                if vs > ts_val:
                    raise ValueError(
                        f"set_validshape {dim_name}={vs} exceeds tile {dim_name}={ts_val}"
                        f"{span_info}. "
                        f"Valid shape must be <= tile shape."
                    )


def _ir_set_validshape(
    tile: Expr,
    shape: Sequence[int | Expr] | _ir_core.MakeTuple,
    *,
    span: Span | None = None,
) -> Expr:
    actual_span = span or _span()
    _validate_validshape_bounds(tile, shape, actual_span)
    shape_0, shape_1 = _normalize_2d_sequence(shape, "shape", actual_span)
    return _ir_core.create_op_call(block_ir_op("set_validshape"), [tile, shape_0, shape_1], {}, actual_span)


_SCALAR_UNSUPPORTED_DTYPES: tuple[DataType, ...] = (
    DataType.FP4,
    DataType.FP8E4M3FN,
    DataType.FP8E5M2,
    DataType.FP8E8M0,
    DataType.FP4E2M1,
    DataType.FP4E1M2,
    DataType.INT4,
    DataType.UINT4,
    DataType.HF4,
    DataType.HF8,
)

_SUPPORTED_DTYPES: tuple[DataType, ...] = (
    DataType.FP8E4M3FN,
    DataType.FP8E5M2,
    DataType.FP8E8M0,
    DataType.FP4E2M1,
    DataType.FP4E1M2,
    DataType.HF8,
    DataType.INT8,
    DataType.FP16,
    DataType.BF16,
    DataType.INT16,
    DataType.FP32,
    DataType.INT32,
    DataType.INT64,
    DataType.UINT8,
    DataType.UINT16,
    DataType.UINT32,
    DataType.UINT64,
)


def _check_scalar_supported_dtype(op_name: str, container: Expr) -> None:
    dtype = container.type.dtype
    if any(dtype == d for d in _SCALAR_UNSUPPORTED_DTYPES):
        raise TypeError(
            f"{op_name} does not support container dtype {dtype}; "
            "low-precision types (FP4/FP8/INT4/UINT4/HF4/HF8) are storage-only "
            "and cannot be used in scalar expressions"
        )


def _try_get_const_offset(off: Expr) -> int | None:
    """Extract compile-time constant offset value, or None if dynamic."""
    if isinstance(off, ConstInt):
        return off.value
    if isinstance(off, _ir_core.Neg) and isinstance(getattr(off, "operand", None), ConstInt):
        return -off.operand.value
    return None


def _validate_load_operands(
    out: Expr,
    tensor: Expr,
    offsets_tuple: _ir_core.MakeTuple,
    tile_dims: list[int] | None,
    op_name: str,
    *,
    use_tile_absolute: bool = False,
) -> tuple[int, list[Any], list[int] | None, Any | None]:
    if not isinstance(out.type, _ir_core.TileType):
        raise ValueError(f"{op_name}: dst must be a Tile, got {type(out.type).__name__}")
    dst_mem = getattr(getattr(out.type, "memref", None), "memory_space", None)
    if dst_mem is not None and dst_mem not in (_ir_core.MemorySpace.Vec, _ir_core.MemorySpace.Mat):
        raise ValueError(f"{op_name}: dst tile must be in Vec (UB) or Mat (L1) memory, got {dst_mem.name}")
    _validate_dtype(getattr(out.type, "dtype", None), "dst tile", op_name)

    if not isinstance(tensor.type, _ir_core.TensorType):
        raise ValueError(f"{op_name}: src must be a Tensor, got {type(tensor.type).__name__}")
    _validate_dtype(getattr(tensor.type, "dtype", None), "src tensor", op_name)

    tensor_ndim = len(tensor.type.shape)
    tile_shape = list(out.type.shape)
    tile_dims, access_size = _validate_tile_dims(tile_dims, tensor_ndim, tile_shape, op_name)
    _validate_offsets(
        offsets_tuple,
        tile_dims,
        tile_shape,
        tensor.type.shape,
        op_name,
        use_tile_absolute=use_tile_absolute,
        access_size=access_size,
    )
    return tensor_ndim, tile_shape, tile_dims, access_size


def _build_store_kwargs(
    *,
    relu_pre_mode: ReluPreMode | None,
    tile_dims: list[int] | None,
    tensor_ndim: int,
    tile_ndim: int,
    atomic: AtomicType,
    phase: STPhase | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if relu_pre_mode is not None:
        kwargs["relu_pre_mode"] = relu_pre_mode
    if tile_dims is not None:
        default_tile_dims = list(range(tensor_ndim - tile_ndim, tensor_ndim))
        if tile_dims != default_tile_dims:
            kwargs["tile_dims"] = tile_dims
    if atomic != AtomicType.AtomicNone:
        kwargs["atomic"] = atomic
    if phase is not None:
        kwargs["phase"] = phase
    return kwargs


def _validate_dtype(dtype: DataType | None, role: str, op_name: str) -> None:
    if dtype is not None and dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"{op_name}: unsupported {role} dtype {dtype}, supported: b8/b16/b32/b64")


def _check_dtype(op_name: str, dtype: DataType | None, allowed: tuple[DataType, ...]) -> None:
    if dtype is not None and dtype not in allowed:
        allowed_names = ", ".join(str(d) for d in allowed)
        raise ValueError(
            f"{op_name}: unsupported dtype {dtype}, supported: {allowed_names}"
        )


def _check_dtype_match(op_name: str, dt: DataType | None, *others: DataType | None) -> None:
    for i, other in enumerate(others):
        if other is not None and other != dt:
            raise ValueError(f"{op_name}: dtype mismatch between arg0 ({dt}) and arg{i + 1} ({other})")


# Per-op supported dtype sets (aligned with ISA static_assert constraints)
_BINARY_DTYPES: tuple[DataType, ...] = (
    DataType.INT8, DataType.UINT8, DataType.INT16, DataType.UINT16,
    DataType.INT32, DataType.UINT32, DataType.INT64, DataType.UINT64,
    DataType.FP16, DataType.FP32, DataType.BF16,
)
_MUL_DTYPES: tuple[DataType, ...] = tuple(
    d for d in _BINARY_DTYPES if d not in (DataType.INT8, DataType.UINT8)
)
_DIV_DTYPES: tuple[DataType, ...] = tuple(
    d for d in _MUL_DTYPES if d != DataType.BF16
)
_BITWISE_DTYPES: tuple[DataType, ...] = (
    DataType.INT8, DataType.UINT8, DataType.INT16, DataType.UINT16,
    DataType.INT32, DataType.UINT32,
)
_NEG_DTYPES: tuple[DataType, ...] = (
    DataType.INT16, DataType.UINT16, DataType.INT32, DataType.UINT32,
    DataType.INT64, DataType.UINT64, DataType.FP16, DataType.FP32, DataType.BF16,
)
_ABS_DTYPES: tuple[DataType, ...] = (
    DataType.INT8, DataType.INT16, DataType.INT32, DataType.FP16, DataType.FP32,
)
_RELU_DTYPES: tuple[DataType, ...] = (DataType.INT32, DataType.FP16, DataType.FP32)
_AXPY_DTYPES: tuple[DataType, ...] = (DataType.FP16, DataType.FP32)
_CMP_DTYPES: tuple[DataType, ...] = _BINARY_DTYPES
_EXPANDS_DTYPES: tuple[DataType, ...] = _BINARY_DTYPES
_SUM_DTYPES: tuple[DataType, ...] = (
    DataType.INT16, DataType.INT32, DataType.FP16, DataType.FP32,
    DataType.INT64, DataType.UINT64,
)
_SEL_DTYPES: tuple[DataType, ...] = _BINARY_DTYPES + (DataType.BOOL,)


def _check_scale_dst_supported(src_dtype: DataType | None, dst_dtype: DataType | None, is_quant_active: bool,
                               op_name: str) -> None:
    """Reject scale quantization to dtype combinations the hardware fixpipe cannot produce.

    The fixpipe has no unsigned requantization (UINT8 output) and only dequantizes
    INT32→FP16 (not INT32→BF16); FP32→BF16 only supports a no-scale truncation.
    These combos compile but fault on device (NPU device error 507015), so surface
    them as parse-time errors instead of a device crash.
    """
    if not is_quant_active:
        return
    if dst_dtype == DataType.UINT8:
        raise ValueError(
            f"{op_name}: scale quantization to UINT8 is not supported — the hardware "
            "fixpipe has no unsigned requantization path. Use an INT8 output, or quantize "
            "in the Vector (UB) domain."
        )
    if dst_dtype == DataType.BF16 and src_dtype in (DataType.FP32, DataType.INT32):
        raise ValueError(
            f"{op_name}: scale quantization from {src_dtype} to BF16 is not supported — "
            "the fixpipe only dequantizes INT32→FP16 and only truncates FP32→BF16 without "
            "a scale. Write FP16 output, or drop the scale and truncate in the Vector (UB) domain."
        )


def _resolve_order(
    order: list[int] | None,
    tensor_ndim: int,
    tile_ndim: int,
    op_name: str,
) -> tuple[list[int], bool]:
    """Resolve the ``order`` kwarg into ``(tile_dims, is_transpose)``.

    ``order`` maps each Tile dimension to a Tensor axis (absolute index).
    Ascending order means no transposition; descending order means transposition
    (DN layout). Internally we decompose into ``tile_dims`` (sorted ascending,
    consumed by C++ codegen) and ``is_transpose`` (bool, consumed by codegen).
    """
    if order is None:
        order = list(range(tensor_ndim - tile_ndim, tensor_ndim))
    is_transpose = len(order) >= 2 and order[0] > order[1]
    tile_dims = sorted(order)
    return tile_dims, is_transpose


def _validate_rank1_tensor_tile_shape(tile_shape: list[Any], op_name: str) -> Any:
    shape_values = [
        dim.value if isinstance(dim, ConstInt) else dim if isinstance(dim, int) else None
        for dim in tile_shape
    ]
    if 1 not in shape_values:
        raise ValueError(
            f"{op_name}: rank-1 Tensor requires a rank-2 Tile with one dimension equal to 1, "
            f"got Tile shape {shape_values}"
        )

    access_size = tile_shape[0] if shape_values[1] == 1 else tile_shape[1]
    return access_size


def _validate_tile_dims(
    tile_dims: list[int] | None,
    tensor_ndim: int,
    tile_shape: list[Any],
    op_name: str,
) -> tuple[list[int] | None, Any | None]:
    if tensor_ndim == 1:
        access_size = _validate_rank1_tensor_tile_shape(tile_shape, op_name)
        return None, access_size

    tile_ndim = len(tile_shape)
    if tile_dims is None:
        tile_dims = list(range(tensor_ndim - tile_ndim, tensor_ndim))
    if len(set(tile_dims)) != len(tile_dims):
        raise ValueError(f"{op_name}: order axes must be unique, got {tile_dims}")
    for dim in tile_dims:
        if dim < 0 or dim >= tensor_ndim:
            raise ValueError(f"{op_name}: order axis {dim} is out of range for Tensor rank {tensor_ndim}")
    return tile_dims, None


def _validate_offsets(
    offsets_tuple: _ir_core.MakeTuple,
    tile_dims: list[int] | None,
    tile_shape: list[Any],
    tensor_shape: Sequence[Any],
    op_name: str,
    *,
    use_tile_absolute: bool = False,
    access_size: Any | None = None,
) -> None:
    if len(tensor_shape) != len(offsets_tuple.elements):
        raise ValueError(
            f"{op_name}: Tensor rank {len(tensor_shape)} requires exactly {len(tensor_shape)} offsets, "
            f"got {len(offsets_tuple.elements)}"
        )
    for i, off in enumerate(offsets_tuple.elements):
        off_val = _try_get_const_offset(off)
        if off_val is not None and off_val < 0:
            raise ValueError(f"{op_name}: offsets[{i}] is {off_val}, negative offset is not allowed")
        if off_val is None:
            continue
        if access_size is not None:
            t_size = access_size
        elif tile_dims is not None and i in tile_dims:
            t_size = tile_shape[tile_dims.index(i)]
        else:
            continue

        t_dim = tensor_shape[i] if i < len(tensor_shape) else None
        check_val = off_val * t_size.value if use_tile_absolute else off_val
        label = f" (absolute offset {check_val})" if use_tile_absolute else ""
        if isinstance(t_size, ConstInt) and isinstance(t_dim, ConstInt) and check_val >= t_dim.value:
            raise ValueError(f"{op_name}: offsets[{i}]={off_val}{label} exceeds tensor dim {i} size {t_dim.value}")


def _ir_getval(container: Expr, offset: int | Expr, *, span: Span | None = None) -> Expr:
    actual_span = span or _span()
    _ctype = container.type
    if not isinstance(_ctype, (_ir_core.TileType, _ir_core.TensorType)):
        from pypto_pro.language.parser.diagnostics import ParserTypeError

        raise ParserTypeError(
            f"getval: 'container' must be a Tile or Tensor, got {type(_ctype).__name__}",
            span=actual_span,
            hint="getval reads a scalar from a Tile/Tensor slot; to access a struct/tiling "
            "field, use attribute access (e.g. tiling.axis1) instead.",
        )
    _check_scalar_supported_dtype("getval", container)
    offset_expr = offset if isinstance(offset, Expr) else _normalize_expr(offset, actual_span, int_dtype=DataType.INDEX)
    return _ir_core.create_op_call(block_ir_op("getval"), [container, offset_expr], {}, actual_span)


def _ir_setval(container: Expr, offset: int | Expr, value: int | float | Expr, *, span: Span | None = None) -> Expr:
    actual_span = span or _span()
    _ctype = container.type
    if not isinstance(_ctype, (_ir_core.TileType, _ir_core.TensorType)):
        from pypto_pro.language.parser.diagnostics import ParserTypeError

        raise ParserTypeError(
            f"setval: 'container' must be a Tile or Tensor, got {type(_ctype).__name__}",
            span=actual_span,
            hint="setval writes a scalar into a Tile/Tensor slot; to write a struct/tiling "
            "field, use attribute assignment (e.g. tiling.axis1 = ...) instead.",
        )
    _check_scalar_supported_dtype("setval", container)
    offset_expr = offset if isinstance(offset, Expr) else _normalize_expr(offset, actual_span, int_dtype=DataType.INDEX)
    if not isinstance(value, Expr):
        container_dtype = container.type.dtype
        value_expr = _normalize_expr(value, actual_span, int_dtype=container_dtype, float_dtype=container_dtype)
    else:
        value_expr = value
    return _ir_core.create_op_call(block_ir_op("setval"), [container, offset_expr, value_expr], {}, actual_span)


def _ir_transpose(
    out: Expr, src: Expr, axis1: int | Expr = 0, axis2: int | Expr = 1, *, span: Span | None = None
) -> Expr:
    dt = getattr(out.type, "dtype", None)
    if dt is not None and dt.get_bit() not in (8, 16, 32):
        raise ValueError(f"transpose: unsupported dtype {dt}, supported: b8/b16/b32")
    src_dt = getattr(src.type, "dtype", None)
    if src_dt is not None and dt is not None and src_dt.get_bit() != dt.get_bit():
        raise ValueError(f"transpose: dtype size mismatch between dst ({dt}) and src ({src_dt})")
    return _ir_core.create_op_call(
        block_ir_op("transpose"),
        [out, src],
        {"axis1": _const_int_attr(axis1, "axis1"), "axis2": _const_int_attr(axis2, "axis2")},
        span or _span(),
    )


def _ir_cast(
    out: Expr,
    src: Expr,
    *,
    span: Span | None = None,
    mode: RoundMode = RoundMode.CAST_ROUND,
) -> Expr:
    target_type = out.type.dtype
    return _ir_core.create_op_call(
        block_ir_op("cast"),
        [out, src],
        {"target_type": target_type, "mode": mode},
        span or _span(),
    )


def _ir_fillpad(
    out: Expr,
    src: Expr,
    *,
    span: Span | None = None,
    mode: FillPadMode = FillPadMode.NORMAL,
) -> Expr:
    name = {
        FillPadMode.NORMAL: "fillpad",
        FillPadMode.EXPAND: "fillpad_expand",
        FillPadMode.INPLACE: "fillpad_inplace",
    }[mode]
    return _ir_core.create_op_call(
        block_ir_op(name),
        [out, src],
        {},
        span or _span(),
    )


def _ir_add_relu_cast(
    out: Expr,
    lhs: Expr,
    rhs: Expr,
    target_type: int | DataType,
    *,
    span: Span | None = None,
    mode: RoundMode = RoundMode.CAST_ROUND,
) -> Expr:
    return _ir_binary_cast("add_relu_cast", out, lhs, rhs, target_type, span=span, mode=mode)


def _ir_sub_relu_cast(
    out: Expr,
    lhs: Expr,
    rhs: Expr,
    target_type: int | DataType,
    *,
    span: Span | None = None,
    mode: RoundMode = RoundMode.CAST_ROUND,
) -> Expr:
    return _ir_binary_cast("sub_relu_cast", out, lhs, rhs, target_type, span=span, mode=mode)


def _ir_mul_cast(
    out: Expr,
    lhs: Expr,
    rhs: Expr,
    target_type: int | DataType,
    *,
    span: Span | None = None,
    mode: RoundMode = RoundMode.CAST_ROUND,
) -> Expr:
    return _ir_binary_cast("mul_cast", out, lhs, rhs, target_type, span=span, mode=mode)


def _ir_cmp(out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None, cmp_mode: int | Expr = 0) -> Expr:
    dt = getattr(lhs.type, "dtype", None)
    _check_dtype("cmp", dt, _CMP_DTYPES)
    _check_dtype_match("cmp", dt, getattr(rhs.type, "dtype", None))
    return _ir_core.create_op_call(
        block_ir_op("cmp"),
        [out, lhs, rhs],
        {"cmp_mode": _const_int_attr(cmp_mode, "cmp_mode")},
        span or _span(),
    )


def _ir_cmps(out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None, cmp_mode: int | Expr = 0) -> Expr:
    return _ir_core.create_op_call(
        block_ir_op("cmps"),
        [out, lhs, rhs],
        {"cmp_mode": _const_int_attr(cmp_mode, "cmp_mode")},
        span or _span(),
    )


def _ir_set_mask_count(*, span: Span | None = None) -> Expr:
    return _ir_core.create_op_call("system.set_mask_count", [], {}, span or _span())


def _ir_set_mask_norm(*, span: Span | None = None) -> Expr:
    return _ir_core.create_op_call("system.set_mask_norm", [], {}, span or _span())


def _ir_set_vec_mask(mask_high: Expr, mask_low: Expr, *, span: Span | None = None) -> Expr:
    return _ir_core.create_op_call("system.set_vec_mask", [mask_high, mask_low], {}, span or _span())


def _ir_reset_mask(*, span: Span | None = None) -> Expr:
    return _ir_core.create_op_call("system.reset_mask", [], {}, span or _span())


def _ir_quant(
    out: Expr,
    src: Expr,
    scale: Expr,
    *,
    span: Span | None = None,
    mode: QuantMode = QuantMode.SYM,
    offset: Expr | None = None,
) -> Expr:
    ins = [out, src, scale]
    if mode == QuantMode.ASYM:
        if offset is None:
            raise ValueError("quant in 'asym' mode requires an offset argument")
        ins.append(offset)
    return _ir_core.create_op_call(block_ir_op("quant"), ins, {"mode": mode}, span or _span())


def _ir_dequant(out: Expr, src: Expr, scale: Expr, offset: Expr, *, span: Span | None = None) -> Expr:
    return _ir_core.create_op_call(block_ir_op("dequant"), [out, src, scale, offset], {}, span or _span())


def _ir_ssbuf_store(*args: Expr, span: Span | None = None) -> Expr:
    return _ir_core.create_op_call(block_ir_op("ssbuf_store"), list(args), {}, span or _span())


def _ir_ssbuf_load(*args: Expr, span: Span | None = None) -> Expr:
    return _ir_core.create_op_call(block_ir_op("ssbuf_load"), list(args), {}, span or _span())


# ---------------------------------------------------------------------------
# TileType descriptor and make_tile_expr
# ---------------------------------------------------------------------------


def _get_current_arch() -> str:
    arch = os.environ.get("PYPTOPRO_JIT_ARCH") or os.environ.get("PYPTOPRO_NPU_ARCH") or "a3"
    arch = arch.strip().lower()
    if arch.startswith("dav-c220") or arch.startswith("dav-2201"):
        return "a3"
    if arch.startswith("dav-c310") or arch.startswith("dav-3510"):
        return "a5"
    if arch in ("a2", "a3", "a5"):
        return arch
    return "a3"


_LAYOUT_TO_BS: dict[TensorLayout, tuple[int, int]] = {
    TensorLayout.ND: (1, 0),
    TensorLayout.DN: (2, 0),
    TensorLayout.NZ: (2, 1),
    TensorLayout.ZN: (1, 2),
    TensorLayout.NN: (2, 2),
    TensorLayout.ZZ: (1, 1),
}

_BS_TO_LAYOUT: dict[tuple[int, int], TensorLayout] = {v: k for k, v in _LAYOUT_TO_BS.items()}


_FRACTAL_TRANSPOSE: dict[TensorLayout, TensorLayout] = {
    TensorLayout.NZ: TensorLayout.ZN,
    TensorLayout.ZN: TensorLayout.NZ,
    TensorLayout.NN: TensorLayout.ZZ,
    TensorLayout.ZZ: TensorLayout.NN,
    TensorLayout.ND: TensorLayout.DN,
    TensorLayout.DN: TensorLayout.ND,
}


def _is_fractal_transpose(a: TensorLayout, b: TensorLayout) -> bool:
    """True when layouts a, b are transpose-pairs (Z<->N flipped)."""
    return _FRACTAL_TRANSPOSE.get(a) == b


# A5 cube buffers (L0A/L0B/L0C) do not support these layouts.
_REJECTED_LAYOUTS_ON_A5 = (TensorLayout.ZZ, TensorLayout.NN)


def _tile_layout(tile_type: "_IRTileType") -> "TensorLayout | None":
    """Return the TensorLayout of a TileType derived from its hardware_info blayout/slayout."""
    hw = getattr(tile_type, "hardware_info", None)
    if hw is None:
        return None
    return _BS_TO_LAYOUT.get((int(hw.blayout), int(hw.slayout)))


_DEFAULT_LAYOUTS_A3: dict[MemorySpace, TensorLayout] = {
    MemorySpace.Mat: TensorLayout.NZ,
    MemorySpace.Left: TensorLayout.ZZ,
    MemorySpace.Right: TensorLayout.ZN,
    MemorySpace.Scaling: TensorLayout.ND,
    MemorySpace.Acc: TensorLayout.NZ,
}

_DEFAULT_LAYOUTS_A5: dict[MemorySpace, TensorLayout] = {
    MemorySpace.Mat: TensorLayout.NZ,
    MemorySpace.Left: TensorLayout.NZ,
    MemorySpace.Right: TensorLayout.ZN,
    MemorySpace.Scaling: TensorLayout.ND,
    MemorySpace.Acc: TensorLayout.NZ,
    MemorySpace.ScaleLeft: TensorLayout.ZZ,
    MemorySpace.ScaleRight: TensorLayout.NN,
}

_MX_SCALE_FRACTAL = 32

mem_id: int = 0

_PAD_VALUES = {
    TilePad.null: 0,
    TilePad.zero: 1,
    TilePad.max: 2,
    TilePad.min: 3,
}


def _normalize_tile_pad(pad: "int | TilePad | None") -> "int | None":
    if pad is None:
        return None
    if pad in _PAD_VALUES:
        return _PAD_VALUES[pad]
    if isinstance(pad, int):
        if pad not in _PAD_VALUES.values():
            raise ValueError("TileType.pad must be one of TilePad.null/zero/max/min")
        return pad
    raise TypeError("TileType.pad must be a enum TilePad or compile-time integer 0/1/2/3")


def _static_last_axis(shape: "Sequence[int] | _ir_core.MakeTuple") -> "int | None":
    """The tile's innermost dimension when it is a compile-time integer, else None."""
    dims = shape.elements if isinstance(shape, _ir_core.MakeTuple) else shape
    if not dims:
        return None
    last = dims[-1]
    if isinstance(last, ConstInt):
        return int(last.value)
    return last if isinstance(last, int) and not isinstance(last, bool) else None


def _validate_subbyte_tile_shape(tt: "TileType") -> None:
    """A sub-byte tile must span whole storage units along the axis its elements pack into.

    Several sub-byte elements share a byte, and only the innermost axis can carry that packing,
    so a tile that ends mid-byte has no address for its own edge. Tensors are exempt: their
    innermost extent may be dynamic, and it is the tile that decides what a transfer touches.
    """
    pack = 8 // tt.dtype.get_bit()
    if pack <= 1:
        return
    last = _static_last_axis(tt.shape)
    if last is not None and last % pack != 0:
        raise ValueError(
            f"a {tt.dtype.to_string()} tile's last dimension must be a multiple of {pack} "
            f"({pack} elements share a storage unit), got {last}. Pad the tile to the next "
            f"multiple of {pack}."
        )


def _has_unit_last_axis(shape: "Sequence[int] | _ir_core.MakeTuple") -> bool:
    """Whether the tile's innermost dimension is statically 1 (dynamic dims answer False)."""
    dims = shape.elements if isinstance(shape, _ir_core.MakeTuple) else shape
    if not dims:
        return False
    last = dims[-1]
    if isinstance(last, ConstInt):
        return int(last.value) == 1
    return isinstance(last, int) and not isinstance(last, bool) and last == 1


def _apply_default_layout(tt: "TileType") -> None:
    arch = _get_current_arch()
    layout_dict = _DEFAULT_LAYOUTS_A5 if arch == "a5" else _DEFAULT_LAYOUTS_A3
    default_layout = layout_dict.get(tt.target_memory)

    if tt.target_memory in (MemorySpace.ScaleLeft, MemorySpace.ScaleRight):
        if default_layout is None:
            raise ValueError(f"{tt.target_memory.name} is only supported on A5, got architecture '{arch}'")

    # A tile whose last axis is 1 has only its rows left to align, so the ISA accepts it in one
    # encoding only: BLayout::RowMajor needs Cols * sizeof(dtype) % 32 == 0, which a 1-wide tile
    # cannot satisfy, and codegen therefore emits ColMajor (DN). A fractal layout would have its
    # block layout silently rewritten to something the user did not ask for -- ZN on a [N, 1] Mat
    # tile would come out NN. Checked before the early return below, because the memory space
    # where column tiles actually live (Vec) has no default layout and would skip it otherwise.
    if tt.layout is not None and _has_unit_last_axis(tt.shape) and tt.layout not in (
        TensorLayout.ND,
        TensorLayout.DN,
    ):
        raise ValueError(
            f"a tile whose last axis is 1 requires layout ND or DN, got {tt.layout.name}: a "
            f"1-wide tile cannot meet the 32-byte column alignment a fractal layout needs, so "
            f"it is always emitted column-major. Drop the layout argument (DN is inferred), or "
            f"widen the tile if you need {tt.layout.name}."
        )

    if default_layout is None:
        return

    if tt.layout is None:
        tt.layout = default_layout

    # L1 ZZ/NN scale views and the dedicated scale buffers use the PTO MX fractal size.
    is_mx_scale_tile = tt.target_memory in (MemorySpace.ScaleLeft, MemorySpace.ScaleRight) or (
        tt.target_memory == MemorySpace.Mat and tt.layout in (TensorLayout.ZZ, TensorLayout.NN)
    )
    if is_mx_scale_tile:
        if tt.fractal not in (None, _MX_SCALE_FRACTAL):
            raise ValueError(
                f"{tt.target_memory.name} MX scale tiles require fractal={_MX_SCALE_FRACTAL}, got {tt.fractal}"
            )
        tt.fractal = _MX_SCALE_FRACTAL

    allowed_layouts = {default_layout}
    if tt.target_memory == MemorySpace.Left:
        allowed_layouts = {_DEFAULT_LAYOUTS_A3[MemorySpace.Left], _DEFAULT_LAYOUTS_A5[MemorySpace.Left]}
    elif tt.target_memory == MemorySpace.Mat:
        allowed_layouts.add(TensorLayout.ZN)
        if tt.dtype in (DataType.UINT64, DataType.INT64):
            allowed_layouts.add(TensorLayout.ND)
        if tt.dtype == DataType.FP8E8M0:
            allowed_layouts.update({TensorLayout.ZZ, TensorLayout.NN})

    if (
        arch == "a5"
        and tt.target_memory in (MemorySpace.Left, MemorySpace.Right, MemorySpace.Acc)
        and tt.layout in _REJECTED_LAYOUTS_ON_A5
    ):
        raise ValueError(
            f"{tt.target_memory.name} tiles do not support {tt.layout.name} layout on '{arch}'; "
            f"use the default {default_layout.name} layout instead."
        )

    if tt.layout not in allowed_layouts:
        space_name = tt.target_memory.name
        allowed_text = ", ".join(f.name for f in sorted(allowed_layouts, key=lambda x: x.name))
        raise ValueError(
            f"{space_name} tiles require layout in {{{allowed_text}}}, "
            f"got {tt.layout.name}. "
            f"Default for '{arch}' is {default_layout.name}."
        )

    if tt.target_memory == MemorySpace.Acc and tt.fractal is None:
        if tt.dtype in (DataType.FP32, DataType.INT32):
            tt.fractal = 1024


# CompactMode enum (pto/type.hpp): Null=0, Normal=1, RowPlusOne=2,
# RowAlignedPadding=3. Any other integer is not a valid compact mode.
_COMPACT_VALUES = frozenset({0, 1, 2, 3})

@dataclass
class TileType:
    """Tile type descriptor containing shape, dtype, and TileView parameters."""

    shape: "Sequence[int] | _ir_core.MakeTuple"
    dtype: DataType
    target_memory: MemorySpace = MemorySpace.Vec
    valid_shape: Optional[Sequence[int]] = None
    layout: Optional[TensorLayout] = None
    fractal: Optional[int] = None
    pad: Optional[int] = None
    compact: Optional[int] = None

    def __post_init__(self):
        _validate_tile_type_params(self)
        self.pad = _normalize_tile_pad(self.pad)
        _validate_subbyte_tile_shape(self)
        _apply_default_layout(self)


# Memory-space address alignment requirements (in bytes).
# Hardware constraint: misaligned tile addresses cause silent corruption or
# device-side runtime errors on move/load/store operations.
# - L1 (Mat) buffer:      32-byte alignment
# - L0A (Left) buffer:   512-byte alignment
# - L0B (Right) buffer:  512-byte alignment
# - L0C (Acc) buffer:     64-byte alignment
# - Vec (UB) buffer:      32-byte alignment
_MEMORY_ALIGNMENT: dict[MemorySpace, int] = {
    MemorySpace.Mat: 32,
    MemorySpace.Vec: 32,
    MemorySpace.Left: 512,
    MemorySpace.Right: 512,
    MemorySpace.Acc: 64,
    MemorySpace.ScaleLeft: 32,
    MemorySpace.ScaleRight: 32,
}


def _validate_tile_addr_alignment(
    addr: int,
    target_memory: MemorySpace,
    span: "Span | None" = None,
) -> None:
    """Validate that a tile address is properly aligned for its memory space.

    Raises ValueError with a descriptive message if the address is misaligned.
    """
    required = _MEMORY_ALIGNMENT.get(target_memory)
    if required is None:
        return  # DDR / Scaling / Bias — no enforced alignment
    if addr % required != 0:
        mem_name = str(target_memory).replace("MemorySpace.", "")
        span_info = f" at {span}" if span else ""
        raise ValueError(
            f"Tile address 0x{addr:05X} ({addr}) is not {required}-byte aligned "
            f"for memory space {mem_name}{span_info}. "
            f"Address must be a multiple of {required}."
        )


_MAKE_TILE_HINT = (
    "A tile occupies a fixed range of its memory space, so it needs an address, "
    "e.g. pl.make_tile(tile_type, addr=0x0); use pl.make_tile_group(type=..., "
    "addrs=..., depth=...) to lay several tiles out from one base address."
)

_MAKE_TILE_TYPE_HINT = (
    "A tile's shape, dtype, memory space and layout all come from a pl.TileType, which is "
    "pl.make_tile()'s only positional input; addr/size are keywords, e.g. "
    "pl.make_tile(pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, "
    "target_memory=pl.MemorySpace.Vec), addr=0x0)."
)


def _static_dim(dim) -> "int | None":
    """A shape dimension as a plain int, or None when it is not known while parsing."""
    if isinstance(dim, bool):
        return None
    if isinstance(dim, int):
        return dim
    if isinstance(dim, ConstInt):
        return int(dim.value)
    return None


def tile_slot_size(shape: "Sequence[int] | _ir_core.MakeTuple", dtype: DataType) -> int:
    """Byte footprint of one tile, derived from its static shape and dtype.

    Shared by pl.make_tile()'s default ``size`` and make_tile_group()'s slot stride,
    so a group slot is exactly as wide as a standalone tile of the same TileType.

    Raises ValueError when the shape is not a tuple of compile-time positive
    integers, which is when the caller has to state ``size`` itself.
    """
    dims = list(shape.elements) if isinstance(shape, _ir_core.MakeTuple) else list(shape)
    static_dims = [_static_dim(dim) for dim in dims]
    if any(dim is None for dim in static_dims):
        raise ValueError(f"tile shape must contain compile-time integers, got {dims}")
    if any(dim <= 0 for dim in static_dims):
        raise ValueError(f"tile shape dimensions must be positive, got {static_dims}")
    elems = 1
    for dim in static_dims:
        elems *= dim
    return elems * max(1, (int(dtype.get_bit()) + 7) // 8)


def _const_shape_ints(shape) -> "list[int] | None":
    """Extract compile-time ints from a tile shape (list/tuple or MakeTuple of ConstInt).

    Returns None when any element is not a compile-time integer constant.
    """
    if isinstance(shape, _ir_core.MakeTuple):
        ints = []
        for elt in shape.elements:
            if isinstance(elt, _ir_core.ConstInt):
                ints.append(elt.value)
            else:
                return None
        return ints
    if isinstance(shape, (list, tuple)) and all(
        isinstance(s, int) and not isinstance(s, bool) for s in shape
    ):
        return list(shape)
    return None


def _validate_tile_type_params(tt: "TileType") -> None:
    """Reject invalid TileType parameter values early with actionable errors.

    Runs in ``TileType.__post_init__`` so both the AST-parsed path
    (``pl.TileType(...)`` inside a kernel) and direct Python construction are
    covered. Invalid values otherwise surface as opaque device/runtime errors
    (e.g. zero/negative tile sizes, misaligned derived addresses).
    """
    if not isinstance(tt.dtype, DataType):
        raise TypeError(
            f"TileType dtype must be a pl.DT_* / DataType value, got {tt.dtype!r} "
            f"({type(tt.dtype).__name__})."
        )
    if not isinstance(tt.target_memory, MemorySpace):
        raise TypeError(
            f"TileType target_memory must be a pl.MemorySpace value, got {tt.target_memory!r} "
            f"({type(tt.target_memory).__name__})."
        )

    shape = _const_shape_ints(tt.shape)
    if shape is None:
        # Symbolic/dynamic dimensions (e.g. a runtime tensor dim) are allowed to
        # survive into the TileType; consuming code (make_tile/make_tile_group)
        # reports them when a compile-time size is actually required.
        pass
    else:
        if not shape:
            raise ValueError(f"TileType shape must not be empty, got {tt.shape!r}.")
        if any(s <= 0 for s in shape):
            raise ValueError(
                f"TileType shape dimensions must be positive, got {tt.shape!r}. "
                f"Zero or negative tile dimensions produce invalid tile allocations."
            )

        if tt.target_memory == MemorySpace.Bias and shape[0] != 1:
            raise ValueError(
                f"TileType Bias tiles must have exactly 1 row (hardware Rows==1 constraint, "
                f"see TMatmul.hpp/TMov.hpp), got shape {tt.shape!r} with {shape[0]} rows. "
                f"Use shape=[1, N] for the Bias (L0B) tile."
            )

    valid_shape = _const_shape_ints(tt.valid_shape)
    if valid_shape is not None and shape is not None:
        if len(valid_shape) != len(shape):
            raise ValueError(
                f"TileType valid_shape {tt.valid_shape!r} has rank {len(valid_shape)}, "
                f"which must match shape rank {len(shape)}."
            )
        for vs, s in zip(valid_shape, shape):
            if vs == -1:
                continue  # -1: dynamic valid dimension, set at runtime
            if vs <= 0:
                raise ValueError(
                    f"TileType valid_shape dimensions must be positive or -1 (dynamic), "
                    f"got {tt.valid_shape!r}."
                )
            if vs > s:
                raise ValueError(
                    f"TileType valid_shape dimension {vs} exceeds tile shape dimension {s}."
                )

    if tt.fractal is not None and not _is_int(tt.fractal):
        raise ValueError(f"TileType fractal must be an integer, got {tt.fractal!r}.")

    if tt.compact is not None:
        if isinstance(tt.compact, bool) or not isinstance(tt.compact, int) or tt.compact not in _COMPACT_VALUES:
            raise ValueError(
                f"TileType compact must be one of {sorted(_COMPACT_VALUES)} "
                f"(CompactMode: null=0/normal=1/row_plus_one=2/row_aligned_padding=3), "
                f"got {tt.compact!r}."
            )


def make_tile_expr(
    shape: "Sequence[int] | _ir_core.MakeTuple",
    dtype: DataType,
    target_memory: MemorySpace,
    *,
    addr: "int",
    size: "int | None" = None,
    valid_shape: "Sequence[int] | _ir_core.MakeTuple | None" = None,
    layout: "TensorLayout | None" = None,
    fractal: "int | None" = None,
    pad: "int | None" = None,
    compact: "int | None" = None,
    span: "Span | None" = None,
) -> Expr:
    """Create the block.make_tile allocation expression used by block buffers.

    This is the IR builder, which takes the tile's fields spread out. The DSL
    entry point is ``pl.make_tile(tile_type, *, addr, size=None)``, whose parse
    handler unpacks the TileType and calls this.

    ``addr`` is required: it is what attaches a MemRef to the tile type, and a
    tile without one is placed nowhere in particular. ``size`` defaults to the
    footprint derived from ``shape`` and ``dtype``; pass it only to reserve more
    than that, as an NZ/ZN tile rounded up to whole fractals does.
    """
    actual_span = span or _span()
    shape_tuple = _to_make_tuple(shape, actual_span)
    valid_shape_tuple = (
        _to_make_tuple(valid_shape, actual_span) if valid_shape is not None else _ir_core.MakeTuple([], actual_span)
    )
    args = [shape_tuple, valid_shape_tuple]
    blayout: "int | None" = None
    slayout: "int | None" = None
    if layout is not None:
        b, s = _LAYOUT_TO_BS[layout]
        blayout = b
        slayout = s
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "target_memory": target_memory,
        "blayout": blayout,
        "slayout": slayout,
        "fractal": fractal,
        "pad": pad,
        "compact": compact,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if addr is None:
        raise ValueError(f"make_tile_expr() requires 'addr', but it is missing. {_MAKE_TILE_HINT}")
    if size is None:
        try:
            size = tile_slot_size(shape_tuple, dtype)
        except ValueError as exc:
            raise ValueError(
                f"pl.make_tile() cannot derive 'size' from the tile type: {exc}. Pass an explicit byte size."
            ) from exc
    if isinstance(addr, int):
        _validate_tile_addr_alignment(addr, target_memory, actual_span)
    global mem_id
    mem_id += 1
    kwargs["memref_addr"] = addr
    kwargs["memref_size"] = size
    kwargs["memref_id"] = mem_id
    return _ir_core.create_op_call(block_ir_op("make_tile"), args, kwargs, actual_span)


# ---------------------------------------------------------------------------
# Parse handlers — registered into _OP_REGISTRY at import time
# ---------------------------------------------------------------------------


@op_impl("TileType")
def _parse_tile_type_call(self, call: ast.Call):
    kwargs = {}
    for kw in call.keywords:
        # TileType shape / valid_shape must be compile-time constants. A runtime
        # scalar var would leak its name into the generated C++ tile declaration
        # (which is hoisted to the function prologue, above where the var is even
        # defined). Reject it here with a clear error; use pl.set_validshape() for
        # a runtime valid shape.
        if kw.arg in ("shape", "valid_shape") and isinstance(kw.value, ast.List):
            kwargs[kw.arg] = [
                self.require_const_value(
                    self.parse_expression(elt),
                    elt,
                    expects="integer",
                    hint="A constant list kwarg (e.g. TileType shape/valid_shape) must be compile-time "
                    "constants; use pl.set_validshape() for a runtime valid shape.",
                    check=_is_int,
                )
                for elt in kw.value.elts
            ]
        else:
            kwargs[kw.arg] = self.resolve_single_kwarg(kw.arg, kw.value)
    return TileType(**kwargs)


def _resolve_make_tile_memref(self, call: ast.Call, kwargs: dict, span: Span) -> dict:
    """Resolve the addr/size pair that places a tile in its memory space.

    ``addr`` is required and fixed while parsing, the same contract
    make_tile_group() enforces for addrs. Without the check a missing addr
    yields a tile with no MemRef, which codegen happily emits — the kernel
    builds and runs, reading and writing whatever happens to sit there.
    ``size`` is optional (it is derived from the tile type), but when given it is
    held to the same compile-time contract.

    Both arrive as keywords, so they are already unwrapped to Python values here;
    they still go through ``require_const_value`` — the same accessor
    make_tile_group() uses — because a runtime expression unwraps to an ir.Expr,
    which an int annotation does not keep out.
    """
    from pypto_pro.language.parser.diagnostics import FinalRejectionError

    memref: dict[str, int] = {}
    for key in ("addr", "size"):
        if key not in kwargs:
            continue
        value = kwargs[key]
        if value is None:  # a literal addr=None parses to an ir.Var and is rejected below
            continue
        memref[key] = self.require_const_value(
            value,
            self._kwarg_node(call, key),
            key=key,
            expects="integer",
            check=_is_int,
            hint=f"{key} is fixed while parsing; pass a literal, or a variable bound to "
            "literals/constants — not a runtime value such as a tensor shape or loop index",
        )
    if "addr" not in memref:
        raise FinalRejectionError("pl.make_tile() missing required keyword 'addr'", span=span, hint=_MAKE_TILE_HINT)
    if "size" in memref and memref["size"] <= 0:
        raise FinalRejectionError(
            f"pl.make_tile() 'size' must be a positive byte count, got {memref['size']}",
            span=span,
            hint="Drop 'size' to reserve the tile type's own footprint, or pass the larger "
            "byte count the layout needs (an NZ/ZN tile rounded up to whole fractals)",
        )

    # addr/size move back as plain ints, so make_tile_expr() runs its alignment
    # check on them rather than on whatever expression the call site wrote.
    return {**kwargs, **memref}


@op_impl("make_tile")
def _parse_make_tile(self, call: ast.Call) -> Expr:
    from pypto_pro.language.parser.diagnostics import FinalRejectionError

    span = self.span_tracker.get_span(call)
    # Only the first positional is parsed: the gate below rejects the rest, and its
    # message should win over whatever parsing an unexpected expression would say.
    tile_type = self.parse_expression(call.args[0]) if call.args else None
    kwargs = self.parse_op_kwargs(call)

    if not isinstance(tile_type, TileType):
        got = ast.unparse(call.args[0]) if call.args else "no positional argument"
        raise FinalRejectionError(
            f"pl.make_tile() takes a pl.TileType as its first argument, got {got}",
            span=span,
            hint=_MAKE_TILE_TYPE_HINT,
        )
    if len(call.args) > 1:
        # addr/size name a placement, not an operand, and a swapped pair of bare ints
        # reads exactly like a correct one — so they are keywords, as in make_tile_group().
        # The values are not echoed: ast.unparse would print 0x40 back as 64, which reads
        # badly beside the source line the diagnostic already quotes.
        raise FinalRejectionError(
            f"pl.make_tile() takes 1 positional argument (the tile type) but {len(call.args)} were given",
            span=span,
            hint="pass addr and size as keywords, e.g. pl.make_tile(tile_type, addr=0x0, size=1024)",
        )

    # Every TileType field names a make_tile_expr() parameter, so the type is spread
    # over the kwargs it did not already carry; an explicit kwarg wins over it.
    for field in fields(tile_type):
        value = getattr(tile_type, field.name)
        if value is not None:
            kwargs.setdefault(field.name, value)

    kwargs = _resolve_make_tile_memref(self, call, kwargs, span)
    return make_tile_expr(**kwargs, span=span)


# ---------------------------------------------------------------------------
# reinterpret — Tile/TileGroup metadata reinterpretation (zero data movement)
# ---------------------------------------------------------------------------


def _span_msg(span: "Span | None") -> str:
    return f" at {span}" if span else ""


def _hw_attr(hw, field: str) -> "int | None":
    """One hardware_info field as an int, or None when unset/negative sentinel."""
    if hw is None:
        return None
    value = getattr(hw, field, None)
    if value is None:
        return None
    value = int(value)
    return value if value >= 0 else None


def _reinterpret_footprint_check(shape: list[int], dtype: DataType, size: int, span: "Span | None") -> None:
    """The new footprint must not exceed the original buffer span (hard constraint)."""
    try:
        footprint = tile_slot_size(shape, dtype)
    except ValueError:
        return
    if footprint > size:
        raise ValueError(
            f"reinterpret: new footprint {footprint} bytes exceeds the original buffer size {size} bytes; "
            f"a reinterpretation may only reuse the buffer at the same size or smaller{_span_msg(span)}"
        )


def _reinterpret_align_check(addr: int, dtype: DataType, span: "Span | None") -> None:
    """The new element width must divide the base address (element-boundary view)."""
    elem_bytes = max(1, (int(dtype.get_bit()) + 7) // 8)
    if addr % elem_bytes != 0:
        raise ValueError(
            f"reinterpret: address 0x{addr:X} is not aligned to the new element size {elem_bytes} bytes"
            f"{_span_msg(span)}"
        )


def _memref_const_addr(memref, span: "Span | None") -> "int | None":
    """The compile-time address of a tile's MemRef, or None when it is not constant."""
    addr_expr = getattr(memref, "addr", None)
    if addr_expr is None:
        return None
    value = getattr(addr_expr, "value", None)
    if not isinstance(value, int):
        return None
    return value


def _ir_reinterpret(
    src: Expr,
    *,
    shape: "Sequence[int] | None" = None,
    dtype: "DataType | None" = None,
    layout: "TensorLayout | None" = None,
    span: "Span | None" = None,
) -> Expr:
    """Build a make_tile call re-declaring the tile's dtype/shape/layout.

    The new tile is re-allocated at the SAME address and size as the source
    (SameAllocation semantics keep the "same buffer" identity), with valid_shape
    NOT inherited (dynamic / -1) and fractal/pad/compact inherited verbatim.
    No MemRef is shared: this path reuses make_tile_expr, so the id counter
    simply advances and the new MemRef is a fresh object.
    """
    actual_span = span or _span()
    if not isinstance(src.type, _IRTileType):
        raise TypeError(f"reinterpret: expected a Tile, got {type(src.type).__name__}")
    t = src.type
    memref = getattr(t, "memref", None)
    if memref is None:
        raise ValueError(
            f"reinterpret: source tile has no MemRef (bind it with pl.make_tile first){_span_msg(actual_span)}"
        )

    old_shape = _static_shape_ints(t.shape, "reinterpret: source")
    hw = getattr(t, "hardware_info", None)
    old_layout = _tile_layout(t)

    new_shape = list(shape) if shape is not None else old_shape
    new_dtype = dtype if dtype is not None else t.dtype
    new_layout = layout if layout is not None else old_layout

    addr = _memref_const_addr(memref, actual_span)
    if addr is None:
        raise ValueError(f"reinterpret: source tile address must be a compile-time constant{_span_msg(actual_span)}")
    _reinterpret_align_check(addr, new_dtype, actual_span)
    _reinterpret_footprint_check(new_shape, new_dtype, int(memref.size), actual_span)

    # Fresh MemRef at the same address & size (new memref_id); valid_shape
    # intentionally omitted so the new handle starts dynamic (-1).
    return make_tile_expr(
        new_shape,
        new_dtype,
        memref.memory_space,
        addr=addr,
        size=int(memref.size),
        valid_shape=None,
        layout=new_layout,
        fractal=_hw_attr(hw, "fractal"),
        pad=_hw_attr(hw, "pad"),
        compact=_hw_attr(hw, "compact"),
        span=actual_span,
    )


@op_impl("reinterpret")
def _parse_reinterpret(self, call: ast.Call) -> Expr:
    from pypto_pro.language.parser.diagnostics import FinalRejectionError

    span = self.span_tracker.get_span(call)
    if len(call.args) != 1:
        raise FinalRejectionError(
            f"pl.reinterpret() takes exactly 1 positional argument (a Tile or TileGroup), got {len(call.args)}",
            span=span,
            hint="Usage: t2 = pl.reinterpret(tile, dtype=..., shape=..., layout=...)",
        )
    src = self.parse_expression(call.args[0])
    kwargs = self.parse_op_kwargs(call)

    if kwargs.get("dtype") is None and kwargs.get("shape") is None and kwargs.get("layout") is None:
        raise FinalRejectionError(
            "pl.reinterpret() requires at least one of dtype/shape/layout to override",
            span=span,
            hint="e.g. pl.reinterpret(tile, shape=[64, 64], dtype=pl.DT_BF16)",
        )

    if kwargs.get("dtype") is not None and kwargs.get("shape") is None:
        raise FinalRejectionError(
            "pl.reinterpret() 'shape' is required when 'dtype' changes",
            span=span,
            hint="state the new element count explicitly: pl.reinterpret(tile, dtype=..., shape=[rows, cols])",
        )

    new_shape = kwargs.get("shape")
    if new_shape is not None and not (
        isinstance(new_shape, (list, tuple)) and len(new_shape) > 0 and all(_is_int(d) for d in new_shape)
    ):
        raise FinalRejectionError(
            "pl.reinterpret() 'shape' must be a non-empty list of compile-time integers",
            span=span,
            hint="e.g. shape=[32, 128]; runtime shapes are not supported — use pl.set_validshape() for runtime windows",
        )

    # All override args are validated once above, before dispatch: neither the
    # tile branch nor the tile_group branch re-validates them. The "is a Tile"
    # check lives in _ir_reinterpret, which both branches go through.
    if self.is_tile_group(src):
        return self._build_reinterpreted_group(src, kwargs, span)

    new_tile = _ir_reinterpret(src, shape=new_shape, dtype=kwargs.get("dtype"), layout=kwargs.get("layout"), span=span)

    # The reinterpreted tile aliases the same buffer, so it inherits the source's
    # mutex binding (same propagation pattern as GetItemExpr in _expression_parser).
    mm = self._tile_mutex_meta.get(src)
    if mm is not None:
        self._tile_mutex_meta[new_tile] = mm
    return new_tile


def _resolve_order_kwarg(self, call: ast.Call, kwargs: dict) -> None:
    """Resolve the ``order`` axis list, which selects tensor axes while parsing.

    A pre-hook runs after ``_make_handler`` parsed the kwargs, so ``order`` is
    already a parse-time value here; only the AST node is fetched, for the
    diagnostic.
    """
    if "order" not in kwargs:
        return
    kwargs["order"] = list(
        self.require_const_value(
            kwargs["order"],
            self._kwarg_node(call, "order"),
            key="order",
            expects="integer list",
            hint="order selects tensor axes at compile time; pass a constant list "
            "(e.g. order=[1, 3]) or a variable bound to one, not a runtime value.",
            check=lambda value: isinstance(value, tuple)
            and all(isinstance(axis, int) and not isinstance(axis, bool) for axis in value),
        )
    )


def _static_shape_ints(shape, what: str) -> list[int]:
    """Resolve a tile/tensor shape to plain ints (compile-time dims only).

    Shared by scale-tile validation and pl.reinterpret's source-shape reader;
    raises ValueError when a dimension is not statically known.
    """
    ints = []
    for dim in shape:
        if isinstance(dim, ConstInt):
            ints.append(int(dim.value))
        elif isinstance(dim, int):
            ints.append(dim)
        else:
            raise ValueError(f"{what} shape must contain static integer dimensions, got {type(dim)}")
    return ints


def _validate_fp_shape_dtype(fp_shape_ints: list[int], scale_dtype: DataType, what: str) -> None:
    """Validate a per-channel scale Tile against the FixPipe deqTensor constraints.

    Hardware FixPipe deqTensor (pto-isa TMov.hpp): row must be 1;
    col * sizeof(int64) 128B-aligned (col % 16 == 0); col * sizeof(int64) <= 4KB FB (col <= 512).
    ``[N, 1]`` per-row scaling is NOT supported — it fails at compile time with
    "TMov: When TileType is Scaling, row must be 1."
    """
    if len(fp_shape_ints) != 2:
        raise ValueError(f"{what} must be 2D, got shape {fp_shape_ints}")
    if fp_shape_ints[0] != 1:
        raise ValueError(
            f"{what} must have shape [1, N] (row == 1), got [{fp_shape_ints[0]}, {fp_shape_ints[1]}]. "
            f"Hardware FixPipe deqTensor only supports per-column scaling ([1, N]); "
            f"[N, 1] per-row scaling is not supported. "
            f"Per-token (row-wise) quantization must be done in the Vector (UB) domain."
        )
    if fp_shape_ints[1] % 16 != 0:
        raise ValueError(
            f"{what} [1, N]: N must be a multiple of 16 (128B alignment for INT64), "
            f"got [{fp_shape_ints[0]}, {fp_shape_ints[1]}]"
        )
    if fp_shape_ints[1] > 512:
        raise ValueError(
            f"{what} [1, N]: N must be <= 512 (4KB fixpipe buffer limit for INT64), "
            f"got [{fp_shape_ints[0]}, {fp_shape_ints[1]}]"
        )
    if scale_dtype == DataType.FP32:
        raise ValueError(
            f"{what} dtype FP32 is not supported. "
            f"Please convert FP32 scale to INT64 using torch_npu.npu_trans_quant_param() before passing to kernel. "
            f"Example: scale_int64 = torch_npu.npu_trans_quant_param(scale_fp32.npu())"
        )
    if scale_dtype != DataType.INT64:
        raise ValueError(
            f"{what} dtype must be INT64, got {scale_dtype}. "
            f"For FP32 scale, use torch_npu.npu_trans_quant_param() to convert to INT64."
        )


def _auto_alloc_scaling_tile_hook(self, call: ast.Call, kwargs: dict) -> None:
    """Pre-hook: validate a user-prepared Scaling tile when scale is a Tile (per-channel quantization).

    Per-channel quantization requires a user-prepared deqTensor tile: the user
    builds a Scaling tile (MemorySpace.Scaling, [1, N] INT64), owns the data
    flow (load -> move -> sync MTE1->FIX before the store/move), and passes it
    as ``scale``. This hook only validates it — no auto-allocation of
    Scaling/Mat tiles and no sync events are emitted. The validated tile stays
    in the ``scale`` kwarg and is resolved by ``_resolve_scale_param`` in the
    builder to the store_fp/move_fp operand.

    A GM Tensor scale is rejected at parse time: the automatic per-channel path
    (auto-allocated Mat intermediate + auto sync events) has been removed; users
    must prepare the Scaling tile themselves.

    Scale tile shape constraint:
    - Hardware FixPipe deqTensor only supports per-column scaling: scale tile must be
      ``[1, N]`` (row == 1), with ``N % 16 == 0`` (128B alignment) and ``N <= 512`` (4KB
      fixpipe buffer). ``[N, 1]`` per-row scaling is rejected here with a clear error
      (hardware would otherwise fail compilation with "TMov: row must be 1").

    FP32 scale tile limitation:
    - Only INT64 scale tiles are supported; FP32 scale tensors must be converted
      on the host side via ``torch_npu.npu_trans_quant_param`` before being loaded
      into the Scaling tile.
    """
    scale = kwargs.get("scale")
    if scale is None:
        return
    if not isinstance(scale, Expr):
        return
    scale_type = getattr(scale, "type", None)

    if isinstance(scale_type, _ir_core.TileType):
        mem = getattr(getattr(scale_type, "memref", None), "memory_space", None)
        if mem != MemorySpace.Scaling:
            raise ValueError(
                f"scale Tile must be allocated in MemorySpace.Scaling for per-channel quantization, got {mem}"
            )
        fp_shape_ints = _static_shape_ints(scale_type.shape, "scale tile")
        _validate_fp_shape_dtype(fp_shape_ints, scale_type.dtype, "scale tile")
        return

    if isinstance(scale_type, _ir_core.TensorType):
        raise ValueError(
            "scale Tensor is not supported for per-channel quantization — pass a "
            "user-prepared Scaling Tile (MemorySpace.Scaling, shape [1, N], INT64) "
            "instead, and ensure it is ready (load -> move -> sync MTE1->FIX) before "
            "the store/move"
        )

    return  # scalar / runtime bits -> handled by _resolve_scale_param


# ---------------------------------------------------------------------------
# Builder helpers for merged interfaces
# ---------------------------------------------------------------------------


def _create_tile_scalar_op(
    out: Expr, lhs: Expr, rhs: Expr, *, tile_op: str, scalar_op: str, span: Span | None = None, **kwargs
) -> Expr:
    """Dispatch to tile-tile or tile-scalar IR op based on rhs type."""
    if isinstance(getattr(rhs, "type", None), _ir_core.TileType):
        target_op = block_ir_op(tile_op)
    else:
        from pypto_pro.language.parser.diagnostics import check_const_expr_fits_dtype

        target_op = block_ir_op(scalar_op)
        out_dtype = getattr(getattr(out, "type", None), "dtype", None)
        check_const_expr_fits_dtype(rhs, out_dtype, span=span, api=f"pl.{tile_op}")
    return _ir_core.create_op_call(target_op, [out, lhs, rhs], kwargs, span)


def _create_dim_op(args: list[Expr], *, row_op: str, col_op: str, dim: int = 0, span: Span | None = None) -> Expr:
    """Dispatch to row-wise or col-wise IR op based on dim."""
    ir_name = row_op if dim == 0 else col_op
    return _ir_core.create_op_call(block_ir_op(ir_name), args, {}, span)


def _ir_select(out: Expr, mask: Expr, lhs: Expr, rhs: Expr, tmp: Expr, *, span: Span | None = None) -> Expr:
    if isinstance(getattr(rhs, "type", None), _ir_core.TileType):
        return _ir_sel(out, mask, lhs, rhs, tmp, span=span)
    return _ir_sels(out, mask, lhs, tmp, rhs, span=span)


def _ir_eq(out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None) -> Expr:
    if isinstance(getattr(rhs, "type", None), _ir_core.TileType):
        return _ir_cmp(out, lhs, rhs, span=span, cmp_mode=CmpMode.EQ.value)
    return _ir_cmps(out, lhs, rhs, span=span, cmp_mode=CmpMode.EQ.value)


def _ir_ne(out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None) -> Expr:
    if isinstance(getattr(rhs, "type", None), _ir_core.TileType):
        return _ir_cmp(out, lhs, rhs, span=span, cmp_mode=CmpMode.NE.value)
    return _ir_cmps(out, lhs, rhs, span=span, cmp_mode=CmpMode.NE.value)


def _ir_lt(out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None) -> Expr:
    if isinstance(getattr(rhs, "type", None), _ir_core.TileType):
        return _ir_cmp(out, lhs, rhs, span=span, cmp_mode=CmpMode.LT.value)
    return _ir_cmps(out, lhs, rhs, span=span, cmp_mode=CmpMode.LT.value)


def _ir_le(out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None) -> Expr:
    if isinstance(getattr(rhs, "type", None), _ir_core.TileType):
        return _ir_cmp(out, lhs, rhs, span=span, cmp_mode=CmpMode.LE.value)
    return _ir_cmps(out, lhs, rhs, span=span, cmp_mode=CmpMode.LE.value)


def _ir_gt(out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None) -> Expr:
    if isinstance(getattr(rhs, "type", None), _ir_core.TileType):
        return _ir_cmp(out, lhs, rhs, span=span, cmp_mode=CmpMode.GT.value)
    return _ir_cmps(out, lhs, rhs, span=span, cmp_mode=CmpMode.GT.value)


def _ir_ge(out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None) -> Expr:
    if isinstance(getattr(rhs, "type", None), _ir_core.TileType):
        return _ir_cmp(out, lhs, rhs, span=span, cmp_mode=CmpMode.GE.value)
    return _ir_cmps(out, lhs, rhs, span=span, cmp_mode=CmpMode.GE.value)


def _ir_sum(out: Expr, src: Expr, tmp: Expr, *, span: Span | None = None, dim: int = 0) -> Expr:
    return _create_dim_op([out, src, tmp], row_op="row_sum", col_op="col_sum", dim=dim, span=span)


def _ir_argmax(out: Expr, src: Expr, tmp: Expr, *, span: Span | None = None, dim: int = 0) -> Expr:
    return _create_dim_op([out, src, tmp], row_op="row_argmax", col_op="col_argmax", dim=dim, span=span)


def _ir_argmin(out: Expr, src: Expr, tmp: Expr, *, span: Span | None = None, dim: int = 0) -> Expr:
    return _create_dim_op([out, src, tmp], row_op="row_argmin", col_op="col_argmin", dim=dim, span=span)


def _ir_expand_max(out: Expr, src: Expr, scalar: Expr, *, span: Span | None = None, dim: int = 0) -> Expr:
    return _create_dim_op([out, src, scalar], row_op="row_expand_max", col_op="col_expand_max", dim=dim, span=span)


def _ir_expand_min(out: Expr, src: Expr, scalar: Expr, *, span: Span | None = None, dim: int = 0) -> Expr:
    return _create_dim_op([out, src, scalar], row_op="row_expand_min", col_op="col_expand_min", dim=dim, span=span)


def _ir_expand_mul(out: Expr, src: Expr, scalar: Expr, *, span: Span | None = None, dim: int = 0) -> Expr:
    return _create_dim_op([out, src, scalar], row_op="row_expand_mul", col_op="col_expand_mul", dim=dim, span=span)


def _ir_expand_sub(out: Expr, src: Expr, scalar: Expr, *, span: Span | None = None, dim: int = 0) -> Expr:
    return _create_dim_op([out, src, scalar], row_op="row_expand_sub", col_op="col_expand_sub", dim=dim, span=span)


def _ir_expand_div(out: Expr, src: Expr, scalar: Expr, *, span: Span | None = None, dim: int = 0) -> Expr:
    return _create_dim_op([out, src, scalar], row_op="row_expand_div", col_op="col_expand_div", dim=dim, span=span)


register_table(
    {
        # args + kwargs -> builder
        "store_fp": OpSpec(builder=_ir_store_fp),
        "move": OpSpec(builder=_ir_move, pre_hooks=[_auto_alloc_scaling_tile_hook]),
        "insert": OpSpec(builder=_ir_insert),
        "getval": OpSpec(builder=_ir_getval),
        "setval": OpSpec(builder=_ir_setval),
        "transpose": OpSpec(builder=_ir_transpose),
        "cast": OpSpec(builder=_ir_cast),
        "fillpad": OpSpec(builder=_ir_fillpad),
        "add_relu_cast": OpSpec(builder=_ir_add_relu_cast),
        "sub_relu_cast": OpSpec(builder=_ir_sub_relu_cast),
        "mul_cast": OpSpec(builder=_ir_mul_cast),
        "set_vec_mask": OpSpec(builder=_ir_set_vec_mask),
        "quant": OpSpec(builder=_ir_quant),
        "dequant": OpSpec(builder=_ir_dequant),
        "ssbuf_store": OpSpec(builder=_ir_ssbuf_store),
        "ssbuf_load": OpSpec(builder=_ir_ssbuf_load),
        # unary / scalar / fused compute ops
        "neg": OpSpec(builder=_ir_neg),
        "abs": OpSpec(builder=_ir_abs),
        "relu": OpSpec(builder=_ir_relu),
        "axpy": OpSpec(builder=_ir_axpy),
        "add_relu": OpSpec(builder=_ir_add_relu),
        "xor": OpSpec(builder=_ir_xor),
        "expands": OpSpec(builder=_ir_expands),
        "row_sum": OpSpec(builder=_ir_row_sum),
        "col_sum": OpSpec(builder=_ir_col_sum),
        # tile-tile / tile-scalar dispatch
        "select": OpSpec(builder=_ir_select),
        "eq": OpSpec(builder=_ir_eq),
        "ne": OpSpec(builder=_ir_ne),
        "lt": OpSpec(builder=_ir_lt),
        "le": OpSpec(builder=_ir_le),
        "gt": OpSpec(builder=_ir_gt),
        "ge": OpSpec(builder=_ir_ge),
        "sum": OpSpec(builder=_ir_sum),
        "argmax": OpSpec(builder=_ir_argmax),
        "argmin": OpSpec(builder=_ir_argmin),
        "expand_max": OpSpec(builder=_ir_expand_max),
        "expand_min": OpSpec(builder=_ir_expand_min),
        "expand_mul": OpSpec(builder=_ir_expand_mul),
        "expand_sub": OpSpec(builder=_ir_expand_sub),
        "expand_div": OpSpec(builder=_ir_expand_div),
    # args + kwargs + order hook (load) / order hook + scaling tile hook (store)
    "load": OpSpec(builder=_ir_load, pre_hooks=[_resolve_order_kwarg]),
    "load_tile": OpSpec(builder=_ir_load_tile, pre_hooks=[_resolve_order_kwarg]),
    "store": OpSpec(builder=_ir_store, pre_hooks=[_auto_alloc_scaling_tile_hook, _resolve_order_kwarg]),
    "store_tile": OpSpec(builder=_ir_store_tile, pre_hooks=[_auto_alloc_scaling_tile_hook, _resolve_order_kwarg]),
        # kwargs only
        "set_mask_count": OpSpec(builder=_ir_set_mask_count, parse_args=False),
        "set_mask_norm": OpSpec(builder=_ir_set_mask_norm, parse_args=False),
        "reset_mask": OpSpec(builder=_ir_reset_mask, parse_args=False),
    }
)


@op_impl("set_validshape")
def _parse_set_validshape(self, call: ast.Call) -> Expr:
    span = self.span_tracker.get_span(call)
    args = [self.parse_expression(arg) for arg in call.args]
    kwargs = self.parse_op_kwargs(call)

    if args and self.is_tile_group(args[0]):
        group_var = args[0]
        shape = args[1] if len(args) > 1 else None

        # tile_group_meta is keyed by the handle expression itself, not by id():
        # an id() lookup never hits, silently collapsing the group to one tile.
        meta = self.tile_group_meta.get(group_var, (1, None))
        n_tiles = meta[0]
        tiles = self.lower_attr_access(group_var, "tiles", span)

        for i in range(n_tiles):
            tile_ir = _ir_core.GetItemExpr(tiles, ConstInt(i, DataType.INDEX, span), span)
            vs_call = _ir_set_validshape(tile_ir, shape, span=span)
            self.builder.emit(_ir_core.EvalStmt(vs_call, span))

        return ConstInt(0, DataType.INDEX, span)

    result = _ir_set_validshape(*args, **kwargs, span=span)

    if args and hasattr(self, 'record_tile_valid_shape') and len(args) >= 2:
        self.record_tile_valid_shape(args[0], args[1])
    return result


def _check_tile_memory_space(
    op_name: str, operand_name: str, expr: Expr, expected: MemorySpace, expected_desc: str
) -> None:
    mem = getattr(getattr(getattr(expr, "type", None), "memref", None), "memory_space_", None)
    if mem is not None and mem != expected:
        raise ValueError(f"{op_name}: {operand_name} must be in {expected_desc}, got {mem.name}")


def _validate_acc_phase(op_name: str, phase: AccPhase | None) -> None:
    if phase is not None and not isinstance(phase, AccPhase):
        raise ValueError(f"{op_name}: invalid phase value {phase!r}, expected AccPhase")


def _ir_matmul(dst: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None, phase: AccPhase | None = None) -> Expr:
    actual_span = span or _span()
    _validate_acc_phase("matmul", phase)
    _check_tile_memory_space("matmul", "dst_tile", dst, MemorySpace.Acc, "L0C (Acc)")
    _check_tile_memory_space("matmul", "lhs_tile", lhs, MemorySpace.Left, "L0A (Left)")
    _check_tile_memory_space("matmul", "rhs_tile", rhs, MemorySpace.Right, "L0B (Right)")
    kwargs: dict[str, Any] = {}
    if phase is not None:
        kwargs["phase"] = phase
    return _ir_core.create_op_call(block_ir_op("matmul"), [dst, lhs, rhs], kwargs, actual_span)


def _ir_matmul_acc(
    dst: Expr, acc: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None, phase: AccPhase | None = None
) -> Expr:
    actual_span = span or _span()
    _validate_acc_phase("matmul_acc", phase)
    _check_tile_memory_space("matmul_acc", "dst_tile", dst, MemorySpace.Acc, "L0C (Acc)")
    _check_tile_memory_space("matmul_acc", "acc_tile", acc, MemorySpace.Acc, "L0C (Acc)")
    _check_tile_memory_space("matmul_acc", "lhs_tile", lhs, MemorySpace.Left, "L0A (Left)")
    _check_tile_memory_space("matmul_acc", "rhs_tile", rhs, MemorySpace.Right, "L0B (Right)")
    kwargs: dict[str, Any] = {}
    if phase is not None:
        kwargs["phase"] = phase
    return _ir_core.create_op_call(block_ir_op("matmul_acc"), [dst, acc, lhs, rhs], kwargs, actual_span)


def _ir_matmul_bias(
    dst: Expr, lhs: Expr, rhs: Expr, bias: Expr, *, span: Span | None = None, phase: AccPhase | None = None
) -> Expr:
    actual_span = span or _span()
    _validate_acc_phase("matmul_bias", phase)
    _check_tile_memory_space("matmul_bias", "dst_tile", dst, MemorySpace.Acc, "L0C (Acc)")
    _check_tile_memory_space("matmul_bias", "lhs_tile", lhs, MemorySpace.Left, "L0A (Left)")
    _check_tile_memory_space("matmul_bias", "rhs_tile", rhs, MemorySpace.Right, "L0B (Right)")
    _check_tile_memory_space("matmul_bias", "bias_tile", bias, MemorySpace.Bias, "L0B (Bias)")
    kwargs: dict[str, Any] = {}
    if phase is not None:
        kwargs["phase"] = phase
    return _ir_core.create_op_call(block_ir_op("matmul_bias"), [dst, lhs, rhs, bias], kwargs, actual_span)


@op_impl("matmul")
def _parse_matmul(self, call: ast.Call) -> Expr:
    span = self.span_tracker.get_span(call)
    args = [self.parse_expression(arg) for arg in call.args]
    kwargs = self.parse_op_kwargs(call)
    if len(args) == 4:
        return _ir_matmul_bias(*args, **kwargs, span=span)
    return _ir_matmul(*args, **kwargs, span=span)


@op_impl("matmul_acc")
def _parse_matmul_acc(self, call: ast.Call) -> Expr:
    span = self.span_tracker.get_span(call)
    args = [self.parse_expression(arg) for arg in call.args]
    kwargs = self.parse_op_kwargs(call)
    return _ir_matmul_acc(*args, **kwargs, span=span)


_MX_FP8_DTYPES = (DataType.FP8E4M3FN, DataType.FP8E5M2)
_MX_FP4_DTYPES = (DataType.FP4E2M1, DataType.FP4E1M2)
_MX_GROUP_SIZE = 32
_MX_SCALE_ADDR_SHIFT = 4


def _check_mx_scale_tile(
    op_name: str,
    scale: Expr | None,
    data: Expr,
    *,
    is_left: bool,
) -> None:
    """Validate an MX scale tile and its shape/address relationship to the paired data tile."""
    if is_left:
        scale_name, data_name, group_axis = "scale_a", "lhs_tile", 1
        expected_space, expected_desc = MemorySpace.ScaleLeft, "L0A (ScaleLeft)"
    else:
        scale_name, data_name, group_axis = "scale_b", "rhs_tile", 0
        expected_space, expected_desc = MemorySpace.ScaleRight, "L0B (ScaleRight)"

    if scale is None:
        raise ValueError(f"{op_name}: {scale_name} is required")
    scale_type = getattr(scale, "type", None)
    if not isinstance(scale_type, _IRTileType):
        raise ValueError(f"{op_name}: {scale_name} must be a Tile")
    if scale_type.dtype != DataType.FP8E8M0:
        raise ValueError(f"{op_name}: {scale_name} must use FP8E8M0 dtype, got {scale_type.dtype}")
    _check_tile_memory_space(op_name, scale_name, scale, expected_space, expected_desc)

    data_type = data.type
    data_shape = _tile_shape_ints(data_type)
    scale_shape = _tile_shape_ints(scale_type)
    if data_shape is not None and scale_shape is not None:
        expected_shape = data_shape.copy()
        expected_shape[group_axis] //= _MX_GROUP_SIZE
        if scale_shape != expected_shape:
            raise ValueError(
                f"{op_name}: {scale_name} shape must match {data_name} MX groups, "
                f"expected {expected_shape} for {data_name} shape {data_shape}, got {scale_shape}."
            )

    # A5 mad_mx receives only the L0A/L0B data pointers and derives the
    # corresponding L0AMX/L0BMX address by shifting right four bits.  Skip
    # auto-allocated tiles or non-constant addresses, which cannot be proven here.
    data_memref = data_type.memref
    scale_memref = scale_type.memref
    if data_memref is None or scale_memref is None:
        return

    data_addr = data_memref.addr
    scale_addr = scale_memref.addr
    if not isinstance(data_addr, ConstInt) or not isinstance(scale_addr, ConstInt):
        return

    expected_scale_addr = data_addr.value >> _MX_SCALE_ADDR_SHIFT
    if scale_addr.value != expected_scale_addr:
        raise ValueError(
            f"{op_name}: {scale_name} address must equal {data_name} address >> {_MX_SCALE_ADDR_SHIFT}, "
            f"got {data_name}=0x{data_addr.value:X}, expected {scale_name}=0x{expected_scale_addr:X}, "
            f"actual {scale_name}=0x{scale_addr.value:X}."
        )


def _check_mx_operands(
    op_name: str,
    dst: Expr,
    lhs: Expr,
    rhs: Expr,
    scale_a: Expr,
    scale_b: Expr,
    acc: Expr | None = None,
) -> None:
    """Run the common declared-type and shape checks for MX matmul variants."""
    _check_tile_memory_space(op_name, "dst_tile", dst, MemorySpace.Acc, "L0C (Acc)")
    _check_tile_memory_space(op_name, "lhs_tile", lhs, MemorySpace.Left, "L0A (Left)")
    _check_tile_memory_space(op_name, "rhs_tile", rhs, MemorySpace.Right, "L0B (Right)")
    if acc is not None:
        _check_tile_memory_space(op_name, "acc_tile", acc, MemorySpace.Acc, "L0C (Acc)")

    a_dtype, b_dtype, dst_dtype = lhs.type.dtype, rhs.type.dtype, dst.type.dtype
    valid_input_dtypes = (a_dtype in _MX_FP8_DTYPES and b_dtype in _MX_FP8_DTYPES) or (
        a_dtype in _MX_FP4_DTYPES and b_dtype in _MX_FP4_DTYPES
    )
    if not valid_input_dtypes or dst_dtype != DataType.FP32:
        raise ValueError(
            f"{op_name}: (lhs,rhs) must be FP8/FP4 combo and dst FP32, "
            f"got ({a_dtype},{b_dtype},{dst_dtype})."
        )

    dst_shape = _tile_shape_ints(dst.type)
    lhs_shape = _tile_shape_ints(lhs.type)
    rhs_shape = _tile_shape_ints(rhs.type)
    lhs_k = lhs_shape[1] if lhs_shape is not None else None
    rhs_k = rhs_shape[0] if rhs_shape is not None else None

    if lhs_k is not None and rhs_k is not None and lhs_k != rhs_k:
        raise ValueError(f"{op_name}: lhs and rhs K dimensions must match, got lhs K={lhs_k}, rhs K={rhs_k}.")
    for name, k_value in (("lhs", lhs_k), ("rhs", rhs_k)):
        if k_value is not None and k_value % 64 != 0:
            raise ValueError(
                f"{op_name}: K dimension must be a multiple of 64 for MX matmul, got {name} K={k_value}."
            )

    if dst_shape is not None and lhs_shape is not None and rhs_shape is not None:
        expected_shape = [lhs_shape[0], rhs_shape[1]]
        if dst_shape != expected_shape:
            raise ValueError(
                f"{op_name}: dst_tile shape must be [lhs M, rhs N], "
                f"expected {expected_shape}, got {dst_shape}."
            )
    if acc is not None:
        acc_shape = _tile_shape_ints(acc.type)
        if dst_shape is not None and acc_shape is not None and acc_shape != dst_shape:
            raise ValueError(
                f"{op_name}: acc_tile shape must match dst_tile shape, "
                f"got acc_tile={acc_shape}, dst_tile={dst_shape}."
            )
        if acc.type.dtype != DataType.FP32:
            raise ValueError(f"{op_name}: acc_tile must use FP32 dtype, got {acc.type.dtype}.")

    _check_mx_scale_tile(op_name, scale_a, lhs, is_left=True)
    _check_mx_scale_tile(op_name, scale_b, rhs, is_left=False)


def _ir_matmul_mx(
    dst: Expr, lhs: Expr, rhs: Expr, scale_a: Expr, scale_b: Expr,
    *, span: Span | None = None, phase: AccPhase | None = None
) -> Expr:
    actual_span = span or _span()
    _validate_acc_phase("matmul_mx", phase)
    _check_mx_operands("matmul_mx", dst, lhs, rhs, scale_a, scale_b)
    kwargs: dict[str, Any] = {}
    if phase is not None:
        kwargs["phase"] = phase
    return _ir_core.create_op_call(
        block_ir_op("matmul_mx"), [dst, lhs, rhs, scale_a, scale_b], kwargs, actual_span
    )


def _ir_matmul_mx_acc(
    dst: Expr, acc: Expr, lhs: Expr, rhs: Expr, scale_a: Expr, scale_b: Expr,
    *, span: Span | None = None, phase: AccPhase | None = None
) -> Expr:
    actual_span = span or _span()
    _validate_acc_phase("matmul_mx_acc", phase)
    _check_mx_operands("matmul_mx_acc", dst, lhs, rhs, scale_a, scale_b, acc)
    kwargs: dict[str, Any] = {}
    if phase is not None:
        kwargs["phase"] = phase
    return _ir_core.create_op_call(
        block_ir_op("matmul_mx_acc"), [dst, acc, lhs, rhs, scale_a, scale_b], kwargs, actual_span
    )


@op_impl("matmul_mx")
def _parse_matmul_mx(self, call: ast.Call) -> Expr:
    span = self.span_tracker.get_span(call)
    args = [self.parse_expression(arg) for arg in call.args]
    kwargs = self.parse_op_kwargs(call)
    return _ir_matmul_mx(*args, **kwargs, span=span)


@op_impl("matmul_mx_acc")
def _parse_matmul_mx_acc(self, call: ast.Call) -> Expr:
    span = self.span_tracker.get_span(call)
    args = [self.parse_expression(arg) for arg in call.args]
    kwargs = self.parse_op_kwargs(call)
    return _ir_matmul_mx_acc(*args, **kwargs, span=span)


def _to_const_int(val: Any, span: Span | None = None) -> Expr:
    if isinstance(val, Expr):
        return val
    if isinstance(val, enum.Enum):
        return _ir_core.ConstInt(int(val), _ir_core.DataType.INT64, span or _span())
    return _ir_core.ConstInt(int(val), _ir_core.DataType.INT64, span or _span())


def _to_const_bool(val: Any, span: Span | None = None) -> Expr:
    if isinstance(val, Expr):
        return val
    return _ir_core.ConstBool(bool(val), span or _span())


def _ir_set_saturation_flag(mode: Any, enable: Any, *, span: Span | None = None) -> Expr:
    actual_span = span or _span()
    kwargs = {"mode": int(mode), "enable": bool(enable)}
    return _ir_core.create_op_call("set_saturation_flag", [], kwargs, actual_span)


def _ir_get_saturation_flag(mode: Any, *, span: Span | None = None) -> Expr:
    actual_span = span or _span()
    kwargs = {"mode": int(mode)}
    return _ir_core.create_op_call("get_saturation_flag", [], kwargs, actual_span)


def _ir_set_ctrl_spr(start_bit: Any, end_bit: Any, value: Any, *, span: Span | None = None) -> Expr:
    actual_span = span or _span()
    args = [_to_const_int(start_bit, actual_span), _to_const_int(end_bit, actual_span),
            _to_const_int(value, actual_span)]
    return _ir_core.create_op_call("set_ctrl_spr", args, {}, actual_span)


def _ir_get_ctrl_spr(start_bit: Any, end_bit: Any, *, span: Span | None = None) -> Expr:
    actual_span = span or _span()
    args = [_to_const_int(start_bit, actual_span), _to_const_int(end_bit, actual_span)]
    return _ir_core.create_op_call("get_ctrl_spr", args, {}, actual_span)


def _ir_reset_ctrl_spr(start_bit: Any, end_bit: Any, *, span: Span | None = None) -> Expr:
    actual_span = span or _span()
    args = [_to_const_int(start_bit, actual_span), _to_const_int(end_bit, actual_span)]
    return _ir_core.create_op_call("reset_ctrl_spr", args, {}, actual_span)


@op_impl("set_saturation_flag")
def _parse_set_saturation_flag(self, call: ast.Call) -> Expr:
    span = self.span_tracker.get_span(call)
    kwargs = self.parse_op_kwargs(call)
    return _ir_set_saturation_flag(kwargs["mode"], kwargs["enable"], span=span)


@op_impl("get_saturation_flag")
def _parse_get_saturation_flag(self, call: ast.Call) -> Expr:
    span = self.span_tracker.get_span(call)
    kwargs = self.parse_op_kwargs(call)
    return _ir_get_saturation_flag(kwargs["mode"], span=span)


@op_impl("set_ctrl_spr")
def _parse_set_ctrl_spr(self, call: ast.Call) -> Expr:
    span = self.span_tracker.get_span(call)
    args = [self.parse_expression(arg) for arg in call.args]
    return _ir_set_ctrl_spr(*args, span=span)


@op_impl("get_ctrl_spr")
def _parse_get_ctrl_spr(self, call: ast.Call) -> Expr:
    span = self.span_tracker.get_span(call)
    args = [self.parse_expression(arg) for arg in call.args]
    return _ir_get_ctrl_spr(*args, span=span)


@op_impl("reset_ctrl_spr")
def _parse_reset_ctrl_spr(self, call: ast.Call) -> Expr:
    span = self.span_tracker.get_span(call)
    args = [self.parse_expression(arg) for arg in call.args]
    return _ir_reset_ctrl_spr(*args, span=span)
