#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for the ``make_tile`` address and footprint contract.

``pl.make_tile(tile_type, *, addr)`` places a tile at a fixed address in
its memory space. Three layers are covered here:

* ``tile_slot_size()`` — the byte footprint derived from a static shape/dtype,
  shared by ``make_tile_expr``'s default ``size`` and ``make_tile_group``'s slot stride;
* ``make_tile_expr()`` — the IR builder behind it, which takes the tile's fields
  spread out, requires ``addr`` and derives ``size``;
* ``pl.make_tile(...)`` as parsed inside a kernel, where a ``pl.TileType`` is the
  only accepted positional argument and ``addr`` is a keyword-only
  compile-time integer. The byte span is always derived from the TileType.

Complements test_block_ops.py, which covers block ops in general.
"""

from pypto_pro._errors import (
    DynamicShapeUnsupported,
    InvalidArgument,
    InvalidOperation,
    InvalidTile,
    InvalidType,
)
from pypto_pro.ir.op.block_ops import make_tile_expr, tile_slot_size
import pypto_pro.language as pl
import pytest

from pypto.pypto_impl import ir


def _span() -> ir.Span:
    return ir.Span("test_make_tile.py", 1, 1, 1, 1)


def _const_tuple(*dims: int) -> ir.MakeTuple:
    """A shape as the parser hands it to make_tile_expr: a MakeTuple of ConstInt."""
    span = _span()
    return ir.MakeTuple([ir.ConstInt(dim, ir.DataType.INDEX, span) for dim in dims], span)


def _dynamic_tuple() -> ir.MakeTuple:
    """A shape with one runtime dimension, which has no compile-time footprint."""
    span = _span()
    return ir.MakeTuple(
        [ir.Var("n", ir.ScalarType(ir.DataType.INT64), span), ir.ConstInt(64, ir.DataType.INDEX, span)],
        span,
    )


def _parse_kernel(kernel_def) -> ir.Program:
    return kernel_def.to_kernel_def().parse_target_program(ir.SectionKind.Vector)[0]


def _kernel_ir(kernel_def) -> str:
    return str(_parse_kernel(kernel_def))


# ---------------------------------------------------------------------------
# tile_slot_size — the shared footprint helper
# ---------------------------------------------------------------------------


def test_slot_size_is_elements_times_dtype_bytes():
    assert tile_slot_size([64, 128], pl.DT_FP16) == 64 * 128 * 2


def test_slot_size_accepts_a_tuple_shape():
    assert tile_slot_size((16, 16), pl.DT_FP32) == 16 * 16 * 4


def test_slot_size_accepts_a_make_tuple_of_const_ints():
    """The parser passes shapes as IR MakeTuples, not Python sequences."""
    assert tile_slot_size(_const_tuple(8, 16), pl.DT_FP32) == 8 * 16 * 4


def test_slot_size_accepts_a_one_dimensional_shape():
    assert tile_slot_size([64], pl.DT_FP16) == 128


@pytest.mark.parametrize(
    ("dtype", "width"),
    [(pl.DT_INT8, 1), (pl.DT_FP16, 2), (pl.DT_FP32, 4), (pl.DT_INT64, 8)],
)
def test_slot_size_scales_with_the_dtype_width(dtype, width):
    assert tile_slot_size([32, 32], dtype) == 32 * 32 * width


def test_slot_size_packs_sub_byte_elements_before_rounding_to_bytes():
    assert tile_slot_size([64, 64], pl.DT_INT4) == 64 * 64 // 2
    assert tile_slot_size([256, 256], pl.DT_FP4E2M1) == 256 * 256 // 2
    assert tile_slot_size([1], pl.DT_INT4) == 1


def test_slot_size_rejects_a_runtime_dimension():
    with pytest.raises(DynamicShapeUnsupported, match="compile-time integers"):
        tile_slot_size(_dynamic_tuple(), pl.DT_FP16)


def test_slot_size_rejects_a_zero_dimension():
    with pytest.raises(ValueError, match="must be positive"):
        tile_slot_size([64, 0], pl.DT_FP16)


def test_slot_size_rejects_a_negative_dimension():
    with pytest.raises(ValueError, match="must be positive"):
        tile_slot_size([64, -8], pl.DT_FP16)


def test_slot_size_rejects_a_bool_dimension():
    """bool subclasses int, but ``True`` is not a shape."""
    with pytest.raises(DynamicShapeUnsupported, match="compile-time integers"):
        tile_slot_size([True, 64], pl.DT_FP16)


# ---------------------------------------------------------------------------
# make_tile_expr() — the IR builder
# ---------------------------------------------------------------------------


def test_builder_requires_addr():
    with pytest.raises(TypeError, match=r"make_tile_expr\(\) missing 1 required keyword-only argument: 'addr'"):
        make_tile_expr([16, 16], pl.DT_FP16, pl.MemorySpace.Vec)


def test_builder_missing_addr_points_at_make_tile_group():
    with pytest.raises(TypeError, match=r"make_tile_expr\(\) missing 1 required keyword-only argument: 'addr'"):
        make_tile_expr([16, 16], pl.DT_FP16, target_memory=pl.MemorySpace.Vec)


def test_builder_treats_address_zero_as_an_address():
    with pytest.raises(TypeError, match=r"make_tile_expr\(\) missing 1 required positional argument: 'target_memory'"):
        assert make_tile_expr([16, 16], pl.DT_FP16, addr=0).kwargs["memref_addr"] == 0


def test_builder_derives_size_from_shape_and_dtype():
    assert make_tile_expr([64, 128], pl.DT_FP16, target_memory=pl.MemorySpace.Vec, \
    addr=0).kwargs["memref_size"] == 64 * 128 * 2


def test_builder_keeps_an_explicit_size():
    """Internal reinterpretation can preserve a span larger than the new shape."""
    assert make_tile_expr([64, 128], pl.DT_FP16, pl.MemorySpace.Vec, addr=0, size=40960).kwargs["memref_size"] == 40960


def test_builder_reports_when_size_cannot_be_derived():
    with pytest.raises(InvalidOperation, match="cannot derive its byte span"):
        make_tile_expr(_dynamic_tuple(), pl.DT_FP16, pl.MemorySpace.Vec, addr=0)


def test_builder_gives_each_tile_its_own_memref_id():
    first = make_tile_expr([16, 16], pl.DT_FP16, pl.MemorySpace.Vec, addr=0).kwargs["memref_id"]
    second = make_tile_expr([16, 16], pl.DT_FP16, pl.MemorySpace.Vec, addr=0).kwargs["memref_id"]
    assert second == first + 1


@pytest.mark.parametrize(
    ("memory_space", "alignment"),
    [
        (pl.MemorySpace.Vec, 32),
        (pl.MemorySpace.Mat, 32),
        (pl.MemorySpace.Acc, 64),
        (pl.MemorySpace.Left, 512),
        (pl.MemorySpace.Right, 512),
    ],
)
def test_builder_enforces_the_alignment_of_each_memory_space(memory_space, alignment):
    make_tile_expr([64, 64], pl.DT_FP16, memory_space, addr=alignment)
    with pytest.raises(ValueError, match=f"not {alignment}-byte aligned"):
        make_tile_expr([64, 64], pl.DT_FP16, memory_space, addr=alignment // 2)


def test_builder_leaves_unaligned_spaces_alone():
    """DDR has no enforced tile alignment, so any address is accepted."""
    assert make_tile_expr([64, 64], pl.DT_FP16, pl.MemorySpace.DDR, addr=1).kwargs["memref_addr"] == 1


# ---------------------------------------------------------------------------
# pl.make_tile(...) inside a kernel — addr binding
# ---------------------------------------------------------------------------


def test_addr_zero_is_accepted_in_a_kernel():
    @pl.jit
    def k(x: pl.Tensor[[1, 64], pl.DT_FP16]):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        t = pl.make_tile(tt, addr=0)
        pl.load(t, x, [0, 0])

    assert "memref_addr=0" in _kernel_ir(k)


def test_addr_none_is_rejected():
    """``addr=None`` parses to IR, not to Python None, so it fails the int check."""

    @pl.jit
    def k(x: pl.Tensor[[1, 64], pl.DT_FP16]):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        t = pl.make_tile(tt, addr=None)
        pl.load(t, x, [0, 0])

    with pytest.raises(InvalidType, match="'addr' must be a compile-time integer"):
        _parse_kernel(k)


def test_bool_addr_is_rejected():
    """bool subclasses int, but ``addr=False`` is a mistake, not address 0."""

    @pl.jit
    def k(x: pl.Tensor[[1, 64], pl.DT_FP16]):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        t = pl.make_tile(tt, addr=False)
        pl.load(t, x, [0, 0])

    with pytest.raises(InvalidType, match="'addr' must be a compile-time integer"):
        _parse_kernel(k)


def test_float_addr_is_rejected():
    @pl.jit
    def k(x: pl.Tensor[[1, 64], pl.DT_FP16]):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        t = pl.make_tile(tt, addr=0.0)
        pl.load(t, x, [0, 0])

    with pytest.raises(InvalidType, match="'addr' must be a compile-time integer"):
        _parse_kernel(k)


def test_loop_index_addr_is_rejected():
    """A tile's address is fixed while parsing, so it cannot follow a loop."""

    @pl.jit
    def k(x: pl.Tensor[[1, 64], pl.DT_FP16]):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        for i in pl.range(2):
            t = pl.make_tile(tt, addr=i * 128)
            pl.load(t, x, [0, 0])

    with pytest.raises(InvalidType) as excinfo:
        _parse_kernel(k)
    assert "'addr' must be a compile-time integer" in str(excinfo.value)
    assert "loop index" in str(excinfo.value)


def test_addr_from_a_kernel_local_constant_is_folded():
    """addr accepts any expression with a parse-time value, not just literals."""

    @pl.jit
    def k(x: pl.Tensor[[1, 64], pl.DT_FP16]):
        base = 0x100
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        t = pl.make_tile(tt, addr=base + 0x20)
        pl.load(t, x, [0, 0])

    assert "memref_addr=288" in _kernel_ir(k)  # 0x120


# ---------------------------------------------------------------------------
# pl.make_tile(...) inside a kernel — derived footprint and removed size argument
# ---------------------------------------------------------------------------


def test_size_is_derived_from_tile_type():
    @pl.jit
    def k(x: pl.Tensor[[1, 64], pl.DT_FP16]):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        t = pl.make_tile(tt, addr=0)
        pl.load(t, x, [0, 0])

    assert "memref_size=128" in _kernel_ir(k)


@pytest.mark.parametrize("size", [-32, 0, 128, 256])
def test_size_keyword_is_rejected(size):
    @pl.jit
    def k(x: pl.Tensor[[1, 64], pl.DT_FP16]):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        t = pl.make_tile(tt, addr=0, size=size)
        pl.load(t, x, [0, 0])

    with pytest.raises(InvalidArgument, match="unexpected keyword argument") as excinfo:
        _parse_kernel(k)
    assert "['size']" in str(excinfo.value)
    assert "only 'addr' is supported" in str(excinfo.value)


def test_runtime_size_is_rejected_before_evaluating_it():
    @pl.jit
    def k(x: pl.Tensor[[pl.DYNAMIC, 64], pl.DT_FP16]):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        t = pl.make_tile(tt, addr=0, size=x.shape[0] * 2)
        pl.load(t, x, [0, 0])

    with pytest.raises(InvalidArgument, match="unexpected keyword argument"):
        _parse_kernel(k)


# ---------------------------------------------------------------------------
# pl.make_tile(...) inside a kernel — argument binding diagnostics
# ---------------------------------------------------------------------------


def test_a_shape_in_place_of_a_tile_type_is_rejected():
    """A tile's shape only reaches make_tile through a TileType."""

    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        t = pl.make_tile([64], pl.DT_FP16, pl.MemorySpace.Vec, 0, 128)
        pl.load(t, x, [0])

    with pytest.raises(InvalidType) as excinfo:
        _parse_kernel(k)
    assert "takes a pl.TileType as its first argument" in str(excinfo.value)
    # the offending expression, quoted from the source
    assert "[64]" in str(excinfo.value)


def test_tile_type_may_be_built_inline():
    """The leading TileType is recognised by value, not by being a named variable."""

    @pl.jit
    def k(x: pl.Tensor[[1, 64], pl.DT_FP16]):
        t = pl.make_tile(
            pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec), addr=0x40
        )
        pl.load(t, x, [0, 0])

    ir_str = _kernel_ir(k)
    assert "memref_addr=64" in ir_str
    assert "memref_size=128" in ir_str


def test_a_tile_type_is_required_even_when_addr_is_given():
    """The builder's spread-out form is not a DSL spelling of make_tile."""

    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        t = pl.make_tile([64], pl.DT_FP16, pl.MemorySpace.Vec, 0x40)
        pl.load(t, x, [0])

    with pytest.raises(InvalidType, match="takes a pl.TileType as its first argument"):
        _parse_kernel(k)


def test_a_call_with_no_positional_argument_is_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        t = pl.make_tile(addr=0x40)
        pl.load(t, x, [0])

    with pytest.raises(InvalidType, match="got no positional argument"):
        _parse_kernel(k)


def test_addr_and_size_given_positionally_are_rejected():
    """A bare pair of ints reads the same swapped, so neither position binds."""

    @pl.jit
    def k(x: pl.Tensor[[1, 64], pl.DT_FP16]):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        t = pl.make_tile(tt, 0, 128)
        pl.load(t, x, [0, 0])

    with pytest.raises(InvalidArgument) as excinfo:
        _parse_kernel(k)
    assert "takes 1 positional argument (the tile type) but 3 were given" in str(excinfo.value)
    assert "addr=" in str(excinfo.value)


def test_a_positional_addr_is_not_silently_shadowed_by_a_keyword_one():
    """Two addrs used to resolve to the positional one, dropping the keyword."""

    @pl.jit
    def k(x: pl.Tensor[[1, 64], pl.DT_FP16]):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        t = pl.make_tile(tt, 0x100, addr=0x2000)
        pl.load(t, x, [0, 0])

    # the keyword does not count toward the arity, exactly as Python reports it
    with pytest.raises(InvalidArgument, match=r"takes 1 positional argument .* but 2 were given"):
        _parse_kernel(k)


def test_a_non_tile_type_first_argument_is_rejected():
    """Anything that is not a TileType is reported, not silently treated as addr."""

    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        t = pl.make_tile(0x40)
        pl.load(t, x, [0])

    with pytest.raises(InvalidType, match="takes a pl.TileType as its first argument"):
        _parse_kernel(k)


# ---------------------------------------------------------------------------
# pl.make_tile(tile_type, ...) — every TileType field reaches the tile
# ---------------------------------------------------------------------------


def test_valid_shape_pad_and_compact_are_spread_from_the_tile_type():
    @pl.jit
    def k(x: pl.Tensor[[128, 128], pl.DT_FP16]):
        tt = pl.TileType(
            shape=[128, 128],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[64, 128],
            pad=pl.TilePad.zero,
            compact=1,
        )
        t = pl.make_tile(tt, addr=0)
        pl.load(t, x, [0, 0])

    ir_str = _kernel_ir(k)
    assert "block.make_tile(tuple(128, 128), tuple(64, 128)" in ir_str
    assert "pad=1" in ir_str
    assert "compact=1" in ir_str


def test_layout_and_fractal_are_spread_from_the_tile_type():
    @pl.jit
    def k(x: pl.Tensor[[128, 128], pl.DT_FP16]):
        tt = pl.TileType(
            shape=[128, 128],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Vec,
            layout=pl.ND,
            fractal=512,
        )
        t = pl.make_tile(tt, addr=0)
        pl.load(t, x, [0, 0])

    ir_str = _kernel_ir(k)
    # ND is (blayout, slayout) = (1, 0).
    assert "blayout=1" in ir_str
    assert "slayout=0" in ir_str
    assert "fractal=512" in ir_str


def test_tile_type_field_cannot_be_overridden_at_make_tile_call():
    @pl.jit
    def k(x: pl.Tensor[[128, 128], pl.DT_FP16]):
        tt = pl.TileType(
            shape=[128, 128],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[8, 8],
        )
        t = pl.make_tile(tt, addr=0, valid_shape=[16, 16])
        pl.load(t, x, [0, 0])

    with pytest.raises(InvalidArgument, match="unexpected keyword argument") as excinfo:
        _parse_kernel(k)
    assert "['valid_shape']" in str(excinfo.value)


def test_unknown_make_tile_keyword_is_rejected():
    @pl.jit
    def k(x: pl.Tensor[[1, 64], pl.DT_FP16]):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        t = pl.make_tile(tt, addr=0, unknown=1)
        pl.load(t, x, [0, 0])

    with pytest.raises(InvalidArgument, match="unexpected keyword argument") as excinfo:
        _parse_kernel(k)
    assert "['unknown']" in str(excinfo.value)


def test_size_is_derived_from_the_shape_not_the_valid_shape():
    """valid_shape narrows what the tile reads; the tile still occupies its full shape."""

    @pl.jit
    def k(x: pl.Tensor[[128, 128], pl.DT_FP16]):
        tt = pl.TileType(
            shape=[128, 128],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[8, 8],
        )
        t = pl.make_tile(tt, addr=0)
        pl.load(t, x, [0, 0])

    assert f"memref_size={128 * 128 * 2}" in _kernel_ir(k)


def test_sub_byte_tile_reserves_packed_bytes():
    """INT4 has no Vec op to consume it, so only the reservation is checked."""

    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_INT4, target_memory=pl.MemorySpace.Vec)
        t = pl.make_tile(tt, addr=0)  # noqa: F841

    assert f"memref_size={64 * 64 // 2}" in _kernel_ir(k)


def test_a_tile_type_with_a_runtime_shape_cannot_derive_its_size():
    """A tuple shape skips the TileType const check, so make_tile_expr reports it."""

    @pl.jit
    def k(x: pl.Tensor[[pl.DYNAMIC, 64], pl.DT_FP16]):
        tt = pl.TileType(shape=(x.shape[0], 64), dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        t = pl.make_tile(tt, addr=0)
        pl.load(t, x, [0, 0])

    with pytest.raises(InvalidOperation, match="cannot derive its byte span"):
        _parse_kernel(k)


# ---------------------------------------------------------------------------
# pl.make_tile(...) inside a kernel — address alignment
# ---------------------------------------------------------------------------


def test_misaligned_acc_addr_is_rejected():
    @pl.jit
    def k(x: pl.Tensor[[128, 128], pl.DT_FP32]):
        tt = pl.TileType(shape=[128, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc)
        t = pl.make_tile(tt, addr=0x20)
        pl.load(t, x, [0, 0])

    with pytest.raises(InvalidTile, match="not 64-byte aligned"):
        _parse_kernel(k)


def test_misaligned_left_addr_is_rejected():
    @pl.jit
    def k(x: pl.Tensor[[128, 128], pl.DT_FP16]):
        tt = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left)
        t = pl.make_tile(tt, addr=0x100)
        pl.load(t, x, [0, 0])

    with pytest.raises(InvalidTile, match="not 512-byte aligned"):
        _parse_kernel(k)


def test_a_folded_addr_is_checked_for_alignment():
    """The alignment check sees the folded value, not the source expression."""

    @pl.jit
    def k(x: pl.Tensor[[1, 64], pl.DT_FP16]):
        base = 0x40
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        t = pl.make_tile(tt, addr=base + 1)
        pl.load(t, x, [0, 0])

    with pytest.raises(InvalidTile, match="not 32-byte aligned"):
        _parse_kernel(k)


# ---------------------------------------------------------------------------
# TileType shape / valid_shape stay compile-time constants
# ---------------------------------------------------------------------------


def test_tile_type_shape_accepts_a_constant_expression():
    @pl.jit
    def k(x: pl.Tensor[[1, 64], pl.DT_FP16]):
        rows = 8
        tt = pl.TileType(shape=[1, rows * 8], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        t = pl.make_tile(tt, addr=0)
        pl.load(t, x, [0, 0])

    assert "block.make_tile(tuple(1, 64)" in _kernel_ir(k)


def test_tile_type_shape_rejects_a_bool_element():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[True, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        t = pl.make_tile(tt, addr=0)
        pl.load(t, x, [0])

    with pytest.raises(InvalidType, match="must be a compile-time integer"):
        _parse_kernel(k)


def test_pad_need_compile_time_value():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16], pad_mode: pl.DT_INT64):
        tt = pl.TileType(
            shape=[64],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Vec,
            pad=pl.TilePad.zero if pad_mode else 0,
        )
        t = pl.make_tile(tt, addr=0)
        pl.load(t, x, [0])

    with pytest.raises(InvalidOperation, match="'pl.TilePad.zero' has no runtime value"):
        _parse_kernel(k)

def test_tile_type_valid_shape_rejects_a_runtime_element():
    """set_validshape() is the runtime path; the TileType field is parse-time only."""

    @pl.jit
    def k(x: pl.Tensor[[pl.DYNAMIC, 64], pl.DT_FP16]):
        tt = pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[x.shape[0], 64],
        )
        t = pl.make_tile(tt, addr=0)
        pl.load(t, x, [0, 0])

    with pytest.raises(InvalidType) as excinfo:
        _parse_kernel(k)
    assert "must be a compile-time integer" in str(excinfo.value)
    assert "pl.set_validshape()" in str(excinfo.value)
