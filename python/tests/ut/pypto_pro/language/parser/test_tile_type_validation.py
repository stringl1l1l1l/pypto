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
"""Unit tests for invalid-parameter validation of pl.TileType and pl.make_tile_group.

Invalid values must be intercepted at parse time with actionable errors instead
of surfacing as opaque device/runtime failures (bad addresses, zero/negative
tile sizes, misaligned base-address expansion, ...).
"""

from pypto_pro._errors import InvalidArgument, InvalidShape, InvalidType, InvalidVal, RuntimeFailure
import pypto_pro.language as pl
import pytest

from pypto.pypto_impl import ir


def _parse(kernel_def) -> ir.Program:
    return kernel_def.to_kernel_def().parse_target_program(ir.SectionKind.Vector)[0]


# ---------------------------------------------------------------------------
# pl.TileType invalid parameters
# ---------------------------------------------------------------------------


def test_tile_type_shape_zero_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[0], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidArgument, match="shape dimensions must be positive"):
        _parse(k)


def test_tile_type_shape_negative_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[-64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidArgument, match="shape dimensions must be positive"):
        _parse(k)


def test_tile_type_shape_empty_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidShape, match="shape must not be empty"):
        _parse(k)


def test_tile_type_bias_multi_row_rejected():
    # Bias (L0B) tiles must have exactly 1 row: [128, 128] is a hardware-invalid
    # bias shape and previously surfaced only as a F0FFFF compile error.
    @pl.jit
    def k(x: pl.Tensor[[128, 128], pl.DT_FP16]):
        tt = pl.TileType(shape=[128, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Bias)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        _ = g.next()

    with pytest.raises(InvalidShape, match="Bias tiles must have exactly 1 row"):
        _parse(k)


def test_tile_type_bias_single_row_allowed():
    @pl.jit
    def k(x: pl.Tensor[[128, 128], pl.DT_FP16]):
        tt = pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Bias)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        _ = g.next()

    ir_str = str(_parse(k))
    assert "block.make_tile" in ir_str


def test_tile_type_bias_non_nd_layout_rejected():
    @pl.jit(arch="3510")
    def k(x: pl.Tensor[[1, 128], pl.DT_FP32]):
        tt = pl.TileType(
            shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Bias, layout=pl.DN
        )
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        _ = g.next()

    with pytest.raises(InvalidVal, match="Bias tiles require layout in \\{ND\\}"):
        _parse(k)


def test_move_mat_fp16_to_bias_allowed():
    # FP16 Mat -> FP32 Bias is the standard bias-feed path (TMovToBt half->float).
    @pl.jit
    def k(x: pl.Tensor[[1, 128], pl.DT_FP16]):
        bias_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[1, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0x30000, mutex_ids=[10])
        bias_l0b = pl.make_tile_group(
            type=pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Bias),
            addrs=0x0000, mutex_ids=[12])
        cur = bias_l1.next()
        pl.load(cur, x, [0, 0])
        pl.move(bias_l0b.current(), cur)

    ir_str = str(_parse(k))
    assert "block.move" in ir_str


def test_tile_type_shape_non_int_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64.0], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidType, match="must be a compile-time integer"):
        _parse(k)


def test_tile_type_shape_bool_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[True], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidType, match="must be a compile-time integer"):
        _parse(k)


def test_tile_type_bad_dtype_direct_construction_rejected():
    # dtype=0 in kernel source resolves to DataType.BOOL (a valid enum value), so
    # the direct-construction path is the right place to probe a non-DataType dtype.
    with pytest.raises(TypeError, match="dtype must be a pl.DT_\\* / DataType value"):
        _ = pl.TileType(shape=[64], dtype="fp16", target_memory=pl.MemorySpace.Vec)


def test_tile_type_bad_target_memory_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64], dtype=pl.DT_FP16, target_memory="Vec")
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidType, match="target_memory must be a pl.MemorySpace"):
        _parse(k)


def test_tile_type_valid_shape_exceeds_shape_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(
            shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, valid_shape=[65]
        )
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidShape, match="exceeds tile shape dimension"):
        _parse(k)


def test_tile_type_valid_shape_rank_mismatch_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(
            shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, valid_shape=[64, 1]
        )
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidShape, match="must match shape rank"):
        _parse(k)


def test_tile_type_valid_shape_bad_value_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(
            shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, valid_shape=[-2]
        )
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidShape, match="positive or -1 \\(dynamic\\)"):
        _parse(k)


def test_tile_type_bad_fractal_rejected():
    # fractal is a pass-through field on every memory space (any int is accepted,
    # e.g. fractal=256 spreads into make_tile IR); only non-int values are invalid.
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, fractal=True
        )
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidType, match="fractal must be an integer"):
        _parse(k)


def test_tile_type_fractal_arbitrary_int_allowed():
    # Any valid integer fractal (512/1024/32) is legal and must spread into make_tile
    # unchanged; see test_make_tile.py::test_layout_and_fractal_are_spread_from_the_tile_type.
    @pl.jit
    def k(x: pl.Tensor[[128, 128], pl.DT_FP16]):
        tt = pl.TileType(
            shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, fractal=512
        )
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        pl.load(g.next(), x, [0, 0])

    ir_str = str(_parse(k))
    assert "fractal=512" in ir_str


def test_tile_type_compact_out_of_range_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, compact=-1
        )
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidArgument, match="compact must be one of \\[0, 1, 2, 3\\]"):
        _parse(k)


# ---------------------------------------------------------------------------
# pl.make_tile_group invalid addrs / mutex_ids
# ---------------------------------------------------------------------------


def test_make_tile_group_negative_addrs_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=-0x100, mutex_ids=[0])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidArgument, match="addrs must be non-negative integers"):
        _parse(k)


def test_make_tile_group_float_addrs_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=0.5, mutex_ids=[0])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidType, match="addrs.*must be a compile-time integer, or list of integers"):
        _parse(k)


def test_make_tile_group_float_in_addrs_list_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=[0, 0.5], mutex_ids=[0, 1])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidType, match="addrs.*must be a compile-time integer, or list of integers"):
        _parse(k)


def test_make_tile_group_bool_addrs_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=True, mutex_ids=[0])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidType, match="addrs.*must be a compile-time integer, or list of integers"):
        _parse(k)


def test_make_tile_group_none_addrs_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=None, mutex_ids=[0])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidType, match="addrs.*must be a compile-time integer, or list of integers"):
        _parse(k)


def test_make_tile_group_str_in_addrs_list_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=["x"], mutex_ids=[0])
        pl.load(g.next(), x, [0])

    with pytest.raises(RuntimeFailure, match="Failed to parse kernel function"):
        _parse(k)


def test_make_tile_group_bool_mutex_id_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[True])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidType, match="mutex_ids must be ints, got True"):
        _parse(k)


def test_make_tile_group_scalar_mutex_id_int_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=5)
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidType, match="mutex_ids must be a list, tuple, or None"):
        _parse(k)


def test_make_tile_group_scalar_mutex_id_float_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=0.0)
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidType, match="mutex_ids must be a list, tuple, or None"):
        _parse(k)


def test_make_tile_group_unknown_kwarg_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0], foo=1)
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidArgument, match="unexpected keyword argument\\(s\\) \\['foo'\\]"):
        _parse(k)


def test_make_tile_group_typo_kwarg_rejected():
    # mutex_ids -> mutex_id typo must point at the unknown keyword, not just
    # report a missing 'mutex_ids'.
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_id=[0])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidArgument, match="unexpected keyword argument\\(s\\) \\['mutex_id'\\]"):
        _parse(k)


def test_make_tile_group_bwd_fwd_ids_ignored_for_compat():
    # bwd_ids/fwd_ids were silently ignored by the legacy parser and are kept
    # accepted for backward compatibility with existing operator repositories.
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0], bwd_ids=[0], fwd_ids=[0])
        _ = g.next()

    ir_str = str(_parse(k))
    assert "block.make_tile" in ir_str


# ---------------------------------------------------------------------------
# Valid inputs must keep parsing
# ---------------------------------------------------------------------------


def test_valid_dynamic_valid_shape_allowed():
    @pl.jit
    def k(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tt = pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
        )
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        pl.load(g.next(), x, [0, 0])

    ir_str = str(_parse(k))
    assert "block.make_tile" in ir_str


def test_valid_contiguous_addrs_expansion():
    @pl.jit
    def k(x: pl.Tensor[[128, 128], pl.DT_FP16]):
        tt = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        db = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
        pl.load(db.next(), x, [0, 0])
        pl.load(db.next(), x, [0, 0])

    ir_str = str(_parse(k))
    assert "32768" in ir_str  # second slot at base + 128*128*2


def test_valid_acc_default_fractal():
    @pl.jit
    def k(x: pl.Tensor[[64, 64], pl.DT_FP32]):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        _ = g.next()

    ir_str = str(_parse(k))
    assert "block.make_tile" in ir_str


def test_valid_fractal_512_on_left_allowed():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, fractal=512
        )
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        _ = g.next()

    ir_str = str(_parse(k))
    assert "block.make_tile" in ir_str


def test_valid_compact_row_plus_one_allowed():
    # CompactMode::RowPlusOne(=2) is a valid compact mode, must not be rejected.
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, compact=2
        )
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        _ = g.next()

    ir_str = str(_parse(k))
    assert "block.make_tile" in ir_str
