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
"""A single-column tile settles its block layout by shape -- and only its block layout.

``BLayout::RowMajor`` requires ``Cols * sizeof(DType) % 32 == 0``, so a tile whose last dim is
1 has to be ``ColMajor``; that is forced by the ISA and is not the user's to choose. But
``ConvertTileType`` used to let that shape rule stand in for the tile's whole ``hw_info``, in
opposite ways on its two paths:

* the memref path took it as an ``else if`` and dropped ``slayout``/``fractal``/``pad``/``compact``;
* the memref-less path (if-merged and intermediate tiles) applied ``hw_info`` *afterwards*,
  overriding the forced ColMajor back to RowMajor -- which the ISA then rejects.

Both paths are covered here. A fractal layout on a 1-wide tile is now rejected outright rather
than silently rewritten, which is checked too.

These read the generated C++ directly, so they need no device and no bisheng.
"""

import re

import pypto_pro.language as pl
import pytest

ROWS = 64
COLS = 64


def _compile_to_cce(kernel) -> str:
    from pypto_pro.runtime.jit import _assemble_cv_source, _parse_and_codegen_targets

    cube, vector = _parse_and_codegen_targets(kernel.to_kernel_def(), "3510", "")
    return _assemble_cv_source(cube, vector).content


def _tile_type_alias(source: str, tile_group: str) -> str:
    """The ``using _tg_<name>_tiles_0_Type = Tile<...>;`` line for one tile group."""
    match = re.search(rf"using _tg_{re.escape(tile_group)}_tiles_0_Type = (Tile<[^;]*);", source)
    assert match is not None, f"no tile type alias found for '{tile_group}'"
    return match.group(1)


def _bare_tile_decl(source: str, var: str) -> str:
    """The ``Tile<...> <var>;`` declaration of a tile that has no memref of its own."""
    match = re.search(rf"^\s*(Tile<[^;>]*>) {re.escape(var)};", source, re.MULTILINE)
    assert match is not None, f"no bare tile declaration found for '{var}'"
    return match.group(1)


@pl.jit
def _compact_tiles_kernel(x: pl.Tensor[[ROWS, COLS], pl.DT_FP32]):
    """One column tile and one ordinary tile, both asking for compact=1."""
    column_type = pl.TileType(
        shape=[ROWS, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, compact=1
    )
    full_type = pl.TileType(
        shape=[ROWS, COLS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, compact=1
    )
    column = pl.make_tile_group(type=column_type, addrs=0x00000, mutex_ids=[0])
    full = pl.make_tile_group(type=full_type, addrs=0x10000, mutex_ids=[1])
    with pl.section_vector():
        pl.load(full.current(), x, [0, 0])
        pl.load(column.current(), x, [0, 0])


@pl.jit
def _column_tile_declared_nd_kernel(x: pl.Tensor[[ROWS, COLS], pl.DT_FP32]):
    """A column tile asking for pl.ND, which the ISA cannot honour for a 1-wide tile."""
    column_type = pl.TileType(
        shape=[ROWS, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.ND
    )
    column = pl.make_tile_group(type=column_type, addrs=0x0000, mutex_ids=[0])
    with pl.section_vector():
        pl.load(column.current(), x, [0, 0])


@pl.jit
def _if_merged_column_tile_kernel(x: pl.Tensor[[ROWS, COLS], pl.DT_FP32], flag: pl.Tensor[[1, 1], pl.DT_INT32]):
    """A column tile merged out of an if, which has no memref of its own."""
    column_type = pl.TileType(shape=[ROWS, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    a = pl.make_tile_group(type=column_type, addrs=0x0000, mutex_ids=[0])
    b = pl.make_tile_group(type=column_type, addrs=0x1000, mutex_ids=[1])
    with pl.section_vector():
        c = flag[0, 0]
        if c > 0:
            merged = a.next()
        else:
            merged = b.next()
        pl.load(merged, x, [0, 0])


def test_column_tile_keeps_its_compact_mode():
    """The shape rule must not swallow the rest of hw_info (memref path).

    The column tile used to come out without CompactMode at all, so ``compact=1`` was silently
    ignored for exactly the reduction-vector tiles that ask for it (FA, softmax, layernorm).
    """
    source = _compile_to_cce(_compact_tiles_kernel)

    column = _tile_type_alias(source, "column")
    assert "BLayout::ColMajor" in column, column
    assert "CompactMode::Normal" in column, column

    # The ordinary tile is the control: it always kept its compact mode.
    full = _tile_type_alias(source, "full")
    assert "BLayout::RowMajor" in full, full
    assert "CompactMode::Normal" in full, full


def test_column_tile_is_col_major_whatever_the_layout_kwarg_says():
    """A 1-wide tile cannot be RowMajor, so the shape still decides the block layout."""
    source = _compile_to_cce(_column_tile_declared_nd_kernel)

    assert "BLayout::ColMajor" in _tile_type_alias(source, "column")


def test_memref_less_column_tile_is_col_major():
    """The memref-less path must apply the shape rule *after* hw_info, not before.

    An if-merged tile has no memref of its own and takes the other branch of ConvertTileType.
    Its hw_info carries the default row_major, which used to override the forced ColMajor and
    produce a 1-wide RowMajor tile -- rejected by the ISA, since RowMajor needs the columns
    32-byte aligned and this tile has one column.
    """
    source = _compile_to_cce(_if_merged_column_tile_kernel)

    merged = _bare_tile_decl(source, "merged_2")
    assert "64, 1," in merged, merged
    assert "BLayout::ColMajor" in merged, merged


def test_fractal_layout_on_a_column_tile_is_rejected():
    """A 1-wide tile can only be ND or DN; a fractal layout would be silently rewritten.

    ZN on a [ROWS, 1] Mat tile used to emit BLayout::ColMajor + SLayout::ColMajor, i.e. NN --
    a layout the user never asked for.
    """
    with pytest.raises(Exception, match="requires layout ND or DN"):

        @pl.jit
        def _fractal_column_kernel(x: pl.Tensor[[ROWS, COLS], pl.DT_FP16]):
            column_type = pl.TileType(
                shape=[ROWS, 1], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN
            )
            column = pl.make_tile_group(type=column_type, addrs=0x0000, mutex_ids=[0])
            with pl.section_cube():
                pl.load(column.current(), x, [0, 0])

        _compile_to_cce(_fractal_column_kernel)
