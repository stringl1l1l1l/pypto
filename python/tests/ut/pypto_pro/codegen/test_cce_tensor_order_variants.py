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
"""CCE code generation for per-access tensor layout (the ``order`` kwarg).

``order`` picks which tensor axes an access walks and whether it walks them transposed, but
that choice lands on the tensor's **GlobalTensor declaration**: the row/col strides are ctor
arguments and ``Layout::DN`` is a template argument, so neither can vary per access. One
declaration per tensor cannot serve two orders, so codegen emits one per layout and each
access goes through the one matching its own ``order``.

Two accesses that differ only in their row axis both come out ``Layout::ND``, so nothing
downstream rejects the mismatch -- that is the silent case, and the reason this is checked
here on the declarations rather than only on results.

These read the generated C++ directly, so they need no device and no bisheng. The same
kernels are run for real in
tests/st/pypto_pro/frontend/datacopy/test_load_order_variants.py.
"""

import re

import pypto_pro.language as pl

DIM = 16
TILE = 64


def _compile_to_cce(kernel) -> str:
    from pypto_pro.runtime.jit import _assemble_cv_source, _parse_and_codegen_targets

    cube, vector = _parse_and_codegen_targets(kernel.to_kernel_def(), "3510", "")
    return _assemble_cv_source(cube, vector).content


def _global_tensor_decls(source: str, tensor: str) -> list[str]:
    """The ``using <name>Type = GlobalTensor<...>;`` lines declared for one tensor."""
    pattern = re.compile(rf"using {re.escape(tensor)}\w*Type = GlobalTensor<[^;]*;")
    return pattern.findall(source)


def _tload_sources(source: str) -> list[str]:
    """The GlobalTensor instance each TLOAD reads through, in program order."""
    return re.findall(r"TLOAD\([^,]+,\s*([^)]+)\);", source)


def _instance_strides(source: str, name: str) -> str:
    match = re.search(rf"{re.escape(name)}Type {re.escape(name)}\([^;]*StrideDim5\(([^)]*)\)", source)
    assert match is not None, f"no instance construction found for '{name}'"
    return match.group(1)


@pl.jit
def _mixed_row_axis_kernel(
    x: pl.Tensor[[DIM, DIM, DIM, DIM], pl.DT_FP16],
    inner: pl.Tensor[[DIM, DIM, DIM, DIM], pl.DT_FP16],
    middle: pl.Tensor[[DIM, DIM, DIM, DIM], pl.DT_FP16],
):
    tile_type = pl.TileType(shape=[DIM, DIM], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    inner_group = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])
    middle_group = pl.make_tile_group(type=tile_type, addrs=0x1000, mutex_ids=[1])
    with pl.section_vector():
        inner_tile = inner_group.current()
        middle_tile = middle_group.current()
        pl.load(inner_tile, x, [0, 0, 0, 0], order=[2, 3])
        pl.load(middle_tile, x, [0, 0, 0, 0], order=[1, 3])
        pl.store(inner, inner_tile, [0, 0, 0, 0], order=[2, 3])
        pl.store(middle, middle_tile, [0, 0, 0, 0], order=[2, 3])


@pl.jit
def _single_order_kernel(
    x: pl.Tensor[[DIM, DIM, DIM, DIM], pl.DT_FP16],
    out: pl.Tensor[[DIM, DIM, DIM, DIM], pl.DT_FP16],
):
    tile_type = pl.TileType(shape=[DIM, DIM], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    group = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])
    with pl.section_vector():
        tile = group.current()
        pl.load(tile, x, [0, 0, 0, 0], order=[2, 3])
        pl.load(tile, x, [0, 0, 0, 0], order=[2, 3])
        pl.store(out, tile, [0, 0, 0, 0], order=[2, 3])


@pl.jit
def _transposed_only_kernel(x: pl.Tensor[[TILE, TILE], pl.DT_FP16], out: pl.Tensor[[TILE, TILE], pl.DT_FP32]):
    l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[0])
    left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000, mutex_ids=[1])
    right = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[2])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000, mutex_ids=[3])
    with pl.section_cube():
        tile = l1.current()
        pl.load(tile, x, [0, 0], order=[1, 0])
        pl.move(left.current(), tile)
        pl.move(right.current(), tile)
        pl.matmul(acc.current(), left.current(), right.current())
        pl.store(out, acc.current(), [0, 0])


def test_two_row_axes_get_one_declaration_each():
    """Two orders that differ only in their row axis still need one declaration each."""
    source = _compile_to_cce(_mixed_row_axis_kernel)
    decls = _global_tensor_decls(source, "x")

    assert len(decls) == 2, f"expected one declaration per layout, got {decls}"


def test_each_load_walks_the_axes_its_own_order_names():
    """The second load must not be routed through the first load's strides.

    This is the silent bug: both declarations are Layout::ND, so nothing rejects the
    mismatch -- the second load simply returned the first load's elements.
    """
    source = _compile_to_cce(_mixed_row_axis_kernel)
    inner_src, middle_src = _tload_sources(source)

    assert inner_src != middle_src, f"both loads read through '{inner_src}'"
    # order=[2, 3] steps rows by axis 2 (DIM elements); order=[1, 3] by axis 1 (DIM * DIM).
    assert _instance_strides(source, inner_src) == "1, 1, 1, 16, 1"
    assert _instance_strides(source, middle_src) == "1, 1, 1, 16 * 16, 1"


def test_row_major_and_transposed_do_not_share_a_declaration():
    """A transposed load must not drag the row-major one into Layout::DN with it."""
    source = _compile_to_cce(_transposed_only_kernel)
    decls = _global_tensor_decls(source, "x")

    assert len(decls) == 1, f"one layout must stay one declaration, got {decls}"
    assert "Layout::DN" in decls[0]
    # The first layout keeps the tensor's plain name whichever layout it happens to be, so a
    # kernel that reads a tensor one way generates exactly what it did before variants existed.
    assert decls[0].startswith("using x_0Type ")


def test_repeating_one_order_reuses_its_declaration():
    """Only a *differing* layout adds a declaration; the same order twice does not."""
    source = _compile_to_cce(_single_order_kernel)
    decls = _global_tensor_decls(source, "x")

    assert len(decls) == 1, f"one layout must stay one declaration, got {decls}"
    assert decls[0].startswith("using x_0Type ")
    assert len(set(_tload_sources(source))) == 1
