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

"""Parse-time guards on the merged-row form of pl.load (``order`` naming 3 tensor axes).

The merged form uses a 3D TLOAD descriptor for both ND2NZ and DN2ZN.
These tests cover the current frontend restrictions and the DMA's addressing constraints:
nd2nz reads each row as contiguous elements and never looks at the column axis's stride,
so a non-innermost column axis would quietly transfer the wrong elements.
"""

from pypto_pro._errors import InvalidArgument, InvalidFormat, InvalidShape, InvalidType
import pypto_pro.language as pl
from pypto_pro.runtime.platform import NpuArch
import pytest

B, S, N, G, D = 1, 64, 3, 2, 128
S_TILE = 16
M = S_TILE * G


def _codegen(kernel):
    from pypto_pro.runtime.jit import _assemble_cv_source, _parse_and_codegen_targets
    from pypto_pro.runtime.kernel import KernelDef

    kernel_def = kernel if isinstance(kernel, KernelDef) else kernel.to_kernel_def()
    cube, vector = _parse_and_codegen_targets(kernel_def, NpuArch.DAV_3510, "")
    return _assemble_cv_source(cube, vector).content


def _mat(shape, layout=pl.NZ, dtype=pl.DT_FP16):
    return pl.TileType(shape=shape, dtype=dtype, target_memory=pl.MemorySpace.Mat, layout=layout)


@pytest.mark.parametrize("axes", [[1, 4], [1, 3, 4]])
@pytest.mark.parametrize("transposed", [False, True])
def test_load_records_order_orientation_in_ir(axes, transposed):
    from pypto_pro import ir
    from pypto_pro.ir.op.block_ops import _ir_load, make_tile_expr

    order = axes[::-1] if transposed else axes
    shape = [D, M] if transposed else [M, D]
    layout = pl.ZN if transposed else pl.NZ
    tile = make_tile_expr(shape, pl.DT_FP16, pl.MemorySpace.Mat, addr=0, layout=layout)
    tensor = ir.Var("x", ir.TensorType([B, S, N, G, D], pl.DT_FP16), ir.Span.unknown())

    call = _ir_load(tile, tensor, [0] * 5, order=order)

    assert call.kwargs.get("is_transpose", False) == transposed
    if len(axes) == 3:
        assert call.kwargs["nd_inner"] == G
    else:
        assert "nd_inner" not in call.kwargs


def test_merged_load_lowers_to_multi_matrix_tload():
    """The descriptor separates matrix and row strides and directly calls TLOAD."""

    @pl.jit(auto_mutex=True)
    def kernel(x: pl.Tensor[[B, S, N, G, D], pl.DT_FP16]):
        t = pl.make_tile_group(type=_mat([M, D]), addrs=0x0, mutex_ids=[0])
        with pl.section_cube():
            pl.load(t.current(), x, [0, 0, 0, 0, 0], order=[1, 3, 4])

    src = _codegen(kernel)
    assert "MultiNd2Nz" not in src
    assert f"const uint64_t __merged_inner = {G};" in src
    # Row stride is the middle axis's (D); matrix stride is the outer axis's (N*G*D).
    assert "StrideDim5(1, 1, uint64_t{1} * 3 * 2 * 128, uint64_t{1} * 128, 1)" in src
    assert "pto::Shape<1, 1, -1, -1, -1>" in src
    assert "pto::Stride<1, 1, -1, -1, 1>" in src
    assert ", Layout::ND>;" in src
    assert "DIM_4>(__merged_count, __merged_rows," in src
    assert "TLOAD(" in src
    assert ".SetStride<" not in src
    assert "pypto_multi_nd2nz.h" not in src


@pytest.mark.parametrize("use_view", [False, True])
def test_merged_load_reuses_each_layout_declaration(use_view):
    """Hoist compatible loads, keeping ordinary, merged ND and merged DN types distinct."""

    @pl.jit(auto_mutex=True)
    def kernel(x: pl.Tensor[[B, S, N, G, D], pl.DT_FP16]):
        t = pl.make_tile_group(type=_mat([M, D]), addrs=0, mutex_ids=[0])
        u = pl.make_tile_group(type=_mat([D, M], pl.ZN), addrs=0x10000, mutex_ids=[1])
        if use_view:
            source = pl.make_tensor(x, [B, S, N, G, D], [S * N * G * D, N * G * D, G * D, D, 1])
        else:
            source = x
        with pl.section_cube():
            pl.load(t.current(), source, [0, 0, 0, 0, 0], order=[1, 3, 4])
            pl.load(t.current(), source, [0, 0, 0, 0, 0], order=[1, 4])
            for i in pl.range(2):
                pl.load(t.current(), source, [0, i, 0, 0, 0], order=[1, 3, 4])
                pl.load(u.current(), source, [0, i, 0, 0, 0], order=[4, 3, 1])
                pl.load(t.current(), source, [0, i + 1, 0, 0, 0], order=[1, 3, 4])

    src = _codegen(kernel)
    assert src.count(" = GlobalTensor<") == 3
    assert src.count("pto::Shape<1, 1, -1, -1, -1>") == 2
    assert "__merged_tensor" not in src
    before_loop, loop = src.split("for (")
    assert before_loop.count(" = GlobalTensor<") == 3
    assert "GlobalTensor<" not in loop
    assert loop.count(".SetShape<") == 3
    assert ".SetStride<" not in src
    assert loop.count("TASSIGN(") == 3
    assert loop.count("TLOAD(") == 3


def test_merged_load_keeps_dynamic_inner_axis_wide():
    """Normalize the wide inner extent and promote dynamic strides before multiplication."""
    dyn = pl.DYNAMIC

    @pl.jit(auto_mutex=True)
    def kernel(x: pl.Tensor[[dyn, dyn, dyn, dyn, dyn], pl.DT_FP16]):
        t = pl.make_tile_group(type=_mat([M, D]), addrs=0x0, mutex_ids=[0])
        with pl.section_cube():
            pl.load(t.current(), x, [0, 0, 0, 0, 0], order=[1, 3, 4])

    src = _codegen(kernel)
    assert "MultiNd2Nz" not in src
    assert "const uint64_t __merged_inner = __pypto_dyn_x_3;" in src
    # Both source strides come from runtime dims too.
    assert ("StrideDim5(1, 1, uint64_t{1} * __pypto_dyn_x_2 * __pypto_dyn_x_3 * __pypto_dyn_x_4, "
            "uint64_t{1} * __pypto_dyn_x_4, 1)") in src
    assert "const int32_t __merged_rows = static_cast<int32_t>(__merged_inner < 32 ? __merged_inner : 32);" in src
    assert "__merged_matrix_stride" not in src
    assert "const int32_t __merged_count =" in src


def test_ordinary_two_axis_load_still_uses_tload():
    """A 2-axis order is untouched by the merged path."""

    @pl.jit(auto_mutex=True)
    def kernel(x: pl.Tensor[[B, S, N, G, D], pl.DT_FP16]):
        t = pl.make_tile_group(type=_mat([S_TILE, D]), addrs=0x0, mutex_ids=[0])
        with pl.section_cube():
            pl.load(t.current(), x, [0, 0, 0, 0, 0], order=[1, 4])

    src = _codegen(kernel)
    assert "TLOAD(" in src
    assert "MultiNd2Nz" not in src


def test_merged_load_requires_innermost_column_axis():
    """nd2nz reads each row as contiguous elements -- it never applies a column stride."""

    @pl.jit(auto_mutex=True)
    def kernel(x: pl.Tensor[[B, S, N, G, D], pl.DT_FP16]):
        t = pl.make_tile_group(
            type=pl.TileType(shape=[M, D], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Mat, layout=pl.NZ), addrs=0x0, mutex_ids=[0])
        with pl.section_cube():
            pl.load(t.current(), x, [0, 0, 0, 0, 0], order=[1, 2, 3])

    with pytest.raises(Exception, match="innermost axis"):
        _codegen(kernel)


def test_merged_load_requires_l1_destination():
    """The current frontend supports merged loads into L1; UB needs the data packed first."""

    @pl.jit(auto_mutex=True)
    def kernel(x: pl.Tensor[[B, S, N, G, D], pl.DT_FP16]):
        t = pl.make_tile_group(
            type=pl.TileType(shape=[M, D], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Vec), addrs=0x0, mutex_ids=[0])
        with pl.section_vector():
            pl.load(t.current(), x, [0, 0, 0, 0, 0], order=[1, 3, 4])

    with pytest.raises(Exception, match=r"currently require a Mat \(L1\) destination") as exc_info:
        _codegen(kernel)
    assert "load the selected data into a packed tile" in str(exc_info.value)
    assert "reinterpret alone does not gather noncontiguous GM data" in str(exc_info.value)


def test_transposed_merged_load_is_the_same_transfer_into_a_zn_tile():
    """The two descriptors use exchanged logical axes over identical GM/L1 bytes."""

    @pl.jit(auto_mutex=True)
    def ascending(x: pl.Tensor[[B, S, N, G, D], pl.DT_FP16]):
        t = pl.make_tile_group(type=_mat([M, D]), addrs=0x0, mutex_ids=[0])
        with pl.section_cube():
            pl.load(t.current(), x, [0, 0, 0, 0, 0], order=[1, 3, 4])

    @pl.jit(auto_mutex=True)
    def reversed_(x: pl.Tensor[[B, S, N, G, D], pl.DT_FP16]):
        t = pl.make_tile_group(type=_mat([D, M], pl.ZN), addrs=0x0, mutex_ids=[0])
        with pl.section_cube():
            pl.load(t.current(), x, [0, 0, 0, 0, 0], order=[4, 3, 1])

    plain = _codegen(ascending)
    transposed = _codegen(reversed_)
    assert ", Layout::ND>;" in plain
    assert ", Layout::DN>;" in transposed
    assert "pto::Stride<1, 1, -1, 1, -1>" in transposed
    assert "DIM_4>(__merged_count, static_cast<int64_t>(" in transposed
    assert "GetValidRow()), __merged_rows)" in transposed
    assert "StrideDim5(1, 1, uint64_t{1} * 3 * 2 * 128, 1, uint64_t{1} * 128)" in transposed
    assert ".SetStride<" not in transposed
    assert f"const uint64_t __merged_inner = {G};" in plain
    assert f"const uint64_t __merged_inner = {G};" in transposed


def test_transposed_merged_order_requires_a_zn_destination():
    """[4, 3, 1] into an NZ tile asks for a transpose the DMA cannot do.

    The reversal is not a different transfer, it is the other label on the same bytes, so the
    Tile has to carry that label. An NZ dst here would take the ascending order's data while
    the kernel reads it as the transpose.
    """

    @pl.jit(auto_mutex=True)
    def kernel(x: pl.Tensor[[B, S, N, G, D], pl.DT_FP16]):
        t = pl.make_tile_group(
            type=pl.TileType(shape=[M, D], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Mat, layout=pl.NZ), addrs=0x0, mutex_ids=[0])
        with pl.section_cube():
            pl.load(t.current(), x, [0, 0, 0, 0, 0], order=[4, 3, 1])

    with pytest.raises(Exception, match="must be the ZN alias"):
        _codegen(kernel)


def test_transposed_merged_order_accepts_a_partial_final_matrix():
    """The 32 merged columns need a remainder DMA with static G=3; 48 is the D window."""

    @pl.jit(auto_mutex=True)
    def kernel(x: pl.Tensor[[B, S, N, 3, 48], pl.DT_FP16]):
        t = pl.make_tile_group(type=_mat([48, 32], pl.ZN), addrs=0x0, mutex_ids=[0])
        with pl.section_cube():
            pl.load(t.current(), x, [0, 0, 0, 0, 0], order=[4, 3, 1])

    source = _codegen(kernel)
    assert "const uint64_t __merged_inner = 3;" in source
    assert "GetValidCol()" in next(line for line in source.splitlines() if "__merged_valid =" in line)
    assert "GetValidRow()), __merged_rows)" in source


def test_merged_load_rejects_a_swapped_column_axis():
    """[1, 4, 3] names axis 3 as the column axis -- it must not pass as [1, 3, 4].

    The sharpest case of the order rule, and the reason neither accepted spelling can be
    recognised from the sorted list: [1, 3, 4], [4, 3, 1] and [1, 4, 3] all sort to tile_dims
    [1, 3, 4]. Validating that would accept this call and transfer the ascending order's data --
    silently, since axis 3 has a plausible extent.
    """

    @pl.jit(auto_mutex=True)
    def kernel(x: pl.Tensor[[B, S, N, G, D], pl.DT_FP16]):
        t = pl.make_tile_group(type=_mat([M, D]), addrs=0x0, mutex_ids=[0])
        with pl.section_cube():
            pl.load(t.current(), x, [0, 0, 0, 0, 0], order=[1, 4, 3])

    with pytest.raises(Exception, match="ascending"):
        _codegen(kernel)


@pytest.mark.parametrize("inner", [3, 64, 65536, 2**31])
@pytest.mark.parametrize("transposed", [False, True])
def test_merged_load_keeps_static_inner_for_partial_matrices(inner, transposed):
    """Keep wide static extents intact until clipping to the tile capacity."""
    shape = [D, 32] if transposed else [32, D]
    layout = pl.ZN if transposed else pl.NZ
    order = [4, 3, 1] if transposed else [1, 3, 4]

    @pl.jit(auto_mutex=True)
    def kernel(x: pl.Tensor[[B, S, N, inner, D], pl.DT_FP16]):
        t = pl.make_tile_group(type=_mat(shape, layout), addrs=0x0, mutex_ids=[0])
        with pl.section_cube():
            pl.load(t.current(), x, [0, 0, 0, 0, 0], order=order)

    source = _codegen(kernel)
    assert f"const uint64_t __merged_inner = {inner};" in source
    assert "const int32_t __merged_rows = static_cast<int32_t>(__merged_inner < 32 ? __merged_inner : 32);" in source
    assert "TLOAD(" in source


def test_merged_order_rejected_for_load_tile():
    """Tile-block offsets have no single stride to scale once rows span two tensor axes."""

    @pl.jit(auto_mutex=True)
    def kernel(x: pl.Tensor[[B, S, N, G, D], pl.DT_FP16]):
        t = pl.make_tile_group(type=_mat([M, D]), addrs=0x0, mutex_ids=[0])
        with pl.section_cube():
            pl.load_tile(t.current(), x, [0, 0, 0, 0, 0], order=[1, 3, 4])

    with pytest.raises(Exception, match="only supported"):
        _codegen(kernel)


@pytest.mark.parametrize("order", [[], [0], [1], [4], [1, 2, 3, 4]])
def test_load_rejects_wrong_order_rank(order):
    from pypto_pro import ir
    from pypto_pro.ir.op.block_ops import _ir_load, make_tile_expr

    tile = make_tile_expr([M, D], pl.DT_FP16, pl.MemorySpace.Mat, addr=0, layout=pl.NZ)
    tensor = ir.Var("x", ir.TensorType([B, S, N, G, D], pl.DT_FP16), ir.Span.unknown())
    with pytest.raises(InvalidShape, match="order.*axes"):
        _ir_load(tile, tensor, [0] * 5, order=order)


@pytest.mark.parametrize("order", [[-1, 4], [1, 5]])
def test_load_rejects_out_of_range_order(order):
    from pypto_pro import ir
    from pypto_pro.ir.op.block_ops import _ir_load, make_tile_expr

    tile = make_tile_expr([1, D], pl.DT_FP16, pl.MemorySpace.Vec, addr=0)
    tensor = ir.Var("x", ir.TensorType([B, S, N, G, D], pl.DT_FP16), ir.Span.unknown())
    with pytest.raises(InvalidShape, match="order axis.*out of range"):
        _ir_load(tile, tensor, [0] * 5, order=order)


@pytest.mark.parametrize("op_name", ["load", "load_tile", "store", "store_tile"])
@pytest.mark.parametrize("shape", [[D], [3, D], [2, 3, D]])
@pytest.mark.parametrize("order", [[0], [1]])
def test_transfer_rejects_single_axis_order(op_name, shape, order):
    """Rank-1 handling must not bypass explicit-order validation for any transfer API."""
    from pypto_pro import ir
    from pypto_pro.ir.op import block_ops
    from pypto_pro.ir.op.block_ops import make_tile_expr

    tile = make_tile_expr([1, D], pl.DT_FP16, pl.MemorySpace.Vec, addr=0)
    tensor = ir.Var("x", ir.TensorType(shape, pl.DT_FP16), ir.Span.unknown())
    operands = (tile, tensor) if op_name.startswith("load") else (tensor, tile)
    error = InvalidArgument if len(shape) == 1 else InvalidShape
    message = "order is not supported for rank-1" if len(shape) == 1 else "order must name 2 axes.*omit order"
    with pytest.raises(error, match=message):
        getattr(block_ops, "_ir_" + op_name)(*operands, [0] * len(shape), order=order)


@pytest.mark.parametrize("op_name", ["store", "store_tile"])
def test_store_rejects_merged_order(op_name):
    from pypto_pro import ir
    from pypto_pro.ir.op import block_ops
    from pypto_pro.ir.op.block_ops import make_tile_expr

    tile = make_tile_expr([M, D], pl.DT_FP16, pl.MemorySpace.Vec, addr=0)
    tensor = ir.Var("x", ir.TensorType([B, S, N, G, D], pl.DT_FP16), ir.Span.unknown())
    with pytest.raises(InvalidShape, match="order.*axes"):
        getattr(block_ops, "_ir_" + op_name)(tensor, tile, [0] * 5, order=[1, 3, 4])


@pytest.mark.parametrize("offset", [1, "dynamic"])
def test_merged_load_rejects_nonzero_inner_offset(offset):
    from pypto_pro import ir
    from pypto_pro.ir.op.block_ops import _ir_load, make_tile_expr

    tile = make_tile_expr([M, D], pl.DT_FP16, pl.MemorySpace.Mat, addr=0, layout=pl.NZ)
    tensor = ir.Var("x", ir.TensorType([B, S, N, G, D], pl.DT_FP16), ir.Span.unknown())
    if offset == "dynamic":
        offset = ir.Var("offset", ir.ScalarType(pl.DT_INT32), ir.Span.unknown())
    with pytest.raises(InvalidArgument, match="middle axis.*zero"):
        _ir_load(tile, tensor, [0, 0, 0, offset, 0], order=[1, 3, 4])


@pytest.mark.parametrize("dtype", [pl.DT_INT64, pl.DT_FP4E2M1, pl.DT_FP4E1M2])
def test_merged_load_rejects_unsupported_element_width(dtype):
    from pypto_pro import ir
    from pypto_pro.ir.op.block_ops import _ir_load, make_tile_expr

    tile = make_tile_expr([M, D], dtype, pl.MemorySpace.Mat, addr=0, layout=pl.NZ)
    tensor = ir.Var("x", ir.TensorType([B, S, N, G, D], dtype), ir.Span.unknown())
    with pytest.raises(InvalidType, match="8, 16 or 32-bit"):
        _ir_load(tile, tensor, [0] * 5, order=[1, 3, 4])


def test_merged_load_rejects_dtype_mismatch():
    from pypto_pro import ir
    from pypto_pro.ir.op.block_ops import _ir_load, make_tile_expr

    tile = make_tile_expr([M, D], pl.DT_BF16, pl.MemorySpace.Mat, addr=0, layout=pl.NZ)
    tensor = ir.Var("x", ir.TensorType([B, S, N, G, D], pl.DT_FP16), ir.Span.unknown())
    with pytest.raises(InvalidType, match="dtype mismatch"):
        _ir_load(tile, tensor, [0] * 5, order=[1, 3, 4])


def test_merged_load_rejects_strided_columns():
    from pypto_pro import ir
    from pypto_pro.ir.op.block_ops import _ir_load, make_tile_expr

    tile = make_tile_expr([M, D], pl.DT_FP16, pl.MemorySpace.Mat, addr=0, layout=pl.NZ)
    stride = [ir.ConstInt(v, pl.DT_INT64, ir.Span.unknown()) for v in [98304, 1536, 512, 256, 2]]
    view = ir.TensorView(stride, ir.TensorLayout.ND)
    tensor = ir.Var("x", ir.TensorType([B, S, N, G, D], pl.DT_FP16, None, view), ir.Span.unknown())
    with pytest.raises(InvalidArgument, match="column stride.*1"):
        _ir_load(tile, tensor, [0] * 5, order=[1, 3, 4])


@pytest.mark.parametrize("rows, fractal, message", [(65536, 512, "destination row stride"),
                                                     (32, 1024, "512-byte fractal")])
def test_merged_load_rejects_unrepresentable_destination(rows, fractal, message):
    from pypto_pro import ir
    from pypto_pro.ir.op.block_ops import _ir_load

    # Build the IR type directly: allocation capacity checks must not mask the DMA limit.
    span = ir.Span.unknown()
    memref = ir.MemRef(pl.MemorySpace.Mat, ir.ConstInt(0, pl.DT_INT64, span), rows * D * 2, 0)
    hardware = ir.HardwareInfo(ir.TileLayout.col_major, ir.TileLayout.row_major, fractal)
    tile = ir.Var("tile", ir.TileType([rows, D], pl.DT_FP16, memref, hardware_info=hardware), span)
    tensor = ir.Var("x", ir.TensorType([B, S, N, G, D], pl.DT_FP16), span)
    error = InvalidShape if fractal == 512 else InvalidFormat
    with pytest.raises(error, match=message):
        _ir_load(tile, tensor, [0] * 5, order=[1, 3, 4])


@pytest.mark.parametrize("dtype", [pl.DT_FP8E4M3FN, pl.DT_FP8E5M2, pl.DT_FP32])
@pytest.mark.parametrize("transposed", [False, True])
def test_merged_load_dtype_and_orientation_codegen(dtype, transposed):
    shape = [D, M] if transposed else [M, D]
    layout = pl.ZN if transposed else pl.NZ
    order = [4, 3, 1] if transposed else [1, 3, 4]

    @pl.jit(auto_mutex=True)
    def kernel(x: pl.Tensor[[B, S, N, G, D], dtype]):
        t = pl.make_tile_group(type=_mat(shape, layout, dtype), addrs=0, mutex_ids=[0])
        with pl.section_cube():
            pl.load(t.current(), x, [0, 0, 0, 0, 0], order=order)

    source = _codegen(kernel)
    assert "MultiNd2Nz" not in source
    assert f"const uint64_t __merged_inner = {G};" in source
    assert "TLOAD(" in source
