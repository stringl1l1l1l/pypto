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

"""System test for fixed-count per-tile TileGroup mutex IDs.

The move consumes a flat single-ID source group and a two-ID destination group
in one operation. Their runtime IDs overlap, so the generated mutex operation
must acquire and release each distinct ID once in the same order.
The scalar getval/setval pair also verifies that all destination mutex IDs
are propagated to PIPE_S synchronization. Additional cases cover overlapping
slots in one group, equal-count metadata merges across groups, tuple input, and
explicit depth with discrete addresses. An ordinary Python helper case verifies
that Tile mutex metadata survives a local if/else across an inline call.
"""

import os

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

TILE_ROWS = 32
TILE_COLS = 128
NUM_SLOTS = 2
NUM_TILES = 8
FULL_ROWS = TILE_ROWS * NUM_TILES


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


@pl.vector_function
def _add_one(tile):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    for row in pl.range(0, TILE_ROWS):
        offset = row * TILE_COLS
        source = vf.load_align(tile, offset)
        result = vf.adds(source, 1.0, mask)
        vf.store_align(tile + offset, result, mask)


def _select_tile_with_if_else(true_tile, false_tile, choose_true):
    selected_tile = true_tile
    if choose_true:
        selected_tile = true_tile
    else:
        selected_tile = false_tile
    return selected_tile


@pl.jit(arch="3510", auto_mutex=True)
def mixed_width_groups_kernel(
    source: pl.Tensor[[FULL_ROWS, TILE_COLS], pl.DT_FP16],
    output: pl.Tensor[[FULL_ROWS, TILE_COLS], pl.DT_FP16],
):
    tile_type = pl.TileType(
        shape=[TILE_ROWS, TILE_COLS],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    source_group = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    output_group = pl.make_tile_group(
        type=tile_type,
        addrs=0x4000,
        mutex_ids=[[0, 2], [1, 3]],
    )

    with pl.section_vector():
        for index in pl.range(0, NUM_TILES):
            source_tile = source_group[index % NUM_SLOTS]
            output_tile = output_group[index % NUM_SLOTS]
            row = index * TILE_ROWS
            pl.load(source_tile, source, [row, 0])
            pl.move(output_tile, source_tile)
            _add_one(output_tile)
            first = output_tile[0, 0]
            output_tile[1, 0] = first
            pl.store(output, output_tile, [row, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_mixed_single_and_multi_id_tile_groups():
    _require_a5(ST_DEVICE)
    torch.manual_seed(0)
    source = torch.rand([FULL_ROWS, TILE_COLS], device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    output = torch.zeros([FULL_ROWS, TILE_COLS], device=ST_DEVICE, dtype=torch.float16)

    mixed_width_groups_kernel(source, output)
    torch.npu.synchronize()

    golden = source.cpu().float() + 1.0
    for index in range(NUM_TILES):
        row = index * TILE_ROWS
        golden[row + 1, 0] = golden[row, 0]
    torch.testing.assert_close(output.cpu().float(), golden, rtol=3e-3, atol=3e-3)


# ===========================================================================
# Multi-ID tests: cube+vector, overlapping groups, dynamic patterns
# ===========================================================================

CUBE_M = 64
CUBE_K = 64
CUBE_N = 64
CUBE_VEC_ROWS = 32

DYN_DEPTH = 4
DYN_NUM_TILES = 8
DYN_FULL_ROWS = TILE_ROWS * DYN_NUM_TILES

PURE_DEPTH = 4
PURE_FULL_ROWS = TILE_ROWS * PURE_DEPTH

CONST_DEPTH = 5

EXPLICIT_DEPTH = 3
EXPLICIT_FULL_ROWS = TILE_ROWS * EXPLICIT_DEPTH

L0A_M = 128
L0A_K_WIDE = 256
L0A_K_TILE = 128

L0B_K = 128
L0B_K_WIDE = 256
L0B_N = 128


@pl.jit(arch="3510", auto_mutex=True)
def cube_vector_multi_id_kernel(
    a: pl.Tensor[[CUBE_M, CUBE_N], pl.DT_FP32],
    b: pl.Tensor[[CUBE_M, CUBE_K], pl.DT_FP16],
    c: pl.Tensor[[CUBE_K, CUBE_N], pl.DT_FP16],
    out: pl.Tensor[[CUBE_M, CUBE_N], pl.DT_FP32],
):
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CUBE_M, CUBE_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x0000,
        mutex_ids=[[0, 1]],
    )
    c_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CUBE_K, CUBE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x2000,
        mutex_ids=[[2, 3]],
    )
    b_left = pl.make_tile_group(
        type=pl.TileType(shape=[CUBE_M, CUBE_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[4],
    )
    c_right = pl.make_tile_group(
        type=pl.TileType(shape=[CUBE_K, CUBE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[5],
    )
    tile_acc = pl.make_tile(
        pl.TileType(shape=[CUBE_M, CUBE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addr=0x0000,
    )
    vec_group = pl.make_tile_group(
        type=pl.TileType(shape=[CUBE_VEC_ROWS, CUBE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids=[[6, 7], [8, 9], [10, 11], [12, 13]],
    )

    with pl.section_cube():
        pl.load(b_l1[0], b, [0, 0])
        pl.load(c_l1[0], c, [0, 0])
        pl.move(b_left[0], b_l1[0])
        pl.move(c_right[0], c_l1[0])
        pl.matmul(tile_acc, b_left[0], c_right[0])
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.move(vec_group[0], tile_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    with pl.section_vector():
        sub_index = pl.get_subblock_idx()
        row_offset = sub_index * CUBE_VEC_ROWS
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=0)
        pl.load(vec_group[1], a, [row_offset, 0])
        pl.add(vec_group[2], vec_group[1], vec_group[0])
        pl.move(vec_group[3], vec_group[2])
        pl.store(out, vec_group[3], [row_offset, 0])


@pl.jit(arch="3510", auto_mutex=True)
def overlap_groups_add_kernel(
    a: pl.Tensor[[DYN_FULL_ROWS, TILE_COLS], pl.DT_FP16],
    b: pl.Tensor[[DYN_FULL_ROWS, TILE_COLS], pl.DT_FP16],
    out: pl.Tensor[[DYN_FULL_ROWS, TILE_COLS], pl.DT_FP16],
):
    tile_type = pl.TileType(shape=[TILE_ROWS, TILE_COLS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    group_a = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[[0, 1], [1, 2]])
    group_b = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[[3, 4, 5], [0, 1, 2]])

    with pl.section_vector():
        for i in pl.range(0, DYN_NUM_TILES):
            tile_a = group_a[i % 2]
            tile_b = group_b[(i + 1) % 2]
            row = i * TILE_ROWS
            pl.load(tile_a, a, [row, 0])
            pl.load(tile_b, b, [row, 0])
            pl.add(tile_a, tile_a, tile_b)
            pl.store(out, tile_a, [row, 0])


@pl.jit(arch="3510", auto_mutex=True)
def dynamic_offset_subscript_kernel(
    a: pl.Tensor[[DYN_FULL_ROWS, TILE_COLS], pl.DT_FP16],
    b: pl.Tensor[[DYN_FULL_ROWS, TILE_COLS], pl.DT_FP16],
    out: pl.Tensor[[DYN_FULL_ROWS, TILE_COLS], pl.DT_FP16],
):
    tile_type = pl.TileType(shape=[TILE_ROWS, TILE_COLS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    group_a = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[[0, 1], [2, 3], [4, 5], [6, 7]])
    group_b = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[[6, 7], [0, 1], [2, 3], [4, 5]])

    with pl.section_vector():
        for i in pl.range(0, DYN_NUM_TILES):
            tile_a = group_a[(i + 2) % DYN_DEPTH]
            tile_b = group_b[(i * 2) % DYN_DEPTH]
            row = i * TILE_ROWS
            pl.load(tile_a, a, [row, 0])
            pl.load(tile_b, b, [row, 0])
            pl.add(tile_a, tile_a, tile_b)
            pl.store(out, tile_a, [row, 0])


@pl.jit(arch="3510", auto_mutex=True)
def pure_iterative_subscript_kernel(
    a: pl.Tensor[[PURE_FULL_ROWS, TILE_COLS], pl.DT_FP16],
    out: pl.Tensor[[PURE_FULL_ROWS, TILE_COLS], pl.DT_FP16],
):
    tile_type = pl.TileType(shape=[TILE_ROWS, TILE_COLS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    g = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[[0, 1], [2, 3], [4, 5], [6, 7]])

    with pl.section_vector():
        for i in pl.range(0, PURE_DEPTH):
            tile = g[i]
            row = i * TILE_ROWS
            pl.load(tile, a, [row, 0])
            pl.exp(tile, tile)
            pl.store(out, tile, [row, 0])


@pl.jit(arch="3510", auto_mutex=True)
def pure_constant_subscript_kernel(
    a: pl.Tensor[[TILE_ROWS, TILE_COLS], pl.DT_FP16],
    b: pl.Tensor[[TILE_ROWS, TILE_COLS], pl.DT_FP16],
    out: pl.Tensor[[TILE_ROWS, TILE_COLS], pl.DT_FP16],
):
    tile_type = pl.TileType(shape=[TILE_ROWS, TILE_COLS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    g = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])

    with pl.section_vector():
        tile_a = g[1]
        tile_b = g[4]
        pl.load(tile_a, a, [0, 0])
        pl.load(tile_b, b, [0, 0])
        pl.add(tile_a, tile_a, tile_b)
        pl.store(out, tile_a, [0, 0])


@pl.jit(arch="3510", auto_mutex=True)
def same_group_overlapping_slots_kernel(
    a: pl.Tensor[[TILE_ROWS, TILE_COLS], pl.DT_FP16],
    b: pl.Tensor[[TILE_ROWS, TILE_COLS], pl.DT_FP16],
    out: pl.Tensor[[TILE_ROWS, TILE_COLS], pl.DT_FP16],
):
    tile_type = pl.TileType(shape=[TILE_ROWS, TILE_COLS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    group = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[[0, 1], [1, 2]])

    with pl.section_vector():
        tile_a = group[0]
        tile_b = group[1]
        pl.load(tile_a, a, [0, 0])
        pl.load(tile_b, b, [0, 0])
        pl.add(tile_a, tile_a, tile_b)
        pl.store(out, tile_a, [0, 0])


@pl.jit(arch="3510", auto_mutex=True)
def control_flow_multi_id_groups_kernel(
    a: pl.Tensor[[FULL_ROWS, TILE_COLS], pl.DT_FP16],
    out: pl.Tensor[[FULL_ROWS, TILE_COLS], pl.DT_FP16],
):
    tile_type = pl.TileType(shape=[TILE_ROWS, TILE_COLS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    group_a = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[[0, 1], [2, 3]])
    group_b = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[[1, 4], [3, 5]])

    with pl.section_vector():
        for i in pl.range(0, NUM_TILES):
            slot = i % 2
            tile = group_a[slot]
            if i % 3 == 0:
                tile = group_b[slot]
            row = i * TILE_ROWS
            pl.load(tile, a, [row, 0])
            pl.add(tile, tile, tile)
            pl.store(out, tile, [row, 0])


@pl.jit(arch="3510", auto_mutex=True)
def subfunction_tile_if_else_kernel(
    true_source: pl.Tensor[[FULL_ROWS, TILE_COLS], pl.DT_FP16],
    false_source: pl.Tensor[[FULL_ROWS, TILE_COLS], pl.DT_FP16],
    out: pl.Tensor[[FULL_ROWS, TILE_COLS], pl.DT_FP16],
):
    tile_type = pl.TileType(shape=[TILE_ROWS, TILE_COLS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    true_group = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[[0, 1], [2, 3]])
    false_group = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[[1, 4], [3, 5]])
    output_group = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[[0, 6], [2, 7]])

    with pl.section_vector():
        for index in pl.range(0, NUM_TILES):
            slot = index % NUM_SLOTS
            true_tile = true_group[slot]
            false_tile = false_group[slot]
            output_tile = output_group[slot]
            row = index * TILE_ROWS
            pl.load(true_tile, true_source, [row, 0])
            pl.load(false_tile, false_source, [row, 0])
            selected_tile = _select_tile_with_if_else(true_tile, false_tile, index % 2 == 0)
            pl.add(output_tile, selected_tile, selected_tile)
            pl.store(out, output_tile, [row, 0])


@pl.jit(arch="3510", auto_mutex=True)
def tuple_mutex_ids_kernel(
    a: pl.Tensor[[TILE_ROWS, TILE_COLS], pl.DT_FP16],
    b: pl.Tensor[[TILE_ROWS, TILE_COLS], pl.DT_FP16],
    out: pl.Tensor[[TILE_ROWS, TILE_COLS], pl.DT_FP16],
):
    tile_type = pl.TileType(shape=[TILE_ROWS, TILE_COLS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    group = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=((6, 7), (8, 9)))

    with pl.section_vector():
        tile_a = group[0]
        tile_b = group[1]
        pl.load(tile_a, a, [0, 0])
        pl.load(tile_b, b, [0, 0])
        pl.add(tile_a, tile_a, tile_b)
        pl.store(out, tile_a, [0, 0])


@pl.jit(arch="3510", auto_mutex=True)
def explicit_depth_discrete_addrs_kernel(
    a: pl.Tensor[[EXPLICIT_FULL_ROWS, TILE_COLS], pl.DT_FP16],
    out: pl.Tensor[[EXPLICIT_FULL_ROWS, TILE_COLS], pl.DT_FP16],
):
    tile_type = pl.TileType(shape=[TILE_ROWS, TILE_COLS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    group = pl.make_tile_group(
        type=tile_type,
        addrs=[0x0000, 0x4000, 0x8000],
        mutex_ids=[[10, 11], [12, 13], [14, 15]],
        depth=EXPLICIT_DEPTH,
    )

    with pl.section_vector():
        for i in pl.range(0, EXPLICIT_DEPTH):
            tile = group[i]
            row = i * TILE_ROWS
            pl.load(tile, a, [row, 0])
            pl.exp(tile, tile)
            pl.store(out, tile, [row, 0])


@pl.jit(arch="3510", auto_mutex=True)
def next_and_subscript_mixed_kernel(
    a: pl.Tensor[[FULL_ROWS, TILE_COLS], pl.DT_FP16],
    out: pl.Tensor[[FULL_ROWS, TILE_COLS], pl.DT_FP16],
):
    tile_type = pl.TileType(shape=[TILE_ROWS, TILE_COLS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    g = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[[0, 1], [2, 3], [4, 5], [6, 7]])

    with pl.section_vector():
        for k in pl.range(0, NUM_TILES):
            tile = g[0]
            row = k * TILE_ROWS
            if k % 2 == 0:
                tile = g.next()
            else:
                tile = g[k % 4]
            pl.load(tile, a, [row, 0])
            pl.add(tile, tile, tile)
            pl.store(out, tile, [row, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_cube_vector_multi_mutex_id():
    _require_a5(ST_DEVICE)
    torch.manual_seed(0)
    a = torch.rand([CUBE_M, CUBE_N], device=ST_DEVICE, dtype=torch.float32) * 2.0 - 1.0
    b = torch.rand([CUBE_M, CUBE_K], device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    c = torch.rand([CUBE_K, CUBE_N], device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    out = torch.zeros([CUBE_M, CUBE_N], device=ST_DEVICE, dtype=torch.float32)
    cube_vector_multi_id_kernel(a, b, c, out)
    torch.npu.synchronize()
    golden = a.cpu().float() + (b.cpu().float() @ c.cpu().float())
    torch.testing.assert_close(out.cpu().float(), golden, rtol=3e-3, atol=3e-3)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_overlapping_groups_dynamic_subscript():
    _require_a5(ST_DEVICE)
    torch.manual_seed(0)
    a = torch.rand([DYN_FULL_ROWS, TILE_COLS], device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    b = torch.rand([DYN_FULL_ROWS, TILE_COLS], device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    out = torch.zeros([DYN_FULL_ROWS, TILE_COLS], device=ST_DEVICE, dtype=torch.float16)
    overlap_groups_add_kernel(a, b, out)
    torch.npu.synchronize()
    golden = a.cpu().float() + b.cpu().float()
    torch.testing.assert_close(out.cpu().float(), golden, rtol=3e-3, atol=3e-3)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dynamic_offset_subscript():
    _require_a5(ST_DEVICE)
    torch.manual_seed(0)
    a = torch.rand([DYN_FULL_ROWS, TILE_COLS], device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    b = torch.rand([DYN_FULL_ROWS, TILE_COLS], device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    out = torch.zeros([DYN_FULL_ROWS, TILE_COLS], device=ST_DEVICE, dtype=torch.float16)
    dynamic_offset_subscript_kernel(a, b, out)
    torch.npu.synchronize()
    golden = a.cpu().float() + b.cpu().float()
    torch.testing.assert_close(out.cpu().float(), golden, rtol=3e-3, atol=3e-3)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_pure_iterative_subscript():
    _require_a5(ST_DEVICE)
    torch.manual_seed(0)
    a = torch.rand([PURE_FULL_ROWS, TILE_COLS], device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    out = torch.zeros([PURE_FULL_ROWS, TILE_COLS], device=ST_DEVICE, dtype=torch.float16)
    pure_iterative_subscript_kernel(a, out)
    torch.npu.synchronize()
    golden = torch.exp(a.cpu().float())
    torch.testing.assert_close(out.cpu().float(), golden, rtol=3e-3, atol=3e-3)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_pure_constant_subscript():
    _require_a5(ST_DEVICE)
    torch.manual_seed(0)
    a = torch.rand([TILE_ROWS, TILE_COLS], device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    b = torch.rand([TILE_ROWS, TILE_COLS], device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    out = torch.zeros([TILE_ROWS, TILE_COLS], device=ST_DEVICE, dtype=torch.float16)
    pure_constant_subscript_kernel(a, b, out)
    torch.npu.synchronize()
    golden = a.cpu().float() + b.cpu().float()
    torch.testing.assert_close(out.cpu().float(), golden, rtol=3e-3, atol=3e-3)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_same_group_overlapping_slots_in_one_op():
    _require_a5(ST_DEVICE)
    torch.manual_seed(0)
    a = torch.rand([TILE_ROWS, TILE_COLS], device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    b = torch.rand([TILE_ROWS, TILE_COLS], device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    out = torch.zeros([TILE_ROWS, TILE_COLS], device=ST_DEVICE, dtype=torch.float16)
    same_group_overlapping_slots_kernel(a, b, out)
    torch.npu.synchronize()
    golden = a.cpu().float() + b.cpu().float()
    torch.testing.assert_close(out.cpu().float(), golden, rtol=3e-3, atol=3e-3)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_control_flow_merges_different_multi_id_groups():
    _require_a5(ST_DEVICE)
    torch.manual_seed(0)
    a = torch.rand([FULL_ROWS, TILE_COLS], device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    out = torch.zeros([FULL_ROWS, TILE_COLS], device=ST_DEVICE, dtype=torch.float16)
    control_flow_multi_id_groups_kernel(a, out)
    torch.npu.synchronize()
    golden = a.cpu().float() + a.cpu().float()
    torch.testing.assert_close(out.cpu().float(), golden, rtol=3e-3, atol=3e-3)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_subfunction_tile_if_else():
    _require_a5(ST_DEVICE)
    torch.manual_seed(0)
    true_source = torch.rand([FULL_ROWS, TILE_COLS], device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    false_source = torch.rand([FULL_ROWS, TILE_COLS], device=ST_DEVICE, dtype=torch.float16) * 2.0 + 2.0
    out = torch.zeros([FULL_ROWS, TILE_COLS], device=ST_DEVICE, dtype=torch.float16)

    subfunction_tile_if_else_kernel(true_source, false_source, out)
    torch.npu.synchronize()

    golden = true_source.cpu().float() * 2.0
    false_golden = false_source.cpu().float() * 2.0
    for index in range(1, NUM_TILES, 2):
        row = index * TILE_ROWS
        golden[row:row + TILE_ROWS] = false_golden[row:row + TILE_ROWS]
    torch.testing.assert_close(out.cpu().float(), golden, rtol=3e-3, atol=3e-3)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_tuple_mutex_ids():
    _require_a5(ST_DEVICE)
    torch.manual_seed(0)
    a = torch.rand([TILE_ROWS, TILE_COLS], device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    b = torch.rand([TILE_ROWS, TILE_COLS], device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    out = torch.zeros([TILE_ROWS, TILE_COLS], device=ST_DEVICE, dtype=torch.float16)
    tuple_mutex_ids_kernel(a, b, out)
    torch.npu.synchronize()
    golden = a.cpu().float() + b.cpu().float()
    torch.testing.assert_close(out.cpu().float(), golden, rtol=3e-3, atol=3e-3)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_explicit_depth_with_discrete_addrs():
    _require_a5(ST_DEVICE)
    torch.manual_seed(0)
    a = torch.rand([EXPLICIT_FULL_ROWS, TILE_COLS], device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    out = torch.zeros([EXPLICIT_FULL_ROWS, TILE_COLS], device=ST_DEVICE, dtype=torch.float16)
    explicit_depth_discrete_addrs_kernel(a, out)
    torch.npu.synchronize()
    golden = torch.exp(a.cpu().float())
    torch.testing.assert_close(out.cpu().float(), golden, rtol=3e-3, atol=3e-3)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_next_and_subscript_mixed():
    _require_a5(ST_DEVICE)
    torch.manual_seed(0)
    a = torch.rand([FULL_ROWS, TILE_COLS], device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    out = torch.zeros([FULL_ROWS, TILE_COLS], device=ST_DEVICE, dtype=torch.float16)
    next_and_subscript_mixed_kernel(a, out)
    torch.npu.synchronize()
    golden = a.cpu().float() + a.cpu().float()
    torch.testing.assert_close(out.cpu().float(), golden, rtol=3e-3, atol=3e-3)


@pl.jit(arch="3510", auto_mutex=True)
def l0a_l0b_overlap_multi_id_kernel(
    a: pl.Tensor[[L0A_M, L0A_K_WIDE], pl.DT_FP16],
    b: pl.Tensor[[L0B_K_WIDE, L0B_N], pl.DT_FP16],
    out: pl.Tensor[[L0A_M, L0B_N], pl.DT_FP32],
):
    l0a_g1 = pl.make_tile_group(
        type=pl.TileType(shape=[L0A_M, L0A_K_TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0000,
        mutex_ids=[0, 1],
    )
    l0a_g2 = pl.make_tile_group(
        type=pl.TileType(shape=[L0A_M, L0A_K_WIDE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0000,
        mutex_ids=[[0, 1]],
    )

    l0b_g1 = pl.make_tile_group(
        type=pl.TileType(shape=[L0B_K, L0B_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0000,
        mutex_ids=[2, 3],
    )
    l0b_g2 = pl.make_tile_group(
        type=pl.TileType(shape=[L0B_K_WIDE, L0B_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0000,
        mutex_ids=[[2, 3]],
    )

    a_l1_g1 = pl.make_tile_group(
        type=pl.TileType(shape=[L0A_M, L0A_K_TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[4, 5],
    )
    b_l1_g1 = pl.make_tile_group(
        type=pl.TileType(shape=[L0B_K, L0B_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000,
        mutex_ids=[6, 7],
    )
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[L0A_M, L0B_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[8],
    )

    with pl.section_cube():
        for i in pl.range(0, 4):
            slot = i % 2
            pl.load(a_l1_g1[slot], a, [0, slot * L0A_K_TILE])
            pl.load(b_l1_g1[slot], b, [slot * L0B_K, 0])
            pl.move(l0a_g1[slot], a_l1_g1[slot])
            pl.move(l0b_g1[slot], b_l1_g1[slot])
            if slot == 1:
                pl.matmul(acc[0], l0a_g2[0], l0b_g2[0])
                pl.store(out, acc[0], [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_l0a_l0b_overlap_multi_id():
    _require_a5(ST_DEVICE)
    torch.manual_seed(0)
    a = torch.rand([L0A_M, L0A_K_WIDE], device=ST_DEVICE, dtype=torch.float16) * 0.1
    b = torch.rand([L0B_K_WIDE, L0B_N], device=ST_DEVICE, dtype=torch.float16) * 0.1
    out = torch.zeros([L0A_M, L0B_N], device=ST_DEVICE, dtype=torch.float32)
    l0a_l0b_overlap_multi_id_kernel(a, b, out)
    torch.npu.synchronize()
    golden = a.cpu().float() @ b.cpu().float()
    torch.testing.assert_close(out.cpu().float(), golden, rtol=3e-3, atol=3e-3)


if __name__ == "__main__":
    test_mixed_single_and_multi_id_tile_groups()
    test_cube_vector_multi_mutex_id()
    test_overlapping_groups_dynamic_subscript()
    test_dynamic_offset_subscript()
    test_pure_iterative_subscript()
    test_pure_constant_subscript()
    test_same_group_overlapping_slots_in_one_op()
    test_control_flow_merges_different_multi_id_groups()
    test_subfunction_tile_if_else()
    test_tuple_mutex_ids()
    test_explicit_depth_with_discrete_addrs()
    test_next_and_subscript_mixed()
    test_l0a_l0b_overlap_multi_id()
