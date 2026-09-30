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

"""ST coverage for ``pl.move`` with mixed Tile and TileGroup operands.

``pl.move`` uses ``(dst, src)`` argument order.  The cases below cover:

* Cube ``move(tile, tile_group)`` and ``move(tile_group, tile)`` on MTE1.
* Vector ``tile_group -> tile -> tile_group`` on PIPE_V.
* Cube/Vector ``Acc tile_group -> Vec tile`` on PIPE_FIX.

Each managed operand under test uses a single-slot TileGroup repeatedly.  The
pure-pipeline cases depend on auto_mutex selecting the move pipe from both
operands even when the other operand is an ordinary Tile.  Explicit
synchronization is used for dependencies owned by ordinary Tiles and for
Cube/Vector cross-core publication and reuse.
"""

import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

CUBE_TILE = 64
CUBE_M = CUBE_TILE * 4
CUBE_K = CUBE_TILE
CUBE_N = CUBE_TILE

VEC_TILE_M = 64
VEC_TILE_N = 128
VEC_NUM_TILES = 8
VEC_M = VEC_TILE_M * VEC_NUM_TILES

CV_TILE_M = 64
CV_TILE_K = 64
CV_TILE_N = 64
CV_VEC_M = CV_TILE_M // 2
CV_NUM_TILES = 4
CV_NUM_CORES = 2
CV_M = CV_TILE_M * CV_NUM_TILES


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


def _make_fp16_pattern(shape, modulus, device=ST_DEVICE):
    """Build deterministic FP16 data without a float or double scalar path."""
    numel = 1
    for extent in shape:
        numel *= extent
    host = torch.arange(numel, device="cpu", dtype=torch.int32)
    host.remainder_(modulus)
    return host.reshape(shape).to(device=device, dtype=torch.float16)


# =============================================================================
# Pure Cube: move(tile, tile_group)
# =============================================================================
@pl.jit(arch="3510", auto_mutex=True)
def cube_move_tile_group_to_tile_kernel(
    a: pl.Tensor[[CUBE_M, CUBE_K], pl.DT_FP16],
    b: pl.Tensor[[CUBE_K, CUBE_N], pl.DT_FP16],
    out: pl.Tensor[[CUBE_M, CUBE_N], pl.DT_FP32],
):
    """Read a managed Mat tile into an ordinary Left tile."""
    a_l1_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[CUBE_TILE, CUBE_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x0000,
        mutex_ids=[0],
    )
    a_l0a_tile = pl.make_tile(
        pl.TileType(
            shape=[CUBE_TILE, CUBE_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addr=0x0000,
    )
    b_l1_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[CUBE_K, CUBE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x4000,
        mutex_ids=[1],
    )
    b_l0b_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[CUBE_K, CUBE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[2],
    )
    acc_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[CUBE_TILE, CUBE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[3],
    )

    with pl.section_cube():
        for row in pl.range(0, CUBE_M, CUBE_TILE):
            a_l1 = a_l1_group.current()
            b_l1 = b_l1_group.current()
            b_l0b = b_l0b_group.current()
            acc = acc_group.current()

            pl.load(a_l1, a, [row, 0])
            pl.load(b_l1, b, [0, 0])
            pl.move(a_l0a_tile, a_l1)
            pl.move(b_l0b, b_l1)

            # a_l0a_tile is ordinary: synchronize its MTE1 -> M dependency manually.
            pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
            pl.matmul(acc, a_l0a_tile, b_l0b)
            # Protect the same ordinary L0A tile before the next MTE1 overwrite.
            pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=1)
            pl.store(out, acc, [row, 0])


# =============================================================================
# Pure Cube: move(tile_group, tile)
# =============================================================================
@pl.jit(arch="3510", auto_mutex=True)
def cube_move_tile_to_tile_group_kernel(
    a: pl.Tensor[[CUBE_M, CUBE_K], pl.DT_FP16],
    b: pl.Tensor[[CUBE_K, CUBE_N], pl.DT_FP16],
    out: pl.Tensor[[CUBE_M, CUBE_N], pl.DT_FP32],
):
    """Load each ordinary Mat tile, move it to managed Left, and run matmul."""
    a_l1_tile = pl.make_tile(
        pl.TileType(
            shape=[CUBE_TILE, CUBE_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addr=0x0000,
    )
    a_l0a_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[CUBE_TILE, CUBE_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[0],
    )
    b_l1_tile = pl.make_tile(
        pl.TileType(
            shape=[CUBE_K, CUBE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addr=0x4000,
    )
    b_l0b_tile = pl.make_tile(
        pl.TileType(
            shape=[CUBE_K, CUBE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addr=0x0000,
    )
    acc_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[CUBE_TILE, CUBE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[1, 2, 3, 4],
    )

    with pl.section_cube():
        # Prepare the loop-invariant ordinary B tiles before the mixed A pipeline.
        pl.load(b_l1_tile, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.move(b_l0b_tile, b_l1_tile)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=1)

        for row in pl.range(0, CUBE_M, CUBE_TILE):
            a_l0a = a_l0a_group.current()
            acc = acc_group.next()

            pl.load(a_l1_tile, a, [row, 0])
            # Protect the ordinary source's load -> mixed-move dependency.
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=2)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=2)
            pl.move(a_l0a, a_l1_tile)
            # Protect the ordinary source before the next loop overwrites it.
            pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=3)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=3)
            pl.matmul(acc, a_l0a, b_l0b_tile)
            pl.store(out, acc, [row, 0])


# =============================================================================
# Pure Vector: group -> tile -> group
# =============================================================================
@pl.jit(arch="3510", auto_mutex=True)
def vector_mixed_move_both_directions_kernel(
    source: pl.Tensor[[VEC_M, VEC_TILE_N], pl.DT_FP16],
    middle_out: pl.Tensor[[VEC_M, VEC_TILE_N], pl.DT_FP16],
    out: pl.Tensor[[VEC_M, VEC_TILE_N], pl.DT_FP16],
):
    """Exercise both mixed move directions and manually synchronize the ordinary Tile."""
    tile_type = pl.TileType(
        shape=[VEC_TILE_M, VEC_TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    source_group = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])
    middle_tile = pl.make_tile(
        tile_type,
        addr=0x8000,
    )
    out_group = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[1])

    with pl.section_vector():
        for row in pl.range(0, VEC_M, VEC_TILE_M):
            source_tile = source_group.current()
            out_tile = out_group.current()
            pl.load(source_tile, source, [row, 0])
            pl.move(middle_tile, source_tile)

            # middle_tile is ordinary: protect its V -> MTE3 access and reuse manually.
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=4)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=4)
            pl.store(middle_out, middle_tile, [row, 0])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=5)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=5)

            pl.move(out_tile, middle_tile)
            pl.add(out_tile, out_tile, out_tile)
            pl.store(out, out_tile, [row, 0])


# =============================================================================
# Cube/Vector: move(Vec tile, Acc tile_group)
# =============================================================================
@pl.jit(arch="3510", auto_mutex=True)
def cv_move_acc_group_to_vec_tile_kernel(
    a: pl.Tensor[[CV_M, CV_TILE_K], pl.DT_FP16],
    b: pl.Tensor[[CV_TILE_K, CV_TILE_N], pl.DT_FP16],
    out: pl.Tensor[[CV_M, CV_TILE_N], pl.DT_FP32],
):
    """Reuse one managed Acc tile across two iterations per Cube core."""
    a_l1_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[CV_TILE_M, CV_TILE_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x0000,
        mutex_ids=[0],
    )
    b_l1_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[CV_TILE_K, CV_TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x4000,
        mutex_ids=[1],
    )
    a_l0a_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[CV_TILE_M, CV_TILE_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[CV_TILE_K, CV_TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    acc_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[CV_TILE_M, CV_TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[8],
    )
    vec_tile = pl.make_tile(
        pl.TileType(
            shape=[CV_VEC_M, CV_TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x0000,
    )

    with pl.section_cube():
        core_id = pl.get_block_idx()
        for tile_index in pl.range(core_id, CV_NUM_TILES, CV_NUM_CORES):
            row = tile_index * CV_TILE_M
            a_l1 = a_l1_group.current()
            b_l1 = b_l1_group.current()
            a_l0a = a_l0a_group.current()
            b_l0b = b_l0b_group.current()
            acc = acc_group.current()

            pl.load(a_l1, a, [row, 0])
            pl.load(b_l1, b, [0, 0])
            pl.move(a_l0a, a_l1)
            pl.move(b_l0b, b_l1)
            pl.matmul(acc, a_l0a, b_l0b)

            # The Vector halves acknowledge the previous use of the ordinary Vec tile.
            pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=6)
            pl.move(vec_tile, acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    with pl.section_vector():
        subblock_id = pl.get_subblock_idx()
        vector_core_id = pl.get_block_idx() // pl.get_subblock_num()
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=6)

        for tile_index in pl.range(vector_core_id, CV_NUM_TILES, CV_NUM_CORES):
            row = tile_index * CV_TILE_M + subblock_id * CV_VEC_M
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
            pl.store(out, vec_tile, [row, 0])
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=6)


def _cube_inputs():
    a_host = _make_fp16_pattern([CUBE_M, CUBE_K], 17, device="cpu")
    b_host = _make_fp16_pattern([CUBE_K, CUBE_N], 13, device="cpu")
    golden = a_host.to(dtype=torch.float32) @ b_host.to(dtype=torch.float32)
    a = a_host.to(device=ST_DEVICE)
    b = b_host.to(device=ST_DEVICE)
    out = torch.zeros([CUBE_M, CUBE_N], device=ST_DEVICE, dtype=torch.float32)
    return a, b, out, golden


def _assert_matmul_close(out, golden):
    actual = out.detach().cpu()
    scale = torch.tensor(1000, device="cpu", dtype=torch.float32)
    margin = torch.tensor(3, device="cpu", dtype=torch.float32)
    difference = torch.abs(actual - golden)
    scaled_tolerance = margin + margin * torch.abs(golden)
    mismatch = ~torch.isfinite(actual) | (difference * scale > scaled_tolerance)
    if torch.any(mismatch):
        mismatch_count = mismatch.sum().item()
        max_difference = difference.max().item()
        pytest.fail(
            f"Tensor-likes are not close: {mismatch_count} mismatched element(s), "
            f"greatest absolute difference is {max_difference}."
        )


@pytest.mark.soc("950")
def test_cube_move_tile_group_to_tile():
    _require_a5(ST_DEVICE)
    a, b, out, golden = _cube_inputs()
    cube_move_tile_group_to_tile_kernel[None, 1](a, b, out)
    torch.npu.synchronize()
    _assert_matmul_close(out, golden)


@pytest.mark.soc("950")
def test_cube_move_tile_to_tile_group():
    _require_a5(ST_DEVICE)
    a, b, out, golden = _cube_inputs()
    cube_move_tile_to_tile_group_kernel[None, 1](a, b, out)
    torch.npu.synchronize()
    _assert_matmul_close(out, golden)


@pytest.mark.soc("950")
def test_vector_mixed_move_both_directions():
    _require_a5(ST_DEVICE)
    source = _make_fp16_pattern([VEC_M, VEC_TILE_N], 19)
    middle_out = torch.zeros([VEC_M, VEC_TILE_N], device=ST_DEVICE, dtype=torch.float16)
    out = torch.zeros([VEC_M, VEC_TILE_N], device=ST_DEVICE, dtype=torch.float16)
    vector_mixed_move_both_directions_kernel[None, 1](source, middle_out, out)
    torch.npu.synchronize()
    torch.testing.assert_close(middle_out.cpu().float(), source.cpu().float(), rtol=3e-3, atol=3e-3)
    source_host = source.cpu().float()
    torch.testing.assert_close(out.cpu().float(), source_host + source_host, rtol=3e-3, atol=3e-3)


def _cv_inputs():
    a_host = _make_fp16_pattern([CV_M, CV_TILE_K], 17, device="cpu")
    b_host = _make_fp16_pattern([CV_TILE_K, CV_TILE_N], 13, device="cpu")
    golden = a_host.to(dtype=torch.float32) @ b_host.to(dtype=torch.float32)
    a = a_host.to(device=ST_DEVICE)
    b = b_host.to(device=ST_DEVICE)
    out = torch.zeros([CV_M, CV_TILE_N], device=ST_DEVICE, dtype=torch.float32)
    return a, b, out, golden


@pytest.mark.soc("950")
def test_cv_move_acc_group_to_vec_tile():
    _require_a5(ST_DEVICE)
    a, b, out, golden = _cv_inputs()
    cv_move_acc_group_to_vec_tile_kernel[None, CV_NUM_CORES](a, b, out)
    torch.npu.synchronize()
    _assert_matmul_close(out, golden)


if __name__ == "__main__":
    test_cube_move_tile_group_to_tile()
    test_cube_move_tile_to_tile_group()
    test_vector_mixed_move_both_directions()
    test_cv_move_acc_group_to_vec_tile()
