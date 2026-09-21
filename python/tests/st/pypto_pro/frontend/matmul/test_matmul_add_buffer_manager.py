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

"""Tiled Matmul + Add with L1 Double Buffer using make_tile_group + auto_mutex.

Computes: out = A @ B + X
  A: [256, 64], B: [64, 256], X: [256, 256]  (FP16 matmul ->FP32 acc)

Tiling:
  tile_m=64, tile_n=64, K=64 (no K tiling)
  M tiles = 4,  N tiles = 4  -> 16 tiles total

L1 double buffer: make_tile_group with next() rotation.
  Each iteration next() advances the cursor and returns the next tile
  (ping->pong->ping->..). No manual advance or buf_idx needed.

Synchronization: auto_mutex. The tile <-> mutex_id mapping is parser
metadata; mutex ids are fully hidden from the user.
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

M_SIZE = 256
K_SIZE = 64
N_SIZE = 256
TILE_M = 64
TILE_N = 64


@pl.jit(auto_mutex=True)
def tiled_matmul_add_db(
    a: pl.Tensor[[M_SIZE, K_SIZE], pl.DT_FP16],
    b: pl.Tensor[[K_SIZE, N_SIZE], pl.DT_FP16],
    x: pl.Tensor[[M_SIZE, N_SIZE], pl.DT_FP32],
    out: pl.Tensor[[M_SIZE, N_SIZE], pl.DT_FP32],
    workspace: pl.Tensor[[M_SIZE, N_SIZE], pl.DT_FP32],
):

    # ========== Buffer declarations ==========
    # L1 double buffer (rotate via next())
    a_l1_db = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M, K_SIZE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0, 1],
    )
    b_l1_db = pl.make_tile_group(
        type=pl.TileType(shape=[K_SIZE, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x04000,
        mutex_ids=[2, 3],
    )

    # L0A / L0B (single tile)
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M, K_SIZE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[4],
    )
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[K_SIZE, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[5],
    )

    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[6],
    )

    mm_res_ub = pl.make_tile_group(
        type=pl.TileType(shape=[32, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids=[7],
    )
    x_ub = pl.make_tile_group(
        type=pl.TileType(shape=[32, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x2000,
        mutex_ids=[8],
    )
    out_ub = pl.make_tile_group(
        type=pl.TileType(shape=[32, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000,
        mutex_ids=[9],
    )

    # ========== Cube Section ==========
    with pl.section_cube():
        for i in pl.range(0, M_SIZE, TILE_M):
            for j in pl.range(0, N_SIZE, TILE_N):
                # next() rotates: ping ->pong ->ping ->...
                cur_a = a_l1_db.next()
                cur_b = b_l1_db.next()
                al = a_left.current()
                br = b_right.current()
                ac = acc.current()

                pl.load(cur_a, a, [i, 0])
                pl.load(cur_b, b, [0, j])
                pl.move(al, cur_a)
                pl.move(br, cur_b)
                pl.matmul(ac, al, br)
                pl.store(workspace, ac, [i, j])

                # Cross-core ->Vector
                pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    # ========== Vector Section ==========
    with pl.section_vector():
        sub_index = pl.get_subblock_idx()

        for i in pl.range(0, M_SIZE, TILE_M):
            for j in pl.range(0, N_SIZE, TILE_N):
                row_off = sub_index * 32

                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=0)

                mm = mm_res_ub.current()
                xb = x_ub.current()
                ob = out_ub.current()
                pl.load(mm, workspace, [i + row_off, j])
                pl.load(xb, x, [i + row_off, j])
                pl.add(ob, mm, xb)
                pl.store(out, ob, [i + row_off, j])


# ============================================================================
# Alias kept for the end-to-end NPU test below.
# ============================================================================


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_tiled_matmul_add_db_npu():
    """End-to-end NPU test (A5 only)."""

    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"
    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, not A5. Skip.", device_name)
        return

    torch.manual_seed(42)
    a = torch.randn(M_SIZE, K_SIZE, device=device, dtype=torch.float16)
    b = torch.randn(K_SIZE, N_SIZE, device=device, dtype=torch.float16)
    x = torch.randn(M_SIZE, N_SIZE, device=device, dtype=torch.float32)
    out = torch.zeros(M_SIZE, N_SIZE, device=device, dtype=torch.float32)
    workspace = torch.zeros(M_SIZE, N_SIZE, device=device, dtype=torch.float32)

    tiled_matmul_add_db[None, 1](a, b, x, out, workspace)
    torch.npu.synchronize()

    out_ref = torch.matmul(a.float(), b.float()) + x
    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)
    logging.info("result equal!")


logging.basicConfig(level=logging.INFO, format="%(message)s")
