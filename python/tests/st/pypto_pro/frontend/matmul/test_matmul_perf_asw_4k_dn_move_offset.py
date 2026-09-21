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

"""ASW Matmul 4K — two-layer K loop using the 2D pl.move offset [offset_m, offset_k].

Shape: M=4096, N=4096, K=4096
Tile: TILE_M=256, TILE_N=256, KL0=64, KL1=128

This is the expert-recommended variant of test_matmul_perf_asw_4k_dn_src_k.py.
Instead of the 1D ``src_k_offset=`` kwarg, it uses the new 2D positional offset:

    pl.set_validshape(a_wide_slot.tile, [256, 64])          # narrow the read window
    pl.move(l0a, a_wide_slot.tile, [offset_m, offset_k]) # 2D start position

Why this form:
  - ONE wide L1 NBuffer per side (NO separate c0/c1 chunk-view buffers). The
    valid_shape chunk-view approach forced shared buf_ids across three views,
    which blocked opening an L1 4-buffer (depth capped at 3 by the 507015
    READ_OVERFLOW rule on the wide declared footprint). With a single wide tile
    + a 2D move offset, L1 can be a true 4-buffer.
  - The src tile keeps its WIDE (256,128) declared shape so the on-chip NZ
    stride stays correct; the [offset_m, offset_k] only shifts where the read
    starts. This mirrors the mainline SDK LoadData2DParamsV2 m/k start position.
  - move() with an offset lowers to pto::TEXTRACT(dst, src, offset_m, offset_k)
    (wide src -> narrow dst), so L0 stays narrow (256,64)/(64,256) = 32KB and
    double-buffers in its 64KB.

K sub-segment selection here is purely by offset:
  - A: K is dim1 (columns) -> offset = [0, ki*KL0]
  - B: K is dim0 (rows)    -> offset = [ki*KL0, 0]

Buffer / perf characteristics (same as the src_k_offset version):
  - A GM->L1: 32 loads, B GM->L1: 32 loads (vs 64 each in single-layer baseline)
  - L1 depth: 4   |   L0 DB: enabled   |   L0C single 256KB

Usage:
    msprof --output=./ python3 test_matmul_perf_asw_4k_dn_move_offset.py
"""

import logging
import os
import time

import pypto_pro.language as pl
import pytest
import torch

import pypto

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ================================================================
#  Shape configuration
# ================================================================
M_SIZE = 4096
N_SIZE = 4096
K_SIZE = 4096
TILE_M = 256
TILE_N = 256
KL0 = 64
KL1 = 128
STEP_KA = KL1 // KL0  # 2
NUM_CORES = 28  # 950 默认流预算 28 cube 核；请求超出预算会直接报错

M_TILES = M_SIZE // TILE_M  # 16
N_TILES = N_SIZE // TILE_N  # 16
K_OUTER = K_SIZE // KL1  # 32
K_INNER = STEP_KA  # 2
K_TOTAL = K_OUTER * K_INNER  # 64
TOTAL_CNT = M_TILES * N_TILES  # 256

# ASW
MAIN_WINDOW = 4
MAIN_ROW = M_TILES // MAIN_WINDOW - 1  # 3
TAIL_WINDOW = M_TILES - MAIN_ROW * MAIN_WINDOW
assert TAIL_WINDOW == MAIN_WINDOW

# ================================================================
#  Buffer addresses
#  L0A / L0B / L0C / L1 are FOUR separate SRAMs; each starts at 0x0.
# ================================================================

L0A_BASE = 0x0000  # 64KB in 64KB L0A
L0B_BASE = 0x0000  # 64KB in 64KB L0B
L0C_BASE = 0x0000  # 256KB in 256KB L0C
A_L1_ADDR_0 = 0x00000  # A wide region (4x64KB = 256KB)
B_L1_BASE = 0x40000  # B wide region (after A's 256KB)


@pl.jit(auto_mutex=True)
def matmul_perf_asw_4k_dn_move_offset_kernel(
    a: pl.Tensor[[M_SIZE, K_SIZE], pl.DT_FP16],
    b: pl.Tensor[[K_SIZE, N_SIZE], pl.DT_FP16],
    out: pl.Tensor[[M_SIZE, N_SIZE], pl.DT_FP16],
):

    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    rounds = (TOTAL_CNT + num_cores - 1) // num_cores

    with pl.section_cube():
        # --- A_L1: 4-buffer, wide shape=(256, 128) ---
        # ONE wide tile, no c0/c1 chunk views -> L1 4-buffer is unblocked.
        a_l1_wide = pl.make_tile_group(
            type=pl.TileType(shape=[TILE_M, KL1], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=A_L1_ADDR_0,
            mutex_ids=[0, 1, 10, 11],
        )

        # --- B_L1: 4-buffer, wide shape=(128, 256) ---
        b_l1_wide = pl.make_tile_group(
            type=pl.TileType(shape=[KL1, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=B_L1_BASE,
            mutex_ids=[2, 3, 12, 13],
        )

        # --- L0A: double buffer, narrow shape=(256, 64), 32KB x 2 = 64KB ---
        # TEXTRACT (emitted by an offset move) allows src wide (256,128) /
        # dst narrow (256,64), unlike TMOV which requires shape equality.
        a_left_db = pl.make_tile_group(
            type=pl.TileType(shape=[TILE_M, KL0], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=L0A_BASE,
            mutex_ids=[4, 5],
        )
        # --- L0B: double buffer, narrow shape=(64, 256), 32KB x 2 = 64KB ---
        b_right_db = pl.make_tile_group(
            type=pl.TileType(shape=[KL0, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
            addrs=L0B_BASE,
            mutex_ids=[6, 7],
        )
        # --- L0C: SINGLE 256KB ---
        acc = pl.make_tile_group(
            type=pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, fractal=1024),
            addrs=L0C_BASE,
            mutex_ids=[8],
        )

        # Enable N-direction fixpipe drain
        pl.system.set_mm_layout_transform(enabled=True)

        for r in pl.range(0, rounds):
            index = core_id + r * num_cores

            if index < TOTAL_CNT:
                row_idx = index // N_TILES // MAIN_WINDOW
                mi = row_idx * MAIN_WINDOW + index % MAIN_WINDOW
                ni_normal = (index // MAIN_WINDOW) % N_TILES

                # Boustrophedon (snake) N-walk: even rows go left->right, odd rows
                # right->left. Written branchless because a variable assigned only
                # inside if/else branches goes out of scope in the DSL codegen
                # (the generated kernel.cpp would reference an undeclared ni_0).
                parity = row_idx % 2
                ni = ni_normal + parity * (N_TILES - 1 - 2 * ni_normal)

                i = mi * TILE_M
                j = ni * TILE_N

                # ===== Outer K loop: 32 iters =====
                for ko in pl.range(0, K_OUTER):
                    k_off = ko * KL1

                    # Load A: 1 x 128 columns into the wide tile (full 256x128).
                    a_wide_slot = a_l1_wide.next()
                    pl.set_validshape(a_wide_slot, [TILE_M, KL1])
                    pl.load(a_wide_slot, a, [i, k_off])

                    # Load B: 1 x 128 rows into the wide tile (full 128x256).
                    b_wide_slot = b_l1_wide.next()
                    pl.set_validshape(b_wide_slot, [KL1, TILE_N])
                    pl.load(b_wide_slot, b, [k_off, j])

                    # Narrow the read window to one baseK sub-tile.
                    pl.set_validshape(a_wide_slot, [TILE_M, KL0])
                    pl.set_validshape(b_wide_slot, [KL0, TILE_N])

                    # --- ki=0: offset [0, 0] ---
                    cur_a_left = a_left_db.next()
                    pl.move(cur_a_left, a_wide_slot, offset=[0, 0])
                    cur_b_right = b_right_db.next()
                    pl.move(cur_b_right, b_wide_slot, offset=[0, 0])

                    if ko == 0:
                        pl.matmul(acc.current(), cur_a_left, cur_b_right, phase=pl.AccPhase.Partial)
                    else:
                        pl.matmul_acc(acc.current(), acc.current(), cur_a_left, cur_b_right, phase=pl.AccPhase.Partial)

                    # --- ki=1: A offset [0, KL0] (K=cols); B offset [KL0, 0] (K=rows) ---
                    cur_a_left = a_left_db.next()
                    pl.move(cur_a_left, a_wide_slot, offset=[0, KL0])
                    cur_b_right = b_right_db.next()
                    pl.move(cur_b_right, b_wide_slot, offset=[KL0, 0])

                    if ko == K_OUTER - 1:
                        pl.matmul_acc(acc.current(), acc.current(), cur_a_left, cur_b_right, phase=pl.AccPhase.Final)
                    else:
                        pl.matmul_acc(acc.current(), acc.current(), cur_a_left, cur_b_right, phase=pl.AccPhase.Partial)

                # FIX: ACC -> FP16 GM
                pl.store(out, acc.current(), [i, j], phase=pl.STPhase.Final)

        # Restore M-direction fixpipe drain
        pl.system.set_mm_layout_transform(enabled=False)


def run_perf_test(num_iters: int = 20, warmup: int = 3):
    logging.info("=" * 60)
    logging.info("ASW Matmul 4K — two-layer K (2D move offset [m, k] + L0 DB)")
    logging.info("=" * 60)
    logging.info("Shape: M=%d, N=%d, K=%d", M_SIZE, N_SIZE, K_SIZE)
    logging.info("Tile:  TILE_M=%d, TILE_N=%d, KL0=%d, KL1=%d", TILE_M, TILE_N, KL0, KL1)
    logging.info("K loop: %d outer x %d inner = %d total mmad", K_OUTER, K_INNER, K_TOTAL)
    logging.info("A/B GM->L1: %d loads each (vs %d in single-layer baseline)", K_OUTER, K_TOTAL)
    logging.info("L1 depth: 4 (one wide tile, no chunk views)")
    logging.info("L0 DB: enabled (offset move -> TEXTRACT, narrow dst)")
    logging.info("ASW: mainWindow=%d, MAIN_ROW=%d", MAIN_WINDOW, MAIN_ROW)
    logging.info("Requested cores: %d", NUM_CORES)
    logging.info("Iters: warmup=%d, measure=%d", warmup, num_iters)
    logging.info("-" * 60)

    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"
    torch.manual_seed(42)

    a = torch.randn(M_SIZE, K_SIZE, device=device, dtype=torch.float16)
    b = torch.randn(K_SIZE, N_SIZE, device=device, dtype=torch.float16)
    out = torch.zeros(M_SIZE, N_SIZE, device=device, dtype=torch.float16)

    logging.info("Running correctness check...")
    matmul_perf_asw_4k_dn_move_offset_kernel[None, NUM_CORES](a, b, out)
    torch.npu.synchronize()
    golden = torch.matmul(a.float(), b.float()).half()
    max_diff = (out.float() - golden.float()).abs().max().item()
    logging.info("Max diff vs golden: %.6f", max_diff)
    torch.testing.assert_close(out, golden, rtol=1e-2, atol=1e-2)
    logging.info("Correctness PASS")

    logging.info("Warming up (%d iters)...", warmup)
    for _ in range(warmup):
        matmul_perf_asw_4k_dn_move_offset_kernel[None, NUM_CORES](a, b, out)
    torch.npu.synchronize()

    logging.info("Running %d iters for averaging...", num_iters)
    timings_us = []
    for it in range(num_iters):
        torch.npu.synchronize()
        start = time.perf_counter()
        matmul_perf_asw_4k_dn_move_offset_kernel[None, NUM_CORES](a, b, out)
        torch.npu.synchronize()
        end = time.perf_counter()
        elapsed_us = (end - start) * 1e6
        timings_us.append(elapsed_us)
        logging.info("  iter %2d: %.3f us", it, elapsed_us)

    timings_us.sort()
    avg_us = sum(timings_us) / len(timings_us)
    median_us = timings_us[len(timings_us) // 2]
    min_us = timings_us[0]
    max_us = timings_us[-1]
    trim = max(1, len(timings_us) // 5)
    trimmed = timings_us[trim:-trim] if len(timings_us) > 2 * trim else timings_us
    trimmed_avg_us = sum(trimmed) / len(trimmed)

    flops = 2 * M_SIZE * N_SIZE * K_SIZE
    tflops = flops / (trimmed_avg_us / 1e6) / 1e12

    logging.info("-" * 60)
    logging.info("Results (us, %d iters):", num_iters)
    logging.info("  min:          %.3f", min_us)
    logging.info("  median:       %.3f", median_us)
    logging.info("  mean:         %.3f", avg_us)
    logging.info("  trimmed mean: %.3f", trimmed_avg_us)
    logging.info("  max:          %.3f", max_us)
    logging.info("  Throughput:   %.3f TFLOPS", tflops)
    logging.info("=" * 60)

    return trimmed_avg_us, tflops


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_matmul_perf_asw_4k_dn_move_offset():
    run_perf_test()


if __name__ == "__main__":
    run_perf_test()
