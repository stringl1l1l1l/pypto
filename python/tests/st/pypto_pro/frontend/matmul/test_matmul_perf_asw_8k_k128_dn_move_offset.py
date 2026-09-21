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

# ASW Matmul 8K-K128 — two-layer K loop using 2D pl.move offset [offset_m, offset_k].
# K_OUTER=1 (one wide GM->L1 load of 128 cols), K_INNER=2 (two L1->L0 sub-tile reads).
# Compare against test_matmul_perf_asw_8k_k128_backup.py (single-layer, 2 GM->L1 loads).
#
# Usage:

import logging
import os
import time

import pypto_pro.language as pl
import pytest
import torch

import pypto

logging.basicConfig(level=logging.INFO, format="%(message)s")

L0C_FRACTAL = int(os.environ.get("L0C_FRACTAL", "1024"))

M_SIZE = 8192
N_SIZE = 8192
K_SIZE = 128
TILE_M = 256
TILE_N = 256
KL0 = 64
KL1 = 128
STEP_KA = KL1 // KL0  # 2
NUM_CORES = 28  # 950 默认流预算 28 cube 核；请求超出预算会直接报错

M_TILES = M_SIZE // TILE_M  # 32
N_TILES = N_SIZE // TILE_N  # 32
K_OUTER = K_SIZE // KL1  # 1
K_INNER = STEP_KA  # 2
K_TOTAL = K_OUTER * K_INNER  # 2
TOTAL_CNT = M_TILES * N_TILES  # 1024

MAIN_WINDOW = 4
MAIN_ROW = M_TILES // MAIN_WINDOW - 1  # 7
TAIL_WINDOW = M_TILES - MAIN_ROW * MAIN_WINDOW
assert TAIL_WINDOW == MAIN_WINDOW

L0A_BASE = 0x0000
L0B_BASE = 0x0000
L0C_BASE = 0x0000
A_L1_ADDR_0 = 0x00000
B_L1_BASE = 0x40000


@pl.jit(auto_mutex=True)
def matmul_perf_asw_8k_k128_dn_move_offset_kernel(
    a: pl.Tensor[[M_SIZE, K_SIZE], pl.DT_FP16],
    b: pl.Tensor[[K_SIZE, N_SIZE], pl.DT_FP16],
    out: pl.Tensor[[M_SIZE, N_SIZE], pl.DT_FP16],
):

    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    rounds = (TOTAL_CNT + num_cores - 1) // num_cores

    with pl.section_cube():
        a_l1_wide = pl.make_tile_group(
            type=pl.TileType(shape=[TILE_M, KL1], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=A_L1_ADDR_0,
            mutex_ids=[0, 1, 10, 11],
        )

        b_l1_wide = pl.make_tile_group(
            type=pl.TileType(shape=[KL1, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=B_L1_BASE,
            mutex_ids=[2, 3, 12, 13],
        )

        a_left_db = pl.make_tile_group(
            type=pl.TileType(shape=[TILE_M, KL0], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=L0A_BASE,
            mutex_ids=[4, 5],
        )
        b_right_db = pl.make_tile_group(
            type=pl.TileType(shape=[KL0, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
            addrs=L0B_BASE,
            mutex_ids=[6, 7],
        )
        acc = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, fractal=L0C_FRACTAL
            ),
            addrs=L0C_BASE,
            mutex_ids=[8],
        )

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

                for ko in pl.range(0, K_OUTER):
                    k_off = ko * KL1

                    a_wide_slot = a_l1_wide.next()
                    pl.set_validshape(a_wide_slot, [TILE_M, KL1])
                    pl.load(a_wide_slot, a, [i, k_off])

                    b_wide_slot = b_l1_wide.next()
                    pl.set_validshape(b_wide_slot, [KL1, TILE_N])
                    pl.load(b_wide_slot, b, [k_off, j])

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

                    # --- ki=1: A offset [0, KL0]; B offset [KL0, 0] ---
                    cur_a_left = a_left_db.next()
                    pl.move(cur_a_left, a_wide_slot, offset=[0, KL0])
                    cur_b_right = b_right_db.next()
                    pl.move(cur_b_right, b_wide_slot, offset=[KL0, 0])

                    if ko == K_OUTER - 1:
                        pl.matmul_acc(acc.current(), acc.current(), cur_a_left, cur_b_right, phase=pl.AccPhase.Final)
                    else:
                        pl.matmul_acc(acc.current(), acc.current(), cur_a_left, cur_b_right, phase=pl.AccPhase.Partial)

                pl.store(out, acc.current(), [i, j], phase=pl.STPhase.Final)

        pl.system.set_mm_layout_transform(enabled=False)


def run_perf_test(num_iters: int = 20, warmup: int = 3):
    logging.info("=" * 60)
    logging.info("ASW Matmul 8K-K128 — two-layer K (2D move offset + L0 DB)")
    logging.info("=" * 60)
    logging.info("Shape: M=%d, N=%d, K=%d", M_SIZE, N_SIZE, K_SIZE)
    logging.info("Tile:  TILE_M=%d, TILE_N=%d, KL0=%d, KL1=%d", TILE_M, TILE_N, KL0, KL1)
    logging.info("K loop: %d outer x %d inner = %d total mmad", K_OUTER, K_INNER, K_TOTAL)
    logging.info("Tiles: %d x %d = %d", M_TILES, N_TILES, TOTAL_CNT)
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
    matmul_perf_asw_8k_k128_dn_move_offset_kernel[None, NUM_CORES](a, b, out)
    torch.npu.synchronize()
    golden = torch.matmul(a.float(), b.float()).half()
    max_diff = (out.float() - golden.float()).abs().max().item()
    logging.info("Max diff vs golden: %.6f", max_diff)
    torch.testing.assert_close(out, golden, rtol=1e-2, atol=1e-2)
    logging.info("Correctness PASS")

    logging.info("Warming up (%d iters)...", warmup)
    for _ in range(warmup):
        matmul_perf_asw_8k_k128_dn_move_offset_kernel[None, NUM_CORES](a, b, out)
    torch.npu.synchronize()

    logging.info("Running %d iters for averaging...", num_iters)
    timings_us = []
    for it in range(num_iters):
        torch.npu.synchronize()
        start = time.perf_counter()
        matmul_perf_asw_8k_k128_dn_move_offset_kernel[None, NUM_CORES](a, b, out)
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
def test_matmul_perf_asw_8k_k128_dn_move_offset():
    run_perf_test()


if __name__ == "__main__":
    run_perf_test()
