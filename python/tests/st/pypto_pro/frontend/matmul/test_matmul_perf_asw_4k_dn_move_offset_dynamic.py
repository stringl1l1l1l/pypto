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

"""ASW Matmul (4K-style) — FULLY DYNAMIC / ARBITRARY M/N/K via the 2D pl.move offset.

Dynamic version of ``test_matmul_perf_asw_4k_dn_move_offset.py``: ``a``/``b``/``out``
are all ``[pl.DYNAMIC, pl.DYNAMIC]`` so a single compiled kernel serves any
``[M, K] x [K, N]`` fp16 matmul (the static kernel was hard-wired to 4096^3).  Tile
*shapes* stay fixed (TILE_M=256, TILE_N=256, KL0=64, KL1=128); only the GM sizes are
dynamic, and every axis' tail is handled with ``pl.set_validshape``.

Arbitrary M/N/K:
  * K -- fully arbitrary incl. tails.  ``K_OUTER = ceil(K/KL1)`` wide GM->L1 loads; the
    last wide block narrows to ``valid_k1 = min(KL1, K - ko*KL1)`` and is split into KL0
    sub-tiles by a *runtime* inner loop ``for ki in range(ceil(valid_k1/KL0))`` (1 or 2
    iterations).  A loop (not an unrolled ``if`` that conditionally skips the 2nd sub-tile)
    keeps the L0 double-buffer ping-pong balanced so ``auto_mutex`` never deadlocks on the
    K tail.  ``phase`` is driven by the global KL0 sub-tile index ``gsub = ko*STEP_KA + ki``
    vs ``last_sub = ceil(K/KL0) - 1`` so the last non-empty sub-tile is tagged ``Final``.
  * M / N -- arbitrary (a partial edge tile is fine).  The trick is that the matmul and the
    fixpipe always operate on a **full 256x256 tile** -- a *partial* fixpipe write/read
    corrupts the tail (garbage on N) or hangs the device (partial M).  So only two things
    are narrowed: the GM **loads** clip to ``valid_m`` / ``valid_n`` (to avoid reading past
    the tensor), and the acc gets ``set_validshape([valid_m, valid_n])`` right before the
    **store** (to write only the live window).  The compute reads the tile's padding region
    (stale L1) into the matmul, producing garbage in the pad rows/cols of the accumulator --
    but those are never stored, and each acc element is an independent dot product, so the
    live ``valid_m x valid_n`` window is exact.

The ASW boustrophedon (snake) scheduling is preserved, which requires the number of
M-tiles to be a multiple of ``MAIN_WINDOW`` (4); so N is fully arbitrary and M is arbitrary
subject to ``ceil(M/256) % 4 == 0`` (same #M-tiles constraint the static kernel has).

Usage:
    python3 test_matmul_perf_asw_4k_dn_move_offset_dynamic.py
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ================================================================
#  Fixed tile configuration (shapes are compile-time; GM sizes are dynamic)
# ================================================================
TILE_M = 256
TILE_N = 256
KL0 = 64
KL1 = 128
STEP_KA = KL1 // KL0  # 2 KL0 sub-tiles per KL1 wide load
MAIN_WINDOW = 4  # ASW window: #M-tiles must be a multiple of this

# ================================================================
#  Buffer addresses (unchanged from the static kernel; tiles keep their fixed shapes).
#  L0A / L0B / L0C / L1 are FOUR separate SRAMs; each starts at 0x0.
# ================================================================
L0A_BASE = 0x0000  # 64KB in 64KB L0A
L0B_BASE = 0x0000  # 64KB in 64KB L0B
L0C_BASE = 0x0000  # 256KB in 256KB L0C
A_L1_ADDR_0 = 0x00000  # A wide region (4x64KB = 256KB)
B_L1_BASE = 0x40000  # B wide region (after A's 256KB)


@pl.jit(auto_mutex=True)
def matmul_perf_asw_4k_dn_move_offset_dynamic_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx() // pl.get_subblock_num()

    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[1]

    with pl.section_cube():
        # --- A_L1: 4-buffer, wide shape=(256, 128); K window set per block via set_validshape ---
        a_l1_wide = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE_M, KL1], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]
            ),
            addrs=A_L1_ADDR_0,
            mutex_ids=[0, 1, 10, 11],
        )
        # --- B_L1: 4-buffer, wide shape=(128, 256) ---
        b_l1_wide = pl.make_tile_group(
            type=pl.TileType(
                shape=[KL1, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]
            ),
            addrs=B_L1_BASE,
            mutex_ids=[2, 3, 12, 13],
        )
        # --- L0A: double buffer, narrow shape=(256, 64) ---
        a_left_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE_M, KL0],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Left,
                layout=pl.NZ,
                valid_shape=[-1, -1],
            ),
            addrs=L0A_BASE,
            mutex_ids=[4, 5],
        )
        # --- L0B: double buffer, narrow shape=(64, 256) ---
        b_right_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[KL0, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, valid_shape=[-1, -1]
            ),
            addrs=L0B_BASE,
            mutex_ids=[6, 7],
        )
        # --- L0C: SINGLE 256KB ---
        acc = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE_M, TILE_N],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Acc,
                fractal=1024,
                valid_shape=[-1, -1],
            ),
            addrs=L0C_BASE,
            mutex_ids=[8],
        )

        # Enable N-direction fixpipe drain
        pl.system.set_mm_layout_transform(enabled=True)

        # Runtime tile counts (ceildiv -> edge tiles covered). #M-tiles must be a multiple
        # of MAIN_WINDOW for ASW.
        m_tiles = (m + TILE_M - 1) // TILE_M
        n_tiles = (n + TILE_N - 1) // TILE_N
        total_cnt = m_tiles * n_tiles
        k_outer = (k + KL1 - 1) // KL1  # #wide KL1 loads
        k_subs = (k + KL0 - 1) // KL0  # #KL0 sub-tiles total
        last_sub = k_subs - 1  # global index of the final accumulation
        round_cnt = (total_cnt + num_cores - 1) // num_cores

        for r in pl.range(0, round_cnt):
            index = core_id + r * num_cores

            if index < total_cnt:
                row_idx = index // n_tiles // MAIN_WINDOW
                mi = row_idx * MAIN_WINDOW + index % MAIN_WINDOW
                ni_normal = (index // MAIN_WINDOW) % n_tiles

                # Boustrophedon (snake) N-walk: even rows go left->right, odd rows
                # right->left. Written branchless because a variable assigned only
                # inside if/else branches goes out of scope in the DSL codegen.
                parity = row_idx % 2
                ni = ni_normal + parity * (n_tiles - 1 - 2 * ni_normal)

                i = mi * TILE_M
                j = ni * TILE_N
                valid_m = pl.min(TILE_M, m - i)  # live rows in this (edge) M-tile
                valid_n = pl.min(TILE_N, n - j)  # live cols in this (edge) N-tile

                # ===== Outer K loop: ceil(K / KL1) wide GM->L1 loads =====
                for ko in pl.range(0, k_outer):
                    k_off = ko * KL1
                    valid_k1 = pl.min(KL1, k - k_off)  # K tail for this wide block

                    # Loads clip to the live window (valid_m x valid_k1 / valid_k1 x valid_n)
                    # so the GM read never runs past the tensor. The compute below still uses
                    # the FULL tile (the pad region holds stale L1 and is never stored).
                    a_wide_slot = a_l1_wide.next()
                    pl.set_validshape(a_wide_slot, [valid_m, valid_k1])
                    pl.load(a_wide_slot, a, [i, k_off])

                    b_wide_slot = b_l1_wide.next()
                    pl.set_validshape(b_wide_slot, [valid_k1, valid_n])
                    pl.load(b_wide_slot, b, [k_off, j])

                    # Inner KL0 sub-tile loop: n_sub = ceil(valid_k1 / KL0) is 1 or 2.
                    # A runtime loop (not an unrolled `if`) keeps the L0 double-buffer
                    # ping-pong balanced -- every iteration does exactly one .next() per
                    # side, so auto_mutex's per-slot handshakes never deadlock on the K tail.
                    n_sub = (valid_k1 + KL0 - 1) // KL0
                    for ki in pl.range(0, n_sub):
                        sub_off = ki * KL0  # 0 or KL0 (2D move start)
                        sub_k = pl.min(KL0, valid_k1 - sub_off)  # KL0, or the K-tail remainder
                        gsub = ko * STEP_KA + ki  # global KL0 sub-tile index

                        # FULL-M / FULL-N move + matmul (fixpipe must see a whole 256x256 tile).
                        pl.set_validshape(a_wide_slot, [TILE_M, sub_k])
                        cur_a_left = a_left_db.next()
                        pl.move(cur_a_left, a_wide_slot, offset=[0, sub_off])
                        pl.set_validshape(cur_a_left, [TILE_M, sub_k])
                        pl.set_validshape(b_wide_slot, [sub_k, TILE_N])
                        cur_b_right = b_right_db.next()
                        pl.move(cur_b_right, b_wide_slot, offset=[sub_off, 0])
                        pl.set_validshape(cur_b_right, [sub_k, TILE_N])

                        # gsub==0 -> first contribution (matmul, init acc); else matmul_acc.
                        # gsub==last_sub -> final K step (AccPhase.Final), else Partial.
                        if gsub == 0:
                            if gsub == last_sub:
                                pl.matmul(acc.current(), cur_a_left, cur_b_right, phase=pl.AccPhase.Final)
                            else:
                                pl.matmul(acc.current(), cur_a_left, cur_b_right, phase=pl.AccPhase.Partial)
                        else:
                            if gsub == last_sub:
                                pl.matmul_acc(
                                    acc.current(), acc.current(), cur_a_left, cur_b_right, phase=pl.AccPhase.Final
                                )
                            else:
                                pl.matmul_acc(
                                    acc.current(), acc.current(), cur_a_left, cur_b_right, phase=pl.AccPhase.Partial
                                )

                # FIX: ACC -> FP16 GM. Clip the accumulator to the live window so only
                # valid_m x valid_n is written (the pad rows/cols hold garbage).
                pl.set_validshape(acc.current(), [valid_m, valid_n])
                pl.store(out, acc.current(), [i, j], phase=pl.STPhase.Final)

        # Restore M-direction fixpipe drain
        pl.system.set_mm_layout_transform(enabled=False)


def _run_case(m, n, k):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"
    torch.manual_seed(42)

    m_tiles = (m + TILE_M - 1) // TILE_M
    assert m_tiles % MAIN_WINDOW == 0, f"ASW requires #M-tiles ({m_tiles}) to be a multiple of {MAIN_WINDOW}"

    logging.info(f"\n=== Test ASW Matmul dynamic M={m} N={n} K={k} (2D move offset, arbitrary M/N/K) ===")
    a = torch.randn(m, k, device=device, dtype=torch.float16)
    b = torch.randn(k, n, device=device, dtype=torch.float16)
    out = torch.zeros(m, n, device=device, dtype=torch.float16)

    total_cnt = m_tiles * ((n + TILE_N - 1) // TILE_N)
    num_cores = min(28, total_cnt)  # 950 默认流预算 28 cube 核
    matmul_perf_asw_4k_dn_move_offset_dynamic_kernel[None, num_cores](a, b, out)
    torch.npu.synchronize()

    golden = torch.matmul(a.float(), b.float()).half()
    max_diff = (out.float() - golden.float()).abs().max().item()
    logging.info(f"  cores={num_cores}  max abs diff: {max_diff:.4f}")
    torch.testing.assert_close(out, golden, rtol=1e-2, atol=1e-2)
    logging.info("  PASS")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_matmul_perf_asw_4k_dn_move_offset_dynamic():
    # (M, N, K): one compiled kernel serves all. N is fully arbitrary; M is arbitrary with
    # #M-tiles a multiple of 4 (ASW); K is fully arbitrary. Tails on every axis are exercised.
    cases = [
        (1024, 512, 256),  # tile-aligned baseline
        (800, 304, 192),  # M tail 32, N tail 48, K=192 -> last wide block single sub-tile
        (1024, 300, 208),  # N=300 (unaligned), K=208 -> last KL0 sub-tile partial (16)
        (1024, 500, 448),  # N=500 (unaligned), larger K
        (2048, 2048, 512),  # larger: M_TILES=8, N_TILES=8, K=4*KL1
    ]
    for m, n, k in cases:
        _run_case(m, n, k)


if __name__ == "__main__":
    logging.info("ASW Matmul 4K move-offset Test (fully dynamic / arbitrary M/N/K)")
    logging.info("=" * 60)
    test_matmul_perf_asw_4k_dn_move_offset_dynamic()
    logging.info("\nTest completed!")
