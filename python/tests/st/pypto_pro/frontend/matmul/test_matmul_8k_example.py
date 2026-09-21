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

from dataclasses import dataclass
import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto


def store_to_out(out, acc_tile, offset):
    pl.store_tile(out, acc_tile, offset)


@dataclass
class OpTiling:
    valid_size: int


@pl.jit(auto_mutex=True)
def matmul_example(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    tiling: OpTiling,
):
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    valid_n = 128
    with pl.section_cube():
        a_mat_4_buffer = pl.make_tile_group(
            type=pl.TileType(
                shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, valid_shape=[valid_n, valid_n]
            ),
            addrs=0,
            mutex_ids=[0, 1, 10, 11],
        )
        b_mat_4_buffer = pl.make_tile_group(
            type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0x20000,
            mutex_ids=[2, 3, 12, 13],
        )
        a_left_db = pl.make_tile_group(
            type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
            addrs=0,
            mutex_ids=[4, 5],
        )
        b_right_db = pl.make_tile_group(
            type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
            addrs=0,
            mutex_ids=[6, 7],
        )
        acc_db = pl.make_tile_group(
            type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
            addrs=0,
            mutex_ids=[8, 9],
        )

        for i in pl.range(core_id, a.shape[0] // 128, num_cores):
            a_l1_tile = a_mat_4_buffer.next()
            pl.load_tile(a_l1_tile, a, [i, 0])
            compute(a_l1_tile, a_mat_4_buffer, b_mat_4_buffer, a_left_db, b_right_db, acc_db, a, b, out, i)


def compute(a_l1_tile, a_mat_4_buffer, b_mat_4_buffer, a_left_db, b_right_db, acc_db, a, b, out, i):
    for j in pl.range(0, b.shape[1] // 128, 1):
        a_l1_tile = a_mat_4_buffer.current()
        b_l1_tile = b_mat_4_buffer.next()
        pl.load_tile(b_l1_tile, b, [0, j])

        cur_a_left = a_left_db.next()
        pl.move(cur_a_left, a_l1_tile)
        cur_b_right = b_right_db.next()
        pl.move(cur_b_right, b_l1_tile)

        acc_tile = acc_db.next()
        pl.matmul(acc_tile, cur_a_left, cur_b_right)

        store_to_out(out, acc_tile, [i, j])


def run_perf_test(num_iters: int = 20, warmup: int = 3):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"
    torch.manual_seed(42)
    m_size = 8192
    n_size = 8192
    k_size = 128
    a = torch.randn(m_size, k_size, device=device, dtype=torch.float16)
    b = torch.randn(k_size, n_size, device=device, dtype=torch.float16)
    out = torch.zeros(m_size, n_size, device=device, dtype=torch.float16)
    core_num = 28  # 950 默认流预算 28 cube 核；请求超出预算会直接报错
    tiling = OpTiling(valid_size=128)
    matmul_example[None, core_num](a, b, out, tiling)
    torch.npu.synchronize()
    golden = torch.matmul(a.float(), b.float()).half()
    max_diff = (out.float() - golden.float()).abs().max().item()
    logging.info("Max diff vs golden: %.6f", max_diff)
    torch.testing.assert_close(out, golden, rtol=1e-2, atol=1e-2)
    logging.info("Correctness PASS")
    return


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_matmul_perf_asw_8k_k128_dn_move_offset():
    run_perf_test()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_perf_test()
