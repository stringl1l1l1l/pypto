# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Move + phase integration test.

Verifies that pl.move with STPhase.Final correctly moves Acc->Vec and
stores the result without deadlock on Ascend 950. The kernel performs
K-dimension accumulated matmul (phase Partial/Final) then moves the
accumulator to Vec memory via phase=STPhase.Final before storing.
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

TILE = 128
K_SIZE_ACC = 256     # 2 K-blocks for accumulation


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


@pl.jit(auto_mutex=True)
def move_phase_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[K_SIZE_ACC, TILE], pl.DT_FP16],
    out: pl.Tensor[[TILE, TILE], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[0, 1])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000, mutex_ids=[2, 3])
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000, mutex_ids=[4, 5])
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[6, 7])
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, fractal=1024),
        addrs=0x0000, mutex_ids=[8])
    vec_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000, mutex_ids=[9])
    mat_res = vec_tile_group.current()
    with pl.section_cube():
        pl.system.set_mm_layout_transform(enabled=True)
        ac = c_l0c.current()
        for k in pl.range(0, K_SIZE_ACC, TILE):
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_l0a.next()
            br = b_l0b.next()
            pl.load(cur_a, a, [0, k])
            pl.load(cur_b, b, [k, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if k == 0:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
        pl.move(mat_res, ac, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0, phase=pl.STPhase.Final)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)
        pl.system.set_mm_layout_transform(enabled=False)
    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
        if sub_id == 0:
            pl.store(out, mat_res, [0, 0])
        pl.system.bar_all()


@pytest.mark.soc("950")
def test_move_phase():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn(TILE, K_SIZE_ACC, device=device, dtype=torch.float16)
    b = torch.randn(K_SIZE_ACC, TILE, device=device, dtype=torch.float16)
    c = torch.zeros(TILE, TILE, device=device, dtype=torch.float32)

    move_phase_kernel(a, b, c)
    torch.npu.synchronize()

    ref = torch.matmul(a.float(), b.float())
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
    logging.info("move + phase result equal!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
