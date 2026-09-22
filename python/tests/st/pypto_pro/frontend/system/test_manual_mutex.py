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
"""End-to-end checks for the manual-synchronization documentation examples.

The kernels in this file intentionally match the examples in:

* mutex_lock.md
* mutex_unlock.md
* sync_all.md

They cover ordinary operator flows rather than boundary IDs, dynamic-ID forms,
or deliberately adversarial control-flow constructions.
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


def _require_a5() -> None:
    try:
        torch.npu.set_device(ST_DEVICE)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        pytest.skip(f"Current device is {device_name}, not A5 (Ascend950). Skip.")


# mutex_lock.md and mutex_unlock.md: one input UB crosses MTE2 -> V and one output
# UB crosses V -> MTE3. No flag synchronization or auto_mutex is present, so
# the result exercises the manually inserted mutex operations directly.
@pl.jit(auto_mutex=False)
def mutex_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_x = pl.make_tile(tt, addr=0x0000)
    tile_out = pl.make_tile(tt, addr=0x4000)
    with pl.section_vector():
        pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=0)
        pl.load(tile_x, x, [0, 0])
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=0)

        pl.system.mutex_lock(pipe=pl.PipeType.V, mutex_id=0)
        pl.system.mutex_lock(pipe=pl.PipeType.V, mutex_id=1)
        pl.add(tile_out, tile_x, tile_x)
        pl.system.mutex_unlock(pipe=pl.PipeType.V, mutex_id=1)
        pl.system.mutex_unlock(pipe=pl.PipeType.V, mutex_id=0)

        pl.system.mutex_lock(pipe=pl.PipeType.MTE3, mutex_id=1)
        pl.store(out, tile_out, [0, 0])
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE3, mutex_id=1)


# sync_all.md: a conventional double-buffered vector loop. Each AIV writes
# disjoint rows, and every participating AIV reaches one loop-external barrier.
@pl.jit(auto_mutex=True)
def sync_all_kernel(
    x: pl.Tensor[[2048, 64], pl.DT_FP32],
    out: pl.Tensor[[2048, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    input_tiles = pl.make_tile_group(type=tt, addrs=[0x0000, 0x0100], mutex_ids=[0, 1])
    output_tiles = pl.make_tile_group(type=tt, addrs=[0x0200, 0x0300], mutex_ids=[2, 3])

    with pl.section_vector():
        for row in pl.range(pl.get_block_idx(), x.shape[0], pl.get_block_num()):
            tile_x = input_tiles.next()
            tile_out = output_tiles.next()
            pl.load(tile_x, x, [row, 0])
            pl.add(tile_out, tile_x, tile_x)
            pl.store(out, tile_out, [row, 0])

        pl.system.sync_all(
            core_type=pl.SyncCoreType.AIV_ONLY,
        )


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_mutex_lock_unlock_doc_example():
    """Manual mutex is the only MTE2 -> V -> MTE3 ordering mechanism."""
    _require_a5()
    torch.manual_seed(0)
    x = torch.randn((64, 64), device=ST_DEVICE, dtype=torch.float32)
    out = torch.empty_like(x)

    mutex_kernel[None, 1](x, out)
    torch.npu.synchronize()

    torch.testing.assert_close(out, x + x, rtol=1e-4, atol=1e-4)
    logging.info("mutex_lock/mutex_unlock documentation example passed")


@pytest.mark.soc("950")
@pytest.mark.parametrize("used_cores", [1, 2], ids=["one-core", "two-cores"])
@pypto.options(pass_options={"enable_slice": False})
def test_sync_all_doc_example(used_cores):
    """All launched AIVs finish normal work and leave the barrier without hanging.

    This case checks barrier participation/liveness and operator precision. It
    does not claim to verify cross-core GM data visibility.
    """
    _require_a5()
    torch.manual_seed(0)
    x = torch.randn((2048, 64), device=ST_DEVICE, dtype=torch.float32)
    out = torch.empty_like(x)

    sync_all_kernel[None, used_cores](x, out)
    torch.npu.synchronize()

    torch.testing.assert_close(out, x + x, rtol=1e-4, atol=1e-4)
    logging.info("sync_all documentation example passed with %d core(s)", used_cores)
