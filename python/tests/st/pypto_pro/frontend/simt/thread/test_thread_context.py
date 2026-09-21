# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 end-to-end test for SIMT thread, block, and grid context interfaces."""

import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
THREADS_X = 4
THREADS_Y = 2
THREADS_Z = 2
THREADS = THREADS_X * THREADS_Y * THREADS_Z


def _require_a5():
    try:
        torch.npu.set_device(ST_DEVICE)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


@pl.vector_function(mode="simt", max_threads=THREADS)
def write_thread_context(dst):
    thread = pl.simt.thread_idx()
    block = pl.simt.block_dim()
    block_id = pl.simt.block_idx()
    grid = pl.simt.grid_dim()
    tid = pl.simt.linear_thread_idx()
    warp = pl.simt.cast(pl.simt.warp_size(), pl.DT_UINT32)
    coordinate_tid = thread.x + thread.y * block.x + thread.z * block.x * block.y
    value = (
        thread.x
        + thread.y * 4
        + thread.z * 8
        + block.x * 16
        + block.y * 128
        + block.z * 256
        + block_id.x * 512
        + block_id.y * 4096
        + block_id.z * 8192
        + grid.x * 16384
        + grid.y * 131072
        + grid.z * 262144
    )
    if tid == coordinate_tid:
        dst[0, tid] = value
        dst[1, tid] = warp


@pl.jit()
def simt_thread_context(out: pl.Tensor[[2, THREADS], pl.DT_UINT32]):
    tile_type = pl.TileType(shape=[2, THREADS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    dst = pl.make_tile(tile_type, addr=0x0000)
    with pl.section_vector():
        write_thread_context[THREADS_X, THREADS_Y, THREADS_Z](dst)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        pl.store(out, dst, [0, 0])


@pytest.mark.soc("950")
def test_thread_context():
    _require_a5()

    out = torch.empty((2, THREADS), dtype=torch.int32).to(torch.uint32).to(ST_DEVICE)
    simt_thread_context[None, 1](out)
    torch.npu.synchronize()

    tid = torch.arange(THREADS, dtype=torch.int64)
    thread_x = tid % THREADS_X
    thread_y = (tid // THREADS_X) % THREADS_Y
    thread_z = tid // (THREADS_X * THREADS_Y)
    expected = (
        thread_x
        + thread_y * 4
        + thread_z * 8
        + THREADS_X * 16
        + THREADS_Y * 128
        + THREADS_Z * 256
        + 16384
        + 131072
        + 262144
    )

    expected = torch.stack((expected, torch.full_like(expected, 32)))
    torch.testing.assert_close(out.cpu().to(torch.int64), expected, rtol=0, atol=0)


if __name__ == "__main__":
    test_thread_context()
