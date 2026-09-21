# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 end-to-end tests for SIMT UB Tile access, runtime Valid Shape, and dynamic UB inference."""

import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
TILE_ROWS = 8
TILE_COLS = 64
VALID_ROWS = 6
VALID_COLS = 37
THREADS = 256
TILE_BYTES = TILE_ROWS * TILE_COLS * 4
MIXED_ELEMENTS = 1024
MIXED_THREADS = 1024
MIXED_TILE_BYTES = MIXED_ELEMENTS * 4
MIXED_TILE_HIGH_WATER = 24 * 1024


def _require_a5():
    try:
        torch.npu.set_device(ST_DEVICE)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


@pl.vector_function(mode="simt", max_threads=THREADS)
def ub_tile_add(
    dst,
    src,
    delta: pl.DT_FP32,
):
    tid = pl.simt.linear_thread_idx()
    rows = src.valid_shape[0]
    cols = src.valid_shape[1]
    row = tid // cols
    col = tid % cols
    if row < rows:
        dst[row, col] = src[row, col] + delta


@pl.vector_function(mode="simt", max_threads=MIXED_THREADS)
def copy_from_gm(src, out):
    tid = pl.simt.linear_thread_idx()
    out[0, tid] = src[0, tid]


@pl.jit()
def simt_ub_tile_access(
    x: pl.Tensor[[VALID_ROWS, VALID_COLS], pl.DT_FP32],
    out: pl.Tensor[[VALID_ROWS, VALID_COLS], pl.DT_FP32],
    valid_rows: pl.DT_UINT32,
    valid_cols: pl.DT_UINT32,
    delta: pl.DT_FP32,
):
    tile_type = pl.TileType(
        shape=[TILE_ROWS, TILE_COLS],
        dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
    )
    src = pl.make_tile(tile_type, addr=0x0000)
    dst = pl.make_tile(tile_type, addr=0x0800)
    with pl.section_vector():
        pl.set_validshape(src, [valid_rows, valid_cols])
        pl.set_validshape(dst, [valid_rows, valid_cols])
        pl.load(src, x, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        ub_tile_add[THREADS](dst, src, delta)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, dst, [0, 0])


@pl.jit()
def mixed_simd_high_water_not_passed_to_simt(
    src: pl.Tensor[[1, MIXED_ELEMENTS], pl.DT_FP32],
    simd_out: pl.Tensor[[1, MIXED_ELEMENTS], pl.DT_FP32],
    simt_out: pl.Tensor[[1, MIXED_ELEMENTS], pl.DT_FP32],
):
    tile_type = pl.TileType(
        shape=[1, MIXED_ELEMENTS],
        dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec,
    )
    scratch_group = pl.make_tile_group(
        type=tile_type,
        addrs=MIXED_TILE_HIGH_WATER - MIXED_TILE_BYTES,
        mutex_ids=[0],
    )
    scratch = scratch_group.current()
    with pl.section_vector():
        pl.load(scratch, src, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.add(scratch, scratch, 0.0)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(simd_out, scratch, [0, 0])
        copy_from_gm[MIXED_THREADS](src, simt_out)


@pytest.mark.soc("950")
@pytest.mark.parametrize(("valid_rows", "valid_cols"), [(1, 1), (VALID_ROWS, VALID_COLS)])
def test_ub_tile_access(valid_rows, valid_cols):
    _require_a5()

    delta = 0.75
    sentinel = -7.0
    x = torch.arange(VALID_ROWS * VALID_COLS, dtype=torch.float32).reshape(VALID_ROWS, VALID_COLS).to(ST_DEVICE)
    out = torch.full_like(x, sentinel)

    simt_ub_tile_access(x, out, valid_rows, valid_cols, delta)
    torch.npu.synchronize()

    expected = torch.full((VALID_ROWS, VALID_COLS), sentinel, dtype=torch.float32)
    expected[:valid_rows, :valid_cols] = x.cpu()[:valid_rows, :valid_cols] + delta
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_mixed_kernel_infers_simd_tile_high_water_not_passed_to_simt():
    _require_a5()

    src = torch.arange(MIXED_ELEMENTS, dtype=torch.float32).reshape(1, MIXED_ELEMENTS)
    src_device = src.to(ST_DEVICE)
    simd_out = torch.full_like(src_device, -1.0)
    simt_out = torch.full_like(src_device, -1.0)

    mixed_simd_high_water_not_passed_to_simt(src_device, simd_out, simt_out)
    torch.npu.synchronize()

    torch.testing.assert_close(simd_out.cpu(), src, rtol=0, atol=0)
    torch.testing.assert_close(simt_out.cpu(), src, rtol=0, atol=0)


if __name__ == "__main__":
    test_ub_tile_access(VALID_ROWS, VALID_COLS)
