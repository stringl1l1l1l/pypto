# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 end-to-end tests for the SIMT Lane interfaces."""

import os

import pypto_pro.language as pl
from pypto_pro.runtime.platform import NpuArch
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
WARP_SIZE = 32

@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def write_lane_id(output):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.lane_id()

@pl.jit(arch=NpuArch.DAV_3510, auto_mutex=True)
def simt_lane_id(output: pl.Tensor[[1, WARP_SIZE], pl.DT_INT32]):
    output_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, WARP_SIZE], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    output_tile = output_tile_group.current()
    with pl.section_vector():
        write_lane_id[WARP_SIZE](output_tile)
        pl.store(output, output_tile, [0, 0])

@pytest.mark.soc("950")
def test_lane_id():
    torch.npu.set_device(ST_DEVICE)
    expected = torch.arange(WARP_SIZE, dtype=torch.int32).reshape(1, WARP_SIZE)
    actual = torch.empty_like(expected).to(ST_DEVICE)
    simt_lane_id(actual)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

# -----------------------------------------------------------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def write_lanemask_eq(output):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.lanemask_eq()

@pl.jit(arch=NpuArch.DAV_3510, auto_mutex=True)
def simt_lanemask_eq(output: pl.Tensor[[1, WARP_SIZE], pl.DT_INT32]):
    output_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, WARP_SIZE], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    output_tile = output_tile_group.current()
    with pl.section_vector():
        write_lanemask_eq[WARP_SIZE](output_tile)
        pl.store(output, output_tile, [0, 0])

@pytest.mark.soc("950")
def test_lanemask_eq():
    torch.npu.set_device(ST_DEVICE)
    lane = torch.arange(WARP_SIZE, dtype=torch.int64)
    expected = (1 << lane).to(torch.int32).reshape(1, WARP_SIZE)
    actual = torch.empty_like(expected).to(ST_DEVICE)
    simt_lanemask_eq(actual)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

# -----------------------------------------------------------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def write_lanemask_ge(output):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.lanemask_ge()

@pl.jit(arch=NpuArch.DAV_3510, auto_mutex=True)
def simt_lanemask_ge(output: pl.Tensor[[1, WARP_SIZE], pl.DT_INT32]):
    output_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, WARP_SIZE], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    output_tile = output_tile_group.current()
    with pl.section_vector():
        write_lanemask_ge[WARP_SIZE](output_tile)
        pl.store(output, output_tile, [0, 0])

@pytest.mark.soc("950")
def test_lanemask_ge():
    torch.npu.set_device(ST_DEVICE)
    lane = torch.arange(WARP_SIZE, dtype=torch.int64)
    expected = (((1 << WARP_SIZE) - 1) ^ ((1 << lane) - 1)).to(torch.int32).reshape(1, WARP_SIZE)
    actual = torch.empty_like(expected).to(ST_DEVICE)
    simt_lanemask_ge(actual)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

# -----------------------------------------------------------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def write_lanemask_gt(output):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.lanemask_gt()

@pl.jit(arch=NpuArch.DAV_3510, auto_mutex=True)
def simt_lanemask_gt(output: pl.Tensor[[1, WARP_SIZE], pl.DT_INT32]):
    output_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, WARP_SIZE], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    output_tile = output_tile_group.current()
    with pl.section_vector():
        write_lanemask_gt[WARP_SIZE](output_tile)
        pl.store(output, output_tile, [0, 0])

@pytest.mark.soc("950")
def test_lanemask_gt():
    torch.npu.set_device(ST_DEVICE)
    lane = torch.arange(WARP_SIZE, dtype=torch.int64)
    expected = (((1 << WARP_SIZE) - 1) ^ ((1 << (lane + 1)) - 1)).to(torch.int32).reshape(1, WARP_SIZE)
    actual = torch.empty_like(expected).to(ST_DEVICE)
    simt_lanemask_gt(actual)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

# -----------------------------------------------------------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def write_lanemask_le(output):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.lanemask_le()

@pl.jit(arch=NpuArch.DAV_3510, auto_mutex=True)
def simt_lanemask_le(output: pl.Tensor[[1, WARP_SIZE], pl.DT_INT32]):
    output_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, WARP_SIZE], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    output_tile = output_tile_group.current()
    with pl.section_vector():
        write_lanemask_le[WARP_SIZE](output_tile)
        pl.store(output, output_tile, [0, 0])

@pytest.mark.soc("950")
def test_lanemask_le():
    torch.npu.set_device(ST_DEVICE)
    lane = torch.arange(WARP_SIZE, dtype=torch.int64)
    expected = ((1 << (lane + 1)) - 1).to(torch.int32).reshape(1, WARP_SIZE)
    actual = torch.empty_like(expected).to(ST_DEVICE)
    simt_lanemask_le(actual)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

# -----------------------------------------------------------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def write_lanemask_lt(output):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.lanemask_lt()

@pl.jit(arch=NpuArch.DAV_3510, auto_mutex=True)
def simt_lanemask_lt(output: pl.Tensor[[1, WARP_SIZE], pl.DT_INT32]):
    output_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, WARP_SIZE], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    output_tile = output_tile_group.current()
    with pl.section_vector():
        write_lanemask_lt[WARP_SIZE](output_tile)
        pl.store(output, output_tile, [0, 0])

@pytest.mark.soc("950")
def test_lanemask_lt():
    torch.npu.set_device(ST_DEVICE)
    lane = torch.arange(WARP_SIZE, dtype=torch.int64)
    expected = ((1 << lane) - 1).to(torch.int32).reshape(1, WARP_SIZE)
    actual = torch.empty_like(expected).to(ST_DEVICE)
    simt_lanemask_lt(actual)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
