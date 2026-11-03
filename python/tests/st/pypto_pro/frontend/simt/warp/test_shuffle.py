# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 end-to-end tests for the SIMT Warp Shuffle interfaces."""

import os

import pypto_pro.language as pl
from pypto_pro.runtime.platform import NpuArch
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
WARP_SIZE = 32
TWO_WARPS = 2 * WARP_SIZE
SUBGROUP_WIDTH = 16
SOURCE_LANE = 3
DELTA = 1
LANE_MASK = 1

@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def write_warp_shfl(
    values,
    output,
):
    tid = pl.simt.linear_thread_idx()
    warp_width = pl.simt.warp_size()
    subgroup_width = pl.simt.cast(warp_width // 2, pl.DT_INT32)
    output[0, tid] = pl.simt.warp_shfl(values[0, tid], SOURCE_LANE, subgroup_width)

@pl.jit(arch=NpuArch.DAV_3510, auto_mutex=True)
def simt_warp_shfl(
    values: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
    output: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
):
    values_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, WARP_SIZE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    values_tile = values_tile_group.current()
    output_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, WARP_SIZE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    output_tile = output_tile_group.current()
    with pl.section_vector():
        pl.load(values_tile, values, [0, 0])
        write_warp_shfl[WARP_SIZE](values_tile, output_tile)
        pl.store(output, output_tile, [0, 0])

@pytest.mark.soc("950")
def test_warp_shfl_logical_subgroups():
    torch.npu.set_device(ST_DEVICE)
    values = torch.arange(WARP_SIZE, dtype=torch.float32).reshape(1, WARP_SIZE)
    lane = torch.arange(WARP_SIZE, dtype=torch.int64)
    expected = ((lane // SUBGROUP_WIDTH) * SUBGROUP_WIDTH + SOURCE_LANE).to(torch.float32).reshape(1, WARP_SIZE)
    actual = torch.empty_like(expected).to(ST_DEVICE)
    simt_warp_shfl(values.to(ST_DEVICE), actual)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

# -----------------------------------------------------------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def write_warp_shfl_down(
    values,
    output,
):
    tid = pl.simt.linear_thread_idx()
    warp_width = pl.simt.warp_size()
    subgroup_width = pl.simt.cast(warp_width // 2, pl.DT_INT32)
    output[0, tid] = pl.simt.warp_shfl_down(values[0, tid], DELTA, subgroup_width)

@pl.jit(arch=NpuArch.DAV_3510, auto_mutex=True)
def simt_warp_shfl_down(
    values: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
    output: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
):
    values_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, WARP_SIZE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    values_tile = values_tile_group.current()
    output_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, WARP_SIZE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    output_tile = output_tile_group.current()
    with pl.section_vector():
        pl.load(values_tile, values, [0, 0])
        write_warp_shfl_down[WARP_SIZE](values_tile, output_tile)
        pl.store(output, output_tile, [0, 0])

@pytest.mark.soc("950")
def test_warp_shfl_down_logical_subgroup_boundary():
    torch.npu.set_device(ST_DEVICE)
    values = torch.arange(WARP_SIZE, dtype=torch.float32).reshape(1, WARP_SIZE)
    lane = torch.arange(WARP_SIZE, dtype=torch.int64)
    subgroup_lane = lane % SUBGROUP_WIDTH
    expected = (
        torch.where(
            subgroup_lane >= SUBGROUP_WIDTH - DELTA,
            lane,
            lane + DELTA,
        )
        .to(torch.float32)
        .reshape(1, WARP_SIZE)
    )
    actual = torch.empty_like(expected).to(ST_DEVICE)
    simt_warp_shfl_down(values.to(ST_DEVICE), actual)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

# -----------------------------------------------------------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def write_warp_shfl_up(
    values,
    output,
):
    tid = pl.simt.linear_thread_idx()
    warp_width = pl.simt.warp_size()
    subgroup_width = pl.simt.cast(warp_width // 2, pl.DT_INT32)
    output[0, tid] = pl.simt.warp_shfl_up(values[0, tid], DELTA, subgroup_width)

@pl.jit(arch=NpuArch.DAV_3510, auto_mutex=True)
def simt_warp_shfl_up(
    values: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
    output: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
):
    values_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, WARP_SIZE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    values_tile = values_tile_group.current()
    output_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, WARP_SIZE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    output_tile = output_tile_group.current()
    with pl.section_vector():
        pl.load(values_tile, values, [0, 0])
        write_warp_shfl_up[WARP_SIZE](values_tile, output_tile)
        pl.store(output, output_tile, [0, 0])

@pytest.mark.soc("950")
def test_warp_shfl_up_logical_subgroup_boundary():
    torch.npu.set_device(ST_DEVICE)
    values = torch.arange(WARP_SIZE, dtype=torch.float32).reshape(1, WARP_SIZE)
    lane = torch.arange(WARP_SIZE, dtype=torch.int64)
    subgroup_lane = lane % SUBGROUP_WIDTH
    expected = torch.where(subgroup_lane < DELTA, lane, lane - DELTA).to(torch.float32).reshape(1, WARP_SIZE)
    actual = torch.empty_like(expected).to(ST_DEVICE)
    simt_warp_shfl_up(values.to(ST_DEVICE), actual)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

# -----------------------------------------------------------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def write_warp_shfl_xor(
    values,
    output,
):
    tid = pl.simt.linear_thread_idx()
    warp_width = pl.simt.warp_size()
    subgroup_width = pl.simt.cast(warp_width // 2, pl.DT_INT32)
    output[0, tid] = pl.simt.warp_shfl_xor(values[0, tid], LANE_MASK, subgroup_width)

@pl.jit(arch=NpuArch.DAV_3510, auto_mutex=True)
def simt_warp_shfl_xor(
    values: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
    output: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
):
    values_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, WARP_SIZE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    values_tile = values_tile_group.current()
    output_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, WARP_SIZE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    output_tile = output_tile_group.current()
    with pl.section_vector():
        pl.load(values_tile, values, [0, 0])
        write_warp_shfl_xor[WARP_SIZE](values_tile, output_tile)
        pl.store(output, output_tile, [0, 0])

@pytest.mark.soc("950")
def test_warp_shfl_xor_logical_subgroups():
    torch.npu.set_device(ST_DEVICE)
    values = torch.arange(WARP_SIZE, dtype=torch.float32).reshape(1, WARP_SIZE)
    lane = torch.arange(WARP_SIZE, dtype=torch.int64)
    expected = (lane ^ LANE_MASK).to(torch.float32).reshape(1, WARP_SIZE)
    actual = torch.empty_like(expected).to(ST_DEVICE)
    simt_warp_shfl_xor(values.to(ST_DEVICE), actual)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

# -----------------------------------------------------------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def write_warp_shfl_bf16(
    values,
    output,
):
    tid = pl.simt.linear_thread_idx()
    src_lane = pl.simt.cast((pl.simt.lane_id() + SOURCE_LANE) % SUBGROUP_WIDTH, pl.DT_INT32)
    output[0, tid] = pl.simt.warp_shfl(values[0, tid], src_lane, SUBGROUP_WIDTH)

@pl.jit(arch=NpuArch.DAV_3510, auto_mutex=True)
def simt_warp_shfl_bf16(
    values: pl.Tensor[[1, WARP_SIZE], pl.DT_BF16],
    output: pl.Tensor[[1, WARP_SIZE], pl.DT_BF16],
):
    values_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, WARP_SIZE], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    values_tile = values_tile_group.current()
    output_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, WARP_SIZE], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    output_tile = output_tile_group.current()
    with pl.section_vector():
        pl.load(values_tile, values, [0, 0])
        write_warp_shfl_bf16[WARP_SIZE](values_tile, output_tile)
        pl.store(output, output_tile, [0, 0])

@pytest.mark.soc("950")
def test_warp_shfl_bf16_runtime_src_lane():
    torch.npu.set_device(ST_DEVICE)
    lane = torch.arange(WARP_SIZE, dtype=torch.int64)
    values = lane.to(torch.bfloat16).reshape(1, WARP_SIZE)
    expected_lane = (lane // SUBGROUP_WIDTH) * SUBGROUP_WIDTH + (lane + SOURCE_LANE) % SUBGROUP_WIDTH
    expected = expected_lane.to(torch.bfloat16).reshape(1, WARP_SIZE)
    actual = torch.empty_like(expected).to(ST_DEVICE)
    simt_warp_shfl_bf16(values.to(ST_DEVICE), actual)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

# -----------------------------------------------------------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def write_warp_shfl_xor_fp16(
    values,
    output,
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.warp_shfl_xor(values[0, tid], LANE_MASK, SUBGROUP_WIDTH)

@pl.jit(arch=NpuArch.DAV_3510, auto_mutex=True)
def simt_warp_shfl_xor_fp16(
    values: pl.Tensor[[1, WARP_SIZE], pl.DT_FP16],
    output: pl.Tensor[[1, WARP_SIZE], pl.DT_FP16],
):
    values_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, WARP_SIZE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    values_tile = values_tile_group.current()
    output_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, WARP_SIZE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    output_tile = output_tile_group.current()
    with pl.section_vector():
        pl.load(values_tile, values, [0, 0])
        write_warp_shfl_xor_fp16[WARP_SIZE](values_tile, output_tile)
        pl.store(output, output_tile, [0, 0])

@pytest.mark.soc("950")
def test_warp_shfl_xor_fp16():
    torch.npu.set_device(ST_DEVICE)
    lane = torch.arange(WARP_SIZE, dtype=torch.int64)
    values = lane.to(torch.float16).reshape(1, WARP_SIZE)
    expected = (lane ^ LANE_MASK).to(torch.float16).reshape(1, WARP_SIZE)
    actual = torch.empty_like(expected).to(ST_DEVICE)
    simt_warp_shfl_xor_fp16(values.to(ST_DEVICE), actual)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

# -----------------------------------------------------------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=TWO_WARPS)
def write_warp_shfl_down_int64(
    values,
    output,
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.warp_shfl_down(values[0, tid], DELTA)

@pl.jit(arch=NpuArch.DAV_3510, auto_mutex=True)
def simt_warp_shfl_down_int64(
    values: pl.Tensor[[1, TWO_WARPS], pl.DT_INT64],
    output: pl.Tensor[[1, TWO_WARPS], pl.DT_INT64],
):
    values_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, TWO_WARPS], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    values_tile = values_tile_group.current()
    output_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, TWO_WARPS], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    output_tile = output_tile_group.current()
    with pl.section_vector():
        pl.load(values_tile, values, [0, 0])
        write_warp_shfl_down_int64[TWO_WARPS](values_tile, output_tile)
        pl.store(output, output_tile, [0, 0])

@pl.vector_function(mode="simt", max_threads=TWO_WARPS)
def write_warp_shfl_down_uint64(
    values,
    output,
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.warp_shfl_down(values[0, tid], DELTA)

@pl.jit(arch=NpuArch.DAV_3510, auto_mutex=True)
def simt_warp_shfl_down_uint64(
    values: pl.Tensor[[1, TWO_WARPS], pl.DT_UINT64],
    output: pl.Tensor[[1, TWO_WARPS], pl.DT_UINT64],
):
    values_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, TWO_WARPS], dtype=pl.DT_UINT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    values_tile = values_tile_group.current()
    output_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, TWO_WARPS], dtype=pl.DT_UINT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    output_tile = output_tile_group.current()
    with pl.section_vector():
        pl.load(values_tile, values, [0, 0])
        write_warp_shfl_down_uint64[TWO_WARPS](values_tile, output_tile)
        pl.store(output, output_tile, [0, 0])

@pytest.mark.soc("950")
@pytest.mark.parametrize(
    "kernel, dtype",
    [(simt_warp_shfl_down_int64, torch.int64), (simt_warp_shfl_down_uint64, torch.uint64)],
)
def test_warp_shfl_down_64bit_two_warps(kernel, dtype):
    torch.npu.set_device(ST_DEVICE)
    lane = torch.arange(TWO_WARPS, dtype=torch.int64)
    values = (((lane + 1) << 40) + lane).to(dtype).reshape(1, TWO_WARPS)
    source = torch.where(lane % WARP_SIZE == WARP_SIZE - DELTA, lane, lane + DELTA)
    expected = (((source + 1) << 40) + source).to(dtype).reshape(1, TWO_WARPS)
    actual = torch.empty_like(expected).to(ST_DEVICE)
    kernel(values.to(ST_DEVICE), actual)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu().to(torch.int64), expected.to(torch.int64), rtol=0, atol=0)
