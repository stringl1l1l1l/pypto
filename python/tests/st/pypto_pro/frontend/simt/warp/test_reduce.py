# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 end-to-end tests for the SIMT Warp Reduce interfaces."""

import os

import pypto_pro.language as pl
from pypto_pro.runtime.platform import NpuArch
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
WARP_SIZE = 32

@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def write_warp_reduce_add(
    values,
    output,
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.warp_reduce_add(values[0, tid])

@pl.jit(arch=NpuArch.DAV_3510, auto_mutex=True)
def simt_warp_reduce_add(
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
        write_warp_reduce_add[WARP_SIZE](values_tile, output_tile)
        pl.store(output, output_tile, [0, 0])

@pytest.mark.soc("950")
def test_warp_reduce_add():
    torch.npu.set_device(ST_DEVICE)
    values = (torch.roll(torch.arange(WARP_SIZE, dtype=torch.float32), shifts=7) - 14).reshape(1, WARP_SIZE)
    expected = torch.full((1, WARP_SIZE), values.sum().item(), dtype=torch.float32)
    actual = torch.empty_like(expected).to(ST_DEVICE)
    simt_warp_reduce_add(values.to(ST_DEVICE), actual)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

# -----------------------------------------------------------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def write_warp_reduce_max(
    values,
    output,
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.warp_reduce_max(values[0, tid])

@pl.jit(arch=NpuArch.DAV_3510, auto_mutex=True)
def simt_warp_reduce_max(
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
        write_warp_reduce_max[WARP_SIZE](values_tile, output_tile)
        pl.store(output, output_tile, [0, 0])

@pytest.mark.soc("950")
def test_warp_reduce_max():
    torch.npu.set_device(ST_DEVICE)
    values = (torch.roll(torch.arange(WARP_SIZE, dtype=torch.float32), shifts=7) - 16).reshape(1, WARP_SIZE)
    expected = torch.full((1, WARP_SIZE), values.max().item(), dtype=torch.float32)
    actual = torch.empty_like(expected).to(ST_DEVICE)
    simt_warp_reduce_max(values.to(ST_DEVICE), actual)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

# -----------------------------------------------------------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def write_warp_reduce_min(
    values,
    output,
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.warp_reduce_min(values[0, tid])

@pl.jit(arch=NpuArch.DAV_3510, auto_mutex=True)
def simt_warp_reduce_min(
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
        write_warp_reduce_min[WARP_SIZE](values_tile, output_tile)
        pl.store(output, output_tile, [0, 0])

@pytest.mark.soc("950")
def test_warp_reduce_min():
    torch.npu.set_device(ST_DEVICE)
    values = (torch.roll(torch.arange(WARP_SIZE, dtype=torch.float32), shifts=7) - 16).reshape(1, WARP_SIZE)
    expected = torch.full((1, WARP_SIZE), values.min().item(), dtype=torch.float32)
    actual = torch.empty_like(expected).to(ST_DEVICE)
    simt_warp_reduce_min(values.to(ST_DEVICE), actual)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

# -----------------------------------------------------------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def write_warp_reduce_fp16_divergent(
    values,
    output,
):
    tid = pl.simt.linear_thread_idx()
    lane = pl.simt.lane_id()
    if lane < WARP_SIZE // 2:
        output[0, tid] = pl.simt.warp_reduce_add(values[0, tid])
        output[1, tid] = pl.simt.warp_reduce_max(values[0, tid])
        output[2, tid] = pl.simt.warp_reduce_min(values[0, tid])
    else:
        value = values[0, tid]
        doubled = value + value
        output[0, tid] = pl.simt.warp_reduce_add(doubled)
        output[1, tid] = pl.simt.warp_reduce_max(doubled)
        output[2, tid] = pl.simt.warp_reduce_min(doubled)

@pl.jit(arch=NpuArch.DAV_3510, auto_mutex=True)
def simt_warp_reduce_fp16_divergent(
    values: pl.Tensor[[1, WARP_SIZE], pl.DT_FP16],
    output: pl.Tensor[[3, WARP_SIZE], pl.DT_FP16],
):
    values_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, WARP_SIZE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    values_tile = values_tile_group.current()
    output_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[3, WARP_SIZE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    output_tile = output_tile_group.current()
    with pl.section_vector():
        pl.load(values_tile, values, [0, 0])
        write_warp_reduce_fp16_divergent[WARP_SIZE](values_tile, output_tile)
        pl.store(output, output_tile, [0, 0])

@pytest.mark.soc("950")
def test_warp_reduce_fp16_divergent():
    torch.npu.set_device(ST_DEVICE)
    # Keep zero as both groups' minimum while doubling the upper group to distinguish its add and max results.
    half_warp_values = torch.arange(WARP_SIZE // 2, dtype=torch.float16)
    values = half_warp_values.repeat(2).reshape(1, WARP_SIZE)
    expected = torch.empty((3, WARP_SIZE), dtype=torch.float16)
    for start in (0, WARP_SIZE // 2):
        group = values[0, start:start + WARP_SIZE // 2]
        if start:
            group = group * 2
        expected[0, start:start + WARP_SIZE // 2] = group.sum()
        expected[1, start:start + WARP_SIZE // 2] = group.max()
        expected[2, start:start + WARP_SIZE // 2] = group.min()
    actual = torch.empty_like(expected).to(ST_DEVICE)
    simt_warp_reduce_fp16_divergent(values.to(ST_DEVICE), actual)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
