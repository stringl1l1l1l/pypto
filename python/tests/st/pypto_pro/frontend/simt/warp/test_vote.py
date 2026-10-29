# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 end-to-end tests for the SIMT Warp Vote interfaces."""

import os

import pypto_pro.language as pl
from pypto_pro.runtime.platform import NpuArch
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
WARP_SIZE = 32

@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def write_warp_active_mask(output):
    tid = pl.simt.linear_thread_idx()
    lane = pl.simt.lane_id()
    warp_size = pl.simt.warp_size()
    output[0, tid] = pl.simt.warp_active_mask()
    if lane < warp_size // 2:
        output[1, tid] = pl.simt.warp_active_mask()
        output[2, tid] = 0
    else:
        output[1, tid] = 0
        output[2, tid] = pl.simt.warp_active_mask()

@pl.jit(arch=NpuArch.DAV_3510, auto_mutex=True)
def simt_warp_active_mask(output: pl.Tensor[[3, WARP_SIZE], pl.DT_UINT32]):
    output_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[3, WARP_SIZE], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    output_tile = output_tile_group.current()
    with pl.section_vector():
        write_warp_active_mask[WARP_SIZE](output_tile)
        pl.store(output, output_tile, [0, 0])

@pytest.mark.soc("950")
def test_warp_active_mask():
    torch.npu.set_device(ST_DEVICE)
    expected = torch.stack(
        (
            torch.full((WARP_SIZE,), 0xFFFFFFFF, dtype=torch.uint32),
            torch.cat(
                (
                    torch.full((WARP_SIZE // 2,), 0x0000FFFF, dtype=torch.uint32),
                    torch.zeros(WARP_SIZE // 2, dtype=torch.uint32),
                )
            ),
            torch.cat(
                (
                    torch.zeros(WARP_SIZE // 2, dtype=torch.uint32),
                    torch.full((WARP_SIZE // 2,), 0xFFFF0000, dtype=torch.uint32),
                )
            ),
        )
    )
    actual = torch.empty_like(expected).to(ST_DEVICE)
    simt_warp_active_mask(actual)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu().to(torch.int64), expected.to(torch.int64), rtol=0, atol=0)

# -----------------------------------------------------------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def write_warp_all(output):
    tid = pl.simt.linear_thread_idx()
    lane = pl.simt.lane_id()
    warp_size = pl.simt.warp_size()
    output[0, tid] = pl.simt.warp_all(lane < warp_size)
    output[1, tid] = pl.simt.warp_all(lane < warp_size - 1)

@pl.jit(arch=NpuArch.DAV_3510, auto_mutex=True)
def simt_warp_all(output: pl.Tensor[[2, WARP_SIZE], pl.DT_INT32]):
    output_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[2, WARP_SIZE], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    output_tile = output_tile_group.current()
    with pl.section_vector():
        write_warp_all[WARP_SIZE](output_tile)
        pl.store(output, output_tile, [0, 0])

@pytest.mark.soc("950")
def test_warp_all_true_and_false_predicates():
    torch.npu.set_device(ST_DEVICE)
    expected = torch.stack((torch.ones(WARP_SIZE, dtype=torch.int32), torch.zeros(WARP_SIZE, dtype=torch.int32)))
    actual = torch.empty_like(expected).to(ST_DEVICE)
    simt_warp_all(actual)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

# -----------------------------------------------------------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def write_warp_any(output):
    tid = pl.simt.linear_thread_idx()
    lane = pl.simt.lane_id()
    warp_size = pl.simt.warp_size()
    output[0, tid] = pl.simt.warp_any(lane == warp_size - 1)
    output[1, tid] = pl.simt.warp_any(lane == warp_size)

@pl.jit(arch=NpuArch.DAV_3510, auto_mutex=True)
def simt_warp_any(output: pl.Tensor[[2, WARP_SIZE], pl.DT_INT32]):
    output_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[2, WARP_SIZE], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    output_tile = output_tile_group.current()
    with pl.section_vector():
        write_warp_any[WARP_SIZE](output_tile)
        pl.store(output, output_tile, [0, 0])

@pytest.mark.soc("950")
def test_warp_any_true_and_false_predicates():
    torch.npu.set_device(ST_DEVICE)
    expected = torch.stack((torch.ones(WARP_SIZE, dtype=torch.int32), torch.zeros(WARP_SIZE, dtype=torch.int32)))
    actual = torch.empty_like(expected).to(ST_DEVICE)
    simt_warp_any(actual)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

# -----------------------------------------------------------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def write_warp_ballot(output):
    tid = pl.simt.linear_thread_idx()
    lane = pl.simt.lane_id()
    warp_size = pl.simt.warp_size()
    output[0, tid] = pl.simt.warp_ballot((lane % 2) == 0)
    output[1, tid] = pl.simt.warp_ballot(lane < warp_size // 2)

@pl.jit(arch=NpuArch.DAV_3510, auto_mutex=True)
def simt_warp_ballot(output: pl.Tensor[[2, WARP_SIZE], pl.DT_UINT32]):
    output_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[2, WARP_SIZE], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    output_tile = output_tile_group.current()
    with pl.section_vector():
        write_warp_ballot[WARP_SIZE](output_tile)
        pl.store(output, output_tile, [0, 0])

@pytest.mark.soc("950")
def test_warp_ballot_predicate_masks():
    torch.npu.set_device(ST_DEVICE)
    expected = torch.stack(
        (
            torch.full((WARP_SIZE,), 0x55555555, dtype=torch.uint32),
            torch.full((WARP_SIZE,), 0x0000FFFF, dtype=torch.uint32),
        )
    )
    actual = torch.empty_like(expected).to(ST_DEVICE)
    simt_warp_ballot(actual)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu().to(torch.int64), expected.to(torch.int64), rtol=0, atol=0)
