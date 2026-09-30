# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Limit a synchronized mixed matmul; independent scalar probes cannot detect missing CV participants."""

import pypto_pro.language as pl
import pytest
from test_core_limits import DEVICE_ID
import torch

CV_TILE_M = 64
CV_TILE_K = 64
CV_TILE_N = 64
CV_VEC_M = CV_TILE_M // 2
CV_NUM_TILES = 13
CV_M = CV_TILE_M * CV_NUM_TILES


@pl.jit(arch="3510", auto_mutex=True)
def limited_matmul(
    a: pl.Tensor[[CV_M, CV_TILE_K], pl.DT_FP16],
    b: pl.Tensor[[CV_TILE_K, CV_TILE_N], pl.DT_FP16],
    out: pl.Tensor[[CV_M, CV_TILE_N], pl.DT_FP32],
):
    """Distribute all tiles using the limited runtime block count on both synchronized engines."""
    a_l1_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[CV_TILE_M, CV_TILE_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x0000,
        mutex_ids=[0],
    )
    b_l1_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[CV_TILE_K, CV_TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x4000,
        mutex_ids=[1],
    )
    a_l0a_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[CV_TILE_M, CV_TILE_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[CV_TILE_K, CV_TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    acc_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[CV_TILE_M, CV_TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[8],
    )
    vec_tile = pl.make_tile(
        pl.TileType(
            shape=[CV_VEC_M, CV_TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x0000,
    )

    with pl.section_cube():
        core_id = pl.get_block_idx()
        for tile_index in pl.range(core_id, CV_NUM_TILES, pl.get_block_num()):
            row = tile_index * CV_TILE_M
            a_l1 = a_l1_group.current()
            b_l1 = b_l1_group.current()
            a_l0a = a_l0a_group.current()
            b_l0b = b_l0b_group.current()
            acc = acc_group.current()

            pl.load(a_l1, a, [row, 0])
            pl.load(b_l1, b, [0, 0])
            pl.move(a_l0a, a_l1)
            pl.move(b_l0b, b_l1)
            pl.matmul(acc, a_l0a, b_l0b)

            # The Vector halves acknowledge the previous use of the ordinary Vec tile.
            pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=6)
            pl.move(vec_tile, acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    with pl.section_vector():
        subblock_id = pl.get_subblock_idx()
        vector_core_id = pl.get_block_idx() // pl.get_subblock_num()
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=6)

        for tile_index in pl.range(vector_core_id, CV_NUM_TILES, pl.get_block_num()):
            row = tile_index * CV_TILE_M + subblock_id * CV_VEC_M
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
            pl.store(out, vec_tile, [row, 0])
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=6)


@pytest.mark.soc("950")
@pytest.mark.parametrize("limits", [(4, 16), (8, 5)])
@pytest.mark.parametrize("capture", [False, True], ids=["eager", "graph"])
def test_limited_matmul(limits, capture):
    """Uneven work and two vector consumers per cube exercise the limited group's cross-core waits."""
    torch.npu.set_device(DEVICE_ID)
    a_host = (torch.arange(CV_M * CV_TILE_K).reshape(CV_M, CV_TILE_K) % 17).to(torch.float16)
    b_host = (torch.arange(CV_TILE_K * CV_TILE_N).reshape(CV_TILE_K, CV_TILE_N) % 13).to(torch.float16)
    golden = a_host.float() @ b_host.float()
    a = a_host.to(f"npu:{DEVICE_ID}")
    b = b_host.to(f"npu:{DEVICE_ID}")
    out = torch.zeros((CV_M, CV_TILE_N), dtype=torch.float32, device=a.device)
    s = torch.npu.Stream()
    torch.npu.synchronize()
    # A direct call carries the auto sentinel: the launch takes the scope's full
    # block budget, the same counts the previous clamped request produced.
    with torch.npu.npugraph_ex.scope.limit_core_num(*limits, stream=s), torch.npu.stream(s):
        limited_matmul(a, b, out)
        s.synchronize()
        if capture:
            out.zero_()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=s):
                limited_matmul(a, b, out)
    if capture:
        graph.replay()
    torch.npu.synchronize()
    actual = out.cpu()
    torch.testing.assert_close(actual, golden, rtol=0, atol=0)
    print(f"mixed matmul limits={limits} capture={capture}: max_abs_error={(actual - golden).abs().max().item()}, "
          "exact_match=100%")
