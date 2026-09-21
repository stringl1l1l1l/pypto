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

"""Static fused matmul + add test with manual synchronization."""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto


@pl.jit()
def matmul_add_matmul_add_cce(
    q: pl.Tensor[[64, 64], pl.DT_FP32],
    k: pl.Tensor[[64, 64], pl.DT_FP32],
    v: pl.Tensor[[64, 64], pl.DT_FP32],
    x1: pl.Tensor[[64, 64], pl.DT_FP32],
    x2: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
    workspace: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tile_p_vec = pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    mm1_res = pl.make_tile(tile_p_vec, addr=0x0000)

    tile_p_vec = pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    mm2_res = pl.make_tile(tile_p_vec, addr=0x2000)

    tile_v1_mat = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
    v1_mat = pl.make_tile(tile_v1_mat, addr=0x10000)

    with pl.section_cube():
        tile_q_mat = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(tile_q_mat, addr=0x0000)

        tile_k_mat = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        k_mat = pl.make_tile(tile_k_mat, addr=0x4000)
        v_mat = pl.make_tile(tile_k_mat, addr=0x8000)

        tile_q_left = pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Left,
            layout=pl.NZ,
        )
        q_left = pl.make_tile(tile_q_left, addr=0x0000)

        tile_k_right = pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Right,
            layout=pl.ZN,
        )
        k_right = pl.make_tile(tile_k_right, addr=0x0000)
        v_right = pl.make_tile(tile_k_right, addr=0x4000)  # kv addr not conflict, no need sync

        tile_c1_type = pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            fractal=1024,
        )
        tile_c1 = pl.make_tile(tile_c1_type, addr=0x0000)
        tile_c2 = pl.make_tile(tile_c1_type, addr=0x8000)

        pl.load(q_mat, q, [0, 0])
        pl.load(k_mat, k, [0, 0])

        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        pl.move(q_left, q_mat)
        pl.move(k_right, k_mat)

        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

        pl.matmul(tile_c1, q_left, k_right)

        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.store(workspace, tile_c1, [0, 0], relu_pre_mode=pl.ReluPreMode.NormalRelu)  # L0C2GM
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

        pl.load(v_mat, v, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.move(v_right, v_mat)

        pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=2)
        pl.move(q_left, v1_mat)

        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.matmul(tile_c2, q_left, v_right)

        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.move(mm2_res, tile_c2, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)  # ACC -> UB
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=1)

    with pl.section_vector():
        sub_index = pl.get_subblock_idx()
        tile_type_x1 = pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        tile_x1 = pl.make_tile(tile_type_x1, addr=0x4000)
        tile_x2 = pl.make_tile(tile_type_x1, addr=0x6000)

        tile_type_out = pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        tile_out = pl.make_tile(tile_type_out, addr=0x8000)

        tile_type_nz = pl.TileType(
            shape=[32, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec,
            layout=pl.NZ,
        )
        tile_nz = pl.make_tile(tile_type_nz, addr=0xA000)  # NZ no bank conflict

        off = sub_index * 32
        pl.load(tile_x1, x1, [off, 0])

        pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=0)
        pl.load(mm1_res, workspace, [off, 0])

        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=1)

        pl.add(tile_out, mm1_res, tile_x1)
        pl.move(tile_nz, tile_out)  # ND2NZ

        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.insert(v1_mat, tile_nz, [off, 0])  # UB2L1 NZ2NZ

        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=2)

        pl.load(tile_x2, x2, [off, 0])

        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=1)
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=1)

        pl.add(tile_out, mm2_res, tile_x2)

        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)

        pl.store(out, tile_out, [off, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_matmul_add_matmul_add():

    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Currrent device is not Ascend950, skip.")
        return
    shape = [64, 64]
    torch.manual_seed(0)
    dtype = torch.float32

    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)
    x1 = torch.randn(shape, device=device, dtype=dtype)
    x2 = torch.randn(shape, device=device, dtype=dtype)
    out = torch.zeros(shape, device=device, dtype=dtype)
    workspace = torch.zeros(shape, device=device, dtype=dtype)  # bmm1

    matmul_add_matmul_add_cce[None, 1](q, k, v, x1, x2, out, workspace)
    torch.npu.synchronize()

    c1 = torch.matmul(q, k)
    c1 = torch.relu(c1)
    v1 = c1 + x1
    c2 = torch.matmul(v1, v)
    out_ref = c2 + x2

    logging.info("***********npu output***********")
    logging.info("%s %s", out.shape, out.dtype)
    logging.info("%s", out)
    logging.info("***********golden output***********")
    logging.info("%s %s", out_ref.shape, out_ref.dtype)
    logging.info("%s", out_ref)

    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)
    logging.info("result equal!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_matmul_add_matmul_add()
    logging.info("\nAll tests passed!")
