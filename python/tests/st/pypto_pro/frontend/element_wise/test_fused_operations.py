# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""UT for doc examples — fused ops.

make_tile_group + auto_mutex style.

Verifies kernel examples from:
  docs/zh/api/pro_api/SIMD-API/memory_vector_computation/fused_vector_computation/mul_add_dst.md
  docs/zh/api/pro_api/SIMD-API/memory_vector_computation/fused_vector_computation/fused_mul_add_relu.md
  docs/zh/api/pro_api/SIMD-API/memory_vector_computation/fused_vector_computation/partadd.md
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

M = 64
N = 128


# ============================================================================
# mul_add_dst
# ============================================================================


@pl.jit(auto_mutex=True)
def mul_add_dst_kernel(
    x: pl.Tensor[[M, N], pl.DT_FP16],
    y: pl.Tensor[[M, N], pl.DT_FP16],
    z: pl.Tensor[[M, N], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[1])
    tile_c = pl.make_tile_group(type=tt, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_c = tile_c.current()
        pl.load(cur_a, x, [0, 0])
        pl.load(cur_b, y, [0, 0])
        pl.load(cur_c, z, [0, 0])
        pl.mul_add_dst(cur_c, cur_a, cur_b)
        pl.store(z, cur_c, [0, 0])


# ============================================================================
# fused_mul_add_relu
# ============================================================================


@pl.jit(auto_mutex=True)
def fused_mul_add_relu_kernel(
    x: pl.Tensor[[M, N], pl.DT_FP16],
    y: pl.Tensor[[M, N], pl.DT_FP16],
    z: pl.Tensor[[M, N], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[1])
    tile_c = pl.make_tile_group(type=tt, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_c = tile_c.current()
        pl.load(cur_a, x, [0, 0])
        pl.load(cur_b, y, [0, 0])
        pl.load(cur_c, z, [0, 0])
        pl.fused_mul_add_relu(cur_c, cur_a, cur_b)
        pl.store(z, cur_c, [0, 0])


# ============================================================================
# partadd
# ============================================================================


@pl.jit(auto_mutex=True)
def partadd_kernel(
    x: pl.Tensor[[M, N], pl.DT_FP16],
    y: pl.Tensor[[M, N], pl.DT_FP16],
    z: pl.Tensor[[M, N], pl.DT_FP16],
):
    tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    tile_a = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[1])
    tile_c = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_c = tile_c.current()
        pl.set_validshape(cur_a, [64, 128])
        pl.load(cur_a, x, [0, 0])
        pl.set_validshape(cur_b, [32, 128])
        pl.load(cur_b, y, [0, 0])
        pl.set_validshape(cur_c, [64, 128])
        pl.partadd(cur_c, cur_a, cur_b)
        pl.store(z, cur_c, [0, 0])


# ============================================================================
# Test functions
# ============================================================================


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_mul_add_dst():
    torch.npu.set_device(ST_DEVICE)
    x = torch.randn(M, N, device=ST_DEVICE, dtype=torch.float16)
    y = torch.randn(M, N, device=ST_DEVICE, dtype=torch.float16)
    z = torch.randn(M, N, device=ST_DEVICE, dtype=torch.float16)
    z_orig = z.clone()
    z_ref = z_orig + x * y
    mul_add_dst_kernel[None, 1](x, y, z)
    torch.npu.synchronize()
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("test_mul_add_dst passed!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_fused_mul_add_relu():
    torch.npu.set_device(ST_DEVICE)
    x = torch.randn(M, N, device=ST_DEVICE, dtype=torch.float16)
    y = torch.randn(M, N, device=ST_DEVICE, dtype=torch.float16)
    z = torch.randn(M, N, device=ST_DEVICE, dtype=torch.float16)
    z_orig = z.clone()
    z_ref = torch.relu(z_orig * x + y)
    fused_mul_add_relu_kernel[None, 1](x, y, z)
    torch.npu.synchronize()
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("test_fused_mul_add_relu passed!")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_partadd():
    torch.npu.set_device(ST_DEVICE)
    x = torch.randn(M, N, device=ST_DEVICE, dtype=torch.float16)
    y = torch.randn(M, N, device=ST_DEVICE, dtype=torch.float16)
    z = torch.empty(M, N, device=ST_DEVICE, dtype=torch.float16)
    z_ref = x.clone()
    z_ref[:32, :] = x[:32, :] + y[:32, :]
    partadd_kernel(x, y, z)
    torch.npu.synchronize()
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("test_partadd passed!")
