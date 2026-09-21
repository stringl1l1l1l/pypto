# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tile-tile comparison and fused scalar multiply-add (axpy).

Doc: docs/zh/api/pro_api/SIMD-API/计算API/Memory矢量计算/{比较/comparison, 复合计算/axpy}

Pure vector mode. Verifies gt producing a bit-packed mask for select, and
in-place axpy (out = scalar * src + out).
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

ALPHA = 2.0


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


# ===========================================================================
# gt + select —— tile-tile 比较生成 bit-packed mask，再用 select 选择
#   gt 输出是 bit-packed UINT8 mask，不能直接 store 为 FP32。
#   配合 select 验证：a > b 处选 a，否则选 b → out = max(a, b) 的近似。
# ===========================================================================
@pl.jit(auto_mutex=True)
def gt_select_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32],
    b: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a_group = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b_group = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out_group = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    tmp_vec_group = pl.make_tile_group(type=tt, addrs=0xC000, mutex_ids=[3])
    mask_vec_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec),
        addrs=0x10000,
        mutex_ids=[4],
    )
    with pl.section_vector():
        tile_a = tile_a_group.current()
        tile_b = tile_b_group.current()
        tile_out = tile_out_group.current()
        tmp_vec = tmp_vec_group.current()
        mask_vec = mask_vec_group.current()
        pl.load(tile_a, a, [0, 0])
        pl.load(tile_b, b, [0, 0])
        pl.gt(mask_vec, tile_a, tile_b)
        pl.select(tile_out, mask_vec, tile_a, tile_b, tmp_vec)
        pl.store(out, tile_out, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_gt_select():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn(64, 64, device=device, dtype=torch.float32)
    b = torch.randn(64, 64, device=device, dtype=torch.float32)
    out = torch.zeros(64, 64, device=device, dtype=torch.float32)
    gt_select_kernel(a, b, out)
    torch.npu.synchronize()
    out_ref = torch.where(a > b, a, b)
    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)
    logging.info("gt+select result equal!")


# ===========================================================================
# axpy —— out = ALPHA * src + out（in-place 融合标量乘加）
# ===========================================================================
@pl.jit(auto_mutex=True)
def axpy_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP32],
    y: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_x_group = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_y_group = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        tile_x = tile_x_group.current()
        tile_y = tile_y_group.current()
        pl.load(tile_x, x, [0, 0])
        pl.load(tile_y, y, [0, 0])
        pl.axpy(tile_y, tile_x, ALPHA)
        pl.store(y, tile_y, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_axpy():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    x = torch.randn(64, 64, device=device, dtype=torch.float32)
    y = torch.randn(64, 64, device=device, dtype=torch.float32)
    y_orig = y.clone()
    axpy_kernel[None, 1](x, y)
    torch.npu.synchronize()
    y_ref = ALPHA * x + y_orig
    torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=1e-2)
    logging.info("axpy result equal!")
