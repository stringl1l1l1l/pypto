# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Vector operations with independent outputs sharing a small set of JIT kernels.

FP32 operations share their inputs and write separate 64-row output regions.
The positive input preserves the domain of sqrt/rsqrt/recip; integer XOR,
in-place fused multiply-add and FP16-to-FP32 casts keep their dtype contracts.
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
SCALE = 0.125
SUB_VALUE = 1.0
DIVISOR = 4.0
ADD_VALUE = 1.0
MAX_SCALAR = 0.0
MIN_SCALAR = 1.0


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


@pl.jit(auto_mutex=True)
def vector_fp32_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32],
    b: pl.Tensor[[64, 64], pl.DT_FP32],
    positive: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[1280, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    tile_positive = pl.make_tile_group(type=tt, addrs=0xC000, mutex_ids=[3])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        cur_positive = tile_positive.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.load(cur_positive, positive, [0, 0])
        pl.add(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
        pl.sub(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [64, 0])
        pl.mul(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [128, 0])
        pl.maximum(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [192, 0])
        pl.div(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [256, 0])
        pl.minimum(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [320, 0])
        pl.minimum(cur_out, cur_a, MIN_SCALAR)
        pl.store(out, cur_out, [384, 0])
        pl.mul(cur_out, cur_a, SCALE)
        pl.store(out, cur_out, [448, 0])
        pl.sub(cur_out, cur_a, SUB_VALUE)
        pl.store(out, cur_out, [512, 0])
        pl.div(cur_out, cur_a, DIVISOR)
        pl.store(out, cur_out, [576, 0])
        pl.add(cur_out, cur_a, ADD_VALUE)
        pl.store(out, cur_out, [640, 0])
        pl.maximum(cur_out, cur_a, MAX_SCALAR)
        pl.store(out, cur_out, [704, 0])
        pl.exp(cur_out, cur_a)
        pl.store(out, cur_out, [768, 0])
        pl.relu(cur_out, cur_a)
        pl.store(out, cur_out, [832, 0])
        pl.neg(cur_out, cur_a)
        pl.store(out, cur_out, [896, 0])
        pl.rsqrt(cur_out, cur_positive)
        pl.store(out, cur_out, [960, 0])
        pl.recip(cur_out, cur_positive)
        pl.store(out, cur_out, [1024, 0])
        pl.sqrt(cur_out, cur_positive)
        pl.store(out, cur_out, [1088, 0])
        pl.add_relu(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [1152, 0])
        # Fused operations use the first input tile as scratch. Restore it so
        # each output checks the same independent input as its standalone test.
        pl.load(cur_a, a, [0, 0])
        pl.sub_relu(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [1216, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_vector_fp32_operations():
    _require_a5(ST_DEVICE)
    torch.manual_seed(0)
    a = torch.randn(64, 64, device=ST_DEVICE)
    b = torch.randn(64, 64, device=ST_DEVICE)
    positive = torch.rand(64, 64, device=ST_DEVICE) + 1.0
    out = torch.full((1280, 64), float("nan"), device=ST_DEVICE)
    vector_fp32_kernel(a, b, positive, out)
    torch.npu.synchronize()
    references = [
        ("add", a + b),
        ("sub", a - b),
        ("mul", a * b),
        ("maximum", torch.maximum(a, b)),
        ("div", a / b),
        ("minimum", torch.minimum(a, b)),
        ("minimum_scalar", torch.clamp_max(a, MIN_SCALAR)),
        ("mul_scalar", a * SCALE),
        ("sub_scalar", a - SUB_VALUE),
        ("div_scalar", a / DIVISOR),
        ("add_scalar", a + ADD_VALUE),
        ("maximum_scalar", torch.clamp_min(a, MAX_SCALAR)),
        ("exp", torch.exp(a)),
        ("relu", torch.relu(a)),
        ("neg", -a),
        ("rsqrt", torch.rsqrt(positive)),
        ("recip", torch.reciprocal(positive)),
        ("sqrt", torch.sqrt(positive)),
        ("add_relu", torch.relu(a + b)),
        ("sub_relu", torch.relu(a - b)),
    ]
    for index, (name, expected) in enumerate(references):
        torch.testing.assert_close(
            out[index * 64:(index + 1) * 64], expected, rtol=1e-2, atol=1e-2, msg=lambda message: f"{name}: {message}"
        )


@pl.jit(auto_mutex=True)
def xor_kernel(
    a: pl.Tensor[[64, 64], pl.DT_INT32],
    b: pl.Tensor[[64, 64], pl.DT_INT32],
    out: pl.Tensor[[64, 64], pl.DT_INT32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_tmp = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    tile_out = pl.make_tile_group(type=tt, addrs=0xC000, mutex_ids=[3])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_tmp = tile_tmp.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.xor(cur_out, cur_a, cur_b, cur_tmp)
        pl.store(out, cur_out, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_xor():
    device = ST_DEVICE
    _require_a5(device)
    a = torch.arange(64 * 64, device=device, dtype=torch.int32).reshape(64, 64) + 2
    b = (torch.arange(64 * 64, device=device, dtype=torch.int32).reshape(64, 64) % 16) + 1
    out = torch.zeros((64, 64), device=device, dtype=torch.int32)
    xor_kernel(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.bitwise_xor(a, b), rtol=0, atol=0)
    logging.info("xor result equal!")


@pl.jit(auto_mutex=True)
def fused_mul_add_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32],
    b: pl.Tensor[[64, 64], pl.DT_FP32],
    c: pl.Tensor[[64, 64], pl.DT_FP32],
):
    # fused_mul_add 为 in-place 融合乘加：c = c * a + b
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_c = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_c = tile_c.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.load(cur_c, c, [0, 0])
        pl.fused_mul_add(cur_c, cur_a, cur_b)
        pl.store(c, cur_c, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_fused_mul_add():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.randn(64, 64, device=device, dtype=torch.float32)
    b = torch.randn(64, 64, device=device, dtype=torch.float32)
    c = torch.randn(64, 64, device=device, dtype=torch.float32)
    c_ref = c * a + b
    # 单核验证：kernel 对 c 做原位读改写，直调现在解析为满核（多核回写同一区域会竞态），这里显式请求 1 块。
    fused_mul_add_kernel[None, 1](a, b, c)
    torch.npu.synchronize()
    torch.testing.assert_close(c, c_ref, rtol=1e-2, atol=1e-2)
    logging.info("fused_mul_add result equal!")


@pl.jit(auto_mutex=True)
def fused_cast_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP16], b: pl.Tensor[[64, 64], pl.DT_FP16], out: pl.Tensor[[192, 64], pl.DT_FP32]
):
    tt_in = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tt_out = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt_in, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt_in, addrs=0x2000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt_out, addrs=0x4000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.add_relu_cast(cur_out, cur_a, cur_b, target_type=pl.DT_FP32, mode=pl.RoundMode.CAST_ROUND)
        pl.store(out, cur_out, [0, 0])
        pl.load(cur_a, a, [0, 0])
        pl.mul_cast(cur_out, cur_a, cur_b, target_type=pl.DT_FP32, mode=pl.RoundMode.CAST_ROUND)
        pl.store(out, cur_out, [64, 0])
        pl.load(cur_a, a, [0, 0])
        pl.sub_relu_cast(cur_out, cur_a, cur_b, target_type=pl.DT_FP32, mode=pl.RoundMode.CAST_ROUND)
        pl.store(out, cur_out, [128, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_fused_cast_operations():
    _require_a5(ST_DEVICE)
    torch.manual_seed(0)
    a = torch.randn(64, 64, device=ST_DEVICE, dtype=torch.float16)
    b = torch.randn(64, 64, device=ST_DEVICE, dtype=torch.float16)
    out = torch.full((192, 64), float("nan"), device=ST_DEVICE, dtype=torch.float32)
    fused_cast_kernel(a, b, out)
    torch.npu.synchronize()
    a, b = a.float(), b.float()
    references = [("add_relu_cast", torch.relu(a + b)), ("mul_cast", a * b), ("sub_relu_cast", torch.relu(a - b))]
    for index, (name, expected) in enumerate(references):
        torch.testing.assert_close(
            out[index * 64:(index + 1) * 64], expected, rtol=1e-2, atol=1e-2, msg=lambda message: f"{name}: {message}"
        )
