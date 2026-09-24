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

"""ST for 20 PyPTO tile-level interfaces from the transfer-to-test checklist.

Covers all 20 interfaces with aligned + unaligned (partial-tile) shapes:
  1.  pl.add       2.  pl.sub        3.  pl.mul        4.  pl.and_
  5.  pl.maximum   6.  pl.neg        7.  pl.abs        8.  pl.axpy
  9.  pl.relu      10. pl.xor        11. pl.cast      12. pl.eq
  13. pl.select    14. pl.div        15. pl.sum        16. pl.load
  17. pl.store     18. pl.expands    19. pl.transpose  20. pl.add_relu

Unaligned shapes (non-multiple of TILE_N=64) exercise the partial-tile path
via ``pl.set_validshape``, verifying correctness on edge tiles.

Requires an Ascend 950 (A5) device; skips otherwise.
"""

import os

from pypto_pro._errors import InvalidOperation
import pypto_pro.language as pl
import pytest
import torch
import torch_npu  # noqa: F401 — registers npu backend

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

TILE_M = 1
TILE_N = 64
TILE_SIZE = 256
UNALIGN_N = 47

SCALAR_VAL = 2.0
DYN = pl.DYNAMIC


def _check_npu():
    try:
        torch.npu.set_device(ST_DEVICE)
        name = torch.npu.get_device_name()
        if "Ascend950" not in name:
            pytest.skip(f"Device {name} is not A5 (Ascend950). Skip.")
        return True
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
        return False


def _run(kernel, *args, **kwargs):
    kernel[None, 1](*args, **kwargs)
    torch.npu.synchronize()


# =============================================================================
# 1. pl.add — element-wise addition
# =============================================================================

@pl.jit()
def kernel_add_fp32(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    b: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tb = pl.make_tile(tf, addr=TILE_SIZE)
    tc = pl.make_tile(tf, addr=TILE_SIZE * 2)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.add(tc, ta, tb)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_add_fp32_scalar(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tc = pl.make_tile(tf, addr=TILE_SIZE)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.add(tc, ta, SCALAR_VAL)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_add_int32(
    a: pl.Tensor[[DYN, DYN], pl.DT_INT32],
    b: pl.Tensor[[DYN, DYN], pl.DT_INT32],
    out: pl.Tensor[[DYN, DYN], pl.DT_INT32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tb = pl.make_tile(tf, addr=TILE_SIZE)
    tc = pl.make_tile(tf, addr=TILE_SIZE * 2)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.add(tc, ta, tb)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_add_fp32_unaligned(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    b: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tb = pl.make_tile(tf, addr=TILE_SIZE)
    tc = pl.make_tile(tf, addr=TILE_SIZE * 2)
    pl.set_validshape(ta, [TILE_M, UNALIGN_N])
    pl.set_validshape(tb, [TILE_M, UNALIGN_N])
    pl.set_validshape(tc, [TILE_M, UNALIGN_N])
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.add(tc, ta, tb)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pytest.mark.soc("950")
def test_add_fp32():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    b = torch.randn(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    _run(kernel_add_fp32, a, b, out)
    torch.testing.assert_close(out, a + b, rtol=1e-5, atol=1e-5)


@pytest.mark.soc("950")
def test_add_fp32_scalar():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    _run(kernel_add_fp32_scalar, a, out)
    torch.testing.assert_close(out, a + SCALAR_VAL, rtol=1e-5, atol=1e-5)


@pytest.mark.soc("950")
def test_add_int32():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randint(-1000, 1000, (TILE_M, TILE_N), device=ST_DEVICE, dtype=torch.int32)
    b = torch.randint(-1000, 1000, (TILE_M, TILE_N), device=ST_DEVICE, dtype=torch.int32)
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.int32)
    _run(kernel_add_int32, a, b, out)
    torch.testing.assert_close(out, a + b, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_add_fp32_unaligned():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32)
    b = torch.randn(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32)
    out = torch.empty(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32)
    _run(kernel_add_fp32_unaligned, a, b, out)
    torch.testing.assert_close(out, a + b, rtol=1e-5, atol=1e-5)


# =============================================================================
# 2. pl.sub — element-wise subtraction
# =============================================================================

@pl.jit()
def kernel_sub_fp32(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    b: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tb = pl.make_tile(tf, addr=TILE_SIZE)
    tc = pl.make_tile(tf, addr=TILE_SIZE * 2)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.sub(tc, ta, tb)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_sub_fp32_scalar(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tc = pl.make_tile(tf, addr=TILE_SIZE)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.sub(tc, ta, SCALAR_VAL)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_sub_int32(
    a: pl.Tensor[[DYN, DYN], pl.DT_INT32],
    b: pl.Tensor[[DYN, DYN], pl.DT_INT32],
    out: pl.Tensor[[DYN, DYN], pl.DT_INT32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tb = pl.make_tile(tf, addr=TILE_SIZE)
    tc = pl.make_tile(tf, addr=TILE_SIZE * 2)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.sub(tc, ta, tb)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_sub_fp32_unaligned(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    b: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tb = pl.make_tile(tf, addr=TILE_SIZE)
    tc = pl.make_tile(tf, addr=TILE_SIZE * 2)
    pl.set_validshape(ta, [TILE_M, UNALIGN_N])
    pl.set_validshape(tb, [TILE_M, UNALIGN_N])
    pl.set_validshape(tc, [TILE_M, UNALIGN_N])
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.sub(tc, ta, tb)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pytest.mark.soc("950")
def test_sub_fp32():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    b = torch.randn(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    _run(kernel_sub_fp32, a, b, out)
    torch.testing.assert_close(out, a - b, rtol=1e-5, atol=1e-5)


@pytest.mark.soc("950")
def test_sub_fp32_scalar():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    _run(kernel_sub_fp32_scalar, a, out)
    torch.testing.assert_close(out, a - SCALAR_VAL, rtol=1e-5, atol=1e-5)


@pytest.mark.soc("950")
def test_sub_int32():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randint(-1000, 1000, (TILE_M, TILE_N), device=ST_DEVICE, dtype=torch.int32)
    b = torch.randint(-1000, 1000, (TILE_M, TILE_N), device=ST_DEVICE, dtype=torch.int32)
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.int32)
    _run(kernel_sub_int32, a, b, out)
    torch.testing.assert_close(out, a - b, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_sub_fp32_unaligned():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32)
    b = torch.randn(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32)
    out = torch.empty(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32)
    _run(kernel_sub_fp32_unaligned, a, b, out)
    torch.testing.assert_close(out, a - b, rtol=1e-5, atol=1e-5)


# =============================================================================
# 3. pl.mul — element-wise multiplication
# =============================================================================

@pl.jit()
def kernel_mul_fp32(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    b: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tb = pl.make_tile(tf, addr=TILE_SIZE)
    tc = pl.make_tile(tf, addr=TILE_SIZE * 2)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.mul(tc, ta, tb)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_mul_fp32_scalar(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tc = pl.make_tile(tf, addr=TILE_SIZE)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.mul(tc, ta, SCALAR_VAL)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_mul_int32(
    a: pl.Tensor[[DYN, DYN], pl.DT_INT32],
    b: pl.Tensor[[DYN, DYN], pl.DT_INT32],
    out: pl.Tensor[[DYN, DYN], pl.DT_INT32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tb = pl.make_tile(tf, addr=TILE_SIZE)
    tc = pl.make_tile(tf, addr=TILE_SIZE * 2)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.mul(tc, ta, tb)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_mul_fp32_unaligned(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    b: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tb = pl.make_tile(tf, addr=TILE_SIZE)
    tc = pl.make_tile(tf, addr=TILE_SIZE * 2)
    pl.set_validshape(ta, [TILE_M, UNALIGN_N])
    pl.set_validshape(tb, [TILE_M, UNALIGN_N])
    pl.set_validshape(tc, [TILE_M, UNALIGN_N])
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.mul(tc, ta, tb)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pytest.mark.soc("950")
def test_mul_fp32():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    b = torch.randn(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    _run(kernel_mul_fp32, a, b, out)
    torch.testing.assert_close(out, a * b, rtol=1e-5, atol=1e-5)


@pytest.mark.soc("950")
def test_mul_fp32_scalar():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    _run(kernel_mul_fp32_scalar, a, out)
    torch.testing.assert_close(out, a * SCALAR_VAL, rtol=1e-5, atol=1e-5)


@pytest.mark.soc("950")
def test_mul_int32():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randint(-100, 100, (TILE_M, TILE_N), device=ST_DEVICE, dtype=torch.int32)
    b = torch.randint(-100, 100, (TILE_M, TILE_N), device=ST_DEVICE, dtype=torch.int32)
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.int32)
    _run(kernel_mul_int32, a, b, out)
    torch.testing.assert_close(out, a * b, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_mul_fp32_unaligned():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32)
    b = torch.randn(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32)
    out = torch.empty(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32)
    _run(kernel_mul_fp32_unaligned, a, b, out)
    torch.testing.assert_close(out, a * b, rtol=1e-5, atol=1e-5)


# =============================================================================
# 4. pl.and_ — element-wise bitwise AND
# =============================================================================

@pl.jit()
def kernel_and_int32(
    a: pl.Tensor[[DYN, DYN], pl.DT_INT32],
    b: pl.Tensor[[DYN, DYN], pl.DT_INT32],
    out: pl.Tensor[[DYN, DYN], pl.DT_INT32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tb = pl.make_tile(tf, addr=TILE_SIZE)
    tc = pl.make_tile(tf, addr=TILE_SIZE * 2)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.and_(tc, ta, tb)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_and_int16(
    a: pl.Tensor[[DYN, DYN], pl.DT_INT16],
    b: pl.Tensor[[DYN, DYN], pl.DT_INT16],
    out: pl.Tensor[[DYN, DYN], pl.DT_INT16],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_INT16, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tb = pl.make_tile(tf, addr=TILE_SIZE)
    tc = pl.make_tile(tf, addr=TILE_SIZE * 2)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.and_(tc, ta, tb)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_and_int32_unaligned(
    a: pl.Tensor[[DYN, DYN], pl.DT_INT32],
    b: pl.Tensor[[DYN, DYN], pl.DT_INT32],
    out: pl.Tensor[[DYN, DYN], pl.DT_INT32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tb = pl.make_tile(tf, addr=TILE_SIZE)
    tc = pl.make_tile(tf, addr=TILE_SIZE * 2)
    pl.set_validshape(ta, [TILE_M, UNALIGN_N])
    pl.set_validshape(tb, [TILE_M, UNALIGN_N])
    pl.set_validshape(tc, [TILE_M, UNALIGN_N])
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.and_(tc, ta, tb)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pytest.mark.soc("950")
def test_and_int32():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randint(0, 65536, (TILE_M, TILE_N), device=ST_DEVICE, dtype=torch.int32)
    b = torch.randint(0, 65536, (TILE_M, TILE_N), device=ST_DEVICE, dtype=torch.int32)
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.int32)
    _run(kernel_and_int32, a, b, out)
    torch.testing.assert_close(out, a & b, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_and_int16():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randint(0, 256, (TILE_M, TILE_N), device=ST_DEVICE, dtype=torch.int16)
    b = torch.randint(0, 256, (TILE_M, TILE_N), device=ST_DEVICE, dtype=torch.int16)
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.int16)
    _run(kernel_and_int16, a, b, out)
    torch.testing.assert_close(out, a & b, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_and_int32_unaligned():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randint(0, 65536, (TILE_M, UNALIGN_N), device=ST_DEVICE, dtype=torch.int32)
    b = torch.randint(0, 65536, (TILE_M, UNALIGN_N), device=ST_DEVICE, dtype=torch.int32)
    out = torch.empty(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.int32)
    _run(kernel_and_int32_unaligned, a, b, out)
    torch.testing.assert_close(out, a & b, rtol=0, atol=0)


# =============================================================================
# 5. pl.maximum — element-wise maximum
# =============================================================================

@pl.jit()
def kernel_maximum_fp32(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    b: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tb = pl.make_tile(tf, addr=TILE_SIZE)
    tc = pl.make_tile(tf, addr=TILE_SIZE * 2)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.maximum(tc, ta, tb)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_maximum_int32(
    a: pl.Tensor[[DYN, DYN], pl.DT_INT32],
    b: pl.Tensor[[DYN, DYN], pl.DT_INT32],
    out: pl.Tensor[[DYN, DYN], pl.DT_INT32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tb = pl.make_tile(tf, addr=TILE_SIZE)
    tc = pl.make_tile(tf, addr=TILE_SIZE * 2)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.maximum(tc, ta, tb)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_maximum_fp32_unaligned(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    b: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tb = pl.make_tile(tf, addr=TILE_SIZE)
    tc = pl.make_tile(tf, addr=TILE_SIZE * 2)
    pl.set_validshape(ta, [TILE_M, UNALIGN_N])
    pl.set_validshape(tb, [TILE_M, UNALIGN_N])
    pl.set_validshape(tc, [TILE_M, UNALIGN_N])
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.maximum(tc, ta, tb)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pytest.mark.soc("950")
def test_maximum_fp32():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    b = torch.randn(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    _run(kernel_maximum_fp32, a, b, out)
    torch.testing.assert_close(out, torch.maximum(a, b), rtol=1e-5, atol=1e-5)


@pytest.mark.soc("950")
def test_maximum_int32():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randint(-100, 100, (TILE_M, TILE_N), device=ST_DEVICE, dtype=torch.int32)
    b = torch.randint(-100, 100, (TILE_M, TILE_N), device=ST_DEVICE, dtype=torch.int32)
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.int32)
    _run(kernel_maximum_int32, a, b, out)
    torch.testing.assert_close(out, torch.maximum(a, b), rtol=0, atol=0)


@pytest.mark.soc("950")
def test_maximum_fp32_unaligned():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32)
    b = torch.randn(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32)
    out = torch.empty(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32)
    _run(kernel_maximum_fp32_unaligned, a, b, out)
    torch.testing.assert_close(out, torch.maximum(a, b), rtol=1e-5, atol=1e-5)


# =============================================================================
# 6. pl.neg — element-wise negate
# =============================================================================

@pl.jit()
def kernel_neg_fp32(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tc = pl.make_tile(tf, addr=TILE_SIZE)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.neg(tc, ta)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_neg_int32(
    a: pl.Tensor[[DYN, DYN], pl.DT_INT32],
    out: pl.Tensor[[DYN, DYN], pl.DT_INT32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tc = pl.make_tile(tf, addr=TILE_SIZE)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.neg(tc, ta)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_neg_fp32_unaligned(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tc = pl.make_tile(tf, addr=TILE_SIZE)
    pl.set_validshape(ta, [TILE_M, UNALIGN_N])
    pl.set_validshape(tc, [TILE_M, UNALIGN_N])
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.neg(tc, ta)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pytest.mark.soc("950")
def test_neg_fp32():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    _run(kernel_neg_fp32, a, out)
    torch.testing.assert_close(out, -a, rtol=1e-5, atol=1e-5)


@pytest.mark.soc("950")
def test_neg_int32():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randint(-1000, 1000, (TILE_M, TILE_N), device=ST_DEVICE, dtype=torch.int32)
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.int32)
    _run(kernel_neg_int32, a, out)
    torch.testing.assert_close(out, -a, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_neg_fp32_unaligned():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32)
    out = torch.empty(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32)
    _run(kernel_neg_fp32_unaligned, a, out)
    torch.testing.assert_close(out, -a, rtol=1e-5, atol=1e-5)


# =============================================================================
# 7. pl.abs — element-wise absolute value
# =============================================================================

@pl.jit()
def kernel_abs_fp32(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tc = pl.make_tile(tf, addr=TILE_SIZE)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.abs(tc, ta)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_abs_int32(
    a: pl.Tensor[[DYN, DYN], pl.DT_INT32],
    out: pl.Tensor[[DYN, DYN], pl.DT_INT32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tc = pl.make_tile(tf, addr=TILE_SIZE)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.abs(tc, ta)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_abs_fp32_unaligned(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tc = pl.make_tile(tf, addr=TILE_SIZE)
    pl.set_validshape(ta, [TILE_M, UNALIGN_N])
    pl.set_validshape(tc, [TILE_M, UNALIGN_N])
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.abs(tc, ta)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pytest.mark.soc("950")
def test_abs_fp32():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32) * 10
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    _run(kernel_abs_fp32, a, out)
    torch.testing.assert_close(out, torch.abs(a), rtol=1e-5, atol=1e-5)


@pytest.mark.soc("950")
def test_abs_int32():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randint(-100, 100, (TILE_M, TILE_N), device=ST_DEVICE, dtype=torch.int32)
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.int32)
    _run(kernel_abs_int32, a, out)
    torch.testing.assert_close(out, torch.abs(a), rtol=0, atol=0)


@pytest.mark.soc("950")
def test_abs_fp32_unaligned():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32) * 10
    out = torch.empty(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32)
    _run(kernel_abs_fp32_unaligned, a, out)
    torch.testing.assert_close(out, torch.abs(a), rtol=1e-5, atol=1e-5)


# =============================================================================
# 8. pl.axpy — AXPY: out = alpha * src + out
# =============================================================================

@pl.jit()
def kernel_axpy_fp32(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tc = pl.make_tile(tf, addr=TILE_SIZE)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tc, out, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.axpy(tc, ta, SCALAR_VAL)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_axpy_fp32_unaligned(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tc = pl.make_tile(tf, addr=TILE_SIZE)
    pl.set_validshape(ta, [TILE_M, UNALIGN_N])
    pl.set_validshape(tc, [TILE_M, UNALIGN_N])
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tc, out, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.axpy(tc, ta, SCALAR_VAL)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pytest.mark.soc("950")
def test_axpy_fp32():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    out = torch.randn(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    ref = SCALAR_VAL * a + out.clone()
    _run(kernel_axpy_fp32, a, out)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.soc("950")
def test_axpy_fp32_unaligned():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32)
    out = torch.randn(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32)
    ref = SCALAR_VAL * a + out.clone()
    _run(kernel_axpy_fp32_unaligned, a, out)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


# =============================================================================
# 9. pl.relu — element-wise ReLU
# =============================================================================

@pl.jit()
def kernel_relu_fp32(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tc = pl.make_tile(tf, addr=TILE_SIZE)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.relu(tc, ta)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_relu_int32(
    a: pl.Tensor[[DYN, DYN], pl.DT_INT32],
    out: pl.Tensor[[DYN, DYN], pl.DT_INT32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tc = pl.make_tile(tf, addr=TILE_SIZE)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.relu(tc, ta)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_relu_fp32_unaligned(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tc = pl.make_tile(tf, addr=TILE_SIZE)
    pl.set_validshape(ta, [TILE_M, UNALIGN_N])
    pl.set_validshape(tc, [TILE_M, UNALIGN_N])
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.relu(tc, ta)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pytest.mark.soc("950")
def test_relu_fp32():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32) * 10
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    _run(kernel_relu_fp32, a, out)
    torch.testing.assert_close(out, torch.relu(a), rtol=1e-5, atol=1e-5)


@pytest.mark.soc("950")
def test_relu_int32():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randint(-100, 100, (TILE_M, TILE_N), device=ST_DEVICE, dtype=torch.int32)
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.int32)
    _run(kernel_relu_int32, a, out)
    torch.testing.assert_close(out, torch.relu(a), rtol=0, atol=0)


@pytest.mark.soc("950")
def test_relu_fp32_unaligned():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32) * 10
    out = torch.empty(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32)
    _run(kernel_relu_fp32_unaligned, a, out)
    torch.testing.assert_close(out, torch.relu(a), rtol=1e-5, atol=1e-5)


# =============================================================================
# 10. pl.xor — element-wise bitwise XOR (dst, lhs, rhs, tmp)
# =============================================================================

@pl.jit()
def kernel_xor_int32(
    a: pl.Tensor[[DYN, DYN], pl.DT_INT32],
    b: pl.Tensor[[DYN, DYN], pl.DT_INT32],
    out: pl.Tensor[[DYN, DYN], pl.DT_INT32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tb = pl.make_tile(tf, addr=TILE_SIZE)
    tc = pl.make_tile(tf, addr=TILE_SIZE * 2)
    ttmp = pl.make_tile(tf, addr=TILE_SIZE * 3)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.xor(tc, ta, tb, ttmp)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_xor_int16(
    a: pl.Tensor[[DYN, DYN], pl.DT_INT16],
    b: pl.Tensor[[DYN, DYN], pl.DT_INT16],
    out: pl.Tensor[[DYN, DYN], pl.DT_INT16],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_INT16, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tb = pl.make_tile(tf, addr=TILE_SIZE)
    tc = pl.make_tile(tf, addr=TILE_SIZE * 2)
    ttmp = pl.make_tile(tf, addr=TILE_SIZE * 3)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.xor(tc, ta, tb, ttmp)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_xor_int32_unaligned(
    a: pl.Tensor[[DYN, DYN], pl.DT_INT32],
    b: pl.Tensor[[DYN, DYN], pl.DT_INT32],
    out: pl.Tensor[[DYN, DYN], pl.DT_INT32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tb = pl.make_tile(tf, addr=TILE_SIZE)
    tc = pl.make_tile(tf, addr=TILE_SIZE * 2)
    ttmp = pl.make_tile(tf, addr=TILE_SIZE * 3)
    pl.set_validshape(ta, [TILE_M, UNALIGN_N])
    pl.set_validshape(tb, [TILE_M, UNALIGN_N])
    pl.set_validshape(tc, [TILE_M, UNALIGN_N])
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.xor(tc, ta, tb, ttmp)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pytest.mark.soc("950")
def test_xor_int32():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randint(0, 65536, (TILE_M, TILE_N), device=ST_DEVICE, dtype=torch.int32)
    b = torch.randint(0, 65536, (TILE_M, TILE_N), device=ST_DEVICE, dtype=torch.int32)
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.int32)
    _run(kernel_xor_int32, a, b, out)
    torch.testing.assert_close(out, a ^ b, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_xor_int16():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randint(0, 256, (TILE_M, TILE_N), device=ST_DEVICE, dtype=torch.int16)
    b = torch.randint(0, 256, (TILE_M, TILE_N), device=ST_DEVICE, dtype=torch.int16)
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.int16)
    _run(kernel_xor_int16, a, b, out)
    torch.testing.assert_close(out, a ^ b, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_xor_int32_unaligned():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randint(0, 65536, (TILE_M, UNALIGN_N), device=ST_DEVICE, dtype=torch.int32)
    b = torch.randint(0, 65536, (TILE_M, UNALIGN_N), device=ST_DEVICE, dtype=torch.int32)
    out = torch.empty(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.int32)
    _run(kernel_xor_int32_unaligned, a, b, out)
    torch.testing.assert_close(out, a ^ b, rtol=0, atol=0)


# =============================================================================
# 11. pl.cast — type conversion
# =============================================================================

@pl.jit()
def kernel_cast_fp32_to_int32(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_INT32],
):
    tf_f = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tf_i = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf_f, addr=0)
    tc = pl.make_tile(tf_i, addr=TILE_SIZE)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.cast(tc, ta, mode=pl.RoundMode.CAST_TRUNC)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_cast_int32_to_fp32(
    a: pl.Tensor[[DYN, DYN], pl.DT_INT32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf_i = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    tf_f = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf_i, addr=0)
    tc = pl.make_tile(tf_f, addr=TILE_SIZE)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.cast(tc, ta)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_cast_fp32_to_int32_unaligned(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_INT32],
):
    tf_f = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tf_i = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf_f, addr=0)
    tc = pl.make_tile(tf_i, addr=TILE_SIZE)
    pl.set_validshape(ta, [TILE_M, UNALIGN_N])
    pl.set_validshape(tc, [TILE_M, UNALIGN_N])
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.cast(tc, ta, mode=pl.RoundMode.CAST_TRUNC)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pytest.mark.soc("950")
def test_cast_fp32_to_int32():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32) * 100
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.int32)
    _run(kernel_cast_fp32_to_int32, a, out)
    torch.testing.assert_close(out, a.to(torch.int32), rtol=0, atol=0)


@pytest.mark.soc("950")
def test_cast_int32_to_fp32():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randint(-1000, 1000, (TILE_M, TILE_N), device=ST_DEVICE, dtype=torch.int32)
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    _run(kernel_cast_int32_to_fp32, a, out)
    torch.testing.assert_close(out, a.to(torch.float32), rtol=1e-5, atol=1e-5)


@pytest.mark.soc("950")
def test_cast_fp32_to_int32_unaligned():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32) * 100
    out = torch.empty(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.int32)
    _run(kernel_cast_fp32_to_int32_unaligned, a, out)
    torch.testing.assert_close(out, a.to(torch.int32), rtol=0, atol=0)


# =============================================================================
# 12. pl.eq — element-wise equality
# =============================================================================

@pl.jit()
def kernel_eq_fp32(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    b: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tm = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tb = pl.make_tile(tf, addr=TILE_SIZE)
    t_mask = pl.make_tile(tm, addr=TILE_SIZE * 2)
    t_one = pl.make_tile(tf, addr=TILE_SIZE * 3)
    t_out = pl.make_tile(tf, addr=TILE_SIZE * 4)
    tmp = pl.make_tile(tf, addr=TILE_SIZE * 5)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.eq(t_mask, ta, tb)
        pl.expands(t_one, 1.0)
        pl.select(t_out, t_mask, t_one, 0.0, tmp)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])


@pl.jit()
def kernel_eq_int32(
    a: pl.Tensor[[DYN, DYN], pl.DT_INT32],
    b: pl.Tensor[[DYN, DYN], pl.DT_INT32],
    out: pl.Tensor[[DYN, DYN], pl.DT_INT32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    tm = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tb = pl.make_tile(tf, addr=TILE_SIZE)
    t_mask = pl.make_tile(tm, addr=TILE_SIZE * 2)
    t_one = pl.make_tile(tf, addr=TILE_SIZE * 3)
    t_out = pl.make_tile(tf, addr=TILE_SIZE * 4)
    tmp = pl.make_tile(tf, addr=TILE_SIZE * 5)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.eq(t_mask, ta, tb)
        pl.expands(t_one, 1)
        pl.select(t_out, t_mask, t_one, 0, tmp)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])


@pl.jit()
def kernel_eq_fp32_unaligned(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    b: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tm = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tb = pl.make_tile(tf, addr=TILE_SIZE)
    t_mask = pl.make_tile(tm, addr=TILE_SIZE * 2)
    t_one = pl.make_tile(tf, addr=TILE_SIZE * 3)
    t_out = pl.make_tile(tf, addr=TILE_SIZE * 4)
    tmp = pl.make_tile(tf, addr=TILE_SIZE * 5)
    pl.set_validshape(ta, [TILE_M, UNALIGN_N])
    pl.set_validshape(tb, [TILE_M, UNALIGN_N])
    pl.set_validshape(t_mask, [TILE_M, UNALIGN_N])
    pl.set_validshape(t_one, [TILE_M, UNALIGN_N])
    pl.set_validshape(t_out, [TILE_M, UNALIGN_N])
    pl.set_validshape(tmp, [TILE_M, UNALIGN_N])
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.eq(t_mask, ta, tb)
        pl.expands(t_one, 1.0)
        pl.select(t_out, t_mask, t_one, 0.0, tmp)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])


@pytest.mark.soc("950")
def test_eq_fp32():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    b = a.clone()
    b[0, ::3] += 1.0
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    _run(kernel_eq_fp32, a, b, out)
    torch.testing.assert_close(out, (a == b).to(torch.float32), rtol=0, atol=0)


@pytest.mark.soc("950")
def test_eq_int32():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randint(0, 10, (TILE_M, TILE_N), device=ST_DEVICE, dtype=torch.int32)
    b = a.clone()
    b[0, ::3] += 1
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.int32)
    _run(kernel_eq_int32, a, b, out)
    torch.testing.assert_close(out, (a == b).to(torch.int32), rtol=0, atol=0)


@pytest.mark.soc("950")
def test_eq_fp32_unaligned():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32)
    b = a.clone()
    b[0, ::3] += 1.0
    out = torch.empty(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32)
    _run(kernel_eq_fp32_unaligned, a, b, out)
    torch.testing.assert_close(out, (a == b).to(torch.float32), rtol=0, atol=0)


# =============================================================================
# 13. pl.select — select by mask
# =============================================================================

@pl.jit()
def kernel_select_fp32(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    b: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tm = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tb = pl.make_tile(tf, addr=TILE_SIZE)
    t_mask = pl.make_tile(tm, addr=TILE_SIZE * 2)
    t_out = pl.make_tile(tf, addr=TILE_SIZE * 3)
    tmp = pl.make_tile(tf, addr=TILE_SIZE * 4)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.eq(t_mask, ta, tb)
        pl.select(t_out, t_mask, ta, tb, tmp)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])


@pl.jit()
def kernel_select_fp32_unaligned(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    b: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tm = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tb = pl.make_tile(tf, addr=TILE_SIZE)
    t_mask = pl.make_tile(tm, addr=TILE_SIZE * 2)
    t_out = pl.make_tile(tf, addr=TILE_SIZE * 3)
    tmp = pl.make_tile(tf, addr=TILE_SIZE * 4)
    pl.set_validshape(ta, [TILE_M, UNALIGN_N])
    pl.set_validshape(tb, [TILE_M, UNALIGN_N])
    pl.set_validshape(t_mask, [TILE_M, UNALIGN_N])
    pl.set_validshape(t_out, [TILE_M, UNALIGN_N])
    pl.set_validshape(tmp, [TILE_M, UNALIGN_N])
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.eq(t_mask, ta, tb)
        pl.select(t_out, t_mask, ta, tb, tmp)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])


@pytest.mark.soc("950")
def test_select_fp32():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    b = torch.randn(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    _run(kernel_select_fp32, a, b, out)
    ref = torch.where(a == b, a, b)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.soc("950")
def test_select_fp32_unaligned():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32)
    b = torch.randn(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32)
    out = torch.empty(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32)
    _run(kernel_select_fp32_unaligned, a, b, out)
    ref = torch.where(a == b, a, b)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


# =============================================================================
# 14. pl.div — element-wise division
# =============================================================================

@pl.jit()
def kernel_div_fp32(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    b: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tb = pl.make_tile(tf, addr=TILE_SIZE)
    tc = pl.make_tile(tf, addr=TILE_SIZE * 2)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.div(tc, ta, tb)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_div_fp32_scalar(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tc = pl.make_tile(tf, addr=TILE_SIZE)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.div(tc, ta, SCALAR_VAL)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_div_int32(
    a: pl.Tensor[[DYN, DYN], pl.DT_INT32],
    b: pl.Tensor[[DYN, DYN], pl.DT_INT32],
    out: pl.Tensor[[DYN, DYN], pl.DT_INT32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tb = pl.make_tile(tf, addr=TILE_SIZE)
    tc = pl.make_tile(tf, addr=TILE_SIZE * 2)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.div(tc, ta, tb)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_div_fp32_unaligned(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    b: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tb = pl.make_tile(tf, addr=TILE_SIZE)
    tc = pl.make_tile(tf, addr=TILE_SIZE * 2)
    pl.set_validshape(ta, [TILE_M, UNALIGN_N])
    pl.set_validshape(tb, [TILE_M, UNALIGN_N])
    pl.set_validshape(tc, [TILE_M, UNALIGN_N])
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.div(tc, ta, tb)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pytest.mark.soc("950")
def test_div_fp32():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32) * 10 + 1.0
    b = torch.randn(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32) * 10 + 1.0
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    _run(kernel_div_fp32, a, b, out)
    torch.testing.assert_close(out, a / b, rtol=1e-5, atol=1e-5)


@pytest.mark.soc("950")
def test_div_fp32_scalar():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32) * 10 + 1.0
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    _run(kernel_div_fp32_scalar, a, out)
    torch.testing.assert_close(out, a / SCALAR_VAL, rtol=1e-5, atol=1e-5)


@pytest.mark.soc("950")
def test_div_int32():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randint(1, 1000, (TILE_M, TILE_N), device=ST_DEVICE, dtype=torch.int32)
    b = torch.randint(1, 100, (TILE_M, TILE_N), device=ST_DEVICE, dtype=torch.int32)
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.int32)
    _run(kernel_div_int32, a, b, out)
    torch.testing.assert_close(out, a // b, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_div_fp32_unaligned():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32) * 10 + 1.0
    b = torch.randn(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32) * 10 + 1.0
    out = torch.empty(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32)
    _run(kernel_div_fp32_unaligned, a, b, out)
    torch.testing.assert_close(out, a / b, rtol=1e-5, atol=1e-5)


# =============================================================================
# 15. pl.sum — sum reduction
# =============================================================================

@pl.jit()
def kernel_sum_fp32_row(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tc = pl.make_tile(tf, addr=TILE_SIZE)
    tmp = pl.make_tile(tf, addr=TILE_SIZE * 2)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.sum(tc, ta, tmp, dim=0)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_sum_fp32_row_unaligned(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tc = pl.make_tile(tf, addr=TILE_SIZE)
    tmp = pl.make_tile(tf, addr=TILE_SIZE * 2)
    pl.set_validshape(ta, [TILE_M, UNALIGN_N])
    pl.set_validshape(tc, [TILE_M, UNALIGN_N])
    pl.set_validshape(tmp, [TILE_M, UNALIGN_N])
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.sum(tc, ta, tmp, dim=0)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pytest.mark.soc("950")
def test_sum_fp32_row():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    _run(kernel_sum_fp32_row, a, out)
    torch.testing.assert_close(out[0, 0], a.sum(dim=1).squeeze(0), rtol=1e-3, atol=1e-3)


@pytest.mark.soc("950")
def test_sum_fp32_row_unaligned():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32)
    out = torch.empty(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32)
    _run(kernel_sum_fp32_row_unaligned, a, out)
    torch.testing.assert_close(out[0, 0], a.sum(dim=1).squeeze(0), rtol=1e-3, atol=1e-3)


# =============================================================================
# 16+17. pl.load + pl.store — data movement round-trip
# =============================================================================

@pl.jit()
def kernel_load_store_fp32(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=0)
        pl.store(out, ta, [0, 0])


@pl.jit()
def kernel_load_store_int32(
    a: pl.Tensor[[DYN, DYN], pl.DT_INT32],
    out: pl.Tensor[[DYN, DYN], pl.DT_INT32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=0)
        pl.store(out, ta, [0, 0])


@pl.jit()
def kernel_load_store_fp32_unaligned(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    pl.set_validshape(ta, [TILE_M, UNALIGN_N])
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=0)
        pl.store(out, ta, [0, 0])


@pytest.mark.soc("950")
def test_load_store_fp32():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    _run(kernel_load_store_fp32, a, out)
    torch.testing.assert_close(out, a, rtol=1e-5, atol=1e-5)


@pytest.mark.soc("950")
def test_load_store_int32():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randint(-1000, 1000, (TILE_M, TILE_N), device=ST_DEVICE, dtype=torch.int32)
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.int32)
    _run(kernel_load_store_int32, a, out)
    torch.testing.assert_close(out, a, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_load_store_fp32_unaligned():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32)
    out = torch.empty(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32)
    _run(kernel_load_store_fp32_unaligned, a, out)
    torch.testing.assert_close(out, a, rtol=1e-5, atol=1e-5)


# =============================================================================
# 18. pl.expands — fill Tile with scalar (splat)
# =============================================================================

@pl.jit()
def kernel_expands_fp32(
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tc = pl.make_tile(tf, addr=0)
    with pl.section_vector():
        pl.expands(tc, SCALAR_VAL)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_expands_int32(
    out: pl.Tensor[[DYN, DYN], pl.DT_INT32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    tc = pl.make_tile(tf, addr=0)
    with pl.section_vector():
        pl.expands(tc, 42)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pytest.mark.soc("950")
def test_expands_fp32():
    if not _check_npu():
        return
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    _run(kernel_expands_fp32, out)
    torch.testing.assert_close(out, torch.full((TILE_M, TILE_N), SCALAR_VAL, device=ST_DEVICE, dtype=torch.float32),
                               rtol=1e-5, atol=1e-5)


@pytest.mark.soc("950")
def test_expands_int32():
    if not _check_npu():
        return
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.int32)
    _run(kernel_expands_int32, out)
    torch.testing.assert_close(out, torch.full((TILE_M, TILE_N), 42, device=ST_DEVICE, dtype=torch.int32),
                               rtol=0, atol=0)


# =============================================================================
# 19. pl.transpose — transpose last two dimensions
# =============================================================================

T_M = 8
T_N = 8
T_M_INT16 = 16
T_N_INT16 = 16


@pl.jit()
def kernel_transpose_fp32(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf = pl.TileType(shape=[T_M, T_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tc = pl.make_tile(tf, addr=T_M * T_N * 4)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.transpose(tc, ta)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_transpose_int16(
    a: pl.Tensor[[DYN, DYN], pl.DT_INT16],
    out: pl.Tensor[[DYN, DYN], pl.DT_INT16],
):
    tf = pl.TileType(shape=[T_M_INT16, T_N_INT16], dtype=pl.DT_INT16, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tc = pl.make_tile(tf, addr=T_M_INT16 * T_N_INT16 * 4)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.transpose(tc, ta)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pytest.mark.soc("950")
def test_transpose_fp32():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(T_M, T_N, device=ST_DEVICE, dtype=torch.float32)
    out = torch.empty(T_M, T_N, device=ST_DEVICE, dtype=torch.float32)
    _run(kernel_transpose_fp32, a, out)
    torch.testing.assert_close(out, a.t(), rtol=1e-5, atol=1e-5)


@pytest.mark.soc("950")
def test_transpose_int16():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randint(-100, 100, (T_M_INT16, T_N_INT16), device=ST_DEVICE, dtype=torch.int16)
    out = torch.empty(T_M_INT16, T_N_INT16, device=ST_DEVICE, dtype=torch.int16)
    _run(kernel_transpose_int16, a, out)
    torch.testing.assert_close(out, a.t(), rtol=0, atol=0)


# =============================================================================
# 20. pl.add_relu — fused add + ReLU
# =============================================================================

@pl.jit()
def kernel_add_relu_fp32(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    b: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tb = pl.make_tile(tf, addr=TILE_SIZE)
    tc = pl.make_tile(tf, addr=TILE_SIZE * 2)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.add_relu(tc, ta, tb)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_add_relu_fp32_unaligned(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    b: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tb = pl.make_tile(tf, addr=TILE_SIZE)
    tc = pl.make_tile(tf, addr=TILE_SIZE * 2)
    pl.set_validshape(ta, [TILE_M, UNALIGN_N])
    pl.set_validshape(tb, [TILE_M, UNALIGN_N])
    pl.set_validshape(tc, [TILE_M, UNALIGN_N])
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.add_relu(tc, ta, tb)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pytest.mark.soc("950")
def test_add_relu_fp32():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32) * 10
    b = torch.randn(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32) * 10
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    _run(kernel_add_relu_fp32, a, b, out)
    torch.testing.assert_close(out, torch.relu(a + b), rtol=1e-5, atol=1e-5)


@pytest.mark.soc("950")
def test_add_relu_fp32_unaligned():
    if not _check_npu():
        return
    torch.manual_seed(42)
    a = torch.randn(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32) * 10
    b = torch.randn(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32) * 10
    out = torch.empty(TILE_M, UNALIGN_N, device=ST_DEVICE, dtype=torch.float32)
    _run(kernel_add_relu_fp32_unaligned, a, b, out)
    torch.testing.assert_close(out, torch.relu(a + b), rtol=1e-5, atol=1e-5)


# =============================================================================
# Tile placement contract rejections: wrong-usage kernels that trip the
# deduce-time memspace checks for the Vec / reduction tile contracts. The
# raise happens at parse (deduce) time during launch.
# =============================================================================

@pl.jit()
def kernel_reject_add_wrong_memspace(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    b: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    # block.add runs on the vector pipe over UB tiles: a Mat-space operand
    # must fail the deduce-time memspace check.
    tf_v = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tf_m = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat)
    ta = pl.make_tile_group(type=tf_m, addrs=[0], mutex_ids=[0]).next()
    tb = pl.make_tile(tf_v, addr=TILE_SIZE)
    tc = pl.make_tile(tf_v, addr=TILE_SIZE * 2)
    with pl.section_vector():
        pl.add(tc, ta, tb)


@pytest.mark.soc("950")
def test_reject_add_nonvec_memspace():
    if not _check_npu():
        return
    a = torch.zeros(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    b = torch.zeros(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    with pytest.raises(InvalidOperation, match="requires argument 1 to be a Vec"):
        _run(kernel_reject_add_wrong_memspace, a, b, out)


@pl.jit()
def kernel_reject_matmul_wrong_memspace(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    b: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    # block.matmul pins out=Acc, lhs=Left, rhs=Right: an out tile taken from a
    # Vec-space group must fail the tile-placement check.
    tf_v = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tf_l = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left)
    tf_r = pl.TileType(shape=[TILE_N, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right)
    to = pl.make_tile_group(type=tf_v, addrs=[0], mutex_ids=[0]).next()
    tl = pl.make_tile(tf_l, addr=TILE_SIZE * 2)
    tr = pl.make_tile(tf_r, addr=TILE_SIZE * 4)
    with pl.section_vector():
        pl.matmul(to, tl, tr)


@pytest.mark.soc("950")
def test_reject_matmul_wrong_memspace():
    if not _check_npu():
        return
    a = torch.zeros(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    b = torch.zeros(TILE_N, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    out = torch.empty(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    with pytest.raises(InvalidOperation, match="dst_tile must be in L0C"):
        _run(kernel_reject_matmul_wrong_memspace, a, b, out)


@pl.jit()
def kernel_reject_row_sum_vec(
    a: pl.Tensor[[DYN, DYN], pl.DT_FP32],
    out: pl.Tensor[[DYN, DYN], pl.DT_FP32],
):
    # The reduction families pin dst/src/tmp to UB tiles: an Acc-space src
    # must fail the deduce-time memspace check. (out/tmp keep [rows, 1]
    # shapes, whose RowMajor stride stays 32B-aligned at rows=16.)
    tf_v = pl.TileType(shape=[16, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tf_o = pl.TileType(shape=[16, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tf_acc = pl.TileType(shape=[16, 16], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc)
    to = pl.make_tile(tf_o, addr=0)
    tl = pl.make_tile_group(type=tf_acc, addrs=[TILE_SIZE], mutex_ids=[1]).next()
    tmp = pl.make_tile(tf_v, addr=TILE_SIZE * 2)
    with pl.section_vector():
        pl.sum(to, tl, tmp, dim=0)


@pytest.mark.soc("950")
def test_reject_row_sum_vec():
    if not _check_npu():
        return
    a = torch.zeros(16, TILE_N, device=ST_DEVICE, dtype=torch.float32)
    out = torch.empty(16, 1, device=ST_DEVICE, dtype=torch.float32)
    with pytest.raises(InvalidOperation, match="requires argument 1 to be a Vec"):
        _run(kernel_reject_row_sum_vec, a, out)
