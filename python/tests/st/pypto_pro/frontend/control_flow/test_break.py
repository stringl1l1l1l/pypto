# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""break control_flow 前端测试。

测试覆盖场景:
  1. 覆盖 break control_flow 场景的前端语法、编译入口与运行验证。
  2. 覆盖 dtype 泛化：DT_FP16/DT_BF16/DT_FP32/DT_INT32; Tensor 维度/shape 泛化：2D。
  3. 覆盖控制流组合：for 循环、while 循环、if/elif/else 分支、with section 上下文、break 跳转、无值 return 提前退出。
  4. 覆盖接口组合：基础算术运算、激活/比较类运算、Tensor 与 Tile 数据搬运、section_vector/section_cube 上下文。
  5. 覆盖边界场景：for/while 内的首轮 break、条件 break、多层嵌套 break 和非对齐尾块。
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


TILE_M = 64
TILE_N = 64

DTYPES_ALL = [
    (pl.DT_FP16, torch.float16, "fp16", 1e-2, 1e-2),
    (pl.DT_BF16, torch.bfloat16, "bf16", 1e-2, 1e-2),
    (pl.DT_FP32, torch.float32, "fp32", 1e-4, 1e-4),
    (pl.DT_INT32, torch.int32, "int32", 0, 0),
]
DTYPES_FP16 = [DTYPES_ALL[0]]


def _gen(shape, tdt, device):
    if tdt == torch.int32:
        x = torch.randint(-100, 100, shape, device=device, dtype=tdt)
        y = torch.randint(-100, 100, shape, device=device, dtype=tdt)
    else:
        x = torch.rand(shape, device=device, dtype=tdt)
        y = torch.rand(shape, device=device, dtype=tdt)
    z = torch.zeros(shape, device=device, dtype=tdt)
    return x, y, z


def _ref(tdt, fn, x, y):
    if tdt == torch.int32:
        return fn(x, y)
    return fn(x.float(), y.float()).to(tdt)


# ===================================================================
# for_break: for-loop add with break (only first tile col)  (FP16/BF16/FP32/INT32)
# ===================================================================


# =============================================================================
# Test 1: for 循环 break 中断 - FP16
#         for-loop break - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def for_break_fp16_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, m // TILE_M, 1):
            for j in pl.range(0, n // TILE_N, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])
                break


# =============================================================================
# Test 2: for 循环 break 中断 - BF16
#         for-loop break - BF16
# =============================================================================
@pl.jit(auto_mutex=True)
def for_break_bf16_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, m // TILE_M, 1):
            for j in pl.range(0, n // TILE_N, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])
                break


# =============================================================================
# Test 3: for 循环 break 中断 - FP32
#         for-loop break - FP32
# =============================================================================
@pl.jit(auto_mutex=True)
def for_break_fp32_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, m // TILE_M, 1):
            for j in pl.range(0, n // TILE_N, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])
                break


# =============================================================================
# Test 4: for 循环 break 中断 - INT32
#         for-loop break - INT32
# =============================================================================
@pl.jit(auto_mutex=True)
def for_break_int32_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, m // TILE_M, 1):
            for j in pl.range(0, n // TILE_N, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])
                break


FOR_BREAK_KERNELS = {
    "fp16": for_break_fp16_kernel,
    "bf16": for_break_bf16_kernel,
    "fp32": for_break_fp32_kernel,
    "int32": for_break_int32_kernel,
}


# ===================================================================
# while_break: while-loop add with break (only first tile col)  (FP16)
# ===================================================================


# =============================================================================
# Test 5: while 循环 break 中断 - FP16
#         while-loop break - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def while_break_fp16_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        i: pl.DT_INT64 = 0
        while i < m // TILE_M:
            j: pl.DT_INT64 = 0
            while j < n // TILE_N:
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])
                break
            i = i + 1


WHILE_BREAK_KERNELS = {
    "fp16": while_break_fp16_kernel,
}


# ===================================================================
# for_if_break: nested for-loop with if(i >= threshold) break  (FP16)
# ===================================================================


# =============================================================================
# Test 6: for + if 内层 break
#         for + if inner break
# =============================================================================
@pl.jit(auto_mutex=True)
def for_if_break_inner_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, m // TILE_M, 1):
            for j in pl.range(0, n // TILE_N, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])
                if j >= 1:
                    break


# ===================================================================
# while_if_break: nested while-loop with if(i >= threshold) break  (FP16)
# ===================================================================


# =============================================================================
# Test 7: while + if 内层 break
#         while + if inner break
# =============================================================================
@pl.jit(auto_mutex=True)
def while_if_break_inner_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        i: pl.DT_INT64 = 0
        while i < m // TILE_M:
            j: pl.DT_INT64 = 0
            while j < n // TILE_N:
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])
                if j >= 1:
                    break
                j = j + 1
            i = i + 1


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_for_break():
    # Let kernel discovery reach every variant before validating any output.
    checks = []
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[128, 128], [256, 128], [128, 256], [256, 256]]
    for shape in shapes:
        for _pl_dt, tdt, label, atol, rtol in DTYPES_ALL:
            kernel = FOR_BREAK_KERNELS[label]
            x, y, z = _gen(shape, tdt, device)
            kernel(x, y, z)
            torch.npu.synchronize()
            z_ref = torch.zeros(shape, device=device, dtype=tdt)
            z_ref[:, :TILE_N] = _ref(tdt, lambda a, b: a + b, x[:, :TILE_N], y[:, :TILE_N])
            checks.append((z, z_ref, atol, rtol, 'test_for_break [%s] shape=%s' % (label, shape)))

    for actual, expected, atol, rtol, case in checks:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=case)
        logging.info("%s passed!", case)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_while_break():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[256, 256]]
    for shape in shapes:
        for _pl_dt, tdt, label, atol, rtol in DTYPES_FP16:
            kernel = WHILE_BREAK_KERNELS[label]
            x, y, z = _gen(shape, tdt, device)
            kernel(x, y, z)
            torch.npu.synchronize()
            z_ref = torch.zeros(shape, device=device, dtype=tdt)
            z_ref[:, :TILE_N] = _ref(tdt, lambda a, b: a + b, x[:, :TILE_N], y[:, :TILE_N])
            torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
            logging.info("test_while_break [%s] passed! shape=%s", label, shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_for_if_break_inner():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[256, 256]]
    for shape in shapes:
        x = torch.rand(shape, device=device, dtype=torch.float16)
        y = torch.rand(shape, device=device, dtype=torch.float16)
        z = torch.zeros(shape, device=device, dtype=torch.float16)
        for_if_break_inner_kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = torch.zeros(shape, device=device, dtype=torch.float16)
        z_ref[:, :TILE_N * 2] = (x[:, :TILE_N * 2].float() + y[:, :TILE_N * 2].float()).to(torch.float16)
        torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
        logging.info("test_for_if_break_inner passed! shape=%s", shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_while_if_break_inner():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[256, 256]]
    for shape in shapes:
        x = torch.rand(shape, device=device, dtype=torch.float16)
        y = torch.rand(shape, device=device, dtype=torch.float16)
        z = torch.zeros(shape, device=device, dtype=torch.float16)
        while_if_break_inner_kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = torch.zeros(shape, device=device, dtype=torch.float16)
        z_ref[:, :TILE_N * 2] = (x[:, :TILE_N * 2].float() + y[:, :TILE_N * 2].float()).to(torch.float16)
        torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
        logging.info("test_while_if_break_inner passed! shape=%s", shape)


# ===================================================================
# for_while_8layer_break: 8-layer alternating for/while with break
#   L1: for i0 in range(0, 1)          — break if i0 >= 0 (always)
#   L2:   while i1 < M // TILE_M       — break if i1 >= 1
#   L3:     for i2 in range(0, 1)      — break if i2 >= 0 (always)
#   L4:      while i3 < N // TILE_N    — break if i3 >= 1
#   L5:        for i4 in range(0, 1)   — break if i4 >= 0 (always)
#   L6:          while i5 < 1          — break if i5 >= 0 (always)
#   L7:            for i6 in range(0, 1) — break if i6 >= 0 (always)
#   L8:              while i7 < 1      — break if i7 >= 0 (always)
# ===================================================================


# =============================================================================
# Test 8: for/while 8 层嵌套 break
#         8-layer nested for/while break
# =============================================================================
@pl.jit(auto_mutex=True)
def for_while_8layer_break_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i0 in pl.range(0, 1, 1):
            i1: pl.DT_INT64 = 0
            while i1 < m // TILE_M:
                for i2 in pl.range(0, 1, 1):
                    i3: pl.DT_INT64 = 0
                    while i3 < n // TILE_N:
                        for i4 in pl.range(0, 1, 1):
                            i5: pl.DT_INT64 = 0
                            while i5 < 1:
                                for i6 in pl.range(0, 1, 1):
                                    i7: pl.DT_INT64 = 0
                                    while i7 < 1:
                                        tile_a = a_db.next()
                                        tile_b = b_db.next()
                                        tile_c = c_db.next()
                                        pl.load_tile(tile_a, x, [i1, i3])
                                        pl.load_tile(tile_b, y, [i1, i3])
                                        pl.add(tile_c, tile_a, tile_b)
                                        pl.store_tile(z, tile_c, [i1, i3])
                                        if i7 >= 0:
                                            break
                                        i7 = i7 + 1
                                    if i6 >= 0:
                                        break
                                if i5 >= 0:
                                    break
                                i5 = i5 + 1
                            if i4 >= 0:
                                break
                        if i3 >= 1:
                            break
                        i3 = i3 + 1
                    if i2 >= 0:
                        break
                if i1 >= 1:
                    break
                i1 = i1 + 1
            if i0 >= 0:
                break


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_for_while_8layer_break():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[256, 256]]
    for shape in shapes:
        x = torch.rand(shape, device=device, dtype=torch.float16)
        y = torch.rand(shape, device=device, dtype=torch.float16)
        z = torch.zeros(shape, device=device, dtype=torch.float16)
        for_while_8layer_break_kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = torch.zeros(shape, device=device, dtype=torch.float16)
        x_f = x[:TILE_M * 2, :TILE_N * 2].float()
        y_f = y[:TILE_M * 2, :TILE_N * 2].float()
        z_ref[:TILE_M * 2, :TILE_N * 2] = (x_f + y_f).to(torch.float16)
        torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
        logging.info("test_for_while_8layer_break passed! shape=%s", shape)


# ===================================================================
# for_break_first: 2-layer for, break at FIRST line of inner body
#   outer pre-op: store x to z[i, 0] (executes every outer iter)
#   inner: break immediately, add/store never execute
# ===================================================================


# =============================================================================
# Test 9: for 首轮 break
#         for-loop first-iteration break
# =============================================================================
@pl.jit(auto_mutex=True)
def for_break_first_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, m // TILE_M, 1):
            tile_pre = c_db.next()
            pl.load_tile(tile_pre, x, [i, 0])
            pl.store_tile(z, tile_pre, [i, 0])
            for j in pl.range(0, n // TILE_N, 1):
                break
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])
            if i >= 1:
                break


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_for_break_first():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[256, 256]]
    for shape in shapes:
        x = torch.rand(shape, device=device, dtype=torch.float16)
        y = torch.rand(shape, device=device, dtype=torch.float16)
        z = torch.zeros(shape, device=device, dtype=torch.float16)
        for_break_first_kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = torch.zeros(shape, device=device, dtype=torch.float16)
        z_ref[:TILE_M * 2, :TILE_N] = x[:TILE_M * 2, :TILE_N].to(torch.float16)
        torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
        logging.info("test_for_break_first passed! shape=%s", shape)


# ===================================================================
# for_while_if_not_break_continue: break/continue with not conditions
# ===================================================================


@pl.jit(auto_mutex=True)
def for_while_if_not_break_continue_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, m // TILE_M, 1):
            j: pl.DT_INT64 = 0
            while j < n // TILE_N:
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.system.bar_mte2()
                pl.load_tile(tile_b, y, [i, j])
                pl.system.bar_mte2()
                if not j >= 1 and not i >= 1:
                    pl.add(tile_c, tile_a, tile_b)
                    pl.store_tile(z, tile_c, [i, j])
                elif not j >= 1 or i >= 1:
                    pl.sub(tile_c, tile_a, tile_b)
                    pl.store_tile(z, tile_c, [i, j])
                    j = j + 1
                    continue
                else:
                    break
                j = j + 1


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_for_while_if_not_break_continue():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [256, 128]
    x = torch.rand(shape, device=device, dtype=torch.float16)
    y = torch.rand(shape, device=device, dtype=torch.float16)
    z = torch.zeros(shape, device=device, dtype=torch.float16)
    for_while_if_not_break_continue_kernel(x, y, z)
    torch.npu.synchronize()
    z_ref = torch.zeros(shape, device=device, dtype=torch.float16)
    x_f = x.float()
    y_f = y.float()
    z_ref[:TILE_M, :TILE_N] = (x_f[:TILE_M, :TILE_N] + y_f[:TILE_M, :TILE_N]).to(torch.float16)
    z_ref[TILE_M:, :2 * TILE_N] = (x_f[TILE_M:, :2 * TILE_N] - y_f[TILE_M:, :2 * TILE_N]).to(torch.float16)
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("test_for_while_if_not_break_continue passed! shape=%s", shape)


# =============================================================================
# Test 10: break 非对齐尾块 - FP16
#         break with unaligned tail block - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def break_unaligned_fp16_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m = x.shape[0]
    n = x.shape[1]
    tile_type = pl.TileType(
        shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
    )
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, m, TILE_M):
            for j in pl.range(0, n, TILE_N):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                valid_r = pl.min(m - i, TILE_M)
                valid_c = pl.min(n - j, TILE_N)
                pl.set_validshape(tile_a, [valid_r, valid_c])
                pl.set_validshape(tile_b, [valid_r, valid_c])
                pl.set_validshape(tile_c, [valid_r, valid_c])
                pl.load(tile_a, x, [i, j])
                pl.load(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store(z, tile_c, [i, j])
                break


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_break_unaligned_shape():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[97, 65]]
    for shape in shapes:
        x = torch.rand(shape, device=device, dtype=torch.float16)
        y = torch.rand(shape, device=device, dtype=torch.float16)
        z = torch.zeros(shape, device=device, dtype=torch.float16)
        break_unaligned_fp16_kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = torch.zeros(shape, device=device, dtype=torch.float16)
        z_ref[:, :TILE_N] = (x[:, :TILE_N].float() + y[:, :TILE_N].float()).to(torch.float16)
        torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
        logging.info("test_break_unaligned_shape passed! shape=%s", shape)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_for_break()
    test_while_break()
    test_for_if_break_inner()
    test_while_if_break_inner()
    test_for_while_8layer_break()
    test_for_break_first()
    test_for_while_if_not_break_continue()
    test_break_unaligned_shape()
    logging.info("\nAll tests passed!")
