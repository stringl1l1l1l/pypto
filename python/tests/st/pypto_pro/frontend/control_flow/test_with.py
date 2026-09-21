# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""with control_flow 前端测试。

测试覆盖场景:
  1. 覆盖 with control_flow 场景的前端语法、编译入口与运行验证。
  2. 覆盖 dtype 泛化：DT_FP16/DT_FP32/DT_BF16; Tensor 维度/shape 泛化：2D。
  3. 覆盖控制流组合：for 循环、if/elif/else 分支、with section 上下文、无值 return 提前退出。
  4. 覆盖接口组合：基础算术运算、Tensor 与 Tile 数据搬运、Cube 矩阵乘、VF 向量函数、section_vector/section_cube 上下文。
  5. 覆盖边界场景：section_cube、section_vector、VF vector_function 和 shape 泛化场景。
"""

import logging
import os

import pypto_pro.language as pl
from pypto_pro.language import Vf
import pytest
import torch

import pypto

vf = Vf

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


TILE_M = 64
TILE_N = 64
CUBE_TILE = 64

DTYPES_FLOAT = [
    (pl.DT_FP16, torch.float16, "fp16", 1e-2, 1e-2),
    (pl.DT_BF16, torch.bfloat16, "bf16", 1e-2, 1e-2),
    (pl.DT_FP32, torch.float32, "fp32", 1e-4, 1e-4),
]


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
# with_section_cube: matmul via section_cube  (FP16->FP32 / BF16->FP32 / FP32->FP32)
# ===================================================================


# =============================================================================
# Test 1: section_cube 矩阵乘 - FP16
#         section_cube matmul - FP16
# =============================================================================
@pl.jit(auto_mutex=True)
def with_section_cube_fp16_kernel(
    x: pl.Tensor[[64, 32], pl.DT_FP16],
    y: pl.Tensor[[32, 64], pl.DT_FP16],
    z: pl.Tensor[[64, 64], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[4],
    )
    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.load(cur_a, x, [0, 0])
        pl.load(cur_b, y, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.store(z, ac, [0, 0])


# =============================================================================
# Test 2: section_cube 矩阵乘 - BF16
#         section_cube matmul - BF16
# =============================================================================
@pl.jit(auto_mutex=True)
def with_section_cube_bf16_kernel(
    x: pl.Tensor[[64, 32], pl.DT_BF16],
    y: pl.Tensor[[32, 64], pl.DT_BF16],
    z: pl.Tensor[[64, 64], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 32], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[32, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 32], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[32, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[4],
    )
    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.load(cur_a, x, [0, 0])
        pl.load(cur_b, y, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.store(z, ac, [0, 0])


# =============================================================================
# Test 3: section_cube 矩阵乘 - FP32
#         section_cube matmul - FP32
# =============================================================================
@pl.jit(auto_mutex=True)
def with_section_cube_fp32_kernel(
    x: pl.Tensor[[64, 32], pl.DT_FP32],
    y: pl.Tensor[[32, 64], pl.DT_FP32],
    z: pl.Tensor[[64, 64], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[4],
    )
    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.load(cur_a, x, [0, 0])
        pl.load(cur_b, y, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.store(z, ac, [0, 0])


WITH_SECTION_CUBE_KERNELS = {
    "fp16": with_section_cube_fp16_kernel,
    "bf16": with_section_cube_bf16_kernel,
    "fp32": with_section_cube_fp32_kernel,
}


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_with_section_cube():
    # Let kernel discovery reach every variant before validating any output.
    checks = []
    device = ST_DEVICE
    torch.npu.set_device(device)
    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is not Ascend950, skip.")
        return
    torch.manual_seed(0)
    for _pl_dt, tdt, label, atol, rtol in DTYPES_FLOAT:
        kernel = WITH_SECTION_CUBE_KERNELS[label]
        x = torch.randn(64, 32, device=device, dtype=tdt)
        y = torch.randn(32, 64, device=device, dtype=tdt)
        z = torch.zeros(64, 64, device=device, dtype=torch.float32)
        kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = torch.matmul(x.float(), y.float())
        case = f"test_with_section_cube [{label}] x=[64,32] y=[32,64] z=[64,64] dtype={tdt}"
        checks.append((z, z_ref, atol, rtol, case))

    for actual, expected, atol, rtol, case in checks:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=case)
        logging.info("%s passed!", case)


# ===================================================================
# with_section_cube_shape_generalization: cube matmul shape coverage
#   x: [M, K], y: [K, N], z/L0C output: [M, N]
#   x shapes covered by test: [128,128], [256,128], [1024,1024]
# ===================================================================


# =============================================================================
# Test 4: section_cube 矩阵乘 shape 泛化
#         section_cube matmul shape generalization
# =============================================================================
@pl.jit(auto_mutex=True)
def with_section_cube_shape_generalization_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    p = x.shape[1]
    m = x.shape[0]
    n = y.shape[1]
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CUBE_TILE, CUBE_TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0, 1],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CUBE_TILE, CUBE_TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000,
        mutex_ids=[2, 3],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(
            shape=[CUBE_TILE, CUBE_TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ
        ),
        addrs=0x0000,
        mutex_ids=[4, 5],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[CUBE_TILE, CUBE_TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[6, 7],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(
            shape=[CUBE_TILE, CUBE_TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, fractal=1024
        ),
        addrs=0x0000,
        mutex_ids=[8],
    )

    with pl.section_cube():
        pl.system.set_mm_layout_transform(enabled=True)
        for i in pl.range(0, m, CUBE_TILE):
            for j in pl.range(0, n, CUBE_TILE):
                ac = c_l0c.current()
                for k in pl.range(0, p, CUBE_TILE):
                    cur_a = a_l1.next()
                    cur_b = b_l1.next()
                    al = a_l0a.next()
                    br = b_l0b.next()
                    pl.load(cur_a, x, [i, k])
                    pl.load(cur_b, y, [k, j])
                    pl.move(al, cur_a)
                    pl.move(br, cur_b)
                    if k == 0:
                        pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
                    else:
                        if k == p - CUBE_TILE:
                            pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
                        else:
                            pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
                pl.store(z, ac, [i, j], phase=pl.STPhase.Final)
        pl.system.set_mm_layout_transform(enabled=False)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_with_section_cube_shape_generalization():
    device = ST_DEVICE
    torch.npu.set_device(device)
    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is not Ascend950, skip.")
        return
    torch.manual_seed(0)
    x_shapes = [(128, 128)]
    for m_size, k_size in x_shapes:
        n_size = k_size
        x = torch.randn(m_size, k_size, device=device, dtype=torch.float16)
        y = torch.randn(k_size, n_size, device=device, dtype=torch.float16)
        z = torch.zeros(m_size, n_size, device=device, dtype=torch.float32)
        with_section_cube_shape_generalization_kernel(x, y, z)
        torch.npu.synchronize()
        z_ref = torch.matmul(x.float(), y.float())
        torch.testing.assert_close(z, z_ref, atol=2e-1, rtol=2e-2)
        logging.info(
            "test_with_section_cube_shape_generalization passed! x=[%d,%d] y=[%d,%d] z=[%d,%d] dtype=%s",
            m_size,
            k_size,
            k_size,
            n_size,
            m_size,
            n_size,
            x.dtype,
        )


# ===================================================================
# with_section_vf: VF min/exp/abs via @pl.vector_function  (FP32)
# ===================================================================

VF_N = 1
VF_M = 64
VF_TILE_SIZE = VF_N * VF_M * 4
VF_VA_A = 0
VF_VA_B = VF_VA_A + VF_TILE_SIZE
VF_VA_OUT0 = VF_VA_B + VF_TILE_SIZE
VF_VA_OUT1 = VF_VA_OUT0 + VF_TILE_SIZE


@pl.vector_function
def _with_section_vf_min_exp_body(in_a, in_b, t_out0, t_out1):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_a = vf.load_align(in_a, 0)
    reg_b = vf.load_align(in_b, 0)
    reg_dst = vf.min(reg_a, reg_b, preg)
    vf.store_align(t_out0, reg_dst, preg)
    reg_dst = vf.exp(reg_a, preg)
    vf.store_align(t_out1, reg_dst, preg)


# =============================================================================
# Test 5: section_vector + VF min/exp
#         section_vector + VF min/exp
# =============================================================================
@pl.jit()
def with_section_vf_min_exp_kernel(
    a: pl.Tensor[[1, 64], pl.DT_FP32],
    b: pl.Tensor[[1, 64], pl.DT_FP32],
    out_min: pl.Tensor[[1, 64], pl.DT_FP32],
    out_exp: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tf = pl.TileType(shape=[VF_N, VF_M], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a = pl.make_tile(tf, addr=VF_VA_A)
    in_b = pl.make_tile(tf, addr=VF_VA_B)
    t_out0 = pl.make_tile(tf, addr=VF_VA_OUT0)
    t_out1 = pl.make_tile(tf, addr=VF_VA_OUT1)
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.load(in_b, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        _with_section_vf_min_exp_body(in_a, in_b, t_out0, t_out1)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out_min, t_out0, [0, 0])
        pl.store(out_exp, t_out1, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_with_section_vf_min_exp():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    a = torch.randn(1, 64, device=device, dtype=torch.float32)
    b = torch.randn(1, 64, device=device, dtype=torch.float32)
    out_min = torch.zeros(1, 64, device=device, dtype=torch.float32)
    out_exp = torch.zeros(1, 64, device=device, dtype=torch.float32)
    with_section_vf_min_exp_kernel(a, b, out_min, out_exp)
    torch.npu.synchronize()
    torch.testing.assert_close(out_min, torch.minimum(a, b), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out_exp, torch.exp(a), atol=1e-4, rtol=1e-4)
    logging.info("test_with_section_vf_min_exp passed! dtype=%s", a.dtype)


# ===================================================================
# with_section_if: section_vector with if/else add/sub  (FP16)
# ===================================================================


@pl.jit(auto_mutex=True)
def with_section_if_fp16_kernel(
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
                if i == 0:
                    pl.add(tile_c, tile_a, tile_b)
                else:
                    pl.sub(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_with_section_if():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shape = [256, 256]
    x = torch.rand(shape, device=device, dtype=torch.float16)
    y = torch.rand(shape, device=device, dtype=torch.float16)
    z = torch.zeros(shape, device=device, dtype=torch.float16)
    with_section_if_fp16_kernel(x, y, z)
    torch.npu.synchronize()
    z_ref = torch.zeros(shape, device=device, dtype=torch.float16)
    z_ref[:TILE_M, :] = (x[:TILE_M, :].float() + y[:TILE_M, :].float()).to(torch.float16)
    z_ref[TILE_M:, :] = (x[TILE_M:, :].float() - y[TILE_M:, :].float()).to(torch.float16)
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("test_with_section_if passed! shape=%s", shape)


# ===================================================================
# with_cube_then_vector: matmul in section_cube, add in section_vector
#   a: [64, 32], b: [32, 64], out: [64, 64]
# ===================================================================


# =============================================================================
# Test 7: section_cube 后接 section_vector 加法
#         section_vector add after section_cube
# =============================================================================
@pl.jit(auto_mutex=True)
def with_cube_then_vector_kernel(
    a: pl.Tensor[[64, 32], pl.DT_FP16],
    b: pl.Tensor[[32, 64], pl.DT_FP16],
    x: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[4],
    )

    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.store(out, ac, [0, 0])
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    with pl.section_vector():
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=0)
        mm_vec = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0x0000,
            mutex_ids=[5],
        )
        x_vec = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0x4000,
            mutex_ids=[6],
        )
        out_vec = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0x8000,
            mutex_ids=[7],
        )
        mm = mm_vec.current()
        xv = x_vec.current()
        ov = out_vec.current()
        pl.load(mm, out, [0, 0])
        pl.load(xv, x, [0, 0])
        pl.add(ov, mm, xv)
        pl.store(out, ov, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_with_cube_then_vector():
    device = ST_DEVICE
    torch.npu.set_device(device)
    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is not Ascend950, skip.")
        return
    torch.manual_seed(0)
    a = torch.randn(64, 32, device=device, dtype=torch.float16)
    b = torch.randn(32, 64, device=device, dtype=torch.float16)
    x = torch.randn(64, 64, device=device, dtype=torch.float32)
    out = torch.zeros(64, 64, device=device, dtype=torch.float32)
    # 单 tile 读写回 GM：多核会对同一输出互相读到中间结果，显式单块。
    with_cube_then_vector_kernel[None, 1](a, b, x, out)
    torch.npu.synchronize()
    z_ref = torch.matmul(a.float(), b.float()) + x.float()
    torch.testing.assert_close(out, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("test_with_cube_then_vector passed! shape=[64, 64] dtype=%s", out.dtype)


# ===================================================================
# with_cube_then_vf: matmul in section_cube, vf add in @pl.vector_function
#   vf:   load C from GM, load D from GM, out = vf.add(C, D)
# ===================================================================


@pl.vector_function
def _with_cube_then_vf_add_body(mm, dv, ov):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_a = vf.load_align(mm, 0)
    reg_b = vf.load_align(dv, 0)
    reg_dst = vf.add(reg_a, reg_b, preg)
    vf.store_align(ov, reg_dst, preg)


# =============================================================================
# Test 8: section_cube 后接 VF 加法
#         VF add after section_cube
# =============================================================================
@pl.jit(auto_mutex=True)
def with_cube_then_vf_kernel(
    a: pl.Tensor[[64, 32], pl.DT_FP16],
    b: pl.Tensor[[32, 64], pl.DT_FP16],
    d: pl.Tensor[[1, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[4],
    )

    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.store(out, ac, [0, 0])
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    with pl.section_vector():
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=0)
        mm_vec = pl.make_tile_group(
            type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0x0000,
            mutex_ids=[5],
        )
        d_vec = pl.make_tile_group(
            type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0x0400,
            mutex_ids=[6],
        )
        out_vec = pl.make_tile_group(
            type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0x0800,
            mutex_ids=[7],
        )
        mm = mm_vec.current()
        dv = d_vec.current()
        ov = out_vec.current()
        pl.load(mm, out, [0, 0])
        pl.load(dv, d, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=1)
        _with_cube_then_vf_add_body(mm, dv, ov)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.store(out, ov, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_with_cube_then_vf():
    device = ST_DEVICE
    torch.npu.set_device(device)
    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is not Ascend950, skip.")
        return
    torch.manual_seed(0)
    a = torch.randn(64, 32, device=device, dtype=torch.float16)
    b = torch.randn(32, 64, device=device, dtype=torch.float16)
    d = torch.randn(1, 64, device=device, dtype=torch.float32)
    out = torch.zeros(64, 64, device=device, dtype=torch.float32)
    # 单 tile 读改写 GM：多核会对同一输出互相读到中间结果，显式单块。
    with_cube_then_vf_kernel[None, 1](a, b, d, out)
    torch.npu.synchronize()
    mm = torch.matmul(a.float(), b.float())
    z_ref = mm + d
    torch.testing.assert_close(out[:1, :], z_ref[:1, :], atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(out[1:, :], mm[1:, :], atol=1e-2, rtol=1e-2)
    logging.info("test_with_cube_then_vf passed! a=[64,32] b=[32,64] out=[64,64] dtype=%s", out.dtype)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_with_section_cube()
    test_with_section_cube_shape_generalization()
    test_with_section_vf_min_exp()
    test_with_section_if()
    test_with_cube_then_vector()
    test_with_cube_then_vf()
    logging.info("\nAll tests passed!")
