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
"""
Cast+Matmul 融合算子 ST 测试脚本。
场景：先 Cast 输入到目标 dtype，再执行 Matmul；另含 Cast+ScaledMM(MX) 的 UB2L1 场景。
支持 2D/3D/4D 输入，3D/4D 为 BatchMatmul 场景（batch 维切片 + 逐 batch Cast）。
支持 pytest 参数化执行和直接执行两种模式。
"""

import os

import pytest
from testcase.matmul_ub2l1_test_case import (
    CAST_3D_MATMUL_TESTS,
    CAST_4D_MATMUL_TESTS,
    CAST_BOTH_MATMUL_TESTS,
    CAST_LEFT_MATMUL_TESTS,
    CAST_RIGHT_MATMUL_TESTS,
    SCALED_MM_UB2L1_TESTS,
    CastMatmulConfig,
    ScaledMmUb2L1Config,
)
import torch
import torch.nn.functional as functional

import pypto

K_BLOCK_SIZE_64 = 64
SCALE_INNER_DIM = 2


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def cast_matmul_pto_kernel(
    a_tensor: pypto.Tensor(),
    b_tensor: pypto.Tensor(),
    out_tensor: pypto.Tensor(),
    config: CastMatmulConfig,
):
    m, k, n = config.shape
    m_view, n_view = config.view_shape

    pypto.set_cube_tile_shapes(*config.cube_tile_shape, config.enable_ksplit)

    m_loop = (m + m_view - 1) // m_view
    n_loop = (n + n_view - 1) // n_view
    # 当设置scope大于5000，即5001以上时，开启mix场景，走入UB2L1
    pypto.set_pass_options(sg_set_scope=10000)
    for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_L0_mIdx", idx_name="m_idx"):
        for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L0_nIdx", idx_name="n_idx"):
            mode = pypto.CastMode.CAST_NONE
            if config.matmul_pto_dtype == pypto.DT_INT8:
                mode = pypto.CastMode.CAST_TRUNC
            m_offset = m_idx * m_view
            n_offset = n_idx * n_view
            if config.a_trans:
                a_tile = pypto.view(
                    a_tensor, [k, m_view], [0, m_offset], valid_shape=[k, (m - m_offset).min(m_view)]
                )
            else:
                a_tile = pypto.view(
                    a_tensor, [m_view, k], [m_offset, 0], valid_shape=[(m - m_offset).min(m_view), k]
                )

            if config.a_cast:
                pypto.set_vec_tile_shapes(*config.a_vec_tile_shape)
                a_compute = pypto.cast(a_tile, config.matmul_pto_dtype, mode)
            else:
                a_compute = a_tile

            if config.b_trans:
                b_tile = pypto.view(
                    b_tensor, [n_view, k], [n_offset, 0], valid_shape=[(n - n_offset).min(n_view), k]
                )
            else:
                b_tile = pypto.view(
                    b_tensor, [k, n_view], [0, n_offset], valid_shape=[k, (n - n_offset).min(n_view)]
                )

            if config.b_cast:
                pypto.set_vec_tile_shapes(*config.b_vec_tile_shape)
                b_compute = pypto.cast(b_tile, config.matmul_pto_dtype, mode)
            else:
                b_compute = b_tile

            out_view = pypto.matmul(
                a_compute,
                b_compute,
                out_dtype=config.out_pto_dtype,
                a_trans=config.a_trans,
                b_trans=config.b_trans,
            )

            out_tensor[
                m_offset:m_offset + m_view,
                n_offset:n_offset + n_view,
            ] = out_view
    # 运行完后设置回-1，关闭mix
    pypto.set_pass_options(sg_set_scope=-1)


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def cast_matmul_pto_kernel_3d(
    a_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
    b_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
    out_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
    config: CastMatmulConfig,
):
    b_size = config.batch_shape[0]
    m, k, n = config.shape
    b_view = config.batch_view_shape[0]
    m_view, n_view = config.view_shape

    pypto.set_cube_tile_shapes(*config.cube_tile_shape, config.enable_ksplit)

    b_loop = (b_size + b_view - 1) // b_view
    m_loop = (m + m_view - 1) // m_view
    n_loop = (n + n_view - 1) // n_view
    # 当设置scope大于5000，即5001以上时，开启mix场景，走入UB2L1
    pypto.set_pass_options(sg_set_scope=10000)
    for b_idx in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx", idx_name="b_idx"):
        for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_L0_mIdx", idx_name="m_idx"):
            for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L1_nIdx", idx_name="n_idx"):
                mode = pypto.CastMode.CAST_NONE
                if config.matmul_pto_dtype == pypto.DT_INT8:
                    mode = pypto.CastMode.CAST_TRUNC
                b_offset = b_idx * b_view
                m_offset = m_idx * m_view
                n_offset = n_idx * n_view
                if config.a_trans:
                    a_tile = pypto.view(
                        a_tensor,
                        [b_view, k, m_view],
                        [b_offset, 0, m_offset],
                        valid_shape=[(b_size - b_offset).min(b_view), k, (m - m_offset).min(m_view)],
                    )
                else:
                    a_tile = pypto.view(
                        a_tensor,
                        [b_view, m_view, k],
                        [b_offset, m_offset, 0],
                        valid_shape=[(b_size - b_offset).min(b_view), (m - m_offset).min(m_view), k],
                    )

                if config.a_cast:
                    pypto.set_vec_tile_shapes(*config.a_vec_tile_shape)
                    a_compute = pypto.cast(a_tile, config.matmul_pto_dtype, mode)
                else:
                    a_compute = a_tile

                if config.b_trans:
                    b_tile = pypto.view(
                        b_tensor,
                        [b_view, n_view, k],
                        [b_offset, n_offset, 0],
                        valid_shape=[(b_size - b_offset).min(b_view), (n - n_offset).min(n_view), k],
                    )
                else:
                    b_tile = pypto.view(
                        b_tensor,
                        [b_view, k, n_view],
                        [b_offset, 0, n_offset],
                        valid_shape=[(b_size - b_offset).min(b_view), k, (n - n_offset).min(n_view)],
                    )

                if config.b_cast:
                    pypto.set_vec_tile_shapes(*config.b_vec_tile_shape)
                    b_compute = pypto.cast(b_tile, config.matmul_pto_dtype, mode)
                else:
                    b_compute = b_tile

                out_view = pypto.matmul(
                    a_compute,
                    b_compute,
                    out_dtype=config.out_pto_dtype,
                    a_trans=config.a_trans,
                    b_trans=config.b_trans,
                )

                out_tensor[
                    b_offset:b_offset + b_view,
                    m_offset:m_offset + m_view,
                    n_offset:n_offset + n_view,
                ] = out_view
    # 运行完后设置回-1，关闭mix
    pypto.set_pass_options(sg_set_scope=-1)


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def cast_matmul_pto_kernel_4d(
    a_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    b_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    out_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    config: CastMatmulConfig,
):
    b0_size, b1_size = config.batch_shape
    m, k, n = config.shape
    b0_view, b1_view = config.batch_view_shape
    m_view, n_view = config.view_shape

    pypto.set_cube_tile_shapes(*config.cube_tile_shape, config.enable_ksplit)

    b0_loop = (b0_size + b0_view - 1) // b0_view
    b1_loop = (b1_size + b1_view - 1) // b1_view
    m_loop = (m + m_view - 1) // m_view
    n_loop = (n + n_view - 1) // n_view
    # 当设置scope大于5000，即5001以上时，开启mix场景，走入UB2L1
    pypto.set_pass_options(sg_set_scope=10000)
    for b0_idx in pypto.loop(0, b0_loop, 1, name="LOOP_L0_b0Idx", idx_name="b0_idx"):
        for b1_idx in pypto.loop(0, b1_loop, 1, name="LOOP_L0_b1Idx", idx_name="b1_idx"):
            for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_L0_mIdx", idx_name="m_idx"):
                for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L1_nIdx", idx_name="n_idx"):
                    mode = pypto.CastMode.CAST_NONE
                    if config.matmul_pto_dtype == pypto.DT_INT8:
                        mode = pypto.CastMode.CAST_TRUNC
                    b0_offset = b0_idx * b0_view
                    b1_offset = b1_idx * b1_view
                    m_offset = m_idx * m_view
                    n_offset = n_idx * n_view
                    if config.a_trans:
                        a_tile = pypto.view(
                            a_tensor,
                            [b0_view, b1_view, k, m_view],
                            [b0_offset, b1_offset, 0, m_offset],
                            valid_shape=[
                                (b0_size - b0_offset).min(b0_view),
                                (b1_size - b1_offset).min(b1_view),
                                k,
                                (m - m_offset).min(m_view),
                            ],
                        )
                    else:
                        a_tile = pypto.view(
                            a_tensor,
                            [b0_view, b1_view, m_view, k],
                            [b0_offset, b1_offset, m_offset, 0],
                            valid_shape=[
                                (b0_size - b0_offset).min(b0_view),
                                (b1_size - b1_offset).min(b1_view),
                                (m - m_offset).min(m_view),
                                k,
                            ],
                        )

                    if config.a_cast:
                        pypto.set_vec_tile_shapes(*config.a_vec_tile_shape)
                        a_compute = pypto.cast(a_tile, config.matmul_pto_dtype, mode)
                    else:
                        a_compute = a_tile

                    if config.b_trans:
                        b_tile = pypto.view(
                            b_tensor,
                            [b0_view, b1_view, n_view, k],
                            [b0_offset, b1_offset, n_offset, 0],
                            valid_shape=[
                                (b0_size - b0_offset).min(b0_view),
                                (b1_size - b1_offset).min(b1_view),
                                (n - n_offset).min(n_view),
                                k,
                            ],
                        )
                    else:
                        b_tile = pypto.view(
                            b_tensor,
                            [b0_view, b1_view, k, n_view],
                            [b0_offset, b1_offset, 0, n_offset],
                            valid_shape=[
                                (b0_size - b0_offset).min(b0_view),
                                (b1_size - b1_offset).min(b1_view),
                                k,
                                (n - n_offset).min(n_view),
                            ],
                        )

                    if config.b_cast:
                        pypto.set_vec_tile_shapes(*config.b_vec_tile_shape)
                        b_compute = pypto.cast(b_tile, config.matmul_pto_dtype, mode)
                    else:
                        b_compute = b_tile

                    out_view = pypto.matmul(
                        a_compute,
                        b_compute,
                        out_dtype=config.out_pto_dtype,
                        a_trans=config.a_trans,
                        b_trans=config.b_trans,
                    )

                    out_tensor[
                        b0_offset:b0_offset + b0_view,
                        b1_offset:b1_offset + b1_view,
                        m_offset:m_offset + m_view,
                        n_offset:n_offset + n_view,
                    ] = out_view
    # 运行完后设置回-1，关闭mix
    pypto.set_pass_options(sg_set_scope=-1)


def run_cast_matmul_test(case: dict):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    config = CastMatmulConfig.from_test_case(case)

    m, k, n = config.shape
    a_shape = [k, m] if config.a_trans else [m, k]
    b_shape = [n, k] if config.b_trans else [k, n]
    c_shape = [m, n]

    a_input_torch_dtype = CastMatmulConfig.get_torch_dtype(case["a_input_dtype"])
    b_input_torch_dtype = CastMatmulConfig.get_torch_dtype(case["b_input_dtype"])
    c_torch_dtype = CastMatmulConfig.get_torch_dtype(case["out_dtype"])

    if a_input_torch_dtype == torch.int8:
        a_tensor_cpu = torch.randint(-5, 6, a_shape, dtype=a_input_torch_dtype)
    else:
        a_tensor_cpu = torch.rand(a_shape, dtype=a_input_torch_dtype)

    if b_input_torch_dtype == torch.int8:
        b_tensor_cpu = torch.randint(-5, 6, b_shape, dtype=b_input_torch_dtype)
    else:
        b_tensor_cpu = torch.rand(b_shape, dtype=b_input_torch_dtype)

    matmul_dtype = CastMatmulConfig.get_torch_dtype(case["matmul_dtype"])

    a_cpu = a_tensor_cpu.to(matmul_dtype).T if config.a_trans else a_tensor_cpu.to(matmul_dtype)
    b_cpu = b_tensor_cpu.to(matmul_dtype).T if config.b_trans else b_tensor_cpu.to(matmul_dtype)

    if matmul_dtype == torch.int8:
        golden = torch.matmul(a_cpu.to(torch.int32), b_cpu.to(torch.int32)).to(c_torch_dtype)
    else:
        golden = torch.matmul(a_cpu.to(torch.float32), b_cpu.to(torch.float32)).to(c_torch_dtype)

    a_tensor = a_tensor_cpu.to(f"npu:{device_id}")
    b_tensor = b_tensor_cpu.to(f"npu:{device_id}")
    c_tensor = torch.zeros(c_shape, dtype=c_torch_dtype, device=f"npu:{device_id}")

    cast_matmul_pto_kernel(a_tensor, b_tensor, c_tensor, config)

    atol, rtol = CastMatmulConfig.get_tolerance(case["out_dtype"])

    assert torch.allclose(c_tensor.cpu(), golden.cpu(), atol=atol, rtol=rtol), (
        f"Test case {case['id']} ({case['name']}) failed"
    )


ALL_CAST_MATMUL_TESTS = CAST_RIGHT_MATMUL_TESTS + CAST_LEFT_MATMUL_TESTS + CAST_BOTH_MATMUL_TESTS


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def cast_scaled_mm_ub2l1_kernel(
    a_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
    b_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
    out_tensor: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
    scale_a_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], dtype=pypto.DT_FP8E8M0),
    scale_b_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], dtype=pypto.DT_FP8E8M0),
    config: ScaledMmUb2L1Config,
):
    m, n = config.m, config.n
    k = config.k
    vm, vn = config.view_shape
    m_loop = (m + vm - 1) // vm
    n_loop = (n + vn - 1) // vn
    scale_k = (k + K_BLOCK_SIZE_64 - 1) // K_BLOCK_SIZE_64

    # 当设置scope大于5000，即5001以上时，开启mix场景，走入UB2L1
    pypto.set_pass_options(sg_set_scope=10000)
    for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_L0_mIdx", idx_name="m_idx"):
        for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L0_nIdx", idx_name="n_idx"):
            m_offset = m_idx * vm
            n_offset = n_idx * vn

            a_view = pypto.view(a_tensor, [vm, k], [m_offset, 0], valid_shape=[min(vm, m - m_offset), k])

            # B 矩阵（转置布局 [n, k]）FP32 输入 Cast 为 FP8E4M3，cast 产出 UB 数据后
            # 作为 ScaledMM(MX) 右矩阵拷入 L1，触发 MX_PADDING_MODE 的 UB2L1 约束
            b_tile = pypto.view(b_tensor, [vn, k], [n_offset, 0], valid_shape=[min(vn, n - n_offset), k])
            pypto.set_vec_tile_shapes(*config.b_vec_tile_shape)
            b_compute = pypto.cast(b_tile, pypto.DT_FP8E4M3, pypto.CastMode.CAST_NONE)

            scale_a_view = pypto.view(
                scale_a_tensor, [vm, scale_k, SCALE_INNER_DIM], [m_offset, 0, 0],
                valid_shape=[min(vm, m - m_offset), scale_k, SCALE_INNER_DIM]
            )
            scale_b_view = pypto.view(
                scale_b_tensor, [scale_k, vn, SCALE_INNER_DIM], [0, n_offset, 0],
                valid_shape=[scale_k, min(vn, n - n_offset), SCALE_INNER_DIM]
            )

            pypto.set_cube_tile_shapes(*config.cube_tile_shape)
            out_view = pypto.scaled_mm(
                a_view,
                b_compute,
                config.out_pto_dtype,
                scale_a_view,
                scale_b_view,
                a_trans=False,
                b_trans=True,
                scale_a_trans=False,
                scale_b_trans=False,
                c_matrix_nz=False,
            )
            pypto.assemble(out_view, [m_offset, n_offset], out_tensor)
    # 运行完后设置回-1，关闭mix
    pypto.set_pass_options(sg_set_scope=-1)


def run_cast_scaled_mm_ub2l1_test(case: dict):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    config = ScaledMmUb2L1Config.from_test_case(case)

    m, k, n = config.m, config.k, config.n
    scale_k = (k + K_BLOCK_SIZE_64 - 1) // K_BLOCK_SIZE_64
    padding_k = scale_k * K_BLOCK_SIZE_64 - k

    # A 直接 FP8E4M3 输入；B 为 FP32 输入，kernel 内 Cast 为 FP8E4M3
    a_cpu = torch.rand([m, k], dtype=torch.float32).uniform_(-3, 3).to(torch.float8_e4m3fn)
    b_src_cpu = torch.rand([n, k], dtype=torch.float32).uniform_(-3, 3)
    b_cpu = b_src_cpu.to(torch.float8_e4m3fn)
    scale_a_cpu = torch.rand([m, scale_k, SCALE_INNER_DIM], dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)
    scale_b_cpu = torch.rand([scale_k, n, SCALE_INNER_DIM], dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)

    # golden 与 scaled_mm 语义对齐：scale 按 32 元素粒度展开到 K 维（K 向 pad 到 scale_k*64）
    scale_a_tmp = scale_a_cpu.view(m, scale_k * SCALE_INNER_DIM).to(torch.float32).repeat_interleave(32, dim=1)
    scale_b_tmp = (
        torch.transpose(scale_b_cpu, -2, -1)
        .reshape(scale_k * SCALE_INNER_DIM, n)
        .to(torch.float32)
        .repeat_interleave(32, dim=0)
    )

    mat_a_tmp = functional.pad(a_cpu.to(torch.float32), ((0, padding_k, 0, 0)), "constant")
    mat_a_tmp = mat_a_tmp * scale_a_tmp
    mat_b_tmp = functional.pad(b_cpu.to(torch.float32).T, ((0, 0, 0, padding_k)), "constant")
    mat_b_tmp = scale_b_tmp * mat_b_tmp

    out_torch_dtype = ScaledMmUb2L1Config.get_torch_dtype(config.out_dtype)
    golden = torch.matmul(mat_a_tmp, mat_b_tmp).to(out_torch_dtype)

    device = f"npu:{device_id}"
    a_npu = a_cpu.to(device)
    b_npu = b_src_cpu.to(device)
    scale_a_npu = scale_a_cpu.to(device)
    scale_b_npu = scale_b_cpu.to(device)
    out_npu = torch.zeros([m, n], dtype=out_torch_dtype, device=device)

    cast_scaled_mm_ub2l1_kernel(a_npu, b_npu, out_npu, scale_a_npu, scale_b_npu, config)

    atol, rtol = ScaledMmUb2L1Config.get_tolerance(config.out_dtype)
    assert torch.allclose(out_npu.cpu(), golden, atol=atol, rtol=rtol), (
        f"Test case {case['id']} ({case['name']}) failed"
    )


def run_cast_matmul_nd_test(case: dict):
    """3D/4D batch cast+matmul 测试入口，按 batch 维个数分发到对应 kernel。"""
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    config = CastMatmulConfig.from_test_case(case)

    m, k, n = config.shape
    batch_shape = list(config.batch_shape)
    if config.a_trans:
        a_shape = batch_shape + [k, m]
    else:
        a_shape = batch_shape + [m, k]
    if config.b_trans:
        b_shape = batch_shape + [n, k]
    else:
        b_shape = batch_shape + [k, n]
    c_shape = batch_shape + [m, n]

    a_input_torch_dtype = CastMatmulConfig.get_torch_dtype(case["a_input_dtype"])
    b_input_torch_dtype = CastMatmulConfig.get_torch_dtype(case["b_input_dtype"])
    c_torch_dtype = CastMatmulConfig.get_torch_dtype(case["out_dtype"])
    matmul_dtype = CastMatmulConfig.get_torch_dtype(case["matmul_dtype"])

    # 对齐用例表 -5_5 datarange：cast 目标为 int8 时浮点源数据取 [-5, 5)，保证截断结果非平凡
    def gen_input(shape, dtype):
        if dtype == torch.int8:
            return torch.randint(-5, 6, shape, dtype=dtype)
        if matmul_dtype == torch.int8:
            return torch.rand(shape, dtype=dtype) * 10 - 5
        return torch.rand(shape, dtype=dtype)

    a_tensor_cpu = gen_input(a_shape, a_input_torch_dtype)
    b_tensor_cpu = gen_input(b_shape, b_input_torch_dtype)

    a_cpu = a_tensor_cpu.to(matmul_dtype).transpose(-2, -1) if config.a_trans else a_tensor_cpu.to(matmul_dtype)
    b_cpu = b_tensor_cpu.to(matmul_dtype).transpose(-2, -1) if config.b_trans else b_tensor_cpu.to(matmul_dtype)

    if matmul_dtype == torch.int8:
        golden = torch.matmul(a_cpu.to(torch.int32), b_cpu.to(torch.int32)).to(c_torch_dtype)
    else:
        golden = torch.matmul(a_cpu.to(torch.float32), b_cpu.to(torch.float32)).to(c_torch_dtype)

    a_tensor = a_tensor_cpu.to(f"npu:{device_id}")
    b_tensor = b_tensor_cpu.to(f"npu:{device_id}")
    c_tensor = torch.zeros(c_shape, dtype=c_torch_dtype, device=f"npu:{device_id}")

    if len(batch_shape) == 1:
        cast_matmul_pto_kernel_3d(a_tensor, b_tensor, c_tensor, config)
    else:
        cast_matmul_pto_kernel_4d(a_tensor, b_tensor, c_tensor, config)

    atol, rtol = CastMatmulConfig.get_tolerance(case["out_dtype"])

    assert torch.allclose(c_tensor.cpu(), golden.cpu(), atol=atol, rtol=rtol), (
        f"Test case {case['id']} ({case['name']}) failed"
    )


@pytest.mark.parametrize(
    "case", [pytest.param(case, marks=pytest.mark.soc(*case["products"])) for case in SCALED_MM_UB2L1_TESTS]
)
@pypto.options(pass_options={"enable_slice": False})
def test_cast_scaled_mm_ub2l1(case: dict):
    run_cast_scaled_mm_ub2l1_test(case)


@pytest.mark.parametrize(
    "case", [pytest.param(case, marks=pytest.mark.soc(*case["products"])) for case in ALL_CAST_MATMUL_TESTS]
)
@pypto.options(pass_options={"enable_slice": True})
def test_cast_matmul(case: dict):
    run_cast_matmul_test(case)


@pytest.mark.parametrize(
    "case", [pytest.param(case, marks=pytest.mark.soc(*case["products"])) for case in CAST_3D_MATMUL_TESTS]
)
@pypto.options(pass_options={"enable_slice": True})
def test_cast_matmul_3d(case: dict):
    run_cast_matmul_nd_test(case)


@pytest.mark.parametrize(
    "case", [pytest.param(case, marks=pytest.mark.soc(*case["products"])) for case in CAST_4D_MATMUL_TESTS]
)
@pypto.options(pass_options={"enable_slice": True})
def test_cast_matmul_4d(case: dict):
    run_cast_matmul_nd_test(case)


def run_cast_matmul_demo(run_mode):
    m_size, k_size, n_size = 256, 256, 256
    m_view_size, n_view_size = 128, 128

    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")

    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0}, runtime_options={"run_mode": mode}
    )
    def cast_matmul_demo_kernel(
        a: pypto.Tensor([], pypto.DT_FP32),
        b: pypto.Tensor([], pypto.DT_FP16),
        out: pypto.Tensor([], pypto.DT_FP16),
    ):
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
        pypto.set_pass_options(sg_set_scope=10000)
        m_loop = (m_size + m_view_size - 1) // m_view_size
        n_loop = (n_size + n_view_size - 1) // n_view_size

        for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_L0_mIdx", idx_name="m_idx"):
            for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L0_nIdx", idx_name="n_idx"):
                a_tile = pypto.view(a, [m_view_size, k_size], [m_idx * m_view_size, 0])
                pypto.set_vec_tile_shapes(m_view_size, k_size)
                a_fp16_tile = pypto.cast(a_tile, pypto.DT_FP16)

                b_view = pypto.view(b, [k_size, n_view_size], [0, n_idx * n_view_size])
                out_view = pypto.matmul(a_fp16_tile, b_view, pypto.DT_FP16)
                out[
                    m_idx * m_view_size:m_idx * m_view_size + m_view_size,
                    n_idx * n_view_size:n_idx * n_view_size + n_view_size,
                ] = out_view

    device = "npu:0" if run_mode == "npu" else "cpu"
    a = torch.randn([m_size, k_size], dtype=torch.float32, device=device)
    b = torch.randn([k_size, n_size], dtype=torch.float16, device=device)
    out = torch.empty(m_size, n_size, dtype=torch.float16, device=device)
    cast_matmul_demo_kernel(a, b, out)


if __name__ == "__main__":
    run_cast_matmul_demo("npu")
