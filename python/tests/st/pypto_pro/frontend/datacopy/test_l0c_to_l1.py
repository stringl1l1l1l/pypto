# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end tests for L0C Acc NZ to L1 Mat FixPipe transfers.

Every kernel produces the source Acc tile with a real matmul, transfers it to
L1, and consumes the L1 result with a second matmul before writing GM output.
The TMOV, TEXTRACT, and TINSERT paths also cover paired AccPhase/STPhase flows.
"""

import logging
import os
import struct

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
SIGNED_INT8_FLAG = 1 << 46


def _require_a5(device: str) -> None:
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip("not A5")


def _scale_bits(scale: float) -> int:
    return struct.unpack("!I", struct.pack("!f", scale))[0]


def _scaling_params(scales: torch.Tensor, device: str, *, signed_int8: bool = True) -> torch.Tensor:
    flag = SIGNED_INT8_FLAG if signed_int8 else 0
    encoded = [flag | _scale_bits(float(value)) for value in scales.tolist()]
    return torch.tensor([encoded], dtype=torch.int64, device=device)


def _assert_result(label: str, actual: torch.Tensor, expected: torch.Tensor, *, atol: float) -> None:
    actual_cpu = actual.cpu()
    expected_cpu = expected.cpu()
    diff = (actual_cpu.to(torch.float64) - expected_cpu.to(torch.float64)).abs()
    max_diff = diff.max().item() if diff.numel() else 0.0
    logging.info("%s: max_abs_diff=%s", label, max_diff)
    try:
        torch.testing.assert_close(actual_cpu, expected_cpu, rtol=0, atol=atol)
    except AssertionError:
        mismatch = diff > atol
        first = mismatch.nonzero()[0].tolist() if mismatch.any() else None
        print(f"[FAIL] {label}: max_abs_diff={max_diff}, first_mismatch={first}")
        print(f"actual[:4, :8]=\n{actual_cpu[:4, :8]}")
        print(f"golden[:4, :8]=\n{expected_cpu[:4, :8]}")
        raise


@pl.jit(auto_mutex=True)
def l0c_to_l1_tmov_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    rhs: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    mat_fp32 = pl.TileType(
        shape=[64, 64],
        dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Mat,
        layout=pl.NZ,
        valid_shape=[-1, -1],
        compact=1,
    )
    a_mat_group = pl.make_tile_group(
        type=mat_fp32,
        addrs=[0x0000, 0x4000],
        mutex_ids=[0, 1],
    )
    b_mat_group = pl.make_tile_group(
        type=mat_fp32,
        addrs=[0x8000, 0xC000],
        mutex_ids=[2, 3],
    )
    result_mat_group = pl.make_tile_group(
        type=mat_fp32,
        addrs=0x10000,
        mutex_ids=[4],
    )
    rhs_mat_group = pl.make_tile_group(
        type=mat_fp32,
        addrs=0x14000,
        mutex_ids=[5],
    )
    left_fp32 = pl.TileType(
        shape=[64, 64],
        dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Left,
        layout=pl.NZ,
        valid_shape=[-1, -1],
        compact=1,
    )
    first_left_group = pl.make_tile_group(type=left_fp32, addrs=[0x0000, 0x4000], mutex_ids=[6, 7])
    second_left_group = pl.make_tile_group(type=left_fp32, addrs=0x8000, mutex_ids=[8])
    right_fp32 = pl.TileType(
        shape=[64, 64],
        dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Right,
        layout=pl.ZN,
        valid_shape=[-1, -1],
        compact=1,
    )
    first_right_group = pl.make_tile_group(type=right_fp32, addrs=[0x0000, 0x4000], mutex_ids=[9, 10])
    second_right_group = pl.make_tile_group(type=right_fp32, addrs=0x8000, mutex_ids=[11])
    acc_fp32 = pl.TileType(
        shape=[64, 64],
        dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Acc,
        layout=pl.NZ,
        fractal=1024,
        valid_shape=[-1, -1],
        compact=1,
    )
    first_acc_group = pl.make_tile_group(type=acc_fp32, addrs=0x0000, mutex_ids=[12])
    second_acc_group = pl.make_tile_group(type=acc_fp32, addrs=0x4000, mutex_ids=[13])

    with pl.section_cube():
        total_m = a.shape[0]
        total_k = a.shape[1]
        total_n = b.shape[1]
        for m_offset in pl.range(0, total_m, 64):
            valid_m = pl.min(total_m - m_offset, 64)
            for n_offset in pl.range(0, total_n, 64):
                valid_n = pl.min(total_n - n_offset, 64)
                first_acc = first_acc_group.current()
                pl.set_validshape(first_acc, [valid_m, valid_n])

                k_blocks = (total_k + 63) // 64
                for k_idx in pl.range(0, k_blocks):
                    k_offset = k_idx * 64
                    valid_k = pl.min(total_k - k_offset, 64)
                    a_mat = a_mat_group.next()
                    b_mat = b_mat_group.next()
                    first_left = first_left_group.next()
                    first_right = first_right_group.next()
                    pl.set_validshape(a_mat, [valid_m, valid_k])
                    pl.set_validshape(b_mat, [valid_k, valid_n])
                    pl.set_validshape(first_left, [valid_m, valid_k])
                    pl.set_validshape(first_right, [valid_k, valid_n])
                    pl.load(a_mat, a, [m_offset, k_offset])
                    pl.load(b_mat, b, [k_offset, n_offset])
                    pl.move(first_left, a_mat)
                    pl.move(first_right, b_mat)
                    if k_idx == 0:
                        if k_idx == k_blocks - 1:
                            pl.matmul(first_acc, first_left, first_right, phase=pl.AccPhase.Final)
                        else:
                            pl.matmul(first_acc, first_left, first_right, phase=pl.AccPhase.Partial)
                    elif k_idx == k_blocks - 1:
                        pl.matmul_acc(
                            first_acc,
                            first_acc,
                            first_left,
                            first_right,
                            phase=pl.AccPhase.Final,
                        )
                    else:
                        pl.matmul_acc(
                            first_acc,
                            first_acc,
                            first_left,
                            first_right,
                            phase=pl.AccPhase.Partial,
                        )

                result_mat = result_mat_group.current()
                rhs_mat = rhs_mat_group.current()
                second_left = second_left_group.current()
                second_right = second_right_group.current()
                second_acc = second_acc_group.current()
                pl.set_validshape(result_mat, [valid_m, valid_n])
                pl.set_validshape(rhs_mat, [valid_n, valid_n])
                pl.set_validshape(second_left, [valid_m, valid_n])
                pl.set_validshape(second_right, [valid_n, valid_n])
                pl.set_validshape(second_acc, [valid_m, valid_n])
                pl.load(rhs_mat, rhs, [n_offset, n_offset])
                pl.move(result_mat, first_acc, phase=pl.STPhase.Final)
                pl.move(second_left, result_mat)
                pl.move(second_right, rhs_mat)
                pl.matmul(second_acc, second_left, second_right)
                pl.store(out, second_acc, [m_offset, n_offset])


@pl.jit(auto_mutex=True)
def l0c_to_l1_scalar_tmov_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    rhs: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    scale_bits: pl.DT_INT32,
):
    mat_fp32 = pl.TileType(
        shape=[64, 64],
        dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Mat,
        layout=pl.NZ,
        valid_shape=[-1, -1],
    )
    mat_fp32_group = pl.make_tile_group(type=mat_fp32, addrs=[0x0000, 0x4000], mutex_ids=[0, 1])
    mat_int8 = pl.TileType(
        shape=[64, 64],
        dtype=pl.DT_INT8,
        target_memory=pl.MemorySpace.Mat,
        layout=pl.NZ,
        valid_shape=[-1, -1],
    )
    mat_int8_group = pl.make_tile_group(type=mat_int8, addrs=[0x8000, 0x9000], mutex_ids=[2, 3])
    first_left_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Left,
            layout=pl.NZ,
            valid_shape=[-1, -1],
        ),
        addrs=0x0000,
        mutex_ids=[4],
    )
    second_left_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_INT8,
            target_memory=pl.MemorySpace.Left,
            layout=pl.NZ,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addrs=0x4000,
        mutex_ids=[5],
    )
    first_right_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Right,
            layout=pl.ZN,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addrs=0x0000,
        mutex_ids=[6],
    )
    second_right_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_INT8,
            target_memory=pl.MemorySpace.Right,
            layout=pl.ZN,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addrs=0x4000,
        mutex_ids=[7],
    )
    first_acc_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            fractal=1024,
            valid_shape=[-1, -1],
        ),
        addrs=0x0000,
        mutex_ids=[8],
    )
    second_acc_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_INT32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            fractal=1024,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addrs=0x4000,
        mutex_ids=[9],
    )

    with pl.section_cube():
        valid_m = a.shape[0]
        valid_k = a.shape[1]
        valid_n = b.shape[1]
        a_mat = mat_fp32_group.next()
        b_mat = mat_fp32_group.next()
        rhs_mat = mat_int8_group.next()
        result_mat = mat_int8_group.next()
        first_left = first_left_group.current()
        second_left = second_left_group.current()
        first_right = first_right_group.current()
        second_right = second_right_group.current()
        first_acc = first_acc_group.current()
        second_acc = second_acc_group.current()

        pl.set_validshape(a_mat, [valid_m, valid_k])
        pl.set_validshape(b_mat, [valid_k, valid_n])
        pl.set_validshape(rhs_mat, [valid_n, valid_n])
        pl.set_validshape(first_left, [valid_m, valid_k])
        pl.set_validshape(first_right, [valid_k, valid_n])
        pl.set_validshape(first_acc, [valid_m, valid_n])
        pl.set_validshape(result_mat, [valid_m, valid_n])
        pl.set_validshape(second_left, [valid_m, valid_n])
        pl.set_validshape(second_right, [valid_n, valid_n])
        pl.set_validshape(second_acc, [valid_m, valid_n])

        pl.load(a_mat, a, [0, 0])
        pl.load(b_mat, b, [0, 0])
        pl.load(rhs_mat, rhs, [0, 0])

        pl.move(first_left, a_mat)
        pl.move(first_right, b_mat)
        pl.move(second_right, rhs_mat)

        pl.matmul(first_acc, first_left, first_right)

        pl.move(result_mat, first_acc, scale=scale_bits)

        pl.move(second_left, result_mat)
        pl.matmul(second_acc, second_left, second_right)
        pl.store(out, second_acc, [0, 0])


@pl.jit(auto_mutex=True)
def l0c_to_l1_scaling_tmov_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    rhs: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    fp_params: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT64],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
):
    mat_fp32 = pl.TileType(
        shape=[64, 64],
        dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Mat,
        layout=pl.NZ,
        valid_shape=[-1, -1],
    )
    mat_fp32_group = pl.make_tile_group(type=mat_fp32, addrs=[0x0000, 0x4000], mutex_ids=[0, 1])
    mat_int8 = pl.TileType(
        shape=[64, 64],
        dtype=pl.DT_INT8,
        target_memory=pl.MemorySpace.Mat,
        layout=pl.NZ,
        valid_shape=[-1, -1],
    )
    mat_int8_group = pl.make_tile_group(type=mat_int8, addrs=[0x8000, 0x9000], mutex_ids=[2, 3])
    fp_mat_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 32],
            dtype=pl.DT_INT64,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ND,
            valid_shape=[-1, -1],
        ),
        addrs=0xA000,
        mutex_ids=[4],
    )
    fp_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 32],
            dtype=pl.DT_INT64,
            target_memory=pl.MemorySpace.Scaling,
            valid_shape=[-1, -1],
        ),
        addrs=0x0000,
        mutex_ids=[5],
    )
    first_left_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Left,
            layout=pl.NZ,
            valid_shape=[-1, -1],
        ),
        addrs=0x0000,
        mutex_ids=[6],
    )
    second_left_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_INT8,
            target_memory=pl.MemorySpace.Left,
            layout=pl.NZ,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addrs=0x4000,
        mutex_ids=[7],
    )
    first_right_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Right,
            layout=pl.ZN,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addrs=0x0000,
        mutex_ids=[8],
    )
    second_right_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_INT8,
            target_memory=pl.MemorySpace.Right,
            layout=pl.ZN,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addrs=0x4000,
        mutex_ids=[9],
    )
    first_acc_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            fractal=1024,
            valid_shape=[-1, -1],
        ),
        addrs=0x0000,
        mutex_ids=[10],
    )
    second_acc_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_INT32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            fractal=1024,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addrs=0x4000,
        mutex_ids=[11],
    )

    with pl.section_cube():
        valid_m = a.shape[0]
        valid_k = a.shape[1]
        valid_n = b.shape[1]
        a_mat = mat_fp32_group.next()
        b_mat = mat_fp32_group.next()
        rhs_mat = mat_int8_group.next()
        result_mat = mat_int8_group.next()
        fp_mat = fp_mat_group.current()
        fp_tile = fp_group.current()
        first_left = first_left_group.current()
        second_left = second_left_group.current()
        first_right = first_right_group.current()
        second_right = second_right_group.current()
        first_acc = first_acc_group.current()
        second_acc = second_acc_group.current()

        pl.set_validshape(a_mat, [valid_m, valid_k])
        pl.set_validshape(b_mat, [valid_k, valid_n])
        pl.set_validshape(rhs_mat, [valid_n, valid_n])
        pl.set_validshape(fp_mat, [1, valid_n])
        pl.set_validshape(fp_tile, [1, valid_n])
        pl.set_validshape(first_left, [valid_m, valid_k])
        pl.set_validshape(first_right, [valid_k, valid_n])
        pl.set_validshape(first_acc, [valid_m, valid_n])
        pl.set_validshape(result_mat, [valid_m, valid_n])
        pl.set_validshape(second_left, [valid_m, valid_n])
        pl.set_validshape(second_right, [valid_n, valid_n])
        pl.set_validshape(second_acc, [valid_m, valid_n])

        pl.load(a_mat, a, [0, 0])
        pl.load(b_mat, b, [0, 0])
        pl.load(rhs_mat, rhs, [0, 0])
        pl.load(fp_mat, fp_params, [0, 0])

        pl.move(first_left, a_mat)
        pl.move(first_right, b_mat)
        pl.move(second_right, rhs_mat)
        pl.move(fp_tile, fp_mat)

        pl.matmul(first_acc, first_left, first_right)

        pl.move(result_mat, first_acc, scale=fp_tile, relu_pre_mode=pl.ReluPreMode.NormalRelu)

        pl.move(second_left, result_mat)
        pl.matmul(second_acc, second_left, second_right)
        pl.store(out, second_acc, [0, 0])


@pl.jit(auto_mutex=True)
def l0c_to_l1_extract_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    rhs: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    fp_params: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT64],
    scalar_scale_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    scaling_tile_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    scale_bits: pl.DT_INT32,
):
    mat_fp32_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
            valid_shape=[-1, -1],
        ),
        addrs=[0x0000, 0x4000],
        mutex_ids=[0, 1],
    )
    rhs_mat_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[32, 32],
            dtype=pl.DT_INT8,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
            valid_shape=[-1, -1],
        ),
        addrs=0x8000,
        mutex_ids=[2],
    )
    result_mat_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[32, 32],
            dtype=pl.DT_INT8,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
            valid_shape=[-1, -1],
        ),
        addrs=[0x8400, 0x9000],
        mutex_ids=[3, 12],
    )
    fp_mat_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 64],
            dtype=pl.DT_INT64,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ND,
            valid_shape=[-1, -1],
        ),
        addrs=0x8C00,
        mutex_ids=[10],
    )
    fp_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 64],
            dtype=pl.DT_INT64,
            target_memory=pl.MemorySpace.Scaling,
            valid_shape=[-1, -1],
        ),
        addrs=0x0000,
        mutex_ids=[11],
    )
    first_left_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Left,
            layout=pl.NZ,
            valid_shape=[-1, -1],
        ),
        addrs=0x0000,
        mutex_ids=[4],
    )
    second_left_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[32, 32],
            dtype=pl.DT_INT8,
            target_memory=pl.MemorySpace.Left,
            layout=pl.NZ,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addrs=[0x4000, 0x4400],
        mutex_ids=[5, 13],
    )
    first_right_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Right,
            layout=pl.ZN,
            valid_shape=[-1, -1],
        ),
        addrs=0x0000,
        mutex_ids=[6],
    )
    second_right_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[32, 32],
            dtype=pl.DT_INT8,
            target_memory=pl.MemorySpace.Right,
            layout=pl.ZN,
            valid_shape=[-1, -1],
        ),
        addrs=0x4000,
        mutex_ids=[7],
    )
    first_acc_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            fractal=1024,
            valid_shape=[-1, -1],
        ),
        addrs=0x0000,
        mutex_ids=[8],
    )
    second_acc_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[32, 32],
            dtype=pl.DT_INT32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            fractal=1024,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addrs=[0x4000, 0x5000],
        mutex_ids=[9, 14],
    )

    with pl.section_cube():
        src_m = a.shape[0]
        src_k = a.shape[1]
        src_n = b.shape[1]
        valid_m = scalar_scale_out.shape[0]
        valid_n = scalar_scale_out.shape[1]
        a_mat = mat_fp32_group.next()
        b_mat = mat_fp32_group.next()
        rhs_mat = rhs_mat_group.current()
        scalar_result_mat = result_mat_group.next()
        scaling_result_mat = result_mat_group.next()
        fp_mat = fp_mat_group.current()
        fp_tile = fp_group.current()
        first_left = first_left_group.current()
        scalar_second_left = second_left_group.next()
        scaling_second_left = second_left_group.next()
        first_right = first_right_group.current()
        second_right = second_right_group.current()
        first_acc = first_acc_group.current()
        scalar_second_acc = second_acc_group.next()
        scaling_second_acc = second_acc_group.next()

        pl.set_validshape(a_mat, [src_m, src_k])
        pl.set_validshape(b_mat, [src_k, src_n])
        pl.set_validshape(first_left, [src_m, src_k])
        pl.set_validshape(first_right, [src_k, src_n])
        pl.set_validshape(first_acc, [src_m, src_n])
        pl.set_validshape(rhs_mat, [valid_n, valid_n])
        pl.set_validshape(scalar_result_mat, [valid_m, valid_n])
        pl.set_validshape(scaling_result_mat, [valid_m, valid_n])
        pl.set_validshape(fp_mat, [1, src_n])
        pl.set_validshape(fp_tile, [1, src_n])
        pl.set_validshape(scalar_second_left, [valid_m, valid_n])
        pl.set_validshape(scaling_second_left, [valid_m, valid_n])
        pl.set_validshape(second_right, [valid_n, valid_n])
        pl.set_validshape(scalar_second_acc, [valid_m, valid_n])
        pl.set_validshape(scaling_second_acc, [valid_m, valid_n])

        pl.load(a_mat, a, [0, 0])
        pl.load(b_mat, b, [0, 0])
        pl.load(rhs_mat, rhs, [0, 0])
        pl.load(fp_mat, fp_params, [0, 0])

        pl.move(first_left, a_mat)
        pl.move(first_right, b_mat)
        pl.move(second_right, rhs_mat)
        pl.move(fp_tile, fp_mat)

        pl.matmul(first_acc, first_left, first_right, phase=pl.AccPhase.Final)

        # Partial keeps the finalized Acc readable; the second drain clears it with Final.
        pl.move(
            scalar_result_mat,
            first_acc,
            [16, 16],
            scale=scale_bits,
            phase=pl.STPhase.Partial,
        )

        pl.move(scalar_second_left, scalar_result_mat)
        pl.matmul(scalar_second_acc, scalar_second_left, second_right)
        pl.store(scalar_scale_out, scalar_second_acc, [0, 0])

        pl.move(
            scaling_result_mat,
            first_acc,
            [16, 16],
            scale=fp_tile,
            relu_pre_mode=pl.ReluPreMode.NormalRelu,
            phase=pl.STPhase.Final,
        )
        pl.move(scaling_second_left, scaling_result_mat)
        pl.matmul(scaling_second_acc, scaling_second_left, second_right)
        pl.store(scaling_tile_out, scaling_second_acc, [0, 0])


@pl.jit(auto_mutex=True)
def l0c_to_l1_insert_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    seed: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    rhs: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    fp_params: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT64],
    scalar_scale_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    scaling_tile_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    scale_bits: pl.DT_INT32,
):
    a_mat_group = pl.make_tile_group(
        type=pl.TileType(
            # TINSERT uses the Acc source tile's static M extent. Match it to
            # the 16-row inserted window so rows outside that window are not
            # copied into result_mat.
            shape=[16, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
            valid_shape=[-1, -1],
        ),
        addrs=0x0000,
        mutex_ids=[0],
    )
    b_mat_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 32],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
            valid_shape=[-1, -1],
        ),
        addrs=0x2000,
        mutex_ids=[1],
    )
    result_mat_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_INT8,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
            valid_shape=[-1, -1],
        ),
        addrs=0x4000,
        mutex_ids=[2],
    )
    rhs_mat_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_INT8,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
            valid_shape=[-1, -1],
        ),
        addrs=0x5000,
        mutex_ids=[3],
    )
    fp_mat_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 32],
            dtype=pl.DT_INT64,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ND,
            valid_shape=[-1, -1],
        ),
        addrs=0x6000,
        mutex_ids=[4],
    )
    fp_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 32],
            dtype=pl.DT_INT64,
            target_memory=pl.MemorySpace.Scaling,
            valid_shape=[-1, -1],
        ),
        addrs=0x0000,
        mutex_ids=[5],
    )
    first_left_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[16, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Left,
            layout=pl.NZ,
            valid_shape=[-1, -1],
        ),
        addrs=0x0000,
        mutex_ids=[6],
    )
    second_left_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_INT8,
            target_memory=pl.MemorySpace.Left,
            layout=pl.NZ,
            valid_shape=[-1, -1],
        ),
        addrs=0x2000,
        mutex_ids=[7],
    )
    first_right_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 32],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Right,
            layout=pl.ZN,
            valid_shape=[-1, -1],
        ),
        addrs=0x0000,
        mutex_ids=[8],
    )
    second_right_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_INT8,
            target_memory=pl.MemorySpace.Right,
            layout=pl.ZN,
            valid_shape=[-1, -1],
        ),
        addrs=0x2000,
        mutex_ids=[9],
    )
    first_acc_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[16, 32],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            fractal=1024,
            valid_shape=[-1, -1],
        ),
        addrs=0x0000,
        mutex_ids=[10],
    )
    second_acc_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_INT32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            fractal=1024,
            valid_shape=[-1, -1],
        ),
        addrs=0x1000,
        mutex_ids=[11],
    )

    with pl.section_cube():
        valid_m = a.shape[0]
        valid_k = a.shape[1]
        valid_n = b.shape[1]
        target_m = seed.shape[0]
        target_n = seed.shape[1]
        a_mat = a_mat_group.current()
        b_mat = b_mat_group.current()
        result_mat = result_mat_group.current()
        rhs_mat = rhs_mat_group.current()
        fp_mat = fp_mat_group.current()
        fp_tile = fp_group.current()
        first_left = first_left_group.current()
        second_left = second_left_group.current()
        first_right = first_right_group.current()
        second_right = second_right_group.current()
        first_acc = first_acc_group.current()
        second_acc = second_acc_group.current()

        pl.set_validshape(a_mat, [valid_m, valid_k])
        pl.set_validshape(b_mat, [valid_k, valid_n])
        pl.set_validshape(first_left, [valid_m, valid_k])
        pl.set_validshape(first_right, [valid_k, valid_n])
        pl.set_validshape(first_acc, [valid_m, valid_n])
        pl.set_validshape(fp_mat, [1, valid_n])
        pl.set_validshape(fp_tile, [1, valid_n])
        pl.set_validshape(result_mat, [target_m, target_n])
        pl.set_validshape(rhs_mat, [target_n, target_n])
        pl.set_validshape(second_left, [target_m, target_n])
        pl.set_validshape(second_right, [target_n, target_n])
        pl.set_validshape(second_acc, [target_m, target_n])

        pl.load(a_mat, a, [0, 0])
        pl.load(b_mat, b, [0, 0])
        pl.load(result_mat, seed, [0, 0])
        pl.load(rhs_mat, rhs, [0, 0])
        pl.load(fp_mat, fp_params, [0, 0])

        pl.move(first_left, a_mat)
        pl.move(first_right, b_mat)
        pl.move(second_right, rhs_mat)
        pl.move(fp_tile, fp_mat)

        pl.matmul(first_acc, first_left, first_right, phase=pl.AccPhase.Final)

        # Partial keeps the finalized Acc readable; the explicit insert below clears it with Final.
        # A destination offset on move is lowered to TINSERT.
        pl.move(
            result_mat,
            first_acc,
            [16, 32],
            scale=scale_bits,
            phase=pl.STPhase.Partial,
        )
        pl.move(second_left, result_mat)
        pl.matmul(second_acc, second_left, second_right)
        pl.store(scalar_scale_out, second_acc, [0, 0])

        pl.load(result_mat, seed, [0, 0])
        pl.insert(
            result_mat,
            first_acc,
            [16, 32],
            scale=fp_tile,
            relu_pre_mode=pl.ReluPreMode.NormalRelu,
            phase=pl.STPhase.Final,
        )

        pl.move(second_left, result_mat)
        pl.matmul(second_acc, second_left, second_right)
        pl.store(scaling_tile_out, second_acc, [0, 0])


def _make_l0c_to_l1_dtype_kernel(
    name: str,
    input_dtype,
    acc_dtype,
    result_dtype,
    output_dtype,
    *,
    scale_value: float | None,
    use_scaling_tile: bool = False,
):
    """Build a tail-block TMOV kernel for one legal dtype combination."""
    use_scalar_scale = scale_value is not None
    cube_dtype = pl.DT_INT8 if result_dtype in (pl.DT_UINT8, pl.DT_FP8E4M3FN) else result_dtype

    @pl.jit(auto_mutex=True, name=name)
    def kernel(
        a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], input_dtype],
        b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], input_dtype],
        rhs: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], cube_dtype],
        fp_params: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT64],
        out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], output_dtype],
    ):
        input_mat_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[64, 64],
                dtype=input_dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
                valid_shape=[-1, -1],
            ),
            addrs=[0x0000, 0x4000],
            mutex_ids=[0, 1],
        )
        result_mat_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[64, 64],
                dtype=result_dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
                valid_shape=[-1, -1],
            ),
            addrs=0x8000,
            mutex_ids=[2],
        )
        rhs_mat_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[64, 64],
                dtype=cube_dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
                valid_shape=[-1, -1],
            ),
            addrs=0xC000,
            mutex_ids=[3],
        )
        fp_mat_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, 64],
                dtype=pl.DT_INT64,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.ND,
                valid_shape=[-1, -1],
            ),
            addrs=0x10000,
            mutex_ids=[10],
        )
        fp_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, 64],
                dtype=pl.DT_INT64,
                target_memory=pl.MemorySpace.Scaling,
                valid_shape=[-1, -1],
            ),
            addrs=0x0000,
            mutex_ids=[11],
        )
        first_left_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[64, 64],
                dtype=input_dtype,
                target_memory=pl.MemorySpace.Left,
                layout=pl.NZ,
                valid_shape=[-1, -1],
            ),
            addrs=0x0000,
            mutex_ids=[4],
        )
        second_left_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[64, 64],
                dtype=cube_dtype,
                target_memory=pl.MemorySpace.Left,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x4000,
            mutex_ids=[5],
        )
        first_right_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[64, 64],
                dtype=input_dtype,
                target_memory=pl.MemorySpace.Right,
                layout=pl.ZN,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0000,
            mutex_ids=[6],
        )
        second_right_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[64, 64],
                dtype=cube_dtype,
                target_memory=pl.MemorySpace.Right,
                layout=pl.ZN,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x4000,
            mutex_ids=[7],
        )
        first_acc_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[64, 64],
                dtype=acc_dtype,
                target_memory=pl.MemorySpace.Acc,
                layout=pl.NZ,
                fractal=1024,
                valid_shape=[-1, -1],
            ),
            addrs=0x0000,
            mutex_ids=[8],
        )
        second_acc_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[64, 64],
                dtype=output_dtype,
                target_memory=pl.MemorySpace.Acc,
                layout=pl.NZ,
                fractal=1024,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x4000,
            mutex_ids=[9],
        )

        with pl.section_cube():
            valid_m = a.shape[0]
            valid_k = a.shape[1]
            valid_n = b.shape[1]
            a_mat = input_mat_group.next()
            b_mat = input_mat_group.next()
            result_mat = result_mat_group.current()
            rhs_mat = rhs_mat_group.current()
            fp_mat = fp_mat_group.current()
            fp_tile = fp_group.current()
            first_left = first_left_group.current()
            second_left = second_left_group.current()
            first_right = first_right_group.current()
            second_right = second_right_group.current()
            first_acc = first_acc_group.current()
            second_acc = second_acc_group.current()
            result_for_cube = pl.reinterpret(result_mat, shape=[64, 64], dtype=cube_dtype)

            pl.set_validshape(a_mat, [valid_m, valid_k])
            pl.set_validshape(first_left, [valid_m, valid_k])
            pl.set_validshape(b_mat, [valid_k, valid_n])
            pl.set_validshape(first_right, [valid_k, valid_n])
            pl.set_validshape(result_mat, [valid_m, valid_n])
            pl.set_validshape(result_for_cube, [valid_m, valid_n])
            pl.set_validshape(second_left, [valid_m, valid_n])
            pl.set_validshape(first_acc, [valid_m, valid_n])
            pl.set_validshape(second_acc, [valid_m, valid_n])
            pl.set_validshape(rhs_mat, [valid_n, valid_n])
            pl.set_validshape(second_right, [valid_n, valid_n])
            pl.set_validshape(fp_mat, [1, valid_n])
            pl.set_validshape(fp_tile, [1, valid_n])

            pl.load(a_mat, a, [0, 0])
            pl.load(b_mat, b, [0, 0])
            pl.load(rhs_mat, rhs, [0, 0])
            if use_scaling_tile:
                pl.load(fp_mat, fp_params, [0, 0])
            pl.move(first_left, a_mat)
            pl.move(first_right, b_mat)
            if use_scaling_tile:
                pl.move(fp_tile, fp_mat)
            pl.matmul(first_acc, first_left, first_right)
            if use_scaling_tile:
                pl.move(result_mat, first_acc, scale=fp_tile)
            elif use_scalar_scale:
                pl.move(result_mat, first_acc, scale=scale_value)
            else:
                pl.move(result_mat, first_acc)
            pl.move(second_left, result_for_cube)
            pl.move(second_right, rhs_mat)
            pl.matmul(second_acc, second_left, second_right)
            pl.store(out, second_acc, [0, 0])

    return kernel


def _dtype_case(
    label,
    source,
    result_dtype,
    output_dtype,
    result_torch_dtype,
    output_torch_dtype,
    *,
    scale_kind="scalar",
    scale=0.5,
    expected_kind="cast",
):
    return {
        "label": label,
        "source": source,
        "result_dtype": result_dtype,
        "output_dtype": output_dtype,
        "result_torch_dtype": result_torch_dtype,
        "output_torch_dtype": output_torch_dtype,
        "scale_kind": scale_kind,
        "scale": scale,
        "expected_kind": expected_kind,
    }


_DTYPE_CASES = [
    _dtype_case(
        "fp32_to_fp16_plain",
        "fp32",
        pl.DT_FP16,
        pl.DT_FP32,
        torch.float16,
        torch.float32,
        scale_kind="none",
    ),
    _dtype_case(
        "fp32_to_bf16_plain",
        "fp32",
        pl.DT_BF16,
        pl.DT_FP32,
        torch.bfloat16,
        torch.float32,
        scale_kind="none",
    ),
    _dtype_case("fp32_to_fp16_scalar", "fp32", pl.DT_FP16, pl.DT_FP32, torch.float16, torch.float32),
    _dtype_case("fp32_to_uint8_scalar", "fp32", pl.DT_UINT8, pl.DT_INT32, torch.uint8, torch.int32),
    _dtype_case("fp32_to_bf16_scalar", "fp32", pl.DT_BF16, pl.DT_FP32, torch.bfloat16, torch.float32),
    _dtype_case(
        "fp32_to_hf8_scalar",
        "fp32",
        pl.DT_HF8,
        pl.DT_FP32,
        torch.uint8,
        torch.float32,
        scale=1.0,
        expected_kind="hf8",
    ),
    _dtype_case(
        "fp32_to_fp8e4m3fn_scalar",
        "fp32",
        pl.DT_FP8E4M3FN,
        pl.DT_INT32,
        torch.float8_e4m3fn,
        torch.int32,
        expected_kind="raw_fp8",
    ),
    _dtype_case(
        "fp32_to_uint8_scaling",
        "fp32",
        pl.DT_UINT8,
        pl.DT_INT32,
        torch.uint8,
        torch.int32,
        scale_kind="scaling",
    ),
    _dtype_case(
        "fp32_to_fp16_scaling",
        "fp32",
        pl.DT_FP16,
        pl.DT_FP32,
        torch.float16,
        torch.float32,
        scale_kind="scaling",
    ),
    _dtype_case(
        "fp32_to_bf16_scaling",
        "fp32",
        pl.DT_BF16,
        pl.DT_FP32,
        torch.bfloat16,
        torch.float32,
        scale_kind="scaling",
    ),
    _dtype_case(
        "fp32_to_hf8_scaling",
        "fp32",
        pl.DT_HF8,
        pl.DT_FP32,
        torch.uint8,
        torch.float32,
        scale_kind="scaling",
        expected_kind="hf8",
    ),
    _dtype_case(
        "fp32_to_fp8e4m3fn_scaling",
        "fp32",
        pl.DT_FP8E4M3FN,
        pl.DT_INT32,
        torch.float8_e4m3fn,
        torch.int32,
        scale_kind="scaling",
        expected_kind="raw_fp8",
    ),
    _dtype_case("int32_to_int8_scalar", "int32", pl.DT_INT8, pl.DT_INT32, torch.int8, torch.int32),
    _dtype_case("int32_to_uint8_scalar", "int32", pl.DT_UINT8, pl.DT_INT32, torch.uint8, torch.int32),
    _dtype_case("int32_to_fp16_scalar", "int32", pl.DT_FP16, pl.DT_FP32, torch.float16, torch.float32),
    _dtype_case("int32_to_bf16_scalar", "int32", pl.DT_BF16, pl.DT_FP32, torch.bfloat16, torch.float32),
    _dtype_case(
        "int32_to_int8_scaling",
        "int32",
        pl.DT_INT8,
        pl.DT_INT32,
        torch.int8,
        torch.int32,
        scale_kind="scaling",
    ),
    _dtype_case(
        "int32_to_uint8_scaling",
        "int32",
        pl.DT_UINT8,
        pl.DT_INT32,
        torch.uint8,
        torch.int32,
        scale_kind="scaling",
    ),
    _dtype_case(
        "int32_to_fp16_scaling",
        "int32",
        pl.DT_FP16,
        pl.DT_FP32,
        torch.float16,
        torch.float32,
        scale_kind="scaling",
    ),
    _dtype_case(
        "int32_to_bf16_scaling",
        "int32",
        pl.DT_BF16,
        pl.DT_FP32,
        torch.bfloat16,
        torch.float32,
        scale_kind="scaling",
    ),
]


def _dtype_params(scale_kind):
    return [pytest.param(case, id=case["label"]) for case in _DTYPE_CASES if case["scale_kind"] == scale_kind]


_T01_CASES = [pytest.param(None, id="tiled_tail"), *_dtype_params("none")]
_T02_CASES = [pytest.param(None, id="n_tail"), *_dtype_params("scalar")]
_T03_CASES = [pytest.param(None, id="n_tail_relu"), *_dtype_params("scaling")]


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
@pytest.mark.parametrize("case", _T01_CASES)
def test_t01_l0c_to_l1_tmov(case):
    device = ST_DEVICE
    _require_a5(device)
    if case is not None:
        label, actual, expected, atol = _run_dtype_and_layout_case(case, device, "t01")
        torch.npu.synchronize()
        _assert_result(label, actual, expected, atol=atol)
        return

    m, n, k = 128, 96, 96
    values = (torch.arange(m * k, dtype=torch.float32).reshape(m, k) % 5 - 2).to(device)
    first_rhs = (torch.arange(k * n, dtype=torch.float32).reshape(k, n) % 5 - 2).to(device)
    second_rhs = torch.eye(n, dtype=torch.float32, device=device)
    out = torch.zeros((m, n), dtype=torch.float32, device=device)

    l0c_to_l1_tmov_kernel(values, first_rhs, second_rhs, out)
    torch.npu.synchronize()
    _assert_result("t01 TMOV", out, values @ first_rhs, atol=1e-3)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
@pytest.mark.parametrize("case", _T02_CASES)
def test_t02_l0c_to_l1_scalar_tmov(case):
    device = ST_DEVICE
    _require_a5(device)
    if case is not None:
        label, actual, expected, atol = _run_dtype_and_layout_case(case, device, "t02")
        torch.npu.synchronize()
        _assert_result(label, actual, expected, atol=atol)
        return

    m, n, k = 64, 32, 64
    scale = 0.5
    values = (torch.arange(m * k, dtype=torch.float32).reshape(m, k) % 17 - 8).to(device)
    identity_fp32 = torch.eye(k, n, dtype=torch.float32, device=device)
    identity_int8 = torch.eye(n, dtype=torch.int8, device=device)
    out = torch.zeros((m, n), dtype=torch.int32, device=device)

    l0c_to_l1_scalar_tmov_kernel(values, identity_fp32, identity_int8, out, _scale_bits(scale))
    expected = torch.clamp(torch.round(values[:, :n] * scale), -128, 127).to(torch.int32)
    torch.npu.synchronize()
    _assert_result("t02 scalar TMOV N-tail", out, expected, atol=1)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
@pytest.mark.parametrize("case", _T03_CASES)
def test_t03_l0c_to_l1_scaling_tmov(case):
    device = ST_DEVICE
    _require_a5(device)
    if case is not None:
        label, actual, expected, atol = _run_dtype_and_layout_case(case, device, "t03")
        torch.npu.synchronize()
        _assert_result(label, actual, expected, atol=atol)
        return

    m, n, k = 64, 32, 64
    values = (torch.arange(m * k, dtype=torch.float32).reshape(m, k) % 17 - 8).to(device)
    identity_fp32 = torch.eye(k, n, dtype=torch.float32, device=device)
    identity_int8 = torch.eye(n, dtype=torch.int8, device=device)
    scales = torch.linspace(0.25, 1.0, n, dtype=torch.float32)
    fp_params = _scaling_params(scales, device)
    out = torch.zeros((m, n), dtype=torch.int32, device=device)

    l0c_to_l1_scaling_tmov_kernel(values, identity_fp32, identity_int8, fp_params, out)
    expected = torch.clamp(torch.round(torch.relu(values[:, :n]) * scales.to(device)), -128, 127).to(torch.int32)
    torch.npu.synchronize()
    _assert_result("t03 Scaling Tile TMOV + ReLU N-tail", out, expected, atol=1)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t04_l0c_to_l1_extract():
    device = ST_DEVICE
    _require_a5(device)
    scale = 0.5
    values = (torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64) % 17 - 8).to(device)
    identity_fp32 = torch.eye(64, dtype=torch.float32, device=device)
    identity_int8 = torch.eye(32, dtype=torch.int8, device=device)
    window_scales = torch.linspace(0.25, 1.0, 32, dtype=torch.float32)
    scales = torch.ones(64, dtype=torch.float32)
    scales[16:48] = window_scales
    fp_params = _scaling_params(scales, device)
    scalar_scale_out = torch.zeros((32, 32), dtype=torch.int32, device=device)
    scaling_tile_out = torch.zeros((32, 32), dtype=torch.int32, device=device)

    l0c_to_l1_extract_kernel(
        values,
        identity_fp32,
        identity_int8,
        fp_params,
        scalar_scale_out,
        scaling_tile_out,
        _scale_bits(scale),
    )
    torch.npu.synchronize()

    window = values[16:48, 16:48]
    scalar_scale_expected = torch.clamp(torch.round(window * scale), -128, 127).to(torch.int32)
    scaling_tile_expected = torch.clamp(
        torch.round(torch.relu(window) * window_scales.to(device)), -128, 127
    ).to(torch.int32)
    _assert_result("t04 TEXTRACT scalar tail", scalar_scale_out, scalar_scale_expected, atol=1)
    _assert_result("t04 TEXTRACT Scaling Tile + ReLU tail", scaling_tile_out, scaling_tile_expected, atol=1)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t05_l0c_to_l1_insert():
    device = ST_DEVICE
    _require_a5(device)
    values = (torch.arange(16 * 64, dtype=torch.float32).reshape(16, 64) % 17 - 8).to(device)
    first_rhs = torch.eye(64, 32, dtype=torch.float32, device=device)
    seed = torch.full((64, 64), -7, dtype=torch.int8, device=device)
    second_rhs = torch.eye(64, dtype=torch.int8, device=device)
    scales = torch.linspace(0.25, 1.0, 32, dtype=torch.float32)
    fp_params = _scaling_params(scales, device)
    scalar_scale_out = torch.zeros((64, 64), dtype=torch.int32, device=device)
    scaling_tile_out = torch.zeros((64, 64), dtype=torch.int32, device=device)

    l0c_to_l1_insert_kernel(
        values,
        first_rhs,
        seed,
        second_rhs,
        fp_params,
        scalar_scale_out,
        scaling_tile_out,
        _scale_bits(0.5),
    )
    torch.npu.synchronize()

    scalar_scale_expected = seed.to(torch.int32)
    scalar_scale_expected[16:32, 32:64] = torch.clamp(torch.round(values[:, :32] * 0.5), -128, 127).to(
        torch.int32
    )
    scaling_tile_expected = seed.to(torch.int32)
    scaling_tile_expected[16:32, 32:64] = torch.clamp(
        torch.round(torch.relu(values[:, :32]) * scales.to(device)), -128, 127
    ).to(torch.int32)
    _assert_result("t05 automatic scalar TINSERT tail", scalar_scale_out, scalar_scale_expected, atol=1)
    _assert_result("t05 explicit Scaling Tile TINSERT + ReLU tail", scaling_tile_out, scaling_tile_expected, atol=1)


def _run_dtype_and_layout_case(case, device, test_id):
    scale_kind = case["scale_kind"]
    scale = case["scale"]
    m, n, k = 64, 32, 64

    if case["source"] == "int32":
        input_dtype = pl.DT_INT8
        acc_dtype = pl.DT_INT32
        values = (torch.arange(m * k, dtype=torch.int32).reshape(m, k) % 17 - 8).to(torch.int8).to(device)
        first_rhs = torch.eye(k, n, dtype=torch.int8, device=device)
        first_acc = values.to(torch.int32)[:, :n]
    else:
        input_dtype = pl.DT_FP32
        acc_dtype = pl.DT_FP32
        values = torch.arange(m * k, dtype=torch.float32).reshape(m, k)
        if case["expected_kind"] in ("hf8", "raw_fp8"):
            values = values % 3 - 1
        else:
            values = (values % 17 - 8) / 8
        values = values.to(device)
        first_rhs = torch.eye(k, n, dtype=torch.float32, device=device)
        first_acc = values[:, :n]

    if case["expected_kind"] == "hf8":
        second_rhs = (torch.eye(n, dtype=torch.uint8) * 0x08).to(device)
    elif case["result_torch_dtype"] in (torch.uint8, torch.float8_e4m3fn):
        second_rhs = torch.eye(n, dtype=torch.int8, device=device)
    else:
        second_rhs = torch.eye(n, dtype=case["result_torch_dtype"], device=device)

    signed_int8 = case["result_torch_dtype"] == torch.int8
    scales = torch.full((n,), scale, dtype=torch.float32)
    fp_params = _scaling_params(scales, device, signed_int8=signed_int8)
    out = torch.zeros((m, n), dtype=case["output_torch_dtype"], device=device)
    kernel = _make_l0c_to_l1_dtype_kernel(
        f"l0c_to_l1_{case['label']}",
        input_dtype,
        acc_dtype,
        case["result_dtype"],
        case["output_dtype"],
        scale_value=scale if scale_kind == "scalar" else None,
        use_scaling_tile=scale_kind == "scaling",
    )

    kernel(values, first_rhs, second_rhs, fp_params, out)

    scaled = first_acc if scale_kind == "none" else first_acc * scale
    if case["expected_kind"] == "hf8":
        expected = scaled.to(torch.float32)
        atol = 1e-3
    elif case["expected_kind"] == "raw_fp8":
        expected = scaled.cpu().to(torch.float8_e4m3fn).view(torch.int8).to(torch.int32)
        atol = 1
    elif case["result_torch_dtype"] in (torch.int8, torch.uint8):
        min_value, max_value = (-128, 127) if case["result_torch_dtype"] == torch.int8 else (0, 255)
        expected = torch.clamp(torch.round(scaled), min_value, max_value).to(torch.int32)
        atol = 1
    else:
        expected = scaled.to(case["result_torch_dtype"]).to(case["output_torch_dtype"])
        atol = 1e-3
    return f"{test_id} {case['label']}", out, expected, atol


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pytest.main([__file__, "-v", "-s"])
