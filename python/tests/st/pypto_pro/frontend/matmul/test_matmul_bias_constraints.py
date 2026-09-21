#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------

"""matmul bias constraint and error tests.

Covers:
  - bias tile dtype errors (FP16, BF16 — must be FP32)
  - bias tile memory space errors (Right, Vec, Mat — must be Bias)
  - bias direct load error (load only supports Vec/Mat)
  - bias shape error ([M,N] instead of [1,N])
  - K-split constraint: matmul(bias) on middle/last block (acc overwritten)
  - phase mismatch: matmul(Final) + store without STPhase
  - phase deadlock: all Partial, no Final
  - illegal move paths: Left→Bias, Acc→Bias, Vec→Bias
"""

import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

TILE = 128
K_SPLIT = 384
DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
DEVICE = f"npu:{DEVICE_ID}"


def _run_expect_error(kernel_func, *args):
    with pytest.raises((RuntimeError, Exception)):
        kernel_func(*args)
        torch.npu.synchronize()


# =============================================================================
# bias tile dtype errors (2 tests)
# =============================================================================


@pl.jit(auto_mutex=True)
def kernel_bias_dtype_fp16(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    with pl.section_cube():
        a_mat = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0,
            mutex_ids=[0, 1],
        )
        b_mat = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0x20000,
            mutex_ids=[2, 3],
        )
        bias_mat = pl.make_tile_group(
            type=pl.TileType(shape=[1, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0x40000,
            mutex_ids=[4, 5],
        )
        a_left = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
            addrs=0,
            mutex_ids=[6, 7],
        )
        b_right = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
            addrs=0,
            mutex_ids=[8, 9],
        )
        bias_l0b = pl.make_tile_group(
            type=pl.TileType(shape=[1, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Bias),
            addrs=0,
            mutex_ids=[10, 11],
        )
        acc = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
            addrs=0,
            mutex_ids=[12, 13],
        )
        a_l1 = a_mat.next()
        pl.load_tile(a_l1, a, [0, 0])
        b_l1 = b_mat.next()
        pl.load_tile(b_l1, b, [0, 0])
        bias_l1 = bias_mat.next()
        pl.load_tile(bias_l1, bias, [0, 0])
        cur_a = a_left.next()
        pl.move(cur_a, a_l1)
        cur_b = b_right.next()
        pl.move(cur_b, b_l1)
        cur_bias = bias_l0b.next()
        pl.move(cur_bias, bias_l1)
        ac = acc.next()
        pl.matmul(ac, cur_a, cur_b, cur_bias)
        pl.store_tile(out, ac, [0, 0])


@pl.jit(auto_mutex=True)
def kernel_bias_dtype_bf16(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
):
    with pl.section_cube():
        a_mat = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat),
            addrs=0,
            mutex_ids=[0, 1],
        )
        b_mat = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat),
            addrs=0x20000,
            mutex_ids=[2, 3],
        )
        bias_mat = pl.make_tile_group(
            type=pl.TileType(shape=[1, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0x40000,
            mutex_ids=[4, 5],
        )
        a_left = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left),
            addrs=0,
            mutex_ids=[6, 7],
        )
        b_right = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right),
            addrs=0,
            mutex_ids=[8, 9],
        )
        bias_l0b = pl.make_tile_group(
            type=pl.TileType(shape=[1, TILE], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Bias),
            addrs=0,
            mutex_ids=[10, 11],
        )
        acc = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
            addrs=0,
            mutex_ids=[12, 13],
        )
        a_l1 = a_mat.next()
        pl.load_tile(a_l1, a, [0, 0])
        b_l1 = b_mat.next()
        pl.load_tile(b_l1, b, [0, 0])
        bias_l1 = bias_mat.next()
        pl.load_tile(bias_l1, bias, [0, 0])
        cur_a = a_left.next()
        pl.move(cur_a, a_l1)
        cur_b = b_right.next()
        pl.move(cur_b, b_l1)
        cur_bias = bias_l0b.next()
        pl.move(cur_bias, bias_l1)
        ac = acc.next()
        pl.matmul(ac, cur_a, cur_b, cur_bias)
        pl.store_tile(out, ac, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_bias_dtype_fp16():
    torch.npu.set_device(DEVICE_ID)
    a = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.float16)
    b = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(1, TILE, device=DEVICE, dtype=torch.float16)
    out = torch.zeros(TILE, TILE, device=DEVICE, dtype=torch.float16)
    _run_expect_error(kernel_bias_dtype_fp16, a, b, bias, out)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_bias_dtype_bf16():
    torch.npu.set_device(DEVICE_ID)
    a = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.bfloat16)
    bias = torch.randn(1, TILE, device=DEVICE, dtype=torch.bfloat16)
    out = torch.zeros(TILE, TILE, device=DEVICE, dtype=torch.bfloat16)
    _run_expect_error(kernel_bias_dtype_bf16, a, b, bias, out)


# =============================================================================
# bias tile memory space errors (3 tests)
# =============================================================================


@pl.jit(auto_mutex=True)
def kernel_bias_mem_right(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    with pl.section_cube():
        a_mat = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0,
            mutex_ids=[0, 1],
        )
        b_mat = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0x20000,
            mutex_ids=[2, 3],
        )
        bias_mat = pl.make_tile_group(
            type=pl.TileType(shape=[1, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0x40000,
            mutex_ids=[4, 5],
        )
        a_left = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
            addrs=0,
            mutex_ids=[6, 7],
        )
        b_right = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
            addrs=0,
            mutex_ids=[8, 9],
        )
        bias_wrong = pl.make_tile_group(
            type=pl.TileType(shape=[1, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right),
            addrs=0,
            mutex_ids=[10, 11],
        )
        acc = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
            addrs=0,
            mutex_ids=[12, 13],
        )
        a_l1 = a_mat.next()
        pl.load_tile(a_l1, a, [0, 0])
        b_l1 = b_mat.next()
        pl.load_tile(b_l1, b, [0, 0])
        bias_l1 = bias_mat.next()
        pl.load_tile(bias_l1, bias, [0, 0])
        cur_a = a_left.next()
        pl.move(cur_a, a_l1)
        cur_b = b_right.next()
        pl.move(cur_b, b_l1)
        cur_bias = bias_wrong.next()
        pl.move(cur_bias, bias_l1)
        ac = acc.next()
        pl.matmul(ac, cur_a, cur_b, cur_bias)
        pl.store_tile(out, ac, [0, 0])


@pl.jit(auto_mutex=True)
def kernel_bias_mem_vec(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    with pl.section_cube():
        a_mat = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0,
            mutex_ids=[0, 1],
        )
        b_mat = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0x20000,
            mutex_ids=[2, 3],
        )
        bias_mat = pl.make_tile_group(
            type=pl.TileType(shape=[1, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0x40000,
            mutex_ids=[4, 5],
        )
        a_left = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
            addrs=0,
            mutex_ids=[6, 7],
        )
        b_right = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
            addrs=0,
            mutex_ids=[8, 9],
        )
        bias_wrong = pl.make_tile_group(
            type=pl.TileType(shape=[1, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0,
            mutex_ids=[10, 11],
        )
        acc = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
            addrs=0,
            mutex_ids=[12, 13],
        )
        a_l1 = a_mat.next()
        pl.load_tile(a_l1, a, [0, 0])
        b_l1 = b_mat.next()
        pl.load_tile(b_l1, b, [0, 0])
        bias_l1 = bias_mat.next()
        pl.load_tile(bias_l1, bias, [0, 0])
        cur_a = a_left.next()
        pl.move(cur_a, a_l1)
        cur_b = b_right.next()
        pl.move(cur_b, b_l1)
        cur_bias = bias_wrong.next()
        pl.move(cur_bias, bias_l1)
        ac = acc.next()
        pl.matmul(ac, cur_a, cur_b, cur_bias)
        pl.store_tile(out, ac, [0, 0])


@pl.jit(auto_mutex=True)
def kernel_bias_mem_mat(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    with pl.section_cube():
        a_mat = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0,
            mutex_ids=[0, 1],
        )
        b_mat = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0x20000,
            mutex_ids=[2, 3],
        )
        bias_mat = pl.make_tile_group(
            type=pl.TileType(shape=[1, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0x40000,
            mutex_ids=[4, 5],
        )
        a_left = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
            addrs=0,
            mutex_ids=[6, 7],
        )
        b_right = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
            addrs=0,
            mutex_ids=[8, 9],
        )
        bias_wrong = pl.make_tile_group(
            type=pl.TileType(shape=[1, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat),
            addrs=0,
            mutex_ids=[10, 11],
        )
        acc = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
            addrs=0,
            mutex_ids=[12, 13],
        )
        a_l1 = a_mat.next()
        pl.load_tile(a_l1, a, [0, 0])
        b_l1 = b_mat.next()
        pl.load_tile(b_l1, b, [0, 0])
        bias_l1 = bias_mat.next()
        pl.load_tile(bias_l1, bias, [0, 0])
        cur_a = a_left.next()
        pl.move(cur_a, a_l1)
        cur_b = b_right.next()
        pl.move(cur_b, b_l1)
        cur_bias = bias_wrong.next()
        pl.move(cur_bias, bias_l1)
        ac = acc.next()
        pl.matmul(ac, cur_a, cur_b, cur_bias)
        pl.store_tile(out, ac, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_bias_mem_right():
    torch.npu.set_device(DEVICE_ID)
    a = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.float16)
    b = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(1, TILE, device=DEVICE, dtype=torch.float16)
    out = torch.zeros(TILE, TILE, device=DEVICE, dtype=torch.float16)
    _run_expect_error(kernel_bias_mem_right, a, b, bias, out)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_bias_mem_vec():
    torch.npu.set_device(DEVICE_ID)
    a = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.float16)
    b = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(1, TILE, device=DEVICE, dtype=torch.float16)
    out = torch.zeros(TILE, TILE, device=DEVICE, dtype=torch.float16)
    _run_expect_error(kernel_bias_mem_vec, a, b, bias, out)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_bias_mem_mat():
    torch.npu.set_device(DEVICE_ID)
    a = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.float16)
    b = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(1, TILE, device=DEVICE, dtype=torch.float16)
    out = torch.zeros(TILE, TILE, device=DEVICE, dtype=torch.float16)
    _run_expect_error(kernel_bias_mem_mat, a, b, bias, out)


# =============================================================================
# bias direct load error (1 test)
# =============================================================================


@pl.jit(auto_mutex=True)
def kernel_bias_load_direct(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    with pl.section_cube():
        a_mat = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0,
            mutex_ids=[0, 1],
        )
        b_mat = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0x20000,
            mutex_ids=[2, 3],
        )
        a_left = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
            addrs=0,
            mutex_ids=[6, 7],
        )
        b_right = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
            addrs=0,
            mutex_ids=[8, 9],
        )
        bias_l0b = pl.make_tile_group(
            type=pl.TileType(shape=[1, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Bias),
            addrs=0,
            mutex_ids=[10, 11],
        )
        acc = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
            addrs=0,
            mutex_ids=[12, 13],
        )
        a_l1 = a_mat.next()
        pl.load_tile(a_l1, a, [0, 0])
        b_l1 = b_mat.next()
        pl.load_tile(b_l1, b, [0, 0])
        pl.load(bias_l0b.next(), bias, [0, 0])
        cur_a = a_left.next()
        pl.move(cur_a, a_l1)
        cur_b = b_right.next()
        pl.move(cur_b, b_l1)
        ac = acc.next()
        pl.matmul(ac, cur_a, cur_b, bias_l0b.current())
        pl.store_tile(out, ac, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_bias_load_direct():
    torch.npu.set_device(DEVICE_ID)
    a = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.float16)
    b = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(1, TILE, device=DEVICE, dtype=torch.float16)
    out = torch.zeros(TILE, TILE, device=DEVICE, dtype=torch.float16)
    _run_expect_error(kernel_bias_load_direct, a, b, bias, out)


# =============================================================================
# bias shape error (1 test)
# =============================================================================


@pl.jit(auto_mutex=True)
def kernel_bias_shape_full_2d(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    with pl.section_cube():
        a_mat = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0,
            mutex_ids=[0, 1],
        )
        b_mat = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0x20000,
            mutex_ids=[2, 3],
        )
        bias_mat = pl.make_tile_group(
            type=pl.TileType(shape=[1, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0x40000,
            mutex_ids=[4, 5],
        )
        a_left = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
            addrs=0,
            mutex_ids=[6, 7],
        )
        b_right = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
            addrs=0,
            mutex_ids=[8, 9],
        )
        bias_l0b = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Bias),
            addrs=0,
            mutex_ids=[10, 11],
        )
        acc = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
            addrs=0,
            mutex_ids=[12, 13],
        )
        a_l1 = a_mat.next()
        pl.load_tile(a_l1, a, [0, 0])
        b_l1 = b_mat.next()
        pl.load_tile(b_l1, b, [0, 0])
        bias_l1 = bias_mat.next()
        pl.load_tile(bias_l1, bias, [0, 0])
        cur_a = a_left.next()
        pl.move(cur_a, a_l1)
        cur_b = b_right.next()
        pl.move(cur_b, b_l1)
        cur_bias = bias_l0b.next()
        pl.move(cur_bias, bias_l1)
        ac = acc.next()
        pl.matmul(ac, cur_a, cur_b, cur_bias)
        pl.store_tile(out, ac, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_bias_shape_full_2d():
    torch.npu.set_device(DEVICE_ID)
    a = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.float16)
    b = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(1, TILE, device=DEVICE, dtype=torch.float16)
    out = torch.zeros(TILE, TILE, device=DEVICE, dtype=torch.float16)
    _run_expect_error(kernel_bias_shape_full_2d, a, b, bias, out)


# =============================================================================
# K-split constraint: matmul(bias) on non-first block (2 tests)
# =============================================================================


@pl.jit(auto_mutex=True)
def kernel_bias_middle_block(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    with pl.section_cube():
        pl.system.set_mm_layout_transform(enabled=True)
        a_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0x00000,
            mutex_ids=[0, 1],
        )
        b_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0x10000,
            mutex_ids=[2, 3],
        )
        bias_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[1, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0x20000,
            mutex_ids=[4, 5],
        )
        a_left = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=0x0000,
            mutex_ids=[6, 7],
        )
        b_right = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
            addrs=0x0000,
            mutex_ids=[8, 9],
        )
        bias_l0b = pl.make_tile_group(
            type=pl.TileType(shape=[1, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Bias),
            addrs=0x0000,
            mutex_ids=[10, 11],
        )
        acc = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, fractal=1024),
            addrs=0x0000,
            mutex_ids=[12],
        )
        ac = acc.current()
        for k in pl.range(0, K_SPLIT, TILE):
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_left.next()
            br = b_right.next()
            pl.load(cur_a, a, [0, k])
            pl.load(cur_b, b, [k, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if k == 0:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
            elif k < K_SPLIT - TILE:
                bias_l1_tile = bias_l1.next()
                pl.load(bias_l1_tile, bias, [0, 0])
                bl = bias_l0b.next()
                pl.move(bl, bias_l1_tile)
                pl.matmul(ac, al, br, bl, phase=pl.AccPhase.Partial)
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
        pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)
        pl.system.set_mm_layout_transform(enabled=False)


@pl.jit(auto_mutex=True)
def kernel_bias_last_block(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    with pl.section_cube():
        pl.system.set_mm_layout_transform(enabled=True)
        a_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0x00000,
            mutex_ids=[0, 1],
        )
        b_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0x10000,
            mutex_ids=[2, 3],
        )
        bias_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[1, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0x20000,
            mutex_ids=[4, 5],
        )
        a_left = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=0x0000,
            mutex_ids=[6, 7],
        )
        b_right = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
            addrs=0x0000,
            mutex_ids=[8, 9],
        )
        bias_l0b = pl.make_tile_group(
            type=pl.TileType(shape=[1, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Bias),
            addrs=0x0000,
            mutex_ids=[10, 11],
        )
        acc = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, fractal=1024),
            addrs=0x0000,
            mutex_ids=[12],
        )
        ac = acc.current()
        for k in pl.range(0, K_SPLIT, TILE):
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_left.next()
            br = b_right.next()
            pl.load(cur_a, a, [0, k])
            pl.load(cur_b, b, [k, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if k == 0:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
            elif k < K_SPLIT - TILE:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
            else:
                bias_l1_tile = bias_l1.next()
                pl.load(bias_l1_tile, bias, [0, 0])
                bl = bias_l0b.next()
                pl.move(bl, bias_l1_tile)
                pl.matmul(ac, al, br, bl, phase=pl.AccPhase.Final)
        pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)
        pl.system.set_mm_layout_transform(enabled=False)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_matmul_bias_middle_block():
    torch.npu.set_device(DEVICE_ID)
    torch.manual_seed(42)
    a = torch.randn(TILE, K_SPLIT, device=DEVICE, dtype=torch.float16)
    b = torch.randn(K_SPLIT, TILE, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(1, TILE, device=DEVICE, dtype=torch.float16)
    out = torch.zeros(TILE, TILE, device=DEVICE, dtype=torch.float16)
    kernel_bias_middle_block(a, b, bias, out)
    torch.npu.synchronize()
    golden = torch.matmul(a.float(), b.float()).half() + bias
    max_diff = (out.float() - golden.float()).abs().max().item()
    assert max_diff > 1.0, f"Expected wrong result (max_diff > 1.0), got {max_diff}"


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_matmul_bias_last_block():
    torch.npu.set_device(DEVICE_ID)
    torch.manual_seed(42)
    a = torch.randn(TILE, K_SPLIT, device=DEVICE, dtype=torch.float16)
    b = torch.randn(K_SPLIT, TILE, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(1, TILE, device=DEVICE, dtype=torch.float16)
    out = torch.zeros(TILE, TILE, device=DEVICE, dtype=torch.float16)
    kernel_bias_last_block(a, b, bias, out)
    torch.npu.synchronize()
    golden = torch.matmul(a.float(), b.float()).half() + bias
    max_diff = (out.float() - golden.float()).abs().max().item()
    assert max_diff > 1.0, f"Expected wrong result (max_diff > 1.0), got {max_diff}"


# =============================================================================
# Illegal move paths to Bias (3 tests)
# =============================================================================


@pl.jit(auto_mutex=True)
def kernel_move_left_to_bias(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    with pl.section_cube():
        a_mat = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0,
            mutex_ids=[0, 1],
        )
        b_mat = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0x20000,
            mutex_ids=[2, 3],
        )
        bias_mat = pl.make_tile_group(
            type=pl.TileType(shape=[1, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0x40000,
            mutex_ids=[4, 5],
        )
        a_left = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
            addrs=0,
            mutex_ids=[6, 7],
        )
        b_right = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
            addrs=0,
            mutex_ids=[8, 9],
        )
        bias_l0b = pl.make_tile_group(
            type=pl.TileType(shape=[1, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Bias),
            addrs=0,
            mutex_ids=[10, 11],
        )
        acc = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
            addrs=0,
            mutex_ids=[12, 13],
        )
        a_l1 = a_mat.next()
        pl.load_tile(a_l1, a, [0, 0])
        b_l1 = b_mat.next()
        pl.load_tile(b_l1, b, [0, 0])
        bias_l1 = bias_mat.next()
        pl.load_tile(bias_l1, bias, [0, 0])
        cur_a = a_left.next()
        pl.move(cur_a, a_l1)
        cur_b = b_right.next()
        pl.move(cur_b, b_l1)
        cur_bias = bias_l0b.next()
        pl.move(cur_bias, cur_a)
        ac = acc.next()
        pl.matmul(ac, cur_a, cur_b, cur_bias)
        pl.store_tile(out, ac, [0, 0])


@pl.jit(auto_mutex=True)
def kernel_move_acc_to_bias(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    with pl.section_cube():
        a_mat = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0,
            mutex_ids=[0, 1],
        )
        b_mat = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0x20000,
            mutex_ids=[2, 3],
        )
        bias_mat = pl.make_tile_group(
            type=pl.TileType(shape=[1, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0x40000,
            mutex_ids=[4, 5],
        )
        a_left = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
            addrs=0,
            mutex_ids=[6, 7],
        )
        b_right = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
            addrs=0,
            mutex_ids=[8, 9],
        )
        bias_l0b = pl.make_tile_group(
            type=pl.TileType(shape=[1, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Bias),
            addrs=0,
            mutex_ids=[10, 11],
        )
        acc = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
            addrs=0,
            mutex_ids=[12, 13],
        )
        a_l1 = a_mat.next()
        pl.load_tile(a_l1, a, [0, 0])
        b_l1 = b_mat.next()
        pl.load_tile(b_l1, b, [0, 0])
        bias_l1 = bias_mat.next()
        pl.load_tile(bias_l1, bias, [0, 0])
        cur_a = a_left.next()
        pl.move(cur_a, a_l1)
        cur_b = b_right.next()
        pl.move(cur_b, b_l1)
        ac = acc.next()
        pl.matmul(ac, cur_a, cur_b)
        cur_bias = bias_l0b.next()
        pl.move(cur_bias, ac)
        pl.store_tile(out, ac, [0, 0])


@pl.jit(auto_mutex=True)
def kernel_move_vec_to_bias(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    with pl.section_cube():
        a_mat = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0,
            mutex_ids=[0, 1],
        )
        b_mat = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0x20000,
            mutex_ids=[2, 3],
        )
        bias_mat = pl.make_tile_group(  # noqa: F841
            type=pl.TileType(shape=[1, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0x40000,
            mutex_ids=[4, 5],
        )
        a_left = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
            addrs=0,
            mutex_ids=[6, 7],
        )
        b_right = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
            addrs=0,
            mutex_ids=[8, 9],
        )
        bias_l0b = pl.make_tile_group(
            type=pl.TileType(shape=[1, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Bias),
            addrs=0,
            mutex_ids=[10, 11],
        )
        acc = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
            addrs=0,
            mutex_ids=[12, 13],
        )
        bias_vec = pl.make_tile_group(
            type=pl.TileType(shape=[1, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0,
            mutex_ids=[14, 15],
        )
        a_l1 = a_mat.next()
        pl.load_tile(a_l1, a, [0, 0])
        b_l1 = b_mat.next()
        pl.load_tile(b_l1, b, [0, 0])
        cur_a = a_left.next()
        pl.move(cur_a, a_l1)
        cur_b = b_right.next()
        pl.move(cur_b, b_l1)
        bias_v = bias_vec.next()
        pl.load_tile(bias_v, bias, [0, 0])
        cur_bias = bias_l0b.next()
        pl.move(cur_bias, bias_v)
        ac = acc.next()
        pl.matmul(ac, cur_a, cur_b, cur_bias)
        pl.store_tile(out, ac, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_move_left_to_bias():
    torch.npu.set_device(DEVICE_ID)
    a = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.float16)
    b = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(1, TILE, device=DEVICE, dtype=torch.float16)
    out = torch.zeros(TILE, TILE, device=DEVICE, dtype=torch.float16)
    _run_expect_error(kernel_move_left_to_bias, a, b, bias, out)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_move_acc_to_bias():
    torch.npu.set_device(DEVICE_ID)
    a = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.float16)
    b = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(1, TILE, device=DEVICE, dtype=torch.float16)
    out = torch.zeros(TILE, TILE, device=DEVICE, dtype=torch.float16)
    _run_expect_error(kernel_move_acc_to_bias, a, b, bias, out)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_move_vec_to_bias():
    torch.npu.set_device(DEVICE_ID)
    a = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.float16)
    b = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(1, TILE, device=DEVICE, dtype=torch.float16)
    out = torch.zeros(TILE, TILE, device=DEVICE, dtype=torch.float16)
    _run_expect_error(kernel_move_vec_to_bias, a, b, bias, out)
