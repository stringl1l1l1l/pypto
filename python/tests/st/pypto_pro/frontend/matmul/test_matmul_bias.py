#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""matmul bias tests: basic dtype, parameter styles, M/N/K split, move conversion, regression.

Covers:
  - FP16/BF16 basic correctness + dynamic M multi-tile broadcast
  - 5 parameter passing combos (positional/no-bias/with-phase) + keyword rejection
  - N-dim multi-tile (2 tiles, 2x2 tiles) + M=4 tiles
  - K-split (2/3/4 blocks) + M/N/K three-way split (K=2/K=3)
  - phase=Final single call + STPhase.Final pairing
  - move type conversion (FP16→FP32, BF16→FP32)
  - regression: matmul/matmul_acc without bias unaffected
  - tail-block valid_shape: M/N/K tail (tail=72) + MNK combined tail with bias
"""

import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

TILE = 128
K_SPLIT_3 = 384
K_SPLIT_2 = 256
K_SPLIT_4 = 512
M_MNK = 256
N_MNK = 256
DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
DEVICE = f"npu:{DEVICE_ID}"


def _make_tile_groups(dtype=pl.DT_FP16):
    a_mat = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=dtype, target_memory=pl.MemorySpace.Mat), addrs=0, mutex_ids=[0, 1]
    )
    b_mat = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=dtype, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000,
        mutex_ids=[2, 3],
    )
    bias_mat = pl.make_tile_group(
        type=pl.TileType(shape=[1, TILE], dtype=dtype, target_memory=pl.MemorySpace.Mat),
        addrs=0x40000,
        mutex_ids=[4, 5],
    )
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=dtype, target_memory=pl.MemorySpace.Left), addrs=0, mutex_ids=[6, 7]
    )
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=dtype, target_memory=pl.MemorySpace.Right), addrs=0, mutex_ids=[8, 9]
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
    return a_mat, b_mat, bias_mat, a_left, b_right, bias_l0b, acc


def _load_and_move(a_mat, b_mat, bias_mat, a_left, b_right, bias_l0b, acc, a, b, bias):
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
    return cur_a, cur_b, cur_bias, acc.next()


def _make_k_split_tile_groups():
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
    return a_l1, b_l1, bias_l1, a_left, b_right, bias_l0b, acc


# =============================================================================
# Basic dtype tests (3 tests)
# =============================================================================


@pl.jit(auto_mutex=True)
def kernel_bias_fp16(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    with pl.section_cube():
        a_mat, b_mat, bias_mat, a_left, b_right, bias_l0b, acc = _make_tile_groups()
        cur_a, cur_b, cur_bias, ac = _load_and_move(a_mat, b_mat, bias_mat, a_left, b_right, bias_l0b, acc, a, b, bias)
        pl.matmul(ac, cur_a, cur_b, cur_bias)
        pl.store_tile(out, ac, [0, 0])


@pl.jit(auto_mutex=True)
def kernel_bias_bf16(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
):
    with pl.section_cube():
        a_mat, b_mat, bias_mat, a_left, b_right, bias_l0b, acc = _make_tile_groups(pl.DT_BF16)
        cur_a, cur_b, cur_bias, ac = _load_and_move(a_mat, b_mat, bias_mat, a_left, b_right, bias_l0b, acc, a, b, bias)
        pl.matmul(ac, cur_a, cur_b, cur_bias)
        pl.store_tile(out, ac, [0, 0])


@pl.jit(auto_mutex=True)
def kernel_bias_dynamic_m(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    m_tiles = a.shape[0] // TILE
    n_tiles = b.shape[1] // TILE
    with pl.section_cube():
        a_mat, b_mat, bias_mat, a_left, b_right, bias_l0b, acc = _make_tile_groups()
        for mi in pl.range(core_id, m_tiles, num_cores):
            for ni in pl.range(0, n_tiles, 1):
                a_l1 = a_mat.next()
                pl.load_tile(a_l1, a, [mi, 0])
                b_l1 = b_mat.next()
                pl.load_tile(b_l1, b, [0, ni])
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
                pl.store_tile(out, ac, [mi, ni])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_matmul_bias_fp16():
    """FP16 input + FP32 bias: basic correctness + Mat(FP16)→Bias(FP32) move type conversion."""
    torch.npu.set_device(DEVICE_ID)
    torch.manual_seed(42)
    a = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.float16)
    b = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(1, TILE, device=DEVICE, dtype=torch.float16)
    out = torch.zeros(TILE, TILE, device=DEVICE, dtype=torch.float16)
    kernel_bias_fp16(a, b, bias, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.matmul(a.float(), b.float()).half() + bias, rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_matmul_bias_bf16():
    """BF16 input + FP32 bias: basic correctness + Mat(BF16)→Bias(FP32) move type conversion."""
    torch.npu.set_device(DEVICE_ID)
    torch.manual_seed(42)
    a = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.bfloat16)
    bias = torch.randn(1, TILE, device=DEVICE, dtype=torch.bfloat16)
    out = torch.zeros(TILE, TILE, device=DEVICE, dtype=torch.bfloat16)
    kernel_bias_bf16(a, b, bias, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.matmul(a.float(), b.float()).to(torch.bfloat16) + bias, rtol=5e-2, atol=5e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_matmul_bias_dynamic_m():
    torch.npu.set_device(DEVICE_ID)
    torch.manual_seed(42)
    a = torch.randn(256, 128, device=DEVICE, dtype=torch.float16)
    b = torch.randn(128, 128, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(1, 128, device=DEVICE, dtype=torch.float16)
    out = torch.zeros(256, 128, device=DEVICE, dtype=torch.float16)
    kernel_bias_dynamic_m(a, b, bias, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.matmul(a.float(), b.float()).half() + bias, rtol=1e-2, atol=1e-2)


# =============================================================================
# Parameter passing styles (6 tests)
# =============================================================================


@pl.jit(auto_mutex=True)
def kernel_no_bias_positional(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    with pl.section_cube():
        a_mat, b_mat, bias_mat, a_left, b_right, bias_l0b, acc = _make_tile_groups()
        a_l1 = a_mat.next()
        pl.load_tile(a_l1, a, [0, 0])
        b_l1 = b_mat.next()
        pl.load_tile(b_l1, b, [0, 0])
        cur_a = a_left.next()
        pl.move(cur_a, a_l1)
        cur_b = b_right.next()
        pl.move(cur_b, b_l1)
        ac = acc.next()
        pl.matmul(ac, cur_a, cur_b)
        pl.store_tile(out, ac, [0, 0])


@pl.jit(auto_mutex=True)
def kernel_bias_positional(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    with pl.section_cube():
        a_mat, b_mat, bias_mat, a_left, b_right, bias_l0b, acc = _make_tile_groups()
        cur_a, cur_b, cur_bias, ac = _load_and_move(a_mat, b_mat, bias_mat, a_left, b_right, bias_l0b, acc, a, b, bias)
        pl.matmul(ac, cur_a, cur_b, cur_bias)
        pl.store_tile(out, ac, [0, 0])


@pl.jit(auto_mutex=True)
def kernel_bias_keyword(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    with pl.section_cube():
        a_mat, b_mat, bias_mat, a_left, b_right, bias_l0b, acc = _make_tile_groups()
        cur_a, cur_b, cur_bias, ac = _load_and_move(a_mat, b_mat, bias_mat, a_left, b_right, bias_l0b, acc, a, b, bias)
        pl.matmul(ac, cur_a, cur_b, bias_tile=cur_bias)
        pl.store_tile(out, ac, [0, 0])


def _run_expect_keyword_error(kernel):
    with pytest.raises(Exception):
        torch.npu.set_device(DEVICE_ID)
        torch.manual_seed(42)
        a = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.float16)
        b = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.float16)
        bias = torch.randn(1, TILE, device=DEVICE, dtype=torch.float16)
        out = torch.zeros(TILE, TILE, device=DEVICE, dtype=torch.float16)
        kernel(a, b, bias, out)
        torch.npu.synchronize()


@pl.jit(auto_mutex=True)
def kernel_no_bias_phase(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    with pl.section_cube():
        pl.system.set_mm_layout_transform(enabled=True)
        a_mat, b_mat, bias_mat, a_left, b_right, bias_l0b, acc = _make_tile_groups()
        a_l1 = a_mat.next()
        pl.load_tile(a_l1, a, [0, 0])
        b_l1 = b_mat.next()
        pl.load_tile(b_l1, b, [0, 0])
        cur_a = a_left.next()
        pl.move(cur_a, a_l1)
        cur_b = b_right.next()
        pl.move(cur_b, b_l1)
        ac = acc.current()
        pl.matmul(ac, cur_a, cur_b, phase=pl.AccPhase.Final)
        pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)
        pl.system.set_mm_layout_transform(enabled=False)


@pl.jit(auto_mutex=True)
def kernel_bias_positional_phase(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    with pl.section_cube():
        pl.system.set_mm_layout_transform(enabled=True)
        a_mat, b_mat, bias_mat, a_left, b_right, bias_l0b, acc = _make_tile_groups()
        cur_a, cur_b, cur_bias, _ = _load_and_move(a_mat, b_mat, bias_mat, a_left, b_right, bias_l0b, acc, a, b, bias)
        ac = acc.current()
        pl.matmul(ac, cur_a, cur_b, cur_bias, phase=pl.AccPhase.Final)
        pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)
        pl.system.set_mm_layout_transform(enabled=False)


@pl.jit(auto_mutex=True)
def kernel_bias_keyword_phase(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    with pl.section_cube():
        pl.system.set_mm_layout_transform(enabled=True)
        a_mat, b_mat, bias_mat, a_left, b_right, bias_l0b, acc = _make_tile_groups()
        cur_a, cur_b, cur_bias, _ = _load_and_move(a_mat, b_mat, bias_mat, a_left, b_right, bias_l0b, acc, a, b, bias)
        ac = acc.current()
        pl.matmul(ac, cur_a, cur_b, bias_tile=cur_bias, phase=pl.AccPhase.Final)
        pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)
        pl.system.set_mm_layout_transform(enabled=False)


def _run_single(kernel, has_bias):
    torch.npu.set_device(DEVICE_ID)
    torch.manual_seed(42)
    a = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.float16)
    b = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.float16)
    out = torch.zeros(TILE, TILE, device=DEVICE, dtype=torch.float16)
    if has_bias:
        bias = torch.randn(1, TILE, device=DEVICE, dtype=torch.float16)
        kernel(a, b, bias, out)
        torch.npu.synchronize()
        torch.testing.assert_close(out, torch.matmul(a.float(), b.float()).half() + bias, rtol=1e-2, atol=1e-2)
    else:
        kernel(a, b, out)
        torch.npu.synchronize()
        torch.testing.assert_close(out, torch.matmul(a.float(), b.float()).half(), rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_matmul_no_bias_positional():
    _run_single(kernel_no_bias_positional, has_bias=False)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_matmul_bias_positional():
    _run_single(kernel_bias_positional, has_bias=True)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_matmul_bias_keyword():
    """bias_tile keyword is rejected: only positional bias_tile is supported."""
    _run_expect_keyword_error(kernel_bias_keyword)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_matmul_no_bias_with_phase():
    _run_single(kernel_no_bias_phase, has_bias=False)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_matmul_bias_positional_with_phase():
    _run_single(kernel_bias_positional_phase, has_bias=True)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_matmul_bias_keyword_with_phase():
    """bias_tile keyword is rejected even with phase: only positional bias_tile is supported."""
    _run_expect_keyword_error(kernel_bias_keyword_phase)


# =============================================================================
# N-dim multi-tile (2 tests)
# =============================================================================


@pl.jit(auto_mutex=True)
def kernel_n_2tiles(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    n_tiles = b.shape[1] // TILE
    with pl.section_cube():
        a_mat, b_mat, bias_mat, a_left, b_right, bias_l0b, acc = _make_tile_groups()
        for ni in pl.range(0, n_tiles):
            a_l1 = a_mat.next()
            pl.load_tile(a_l1, a, [0, 0])
            b_l1 = b_mat.next()
            pl.load_tile(b_l1, b, [0, ni])
            bias_l1 = bias_mat.next()
            pl.load_tile(bias_l1, bias, [0, ni])
            cur_a = a_left.next()
            pl.move(cur_a, a_l1)
            cur_b = b_right.next()
            pl.move(cur_b, b_l1)
            cur_bias = bias_l0b.next()
            pl.move(cur_bias, bias_l1)
            ac = acc.next()
            pl.matmul(ac, cur_a, cur_b, cur_bias)
            pl.store_tile(out, ac, [0, ni])


@pl.jit(auto_mutex=True)
def kernel_mn_4tiles(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m_tiles = a.shape[0] // TILE
    n_tiles = b.shape[1] // TILE
    with pl.section_cube():
        a_mat, b_mat, bias_mat, a_left, b_right, bias_l0b, acc = _make_tile_groups()
        for mi in pl.range(0, m_tiles):
            for ni in pl.range(0, n_tiles):
                a_l1 = a_mat.next()
                pl.load_tile(a_l1, a, [mi, 0])
                b_l1 = b_mat.next()
                pl.load_tile(b_l1, b, [0, ni])
                bias_l1 = bias_mat.next()
                pl.load_tile(bias_l1, bias, [0, ni])
                cur_a = a_left.next()
                pl.move(cur_a, a_l1)
                cur_b = b_right.next()
                pl.move(cur_b, b_l1)
                cur_bias = bias_l0b.next()
                pl.move(cur_bias, bias_l1)
                ac = acc.next()
                pl.matmul(ac, cur_a, cur_b, cur_bias)
                pl.store_tile(out, ac, [mi, ni])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_matmul_bias_n_2tiles():
    torch.npu.set_device(DEVICE_ID)
    torch.manual_seed(42)
    a = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.float16)
    b = torch.randn(TILE, 256, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(1, 256, device=DEVICE, dtype=torch.float16)
    out = torch.zeros(TILE, 256, device=DEVICE, dtype=torch.float16)
    kernel_n_2tiles(a, b, bias, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.matmul(a.float(), b.float()).half() + bias, rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_matmul_bias_mn_4tiles():
    torch.npu.set_device(DEVICE_ID)
    torch.manual_seed(42)
    a = torch.randn(256, TILE, device=DEVICE, dtype=torch.float16)
    b = torch.randn(TILE, 256, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(1, 256, device=DEVICE, dtype=torch.float16)
    out = torch.zeros(256, 256, device=DEVICE, dtype=torch.float16)
    kernel_mn_4tiles(a, b, bias, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.matmul(a.float(), b.float()).half() + bias, rtol=1e-2, atol=1e-2)


# =============================================================================
# M=4 tiles (1 test)
# =============================================================================


@pl.jit(auto_mutex=True)
def kernel_m_4tiles(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    m_tiles = a.shape[0] // TILE
    with pl.section_cube():
        a_mat, b_mat, bias_mat, a_left, b_right, bias_l0b, acc = _make_tile_groups()
        for mi in pl.range(core_id, m_tiles, num_cores):
            a_l1 = a_mat.next()
            pl.load_tile(a_l1, a, [mi, 0])
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
            pl.store_tile(out, ac, [mi, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_matmul_bias_m_4tiles():
    torch.npu.set_device(DEVICE_ID)
    torch.manual_seed(42)
    a = torch.randn(512, 128, device=DEVICE, dtype=torch.float16)
    b = torch.randn(128, 128, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(1, 128, device=DEVICE, dtype=torch.float16)
    out = torch.zeros(512, 128, device=DEVICE, dtype=torch.float16)
    kernel_m_4tiles(a, b, bias, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.matmul(a.float(), b.float()).half() + bias, rtol=1e-2, atol=1e-2)


# =============================================================================
# K-split: K=2/3/4 blocks + M/N/K three-way (6 tests)
# =============================================================================


@pl.jit(auto_mutex=True)
def kernel_k_split_bias(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    with pl.section_cube():
        pl.system.set_mm_layout_transform(enabled=True)
        a_l1, b_l1, bias_l1, a_left, b_right, bias_l0b, acc = _make_k_split_tile_groups()
        ac = acc.current()
        for k in pl.range(0, K_SPLIT_3, TILE):
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_left.next()
            br = b_right.next()
            pl.load(cur_a, a, [0, k])
            pl.load(cur_b, b, [k, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if k == 0:
                bias_l1_tile = bias_l1.next()
                pl.load(bias_l1_tile, bias, [0, 0])
                bl = bias_l0b.next()
                pl.move(bl, bias_l1_tile)
                pl.matmul(ac, al, br, bl, phase=pl.AccPhase.Partial)
            elif k < K_SPLIT_3 - TILE:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
        pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)
        pl.system.set_mm_layout_transform(enabled=False)


@pl.jit(auto_mutex=True)
def kernel_k2_split_bias(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    with pl.section_cube():
        pl.system.set_mm_layout_transform(enabled=True)
        a_l1, b_l1, bias_l1, a_left, b_right, bias_l0b, acc = _make_k_split_tile_groups()
        ac = acc.current()
        for k in pl.range(0, K_SPLIT_2, TILE):
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_left.next()
            br = b_right.next()
            pl.load(cur_a, a, [0, k])
            pl.load(cur_b, b, [k, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if k == 0:
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
def kernel_k4_split_bias(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    with pl.section_cube():
        pl.system.set_mm_layout_transform(enabled=True)
        a_l1, b_l1, bias_l1, a_left, b_right, bias_l0b, acc = _make_k_split_tile_groups()
        ac = acc.current()
        for k in pl.range(0, K_SPLIT_4, TILE):
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_left.next()
            br = b_right.next()
            pl.load(cur_a, a, [0, k])
            pl.load(cur_b, b, [k, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if k == 0:
                bias_l1_tile = bias_l1.next()
                pl.load(bias_l1_tile, bias, [0, 0])
                bl = bias_l0b.next()
                pl.move(bl, bias_l1_tile)
                pl.matmul(ac, al, br, bl, phase=pl.AccPhase.Partial)
            elif k < K_SPLIT_4 - TILE:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
        pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)
        pl.system.set_mm_layout_transform(enabled=False)


@pl.jit(auto_mutex=True)
def kernel_mnk_split_bias(
    a: pl.Tensor[[M_MNK, K_SPLIT_3], pl.DT_FP16],
    b: pl.Tensor[[K_SPLIT_3, N_MNK], pl.DT_FP16],
    bias: pl.Tensor[[1, N_MNK], pl.DT_FP16],
    out: pl.Tensor[[M_MNK, N_MNK], pl.DT_FP16],
):
    m_tiles = M_MNK // TILE
    n_tiles = N_MNK // TILE
    with pl.section_cube():
        pl.system.set_mm_layout_transform(enabled=True)
        a_l1, b_l1, bias_l1, a_left, b_right, bias_l0b, acc = _make_k_split_tile_groups()
        ac = acc.current()
        for mi in pl.range(0, m_tiles):
            for ni in pl.range(0, n_tiles):
                for k in pl.range(0, K_SPLIT_3, TILE):
                    cur_a = a_l1.next()
                    cur_b = b_l1.next()
                    al = a_left.next()
                    br = b_right.next()
                    pl.load(cur_a, a, [mi * TILE, k])
                    pl.load(cur_b, b, [k, ni * TILE])
                    pl.move(al, cur_a)
                    pl.move(br, cur_b)
                    if k == 0:
                        bias_l1_tile = bias_l1.next()
                        pl.load(bias_l1_tile, bias, [0, ni * TILE])
                        bl = bias_l0b.next()
                        pl.move(bl, bias_l1_tile)
                        pl.matmul(ac, al, br, bl, phase=pl.AccPhase.Partial)
                    elif k < K_SPLIT_3 - TILE:
                        pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
                    else:
                        pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
                        pl.store(out, ac, [mi * TILE, ni * TILE], phase=pl.STPhase.Final)
        pl.system.set_mm_layout_transform(enabled=False)


@pl.jit(auto_mutex=True)
def kernel_mnk_k2_split_bias(
    a: pl.Tensor[[M_MNK, K_SPLIT_2], pl.DT_FP16],
    b: pl.Tensor[[K_SPLIT_2, N_MNK], pl.DT_FP16],
    bias: pl.Tensor[[1, N_MNK], pl.DT_FP16],
    out: pl.Tensor[[M_MNK, N_MNK], pl.DT_FP16],
):
    m_tiles = M_MNK // TILE
    n_tiles = N_MNK // TILE
    with pl.section_cube():
        pl.system.set_mm_layout_transform(enabled=True)
        a_l1, b_l1, bias_l1, a_left, b_right, bias_l0b, acc = _make_k_split_tile_groups()
        ac = acc.current()
        for mi in pl.range(0, m_tiles):
            for ni in pl.range(0, n_tiles):
                for k in pl.range(0, K_SPLIT_2, TILE):
                    cur_a = a_l1.next()
                    cur_b = b_l1.next()
                    al = a_left.next()
                    br = b_right.next()
                    pl.load(cur_a, a, [mi * TILE, k])
                    pl.load(cur_b, b, [k, ni * TILE])
                    pl.move(al, cur_a)
                    pl.move(br, cur_b)
                    if k == 0:
                        bias_l1_tile = bias_l1.next()
                        pl.load(bias_l1_tile, bias, [0, ni * TILE])
                        bl = bias_l0b.next()
                        pl.move(bl, bias_l1_tile)
                        pl.matmul(ac, al, br, bl, phase=pl.AccPhase.Partial)
                    else:
                        pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
                        pl.store(out, ac, [mi * TILE, ni * TILE], phase=pl.STPhase.Final)
        pl.system.set_mm_layout_transform(enabled=False)


@pl.jit(auto_mutex=True)
def kernel_bias_phase_final(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    with pl.section_cube():
        pl.system.set_mm_layout_transform(enabled=True)
        a_l1, b_l1, bias_l1, a_left, b_right, bias_l0b, acc = _make_k_split_tile_groups()
        a_l1_tile = a_l1.next()
        pl.load(a_l1_tile, a, [0, 0])
        b_l1_tile = b_l1.next()
        pl.load(b_l1_tile, b, [0, 0])
        bias_l1_tile = bias_l1.next()
        pl.load(bias_l1_tile, bias, [0, 0])
        al = a_left.next()
        pl.move(al, a_l1_tile)
        br = b_right.next()
        pl.move(br, b_l1_tile)
        bl = bias_l0b.next()
        pl.move(bl, bias_l1_tile)
        ac = acc.current()
        pl.matmul(ac, al, br, bl, phase=pl.AccPhase.Final)
        pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)
        pl.system.set_mm_layout_transform(enabled=False)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_matmul_bias_k_split():
    torch.npu.set_device(DEVICE_ID)
    torch.manual_seed(42)
    a = torch.randn(TILE, K_SPLIT_3, device=DEVICE, dtype=torch.float16)
    b = torch.randn(K_SPLIT_3, TILE, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(1, TILE, device=DEVICE, dtype=torch.float16)
    out = torch.zeros(TILE, TILE, device=DEVICE, dtype=torch.float16)
    kernel_k_split_bias(a, b, bias, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.matmul(a.float(), b.float()).half() + bias, rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_matmul_bias_k2_split():
    torch.npu.set_device(DEVICE_ID)
    torch.manual_seed(42)
    a = torch.randn(TILE, K_SPLIT_2, device=DEVICE, dtype=torch.float16)
    b = torch.randn(K_SPLIT_2, TILE, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(1, TILE, device=DEVICE, dtype=torch.float16)
    out = torch.zeros(TILE, TILE, device=DEVICE, dtype=torch.float16)
    kernel_k2_split_bias(a, b, bias, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.matmul(a.float(), b.float()).half() + bias, rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_matmul_bias_k4_split():
    torch.npu.set_device(DEVICE_ID)
    torch.manual_seed(42)
    a = torch.randn(TILE, K_SPLIT_4, device=DEVICE, dtype=torch.float16)
    b = torch.randn(K_SPLIT_4, TILE, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(1, TILE, device=DEVICE, dtype=torch.float16)
    out = torch.zeros(TILE, TILE, device=DEVICE, dtype=torch.float16)
    kernel_k4_split_bias(a, b, bias, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.matmul(a.float(), b.float()).half() + bias, rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_matmul_bias_mnk_split():
    torch.npu.set_device(DEVICE_ID)
    torch.manual_seed(42)
    a = torch.randn(M_MNK, K_SPLIT_3, device=DEVICE, dtype=torch.float16)
    b = torch.randn(K_SPLIT_3, N_MNK, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(1, N_MNK, device=DEVICE, dtype=torch.float16)
    out = torch.zeros(M_MNK, N_MNK, device=DEVICE, dtype=torch.float16)
    kernel_mnk_split_bias(a, b, bias, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.matmul(a.float(), b.float()).half() + bias, rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_matmul_bias_mnk_k2_split():
    torch.npu.set_device(DEVICE_ID)
    torch.manual_seed(42)
    a = torch.randn(M_MNK, K_SPLIT_2, device=DEVICE, dtype=torch.float16)
    b = torch.randn(K_SPLIT_2, N_MNK, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(1, N_MNK, device=DEVICE, dtype=torch.float16)
    out = torch.zeros(M_MNK, N_MNK, device=DEVICE, dtype=torch.float16)
    kernel_mnk_k2_split_bias(a, b, bias, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.matmul(a.float(), b.float()).half() + bias, rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_matmul_bias_phase_final():
    torch.npu.set_device(DEVICE_ID)
    torch.manual_seed(42)
    a = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.float16)
    b = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(1, TILE, device=DEVICE, dtype=torch.float16)
    out = torch.zeros(TILE, TILE, device=DEVICE, dtype=torch.float16)
    kernel_bias_phase_final(a, b, bias, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.matmul(a.float(), b.float()).half() + bias, rtol=1e-2, atol=1e-2)


# =============================================================================
# Regression (3 tests)
# =============================================================================


@pl.jit(auto_mutex=True)
def kernel_regress_no_bias(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    with pl.section_cube():
        a_mat, b_mat, bias_mat, a_left, b_right, bias_l0b, acc = _make_tile_groups()
        a_l1 = a_mat.next()
        pl.load_tile(a_l1, a, [0, 0])
        b_l1 = b_mat.next()
        pl.load_tile(b_l1, b, [0, 0])
        cur_a = a_left.next()
        pl.move(cur_a, a_l1)
        cur_b = b_right.next()
        pl.move(cur_b, b_l1)
        ac = acc.next()
        pl.matmul(ac, cur_a, cur_b)
        pl.store_tile(out, ac, [0, 0])


@pl.jit(auto_mutex=True)
def kernel_regress_no_bias_phase(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    with pl.section_cube():
        pl.system.set_mm_layout_transform(enabled=True)
        a_mat, b_mat, bias_mat, a_left, b_right, bias_l0b, acc = _make_tile_groups()
        a_l1 = a_mat.next()
        pl.load_tile(a_l1, a, [0, 0])
        b_l1 = b_mat.next()
        pl.load_tile(b_l1, b, [0, 0])
        cur_a = a_left.next()
        pl.move(cur_a, a_l1)
        cur_b = b_right.next()
        pl.move(cur_b, b_l1)
        ac = acc.current()
        pl.matmul(ac, cur_a, cur_b, phase=pl.AccPhase.Final)
        pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)
        pl.system.set_mm_layout_transform(enabled=False)


@pl.jit(auto_mutex=True)
def kernel_regress_matmul_acc(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    with pl.section_cube():
        pl.system.set_mm_layout_transform(enabled=True)
        a_l1, b_l1, bias_l1, a_left, b_right, bias_l0b, acc = _make_k_split_tile_groups()
        ac = acc.current()
        for k in pl.range(0, K_SPLIT_2, TILE):
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
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
        pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)
        pl.system.set_mm_layout_transform(enabled=False)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_regress_matmul_no_bias():
    torch.npu.set_device(DEVICE_ID)
    torch.manual_seed(42)
    a = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.float16)
    b = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.float16)
    out = torch.zeros(TILE, TILE, device=DEVICE, dtype=torch.float16)
    kernel_regress_no_bias(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.matmul(a.float(), b.float()).half(), rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_regress_matmul_no_bias_phase():
    torch.npu.set_device(DEVICE_ID)
    torch.manual_seed(42)
    a = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.float16)
    b = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.float16)
    out = torch.zeros(TILE, TILE, device=DEVICE, dtype=torch.float16)
    kernel_regress_no_bias_phase(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.matmul(a.float(), b.float()).half(), rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_regress_matmul_acc():
    torch.npu.set_device(DEVICE_ID)
    torch.manual_seed(42)
    a = torch.randn(TILE, K_SPLIT_2, device=DEVICE, dtype=torch.float16)
    b = torch.randn(K_SPLIT_2, TILE, device=DEVICE, dtype=torch.float16)
    out = torch.zeros(TILE, TILE, device=DEVICE, dtype=torch.float16)
    kernel_regress_matmul_acc(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.matmul(a.float(), b.float()).half(), rtol=1e-2, atol=1e-2)


# =============================================================================
# Tail-block valid_shape tests (4 tests)
# =============================================================================
# M/N/K dimension not aligned to TILE (128): last tile carries valid_shape < TILE.
# Uses set_validshape on L1/L0A/L0B/Acc/Bias tiles to narrow the live region.
#
# Coverage:
#   - M-tail (M=200, tail=72): a/acc valid_shape narrowed on M
#   - N-tail (N=200, tail=72): b/bias/acc valid_shape narrowed on N
#   - K-tail (K=200, tail=72): a/b valid_shape narrowed on K, K-split + phase
#   - MNK-tail (M=N=K=200): all three tails combined, K-split + M/N tiling

K_TAIL = 200
TAIL_REM = 72  # 200 - 128 = 72


def _make_tail_tile_groups(dtype=pl.DT_FP16):
    a_mat = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=dtype, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1], compact=1
        ),
        addrs=0,
        mutex_ids=[0, 1],
    )
    b_mat = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=dtype, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1], compact=1
        ),
        addrs=0x20000,
        mutex_ids=[2, 3],
    )
    bias_mat = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, TILE], dtype=dtype, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1], compact=1
        ),
        addrs=0x40000,
        mutex_ids=[4, 5],
    )
    a_left = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=dtype, target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1], compact=1
        ),
        addrs=0,
        mutex_ids=[6, 7],
    )
    b_right = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=dtype, target_memory=pl.MemorySpace.Right, valid_shape=[-1, -1], compact=1
        ),
        addrs=0,
        mutex_ids=[8, 9],
    )
    bias_l0b = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Bias, valid_shape=[-1, -1], compact=1
        ),
        addrs=0,
        mutex_ids=[10, 11],
    )
    acc = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1], compact=1
        ),
        addrs=0,
        mutex_ids=[12, 13],
    )
    return a_mat, b_mat, bias_mat, a_left, b_right, bias_l0b, acc


@pl.jit(auto_mutex=True)
def kernel_bias_m_tail(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m_total = a.shape[0]
    with pl.section_cube():
        a_mat, b_mat, bias_mat, a_left, b_right, bias_l0b, acc = _make_tail_tile_groups()
        b_l1 = b_mat.next()
        pl.load(b_l1, b, [0, 0])
        bias_l1 = bias_mat.next()
        pl.load(bias_l1, bias, [0, 0])
        cur_b = b_right.next()
        pl.move(cur_b, b_l1)
        cur_bias = bias_l0b.next()
        pl.move(cur_bias, bias_l1)
        for mi in pl.range(0, m_total, TILE):
            valid_m = pl.min(m_total - mi, TILE)
            a_l1 = a_mat.next()
            pl.set_validshape(a_l1, [valid_m, TILE])
            al = a_left.next()
            pl.set_validshape(al, [valid_m, TILE])
            pl.load(a_l1, a, [mi, 0])
            pl.move(al, a_l1)
            ac = acc.next()
            pl.set_validshape(ac, [valid_m, TILE])
            pl.matmul(ac, al, cur_b, cur_bias)
            pl.store(out, ac, [mi, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_matmul_bias_m_tail():
    """M-axis tail (M=200, tile=128, tail=72) with bias: valid_shape on a/acc."""
    torch.npu.set_device(DEVICE_ID)
    torch.manual_seed(42)
    a = torch.randn(K_TAIL, TILE, device=DEVICE, dtype=torch.float16)
    b = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(1, TILE, device=DEVICE, dtype=torch.float16)
    out = torch.zeros(K_TAIL, TILE, device=DEVICE, dtype=torch.float16)
    kernel_bias_m_tail(a, b, bias, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.matmul(a.float(), b.float()).half() + bias, rtol=1e-2, atol=1e-2)


@pl.jit(auto_mutex=True)
def kernel_bias_n_tail(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    n_total = b.shape[1]
    with pl.section_cube():
        a_mat, b_mat, bias_mat, a_left, b_right, bias_l0b, acc = _make_tail_tile_groups()
        a_l1 = a_mat.next()
        pl.load(a_l1, a, [0, 0])
        cur_a = a_left.next()
        pl.move(cur_a, a_l1)
        for ni in pl.range(0, n_total, TILE):
            valid_n = pl.min(n_total - ni, TILE)
            b_l1 = b_mat.next()
            pl.set_validshape(b_l1, [TILE, valid_n])
            br = b_right.next()
            pl.set_validshape(br, [TILE, valid_n])
            pl.load(b_l1, b, [0, ni])
            pl.move(br, b_l1)
            bias_l1 = bias_mat.next()
            pl.set_validshape(bias_l1, [1, valid_n])
            pl.load(bias_l1, bias, [0, ni])
            cur_bias = bias_l0b.next()
            pl.set_validshape(cur_bias, [1, valid_n])
            pl.move(cur_bias, bias_l1)
            ac = acc.next()
            pl.set_validshape(ac, [TILE, valid_n])
            pl.matmul(ac, cur_a, br, cur_bias)
            pl.store(out, ac, [0, ni])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_matmul_bias_n_tail():
    """N-axis tail (N=200, tile=128, tail=72) with bias: valid_shape on b/bias/acc."""
    torch.npu.set_device(DEVICE_ID)
    torch.manual_seed(42)
    a = torch.randn(TILE, TILE, device=DEVICE, dtype=torch.float16)
    b = torch.randn(TILE, K_TAIL, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(1, K_TAIL, device=DEVICE, dtype=torch.float16)
    out = torch.zeros(TILE, K_TAIL, device=DEVICE, dtype=torch.float16)
    kernel_bias_n_tail(a, b, bias, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.matmul(a.float(), b.float()).half() + bias, rtol=1e-2, atol=1e-2)


@pl.jit(auto_mutex=True)
def kernel_bias_k_tail(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    k_total = a.shape[1]
    with pl.section_cube():
        pl.system.set_mm_layout_transform(enabled=True)
        a_mat, b_mat, bias_mat, a_left, b_right, bias_l0b, acc = _make_tail_tile_groups()
        ac = acc.current()
        for ki in pl.range(0, k_total, TILE):
            valid_k = pl.min(k_total - ki, TILE)
            a_l1 = a_mat.next()
            pl.set_validshape(a_l1, [TILE, valid_k])
            b_l1 = b_mat.next()
            pl.set_validshape(b_l1, [valid_k, TILE])
            al = a_left.next()
            pl.set_validshape(al, [TILE, valid_k])
            br = b_right.next()
            pl.set_validshape(br, [valid_k, TILE])
            pl.load(a_l1, a, [0, ki])
            pl.load(b_l1, b, [ki, 0])
            pl.move(al, a_l1)
            pl.move(br, b_l1)
            if ki == 0:
                bias_l1 = bias_mat.next()
                pl.load(bias_l1, bias, [0, 0])
                cur_bias = bias_l0b.next()
                pl.move(cur_bias, bias_l1)
                if k_total <= TILE:
                    pl.matmul(ac, al, br, cur_bias, phase=pl.AccPhase.Final)
                else:
                    pl.matmul(ac, al, br, cur_bias, phase=pl.AccPhase.Partial)
            elif ki < k_total - TILE:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
        pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)
        pl.system.set_mm_layout_transform(enabled=False)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_matmul_bias_k_tail():
    """K-axis tail (K=200, tile=128, tail=72) with bias: K-split + valid_shape on last K-block."""
    torch.npu.set_device(DEVICE_ID)
    torch.manual_seed(42)
    a = torch.randn(TILE, K_TAIL, device=DEVICE, dtype=torch.float16)
    b = torch.randn(K_TAIL, TILE, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(1, TILE, device=DEVICE, dtype=torch.float16)
    out = torch.zeros(TILE, TILE, device=DEVICE, dtype=torch.float16)
    kernel_bias_k_tail(a, b, bias, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.matmul(a.float(), b.float()).half() + bias, rtol=1e-2, atol=1e-2)


@pl.jit(auto_mutex=True)
def kernel_bias_mnk_tail(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    m_total = a.shape[0]
    n_total = b.shape[1]
    k_total = a.shape[1]
    with pl.section_cube():
        pl.system.set_mm_layout_transform(enabled=True)
        a_mat, b_mat, bias_mat, a_left, b_right, bias_l0b, acc = _make_tail_tile_groups()
        ac = acc.current()
        for mi in pl.range(0, m_total, TILE):
            valid_m = pl.min(m_total - mi, TILE)
            for ni in pl.range(0, n_total, TILE):
                valid_n = pl.min(n_total - ni, TILE)
                for ki in pl.range(0, k_total, TILE):
                    valid_k = pl.min(k_total - ki, TILE)
                    a_l1 = a_mat.next()
                    pl.set_validshape(a_l1, [valid_m, valid_k])
                    b_l1 = b_mat.next()
                    pl.set_validshape(b_l1, [valid_k, valid_n])
                    al = a_left.next()
                    pl.set_validshape(al, [valid_m, valid_k])
                    br = b_right.next()
                    pl.set_validshape(br, [valid_k, valid_n])
                    pl.load(a_l1, a, [mi, ki])
                    pl.load(b_l1, b, [ki, ni])
                    pl.move(al, a_l1)
                    pl.move(br, b_l1)
                    if ki == 0:
                        bias_l1 = bias_mat.next()
                        pl.set_validshape(bias_l1, [1, valid_n])
                        pl.load(bias_l1, bias, [0, ni])
                        cur_bias = bias_l0b.next()
                        pl.set_validshape(cur_bias, [1, valid_n])
                        pl.move(cur_bias, bias_l1)
                        pl.set_validshape(ac, [valid_m, valid_n])
                        if k_total <= TILE:
                            pl.matmul(ac, al, br, cur_bias, phase=pl.AccPhase.Final)
                            pl.store(out, ac, [mi, ni], phase=pl.STPhase.Final)
                        else:
                            pl.matmul(ac, al, br, cur_bias, phase=pl.AccPhase.Partial)
                    elif ki < k_total - TILE:
                        pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
                    else:
                        pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
                        pl.store(out, ac, [mi, ni], phase=pl.STPhase.Final)
        pl.system.set_mm_layout_transform(enabled=False)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_matmul_bias_mnk_tail():
    """MNK combined tail (M=N=K=200, tile=128, tail=72) with bias: all three axes have tail."""
    torch.npu.set_device(DEVICE_ID)
    torch.manual_seed(42)
    a = torch.randn(K_TAIL, K_TAIL, device=DEVICE, dtype=torch.float16)
    b = torch.randn(K_TAIL, K_TAIL, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(1, K_TAIL, device=DEVICE, dtype=torch.float16)
    out = torch.zeros(K_TAIL, K_TAIL, device=DEVICE, dtype=torch.float16)
    kernel_bias_mnk_tail(a, b, bias, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.matmul(a.float(), b.float()).half() + bias, rtol=1e-2, atol=1e-2)
