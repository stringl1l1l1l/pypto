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
"""ST: precision checks for InferMultiIterOverlap assemble scenes.

Numerical results only. Mark / encode WAW are guarded by C++ UTs.
"""

import os

import torch
import torch_npu

import pypto

BS = 32
HIDDEN = 64
TILE = 8
TILE_N = 16


def _device_id():
    return int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))


def _run_case(kernel, golden_fn, tag, seed):
    torch.manual_seed(seed)
    x = torch.randn(BS, HIDDEN, dtype=torch.float32)
    device = f"npu:{_device_id()}"
    torch.npu.set_device(_device_id())
    x_npu = x.to(device)
    out = torch.zeros(BS, HIDDEN, dtype=torch.float32, device=device)
    kernel(x_npu, out)
    torch_npu.npu.synchronize()
    assert torch.allclose(out.cpu(), golden_fn(x), atol=1e-4, rtol=1e-4), f"{tag}: precision mismatch"


# ---------- kernels (concrete int shapes) ----------


@pypto.frontend.jit()
def _kernel_assemble_disjoint(
    in_t: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    out_t: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
):
    """Assemble with offset=[i*tile, 0]; windows are disjoint across iters."""
    n = (BS + TILE - 1) // TILE
    for i in pypto.loop(n, name="ST_DJ", idx_name="i"):
        pypto.set_vec_tile_shapes(TILE, TILE)
        v = in_t.view([TILE, HIDDEN], [i * TILE, 0])
        pypto.assemble(pypto.mul(v, 4.0), [i * TILE, 0], out_t)


@pypto.frontend.jit()
def _kernel_assemble_overlap(
    in_t: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    out_t: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
):
    """Assemble with offset=[i*(tile//2), 0]; half-window overlap across iters."""
    stride = TILE // 2
    n = (BS + TILE - 1) // TILE
    for i in pypto.loop(n, name="ST_OV", idx_name="i"):
        pypto.set_vec_tile_shapes(TILE, TILE)
        v = in_t.view([TILE, HIDDEN], [i * TILE, 0])
        pypto.assemble(pypto.mul(v, 4.0), [i * stride, 0], out_t)


@pypto.frontend.jit()
def _kernel_nested_disjoint(
    in_t: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    out_t: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
):
    """Nested assemble [i*tile, j*tile_n]; separable on both dims."""
    n_row = (BS + TILE - 1) // TILE
    n_col = (HIDDEN + TILE_N - 1) // TILE_N
    for i in pypto.loop(n_row, name="ST_NDJ_I", idx_name="i"):
        for j in pypto.loop(n_col, name="ST_NDJ_J", idx_name="j"):
            pypto.set_vec_tile_shapes(TILE, TILE)
            v = in_t.view([TILE, TILE_N], [i * TILE, j * TILE_N])
            pypto.assemble(pypto.mul(v, 4.0), [i * TILE, j * TILE_N], out_t)


@pypto.frontend.jit()
def _kernel_nested_overlap(
    in_t: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    out_t: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
):
    """Nested assemble; column stride j*(tile_n//2) half-window overlap."""
    stride_c = TILE_N // 2
    n_row = (BS + TILE - 1) // TILE
    n_col = (HIDDEN - TILE_N) // stride_c + 1
    for i in pypto.loop(n_row, name="ST_NOV_I", idx_name="i"):
        for j in pypto.loop(n_col, name="ST_NOV_J", idx_name="j"):
            pypto.set_vec_tile_shapes(TILE, TILE)
            v = in_t.view([TILE, TILE_N], [i * TILE, j * stride_c])
            pypto.assemble(pypto.mul(v, 4.0), [i * TILE, j * stride_c], out_t)


@pypto.frontend.jit()
def _kernel_fixed_offset(
    in_t: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    out_t: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
):
    """Assemble to fixed offset [0, 0] every iter; same region overlaps."""
    n = (BS + TILE - 1) // TILE
    for i in pypto.loop(n, name="ST_FIX", idx_name="i"):
        pypto.set_vec_tile_shapes(TILE, TILE)
        v = in_t.view([TILE, HIDDEN], [i * TILE, 0])
        pypto.assemble(pypto.mul(v, 4.0), [0, 0], out_t)


@pypto.frontend.jit()
def _kernel_same_idx_diff_step(
    in_t: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    out_ov: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    out_dj: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
):
    """Sibling loops share idx_name "i" but write different outs / different steps."""
    stride = TILE // 2
    n_ov = (BS + TILE - 1) // TILE
    for i in pypto.loop(0, n_ov, 1, name="ST_SIB_OV", idx_name="i"):
        pypto.set_vec_tile_shapes(TILE, TILE)
        v = in_t.view([TILE, HIDDEN], [i * TILE, 0])
        pypto.assemble(pypto.mul(v, 4.0), [i * stride, 0], out_ov)

    n_dj = 2 * ((BS + TILE - 1) // TILE)
    for i in pypto.loop(0, n_dj, 2, name="ST_SIB_DJ", idx_name="i"):
        pypto.set_vec_tile_shapes(TILE, TILE)
        v = in_t.view([TILE, HIDDEN], [i * stride, 0])
        pypto.assemble(pypto.mul(v, 4.0), [i * stride, 0], out_dj)


@pypto.frontend.jit()
def _kernel_nested_inner_immediate_offset(
    in_t: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    out_t: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
):
    """Innermost j does not appear in assemble offset: window is invariant in j (last-j wins)."""
    n_row = (BS + TILE - 1) // TILE
    n_col = (HIDDEN + TILE_N - 1) // TILE_N
    for i in pypto.loop(n_row, name="ST_IMM_I", idx_name="i"):
        for j in pypto.loop(n_col, name="ST_IMM_J", idx_name="j"):
            pypto.set_vec_tile_shapes(TILE, TILE)
            v = in_t.view([TILE, TILE_N], [i * TILE, j * TILE_N])
            pypto.assemble(pypto.mul(v, 4.0), [i * TILE, 0], out_t)


# ---------- goldens ----------


def _golden_disjoint(x):
    out = torch.zeros_like(x)
    for i in range((BS + TILE - 1) // TILE):
        r0 = i * TILE
        rows = min(TILE, BS - r0)
        out[r0:r0 + rows, :] = 4.0 * x[r0:r0 + rows, :]
    return out


def _golden_overlap(x):
    out = torch.zeros_like(x)
    stride = TILE // 2
    n = (BS + TILE - 1) // TILE
    for i in range(n):
        rows = min(TILE, BS - i * TILE)
        dst = i * stride
        rows = min(rows, BS - dst)
        if rows <= 0:
            break
        out[dst:dst + rows, :] = 4.0 * x[i * TILE:i * TILE + rows, :]
    return out


def _golden_nested_disjoint(x):
    out = torch.zeros_like(x)
    for i in range((BS + TILE - 1) // TILE):
        for j in range((HIDDEN + TILE_N - 1) // TILE_N):
            r0, c0 = i * TILE, j * TILE_N
            rows = min(TILE, BS - r0)
            cols = min(TILE_N, HIDDEN - c0)
            out[r0:r0 + rows, c0:c0 + cols] = 4.0 * x[r0:r0 + rows, c0:c0 + cols]
    return out


def _golden_nested_overlap(x):
    out = torch.zeros_like(x)
    stride_c = TILE_N // 2
    n_row = (BS + TILE - 1) // TILE
    n_col = (HIDDEN - TILE_N) // stride_c + 1
    for i in range(n_row):
        for j in range(n_col):
            r0, c0 = i * TILE, j * stride_c
            if c0 >= HIDDEN:
                break
            rows = min(TILE, BS - r0)
            cols = min(TILE_N, HIDDEN - c0)
            out[r0:r0 + rows, c0:c0 + cols] = 4.0 * x[r0:r0 + rows, c0:c0 + cols]
    return out


def _golden_fixed_offset(x):
    out = torch.zeros_like(x)
    n = (BS + TILE - 1) // TILE
    last = n - 1
    r0 = last * TILE
    rows = min(TILE, BS - r0)
    out[0:rows, :] = 4.0 * x[r0:r0 + rows, :]
    return out


def _golden_same_idx_diff_step(x):
    out_ov = torch.zeros_like(x)
    out_dj = torch.zeros_like(x)
    stride = TILE // 2
    n_ov = (BS + TILE - 1) // TILE
    for i in range(n_ov):
        rows = min(TILE, BS - i * TILE)
        dst = i * stride
        rows = min(rows, BS - dst)
        if rows <= 0:
            break
        out_ov[dst:dst + rows, :] = 4.0 * x[i * TILE:i * TILE + rows, :]

    n_dj = 2 * ((BS + TILE - 1) // TILE)
    for i in range(0, n_dj, 2):
        r0 = i * stride
        if r0 >= BS:
            break
        rows = min(TILE, BS - r0)
        out_dj[r0:r0 + rows, :] = 4.0 * x[r0:r0 + rows, :]
    return out_ov, out_dj


def _golden_nested_inner_immediate_offset(x):
    out = torch.zeros_like(x)
    n_row = (BS + TILE - 1) // TILE
    n_col = (HIDDEN + TILE_N - 1) // TILE_N
    for i in range(n_row):
        j = n_col - 1
        r0, c0 = i * TILE, j * TILE_N
        rows = min(TILE, BS - r0)
        cols = min(TILE_N, HIDDEN - c0)
        out[r0:r0 + rows, 0:cols] = 4.0 * x[r0:r0 + rows, c0:c0 + cols]
    return out


# ---------- tests ----------


def test_multi_iter_waw_assemble_disjoint():
    _run_case(_kernel_assemble_disjoint, _golden_disjoint, tag="assemble_disjoint", seed=0)


def test_multi_iter_waw_assemble_overlap():
    _run_case(_kernel_assemble_overlap, _golden_overlap, tag="assemble_overlap", seed=1)


def test_multi_iter_waw_nested_disjoint():
    _run_case(_kernel_nested_disjoint, _golden_nested_disjoint, tag="nested_disjoint", seed=2)


def test_multi_iter_waw_nested_overlap():
    _run_case(_kernel_nested_overlap, _golden_nested_overlap, tag="nested_overlap", seed=3)


def test_multi_iter_waw_fixed_offset():
    _run_case(_kernel_fixed_offset, _golden_fixed_offset, tag="fixed_offset", seed=4)


def test_multi_iter_waw_same_idx_diff_step():
    torch.manual_seed(5)
    x = torch.randn(BS, HIDDEN, dtype=torch.float32)
    device = f"npu:{_device_id()}"
    torch.npu.set_device(_device_id())
    x_npu = x.to(device)
    out_ov = torch.zeros(BS, HIDDEN, dtype=torch.float32, device=device)
    out_dj = torch.zeros(BS, HIDDEN, dtype=torch.float32, device=device)
    _kernel_same_idx_diff_step(x_npu, out_ov, out_dj)
    torch_npu.npu.synchronize()
    g_ov, g_dj = _golden_same_idx_diff_step(x)
    assert torch.allclose(out_ov.cpu(), g_ov, atol=1e-4, rtol=1e-4), "same_idx_diff_step out_ov mismatch"
    assert torch.allclose(out_dj.cpu(), g_dj, atol=1e-4, rtol=1e-4), "same_idx_diff_step out_dj mismatch"


def test_multi_iter_waw_nested_inner_immediate_offset():
    _run_case(
        _kernel_nested_inner_immediate_offset,
        _golden_nested_inner_immediate_offset,
        tag="nested_inner_immediate_offset",
        seed=6,
    )
