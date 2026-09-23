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

"""Test layout / shape / is_transpose scenarios for GM->L1->L0A/L0B->L0C->GM pipeline.

All kernels compute  C[M,N] = A[M,K] @ B[K,N]  and verify against torch.matmul.
Uses make_tile_group + auto_mutex (no manual sync needed).

Organization:
  Part A — Non-square (M=64, K=128, N=32) test cases:  t01 ~ t13
  Part B — Square (M=K=N=128) test cases:              t14 ~ t24

L1 layout constraint (from TLOAD dispatch):
  - NZ L1: GM ND  → TLOAD ND→NZ (no transpose)
  - ZN L1: GM DN  → TLOAD DN→ZN (transpose only; is_transpose=True required)
  - ND→ZN path does NOT exist; ZN L1 always requires is_transpose=True (DN GlobalTensor).

Valid L1 layout combinations (7 total, all explicit is_transpose):
  NN: Left NZ, Right NZ        (ND→NZ, ND→NZ)
  NT: Left NZ, Right ZN        (ND→NZ, DN→ZN)
  TN: Left ZN, Right NZ        (DN→ZN, ND→NZ)
  TT: Left ZN, Right ZN        (DN→ZN, DN→ZN)

Naming convention:
  t<NN>_<write>_<shape>_<gm_layout>_<l1_layout>_<flow>
    write:     exp (explicit is_transpose=True)
    shape:     nsq (M!=K!=N non-square)           / sq  (M=K=N square)
    gm_layout: nd  (GM normal [M,K])              / dn  (GM transposed [K,M] or [N,K])
    l1_layout: nz  (Mat NZ)                       / zn  (Mat ZN)
    flow:      single / kloop / kloop_tail / kloop_if / m_tiling / m_tail
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


logging.basicConfig(level=logging.INFO)

M_NSQ = 64
K_NSQ = 128
N_NSQ = 32

M_SQ = 128
K_SQ = 128
N_SQ = 128

TILE_SQ = 128


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip("not A5")


def _inputs(device, shape, dtype=torch.float16):
    torch.manual_seed(42)
    return torch.randn(shape, device=device, dtype=dtype)


def _run(kernel, a_arg, b_arg, a, b, out_shape, device=ST_DEVICE):
    out = torch.zeros(out_shape, device=device, dtype=torch.float32)
    kernel(a_arg, b_arg, out)
    torch.npu.synchronize()
    ref = torch.matmul(a.float(), b.float())
    diff = (out - ref).abs().max().item()
    logging.info("%s: max|out - A@B| = %s", kernel.__name__, diff)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


# #####################################################################################
# #####################################################################################


# === A1: NN non-square — Left NZ, Right NZ (only valid L1 combo for NN) =============


@pl.jit(auto_mutex=True)
def t01_nsq_nd_nz_single(
    a: pl.Tensor[[M_NSQ, K_NSQ], pl.DT_FP16],
    b: pl.Tensor[[K_NSQ, N_NSQ], pl.DT_FP16],
    out: pl.Tensor[[M_NSQ, N_NSQ], pl.DT_FP32],
):
    """NN nsq: both L1 NZ, no transpose, single tile."""
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M_NSQ, K_NSQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[K_NSQ, N_NSQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x10000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[M_NSQ, K_NSQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[K_NSQ, N_NSQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(
            shape=[M_NSQ, N_NSQ], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        ),
        addrs=0x0,
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


@pl.jit(auto_mutex=True)
def t02_nsq_nd_nz_kloop_if(
    a: pl.Tensor[[M_NSQ, 256], pl.DT_FP16],
    b: pl.Tensor[[256, N_NSQ], pl.DT_FP16],
    out: pl.Tensor[[M_NSQ, N_NSQ], pl.DT_FP32],
):
    """NN nsq with K-loop + if: A[M,256]@B[256,N], 2 K-blocks of 128."""
    k_total = 256
    tile_k = 128
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M_NSQ, tile_k], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x00000,
        mutex_ids=[0, 1],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[tile_k, N_NSQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x10000,
        mutex_ids=[2, 3],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[M_NSQ, tile_k], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0,
        mutex_ids=[4, 5],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[tile_k, N_NSQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0,
        mutex_ids=[6, 7],
    )
    acc_grp = pl.make_tile_group(
        type=pl.TileType(
            shape=[M_NSQ, N_NSQ], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        ),
        addrs=0x0,
        mutex_ids=[8],
    )
    with pl.section_cube():
        ac = acc_grp.current()
        for ki in pl.range(0, k_total, tile_k):
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_l0a.next()
            br = b_l0b.next()
            pl.load(cur_a, a, [0, ki])
            pl.load(cur_b, b, [ki, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if ki == 0 and ki != k_total - tile_k:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
            elif ki == 0 and ki == k_total - tile_k:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Final)
            elif ki < k_total - tile_k:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
        pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)


# === A2: NT non-square (right transpose) — rec NZ(relabel) vs exp ZN ===============


@pl.jit(auto_mutex=True)
def t03_nsq_dn_zn_single(
    a: pl.Tensor[[M_NSQ, K_NSQ], pl.DT_FP16],
    b_t: pl.Tensor[[N_NSQ, K_NSQ], pl.DT_FP16],
    out: pl.Tensor[[M_NSQ, N_NSQ], pl.DT_FP32],
):
    """NT nsq explicit: B^T[N,K] with is_transpose=True into Mat ZN [K,N]."""
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M_NSQ, K_NSQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[K_NSQ, N_NSQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=0x10000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[M_NSQ, K_NSQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[K_NSQ, N_NSQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(
            shape=[M_NSQ, N_NSQ], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        ),
        addrs=0x0,
        mutex_ids=[4],
    )
    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b_t, [0, 0], order=[1, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.store(out, ac, [0, 0])


@pl.jit(auto_mutex=True)
def t05_nsq_dn_zn_kloop(
    a: pl.Tensor[[M_NSQ, 256], pl.DT_FP16],
    b_t: pl.Tensor[[N_NSQ, 256], pl.DT_FP16],
    out: pl.Tensor[[M_NSQ, N_NSQ], pl.DT_FP32],
):
    """NT nsq explicit with K-loop: A[M,K] @ B^T[N,K], K split into 2 blocks."""
    k_total = 256
    tile_k = 128
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M_NSQ, tile_k], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x00000,
        mutex_ids=[0, 1],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[tile_k, N_NSQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=0x10000,
        mutex_ids=[2, 3],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[M_NSQ, tile_k], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0,
        mutex_ids=[4, 5],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[tile_k, N_NSQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0,
        mutex_ids=[6, 7],
    )
    acc_grp = pl.make_tile_group(
        type=pl.TileType(
            shape=[M_NSQ, N_NSQ], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        ),
        addrs=0x0,
        mutex_ids=[8],
    )
    with pl.section_cube():
        ac = acc_grp.current()
        for ki in pl.range(0, k_total, tile_k):
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_l0a.next()
            br = b_l0b.next()
            pl.load(cur_a, a, [0, ki])
            pl.load(cur_b, b_t, [0, ki], order=[1, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if ki == 0 and ki != k_total - tile_k:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
            elif ki == 0 and ki == k_total - tile_k:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Final)
            elif ki < k_total - tile_k:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
        pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)


@pl.jit(auto_mutex=True)
def t06_nsq_dn_zn_kloop_tail(
    a: pl.Tensor[[M_NSQ, 200], pl.DT_FP16],
    b_t: pl.Tensor[[N_NSQ, 200], pl.DT_FP16],
    out: pl.Tensor[[M_NSQ, N_NSQ], pl.DT_FP32],
):
    """NT nsq explicit with K-loop + tail: K=200, tile=128, tail=72."""
    k_total = 200
    tile_k = 128
    a_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[M_NSQ, tile_k],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addrs=0x00000,
        mutex_ids=[0, 1],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[tile_k, N_NSQ],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZN,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addrs=0x10000,
        mutex_ids=[2, 3],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(
            shape=[M_NSQ, tile_k],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Left,
            layout=pl.NZ,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addrs=0x0,
        mutex_ids=[4, 5],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(
            shape=[tile_k, N_NSQ],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Right,
            layout=pl.ZN,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addrs=0x0,
        mutex_ids=[6, 7],
    )
    acc_grp = pl.make_tile_group(
        type=pl.TileType(
            shape=[M_NSQ, N_NSQ], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        ),
        addrs=0x0,
        mutex_ids=[8],
    )
    with pl.section_cube():
        ac = acc_grp.current()
        for ki in pl.range(0, k_total, tile_k):
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_l0a.next()
            br = b_l0b.next()
            remaining = k_total - ki
            if remaining < tile_k:
                pl.set_validshape(cur_a, [M_NSQ, remaining])
                pl.set_validshape(cur_b, [remaining, N_NSQ])
                pl.set_validshape(al, [M_NSQ, remaining])
                pl.set_validshape(br, [remaining, N_NSQ])
            pl.load(cur_a, a, [0, ki])
            pl.load(cur_b, b_t, [0, ki], order=[1, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if ki == 0 and ki != k_total - tile_k:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
            elif ki == 0 and ki == k_total - tile_k:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Final)
            elif ki < k_total - tile_k:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
        pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)


# === A3: TN non-square (left transpose) — rec NZ(relabel) vs exp ZN ===============


@pl.jit(auto_mutex=True)
def t07_nsq_dn_zn_tn_single(
    a_t: pl.Tensor[[K_NSQ, M_NSQ], pl.DT_FP16],
    b: pl.Tensor[[K_NSQ, N_NSQ], pl.DT_FP16],
    out: pl.Tensor[[M_NSQ, N_NSQ], pl.DT_FP32],
):
    """TN nsq explicit single: A^T[K,M] is_transpose=True into Mat ZN [M,K]."""
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M_NSQ, K_NSQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[K_NSQ, N_NSQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x10000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[M_NSQ, K_NSQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[K_NSQ, N_NSQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(
            shape=[M_NSQ, N_NSQ], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        ),
        addrs=0x0,
        mutex_ids=[4],
    )
    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.load(cur_a, a_t, [0, 0], order=[1, 0])
        pl.load(cur_b, b, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.store(out, ac, [0, 0])


@pl.jit(auto_mutex=True)
def t09_nsq_dn_zn_tn_kloop_if(
    a_t: pl.Tensor[[K_NSQ, M_NSQ], pl.DT_FP16],
    b: pl.Tensor[[K_NSQ, N_NSQ], pl.DT_FP16],
    out: pl.Tensor[[M_NSQ, N_NSQ], pl.DT_FP32],
):
    """TN nsq explicit with K-loop + if: A^T[K,M] is_transpose=True, Mat ZN, B normal."""
    tile_k = 64
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M_NSQ, tile_k], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=0x00000,
        mutex_ids=[0, 1],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[tile_k, N_NSQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x10000,
        mutex_ids=[2, 3],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[M_NSQ, tile_k], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0,
        mutex_ids=[4, 5],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[tile_k, N_NSQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0,
        mutex_ids=[6, 7],
    )
    acc_grp = pl.make_tile_group(
        type=pl.TileType(
            shape=[M_NSQ, N_NSQ], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        ),
        addrs=0x0,
        mutex_ids=[8],
    )
    with pl.section_cube():
        ac = acc_grp.current()
        for ki in pl.range(0, K_NSQ, tile_k):
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_l0a.next()
            br = b_l0b.next()
            pl.load(cur_a, a_t, [ki, 0], order=[1, 0])
            pl.load(cur_b, b, [ki, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if ki == 0 and ki != K_NSQ - tile_k:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
            elif ki == 0 and ki == K_NSQ - tile_k:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Final)
            elif ki < K_NSQ - tile_k:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
        pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)


@pl.jit(auto_mutex=True)
def t10_nsq_dn_zn_tn_kloop_tail(
    a_t: pl.Tensor[[K_NSQ, M_NSQ], pl.DT_FP16],
    b: pl.Tensor[[K_NSQ, N_NSQ], pl.DT_FP16],
    out: pl.Tensor[[M_NSQ, N_NSQ], pl.DT_FP32],
):
    """TN nsq explicit with K-loop + if + tail: A^T is_transpose, B normal.

    K=128 split into 2 blocks of 64, last block uses valid_shape.
    """
    k_total = K_NSQ
    tile_k = 64
    a_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[M_NSQ, tile_k],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZN,
            valid_shape=[-1, -1],
        ),
        addrs=0x00000,
        mutex_ids=[0, 1],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[tile_k, N_NSQ],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
            valid_shape=[-1, -1],
        ),
        addrs=0x10000,
        mutex_ids=[2, 3],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[M_NSQ, tile_k], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0,
        mutex_ids=[4, 5],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[tile_k, N_NSQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0,
        mutex_ids=[6, 7],
    )
    acc_grp = pl.make_tile_group(
        type=pl.TileType(
            shape=[M_NSQ, N_NSQ], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        ),
        addrs=0x0,
        mutex_ids=[8],
    )
    with pl.section_cube():
        ac = acc_grp.current()
        for ki in pl.range(0, k_total, tile_k):
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_l0a.next()
            br = b_l0b.next()
            remaining = k_total - ki
            if remaining < tile_k:
                pl.set_validshape(cur_a, [M_NSQ, remaining])
                pl.set_validshape(cur_b, [remaining, N_NSQ])
            pl.load(cur_a, a_t, [ki, 0], order=[1, 0])
            pl.load(cur_b, b, [ki, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if ki == 0 and ki != k_total - tile_k:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
            elif ki == 0 and ki == k_total - tile_k:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Final)
            elif ki < k_total - tile_k:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
        pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)


# === A4: TT non-square (both transpose) — rec NZ(relabel) vs exp ZN ===============


@pl.jit(auto_mutex=True)
def t11_nsq_dn_zn_tt_single(
    a_t: pl.Tensor[[K_NSQ, M_NSQ], pl.DT_FP16],
    b_t: pl.Tensor[[N_NSQ, K_NSQ], pl.DT_FP16],
    out: pl.Tensor[[M_NSQ, N_NSQ], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M_NSQ, K_NSQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[K_NSQ, N_NSQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=0x10000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[M_NSQ, K_NSQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[K_NSQ, N_NSQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(
            shape=[M_NSQ, N_NSQ], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        ),
        addrs=0x0,
        mutex_ids=[4],
    )
    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.load(cur_a, a_t, [0, 0], order=[1, 0])
        pl.load(cur_b, b_t, [0, 0], order=[1, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.store(out, ac, [0, 0])


@pl.jit(auto_mutex=True)
def t12_nsq_dn_zn_tt_kloop(
    a_t: pl.Tensor[[256, M_NSQ], pl.DT_FP16],
    b_t: pl.Tensor[[N_NSQ, 256], pl.DT_FP16],
    out: pl.Tensor[[M_NSQ, N_NSQ], pl.DT_FP32],
):
    """TT nsq explicit with K-loop: both is_transpose=True, both Mat ZN.

    K=256 split into 2 blocks of 128.
    """
    k_total = 256
    tile_k = 128
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M_NSQ, tile_k], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=0x00000,
        mutex_ids=[0, 1],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[tile_k, N_NSQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=0x10000,
        mutex_ids=[2, 3],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[M_NSQ, tile_k], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0,
        mutex_ids=[4, 5],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[tile_k, N_NSQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0,
        mutex_ids=[6, 7],
    )
    acc_grp = pl.make_tile_group(
        type=pl.TileType(
            shape=[M_NSQ, N_NSQ], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        ),
        addrs=0x0,
        mutex_ids=[8],
    )
    with pl.section_cube():
        ac = acc_grp.current()
        for ki in pl.range(0, k_total, tile_k):
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_l0a.next()
            br = b_l0b.next()
            pl.load(cur_a, a_t, [ki, 0], order=[1, 0])
            pl.load(cur_b, b_t, [0, ki], order=[1, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if ki == 0 and ki != k_total - tile_k:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
            elif ki == 0 and ki == k_total - tile_k:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Final)
            elif ki < k_total - tile_k:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
        pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)


@pl.jit(auto_mutex=True)
def t13_nsq_dn_zn_tt_kloop_if(
    a_t: pl.Tensor[[K_NSQ, M_NSQ], pl.DT_FP16],
    b_t: pl.Tensor[[N_NSQ, K_NSQ], pl.DT_FP16],
    out: pl.Tensor[[M_NSQ, N_NSQ], pl.DT_FP32],
):
    """TT nsq explicit with K-loop + if: both is_transpose=True, both Mat ZN."""
    tile_k = 64
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M_NSQ, tile_k], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=0x00000,
        mutex_ids=[0, 1],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[tile_k, N_NSQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=0x10000,
        mutex_ids=[2, 3],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[M_NSQ, tile_k], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0,
        mutex_ids=[4, 5],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[tile_k, N_NSQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0,
        mutex_ids=[6, 7],
    )
    acc_grp = pl.make_tile_group(
        type=pl.TileType(
            shape=[M_NSQ, N_NSQ], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        ),
        addrs=0x0,
        mutex_ids=[8],
    )
    with pl.section_cube():
        ac = acc_grp.current()
        for ki in pl.range(0, K_NSQ, tile_k):
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_l0a.next()
            br = b_l0b.next()
            pl.load(cur_a, a_t, [ki, 0], order=[1, 0])
            pl.load(cur_b, b_t, [0, ki], order=[1, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if ki == 0 and ki != K_NSQ - tile_k:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
            elif ki == 0 and ki == K_NSQ - tile_k:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Final)
            elif ki < K_NSQ - tile_k:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
        pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)


# #####################################################################################
# #####################################################################################


# === B1: NN square, recommended =====================================================


@pl.jit(auto_mutex=True)
def t14_sq_nd_nz_single(
    a: pl.Tensor[[M_SQ, K_SQ], pl.DT_FP16],
    b: pl.Tensor[[K_SQ, N_SQ], pl.DT_FP16],
    out: pl.Tensor[[M_SQ, N_SQ], pl.DT_FP32],
):
    """NN square recommended: no transpose, baseline."""
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M_SQ, K_SQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[K_SQ, N_SQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x10000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[M_SQ, K_SQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[K_SQ, N_SQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(
            shape=[M_SQ, N_SQ], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        ),
        addrs=0x0,
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


@pl.jit(auto_mutex=True)
def t15_sq_nd_nz_kloop_if(
    a: pl.Tensor[[M_SQ, 256], pl.DT_FP16],
    b: pl.Tensor[[256, N_SQ], pl.DT_FP16],
    out: pl.Tensor[[M_SQ, N_SQ], pl.DT_FP32],
):
    """NN square recommended with K-loop + if: 2 K-blocks."""
    k_total = 256
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M_SQ, TILE_SQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x00000,
        mutex_ids=[0, 1],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_SQ, N_SQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x10000,
        mutex_ids=[2, 3],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[M_SQ, TILE_SQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0,
        mutex_ids=[4, 5],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_SQ, N_SQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0,
        mutex_ids=[6, 7],
    )
    acc_grp = pl.make_tile_group(
        type=pl.TileType(
            shape=[M_SQ, N_SQ], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        ),
        addrs=0x0,
        mutex_ids=[8],
    )
    with pl.section_cube():
        ac = acc_grp.current()
        for ki in pl.range(0, k_total, TILE_SQ):
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_l0a.next()
            br = b_l0b.next()
            pl.load(cur_a, a, [0, ki])
            pl.load(cur_b, b, [ki, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if ki == 0 and ki != k_total - TILE_SQ:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
            elif ki == 0 and ki == k_total - TILE_SQ:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Final)
            elif ki < k_total - TILE_SQ:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
        pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)


@pl.jit(auto_mutex=True)
def t16_sq_nd_nz_kacc_3blocks(
    a: pl.Tensor[[128, 384], pl.DT_FP16],
    b: pl.Tensor[[384, 128], pl.DT_FP16],
    out: pl.Tensor[[128, 128], pl.DT_FP32],
):
    """NN square 3-block K-acc: Partial/Partial/Final 3-way if branch."""
    tile = 128
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[tile, tile], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x00000,
        mutex_ids=[0, 1],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[tile, tile], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x10000,
        mutex_ids=[2, 3],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[tile, tile], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0,
        mutex_ids=[4, 5],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[tile, tile], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0,
        mutex_ids=[6, 7],
    )
    acc_grp = pl.make_tile_group(
        type=pl.TileType(
            shape=[tile, tile], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        ),
        addrs=0x0,
        mutex_ids=[8],
    )
    with pl.section_cube():
        pl.system.set_mm_layout_transform(enabled=True)
        ac = acc_grp.current()
        for k in pl.range(0, 384, tile):
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_l0a.next()
            br = b_l0b.next()
            pl.load(cur_a, a, [0, k])
            pl.load(cur_b, b, [k, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if k == 0 and k != 384 - tile:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
            elif k == 0 and k == 384 - tile:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Final)
            elif k < 384 - tile:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
        pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)
        pl.system.set_mm_layout_transform(enabled=False)


# === B2: NN square, M-tiling and M-tail =============================================


@pl.jit(auto_mutex=True)
def t17_sq_nd_nz_m_tiling(
    a: pl.Tensor[[256, 64], pl.DT_FP16],
    b: pl.Tensor[[64, 64], pl.DT_FP16],
    out: pl.Tensor[[256, 64], pl.DT_FP32],
):
    """NN M-axis tiling: M=256 split into 4 blocks of 64, store per block."""
    tile = 64
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[tile, tile], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x00000,
        mutex_ids=[0, 1],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[tile, tile], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x10000,
        mutex_ids=[2],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[tile, tile], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0,
        mutex_ids=[3],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[tile, tile], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0,
        mutex_ids=[4],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(
            shape=[tile, tile], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        ),
        addrs=0x0,
        mutex_ids=[5],
    )
    with pl.section_cube():
        for i in pl.range(0, 4, 1):
            cur_a = a_l1.next()
            cur_b = b_l1.current()
            al = a_l0a.next()
            br = b_l0b.current()
            ac = c_l0c.current()
            pl.load(cur_a, a, [i * tile, 0])
            pl.load(cur_b, b, [0, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            pl.matmul(ac, al, br)
            pl.store(out, ac, [i * tile, 0])


@pl.jit(auto_mutex=True)
def t18_sq_nd_nz_m_tail(
    a: pl.Tensor[[200, 128], pl.DT_FP16],
    b: pl.Tensor[[128, 128], pl.DT_FP16],
    out: pl.Tensor[[128, 128], pl.DT_FP32],
):
    """NN M-axis tail: M=200, tile=128, tail=72 with valid_shape on M dimension."""
    tile = 128
    a_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[tile, tile], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ, valid_shape=[-1, -1]
        ),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[tile, tile], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x10000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[tile, tile], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[tile, tile], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(
            shape=[tile, tile],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            fractal=1024,
            valid_shape=[-1, -1],
        ),
        addrs=0x0,
        mutex_ids=[4],
    )
    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.set_validshape(cur_a, [72, tile])
        pl.set_validshape(ac, [72, tile])
        pl.load(cur_a, a, [128, 0])
        pl.load(cur_b, b, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.store(out, ac, [0, 0])


# === B3: NN square FP32 =============================================================


@pl.jit(auto_mutex=True)
def t19_sq_nd_nz_fp32_single(
    a: pl.Tensor[[64, 64], pl.DT_FP32],
    b: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    """NN square FP32: tests FP32 dtype path (c0Size=8 instead of 16)."""
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x10000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        ),
        addrs=0x0,
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


# === B4: NT/TN/TT square explicit (is_transpose required for square) ================


@pl.jit(auto_mutex=True)
def t20_sq_dn_zn_single(
    a: pl.Tensor[[M_SQ, K_SQ], pl.DT_FP16],
    b_t: pl.Tensor[[N_SQ, K_SQ], pl.DT_FP16],
    out: pl.Tensor[[M_SQ, N_SQ], pl.DT_FP32],
):
    """NT square explicit single: is_transpose=True required (Pass cannot detect square reversed)."""
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M_SQ, K_SQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[K_SQ, N_SQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=0x10000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[M_SQ, K_SQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[K_SQ, N_SQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(
            shape=[M_SQ, N_SQ], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        ),
        addrs=0x0,
        mutex_ids=[4],
    )
    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b_t, [0, 0], order=[1, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.store(out, ac, [0, 0])


@pl.jit(auto_mutex=True)
def t21_sq_dn_zn_kloop(
    a: pl.Tensor[[M_SQ, K_SQ], pl.DT_FP16],
    b_t: pl.Tensor[[N_SQ, K_SQ], pl.DT_FP16],
    out: pl.Tensor[[M_SQ, N_SQ], pl.DT_FP32],
):
    """NT square explicit with K-loop + if."""
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M_SQ, TILE_SQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x00000,
        mutex_ids=[0, 1],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_SQ, N_SQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=0x10000,
        mutex_ids=[2, 3],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[M_SQ, TILE_SQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0,
        mutex_ids=[4, 5],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_SQ, N_SQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0,
        mutex_ids=[6, 7],
    )
    acc_grp = pl.make_tile_group(
        type=pl.TileType(
            shape=[M_SQ, N_SQ], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        ),
        addrs=0x0,
        mutex_ids=[8],
    )
    with pl.section_cube():
        ac = acc_grp.current()
        for ki in pl.range(0, K_SQ, TILE_SQ):
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_l0a.next()
            br = b_l0b.next()
            pl.load(cur_a, a, [0, ki])
            pl.load(cur_b, b_t, [0, ki], order=[1, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if ki == 0 and ki != K_SQ - TILE_SQ:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
            elif ki == 0 and ki == K_SQ - TILE_SQ:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Final)
            elif ki < K_SQ - TILE_SQ:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
        pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)


@pl.jit(auto_mutex=True)
def t22_sq_dn_zn_tn_kloop_if(
    a_t: pl.Tensor[[K_SQ, M_SQ], pl.DT_FP16],
    b: pl.Tensor[[K_SQ, N_SQ], pl.DT_FP16],
    out: pl.Tensor[[M_SQ, N_SQ], pl.DT_FP32],
):
    """TN square explicit with K-loop + if: A^T[K,M] is_transpose=True, B normal."""
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M_SQ, K_SQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=0x00000,
        mutex_ids=[0, 1],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[K_SQ, N_SQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x10000,
        mutex_ids=[2, 3],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[M_SQ, K_SQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0,
        mutex_ids=[4, 5],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[K_SQ, N_SQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0,
        mutex_ids=[6, 7],
    )
    acc_grp = pl.make_tile_group(
        type=pl.TileType(
            shape=[M_SQ, N_SQ], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        ),
        addrs=0x0,
        mutex_ids=[8],
    )
    with pl.section_cube():
        ac = acc_grp.current()
        for ki in pl.range(0, K_SQ, TILE_SQ):
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_l0a.next()
            br = b_l0b.next()
            pl.load(cur_a, a_t, [ki, 0], order=[1, 0])
            pl.load(cur_b, b, [ki, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if ki == 0 and ki != K_SQ - TILE_SQ:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
            elif ki == 0 and ki == K_SQ - TILE_SQ:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Final)
            elif ki < K_SQ - TILE_SQ:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
        pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)


@pl.jit(auto_mutex=True)
def t23_sq_dn_zn_tt_kloop_if(
    a_t: pl.Tensor[[K_SQ, M_SQ], pl.DT_FP16],
    b_t: pl.Tensor[[N_SQ, K_SQ], pl.DT_FP16],
    out: pl.Tensor[[M_SQ, N_SQ], pl.DT_FP32],
):
    """TT square explicit with K-loop + if: both is_transpose=True, both Mat ZN."""
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M_SQ, K_SQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=0x00000,
        mutex_ids=[0, 1],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[K_SQ, N_SQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=0x10000,
        mutex_ids=[2, 3],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[M_SQ, K_SQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0,
        mutex_ids=[4, 5],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[K_SQ, N_SQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0,
        mutex_ids=[6, 7],
    )
    acc_grp = pl.make_tile_group(
        type=pl.TileType(
            shape=[M_SQ, N_SQ], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        ),
        addrs=0x0,
        mutex_ids=[8],
    )
    with pl.section_cube():
        ac = acc_grp.current()
        for ki in pl.range(0, K_SQ, TILE_SQ):
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_l0a.next()
            br = b_l0b.next()
            pl.load(cur_a, a_t, [ki, 0], order=[1, 0])
            pl.load(cur_b, b_t, [0, ki], order=[1, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if ki == 0 and ki != K_SQ - TILE_SQ:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
            elif ki == 0 and ki == K_SQ - TILE_SQ:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Final)
            elif ki < K_SQ - TILE_SQ:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
        pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)


@pl.jit(auto_mutex=True)
def t24_sq_dn_zn_kloop_tail(
    a: pl.Tensor[[M_SQ, 200], pl.DT_FP16],
    b_t: pl.Tensor[[N_SQ, 200], pl.DT_FP16],
    out: pl.Tensor[[M_SQ, N_SQ], pl.DT_FP32],
):
    """NT square explicit with K-loop + if + tail: K=200, tile=128, tail=72.

    is_transpose=True required (square cannot auto-detect).
    """
    k_total = 200
    tile_k = 128
    a_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[M_SQ, tile_k],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addrs=0x00000,
        mutex_ids=[0, 1],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[tile_k, N_SQ],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZN,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addrs=0x10000,
        mutex_ids=[2, 3],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(
            shape=[M_SQ, tile_k],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Left,
            layout=pl.NZ,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addrs=0x0,
        mutex_ids=[4, 5],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(
            shape=[tile_k, N_SQ],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Right,
            layout=pl.ZN,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addrs=0x0,
        mutex_ids=[6, 7],
    )
    acc_grp = pl.make_tile_group(
        type=pl.TileType(
            shape=[M_SQ, N_SQ], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        ),
        addrs=0x0,
        mutex_ids=[8],
    )
    with pl.section_cube():
        ac = acc_grp.current()
        for ki in pl.range(0, k_total, tile_k):
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_l0a.next()
            br = b_l0b.next()
            remaining = k_total - ki
            if remaining < tile_k:
                pl.set_validshape(cur_a, [M_SQ, remaining])
                pl.set_validshape(cur_b, [remaining, N_SQ])
                pl.set_validshape(al, [M_SQ, remaining])
                pl.set_validshape(br, [remaining, N_SQ])
            pl.load(cur_a, a, [0, ki])
            pl.load(cur_b, b_t, [0, ki], order=[1, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if ki == 0 and ki != k_total - tile_k:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
            elif ki == 0 and ki == k_total - tile_k:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Final)
            elif ki < k_total - tile_k:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
        pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)


# === Per-channel (Scaling Tile) scale + STPhase store ==============================
# Regression: the CCE Scaling-Tile store branch used to drop the phase kwarg, so the
# TSTORE_FP emission lost the STPhase unit-flag. A 4-block K-loop builds a balanced UF
# chain (matmul Partial -> matmul_acc Partial x2 -> matmul_acc Final); the per-channel
# store closes it with STPhase.Final, exercising phase forwarding on the fp-tile path.


@pl.jit(auto_mutex=True)
def t25_nsq_per_channel_scale_phase(
    a: pl.Tensor[[M_NSQ, 512], pl.DT_FP16],
    b: pl.Tensor[[512, N_NSQ], pl.DT_FP16],
    fp_params: pl.Tensor[[1, N_NSQ], pl.DT_INT64],
    out: pl.Tensor[[M_NSQ, N_NSQ], pl.DT_INT8],
):
    """NN nsq K-loop (4 blocks), per-channel Scaling-Tile scale + STPhase.Final store."""
    k_total = 512
    tile_k = 128
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M_NSQ, tile_k], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x00000,
        mutex_ids=[0, 1],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[tile_k, N_NSQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x10000,
        mutex_ids=[2, 3],
    )
    fp_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[1, N_NSQ], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Mat, layout=pl.ND),
        addrs=0x18000,
        mutex_ids=[4],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[M_NSQ, tile_k], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0,
        mutex_ids=[5, 6],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[tile_k, N_NSQ], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0,
        mutex_ids=[7, 8],
    )
    acc_grp = pl.make_tile_group(
        type=pl.TileType(
            shape=[M_NSQ, N_NSQ], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        ),
        addrs=0x0,
        mutex_ids=[9],
    )
    fp_scaling = pl.make_tile_group(
        type=pl.TileType(shape=[1, N_NSQ], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Scaling),
        addrs=0x0,
        mutex_ids=[10],
    )
    with pl.section_cube():
        ac = acc_grp.current()
        fp_tile = fp_scaling.current()
        cur_fp_l1 = fp_l1.current()
        pl.load(cur_fp_l1, fp_params, [0, 0])
        pl.move(fp_tile, cur_fp_l1)
        for ki in pl.range(0, k_total, tile_k):
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_l0a.next()
            br = b_l0b.next()
            pl.load(cur_a, a, [0, ki])
            pl.load(cur_b, b, [ki, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if ki == 0:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
            elif ki < k_total - tile_k:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
        pl.store(out, ac, [0, 0], scale=fp_tile, phase=pl.STPhase.Final)


# #####################################################################################
# Test functions
# #####################################################################################


@pytest.fixture(scope="module")
def _device():
    dev = ST_DEVICE
    _require_a5(dev)
    return dev


# --- Part A: Non-square tests ---


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t01(_device):
    a = _inputs(_device, [M_NSQ, K_NSQ])
    b = _inputs(_device, [K_NSQ, N_NSQ])
    _run(t01_nsq_nd_nz_single, a, b, a, b, [M_NSQ, N_NSQ], _device)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t02(_device):
    a = _inputs(_device, [M_NSQ, 256])
    b = _inputs(_device, [256, N_NSQ])
    _run(t02_nsq_nd_nz_kloop_if, a, b, a, b, [M_NSQ, N_NSQ], _device)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t03(_device):
    a = _inputs(_device, [M_NSQ, K_NSQ])
    b = _inputs(_device, [K_NSQ, N_NSQ])
    _run(t03_nsq_dn_zn_single, a, b.t().contiguous(), a, b, [M_NSQ, N_NSQ], _device)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t05(_device):
    a = _inputs(_device, [M_NSQ, 256])
    b = _inputs(_device, [256, N_NSQ])
    _run(t05_nsq_dn_zn_kloop, a, b.t().contiguous(), a, b, [M_NSQ, N_NSQ], _device)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t06(_device):
    a = _inputs(_device, [M_NSQ, 200])
    b = _inputs(_device, [200, N_NSQ])
    _run(t06_nsq_dn_zn_kloop_tail, a, b.t().contiguous(), a, b, [M_NSQ, N_NSQ], _device)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t07(_device):
    a = _inputs(_device, [M_NSQ, K_NSQ])
    b = _inputs(_device, [K_NSQ, N_NSQ])
    _run(t07_nsq_dn_zn_tn_single, a.t().contiguous(), b, a, b, [M_NSQ, N_NSQ], _device)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t09(_device):
    a = _inputs(_device, [M_NSQ, K_NSQ])
    b = _inputs(_device, [K_NSQ, N_NSQ])
    _run(t09_nsq_dn_zn_tn_kloop_if, a.t().contiguous(), b, a, b, [M_NSQ, N_NSQ], _device)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t10(_device):
    a = _inputs(_device, [M_NSQ, K_NSQ])
    b = _inputs(_device, [K_NSQ, N_NSQ])
    _run(t10_nsq_dn_zn_tn_kloop_tail, a.t().contiguous(), b, a, b, [M_NSQ, N_NSQ], _device)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t11(_device):
    a = _inputs(_device, [M_NSQ, K_NSQ])
    b = _inputs(_device, [K_NSQ, N_NSQ])
    _run(t11_nsq_dn_zn_tt_single, a.t().contiguous(), b.t().contiguous(), a, b, [M_NSQ, N_NSQ], _device)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t12(_device):
    a = _inputs(_device, [M_NSQ, 256])
    b = _inputs(_device, [256, N_NSQ])
    _run(t12_nsq_dn_zn_tt_kloop, a.t().contiguous(), b.t().contiguous(), a, b, [M_NSQ, N_NSQ], _device)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t13(_device):
    a = _inputs(_device, [M_NSQ, K_NSQ])
    b = _inputs(_device, [K_NSQ, N_NSQ])
    _run(t13_nsq_dn_zn_tt_kloop_if, a.t().contiguous(), b.t().contiguous(), a, b, [M_NSQ, N_NSQ], _device)


# --- Part B: Square tests ---


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t14(_device):
    a = _inputs(_device, [M_SQ, K_SQ])
    b = _inputs(_device, [K_SQ, N_SQ])
    _run(t14_sq_nd_nz_single, a, b, a, b, [M_SQ, N_SQ], _device)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t15(_device):
    a = _inputs(_device, [M_SQ, 256])
    b = _inputs(_device, [256, N_SQ])
    _run(t15_sq_nd_nz_kloop_if, a, b, a, b, [M_SQ, N_SQ], _device)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t16(_device):
    a = _inputs(_device, [128, 384])
    b = _inputs(_device, [384, 128])
    _run(t16_sq_nd_nz_kacc_3blocks, a, b, a, b, [128, 128], _device)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t17(_device):
    a = _inputs(_device, [256, 64])
    b = _inputs(_device, [64, 64])
    _run(t17_sq_nd_nz_m_tiling, a, b, a, b, [256, 64], _device)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t18(_device):
    a = _inputs(_device, [200, 128])
    b = _inputs(_device, [128, 128])
    out = torch.zeros([128, 128], device=_device, dtype=torch.float32)
    t18_sq_nd_nz_m_tail(a, b, out)
    torch.npu.synchronize()
    ref = torch.matmul(a[128:200, :].float(), b.float())
    diff = (out[:72, :] - ref).abs().max().item()
    logging.info("t18: max|out - A@B| = %s", diff)
    torch.testing.assert_close(out[:72, :], ref, rtol=2e-2, atol=2e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t19(_device):
    a = _inputs(_device, [64, 64], dtype=torch.float32)
    b = _inputs(_device, [64, 64], dtype=torch.float32)
    out = torch.zeros([64, 64], device=_device, dtype=torch.float32)
    t19_sq_nd_nz_fp32_single(a, b, out)
    torch.npu.synchronize()
    ref = torch.matmul(a, b)
    diff = (out - ref).abs().max().item()
    logging.info("t19: max|out - A@B| = %s", diff)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t20(_device):
    a = _inputs(_device, [M_SQ, K_SQ])
    b = _inputs(_device, [K_SQ, N_SQ])
    _run(t20_sq_dn_zn_single, a, b.t().contiguous(), a, b, [M_SQ, N_SQ], _device)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t21(_device):
    a = _inputs(_device, [M_SQ, K_SQ])
    b = _inputs(_device, [K_SQ, N_SQ])
    _run(t21_sq_dn_zn_kloop, a, b.t().contiguous(), a, b, [M_SQ, N_SQ], _device)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t22(_device):
    a = _inputs(_device, [M_SQ, K_SQ])
    b = _inputs(_device, [K_SQ, N_SQ])
    _run(t22_sq_dn_zn_tn_kloop_if, a.t().contiguous(), b, a, b, [M_SQ, N_SQ], _device)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t23(_device):
    a = _inputs(_device, [M_SQ, K_SQ])
    b = _inputs(_device, [K_SQ, N_SQ])
    _run(t23_sq_dn_zn_tt_kloop_if, a.t().contiguous(), b.t().contiguous(), a, b, [M_SQ, N_SQ], _device)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t24(_device):
    a = _inputs(_device, [M_SQ, 200])
    b = _inputs(_device, [200, N_SQ])
    _run(t24_sq_dn_zn_kloop_tail, a, b.t().contiguous(), a, b, [M_SQ, N_SQ], _device)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t25_per_channel_scale_phase(_device):
    # Integer-valued inputs so raw_ref * scale lands exactly on integers: no
    # round-half ambiguity between torch (half-to-even) and the hardware FixPipe,
    # so the int8 output can be checked exactly (atol=0).
    #   a[m, k] = 1, b[k, n] = v_n  ->  raw_ref[m, n] = 512 * v_n
    # scale = 1/512 (exact in fp32) then cancels K, giving expected[m, n] = v_n.
    k_dim = 512
    scale_value = 1.0 / k_dim
    scale_bits = (1 << 46) | struct.unpack("!I", struct.pack("!f", scale_value))[0]
    fp_params = torch.full((1, N_NSQ), scale_bits, device=_device, dtype=torch.int64)

    a = torch.ones([M_NSQ, k_dim], device=_device, dtype=torch.float16)
    col = (torch.arange(N_NSQ, device=_device) % 64 - 32).to(torch.float16)  # -32..31, int8-safe
    b = col.unsqueeze(0).repeat(k_dim, 1).contiguous()
    out = torch.zeros([M_NSQ, N_NSQ], device=_device, dtype=torch.int8)

    t25_nsq_per_channel_scale_phase(a, b, fp_params, out)
    torch.npu.synchronize()

    raw_ref = torch.matmul(a.float(), b.float())
    expected = torch.clamp(torch.round(raw_ref * scale_value), -128, 127).to(torch.int8)
    torch.testing.assert_close(out.to(torch.int32), expected.to(torch.int32), rtol=0, atol=0)
    logging.info("test_t25_per_channel_scale_phase passed.")


# #####################################################################################
# Main
# #####################################################################################
if __name__ == "__main__":
    dev = ST_DEVICE
    _require_a5(dev)

    all_tests = [
        # Part A: Non-square (t01 ~ t13)
        ("t01_nsq_nd_nz_single", test_t01, dev),
        ("t02_nsq_nd_nz_kloop_if", test_t02, dev),
        ("t03_nsq_dn_zn_single", test_t03, dev),
        ("t05_nsq_dn_zn_kloop", test_t05, dev),
        ("t06_nsq_dn_zn_kloop_tail", test_t06, dev),
        ("t07_nsq_dn_zn_tn_single", test_t07, dev),
        ("t09_nsq_dn_zn_tn_kloop_if", test_t09, dev),
        ("t10_nsq_dn_zn_tn_kloop_tail", test_t10, dev),
        ("t11_nsq_dn_zn_tt_single", test_t11, dev),
        ("t12_nsq_dn_zn_tt_kloop", test_t12, dev),
        ("t13_nsq_dn_zn_tt_kloop_if", test_t13, dev),
        # Part B: Square (t14 ~ t24)
        ("t14_sq_nd_nz_single", test_t14, dev),
        ("t15_sq_nd_nz_kloop_if", test_t15, dev),
        ("t16_sq_nd_nz_kacc_3blocks", test_t16, dev),
        ("t17_sq_nd_nz_m_tiling", test_t17, dev),
        ("t18_sq_nd_nz_m_tail", test_t18, dev),
        ("t19_sq_nd_nz_fp32_single", test_t19, dev),
        ("t20_sq_dn_zn_single", test_t20, dev),
        ("t21_sq_dn_zn_kloop", test_t21, dev),
        ("t22_sq_dn_zn_tn_kloop_if", test_t22, dev),
        ("t23_sq_dn_zn_tt_kloop_if", test_t23, dev),
        ("t24_sq_dn_zn_kloop_tail", test_t24, dev),
        ("t25_nsq_per_channel_scale_phase", test_t25_per_channel_scale_phase, dev),
    ]

    logging.info("=" * 80)
    logging.info("Running layout / is_transpose / shape / flow test suite")
    logging.info("  Part A: t01-t13 Non-square (M=64, K=128, N=32)")
    logging.info("  Part B: t14-t24 Square (M=K=N=128)")
    logging.info("=" * 80)

    passed = 0
    failed = 0
    for name, fn, d in all_tests:
        try:
            fn(d)
            passed += 1
            logging.info("PASS: %s", name)
        except Exception as exc:
            failed += 1
            logging.error("FAIL: %s -- %s", name, exc)

    logging.info("=" * 80)
    logging.info("Results: %d passed, %d failed, %d total", passed, failed, passed + failed)
    logging.info("=" * 80)
