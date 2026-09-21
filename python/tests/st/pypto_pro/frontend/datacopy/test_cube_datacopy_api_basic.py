# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# -----------------------------------------------------------------------------------------------------------

"""Integrated API test cases for pl.load / pl.load_tile / pl.store / pl.store_tile / pl.move.

Merged from test_load_api.py, test_load_tile_api.py, test_store_api.py,
test_store_tile_api.py, test_move_api.py. Only normal (non-abnormal) test cases.
"""

import logging
import os

from pypto_pro._errors import InvalidVal
import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

logging.basicConfig(level=logging.INFO)

TILE = 64
TILE_LARGE = 128


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


def _inputs(device, shape, dtype=torch.float16):
    torch.manual_seed(0)
    return torch.randn(shape, device=device, dtype=dtype)


# =====================================================================================
# Section 1: Basic — FP16, BF16, FP32
# =====================================================================================


@pl.jit(auto_mutex=True)
def call_kernel_basic_fp16(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    out: pl.Tensor[[TILE, TILE], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
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


@pl.jit(auto_mutex=True)
def call_kernel_basic_bf16(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    b: pl.Tensor[[TILE, TILE], pl.DT_BF16],
    out: pl.Tensor[[TILE, TILE], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
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


@pl.jit(auto_mutex=True)
def call_kernel_basic_fp32(
    a: pl.Tensor[[TILE, TILE], pl.DT_FP32],
    b: pl.Tensor[[TILE, TILE], pl.DT_FP32],
    out: pl.Tensor[[TILE, TILE], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
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


# =====================================================================================
# Section 2: Offsets — load/store row, col, both + load_tile/store_tile combined
# =====================================================================================


@pl.jit(auto_mutex=True)
def call_kernel_offset_row(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    out: pl.Tensor[[TILE_LARGE, TILE], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[4],
    )
    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.load(cur_a, a, [32, 0])
        pl.load(cur_b, b, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.store(out, ac, [32, 0])


@pl.jit(auto_mutex=True)
def call_kernel_offset_col(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    out: pl.Tensor[[TILE, TILE_LARGE], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[4],
    )
    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.load(cur_a, a, [0, 32])
        pl.load(cur_b, b, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.store(out, ac, [0, 32])


@pl.jit(auto_mutex=True)
def call_kernel_offset_both(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    out: pl.Tensor[[TILE_LARGE, TILE_LARGE], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[4],
    )
    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.load(cur_a, a, [32, 32])
        pl.load(cur_b, b, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.store(out, ac, [32, 32])


@pl.jit(auto_mutex=True)
def call_kernel_tile_offset(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    out: pl.Tensor[[TILE_LARGE, TILE_LARGE], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[4],
    )
    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.load_tile(cur_a, a, [1, 0])
        pl.load_tile(cur_b, b, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.store_tile(out, ac, [0, 1])


# =====================================================================================
# Section 3: Layout / is_transpose
# =====================================================================================


@pl.jit(auto_mutex=True)
def call_kernel_dn_transpose(
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[TILE, TILE], pl.DT_FP32],
):
    k_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=0x00000,
        mutex_ids=[0],
    )
    q_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000,
        mutex_ids=[1],
    )
    k_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    q_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[4],
    )
    with pl.section_cube():
        cur_k = k_l1.current()
        cur_q = q_l1.current()
        kl = k_l0a.current()
        qr = q_l0b.current()
        ac = c_l0c.current()
        pl.load(cur_k, k, [0, 0], order=[1, 0])
        pl.load(cur_q, q, [0, 0], order=[1, 0])
        pl.move(kl, cur_k)
        pl.move(qr, cur_q)
        pl.matmul(ac, kl, qr)
        pl.store(out, ac, [0, 0])


# =====================================================================================
# Section 4: 4D tensor + tile_dims
# =====================================================================================


@pl.jit(auto_mutex=True)
def call_kernel_4d_tile_dims(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[4],
    )
    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.load(cur_a, a, [0, 0, 0, 0], order=[2, 3])
        pl.load(cur_b, b, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.store(out, ac, [0, 0, 0, 0], order=[2, 3])


# =====================================================================================
# Section 4a: Negative test kernels — store/store_tile with descending order
# =====================================================================================


@pl.jit(auto_mutex=True)
def call_kernel_store_descending_order(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[1],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[2],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[3],
    )
    with pl.section_cube():
        cur_a = a_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.load(cur_a, a, [0, 0, 0, 0], order=[2, 3])
        pl.move(al, cur_a)
        pl.matmul(ac, al, br)
        pl.store(out, ac, [0, 0, 0, 0], order=[3, 1])


@pl.jit(auto_mutex=True)
def call_kernel_store_tile_descending_order(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[1],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[2],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[3],
    )
    with pl.section_cube():
        cur_a = a_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.load(cur_a, a, [0, 0, 0, 0], order=[2, 3])
        pl.move(al, cur_a)
        pl.matmul(ac, al, br)
        pl.store_tile(out, ac, [0, 0, 0, 0], order=[3, 1])


# =====================================================================================
# Section 5: Control flow — for-loop, if/else, while+for+if/elif/else
# =====================================================================================


@pl.jit(auto_mutex=True)
def call_kernel_for_loop(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0, 1],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000,
        mutex_ids=[2],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[3, 4],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[5],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[6, 7],
    )
    with pl.section_cube():
        cur_b = b_l1.next()
        pl.load(cur_b, b, [0, 0])
        for i in pl.range(0, 4, 1):
            cur_a = a_l1.next()
            al = a_l0a.next()
            br = b_l0b.next()
            ac = c_l0c.next()
            pl.load(cur_a, a, [i * TILE, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            pl.matmul(ac, al, br)
            pl.store(out, ac, [i * TILE, 0])


@pl.jit(auto_mutex=True)
def call_kernel_if_else(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    out: pl.Tensor[[TILE, TILE], pl.DT_FP32],
    sel: pl.DT_INT32,
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[4],
    )
    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        if sel == 0:
            pl.load(cur_a, a, [0, 0])
        else:
            pl.load(cur_a, a, [TILE, 0])
        pl.load(cur_b, b, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.store(out, ac, [0, 0])


@pl.jit(auto_mutex=True)
def call_kernel_while_for_ifelse(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    n_groups: pl.DT_INT32,
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0, 1],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000,
        mutex_ids=[2],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[3, 4],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[5],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[6, 7],
    )
    with pl.section_cube():
        cur_b = b_l1.next()
        pl.load(cur_b, b, [0, 0])
        gi = 0
        while gi < n_groups:
            for j in pl.range(0, 2, 1):
                cur_a = a_l1.next()
                al = a_l0a.next()
                br = b_l0b.next()
                ac = c_l0c.next()
                if gi < 1:
                    pl.load(cur_a, a, [j * TILE, 0])
                elif gi < 2:
                    pl.load(cur_a, a, [TILE * 2 + j * TILE, 0])
                else:
                    pl.load(cur_a, a, [gi * TILE * 2 + j * TILE, 0])
                pl.move(al, cur_a)
                pl.move(br, cur_b)
                pl.matmul(ac, al, br)
                if gi < 1:
                    pl.store(out, ac, [j * TILE, 0])
                elif gi < 2:
                    pl.store(out, ac, [TILE * 2 + j * TILE, 0])
                else:
                    pl.store(out, ac, [gi * TILE * 2 + j * TILE, 0])
            gi = gi + 1


# =====================================================================================
# Section 6: Tail blocks
# =====================================================================================


@pl.jit(auto_mutex=True)
def call_kernel_tail_row(
    a: pl.Tensor[[72, TILE_LARGE], pl.DT_FP16],
    b: pl.Tensor[[TILE_LARGE, TILE_LARGE], pl.DT_FP16],
    out: pl.Tensor[[72, TILE_LARGE], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE_LARGE, TILE_LARGE],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Mat,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_LARGE, TILE_LARGE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE_LARGE, TILE_LARGE],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Left,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_LARGE, TILE_LARGE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE_LARGE, TILE_LARGE],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addrs=0x0000,
        mutex_ids=[4],
    )
    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.set_validshape(cur_a, [72, TILE_LARGE])
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.set_validshape(al, [72, TILE_LARGE])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.set_validshape(ac, [72, TILE_LARGE])
        pl.matmul(ac, al, br)
        pl.store(out, ac, [0, 0])


@pl.jit(auto_mutex=True)
def call_kernel_tail_loop(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[TILE_LARGE, TILE_LARGE], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE_LARGE, TILE_LARGE],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Mat,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addrs=0x00000,
        mutex_ids=[0, 1],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_LARGE, TILE_LARGE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000,
        mutex_ids=[2],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE_LARGE, TILE_LARGE],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Left,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addrs=0x0000,
        mutex_ids=[3, 4],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_LARGE, TILE_LARGE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[5],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE_LARGE, TILE_LARGE],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addrs=0x0000,
        mutex_ids=[6, 7],
    )
    with pl.section_cube():
        cur_b = b_l1.next()
        pl.load(cur_b, b, [0, 0])
        for i in pl.range(0, 3, 1):
            cur_a = a_l1.next()
            al = a_l0a.next()
            br = b_l0b.next()
            ac = c_l0c.next()
            rem_r = 328 - i * TILE_LARGE
            if rem_r >= TILE_LARGE:
                pl.set_validshape(cur_a, [TILE_LARGE, TILE_LARGE])
                pl.set_validshape(al, [TILE_LARGE, TILE_LARGE])
                pl.set_validshape(ac, [TILE_LARGE, TILE_LARGE])
            else:
                pl.set_validshape(cur_a, [rem_r, TILE_LARGE])
                pl.set_validshape(al, [rem_r, TILE_LARGE])
                pl.set_validshape(ac, [rem_r, TILE_LARGE])
            pl.load(cur_a, a, [i * TILE_LARGE, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            pl.matmul(ac, al, br)
            pl.store(out, ac, [i * TILE_LARGE, 0])


@pl.jit(auto_mutex=True)
def call_kernel_tail_pad_zero(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[TILE_LARGE, TILE_LARGE], pl.DT_FP16],
    out: pl.Tensor[[TILE_LARGE, TILE_LARGE], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE_LARGE, TILE_LARGE],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Mat,
            valid_shape=[-1, -1],
            pad=pl.TilePad.zero,
            compact=1,
        ),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_LARGE, TILE_LARGE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_LARGE, TILE_LARGE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_LARGE, TILE_LARGE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_LARGE, TILE_LARGE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[4],
    )
    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.set_validshape(cur_a, [72, TILE_LARGE])
        pl.load(cur_a, a, [TILE_LARGE, 0])
        pl.load(cur_b, b, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.store(out, ac, [0, 0])


@pl.jit(auto_mutex=True)
def call_kernel_store_relu(
    a: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    b: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    out: pl.Tensor[[TILE, TILE], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
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
        pl.store(out, ac, [0, 0], relu_pre_mode=pl.ReluPreMode.NormalRelu)


@pl.jit(auto_mutex=True)
def call_kernel_store_atomic_add(
    a: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    b: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    out: pl.Tensor[[TILE, TILE], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
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
        pl.store(out, ac, [0, 0], atomic=pl.AtomicType.AtomicAdd)
        pl.store(out, ac, [0, 0], atomic=pl.AtomicType.AtomicAdd)


@pl.jit(auto_mutex=True)
def call_kernel_store_phase_final(
    a: pl.Tensor[[TILE, 192], pl.DT_FP16],
    b: pl.Tensor[[192, TILE], pl.DT_FP16],
    out: pl.Tensor[[TILE, TILE], pl.DT_FP32],
):
    tile_k = 64
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, tile_k], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0, 1],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[tile_k, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000,
        mutex_ids=[2, 3],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, tile_k], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[4, 5],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[tile_k, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[6, 7],
    )
    acc_grp = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[8],
    )
    with pl.section_cube():
        pl.system.set_mm_layout_transform(enabled=True)
        ac = acc_grp.current()
        for k in pl.range(0, 192, tile_k):
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_l0a.next()
            br = b_l0b.next()
            pl.load(cur_a, a, [0, k])
            pl.load(cur_b, b, [k, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if k == 0:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
            elif k < 128:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
        pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)
        pl.system.set_mm_layout_transform(enabled=False)


# =====================================================================================
# Section 8: Move — offset sub-block (K-axis, M-axis) + acc_to_vec (5 modes)
# =====================================================================================


@pl.jit(auto_mutex=True)
def call_kernel_move_offset_k(
    a: pl.Tensor[[TILE, TILE_LARGE], pl.DT_FP16],
    b: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    out: pl.Tensor[[TILE, TILE], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE_LARGE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
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
        pl.move(al, cur_a, [0, TILE])
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.store(out, ac, [0, 0])


@pl.jit(auto_mutex=True)
def call_kernel_move_offset_m(
    a: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    b: pl.Tensor[[TILE_LARGE, TILE], pl.DT_FP16],
    out: pl.Tensor[[TILE, TILE], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_LARGE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
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
        pl.move(br, cur_b, [TILE, 0])
        pl.matmul(ac, al, br)
        pl.store(out, ac, [0, 0])


@pl.jit(auto_mutex=True)
def call_kernel_move_acc_to_vec_single(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    out: pl.Tensor[[TILE, TILE], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[4],
    )
    vec_tile = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids=[5],
    )
    mat_res = vec_tile.current()
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
        pl.move(mat_res, ac, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)
    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
        if sub_id == 0:
            pl.store(out, mat_res, [0, 0])
        pl.system.bar_all()


@pl.jit(auto_mutex=True)
def call_kernel_move_acc_to_vec_dual_m(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    out: pl.Tensor[[TILE, TILE], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[4],
    )
    vec_tile = pl.make_tile_group(
        type=pl.TileType(shape=[32, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids=[5],
    )
    mat_res = vec_tile.current()
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
        pl.move(mat_res, ac, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)
    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
        pl.store(out, mat_res, [sub_id * 32, 0])
        pl.system.bar_all()


@pl.jit(auto_mutex=True)
def call_kernel_move_acc_to_vec_dual_n(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    out: pl.Tensor[[TILE, TILE], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[4],
    )
    vec_tile = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids=[5],
    )
    mat_res = vec_tile.current()
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
        pl.move(mat_res, ac, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)
    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
        pl.store(out, mat_res, [0, sub_id * 32])
        pl.system.bar_all()


@pl.jit(auto_mutex=True)
def call_kernel_move_acc_to_vec_relu(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    out: pl.Tensor[[TILE, TILE], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[4],
    )
    vec_tile = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids=[5],
    )
    mat_res = vec_tile.current()
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
        pl.move(mat_res, ac, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0, relu_pre_mode=pl.ReluPreMode.NormalRelu)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)
    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
        if sub_id == 0:
            pl.store(out, mat_res, [0, 0])
        pl.system.bar_all()


@pl.jit(auto_mutex=True)
def call_kernel_move_acc_to_vec_tail(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    out: pl.Tensor[[TILE, TILE], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1], compact=1
        ),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000,
        mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1], compact=1
        ),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1], compact=1
        ),
        addrs=0x0000,
        mutex_ids=[4],
    )
    vec_tile = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids=[5],
    )
    mat_res = vec_tile.current()
    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.set_validshape(cur_a, [40, TILE])
        pl.set_validshape(al, [40, TILE])
        pl.set_validshape(ac, [40, TILE])
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.move(mat_res, ac, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)
    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
        if sub_id == 0:
            pl.store(out, mat_res, [0, 0])
        pl.system.bar_all()


# =====================================================================================
# Test functions
# =====================================================================================


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_basic_fp16():
    device = ST_DEVICE
    _require_a5(device)
    a = _inputs(device, [TILE, TILE])
    b = torch.eye(TILE, device=device, dtype=torch.float16)
    out = torch.zeros([TILE, TILE], device=device, dtype=torch.float32)
    call_kernel_basic_fp16(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_basic_bf16():
    device = ST_DEVICE
    _require_a5(device)
    a = _inputs(device, [TILE, TILE], dtype=torch.bfloat16)
    b = torch.eye(TILE, device=device, dtype=torch.bfloat16)
    out = torch.zeros([TILE, TILE], device=device, dtype=torch.float32)
    call_kernel_basic_bf16(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_basic_fp32():
    device = ST_DEVICE
    _require_a5(device)
    a = _inputs(device, [TILE, TILE], dtype=torch.float32)
    b = torch.eye(TILE, device=device, dtype=torch.float32)
    out = torch.zeros([TILE, TILE], device=device, dtype=torch.float32)
    call_kernel_basic_fp32(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_offset_row():
    device = ST_DEVICE
    _require_a5(device)
    a = _inputs(device, [TILE_LARGE, TILE])
    b = torch.eye(TILE, device=device, dtype=torch.float16)
    out = torch.zeros([TILE_LARGE, TILE], device=device, dtype=torch.float32)
    call_kernel_offset_row(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out[32:96, :], a[32:96, :].float(), rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_offset_col():
    device = ST_DEVICE
    _require_a5(device)
    a = _inputs(device, [TILE, TILE_LARGE])
    b = torch.eye(TILE, device=device, dtype=torch.float16)
    out = torch.zeros([TILE, TILE_LARGE], device=device, dtype=torch.float32)
    call_kernel_offset_col(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out[:, 32:96], a[:, 32:96].float(), rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_offset_both():
    device = ST_DEVICE
    _require_a5(device)
    a = _inputs(device, [TILE_LARGE, TILE_LARGE])
    b = torch.eye(TILE, device=device, dtype=torch.float16)
    out = torch.zeros([TILE_LARGE, TILE_LARGE], device=device, dtype=torch.float32)
    call_kernel_offset_both(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out[32:96, 32:96], a[32:96, 32:96].float(), rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_tile_offset():
    device = ST_DEVICE
    _require_a5(device)
    a = _inputs(device, [TILE_LARGE, TILE])
    b = torch.eye(TILE, device=device, dtype=torch.float16)
    out = torch.zeros([TILE_LARGE, TILE_LARGE], device=device, dtype=torch.float32)
    call_kernel_tile_offset(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out[:TILE, TILE:], a[TILE:, :].float(), rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dn_transpose():
    device = ST_DEVICE
    _require_a5(device)
    k = _inputs(device, [TILE, TILE])
    q = _inputs(device, [TILE, TILE])
    out = torch.zeros([TILE, TILE], device=device, dtype=torch.float32)
    call_kernel_dn_transpose(k, q, out)
    torch.npu.synchronize()
    ref = torch.matmul(k.float().T, q.float().T)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_4d_tile_dims():
    device = ST_DEVICE
    _require_a5(device)
    a = _inputs(device, [2, 4, TILE, TILE])
    b = torch.eye(TILE, device=device, dtype=torch.float16)
    out = torch.zeros([2, 4, TILE, TILE], device=device, dtype=torch.float32)
    call_kernel_4d_tile_dims(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out[0, 0, :TILE, :TILE], a[0, 0, :TILE, :TILE].float(), rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_for_loop():
    device = ST_DEVICE
    _require_a5(device)
    a = _inputs(device, [256, TILE])
    b = torch.eye(TILE, device=device, dtype=torch.float16)
    out = torch.zeros([256, TILE], device=device, dtype=torch.float32)
    call_kernel_for_loop(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_if_else():
    device = ST_DEVICE
    _require_a5(device)
    a = _inputs(device, [TILE_LARGE, TILE])
    b = torch.eye(TILE, device=device, dtype=torch.float16)
    out0 = torch.zeros([TILE, TILE], device=device, dtype=torch.float32)
    out1 = torch.zeros([TILE, TILE], device=device, dtype=torch.float32)
    call_kernel_if_else(a, b, out0, 0)
    call_kernel_if_else(a, b, out1, 1)
    torch.npu.synchronize()
    torch.testing.assert_close(out0, a[:TILE, :].float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(out1, a[TILE:, :].float(), rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_while_for_ifelse():
    device = ST_DEVICE
    _require_a5(device)
    a = _inputs(device, [256, TILE])
    b = torch.eye(TILE, device=device, dtype=torch.float16)
    out = torch.zeros([256, TILE], device=device, dtype=torch.float32)
    call_kernel_while_for_ifelse(a, b, out, 2)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_tail_row():
    device = ST_DEVICE
    _require_a5(device)
    a = _inputs(device, [72, TILE_LARGE])
    b = torch.eye(TILE_LARGE, device=device, dtype=torch.float16)
    out = torch.zeros([72, TILE_LARGE], device=device, dtype=torch.float32)
    call_kernel_tail_row(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.matmul(a.float(), b.float()), rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_tail_loop():
    device = ST_DEVICE
    _require_a5(device)
    a = _inputs(device, [328, TILE_LARGE])
    b = torch.eye(TILE_LARGE, device=device, dtype=torch.float16)
    out = torch.zeros([328, TILE_LARGE], device=device, dtype=torch.float32)
    call_kernel_tail_loop(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.matmul(a.float(), b.float()), rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_tail_pad_zero():
    device = ST_DEVICE
    _require_a5(device)
    a = _inputs(device, [200, TILE_LARGE])
    b = torch.eye(TILE_LARGE, device=device, dtype=torch.float16)
    out = torch.zeros([TILE_LARGE, TILE_LARGE], device=device, dtype=torch.float32)
    call_kernel_tail_pad_zero(a, b, out)
    torch.npu.synchronize()
    ref = torch.matmul(a[TILE_LARGE:, :].float(), b.float())
    torch.testing.assert_close(out[:72, :], ref, rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_store_relu():
    device = ST_DEVICE
    _require_a5(device)
    a = _inputs(device, [TILE, TILE])
    b = _inputs(device, [TILE, TILE])
    out = torch.zeros([TILE, TILE], device=device, dtype=torch.float32)
    call_kernel_store_relu(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.relu(torch.matmul(a.float(), b.float())), rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_store_atomic_add():
    device = ST_DEVICE
    _require_a5(device)
    a = _inputs(device, [TILE, TILE])
    b = torch.eye(TILE, device=device, dtype=torch.float16)
    out = torch.zeros([TILE, TILE], device=device, dtype=torch.float32)
    call_kernel_store_atomic_add[None, 1](a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a.float() * 2, rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_store_phase_final():
    device = ST_DEVICE
    _require_a5(device)
    a = _inputs(device, [TILE, 192])
    b = _inputs(device, [192, TILE])
    out = torch.zeros([TILE, TILE], device=device, dtype=torch.float32)
    call_kernel_store_phase_final(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.matmul(a.float(), b.float()), rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_move_offset_k():
    device = ST_DEVICE
    _require_a5(device)
    a = _inputs(device, [TILE, TILE_LARGE])
    b = torch.eye(TILE, device=device, dtype=torch.float16)
    out = torch.zeros([TILE, TILE], device=device, dtype=torch.float32)
    call_kernel_move_offset_k(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.matmul(a[:, TILE:].float(), b.float()), rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_move_offset_m():
    device = ST_DEVICE
    _require_a5(device)
    a = _inputs(device, [TILE, TILE])
    b = _inputs(device, [TILE_LARGE, TILE])
    out = torch.zeros([TILE, TILE], device=device, dtype=torch.float32)
    call_kernel_move_offset_m(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.matmul(a.float(), b[TILE:, :].float()), rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_move_acc_to_vec_single():
    device = ST_DEVICE
    _require_a5(device)
    a = _inputs(device, [TILE, TILE])
    b = _inputs(device, [TILE, TILE])
    out = torch.zeros([TILE, TILE], device=device, dtype=torch.float32)
    call_kernel_move_acc_to_vec_single(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.matmul(a.float(), b.float()), rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_move_acc_to_vec_dual_m():
    device = ST_DEVICE
    _require_a5(device)
    a = _inputs(device, [TILE, TILE])
    b = _inputs(device, [TILE, TILE])
    out = torch.zeros([TILE, TILE], device=device, dtype=torch.float32)
    call_kernel_move_acc_to_vec_dual_m(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.matmul(a.float(), b.float()), rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_move_acc_to_vec_dual_n():
    device = ST_DEVICE
    _require_a5(device)
    a = _inputs(device, [TILE, TILE])
    b = _inputs(device, [TILE, TILE])
    out = torch.zeros([TILE, TILE], device=device, dtype=torch.float32)
    call_kernel_move_acc_to_vec_dual_n(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.matmul(a.float(), b.float()), rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_move_acc_to_vec_relu():
    device = ST_DEVICE
    _require_a5(device)
    a = _inputs(device, [TILE, TILE])
    b = torch.eye(TILE, device=device, dtype=torch.float16)
    out = torch.zeros([TILE, TILE], device=device, dtype=torch.float32)
    call_kernel_move_acc_to_vec_relu(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.relu(a.float()), rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_move_acc_to_vec_tail():
    device = ST_DEVICE
    _require_a5(device)
    a = _inputs(device, [40, TILE])
    b = torch.eye(TILE, device=device, dtype=torch.float16)
    out = torch.zeros([TILE, TILE], device=device, dtype=torch.float32)
    call_kernel_move_acc_to_vec_tail(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out[:40, :], a.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_store_order_descending():
    device = ST_DEVICE
    _require_a5(device)
    a = _inputs(device, [2, 4, TILE, TILE])
    out = torch.zeros([2, 4, TILE, TILE], device=device, dtype=torch.float32)
    with pytest.raises(InvalidVal, match="order must be ascending"):
        call_kernel_store_descending_order(a, out)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_store_tile_order_descending():
    device = ST_DEVICE
    _require_a5(device)
    a = _inputs(device, [2, 4, TILE, TILE])
    out = torch.zeros([2, 4, TILE, TILE], device=device, dtype=torch.float32)
    with pytest.raises(InvalidVal, match="order must be ascending"):
        call_kernel_store_tile_descending_order(a, out)


if __name__ == "__main__":
    all_tests = [
        test_basic_fp16,
        test_basic_bf16,
        test_basic_fp32,
        test_offset_row,
        test_offset_col,
        test_offset_both,
        test_tile_offset,
        test_dn_transpose,
        test_4d_tile_dims,
        test_for_loop,
        test_if_else,
        test_while_for_ifelse,
        test_tail_row,
        test_tail_loop,
        test_tail_pad_zero,
        test_store_relu,
        test_store_atomic_add,
        test_store_phase_final,
        test_move_offset_k,
        test_move_offset_m,
        test_move_acc_to_vec_single,
        test_move_acc_to_vec_dual_m,
        test_move_acc_to_vec_dual_n,
        test_move_acc_to_vec_relu,
        test_move_acc_to_vec_tail,
        test_store_order_descending,
        test_store_tile_order_descending,
    ]
    logging.info("Running integrated API test suite")
    passed = 0
    failed = 0
    for fn in all_tests:
        try:
            fn()
            passed += 1
            logging.info("PASS: %s", fn.__name__)
        except Exception as exc:
            failed += 1
            logging.error("FAIL: %s -- %s", fn.__name__, exc)
    logging.info("Results: %d passed, %d failed, %d total", passed, failed, passed + failed)
