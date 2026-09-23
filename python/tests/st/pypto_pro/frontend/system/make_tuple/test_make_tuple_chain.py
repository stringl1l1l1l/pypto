# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""NPU ST for make_tuple chain calls: t.a.b attribute access and t.field.next().

Covers:
  - Nested make_tuple chain attribute access (t.a.b)
  - make_tuple wrapping a tile_group with chain .next() / .current()
  - Three-level chain (t.a.b.v)
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


# =============================================================================
# Chain attribute access: t.a.b (nested make_tuple)
# =============================================================================

@pl.jit(arch="3510", auto_mutex=True)
def make_tuple_chain_attr_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    inner = pl.make_tuple(x=10, y=20)
    outer = pl.make_tuple(a=inner)
    with pl.section_vector():
        val: pl.DT_INT32 = outer.a.x + outer.a.y
        pl.setval(out, 0, val)


@pytest.mark.soc("950")
def test_make_tuple_chain_attr():
    _require_a5(ST_DEVICE)
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    make_tuple_chain_attr_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([30], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Three-level chain: t.a.b.v
# =============================================================================

@pl.jit(arch="3510", auto_mutex=True)
def make_tuple_chain_three_level_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    leaf = pl.make_tuple(v=42)
    mid = pl.make_tuple(b=leaf)
    root = pl.make_tuple(a=mid)
    with pl.section_vector():
        pl.setval(out, 0, root.a.b.v)


@pytest.mark.soc("950")
def test_make_tuple_chain_three_level():
    _require_a5(ST_DEVICE)
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    make_tuple_chain_three_level_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([42], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# make_tuple wrapping tile_group: t.field.next()
# =============================================================================

@pl.jit(arch="3510", auto_mutex=True)
def make_tuple_chain_tile_group_next_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tt, addrs=[0x00000], mutex_ids=[20])
    tensor_info = pl.make_tuple(acc_db=a_db)
    with pl.section_vector():
        t = tensor_info.acc_db.next()
        pl.load(t, a, [0, 0])
        pl.exp(t, t)
        pl.store(a, t, [0, 0])


@pytest.mark.soc("950")
def test_make_tuple_chain_tile_group_next():
    _require_a5(ST_DEVICE)
    a = torch.randn(64, 128, device=ST_DEVICE, dtype=torch.float16)
    expected = a.float().exp().half()
    make_tuple_chain_tile_group_next_kernel[None, 1](a)
    torch.npu.synchronize()
    assert torch.allclose(a, expected, rtol=1e-3, atol=1e-3), f"max diff: {(a - expected).abs().max().item()}"


# =============================================================================
# make_tuple wrapping tile_group: t.field.current()
# =============================================================================

@pl.jit(arch="3510", auto_mutex=True)
def make_tuple_chain_tile_group_current_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tt, addrs=[0x00000], mutex_ids=[20])
    tensor_info = pl.make_tuple(buf=a_db)
    with pl.section_vector():
        t = tensor_info.buf.next()
        pl.load(t, a, [0, 0])
        pl.exp(t, t)
        cur = tensor_info.buf.current()
        pl.store(a, cur, [0, 0])


@pytest.mark.soc("950")
def test_make_tuple_chain_tile_group_current():
    _require_a5(ST_DEVICE)
    a = torch.randn(64, 128, device=ST_DEVICE, dtype=torch.float16)
    expected = a.float().exp().half()
    make_tuple_chain_tile_group_current_kernel[None, 1](a)
    torch.npu.synchronize()
    assert torch.allclose(a, expected, rtol=1e-3, atol=1e-3), f"max diff: {(a - expected).abs().max().item()}"


# =============================================================================
# Four-level chain: t.a.b.c.d
# =============================================================================

@pl.jit(arch="3510", auto_mutex=True)
def make_tuple_chain_four_level_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    l1 = pl.make_tuple(w=1)
    l2 = pl.make_tuple(c=l1)
    l3 = pl.make_tuple(b=l2)
    root = pl.make_tuple(a=l3)
    with pl.section_vector():
        pl.setval(out, 0, root.a.b.c.w)


@pytest.mark.soc("950")
def test_make_tuple_chain_four_level():
    _require_a5(ST_DEVICE)
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    make_tuple_chain_four_level_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([1], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Chain reads in arithmetic: t.a.x * 2 + t.a.y
# =============================================================================

@pl.jit(arch="3510", auto_mutex=True)
def make_tuple_chain_arithmetic_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    inner = pl.make_tuple(x=10, y=20)
    outer = pl.make_tuple(a=inner)
    with pl.section_vector():
        pl.setval(out, 0, outer.a.x * 2 + outer.a.y)


@pytest.mark.soc("950")
def test_make_tuple_chain_arithmetic():
    _require_a5(ST_DEVICE)
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    make_tuple_chain_arithmetic_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([40], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# make_tuple wrapping a struct_array element: t.s.v
# =============================================================================

@pl.jit(arch="3510", auto_mutex=True)
def make_tuple_chain_struct_array_elem_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    slots = pl.struct_array(2, "Slot", v=0)
    t = pl.make_tuple(s=slots[0])
    with pl.section_vector():
        slots[0].v = 7
        pl.setval(out, 0, t.s.v)


@pytest.mark.soc("950")
def test_make_tuple_chain_struct_array_elem():
    _require_a5(ST_DEVICE)
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    make_tuple_chain_struct_array_elem_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([7], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# make_tuple wrapping tile_group: t.field.previous()
# =============================================================================

@pl.jit(arch="3510", auto_mutex=True)
def make_tuple_chain_tile_group_previous_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tt, addrs=[0x00000, 0x10000], mutex_ids=[20, 21])
    tensor_info = pl.make_tuple(acc_db=a_db)
    with pl.section_vector():
        t = tensor_info.acc_db.previous()
        pl.load(t, a, [0, 0])
        pl.exp(t, t)
        pl.store(a, t, [0, 0])


@pytest.mark.soc("950")
def test_make_tuple_chain_tile_group_previous():
    _require_a5(ST_DEVICE)
    a = torch.randn(64, 128, device=ST_DEVICE, dtype=torch.float16)
    expected = a.float().exp().half()
    make_tuple_chain_tile_group_previous_kernel[None, 1](a)
    torch.npu.synchronize()
    assert torch.allclose(a, expected, rtol=1e-3, atol=1e-3), f"max diff: {(a - expected).abs().max().item()}"


# =============================================================================
# make_tuple wrapping tile_group: double buffer via t.field.next() in a loop
# =============================================================================

@pl.jit(arch="3510", auto_mutex=True)
def make_tuple_chain_double_buffer_loop_kernel(
    a: pl.Tensor[[128, 128], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tt, addrs=[0x00000, 0x10000], mutex_ids=[20, 21])
    tensor_info = pl.make_tuple(acc_db=a_db)
    with pl.section_vector():
        for i in pl.range(0, 2):
            t = tensor_info.acc_db.next()
            pl.load(t, a, [i * 64, 0])
            pl.exp(t, t)
            pl.store(a, t, [i * 64, 0])


@pytest.mark.soc("950")
def test_make_tuple_chain_double_buffer_loop():
    _require_a5(ST_DEVICE)
    a = torch.randn(128, 128, device=ST_DEVICE, dtype=torch.float16)
    expected = a.float().exp().half()
    make_tuple_chain_double_buffer_loop_kernel[None, 1](a)
    torch.npu.synchronize()
    assert torch.allclose(a, expected, rtol=1e-3, atol=1e-3), f"max diff: {(a - expected).abs().max().item()}"


# =============================================================================
# make_tuple wrapping tile_group: ring buffer wraps around 3 slots
# =============================================================================

@pl.jit(arch="3510", auto_mutex=True)
def make_tuple_chain_ring_buffer_loop_kernel(
    a: pl.Tensor[[256, 128], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tt, addrs=[0x00000, 0x10000, 0x20000], mutex_ids=[20, 21, 22])
    tensor_info = pl.make_tuple(acc_db=a_db)
    with pl.section_vector():
        for i in pl.range(0, 4):
            t = tensor_info.acc_db.next()
            pl.load(t, a, [i * 64, 0])
            pl.exp(t, t)
            pl.store(a, t, [i * 64, 0])


@pytest.mark.soc("950")
def test_make_tuple_chain_ring_buffer_loop():
    _require_a5(ST_DEVICE)
    a = torch.randn(256, 128, device=ST_DEVICE, dtype=torch.float16)
    expected = a.float().exp().half()
    make_tuple_chain_ring_buffer_loop_kernel[None, 1](a)
    torch.npu.synchronize()
    assert torch.allclose(a, expected, rtol=1e-3, atol=1e-3), f"max diff: {(a - expected).abs().max().item()}"


# =============================================================================
# Chain reads accumulated in a for loop
# =============================================================================

@pl.jit(arch="3510", auto_mutex=True)
def make_tuple_chain_loop_accum_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    inner = pl.make_tuple(x=10)
    outer = pl.make_tuple(a=inner)
    total = 0
    with pl.section_vector():
        for i in pl.range(0, 4):
            total = total + outer.a.x
        pl.setval(out, 0, total)


@pytest.mark.soc("950")
def test_make_tuple_chain_loop_accum():
    _require_a5(ST_DEVICE)
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    make_tuple_chain_loop_accum_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([40], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Chain reads selected by an if/else branch
# =============================================================================

@pl.jit(arch="3510", auto_mutex=True)
def make_tuple_chain_if_branch_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    inner = pl.make_tuple(x=10, y=20)
    outer = pl.make_tuple(a=inner)
    total = 0
    with pl.section_vector():
        for i in pl.range(0, 4):
            if i < 2:
                total = total + outer.a.x
            else:
                total = total + outer.a.y
        pl.setval(out, 0, total)


@pytest.mark.soc("950")
def test_make_tuple_chain_if_branch():
    _require_a5(ST_DEVICE)
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    make_tuple_chain_if_branch_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([60], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Standalone runner
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tests = [
        test_make_tuple_chain_attr,
        test_make_tuple_chain_three_level,
        test_make_tuple_chain_tile_group_next,
        test_make_tuple_chain_tile_group_current,
        test_make_tuple_chain_four_level,
        test_make_tuple_chain_arithmetic,
        test_make_tuple_chain_struct_array_elem,
        test_make_tuple_chain_tile_group_previous,
        test_make_tuple_chain_double_buffer_loop,
        test_make_tuple_chain_ring_buffer_loop,
        test_make_tuple_chain_loop_accum,
        test_make_tuple_chain_if_branch,
    ]
    for t in tests:
        t()
        logging.info("%s passed!", t.__name__)
    logging.info("All make_tuple chain call ST tests passed!")
