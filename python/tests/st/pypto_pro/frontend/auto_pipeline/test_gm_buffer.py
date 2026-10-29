#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software and you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""GM (workspace) views as cross-core buffers: ``pl.make_tensor(..., depth=, fwd_ids=, bwd_ids=)``.

Two three-stage kernels, each handing data over through two GM buffers, one per direction:

  * VCV: vector -> gm_a -> cube -> gm_b -> vector
  * CVC: cube -> gm_a -> vector -> gm_b -> cube

The third stage is the second one on its core, so it is the one ``preload`` delays; gm_b,
which feeds it, needs ``max(preload, 1)`` slots, gm_a needs 2 once anything is pipelined.

Both buffers live in one uint8 workspace and reinterpret it with ``dtype=``. The workspace
is filled with a sentinel first, so a slot written at the wrong offset or with the wrong
size shows up as a changed byte in a row's padding or past the last slot.

Two more, smaller kernels (vector -> gm_a -> cube) write their slot shape as a list of
runtime expressions, so the slot offset is computed on device instead of folded into a
constant. One declares the slot with two axes, the other with three.
"""

import ast
import logging
import os
import re

import pypto_pro.language as pl
from pypto_pro.runtime.pipeline._analyzer import probe_kernel_facts
from pypto_pro.runtime.pipeline._transformer import transform_pipeline
from pypto_pro.runtime.platform import NpuArch
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

TM = 64  # rows per task
K = 64
N = 64
P = 64
NT = 6  # tasks per core: enough for every depth below to wrap around
CORES = 2  # GM is shared by every core, so each one works on its own band and its own region
S0, S1 = 2.0, 0.5
SENTINEL = 0x7B
GUARD = 4096  # bytes after each buffer that must stay untouched
ATOL = 5e-2


def _slots(preload: int) -> tuple[int, int]:
    """(gm_a, gm_b) slot counts a preload needs: the consumer's delay minus the producer's, plus one."""
    return (2 if preload else 1), max(preload, 1)


def _ids(first: int, count: int) -> list[int]:
    return list(range(first, first + count))


class _Layout:
    """Where the two buffers sit in the workspace, in bytes and in elements."""

    def __init__(self, depth_a, depth_b, pitched, a_cols, a_bytes, b_cols, b_bytes):
        self.pitch_a = 2 * a_cols if pitched else a_cols
        self.pitch_b = 2 * b_cols if pitched else b_cols
        self.a_cols, self.b_cols = a_cols, b_cols
        self.a_elem, self.b_elem = a_bytes, b_bytes
        self.depth_a, self.depth_b = depth_a, depth_b
        self.a_size = depth_a * TM * self.pitch_a * a_bytes
        self.b_off = (self.a_size + GUARD + 511) // 512 * 512
        self.b_size = depth_b * TM * self.pitch_b * b_bytes
        self.per_core = (self.b_off + self.b_size + GUARD + 511) // 512 * 512
        self.total = CORES * self.per_core


def _check_untouched(ws: torch.Tensor, lay: _Layout):
    """Every byte outside the slots' valid rows still holds the sentinel, in every core's region."""
    ws = ws.cpu()
    for core in range(CORES):
        base = core * lay.per_core
        for off, depth, pitch, cols, elem in (
            (0, lay.depth_a, lay.pitch_a, lay.a_cols, lay.a_elem),
            (lay.b_off, lay.depth_b, lay.pitch_b, lay.b_cols, lay.b_elem),
        ):
            start = base + off
            region = ws[start:start + depth * TM * pitch * elem].view(depth, TM, pitch * elem)
            assert bool((region[:, :, cols * elem:] == SENTINEL).all()), "a slot wrote into its row padding"
        assert bool((ws[base + lay.a_size:base + lay.b_off] == SENTINEL).all()), "gm_a wrote past its last slot"
        tail = ws[base + lay.b_off + lay.b_size:base + lay.per_core]
        assert bool((tail == SENTINEL).all()), "gm_b wrote past its last slot"


# ---------------------------------------------------------------------------
# VCV: vector -> gm_a (fp16) -> cube -> gm_b (fp32) -> vector
# ---------------------------------------------------------------------------


@pl.pipeline.stage
def vcv_scale(k, src, gm_a, ub_a):
    t = ub_a.next()
    pl.load(t, src, [k * TM, 0])
    pl.muls(t, t, S0)
    pl.store(gm_a, t, [0, 0])


@pl.pipeline.stage
def vcv_matmul(gm_a, b, gm_b, mat_a, mat_b, left, right, acc):
    ma = mat_a.next()
    mb = mat_b.next()
    lt = left.next()
    rt = right.next()
    c = acc.next()
    pl.load(ma, gm_a, [0, 0])
    pl.load(mb, b, [0, 0])
    pl.move(lt, ma)
    pl.move(rt, mb)
    pl.matmul(c, lt, rt)
    pl.store(gm_b, c, [0, 0])


@pl.pipeline.stage
def vcv_post(k, gm_b, out, ub_b):
    t = ub_b.next()
    pl.load(t, gm_b, [0, 0])
    pl.muls(t, t, S1)
    pl.store(out, t, [k * TM, 0])


def _vcv_kernel(preload: int, depth_a: int, depth_b: int, pitched: bool, name: str):
    lay = _Layout(depth_a, depth_b, pitched, K, 2, N, 4)
    stride_a, stride_b = [lay.pitch_a, 1], [lay.pitch_b, 1]
    off_b, per_core = lay.b_off, lay.per_core
    fwd_a, bwd_a = _ids(0, depth_a), _ids(depth_a, depth_a)
    fwd_b, bwd_b = _ids(2 * depth_a, depth_b), _ids(2 * depth_a + depth_b, depth_b)

    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=True, name=name, pipeline=pl.pipeline.PipelineConfig(preload=preload))
    def vcv(
        src: pl.Tensor[[CORES * NT * TM, K], pl.DT_FP16],
        b: pl.Tensor[[K, N], pl.DT_FP16],
        ws: pl.Ptr[pl.DT_UINT8],
        out: pl.Tensor[[CORES * NT * TM, N], pl.DT_FP32],
    ):
        # One workspace region per core, and the band of tasks that core owns. The cube and the
        # vector side must agree on both, hence the task ration in the id.
        core_id = pl.get_block_idx() // pl.get_subblock_num()
        task_base = core_id * NT
        ws_core = pl.addptr(ws, core_id * per_core)
        gm_a = pl.make_tensor(ws_core, [TM, K], stride_a, dtype=pl.DT_FP16,
                              depth=depth_a, fwd_ids=fwd_a, bwd_ids=bwd_a)
        gm_b = pl.make_tensor(pl.addptr(ws_core, off_b), [TM, N], stride_b, dtype=pl.DT_FP32,
                              depth=depth_b, fwd_ids=fwd_b, bwd_ids=bwd_b)
        with pl.section_cube():
            mat_a = pl.make_tile_group(
                type=pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
                addrs=0x00000, mutex_ids=[0, 1])
            mat_b = pl.make_tile_group(
                type=pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
                addrs=0x10000, mutex_ids=[2, 3])
            left = pl.make_tile_group(
                type=pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
                addrs=0x0000, mutex_ids=[4])
            right = pl.make_tile_group(
                type=pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
                addrs=0x0000, mutex_ids=[5])
            acc = pl.make_tile_group(
                type=pl.TileType(shape=[TM, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
                addrs=0x0000, mutex_ids=[6])
        with pl.section_vector():
            ub_a = pl.make_tile_group(
                type=pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
                addrs=0x00000, mutex_ids=[7, 8])
            ub_b = pl.make_tile_group(
                type=pl.TileType(shape=[TM, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                addrs=0x08000, mutex_ids=[9, 10])

        for i in pl.range(0, NT):
            with pl.section_vector():
                vcv_scale(task_base + i, src, gm_a, ub_a)
            with pl.section_cube():
                vcv_matmul(gm_a, b, gm_b, mat_a, mat_b, left, right, acc)
            with pl.section_vector():
                vcv_post(task_base + i, gm_b, out, ub_b)

    return vcv, lay


def _run_vcv(kernel, lay):
    torch.manual_seed(11)
    src = torch.rand((CORES * NT * TM, K), device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    b = torch.rand((K, N), device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    ws = torch.full((lay.total,), SENTINEL, device=ST_DEVICE, dtype=torch.uint8)
    out = torch.zeros((CORES * NT * TM, N), device=ST_DEVICE, dtype=torch.float32)
    kernel[None, CORES](src, b, ws, out)
    torch.npu.synchronize()
    scaled = (src.cpu().float() * S0).half().float()
    golden = (scaled @ b.cpu().float()) * S1
    return out.cpu(), golden, ws


# ---------------------------------------------------------------------------
# CVC: cube -> gm_a (fp32) -> vector -> gm_b (fp16) -> cube
# ---------------------------------------------------------------------------


@pl.pipeline.stage
def cvc_matmul(k, a, b, gm_a, mat_a, mat_b, left, right, acc):
    ma = mat_a.next()
    mb = mat_b.next()
    lt = left.next()
    rt = right.next()
    c = acc.next()
    pl.load(ma, a, [k * TM, 0])
    pl.load(mb, b, [0, 0])
    pl.move(lt, ma)
    pl.move(rt, mb)
    pl.matmul(c, lt, rt)
    pl.store(gm_a, c, [0, 0])


@pl.pipeline.stage
def cvc_scale(gm_a, gm_b, ub32, ub16):
    t = ub32.next()
    h = ub16.next()
    pl.load(t, gm_a, [0, 0])
    pl.muls(t, t, S0)
    pl.cast(h, t)
    pl.store(gm_b, h, [0, 0])


@pl.pipeline.stage
def cvc_matmul2(k, gm_b, c_in, out, mat_a, mat_b, left, right, acc):
    ma = mat_a.next()
    mb = mat_b.next()
    lt = left.next()
    rt = right.next()
    c = acc.next()
    pl.load(ma, gm_b, [0, 0])
    pl.load(mb, c_in, [0, 0])
    pl.move(lt, ma)
    pl.move(rt, mb)
    pl.matmul(c, lt, rt)
    pl.store(out, c, [k * TM, 0])


def _cvc_kernel(preload: int, depth_a: int, depth_b: int, pitched: bool, name: str):
    lay = _Layout(depth_a, depth_b, pitched, N, 4, N, 2)
    stride_a, stride_b = [lay.pitch_a, 1], [lay.pitch_b, 1]
    off_b, per_core = lay.b_off, lay.per_core
    fwd_a, bwd_a = _ids(0, depth_a), _ids(depth_a, depth_a)
    fwd_b, bwd_b = _ids(2 * depth_a, depth_b), _ids(2 * depth_a + depth_b, depth_b)

    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=True, name=name, pipeline=pl.pipeline.PipelineConfig(preload=preload))
    def cvc(
        a: pl.Tensor[[CORES * NT * TM, K], pl.DT_FP16],
        b: pl.Tensor[[K, N], pl.DT_FP16],
        c_in: pl.Tensor[[N, P], pl.DT_FP16],
        ws: pl.Ptr[pl.DT_UINT8],
        out: pl.Tensor[[CORES * NT * TM, P], pl.DT_FP32],
    ):
        core_id = pl.get_block_idx() // pl.get_subblock_num()
        task_base = core_id * NT
        ws_core = pl.addptr(ws, core_id * per_core)
        gm_a = pl.make_tensor(ws_core, [TM, N], stride_a, dtype=pl.DT_FP32,
                              depth=depth_a, fwd_ids=fwd_a, bwd_ids=bwd_a)
        gm_b = pl.make_tensor(pl.addptr(ws_core, off_b), [TM, N], stride_b, dtype=pl.DT_FP16,
                              depth=depth_b, fwd_ids=fwd_b, bwd_ids=bwd_b)
        with pl.section_cube():
            mat_a = pl.make_tile_group(
                type=pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
                addrs=0x00000, mutex_ids=[0, 1])
            mat_b = pl.make_tile_group(
                type=pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
                addrs=0x10000, mutex_ids=[2, 3])
            left = pl.make_tile_group(
                type=pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
                addrs=0x0000, mutex_ids=[4])
            right = pl.make_tile_group(
                type=pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
                addrs=0x0000, mutex_ids=[5])
            acc = pl.make_tile_group(
                type=pl.TileType(shape=[TM, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
                addrs=0x0000, mutex_ids=[6])
        with pl.section_vector():
            ub32 = pl.make_tile_group(
                type=pl.TileType(shape=[TM, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                addrs=0x00000, mutex_ids=[7, 8])
            ub16 = pl.make_tile_group(
                type=pl.TileType(shape=[TM, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
                addrs=0x10000, mutex_ids=[9, 10])

        for i in pl.range(0, NT):
            with pl.section_cube():
                cvc_matmul(task_base + i, a, b, gm_a, mat_a, mat_b, left, right, acc)
            with pl.section_vector():
                cvc_scale(gm_a, gm_b, ub32, ub16)
            with pl.section_cube():
                cvc_matmul2(task_base + i, gm_b, c_in, out, mat_a, mat_b, left, right, acc)

    return cvc, lay


def _run_cvc(kernel, lay):
    torch.manual_seed(13)
    a = torch.rand((CORES * NT * TM, K), device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    b = torch.rand((K, N), device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    c_in = torch.rand((N, P), device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    ws = torch.full((lay.total,), SENTINEL, device=ST_DEVICE, dtype=torch.uint8)
    out = torch.zeros((CORES * NT * TM, P), device=ST_DEVICE, dtype=torch.float32)
    kernel[None, CORES](a, b, c_in, ws, out)
    torch.npu.synchronize()
    mid = ((a.cpu().float() @ b.cpu().float()) * S0).half().float()
    golden = mid @ c_in.cpu().float()
    return out.cpu(), golden, ws


_BUILDERS = {"vcv": (_vcv_kernel, _run_vcv), "cvc": (_cvc_kernel, _run_cvc)}

# (kind, preload, slack slots on gm_b beyond the minimum, pitched rows)
_CASES = [
    *[(kind, p, 0, True) for kind in ("vcv", "cvc") for p in (0, 1, 2, 3)],
    *[(kind, 2, 0, False) for kind in ("vcv", "cvc")],
    *[(kind, p, 1, True) for kind in ("vcv", "cvc") for p in (0, 2)],
]


def _case_name(kind, preload, slack, pitched):
    return f"gm_{kind}_preload{preload}_slack{slack}_{'pitched' if pitched else 'contig'}"


@pytest.mark.soc("950")
@pytest.mark.parametrize("kind, preload, slack, pitched", _CASES, ids=[_case_name(*c) for c in _CASES])
def test_gm_buffer(kind, preload, slack, pitched):
    torch.npu.set_device(ST_DEVICE)
    build, run = _BUILDERS[kind]
    depth_a, depth_b = _slots(preload)
    name = _case_name(kind, preload, slack, pitched)
    kernel, lay = build(preload, depth_a, depth_b + slack, pitched, name)

    out, golden, ws = run(kernel, lay)

    torch.testing.assert_close(out, golden, rtol=1e-2, atol=ATOL)
    _check_untouched(ws, lay)
    logging.info("%s passed, max|diff| = %.6f", name, (out - golden).abs().max().item())


# ---------------------------------------------------------------------------
# Generated code
# ---------------------------------------------------------------------------


def _generated(kernel) -> str:
    """The transformed kernel as source, the way the framework hands it to codegen."""
    kd = kernel.to_kernel_def()
    if_const_map, var_types, arg_types, field_types = probe_kernel_facts(kd, None)
    tree = transform_pipeline(
        kd._func_def, kd._closure_vars, kd._pipeline,
        if_const_map=if_const_map, var_types=var_types, arg_types=arg_types, field_types=field_types,
        tilingkey_consts=kd._tilingkey_consts, datatype_consts=kd._datatype_consts,
    )
    return ast.unparse(tree)


def _guarded_bodies(src: str) -> list[list[str]]:
    """The statements under each stage's is_valid guard, unparsed."""
    tree = ast.parse(src)
    return [
        [ast.unparse(s) for s in node.body]
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and ast.unparse(node.test).endswith("._pl_is_valid")
    ]


# The event-id variables each buffer is declared with in _vcv_kernel.
_IDS = {"gm_a": ("fwd_a", "bwd_a"), "gm_b": ("fwd_b", "bwd_b")}

_PICK = re.compile(
    r"^(_pl_gm_(\w+)) = pl\.make_tensor\(pl\.addptr\(pl\.make_ptr\(\2\), (\w+)\._pl_task_id % (\d+) \* (\d+)\)"
)


def test_generated_slot_selection():
    """Each call gets its task's slot, picked after the waits with the event id's own index."""
    kernel, lay = _vcv_kernel(2, 2, 2, True, "gm_vcv_generated")
    bodies = _guarded_bodies(_generated(kernel))
    sizes = {"gm_a": lay.pitch_a * TM, "gm_b": lay.pitch_b * TM}
    picked = set()
    for body in bodies:
        call = next(i for i, s in enumerate(body) if s.startswith(("vcv_scale(", "vcv_matmul(", "vcv_post(")))
        picks = [(i, _PICK.match(s)) for i, s in enumerate(body[:call]) if s.startswith("_pl_gm_")]
        assert picks, f"no slot picked before `{body[call]}`"
        last_wait = max((i for i, s in enumerate(body[:call]) if "wait_cross_core" in s), default=-1)
        for i, m in picks:
            assert m, f"unexpected slot pick `{body[i]}`"
            var, buf, ctx, slots, size = m.groups()
            assert i > last_wait, f"`{var}` is picked before a wait"
            assert (int(slots), int(size)) == (2, sizes[buf])
            assert f"{var}," in body[call] or f"{var})" in body[call]
            ids = [s for s in body if any(f"event_id={name}[" in s for name in _IDS[buf])]
            assert ids and all(f"[{ctx}._pl_task_id % 2]" in s for s in ids)
            picked.add(buf)
    assert picked == {"gm_a", "gm_b"}


def test_generated_pipes():
    """Each GM access waits and sets on the pipe of the op touching the buffer."""
    kernel, _ = _vcv_kernel(1, 2, 1, True, "gm_vcv_pipes")
    src = _generated(kernel)
    sync = re.compile(r"pl\.system\.(\w+_cross_core)\(pipe=pl\.PipeType\.(\w+), event_id=(\w+)\[")
    buffer_of = {name: buf for buf, names in _IDS.items() for name in names}
    pipes = {(op, pipe, buffer_of[ids]) for op, pipe, ids in sync.findall(src)}
    assert ("set_cross_core", "MTE3", "gm_a") in pipes  # vector store
    assert ("wait_cross_core", "MTE2", "gm_a") in pipes  # cube load
    assert ("set_cross_core", "FIX", "gm_b") in pipes  # cube store from Acc
    assert ("wait_cross_core", "MTE2", "gm_b") in pipes  # vector load


def test_single_slot_passes_buffer_through():
    """One slot needs no selection: the stage gets the declared view itself."""
    kernel, _ = _vcv_kernel(0, 1, 1, True, "gm_vcv_single")
    src = _generated(kernel)
    assert "_pl_gm_" not in src
    assert "src, gm_a, ub_a)" in src


# ---------------------------------------------------------------------------
# A slot shape that is only known at run time
# ---------------------------------------------------------------------------

DYN_DEPTH = 2
DYN_PITCH = 2 * K  # pitched rows, so the slot size goes through max(shape[i] * stride[i])
DYN_SLOT = TM * DYN_PITCH * 2  # bytes per slot, fp16
DYN_PER_CORE = (DYN_DEPTH * DYN_SLOT + GUARD + 511) // 512 * 512


@pl.pipeline.stage
def dyn_matmul(k, gm_a, b, out, mat_a, mat_b, left, right, acc):
    ma = mat_a.next()
    mb = mat_b.next()
    lt = left.next()
    rt = right.next()
    c = acc.next()
    pl.load(ma, gm_a, [0, 0])
    pl.load(mb, b, [0, 0])
    pl.move(lt, ma)
    pl.move(rt, mb)
    pl.matmul(c, lt, rt)
    pl.store(out, c, [k * TM, 0])


def _dyn_kernel(name: str):
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=True, name=name, pipeline=pl.pipeline.PipelineConfig(preload=1))
    def dyn(
        src: pl.Tensor[[CORES * NT * TM, K], pl.DT_FP16],
        b: pl.Tensor[[K, N], pl.DT_FP16],
        ws: pl.Ptr[pl.DT_UINT8],
        out: pl.Tensor[[CORES * NT * TM, N], pl.DT_FP32],
    ):
        core_id = pl.get_block_idx() // pl.get_subblock_num()
        task_base = core_id * NT
        ws_core = pl.addptr(ws, core_id * DYN_PER_CORE)
        # rows comes off the input shape, so the slot size is an expression, not a constant.
        rows = src.shape[0] // (CORES * NT)
        gm_a = pl.make_tensor(ws_core, [rows, K], [DYN_PITCH, 1], dtype=pl.DT_FP16,
                              depth=DYN_DEPTH, fwd_ids=[0, 1], bwd_ids=[2, 3])
        with pl.section_cube():
            mat_a = pl.make_tile_group(
                type=pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
                addrs=0x00000, mutex_ids=[0, 1])
            mat_b = pl.make_tile_group(
                type=pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
                addrs=0x10000, mutex_ids=[2, 3])
            left = pl.make_tile_group(
                type=pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
                addrs=0x0000, mutex_ids=[4])
            right = pl.make_tile_group(
                type=pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
                addrs=0x0000, mutex_ids=[5])
            acc = pl.make_tile_group(
                type=pl.TileType(shape=[TM, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
                addrs=0x0000, mutex_ids=[6])
        with pl.section_vector():
            ub_a = pl.make_tile_group(
                type=pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
                addrs=0x00000, mutex_ids=[7, 8])

        for i in pl.range(0, NT):
            with pl.section_vector():
                vcv_scale(task_base + i, src, gm_a, ub_a)
            with pl.section_cube():
                dyn_matmul(task_base + i, gm_a, b, out, mat_a, mat_b, left, right, acc)

    return dyn


@pl.pipeline.stage
def dyn3_scale(k, src, gm_a, ub_a):
    t = ub_a.next()
    pl.load(t, src, [k * TM, 0])
    pl.muls(t, t, S0)
    pl.store(gm_a, t, [0, 0, 0])


@pl.pipeline.stage
def dyn3_matmul(k, gm_a, b, out, mat_a, mat_b, left, right, acc):
    ma = mat_a.next()
    mb = mat_b.next()
    lt = left.next()
    rt = right.next()
    c = acc.next()
    pl.load(ma, gm_a, [0, 0, 0])
    pl.load(mb, b, [0, 0])
    pl.move(lt, ma)
    pl.move(rt, mb)
    pl.matmul(c, lt, rt)
    pl.store(out, c, [k * TM, 0])


def _dyn3_kernel(name: str):
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=True, name=name, pipeline=pl.pipeline.PipelineConfig(preload=1))
    def dyn3(
        src: pl.Tensor[[CORES * NT * TM, K], pl.DT_FP16],
        b: pl.Tensor[[K, N], pl.DT_FP16],
        ws: pl.Ptr[pl.DT_UINT8],
        out: pl.Tensor[[CORES * NT * TM, N], pl.DT_FP32],
    ):
        core_id = pl.get_block_idx() // pl.get_subblock_num()
        task_base = core_id * NT
        ws_core = pl.addptr(ws, core_id * DYN_PER_CORE)
        # Three axes, so the slot size takes a max per axis.
        rows = src.shape[0] // (CORES * NT)
        gm_a = pl.make_tensor(ws_core, [1, rows, K], [rows * DYN_PITCH, DYN_PITCH, 1],
                              dtype=pl.DT_FP16, depth=DYN_DEPTH, fwd_ids=[0, 1], bwd_ids=[2, 3])
        with pl.section_cube():
            mat_a = pl.make_tile_group(
                type=pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
                addrs=0x00000, mutex_ids=[0, 1])
            mat_b = pl.make_tile_group(
                type=pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
                addrs=0x10000, mutex_ids=[2, 3])
            left = pl.make_tile_group(
                type=pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
                addrs=0x0000, mutex_ids=[4])
            right = pl.make_tile_group(
                type=pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
                addrs=0x0000, mutex_ids=[5])
            acc = pl.make_tile_group(
                type=pl.TileType(shape=[TM, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
                addrs=0x0000, mutex_ids=[6])
        with pl.section_vector():
            ub_a = pl.make_tile_group(
                type=pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
                addrs=0x00000, mutex_ids=[7, 8])

        for i in pl.range(0, NT):
            with pl.section_vector():
                dyn3_scale(task_base + i, src, gm_a, ub_a)
            with pl.section_cube():
                dyn3_matmul(task_base + i, gm_a, b, out, mat_a, mat_b, left, right, acc)

    return dyn3


def _check_dyn(name: str, build, expect_size: str):
    """Run a runtime-shaped kernel and check the slot size it picked with."""
    kernel = build(name)
    pick = next(s for s in _generated(kernel).splitlines() if "_pl_gm_gm_a = " in s)
    assert expect_size in pick, pick

    torch.manual_seed(13)
    src = torch.rand((CORES * NT * TM, K), device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    b = torch.rand((K, N), device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    ws = torch.full((CORES * DYN_PER_CORE,), SENTINEL, device=ST_DEVICE, dtype=torch.uint8)
    out = torch.zeros((CORES * NT * TM, N), device=ST_DEVICE, dtype=torch.float32)
    kernel[None, CORES](src, b, ws, out)
    torch.npu.synchronize()

    golden = (src.cpu().float() * S0).half().float() @ b.cpu().float()
    torch.testing.assert_close(out.cpu(), golden, rtol=1e-2, atol=ATOL)

    ws = ws.cpu()
    for core in range(CORES):
        base = core * DYN_PER_CORE
        region = ws[base:base + DYN_DEPTH * DYN_SLOT].view(DYN_DEPTH, TM, DYN_PITCH * 2)
        assert bool((region[:, :, K * 2:] == SENTINEL).all()), "a slot wrote into its row padding"
        tail = ws[base + DYN_DEPTH * DYN_SLOT:base + DYN_PER_CORE]
        assert bool((tail == SENTINEL).all()), "gm_a wrote past its last slot"
    logging.info("%s passed, max|diff| = %.6f", name, (out.cpu() - golden).abs().max().item())


@pytest.mark.soc("950")
def test_gm_dynamic_slot_shape():
    """A slot shape of runtime expressions: the size stays an expression and still addresses right."""
    torch.npu.set_device(ST_DEVICE)
    _check_dyn("gm_dynamic_slot_shape", _dyn_kernel, "% 2 * max(rows * DYN_PITCH, K * 1)")


@pytest.mark.soc("950")
def test_gm_dynamic_slot_shape_3d():
    """Three axes: the per-axis max nests, because the kernel-side max takes two arguments."""
    torch.npu.set_device(ST_DEVICE)
    _check_dyn(
        "gm_dynamic_slot_shape_3d",
        _dyn3_kernel,
        "% 2 * max(max(1 * (rows * DYN_PITCH), rows * DYN_PITCH), K * 1)",
    )


# ---------------------------------------------------------------------------
# Refused declarations and accesses
# ---------------------------------------------------------------------------


def _gm_decl_kernel(depth, name):
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=True, name=name, pipeline=pl.pipeline.PipelineConfig(preload=1))
    def k(
        src: pl.Tensor[[NT * TM, K], pl.DT_FP16],
        b: pl.Tensor[[K, N], pl.DT_FP16],
        ws: pl.Ptr[pl.DT_UINT8],
        out: pl.Tensor[[NT * TM, N], pl.DT_FP32],
    ):
        gm_a = pl.make_tensor(ws, [TM, K], dtype=pl.DT_FP16, depth=depth, fwd_ids=[0], bwd_ids=[2])
        gm_b = pl.make_tensor(pl.addptr(ws, 65536), [TM, N], dtype=pl.DT_FP32, depth=1, fwd_ids=[4], bwd_ids=[5])
        with pl.section_cube():
            mat_a = pl.make_tile_group(
                type=pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
                addrs=0x00000, mutex_ids=[0, 1])
            mat_b = pl.make_tile_group(
                type=pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
                addrs=0x10000, mutex_ids=[2, 3])
            left = pl.make_tile_group(
                type=pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
                addrs=0x0000, mutex_ids=[4])
            right = pl.make_tile_group(
                type=pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
                addrs=0x0000, mutex_ids=[5])
            acc = pl.make_tile_group(
                type=pl.TileType(shape=[TM, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
                addrs=0x0000, mutex_ids=[6])
        with pl.section_vector():
            ub_a = pl.make_tile_group(
                type=pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
                addrs=0x00000, mutex_ids=[7, 8])
            ub_b = pl.make_tile_group(
                type=pl.TileType(shape=[TM, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                addrs=0x08000, mutex_ids=[9, 10])
        for i in pl.range(0, NT):
            with pl.section_vector():
                vcv_scale(i, src, gm_a, ub_a)
            with pl.section_cube():
                vcv_matmul(gm_a, b, gm_b, mat_a, mat_b, left, right, acc)
            with pl.section_vector():
                vcv_post(i, gm_b, out, ub_b)

    return k


def _declared_in_helper(view):
    pl.make_tensor(view, [1, 1], depth=1, fwd_ids=[9], bwd_ids=[10])


@pl.pipeline.stage
def scale_with_helper_decl(k, src, gm_a, ub_a):
    t = ub_a.next()
    pl.load(t, src, [k * TM, 0])
    _declared_in_helper(gm_a)
    pl.store(gm_a, t, [0, 0])


@pl.pipeline.stage
def consume(gm_a, mat_a):
    ma = mat_a.next()
    pl.load(ma, gm_a, [0, 0])


@pl.jit(arch=NpuArch.DAV_3510, auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=1))
def helper_decl_kernel(src: pl.Tensor[[NT * TM, K], pl.DT_FP16], ws: pl.Ptr[pl.DT_FP16]):
    gm_a = pl.make_tensor(ws, [TM, K], depth=2, fwd_ids=[0, 1], bwd_ids=[2, 3])
    with pl.section_cube():
        mat_a = pl.make_tile_group(
            type=pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0x00000, mutex_ids=[0, 1])
    with pl.section_vector():
        ub_a = pl.make_tile_group(
            type=pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addrs=0x00000, mutex_ids=[2, 3])
    for i in pl.range(0, NT):
        with pl.section_vector():
            scale_with_helper_decl(i, src, gm_a, ub_a)
        with pl.section_cube():
            consume(gm_a, mat_a)


@pl.pipeline.stage
def scale(k, src, gm_a, ub_a):
    t = ub_a.next()
    pl.load(t, src, [k * TM, 0])
    pl.store(gm_a, t, [0, 0])


@pl.jit(arch=NpuArch.DAV_3510, auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=1))
def outside_stage_kernel(src: pl.Tensor[[NT * TM, K], pl.DT_FP16], ws: pl.Ptr[pl.DT_FP16]):
    gm_a = pl.make_tensor(ws, [TM, K], depth=2, fwd_ids=[0, 1], bwd_ids=[2, 3])
    with pl.section_cube():
        mat_a = pl.make_tile_group(
            type=pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0x00000, mutex_ids=[0, 1])
    with pl.section_vector():
        ub_a = pl.make_tile_group(
            type=pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addrs=0x00000, mutex_ids=[2, 3])
    for i in pl.range(0, NT):
        with pl.section_vector():
            scale(i, src, gm_a, ub_a)
        with pl.section_cube():
            consume(gm_a, mat_a)
        _ = gm_a[0, 0]


@pl.jit(arch=NpuArch.DAV_3510, auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=1))
def no_depth_kernel(src: pl.Tensor[[NT * TM, K], pl.DT_FP16], ws: pl.Ptr[pl.DT_FP16]):
    gm_a = pl.make_tensor(ws, [TM, K], fwd_ids=[0, 1], bwd_ids=[2, 3])
    with pl.section_cube():
        mat_a = pl.make_tile_group(
            type=pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0x00000, mutex_ids=[0, 1])
    with pl.section_vector():
        ub_a = pl.make_tile_group(
            type=pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addrs=0x00000, mutex_ids=[2, 3])
    for i in pl.range(0, NT):
        with pl.section_vector():
            scale(i, src, gm_a, ub_a)
        with pl.section_cube():
            consume(gm_a, mat_a)


@pl.jit(arch=NpuArch.DAV_3510, auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=1))
def unknown_shape_kernel(src: pl.Tensor[[NT * TM, K], pl.DT_FP16], ws: pl.Ptr[pl.DT_FP16]):
    gm_a = pl.make_tensor(ws, src.shape, depth=2, fwd_ids=[0, 1], bwd_ids=[2, 3])
    with pl.section_cube():
        mat_a = pl.make_tile_group(
            type=pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0x00000, mutex_ids=[0, 1])
    with pl.section_vector():
        ub_a = pl.make_tile_group(
            type=pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addrs=0x00000, mutex_ids=[2, 3])
    for i in pl.range(0, NT):
        with pl.section_vector():
            scale(i, src, gm_a, ub_a)
        with pl.section_cube():
            consume(gm_a, mat_a)


@pytest.mark.parametrize(
    "kernel, message",
    [
        (lambda:no_depth_kernel, "needs depth="),
        (lambda:_gm_decl_kernel(0, "gm_zero_depth"), "needs depth="),
        (lambda:_gm_decl_kernel(1, "gm_too_few_slots"), "slots are live at once"),
        (lambda:unknown_shape_kernel, "cannot determine the slot size"),
        (lambda:helper_decl_kernel, "declares a buffer with `pl.make_tensor()`"),
        (lambda:outside_stage_kernel, "is not inside a stage"),
    ],
    ids=["no_depth", "zero_depth", "too_few_slots", "unknown_shape", "declared_in_helper", "outside_stage"],
)
def test_refused(kernel, message):
    with pytest.raises(Exception, match=re.escape(message)):
        _generated(kernel())
