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

"""What may be written inside a pipeline loop, and where the transform puts it.

A stage runs once per TASK; every other statement in the loop body runs once per BEAT,
fill and drain beats included. That splits the body in three:

    for ki in ...:
        <zone 1>                  statements, run per beat, read by the snapshot after them
        with pl.section_*():      the stage chain: stages and nothing else, and each
            stage(...)            section holds its one stage call and nothing else
        <zone 3>                  statements, run per beat, not in this beat's snapshot

The kernel runs on device so the per-iteration values a zone-1 statement supplies are
checked for real; placement and the refusals are asserted on the transformed AST, which
costs nothing and catches what numbers cannot.
"""

import ast
import os

from pypto_pro._errors import PyptoProError
import pypto_pro.language as pl
from pypto_pro.runtime.pipeline import transform_pipeline
from pypto_pro.runtime.pipeline._analyzer import probe_kernel_facts
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

TM, TN = 64, 64
NT = 4
FULL_M = TM * (NT + 2)
ATOL = 1e-2


def _generated(kernel) -> str:
    """The transformed kernel as source, the way the framework hands it to codegen."""
    kd = kernel.to_kernel_def()
    if_const_map, var_types = probe_kernel_facts(kd, None)
    tree = transform_pipeline(
        kd._func_def, kd._closure_vars, kd._pipeline,
        if_const_map=if_const_map, var_types=var_types,
        tilingkey_consts=kd._tilingkey_consts, datatype_consts=kd._datatype_consts,
    )
    return ast.unparse(tree)


# ---------------------------------------------------------------------------
# Stages and helpers shared by every kernel below.
# ---------------------------------------------------------------------------


@pl.pipeline.stage
def produce(k, scale, src, out, sh, ubg):
    """Vector side: computes, stores, and hands a tile to the cube core."""
    tile = ubg.next()
    pl.load(tile, src, [k * TM, 0])
    pl.muls(tile, tile, scale)
    pl.store(out, tile, [k * TM, 0])
    shared = sh.next()
    pl.move(shared, tile)


@pl.pipeline.stage
def produce_cfg(k, cfg, src, out, sh, ubg):
    """Takes the whole struct, so its fields travel in the ctx slot."""
    tile = ubg.next()
    pl.load(tile, src, [k * TM, 0])
    pl.muls(tile, tile, cfg.f)
    pl.store(out, tile, [k * TM, 0])
    shared = sh.next()
    pl.move(shared, tile)


@pl.pipeline.stage
def consume(k, sh, leftg):
    """Cube side: the other half of the handover."""
    shared = sh.next()
    left = leftg.next()
    pl.move(left, shared)


def take_a_slot(g):
    """Advances a group's cursor — legal only inside a stage, wherever it is written."""
    _tile = g.next()


def move_between(shared, left):
    """Touches the CROSS-CORE buffer through its parameters — nobody would synchronise it."""
    pl.move(left, shared)


def fill_in(box, v):
    """Fills in its own parameter — a value reaching the caller with no assignment.

    The shape the sparse kernel uses (``set_tile_info(run_info, ki)``): the struct the
    stages read is updated in place, so the call has to run before the snapshot even
    though the call site assigns nothing.
    """
    box.f = v


def hand_back_slot(g):
    """Returns a slot: the caller binds it to a name that resolves to no buffer."""
    return g.next()


# ---------------------------------------------------------------------------
# Every shape that is ALLOWED, in one kernel that runs.
# ---------------------------------------------------------------------------


@pl.jit(arch="3510", auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=1))
def loop_body_shapes(
    src: pl.Tensor[[FULL_M, TN], pl.DT_FP16], out: pl.Tensor[[FULL_M, TN], pl.DT_FP16]
):
    sh = pl.make_tile_group(
        type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=[0x0000, 0x8000], mutex_ids=[0, 1], fwd_ids=[0, 1], bwd_ids=[2, 3])
    with pl.section_cube():
        leftg = pl.make_tile_group(
            type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=[0x0000, 0x2000], mutex_ids=[4, 5])
    with pl.section_vector():
        ub = pl.make_tile_group(
            type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addrs=[0x0000, 0x2000], mutex_ids=[6, 7])
        ub2 = pl.make_tile_group(
            type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addrs=[0x4000, 0x6000], mutex_ids=[8, 9])
        spare = ub2.next()            # taken OUTSIDE the loop, so it is a stable tile
    tick = 0
    for ki in pl.range(0, NT):
        scale = tick + 2            # zone 1: supplies a stage argument
        with pl.section_vector():
            spare2 = ub2.next()       # a purely local group: its rotation is the user's own
            pl.muls(spare2, spare2, 1.0)
        with pl.section_vector():
            produce(ki, scale, src, out, sh, ub)
        with pl.section_cube():
            consume(ki, sh, leftg)
        with pl.section_vector():
            pl.muls(spare, spare, 2.0)    # zone 3: work on a purely local buffer
        tick = tick + 1                   # zone 3: prepares the NEXT iteration


@pytest.mark.soc("950")
def test_loop_body_shapes_on_device():
    """Iteration ki must see the scale zone 1 computed for it, not another beat's."""
    torch.npu.set_device(ST_DEVICE)
    torch.manual_seed(7)
    src = torch.rand((FULL_M, TN), device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    out = torch.zeros((FULL_M, TN), device=ST_DEVICE, dtype=torch.float16)
    loop_body_shapes(src, out)
    torch.npu.synchronize()

    golden = torch.zeros_like(src, dtype=torch.float32)
    for ki in range(NT):
        golden[ki * TM:(ki + 1) * TM] = src[ki * TM:(ki + 1) * TM].float() * (ki + 2.0)
    assert (out[:NT * TM].float().cpu() - golden[:NT * TM].cpu()).abs().max().item() <= ATOL


@pytest.mark.soc("950")
def test_allowed_shapes_are_placed_correctly():
    src = _generated(loop_body_shapes)
    body = src[src.index("for ki in pl.range"):].split("for _pl_drain")[0]
    lines = [line.strip() for line in body.splitlines()]

    def at(fragment):
        return next(i for i, line in enumerate(lines) if fragment in line)

    # A zone-1 statement stays where it was written; the snapshot after it reads what it
    # produced, so no value has to be moved to be seen.
    assert at("scale = tick + 2") < at("_pl_ctx_0.scale = scale") < at("produce(")

    # One fill per field, and the stage reads it from the slot rather than live.
    assert lines.count("_pl_ctx_0.ki = ki") == 1
    assert "produce(_pl_ctx_0.ki, _pl_ctx_0.scale, src, out, sh, ub)" in lines

    # Sync wraps the stage call alone.
    assert at("wait_cross_core") < at("produce(") < at("set_cross_core")

    # A zone-3 section is left where it was written, after the last stage.
    assert at("consume(") < at("pl.muls(spare, spare, 2.0)")

    # A purely local group may be rotated in the loop body: nothing indexes it by task.
    assert at("spare2 = ub2.next()") < at("produce(")

    # Written after the last stage, so it belongs to the next iteration: not snapshotted
    # into this beat's slot.
    assert at("consume(") < at("tick = tick + 1")


@pl.jit(arch="3510", auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=1))
def struct_filled_in_by_helper(
    src: pl.Tensor[[FULL_M, TN], pl.DT_FP16], out: pl.Tensor[[FULL_M, TN], pl.DT_FP16]
):
    sh = pl.make_tile_group(
        type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=[0x0000, 0x8000], mutex_ids=[0, 1], fwd_ids=[0, 1], bwd_ids=[2, 3])
    with pl.section_cube():
        leftg = pl.make_tile_group(
            type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=[0x0000, 0x2000], mutex_ids=[4, 5])
    with pl.section_vector():
        ub = pl.make_tile_group(
            type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addrs=[0x0000, 0x2000], mutex_ids=[6, 7])
    box = pl.struct("Box", f=0.0)
    for ki in pl.range(0, NT):
        fill_in(box, 2.0)
        with pl.section_vector():
            produce_cfg(ki, box, src, out, sh, ub)
        with pl.section_cube():
            consume(ki, sh, leftg)


@pytest.mark.soc("950")
def test_zone1_call_runs_before_the_snapshot():
    """A zone-1 call that assigns nothing but fills in a struct still supplies the snapshot."""
    src = _generated(struct_filled_in_by_helper)
    body = src[src.index("for ki in pl.range"):].split("for _pl_drain")[0]
    lines = [line.strip() for line in body.splitlines()]
    call = next(i for i, line in enumerate(lines) if "fill_in(box, 2.0)" in line)
    fill = next(i for i, line in enumerate(lines) if line.startswith("_pl_ctx_0.f ="))
    assert call < fill, "the call must run before the snapshot reads the struct"


# ---------------------------------------------------------------------------
# Shapes that are refused. Each is a whole kernel because the first error wins.
# ---------------------------------------------------------------------------


@pl.jit(arch="3510", auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=1))
def stmt_between_stages(
    src: pl.Tensor[[FULL_M, TN], pl.DT_FP16], out: pl.Tensor[[FULL_M, TN], pl.DT_FP16]
):
    sh = pl.make_tile_group(
        type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=[0x0000, 0x8000], mutex_ids=[0, 1], fwd_ids=[0, 1], bwd_ids=[2, 3])
    with pl.section_cube():
        leftg = pl.make_tile_group(
            type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=[0x0000, 0x2000], mutex_ids=[4, 5])
    with pl.section_vector():
        ub = pl.make_tile_group(
            type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addrs=[0x0000, 0x2000], mutex_ids=[6, 7])
    tick = 0
    for ki in pl.range(0, NT):
        with pl.section_vector():
            produce(ki, 2.0, src, out, sh, ub)
        tick = tick + 1                       # between two stages: neither beat nor task
        with pl.section_cube():
            consume(ki, sh, leftg)


@pl.jit(arch="3510", auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=1))
def section_without_stage_between_stages(
    src: pl.Tensor[[FULL_M, TN], pl.DT_FP16], out: pl.Tensor[[FULL_M, TN], pl.DT_FP16]
):
    sh = pl.make_tile_group(
        type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=[0x0000, 0x8000], mutex_ids=[0, 1], fwd_ids=[0, 1], bwd_ids=[2, 3])
    with pl.section_cube():
        leftg = pl.make_tile_group(
            type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=[0x0000, 0x2000], mutex_ids=[4, 5])
    with pl.section_vector():
        ub = pl.make_tile_group(
            type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addrs=[0x0000, 0x2000], mutex_ids=[6, 7])
        ub2 = pl.make_tile_group(
            type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addrs=[0x4000, 0x6000], mutex_ids=[8, 9])
        spare = ub2.next()
    for ki in pl.range(0, NT):
        with pl.section_vector():
            produce(ki, 2.0, src, out, sh, ub)
        with pl.section_vector():
            pl.muls(spare, spare, 2.0)        # a section is still a statement
        with pl.section_cube():
            consume(ki, sh, leftg)


@pl.jit(arch="3510", auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=1))
def stmt_beside_stage_in_section(
    src: pl.Tensor[[FULL_M, TN], pl.DT_FP16], out: pl.Tensor[[FULL_M, TN], pl.DT_FP16]
):
    sh = pl.make_tile_group(
        type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=[0x0000, 0x8000], mutex_ids=[0, 1], fwd_ids=[0, 1], bwd_ids=[2, 3])
    with pl.section_cube():
        leftg = pl.make_tile_group(
            type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=[0x0000, 0x2000], mutex_ids=[4, 5])
    with pl.section_vector():
        ub = pl.make_tile_group(
            type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addrs=[0x0000, 0x2000], mutex_ids=[6, 7])
    for ki in pl.range(0, NT):
        with pl.section_vector():
            scale = 2.0                       # the section holds its stage and nothing else
            produce(ki, scale, src, out, sh, ub)
        with pl.section_cube():
            consume(ki, sh, leftg)


@pl.jit(arch="3510", auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=1))
def slot_taken_in_loop_body(
    src: pl.Tensor[[FULL_M, TN], pl.DT_FP16], out: pl.Tensor[[FULL_M, TN], pl.DT_FP16]
):
    sh = pl.make_tile_group(
        type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=[0x0000, 0x8000], mutex_ids=[0, 1], fwd_ids=[0, 1], bwd_ids=[2, 3])
    with pl.section_cube():
        leftg = pl.make_tile_group(
            type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=[0x0000, 0x2000], mutex_ids=[4, 5])
    with pl.section_vector():
        ub = pl.make_tile_group(
            type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addrs=[0x0000, 0x2000], mutex_ids=[6, 7])
    for ki in pl.range(0, NT):
        _spare = sh.next()        # a tracked buffer: its slot belongs to a task
        with pl.section_vector():
            produce(ki, 2.0, src, out, sh, ub)
        with pl.section_cube():
            consume(ki, sh, leftg)


@pl.jit(arch="3510", auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=1))
def slot_taken_inside_a_helper(
    src: pl.Tensor[[FULL_M, TN], pl.DT_FP16], out: pl.Tensor[[FULL_M, TN], pl.DT_FP16]
):
    sh = pl.make_tile_group(
        type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=[0x0000, 0x8000], mutex_ids=[0, 1], fwd_ids=[0, 1], bwd_ids=[2, 3])
    with pl.section_cube():
        leftg = pl.make_tile_group(
            type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=[0x0000, 0x2000], mutex_ids=[4, 5])
    with pl.section_vector():
        ub = pl.make_tile_group(
            type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addrs=[0x0000, 0x2000], mutex_ids=[6, 7])
    for ki in pl.range(0, NT):
        with pl.section_cube():
            take_a_slot(sh)       # the call site shows no pick at all
        with pl.section_vector():
            produce(ki, 2.0, src, out, sh, ub)
        with pl.section_cube():
            consume(ki, sh, leftg)


@pl.jit(arch="3510", auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=1))
def cross_core_touched_outside_stage(
    src: pl.Tensor[[FULL_M, TN], pl.DT_FP16], out: pl.Tensor[[FULL_M, TN], pl.DT_FP16]
):
    sh = pl.make_tile_group(
        type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=[0x0000, 0x8000], mutex_ids=[0, 1], fwd_ids=[0, 1], bwd_ids=[2, 3])
    with pl.section_cube():
        leftg = pl.make_tile_group(
            type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=[0x0000, 0x2000], mutex_ids=[4, 5])
        cur_left = leftg.next()
        cur_sh = sh.next()
    with pl.section_vector():
        ub = pl.make_tile_group(
            type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addrs=[0x0000, 0x2000], mutex_ids=[6, 7])
    for ki in pl.range(0, NT):
        with pl.section_vector():
            produce(ki, 2.0, src, out, sh, ub)
        with pl.section_cube():
            consume(ki, sh, leftg)
        with pl.section_cube():
            move_between(cur_sh, cur_left)    # zone 3, but on a buffer sync tracks


@pl.pipeline.stage
def take_slot(k, sh, leftg):
    shared = hand_back_slot(sh)
    left = leftg.next()
    pl.move(left, shared)


@pl.jit(arch="3510", auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=1))
def helper_returns_a_slot(
    src: pl.Tensor[[FULL_M, TN], pl.DT_FP16], out: pl.Tensor[[FULL_M, TN], pl.DT_FP16]
):
    sh = pl.make_tile_group(
        type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=[0x0000, 0x8000], mutex_ids=[0, 1], fwd_ids=[0, 1], bwd_ids=[2, 3])
    with pl.section_cube():
        leftg = pl.make_tile_group(
            type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=[0x0000, 0x2000], mutex_ids=[4, 5])
    with pl.section_vector():
        ub = pl.make_tile_group(
            type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addrs=[0x0000, 0x2000], mutex_ids=[6, 7])
    for ki in pl.range(0, NT):
        with pl.section_vector():
            produce(ki, 2.0, src, out, sh, ub)
        with pl.section_cube():
            take_slot(ki, sh, leftg)


@pl.jit(arch="3510", auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=1))
def stage_call_without_a_section(
    src: pl.Tensor[[FULL_M, TN], pl.DT_FP16], out: pl.Tensor[[FULL_M, TN], pl.DT_FP16]
):
    sh = pl.make_tile_group(
        type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=[0x0000, 0x8000], mutex_ids=[0, 1], fwd_ids=[0, 1], bwd_ids=[2, 3])
    # Declared outside any section, so the bare call below resolves in both targets and the
    # missing section is what fails.
    ub = pl.make_tile_group(
        type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=[0x0000, 0x2000], mutex_ids=[6, 7])
    with pl.section_cube():
        leftg = pl.make_tile_group(
            type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=[0x0000, 0x2000], mutex_ids=[4, 5])
    for ki in pl.range(0, NT):
        produce(ki, 2.0, src, out, sh, ub)    # no section says which core runs it
        with pl.section_cube():
            consume(ki, sh, leftg)


@pl.jit(arch="3510", auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=1))
def one_stage_called_twice(
    src: pl.Tensor[[FULL_M, TN], pl.DT_FP16], out: pl.Tensor[[FULL_M, TN], pl.DT_FP16]
):
    sh = pl.make_tile_group(
        type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=[0x0000, 0x8000], mutex_ids=[0, 1], fwd_ids=[0, 1], bwd_ids=[2, 3])
    with pl.section_cube():
        leftg = pl.make_tile_group(
            type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=[0x0000, 0x2000], mutex_ids=[4, 5])
    with pl.section_vector():
        ub = pl.make_tile_group(
            type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addrs=[0x0000, 0x2000], mutex_ids=[6, 7])
    for ki in pl.range(0, NT):
        with pl.section_vector():
            produce(ki, 2.0, src, out, sh, ub)
        with pl.section_cube():
            consume(ki, sh, leftg)
        with pl.section_vector():
            produce(ki, 3.0, src, out, sh, ub)   # sync belongs to the NAME


@pytest.mark.soc("950")
@pytest.mark.parametrize(
    "kernel, message",
    [
        (stmt_between_stages, "sits between two stage calls"),
        (section_without_stage_between_stages, "sits between two stage calls"),
        (stmt_beside_stage_in_section, "may hold nothing else"),
        (slot_taken_in_loop_body, "takes a slot from tile group"),
        (slot_taken_inside_a_helper, "takes a slot from tile group"),
        (cross_core_touched_outside_stage, "cross-core sync tracks"),
        (helper_returns_a_slot, "a tile or tile group"),
        (stage_call_without_a_section, "not inside a `with pl.section_cube"),
        (one_stage_called_twice, "called more than once"),
    ],
)
def test_refused_shapes(kernel, message):
    # 8 of the parametrised shapes raise InvalidOperation and one raises
    # NotSupported, whose builtin bases differ, so the shared base is what fits.
    with pytest.raises(PyptoProError, match=message):
        _generated(kernel)
