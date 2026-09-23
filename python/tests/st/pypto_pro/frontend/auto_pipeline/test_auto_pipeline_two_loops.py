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

"""Two independent pipeline loops in one kernel, each with its own preload.

Every pipeline loop gets its own ctx ring, task counter, drain and sync, so this file
covers what a single-pipeline kernel cannot:

  * the second pipeline's generated names do not collide with the first's;
  * each loop syncs only the buffers IT uses, and declares only those event ids;
  * two buffers used in different loops may declare the SAME event ids — the loops run one
    after the other and each loop's sync balances within itself, so the ids are free again
    by the time the next loop starts (the ``a_*`` and ``b_*`` buffers below share them on
    purpose);
  * ``preload`` given per loop actually reaches each loop, including the mixed case where
    one stays serial while the other is pipelined.

Four stages per loop is what makes the preload visible: a core's FIRST stage always runs
one beat behind its upstream, and only the SECOND stage on the same core is placed
``preload`` beats after it. With two stages per loop (one per core) every preload would
produce the same schedule.

The cross-core handover runs vector -> cube: the vector stage writes a Mat buffer (a
``Vec->Mat`` move) and the cube stage reads it (``Mat->Left``). The cube stages only
consume — they are the other half of a pair, which every pipeline loop has to hold — while
the numbers this test checks are produced and stored by the vector side.

The whole-core barrier between the two loops is the user's own, as it always is: the
framework synchronises inside a pipeline loop, never between two of them.
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

TM, TN = 64, 64
NT = 4
LO, HI = 0, NT * TM  # the two row bands the low/high stage pair writes
FULL_M = 2 * NT * TM
ATOL = 1e-2


# The tile types are spelled out at every declaration rather than built by a helper: the
# declaration scan reads target_memory off the make_tile_group call itself and needs it as a
# literal there (see _cross_core_scanner._validate_buffer_memory).


# ---------------------------------------------------------------------------
# Stages. Each loop runs vector, cube, vector, cube — two per core, which is what
# makes preload change the schedule.
# ---------------------------------------------------------------------------


@pl.pipeline.stage
def a_prep_lo(k, src, out, sh, ubg):
    tile = ubg.next()
    pl.load(tile, src, [LO + k * TM, 0])
    pl.muls(tile, tile, 2.0)
    pl.store(out, tile, [LO + k * TM, 0])
    shared = sh.next()
    pl.move(shared, tile)


@pl.pipeline.stage
def a_use_lo(k, sh, leftg):
    shared = sh.next()
    left = leftg.next()
    pl.move(left, shared)


@pl.pipeline.stage
def a_prep_hi(k, src, out, sh, ubg):
    tile = ubg.next()
    pl.load(tile, src, [HI + k * TM, 0])
    pl.muls(tile, tile, 3.0)
    pl.store(out, tile, [HI + k * TM, 0])
    shared = sh.next()
    pl.move(shared, tile)


@pl.pipeline.stage
def a_use_hi(k, sh, leftg):
    shared = sh.next()
    left = leftg.next()
    pl.move(left, shared)


@pl.pipeline.stage
def b_prep_lo(k, src, out, sh, ubg):
    tile = ubg.next()
    pl.load(tile, src, [LO + k * TM, 0])
    pl.muls(tile, tile, 4.0)
    pl.store(out, tile, [LO + k * TM, 0])
    shared = sh.next()
    pl.move(shared, tile)


@pl.pipeline.stage
def b_use_lo(k, sh, leftg):
    shared = sh.next()
    left = leftg.next()
    pl.move(left, shared)


@pl.pipeline.stage
def b_prep_hi(k, src, out, sh, ubg):
    tile = ubg.next()
    pl.load(tile, src, [HI + k * TM, 0])
    pl.muls(tile, tile, 5.0)
    pl.store(out, tile, [HI + k * TM, 0])
    shared = sh.next()
    pl.move(shared, tile)


@pl.pipeline.stage
def b_use_hi(k, sh, leftg):
    shared = sh.next()
    left = leftg.next()
    pl.move(left, shared)


@pl.jit(arch="3510", auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=[1, 2]))
def four_stage_kernel(
    src: pl.Tensor[[FULL_M, TN], pl.DT_FP16],
    out: pl.Tensor[[FULL_M, TN], pl.DT_FP16],
):
    """Two pipelines of four stages, preloaded 1 and 2 — different depths, same kernel.

    The b_* buffers repeat the a_* event ids deliberately: they belong to the other loop.
    """
    a_lo = pl.make_tile_group(
        type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=[0x00000, 0x08000], mutex_ids=[0, 1], fwd_ids=[0, 1], bwd_ids=[2, 3],
    )
    a_hi = pl.make_tile_group(
        type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=[0x10000, 0x18000], mutex_ids=[2, 3], fwd_ids=[4, 5], bwd_ids=[6, 7],
    )
    b_lo = pl.make_tile_group(
        type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=[0x20000, 0x28000], mutex_ids=[4, 5], fwd_ids=[0, 1], bwd_ids=[2, 3],
    )
    b_hi = pl.make_tile_group(
        type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=[0x30000, 0x38000], mutex_ids=[6, 7], fwd_ids=[4, 5], bwd_ids=[6, 7],
    )
    with pl.section_cube():
        a_left = pl.make_tile_group(
            type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=[0x0000, 0x2000], mutex_ids=[8, 9],
        )
        b_left = pl.make_tile_group(
            type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=[0x4000, 0x6000], mutex_ids=[10, 11],
        )
    with pl.section_vector():
        a_ub = pl.make_tile_group(
            type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addrs=[0x0000, 0x2000], mutex_ids=[12, 13],
        )
        b_ub = pl.make_tile_group(
            type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addrs=[0x4000, 0x6000], mutex_ids=[14, 15],
        )

    for i in pl.range(0, NT):
        with pl.section_vector():
            a_prep_lo(i, src, out, a_lo, a_ub)
        with pl.section_cube():
            a_use_lo(i, a_lo, a_left)
        with pl.section_vector():
            a_prep_hi(i, src, out, a_hi, a_ub)
        with pl.section_cube():
            a_use_hi(i, a_hi, a_left)

    pl.system.sync_all(core_type=pl.SyncCoreType.MIX)

    for j in pl.range(0, NT):
        with pl.section_vector():
            b_prep_lo(j, src, out, b_lo, b_ub)
        with pl.section_cube():
            b_use_lo(j, b_lo, b_left)
        with pl.section_vector():
            b_prep_hi(j, src, out, b_hi, b_ub)
        with pl.section_cube():
            b_use_hi(j, b_hi, b_left)


@pl.jit(arch="3510", auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=[0, 2]))
def mixed_preload_kernel(
    src: pl.Tensor[[FULL_M, TN], pl.DT_FP16],
    out: pl.Tensor[[FULL_M, TN], pl.DT_FP16],
):
    """preload=0 on the first loop, 2 on the second.

    The first keeps its serial loop and gets only cross-core sync inserted; the second is
    pipelined. The two emission paths have to coexist in one kernel, each with its own
    counter (`_pl_sync_id` and `_pl_task_id_1`). Two stages per loop is enough here — what
    is under test is the two paths meeting, not the schedule.
    """
    a_lo = pl.make_tile_group(
        type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=[0x00000, 0x08000], mutex_ids=[0, 1], fwd_ids=[0, 1], bwd_ids=[2, 3],
    )
    b_lo = pl.make_tile_group(
        type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=[0x20000, 0x28000], mutex_ids=[4, 5], fwd_ids=[0, 1], bwd_ids=[2, 3],
    )
    with pl.section_cube():
        a_left = pl.make_tile_group(
            type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=[0x0000, 0x2000], mutex_ids=[8, 9],
        )
        b_left = pl.make_tile_group(
            type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=[0x4000, 0x6000], mutex_ids=[10, 11],
        )
    with pl.section_vector():
        a_ub = pl.make_tile_group(
            type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addrs=[0x0000, 0x2000], mutex_ids=[12, 13],
        )
        b_ub = pl.make_tile_group(
            type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addrs=[0x4000, 0x6000], mutex_ids=[14, 15],
        )

    for i in pl.range(0, NT):
        with pl.section_vector():
            a_prep_lo(i, src, out, a_lo, a_ub)
        with pl.section_cube():
            a_use_lo(i, a_lo, a_left)

    pl.system.sync_all(core_type=pl.SyncCoreType.MIX)

    for j in pl.range(0, NT):
        with pl.section_vector():
            b_prep_lo(j, src, out, b_lo, b_ub)
        with pl.section_cube():
            b_use_lo(j, b_lo, b_left)


def _golden(src, hi_stages: bool):
    """Loop B runs after loop A over the same rows, so its factor is the one that survives.

    That overlap is the point: if the two pipelines were interleaved, or the second loop's
    drain ran before the first's tail, these rows would not come out at 4x and 5x.
    """
    src = src.cpu().float()
    golden = torch.zeros_like(src)
    golden[LO:LO + NT * TM] = 4.0 * src[LO:LO + NT * TM]
    if hi_stages:
        golden[HI:HI + NT * TM] = 5.0 * src[HI:HI + NT * TM]
    return golden


@pytest.mark.soc("950")
@pytest.mark.parametrize(
    "kernel, hi_stages, label",
    [
        (four_stage_kernel, True, "4 stages/loop, preload=[1, 2]"),
        (mixed_preload_kernel, False, "2 stages/loop, preload=[0, 2]"),
    ],
)
def test_two_pipeline_loops(kernel, hi_stages, label):
    torch.npu.set_device(ST_DEVICE)
    torch.manual_seed(7)
    src = torch.rand((FULL_M, TN), device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    out = torch.zeros((FULL_M, TN), device=ST_DEVICE, dtype=torch.float16)
    golden = _golden(src, hi_stages)

    kernel(src, out)
    torch.npu.synchronize()

    diff = (out.cpu().float() - golden).abs().max().item()
    assert diff <= ATOL, f"{label}: max|diff| = {diff}"
    logging.info("two pipeline loops [%s] passed, max|diff| = %.6f", label, diff)
