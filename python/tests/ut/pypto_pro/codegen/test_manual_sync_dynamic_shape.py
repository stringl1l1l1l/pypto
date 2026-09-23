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
"""Supplemental dynamic-shape CCE checks for manual in-core synchronization.

The case IDs in this file correspond to the review matrix in the repository
root ``example`` file.  Positive interface behaviour is covered by A5 ST; this
file only inspects lowering details that are not directly observable from an
ST result, such as exact instruction spelling, count, order, and control-flow
placement.  It is supplementary and must not be counted as positive interface
coverage.
"""

import re

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField
import pytest

EVENT_ID = 2
EVENT_IDS = (4, 5)
PING_PONG_EVENTS = (0, 1)
PING_PONG_MUTEXES = (6, 7)
USE_STATIC_EVENT = True


class _ManualSyncTilingKey:
    SyncEvent = TilingKeyField(bits=3, values=[2, 7])


def _compile_to_cce(kernel_def) -> str:
    from pypto_pro.runtime.jit import _assemble_cv_source, _parse_and_codegen_targets

    cube, vector = _parse_and_codegen_targets(kernel_def, "3510", "")
    return _assemble_cv_source(cube, vector).content


def _assert_in_order(source: str, *snippets: str) -> None:
    position = -1
    for snippet in snippets:
        position = source.find(snippet, position + 1)
        assert position >= 0, f"missing or reordered CCE snippet: {snippet}\n{source}"


@pl.jit
def _vector_pipeline_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tile_type = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_x = pl.make_tile(tile_type, addr=0x0000)
    tile_out = pl.make_tile(tile_type, addr=0x0100)
    with pl.section_vector():
        for row in pl.range(0, x.shape[0]):
            pl.load(tile_x, x, [row, 0])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.add(tile_out, tile_x, tile_x)
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.store(out, tile_out, [row, 0])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=2)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=2)
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=3)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=3)
            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=4)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=4)


@pl.jit
def _cube_pipeline_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
    left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left)
    right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right)
    acc_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc)
    a_l1 = pl.make_tile(mat_type, addr=0x0000)
    b_l1 = pl.make_tile(mat_type, addr=0x2000)
    a_l0 = pl.make_tile(left_type, addr=0x0000)
    b_l0 = pl.make_tile(right_type, addr=0x0000)
    acc = pl.make_tile(acc_type, addr=0x0000)
    with pl.section_cube():
        for row in pl.range(0, a.shape[0], 64):
            pl.load(a_l1, a, [row, 0])
            pl.load(b_l1, b, [0, 0])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
            pl.move(a_l0, a_l1)
            pl.move(b_l0, b_l1)
            pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=1)
            pl.matmul(acc, a_l0, b_l0)
            pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=2)
            pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=2)
            pl.store(out, acc, [row, 0])
            pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=3)
            pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=3)
            pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=4)
            pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=4)
            pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=5)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=5)
            pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.FIX, event_id=6)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.FIX, event_id=6)


@pl.jit
def _scalar_pipeline_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
):
    tile_type = pl.TileType(shape=[1, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    tile = pl.make_tile(tile_type, addr=0x0000)
    with pl.section_vector():
        pl.load(tile, x, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.S, event_id=6)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.S, event_id=6)
        value = tile[0, 0]
        tile[0, 0] = value + x.shape[0]
        pl.system.sync_src(set_pipe=pl.PipeType.S, wait_pipe=pl.PipeType.MTE3, event_id=7)
        pl.system.sync_dst(set_pipe=pl.PipeType.S, wait_pipe=pl.PipeType.MTE3, event_id=7)
        pl.store(out, tile, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=0)


@pl.jit
def _static_event_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    local_event = 2
    folded_event = (1 + 2) % 8
    with pl.section_vector():
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=7)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=7)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=local_event)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=EVENT_ID)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=folded_event)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=folded_event)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=EVENT_IDS[0])
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=EVENT_IDS[0])
        if USE_STATIC_EVENT:
            pl.system.sync_src(set_pipe=pl.PipeType.S, wait_pipe=pl.PipeType.MTE3, event_id=6)
            pl.system.sync_dst(set_pipe=pl.PipeType.S, wait_pipe=pl.PipeType.MTE3, event_id=6)
        if x.shape[0] > 0:
            pl.system.bar_all()


@pl.jit(auto_mutex=False, tiling_key=_ManualSyncTilingKey)
def _tilingkey_static_event_kernel(x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32]):
    with pl.section_vector():
        if SyncEvent == 2:  # noqa: F821
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=2)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=2)
        else:
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=7)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=7)


@pl.jit(auto_mutex=False, datatype={"x": "io_dtype"})
def _datatype_static_event_kernel(x: pl.Ptr[pl.DT_UINT8]):
    with pl.section_vector():
        if io_dtype == pl.DT_FP16:  # noqa: F821
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=1)
        else:
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=6)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=6)


@pl.jit
def _dynamic_event_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    event_id: pl.DT_INT64,
    slot: pl.DT_INT64,
    condition: pl.DT_BOOL,
):
    selected_event = event_id if condition else slot % 8
    branch_event = 0
    if condition:
        branch_event = event_id
    else:
        branch_event = slot % 8
    with pl.section_vector():
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=event_id)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=event_id)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=EVENT_IDS[slot % 2])
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=EVENT_IDS[slot % 2])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=selected_event)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=selected_event)
        pl.system.sync_src(set_pipe=pl.PipeType.S, wait_pipe=pl.PipeType.MTE3, event_id=branch_event)
        pl.system.sync_dst(set_pipe=pl.PipeType.S, wait_pipe=pl.PipeType.MTE3, event_id=branch_event)
        for row in pl.range(0, x.shape[0]):
            ping_pong = PING_PONG_EVENTS[row % 2]
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=ping_pong)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=ping_pong)


@pl.jit
def _control_flow_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    condition: pl.DT_BOOL,
):
    with pl.section_vector():
        if condition:
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=4)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=4)

        if condition:
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        else:
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)

        pl.system.sync_src(set_pipe=pl.PipeType.S, wait_pipe=pl.PipeType.MTE2, event_id=5)
        for row in pl.range(0, x.shape[0]):
            if row % 2 == 0:
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
        pl.system.sync_dst(set_pipe=pl.PipeType.S, wait_pipe=pl.PipeType.MTE2, event_id=5)

        index: pl.DT_INT64 = 0
        while index < x.shape[0]:
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.S, event_id=3)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.S, event_id=3)
            index = index + 1


@pl.jit
def _barrier_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    condition: pl.DT_BOOL,
):
    with pl.section_vector():
        for _ in pl.range(0, x.shape[0]):
            pl.system.bar_mte1()
            pl.system.bar_mte2()
            pl.system.bar_mte3()
        if condition:
            pl.system.bar_m()
            pl.system.bar_fix()
        pl.system.bar_all()
        pl.system.bar_all()
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.bar_all()
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)


@pl.jit
def _sync_all_hard_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    with pl.section_cube():
        pl.system.sync_all(core_type=pl.SyncCoreType.AIC_ONLY)
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
    with pl.section_vector():
        pl.system.sync_all(core_type=pl.SyncCoreType.AIV_ONLY)
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)


@pl.jit
def _mutex_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    slot: pl.DT_INT64,
    condition: pl.DT_BOOL,
):
    tile_type = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile = pl.make_tile(tile_type, addr=0x0000)
    local_mutex = 5
    selected_mutex = 8 if condition else 9
    with pl.section_vector():
        pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=0)
        pl.load(tile, x, [0, 0])
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=0)
        pl.system.mutex_lock(pipe=pl.PipeType.MTE3, mutex_id=local_mutex)
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE3, mutex_id=local_mutex)
        pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=31)
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=31)

        pl.system.mutex_lock(
            pipe=pl.PipeType.MTE3,
            mutex_id=PING_PONG_MUTEXES[slot % 2],
        )
        pl.store(out, tile, [0, 0])
        pl.system.mutex_unlock(
            pipe=pl.PipeType.MTE3,
            mutex_id=PING_PONG_MUTEXES[slot % 2],
        )

        pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=selected_mutex)
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=selected_mutex)
        pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=10)
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=10)
        pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=10)
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=10)
        if x.shape[0] > 0:
            pl.system.mutex_lock(pipe=pl.PipeType.M, mutex_id=11)
            pl.system.mutex_unlock(pipe=pl.PipeType.M, mutex_id=11)


@pl.jit(auto_mutex=True)
def _auto_mutex_combination_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tile_type = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0x0000, 0x0100], mutex_ids=[0, 1])
    outputs = pl.make_tile_group(type=tile_type, addrs=[0x0200, 0x0300], mutex_ids=[2, 3])
    with pl.section_vector():
        pl.system.set_cross_core(
            pipe=pl.PipeType.V, event_id=14, sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK
        )
        pl.system.wait_cross_core(
            pipe=pl.PipeType.V, event_id=14, sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK
        )
        for row in pl.range(0, x.shape[0]):
            tile_x = inputs.next()
            tile_out = outputs.next()
            pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=12)
            pl.load(tile_x, x, [row, 0])
            pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=12)
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=4)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=4)
            pl.add(tile_out, tile_x, tile_x)
            pl.system.bar_all()
            pl.store(out, tile_out, [row, 0])
        pl.system.sync_all(core_type=pl.SyncCoreType.AIV_ONLY)


def test_a01_a03_a04_a06_a07_a08_vector_and_scalar_pipeline_order():
    vector = _compile_to_cce(_vector_pipeline_kernel.to_kernel_def())
    scalar = _compile_to_cce(_scalar_pipeline_kernel.to_kernel_def())

    _assert_in_order(
        vector,
        "set_flag(PIPE_MTE2, PIPE_V, (event_t)0);",
        "wait_flag(PIPE_MTE2, PIPE_V, (event_t)0);",
        "set_flag(PIPE_V, PIPE_MTE3, (event_t)1);",
        "wait_flag(PIPE_V, PIPE_MTE3, (event_t)1);",
        "set_flag(PIPE_MTE3, PIPE_V, (event_t)2);",
        "wait_flag(PIPE_MTE3, PIPE_V, (event_t)2);",
        "set_flag(PIPE_V, PIPE_MTE2, (event_t)3);",
        "wait_flag(PIPE_V, PIPE_MTE2, (event_t)3);",
        "set_flag(PIPE_MTE3, PIPE_MTE2, (event_t)4);",
        "wait_flag(PIPE_MTE3, PIPE_MTE2, (event_t)4);",
    )
    _assert_in_order(
        scalar,
        "set_flag(PIPE_MTE2, PIPE_S, (event_t)6);",
        "wait_flag(PIPE_MTE2, PIPE_S, (event_t)6);",
        "set_flag(PIPE_S, PIPE_MTE3, (event_t)7);",
        "wait_flag(PIPE_S, PIPE_MTE3, (event_t)7);",
        "set_flag(PIPE_MTE3, PIPE_MTE2, (event_t)0);",
        "wait_flag(PIPE_MTE3, PIPE_MTE2, (event_t)0);",
    )


def test_a02_a05_a09_a10_a11_cube_forward_and_reverse_reuse_order():
    cpp = _compile_to_cce(_cube_pipeline_kernel.to_kernel_def())
    _assert_in_order(
        cpp,
        "set_flag(PIPE_MTE2, PIPE_MTE1, (event_t)0);",
        "wait_flag(PIPE_MTE2, PIPE_MTE1, (event_t)0);",
        "set_flag(PIPE_MTE1, PIPE_M, (event_t)1);",
        "wait_flag(PIPE_MTE1, PIPE_M, (event_t)1);",
        "set_flag(PIPE_M, PIPE_FIX, (event_t)2);",
        "wait_flag(PIPE_M, PIPE_FIX, (event_t)2);",
        "set_flag(PIPE_FIX, PIPE_M, (event_t)3);",
        "wait_flag(PIPE_FIX, PIPE_M, (event_t)3);",
        "set_flag(PIPE_M, PIPE_MTE1, (event_t)4);",
        "wait_flag(PIPE_M, PIPE_MTE1, (event_t)4);",
        "set_flag(PIPE_MTE1, PIPE_MTE2, (event_t)5);",
        "wait_flag(PIPE_MTE1, PIPE_MTE2, (event_t)5);",
        "set_flag(PIPE_MTE1, PIPE_FIX, (event_t)6);",
        "wait_flag(PIPE_MTE1, PIPE_FIX, (event_t)6);",
    )


@pytest.mark.parametrize(
    "event_token",
    ["(event_t)0", "(event_t)7", "(event_t)2", "(event_t)3", "(event_t)4", "(event_t)6"],
    ids=[
        "B01-lower",
        "B01-upper",
        "B02-B03-local-global",
        "B04-fold",
        "B05-static-tuple",
        "B07-static-condition",
    ],
)
def test_b01_b07_static_event_id_generalization(event_token):
    cpp = _compile_to_cce(_static_event_kernel.to_kernel_def())
    assert event_token in cpp
    if event_token == "(event_t)2":
        assert cpp.count("(event_t)2") >= 2


@pytest.mark.parametrize("event_id", [2, 7])
def test_b07_tilingkey_branch_folds_selected_event_id(event_id):
    kernel_def = _tilingkey_static_event_kernel.to_kernel_def({"SyncEvent": event_id})
    cpp = _compile_to_cce(kernel_def)

    other_event_id = 7 if event_id == 2 else 2
    assert f"(event_t){event_id}" in cpp
    assert f"(event_t){other_event_id}" not in cpp


@pytest.mark.parametrize(("dtype", "event_id"), [(pl.DT_FP16, 1), (pl.DT_BF16, 6)])
def test_b07_datatype_branch_folds_selected_event_id(dtype, event_id):
    kernel_def = _datatype_static_event_kernel.to_kernel_def(datatype_consts={"io_dtype": dtype})
    cpp = _compile_to_cce(kernel_def)

    other_event_id = 6 if event_id == 1 else 1
    assert f"(event_t){event_id}" in cpp
    assert f"(event_t){other_event_id}" not in cpp


def test_b06_b10_runtime_event_id_generalization():
    cpp = _compile_to_cce(_dynamic_event_kernel.to_kernel_def())
    sync_calls = re.findall(r"(?:set_flag|wait_flag)\([^;\n]+\);", cpp)
    assert sum(call.startswith("set_flag(") for call in sync_calls) >= 4
    assert sum(call.startswith("wait_flag(") for call in sync_calls) >= 4
    assert all("(event_t)" in call for call in sync_calls)
    assert "set_flag(PIPE_MTE2, PIPE_V, (event_t)" in cpp
    assert "wait_flag(PIPE_MTE2, PIPE_V, (event_t)" in cpp
    assert "set_flag(PIPE_V, PIPE_MTE3, (event_t)" in cpp
    assert "wait_flag(PIPE_V, PIPE_MTE3, (event_t)" in cpp


@pytest.mark.parametrize(
    ("control_token", "inner_sync"),
    [
        ("if (", "set_flag(PIPE_MTE2, PIPE_V, (event_t)4);"),
        ("for (", "set_flag(PIPE_MTE3, PIPE_MTE2, (event_t)2);"),
        ("while (", "set_flag(PIPE_MTE2, PIPE_S, (event_t)3);"),
    ],
    ids=["C01-C02-runtime-if-else", "C03-C05-nested-for-if", "C06-while"],
)
def test_c01_c06_sync_is_preserved_in_control_flow(control_token, inner_sync):
    cpp = _compile_to_cce(_control_flow_kernel.to_kernel_def())
    assert control_token in cpp
    assert cpp.count("set_flag(") == cpp.count("wait_flag(")
    _assert_in_order(cpp, control_token, inner_sync)


@pytest.mark.parametrize(
    ("case_id", "barrier", "minimum_count", "container"),
    [
        ("D02", "pipe_barrier(PIPE_M);", 1, "if ("),
        ("D03", "pipe_barrier(PIPE_MTE1);", 1, "for ("),
        ("D04", "pipe_barrier(PIPE_MTE2);", 1, "for ("),
        ("D05", "pipe_barrier(PIPE_MTE3);", 1, "for ("),
        ("D06", "pipe_barrier(PIPE_FIX);", 1, "if ("),
        ("D07-D11", "pipe_barrier(PIPE_ALL);", 3, None),
    ],
)
def test_d02_d11_barrier_codegen(case_id, barrier, minimum_count, container):
    del case_id
    cpp = _compile_to_cce(_barrier_kernel.to_kernel_def())
    assert cpp.count(barrier) >= minimum_count
    if container is not None:
        _assert_in_order(cpp, container, barrier)
    _assert_in_order(
        cpp,
        "set_flag(PIPE_MTE2, PIPE_V, (event_t)0);",
        "pipe_barrier(PIPE_ALL);",
        "wait_flag(PIPE_MTE2, PIPE_V, (event_t)0);",
    )


@pytest.mark.parametrize(
    "syncall",
    [
        "SYNCALL<SyncCoreType::AIVOnly>()",
        "SYNCALL<SyncCoreType::AICOnly>()",
        "SYNCALL<SyncCoreType::Mix>()",
    ],
    ids=["E01-hard-aiv", "E02-hard-aic", "E03-hard-mix"],
)
def test_e01_e03_hard_sync_all(syncall):
    assert syncall in _compile_to_cce(_sync_all_hard_kernel.to_kernel_def())


def test_f01_f09_static_dynamic_and_control_flow_mutex():
    cpp = _compile_to_cce(_mutex_kernel.to_kernel_def())
    assert "get_buf(PIPE_MTE2, 0, 0);" in cpp
    assert "rls_buf(PIPE_MTE2, 0, 0);" in cpp
    assert "get_buf(PIPE_MTE3" in cpp
    assert "rls_buf(PIPE_MTE3" in cpp
    assert "get_buf(PIPE_MTE3, 5, 0);" in cpp
    assert "rls_buf(PIPE_MTE3, 5, 0);" in cpp
    assert "get_buf(PIPE_MTE2, 31, 0);" in cpp
    assert "rls_buf(PIPE_MTE2, 31, 0);" in cpp
    assert "get_buf(PIPE_M, 11, 0);" in cpp
    assert "rls_buf(PIPE_M, 11, 0);" in cpp
    assert "if (" in cpp
    assert cpp.count("get_buf(PIPE_MTE2, 10, 0);") == 2
    assert cpp.count("rls_buf(PIPE_MTE2, 10, 0);") == 2
    assert cpp.count("get_buf(") == cpp.count("rls_buf(")


def test_h01_h05_auto_mutex_with_all_manual_sync_forms():
    cpp = _compile_to_cce(_auto_mutex_combination_kernel.to_kernel_def())
    assert "get_buf(" in cpp
    assert "rls_buf(" in cpp
    assert "set_flag(PIPE_MTE2, PIPE_V, (event_t)4);" in cpp
    assert "wait_flag(PIPE_MTE2, PIPE_V, (event_t)4);" in cpp
    assert "get_buf(PIPE_MTE2, 12, 0);" in cpp
    assert "rls_buf(PIPE_MTE2, 12, 0);" in cpp
    assert "SYNCALL<SyncCoreType::AIVOnly>()" in cpp
    assert "pipe_barrier(PIPE_ALL);" in cpp
    assert "getFFTSMsg(1, 14)" in cpp
    assert "wait_flag_dev(PIPE_V, 14);" in cpp
