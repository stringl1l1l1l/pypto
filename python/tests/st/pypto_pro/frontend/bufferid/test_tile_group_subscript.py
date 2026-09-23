# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TileGroup 下标访问前端测试。

测试覆盖场景:
  1. 覆盖边界内常量下标、动态下标、显式取模和单槽 TileGroup。
  2. 覆盖 mutex_ids 缺省、空列表、重复 ID 以及重叠候选 ID。
  3. 覆盖多槽预取、同一算子的双槽输入、循环携带 Tile 和 cursor 混合访问。
  4. 覆盖 Vector、Cube、Cube/Vector 混合和 VF 场景，并通过 torch golden 验证运行结果。
  5. 覆盖 pipeline.stage 中 TileGroup 的边界内常量和动态下标访问。
"""

import logging
import os

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

TILE_M = 64
TILE_N = 128
NTILE = 8
FULL_M = TILE_M * NTILE

PIPELINE_NTILE = 4
PIPELINE_FULL_M = TILE_M * PIPELINE_NTILE

REPEATED_TILE_M = 32
REPEATED_DEPTH = 16
REPEATED_NTILE = REPEATED_DEPTH * 2
REPEATED_FULL_M = REPEATED_TILE_M * REPEATED_NTILE

MM_M, MM_K, MM_N = 256, 128, 256
MM_TILE = 128

CV_M = 64
CV_K = 64
CV_N = 64
CV_VEC_ROWS = 32

VF_TILE_ROWS = 128
VF_TILE_COLS = 128
VF_VALID_COLS = 64


def _vec_tile_type():
    return pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


# =============================================================================
# Test 1: TileGroup 常量下标访问
#         TileGroup constant subscript
# =============================================================================
@pl.jit(arch="3510", auto_mutex=True)
def const_index_kernel(
    a: pl.Tensor[[FULL_M, TILE_N], pl.DT_FP16],
    out: pl.Tensor[[FULL_M, TILE_N], pl.DT_FP16],
):
    """Alternate two slots by literal index — every lock is a static mutex id."""
    g = pl.make_tile_group(type=_vec_tile_type(), addrs=0x0000, mutex_ids=[20, 21])
    with pl.section_vector():
        for k in pl.range(0, NTILE, 2):
            even = g[0]
            pl.load(even, a, [k * TILE_M, 0])
            pl.exp(even, even)
            pl.store(out, even, [k * TILE_M, 0])

            odd = g[1]
            pl.load(odd, a, [(k + 1) * TILE_M, 0])
            pl.add(odd, odd, odd)
            pl.store(out, odd, [(k + 1) * TILE_M, 0])


# =============================================================================
# Test 2: TileGroup 动态下标访问
#         TileGroup dynamic subscript
# =============================================================================
@pl.jit(arch="3510", auto_mutex=True)
def dynamic_index_kernel(
    a: pl.Tensor[[FULL_M, TILE_N], pl.DT_FP16],
    out: pl.Tensor[[FULL_M, TILE_N], pl.DT_FP16],
):
    """Rotate with an explicitly wrapped runtime index and lock dynamically."""
    g = pl.make_tile_group(type=_vec_tile_type(), addrs=0x0000, mutex_ids=[20, 21])
    with pl.section_vector():
        for k in pl.range(0, NTILE):
            t = g[k % 2]
            pl.load(t, a, [k * TILE_M, 0])
            if k < 4:
                pl.exp(t, t)
            else:
                pl.add(t, t, t)
            pl.store(out, t, [k * TILE_M, 0])


# =============================================================================
# Test 3: mutex_ids=None 时通过 depth 创建 TileGroup
#         TileGroup depth with mutex_ids=None
# =============================================================================
@pl.jit(arch="3510", auto_mutex=True)
def no_mutex_none_kernel(
    a: pl.Tensor[[FULL_M, TILE_N], pl.DT_FP16],
    out: pl.Tensor[[FULL_M, TILE_N], pl.DT_FP16],
):
    """mutex_ids=None uses depth and leaves synchronization to the kernel."""
    g = pl.make_tile_group(type=_vec_tile_type(), addrs=0x0000, mutex_ids=None, depth=2)
    with pl.section_vector():
        for k in pl.range(0, NTILE):
            pl.system.bar_all()
            t = g[k % 2]
            pl.load(t, a, [k * TILE_M, 0])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.exp(t, t)
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.store(out, t, [k * TILE_M, 0])


# =============================================================================
# Test 4: mutex_ids=[] 时通过 depth 创建 TileGroup
#         TileGroup depth with empty mutex_ids
# =============================================================================
@pl.jit(arch="3510", auto_mutex=True)
def no_mutex_empty_kernel(
    a: pl.Tensor[[FULL_M, TILE_N], pl.DT_FP16],
    out: pl.Tensor[[FULL_M, TILE_N], pl.DT_FP16],
):
    """An empty mutex_ids list has the same no-auto-mutex behavior as None."""
    g = pl.make_tile_group(type=_vec_tile_type(), addrs=0x0000, mutex_ids=[], depth=2)
    with pl.section_vector():
        for k in pl.range(0, NTILE):
            pl.system.bar_all()
            t = g.next()
            pl.load(t, a, [k * TILE_M, 0])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.exp(t, t)
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.store(out, t, [k * TILE_M, 0])


# =============================================================================
# Test 5: pipeline.stage 中混合使用常量和动态下标
#         Constant and dynamic subscripts in pipeline.stage
# =============================================================================
@pl.pipeline.stage
def pipeline_mixed_subscript_stage(k, a, out, g):
    tile = g[0]
    if k == 1:
        tile = g[1]
    elif k >= 2:
        tile = g[k]

    pl.load(tile, a, [k * TILE_M, 0])
    pl.add(tile, tile, tile)
    pl.store(out, tile, [k * TILE_M, 0])


@pl.jit(arch="3510", auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=1))
def pipeline_stage_subscript_kernel(
    a: pl.Tensor[[PIPELINE_FULL_M, TILE_N], pl.DT_FP16],
    out: pl.Tensor[[PIPELINE_FULL_M, TILE_N], pl.DT_FP16],
):
    g = pl.make_tile_group(type=_vec_tile_type(), addrs=0x0000, mutex_ids=[20, 21, 22, 23])
    for k in pl.range(0, PIPELINE_NTILE):
        with pl.section_vector():
            pipeline_mixed_subscript_stage(k, a, out, g)


# =============================================================================
# Test 6: 重复 mutex ID 的 16 槽轮转访问
#         Round-robin access with repeated mutex IDs
# =============================================================================
@pl.jit(arch="3510", auto_mutex=True)
def repeated_mutex_ids_kernel(
    a: pl.Tensor[[REPEATED_FULL_M, TILE_N], pl.DT_FP16],
    out: pl.Tensor[[REPEATED_FULL_M, TILE_N], pl.DT_FP16],
):
    """Round-robin over 16 slots whose latter eight mutex IDs repeat 0..7."""
    tile_type = pl.TileType(
        shape=[REPEATED_TILE_M, TILE_N],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    g = pl.make_tile_group(
        type=tile_type,
        addrs=0x0000,
        mutex_ids=[0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7],
        depth=REPEATED_DEPTH,
    )
    with pl.section_vector():
        for k in pl.range(0, REPEATED_NTILE):
            t = g[k % REPEATED_DEPTH]
            offset = k * REPEATED_TILE_M
            pl.load(t, a, [offset, 0])
            pl.exp(t, t)
            pl.store(out, t, [offset, 0])


# =============================================================================
# Test 7: 动态下标访问结果与 next() 一致
#         Dynamic subscript matches next()
# =============================================================================
@pl.jit(arch="3510", auto_mutex=True)
def next_reference_kernel(
    a: pl.Tensor[[FULL_M, TILE_N], pl.DT_FP16],
    out: pl.Tensor[[FULL_M, TILE_N], pl.DT_FP16],
):
    """The next()-based spelling of dynamic_index_kernel, used as a cross-check."""
    g = pl.make_tile_group(type=_vec_tile_type(), addrs=0x0000, mutex_ids=[20, 21])
    with pl.section_vector():
        for k in pl.range(0, NTILE):
            t = g.next()
            pl.load(t, a, [k * TILE_M, 0])
            if k < 4:
                pl.exp(t, t)
            else:
                pl.add(t, t, t)
            pl.store(out, t, [k * TILE_M, 0])


# =============================================================================
# Test 8: 两个 Tile 槽并行存活的预取场景
#         Prefetch with two live Tile slots
# =============================================================================
@pl.jit(arch="3510", auto_mutex=True)
def prefetch_kernel(
    a: pl.Tensor[[FULL_M, TILE_N], pl.DT_FP16],
    out: pl.Tensor[[FULL_M, TILE_N], pl.DT_FP16],
):
    """Two slots live at once: load slot k+1 while computing on slot k.

    This is the schedule subscript exists for — the cursor accessors cannot name
    two different future slots in one iteration. Correctness depends entirely on
    auto_mutex locking the two slots on their own ids: if both resolved to the
    same buffer token the MTE2 load would overwrite the tile the V pipe is still
    reading, and the tail tiles would come back wrong.
    """
    g = pl.make_tile_group(type=_vec_tile_type(), addrs=0x0000, mutex_ids=[20, 21])
    with pl.section_vector():
        pl.load(g[0], a, [0, 0])
        for k in pl.range(0, NTILE - 1):
            pl.load(g[(k + 1) % 2], a, [(k + 1) * TILE_M, 0])
            cur = g[k % 2]
            pl.exp(cur, cur)
            pl.store(out, cur, [k * TILE_M, 0])
        last = g[(NTILE - 1) % 2]
        pl.exp(last, last)
        pl.store(out, last, [(NTILE - 1) * TILE_M, 0])


# =============================================================================
# Test 9: 同一算子访问两个 Tile 槽
#          Two Tile slots used by one operation
# =============================================================================
@pl.jit(arch="3510", auto_mutex=True)
def two_slots_one_op_kernel(
    a: pl.Tensor[[FULL_M, TILE_N], pl.DT_FP16],
    out: pl.Tensor[[FULL_M, TILE_N], pl.DT_FP16],
):
    """Both operands of one op come from the same group.

    The two runtime ids go into a single mutex_lock_dyn whose codegen guards the
    second acquire with `!=`. Without that dedup, an iteration where both indices
    land on the same slot would issue two get_buf for one id on one pipe and hang
    the device — so this test passing at all is the assertion that matters.
    """
    g = pl.make_tile_group(type=_vec_tile_type(), addrs=0x0000, mutex_ids=[20, 21])
    with pl.section_vector():
        for k in pl.range(0, NTILE):
            src = g[k % 2]
            pl.load(src, a, [k * TILE_M, 0])
            dst = g[(k + 1) % 2]
            pl.move(dst, src)
            pl.exp(dst, dst)
            pl.store(out, dst, [k * TILE_M, 0])


# =============================================================================
# Test 10: 循环携带 g[0] 和 g[3] 后执行 add
#          Loop-carried g[0] and g[3] add
# =============================================================================
@pl.jit(arch="3510", auto_mutex=True)
def four_slots_loop_carried_add_kernel(
    a: pl.Tensor[[TILE_M, TILE_N], pl.DT_FP16],
    b: pl.Tensor[[TILE_M, TILE_N], pl.DT_FP16],
    out: pl.Tensor[[TILE_M, TILE_N], pl.DT_FP16],
):
    """Carry g[0] and g[3] selections out of a loop and add them."""
    g = pl.make_tile_group(type=_vec_tile_type(), addrs=0x0000, mutex_ids=[20, 21, 22, 23])
    with pl.section_vector():
        # Deliberately alias both initial values. If loop-carried Tile assignment
        # is broken, both remain g[1] and the result becomes b + b.
        first = g[1]
        last = g[1]
        for index in pl.range(0, 4):
            if index == 0:
                first = g[index]
            if index == 3:
                last = g[index]

        pl.load(first, a, [0, 0])
        pl.load(last, b, [0, 0])
        pl.add(first, first, last)
        pl.store(out, first, [0, 0])


# =============================================================================
# Test 11: if/else 合并重叠 mutex 候选 ID
#          Merge overlapping mutex candidates across if/else
# =============================================================================
@pl.jit(arch="3510", auto_mutex=True)
def ifelse_overlapping_mutex_ids_kernel(
    a: pl.Tensor[[FULL_M, TILE_N], pl.DT_FP16],
    out: pl.Tensor[[FULL_M, TILE_N], pl.DT_FP16],
):
    """Merge two overlapping mutex candidate sets across an if/else.

    After SSA merging, src can use mutex {0, 1}, while dst can use {1, 2}.
    Even iterations select mutex 1 for both operands and require dynamic dedup;
    odd iterations select the non-overlapping runtime pair 0/2.
    """
    src0 = pl.make_tile_group(type=_vec_tile_type(), addrs=0x0000, mutex_ids=[0])
    src1 = pl.make_tile_group(type=_vec_tile_type(), addrs=0x4000, mutex_ids=[1])
    dst2 = pl.make_tile_group(type=_vec_tile_type(), addrs=0x8000, mutex_ids=[2])
    dst1 = pl.make_tile_group(type=_vec_tile_type(), addrs=0xC000, mutex_ids=[1])
    with pl.section_vector():
        for k in pl.range(0, NTILE):
            src = src0[0]
            dst = dst2[0]
            if k % 2 == 0:
                src = src1[0]
                dst = dst1[0]
            else:
                src = src0[0]
                dst = dst2[0]
            pl.load(src, a, [k * TILE_M, 0])
            pl.move(dst, src)
            pl.exp(dst, dst)
            pl.store(out, dst, [k * TILE_M, 0])


# =============================================================================
# Test 12: 单槽 TileGroup 的动态下标访问
#          Dynamic subscript on a single-slot TileGroup
# =============================================================================
@pl.jit(arch="3510", auto_mutex=True)
def single_slot_kernel(
    a: pl.Tensor[[FULL_M, TILE_N], pl.DT_FP16],
    out: pl.Tensor[[FULL_M, TILE_N], pl.DT_FP16],
):
    """A one-slot group requires its runtime index to be reduced to zero."""
    g = pl.make_tile_group(type=_vec_tile_type(), addrs=0x0000, mutex_ids=[22])
    with pl.section_vector():
        for k in pl.range(0, NTILE):
            t = g[k % 1]
            pl.load(t, a, [k * TILE_M, 0])
            pl.exp(t, t)
            pl.store(out, t, [k * TILE_M, 0])


# =============================================================================
# Test 13: TileGroup 边界内常量下标访问
#          TileGroup bounded constant subscripts
# =============================================================================
@pl.jit(arch="3510", auto_mutex=True)
def bounded_constant_index_kernel(
    a: pl.Tensor[[FULL_M, TILE_N], pl.DT_FP16],
    out: pl.Tensor[[FULL_M, TILE_N], pl.DT_FP16],
):
    """Use g[1] and g[3] from four explicitly separated TileGroup slots."""
    g = pl.make_tile_group(
        type=_vec_tile_type(),
        addrs=[0x0000, 0x8000, 0x10000, 0x18000],
        mutex_ids=[20, 21, 22, 23],
    )
    with pl.section_vector():
        exp_tile = g[1]
        add_tile = g[3]
        pl.load(exp_tile, a, [0, 0])
        pl.load(add_tile, a, [TILE_M, 0])
        pl.exp(exp_tile, exp_tile)
        pl.add(add_tile, add_tile, add_tile)
        pl.store(out, exp_tile, [0, 0])
        pl.store(out, add_tile, [TILE_M, 0])


# =============================================================================
# Test 14: 下标 Tile 的 getval/setval 标量访问
#          Scalar getval/setval on a subscript-selected Tile
# =============================================================================
@pl.jit(arch="3510", auto_mutex=True)
def getval_setval_kernel(
    a: pl.Tensor[[FULL_M, TILE_N], pl.DT_FP16],
    out: pl.Tensor[[FULL_M, TILE_N], pl.DT_FP16],
):
    """Scalar access through a subscript-selected tile needs the same V->S order.

    getval/setval run on PIPE_S; the element read must observe the exp() the V
    pipe just wrote to the same slot.
    """
    g = pl.make_tile_group(type=_vec_tile_type(), addrs=0x0000, mutex_ids=[20, 21])
    with pl.section_vector():
        for k in pl.range(0, NTILE):
            t = g[k % 2]
            pl.load(t, a, [k * TILE_M, 0])
            pl.exp(t, t)
            v = t[0, 0]
            t[1, 0] = v
            pl.store(out, t, [k * TILE_M, 0])


# =============================================================================
# Test 15: TileGroup 下标与 next() 混合访问
#          Mixed TileGroup subscript and next() access
# =============================================================================
@pl.jit(arch="3510", auto_mutex=True)
def mixed_cursor_and_subscript_kernel(
    a: pl.Tensor[[FULL_M, TILE_N], pl.DT_FP16],
    out: pl.Tensor[[FULL_M, TILE_N], pl.DT_FP16],
):
    """Subscript leaves the cursor alone, so an interleaved next() keeps rotating.

    Slot k%2 is written through the subscript and slot (k+1)%2 through next();
    if the subscript had moved the cursor the two would collide and one of the
    two halves of the output would be stale.
    """
    g = pl.make_tile_group(type=_vec_tile_type(), addrs=0x0000, mutex_ids=[20, 21])
    with pl.section_vector():
        for k in pl.range(0, NTILE):
            direct = g[k % 2]
            pl.load(direct, a, [k * TILE_M, 0])
            pl.exp(direct, direct)
            pl.store(out, direct, [k * TILE_M, 0])

            rotated = g.next()
            pl.load(rotated, a, [k * TILE_M, 0])
            pl.exp(rotated, rotated)
            pl.exp(rotated, rotated)


# =============================================================================
# Test 16: Cube 双缓冲矩阵乘
#          Cube double-buffered matmul
# =============================================================================
@pl.jit(arch="3510", auto_mutex=True)
def matmul_subscript_kernel(
    a: pl.Tensor[[MM_M, MM_K], pl.DT_FP16],
    b: pl.Tensor[[MM_K, MM_N], pl.DT_FP16],
    c: pl.Tensor[[MM_M, MM_N], pl.DT_FP32],
):
    """The documented double-buffered matmul with L1 slots addressed by subscript."""
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[MM_TILE, MM_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0, 1],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[MM_K, MM_TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000,
        mutex_ids=[2, 3],
    )
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[MM_TILE, MM_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[4],
    )
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[MM_K, MM_TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[5],
    )
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[MM_TILE, MM_TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[6],
    )

    with pl.section_cube():
        step = 0
        for i in pl.range(0, MM_M, MM_TILE):
            for j in pl.range(0, MM_N, MM_TILE):
                cur_a = a_l1[step % 2]
                cur_b = b_l1[step % 2]
                al = a_left[0]
                br = b_right[0]
                ac = acc[0]
                pl.load(cur_a, a, [i, 0])
                pl.load(cur_b, b, [0, j])
                pl.move(al, cur_a)
                pl.move(br, cur_b)
                pl.matmul(ac, al, br)
                pl.store(c, ac, [i, j])
                step = step + 1


@pl.vector_function
def vf_add_64_columns(tile, value):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for row in pl.range(0, VF_TILE_ROWS):
        offset = row * VF_TILE_COLS
        vreg = vf.load_align(tile, offset)
        result = vf.adds(vreg, value, preg)
        vf.store_align(tile + offset, result, preg)


# =============================================================================
# Test 17: 同地址 TileGroup 子视图 VF 计算
#          VF computation on aliased TileGroup subviews
# =============================================================================
@pl.jit(auto_mutex=True)
def tile_group_alias_subview_vf_kernel(
    x: pl.Tensor[[VF_TILE_ROWS, VF_TILE_COLS], pl.DT_FP32],
    out: pl.Tensor[[VF_TILE_ROWS, VF_TILE_COLS], pl.DT_FP32],
):
    tt = pl.TileType(shape=[VF_TILE_ROWS, VF_TILE_COLS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    group_a = pl.make_tile_group(type=tt, addrs=[0x0000, 0x0000], mutex_ids=[0, 1])

    with pl.section_vector():
        valid_row = 128
        valid_col = 64

        tile_ping = group_a[0]
        tile_ping_first_part = tile_ping[:, 0:VF_VALID_COLS]
        pl.set_validshape(tile_ping, [valid_row, valid_col])
        pl.load(tile_ping_first_part, x, [0, 0])
        vf_add_64_columns(tile_ping_first_part, 1.0)
        pl.store(out, tile_ping_first_part, [0, 0])

        tile_pong = group_a[1]
        tile_pong_second_part = tile_pong[:, VF_VALID_COLS:]
        pl.set_validshape(tile_pong, [valid_row, valid_col])
        pl.load(tile_pong_second_part, x, [0, VF_VALID_COLS])
        vf_add_64_columns(tile_pong_second_part, 3.0)
        pl.store(out, tile_pong_second_part, [0, VF_VALID_COLS])


# =============================================================================
# Test 18: 偏移地址 TileGroup VF 计算
#          VF computation on shifted TileGroup addresses
# =============================================================================
@pl.jit(auto_mutex=True)
def tile_group_shifted_addr_vf_kernel(
    x: pl.Tensor[[VF_TILE_ROWS, VF_TILE_COLS], pl.DT_FP32],
    out: pl.Tensor[[VF_TILE_ROWS, VF_TILE_COLS], pl.DT_FP32],
):
    tt = pl.TileType(shape=[VF_TILE_ROWS, VF_TILE_COLS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    group_a = pl.make_tile_group(type=tt, addrs=[0x0000, 256], mutex_ids=[0, 1])

    with pl.section_vector():
        valid_row = 128
        valid_col = 64

        tile_ping = group_a[0]
        pl.set_validshape(tile_ping, [valid_row, valid_col])
        pl.load(tile_ping, x, [0, 0])
        vf_add_64_columns(tile_ping, 1.0)
        pl.store(out, tile_ping, [0, 0])

        tile_pong = group_a[1]
        pl.set_validshape(tile_pong, [valid_row, valid_col])
        pl.load(tile_pong, x, [0, VF_VALID_COLS])
        vf_add_64_columns(tile_pong, 3.0)
        pl.store(out, tile_pong, [0, VF_VALID_COLS])


# =============================================================================
# Test 19: Cube/Vector 混合计算 a + b @ c
#          Cube/Vector mixed computation for a + b @ c
# =============================================================================
@pl.jit(arch="3510", auto_mutex=True)
def cube_vector_four_mutex_kernel(
    a: pl.Tensor[[CV_M, CV_N], pl.DT_FP32],
    b: pl.Tensor[[CV_M, CV_K], pl.DT_FP16],
    c: pl.Tensor[[CV_K, CV_N], pl.DT_FP16],
    out: pl.Tensor[[CV_M, CV_N], pl.DT_FP32],
):
    """Compute a + b @ c with Cube and Vector while using mutex IDs 0..3."""
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CV_M, CV_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x0000,
        mutex_ids=[0],
    )
    c_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CV_K, CV_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x2000,
        mutex_ids=[1],
    )
    b_left = pl.make_tile_group(
        type=pl.TileType(shape=[CV_M, CV_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    c_right = pl.make_tile_group(
        type=pl.TileType(shape=[CV_K, CV_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    tile_acc = pl.make_tile(
        pl.TileType(shape=[CV_M, CV_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addr=0x0000,
    )
    vec_group = pl.make_tile_group(
        type=pl.TileType(shape=[CV_VEC_ROWS, CV_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids=[0, 1, 2, 3],
    )
    tile_product = vec_group[1]

    with pl.section_cube():
        tile_b_l1 = b_l1[0]
        tile_c_l1 = c_l1[0]
        tile_b_left = b_left[0]
        tile_c_right = c_right[0]
        pl.load(tile_b_l1, b, [0, 0])
        pl.load(tile_c_l1, c, [0, 0])
        pl.move(tile_b_left, tile_b_l1)
        pl.move(tile_c_right, tile_c_l1)
        pl.matmul(tile_acc, tile_b_left, tile_c_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.move(tile_product, tile_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    with pl.section_vector():
        sub_index = pl.get_subblock_idx()
        row_offset = sub_index * CV_VEC_ROWS
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=0)

        tile_a = vec_group[0]
        tile_sum = vec_group[2]
        tile_out = vec_group[3]

        pl.load(tile_a, a, [row_offset, 0])
        pl.add(tile_sum, tile_a, tile_product)
        pl.move(tile_out, tile_sum)
        pl.store(out, tile_out, [row_offset, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_subscript_const_index():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[FULL_M, TILE_N]]
    for shape in shapes:
        a = torch.rand(shape, device=device, dtype=torch.float16) * 2.0 - 1.0
        out = torch.zeros(shape, device=device, dtype=torch.float16)
        const_index_kernel(a, out)
        torch.npu.synchronize()
        golden = a.cpu().float()
        for k in range(0, NTILE, 2):
            even_rows = slice(k * TILE_M, (k + 1) * TILE_M)
            odd_rows = slice((k + 1) * TILE_M, (k + 2) * TILE_M)
            golden[even_rows] = torch.exp(golden[even_rows])
            golden[odd_rows] = golden[odd_rows] + golden[odd_rows]
        torch.testing.assert_close(out.cpu().float(), golden, rtol=3e-3, atol=3e-3)
        logging.info("test_subscript_const_index passed! shape=%s", shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_subscript_dynamic_index():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[FULL_M, TILE_N]]
    for shape in shapes:
        a = torch.rand(shape, device=device, dtype=torch.float16) * 2.0 - 1.0
        out = torch.zeros(shape, device=device, dtype=torch.float16)
        dynamic_index_kernel(a, out)
        torch.npu.synchronize()
        golden = a.cpu().float()
        split_row = 4 * TILE_M
        golden[:split_row] = torch.exp(golden[:split_row])
        golden[split_row:] = golden[split_row:] + golden[split_row:]
        torch.testing.assert_close(out.cpu().float(), golden, rtol=3e-3, atol=3e-3)
        logging.info("test_subscript_dynamic_index passed! shape=%s", shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_subscript_depth_without_mutex_ids():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[FULL_M, TILE_N]]
    for shape in shapes:
        a = torch.rand(shape, device=device, dtype=torch.float16) * 2.0 - 1.0
        out = torch.zeros(shape, device=device, dtype=torch.float16)
        no_mutex_none_kernel(a, out)
        torch.npu.synchronize()
        golden = torch.exp(a.cpu().float())
        torch.testing.assert_close(out.cpu().float(), golden, rtol=3e-3, atol=3e-3)
        logging.info("test_subscript_depth_without_mutex_ids passed! shape=%s", shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_next_depth_with_empty_mutex_ids():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[FULL_M, TILE_N]]
    for shape in shapes:
        a = torch.rand(shape, device=device, dtype=torch.float16) * 2.0 - 1.0
        out = torch.zeros(shape, device=device, dtype=torch.float16)
        no_mutex_empty_kernel(a, out)
        torch.npu.synchronize()
        golden = torch.exp(a.cpu().float())
        torch.testing.assert_close(out.cpu().float(), golden, rtol=3e-3, atol=3e-3)
        logging.info("test_next_depth_with_empty_mutex_ids passed! shape=%s", shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_pipeline_stage_mixed_subscripts():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[PIPELINE_FULL_M, TILE_N]]
    for shape in shapes:
        a = torch.rand(shape, device=device, dtype=torch.float16) * 2.0 - 1.0
        out = torch.zeros(shape, device=device, dtype=torch.float16)
        pipeline_stage_subscript_kernel(a, out)
        torch.npu.synchronize()
        golden = a.cpu().float() * 2.0
        torch.testing.assert_close(out.cpu().float(), golden, rtol=3e-3, atol=3e-3)
        logging.info("test_pipeline_stage_mixed_subscripts passed! shape=%s", shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_subscript_repeated_mutex_ids_round_robin():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[REPEATED_FULL_M, TILE_N]]
    for shape in shapes:
        a = torch.rand(shape, device=device, dtype=torch.float16) * 2.0 - 1.0
        out = torch.zeros(shape, device=device, dtype=torch.float16)
        repeated_mutex_ids_kernel(a, out)
        torch.npu.synchronize()
        golden = torch.exp(a.cpu().float())
        torch.testing.assert_close(out.cpu().float(), golden, rtol=3e-3, atol=3e-3)
        logging.info("test_subscript_repeated_mutex_ids_round_robin passed! shape=%s", shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_subscript_matches_next():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[FULL_M, TILE_N]]
    for shape in shapes:
        a = torch.rand(shape, device=device, dtype=torch.float16) * 2.0 - 1.0
        sub_out = torch.zeros(shape, device=device, dtype=torch.float16)
        next_out = torch.zeros(shape, device=device, dtype=torch.float16)
        dynamic_index_kernel(a, sub_out)
        next_reference_kernel(a, next_out)
        torch.npu.synchronize()
        golden = a.cpu().float()
        split_row = 4 * TILE_M
        golden[:split_row] = torch.exp(golden[:split_row])
        golden[split_row:] = golden[split_row:] + golden[split_row:]
        torch.testing.assert_close(sub_out, next_out, rtol=0, atol=0)
        torch.testing.assert_close(sub_out.cpu().float(), golden, rtol=3e-3, atol=3e-3)
        logging.info("test_subscript_matches_next passed! shape=%s", shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_subscript_prefetch_two_slots_live():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[FULL_M, TILE_N]]
    for shape in shapes:
        a = torch.rand(shape, device=device, dtype=torch.float16) * 2.0 - 1.0
        out = torch.zeros(shape, device=device, dtype=torch.float16)
        prefetch_kernel(a, out)
        torch.npu.synchronize()
        golden = torch.exp(a.cpu().float())
        torch.testing.assert_close(out.cpu().float(), golden, rtol=3e-3, atol=3e-3)
        logging.info("test_subscript_prefetch_two_slots_live passed! shape=%s", shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_subscript_two_slots_in_one_op():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[FULL_M, TILE_N]]
    for shape in shapes:
        a = torch.rand(shape, device=device, dtype=torch.float16) * 2.0 - 1.0
        out = torch.zeros(shape, device=device, dtype=torch.float16)
        two_slots_one_op_kernel(a, out)
        torch.npu.synchronize()
        golden = torch.exp(a.cpu().float())
        torch.testing.assert_close(out.cpu().float(), golden, rtol=3e-3, atol=3e-3)
        logging.info("test_subscript_two_slots_in_one_op passed! shape=%s", shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_subscript_four_slots_loop_carried_add():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[TILE_M, TILE_N]]
    for shape in shapes:
        a = torch.rand(shape, device=device, dtype=torch.float16) * 2.0 - 1.0
        b = torch.rand(shape, device=device, dtype=torch.float16) * 2.0 - 1.0
        out = torch.zeros(shape, device=device, dtype=torch.float16)
        four_slots_loop_carried_add_kernel(a, b, out)
        torch.npu.synchronize()
        golden = a.cpu().float() + b.cpu().float()
        torch.testing.assert_close(out.cpu().float(), golden, rtol=3e-3, atol=3e-3)
        logging.info("test_subscript_four_slots_loop_carried_add passed! shape=%s", shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_subscript_ifelse_overlapping_mutex_ids():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[FULL_M, TILE_N]]
    for shape in shapes:
        a = torch.rand(shape, device=device, dtype=torch.float16) * 2.0 - 1.0
        out = torch.zeros(shape, device=device, dtype=torch.float16)
        ifelse_overlapping_mutex_ids_kernel(a, out)
        torch.npu.synchronize()
        golden = torch.exp(a.cpu().float())
        torch.testing.assert_close(out.cpu().float(), golden, rtol=3e-3, atol=3e-3)
        logging.info("test_subscript_ifelse_overlapping_mutex_ids passed! shape=%s", shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_subscript_single_slot_group():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[FULL_M, TILE_N]]
    for shape in shapes:
        a = torch.rand(shape, device=device, dtype=torch.float16) * 2.0 - 1.0
        out = torch.zeros(shape, device=device, dtype=torch.float16)
        single_slot_kernel(a, out)
        torch.npu.synchronize()
        golden = torch.exp(a.cpu().float())
        torch.testing.assert_close(out.cpu().float(), golden, rtol=3e-3, atol=3e-3)
        logging.info("test_subscript_single_slot_group passed! shape=%s", shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_subscript_bounded_constant_indices():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[FULL_M, TILE_N]]
    for shape in shapes:
        a = torch.rand(shape, device=device, dtype=torch.float16) * 2.0 - 1.0
        out = torch.zeros(shape, device=device, dtype=torch.float16)
        bounded_constant_index_kernel(a, out)
        torch.npu.synchronize()
        got = out.cpu().float()
        exp_golden = torch.exp(a[:TILE_M].cpu().float())
        add_golden = a[TILE_M:2 * TILE_M].cpu().float() * 2.0
        torch.testing.assert_close(got[:TILE_M], exp_golden, rtol=3e-3, atol=3e-3)
        torch.testing.assert_close(got[TILE_M:2 * TILE_M], add_golden, rtol=3e-3, atol=3e-3)
        logging.info("test_subscript_bounded_constant_indices passed! shape=%s", shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_subscript_getval_setval():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[FULL_M, TILE_N]]
    for shape in shapes:
        a = torch.rand(shape, device=device, dtype=torch.float16) * 2.0 - 1.0
        out = torch.zeros(shape, device=device, dtype=torch.float16)
        getval_setval_kernel(a, out)
        torch.npu.synchronize()
        golden = torch.exp(a.cpu().float())
        for k in range(NTILE):
            golden[k * TILE_M + 1, 0] = golden[k * TILE_M, 0]
        torch.testing.assert_close(out.cpu().float(), golden, rtol=3e-3, atol=3e-3)
        logging.info("test_subscript_getval_setval passed! shape=%s", shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_subscript_mixed_with_next():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[FULL_M, TILE_N]]
    for shape in shapes:
        a = torch.rand(shape, device=device, dtype=torch.float16) * 2.0 - 1.0
        out = torch.zeros(shape, device=device, dtype=torch.float16)
        mixed_cursor_and_subscript_kernel(a, out)
        torch.npu.synchronize()
        golden = torch.exp(a.cpu().float())
        torch.testing.assert_close(out.cpu().float(), golden, rtol=3e-3, atol=3e-3)
        logging.info("test_subscript_mixed_with_next passed! shape=%s", shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_subscript_matmul_double_buffer():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[MM_M, MM_K, MM_N]]
    for m, k, n in shapes:
        a = torch.rand(m, k, device=device, dtype=torch.float16) * 0.1
        b = torch.rand(k, n, device=device, dtype=torch.float16) * 0.1
        c = torch.zeros(m, n, device=device, dtype=torch.float32)
        matmul_subscript_kernel(a, b, c)
        torch.npu.synchronize()
        golden = a.cpu().float() @ b.cpu().float()
        torch.testing.assert_close(c.cpu(), golden, rtol=5e-3, atol=5e-3)
        logging.info("test_subscript_matmul_double_buffer passed! shape=%s", [m, k, n])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_tile_group_alias_subview_vf():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(3)
    shapes = [[VF_TILE_ROWS, VF_TILE_COLS]]
    for shape in shapes:
        x = torch.rand(shape, device=device, dtype=torch.float32)
        out = torch.empty(shape, device=device, dtype=torch.float32)
        tile_group_alias_subview_vf_kernel(x, out)
        torch.npu.synchronize()
        golden = x.clone()
        golden[:, :VF_VALID_COLS] += 1.0
        golden[:, VF_VALID_COLS:] += 3.0
        torch.testing.assert_close(out, golden, rtol=0, atol=0)
        logging.info("test_tile_group_alias_subview_vf passed! shape=%s", shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_tile_group_shifted_addr_vf():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(4)
    shapes = [[VF_TILE_ROWS, VF_TILE_COLS]]
    for shape in shapes:
        x = torch.rand(shape, device=device, dtype=torch.float32)
        out = torch.empty(shape, device=device, dtype=torch.float32)
        tile_group_shifted_addr_vf_kernel(x, out)
        torch.npu.synchronize()
        golden = x.clone()
        golden[:, :VF_VALID_COLS] += 1.0
        golden[:, VF_VALID_COLS:] += 3.0
        torch.testing.assert_close(out, golden, rtol=0, atol=0)
        logging.info("test_tile_group_shifted_addr_vf passed! shape=%s", shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_cube_vector_four_mutex():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    shapes = [[CV_M, CV_K, CV_N]]
    for m, k, n in shapes:
        a = torch.rand([m, n], device=device, dtype=torch.float32) * 2.0 - 1.0
        b = torch.rand([m, k], device=device, dtype=torch.float16) * 0.1
        c = torch.rand([k, n], device=device, dtype=torch.float16) * 0.1
        out = torch.zeros([m, n], device=device, dtype=torch.float32)
        cube_vector_four_mutex_kernel(a, b, c, out)
        torch.npu.synchronize()
        golden = a.cpu() + b.cpu().float() @ c.cpu().float()
        torch.testing.assert_close(out.cpu(), golden, rtol=5e-3, atol=5e-3)
        logging.info("test_cube_vector_four_mutex passed! shape=%s", [m, k, n])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_subscript_const_index()
    test_subscript_dynamic_index()
    test_subscript_depth_without_mutex_ids()
    test_next_depth_with_empty_mutex_ids()
    test_pipeline_stage_mixed_subscripts()
    test_subscript_repeated_mutex_ids_round_robin()
    test_subscript_matches_next()
    test_subscript_prefetch_two_slots_live()
    test_subscript_two_slots_in_one_op()
    test_subscript_four_slots_loop_carried_add()
    test_subscript_ifelse_overlapping_mutex_ids()
    test_subscript_single_slot_group()
    test_subscript_bounded_constant_indices()
    test_subscript_getval_setval()
    test_subscript_mixed_with_next()
    test_subscript_matmul_double_buffer()
    test_tile_group_alias_subview_vf()
    test_tile_group_shifted_addr_vf()
    test_cube_vector_four_mutex()
    logging.info("\nAll tests passed!")
