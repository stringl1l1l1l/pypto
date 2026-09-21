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

"""FlashAttention performance kernel -- DN mode, VF softmax + flash_update, TILE-GROUP form.

This is the make_tile_group / auto_mutex port of test_fa_perf_tkv_preload_dn_vf_tail.py.
It keeps the *dynamic-shape + tail* semantics of the manual (_tail) kernel but adopts the
buffer-rotation and synchronization model of test_fa_perf_tkv_preload_dn_vf_bufid.py:

  * All multi-buffers are pl.make_tile_group(type=, addrs=, mutex_ids=) and accessed via
    .next() (rotating cursor) instead of manual ping-pong indices (l0ab_idx / l0c_idx /
    task_id % N tuple indexing).
  * @pl.jit(auto_mutex=True) removes all manual intra-core pl.system.sync_src/sync_dst;
    only cross-core set_cross_core / wait_cross_core remain (cube <-> vector handshake).
  * Dynamic tail handling is preserved: per-tile pl.set_validshape on each .next() slot,
    tail-aware VF masks (update_mask(actual_sq_half)), pad=min on qk_vec, compact=1 on the
    L0/L1 cube operands + qk_vec, compact=2 (RowPlusOne) on the p_f16 insert halves, and the
    real_rows / row_off store clip so junk tail rows are never written to global o.

Dynamic cube operands (left/right/acc/qk_vec) carry compact=1 and p_f16 halves carry
compact=2 on purpose: with a runtime valid window smaller than the declared tile these
fractal-repacking copies must pack to the valid dims. (Vec-only tiles processed by plain
vector ops -- see the layernorm/softmax tile-group tests -- do not need compact.)
"""

import logging
import math
import os

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
import pytest
import torch

import pypto

# ================================================================
#  Configuration -- change QK_PRELOAD to tune pre-compute depth
# ================================================================
QK_PRELOAD = 2
FIFO_SIZE = QK_PRELOAD + 1

# ================================================================
#  Tile dimensions and constants
# ================================================================
TS = 128
TKV = 128
TD = 128
TS_HALF = TS // 2
SCALE = 1.0 / math.sqrt(TD)
FLOAT_REP_SIZE = 64
D_LOOPS = TD // FLOAT_REP_SIZE
REDUCE_SIZE = 1
minValue = -1e9  # noqa: N816
block_stride_dn = TKV >> 1 | 0x1
REPEAT_STRIDE_DN = 1

# Cube tile byte sizes
Q_F16 = TS * TD * 2
KT_F16 = TKV * TD * 2
V_F16 = TKV * TD * 2
P_F16 = TS * TKV * 2
QK_HALF_F32 = TKV * TS * 4
PV_HALF_F32 = TS * TD * 4

# ---- MAT (512KB) : tile_group base addresses (slots derived from type size) ----
MA0 = 0  # q_l1_db   (2 slots)
MA1 = Q_F16 * 2  # k_l1_db   (2 slots)
MA2 = MA1 + KT_F16 * 2  # p_mat_db  (3 slots)
MA3 = MA2 + P_F16 * 3  # v_l1_db   (2 slots)
LA0 = 0  # left_db   (2 slots)
RA0 = 0  # right_db  (2 slots)
CA0 = 0  # acc_db    (4 slots)

# ---- VEC addresses (248KB on a5) ----
VB4_KV = TS_HALF * TKV * 4
VB2_KV = TS_HALF * 2 * (TKV // 2 + 1) * 2
VB2_KV_HALF = VB2_KV // 2
VB4 = TS_HALF * TD * 4
VB2 = TS_HALF * TD * 2
VB_RED = TS_HALF * 1 * 4

VA0 = 0  # qk_vec_db (2 slots)
VA2 = VA0 + 2 * VB4_KV  # p_f16 region (main halves)
VA2_DN = VA2 + VB2_KV_HALF  # p_f16 back half (slot 0)
VA2B = VA2 + VB2_KV  # p_f16 (slot 1)
VA2B_DN = VA2B + VB2_KV_HALF  # p_f16 back half (slot 1)
VA8 = VA2 + 2 * VB2_KV  # pv_vec_db (2 slots)
VA7 = VA8 + 2 * VB4  # running_o (1 slot)
VA9 = VA7 + VB4  # o_f16     (1 slot)
VA_GMAX0 = VA9 + VB2
VA_GMAX1 = VA_GMAX0 + VB_RED
VA_GMAX2 = VA_GMAX1 + VB_RED
VA_GSUM0 = VA_GMAX2 + VB_RED
VA_GSUM1 = VA_GSUM0 + VB_RED
VA_GSUM2 = VA_GSUM1 + VB_RED
VA_EC0 = VA_GSUM2 + VB_RED
VA_EC1 = VA_EC0 + VB_RED
VA_EC2 = VA_EC1 + VB_RED
VA_END = VA_EC2 + VB_RED
assert VA_END <= 248 * 1024, f"VEC overflow: {VA_END} > {248 * 1024}"

# Cross-core event IDs
QK_READY_FORWARD_IDS = (0, 1)
QK_READY_BARKWARD_IDS = (2, 3)
P_READY_FORWARD_IDS = (4, 5, 6)
PV_READY_FORWARD_IDS = (7, 8)
PV_READY_BARKWARD_IDS = (9, 10)

PV_CORE_STRIDE = 2 * FIFO_SIZE * TS


# ================================================================
#  VF softmax functions (tail-aware: take actual_sq_half / actual_skv)
# ================================================================
@pl.vector_function
def process_vec1_dn_no_update_vf(input_tile, x_exp_tile, max_tile, sum_tile, actual_sq_half, actual_skv):
    """Softmax DN VF kernel -- full version with Cast + DeInterleave + Store."""
    preg_main = vf.update_mask(actual_sq_half, dtype=pl.DT_FP32)
    preg_store = vf.update_mask(TS_HALF * 2, dtype=pl.DT_FP16)

    src_ub0 = input_tile
    src_ub1 = input_tile + TS_HALF
    src_ub2 = input_tile + TS_HALF * 2
    src_ub3 = input_tile + TS_HALF * 3
    x_exp_1 = x_exp_tile + TKV * 4

    # Phase 0: Fill invalid rows with minValue
    vreg_fill = vf.full(minValue)
    for r in pl.range(actual_skv, TKV):
        vf.store_align(input_tile + (r * TS_HALF), vreg_fill, preg_main)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)

    # Phase 1: ReduceMax (4-way parallel, only valid rows)
    max0 = vf.full(minValue)
    max1 = vf.full(minValue)
    max2 = vf.full(minValue)
    max3 = vf.full(minValue)
    for iter_m in pl.range(0, TKV // 4):
        src0 = vf.load_align(src_ub0, iter_m * TS_HALF * 4)
        src1 = vf.load_align(src_ub1, iter_m * TS_HALF * 4)
        src2 = vf.load_align(src_ub2, iter_m * TS_HALF * 4)
        src3 = vf.load_align(src_ub3, iter_m * TS_HALF * 4)
        max0 = vf.max(max0, src0, preg_main)
        max1 = vf.max(max1, src1, preg_main)
        max2 = vf.max(max2, src2, preg_main)
        max3 = vf.max(max3, src3, preg_main)
    max0 = vf.max(max0, max2, preg_main)
    max1 = vf.max(max1, max3, preg_main)
    max0 = vf.max(max0, max1, preg_main)
    max0 = vf.muls(max0, SCALE, preg_main)
    vf.store_align(max_tile, max0, preg_main)

    # Phase 2: ExpSub + Cast + DeInterleave + Store + Sum
    vreg_x_sum_0 = vf.full(0.0, preg_main)
    vreg_x_sum_1 = vf.full(0.0, preg_main)
    vreg_x_sum_2 = vf.full(0.0, preg_main)
    vreg_x_sum_3 = vf.full(0.0, preg_main)
    for i0 in pl.range(0, TKV // 4):
        vreg_x_f32_0 = vf.load_align(input_tile, i0 * TS_HALF)
        vreg_x_f32_1 = vf.load_align(input_tile, TKV * TS_HALF // 4 + i0 * TS_HALF)
        vreg_x_f32_2 = vf.load_align(input_tile, TKV * TS_HALF // 2 + i0 * TS_HALF)
        vreg_x_f32_3 = vf.load_align(input_tile, TKV * TS_HALF // 2 + TKV * TS_HALF // 4 + i0 * TS_HALF)
        vreg_x_f32_0 = vf.muls(vreg_x_f32_0, SCALE, preg_main)
        vreg_x_f32_1 = vf.muls(vreg_x_f32_1, SCALE, preg_main)
        vreg_x_f32_2 = vf.muls(vreg_x_f32_2, SCALE, preg_main)
        vreg_x_f32_3 = vf.muls(vreg_x_f32_3, SCALE, preg_main)
        vreg_x_exp_0 = vf.exp_sub(vreg_x_f32_0, max0, preg_main)
        vreg_x_exp_1 = vf.exp_sub(vreg_x_f32_1, max0, preg_main)
        vreg_x_exp_2 = vf.exp_sub(vreg_x_f32_2, max0, preg_main)
        vreg_x_exp_3 = vf.exp_sub(vreg_x_f32_3, max0, preg_main)
        vreg_x_exp_even_f16 = vf.astype(vreg_x_exp_0, preg_main, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP16)
        vreg_x_exp_odd_f16 = vf.astype(vreg_x_exp_2, preg_main, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP16)
        vreg_x_exp_f16_pack, vreg_x_exp_f16_packa = vf.de_interleave(vreg_x_exp_even_f16, vreg_x_exp_odd_f16)
        vreg_x_exp_even_f16_1 = vf.astype(vreg_x_exp_1, preg_main, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP16)
        vreg_x_exp_odd_f16_1 = vf.astype(vreg_x_exp_3, preg_main, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP16)
        vreg_x_exp_f16_1_pack, vreg_x_exp_f16_1_packa = vf.de_interleave(vreg_x_exp_even_f16_1, vreg_x_exp_odd_f16_1)
        vf.store_align(
            x_exp_tile,
            vreg_x_exp_f16_pack,
            preg_store,
            block_stride=block_stride_dn,
            repeat_stride=REPEAT_STRIDE_DN,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            post_update=True,
        )
        vreg_x_sum_0 = vf.add(vreg_x_exp_0, vreg_x_sum_0, preg_main)
        vreg_x_sum_2 = vf.add(vreg_x_exp_2, vreg_x_sum_2, preg_main)
        vf.store_align(
            x_exp_1,
            vreg_x_exp_f16_1_pack,
            preg_store,
            block_stride=block_stride_dn,
            repeat_stride=REPEAT_STRIDE_DN,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            post_update=True,
        )
        vreg_x_sum_1 = vf.add(vreg_x_exp_1, vreg_x_sum_1, preg_main)
        vreg_x_sum_3 = vf.add(vreg_x_exp_3, vreg_x_sum_3, preg_main)

    # Phase 3: Sum merge + store
    vreg_x_sum0 = vf.add(vreg_x_sum_2, vreg_x_sum_0, preg_main)
    vreg_x_sum1 = vf.add(vreg_x_sum_3, vreg_x_sum_1, preg_main)
    vreg_x_sum0 = vf.add(vreg_x_sum0, vreg_x_sum1, preg_main)
    vf.store_align(sum_tile, vreg_x_sum0, preg_main)


@pl.vector_function
def process_vec1_dn_update_vf(input_tile, x_exp_tile, max_tile, exp_max_tile, sum_tile, actual_sq_half, actual_skv):
    """Softmax DN VF kernel -- Update version with max/exp correction."""
    preg_main = vf.update_mask(actual_sq_half, dtype=pl.DT_FP32)
    preg_store = vf.update_mask(TS_HALF * 2, dtype=pl.DT_FP16)

    src_ub0 = input_tile
    src_ub1 = input_tile + TS_HALF
    src_ub2 = input_tile + TS_HALF * 2
    src_ub3 = input_tile + TS_HALF * 3
    x_exp_1 = x_exp_tile + TKV * 4

    # Phase 0: Fill invalid rows with minValue
    vreg_fill = vf.full(minValue)
    for r in pl.range(actual_skv, TKV):
        vf.store_align(input_tile + (r * TS_HALF), vreg_fill, preg_main)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)

    # Phase 1: ReduceMax (4-way parallel, only valid rows)
    max0 = vf.full(minValue)
    max1 = vf.full(minValue)
    max2 = vf.full(minValue)
    max3 = vf.full(minValue)
    for iter_m in pl.range(0, TKV // 4):
        src0 = vf.load_align(src_ub0, iter_m * TS_HALF * 4)
        src1 = vf.load_align(src_ub1, iter_m * TS_HALF * 4)
        src2 = vf.load_align(src_ub2, iter_m * TS_HALF * 4)
        src3 = vf.load_align(src_ub3, iter_m * TS_HALF * 4)
        max0 = vf.max(max0, src0, preg_main)
        max1 = vf.max(max1, src1, preg_main)
        max2 = vf.max(max2, src2, preg_main)
        max3 = vf.max(max3, src3, preg_main)
    vreg_x_max_f32_b = vf.load_align(max_tile, 0)
    max0 = vf.max(max0, max2, preg_main)
    max1 = vf.max(max1, max3, preg_main)
    max0 = vf.max(max0, max1, preg_main)
    max0 = vf.muls(max0, SCALE, preg_main)
    max0 = vf.max(max0, vreg_x_max_f32_b, preg_main)
    vreg_x_max_f32_b = vf.exp_sub(vreg_x_max_f32_b, max0, preg_main)
    vf.store_align(max_tile, max0, preg_main)
    vf.store_align(exp_max_tile, vreg_x_max_f32_b, preg_main)

    # Phase 2: ExpSub + Cast + DeInterleave + Store + Sum
    vreg_x_sum_0 = vf.full(0.0, preg_main)
    vreg_x_sum_1 = vf.full(0.0, preg_main)
    vreg_x_sum_2 = vf.full(0.0, preg_main)
    vreg_x_sum_3 = vf.full(0.0, preg_main)
    for i0 in pl.range(0, TKV // 4):
        vreg_x_f32_0 = vf.load_align(input_tile, i0 * TS_HALF)
        vreg_x_f32_2 = vf.load_align(input_tile, TKV * TS_HALF // 2 + i0 * TS_HALF)
        vreg_x_f32_0 = vf.muls(vreg_x_f32_0, SCALE, preg_main)
        vreg_x_f32_2 = vf.muls(vreg_x_f32_2, SCALE, preg_main)
        vreg_x_exp_0 = vf.exp_sub(vreg_x_f32_0, max0, preg_main)
        vreg_x_exp_2 = vf.exp_sub(vreg_x_f32_2, max0, preg_main)
        vreg_x_exp_even_f16 = vf.astype(vreg_x_exp_0, preg_main, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP16)
        vreg_x_exp_odd_f16 = vf.astype(vreg_x_exp_2, preg_main, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP16)
        vreg_x_exp_f16_pack, vreg_x_exp_f16_packa = vf.de_interleave(vreg_x_exp_even_f16, vreg_x_exp_odd_f16)
        vreg_x_f32_1 = vf.load_align(input_tile, TKV * TS_HALF // 4 + i0 * TS_HALF)
        vreg_x_f32_3 = vf.load_align(input_tile, TKV * TS_HALF // 2 + TKV * TS_HALF // 4 + i0 * TS_HALF)
        vreg_x_f32_1 = vf.muls(vreg_x_f32_1, SCALE, preg_main)
        vreg_x_f32_3 = vf.muls(vreg_x_f32_3, SCALE, preg_main)
        vreg_x_exp_1 = vf.exp_sub(vreg_x_f32_1, max0, preg_main)
        vreg_x_exp_3 = vf.exp_sub(vreg_x_f32_3, max0, preg_main)
        vreg_x_exp_even_f16_1 = vf.astype(vreg_x_exp_1, preg_main, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP16)
        vreg_x_exp_odd_f16_1 = vf.astype(vreg_x_exp_3, preg_main, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP16)
        vreg_x_exp_f16_1_pack, vreg_x_exp_f16_1_packa = vf.de_interleave(vreg_x_exp_even_f16_1, vreg_x_exp_odd_f16_1)
        vf.store_align(
            x_exp_tile,
            vreg_x_exp_f16_pack,
            preg_store,
            block_stride=block_stride_dn,
            repeat_stride=REPEAT_STRIDE_DN,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            post_update=True,
        )
        vreg_x_sum_0 = vf.add(vreg_x_exp_0, vreg_x_sum_0, preg_main)
        vreg_x_sum_2 = vf.add(vreg_x_exp_2, vreg_x_sum_2, preg_main)
        vf.store_align(
            x_exp_1,
            vreg_x_exp_f16_1_pack,
            preg_store,
            block_stride=block_stride_dn,
            repeat_stride=REPEAT_STRIDE_DN,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            post_update=True,
        )
        vreg_x_sum_1 = vf.add(vreg_x_exp_1, vreg_x_sum_1, preg_main)
        vreg_x_sum_3 = vf.add(vreg_x_exp_3, vreg_x_sum_3, preg_main)

    # Phase 3: Sum merge + update old sum
    vreg_x_sum0 = vf.add(vreg_x_sum_2, vreg_x_sum_0, preg_main)
    vreg_x_sum1 = vf.add(vreg_x_sum_3, vreg_x_sum_1, preg_main)
    vreg_x_sum0 = vf.add(vreg_x_sum0, vreg_x_sum1, preg_main)
    vreg_l0 = vf.load_align(sum_tile, 0)
    vreg_l0 = vf.mul(vreg_x_max_f32_b, vreg_l0, preg_main)
    vreg_l0 = vf.add(vreg_l0, vreg_x_sum0, preg_main)
    vf.store_align(sum_tile, vreg_l0, preg_main)


# ================================================================
#  FlashUpdate VF functions (compute_gu)
# ================================================================
@pl.vector_function
def flash_update_basic_vf(dst_tile, cur_tile, pre_tile, exp_max_tile):
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for i in pl.range(0, TS_HALF):
        vreg_exp_max = vf.load_align(exp_max_tile, i * REDUCE_SIZE, dist=pl.LoadDist.BRC_B32)
        for j in pl.range(0, D_LOOPS):
            vreg_input_pre = vf.load_align(pre_tile, i * TD + j * FLOAT_REP_SIZE)
            vreg_input_cur = vf.load_align(cur_tile, i * TD + j * FLOAT_REP_SIZE)
            vreg_mul = vf.mul(vreg_exp_max, vreg_input_pre, preg_all)
            vreg_add = vf.add(vreg_mul, vreg_input_cur, preg_all)
            vf.store_align(dst_tile + (i * TD + j * FLOAT_REP_SIZE), vreg_add, preg_all)


@pl.vector_function
def flash_update_last_basic_vf(dst_tile, cur_tile, pre_tile, exp_max_tile, exp_sum_tile):
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for i in pl.range(0, TS_HALF):
        vreg_exp_max = vf.load_align(exp_max_tile, i * REDUCE_SIZE, dist=pl.LoadDist.BRC_B32)
        vreg_exp_sum = vf.load_align(exp_sum_tile, i * REDUCE_SIZE, dist=pl.LoadDist.BRC_B32)
        for j in pl.range(0, D_LOOPS):
            vreg_input_pre = vf.load_align(pre_tile, i * TD + j * FLOAT_REP_SIZE)
            vreg_input_cur = vf.load_align(cur_tile, i * TD + j * FLOAT_REP_SIZE)
            vreg_mul = vf.mul(vreg_exp_max, vreg_input_pre, preg_all)
            vreg_add = vf.add(vreg_mul, vreg_input_cur, preg_all)
            vreg_div = vf.div(vreg_add, vreg_exp_sum, preg_all)
            vf.store_align(dst_tile + (i * TD + j * FLOAT_REP_SIZE), vreg_div, preg_all)


@pl.vector_function
def last_div_vf(dst_tile, cur_tile, exp_sum_tile):
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for i in pl.range(0, TS_HALF):
        vreg_exp_sum = vf.load_align(exp_sum_tile, i * REDUCE_SIZE, dist=pl.LoadDist.BRC_B32)
        for j in pl.range(0, D_LOOPS):
            vreg_input_cur = vf.load_align(cur_tile, i * TD + j * FLOAT_REP_SIZE)
            vreg_div = vf.div(vreg_input_cur, vreg_exp_sum, preg_all)
            vf.store_align(dst_tile + (i * TD + j * FLOAT_REP_SIZE), vreg_div, preg_all)


def compute_qk(
    ki,
    sq_off,
    skv_off,
    task_id_mod2,
    actual_skv,
    actual_sq,
    actual_sq_pad,
    actual_sq_half,
    cur_q,
    q,
    k,
    k_l1_db,
    left_db,
    right_db,
    acc_db,
    qk_vec_db,
):
    if ki == 0:
        pl.set_validshape(cur_q, [TD, actual_sq])
        pl.load(cur_q, q, [sq_off, 0], order=[1, 0])
    cur_k = k_l1_db.next()
    pl.set_validshape(cur_k, [actual_skv, TD])
    pl.load(cur_k, k, [skv_off, 0])

    qk_left = left_db.next()
    qk_right = right_db.next()
    qk_acc = acc_db.next()
    pl.set_validshape(qk_left, [actual_skv, TD])
    pl.move(qk_left, cur_k)
    pl.set_validshape(qk_right, [TD, actual_sq])
    pl.move(qk_right, cur_q)
    pl.set_validshape(qk_acc, [actual_skv, actual_sq_pad])
    pl.matmul(qk_acc, qk_left, qk_right)

    qk_dst = qk_vec_db.next()
    pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY_BARKWARD_IDS[task_id_mod2])
    pl.set_validshape(qk_acc, [actual_skv, actual_sq_pad])
    pl.set_validshape(qk_dst, [actual_skv, actual_sq_half])
    pl.move(qk_dst, qk_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN)
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY_FORWARD_IDS[task_id_mod2])


def compute_pv(
    ki,
    task_id_mod2,
    task_id_mod3,
    actual_skv,
    actual_sq_pad,
    actual_sq_half,
    v,
    v_l1_db,
    p_mat_db,
    left_db,
    right_db,
    acc_db,
    pv_vec_db,
):
    sv_off = ki * TKV
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=P_READY_FORWARD_IDS[task_id_mod3])
    cur_v = v_l1_db.next()
    pl.set_validshape(cur_v, [actual_skv, TD])
    pl.load(cur_v, v, [sv_off, 0])

    cur_p = p_mat_db.next()
    pv_left = left_db.next()
    pv_right = right_db.next()
    pv_acc = acc_db.next()
    pl.set_validshape(pv_left, [actual_sq_pad, actual_skv])
    pl.set_validshape(cur_p, [actual_skv, actual_sq_pad])
    pl.move(pv_left, cur_p)
    pl.set_validshape(pv_right, [actual_skv, TD])
    pl.move(pv_right, cur_v)
    pl.set_validshape(pv_acc, [actual_sq_pad, TD])
    pl.matmul(pv_acc, pv_left, pv_right)

    pv_dst = pv_vec_db.next()
    pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY_BARKWARD_IDS[task_id_mod2])
    pl.set_validshape(pv_acc, [actual_sq_pad, TD])
    pl.set_validshape(pv_dst, [actual_sq_half, TD])
    pl.move(pv_dst, pv_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY_FORWARD_IDS[task_id_mod2])


# ================================================================
#  Vector: compute_p (softmax) / compute_gu (flash update)
# ================================================================
def compute_p(
    ki,
    task_id_mod2,
    task_id_mod3,
    q_count_mod3,
    actual_skv,
    actual_sq_half,
    sub_id,
    global_max_rm_buf,
    global_sum_rm_buf,
    exp_corr_rm_fifo,
    qk_vec_db,
    p_f16_db,
    p_f16_main_db,
    p_f16_back_db,
    p_mat_db,
):
    """Softmax on KQ tile -> P. Column-direction VF; TINSERT with dynamic col offset."""
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_FORWARD_IDS[task_id_mod2])
    global_max_rm_cur = global_max_rm_buf[q_count_mod3]
    global_sum_rm_cur = global_sum_rm_buf[q_count_mod3]
    exp_corr_rm_cur = exp_corr_rm_fifo[task_id_mod3]

    qk_slot = qk_vec_db.next()
    p_f16_slot = p_f16_db.next()
    p_f16_main_slot = p_f16_main_db.next()
    p_f16_back_slot = p_f16_back_db.next()
    p_mat_slot = p_mat_db.next()

    if ki == 0:
        process_vec1_dn_no_update_vf(
            qk_slot, p_f16_slot, global_max_rm_cur, global_sum_rm_cur, actual_sq_half, actual_skv
        )
    if ki > 0:
        process_vec1_dn_update_vf(
            qk_slot, p_f16_slot, global_max_rm_cur, exp_corr_rm_cur, global_sum_rm_cur, actual_sq_half, actual_skv
        )
    pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_BARKWARD_IDS[task_id_mod2])

    # DN: TINSERT with dynamic column offset (actual_sq_half * sub_id). Full TKV//2 rows per half.
    pl.set_validshape(p_f16_main_slot, [TKV // 2, actual_sq_half])
    pl.set_validshape(p_f16_back_slot, [TKV // 2, actual_sq_half])
    pl.insert(p_mat_slot, p_f16_main_slot, [0, actual_sq_half * sub_id])
    pl.insert(p_mat_slot, p_f16_back_slot, [TKV // 2, actual_sq_half * sub_id])
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=P_READY_FORWARD_IDS[task_id_mod3])


def compute_gu(
    ki,
    skv_tiles,
    sq_off,
    task_id_mod2,
    task_id_mod3,
    q_count_mod3,
    actual_sq_half,
    real_rows,
    row_off,
    o,
    pv_vec_db,
    running_o,
    o_f16,
    exp_corr_rm_fifo,
    global_sum_rm_buf,
):
    """GU: running output update. running_o/pv have actual_sq_half rows; only real_rows stored to o."""
    pv_slot = pv_vec_db.next()
    gsum_gu = global_sum_rm_buf[q_count_mod3]
    exp_corr_gu = exp_corr_rm_fifo[task_id_mod3]
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_FORWARD_IDS[task_id_mod2])

    pl.set_validshape(running_o, [actual_sq_half, TD])
    pl.set_validshape(pv_slot, [actual_sq_half, TD])
    if ki == 0:
        pl.move(running_o, pv_slot)
    else:
        if ki < skv_tiles - 1:
            flash_update_basic_vf(running_o, pv_slot, running_o, exp_corr_gu)
        else:
            flash_update_last_basic_vf(running_o, pv_slot, running_o, exp_corr_gu, gsum_gu)
            pl.set_validshape(o_f16, [real_rows, TD])
            pl.cast(o_f16, running_o, mode=pl.RoundMode.CAST_ROUND)
            pl.store(o, o_f16, [sq_off + row_off, 0])
    pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_BARKWARD_IDS[task_id_mod2])

    if ki == skv_tiles - 1:
        if ki == 0:
            last_div_vf(running_o, running_o, gsum_gu)
            pl.set_validshape(o_f16, [real_rows, TD])
            pl.cast(o_f16, running_o, mode=pl.RoundMode.CAST_ROUND)
            pl.store(o, o_f16, [sq_off + row_off, 0])


# ================================================================
#  Kernel -- tile_group + auto_mutex
# ================================================================
@pl.jit(auto_mutex=True)
def fa_perf_tkv_preload_dn_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    v: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    o: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    sq_dim = q.shape[0]
    skv_dim = k.shape[0]
    sq_tiles = (sq_dim + (TS - 1)) // TS
    skv_tiles = (skv_dim + (TKV - 1)) // TKV
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx() // pl.get_subblock_num()

    # ===== Cross-core shared buffers (declared outside the sections) =====
    qk_vec_db = pl.make_tile_group(
        type=pl.TileType(
            shape=[TKV, TS_HALF],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
            compact=1,
            pad=pl.TilePad.min,
        ),
        addrs=VA0,
        mutex_ids=[14, 15],
    )
    pv_vec_db = pl.make_tile_group(
        type=pl.TileType(shape=[TS_HALF, TD], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addrs=VA8,
        mutex_ids=[16, 17],
    )
    p_mat_db = pl.make_tile_group(
        type=pl.TileType(
            shape=[TS, TKV], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN, valid_shape=[-1, -1]
        ),
        addrs=MA2,
        mutex_ids=[18, 19, 20],
    )

    # =================== CUBE SECTION ===================
    with pl.section_cube():
        q_l1_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TD, TS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN, valid_shape=[-1, -1]
            ),
            addrs=MA0,
            mutex_ids=[0, 1],
        )
        k_l1_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TKV, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ, valid_shape=[-1, -1]
            ),
            addrs=MA1,
            mutex_ids=[2, 3],
        )
        v_l1_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TKV, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ, valid_shape=[-1, -1]
            ),
            addrs=MA3,
            mutex_ids=[4, 5],
        )
        left_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TKV, TD],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Left,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=LA0,
            mutex_ids=[6, 7],
        )
        right_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TD, TS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, valid_shape=[-1, -1], compact=1
            ),
            addrs=RA0,
            mutex_ids=[8, 9],
        )
        acc_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TKV, TS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1], compact=1
            ),
            addrs=CA0,
            mutex_ids=[10, 11, 12, 13],
        )

        task_id = 0
        ctx_arr = pl.struct_array(
            4,
            "CubeCtx",
            ki=0,
            task_id_mod2=0,
            task_id_mod3=0,
            actual_skv=0,
            actual_sq=0,
            actual_sq_pad=0,
            actual_sq_half=0,
        )
        for qi in pl.range(core_id, sq_tiles, num_cores):
            sq_off = qi * TS
            actual_sq = pl.min(sq_dim - sq_off, TS)
            actual_sq_pad = ((actual_sq + 31) // 32) * 32
            actual_sq_half = actual_sq_pad // 2
            cur_q = q_l1_db.next()
            ki_end = skv_tiles
            if qi + num_cores >= sq_tiles:
                ki_end = skv_tiles + QK_PRELOAD
            for ki in pl.range(0, ki_end):
                if ki < skv_tiles:
                    skv_off = ki * TKV
                    actual_skv = pl.min(skv_dim - skv_off, TKV)
                    ctx_curr = ctx_arr[task_id % 4]
                    ctx_curr.ki = ki
                    ctx_curr.task_id_mod2 = task_id % 2
                    ctx_curr.task_id_mod3 = task_id % 3
                    ctx_curr.actual_skv = actual_skv
                    ctx_curr.actual_sq = actual_sq
                    ctx_curr.actual_sq_pad = actual_sq_pad
                    ctx_curr.actual_sq_half = actual_sq_half
                    compute_qk(
                        ki,
                        sq_off,
                        skv_off,
                        task_id % 2,
                        actual_skv,
                        actual_sq,
                        actual_sq_pad,
                        actual_sq_half,
                        cur_q,
                        q,
                        k,
                        k_l1_db,
                        left_db,
                        right_db,
                        acc_db,
                        qk_vec_db,
                    )
                if task_id > 1:
                    prev = ctx_arr[(task_id + 2) % 4]
                    compute_pv(
                        prev.ki,
                        prev.task_id_mod2,
                        prev.task_id_mod3,
                        prev.actual_skv,
                        prev.actual_sq_pad,
                        prev.actual_sq_half,
                        v,
                        v_l1_db,
                        p_mat_db,
                        left_db,
                        right_db,
                        acc_db,
                        pv_vec_db,
                    )
                task_id = task_id + 1

    # =================== VECTOR SECTION ===================
    with pl.section_vector():
        p_f16_db = pl.make_tile_group(
            type=pl.TileType(shape=[TKV // 2 + 1, TS_HALF * 2], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addrs=[VA2, VA2B],
            mutex_ids=[0, 1],
        )
        p_f16_main_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TKV // 2 + 1, TS_HALF],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Vec,
                valid_shape=[-1, -1],
                layout=pl.NZ,
                compact=2,
            ),
            addrs=[VA2, VA2B],
            mutex_ids=[0, 1],
        )
        p_f16_back_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TKV // 2 + 1, TS_HALF],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Vec,
                valid_shape=[-1, -1],
                layout=pl.NZ,
                compact=2,
            ),
            addrs=[VA2_DN, VA2B_DN],
            mutex_ids=[0, 1],
        )

        red_rm_type = pl.TileType(shape=[1, TS_HALF], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        gmax_rm_0 = pl.make_tile(red_rm_type, addr=VA_GMAX0)
        gmax_rm_1 = pl.make_tile(red_rm_type, addr=VA_GMAX1)
        gmax_rm_2 = pl.make_tile(red_rm_type, addr=VA_GMAX2)
        global_max_rm_buf = (gmax_rm_0, gmax_rm_1, gmax_rm_2)
        gsum_rm_0 = pl.make_tile(red_rm_type, addr=VA_GSUM0)
        gsum_rm_1 = pl.make_tile(red_rm_type, addr=VA_GSUM1)
        gsum_rm_2 = pl.make_tile(red_rm_type, addr=VA_GSUM2)
        global_sum_rm_buf = (gsum_rm_0, gsum_rm_1, gsum_rm_2)
        ec_rm_0 = pl.make_tile(red_rm_type, addr=VA_EC0)
        ec_rm_1 = pl.make_tile(red_rm_type, addr=VA_EC1)
        ec_rm_2 = pl.make_tile(red_rm_type, addr=VA_EC2)
        exp_corr_rm_fifo = (ec_rm_0, ec_rm_1, ec_rm_2)

        o_f16_buf = pl.make_tile_group(
            type=pl.TileType(
                shape=[TS_HALF, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
            ),
            addrs=VA9,
            mutex_ids=[11],
        )
        running_o_buf = pl.make_tile_group(
            type=pl.TileType(
                shape=[TS_HALF, TD], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
            ),
            addrs=VA7,
            mutex_ids=[12],
        )
        o_f16 = o_f16_buf.next()
        running_o = running_o_buf.next()

        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_BARKWARD_IDS[0])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_BARKWARD_IDS[1])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_BARKWARD_IDS[0])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_BARKWARD_IDS[1])

        task_id = 0
        q_count = 0
        sub_id = pl.get_subblock_idx()
        ctx_arr = pl.struct_array(
            4,
            "VecCtx",
            sq_off=0,
            ki=0,
            skv_tiles=0,
            task_id_mod2=0,
            task_id_mod3=0,
            q_count_mod3=0,
            actual_skv=0,
            actual_sq_half=0,
            real_rows=0,
            row_off=0,
        )
        for qi in pl.range(core_id, sq_tiles, num_cores):
            sq_off = qi * TS
            actual_sq = pl.min(sq_dim - sq_off, TS)
            actual_sq_pad = ((actual_sq + 31) // 32) * 32
            actual_sq_half = actual_sq_pad // 2
            sub0_real_rows = pl.min(actual_sq, actual_sq_half)
            sub1_real_rows = actual_sq - sub0_real_rows
            real_rows = (1 - sub_id) * sub0_real_rows + sub_id * sub1_real_rows
            row_off = sub_id * actual_sq_half
            ki_end = skv_tiles
            if qi + num_cores >= sq_tiles:
                ki_end = skv_tiles + 3
            for ki in pl.range(0, ki_end):
                if ki < skv_tiles:
                    skv_off = ki * TKV
                    actual_skv = pl.min(skv_dim - skv_off, TKV)
                    ctx_curr = ctx_arr[task_id % 4]
                    ctx_curr.sq_off = sq_off
                    ctx_curr.ki = ki
                    ctx_curr.skv_tiles = skv_tiles
                    ctx_curr.task_id_mod2 = task_id % 2
                    ctx_curr.task_id_mod3 = task_id % 3
                    ctx_curr.q_count_mod3 = q_count % 3
                    ctx_curr.actual_skv = actual_skv
                    ctx_curr.actual_sq_half = actual_sq_half
                    ctx_curr.real_rows = real_rows
                    ctx_curr.row_off = row_off
                if task_id > 0:
                    if ki < skv_tiles + 1:
                        p_ctx = ctx_arr[(task_id + 3) % 4]
                        compute_p(
                            p_ctx.ki,
                            p_ctx.task_id_mod2,
                            p_ctx.task_id_mod3,
                            p_ctx.q_count_mod3,
                            p_ctx.actual_skv,
                            p_ctx.actual_sq_half,
                            sub_id,
                            global_max_rm_buf,
                            global_sum_rm_buf,
                            exp_corr_rm_fifo,
                            qk_vec_db,
                            p_f16_db,
                            p_f16_main_db,
                            p_f16_back_db,
                            p_mat_db,
                        )
                if task_id > 2:
                    g_ctx = ctx_arr[(task_id + 1) % 4]
                    compute_gu(
                        g_ctx.ki,
                        g_ctx.skv_tiles,
                        g_ctx.sq_off,
                        g_ctx.task_id_mod2,
                        g_ctx.task_id_mod3,
                        g_ctx.q_count_mod3,
                        g_ctx.actual_sq_half,
                        g_ctx.real_rows,
                        g_ctx.row_off,
                        o,
                        pv_vec_db,
                        running_o,
                        o_f16,
                        exp_corr_rm_fifo,
                        global_sum_rm_buf,
                    )
                task_id = task_id + 1
            q_count = q_count + 1


# ================================================================
#  Reference + Tests
# ================================================================
def flash_attention_ref(q, k, v, d):
    scale_val = 1.0 / math.sqrt(d)
    qk = torch.matmul(q.float(), k.float().T)
    scale = qk * scale_val
    max_val = torch.max(scale, dim=-1, keepdim=True).values
    x_sub = scale - max_val
    x_exp = torch.exp(x_sub)
    attn = x_exp / torch.sum(x_exp, dim=-1, keepdim=True)
    return qk, x_exp, torch.matmul(attn, v.float()).half()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_fa_perf():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"
    torch.manual_seed(42)
    for sq, skv, d, num_cores in [
        (64, 64, TD, 1),
        (34, 62, TD, 1),
        (162, 190, TD, 1),
        (8192, 8192, TD, 28),  # 950 默认流预算 28 cube 核
        (8128, 8128, TD, 28),
        (8111, 7777, TD, 28),
    ]:
        logging.info("\nFA-Perf DN tile-group (%s,%s,%s) cores=%s  QK_PRELOAD=%s", sq, skv, d, num_cores, QK_PRELOAD)
        q_t = torch.rand((sq, d), device=device, dtype=torch.float16)
        k_t = torch.rand((skv, d), device=device, dtype=torch.float16)
        v_t = torch.rand((skv, d), device=device, dtype=torch.float16)
        o_t = torch.zeros((sq, d), device=device, dtype=torch.float16)

        qk_ref, x_exp_ref, o_ref = flash_attention_ref(q_t, k_t, v_t, d)

        prev_diff = None
        for i in range(20):
            o_t.zero_()
            fa_perf_tkv_preload_dn_kernel[None, num_cores](q_t, k_t, v_t, o_t)
            torch.npu.synchronize()
            diff = (o_t - o_ref).abs().max().item()
            logging.info("  run %s: max|diff|=%.6f", i, diff)
            if prev_diff is not None and diff != prev_diff:
                raise AssertionError(f"Non-deterministic result! run {i} diff={diff:.6f} vs prev={prev_diff:.6f}")
            prev_diff = diff

        torch.testing.assert_close(o_t, o_ref, rtol=5e-3, atol=5e-3)
        logging.info("  PASS (deterministic, max|diff|=%.6f)", prev_diff)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info(
        "FA perf DN tile-group: double-buffer + QK pre-compute (QK_PRELOAD=%s, FIFO=%s)", QK_PRELOAD, FIFO_SIZE
    )
    logging.info("%s", '=' * 60)
    test_fa_perf()
    logging.info("\nAll FlashAttention DN tile-group tests passed!")
