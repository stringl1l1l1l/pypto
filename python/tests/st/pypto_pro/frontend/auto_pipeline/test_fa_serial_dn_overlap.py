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

"""FlashAttention performance kernel using PyPTO IR manual (non-SSA) mode — DN mode.

DN (DecN) mode differences vs standard ND mode:
  - compute_qk: K x Q^T (left/right swapped vs Q x K^T),
                Q loaded with is_transpose=True -> L1 shape [TD, TS] layout=pl.ZN,
                K loaded normally -> L1 shape [TKV, TD] layout=pl.NZ,
                matmul Left=K, Right=Q^T, acc output shape [TKV, TS],
                acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN (split along N=TS axis)
  - softmax:    column-direction ops on qk_vec[TKV, TS_HALF]
                (col_max / col_expand_sub / col_sum replacing row_* equivalents)
  - compute_pv: P[TS, TKV] (layout=pl.ZN) x V[TKV, TD] -> acc[TS, TD],
                TINSERT uses offset[1]=TS_HALF*sub_id (column offset)

Reference: fa_performance_dn_kernel.cpp
"""

import logging
import math
import os

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

# ================================================================
#  Configuration — change QK_PRELOAD to tune pre-compute depth
# ================================================================
QK_PRELOAD = 2          # How many KV tiles to pre-compute QK ahead
FIFO_SIZE = QK_PRELOAD + 1  # Exp-corr FIFO depth (avoids read/write collision)

# ================================================================
#  Tile dimensions and constants
# ================================================================
TS = 128
TKV = 128
TD = 128
TS_HALF = TS // 2
SCALE = 1.0 / math.sqrt(TD)
FLOAT_REP_SIZE = 64  # elements per fp32 register
D_LOOPS = TD // FLOAT_REP_SIZE
REDUCE_SIZE = 1
minValue = -1e9  # noqa: N816
block_stride_dn = TKV >> 1 | 0x1
REPEAT_STRIDE_DN = 1
vec1_s2_stride_dn = TKV * 8

# Cube tile byte sizes
Q_F16 = TS * TD * 2
KT_F16 = TKV * TD * 2
V_F16 = TKV * TD * 2
P_F16 = TS * TKV * 2
QK_HALF_F32 = TKV * TS * 4
PV_HALF_F32 = TS * TD * 4

# ---- MAT (512KB) ----
MA0 = 0
MA0_PONG = MA0 + Q_F16
MA1 = Q_F16 * 2
MA1_PONG = MA1 + KT_F16
MA2 = MA1 + KT_F16 * 2
MA2_PONG = MA2 + P_F16
MA2_PONG2 = MA2_PONG + P_F16
MA3 = MA2 + P_F16 * 3
MA3_PONG = MA3 + V_F16
LA0 = 0
LA1 = KT_F16
RA0 = 0
RA1 = Q_F16
CA0 = 0
CA1 = QK_HALF_F32
CA2 = QK_HALF_F32 * 2
CA3 = QK_HALF_F32 * 3

# ---- VEC addresses (248KB on a5) ----
VB4_KV = TS_HALF * TKV * 4
VB2_KV = TS_HALF * 2 * (TKV // 2 + 1) * 2
VB2_KV_HALF = VB2_KV // 2
VB4 = TS_HALF * TD * 4
VB2 = TS_HALF * TD * 2
VB_RED = TS_HALF * 1 * 4

VA0 = 0
VA2 = VA0 + 2 * VB4_KV
VA2_DN = VA2 + VB2_KV_HALF
VA2B = VA2 + VB2_KV
VA2B_DN = VA2B + VB2_KV_HALF
VA8 = VA2 + 2 * VB2_KV
VA7 = VA8 + 2 * VB4
VA9 = VA7 + VB4
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


PV_CORE_STRIDE = 2 * FIFO_SIZE * TS


# ================================================================
#  VF softmax functions (take explicit tile pointers, no NBuffer access)
# ================================================================
@pl.vector_function
def process_vec1_dn_no_update_vf(input_tile, x_exp_tile, max_tile, sum_tile):
    """Softmax DN VF kernel — full version with Cast + DeInterleave + Store."""
    preg_108 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_136 = vf.update_mask(128, dtype=pl.DT_FP16)
    src_ub0 = input_tile
    src_ub1 = input_tile + TS_HALF
    src_ub2 = input_tile + TS_HALF * 2
    src_ub3 = input_tile + TS_HALF * 3
    x_exp_1 = x_exp_tile + TKV * 4
    max0 = vf.full(minValue)
    max1 = vf.full(minValue)
    max2 = vf.full(minValue)
    max3 = vf.full(minValue)
    for iter_m in pl.range(0, TKV // 4):
        src0 = vf.load_align(src_ub0, iter_m * TS_HALF * 4)
        src1 = vf.load_align(src_ub1, iter_m * TS_HALF * 4)
        src2 = vf.load_align(src_ub2, iter_m * TS_HALF * 4)
        src3 = vf.load_align(src_ub3, iter_m * TS_HALF * 4)
        max0 = vf.max(max0, src0, preg_108)
        max1 = vf.max(max1, src1, preg_108)
        max2 = vf.max(max2, src2, preg_108)
        max3 = vf.max(max3, src3, preg_108)
    max0 = vf.max(max0, max2, preg_108)
    max1 = vf.max(max1, max3, preg_108)
    max0 = vf.max(max0, max1, preg_108)
    max0 = vf.muls(max0, SCALE, preg_108)
    vf.store_align(max_tile, max0, preg_108)
    vreg_x_sum_0 = vf.full(0.0, preg_108)
    vreg_x_sum_1 = vf.full(0.0, preg_108)
    vreg_x_sum_2 = vf.full(0.0, preg_108)
    vreg_x_sum_3 = vf.full(0.0, preg_108)
    for i0 in pl.range(0, TKV // 4):
        vreg_x_f32_0 = vf.load_align(input_tile, i0 * TS_HALF)
        vreg_x_f32_1 = vf.load_align(input_tile, TKV * TS_HALF // 4 + i0 * TS_HALF)
        vreg_x_f32_2 = vf.load_align(input_tile, TKV * TS_HALF // 2 + i0 * TS_HALF)
        vreg_x_f32_3 = vf.load_align(input_tile, TKV * TS_HALF // 2 + TKV * TS_HALF // 4 + i0 * TS_HALF)
        vreg_x_f32_0 = vf.muls(vreg_x_f32_0, SCALE, preg_108)
        vreg_x_f32_1 = vf.muls(vreg_x_f32_1, SCALE, preg_108)
        vreg_x_f32_2 = vf.muls(vreg_x_f32_2, SCALE, preg_108)
        vreg_x_f32_3 = vf.muls(vreg_x_f32_3, SCALE, preg_108)
        vreg_x_exp_0 = vf.exp_sub(vreg_x_f32_0, max0, preg_108)
        vreg_x_exp_1 = vf.exp_sub(vreg_x_f32_1, max0, preg_108)
        vreg_x_exp_2 = vf.exp_sub(vreg_x_f32_2, max0, preg_108)
        vreg_x_exp_3 = vf.exp_sub(vreg_x_f32_3, max0, preg_108)
        vreg_x_exp_even_f16 = vf.astype(vreg_x_exp_0, preg_108, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP16)
        vreg_x_exp_odd_f16 = vf.astype(vreg_x_exp_2, preg_108, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP16)
        vreg_x_exp_f16_pack, vreg_x_exp_f16_packa = vf.de_interleave(vreg_x_exp_even_f16, vreg_x_exp_odd_f16)
        vreg_x_exp_even_f16_1 = vf.astype(vreg_x_exp_1, preg_108, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP16)
        vreg_x_exp_odd_f16_1 = vf.astype(vreg_x_exp_3, preg_108, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP16)
        vreg_x_exp_f16_1_pack, vreg_x_exp_f16_1_packa = vf.de_interleave(vreg_x_exp_even_f16_1, vreg_x_exp_odd_f16_1)
        vf.store_align(x_exp_tile, vreg_x_exp_f16_pack, preg_136,
                      block_stride=block_stride_dn, repeat_stride=REPEAT_STRIDE_DN,
                      data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)
        vreg_x_sum_0 = vf.add(vreg_x_exp_0, vreg_x_sum_0, preg_108)
        vreg_x_sum_2 = vf.add(vreg_x_exp_2, vreg_x_sum_2, preg_108)
        vf.store_align(x_exp_1, vreg_x_exp_f16_1_pack, preg_136,
                      block_stride=block_stride_dn, repeat_stride=REPEAT_STRIDE_DN,
                      data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)
        vreg_x_sum_1 = vf.add(vreg_x_exp_1, vreg_x_sum_1, preg_108)
        vreg_x_sum_3 = vf.add(vreg_x_exp_3, vreg_x_sum_3, preg_108)
    vreg_x_sum0 = vf.add(vreg_x_sum_2, vreg_x_sum_0, preg_108)
    vreg_x_sum1 = vf.add(vreg_x_sum_3, vreg_x_sum_1, preg_108)
    vreg_x_sum0 = vf.add(vreg_x_sum0, vreg_x_sum1, preg_108)
    vf.store_align(sum_tile, vreg_x_sum0, preg_108)


@pl.vector_function
def process_vec1_dn_update_vf(input_tile, x_exp_tile, max_tile, exp_max_tile, sum_tile):
    """Softmax DN VF kernel — Update version with max/exp correction."""
    preg_108 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_136 = vf.update_mask(128, dtype=pl.DT_FP16)
    src_ub0 = input_tile
    src_ub1 = input_tile + TS_HALF
    src_ub2 = input_tile + TS_HALF * 2
    src_ub3 = input_tile + TS_HALF * 3
    x_exp_1 = x_exp_tile + TKV * 4
    max0 = vf.full(minValue)
    max1 = vf.full(minValue)
    max2 = vf.full(minValue)
    max3 = vf.full(minValue)
    for iter_m in pl.range(0, TKV // 4):
        src0 = vf.load_align(src_ub0, iter_m * TS_HALF * 4)
        src1 = vf.load_align(src_ub1, iter_m * TS_HALF * 4)
        src2 = vf.load_align(src_ub2, iter_m * TS_HALF * 4)
        src3 = vf.load_align(src_ub3, iter_m * TS_HALF * 4)
        max0 = vf.max(max0, src0, preg_108)
        max1 = vf.max(max1, src1, preg_108)
        max2 = vf.max(max2, src2, preg_108)
        max3 = vf.max(max3, src3, preg_108)
    vreg_x_max_f32_b = vf.load_align(max_tile, 0)
    max0 = vf.max(max0, max2, preg_108)
    max1 = vf.max(max1, max3, preg_108)
    max0 = vf.max(max0, max1, preg_108)
    max0 = vf.muls(max0, SCALE, preg_108)
    max0 = vf.max(max0, vreg_x_max_f32_b, preg_108)
    vreg_x_max_f32_b = vf.exp_sub(vreg_x_max_f32_b, max0, preg_108)
    vf.store_align(max_tile, max0, preg_108)
    vf.store_align(exp_max_tile, vreg_x_max_f32_b, preg_108)
    vreg_x_sum_0 = vf.full(0.0, preg_108)
    vreg_x_sum_1 = vf.full(0.0, preg_108)
    vreg_x_sum_2 = vf.full(0.0, preg_108)
    vreg_x_sum_3 = vf.full(0.0, preg_108)
    for i0 in pl.range(0, TKV // 4):
        vreg_x_f32_0 = vf.load_align(input_tile, i0 * TS_HALF)
        vreg_x_f32_1 = vf.load_align(input_tile, TKV * TS_HALF // 4 + i0 * TS_HALF)
        vreg_x_f32_2 = vf.load_align(input_tile, TKV * TS_HALF // 2 + i0 * TS_HALF)
        vreg_x_f32_3 = vf.load_align(input_tile, TKV * TS_HALF // 2 + TKV * TS_HALF // 4 + i0 * TS_HALF)
        vreg_x_f32_0 = vf.muls(vreg_x_f32_0, SCALE, preg_108)
        vreg_x_f32_1 = vf.muls(vreg_x_f32_1, SCALE, preg_108)
        vreg_x_f32_2 = vf.muls(vreg_x_f32_2, SCALE, preg_108)
        vreg_x_f32_3 = vf.muls(vreg_x_f32_3, SCALE, preg_108)
        vreg_x_exp_0 = vf.exp_sub(vreg_x_f32_0, max0, preg_108)
        vreg_x_exp_1 = vf.exp_sub(vreg_x_f32_1, max0, preg_108)
        vreg_x_exp_2 = vf.exp_sub(vreg_x_f32_2, max0, preg_108)
        vreg_x_exp_3 = vf.exp_sub(vreg_x_f32_3, max0, preg_108)
        vreg_x_exp_even_f16 = vf.astype(vreg_x_exp_0, preg_108, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP16)
        vreg_x_exp_odd_f16 = vf.astype(vreg_x_exp_2, preg_108, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP16)
        vreg_x_exp_f16_pack, vreg_x_exp_f16_packa = vf.de_interleave(vreg_x_exp_even_f16, vreg_x_exp_odd_f16)
        vreg_x_exp_even_f16_1 = vf.astype(vreg_x_exp_1, preg_108, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP16)
        vreg_x_exp_odd_f16_1 = vf.astype(vreg_x_exp_3, preg_108, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP16)
        vreg_x_exp_f16_1_pack, vreg_x_exp_f16_1_packa = vf.de_interleave(vreg_x_exp_even_f16_1, vreg_x_exp_odd_f16_1)
        vf.store_align(x_exp_tile, vreg_x_exp_f16_pack, preg_136,
                      block_stride=block_stride_dn, repeat_stride=REPEAT_STRIDE_DN,
                      data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)
        vreg_x_sum_0 = vf.add(vreg_x_exp_0, vreg_x_sum_0, preg_108)
        vreg_x_sum_2 = vf.add(vreg_x_exp_2, vreg_x_sum_2, preg_108)
        vf.store_align(x_exp_1, vreg_x_exp_f16_1_pack, preg_136,
                      block_stride=block_stride_dn, repeat_stride=REPEAT_STRIDE_DN,
                      data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)
        vreg_x_sum_1 = vf.add(vreg_x_exp_1, vreg_x_sum_1, preg_108)
        vreg_x_sum_3 = vf.add(vreg_x_exp_3, vreg_x_sum_3, preg_108)
    vreg_x_sum0 = vf.add(vreg_x_sum_2, vreg_x_sum_0, preg_108)
    vreg_x_sum1 = vf.add(vreg_x_sum_3, vreg_x_sum_1, preg_108)
    vreg_x_sum0 = vf.add(vreg_x_sum0, vreg_x_sum1, preg_108)
    vreg_l0 = vf.load_align(sum_tile, 0)
    vreg_l0 = vf.mul(vreg_x_max_f32_b, vreg_l0, preg_108)
    vreg_l0 = vf.add(vreg_l0, vreg_x_sum0, preg_108)
    vf.store_align(sum_tile, vreg_l0, preg_108)


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

# ================================================================
#  Stage functions — each receives NBuffers/tiles/scalars as params
# ================================================================


@pl.pipeline.stage
def compute_qk(ki, sq_off, tick, q, k, cur_q_slot, k_l1_db, left_db, right_db, acc_db, qk_vec_db):
    """QK matmul stage (Cube). Writes qk_vec."""
    skv_off = ki * TKV
    if ki == 0:
        pl.load(cur_q_slot, q, [sq_off, 0], order=[1, 0])
    cur_k_slot = k_l1_db.next()
    pl.load(cur_k_slot, k, [skv_off, 0])
    qk_left = left_db.next()
    qk_right = right_db.next()
    qk_acc = acc_db.next()
    pl.move(qk_left, cur_k_slot)
    pl.move(qk_right, cur_q_slot)
    pl.matmul(qk_acc, qk_left, qk_right)
    qk_dst = qk_vec_db.next()
    pl.move(qk_dst, qk_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN)


@pl.pipeline.stage
def compute_pv(ki, tick, v, v_l1_db, p_mat_db, left_db, right_db, acc_db, pv_vec_db):
    """PV matmul stage (Cube). Reads p_mat, writes pv_vec."""
    sv_off = ki * TKV
    cur_v_slot = v_l1_db.next()
    cur_p_slot = p_mat_db.next()
    pv_left = left_db.next()
    pv_right = right_db.next()
    pv_acc = acc_db.next()
    pl.load(cur_v_slot, v, [sv_off, 0])
    pl.move(pv_left, cur_p_slot)
    pl.move(pv_right, cur_v_slot)
    pl.matmul(pv_acc, pv_left, pv_right)
    pv_dst = pv_vec_db.next()
    pl.move(pv_dst, pv_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)


@pl.pipeline.stage
def compute_p(ki, tick, q_count, sub_id,
              qk_vec_db, p_f16_db, p_f16_main_db, p_f16_back_db, p_mat_db,
              global_max_rm_buf, global_sum_rm_buf, exp_corr_rm_fifo):
    """Softmax stage (Vector). Reads qk_vec, writes p_mat."""
    global_max_rm_cur = global_max_rm_buf[q_count % 3]
    global_sum_rm_cur = global_sum_rm_buf[q_count % 3]
    exp_corr_rm_cur = exp_corr_rm_fifo[tick % 3]
    qk_slot = qk_vec_db.next()
    p_f16_slot = p_f16_db.next()
    p_f16_main_slot = p_f16_main_db.next()
    p_f16_back_slot = p_f16_back_db.next()
    p_mat_slot = p_mat_db.next()
    if ki == 0:
        process_vec1_dn_no_update_vf(
            qk_slot,
            p_f16_slot,
            global_max_rm_cur,
            global_sum_rm_cur,
        )
    if ki > 0:
        process_vec1_dn_update_vf(
            qk_slot,
            p_f16_slot,
            global_max_rm_cur,
            exp_corr_rm_cur,
            global_sum_rm_cur,
        )
    pl.insert(p_mat_slot, p_f16_main_slot, [0, TS_HALF * sub_id])
    pl.insert(p_mat_slot, p_f16_back_slot, [TKV // 2, TS_HALF * sub_id])


@pl.pipeline.stage
def compute_gu(ki, skv_tiles, tick, q_count, sq_off, row_off, o,
               pv_vec_db, running_o_buf, o_f16_buf,
               global_sum_rm_buf, exp_corr_rm_fifo):
    """Flash update stage (Vector). Reads pv_vec, writes output."""
    pv_slot = pv_vec_db.next()
    running_o = running_o_buf.next()
    o_f16 = o_f16_buf.next()
    gsum_gu = global_sum_rm_buf[q_count % 3]
    exp_corr_gu = exp_corr_rm_fifo[tick % 3]
    if ki == 0:
        pl.move(running_o, pv_slot)
    else:
        if ki < skv_tiles - 1:
            flash_update_basic_vf(running_o, pv_slot, running_o, exp_corr_gu)
        else:
            flash_update_last_basic_vf(
                running_o,
                pv_slot,
                running_o,
                exp_corr_gu,
                gsum_gu,
            )
            pl.cast(o_f16, running_o, mode=pl.RoundMode.CAST_ROUND)
            pl.store(o, o_f16, [sq_off + row_off, 0])
    if ki == skv_tiles - 1:
        if ki == 0:
            last_div_vf(running_o, running_o, gsum_gu)
            pl.cast(o_f16, running_o, mode=pl.RoundMode.CAST_ROUND)
            pl.store(o, o_f16, [sq_off + row_off, 0])

# ================================================================
#  Kernel — serial execution, AscendC-style interleaved sections
# ================================================================


@pl.jit(auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=1))
def fa_perf_tkv_preload_dn_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    v: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    o: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    qk_buf: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    p_buf: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    pv_buf: pl.Tensor[[48 * PV_CORE_STRIDE, pl.DYNAMIC], pl.DT_FP32],
):

    sq_dim = q.shape[0]
    skv_dim = k.shape[0]
    sq_tiles = (sq_dim + (TS - 1)) // TS
    skv_tiles = (skv_dim + (TKV - 1)) // TKV
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx() // pl.get_subblock_num()

    # ===== Cross-core shared buffers (outside sections) =====
    qk_vec_db = pl.make_tile_group(
        type=pl.TileType(shape=[TKV, TS_HALF], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=VA0, mutex_ids=[14, 15],
        fwd_ids=[0, 1], bwd_ids=[2, 3])
    # OVERLAP PROBE: share physical address with qk_vec_db (both VA0, both 2-slot,
    # both Vec) to trigger the framework's address-overlap reverse-sync path and
    # test whether preload=2 reverse (inverse-time) sync deadlocks in practice.
    # mutex_ids match qk_vec_db's: a mutex locks the ADDRESS, so two buffers over one
    # region must take the same lock or they do not exclude each other at all.
    pv_vec_db = pl.make_tile_group(
        type=pl.TileType(shape=[TS_HALF, TD], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=VA0, mutex_ids=[14, 15],
        fwd_ids=[7, 8], bwd_ids=[9, 10])
    p_mat_db = pl.make_tile_group(
        type=pl.TileType(shape=[TS, TKV], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat,
                         layout=pl.ZN),
        addrs=MA2, mutex_ids=[18, 19, 20],
        fwd_ids=[4, 5, 6])

    # ===== Cube-local buffer declarations (in section) =====
    with pl.section_cube():
        q_l1_db = pl.make_tile_group(
            type=pl.TileType(shape=[TD, TS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat,
                             layout=pl.ZN),
            addrs=MA0, mutex_ids=[0, 1])
        k_l1_db = pl.make_tile_group(
            type=pl.TileType(shape=[TKV, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat,
                             layout=pl.NZ),
            addrs=MA1, mutex_ids=[2, 3])
        v_l1_db = pl.make_tile_group(
            type=pl.TileType(shape=[TKV, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat,
                             layout=pl.NZ),
            addrs=MA3, mutex_ids=[4, 5])
        left_db = pl.make_tile_group(
            type=pl.TileType(shape=[TKV, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left,
                             layout=pl.NZ),
            addrs=LA0, mutex_ids=[6, 7])
        right_db = pl.make_tile_group(
            type=pl.TileType(shape=[TD, TS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
            addrs=RA0, mutex_ids=[8, 9])
        acc_db = pl.make_tile_group(
            type=pl.TileType(shape=[TKV, TS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
            addrs=CA0, mutex_ids=[10, 11, 12, 13])

    # ===== Vector-local buffer declarations (in section) =====
    with pl.section_vector():
        p_f16_db = pl.make_tile_group(
            type=pl.TileType(shape=[TKV // 2 + 1, TS_HALF * 2], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[VA2, VA2B], mutex_ids=[0, 1])
        p_f16_main_db = pl.make_tile_group(
            type=pl.TileType(shape=[TKV // 2 + 1, TS_HALF], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Vec,
                             valid_shape=[TKV // 2, TS_HALF], layout=pl.NZ),
            addrs=[VA2, VA2B], mutex_ids=[0, 1])
        p_f16_back_db = pl.make_tile_group(
            type=pl.TileType(shape=[TKV // 2 + 1, TS_HALF], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Vec,
                             valid_shape=[TKV // 2, TS_HALF], layout=pl.NZ),
            addrs=[VA2_DN, VA2B_DN], mutex_ids=[0, 1])
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
            type=pl.TileType(shape=[TS_HALF, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addrs=VA9, mutex_ids=[11])
        running_o_buf = pl.make_tile_group(
            type=pl.TileType(shape=[TS_HALF, TD], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=VA7, mutex_ids=[12])
        sub_id = pl.get_subblock_idx()
        row_off = sub_id * TS_HALF

    # ===== Main loop: interleaved cube/vector sections =====
    # NO sync here — framework auto-inserts cross-core sync (pipeline transform).
    tick = 0
    q_count = 0
    for qi in pl.range(core_id, sq_tiles, num_cores):
        sq_off = qi * TS
        with pl.section_cube():
            cur_q_slot = q_l1_db.next()
        for ki in pl.range(0, skv_tiles):
            with pl.section_cube():
                compute_qk(ki, sq_off, tick, q, k, cur_q_slot,
                           k_l1_db, left_db, right_db, acc_db, qk_vec_db)
            with pl.section_vector():
                compute_p(ki, tick, q_count, sub_id,
                          qk_vec_db, p_f16_db, p_f16_main_db, p_f16_back_db, p_mat_db,
                          global_max_rm_buf, global_sum_rm_buf, exp_corr_rm_fifo)
            with pl.section_cube():
                compute_pv(ki, tick, v, v_l1_db, p_mat_db,
                           left_db, right_db, acc_db, pv_vec_db)
            with pl.section_vector():
                compute_gu(ki, skv_tiles, tick, q_count, sq_off, row_off, o,
                           pv_vec_db, running_o_buf, o_f16_buf,
                           global_sum_rm_buf, exp_corr_rm_fifo)
            tick = tick + 1
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
def test_fa_perf():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(42)
    for sq, skv, d, num_cores in [
        (8192, 256, TD, 28),  # 950 默认流预算 28 cube 核
    ]:
        logging.info("\nFA-Perf DN (%s,%s,%s) cores=%s", sq, skv, d, num_cores)
        q_t = torch.rand((sq, d), device=device, dtype=torch.float16)
        k_t = torch.rand((skv, d), device=device, dtype=torch.float16)
        v_t = torch.rand((skv, d), device=device, dtype=torch.float16)
        o_t = torch.zeros((sq, d), device=device, dtype=torch.float16)
        qk_t = torch.zeros((sq * FIFO_SIZE, skv), device=device, dtype=torch.float32)
        p_t = torch.zeros((sq * FIFO_SIZE, skv), device=device, dtype=torch.float16)
        pv_t = torch.zeros((48 * PV_CORE_STRIDE, d), device=device, dtype=torch.float32)
        qk_ref, x_exp_ref, o_ref = flash_attention_ref(q_t, k_t, v_t, d)
        prev_diff = None
        for i in range(20):
            o_t.zero_()
            fa_perf_tkv_preload_dn_kernel[None, num_cores](q_t, k_t, v_t, o_t, qk_t, p_t, pv_t)
            torch.npu.synchronize()
            diff = (o_t - o_ref).abs().max().item()
            logging.info("  run %s: max|diff|=%.6f", i, diff)
            if prev_diff is not None and diff != prev_diff:
                raise AssertionError(f"Non-deterministic! run {i} diff={diff:.6f} vs prev={prev_diff:.6f}")
            prev_diff = diff
        torch.testing.assert_close(o_t, o_ref, rtol=5e-3, atol=5e-3)
        logging.info("  PASS (deterministic, max|diff|=%.6f)", prev_diff)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("FA perf DN: double-buffer + QK pre-compute (QK_PRELOAD=%s, FIFO=%s)", QK_PRELOAD, FIFO_SIZE)
    logging.info("%s", '=' * 60)
    test_fa_perf()
    logging.info("\nAll FlashAttention DN tests passed!")
