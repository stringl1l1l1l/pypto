#!/usr/bin/env python3
# coding: utf-8
# ruff: noqa: F821
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""FlashAttention performance kernel (DN mode) — *dynamic rank* variant with BSND inputs.

Kernel signature matches ops-transformer's FlashAttentionScore (23 params + workspace + tiling):
    query, key, value, real_shift, drop_mask, padding_mask, atten_mask,
    prefix, actual_seq_qlen, actual_seq_kvlen, q_start_idx, kv_start_idx,
    d_scale_q, d_scale_k, d_scale_v, query_rope, key_rope, sink, p_scale,
    softmax_max, softmax_sum, softmax_out, attention_out, workspace, tiling

All tensor params are declared as typed pointers (pl.Ptr[pl.DT_FP16] etc).
Unused optional params are declared but not accessed in the kernel body.
The actual shape/stride is supplied at runtime through a *tiling* dataclass:

    @dataclass
    class OpTiling:
        b: int
        sq: int
        n: int
        skv: int
        d: int

The kernel reconstructs typed tensor views via ``pl.make_tensor``.
Work distribution: total work ``B * N`` is flattened and distributed across cores
using ``pl.get_block_num()`` and ``pl.get_block_idx()``.

Attention mask: a fixed [FIXED_MASK_S, FIXED_MASK_S] UINT8 causal mask (1=mask out, 0=keep),
loaded into VEC UB and applied during softmax on diagonal KV tiles (ki==skv_tiles-1).
"""

from dataclasses import dataclass
import logging
import math
import os

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.tilingkey import TilingKeyField
import pytest
import torch

import pypto

logging.basicConfig(level=logging.INFO)
# ================================================================
#  Configuration — change QK_PRELOAD to tune pre-compute depth
# ================================================================
QK_PRELOAD = 2  # How many KV tiles to pre-compute QK ahead

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

# Cube tile byte sizes
Q_F16 = TS * TD * 2
KT_F16 = TKV * TD * 2
P_F16 = TS * TKV * 2

# ---- MAT (512KB) ----
MA0 = 0
MA1 = Q_F16 * 2
MA2 = MA1 + KT_F16 * 2
MA3 = MA2 + P_F16 * 3
LA0 = 0
RA0 = 0
CA0 = 0

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
VB_MASK = TKV * TS_HALF
VA_MASK = VA_EC2 + VB_RED
VA_END = VA_MASK + 2 * VB_MASK
assert VA_END <= 248 * 1024, f"VEC overflow: {VA_END} > {248 * 1024}"

FIXED_MASK_S = 2048


# Cross-core event IDs
QK_READY_FORWARD_IDS = (0, 1)
QK_READY_BARKWARD_IDS = (2, 3)
P_READY_FORWARD_IDS = (4, 5, 6)
PV_READY_FORWARD_IDS = (7, 8)
PV_READY_BARKWARD_IDS = (9, 10)


# ================================================================
#  Tiling data — describes the runtime shape of the (rank-erased) BSND inputs
# ================================================================


@dataclass
class OpTiling:
    b: int
    sq: int
    n: int
    skv: int
    d: int
    layout: int


# ================================================================
#  TilingKey — compile-time template specialization (matches ops-transformer)
# ================================================================
# Dict for JIT launch (concrete key)
_FA_TILING_KEY_LAUNCH = {
    "KernelTypeKey": 0,
    "ImplMode": 0,
    "Layout": 1,
    "S1TemplateType": 128,
    "S2TemplateType": 128,
    "DTemplateType": 128,
    "DvTemplateType": 128,
    "PseMode": 9,
    "HasAtten": 0,
    "HasDrop": 0,
    "HasRope": 0,
    "OutDtype": 0,
    "Regbase": 1,
    "OptionalDn": 0,
}


class FaTilingKey:
    KernelTypeKey = TilingKeyField(bits=2, values=[0, 1])
    ImplMode = TilingKeyField(bits=2, values=[0, 1, 2])
    Layout = TilingKeyField(bits=4, values=[0, 1, 2, 3, 4])
    S1TemplateType = TilingKeyField(bits=10, values=[0, 16, 64, 128, 256])
    S2TemplateType = TilingKeyField(bits=10, values=[0, 16, 32, 64, 128, 256, 512])
    DTemplateType = TilingKeyField(bits=12, values=[0, 16, 32, 48, 64, 80, 96, 128, 160, 192, 256, 768])
    DvTemplateType = TilingKeyField(bits=12, values=[0, 16, 32, 48, 64, 80, 96, 128, 160, 192, 256, 768])
    PseMode = TilingKeyField(bits=4, values=[0, 1, 2, 3, 4, 9])
    HasAtten = TilingKeyField(bits=1, values=[0, 1])
    HasDrop = TilingKeyField(bits=1, values=[0, 1])
    HasRope = TilingKeyField(bits=1, values=[0, 1])
    OutDtype = TilingKeyField(bits=2, values=[0, 1, 2])
    Regbase = TilingKeyField(bits=1, values=[0, 1])
    OptionalDn = TilingKeyField(bits=1, values=[0, 1])

    @classmethod
    def is_valid(cls, key):
        (
            kernel_type_key,
            impl_mode,
            layout,
            s1_template_type,
            s2_template_type,
            d_template_type,
            dv_template_type,
            pse_mode,
            has_atten,
            has_drop,
            has_rope,
            out_dtype,
            regbase,
            optional_dn,
        ) = key
        if kernel_type_key != 0:
            return False
        if impl_mode != 0:
            return False
        if layout != 1:
            return False
        if s1_template_type != 128:
            return False
        if s2_template_type != 128:
            return False
        if d_template_type != 128:
            return False
        if dv_template_type != 128:
            return False
        if pse_mode != 9:
            return False
        if has_drop != 0:
            return False
        if has_rope != 0:
            return False
        if out_dtype != 0:
            return False
        if regbase != 1:
            return False
        if optional_dn != 0:
            return False
        return True


# ================================================================
#  VF softmax functions (merged no-update / update with is_update flag)
# ================================================================


@pl.vector_function
def process_vec1_dn_vf(input_tile, x_exp_tile, max_tile, exp_max_tile, sum_tile, is_update, mask_tile, need_mask):
    """Softmax DN VF kernel — merged no-update / update / causal-mask versions."""
    preg_108 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_136 = vf.update_mask(128, dtype=io_dtype)
    preg_compare0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    preg_compare1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    preg_compare2 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    preg_compare3 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    src_ub0 = input_tile
    src_ub1 = input_tile + TS_HALF
    src_ub2 = input_tile + TS_HALF * 2
    src_ub3 = input_tile + TS_HALF * 3
    x_exp_1 = x_exp_tile + TKV * 4
    max0 = vf.full(minValue)
    max1 = vf.full(minValue)
    max2 = vf.full(minValue)
    max3 = vf.full(minValue)
    vreg_min = vf.full(minValue)
    for iter_m in pl.range(0, TKV // 4):
        src0 = vf.load_align(src_ub0, iter_m * TS_HALF * 4)
        src1 = vf.load_align(src_ub1, iter_m * TS_HALF * 4)
        src2 = vf.load_align(src_ub2, iter_m * TS_HALF * 4)
        src3 = vf.load_align(src_ub3, iter_m * TS_HALF * 4)
        if need_mask:
            preg_compare0 = vf.load_align(mask_tile, iter_m * TS_HALF * 4, dist=pl.LoadDist.DS)
            preg_compare1 = vf.load_align(mask_tile, TS_HALF + iter_m * TS_HALF * 4, dist=pl.LoadDist.DS)
            preg_compare2 = vf.load_align(mask_tile, TS_HALF * 2 + iter_m * TS_HALF * 4, dist=pl.LoadDist.DS)
            preg_compare3 = vf.load_align(mask_tile, TS_HALF * 3 + iter_m * TS_HALF * 4, dist=pl.LoadDist.DS)
            src0 = vf.select(vreg_min, src0, preg_compare0)
            src1 = vf.select(vreg_min, src1, preg_compare1)
            src2 = vf.select(vreg_min, src2, preg_compare2)
            src3 = vf.select(vreg_min, src3, preg_compare3)
            vf.store_align(input_tile + (iter_m * TS_HALF * 4), src0, preg_108)
            vf.store_align(input_tile + (TS_HALF + iter_m * TS_HALF * 4), src1, preg_108)
            vf.store_align(input_tile + (TS_HALF * 2 + iter_m * TS_HALF * 4), src2, preg_108)
            vf.store_align(input_tile + (TS_HALF * 3 + iter_m * TS_HALF * 4), src3, preg_108)
        max0 = vf.max(max0, src0, preg_108)
        max1 = vf.max(max1, src1, preg_108)
        max2 = vf.max(max2, src2, preg_108)
        max3 = vf.max(max3, src3, preg_108)
    if is_update:
        vreg_x_max_f32_b = vf.load_align(max_tile, 0)
    max0 = vf.max(max0, max2, preg_108)
    max1 = vf.max(max1, max3, preg_108)
    max0 = vf.max(max0, max1, preg_108)
    max0 = vf.muls(max0, SCALE, preg_108)
    if is_update:
        max0 = vf.max(max0, vreg_x_max_f32_b, preg_108)
        vreg_x_max_f32_b = vf.exp_sub(vreg_x_max_f32_b, max0, preg_108)
    vf.store_align(max_tile, max0, preg_108)
    if is_update:
        vf.store_align(exp_max_tile, vreg_x_max_f32_b, preg_108)
    vreg_x_sum_0 = vf.full(0.0, preg_108)
    vreg_x_sum_1 = vf.full(0.0, preg_108)
    vreg_x_sum_2 = vf.full(0.0, preg_108)
    vreg_x_sum_3 = vf.full(0.0, preg_108)
    if need_mask:
        vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
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
        vreg_x_exp_even_dtype = vf.astype(vreg_x_exp_0, preg_108, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        vreg_x_exp_odd_dtype = vf.astype(vreg_x_exp_2, preg_108, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        vreg_x_exp_dtype_pack, vreg_x_exp_dtype_packa = vf.de_interleave(vreg_x_exp_even_dtype, vreg_x_exp_odd_dtype)
        vreg_x_exp_even_dtype_1 = vf.astype(vreg_x_exp_1, preg_108, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        vreg_x_exp_odd_dtype_1 = vf.astype(vreg_x_exp_3, preg_108, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        vreg_x_exp_dtype_1_pack, vreg_x_exp_dtype_1_packa = vf.de_interleave(
            vreg_x_exp_even_dtype_1, vreg_x_exp_odd_dtype_1
        )
        vf.store_align(
            x_exp_tile,
            vreg_x_exp_dtype_pack,
            preg_136,
            block_stride=block_stride_dn,
            repeat_stride=REPEAT_STRIDE_DN,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            post_update=True,
        )
        vreg_x_sum_0 = vf.add(vreg_x_exp_0, vreg_x_sum_0, preg_108)
        vreg_x_sum_2 = vf.add(vreg_x_exp_2, vreg_x_sum_2, preg_108)
        vf.store_align(
            x_exp_1,
            vreg_x_exp_dtype_1_pack,
            preg_136,
            block_stride=block_stride_dn,
            repeat_stride=REPEAT_STRIDE_DN,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            post_update=True,
        )
        vreg_x_sum_1 = vf.add(vreg_x_exp_1, vreg_x_sum_1, preg_108)
        vreg_x_sum_3 = vf.add(vreg_x_exp_3, vreg_x_sum_3, preg_108)
    vreg_x_sum0 = vf.add(vreg_x_sum_2, vreg_x_sum_0, preg_108)
    vreg_x_sum1 = vf.add(vreg_x_sum_3, vreg_x_sum_1, preg_108)
    vreg_x_sum0 = vf.add(vreg_x_sum0, vreg_x_sum1, preg_108)
    if is_update:
        vreg_l0 = vf.load_align(sum_tile, 0)
        vreg_l0 = vf.mul(vreg_x_max_f32_b, vreg_l0, preg_108)
        vreg_l0 = vf.add(vreg_l0, vreg_x_sum0, preg_108)
        vf.store_align(sum_tile, vreg_l0, preg_108)
    else:
        vf.store_align(sum_tile, vreg_x_sum0, preg_108)


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


def compute_qk(tile_dims, sq_off, sk_off, ki, cur_q_slot, q, k, k_l1_db, left_db, right_db, acc_db, qk_vec_db, task_id):
    if ki == 0:
        pl.load_tile(cur_q_slot, q, sq_off, order=[3, 1])
    cur_k_slot = k_l1_db.next()
    pl.load_tile(cur_k_slot, k, sk_off, order=tile_dims)
    qk_left = left_db.next()
    qk_right = right_db.next()
    qk_acc = acc_db.next()
    pl.move(qk_left, cur_k_slot)
    pl.move(qk_right, cur_q_slot)
    pl.matmul(qk_acc, qk_left, qk_right)
    qk_dst = qk_vec_db.next()
    pl.system.wait_cross_core(
        pipe=pl.PipeType.FIX,
        event_id=QK_READY_BARKWARD_IDS[task_id % 2],
    )
    pl.move(qk_dst, qk_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN)
    pl.system.set_cross_core(
        pipe=pl.PipeType.FIX,
        event_id=QK_READY_FORWARD_IDS[task_id % 2],
    )


def compute_pv(b_idx, n_idx, ctx, v_l1_db, p_mat_db, left_db, right_db, acc_db, pv_vec_db, v):
    pl.system.wait_cross_core(
        pipe=pl.PipeType.MTE1,
        event_id=P_READY_FORWARD_IDS[ctx.task_id_mod3],
    )
    cur_v_slot = v_l1_db.next()
    cur_p_slot = p_mat_db.next()
    pv_left = left_db.next()
    pv_right = right_db.next()
    pv_acc = acc_db.next()
    pl.load_tile(cur_v_slot, v, [b_idx, ctx.ki, n_idx, 0], order=[1, 3])
    pl.move(pv_left, cur_p_slot)
    pl.move(pv_right, cur_v_slot)
    pl.matmul(pv_acc, pv_left, pv_right)
    pv_dst = pv_vec_db.next()
    pl.system.wait_cross_core(
        pipe=pl.PipeType.FIX,
        event_id=PV_READY_BARKWARD_IDS[ctx.task_id_mod2],
    )
    pl.move(pv_dst, pv_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    pl.system.set_cross_core(
        pipe=pl.PipeType.FIX,
        event_id=PV_READY_FORWARD_IDS[ctx.task_id_mod2],
    )


def compute_p(
    p_ctx,
    global_max_rm_buf,
    global_sum_rm_buf,
    exp_corr_rm_fifo,
    qk_vec_db,
    p_dtype_db,
    p_dtype_main_db,
    p_dtype_back_db,
    p_mat_db,
    sub_id,
    attn_mask,
    mask_db,
    need_mask,
):
    pl.system.wait_cross_core(
        pipe=pl.PipeType.V,
        event_id=QK_READY_FORWARD_IDS[p_ctx.task_id_mod2],
    )
    global_max_rm_cur = global_max_rm_buf[p_ctx.q_count_mod3]
    global_sum_rm_cur = global_sum_rm_buf[p_ctx.q_count_mod3]
    exp_corr_rm_cur = exp_corr_rm_fifo[p_ctx.task_id_mod3]
    qk_slot = qk_vec_db.next()
    p_dtype_slot = p_dtype_db.next()
    p_dtype_main_slot = p_dtype_main_db.next()
    p_dtype_back_slot = p_dtype_back_db.next()
    p_mat_slot = p_mat_db.next()
    if need_mask:
        if p_ctx.ki == p_ctx.skv_tiles - 1:
            mask_slot = mask_db.next()
            pl.load(mask_slot, attn_mask, [0, sub_id * TS_HALF])
            if p_ctx.ki == 0:
                process_vec1_dn_vf(
                    qk_slot, p_dtype_slot, global_max_rm_cur, exp_corr_rm_cur, global_sum_rm_cur, 0, mask_slot, 1
                )
            else:
                process_vec1_dn_vf(
                    qk_slot, p_dtype_slot, global_max_rm_cur, exp_corr_rm_cur, global_sum_rm_cur, 1, mask_slot, 1
                )
        else:
            if p_ctx.ki == 0:
                process_vec1_dn_vf(
                    qk_slot, p_dtype_slot, global_max_rm_cur, exp_corr_rm_cur, global_sum_rm_cur, 0, qk_slot, 0
                )
            else:
                process_vec1_dn_vf(
                    qk_slot, p_dtype_slot, global_max_rm_cur, exp_corr_rm_cur, global_sum_rm_cur, 1, qk_slot, 0
                )
    else:
        if p_ctx.ki == 0:
            process_vec1_dn_vf(
                qk_slot, p_dtype_slot, global_max_rm_cur, exp_corr_rm_cur, global_sum_rm_cur, 0, qk_slot, 0
            )
        else:
            process_vec1_dn_vf(
                qk_slot, p_dtype_slot, global_max_rm_cur, exp_corr_rm_cur, global_sum_rm_cur, 1, qk_slot, 0
            )
    pl.system.set_cross_core(
        pipe=pl.PipeType.V,
        event_id=QK_READY_BARKWARD_IDS[p_ctx.task_id_mod2],
    )
    pl.insert(p_mat_slot, p_dtype_main_slot, [0, TS_HALF * sub_id])
    pl.insert(p_mat_slot, p_dtype_back_slot, [TKV // 2, TS_HALF * sub_id])
    pl.system.set_cross_core(
        pipe=pl.PipeType.MTE3,
        event_id=P_READY_FORWARD_IDS[p_ctx.task_id_mod3],
    )


def compute_gu(
    b_idx, n_idx, g_ctx, sub_id, pv_vec_db, global_sum_rm_buf, exp_corr_rm_fifo, running_o_buf, o_dtype_buf, o
):
    pv_slot = pv_vec_db.next()
    gsum_gu = global_sum_rm_buf[g_ctx.q_count_mod3]
    exp_corr_gu = exp_corr_rm_fifo[g_ctx.task_id_mod3]
    pl.system.wait_cross_core(
        pipe=pl.PipeType.V,
        event_id=PV_READY_FORWARD_IDS[g_ctx.task_id_mod2],
    )
    if g_ctx.ki == 0:
        pl.move(running_o_buf, pv_slot)
    else:
        if g_ctx.ki < g_ctx.skv_tiles - 1:
            flash_update_basic_vf(running_o_buf, pv_slot, running_o_buf, exp_corr_gu)
        else:
            flash_update_last_basic_vf(
                running_o_buf,
                pv_slot,
                running_o_buf,
                exp_corr_gu,
                gsum_gu,
            )
            pl.cast(
                o_dtype_buf,
                running_o_buf,
                mode=pl.RoundMode.CAST_ROUND,
            )
            pl.store_tile(o, o_dtype_buf, [b_idx, g_ctx.qi * 2 + sub_id, n_idx, 0], order=[1, 3])
    pl.system.set_cross_core(
        pipe=pl.PipeType.V,
        event_id=PV_READY_BARKWARD_IDS[g_ctx.task_id_mod2],
    )
    if g_ctx.ki == g_ctx.skv_tiles - 1:
        if g_ctx.ki == 0:
            last_div_vf(running_o_buf, running_o_buf, gsum_gu)
            pl.cast(
                o_dtype_buf,
                running_o_buf,
                mode=pl.RoundMode.CAST_ROUND,
            )
            pl.store_tile(o, o_dtype_buf, [b_idx, g_ctx.qi * 2 + sub_id, n_idx, 0], order=[1, 3])


# ================================================================
#  Kernel — dynamic rank: inputs are raw pointers, shapes come from tiling
# ================================================================


@pl.jit(
    auto_mutex=True,
    tiling_key=FaTilingKey,
    datatype={
        "query": "io_dtype",
        "key": "io_dtype",
        "value": "io_dtype",
        "attention_out": "io_dtype",
    },
    compile_timeout=300,
)
def flash_attention_score(
    query: pl.Ptr[pl.DT_UINT8],
    key: pl.Ptr[pl.DT_UINT8],
    value: pl.Ptr[pl.DT_UINT8],
    real_shift: pl.Ptr[pl.DT_UINT8],
    drop_mask: pl.Ptr[pl.DT_UINT8],
    padding_mask: pl.Ptr[pl.DT_UINT8],
    atten_mask: pl.Ptr[pl.DT_UINT8],
    prefix: pl.Ptr[pl.DT_UINT8],
    actual_seq_qlen: pl.Ptr[pl.DT_UINT8],
    actual_seq_kvlen: pl.Ptr[pl.DT_UINT8],
    q_start_idx: pl.Ptr[pl.DT_UINT8],
    kv_start_idx: pl.Ptr[pl.DT_UINT8],
    d_scale_q: pl.Ptr[pl.DT_UINT8],
    d_scale_k: pl.Ptr[pl.DT_UINT8],
    d_scale_v: pl.Ptr[pl.DT_UINT8],
    query_rope: pl.Ptr[pl.DT_UINT8],
    key_rope: pl.Ptr[pl.DT_UINT8],
    sink: pl.Ptr[pl.DT_UINT8],
    p_scale: pl.Ptr[pl.DT_UINT8],
    softmax_max: pl.Ptr[pl.DT_UINT8],
    softmax_sum: pl.Ptr[pl.DT_UINT8],
    softmax_out: pl.Ptr[pl.DT_UINT8],
    attention_out: pl.Ptr[pl.DT_UINT8],
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: OpTiling,
):

    tensor_q = pl.make_tensor(
        query,
        [tiling.b, tiling.sq, tiling.n, tiling.d],
        dtype=io_dtype,
    )
    tensor_k = pl.make_tensor(
        key,
        [tiling.b, tiling.skv, tiling.n, tiling.d],
        dtype=io_dtype,
    )
    tensor_v = pl.make_tensor(
        value,
        [tiling.b, tiling.skv, tiling.n, tiling.d],
        dtype=io_dtype,
    )
    tensor_o = pl.make_tensor(
        attention_out,
        [tiling.b, tiling.sq, tiling.n, tiling.d],
        dtype=io_dtype,
    )
    tensor_attn_mask = pl.make_tensor(atten_mask, [FIXED_MASK_S, FIXED_MASK_S])

    n_dim = tiling.n
    sq_dim = tiling.sq
    skv_dim = tiling.skv
    sq_tiles = (sq_dim + (TS - 1)) // TS
    skv_tiles = (skv_dim + (TKV - 1)) // TKV
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    num_cores = pl.get_block_num()
    total_work = tiling.b * tiling.n
    work_per_core = (total_work + num_cores - 1) // num_cores
    work_start = core_id * work_per_core
    work_end = work_start + work_per_core
    if work_end > total_work:
        work_end = total_work

    # ===== Cross-core shared buffers =====
    qk_vec_db = pl.make_tile_group(
        type=pl.TileType(shape=[TKV, TS_HALF], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=VA0,
        mutex_ids=[14, 15],
    )
    pv_vec_db = pl.make_tile_group(
        type=pl.TileType(shape=[TS_HALF, TD], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=VA8,
        mutex_ids=[16, 17],
    )
    p_mat_db = pl.make_tile_group(
        type=pl.TileType(shape=[TS, TKV], dtype=io_dtype, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=MA2,
        mutex_ids=[18, 19, 20],
    )

    # =================== CUBE SECTION ===================
    with pl.section_cube():
        q_l1_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TD, TS],
                dtype=io_dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.ZN,
            ),
            addrs=MA0,
            mutex_ids=[0, 1],
        )
        k_l1_db = pl.make_tile_group(
            type=pl.TileType(shape=[TKV, TD], dtype=io_dtype, target_memory=pl.MemorySpace.Mat),
            addrs=MA1,
            mutex_ids=[2, 3],
        )
        v_l1_db = pl.make_tile_group(
            type=pl.TileType(shape=[TKV, TD], dtype=io_dtype, target_memory=pl.MemorySpace.Mat),
            addrs=MA3,
            mutex_ids=[4, 5],
        )
        left_db = pl.make_tile_group(
            type=pl.TileType(shape=[TKV, TD], dtype=io_dtype, target_memory=pl.MemorySpace.Left),
            addrs=LA0,
            mutex_ids=[6, 7],
        )
        right_db = pl.make_tile_group(
            type=pl.TileType(shape=[TD, TS], dtype=io_dtype, target_memory=pl.MemorySpace.Right),
            addrs=RA0,
            mutex_ids=[8, 9],
        )
        acc_db = pl.make_tile_group(
            type=pl.TileType(shape=[TKV, TS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
            addrs=CA0,
            mutex_ids=[10, 11, 12, 13],
        )

        task_id = 0
        qkv_tile_dims = [1, 3]
        ctx_arr = pl.struct_array(4, "CubeCtx", b_idx=0, n_idx=0, ki=0, task_id_mod2=0, task_id_mod3=0)
        for work_id in pl.range(work_start, work_end):
            b_idx = work_id // n_dim
            n_idx = work_id % n_dim
            for qi in pl.range(0, sq_tiles):
                cur_q_slot = q_l1_db.next()
                causal_skv = skv_tiles
                if HasAtten == 1:  # noqa: F821
                    causal_skv = qi + 1
                ki_end = causal_skv
                if work_id + 1 >= work_end:
                    if qi + 1 >= sq_tiles:
                        ki_end = causal_skv + QK_PRELOAD
                sq_off = [b_idx, qi, n_idx, 0]
                for ki in pl.range(0, ki_end):
                    if ki < causal_skv:
                        ctx_curr = ctx_arr[task_id % 4]
                        ctx_curr.b_idx = b_idx
                        ctx_curr.n_idx = n_idx
                        ctx_curr.ki = ki
                        ctx_curr.task_id_mod2 = task_id % 2
                        ctx_curr.task_id_mod3 = task_id % 3
                        sk_off = [b_idx, ki, n_idx, 0]
                        compute_qk(
                            qkv_tile_dims,
                            sq_off,
                            sk_off,
                            ki,
                            cur_q_slot,
                            tensor_q,
                            tensor_k,
                            k_l1_db,
                            left_db,
                            right_db,
                            acc_db,
                            qk_vec_db,
                            task_id,
                        )

                    if task_id > 1:
                        prev = ctx_arr[(task_id + 2) % 4]
                        compute_pv(
                            prev.b_idx,
                            prev.n_idx,
                            prev,
                            v_l1_db,
                            p_mat_db,
                            left_db,
                            right_db,
                            acc_db,
                            pv_vec_db,
                            tensor_v,
                        )
                    task_id = task_id + 1

    # =================== VECTOR SECTION ===================
    with pl.section_vector():
        p_dtype_db = pl.make_tile_group(
            type=pl.TileType(shape=[TKV // 2 + 1, TS_HALF * 2], dtype=io_dtype, target_memory=pl.MemorySpace.Vec),
            addrs=[VA2, VA2B],
            mutex_ids=[0, 1],
        )
        p_dtype_main_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TKV // 2 + 1, TS_HALF],
                dtype=io_dtype,
                target_memory=pl.MemorySpace.Vec,
                valid_shape=[TKV // 2, TS_HALF],
                layout=pl.NZ,
            ),
            addrs=[VA2, VA2B],
            mutex_ids=[0, 1],
        )
        p_dtype_back_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TKV // 2 + 1, TS_HALF],
                dtype=io_dtype,
                target_memory=pl.MemorySpace.Vec,
                valid_shape=[TKV // 2, TS_HALF],
                layout=pl.NZ,
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
        o_dtype_g = pl.make_tile_group(
            type=pl.TileType(shape=[TS_HALF, TD], dtype=io_dtype, target_memory=pl.MemorySpace.Vec),
            addrs=VA9,
            mutex_ids=[11],
        )
        o_dtype_buf = o_dtype_g.next()
        running_o_g = pl.make_tile_group(
            type=pl.TileType(shape=[TS_HALF, TD], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=VA7,
            mutex_ids=[12],
        )
        running_o_buf = running_o_g.next()
        mask_db = pl.make_tile_group(
            type=pl.TileType(shape=[TKV, TS_HALF], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec),
            addrs=VA_MASK,
            mutex_ids=[8, 9],
        )

        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_BARKWARD_IDS[0])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_BARKWARD_IDS[1])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_BARKWARD_IDS[0])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_BARKWARD_IDS[1])

        sub_id = pl.get_subblock_idx()
        task_id = 0
        q_count = 0
        ctx_arr = pl.struct_array(
            4, "VecCtx", b_idx=0, n_idx=0, qi=0, ki=0, skv_tiles=0, task_id_mod2=0, task_id_mod3=0, q_count_mod3=0
        )
        for work_id in pl.range(work_start, work_end):
            b_idx = work_id // n_dim
            n_idx = work_id % n_dim
            for qi in pl.range(0, sq_tiles):
                causal_skv = skv_tiles
                if HasAtten == 1:  # noqa: F821
                    causal_skv = qi + 1
                ki_end = causal_skv
                if work_id + 1 >= work_end:
                    if qi + 1 >= sq_tiles:
                        ki_end = causal_skv + 3
                for ki in pl.range(0, ki_end):
                    if ki < causal_skv:
                        ctx_curr = ctx_arr[task_id % 4]
                        ctx_curr.b_idx = b_idx
                        ctx_curr.n_idx = n_idx
                        ctx_curr.qi = qi
                        ctx_curr.ki = ki
                        ctx_curr.skv_tiles = causal_skv
                        ctx_curr.task_id_mod2 = task_id % 2
                        ctx_curr.task_id_mod3 = task_id % 3
                        ctx_curr.q_count_mod3 = q_count % 3
                    if task_id > 0:
                        if ki < causal_skv + 1:
                            p_ctx = ctx_arr[(task_id + 3) % 4]
                            compute_p(
                                p_ctx,
                                global_max_rm_buf,
                                global_sum_rm_buf,
                                exp_corr_rm_fifo,
                                qk_vec_db,
                                p_dtype_db,
                                p_dtype_main_db,
                                p_dtype_back_db,
                                p_mat_db,
                                sub_id,
                                tensor_attn_mask,
                                mask_db,
                                HasAtten,
                            )  # noqa: F821
                    if task_id > 2:
                        g_ctx = ctx_arr[(task_id + 1) % 4]
                        compute_gu(
                            g_ctx.b_idx,
                            g_ctx.n_idx,
                            g_ctx,
                            sub_id,
                            pv_vec_db,
                            global_sum_rm_buf,
                            exp_corr_rm_fifo,
                            running_o_buf,
                            o_dtype_buf,
                            tensor_o,
                        )
                    task_id = task_id + 1
                q_count = q_count + 1


# ================================================================
#  Reference + Tests
# ================================================================


def flash_attention_causal_ref_bs(q, k, v, d):
    scale_val = 1.0 / math.sqrt(d)
    b, sq, n, _ = q.shape
    _, skv, _, _ = k.shape
    o_ref = torch.zeros_like(q)
    causal_mask = torch.triu(torch.ones(sq, skv, dtype=torch.bool, device=q.device), diagonal=1)
    for bi in range(b):
        for ni in range(n):
            qk = torch.matmul(q[bi, :, ni, :].float(), k[bi, :, ni, :].float().T) * scale_val
            qk = qk.masked_fill(causal_mask, float("-inf"))
            attn = torch.softmax(qk, dim=-1)
            o_ref[bi, :, ni, :] = torch.matmul(attn, v[bi, :, ni, :].float()).to(q.dtype)
    return o_ref


def flash_attention_full_ref_bs(q, k, v, d):
    scale_val = 1.0 / math.sqrt(d)
    b, sq, n, _ = q.shape
    o_ref = torch.zeros_like(q)
    for bi in range(b):
        for ni in range(n):
            qk = torch.matmul(q[bi, :, ni, :].float(), k[bi, :, ni, :].float().T) * scale_val
            attn = torch.softmax(qk, dim=-1)
            o_ref[bi, :, ni, :] = torch.matmul(attn, v[bi, :, ni, :].float()).to(q.dtype)
    return o_ref


def make_causal_mask_dn_fixed_u8(device):
    return torch.tril(torch.ones((FIXED_MASK_S, FIXED_MASK_S), dtype=torch.uint8, device=device), diagonal=-1)


@pytest.mark.soc("950")
@pytest.mark.parametrize(
    ("torch_dtype", "pl_dtype", "rtol", "atol"),
    [
        (torch.float16, pl.DT_FP16, 5e-3, 5e-3),
        (torch.bfloat16, pl.DT_BF16, 1e-2, 1e-2),
    ],
)
@pypto.options(pass_options={"enable_slice": False})
def test_fa_perf(torch_dtype, pl_dtype, rtol, atol):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"
    torch.manual_seed(42)
    attn_mask = make_causal_mask_dn_fixed_u8(device)
    b, sq, n, skv, d, num_cores = 8, 4096, 8, 4096, TD, 28  # 950 默认流预算 28 cube 核
    logging.info(
        "\nFA-Perf BSND DN dynrank tilingkey dtype=%s (b=%s,sq=%s,n=%s,skv=%s,d=%s) cores=%s QK_PRELOAD=%s",
        torch_dtype,
        b,
        sq,
        n,
        skv,
        d,
        num_cores,
        QK_PRELOAD,
    )
    q_t = torch.rand((b, sq, n, d), device=device, dtype=torch_dtype)
    k_t = torch.rand((b, skv, n, d), device=device, dtype=torch_dtype)
    v_t = torch.rand((b, skv, n, d), device=device, dtype=torch_dtype)
    total_work = b * n
    actual_num_cores = min(num_cores, total_work)
    tiling = OpTiling(b=b, sq=sq, n=n, skv=skv, d=d, layout=1)
    softmax_max = torch.zeros((b, n, sq), device=device, dtype=torch.float32)
    softmax_sum = torch.zeros((b, n, sq), device=device, dtype=torch.float32)
    softmax_out = torch.zeros((b, sq, n, skv), device=device, dtype=torch_dtype)
    datatype = {"query": pl_dtype, "key": pl_dtype, "value": pl_dtype, "attention_out": pl_dtype}

    o_causal = torch.zeros((b, sq, n, d), device=device, dtype=torch_dtype)
    tk_causal = {**_FA_TILING_KEY_LAUNCH, "HasAtten": 1}
    flash_attention_score[None, actual_num_cores, tk_causal, datatype](
        q_t,
        k_t,
        v_t,
        None,
        None,
        None,  # real_shift, drop_mask, padding_mask
        attn_mask,  # atten_mask
        None,
        None,
        None,  # prefix, actual_seq_qlen, actual_seq_kvlen
        None,
        None,  # q_start_idx, kv_start_idx
        None,
        None,
        None,  # d_scale_q, d_scale_k, d_scale_v
        None,
        None,  # query_rope, key_rope
        None,
        None,  # sink, p_scale
        softmax_max,
        softmax_sum,
        softmax_out,  # softmax outputs
        o_causal,  # attention_out
        None,  # workspace
        tiling,
    )
    torch.npu.synchronize()
    o_full = torch.zeros((b, sq, n, d), device=device, dtype=torch_dtype)
    tk_full = {**_FA_TILING_KEY_LAUNCH, "HasAtten": 0}
    flash_attention_score[None, actual_num_cores, tk_full, datatype](
        q_t,
        k_t,
        v_t,
        None,
        None,
        None,
        attn_mask,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        softmax_max,
        softmax_sum,
        softmax_out,
        o_full,
        None,  # workspace
        tiling,
    )
    torch.npu.synchronize()
    # Reach both tiling keys before output checks stop precompile discovery.
    o_causal_ref = flash_attention_causal_ref_bs(q_t, k_t, v_t, d)
    torch.testing.assert_close(o_causal, o_causal_ref, rtol=rtol, atol=atol)
    logging.info("HasAtten=1 (causal+mask) PASS: max|diff|=%.6f", (o_causal - o_causal_ref).abs().max().item())

    o_full_ref = flash_attention_full_ref_bs(q_t, k_t, v_t, d)
    torch.testing.assert_close(o_full, o_full_ref, rtol=rtol, atol=atol)
    logging.info("HasAtten=0 (full) PASS: max|diff|=%.6f", (o_full - o_full_ref).abs().max().item())
