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

"""FlashAttention performance kernel using NBuffer + auto_mutex.

Refactored from test_fa_perf_tkv_preload.py to use:
  - NBuffer with current() auto-rotate cursor (no manual buf_idx)
  - auto_mutex=True for automatic Mutex synchronization
  - Cross-core event_id synchronization preserved (QK_READY, P_READY, PV_READY)

Features:
  1. Multi-core: each Cube core processes multiple Q tiles via strided loop
  2. Double buffer: NBuffer with current() auto-rotate for Q/K/V/P/L0A/L0B/L0C
  3. FIFO cross-core GM buffers with configurable depth
  4. Cross-core event ID ping/pong
  5. QK pre-compute: Cube runs QK_PRELOAD tiles ahead
  6. Vector: FIFO exp_corr, double-buffered global_max/global_sum

Usage:
    python3 python/tests/st/pypto_pro/frontend/fa/test_flex_attention.py
"""

import itertools
import logging
import math
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

# ================================================================
#  Configuration
# ================================================================
QK_PRELOAD = 3
FIFO_SIZE = QK_PRELOAD + 1

# ================================================================
#  Tile dimensions and constants
# ================================================================
TS = 128
TKV = 128
TD = 128
TS_HALF = TS // 2
NEG_INF = -1e9
SCALE = 1.0 / math.sqrt(TD)

# Per query row up to MAX_HOLES invisible bands (holes) may be masked out, in
# addition to the causal right bound. Holes are arbitrary-position and
# non-overlapping; unused hole slots carry holes=0 (a no-op in decode_mask).
MAX_HOLES = 3

BLOCK_STRIDE_ND = TS >> 1 | 0x1
REPEAT_STRIDE_ND = 1
FLOAT_REP_SIZE = 64  # elements per fp32 register
D_LOOPS = TD // FLOAT_REP_SIZE
TAIL_D = TD % FLOAT_REP_SIZE
REDUCE_SIZE = 1

# Buffer sizes (bytes)
Q_F16 = TS * TD * 2
KT_F16 = TD * TKV * 2
V_F16 = TKV * TD * 2
P_F16 = TS * TKV * 2
QK_HALF_F32 = TS * TKV * 4
PV_HALF_F32 = TS * TD * 4

# VEC buffer sizes
VB4_KV = TS_HALF * TKV * 4
VB2_KV = TS_HALF * TKV * 2
VB4 = TS_HALF * TD * 4
VB2 = TS_HALF * TD * 2
VB6 = (TS_HALF + 1) * TKV * 2
VB_RED = TS_HALF * 1 * 4
VB_MASK = TS_HALF * TKV

# ================================================================
#  Buffer addresses
# ================================================================
# MAT (512KB) - L1 buffers
MA0_Q = 0
MA1_K = Q_F16 * 2
MA2_P = MA1_K + KT_F16 * 2
MA3_V = MA2_P + P_F16 * 3

# L0A/L0B/L0C addresses
LA0 = 0
LA1 = P_F16
RA0 = 0
RA1 = KT_F16
CA0 = 0
CA1 = QK_HALF_F32

# VEC (248KB) addresses
VA0 = 0
VA1 = VA0 + VB4_KV * 2
VA_GMAX0 = VA1 + VB6 * 2
VA_GMAX1 = VA_GMAX0 + VB_RED
VA_GMAX2 = VA_GMAX1 + VB_RED
VA_GSUM0 = VA_GMAX2 + VB_RED
VA_GSUM1 = VA_GSUM0 + VB_RED
VA_GSUM2 = VA_GSUM1 + VB_RED
VA_EXPMAX0 = VA_GSUM2 + VB_RED
VA_EXPMAX1 = VA_EXPMAX0 + VB_RED
VA_EXPMAX2 = VA_EXPMAX1 + VB_RED
VA7 = VA_EXPMAX2 + VB_RED
VA8 = VA7 + VB4
VA9 = VA8 + VB4 * 2
VA10 = VA9 + VB2
VA11 = VA10 + 8192 * 2
VA12 = VA11 + VB_RED
VA13 = VA12 + VB_RED
VA14 = VA13 + VB_RED * 2
# holel/holes double buffers hold MAX_HOLES packed hole channels each.
VA15 = VA14 + VB_RED * 2 * MAX_HOLES
VA16 = VA15 + VB_RED * 2 * MAX_HOLES
assert VA16 <= 248 * 1024

# Cross-core shared buffer IDs
QK_READY_FORWARD_IDS = (0, 1)
QK_READY_BARKWARD_IDS = (2, 3)
P_READY_FORWARD_IDS = (4, 5, 6)
PV_READY_FORWARD_IDS = (7, 8)
PV_READY_BARKWARD_IDS = (9, 10)


@pl.vector_function
def process_vec1_nd_no_update_vf_unalign64(
    input_tile, dst_tile, max_tile, max_tile_st, sum_tile, mask_tile, s1_size, s2_size
):
    preg_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    ureg_max = vf.unalign_reg_for_store()
    ureg_exp_sum = vf.unalign_reg_for_store()

    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(s2_size, dtype=pl.DT_FP32)
    preg_all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)

    vreg_min = vf.full(NEG_INF)

    for m in pl.range(s1_size):
        vreg_x = vf.load_align(input_tile, m * TKV)
        vreg_x = vf.muls(vreg_x, SCALE, preg_tail)

        preg_mask = vf.load_align(mask_tile, m * TKV, dist=pl.LoadDist.DS)
        vreg_x = vf.select(vreg_min, vreg_x, preg_mask)
        vf.store_align(input_tile + m * TKV, vreg_x, preg_all)
        vreg_max = vf.reduce_max(vreg_x, preg_tail, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(max_tile, vreg_max, ureg_max, 1, post_update=True)
    vf.store_unalign_post(max_tile, ureg_max, 0, post_update=True)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)

    for i in pl.range(s1_size):
        vreg_max_2 = vf.load_align(max_tile_st, i, dist=pl.LoadDist.BRC_B32)
        vreg_x_2 = vf.load_align(input_tile, i * TKV)
        vreg_exp_even = vf.exp_sub(vreg_x_2, vreg_max_2, preg_tail)

        vreg_exp_sum = vf.reduce_sum(vreg_exp_even, preg_tail, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(sum_tile, vreg_exp_sum, ureg_exp_sum, 1, post_update=True)

        vreg_exp_even_f16 = vf.astype(vreg_exp_even, preg_all_f16, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP16)
        vreg_dst_even_f16, vreg_dst_odd_f16 = vf.de_interleave(vreg_exp_even_f16, vreg_exp_even_f16)
        vf.store_align(
            dst_tile,
            vreg_dst_even_f16,
            preg_all_f16,
            block_stride=BLOCK_STRIDE_ND,
            repeat_stride=REPEAT_STRIDE_ND,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            post_update=True,
        )
    vf.store_unalign_post(sum_tile, ureg_exp_sum, 0, post_update=True)


@pl.vector_function
def process_vec1_nd_no_update_vf_unalign(
    input_tile, dst_tile, max_tile, max_tile_st, sum_tile, mask_tile, s1_size, s2_size
):
    preg_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    preg_mask_unroll = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    ureg_max = vf.unalign_reg_for_store()
    ureg_exp_sum = vf.unalign_reg_for_store()

    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(s2_size - 64, dtype=pl.DT_FP32)
    preg_all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)

    vreg_min = vf.full(NEG_INF)

    for m in pl.range(s1_size):
        vreg_x = vf.load_align(input_tile, m * TKV)
        vreg_x_unroll = vf.load_align(input_tile, m * TKV + 64)

        vreg_x = vf.muls(vreg_x, SCALE, preg_all)
        vreg_x_unroll = vf.muls(vreg_x_unroll, SCALE, preg_tail)

        preg_mask = vf.load_align(mask_tile, m * TKV, dist=pl.LoadDist.DS)
        preg_mask_unroll = vf.load_align(mask_tile, m * TKV + 64, dist=pl.LoadDist.DS)
        vreg_x = vf.select(vreg_min, vreg_x, preg_mask)
        vreg_x_unroll = vf.select(vreg_min, vreg_x_unroll, preg_mask_unroll)
        vreg_x_unroll = vf.select(vreg_x_unroll, vreg_min, preg_tail)
        vf.store_align(input_tile + m * TKV, vreg_x, preg_all)
        vf.store_align(input_tile + m * TKV + 64, vreg_x_unroll, preg_all)
        vreg_max_tmp = vf.max(vreg_x, vreg_x_unroll, preg_all)
        vreg_max = vf.reduce_max(vreg_max_tmp, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(max_tile, vreg_max, ureg_max, 1, post_update=True)
    vf.store_unalign_post(max_tile, ureg_max, 0, post_update=True)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)

    for i in pl.range(s1_size):
        vreg_max_2 = vf.load_align(max_tile_st, i, dist=pl.LoadDist.BRC_B32)
        vreg_x_2, vreg_x_unroll_2 = vf.load_align(input_tile, i * TKV, dist=pl.LoadDist.DINTLV_B32)
        vreg_exp_even = vf.exp_sub(vreg_x_2, vreg_max_2, preg_all)
        vreg_exp_odd = vf.exp_sub(vreg_x_unroll_2, vreg_max_2, preg_all)

        vreg_exp_sum = vf.add(vreg_exp_even, vreg_exp_odd, preg_all)
        vreg_exp_sum = vf.reduce_sum(vreg_exp_sum, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(sum_tile, vreg_exp_sum, ureg_exp_sum, 1, post_update=True)

        vreg_exp_even_f16 = vf.astype(vreg_exp_even, preg_all, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP16)
        vreg_exp_odd_f16 = vf.astype(vreg_exp_odd, preg_all, layout=pl.CastLayout.ONE, dtype=pl.DT_FP16)
        vreg_exp_f16 = vf.or_(vreg_exp_even_f16, vreg_exp_odd_f16, preg_all_f16)
        vf.store_align(
            dst_tile,
            vreg_exp_f16,
            preg_all_f16,
            block_stride=BLOCK_STRIDE_ND,
            repeat_stride=REPEAT_STRIDE_ND,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            post_update=True,
        )
    vf.store_unalign_post(sum_tile, ureg_exp_sum, 0, post_update=True)


@pl.vector_function
def process_vec1_nd_no_update_vf(input_tile, dst_tile, max_tile, max_tile_st, sum_tile, mask_tile, s1_size):
    preg_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    preg_mask_unroll = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    ureg_max = vf.unalign_reg_for_store()
    ureg_exp_sum = vf.unalign_reg_for_store()

    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)

    vreg_min = vf.full(NEG_INF)

    for m in pl.range(s1_size):
        vreg_x = vf.load_align(input_tile, m * TKV)
        vreg_x_unroll = vf.load_align(input_tile, m * TKV + 64)

        vreg_x = vf.muls(vreg_x, SCALE, preg_all)
        vreg_x_unroll = vf.muls(vreg_x_unroll, SCALE, preg_all)

        preg_mask = vf.load_align(mask_tile, m * TKV, dist=pl.LoadDist.DS)
        preg_mask_unroll = vf.load_align(mask_tile, m * TKV + 64, dist=pl.LoadDist.DS)
        vreg_x = vf.select(vreg_min, vreg_x, preg_mask)
        vreg_x_unroll = vf.select(vreg_min, vreg_x_unroll, preg_mask_unroll)
        vf.store_align(input_tile + m * TKV, vreg_x, preg_all)
        vf.store_align(input_tile + m * TKV + 64, vreg_x_unroll, preg_all)
        vreg_max_tmp = vf.max(vreg_x, vreg_x_unroll, preg_all)
        vreg_max = vf.reduce_max(vreg_max_tmp, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(max_tile, vreg_max, ureg_max, 1, post_update=True)
    vf.store_unalign_post(max_tile, ureg_max, 0, post_update=True)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)

    for i in pl.range(s1_size):
        vreg_max_2 = vf.load_align(max_tile_st, i, dist=pl.LoadDist.BRC_B32)
        vreg_x_2, vreg_x_unroll_2 = vf.load_align(input_tile, i * TKV, dist=pl.LoadDist.DINTLV_B32)
        vreg_exp_even = vf.exp_sub(vreg_x_2, vreg_max_2, preg_all)
        vreg_exp_odd = vf.exp_sub(vreg_x_unroll_2, vreg_max_2, preg_all)

        vreg_exp_sum = vf.add(vreg_exp_even, vreg_exp_odd, preg_all)
        vreg_exp_sum = vf.reduce_sum(vreg_exp_sum, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(sum_tile, vreg_exp_sum, ureg_exp_sum, 1, post_update=True)

        vreg_exp_even_f16 = vf.astype(vreg_exp_even, preg_all, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP16)
        vreg_exp_odd_f16 = vf.astype(vreg_exp_odd, preg_all, layout=pl.CastLayout.ONE, dtype=pl.DT_FP16)
        vreg_exp_f16 = vf.or_(vreg_exp_even_f16, vreg_exp_odd_f16, preg_all_f16)
        vf.store_align(
            dst_tile,
            vreg_exp_f16,
            preg_all_f16,
            block_stride=BLOCK_STRIDE_ND,
            repeat_stride=REPEAT_STRIDE_ND,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            post_update=True,
        )
    vf.store_unalign_post(sum_tile, ureg_exp_sum, 0, post_update=True)


@pl.vector_function
def process_vec1_nd_update_vf_unalign64(
    input_tile, dst_tile, max_tile, mask_tile, tmp_max, tmp_max_st, tmp_exp_sum, s1_size, s2_size
):
    preg_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    ureg_max = vf.unalign_reg_for_store()
    ureg_exp_sum = vf.unalign_reg_for_store()

    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(s2_size, dtype=pl.DT_FP32)
    preg_all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)

    vreg_min = vf.full(NEG_INF)

    for m in pl.range(s1_size):
        vreg_x = vf.load_align(input_tile, m * TKV)
        vreg_x = vf.muls(vreg_x, SCALE, preg_tail)

        preg_mask = vf.load_align(mask_tile, m * TKV, dist=pl.LoadDist.DS)
        vreg_x = vf.select(vreg_min, vreg_x, preg_mask)
        vf.store_align(input_tile + m * TKV, vreg_x, preg_all)
        vreg_max = vf.reduce_max(vreg_x, preg_tail, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(tmp_max, vreg_max, ureg_max, 1, post_update=True)
    vf.store_unalign_post(tmp_max, ureg_max, 0, post_update=True)
    vreg_in_max = vf.load_align(max_tile, 0)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    vreg_max_new = vf.load_align(tmp_max_st, 0)
    vreg_max_new = vf.max(vreg_in_max, vreg_max_new, preg_all)
    vf.store_align(tmp_max_st, vreg_max_new, preg_all)

    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    for i in pl.range(s1_size):
        vreg_max_2 = vf.load_align(tmp_max_st, i, dist=pl.LoadDist.BRC_B32)
        vreg_x_2 = vf.load_align(input_tile, i * TKV)
        vreg_exp = vf.exp_sub(vreg_x_2, vreg_max_2, preg_tail)

        vreg_exp_sum = vf.reduce_sum(vreg_exp, preg_tail, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(tmp_exp_sum, vreg_exp_sum, ureg_exp_sum, 1, post_update=True)

        vreg_exp_even_f16 = vf.astype(vreg_exp, preg_all, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP16)
        vreg_dst_even_f16, vreg_dst_odd_f16 = vf.de_interleave(vreg_exp_even_f16, vreg_exp_even_f16)
        vf.store_align(
            dst_tile,
            vreg_dst_even_f16,
            preg_all_f16,
            block_stride=BLOCK_STRIDE_ND,
            repeat_stride=REPEAT_STRIDE_ND,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            post_update=True,
        )
    vf.store_unalign_post(tmp_exp_sum, ureg_exp_sum, 0, post_update=True)


@pl.vector_function
def process_vec1_nd_update_vf_unalign(
    input_tile, dst_tile, max_tile, mask_tile, tmp_max, tmp_max_st, tmp_exp_sum, s1_size, s2_size
):
    preg_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    preg_mask_unroll = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    ureg_max = vf.unalign_reg_for_store()
    ureg_exp_sum = vf.unalign_reg_for_store()

    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(s2_size - 64, dtype=pl.DT_FP32)
    preg_all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)

    vreg_min = vf.full(NEG_INF)

    for m in pl.range(s1_size):
        vreg_x = vf.load_align(input_tile, m * TKV)
        vreg_x_unroll = vf.load_align(input_tile, m * TKV + 64)

        vreg_x = vf.muls(vreg_x, SCALE, preg_all)
        vreg_x_unroll = vf.muls(vreg_x_unroll, SCALE, preg_tail)

        preg_mask = vf.load_align(mask_tile, m * TKV, dist=pl.LoadDist.DS)
        preg_mask_unroll = vf.load_align(mask_tile, m * TKV + 64, dist=pl.LoadDist.DS)
        vreg_x = vf.select(vreg_min, vreg_x, preg_mask)
        vreg_x_unroll = vf.select(vreg_min, vreg_x_unroll, preg_mask_unroll)
        vreg_x_unroll = vf.select(vreg_x_unroll, vreg_min, preg_tail)
        vf.store_align(input_tile + m * TKV, vreg_x, preg_all)
        vf.store_align(input_tile + m * TKV + 64, vreg_x_unroll, preg_all)
        vreg_max_tmp = vf.max(vreg_x, vreg_x_unroll, preg_all)
        vreg_max = vf.reduce_max(vreg_max_tmp, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(tmp_max, vreg_max, ureg_max, 1, post_update=True)
    vf.store_unalign_post(tmp_max, ureg_max, 0, post_update=True)
    vreg_in_max = vf.load_align(max_tile, 0)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    vreg_max_new = vf.load_align(tmp_max_st, 0)
    vreg_max_new = vf.max(vreg_in_max, vreg_max_new, preg_all)
    vf.store_align(tmp_max_st, vreg_max_new, preg_all)

    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    for i in pl.range(s1_size):
        vreg_max_2 = vf.load_align(tmp_max_st, i, dist=pl.LoadDist.BRC_B32)
        vreg_x_2, vreg_x_unroll_2 = vf.load_align(input_tile, i * TKV, dist=pl.LoadDist.DINTLV_B32)
        vreg_exp_even = vf.exp_sub(vreg_x_2, vreg_max_2, preg_all)
        vreg_exp_odd = vf.exp_sub(vreg_x_unroll_2, vreg_max_2, preg_all)

        vreg_exp_sum = vf.add(vreg_exp_even, vreg_exp_odd, preg_all)
        vreg_exp_sum = vf.reduce_sum(vreg_exp_sum, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(tmp_exp_sum, vreg_exp_sum, ureg_exp_sum, 1, post_update=True)

        vreg_exp_even_f16 = vf.astype(vreg_exp_even, preg_all, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP16)
        vreg_exp_odd_f16 = vf.astype(vreg_exp_odd, preg_all, layout=pl.CastLayout.ONE, dtype=pl.DT_FP16)
        vreg_exp_f16 = vf.or_(vreg_exp_even_f16, vreg_exp_odd_f16, preg_all_f16)
        vf.store_align(
            dst_tile,
            vreg_exp_f16,
            preg_all_f16,
            block_stride=BLOCK_STRIDE_ND,
            repeat_stride=REPEAT_STRIDE_ND,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            post_update=True,
        )
    vf.store_unalign_post(tmp_exp_sum, ureg_exp_sum, 0, post_update=True)


@pl.vector_function
def process_vec1_nd_update_vf(input_tile, dst_tile, max_tile, mask_tile, tmp_max, tmp_max_st, tmp_exp_sum, s1_size):
    preg_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    preg_mask_unroll = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    ureg_max = vf.unalign_reg_for_store()
    ureg_exp_sum = vf.unalign_reg_for_store()

    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)

    vreg_min = vf.full(NEG_INF)

    for m in pl.range(s1_size):
        vreg_x = vf.load_align(input_tile, m * TKV)
        vreg_x_unroll = vf.load_align(input_tile, m * TKV + 64)

        vreg_x = vf.muls(vreg_x, SCALE, preg_all)
        vreg_x_unroll = vf.muls(vreg_x_unroll, SCALE, preg_all)

        preg_mask = vf.load_align(mask_tile, m * TKV, dist=pl.LoadDist.DS)
        preg_mask_unroll = vf.load_align(mask_tile, m * TKV + 64, dist=pl.LoadDist.DS)
        vreg_x = vf.select(vreg_min, vreg_x, preg_mask)
        vreg_x_unroll = vf.select(vreg_min, vreg_x_unroll, preg_mask_unroll)
        vf.store_align(input_tile + m * TKV, vreg_x, preg_all)
        vf.store_align(input_tile + m * TKV + 64, vreg_x_unroll, preg_all)
        vreg_max_tmp = vf.max(vreg_x, vreg_x_unroll, preg_all)
        vreg_max = vf.reduce_max(vreg_max_tmp, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(tmp_max, vreg_max, ureg_max, 1, post_update=True)
    vf.store_unalign_post(tmp_max, ureg_max, 0, post_update=True)
    vreg_in_max = vf.load_align(max_tile, 0)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    vreg_max_new = vf.load_align(tmp_max_st, 0)
    vreg_max_new = vf.max(vreg_in_max, vreg_max_new, preg_all)
    vf.store_align(tmp_max_st, vreg_max_new, preg_all)

    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    for i in pl.range(s1_size):
        vreg_max_2 = vf.load_align(tmp_max_st, i, dist=pl.LoadDist.BRC_B32)
        vreg_x_2, vreg_x_unroll_2 = vf.load_align(input_tile, i * TKV, dist=pl.LoadDist.DINTLV_B32)
        vreg_exp_even = vf.exp_sub(vreg_x_2, vreg_max_2, preg_all)
        vreg_exp_odd = vf.exp_sub(vreg_x_unroll_2, vreg_max_2, preg_all)

        vreg_exp_sum = vf.add(vreg_exp_even, vreg_exp_odd, preg_all)
        vreg_exp_sum = vf.reduce_sum(vreg_exp_sum, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(tmp_exp_sum, vreg_exp_sum, ureg_exp_sum, 1, post_update=True)

        vreg_exp_even_f16 = vf.astype(vreg_exp_even, preg_all, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP16)
        vreg_exp_odd_f16 = vf.astype(vreg_exp_odd, preg_all, layout=pl.CastLayout.ONE, dtype=pl.DT_FP16)
        vreg_exp_f16 = vf.or_(vreg_exp_even_f16, vreg_exp_odd_f16, preg_all_f16)
        vf.store_align(
            dst_tile,
            vreg_exp_f16,
            preg_all_f16,
            block_stride=BLOCK_STRIDE_ND,
            repeat_stride=REPEAT_STRIDE_ND,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            post_update=True,
        )
    vf.store_unalign_post(tmp_exp_sum, ureg_exp_sum, 0, post_update=True)


@pl.vector_function
def update_exp_sum(exp_diff, max_tile, tmp_max, sum_tile, tmp_sum):
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)

    vreg_max = vf.load_align(max_tile, 0)
    vreg_max_tmp = vf.load_align(tmp_max, 0)
    vreg_exp_max = vf.exp_sub(vreg_max, vreg_max_tmp, preg_all)
    vf.store_align(exp_diff, vreg_exp_max, preg_all)
    vf.store_align(max_tile, vreg_max_tmp, preg_all)

    vreg_sum = vf.load_align(sum_tile, 0)
    vreg_sum_tmp = vf.load_align(tmp_sum, 0)
    vreg_exp_update = vf.mul(vreg_sum, vreg_exp_max, preg_all)
    vreg_exp_update = vf.add(vreg_exp_update, vreg_sum_tmp, preg_all)
    vf.store_align(sum_tile, vreg_exp_update, preg_all)


@pl.vector_function
def flash_update_basic_vf(dst_tile, cur_tile, pre_tile, exp_max_tile, s1_size, has_tail):
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(TAIL_D, dtype=pl.DT_FP32)
    for i in pl.range(0, s1_size):
        vreg_exp_max = vf.load_align(exp_max_tile, i * REDUCE_SIZE, dist=pl.LoadDist.BRC_B32)
        for j in pl.range(0, D_LOOPS):
            vreg_input_pre = vf.load_align(pre_tile, i * TD + j * FLOAT_REP_SIZE)
            vreg_input_cur = vf.load_align(cur_tile, i * TD + j * FLOAT_REP_SIZE)
            vreg_mul = vf.mul(vreg_exp_max, vreg_input_pre, preg_all)
            vreg_add = vf.add(vreg_mul, vreg_input_cur, preg_all)
            vf.store_align(dst_tile + (i * TD + j * FLOAT_REP_SIZE), vreg_add, preg_all)
        for _ in pl.range(0, has_tail):
            vreg_input_pre = vf.load_align(pre_tile, i * TD + D_LOOPS * FLOAT_REP_SIZE)
            vreg_input_cur = vf.load_align(cur_tile, i * TD + D_LOOPS * FLOAT_REP_SIZE)
            vreg_mul = vf.mul(vreg_exp_max, vreg_input_pre, preg_tail)
            vreg_add = vf.add(vreg_mul, vreg_input_cur, preg_tail)
            vf.store_align(dst_tile + (i * TD + D_LOOPS * FLOAT_REP_SIZE), vreg_add, preg_tail)


@pl.vector_function
def flash_update_last_basic_vf(dst_tile, cur_tile, pre_tile, exp_max_tile, exp_sum_tile, s1_size, has_tail):
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(TAIL_D, dtype=pl.DT_FP32)
    for i in pl.range(0, s1_size):
        vreg_exp_max = vf.load_align(exp_max_tile, i * REDUCE_SIZE, dist=pl.LoadDist.BRC_B32)
        vreg_exp_sum = vf.load_align(exp_sum_tile, i * REDUCE_SIZE, dist=pl.LoadDist.BRC_B32)
        for j in pl.range(0, D_LOOPS):
            vreg_input_pre = vf.load_align(pre_tile, i * TD + j * FLOAT_REP_SIZE)
            vreg_input_cur = vf.load_align(cur_tile, i * TD + j * FLOAT_REP_SIZE)
            vreg_mul = vf.mul(vreg_exp_max, vreg_input_pre, preg_all)
            vreg_add = vf.add(vreg_mul, vreg_input_cur, preg_all)
            vreg_div = vf.div(vreg_add, vreg_exp_sum, preg_all)
            vf.store_align(dst_tile + (i * TD + j * FLOAT_REP_SIZE), vreg_div, preg_all)
        for _ in pl.range(0, has_tail):
            vreg_input_pre = vf.load_align(pre_tile, i * TD + D_LOOPS * FLOAT_REP_SIZE)
            vreg_input_cur = vf.load_align(cur_tile, i * TD + D_LOOPS * FLOAT_REP_SIZE)
            vreg_mul = vf.mul(vreg_exp_max, vreg_input_pre, preg_tail)
            vreg_add = vf.add(vreg_mul, vreg_input_cur, preg_tail)
            vreg_div = vf.div(vreg_add, vreg_exp_sum, preg_tail)
            vf.store_align(dst_tile + (i * TD + D_LOOPS * FLOAT_REP_SIZE), vreg_div, preg_tail)


@pl.vector_function
def last_div_vf(dst_tile, cur_tile, exp_sum_tile, s1_size, has_tail):
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(TAIL_D, dtype=pl.DT_FP32)
    for i in pl.range(0, s1_size):
        vreg_exp_sum = vf.load_align(exp_sum_tile, i * REDUCE_SIZE, dist=pl.LoadDist.BRC_B32)
        for j in pl.range(0, D_LOOPS):
            vreg_input_cur = vf.load_align(cur_tile, i * TD + j * FLOAT_REP_SIZE)
            vreg_div = vf.div(vreg_input_cur, vreg_exp_sum, preg_all)
            vf.store_align(dst_tile + (i * TD + j * FLOAT_REP_SIZE), vreg_div, preg_all)
        for _ in pl.range(0, has_tail):
            vreg_input_cur = vf.load_align(cur_tile, i * TD + D_LOOPS * FLOAT_REP_SIZE)
            vreg_div = vf.div(vreg_input_cur, vreg_exp_sum, preg_tail)
            vf.store_align(dst_tile + (i * TD + D_LOOPS * FLOAT_REP_SIZE), vreg_div, preg_tail)


@pl.vector_function
def decode_mask(mask, maskr, holel, holes, col, s1_size, hole_num):
    index = vf.arange(col, dtype=pl.DT_INT32)
    index_unroll = vf.arange(col + 64, dtype=pl.DT_INT32)
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    preg_all_b16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    merge_bit = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    merge_unroll_bit = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    row_reg = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    temp_reg = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    vreg_zero = vf.full(0, dtype=pl.DT_UINT16)
    vreg_one = vf.full(1, dtype=pl.DT_UINT16)
    for m in pl.range(0, s1_size):
        maskr_reg = vf.load_align(maskr, m, dist=pl.LoadDist.BRC_B32, dtype=pl.DT_INT32)
        # Start from the causal right bound: visible where c < maskr.
        merge_bit = vf.lt(index, maskr_reg, preg_all)
        merge_unroll_bit = vf.lt(index_unroll, maskr_reg, preg_all)
        # AND in "c not in hole h" for each of the row's up-to-MAX_HOLES holes.
        # hole h, row m lives at offset h * TS_HALF + m in the packed buffers.
        # (c - holel) >= holes (unsigned) means c is outside [holel, holel+holes):
        # when c < holel the uint32 subtraction wraps to a large value, so a hole
        # at an arbitrary position (holel > 0) is handled without a signed branch;
        # holes == 0 makes the condition trivially true (an unused hole slot).
        for h in pl.range(0, hole_num):
            holel_reg = vf.load_align(holel, h * TS_HALF + m, dist=pl.LoadDist.BRC_B32, dtype=pl.DT_INT32)
            holes_reg = vf.load_align(holes, h * TS_HALF + m, dist=pl.LoadDist.BRC_B32, dtype=pl.DT_INT32)
            diff = vf.sub(index, holel_reg, preg_all)
            diff_unroll = vf.sub(index_unroll, holel_reg, preg_all)
            hole_bit = vf.ge(diff, holes_reg, preg_all)
            hole_unroll_bit = vf.ge(diff_unroll, holes_reg, preg_all)
            merge_bit = vf.and_(merge_bit, hole_bit, preg_all)
            merge_unroll_bit = vf.and_(merge_unroll_bit, hole_unroll_bit, preg_all)
        row_reg, temp_reg = vf.de_interleave(merge_bit, merge_unroll_bit, dtype=pl.DT_UINT16)
        mask_b16 = vf.select(vreg_zero, vreg_one, row_reg)
        vf.store_align(mask + (m * TKV), mask_b16, preg_all_b16, dist=pl.StoreDist.PACK)


def compute_qk(
    ctx, ki, sq_off, q, k, cur_q_slot, k_l1_db, left_db, right_db, acc_db, qk_vec_db, task_id, left2, right2, acc2
):
    # --- compute_qk inlined ---
    skv_off = ctx.s2_size_acc + ctx.ki * TKV
    cur_k_slot = k_l1_db.next()
    qk_left = left_db.next()
    qk_right = right_db.next()
    qk_acc = acc_db.next()
    left2.next()
    right2.next()
    acc2.next()

    if ctx.loop_count == 0:
        pl.set_validshape(cur_q_slot, [ctx.s1_size, TD])
        pl.load(cur_q_slot, q, [sq_off, ctx.n_idx, 0], order=[0, 2])
    pl.set_validshape(cur_k_slot, [TD, ctx.s2_size])
    pl.load(cur_k_slot, k, [skv_off, ctx.n_idx, 0], order=[2, 0])

    pl.set_validshape(qk_left, [ctx.s1_size, TD])
    pl.move(qk_left, cur_q_slot)
    pl.set_validshape(qk_right, [TD, ctx.s2_size])
    pl.move(qk_right, cur_k_slot)
    pl.set_validshape(qk_acc, [ctx.s1_size, ctx.s2_size])
    pl.matmul(qk_acc, qk_left, qk_right)

    qk_slot = qk_vec_db.next()
    pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY_BARKWARD_IDS[task_id % 2])
    pl.set_validshape(qk_slot, [(ctx.s1_size + 1) // 2, ctx.s2_size])
    pl.set_validshape(qk_acc, [(ctx.s1_size + 1) // 2 * 2, (ctx.s2_size + 7) // 8 * 8])
    pl.move(qk_slot, qk_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY_FORWARD_IDS[task_id % 2])


def compute_pv(ctx, v_l1_db, p_mat_db, left_db, right_db, acc_db, pv_vec_db, v, left2, right2, acc2):
    sv_off = ctx.s2_size_acc + ctx.ki * TKV
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=P_READY_FORWARD_IDS[ctx.task_id % 3])
    cur_v_slot = v_l1_db.next()
    # # current() advances p_mat_db cursor to stay in sync with Vec,
    cur_p_slot = p_mat_db.next()
    pv_left = left_db.next()
    pv_right = right_db.next()
    pv_acc = acc_db.next()
    left2.next()
    right2.next()
    acc2.next()

    pl.set_validshape(cur_v_slot, [ctx.s2_size, TD])
    pl.load(cur_v_slot, v, [sv_off, ctx.n_idx, 0], order=[0, 2])
    pl.set_validshape(cur_p_slot, [ctx.s1_size, ctx.s2_size])
    pl.set_validshape(pv_left, [ctx.s1_size, ctx.s2_size])
    pl.move(pv_left, cur_p_slot)
    pl.set_validshape(pv_right, [ctx.s2_size, TD])
    pl.move(pv_right, cur_v_slot)
    pl.set_validshape(pv_acc, [ctx.s1_size, TD])
    pl.matmul(pv_acc, pv_left, pv_right)

    pv_slot = pv_vec_db.next()
    pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY_BARKWARD_IDS[ctx.task_id % 2])
    pl.set_validshape(pv_slot, [(ctx.s1_size + 1) // 2, TD])
    pl.set_validshape(pv_acc, [(ctx.s1_size + 1) // 2 * 2, TD])
    pl.move(pv_slot, pv_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY_FORWARD_IDS[ctx.task_id % 2])


def compute_p(
    ctx_p,
    sub_id,
    qk_vec_db,
    tile_nz_db,
    tmp_max,
    tmp_sum,
    global_max,
    global_sum,
    exp_corr_db,
    p_mat_db,
    mask_db,
    maskr_db,
    holel_db,
    holes_db,
    maskr_gm,
    holel_gm,
    holes_gm,
    zero_mask_gm,
):
    """Softmax on KQ tile -> P. Includes cross-core sync."""
    p_eid = ctx_p.task_id % 2
    q_idx_p = ctx_p.q_count % 3

    qk_slot = qk_vec_db.next()
    tile_nz = tile_nz_db.next()
    mask_buf = mask_db.next()
    maskr_buf = maskr_db.next()
    holel_buf = holel_db.next()
    holes_buf = holes_db.next()
    gmax_p = global_max[q_idx_p]
    gsum_p = global_sum[q_idx_p]
    exp_diff = exp_corr_db[ctx_p.task_id % 3]
    row_offset = ctx_p.first_s1 * sub_id
    row = ctx_p.s1_idx * TS
    col = ctx_p.ki * TKV

    if ctx_p.mask_bit == 0:
        # Fully-visible block: no per-element decode. Copy a fixed [TS_HALF, TKV]
        # all-zero mask in from GM (mask 0 == visible) instead of synthesizing it
        # on the vector unit with pl.expands.
        pl.set_validshape(mask_buf, [TS_HALF, TKV])
        pl.load(mask_buf, zero_mask_gm, [0, 0])
    else:
        # sub_id splits the Q tile's rows into two halves; each subblock must read
        # its own half's per-row mask params, so offset the GM row by row_offset.
        mask_row = row + row_offset
        pl.set_validshape(maskr_buf, [1, ctx_p.half_s1])
        pl.load(maskr_buf, maskr_gm, [ctx_p.b_idx, mask_row])

        pl.set_validshape(holel_buf, [MAX_HOLES, ctx_p.half_s1])
        pl.load(holel_buf, holel_gm, [ctx_p.b_idx, 0, mask_row], order=[1, 2])
        pl.set_validshape(holes_buf, [MAX_HOLES, ctx_p.half_s1])
        pl.load(holes_buf, holes_gm, [ctx_p.b_idx, 0, mask_row], order=[1, 2])
        decode_mask(mask_buf, maskr_buf, holel_buf, holes_buf, col, ctx_p.half_s1, ctx_p.hole_num)

    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_FORWARD_IDS[p_eid])
    if ctx_p.loop_count == 0:
        if ctx_p.s2_size == 128:
            process_vec1_nd_no_update_vf(qk_slot, tile_nz, gmax_p, gmax_p, gsum_p, mask_buf, ctx_p.half_s1)
        elif ctx_p.s2_size <= 64:
            process_vec1_nd_no_update_vf_unalign64(
                qk_slot, tile_nz, gmax_p, gmax_p, gsum_p, mask_buf, ctx_p.half_s1, ctx_p.s2_size
            )
        else:
            process_vec1_nd_no_update_vf_unalign(
                qk_slot, tile_nz, gmax_p, gmax_p, gsum_p, mask_buf, ctx_p.half_s1, ctx_p.s2_size
            )
    else:
        if ctx_p.s2_size == 128:
            process_vec1_nd_update_vf(qk_slot, tile_nz, gmax_p, mask_buf, tmp_max, tmp_max, tmp_sum, ctx_p.half_s1)
        elif ctx_p.s2_size <= 64:
            process_vec1_nd_update_vf_unalign64(
                qk_slot, tile_nz, gmax_p, mask_buf, tmp_max, tmp_max, tmp_sum, ctx_p.half_s1, ctx_p.s2_size
            )
        else:
            process_vec1_nd_update_vf_unalign(
                qk_slot, tile_nz, gmax_p, mask_buf, tmp_max, tmp_max, tmp_sum, ctx_p.half_s1, ctx_p.s2_size
            )
        update_exp_sum(exp_diff, gmax_p, tmp_max, gsum_p, tmp_sum)
    pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_BARKWARD_IDS[p_eid])

    cur_p_slot = p_mat_db.next()
    pl.set_validshape(tile_nz, [ctx_p.half_s1, ctx_p.s2_size])
    pl.insert(cur_p_slot, tile_nz, [row_offset, 0])
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=P_READY_FORWARD_IDS[ctx_p.task_id % 3])


def compute_gu(ctx_gu, pv_vec_db, exp_corr_db, global_sum_buf, running_o, o_f16, o):
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_FORWARD_IDS[ctx_gu.task_id % 2])

    sub_id = pl.get_subblock_idx()
    row_offset = ctx_gu.first_s1 * sub_id
    pv_slot = pv_vec_db.next()
    gsum_gu = global_sum_buf[ctx_gu.q_count % 3]
    exp_corr_gu = exp_corr_db[ctx_gu.task_id % 3]
    pl.set_validshape(running_o, [ctx_gu.half_s1, TD])
    pl.set_validshape(pv_slot, [ctx_gu.half_s1, TD])
    has_tail = 0
    if TAIL_D != 0:
        has_tail = 1
    if ctx_gu.loop_count == 0:
        pl.move(running_o, pv_slot)
    else:
        if ctx_gu.ki < ctx_gu.kv_loop - 1:
            flash_update_basic_vf(running_o, pv_slot, running_o, exp_corr_gu, ctx_gu.half_s1, has_tail)
        else:
            flash_update_last_basic_vf(running_o, pv_slot, running_o, exp_corr_gu, gsum_gu, ctx_gu.half_s1, has_tail)
            pl.set_validshape(o_f16, [ctx_gu.half_s1, TD])
            pl.cast(o_f16, running_o, mode=pl.RoundMode.CAST_ROUND)
            pl.store(o, o_f16, [ctx_gu.q_off + row_offset, ctx_gu.n_idx, 0], order=[0, 2])

    pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_BARKWARD_IDS[ctx_gu.task_id % 2])
    if ctx_gu.ki == ctx_gu.kv_loop - 1:
        if ctx_gu.loop_count == 0:
            last_div_vf(running_o, running_o, gsum_gu, ctx_gu.half_s1, has_tail)
            pl.set_validshape(o_f16, [ctx_gu.half_s1, TD])
            pl.cast(o_f16, running_o, mode=pl.RoundMode.CAST_ROUND)
            pl.store(o, o_f16, [ctx_gu.q_off + row_offset, ctx_gu.n_idx, 0], order=[0, 2])


# ================================================================
#  Kernel with NBuffer + auto_mutex
# ================================================================
@pl.jit(arch="3510", auto_mutex=True, compile_timeout=200)
def flex_attention(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    v: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    o: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    tile_range: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    sparse_compute: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
    sparse_mask: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
    maskr: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    holel: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    holes: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    zero_mask: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT8],
    actual_seq_q: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    actual_seq_kv: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    work_ranges: pl.Tensor[[pl.DYNAMIC, 2], pl.DT_INT32],
):
    n_dim = q.shape[1]
    batch = tile_range.shape[0]
    max_block_m = sparse_compute.shape[1]
    max_block_n = sparse_compute.shape[2]
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    hole_num = holel.shape[1]

    # ========== Cross-core shared buffers (UBNBuffer for double-buffer) ==========
    # P MAT - Vector insert, Cube PV read
    p_mat_db = pl.make_tile_group(
        type=pl.TileType(shape=[TS, TKV], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=MA2_P,
        mutex_ids=[14, 15, 16],
    )

    # qk_vec UB - Cube store from ACC, Vector softmax (double-buffer for FIFO)
    qk_vec_db = pl.make_tile_group(
        type=pl.TileType(
            shape=[TS_HALF, TKV],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
            compact=1,
            pad=pl.TilePad.min,
        ),
        addrs=VA0,
        mutex_ids=[17, 18],
    )

    # pv_vec UB - Cube store from ACC, Vector GU (double-buffer for FIFO)
    pv_vec_db = pl.make_tile_group(
        type=pl.TileType(shape=[TS_HALF, TD], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=VA8,
        mutex_ids=[19, 20],
    )

    # =================== CUBE SECTION ===================
    with pl.section_cube():
        # Cube-only buffers (independent buf_id space: 0-11)
        q_l1_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TS, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1], compact=1
            ),
            addrs=MA0_Q,
            mutex_ids=[0, 1],
        )
        k_l1_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TD, TKV],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.ZN,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=MA1_K,
            mutex_ids=[2, 3],
        )
        v_l1_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TKV, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1], compact=1
            ),
            addrs=MA3_V,
            mutex_ids=[4, 5],
        )

        left_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TS, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1], compact=1
            ),
            addrs=[0, 32768],
            mutex_ids=[6, 7],
        )
        right_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TD, TKV], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, valid_shape=[-1, -1], compact=1
            ),
            addrs=[0, 32768],
            mutex_ids=[8, 9],
        )
        acc_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TS, TKV], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1], compact=1
            ),
            addrs=[0, 65536, 131072, 196608],
            mutex_ids=[10, 11, 12, 13],
        )
        left_db2 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TS, TKV], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1], compact=1
            ),
            addrs=[0, 32768],
            mutex_ids=[6, 7],
        )
        right_db2 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TKV, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, valid_shape=[-1, -1], compact=1
            ),
            addrs=[0, 32768],
            mutex_ids=[8, 9],
        )
        acc_db2 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TS, TD], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1], compact=1
            ),
            addrs=[0, 65536, 131072, 196608],
            mutex_ids=[10, 11, 12, 13],
        )

        work_start = pl.getval(work_ranges, core_id * 2)
        work_end = pl.getval(work_ranges, core_id * 2 + 1)
        task_id = 0
        b_idx = 0
        s1_size_acc = 0
        s2_size_acc = 0
        actual_s1 = 0
        actual_s2 = 0
        s1o_acc = 0
        ctx_arr = pl.struct_array(
            4,
            "CubeCtx",
            n_idx=0,
            qi=0,
            ki=0,
            task_id=0,
            s1_size_acc=0,
            s2_size_acc=0,
            s1_size=0,
            s2_size=0,
            loop_count=0,
        )
        s1o_size = 0  # tmp-val
        for idx in pl.range(batch):
            if idx == 0:
                actual_s1 = pl.getval(actual_seq_q, idx)
                actual_s2 = pl.getval(actual_seq_kv, idx)
            else:
                actual_s1 = pl.getval(actual_seq_q, idx) - pl.getval(actual_seq_q, idx - 1)
                actual_s2 = pl.getval(actual_seq_kv, idx) - pl.getval(actual_seq_kv, idx - 1)
            s1o_size = s1o_size + (actual_s1 + TS - 1) // TS * n_dim
            if work_start >= s1o_size:
                s1o_acc = s1o_size
                s1_size_acc = s1_size_acc + actual_s1
                s2_size_acc = s2_size_acc + actual_s2
                b_idx = b_idx + 1
                continue
            break
        for work_id in pl.range(work_start, work_end):
            for _ in pl.range(b_idx, batch):
                if b_idx == 0:
                    actual_s1 = pl.getval(actual_seq_q, b_idx)
                    actual_s2 = pl.getval(actual_seq_kv, b_idx)
                else:
                    actual_s1 = pl.getval(actual_seq_q, b_idx) - pl.getval(actual_seq_q, b_idx - 1)
                    actual_s2 = pl.getval(actual_seq_kv, b_idx) - pl.getval(actual_seq_kv, b_idx - 1)
                s1o_size = s1o_acc + (actual_s1 + TS - 1) // TS * n_dim
                if work_id >= s1o_size:
                    s1o_acc = s1o_size
                    s1_size_acc = s1_size_acc + actual_s1
                    s2_size_acc = s2_size_acc + actual_s2
                    b_idx = b_idx + 1
                    continue
                break
            cur_b_s1o = (actual_s1 + TS - 1) // TS
            s1o_size = work_id - s1o_acc
            n_idx = s1o_size // cur_b_s1o
            s1_idx = s1o_size % cur_b_s1o
            s1_size = pl.min(TS, actual_s1 - s1_idx * TS)

            sq_off = s1_size_acc + s1_idx * TS
            cur_q_slot = q_l1_db.next()
            kv_loop = pl.getval(tile_range, b_idx * max_block_m + s1_idx) + 1
            kv_end = pl.min(kv_loop * TKV, actual_s2)
            drain = 0
            if work_id == work_end - 1:
                drain = 2
            q_loop_count = 0  # index of the computed KV block within this Q tile
            for ki in pl.range(0, kv_loop + drain):
                if ki < kv_loop:
                    base_offset = b_idx * max_block_m * max_block_n + s1_idx * max_block_n
                    word_offset = ki // 32
                    work_index = ki % 32
                    compute_bit = pl.getval(sparse_compute, base_offset + word_offset)
                    cur_bit = (compute_bit >> work_index) & 1
                    if cur_bit == 0:
                        continue
                    # Save current context
                    ctx_curr = ctx_arr[task_id % 4]
                    ctx_curr.task_id = task_id
                    ctx_curr.n_idx = n_idx
                    ctx_curr.ki = ki
                    ctx_curr.s1_size_acc = s1_size_acc
                    ctx_curr.s2_size_acc = s2_size_acc
                    ctx_curr.s1_size = s1_size
                    ctx_curr.s2_size = pl.min(TKV, kv_end - ki * TKV)
                    ctx_curr.loop_count = q_loop_count  # 0 on first computed block -> load Q
                    q_loop_count = q_loop_count + 1

                    # ========== compute_qk (current step) ==========
                    compute_qk(
                        ctx_curr,
                        ki,
                        sq_off,
                        q,
                        k,
                        cur_q_slot,
                        k_l1_db,
                        left_db,
                        right_db,
                        acc_db,
                        qk_vec_db,
                        task_id,
                        left_db2,
                        right_db2,
                        acc_db2,
                    )

                # ========== compute_pv (delayed 1 step: uses ctx from task_id-1) ==========
                if task_id > 1:
                    ctx_pre2 = ctx_arr[(task_id + 2) % 4]
                    compute_pv(
                        ctx_pre2,
                        v_l1_db,
                        p_mat_db,
                        left_db2,
                        right_db2,
                        acc_db2,
                        pv_vec_db,
                        v,
                        left_db,
                        right_db,
                        acc_db,
                    )
                task_id = task_id + 1

    # =================== VECTOR SECTION ===================
    with pl.section_vector():
        tile_nz_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[65, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], layout=pl.NZ
            ),
            addrs=VA1,
            mutex_ids=[0, 1],
        )

        running_o = pl.make_tile(
            pl.TileType(shape=[TS_HALF, TD], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addr=VA7
        )

        o_f16_g = pl.make_tile_group(
            type=pl.TileType(shape=[TS_HALF, TD], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addrs=VA9,
            mutex_ids=[2],
        )
        o_f16 = o_f16_g.next()

        mask_db = pl.make_tile_group(
            type=pl.TileType(shape=[TS_HALF, TKV], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec),
            addrs=VA10,
            mutex_ids=[3, 4],
        )

        maskr_db = pl.make_tile_group(
            type=pl.TileType(shape=[1, TS_HALF], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
            addrs=VA13,
            mutex_ids=[5, 6],
        )

        holel_db = pl.make_tile_group(
            type=pl.TileType(shape=[MAX_HOLES, TS_HALF], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
            addrs=VA14,
            mutex_ids=[7, 8],
        )

        holes_db = pl.make_tile_group(
            type=pl.TileType(shape=[MAX_HOLES, TS_HALF], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
            addrs=VA15,
            mutex_ids=[9, 10],
        )

        # Double-buffered global state (per Q tile) -use tile tuples for dynamic
        # indexing by q_count % 2, since StructArray ctx references need runtime index.
        red_type = pl.TileType(shape=[TS_HALF, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN)
        gmax_0 = pl.make_tile(red_type, addr=VA_GMAX0)
        gmax_1 = pl.make_tile(red_type, addr=VA_GMAX1)
        gmax_2 = pl.make_tile(red_type, addr=VA_GMAX2)
        global_max = (gmax_0, gmax_1, gmax_2)

        gsum_0 = pl.make_tile(red_type, addr=VA_GSUM0)
        gsum_1 = pl.make_tile(red_type, addr=VA_GSUM1)
        gsum_2 = pl.make_tile(red_type, addr=VA_GSUM2)
        global_sum = (gsum_0, gsum_1, gsum_2)

        tmp_max = pl.make_tile(red_type, addr=VA11)
        tmp_sum = pl.make_tile(red_type, addr=VA12)

        # FIFO exp_corr -use NBuffer with current() auto-rotate
        exp_max0 = pl.make_tile(red_type, addr=VA_EXPMAX0)
        exp_max1 = pl.make_tile(red_type, addr=VA_EXPMAX1)
        exp_max2 = pl.make_tile(red_type, addr=VA_EXPMAX2)
        exp_corr_db = (exp_max0, exp_max1, exp_max2)

        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_BARKWARD_IDS[0])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_BARKWARD_IDS[1])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_BARKWARD_IDS[0])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_BARKWARD_IDS[1])

        work_start = pl.getval(work_ranges, core_id * 2)
        work_end = pl.getval(work_ranges, core_id * 2 + 1)
        sub_id = pl.get_subblock_idx()

        task_id = 0
        q_count = 0
        b_idx = 0
        s1_size_acc = 0
        s2_size_acc = 0
        actual_s1 = 0
        actual_s2 = 0
        s1o_acc = 0

        # StructArray(3) for pipeline context tracking (same as original)
        ctx_arr = pl.struct_array(
            4,
            "VecCtx",
            b_idx=0,
            n_idx=0,
            s1_idx=0,
            q_off=0,
            ki=0,
            q_count=0,
            sub_id=0,
            task_id=0,
            kv_loop=0,
            half_s1=0,
            first_s1=0,
            s1_size=0,
            s2_size=0,
            compute_bit=0,
            mask_bit=0,
            loop_count=0,
            hole_num=0,
        )

        # calc start
        s1o_size = 0  # tmp-val
        for idx in pl.range(batch):
            if idx == 0:
                actual_s1 = pl.getval(actual_seq_q, idx)
                actual_s2 = pl.getval(actual_seq_kv, idx)
            else:
                actual_s1 = pl.getval(actual_seq_q, idx) - pl.getval(actual_seq_q, idx - 1)
                actual_s2 = pl.getval(actual_seq_kv, idx) - pl.getval(actual_seq_kv, idx - 1)
            s1o_size = s1o_size + (actual_s1 + TS - 1) // TS * n_dim
            if work_start >= s1o_size:
                s1o_acc = s1o_size
                s1_size_acc = s1_size_acc + actual_s1
                s2_size_acc = s2_size_acc + actual_s2
                b_idx = b_idx + 1
                continue
            break
        for work_id in pl.range(work_start, work_end):
            for _ in pl.range(b_idx, batch):
                if b_idx == 0:
                    actual_s1 = pl.getval(actual_seq_q, b_idx)
                    actual_s2 = pl.getval(actual_seq_kv, b_idx)
                else:
                    actual_s1 = pl.getval(actual_seq_q, b_idx) - pl.getval(actual_seq_q, b_idx - 1)
                    actual_s2 = pl.getval(actual_seq_kv, b_idx) - pl.getval(actual_seq_kv, b_idx - 1)
                s1o_size = s1o_acc + (actual_s1 + TS - 1) // TS * n_dim
                if work_id >= s1o_size:
                    s1o_acc = s1o_size
                    s1_size_acc = s1_size_acc + actual_s1
                    s2_size_acc = s2_size_acc + actual_s2
                    b_idx = b_idx + 1
                    continue
                break
            cur_b_s1o = (actual_s1 + TS - 1) // TS
            s1o_size = work_id - s1o_acc
            n_idx = s1o_size // cur_b_s1o
            s1_idx = s1o_size % cur_b_s1o
            s1_size = pl.min(TS, actual_s1 - s1_idx * TS)

            sq_acc = 0
            if b_idx > 0:
                sq_acc = pl.getval(actual_seq_q, b_idx - 1)
            q_off = sq_acc + s1_idx * TS
            kv_loop = pl.getval(tile_range, b_idx * max_block_m + s1_idx) + 1
            kv_end = pl.min(kv_loop * TKV, actual_s2)

            drain = 0
            if work_id == work_end - 1:
                drain = 3
            q_loop_count = 0  # reset per Q tile: first computed KV block gets loop_count 0
            for ki in pl.range(0, kv_loop + drain):
                if ki < kv_loop:
                    base_offset = b_idx * max_block_m * max_block_n + s1_idx * max_block_n
                    word_offset = ki // 32
                    work_index = ki % 32
                    compute_bit = pl.getval(sparse_compute, base_offset + word_offset)
                    mask_bit = pl.getval(sparse_mask, base_offset + word_offset)
                    cur_compute = (compute_bit >> work_index) & 1
                    cur_mask = (mask_bit >> work_index) & 1
                    if cur_compute == 0:
                        continue
                    # ===== producer: build this task's context =====
                    ctx_curr = ctx_arr[task_id % 4]
                    ctx_curr.b_idx = b_idx
                    ctx_curr.n_idx = n_idx
                    ctx_curr.s1_idx = s1_idx
                    ctx_curr.q_off = q_off
                    ctx_curr.task_id = task_id
                    ctx_curr.ki = ki
                    ctx_curr.kv_loop = kv_loop
                    ctx_curr.q_count = q_count
                    ctx_curr.s1_size = s1_size
                    ctx_curr.s2_size = pl.min(TKV, kv_end - ki * TKV)
                    half_s1 = (ctx_curr.s1_size + 1) // 2
                    first_s1 = half_s1
                    if sub_id == 1:
                        half_s1 = ctx_curr.s1_size - half_s1
                    ctx_curr.first_s1 = first_s1
                    ctx_curr.half_s1 = half_s1
                    ctx_curr.compute_bit = cur_compute
                    ctx_curr.mask_bit = cur_mask
                    ctx_curr.loop_count = q_loop_count
                    ctx_curr.hole_num = hole_num
                    q_loop_count = q_loop_count + 1

                # ===== compute_p (consumer, lag 1: real steps + 1st drain step) =====
                if task_id > 0:
                    if ki <= kv_loop:
                        ctx_p = ctx_arr[(task_id + 3) % 4]
                        compute_p(
                            ctx_p,
                            sub_id,
                            qk_vec_db,
                            tile_nz_g,
                            tmp_max,
                            tmp_sum,
                            global_max,
                            global_sum,
                            exp_corr_db,
                            p_mat_db,
                            mask_db,
                            maskr_db,
                            holel_db,
                            holes_db,
                            maskr,
                            holel,
                            holes,
                            zero_mask,
                        )

                # ===== compute_gu (consumer, lag 3: every step incl. all drains) =====
                if task_id > 2:
                    ctx_gu = ctx_arr[(task_id + 1) % 4]
                    compute_gu(ctx_gu, pv_vec_db, exp_corr_db, global_sum, running_o, o_f16, o)

                task_id = task_id + 1
            q_count = q_count + 1


def flash_attention_ref_tnd(q_tnd, k_tnd, v_tnd, seq_q_list, seq_kv_list, d, masks):
    """Reference attention operating on TND tensors, per-batch.

    ``masks[bi]`` is a bool tensor of shape [sq, skv] where True == masked
    (the cell is invisible and gets NEG_INF before softmax). It is the single
    source of truth shared with the device inputs (maskr/holel/holes + sparse).

    This mirrors the *on-device flash-attention compute flow* rather than a
    single fused fp32 softmax, so the reference stays faithful to what the
    kernel actually computes:
      * KV is consumed in TKV-wide blocks with an online (running max/sum)
        softmax, exactly like the Cube/Vector pipeline;
      * the P (probability) tile is cast to fp16 *before* the P@V matmul, as the
        kernel does (``vf.astype(..., DT_FP16)`` in ``process_vec1_*``), so P@V
        runs in fp16 with fp32 accumulation;
      * the running denominator uses the fp32 exp sum (the kernel reduces the
        sum from the fp32 exp, before the fp16 cast).
    Modelling the fp16-P quantization is what lets the reference agree with the
    kernel to ~fp16 ulp even for sliding-window masks, where the attention peak
    lands in a later block and earlier blocks' fp16 P is downscaled by exp_corr.
    """
    q_tnd = q_tnd.cpu()
    k_tnd = k_tnd.cpu()
    v_tnd = v_tnd.cpu()
    seq_q_list = seq_q_list.cpu() if torch.is_tensor(seq_q_list) else seq_q_list
    seq_kv_list = seq_kv_list.cpu() if torch.is_tensor(seq_kv_list) else seq_kv_list
    scale_val = 1.0 / math.sqrt(d)
    n = q_tnd.shape[1]
    o_tnd = torch.zeros_like(q_tnd)
    q_off = 0
    kv_off = 0
    for bi, sq_val in enumerate(seq_q_list):
        sq = int(sq_val)
        skv = int(seq_kv_list[bi])
        mask = masks[bi].cpu()
        for ni in range(n):
            qi = q_tnd[q_off:q_off + sq, ni, :].float()  # [sq, d]
            ki = k_tnd[kv_off:kv_off + skv, ni, :].float()  # [skv, d]
            vi = v_tnd[kv_off:kv_off + skv, ni, :].float()  # [skv, d]
            running_m = torch.full((sq, 1), float(NEG_INF))
            running_l = torch.zeros((sq, 1))
            acc = torch.zeros((sq, d))
            for c0 in range(0, skv, TKV):
                c1 = min(c0 + TKV, skv)
                qk = torch.matmul(qi, ki[c0:c1].T) * scale_val  # [sq, blk] fp32
                qk = qk.masked_fill(mask[:, c0:c1], NEG_INF)
                blk_max = qk.max(dim=1, keepdim=True).values  # [sq, 1]
                new_m = torch.maximum(running_m, blk_max)
                corr = torch.exp(running_m - new_m)  # [sq, 1]
                p = torch.exp(qk - new_m)  # [sq, blk] fp32
                p16 = p.half().float()  # fp16 P (FA flow)
                acc = acc * corr + torch.matmul(p16, vi[c0:c1])
                running_l = running_l * corr + p.sum(dim=1, keepdim=True)
                running_m = new_m
            o_tnd[q_off:q_off + sq, ni, :] = (acc / running_l).half()
        q_off += sq
        kv_off += skv
    return o_tnd


def build_causal_window_mask_inputs(seq_q_list, seq_kv_list, window):
    """Build the flex-attention mask inputs for a causal + sliding-window mask.

    Per query row r the visible KV columns are ``[max(0, r-window+1), r]``, encoded as:
      * maskr[r] = r + 1                         (visible right bound, c < maskr)
      * holel[r] = 0, holes[r] = max(0, r-window+1)  (invisible hole [0, holes))

    This uses a single hole channel (hole_num == 1); the holel/holes tensors are
    shaped [b, MAX_HOLES, pad] for a uniform interface with the multi-hole path,
    with channels 1..MAX_HOLES-1 left zero (a no-op in decode_mask).

    Returns the seven device tensors (CPU) plus a list of per-batch bool masks
    (True == masked) that is the shared source of truth for the reference.
    """
    import numpy as np

    b = len(seq_q_list)
    block_num = max((sq + TS - 1) // TS for sq in seq_q_list)
    col_block_num = max((skv + TKV - 1) // TKV for skv in seq_kv_list)
    word_num = (col_block_num + 31) // 32
    pad = block_num * TS

    tile_range = torch.zeros((b, block_num), dtype=torch.int32)
    sparse_compute = np.zeros((b, block_num, word_num), dtype=np.uint32)
    sparse_mask = np.zeros((b, block_num, word_num), dtype=np.uint32)
    maskr = torch.zeros((b, pad), dtype=torch.int32)
    holel = torch.zeros((b, MAX_HOLES, pad), dtype=torch.int32)
    holes = torch.zeros((b, MAX_HOLES, pad), dtype=torch.int32)
    dense_masks = []

    for bi in range(b):
        sq = seq_q_list[bi]
        skv = seq_kv_list[bi]
        vis = torch.zeros((sq, skv), dtype=torch.bool)
        for r in range(sq):
            right = r + 1
            hole_len = max(0, r - window + 1)
            maskr[bi, r] = right
            holel[bi, 0, r] = 0
            holes[bi, 0, r] = hole_len
            lo = hole_len
            hi = min(right, skv)
            if hi > lo:
                vis[r, lo:hi] = True

        blk_m = (sq + TS - 1) // TS
        blk_n = (skv + TKV - 1) // TKV
        for mi in range(blk_m):
            r0, r1 = mi * TS, min((mi + 1) * TS, sq)
            max_blk = 0
            for kj in range(blk_n):
                c0, c1 = kj * TKV, min((kj + 1) * TKV, skv)
                sub = vis[r0:r1, c0:c1]
                if bool(sub.any()):
                    max_blk = kj
                    sparse_compute[bi, mi, kj // 32] |= np.uint32(1 << (kj % 32))
                    if not bool(sub.all()):
                        # partial block -> needs per-element decode (mask_bit=1)
                        sparse_mask[bi, mi, kj // 32] |= np.uint32(1 << (kj % 32))
            tile_range[bi, mi] = max_blk
        dense_masks.append(~vis)

    return (
        tile_range,
        torch.from_numpy(sparse_compute),
        torch.from_numpy(sparse_mask),
        maskr,
        holel,
        holes,
        dense_masks,
    )


def build_work_ranges(seq_q_list, sparse_compute, n, num_cores):
    """Assign a contiguous work_id range to each core, balancing *cost* rather
    than the raw work_id count.

    Each work_id is one (batch, head, Q-tile) triple; the kernel walks them in
    the order  for bi: for n_idx: for s1_idx.  A work_id's cost is the number of
    KV blocks it actually computes (popcount of sparse_compute[bi, s1_idx]),
    which under a causal / sliding-window mask is highly non-uniform (a lower Q
    tile touches far fewer KV blocks). Splitting by count alone therefore leaves
    the core holding the heaviest tiles as the straggler; splitting so each
    core's summed cost is ~equal removes that skew.

    The kernel requires each core's range to be a *contiguous* [start, end)
    (its b_idx / seq accumulators advance monotonically with work_id), so this
    is a "split array into num_cores parts minimizing the largest part-sum"
    problem, solved exactly by binary-searching the max part cost.

    Returns an int32 [num_cores, 2] tensor of [start, end) work_id ranges.
    """
    sc = sparse_compute.cpu().numpy() if torch.is_tensor(sparse_compute) else sparse_compute
    b = len(seq_q_list)

    # Per-work_id cost, in kernel work_id order (bi outer, then n_idx, then s1_idx).
    costs = []
    for bi in range(b):
        blk_m = (seq_q_list[bi] + TS - 1) // TS
        word = sc.shape[2]
        tile_cost = [sum(int(sc[bi, s1, w]).bit_count() for w in range(word)) for s1 in range(blk_m)]
        for _n in range(n):
            costs.extend(tile_cost)
    total_work = len(costs)

    prefix = [0] * (total_work + 1)
    for i, c in enumerate(costs):
        prefix[i + 1] = prefix[i] + c

    def _num_parts(cap):
        # greedily pack contiguous work_ids into as few cores as possible s.t.
        # no core's cost exceeds cap; returns parts needed (inf if a single
        # work_id already exceeds cap -> cap too small).
        parts, cur, i = 1, 0, 0
        while i < total_work:
            c = costs[i]
            if c > cap:
                return total_work + 1  # infeasible cap
            if cur + c > cap:
                parts += 1
                cur = 0
            cur += c
            i += 1
        return parts

    # Binary-search the smallest feasible max-part cost (min-max objective).
    lo, hi = (max(costs) if costs else 0), (prefix[total_work] or 1)
    while lo < hi:
        mid = (lo + hi) // 2
        if _num_parts(mid) <= num_cores:
            hi = mid
        else:
            lo = mid + 1
    cap = lo

    # Emit contiguous ranges packed under `cap`. If the greedy packing uses
    # fewer than num_cores cores, the remaining cores get empty [end, end)
    # ranges (the kernel iterates range(start, end) -> no work).
    work_ranges = torch.zeros((num_cores, 2), dtype=torch.int32)
    core, start, cur, i = 0, 0, 0, 0
    while i < total_work and core < num_cores:
        c = costs[i]
        if cur + c > cap and cur > 0:
            work_ranges[core, 0] = start
            work_ranges[core, 1] = i
            core += 1
            start, cur = i, 0
            continue
        cur += c
        i += 1
    if core < num_cores:
        work_ranges[core, 0] = start
        work_ranges[core, 1] = total_work
        core += 1
    for c in range(core, num_cores):
        work_ranges[c, 0] = total_work
        work_ranges[c, 1] = total_work
    return work_ranges


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_fa_perf_nbuf():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(42)

    # Each cfg targets a specific code path of the kernel. The three softmax
    # branches in compute_p() are selected by the KV-tile width s2_size:
    #   s2_size == 128            -> process_vec1_nd_*_vf            (aligned)
    #   s2_size <= 64             -> process_vec1_nd_*_vf_unalign64
    #   64 < s2_size < 128        -> process_vec1_nd_*_vf_unalign
    # s2_size for the last KV block of a batch == skv % 128 (when non-zero).
    #
    # The sliding-window cases additionally exercise the path where a Q-tile's
    # leading KV block(s) are fully masked and skipped, so the first *computed*
    # block is at ki>0 (loop_count==0 but ki>0) -- a regression guard for the
    # stale-Q / first-block-init bugs. With those fixed the kernel matches the
    # FA-flow reference to ~1 fp16 ulp for every case, so all cfgs use a tight atol.
    failures = []
    test_cfgs = [
        ([512], [512], 1, TD, 100000, 28, 1e-3),  # baseline: aligned, full causal, s2=128 aligned path
        ([512], [512], 1, TD, 128, 28, 1e-3),  # sliding window=128 -> leading-block skip (loop_count0@ki>0)
        ([512], [512], 1, TD, 256, 28, 1e-3),  # sliding window=256 -> multi-block window + skip
        ([200], [200], 1, TD, 100000, 28, 1e-3),  # q-tail(72) + kv tail s2=72 -> unalign path
        ([160], [160], 1, TD, 100000, 28, 1e-3),  # kv tail s2=32 -> unalign64 path
        ([100], [100], 1, TD, 100000, 28, 1e-3),  # single sub-128 block, s2=100 -> unalign path
        ([50], [50], 1, TD, 100000, 28, 1e-3),  # single tiny block, s2=50 -> unalign64 path
        ([256], [256], 2, TD, 100000, 28, 1e-3),  # multi-head n=2
        ([128, 256, 200], [128, 256, 200], 1, TD, 100000, 28, 1e-3),  # multi-batch varlen TND
        ([384, 100], [384, 100], 2, TD, 256, 28, 1e-3),  # varlen + multi-head + sliding window
    ]
    for test_cfg in test_cfgs:
        seq_q_list, seq_kv_list, n, d, window, num_cores, atol = test_cfg
        b = len(seq_q_list)
        tq = sum(seq_q_list)
        tkv = sum(seq_kv_list)
        logging.info(
            "\nFA-TND-DN-A5 (b=%s, seq_q=%s, seq_kv=%s, n=%s, d=%s) window=%s cores=%s",
            b,
            seq_q_list,
            seq_kv_list,
            n,
            d,
            window,
            num_cores,
        )

        q = torch.rand((tq, n, d), device="cpu", dtype=torch.float16)
        k = torch.rand((tkv, n, d), device="cpu", dtype=torch.float16)
        v = torch.rand((tkv, n, d), device="cpu", dtype=torch.float16)
        o = torch.zeros((tq, n, d), device="cpu", dtype=torch.float16)

        # Build the causal + sliding-window mask inputs; `masks` is the shared
        # source of truth also fed to the reference.
        (tile_range, sparse_compute, sparse_mask, maskr, holel, holes, masks) = build_causal_window_mask_inputs(
            seq_q_list, seq_kv_list, window
        )

        actual_q_len = list(itertools.accumulate(seq_q_list))
        actual_kv_len = list(itertools.accumulate(seq_kv_list))
        total_work = sum((x + 127) // 128 for x in seq_q_list) * n
        # Cost-balanced contiguous partition (see build_work_ranges): splits so
        # each core's summed KV-block cost is ~equal instead of its tile count.
        work_ranges = build_work_ranges(seq_q_list, sparse_compute, n, num_cores)
        ################h2d###############
        actual_seq_q = torch.tensor(actual_q_len, device=device, dtype=torch.int32)
        actual_seq_kv = torch.tensor(actual_kv_len, device=device, dtype=torch.int32)
        q = q.to(device)
        k = k.to(device)
        v = v.to(device)
        o = o.to(device)
        tile_range = tile_range.to(device)
        sparse_compute = sparse_compute.to(device)
        sparse_mask = sparse_mask.to(device)
        maskr = maskr.to(device)
        holel = holel.to(device)
        holes = holes.to(device)
        zero_mask = torch.zeros((TS_HALF, TKV), device=device, dtype=torch.uint8)
        work_ranges = work_ranges.to(device)
        actual_num_cores = min(num_cores, total_work)
        flex_attention[None, actual_num_cores](
            q,
            k,
            v,
            o,
            tile_range,
            sparse_compute,
            sparse_mask,
            maskr,
            holel,
            holes,
            zero_mask,
            actual_seq_q,
            actual_seq_kv,
            work_ranges,
        )
        torch.npu.synchronize()

        o_ref = flash_attention_ref_tnd(q, k, v, seq_q_list, seq_kv_list, d, masks)
        o = o.cpu()
        diff = (o - o_ref).abs().max().item()
        # Guard against the all-zero regression (host `o` never written back).
        out_is_zero = bool(o.abs().max().item() == 0.0)
        status = "PASS"
        try:
            assert not out_is_zero, "kernel output is all-zero (o not written back to host)"
            torch.testing.assert_close(o, o_ref, rtol=atol, atol=atol)
        except AssertionError as e:
            status = "FAIL"
            failures.append((test_cfg, diff, str(e).splitlines()[0]))
        logging.info(
            "  [%s] b=%d seq_q=%s seq_kv=%s n=%d window=%s atol=%.0e max|diff|=%.5f",
            status,
            b,
            seq_q_list,
            seq_kv_list,
            n,
            window,
            atol,
            diff,
        )

    if failures:
        msg = "\n".join("  cfg=%s max|diff|=%.5f : %s" % (c, d, m) for c, d, m in failures)
        raise AssertionError("%d/%d precision cases failed:\n%s" % (len(failures), len(test_cfgs), msg))
    logging.info("  PASS")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.npu.set_device(ST_DEVICE)
    logging.info("FlexAttention with NBuffer + auto_mutex (QK_PRELOAD=%s, FIFO=%s)", QK_PRELOAD, FIFO_SIZE)
    logging.info("%s", '=' * 60)
    test_fa_perf_nbuf()
    logging.info("\nAll tests passed!")
