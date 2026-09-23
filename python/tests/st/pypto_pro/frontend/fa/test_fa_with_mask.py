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
    python3 python/tests/st/pypto_pro/frontend/fa/test_fa_with_mask.py
"""

import itertools
import logging
import math
import os

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

# ================================================================
#  Configuration
# ================================================================
QK_PRELOAD = 1
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
assert VA13 <= 248 * 1024


# ================================================================
#  Mutex IDs - Cube and Vector use independent buf_id spaces
# ================================================================
# Cube-only (inside section_cube): 0-11
#   Q L1: (0, 1), K L1: (2, 3), V L1: (4, 5)
#   L0A: (6, 7), L0B: (8, 9), L0C: (10, 11)
#
# Vector-only (inside section_vector): 0-11
#   tmp_vec: 0, p_f16: 1, reduce_dst: 2
#   gmax_rm: (3, 4), gsum: (5, 6)
#   exp_corr: (7, 8), running_o: 9, o_f16: 10, tile_nz: 11
#
# Cross-core shared (outside sections): 12-17
#   P MAT: (12, 13)
#   qk_vec UB: (14, 15)
#   pv_vec UB: (16, 17)

# Cross-core shared buffer IDs
QK_READY_FORWARD_IDS = (0, 1)
QK_READY_BARKWARD_IDS = (2, 3)
P_READY_FORWARD_IDS = (4, 5, 6)
PV_READY_FORWARD_IDS = (7, 8)
PV_READY_BARKWARD_IDS = (9, 10)


PV_CORE_STRIDE = 2 * FIFO_SIZE * TS


def calc_segment_idx(b_idx, s1_idx, segment_starts, actual_s1):
    """Return (segment_idx, segment_acc, cross_segment) for this Q-tile.

    The Q-tile occupies rows [s1_start, s1_end) = [s1_idx*TS, s1_idx*TS+TS)
    within batch b_idx.  Walking the cumulative segment ends once yields all
    three values:

      segment_idx   = index of the segment owning the tile (containing s1_start);
                      a row exactly on a boundary belongs to the next segment.
      segment_acc   = that segment's start offset (== own_start).
      cross_segment = how many segments the tile's rows span (>= 1).
    """
    seq_num = segment_starts.shape[1]
    s1_start = s1_idx * TS
    s1_end = pl.min(s1_start + TS, actual_s1)
    # --- owning segment (the one containing s1_start) ---
    segment_idx = 0
    segment_acc = 0  # start offset of the owning segment (== own_start)
    acc = 0
    for si in pl.range(seq_num):
        seg = pl.getval(segment_starts, b_idx * seq_num + si)
        if s1_start < acc + seg:
            segment_idx = si
            segment_acc = acc
            break
        acc = acc + seg
    # --- cross_segment ---
    # The owning segment always counts.  If it is the last segment, the tile
    # cannot spill into any following segment, and its last tile may be a padded
    # tail block (s1_end overshoots the real rows there) -- so force cross == 1.
    # Otherwise count how far the tile reaches into the following segments using
    # s1_end (a non-last owning segment always carries a full TS-row tile).
    cross_segment = 1
    if segment_idx < seq_num - 1:
        seg_end = segment_acc + pl.getval(segment_starts, b_idx * seq_num + segment_idx)
        for sj in pl.range(segment_idx + 1, seq_num):
            if s1_end > seg_end:
                cross_segment = cross_segment + 1
                seg_end = seg_end + pl.getval(segment_starts, b_idx * seq_num + sj)
            else:
                break
    return segment_idx, segment_acc, cross_segment


def seg_window(rule, gs, ge, r0, c0):
    """Template window (mi, mj) for ONE segment's rows on KV tile [c0, c0+TKV).

    Given a segment with global rows [gs, ge) and its rule, return the template
    anchor (mi, mj) such that loading ``mask[mi + tile_row, mj : mj + TKV]`` for
    tile_row in [gs-r0, ge-r0) reproduces that segment's visibility across the
    128 KV columns.  Because a segment sees *all strictly-previous* columns in
    full, each segment needs at most a single vertical cut (at its own start)
    plus (for diag) its own diagonal -- i.e. exactly one template primitive.

    This is the per-segment primitive of the variable-length loop (DESIGN 3.2);
    it generalises the fixed base+overlay windows to arbitrary/small segments.
    """
    mi = 128
    mj = 0
    diag_tile = r0 == c0
    if rule == 0:  # causal: diag tile -> lower-tri block; else visible
        if diag_tile:
            mi = 0
            mj = 0
        else:
            mi = 128
            mj = 0
    elif rule == 1:  # full: visible up to ge, then vertical cut
        if c0 + TKV <= ge:
            mi = 128
            mj = 0
        elif c0 >= ge:
            mi = 264
            mj = 136
        else:
            mi = 264
            mj = (c0 + TKV - ge) + 8
    else:  # rule == 2 diag: prefix cut at gs + own diagonal
        if diag_tile:
            p = gs - c0
            if p <= 0:
                mi = 136
                mj = 136
            else:
                # small segment: own_start falls inside the tile -> align this
                # segment's diagonal to the global main diagonal (DESIGN 3.2).
                mi = 136 - p
                mj = 136 - p
        else:
            if c0 + TKV <= gs:
                mi = 128
                mj = 0
            elif c0 >= gs:
                mi = 264
                mj = 136
            else:
                mi = 264
                mj = (c0 + TKV - gs) + 8
    return mi, mj


def calc_loop(start, end, work_id, b_idx, s1_o_acc, s1_size_acc, s2_size_acc, actual_seq_q, actual_seq_kv, n_dim):
    for _ in pl.range(start, end):
        actual_s1 = 0
        actual_s2 = 0
        if b_idx == 0:
            actual_s1 = pl.getval(actual_seq_q, b_idx)
            actual_s2 = pl.getval(actual_seq_kv, b_idx)
        else:
            actual_s1 = pl.getval(actual_seq_q, b_idx) - pl.getval(actual_seq_q, b_idx - 1)
            actual_s2 = pl.getval(actual_seq_kv, b_idx) - pl.getval(actual_seq_kv, b_idx - 1)
        s1o_size = s1_o_acc + (actual_s1 + TS - 1) // TS * n_dim
        if work_id >= s1o_size:
            s1_o_acc = s1o_size
            s1_size_acc = s1_size_acc + actual_s1
            s2_size_acc = s2_size_acc + actual_s2
            b_idx = b_idx + 1
            continue
        break
    return b_idx, s1_o_acc, s1_size_acc, s2_size_acc, actual_s1, actual_s2


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


def compute_qk(
    ctx, ki, sq_off, q, k, cur_q_slot, k_l1_db, left_db, right_db, acc_db, qk_vec_db, task_id, left2, right2, acc2
):
    # --- compute_qk inlined ---
    skv_off = ctx.s2SizeAcc + ctx.ki * TKV
    cur_k_slot = k_l1_db.next()
    qk_left = left_db.next()
    qk_right = right_db.next()
    qk_acc = acc_db.next()
    tmp1, tmp2, tmp3 = left2.next(), right2.next(), acc2.next()  # noqa: F841

    if ki == 0:
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
    sv_off = ctx.s2SizeAcc + ctx.ki * TKV
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=P_READY_FORWARD_IDS[ctx.task_id % 3])
    cur_v_slot = v_l1_db.next()
    # # current() advances p_mat_db cursor to stay in sync with Vec,
    cur_p_slot = p_mat_db.next()
    pv_left = left_db.next()
    pv_right = right_db.next()
    pv_acc = acc_db.next()
    tmp1, tmp2, tmp3 = left2.next(), right2.next(), acc2.next()  # noqa: F841

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
    mask,
    segment_starts,
    rules,
) -> None:
    """Softmax on KQ tile -> P. Includes cross-core sync."""
    seq_num = rules.shape[0]
    p_eid = ctx_p.task_id % FIFO_SIZE

    qk_slot = qk_vec_db.next()
    tile_nz = tile_nz_db.next()
    mask_buf = mask_db.next()
    q_idx_p = ctx_p.q_count % 3
    gmax_p = global_max[q_idx_p]
    gsum_p = global_sum[q_idx_p]
    exp_diff = exp_corr_db[ctx_p.task_id % 3]
    row_off = ctx_p.first_s1 * sub_id
    r0 = ctx_p.s1_idx * TS
    c0 = ctx_p.ki * TKV
    # ===== unified per-segment mask fill (coalesced seg_window loop) =====
    # One code path for every tile: walk the segments the Q-tile spans and, for
    # each, load its single template window (mi, mj) from seg_window into that
    # segment's rows.  Adjacent segments whose window matches (same mi/mj,
    # contiguous rows) are COALESCED into one load, so a run of fully-visible /
    # fully-masked segments (the common non-diagonal case) costs a single load
    # while a diagonal staircase still gets one load per distinct step.  This
    # subsumes the former base+overlay coordinate scheme exactly, with no
    # cross_segment ceiling.  mask_buf[j] == tile row (row_off + j); the template
    # row for tile row t is (mi + t).
    pl.set_validshape(mask_buf, [ctx_p.half_s1, ctx_p.s2_size])
    lo_tile = row_off
    hi_tile = row_off + ctx_p.half_s1
    # Recompute the owning segment's global start (gs == own_start) from
    # segment_idx; segment_acc is carried in ctx.
    gs = ctx_p.segment_acc
    # pending coalesced run: have==1 once a window is open; (p_mi, p_mj) its
    # template anchor; [p_rlo, p_rhi) its accumulated tile-row range.
    have = 0
    p_mi = 0
    p_mj = 0
    p_rlo = 0
    p_rhi = 0
    for k in pl.range(ctx_p.cross_segment):
        g = ctx_p.segment_idx + k
        ge = gs + pl.getval(segment_starts, ctx_p.b_idx * seq_num + g)
        rule_g = pl.getval(rules, g)
        rlo = pl.max(gs - r0, lo_tile)  # this seg's rows within sub-block
        rhi = pl.min(ge - r0, hi_tile)
        cnt = rhi - rlo
        if cnt > 0:
            mi, mj = seg_window(rule_g, gs, ge, r0, c0)
            # merge into the open run iff same window and rows continue it
            merged = 0
            if have == 1:
                if mi == p_mi:
                    if mj == p_mj:
                        if rlo == p_rhi:
                            merged = 1
            if merged == 1:
                p_rhi = rhi
            else:
                # window changed: flush the previous run, then open a new one
                if have == 1:
                    cntp = p_rhi - p_rlo
                    tmp = mask_buf[(p_rlo - lo_tile):, :]
                    pl.set_validshape(tmp, [cntp, ctx_p.s2_size])
                    pl.load(tmp, mask, [p_mi + p_rlo, p_mj])
                p_mi = mi
                p_mj = mj
                p_rlo = rlo
                p_rhi = rhi
                have = 1
        gs = ge
    # flush the final open run
    if have == 1:
        cntp = p_rhi - p_rlo
        tmp = mask_buf[(p_rlo - lo_tile):, :]
        pl.set_validshape(tmp, [cntp, ctx_p.s2_size])
        pl.load(tmp, mask, [p_mi + p_rlo, p_mj])
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_FORWARD_IDS[p_eid])
    if ctx_p.ki == 0:
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
    pl.insert(cur_p_slot, tile_nz, [row_off, 0])
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=P_READY_FORWARD_IDS[ctx_p.task_id % 3])


def compute_gu(ctx_gu, pv_vec_db, exp_corr_db, global_sum_buf, running_o, o_f16, o):
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_FORWARD_IDS[ctx_gu.task_id % 2])

    sub_id = pl.get_subblock_idx()
    row_off = ctx_gu.first_s1 * sub_id
    pv_slot = pv_vec_db.next()
    gsum_gu = global_sum_buf[ctx_gu.q_count % 3]
    exp_corr_gu = exp_corr_db[ctx_gu.task_id % 3]
    pl.set_validshape(running_o, [ctx_gu.half_s1, TD])
    pl.set_validshape(pv_slot, [ctx_gu.half_s1, TD])
    has_tail = 0
    if TAIL_D != 0:
        has_tail = 1
    if ctx_gu.ki == 0:
        pl.move(running_o, pv_slot)
    else:
        if ctx_gu.ki < ctx_gu.kv_loop - 1:
            flash_update_basic_vf(running_o, pv_slot, running_o, exp_corr_gu, ctx_gu.half_s1, has_tail)
        else:
            flash_update_last_basic_vf(running_o, pv_slot, running_o, exp_corr_gu, gsum_gu, ctx_gu.half_s1, has_tail)
            pl.set_validshape(o_f16, [ctx_gu.half_s1, TD])
            pl.cast(o_f16, running_o, mode=pl.RoundMode.CAST_ROUND)
            pl.store(o, o_f16, [ctx_gu.q_off + row_off, ctx_gu.n_idx, 0], order=[0, 2])

    pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_BARKWARD_IDS[ctx_gu.task_id % 2])
    if ctx_gu.ki == ctx_gu.kv_loop - 1:
        if ctx_gu.ki == 0:
            last_div_vf(running_o, running_o, gsum_gu, ctx_gu.half_s1, has_tail)
            pl.set_validshape(o_f16, [ctx_gu.half_s1, TD])
            pl.cast(o_f16, running_o, mode=pl.RoundMode.CAST_ROUND)
            pl.store(o, o_f16, [ctx_gu.q_off + row_off, ctx_gu.n_idx, 0], order=[0, 2])


# ================================================================
#  Kernel with NBuffer + auto_mutex
# ================================================================
@pl.jit(arch="3510", auto_mutex=True, compile_timeout=200)
def fa_tnd_with_mask_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    v: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    o: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    mask: pl.Tensor[[520, 520], pl.DT_UINT8],
    segment_starts: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    rules: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    actual_seq_q: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    actual_seq_kv: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    work_ranges: pl.Tensor[[pl.DYNAMIC, 2], pl.DT_INT32],
):
    n_dim = q.shape[1]
    batch = segment_starts.shape[0]
    seq_num = rules.shape[0]
    core_id = pl.get_block_idx() // pl.get_subblock_num()

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
                shape=[TS, TD],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Mat,
                valid_shape=[-1, -1],
                compact=1,
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
                shape=[TKV, TD],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Mat,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=MA3_V,
            mutex_ids=[4, 5],
        )

        left_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TS, TD],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Left,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=[0, 32768],
            mutex_ids=[6, 7],
        )
        right_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TD, TKV],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Right,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=[0, 32768],
            mutex_ids=[8, 9],
        )
        acc_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TS, TKV],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Acc,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=[0, 65536, 131072, 196608],
            mutex_ids=[10, 11, 12, 13],
        )
        left_db2 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TS, TKV],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Left,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=[0, 32768],
            mutex_ids=[6, 7],
        )
        right_db2 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TKV, TD],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Right,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=[0, 32768],
            mutex_ids=[8, 9],
        )
        acc_db2 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TS, TD],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Acc,
                valid_shape=[-1, -1],
                compact=1,
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
        s1_o_acc = 0
        ctx_arr = pl.struct_array(
            4, "CubeCtx", n_idx=0, qi=0, ki=0, task_id=0, s1SizeAcc=0, s2SizeAcc=0, s1_size=0, s2_size=0
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
                s1_o_acc = s1o_size
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
                s1o_size = s1_o_acc + (actual_s1 + TS - 1) // TS * n_dim
                if work_id >= s1o_size:
                    s1_o_acc = s1o_size
                    s1_size_acc = s1_size_acc + actual_s1
                    s2_size_acc = s2_size_acc + actual_s2
                    b_idx = b_idx + 1
                    continue
                break
            cur_b_s1o = (actual_s1 + TS - 1) // TS
            s1o_size = work_id - s1_o_acc
            n_idx = s1o_size // cur_b_s1o
            s1_idx = s1o_size % cur_b_s1o
            s1_size = pl.min(TS, actual_s1 - s1_idx * TS)

            sq_off = s1_size_acc + s1_idx * TS
            cur_q_slot = q_l1_db.next()
            kv_loop = s1_idx + 1
            kv_end = pl.min((s1_idx + 1) * TS, actual_s2)
            first_rule = pl.getval(rules, 0)
            if first_rule == 1:
                seg0 = pl.getval(segment_starts, b_idx * seq_num)
                if s1_idx * TS < seg0:
                    kv_loop = (seg0 + TS - 1) // TS
                    kv_end = pl.min(kv_loop * TS, actual_s2)
            drain = 0
            if work_id == work_end - 1:
                drain = 2
            for ki in pl.range(0, kv_loop + drain):
                if ki < kv_loop:
                    # Save current context
                    ctx_curr = ctx_arr[task_id % 4]
                    ctx_curr.task_id = task_id
                    ctx_curr.n_idx = n_idx
                    ctx_curr.ki = ki
                    ctx_curr.s1SizeAcc = s1_size_acc
                    ctx_curr.s2SizeAcc = s2_size_acc
                    ctx_curr.s1_size = s1_size
                    ctx_curr.s2_size = pl.min(TKV, kv_end - ki * TKV)

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
        s1_o_acc = 0
        segment_idx = 0

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
            segment_acc=0,
            segment_idx=0,
            cross_segment=0,
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
                s1_o_acc = s1o_size
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
                s1o_size = s1_o_acc + (actual_s1 + TS - 1) // TS * n_dim
                if work_id >= s1o_size:
                    s1_o_acc = s1o_size
                    s1_size_acc = s1_size_acc + actual_s1
                    s2_size_acc = s2_size_acc + actual_s2
                    b_idx = b_idx + 1
                    continue
                break
            cur_b_s1o = (actual_s1 + TS - 1) // TS
            s1o_size = work_id - s1_o_acc
            n_idx = s1o_size // cur_b_s1o
            s1_idx = s1o_size % cur_b_s1o
            segment_idx, segment_acc, cross_segment = calc_segment_idx(b_idx, s1_idx, segment_starts, actual_s1)
            s1_size = pl.min(TS, actual_s1 - s1_idx * TS)

            sq_acc = 0
            if b_idx > 0:
                sq_acc = pl.getval(actual_seq_q, b_idx - 1)
            q_off = sq_acc + s1_idx * TS
            kv_loop = s1_idx + 1
            kv_end = pl.min((s1_idx + 1) * TS, actual_s2)
            first_rule = pl.getval(rules, 0)
            if first_rule == 1:
                seg0 = pl.getval(segment_starts, b_idx * seq_num)
                if s1_idx * TS < seg0:
                    kv_loop = (seg0 + TKV - 1) // TKV
                    kv_end = pl.min(kv_loop * TS, actual_s2)

            # Pipeline flush folded into the ki loop: on the LAST work item extend by
            # the max consumer lag (compute_gu = 3) with the producer gated off.
            # compute_p (lag 1) drains once -> gate to ki <= kv_loop; compute_gu
            # (lag 3) drains on all extra steps.  No separate epilogue.
            drain = 0
            if work_id == work_end - 1:
                drain = 3
            for ki in pl.range(0, kv_loop + drain):
                if ki < kv_loop:
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

                    # Mask coordinates are computed entirely in compute_p by the
                    # coalesced seg_window loop; the producer only forwards the
                    # segment span (owning index + how many segments the rows cross).
                    ctx_curr.segment_acc = segment_acc
                    ctx_curr.segment_idx = segment_idx
                    ctx_curr.cross_segment = cross_segment

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
                            mask,
                            segment_starts,
                            rules,
                        )

                # ===== compute_gu (consumer, lag 3: every step incl. all drains) =====
                if task_id > 2:
                    ctx_gu = ctx_arr[(task_id + 1) % 4]
                    compute_gu(ctx_gu, pv_vec_db, exp_corr_db, global_sum, running_o, o_f16, o)

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


def build_segment_mask(segment, rules, device):
    """Boolean attention mask (True = masked / -inf) of shape [L, L] for one
    batch, generalised for arbitrary segments / rules.

    Unified model (matches the kernel's seg_window fill):
      * every segment attends to *all strictly-previous* segments in full;
      * within its own segment it applies its rule:
          rule 0 = causal (lower-triangular), 1 = full (all), 2 = diag (only the
          main diagonal);
      * the main diagonal is always visible.
    """
    seq_len = sum(segment)
    mask = torch.ones(seq_len, seq_len, device=device)  # 1 = masked
    starts = [0]
    for s in segment:
        starts.append(starts[-1] + s)
    for g, sz in enumerate(segment):
        s0, s1 = starts[g], starts[g + 1]
        # see all strictly-previous segments in full
        if s0 > 0:
            mask[s0:s1, :s0] = 0
        rule = rules[g]
        if rule == 0:
            # causal within own segment (visible where col <= row)
            # Build the lower-triangular block and assign it with a basic slice.
            # NB: do NOT use `mask[s0:s1, s0:s1][tri] = 0` -- that chained boolean
            # write into a sliced view does not propagate to the base tensor for
            # a non-zero offset (s0 > 0) on the NPU backend, leaving the block as
            # diag instead of causal.
            rr = torch.arange(sz, device=device).view(-1, 1)
            cc = torch.arange(sz, device=device).view(1, -1)
            mask[s0:s1, s0:s1] = (cc > rr).to(mask.dtype)  # 1 = masked (col > row)
        elif rule == 1:  # full within own segment
            mask[s0:s1, s0:s1] = 0
        # rule == 2 (diag): only the main diagonal (filled below)
    mask.diagonal(0).fill_(0)
    return mask.bool()


def flash_attention_ref_tnd(q_tnd, k_tnd, v_tnd, seq_q_list, seq_kv_list, d, segments, rules):
    """Reference attention operating on TND tensors, per-batch."""
    q_tnd = q_tnd.cpu()
    k_tnd = k_tnd.cpu()
    v_tnd = v_tnd.cpu()
    # seq_q_list/seq_kv_list/segments/rules may be passed as plain Python lists
    # (from test_cfg) or as tensors; normalize tensors to cpu, leave lists as-is.
    seq_q_list = seq_q_list.cpu() if torch.is_tensor(seq_q_list) else seq_q_list
    seq_kv_list = seq_kv_list.cpu() if torch.is_tensor(seq_kv_list) else seq_kv_list
    segments = segments.cpu() if torch.is_tensor(segments) else segments
    rules = rules.cpu() if torch.is_tensor(rules) else rules
    scale_val = 1.0 / math.sqrt(d)
    n = q_tnd.shape[1]
    o_tnd = torch.zeros_like(q_tnd)
    q_off = 0
    kv_off = 0
    for sq, skv, segment in zip(seq_q_list, seq_kv_list, segments):
        mask = build_segment_mask(segment, rules, q_tnd.device)
        for ni in range(n):
            qi = q_tnd[q_off:q_off + sq, ni, :].float()
            ki = k_tnd[kv_off:kv_off + skv, ni, :].float()
            vi = v_tnd[kv_off:kv_off + skv, ni, :].float()
            qk = torch.matmul(qi, ki.T)
            qk = qk * scale_val
            qk.masked_fill_(mask, NEG_INF)
            attn = torch.softmax(qk, dim=-1)
            pv = torch.matmul(attn, vi)
            o_tnd[q_off:q_off + sq, ni, :] = pv.half()
        q_off += sq
        kv_off += skv
    return o_tnd


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_fa_perf_nbuf():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(42)

    for test_cfg in [
        # small diag middle segments (size 5) -> diagonal tile spans 11 segments
        # (cross_segment==11), exercises the variable-length overlay loop.
        (None, None, [[2200, 8, 300, 5, 200, 110, 5, 400, 5, 5, 1024]], [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2], 1, TD, 28),
        # ===== variable-length overlay loop: generalization suite =====
        # All validated against golden by scratchpad/matrix_test.py (per-core,
        # per-basic-block replay, 0 mismatch).  Each exercises the diagonal-tile
        # loop; segments use +/* expressions (the list is built at runtime).
        # --- first_rule=0, diag middles: size 1 / 7 / mixed, big cross ---
        (None, None, [[2048, 8] + [1] * 40 + [1024]], [0, 1] + [2] * 40 + [2], 1, TD, 28),
        (None, None, [[1024, 8] + [1] * 120 + [1024]], [0, 1] + [2] * 120 + [2], 1, TD, 28),
        (None, None, [[2048, 8] + [7] * 40 + [936]], [0, 1] + [2] * 40 + [2], 1, TD, 28),
        (None, None, [[1280, 8, 3, 7, 1, 5, 2, 9, 1, 4, 6, 1, 8, 2, 5, 3, 1000]], [0, 1] + [2] * 14 + [2], 1, TD, 28),
        # --- non-128 total (unalign tail) with the loop mid-sequence ---
        (None, None, [[2200, 8] + [5] * 8 + [967]], [0, 1] + [2] * 8 + [2], 1, TD, 28),
        (None, None, [[1536, 5, 5, 5, 5, 1000]], [1, 0, 0, 0, 0, 2], 1, TD, 28),
        # --- causal middles (loop still fires on diag tile, rule-0 windows) ---
        (None, None, [[2048, 8] + [5] * 10 + [1024]], [0, 1] + [0] * 10 + [2], 1, TD, 28),
        (None, None, [[1536] + [5] * 10 + [1024]], [1] + [2] * 10 + [2], 1, TD, 28),
        (None, None, [[1024] + [1] * 60 + [1000]], [1] + [2] * 60 + [2], 1, TD, 28),
        (None, None, [[1536] + [5] * 12 + [1000]], [1] + [0] * 12 + [2], 1, TD, 28),
        # --- loop ON the final PARTIAL tile (s2_size<128, half_s1<64) ---
        (None, None, [[2048, 8] + [5] * 32], [0, 1] + [2] * 32, 1, TD, 28),  # tail 40
        (None, None, [[2048, 8] + [5] * 44], [0, 1] + [2] * 44, 1, TD, 28),  # tail 100
        (None, None, [[1536] + [3] * 40], [1] + [2] * 40, 1, TD, 28),  # fr1 tail 120
        # --- multi-head (n>1) and multi-batch (shared SeqNum/rules) ---
        (None, None, [[2200, 8] + [5] * 8 + [1024]], [0, 1] + [2] * 8 + [2], 4, TD, 28),
        (
            None,
            None,
            [[2200, 8] + [5] * 8 + [1024], [1536, 8, 7, 3, 1, 5, 2, 9, 4, 6, 1000]],
            [0, 1] + [2] * 8 + [2],
            1,
            TD,
            28,
        ),
        # ----- HETEROGENEOUS-RULE middles (causal+diag interleaved in one seq) -----
        # Stresses per-segment seg_window rule dispatch AND coalescing across
        # DIFFERENT windows (merge equal runs, break+restart where they differ).
        # All validated against golden by scratchpad/matrix_test.py (0 mismatch).
        (None, None, [[2048, 8] + [5] * 8 + [1024]], [0, 1] + [0, 2] * 4 + [2], 1, TD, 28),
        (None, None, [[1536, 8, 3, 7, 1, 9, 2, 5, 4, 6, 1000]], [0, 1] + [2, 0, 2, 0, 2, 0, 2, 0] + [2], 1, TD, 28),
        (None, None, [[2048, 8] + [1] * 40 + [1024]], [0, 1] + [0, 2] * 20 + [2], 1, TD, 28),
        (None, None, [[2048, 8] + [3] * 18 + [1000]], [0, 1] + [2] * 6 + [0] * 6 + [2] * 6 + [2], 1, TD, 28),
        (None, None, [[1536] + [5] * 10 + [1024]], [1] + [0, 2] * 5 + [2], 1, TD, 28),
        (None, None, [[1280, 7, 1, 5, 2, 9, 3, 1, 6, 4, 1000]], [1] + [2, 0, 2, 0, 2, 0, 2, 0, 2] + [2], 1, TD, 28),
        (None, None, [[2200, 8] + [5] * 8 + [967]], [0, 1] + [0, 2] * 4 + [2], 1, TD, 28),
        (None, None, [[2048, 8] + [5] * 8 + [1024]], [0, 1] + [0, 2] * 4 + [2], 4, TD, 28),
        (
            None,
            None,
            [[2048, 8] + [5] * 8 + [1024], [1536, 8, 3, 7, 1, 9, 2, 5, 4, 6, 1000]],
            [0, 1] + [0, 2] * 4 + [2],
            1,
            TD,
            28,
        ),
        # ===== end generalization suite =====
        # ===== supplementary regression (board-verified, max|diff|=0.000488) =====
        # original cross>=4 cases, now routed through the variable loop (n=8):
        (None, None, [[1800, 8, 100, 1500], [1500, 8, 300, 1200]], [0, 1, 0, 2], 8, TD, 28),
        (None, None, [[1548, 8, 100, 100, 100, 1088]], [0, 1, 2, 2, 2, 2], 8, TD, 28),
        (
            None,
            None,
            [
                [2200, 8, 200, 1048],
                [1800, 8, 100, 1164],
                [2440, 8, 200, 2088],
                [1600, 8, 600, 2016],
                [3300, 8, 200, 1356],
                [1700, 8, 300, 2216],
                [1780, 8, 700, 1224],
                [2048, 8, 500, 1924],
            ],
            [0, 1, 2, 2],
            8,
            TD,
            28,
        ),
        # fixed first_rule==1 causal-middle pattern in new combinations:
        (
            None,
            None,
            [[1536] + [5] * 8 + [1000], [1536, 7, 3, 5, 1, 9, 2, 4, 6, 1000]],
            [1] + [0] * 8 + [2],
            1,
            TD,
            28,
        ),  # 2-batch
        (None, None, [[1536] + [5] * 8 + [1000]], [1] + [0] * 8 + [2], 4, TD, 28),  # n=4
        (None, None, [[1536] + [3] * 40], [1] + [0] * 39 + [2], 1, TD, 28),  # tail staircase
        (None, None, [[1536] + [1] * 50 + [1000]], [1] + [0] * 50 + [2], 1, TD, 28),  # size-1 x50
        (None, None, [[2048, 8] + [5] * 32], [0, 1] + [0] * 31 + [2], 1, TD, 28),  # fr0 causal tail
        # ===== end supplementary regression =====
        ([600], [600], [[256, 8, 128, 208]], [0, 1, 2, 2], 2, TD, 8),
        ([3584], [3584], [[1536, 8, 800, 1240]], [0, 1, 2, 2], 1, TD, 28),
        ([3712], [3712], [[1536, 8, 800, 1368]], [0, 1, 2, 2], 1, TD, 28),
        ([2048, 2048], [2048, 2048], [[1536, 8, 200, 304], [1536, 8, 180, 324]], [0, 1, 2, 2], 1, TD, 28),
        ([639], [639], [[256, 8, 128, 247]], [0, 1, 2, 2], 1, TD, 28),  # tail 127 (unalign)
        ([576], [576], [[256, 8, 128, 184]], [0, 1, 2, 2], 1, TD, 28),  # tail 64  (unalign64 boundary)
        ([520], [520], [[256, 8, 128, 128]], [0, 1, 2, 2], 1, TD, 28),  # tail 8   (tiny, unalign64)
        ([612], [612], [[256, 8, 128, 220]], [0, 1, 0, 2], 1, TD, 28),  # tail 100, causal middle (unalign)
        # tail tile STRADDLES a segment boundary (cross2) under non-128: last seg
        # is only 48 rows, so rows [512,576) span seg2(diag)->seg3(diag-last).
        ([576], [576], [[256, 8, 264, 48]], [0, 1, 2, 2], 1, TD, 28),
        # first_rule==1 (merged full leading), non-128 tails:
        ([583], [583], [[148, 120, 120, 195]], [1, 2, 2, 2], 1, TD, 28),  # tail 71 (unalign)
        ([550], [550], [[148, 120, 120, 162]], [1, 2, 2, 2], 1, TD, 28),  # tail 38 (unalign64)
        # multi-batch, different non-128 tails per batch (shared SeqNum/rules):
        ([576, 639], [576, 639], [[256, 8, 128, 184], [256, 8, 128, 247]], [0, 1, 2, 2], 1, TD, 28),
        # #######################JD CASE#############################
        (None, None, [[1600, 8, 200, 1200]], [0, 1, 2, 2], 8, TD, 28),
        (None, None, [[1600, 8, 200, 1200]], [0, 1, 0, 2], 8, TD, 28),
        (None, None, [[1600, 8, 200, 1200], [1700, 8, 300, 1024]], [0, 1, 2, 2], 8, TD, 28),
        (None, None, [[1800, 8, 100, 1500], [1500, 8, 300, 1200]], [0, 1, 0, 2], 8, TD, 28),
        (
            None,
            None,
            [[1600, 8, 200, 1200], [1700, 8, 300, 1024], [1680, 8, 200, 1280], [2000, 8, 700, 2048]],
            [0, 1, 2, 2],
            8,
            TD,
            28,
        ),
        (
            None,
            None,
            [[3200, 8, 200, 1200], [2300, 8, 400, 1800], [2080, 8, 200, 1800], [1700, 8, 100, 1024]],
            [0, 1, 0, 2],
            8,
            TD,
            28,
        ),
        (
            None,
            None,
            [
                [2200, 8, 200, 1024],
                [1700, 8, 100, 1100],
                [2440, 8, 200, 2048],
                [1600, 8, 600, 1900],
                [3300, 8, 200, 1300],
                [1700, 8, 300, 2100],
                [1780, 8, 700, 1200],
                [2048, 8, 500, 1800],
            ],
            [0, 1, 2, 2],
            8,
            TD,
            28,
        ),
        (
            None,
            None,
            [
                [2200, 8, 200, 1024],
                [1700, 8, 300, 1100],
                [2440, 8, 200, 2048],
                [1600, 8, 600, 1800],
                [3300, 8, 200, 1300],
                [1700, 8, 300, 2048],
                [1780, 8, 700, 1024],
                [2048, 8, 500, 1800],
            ],
            [0, 1, 0, 2],
            8,
            TD,
            28,
        ),
        (None, None, [[1600, 200, 1024]], [1, 0, 2], 8, TD, 28),
        (None, None, [[1600, 200, 1000], [2000, 300, 1100]], [1, 0, 2], 8, TD, 28),
        (
            None,
            None,
            [[1600, 200, 1024], [1700, 300, 1600], [1680, 200, 1200], [2000, 700, 2400]],
            [1, 0, 2],
            8,
            TD,
            28,
        ),
        (
            None,
            None,
            [
                [2200, 200, 1024],
                [1700, 300, 1100],
                [2440, 200, 2048],
                [1600, 400, 1800],
                [2200, 200, 1300],
                [1700, 300, 2048],
                [1680, 300, 1024],
                [2048, 700, 1800],
            ],
            [1, 0, 2],
            8,
            TD,
            28,
        ),
        (None, None, [[1600, 8, 10, 1200]], [0, 1, 2, 2], 8, TD, 28),
        (None, None, [[1600, 8, 8, 12, 1200], [1700, 8, 6, 15, 1024]], [0, 1, 0, 0, 2], 8, TD, 28),
        (
            None,
            None,
            [
                [1600, 8, 5, 6, 7, 8, 1200],
                [1700, 8, 10, 12, 13, 15, 1024],
                [1680, 8, 10, 12, 13, 15, 1280],
                [2000, 8, 13, 14, 15, 15, 2048],
            ],
            [0, 1, 2, 2, 2, 2, 2],
            8,
            TD,
            28,
        ),
        (
            None,
            None,
            [
                [2200, 8] + [5] * 8 + [1024],
                [1700, 8, 5, 5, 5, 5, 5, 5, 5, 10, 1100],
                [2440, 8, 5, 5, 5, 5, 5, 5, 5, 12, 2048],
                [1600, 8, 5, 15, 5, 5, 5, 5, 5, 5, 1900],
                [3300, 8, 5, 5, 10, 10, 5, 5, 5, 5, 1300],
                [1700, 8, 15, 15, 5, 5, 5, 5, 5, 5, 2100],
                [1780, 8, 15, 15, 15, 5, 5, 5, 5, 5, 1200],
                [2048, 8, 15, 15, 15, 15, 5, 5, 5, 5, 1800],
            ],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            8,
            TD,
            28,
        ),
        (None, None, [[1600, 8] + [5] * 1 + [1200]], [0, 1, 2, 2], 8, TD, 28),
        (None, None, [[1600, 8] + [5] * 2 + [1200]], [0, 1] + [2] * 2 + [2], 8, TD, 28),
        (None, None, [[1600, 8] + [5] * 4 + [1200]], [0, 1] + [2] * 4 + [2], 8, TD, 28),
        (None, None, [[1600, 8] + [5] * 8 + [1200]], [0, 1] + [2] * 8 + [2], 8, TD, 28),
        (None, None, [[1600, 8] + [5] * 16 + [1200]], [0, 1] + [2] * 16 + [2], 8, TD, 28),
        (None, None, [[1600, 8] + [5] * 32 + [1200]], [0, 1] + [2] * 32 + [2], 8, TD, 28),
        (None, None, [[1600, 8] + [5] * 64 + [1200]], [0, 1] + [2] * 64 + [2], 8, TD, 28),
        (None, None, [[1600, 8] + [5] * 128 + [1200]], [0, 1] + [2] * 128 + [2], 8, TD, 28),
        (None, None, [[1600, 8] + [5] * 256 + [1200]], [0, 1] + [2] * 256 + [2], 8, TD, 28),
        (None, None, [[1600, 8] + [5] * 512 + [1200]], [0, 1] + [2] * 512 + [2], 8, TD, 28),
        (None, None, [[1600, 8] + [5] * 1024 + [1200]], [0, 1] + [2] * 1024 + [2], 8, TD, 28),
        # ======================================================================= #
        (None, None, [[1600, 200, 1024]], [1, 2, 2], 8, TD, 28),
        (None, None, [[1600, 200, 1000], [2000, 300, 1100]], [1, 2, 2], 8, TD, 28),
        (
            None,
            None,
            [[1600, 200, 1024], [1700, 300, 1600], [1680, 200, 1200], [2000, 700, 2400]],
            [1, 2, 2],
            8,
            TD,
            28,
        ),
        (
            None,
            None,
            [
                [2200, 200, 1024],
                [1700, 300, 1100],
                [2440, 200, 2048],
                [1600, 400, 1800],
                [2200, 200, 1300],
                [1700, 300, 2048],
                [1680, 300, 1024],
                [2048, 700, 1800],
            ],
            [1, 2, 2],
            8,
            TD,
            28,
        ),
        ####################################################################################
        ####################################################################################
        # NB: each batch's sum(segment) must equal its seq length and be a
        # --- baseline: 4 segments [0,1,2,2], 8 batches (128-aligned) ---
        (
            [3456, 3072, 4736, 4224, 4864, 4224, 3712, 4480],
            [3456, 3072, 4736, 4224, 4864, 4224, 3712, 4480],
            [
                [2200, 8, 200, 1048],  # 1024 + 24
                [1800, 8, 100, 1164],  # 1100 + 36
                [2440, 8, 200, 2088],  # 2048 + 40
                [1600, 8, 600, 2016],  # 1900 + 116
                [3300, 8, 200, 1356],  # 1300 + 56
                [1700, 8, 300, 2216],  # 2100 + 116
                [1780, 8, 700, 1224],  # 1200 + 24
                [2048, 8, 500, 1924],
            ],
            [0, 1, 2, 2],
            8,
            TD,
            28,
        ),
        # --- 5 segments, diag middle: seg0 tail tile spans seg0+seg1+seg2 (cross3) ---
        ([2816], [2816], [[1548, 8, 120, 120, 1020]], [0, 1, 2, 2, 2], 8, TD, 28),
        # --- 6 segments, diag middle: seg0 tail tile spans 4 segments (cross4) ---
        ([2944], [2944], [[1548, 8, 100, 100, 100, 1088]], [0, 1, 2, 2, 2, 2], 8, TD, 28),
        # --- 7 segments, diag middle (more middle sub-segments) ---
        ([3072], [3072], [[1536, 8, 120, 120, 120, 120, 1048]], [0, 1, 2, 2, 2, 2, 2], 8, TD, 28),
        # --- 6 segments, SMALL diag middles: a middle diag tile spans 3 segments
        #     (diag->diag->diag, exercises the cross>=3 second overlay) ---
        ([2944], [2944], [[1536, 8, 110, 110, 110, 1070]], [0, 1, 2, 2, 2, 2], 8, TD, 28),
        # --- 5 segments, causal middle (the all-0 variant) ---
        ([2816], [2816], [[1548, 8, 120, 120, 1020]], [0, 1, 0, 0, 2], 8, TD, 28),
        # --- 7 segments, causal middle (all-0 variant, more sub-segments) ---
        ([2944], [2944], [[1536, 8, 100, 100, 100, 100, 1000]], [0, 1, 0, 0, 0, 0, 2], 8, TD, 28),
        # --- 3 segments, NO middle part: [causal, full(8), diag-last] ---
        ([2944], [2944], [[1280, 8, 1656]], [0, 1, 2], 8, TD, 28),
        # --- 2 batches, 5-segment diag middle (different segment offsets) ---
        (
            [2816, 2944],
            [2816, 2944],
            [[1548, 8, 120, 120, 1020], [1676, 8, 120, 120, 1020]],
            [0, 1, 2, 2, 2],
            8,
            TD,
            28,
        ),
        ([2816, 2944], [2816, 2944], [[1556, 120, 120, 1020], [1684, 120, 120, 1020]], [1, 2, 2, 2], 8, TD, 28),
        ([1408], [1408], [[148, 120, 120, 1020]], [1, 2, 2, 2], 8, TD, 28),
        # --- first_rule==1, large full leading + CAUSAL middle (all-0) ---
        ([2816], [2816], [[1556, 120, 120, 1020]], [1, 0, 0, 2], 8, TD, 28),
        # --- first_rule==1, 5 segments, diag middle ---
        ([1536], [1536], [[148, 120, 120, 120, 1028]], [1, 2, 2, 2, 2], 8, TD, 28),
        # --- first_rule==1, NO middle part: [full(merged), diag-last, diag-last]? ---
        #     i.e. full leading directly followed by diag last (one diag middle).
        ([1280], [1280], [[148, 120, 1012]], [1, 2, 2], 8, TD, 28),
        # --- first_rule==1, full leading CROSS3: small (100) next segments so the
        #     seg0 tail tile spans seg0+seg1+seg2 -> two overlays (diag) ---
        ([1408], [1408], [[148, 100, 100, 1060]], [1, 2, 2, 2], 8, TD, 28),
        # --- first_rule==1, full leading CROSS3 into CAUSAL middles ---
        ([1408], [1408], [[148, 100, 100, 1060]], [1, 0, 0, 2], 8, TD, 28),
        # --- first_rule==1, 6 segments, large full leading + diag middles ---
        ([2944], [2944], [[1556, 120, 120, 120, 120, 908]], [1, 2, 2, 2, 2, 2], 8, TD, 28),
    ]:
        seq_q_list, seq_kv_list, segments, rules, n, d, num_cores = test_cfg
        if seq_q_list is None:
            seq_q_list = [sum(sublist) for sublist in segments]
            seq_kv_list = seq_q_list
        b = len(seq_q_list)
        tq = sum(seq_q_list)
        tkv = sum(seq_kv_list)
        logging.info(
            "\nFA-TND-DN-A5 (b=%s, seq_q=%s, seq_kv=%s, n=%s, d=%s) rules=%s cores=%s",
            b,
            seq_q_list,
            seq_kv_list,
            n,
            d,
            rules,
            num_cores,
        )

        q = torch.rand((tq, n, d), device="cpu", dtype=torch.float16)
        k = torch.rand((tkv, n, d), device="cpu", dtype=torch.float16)
        v = torch.rand((tkv, n, d), device="cpu", dtype=torch.float16)
        o = torch.zeros((tq, n, d), device="cpu", dtype=torch.float16)

        mask = torch.ones(520, 520).to(torch.uint8).cpu()
        lower_tri_mask = torch.tril(torch.ones(128, 128), diagonal=0).bool().cpu()
        mask[:128, :128][lower_tri_mask] = 0
        mask.diagonal(0).fill_(0)
        mask[128:136 + 256, :136] = 0
        mask[136 + 256:, :136 + 256] = 0
        actual_q_len = list(itertools.accumulate(seq_q_list))
        actual_kv_len = list(itertools.accumulate(seq_kv_list))
        total_work = sum((x + 127) // 128 for x in seq_q_list) * n
        work_ranges = torch.zeros((num_cores, 2), device="cpu", dtype=torch.int32)
        work_per_core = total_work // num_cores
        last = total_work % num_cores
        idx = 0
        for core in range(num_cores):
            work_ranges[core, 0] = idx
            sec = idx + work_per_core if core >= last else idx + work_per_core + 1
            work_ranges[core, 1] = min(sec, total_work)
            idx = sec
        # h2d
        actual_seq_q = torch.tensor(actual_q_len, device=device, dtype=torch.int32)
        actual_seq_kv = torch.tensor(actual_kv_len, device=device, dtype=torch.int32)
        segments_dev = torch.tensor(segments, device=device, dtype=torch.int32)
        rules_dev = torch.tensor(rules, device=device, dtype=torch.int32)
        q = q.to(device)
        k = k.to(device)
        v = v.to(device)
        o = o.to(device)
        mask = mask.to(device)
        work_ranges = work_ranges.to(device)
        actual_num_cores = min(num_cores, total_work)
        fa_tnd_with_mask_kernel[None, actual_num_cores](
            q, k, v, o, mask, segments_dev, rules_dev, actual_seq_q, actual_seq_kv, work_ranges
        )
        torch.npu.synchronize()

        o_ref = flash_attention_ref_tnd(q, k, v, seq_q_list, seq_kv_list, d, segments, rules)
        o = o.cpu()
        diff = (o - o_ref).abs().max().item()
        logging.info("  max|diff|=%.4f", diff)
        torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
        logging.info("  PASS")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("FA perf with NBuffer + auto_mutex (QK_PRELOAD=%s, FIFO=%s)", QK_PRELOAD, FIFO_SIZE)
    logging.info("%s", '=' * 60)
    test_fa_perf_nbuf()  # uncomment on A5 NPU
    logging.info("\nParse test passed!")
