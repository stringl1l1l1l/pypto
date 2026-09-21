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

"""QuantLightningIndexer kernel using VF API (direct VF instructions).

Pipeline:
  1. Cube: Q @ K^T -> scores_workspace[DT_FP32]
  2. Vector1: Load weight/qScale/kScale -> Mul + cast to DT_UINT16 sortable key
  3. TopK VF: Histogram-based sorting using VF instructions
     - HistogramsHigh: DT_UINT16 high byte histogram
     - FindHighTargetBin: Compare + Squeeze + StoreUnAlign
     - HistogramsLow: Low byte histogram (filtered by MSB)
     - FindKth: Find kth value
     - CollectIndices: Gather GT + EQ indices

Usage:
    python3 python/tests/st/pypto_pro/frontend/lightning_indexer/test_quant_lightning_indexer_vf.py
"""

import logging
import os

import numpy as np
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
import pytest
import torch

import pypto

logging.basicConfig(level=logging.INFO, format="%(message)s")


# ================================================================
# ================================================================
S2_BASE_SIZE = 128
S1_BASE_SIZE = 4
HEAD_DIM = 128
M_BASE_SIZE = 256  # L1/L0 buffer pre-allocation (max mBaseSize)
TRUNK_LEN = 16384
TOPK_ALIGN_MAX = 2048
TOPK_OUTPUT_BUF_LEN = TRUNK_LEN + TOPK_ALIGN_MAX
TOPK_OUTPUT_BUF_BYTES = TOPK_OUTPUT_BUF_LEN * 2

TS = S2_BASE_SIZE
TD = HEAD_DIM
TM = S1_BASE_SIZE

# Dynamic dimensions (passed at runtime)

# TopK VF address offsets (relative to va_after_v1)
TK_HIST_HIGH_OFF = 0
TK_HIST_LOW_OFF = TK_HIST_HIGH_OFF + 1024
TK_SQZ_IDX_OFF = TK_HIST_LOW_OFF + 1024
TK_IDX_HIGH_OFF = TK_SQZ_IDX_OFF + 1024
TK_NK_VALUE_OFF = TK_IDX_HIGH_OFF + 256
TK_KTH_VALUE_OFF = TK_NK_VALUE_OFF + 256
TK_OUTPUT_IDX_OFF = TK_KTH_VALUE_OFF + 256

# Cross-core event IDs (ping-pong)
CUBE_TO_VEC_0 = 8
CUBE_TO_VEC_1 = 9
VEC_TO_CUBE_0 = 10
VEC_TO_CUBE_1 = 11

# ================================================================
# reduce_and_cast_sk_vf: reduce 8 heads from mm1_vec [m_half, TS] -> [sq_rows_per_sub, TS] u16
# ================================================================


@pl.vector_function
def reduce_and_cast_sk_vf(
    mm1_vec_tile, wq_tile, kscale_tile, score_u16_sk_tile, gsize: pl.DT_INT64, wq_col_pad: pl.DT_INT64
):
    """Reduce mm1_vec [m_half, TS] DT_FP32 across gsize heads -> [2, 128] u16 sortable key.

    mm1_vec_tile: [m_half, TS] — dual_split_m result with all TS columns per row.
    VF reads lo half (cols 0..TS/2) at offset h*TS, hi half (cols TS/2..TS) at h*TS+TS/2.

    wq_tile already contains weight*qscale (precomputed outside the sk loop).
    Both rows broadcast wq[h] from UB via BRC_B32 each iteration.
    Single accumulator per (row, half): Mul+Add, no dual-accumulator merge step.
    """
    preg_b32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_u16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)

    # ---- FloatToSortableKey constants ----
    sign_mask = vf.full(0x8000, dtype=pl.DT_UINT16)
    all_one = vf.full(0xFFFF, dtype=pl.DT_UINT16)
    zero_u16 = vf.full(0, dtype=pl.DT_UINT16)
    nan_mask = vf.full(0x7FC0, dtype=pl.DT_UINT16)
    zero_fp32 = vf.full(0, dtype=pl.DT_FP32)

    # ---- kscale (shared across both rows) ----
    ks_reg_0 = vf.load_align(kscale_tile, 0)
    ks_reg_1 = vf.load_align(kscale_tile, 64)

    # ---- accumulators: first head iteration writes directly (no zero-init add) ----
    # ---- working registers ----
    dst_stride = 128  # dstStride in elements (SLayout=512B / 4B = 128)
    nd1_off = gsize * 2 * dst_stride
    # ---- head 0: mul -> sum directly (skip 4x duplicate-zero + add) ----
    qk_reg_0 = vf.load_align(mm1_vec_tile, 0)
    qk_reg_1 = vf.load_align(mm1_vec_tile, nd1_off)
    qk_reg_0 = vf.max(qk_reg_0, zero_fp32, preg_b32)
    qk_reg_1 = vf.max(qk_reg_1, zero_fp32, preg_b32)
    wbrc0_reg = vf.load_align(wq_tile, 0, dist=pl.LoadDist.BRC_B32)
    sum0_r0 = vf.mul(qk_reg_0, wbrc0_reg, preg_b32)
    sum1_r0 = vf.mul(qk_reg_1, wbrc0_reg, preg_b32)
    qk_reg_0 = vf.load_align(mm1_vec_tile, gsize * dst_stride)
    qk_reg_1 = vf.load_align(mm1_vec_tile, nd1_off + gsize * dst_stride)
    qk_reg_0 = vf.max(qk_reg_0, zero_fp32, preg_b32)
    qk_reg_1 = vf.max(qk_reg_1, zero_fp32, preg_b32)
    wbrc1_reg = vf.load_align(wq_tile + wq_col_pad, 0, dist=pl.LoadDist.BRC_B32)
    sum0_r1 = vf.mul(qk_reg_0, wbrc1_reg, preg_b32)
    sum1_r1 = vf.mul(qk_reg_1, wbrc1_reg, preg_b32)

    # ---- head loop: h = 1..gsize-1, both rows per iteration ----
    for h in pl.range(1, gsize):
        qk_reg_0 = vf.load_align(mm1_vec_tile, h * dst_stride)
        qk_reg_1 = vf.load_align(mm1_vec_tile, nd1_off + h * dst_stride)
        qk_reg_0 = vf.max(qk_reg_0, zero_fp32, preg_b32)
        qk_reg_1 = vf.max(qk_reg_1, zero_fp32, preg_b32)
        wbrc0_reg = vf.load_align(wq_tile, h, dist=pl.LoadDist.BRC_B32)
        sum0_r0 = vf.mul_add_dst(qk_reg_0, wbrc0_reg, preg_b32)
        sum1_r0 = vf.mul_add_dst(qk_reg_1, wbrc0_reg, preg_b32)
        qk_reg_0 = vf.load_align(mm1_vec_tile, (gsize + h) * dst_stride)
        qk_reg_1 = vf.load_align(mm1_vec_tile, nd1_off + (gsize + h) * dst_stride)
        qk_reg_0 = vf.max(qk_reg_0, zero_fp32, preg_b32)
        qk_reg_1 = vf.max(qk_reg_1, zero_fp32, preg_b32)
        wbrc1_reg = vf.load_align(wq_tile + wq_col_pad, h, dist=pl.LoadDist.BRC_B32)
        sum0_r1 = vf.mul_add_dst(qk_reg_0, wbrc1_reg, preg_b32)
        sum1_r1 = vf.mul_add_dst(qk_reg_1, wbrc1_reg, preg_b32)

    # ---- apply kscale ----
    sum0_r0 = vf.mul(sum0_r0, ks_reg_0, preg_b32)
    sum1_r0 = vf.mul(sum1_r0, ks_reg_1, preg_b32)
    sum0_r1 = vf.mul(sum0_r1, ks_reg_0, preg_b32)
    sum1_r1 = vf.mul(sum1_r1, ks_reg_1, preg_b32)

    # ---- cast + FloatToSortableKey + store (both rows) ----
    # row0
    di_reg_0, di_reg_1 = vf.de_interleave(sum0_r0, sum1_r0)
    f16_even = vf.astype(di_reg_0, preg_b32, layout=pl.CastLayout.ZERO, dtype=pl.DT_BF16)
    f16_odd = vf.astype(di_reg_1, preg_b32, layout=pl.CastLayout.ONE, dtype=pl.DT_BF16)
    f16_combined = vf.xor(f16_even, f16_odd, preg_u16)
    preg_nan = vf.eq(vf.bit_cast(f16_combined, dtype=pl.DT_UINT16), nan_mask, preg_u16)
    key_reg = vf.select(all_one, f16_combined, preg_nan)
    sign_reg = vf.and_(key_reg, sign_mask, preg_u16)
    preg_sign = vf.gt(sign_reg, zero_u16, preg_u16)
    xor_mask = vf.select(all_one, sign_mask, preg_sign)
    key_reg = vf.xor(key_reg, xor_mask, preg_u16)
    vf.store_align(score_u16_sk_tile, key_reg, preg_u16, dist=pl.StoreDist.NORM_B16)

    # row1
    di_reg_0, di_reg_1 = vf.de_interleave(sum0_r1, sum1_r1)
    f16_even = vf.astype(di_reg_0, preg_b32, layout=pl.CastLayout.ZERO, dtype=pl.DT_BF16)
    f16_odd = vf.astype(di_reg_1, preg_b32, layout=pl.CastLayout.ONE, dtype=pl.DT_BF16)
    f16_combined = vf.xor(f16_even, f16_odd, preg_u16)
    preg_nan = vf.eq(vf.bit_cast(f16_combined, dtype=pl.DT_UINT16), nan_mask, preg_u16)
    key_reg = vf.select(all_one, f16_combined, preg_nan)
    sign_reg = vf.and_(key_reg, sign_mask, preg_u16)
    preg_sign = vf.gt(sign_reg, zero_u16, preg_u16)
    xor_mask = vf.select(all_one, sign_mask, preg_sign)
    key_reg = vf.xor(key_reg, xor_mask, preg_u16)
    vf.store_align(score_u16_sk_tile + TS, key_reg, preg_u16, dist=pl.StoreDist.NORM_B16)


# ================================================================
# TopK VF Implementation
# ================================================================


@pl.vector_function
def histograms_high_vf(hist_high_tile, score_u16_tile, hist_loop: pl.DT_INT64):
    """Histogram high byte (BYTE_1, MSB) — chistv2 accumulates into dst reg."""
    preg_b8 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT8)
    preg_b16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    preg_b32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)

    cout0 = vf.full(0, preg_b16, dtype=pl.DT_UINT16)
    cout1 = vf.full(0, preg_b16, dtype=pl.DT_UINT16)

    for i in pl.range(0, hist_loop):
        vreg_low, vreg_high = vf.load_align(score_u16_tile, i * 256, dist=pl.LoadDist.DINTLV_B8, dtype=pl.DT_UINT8)
        cout0 = vf.histograms(vreg_high, preg_b8, bin_type=pl.BinType.BIN0, hist_type=pl.HistType.ACCUMULATE)
        cout1 = vf.histograms(vreg_high, preg_b8, bin_type=pl.BinType.BIN1, hist_type=pl.HistType.ACCUMULATE)

    c0_even = vf.astype(cout0, preg_b16, layout=pl.CastLayout.ZERO, dtype=pl.DT_UINT32)
    c0_odd = vf.astype(cout0, preg_b16, layout=pl.CastLayout.ONE, dtype=pl.DT_UINT32)
    c1_even = vf.astype(cout1, preg_b16, layout=pl.CastLayout.ZERO, dtype=pl.DT_UINT32)
    c1_odd = vf.astype(cout1, preg_b16, layout=pl.CastLayout.ONE, dtype=pl.DT_UINT32)
    vf.store_align(hist_high_tile, c0_even, c0_odd, preg_b32, dist=pl.StoreDist.INTLV_B32)
    vf.store_align(hist_high_tile + 128, c1_even, c1_odd, preg_b32, dist=pl.StoreDist.INTLV_B32)
    vf.mem_bar()


@pl.vector_function
def find_high_bin_and_histograms_low(
    hist_low_tile,
    idx_high_tile,
    hist_high_tile,
    score_u16_tile,
    sqz_idx_buf,
    nk_value_tile,
    bottom_k: pl.DT_INT64,
    hist_loop: pl.DT_INT64,
):
    """Find high target bin + histogram low byte (merged).

    Flow (mirrors arch35/vf/vf_topk.h FindFirstTargetBinVFImpl):
      Squeeze loop -> StoreUnAlign -> StoreUnAlignPost -> MemBar ->
      load_align BRC_B8 -> byte-wise compute idx_prev0 -> Gather prev_bin_value ->
      Select with preg_eq0 -> Sub next_k -> store_align idx_high & next_k ->
      Histograms low byte (filtered by EQ idx_high).
    """
    # DINTLV_B8 splits the u16 score stream into low/high byte halves. Regs
    # are declared u16 so load_align's b16 stride matches; backend reinterprets
    # as RegTensor<uint8_t>& for chistv2 / vcmp_eq (mirrors vf_topk_16_gather.h).
    preg_b8 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT8)
    preg_b16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    preg_b32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)

    # AR special purpose register tracks StoreUnAlign cumulative byte count.
    vf.clear_spr()
    align_idx_high = vf.unalign_reg_for_store()

    btm_k = vf.full(bottom_k, dtype=pl.DT_UINT32)

    # Step 1: Find first high bin >= bottom_k. Each loop iteration writes a
    # variable-length squeeze output to sqz_idx_buf via POST_UPDATE.
    for i in pl.range(0, 4):
        idx_c = vf.arange(i * 64, dtype=pl.DT_INT32)
        cout = vf.load_align(hist_high_tile, i * 64)
        preg_ge = vf.ge(cout, btm_k, preg_b32)
        sqz_idx_high = vf.squeeze(idx_c, preg_ge, gather_mode=pl.SqueezeMode.STORE_REG, dtype=pl.DT_UINT32)
        vf.squeeze_store_unalign(sqz_idx_buf, sqz_idx_high, align_idx_high)
    vf.squeeze_store_unalign_post(sqz_idx_buf, align_idx_high)
    vf.mem_bar()

    # BRC_B8 broadcasts byte0 of sqz_idx_buf into every byte position of every lane,
    # so each lane holds 0xXXXXXXXX where XX = first high-bin index (0..255).
    idx_high = vf.load_align(sqz_idx_buf, 0, dist=pl.LoadDist.BRC_B8)

    # Compute prev bin: subtract byte-wise 0x01010101 then keep low byte via >> 24
    # (high byte after Sub equals (idx_high - 1) due to byte-wise underflow propagating
    # the same way as in the reference impl).
    idx_all1 = vf.full(0x01010101, dtype=pl.DT_UINT32)
    zero_all = vf.full(0, dtype=pl.DT_UINT32)
    idx_prev0 = vf.sub(idx_high, idx_all1, preg_b32)
    idx_prev0 = vf.shift_right(idx_prev0, 24, preg_b32)

    # Gather previous bin value from histogram
    prev_bin_value = vf.gather(hist_high_tile, idx_prev0, preg_b32)

    # If idx_high == 0, override prev_bin_value to 0
    preg_eq0 = vf.eq(idx_high, zero_all, preg_b32)
    prev_bin_value = vf.select(zero_all, prev_bin_value, preg_eq0)

    next_k = vf.sub(btm_k, prev_bin_value, preg_b32)

    # Persist idx_high (low byte) and next_k for find_kth_vf in subsequent vf scope
    vf.store_align(idx_high_tile, idx_high, preg_b32)
    vf.store_align(nk_value_tile, next_k, preg_b32)

    # Step 2: Histogram low byte filtered by high-byte EQ idx_high.
    # chistv2 accumulates into the u16 dst reg; widen to u32 once after the loop.
    cout0 = vf.full(0, preg_b16, dtype=pl.DT_UINT16)
    cout1 = vf.full(0, preg_b16, dtype=pl.DT_UINT16)

    for i in pl.range(0, hist_loop):
        vreg_low, vreg_high = vf.load_align(score_u16_tile, i * 256, dist=pl.LoadDist.DINTLV_B8, dtype=pl.DT_UINT8)
        preg_eq = vf.eq(vreg_high, vf.bit_cast(idx_high, dtype=pl.DT_UINT8), preg_b8)
        cout0 = vf.histograms(vreg_low, preg_eq, bin_type=pl.BinType.BIN0, hist_type=pl.HistType.ACCUMULATE)
        cout1 = vf.histograms(vreg_low, preg_eq, bin_type=pl.BinType.BIN1, hist_type=pl.HistType.ACCUMULATE)

    c0_even = vf.astype(cout0, preg_b16, layout=pl.CastLayout.ZERO, dtype=pl.DT_UINT32)
    c0_odd = vf.astype(cout0, preg_b16, layout=pl.CastLayout.ONE, dtype=pl.DT_UINT32)
    c1_even = vf.astype(cout1, preg_b16, layout=pl.CastLayout.ZERO, dtype=pl.DT_UINT32)
    c1_odd = vf.astype(cout1, preg_b16, layout=pl.CastLayout.ONE, dtype=pl.DT_UINT32)
    vf.store_align(hist_low_tile, c0_even, c0_odd, preg_b32, dist=pl.StoreDist.INTLV_B32)
    vf.store_align(hist_low_tile + 128, c1_even, c1_odd, preg_b32, dist=pl.StoreDist.INTLV_B32)
    vf.mem_bar()  # ensure hist_low visible to find_kth_vf's vlds


@pl.vector_function
def find_kth_vf(kth_value_tile, hist_low_tile, idx_high_tile, sqz_idx_buf, nk_value_tile):
    """Find kth value from low histogram and assemble (idx_high & 0xff00) + idx_low."""
    preg_b16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    preg_b32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)

    vf.clear_spr()
    align_idx_low = vf.unalign_reg_for_store()

    # next_k was persisted to nk_value_tile by find_high_bin_and_histograms_low
    btm_k = vf.load_align(nk_value_tile, 0, dist=pl.LoadDist.BRC_B32)

    # Find first low bin >= next_k, store squeeze output to sqz_idx_buf.
    for i in pl.range(0, 4):
        idx_c = vf.arange(i * 64, dtype=pl.DT_INT32)
        cout = vf.load_align(hist_low_tile, i * 64)
        preg_ge = vf.ge(cout, btm_k, preg_b32)
        sqz_idx_low = vf.squeeze(idx_c, preg_ge, gather_mode=pl.SqueezeMode.STORE_REG, dtype=pl.DT_UINT32)
        vf.squeeze_store_unalign(sqz_idx_buf, sqz_idx_low, align_idx_low)
    vf.squeeze_store_unalign_post(sqz_idx_buf, align_idx_low)
    vf.mem_bar()

    # BRC_B8:  broadcast byte0 of idxHighBuf -> every u8 lane = idx_high byte
    # AND 0xff00 (u16 mask): keeps high byte of each u16 -> 0x00XX00XX
    # BRC_B16: broadcast u16[0] of sqz_idx_buf -> 0x00YY00YY
    # u16 Add: idx_k = 0x00XX00XX + 0x00YY00YY = 0x00(XX+YY)... (no carry across u16)
    # NORM_B16 store: write as u16 array so BRC_B16 in collect_gt reads correctly.
    idx_high = vf.load_align(idx_high_tile, 0, dist=pl.LoadDist.BRC_B8, dtype=pl.DT_UINT16)
    idx_low = vf.load_align(sqz_idx_buf, 0, dist=pl.LoadDist.BRC_B16, dtype=pl.DT_UINT16)
    idx_mask = vf.full(0xFF00, dtype=pl.DT_UINT16)
    idx_high = vf.and_(idx_high, idx_mask, preg_b32)
    idx_k = vf.add(idx_high, idx_low, preg_b16)
    vf.store_align(kth_value_tile, idx_k, preg_b32, dist=pl.StoreDist.NORM_B16)
    vf.mem_bar()  # ensure kth_value visible to collect_gt's vlds


@pl.vector_function
def collect_indices_gt_eq_vf(
    output_idx_tile,
    score_u16_tile,
    kth_value_tile,
    begin_idx: pl.DT_INT64,
    vf_loop: pl.DT_INT64,
):
    """Collect indices where value > kth (GT), then value == kth (EQ).

    GT indices come first, EQ indices follow. The downstream pl.store
    with set_validshape(topk) truncates to exactly topk entries.
    Output indices are DT_INT32 (matching reference implementation).
    """
    preg_b16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)

    vf.clear_spr()
    align_idx = vf.unalign_reg_for_store()

    kth_value = vf.load_align(kth_value_tile, 0, dist=pl.LoadDist.BRC_B16, dtype=pl.DT_UINT16)

    # Pass 1: collect GT indices
    for i in pl.range(0, vf_loop):
        idx_c = vf.arange(begin_idx + i * 128, dtype=pl.DT_INT16)
        vreg_input = vf.load_align(score_u16_tile, i * 128)
        pout_gt = vf.gt(vreg_input, kth_value, preg_b16)
        sqz_idx_out = vf.squeeze(idx_c, pout_gt, gather_mode=pl.SqueezeMode.STORE_REG, dtype=pl.DT_UINT16)
        vf.squeeze_store_unalign(output_idx_tile, sqz_idx_out, align_idx)

    # Pass 2: collect EQ indices (appends after GT)
    for i in pl.range(0, vf_loop):
        idx_c = vf.arange(begin_idx + i * 128, dtype=pl.DT_INT16)
        vreg_input = vf.load_align(score_u16_tile, i * 128)
        pout_eq = vf.eq(vreg_input, kth_value, preg_b16)
        sqz_idx_out = vf.squeeze(idx_c, pout_eq, gather_mode=pl.SqueezeMode.STORE_REG, dtype=pl.DT_UINT16)
        vf.squeeze_store_unalign(output_idx_tile, sqz_idx_out, align_idx)

    vf.squeeze_store_unalign_post(output_idx_tile, align_idx)


# ================================================================
# Kernel Factory
# ================================================================


def make_quant_lightning_indexer_vf_kernel(
    b: int, n1: int, n2: int, sq: int, sk: int, d: int, topk: int, num_cores: int, sparse_mode: int
):
    _gsize = n1 // n2
    assert n2 == 1, f"n2={n2} not supported; only n2=1 (MHA/GQA with single KV head) is supported."
    assert _gsize in (8, 16, 24, 32, 64), (
        f"N1 is {n1}, N2 is {n2}, N1 divided by N2 must equal 64 or 32 or 24 or 16 or 8."
    )
    _s2_row_coeff = 1 if sparse_mode == 3 else 0
    _s2_sq_coeff = -1 if sparse_mode == 3 else 0
    _s2_const = 1 if sparse_mode == 3 else 0
    _atten_mask_flag = sparse_mode == 3
    _input_dtype = pl.DT_FP8E4M3FN
    _group_size = n1 // n2

    # Dynamic constants derived from n1/n2 (replace former global hardcodes)
    g_size_static = _group_size  # heads per KV group
    m_cube_runtime = TM * n1  # actual cube M dim
    m_half_runtime = m_cube_runtime // 2  # per sub-AIV after dual_split_m
    sq_rows_per_sub = m_half_runtime // g_size_static  # rows per sub-AIV
    # weight/qscale col: pad g_size_static up to next multiple of 16 for 32B DT_FP16 alignment
    wq_col_pad = ((g_size_static + 15) // 16) * 16

    # M dimension includes all heads (GQA optimization)
    # Tile shape uses M_BASE_SIZE=256 (max), actual data uses m_cube=TM*n1 via SetValidShape.
    # mBaseSize=S1_BASE*g_size rows are valid at runtime.
    m_cube = TM * n1
    m_half = m_cube // 2

    # Cube buffer byte sizes — L1 uses M_BASE_SIZE (256) for max allocation;
    # L0A/L0C use m_cube_runtime for NZ dstNzC0Stride to match actual data layout.
    q_bytes = m_cube_runtime * TD
    k_bytes = TD * TS
    acc_bytes = m_cube_runtime * TS * 4

    # ---- Cube buffer addresses (computed at compile time) ----
    # MAT addresses (Query L1: 2-buf, Key L1: 3-buf)
    ma_q0 = 0
    ma_q1 = ma_q0 + q_bytes
    ma0 = ma_q1 + q_bytes
    ma1 = ma0 + k_bytes
    ma2 = ma1 + k_bytes  # noqa: F841
    # L0A/L0B addresses (L0B is 2-buf)
    la0 = 0
    ra0 = 0
    # ACC addresses (2-buf for double buffer)
    # Each buf holds [m_cube_runtime, TS] DT_FP32 in NZ format.
    # For ndNum=2 emulation: split N=128 into two 64-col halves.
    # NZ layout: 4 n-fractals × (m_cube_runtime/16) × 16 × 16 × 4B per half.
    acc_half_n_offset = 4 * (m_cube_runtime // 16) * 16 * 16 * 4
    ca0 = 0
    ca0_hi = ca0 + acc_half_n_offset  # noqa: F841
    ca1 = ca0 + acc_bytes
    ca1_hi = ca1 + acc_half_n_offset  # noqa: F841

    # ---- Vector UB buffer addresses ----
    # mm1_vec: ndNum=2 emulation with interleaved pingpong.
    # SLayout=512B → dstStride=128 elements. Each fixpipe writes 64 cols per row,
    # rows spaced 128 elements apart. Ping at row offset 0, pong at row offset 64.
    # Both pingpong share same region.
    dst_stride = 128  # dstStride in elements
    nd1_off_elems = m_half_runtime * dst_stride  # dstNdStride in elements
    mm1_vec_nd_bytes = nd1_off_elems * 4  # bytes per ND block
    pp_offset_bytes = 256  # 64 elements * 4B
    va_mm1_vec_ping = 0
    va_mm1_vec_ping_hi = mm1_vec_nd_bytes
    va_mm1_vec_pong = pp_offset_bytes
    va_mm1_vec_pong_hi = mm1_vec_nd_bytes + pp_offset_bytes
    mm1_vec_pong_end = mm1_vec_nd_bytes * 2 + pp_offset_bytes
    # weight/qscale: GM DT_FP16, load to DT_FP16 tile [2, wq_col_pad=16] (pad col 8→16 for 32B alignment),
    # then pl.cast to DT_FP32 tile [2, 16]. DT_FP32 size = 2*16*4 = 128B per tensor.
    # The extra 8 padded DT_FP16 values cast to 0.0 and don't affect the BRC_B32 computation
    # since wbrc = vf.load_align(wq_tile, h, dist=pl.LoadDist.BRC_B32) only reads element h (0..7).
    sq_rows_per_sub = m_half // n1  # = 2 for n1=8
    vb_weight_sk_f16 = sq_rows_per_sub * wq_col_pad * 2  # DT_FP16 [2,16]: 64B
    vb_weight_sk = sq_rows_per_sub * wq_col_pad * 4  # DT_FP32 [2,16]: 128B
    va_weight_f16 = mm1_vec_pong_end
    va_qscale_f16 = va_weight_f16 + vb_weight_sk_f16
    va_weight = va_qscale_f16 + vb_weight_sk_f16  # DT_FP32 compute buffer
    va_qscale = va_weight + vb_weight_sk  # DT_FP32 compute buffer
    # score_u16_sk: vec1 output per sk_tile [sq_rows_per_sub, TS] u16, ping-pong 2-buf
    vb_score_u16_sk = sq_rows_per_sub * TS * 2
    va_score_u16_sk_ping = va_qscale + vb_weight_sk
    va_score_u16_sk_pong = va_score_u16_sk_ping + vb_score_u16_sk
    # kscale: per 16-sk_tile batch load. Use full sk_align128 tile shape (same as original)
    # but re-load every 16 tiles with different offset. This avoids tile_dims issues.
    kscale_batch = 16
    kscale_batch_elems = kscale_batch * TS  # 2048
    sk_align128 = (sk + 127) // 128 * 128
    vb_kscale_all_f16 = sk_align128 * 2  # keep full-size tile for codegen compatibility
    vb_kscale_all_f32 = sk_align128 * 4
    # score_u16_row: needs to hold one trunk (up to TRUNK_LEN elements) aligned to 256.
    sk_align256 = (sk + 255) // 256 * 256
    vb_score_u16_row = sk_align256 * 2  # noqa: F841
    va_kscale_f16 = va_score_u16_sk_pong + vb_score_u16_sk
    va_kscale_f32 = va_kscale_f16 + vb_kscale_all_f16
    va_score_u16_row = va_kscale_f16  # Phase 2 reuse (non-overlapping with Phase 1)
    va_after_v1 = va_kscale_f32 + vb_kscale_all_f32

    # TopK VF addresses — dynamic based on actual topk and sk
    topk_align256 = (topk + 255) // 256 * 256  # align topk up to 256 for output buffer
    # output_idx_buf: GT indices (up to sk_align256) + EQ indices (up to topk_align256)
    # VF Squeeze writes DT_UINT16 indices (matching reference FindIdxGTOutputVFImpl/FindIdxEQOutputVFImpl).
    # A separate DT_INT32 cast buffer holds the final topk indices after UINT16→INT32 expansion.
    _topk_output_buf_len = sk_align256 + topk_align256
    _topk_output_buf_bytes = _topk_output_buf_len * 2  # DT_UINT16: 2 bytes each
    tk_hist_high = va_after_v1 + TK_HIST_HIGH_OFF
    tk_hist_low = va_after_v1 + TK_HIST_LOW_OFF
    tk_sqz_idx = va_after_v1 + TK_SQZ_IDX_OFF
    tk_idx_high = va_after_v1 + TK_IDX_HIGH_OFF
    tk_nk_value = va_after_v1 + TK_NK_VALUE_OFF
    tk_kth_value = va_after_v1 + TK_KTH_VALUE_OFF
    tk_output_idx = va_after_v1 + TK_KTH_VALUE_OFF + 256  # DT_UINT16 collect buffer

    @pl.vector_function
    def _vf_quant_lightning_indexer_vf_kernel_0(vec_valid_s2_len, output_idx_tile):
        preg_b32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT32)
        preg_u16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)  # noqa: F841
        neg_one_i32 = vf.full(-1, dtype=pl.DT_INT32)
        threshold = vf.full(vec_valid_s2_len, dtype=pl.DT_INT32)
        vf.clear_spr()
        align_idx = vf.unalign_reg_for_store()
        # Output topk elements: [0..valid-1] then [-1...-1]
        topk_loops = (topk + 63) // 64
        for idx_i in pl.range(0, topk_loops):
            idx_c = vf.arange(idx_i * 64, dtype=pl.DT_INT32)
            # idx >= valid_len → select -1; else keep index
            preg_ge = vf.ge(idx_c, threshold, preg_b32)
            idx_c = vf.select(neg_one_i32, idx_c, preg_ge)
            # squeeze all 64 elements (truncate INT32→UINT16)
            sqz_out = vf.squeeze(idx_c, preg_b32, gather_mode=pl.SqueezeMode.STORE_REG, dtype=pl.DT_UINT32)
            vf.squeeze_store_unalign(output_idx_tile, sqz_out, align_idx)
        vf.squeeze_store_unalign_post(output_idx_tile, align_idx)
        vf.mem_bar()

    @pl.jit(auto_mutex=True)
    def quant_lightning_indexer_vf_kernel(
        query: pl.Tensor[[b, pl.DYNAMIC, n1, d], _input_dtype],
        key: pl.Tensor[[b, pl.DYNAMIC, n2, d], _input_dtype],
        weights: pl.Tensor[[b, pl.DYNAMIC, n1], pl.DT_FP16],
        q_scale: pl.Tensor[[b, pl.DYNAMIC, n1], pl.DT_FP16],
        k_scale: pl.Tensor[[b, pl.DYNAMIC, n2], pl.DT_FP16],
        score_gm: pl.Tensor[[num_cores, TM, pl.DYNAMIC], pl.DT_UINT16],
        sorted_indices: pl.Tensor[[b, pl.DYNAMIC, n2, topk], pl.DT_INT16],
    ):

        sq_real = query.shape[1]
        sk_real = key.shape[1]
        gs1_tiles = (sq_real * n1 + m_cube_runtime - 1) // m_cube_runtime
        sk_tiles = (sk_real + TS - 1) // TS
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx() // pl.get_subblock_num()

        # Vec reads via full-slot tile group (interleaved pingpong layout)
        mm1_vec_group = pl.make_tile_group(
            type=pl.TileType(shape=[m_half_runtime, TS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=[va_mm1_vec_ping, va_mm1_vec_pong],
            mutex_ids=[15, 16],
        )

        pl.system.bar_all()

        with pl.section_cube():
            q_l1_db = pl.make_tile_group(
                type=pl.TileType(shape=[m_cube_runtime, TD], dtype=_input_dtype, target_memory=pl.MemorySpace.Mat),
                addrs=ma_q0,
                mutex_ids=[0, 1],
            )
            k_l1_db = pl.make_tile_group(
                type=pl.TileType(shape=[TD, TS], dtype=_input_dtype, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
                addrs=ma0,
                mutex_ids=[2, 3, 4],
            )
            left_db = pl.make_tile_group(
                type=pl.TileType(
                    shape=[m_cube_runtime, TD], dtype=_input_dtype, target_memory=pl.MemorySpace.Left, compact=1
                ),
                addrs=la0,
                mutex_ids=[5, 6],
            )
            right_db = pl.make_tile_group(
                type=pl.TileType(shape=[TD, TS], dtype=_input_dtype, target_memory=pl.MemorySpace.Right, compact=1),
                addrs=ra0,
                mutex_ids=[7, 8],
            )

            # ACC: single make_tile_group for matmul + fixpipe (auto_mutex handles M↔FIX sync)
            acc_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[m_cube_runtime, TS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, compact=1
                ),
                addrs=[ca0, ca1],
                mutex_ids=[9, 10],
            )

            # UB dst: ND0 and ND1 each as make_tile_group (auto_mutex handles FIX↔V sync)
            mm1_nd0_group = pl.make_tile_group(
                type=pl.TileType(shape=[m_half_runtime, TS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                addrs=[va_mm1_vec_ping, va_mm1_vec_pong],
                mutex_ids=[11, 12],
            )
            mm1_nd1_group = pl.make_tile_group(
                type=pl.TileType(shape=[m_half_runtime, TS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                addrs=[va_mm1_vec_ping_hi, va_mm1_vec_pong_hi],
                mutex_ids=[13, 14],
            )

            loop = 0

            for b_id in pl.range(0, b):
                for gs1_tile_id in pl.range(core_id, gs1_tiles, num_cores):
                    gs1_off = gs1_tile_id * m_cube_runtime
                    gs1_valid = pl.min(sq_real * n1 - gs1_off, m_cube_runtime)
                    s1_base = gs1_off // n1

                    # Q L1 slot pinned at gS1_tile granularity; load deferred
                    # to sk_tile_id == 0 inside the sk loop so K MTE2 fires
                    # first each iteration (mirrors reference ComputeMm1:
                    # KeyNd2Nz before QueryNd2Nz, Q only on isFirstS2InnerLoop).
                    q_slot = q_l1_db.next()
                    pl.set_validshape(q_slot, [gs1_valid, TD])

                    for sk_tile_id in pl.range(0, sk_tiles):
                        sk_off = sk_tile_id * TS
                        pingpong = loop % 2

                        # CrossCoreWait at iteration start (mirrors reference:
                        # CrossCoreWaitFlag before KeyNd2Nz, letting MTE2 overlap).
                        if pingpong == 0:
                            pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=VEC_TO_CUBE_0)
                        else:
                            pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=VEC_TO_CUBE_1)

                        k_slot = k_l1_db.next()
                        left_slot = left_db.next()
                        right_slot = right_db.next()

                        # K validshape only — issue K MTE2 ASAP, then set the
                        # remaining shapes (right/acc/left) while K MTE2 is in
                        # flight. This shrinks the scalar prologue between
                        # k_l1_db.next() and pl.load(K).
                        k_valid = pl.min(sk_real - sk_off, TS)
                        pl.set_validshape(k_slot, [TD, k_valid])
                        pl.load(k_slot, key, [b_id, sk_off, 0, 0], order=[3, 1])

                        # Q MTE2: only on first sk_tile, after K MTE2 issued.
                        if sk_tile_id == 0:
                            pl.load(q_slot, query, [b_id, s1_base, 0, 0], order=[2, 3])

                        # Remaining set_validshape after K/Q MTE2 are in flight.
                        pl.set_validshape(right_slot, [TD, k_valid])
                        pl.set_validshape(left_slot, [gs1_valid, TD])

                        # Move Q/K -> L0A/L0B (NBuffer auto_mutex handles MTE1↔M sync)
                        pl.move(left_slot, q_slot)
                        pl.move(right_slot, k_slot)

                        # Matmul: write to ACC (make_tile_group auto_mutex handles M↔FIX sync)
                        acc_tile = acc_group.next()
                        pl.set_validshape(acc_tile, [gs1_valid, k_valid])
                        pl.matmul(acc_tile, left_slot, right_slot)

                        mm1_nd0_slot = mm1_nd0_group.next()
                        mm1_nd1_slot = mm1_nd1_group.next()
                        # fixpipe 1: ACC[M, 0:64] → UB ND0
                        pl.set_validshape(mm1_nd0_slot, [gs1_valid, 64])
                        pl.set_validshape(acc_tile, [gs1_valid, 64])
                        pl.move(mm1_nd0_slot, acc_tile, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                        pl.set_validshape(mm1_nd1_slot, [gs1_valid, 64])
                        pl.move(mm1_nd1_slot, acc_tile, [0, 64], acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)

                        # Notify vector: mm1_vec slot ready
                        if pingpong == 0:
                            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=CUBE_TO_VEC_0)
                        else:
                            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=CUBE_TO_VEC_1)

                        loop = loop + 1

            # Drain: wait vec to release both UB slots
            pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=VEC_TO_CUBE_0)
            pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=VEC_TO_CUBE_1)
        with pl.section_vector():
            # weight/qscale: DT_FP16 load tiles [2, 16] (col padded 8→16 for 32B alignment),
            # DT_FP32 compute tiles [2, 16] at separate addresses.
            # The DT_FP16 load buffers are make_tile_group so auto_mutex inserts the
            # MTE2(load)→V(cast) handshake and WAR reuse guard (replaces the manual
            weight_f16_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[sq_rows_per_sub, wq_col_pad], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec
                ),
                addrs=[va_weight_f16],
                mutex_ids=[19],
            )
            weight_tile = pl.make_tile(
                pl.TileType(shape=[sq_rows_per_sub, wq_col_pad], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                addr=va_weight,
            )
            qscale_f16_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[sq_rows_per_sub, wq_col_pad], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec
                ),
                addrs=[va_qscale_f16],
                mutex_ids=[20],
            )
            qscale_tile = pl.make_tile(
                pl.TileType(shape=[sq_rows_per_sub, wq_col_pad], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                addr=va_qscale,
            )
            # kscale: full-size tile (same shape as original for codegen compatibility).
            # DT_FP16 load buffer is a make_tile_group so auto_mutex inserts the
            # MTE2(load)→V(cast) handshake (replaces manual sync_src/sync_dst event_id=4).
            kscale_f16_group = pl.make_tile_group(
                type=pl.TileType(shape=[1, sk_align128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
                addrs=[va_kscale_f16],
                mutex_ids=[21],
            )
            kscale_f32_tile = pl.make_tile(
                pl.TileType(shape=[1, sk_align128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                addr=va_kscale_f32,
            )
            # score_u16_sk: make_tile_group (auto_mutex handles V↔MTE3 sync)
            score_u16_sk_group = pl.make_tile_group(
                type=pl.TileType(shape=[sq_rows_per_sub, TS], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec),
                addrs=[va_score_u16_sk_ping, va_score_u16_sk_pong],
                mutex_ids=[17, 18],
            )
            # score_u16_row: make_tile_group so auto_mutex inserts the MTE2(load)→V(vf)
            # handshake and the WAR reuse guard across trunks (replaces the manual
            score_u16_row_group = pl.make_tile_group(
                type=pl.TileType(shape=[1, sk_align256], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec, pad=0),
                addrs=[va_score_u16_row],
                mutex_ids=[22],
            )
            hist_high_tile = pl.make_tile(
                pl.TileType(shape=[1, 256], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
                addr=tk_hist_high,
            )
            hist_low_tile = pl.make_tile(
                pl.TileType(shape=[1, 256], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
                addr=tk_hist_low,
            )
            sqz_idx_buf = pl.make_tile(
                pl.TileType(shape=[1, 256], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
                addr=tk_sqz_idx,
            )
            idx_high_tile = pl.make_tile(
                pl.TileType(shape=[1, 64], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
                addr=tk_idx_high,
            )
            nk_value_tile = pl.make_tile(
                pl.TileType(shape=[1, 64], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
                addr=tk_nk_value,
            )
            kth_value_tile = pl.make_tile(
                pl.TileType(shape=[1, 64], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
                addr=tk_kth_value,
            )
            # DT_UINT16 collect buffer (VF Squeeze writes DT_UINT16, matching reference tmpIdxLocal).
            # make_tile_group so auto_mutex inserts the V(vf)→MTE3(store) handshake and the
            output_idx_group = pl.make_tile_group(
                type=pl.TileType(shape=[1, _topk_output_buf_len], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec),
                addrs=[tk_output_idx],
                mutex_ids=[23],
            )

            # Vec pre-sends two VEC_TO_CUBE signals so cube can start immediately
            # (mirrors reference: CrossCoreSetFlag(CROSS_VC_EVENT+0/1) at vec init)
            pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=VEC_TO_CUBE_0)
            pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=VEC_TO_CUBE_1)

            sub_id = pl.get_subblock_idx()
            loop = 0

            for b_id in pl.range(0, b):
                for gs1_tile_id in pl.range(core_id, gs1_tiles, num_cores):
                    gs1_off = gs1_tile_id * m_cube_runtime
                    s1_base = gs1_off // n1
                    sq_row_base = sub_id * sq_rows_per_sub

                    # ---- Phase 1: load weight/qscale as DT_FP16, cast to DT_FP32 ----
                    # weight/qscale: load FP16 [2, n1] into [2, 16] tile (col padded, 32B-aligned).
                    # auto_mutex on these tile-group slots provides the load↔cast sync.
                    weight_f16_slot = weight_f16_group.next()
                    qscale_f16_slot = qscale_f16_group.next()
                    pl.set_validshape(weight_f16_slot, [sq_rows_per_sub, n1])
                    pl.load(weight_f16_slot, weights, [b_id, s1_base + sq_row_base, 0], order=[1, 2])
                    pl.set_validshape(qscale_f16_slot, [sq_rows_per_sub, n1])
                    pl.load(qscale_f16_slot, q_scale, [b_id, s1_base + sq_row_base, 0], order=[1, 2])
                    # Cast weight/qscale FP16 -> FP32
                    pl.set_validshape(weight_tile, [sq_rows_per_sub, n1])
                    pl.cast(weight_tile, weight_f16_slot, mode=pl.RoundMode.CAST_ROUND)
                    pl.set_validshape(qscale_tile, [sq_rows_per_sub, n1])
                    pl.cast(qscale_tile, qscale_f16_slot, mode=pl.RoundMode.CAST_ROUND)
                    # Compute wq = weight * qscale in-place on weight_tile (once per gS1_tile).
                    pl.mul(weight_tile, weight_tile, qscale_tile)

                    # ---- Phase 1: per sk_tile ----
                    for sk_tile_id in pl.range(0, sk_tiles):
                        sk_off = sk_tile_id * TS
                        pingpong = loop % 2

                        # kscale: load 2048 DT_FP16 every 16 sk_tiles (before cross_core wait,
                        # overlaps with cube fixpipe). order=[0,1] → offset is element-level
                        # on dim1(Sk) since stride[dim4]=1 in generated code.
                        # kscale: load 2048 DT_FP16 every 16 sk_tiles, before cross_core wait
                        if sk_tile_id % kscale_batch == 0:
                            kscale_batch_start = (sk_tile_id // kscale_batch) * kscale_batch_elems
                            # auto_mutex on the kscale_f16 tile-group slot provides load↔cast sync.
                            kscale_f16_slot = kscale_f16_group.next()
                            pl.set_validshape(kscale_f16_slot, [1, kscale_batch_elems])
                            pl.load(kscale_f16_slot, k_scale, [b_id, kscale_batch_start, 0], order=[0, 1])
                            pl.set_validshape(kscale_f32_tile, [1, kscale_batch_elems])
                            pl.cast(kscale_f32_tile, kscale_f16_slot, mode=pl.RoundMode.CAST_ROUND)

                        # Wait for cube fixpipe on V pipe
                        if pingpong == 0:
                            pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=CUBE_TO_VEC_0)
                        else:
                            pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=CUBE_TO_VEC_1)

                        # kscale offset within the 16-tile batch
                        kscale_off = (sk_tile_id % kscale_batch) * TS

                        mm1_slot = mm1_vec_group.next()
                        score_slot = score_u16_sk_group.next()
                        reduce_and_cast_sk_vf(
                            mm1_slot,
                            weight_tile,
                            kscale_f32_tile[:, kscale_off:],
                            score_slot,
                            g_size_static,
                            wq_col_pad,
                        )

                        pl.store(score_gm, score_slot, [core_id, sq_row_base, sk_off], order=[1, 2])

                        # Notify cube on V pipe (parallel with MTE3 store): mm1_vec slot consumed
                        if pingpong == 0:
                            pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=VEC_TO_CUBE_0)
                        else:
                            pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=VEC_TO_CUBE_1)

                        loop = loop + 1

                    # ---- Phase 2: topk per row from score_gm ----
                    # MTE3 -> MTE2 fence: ensure Phase 1's score_gm stores are flushed
                    # to GM before Phase 2's loads start (mirrors reference ProcessTopK
                    # which issues SetFlag<MTE3_MTE2>/WaitFlag<MTE3_MTE2> at entry).
                    # NOTE: this is the one sync that make_tile_group cannot replace — it
                    # orders a store→load hazard on the score_gm *GM* scratch tensor (a
                    # kernel argument), not a reuse hazard on a UB tile. auto_mutex only
                    # synchronises accesses to tile-group UB slots, so this GM fence stays.
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=0)
                    for row_within in pl.range(0, TM):
                        row_abs = s1_base + row_within
                        if row_abs < sq_real:
                            vec_valid_s2_len = sk_real + _s2_row_coeff * row_abs + _s2_sq_coeff * sq_real + _s2_const

                            if vec_valid_s2_len <= 0:
                                pass  # No valid data, keep sorted_indices as -1
                            elif vec_valid_s2_len <= topk:
                                # validS2Len <= topk: output sequential indices [0..validS2Len-1, -1...-1].
                                # then Duplicate -1 for tail positions [validS2Len..topk).
                                # VF API equivalent: squeeze(arange) + store_unalign for seq indices,
                                # then squeeze(negOne) + store_unalign for -1 fill.
                                row_sub_owner = row_within // sq_rows_per_sub
                                if sub_id == row_sub_owner:
                                    # auto_mutex on the output_idx slot provides the vf→store sync.
                                    output_idx_slot = output_idx_group.next()
                                    _vf_quant_lightning_indexer_vf_kernel_0(vec_valid_s2_len, output_idx_slot)
                                    pl.set_validshape(output_idx_slot, [1, topk])
                                    pl.store(sorted_indices, output_idx_slot, [b_id, row_abs, 0, 0])
                            else:
                                row_sub_owner = row_within // sq_rows_per_sub
                                if sub_id == row_sub_owner:
                                    for trunk_id in pl.range(0, (vec_valid_s2_len + TRUNK_LEN - 1) // TRUNK_LEN):
                                        trunk_start = trunk_id * TRUNK_LEN
                                        trunk_end = pl.min(vec_valid_s2_len, trunk_start + TRUNK_LEN)
                                        trunk_len_actual = trunk_end - trunk_start
                                        _align256_extra = (256 - trunk_len_actual % 256) % 256
                                        trunk_len_align256 = trunk_len_actual + _align256_extra
                                        _align128_extra = (128 - trunk_len_actual % 128) % 128
                                        trunk_len_align128 = trunk_len_actual + _align128_extra
                                        collect_loop = trunk_len_align128 // 128
                                        hist_loop = trunk_len_align256 // 256

                                        # auto_mutex on the score_u16_row slot provides the
                                        # MTE2(load)→V(vf) handshake and cross-trunk WAR guard.
                                        score_row_slot = score_u16_row_group.next()
                                        if _atten_mask_flag:
                                            pl.expands(score_row_slot, 0)

                                        pl.set_validshape(score_row_slot, [1, trunk_len_actual])
                                        pl.load(
                                            score_row_slot, score_gm, [core_id, row_within, trunk_start], order=[1, 2]
                                        )

                                        histograms_high_vf(hist_high_tile, score_row_slot, hist_loop)

                                        bottom_k = pl.max(1, trunk_len_align256 - topk + 1)
                                        find_high_bin_and_histograms_low(
                                            hist_low_tile,
                                            idx_high_tile,
                                            hist_high_tile,
                                            score_row_slot,
                                            sqz_idx_buf,
                                            nk_value_tile,
                                            bottom_k,
                                            hist_loop,
                                        )

                                        find_kth_vf(
                                            kth_value_tile,
                                            hist_low_tile,
                                            idx_high_tile,
                                            sqz_idx_buf,
                                            nk_value_tile,
                                        )

                                        # auto_mutex on the output_idx slot provides the vf→store sync.
                                        output_idx_slot = output_idx_group.next()
                                        collect_indices_gt_eq_vf(
                                            output_idx_slot,
                                            score_row_slot,
                                            kth_value_tile,
                                            trunk_start,
                                            collect_loop,
                                        )

                                        pl.set_validshape(output_idx_slot, [1, topk])
                                        pl.store(sorted_indices, output_idx_slot, [b_id, row_abs, 0, 0])

    return quant_lightning_indexer_vf_kernel


# ================================================================
# Reference + Tests
# ================================================================


def reference_quant_lightning_indexer(query, key, weights, q_scale, k_scale, topk, sparse_mode, row_indices=None):
    """Reference implementation supporting GQA.

    n1 = query head num, n2 = kv head num, g_size = n1/n2.
    row_indices selects original query rows; outputs use that row order. Keeping
    original row numbers and Sq preserves the causal mask when sampling.
    Returns (indices [b, selected_sq, n2, topk], scores [b, selected_sq, n2, sk]).
    """
    b, sq, n1, d = query.shape
    _, sk, n2, _ = key.shape
    g_size = n1 // n2

    q = query.detach().cpu().float()  # [b, sq, n1, d]
    k = key.detach().cpu().float()  # [b, sk, n2, d]
    w = weights.detach().cpu().float()  # [b, sq, n1]
    qs = q_scale.detach().cpu().float()  # [b, sq, n1]
    ks = k_scale.detach().cpu().float()  # [b, sk, n2]

    original_sq = sq
    row_ids = torch.arange(sq) if row_indices is None else torch.tensor(list(row_indices), dtype=torch.int64)
    if row_indices is not None:
        q, w, qs = q[:, row_ids], w[:, row_ids], qs[:, row_ids]
        sq = len(row_ids)

    # Accumulate per-head weighted scores into qk_reduced[b, sq, n2, sk]
    qk_reduced = torch.zeros(b, sq, n2, sk, dtype=torch.float32)

    for ni in range(n1):
        gi = ni // g_size

        q_ni = q[:, :, ni, :]
        k_gi = k[:, :, gi, :]
        w_ni = w[:, :, ni]
        qs_ni = qs[:, :, ni]

        qk = torch.matmul(q_ni, k_gi.transpose(-1, -2))
        qk = torch.relu(qk)
        qk = qk * w_ni.unsqueeze(-1)
        qk = qk * qs_ni.unsqueeze(-1)
        qk_reduced[:, :, gi, :] += qk

    # Apply kscale per kv head
    for gi in range(n2):
        qk_reduced[:, :, gi, :] = qk_reduced[:, :, gi, :] * ks[:, :, gi].unsqueeze(1)

    if sparse_mode == 3:
        mask = torch.arange(sk).unsqueeze(0) <= row_ids.unsqueeze(1) + sk - original_sq
        qk_reduced = qk_reduced * mask.unsqueeze(0).unsqueeze(2)

    # Convert to DT_BF16 sortable key for ranking (matches NPU behavior:
    # DT_FP32 -> DT_BF16 cast -> FloatToSortableKey -> histogram topk)
    qk_bf16 = qk_reduced.to(torch.bfloat16)
    qk_u16 = qk_bf16.view(torch.int16).to(torch.int32) & 0xFFFF

    # FloatToSortableKey: NaN -> 0xFFFF, then XOR
    is_nan = (qk_u16 & 0x7FFF) > 0x7F80
    key = torch.where(is_nan, torch.tensor(0xFFFF, dtype=torch.int32), qk_u16)

    sign_bit = (key >> 15) & 1
    sortable_key = torch.where(sign_bit > 0, key ^ 0xFFFF, key ^ 0x8000)

    indices = torch.full((b, sq, n2, topk), -1, dtype=torch.int16)
    k_eff = min(topk, sk)
    for gi in range(n2):
        topk_idx = sortable_key[:, :, gi, :].argsort(dim=-1, descending=True, stable=True)[..., :k_eff]
        indices[:, :, gi, :k_eff] = topk_idx.to(torch.int16)

    return indices, qk_reduced


def _evaluate_topk_quality(
    npu_indices, ref_indices, ref_qk, topk, sk, recall_threshold=0.80, sparse_mode=0, sq=None, row_indices=None
):
    """Score the NPU output against the reference top-k indices.

    Recall = |NPU top-k ∩ Ref top-k| / |Ref top-k|
    Score ratio = mean(qk[npu_indices]) / mean(qk[ref_indices])

    npu_indices: [b, sq, n2, topk]
    ref_indices: [b, selected_sq, n2, topk]
    ref_qk: [b, selected_sq, n2, sk] — reduced scores per kv head (used only for score ratio)
    row_indices gives the original query rows, in the same order as the reference.
    """
    b, sq_dim, n2_dim, _ = npu_indices.shape
    row_indices = list(range(sq_dim) if row_indices is None else row_indices)
    npu = npu_indices.cpu()[:, row_indices].to(torch.int64).numpy()
    ref = ref_indices.cpu().to(torch.int64).numpy()
    qk = ref_qk.cpu().numpy()  # [b, sq, n2, sk]

    total_valid = 0
    total_npu_idx = 0
    recall_sum = 0.0
    score_ratio_sum = 0.0
    rows_checked = 0
    row_recalls = []

    for bi in range(b):
        for reference_row, si in enumerate(row_indices):
            for gi in range(n2_dim):
                # Skip rows where validS2Len <= 0 (sparse_mode=3 only)
                if sparse_mode == 3 and sq is not None:
                    vec_valid_s2_len = sk + si - sq + 1
                    if vec_valid_s2_len <= 0:
                        continue
                    # Skip rows where validS2Len <= topk (NPU outputs sequential indices,
                    # which is correct — all valid indices are included regardless of order)
                    if vec_valid_s2_len <= topk:
                        rows_checked += 1
                        recall_sum += 1.0
                        score_ratio_sum += 1.0  # Perfect score (all valid indices present)
                        row_recalls.append((bi, gi, si, 1.0, vec_valid_s2_len, vec_valid_s2_len))
                        continue

                npu_row = npu[bi, reference_row, gi]
                ref_row = ref[bi, reference_row, gi]
                qk_row = qk[bi, reference_row, gi]  # [sk]

                # NPU valid indices (in [0, sk))
                valid_mask = (npu_row >= 0) & (npu_row < sk)
                npu_valid = npu_row[valid_mask]
                # Ref valid indices (filter -1 padding and out-of-range)
                ref_valid_mask = (ref_row >= 0) & (ref_row < sk)
                ref_valid = ref_row[ref_valid_mask]

                total_npu_idx += npu_row.size
                total_valid += npu_valid.size

                # Use sets so duplicates don't double-count
                npu_set = set(npu_valid.tolist())
                ref_set = set(ref_valid.tolist())

                if len(ref_set) == 0:
                    rows_checked += 1
                    continue

                hit = len(npu_set & ref_set)
                recall = hit / len(ref_set)
                recall_sum += recall
                row_recalls.append((bi, gi, si, recall, hit, len(ref_set)))

                # Score ratio: mean qk over selected indices.
                npu_qk_mean = float(qk_row[npu_valid].mean()) if npu_valid.size > 0 else 0.0
                ref_qk_mean = float(qk_row[ref_valid].mean()) if ref_valid.size > 0 else 0.0
                if ref_qk_mean != 0:
                    score_ratio_sum += npu_qk_mean / ref_qk_mean

                rows_checked += 1

    valid_rate = total_valid / max(total_npu_idx, 1)
    mean_recall = recall_sum / max(rows_checked, 1)
    mean_score_ratio = score_ratio_sum / max(rows_checked, 1)

    logging.info(f"  Rows sampled:        {rows_checked}")
    logging.info(f"  Valid-index rate:    {valid_rate:.4f}")
    logging.info(f"  Mean recall:         {mean_recall:.4f}")
    logging.info(f"  Mean score ratio:    {mean_score_ratio:.4f}")
    display_index = -1 if sparse_mode == 3 and sq is not None else 0
    display_row = row_indices[display_index]
    logging.info(f"  NPU[0,{display_row},0,:8]:   {npu[0, display_index, 0, :8].tolist()}")
    logging.info(f"  Ref[0,{display_row},0,:8]:   {ref[0, display_index, 0, :8].tolist()}")
    npu_sorted = sorted(npu[0, display_index, 0, :topk].tolist())
    ref_sorted = sorted(ref[0, display_index, 0, :topk].tolist())
    logging.info(f"  NPU sorted[:16]:      {npu_sorted[:16]}")
    logging.info(f"  Ref sorted[:16]:      {ref_sorted[:16]}")
    logging.info(f"  NPU sorted[-16:]:     {npu_sorted[-16:]}")
    logging.info(f"  Ref sorted[-16:]:     {ref_sorted[-16:]}")

    if mean_recall < recall_threshold or valid_rate < 0.5:
        sorted_rows = sorted(row_recalls, key=lambda x: x[3])[:8]
        logging.info("  Lowest recall rows (top 8):")
        for bi, gi, si, recall, hit, ref_total in sorted_rows:
            logging.info(f"    b={bi + 1} n2={gi + 1} sq={si + 1}: recall={recall:.4f} (hit={hit}/{ref_total})")

    assert valid_rate >= 0.5, f"valid_rate={valid_rate:.4f} too low"
    assert mean_recall >= recall_threshold, f"Mean recall {mean_recall:.4f} below threshold {recall_threshold}"

    return {
        "valid_rate": valid_rate,
        "mean_recall": mean_recall,
        "mean_score_ratio": mean_score_ratio,
    }


@pytest.mark.soc("950")
@pytest.mark.skip_jit_discovery(reason="Single kernel with an expensive CPU reference; run both only once")
@pl.jit()
@pypto.options(pass_options={"enable_slice": False})
def test_quant_lightning_indexer_vf():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"

    logging.info("\n=== Test QuantLightningIndexer VF ===")

    test_cases = [
        (1, 64, 1, 8192, 8192, HEAD_DIM, 2048, 28, 0, "fp8"),  # 950 默认流预算 28 cube 核
    ]

    for b, n1, n2, sq, sk, d, topk, num_cores, sparse_mode, input_dtype in test_cases:
        logging.info(
            f"\nCase: B={b} N1={n1} N2={n2} Sq={sq} Sk={sk} topk={topk} dtype={input_dtype} sparse_mode={sparse_mode}"
        )

        np.random.seed(42)

        assert input_dtype == "fp8", f"input_dtype={input_dtype!r} not supported; only 'fp8' (FP8E4M3FN) is supported."
        torch_input_dtype = torch.float8_e4m3fn
        query = (
            torch.tensor(np.random.uniform(-128, 127, (b, sq, n1, d)), dtype=torch.float32)
            .to(torch_input_dtype)
            .to(device)
        )
        key = (
            torch.tensor(np.random.uniform(-128, 127, (b, sk, n2, d)), dtype=torch.float32)
            .to(torch_input_dtype)
            .to(device)
        )
        weights = (
            torch.tensor(np.random.uniform(0, 0.01, (b, sq, n1)), dtype=torch.float32).to(torch.float16).to(device)
        )
        q_scale = torch.tensor(np.random.uniform(0, 10, (b, sq, n1)), dtype=torch.float32).to(torch.float16).to(device)
        k_scale = torch.tensor(np.random.uniform(0, 10, (b, sk, n2)), dtype=torch.float32).to(torch.float16).to(device)

        sorted_indices = torch.full([b, sq, n2, topk], -1, device=device, dtype=torch.int16)

        sk_aligned = ((sk + S2_BASE_SIZE - 1) // S2_BASE_SIZE) * S2_BASE_SIZE
        score_gm = torch.zeros([num_cores, S1_BASE_SIZE, sk_aligned], device=device, dtype=torch.int16)

        kernel = make_quant_lightning_indexer_vf_kernel(
            b,
            n1,
            n2,
            sq,
            sk,
            d,
            topk,
            num_cores,
            sparse_mode,
        )

        for run_i in range(10):
            sorted_indices.fill_(-1)
            kernel[None, num_cores](query, key, weights, q_scale, k_scale, score_gm, sorted_indices)
            torch.npu.synchronize()
            logging.info(f"  run {run_i} done")

        # Convert DT_INT16 output to DT_INT32 on CPU (kernel outputs DT_INT16 indices,
        # reference implementation uses DT_INT32; values are identical since sk <= 32767).
        sorted_indices_i32 = sorted_indices.cpu().to(torch.int32)

        # Preserve the existing coverage: all small-case rows, every fourth row
        # for large cases (2,048 of 8,192), with the same stable top-k tie ordering.
        row_indices = range(0, sq, 1 if sq <= 2048 else 4)
        ref_indices, ref_qk = reference_quant_lightning_indexer(
            query, key, weights.float(), q_scale.float(), k_scale.float(), topk, sparse_mode, row_indices=row_indices
        )
        metrics = _evaluate_topk_quality(
            sorted_indices_i32,
            ref_indices,
            ref_qk,
            topk=topk,
            sk=sk,
            recall_threshold=0.999,
            sparse_mode=sparse_mode,
            sq=sq,
            row_indices=row_indices,
        )

        assert metrics['mean_recall'] >= 0.999, (
            f"FAIL: recall={metrics['mean_recall']:.4f}, score_ratio={metrics['mean_score_ratio']:.4f}"
        )
        logging.info(f"  PASS  (recall={metrics['mean_recall']:.4f}, score_ratio={metrics['mean_score_ratio']:.4f})")

        torch.save(sorted_indices_i32, "pypto_Li_vf_sorted_indices.bin")


if __name__ == "__main__":
    logging.info("QuantLightningIndexer VF Test")
    logging.info("=" * 60)
    test_quant_lightning_indexer_vf()
    logging.info("\nTest completed!")
