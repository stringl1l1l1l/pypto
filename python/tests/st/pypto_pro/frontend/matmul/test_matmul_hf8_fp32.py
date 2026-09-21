#!/usr/bin/env python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

from functools import lru_cache
import logging
import os
import struct

import numpy as np
import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

logging.basicConfig(level=logging.INFO)

M_TOTAL = 128  # 2 M rounds
K_TOTAL = 192  # 3 K rounds -> first / middle / last
N_TOTAL = 128  # 2 N rounds
TILE = 64

M_TAIL_TOTAL = 200
K_TAIL_TOTAL = 152
N_TAIL_TOTAL = 168

K_CORES = 3

BATCH = 2
HEADS = 2


def _ceil_blocks(total, tile):
    """Number of tile-sized blocks needed to cover ``total``, including a partial one.

    Block-index loops must use this, not ``total // tile``: plain floor division silently
    drops the tail block, so the kernel would compute a smaller product and still pass
    every shape check (the missing rows/cols just never get written).
    """
    return -(-total // tile)


# Identity store scale (fp32 bit-pattern of 1.0): cast FP32 acc -> HF8 without rescale.
# Unit scale: the quantized store path converts the accumulator dtype without
# rescaling. Passed as a plain float -- the framework reinterprets an FP32 scale as
# its IEEE-754 bit pattern in codegen, so no struct.pack is needed here any more.
_SCALE_ONE = 1.0


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip("not A5")


# #####################################################################################
# Kernel factories — M/N/K loop bodies, dtype per memory space taken as arguments
# #####################################################################################


def _kernel_name(tag, in_dtype, acc, out):
    """Unique name per dtype combo.

    Artifact directories include this name. Distinct specializations need distinct
    names so parallel compilation cannot overwrite another specialization's source.
    """
    return f"{tag}_in{in_dtype}_acc{acc}_out{out}"


def make_matmul_mnk_if(in_dtype, out_dtype, acc_dtype=pl.DT_FP32, m_total=M_TOTAL, k_total=K_TOTAL, n_total=N_TOTAL):
    """M/N/K multi-round loops with if/elif/else control flow, tail blocks supported."""
    # One dtype from GM through L1 and L0A/L0B: pl.load and pl.move never cast, so the
    # whole operand path carries in_dtype. Only the accumulator (FP32) and the output
    # differ.
    acc = acc_dtype
    quantized_out = out_dtype != acc
    mt, kt, nt = m_total, k_total, n_total
    _kname = _kernel_name("t01_mnkif", in_dtype, acc, out_dtype) + f"_{mt}x{kt}x{nt}"

    @pl.jit(auto_mutex=True, name=_kname)
    def t01_mnk_loop_if(
        a: pl.Tensor[[mt, kt], in_dtype],
        b: pl.Tensor[[kt, nt], in_dtype],
        out: pl.Tensor[[mt, nt], out_dtype],
    ):
        a_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[0, 1],
        )
        b_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x40000,
            mutex_ids=[2, 3],
        )
        a_l0a = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Left,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[4, 5],
        )
        b_l0b = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Right,
                layout=pl.ZN,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[6, 7],
        )
        c_l0c = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=acc,
                target_memory=pl.MemorySpace.Acc,
                layout=pl.NZ,
                fractal=1024,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[8],
        )
        with pl.section_cube():
            for mi in pl.range(0, mt, TILE):
                m_valid = pl.min(TILE, mt - mi)
                for ni in pl.range(0, nt, TILE):
                    n_valid = pl.min(TILE, nt - ni)
                    ac = c_l0c.current()
                    # Output window for this tile: clips on the M and N tails.
                    pl.set_validshape(ac, [m_valid, n_valid])
                    for ki in pl.range(0, kt, TILE):
                        k_valid = pl.min(TILE, kt - ki)
                        cur_a = a_l1.next()
                        cur_b = b_l1.next()
                        al = a_l0a.next()
                        br = b_l0b.next()
                        # Both L1 and L0 tiles need the window, on every side.
                        pl.set_validshape(cur_a, [m_valid, k_valid])
                        pl.set_validshape(cur_b, [k_valid, n_valid])
                        pl.set_validshape(al, [m_valid, k_valid])
                        pl.set_validshape(br, [k_valid, n_valid])
                        pl.load(cur_a, a, [mi, ki])
                        pl.load(cur_b, b, [ki, ni])
                        pl.move(al, cur_a)
                        pl.move(br, cur_b)
                        # Arm selection is positional and stays correct with a K tail:
                        # the partial block is simply the last one.
                        if ki == 0 and ki + TILE >= kt:
                            pl.matmul(ac, al, br, phase=pl.AccPhase.Final)
                        elif ki == 0:
                            pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
                        elif ki + TILE < kt:
                            pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
                        else:
                            pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
                    if quantized_out:
                        pl.store(out, ac, [mi, ni], scale=_SCALE_ONE, phase=pl.STPhase.Final)
                    else:
                        pl.store(out, ac, [mi, ni], phase=pl.STPhase.Final)

    return t01_mnk_loop_if


def make_matmul_no_phase(in_dtype, out_dtype, acc_dtype=pl.DT_FP32, m_total=M_TOTAL, k_total=K_TOTAL, n_total=N_TOTAL):
    """t02: M/N/K multi-round loops, matmul/matmul_acc without ``phase``."""
    # One dtype from GM through L1 and L0A/L0B: pl.load and pl.move never cast, so the
    # whole operand path carries in_dtype. Only the accumulator (FP32) and the output
    # differ.
    acc = acc_dtype
    quantized_out = out_dtype != acc
    mt, kt, nt = m_total, k_total, n_total
    _kname = _kernel_name("t02_nophase", in_dtype, acc, out_dtype) + f"_{mt}x{kt}x{nt}"

    @pl.jit(auto_mutex=True, name=_kname)
    def t02_mnk_no_phase(
        a: pl.Tensor[[mt, kt], in_dtype],
        b: pl.Tensor[[kt, nt], in_dtype],
        out: pl.Tensor[[mt, nt], out_dtype],
    ):
        a_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[0, 1],
        )
        b_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x40000,
            mutex_ids=[2, 3],
        )
        a_l0a = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Left,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[4, 5],
        )
        b_l0b = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Right,
                layout=pl.ZN,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[6, 7],
        )
        c_l0c = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=acc,
                target_memory=pl.MemorySpace.Acc,
                layout=pl.NZ,
                fractal=1024,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[8],
        )
        with pl.section_cube():
            for mi in pl.range(0, mt, TILE):
                m_valid = pl.min(TILE, mt - mi)
                for ni in pl.range(0, nt, TILE):
                    n_valid = pl.min(TILE, nt - ni)
                    ac = c_l0c.current()
                    pl.set_validshape(ac, [m_valid, n_valid])
                    for ki in pl.range(0, kt, TILE):
                        k_valid = pl.min(TILE, kt - ki)
                        cur_a = a_l1.next()
                        cur_b = b_l1.next()
                        al = a_l0a.next()
                        br = b_l0b.next()
                        pl.set_validshape(cur_a, [m_valid, k_valid])
                        pl.set_validshape(cur_b, [k_valid, n_valid])
                        pl.set_validshape(al, [m_valid, k_valid])
                        pl.set_validshape(br, [k_valid, n_valid])
                        pl.load(cur_a, a, [mi, ki])
                        pl.load(cur_b, b, [ki, ni])
                        pl.move(al, cur_a)
                        pl.move(br, cur_b)
                        if ki == 0:
                            pl.matmul(ac, al, br)  # no phase
                        else:
                            pl.matmul_acc(ac, ac, al, br)  # no phase
                    if quantized_out:
                        pl.store(out, ac, [mi, ni], scale=_SCALE_ONE)
                    else:
                        pl.store(out, ac, [mi, ni])  # no STPhase

    return t02_mnk_no_phase


def make_matmul_atomic(in_dtype, out_dtype, acc_dtype=pl.DT_FP32, m_total=M_TOTAL, k_total=K_TOTAL, n_total=N_TOTAL):
    """t03: M/N/K multi-round loops, K accumulated by an atomic store."""
    # One dtype from GM through L1 and L0A/L0B: pl.load and pl.move never cast, so the
    # whole operand path carries in_dtype. Only the accumulator (FP32) and the output
    # differ.
    acc = acc_dtype
    mt, kt, nt = m_total, k_total, n_total
    _kname = _kernel_name("t03_atomic", in_dtype, acc, out_dtype) + f"_{mt}x{kt}x{nt}"

    @pl.jit(auto_mutex=True, name=_kname)
    def t03_mnk_atomic(
        a: pl.Tensor[[mt, kt], in_dtype],
        b: pl.Tensor[[kt, nt], in_dtype],
        out: pl.Tensor[[mt, nt], out_dtype],
    ):
        a_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[0, 1],
        )
        b_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x40000,
            mutex_ids=[2, 3],
        )
        a_l0a = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Left,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[4, 5],
        )
        b_l0b = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Right,
                layout=pl.ZN,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[6, 7],
        )
        c_l0c = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=acc,
                target_memory=pl.MemorySpace.Acc,
                layout=pl.NZ,
                fractal=1024,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[8],
        )
        with pl.section_cube():
            for mi in pl.range(0, mt, TILE):
                m_valid = pl.min(TILE, mt - mi)
                for ni in pl.range(0, nt, TILE):
                    n_valid = pl.min(TILE, nt - ni)
                    for ki in pl.range(0, kt, TILE):
                        k_valid = pl.min(TILE, kt - ki)
                        cur_a = a_l1.next()
                        cur_b = b_l1.next()
                        al = a_l0a.next()
                        br = b_l0b.next()
                        ac = c_l0c.current()
                        pl.set_validshape(ac, [m_valid, n_valid])
                        pl.set_validshape(cur_a, [m_valid, k_valid])
                        pl.set_validshape(cur_b, [k_valid, n_valid])
                        pl.set_validshape(al, [m_valid, k_valid])
                        pl.set_validshape(br, [k_valid, n_valid])
                        pl.load(cur_a, a, [mi, ki])
                        pl.load(cur_b, b, [ki, ni])
                        pl.move(al, cur_a)
                        pl.move(br, cur_b)
                        pl.matmul(ac, al, br)
                        pl.store(out, ac, [mi, ni], atomic=pl.AtomicType.AtomicAdd)

    return t03_mnk_atomic


def make_matmul_katomic_multicore(
    in_dtype, out_dtype, acc_dtype=pl.DT_FP32, m_total=M_TOTAL, k_total=K_TOTAL, n_total=N_TOTAL, k_cores=K_CORES
):
    """t12: K axis split across cores, the partial sums joined by an atomic store."""
    acc = acc_dtype
    mt, kt, nt = m_total, k_total, n_total
    _kname = _kernel_name("t12_katomic_mc", in_dtype, acc, out_dtype) + f"_{mt}x{kt}x{nt}_c{k_cores}"

    @pl.jit(auto_mutex=True, name=_kname)
    def t12_k_atomic_multicore(
        a: pl.Tensor[[mt, kt], in_dtype],
        b: pl.Tensor[[kt, nt], in_dtype],
        out: pl.Tensor[[mt, nt], out_dtype],
    ):
        a_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[0, 1],
        )
        b_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x40000,
            mutex_ids=[2, 3],
        )
        a_l0a = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Left,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[4, 5],
        )
        b_l0b = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Right,
                layout=pl.ZN,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[6, 7],
        )
        c_l0c = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=acc,
                target_memory=pl.MemorySpace.Acc,
                layout=pl.NZ,
                fractal=1024,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[8],
        )
        with pl.section_cube():
            core_id = pl.get_block_idx()
            for mi in pl.range(0, mt, TILE):
                m_valid = pl.min(TILE, mt - mi)
                for ni in pl.range(0, nt, TILE):
                    n_valid = pl.min(TILE, nt - ni)
                    for ki in pl.range(core_id * TILE, kt, k_cores * TILE):
                        k_valid = pl.min(TILE, kt - ki)
                        cur_a = a_l1.next()
                        cur_b = b_l1.next()
                        al = a_l0a.next()
                        br = b_l0b.next()
                        ac = c_l0c.current()
                        pl.set_validshape(ac, [m_valid, n_valid])
                        pl.set_validshape(cur_a, [m_valid, k_valid])
                        pl.set_validshape(cur_b, [k_valid, n_valid])
                        pl.set_validshape(al, [m_valid, k_valid])
                        pl.set_validshape(br, [k_valid, n_valid])
                        pl.load(cur_a, a, [mi, ki])
                        pl.load(cur_b, b, [ki, ni])
                        pl.move(al, cur_a)
                        pl.move(br, cur_b)
                        pl.matmul(ac, al, br)
                        pl.store(out, ac, [mi, ni], atomic=pl.AtomicType.AtomicAdd)

    return t12_k_atomic_multicore


def make_matmul_insert(in_dtype, out_dtype, acc_dtype=pl.DT_FP32, m_total=M_TOTAL, k_total=K_TOTAL, n_total=N_TOTAL):
    acc = acc_dtype
    quantized_out = out_dtype != acc
    _kname = _kernel_name("t13_insert", in_dtype, acc, out_dtype)

    @pl.jit(auto_mutex=True, name=_kname)
    def t13_insert(
        a: pl.Tensor[[TILE, TILE], in_dtype],
        b: pl.Tensor[[TILE, TILE], in_dtype],
        out: pl.Tensor[[TILE, TILE], out_dtype],
    ):
        a_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=in_dtype, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=0x0,
            mutex_ids=[0],
        )
        a_ub_nd = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=in_dtype, target_memory=pl.MemorySpace.Vec),
            addrs=0x0,
            mutex_ids=[9, 10],
        )
        a_ub_nz = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=in_dtype, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
            addrs=0x10000,
            mutex_ids=[11, 12],
        )
        b_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=in_dtype, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=0x40000,
            mutex_ids=[2, 3],
        )
        a_l0a = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=in_dtype, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=0x0,
            mutex_ids=[4, 5],
        )
        b_l0b = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=in_dtype, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addrs=0x0,
            mutex_ids=[6, 7],
        )
        c_l0c = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE], dtype=acc, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
            ),
            addrs=0x0,
            mutex_ids=[8],
        )

        with pl.section_vector():
            ub_nd = a_ub_nd.next()
            ub_nz = a_ub_nz.next()
            cur_a = a_l1.current()
            pl.load(ub_nd, a, [0, 0])
            pl.move(ub_nz, ub_nd)
            pl.insert(cur_a, ub_nz, [0, 0])
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=2)

        with pl.section_cube():
            cur_a = a_l1.current()
            cur_b = b_l1.next()
            al = a_l0a.next()
            br = b_l0b.next()
            ac = c_l0c.current()
            pl.load(cur_b, b, [0, 0])
            pl.move(br, cur_b)
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=2)
            pl.move(al, cur_a)
            pl.matmul(ac, al, br)
            if quantized_out:
                pl.store(out, ac, [0, 0], scale=_SCALE_ONE)
            else:
                pl.store(out, ac, [0, 0])

    return t13_insert


def make_matmul_transpose(in_dtype, out_dtype, m_total, k_total, n_total, acc_dtype=pl.DT_FP32):
    """t04/t05: M/N/K multi-round loops, both operands transposed (TT)."""
    # One dtype from GM through L1 and L0A/L0B: pl.load and pl.move never cast, so the
    # whole operand path carries in_dtype. Only the accumulator (FP32) and the output
    # differ.
    acc = acc_dtype
    quantized_out = out_dtype != acc
    mt, kt, nt = m_total, k_total, n_total
    _kname = _kernel_name("t04_tt", in_dtype, acc, out_dtype) + f"_{mt}x{kt}x{nt}"

    @pl.jit(auto_mutex=True, name=_kname)
    def t04_mnk_transpose(
        a_t: pl.Tensor[[kt, mt], in_dtype],  # A^T in GM
        b_t: pl.Tensor[[nt, kt], in_dtype],  # B^T in GM
        out: pl.Tensor[[mt, nt], out_dtype],
    ):
        # Tile shapes stay logical; only the L1 layout flips to ZN.
        a_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.ZN,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[0, 1],
        )
        b_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.ZN,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x40000,
            mutex_ids=[2, 3],
        )
        a_l0a = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Left,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[4, 5],
        )
        b_l0b = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Right,
                layout=pl.ZN,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[6, 7],
        )
        c_l0c = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=acc,
                target_memory=pl.MemorySpace.Acc,
                layout=pl.NZ,
                fractal=1024,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[8],
        )
        with pl.section_cube():
            for mi in pl.range(0, mt, TILE):
                m_valid = pl.min(TILE, mt - mi)
                for ni in pl.range(0, nt, TILE):
                    n_valid = pl.min(TILE, nt - ni)
                    ac = c_l0c.current()
                    pl.set_validshape(ac, [m_valid, n_valid])
                    for ki in pl.range(0, kt, TILE):
                        k_valid = pl.min(TILE, kt - ki)
                        cur_a = a_l1.next()
                        cur_b = b_l1.next()
                        al = a_l0a.next()
                        br = b_l0b.next()
                        pl.set_validshape(cur_a, [m_valid, k_valid])
                        pl.set_validshape(cur_b, [k_valid, n_valid])
                        pl.set_validshape(al, [m_valid, k_valid])
                        pl.set_validshape(br, [k_valid, n_valid])
                        pl.load(cur_a, a_t, [ki, mi], order=[1, 0])
                        pl.load(cur_b, b_t, [ni, ki], order=[1, 0])
                        pl.move(al, cur_a)
                        pl.move(br, cur_b)
                        if ki == 0:
                            pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
                        elif ki + TILE < kt:
                            pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
                        else:
                            pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
                    if quantized_out:
                        pl.store(out, ac, [mi, ni], scale=_SCALE_ONE, phase=pl.STPhase.Final)
                    else:
                        pl.store(out, ac, [mi, ni], phase=pl.STPhase.Final)

    return t04_mnk_transpose


def make_matmul_quant_scalar(
    in_dtype, out_dtype, acc_dtype=pl.DT_FP32, m_total=M_TOTAL, k_total=K_TOTAL, n_total=N_TOTAL
):
    """t06: M/N/K multi-round loops, store with a scalar ``scale``, out HF8/FP32."""
    acc = acc_dtype
    mt, kt, nt = m_total, k_total, n_total
    _kname = _kernel_name("t06_qscalar", in_dtype, acc, out_dtype) + f"_{mt}x{kt}x{nt}"

    @pl.jit(auto_mutex=True, name=_kname)
    def t06_mnk_quant_scalar(
        a: pl.Tensor[[mt, kt], in_dtype],
        b: pl.Tensor[[kt, nt], in_dtype],
        out: pl.Tensor[[mt, nt], out_dtype],
    ):
        a_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[0, 1],
        )
        b_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x40000,
            mutex_ids=[2, 3],
        )
        a_l0a = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Left,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[4, 5],
        )
        b_l0b = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Right,
                layout=pl.ZN,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[6, 7],
        )
        c_l0c = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=acc,
                target_memory=pl.MemorySpace.Acc,
                layout=pl.NZ,
                fractal=1024,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[8],
        )
        with pl.section_cube():
            for mi in pl.range(0, mt, TILE):
                m_valid = pl.min(TILE, mt - mi)
                for ni in pl.range(0, nt, TILE):
                    n_valid = pl.min(TILE, nt - ni)
                    ac = c_l0c.current()
                    pl.set_validshape(ac, [m_valid, n_valid])
                    for ki in pl.range(0, kt, TILE):
                        k_valid = pl.min(TILE, kt - ki)
                        cur_a = a_l1.next()
                        cur_b = b_l1.next()
                        al = a_l0a.next()
                        br = b_l0b.next()
                        pl.set_validshape(cur_a, [m_valid, k_valid])
                        pl.set_validshape(cur_b, [k_valid, n_valid])
                        pl.set_validshape(al, [m_valid, k_valid])
                        pl.set_validshape(br, [k_valid, n_valid])
                        pl.load(cur_a, a, [mi, ki])
                        pl.load(cur_b, b, [ki, ni])
                        pl.move(al, cur_a)
                        pl.move(br, cur_b)
                        if ki == 0:
                            pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
                        elif ki + TILE < kt:
                            pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
                        else:
                            pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
                    # Scalar pre-quant on every output dtype, scale = 1.0.
                    pl.store(out, ac, [mi, ni], scale=_SCALE_ONE, phase=pl.STPhase.Final)

    return t06_mnk_quant_scalar


def make_matmul_fp_tile(in_dtype, out_dtype, acc_dtype=pl.DT_FP32, m_total=M_TOTAL, k_total=K_TOTAL, n_total=N_TOTAL):
    """t07: M/N/K multi-round loops, store with a per-column scale, out HF8/FP32."""
    acc = acc_dtype
    mt, kt, nt = m_total, k_total, n_total
    _kname = _kernel_name("t07_fptile", in_dtype, acc, out_dtype) + f"_{mt}x{kt}x{nt}"

    @pl.jit(auto_mutex=True, name=_kname)
    def t07_mnk_fp_tile(
        a: pl.Tensor[[mt, kt], in_dtype],
        b: pl.Tensor[[kt, nt], in_dtype],
        fp_params: pl.Tensor[[1, nt], pl.DT_INT64],
        out: pl.Tensor[[mt, nt], out_dtype],
    ):
        a_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[0, 1],
        )
        b_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x40000,
            mutex_ids=[2, 3],
        )
        fp_mat = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, TILE],
                dtype=pl.DT_INT64,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.ND,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x60000,
            mutex_ids=[9],
        )
        a_l0a = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Left,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[4, 5],
        )
        b_l0b = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Right,
                layout=pl.ZN,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[6, 7],
        )
        fp_scaling = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, TILE],
                dtype=pl.DT_INT64,
                target_memory=pl.MemorySpace.Scaling,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[10],
        )
        c_l0c = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=acc,
                target_memory=pl.MemorySpace.Acc,
                layout=pl.NZ,
                fractal=1024,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[8],
        )
        with pl.section_cube():
            for mi in pl.range(0, mt, TILE):
                m_valid = pl.min(TILE, mt - mi)
                for ni in pl.range(0, nt, TILE):
                    n_valid = pl.min(TILE, nt - ni)
                    ac = c_l0c.current()
                    pl.set_validshape(ac, [m_valid, n_valid])
                    for ki in pl.range(0, kt, TILE):
                        k_valid = pl.min(TILE, kt - ki)
                        cur_a = a_l1.next()
                        cur_b = b_l1.next()
                        al = a_l0a.next()
                        br = b_l0b.next()
                        pl.set_validshape(cur_a, [m_valid, k_valid])
                        pl.set_validshape(cur_b, [k_valid, n_valid])
                        pl.set_validshape(al, [m_valid, k_valid])
                        pl.set_validshape(br, [k_valid, n_valid])
                        pl.load(cur_a, a, [mi, ki])
                        pl.load(cur_b, b, [ki, ni])
                        pl.move(al, cur_a)
                        pl.move(br, cur_b)
                        if ki == 0:
                            pl.matmul(ac, al, br)
                        else:
                            pl.matmul_acc(ac, ac, al, br)
                    cur_fp = fp_mat.current()
                    fp_tile = fp_scaling.current()
                    pl.set_validshape(cur_fp, [1, n_valid])
                    pl.set_validshape(fp_tile, [1, n_valid])
                    pl.load(cur_fp, fp_params, [0, ni])
                    pl.move(fp_tile, cur_fp)
                    pl.store(out, ac, [mi, ni], scale=fp_tile)

    return t07_mnk_fp_tile


def make_matmul_tile_offsets(
    in_dtype, out_dtype, acc_dtype=pl.DT_FP32, m_total=M_TOTAL, k_total=K_TOTAL, n_total=N_TOTAL
):
    """t08: M/N/K multi-round loops, using load_tile / store_tile."""
    acc = acc_dtype
    quantized_out = out_dtype != acc
    mt, kt, nt = m_total, k_total, n_total
    m_blocks, k_blocks, n_blocks = (_ceil_blocks(mt, TILE), _ceil_blocks(kt, TILE), _ceil_blocks(nt, TILE))
    _kname = _kernel_name("t08_tileoff", in_dtype, acc, out_dtype) + f"_{mt}x{kt}x{nt}"

    @pl.jit(auto_mutex=True, name=_kname)
    def t08_mnk_tile_offsets(
        a: pl.Tensor[[mt, kt], in_dtype],
        b: pl.Tensor[[kt, nt], in_dtype],
        out: pl.Tensor[[mt, nt], out_dtype],
    ):
        a_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[0, 1],
        )
        b_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x40000,
            mutex_ids=[2, 3],
        )
        a_l0a = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Left,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[4, 5],
        )
        b_l0b = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Right,
                layout=pl.ZN,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[6, 7],
        )
        c_l0c = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=acc,
                target_memory=pl.MemorySpace.Acc,
                layout=pl.NZ,
                fractal=1024,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[8],
        )
        with pl.section_cube():
            for mi in pl.range(0, m_blocks, 1):
                m_valid = pl.min(TILE, mt - mi * TILE)
                for ni in pl.range(0, n_blocks, 1):
                    n_valid = pl.min(TILE, nt - ni * TILE)
                    ac = c_l0c.current()
                    pl.set_validshape(ac, [m_valid, n_valid])
                    for ki in pl.range(0, k_blocks, 1):
                        k_valid = pl.min(TILE, kt - ki * TILE)
                        cur_a = a_l1.next()
                        cur_b = b_l1.next()
                        al = a_l0a.next()
                        br = b_l0b.next()
                        pl.set_validshape(cur_a, [m_valid, k_valid])
                        pl.set_validshape(cur_b, [k_valid, n_valid])
                        pl.set_validshape(al, [m_valid, k_valid])
                        pl.set_validshape(br, [k_valid, n_valid])
                        # Block indices, not element offsets.
                        pl.load_tile(cur_a, a, [mi, ki])
                        pl.load_tile(cur_b, b, [ki, ni])
                        pl.move(al, cur_a)
                        pl.move(br, cur_b)
                        if ki == 0:
                            pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
                        elif ki + 1 < k_blocks:
                            pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
                        else:
                            pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
                    if quantized_out:
                        pl.store_tile(out, ac, [mi, ni], scale=_SCALE_ONE, phase=pl.STPhase.Final)
                    else:
                        pl.store_tile(out, ac, [mi, ni], phase=pl.STPhase.Final)

    return t08_mnk_tile_offsets


def make_matmul_tile_transpose(
    in_dtype, out_dtype, acc_dtype=pl.DT_FP32, m_total=M_TOTAL, k_total=K_TOTAL, n_total=N_TOTAL
):
    """t09: M/N/K multi-round loops, transposing ``load_tile`` (descending ``order``)."""
    acc = acc_dtype
    quantized_out = out_dtype != acc
    mt, kt, nt = m_total, k_total, n_total
    m_blocks, k_blocks, n_blocks = (_ceil_blocks(mt, TILE), _ceil_blocks(kt, TILE), _ceil_blocks(nt, TILE))
    _kname = _kernel_name("t09_tiletrans", in_dtype, acc, out_dtype) + f"_{mt}x{kt}x{nt}"

    @pl.jit(auto_mutex=True, name=_kname)
    def t09_mnk_tile_transpose(
        a_t: pl.Tensor[[kt, mt], in_dtype],  # A^T in GM
        b: pl.Tensor[[kt, nt], in_dtype],
        out: pl.Tensor[[mt, nt], out_dtype],
    ):
        a_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.ZN,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[0, 1],
        )
        b_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x40000,
            mutex_ids=[2, 3],
        )
        a_l0a = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Left,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[4, 5],
        )
        b_l0b = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Right,
                layout=pl.ZN,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[6, 7],
        )
        c_l0c = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=acc,
                target_memory=pl.MemorySpace.Acc,
                layout=pl.NZ,
                fractal=1024,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[8],
        )
        with pl.section_cube():
            for mi in pl.range(0, m_blocks, 1):
                m_valid = pl.min(TILE, mt - mi * TILE)
                for ni in pl.range(0, n_blocks, 1):
                    n_valid = pl.min(TILE, nt - ni * TILE)
                    ac = c_l0c.current()
                    pl.set_validshape(ac, [m_valid, n_valid])
                    for ki in pl.range(0, k_blocks, 1):
                        k_valid = pl.min(TILE, kt - ki * TILE)
                        cur_a = a_l1.next()
                        cur_b = b_l1.next()
                        al = a_l0a.next()
                        br = b_l0b.next()
                        pl.set_validshape(cur_a, [m_valid, k_valid])
                        pl.set_validshape(cur_b, [k_valid, n_valid])
                        pl.set_validshape(al, [m_valid, k_valid])
                        pl.set_validshape(br, [k_valid, n_valid])
                        pl.load_tile(cur_a, a_t, [ki, mi], order=[1, 0])
                        pl.load_tile(cur_b, b, [ki, ni])
                        pl.move(al, cur_a)
                        pl.move(br, cur_b)
                        if ki == 0:
                            pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
                        elif ki + 1 < k_blocks:
                            pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
                        else:
                            pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
                    if quantized_out:
                        pl.store_tile(out, ac, [mi, ni], scale=_SCALE_ONE, phase=pl.STPhase.Final)
                    else:
                        pl.store_tile(out, ac, [mi, ni], phase=pl.STPhase.Final)

    return t09_mnk_tile_transpose


def make_matmul_tile_mixed(
    in_dtype, out_dtype, acc_dtype=pl.DT_FP32, m_total=M_TOTAL, k_total=K_TOTAL, n_total=N_TOTAL
):
    """t10: M/N/K multi-round loops mixing tile-indexed and element-indexed addressing."""
    # One dtype from GM through L1 and L0A/L0B: pl.load and pl.move never cast, so the
    # whole operand path carries in_dtype. Only the accumulator (FP32) and the output
    # differ.
    acc = acc_dtype
    quantized_out = out_dtype != acc
    mt, kt, nt = m_total, k_total, n_total
    m_blocks, k_blocks, n_blocks = (_ceil_blocks(mt, TILE), _ceil_blocks(kt, TILE), _ceil_blocks(nt, TILE))
    _kname = _kernel_name("t10_tilemixed", in_dtype, acc, out_dtype) + f"_{mt}x{kt}x{nt}"

    @pl.jit(auto_mutex=True, name=_kname)
    def t10_mnk_tile_mixed(
        a: pl.Tensor[[mt, kt], in_dtype],
        b: pl.Tensor[[kt, nt], in_dtype],
        out: pl.Tensor[[mt, nt], out_dtype],
    ):
        a_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[0, 1],
        )
        b_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x40000,
            mutex_ids=[2, 3],
        )
        a_l0a = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Left,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[4, 5],
        )
        b_l0b = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Right,
                layout=pl.ZN,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[6, 7],
        )
        c_l0c = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=acc,
                target_memory=pl.MemorySpace.Acc,
                layout=pl.NZ,
                fractal=1024,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[8],
        )
        with pl.section_cube():
            for mb in pl.range(0, m_blocks, 1):
                m_valid = pl.min(TILE, mt - mb * TILE)
                for nb in pl.range(0, n_blocks, 1):
                    n_valid = pl.min(TILE, nt - nb * TILE)
                    ac = c_l0c.current()
                    pl.set_validshape(ac, [m_valid, n_valid])
                    for kb in pl.range(0, k_blocks, 1):
                        k_valid = pl.min(TILE, kt - kb * TILE)
                        cur_a = a_l1.next()
                        cur_b = b_l1.next()
                        al = a_l0a.next()
                        br = b_l0b.next()
                        pl.set_validshape(cur_a, [m_valid, k_valid])
                        pl.set_validshape(cur_b, [k_valid, n_valid])
                        pl.set_validshape(al, [m_valid, k_valid])
                        pl.set_validshape(br, [k_valid, n_valid])
                        pl.load_tile(cur_a, a, [mb, kb])
                        pl.load(cur_b, b, [kb * TILE, nb * TILE])
                        pl.move(al, cur_a)
                        pl.move(br, cur_b)
                        if kb == 0:
                            pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
                        elif kb + 1 < k_blocks:
                            pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
                        else:
                            pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
                    if quantized_out:
                        pl.store(out, ac, [mb * TILE, nb * TILE], scale=_SCALE_ONE, phase=pl.STPhase.Final)
                    else:
                        pl.store(out, ac, [mb * TILE, nb * TILE], phase=pl.STPhase.Final)

    return t10_mnk_tile_mixed


def make_matmul_tile_order(
    in_dtype, out_dtype, acc_dtype=pl.DT_FP32, m_total=M_TOTAL, k_total=K_TOTAL, n_total=N_TOTAL
):
    """t11: batched (4-D) operands addressed by load_tile / store_tile with ``order``."""
    acc = acc_dtype
    quantized_out = out_dtype != acc
    mt, kt, nt = m_total, k_total, n_total
    m_blocks, k_blocks, n_blocks = (_ceil_blocks(mt, TILE), _ceil_blocks(kt, TILE), _ceil_blocks(nt, TILE))
    _kname = _kernel_name("t11_tileorder", in_dtype, acc, out_dtype) + f"_{mt}x{kt}x{nt}"

    @pl.jit(auto_mutex=True, name=_kname)
    def t11_batched_tile_order(
        a: pl.Tensor[[BATCH, HEADS, mt, kt], in_dtype],
        b: pl.Tensor[[BATCH, HEADS, kt, nt], in_dtype],
        out: pl.Tensor[[BATCH, HEADS, mt, nt], out_dtype],
    ):
        a_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[0, 1],
        )
        b_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x40000,
            mutex_ids=[2, 3],
        )
        a_l0a = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Left,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[4, 5],
        )
        b_l0b = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=in_dtype,
                target_memory=pl.MemorySpace.Right,
                layout=pl.ZN,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[6, 7],
        )
        c_l0c = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=acc,
                target_memory=pl.MemorySpace.Acc,
                layout=pl.NZ,
                fractal=1024,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x0,
            mutex_ids=[8],
        )
        with pl.section_cube():
            for bi in pl.range(0, BATCH, 1):
                for hi in pl.range(0, HEADS, 1):
                    for mb in pl.range(0, m_blocks, 1):
                        m_valid = pl.min(TILE, mt - mb * TILE)
                        for nb in pl.range(0, n_blocks, 1):
                            n_valid = pl.min(TILE, nt - nb * TILE)
                            ac = c_l0c.current()
                            pl.set_validshape(ac, [m_valid, n_valid])
                            for kb in pl.range(0, k_blocks, 1):
                                k_valid = pl.min(TILE, kt - kb * TILE)
                                cur_a = a_l1.next()
                                cur_b = b_l1.next()
                                al = a_l0a.next()
                                br = b_l0b.next()
                                pl.set_validshape(cur_a, [m_valid, k_valid])
                                pl.set_validshape(cur_b, [k_valid, n_valid])
                                pl.set_validshape(al, [m_valid, k_valid])
                                pl.set_validshape(br, [k_valid, n_valid])
                                # bi/hi absolute; mb/kb/nb scaled by TILE via order=[2,3].
                                pl.load_tile(cur_a, a, [bi, hi, mb, kb], order=[2, 3])
                                pl.load_tile(cur_b, b, [bi, hi, kb, nb], order=[2, 3])
                                pl.move(al, cur_a)
                                pl.move(br, cur_b)
                                if kb == 0:
                                    pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
                                elif kb + 1 < k_blocks:
                                    pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
                                else:
                                    pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
                            if quantized_out:
                                pl.store_tile(
                                    out, ac, [bi, hi, mb, nb], order=[2, 3], scale=_SCALE_ONE, phase=pl.STPhase.Final
                                )
                            else:
                                pl.store_tile(out, ac, [bi, hi, mb, nb], order=[2, 3], phase=pl.STPhase.Final)

    return t11_batched_tile_order


# #####################################################################################
# HiFloat8 codec — golden reference, pure numpy
#
# Merged in from what used to be a separate quant_golden.py. That module also carried
# MXFP8 (e4m3) and MXFP4 (e2m1) encoders plus a CLI for writing .bin goldens; this file
# only ever called quant_hif8/dequant_hif8, and nothing else in the repo imported it, so
# only the HF8 half came across.
#
# Semantics mirror the CPU simulator ``include/pto/cpu/TQuant.hpp`` and the testcase
# generator ``tests/npu/a5/src/st/testcase/tquant/gen_data.py`` bit-for-bit, so goldens
# computed here compare directly against kernel output.
#
# HiF8 is 1 sign bit + a variable-length "dot" field that selects the exponent/mantissa
# split (tapered precision), with no per-group scale -- a plain elementwise fp32 -> hif8
# cast using ROUND_A (round away from zero on ties) as the A5 hardware requires.
# #####################################################################################


def _hif8_decode_bitstring(bstr: str) -> float:
    """Decode an 8-char '0'/'1' string (MSB first) to a python float / nan / inf.

    Direct port of the HF8() reference decoder.
    """
    s = bstr[0]
    m = bstr[5:]
    m1, m2, m3 = int(bstr[5]), int(bstr[6]), int(bstr[7])
    if bstr[1] == "1" or bstr[2] == "1":
        d, e = bstr[1:3], bstr[3:5]
    elif bstr[3] == "1":
        d, e = bstr[1:4], bstr[4]
    else:
        d, e = bstr[1:5], ""

    f1 = -1.0 if s == "1" else 1.0

    if d == "0000":
        # Ladder around zero: 2^(m-23). The all-zero-mantissa slot is +0 for
        # s=0 and the single NaN encoding for s=1 (so there is no negative zero).
        if m == "000":
            return float("nan") if s == "1" else 0.0
        return 2.0 ** (m1 * 4 + m2 * 2 + m3 - 23) * f1
    if d == "0001":
        return (1 + (m1 * 4 + m2 * 2 + m3) / 8) * (2.0**0) * f1
    if d == "001":
        f2 = -1 if e == "1" else 1
        return (1 + (m1 * 4 + m2 * 2 + m3) / 8) * (2.0**f2) * f1
    if d == "01":
        e1, e2 = int(bstr[3]), int(bstr[4])
        f2 = -1 if e1 == 1 else 1
        return (1 + (m1 * 4 + m2 * 2 + m3) / 8) * (2.0 ** (f2 * (2 + e2))) * f1
    if d == "10":
        e2, e3 = int(bstr[4]), int(bstr[5])
        f2 = -1 if int(bstr[3]) == 1 else 1
        return (1 + (m2 * 2 + m3) / 4) * (2.0 ** (f2 * (4 + e2 * 2 + e3))) * f1
    # d == "11"
    e2, e3, e4 = int(bstr[4]), int(bstr[5]), int(bstr[6])
    f2 = -1 if int(bstr[3]) == 1 else 1
    if e == "01" and m == "111":
        return f1 * float("inf")
    return (1 + m3 / 2) * (2.0 ** (f2 * (8 + e2 * 4 + e3 * 2 + e4))) * f1


def _build_hif8_tables():
    """Enumerate all 256 codes -> decoded fp32 value. Returns (values, is_special, nan_code).

    ``values`` holds the finite decoded magnitude/value for each code (nan/inf entries hold
    nan). ``is_special`` marks nan/inf codes so the encoder skips them as round targets.
    """
    values = np.empty(256, dtype=np.float64)
    is_special = np.zeros(256, dtype=bool)
    nan_code = None
    for code in range(256):
        bstr = format(code, "08b")
        v = _hif8_decode_bitstring(bstr)
        if v != v:  # nan
            values[code] = np.nan
            is_special[code] = True
            nan_code = code
        elif v in (float("inf"), float("-inf")):
            values[code] = np.nan
            is_special[code] = True
        else:
            values[code] = v
    return values, is_special, nan_code


_HIF8_VALUES, _HIF8_SPECIAL, _HIF8_NAN_CODE = _build_hif8_tables()
# Finite round targets, and the codes they map to.
_HIF8_FINITE_MASK = ~_HIF8_SPECIAL
_HIF8_FINITE_VALS = _HIF8_VALUES[_HIF8_FINITE_MASK].astype(np.float64)
_HIF8_FINITE_CODES = np.nonzero(_HIF8_FINITE_MASK)[0].astype(np.uint8)
_HIF8_MAX_MAG = float(np.nanmax(np.abs(_HIF8_FINITE_VALS)))
# Sorted, unique targets for vectorized nearest-neighbor lookup. Keep the first
# original code for duplicate values, as the scalar reference does.
_HIF8_SORTED_VALS, _HIF8_FIRST = np.unique(_HIF8_FINITE_VALS, return_index=True)
_HIF8_SORTED_CODES = _HIF8_FINITE_CODES[_HIF8_FIRST]


def encode_hif8_scalar(value: float) -> np.uint8:
    """fp32 -> HiF8 byte, nearest value with round-away-from-zero on ties (ROUND_A)."""
    v = float(value)
    if v != v:  # nan
        return np.uint8(_HIF8_NAN_CODE if _HIF8_NAN_CODE is not None else 0x80)
    # Saturate to the representable finite range (SaturationMode default clamps).
    if v > _HIF8_MAX_MAG:
        v = _HIF8_MAX_MAG
    elif v < -_HIF8_MAX_MAG:
        v = -_HIF8_MAX_MAG
    diffs = np.abs(_HIF8_FINITE_VALS - v)
    best = float(diffs.min())
    # Tie -> round away from zero: pick the candidate with the larger magnitude.
    tie_mask = np.isclose(diffs, best, rtol=0.0, atol=0.0)
    cand_vals = _HIF8_FINITE_VALS[tie_mask]
    cand_codes = _HIF8_FINITE_CODES[tie_mask]
    pick = int(np.argmax(np.abs(cand_vals)))
    return np.uint8(cand_codes[pick])


def quant_hif8(src) -> np.ndarray:
    """Elementwise fp32 -> HiF8 byte tile. Returns uint8 array, same shape as src.

    HiF8 carries no per-group scale: this is the pure cast path (cast32toH8).
    """
    src_fp32 = np.asarray(src, dtype=np.float32)
    # Use float64 distances, matching encode_hif8_scalar, including midpoint ties.
    # Widening a signaling NaN quiets it; all NaNs map to the same code below.
    with np.errstate(invalid="ignore"):
        flat = np.clip(src_fp32.reshape(-1).astype(np.float64), -_HIF8_MAX_MAG, _HIF8_MAX_MAG)
    upper = np.searchsorted(_HIF8_SORTED_VALS, flat).clip(0, len(_HIF8_SORTED_VALS) - 1)
    lower = (upper - 1).clip(0)
    low_value, high_value = _HIF8_SORTED_VALS[lower], _HIF8_SORTED_VALS[upper]
    low_distance, high_distance = np.abs(flat - low_value), np.abs(high_value - flat)
    take_high = (high_distance < low_distance) | (
        (high_distance == low_distance) & (np.abs(high_value) > np.abs(low_value))
    )
    out = _HIF8_SORTED_CODES[np.where(take_high, upper, lower)]
    out[np.isnan(flat)] = _HIF8_NAN_CODE if _HIF8_NAN_CODE is not None else 0x80
    return out.reshape(src_fp32.shape)


def dequant_hif8(codes) -> np.ndarray:
    """HiF8 byte tile -> fp32 (for tolerance/debug). nan/inf preserved."""
    c = np.asarray(codes, dtype=np.uint8).reshape(-1)
    out = _HIF8_VALUES[c].astype(np.float32)
    return out.reshape(np.asarray(codes).shape)


# #####################################################################################
# Per-dtype operand builders + golden helpers
# #####################################################################################


def _make_operand(dtype_label, shape, seed, device):
    """Return (device tensor for the kernel, exact FP32 values for the golden).

    HF8: the kernel param is 1 byte/elem, so the device tensor holds HF8 codes as uint8;
    the golden uses the decoded values, which are exactly what the Cube consumes.
    FP32: both are the same fp32 tensor.
    """
    rng = np.random.default_rng(seed)
    ref = rng.standard_normal(size=shape).astype(np.float32)
    if dtype_label == "hf8":
        codes = quant_hif8(ref)
        dev_t = torch.from_numpy(np.ascontiguousarray(codes)).to(device)
        gold = torch.from_numpy(np.ascontiguousarray(dequant_hif8(codes))).to(torch.float32)
    elif dtype_label == "fp32":
        dev_t = torch.from_numpy(np.ascontiguousarray(ref)).to(device)
        gold = torch.from_numpy(np.ascontiguousarray(ref)).to(torch.float32)
    else:
        raise ValueError(f"unsupported input dtype {dtype_label!r}")
    return dev_t, gold


# label -> pl dtype constant / torch output dtype / tolerance per input dtype
_IN_DTYPE = {"hf8": pl.DT_HF8, "fp32": pl.DT_FP32}
_OUT_DTYPE = {
    "fp32": (pl.DT_FP32, torch.float32),
    "hf8": (pl.DT_HF8, torch.uint8),  # HF8 has no torch dtype; carried as raw bytes
}
_IN_TOL = {"hf8": dict(rtol=1e-3, atol=1e-3), "fp32": dict(rtol=1e-3, atol=1e-3)}


def make_fp_params(scales, device):
    """Encode a per-column fp32 scale vector as the INT64 scaling-tile payload.

    Layout mirrors test_scale_store_tile_per_channel.py: the low 32 bits hold the fp32
    bit-pattern of the scale.
    """
    payload = [struct.unpack("!I", struct.pack("!f", np.float32(s)))[0] for s in scales]
    return torch.tensor(payload, dtype=torch.int64).reshape(1, -1).to(device)


def _hif8_rank_table():
    """Sorted table of every finite value HF8 can represent, plus a value->rank lookup.

    Used to measure a code mismatch as a distance in *grid steps* rather than as a
    relative error. HF8 is tapered: the relative gap between neighbouring representable
    values ranges from 1/9 up to 1.0 (near the bottom of the range consecutive codes
    simply double), so no single relative-error threshold means "one step" everywhere.
    Ranking the values sidesteps that entirely.
    """
    vals = dequant_hif8(np.arange(256, dtype=np.uint8)).astype(np.float64)
    finite = np.unique(vals[np.isfinite(vals)])
    return finite


_HIF8_GRID = _hif8_rank_table()


def _assert_hif8_codes_close(out_codes, golden_codes, label, max_steps=1):
    """HF8-output comparison: mostly bit-exact, mismatches within one HF8 grid step.

    A mismatch is scored by how far apart the two codes sit in the sorted table of
    representable HF8 values (see _hif8_rank_table). One step means the two codes are
    neighbours on the grid, i.e. a rounding tie went the other way.
    """
    out_np = out_codes.detach().to("cpu").numpy().astype(np.uint8).reshape(-1)
    gold = golden_codes.astype(np.uint8).reshape(-1)
    mismatch = np.nonzero(out_np != gold)[0]
    frac = 1.0 - mismatch.size / out_np.size
    logging.info("%s: HF8 code exact-match=%.4f (%d/%d differ)", label, frac, mismatch.size, out_np.size)
    if mismatch.size:
        got_v = dequant_hif8(out_np[mismatch]).astype(np.float64)
        exp_v = dequant_hif8(gold[mismatch]).astype(np.float64)
        finite = np.isfinite(got_v) & np.isfinite(exp_v)
        assert finite.all(), f"{label}: non-finite HF8 value among mismatches"
        steps = np.abs(np.searchsorted(_HIF8_GRID, got_v) - np.searchsorted(_HIF8_GRID, exp_v))
        worst = int(steps.max())
        logging.info("%s: worst mismatch = %d grid step(s)", label, worst)
        assert worst <= max_steps, f"{label}: HF8 code differs by {worst} grid steps (max {max_steps})"
    assert frac >= 0.98, f"{label}: only {frac:.3f} of HF8 codes bit-exact"


def _check_output(out, ref, in_dtype, out_dtype, label):
    """Compare a kernel result against the FP32 golden, per output dtype."""
    if out_dtype == "hf8":
        golden_codes = quant_hif8(ref.numpy().astype(np.float32))
        _assert_hif8_codes_close(out, golden_codes, label)
        return
    got = out.cpu()
    expect = ref
    tol = _IN_TOL[in_dtype]
    diff = (got - expect).abs()
    logging.info(
        "%s: max|out-golden|=%.6g  max_rel=%.6g",
        label,
        float(diff.max()),
        float((diff / expect.abs().clamp_min(1e-6)).max()),
    )
    torch.testing.assert_close(got, expect, **tol)


@lru_cache(maxsize=None)
def _cached_kernel(factory, in_dtype, out_dtype, factory_args, factory_kwargs):
    """Reuse the same JIT object during discovery and the real device run.

    Cache dtype labels and scalar shape options; the PyPTO dtype objects themselves
    are not hashable. Runtime operands and per-column scale values are not cached.
    """
    return factory(_IN_DTYPE[in_dtype], _OUT_DTYPE[out_dtype][0], *factory_args, **dict(factory_kwargs))


def _run(
    factory,
    in_dtype,
    out_dtype,
    device,
    mkn=None,
    a_shape=None,
    b_shape=None,
    out_shape=None,
    a_trans=False,
    b_trans=False,
    scales=None,
    factory_args=(),
    label=None,
    block_dim=None,
    **kwargs,
):
    """Build the kernel for one dtype combo, run it, and check against the golden.

    All cases share this one path; the variants differ only in how the operands are shaped
    and how the golden is formed:

    ``a_shape``/``b_shape``/``out_shape``
        Default to the plain 2-D [M,K] @ [K,N] -> [M,N]. Pass explicit shapes for the
        transposed cases (A^T is [K,M], B^T is [N,K]) or the batched case (4-D, where
        torch.matmul batches over the leading axes on its own).
    ``a_trans``/``b_trans``
        Transpose that operand back before computing the golden, i.e. the kernel is
        expected to undo the transposition it was handed.
    ``scales``
        A per-column fp32 array switches on the per-channel store path (scale=<Tile>):
        the kernel takes an extra INT64 [1, N] parameter and the golden is scaled per
        column.
    ``factory_args``
        Positional arguments inserted after (in_dtype, out_dtype) -- used by the transpose
        factory, which takes explicit M/K/N.
    ``mkn``
        (M, K, N) for this run. Drives the operand shapes, the factory's shape arguments
        and hence the golden, all from one place -- pass _TAIL_MKN for the tail variant.
        Ignored for shapes given explicitly via a_shape/b_shape/out_shape.
    ``block_dim``
        Number of cores to launch on, via ``kernel[None, block_dim](...)``. Default None
        launches the single-core way every other case uses. Only t12 needs this: its K
        slices are assigned by ``pl.get_block_idx()``, so the core count is what makes the
        partition happen at all.
    """
    out_torch = _OUT_DTYPE[out_dtype][1]
    m, k, n = mkn or (M_TOTAL, K_TOTAL, N_TOTAL)
    a_shape = a_shape or [m, k]
    b_shape = b_shape or [k, n]
    out_shape = out_shape or [m, n]
    # The factory needs the same shape unless it takes it positionally (transpose case).
    if not factory_args:
        kwargs.setdefault("m_total", m)
        kwargs.setdefault("k_total", k)
        kwargs.setdefault("n_total", n)

    kernel = _cached_kernel(factory, in_dtype, out_dtype, tuple(factory_args), tuple(sorted(kwargs.items())))
    a_dev, a_gold = _make_operand(in_dtype, a_shape, seed=42, device=device)
    b_dev, b_gold = _make_operand(in_dtype, b_shape, seed=43, device=device)
    out = torch.zeros(out_shape, device=device, dtype=out_torch)

    # fp32 golden over the values the Cube actually saw.
    ref = torch.matmul(a_gold.t() if a_trans else a_gold, b_gold.t() if b_trans else b_gold)
    launch = kernel if block_dim is None else kernel[None, block_dim]
    if scales is None:
        launch(a_dev, b_dev, out)
    else:
        fp_params = make_fp_params(scales, device)
        launch(a_dev, b_dev, fp_params, out)
        ref = ref * torch.from_numpy(np.asarray(scales, dtype=np.float32)).reshape(1, -1)
    torch.npu.synchronize()

    label = label or factory.__name__
    tail = "tail" if (m % TILE or k % TILE or n % TILE) else "exact"
    _check_output(out, ref, in_dtype, out_dtype, f"{label}[in={in_dtype},out={out_dtype},{m}x{k}x{n},{tail}]")


# #####################################################################################
# Tests — each loop body swept over dtype combinations
# #####################################################################################

# testcase.md asks for HF8 in / FP32 out; the other combos come free from the factory.
_COMBOS = [
    ("hf8", "fp32"),
    ("fp32", "fp32"),
    ("hf8", "hf8"),
    ("fp32", "hf8"),
]


@pytest.fixture(scope="module")
def _device():
    dev = ST_DEVICE
    _require_a5(dev)
    return dev


# Quantized-store output dtypes (t06/t07). BF16 and INT8 are also legal targets of these
# fixpipe modes (TStore.hpp:141 isQuant branch), but this file is scoped to HF8 and FP32.
_QUANT_OUTS = ["hf8", "fp32"]
_TAIL_MKN = (M_TAIL_TOTAL, K_TAIL_TOTAL, N_TAIL_TOTAL)  # 200 x 152 x 168
_MKN_VARIANTS = [("tail", _TAIL_MKN)]
_TT_SHAPES = [
    ("square_tail", 136, 136, 136),
    ("nonsquare_tail", 136, 152, 200),
]


@pytest.mark.soc("950")
@pytest.mark.parametrize("mkn", [v[1] for v in _MKN_VARIANTS], ids=[v[0] for v in _MKN_VARIANTS])
@pytest.mark.parametrize("in_dtype,out_dtype", _COMBOS)
@pypto.options(pass_options={"enable_slice": False})
def test_t01(in_dtype, out_dtype, mkn, _device):
    """M/N/K multi-round loops with if/elif/else control flow, tail shape."""
    _run(make_matmul_mnk_if, in_dtype, out_dtype, _device, mkn=mkn)


@pytest.mark.soc("950")
@pytest.mark.parametrize("mkn", [v[1] for v in _MKN_VARIANTS], ids=[v[0] for v in _MKN_VARIANTS])
@pytest.mark.parametrize("in_dtype,out_dtype", _COMBOS)
@pypto.options(pass_options={"enable_slice": False})
def test_t02_no_phase(in_dtype, out_dtype, mkn, _device):
    """matmul/matmul_acc without a phase argument (L0C accumulation)."""
    _run(make_matmul_no_phase, in_dtype, out_dtype, _device, mkn=mkn)


@pytest.mark.soc("950")
@pytest.mark.parametrize("mkn", [v[1] for v in _MKN_VARIANTS], ids=[v[0] for v in _MKN_VARIANTS])
@pytest.mark.parametrize("in_dtype", ["hf8", "fp32"])
@pypto.options(pass_options={"enable_slice": False})
def test_t03_atomic(in_dtype, mkn, _device):
    """K accumulated through an atomic-add store into GM.

    FP32 output only: atomic accumulation needs the GM dtype to match the accumulator.
    """
    # K 的原子累加按单核语义验证：直调现为满核，显式请求 1 块。
    _run(make_matmul_atomic, in_dtype, "fp32", _device, mkn=mkn, block_dim=1)


@pytest.mark.soc("950")
@pytest.mark.parametrize("shape_id,m,k,n", _TT_SHAPES, ids=[s[0] for s in _TT_SHAPES])
@pytest.mark.parametrize("in_dtype", ["hf8", "fp32"])
@pypto.options(pass_options={"enable_slice": False})
def test_t04_transpose(in_dtype, shape_id, m, k, n, _device):
    """M/N/K multi-round loops with both operands transposed, square and non-square."""
    _run(
        make_matmul_transpose,
        in_dtype,
        "fp32",
        _device,
        factory_args=(m, k, n),
        a_shape=[k, m],
        b_shape=[n, k],
        out_shape=[m, n],
        a_trans=True,
        b_trans=True,
        label=f"t04_transpose_{m}x{k}x{n}",
    )


@pytest.mark.soc("950")
@pytest.mark.parametrize("out_dtype", _QUANT_OUTS)
@pytest.mark.parametrize("mkn", [v[1] for v in _MKN_VARIANTS], ids=[v[0] for v in _MKN_VARIANTS])
@pytest.mark.parametrize("in_dtype", ["hf8", "fp32"])
@pypto.options(pass_options={"enable_slice": False})
def test_t06_quant_scalar(in_dtype, out_dtype, mkn, _device):
    """store with a scalar scale, out HF8/FP32."""
    _run(make_matmul_quant_scalar, in_dtype, out_dtype, _device, mkn=mkn)


@pytest.mark.soc("950")
@pytest.mark.parametrize("out_dtype", _QUANT_OUTS)
@pytest.mark.parametrize("mkn", [v[1] for v in _MKN_VARIANTS], ids=[v[0] for v in _MKN_VARIANTS])
@pytest.mark.parametrize("in_dtype", ["hf8", "fp32"])
@pypto.options(pass_options={"enable_slice": False})
def test_t07_fp_tile(in_dtype, out_dtype, mkn, _device):
    """store with a per-column scale (Tile form), out HF8/FP32."""
    # One scale per output column, so the vector length follows N (not a fixed constant).
    # Scales near 1.0 keep the product inside every output dtype's range.
    scales = np.linspace(0.5, 1.5, mkn[2]).astype(np.float32)
    _run(make_matmul_fp_tile, in_dtype, out_dtype, _device, mkn=mkn, scales=scales)


@pytest.mark.soc("950")
@pytest.mark.parametrize("mkn", [v[1] for v in _MKN_VARIANTS], ids=[v[0] for v in _MKN_VARIANTS])
@pytest.mark.parametrize("in_dtype,out_dtype", _COMBOS)
@pypto.options(pass_options={"enable_slice": False})
def test_t08_tile_offsets(in_dtype, out_dtype, mkn, _device):
    """load_tile / store_tile block indices (ceil bounds cover the tail)."""
    _run(make_matmul_tile_offsets, in_dtype, out_dtype, _device, mkn=mkn)


@pytest.mark.soc("950")
@pytest.mark.parametrize("mkn", [v[1] for v in _MKN_VARIANTS], ids=[v[0] for v in _MKN_VARIANTS])
@pytest.mark.parametrize("in_dtype,out_dtype", _COMBOS)
@pypto.options(pass_options={"enable_slice": False})
def test_t09_tile_transpose(in_dtype, out_dtype, mkn, _device):
    """Transposing load_tile (descending order), store_tile ascending."""
    _m, _k, _n = mkn
    _run(
        make_matmul_tile_transpose,
        in_dtype,
        out_dtype,
        _device,
        mkn=mkn,
        a_shape=[_k, _m],
        a_trans=True,
        label="t09_tile_transpose",
    )


@pytest.mark.soc("950")
@pytest.mark.parametrize("mkn", [v[1] for v in _MKN_VARIANTS], ids=[v[0] for v in _MKN_VARIANTS])
@pytest.mark.parametrize("in_dtype,out_dtype", _COMBOS)
@pypto.options(pass_options={"enable_slice": False})
def test_t10_tile_mixed(in_dtype, out_dtype, mkn, _device):
    """Tile-indexed and element-indexed addressing mixed in one kernel."""
    _run(make_matmul_tile_mixed, in_dtype, out_dtype, _device, mkn=mkn)


@pytest.mark.soc("950")
@pytest.mark.parametrize("mkn", [v[1] for v in _MKN_VARIANTS], ids=[v[0] for v in _MKN_VARIANTS])
@pytest.mark.parametrize("in_dtype,out_dtype", _COMBOS)
@pypto.options(pass_options={"enable_slice": False})
def test_t11_batched_tile_order(in_dtype, out_dtype, mkn, _device):
    """Batched 4-D operands via load_tile / store_tile with order=[2, 3]."""
    _m, _k, _n = mkn
    _run(
        make_matmul_tile_order,
        in_dtype,
        out_dtype,
        _device,
        mkn=mkn,
        a_shape=[BATCH, HEADS, _m, _k],
        b_shape=[BATCH, HEADS, _k, _n],
        out_shape=[BATCH, HEADS, _m, _n],
        label="t11_batched_tile_order",
    )


@pytest.mark.soc("950")
@pytest.mark.parametrize("mkn", [v[1] for v in _MKN_VARIANTS], ids=[v[0] for v in _MKN_VARIANTS])
@pytest.mark.parametrize("in_dtype", ["hf8", "fp32"])
@pypto.options(pass_options={"enable_slice": False})
def test_t12_katomic_multicore(in_dtype, mkn, _device):
    """K axis split across K_CORES cores, partial sums joined by atomic-add.

    FP32 output only, as in t03: atomic accumulation needs the GM dtype to match the
    accumulator. Unlike t03 the adds come from different cores with no ordering between
    them, so this also covers that the sum is order-independent.
    """
    _run(
        make_matmul_katomic_multicore,
        in_dtype,
        "fp32",
        _device,
        mkn=mkn,
        block_dim=K_CORES,
        label="t12_katomic_multicore",
    )


@pytest.mark.soc("950")
@pytest.mark.parametrize("in_dtype,out_dtype", _COMBOS)
@pypto.options(pass_options={"enable_slice": False})
def test_t13_insert(in_dtype, out_dtype, _device):
    """A staged GM -> UB -> L1 by pl.insert, i.e. whether UB -> L1 carries the input dtype.

    Single [TILE, TILE] block, no shape parametrization: the factory fixes the shapes
    because the question is whether insert moves the bytes intact, not how tiling behaves.
    """
    _run(make_matmul_insert, in_dtype, out_dtype, _device, mkn=(TILE, TILE, TILE), label="t13_insert")


# #####################################################################################
# Main
# #####################################################################################
if __name__ == "__main__":
    dev = ST_DEVICE
    _require_a5(dev)

    # Call the pytest test functions directly so both entry points share one code path.
    # Argument order must match each test's signature after the parametrize decorators.
    all_tests = []
    for _mid, _mkn in _MKN_VARIANTS:
        for _in, _out in _COMBOS:
            for _tag, _fn in (
                ("t01_mnk_if", test_t01),
                ("t02_no_phase", test_t02_no_phase),
                ("t08_tile_offsets", test_t08_tile_offsets),
                ("t09_tile_transpose", test_t09_tile_transpose),
                ("t10_tile_mixed", test_t10_tile_mixed),
                ("t11_batched_tile_order", test_t11_batched_tile_order),
            ):
                all_tests.append((f"{_tag}[in={_in},out={_out},{_mid}]", _fn, (_in, _out, _mkn, dev)))
        for _in in ("hf8", "fp32"):
            all_tests.append((f"t03_atomic[in={_in},{_mid}]", test_t03_atomic, (_in, _mkn, dev)))
            all_tests.append((f"t12_katomic_multicore[in={_in},{_mid}]", test_t12_katomic_multicore, (_in, _mkn, dev)))
            for _out in _QUANT_OUTS:
                all_tests.append(
                    (f"t06_quant_scalar[in={_in},out={_out},{_mid}]", test_t06_quant_scalar, (_in, _out, _mkn, dev))
                )
                all_tests.append((f"t07_fp_tile[in={_in},out={_out},{_mid}]", test_t07_fp_tile, (_in, _out, _mkn, dev)))
    # t04 carries its own shape list (square / non-square, both tail-block shapes).
    for _in in ("hf8", "fp32"):
        for _sid, _m, _k, _n in _TT_SHAPES:
            all_tests.append((f"t04_transpose[in={_in},{_sid}]", test_t04_transpose, (_in, _sid, _m, _k, _n, dev)))
    # t13 is a single TILE-sized block, so it takes no shape argument.
    for _in, _out in _COMBOS:
        all_tests.append((f"t13_insert[in={_in},out={_out}]", test_t13_insert, (_in, _out, dev)))

    logging.info("=" * 80)
    logging.info("Datatype-parametrized matmul, M/N/K multi-round, TILE=%d", TILE)
    logging.info(
        "  tail shapes: M=%d K=%d N=%d (tails %d/%d/%d, none 16-aligned)",
        *_TAIL_MKN,
        _TAIL_MKN[0] % TILE,
        _TAIL_MKN[1] % TILE,
        _TAIL_MKN[2] % TILE,
    )
    logging.info("  full blocks run first on every axis, so the exact-multiple path is covered")
    logging.info("  combos (in,out) = %s", _COMBOS)
    logging.info("  total cases = %d", len(all_tests))
    logging.info("=" * 80)

    passed = 0
    failed = 0
    for name, fn, args in all_tests:
        try:
            fn(*args)
            passed += 1
            logging.info("PASS: %s", name)
        except Exception as exc:
            failed += 1
            logging.error("FAIL: %s -- %s", name, exc)

    logging.info("=" * 80)
    logging.info("Results: %d passed, %d failed, %d total", passed, failed, passed + failed)
    logging.info("=" * 80)
