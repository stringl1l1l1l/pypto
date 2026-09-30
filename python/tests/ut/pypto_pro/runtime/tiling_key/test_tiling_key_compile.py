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
"""Compile smoke tests for wide TilingKeyField bit widths.

The cases verify kernels with 8-bit and 16-bit tiling-key fields compile and
produce kernel.cpp artifacts.
"""

from __future__ import annotations

from pathlib import Path

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField


class TkBits8:
    OpType = TilingKeyField(bits=8, values=list(range(256)))


class TkBits16:
    OpType = TilingKeyField(bits=16, values=list(range(65536)))


def _make_kernel(tiling_key_cls):
    @pl.jit(arch="3510", auto_mutex=True, tiling_key=tiling_key_cls)
    def _kernel(
        x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
        y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
        z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    ):
        m = x.shape[0]
        n = x.shape[1]
        tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        tile_a = pl.make_tile(tile_type, addr=0x0000)
        tile_b = pl.make_tile(tile_type, addr=0x4000)
        tile_c = pl.make_tile(tile_type, addr=0x8000)

        with pl.section_vector():
            for i in pl.range(0, m, 64):
                for j in pl.range(0, n, 128):
                    pl.system.bar_all()
                    pl.load(tile_a, x, [i, j])
                    pl.load(tile_b, y, [i, j])
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

                    if OpType == 0:  # noqa: F821
                        pl.add(tile_c, tile_a, tile_b)
                    elif OpType == 1:  # noqa: F821
                        pl.sub(tile_c, tile_a, tile_b)
                    else:
                        pl.mul(tile_c, tile_a, tile_b)

                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.store(z, tile_c, [i, j])

    return _kernel


def _compile_for_key(kernel, *args, **kwargs):
    return getattr(kernel, "_compile_for_key")(*args, **kwargs)


kernel_bits8 = _make_kernel(TkBits8)


kernel_bits16 = _make_kernel(TkBits16)


def test_bits_8_compile():
    compiled = _compile_for_key(kernel_bits8, {"OpType": 0})
    binary_dir = str(Path(compiled.lib_path).parent)
    assert Path(binary_dir).exists()


def test_bits_16_compile():
    compiled = _compile_for_key(kernel_bits16, {"OpType": 0})
    binary_dir = str(Path(compiled.lib_path).parent)
    assert Path(binary_dir).exists()
