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
"""TilingKey cache tests for multi-field key combinations.

The cases verify selected ModeA/ModeB/Flag combinations produce independent
compiled variants and artifact directories.
"""

from __future__ import annotations

import os
from pathlib import Path

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField


class TkPermutation:
    ModeA = TilingKeyField(bits=2, values=[0, 1, 2, 3])
    ModeB = TilingKeyField(bits=3, values=[0, 1, 2, 3, 4, 5, 6, 7])
    Flag = TilingKeyField(bits=1, values=[0, 1])


def _compile_for_key(kernel, *args, **kwargs):
    return getattr(kernel, "_compile_for_key")(*args, **kwargs)


def _compiled_cache(kernel):
    return getattr(kernel, "_compiled_by_signature")


@pl.jit(arch="3510", auto_mutex=True, tiling_key=TkPermutation)
def kernel_permutation(
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

                if Flag == 0:  # noqa: F821
                    if ModeA == 0:  # noqa: F821
                        pl.add(tile_c, tile_a, tile_b)
                    elif ModeA == 1:  # noqa: F821
                        pl.sub(tile_c, tile_a, tile_b)
                    elif ModeA == 2:  # noqa: F821
                        pl.mul(tile_c, tile_a, tile_b)
                    elif ModeA == 3:  # noqa: F821
                        pl.sub(tile_c, tile_a, tile_b)
                else:
                    if ModeA == 0:  # noqa: F821
                        pl.add(tile_c, tile_a, tile_b)
                    elif ModeA == 1:  # noqa: F821
                        pl.sub(tile_c, tile_a, tile_b)
                    elif ModeA == 2:  # noqa: F821
                        pl.mul(tile_c, tile_a, tile_b)
                    elif ModeA == 3:  # noqa: F821
                        pl.sub(tile_c, tile_a, tile_b)
                    pl.sub(tile_c, tile_c, tile_b)

                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.store(z, tile_c, [i, j])


def test_tilingkey_cache_subdirs():
    """Selected tiling-key combinations compile into independent .so files."""
    combos = [
        {"ModeA": 0, "ModeB": 0, "Flag": 0},
        {"ModeA": 1, "ModeB": 1, "Flag": 0},
        {"ModeA": 2, "ModeB": 3, "Flag": 0},
        {"ModeA": 3, "ModeB": 7, "Flag": 0},
        {"ModeA": 0, "ModeB": 0, "Flag": 1},
        {"ModeA": 1, "ModeB": 1, "Flag": 1},
        {"ModeA": 2, "ModeB": 3, "Flag": 1},
        {"ModeA": 3, "ModeB": 7, "Flag": 1},
    ]

    compiled_variants = []
    for key in combos:
        compiled_variants.append(_compile_for_key(kernel_permutation, key))

    cached = _compiled_cache(kernel_permutation)
    assert len(cached) >= 8, (
        f"Expected >=8 compiled variants, got {len(cached)}"
    )

    lib_paths = set()
    for compiled in compiled_variants:
        assert compiled.lib_path is not None, "Missing lib_path for compiled tiling key"
        assert os.path.isfile(compiled.lib_path), (
            f".so file does not exist: {compiled.lib_path}"
        )
        lib_paths.add(compiled.lib_path)

    assert len(lib_paths) >= 8, (
        f"Expected >=8 unique .so paths, got {len(lib_paths)}"
    )


def test_compile_produces_valid_binary_dir():
    """Compiled artifacts include a binary directory and kernel.cpp."""
    compiled = _compile_for_key(kernel_permutation, {"ModeA": 0, "ModeB": 0, "Flag": 0})
    binary_dir = str(Path(compiled.lib_path).parent)
    bin_path = Path(binary_dir)
    assert bin_path.is_dir()
    kcpp = bin_path / "kernel.cpp"
    assert kcpp.is_file(), f"kernel.cpp missing in {binary_dir}"
