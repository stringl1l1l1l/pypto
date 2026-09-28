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
"""CCE codegen tests for per-tile-aware dynamic mutex deduplication."""

from pypto_pro._arch import NpuArch
import pypto_pro.language as pl


def _compile_to_cce(kernel) -> str:
    from pypto_pro.runtime.jit import _assemble_cv_source, _parse_and_codegen_targets

    cube, vector = _parse_and_codegen_targets(kernel.to_kernel_def(), NpuArch.DAV_3510, "")
    return _assemble_cv_source(cube, vector).content


@pl.jit(auto_mutex=True)
def _mixed_multi_mutex_ids_kernel(x: pl.Tensor[[64, 32], pl.DT_FP16]):
    tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    source_group = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    output_group = pl.make_tile_group(type=tile_type, addrs=0x1000, mutex_ids=[[0, 2], [1, 3]])
    with pl.section_vector():
        for index in pl.range(0, 2):
            source_tile = source_group[index]
            output_tile = output_group[index]
            pl.load(source_tile, x, [index * 32, 0])
            pl.move(output_tile, source_tile)
            value = output_tile[0, 0]
            output_tile[1, 0] = value


def test_cce_dynamic_mutex_dedup_skips_same_tile_comparisons():
    cpp = _compile_to_cce(_mixed_multi_mutex_ids_kernel)

    source_mutex = "_tg_source_group_mutex_ids_0[index__iterator_0]"
    output_mutex0 = "_tg_output_group_mutex_ids_0[index__iterator_0]"
    output_mutex1 = "_tg_output_group_mutex_ids_1_0[index__iterator_0]"
    same_tile_guard = f"({output_mutex1} != {output_mutex0})"
    cross_tile_guard = (
        f"({source_mutex} != {output_mutex0}) && "
        f"({source_mutex} != {output_mutex1})"
    )
    assert "source_tile__mutexid" not in cpp
    assert "output_tile__mutexid" not in cpp
    assert same_tile_guard not in cpp
    assert cpp.count(cross_tile_guard) == 2
    assert cpp.count(f"get_buf(PIPE_S, {output_mutex0}, 0);") == 2
    assert cpp.count(f"get_buf(PIPE_S, {output_mutex1}, 0);") == 2
    assert cpp.count(f"rls_buf(PIPE_S, {output_mutex1}, 0);") == 2
    assert cpp.count(f"rls_buf(PIPE_S, {output_mutex0}, 0);") == 2

    acquire_output0 = cpp.index(f"get_buf(PIPE_V, {output_mutex0}, 0);")
    acquire_output1 = cpp.index(f"get_buf(PIPE_V, {output_mutex1}, 0);", acquire_output0)
    acquire_source = cpp.index(cross_tile_guard, acquire_output1)
    release_output0 = cpp.index(f"rls_buf(PIPE_V, {output_mutex0}, 0);", acquire_source)
    release_output1 = cpp.index(f"rls_buf(PIPE_V, {output_mutex1}, 0);", release_output0)
    release_source = cpp.index(cross_tile_guard, release_output1)
    assert acquire_output0 < acquire_output1 < acquire_source
    assert release_output0 < release_output1 < release_source


@pl.jit(auto_mutex=True)
def _auto_mutex_ids_kernel(x: pl.Tensor[[64, 32], pl.DT_FP16]):
    tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    manual_group = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])  # noqa: F841
    auto_group = pl.make_tile_group(type=tile_type, addrs=0x2000, mutex_ids="auto", depth=2)
    with pl.section_vector():
        for index in pl.range(0, 2):
            auto_tile = auto_group[index]
            pl.load(auto_tile, x, [index * 32, 0])
            pl.store(x, auto_tile, [index * 32, 0])


def test_cce_auto_mutex_ids_use_existing_dynamic_lock_path():
    cpp = _compile_to_cce(_auto_mutex_ids_kernel)

    auto_mutex = "_tg_auto_group_mutex_ids_0[index__iterator_0]"
    assert f"get_buf(PIPE_MTE2, {auto_mutex}, 0);" in cpp
    assert f"rls_buf(PIPE_MTE2, {auto_mutex}, 0);" in cpp
    assert f"get_buf(PIPE_MTE3, {auto_mutex}, 0);" in cpp
    assert f"rls_buf(PIPE_MTE3, {auto_mutex}, 0);" in cpp
