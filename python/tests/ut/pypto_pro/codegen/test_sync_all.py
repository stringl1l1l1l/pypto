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
"""CCE codegen smoke tests for system.sync_all.

Each test compiles a @pl.jit with pl.system.sync_all() and verifies that the
generated CCE C++ output contains the expected SYNCALL calls with the correct
core_type attribute. Covered cases include the default Mix and explicit AIVOnly.
"""

import logging

import pypto_pro.language as pl


def _compile_to_cce(kernel) -> str:
    from pypto_pro.runtime.jit import _assemble_cv_source, _parse_and_codegen_targets

    cube, vector = _parse_and_codegen_targets(kernel.to_kernel_def(), "a5", "")
    return _assemble_cv_source(cube, vector).content


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@pl.jit
def _sync_all_mix_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile(tile_type, addr=0x0000)
    pl.load(tile_a, a, [0, 0])
    pl.system.sync_all()


@pl.jit
def _sync_all_aiv_only_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile(tile_type, addr=0x0000)
    pl.load(tile_a, a, [0, 0])
    pl.system.sync_all(core_type=pl.SyncCoreType.AIV_ONLY)


# ---------------------------------------------------------------------------
# CCE C++ tests
# ---------------------------------------------------------------------------


def test_sync_all_mix_cce():
    cpp = _compile_to_cce(_sync_all_mix_kernel)
    logging.info("CCE C++ output:\n%s", cpp)
    assert "SYNCALL<SyncCoreType::Mix>()" in cpp


def test_sync_all_aiv_only_cce():
    cpp = _compile_to_cce(_sync_all_aiv_only_kernel)
    logging.info("CCE C++ output:\n%s", cpp)
    assert "SYNCALL<SyncCoreType::AIVOnly>()" in cpp


# ---------------------------------------------------------------------------
# Negative: old manual-expansion primitives should NOT appear
# ---------------------------------------------------------------------------


def test_no_manual_sync_primitives_in_cce():
    cpp = _compile_to_cce(_sync_all_mix_kernel)
    assert "ffts_cross_core_sync" not in cpp
    assert "wait_flag_dev" not in cpp
    assert "set_intra_block" not in cpp
    assert "wait_intra_block" not in cpp
