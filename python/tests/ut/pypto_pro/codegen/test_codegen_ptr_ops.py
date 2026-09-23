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
"""CCE code-generation tests for pointer and tensor re-view APIs."""

import pypto_pro.language as pl


def _compile_to_cce(kernel) -> str:
    from pypto_pro.runtime.jit import _assemble_cv_source, _parse_and_codegen_targets

    cube, vector = _parse_and_codegen_targets(kernel.to_kernel_def(), "3510", "")
    return _assemble_cv_source(cube, vector).content


@pl.jit
def _make_ptr_cce_kernel(p: pl.Ptr[pl.DT_UINT8], out: pl.Tensor[[64, 128], pl.DT_FP16]):
    fp16_ptr = pl.make_ptr(p, dtype=pl.DT_FP16)
    ws = pl.make_tensor(fp16_ptr, [64, 128], [128, 1])
    tile = pl.make_tile(
        pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addr=0x0,
    )
    with pl.section_vector():
        pl.load(tile, ws, [0, 0])
        pl.store(out, tile, [0, 0])


@pl.jit
def _retensor_cce_kernel(src: pl.Tensor[[64, 128], pl.DT_FP16], out: pl.Tensor[[64, 128], pl.DT_FP16]):
    reshaped = pl.make_tensor(src, [64, 128], [128, 1])
    tile = pl.make_tile(
        pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addr=0x0,
    )
    with pl.section_vector():
        pl.load(tile, reshaped, [0, 0])
        pl.store(out, tile, [0, 0])


@pl.jit
def _retensor_cce_dtype_kernel(src: pl.Tensor[[64, 128], pl.DT_FP16], out: pl.Tensor[[64, 256], pl.DT_UINT8]):
    as_u8 = pl.make_tensor(src, [64, 256], [256, 1], dtype=pl.DT_UINT8)
    tile = pl.make_tile(
        pl.TileType(shape=[64, 256], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec),
        addr=0x0,
    )
    with pl.section_vector():
        pl.load(tile, as_u8, [0, 0])
        pl.store(out, tile, [0, 0])


def test_make_ptr_cce_emits_reinterpret_cast():
    cpp = _compile_to_cce(_make_ptr_cce_kernel)
    # Raw uint8 param p reinterpreted as half*.
    assert "((__gm__ half*)(p_0))" in cpp


def test_make_tensor_from_tensor_cce_reuses_source_pointer():
    cpp = _compile_to_cce(_retensor_cce_kernel)
    # The new view's GlobalTensor is constructed from the source param's pointer (src_0_ptr),
    # not from a fresh allocation.
    assert "src_0_ptr" in cpp
    assert "reshaped_0(src_0_ptr" in cpp


def test_make_tensor_from_tensor_cce_dtype_cast():
    cpp = _compile_to_cce(_retensor_cce_dtype_kernel)
    # Reinterpreting fp16 -> uint8 emits a cast of the source pointer.
    assert "(__gm__ uint8_t*)(src_0_ptr)" in cpp
