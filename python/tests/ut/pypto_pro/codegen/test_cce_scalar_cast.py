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

"""CCE code-generation tests for A5 SIMT scalar_cast."""

import re

import pypto_pro.language as pl


def _compile_to_cce(kernel) -> str:
    from pypto_pro.runtime.jit import _add_kernel_header, _assemble_cv_source, _parse_and_codegen_targets

    cube, vector = _parse_and_codegen_targets(kernel.to_kernel_def(), "3510", "")
    return _add_kernel_header(_assemble_cv_source(cube, vector)).content


@pl.vector_function(mode="simt", max_threads=1)
def _scalar_cast_intrinsics(
    half_out,
    bfloat_out,
    integer_out,
    float_out,
    plain_out,
    value: pl.DT_FP32,
    integer: pl.DT_INT64,
):
    half_out[0, 0] = pl.simt.cast(value, pl.DT_FP16, mode=pl.RoundMode.CAST_RINT)
    bfloat_out[0, 0] = pl.simt.cast(value, pl.DT_BF16, mode=pl.RoundMode.CAST_ROUND)
    integer_out[0, 0] = pl.simt.cast(value, pl.DT_INT32, mode=pl.RoundMode.CAST_FLOOR)
    float_out[0, 0] = pl.simt.cast(integer, pl.DT_FP32, mode=pl.RoundMode.CAST_CEIL)
    plain_out[0, 0] = pl.simt.cast(integer, pl.DT_INT32)


@pl.jit
def _scalar_cast_codegen_kernel(value: pl.DT_FP32, integer: pl.DT_INT64):
    half_type = pl.TileType(shape=[1, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    bfloat_type = pl.TileType(shape=[1, 16], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec)
    int_type = pl.TileType(shape=[1, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    float_type = pl.TileType(shape=[1, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    half_out = pl.make_tile(half_type, addr=0x0000)
    bfloat_out = pl.make_tile(bfloat_type, addr=0x0040)
    integer_out = pl.make_tile(int_type, addr=0x0080)
    float_out = pl.make_tile(float_type, addr=0x00C0)
    plain_out = pl.make_tile(int_type, addr=0x0100)
    with pl.section_vector():
        _scalar_cast_intrinsics[1](half_out, bfloat_out, integer_out, float_out, plain_out, value, integer)


def _simt_function_source(cpp: str, name: str) -> str:
    start = cpp.index(f"inline void {name}(")
    end = cpp.index("\n}\n", start)
    return cpp[start:end]


def test_scalar_cast_codegen_maps_round_modes_and_cce_builtins():
    cpp = _compile_to_cce(_scalar_cast_codegen_kernel)
    function = _simt_function_source(cpp, "_scalar_cast_intrinsics")

    assert "__cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(" in function
    assert "__cvt_bfloat16_t<ROUND::A, RoundingSaturation::RS_DISABLE_VALUE>(" in function
    assert "__cvt_int32_t<ROUND::F, RoundingSaturation::RS_ENABLE_VALUE>(" in function
    assert "__cvt_float<ROUND::C, RoundingSaturation::RS_DISABLE_VALUE>(" in function
    assert re.search(r"plain_out(?:_\d+)?\[[^]]+\]\s*=\s*\(\(int32_t\)integer(?:_\d+)?\)", function)
    assert "simt_api/" not in cpp
    assert "__float2half" not in cpp
    assert "__float2bfloat16" not in cpp
    assert "__float2int" not in cpp


@pl.vector_function(mode="simt", max_threads=1)
def _scalar_cast_all_intrinsics(
    half_out,
    bfloat_out,
    int32_out,
    uint32_out,
    int64_out,
    uint64_out,
    float_out,
    value: pl.DT_FP32,
    int32_value: pl.DT_INT32,
    uint32_value: pl.DT_UINT32,
    int64_value: pl.DT_INT64,
    uint64_value: pl.DT_UINT64,
):
    half_out[0, 0] = pl.simt.cast(value, pl.DT_FP16)
    half_out[0, 0] = pl.simt.cast(value, pl.DT_FP16, mode=pl.RoundMode.CAST_RINT)
    half_out[0, 0] = pl.simt.cast(value, pl.DT_FP16, mode=pl.RoundMode.CAST_ROUND)
    half_out[0, 0] = pl.simt.cast(value, pl.DT_FP16, mode=pl.RoundMode.CAST_FLOOR)
    half_out[0, 0] = pl.simt.cast(value, pl.DT_FP16, mode=pl.RoundMode.CAST_CEIL)
    half_out[0, 0] = pl.simt.cast(value, pl.DT_FP16, mode=pl.RoundMode.CAST_TRUNC)
    half_out[0, 0] = pl.simt.cast(value, pl.DT_FP16, mode=pl.RoundMode.CAST_ODD)

    bfloat_out[0, 0] = pl.simt.cast(value, pl.DT_BF16)
    bfloat_out[0, 0] = pl.simt.cast(value, pl.DT_BF16, mode=pl.RoundMode.CAST_RINT)
    bfloat_out[0, 0] = pl.simt.cast(value, pl.DT_BF16, mode=pl.RoundMode.CAST_ROUND)
    bfloat_out[0, 0] = pl.simt.cast(value, pl.DT_BF16, mode=pl.RoundMode.CAST_FLOOR)
    bfloat_out[0, 0] = pl.simt.cast(value, pl.DT_BF16, mode=pl.RoundMode.CAST_CEIL)
    bfloat_out[0, 0] = pl.simt.cast(value, pl.DT_BF16, mode=pl.RoundMode.CAST_TRUNC)

    float_out[0, 0] = pl.simt.cast(pl.simt.cast(value, pl.DT_FP16), pl.DT_FP32)
    float_out[0, 0] = pl.simt.cast(pl.simt.cast(value, pl.DT_BF16), pl.DT_FP32)
    int32_out[0, 0] = pl.simt.cast(value, pl.DT_INT32, mode=pl.RoundMode.CAST_RINT)
    uint32_out[0, 0] = pl.simt.cast(value, pl.DT_UINT32, mode=pl.RoundMode.CAST_ROUND)
    int64_out[0, 0] = pl.simt.cast(value, pl.DT_INT64, mode=pl.RoundMode.CAST_FLOOR)
    uint64_out[0, 0] = pl.simt.cast(value, pl.DT_UINT64, mode=pl.RoundMode.CAST_CEIL)
    int32_out[0, 0] = pl.simt.cast(value, pl.DT_INT32, mode=pl.RoundMode.CAST_TRUNC)
    float_out[0, 0] = pl.simt.cast(int32_value, pl.DT_FP32, mode=pl.RoundMode.CAST_RINT)
    float_out[0, 0] = pl.simt.cast(uint32_value, pl.DT_FP32, mode=pl.RoundMode.CAST_ROUND)
    float_out[0, 0] = pl.simt.cast(int64_value, pl.DT_FP32, mode=pl.RoundMode.CAST_FLOOR)
    float_out[0, 0] = pl.simt.cast(uint64_value, pl.DT_FP32, mode=pl.RoundMode.CAST_CEIL)
    float_out[0, 0] = pl.simt.cast(int32_value, pl.DT_FP32, mode=pl.RoundMode.CAST_TRUNC)
    int32_out[0, 0] = pl.simt.cast(int64_value, pl.DT_INT32)


@pl.jit
def _scalar_cast_all_codegen_kernel(
    value: pl.DT_FP32,
    int32_value: pl.DT_INT32,
    uint32_value: pl.DT_UINT32,
    int64_value: pl.DT_INT64,
    uint64_value: pl.DT_UINT64,
):
    half_type = pl.TileType(shape=[1, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    bfloat_type = pl.TileType(shape=[1, 16], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec)
    int32_type = pl.TileType(shape=[1, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    uint32_type = pl.TileType(shape=[1, 8], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    int64_type = pl.TileType(shape=[1, 4], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec)
    uint64_type = pl.TileType(shape=[1, 4], dtype=pl.DT_UINT64, target_memory=pl.MemorySpace.Vec)
    float_type = pl.TileType(shape=[1, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    half_out = pl.make_tile(half_type, addr=0x0000)
    bfloat_out = pl.make_tile(bfloat_type, addr=0x0040)
    int32_out = pl.make_tile(int32_type, addr=0x0080)
    uint32_out = pl.make_tile(uint32_type, addr=0x00C0)
    int64_out = pl.make_tile(int64_type, addr=0x0100)
    uint64_out = pl.make_tile(uint64_type, addr=0x0140)
    float_out = pl.make_tile(float_type, addr=0x0180)
    with pl.section_vector():
        _scalar_cast_all_intrinsics[1](
            half_out,
            bfloat_out,
            int32_out,
            uint32_out,
            int64_out,
            uint64_out,
            float_out,
            value,
            int32_value,
            uint32_value,
            int64_value,
            uint64_value,
        )


def test_scalar_cast_codegen_covers_supported_intrinsic_families():
    cpp = _compile_to_cce(_scalar_cast_all_codegen_kernel)
    function = _simt_function_source(cpp, "_scalar_cast_all_intrinsics")

    for intrinsic in (
        "__cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(",
        "__cvt_half<ROUND::A, RoundingSaturation::RS_DISABLE_VALUE>(",
        "__cvt_half<ROUND::F, RoundingSaturation::RS_DISABLE_VALUE>(",
        "__cvt_half<ROUND::C, RoundingSaturation::RS_DISABLE_VALUE>(",
        "__cvt_half<ROUND::Z, RoundingSaturation::RS_DISABLE_VALUE>(",
        "__cvt_half<ROUND::O, RoundingSaturation::RS_DISABLE_VALUE>(",
        "__cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(",
        "__cvt_bfloat16_t<ROUND::A, RoundingSaturation::RS_DISABLE_VALUE>(",
        "__cvt_bfloat16_t<ROUND::F, RoundingSaturation::RS_DISABLE_VALUE>(",
        "__cvt_bfloat16_t<ROUND::C, RoundingSaturation::RS_DISABLE_VALUE>(",
        "__cvt_bfloat16_t<ROUND::Z, RoundingSaturation::RS_DISABLE_VALUE>(",
        "__cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(",
        "__cvt_float<ROUND::A, RoundingSaturation::RS_DISABLE_VALUE>(",
        "__cvt_float<ROUND::F, RoundingSaturation::RS_DISABLE_VALUE>(",
        "__cvt_float<ROUND::C, RoundingSaturation::RS_DISABLE_VALUE>(",
        "__cvt_float<ROUND::Z, RoundingSaturation::RS_DISABLE_VALUE>(",
        "__cvt_int32_t<ROUND::R, RoundingSaturation::RS_ENABLE_VALUE>(",
        "__cvt_uint32_t<ROUND::A, RoundingSaturation::RS_ENABLE_VALUE>(",
        "__cvt_int64_t<ROUND::F, RoundingSaturation::RS_ENABLE_VALUE>(",
        "__cvt_uint64_t<ROUND::C, RoundingSaturation::RS_ENABLE_VALUE>(",
        "__cvt_int32_t<ROUND::Z, RoundingSaturation::RS_ENABLE_VALUE>(",
    ):
        assert intrinsic in function
    assert "simt_api/" not in cpp
    assert "__float2half" not in cpp
    assert "__float2bfloat16" not in cpp
    assert "__float2int" not in cpp
