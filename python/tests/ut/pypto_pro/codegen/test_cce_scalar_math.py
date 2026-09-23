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

"""Direct CCE code-generation tests for A5 SIMT scalar math."""

import pypto_pro.language as pl


def _compile_to_cce(kernel) -> str:
    from pypto_pro.runtime.jit import _add_kernel_header, _assemble_cv_source, _parse_and_codegen_targets

    cube, vector = _parse_and_codegen_targets(kernel.to_kernel_def(), "3510", "")
    return _add_kernel_header(_assemble_cv_source(cube, vector)).content


@pl.vector_function(mode="simt", max_threads=1)
def _fp32_math_intrinsics(
    out,
    flags,
    value: pl.DT_FP32,
):
    out[0, 0] = pl.simt.abs(value)
    out[0, 1] = pl.simt.min(value, value)
    out[0, 2] = pl.simt.max(value, value)
    out[0, 3] = pl.simt.sqrt(value)
    out[0, 4] = pl.simt.rsqrt(value)
    out[0, 5] = pl.simt.exp(value)
    out[0, 6] = pl.simt.exp2(value)
    out[0, 7] = pl.simt.log(value)
    out[0, 8] = pl.simt.log2(value)
    out[0, 9] = pl.simt.log1p(value)
    out[0, 10] = pl.simt.sin(value)
    out[0, 11] = pl.simt.cos(value)
    out[0, 12] = pl.simt.tanh(value)
    out[0, 13] = pl.simt.rint(value)
    out[0, 14] = pl.simt.round(value)
    out[0, 15] = pl.simt.floor(value)
    out[0, 16] = pl.simt.ceil(value)
    out[0, 17] = pl.simt.trunc(value)
    out[0, 18] = pl.simt.fma(value, value, value)
    flags[0, 0] = pl.simt.isnan(value)
    flags[0, 1] = pl.simt.isinf(value)


@pl.jit
def _fp32_math_codegen_kernel(value: pl.DT_FP32):
    out_type = pl.TileType(shape=[1, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    flags_type = pl.TileType(shape=[1, 32], dtype=pl.DT_BOOL, target_memory=pl.MemorySpace.Vec)
    out = pl.make_tile(out_type, addr=0x0000)
    flags = pl.make_tile(flags_type, addr=0x0080)
    with pl.section_vector():
        _fp32_math_intrinsics[1](out, flags, value)


@pl.vector_function(mode="simt", max_threads=1)
def _fp16_math_intrinsics(
    out,
    flags,
    source,
):
    value = source[0, 0]
    out[0, 0] = pl.simt.abs(value)
    out[0, 1] = pl.simt.min(value, value)
    out[0, 2] = pl.simt.max(value, value)
    out[0, 3] = pl.simt.sqrt(value)
    out[0, 4] = pl.simt.rsqrt(value)
    out[0, 5] = pl.simt.exp(value)
    out[0, 6] = pl.simt.exp2(value)
    out[0, 7] = pl.simt.log(value)
    out[0, 8] = pl.simt.log2(value)
    out[0, 9] = pl.simt.sin(value)
    out[0, 10] = pl.simt.cos(value)
    out[0, 11] = pl.simt.tanh(value)
    out[0, 12] = pl.simt.rint(value)
    out[0, 13] = pl.simt.round(value)
    out[0, 14] = pl.simt.floor(value)
    out[0, 15] = pl.simt.ceil(value)
    out[0, 16] = pl.simt.trunc(value)
    out[0, 17] = pl.simt.fma(value, value, value)
    flags[0, 0] = pl.simt.isnan(value)
    flags[0, 1] = pl.simt.isinf(value)


@pl.jit
def _fp16_math_codegen_kernel(_jit_entry: pl.DT_INT64):
    out_type = pl.TileType(shape=[1, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    flags_type = pl.TileType(shape=[1, 32], dtype=pl.DT_BOOL, target_memory=pl.MemorySpace.Vec)
    source_type = pl.TileType(shape=[1, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    out = pl.make_tile(out_type, addr=0x0000)
    flags = pl.make_tile(flags_type, addr=0x0040)
    source = pl.make_tile(source_type, addr=0x0080)
    with pl.section_vector():
        _fp16_math_intrinsics[1](out, flags, source)


@pl.vector_function(mode="simt", max_threads=1)
def _bf16_math_intrinsics(
    out,
    flags,
    source,
):
    value = source[0, 0]
    out[0, 0] = pl.simt.abs(value)
    out[0, 1] = pl.simt.min(value, value)
    out[0, 2] = pl.simt.max(value, value)
    out[0, 3] = pl.simt.sqrt(value)
    out[0, 4] = pl.simt.rsqrt(value)
    out[0, 5] = pl.simt.exp(value)
    out[0, 6] = pl.simt.exp2(value)
    out[0, 7] = pl.simt.log(value)
    out[0, 8] = pl.simt.log2(value)
    out[0, 9] = pl.simt.sin(value)
    out[0, 10] = pl.simt.cos(value)
    out[0, 11] = pl.simt.tanh(value)
    out[0, 12] = pl.simt.rint(value)
    out[0, 13] = pl.simt.round(value)
    out[0, 14] = pl.simt.floor(value)
    out[0, 15] = pl.simt.ceil(value)
    out[0, 16] = pl.simt.trunc(value)
    out[0, 17] = pl.simt.fma(value, value, value)
    flags[0, 0] = pl.simt.isnan(value)
    flags[0, 1] = pl.simt.isinf(value)


@pl.jit
def _bf16_math_codegen_kernel(_jit_entry: pl.DT_INT64):
    out_type = pl.TileType(shape=[1, 32], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec)
    flags_type = pl.TileType(shape=[1, 32], dtype=pl.DT_BOOL, target_memory=pl.MemorySpace.Vec)
    source_type = pl.TileType(shape=[1, 32], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec)
    out = pl.make_tile(out_type, addr=0x0000)
    flags = pl.make_tile(flags_type, addr=0x0040)
    source = pl.make_tile(source_type, addr=0x0080)
    with pl.section_vector():
        _bf16_math_intrinsics[1](out, flags, source)


@pl.vector_function(mode="simt", max_threads=1)
def _int64_math_intrinsics(
    out,
    source,
):
    out[0, 0] = pl.simt.abs(source[0, 0])
    out[0, 1] = pl.simt.min(source[0, 0], source[0, 1])
    out[0, 2] = pl.simt.max(source[0, 0], source[0, 1])


@pl.jit
def _int64_math_codegen_kernel(_jit_entry: pl.DT_INT64):
    out_type = pl.TileType(shape=[1, 32], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec)
    source_type = pl.TileType(shape=[1, 32], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec)
    out = pl.make_tile(out_type, addr=0x0000)
    source = pl.make_tile(source_type, addr=0x0040)
    with pl.section_vector():
        _int64_math_intrinsics[1](out, source)


def test_fp32_scalar_math_codegen_maps_native_cce_intrinsics():
    cpp = _compile_to_cce(_fp32_math_codegen_kernel)

    for intrinsic in (
        "__fabsf(",
        "__fminf(",
        "__fmaxf(",
        "__sqrtf(",
        "__expf(",
        "__logf(",
        "__rintf(",
        "__roundf(",
        "__floorf(",
        "__ceilf(",
        "__isnan(",
        "__isinf(",
        "__fma(",
    ):
        assert intrinsic in cpp
    assert "1.0f / __sqrtf(" in cpp
    assert "__expf(" in cpp and "0.6931471805599453f" in cpp
    assert "__logf(1.0f +" in cpp
    assert "__logf(2.0f)" in cpp
    assert "1.0f - (2.0f / (__expf(2.0f *" in cpp
    assert "__fabsf(__t)" in cpp
    assert "float __t = __fma(" not in cpp
    assert "__t = __fma(__t, 0.0f, __t);" in cpp
    assert "simt_api/math_functions.h" not in cpp


def test_fp16_scalar_math_codegen_maps_cce_intrinsics_without_asc_headers():
    cpp = _compile_to_cce(_fp16_math_codegen_kernel)

    for intrinsic in (
        "__sqrtf(",
        "__expf(",
        "__logf(",
        "__rintf(",
        "__floorf(",
        "__ceilf(",
        "__isnan(",
        "__isinf(",
        "__fma(",
        "__hmin_nan(",
        "__hmax_nan(",
        "__cvt_float<",
        "__cvt_half<",
    ):
        assert intrinsic in cpp
    assert "(half)1.0 / __sqrtf(" in cpp
    assert "__cvt_half<ROUND::A," in cpp
    assert "simt_api/asc_fp16.h" not in cpp


def test_bf16_scalar_math_codegen_maps_cce_intrinsics_without_asc_headers():
    cpp = _compile_to_cce(_bf16_math_codegen_kernel)

    for intrinsic in (
        "__rintf(",
        "__floorf(",
        "__ceilf(",
        "__isnan(",
        "__isinf(",
        "__fma(",
        "__min(",
        "__max(",
        "__cvt_float<",
        "__cvt_bfloat16_t<",
    ):
        assert intrinsic in cpp
    assert "__cvt_bfloat16_t<ROUND::A," in cpp
    assert "simt_api/asc_bf16.h" not in cpp


def test_int64_scalar_math_codegen_maps_abs_min_max_and_header():
    cpp = _compile_to_cce(_int64_math_codegen_kernel)

    assert "= abs(" in cpp
    assert "min((int64_t)" in cpp
    assert "max((int64_t)" in cpp
    assert "simt_api/math_functions.h" not in cpp
