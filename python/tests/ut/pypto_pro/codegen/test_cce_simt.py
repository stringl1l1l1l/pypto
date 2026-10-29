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

"""Representative DSL-to-CCE integration tests for the A5 SIMT path.

Exhaustive dtype contracts and operation-to-intrinsic mappings belong to the
native tests. This module covers representative frontend-to-backend behavior.
"""

import re

from pypto_pro._errors import NotSupported
import pypto_pro.language as pl
from pypto_pro.runtime.platform import NpuArch
import pytest


def _compile_to_cce(kernel, arch: NpuArch = NpuArch.DAV_3510) -> str:
    from pypto_pro.runtime.jit import _add_kernel_header, _assemble_cv_source, _parse_and_codegen_targets
    from pypto_pro.runtime.platform import normalize_arch

    cube, vector = _parse_and_codegen_targets(kernel.to_kernel_def(), normalize_arch(arch), "")
    return _add_kernel_header(_assemble_cv_source(cube, vector)).content


@pl.vector_function(mode="simt", max_threads=256)
def _tile_add(
    dst,
    src,
    n: pl.DT_UINT32,
    delta: pl.DT_FP32,
):
    tid = pl.simt.linear_thread_idx()
    if tid < n:
        dst[0, tid] = src[0, tid] + delta


@pl.vector_function(mode="simt", max_threads=256)
def _gm_add(
    dst: pl.Tensor[[1, 256], pl.DT_FP32],
    src: pl.Tensor[[1, 256], pl.DT_FP32],
    n: pl.DT_UINT32,
    delta: pl.DT_FP32,
):
    tid = pl.simt.linear_thread_idx()
    if tid < n:
        dst[0, tid] = src[0, tid] + delta


@pl.vector_function(mode="simt")
def _callee_add(value: pl.DT_INT32, delta: pl.DT_INT32) -> pl.DT_INT32:
    return value + delta


@pl.vector_function(mode="simt")
def _callee_load(src: pl.Tensor[[1, 32], pl.DT_INT32], index: pl.DT_UINT32) -> pl.DT_INT32:
    return src[0, index]


@pl.vector_function(mode="simt")
def _callee_store(
    dst,
    index: pl.DT_UINT32,
    value: pl.DT_INT32,
):
    dst[0, index] = value


@pl.vector_function(mode="simt")
def _callee_apply(
    dst,
    src: pl.Tensor[[1, 32], pl.DT_INT32],
    index: pl.DT_UINT32,
    delta: pl.DT_INT32,
):
    value = _callee_add(_callee_load(src, index), delta)
    _callee_store(dst, index, value)


@pl.vector_function(mode="simt", max_threads=32)
def _callee_entry(
    dst,
    src: pl.Tensor[[1, 32], pl.DT_INT32],
    delta: pl.DT_INT32,
):
    tid = pl.simt.linear_thread_idx()
    _callee_apply(dst, src, tid, delta)


@pl.vector_function(mode="simt", max_threads=256)
def _context_probe(dst):
    thread = pl.simt.thread_idx()
    block = pl.simt.block_dim()
    block_id = pl.simt.block_idx()
    grid = pl.simt.grid_dim()
    tid = pl.simt.linear_thread_idx()
    value = (
        thread.x
        + thread.y
        + thread.z
        + block.x
        + block.y
        + block.z
        + block_id.x
        + block_id.y
        + block_id.z
        + grid.x
        + grid.y
        + grid.z
        + pl.simt.warp_size()
    )
    dst[0, tid] = value


@pl.vector_function(mode="simt", max_threads=32)
def _warp_codegen_probe(value: pl.DT_FP32, int_output, uint_output, float_output):
    lane = pl.simt.lane_id()
    int_output[0, lane] = lane
    predicate = lane < 16
    uint_output[0, lane] = pl.simt.warp_ballot(predicate)
    float_output[0, lane] = pl.simt.warp_shfl_down(value, 1, width=8)
    float_output[1, lane] = pl.simt.warp_reduce_add(value)


@pl.vector_function(mode="simt", max_threads=256)
def _tile_valid_shape_access(
    dst,
    src,
):
    tid = pl.simt.linear_thread_idx()
    rows = src.valid_shape[0]
    cols = src.valid_shape[1]
    row = tid // cols
    col = tid % cols
    if row < rows:
        dst[row, col] = src[row, col]


@pl.jit
def _simt_tile_codegen_kernel(
    x: pl.Tensor[[1, 256], pl.DT_FP32],
    out: pl.Tensor[[1, 256], pl.DT_FP32],
    n: pl.DT_UINT32,
    delta: pl.DT_FP32,
):
    tile_type = pl.TileType(shape=[1, 256], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    src = pl.make_tile(tile_type, addr=0x0000)
    dst = pl.make_tile(tile_type, addr=0x0400)
    with pl.section_vector():
        pl.load(src, x, [0, 0])
        pl.load(dst, x, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        _tile_add[256](dst, src, n, delta)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, dst, [0, 0])


@pl.jit
def _simt_gm_codegen_kernel(
    x: pl.Tensor[[1, 256], pl.DT_FP32],
    out: pl.Tensor[[1, 256], pl.DT_FP32],
    n: pl.DT_UINT32,
    delta: pl.DT_FP32,
):
    with pl.section_vector():
        _gm_add[256](out, x, n, delta)


@pl.jit
def _simt_context_codegen_kernel(_jit_entry: pl.DT_INT64):
    tile_type = pl.TileType(shape=[1, 256], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    dst = pl.make_tile(tile_type, addr=0x0000)
    with pl.section_vector():
        _context_probe[8, 4, 8](dst)


@pl.jit
def _simt_warp_codegen_kernel(value: pl.DT_FP32):
    int_output = pl.make_tile(
        pl.TileType(shape=[1, 32], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addr=0x0000,
    )
    uint_output = pl.make_tile(
        pl.TileType(shape=[1, 32], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addr=0x0080,
    )
    float_output = pl.make_tile(
        pl.TileType(shape=[2, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x0100,
    )
    with pl.section_vector():
        _warp_codegen_probe[32](value, int_output, uint_output, float_output)


@pl.jit
def _simt_callee_codegen_kernel(src: pl.Tensor[[1, 32], pl.DT_INT32], delta: pl.DT_INT32):
    tile_type = pl.TileType(shape=[1, 32], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    dst = pl.make_tile(tile_type, addr=0x0000)
    with pl.section_vector():
        _callee_entry[32](dst, src, delta)


@pl.jit
def _simt_valid_shape_codegen_kernel(valid_rows: pl.DT_UINT32, valid_cols: pl.DT_UINT32):
    tile_type = pl.TileType(
        shape=[8, 64],
        dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
    )
    dst = pl.make_tile(tile_type, addr=0x0000)
    src = pl.make_tile(tile_type, addr=0x0800)
    with pl.section_vector():
        pl.set_validshape(dst, [valid_rows, valid_cols])
        pl.set_validshape(src, [valid_rows, valid_cols])
        _tile_valid_shape_access[256](dst, src)


@pl.vector_function(mode="simt", max_threads=32)
def _atomic_add_ub(
    dst,
    old_values,
    value: pl.DT_INT32,
):
    tid = pl.simt.linear_thread_idx()
    old_values[0, tid] = pl.simt.atomic_add(dst[0, 0], value)


@pl.vector_function(mode="simt", max_threads=32)
def _atomic_add_gm(
    dst: pl.Tensor[[1, 32], pl.DT_INT64],
    old_values: pl.Tensor[[1, 32], pl.DT_INT64],
    value: pl.DT_INT64,
):
    tid = pl.simt.linear_thread_idx()
    old_values[0, tid] = pl.simt.atomic_add(dst[0, 0], value)


@pl.vector_function(mode="simt", max_threads=1)
def _atomic_add_discard(dst, value: pl.DT_INT32):
    pl.simt.atomic_add(dst[0, 0], value)


@pl.vector_function(mode="simt", max_threads=1)
def _atomic_half_discard(
    ub,
    gm: pl.Tensor[[1, 3], pl.DT_BF16],
):
    pl.simt.atomic_add(ub[0, 0], 1.0)
    pl.simt.atomic_max(ub[0, 1], 2.0)
    pl.simt.atomic_min(ub[0, 2], 3.0)
    pl.simt.atomic_add(gm[0, 0], 1.0)
    pl.simt.atomic_max(gm[0, 1], 2.0)
    pl.simt.atomic_min(gm[0, 2], 3.0)


@pl.jit
def _atomic_add_ub_codegen_kernel(value: pl.DT_INT32):
    tile_type = pl.TileType(shape=[1, 32], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    dst = pl.make_tile(tile_type, addr=0x0000)
    old_values = pl.make_tile(tile_type, addr=0x0080)
    with pl.section_vector():
        _atomic_add_ub[32](dst, old_values, value)


@pl.jit
def _atomic_add_gm_codegen_kernel(
    dst: pl.Tensor[[1, 32], pl.DT_INT64],
    old_values: pl.Tensor[[1, 32], pl.DT_INT64],
    value: pl.DT_INT64,
):
    with pl.section_vector():
        _atomic_add_gm[32](dst, old_values, value)


@pl.jit
def _atomic_add_discard_codegen_kernel(value: pl.DT_INT32):
    tile_type = pl.TileType(shape=[1, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    dst = pl.make_tile(tile_type, addr=0x0000)
    with pl.section_vector():
        _atomic_add_discard[1](dst, value)


@pl.jit
def _atomic_half_discard_codegen_kernel(gm: pl.Tensor[[1, 3], pl.DT_BF16]):
    tile_type = pl.TileType(shape=[1, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    ub = pl.make_tile(tile_type, addr=0x0000)
    with pl.section_vector():
        _atomic_half_discard[1](ub, gm)


def _simt_function_source(cpp: str, name: str) -> str:
    start = cpp.index(f"inline void {name}(")
    end = cpp.index("\n}\n", start)
    return cpp[start:end]


def test_simt_tile_codegen_emits_direct_cce_function_launch_and_ub_accesses():
    cpp = _compile_to_cce(_simt_tile_codegen_kernel)

    signature = next(line for line in cpp.splitlines() if "inline void _tile_add(" in line)
    function_pos = cpp.index(signature)
    kernel_pos = cpp.index("__aicore__ inline void _simt_tile_codegen_kernel_impl_vector")
    launch_line = next(line for line in cpp.splitlines() if "cce::async_invoke<_tile_add>" in line)
    function = cpp[function_pos:kernel_pos]

    assert function_pos < kernel_pos
    assert "__simt_vf__ __launch_bounds__(256)" in signature
    for tile in ("dst", "src"):
        assert re.search(rf"__ubuf__\s+float\s*\*\s*{tile}(?:_\d+)*\b", signature)
        for axis in ("row", "col"):
            assert re.search(rf"\buint32_t\s+{tile}(?:_\d+)*__valid_{axis}\b", signature)
    assert "cce::async_invoke<_tile_add>(cce::dim3{256, 1, 1}" in launch_line
    assert launch_line.count("(__ubuf__ float*)") == 2
    assert launch_line.count(".tile.data()") == 2
    assert launch_line.count(".tile.GetValidRow()") == 2
    assert launch_line.count(".tile.GetValidCol()") == 2
    assert "pipe_barrier(PIPE_ALL);" not in cpp
    assert re.search(r"\bdst(?:_\d+)?\[", function)
    assert re.search(r"\bsrc(?:_\d+)?\[", function)
    assert ".GetValue(" not in function
    assert ".SetValue(" not in function
    assert "PTO_INLINE" not in cpp
    assert "asc_vf_call" not in cpp


def test_simt_context_codegen_uses_native_cce_xyz_context_and_tuple_launch():
    cpp = _compile_to_cce(_simt_context_codegen_kernel)
    launch_line = next(line for line in cpp.splitlines() if "cce::async_invoke<_context_probe>" in line)

    for context in ("threadIdx", "blockDim", "blockIdx", "gridDim"):
        for axis in ("x", "y", "z"):
            assert f"{context}.{axis}" in cpp
    assert "warpSize" in cpp
    assert "cce::async_invoke<_context_probe>(cce::dim3{8, 4, 8}" in launch_line


def test_simt_gm_tensor_codegen_uses_native_gm_pointers():
    cpp = _compile_to_cce(_simt_gm_codegen_kernel)
    function = _simt_function_source(cpp, "_gm_add")
    launch_line = next(line for line in cpp.splitlines() if "cce::async_invoke<_gm_add>" in line)

    assert re.search(r"__gm__\s+float\s*\*\s*dst", function)
    assert re.search(r"__gm__\s+float\s*\*\s*src", function)
    assert function.count("(__gm__ float*)") >= 2
    assert ".GetValue(" not in function
    assert ".SetValue(" not in function
    assert launch_line.count("(__gm__ float*)") == 2
    assert ".data()" not in launch_line


def test_simt_launch_passes_runtime_tile_valid_shape_to_native_function():
    cpp = _compile_to_cce(_simt_valid_shape_codegen_kernel)
    function = _simt_function_source(cpp, "_tile_valid_shape_access")
    launch_line = next(line for line in cpp.splitlines() if "cce::async_invoke<_tile_valid_shape_access>" in line)

    assert re.search(r"\bsrc(?:_\d+)*__valid_row\b", function)
    assert re.search(r"\bsrc(?:_\d+)*__valid_col\b", function)
    assert launch_line.count(".tile.GetValidRow()") == 2
    assert launch_line.count(".tile.GetValidCol()") == 2


def test_simt_warp_codegen_connects_each_operation_family_to_cce():
    cpp = _compile_to_cce(_simt_warp_codegen_kernel)
    function = _simt_function_source(cpp, "_warp_codegen_probe")

    for intrinsic in (
        "laneid()",
        "__ballot(",
        "__shfl_down(",
        "__reduce_add(",
    ):
        assert intrinsic in function
    assert "asc_" not in function


def test_atomic_add_gm_int64_codegen_uses_element_pointer():
    cpp = _compile_to_cce(_atomic_add_gm_codegen_kernel)
    function = _simt_function_source(cpp, "_atomic_add_gm")

    assert re.search(r"__gm__\s+int64_t\s*\*\s*dst(?:_\d+)*\b", function)
    assert "atomicAdd(" in function


def test_atomic_add_codegen_preserves_side_effect_when_return_value_is_discarded():
    cpp = _compile_to_cce(_atomic_add_discard_codegen_kernel)
    function = _simt_function_source(cpp, "_atomic_add_discard")

    assert "atomicAdd(" in function


def test_half_precision_atomic_codegen_emits_void_ub_and_gm_calls():
    cpp = _compile_to_cce(_atomic_half_discard_codegen_kernel)
    function = _simt_function_source(cpp, "_atomic_half_discard")

    assert re.search(r"__ubuf__\s+half\s*\*\s*ub(?:_\d+)*\b", function)
    assert re.search(r"__gm__\s+bfloat16_t\s*\*\s*gm(?:_\d+)*\b", function)
    for intrinsic in ("atomicAdd", "atomicMax", "atomicMin"):
        assert f"{intrinsic}(" in function


def test_simt_callee_codegen_emits_native_nested_calls_and_tile_abi():
    cpp = _compile_to_cce(_simt_callee_codegen_kernel)

    add_signature = next(line for line in cpp.splitlines() if "_callee_add(" in line and "__simt_callee__" in line)
    load_signature = next(line for line in cpp.splitlines() if "_callee_load(" in line and "__simt_callee__" in line)
    store_signature = next(line for line in cpp.splitlines() if "_callee_store(" in line and "__simt_callee__" in line)
    apply_signature = next(line for line in cpp.splitlines() if "_callee_apply(" in line and "__simt_callee__" in line)
    entry_signature = next(line for line in cpp.splitlines() if "inline void _callee_entry(" in line)

    assert "__simt_callee__ inline int32_t" in add_signature
    assert "__simt_callee__ inline int32_t" in load_signature
    assert "__simt_callee__ inline void" in store_signature
    for signature in (store_signature, apply_signature):
        assert re.search(r"__ubuf__\s+int32_t\s*\*\s*dst(?:_\d+)*\b", signature)
        assert re.search(r"\bdst(?:_\d+)*__valid_row\b", signature)
        assert re.search(r"\bdst(?:_\d+)*__valid_col\b", signature)
    for signature in (load_signature, apply_signature):
        assert re.search(r"__gm__\s+int32_t\s*\*\s*src(?:_\d+)*\b", signature)
        assert not re.search(r"\bsrc(?:_\d+)*__valid_row\b", signature)
        assert not re.search(r"\bsrc(?:_\d+)*__valid_col\b", signature)

    add_pos = cpp.index(add_signature)
    load_pos = cpp.index(load_signature)
    store_pos = cpp.index(store_signature)
    apply_pos = cpp.index(apply_signature)
    entry_pos = cpp.index(entry_signature)
    assert max(add_pos, load_pos, store_pos) < apply_pos < entry_pos
    apply_function = cpp[apply_pos:entry_pos]
    for callee in ("_callee_load(", "_callee_add(", "_callee_store("):
        assert callee in apply_function
    assert "_callee_apply(" in cpp[entry_pos:]


@pl.vector_function(mode="simt", max_threads=256)
def _simt_inplace_add(data, delta: pl.DT_FP32):
    tid = pl.simt.linear_thread_idx()
    data[0, tid] = data[0, tid] + delta


@pl.jit
def _simt_auto_mutex_kernel(
    x: pl.Tensor[[1, 256], pl.DT_FP32],
    out: pl.Tensor[[1, 256], pl.DT_FP32],
    delta: pl.DT_FP32,
):
    tile_type = pl.TileType(shape=[1, 256], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    data = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])
    with pl.section_vector():
        pl.load(data.current(), x, [0, 0])
        _simt_inplace_add[256](data.current(), delta)
        pl.store(out, data.current(), [0, 0])


def test_simt_auto_mutex_emits_pipe_v_lock_unlock_around_launch():
    cpp = _compile_to_cce(_simt_auto_mutex_kernel)

    launch_line = next(line for line in cpp.splitlines() if "cce::async_invoke<_simt_inplace_add>" in line)
    launch_pos = cpp.index(launch_line)

    lock_match = re.search(r"get_buf\(PIPE_V,\s*\w+,\s*0\);", cpp[:launch_pos])
    assert lock_match is not None, "Expected get_buf(PIPE_V, ...) before cce::async_invoke"

    unlock_match = re.search(r"rls_buf\(PIPE_V,\s*\w+,\s*0\);", cpp[launch_pos + len(launch_line):])
    assert unlock_match is not None, "Expected rls_buf(PIPE_V, ...) after cce::async_invoke"

    assert "pipe_barrier(PIPE_ALL);" not in cpp


def test_simt_codegen_rejects_pre_a5_architecture():
    with pytest.raises(NotSupported, match="requires arch='3510'"):
        _compile_to_cce(_simt_tile_codegen_kernel, arch=NpuArch.DAV_2201)


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


def test_scalar_cast_codegen_maps_round_modes_and_cce_builtins():
    cpp = _compile_to_cce(_scalar_cast_codegen_kernel)
    function = _simt_function_source(cpp, "_scalar_cast_intrinsics")

    assert "__cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(" in function
    assert "__cvt_bfloat16_t<ROUND::A, RoundingSaturation::RS_DISABLE_VALUE>(" in function
    assert "__cvt_int32_t<ROUND::F, RoundingSaturation::RS_ENABLE_VALUE>(" in function
    assert "__cvt_float<ROUND::C, RoundingSaturation::RS_DISABLE_VALUE>(" in function
    plain_cast = re.search(r"auto (\w+) = \(\(int32_t\)integer(?:_\d+)?\);", function)
    assert plain_cast is not None
    assert re.search(rf"plain_out(?:_\d+)?\[[^]]+\]\s*=\s*{plain_cast[1]};", function)
    assert "simt_api/" not in cpp
    assert "__float2half" not in cpp
    assert "__float2bfloat16" not in cpp
    assert "__float2int" not in cpp


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
    function = _simt_function_source(cpp, "_fp32_math_intrinsics")

    for intrinsic in ("__fminf(", "__sqrtf(", "__isnan(", "__fma("):
        assert intrinsic in function
    assert "simt_api/math_functions.h" not in cpp


def test_fp16_scalar_math_codegen_maps_cce_intrinsics_without_simt_headers():
    cpp = _compile_to_cce(_fp16_math_codegen_kernel)
    function = _simt_function_source(cpp, "_fp16_math_intrinsics")

    for intrinsic in ("__hmin_nan(", "__cvt_float<", "__cvt_half<"):
        assert intrinsic in function
    assert "simt_api/" not in cpp


def test_bf16_scalar_math_codegen_maps_cce_intrinsics_without_simt_headers():
    cpp = _compile_to_cce(_bf16_math_codegen_kernel)
    function = _simt_function_source(cpp, "_bf16_math_intrinsics")

    for intrinsic in ("__min(", "__cvt_float<", "__cvt_bfloat16_t<"):
        assert intrinsic in function
    assert "simt_api/" not in cpp


def test_int64_scalar_math_codegen_maps_abs_min_max_and_header():
    cpp = _compile_to_cce(_int64_math_codegen_kernel)
    function = _simt_function_source(cpp, "_int64_math_intrinsics")

    assert "= abs(" in function
    assert "min((int64_t)" in function
    assert "max((int64_t)" in function
    assert "simt_api/math_functions.h" not in cpp


@pl.vector_function(mode="simt", max_threads=1)
def _fp32_transcendental_intrinsics(out, value: pl.DT_FP32):
    out[0, 0] = pl.simt.exp10(value)
    out[0, 1] = pl.simt.log10(value)
    out[0, 2] = pl.simt.tan(value)
    out[0, 3] = pl.simt.atan(value)
    out[0, 4] = pl.simt.expm1(value)
    out[0, 5] = pl.simt.logb(value)
    out[0, 6] = pl.simt.cosh(value)
    out[0, 7] = pl.simt.acos(value)
    out[0, 8] = pl.simt.sinh(value)
    out[0, 9] = pl.simt.asin(value)
    out[0, 10] = pl.simt.cbrt(value)


@pl.jit
def _fp32_transcendental_codegen_kernel(value: pl.DT_FP32):
    out_type = pl.TileType(shape=[1, 16], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    out = pl.make_tile(out_type, addr=0x0000)
    with pl.section_vector():
        _fp32_transcendental_intrinsics[1](out, value)


@pl.vector_function(mode="simt", max_threads=1)
def _fp16_exp_log_reciprocal_intrinsics(out, source):
    value = source[0, 0]
    out[0, 0] = pl.simt.exp10(value)
    out[0, 1] = pl.simt.log10(value)
    out[0, 2] = pl.simt.rcp(value)


@pl.jit
def _fp16_exp_log_reciprocal_codegen_kernel(_jit_entry: pl.DT_INT64):
    tile_type = pl.TileType(shape=[1, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    source_type = pl.TileType(shape=[1, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    out = pl.make_tile(tile_type, addr=0x0000)
    source = pl.make_tile(source_type, addr=0x0040)
    with pl.section_vector():
        _fp16_exp_log_reciprocal_intrinsics[1](out, source)


@pl.vector_function(mode="simt", max_threads=1)
def _bf16_exp_log_reciprocal_intrinsics(out, source):
    value = source[0, 0]
    out[0, 0] = pl.simt.exp10(value)
    out[0, 1] = pl.simt.log10(value)
    out[0, 2] = pl.simt.rcp(value)


@pl.jit
def _bf16_exp_log_reciprocal_codegen_kernel(_jit_entry: pl.DT_INT64):
    tile_type = pl.TileType(shape=[1, 16], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec)
    source_type = pl.TileType(shape=[1, 16], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec)
    out = pl.make_tile(tile_type, addr=0x0000)
    source = pl.make_tile(source_type, addr=0x0040)
    with pl.section_vector():
        _bf16_exp_log_reciprocal_intrinsics[1](out, source)


@pl.vector_function(mode="simt", max_threads=1)
def _nan_intrinsics(out_fp16, source_fp16, out_bf16, source_bf16):
    out_fp16[0, 0] = pl.simt.max_nan(source_fp16[0, 0], source_fp16[0, 1])
    out_fp16[0, 1] = pl.simt.min_nan(source_fp16[0, 0], source_fp16[0, 1])
    out_bf16[0, 0] = pl.simt.max_nan(source_bf16[0, 0], source_bf16[0, 1])
    out_bf16[0, 1] = pl.simt.min_nan(source_bf16[0, 0], source_bf16[0, 1])


@pl.jit
def _nan_intrinsics_codegen_kernel(_jit_entry: pl.DT_INT64):
    fp16_type = pl.TileType(shape=[1, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    bf16_type = pl.TileType(shape=[1, 16], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec)
    out_fp16 = pl.make_tile(fp16_type, addr=0x0000)
    source_fp16 = pl.make_tile(fp16_type, addr=0x0040)
    out_bf16 = pl.make_tile(bf16_type, addr=0x0080)
    source_bf16 = pl.make_tile(bf16_type, addr=0x00C0)
    with pl.section_vector():
        _nan_intrinsics[1](out_fp16, source_fp16, out_bf16, source_bf16)


@pl.vector_function(mode="simt", max_threads=1)
def _pi_hyperbolic_math_intrinsics(out_fp32, value: pl.DT_FP32):
    out_fp32[0, 0] = pl.simt.tanpi(value)
    out_fp32[0, 1] = pl.simt.atanh(value)
    out_fp32[0, 2] = pl.simt.cospi(value)
    out_fp32[0, 3] = pl.simt.acosh(value)
    out_fp32[0, 4] = pl.simt.sinpi(value)
    out_fp32[0, 5] = pl.simt.asinh(value)
    out_fp32[0, 6] = pl.simt.rcbrt(value)


@pl.jit
def _pi_hyperbolic_math_codegen_kernel(value: pl.DT_FP32):
    out_fp32 = pl.make_tile(
        pl.TileType(shape=[1, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x0000,
    )
    with pl.section_vector():
        _pi_hyperbolic_math_intrinsics[1](out_fp32, value)


def test_fp32_transcendental_math_codegen_is_self_contained():
    cpp = _compile_to_cce(_fp32_transcendental_codegen_kernel)
    function = _simt_function_source(cpp, "_fp32_transcendental_intrinsics")

    assert "pypto_simt_" not in cpp
    assert "simt_api/" not in cpp
    assert "__logf(" in function


def test_fp16_exp_log_reciprocal_codegen_is_self_contained():
    cpp = _compile_to_cce(_fp16_exp_log_reciprocal_codegen_kernel)

    assert "pypto_simt_" not in cpp
    assert "simt_api/" not in cpp


def test_bf16_exp_log_reciprocal_codegen_is_self_contained():
    cpp = _compile_to_cce(_bf16_exp_log_reciprocal_codegen_kernel)
    function = _simt_function_source(cpp, "_bf16_exp_log_reciprocal_intrinsics")

    assert "pypto_simt_" not in cpp
    assert "simt_api/" not in cpp
    assert "__cvt_float<" in function
    assert "__cvt_bfloat16_t<" in function


def test_nan_select_codegen_uses_low_precision_intrinsics():
    cpp = _compile_to_cce(_nan_intrinsics_codegen_kernel)

    assert "__hmax_nan(" in cpp
    assert "__hmin_nan(" in cpp
    assert "__cvt_float<" not in cpp
    assert "simt_api/" not in cpp


def test_pi_hyperbolic_math_codegen_is_self_contained():
    cpp = _compile_to_cce(_pi_hyperbolic_math_codegen_kernel)
    function = _simt_function_source(cpp, "_pi_hyperbolic_math_intrinsics")

    assert "pypto_simt_" not in cpp
    assert "simt_api/" not in cpp
    for wrapper in ("tanpif(", "atanhf(", "cospif(", "acoshf(", "sinpif(", "asinhf(", "rcbrtf("):
        assert wrapper not in function


@pl.vector_function(mode="simt", max_threads=1)
def _bessel_rounding_and_saturate_intrinsics(out_fp32, out_i64, value: pl.DT_FP32, order: pl.DT_INT32):
    out_fp32[0, 0] = pl.simt.jn(order, value)
    out_fp32[0, 1] = pl.simt.yn(order, value)
    out_fp32[0, 2] = pl.simt.cyl_bessel_i0(value)
    out_fp32[0, 3] = pl.simt.cyl_bessel_i1(value)
    out_fp32[0, 4] = pl.simt.j0(value)
    out_fp32[0, 5] = pl.simt.j1(value)
    out_fp32[0, 6] = pl.simt.y0(value)
    out_fp32[0, 7] = pl.simt.y1(value)
    out_fp32[0, 8] = pl.simt.saturate(value)
    out_i64[0, 0] = pl.simt.rint(value, dtype=pl.DT_INT64)
    out_i64[0, 1] = pl.simt.round(value, dtype=pl.DT_INT64)


@pl.jit
def _bessel_rounding_and_saturate_codegen_kernel(value: pl.DT_FP32, order: pl.DT_INT32):
    out_fp32 = pl.make_tile(
        pl.TileType(shape=[1, 16], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x0000,
    )
    out_i64 = pl.make_tile(
        pl.TileType(shape=[1, 4], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec),
        addr=0x0040,
    )
    with pl.section_vector():
        _bessel_rounding_and_saturate_intrinsics[1](out_fp32, out_i64, value, order)


def test_bessel_rounding_and_saturate_codegen_is_self_contained():
    cpp = _compile_to_cce(_bessel_rounding_and_saturate_codegen_kernel)
    function = _simt_function_source(cpp, "_bessel_rounding_and_saturate_intrinsics")

    assert "simt_api/" not in cpp
    assert "__cvt_int64_t<ROUND::R, RoundingSaturation::RS_ENABLE_VALUE>(" in function
    assert "__cvt_int64_t<ROUND::A, RoundingSaturation::RS_ENABLE_VALUE>(" in function
    assert "__saturatef(" in function
    for wrapper in ("j0f(", "j1f(", "y0f(", "y1f(", "jnf(", "ynf("):
        assert wrapper not in function


@pl.vector_function(mode="simt", max_threads=1)
def _integer_bit_arithmetic_intrinsics(
    signed_out,
    counts,
    unsigned_out,
    unsigned_wide,
    signed_source,
    unsigned_source,
    signed64_source,
    unsigned64_source,
):
    signed_lhs = signed_source[0, 0]
    signed_rhs = signed_source[0, 1]
    unsigned_lhs = unsigned_source[0, 0]
    unsigned_rhs = unsigned_source[0, 1]
    selector = unsigned_source[0, 2]
    unsigned_out[0, 0] = pl.simt.brev(unsigned_lhs)
    unsigned_wide[0, 0] = pl.simt.brev(unsigned64_source[0, 0])
    counts[0, 0] = pl.simt.clz(signed_lhs)
    counts[0, 1] = pl.simt.clz(signed64_source[0, 0])
    counts[0, 2] = pl.simt.ffs(signed_lhs)
    counts[0, 3] = pl.simt.ffs(signed64_source[0, 0])
    unsigned_out[0, 1] = pl.simt.byte_perm(unsigned_lhs, unsigned_rhs, selector)
    unsigned_out[0, 2] = pl.simt.sad(signed_lhs, signed_rhs, selector)
    unsigned_out[0, 3] = pl.simt.sad(unsigned_lhs, unsigned_rhs, selector)
    signed_out[0, 0] = pl.simt.mul24(signed_lhs, signed_rhs)
    unsigned_out[0, 4] = pl.simt.mul24(unsigned_lhs, unsigned_rhs)
    signed_out[0, 1] = pl.simt.hadd(signed_lhs, signed_rhs)
    unsigned_out[0, 5] = pl.simt.hadd(unsigned_lhs, unsigned_rhs)
    signed_out[0, 2] = pl.simt.rhadd(signed_lhs, signed_rhs)
    unsigned_out[0, 6] = pl.simt.rhadd(unsigned_lhs, unsigned_rhs)


@pl.jit
def _integer_bit_arithmetic_codegen_kernel(_jit_entry: pl.DT_INT64):
    signed_out = pl.make_tile(
        pl.TileType(shape=[1, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addr=0x0000,
    )
    counts = pl.make_tile(
        pl.TileType(shape=[1, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addr=0x0080,
    )
    unsigned_out = pl.make_tile(
        pl.TileType(shape=[1, 8], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addr=0x00C0,
    )
    unsigned_wide = pl.make_tile(
        pl.TileType(shape=[1, 4], dtype=pl.DT_UINT64, target_memory=pl.MemorySpace.Vec),
        addr=0x0100,
    )
    signed_source = pl.make_tile(
        pl.TileType(shape=[1, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addr=0x0140,
    )
    unsigned_source = pl.make_tile(
        pl.TileType(shape=[1, 8], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addr=0x0180,
    )
    signed64_source = pl.make_tile(
        pl.TileType(shape=[1, 4], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec),
        addr=0x01C0,
    )
    unsigned64_source = pl.make_tile(
        pl.TileType(shape=[1, 4], dtype=pl.DT_UINT64, target_memory=pl.MemorySpace.Vec),
        addr=0x0200,
    )
    with pl.section_vector():
        _integer_bit_arithmetic_intrinsics[1](
            signed_out,
            counts,
            unsigned_out,
            unsigned_wide,
            signed_source,
            unsigned_source,
            signed64_source,
            unsigned64_source,
        )


@pl.vector_function(mode="simt", max_threads=1)
def _low_precision_fma_comparison_intrinsics(out, flags, source):
    lhs = source[0, 0]
    rhs = source[0, 1]
    out[0, 0] = pl.simt.fma_relu(lhs, rhs, lhs)
    flags[0, 0] = pl.simt.equ(lhs, rhs)
    flags[0, 1] = pl.simt.leu(lhs, rhs)
    flags[0, 2] = pl.simt.geu(lhs, rhs)
    flags[0, 3] = pl.simt.ltu(lhs, rhs)
    flags[0, 4] = pl.simt.gtu(lhs, rhs)


@pl.jit
def _fp16_fma_comparison_codegen_kernel(_jit_entry: pl.DT_INT64):
    out = pl.make_tile(
        pl.TileType(shape=[1, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addr=0x0000,
    )
    flags = pl.make_tile(
        pl.TileType(shape=[1, 32], dtype=pl.DT_BOOL, target_memory=pl.MemorySpace.Vec),
        addr=0x0040,
    )
    source = pl.make_tile(
        pl.TileType(shape=[1, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addr=0x0080,
    )
    with pl.section_vector():
        _low_precision_fma_comparison_intrinsics[1](out, flags, source)


@pl.jit
def _bf16_fma_comparison_codegen_kernel(_jit_entry: pl.DT_INT64):
    out = pl.make_tile(
        pl.TileType(shape=[1, 16], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addr=0x0000,
    )
    flags = pl.make_tile(
        pl.TileType(shape=[1, 32], dtype=pl.DT_BOOL, target_memory=pl.MemorySpace.Vec),
        addr=0x0040,
    )
    source = pl.make_tile(
        pl.TileType(shape=[1, 16], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addr=0x0080,
    )
    with pl.section_vector():
        _low_precision_fma_comparison_intrinsics[1](out, flags, source)


@pl.vector_function(mode="simt", max_threads=1)
def _fp32_binary_math_intrinsics(out, lhs: pl.DT_FP32, rhs: pl.DT_FP32):
    out[0, 0] = pl.simt.hypot(lhs, rhs)
    out[0, 1] = pl.simt.rhypot(lhs, rhs)
    out[0, 2] = pl.simt.fdim(lhs, rhs)
    out[0, 3] = pl.simt.remainder(lhs, rhs)
    out[0, 4] = pl.simt.pow(lhs, rhs)


@pl.jit
def _fp32_binary_math_codegen_kernel(lhs: pl.DT_FP32, rhs: pl.DT_FP32):
    out = pl.make_tile(
        pl.TileType(shape=[1, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x0000,
    )
    with pl.section_vector():
        _fp32_binary_math_intrinsics[1](out, lhs, rhs)


@pl.vector_function(mode="simt", max_threads=1)
def _recognized_special_math_intrinsics(
    out_fp32,
    value: pl.DT_FP32,
    order: pl.DT_INT32,
    long_exponent: pl.DT_INT64,
):
    out_fp32[0, 0] = pl.simt.ldexp(value, order)
    out_fp32[0, 1] = pl.simt.scalbn(value, order)
    out_fp32[0, 2] = pl.simt.scalbn(value, long_exponent)
    out_fp32[0, 3] = pl.simt.norm3d(value, value, value)
    out_fp32[0, 4] = pl.simt.rnorm3d(value, value, value)
    out_fp32[0, 5] = pl.simt.norm4d(value, value, value, value)
    out_fp32[0, 6] = pl.simt.rnorm4d(value, value, value, value)
    out_fp32[0, 7] = pl.simt.erf(value)
    out_fp32[0, 8] = pl.simt.erfc(value)
    out_fp32[0, 9] = pl.simt.erfinv(value)
    out_fp32[0, 10] = pl.simt.erfcinv(value)
    out_fp32[0, 11] = pl.simt.erfcx(value)
    out_fp32[0, 12] = pl.simt.tgamma(value)
    out_fp32[0, 13] = pl.simt.lgamma(value)
    out_fp32[0, 14] = pl.simt.normcdf(value)
    out_fp32[0, 15] = pl.simt.normcdfinv(value)


@pl.jit
def _recognized_special_math_codegen_kernel(
    value: pl.DT_FP32,
    order: pl.DT_INT32,
    long_exponent: pl.DT_INT64,
):
    out_fp32 = pl.make_tile(
        pl.TileType(shape=[1, 16], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x0000,
    )
    with pl.section_vector():
        _recognized_special_math_intrinsics[1](out_fp32, value, order, long_exponent)


def test_recognized_special_math_codegen_is_self_contained():
    cpp = _compile_to_cce(_recognized_special_math_codegen_kernel)
    function = _simt_function_source(cpp, "_recognized_special_math_intrinsics")

    assert "simt_api/" not in cpp
    assert "__powf(" in function
    for wrapper in (
        "erff(",
        "erfcf(",
        "erfinvf(",
        "erfcinvf(",
        "erfcxf(",
        "tgammaf(",
        "lgammaf(",
        "normcdff(",
        "normcdfinvf(",
    ):
        assert wrapper not in function


@pl.vector_function(mode="simt", max_threads=1)
def _mul_i32_to_i64_intrinsics(signed_out, unsigned_out, signed_source, unsigned_source):
    signed_out[0, 0] = pl.simt.mul_i32_to_i64(signed_source[0, 0], signed_source[0, 1])
    unsigned_out[0, 0] = pl.simt.mul_i32_to_i64(unsigned_source[0, 0], unsigned_source[0, 1])


@pl.jit
def _mul_i32_to_i64_codegen_kernel(_jit_entry: pl.DT_INT64):
    signed_out = pl.make_tile(
        pl.TileType(shape=[1, 4], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec),
        addr=0x0000,
    )
    unsigned_out = pl.make_tile(
        pl.TileType(shape=[1, 4], dtype=pl.DT_UINT64, target_memory=pl.MemorySpace.Vec),
        addr=0x0040,
    )
    signed_source = pl.make_tile(
        pl.TileType(shape=[1, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addr=0x0080,
    )
    unsigned_source = pl.make_tile(
        pl.TileType(shape=[1, 8], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addr=0x00C0,
    )
    with pl.section_vector():
        _mul_i32_to_i64_intrinsics[1](signed_out, unsigned_out, signed_source, unsigned_source)


def test_mul_i32_to_i64_codegen_uses_cce_intrinsic():
    cpp = _compile_to_cce(_mul_i32_to_i64_codegen_kernel)

    assert "__mul_i32toi64((int)(" in cpp
    assert "__mul_i32toi64((unsigned int)(" in cpp


def test_integer_bit_arithmetic_codegen_uses_cce_intrinsics():
    cpp = _compile_to_cce(_integer_bit_arithmetic_codegen_kernel)

    assert "simt_api/" not in cpp
    for intrinsic in (
        "__brev",
        "__clz",
        "__ffs",
        "__byte_perm",
        "__sad",
        "__usad",
        "__mul24",
        "__umul24",
        "__hadd",
        "__uhadd",
        "__rhadd",
        "__urhadd",
    ):
        assert f"{intrinsic}(" in cpp


def test_fp16_fma_comparison_codegen_uses_cce_intrinsics():
    cpp = _compile_to_cce(_fp16_fma_comparison_codegen_kernel)

    assert "simt_api/" not in cpp
    for intrinsic in ("__hfma_relu", "__hequ", "__hleu", "__hgeu", "__hltu", "__hgtu"):
        assert f"{intrinsic}(" in cpp


def test_bf16_fma_comparison_codegen_uses_cce_intrinsics():
    cpp = _compile_to_cce(_bf16_fma_comparison_codegen_kernel)

    assert "simt_api/" not in cpp
    for intrinsic in ("__hfma_relu", "__hequ", "__hleu", "__hgeu", "__hltu", "__hgtu"):
        assert f"{intrinsic}(" in cpp


def test_fp32_binary_math_codegen_is_self_contained():
    cpp = _compile_to_cce(_fp32_binary_math_codegen_kernel)
    function = _simt_function_source(cpp, "_fp32_binary_math_intrinsics")

    assert "simt_api/" not in cpp
    assert "__sqrtf(" in function
    assert "__fma(" in function
    for wrapper in ("hypotf", "rhypotf", "fdimf", "remainderf", "powf"):
        assert f"{wrapper}(" not in function
