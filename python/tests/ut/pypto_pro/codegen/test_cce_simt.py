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

"""CCE code-generation tests for the Tile-first A5 SIMT path."""

import re

import pypto_pro.language as pl
import pytest


def _compile_to_cce(kernel, arch: str = "a5") -> str:
    from pypto_pro.runtime.jit import _assemble_cv_source, _parse_and_codegen_targets

    cube, vector = _parse_and_codegen_targets(kernel.to_kernel_def(), arch, "")
    return _assemble_cv_source(cube, vector).content


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


@pl.vector_function(mode="simt", max_threads=1)
def _atomic_rmw_discard(
    numeric,
    bitwise,
    counter,
    compare: pl.DT_INT32,
    replacement: pl.DT_INT32,
    mask: pl.DT_UINT32,
    limit: pl.DT_UINT32,
):
    pl.simt.atomic_sub(numeric[0, 0], replacement)
    pl.simt.atomic_exch(numeric[0, 0], replacement)
    pl.simt.atomic_max(numeric[0, 0], replacement)
    pl.simt.atomic_min(numeric[0, 0], replacement)
    pl.simt.atomic_cas(numeric[0, 0], compare, replacement)
    pl.simt.atomic_and(bitwise[0, 0], mask)
    pl.simt.atomic_or(bitwise[0, 0], mask)
    pl.simt.atomic_xor(bitwise[0, 0], mask)
    pl.simt.atomic_inc(counter[0, 0], limit)
    pl.simt.atomic_dec(counter[0, 0], limit)


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


@pl.jit
def _atomic_rmw_discard_codegen_kernel(
    compare: pl.DT_INT32,
    replacement: pl.DT_INT32,
    mask: pl.DT_UINT32,
    limit: pl.DT_UINT32,
):
    numeric_type = pl.TileType(shape=[1, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    unsigned_type = pl.TileType(shape=[1, 8], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    numeric = pl.make_tile(numeric_type, addr=0x0000)
    bitwise = pl.make_tile(unsigned_type, addr=0x0040)
    counter = pl.make_tile(unsigned_type, addr=0x0080)
    with pl.section_vector():
        _atomic_rmw_discard[1](numeric, bitwise, counter, compare, replacement, mask, limit)


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
    assert launch_line.count(".data()") == 2
    assert launch_line.count(".GetValidRow()") == 2
    assert launch_line.count(".GetValidCol()") == 2
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
    assert "threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y" in cpp
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
    assert launch_line.count(".GetValidRow()") == 2
    assert launch_line.count(".GetValidCol()") == 2


def test_atomic_add_gm_int64_codegen_uses_element_pointer_and_returns_old_value():
    cpp = _compile_to_cce(_atomic_add_gm_codegen_kernel)
    function = _simt_function_source(cpp, "_atomic_add_gm")
    atomic_line = next(line for line in function.splitlines() if "atomicAdd(" in line)
    store_line = next(
        line for line in function.splitlines() if "old_values" in line and "__simt_atomic_result_" in line
    )
    result_match = re.search(r"^\s*auto\s+(__simt_atomic_result_\d+)\s*=", atomic_line)

    assert re.search(r"__gm__\s+int64_t\s*\*\s*dst(?:_\d+)*\b", function)
    assert result_match is not None
    assert re.search(r"atomicAdd\(dst(?:_\d+)?\s*\+\s*\(", atomic_line)
    assert re.search(r",\s*value(?:_\d+)?\);$", atomic_line.strip())
    assert re.search(r"\*\(\(__gm__ int64_t\*\)old_values(?:_\d+)?\s*\+", store_line)
    assert result_match.group(1) in store_line


def test_atomic_add_codegen_preserves_side_effect_when_return_value_is_discarded():
    cpp = _compile_to_cce(_atomic_add_discard_codegen_kernel)
    function = _simt_function_source(cpp, "_atomic_add_discard")
    atomic_line = next(line for line in function.splitlines() if "atomicAdd(" in line)

    assert re.search(
        r"^\s*auto\s+__simt_atomic_result_\d+\s*=\s*atomicAdd\(dst(?:_\d+)?\s*\+\s*\(", atomic_line)
    assert atomic_line.strip().endswith(";")


def test_half_precision_atomic_codegen_emits_void_ub_and_gm_calls():
    cpp = _compile_to_cce(_atomic_half_discard_codegen_kernel)
    function = _simt_function_source(cpp, "_atomic_half_discard")

    assert re.search(r"__ubuf__\s+half\s*\*\s*ub(?:_\d+)*\b", function)
    assert re.search(r"__gm__\s+bfloat16_t\s*\*\s*gm(?:_\d+)*\b", function)
    assert not re.search(r"\bauto\s+ignored(?:_\d+)*\s*=", function)
    for intrinsic in ("atomicAdd", "atomicMax", "atomicMin"):
        matching_lines = [line.strip() for line in function.splitlines() if f"{intrinsic}(" in line]
        assert len(matching_lines) == 2
        assert all(line.startswith(intrinsic) and line.endswith(";") for line in matching_lines)


def test_atomic_rmw_codegen_maps_all_intrinsics_and_preserves_discarded_calls():
    cpp = _compile_to_cce(_atomic_rmw_discard_codegen_kernel)
    function = _simt_function_source(cpp, "_atomic_rmw_discard")
    assert ".data()" not in function

    for intrinsic in (
        "atomicSub",
        "atomicExch",
        "atomicMax",
        "atomicMin",
        "atomicCAS",
        "atomicAnd",
        "atomicOr",
        "atomicXOr",
        "atomicInc",
        "atomicDec",
    ):
        matching_lines = [line.strip() for line in function.splitlines() if f"{intrinsic}(" in line]
        assert len(matching_lines) == 1
        assert matching_lines[0].endswith(";")

    cas_line = next(line for line in function.splitlines() if "atomicCAS(" in line)
    assert re.search(r",\s*compare(?:_\d+)?\s*,\s*replacement(?:_\d+)?\s*\);$", cas_line.strip())


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
    add_function = cpp[add_pos:cpp.index("\n}\n", add_pos)]
    apply_function = cpp[apply_pos:entry_pos]
    assert re.search(r"\bvalue(?:_\d+)*\s*\+\s*delta(?:_\d+)*", add_function)
    assert re.search(r"\breturn\s+[^;]+;", add_function)
    assert re.search(r"_callee_load\(src(?:_\d+)?,\s*index(?:_\d+)?\)", apply_function)
    assert re.search(r"_callee_add\([^;]+\)", apply_function)
    assert re.search(
        r"_callee_store\([^;]+dst(?:_\d+)?__valid_row,\s*dst(?:_\d+)?__valid_col[^;]+\);",
        apply_function,
    )
    assert re.search(
        r"_callee_apply\([^;]+dst(?:_\d+)?__valid_row,\s*dst(?:_\d+)?__valid_col"
        r"[^;]+src(?:_\d+)?[^;]+\);",
        cpp[entry_pos:],
    )


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
    with pytest.raises(RuntimeError, match="requires arch='a5'"):
        _compile_to_cce(_simt_tile_codegen_kernel, arch="a3")
