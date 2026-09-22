# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 end-to-end tests for the SIMT atomic_add interface."""

import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
THREADS = 64
ELEMENTS = 64
SEMANTIC_THREADS = 32
ATOMIC_BUCKETS = 8
ATOMIC_GRID_BLOCKS = 4


def _require_a5():
    try:
        torch.npu.set_device(ST_DEVICE)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


def _run_kernel(kernel, states):
    _require_a5()
    device_states = [state.to(ST_DEVICE) for state in states]
    kernel[None, 1](*device_states)
    torch.npu.synchronize()
    return [state.cpu() for state in device_states]


def _assert_target(state, expected):
    expected = torch.tensor(expected, dtype=state.dtype)
    torch.testing.assert_close(state[0, 0], expected, rtol=0, atol=0)


@pl.vector_function(mode="simt", max_threads=THREADS)
def atomic_add_loaded_gm_value(
    target: pl.Tensor[[1, THREADS], pl.DT_INT32],
    value_tensor: pl.Tensor[[1, 1], pl.DT_INT32],
):
    pl.simt.atomic_add(target[0, 0], value_tensor[0, 0])


@pl.jit(auto_mutex=True, arch="a5")
def simt_atomic_add_loaded_gm_value(
    target: pl.Tensor[[1, THREADS], pl.DT_INT32],
    value_tensor: pl.Tensor[[1, 1], pl.DT_INT32],
):
    with pl.section_vector():
        atomic_add_loaded_gm_value[THREADS](target, value_tensor)


@pytest.mark.soc("950")
def test_atomic_add_accepts_loaded_gm_value():
    initial = torch.full((1, THREADS), 7, dtype=torch.int32)
    value = torch.tensor([[-3]], dtype=torch.int32)

    result, loaded_value = _run_kernel(simt_atomic_add_loaded_gm_value, (initial, value))

    expected = initial.clone()
    expected[0, 0] = 7 + THREADS * value[0, 0]
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(loaded_value, value, rtol=0, atol=0)


@pytest.mark.soc("950")
@pytest.mark.parametrize(
    "dtype,torch_dtype,value",
    [(pl.DT_INT32, torch.int32, -3), (pl.DT_UINT32, torch.uint32, 3), (pl.DT_FP32, torch.float32, 0.25)],
)
@pytest.mark.parametrize("use_tile", [False, True], ids=["tensor", "tile"])
def test_atomic_add_from_loaded_elements(dtype, torch_dtype, value, use_tile):
    @pl.vector_function(mode="simt", max_threads=THREADS)
    def add_loaded(dst, values):
        pl.simt.atomic_add(dst[0, 0], values[0, 0])
        scalar = values[0, 1]
        pl.simt.atomic_add(dst[0, 1], scalar)

    @pl.jit(auto_mutex=True)
    def kernel(dst: pl.Tensor[[1, ELEMENTS], dtype], values: pl.Tensor[[1, ELEMENTS], dtype]):
        with pl.section_vector():
            if use_tile:
                tile_type = pl.TileType(shape=[1, ELEMENTS], dtype=dtype, target_memory=pl.MemorySpace.Vec)
                dst_group = pl.make_tile_group(type=tile_type, addrs=0, mutex_ids=[0])
                values_group = pl.make_tile_group(type=tile_type, addrs=0x0400, mutex_ids=[1])
                dst_tile = dst_group.current()
                values_tile = values_group.current()
                pl.load(dst_tile, dst, [0, 0])
                pl.load(values_tile, values, [0, 0])
                add_loaded[THREADS](dst_tile, values_tile)
                pl.store(dst, dst_tile, [0, 0])
            else:
                add_loaded[THREADS](dst, values)

    initial = torch.full((1, ELEMENTS), 7, dtype=torch_dtype)
    values = torch.full((1, ELEMENTS), value, dtype=torch_dtype)
    result, loaded_values = _run_kernel(kernel, (initial, values))
    expected = initial.clone()
    expected[0, 0] = 7 + THREADS * value
    expected[0, 1] = 7 + THREADS * value
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(loaded_values, values, rtol=0, atol=0)


@pl.vector_function(mode="simt", max_threads=THREADS)
def atomic_add_ub_all_dtypes(
    int32_tile,
    uint32_tile,
    fp16_tile,
    bf16_tile,
    fp32_tile,
):
    pl.simt.atomic_add(int32_tile[0, 0], 1)
    pl.simt.atomic_add(uint32_tile[0, 0], 1)
    pl.simt.atomic_add(fp16_tile[0, 0], 1.0)
    pl.simt.atomic_add(bf16_tile[0, 0], 1.0)
    pl.simt.atomic_add(fp32_tile[0, 0], 1.0)


@pl.jit()
def simt_atomic_add_ub_all_dtypes(
    int32_state: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    uint32_state: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    fp16_state: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    bf16_state: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    fp32_state: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    int32_tile = pl.make_tile(
        pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addr=0x0000,
    )
    uint32_tile = pl.make_tile(
        pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addr=0x0400,
    )
    fp16_tile = pl.make_tile(
        pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addr=0x0800,
    )
    bf16_tile = pl.make_tile(
        pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addr=0x0C00,
    )
    fp32_tile = pl.make_tile(
        pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x1000,
    )
    with pl.section_vector():
        pl.load(int32_tile, int32_state, [0, 0])
        pl.load(uint32_tile, uint32_state, [0, 0])
        pl.load(fp16_tile, fp16_state, [0, 0])
        pl.load(bf16_tile, bf16_state, [0, 0])
        pl.load(fp32_tile, fp32_state, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        atomic_add_ub_all_dtypes[THREADS](int32_tile, uint32_tile, fp16_tile, bf16_tile, fp32_tile)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(int32_state, int32_tile, [0, 0])
        pl.store(uint32_state, uint32_tile, [0, 0])
        pl.store(fp16_state, fp16_tile, [0, 0])
        pl.store(bf16_state, bf16_tile, [0, 0])
        pl.store(fp32_state, fp32_tile, [0, 0])


@pl.vector_function(mode="simt", max_threads=THREADS)
def atomic_add_gm_all_dtypes(
    int32_state: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    uint32_state: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    fp16_state: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    bf16_state: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    fp32_state: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    int64_state: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    uint64_state: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
):
    pl.simt.atomic_add(int32_state[0, 0], 1)
    pl.simt.atomic_add(uint32_state[0, 0], 1)
    pl.simt.atomic_add(fp16_state[0, 0], 1.0)
    pl.simt.atomic_add(bf16_state[0, 0], 1.0)
    pl.simt.atomic_add(fp32_state[0, 0], 1.0)
    pl.simt.atomic_add(int64_state[0, 0], 1)
    pl.simt.atomic_add(uint64_state[0, 0], 1)


@pl.jit()
def simt_atomic_add_gm_all_dtypes(
    int32_state: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    uint32_state: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    fp16_state: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    bf16_state: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    fp32_state: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    int64_state: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    uint64_state: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
):
    with pl.section_vector():
        atomic_add_gm_all_dtypes[THREADS](
            int32_state, uint32_state, fp16_state, bf16_state, fp32_state, int64_state, uint64_state
        )


@pl.vector_function(mode="simt", max_threads=1)
def atomic_add_return_value_gm(
    state: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    old_values: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
):
    old_values[0, 0] = pl.simt.atomic_add(state[0, 0], 5)


@pl.jit()
def simt_atomic_add_return_value_gm(
    state: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    old_values: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
):
    with pl.section_vector():
        atomic_add_return_value_gm[1](state, old_values)


@pl.vector_function(mode="simt", max_threads=SEMANTIC_THREADS)
def atomic_add_histogram_gm(histogram: pl.Tensor[[1, ATOMIC_BUCKETS], pl.DT_INT32]):
    tid = pl.simt.linear_thread_idx()
    pl.simt.atomic_add(histogram[0, tid % ATOMIC_BUCKETS], 1)


@pl.vector_function(mode="simt", max_threads=SEMANTIC_THREADS)
def atomic_add_histogram_ub(histogram):
    tid = pl.simt.linear_thread_idx()
    pl.simt.atomic_add(histogram[0, tid % ATOMIC_BUCKETS], 1)


@pl.jit()
def simt_atomic_add_histogram_gm(histogram: pl.Tensor[[1, ATOMIC_BUCKETS], pl.DT_INT32]):
    with pl.section_vector():
        atomic_add_histogram_gm[SEMANTIC_THREADS](histogram)


@pl.jit()
def simt_atomic_add_histogram_ub(histogram_tensor: pl.Tensor[[1, ATOMIC_BUCKETS], pl.DT_INT32]):
    tile_type = pl.TileType(shape=[1, ATOMIC_BUCKETS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    histogram = pl.make_tile(tile_type, addr=0x0000)
    with pl.section_vector():
        pl.load(histogram, histogram_tensor, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        atomic_add_histogram_ub[SEMANTIC_THREADS](histogram)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(histogram_tensor, histogram, [0, 0])


@pl.vector_function(mode="simt", max_threads=SEMANTIC_THREADS)
def atomic_add_multicore(state: pl.Tensor[[1, 1], pl.DT_INT64]):
    pl.simt.atomic_add(state[0, 0], 1)


@pl.jit()
def simt_atomic_add_multicore(state: pl.Tensor[[1, 1], pl.DT_INT64]):
    with pl.section_vector():
        atomic_add_multicore[SEMANTIC_THREADS](state)


def _assert_histogram(kernel):
    histogram = torch.zeros((1, ATOMIC_BUCKETS), dtype=torch.int32, device=ST_DEVICE)
    kernel[None, 1](histogram)
    torch.npu.synchronize()
    expected = torch.full((1, ATOMIC_BUCKETS), SEMANTIC_THREADS // ATOMIC_BUCKETS, dtype=torch.int32)
    torch.testing.assert_close(histogram.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_atomic_add_ub_all_supported_dtypes():
    states = [
        torch.full((1, ELEMENTS), 0, dtype=torch.int32),
        torch.full((1, ELEMENTS), 0, dtype=torch.uint32),
        torch.full((1, ELEMENTS), 0, dtype=torch.float16),
        torch.full((1, ELEMENTS), 0, dtype=torch.bfloat16),
        torch.full((1, ELEMENTS), 0, dtype=torch.float32),
    ]
    for state in _run_kernel(simt_atomic_add_ub_all_dtypes, states):
        _assert_target(state, THREADS)


@pytest.mark.soc("950")
def test_atomic_add_gm_all_supported_dtypes():
    states = [
        torch.full((1, ELEMENTS), 0, dtype=torch.int32),
        torch.full((1, ELEMENTS), 0, dtype=torch.uint32),
        torch.full((1, ELEMENTS), 0, dtype=torch.float16),
        torch.full((1, ELEMENTS), 0, dtype=torch.bfloat16),
        torch.full((1, ELEMENTS), 0, dtype=torch.float32),
        torch.full((1, ELEMENTS), 0, dtype=torch.int64),
        torch.full((1, ELEMENTS), 0, dtype=torch.uint64),
    ]
    for state in _run_kernel(simt_atomic_add_gm_all_dtypes, states):
        _assert_target(state, THREADS)


@pytest.mark.soc("950")
def test_atomic_add_returns_old_value_and_preserves_other_elements():
    _require_a5()
    state = torch.full((1, ELEMENTS), 123, dtype=torch.int32)
    state[0, 0] = 10
    expected_state = state.clone()
    expected_state[0, 0] = 15
    expected_old = torch.full((1, ELEMENTS), 123, dtype=torch.int32)
    expected_old[0, 0] = 10
    state_device = state.to(ST_DEVICE)
    old_values = torch.full((1, ELEMENTS), 123, dtype=torch.int32).to(ST_DEVICE)
    simt_atomic_add_return_value_gm[None, 1](state_device, old_values)
    torch.npu.synchronize()
    torch.testing.assert_close(state_device.cpu(), expected_state, rtol=0, atol=0)
    torch.testing.assert_close(old_values.cpu(), expected_old, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_atomic_add_dynamic_gm_offsets():
    _require_a5()
    _assert_histogram(simt_atomic_add_histogram_gm)


@pytest.mark.soc("950")
def test_atomic_add_dynamic_ub_offsets():
    _require_a5()
    _assert_histogram(simt_atomic_add_histogram_ub)


@pytest.mark.soc("950")
def test_atomic_add_gm_is_atomic_across_aiv_cores():
    _require_a5()
    state = torch.zeros((1, 1), dtype=torch.int64, device=ST_DEVICE)
    simt_atomic_add_multicore[None, ATOMIC_GRID_BLOCKS](state)
    torch.npu.synchronize()
    expected = torch.tensor([[ATOMIC_GRID_BLOCKS * SEMANTIC_THREADS]], dtype=torch.int64)
    torch.testing.assert_close(state.cpu(), expected, rtol=0, atol=0)
