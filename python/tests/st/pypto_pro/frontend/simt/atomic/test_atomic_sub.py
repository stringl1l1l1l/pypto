# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 end-to-end tests for the SIMT atomic_sub interface."""

import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
THREADS = 64
ELEMENTS = 64


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
def atomic_sub_ub_all_dtypes(
    int32_tile,
    uint32_tile,
    fp32_tile,
):
    pl.simt.atomic_sub(int32_tile[0, 0], 1)
    pl.simt.atomic_sub(uint32_tile[0, 0], 1)
    pl.simt.atomic_sub(fp32_tile[0, 0], 1.0)


@pl.jit()
def simt_atomic_sub_ub_all_dtypes(
    int32_state: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    uint32_state: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
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
    fp32_tile = pl.make_tile(
        pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x0800,
    )
    with pl.section_vector():
        pl.load(int32_tile, int32_state, [0, 0])
        pl.load(uint32_tile, uint32_state, [0, 0])
        pl.load(fp32_tile, fp32_state, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        atomic_sub_ub_all_dtypes[THREADS](int32_tile, uint32_tile, fp32_tile)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(int32_state, int32_tile, [0, 0])
        pl.store(uint32_state, uint32_tile, [0, 0])
        pl.store(fp32_state, fp32_tile, [0, 0])


@pl.vector_function(mode="simt", max_threads=THREADS)
def atomic_sub_gm_all_dtypes(
    int32_state: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    uint32_state: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    fp32_state: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    int64_state: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    uint64_state: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
):
    pl.simt.atomic_sub(int32_state[0, 0], 1)
    pl.simt.atomic_sub(uint32_state[0, 0], 1)
    pl.simt.atomic_sub(fp32_state[0, 0], 1.0)
    pl.simt.atomic_sub(int64_state[0, 0], 1)
    pl.simt.atomic_sub(uint64_state[0, 0], 1)


@pl.jit()
def simt_atomic_sub_gm_all_dtypes(
    int32_state: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    uint32_state: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    fp32_state: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    int64_state: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    uint64_state: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
):
    with pl.section_vector():
        atomic_sub_gm_all_dtypes[THREADS](int32_state, uint32_state, fp32_state, int64_state, uint64_state)


@pl.vector_function(mode="simt", max_threads=1)
def atomic_sub_return_value_gm(
    state: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    old_values: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
):
    old_values[0, 0] = pl.simt.atomic_sub(state[0, 0], 3)


@pl.jit()
def simt_atomic_sub_return_value_gm(
    state: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    old_values: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
):
    with pl.section_vector():
        atomic_sub_return_value_gm[1](state, old_values)


@pytest.mark.soc("950")
def test_atomic_sub_ub_all_supported_dtypes():
    states = [
        torch.full((1, ELEMENTS), 100, dtype=torch.int32),
        torch.full((1, ELEMENTS), 100, dtype=torch.uint32),
        torch.full((1, ELEMENTS), 100, dtype=torch.float32),
    ]
    for state in _run_kernel(simt_atomic_sub_ub_all_dtypes, states):
        _assert_target(state, 100 - THREADS)


@pytest.mark.soc("950")
def test_atomic_sub_gm_all_supported_dtypes():
    states = [
        torch.full((1, ELEMENTS), 100, dtype=torch.int32),
        torch.full((1, ELEMENTS), 100, dtype=torch.uint32),
        torch.full((1, ELEMENTS), 100, dtype=torch.float32),
        torch.full((1, ELEMENTS), 100, dtype=torch.int64),
        torch.full((1, ELEMENTS), 100, dtype=torch.uint64),
    ]
    for state in _run_kernel(simt_atomic_sub_gm_all_dtypes, states):
        _assert_target(state, 100 - THREADS)


@pytest.mark.soc("950")
def test_atomic_sub_returns_old_value_and_preserves_other_elements():
    _require_a5()
    state = torch.full((1, ELEMENTS), 123, dtype=torch.int32)
    state[0, 0] = 10
    expected_state = state.clone()
    expected_state[0, 0] = 7
    expected_old = torch.full((1, ELEMENTS), 123, dtype=torch.int32)
    expected_old[0, 0] = 10
    state_device = state.to(ST_DEVICE)
    old_values = torch.full((1, ELEMENTS), 123, dtype=torch.int32).to(ST_DEVICE)
    simt_atomic_sub_return_value_gm[None, 1](state_device, old_values)
    torch.npu.synchronize()
    torch.testing.assert_close(state_device.cpu(), expected_state, rtol=0, atol=0)
    torch.testing.assert_close(old_values.cpu(), expected_old, rtol=0, atol=0)
