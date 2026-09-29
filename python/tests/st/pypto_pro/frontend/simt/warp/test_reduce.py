# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 end-to-end tests for the SIMT Warp Reduce interfaces."""

import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
WARP_SIZE = 32


def _require_a5():
    try:
        torch.npu.set_device(ST_DEVICE)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


def _run_warp(kernel, expected, values):
    _require_a5()
    device_values = values.to(ST_DEVICE)
    actual = torch.empty_like(expected).to(ST_DEVICE)
    kernel(device_values, actual)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def write_warp_reduce_add(
    values: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
    output: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.warp_reduce_add(values[0, tid])


@pl.jit(arch="3510")
def simt_warp_reduce_add(
    values: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
    output: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
):
    with pl.section_vector():
        write_warp_reduce_add[WARP_SIZE](values, output)


@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def write_warp_reduce_max(
    values: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
    output: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.warp_reduce_max(values[0, tid])


@pl.jit(arch="3510")
def simt_warp_reduce_max(
    values: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
    output: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
):
    with pl.section_vector():
        write_warp_reduce_max[WARP_SIZE](values, output)


@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def write_warp_reduce_min(
    values: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
    output: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.warp_reduce_min(values[0, tid])


@pl.jit(arch="3510")
def simt_warp_reduce_min(
    values: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
    output: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
):
    with pl.section_vector():
        write_warp_reduce_min[WARP_SIZE](values, output)


@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def write_warp_reduce_fp16_divergent(
    values: pl.Tensor[[1, WARP_SIZE], pl.DT_FP16],
    output: pl.Tensor[[3, WARP_SIZE], pl.DT_FP16],
):
    tid = pl.simt.linear_thread_idx()
    lane = pl.simt.lane_id()
    if lane < WARP_SIZE // 2:
        output[0, tid] = pl.simt.warp_reduce_add(values[0, tid])
        output[1, tid] = pl.simt.warp_reduce_max(values[0, tid])
        output[2, tid] = pl.simt.warp_reduce_min(values[0, tid])
    else:
        value = values[0, tid]
        doubled = value + value
        output[0, tid] = pl.simt.warp_reduce_add(doubled)
        output[1, tid] = pl.simt.warp_reduce_max(doubled)
        output[2, tid] = pl.simt.warp_reduce_min(doubled)


@pl.jit(arch="3510")
def simt_warp_reduce_fp16_divergent(
    values: pl.Tensor[[1, WARP_SIZE], pl.DT_FP16],
    output: pl.Tensor[[3, WARP_SIZE], pl.DT_FP16],
):
    with pl.section_vector():
        write_warp_reduce_fp16_divergent[WARP_SIZE](values, output)


@pytest.mark.soc("950")
def test_warp_reduce_add():
    values = (torch.roll(torch.arange(WARP_SIZE, dtype=torch.float32), shifts=7) - 14).reshape(1, WARP_SIZE)
    expected = torch.full((1, WARP_SIZE), values.sum().item(), dtype=torch.float32)
    _run_warp(simt_warp_reduce_add, expected, values)


@pytest.mark.soc("950")
def test_warp_reduce_max():
    values = (torch.roll(torch.arange(WARP_SIZE, dtype=torch.float32), shifts=7) - 16).reshape(1, WARP_SIZE)
    expected = torch.full((1, WARP_SIZE), values.max().item(), dtype=torch.float32)
    _run_warp(simt_warp_reduce_max, expected, values)


@pytest.mark.soc("950")
def test_warp_reduce_min():
    values = (torch.roll(torch.arange(WARP_SIZE, dtype=torch.float32), shifts=7) - 16).reshape(1, WARP_SIZE)
    expected = torch.full((1, WARP_SIZE), values.min().item(), dtype=torch.float32)
    _run_warp(simt_warp_reduce_min, expected, values)


@pytest.mark.soc("950")
def test_warp_reduce_fp16_divergent():
    # Keep zero as both groups' minimum while doubling the upper group to distinguish its add and max results.
    half_warp_values = torch.arange(WARP_SIZE // 2, dtype=torch.float16)
    values = half_warp_values.repeat(2).reshape(1, WARP_SIZE)
    expected = torch.empty((3, WARP_SIZE), dtype=torch.float16)
    for start in (0, WARP_SIZE // 2):
        group = values[0, start:start + WARP_SIZE // 2]
        if start:
            group = group * 2
        expected[0, start:start + WARP_SIZE // 2] = group.sum()
        expected[1, start:start + WARP_SIZE // 2] = group.max()
        expected[2, start:start + WARP_SIZE // 2] = group.min()
    _run_warp(simt_warp_reduce_fp16_divergent, expected, values)
