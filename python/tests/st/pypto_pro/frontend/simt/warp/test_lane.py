# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 end-to-end tests for the SIMT Lane interfaces."""

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


def _run_warp(kernel, expected):
    _require_a5()
    actual = torch.empty_like(expected).to(ST_DEVICE)
    kernel(actual)
    torch.npu.synchronize()
    actual = actual.cpu()
    if actual.dtype in (torch.uint32, torch.uint64):
        actual = actual.to(torch.int64)
        expected = expected.to(torch.int64)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def write_lane_id(output: pl.Tensor[[1, WARP_SIZE], pl.DT_INT32]):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.lane_id()


@pl.jit(arch="3510")
def simt_lane_id(output: pl.Tensor[[1, WARP_SIZE], pl.DT_INT32]):
    with pl.section_vector():
        write_lane_id[WARP_SIZE](output)


@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def write_lanemask_eq(output: pl.Tensor[[1, WARP_SIZE], pl.DT_INT32]):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.lanemask_eq()


@pl.jit(arch="3510")
def simt_lanemask_eq(output: pl.Tensor[[1, WARP_SIZE], pl.DT_INT32]):
    with pl.section_vector():
        write_lanemask_eq[WARP_SIZE](output)


@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def write_lanemask_ge(output: pl.Tensor[[1, WARP_SIZE], pl.DT_INT32]):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.lanemask_ge()


@pl.jit(arch="3510")
def simt_lanemask_ge(output: pl.Tensor[[1, WARP_SIZE], pl.DT_INT32]):
    with pl.section_vector():
        write_lanemask_ge[WARP_SIZE](output)


@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def write_lanemask_gt(output: pl.Tensor[[1, WARP_SIZE], pl.DT_INT32]):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.lanemask_gt()


@pl.jit(arch="3510")
def simt_lanemask_gt(output: pl.Tensor[[1, WARP_SIZE], pl.DT_INT32]):
    with pl.section_vector():
        write_lanemask_gt[WARP_SIZE](output)


@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def write_lanemask_le(output: pl.Tensor[[1, WARP_SIZE], pl.DT_INT32]):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.lanemask_le()


@pl.jit(arch="3510")
def simt_lanemask_le(output: pl.Tensor[[1, WARP_SIZE], pl.DT_INT32]):
    with pl.section_vector():
        write_lanemask_le[WARP_SIZE](output)


@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def write_lanemask_lt(output: pl.Tensor[[1, WARP_SIZE], pl.DT_INT32]):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.lanemask_lt()


@pl.jit(arch="3510")
def simt_lanemask_lt(output: pl.Tensor[[1, WARP_SIZE], pl.DT_INT32]):
    with pl.section_vector():
        write_lanemask_lt[WARP_SIZE](output)


@pytest.mark.soc("950")
def test_lane_id():
    expected = torch.arange(WARP_SIZE, dtype=torch.int32).reshape(1, WARP_SIZE)
    _run_warp(simt_lane_id, expected)


@pytest.mark.soc("950")
def test_lanemask_eq():
    lane = torch.arange(WARP_SIZE, dtype=torch.int64)
    expected = (1 << lane).to(torch.int32).reshape(1, WARP_SIZE)
    _run_warp(simt_lanemask_eq, expected)


@pytest.mark.soc("950")
def test_lanemask_ge():
    lane = torch.arange(WARP_SIZE, dtype=torch.int64)
    expected = (((1 << WARP_SIZE) - 1) ^ ((1 << lane) - 1)).to(torch.int32).reshape(1, WARP_SIZE)
    _run_warp(simt_lanemask_ge, expected)


@pytest.mark.soc("950")
def test_lanemask_gt():
    lane = torch.arange(WARP_SIZE, dtype=torch.int64)
    expected = (((1 << WARP_SIZE) - 1) ^ ((1 << (lane + 1)) - 1)).to(torch.int32).reshape(1, WARP_SIZE)
    _run_warp(simt_lanemask_gt, expected)


@pytest.mark.soc("950")
def test_lanemask_le():
    lane = torch.arange(WARP_SIZE, dtype=torch.int64)
    expected = ((1 << (lane + 1)) - 1).to(torch.int32).reshape(1, WARP_SIZE)
    _run_warp(simt_lanemask_le, expected)


@pytest.mark.soc("950")
def test_lanemask_lt():
    lane = torch.arange(WARP_SIZE, dtype=torch.int64)
    expected = ((1 << lane) - 1).to(torch.int32).reshape(1, WARP_SIZE)
    _run_warp(simt_lanemask_lt, expected)
