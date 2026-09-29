# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 end-to-end tests for the SIMT Warp Vote interfaces."""

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
def write_warp_active_mask(output: pl.Tensor[[3, WARP_SIZE], pl.DT_UINT32]):
    tid = pl.simt.linear_thread_idx()
    lane = pl.simt.lane_id()
    warp_size = pl.simt.warp_size()
    output[0, tid] = pl.simt.warp_active_mask()
    if lane < warp_size // 2:
        output[1, tid] = pl.simt.warp_active_mask()
        output[2, tid] = 0
    else:
        output[1, tid] = 0
        output[2, tid] = pl.simt.warp_active_mask()


@pl.jit(arch="3510")
def simt_warp_active_mask(output: pl.Tensor[[3, WARP_SIZE], pl.DT_UINT32]):
    with pl.section_vector():
        write_warp_active_mask[WARP_SIZE](output)


@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def write_warp_all(output: pl.Tensor[[2, WARP_SIZE], pl.DT_INT32]):
    tid = pl.simt.linear_thread_idx()
    lane = pl.simt.lane_id()
    warp_size = pl.simt.warp_size()
    output[0, tid] = pl.simt.warp_all(lane < warp_size)
    output[1, tid] = pl.simt.warp_all(lane < warp_size - 1)


@pl.jit(arch="3510")
def simt_warp_all(output: pl.Tensor[[2, WARP_SIZE], pl.DT_INT32]):
    with pl.section_vector():
        write_warp_all[WARP_SIZE](output)


@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def write_warp_any(output: pl.Tensor[[2, WARP_SIZE], pl.DT_INT32]):
    tid = pl.simt.linear_thread_idx()
    lane = pl.simt.lane_id()
    warp_size = pl.simt.warp_size()
    output[0, tid] = pl.simt.warp_any(lane == warp_size - 1)
    output[1, tid] = pl.simt.warp_any(lane == warp_size)


@pl.jit(arch="3510")
def simt_warp_any(output: pl.Tensor[[2, WARP_SIZE], pl.DT_INT32]):
    with pl.section_vector():
        write_warp_any[WARP_SIZE](output)


@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def write_warp_ballot(output: pl.Tensor[[2, WARP_SIZE], pl.DT_UINT32]):
    tid = pl.simt.linear_thread_idx()
    lane = pl.simt.lane_id()
    warp_size = pl.simt.warp_size()
    output[0, tid] = pl.simt.warp_ballot((lane % 2) == 0)
    output[1, tid] = pl.simt.warp_ballot(lane < warp_size // 2)


@pl.jit(arch="3510")
def simt_warp_ballot(output: pl.Tensor[[2, WARP_SIZE], pl.DT_UINT32]):
    with pl.section_vector():
        write_warp_ballot[WARP_SIZE](output)


@pytest.mark.soc("950")
def test_warp_active_mask():
    expected = torch.stack(
        (
            torch.full((WARP_SIZE,), 0xFFFFFFFF, dtype=torch.uint32),
            torch.cat(
                (
                    torch.full((WARP_SIZE // 2,), 0x0000FFFF, dtype=torch.uint32),
                    torch.zeros(WARP_SIZE // 2, dtype=torch.uint32),
                )
            ),
            torch.cat(
                (
                    torch.zeros(WARP_SIZE // 2, dtype=torch.uint32),
                    torch.full((WARP_SIZE // 2,), 0xFFFF0000, dtype=torch.uint32),
                )
            ),
        )
    )
    _run_warp(simt_warp_active_mask, expected)


@pytest.mark.soc("950")
def test_warp_all_true_and_false_predicates():
    expected = torch.stack((torch.ones(WARP_SIZE, dtype=torch.int32), torch.zeros(WARP_SIZE, dtype=torch.int32)))
    _run_warp(simt_warp_all, expected)


@pytest.mark.soc("950")
def test_warp_any_true_and_false_predicates():
    expected = torch.stack((torch.ones(WARP_SIZE, dtype=torch.int32), torch.zeros(WARP_SIZE, dtype=torch.int32)))
    _run_warp(simt_warp_any, expected)


@pytest.mark.soc("950")
def test_warp_ballot_predicate_masks():
    expected = torch.stack(
        (
            torch.full((WARP_SIZE,), 0x55555555, dtype=torch.uint32),
            torch.full((WARP_SIZE,), 0x0000FFFF, dtype=torch.uint32),
        )
    )
    _run_warp(simt_warp_ballot, expected)
