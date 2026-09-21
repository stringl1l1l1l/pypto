# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Device regressions for simultaneous loop-carried assignments."""

import math
import os

import pypto_pro.language as pl
import pytest
import torch
import torch_npu  # noqa: F401


def _integer_sqrt(x):
    if x < 2:
        return x
    res = x
    nxt = (res + 1) // 2
    while nxt < res:
        res = nxt
        nxt = (res + x // res) // 2
    return res


@pl.jit
def _while_isqrt_carry_kernel(x: pl.DT_INT64, out: pl.Tensor[[1, 3], pl.DT_INT64]):
    with pl.section_vector():
        out[0, 0] = _integer_sqrt(x)


@pl.jit
def _while_swap_carry_kernel(x: pl.DT_INT64, out: pl.Tensor[[1, 3], pl.DT_INT64]):
    with pl.section_vector():
        left = x
        right = x + 1
        steps = 0
        # A bounded guard makes broken parallel copies fail without hanging a test.
        while right != x and steps < 4:
            old_left = left
            left = right
            right = old_left
            steps += 1
            continue
        out[0, 0] = left
        out[0, 1] = right
        out[0, 2] = steps


@pl.jit
def _for_swap_carry_kernel(x: pl.DT_INT64, out: pl.Tensor[[1, 3], pl.DT_INT64]):
    with pl.section_vector():
        left = x
        right = x + 1
        for i in pl.range(3):
            old_left = left
            left = right
            right = old_left
            if i == 2:
                break
            continue
        out[0, 0] = left
        out[0, 1] = right


@pytest.mark.soc("950")
@pytest.mark.parametrize("x", [0, 1, 2, 3, 4, 8, 9, 15, 16, 31, 32, 2147395600])
def test_while_isqrt_carry(x):
    device = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device)
    out = torch.zeros((1, 3), dtype=torch.int64, device=f"npu:{device}")
    _while_isqrt_carry_kernel[None, 1](x, out)
    torch.npu.synchronize()
    assert out.cpu().tolist() == [[math.isqrt(x), 0, 0]]


@pytest.mark.soc("950")
@pytest.mark.parametrize(
    "kernel, expected",
    [(_while_swap_carry_kernel, [8, 7, 1]), (_for_swap_carry_kernel, [8, 7, 0])],
)
def test_loop_swap_carry(kernel, expected):
    device = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device)
    out = torch.zeros((1, 3), dtype=torch.int64, device=f"npu:{device}")
    kernel[None, 1](7, out)
    torch.npu.synchronize()
    assert out.cpu().tolist() == [expected]
