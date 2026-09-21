# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""NPU regressions for nested carries, loop targets, and multiple returns."""

import os

import pypto_pro.language as pl
import pytest
import torch
import torch_npu  # noqa: F401


@pl.jit
def nested_for(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[1, 8], pl.DT_INT64]):
    with pl.section_vector():
        state = pl.make_tuple(a=pl.make_tuple(v=x), b=pl.make_tuple(v=x + 1))
        for i in pl.range(n):
            state = pl.make_tuple(a=state.b, b=state.a)
        out[0, 0] = state.a.v
        out[0, 1] = state.b.v


def multi(a, b, n):
    for i in pl.range(n):
        if i == 2:
            return b, a
        a, b = b, a
    return a, b


@pl.jit
def multi_return(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[1, 8], pl.DT_INT64]):
    with pl.section_vector():
        a, b = multi(x, x + 1, n)
        out[0, 0] = a
        out[0, 1] = b


@pl.jit
def nested_while(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[1, 8], pl.DT_INT64]):
    with pl.section_vector():
        state = pl.make_tuple(a=pl.make_tuple(v=x), b=pl.make_tuple(v=x + 1), count=0)
        while state.count < n:
            state = pl.make_tuple(a=state.b, b=state.a, count=state.count + 1)
            continue
        out[0, 0] = state.a.v
        out[0, 1] = state.b.v
        out[0, 2] = state.count


@pl.jit
def nested_break(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[1, 8], pl.DT_INT64]):
    with pl.section_vector():
        state = pl.make_tuple(a=pl.make_tuple(v=x), b=pl.make_tuple(v=x + 1))
        for i in pl.range(n):
            state = pl.make_tuple(a=state.b, b=state.a)
            break
        out[0, 0] = state.a.v
        out[0, 1] = state.b.v


@pl.jit
def composite_rebind(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[1, 8], pl.DT_INT64]):
    with pl.section_vector():
        state = pl.make_tuple(a=(x, x + 1), b=(x + 2, x + 3))
        for i in pl.range(n):
            state = pl.make_tuple(a=state.b, b=state.a)
        out[0, 0] = state.a[0]
        out[0, 1] = state.a[1]
        out[0, 2] = state.b[0]
        out[0, 3] = state.b[1]


@pl.jit
def nested_induction(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[1, 8], pl.DT_INT64]):
    with pl.section_vector():
        i = x
        for j in pl.range(n):
            for i in pl.range(3):
                pass
        out[0, 0] = i


@pl.jit
def branch_induction(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[1, 8], pl.DT_INT64]):
    with pl.section_vector():
        i = x
        if n > 0:
            for i in pl.range(3):
                pass
        else:
            for i in pl.range(5):
                pass
        out[0, 0] = i


@pl.jit
def while_induction(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[1, 8], pl.DT_INT64]):
    with pl.section_vector():
        i = 0
        steps = 0
        while i < n and steps < 4:
            for i in pl.range(n + 1):
                pass
            steps += 1
        out[0, 0] = i
        out[0, 1] = steps


def hetero_multi_helper(state, n):
    for i in pl.range(n):
        state = pl.make_tuple(a=state.b, b=state.a)
        if i == 1:
            return pl.make_tuple(first=state.b, second=state.a)
    return pl.make_tuple(first=state.a, second=state.b)


@pl.jit
def hetero_multi(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[1, 8], pl.DT_INT64]):
    with pl.section_vector():
        state = pl.make_tuple(a=pl.make_tuple(v=x), b=pl.make_tuple(v=x + 1))
        a, b = hetero_multi_helper(state, n)
        out[0, 0] = a.v
        out[0, 1] = b.v


def early_return_helper(a, b, n):
    for i in pl.range(n):
        j = 0
        while j < n:
            if j == 1:
                if i == 0:
                    return b, a, i + j
                else:
                    return a, b, i - j
            a, b = b, a
            j += 1
    return a, b, n


@pl.jit
def nested_returns(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[1, 8], pl.DT_INT64]):
    with pl.section_vector():
        a, b, count = early_return_helper(x, x + 1, n)
        out[0, 0] = a
        out[0, 1] = b
        out[0, 2] = count


def collision_return_val(return_val, other):
    return return_val, other


@pl.jit
def return_name_collision(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[1, 8], pl.DT_INT64]):
    with pl.section_vector():
        a, b = collision_return_val(x, x + 1)
        out[0, 0] = a
        out[0, 1] = b


def collision_returned(a, b, n):
    returned = n
    for i in pl.range(2):
        if i == 1:
            return a, b, returned
        a, b = b, a
    return a, b, returned


@pl.jit
def returned_name_collision(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[1, 8], pl.DT_INT64]):
    with pl.section_vector():
        a, b, c = collision_returned(x, x + 1, n)
        out[0, 0] = a
        out[0, 1] = b
        out[0, 2] = c


def side_effect_predicate(out):
    out[0, 0] = out[0, 0] + 1
    return out[0, 0] > 0


@pl.jit
def ternary_condition_twice(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[1, 8], pl.DT_INT64]):
    with pl.section_vector():
        selected = x if side_effect_predicate(out) else n
        out[0, 1] = selected


def _swap_expected(x, n):
    return [x + n % 2, x + 1 - n % 2]


@pytest.mark.soc("950")
@pytest.mark.parametrize("n", range(5))
@pytest.mark.parametrize(
    "kernel, expected",
    [
        (nested_for, _swap_expected),
        (ternary_condition_twice, lambda x, n:[1, x]),
        (return_name_collision, lambda x, n:[x, x + 1]),
        (returned_name_collision, lambda x, n:[x + 1, x, n]),
        (nested_while, lambda x, n:[*_swap_expected(x, n), n]),
        (nested_break, lambda x, n:_swap_expected(x, min(n, 1))),
        (composite_rebind, lambda x, n:[x + (2 * n + k) % 4 for k in range(4)]),
        (nested_induction, lambda x, n:[2 if n else x]),
        (branch_induction, lambda x, n:[2 if n else 4]),
        (while_induction, lambda x, n:[n, int(n > 0)]),
        (multi_return, lambda x, n:_swap_expected(x, min(n, 3))),
        (hetero_multi, lambda x, n:_swap_expected(x, min(n, 1))),
        (nested_returns, lambda x, n:[*_swap_expected(x, int(n == 1)), min(n, 1)]),
    ],
    ids=lambda value: value.__name__,
)
def test_control_flow_values(kernel, expected, n):
    device = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device)
    out = torch.zeros((1, 8), dtype=torch.int64, device=f"npu:{device}")
    kernel[None, 1](7, n, out)
    torch.npu.synchronize()
    values = expected(7, n)
    assert out.cpu().tolist() == [values + [0] * (8 - len(values))]
