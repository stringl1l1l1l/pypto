# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 NPU coverage for Python-compatible scalar numeric semantics."""

import math
import os

import pypto_pro.language as pl
import pytest
import torch
import torch_npu  # noqa: F401 — registers npu backend

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
FLOAT_FLOORDIV_OVERFLOW_CASES = [
    (3e38, 0.5, math.inf),
    (-3e38, 0.5, -math.inf),
    (3e38, -0.5, -math.inf),
    (-3e38, -0.5, math.inf),
]


@pl.jit()
def scalar_numeric_semantics_kernel(
    lhs: pl.DT_INT32,
    rhs: pl.DT_INT32,
    float_out: pl.Tensor[[1], pl.DT_FP32],
    int_out: pl.Tensor[[6], pl.DT_INT32],
):
    with pl.section_vector():
        flag = lhs > rhs
        float_out[0] = lhs / rhs
        int_out[0] = flag + 7
        int_out[1] = flag & 3
        int_out[2] = flag | 2
        int_out[3] = flag ^ 3
        int_out[4] = flag == 1
        int_out[5] = flag < 2


@pl.jit()
def scalar_float_divmod_kernel(
    lhs: pl.DT_FP32,
    rhs: pl.DT_FP32,
    out: pl.Tensor[[2], pl.DT_FP32],
):
    with pl.section_vector():
        out[0] = lhs // rhs
        out[1] = lhs % rhs


@pl.jit()
def scalar_float_static_shape_kernel(src: pl.Tensor[[1, 8], pl.DT_FP32], out: pl.Tensor[[1, 8], pl.DT_FP32]):
    rows = 7.0 % 2.0
    cols = 16.0 // 2.0
    tile_type = pl.TileType(shape=[int(rows), int(cols)], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile = pl.make_tile(tile_type, addr=0)
    with pl.section_vector():
        pl.load(tile, src, [0, 0])
        pl.store(out, tile, [0, 0])


def _run_scalar_case(lhs, rhs, expected_div, expected_int):
    torch.npu.set_device(ST_DEVICE)
    float_out = torch.zeros(1, device=ST_DEVICE, dtype=torch.float32)
    int_out = torch.zeros(6, device=ST_DEVICE, dtype=torch.int32)

    scalar_numeric_semantics_kernel(lhs, rhs, float_out, int_out)
    torch.npu.synchronize()

    expected_float = torch.tensor([expected_div], device=ST_DEVICE, dtype=torch.float32)
    expected_numeric = torch.tensor(expected_int, device=ST_DEVICE, dtype=torch.int32)
    torch.testing.assert_close(float_out, expected_float, rtol=0, atol=0)
    assert torch.equal(int_out, expected_numeric), (
        f"got {int_out.cpu().tolist()}, expected {expected_numeric.cpu().tolist()}"
    )


def _run_float_divmod_case(lhs, rhs, expected_quotient):
    torch.npu.set_device(ST_DEVICE)
    out = torch.empty(2, device=ST_DEVICE, dtype=torch.float32)

    scalar_float_divmod_kernel(lhs, rhs, out)
    torch.npu.synchronize()

    expected_remainder = torch.remainder(
        torch.tensor(lhs, dtype=torch.float32), torch.tensor(rhs, dtype=torch.float32)
    ).item()
    expected = torch.tensor([expected_quotient, expected_remainder], device=ST_DEVICE, dtype=torch.float32)
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)
    return out.cpu()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_runtime_int_truediv_and_bool_numeric_semantics():
    _run_scalar_case(7, 2, 3.5, [8, 1, 3, 2, 1, 1])
    _run_scalar_case(1, 2, 0.5, [7, 0, 2, 3, 0, 1])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_runtime_float_floordiv_and_mod_semantics():
    for lhs, rhs, expected_quotient in [
        (7.0, 2.0, 3.0),
        (-7.0, 2.0, -4.0),
        (7.0, -2.0, -4.0),
        (-7.0, -2.0, 3.0),
        (1.0, 0.1, 9.0),
    ]:
        _run_float_divmod_case(lhs, rhs, expected_quotient)

    negative_zero = _run_float_divmod_case(4.0, -2.0, -2.0)[1]
    positive_zero = _run_float_divmod_case(-4.0, 2.0, -2.0)[1]
    assert torch.signbit(negative_zero).item()
    assert not torch.signbit(positive_zero).item()


@pytest.mark.soc("950")
@pytest.mark.parametrize(
    "lhs,rhs,expected_quotient",
    FLOAT_FLOORDIV_OVERFLOW_CASES,
)
@pypto.options(pass_options={"enable_slice": False})
def test_runtime_float_floordiv_preserves_overflow(lhs, rhs, expected_quotient):
    _run_float_divmod_case(lhs, rhs, expected_quotient)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_float_divmod_defines_static_tile_shape():
    torch.npu.set_device(ST_DEVICE)
    src = torch.arange(8, dtype=torch.float32, device=ST_DEVICE).reshape(1, 8)
    out = torch.empty_like(src)
    # 单 tile 语义验证：直调现为满核，显式请求 1 块保持单核语义。
    scalar_float_static_shape_kernel[None, 1](src, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, src, rtol=0, atol=0)


if __name__ == "__main__":
    test_runtime_int_truediv_and_bool_numeric_semantics()
    test_runtime_float_floordiv_and_mod_semantics()
    for overflow_case in FLOAT_FLOORDIV_OVERFLOW_CASES:
        test_runtime_float_floordiv_preserves_overflow(*overflow_case)
    test_float_divmod_defines_static_tile_shape()
