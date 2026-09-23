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
"""CCE codegen tests for float-scalar-to-integer coercion (CoerceScalarToInt).

A float literal applied to an integer register must be emitted as a correctly-typed integer literal,
not as a float literal that would trigger -Wliteral-conversion. The value is truncated to the width of
the source dtype; the frontend range check rejects anything the dtype cannot represent before it gets
here, so only the fractional part is dropped.

Each case asserts the IR as well as the generated CCE, because the two say different things: the
parser hands codegen the scalar exactly as it was written, and the truncation is codegen's own doing.
"""

from pypto_pro import ir
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
import pytest

_N, _M = 1, 64


def _compile_to_cce(kernel) -> str:
    from pypto_pro.runtime.jit import _assemble_cv_source, _parse_and_codegen_targets

    cube, vector = _parse_and_codegen_targets(kernel.to_kernel_def(), "3510", "")
    return _assemble_cv_source(cube, vector).content


def _parse_to_ir(kernel) -> str:
    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    return str(program)


def _adds_kernel(dtype, scalar, elem_bytes):
    size = _N * _M * elem_bytes

    @pl.vector_function
    def vf_body(in_a, t_out):
        preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=dtype)
        reg_a = vf.load_align(in_a, 0)
        reg_out = vf.adds(reg_a, scalar, preg)
        vf.store_align(t_out, reg_out, preg)

    @pl.jit()
    def kernel(a: pl.Tensor[[_N, _M], dtype], out: pl.Tensor[[_N, _M], dtype]):
        tf = pl.TileType(shape=[_N, _M], dtype=dtype, target_memory=pl.MemorySpace.Vec)
        in_a = pl.make_tile(tf, addr=0)
        t_out = pl.make_tile(tf, addr=size)
        with pl.section_vector():
            pl.load(in_a, a, [0, 0])
            vf_body(in_a, t_out)
            pl.store(out, t_out, [0, 0])

    return kernel


def _vadds_arg(cpp: str) -> str:
    """The scalar operand of the emitted vadds, i.e. its third argument."""
    line = next(ln for ln in cpp.splitlines() if "vadds(" in ln)
    return line.split("vadds(", 1)[1].split(",")[2].strip()


def _vf_adds_scalar(ir_text: str) -> str:
    """The scalar operand of the vf.adds statement in the IR, i.e. its third argument."""
    line = next(ln for ln in ir_text.splitlines() if "vf.adds(" in ln)
    return line.split("vf.adds(", 1)[1].split(",")[2].strip()


@pytest.mark.soc("950")
@pytest.mark.parametrize(
    ("dtype", "elem_bytes", "scalar", "expected"),
    [
        # 8-bit had no branch at all before: the float literal fell straight through.
        (pl.DT_INT8, 1, 1.9, "1"),
        (pl.DT_INT8, 1, -1.9, "-1"),
        (pl.DT_UINT8, 1, 200.7, "200u"),
        # Widths the original cascade already covered, kept as a regression guard.
        (pl.DT_INT16, 2, 3.5, "3"),
        (pl.DT_INT32, 4, -7.9, "-7"),
        (pl.DT_UINT32, 4, 9.9, "9u"),
    ],
)
def test_float_scalar_is_emitted_as_a_truncated_integer_literal(dtype, elem_bytes, scalar, expected):
    kernel = _adds_kernel(dtype, scalar, elem_bytes)

    ir_text = _parse_to_ir(kernel)
    assert "vf.adds(" in ir_text
    # The IR carries the scalar as written -- nothing has been truncated yet.
    assert _vf_adds_scalar(ir_text) == str(scalar)

    cpp = _compile_to_cce(kernel)

    assert _vadds_arg(cpp) == expected


@pytest.mark.soc("950")
def test_float_scalar_keeps_its_float_form_for_a_float_dtype():
    """Coercion applies only to integer source dtypes."""
    kernel = _adds_kernel(pl.DT_FP32, 1.5, 4)

    ir_text = _parse_to_ir(kernel)
    assert "vf.reg_tensor(dtype=float)" in ir_text
    assert _vf_adds_scalar(ir_text) == "1.5"

    cpp = _compile_to_cce(kernel)

    assert _vadds_arg(cpp) == "1.5f"
