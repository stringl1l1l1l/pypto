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
"""Front-end validation of pl.range() start/stop bounds.

Three rules are enforced at parse time, matching what codegen can lower:

  * Outside a VF section the bound must fit int64 (the loop variable is int64_t).
  * Inside a VF section the bound must be a non-negative uint16 in [0, 65535] (the loop variable
    is uint16_t and the stop expression is cast to uint16_t).
  * A bound must be an integer scalar: float / bool / tuple / str / dtype / tile are rejected.

step is assumed to be a positive integer and is not validated here.
"""
from pypto_pro import ir
from pypto_pro._errors import InvalidArgument, InvalidVal, OutOfRange
from pypto_pro.ir._limits import INT64_MAX, INT64_MIN
import pypto_pro.language as pl
import pytest

_UINT16_MAX = 65535


# ---------------------------------------------------------------------------
# Non-VF range: bounds must fit int64, negatives allowed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stop", [0, 1, 10, INT64_MAX, _UINT16_MAX + 1])
def test_non_vf_accepts_int64_stop(stop):
    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[64], pl.DT_FP32]):
        for _ in pl.range(stop):
            pl.system.bar_all()

    kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


@pytest.mark.parametrize("start", [-1, INT64_MIN])
def test_non_vf_accepts_negative_start(start):
    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[64], pl.DT_FP32]):
        for _ in pl.range(start, 10):
            pl.system.bar_all()

    kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_non_vf_rejects_folded_stop_beyond_int64():
    """An INT64 constant expression rejects overflow before range lowering."""

    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[64], pl.DT_FP32]):
        for _ in pl.range(INT64_MAX + 1):
            pl.system.bar_all()

    with pytest.raises(OutOfRange, match=r"integer constant must be representable in int64"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


@pytest.mark.parametrize("stop", [2**200])
def test_non_vf_rejects_stop_beyond_storage_band(stop):
    """Above UINT64_MAX the value cannot even materialise into ir.ConstInt, so the storage-band
    check fires first, before pl.range's own bound check."""

    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[64], pl.DT_FP32]):
        for _ in pl.range(stop):
            pl.system.bar_all()

    with pytest.raises(OutOfRange, match=r"integer constant must be in"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_non_vf_rejects_start_below_storage_band():
    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[64], pl.DT_FP32]):
        for _ in pl.range(INT64_MIN - 1, 10):
            pl.system.bar_all()

    with pytest.raises(OutOfRange, match=r"integer constant must be representable in int64"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


# ---------------------------------------------------------------------------
# VF range: bounds must be non-negative uint16 in [0, 65535]
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stop", [0, 1, _UINT16_MAX])
def test_vf_accepts_uint16_stop(stop):
    @pl.vector_function
    def vf_body(captured_stop):
        for _ in pl.range(captured_stop):
            pass

    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[64], pl.DT_FP32]):
        vf_body(stop)

    kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_vf_rejects_stop_above_uint16():
    @pl.vector_function
    def vf_body():
        for _ in pl.range(_UINT16_MAX + 1):
            pass

    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[64], pl.DT_FP32]):
        vf_body()

    with pytest.raises(OutOfRange, match=r"pl\.range\(\) stop must be in \[0, 65535\]"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_vf_rejects_negative_start():
    @pl.vector_function
    def vf_body():
        for _ in pl.range(-1, 10):
            pass

    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[64], pl.DT_FP32]):
        vf_body()

    with pytest.raises(OutOfRange, match=r"pl\.range\(\) start must be in \[0, 65535\]"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


# ---------------------------------------------------------------------------
# Type: float / bool bounds are rejected everywhere
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stop", [1.5, 1e3])
def test_rejects_float_literal_stop(stop):
    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[64], pl.DT_FP32]):
        for _ in pl.range(stop):
            pl.system.bar_all()

    with pytest.raises(InvalidVal, match=r"pl\.range\(\): stop must be an integer, got float"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


@pytest.mark.parametrize("stop", [True, False])
def test_rejects_bool_literal_stop(stop):
    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[64], pl.DT_FP32]):
        for _ in pl.range(stop):
            pl.system.bar_all()

    with pytest.raises(InvalidVal, match=r"pl\.range\(\): stop must be an integer, got bool"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_rejects_float_start():
    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[64], pl.DT_FP32]):
        for _ in pl.range(0.0, 10):
            pl.system.bar_all()

    with pytest.raises(InvalidVal, match=r"pl\.range\(\): start must be an integer, got float"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_rejects_float_scalar_param_stop():
    @pl.jit(auto_mutex=False)
    def kernel(n: pl.DT_FP32, x: pl.Tensor[[64], pl.DT_FP32]):
        for _ in pl.range(n):
            pl.system.bar_all()

    with pytest.raises(InvalidVal, match=r"pl\.range\(\): stop must be an integer, got float"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_rejects_bool_scalar_param_stop():
    @pl.jit(auto_mutex=False)
    def kernel(flag: pl.DT_BOOL, x: pl.Tensor[[64], pl.DT_FP32]):
        for _ in pl.range(flag):
            pl.system.bar_all()

    with pytest.raises(InvalidVal, match=r"pl\.range\(\): stop must be an integer, got bool"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


# ---------------------------------------------------------------------------
# Type: non-scalar bounds (tuple / list / str / dtype) are rejected
# ---------------------------------------------------------------------------
def test_rejects_tuple_stop():
    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[64], pl.DT_FP32]):
        for _ in pl.range((0, 10)):
            pl.system.bar_all()

    with pytest.raises(InvalidVal, match=r"pl\.range\(\): stop must be an integer scalar"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_rejects_list_stop():
    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[64], pl.DT_FP32]):
        for _ in pl.range([0, 10]):
            pl.system.bar_all()

    with pytest.raises(InvalidVal, match=r"pl\.range\(\): stop must be an integer scalar"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_rejects_string_stop():
    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[64], pl.DT_FP32]):
        for _ in pl.range("10"):
            pl.system.bar_all()

    with pytest.raises(InvalidVal, match=r"pl\.range\(\): stop must be an integer scalar"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_rejects_dtype_stop():
    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[64], pl.DT_FP32]):
        for _ in pl.range(pl.DT_INT32):
            pl.system.bar_all()

    with pytest.raises(InvalidVal, match=r"pl\.range\(\): stop must be an integer scalar"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


# ---------------------------------------------------------------------------
# step must be a positive integer
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("step", [1, 2, 10])
def test_accepts_positive_step(step):
    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[64], pl.DT_FP32]):
        for _ in pl.range(0, 10, step):
            pl.system.bar_all()

    kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


@pytest.mark.parametrize("step", [0, -1])
def test_rejects_non_positive_step(step):
    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[64], pl.DT_FP32]):
        for _ in pl.range(0, 10, step):
            pl.system.bar_all()

    with pytest.raises(OutOfRange, match=r"pl\.range\(\) step must be in \[1,"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


@pytest.mark.parametrize("step", [1.0, True])
def test_rejects_non_integer_step(step):
    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[64], pl.DT_FP32]):
        for _ in pl.range(0, 10, step):
            pl.system.bar_all()

    with pytest.raises(InvalidVal, match=r"pl\.range\(\): step must be an integer"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_accepts_runtime_step():
    @pl.jit(auto_mutex=False)
    def kernel(s: pl.DT_INT64, x: pl.Tensor[[64], pl.DT_FP32]):
        for _ in pl.range(0, 10, s):
            pl.system.bar_all()

    kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


# ---------------------------------------------------------------------------
# Loop-variable overflow on the final increment (start/stop/step combined)
# ---------------------------------------------------------------------------
def test_vf_rejects_final_step_overflow():
    @pl.vector_function
    def vf_body():
        for _ in pl.range(0, _UINT16_MAX, 10):
            pass

    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[64], pl.DT_FP32]):
        vf_body()

    with pytest.raises(OutOfRange, match=r"loop variable reaches 65540 on the final step"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_vf_accepts_when_final_step_stays_in_range():
    @pl.vector_function
    def vf_body():
        for _ in pl.range(0, _UINT16_MAX - 5, 10):
            pass

    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[64], pl.DT_FP32]):
        vf_body()

    kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_non_vf_rejects_final_step_overflow():
    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[64], pl.DT_FP32]):
        for _ in pl.range(0, INT64_MAX, INT64_MAX - 5):
            pl.system.bar_all()

    with pytest.raises(OutOfRange, match=r"loop variable reaches .* on the final step"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_empty_loop_does_not_trigger_overflow():
    @pl.vector_function
    def vf_body():
        for _ in pl.range(_UINT16_MAX, 0, 10):
            pass

    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[64], pl.DT_FP32]):
        vf_body()

    kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_runtime_bound_skips_overflow_check():
    @pl.jit(auto_mutex=False)
    def kernel(n: pl.DT_INT64, x: pl.Tensor[[64], pl.DT_FP32]):
        for _ in pl.range(0, n, 10):
            pl.system.bar_all()

    kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


# ---------------------------------------------------------------------------
# Legal integer bounds still parse, including runtime scalars and expressions
# ---------------------------------------------------------------------------
def test_accepts_integer_scalar_param():
    @pl.jit(auto_mutex=False)
    def kernel(n: pl.DT_INT64, x: pl.Tensor[[64], pl.DT_FP32]):
        for _ in pl.range(0, n):
            pl.system.bar_all()

    kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_accepts_integer_expression():
    @pl.jit(auto_mutex=False)
    def kernel(n: pl.DT_INT64, x: pl.Tensor[[64], pl.DT_FP32]):
        for _ in pl.range(n * 2 + 1):  # type: ignore[operator]
            pl.system.bar_all()

    kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


# ---------------------------------------------------------------------------
# The rejection is final and not retried as a plain Python expression
# ---------------------------------------------------------------------------
def test_rejection_is_final():
    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[64], pl.DT_FP32]):
        for _ in pl.range(1.5):
            pl.system.bar_all()

    with pytest.raises(InvalidVal):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


# ---------------------------------------------------------------------------
# Arity / kwargs errors keep their existing InvalidType behaviour
# ---------------------------------------------------------------------------
def test_no_args_still_rejected():
    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[64], pl.DT_FP32]):
        for _ in pl.range():  # type: ignore[call-arg]
            pl.system.bar_all()

    with pytest.raises(InvalidArgument, match=r"requires at least 1 argument"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_keyword_args_still_rejected():
    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[64], pl.DT_FP32]):
        for _ in pl.range(stop=10):  # type: ignore[call-arg]
            pl.system.bar_all()

    with pytest.raises(InvalidArgument, match=r"does not support keyword arguments"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
