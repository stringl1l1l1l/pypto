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
"""Cache/control-flow tests for shape-specialized PyPTO Pro JIT kernels."""

import importlib
import inspect

import pypto_pro.language as pl

jit_runtime = importlib.import_module("pypto_pro.runtime.jit")


def _private(obj, name):
    return getattr(obj, name)


def _bind_static_shapes(kernel, static_shapes):
    return _private(kernel, "_bind_static_shapes")(static_shapes)


def _static_signature_suffix(signature):
    return _private(jit_runtime, "_static_signature_suffix")(signature)


@pl.jit(arch="3510")
def mixed_policy_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.STATIC, 4], pl.DT_FP32],
):
    return


@pl.jit(arch="3510")
def ellipsis_policy_kernel(
    x: pl.Tensor[[pl.DYNAMIC, ...], pl.DT_FP32],
):
    return


@pl.jit(arch="3510")
def call_style_policy_kernel(
    x: pl.Tensor([pl.DYNAMIC, pl.STATIC, 4], pl.DT_FP32),
):
    return


def test_binary_static_shape_binding_uses_same_signature_model():
    bound = _bind_static_shapes(mixed_policy_kernel, {"x": [2, 7, 4]})

    assert bound.static_signature == ((0, 1, 7),)


def test_call_style_annotation_uses_same_shape_policy_model():
    bound = _bind_static_shapes(call_style_policy_kernel, {"x": [2, 7, 4]})

    assert bound.static_signature == ((0, 1, 7),)


def test_to_kernel_def_preserves_concrete_tilingkey_positional_api():
    concrete_key = {"Mode": 1}

    kernel_def = mixed_policy_kernel.to_kernel_def(concrete_key)

    assert getattr(kernel_def, "_tilingkey_consts") == concrete_key


def test_codegen_preserves_opc_tilingkey_arguments():
    parameters = inspect.signature(_private(jit_runtime, "_codegen")).parameters

    assert "tilingkey_packed" in parameters
    assert "out_dir" in parameters


def test_static_signature_produces_stable_distinct_artifact_suffixes():
    first = _static_signature_suffix(((0, 1, 7),))
    same = _static_signature_suffix(((0, 1, 7),))
    second = _static_signature_suffix(((0, 1, 9),))

    assert first == same
    assert first != second
    assert _static_signature_suffix(()) == ""


def test_bound_signature_is_consumed_by_ast_parameter_lowering():
    from pypto.pypto_impl import ir

    bound = _bind_static_shapes(mixed_policy_kernel, {"x": [2, 7, 4]})
    program = mixed_policy_kernel.to_kernel_def().parse_target_program(
        ir.SectionKind.Vector, bound_signature=bound
    )[0]
    func = program.functions["mixed_policy_kernel"]
    shape = func.params[0].type.shape

    assert isinstance(shape[0], ir.Var)
    assert shape[0].name == "__pypto_dyn_x_0"
    assert isinstance(shape[1], ir.ConstInt) and shape[1].value == 7
    assert isinstance(shape[2], ir.ConstInt) and shape[2].value == 4


def test_ellipsis_expands_to_static_tail_before_parsing():
    from pypto.pypto_impl import ir

    bound = _bind_static_shapes(ellipsis_policy_kernel, {"x": [2, 7, 9]})
    program = ellipsis_policy_kernel.to_kernel_def().parse_target_program(
        ir.SectionKind.Vector, bound_signature=bound
    )[0]
    shape = program.functions["ellipsis_policy_kernel"].params[0].type.shape

    assert isinstance(shape[0], ir.Var)
    assert [dim.value for dim in shape[1:]] == [7, 9]
    assert bound.static_signature == ((0, 1, 7), (0, 2, 9))
