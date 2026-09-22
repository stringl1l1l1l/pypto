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
"""Smoke tests for automatic span capture in IR operation helpers."""

import inspect

from pypto_pro import DataType, ir
from pypto_pro.ir._utils import _get_span_or_capture
from pypto_pro.ir.op import ptr as ptr_ops


def _current_line():
    frame = inspect.currentframe()
    if frame and frame.f_back:
        return frame.f_back.f_lineno
    return -1


def _tensor_var(name: str):
    return ir.Var(name, ir.TensorType([64], DataType.FP32), ir.Span.unknown())


def test_tensor_op_helper_captures_caller_span():
    x = _tensor_var("x")

    line_before = _current_line()
    result = ptr_ops.make_tensor(x, [64], [1])

    assert result.span.filename.endswith("test_operation_span_capture.py")
    assert result.span.is_valid()
    assert result.span.begin_line == line_before + 1


def test_explicit_span_overrides_auto_capture():
    x = _tensor_var("x")
    explicit = ir.Span("custom.py", 100, 20)

    result = ptr_ops.make_tensor(x, [64], [1], span=explicit)

    assert result.span.filename == "custom.py"
    assert result.span.begin_line == 100
    assert result.span.begin_column == 20


def test_get_span_or_capture_respects_frame_offset():
    def wrapper():
        return _get_span_or_capture(frame_offset=1)

    line_before = _current_line()
    result = wrapper()

    assert result.filename.endswith("test_operation_span_capture.py")
    assert result.is_valid()
    assert result.begin_line == line_before + 1
