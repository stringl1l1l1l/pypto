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
"""Unit tests for the declared-annotations extraction (``_direct_declared_annotations``).

Extracts author-pinned literal shapes/dtypes from a DIRECT ``@pypto.frontend.jit`` kernel's annotations
(reads the cached parsed signature). Building a real ``JitCallableWrapper`` needs the native ``pypto_impl``
on the import path; unlike ``test_from_jit_kernel.py`` there is no skip guard, so collection fails without
it. The helper reads only ``op._bare_kernel_fn``, so a plain namespace stands in for the ``ExportedCustomOp``.
"""
import types

import kernel_compile_samples as samples  # sibling fixture module (pytest puts this dir on sys.path)

from pypto.extensions.torch_custom_op_litenpu.common.authoring import _direct_declared_annotations


def _op(kernel):
    return types.SimpleNamespace(_bare_kernel_fn=kernel)


def test_extract_pinned_returns_full_entry_list():
    ann = _direct_declared_annotations(_op(samples.direct_add_kernel_pinned))
    # 3 params (input0, input1, output), each pinning shape [1,1,8,64] + DT_FP16
    assert ann == [
        {"shape": [1, 1, 8, 64], "dtype": "DT_FP16"},
        {"shape": [1, 1, 8, 64], "dtype": "DT_FP16"},
        {"shape": [1, 1, 8, 64], "dtype": "DT_FP16"},
    ]


def test_extract_erased_returns_none():
    # Fully-erased kernel (every param [...] with no dtype) -> None so erased demos stay byte-identical.
    assert _direct_declared_annotations(_op(samples.direct_add_kernel_erased)) is None


def test_extract_no_bare_kernel_returns_none():
    assert _direct_declared_annotations(_op(None)) is None
