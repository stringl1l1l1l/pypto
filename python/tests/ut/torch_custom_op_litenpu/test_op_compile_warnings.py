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
"""Unit tests for the declared-annotation guard in ``CompileEntry``: a shape mismatch warns, a dtype
mismatch raises. The compile module's ``torch``/``pypto`` attributes are monkeypatched with fakes.
"""
import importlib
import types
import warnings

import pytest

from pypto.extensions.torch_custom_op_litenpu.common.compile import CompileEntry

_cop = importlib.import_module("pypto.extensions.torch_custom_op_litenpu.common.compile")


def _install_mocks(monkeypatch):
    class _T:
        def __init__(self, shape, dtype):
            self.shape, self.dtype = tuple(shape), dtype

    fake_torch = types.SimpleNamespace(
        float16="torch.float16", int32="torch.int32",
        empty=lambda shape, dtype, device="cpu": _T(shape, dtype),
    )

    # Stand-in DataType enum: like the real pypto.DataType it stringifies to its int value, not to the
    # DT_ token, so the guard has to compare the enum objects rather than their string forms.
    class _DT:
        def __init__(self, token, val):
            self.name, self._val = token, val

        def __repr__(self):
            return str(self._val)
    fp16, int32 = _DT("DT_FP16", 6), _DT("DT_INT32", 3)
    pto_by_torch_dtype = {"torch.float16": fp16, "torch.int32": int32}
    fake_pypto = types.SimpleNamespace(
        DT_FP16=fp16, DT_INT32=int32,  # top-level DT_* alias -> same obj
        set_codegen_options=lambda **_k: None,  # a DIRECT jit_kernel pins soc via this global
    )
    monkeypatch.setattr(_cop, "torch", fake_torch)
    monkeypatch.setattr(_cop, "_pto_dtype", lambda td: pto_by_torch_dtype[td])
    monkeypatch.setattr(_cop, "pypto", fake_pypto)


def _fake_factory(*_a, **_k):
    class _Jit:
        def build_kernel(self, *tensors):
            return "/tmp/k.o"
    return _Jit()


def _entry(declared):
    # only the constructor carries declared_annotations on the factory form
    return CompileEntry(
        factory=_fake_factory, num_inputs=2, num_outputs=1,
        factory_signature="lists", declared_annotations=declared,
    )


def test_no_warning_when_declared_matches(monkeypatch):
    _install_mocks(monkeypatch)
    entry = _entry([
        {"shape": [8, 64], "dtype": "DT_FP16"},
        {"shape": [8, 64], "dtype": "DT_FP16"},
        {"shape": None, "dtype": None},
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning -> failure
        entry(((8, 64), (8, 64)), ("float16", "float16"), {}, "Kirin9030")


def test_shape_mismatch_warns(monkeypatch):
    _install_mocks(monkeypatch)
    entry = _entry([
        {"shape": [8, 64], "dtype": None},
        {"shape": None, "dtype": None},
        {"shape": None, "dtype": None},
    ])
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        entry(((16, 64), (8, 64)), ("float16", "float16"), {}, "Kirin9030")
    msgs = [str(w.message) for w in rec]
    assert any("input[0] shape [8, 64] != runtime [16, 64]" in m for m in msgs), msgs


def test_dtype_mismatch_raises(monkeypatch):
    _install_mocks(monkeypatch)
    # input0 declared int32, fed float16 -> raise (a pinned dtype would silently override the runtime one)
    entry = _entry([
        {"shape": None, "dtype": "DT_INT32"},
        {"shape": None, "dtype": None},
        {"shape": None, "dtype": None},
    ])
    with pytest.raises(ValueError, match=r"input\[0\] dtype DT_INT32"):
        entry(((8, 64), (8, 64)), ("float16", "float16"), {}, "Kirin9030")


def test_none_declared_skips_checks(monkeypatch):
    _install_mocks(monkeypatch)
    entry = CompileEntry(
        factory=_fake_factory, num_inputs=2, num_outputs=1, factory_signature="lists",
    )  # no declared_annotations at all
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        entry(((8, 64), (8, 64)), ("float16", "float16"), {}, "Kirin9030")


class _FakeJit:
    __name__ = "attention_kernel"

    def build_kernel(self, *tensors):
        return "/tmp/k.o"


def _jit_entry(declared):
    return CompileEntry(
        jit_kernel=_FakeJit(), num_inputs=3, num_outputs=1, declared_annotations=declared,
    )


# 4 pinned entries (q, k, v, output) mirroring the attention demo
_ATTN_DECL = [{"shape": [1, 1, 8, 64], "dtype": "DT_FP16"}] * 4
_ATTN_DTYPES = ("float16", "float16", "float16")


def test_jit_kernel_shape_mismatch_warns(monkeypatch):
    # DIRECT (jit_kernel=) entries run the same guard as the factory form
    _install_mocks(monkeypatch)
    entry = _jit_entry(_ATTN_DECL)
    with pytest.warns(UserWarning, match=r"input\[0\] shape \[1, 1, 8, 64\] != runtime \[2, 1, 8, 64\]"):
        entry(((2, 1, 8, 64), (1, 1, 8, 64), (1, 1, 8, 64)), _ATTN_DTYPES, {}, "Kirin9030")
