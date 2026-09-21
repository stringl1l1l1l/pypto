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
"""Unit tests for the factory form of ``CompileEntry``.

The compile module's ``torch``/``pypto`` attributes are monkeypatched with fakes, so the contract logic
runs without the NPU toolchain; importing the module under test still requires pypto on the path.
"""
import importlib
import types

import pytest

from pypto.extensions.torch_custom_op_litenpu.common.compile import CompileEntry, _as_tuple_of

_cop = importlib.import_module("pypto.extensions.torch_custom_op_litenpu.common.compile")  # to patch its seams


class _FakeTensor:
    def __init__(self, shape, dtype):
        self.shape, self.dtype = tuple(shape), dtype

    def __repr__(self):
        return f"T{self.shape}:{self.dtype}"


@pytest.fixture
def fake_backend(monkeypatch):
    """Patch the compile module's torch/pypto seams with toolchain-free fakes."""
    fake_torch = types.SimpleNamespace(
        float16="torch.float16", float32="torch.float32", int32="torch.int32",
        bfloat16="torch.bfloat16",
        empty=lambda shape, dtype, device="cpu": _FakeTensor(shape, dtype),
    )
    monkeypatch.setattr(_cop, "torch", fake_torch)
    monkeypatch.setattr(_cop, "_pto_dtype", lambda td: f"pto:{td}")
    return fake_torch


def test_as_tuple_of():
    # single vs N-tuple disambiguation, incl. the 0-d output case
    assert _as_tuple_of((8, 64), 1) == ((8, 64),)
    assert _as_tuple_of((), 1) == ((),)
    assert _as_tuple_of("torch.float16", 1) == ("torch.float16",)
    assert _as_tuple_of(((8, 64), (8,)), 2) == ((8, 64), (8,))
    assert _as_tuple_of(("a", "b"), 2) == ("a", "b")
    with pytest.raises(ValueError):
        _as_tuple_of(((8, 64),), 2)  # wrong count
    with pytest.raises(ValueError):
        _as_tuple_of((8, 64), 2)  # a single flat shape must not be split into 2 scalar "shapes"


def test_from_factory_basic_and_caching(fake_backend):
    calls = {"factory": [], "build": []}

    def fake_create_add_kernel(shape, dtype, soc_version):
        calls["factory"].append((tuple(shape), dtype, soc_version))

        class _Jit:
            def build_kernel(self, *tensors):
                calls["build"].append([(t.shape, t.dtype) for t in tensors])
                return f"/tmp/kernel_{tuple(shape)}.o"
        return _Jit()

    add_op_compile = CompileEntry(factory=fake_create_add_kernel, num_inputs=2)

    p1 = add_op_compile(((8, 64), (8, 64)), ("float16", "float16"), {}, "Kirin9030")
    assert p1 == "/tmp/kernel_(8, 64).o", p1
    assert calls["factory"] == [((8, 64), "pto:torch.float16", "Kirin9030")], calls["factory"]
    # 2 inputs + 1 inferred output
    assert calls["build"][-1] == [((8, 64), "torch.float16")] * 3, calls["build"][-1]

    # an identical key hits the cache; a different shape recompiles
    add_op_compile(((8, 64), (8, 64)), ("float16", "float16"), {}, "Kirin9030")
    assert len(calls["factory"]) == 1, calls["factory"]
    add_op_compile(((256, 64), (256, 64)), ("float16", "float16"), {}, "Kirin9030")
    assert len(calls["factory"]) == 2, calls["factory"]

    # single-input op; soc_version is required (the host always supplies it)
    op1 = CompileEntry(
        factory=lambda s, d, v: types.SimpleNamespace(build_kernel=lambda *t: f"/tmp/one_{tuple(s)}.o"),
        num_inputs=1)
    assert op1(((8, 64),), ("float32",), {}, "Kirin9030").endswith("one_(8, 64).o")
    with pytest.raises(ValueError, match="soc_version"):
        op1(((8, 64),), ("float32",), {}, None)


def test_matmul_lists_factory(fake_backend):
    # matmul: different input shapes, heterogeneous input dtypes, out shape != in[0]
    mm = {}

    def fake_mm_factory(shapes, dtypes, soc):
        mm["factory"] = ([tuple(s) for s in shapes], list(dtypes), soc)

        def build_kernel(*tensors):
            mm["build"] = [(t.shape, t.dtype) for t in tensors]
            return "/tmp/mm.o"
        return types.SimpleNamespace(build_kernel=build_kernel)

    matmul_op = CompileEntry(factory=fake_mm_factory, num_inputs=2, factory_signature="lists")

    def _mm_infer_shape(a_shape, b_shape):
        return (a_shape[0], b_shape[1])                 # C[M,N] = A[M,K] @ B[K,N]

    matmul_op.infer_shape = _mm_infer_shape

    pmm = matmul_op(((8, 16), (16, 32)), ("float16", "int32"), {}, "Kirin9030")
    assert pmm == "/tmp/mm.o"
    assert mm["factory"] == ([(8, 16), (16, 32)],
                             ["pto:torch.float16", "pto:torch.int32"], "Kirin9030"), mm["factory"]
    # A(8,16) fp16 + B(16,32) int32 inputs, then the hook-inferred C(8,32) output (dtype of in[0])
    assert mm["build"] == [((8, 16), "torch.float16"), ((16, 32), "torch.int32"),
                           ((8, 32), "torch.float16")], mm["build"]


def test_multi_output_mixed_dtypes(fake_backend):
    # norm -> (y, rstd), mixed output dtypes (y=fp16, rstd=fp32)
    nm = {}

    def fake_norm_factory(shapes, dtypes, soc):
        def build_kernel(*tensors):
            nm["build"] = [(t.shape, t.dtype) for t in tensors]
            return "/tmp/norm.o"
        return types.SimpleNamespace(build_kernel=build_kernel)

    norm_op = CompileEntry(
        factory=fake_norm_factory, num_inputs=2, num_outputs=2, factory_signature="lists")

    def _norm_infer_shape(x_shape, w_shape):
        return (x_shape, (x_shape[0],))                 # y like x; rstd is per-row

    norm_op.infer_shape = _norm_infer_shape

    def _norm_infer_dtype(x_dtype, w_dtype):
        return (x_dtype, fake_backend.float32)          # y keeps dtype; rstd is fp32

    norm_op.infer_dtype = _norm_infer_dtype

    pn = norm_op(((8, 64), (64,)), ("float16", "float16"), {}, "Kirin9030")
    assert pn == "/tmp/norm.o"
    # inputs x(8,64) w(64,); outputs y(8,64) fp16, rstd(8,) fp32
    assert nm["build"] == [((8, 64), "torch.float16"), ((64,), "torch.float16"),
                           ((8, 64), "torch.float16"), ((8,), "torch.float32")], nm["build"]


def test_full_factory_attrs(fake_backend):
    # factory_signature="full": attrs forwarded to the factory (e.g. rmsnorm eps)
    fl = {}

    def fake_full_factory(shapes, dtypes, attrs, soc):
        fl["call"] = ([tuple(s) for s in shapes], list(dtypes), dict(attrs), soc)
        return types.SimpleNamespace(build_kernel=lambda *t: "/tmp/full.o")

    full_op = CompileEntry(factory=fake_full_factory, num_inputs=2, factory_signature="full")
    pf = full_op(((8, 64), (64,)), ("float16", "float16"), {"eps": "1e-6"}, "Kirin9030")
    assert pf == "/tmp/full.o"
    assert fl["call"] == ([(8, 64), (64,)], ["pto:torch.float16", "pto:torch.float16"],
                          {"eps": "1e-6"}, "Kirin9030"), fl["call"]


@pytest.mark.parametrize("hook", ["infer_shape", "infer_dtype"])
def test_infer_hook_setters_reject_non_callables(hook):
    """The hooks are assigned by name (``entry.infer_shape = fn``), so a typo'd non-callable would
    otherwise only surface deep inside dispatch. ``None`` stays valid: a hook-less op has none."""
    entry = CompileEntry(factory=lambda shape, dtype, soc: None, num_inputs=1)

    with pytest.raises(TypeError, match=f"{hook} must be callable or None, got int"):
        setattr(entry, hook, 42)

    setattr(entry, hook, None)
    assert getattr(entry, hook) is None

    def _hook(a):
        return a

    setattr(entry, hook, _hook)
    assert getattr(entry, hook) is _hook
