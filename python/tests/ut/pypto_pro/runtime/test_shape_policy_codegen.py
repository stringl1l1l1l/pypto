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
"""Shape-policy tests that cross the JIT and codegen preparation path.

The cases verify STATIC dimensions are baked into parsed IR, DYNAMIC dimensions
reuse one compiled variant, and STATIC signature changes select distinct cache
entries and artifact suffixes.
"""

import importlib

import pypto_pro.language as pl
import torch

jit_runtime = importlib.import_module("pypto_pro.runtime.jit")


def _private(obj, name):
    return getattr(obj, name)


def _ensure_compiled(kernel, *args, **kwargs):
    return _private(kernel, "_ensure_compiled")(*args, **kwargs)


def _bind_static_shapes(kernel, static_shapes):
    return _private(kernel, "_bind_static_shapes")(static_shapes)


def _compiled_cache(kernel):
    return _private(kernel, "_compiled_by_signature")


def _static_signature_suffix(signature):
    return _private(jit_runtime, "_static_signature_suffix")(signature)


@pl.jit(arch="3510")
def static_dynamic_kernel(
    x: pl.Tensor[[pl.STATIC, pl.DYNAMIC], pl.DT_FP16],
):
    return


@pl.jit(arch="3510")
def dynamic_dynamic_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    return


def _reset_caches(kernel):
    _compiled_cache(kernel).clear()


def _install_fake_build(monkeypatch, kernel):
    _reset_caches(kernel)
    monkeypatch.setattr(jit_runtime, "_build_jit_so", lambda *a, **kw: "/fake/path.so")


def test_static_specialization_produces_different_artifacts():
    """Different STATIC values bake distinct ConstInt dimensions into the IR.

    This is the core guarantee that STATIC specialization is not a no-op: the
    STATIC axis is lowered as a ConstInt in the tensor type shape, so each
    distinct value produces a distinct IR variant feeding codegen. A kernel body
    that does not reference the STATIC axis yields identical C++ text, but the
    IR-level difference is what lets the CCE compiler fold the constant when the
    body does use it.
    """
    from pypto.pypto_impl import ir

    bound_m8 = _bind_static_shapes(static_dynamic_kernel, {"x": [8, 64]})
    bound_m16 = _bind_static_shapes(static_dynamic_kernel, {"x": [16, 64]})
    prog_m8 = static_dynamic_kernel.to_kernel_def().parse_target_program(
        ir.SectionKind.Vector, bound_signature=bound_m8
    )[0]
    prog_m16 = static_dynamic_kernel.to_kernel_def().parse_target_program(
        ir.SectionKind.Vector, bound_signature=bound_m16
    )[0]
    shape_m8 = prog_m8.functions["static_dynamic_kernel"].params[0].type.shape
    shape_m16 = prog_m16.functions["static_dynamic_kernel"].params[0].type.shape

    assert isinstance(shape_m8[0], ir.ConstInt) and shape_m8[0].value == 8, (
        "STATIC value 8 was not baked into the IR as a ConstInt"
    )
    assert isinstance(shape_m16[0], ir.ConstInt) and shape_m16[0].value == 16, (
        "STATIC value 16 was not baked into the IR as a ConstInt"
    )
    assert shape_m8[0].value != shape_m16[0].value, (
        "STATIC specialization did not change the IR tensor shape; "
        "the runtime value was not baked into the variant"
    )


def test_dynamic_only_reuses_single_compiled_variant_across_shape_changes(monkeypatch):
    """A fully-DYNAMIC kernel must produce exactly one compiled variant for any shapes.

    DYNAMIC dimensions are excluded from static_signature, so the compiled cache key is
    constant () across all runtime sizes. This verifies the 'one .so serves all sizes'
    contract at the compiled-cache layer.
    """
    _install_fake_build(monkeypatch, dynamic_dynamic_kernel)

    _ensure_compiled(dynamic_dynamic_kernel, (torch.empty((8, 64), dtype=torch.float16),))
    _ensure_compiled(dynamic_dynamic_kernel, (torch.empty((16, 128), dtype=torch.float16),))
    _ensure_compiled(dynamic_dynamic_kernel, (torch.empty((1024, 4096), dtype=torch.float16),))

    assert len(_compiled_cache(dynamic_dynamic_kernel)) == 1, (
        "DYNAMIC-only kernel produced more than one compiled variant; "
        "shape changes should not trigger re-compile"
    )


def test_static_axis_change_triggers_new_variant_but_stability_reuses(monkeypatch):
    """STATIC axis changes produce a new variant; unchanged STATIC reuses the cached one."""
    _install_fake_build(monkeypatch, static_dynamic_kernel)

    _ensure_compiled(static_dynamic_kernel, (torch.empty((8, 64), dtype=torch.float16),))
    assert len(_compiled_cache(static_dynamic_kernel)) == 1

    # DYNAMIC axis changes, STATIC axis unchanged -> must reuse variant A
    _ensure_compiled(static_dynamic_kernel, (torch.empty((8, 4096), dtype=torch.float16),))
    assert len(_compiled_cache(static_dynamic_kernel)) == 1, (
        "DYNAMIC axis change triggered a new compiled variant; only STATIC axes should key the cache"
    )

    # STATIC axis changes (8 -> 16) -> must produce variant B
    _ensure_compiled(static_dynamic_kernel, (torch.empty((16, 64), dtype=torch.float16),))
    assert len(_compiled_cache(static_dynamic_kernel)) == 2, (
        "STATIC axis change did not trigger a new compiled variant"
    )

    # Back to STATIC=8 -> must reuse variant A, not produce a third variant
    _ensure_compiled(static_dynamic_kernel, (torch.empty((8, 64), dtype=torch.float16),))
    assert len(_compiled_cache(static_dynamic_kernel)) == 2, (
        "Revisiting a prior STATIC value created a duplicate variant instead of reusing the cache"
    )


def test_static_signature_keys_build_dir_suffix():
    """Distinct STATIC signatures must produce distinct build directories.

    The build-dir suffix (_static_signature_suffix) is what keeps two specialized
    kernel.cpp files from clobbering each other on disk. If two different signatures
    ever mapped to the same dir, one variant would overwrite the other.
    """
    sig_a = ((0, 0, 8),)
    sig_b = ((0, 0, 16),)

    suffix_a = _static_signature_suffix(sig_a)
    suffix_b = _static_signature_suffix(sig_b)

    assert suffix_a != suffix_b, "Distinct STATIC signatures share a build-dir suffix"
    assert suffix_a != "", "Non-empty static signature produced an empty suffix"
    assert _static_signature_suffix(()) == "", (
        "Empty static signature (DYNAMIC-only) should produce no suffix"
    )
