#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software and can be redistributed and modified under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.huawei.com/
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
# PARTICULAR PURPOSE.
# See the License for the specific language governing permissions and limitations under the License.
# -----------------------------------------------------------------------------------------------------------
"""Arch identity and legacy-alias compatibility tests."""

import os
import warnings

from pypto_pro._errors import InvalidVal, NotSupported
from pypto_pro.runtime.platform import NpuArch, parse_arch
import pytest


def test_parse_arch_canonical_forms():
    assert parse_arch("3510") is NpuArch.DAV_3510
    assert parse_arch(" 3510 ") is NpuArch.DAV_3510
    assert parse_arch(3510) is NpuArch.DAV_3510
    assert parse_arch(NpuArch.DAV_3510) is NpuArch.DAV_3510


def test_npuarch_str_is_canonical_form():
    assert str(NpuArch.DAV_3510) == "3510"
    assert f"{NpuArch.DAV_3510}" == "3510"


def test_parse_arch_legacy_a5_maps_to_3510_with_deprecation():
    with pytest.warns(DeprecationWarning, match="'a5' is deprecated"):
        assert parse_arch("a5") is NpuArch.DAV_3510


def test_parse_arch_legacy_a2_a3_map_to_2201_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert parse_arch("a2") is NpuArch.DAV_2201
        assert parse_arch("a3") is NpuArch.DAV_2201


def test_parse_arch_unknown_raises():
    for bad in ("future", "a6", ""):
        with pytest.raises(InvalidVal):
            parse_arch(bad)


def test_setup_arch_env_accepts_deprecated_a5(monkeypatch):
    from pypto_pro.runtime.jit import _setup_arch_env

    monkeypatch.delenv("PYPTOPRO_JIT_ARCH", raising=False)
    with pytest.warns(DeprecationWarning):
        parsed = _setup_arch_env("a5")
    assert parsed is NpuArch.DAV_3510
    assert os.environ["PYPTOPRO_JIT_ARCH"] == "3510"


def test_setup_arch_env_canonicalises_env_value(monkeypatch):
    from pypto_pro.runtime.jit import _setup_arch_env, get_current_arch

    monkeypatch.delenv("PYPTOPRO_JIT_ARCH", raising=False)
    assert _setup_arch_env("3510") is NpuArch.DAV_3510
    assert get_current_arch() is NpuArch.DAV_3510


def test_setup_arch_env_rejects_non_3510():
    from pypto_pro.runtime.jit import _setup_arch_env

    with pytest.raises(NotSupported, match="only supports arch '3510'"):
        _setup_arch_env("a3")


def test_get_current_arch_env_a5_warns_and_canonicalises(monkeypatch):
    from pypto_pro.runtime.jit import get_current_arch

    monkeypatch.setenv("PYPTOPRO_JIT_ARCH", "a5")
    with pytest.warns(DeprecationWarning):
        assert get_current_arch() is NpuArch.DAV_3510


def test_jit_decorator_warns_on_deprecated_a5():
    import pypto_pro.language as pl

    with pytest.warns(DeprecationWarning):

        @pl.jit(arch="a5")
        def kernel(x):
            return x


def test_resolve_kernel_target_treats_a5_as_3510():
    from pypto_pro.runtime.compile_config import get_jit_compile_config

    cfg = get_jit_compile_config()
    with pytest.warns(DeprecationWarning):
        legacy = cfg.resolve_kernel_target("a5", has_cube=True, has_vector=True)
    canonical = cfg.resolve_kernel_target("3510", has_cube=True, has_vector=True)
    assert legacy == canonical


def test_platform_arch_is_enum():
    from pypto_pro.runtime.platform import get_platform_info

    arch = get_platform_info().arch
    assert arch is None or isinstance(arch, NpuArch)
