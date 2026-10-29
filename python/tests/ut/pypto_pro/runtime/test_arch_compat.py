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
from pypto_pro.runtime.platform import NpuArch, normalize_arch
import pytest


def test_parse_arch_rejects_numeric_strings():
    for bad in ("3510", "DAV_3510"):
        with pytest.raises(InvalidVal, match="expect a valid NpuArch value"):
            normalize_arch(bad)


def test_parse_arch_rejects_non_str_non_enum_inputs():
    for bad in (None, 3510):
        with pytest.raises(InvalidVal, match="expected str or NpuArch"):
            normalize_arch(bad)


def test_parse_arch_accepts_enum_values():
    members = (
        NpuArch.DAV_3510,
        NpuArch.DAV_2201,
        NpuArch.DAV_1001,
        NpuArch.DAV_3003,
        NpuArch.DAV_3113,
        NpuArch.DAV_UNKNOWN,
    )
    for member in members:
        assert normalize_arch(member) == member


def test_npuarch_construction_from_value_is_forbidden():
    for value in (3510, "DAV_3510"):
        with pytest.raises(TypeError, match="cannot be constructed from a value"):
            NpuArch(value)


def test_npuarch_name_is_canonical_form():
    assert NpuArch.DAV_3510.name == "DAV_3510"
    assert NpuArch.DAV_2201.name == "DAV_2201"


def test_npuarch_str_is_numeric_for_artifact_paths():
    assert str(NpuArch.DAV_3510) == "3510"
    assert f"{NpuArch.DAV_3510}" == "3510"


def test_parse_arch_legacy_names_warn_and_resolve():
    with pytest.warns(DeprecationWarning, match="'a5' is deprecated"):
        assert normalize_arch("a5") == NpuArch.DAV_3510
    with pytest.warns(DeprecationWarning):
        assert normalize_arch(" A5 ") == NpuArch.DAV_3510
        assert normalize_arch("a2") == NpuArch.DAV_2201
        assert normalize_arch("a3") == NpuArch.DAV_2201


def test_parse_arch_unknown_raises():
    for bad in ("future", "a6"):
        with pytest.raises(InvalidVal):
            normalize_arch(bad)


def test_jit_rejects_numeric_string_arch():
    import pypto_pro.language as pl

    with pytest.raises(InvalidVal, match="expect a valid NpuArch value"):
        pl.jit(arch="3510")


def test_setup_arch_env_accepts_deprecated_a5(monkeypatch):
    from pypto_pro.runtime.jit import _setup_arch_env

    monkeypatch.delenv("PYPTOPRO_JIT_ARCH", raising=False)
    with pytest.warns(DeprecationWarning):
        parsed = normalize_arch("a5")
    assert _setup_arch_env(parsed) == NpuArch.DAV_3510
    assert os.environ["PYPTOPRO_JIT_ARCH"] == NpuArch.DAV_3510.name


def test_setup_arch_env_canonicalises_env_value(monkeypatch):
    from pypto_pro.runtime.jit import _setup_arch_env, get_current_arch

    monkeypatch.delenv("PYPTOPRO_JIT_ARCH", raising=False)
    assert _setup_arch_env(NpuArch.DAV_3510) == NpuArch.DAV_3510
    assert os.environ["PYPTOPRO_JIT_ARCH"] == NpuArch.DAV_3510.name
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert get_current_arch() == NpuArch.DAV_3510


def test_setup_arch_env_rejects_non_3510():
    from pypto_pro.runtime.jit import _setup_arch_env

    with pytest.raises(NotSupported, match="only supports arch '3510'"):
        _setup_arch_env(NpuArch.DAV_2201)


def test_get_current_arch_env_a5_warns_and_canonicalises(monkeypatch):
    from pypto_pro.runtime.jit import get_current_arch

    monkeypatch.setenv("PYPTOPRO_JIT_ARCH", "a5")
    with pytest.warns(DeprecationWarning):
        assert get_current_arch() == NpuArch.DAV_3510


def test_jit_decorator_warns_on_deprecated_a5():
    import pypto_pro.language as pl

    with pytest.warns(DeprecationWarning):

        @pl.jit(arch="a5")
        def kernel(x):
            return x


def test_platform_arch_is_enum():
    from pypto_pro.runtime.platform import get_platform_info

    arch = get_platform_info().arch
    assert arch is None or isinstance(arch, NpuArch)
