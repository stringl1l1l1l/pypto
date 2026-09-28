#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software and can be redistributed and modified under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Arch input parsing over the pybind-bound ``NpuArch`` enum (the C++ ``NPUArch`` in tilefwk/platform.h)."""

from __future__ import annotations

import warnings

from pypto.pypto_impl import NpuArch, ParseNPUArch

from ._errors import InvalidVal

__all__ = ["NpuArch", "parse_arch"]

_DEPRECATED_TO_CANONICAL = {"a5": "3510"}


def parse_arch(value: str) -> NpuArch:
    """Resolve an arch name: the __NPU_ARCH__ version number ("3510") or a legacy name.

    Deprecated names emit a DeprecationWarning and resolve to their canonical
    replacement. Name resolution itself is the bound C++ ParseNPUArch -- the
    same implementation the C++ codegen entry uses -- so Python and C++ accept
    the same names.
    """
    text = value.strip().lower()
    canonical = _DEPRECATED_TO_CANONICAL.get(text)
    if canonical is not None:
        warnings.warn(f"arch {value!r} is deprecated; use arch {canonical!r}", DeprecationWarning, stacklevel=2)
        text = canonical
    arch = ParseNPUArch(text)
    if arch == NpuArch.DAV_UNKNOWN:
        raise InvalidVal(f"unknown arch {value!r}; expected an __NPU_ARCH__ version name such as '3510'")
    return arch
