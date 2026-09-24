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
"""Compilation arch identity: NpuArch enum and its parsing (kept dependency-free; see pypto_pro._errors)."""

from __future__ import annotations

from enum import IntEnum
import warnings

from ._errors import InvalidVal


class NpuArch(IntEnum):
    """Compilation arch identity, mirroring the C++ ``NPUArch`` enum in tilefwk/platform.h."""

    DAV_1001 = 1001  # 910
    DAV_2201 = 2201  # 910B/910C (Atlas A2/A3)
    DAV_3510 = 3510  # Ascend 950PR/950DT
    DAV_3003 = 3003
    DAV_3113 = 3113

    def __str__(self) -> str:
        return str(self.value)


# Legacy internal arch names kept as accepted inputs for compatibility.
_ARCH_INPUT_ALIASES = {
    "a5": NpuArch.DAV_3510,
    "a2": NpuArch.DAV_2201,
    "a3": NpuArch.DAV_2201,
}
_DEPRECATED_ARCH_INPUTS = {"a5"}


def parse_arch(value: NpuArch | int | str) -> NpuArch:
    """Resolve an arch value: a NpuArch member, its __NPU_ARCH__ number, or a legacy name.

    The legacy name "a5" resolves to DAV_3510 with a DeprecationWarning.
    """
    if isinstance(value, NpuArch):
        return value
    text = str(value).strip().lower()
    if text in _DEPRECATED_ARCH_INPUTS:
        warnings.warn(f"arch {value!r} is deprecated; use arch '3510'", DeprecationWarning, stacklevel=2)
    if text in _ARCH_INPUT_ALIASES:
        return _ARCH_INPUT_ALIASES[text]
    try:
        return NpuArch(int(text))
    except ValueError:
        raise InvalidVal(f"unknown arch {value!r}; expected an __NPU_ARCH__ version such as '3510'") from None
