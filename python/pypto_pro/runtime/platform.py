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
"""Platform information utilities for pypto_pro.runtime.

Provides runtime access to NPU hardware information (SOC version, core count).
Uses pypto_impl binding for arch detection and core count query.

Usage::

    from pypto_pro.runtime.platform import get_platform_info

    info = get_platform_info()
    print(info.soc_version)  # e.g. "DAV_2201"
    print(info.core_num)     # e.g. 20

    # Use in kernel launch:
    kernel[None, info.core_num](q, k, v, o)
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# NPUArch string → compilation arch mapping
_ARCH_MAP = {
    "DAV_1001": "a3",  # 910
    "DAV_2201": "a3",  # 910B/910C
    "DAV_3510": "3510",  # 950
}
# Canonical arch -> legacy string still expected by the C++ side (CCECodegen / GetMemoryLimitForArch).
_ARCH_TO_CPP_ARCH = {"3510": "a5"}


@dataclass
class PlatformInfo:
    """NPU platform hardware information."""

    soc_version: str = ""
    core_num: int = 0
    # On architectures where cube (AIC) and vector (AIV) cores are counted
    # separately, these hold the per-type core counts. They fall back to
    # ``core_num`` when the finer-grained query is unavailable.
    cube_core_num: int = 0
    vector_core_num: int = 0

    @property
    def arch(self) -> str:
        """Infer compilation arch from SOC version string.

        Returns:
            "3510" for DAV_3510 (950 series), "a3" for DAV_2201/DAV_1001, "" if unknown.
        """
        if not self.soc_version:
            return ""
        return _ARCH_MAP.get(self.soc_version, "a3")


_cached_info: Optional[PlatformInfo] = None


def _get_npu_arch() -> str:
    """Get NPU architecture string via pypto_impl binding.

    Returns:
        NPUArch string like "DAV_2201", "DAV_3510", or "" if unavailable.
    """
    try:
        from pypto import pypto_impl

        return pypto_impl.GetNPUArch()
    except (ImportError, AttributeError, RuntimeError) as e:
        logger.debug("pypto_impl.GetNPUArch() not available: %s", e)
        return ""


def _get_ai_core_num() -> int:
    """Get AI Core count via pypto_impl binding.

    Returns:
        Number of AI Cores, or 0 if unavailable.
    """
    try:
        from pypto import pypto_impl

        return pypto_impl.GetAICoreNum()
    except (ImportError, AttributeError, RuntimeError) as e:
        logger.debug("pypto_impl.GetAICoreNum() not available: %s", e)
        return 0


def _get_aic_core_num() -> int:
    """Get cube (AIC) core count via pypto_impl binding.

    Returns:
        Number of cube cores, or 0 if unavailable.
    """
    try:
        from pypto import pypto_impl

        return pypto_impl.GetAICCoreNum()
    except (ImportError, AttributeError, RuntimeError) as e:
        logger.debug("pypto_impl.GetAICCoreNum() not available: %s", e)
        return 0


def _get_aiv_core_num() -> int:
    """Get vector (AIV) core count via pypto_impl binding.

    Returns:
        Number of vector cores, or 0 if unavailable.
    """
    try:
        from pypto import pypto_impl

        return pypto_impl.GetAIVCoreNum()
    except (ImportError, AttributeError, RuntimeError) as e:
        logger.debug("pypto_impl.GetAIVCoreNum() not available: %s", e)
        return 0


def get_memory_limit(arch: str, memory_space: str) -> int:
    """Query on-chip buffer capacity (bytes) for the target arch.

    Args:
        arch: Compilation arch ("3510", from PYPTOPRO_JIT_ARCH).
        memory_space: pypto_pro MemorySpace name (e.g. "Vec"/"Mat"/"Left"/
              "Right"/"Acc") of the buffer to query.

    Returns:
        Buffer capacity in bytes, or 0 if unavailable.
    """
    try:
        from pypto import pypto_impl

        return pypto_impl.GetMemoryLimitForArch(_ARCH_TO_CPP_ARCH.get(arch, arch), str(memory_space))
    except (ImportError, AttributeError, RuntimeError) as e:
        logger.debug("pypto_impl memory-limit query not available: %s", e)
        return 0


def get_platform_info(force_refresh: bool = False) -> PlatformInfo:
    """Get NPU platform information.

    Uses pypto_impl binding for both arch detection and core count query.
    Results are cached after the first call.

    Args:
        force_refresh: If True, re-query the hardware (ignore cache).

    Returns:
        PlatformInfo dataclass with hardware details.
    """
    global _cached_info
    if _cached_info is not None and not force_refresh:
        return _cached_info

    info = PlatformInfo()

    info.soc_version = _get_npu_arch()
    info.core_num = _get_ai_core_num()
    # Fall back to core_num when the per-type query is unavailable so callers
    # can rely on these being non-zero whenever core_num is.
    info.cube_core_num = _get_aic_core_num() or info.core_num
    info.vector_core_num = _get_aiv_core_num() or info.core_num

    _cached_info = info

    if info.soc_version:
        logger.info(
            "Platform: %s (arch=%s), core_num=%d, cube_core_num=%d, vector_core_num=%d",
            info.soc_version,
            info.arch,
            info.core_num,
            info.cube_core_num,
            info.vector_core_num,
        )
    else:
        logger.debug("Platform info not available (pypto_impl not loaded)")

    return info
