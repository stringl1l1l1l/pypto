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
"""Experimental auto mix partition option.

Use auto_mix_partition to configure the auto CV Mix merging switch of the
ReduceCopyMerge pass, which is not yet stable enough for pypto.set_pass_options.
"""

from ..config import set_options


def auto_mix_partition(value: int):
    """
    Set the auto mix partition option.

    Parameters
    ---------
    value : int
        Experimental. Controls the automatic CV Mix subgraph merging in the
        ReduceCopyMerge pass. 0 disables auto CV Mix graph merging; 1 enables
        it with Cube ops <= 2000 and Vector ops <= 2240 per merged subgraph.
        Defaults to 0. This is an experimental feature that may change or be
        removed in future releases.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise ValueError(f"Invalid auto_mix_partition: '{value}'. Expected 0 (disable) or 1 (enable).")
    set_options(pass_options={'auto_mix_partition': value})
