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
"""Public scalar aliases and core-side barriers have complete DSL contracts."""

from pypto_pro import ir
from pypto_pro._errors import NotSupported
import pypto_pro.language as pl
from pypto_pro.runtime.platform import NpuArch
import pytest


@pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
def cube_barriers(x: pl.Tensor[[1], pl.DT_FP32]):
    with pl.section_cube():
        pl.system.bar_m()
        pl.system.bar_mte1()
        pl.system.bar_fix()


def _vector_barrier(which):
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(x: pl.Tensor[[1], pl.DT_FP32]):
        with pl.section_vector():
            if which == 0:
                pl.system.bar_m()
            elif which == 1:
                pl.system.bar_mte1()
            else:
                pl.system.bar_fix()
    return kernel


def test_cube_barriers_remain_supported():
    assert cube_barriers.to_kernel_def().parse_target_program(ir.SectionKind.Cube)[1]


@pytest.mark.parametrize("which", [0, 1, 2])
def test_cube_only_barriers_reject_vector_target_with_location(which):
    with pytest.raises(NotSupported) as caught:
        _vector_barrier(which).to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    assert "requires a Cube section" in caught.value.message
    assert caught.value.span["filename"] == __file__
