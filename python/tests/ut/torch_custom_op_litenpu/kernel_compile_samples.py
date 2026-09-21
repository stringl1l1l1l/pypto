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
"""DIRECT (``@pypto.frontend.jit``) kernel samples whose parameter annotations drive the
declared-annotations extraction: one pinning a literal shape + dtype on every param, one fully erased.

They live in a real module so ``inspect.getsource`` and pypto's jit parser both see real source on
disk. The module is a leaf: it references only ``pypto``, so importing it needs no fixture package.
"""

import pypto


# Pinned DIRECT kernel: literal shape + dtype on every param; drives the declared-annotations extraction.
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def direct_add_kernel_pinned(
    input0: pypto.Tensor([1, 1, 8, 64], pypto.DT_FP16),
    input1: pypto.Tensor([1, 1, 8, 64], pypto.DT_FP16),
    output: pypto.Tensor([1, 1, 8, 64], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    output.move(input0 + input1)


# Erased (pinned-free) DIRECT kernel referencing only pypto: no pinned constant, no helper def to pack.
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def direct_add_kernel_erased(
    input0: pypto.Tensor([...]),
    input1: pypto.Tensor([...]),
    output: pypto.Tensor([...]),
):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    output.move(input0 + input1)
