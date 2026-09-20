# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""IR snapshot for an Assemble write in a nested loop helper."""

from pathlib import Path

import pypto
from pypto import pil

from .test_common import check_snapshot, ssa_verify

IR = Path(__file__).with_suffix(".pypto")


def test_nested_assemble_loop_carry():
    def foo(src, out):
        for i in pypto.loop(2):
            def inner():
                out[i * 16:(i + 1) * 16, :] = src

            inner()

    src = pypto.Tensor([16, 16], pypto.DT_FP32, "src")
    out = pypto.Tensor([32, 16], pypto.DT_FP32, "out")
    func = pil.compile(foo, src, out, create_new_logical_tensor=True)

    ssa_verify(func, "nested_assemble_loop_carry")
    check_snapshot(func, IR)
