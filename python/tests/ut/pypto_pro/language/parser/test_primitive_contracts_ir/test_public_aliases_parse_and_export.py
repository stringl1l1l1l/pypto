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
"""Extracted from language/parser/test_primitive_contracts.py: test_public_aliases_parse_and_export.
Auto-assigned memref ids are process-global, so they are normalized to ``<id>``
before the whole-file comparison.
The sibling ``.pypto`` file holds the ``__str__`` (TextDump) output.
"""

from pathlib import Path
import re

from pypto_pro import ir
from pypto_pro._errors import InvalidOperation
import pypto_pro.language as pl
from pypto_pro.runtime.platform import NpuArch
import pytest


@pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
def scalar_aliases(x: pl.Tensor[[1], pl.DT_FP32]):
    tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    a = pl.make_tile(tt, addr=0)
    b = pl.make_tile(tt, addr=256)
    with pl.section_vector():
        pl.adds(b, a, 0.5)
        pl.subs(b, b, 0.25)
        pl.muls(b, b, 2.0)


def test_public_aliases_parse_and_export():
    program, _ = scalar_aliases.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    text = str(program)
    printed = text
    printed = re.sub(r'memref_id=\d+', 'memref_id=<id>', printed)
    golden = Path(__file__).with_suffix(".pypto").read_text(encoding="utf-8")
    assert printed.rstrip("\n") == golden.rstrip("\n")
    for name in ("adds", "subs", "muls"):
        assert name in pl.__all__ and callable(getattr(pl, name))
        with pytest.raises(InvalidOperation):
            getattr(pl, name)(None, None, 1)
