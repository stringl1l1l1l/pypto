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
"""Extracted from runtime/test_parse_observer.py: test_observer_sees_final_program_once.
Auto-assigned memref ids are process-global, so they are normalized to ``<id>``
before the whole-file comparison.
The sibling ``.pypto`` file holds the ``__str__`` (TextDump) output.
"""

from pathlib import Path
import re

from pypto_pro import ir
import pypto_pro.language as pl
from pypto_pro.runtime.platform import NpuArch


@pl.jit(arch=NpuArch.DAV_3510)
def grouped_copy(a: pl.Tensor[[1, 64], pl.DT_FP32], out: pl.Tensor[[1, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    group = pl.make_tile_group(type=tt, addrs=[0, 256], mutex_ids="auto", depth=2)
    with pl.section_vector():
        tile = group.next()
        pl.load(tile, a, [0, 0])
        pl.store(out, tile, [0, 0])



def test_observer_sees_final_program_once():
    calls = []
    definition = grouped_copy.to_kernel_def()
    program, matched = definition.parse_target_program(
        ir.SectionKind.Vector, observer=lambda p, parser: calls.append((p, parser)))
    assert matched and len(calls) == 1
    assert calls[0][0].same_as(program)
    assert calls[0][1].tile_group_meta
    printed = str(program)
    printed = re.sub(r'memref_id=\d+', 'memref_id=<id>', printed)
    golden = Path(__file__).with_suffix(".pypto").read_text(encoding="utf-8")
    assert printed.rstrip("\n") == golden.rstrip("\n")
    definition.parse_target_program(ir.SectionKind.Cube)
    assert len(calls) == 1
