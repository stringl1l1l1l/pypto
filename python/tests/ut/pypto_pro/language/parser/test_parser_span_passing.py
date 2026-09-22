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
"""Smoke tests for parser-provided operation spans."""

from pypto_pro import ir
import pypto_pro.language as pl


def _call_spans(func):
    spans = []
    for stmt in func.body.stmts:
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call):
            spans.append((stmt.value.name, stmt.value.span))
    return spans


def test_parser_passes_valid_spans_to_operations():
    @pl.jit(auto_mutex=False)
    def parsed_ops(
        x: pl.Tensor[[1, 64], pl.DT_FP32],
        y: pl.Tensor[[1, 64], pl.DT_FP32],
    ):
        tile_type = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        a = pl.make_tile(tile_type, addr=0)
        b = pl.make_tile(tile_type, addr=256)
        pl.load(a, x, [0, 0])
        pl.load(b, y, [0, 0])
        pl.add(b, a, b)
        _test_result = b

    parsed_ops_program, _ = parsed_ops.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    parsed_ops = parsed_ops_program.get_function(parsed_ops.__name__)

    spans = _call_spans(parsed_ops)

    # Two separate op calls on two source lines: the assertions below are about the
    # spans the parser attached, not about which op each line happens to build.
    assert [name for name, _ in spans] == ["block.make_tile", "block.make_tile"]
    for _, span in spans:
        assert span.is_valid()
        assert span.begin_line > 0
        assert span.begin_column > 0
        assert span.end_line >= span.begin_line
        if span.end_line == span.begin_line:
            assert span.end_column >= span.begin_column
    assert spans[0][1].begin_line != spans[1][1].begin_line
