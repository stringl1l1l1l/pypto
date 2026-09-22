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
"""Tests for scalar operation dispatch in the DSL parser.

Verifies that pl.min, pl.max dispatch to scalar IR ops
when called with scalar arguments.
"""

import pypto_pro.language as pl
import pytest

from pypto.pypto_impl import ir


@pytest.mark.parametrize(
    "dtype,expected",
    [(pl.DT_INT8, pl.DT_INT64), (pl.DT_UINT32, pl.DT_INT64),
     (pl.DT_UINT64, pl.DT_UINT64), (pl.DT_FP32, pl.DT_FP32)],
)
@pytest.mark.parametrize("use_tile", [False, True], ids=["tensor", "tile"])
def test_simd_element_reads_keep_scalar_promotion(dtype, expected, use_tile):
    @pl.jit(auto_mutex=False)
    def kernel(data: pl.Tensor[[1, 32], dtype]):
        with pl.section_vector():
            if use_tile:
                container = pl.make_tile(
                    pl.TileType(shape=[1, 32], dtype=dtype, target_memory=pl.MemorySpace.Vec), addr=0
                )
            else:
                container = data
            first = pl.getval(container, 0)
            second = container[0, 1]
            pl.setval(container, 2, first + second)

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function = program.get_function("kernel")
    loads = [
        stmt.value
        for stmt in function.body.stmts
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call) and stmt.value.name == "block.getval"
    ]
    assert len(loads) == 2
    assert all(load.type.dtype == expected for load in loads)


def test_simd_integer_reads_merge_as_int64_across_branches():
    @pl.jit(auto_mutex=False)
    def kernel(
        narrow: pl.Tensor[[1], pl.DT_INT8],
        unsigned: pl.Tensor[[1], pl.DT_UINT32],
        out: pl.Tensor[[1], pl.DT_INT64],
        flag: pl.DT_BOOL,
    ):
        with pl.section_vector():
            if flag:
                value = narrow[0]
            else:
                value = pl.getval(unsigned, 0)
            out[0] = value

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function = program.get_function("kernel")
    branch = next(stmt for stmt in function.body.stmts if isinstance(stmt, ir.IfStmt))
    assert len(branch.return_vars) == 1
    assert branch.return_vars[0].type.dtype == pl.DT_INT64
    for body in (branch.then_body, branch.else_body):
        loads = [stmt.value for stmt in body.stmts if isinstance(stmt, ir.AssignStmt)]
        load = next(value for value in loads if isinstance(value, ir.Call) and value.name == "block.getval")
        assert load.type.dtype == pl.DT_INT64


def test_scalar_min():
    """Test pl.min(scalar, scalar) dispatches to ir.min_."""

    @pl.jit(auto_mutex=False)
    def test_min(
        config: pl.Tensor[[2], pl.DT_INT64],
        out: pl.Tensor[[2, 16, 128], pl.DT_FP32],
    ):
        a: pl.DT_UINT64 = pl.getval(config, 0)
        b: pl.DT_UINT64 = pl.getval(config, 1)
        c = pl.min(a, b)
        _ = c + 1
        _test_result = out

    test_min_program, _ = test_min.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    test_min = test_min_program.get_function(test_min.__name__)

    assert isinstance(test_min, ir.Function)
    ir_text = ir.python_print(test_min)
    assert "min" in ir_text.lower()


def test_scalar_max():
    """Test pl.max(scalar, scalar) dispatches to ir.max_."""

    @pl.jit(auto_mutex=False)
    def test_max(
        config: pl.Tensor[[2], pl.DT_INT64],
        out: pl.Tensor[[2, 16, 128], pl.DT_FP32],
    ):
        a: pl.DT_UINT64 = pl.getval(config, 0)
        b: pl.DT_UINT64 = pl.getval(config, 1)
        c = pl.max(a, b)
        _ = c + 1
        _test_result = out

    test_max_program, _ = test_max.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    test_max = test_max_program.get_function(test_max.__name__)

    assert isinstance(test_max, ir.Function)
    ir_text = ir.python_print(test_max)
    assert "max" in ir_text.lower()


def test_scalar_min_with_literal():
    """Test pl.min(scalar, int_literal) -the paged_attention use case."""

    @pl.jit(auto_mutex=False)
    def test_min_lit(
        config: pl.Tensor[[2], pl.DT_INT64],
        out: pl.Tensor[[2, 16, 128], pl.DT_FP32],
    ):
        a: pl.DT_UINT64 = pl.getval(config, 0)
        c = pl.min(a, 128)
        _ = c + 1
        _test_result = out

    test_min_lit_program, _ = test_min_lit.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    test_min_lit = test_min_lit_program.get_function(test_min_lit.__name__)

    assert isinstance(test_min_lit, ir.Function)
    ir_text = ir.python_print(test_min_lit)
    assert "min" in ir_text.lower()


def test_tile_min_still_works():
    """Ensure pl.min(tile, axis=...) still works as tile reduction."""

    @pl.jit(auto_mutex=False)
    def test_tile_min(
        x: pl.Tensor[[32, 32], pl.DT_FP32],
    ):
        tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
        reduced_type = pl.TileType(shape=[32, 1], dtype=pl.DT_FP32)
        tile_a = pl.make_tile(tile_type, addr=0)
        tmp = pl.make_tile(tile_type, addr=4096)
        tile_c = pl.make_tile(reduced_type, addr=8192)
        pl.load(tile_a, x, [0, 0])
        pl.minimum(tile_a, tmp, tile_c, dim=0)
        out: pl.Tensor[[32, 32], pl.DT_FP32] = pl.store(x, tile_c, [0, 0])
        _test_result = out

    test_tile_min_program, _ = test_tile_min.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    test_tile_min = test_tile_min_program.get_function(test_tile_min.__name__)

    assert isinstance(test_tile_min, ir.Function)
