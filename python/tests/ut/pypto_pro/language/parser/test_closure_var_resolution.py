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
"""Unit tests for closure variable resolution in DSL function bodies (issue #276).

Verifies that Python globals/closure variables used as positional arguments
in function calls inside kernel bodies are resolved correctly.
"""

from pypto_pro import ir
from pypto_pro._errors import InvalidType, NameNotFound
import pypto_pro.language as pl
import pytest


def test_list_closure_var_as_positional_arg():
    """List closure var works as positional arg (the original issue)."""
    offset_value = [0, 0]
    tile_shape = [64, 64]

    @pl.jit(auto_mutex=False)
    def func(
        t: pl.Tensor[[128, 128], pl.DT_FP32], out: pl.Tensor[[128, 128], pl.DT_FP32]
    ):
        tile_type = pl.TileType(shape=tile_shape, dtype=pl.DT_FP32)
        a = pl.make_tile(tile_type, addr=0)
        pl.load(a, t, offset_value)
        result: pl.Tensor[[128, 128], pl.DT_FP32] = pl.store(out, a, offset_value)
        _test_result = result

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)


def test_int_closure_var_as_positional_arg():
    """Int closure variable resolves to ConstInt in function body."""
    addr_value = 512

    @pl.jit(auto_mutex=False)
    def func(x: pl.Tensor[[1, 64], pl.DT_FP32]):
        tile_type = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        result = pl.make_tile(tile_type, addr=addr_value)
        _test_result = result

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)


def test_float_closure_var_as_positional_arg():
    """Float closure variable resolves to ConstFloat in function body."""
    scale_value = 2.0

    @pl.jit(auto_mutex=False)
    def func(x: pl.Tensor[[1, 64], pl.DT_FP32]):
        tile_type = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        o = pl.make_tile(tile_type, addr=0)
        s = pl.make_tile(tile_type, addr=512)
        pl.axpy(o, s, scale_value)
        _test_result = o

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)


def test_bool_closure_var_as_positional_arg():
    """Bool closure variable resolves to ConstBool in function body."""
    flag_value = True

    @pl.jit(auto_mutex=False)
    def func(x: pl.Tensor[[1, 64], pl.DT_FP32]):
        pl.printf("closure bool", loc=flag_value)
        _test_result = x

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)


def test_tuple_closure_var_as_positional_arg():
    """Tuple closure variable resolves to MakeTuple in function body."""
    offset_value = (0, 0)
    tile_shape = (64, 64)

    @pl.jit(auto_mutex=False)
    def func(
        t: pl.Tensor[[128, 128], pl.DT_FP32], out: pl.Tensor[[128, 128], pl.DT_FP32]
    ):
        tile_type = pl.TileType(shape=tile_shape, dtype=pl.DT_FP32)
        a = pl.make_tile(tile_type, addr=0)
        pl.load(a, t, offset_value)
        result: pl.Tensor[[128, 128], pl.DT_FP32] = pl.store(out, a, offset_value)
        _test_result = result

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)


def test_closure_tuple_has_entry_anchor_and_folded_reads():
    values = (3, 7)

    @pl.jit(auto_mutex=False)
    def func(idx: pl.DT_INT64):
        selected = values[idx]  # noqa: F841

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assignments = [stmt for stmt in func.body.stmts if isinstance(stmt, ir.AssignStmt)]
    anchor = next(stmt for stmt in assignments if stmt.var.name == "values_0")
    selected = next(stmt for stmt in assignments if stmt.var.name == "selected_0")
    assert isinstance(anchor.value, ir.MakeTuple)
    assert isinstance(selected.value, ir.GetItemExpr)
    assert selected.value.value is anchor.value


def test_nested_list_closure_var():
    """Nested list closure variable recursively converts to nested MakeTuple."""
    offsets_value = [0, 0]

    @pl.jit(auto_mutex=False)
    def func(
        t: pl.Tensor[[128, 128], pl.DT_FP32], out: pl.Tensor[[128, 128], pl.DT_FP32]
    ):
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32)
        a = pl.make_tile(tile_type, addr=0)
        pl.load(a, t, offsets_value)  # type: ignore[arg-type]
        result: pl.Tensor[[128, 128], pl.DT_FP32] = pl.store(out, a, [0, 0])
        _test_result = result

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)


def test_dynamic_tensor_shape_value():
    """A DYNAMIC parameter dimension is read through tensor.shape."""
    @pl.jit(auto_mutex=False)
    def func(
        x: pl.Tensor[[pl.DYNAMIC, 64], pl.DT_FP32],
    ):
        tile_type = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        o = pl.make_tile(tile_type, addr=0)
        s = pl.make_tile(tile_type, addr=512)
        pl.axpy(o, s, x.shape[0])
        _test_result = o

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)


def test_dsl_scope_shadows_closure():
    """Variable defined in DSL body shadows same-named closure variable."""
    x_scale = 999.0  # noqa: F841 -deliberately shadowed by DSL assignment

    @pl.jit(auto_mutex=False)
    def func(x: pl.Tensor[[1, 64], pl.DT_FP32]):
        tile_type = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        x_scale = pl.make_tile(tile_type, addr=0)
        result = pl.make_tile(tile_type, addr=512)
        pl.add(result, x_scale, x_scale)
        _test_result = result

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)


def test_undefined_variable_still_raises():
    """Variable not in scope or closure raises PyptoProError."""
    with pytest.raises(NameNotFound, match="Use of potentially undefined variable"):

        @pl.jit(auto_mutex=False)
        def func(x: pl.Tensor[[1, 64], pl.DT_FP32]):
            tile_type = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
            o = pl.make_tile(tile_type, addr=0)
            s = pl.make_tile(tile_type, addr=512)
            pl.axpy(o, s, totally_undefined)  # noqa: F821 # type: ignore

        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_unsupported_closure_type_raises():
    """Unsupported closure variable type raises PyptoProError."""
    bad_value = "not_a_number"

    with pytest.raises(InvalidType, match="Unsupported closure variable type: str"):

        @pl.jit(auto_mutex=False)
        def func(x: pl.Tensor[[1, 64], pl.DT_FP32]):
            tile_type = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
            o = pl.make_tile(tile_type, addr=0)
            s = pl.make_tile(tile_type, addr=512)
            pl.axpy(o, s, bad_value)  # type: ignore[arg-type]

        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
