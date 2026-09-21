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
"""Unit tests for control flow parsing (for loops, if statements)."""

from pypto_pro import ir
from pypto_pro._errors import InvalidVal, NameNotFound, NotSupported
import pypto_pro.language as pl
import pytest


def test_loop_without_iter_args():
    """Test loop without iter_args."""

    @pl.jit(auto_mutex=False)
    def loop_without_iter_args(x: pl.Tensor[[64], pl.DT_FP32]):
        result: pl.Tensor[[64], pl.DT_FP32] = x
        for i in pl.range(3):
            if i > 0:
                temp = pl.tensor.mul(result, 2.0)
                result = temp
            else:
                temp = pl.tensor.add(result, 1.0)
                result = temp
        _test_result = result

    loop_without_iter_args_program, _ = loop_without_iter_args.to_kernel_def().parse_target_program(
        ir.SectionKind.Vector
    )
    loop_without_iter_args = loop_without_iter_args_program.get_function(loop_without_iter_args.__name__)

    assert isinstance(loop_without_iter_args, ir.Function)


def test_target_section_jump_is_dispatched_by_enclosing_loop_body():
    @pl.jit(auto_mutex=False)
    def func(value: pl.DT_INT64):
        for _ in pl.range(1):
            with pl.section_vector():
                break
            unreachable = value + 1  # noqa: F841
        _test_result = value

    program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func_ir = program.get_function(func.__name__)
    for_stmt = next(stmt for stmt in func_ir.body.stmts if isinstance(stmt, ir.ForStmt))

    assert sum(isinstance(stmt, ir.BreakStmt) for stmt in for_stmt.body.stmts) == 1
    assert all(
        not (isinstance(stmt, ir.AssignStmt) and stmt.var.name.startswith("unreachable"))
        for stmt in for_stmt.body.stmts
    )


def test_loop_carried_constant_is_not_folded_in_body():
    @pl.jit(auto_mutex=False)
    def func(n: pl.DT_INT64):
        x = 0
        for _ in pl.range(n):
            observed = x  # noqa: F841
            x += 1

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    for_stmt = _find_for_stmt(func)
    body = for_stmt.body.stmts
    observed_assignment = next(
        stmt
        for stmt in body
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name.startswith("observed")
    )
    assert isinstance(observed_assignment.value, ir.Var)


def test_for_target_is_rebound_each_iteration_and_merged_after_loop():
    @pl.jit(auto_mutex=False)
    def func(n: pl.DT_INT64):
        i = 5
        for i in pl.range(n):
            i = i + 10
        _test_result = i + 1

    program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func_ir = program.get_function(func.__name__)
    for_stmt = _find_for_stmt(func_ir)

    assert for_stmt.loop_var.name.startswith("i__iterator")
    assert len(for_stmt.iter_args) == 1
    assert len(for_stmt.return_vars) == 1

    bind_target = for_stmt.body.stmts[0]
    assert isinstance(bind_target, ir.AssignStmt)
    assert ir.structural_equal(bind_target.value, for_stmt.loop_var, enable_auto_mapping=False)

    update_target = next(
        stmt
        for stmt in for_stmt.body.stmts
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Add)
    )
    assert ir.structural_equal(update_target.value.left, bind_target.var, enable_auto_mapping=False)
    assert not ir.structural_equal(
        update_target.value.left,
        for_stmt.iter_args[0].iterVar,
        enable_auto_mapping=False,
    )

    continue_stmt = for_stmt.body.stmts[-1]
    assert isinstance(continue_stmt, ir.ContinueStmt)
    assert ir.structural_equal(continue_stmt.value[0], update_target.var, enable_auto_mapping=False)

    result_assignment = next(
        stmt
        for stmt in func_ir.body.stmts
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name.startswith("_test_result")
    )
    assert isinstance(result_assignment.value, ir.Add)
    assert ir.structural_equal(
        result_assignment.value.left,
        for_stmt.return_vars[0],
        enable_auto_mapping=False,
    )


def test_for_target_without_preexisting_binding_is_not_carried():
    @pl.jit(auto_mutex=False)
    def func(n: pl.DT_INT64):
        for i in pl.range(n):
            _test_result = i + 1  # noqa: F841

    program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    for_stmt = _find_for_stmt(program.get_function(func.__name__))

    assert not for_stmt.iter_args
    assert not for_stmt.return_vars
    bind_target = for_stmt.body.stmts[0]
    assert isinstance(bind_target, ir.AssignStmt)
    assert ir.structural_equal(bind_target.value, for_stmt.loop_var, enable_auto_mapping=False)


def test_nested_for_target_is_carried_by_enclosing_loop():
    @pl.jit(auto_mutex=False)
    def func(n: pl.DT_INT64):
        i = 7
        for _ in pl.range(n):
            for i in pl.range(3):
                pass
        _test_result = i + 1

    program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    outer = _find_for_stmt(program.get_function(func.__name__))
    inner = next(stmt for stmt in outer.body.stmts if isinstance(stmt, ir.ForStmt))
    assert len(outer.iter_args) == len(inner.iter_args) == 1
    assert ir.structural_equal(inner.iter_args[0].initValue, outer.iter_args[0].iterVar, enable_auto_mapping=False)
    assert ir.structural_equal(outer.body.stmts[-1].value[0], inner.return_vars[0], enable_auto_mapping=False)


def test_static_if_only_emits_selected_branch():
    @pl.jit(auto_mutex=False)
    def func(_jit_entry: pl.DT_INT64):
        if True:
            selected = 7
        else:
            selected = (1, 2)[3]  # noqa: F841
        observed = selected + 1  # noqa: F841

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert all(not isinstance(stmt, ir.IfStmt) for stmt in func.body.stmts)


def test_dynamic_if_type_mismatch_is_deferred_until_use():
    @pl.jit(auto_mutex=False)
    def unused_mismatch(flag: pl.DT_BOOL):
        if flag:
            value = pl.const(1, pl.DT_INT32)
        else:
            value = pl.const(2, pl.DT_INT64)  # noqa: F841

    unused_program, _ = unused_mismatch.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    unused_mismatch = unused_program.get_function(unused_mismatch.__name__)
    if_stmt = next(stmt for stmt in unused_mismatch.body.stmts if isinstance(stmt, ir.IfStmt))
    assert isinstance(if_stmt.return_vars[0].type, ir.NoneType)

    with pytest.raises(NameNotFound, match="Use of potentially undefined variable"):

        @pl.jit(auto_mutex=False)
        def used_mismatch(flag: pl.DT_BOOL):
            if flag:
                value = pl.const(1, pl.DT_INT32)
            else:
                value = pl.const(2, pl.DT_INT64)
            result = value + 1
            _test_result = result

        used_mismatch.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_dynamic_if_parser_only_merge_is_deferred_until_use():
    @pl.jit(auto_mutex=False)
    def unused_parser_value(flag: pl.DT_BOOL):
        if flag:
            tile_type = pl.TileType(shape=[16], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        else:
            tile_type = pl.TileType(shape=[32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)  # noqa: F841

    unused_program, _ = unused_parser_value.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    unused_func = unused_program.get_function(unused_parser_value.__name__)
    if_stmt = next(stmt for stmt in unused_func.body.stmts if isinstance(stmt, ir.IfStmt))
    assert isinstance(if_stmt.return_vars[0].type, ir.NoneType)

    with pytest.raises(NameNotFound, match="Use of potentially undefined variable"):

        @pl.jit(auto_mutex=False)
        def used_parser_value(flag: pl.DT_BOOL):
            if flag:
                tile_type = pl.TileType(shape=[16], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
            else:
                tile_type = pl.TileType(shape=[32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
            _test_result = tile_type

        used_parser_value.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_dynamic_if_missing_binding_uses_none_type_input():
    @pl.jit(auto_mutex=False)
    def func(flag: pl.DT_BOOL):
        if flag:
            value = 1  # noqa: F841

    program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func_ir = program.get_function(func.__name__)
    if_stmt = next(stmt for stmt in func_ir.body.stmts if isinstance(stmt, ir.IfStmt))
    else_yield = if_stmt.else_body.stmts[-1]

    assert isinstance(else_yield, ir.YieldStmt)
    assert isinstance(else_yield.value[0].type, ir.NoneType)
    assert isinstance(if_stmt.return_vars[0].type, ir.NoneType)


def test_none_type_binding_is_rejected_when_used():
    with pytest.raises(NameNotFound, match="Use of potentially undefined variable"):

        @pl.jit(auto_mutex=False)
        def func(_jit_entry: pl.DT_INT64):
            value = None
            observed = value  # noqa: F841

        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_dynamic_if_nested_tile_group_merge_is_deferred_until_use():
    @pl.jit(auto_mutex=False)
    def unused_tile_group(flag: pl.DT_BOOL):
        tile_type = pl.TileType(
            shape=[1, 16], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
        )
        group = pl.make_tile_group(type=tile_type, addrs=0, mutex_ids=[0])
        container = (group, 1)
        if flag:
            selected = container
        else:
            selected = container  # noqa: F841

    program, _ = unused_tile_group.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func_ir = program.get_function(unused_tile_group.__name__)
    if_stmt = next(stmt for stmt in func_ir.body.stmts if isinstance(stmt, ir.IfStmt))
    assert isinstance(if_stmt.return_vars[0].type, ir.NoneType)

    with pytest.raises(NameNotFound, match="Use of potentially undefined variable"):

        @pl.jit(auto_mutex=False)
        def used_tile_group(flag: pl.DT_BOOL):
            tile_type = pl.TileType(
                shape=[1, 16], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
            )
            group = pl.make_tile_group(type=tile_type, addrs=0, mutex_ids=[0])
            container = (group, 1)
            if flag:
                selected = container
            else:
                selected = container
            observed = selected  # noqa: F841

        used_tile_group.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_dynamic_loop_parser_only_merge_is_deferred_until_use():
    @pl.jit(auto_mutex=False)
    def unused_parser_value(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(shape=[16], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        for _ in pl.range(1):
            tile_type = pl.TileType(shape=[32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)  # noqa: F841

    unused_program, _ = unused_parser_value.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    unused_func = unused_program.get_function(unused_parser_value.__name__)
    for_stmt = next(stmt for stmt in unused_func.body.stmts if isinstance(stmt, ir.ForStmt))
    assert isinstance(for_stmt.return_vars[0].type, ir.NoneType)

    with pytest.raises(NameNotFound, match="Use of potentially undefined variable"):

        @pl.jit(auto_mutex=False)
        def used_parser_value(_jit_entry: pl.DT_INT64):
            tile_type = pl.TileType(shape=[16], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
            for _ in pl.range(1):
                tile_type = pl.TileType(shape=[32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
            _test_result = tile_type

        used_parser_value.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_astype_reconciles_dynamic_if_scalar_types():
    @pl.jit(auto_mutex=False)
    def reconciled(flag: pl.DT_BOOL):
        if flag:
            value = pl.const(1, pl.DT_INT32)
        else:
            value = pl.astype(pl.const(2, pl.DT_INT64), pl.DT_INT32)
        result = value + 1
        _test_result = result

    reconciled_program, _ = reconciled.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    reconciled = reconciled_program.get_function(reconciled.__name__)
    if_stmt = next(stmt for stmt in reconciled.body.stmts if isinstance(stmt, ir.IfStmt))
    merged_type = if_stmt.return_vars[0].type
    assert isinstance(merged_type, ir.ScalarType)
    assert merged_type.dtype == ir.DataType.INT32
    assert any(
        isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Cast)
        for stmt in if_stmt.else_body.stmts
    )


def test_ssa_names_avoid_user_suffix_collisions():
    @pl.jit(auto_mutex=False)
    def func(_jit_entry: pl.DT_INT64):
        x = 1
        x_1 = 2  # noqa: F841
        x = 3  # noqa: F841

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)
    names = [
        stmt.var.name
        for stmt in func.body.stmts
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name.startswith("x")
    ]
    assert names == ["x_0", "x_1_0", "x_1"]


def test_loop_carries_preexisting_break_value():
    @pl.jit(auto_mutex=False)
    def func(n: pl.DT_INT64):
        value = pl.const(0, pl.DT_INT32)
        for _ in pl.range(n):
            value = pl.const(1, pl.DT_INT32)
            break
        _test_result = value

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)
    for_stmt = _find_for_stmt(func)
    assert len(for_stmt.iter_args) == 1
    assert for_stmt.iter_args[0].initValue.type.dtype == ir.DataType.INT32
    assert for_stmt.iter_args[0].iterVar.type.dtype == ir.DataType.INT32
    assert for_stmt.return_vars[0].type.dtype == ir.DataType.INT32
    break_stmt = next(stmt for stmt in for_stmt.body.stmts if isinstance(stmt, ir.BreakStmt))
    assert break_stmt.value[0].type.dtype == ir.DataType.INT32


def test_loop_does_not_carry_body_local_binding():
    @pl.jit(auto_mutex=False)
    def func(n: pl.DT_INT64):
        for _ in pl.range(n):
            local = pl.const(1, pl.DT_INT32)  # noqa: F841

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)
    for_stmt = _find_for_stmt(func)
    assert not for_stmt.iter_args
    assert not for_stmt.return_vars
    assert isinstance(for_stmt.body.stmts[-1], ir.ContinueStmt)
    assert not for_stmt.body.stmts[-1].value


def test_if_continue_branch_does_not_yield_and_loop_fallthrough_continues():
    @pl.jit(auto_mutex=False)
    def func(n: pl.DT_INT64):
        value = pl.const(0, pl.DT_INT64)
        for _ in pl.range(n):
            if value == 1:
                value = pl.const(2, pl.DT_INT64)
                continue
            value = value + 1
        _test_result = value

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)
    for_stmt = _find_for_stmt(func)
    guard = next(stmt for stmt in for_stmt.body.stmts if isinstance(stmt, ir.IfStmt))

    assert isinstance(guard.then_body.stmts[-1], ir.ContinueStmt)
    assert all(not isinstance(stmt, ir.YieldStmt) for stmt in guard.then_body.stmts)
    assert isinstance(guard.else_body.stmts[-1], ir.YieldStmt)
    assert isinstance(for_stmt.body.stmts[-1], ir.ContinueStmt)


def test_if_else_break_uses_only_then_yield_for_if_phi():
    @pl.jit(auto_mutex=False)
    def func(n: pl.DT_INT64):
        value = pl.const(0, pl.DT_INT64)
        for i in pl.range(n):
            if i > 0:
                value = pl.const(1, pl.DT_INT64)
            else:
                value = pl.const(2, pl.DT_INT64)
                break
            value = value + 1
        _test_result = value

    program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    for_stmt = _find_for_stmt(program.get_function(func.__name__))
    branch = next(stmt for stmt in for_stmt.body.stmts if isinstance(stmt, ir.IfStmt))

    assert len(branch.return_vars) == 1
    assert isinstance(branch.then_body.stmts[-1], ir.YieldStmt)
    assert len(branch.then_body.stmts[-1].value) == 1
    assert isinstance(branch.else_body.stmts[-1], ir.BreakStmt)
    assert len(branch.else_body.stmts[-1].value) == 1
    assert isinstance(for_stmt.body.stmts[-1], ir.ContinueStmt)


def test_if_both_branches_jump_stops_enclosing_statement_list():
    @pl.jit(auto_mutex=False)
    def func(n: pl.DT_INT64):
        value = pl.const(0, pl.DT_INT64)
        for i in pl.range(n):
            if i > 0:
                break
            else:
                continue
            invalid = (1, 2)[3]  # noqa: F841
        _test_result = value

    program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    for_stmt = _find_for_stmt(program.get_function(func.__name__))

    branch_index = next(i for i, stmt in enumerate(for_stmt.body.stmts) if isinstance(stmt, ir.IfStmt))
    branch = for_stmt.body.stmts[branch_index]
    assert isinstance(branch.then_body.stmts[-1], ir.BreakStmt)
    assert isinstance(for_stmt.body.stmts[branch_index + 1], ir.ContinueStmt)
    assert sum(isinstance(stmt, ir.ContinueStmt) for stmt in for_stmt.body.stmts) == 1
    assert all(
        not (isinstance(stmt, ir.AssignStmt) and stmt.var.name.startswith("invalid"))
        for stmt in for_stmt.body.stmts
    )


def test_break_type_conflict_fails_eagerly():
    with pytest.raises(InvalidVal, match="Inconsistent types in control flow"):

        @pl.jit(auto_mutex=False)
        def func(n: pl.DT_INT64):
            value = pl.const(0, pl.DT_INT32)
            for i in pl.range(n):
                if i > 0:
                    value = pl.const(1, pl.DT_INT64)
                    break
                value = pl.const(2, pl.DT_INT32)  # noqa: F841

        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_continue_type_conflict_fails_eagerly():
    with pytest.raises(InvalidVal, match="Inconsistent types in control flow"):

        @pl.jit(auto_mutex=False)
        def func(n: pl.DT_INT64):
            value = pl.const(0, pl.DT_INT32)
            for _ in pl.range(n):
                value = pl.const(1, pl.DT_INT64)  # noqa: F841

        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_while_body_and_result_types_are_independent():
    @pl.jit(auto_mutex=False)
    def func(flag: pl.DT_BOOL):
        value = pl.const(0, pl.DT_INT32)
        while True:
            if flag:
                value = pl.const(1, pl.DT_INT64)
                break
            value = pl.const(2, pl.DT_INT32)
            continue
        _test_result = value

    program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    while_stmt = next(
        stmt
        for stmt in program.get_function(func.__name__).body.stmts
        if isinstance(stmt, ir.WhileStmt)
    )

    assert while_stmt.iter_args[0].iterVar.type.dtype == ir.DataType.INT32
    assert while_stmt.return_vars[0].type.dtype == ir.DataType.INT64


def test_while_body_type_conflict_fails_eagerly_after_valid_iter_use():
    with pytest.raises(InvalidVal, match="Inconsistent types in control flow"):

        @pl.jit(auto_mutex=False)
        def func(flag: pl.DT_BOOL):
            value = pl.const(0, pl.DT_INT32)
            while True:
                next_value = value + 1
                if flag:
                    value = pl.const(1, pl.DT_INT64)
                    continue
                value = next_value
                continue

        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_while_result_type_conflict_is_deferred_when_result_is_unused():
    @pl.jit(auto_mutex=False)
    def func(flag: pl.DT_BOOL):
        value = pl.const(0, pl.DT_INT32)
        while True:
            if flag:
                value = pl.const(1, pl.DT_INT64)
                break
            value = pl.const(2.0, pl.DT_FP32)  # noqa: F841
            break
        _test_result = flag

    program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    while_stmt = next(
        stmt
        for stmt in program.get_function(func.__name__).body.stmts
        if isinstance(stmt, ir.WhileStmt)
    )

    assert while_stmt.iter_args[0].iterVar.type.dtype == ir.DataType.INT32
    assert isinstance(while_stmt.return_vars[0].type, ir.NoneType)


def test_statement_list_stops_after_direct_break():
    @pl.jit(auto_mutex=False)
    def func(n: pl.DT_INT64):
        value = pl.const(0, pl.DT_INT64)
        for _ in pl.range(n):
            break
            value = (1, 2)[3]
        _test_result = value

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)
    for_stmt = _find_for_stmt(func)

    assert len(for_stmt.body.stmts) == 2
    assert isinstance(for_stmt.body.stmts[0], ir.AssignStmt)
    assert isinstance(for_stmt.body.stmts[1], ir.BreakStmt)


def test_phi_state_propagates_equal_branch_constant():
    @pl.jit(auto_mutex=False)
    def func(flag: pl.DT_BOOL):
        if flag:
            value = 1
        else:
            value = 1
        if value == 1:
            selected = 2  # noqa: F841
        else:
            selected = (1, 2)[3]  # noqa: F841

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)
    if_stmts = [stmt for stmt in func.body.stmts if isinstance(stmt, ir.IfStmt)]

    assert len(if_stmts) == 1


def test_tuple_is_one_if_merge_value_and_one_constant():
    @pl.jit(auto_mutex=False)
    def func(flag: pl.DT_BOOL):
        if flag:
            pair = (1, 2)
        else:
            pair = (1, 2)
        first = pair[0]
        _test_result = first

    program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func_ir = program.get_function(func.__name__)
    if_stmt = next(stmt for stmt in func_ir.body.stmts if isinstance(stmt, ir.IfStmt))
    then_yield = if_stmt.then_body.stmts[-1]
    else_yield = if_stmt.else_body.stmts[-1]

    assert len(if_stmt.return_vars) == 1
    assert isinstance(if_stmt.return_vars[0].type, ir.TupleType)
    assert isinstance(then_yield, ir.YieldStmt)
    assert isinstance(else_yield, ir.YieldStmt)
    assert len(then_yield.value) == 1
    assert len(else_yield.value) == 1
    assert isinstance(then_yield.value[0], ir.MakeTuple)
    assert isinstance(else_yield.value[0], ir.MakeTuple)
    first = next(
        stmt
        for stmt in func_ir.body.stmts
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name.startswith("first")
    )
    assert isinstance(first.value, ir.ConstInt)
    assert first.value.value == 1


def test_if_merges_named_tuples_with_the_same_fields():
    @pl.jit(auto_mutex=False)
    def func(flag: pl.DT_BOOL, value: pl.DT_INT64):
        if flag:
            record = pl.make_tuple(item=value)
        else:
            record = pl.make_tuple(item=value + 1)
        observed = record.item
        _test_result = observed

    program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func_ir = program.get_function(func.__name__)
    if_stmt = next(stmt for stmt in func_ir.body.stmts if isinstance(stmt, ir.IfStmt))

    assert isinstance(if_stmt.return_vars[0].type, ir.TupleType)
    assert any(
        isinstance(stmt, ir.AssignStmt)
        and stmt.var.name.startswith("observed")
        and isinstance(stmt.value, ir.GetItemExpr)
        for stmt in func_ir.body.stmts
    )


def test_if_rejects_plain_and_named_tuple_with_equal_element_types():
    with pytest.raises(NameNotFound, match="Use of potentially undefined variable"):

        @pl.jit(auto_mutex=False)
        def func(flag: pl.DT_BOOL, value: pl.DT_INT64):
            if flag:
                record = (value,)
            else:
                record = pl.make_tuple(item=value)
            observed = record[0]
            _test_result = observed

        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_if_rejects_named_tuples_with_different_fields():
    with pytest.raises(NameNotFound, match="Use of potentially undefined variable"):

        @pl.jit(auto_mutex=False)
        def func(flag: pl.DT_BOOL, value: pl.DT_INT64):
            if flag:
                record = pl.make_tuple(left=value)
            else:
                record = pl.make_tuple(right=value)
            observed = record.left
            _test_result = observed

        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_if_merges_structs_with_the_same_name_and_fields():
    @pl.jit(auto_mutex=False)
    def func(flag: pl.DT_BOOL, value: pl.DT_INT64):
        if flag:
            record = pl.struct("Record", item=value)
        else:
            record = pl.struct("Record", item=value + 1)
        observed = record.item
        _test_result = observed

    program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func_ir = program.get_function(func.__name__)
    if_stmt = next(stmt for stmt in func_ir.body.stmts if isinstance(stmt, ir.IfStmt))

    assert isinstance(if_stmt.return_vars[0].type, ir.TupleType)


def test_if_rejects_structs_with_different_names():
    with pytest.raises(NameNotFound, match="Use of potentially undefined variable"):

        @pl.jit(auto_mutex=False)
        def func(flag: pl.DT_BOOL, value: pl.DT_INT64):
            if flag:
                record = pl.struct("LeftRecord", item=value)
            else:
                record = pl.struct("RightRecord", item=value)
            observed = record.item
            _test_result = observed

        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_if_rejects_nested_tuple_with_different_inner_tuple_kinds():
    with pytest.raises(NameNotFound, match="Use of potentially undefined variable"):

        @pl.jit(auto_mutex=False)
        def func(flag: pl.DT_BOOL, value: pl.DT_INT64):
            if flag:
                record = ((value,),)
            else:
                record = (pl.make_tuple(item=value),)
            observed = record[0]
            _test_result = observed

        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_for_rejects_tuple_kind_change_in_loop_carry():
    with pytest.raises(InvalidVal, match="Inconsistent types in control flow"):

        @pl.jit(auto_mutex=False)
        def func(value: pl.DT_INT64):
            record = pl.make_tuple(item=value)
            for _ in pl.range(1):
                record = (value,)
            _test_result = record

        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_while_rejects_tuple_kind_change_in_loop_carry():
    with pytest.raises(InvalidVal, match="Inconsistent types in control flow"):

        @pl.jit(auto_mutex=False)
        def func(value: pl.DT_INT64):
            record = pl.make_tuple(item=value)
            while value > 0:
                record = (value,)
                continue
            _test_result = record

        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_const_env_uses_latest_physical_ssa_binding():
    @pl.jit(auto_mutex=False)
    def func(n: pl.DT_INT64):
        value = 1
        value = n
        if value == 1:
            selected = 2  # noqa: F841
        else:
            selected = 3  # noqa: F841

    program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func_ir = program.get_function(func.__name__)

    assert any(isinstance(stmt, ir.IfStmt) for stmt in func_ir.body.stmts)


def test_for_result_preserves_equal_constant_across_zero_and_continue_paths():
    @pl.jit(auto_mutex=False)
    def func(n: pl.DT_INT64):
        value = 0
        for _ in pl.range(n):
            value = 0
        if value == 0:
            selected = 1  # noqa: F841
        else:
            selected = (1, 2)[3]  # noqa: F841

    program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func_ir = program.get_function(func.__name__)
    for_stmt = next(stmt for stmt in func_ir.body.stmts if isinstance(stmt, ir.ForStmt))

    assert sum(isinstance(stmt, ir.IfStmt) for stmt in func_ir.body.stmts) == 0
    assert isinstance(for_stmt.iter_args[0].initValue, ir.ConstInt)
    assert isinstance(for_stmt.body.stmts[-1].value[0], ir.ConstInt)


def test_dynamic_ifexp_returns_constant_when_phi_is_constant():
    @pl.jit(auto_mutex=False)
    def func(flag: pl.DT_BOOL):
        selected = 1 if flag else 1
        _test_result = selected

    program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func_ir = program.get_function(func.__name__)
    if_stmt = next(stmt for stmt in func_ir.body.stmts if isinstance(stmt, ir.IfStmt))
    selected = next(
        stmt
        for stmt in func_ir.body.stmts
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name.startswith("selected")
    )
    then_yield = if_stmt.then_body.stmts[-1]
    else_yield = if_stmt.else_body.stmts[-1]

    assert len(if_stmt.return_vars) == 1
    assert isinstance(then_yield.value[0], ir.ConstInt)
    assert isinstance(else_yield.value[0], ir.ConstInt)
    assert isinstance(selected.value, ir.ConstInt)
    assert selected.value.value == 1


def test_function_scope_gets_implicit_return_jump():
    @pl.jit(auto_mutex=False)
    def func(_jit_entry: pl.DT_INT64):
        value = 1  # noqa: F841

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func.body.stmts[-1], ir.ReturnStmt)


def _find_for_stmt(func: ir.Function) -> ir.ForStmt:
    """Helper to extract the first ForStmt from a function body."""
    body = func.body
    if isinstance(body, ir.ForStmt):
        return body
    if isinstance(body, ir.SeqStmts):
        for stmt in body.stmts:
            if isinstance(stmt, ir.ForStmt):
                return stmt
    raise AssertionError("No ForStmt found in function body")


def _find_while_stmt(func: ir.Function) -> ir.WhileStmt:
    """Helper to find WhileStmt in function body."""
    stmt = func.body
    # Handle SeqStmts wrapper
    if isinstance(stmt, ir.SeqStmts):
        for s in stmt.stmts:
            if isinstance(s, ir.WhileStmt):
                return s
            # Check nested statements
            if isinstance(s, ir.ForStmt) and isinstance(s.body, ir.SeqStmts):
                for nested in s.body.stmts:
                    if isinstance(nested, ir.WhileStmt):
                        return nested
    # Direct while statement
    if isinstance(stmt, ir.WhileStmt):
        return stmt
    raise ValueError("No WhileStmt found in function body")


def _assert_materialized_stop(func: ir.Function, expr_type) -> None:
    for_stmt = _find_for_stmt(func)
    assert isinstance(for_stmt.stop, ir.Var)
    assignment = next(
        stmt
        for stmt in func.body.stmts
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name == for_stmt.stop.name
    )
    assert isinstance(assignment.value, expr_type)


def _assert_lowered_while_guard(while_stmt: ir.WhileStmt, expr_type) -> None:
    assert isinstance(while_stmt.condition, ir.ConstBool)
    assert while_stmt.condition.value
    guard_index = next(
        index for index, stmt in enumerate(while_stmt.body.stmts) if isinstance(stmt, ir.IfStmt)
    )
    assert any(
        isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, expr_type)
        for stmt in while_stmt.body.stmts[:guard_index]
    )
    guard = while_stmt.body.stmts[guard_index]
    assert any(isinstance(stmt, ir.BreakStmt) for stmt in guard.then_body.stmts)


def test_scalar_param_as_stop():
    """Test pl.range(n) where n is a DT_INT64 scalar parameter."""

    @pl.jit(auto_mutex=False)
    def scalar_stop(n: pl.DT_INT64, x: pl.Tensor[[64], pl.DT_FP32]):
        y = x
        for _ in pl.range(n):
            y: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(x, 1.0)
        _test_result = y

    scalar_stop_program, _ = scalar_stop.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    scalar_stop = scalar_stop_program.get_function(scalar_stop.__name__)

    assert isinstance(scalar_stop, ir.Function)
    for_stmt = _find_for_stmt(scalar_stop)
    # stop should be a Var reference to the Scalar parameter 'n'
    assert isinstance(for_stmt.stop, ir.Var)
    assert for_stmt.stop.name == "n_0"
    assert isinstance(for_stmt.stop.type, ir.ScalarType)


def test_scalar_param_as_start_stop():
    """Test pl.range(0, n) where n is a DT_INT64 scalar parameter."""

    @pl.jit(auto_mutex=False)
    def scalar_start_stop(
        n: pl.DT_INT64, x: pl.Tensor[[64], pl.DT_FP32]
    ):
        y = x
        for _ in pl.range(0, n):
            y: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(x, 1.0)
        _test_result = y

    scalar_start_stop_program, _ = scalar_start_stop.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    scalar_start_stop = scalar_start_stop_program.get_function(scalar_start_stop.__name__)

    assert isinstance(scalar_start_stop, ir.Function)
    for_stmt = _find_for_stmt(scalar_start_stop)
    assert isinstance(for_stmt.start, ir.ConstInt)
    assert isinstance(for_stmt.stop, ir.Var)
    assert for_stmt.stop.name == "n_0"


def test_scalar_param_as_for_step_is_accepted():
    @pl.jit(auto_mutex=False)
    def scalar_full_range(n: pl.DT_INT64, step: pl.DT_INT64):
        for _ in pl.range(0, n, step):
            pass

    program, _ = scalar_full_range.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    for_stmt = _find_for_stmt(program.get_function(scalar_full_range.__name__))
    assert isinstance(for_stmt.step, ir.Var)
    assert for_stmt.step.name == "step_0"


def test_positive_folded_for_step_is_accepted():
    @pl.jit(auto_mutex=False)
    def folded_step(n: pl.DT_INT64):
        step = 1 + 1
        for _ in pl.range(0, n, step):
            pass

    program, _ = folded_step.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    for_stmt = _find_for_stmt(program.get_function(folded_step.__name__))
    assert isinstance(for_stmt.step, ir.ConstInt)
    assert for_stmt.step.value == 2


def test_scalar_expression_as_stop():
    """Test pl.range(n * 2) where n is a DT_INT64 scalar parameter."""

    @pl.jit(auto_mutex=False)
    def scalar_expr_stop(n: pl.DT_INT64, x: pl.Tensor[[64], pl.DT_FP32]):
        y = x
        for _ in pl.range(n * 2):  # type: ignore[operator]
            y: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(x, 1.0)
        _test_result = y

    scalar_expr_stop_program, _ = scalar_expr_stop.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    scalar_expr_stop = scalar_expr_stop_program.get_function(scalar_expr_stop.__name__)

    assert isinstance(scalar_expr_stop, ir.Function)
    _assert_materialized_stop(scalar_expr_stop, ir.Mul)


def test_scalar_complex_expression_as_stop():
    """Test pl.range(n * 2 + 1) where n is a DT_INT64 scalar parameter."""

    @pl.jit(auto_mutex=False)
    def scalar_complex_expr(
        n: pl.DT_INT64, x: pl.Tensor[[64], pl.DT_FP32]
    ):
        y = x
        for _ in pl.range(n * 2 + 1):  # type: ignore[operator]
            y: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(x, 1.0)
        _test_result = y

    scalar_complex_expr_program, _ = scalar_complex_expr.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    scalar_complex_expr = scalar_complex_expr_program.get_function(scalar_complex_expr.__name__)

    assert isinstance(scalar_complex_expr, ir.Function)
    _assert_materialized_stop(scalar_complex_expr, ir.Add)


def test_scalar_floordiv_expression_as_stop():
    """Test pl.range(n // 4) where n is a DT_INT64 scalar parameter."""

    @pl.jit(auto_mutex=False)
    def scalar_floordiv_expr(
        n: pl.DT_INT64, x: pl.Tensor[[64], pl.DT_FP32]
    ):
        y = x
        for _ in pl.range(n // 4):  # type: ignore[operator]
            y: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(x, 1.0)
        _test_result = y

    scalar_floordiv_expr_program, _ = scalar_floordiv_expr.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    scalar_floordiv_expr = scalar_floordiv_expr_program.get_function(scalar_floordiv_expr.__name__)

    assert isinstance(scalar_floordiv_expr, ir.Function)
    _assert_materialized_stop(scalar_floordiv_expr, ir.FloorDiv)


def test_natural_while_loop():
    """Test natural while loop syntax."""

    @pl.jit(auto_mutex=False)
    def natural_while(n: pl.DT_INT64):
        x: pl.DT_INT64 = 0
        while x < n:
            x = x + 1
        _test_result = x

    natural_while_program, _ = natural_while.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    natural_while = natural_while_program.get_function(natural_while.__name__)

    assert isinstance(natural_while, ir.Function)
    assert natural_while.name == "natural_while"

    # Find the while statement
    while_stmt = _find_while_stmt(natural_while)
    assert isinstance(while_stmt, ir.WhileStmt)

    assert len(while_stmt.iter_args) == 1
    assert len(while_stmt.return_vars) == 1

    _assert_lowered_while_guard(while_stmt, ir.Lt)

    # Body should be present
    assert while_stmt.body is not None


def test_while_true_with_only_top_level_break_is_flattened():
    @pl.jit(auto_mutex=False)
    def flatten_once(value: pl.DT_INT64):
        result = value
        while True:
            result = result + 1
            break
        _test_result = result

    program, _ = flatten_once.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = program.get_function(flatten_once.__name__)

    assert not any(isinstance(stmt, ir.WhileStmt) for stmt in func.body.stmts)
    assert any(
        isinstance(stmt, ir.AssignStmt) and stmt.var.name.startswith("_test_result")
        for stmt in func.body.stmts
    )


def test_while_true_with_nested_break_is_not_flattened():
    @pl.jit(auto_mutex=False)
    def keep_loop(value: pl.DT_INT64):
        result = value
        while True:
            if result > 0:
                break
            result = result + 1
            break
        _test_result = result

    program, _ = keep_loop.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = program.get_function(keep_loop.__name__)

    assert isinstance(_find_while_stmt(func), ir.WhileStmt)


def test_while_flatten_ignores_jumps_owned_by_nested_loop():
    @pl.jit(auto_mutex=False)
    def flatten_outer(value: pl.DT_INT64):
        result = value
        while True:
            for i in pl.range(1):
                if i > 0:
                    break
                continue
            result = result + 1
            break
        _test_result = result

    program, _ = flatten_outer.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = program.get_function(flatten_outer.__name__)

    assert not any(isinstance(stmt, ir.WhileStmt) for stmt in func.body.stmts)
    assert any(isinstance(stmt, ir.ForStmt) for stmt in func.body.stmts)


def test_while_true_with_nested_continue_is_not_flattened():
    @pl.jit(auto_mutex=False)
    def keep_loop(flag: pl.DT_BOOL):
        while True:
            if flag:
                continue
            break

    program, _ = keep_loop.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = program.get_function(keep_loop.__name__)

    assert isinstance(_find_while_stmt(func), ir.WhileStmt)


def test_while_without_exit_candidate_has_empty_result():
    @pl.jit(auto_mutex=False)
    def loop_forever(value: pl.DT_INT64):
        while True:
            value = value + 1

    program, _ = loop_forever.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = program.get_function(loop_forever.__name__)
    while_stmt = _find_while_stmt(func)

    assert isinstance(while_stmt.iter_args[0].iterVar.type, ir.ScalarType)
    assert isinstance(while_stmt.return_vars[0].type, ir.NoneType)


def test_for_else_is_rejected_before_loop_parsing():
    @pl.jit(auto_mutex=False)
    def invalid_for_else(_jit_entry: pl.DT_INT64):
        for _ in pl.range(1):
            pass
        else:
            pass

    with pytest.raises(NotSupported, match="'for-else' is not supported"):
        invalid_for_else.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_while_else_is_rejected_before_loop_parsing():
    @pl.jit(auto_mutex=False)
    def invalid_while_else(value: pl.DT_INT64):
        while value > 0:
            break
        else:
            pass

    with pytest.raises(NotSupported, match="'while-else' is not supported"):
        invalid_while_else.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_natural_while_loop_with_initialization():
    """Test natural while loop with explicit initialization."""

    @pl.jit(auto_mutex=False)
    def natural_while_init(limit: pl.DT_INT64):
        counter: pl.DT_INT64 = 0
        sum_val: pl.DT_INT64 = 0
        while counter < limit:
            sum_val = sum_val + counter
            counter = counter + 1
        _test_result = sum_val

    natural_while_init_program, _ = natural_while_init.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    natural_while_init = natural_while_init_program.get_function(natural_while_init.__name__)

    assert isinstance(natural_while_init, ir.Function)

    # Find the while statement
    while_stmt = _find_while_stmt(natural_while_init)
    assert isinstance(while_stmt, ir.WhileStmt)

    _assert_lowered_while_guard(while_stmt, ir.Lt)

    assert len(while_stmt.iter_args) == 2
    assert len(while_stmt.return_vars) == 2


def test_while_loop_with_tensors():
    """Test while loop with tensor operations."""

    @pl.jit(auto_mutex=False)
    def while_tensors(n: pl.DT_INT64, x: pl.Tensor[[64], pl.DT_FP32]):
        i: pl.DT_INT64 = 0
        acc: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.create_tensor([64], dtype=pl.DT_FP32)
        while i < n:
            i = i + 1
            acc = pl.tensor.add(acc, x)
        _test_result = acc

    while_tensors_program, _ = while_tensors.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    while_tensors = while_tensors_program.get_function(while_tensors.__name__)

    assert isinstance(while_tensors, ir.Function)

    # Find the while statement
    while_stmt = _find_while_stmt(while_tensors)
    assert isinstance(while_stmt, ir.WhileStmt)

    # Body should contain assignments
    assert while_stmt.body is not None
    if isinstance(while_stmt.body, ir.SeqStmts):
        assert len(while_stmt.body.stmts) >= 1


def test_nested_while_loops():
    """Test nested while loops."""

    @pl.jit(auto_mutex=False)
    def nested_while(n: pl.DT_INT64):
        x: pl.DT_INT64 = 0
        while x < n:
            y: pl.DT_INT64 = 0
            while y < 3:
                y = y + 1
            x = x + 1
        _test_result = x

    nested_while_program, _ = nested_while.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    nested_while = nested_while_program.get_function(nested_while.__name__)

    assert isinstance(nested_while, ir.Function)

    # Find the outer while statement
    outer_while = _find_while_stmt(nested_while)
    assert isinstance(outer_while, ir.WhileStmt)

    # Find inner while statement in the outer while's body
    inner_while = None
    if isinstance(outer_while.body, ir.SeqStmts):
        for stmt in outer_while.body.stmts:
            if isinstance(stmt, ir.WhileStmt):
                inner_while = stmt
                break

    assert inner_while is not None, "Expected nested WhileStmt in outer while body"
    assert isinstance(inner_while, ir.WhileStmt)

    _assert_lowered_while_guard(outer_while, ir.Lt)
    _assert_lowered_while_guard(inner_while, ir.Lt)


def test_while_tensor_getval_condition_is_recomputed_in_body():
    """Tensor getval and its comparison must execute before the guard each iteration."""

    @pl.jit(auto_mutex=False)
    def while_tensor_condition(values: pl.Tensor[[8], pl.DT_INT32], limit: pl.DT_INT64):
        i: pl.DT_INT64 = 0
        while values[i] > 0:
            i = i + 1
            if i >= limit:
                break
        _test_result = i

    while_tensor_condition_program, _ = while_tensor_condition.to_kernel_def().parse_target_program(
        ir.SectionKind.Vector
    )
    while_tensor_condition = while_tensor_condition_program.get_function(while_tensor_condition.__name__)

    while_stmt = _find_while_stmt(while_tensor_condition)
    _assert_lowered_while_guard(while_stmt, ir.Gt)


def test_while_not_condition_uses_expression_parser():
    """The source-level not condition is materialized inside the lowered loop guard."""

    @pl.jit(auto_mutex=False)
    def while_not_condition(n: pl.DT_INT64):
        i: pl.DT_INT64 = 0
        while not i >= n:
            i = i + 1
        _test_result = i

    while_not_condition_program, _ = while_not_condition.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    while_not_condition = while_not_condition_program.get_function(while_not_condition.__name__)

    while_stmt = _find_while_stmt(while_not_condition)
    _assert_lowered_while_guard(while_stmt, ir.Not)


def test_while_with_multiple_updates():
    """Test while loop with multiple variable updates."""

    @pl.jit(auto_mutex=False)
    def while_multi_update(n: pl.DT_INT64):
        x: pl.DT_INT64 = 0
        y: pl.DT_INT64 = 1

        while x < n:
            x = x + 1
            y = y * 2

        _test_result = y

    while_multi_update_program, _ = while_multi_update.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    while_multi_update = while_multi_update_program.get_function(while_multi_update.__name__)

    assert isinstance(while_multi_update, ir.Function)

    # Find the while statement
    while_stmt = _find_while_stmt(while_multi_update)
    assert isinstance(while_stmt, ir.WhileStmt)

    # Body should contain multiple assignments
    assert while_stmt.body is not None
    if isinstance(while_stmt.body, ir.SeqStmts):
        # Should have at least 2 statements (x = x + 1, y = y * 2)
        assert len(while_stmt.body.stmts) >= 2


closure_allowed = [5, 6]


def test_in_operator_with_kernel_local_list():
    """``x in <list>`` accepts a list bound inside the function, not just a literal."""

    @pl.jit(auto_mutex=False)
    def func(n: pl.DT_INT64):
        allowed = [0, 1]
        total: pl.DT_INT64 = 0
        for i in pl.range(n):
            if i in allowed:
                total = total + 1
        _test_result = total

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)


def test_in_operator_kernel_local_list_shadows_closure():
    """A function-local list wins over a same-named closure list, as in Python."""

    @pl.jit(auto_mutex=False)
    def func(n: pl.DT_INT64):
        closure_allowed = [0, 1]  # noqa: F841 - shadows the module-level list
        total: pl.DT_INT64 = 0
        for i in pl.range(n):
            if i in closure_allowed:
                total = total + 1
        _test_result = total

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    text = str(func)
    # The local values (0, 1) drive the eq-chain; the closure values (5, 6) do not.
    assert "== 5" not in text and "== 6" not in text, text


def test_in_operator_with_closure_list_still_works():
    @pl.jit(auto_mutex=False)
    def func(n: pl.DT_INT64):
        total: pl.DT_INT64 = 0
        for i in pl.range(n):
            if i in closure_allowed:
                total = total + 1
        _test_result = total

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)


_FP32_INT32 = (pl.DT_FP32, pl.DT_INT32)


def _addr_for(dtype):
    """Plain Python helper the parser inlines when it appears in a kwarg."""
    return 0x8000 if dtype in _FP32_INT32 else 0x4000


def _addr_for_not_in(dtype):
    return 0x1000 if dtype not in _FP32_INT32 else 0x2000


def _tile_group_addr_ir(dtype, helper) -> str:
    @pl.jit(auto_mutex=True)
    def k(a: pl.Tensor[[128, 128], pl.DT_FP16]):
        tt = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        dt = dtype
        g = pl.make_tile_group(type=tt, addrs=helper(dt), mutex_ids=[0, 1])
        pl.load(g.next(), a, [0, 0])

    return str(k.to_kernel_def().parse_target_program(ir.SectionKind.Vector)[0])


def test_enum_in_tuple_folds_to_selected_branch():
    """``dtype in (pl.DT_FP32, ...)`` has no runtime form; it folds at parse time.

    An enum operand cannot be lowered to an IR constant, so the membership test
    must be decided while parsing and the enclosing ternary collapsed.
    """
    assert "target_memory=Mat" in _tile_group_addr_ir(pl.DT_FP32, _addr_for)
    assert "memref_addr=32768" in _tile_group_addr_ir(pl.DT_FP32, _addr_for)
    assert "memref_addr=16384" in _tile_group_addr_ir(pl.DT_FP16, _addr_for)


def test_enum_not_in_tuple_folds_to_selected_branch():
    assert "memref_addr=4096" in _tile_group_addr_ir(pl.DT_FP16, _addr_for_not_in)
    assert "memref_addr=8192" in _tile_group_addr_ir(pl.DT_FP32, _addr_for_not_in)
