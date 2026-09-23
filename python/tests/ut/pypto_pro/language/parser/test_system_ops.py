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
"""Tests for system operation DSL parsing and unified printing."""

import pypto_pro
from pypto_pro import ir
from pypto_pro._errors import InvalidArgument, InvalidOperation, InvalidType, InvalidVal, NotSupported, OutOfRange
import pypto_pro.language as pl
import pytest


def _const_int(value: int):
    return pypto_pro.ir.ConstInt(value, pypto_pro.ir.DataType.INDEX, pypto_pro.ir.Span.unknown())


def test_sync_src_print_style():
    """Test unified printing for pl.system.sync_src."""

    @pl.jit(auto_mutex=False)
    def main(x: pl.Tensor[[64], pl.DT_FP32]):
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        _test_result = x

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    printed = pypto_pro.ir.python_print(main)
    assert "ir.call @system.sync_src_dyn(" in printed
    assert "set_pipe" in printed
    assert "MTE2" in printed
    assert "wait_pipe" in printed
    assert "PipeType.V" in printed
    assert "0" in printed


def test_bar_all_print_style():
    """Test unified printing for pl.system.bar_all."""

    @pl.jit(auto_mutex=False)
    def main(x: pl.Tensor[[64], pl.DT_FP32]):
        pl.system.bar_all()
        _test_result = x

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    printed = pypto_pro.ir.python_print(main)
    assert "ir.call @system.bar_all()" in printed


def test_multiple_system_ops_print_style():
    """Test unified printing with multiple system ops in a single function."""

    @pl.jit(auto_mutex=False)
    def main(x: pl.Tensor[[64], pl.DT_FP32]):
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        pl.system.bar_all()
        _test_result = x

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    printed = pypto_pro.ir.python_print(main)
    assert "ir.call @system.sync_src_dyn(" in printed
    assert "ir.call @system.sync_dst_dyn(" in printed
    assert "ir.call @system.bar_all()" in printed


def test_sync_with_different_pipe_types():
    """Test sync ops with various PipeType enum values."""

    @pl.jit(auto_mutex=False)
    def main(x: pl.Tensor[[64], pl.DT_FP32]):
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.S, event_id=2)
        _test_result = x

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Cube)
    main = main_program.get_function(main.__name__)

    printed = pypto_pro.ir.python_print(main)
    assert "ir.PipeType.MTE1" in printed
    assert "ir.PipeType.M" in printed
    assert "ir.PipeType.MTE3" in printed
    assert "ir.PipeType.S" in printed


def test_sync_src_pipe_kwarg_rejects_plain_integer():
    with pytest.raises(InvalidVal, match="'set_pipe' expects an enum value"):

        @pl.jit(auto_mutex=False)
        def func(_jit_entry: pl.DT_INT64):
            pl.system.sync_src(set_pipe=1, wait_pipe=pl.PipeType.V, event_id=0)


        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_sync_dst_pipe_kwarg_rejects_plain_integer():
    with pytest.raises(InvalidVal, match="'wait_pipe' expects an enum value"):

        @pl.jit(auto_mutex=False)
        def func(_jit_entry: pl.DT_INT64):
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=5, event_id=0)


        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_sync_pipe_kwarg_rejects_other_enum_type():
    with pytest.raises(InvalidType, match="set_pipe must be a PipeType"):

        @pl.jit(auto_mutex=False)
        def func(_jit_entry: pl.DT_INT64):
            pl.system.sync_src(
                set_pipe=pl.SyncCoreType.AIV_ONLY,
                wait_pipe=pl.PipeType.V,
                event_id=0,
            )

        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_mutex_lock_pipe_kwarg_rejects_plain_integer():
    with pytest.raises(InvalidVal, match="'pipe' expects an enum value"):

        @pl.jit(auto_mutex=False)
        def func(_jit_entry: pl.DT_INT64):
            pl.system.mutex_lock(pipe=5, mutex_id=0)


        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


@pytest.mark.parametrize("builder", [pl.system.mutex_lock, pl.system.mutex_unlock])
@pytest.mark.parametrize("pipe", [1, pl.SyncCoreType.AIV_ONLY])
def test_mutex_ir_builder_requires_pipe_type(builder, pipe):
    with pytest.raises(TypeError, match="pipe must be a PipeType"):
        builder(pipe=pipe, mutex_id=0)


def test_sync_all_core_type_rejects_plain_integer():
    with pytest.raises(InvalidVal, match="'core_type' expects an enum value"):

        @pl.jit(auto_mutex=False)
        def func(_jit_entry: pl.DT_INT64):
            pl.system.sync_all(core_type=2)


        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


@pytest.mark.parametrize("event_id", [True, False])
def test_sync_event_id_rejects_bool(event_id):
    @pl.jit(auto_mutex=False)
    def func(_jit_entry: pl.DT_INT64):
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2,
            wait_pipe=pl.PipeType.V,
            event_id=event_id,
        )

    with pytest.raises(InvalidType, match="event_id must be .*integer scalar expression"):
        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_sync_parser_normalizes_python_int_id():
    @pl.jit(auto_mutex=False)
    def func(_jit_entry: pl.DT_INT64):
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2,
            wait_pipe=pl.PipeType.V,
            event_id=3,
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.MTE2,
            wait_pipe=pl.PipeType.V,
            event_id=3,
        )

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)
    calls = [stmt.expr for stmt in func.body.stmts if isinstance(stmt, pypto_pro.ir.EvalStmt)]

    assert [call.name for call in calls] == ["system.sync_src_dyn", "system.sync_dst_dyn"]
    for sync_call in calls:
        assert isinstance(sync_call.args[0], pypto_pro.ir.ConstInt)
        assert sync_call.args[0].value == 3


@pytest.mark.parametrize("builder", [pl.system.mutex_lock, pl.system.mutex_unlock])
@pytest.mark.parametrize("mutex_id", [True, False])
def test_mutex_id_rejects_bool(builder, mutex_id):
    with pytest.raises(TypeError, match="mutex_id must be a Python int or an integer scalar expression"):
        builder(pipe=pl.PipeType.MTE2, mutex_id=mutex_id)


@pytest.mark.parametrize("builder", [pl.system.mutex_lock, pl.system.mutex_unlock])
def test_mutex_ir_builder_normalizes_python_int_id(builder):
    call = builder(pipe=pl.PipeType.MTE2, mutex_id=0)

    assert isinstance(call.args[0], pypto_pro.ir.ConstInt)
    assert call.args[0].value == 0


@pytest.mark.parametrize("event_id", [-1, 8])
def test_sync_static_event_id_range_is_validated_by_parser(event_id):
    @pl.jit(auto_mutex=False)
    def func(_jit_entry: pl.DT_INT64):
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2,
            wait_pipe=pl.PipeType.V,
            event_id=event_id,
        )

    with pytest.raises(OutOfRange, match=r"event_id must be in \[0, 7\]"):
        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_sync_rejects_equal_pipes_in_parser():
    @pl.jit(auto_mutex=False)
    def func(_jit_entry: pl.DT_INT64):
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2,
            wait_pipe=pl.PipeType.MTE2,
            event_id=0,
        )

    with pytest.raises(InvalidOperation, match="set_pipe and wait_pipe must differ"):
        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_sync_rejects_all_set_pipe_in_parser():
    @pl.jit(auto_mutex=False)
    def func(_jit_entry: pl.DT_INT64):
        pl.system.sync_src(
            set_pipe=pl.PipeType.ALL,
            wait_pipe=pl.PipeType.V,
            event_id=0,
        )

    with pytest.raises(InvalidArgument, match="set_pipe must identify one concrete pipe"):
        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_sync_rejects_all_wait_pipe_in_parser():
    @pl.jit(auto_mutex=False)
    def func(_jit_entry: pl.DT_INT64):
        pl.system.sync_dst(
            set_pipe=pl.PipeType.MTE2,
            wait_pipe=pl.PipeType.ALL,
            event_id=0,
        )

    with pytest.raises(InvalidArgument, match="wait_pipe must identify one concrete pipe"):
        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_sync_rejects_aic_only_path_in_vector_section():
    @pl.jit(auto_mutex=False)
    def func(_jit_entry: pl.DT_INT64):
        with pl.section_vector():
            pl.system.sync_src(
                set_pipe=pl.PipeType.MTE1,
                wait_pipe=pl.PipeType.M,
                event_id=0,
            )

    with pytest.raises(NotSupported, match="unsupported 3510 synchronization path on AIV"):
        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_sync_rejects_aiv_only_path_in_cube_section():
    @pl.jit(auto_mutex=False)
    def func(_jit_entry: pl.DT_INT64):
        with pl.section_cube():
            pl.system.sync_dst(
                set_pipe=pl.PipeType.MTE2,
                wait_pipe=pl.PipeType.V,
                event_id=0,
            )

    with pytest.raises(NotSupported, match="unsupported 3510 synchronization path on AIC"):
        func.to_kernel_def().parse_target_program(ir.SectionKind.Cube)


def test_sync_rejects_unsupported_aic_pipe_path():
    @pl.jit(auto_mutex=False)
    def func(_jit_entry: pl.DT_INT64):
        with pl.section_cube():
            pl.system.sync_src(
                set_pipe=pl.PipeType.MTE1,
                wait_pipe=pl.PipeType.S,
                event_id=0,
            )

    with pytest.raises(NotSupported, match="unsupported 3510 synchronization path on AIC"):
        func.to_kernel_def().parse_target_program(ir.SectionKind.Cube)


@pytest.mark.parametrize("builder", [pl.system.mutex_lock, pl.system.mutex_unlock])
@pytest.mark.parametrize("mutex_id", [-1, 32])
def test_mutex_static_id_range_is_validated_by_frontend(builder, mutex_id):
    with pytest.raises(OutOfRange, match=r"mutex_id must be in \[0, 31\]"):
        builder(pipe=pl.PipeType.MTE2, mutex_id=_const_int(mutex_id))


@pytest.mark.parametrize("builder", [pl.system.mutex_lock, pl.system.mutex_unlock])
def test_mutex_rejects_all_pipe_in_frontend(builder):
    with pytest.raises(ValueError, match="pipe must identify one concrete pipe"):
        builder(pipe=pl.PipeType.ALL, mutex_id=_const_int(0))


@pytest.mark.parametrize("builder", [pl.system.mutex_lock, pl.system.mutex_unlock])
def test_manual_mutex_has_no_auto_candidate_metadata(builder):
    call = builder(pipe=pl.PipeType.MTE2, mutex_id=_const_int(8))

    assert "auto_mutex" not in call.kwargs
    assert "max_mutex_id" not in str(call)
    assert "mutex_ids" not in str(call)


def test_sync_all_rejects_workspaces_in_frontend():
    with pytest.raises(TypeError, match="positional argument"):
        pl.system.sync_all([0])


def test_sync_all_rejects_mode_in_frontend():
    with pytest.raises(TypeError, match="unexpected keyword argument 'mode'"):
        pl.system.sync_all(mode=0)


def test_sync_all_rejects_workspaces_in_parser():
    @pl.jit(auto_mutex=False)
    def func(_jit_entry: pl.DT_INT64):
        pl.system.sync_all([0])

    with pytest.raises(InvalidArgument, match="does not accept positional arguments"):
        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_complex_integer_event_id_expression_is_accepted():
    @pl.jit(auto_mutex=False)
    def func(base: pl.DT_INT64):
        event_id = (base * 3 + 1) % 8
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2,
            wait_pipe=pl.PipeType.V,
            event_id=event_id,
        )


    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)

    func = func_program.get_function(func.__name__)

    assert "system.sync_src_dyn" in str(func)


def test_constant_integer_event_id_expression_is_folded_to_operand():
    @pl.jit(auto_mutex=False)
    def func(_jit_entry: pl.DT_INT64):
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2,
            wait_pipe=pl.PipeType.V,
            event_id=(1 + 2) % 8,
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.MTE2,
            wait_pipe=pl.PipeType.V,
            event_id=(1 + 2) % 8,
        )


    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)

    func = func_program.get_function(func.__name__)

    calls = [stmt.expr for stmt in func.body.stmts if isinstance(stmt, pypto_pro.ir.EvalStmt)]
    assert [call.name for call in calls] == ["system.sync_src_dyn", "system.sync_dst_dyn"]
    for call in calls:
        assert isinstance(call.args[0], pypto_pro.ir.ConstInt)
        assert call.args[0].value == 3


def test_constant_integer_mutex_id_expression_is_folded_to_operand():
    @pl.jit(auto_mutex=False)
    def func(_jit_entry: pl.DT_INT64):
        pl.system.mutex_lock(
            pipe=pl.PipeType.MTE2,
            mutex_id=(5 * 2 + 1) % 32,
        )
        pl.system.mutex_unlock(
            pipe=pl.PipeType.MTE2,
            mutex_id=(5 * 2 + 1) % 32,
        )


    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)

    func = func_program.get_function(func.__name__)

    calls = [stmt.expr for stmt in func.body.stmts if isinstance(stmt, pypto_pro.ir.EvalStmt)]
    assert [call.name for call in calls] == ["system.mutex_lock_dyn", "system.mutex_unlock_dyn"]
    for call in calls:
        assert isinstance(call.args[0], pypto_pro.ir.ConstInt)
        assert call.args[0].value == 11


def test_bool_event_id_expression_is_rejected_by_parser():
    with pytest.raises(InvalidType, match="event_id must be an integer scalar expression"):

        @pl.jit(auto_mutex=False)
        def func(event_id: pl.DT_BOOL):
            pl.system.sync_src(
                set_pipe=pl.PipeType.MTE2,
                wait_pipe=pl.PipeType.V,
                event_id=event_id,
            )


        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_bool_mutex_id_expression_is_rejected_by_parser():
    with pytest.raises(InvalidType, match="mutex_id must be an integer scalar expression"):

        @pl.jit(auto_mutex=False)
        def func(mutex_id: pl.DT_BOOL):
            pl.system.mutex_lock(
                pipe=pl.PipeType.MTE2,
                mutex_id=mutex_id,
            )


        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_complex_float_event_id_expression_is_rejected_by_parser():
    with pytest.raises(InvalidType, match="event_id must be an integer scalar expression"):

        @pl.jit(auto_mutex=False)
        def func(base: pl.DT_INT64):
            event_id = (base * 3 + 1) / 2
            pl.system.sync_src(
                set_pipe=pl.PipeType.MTE2,
                wait_pipe=pl.PipeType.V,
                event_id=event_id,
            )


        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_complex_float_mutex_id_expression_is_rejected_by_parser():
    with pytest.raises(InvalidType, match="mutex_id must be an integer scalar expression"):

        @pl.jit(auto_mutex=False)
        def func(base: pl.DT_INT64):
            mutex_id = (base * 3 + 1) / 2
            pl.system.mutex_lock(
                pipe=pl.PipeType.MTE2,
                mutex_id=mutex_id,
            )


        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_control_flow_integer_event_id_expression_is_accepted():
    @pl.jit(auto_mutex=False)
    def func(base: pl.DT_INT64):
        event_id = base % 8
        if base > 4:
            event_id = (base * 3 + 1) % 8
        else:
            event_id = (base * 5 + 1) % 8
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2,
            wait_pipe=pl.PipeType.V,
            event_id=event_id,
        )


    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)

    func = func_program.get_function(func.__name__)

    assert "system.sync_src_dyn" in str(func)


def test_control_flow_float_event_id_expression_is_rejected_by_parser():
    with pytest.raises(InvalidType, match="event_id must be an integer scalar expression"):

        @pl.jit(auto_mutex=False)
        def func(base: pl.DT_INT64):
            event_id = (base + 1) / 2
            if base > 4:
                event_id = (base * 3 + 1) / 2
            else:
                event_id = (base * 5 + 1) / 2
            pl.system.sync_src(
                set_pipe=pl.PipeType.MTE2,
                wait_pipe=pl.PipeType.V,
                event_id=event_id,
            )


        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


@pytest.mark.parametrize("builder", [pl.system.mutex_lock, pl.system.mutex_unlock])
@pytest.mark.parametrize(
    ("kwarg", "value"),
    [("mode", 0), ("max_mutex_id", 2), ("mutex_ids", [0, 1])],
)
def test_mutex_rejects_non_public_kwargs(builder, kwarg, value):
    with pytest.raises(TypeError, match=rf"unexpected keyword argument '{kwarg}'"):
        builder(pipe=pl.PipeType.MTE2, mutex_id=_const_int(0), **{kwarg: value})


def test_dcci_gm_tensor_with_offset_print_style():
    """Test unified printing for pl.system.dcci with GM tensor and offset."""

    @pl.jit(auto_mutex=False)
    def main(x: pl.Tensor[[16, 16], pl.DT_FP32]):
        pl.system.dcci(x, [0, 0], cache_line=pl.CacheLine.SINGLE_CACHE_LINE, dst=pl.DcciDst.CACHELINE_OUT)
        _test_result = x

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    printed = pypto_pro.ir.python_print(main)
    assert "ir.call @system.dcci(" in printed
    assert "cache_line=" in printed
    assert "dst=" in printed


def test_dcci_gm_tensor_rejects_float_scalar_offset():
    """Test pl.system.dcci rejects float scalar offset for GM tensor."""

    with pytest.raises(InvalidType, match="scalar integer element offset"):

        @pl.jit(auto_mutex=False)
        def main(x: pl.Tensor[[16, 16], pl.DT_FP32]):
            pl.system.dcci(x, 1.5)
            _test_result = x

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_dcci_gm_tensor_rejects_float_tuple_offset():
    """Test pl.system.dcci rejects float tuple element offset for GM tensor."""

    with pytest.raises(InvalidType, match="per-dimension list/tuple"):

        @pl.jit(auto_mutex=False)
        def main(x: pl.Tensor[[16, 16], pl.DT_FP32]):
            pl.system.dcci(x, [1.5, 0])
            _test_result = x

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_dcci_with_tuple_offset_print_style():
    """Test unified printing for pl.system.dcci with tuple offset."""

    @pl.jit(auto_mutex=False)
    def main(x: pl.Tensor[[16, 16], pl.DT_FP32]):
        pl.system.dcci(x, (1, 2))
        _test_result = x

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    printed = pypto_pro.ir.python_print(main)
    assert "ir.call @system.dcci(" in printed
