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
"""Unit tests for IR Builder."""

from pypto_pro import DataType, ir
from pypto_pro._errors import InvalidType
from pypto_pro.ir import IRBuilder
import pytest


def _single_stmt(stmt: ir.Stmt) -> ir.Stmt:
    assert isinstance(stmt, ir.SeqStmts)
    assert len(stmt.stmts) == 1
    return stmt.stmts[0]


def test_simple_function_with_auto_span():
    """Test building a simple function with automatic span capture."""
    ib = IRBuilder()

    with ib.function("my_func") as f:
        x = f.param("x", ir.ScalarType(DataType.INT64))
        y = f.param("y", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        # Build body
        result = ib.var("result", ir.ScalarType(DataType.INT64))
        add_expr = ir.Add(x, y, DataType.INT64, ir.Span.unknown())
        ib.assign(result, add_expr)

    func = f.get_result()

    assert func is not None
    assert func.name == "my_func"
    assert len(func.params) == 2
    assert len(func.return_types) == 1
    assert func.params[0].name == "x_0"
    assert func.params[1].name == "y_0"
    assert func.body is not None


def test_function_with_explicit_span():
    """Test building a function with explicit span."""
    ib = IRBuilder()
    my_span = ir.Span("test.py", 10, 1)

    with ib.function("explicit_func", span=my_span) as f:
        _x = f.param("x", ir.ScalarType(DataType.INT32), span=my_span)
        f.return_type(ir.ScalarType(DataType.INT32))

    func = f.get_result()

    assert func is not None
    assert func.name == "explicit_func"
    assert func.span.filename == "test.py"
    assert func.span.begin_line == 10


def test_function_with_multiple_statements():
    """Test function with multiple statements in body."""
    ib = IRBuilder()

    with ib.function("multi_stmt") as f:
        x = f.param("x", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        a = ib.var("a", ir.ScalarType(DataType.INT64))
        b = ib.var("b", ir.ScalarType(DataType.INT64))

        ib.assign(a, x)
        const_one = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
        add_expr = ir.Add(a, const_one, DataType.INT64, ir.Span.unknown())
        ib.assign(b, add_expr)

    func = f.get_result()

    assert func is not None
    assert isinstance(func.body, ir.SeqStmts)
    assert len(func.body.stmts) == 2


def test_var_name_is_unique_within_function():
    ib = IRBuilder()

    with ib.function("reassign") as f:
        param = f.param("x", ir.ScalarType(DataType.INT64))
        first = ib.var("x", ir.ScalarType(DataType.INT64))
        second = ib.var(first.name, ir.ScalarType(DataType.INT64))

    assert param.name == "x_0"
    assert first.name == "x_1"
    assert second.name == "x_2"


def test_var_names_reset_for_each_function():
    ib = IRBuilder()

    with ib.function("first"):
        first = ib.var("x", ir.ScalarType(DataType.INT64))
    with ib.function("second"):
        second = ib.var("x", ir.ScalarType(DataType.INT64))

    assert first.name == "x_0"
    assert second.name == "x_0"


def test_nested_function_error():
    """Test that nested functions raise an error."""
    ib = IRBuilder()

    with pytest.raises(RuntimeError, match="Cannot begin function"):
        with ib.function("outer") as f:
            f.return_type(ir.ScalarType(DataType.INT64))

            # Try to nest another function - should fail
            with ib.function("inner") as _f2:
                pass


def test_function_type_default():
    """Test that function_type defaults to Opaque."""
    ib = IRBuilder()

    with ib.function("test_func", func_type=ir.FunctionType.Opaque) as f:
        f.return_type(ir.ScalarType(DataType.INT64))

    func = f.get_result()

    assert func is not None
    assert func.func_type == ir.FunctionType.Opaque


def test_function_type_explicit_incore():
    """Test explicit INCORE function_type."""
    ib = IRBuilder()

    with ib.function("test_kernel", func_type=ir.FunctionType.InCore) as f:
        f.return_type(ir.TileType([16, 16], DataType.FP32))

    func = f.get_result()

    assert func is not None
    assert func.func_type == ir.FunctionType.InCore


def test_in_function_query():
    """Test InFunction query."""
    ib = IRBuilder()

    assert not ib.in_function()

    with ib.function("test") as f:
        assert ib.in_function()
        f.return_type(ir.ScalarType(DataType.INT64))

    assert not ib.in_function()


def test_in_loop_query():
    """Test InLoop query."""
    ib = IRBuilder()

    with ib.function("test") as f:
        f.return_type(ir.ScalarType(DataType.INT64))

        assert not ib.in_loop()

        i = ib.var("i", ir.ScalarType(DataType.INT64))

        with ib.for_loop(i, 0, 10, 1):
            assert ib.in_loop()

        assert not ib.in_loop()


def test_in_if_query():
    """Test InIf query."""
    ib = IRBuilder()

    with ib.function("test") as f:
        f.return_type(ir.ScalarType(DataType.INT64))

        assert not ib.in_if()

        with ib.if_stmt(1):
            assert ib.in_if()

        assert not ib.in_if()


def test_let_with_inferred_type():
    """Test basic let() usage with type inference from expression."""
    ib = IRBuilder()

    with ib.function("let_test") as f:
        f.return_type(ir.ScalarType(DataType.INT64))

        # Create an expression with known type
        const = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())

        # let() should infer the type from the expression
        x = ib.let("x", const)

        assert x.name == "x_0"
        assert isinstance(x.type, ir.ScalarType)
        assert x.type.dtype == DataType.INT64

    func = f.get_result()
    assert func is not None


def test_let_with_type_validation():
    """Test let() with explicit type that matches inferred type."""
    ib = IRBuilder()

    with ib.function("validation_test") as f:
        f.return_type(ir.ScalarType(DataType.INT64))

        # Create an expression
        const = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())

        # Provide matching type for validation
        explicit_type = ir.ScalarType(DataType.INT64)
        x = ib.let("x", const, var_type=explicit_type)

        assert x.name == "x_0"
        assert isinstance(x.type, ir.ScalarType)
        assert x.type.dtype == DataType.INT64

    func = f.get_result()
    assert func is not None


def test_let_with_compatible_type_override():
    """Test that let() allows type override with same-kind type (e.g., adding memref)."""
    ib = IRBuilder()

    with ib.function("override_test") as f:
        f.return_type(ir.TensorType([64], DataType.FP32))

        # Create a tensor expression
        param = f.param("x", ir.TensorType([64], DataType.FP32))

        # Override with same-kind type that includes memref
        span = ir.Span.unknown()
        memref = ir.MemRef(ir.MemorySpace.DDR, ir.ConstInt(0, DataType.INT64, span), 256, 0)
        override_type = ir.TensorType([64], DataType.FP32, memref)

        x = ib.let("x", param, var_type=override_type)
        assert isinstance(x.type, ir.TensorType)
        assert x.type.memref is not None


def test_let_with_incompatible_type_override():
    """Test that let() rejects incompatible type overrides (different type kinds)."""
    ib = IRBuilder()

    with pytest.raises(TypeError, match="incompatible"):
        with ib.function("mismatch_test") as f:
            f.return_type(ir.ScalarType(DataType.INT64))

            # Create INT64 scalar but try to override with TensorType
            const = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())
            wrong_type = ir.TensorType([64], DataType.FP32)

            ib.let("x", const, var_type=wrong_type)


def test_let_with_scalar_value():
    """Test let() with int/float values that get normalized."""
    ib = IRBuilder()

    with ib.function("scalar_test") as f:
        f.return_type(ir.ScalarType(DataType.INT64))

        # let() should handle int values via _normalize_expr
        x = ib.let("x", 42)

        assert x.name == "x_0"
        # Type should be inferred from the normalized expression
        assert isinstance(x.type, ir.ScalarType)

    func = f.get_result()
    assert func is not None


def test_let_with_tensor_expr():
    """Test let() with tensor operation result."""
    ib = IRBuilder()

    with ib.function("tensor_test") as f:
        f.return_type(ir.ScalarType(DataType.INT64))

        # Create an op whose result carries a TensorType
        src = ir.Var("src", ir.TensorType([4, 8], DataType.FP32), ir.Span.unknown())
        tensor_view = ir.op.ptr.make_tensor(src, [4, 8], [8, 1], dtype=DataType.FP32)

        # let() should infer TensorType from the op's result type
        t = ib.let("t", tensor_view)

        assert t.name == "t_0"
        assert isinstance(t.type, ir.TensorType)
        assert t.type.dtype == DataType.FP32

    func = f.get_result()
    assert func is not None


def test_let_with_binary_expr():
    """Test let() with binary expression result."""
    ib = IRBuilder()

    with ib.function("binary_test") as f:
        x = f.param("x", ir.ScalarType(DataType.INT64))
        y = f.param("y", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        # Create binary expression
        add_expr = ir.Add(x, y, DataType.INT64, ir.Span.unknown())

        # let() should infer type from Add expression
        result = ib.let("result", add_expr)

        assert result.name == "result_0"
        assert isinstance(result.type, ir.ScalarType)
        assert result.type.dtype == DataType.INT64

    func = f.get_result()
    assert func is not None


def test_let_with_explicit_span():
    """Test let() with explicit span parameter."""
    ib = IRBuilder()
    my_span = ir.Span("test.py", 100, 5)

    with ib.function("span_test") as f:
        f.return_type(ir.ScalarType(DataType.INT64))

        const = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())
        x = ib.let("x", const, span=my_span)

        assert x.name == "x_0"
        assert x.span.filename == "test.py"
        assert x.span.begin_line == 100

    func = f.get_result()
    assert func is not None


def test_iter_arg_with_inferred_type():
    """Test iter_arg with type inference from init_value."""
    ib = IRBuilder()

    with ib.function("iter_test") as f:
        f.return_type(ir.ScalarType(DataType.INT64))

        i = ib.var("i", ir.ScalarType(DataType.INT64))

        with ib.for_loop(i, 0, 10, 1) as loop:
            # Type should be inferred from initial value
            sum_iter = loop.iter_arg("sum", 0)
            # Must have matching return_var
            _ = loop.return_var("sum_final")

            assert sum_iter.name == "sum"
            assert isinstance(sum_iter.iterVar.type, ir.ScalarType)
            assert sum_iter.iterVar.type.dtype == DataType.INT64

    func = f.get_result()
    assert func is not None


def test_iter_arg_with_type_validation():
    """Test iter_arg with explicit type that matches inferred type."""
    ib = IRBuilder()

    with ib.function("iter_validation_test") as f:
        f.return_type(ir.ScalarType(DataType.INT64))

        i = ib.var("i", ir.ScalarType(DataType.INT64))

        with ib.for_loop(i, 0, 10, 1) as loop:
            # Provide matching type for validation
            explicit_type = ir.ScalarType(DataType.INT64)
            sum_iter = loop.iter_arg("sum", 0, iter_type=explicit_type)
            # Must have matching return_var
            _ = loop.return_var("sum_final")

            assert sum_iter.name == "sum"
            assert isinstance(sum_iter.iterVar.type, ir.ScalarType)
            assert sum_iter.iterVar.type.dtype == DataType.INT64

    func = f.get_result()
    assert func is not None


def test_iter_arg_with_type_mismatch():
    """Test that iter_arg raises error when explicit type doesn't match inferred type."""
    ib = IRBuilder()

    with pytest.raises(InvalidType, match="Type mismatch"):
        with ib.function("iter_mismatch_test") as f:
            f.return_type(ir.ScalarType(DataType.INT64))

            i = ib.var("i", ir.ScalarType(DataType.INT64))

            with ib.for_loop(i, 0, 10, 1) as loop:
                # Wrong type - init_value is INT64 but we provide FP32
                wrong_type = ir.ScalarType(DataType.FP32)
                loop.iter_arg("sum", 0, iter_type=wrong_type)


def test_return_var_with_inferred_type():
    """Test return_var with type inference from corresponding iter_arg."""
    ib = IRBuilder()

    with ib.function("return_var_test") as f:
        f.return_type(ir.ScalarType(DataType.INT64))

        i = ib.var("i", ir.ScalarType(DataType.INT64))

        with ib.for_loop(i, 0, 10, 1) as loop:
            _ = loop.iter_arg("sum", 0)
            # Type should be inferred from corresponding iter_arg
            sum_final = loop.return_var("sum_final")

            assert sum_final.name == "sum_final"
            assert isinstance(sum_final.type, ir.ScalarType)
            assert sum_final.type.dtype == DataType.INT64

    func = f.get_result()
    assert func is not None


def test_return_var_with_multiple_iter_args():
    """Test return_var inference with multiple iter_args."""
    ib = IRBuilder()

    with ib.function("multi_return_var_test") as f:
        f.return_type(ir.ScalarType(DataType.INT64))

        i = ib.var("i", ir.ScalarType(DataType.INT64))

        with ib.for_loop(i, 0, 10, 1) as loop:
            # Multiple iter_args with different types
            _ = loop.iter_arg("sum", 0)  # INT64
            _ = loop.iter_arg("count", 1)  # INT64

            # Return vars should match iter_args by index
            sum_final = loop.return_var("sum_final")  # Should be INT64
            count_final = loop.return_var("count_final")  # Should be INT64

            assert isinstance(sum_final.type, ir.ScalarType)
            assert isinstance(count_final.type, ir.ScalarType)
            assert sum_final.type.dtype == DataType.INT64
            assert count_final.type.dtype == DataType.INT64

    func = f.get_result()
    assert func is not None


def test_return_var_explicit_type_validation():
    """Test return_var with explicit type that matches inferred type."""
    ib = IRBuilder()

    with ib.function("return_var_validation_test") as f:
        f.return_type(ir.ScalarType(DataType.INT64))

        i = ib.var("i", ir.ScalarType(DataType.INT64))

        with ib.for_loop(i, 0, 10, 1) as loop:
            _ = loop.iter_arg("sum", 0)
            # Provide explicit type that matches iter_arg type
            explicit_type = ir.ScalarType(DataType.INT64)
            sum_final = loop.return_var("sum_final", var_type=explicit_type)

            assert isinstance(sum_final.type, ir.ScalarType)
            assert sum_final.type.dtype == DataType.INT64

    func = f.get_result()
    assert func is not None
