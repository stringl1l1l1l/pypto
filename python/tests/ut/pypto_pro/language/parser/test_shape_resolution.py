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
"""Unit tests for TypeResolver."""
# DSL function bodies are parsed as AST, not executed -suppress pyright errors
# from type-checking annotations and kwargs inside parsed function bodies.
import ast
from typing import TYPE_CHECKING, Any

from pypto_pro import DataType, ir
from pypto_pro._errors import InvalidArgument, InvalidType
import pypto_pro.language as pl
from pypto_pro.language.parser._expr_evaluator import ExprEvaluator
from pypto_pro.language.parser._type_resolver import TypeResolver
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable


def _make_resolver(
    closure_vars: dict | None = None, scope_lookup: "Callable[[str], Any | None] | None" = None
) -> TypeResolver:
    """Create a TypeResolver with ExprEvaluator from closure_vars."""
    ev = ExprEvaluator(closure_vars=closure_vars or {})
    return TypeResolver(expr_evaluator=ev, scope_lookup=scope_lookup)


@pytest.mark.parametrize(
    "expr, closure_vars, expected",
    [
        ("[rows, cols]", {"rows": 128, "cols": 64}, [128, 64]),
        ("[rows, 64]", {"rows": 128}, [128, 64]),
        ("(rows, cols)", {"rows": 128, "cols": 64}, [128, 64]),
        ("shape", {"shape": [128, 64]}, [128, 64]),
        ("shape", {"shape": (32, 64)}, [32, 64]),
        ("[base * 2, base]", {"base": 64}, [128, 64]),
        ("[len(data), 64]", {"data": [1, 2, 3, 4]}, [4, 64]),
        ("[dims[0], dims[1]]", {"dims": [128, 64, 32]}, [128, 64]),
    ],
)
def test_parse_shape_supported_exprs(expr, closure_vars, expected):
    """parse_shape resolves supported literals, variables and simple expressions."""
    resolver = _make_resolver(closure_vars=closure_vars)
    node = ast.parse(expr, mode="eval").body
    shape = resolver.parse_shape(node)
    assert shape == expected


def test_resolve_dynamic_policy_shape():
    """DYNAMIC creates a deterministic parameter/axis ir.Var."""
    resolver = _make_resolver(closure_vars={"pl": pl})
    node = ast.parse("pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32]", mode="eval").body
    tensor_type = resolver.resolve_param_type(node, parameter_name="x")
    assert tensor_type.shape[0].name == "__pypto_dyn_x_0"
    assert tensor_type.shape[1].name == "__pypto_dyn_x_1"


def test_resolve_dynamic_policy_and_literal_mixed():
    resolver = _make_resolver(closure_vars={"pl": pl})
    node = ast.parse("pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP32]", mode="eval").body
    tensor_type = resolver.resolve_param_type(node, parameter_name="x")
    assert tensor_type.shape[0].name == "__pypto_dyn_x_0"
    assert tensor_type.shape[1].value == 128


def test_resolve_dynamic_policy_and_int_variable_mixed():
    resolver = _make_resolver(closure_vars={"pl": pl, "cols": 64})
    node = ast.parse("pl.Tensor[[pl.DYNAMIC, cols], pl.DT_FP32]", mode="eval").body
    tensor_type = resolver.resolve_param_type(node, parameter_name="x")
    assert isinstance(tensor_type.shape[0], ir.Var)
    assert tensor_type.shape[1].value == 64


def test_dynamic_policy_has_int64_scalar_type():
    resolver = _make_resolver(closure_vars={"pl": pl})
    node = ast.parse("pl.Tensor[[pl.DYNAMIC], pl.DT_FP32]", mode="eval").body
    tensor_type = resolver.resolve_param_type(node, parameter_name="x")
    assert isinstance(tensor_type.shape[0].type, ir.ScalarType)
    assert tensor_type.shape[0].type.dtype == DataType.INT64


def test_parse_shape_with_scope_variable():
    """Scalar variable from parser scope used in inline annotation."""
    mock_var = ir.Var("q_tile", ir.ScalarType(DataType.UINT64), ir.Span.unknown())
    scope = {"q_tile": mock_var}
    resolver = _make_resolver(scope_lookup=lambda name: scope.get(name))
    node = ast.parse("[q_tile, 128]", mode="eval").body
    shape = resolver.parse_shape(node)
    assert shape[0] is mock_var
    assert shape[1] == 128


def test_closure_vars_take_precedence_over_scope():
    """Closure variables are checked before parser scope."""
    scope = {"x": ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())}
    resolver = _make_resolver(closure_vars={"x": 42}, scope_lookup=lambda name: scope.get(name))
    node = ast.parse("[x]", mode="eval").body
    shape = resolver.parse_shape(node)
    assert shape == [42]


def test_to_ir_shape_converts_ints_only_when_mixed_with_exprs():
    """Pure int lists stay plain; mixed lists convert ints to ConstInt."""
    resolver = _make_resolver()
    assert resolver.to_ir_shape([64, 128]) == [64, 128]

    var = ir.Var("M", ir.ScalarType(DataType.INT64), ir.Span.unknown())
    result = resolver.to_ir_shape([var, 128])
    assert len(result) == 2
    assert result[0] is var
    assert isinstance(result[1], ir.ConstInt)
    assert result[1].value == 128


def test_resolve_tensor_type_with_int_vars():
    """Full TensorType resolution with int closure variables."""
    resolver = _make_resolver(closure_vars={"rows": 128, "cols": 64})
    node = ast.parse("pl.Tensor[[rows, cols], pl.DT_FP32]", mode="eval").body
    result = resolver.resolve_type(node)
    assert isinstance(result, ir.TensorType)
    assert len(result.shape) == 2
    assert result.dtype == DataType.FP32


def test_resolve_tensor_type_with_dynamic_policy():
    resolver = _make_resolver(closure_vars={"pl": pl})
    node = ast.parse("pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP32]", mode="eval").body
    result = resolver.resolve_param_type(node, parameter_name="x")
    assert isinstance(result, ir.TensorType)
    assert isinstance(result.shape[0], ir.Var)
    assert result.shape[0].name == "__pypto_dyn_x_0"


@pytest.mark.parametrize(
    "expr, closure_vars, error",
    [
        ("[undefined_var]", {}, "Unknown shape variable"),
        ("[x]", {"x": 3.14}, "positive integer"),
        ("[x]", {"x": "hello"}, "positive integer"),
        ("shape", {"shape": 42}, "must be a list or tuple"),
        ("shape", {"shape": [128, "bad"]}, "element 1 must be a positive integer"),
    ],
)
def test_parse_shape_rejects_invalid_exprs(expr, closure_vars, error):
    """parse_shape rejects unsupported shape variables and invalid dimensions."""
    resolver = _make_resolver(closure_vars=closure_vars)
    node = ast.parse(expr, mode="eval").body
    with pytest.raises((InvalidArgument, InvalidType), match=error):
        resolver.parse_shape(node)


def test_shape_variable_with_dynamic_policies():
    resolver = _make_resolver(closure_vars={"pl": pl, "shape": [pl.DYNAMIC, pl.DYNAMIC]})
    node = ast.parse("pl.Tensor[shape, pl.DT_FP32]", mode="eval").body
    tensor_type = resolver.resolve_param_type(node, parameter_name="x")
    assert tensor_type.shape[0].name == "__pypto_dyn_x_0"
    assert tensor_type.shape[1].name == "__pypto_dyn_x_1"


def test_shape_variable_mixes_dynamic_policy_and_integer():
    resolver = _make_resolver(closure_vars={"pl": pl, "shape": [pl.DYNAMIC, 128]})
    node = ast.parse("pl.Tensor[shape, pl.DT_FP32]", mode="eval").body
    tensor_type = resolver.resolve_param_type(node, parameter_name="x")
    assert tensor_type.shape[0].name == "__pypto_dyn_x_0"
    assert tensor_type.shape[1].value == 128


def test_resolve_dtype_from_closure():
    """DataType closure values resolve; non-DataType values raise."""
    resolver = _make_resolver(closure_vars={"dtype": DataType.FP32})
    node = ast.parse("dtype", mode="eval").body
    result = resolver.resolve_dtype(node)
    assert result == DataType.FP32

    resolver = _make_resolver(closure_vars={"dtype": "FP32"})
    node = ast.parse("dtype", mode="eval").body
    with pytest.raises(InvalidType, match="must be a DataType"):
        resolver.resolve_dtype(node)


def test_resolve_tensor_with_shape_and_dtype_variables():
    """Full TensorType resolution with shape list and dtype from closure."""
    resolver = _make_resolver(closure_vars={"shape": [128, 64], "dtype": DataType.FP16})
    node = ast.parse("pl.Tensor[shape, dtype]", mode="eval").body
    result = resolver.resolve_type(node)
    assert isinstance(result, ir.TensorType)
    assert len(result.shape) == 2
    assert result.dtype == DataType.FP16


def test_resolve_tensor_with_expression_dims():
    """Full TensorType resolution with expression-based dimensions."""
    resolver = _make_resolver(closure_vars={"base": 64})
    node = ast.parse("pl.Tensor[[base * 2, base], pl.DT_FP32]", mode="eval").body
    result = resolver.resolve_type(node)
    assert isinstance(result, ir.TensorType)
    assert len(result.shape) == 2
    assert result.dtype == DataType.FP32


def test_function_with_int_variable_shape():
    """Parse a function with int variables from enclosing scope."""
    rows, cols = 128, 64

    @pl.jit(auto_mutex=False)
    def func(
        x: pl.Tensor[[rows, cols], pl.DT_FP32],
    ):
        _test_result = x

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)
    param_type = func.params[0].type
    assert isinstance(param_type, ir.TensorType)
    assert len(param_type.shape) == 2


def test_function_with_var_shape():
    """Parse a function with a DYNAMIC policy."""
    @pl.jit(auto_mutex=False)
    def func(
        x: pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP32],
    ):
        _test_result = x

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    param_type = func.params[0].type
    assert isinstance(param_type, ir.TensorType)
    assert isinstance(param_type.shape[0], ir.Var)
    assert param_type.shape[0].name == "__pypto_dyn_x_0"
    # Second dim is still a ConstInt
    assert isinstance(param_type.shape[1], ir.ConstInt)
    assert param_type.shape[1].value == 128


def test_function_with_multiple_vars():
    """Each DYNAMIC parameter axis receives an independent ABI name."""
    @pl.jit(auto_mutex=False)
    def func(
        a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    ):
        _test_result = a

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    a_type = func.params[0].type
    b_type = func.params[1].type
    assert isinstance(a_type, ir.TensorType)
    assert isinstance(b_type, ir.TensorType)
    assert isinstance(a_type.shape[0], ir.Var)
    assert isinstance(a_type.shape[1], ir.Var)
    assert isinstance(b_type.shape[0], ir.Var)
    assert isinstance(b_type.shape[1], ir.Var)
    assert a_type.shape[0].name == "__pypto_dyn_a_0"
    assert a_type.shape[1].name == "__pypto_dyn_a_1"
    assert b_type.shape[0].name == "__pypto_dyn_b_0"
    assert b_type.shape[1].name == "__pypto_dyn_b_1"


def test_function_with_shape_variable():
    """Parse a function with shape as a list variable (issue #205)."""
    shape = [128, 128]
    dtype = pl.DT_FP32

    @pl.jit(auto_mutex=False)
    def func(t: pl.Tensor[shape, dtype]):
        _test_result = t

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)
    param_type = func.params[0].type
    assert isinstance(param_type, ir.TensorType)
    assert len(param_type.shape) == 2
    assert param_type.dtype == DataType.FP32


def test_function_with_multiple_shape_variables():
    """Parse a function with different shape variables per param (issue #205 pattern)."""
    tensor_shape = [128, 128]
    tile_shape = [64, 64]
    dtype = pl.DT_FP32

    @pl.jit(auto_mutex=False)
    def func(
        t: pl.Tensor[tensor_shape, dtype], tile: pl.Tensor[tile_shape, dtype]
    ):
        _test_result = t

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)
    assert isinstance(func.params[0].type, ir.TensorType)
    assert len(func.params[0].type.shape) == 2
    assert isinstance(func.params[1].type, ir.TensorType)
    assert len(func.params[1].type.shape) == 2


def test_arithmetic_shape_dims():
    """User computes shape dims from a base size."""
    base = 64

    @pl.jit(auto_mutex=False)
    def func(
        x: pl.Tensor[[base * 2, base], pl.DT_FP32],
    ):
        _test_result = x

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    param_type = func.params[0].type
    assert isinstance(param_type, ir.TensorType)
    assert param_type.shape == [128, 64]


def test_floor_division_in_shape():
    """User splits a dimension with //."""
    total = 256

    @pl.jit(auto_mutex=False)
    def func(
        x: pl.Tensor[[total // 4, total], pl.DT_FP32],
    ):
        _test_result = x

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    param_type = func.params[0].type
    assert isinstance(param_type, ir.TensorType)
    assert param_type.shape == [64, 256]


def test_multi_variable_arithmetic():
    """User combines multiple variables in shape expressions."""
    batch = 4
    heads = 8
    seq_len = 128

    @pl.jit(auto_mutex=False)
    def func(
        x: pl.Tensor[[batch * heads, seq_len], pl.DT_FP16],
    ):
        _test_result = x

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    param_type = func.params[0].type
    assert isinstance(param_type, ir.TensorType)
    assert param_type.shape == [32, 128]


def test_shape_from_dict_config():
    """User stores config in a dict, uses subscript for dims."""
    config = {"rows": 128, "cols": 64}

    @pl.jit(auto_mutex=False)
    def func(
        x: pl.Tensor[[config["rows"], config["cols"]], pl.DT_FP32],  # noqa: F821
    ):  # noqa: F821
        _test_result = x

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    param_type = func.params[0].type
    assert isinstance(param_type, ir.TensorType)
    assert param_type.shape == [128, 64]


def test_shape_from_list_indexing():
    """User picks dims from a list of predefined sizes."""
    sizes = [32, 64, 128, 256]

    @pl.jit(auto_mutex=False)
    def func(
        x: pl.Tensor[[sizes[2], sizes[1]], pl.DT_FP32],
    ):
        _test_result = x

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    param_type = func.params[0].type
    assert isinstance(param_type, ir.TensorType)
    assert param_type.shape == [128, 64]


def test_tuple_variable_as_shape():
    """User passes shape as a tuple (not list)."""
    shape = (128, 64)

    @pl.jit(auto_mutex=False)
    def func(x: pl.Tensor[shape, pl.DT_FP32]):
        _test_result = x

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    param_type = func.params[0].type
    assert isinstance(param_type, ir.TensorType)
    assert len(param_type.shape) == 2


def test_different_shapes_per_param():
    """User has different shapes for different parameters."""
    input_shape = [256, 128]
    output_shape = [256, 64]

    @pl.jit(auto_mutex=False)
    def func(
        x: pl.Tensor[input_shape, pl.DT_FP32],
        out: pl.Tensor[output_shape, pl.DT_FP32],
    ):
        _test_result = x

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    in_type = func.params[0].type
    out_type = func.params[1].type
    assert isinstance(in_type, ir.TensorType)
    assert isinstance(out_type, ir.TensorType)
    assert len(in_type.shape) == 2
    assert len(out_type.shape) == 2


def test_different_dtypes_per_param():
    """User has different dtypes for input and output."""
    dtype_in = pl.DT_FP16
    dtype_out = pl.DT_FP32

    @pl.jit(auto_mutex=False)
    def func(
        x: pl.Tensor[[128, 64], dtype_in],
        out: pl.Tensor[[128, 64], dtype_out],
    ):
        _test_result = pl.make_tensor(x, [128, 64], [64, 1], dtype=dtype_out)

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    in_type = func.params[0].type
    out_type = func.params[1].type
    assert isinstance(in_type, ir.TensorType)
    assert isinstance(out_type, ir.TensorType)
    assert in_type.dtype == DataType.FP16
    assert out_type.dtype == DataType.FP32


@pytest.mark.parametrize(
    "rows, cols, dtype",
    [
        (64, 128, pl.DT_FP16),
        (128, 256, pl.DT_FP32),
    ],
)
def test_parametrized_shape_and_dtype(rows, cols, dtype):
    """User parametrizes both shape and dtype together."""

    @pl.jit(auto_mutex=False)
    def func(
        x: pl.Tensor[[rows, cols], dtype],
    ):
        _test_result = x

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    param_type = func.params[0].type
    assert isinstance(param_type, ir.TensorType)
    assert len(param_type.shape) == 2
    assert param_type.dtype == dtype


def test_tile_shape_and_dtype_from_closure():
    """User defines tile shape and dtype outside, uses in body annotations."""
    tensor_shape = [128, 128]
    tile_shape = [64, 64]
    dtype = pl.DT_FP32

    @pl.jit(auto_mutex=False)
    def func(
        t: pl.Tensor[tensor_shape, dtype], out: pl.Tensor[tensor_shape, dtype]
    ):
        tile_type = pl.TileType(shape=tile_shape, dtype=dtype)
        a = pl.make_tile(tile_type, addr=0)
        pl.load(a, t, [0, 0])
        result: pl.Tensor[tensor_shape, dtype] = pl.store(out, a, [0, 0])
        _test_result = result

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)


def test_shapes_kwarg_from_variable():
    """User passes shapes= kwarg as a closure variable (t.py pattern)."""
    tile_shape = [32, 32]

    @pl.jit(auto_mutex=False)
    def func(
        t: pl.Tensor[[128, 128], pl.DT_FP32], out: pl.Tensor[[128, 128], pl.DT_FP32]
    ):
        tile_type = pl.TileType(shape=tile_shape, dtype=pl.DT_FP32)
        a = pl.make_tile(tile_type, addr=0)
        pl.load(a, t, [0, 0])
        result: pl.Tensor[[128, 128], pl.DT_FP32] = pl.store(out, a, [0, 0])
        _test_result = result

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)


def test_int_kwarg_from_closure():
    """User passes an int kwarg (like axis) from closure."""
    addr_from_closure = 512

    @pl.jit(auto_mutex=False)
    def func(
        x: pl.Tensor[[64, 128], pl.DT_FP32],
    ):
        tile_type = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        result = pl.make_tile(tile_type, addr=addr_from_closure)
        _test_result = result

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)


def test_dtype_kwarg_from_closure():
    """User passes dtype= kwarg from closure variable."""
    out_dtype = pl.DT_FP16

    @pl.jit(auto_mutex=False)
    def func(x: pl.Tensor[[64, 64], pl.DT_FP32]):
        result: pl.Tensor[[64, 64], pl.DT_FP16] = pl.make_tensor(x, [64, 64], [64, 1], dtype=out_dtype)
        _test_result = result

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)


def test_full_parametrized_kernel():
    """Realistic pattern: fully parametrized kernel (the t.py use case)."""
    dtype = pl.DT_FP32
    x, y = 128, 128
    shape = [128, 128]
    tile_shape = [64, 64]

    @pl.jit(auto_mutex=False)
    def kernel_add(t: pl.Tensor[[x, y], dtype], out: pl.Tensor[shape, dtype]):
        tile_type = pl.TileType(shape=tile_shape, dtype=dtype)
        a = pl.make_tile(tile_type, addr=0)
        b = pl.make_tile(tile_type, addr=16384)
        pl.load(a, t, [0, 0])
        pl.add(b, a, 5)
        result: pl.Tensor[shape, dtype] = pl.store(out, b, [0, 0])
        _test_result = result

    kernel_add_program, _ = kernel_add.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    kernel_add = kernel_add_program.get_function(kernel_add.__name__)

    assert isinstance(kernel_add, ir.Function)
    assert len(kernel_add.params) == 2
    for p in kernel_add.params:
        assert isinstance(p.type, ir.TensorType)
        assert p.type.dtype == DataType.FP32


def test_function_with_closure_shapes_in_body():
    """Function bodies resolve closure-provided shapes and dtypes."""
    shape = [128, 128]
    tile_shape = [64, 64]
    dtype = pl.DT_FP32

    @pl.jit(auto_mutex=False)
    def compute(
        t: pl.Tensor[shape, dtype], out: pl.Tensor[shape, dtype]
    ):
        tile_type = pl.TileType(shape=tile_shape, dtype=dtype)
        a = pl.make_tile(tile_type, addr=0)
        pl.load(a, t, [0, 0])
        result: pl.Tensor[shape, dtype] = pl.store(out, a, [0, 0])
        _test_result = result

    compute_program, _ = compute.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    compute = compute_program.get_function(compute.__name__)

    assert isinstance(compute, ir.Function)


def test_var_with_variable_shape():
    """A shape variable may contain DYNAMIC policies."""
    shape = [pl.DYNAMIC, pl.DYNAMIC]

    @pl.jit(auto_mutex=False)
    def func(x: pl.Tensor[shape, pl.DT_FP32]):
        _test_result = x

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    param_type = func.params[0].type
    assert isinstance(param_type, ir.TensorType)
    assert isinstance(param_type.shape[0], ir.Var)
    assert param_type.shape[0].name == "__pypto_dyn_x_0"
    assert isinstance(param_type.shape[1], ir.Var)
    assert param_type.shape[1].name == "__pypto_dyn_x_1"


def test_var_mixed_with_int_in_variable_shape():
    """A shape variable may mix DYNAMIC and a fixed integer."""
    shape = [pl.DYNAMIC, 128]

    @pl.jit(auto_mutex=False)
    def func(x: pl.Tensor[shape, pl.DT_FP32]):
        _test_result = x

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    param_type = func.params[0].type
    assert isinstance(param_type, ir.TensorType)
    assert isinstance(param_type.shape[0], ir.Var)
    assert param_type.shape[0].name == "__pypto_dyn_x_0"
    assert isinstance(param_type.shape[1], ir.ConstInt)
    assert param_type.shape[1].value == 128


def test_var_mixed_with_computed_dim():
    """A DYNAMIC axis may be mixed with a constant expression."""
    base = 64

    @pl.jit(auto_mutex=False)
    def func(
        x: pl.Tensor[[pl.DYNAMIC, base * 2], pl.DT_FP32],
    ):
        _test_result = x

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    param_type = func.params[0].type
    assert isinstance(param_type, ir.TensorType)
    assert isinstance(param_type.shape[0], ir.Var)
    assert param_type.shape[0].name == "__pypto_dyn_x_0"
    # base * 2 evaluates to 128 at compile time, then gets wrapped as ConstInt
    # because the shape is mixed (dynamic Expr + fixed integer)
    assert isinstance(param_type.shape[1], ir.ConstInt)
    assert param_type.shape[1].value == 128


def test_same_shape_reused_across_params():
    """A reused shape variable produces consistent parameter types."""
    shape = [64, 64]
    dtype = pl.DT_FP32

    @pl.jit(auto_mutex=False)
    def func(
        a: pl.Tensor[shape, dtype],
        b: pl.Tensor[shape, dtype],
        out: pl.Tensor[shape, dtype],
    ):
        _test_result = a

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    all_types = [p.type for p in func.params]
    for t in all_types:
        assert isinstance(t, ir.TensorType)
        assert len(t.shape) == 2
        assert t.dtype == DataType.FP32


def test_tile_type_with_variable_shape():
    """User uses a variable for Tile shape annotation."""
    tile_shape = [32, 32]

    @pl.jit(auto_mutex=False)
    def func(
        t: pl.Tensor[[128, 128], pl.DT_FP32], out: pl.Tensor[[128, 128], pl.DT_FP32]
    ):
        tile_type = pl.TileType(shape=tile_shape, dtype=pl.DT_FP32)
        a = pl.make_tile(tile_type, addr=0)
        pl.load(a, t, [0, 0])
        result: pl.Tensor[[128, 128], pl.DT_FP32] = pl.store(out, a, [0, 0])
        _test_result = result

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)


def test_shape_variable_not_defined_raises_error():
    """User typos variable name -should get a clear error."""
    shape = [128, 64]  # noqa: F841 -intentionally unused; typo below

    with pytest.raises(Exception, match="shaep|Cannot resolve|Unknown|undefined"):

        @pl.jit(auto_mutex=False)
        def func(x: pl.Tensor[shaep, pl.DT_FP32]):  # noqa: F821
            _test_result = x

        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_non_list_shape_variable_raises_error():
    """User accidentally passes a string as shape."""
    shape = "128x64"

    with pytest.raises(Exception, match="must be a list or tuple|Failed to evaluate"):

        @pl.jit(auto_mutex=False)
        def func(x: pl.Tensor[shape, pl.DT_FP32]):
            _test_result = x

        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_float_in_shape_raises_error():
    """User accidentally uses floats in shape."""
    shape = [128.0, 64.0]

    with pytest.raises(Exception, match="positive integer"):

        @pl.jit(auto_mutex=False)
        def func(x: pl.Tensor[shape, pl.DT_FP32]):
            _test_result = x

        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_nested_function_captures_correct_scope():
    """Shape variables from an outer function are captured by the parsed helper."""

    def make_kernel(rows, cols, dtype):
        @pl.jit(auto_mutex=False)
        def kernel(
            x: pl.Tensor[[rows, cols], dtype],
        ):
            _test_result = x

        kernel_program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
        kernel = kernel_program.get_function(kernel.__name__)

        return kernel

    k1 = make_kernel(64, 64, pl.DT_FP16)
    k2 = make_kernel(128, 256, pl.DT_FP32)

    assert isinstance(k1, ir.Function)
    assert isinstance(k2, ir.Function)
    k1_type = k1.params[0].type
    k2_type = k2.params[0].type
    assert isinstance(k1_type, ir.TensorType)
    assert isinstance(k2_type, ir.TensorType)
    assert k1_type.dtype == DataType.FP16
    assert k2_type.dtype == DataType.FP32


def test_factory_with_shape_variable():
    """User writes a factory function that parametrizes shape."""

    def make_kernel(shape, dtype):
        @pl.jit(auto_mutex=False)
        def kernel(x: pl.Tensor[shape, dtype]):
            _test_result = x

        kernel_program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
        kernel = kernel_program.get_function(kernel.__name__)

        return kernel

    k = make_kernel([128, 128], pl.DT_FP32)
    assert isinstance(k, ir.Function)
    param_type = k.params[0].type
    assert isinstance(param_type, ir.TensorType)
    assert len(param_type.shape) == 2
    assert param_type.dtype == DataType.FP32
