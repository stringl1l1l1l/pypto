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
from __future__ import annotations

# DSL function bodies are parsed as AST, not executed -suppress pyright errors
# from type-checking the annotations and kwargs inside parsed DSL bodies.
import ast
from enum import Enum
from typing import TYPE_CHECKING, Any

from pypto_pro import DataType, ir
from pypto_pro._errors import InvalidArgument, InvalidFormat, InvalidType, NotSupported
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


def test_language_exports_only_dt_dtype_constants():
    """The language frontend exposes DT_* names and removes legacy aliases."""
    assert pl.DT_INT32 == DataType.INT32
    assert pl.DT_FP32 == DataType.FP32
    assert pl.DT_BOOL == DataType.BOOL
    assert not hasattr(pl, "INT32")
    assert not hasattr(pl, "FP32")
    assert not hasattr(pl, "BOOL")


def test_input_and_output_are_public_enum_values():
    assert isinstance(pl.Input, Enum)
    assert isinstance(pl.Output, Enum)
    assert pl.Input.value == 0
    assert pl.Output.value == 1
    assert type(pl.Input) is type(pl.Output)
    assert not hasattr(pl, "TensorDirection")

    for annotation in (
        pl.Tensor[[64], pl.DT_FP32, pl.Output],
        pl.Tensor([64], pl.DT_FP32, pl.Output),
    ):
        assert isinstance(annotation, pl.Tensor)
        assert not hasattr(annotation, "direction")


@pytest.mark.parametrize(
    "code",
    [
        "pl.Tensor[[64], pl.DT_FP32, pl.Output, pl.ND, "
        "pl.MemRef(pl.MemorySpace.DDR, 0, 1024, 0)]",
        "pl.Tensor[[64], pl.DT_FP32, pl.ND, pl.Output, "
        "pl.MemRef(pl.MemorySpace.DDR, 0, 1024, 0)]",
        "pl.Tensor[[64], pl.DT_FP32, pl.ND, pl.MemRef(pl.MemorySpace.DDR, 0, 1024, 0), "
        "pl.Output]",
    ],
)
def test_tensor_direction_accepts_any_optional_position_after_dtype(code):
    resolver = _make_resolver({"pl": pl})
    node = ast.parse(code, mode="eval").body

    result = resolver.resolve_param_type(node, parameter_name="out", parameter_index=3)

    assert isinstance(result, ir.TensorType)
    assert result.memref is not None
    assert result.tensor_view.layout == ir.TensorLayout.ND
    assert resolver.param_directions == {3: pl.Output}


def test_call_style_tensor_direction_is_recorded_by_parameter_index():
    resolver = _make_resolver({"pl": pl})
    node = ast.parse(
        "pl.Tensor([64], pl.DT_FP32, pl.Output)",
        mode="eval",
    ).body

    result = resolver.resolve_param_type(node, parameter_name="out", parameter_index=2)

    assert isinstance(result, ir.TensorType)
    assert resolver.param_directions == {2: pl.Output}


@pytest.mark.parametrize("direction", ["input", "output", "in", "out"])
def test_call_style_tensor_direction_rejects_strings(direction):
    resolver = _make_resolver({"pl": pl})
    node = ast.parse(
        f"pl.Tensor([64], pl.DT_FP32, {direction!r})",
        mode="eval",
    ).body

    with pytest.raises(InvalidType, match="must be pl.Input or pl.Output"):
        resolver.resolve_param_type(node, parameter_name="tensor", parameter_index=0)


@pytest.mark.parametrize("direction", ["input", "output", "in", "out"])
def test_tensor_direction_rejects_strings(direction):
    resolver = _make_resolver({"pl": pl})
    node = ast.parse(
        f"pl.Tensor[[64], pl.DT_FP32, {direction!r}]",
        mode="eval",
    ).body

    with pytest.raises(InvalidFormat):
        resolver.resolve_param_type(node, parameter_name="tensor", parameter_index=0)


def test_tensor_direction_rejects_duplicate_values():
    resolver = _make_resolver({"pl": pl})
    node = ast.parse(
        "pl.Tensor[[64], pl.DT_FP32, pl.Input, pl.Output]",
        mode="eval",
    ).body

    with pytest.raises(InvalidArgument, match="Tensor direction can be specified only once"):
        resolver.resolve_param_type(node, parameter_name="out", parameter_index=0)


def test_resolve_tensor_type_subscript():
    """Test resolving tensor type with subscript notation."""
    resolver = _make_resolver()

    code = "pl.Tensor[[64, 128], pl.DT_FP16]"
    node = ast.parse(code, mode="eval").body

    result = resolver.resolve_type(node)

    assert isinstance(result, ir.TensorType)
    assert len(result.shape) == 2
    # Shape elements are ConstInt expressions
    assert result.dtype == DataType.FP16


def test_resolve_tensor_type_different_dtypes():
    """Test resolving tensor types with different data types."""
    resolver = _make_resolver()

    test_cases = [
        ("pl.Tensor[[64], pl.DT_FP32]", DataType.FP32),
        ("pl.Tensor[[32, 64], pl.DT_INT32]", DataType.INT32),
        ("pl.Tensor[[1, 2, 3], pl.DT_FP16]", DataType.FP16),
    ]

    for code, expected_dtype in test_cases:
        node = ast.parse(code, mode="eval").body
        result = resolver.resolve_type(node)

        assert isinstance(result, ir.TensorType)
        assert result.dtype == expected_dtype


def test_resolve_dtype_attribute():
    """Test resolving dtype from attribute access."""
    resolver = _make_resolver()

    code = "pl.DT_FP16"
    node = ast.parse(code, mode="eval").body

    result = resolver.resolve_dtype(node)
    assert result == DataType.FP16


def test_resolve_dt_scalar_type():
    """DT_* attributes resolve to scalar types."""
    resolver = _make_resolver()
    result = resolver.resolve_type(ast.parse("pl.DT_INT32", mode="eval").body)

    assert isinstance(result, ir.ScalarType)
    assert result.dtype == DataType.INT32


@pytest.mark.parametrize("code", ["pl.INT32", "pl.FP32", "pl.BOOL"])
def test_removed_scalar_annotation_syntax_raises(code):
    """Legacy scalar annotation spellings are no longer accepted."""
    resolver = _make_resolver()

    with pytest.raises(InvalidType):
        resolver.resolve_type(ast.parse(code, mode="eval").body)


@pytest.mark.parametrize("dtype_name", ["DT_INT4", "DT_FP4"])
def test_scalar_unsupported_low_precision_dtypes(dtype_name):
    """Low-precision dtypes are rejected as scalar type annotations."""
    resolver = _make_resolver()
    node = ast.parse(f"pl.{dtype_name}", mode="eval").body

    with pytest.raises(NotSupported, match="Scalar type does not support dtype"):
        resolver.resolve_type(node)


def test_resolve_dtype_all_types():
    """Test all supported dtype values."""
    resolver = _make_resolver()

    dtypes = [
        ("pl.DT_FP16", DataType.FP16),
        ("pl.DT_FP32", DataType.FP32),
        ("pl.DT_INT32", DataType.INT32),
        ("pl.DT_INT64", DataType.INT64),
        ("pl.DT_BOOL", DataType.BOOL),
    ]

    for code, expected in dtypes:
        node = ast.parse(code, mode="eval").body
        result = resolver.resolve_dtype(node)
        assert result == expected


def test_resolve_invalid_dtype():
    """Test error on invalid dtype."""
    resolver = _make_resolver()

    code = "pl.INVALID_TYPE"
    node = ast.parse(code, mode="eval").body

    with pytest.raises(InvalidType, match="Unknown dtype"):
        resolver.resolve_dtype(node)


def test_resolve_invalid_tensor_syntax():
    """Test error on invalid tensor syntax."""
    resolver = _make_resolver()

    # Missing dtype
    code = "pl.Tensor[[64, 128]]"
    node = ast.parse(code, mode="eval").body

    with pytest.raises(InvalidArgument, match="requires"):
        resolver.resolve_type(node)


def test_parse_shape_list():
    """Test parsing shape from list literal."""
    resolver = _make_resolver()

    code = "[64, 128, 256]"
    node = ast.parse(code, mode="eval").body

    shape = resolver.parse_shape(node)
    assert len(shape) == 3
    assert shape == [64, 128, 256]


def test_parse_shape_tuple():
    """Test parsing shape from tuple literal."""
    resolver = _make_resolver()

    code = "(32, 64)"
    node = ast.parse(code, mode="eval").body

    shape = resolver.parse_shape(node)
    assert len(shape) == 2
    assert shape == [32, 64]


def test_parse_shape_invalid():
    """Test error on invalid shape (not a list, tuple, or known variable)."""
    resolver = _make_resolver()

    # Bare variable name not in closure_vars
    code = "x"
    node = ast.parse(code, mode="eval").body

    with pytest.raises(InvalidArgument, match="Unknown shape variable"):
        resolver.parse_shape(node)


def test_resolve_tuple_two_tensors():
    """Test resolving tuple[pl.Tensor[...], pl.Tensor[...]]."""
    resolver = _make_resolver()

    code = "tuple[pl.Tensor[[64], pl.DT_FP32], pl.Tensor[[128], pl.DT_FP16]]"
    node = ast.parse(code, mode="eval").body

    result = resolver.resolve_type(node)
    assert isinstance(result, ir.TupleType)
    assert len(result.types) == 2
    assert isinstance(result.types[0], ir.TensorType)
    assert result.types[0].dtype == DataType.FP32
    assert isinstance(result.types[1], ir.TensorType)
    assert result.types[1].dtype == DataType.FP16


def test_resolve_tuple_mixed_types():
    """Test resolving tuple with mixed Tensor and Scalar types."""
    resolver = _make_resolver()

    code = "tuple[pl.Tensor[[32, 64], pl.DT_FP32], pl.DT_INT64]"
    node = ast.parse(code, mode="eval").body

    result = resolver.resolve_type(node)
    assert isinstance(result, ir.TupleType)
    assert len(result.types) == 2
    assert isinstance(result.types[0], ir.TensorType)
    assert isinstance(result.types[1], ir.ScalarType)
    assert result.types[1].dtype == DataType.INT64


def test_resolve_tuple_single_element():
    """Test resolving tuple with a single element."""
    resolver = _make_resolver()

    code = "tuple[pl.Tensor[[64], pl.DT_FP32]]"
    node = ast.parse(code, mode="eval").body

    result = resolver.resolve_type(node)
    assert isinstance(result, ir.TupleType)
    assert len(result.types) == 1
    assert isinstance(result.types[0], ir.TensorType)


def test_resolve_nested_tuple_error():
    """Test that nested tuple types raise an error."""
    resolver = _make_resolver()

    code = "tuple[tuple[pl.Tensor[[64], pl.DT_FP32]], pl.Tensor[[128], pl.DT_FP16]]"
    node = ast.parse(code, mode="eval").body

    with pytest.raises(NotSupported, match="Nested tuple types"):
        resolver.resolve_type(node)
