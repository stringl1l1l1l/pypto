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

"""IR builders and parse handlers used by the PyPTO Pro SIMT frontend."""

from __future__ import annotations

import ast
from collections.abc import Callable, Sequence
from typing import Any

from pypto.pypto_impl import ir as _ir_core
from pypto.pypto_impl.ir import Call, Expr, Function, Span

from ..._errors import (
    CommonExternal,
    InvalidArgument,
    InvalidOperation,
    InvalidShape,
    InvalidType,
    InvalidVal,
    NotSupported,
    message_of,
)
from .._utils import _get_span_or_capture, _normalize_expr
from ._op_registry import op_impl

_DIM3_FIELDS = ("x", "y", "z")
_DIM3_CONTEXT_NAME = "dim3_context"


def _make_dim3_components(op_name: str, span: Span) -> list[Call]:
    return [_ir_core.create_op_call(op_name, [], {"axis": axis}, span) for axis in range(len(_DIM3_FIELDS))]


def _make_dim3_context(op_name: str, span: Span | None = None) -> Expr:
    """Build a three-component SIMT context tuple."""
    actual_span = _get_span_or_capture(span)
    return _ir_core.MakeTuple(_make_dim3_components(op_name, actual_span), actual_span)


def linear_thread_idx(span: Span | None = None) -> Call:
    """Return the x-major flattened thread index within the current block."""
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("simt.linear_thread_idx", [], {}, actual_span)


def warp_size(span: Span | None = None) -> Call:
    """Return the target SIMT warp size."""
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("simt.warp_size", [], {}, actual_span)


def syncthreads(span: Span | None = None) -> Call:
    """Build a block-wide SIMT thread barrier."""
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("simt.syncthreads", [], {}, actual_span)


def threadfence_block(span: Span | None = None) -> Call:
    """Build a block-scoped SIMT memory fence."""
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("simt.threadfence_block", [], {}, actual_span)


def threadfence(span: Span | None = None) -> Call:
    """Build a device-scoped SIMT memory fence."""
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("simt.threadfence", [], {}, actual_span)


def cast(
    value: Expr,
    dtype: _ir_core.DataType,
    mode: _ir_core.RoundMode = _ir_core.RoundMode.CAST_NONE,
    span: Span | None = None,
) -> Call:
    """Build a registered SIMT scalar-cast operation."""
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call(
        "simt.cast",
        [value],
        {"target_type": dtype, "mode": mode},
        actual_span,
    )


def bitcast(value: Expr, dtype: _ir_core.DataType, span: Span | None = None) -> Call:
    """Build a registered SIMT scalar bit-reinterpretation operation."""
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call(
        "simt.bitcast",
        [value],
        {"target_type": dtype},
        actual_span,
    )


def _create_math_call(op_name: str, *operands: Expr, span: Span | None) -> Call:
    """Create an IR call for one explicit SIMT scalar math operation."""
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call(f"simt.{op_name}", list(operands), {}, actual_span)


def abs(value: Expr, span: Span | None = None) -> Call:
    """Build a supported scalar absolute-value call."""
    return _create_math_call("abs", value, span=span)


def min(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Build a same-dtype scalar minimum call."""
    return _create_math_call("min", lhs, rhs, span=span)


def max(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Build a same-dtype scalar maximum call."""
    return _create_math_call("max", lhs, rhs, span=span)


def sqrt(value: Expr, span: Span | None = None) -> Call:
    """Build a floating-point scalar square-root call."""
    return _create_math_call("sqrt", value, span=span)


def rsqrt(value: Expr, span: Span | None = None) -> Call:
    """Build a floating-point scalar reciprocal-square-root call."""
    return _create_math_call("rsqrt", value, span=span)


def exp(value: Expr, span: Span | None = None) -> Call:
    """Build a floating-point scalar exponential call."""
    return _create_math_call("exp", value, span=span)


def exp2(value: Expr, span: Span | None = None) -> Call:
    """Build a floating-point scalar base-two exponential call."""
    return _create_math_call("exp2", value, span=span)


def log(value: Expr, span: Span | None = None) -> Call:
    """Build a floating-point scalar natural-logarithm call."""
    return _create_math_call("log", value, span=span)


def log2(value: Expr, span: Span | None = None) -> Call:
    """Build a floating-point scalar base-two logarithm call."""
    return _create_math_call("log2", value, span=span)


def log1p(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32 scalar log-one-plus call."""
    return _create_math_call("log1p", value, span=span)


def sin(value: Expr, span: Span | None = None) -> Call:
    """Build a floating-point scalar sine call."""
    return _create_math_call("sin", value, span=span)


def cos(value: Expr, span: Span | None = None) -> Call:
    """Build a floating-point scalar cosine call."""
    return _create_math_call("cos", value, span=span)


def tanh(value: Expr, span: Span | None = None) -> Call:
    """Build a floating-point scalar hyperbolic-tangent call."""
    return _create_math_call("tanh", value, span=span)


def rint(value: Expr, span: Span | None = None) -> Call:
    """Build a floating-point scalar round-to-nearest-even call."""
    return _create_math_call("rint", value, span=span)


def round(value: Expr, span: Span | None = None) -> Call:
    """Build a floating-point scalar round-half-away-from-zero call."""
    return _create_math_call("round", value, span=span)


def floor(value: Expr, span: Span | None = None) -> Call:
    """Build a floating-point scalar floor call."""
    return _create_math_call("floor", value, span=span)


def ceil(value: Expr, span: Span | None = None) -> Call:
    """Build a floating-point scalar ceiling call."""
    return _create_math_call("ceil", value, span=span)


def trunc(value: Expr, span: Span | None = None) -> Call:
    """Build a floating-point scalar truncate-toward-zero call."""
    return _create_math_call("trunc", value, span=span)


def isnan(value: Expr, span: Span | None = None) -> Call:
    """Build a floating-point scalar NaN-classification call."""
    return _create_math_call("isnan", value, span=span)


def isinf(value: Expr, span: Span | None = None) -> Call:
    """Build a floating-point scalar infinity-classification call."""
    return _create_math_call("isinf", value, span=span)


def isfinite(value: Expr, span: Span | None = None) -> Call:
    """Build an FP16 or FP32 scalar finiteness-classification call."""
    return _create_math_call("isfinite", value, span=span)


def popcount(value: Expr, span: Span | None = None) -> Call:
    """Build an unsigned scalar population count with an INT32 result."""
    return _create_math_call("popcount", value, span=span)


def mul_hi(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Build a same-dtype integer multiply-high call."""
    return _create_math_call("mul_hi", lhs, rhs, span=span)


def fmod(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Build an FP32 remainder call with a quotient truncated toward zero."""
    return _create_math_call("fmod", lhs, rhs, span=span)


def fma(lhs: Expr, rhs: Expr, addend: Expr, span: Span | None = None) -> Call:
    """Build a same-dtype floating-point fused multiply-add call."""
    return _create_math_call("fma", lhs, rhs, addend, span=span)


def _create_atomic_call(
    op_name: str,
    container: Expr,
    offset: Expr,
    *operands: Expr,
    span: Span | None,
) -> Call:
    """Create an IR call for one SIMT atomic operation."""
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call(
        f"simt.{op_name}",
        [container, offset, *operands],
        {},
        actual_span,
    )


def atomic_add(container: Expr, offset: Expr, value: Expr, span: Span | None = None) -> Call:
    """Build an atomic add on one element of a SIMT Tile or Tensor."""
    return _create_atomic_call("atomic_add", container, offset, value, span=span)


def atomic_sub(container: Expr, offset: Expr, value: Expr, span: Span | None = None) -> Call:
    """Build an atomic subtract on one element of a SIMT Tile or Tensor."""
    return _create_atomic_call("atomic_sub", container, offset, value, span=span)


def atomic_exch(container: Expr, offset: Expr, value: Expr, span: Span | None = None) -> Call:
    """Build an atomic exchange on one element of a SIMT Tile or Tensor."""
    return _create_atomic_call("atomic_exch", container, offset, value, span=span)


def atomic_max(container: Expr, offset: Expr, value: Expr, span: Span | None = None) -> Call:
    """Build an atomic maximum on one element of a SIMT Tile or Tensor."""
    return _create_atomic_call("atomic_max", container, offset, value, span=span)


def atomic_min(container: Expr, offset: Expr, value: Expr, span: Span | None = None) -> Call:
    """Build an atomic minimum on one element of a SIMT Tile or Tensor."""
    return _create_atomic_call("atomic_min", container, offset, value, span=span)


def atomic_inc(container: Expr, offset: Expr, limit: Expr, span: Span | None = None) -> Call:
    """Build an atomic wrapping increment on one SIMT counter element."""
    return _create_atomic_call("atomic_inc", container, offset, limit, span=span)


def atomic_dec(container: Expr, offset: Expr, limit: Expr, span: Span | None = None) -> Call:
    """Build an atomic wrapping decrement on one SIMT counter element."""
    return _create_atomic_call("atomic_dec", container, offset, limit, span=span)


def atomic_cas(
    container: Expr,
    offset: Expr,
    compare: Expr,
    value: Expr,
    span: Span | None = None,
) -> Call:
    """Build an atomic compare-and-swap on one SIMT element."""
    return _create_atomic_call("atomic_cas", container, offset, compare, value, span=span)


def atomic_and(container: Expr, offset: Expr, value: Expr, span: Span | None = None) -> Call:
    """Build an atomic bitwise AND on one SIMT element."""
    return _create_atomic_call("atomic_and", container, offset, value, span=span)


def atomic_or(container: Expr, offset: Expr, value: Expr, span: Span | None = None) -> Call:
    """Build an atomic bitwise OR on one SIMT element."""
    return _create_atomic_call("atomic_or", container, offset, value, span=span)


def atomic_xor(container: Expr, offset: Expr, value: Expr, span: Span | None = None) -> Call:
    """Build an atomic bitwise XOR on one SIMT element."""
    return _create_atomic_call("atomic_xor", container, offset, value, span=span)


def launch(
    callee: Function,
    *,
    threads: int | Expr | tuple[int | Expr, ...],
    args: Sequence[Expr],
    span: Span | None = None,
) -> Call:
    """Build a SIMT launch call.

    Normal DSL source is parsed specially so the callee Function is retained in
    the enclosing Program. This builder is primarily useful for direct IR tests.
    """
    actual_span = _get_span_or_capture(span)
    thread_dims = list(threads) if isinstance(threads, tuple) else [threads]
    if not 1 <= len(thread_dims) <= 3:
        raise InvalidShape("SIMT launch threads must contain one to three dimensions")
    thread_dims.extend([1] * (3 - len(thread_dims)))
    normalized_dims = [_normalize_expr(dim, actual_span) for dim in thread_dims]
    return _ir_core.create_op_call(
        "simt.launch",
        [*normalized_dims, *args],
        {"callee": callee.name, "max_threads": callee.get_attr("max_threads")},
        actual_span,
    )


def _validate_simt_body_op(parser: Any, call: ast.Call, _kwargs: dict[str, Any]) -> None:
    """Validate the common source constraints of a SIMT body operation."""

    span = parser.span_tracker.get_span(call)
    op_name = parser._extract_op_name(call.func)
    if parser._current_func_type not in (_ir_core.FunctionType.SimtVF, _ir_core.FunctionType.SimtCallee):
        raise InvalidOperation(
            f"pl.{op_name}() can only be used inside a SIMT function",
            span=span,
            hint="Move SIMT-context-dependent logic into @pl.vector_function(mode=\"simt\").",
        )
    if call.args:
        raise InvalidArgument(f"pl.{op_name}() does not accept positional arguments", span=span)
    if call.keywords:
        raise InvalidArgument(f"pl.{op_name}() does not accept keyword arguments", span=span)


def _parse_dim3_context(parser: Any, call: ast.Call, op_name: str) -> Expr:
    """Parse a SIMT context query as an IR named tuple."""
    _validate_simt_body_op(parser, call, {})
    span = parser.span_tracker.get_span(call)
    return parser.make_named_tuple(
        _make_dim3_components(op_name, span),
        _DIM3_FIELDS,
        span,
        name=_DIM3_CONTEXT_NAME,
    )


@op_impl("simt.thread_idx")
def _parse_thread_idx(parser: Any, call: ast.Call) -> Expr:
    return _parse_dim3_context(parser, call, "simt.thread_idx")


@op_impl("simt.block_dim")
def _parse_block_dim(parser: Any, call: ast.Call) -> Expr:
    return _parse_dim3_context(parser, call, "simt.block_dim")


@op_impl("simt.block_idx")
def _parse_block_idx(parser: Any, call: ast.Call) -> Expr:
    return _parse_dim3_context(parser, call, "simt.block_idx")


@op_impl("simt.grid_dim")
def _parse_grid_dim(parser: Any, call: ast.Call) -> Expr:
    return _parse_dim3_context(parser, call, "simt.grid_dim")


@op_impl("simt.linear_thread_idx")
def _parse_linear_thread_idx(parser: Any, call: ast.Call) -> Expr:
    _validate_simt_body_op(parser, call, {})
    span = parser.span_tracker.get_span(call)
    return linear_thread_idx(span)


@op_impl("simt.warp_size")
def _parse_warp_size(parser: Any, call: ast.Call) -> Expr:
    _validate_simt_body_op(parser, call, {})
    span = parser.span_tracker.get_span(call)
    return warp_size(span)


@op_impl("simt.syncthreads")
def _parse_syncthreads(parser: Any, call: ast.Call) -> Expr:
    _validate_simt_body_op(parser, call, {})
    span = parser.span_tracker.get_span(call)
    return syncthreads(span)


@op_impl("simt.threadfence_block")
def _parse_threadfence_block(parser: Any, call: ast.Call) -> Expr:
    _validate_simt_body_op(parser, call, {})
    span = parser.span_tracker.get_span(call)
    return threadfence_block(span)


@op_impl("simt.threadfence")
def _parse_threadfence(parser: Any, call: ast.Call) -> Expr:
    _validate_simt_body_op(parser, call, {})
    span = parser.span_tracker.get_span(call)
    return threadfence(span)


def _parse_simt_launch(parser: Any, call: ast.Call, local_name: str, callee_template: Callable) -> Expr:
    """Parse ``simt_func[threads](args...)`` and lower it to ``simt.launch`` IR."""
    from pypto_pro.language.parser.decorator import get_simt_max_threads

    span = parser.span_tracker.get_span(call)
    if parser._current_func_type in (_ir_core.FunctionType.SimtVF, _ir_core.FunctionType.SimtCallee):
        raise NotSupported("Nested SIMT vector-function invocation is not supported", span=span)
    if parser.target != _ir_core.SectionKind.Vector:
        raise InvalidOperation(
            "SIMT vector functions must be invoked inside 'with pl.section_vector():'",
            span=span,
            hint="SIMT vector functions execute on AIV and cannot be invoked from Cube or shared kernel code.",
        )

    if call.keywords or any(isinstance(arg, ast.Starred) for arg in call.args):
        raise InvalidArgument(
            "SIMT vector-function invocation accepts positional arguments only",
            span=span,
            hint="Use: simt_func[threads](arg0, arg1, ...)",
        )

    if get_simt_max_threads(callee_template) is None:
        raise InvalidVal(
            f"'{local_name}' is not a launchable "
            '@pl.vector_function(mode="simt", max_threads=...)',
            span=parser.span_tracker.get_span(call.func.value),
            parser_retry=True,
        )

    threads_node = call.func.slice
    if isinstance(threads_node, ast.Tuple):
        if not 1 <= len(threads_node.elts) <= 3:
            raise InvalidShape(
                "SIMT thread configuration must contain one to three dimensions",
                span=parser.span_tracker.get_span(threads_node),
                hint="Use simt_func[N](...), simt_func[x, y](...), or simt_func[x, y, z](...).",
            )
        thread_nodes = list(threads_node.elts)
    else:
        thread_nodes = [threads_node]
    thread_dims = [parser.parse_expression(dim) for dim in thread_nodes]

    launch_args = [parser.parse_expression(arg) for arg in call.args]
    callee = parser._instantiate_simt_function(local_name, callee_template, launch_args, call.args, span)
    parser._validate_simt_function_arguments(callee, launch_args, call.args, span)
    try:
        result = launch(callee, threads=tuple(thread_dims), args=launch_args, span=span)
    except RuntimeError as error:
        raise CommonExternal(message_of(error), span=span, parser_retry=True) from error
    parser.requires_simt = True
    return result

def _numeric_literal_value(node: ast.expr) -> int | float | None:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
        and type(node.operand.value) in (int, float)
    ):
        return node.operand.value if isinstance(node.op, ast.UAdd) else -node.operand.value
    return None


def _parse_atomic_operand(
    parser: Any,
    op_name: str,
    operand_name: str,
    operand_node: ast.expr,
    dtype: _ir_core.DataType,
) -> Expr:
    """Parse one atomic operand, contextually typing numeric literals."""

    operand_span = parser.span_tracker.get_span(operand_node)
    literal_value = _numeric_literal_value(operand_node)
    if literal_value is not None:
        if dtype.is_int() and type(literal_value) is not int:
            raise InvalidType(
                f"pl.{op_name}() requires an integer {operand_name} for target dtype {dtype}",
                span=operand_span,
                hint=f"Use an integer literal or a Scalar with dtype {dtype}.",
                parser_retry=True,
            )
        return parser._make_scalar_constant(literal_value, dtype, operand_span)

    return parser.parse_expression(operand_node)


def _parse_atomic_call(
    parser: Any,
    op_name: str,
    builder: Callable[..., Call],
    operand_names: tuple[str, ...],
    call: ast.Call,
) -> Expr:
    """Parse one ``pl.simt.atomic_*`` call without loading its target."""

    span = parser.span_tracker.get_span(call)
    if parser._current_func_type not in (_ir_core.FunctionType.SimtVF, _ir_core.FunctionType.SimtCallee):
        raise InvalidOperation(
            f"pl.{op_name}() can only be used inside a SIMT function",
            span=span,
            hint="Move SIMT-context-dependent logic into @pl.vector_function(mode=\"simt\").",
        )

    short_name = op_name[len("simt."):]
    expected_args = 1 + len(operand_names)
    if len(call.args) != expected_args or call.keywords:
        signature = ", ".join(("target", *operand_names))
        raise InvalidArgument(
            f"pl.simt.{short_name}() requires exactly {expected_args} positional arguments",
            span=span,
            hint=f"Use pl.simt.{short_name}({signature}).",
        )

    target_node = call.args[0]
    if not isinstance(target_node, ast.Subscript):
        raise InvalidVal(
            f"pl.simt.{short_name}() target must be a direct Tile or Tensor subscript",
            span=parser.span_tracker.get_span(target_node),
            hint=f"Use pl.simt.{short_name}(tile[row, col], ...), not a previously loaded Scalar.",
        )
    if isinstance(target_node.slice, ast.Slice) or (
        isinstance(target_node.slice, ast.Tuple)
        and any(isinstance(index, ast.Slice) for index in target_node.slice.elts)
    ):
        raise NotSupported(
            f"pl.simt.{short_name}() target does not support slices",
            span=parser.span_tracker.get_span(target_node),
            hint="Select exactly one element with integer indices.",
        )

    container = parser.parse_expression(target_node.value)
    container_type = container.type if isinstance(container, Expr) else None
    if not isinstance(container_type, (_ir_core.TileType, _ir_core.TensorType)):
        raise InvalidType(
            f"pl.simt.{short_name}() target container must be a Tile or Tensor",
            span=parser.span_tracker.get_span(target_node.value),
            parser_retry=True,
        )

    dtype = container_type.dtype
    target_span = parser.span_tracker.get_span(target_node)
    offset = parser._parse_scalar_subscript_index(container, target_node.slice, target_span)
    operands = [
        _parse_atomic_operand(parser, op_name, operand_name, operand_node, dtype)
        for operand_name, operand_node in zip(operand_names, call.args[1:])
    ]
    try:
        return builder(container, offset, *operands, span=span)
    except RuntimeError as exc:
        raise CommonExternal(message_of(exc), span=span, parser_retry=True) from exc


@op_impl("simt.cast")
def _parse_simt_cast(parser: Any, call: ast.Call) -> Expr:
    """Parse ``pl.simt.cast`` while preserving its scalar-specific diagnostics."""

    span = parser.span_tracker.get_span(call)
    if parser._current_func_type not in (_ir_core.FunctionType.SimtVF, _ir_core.FunctionType.SimtCallee):
        raise InvalidOperation(
            "pl.simt.cast() can only be used inside a SIMT function",
            span=span,
            hint="Move SIMT-context-dependent logic into @pl.vector_function(mode=\"simt\").",
        )

    if len(call.args) != 2:
        raise InvalidArgument(
            f"pl.simt.cast() requires exactly 2 positional arguments, got {len(call.args)}",
            span=span,
            hint="Use pl.simt.cast(value, pl.DT_FP32, mode=pl.RoundMode.CAST_RINT).",
        )

    mode_keywords = [kw for kw in call.keywords if kw.arg == "mode"]
    if len(mode_keywords) != len(call.keywords) or len(mode_keywords) > 1:
        raise InvalidArgument(
            "pl.simt.cast() only accepts one optional keyword argument: mode",
            span=span,
        )

    value = parser.parse_expression(call.args[0])
    if not isinstance(value, Expr) or not isinstance(value.type, _ir_core.ScalarType):
        raise InvalidType(
            "pl.simt.cast() value must be a scalar expression",
            span=parser.span_tracker.get_span(call.args[0]),
            parser_retry=True,
        )

    dtype = parser.parse_expression(call.args[1])
    if not isinstance(dtype, _ir_core.DataType):
        raise InvalidType(
            "pl.simt.cast() dtype must be a pl.DT_* value",
            span=parser.span_tracker.get_span(call.args[1]),
            parser_retry=True,
        )

    mode = _ir_core.RoundMode.CAST_NONE
    if mode_keywords:
        mode = parser.resolve_single_kwarg("mode", mode_keywords[0].value)
        if not isinstance(mode, _ir_core.RoundMode):
            raise InvalidType(
                "pl.simt.cast() mode must be a pl.RoundMode value",
                span=parser.span_tracker.get_span(mode_keywords[0].value),
                parser_retry=True,
            )

    try:
        return cast(value, dtype, mode, span)
    except (ValueError, RuntimeError) as err:
        message = str(err).replace("simt.cast", "pl.simt.cast()")
        raise CommonExternal(message, span=span, parser_retry=True) from err


@op_impl("simt.bitcast")
def _parse_simt_bitcast(parser: Any, call: ast.Call) -> Expr:
    """Parse ``pl.simt.bitcast`` with scalar and dtype validation."""

    span = parser.span_tracker.get_span(call)
    if parser._current_func_type not in (
        _ir_core.FunctionType.SimtVF,
        _ir_core.FunctionType.SimtCallee,
    ):
        raise InvalidOperation(
            "pl.simt.bitcast() can only be used inside a SIMT function",
            span=span,
            hint="Move SIMT-context-dependent logic into @pl.vector_function(mode=\"simt\").",
        )

    if call.keywords:
        raise InvalidArgument("pl.simt.bitcast() does not accept keyword arguments", span=span)
    if len(call.args) != 2:
        raise InvalidArgument(
            f"pl.simt.bitcast() requires exactly 2 positional arguments, got {len(call.args)}",
            span=span,
            hint="Use pl.simt.bitcast(value, pl.DT_UINT32).",
        )

    value = parser.parse_expression(call.args[0])
    if not isinstance(value, Expr) or not isinstance(value.type, _ir_core.ScalarType):
        raise InvalidType(
            "pl.simt.bitcast() value must be a scalar expression",
            span=parser.span_tracker.get_span(call.args[0]),
            parser_retry=True,
        )

    dtype = parser.parse_expression(call.args[1])
    if not isinstance(dtype, _ir_core.DataType):
        raise InvalidType(
            "pl.simt.bitcast() dtype must be a pl.DT_* value",
            span=parser.span_tracker.get_span(call.args[1]),
            parser_retry=True,
        )

    try:
        return bitcast(value, dtype, span)
    except (ValueError, RuntimeError) as err:
        message = str(err).replace("simt.bitcast", "pl.simt.bitcast()")
        raise CommonExternal(message, span=span, parser_retry=True) from err


def _parse_scalar_math(
    parser: Any,
    op_name: str,
    call: ast.Call,
    expected_args: int,
    builder: Callable[..., Call],
) -> Expr:

    short_name = op_name[len("simt."):]
    span = parser.span_tracker.get_span(call)
    if parser._current_func_type not in (_ir_core.FunctionType.SimtVF, _ir_core.FunctionType.SimtCallee):
        raise InvalidOperation(
            f"pl.{op_name}() can only be used inside a SIMT function",
            span=span,
            hint="Move SIMT-context-dependent logic into @pl.vector_function(mode=\"simt\").",
        )
    if len(call.args) != expected_args or call.keywords:
        raise InvalidArgument(
            f"pl.simt.{short_name}() requires exactly {expected_args} positional argument"
            f"{'s' if expected_args != 1 else ''}",
            span=span,
        )
    args = [parser.parse_expression(arg) for arg in call.args]
    for index, (arg, arg_node) in enumerate(zip(args, call.args)):
        if not isinstance(arg, Expr) or not isinstance(arg.type, _ir_core.ScalarType):
            raise InvalidVal(
                f"pl.simt.{short_name}() argument {index + 1} must be a scalar expression",
                span=parser.span_tracker.get_span(arg_node),
                hint="Subscript a Tile or Tensor first to obtain one scalar element.",
                parser_retry=True,
            )
    try:
        return builder(*args, span=span)
    except (ValueError, RuntimeError) as err:
        raise CommonExternal(message_of(err), span=span, parser_retry=True) from err


def _register_scalar_math_parser(op_name: str, expected_args: int, builder: Callable[..., Call]) -> None:
    """Register one SIMT scalar-math op using the shared parser."""

    @op_impl(op_name)
    def _parser_impl(parser: Any, call: ast.Call) -> Expr:
        return _parse_scalar_math(parser, op_name, call, expected_args, builder)


_register_scalar_math_parser("simt.abs", 1, abs)
_register_scalar_math_parser("simt.min", 2, min)
_register_scalar_math_parser("simt.max", 2, max)
_register_scalar_math_parser("simt.sqrt", 1, sqrt)
_register_scalar_math_parser("simt.rsqrt", 1, rsqrt)
_register_scalar_math_parser("simt.exp", 1, exp)
_register_scalar_math_parser("simt.exp2", 1, exp2)
_register_scalar_math_parser("simt.log", 1, log)
_register_scalar_math_parser("simt.log2", 1, log2)
_register_scalar_math_parser("simt.log1p", 1, log1p)
_register_scalar_math_parser("simt.sin", 1, sin)
_register_scalar_math_parser("simt.cos", 1, cos)
_register_scalar_math_parser("simt.tanh", 1, tanh)
_register_scalar_math_parser("simt.rint", 1, rint)
_register_scalar_math_parser("simt.round", 1, round)
_register_scalar_math_parser("simt.floor", 1, floor)
_register_scalar_math_parser("simt.ceil", 1, ceil)
_register_scalar_math_parser("simt.trunc", 1, trunc)
_register_scalar_math_parser("simt.isnan", 1, isnan)
_register_scalar_math_parser("simt.isinf", 1, isinf)
_register_scalar_math_parser("simt.isfinite", 1, isfinite)
_register_scalar_math_parser("simt.popcount", 1, popcount)
_register_scalar_math_parser("simt.mul_hi", 2, mul_hi)
_register_scalar_math_parser("simt.fmod", 2, fmod)
_register_scalar_math_parser("simt.fma", 3, fma)


@op_impl("simt.atomic_add")
def _parse_atomic_add(parser: Any, call: ast.Call) -> Expr:
    return _parse_atomic_call(parser, "simt.atomic_add", atomic_add, ("value",), call)


@op_impl("simt.atomic_sub")
def _parse_atomic_sub(parser: Any, call: ast.Call) -> Expr:
    return _parse_atomic_call(parser, "simt.atomic_sub", atomic_sub, ("value",), call)


@op_impl("simt.atomic_exch")
def _parse_atomic_exch(parser: Any, call: ast.Call) -> Expr:
    return _parse_atomic_call(parser, "simt.atomic_exch", atomic_exch, ("value",), call)


@op_impl("simt.atomic_max")
def _parse_atomic_max(parser: Any, call: ast.Call) -> Expr:
    return _parse_atomic_call(parser, "simt.atomic_max", atomic_max, ("value",), call)


@op_impl("simt.atomic_min")
def _parse_atomic_min(parser: Any, call: ast.Call) -> Expr:
    return _parse_atomic_call(parser, "simt.atomic_min", atomic_min, ("value",), call)


@op_impl("simt.atomic_inc")
def _parse_atomic_inc(parser: Any, call: ast.Call) -> Expr:
    return _parse_atomic_call(parser, "simt.atomic_inc", atomic_inc, ("limit",), call)


@op_impl("simt.atomic_dec")
def _parse_atomic_dec(parser: Any, call: ast.Call) -> Expr:
    return _parse_atomic_call(parser, "simt.atomic_dec", atomic_dec, ("limit",), call)


@op_impl("simt.atomic_cas")
def _parse_atomic_cas(parser: Any, call: ast.Call) -> Expr:
    return _parse_atomic_call(parser, "simt.atomic_cas", atomic_cas, ("compare", "value"), call)


@op_impl("simt.atomic_and")
def _parse_atomic_and(parser: Any, call: ast.Call) -> Expr:
    return _parse_atomic_call(parser, "simt.atomic_and", atomic_and, ("value",), call)


@op_impl("simt.atomic_or")
def _parse_atomic_or(parser: Any, call: ast.Call) -> Expr:
    return _parse_atomic_call(parser, "simt.atomic_or", atomic_or, ("value",), call)


@op_impl("simt.atomic_xor")
def _parse_atomic_xor(parser: Any, call: ast.Call) -> Expr:
    return _parse_atomic_call(parser, "simt.atomic_xor", atomic_xor, ("value",), call)
