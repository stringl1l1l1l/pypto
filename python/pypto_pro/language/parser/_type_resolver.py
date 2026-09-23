# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Type annotation resolution for IR parsing."""

from __future__ import annotations

__all__ = ["TypeResolver"]


import ast
from collections.abc import Callable
import logging
import typing
from typing import TYPE_CHECKING, Any

from pypto.pypto_impl import ir
from pypto.pypto_impl.ir import DataType
from pypto_pro.language.typing.direction import TensorDirection

from ..._errors import (
    InvalidArgument,
    InvalidFormat,
    InvalidShape,
    InvalidType,
    InvalidVal,
    NotSupported,
    OutOfRange,
    PyptoProError,
    message_of,
)
from ._expr_evaluator import ExprEvaluator

if TYPE_CHECKING:
    from pypto_pro.runtime.shape_policy import BoundKernelSignature, TensorShapeSpec

    from ._span_tracker import SpanTracker


logger = logging.getLogger(__name__)


class TypeResolver:
    """Resolves Python type annotations to IR types."""

    _DTYPE_MAP: dict[str, DataType] = {
        "DT_FP4": DataType.FP4,
        "DT_FP8E4M3FN": DataType.FP8E4M3FN,
        "DT_FP8E5M2": DataType.FP8E5M2,
        "DT_FP8E8M0": DataType.FP8E8M0,
        "DT_FP4E2M1": DataType.FP4E2M1,
        "DT_FP4E1M2": DataType.FP4E1M2,
        "DT_FP16": DataType.FP16,
        "DT_FP32": DataType.FP32,
        "DT_BF16": DataType.BF16,
        "DT_HF4": DataType.HF4,
        "DT_HF8": DataType.HF8,
        "DT_INT4": DataType.INT4,
        "DT_INT8": DataType.INT8,
        "DT_INT16": DataType.INT16,
        "DT_INT32": DataType.INT32,
        "DT_INT64": DataType.INT64,
        "DT_UINT4": DataType.UINT4,
        "DT_UINT8": DataType.UINT8,
        "DT_UINT16": DataType.UINT16,
        "DT_UINT32": DataType.UINT32,
        "DT_UINT64": DataType.UINT64,
        "DT_BOOL": DataType.BOOL,
    }

    _SCALAR_UNSUPPORTED_DTYPE_NAMES: frozenset[str] = frozenset(
        {
            "DT_FP4",
            "DT_FP8E4M3FN",
            "DT_FP8E5M2",
            "DT_FP8E8M0",
            "DT_FP4E2M1",
            "DT_FP4E1M2",
            "DT_INT4",
            "DT_UINT4",
            "DT_HF4",
            "DT_HF8",
        }
    )

    _LAYOUT_MAP: dict[str, "ir.TensorLayout"] = {
        "ND": ir.TensorLayout.ND,
        "DN": ir.TensorLayout.DN,
        "NZ": ir.TensorLayout.NZ,
    }

    _MEMORY_SPACE_MAP: dict[str, "ir.MemorySpace"] = {
        "DDR": ir.MemorySpace.DDR,
        "Vec": ir.MemorySpace.Vec,
        "Mat": ir.MemorySpace.Mat,
        "Left": ir.MemorySpace.Left,
        "Right": ir.MemorySpace.Right,
        "Scaling": ir.MemorySpace.Scaling,
        "Acc": ir.MemorySpace.Acc,
        "Bias": ir.MemorySpace.Bias,
        "ScaleLeft": ir.MemorySpace.ScaleLeft,
        "ScaleRight": ir.MemorySpace.ScaleRight,
    }

    def __init__(
        self,
        expr_evaluator: ExprEvaluator,
        scope_lookup: Callable[[str], Any | None] | None = None,
        span_tracker: "SpanTracker | None" = None,
        bound_signature: "BoundKernelSignature | None" = None,
    ):
        """Initialize type resolver.

        Args:
            expr_evaluator: Evaluator for resolving expressions from closure variables
            scope_lookup: Callback to look up variables in the parser scope
                (for Scalar IR vars used in inline annotations)
            span_tracker: Optional span tracker for accurate source locations
        """
        self.expr_evaluator = expr_evaluator
        self.scope_lookup = scope_lookup
        self.span_tracker = span_tracker
        self.bound_signature = bound_signature
        self._parameter_name: str | None = None
        self._parameter_index: int | None = None
        self.param_directions: dict[int, TensorDirection] = {}

    @staticmethod
    def _resolve_shape_spec_and_rank(
        name: str,
        index: int,
        shape_annotation,
        actual: ir.TensorType,
        span,
    ) -> tuple["TensorShapeSpec", bool]:
        """Create TensorShapeSpec from annotation and check rank against actual shape."""
        from pypto_pro.runtime.shape_policy import TensorShapeSpec

        try:
            shape_spec = TensorShapeSpec.from_annotation(name, index, shape_annotation)
        except (TypeError, ValueError) as exc:
            raise InvalidShape(
                message_of(exc),
                span=span,
                parser_retry=True,
            ) from exc

        explicit_rank = len(shape_spec.dimensions) - (1 if shape_spec.has_ellipsis else 0)
        rank_matches = (
            len(actual.shape) >= explicit_rank if shape_spec.has_ellipsis else len(actual.shape) == explicit_rank
        )
        return shape_spec, rank_matches, explicit_rank

    @classmethod
    def to_ir_shape(cls, shape: list[int | ir.Expr]) -> list[int] | list[ir.Expr]:
        """Convert shape to format accepted by IR constructors.

        TensorType accepts either list[int] or list[Expr], not mixed.
        When the shape contains any Expr elements, all int elements are
        converted to ConstInt.

        Args:
            shape: Mixed list of int and ir.Expr dimensions

        Returns:
            Pure int list or pure Expr list
        """
        if all(isinstance(d, int) for d in shape):
            return shape

        # Convert all to Expr
        return [ir.ConstInt(d, DataType.INT64, ir.Span.unknown()) if isinstance(d, int) else d for d in shape]

    @classmethod
    def dtype_from_value(cls, value: int) -> DataType | None:
        """Resolve a DataType enum from its integer value."""
        for dtype in cls._DTYPE_MAP.values():
            if int(dtype) == value:
                return dtype
        return None

    @classmethod
    def _get_type_name(cls, node: ast.expr) -> str | None:
        """Extract the type name from an AST node referencing Tensor or Ptr.

        Handles both ``pl.Tensor`` (ast.Attribute) and bare ``Tensor`` (ast.Name).

        Args:
            node: AST expression to check

        Returns:
            Type name string if recognized, None otherwise
        """
        if isinstance(node, ast.Attribute) and node.attr in ("Tensor", "Ptr"):
            return node.attr
        if isinstance(node, ast.Name) and node.id in ("Tensor", "Ptr"):
            return node.id
        return None

    @classmethod
    def _validate_shape_value(cls, value: Any, source_name: str, span: ir.Span) -> list[int | ir.Expr]:
        """Validate a Python value as a shape (list/tuple of positive integers).

        Args:
            value: Python value to validate
            source_name: Description of value source for error messages
            span: Source span for error messages

        Returns:
            List of shape dimensions
        """
        if not isinstance(value, (list, tuple)):
            raise InvalidType(
                f"Shape '{source_name}' must be a list or tuple, got {type(value).__name__}",
                span=span,
                hint="Use a list like [64, 128] or a variable holding a list",
                parser_retry=True,
            )

        dims: list[int | ir.Expr] = []
        for i, elem in enumerate(value):
            if type(elem) is not int or elem <= 0:
                raise InvalidType(
                    f"Shape '{source_name}' element {i} must be a positive integer, got {elem!r}",
                    span=span,
                    parser_retry=True,
                )
            dims.append(elem)
        return dims

    @classmethod
    def _validate_dim_value(cls, value: Any, source_name: str, span: ir.Span) -> int | ir.Expr:
        """Validate a Python value as a single shape dimension.

        Args:
            value: Python value to validate
            source_name: Description of value source for error messages
            span: Source span for error messages

        Returns:
            int for static dimension, ir.Expr for dynamic
        """
        if type(value) is int and value > 0:
            return value
        raise InvalidType(
            f"Shape variable '{source_name}' must be a positive integer, got {value!r}",
            span=span,
            parser_retry=True,
        )

    @classmethod
    def _is_memref_node(cls, node: ast.expr) -> bool:
        """Check if an AST node is a pl.MemRef(...) call."""
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        return (isinstance(func, ast.Attribute) and func.attr == "MemRef") or (
            isinstance(func, ast.Name) and func.id == "MemRef"
        )

    def resolve_param_type(
        self,
        type_node: ast.expr,
        parameter_name: str | None = None,
        parameter_index: int | None = None,
    ) -> "ir.Type":
        """Resolve AST type annotation to ir.Type for function parameters.

        Args:
            type_node: AST expression representing the type annotation
            parameter_name: Source parameter name used by shape policies
            parameter_index: Stable parameter position used by direction metadata

        Returns:
            Resolved IR type

        Raises:
            NotSupported: If the annotation is a tuple, which a parameter cannot be
        """
        if parameter_index is not None:
            self.param_directions.pop(parameter_index, None)

        previous_parameter = self._parameter_name
        previous_index = self._parameter_index
        self._parameter_name = parameter_name
        self._parameter_index = parameter_index
        try:
            resolved = self.resolve_type(type_node)
        finally:
            self._parameter_name = previous_parameter
            self._parameter_index = previous_index
        if isinstance(resolved, ir.TupleType):
            raise NotSupported(
                "Parameter type cannot be a tuple",
                hint="Tuple types are only supported as return types",
                parser_retry=True,
            )

        return resolved

    def annotation_has_shape_policy(self, type_node: ast.expr | None) -> bool:
        """Return whether an annotation contains DYNAMIC, STATIC, or ellipsis."""
        if type_node is None:
            return False
        for node in ast.walk(type_node):
            if isinstance(node, ast.Attribute) and node.attr in {"DYNAMIC", "STATIC"}:
                return True
            if isinstance(node, ast.Name) and node.id in {"DYNAMIC", "STATIC"}:
                return True
            if isinstance(node, ast.Constant) and node.value is Ellipsis:
                return True
        success, annotation = self.expr_evaluator.try_eval_expr(type_node)
        if success:
            from pypto_pro.language.typing.shape import _ShapePolicy
            from pypto_pro.language.typing.tensor import Tensor

            def contains_policy(value: Any) -> bool:
                if isinstance(value, Tensor):
                    return any(isinstance(dim, _ShapePolicy) or dim is Ellipsis for dim in value.shape)
                return any(contains_policy(item) for item in typing.get_args(value))

            return contains_policy(annotation)
        return False

    def validate_policy_parameter_type(
        self,
        type_node: ast.expr,
        parameter_name: str,
        actual: ir.Type,
    ) -> None:
        """Validate a policy-bearing helper annotation against its call-site type."""
        success, annotation = self.expr_evaluator.try_eval_expr(type_node)
        if not success:
            raise InvalidVal(
                f"Cannot evaluate annotation for helper parameter '{parameter_name}'",
                span=self._get_span(type_node),
                parser_retry=True,
            )
        from pypto_pro.language.typing.tensor import Tensor
        from pypto_pro.runtime.shape_policy import FixedDim, StaticDim, StaticTail

        if not isinstance(annotation, Tensor) or not isinstance(actual, ir.TensorType):
            raise InvalidArgument(
                f"Helper parameter '{parameter_name}' policy annotation requires a Tensor argument",
                span=self._get_span(type_node),
                parser_retry=True,
            )
        if annotation.dtype != actual.dtype:
            raise InvalidType(
                f"Helper parameter '{parameter_name}' dtype mismatch: expected {annotation.dtype}, got {actual.dtype}",
                span=self._get_span(type_node),
                parser_retry=True,
            )
        shape_spec, rank_matches, explicit_rank = self._resolve_shape_spec_and_rank(
            parameter_name,
            0,
            annotation.shape,
            actual,
            self._get_span(type_node),
        )
        if not rank_matches:
            raise InvalidShape(
                f"Helper parameter '{parameter_name}' rank does not match {annotation.shape}",
                span=self._get_span(type_node),
                parser_retry=True,
            )
        explicit_rank = len(shape_spec.dimensions) - (1 if shape_spec.has_ellipsis else 0)
        for axis, dimension in enumerate(shape_spec.dimensions[:explicit_rank]):
            actual_dim = actual.shape[axis]
            if isinstance(dimension, FixedDim):
                if not isinstance(actual_dim, ir.ConstInt) or actual_dim.value != dimension.value:
                    raise InvalidShape(
                        f"Helper parameter '{parameter_name}' axis {axis} must equal {dimension.value}",
                        span=self._get_span(type_node),
                        parser_retry=True,
                    )
            elif isinstance(dimension, StaticDim) and not isinstance(actual_dim, ir.ConstInt):
                raise InvalidShape(
                    f"Helper parameter '{parameter_name}' STATIC axis {axis} requires a constant caller dimension",
                    span=self._get_span(type_node),
                    parser_retry=True,
                )
        if shape_spec.has_ellipsis:
            tail = shape_spec.dimensions[-1]
            if isinstance(tail, StaticTail):
                for axis in range(tail.start_axis, len(actual.shape)):
                    if not isinstance(actual.shape[axis], ir.ConstInt):
                        raise InvalidShape(
                            f"Helper parameter '{parameter_name}' ellipsis axis {axis} "
                            "requires a constant caller dimension",
                            span=self._get_span(type_node),
                            parser_retry=True,
                        )

    def parse_shape(self, shape_node: ast.expr) -> list[int | ir.Expr]:
        """Parse shape from AST node.

        Supports integer literals, variable names that resolve to int values
        from the enclosing scope and Scalar IR
        variables from the parser scope, and arbitrary expressions that
        evaluate to lists/tuples via ExprEvaluator.

        Args:
            shape_node: AST node representing shape (tuple or list)

        Returns:
            List of shape dimensions (int for static, ir.Expr for dynamic)

        Raises:
            InvalidType: If the shape node is not a list, tuple, or variable
            InvalidArgument: If a shape variable does not resolve
        """
        if isinstance(shape_node, (ast.Tuple, ast.List)):
            return self._parse_dim_elements(shape_node.elts)

        # Handle variable name or arbitrary expression that resolves to a list/tuple
        if isinstance(shape_node, ast.Name):
            # Try eval first -> handles both simple names and expressions
            success, value = self.expr_evaluator.try_eval_expr(shape_node)
            if success:
                return self._validate_shape_value(value, shape_node.id, self._get_span(shape_node))
            raise InvalidArgument(
                f"Unknown shape variable: {shape_node.id}",
                span=self._get_span(shape_node),
                hint="Use a list like [64, 128] or a variable holding a list",
                parser_retry=True,
            )

        # Try evaluating arbitrary expressions (e.g., get_shape(), dims[0:2])
        success, value = self.expr_evaluator.try_eval_expr(shape_node)
        if success:
            return self._validate_shape_value(value, ast.unparse(shape_node), self._get_span(shape_node))

        raise InvalidType(
            f"Shape must be a list, tuple, or variable: {ast.unparse(shape_node)}",
            hint="Use a list like [64, 128] or a variable holding a list",
            parser_retry=True,
        )

    def resolve_type(self, type_node: ast.expr) -> "ir.Type":
        """Resolve AST type annotation to an ir.Type.

        A ``tuple[T1, T2, ...]`` annotation resolves to a single ``ir.TupleType``.

        Args:
            type_node: AST expression representing the type annotation

        Returns:
            Corresponding IR type

        Raises:
            ValueError: If type annotation cannot be resolved
        """
        # Handle subscript notation: pl.Tensor[...], pl.Ptr[...], tuple[...]
        if isinstance(type_node, ast.Subscript):
            # Check for tuple[T1, T2, ...] return type annotation
            value = type_node.value
            if isinstance(value, ast.Name) and value.id == "tuple":
                return self._resolve_tuple_type(type_node)
            return self._resolve_subscript_type(type_node)

        # Handle pl.Tensor((64, 128), pl.DT_FP16) call notation
        if isinstance(type_node, ast.Call):
            return self._resolve_call_type(type_node)

        # A DT_* dtype is the scalar type annotation, e.g. pl.DT_INT64.
        if isinstance(type_node, ast.Attribute):
            dtype_name = type_node.attr
            if dtype_name.startswith("DT_") and dtype_name in self._DTYPE_MAP:
                dtype = self._DTYPE_MAP[dtype_name]
                if dtype_name in self._SCALAR_UNSUPPORTED_DTYPE_NAMES:
                    raise NotSupported(
                        f"Scalar type does not support dtype {dtype_name}; "
                        "low-precision types (FP4/FP4E2M1/FP4E1M2/FP8/FP8E4M3FN/FP8E5M2/"
                        "FP8E8M0/INT4/UINT4/HF4/HF8) are storage-only "
                        "and cannot be used in scalar expressions",
                        span=self._get_span(type_node),
                        hint="Use a supported scalar dtype: DT_BOOL, DT_INT8, DT_INT16, DT_INT32, "
                        "DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_FP16, DT_BF16, DT_FP32",
                        parser_retry=True,
                    )
                return ir.ScalarType(dtype)
            raise InvalidType(
                f"Incomplete type annotation: {ast.unparse(type_node)}",
                span=self._get_span(type_node),
                hint="Use pl.Tensor[[shape], dtype], pl.Ptr[dtype], or a dtype like pl.DT_INT64 for scalars",
                parser_retry=True,
            )

        raise InvalidType(
            f"Unsupported type annotation: {ast.unparse(type_node)}",
            hint="Use pl.Tensor[[shape], dtype], pl.Ptr[dtype], or a dtype like pl.DT_INT64 for scalars",
            parser_retry=True,
        )

    def resolve_dtype(self, dtype_node: ast.expr) -> DataType:
        """Resolve dtype annotation.

        Args:
            dtype_node: AST node representing dtype

        Returns:
            DataType enum value

        Raises:
            ValueError: If dtype cannot be resolved
        """
        span = self._get_span(dtype_node)

        # Handle pl.DT_FP16, pl.DT_FP32, etc.
        if isinstance(dtype_node, ast.Attribute):
            dtype_name = dtype_node.attr
            if dtype_name in self._DTYPE_MAP:
                return self._DTYPE_MAP[dtype_name]

            # Distinguish DataType.UNKNOWN from pl.UNKNOWN for error message quality
            if isinstance(dtype_node.value, ast.Name) and dtype_node.value.id == "DataType":
                raise InvalidType(
                    f"Unknown DataType: {dtype_name}",
                    span=span,
                    hint="Use a valid dtype like pl.DT_FP32, pl.DT_INT32, etc. Available: "
                    f"{', '.join(self._DTYPE_MAP.keys())}",
                    parser_retry=True,
                )

            raise InvalidType(
                f"Unknown dtype: {dtype_name}",
                span=span,
                hint="Use a valid dtype like pl.DT_FP32, pl.DT_INT32, etc. Available: "
                f"{', '.join(self._DTYPE_MAP.keys())}",
                parser_retry=True,
            )

        # Handle simple name like FP16 (if imported directly) or variable from closure
        if isinstance(dtype_node, ast.Name):
            dtype_name = dtype_node.id
            if dtype_name in self._DTYPE_MAP:
                return self._DTYPE_MAP[dtype_name]

            # Try evaluating via ExprEvaluator for DataType values from closure
            success, value = self.expr_evaluator.try_eval_expr(dtype_node)
            if success:
                if isinstance(value, DataType):
                    return value
                raise InvalidType(
                    f"Dtype variable '{dtype_name}' must be a DataType, got {type(value).__name__}",
                    span=span,
                    hint="Use a valid dtype like pl.DT_FP32, pl.DT_INT32, etc.",
                    parser_retry=True,
                )

            raise InvalidType(
                f"Unknown dtype: {dtype_name}",
                span=span,
                hint="Use a valid dtype like pl.DT_FP32, pl.DT_INT32, etc. Available: "
                f"{', '.join(self._DTYPE_MAP.keys())}",
                parser_retry=True,
            )

        raise InvalidType(
            f"Cannot resolve dtype: {ast.unparse(dtype_node)}",
            span=span,
            hint="Use pl.DT_FP32, pl.DT_INT32, or other supported dtype constants",
            parser_retry=True,
        )

    def resolve_layout(self, layout_node: ast.expr) -> "ir.TensorLayout":
        """Resolve layout annotation to ir.TensorLayout.

        Args:
            layout_node: AST node representing layout (e.g., pl.NZ, NZ, or a variable)

        Returns:
            TensorLayout enum value

        Raises:
            InvalidArgument: If the layout name is not a known layout
            InvalidType: If a layout variable holds something other than a TensorLayout
            InvalidFormat: If the layout node cannot be resolved at all
        """
        span = self._get_span(layout_node)

        if isinstance(layout_node, ast.Attribute):
            layout_name = layout_node.attr
            if layout_name in self._LAYOUT_MAP:
                return self._LAYOUT_MAP[layout_name]
            raise InvalidArgument(
                f"Unknown layout: {layout_name}",
                span=span,
                hint=f"Use a valid layout: {', '.join(self._LAYOUT_MAP.keys())}",
                parser_retry=True,
            )

        if isinstance(layout_node, ast.Name):
            layout_name = layout_node.id
            if layout_name in self._LAYOUT_MAP:
                return self._LAYOUT_MAP[layout_name]

            success, value = self.expr_evaluator.try_eval_expr(layout_node)
            if success:
                if isinstance(value, ir.TensorLayout):
                    return value
                raise InvalidType(
                    f"Layout variable '{layout_name}' must be a TensorLayout, got {type(value).__name__}",
                    span=span,
                    hint=f"Use a valid layout: {', '.join(self._LAYOUT_MAP.keys())}",
                    parser_retry=True,
                )

            raise InvalidArgument(
                f"Unknown layout: {layout_name}",
                span=span,
                hint=f"Use a valid layout: {', '.join(self._LAYOUT_MAP.keys())}",
                parser_retry=True,
            )

        raise InvalidFormat(
            f"Cannot resolve layout: {ast.unparse(layout_node)}",
            span=span,
            hint="Use pl.ND, pl.DN, or pl.NZ",
            parser_retry=True,
        )

    def resolve_type_if_memref(self, annotation: ast.expr | None) -> "ir.Type | None":
        """Resolve annotation type only when it contains MemRef information.

        Returns the resolved type if the annotation includes a pl.MemRef(...)
        argument, or None to fall back to the default inferred type.

        Args:
            annotation: Type annotation AST node, or None if not annotated

        Returns:
            Resolved IR type with memref, or None if no memref in annotation
        """
        if not isinstance(annotation, ast.Subscript):
            return None
        slice_value = annotation.slice
        if not isinstance(slice_value, ast.Tuple):
            return None
        if not any(self._is_memref_node(elt) for elt in slice_value.elts):
            return None
        resolved = self.resolve_type(annotation)
        if isinstance(resolved, ir.TupleType):
            return None
        return resolved

    def resolve_memref(self, node: ast.expr) -> "ir.MemRef":
        """Resolve a pl.MemRef(memory_space, addr, size, id) AST call to ir.MemRef.

        Args:
            node: AST Call node for pl.MemRef(...)

        Returns:
            ir.MemRef instance

        Raises:
            InvalidType: If the node is not a pl.MemRef(...) call
            InvalidArgument: If the call does not have 3 or 4 arguments
        """
        if not isinstance(node, ast.Call):
            raise InvalidType(
                f"Expected pl.MemRef(...) call, got: {ast.unparse(node)}",
                hint="Use pl.MemRef(pl.MemorySpace.DDR, addr, size)",
                parser_retry=True,
            )

        span = self._get_span(node)

        if len(node.args) not in (3, 4):
            raise InvalidArgument(
                f"pl.MemRef requires 3 or 4 arguments (memory_space, addr, size[, id]), got {len(node.args)}",
                span=span,
                hint="Use pl.MemRef(pl.MemorySpace.DDR, 0, 1024)",
                parser_retry=True,
            )

        memory_space = self.resolve_memory_space(node.args[0])
        addr_expr = self._resolve_memref_addr(node.args[1])
        size = self._resolve_int_literal(node.args[2], "size", non_negative=True)

        if len(node.args) == 4:
            memref_id = self._resolve_int_literal(node.args[3], "id", non_negative=True)
            return ir.MemRef(memory_space, addr_expr, size, memref_id, span)

        return ir.MemRef(memory_space, addr_expr, size, span)

    def resolve_memory_space(self, node: ast.expr) -> "ir.MemorySpace":
        """Resolve a memory space AST node (e.g., pl.MemorySpace.DDR)."""
        span = self._get_span(node)

        if isinstance(node, ast.Attribute):
            name = node.attr
            if name in self._MEMORY_SPACE_MAP:
                return self._MEMORY_SPACE_MAP[name]
            raise InvalidArgument(
                f"Unknown memory space: {name}",
                span=span,
                hint=f"Use one of: {', '.join(self._MEMORY_SPACE_MAP.keys())}",
                parser_retry=True,
            )

        if isinstance(node, ast.Name):
            name = node.id
            if name in self._MEMORY_SPACE_MAP:
                return self._MEMORY_SPACE_MAP[name]

        raise InvalidVal(
            f"Cannot resolve memory space: {ast.unparse(node)}",
            span=span,
            hint="Use pl.MemorySpace.DDR, pl.MemorySpace.Vec, etc.",
            parser_retry=True,
        )

    def _resolve_annotation_shape(self, shape_node: ast.expr, type_name: str) -> list[int] | list[ir.Expr]:
        """Resolve a shaped annotation, applying a bound tensor policy when present."""
        parameter_name = self._parameter_name
        if type_name != "Tensor" or parameter_name is None:
            return self.to_ir_shape(self.parse_shape(shape_node))

        span = self._get_span(shape_node)
        if self.bound_signature is not None:
            bound_tensor = self.bound_signature.get_tensor(parameter_name)
            if bound_tensor is not None:
                return bound_tensor.to_ir_shape(span)

        success, raw_shape = self.expr_evaluator.try_eval_expr(shape_node)
        if not success:
            raise InvalidShape(
                f"Cannot resolve shape policy for tensor parameter '{parameter_name}'",
                span=span,
                hint="Use DYNAMIC, STATIC, positive integers, or a final ellipsis",
                parser_retry=True,
            )

        from pypto_pro.runtime.shape_policy import TensorShapeSpec

        try:
            shape_spec = TensorShapeSpec.from_annotation(parameter_name, 0, raw_shape)
            if shape_spec.requires_binding:
                raise InvalidShape(
                    f"Tensor parameter '{parameter_name}' uses STATIC or ellipsis without a concrete shape binding",
                    span=span,
                    hint="Use @pl.jit with runtime arguments, or provide static_shapes for STATIC/ellipsis dimensions",
                    parser_retry=True,
                )
            return shape_spec.bind(None).to_ir_shape(span)
        except PyptoProError:
            logger.debug("Re-raising parse error in _resolve_annotation_shape", exc_info=True)
            raise
        except (TypeError, ValueError) as exc:
            raise InvalidShape(message_of(exc), span=span, parser_retry=True) from exc

    def _validate_tensor_layout(
        self,
        shape: list[int] | list[ir.Expr],
        layout: "ir.TensorLayout",
        node: ast.expr,
    ) -> None:
        """Validate layout constraints known when a Tensor annotation is parsed."""
        if layout != ir.TensorLayout.NZ:
            return

        span = self._get_span(node)
        if len(shape) < 2:
            raise InvalidShape(
                "NZ Tensor requires rank >= 2",
                span=span,
                parser_retry=True,
            )

    def _resolve_tensor_direction(self, node: ast.expr) -> TensorDirection | None:
        """Resolve a public direction enum used in a Tensor annotation."""
        if isinstance(node, ast.Attribute):
            if node.attr not in {"Input", "Output"}:
                return None
        elif not isinstance(node, ast.Name):
            return None

        success, value = self.expr_evaluator.try_eval_expr(node)
        return value if success and isinstance(value, TensorDirection) else None

    def _resolve_subscript_type(self, subscript_node: ast.Subscript) -> ir.Type:
        """Resolve subscript type annotation.

        Supports:
        - pl.Tensor[[64, 128], pl.DT_FP16]
        - pl.Tensor[[64, 128], pl.DT_FP16, pl.NZ]
        - pl.Tensor[[64, 128], pl.DT_FP16, pl.MemRef(...)]
        - pl.Tensor[[64, 128], pl.DT_FP16, pl.NZ, pl.MemRef(...)]

        Args:
            subscript_node: AST Subscript node

        Returns:
            IR type

        Raises:
            InvalidArgument: If the subscript names an unknown type
            InvalidType: If the Tensor 4th argument is not pl.MemRef(...)
        """
        value = subscript_node.value
        type_name = self._get_type_name(value)

        if type_name is None:
            raise InvalidArgument(
                f"Unknown type in subscript: {ast.unparse(value)}",
                hint="Use pl.Tensor for tensor types or pl.Ptr for pointer types",
                parser_retry=True,
            )

        slice_value = subscript_node.slice

        if type_name == "Ptr":
            dtype = self.resolve_dtype(slice_value)
            return ir.PtrType(dtype)

        # Tensor: [shape, dtype], [shape, dtype, layout_or_memref], [shape, dtype, layout, memref]
        valid_counts = (2, 3, 4, 5) if type_name == "Tensor" else (2, 3)
        if not isinstance(slice_value, ast.Tuple) or len(slice_value.elts) not in valid_counts:
            if type_name == "Tensor":
                message = (
                    f"{type_name} subscript requires [shape, dtype], "
                    f"[shape, dtype, layout_or_memref], "
                    f"or [shape, dtype, layout, memref], got: {ast.unparse(slice_value)}"
                )
                hint = (
                    "Use pl.Tensor[[shape], dtype], pl.Tensor[[shape], dtype, layout], "
                    "pl.Tensor[[shape], dtype, pl.MemRef(...)], "
                    "or pl.Tensor[[shape], dtype, layout, pl.MemRef(...)] format"
                )
            else:
                message = (
                    f"{type_name} subscript requires [shape, dtype] or [shape, dtype, memref], "
                    f"got: {ast.unparse(slice_value)}"
                )
                hint = f"Use pl.{type_name}[[shape], dtype] or pl.{type_name}[[shape], dtype, pl.MemRef(...)]"
            raise InvalidArgument(
                message, span=self._get_span(slice_value), hint=hint, parser_retry=True,
            )

        shape_node = slice_value.elts[0]
        dtype_node = slice_value.elts[1]

        shape = self._resolve_annotation_shape(shape_node, type_name)
        dtype = self.resolve_dtype(dtype_node)

        # Direction is parser metadata only; remove it before resolving the IR TensorType.
        direction: TensorDirection | None = None
        elts: list[ast.expr] = []
        for element in slice_value.elts:
            element_direction = self._resolve_tensor_direction(element)
            if element_direction is None:
                elts.append(element)
            elif direction is not None:
                raise InvalidArgument(
                    "Tensor direction can be specified only once",
                    span=self._get_span(element),
                    hint="Use one pl.Input or pl.Output value",
                    parser_retry=True,
                )
            else:
                direction = element_direction
        if len(elts) not in (2, 3, 4):
            raise InvalidArgument(
                f"Tensor subscript has invalid arguments: {ast.unparse(slice_value)}",
                hint="Use pl.Tensor[[shape], dtype, optional layout/memref, optional pl.Input/pl.Output]",
                parser_retry=True,
            )
        slice_value = ast.Tuple(
            elts=elts,
            ctx=slice_value.ctx,
            lineno=slice_value.lineno,
            col_offset=slice_value.col_offset,
            end_lineno=getattr(slice_value, "end_lineno", None),
            end_col_offset=getattr(slice_value, "end_col_offset", None),
        )
        n_elts = len(elts)

        def with_direction(tensor_type: ir.TensorType) -> ir.TensorType:
            if direction is not None and self._parameter_index is not None:
                self.param_directions[self._parameter_index] = direction
            return tensor_type

        if n_elts == 2:
            return with_direction(ir.TensorType(shape, dtype))

        # 3 args: [shape, dtype, layout_or_memref_or_view]
        if n_elts == 3:
            third = slice_value.elts[2]
            # Disambiguate 3rd arg (backward compat)
            if self._is_memref_node(third):
                memref = self.resolve_memref(third)
                return with_direction(ir.TensorType(shape, dtype, memref))
            layout = self.resolve_layout(third)
            self._validate_tensor_layout(shape, layout, third)
            tensor_view = ir.TensorView([], layout)
            return with_direction(ir.TensorType(shape, dtype, None, tensor_view))

        # 4 args: [shape, dtype, layout, memref] -> Tensor only
        layout = self.resolve_layout(slice_value.elts[2])
        self._validate_tensor_layout(shape, layout, slice_value.elts[2])
        tensor_view = ir.TensorView([], layout)
        memref_node = slice_value.elts[3]
        if not self._is_memref_node(memref_node):
            raise InvalidType(
                "Tensor 4th argument must be pl.MemRef(...)",
                hint="Use pl.Tensor[[shape], dtype, layout, pl.MemRef(...)]",
                parser_retry=True,
            )
        memref = self.resolve_memref(memref_node)
        return with_direction(ir.TensorType(shape, dtype, memref, tensor_view))

    def _resolve_tuple_type(self, subscript_node: ast.Subscript) -> "ir.TupleType":
        """Resolve tuple[T1, T2, ...] return type annotation to a single TupleType.

        A tuple return is a single value (one MakeTuple expr), so it resolves to one
        ``ir.TupleType`` rather than a flattened list of element types.

        Args:
            subscript_node: AST Subscript node with tuple base

        Returns:
            A TupleType wrapping the element types
        """
        slice_value = subscript_node.slice
        elts = slice_value.elts if isinstance(slice_value, ast.Tuple) else [slice_value]

        types = []
        for elt in elts:
            resolved = self.resolve_type(elt)
            if isinstance(resolved, ir.TupleType):
                raise NotSupported(
                    "Nested tuple types are not supported",
                    span=self._get_span(elt),
                    hint="Use a flat tuple like tuple[pl.Tensor[...], pl.Tensor[...]]",
                    parser_retry=True,
                )
            types.append(resolved)
        return ir.TupleType(types)

    def _resolve_call_type(self, call_node: ast.Call) -> ir.Type:
        """Resolve a function call type annotation.

        Args:
            call_node: AST Call node

        Returns:
            IR type

        Raises:
            ValueError: If call cannot be resolved to a type
        """
        func = call_node.func
        type_name = self._get_type_name(func)

        resolvers = {"Tensor": self._resolve_tensor_type}
        resolver = resolvers.get(type_name) if type_name is not None else None
        if resolver is not None:
            return resolver(call_node)

        raise InvalidType(
            f"Unknown type constructor: {ast.unparse(func)}",
            hint="Use pl.Tensor[[shape], dtype] or a dtype like pl.DT_INT64 for scalars",
            parser_retry=True,
        )

    def _resolve_tensor_type(self, call_node: ast.Call) -> ir.TensorType:
        """Resolve a pl.Tensor((shape), dtype, optional direction) annotation."""
        result = self._resolve_shaped_type(call_node, "Tensor", ir.TensorType)
        if not isinstance(result, ir.TensorType):
            raise InvalidType("Expected TensorType result")
        return result

    def _resolve_shaped_type(
        self,
        call_node: ast.Call,
        type_name: str,
        type_ctor: type[ir.TensorType],
    ) -> ir.TensorType:
        """Resolve a tensor type from a call-style annotation.

        Args:
            call_node: AST Call node for the type constructor
            type_name: "Tensor" for error messages
            type_ctor: IR tensor type constructor

        Returns:
            Constructed IR type

        Raises:
            InvalidType: If the call does not carry both shape and dtype
        """
        if len(call_node.args) not in (2, 3):
            raise InvalidType(
                f"{type_name} type requires shape, dtype, and optional direction arguments, "
                f"got {len(call_node.args)}",
                hint=f"Use pl.{type_name}([shape], dtype) or pl.{type_name}([shape], dtype, pl.Input/pl.Output)",
                parser_retry=True,
            )

        direction = None
        if len(call_node.args) == 3:
            direction = self._resolve_tensor_direction(call_node.args[2])
            if direction is None:
                raise InvalidType(
                    "Tensor direction must be pl.Input or pl.Output",
                    span=self._get_span(call_node.args[2]),
                    parser_retry=True,
                )

        shape = self.to_ir_shape(self.parse_shape(call_node.args[0]))
        dtype = self.resolve_dtype(call_node.args[1])
        result = type_ctor(shape, dtype)
        if direction is not None and self._parameter_index is not None:
            self.param_directions[self._parameter_index] = direction
        return result

    def _parse_dim_elements(self, elts: list[ast.expr]) -> list[int | ir.Expr]:
        """Parse a list of dimension elements (int literal, variable, or evaluable expression).

        Used by both shape and stride parsing since both follow identical element syntax.

        Args:
            elts: List of AST expression nodes for each dimension

        Returns:
            List of dimensions
        """
        dims: list[int | ir.Expr] = []
        for elt in elts:
            if isinstance(elt, ast.Constant) and type(elt.value) is int:
                dims.append(self._validate_dim_value(elt.value, ast.unparse(elt), self._get_span(elt)))
            elif isinstance(elt, ast.Name):
                dims.append(self._resolve_shape_dim(elt))
            else:
                # Try evaluating arbitrary expressions (e.g., x * 2, len(shape))
                success, value = self.expr_evaluator.try_eval_expr(elt)
                if success:
                    dims.append(self._validate_dim_value(value, ast.unparse(elt), self._get_span(elt)))
                else:
                    raise InvalidType(
                        f"Dimension must be int literal, variable, or evaluable expression: {ast.unparse(elt)}",
                        hint="Use integer literals, variables, or expressions for dimensions",
                        parser_retry=True,
                    )
        return dims

    def _get_span(self, node: ast.AST) -> ir.Span:
        """Get span for an AST node, falling back to unknown."""
        if self.span_tracker is not None:
            return self.span_tracker.get_span(node)
        return ir.Span.unknown()

    def _resolve_shape_dim(self, name_node: ast.Name) -> int | ir.Expr:
        """Resolve a variable name used as a shape dimension.

        Resolution order:
        1. ExprEvaluator (compile-time integer from closure)
        2. Parser scope variables (Scalar IR vars from function body)

        Args:
            name_node: AST Name node for the variable

        Returns:
            int for compile-time constants, ir.Expr for dynamic dimensions
        """
        name = name_node.id
        span = self._get_span(name_node)

        # Fast path: direct dict lookup avoids compile+eval overhead for simple names
        if name in self.expr_evaluator.closure_vars:
            return self._validate_dim_value(self.expr_evaluator.closure_vars[name], name, span)

        # 2. Check parser scope (Scalar IR vars in function body)
        if self.scope_lookup:
            var = self.scope_lookup(name)
            if var is not None:
                return var

        raise InvalidArgument(
            f"Unknown shape variable: {name}",
            span=span,
            hint="Use a positive integer or a Scalar variable defined earlier",
            parser_retry=True,
        )

    def _resolve_memref_addr(self, node: ast.expr) -> "ir.Expr":
        """Resolve a MemRef address to an IR expression."""
        value = self._try_resolve_int(node)
        if value is not None:
            return ir.ConstInt(value, DataType.INT64, self._get_span(node))

        raise InvalidType(
            f"MemRef address must be an integer, got: {ast.unparse(node)}",
            span=self._get_span(node),
            hint="Use an integer value for the address, e.g., 0 or 1024",
            parser_retry=True,
        )

    def _resolve_int_literal(self, node: ast.expr, name: str, *, non_negative: bool = False) -> int:
        """Resolve an AST node to an integer literal."""
        value = self._try_resolve_int(node)
        if value is not None:
            if non_negative and value < 0:
                raise OutOfRange(
                    f"MemRef {name} must be >= 0, got: {value}",
                    span=self._get_span(node),
                    hint=f"Use a non-negative integer value for {name}",
                    parser_retry=True,
                )
            return value

        raise InvalidType(
            f"MemRef {name} must be an integer, got: {ast.unparse(node)}",
            span=self._get_span(node),
            hint=f"Use an integer value for {name}",
            parser_retry=True,
        )

    def _try_resolve_int(self, node: ast.expr) -> int | None:
        """Try to resolve an AST node to a Python int.

        Handles integer literals, unary negation of integer literals,
        and expressions evaluable via ExprEvaluator.

        Args:
            node: AST expression node

        Returns:
            Integer value, or None if the node cannot be resolved to an int
        """
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value

        is_neg_int_literal = (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, int)
        )
        if is_neg_int_literal:
            return -node.operand.value

        success, value = self.expr_evaluator.try_eval_expr(node)
        if success and isinstance(value, int):
            return value

        return None
