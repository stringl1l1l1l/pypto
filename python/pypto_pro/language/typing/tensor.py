# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tensor wrapper type for PyPTO Language DSL."""

from __future__ import annotations

__all__ = ["Tensor"]


from collections.abc import Sequence

from pypto.pypto_impl.ir import DataType, Expr, MemRef, TensorLayout
from pypto_pro.language.typing.shape import _ShapePolicy

from ..._errors import InvalidOperation, InvalidType

EllipsisType = type(Ellipsis)


class Tensor:
    """Tensor type for PyPTO Language DSL.

    This class serves dual purposes:
    1. Type annotation helper for function signatures
    2. Runtime wrapper around IR Expr/Call objects

    Annotation mode (used in type hints):
        x: pl.Tensor[[64, 128], pl.DT_FP16]
        y: pl.Tensor[[64, 128], pl.DT_FP16, pl.NZ]

    Runtime mode (wraps IR expressions):
        tensor = pl.make_tensor(src, [64, 128], [128, 1], dtype=pl.DT_FP32)
        # Returns Tensor wrapping the Call expression

    Examples:
        >>> import pypto_pro.language as pl
        >>>
        >>> @pl.jit
        ... def my_func(
        ...     x: pl.Tensor[[64, 128], pl.DT_FP16, pl.NZ],
        ...     out: pl.Tensor[[64, 128], pl.DT_FP32],
        ... ):
        ...     # Build the kernel body with explicit tile operations.
        ...     ...
    """

    def __init__(
        self,
        shape: Sequence[int | _ShapePolicy | EllipsisType] | None = None,
        dtype: DataType | None = None,
        expr: Expr | None = None,
        layout: TensorLayout | None = None,
        memref: MemRef | None = None,
        _annotation_only: bool = False,
        direction: str | None = None,
    ):
        """Initialize Tensor.

        Args:
            shape: Shape (for annotation mode)
            dtype: Data type (for annotation mode)
            expr: IR expression to wrap (for runtime mode)
            layout: Optional tensor layout (ND, DN, NZ)
            memref: Optional memory reference
            _annotation_only: Whether this is annotation-only mode
            direction: Optional direction marker ("in"/"out") from pl.Input/pl.Output
        """
        self.direction = direction
        if expr is not None:
            if _annotation_only or shape is not None or dtype is not None:
                raise InvalidOperation("Runtime Tensor wrapping cannot include annotation arguments")
            self.expr = expr
            self.shape = None
            self.dtype = None
            self.layout = None
            self.memref = None
        elif shape is not None and dtype is not None:
            self.shape = shape
            self.dtype = dtype
            self.layout = layout
            self.memref = memref
            self.expr = None
        else:
            raise InvalidType(
                "Tensor must be initialized with either (shape, dtype) for annotations or expr for runtime wrapping"
            )

    @classmethod
    def __class_getitem__(cls, item: tuple) -> "Tensor":
        """Enable Tensor[[shape], dtype] and extended subscript syntax."""
        if not isinstance(item, tuple) or len(item) not in (2, 3, 4, 5):
            raise InvalidType(
                "Tensor requires [shape, dtype], [shape, dtype, layout_or_memref_or_view], "
                "[shape, dtype, layout, memref], or those plus a direction marker "
                "(pl.Input / pl.Output) notation"
            )

        # Direction marker may appear in any position after dtype; extract it so the
        # remaining elements follow the legacy layout/memref shapes.
        from pypto_pro.language import _DirectionMarker

        direction = None
        elts = list(item)
        for idx in range(len(elts) - 1, 1, -1):
            if isinstance(elts[idx], _DirectionMarker):
                direction = elts[idx].name.lower()
                elts.pop(idx)
        item = tuple(elts)

        if len(item) == 4:
            shape, dtype, layout, memref = item
            return cls(shape, dtype, layout=layout, memref=memref, _annotation_only=True, direction=direction)
        if len(item) == 3:
            shape, dtype, third = item
            if isinstance(third, MemRef):
                return cls(shape, dtype, memref=third, _annotation_only=True, direction=direction)
            return cls(shape, dtype, layout=third, _annotation_only=True, direction=direction)
        shape, dtype = item
        return cls(shape, dtype, _annotation_only=True, direction=direction)

    def __repr__(self) -> str:
        """String representation."""
        if self.expr is not None:
            return f"Tensor(expr={self.expr})"
        if self.layout is not None:
            return f"Tensor[[{self.shape}], {self.dtype}, {self.layout}]"
        return f"Tensor[[{self.shape}], {self.dtype}]"

    def unwrap(self) -> Expr:
        """Get underlying IR expression.

        Returns:
            The wrapped Expr/Call object

        Raises:
            ValueError: If called on an annotation-only Tensor
        """
        if self.expr is None:
            raise InvalidOperation("Cannot unwrap annotation-only Tensor (used in type hints)")
        return self.expr
