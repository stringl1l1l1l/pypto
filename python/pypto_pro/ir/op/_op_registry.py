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

"""Op dispatch registry for CallParserMixin.

Kept in ir/op/ so that ir/op/*.py can use relative imports, avoiding the
circular-import cycle:
  ir/op/__init__.py -> block_ops.py -> language/parser/op_registry.py
    -> language/__init__.py -> pypto_pro.ir (partially initialized)
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass, field

from pypto.pypto_impl import ir as _ir_core

from ..._errors import InvalidOperation, current_span

_OP_REGISTRY: dict[str, Callable] = {}


def op_impl(name: str) -> Callable:
    """Register a function as the parse handler for the given op_name.

    Decorated functions receive ``(self, call: ast.Call) -> ir.Expr`` where
    ``self`` is the ``CallParserMixin`` (i.e. ``ASTParser``) instance.
    """

    def decorator(func: Callable) -> Callable:
        if name in _OP_REGISTRY:
            raise InvalidOperation(f"Duplicate op_impl registration: '{name}'")
        _OP_REGISTRY[name] = func
        return func

    return decorator


@dataclass
class OpSpec:
    """Declarative parse specification for a template op handler.

    Attributes:
        builder: Builder function to forward parsed args/kwargs to.
            If None, ``_ir_core.create_op_call(ir_name, ...)`` is used directly.
        ir_name: IR op name used when ``builder`` is None.
        parse_args: Whether to parse positional args from the AST call.
        parse_kwargs: Whether to parse keyword args from the AST call.
        pre_hooks: Callables ``(self, call, kwargs) -> None`` invoked on the
            kwargs dict before forwarding to the builder.
    """

    builder: Callable | None = None
    ir_name: str | None = None
    parse_args: bool = True
    parse_kwargs: bool = True
    pre_hooks: list[Callable] = field(default_factory=list)


def _make_handler(spec: OpSpec) -> Callable:
    """Generate a unified parse handler from an OpSpec."""

    def handler(self, call: ast.Call):
        span = self.span_tracker.get_span(call)
        args = [self.parse_expression(a) for a in call.args] if spec.parse_args else []
        kwargs = self.parse_op_kwargs(call) if spec.parse_kwargs else {}
        for hook in spec.pre_hooks:
            hook(self, call, kwargs)
        # Publish where the caller wrote each argument, so a builder that rejects
        # one can point at that argument rather than at the whole call. Every
        # @op_impl operator is dispatched here, so this one site covers them all.
        with current_span(span, self.span_tracker.argument_spans(call)):
            if spec.builder is not None:
                return spec.builder(*args, **kwargs, span=span)
            return _ir_core.create_op_call(spec.ir_name, args, kwargs, span)

    return handler


def register_table(specs: dict[str, OpSpec]) -> None:
    """Batch-register declarative op specs into _OP_REGISTRY."""
    for op_name, spec in specs.items():
        if op_name in _OP_REGISTRY:
            raise InvalidOperation(f"Duplicate op_impl registration: '{op_name}'")
        _OP_REGISTRY[op_name] = _make_handler(spec)
