# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Expression parsing helpers for ASTParser."""

from __future__ import annotations

import ast
from typing import Any

from pypto.pypto_impl import ir
from pypto.pypto_impl.ir import DataType
from pypto_pro.ir import op as ir_op
from pypto_pro.ir._operators import make_binary as _make_binary
from pypto_pro.ir._utils import _normalize_expr

from ..._errors import (
    InvalidFormat,
    InvalidOperation,
    InvalidShape,
    InvalidType,
    InvalidVal,
    NameNotFound,
    NotSupported,
    OutOfRange,
    PyptoProError,
)
from ._expr_evaluator import ExprEvaluator
from ._tuple_type_registry import TupleTypeInfo
from ._utils import _const_int_value, _is_const_expr
from .diagnostics import (
    check_fits_dtype,
    make_const_int,
)


def _is_enum_value(value: Any) -> bool:
    """Whether ``value`` is an enum constant (DataType or a pybind enum).

    DataType is a py::class_ without ``__members__``, so it is matched by type;
    generic pybind enums (MemorySpace, RoundMode, ...) expose ``.value`` and a
    class-level ``__members__``. Mirrors the predicate used in parse_attribute /
    parse_name.
    """
    return isinstance(value, ir.DataType) or (
        hasattr(type(value), "__members__") and hasattr(value, "value")
    )


class ExpressionParserMixin:
    """Mixin containing expression, attribute, and subscript parsing."""

    def lookup_expr_by_name(self, name: str) -> Any | None:
        """Resolve a parser name to its current expression."""
        value = self.scope_manager.lookup_var_bounded(name)
        if not isinstance(value, ir.Var):
            return value

        constant = self.const_env.get(value.name)
        if constant is not None:
            return constant

        make_tuple = self.make_tuple_env.get(value.name)
        return make_tuple if make_tuple is not None else value

    def local_binding(self, name: str) -> "tuple[str, Any] | None":
        """How *name* is bound inside the kernel, or None if it is not bound there.

        The single definition of kernel-local name precedence, shared by name
        parsing and by compile-time evaluation so the two cannot disagree:

        * ``("const", <IR constant>)`` — folded in ``const_env`` (``addr = 0x10000``);
        * ``("runtime", <MakeTuple>)`` — an immutable tuple expression containing
          one or more runtime values;
        * ``("parse_time", <object>)`` — bound to a value that has no IR form and
          therefore only exists while parsing (``dt = pl.DT_FP16``);
        * ``("runtime", <IR expression>)`` — a value that only exists at run time.

        A name not bound in the kernel returns None and falls back to the
        enclosing Python scope, matching Python's own scoping.
        """
        value = self.lookup_expr_by_name(name)
        if value is None:
            return None
        if _is_const_expr(value):
            return ("const", value)
        return ("runtime", value) if isinstance(value, ir.Expr) else ("parse_time", value)

    @staticmethod
    def _const_scalar_value(expr: ir.Expr) -> bool | int | float | None:
        """Return a scalar constant's value, including the effect of constant casts."""
        found, value = ExprEvaluator.ir_to_python_value(expr)
        return value if found and isinstance(value, (bool, int, float)) else None

    @staticmethod
    def _const_truth(expr: ir.Expr) -> bool | None:
        value = ExpressionParserMixin._const_scalar_value(expr)
        return None if value is None else bool(value)

    @staticmethod
    def _make_scalar_constant(value: bool | int | float, dtype: DataType, span: ir.Span) -> ir.Expr:
        if dtype == DataType.BOOL:
            return ir.ConstBool(bool(value), span)
        if dtype.is_float():
            # Folding must not change a program's meaning: a result the dtype cannot represent would
            # otherwise become inf (or a wrapped integer, on the branch below) with no diagnostic.
            check_fits_dtype(float(value), dtype, subject="float constant", span=span)
            return ir.ConstFloat(float(value), dtype, span)
        return make_const_int(int(value), dtype, span=span)

    @staticmethod
    def _check_uniform_tuple_types(value_type: ir.TupleType, span: ir.Span) -> None:
        """Validate that all elements of a tuple share the same type (required for variable indexing)."""
        elem_types = list(value_type.types)
        if not elem_types:
            raise OutOfRange("Cannot index into empty tuple", span=span, parser_retry=True)
        first_type = elem_types[0]
        for i, t in enumerate(elem_types[1:], 1):
            if not ir.structural_equal(t, first_type, enable_auto_mapping=False):
                raise InvalidType(
                    f"Variable tuple index requires all elements to have the same type, "
                    f"but element 0 has type {first_type} and element {i} has type {t}",
                    span=span,
                    hint="Use a constant index to access elements of different types",
                    parser_retry=True,
                )

    @staticmethod
    def _is_pl_range_call(call: ast.Call) -> bool:
        """Check if call node is pl.range()."""
        func = call.func
        return isinstance(func, ast.Attribute) and func.attr == "range"

    @staticmethod
    def _try_const_fold_in(left, elements, is_not_in, span):
        """Decide ``x in (...)`` when every operand is known at parse time.

        The single place membership is evaluated. Handles IR constants and
        parse-time-only values alike, so both callers share it: ``_desugar_in_literal``
        uses it as an optimization before building an eq-chain.
        Returns None when any operand is a runtime value, leaving the caller to
        lower the comparison instead.
        """
        known, left_value = ExprEvaluator.ir_to_python_value(left)
        if not known:
            return None
        values = []
        for elt in elements:
            known, value = ExprEvaluator.ir_to_python_value(elt)
            if not known:
                return None
            values.append(value)
        found = any(left_value == value for value in values)
        return ir.ConstBool(found != is_not_in, span)

    def parse_evaluation_statement(self, stmt: ast.Expr) -> None:
        """Parse evaluation statement (EvalStmt).

        Evaluation statements represent operations executed for their side effects,
        with the return value discarded (e.g., synchronization barriers).

        Args:
            stmt: Expr AST node
        """
        expr = self.parse_expression(stmt.value, nested=False)
        span = self.span_tracker.get_span(stmt)

        # Void inline functions or python_var method calls (e.g. nbuf.advance())
        # return None or a non-IR sentinel — nothing to emit.
        if expr is None or not isinstance(expr, ir.Expr):
            return

        # Emit EvalStmt using builder method
        self.builder.eval_stmt(expr, span)

        # Auto-mutex: emit deferred mutex_unlock AFTER the op
        if self._auto_mutex:
            self._emit_auto_mutex_unlocks()

    def parse_expression(self, expr: ast.expr, *, nested: bool = True) -> Any:
        """Parse expression and return IR Expr.

        Args:
            expr: AST expression node
            nested: True (default) when this expression should materialize a
                ir.Call result through a let-binding so the outer expression
                only references the temporary Var. Callers at assignment RHS
                positions (``var = expr``, ``struct.field = expr``) and at
                EvalStmt positions should pass ``nested=False`` to avoid
                redundant ``_expr_tmp_N`` indirection.

        Returns:
            IR expression
        """
        if expr in self._parsed_expr_cache:
            return self._parsed_expr_cache[expr]
        self._current_node = expr
        try:
            result = self._parse_expression_node(expr)
        except PyptoProError as rejection:
            # Retry is opt-in: only a rejection raised with ``parser_retry=True``
            # may be re-evaluated as Python. Everything else — including a
            # position that deliberately rejected this value — propagates, so a
            # retry can never accept what the IR path refused.
            if not rejection.parser_retry:
                raise
            # Some compile-time expressions have no IR form to build at all: a
            # plain Python helper call, a dict lookup, a test against a DataType.
            # Evaluate those as Python and keep the constant; kernel-local names
            # resolve because the evaluator reads const_env as well as the closure.
            #
            # A value Python cannot produce re-raises the original IR error, and
            # one ``python_value_to_ir`` cannot lower reports its own type error,
            # so an unusable value never silently survives the retry.
            #
            # ast.Slice is not an expression position (it only appears inside a
            # rejected subscript); evaluating it would produce a Python slice
            # object and bury the real "Unsupported expression type" error.
            if isinstance(expr, ast.Slice):
                raise
            success, value = self.expr_evaluator.try_eval_expr(expr)
            if not success:
                raise
            if _is_enum_value(value):
                result = value
            else:
                result = self.expr_evaluator.python_value_to_ir(value, self.span_tracker.get_span(expr))

        if nested:
            result = self._materialize_nested_expr(result, self.span_tracker.get_span(expr))
        self._parsed_expr_cache[expr] = result
        return result

    def _parse_expression_node(self, expr: ast.expr) -> Any:
        """Dispatch an AST expression node to its parser."""
        if isinstance(expr, ast.Name):
            result = self.parse_name(expr)
        elif isinstance(expr, ast.Constant):
            result = self.parse_constant(expr)
        elif isinstance(expr, ast.BinOp):
            result = self.parse_binop(expr)
        elif isinstance(expr, ast.Compare):
            result = self.parse_compare(expr)
        elif isinstance(expr, ast.Call):
            result = self.parse_call(expr)
        elif isinstance(expr, ast.Attribute):
            result = self.parse_attribute(expr)
        elif isinstance(expr, ast.UnaryOp):
            result = self.parse_unaryop(expr)
        elif isinstance(expr, ast.List):
            result = self.parse_list(expr)
        elif isinstance(expr, ast.Tuple):
            result = self.parse_tuple_literal(expr)
        elif isinstance(expr, ast.Subscript):
            result = self.parse_subscript(expr)
        elif isinstance(expr, ast.BoolOp):
            result = self.parse_boolop(expr)
        elif isinstance(expr, ast.IfExp):
            result = self.parse_ifexp(expr)
        else:
            raise InvalidType(
                f"Unsupported expression type: {type(expr).__name__}",
                span=self.span_tracker.get_span(expr),
                hint="Use supported expressions like variables, constants, operations, or function calls",
                parser_retry=True,
            )
        return result

    def parse_name(self, name: ast.Name) -> ir.Expr | Any:
        """Parse variable name reference.

        Resolves names by checking the DSL scope first, then falling back
        to closure variables from the enclosing Python scope.

        Args:
            name: Name AST node

        Returns:
            IR expression (Var from scope, or constant/tuple from closure)
        """
        var_name = name.id
        # An inline helper may only resolve its own parameters and locals from
        # the DSL scope.  ``lookup_var_bounded`` behaves like the normal lookup
        # outside an inline scope, but stops at the inline boundary so a helper
        # cannot silently capture a caller IR variable.
        binding = self.local_binding(var_name)
        if binding is not None:
            value = binding[1]
            if not (isinstance(value, ir.Expr) and isinstance(value.type, ir.NoneType)):
                return value

        raise NameNotFound(
            f"Use of potentially undefined variable '{var_name}'",
            span=self.span_tracker.get_span(name),
            hint="Ensure the variable is defined before using it.",
            parser_retry=True,
        )

    def none_with_mutex_meta(self, span: ir.Span) -> ir.Expr:
        """The ``None`` sentinel, carrying mutex metadata that stands for "no buffer".

        A merge slot with no initial value is seeded with ``slot = None`` before its control
        flow. Treating that seed as a tile whose buffer is unknown gives the mutex-id companion
        a definition at the same place, so loop construction carries the companion in lockstep
        with the slot itself and a use after the loop locks the buffer the run actually selected.
        The empty candidate list marks the seed as holding no buffer of its own, and
        auto_mutex skips locking on it (see tile_mutex_lock_meta).
        """
        none_expr = self.builder.builder.none()
        if self._auto_mutex and none_expr not in self._tile_mutex_meta:
            self._tile_mutex_meta[none_expr] = ((ir.ConstInt(-1, DataType.INT64, span),), [])
        return none_expr

    def tile_mutex_lock_meta(self, expr):
        """Mutex metadata for *expr*, but only when it names a buffer worth locking.

        Propagation sites want the metadata as recorded; lock sites want it only when there are
        candidate ids, so the ``None`` seed (which carries none) never turns into a lock.
        """
        meta = self._tile_mutex_meta.get(expr)
        return meta if meta is not None and meta[1] else None

    def parse_constant(self, const: ast.Constant) -> ir.Expr:
        """Parse constant value.

        Args:
            const: Constant AST node

        Returns:
            IR constant expression
        """
        span = self.span_tracker.get_span(const)
        value = const.value

        if isinstance(value, bool):
            return ir.ConstBool(value, span)
        elif isinstance(value, int):
            return make_const_int(value, span=span)
        elif isinstance(value, float):
            return ir.ConstFloat(value, DataType.DEFAULT_CONST_FLOAT, span)
        elif isinstance(value, str):
            return value
        elif value is None:
            return self.none_with_mutex_meta(span)
        else:
            raise InvalidType(
                f"Unsupported constant type: {type(value)}",
                span=self.span_tracker.get_span(const),
                hint="Use int, float, or bool constants",
                parser_retry=True,
            )

    def _check_vf_scalar_division(self, op: ast.operator, left: ir.Expr, right: ir.Expr,
                                  span: ir.Span) -> None:
        """Reject scalar division-family ops inside a vector function.

        '/' always promotes to a scalar FloatDiv, and float '//', '%' emit
        float math; neither can be lowered inside the VF vector scope.
        """
        if self.inline_vf_depth == 0:
            return
        if not (isinstance(left.type, ir.ScalarType) and isinstance(right.type, ir.ScalarType)):
            return
        if isinstance(op, ast.Div) or (
            isinstance(op, (ast.FloorDiv, ast.Mod))
            and (left.type.dtype.is_float() or right.type.dtype.is_float())
        ):
            raise NotSupported(
                "Scalar division ('/', or float '//', '%') is not supported "
                "inside a vector function",
                span=span,
                hint="Use '//' on integer scalars for integer division, or "
                "hoist the division to the kernel level.",
            )

    def parse_binop(self, binop: ast.BinOp) -> ir.Expr:
        """Parse binary operation.

        Args:
            binop: BinOp AST node

        Returns:
            IR binary expression
        """
        span = self.span_tracker.get_span(binop)
        left = self.parse_expression(binop.left)
        right = self.parse_expression(binop.right)
        self._check_vf_scalar_division(binop.op, left, right, span)

        # Tile + offset: only supported in VF section as pointer arithmetic.
        # Non-VF sections require slice syntax (tile[r:r+h, c:c+w]) for sub-views.
        if isinstance(binop.op, ast.Add) and isinstance(left.type, ir.TileType):
            if self.inline_vf_depth == 0:
                raise NotSupported(
                    "Tile + offset is not supported outside VF section; "
                    "use tile[i:i+h, j:j+w] slice syntax for sub-view",
                    span=span,
                    hint="Replace `tile + offset` with `tile[row:row+h, :]` for offset access",
                )
            # VF section: lower to block.subview with original shape (VF codegen
            # only uses the offset as pointer arithmetic, shape is ignored).
            from pypto_pro.ir.op.block_ops import block_ir_op
            shape_tuple = ir.MakeTuple(left.type.shape, span)
            result = ir.create_op_call(
                block_ir_op("subview"),
                [left, right, shape_tuple],
                {},
                span,
            )
            return result

        # Raw pointer arithmetic: only `ptr + offset` is meaningful (→ pl.addptr).
        # Every other operator on a pointer (-, *, /, //, %) is forbidden.
        if isinstance(left.type, ir.PtrType):
            if isinstance(binop.op, ast.Add):
                # Sub-byte element types (INT4/UINT4/FP4/FP4E2M1/FP4E1M2/HF4) pack two elements per
                # byte, so an element offset cannot address a half-byte and there is
                # no valid C element type to lower to. Forbid `ptr + offset` on them.
                if left.type.dtype.get_bit() < 8:
                    raise NotSupported(
                        f"Pointer arithmetic ('ptr + offset') is not supported on the "
                        f"sub-byte element type '{left.type.dtype.to_string()}'",
                        span=span,
                        hint="Offsetting by elements cannot address a half-byte. "
                        "Reinterpret the pointer as a byte-addressable dtype via "
                        "pl.make_ptr (e.g. pl.DT_UINT8) before pointer arithmetic.",
                        parser_retry=True,
                    )
                return ir_op.ptr.addptr(left, right, span=span)
            raise NotSupported(
                f"Unsupported operator '{type(binop.op).__name__}' on a pointer (pl.Ptr)",
                span=span,
                hint="Only 'ptr + offset' is supported for pointer arithmetic (equivalent to pl.addptr(ptr, offset)).",
                parser_retry=True,
            )

        # Map AST operators to IR builder names. Everything is routed through
        # ``make_binary`` so mixed int/float operands (e.g. ``1.0 / G`` or ``1.0 + G``
        # with ``G`` an index Var) get Python-consistent promotion to float.
        op_map = {
            ast.Add: "add",
            ast.Sub: "sub",
            ast.Mult: "mul",
            ast.Div: "truediv",
            ast.FloorDiv: "floordiv",
            ast.Mod: "mod",
            ast.BitAnd: "bit_and",
            ast.BitOr: "bit_or",
            ast.BitXor: "bit_xor",
            ast.LShift: "bit_shift_left",
            ast.RShift: "bit_shift_right",
        }

        op_type = type(binop.op)
        if op_type not in op_map:
            raise NotSupported(
                f"Unsupported binary operator: {op_type.__name__}",
                span=self.span_tracker.get_span(binop),
                hint="Use supported operators: +, -, *, /, //, %, &, |, ^, <<, >>",
                parser_retry=True,
            )

        op_name = op_map[op_type]
        operation = _make_binary(op_name, left, right, span)
        folded = self._fold_const_binop(op_name, operation, span)
        return folded if folded is not None else operation

    def parse_compare(self, compare: ast.Compare) -> ir.Expr:
        """Parse comparison operation.

        Args:
            compare: Compare AST node

        Returns:
            IR comparison expression
        """
        if len(compare.ops) != 1 or len(compare.comparators) != 1:
            raise NotSupported(
                "Only simple comparisons supported",
                span=self.span_tracker.get_span(compare),
                hint="Use single comparison operators like: a < b, not chained comparisons",
            )

        span = self.span_tracker.get_span(compare)
        op_type = type(compare.ops[0])

        # Dispatch `in` / `not in` to the desugar path: they are not scalar
        # binary comparisons, so they cannot enter the op_map below.
        if op_type in (ast.In, ast.NotIn):
            return self._parse_in_operator(compare, span)

        # ── Handle 'is None' / 'is not None' for pointer null checks ──
        # Converts to runtime comparison: cast(ptr, UINT64) == 0 / cast(ptr, UINT64) != 0
        # This allows optional pl.Ptr parameters to be checked at runtime without
        # compile-time constant folding or multiple kernel variants.
        if op_type in (ast.Is, ast.IsNot):
            comparator = compare.comparators[0]
            if isinstance(comparator, ast.Constant) and comparator.value is None:
                # Left side must be a simple variable name (typically a pl.Ptr or pl.Tensor parameter)
                if not isinstance(compare.left, ast.Name):
                    raise InvalidVal(
                        "'is None' only supported on simple variable names",
                        span=span,
                        hint="Use 'param_name is None' for null checks on pl.Ptr parameters",
                    )
                # Parse the left operand (must be a pl.Ptr parameter)
                left = self.parse_expression(compare.left)
                # 'is None' only makes sense for pointer parameters. pl.Tensor is a
                # descriptor (ptr + shape); null-checking it has no meaning and the
                # generated code would reference an undeclared identifier. Guide users
                # to pl.Ptr for optional inputs (reviewer guidance: optional inputs
                # should be declared as pl.Ptr since pl.Tensor requires a shape even
                # when the argument is None, making the shape meaningless).
                if not isinstance(left.type, ir.PtrType):
                    raise InvalidVal(
                        "'is None' / 'is not None' is only supported on pl.Ptr parameters",
                        span=span,
                        hint="Use pl.Ptr[dtype] for optional pointer inputs; "
                        "pl.Tensor requires a shape even when the argument is None",
                        parser_retry=True,
                    )
                # Cast pointer to UINT64 for comparison (IR requires ScalarType for eq/ne)
                left_as_int = ir.cast(left, ir.DataType.UINT64, span)
                zero = ir.ConstInt(0, ir.DataType.UINT64, span)
                # Generate runtime comparison: cast(ptr, UINT64) == 0 or != 0
                op = "eq" if op_type is ast.Is else "ne"
                return _make_binary(op, left_as_int, zero, span)
            else:
                raise InvalidVal(
                    "'is' / 'is not' only supported with None",
                    span=span,
                    hint="Use '==' for value comparison, or 'param is None' for pointer null checks",
                    parser_retry=True,
                )

        # ── Standard comparison operators (==, !=, <, <=, >, >=) ──
        left = self.parse_expression(compare.left)
        right = self.parse_expression(compare.comparators[0])

        # Compile-time enum comparison (e.g. dtype generalization: ``x_dtype ==
        # pl.DT_FP16``). Enum operands (DataType / pybind enum) are Python objects,
        # not ir.Expr, so they cannot go through make_binary/ir.eq; fold ``==`` /
        # ``!=`` here into a ConstBool. Only equality is meaningful for enums.
        if _is_enum_value(left) and _is_enum_value(right):
            if op_type is ast.Eq:
                return ir.ConstBool(left == right, span)
            if op_type is ast.NotEq:
                return ir.ConstBool(left != right, span)
            raise NotSupported(
                f"Unsupported comparison {op_type.__name__} between enum values",
                span=span,
                hint="Only == and != are supported for enum comparisons",
                parser_retry=True,
            )

        # Comparisons also promote mixed int/float operands to float via make_binary.
        op_map = {
            ast.Eq: "eq",
            ast.NotEq: "ne",
            ast.Lt: "lt",
            ast.LtE: "le",
            ast.Gt: "gt",
            ast.GtE: "ge",
        }

        if op_type not in op_map:
            raise NotSupported(
                f"Unsupported comparison: {op_type.__name__}",
                span=span,
                hint="Use supported comparisons: ==, !=, <, <=, >, >=",
                parser_retry=True,
            )

        op_name = op_map[op_type]
        operation = _make_binary(op_name, left, right, span)
        folded = self._fold_const_binop(op_name, operation, span)
        return folded if folded is not None else operation

    def parse_unaryop(self, unary: ast.UnaryOp) -> ir.Expr:
        """Parse unary operation.

        Args:
            unary: UnaryOp AST node

        Returns:
            IR unary expression
        """
        span = self.span_tracker.get_span(unary)
        operand = self.parse_expression(unary.operand)

        op_map = {
            ast.USub: ("neg", ir.neg),
            ast.Not: ("not", ir.not_),
            ast.Invert: ("invert", ir.bit_not),
            ast.UAdd: ("pos", lambda value, _span: value),
        }
        op_type = type(unary.op)
        if op_type not in op_map:
            raise NotSupported(
                f"Unsupported unary operator: {op_type.__name__}",
                span=self.span_tracker.get_span(unary),
                hint="Use supported unary operators: +, -, not, ~",
                parser_retry=True,
            )

        op_name, make_operation = op_map[op_type]
        operation = make_operation(operand, span)
        folded = self._fold_const_unaryop(op_name, operation, span)
        return folded if folded is not None else operation

    def parse_boolop(self, expr: ast.BoolOp) -> ir.Expr:
        span = self.span_tracker.get_span(expr)
        bool_dtype = DataType.BOOL
        if isinstance(expr.op, ast.And):
            fold_fn = ir.And
        elif isinstance(expr.op, ast.Or):
            fold_fn = ir.Or
        else:
            raise InvalidType(
                f"Unsupported boolean operator: {type(expr.op).__name__}",
                span=span,
                parser_retry=True,
            )

        result = self.parse_expression(expr.values[0])
        for value_node in expr.values[1:]:
            result_truth = self._const_truth(result)
            if isinstance(expr.op, ast.And) and result_truth is False:
                return ir.ConstBool(False, span)
            if isinstance(expr.op, ast.Or) and result_truth is True:
                return ir.ConstBool(True, span)
            operand = self.parse_expression(value_node)
            operand_truth = self._const_truth(operand)
            if result_truth is not None and operand_truth is not None:
                value = (
                    result_truth and operand_truth
                    if isinstance(expr.op, ast.And)
                    else result_truth or operand_truth
                )
                result = ir.ConstBool(value, span)
            else:
                result = fold_fn(result, operand, bool_dtype, span)
        return result

    def _reject_ternary_branch(self, value: Any, node: ast.expr, branch: str, test_node: "ast.expr | None"):
        """The one diagnostic for a ternary branch that has no value to select.

        ``test_node`` is the condition when the ternary became a runtime select, and
        None when it const-folded. The distinction is the whole message: a
        compile-time enum is a legal branch once the condition folds (parse_ifexp
        returns it directly), so it is the condition — not the branch — that makes
        the same source text illegal here, and the diagnostic has to say so.
        """
        written = ast.unparse(node)
        if test_node is not None and _is_enum_value(value):
            raise InvalidOperation(
                f"'{written}' has no runtime value, so a runtime condition cannot select it",
                span=self.span_tracker.get_span(node),
                hint=f"'{ast.unparse(test_node)}' is a runtime value, which makes this ternary a "
                "runtime select rather than a compile-time choice. Give it a compile-time "
                "condition — a Python constant, or a tiling_key field, which "
                "@pl.jit(tiling_key=...) bakes to one constant per specialization — or write a "
                "branch that has a runtime value.",
                parser_retry=True,
            )
        raise InvalidVal(
            f"the {branch} branch of this ternary has no value to select, got '{written}'",
            span=self.span_tracker.get_span(node),
            hint="Each branch must be a scalar, tile or tensor expression. A call that performs "
            "an action rather than producing a value (pl.load(...), pl.store(...)) and a type "
            "descriptor (pl.TileType(...)) have nothing to select.",
            parser_retry=True,
        )

    def parse_ifexp(self, expr: ast.IfExp) -> ir.Expr:
        if self.inline_vf_depth > 0:
            raise NotSupported(
                "Ternary conditional expressions are not supported inside a "
                "vector function",
                span=self.span_tracker.get_span(expr),
                hint="Use an if/else statement instead, or compute the scalar "
                "arithmetically (pl.min/pl.max on comparisons).",
            )
        condition = self.parse_expression(expr.test)

        if isinstance(condition, (ir.ConstBool, ir.ConstInt)):
            is_true = condition.value if isinstance(condition, ir.ConstBool) else condition.value != 0
            chosen = expr.body if is_true else expr.orelse
            result = self.parse_expression(chosen, nested=False)
            if _is_enum_value(result):
                return result
            if not isinstance(result, ir.Expr):
                self._reject_ternary_branch(result, chosen, "chosen", None)
            return result

        for branch, branch_name in ((expr.body, "then"), (expr.orelse, "else")):
            success, branch_value = self.expr_evaluator.try_eval_expr(branch)
            if success and _is_enum_value(branch_value):
                self._reject_ternary_branch(branch_value, branch, branch_name, expr.test)

        tmp_name = f"_ifexpr_tmp_{self._ifexpr_tmp_counter}"
        self._ifexpr_tmp_counter += 1
        then_assign = ast.copy_location(
            ast.Assign(targets=[ast.Name(id=tmp_name, ctx=ast.Store())], value=expr.body),
            expr.body,
        )
        else_assign = ast.copy_location(
            ast.Assign(targets=[ast.Name(id=tmp_name, ctx=ast.Store())], value=expr.orelse),
            expr.orelse,
        )
        lowered_if = ast.copy_location(
            ast.If(test=expr.test, body=[then_assign], orelse=[else_assign]),
            expr,
        )
        ast.fix_missing_locations(lowered_if)
        self.parse_if_statement(lowered_if)
        result = self.lookup_expr_by_name(tmp_name)
        if not isinstance(result, ir.Expr):
            branch = expr.body if result is None else expr.orelse
            self._reject_ternary_branch(result, branch, "runtime", expr.test)
        return result

    def make_named_tuple(
        self,
        elements: list,
        field_names,
        span: ir.Span,
        *,
        name: str | None = None,
    ) -> ir.Expr:
        """Build a named ``MakeTuple`` and register its semantic tuple metadata."""
        field_names = tuple(field_names)
        mt = ir.MakeTuple(elements, span)
        self.tuple_type_registry.register_named_tuple(mt.type, field_names, name=name)
        return mt

    def classify_tuple_type(self, tuple_type: ir.TupleType) -> TupleTypeInfo:
        """Classify a tuple type by its parser-session kind, name, and fields."""
        return self.tuple_type_registry.classify(tuple_type)

    def named_fields(self, expr) -> list[str]:
        """Return parser-registered field names for a named tuple or struct expression."""
        if not (isinstance(expr, ir.Expr) and isinstance(expr.type, ir.TupleType)):
            return []
        return list(self.classify_tuple_type(expr.type).fields)

    def register_struct_type(self, expr: ir.Expr, struct_name: str, field_names) -> ir.Expr:
        """Register a struct's semantic tuple metadata in IRDebugInfo."""
        expr_type = expr.type
        if not isinstance(expr_type, ir.TupleType):
            raise InvalidVal(f"Struct '{struct_name}' must have TupleType")
        self.tuple_type_registry.register_struct(expr_type, struct_name, field_names)
        return expr

    def lower_attr_access(self, base: ir.Expr, field_name: str, span: ir.Span):
        """Lower attribute read ``base.field`` to its named tuple element.

        Resolves the field index from the parser's tuple type registry. When *base* is a parse-time
        constant ``MakeTuple`` (never for struct_array, whose elements are
        mutable structs with backing-array alias semantics), constant-folds the
        read to the static element instead of emitting a ``GetItemExpr``.
        Returns None if ``base`` is not a named tuple or has no such field
        (caller keeps its own error handling).
        """
        fields = self.named_fields(base)
        if not fields or field_name not in fields:
            return None
        idx = fields.index(field_name)
        if isinstance(base, ir.MakeTuple) and not self._is_struct_array_tuple(base):
            return base.elements[idx]
        return ir.GetItemExpr(base, ir.ConstInt(idx, DataType.INT64, span), span)

    def parse_attribute(self, attr: ast.Attribute) -> ir.Expr:
        """Parse attribute access.

        Args:
            attr: Attribute AST node

        Returns:
            IR expression
        """
        span = self.span_tracker.get_span(attr)

        # Generic pybind enum constant (pl.MemorySpace.Vec, pl.RoundMode.CAST_FLOOR,
        # pl.DT_FP16, pl.QuantMode.SYM, pl.STPhase.Partial, ...): evaluate the
        # attribute and, if it is an enum value, return the enum object. kwarg
        # consumers use it directly / the C++ boundary extracts .value as int.
        # Generic pybind enums (MemorySpace, RoundMode, ...) expose .value and a
        # class-level __members__; DataType is a py::class_ without __members__, so
        # match it by type. NOTE: enum objects are not IR Exprs, so they cannot be
        # used as positional op args or list elements (only as kwargs).
        success, val = self.expr_evaluator.try_eval_expr(attr)
        if success and _is_enum_value(val):
            return val

        # Attribute access on a parsed base (e.g. arr[0].field, arr[i].field, or
        # x.shape on a Tensor): Tensor.shape yields its shape MakeTuple; any other
        # base lowers through the named-field lookup.
        base_expr = self.parse_expression(attr.value)
        if attr.attr == "shape" and isinstance(base_expr, ir.Expr) and isinstance(base_expr.type, ir.TensorType):
            return ir.MakeTuple(list(base_expr.type.shape), span)
        if isinstance(base_expr, ir.Expr):
            lowered = self.lower_attr_access(base_expr, attr.attr, span)
            if lowered is not None:
                return lowered

        raise InvalidShape(
            f"Standalone attribute access not supported: {ast.unparse(attr)}",
            span=span,
            hint="Attribute access is supported on named tuples, structs, Tensor values (tensor.shape), "
            "tiling parameters, or enum constants",
            parser_retry=True,
        )

    def parse_list(self, list_node: ast.List) -> ir.MakeTuple:
        """Parse list literal into MakeTuple IR expression.


        Args:
            list_node: List AST node

        Returns:
            MakeTuple IR expression
        """
        span = self.span_tracker.get_span(list_node)
        elements = [self.parse_expression(elt) for elt in list_node.elts]
        return ir.MakeTuple(elements, span)

    def parse_tuple_literal(self, tuple_node: ast.Tuple) -> ir.MakeTuple:
        """Parse tuple literal like (x, y, z).

        Args:
            tuple_node: Tuple AST node

        Returns:
            MakeTuple IR expression
        """
        span = self.span_tracker.get_span(tuple_node)
        elements = [self.parse_expression(elt) for elt in tuple_node.elts]
        return ir.MakeTuple(elements, span)

    def parse_subscript(self, subscript: ast.Subscript) -> ir.Expr:
        span = self.span_tracker.get_span(subscript)

        if isinstance(subscript.value, ast.Attribute) and subscript.value.attr == "valid_shape":
            base_expr = self.parse_expression(subscript.value.value)
            if isinstance(base_expr, ir.Expr) and isinstance(base_expr.type, ir.TileType):
                return self._parse_tile_valid_shape_subscript(base_expr, subscript.slice, span)

        if isinstance(subscript.value, ast.Attribute) and subscript.value.attr == "shape":
            base_expr = self.parse_expression(subscript.value.value)
            if isinstance(base_expr, ir.Expr) and isinstance(base_expr.type, ir.TensorType):
                return self._parse_tensor_shape_subscript(base_expr, subscript.slice, span)

        value_expr = self.parse_expression(subscript.value)

        # A tile-group handle is a named tuple, so this must precede the TupleType
        # dispatch below: g[0] means "slot 0 of the group", not "field 0 of the handle".
        if self.is_tile_group(value_expr):
            return self.lower_group_subscript(value_expr, subscript.slice, span)

        value_type = value_expr.type

        # Tile/Tensor subscript: dispatch by index type
        #   - Integer index (no ':') → getval (scalar access, Tile and Tensor)
        #   - Slice index (has ':')  → sub-view (block.subview, Tile only)
        if isinstance(value_type, (ir.TensorType, ir.TileType)):
            has_slice = isinstance(subscript.slice, ast.Slice) or (
                isinstance(subscript.slice, ast.Tuple)
                and any(isinstance(e, ast.Slice) for e in subscript.slice.elts)
            )
            if has_slice:
                return self._parse_slice_subscript(value_expr, subscript.slice, span)
            # All-integer index: A[i, j] → getval(A, i*cols+j)
            index_expr = self._parse_scalar_subscript_index(value_expr, subscript.slice, span)
            from pypto_pro.ir.op.block_ops import _ir_getval
            result = _ir_getval(value_expr, index_expr, span=span)
            mutex_locked = False
            if self._auto_mutex:
                from ._op_pipeline import get_op_pipe
                pipe = get_op_pipe("getval")
                mutex_locked = self._emit_mutex_for_tile(value_expr, pipe, span, is_lock=True)
            if mutex_locked:
                result = self._materialize_nested_expr(result, span)
                self._emit_mutex_for_tile(value_expr, pipe, span, is_lock=False)
            return result

        if not isinstance(value_type, ir.TupleType):
            raise InvalidType(
                f"Subscript requires tuple, tile, or tensor type, got {type(value_type).__name__}",
                span=span,
                hint="Subscript access is supported on Tuple, Tile, and Tensor types",
                parser_retry=True,
            )

        # TupleType subscript: tuple[i] element access
        if isinstance(subscript.slice, ast.Constant):
            if not isinstance(subscript.slice.value, int):
                raise InvalidType(
                    "Tuple index must be an integer",
                    span=span,
                    hint="Use integer index like tuple[0]",
                )
        else:
            if isinstance(subscript.slice, ast.Tuple):
                raise InvalidShape(
                    "Multi-dimensional subscript is not supported for tuples",
                    span=span,
                    hint="Use a scalar index like tuple[0]",
                )
            self._check_uniform_tuple_types(value_type, span)

        index_expr = self.parse_expression(subscript.slice)
        if (
            isinstance(value_expr, ir.MakeTuple)
            and not self._is_struct_array_tuple(value_expr)
            and isinstance(index_expr, ir.ConstInt)
        ):
            index = index_expr.value
            if 0 <= index < len(value_expr.elements):
                return value_expr.elements[index]

        # CCE resolves a non-folded GetItem through the underlying MakeTuple's
        # backing array. struct_array stays as a Var here because it is not a
        # parser constant; ordinary tuples may remain folded MakeTuple values.
        item_expr = ir.GetItemExpr(value_expr, index_expr, span)
        # A subscript aliases the base's buffer, so carry the base's mutex
        # metadata onto the item.  auto_mutex then locks accesses to the item on
        # the base's buf_id; it is consumed when the item is bound to a var (see
        # _transfer_tile_sync_metadata).
        meta = self._tile_mutex_meta.get(value_expr)
        if meta is not None:
            self._tile_mutex_meta[item_expr] = meta
        return item_expr

    def _is_struct_array_tuple(self, tuple_value: ir.MakeTuple) -> bool:
        return tuple_value in self._struct_array_tuples

    def _update_var_envs(self, target: Any, value: Any) -> None:
        """Record parser-known values under the destination's physical SSA name."""
        if not isinstance(target, ir.Var):
            return
        if _is_const_expr(value):
            self.const_env[target.name] = value
        if isinstance(value, ir.MakeTuple) and not self._is_struct_array_tuple(value):
            self.make_tuple_env[target.name] = value

    def _fold_const_binop(self, op_name: str, operation: ir.BinaryExpr, span: ir.Span) -> ir.Expr | None:
        """Fold a validated binary scalar operation using its promoted operands."""
        left_value = self._const_scalar_value(operation.left)
        right_value = self._const_scalar_value(operation.right)
        if op_name in ("truediv", "floordiv", "mod") and right_value == 0:
            operator = {"truediv": "/", "floordiv": "//", "mod": "%"}[op_name]
            raise InvalidVal(
                f"Operator '{operator}' does not allow a zero divisor",
                span=span,
                hint="Use a nonzero divisor",
            )
        if left_value is None or right_value is None:
            return None
        dtype = operation.type.dtype
        binary_ops = {
            "add": lambda: left_value + right_value,
            "sub": lambda: left_value - right_value,
            "mul": lambda: left_value * right_value,
            "truediv": lambda: left_value / right_value,
            "floordiv": lambda: left_value // right_value,
            "mod": lambda: left_value % right_value,
            "bit_and": lambda: left_value & right_value,
            "bit_or": lambda: left_value | right_value,
            "bit_xor": lambda: left_value ^ right_value,
            "bit_shift_left": lambda: left_value << right_value,
            "bit_shift_right": lambda: left_value >> right_value,
            "min": lambda: min(left_value, right_value),
            "max": lambda: max(left_value, right_value),
        }
        if op_name in binary_ops:
            try:
                value = binary_ops[op_name]()
            except (ArithmeticError, ValueError):
                return None
            return self._make_scalar_constant(value, dtype, span)
        comparisons = {
            "eq": lambda: left_value == right_value,
            "ne": lambda: left_value != right_value,
            "lt": lambda: left_value < right_value,
            "le": lambda: left_value <= right_value,
            "gt": lambda: left_value > right_value,
            "ge": lambda: left_value >= right_value,
        }
        if op_name in comparisons:
            return ir.ConstBool(comparisons[op_name](), span)
        raise NotSupported(f"Unsupported constant-folding binary operator: {op_name}")

    def _fold_const_unaryop(self, op_name: str, operation: ir.Expr, span: ir.Span) -> ir.Expr | None:
        """Fold a validated unary scalar operation using its operand and result dtype."""
        operand = operation.operand if isinstance(operation, ir.UnaryExpr) else operation
        value = self._const_scalar_value(operand)
        if value is None:
            return None
        unary_ops = {
            "neg": lambda: -value,
            "not": lambda: not value,
            "invert": lambda: ~value,
            "pos": lambda: value,
        }
        return self._make_scalar_constant(unary_ops[op_name](), operation.type.dtype, span)

    def _desugar_in_literal(self, left, elements, is_not_in, span):
        """Desugar x in (a,b,c) -> Or-chain of eq; x not in -> And-chain of ne."""
        folded = self._try_const_fold_in(left, elements, is_not_in, span)
        if folded is not None:
            return folded
        if not elements:
            return ir.ConstBool(is_not_in, span)
        cmp_name = "ne" if is_not_in else "eq"
        fold_fn = ir.And if is_not_in else ir.Or

        result = _make_binary(cmp_name, left, elements[0], span)
        for elt in elements[1:]:
            result = fold_fn(result, _make_binary(cmp_name, left, elt, span), DataType.BOOL, span)
        return result

    def _desugar_in_range(self, left, range_call, is_not_in, span):
        """Desugar x in pl.range(start, stop, step).

        step == 1:  x >= start and x < stop
        step != 1:  x >= start and x < stop and (x - start) % step == 0
        """
        args = self._parse_range_call(range_call)
        start = _normalize_expr(args["start"], span)
        stop = _normalize_expr(args["stop"], span)
        step = _normalize_expr(args["step"], span)
        ge_start = _make_binary("ge", left, start, span)
        lt_stop = _make_binary("lt", left, stop, span)
        result = ir.And(ge_start, lt_stop, DataType.BOOL, span)
        step_is_one = (
            isinstance(step, ir.ConstInt) and step.value == 1 or isinstance(step, ir.ConstFloat) and step.value == 1.0
        )
        if not step_is_one:
            diff = _make_binary("sub", left, start, span)
            mod_val = _make_binary("mod", diff, step, span)
            zero = ir.ConstInt(0, DataType.INT64, span)
            result = ir.And(result, _make_binary("eq", mod_val, zero, span), DataType.BOOL, span)
        return ir.not_(result, span) if is_not_in else result

    def _parse_in_operator(self, compare: ast.Compare, span: ir.Span) -> ir.Expr:
        """Parse ``x in (...)`` / ``x not in (...)``.

        Supports three container forms, all desugared to existing IR ops:
          1. Literal tuple/list:  x in (a, b, c)      -> or-chain of eq
          2. pl.range() call:     x in pl.range(s,e,k) -> range bounds + modulo check
          3. Closure variable:    x in my_list          -> eval at compile time, expand to eq-chain
        """
        is_not_in = isinstance(compare.ops[0], ast.NotIn)
        left = self.parse_expression(compare.left)
        container = compare.comparators[0]

        if isinstance(container, ast.Call) and self._is_pl_range_call(container):
            return self._desugar_in_range(left, container, is_not_in, span)

        # A literal, a kernel-local list and a closure list all parse to a MakeTuple.
        parsed = self.parse_expression(container)
        if isinstance(parsed, ir.MakeTuple):
            return self._desugar_in_literal(left, list(parsed.elements), is_not_in, span)

        raise NotSupported(
            f"'{'not in' if is_not_in else 'in'}' only supports tuple/list literals, "
            f"pl.range(), or compile-time list/tuple variables, "
            f"got {ast.unparse(container)}",
            span=span,
            hint="Use: x in (a, b, c), x in pl.range(10), or x in <compile-time-list>",
        )

    def _parse_tensor_shape_subscript(
        self,
        base_expr: ir.Expr,
        index_node: ast.expr,
        span: ir.Span,
    ) -> ir.Expr:
        """Resolve ``tensor.shape[index]`` to the dimension stored in its TensorType.

        TensorType shape expressions are the logical tensor dimensions and are the
        source of truth for both static dimensions and dynamic shape ABI variables.
        No runtime tensor operation is emitted.
        """
        if not isinstance(base_expr.type, ir.TensorType):
            raise InvalidType(
                "tensor.shape requires TensorType input",
                span=span,
                hint="Use tensor.shape[index] only on Tensor values",
                parser_retry=True,
            )

        success, axis = self.expr_evaluator.try_eval_expr(index_node)
        if not success or type(axis) is not int:
            raise InvalidType(
                "tensor.shape index must be a compile-time integer",
                span=span,
                hint="Use tensor.shape[0], tensor.shape[-1], or a compile-time integer expression",
            )

        original_axis = axis
        tensor_type = base_expr.type
        rank = len(tensor_type.shape)
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            raise InvalidShape(
                f"shape index {original_axis} out of range for tensor of rank {rank}",
                span=span,
                parser_retry=True,
            )
        return tensor_type.shape[axis]

    def _parse_tile_valid_shape_subscript(
        self,
        base_expr: ir.Expr,
        index_node: ast.expr,
        span: ir.Span,
    ) -> ir.Expr:
        """Parse a runtime ``tile.valid_shape[axis]`` query."""
        success, axis = self.expr_evaluator.try_eval_expr(index_node)
        if not success or type(axis) is not int:
            raise InvalidType(
                "tile.valid_shape index must be a compile-time integer",
                span=span,
                hint="Use tile.valid_shape[0] or tile.valid_shape[1].",
            )

        rank = len(base_expr.type.shape)
        original_axis = axis
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            raise InvalidShape(
                f"valid_shape index {original_axis} out of range for Tile rank {rank}",
                span=span,
                parser_retry=True,
            )

        return ir.create_op_call("block.tile_valid_shape", [base_expr], {"axis": axis}, span)

    @staticmethod
    def _validate_subscript_value(
        value: ir.Expr,
        kind: str,
        axis: int,
        span: ir.Span,
        upper_bound: ir.Expr | None = None,
    ) -> int | None:
        """Validate an index or slice bound and return its static value when available."""
        value_type = getattr(value, "type", None)
        if not (isinstance(value_type, ir.ScalarType) and value_type.dtype.is_int()):
            raise InvalidType(
                f"{kind} for axis {axis} must be an integer scalar, got {value_type}",
                span=span,
                parser_retry=True,
            )

        value_int = _const_int_value(value)
        if value_int is not None and value_int < 0:
            raise InvalidVal(
                f"{kind} for axis {axis} must be non-negative, got {value_int}",
                span=span,
                hint="Use an index greater than or equal to 0; Tile and Tensor subscripts do not "
                "support Python-style negative indexing",
            )

        upper_value = _const_int_value(upper_bound) if upper_bound is not None else None
        if value_int is not None and upper_value is not None and value_int >= upper_value:
            raise OutOfRange(
                f"{kind} {value_int} for axis {axis} is out of range for dimension size {upper_value}",
                span=span,
                hint=f"Use an index in [0, {upper_value})",
                parser_retry=True,
            )
        return value_int

    def _parse_scalar_subscript_index(
        self,
        container_expr: ir.Expr,
        slice_node: ast.expr,
        span: ir.Span,
    ) -> ir.Expr:
        """Parse A[i, j, ...] into a linear offset expression for getval/setval.

        Multi-index: A[i, j] → i * N + j (row-major linearization).
        Single index: A[i] → i (only valid for 1D containers).
        """
        container_type = container_expr.type
        shape = container_type.shape

        # Single index A[x]: only valid for 1D containers (rank 1)
        if not isinstance(slice_node, ast.Tuple):
            if len(shape) != 1:
                if self.inline_vf_depth > 0:
                    raise NotSupported(
                        "Tile[x] in VF section is not supported; use `tile + x` for pointer offset",
                        span=span,
                        hint="Replace `tile[x]` with `tile + x`",
                    )
                raise InvalidShape(
                    f"Subscript A[x] requires 1D container, but got rank {len(shape)}; "
                    f"use {len(shape)} indices or add ':' for sub-view (Tile)",
                    span=span,
                    hint="Use A[i, j] for scalar access or A[x:, :] for sub-view",
                )
            index = self.parse_expression(slice_node)
            self._validate_subscript_value(index, "Subscript index", 0, span, shape[0])
            return index

        # A[i, j, ...] — multi-dimensional coordinate
        elts = slice_node.elts

        if len(elts) != len(shape):
            raise InvalidShape(
                f"Subscript has {len(elts)} indices but container has rank {len(shape)}",
                span=span,
                hint=f"Use {len(shape)} indices to match the container shape",
                parser_retry=True,
            )
        indices = [self.parse_expression(e) for e in elts]

        for axis, index in enumerate(indices):
            self._validate_subscript_value(index, "Subscript index", axis, span, shape[axis])

        offset = indices[-1]
        stride = shape[-1]
        for dim in range(len(indices) - 2, -1, -1):
            # Fold constant stride into a single ConstInt to avoid mul IR nodes
            stride_val = _const_int_value(stride)
            if stride_val == 1:
                product = indices[dim]
            else:
                product = indices[dim] * stride
            offset = product + offset
            if dim > 0:
                next_shape_val = _const_int_value(shape[dim])
                cur_stride_val = _const_int_value(stride)
                if next_shape_val is not None and cur_stride_val is not None:
                    stride = ir.ConstInt(next_shape_val * cur_stride_val, DataType.INT64, span)
                else:
                    stride = shape[dim] * stride
        return offset

    def _parse_slice_subscript(
        self,
        container_expr: ir.Expr,
        slice_node: ast.expr,
        span: ir.Span,
    ) -> ir.Expr:
        """Parse tile[r:, c:] into a block.subview op (Tile only, 2D).

        Result tile preserves the original shape (row_stride unchanged);
        codegen auto-emits SetValidShape with the sub-window dimensions.
        - If slice start reaches/exceeds tile shape → error.
        - If slice end exceeds tile shape → clamp to the tile shape.
        - If slice exceeds tile valid_shape → clamp to (valid_shape - start).
        """
        container_type = container_expr.type
        shape = container_type.shape

        if isinstance(container_type, ir.TensorType):
            raise NotSupported(
                "Tensor slice sub-view is not supported",
                span=span,
                hint="Use pl.load/pl.store with offset lists for tensor access",
            )

        if self._current_func_type in (ir.FunctionType.SimtVF, ir.FunctionType.SimtCallee):
            raise InvalidOperation(
                "Tile subview is not supported inside a SIMT function",
                span=span,
                hint="Access the Tile directly with tile[row, col].",
                parser_retry=True,
            )

        # VF section: tile slice 'tile[a:b, c:d]' is not supported.
        # Use load_align(tile, [row, col]) / store_align(tile, ..., [row, col]) instead.
        if self.inline_vf_depth > 0:
            raise NotSupported(
                "Tile slice 'tile[a:b, c:d]' is not supported in VF section; "
                "use load_align(tile, [row, col]) or store_align(tile, reg, mask, [row, col]) instead",
                span=span,
                hint="Replace `tile[a:a+1, b:b+1]` with `tile, [a, b]` in load_align/store_align",
            )

        if isinstance(container_type, ir.TileType):
            memref = getattr(container_type, 'memref', None)
            mem_space = getattr(memref, 'memory_space', None) if memref is not None else None
            hw_info = getattr(container_type, 'hardware_info', None)
            blayout = getattr(hw_info, 'blayout', None) if hw_info is not None else None
            slayout = getattr(hw_info, 'slayout', None) if hw_info is not None else None
            is_nd = (
                blayout is not None
                and blayout.name == 'row_major'
                and slayout is not None
                and slayout.name == 'none_box'
            )
            is_dn = (
                blayout is not None
                and blayout.name == 'col_major'
                and slayout is not None
                and slayout.name == 'none_box'
            )
            if mem_space is None or mem_space.name != 'Vec' or not (is_nd or is_dn):
                raise InvalidFormat(
                    "Tile slice is only supported on UB (Vec) tiles with ND/DN layout",
                    span=span,
                    hint="Use pl.load with offset or pl.move with offset for unsupported tiles",
                )

        if not isinstance(slice_node, ast.Tuple):
            raise NotSupported(
                "1D tile slice is not supported; use 2D slice like tile[i:i+h, j:j+w]",
                span=span,
                hint="Use tile[i:i+h, j:j+w] for sub-view access",
            )

        slices = slice_node.elts
        if len(slices) != len(shape):
            raise InvalidShape(
                f"Slice subscript has {len(slices)} dimensions but tile has rank {len(shape)}",
                span=span,
                hint=f"Use {len(shape)} indices to match the tile shape",
                parser_retry=True,
            )

        # valid_shape for clamping: compile-time (TileType.tile_view) or runtime
        # (set_validshape, tracked by tile var name).
        ct_valid = None
        tile_view = getattr(container_type, 'tile_view', None)
        if tile_view is not None:
            ct_valid = getattr(tile_view, 'valid_shape', None)
        rt_valid = None
        if isinstance(container_expr, ir.Var):
            rt_valid = self.get_tile_valid_shape(container_expr)
        dim_starts = []
        new_shape_exprs = []

        for i, s in enumerate(slices):
            if not isinstance(s, ast.Slice):
                raise NotSupported(
                    f"Tile slice does not support integer index in dimension {i}",
                    span=span,
                    hint="Tile slices must use ':' for all dimensions",
                )
            if s.step is not None:
                step = self.parse_expression(s.step)
                if _const_int_value(step) != 1:
                    raise InvalidShape(
                        f"Tile slice step for axis {i} must be the compile-time integer 1",
                        span=span,
                        hint="Omit the step or use a contiguous slice with step 1",
                    )
            start = ir.ConstInt(0, DataType.INT64, span) if s.lower is None else self.parse_expression(s.lower)
            dim_starts.append(start)
            start_val = self._validate_subscript_value(start, "Tile slice start", i, span)

            # Compute slice size: upper - start (upper defaults to shape[i]).
            # Clamp upper to shape[i] (Python slice semantics): a[16:77] on a
            # length-64 dim becomes a[16:64], not an error.
            upper = shape[i] if s.upper is None else self.parse_expression(s.upper)
            shape_val = _const_int_value(shape[i])
            upper_val = self._validate_subscript_value(upper, "Tile slice end", i, span)
            effective_upper_val = upper_val
            if shape_val is not None:
                effective_upper_val = (
                    shape_val if effective_upper_val is None else min(effective_upper_val, shape_val)
                )
            if shape_val is not None and upper_val is not None and upper_val > shape_val:
                upper = ir.ConstInt(shape_val, DataType.INT64, span)
                upper_val = shape_val
            if (
                start_val is not None
                and effective_upper_val is not None
                and start_val >= effective_upper_val
            ):
                raise InvalidOperation(
                    f"Tile slice for axis {i} must satisfy start < min(end, shape), "
                    f"got start={start_val}, end={upper_val}, shape={shape_val}",
                    span=span,
                    hint="Tile slices cannot be empty, reversed, or start outside the Tile shape",
                )
            size = upper if start_val == 0 else upper - start
            size_val = _const_int_value(size)

            for vs in (
                ct_valid[i] if ct_valid is not None and i < len(ct_valid) else None,
                rt_valid[i] if rt_valid is not None and i < len(rt_valid) else None,
            ):
                if vs is None:
                    continue
                vs_val = _const_int_value(vs)
                if vs_val is not None and vs_val >= 0 and start_val is not None:
                    remaining = vs_val - start_val
                    if remaining <= 0:
                        raise InvalidShape(
                            f"Tile slice start ({start_val}) must be less than valid_shape dim {i} ({vs_val})",
                            span=span,
                            hint=f"Use a slice start in [0, valid_shape[{i}])",
                        )
                    if size_val is not None:
                        if size_val > remaining:
                            size = ir.ConstInt(remaining, DataType.INT64, span)
                            size_val = remaining
                    else:
                        size = _make_binary("min_", size, ir.ConstInt(remaining, DataType.INT64, span), span)
                elif vs_val is None and start_val is not None:
                    # Dynamic valid_shape: emit runtime clamp min(size, vs - start)
                    remaining_expr = vs - start
                    size = _make_binary("min_", size, remaining_expr, span)

            new_shape_exprs.append(size)

        # Pass [row_start, col_start] as a tuple; codegen computes the byte offset
        # according to the tile's layout (ND row-major, DN col-major, NZ fractal).
        from pypto_pro.ir.op.block_ops import block_ir_op
        shape_tuple = ir.MakeTuple(new_shape_exprs, span)
        offset_tuple = ir.MakeTuple([dim_starts[0], dim_starts[1]], span)
        view_expr = ir.create_op_call(
            block_ir_op("subview"),
            [container_expr, offset_tuple, shape_tuple],
            {},
            span,
        )

        # Propagate tile mutex metadata for auto_mutex
        self._transfer_tile_sync_metadata(view_expr, container_expr)
        return view_expr

    def _mark_make_tuple_anchor(self, result) -> None:
        """Record that a MakeTuple already has an emitted assignment anchor."""
        if isinstance(result, ir.MakeTuple):
            self._anchored_make_tuples.add(result)


    def _materialize_nested_expr(self, result, span: ir.Span):
        """Materialize nested expressions and anchor nested MakeTuples.

        Non-trivial expressions return a temporary Var so an enclosing expression
        does not duplicate their computation. Vars and constants are already
        atomic and remain inline. A MakeTuple instead keeps its expression result:
        the temporary let is only a CCE backing-array anchor, so callers can
        continue folding the enclosing expression.

        Expressions with UnknownType are skipped to avoid materializing
        void/sentinel results (e.g. nbuf.advance() void calls).
        """
        if isinstance(result, ir.MakeTuple):
            if result not in self._anchored_make_tuples:
                name = f"_tuple_anchor_{self._expr_tmp_counter}"
                self._expr_tmp_counter += 1
                self.builder.let(name, result, span=span)
                self._mark_make_tuple_anchor(result)
            return result
        if isinstance(result, (ir.Var, ir.ConstInt, ir.ConstFloat, ir.ConstBool)):
            return result
        if not isinstance(result, ir.Expr):
            return result
        if isinstance(result.type, ir.UnknownType):
            return result
        name = f"_expr_tmp_{self._expr_tmp_counter}"
        self._expr_tmp_counter += 1
        value = self.builder.let(name, result, span=span)
        self._transfer_tile_sync_metadata(value, result)
        self._emit_auto_mutex_unlocks()
        return value

    def _route_ir_node_method(self, node: ast.Call):
        """Route ``xxx.f(...)`` to an op func when ``xxx`` is a tile-group handle.

        Tile-group handle: next/current/previous (each returns a bare tile).
        Returns the IR result, or None when not routed.
        """
        if not isinstance(node.func, ast.Attribute):
            return None
        method = node.func.attr
        if method not in ("next", "current", "previous"):
            return None
        obj = self.parse_expression(node.func.value)
        if isinstance(obj, ir.Expr) and self.is_tile_group(obj):
            return self._lower_group_accessor(obj, method, self.span_tracker.get_span(node))
        return None
