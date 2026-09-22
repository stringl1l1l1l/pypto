# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Assignment parsing helpers for ASTParser."""

from __future__ import annotations

import ast

from pypto.pypto_impl import ir
from pypto.pypto_impl.ir import DataType

from ..._errors import (
    InvalidArgument,
    InvalidOperation,
    InvalidShape,
    InvalidType,
    InvalidVal,
    NotSupported,
    OutOfRange,
)
from ._tuple_type_registry import TupleTypeKind


class AssignmentParserMixin:
    """Mixin containing assignment parsing methods for ``ASTParser``."""

    def parse_annotated_assignment(self, stmt: ast.AnnAssign) -> None:
        """Parse annotated assignment: var: type = value.

        Args:
            stmt: AnnAssign AST node
        """
        if not isinstance(stmt.target, ast.Name):
            raise NotSupported(
                "Only simple variable assignments supported",
                span=self.span_tracker.get_span(stmt.target),
                hint="Use a simple variable name for assignment targets",
            )

        var_name = stmt.target.id
        span = self.span_tracker.get_span(stmt)

        # Parse value expression
        if stmt.value is None:
            raise InvalidArgument(
                "Annotated assignment with no value is not supported",
                span=self.span_tracker.get_span(stmt),
                hint="Provide a value for the assignment",
                parser_retry=True,
            )
        value_expr = self.parse_expression(stmt.value, nested=False)

        # Use annotation type as override when it carries memref info
        annotation_type = self.type_resolver.resolve_type_if_memref(stmt.annotation)
        var = self.builder.let(var_name, value_expr, var_type=annotation_type, span=span)
        self._mark_make_tuple_anchor(value_expr)
        self._transfer_tile_sync_metadata(var, value_expr)

        # Register in scope
        self.scope_manager.define_var(var_name, var, span=span)
        self._update_var_envs(var, value_expr)

    def parse_assignment(self, stmt: ast.Assign) -> None:
        """Parse regular assignment: var = value or tuple unpacking.

        Args:
            stmt: Assign AST node
        """
        if len(stmt.targets) != 1:
            raise NotSupported(
                f"Unsupported assignment: {ast.unparse(stmt)}",
                span=self.span_tracker.get_span(stmt),
                hint="Use simple variable assignments or tuple unpacking",
            )

        target = stmt.targets[0]
        span = self.span_tracker.get_span(stmt)

        # Intercept VF op assignment form: reg = vf.xxx(...) or reg_lo, reg_hi = vf.xxx(...)
        # Only for compute ops that have dst registers (dst_count > 0).
        # Declaration ops (create_mask, RegTensor, compare, etc.) use normal return-value
        # assignment and must NOT be intercepted.
        vf_op = self._is_vf_op_call(stmt.value)
        if vf_op is not None:
            self._validate_op_scope(f"vf.{vf_op}", stmt.value)
            dst_count = self._get_vf_op_dst_count(vf_op)
            if dst_count is not None and dst_count > 0:
                self._parse_vf_assignment(target, stmt, vf_op, span)
                return

        if isinstance(target, ast.Tuple):
            self._parse_tuple_unpacking(target, stmt)
            return

        if isinstance(target, ast.Name):
            self._parse_name_assignment(target.id, stmt)
            return

        if isinstance(target, ast.Attribute):
            self._parse_struct_field_assignment(target, stmt, span)
            return

        if isinstance(target, ast.Subscript):
            self._parse_subscript_assignment(target, stmt, span)
            return

        raise NotSupported(
            f"Unsupported assignment target: {ast.unparse(stmt)}",
            span=span,
            hint="Only `var = ...`, `a.field = ...` (struct), `tensor[i] = ...`, "
            "or tuple unpacking are supported",
        )

    # Ops where dst type differs from src type — dtype kwarg is mandatory.
    _TYPE_CHANGING_OPS = frozenset({"astype", "muls_cast", "pack", "unpack", "bit_cast"})

    @staticmethod
    def _infer_src_dtype(parsed_src_args: list) -> DataType | None:
        """The dtype of the first source that carries one, or None if no source does."""
        for src_expr in parsed_src_args:
            dtype = getattr(getattr(src_expr, 'type', None), 'dtype', None)
            if dtype is not None:
                return dtype
        return None

    def _parse_vf_assignment(
        self,
        target: ast.expr,
        stmt: ast.Assign,
        vf_op_name: str,
        span: ir.Span,
    ) -> None:
        """Handle ``reg = vf.xxx(src, ...)`` assignment form.

        Transparently rewrites to the statement form by:
        1. Emitting a ``vf.reg_tensor(dtype=...)`` declaration for the LHS variable
           (if not already declared), so the CCE backend emits ``RegTensor<T> var;``.
        2. Inserting the LHS variable(s) as dst arg(s) at the front of the call.
        3. Emitting the call as an EvalStmt (side-effect).

        The dtype is inferred from the first source-register argument. For ops
        with only scalar sources (e.g. ``vf.full(0.0, preg)``), the user
        must pass ``dtype=pl.DT_FP32`` as a kwarg on the VF call.

        Supports:
            reg = vf.add(a, b, preg)              → RegTensor reg; vf.add(reg, a, b, preg)
            reg_lo, reg_hi = vf.mull(a, b, preg)  → RegTensor reg_lo, reg_hi; vf.mull(reg_lo, reg_hi, a, b, pred)
        """
        dst_count = self._get_vf_op_dst_count(vf_op_name)

        call = stmt.value  # ast.Call

        # Determine dst variable names.
        # Some ops (like load_align with DINTLV mode) can produce 2 outputs even
        # though their static dst_count is 1.  When the LHS is a tuple and
        # dst_count == 1, promote to 2-dst form so the backend receives the
        # correct 4-arg call (dst0, dst1, ptr, offset).
        actual_dst_count = dst_count
        if dst_count == 1 and isinstance(target, ast.Tuple) and len(target.elts) == 2:
            actual_dst_count = 2

        if actual_dst_count == 1:
            if not isinstance(target, ast.Name):
                raise InvalidType(
                    f"vf.{vf_op_name} assignment target must be a variable name, got {ast.unparse(target)}",
                    span=span,
                )
            dst_names = [target.id]
            dst_targets = [target]
        elif actual_dst_count == 2:
            if not isinstance(target, ast.Tuple) or len(target.elts) != 2:
                raise NotSupported(
                    f"vf.{vf_op_name} produces 2 outputs, use tuple unpacking: reg0, reg1 = vf.{vf_op_name}(...)",
                    span=span,
                )
            for elt in target.elts:
                if not isinstance(elt, ast.Name):
                    raise NotSupported(
                        f"Tuple unpacking target must be variable names, got {ast.unparse(elt)}",
                        span=span,
                    )
            dst_names = [target.elts[0].id, target.elts[1].id]
            dst_targets = [target.elts[0], target.elts[1]]

        # Parse kwargs once; dtype is taken from here (explicit) or inferred
        # from source args below.
        parsed_kwargs = self.parse_op_kwargs(call)
        dtype_val = parsed_kwargs.pop("dtype", None)

        # Detect MaskReg data source — reused both for the type-changing-op
        # dtype exemption and for dst-kind inference below.
        # Only the first positional arg is checked: it is always a data source,
        # never the predicate mask (prego), which is always the LAST positional
        # arg. Checking all args would mistake preg for a MaskReg data source
        # and wrongly declare the dst as MaskReg (e.g. vf.or_(reg_a, reg_b, preg)).
        first_arg = call.args[0] if call.args else None
        is_mask_src = (
            isinstance(first_arg, ast.Name)
            and self.scope_manager.is_mask_reg_var(first_arg.id)
        )

        # For type-changing ops (vf.astype), dtype kwarg is mandatory because
        # the dst type differs from src type.  Unified ops (pack, unpack) only
        # change type for the RegTensor variant; the MaskReg variant preserves
        # the source type, so dtype= is not required when sources are MaskReg.
        if dtype_val is None and vf_op_name in self._TYPE_CHANGING_OPS and not is_mask_src:
            raise InvalidType(
                f"vf.{vf_op_name} requires explicit dtype kwarg because dst type differs from src. "
                f"Example: vf.{vf_op_name}(src, mask, dtype=pl.DT_FP16)",
                span=span,
                parser_retry=True,
            )

        # Step 1: Parse source args FIRST, before any dst manipulation.
        # This is critical for self-referential ops (e.g. max0 = vf.max(max0, src, preg))
        # where the src must resolve to the OLD value, not a newly created register.
        # Also use this pass to infer dtype if not already determined.
        parsed_src_args = [self.parse_expression(src_arg) for src_arg in call.args]

        src_dtype = self._infer_src_dtype(parsed_src_args)

        if dtype_val is None:
            dtype_val = src_dtype

        if dtype_val is None:
            raise InvalidType(
                f"Cannot infer dtype for vf.{vf_op_name} assignment. "
                f"Pass dtype=pl.DT_FP32 (or appropriate type) as a kwarg.",
                span=span,
                parser_retry=True,
            )

        # A scalar operand is checked against the dtype it is actually encoded in. That is dtype_val
        # -- the dst dtype -- except for a type-changing op, where dtype_val names the dst and the
        # operand still lives in the source domain (vf.muls_cast scales an fp32 src, then casts).
        operand_dtype = src_dtype if vf_op_name in self._TYPE_CHANGING_OPS else dtype_val
        self._check_vf_scalar_operands(vf_op_name, parsed_src_args, operand_dtype, span)

        # Re-inject dtype into kwargs iff the backend op carries it as an IR
        # attribute; otherwise it has already been stripped above.
        ir_op_name = f"vf.{vf_op_name}"
        if ir.is_op_registered(ir_op_name) and ir.get_op(ir_op_name).has_attr("dtype"):
            parsed_kwargs["dtype"] = dtype_val

        # Step 2: Declare dst variables (first definition only). For existing
        # variables, reuse them directly — do NOT create a new register, as that
        # would lose the accumulated value in self-referential ops.
        # Ops in _VF_MASK_DST_OPS produce MaskReg dst(s); declare them via
        # vf.mask_reg. Unified ops (move, interleave, etc.) infer the dst kind
        # from source operands: if any source is a known MaskReg variable, the
        # dst is declared as MaskReg.
        is_mask_dst = self._is_vf_mask_dst_op(vf_op_name)
        if not is_mask_dst and vf_op_name in self._VF_UNIFIED_OPS:
            is_mask_dst = is_mask_src
        decl_op = "vf.mask_reg" if is_mask_dst else "vf.reg_tensor"
        dst_vars = []
        for name, _ in zip(dst_names, dst_targets):
            existing = self.scope_manager.lookup_var_bounded(name)
            if existing is None:
                self.current_target_name = name
                decl_call = ir.create_op_call(decl_op, [], {"dtype": dtype_val}, span)
                var = self.builder.let(name, decl_call, span=span)
                self.scope_manager.define_var(name, var, span=span)
                if is_mask_dst:
                    self.scope_manager.register_mask_reg_var(name)
                dst_vars.append(var)
            else:
                dst_vars.append(existing)

        # Step 3: Construct the VF op call with dst vars + parsed src args.
        # vf.shift_left / vf.shift_right are unified ops: the backend codegen
        # picks the per-lane (vshl/vshr) vs uniform-scalar (vshls/vshrs) form
        # from the shift-amount argument type (see EmitVFShiftLeft/Right), so no
        # special routing is needed here.
        all_args = dst_vars + parsed_src_args
        call_expr = ir.create_op_call(ir_op_name, all_args, parsed_kwargs, span)
        self.builder.emit(ir.EvalStmt(call_expr, span))

    def _parse_tuple_unpacking(self, target: ast.Tuple, stmt: ast.Assign) -> None:
        span = self.span_tracker.get_span(stmt)
        value_expr = self.parse_expression(stmt.value, nested=False)
        if not isinstance(value_expr, ir.Expr) or not isinstance(value_expr.type, ir.TupleType):
            raise InvalidType(
                f"Cannot unpack non-tuple value: {ast.unparse(stmt.value)}",
                span=span,
                parser_retry=True,
            )
        expected = len(value_expr.type.types)
        actual = len(target.elts)
        if actual != expected:
            raise InvalidShape(
                f"Cannot unpack tuple with {expected} items into {actual} targets",
                span=span,
                parser_retry=True,
            )
        tuple_var = self.builder.let(f"_tuple_tmp_{self._tuple_idx_counter}", value_expr, span=span)
        self._mark_make_tuple_anchor(value_expr)
        self._tuple_idx_counter += 1
        for i, elt in enumerate(target.elts):
            if not isinstance(elt, ast.Name):
                raise InvalidType(
                    f"Tuple unpacking target must be a variable name, got {ast.unparse(elt)}",
                    span=self.span_tracker.get_span(elt),
                    hint="Use simple variable names in tuple unpacking: a, b, c = func()",
                )
            if elt.id == "_":
                continue
            item_expr = ir.GetItemExpr(
                tuple_var,
                ir.ConstInt(i, DataType.INT64, ir.Span.unknown()),
                span,
            )
            var = self.builder.let(elt.id, item_expr, span=span)
            self.scope_manager.define_var(elt.id, var, span=span)
            const_item = value_expr.elements[i] if isinstance(value_expr, ir.MakeTuple) else item_expr
            self._transfer_tile_sync_metadata(var, const_item)
            self._update_var_envs(var, const_item)

    def _parse_name_assignment(self, var_name: str, stmt: ast.Assign) -> None:
        span = self.span_tracker.get_span(stmt)
        self.current_target_name = var_name
        value_expr = self.parse_expression(stmt.value, nested=False)
        if value_expr is None:
            raise InvalidOperation(
                f"Cannot assign void function result to '{var_name}'",
                span=span,
                hint="Functions used as expressions must return a value",
                parser_retry=True,
            )
        if isinstance(value_expr, ir.Expr):
            var = self.builder.let(var_name, value_expr, span=span)
            self._mark_make_tuple_anchor(value_expr)
            self._transfer_tile_sync_metadata(var, value_expr)
            self.scope_manager.define_var(var_name, var, span=span)
            # Track MaskReg-producing VF ops (create_mask, update_mask, etc.)
            # so unified ops can infer dst register kind from source operands.
            vf_op = self._is_vf_op_call(stmt.value)
            if vf_op is not None and vf_op in self._VF_MASK_PRODUCING_OPS:
                self.scope_manager.register_mask_reg_var(var_name)
            if self._auto_mutex:
                self._emit_auto_mutex_unlocks()
            self._update_var_envs(var, value_expr)
        else:
            self.scope_manager.define_var(var_name, value_expr, span=span)

    def _is_array_field(self, field_type: ir.Type) -> bool:
        """True when *field_type* is a 1-D array field (plain nested TupleType)."""
        return (
            isinstance(field_type, ir.TupleType)
            and self.classify_tuple_type(field_type).kind == TupleTypeKind.TUPLE
        )

    def _is_struct_value(self, expr) -> bool:
        """True when *expr* is a mutable ``pl.struct`` / struct_array slot."""
        return (
            isinstance(getattr(expr, "type", None), ir.TupleType)
            and self.classify_tuple_type(expr.type).kind == TupleTypeKind.STRUCT
        )

    def _parse_struct_field_assignment(self, target: ast.Attribute, stmt: ast.Assign, span: ir.Span) -> None:
        """Lower ``base.field = rhs`` to ``EvalStmt(struct.set(base, rhs, field=...))``.

        The field name (not an index) is carried on the call, mirroring struct.create's
        ``fields``; codegen emits ``base.field = rhs;``. This keeps the write off the
        AssignStmt left-value, which is restricted to plain Vars.
        """
        base = self.parse_expression(target.value)
        field_name = target.attr
        if not self._is_struct_value(base):
            raise InvalidOperation(
                f"Cannot assign field '{field_name}': the target is not a struct",
                span=span,
                hint="Field assignment is only supported on pl.struct() or pl.struct_array() elements, "
                "e.g. s.field = v or arr[i].field = v.",
                parser_retry=True,
            )
        fields = self.named_fields(base)
        if field_name not in fields:
            raise InvalidVal(
                f"Struct has no field '{field_name}'",
                span=span,
                parser_retry=True,
            )
        value_expr = self.parse_expression(stmt.value, nested=False)
        if not isinstance(value_expr, ir.Expr):
            raise InvalidVal(
                f"Right-hand side of '{ast.unparse(target)}' must be an IR expression",
                span=span,
                parser_retry=True,
            )
        field_idx = fields.index(field_name)
        field_type = base.type.types[field_idx]
        self._check_scalar_or_array_field(field_name, value_expr, span)
        if not ir.structural_equal(field_type, value_expr.type, enable_auto_mapping=False):
            raise InvalidType(
                f"Struct field '{field_name}' expects type {field_type}, got {value_expr.type}",
                span=span,
                hint="Assign a value whose type matches the field.",
            )
        if self._is_array_field(field_type):
            for j, elem in enumerate(value_expr.elements):
                idx_const = ir.ConstInt(j, DataType.INT64, span)
                call = ir.create_op_call(
                    "struct.set", [base, idx_const, elem],
                    {"field": field_name}, span,
                )
                self.builder.emit(ir.EvalStmt(call, span))
            return
        call = ir.create_op_call("struct.set", [base, value_expr], {"field": field_name}, span)
        self.builder.emit(ir.EvalStmt(call, span))

    def _parse_subscript_assignment(self, target: ast.Subscript, stmt: ast.Assign, span: ir.Span) -> None:
        value_expr = self.parse_expression(stmt.value, nested=False)
        if not isinstance(value_expr, ir.Expr):
            raise InvalidType(
                "Right-hand side of subscript assignment must be an IR expression",
                span=span,
                parser_retry=True,
            )
        # Struct (or struct_array element) array-field element write:
        # base.field[idx] = val, e.g. s.arr[i] = v or a[i].b[j] = v.
        if isinstance(target.value, ast.Attribute):
            base = self.parse_expression(target.value.value)
            field_name = target.value.attr
            fields = self.named_fields(base)
            if fields:
                if not self._is_struct_value(base):
                    raise InvalidOperation(
                        f"Cannot assign field '{field_name}': the target is not a struct",
                        span=span,
                        hint="Field assignment is only supported on pl.struct() or pl.struct_array() elements, "
                        "e.g. s.field = v or arr[i].field = v.",
                        parser_retry=True,
                    )
                if field_name not in fields:
                    raise InvalidVal(
                        f"Struct has no field '{field_name}'",
                        span=span,
                        parser_retry=True,
                    )
                field_type = base.type.types[fields.index(field_name)]
                # An array field is a plain nested TupleType. A nested named tuple or
                # struct has a TupleTypeInfo entry and is not subscript-writable.
                if not self._is_array_field(field_type):
                    raise InvalidOperation(
                        f"Cannot subscript struct field '{field_name}': field is not an array",
                        span=span,
                        hint="Only array fields (e.g. field=[0, 0, 0, 0]) support element assignment",
                        parser_retry=True,
                    )
                # Single scalar index only; mirror the read path's tuple-subscript rules.
                if isinstance(target.slice, ast.Tuple):
                    raise InvalidShape(
                        "Multi-dimensional subscript is not supported for struct array fields",
                        span=span,
                        hint="Use a scalar index like s.arr[0]",
                    )
                index_expr = self.parse_expression(target.slice)
                if isinstance(index_expr, ir.ConstInt) and not (
                    0 <= index_expr.value < len(field_type.types)
                ):
                    raise OutOfRange(
                        f"Array field '{field_name}' index {index_expr.value} out of bounds "
                        f"for {len(field_type.types)} elements",
                        span=span,
                    )
                elem_type = field_type.types[0]
                if not ir.structural_equal(elem_type, value_expr.type, enable_auto_mapping=False):
                    raise InvalidType(
                        f"Struct field '{field_name}' expects type {elem_type}, got {value_expr.type}",
                        span=span,
                        hint="Assign a value whose type matches the field.",
                    )
                call = ir.create_op_call(
                    "struct.set", [base, index_expr, value_expr], {"field": field_name}, span,
                )
                self.builder.emit(ir.EvalStmt(call, span))
                return

        container_expr = self.parse_expression(target.value)
        container_type = container_expr.type
        if not isinstance(container_type, (ir.TileType, ir.TensorType)):
            raise InvalidType(
                f"Subscript assignment requires Tile or Tensor, got {type(container_type).__name__}",
                span=span,
                hint="Only Tile and Tensor support element assignment via A[i] = v",
                parser_retry=True,
            )
        index_expr = self._parse_scalar_subscript_index(container_expr, target.slice, span)
        from pypto_pro.ir.op.block_ops import _ir_setval
        setval_call = _ir_setval(container_expr, index_expr, value_expr, span=span)
        mutex_locked = False
        if self._auto_mutex:
            from ._op_pipeline import get_op_pipe
            pipe = get_op_pipe("setval")
            mutex_locked = self._emit_mutex_for_tile(container_expr, pipe, span, is_lock=True)
        self.builder.emit(ir.EvalStmt(setval_call, span))
        if mutex_locked:
            self._emit_mutex_for_tile(container_expr, pipe, span, is_lock=False)
