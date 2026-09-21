# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Control-flow parsing helpers for ASTParser."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Callable

from pypto.pypto_impl import ir
from pypto.pypto_impl.ir import DataType
from pypto_pro.ir._limits import INT64_MAX, INT64_MIN, from_storage_int

from ..._errors import InvalidArgument, InvalidOperation, InvalidType, InvalidVal, NotSupported, OutOfRange
from ._expr_evaluator import ExprEvaluator
from ._scope_manager import (
    ConstantState,
    ControlFlowInfo,
    JumpInfo,
    JumpKind,
    LocalScope,
    LoopVarState,
    PhiState,
)
from .diagnostics import (
    check_in_range,
)

_UINT16_MAX = 65535


def _is_bare_return(stmt: ast.Return) -> bool:
    """Return whether *stmt* is a bare ``return`` or ``return None``."""
    return stmt.value is None or (isinstance(stmt.value, ast.Constant) and stmt.value.value is None)


def _body_without_docstring(func_def: ast.FunctionDef) -> list[ast.stmt]:
    body = list(func_def.body)
    if body and isinstance(body[0], ast.Expr):
        value = body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return body[1:]
    return body


def _target_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*(_target_names(elt) for elt in target.elts)) if target.elts else set()
    return set()


def collect_written_vars(
    statements: list[ast.stmt], *, ignore_assignment: Callable[[ast.Assign], bool] | None = None
) -> set[str]:
    """Collect names that a control-flow region can rebind.

    A Python-visible for target is an assignment too: an enclosing branch or
    loop must merge it when that binding already exists outside the inner loop.
    """
    writes: set[str] = set()

    def visit(stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.Assign):
            if ignore_assignment is not None and ignore_assignment(stmt):
                return
            for target in stmt.targets:
                writes.update(_target_names(target))
        elif isinstance(stmt, ast.AugAssign):
            writes.update(_target_names(stmt.target))
        elif isinstance(stmt, ast.AnnAssign):
            writes.update(_target_names(stmt.target))
        elif isinstance(stmt, ast.If):
            for child in [*stmt.body, *stmt.orelse]:
                visit(child)
        elif isinstance(stmt, (ast.For, ast.While)):
            if isinstance(stmt, ast.For):
                writes.update(_target_names(stmt.target))
            for child in stmt.body:
                visit(child)
            for child in stmt.orelse:
                visit(child)
        elif isinstance(stmt, ast.With):
            for child in stmt.body:
                visit(child)

    for statement in statements:
        visit(statement)
    return writes


@dataclass
class _LoopMergeSlot:
    """The minimum state needed for one loop-carried source binding."""

    name: str
    init_value: ir.Expr
    iter_var: ir.Var
    state: LoopVarState
    mutex_iter_vars: tuple[ir.Var, ...] = ()


def validate_single_tail_return(func_def: ast.FunctionDef, context: str) -> tuple[ast.Return, str, str] | None:
    """Require at most one return, and only as the top-level final statement."""
    returns = [node for node in ast.walk(func_def) if isinstance(node, ast.Return)]
    if not returns:
        return None

    if len(returns) > 1:
        return (
            returns[1],
            f"{context} only supports a single return statement.",
            "Keep one top-level return at the end of the function. Only @pl.jit supports early return.",
        )

    body = _body_without_docstring(func_def)
    tail_stmt = body[-1] if body else None
    if returns[0] is not tail_stmt:
        return (
            returns[0],
            f"{context} only supports return as the final top-level statement.",
            "Move the return to the end of the function body, or use @pl.jit when early return is required.",
        )

    return None


class ControlFlowParserMixin:
    """Mixin containing loop, branch, with-scope, and return parsing."""

    _VALID_ITERATORS = {"range"}
    _ITERATOR_ERROR = "For loop must use pl.range()"
    _ITERATOR_HINT = "Use pl.range() as the iterator"

    def _is_empty_control_flow_value(self, value: ir.Expr) -> bool:
        return isinstance(value, ir.Var) and value.name in self._empty_control_flow_values

    def _mark_empty_control_flow_value(self, value: ir.Var) -> None:
        self._empty_control_flow_values.add(value.name)

    def collect_written_vars(self, statements: list[ast.stmt]) -> set[str]:
        """Collect source rebindings, excluding mutable VF register writes."""

        def is_vf_register_write(stmt: ast.Assign) -> bool:
            op_name = self._is_vf_op_call(stmt.value)
            if op_name is None:
                return False
            dst_count = self._get_vf_op_dst_count(op_name)
            return dst_count is not None and dst_count > 0

        return collect_written_vars(statements, ignore_assignment=is_vf_register_write)

    def _merge_tile_mutex_meta_pair(
        self, first_value: ir.Expr, second_value: ir.Expr, span: ir.Span
    ) -> tuple[tuple[ir.Expr, ...], tuple[ir.Expr, ...], list[Any]] | None:
        """Align two runtime mutex-id lists and union their static candidates."""
        if not self._auto_mutex:
            return None
        first_meta = self._tile_mutex_meta.get(first_value)
        second_meta = self._tile_mutex_meta.get(second_value)
        if first_meta is None and second_meta is None:
            return None
        if first_meta is None or second_meta is None:
            present_meta = first_meta if first_meta is not None else second_meta
            if not present_meta[1]:
                return None
            raise InvalidOperation(
                "Cannot merge Tile values when only one input carries mutex metadata",
                span=span,
                hint="Ensure all values come from the same auto-mutex tile-group flow",
                parser_retry=True,
            )
        first_mutex_ids, first_candidates = first_meta
        second_mutex_ids, second_candidates = second_meta
        if len(first_mutex_ids) != len(second_mutex_ids):
            if not first_candidates and len(first_mutex_ids) == 1:
                first_mutex_ids = tuple(first_mutex_ids) * len(second_mutex_ids)
            elif not second_candidates and len(second_mutex_ids) == 1:
                second_mutex_ids = tuple(second_mutex_ids) * len(first_mutex_ids)
            else:
                raise InvalidOperation(
                    "cannot merge tile mutex metadata with different ID counts: "
                    f"{len(first_mutex_ids)} and {len(second_mutex_ids)}",
                    span=span,
                    parser_retry=True,
                )
        candidates = list(
            dict.fromkeys(list(first_candidates or ()) + list(second_candidates or ()))
        )
        return tuple(first_mutex_ids), tuple(second_mutex_ids), candidates

    def _merge_control_flow_mutex_ids(
        self, values: list[ir.Expr], span: ir.Span
    ) -> tuple[list[tuple[ir.Expr, ...]], list[Any]] | None:
        """Align one explicit Tile slot's mutex ids across its CFG inputs."""
        if not self._auto_mutex or not values:
            return None
        if len(values) == 1:
            mutex_meta = self._tile_mutex_meta.get(values[0])
            if mutex_meta is None:
                return None
            mutex_ids, candidates = mutex_meta
            return [tuple(mutex_ids)], list(candidates or ())
        if len(values) == 2:
            merged = self._merge_tile_mutex_meta_pair(values[0], values[1], span)
            if merged is None:
                return None
            first_mutex_ids, second_mutex_ids, candidates = merged
            return [tuple(first_mutex_ids), tuple(second_mutex_ids)], candidates

        exemplar = next(
            (
                value
                for value in values
                if (meta := self._tile_mutex_meta.get(value)) is not None and meta[1]
            ),
            None,
        )
        if exemplar is None:
            return None

        tile_mutex_id_outputs: list[tuple[ir.Expr, ...]] = []
        candidates: list[Any] = []
        for value in values:
            merged = self._merge_tile_mutex_meta_pair(value, exemplar, span)
            if merged is None:
                raise InvalidOperation("Cannot merge Tile values without mutex metadata", span=span, parser_retry=True)
            value_mutex_ids, _, value_candidates = merged
            tile_mutex_id_outputs.append(tuple(value_mutex_ids))
            candidates.extend(value_candidates)
        return tile_mutex_id_outputs, list(dict.fromkeys(candidates))

    def _create_mutex_id_vars(
        self, name: str, count: int, span: ir.Span
    ) -> tuple[ir.Var, ...]:
        return tuple(
            self.builder.var(
                f"{name}__mutexid" if index == 0 else f"{name}__mutexid_{index}",
                ir.ScalarType(DataType.INT64),
                span,
            )
            for index in range(count)
        )

    def _parse_statement_list(self, statements: list[ast.stmt]) -> JumpKind | None:
        """Parse statements until the current scope has terminated."""
        local_scope = self.scope_manager.current_scope
        for statement in statements:
            self.parse_statement(statement)
            if local_scope.jump_kind is not None:
                break
        return local_scope.jump_kind

    def _parse_control_flow_region(
        self, statements: list[ast.stmt], scope_type: str
    ) -> JumpKind:
        """Parse one if/loop block, add its default jump, and dispatch it."""
        jump_kind = self._parse_statement_list(statements)
        local_scope = self.scope_manager.current_scope
        if jump_kind is None:
            jump_kind = self._default_jump_kind(scope_type)
            if jump_kind is None:
                raise NotSupported(f"Unsupported control-flow scope type: {scope_type}")
            local_scope.set_jump(jump_kind)

        loop_info = self.scope_manager.loop_info
        if scope_type in ("for", "while") and loop_info is not None and loop_info.flatten:
            return jump_kind

        self.dispatch_jump(jump_kind)
        return jump_kind

    @staticmethod
    def _default_jump_kind(scope_type: str) -> JumpKind | None:
        if scope_type == "if":
            return JumpKind.YIELD
        if scope_type in ("for", "while"):
            return JumpKind.CONTINUE
        return None

    def _current_span(self) -> ir.Span:
        return self.span_tracker.get_span(self._current_node)

    def _as_control_flow_value(self, name: str, value: Any, span: ir.Span) -> ir.Expr:
        """Convert the current parser binding into one legal jump output."""
        if (
            isinstance(value, ir.Expr)
            and not isinstance(value.type, ir.UnknownType)
            and not self._contains_tile_group_type(value.type)
        ):
            return value
        return self.builder.var(name, ir.NoneType.get(), span)

    def dispatch_jump(self, jump_kind: JumpKind) -> JumpInfo | None:
        """Create an empty terminator and record its named merge outputs."""
        if jump_kind is JumpKind.RETURN:
            return None

        info = self.scope_manager.if_info if jump_kind is JumpKind.YIELD else self.scope_manager.loop_info
        if info is None:
            raise InvalidVal(f"{jump_kind.value} is missing its control-flow owner")

        span = self._current_span()
        outputs = [
            self._as_control_flow_value(
                name, self.lookup_expr_by_name(name), span
            )
            for name in info.merge_names
        ]

        if jump_kind is JumpKind.YIELD:
            jump_op = ir.YieldStmt([], span)
        elif jump_kind is JumpKind.BREAK:
            jump_op = ir.BreakStmt([], span)
        elif jump_kind is JumpKind.CONTINUE:
            jump_op = ir.ContinueStmt([], span)
        else:
            raise NotSupported(f"Unsupported jump kind: {jump_kind}")
        self.builder.emit(jump_op)
        jump_info = JumpInfo(jump_op, tuple(outputs))
        info.jumps.append(jump_info)
        return jump_info

    def _infer_phi_states(self, info: ControlFlowInfo) -> list[PhiState]:
        if not info.jumps:
            return []
        output_count = len(info.jumps[0].outputs)
        states = [PhiState(type_equal=self.tuple_type_registry.types_equal) for _ in range(output_count)]
        for jump in info.jumps:
            if len(jump.outputs) != output_count:
                raise InvalidVal("control-flow jump output counts must match")
            for state, value in zip(states, jump.outputs, strict=True):
                if self._is_empty_control_flow_value(value):
                    continue
                state.propagate(value)
        return states

    def _update_phi_constant(self, result: ir.Var, state: PhiState) -> None:
        """Record PhiState's complete constant value under the result SSA name."""
        if state.constant_state is not ConstantState.MAY_BE_CONSTANT:
            return
        assert state.constant_value is not None
        self.const_env[result.name] = state.constant_value

    def _enter_region(self, body: ir.SeqStmts) -> None:
        self.builder.builder.set_insert_point(ir.InsertPoint(body))

    def _leave_region(self) -> None:
        self.builder.builder.clear_insert_point()

    def _parse_if_region(
        self,
        statements: list[ast.stmt],
        span: ir.Span,
    ) -> tuple[ir.SeqStmts, LocalScope]:
        """Parse one branch/body into an off-tree SeqStmts."""
        body = ir.SeqStmts(span)
        self._enter_region(body)
        self.scope_manager.enter_scope("if")
        local_scope = self.scope_manager.current_scope
        try:
            self._parse_control_flow_region(statements, "if")
        finally:
            self.scope_manager.exit_scope(leak_vars=False)
            self._leave_region()
        return body, local_scope

    def _flatten_while_region(self, statements: list[ast.stmt]) -> None:
        """Parse a flattened while and consume its loop-local jump."""
        self.scope_manager.enter_scope("while")
        previous_while = self.in_while_loop
        self.in_while_loop = True
        with self.scope_manager.change_loop_info(ControlFlowInfo((), flatten=True)):
            self._parse_control_flow_region(statements, "while")
        self.scope_manager.exit_scope(leak_vars=True)
        self.in_while_loop = previous_while

    @staticmethod
    def _can_flatten_while(statements: list[ast.stmt]) -> bool:
        """Return whether the lowered loop reaches an unconditional top-level break."""
        break_index = next(
            (index for index, statement in enumerate(statements) if isinstance(statement, ast.Break)),
            None,
        )
        if break_index is None:
            return False

        def has_current_loop_jump(node: ast.AST) -> bool:
            if isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
                return False
            if isinstance(node, (ast.Break, ast.Continue)):
                return True
            return any(has_current_loop_jump(child) for child in ast.iter_child_nodes(node))

        return not any(has_current_loop_jump(statement) for statement in statements[:break_index])

    def _validate_loop_orelse(self, stmt: ast.For | ast.While) -> None:
        if stmt.orelse:
            kind = "for" if isinstance(stmt, ast.For) else "while"
            raise NotSupported(
                f"'{kind}-else' is not supported",
                span=self.span_tracker.get_span(stmt.orelse[0]),
            )

    @staticmethod
    def _get_with_context_attr(stmt: ast.With) -> str | None:
        """Return the context manager attribute name for supported with calls."""
        context_expr = stmt.items[0].context_expr
        if not isinstance(context_expr, ast.Call):
            return None
        func = context_expr.func
        return func.attr if isinstance(func, ast.Attribute) else None

    @staticmethod
    def _describe_with_context(stmt: ast.With) -> str:
        """Render the offending context manager(s) for a diagnostic message.

        Returns a source-like description (e.g. ``pl.section_vf()``) so the error
        names exactly what was written, instead of only listing what is allowed.
        """
        try:
            return ", ".join(ast.unparse(item.context_expr) for item in stmt.items)
        except Exception:
            return "<unknown>"

    @staticmethod
    def _as_index_expr(value: int | ir.Expr, span: ir.Span) -> ir.Expr:
        return value if isinstance(value, ir.Expr) else ir.ConstInt(value, DataType.INT64, span)

    def _select_loop_merge_inputs(
        self, writes: set[str], span: ir.Span
    ) -> tuple[tuple[str, ir.Expr], ...]:
        """Keep explicit loop writes whose values exist before the loop."""
        merge_inputs = []
        for name in sorted(writes):
            value = self.lookup_expr_by_name(name)
            if value is None:
                continue
            merge_inputs.append((name, self._as_control_flow_value(name, value, span)))
        return tuple(merge_inputs)

    def _create_loop_slots(
        self, merge_inputs: tuple[tuple[str, ir.Expr], ...], span: ir.Span
    ) -> list[_LoopMergeSlot]:
        """Create main-value carries for the selected explicit merge names."""
        slots: list[_LoopMergeSlot] = []
        for name, init_value in merge_inputs:
            state = LoopVarState(
                PhiState(
                    constant_state=ConstantState.NONCONSTANT,
                    type_equal=self.tuple_type_registry.types_equal,
                ),
                PhiState(type_equal=self.tuple_type_registry.types_equal),
            )
            if self._is_empty_control_flow_value(init_value):
                iter_var = self.builder.var(name, init_value.type, span)
                self._mark_empty_control_flow_value(iter_var)
                self._transfer_tile_sync_metadata(iter_var, init_value)
            else:
                state.body_phi.propagate(init_value)
                if state.body_phi.ty is None:
                    raise InvalidVal(f"Loop phi state for '{name}' has no IR type")
                iter_var = self.builder.var(name, state.body_phi.ty, span)
            slots.append(
                _LoopMergeSlot(
                    name=name,
                    init_value=init_value,
                    iter_var=iter_var,
                    state=state,
                )
            )
        return slots

    def _prepare_loop_mutex_iter_vars(
        self, slots: list[_LoopMergeSlot], span: ir.Span
    ) -> None:
        """Expose provisional mutex iter vars while parsing the loop body."""
        if not self._auto_mutex:
            return
        for slot in slots:
            if not isinstance(slot.init_value.type, ir.TileType):
                continue
            mutex_meta = self._tile_mutex_meta.get(slot.init_value)
            if mutex_meta is None:
                continue
            mutex_ids, candidates = mutex_meta
            slot.mutex_iter_vars = self._create_mutex_id_vars(
                slot.name, len(mutex_ids), span
            )
            self._tile_mutex_meta[slot.iter_var] = (
                slot.mutex_iter_vars,
                list(candidates or ()),
            )

    def _parse_loop_region(
        self,
        statements: list[ast.stmt],
        scope_type: str,
        span: ir.Span,
        slots: list[_LoopMergeSlot],
        loop_binding: tuple[str, ir.Var] | None = None,
    ) -> ir.SeqStmts:
        body = ir.SeqStmts(span)
        self._enter_region(body)
        self.scope_manager.enter_scope(scope_type)
        previous_for = self.in_for_loop
        previous_while = self.in_while_loop
        self.in_for_loop = scope_type == "for" or previous_for
        self.in_while_loop = scope_type == "while" or previous_while
        try:
            if loop_binding is not None:
                self.scope_manager.define_var(loop_binding[0], loop_binding[1], allow_redef=True, span=span)
            for slot in slots:
                self.scope_manager.define_var(slot.name, slot.iter_var, allow_redef=True, span=span)
            self._parse_control_flow_region(statements, scope_type)
        finally:
            self.scope_manager.exit_scope(leak_vars=False)
            self.in_for_loop = previous_for
            self.in_while_loop = previous_while
            self._leave_region()
        return body

    def _finalize_loop_slots(
        self,
        slots: list[_LoopMergeSlot],
        info: ControlFlowInfo,
        span: ir.Span,
        *,
        is_for_loop: bool,
    ) -> tuple[
        list[ir.IterArg],
        list[ir.Var],
        list[tuple[str, ir.Var, PhiState]],
    ]:
        finalized: list[tuple[_LoopMergeSlot, ir.Var, ir.Var]] = []
        for jump in info.jumps:
            if len(jump.outputs) != len(slots):
                raise InvalidVal("loop jump output count must match the collected merge slots")
            is_continue = isinstance(jump.jump_op, ir.ContinueStmt)
            is_break = isinstance(jump.jump_op, ir.BreakStmt)
            if not is_continue and not is_break:
                raise InvalidVal("loop ControlFlowInfo can only contain break or continue jumps")
            for slot, value in zip(slots, jump.outputs, strict=True):
                if self._is_empty_control_flow_value(value):
                    continue
                if is_for_loop or is_continue:
                    slot.state.body_phi.propagate(value, fail_eagerly=True)
                if is_for_loop or is_break:
                    slot.state.result_phi.propagate(value)

        for slot in slots:
            state = slot.state
            merged_type = state.body_phi.ty or slot.iter_var.type
            final_iter_var = slot.iter_var
            if not ir.structural_equal(slot.iter_var.type, merged_type, enable_auto_mapping=False):
                final_iter_var = ir.Var(slot.iter_var.name, merged_type, slot.iter_var.span)
                slot.iter_var = final_iter_var
            if state.result_phi.ty is None:
                return_var = self.builder.var(slot.name, ir.NoneType.get(), span)
                self._mark_empty_control_flow_value(return_var)
            else:
                return_var = self.builder.var(slot.name, state.result_phi.ty, span)
            finalized.append((slot, final_iter_var, return_var))

        iter_args: list[ir.IterArg] = []
        return_vars: list[ir.Var] = []
        merged_bindings: list[tuple[str, ir.Var, PhiState]] = []
        for slot, iter_var, return_var in finalized:
            iter_args.append(self.builder.builder.create_iter_arg(iter_var, slot.init_value))
            return_vars.append(return_var)
            merged_bindings.append((slot.name, return_var, slot.state.result_phi))

        # Mutex ids are framework-generated companion values. Infer and append
        # them only after all source-visible main slots have completed Phi inference.
        mutex_outputs_by_jump: list[list[ir.Expr]] = [[] for _ in info.jumps]
        for index, (slot, iter_var, return_var) in enumerate(finalized):
            if not isinstance(iter_var.type, ir.TileType) and not isinstance(return_var.type, ir.TileType):
                continue

            values = [slot.init_value]
            values.extend(jump.outputs[index] for jump in info.jumps)
            mutex_merge = self._merge_control_flow_mutex_ids(values, span)
            if mutex_merge is None:
                continue
            mutex_ids_by_source, candidates = mutex_merge
            init_mutex_ids, *mutex_ids_by_jump = mutex_ids_by_source
            if not slot.mutex_iter_vars:
                slot.mutex_iter_vars = self._create_mutex_id_vars(
                    slot.name, len(init_mutex_ids), span
                )
            mutex_return_vars = self._create_mutex_id_vars(
                slot.name, len(init_mutex_ids), span
            )
            for mutex_iter_var, mutex_init_value in zip(
                slot.mutex_iter_vars, init_mutex_ids, strict=True
            ):
                iter_args.append(
                    self.builder.builder.create_iter_arg(
                        mutex_iter_var, mutex_init_value
                    )
                )
            return_vars.extend(mutex_return_vars)
            for jump_mutex_ids, tile_mutex_ids in zip(
                mutex_outputs_by_jump, mutex_ids_by_jump, strict=True
            ):
                jump_mutex_ids.extend(tile_mutex_ids)

            if isinstance(return_var.type, ir.TileType):
                self._tile_mutex_meta[return_var] = (mutex_return_vars, candidates)

        for jump, jump_mutex_ids in zip(info.jumps, mutex_outputs_by_jump, strict=True):
            if jump.jump_op is None:
                raise InvalidVal("loop jump is missing its terminator")
            self.builder.builder.update_jump_values(jump.jump_op, [*jump.outputs, *jump_mutex_ids])

        return iter_args, return_vars, merged_bindings

    def parse_for_loop(self, stmt: ast.For) -> None:
        """Parse a natural ``for`` directly into an SSA ForStmt."""
        self._validate_loop_orelse(stmt)
        iter_call = self._validate_for_loop_iterator(stmt)
        stmt = self._lower_for_loop(stmt)
        loop_var_name = self._parse_for_loop_target(stmt)
        range_args = self._parse_range_call(iter_call)
        span = self.span_tracker.get_span(stmt)
        writes = self.collect_written_vars(stmt.body)
        merge_inputs = self._select_loop_merge_inputs(writes, span)
        merge_names = tuple(name for name, _ in merge_inputs)
        slots = self._create_loop_slots(merge_inputs, span)
        for slot in slots:
            if not self._is_empty_control_flow_value(slot.init_value):
                slot.state.result_phi.propagate(slot.init_value)
        info = ControlFlowInfo(merge_names)
        self._prepare_loop_mutex_iter_vars(slots, span)
        loop_var = self.builder.var(loop_var_name, ir.ScalarType(DataType.INT64), span)
        with self.scope_manager.change_loop_info(info):
            body = self._parse_loop_region(
                stmt.body, "for", span, slots, (loop_var_name, loop_var)
            )
        iter_args, return_vars, merged_bindings = self._finalize_loop_slots(
            slots, info, span, is_for_loop=True
        )
        for_stmt = self.builder.builder.create_for_stmt(
            loop_var,
            self._as_index_expr(range_args["start"], span),
            self._as_index_expr(range_args["stop"], span),
            self._as_index_expr(range_args["step"], span),
            iter_args,
            body,
            return_vars,
            span,
        )
        self.builder.emit(for_stmt)
        for name, return_var, state in merged_bindings:
            self.scope_manager.define_var(name, return_var, allow_redef=True, span=span)
            self._update_phi_constant(return_var, state)

    def parse_while_loop(self, stmt: ast.While) -> None:
        """Lower a natural while to ``while True`` with an SSA-carrying break guard."""
        self._validate_loop_orelse(stmt)
        span = self.span_tracker.get_span(stmt)
        loop_body = stmt.body
        if not (isinstance(stmt.test, ast.Constant) and stmt.test.value is True):
            break_test = ast.copy_location(ast.UnaryOp(op=ast.Not(), operand=stmt.test), stmt.test)
            break_stmt = ast.copy_location(ast.Break(), stmt)
            break_if = ast.copy_location(
                ast.If(test=break_test, body=[break_stmt], orelse=[]), stmt
            )
            ast.fix_missing_locations(break_if)
            loop_body = [break_if, *stmt.body]

        if self._can_flatten_while(loop_body):
            self._flatten_while_region(loop_body)
            return

        writes = self.collect_written_vars(stmt.body)
        merge_inputs = self._select_loop_merge_inputs(writes, span)
        merge_names = tuple(name for name, _ in merge_inputs)
        slots = self._create_loop_slots(merge_inputs, span)
        info = ControlFlowInfo(merge_names)
        self._prepare_loop_mutex_iter_vars(slots, span)
        with self.scope_manager.change_loop_info(info):
            body = self._parse_loop_region(loop_body, "while", span, slots)
        iter_args, return_vars, merged_bindings = self._finalize_loop_slots(
            slots, info, span, is_for_loop=False
        )
        while_stmt = self.builder.builder.create_while_stmt(
            ir.ConstBool(True, span), iter_args, body, return_vars, span
        )
        self.builder.emit(while_stmt)
        for name, return_var, state in merged_bindings:
            self.scope_manager.define_var(name, return_var, allow_redef=True, span=span)
            self._update_phi_constant(return_var, state)

    def _record_if_const(self, test_node: ast.expr, is_const: bool, value) -> None:
        """Record each ``if`` condition's compile-time constness for the auto-pipeline's branch pruning."""
        if not getattr(self, "collect_if_const", False) or self.inline_call_stack:
            return
        self.if_const_map[(test_node.lineno, test_node.col_offset)] = (is_const, value)

    def parse_if_statement(self, stmt: ast.If) -> None:
        """Parse an IfStmt using only its reachable Yield predecessors."""
        condition = self.parse_expression(stmt.test)
        span = self.span_tracker.get_span(stmt)

        if isinstance(condition, (ir.ConstBool, ir.ConstInt)):
            is_true = condition.value if isinstance(condition, ir.ConstBool) else condition.value != 0
            self._record_if_const(stmt.test, True, bool(is_true))
            self._parse_statement_list(stmt.body if is_true else stmt.orelse)
            return

        self._record_if_const(stmt.test, False, None)
        writes = tuple(sorted(self.collect_written_vars([*stmt.body, *stmt.orelse])))
        info = ControlFlowInfo(writes)
        with self.scope_manager.change_if_info(info):
            then_body, _ = self._parse_if_region(stmt.body, span)

        # If then has no Yield predecessor, emit a result-less structured If
        # and flatten the original else into the parent region.
        if not info.jumps:
            synthetic_else = ir.SeqStmts([ir.YieldStmt([], span)], span)
            self.builder.emit(
                self.builder.builder.create_if_stmt(condition, then_body, synthetic_else, [], span)
            )
            self._parse_statement_list(stmt.orelse)
            return

        with self.scope_manager.change_if_info(info):
            else_body, _ = self._parse_if_region(stmt.orelse, span)

        phi_states = self._infer_phi_states(info)
        if len(phi_states) != len(writes):
            raise InvalidVal("If jump output count must match the collected merge names")

        merged_bindings: list[tuple[str, ir.Var, PhiState]] = []
        return_vars: list[ir.Var] = []
        jump_mutex_id_outputs: list[list[ir.Expr]] = [[] for _ in info.jumps]
        for name, state in zip(writes, phi_states, strict=True):
            if state.ty is None:
                merged_var = self.builder.var(name, ir.NoneType.get(), span)
                self._mark_empty_control_flow_value(merged_var)
            else:
                merged_var = self.builder.var(name, state.ty, span)
            return_vars.append(merged_var)
            merged_bindings.append((name, merged_var, state))

        # Mutex ids are framework-generated companion values. Add them only
        # after all source-visible main slots have completed Phi inference.
        for index, (name, merged_var, state) in enumerate(merged_bindings):
            if not isinstance(state.ty, ir.TileType):
                continue
            values = [jump.outputs[index] for jump in info.jumps]
            mutex_merge = self._merge_control_flow_mutex_ids(values, span)
            if mutex_merge is None:
                continue
            tile_mutex_id_outputs, candidates = mutex_merge
            mutex_vars = self._create_mutex_id_vars(
                name, len(tile_mutex_id_outputs[0]), span
            )
            return_vars.extend(mutex_vars)
            for jump_mutex_ids, tile_mutex_ids in zip(
                jump_mutex_id_outputs, tile_mutex_id_outputs, strict=True
            ):
                jump_mutex_ids.extend(tile_mutex_ids)
            self._tile_mutex_meta[merged_var] = (mutex_vars, candidates)

        for jump, jump_mutex_ids in zip(info.jumps, jump_mutex_id_outputs, strict=True):
            if jump.jump_op is None:
                raise InvalidVal("materialized If yield is missing its terminator")
            self.builder.builder.update_jump_values(jump.jump_op, [*jump.outputs, *jump_mutex_ids])

        self.builder.emit(
            self.builder.builder.create_if_stmt(condition, then_body, else_body, return_vars, span)
        )
        for name, merged_var, state in merged_bindings:
            self.scope_manager.define_var(name, merged_var, allow_redef=True, span=span)
            self._update_phi_constant(merged_var, state)

    def parse_with_statement(self, stmt: ast.With) -> None:
        """Parse with statement for scope contexts.

        Currently supports target-specific Cube and Vector sections.

        Args:
            stmt: With AST node
        """
        # Check that we have exactly one context manager
        if len(stmt.items) != 1:
            raise InvalidArgument(
                f"Only a single context manager is supported in a 'with' statement, "
                f"but got {len(stmt.items)}: 'with {self._describe_with_context(stmt)}:'",
                span=self.span_tracker.get_span(stmt),
                hint="Use 'with pl.section_vector():' or 'with pl.section_cube():'",
            )

        attr = self._get_with_context_attr(stmt)
        span = self.span_tracker.get_span(stmt)

        if attr in ("section_vector", "section_cube") and self.inline_vf_depth:
            raise InvalidOperation(
                f"Section 'pl.{attr}' cannot be nested inside @pl.vector_function",
                span=span,
                hint="Place Cube/Vector sections in the calling kernel and invoke the vector function from there.",
            )

        # Check if this is pl.section_vector() or pl.section_cube()
        if attr == "section_vector":
            self._parse_target_section(stmt.body, ir.SectionKind.Vector, span)
            return
        if attr == "section_cube":
            self._parse_target_section(stmt.body, ir.SectionKind.Cube, span)
            return

        # Unsupported context manager
        raise NotSupported(
            f"Unsupported context manager 'with {self._describe_with_context(stmt)}:'"
            + (f" (pl.{attr}() is not a valid section here)" if attr else ""),
            span=self.span_tracker.get_span(stmt),
            hint=(
                "Only 'with pl.section_vector():' or 'with pl.section_cube():' "
                "are currently supported. VF code must be placed in a @pl.vector_function "
                "decorated function."
            ),
            parser_retry=True,
        )

    def parse_return(self, stmt: ast.Return) -> None:
        """Parse return statement.

        Args:
            stmt: Return AST node
        """
        span = self.span_tracker.get_span(stmt)
        if self.inline_vf_depth != 0:
            raise InvalidOperation(
                "Vector function cannot contain return",
                span=span,
            )

        if _is_bare_return(stmt):
            self.builder.emit(ir.ReturnStmt([], span))
            self.scope_manager.current_scope.set_jump(JumpKind.RETURN)
            return

        if self._current_func_type == ir.FunctionType.SimtCallee:
            return_expr = self.parse_expression(stmt.value)
            if not isinstance(return_expr.type, ir.ScalarType):
                raise InvalidVal(
                    'A helper @pl.vector_function(mode="simt") must return None or one scalar value',
                    span=span,
                    hint="Return Tile/Tensor data through an input parameter; return scalar results directly.",
                    parser_retry=True,
                )
        else:
            if self._void_return_only:
                raise NotSupported(
                    f"{self._void_return_context} only supports bare return or return None; "
                    "returning values is not supported.",
                    span=span,
                    hint=(
                        "Do not write `return <value>`; only use `return` or `return None`. "
                        "Pass output Tensor/Tile/buffer parameters for data results."
                    ),
                )
            return_expr = self.parse_expression(stmt.value)
        self.builder.emit(ir.ReturnStmt([return_expr], span))
        self.scope_manager.current_scope.set_jump(JumpKind.RETURN)

    def parse_break(self, stmt: ast.Break) -> None:
        """Parse break statement.

        Args:
            stmt: Break AST node
        """
        if not self.in_for_loop and not self.in_while_loop:
            raise InvalidOperation(
                "'break' statement outside of a loop",
                span=self.span_tracker.get_span(stmt),
                hint="break can only be used inside a for or while loop",
            )
        self.scope_manager.current_scope.set_jump(JumpKind.BREAK)

    def parse_continue(self, stmt: ast.Continue) -> None:
        """Parse continue statement.

        Args:
            stmt: Continue AST node
        """
        if not self.in_for_loop and not self.in_while_loop:
            raise InvalidOperation(
                "'continue' statement outside of a loop",
                span=self.span_tracker.get_span(stmt),
                hint="continue can only be used inside a for or while loop",
            )
        self.scope_manager.current_scope.set_jump(JumpKind.CONTINUE)

    def _parse_target_section(
        self,
        body: list[ast.stmt],
        kind: ir.SectionKind,
        span,
    ) -> None:
        """Project a Cube/Vector section into the current target Program."""
        if kind != self.target:
            return

        self.matched_target = True
        self._parse_statement_list(body)

    def _validate_for_loop_iterator(self, stmt: ast.For) -> ast.Call:
        """Validate that for loop uses pl.range().

        Returns:
            The call node for pl.range()
        """
        if not isinstance(stmt.iter, ast.Call):
            raise NotSupported(
                self._ITERATOR_ERROR,
                span=self.span_tracker.get_span(stmt.iter),
                hint=self._ITERATOR_HINT,
            )

        iter_call = stmt.iter
        func = iter_call.func
        if isinstance(func, ast.Attribute) and func.attr in self._VALID_ITERATORS:
            return iter_call

        raise NotSupported(
            self._ITERATOR_ERROR,
            span=self.span_tracker.get_span(stmt.iter),
            hint=self._ITERATOR_HINT,
        )

    def _validate_loop_orelse(self, stmt: ast.For | ast.While) -> None:
        if stmt.orelse:
            kind = "for" if isinstance(stmt, ast.For) else "while"
            raise NotSupported(
                f"'{kind}-else' is not supported",
                span=self.span_tracker.get_span(stmt.orelse[0]),
            )

    def _parse_for_loop_target(self, stmt: ast.For) -> str:
        """Parse for loop target, returning the loop variable name."""
        if not isinstance(stmt.target, ast.Name):
            raise InvalidType(
                "For loop target must be a simple name",
                span=self.span_tracker.get_span(stmt.target),
                hint="Use: for i in pl.range(n)",
            )
        return stmt.target.id

    def _lower_for_loop(self, stmt: ast.For) -> ast.For:
        """Lower the Python-visible target to an assignment from an internal iterator."""
        loop_var_name = self._parse_for_loop_target(stmt)
        iterator_name = f"{loop_var_name}__iterator"

        target = ast.copy_location(ast.Name(id=loop_var_name, ctx=ast.Store()), stmt.target)
        iterator = ast.copy_location(ast.Name(id=iterator_name, ctx=ast.Load()), stmt.target)
        bind_target = ast.copy_location(ast.Assign(targets=[target], value=iterator), stmt.target)
        lowered = ast.copy_location(
            ast.For(
                target=ast.copy_location(ast.Name(id=iterator_name, ctx=ast.Store()), stmt.target),
                iter=stmt.iter,
                body=[bind_target, *stmt.body],
                orelse=stmt.orelse,
                type_comment=stmt.type_comment,
            ),
            stmt,
        )
        ast.fix_missing_locations(lowered)
        return lowered

    def _parse_range_call(self, call: ast.Call) -> dict[str, Any]:
        """Parse pl.range() call arguments.

        Args:
            call: AST Call node for pl.range()

        Returns:
            Dictionary with start, stop, step
        """
        if call.keywords:
            raise InvalidArgument(
                "pl.range() does not support keyword arguments",
                span=self.span_tracker.get_span(call),
                hint="Use: pl.range(stop), pl.range(start, stop), or pl.range(start, stop, step)",
            )

        if len(call.args) < 1:
            raise InvalidArgument(
                "pl.range() requires at least 1 argument (stop)",
                span=self.span_tracker.get_span(call),
                hint="Provide at least the stop value: pl.range(10) or pl.range(0, 10)",
            )

        start = 0
        step = 1

        if len(call.args) == 1:
            stop = self.parse_expression(call.args[0])
            self._check_range_bound(stop, call.args[0], subject="stop")
        elif len(call.args) == 2:
            start = self.parse_expression(call.args[0])
            stop = self.parse_expression(call.args[1])
            self._check_range_bound(start, call.args[0], subject="start")
            self._check_range_bound(stop, call.args[1], subject="stop")
        else:
            start = self.parse_expression(call.args[0])
            stop = self.parse_expression(call.args[1])
            step = self.parse_expression(call.args[2])
            self._check_range_bound(start, call.args[0], subject="start")
            self._check_range_bound(stop, call.args[1], subject="stop")
            self._check_range_bound(step, call.args[2], subject="step", is_step=True)

        self._check_range_overflow(start, stop, step, call)

        return {"start": start, "stop": stop, "step": step}

    def _reject_non_integer_scalar(self, expr: Any, node: ast.expr, *, subject: str) -> None:
        """Reject a pl.range() bound that is not an integer scalar."""
        scalar_type = getattr(expr, "type", None) if isinstance(expr, ir.Expr) else None
        if not isinstance(scalar_type, ir.ScalarType):
            raise InvalidVal(
                f"pl.range(): {subject} must be an integer scalar, got '{ast.unparse(node)}'",
                span=self.span_tracker.get_span(node),
                hint=(
                    "pl.range() bounds must be an integer scalar "
                    "(not a tuple/list, string, dtype, tile/tensor, or None)."
                ),
            )

        dtype = scalar_type.dtype
        if dtype == DataType.BOOL:
            bad = "bool"
        elif dtype.is_float():
            bad = "float"
        else:
            return
        raise InvalidVal(
            f"pl.range(): {subject} must be an integer, got {bad}",
            span=self.span_tracker.get_span(node),
            hint="pl.range() bounds must be integer-typed (no float or bool).",
        )

    def _check_range_bound(self, expr: Any, node: ast.expr, *, subject: str, is_step: bool = False) -> None:
        """Validate the type and numeric range of a single pl.range() argument."""
        self._reject_non_integer_scalar(expr, node, subject=subject)

        found, value = ExprEvaluator.ir_to_python_value(expr)
        if not found:
            return
        value = from_storage_int(expr.value, expr.type.dtype)

        in_vf = self.inline_vf_depth > 0
        hi = _UINT16_MAX if in_vf else INT64_MAX
        if is_step:
            lo = 1
            hint = f"pl.range() step must be a positive integer in [1, {hi}]."
        elif in_vf:
            lo = 0
            hint = "Inside a vector function, pl.range() bounds must be in [0, 65535] (uint16)."
        else:
            lo = INT64_MIN
            hint = "pl.range() bounds must fit int64."

        check_in_range(
            value,
            lo,
            hi,
            subject=f"pl.range() {subject}",
            span=self.span_tracker.get_span(node),
            hint=hint,
        )

    def _check_range_overflow(self, start: Any, stop: Any, step: Any, call: ast.Call) -> None:
        """Reject a loop whose variable would overflow its type on the final step."""
        fs, s = ExprEvaluator.ir_to_python_value(start)
        ft, t = ExprEvaluator.ir_to_python_value(stop)
        fp, p = ExprEvaluator.ir_to_python_value(step)
        if not (fs and ft and fp) or t <= s:
            return

        peak = s + ((t - s - 1) // p) * p + p
        hi = _UINT16_MAX if self.inline_vf_depth > 0 else INT64_MAX
        if peak > hi:
            raise OutOfRange(
                f"pl.range(): loop variable reaches {peak} on the final step, exceeding {hi}",
                span=self.span_tracker.get_span(call),
                hint="Reduce stop/step so that the last increment stays within the loop variable's range.",
            )
