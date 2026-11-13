# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""AST parsing for converting Python DSL to IR builder calls."""

from __future__ import annotations

__all__ = ["ASTParser", "ParseFacts"]


import ast
from dataclasses import dataclass, field
from functools import singledispatchmethod
from typing import Any

from pypto.pypto_impl import ir
from pypto_pro.ir import IRBuilder
from pypto_pro.ir.op.ptr_ops import TensorFacts

from ..._errors import (
    CommonInner,
    InvalidArgument,
    InvalidOperation,
    InvalidType,
    InvalidVal,
    NotSupported,
    PyptoProError,
    UntrackedTile,
    _warn_located,
    current_span,
)
from ..typing._tiling import get_tiling_fields, get_tiling_tuple_type, is_tiling_class
from ..typing.shape import _ShapePolicy
from ._assignment_parser import AssignmentParserMixin
from ._buffer_parser import BufferParserMixin, SelectorKind, TileGroupFacts
from ._call_parser import CallParserMixin, _check_type_compatible, _validate_simt_tensor_type
from ._control_flow_parser import ControlFlowParserMixin, validate_single_tail_return
from ._expr_evaluator import ExprEvaluator
from ._expression_parser import ExpressionParserMixin
from ._gemm_lowering import lower_gemm
from ._scope_manager import JumpKind, ScopeManager
from ._span_tracker import SpanTracker
from ._struct_parser import StructParserMixin
from ._tuple_type_registry import TupleTypeRegistry
from ._type_resolver import TypeResolver

# Statements whose span covers a whole indented suite rather than one line, so
# publishing it as the ambient location would underline the entire block.
_COMPOUND_STATEMENTS = (ast.For, ast.While, ast.If, ast.With, ast.Try, ast.FunctionDef)



def _snake_visit_name(node: ast.AST) -> str:
    """Return the snake_case visitor handler for an AST node type."""
    chars: list[str] = []
    for char in type(node).__name__:
        if char.isupper() and chars:
            chars.append("_")
        chars.append(char.lower())
    return f"visit_{''.join(chars)}"


def _infer_return_types_from_body(
    body: ir.Stmt,
    func_name: str,
    inferred: list[ir.Type] | None = None,
) -> list[ir.Type] | None:
    """Infer one return signature and reject conflicting value-returning paths."""
    if isinstance(body, ir.ReturnStmt):
        if body.value:
            actual = [value.type for value in body.value]
            if inferred is None:
                return actual
            consistent = len(actual) == len(inferred) and all(
                ir.structural_equal(reference, current, enable_auto_mapping=False)
                for reference, current in zip(inferred, actual)
            )
            if not consistent:
                reference_text = ", ".join(str(return_type) for return_type in inferred)
                actual_text = ", ".join(str(return_type) for return_type in actual)
                raise InvalidType(
                    f"Function '{func_name}' has inconsistent return types across return paths: "
                    f"{reference_text} and {actual_text}",
                    span=body.span,
                    hint="Make every value-returning path return the same type.",
                    parser_retry=True,
                )
        return inferred
    if isinstance(body, ir.SeqStmts):
        for stmt in body.stmts:
            inferred = _infer_return_types_from_body(stmt, func_name, inferred)
        return inferred
    if isinstance(body, ir.IfStmt):
        for branch in (body.then_body, body.else_body):
            if branch is not None:
                inferred = _infer_return_types_from_body(branch, func_name, inferred)
        return inferred
    if isinstance(body, (ir.ForStmt, ir.WhileStmt)):
        return _infer_return_types_from_body(body.body, func_name, inferred)
    return inferred


@dataclass(frozen=True, eq=False)
class SlotSelector:
    """One tile-group slot-selection call, shared by every operand reusing its tile."""

    kind: SelectorKind  # next / current / previous / index
    slot: int | None  # the static slot of an index selection, else None
    index_node: ast.expr | None  # the index expression when it is not a constant
    order: int  # parser selection order


@dataclass(frozen=True, eq=False)
class BufferRef:
    """A declared buffer that an expression may refer to, and how it selected a slot."""

    buffer: ir.Expr  # the declaration handle
    slot: int | None  # None when any slot may be referred to
    selector: SlotSelector | None = None

    @property
    def kind(self) -> str | None:
        return self.selector.kind if self.selector is not None else None

    @property
    def index_node(self) -> ast.expr | None:
        return self.selector.index_node if self.selector is not None else None

    @property
    def selector_id(self) -> int | None:
        return self.selector.order if self.selector is not None else None


@dataclass(frozen=True, eq=False)
class Operand:
    """One argument of an op that refers to a buffer."""

    position: int | None  # index among the positional arguments as written
    keyword: str | None  # keyword name when passed by keyword
    buffers: tuple[BufferRef, ...]


@dataclass(frozen=True, eq=False)
class OpAccess:
    """One op the parse emitted, with the arguments that refer to a buffer."""

    op_name: str  # "move", "getval", "setval", "vf.load_align", ...
    operands: tuple[Operand, ...]
    keyword_names: tuple[str, ...]  # every keyword the call wrote
    in_vf: bool  # whether the op is inside a vector function
    span: ir.Span


@dataclass
class ParseFacts:
    """Facts recorded during the parse for readers of its result.

    The automatic pipeline transform is today's only reader; a new fact is added
    here rather than as another field on the parser.
    """

    # Maps any Expr -> the buffers it may refer to. Buffers are always
    # declaration handles, never an alias, so one lookup is final.
    buffer_refs: dict[ir.Expr, tuple[BufferRef, ...]] = field(default_factory=dict)
    # Maps a buffer declaration handle -> that declaration's resolved facts.
    buffer_decls: dict[ir.Expr, TileGroupFacts | TensorFacts] = field(default_factory=dict)
    # Maps the (lineno, col_offset) of a kernel-body node -> the ops parsed under it, in
    # order. A key holding an empty list is a call that was expanded and reaches no buffer.
    op_accesses: dict[tuple[int, int] | None, list[OpAccess]] = field(default_factory=dict)
    # Maps the (lineno, col_offset) of a kernel-body `if` test -> (is_const, value).
    if_const_map: dict[tuple[int, int], tuple[bool, object]] = field(default_factory=dict)


class ASTParser(
    AssignmentParserMixin,
    ControlFlowParserMixin,
    ExpressionParserMixin,
    StructParserMixin,
    BufferParserMixin,
    CallParserMixin,
):
    """Parses Python AST and builds IR using IRBuilder."""

    def _in_simd_vf_body(self) -> bool:
        """True while parsing a __simd_vf__/__simd_callee__ function body."""
        return self._current_func_type in (
            ir.FunctionType.SimdVF, ir.FunctionType.SimdCallee,
            ir.FunctionType.SimdTileVF, ir.FunctionType.SimdTileCallee,
        )

    def _in_simd_tile_body(self) -> bool:
        return self._current_func_type in (ir.FunctionType.SimdTileVF, ir.FunctionType.SimdTileCallee)

    def __init__(
        self,
        source_file: str,
        source_lines: list[str],
        target: ir.SectionKind,
        line_offset: int = 0,
        col_offset: int = 0,
        strict_ssa: bool = False,
        closure_vars: dict[str, Any] | None = None,
        auto_mutex: bool = False,
        auto_pipeline: bool = False,
        debug_info: ir.IRDebugInfo | None = None,
        tilingkey_consts: dict[str, int] | None = None,
        datatype_consts: dict[str, Any] | None = None,
        bound_signature=None,
        void_return_only: bool = False,
        void_return_context: str = "this function",
        allow_early_return: bool = False,
    ):
        """Initialize AST parser.

        Args:
            source_file: Path to source file
            source_lines: Lines of source code (dedented for parsing)
            debug_info: Program-owned semantic tuple metadata table
            line_offset: Line number offset to add to AST line numbers (for dedented code)
            col_offset: Column offset to add to AST column numbers (for dedented code)
            strict_ssa: If True, enforce SSA (single assignment). If False (default), allow reassignment.
            closure_vars: Optional variables from the enclosing scope for dynamic shape resolution
            auto_mutex: If True, automatically insert mutex lock/unlock around buffer-managed tile ops.
            auto_pipeline: If True, this is the automatic pipeline's probe parse: collect the
                facts only the pipeline reads (compile-time if conditions) and warn about
                code it cannot track.
            void_return_only: If True, reject return values and non-None return annotations.
            void_return_context: User-facing name used only in diagnostics
                (for example, "@pl.jit" or "@pl.vector_function").
            allow_early_return: If True, skip the single-tail-return restriction.
                This is intentionally separate from void_return_only: @pl.jit
                are void-only but allow early/multiple returns, while @pl.vector_function
                and @pl.pipeline.stage are void-only and still require a single tail return.
            target: Required Cube/Vector target for target-specific parsing.
        """
        if target not in (ir.SectionKind.Cube, ir.SectionKind.Vector):
            raise NotSupported(f"Unsupported parser target: {target}")
        self._target = target
        self.matched_target = False
        self.has_tile_function: bool = False
        # Launch facts collected while this target Program is parsed. Tile
        # placement is compile-time constant, so keeping the Vec high-water
        # mark here avoids rediscovering it by walking the completed IR.
        self.max_vec_tile_end: int = 0
        self.requires_simt: bool = False
        self.span_tracker = SpanTracker(source_file, source_lines, line_offset, col_offset)
        self.scope_manager = ScopeManager(strict_ssa=strict_ssa)
        self._tilingkey_consts = tilingkey_consts
        self._datatype_consts = datatype_consts
        # Concrete tilingkey field values (per launch key) are injected as closure constants
        # so the parser folds field references (e.g. NeedAttnMask) to ConstInt — no template
        # params reach the IR. Field values win over any same-named closure var.
        merged_closure = {
            **(closure_vars or {}),
            **(tilingkey_consts or {}),
            **(datatype_consts or {}),
        }
        # local_binding lets every compile-time position (make_tile_group addrs/mutex_ids,
        # shapes, axes, ``x in <list>``, ...) read kernel-local names such as
        # ``addr = 0x10000``, which live in the parser rather than in the closure.
        # Bound as a method: const_env is created further down and remains append-only
        # because its keys are globally unique physical SSA names.
        self.expr_evaluator = ExprEvaluator(
            closure_vars=merged_closure,
            span_tracker=self.span_tracker,
            local_binding=self.local_binding,
        )
        self.type_resolver = TypeResolver(
            expr_evaluator=self.expr_evaluator,
            scope_lookup=self.scope_manager.lookup_var_bounded,
            span_tracker=self.span_tracker,
            bound_signature=bound_signature,
        )
        self.builder = IRBuilder()
        if debug_info is None:
            raise ValueError("ASTParser requires a non-null IRDebugInfo")
        # Tuple metadata is owned by the Program being built. Nested parsers share
        # this same IRDebugInfo instance.
        self.debug_info = debug_info
        self.tuple_type_registry = TupleTypeRegistry(self.debug_info)
        self.external_funcs: dict[str, ir.Function] = {}  # Track external functions referenced

        # Track active control-flow builders while parsing nested statements.
        self.in_for_loop = False
        self.in_while_loop = False
        self.in_if_stmt = False
        self.current_if_builder = None
        self.current_loop_builder = None

        # Immutable source templates for Python functions expanded at their call sites.
        self.inline_func_cache: dict[int, Any] = {}
        self.inline_call_stack: list[int] = []
        self.inline_counter: int = 0
        self.vf_func_cache: dict[tuple, ir.Function] = {}
        self.vf_call_stack: list[int] = []
        self._simt_entry_type = ir.FunctionType.Opaque

        # Counter for anonymous buffer tile variables (auto-named _buf_tile_N).
        self._buf_tile_counter: int = 0
        # Counter for let-bound buf_idx variables in tile-group cursor selection.
        self._tuple_idx_counter: int = 0
        self._slot_selector_counter: int = 0
        self._auto_mutex_placeholder_counter: int = 0
        self._expr_tmp_counter: int = 0
        self._ifexpr_tmp_counter: int = 0
        # Maps group Expr objects -> (depth, mutex_ids, memory).
        self.tile_group_meta: dict[ir.Expr, tuple] = {}
        self.parse_facts = ParseFacts()
        # Where an op parsed from here on is recorded: the kernel-body node it sits under.
        self._access_site: tuple[int, int] | None = None
        # Parser-only constant environment. Runtime bindings remain exclusively in
        # ScopeManager as Vars; this map records constants by physical SSA name.
        self.const_env: dict[str, ir.Expr] = {}
        # Immutable tuple expressions remain addressable through their physical SSA
        # names even when their elements are runtime values. This is separate from
        # const_env so dynamic tuples do not participate in constant propagation.
        self.make_tuple_env: dict[str, ir.MakeTuple] = {}
        # Physical SSA names that only carry the valueless initial retval used by
        # inline-return lowering.  They are valid IR operands, but not semantic
        # inputs to control-flow type or constant inference.
        self._empty_control_flow_values: set[str] = set()
        # struct_array is the sole mutable tuple container and must retain
        # GetItemExpr alias semantics; every other MakeTuple is immutable.
        self._struct_array_tuples: set[ir.MakeTuple] = set()

        # Maps tile expressions to parser-tracked [row, col] valid shapes from
        # explicit set_validshape calls. This metadata is used to clamp subviews;
        # public tile.valid_shape reads are represented by block.tile_valid_shape.
        self._tile_valid_shape: dict[ir.Expr, list] = {}

        # MakeTuples that already have an emitted assignment anchor. Keeping the
        # Expr objects themselves alive is required: a Python wrapper's id() can
        # be reused while parsing, which would otherwise suppress an unrelated
        # tuple's anchor.
        self._anchored_make_tuples: set[ir.MakeTuple] = set()

        # Cache: (tuple_var_name, index_ssa_var_name) -> phi ir.Var from _build_tuple_index_chain.
        # Applies to all tuple types (tile, tensor, event ID, etc.).
        # Prevents re-emitting an if-else chain when the same buf[idx] expression
        # appears multiple times in the same linear code region.
        self._tuple_select_cache: dict[tuple[str, int], ir.Var] = {}

        self._auto_mutex = auto_mutex
        self._auto_pipeline = auto_pipeline
        self._parsed_expr_cache: dict[ast.expr, Any] = {}
        self._gemm_counter = 0
        self._gemm_used_names: set[str] = set()
        self._current_func_type = ir.FunctionType.Opaque
        self._void_return_only = void_return_only
        self._void_return_context = void_return_context
        self._allow_early_return = allow_early_return
        # Current assignment LHS name; set before parse_expression so helpers
        # (_build_tile_group_ir, _parse_struct_array_expr) can name intermediate vars.
        self.current_target_name: str = ""

        self._current_node: ast.AST | None = None

    @property
    def auto_mutex_enabled(self) -> bool:
        """Return whether automatic mutex emission is enabled."""
        return self._auto_mutex

    @property
    def auto_pipeline_enabled(self) -> bool:
        """Return whether this is the automatic pipeline's probe parse."""
        return self._auto_pipeline

    @property
    def target(self) -> ir.SectionKind:
        """Return the immutable Cube/Vector parse target."""
        return self._target

    def _record_tile_high_water(self, tile: ir.Expr) -> None:
        """Record the Vec high-water mark of an already-created Tile expression."""
        memref = tile.type.memref
        if memref is not None and memref.memory_space_ == ir.MemorySpace.Vec:
            addr = memref.addr
            size = memref.size_
            if not isinstance(addr, ir.ConstInt):
                raise CommonInner("Vec Tile address must be a compile-time constant for mixed-kernel UB analysis")
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                raise CommonInner(f"Vec Tile size must be a positive compile-time integer, got {size!r}")
            self.max_vec_tile_end = max(self.max_vec_tile_end, int(addr.value) + size)

    @staticmethod
    def _attach_ptr_to_tensor_type(name: str, param_type: ir.Type, span: ir.Span) -> ir.Type:
        if not isinstance(param_type, ir.TensorType):
            return param_type
        ptr_var = ir.Var(name + "_ptr", ir.PtrType(param_type.dtype), span)
        tv = (
            ir.TensorView()
            if param_type.tensor_view is None
            else ir.TensorView(
                param_type.tensor_view.valid_shape, param_type.tensor_view.stride, param_type.tensor_view.layout
            )
        )
        tv.ptr = ptr_var
        return ir.TensorType(param_type.shape, param_type.dtype, param_type.memref, tv)

    def record_tile_valid_shape(self, tile_expr: ir.Expr, valid_shape) -> None:
        elements = valid_shape.elements if isinstance(valid_shape, ir.MakeTuple) else valid_shape
        self._tile_valid_shape[tile_expr] = list(elements)

    def get_tile_valid_shape(self, tile_expr: ir.Expr):
        return self._tile_valid_shape.get(tile_expr)

    def set_auto_mutex_enabled(self, enabled: bool) -> None:
        """Set automatic mutex emission for helper parsing."""
        self._auto_mutex = enabled

    def resolve_tiling_class(self, annotation: ast.expr) -> type | None:
        """Return the tiling class if annotation refers to one in closure_vars, else None.

        Args:
            annotation: AST expression node for the annotation

        Returns:
            The resolved tiling class, or None if the annotation is not a tiling class
        """
        if not isinstance(annotation, ast.Name):
            return None
        cls = self.expr_evaluator.closure_vars.get(annotation.id)
        return cls if is_tiling_class(cls) else None

    @staticmethod
    def _window_param_class() -> type | None:
        """The pybind Window class from pypto_impl.dist, or None when unavailable.

        Absent on non-CANN builds (the dist submodule is compiled under BUILD_WITH_CANN),
        where window annotations are simply not a thing.
        """
        from pypto import pypto_impl

        return getattr(getattr(pypto_impl, "dist", None), "Window", None)

    def is_window_annotation(self, annotation: ast.expr) -> bool:
        """True when the annotation evaluates to the pld.Window class (``win: pld.Window``)."""
        if not isinstance(annotation, (ast.Name, ast.Attribute)):
            return False
        window_cls = self._window_param_class()
        if window_cls is None:
            return False
        try:
            success, value = self.expr_evaluator.try_eval_expr(annotation)
        except Exception:  # probing an arbitrary annotation; evaluation failure means "not ours"
            return False
        return success and value is window_cls

    def parse_function(
        self,
        func_def: ast.FunctionDef,
        func_type: ir.FunctionType = ir.FunctionType.Opaque,
        callsite_param_types: dict[str, ir.Type] | None = None,
        const_param_values: dict[str, ir.Expr] | None = None,
        callsite_param_buffers: dict[str, tuple] | None = None,
        func_attrs: dict[str, Any] | None = None,
    ) -> ir.Function:
        """Parse function definition and build IR.

        Args:
            func_def: AST FunctionDef node
            func_type: Function type (default: Opaque)

            callsite_param_types: Concrete parameter types supplied while parsing
                a SIMT or SIMD VF function at its call site.

            const_param_values: Compile-time constant call-site arguments keyed
                by parameter name, supplied while instantiating a SIMD VF
                function. The parameter name resolves to the constant inside the
                body (folding ``if`` conditions on it), and the parameter is
                specialized out of the emitted signature.

            callsite_param_buffers: The buffers each call-site argument reaches,
                keyed by parameter name, supplied while instantiating a SIMD VF
                function for the automatic pipeline's probe parse. A parameter
                then reaches the caller's buffer, so the ops of the body are
                recorded against it like the caller's own.

            func_attrs: Attributes attached to the parsed IR function.

        Returns:
            IR Function object
        """
        if (
            callsite_param_types is not None
            or const_param_values is not None
            or callsite_param_buffers is not None
        ) and func_type not in (
            ir.FunctionType.SimtVF,
            ir.FunctionType.SimtCallee,
            ir.FunctionType.SimdVF,
            ir.FunctionType.SimdCallee,
            ir.FunctionType.SimdTileVF,
            ir.FunctionType.SimdTileCallee,
        ):
            raise NotSupported(
                "callsite_param_types/const_param_values/callsite_param_buffers is only "
                "supported for SIMT/SIMD VF functions"
            )

        func_name = func_def.name
        self._gemm_used_names.update(node.id for node in ast.walk(func_def) if isinstance(node, ast.Name))
        func_span = self.span_tracker.get_span(func_def)

        if (
            self._void_return_only
            and func_def.returns is not None
            and not (isinstance(func_def.returns, ast.Constant) and func_def.returns.value is None)
        ):
            raise NotSupported(
                f"{self._void_return_context} only supports a None return annotation; "
                "returning values is not supported.",
                span=func_span,
                hint=(
                    "Remove the return annotation or use -> None. Do not write `return <value>`; "
                    "only use `return` or `return None`. Pass output Tensor/Tile/buffer parameters "
                    "for data results."
                ),
            )

        if not self._allow_early_return:
            context = self._void_return_context if self._void_return_only else f"Function '{func_name}'"
            return_error = validate_single_tail_return(func_def, context)
            if return_error is not None:
                return_node, message, hint = return_error
                raise InvalidOperation(
                    message,
                    span=self.span_tracker.get_span(return_node),
                    hint=hint,
                )

        self._current_func_type = func_type
        if func_type in (ir.FunctionType.SimtKernel, ir.FunctionType.SimtVF):
            self._simt_entry_type = func_type
        self.scope_manager.enter_scope("function")

        # Collect args to process, filtering out bare 'self'
        args_to_process = [arg for arg in func_def.args.args if not (arg.arg == "self" and arg.annotation is None)]

        self._validate_tiling_params(args_to_process, func_def)

        self._anchored_make_tuples.clear()

        with self.builder.function(func_name, func_span, func_type=func_type) as f:
            for parameter_index, arg in enumerate(args_to_process):
                const_value = const_param_values.get(arg.arg) if const_param_values is not None else None
                if const_value is not None:
                    self.scope_manager.define_var(
                        arg.arg, const_value, allow_redef=True, span=self.span_tracker.get_span(arg)
                    )
                    continue
                callsite_type = callsite_param_types.get(arg.arg) if callsite_param_types is not None else None
                self._parse_function_param(arg, f, parameter_index, callsite_type)
                if callsite_param_buffers:
                    # The parameter reaches whatever the caller passed, so an op of this body
                    # is recorded against the caller's buffer rather than against a tile with
                    # no declaration behind it.
                    refs = callsite_param_buffers.get(arg.arg)
                    if refs:
                        self.parse_facts.buffer_refs[self.lookup_expr_by_name(arg.arg)] = refs

            # Give closure tuples a function-entry array-materialization anchor.
            self._hoist_closure_tuples()

            # Collect function body statements (skip docstrings)
            body_stmts: list[ast.stmt] = []
            for i, stmt in enumerate(func_def.body):
                if i == 0 and isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                    if isinstance(stmt.value.value, str):
                        continue  # Skip docstring
                body_stmts.append(stmt)

            # Standalone __simd_vf__/__simd_callee__ body: the "inside a VF
            # body" semantics (UINT16 loop-bound clamps, tile+offset pointer
            # arithmetic) are keyed off
            # the function type via _in_simd_vf_body() rather than a counter.
            self._parse_statement_list(body_stmts)

            if self.scope_manager.current_scope.jump_kind is None:
                self.builder.emit(ir.ReturnStmt([], self._current_span()))
                self.scope_manager.current_scope.set_jump(JumpKind.RETURN)

        self.scope_manager.exit_scope()
        result = f.get_result()
        inferred = (_infer_return_types_from_body(result.body, func_name) if result.body else None) or []
        if func_def.returns is not None:
            # tuple[...] resolves to a single TupleType, matching a `return a, b`
            # body (one MakeTuple expr), so a tuple return is one return type.
            annotated_return_types = [self.type_resolver.resolve_type(func_def.returns)]

            return_span = self.span_tracker.get_span(func_def.returns)
            if len(annotated_return_types) != len(inferred):
                raise InvalidArgument(
                    f"Return annotation for '{func_name}' expects "
                    f"{len(annotated_return_types)} value(s), got {len(inferred)}",
                    span=return_span,
                    parser_retry=True,
                )
            for annotated, actual in zip(annotated_return_types, inferred):
                with current_span(return_span):
                    _check_type_compatible(
                        annotated,
                        actual,
                        what="Return",
                        name=func_name,
                    )

        attrs = dict(result.attrs)
        attrs.update(func_attrs or {})
        result = ir.Function(
            result.name,
            list(result.params),
            inferred,
            result.body,
            result.span,
            result.func_type,
            result.entry,
            attrs,
        )
        return result

    def parse_statement(self, stmt: ast.stmt) -> None:
        """Parse a statement node.

        Args:
            stmt: AST statement node
        """
        self._current_node = stmt
        # Where the ops this statement emits are recorded, while it is the kernel body's own.
        # A vector function body is parsed at its call site and keeps the site handed to it.
        if not self.inline_call_stack and not self._in_simd_vf_body():
            self._access_site = (stmt.lineno, stmt.col_offset)
        # Publish where this statement was written, for anything raised below that
        # passed no span of its own. A compound statement is excluded: its span
        # covers the whole suite, so an error from deep inside the body would
        # underline screenfuls of unrelated code. Their headers already build an
        # explicit span, and the statements in their bodies come back through here.
        if isinstance(stmt, _COMPOUND_STATEMENTS):
            self._dispatch_statement(stmt)
            return
        with current_span(self.span_tracker.get_span(stmt)):
            self._dispatch_statement(stmt)

    @singledispatchmethod
    def _dispatch_statement(self, stmt: ast.stmt) -> None:
        raise InvalidType(
            f"Unsupported statement type: {type(stmt).__name__}",
            span=self.span_tracker.get_span(stmt),
            hint="Only assignments, for loops, while loops, if statements, "
            "with statements, returns, break, and continue are supported in DSL functions",
            parser_retry=True,
        )

    @_dispatch_statement.register
    def _parse_annotated_assignment_statement(self, stmt: ast.AnnAssign) -> None:
        self.parse_annotated_assignment(stmt)

    @_dispatch_statement.register
    def _parse_assignment_statement(self, stmt: ast.Assign) -> None:
        self.parse_assignment(stmt)

    @_dispatch_statement.register
    def _parse_augmented_assignment_statement(self, stmt: ast.AugAssign) -> None:
        if isinstance(stmt.target, ast.Subscript):
            target_load = ast.copy_location(
                ast.Subscript(value=stmt.target.value, slice=stmt.target.slice, ctx=ast.Load()),
                stmt.target,
            )
        elif isinstance(stmt.target, ast.Name):
            target_load = ast.copy_location(
                ast.Name(id=stmt.target.id, ctx=ast.Load()),
                stmt.target,
            )
        else:
            target_load = stmt.target
        equivalent = ast.Assign(
            targets=[stmt.target],
            value=ast.BinOp(left=target_load, op=stmt.op, right=stmt.value),
            type_comment=None,
        )
        ast.copy_location(equivalent, stmt)
        ast.fix_missing_locations(equivalent)
        self.parse_assignment(equivalent)

    @_dispatch_statement.register
    def _parse_for_statement(self, stmt: ast.For) -> None:
        self.parse_for_loop(stmt)

    @_dispatch_statement.register
    def _parse_while_statement(self, stmt: ast.While) -> None:
        self.parse_while_loop(stmt)

    @_dispatch_statement.register
    def _parse_if_statement(self, stmt: ast.If) -> None:
        self.parse_if_statement(stmt)

    @_dispatch_statement.register
    def _parse_with_statement(self, stmt: ast.With) -> None:
        self.parse_with_statement(stmt)

    @_dispatch_statement.register
    def _parse_return_statement(self, stmt: ast.Return) -> None:
        self.parse_return(stmt)

    @_dispatch_statement.register
    def _parse_break_statement(self, stmt: ast.Break) -> None:
        self.parse_break(stmt)

    @_dispatch_statement.register
    def _parse_continue_statement(self, stmt: ast.Continue) -> None:
        self.parse_continue(stmt)

    @_dispatch_statement.register
    def _parse_evaluation_statement(self, stmt: ast.Expr) -> None:
        lowered = lower_gemm(self, stmt)
        if lowered is not None:
            for inner in lowered:
                self.parse_statement(inner)
            return
        self.parse_evaluation_statement(stmt)

    @_dispatch_statement.register
    def _parse_pass_statement(self, stmt: ast.Pass) -> None:
        pass  # No-op: pass statements are valid in DSL functions

    def _transfer_tile_group_metadata(self, target: ir.Expr, *sources: ir.Expr | None) -> None:
        """Transfer tile-group metadata from ``sources`` to ``target``.

        Tile-group accessor metadata is copied only from a single source; the
        buffer slots referred to are merged from all sources.
        """
        if len(sources) == 1 and sources[0] is not None:
            gm = self.tile_group_meta.get(sources[0])
            if gm is not None:
                self.tile_group_meta[target] = gm
        refs = {
            (id(ref.buffer), ref.slot, id(ref.selector)): ref
            for source in sources
            for ref in self.buffer_refs_of(source)
        }
        if refs:
            self.parse_facts.buffer_refs[target] = tuple(refs.values())

    def _bind_buffer_ref(
        self,
        target: ir.Expr,
        buffer: ir.Expr,
        slot: int | None,
        selector: SlotSelector | None = None,
    ) -> None:
        """Record that ``target`` refers to ``slot`` of the buffers ``buffer`` refers to."""
        refs = tuple(BufferRef(decl, slot, selector) for decl in self.buffers_of(buffer))
        if refs:
            self.parse_facts.buffer_refs[target] = refs

    def _new_slot_selector(
        self, kind: SelectorKind, slot: int | None = None, index_node: ast.expr | None = None
    ) -> SlotSelector:
        """Allocate one selector for a single slot-selection call."""
        selector = SlotSelector(kind, slot, index_node, self._slot_selector_counter)
        self._slot_selector_counter += 1
        return selector

    def buffer_refs_of(self, expr) -> tuple[BufferRef, ...]:
        """Return the buffers ``expr`` may refer to, empty when it refers to none.

        A recorded entry answers first, so an alias or a rebuilt handle reports
        the declaration it came from rather than itself.
        """
        if not isinstance(expr, ir.Expr):
            return ()
        refs = self.parse_facts.buffer_refs.get(expr)
        if refs is not None:
            return refs
        if expr in self.parse_facts.buffer_decls:
            return (BufferRef(expr, None),)
        return ()

    def buffers_of(self, expr) -> tuple[ir.Expr, ...]:
        """Return the buffer declarations ``expr`` refers to, empty when it refers to none."""
        return tuple({id(ref.buffer): ref.buffer for ref in self.buffer_refs_of(expr)}.values())

    def static_slot(self, expr) -> BufferRef | None:
        """Return the one slot ``expr`` refers to when it is known at compile time."""
        refs = self.buffer_refs_of(expr)
        if len(refs) == 1 and refs[0].slot is not None:
            return refs[0]
        return None

    def is_runtime_slot(self, expr) -> bool:
        """Whether ``expr`` is a tile referring to a buffer slot not known at compile time.

        Only a tile has a slot; a whole buffer declaration carries no slot of its own.
        """
        if not isinstance(getattr(expr, "type", None), ir.TileType):
            return False
        return bool(self.buffer_refs_of(expr)) and self.static_slot(expr) is None

    def _record_op_access(
        self,
        op_name: str,
        operands: list[tuple[int | None, str | None, Any]],
        keyword_names: tuple[str, ...],
        span: ir.Span,
    ) -> None:
        """Record one emitted op, from the arguments of it that reach a buffer.

        Only the automatic pipeline reads these records, so nothing is recorded without
        auto_pipeline. ``operands`` holds (position, keyword, parsed value) as the call wrote
        them. An op no argument of which reaches a buffer is not recorded. A tile that
        resolves to no declaration is recorded with no buffers, so a consumer needing its
        buffer can tell it apart from an argument that is not a tile at all, and is warned
        about: the pipeline inserts no sync for it.
        """
        if not self._auto_pipeline:
            return
        recorded = []
        reaches_buffer = False
        untracked = None
        for position, keyword, value in operands:
            refs = self.buffer_refs_of(value)
            if refs:
                reaches_buffer = True
            elif not isinstance(getattr(value, "type", None), ir.TileType):
                continue
            elif untracked is None:
                untracked = keyword if keyword is not None else f"#{position}"
            recorded.append(Operand(position, keyword, refs))
        if untracked is not None:
            shown = op_name if op_name.startswith("vf.") else f"pl.{op_name}"
            _warn_located(
                f"pipeline: argument {untracked} of `{shown}()` is a tile that no "
                f"pl.make_tile_group() declaration provides, so the automatic pipeline inserts "
                f"no sync for it. Declare the tiles taking part in the pipeline with "
                f"pl.make_tile_group().",
                UntrackedTile,
                span,
            )
        if not reaches_buffer:
            return
        self.parse_facts.op_accesses.setdefault(self._access_site, []).append(
            OpAccess(op_name, tuple(recorded), keyword_names, self._in_simd_vf_body(), span)
        )

    def _validate_tiling_params(
        self,
        args_to_process: list[ast.arg],
        func_def: ast.FunctionDef,
    ) -> None:
        """Pre-validate tiling constraints: at most 1 tiling param, must be last."""
        tiling_param_names = [
            arg.arg
            for arg in args_to_process
            if arg.annotation is not None and self.resolve_tiling_class(arg.annotation) is not None
        ]
        if len(tiling_param_names) > 1:
            raise InvalidArgument(
                f"Function '{func_def.name}' has {len(tiling_param_names)} tiling parameters "
                f"({', '.join(tiling_param_names)}), but at most 1 is allowed",
                span=self.span_tracker.get_span(func_def),
                hint="A kernel may have at most one tiling parameter",
            )
        if len(tiling_param_names) == 1:
            if not args_to_process or args_to_process[-1].arg != tiling_param_names[0]:
                tiling_arg = next(a for a in args_to_process if a.arg == tiling_param_names[0])
                raise InvalidOperation(
                    f"Tiling parameter '{tiling_param_names[0]}' must be the last parameter",
                    span=self.span_tracker.get_span(tiling_arg),
                    hint="Move the tiling parameter to the last position",
                )

    @staticmethod
    def _validate_simt_kernel_param(param_name: str, param_type: ir.Type, param_span: ir.Span) -> None:
        if isinstance(param_type, ir.ScalarType):
            return
        if not isinstance(param_type, ir.TensorType):
            raise InvalidVal(
                f"SIMT kernel parameter '{param_name}' must resolve to a scalar or Tensor, got {param_type}",
                span=param_span,
                hint="Tile, Ptr, tuple, and tiling parameters are not supported.",
            )
        subject = f"SIMT kernel Tensor parameter '{param_name}'"
        with current_span(param_span):
            _validate_simt_tensor_type(param_type, subject)
        if any(not isinstance(dim, ir.ConstInt) for dim in param_type.shape):
            raise InvalidType(f"{subject} requires a static shape", span=param_span)

    def _parse_function_param(
        self,
        arg: ast.arg,
        f: Any,
        parameter_index: int,
        callsite_type: ir.Type | None = None,
    ) -> None:
        """Parse a single function parameter and register it in scope or tiling registry."""
        param_name = arg.arg
        param_span = self.span_tracker.get_span(arg)

        # 1) tiling-class annotation: always expand via the tiling path (even if a
        # call-site type exists for this name).
        tiling_cls = self.resolve_tiling_class(arg.annotation) if arg.annotation else None
        if tiling_cls is not None:
            if self._current_func_type == ir.FunctionType.SimtKernel:
                raise InvalidType(
                    f"SIMT kernel parameter '{param_name}' must be a scalar or pl.Tensor",
                    span=param_span,
                )
            fields = get_tiling_fields(tiling_cls)
            # Single struct parameter: the tiling class is lowered to a TupleType whose
            # STRUCT TupleTypeInfo is recorded in IRDebugInfo.
            tuple_type = get_tiling_tuple_type(tiling_cls)
            tiling_var = f.param(param_name, tuple_type, param_span)
            # A tiling param is a struct: register its fields and the Python class name, so
            # codegen emits `struct <ClassName>` (matching the host-side struct).
            self.register_struct_type(tiling_var, tiling_cls.__name__, list(fields.keys()))
            self.scope_manager.define_var(param_name, tiling_var, allow_redef=True)
            return

        # 1b) pld.Window annotation: an opaque WindowType handle. The launch ABI passes
        # the device dist::Window descriptor pointer; field reads lower to scalar GM
        # loads via dist.window_get.
        if arg.annotation is not None and self.is_window_annotation(arg.annotation):
            window_var = f.param(param_name, ir.WindowType(), param_span)
            self.scope_manager.define_var(param_name, window_var, allow_redef=True)
            return

        # 2) A call-site type is available for delayed SIMT function parsing.
        if callsite_type is not None:
            if arg.annotation is not None:
                annotated_type = self.type_resolver.resolve_param_type(
                    arg.annotation, parameter_name=param_name, parameter_index=parameter_index
                )
                with current_span(param_span):
                    _check_type_compatible(
                        annotated_type, callsite_type, what="SIMT parameter", name=param_name
                    )
            param_type = self._attach_ptr_to_tensor_type(param_name, callsite_type, param_span)
            param_var = f.param(param_name, param_type, param_span)
            self.scope_manager.define_var(param_name, param_var, allow_redef=True)
            return

        # 3) Annotation-only path for eagerly parsed functions.
        if arg.annotation is None:
            raise InvalidType(
                f"Parameter '{param_name}' missing type annotation",
                span=param_span,
                hint="Add a type annotation like: x: pl.Tensor[[64], pl.DT_FP32]",
                parser_retry=True,
            )
        param_type = self.type_resolver.resolve_param_type(
            arg.annotation, parameter_name=param_name, parameter_index=parameter_index
        )
        param_type = self._attach_ptr_to_tensor_type(param_name, param_type, param_span)
        if self._current_func_type == ir.FunctionType.SimtKernel:
            self._validate_simt_kernel_param(param_name, param_type, param_span)
        param_var = f.param(param_name, param_type, param_span)
        self.scope_manager.define_var(param_name, param_var, allow_redef=True)

    def _hoist_closure_tuples(self) -> None:
        """Anchor convertible closure tuple/list values at function entry.

        Name reads intentionally continue to resolve to the original MakeTuple.
        The emitted lets exist only to give CCE a stable lexical location for
        backing-array materialization.
        """
        seen: dict[int, tuple[ir.MakeTuple, ir.Var]] = {}
        span = ir.Span.unknown()
        for var_name, value in self.expr_evaluator.closure_vars.items():
            has_scope_binding = self.scope_manager.lookup_var_bounded(var_name) is not None
            if not isinstance(value, (tuple, list)) or has_scope_binding:
                continue
            if any(isinstance(v, _ShapePolicy) or v is Ellipsis for v in value):
                continue
            # Only flat sequences are anchored. A nested one is a multi-dimensional array,
            # which has no CCE backing-array form, so hoisting it would emit a tuple whose
            # elements have no C++ name of their own -- and the point of the hoist is to give
            # the backing array a home. Names like these reach the body through
            # the parser's value environments.
            if any(isinstance(v, (tuple, list)) for v in value):
                continue
            entry = seen.get(id(value))
            if entry is None:
                try:
                    tuple_expr = self.expr_evaluator.python_value_to_ir(value, span)
                except (PyptoProError, TypeError, ValueError):
                    continue
                if not isinstance(tuple_expr, ir.MakeTuple):
                    continue
                tuple_var = self.builder.let(var_name, tuple_expr, span=span)
                self._anchored_make_tuples.add(tuple_expr)
                seen[id(value)] = (tuple_expr, tuple_var)
            else:
                tuple_expr, tuple_var = entry
            self.scope_manager.define_var(var_name, tuple_var, allow_redef=True)
            self._update_var_envs(tuple_var, tuple_expr)
