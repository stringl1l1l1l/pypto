# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Call and operation parsing helpers for ASTParser."""

from __future__ import annotations

import ast
from collections import namedtuple
import copy
from dataclasses import dataclass
import logging
from typing import Any, Callable

from pypto.pypto_impl import ir
from pypto_pro.ir._utils import _is_int
from pypto_pro.ir.op._op_registry import _OP_REGISTRY
from pypto_pro.ir.op.block_ops import block_ir_op

from ._control_flow_parser import _is_bare_return
from ._expr_evaluator import ExprEvaluator
from ._span_tracker import SpanTracker
from .diagnostics import (
    FinalRejectionError,
    ParserSyntaxError,
    ParserTypeError,
    UndefinedVariableError,
    UnsupportedFeatureError,
)

logger = logging.getLogger(__name__)


# Tile reference for auto mutex (optional mutex metadata, memory, dedup id).
_MutexRef = namedtuple("_MutexRef", "buf_ids mutex_ids memory slot_id")


@dataclass(frozen=True)
class _InlineFunctionTemplate:
    """Immutable source template for one directly-expanded Python callable."""

    func_def: ast.FunctionDef
    closure_vars: dict[str, Any]
    is_vector_function: bool
    span_tracker: SpanTracker


class _InlineLocalRenamer(ast.NodeTransformer):
    """Rename bindings local to one helper expansion without touching closures."""

    def __init__(self, names: set[str], prefix: str):
        self._names = names
        self._rename = {name: f"{prefix}{name}" for name in names}

    def generic_visit(self, node):
        if isinstance(node, ast.Name) and node.id in self._rename:
            node.id = self._rename[node.id]
        return super().generic_visit(node)


class _InlineReturnLowerer(ast.NodeTransformer):
    """Lower helper returns and their loop-propagation checks in Python AST."""

    # ast.NodeTransformer dispatches by reflection on `visit_<NodeClassName>`, so the CamelCase
    # halves of visit_Return / visit_For / visit_While are the library's spelling, not ours.
    # Renaming them to snake_case silently turns the visitor into a no-op.
    # pylint: disable=huawei-invalid-name

    def __init__(self, return_val_name: str, returned_name: str):
        self._return_val_name = return_val_name
        self._returned_name = returned_name

    @staticmethod
    def _assignment(name: str, value: ast.expr, location: ast.AST) -> ast.Assign:
        target = ast.copy_location(ast.Name(id=name, ctx=ast.Store()), location)
        return ast.copy_location(ast.Assign(targets=[target], value=value), location)

    def _return_checkpoint(self, location: ast.AST) -> ast.If:
        condition = ast.copy_location(ast.Name(id=self._returned_name, ctx=ast.Load()), location)
        break_stmt = ast.copy_location(ast.Break(), location)
        return ast.copy_location(ast.If(test=condition, body=[break_stmt], orelse=[]), location)

    def visit_Return(self, node: ast.Return):
        lowered: list[ast.stmt] = []
        if not _is_bare_return(node):
            lowered.append(self._assignment(self._return_val_name, node.value, node))
        returned = ast.copy_location(ast.Constant(value=True), node)
        lowered.append(self._assignment(self._returned_name, returned, node))
        lowered.append(ast.copy_location(ast.Break(), node))
        return lowered

    def _lower_loop(self, node: ast.For | ast.While) -> list[ast.stmt]:
        lowered_loop = self.generic_visit(node)
        return [lowered_loop, self._return_checkpoint(node)]

    def visit_For(self, node: ast.For):
        return self._lower_loop(node)

    def visit_While(self, node: ast.While):
        return self._lower_loop(node)

    def lower(self, body: list[ast.stmt]) -> list[ast.stmt]:
        module = ast.Module(body=body, type_ignores=[])
        self.visit(module)
        location = body[0]
        return_val_init = self._assignment(
            self._return_val_name,
            ast.copy_location(ast.Constant(value=None), location),
            location,
        )
        returned_init = self._assignment(
            self._returned_name,
            ast.copy_location(ast.Constant(value=False), location),
            location,
        )
        wrapper = ast.copy_location(
            ast.While(
                test=ast.copy_location(ast.Constant(value=True), location),
                body=[returned_init, *module.body, ast.copy_location(ast.Break(), location)],
                orelse=[],
            ),
            location,
        )
        ast.fix_missing_locations(wrapper)
        return [return_val_init, wrapper]


# Builtin function names that map to pl.* ops (syntax sugar).
_BUILTIN_TO_OP: dict[str, str] = {
    "min": "min",
    "max": "max",
}


def _dtypes_compatible(a: ir.DataType, b: ir.DataType) -> bool:
    """Lenient dtype compatibility: same dtype, or both integer-like / both float-like.

    Different numeric dtypes of the same family (e.g. INT32 and INT64, FP16 and FP32) are
    interconvertible and treated as compatible.
    """
    return a == b or (a.is_int() and b.is_int()) or (a.is_float() and b.is_float())


def _types_compatible(annotated: ir.Type, actual: ir.Type) -> bool:
    """Lenient type compatibility: same kind + compatible dtype.

    Ignores shape/memref/layout details. ``UnknownType`` is compatible with anything.
    Used to validate a function annotation against the type derived from the actual
    argument/return-value expression.
    """
    if isinstance(annotated, ir.UnknownType) or isinstance(actual, ir.UnknownType):
        return True
    if isinstance(annotated, ir.ScalarType) and isinstance(actual, ir.ScalarType):
        return _dtypes_compatible(annotated.dtype, actual.dtype)
    if isinstance(annotated, ir.PtrType) and isinstance(actual, ir.PtrType):
        return _dtypes_compatible(annotated.dtype, actual.dtype)
    # TensorType / TileType both derive from ShapedType; compare dtype, ignore shape/memref.
    if isinstance(annotated, ir.ShapedType) and isinstance(actual, ir.ShapedType):
        return type(annotated) is type(actual) and _dtypes_compatible(annotated.dtype, actual.dtype)
    if isinstance(annotated, ir.TupleType) and isinstance(actual, ir.TupleType):
        if len(annotated.types) != len(actual.types):
            return False
        return all(_types_compatible(a, b) for a, b in zip(annotated.types, actual.types))
    return False


def _check_type_compatible(annotated: ir.Type, actual: ir.Type, *, what: str, name: str, span) -> None:
    """Raise ParserTypeError if ``annotated`` is not compatible with ``actual``."""
    if not _types_compatible(annotated, actual):
        raise ParserTypeError(
            f"{what} '{name}' annotated as {annotated} but called/returned with {actual}",
            span=span,
            hint="Make the annotation match the actual argument/return type, or remove it.",
        )


def _validate_simt_parameters(func: ir.Function, role: str) -> None:
    """Validate the common native CCE ABI accepted by SIMT functions."""
    for param in func.params:
        if not isinstance(param.type, (ir.ScalarType, ir.TensorType, ir.TileType)):
            raise ParserTypeError(
                f"{role} parameter '{param.name}' must resolve to a scalar, Tensor, or Tile value, got {param.type}",
                span=param.span,
                hint="Ptr, Tuple, and tiling parameters are not supported.",
            )
        if isinstance(param.type, ir.TensorType):
            if param.type.dtype.get_bit() < 8:
                raise ParserTypeError(
                    f"{role} Tensor parameter '{param.name}' uses an unsupported sub-byte dtype",
                    span=param.span,
                    hint="Use a byte-addressable Tensor element type such as pl.DT_UINT8 or wider.",
                )
            tensor_view = param.type.tensor_view
            if tensor_view is None or tensor_view.layout != ir.TensorLayout.ND:
                raise ParserTypeError(
                    f"{role} Tensor parameter '{param.name}' requires ND layout",
                    span=param.span,
                    hint="Use pl.Tensor[[shape], dtype] or specify pl.ND explicitly.",
                )
        if isinstance(param.type, ir.TileType):
            if len(param.type.shape) != 2 or not all(isinstance(dim, ir.ConstInt) for dim in param.type.shape):
                raise ParserTypeError(
                    f"{role} Tile parameter '{param.name}' must have a static two-dimensional shape",
                    span=param.span,
                    hint="Pass a static two-dimensional Tile created with pl.make_tile().",
                )
            if param.type.dtype.get_bit() < 8:
                raise ParserTypeError(
                    f"{role} Tile parameter '{param.name}' uses an unsupported sub-byte dtype",
                    span=param.span,
                    hint="Use a byte-addressable Tile element type such as pl.DT_UINT8 or wider.",
                )

class CallParserMixin:
    """Mixin containing call and operation parsing methods for ``ASTParser``."""

    # Kwargs whose value must be an enum, not a plain int. Writing e.g. ``phase=2``
    # or a closure ``a=2; phase=a`` is rejected — use the enum (``phase=pl.STPhase.X``)
    # or an enum variable. Covers block/tensor/system op enum kwargs and VF-op enum
    # kwargs. ``cmp_mode`` is intentionally excluded: it is a plain-int parameter of
    # block gather/cmp (not an enum kwarg), despite _VF_KWARG_ENUMS mapping it.
    _ENUM_KWARGS: frozenset = frozenset({
        # block / tensor / system op enum kwargs
        "phase",
        "atomic",
        "mode",
        "relu_pre_mode",
        "acc_to_vec_mode",
        "sync_mode",
        "core_type",
        "cache_line",
        "dst",
        "target_memory",
        # dtype-family kwargs (all resolve to a DataType at the C++ boundary)
        "dtype",
        "target_type",
        "out_dtype",
        "cmp_dtype",
        # VF-op enum kwargs (from _VF_KWARG_ENUMS, minus cmp_mode)
        "pattern",
        "merge_mode",
        "reduce_mode",
        "reduce_type",
        "pos",
        "layout",
        "round_mode",
        "saturate",
        "bin_type",
        "hist_type",
        "gather_mode",
        "part",
        "width",
        "dist",
        "data_copy_mode",
        "index_order",
    })

    # Mapping of VF kwarg names to their expected enum classes (tuple of types).
    # When a kwarg value resolves to an instance of any mapped enum, its .value
    # VF enum kwarg validation: if a kwarg is mapped to enum classes, the parser
    # passes the enum object through to ConvertKwargsDict (which extracts .value as int).
    # If a raw string is passed for a mapped kwarg, the parser raises an error.
    _VF_KWARG_ENUMS: dict[str, tuple] = {}

    # -------------------------------------------------------------------------
    # VF ops
    # -------------------------------------------------------------------------

    # VF op names are now unified to snake_case across Python API, IR, and C++
    # backend. This map is kept for potential future name aliasing; currently
    # all entries are identity (Python name == IR op name).
    _VF_OP_NAME_MAP: dict[str, str] = {}

    # --- VF assignment-form support --------------------------------------------

    # Number of destination registers at the front of the arg list for each VF op.
    # 0 = no dst (store/side-effect ops, or already return-value ops like compare).
    # 1 = single dst at args_[0].
    # 2 = two dsts at args_[0], args_[1].
    _VF_OP_DST_COUNT: dict[str, int] = {
        # 1 dst
        "add": 1,
        "sub": 1,
        "mul": 1,
        "div": 1,
        "max": 1,
        "min": 1,
        "and_": 1,
        "or_": 1,
        "xor": 1,
        "abs_sub": 1,
        "select": 1,
        "shift_left": 1,
        "shift_right": 1,
        "prelu": 1,
        "ln": 1,
        "log": 1,
        "exp": 1,
        "abs": 1,
        "not_": 1,
        "sqrt": 1,
        "relu": 1,
        "neg": 1,
        "pair_reduce_sum": 1,
        "squeeze": 1,
        "truncate": 1,
        "astype": 1,
        "log2": 1,
        "log10": 1,
        "reduce_sum": 1,
        "reduce_max": 1,
        "reduce_min": 1,
        "muls": 1,
        "adds": 1,
        "subs": 1,
        "mins": 1,
        "maxs": 1,
        "leaky_relu": 1,
        "muls_cast": 1,
        "axpy": 1,
        "exp_sub": 1,
        "mul_add_dst": 1,
        "mul_dst_add": 1,
        "pack": 1,
        "unpack": 1,
        "arange": 1,
        "unsqueeze": 1,
        "full": 1,
        "load_align": 1,
        "load": 1,
        "load_unalign": 1,
        "gather": 1,
        "move": 1,
        "eq": 1,
        "ne": 1,
        "lt": 1,
        "gt": 1,
        "le": 1,
        "ge": 1,
        "histograms": 1,
        "bit_cast": 1,
        # 2 dsts
        "interleave": 2,
        "de_interleave": 2,
        "mull": 2,
        "addc": 2,
        "subc": 2,
        # 0 dst (no assignment form)
        "store_align": 0,
        "store_unalign": 0,
        "store_unalign_post": 0,
        "scatter": 0,
        "store": 0,
        "mem_bar": 0,
        "clear_spr": 0,
        "load_unalign_pre": 0,
        "store_align_pack": 0,
        "store_align_intlv": 0,
        "store_align_pack_postupdate": 0,
    }

    # VF ops whose dst(s) are MaskReg (not RegTensor). The assignment parser
    # declares these via vf.create_mask instead of vf.reg_tensor.
    _VF_MASK_DST_OPS: frozenset[str] = frozenset(
        {
            "eq",
            "ne",
            "lt",
            "gt",
            "le",
            "ge",
        }
    )

    # Ops that support both RegTensor and MaskReg dsts. When the dst is freshly
    # declared, its register kind is inferred from the source operands: if any
    # source argument is a known MaskReg variable, the dst is declared as
    # MaskReg; otherwise RegTensor.
    _VF_UNIFIED_OPS: frozenset[str] = frozenset(
        {
            "move",
            "and_",
            "or_",
            "xor",
            "not_",
            "select",
            "pack",
            "unpack",
            "interleave",
            "de_interleave",
            "load_align",
            "store_align",
            "store_unalign",
        }
    )

    # VF ops whose return value is a MaskReg (used by _parse_name_assignment to
    # track MaskReg variables for unified-op dst inference).
    _VF_MASK_PRODUCING_OPS: frozenset[str] = frozenset(
        {
            "create_mask",
            "update_mask",
            "get_mask_spr",
            "mask_gen_with_reg_tensor",
        }
    )

    @staticmethod
    def _extract_op_name(func: ast.expr) -> str | None:
        """Extract a normalized op name from an attribute-access call node.

        pl.tensor.add  -> "tensor.add"
        pl.add_scalar  -> "add_scalar"
        vf.add         -> "vf.add"  (vf prefix retained)
        bare_name      -> None      (no module prefix)
        """
        attrs: list[str] = []
        node: ast.expr = func
        while isinstance(node, ast.Attribute):
            attrs.insert(0, node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            attrs.insert(0, node.id)
        if len(attrs) < 2:
            return None
        if attrs[0] == "vf":
            return ".".join(attrs)
        return ".".join(attrs[1:])

    @staticmethod
    def _needs_return_lowering(func_def: ast.FunctionDef) -> bool:
        """Check whether a helper's returns have to be lowered onto a merge variable.

        A lone return that is the helper's final top-level statement does not: there is only one
        exit, so the body runs as-is and the return expression is parsed at the call site. The
        call then yields whatever that expression evaluates to, including Python-level values
        such as tile groups, which have no IR representation to merge in the first place.

        Every other shape -- an early return, a return nested in control flow, several returns,
        or none at all -- goes through `_InlineReturnLowerer`.
        """
        body = [stmt for stmt in func_def.body if not CallParserMixin._is_docstring(stmt)]
        returns = [node for node in ast.walk(func_def) if isinstance(node, ast.Return)]
        return not (len(returns) == 1 and bool(body) and body[-1] is returns[0])

    @staticmethod
    def _is_docstring(stmt: ast.stmt) -> bool:
        """Check if an AST statement is a docstring (string constant expression)."""
        return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)

    @staticmethod
    def _is_vf_op_call(call_node: ast.expr) -> str | None:
        """Check if an AST node is a ``vf.xxx(...)`` call.

        Returns the VF op name (e.g. ``"add"``) if yes, ``None`` otherwise.
        """
        if not isinstance(call_node, ast.Call):
            return None
        op_name = CallParserMixin._extract_op_name(call_node.func)
        if op_name is None or not op_name.startswith("vf."):
            return None
        return op_name[3:]  # strip "vf." prefix

    @staticmethod
    def _inline_param_list(func_def: ast.FunctionDef) -> list[ast.arg]:
        return list(func_def.args.args)

    @staticmethod
    def _inline_local_names(func_def: ast.FunctionDef, params: list[ast.arg]) -> set[str]:
        """Collect names local to a helper body, including nested block bindings."""
        names = {param.arg for param in params}

        class _Collector(ast.NodeVisitor):
            def generic_visit(self, node):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    return None
                if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                    names.add(node.id)
                    return None
                return super().generic_visit(node)

        collector = _Collector()
        for stmt in func_def.body:
            collector.visit(stmt)
        return names

    @staticmethod
    def _inline_reassigned_params(func_def: ast.FunctionDef, params: list[ast.arg]) -> set[str]:
        """Return helper parameters that are rebound by its body.

        A directly inlined parameter normally aliases the caller expression.  If
        the helper assigns to it, however, it needs its own IR definition before
        entering control flow so ConvertToSSA can carry it through branches and
        loops.
        """
        param_names = {param.arg for param in params}
        assigned: set[str] = set()

        class _Collector(ast.NodeVisitor):
            def generic_visit(self, node):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    return None
                if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and node.id in param_names:
                    assigned.add(node.id)
                    return None
                return super().generic_visit(node)

        collector = _Collector()
        for stmt in func_def.body:
            collector.visit(stmt)
        return assigned

    @staticmethod
    def _has_unsupported_inline_params(args: ast.arguments) -> bool:
        if args.posonlyargs or args.vararg is not None:
            return True
        if args.kwarg is not None or args.kwonlyargs:
            return True
        return bool(args.kw_defaults)

    # -------------------------------------------------------------------------
    # Mutex dedup helpers (shared by _emit_auto_mutex and _emit_vf_func_mutex_lock)
    # -------------------------------------------------------------------------

    @staticmethod
    def _group_refs_by_mutex_overlap(refs: list) -> list:
        """Group tilerefs by mutex_ids overlap (connected components via union-find).

        Two refs whose mutex_ids lists have any common value are in the same group.
        Returns a list of groups, each group is a list of _MutexRef.
        """
        n = len(refs)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Build mutex_id sets for each ref
        id_sets = [set(ref.mutex_ids) if ref.mutex_ids else set() for ref in refs]

        # Union refs that have overlapping mutex_ids
        for i in range(n):
            for j in range(i + 1, n):
                if id_sets[i] & id_sets[j]:
                    union(i, j)

        # Collect groups
        groups: dict[int, list] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(refs[i])
        return list(groups.values())

    @classmethod
    def _make_call_with_return_type(
        cls,
        op: ir.Op,
        args: list[ir.Expr],
        return_types: list[ir.Type],
        span: ir.Span,
    ) -> ir.Expr:
        """Create an ir.Call, attaching the return type when known.

        Args:
            op: Op identifying the callee
            args: Parsed argument expressions
            return_types: The callee's return type list (may be empty)
            span: Source span for the call
        """
        if not return_types:
            return ir.Call(op, args, span)
        if len(return_types) == 1:
            return ir.Call(op, args, return_types[0], span)
        return ir.Call(op, args, ir.TupleType(return_types), span)

    @classmethod
    def _retrieve_function_source(
        cls,
        func_name: str,
        fn: Callable,
        span: ir.Span,
        decorator_hint: str,
    ) -> tuple[str, list[str], int, int, ast.FunctionDef]:
        """Retrieve source, parse AST, and locate FunctionDef for a callable.

        Returns (source_file, source_lines, line_offset, col_offset, func_def).
        """
        import textwrap as _tw

        from .decorator import _get_source_info

        try:
            source_file, source_lines_raw, starting_line = _get_source_info(fn, "function")
        except Exception as e:
            raise UnsupportedFeatureError(
                f"Cannot compile '{func_name}': unable to retrieve source -{e}",
                span=span,
                hint=f"Define '{func_name}' in a .py file, or use {decorator_hint}",
            ) from e

        source_code = _tw.dedent("".join(source_lines_raw))
        col_offset = len(source_lines_raw[0]) - len(source_lines_raw[0].lstrip()) if source_lines_raw else 0
        line_offset = starting_line - 1
        source_lines = source_code.split("\n")

        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            raise UnsupportedFeatureError(
                f"Cannot parse '{func_name}': {e}",
                span=span,
                hint=f"Use {decorator_hint} to explicitly mark '{func_name}'",
            ) from e

        func_def = next(
            (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fn.__name__),
            None,
        )
        if func_def is None:
            raise UnsupportedFeatureError(
                f"Cannot find function definition for '{func_name}' in source",
                span=span,
                hint=f"Use {decorator_hint} to explicitly mark '{func_name}'",
            )

        return source_file, source_lines, line_offset, col_offset, func_def

    @classmethod
    def _build_function_closure(cls, fn: Callable) -> dict[str, Any]:
        """Build closure dict from a callable's globals and free variables."""
        fn_closure: dict[str, Any] = {**fn.__globals__}
        if fn.__closure__ and fn.__code__.co_freevars:
            fn_closure.update(dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in fn.__closure__))))
        return fn_closure

    @classmethod
    def _resolve_auto_mutex_pipe(cls, op_name: str, tilerefs: list):
        """Determine the pipe for auto_mutex from op_name and tile memory spaces."""
        from ._op_pipeline import get_move_pipe, get_op_pipe, get_store_pipe

        if op_name == "simt.launch":
            return ir.PipeType.V
        if op_name == "move":
            dst_mem = tilerefs[0].memory if tilerefs[0] else None
            src_mem = tilerefs[1].memory if len(tilerefs) > 1 and tilerefs[1] else None
            if src_mem is not None and dst_mem is not None:
                return get_move_pipe(src_mem, dst_mem)
            return None
        if op_name in ("store", "store_tile"):
            src_mem = tilerefs[1].memory if len(tilerefs) > 1 and tilerefs[1] else None
            if src_mem is not None:
                return get_store_pipe(src_mem)
            return None
        return get_op_pipe(op_name)

    @classmethod
    def _init_vf_kwarg_enums(cls):
        if cls._VF_KWARG_ENUMS:
            return
        from pypto.ir import (
            BinType,
            CastLayout,
            CompareMode,
            DataCopyMode,
            DuplicatePos,
            HistType,
            IndexOrder,
            LoadDist,
            MaskPattern,
            MaskWidth,
            MemBarMode,
            MergeMode,
            PackPart,
            ReduceMode,
            SaturateMode,
            SqueezeMode,
            StoreDist,
            VFRoundMode,
        )

        cls._VF_KWARG_ENUMS = {
            "pattern": (MaskPattern,),
            "mode": (MergeMode, MemBarMode),
            "merge_mode": (MergeMode,),
            "reduce_mode": (ReduceMode,),
            "reduce_type": (ReduceMode,),
            "cmp_mode": (CompareMode,),
            "pos": (DuplicatePos,),
            "layout": (CastLayout,),
            "round_mode": (VFRoundMode,),
            "saturate": (SaturateMode,),
            "bin_type": (BinType,),
            "hist_type": (HistType,),
            "gather_mode": (SqueezeMode,),
            "part": (PackPart,),
            "width": (MaskWidth,),
            "dist": (LoadDist, StoreDist),
            "data_copy_mode": (DataCopyMode,),
            "index_order": (IndexOrder,),
        }

    @classmethod
    def _get_vf_op_dst_count(cls, op_name: str) -> int | None:
        """Return the number of dst registers for a VF op, or None if unknown."""
        return cls._VF_OP_DST_COUNT.get(op_name)

    @classmethod
    def _is_vf_mask_dst_op(cls, op_name: str) -> bool:
        """Return True if the VF op's dst(s) are MaskReg (not RegTensor)."""
        return op_name in cls._VF_MASK_DST_OPS

    def parse_call(self, call: ast.Call) -> Any:
        """Parse function call.

        Args:
            call: Call AST node

        Returns:
            IR expression from call
        """
        func = call.func

        # Handle cross-function calls via self.method_name() in @pl.program classes
        if isinstance(func, ast.Attribute):
            # Check for self.method_name pattern
            if isinstance(func.value, ast.Name) and func.value.id == "self":
                method_name = func.attr
                if method_name in self.global_vars:
                    func_obj = self.gvar_to_func.get(method_name)
                    args = [self.parse_expression(arg) for arg in call.args]
                    span = self.span_tracker.get_span(call)

                    # Use return type from the parsed function if available
                    return_types = func_obj.return_types if func_obj else []
                    op = ir.Op(method_name)
                    return self._make_call_with_return_type(op, args, return_types, span)
                else:
                    raise UndefinedVariableError(
                        f"Function '{method_name}' not defined in program",
                        span=self.span_tracker.get_span(call),
                        hint=f"Available functions: {sorted(self.global_vars)}",
                    )

            # Handle pl.tensor.*, pl.system.*, and pl.* operation calls
            return self.parse_op_call(call)

        # Handle bare-name calls to external IR functions or inline Python callables.
        if isinstance(func, ast.Name):
            func_name = func.id

            # Builtin min/max -> route to pl.min/pl.max (scalar_ops.py)
            if func_name in _BUILTIN_TO_OP:
                op_func = _OP_REGISTRY.get(_BUILTIN_TO_OP[func_name])
                if op_func is not None:
                    return op_func(self, call)

            resolved = self.expr_evaluator.closure_vars.get(func_name)
            if callable(resolved) and not isinstance(resolved, type):
                return self._implicit_func_call(func_name, resolved, call)

        raise UnsupportedFeatureError(
            f"Unsupported function call: {ast.unparse(call)}",
            span=self.span_tracker.get_span(call),
            hint="Use pl.* operations, self.method() for cross-function calls, or call an inline Python helper by name",
        )

    def parse_op_call(self, call: ast.Call) -> Any:
        """Parse operation call like pl.tensor.create_tensor() or pl.add().

        Args:
            call: Call AST node

        Returns:
            IR expression from operation
        """
        op_name = self._extract_op_name(call.func)

        result = self._route_ir_node_method(call)
        if result is not None:
            return result

        span = self.span_tracker.get_span(call)
        if op_name is None:
            raise UnsupportedFeatureError(
                f"Unsupported operation call: {ast.unparse(call)}",
                span=span,
                hint="Use pl.*, pl.tensor.*, or pl.system.* operations",
            )

        if op_name == "constexpr":
            raise ParserSyntaxError(
                "pl.constexpr() can only be used as an 'if' condition or in a ternary expression, "
                "e.g. 'if pl.constexpr(cond):' or 'x = a if pl.constexpr(cond) else b'",
                span=span,
                hint="Use 'pl.constexpr(condition):' in an if statement or ternary expression — "
                "constexpr is not allowed in for/while/with/break/continue/return or as a standalone statement",
            )

        if self._current_func_type in (ir.FunctionType.SimtVF, ir.FunctionType.SimtCallee) and not op_name.startswith(
            "simt."
        ):
            raise UnsupportedFeatureError(
                f"Operation '{op_name}' is not supported inside a SIMT function",
                span=span,
                hint=(
                    "The current SIMT slice supports SIMT context queries, scalar expressions, restricted "
                    "loops, and scalar Tensor/Tile accesses."
                ),
            )
        if self._auto_mutex and not op_name.startswith("vf."):
            self._emit_auto_mutex(op_name, call, span)

        op_func = _OP_REGISTRY.get(op_name)
        if op_func is not None:
            return op_func(self, call)

        return self._default_op_func(op_name, call)

    def parse_op_kwargs(self, call: ast.Call) -> dict[str, Any]:
        """Parse keyword arguments for an operation call."""
        return {kw.arg: self.resolve_single_kwarg(kw.arg, kw.value) for kw in call.keywords}

    def resolve_single_kwarg(self, key: str, value: ast.expr) -> Any:
        """Resolve a single keyword argument value to a Python or IR value.

        Kwargs are resolved uniformly through ``parse_expression`` and then
        unwrapped: a constant IR value (ConstInt/ConstBool/ConstFloat) becomes a
        plain Python scalar. This is how an enum written as ``pl.RoundMode.X``
        (lowered to a ConstInt by ``parse_attribute``) turns into the int that op
        builders / the C++ boundary expect. A dtype written as ``pl.DT_FP16``
        resolves to a DataType object (``parse_attribute`` returns it as-is) and
        passes through unchanged, since dtype consumers and the C++ boundary
        expect a DataType. Other non-constant results (Var, MakeTuple, ...) also
        pass through unchanged.

        A list of constants (``order=[0, 1]``, ``mutex_ids=[VA]``) is parsed to a
        MakeTuple and then unwrapped to a Python list, since such consumers expect
        a list; a list containing an IR Var stays a MakeTuple.

        This is the lenient policy: an argument with no parse-time value keeps its
        IR, because most kwargs accept a runtime value. A position that cannot
        (an address, an axis list) uses :meth:`require_const_value` instead,
        which rejects it.

        ``key`` is retained for call-site compatibility (some callers pass it).
        """
        parsed = self.parse_expression(value)
        # An enum kwarg written directly (pl.RoundMode.X) or via an enum variable
        # parses to the enum object, not a Const*, so a Const* here means a plain
        # int/bool/float was passed — reject it for enum kwargs.
        if key in self._ENUM_KWARGS and isinstance(parsed, (ir.ConstBool, ir.ConstInt, ir.ConstFloat)):
            # Final: the int is a perfectly good Python value, so a retry would
            # accept it and lose the rejection.
            raise FinalRejectionError(
                f"'{key}' expects an enum value, not {parsed.value!r}",
                span=self.span_tracker.get_span(value),
                hint=f"Use the enum, e.g. {key}=pl.RoundMode.X / dtype=pl.DT_FP16, "
                "or an enum-valued variable",
            )
        found, python_value = ExprEvaluator.ir_to_python_value(parsed)
        if not found:
            return parsed
        return python_value

    @staticmethod
    def _kwarg_node(call: ast.Call, key: str) -> "ast.expr | None":
        """Value node of keyword *key*, or None when the call does not pass it."""
        return next((kw.value for kw in call.keywords if kw.arg == key), None)

    def require_const_value(
        self, parsed: Any, node: ast.expr, *, expects: str, hint: str, check=None, key: str | None = None
    ) -> Any:
        """Python value of an expression that has to be known while parsing.

        The single compile-time accessor: take the parse-time value of *parsed*,
        and report one consistent diagnostic when it has none or fails the caller's
        *check*. Callers add only what is specific to them — what the position
        expects and how to fix it.

        *parsed* is the caller's own ``parse_expression(node)``: a handler that
        parsed its arguments up front already holds it, and re-parsing to reach the
        check is not free — ``parse_expression`` can emit IR and falls back to
        evaluating the expression in Python. *node* is kept only for the
        diagnostic's text and span, so it must be the node *parsed* came from.
        """
        found, value = ExprEvaluator.ir_to_python_value(parsed)
        if not found or (check is not None and not check(value)):
            subject = f"'{key}'" if key else f"'{ast.unparse(node)}'"
            detail = f", got '{ast.unparse(node)}'" if key else ""
            # Final for the same reason: *check* rejects values Python evaluation
            # would happily produce.
            raise FinalRejectionError(
                f"{subject} must be a compile-time {expects}{detail}",
                span=self.span_tracker.get_span(node),
                hint=hint,
            )
        return value

    def _default_op_func(self, op_name: str, call: ast.Call) -> ir.Expr:
        if op_name.startswith("vf."):
            return self._parse_vf_op(op_name[3:], call)
        return self._parse_block_default(op_name, call)

    def _parse_external_function_call(self, _local_name: str, ext_func: ir.Function, call: ast.Call) -> ir.Expr:
        """Parse a call to an externally-defined ir.Function.

        Args:
            _local_name: The name used in the caller's scope (may be aliased)
            ext_func: The external ir.Function object
            call: The AST Call node
        """
        func_name = ext_func.name
        span = self.span_tracker.get_span(call)

        # Validate no naming conflict with internal program functions
        if func_name in self.global_vars:
            raise ParserSyntaxError(
                f"External function '{func_name}' conflicts with program function '{func_name}'",
                span=span,
                hint="Rename either the external or program function to avoid the name conflict",
            )

        # Check for conflicting externals with same .name but different objects
        if func_name in self.external_funcs and self.external_funcs[func_name] is not ext_func:
            raise ParserSyntaxError(
                f"Conflicting external functions with name '{func_name}'",
                span=span,
                hint="External functions must have unique names; rename one of the functions",
            )

        # Track the external function
        self.external_funcs[func_name] = ext_func

        args = [self.parse_expression(arg) for arg in call.args]
        op = ir.Op(func_name)
        return self._make_call_with_return_type(op, args, ext_func.return_types, span)

    def _parse_simt_template_call(self, local_name: str, fn: Callable, call: ast.Call) -> ir.Expr:
        """Instantiate and call one helper @pl.simt.function template."""
        span = self.span_tracker.get_span(call)
        if call.keywords or any(isinstance(arg, ast.Starred) for arg in call.args):
            raise ParserSyntaxError(
                "Helper @pl.simt.function calls accept positional arguments only",
                span=span,
            )
        args = [self.parse_expression(arg) for arg in call.args]
        callee = self._instantiate_simt_function(local_name, fn, args, call.args, span)
        self._validate_simt_function_arguments(callee, args, call.args, span)
        return self._make_call_with_return_type(ir.Op(callee.name), args, callee.return_types, span)

    def _instantiate_simt_function(
        self,
        local_name: str,
        fn: Callable,
        args: list[ir.Expr],
        arg_nodes: list[ast.expr],
        span,
    ) -> ir.Function:
        """Parse a marked SIMT callable once its actual argument types are known."""
        from ._ast_parser import ASTParser
        from .decorator import get_simt_max_threads

        max_threads = get_simt_max_threads(fn)
        launchable = max_threads is not None
        if launchable:
            if not _is_int(max_threads):
                raise TypeError("max_threads must be an integer")
            if not 1 <= max_threads <= 2048:
                raise ValueError("max_threads must be in the range [1, 2048]")

        cached = self.simt_func_cache.get(id(fn))
        if cached is not None:
            self._register_simt_external(cached, span)
            return cached
        if id(fn) in self.simt_call_stack:
            raise ParserSyntaxError(
                f"Recursive helper @pl.simt.function call involving '{fn.__name__}' is not supported",
                span=span,
            )

        source_file, source_lines, line_offset, col_offset, func_def = self._retrieve_function_source(
            local_name,
            fn,
            span,
            "@pl.simt.function",
        )
        func_args = func_def.args
        if (
            func_args.posonlyargs
            or func_args.vararg is not None
            or func_args.kwarg is not None
            or func_args.kwonlyargs
            or func_args.defaults
            or func_args.kw_defaults
        ):
            raise ParserSyntaxError(
                f"SIMT function '{fn.__name__}' only supports required positional parameters",
                span=span,
            )
        params = list(func_args.args)
        if len(args) != len(params):
            role = "SIMT function" if launchable else "SIMT callee"
            raise ParserTypeError(
                f"{role} '{fn.__name__}' expects {len(params)} argument(s), got {len(args)}",
                span=span,
            )

        func_type = ir.FunctionType.SimtVF if launchable else ir.FunctionType.SimtCallee
        role = "SIMT function" if launchable else "SIMT helper"
        parser = ASTParser(
            source_file,
            source_lines,
            ir.SectionKind.Vector,
            line_offset,
            col_offset,
            global_vars=self.global_vars,
            closure_vars=self._build_function_closure(fn),
            debug_info=self.debug_info,
            tilingkey_consts=self._tilingkey_consts,
            datatype_consts=self._datatype_consts,
            void_return_only=launchable,
            void_return_context="@pl.simt.function(max_threads=...)",
            allow_early_return=True,
        )
        parser.external_funcs = self.external_funcs
        parser.simt_func_cache = self.simt_func_cache
        parser.simt_call_stack = self.simt_call_stack

        self.simt_call_stack.append(id(fn))
        try:
            parsed = parser.parse_function(
                func_def,
                func_type=func_type,
                callsite_param_types={param.arg: actual.type for param, actual in zip(params, args)},
            )

            _validate_simt_parameters(parsed, role)
            if launchable:
                result = ir.Function(
                    parsed.name,
                    list(parsed.params),
                    list(parsed.return_types),
                    parsed.body,
                    parsed.span,
                    func_type,
                    parsed.entry,
                    {"max_threads": max_threads},
                )
            else:
                result = parsed
            self._register_simt_external(result, span)
            self.simt_func_cache[id(fn)] = result
        finally:
            self.simt_call_stack.pop()

        return result

    def _validate_simt_function_arguments(
        self,
        callee: ir.Function,
        args: list[ir.Expr],
        arg_nodes: list[ast.expr],
        span,
    ) -> None:
        """Validate call-site arguments against an instantiated SIMT signature."""
        if len(args) != len(callee.params):
            raise ParserTypeError(
                f"SIMT function '{callee.name}' expects {len(callee.params)} argument(s), got {len(args)}",
                span=span,
            )
        for param, actual, actual_node in zip(callee.params, args, arg_nodes):
            _check_type_compatible(
                param.type,
                actual.type,
                what="SIMT parameter",
                name=param.name,
                span=self.span_tracker.get_span(actual_node),
            )

    def _register_simt_external(self, func: ir.Function, span) -> None:
        """Register one instantiated SIMT function in the enclosing Program."""
        if func.name in self.global_vars:
            raise ParserSyntaxError(
                f"SIMT function '{func.name}' conflicts with an internal program function",
                span=span,
            )
        if func.name in self.external_funcs and self.external_funcs[func.name] is not func:
            raise ParserSyntaxError(
                f"Conflicting external functions with name '{func.name}'",
                span=span,
            )
        self.external_funcs[func.name] = func

    def _inline_template(self, func_name: str, fn: Callable, span) -> _InlineFunctionTemplate:
        template = self.inline_func_cache.get(id(fn))
        if template is not None:
            return template
        from .decorator import is_vector_function

        is_vf = is_vector_function(fn)
        source_info = self._retrieve_function_source(
            func_name,
            fn,
            span,
            "@pl.vector_function" if is_vf else "an annotated Python helper",
        )
        template = _InlineFunctionTemplate(
            func_def=source_info[4],
            closure_vars=self._build_function_closure(fn),
            is_vector_function=is_vf,
            span_tracker=SpanTracker(source_info[0], source_info[1], source_info[2], source_info[3]),
        )
        self.inline_func_cache[id(fn)] = template
        return template

    def _bind_inline_arguments(
        self,
        func_name: str,
        func_def: ast.FunctionDef,
        call: ast.Call,
        span,
    ) -> dict[str, tuple[ir.Expr, ast.expr]]:
        """Bind one positional argument to each helper parameter."""
        args = func_def.args
        if self._has_unsupported_inline_params(args):
            raise ParserSyntaxError(
                f"Inline function '{func_name}' only supports positional parameters with optional defaults",
                span=span,
            )
        if call.keywords or any(isinstance(arg, ast.Starred) for arg in call.args):
            raise ParserSyntaxError(
                f"Call to inline function '{func_name}' only supports plain positional arguments",
                span=span,
            )

        params = self._inline_param_list(func_def)
        required_count = len(params) - len(args.defaults)
        if not required_count <= len(call.args) <= len(params):
            raise ParserTypeError(
                f"Function '{func_name}' expects {required_count} to {len(params)} positional argument(s), "
                f"got {len(call.args)}",
                span=span,
            )
        argument_nodes = [*call.args, *args.defaults[len(call.args) - required_count:]]
        bound = {param.arg: (self.parse_expression(arg), arg) for param, arg in zip(params, argument_nodes)}
        for param in params:
            actual = bound[param.arg][0]
            if param.annotation is None or self.resolve_tiling_class(param.annotation):
                continue
            if self.type_resolver.annotation_has_shape_policy(param.annotation):
                self.type_resolver.validate_policy_parameter_type(param.annotation, param.arg, actual.type)
            else:
                annotated = self.type_resolver.resolve_param_type(param.annotation)
                _check_type_compatible(
                    annotated,
                    actual.type,
                    what="Parameter",
                    name=param.arg,
                    span=span,
                )
        return bound

    def _implicit_func_call(self, func_name: str, fn: Callable, call: ast.Call) -> ir.Expr | None:
        """Expand a Python helper body directly into the caller's IR builder.

        For SIMT functions, routes to ``_parse_simt_template_call`` instead of
        inlining. For regular callables, expands the body directly.
        """
        from .decorator import get_simt_max_threads, is_simt_function

        if is_simt_function(fn):
            if self._current_func_type in (ir.FunctionType.SimtVF, ir.FunctionType.SimtCallee):
                if get_simt_max_threads(fn) is not None:
                    raise ParserTypeError(
                        f"SIMT helper '{func_name}' must be decorated with @pl.simt.function without max_threads",
                        span=self.span_tracker.get_span(call),
                    )
                return self._parse_simt_template_call(func_name, fn, call)
            raise ParserSyntaxError(
                f"@pl.simt.function '{func_name}' must be invoked through pl.simt.launch()",
                span=self.span_tracker.get_span(call),
            )

        span = self.span_tracker.get_span(call)
        template = self._inline_template(func_name, fn, span)
        if self.inline_vf_depth != 0 and not template.is_vector_function:
            raise ParserSyntaxError(
                f"Vector function cannot call non-vector inline function '{func_name}'",
                span=span,
            )
        if id(fn) in self.inline_call_stack:
            raise ParserSyntaxError(
                f"Recursive inline function call detected for '{func_name}'",
                span=span,
            )

        old_closure = self.expr_evaluator.closure_vars
        old_const_env = self.const_env
        self.expr_evaluator.closure_vars = {**template.closure_vars, **old_closure}
        try:
            bound = self._bind_inline_arguments(func_name, template.func_def, call, span)
            params = self._inline_param_list(template.func_def)
            inline_id = self.inline_counter
            self.inline_counter += 1
            prefix = f"__inline_{inline_id}_"
            return_val_name = f"{prefix}return_val"
            returned_name = f"{prefix}returned"
            local_names = self._inline_local_names(template.func_def, params)
            reassigned_params = self._inline_reassigned_params(template.func_def, params)
            func_def = copy.deepcopy(template.func_def)
            renamer = _InlineLocalRenamer(local_names, prefix)
            body = [renamer.visit(stmt) for stmt in func_def.body if not self._is_docstring(stmt)]
            needs_return_lowering = not template.is_vector_function and self._needs_return_lowering(func_def)
            trailing_return = None
            if needs_return_lowering:
                body = _InlineReturnLowerer(return_val_name, returned_name).lower(body)
            else:
                ast.fix_missing_locations(func_def)
                if not template.is_vector_function and body and isinstance(body[-1], ast.Return):
                    trailing_return = body.pop()

            locked_vf_refs: list = []
            if template.is_vector_function and self._auto_mutex:
                arg_nodes = [bound[param.arg][1] for param in params]
                locked_vf_refs = self._emit_vf_func_mutex_lock(
                    [param.arg for param in params],
                    [self._try_resolve_tileref(arg) for arg in arg_nodes],
                    {param.arg for param in params},
                    span,
                )

            self.inline_call_stack.append(id(fn))
            is_outermost_vf = template.is_vector_function and self.inline_vf_depth == 0
            if template.is_vector_function:
                self.inline_vf_depth += 1
            self.scope_manager.enter_scope("inline")
            self.const_env = dict(self.const_env)

            old_span_tracker = self.span_tracker
            self.span_tracker = template.span_tracker

            try:
                for param in params:
                    expr, _arg_node = bound[param.arg]
                    renamed_name = f"{prefix}{param.arg}"
                    # A parameter that the helper rebinds must first become a
                    # real local IR value.  Binding it directly to the caller
                    # expression leaves no version for SSA to merge when the
                    # assignment is nested in an if/while.
                    if param.arg in reassigned_params:
                        value = self.builder.let(renamed_name, expr, span=span)
                        self._transfer_tile_sync_metadata(value, expr)
                    else:
                        value = expr
                    self.scope_manager.define_var(renamed_name, value, allow_redef=True)
                    self._update_const_env(renamed_name, expr)

                if is_outermost_vf:
                    with self.builder.section(ir.SectionKind.VF, span):
                        self.scope_manager.enter_scope("section")
                        try:
                            for stmt in body:
                                self.parse_statement(stmt)
                        finally:
                            self.scope_manager.exit_scope(leak_vars=False)
                else:
                    for stmt in body:
                        self.parse_statement(stmt)
                if template.is_vector_function:
                    return None
                if needs_return_lowering:
                    return self.scope_manager.lookup_var_bounded(return_val_name)
                if trailing_return is not None and trailing_return.value is not None:
                    return self.parse_expression(trailing_return.value)
                return None
            finally:
                self.span_tracker = old_span_tracker
                self.scope_manager.exit_scope(leak_vars=False)
                self.inline_call_stack.pop()
                if template.is_vector_function:
                    self.inline_vf_depth -= 1
                self.const_env = old_const_env
                if locked_vf_refs:
                    self._emit_inline_vf_mutex_unlock(locked_vf_refs, span)
        finally:
            self.expr_evaluator.closure_vars = old_closure

    # -------------------------------------------------------------------------
    # Keyword argument resolution helpers
    # -------------------------------------------------------------------------

    def _parse_vf_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse a VF API operation call: vf.{op_name}(...).

        VF ops directly emit VF instructions. Arguments and kwargs are passed
        through to ir.create_op_call with the "vf." prefix.

        Note: auto_mutex is NOT applied per-vf-op here, because bisheng requires
        VEC_SCOPE to contain only VF instructions (get_buf/rls_buf are plain CCE
        intrinsics and cause "Do not know how to expand this operator's operand"
        errors if emitted inside the scope). Instead, auto_mutex wraps the whole
        VF inline-function call at its call site -see _parse_inline_call.

        Args:
            op_name: Name of the VF operation (without ``vf.`` prefix).
            call: Call AST node.

        Returns:
            IR expression for the VF op call.
        """
        span = self.span_tracker.get_span(call)

        # Reject the legacy statement form `vf.xxx(dst, ...)` for ops that produce
        # a result. Only store/side-effect ops (dst_count == 0) may be called as a
        # bare statement; all others must use the assignment form `dst = vf.xxx(...)`.
        # (The assignment form is handled in _assignment_parser, which never routes
        # through here — so reaching this point means it's a statement-form call.)
        dst_count = self._VF_OP_DST_COUNT.get(op_name)
        if dst_count is not None and dst_count > 0 and op_name != "bit_cast":
            if dst_count == 1:
                correct = f"dst = vf.{op_name}(...)"
            else:
                dst_list = ", ".join(f"dst{i}" for i in range(dst_count))
                correct = f"{dst_list} = vf.{op_name}(...)"
            raise ParserSyntaxError(
                f"vf.{op_name} produces a result and must use the assignment form. "
                f"The statement form vf.{op_name}(dst, ...) is no longer supported.",
                span=span,
                hint=f"Use: {correct}",
            )

        # Block direct use of vf.reg_tensor / vf.mask_reg — registers are now
        # declared implicitly by the assignment form (dst = vf.xxx(...)); users
        # cannot call them directly.
        if op_name in ("reg_tensor", "mask_reg"):
            raise ParserSyntaxError(
                f"vf.{op_name} cannot be called directly. VF registers are declared "
                "automatically by the assignment form.",
                span=span,
                hint="Use: dst = vf.load_align(...)  # or any VF compute op",
            )

        args = [self.parse_expression(arg) for arg in call.args]
        kwargs = self.parse_op_kwargs(call)

        return ir.create_op_call(f"vf.{op_name}", args, kwargs, span)

    # --- auto_mutex helpers ---------------------------------------------------

    def _try_resolve_tileref(self, node: ast.expr):
        """Resolve a tile argument to its memory and optional mutex metadata.

        Returns _MutexRef(buf_ids, mutex_ids, memory, tile_id) for tile
        arguments. Ordinary tiles have empty ``buf_ids`` and ``mutex_ids``;
        their memory is still needed to infer the move pipe when paired with a
        tile-group tile.

        Only positional arguments are scanned by auto-mutex. Subscript
        arguments (``tile[off]``, ``buf[idx]``) alias the base's
        buffer; their mutex metadata is propagated onto the GetItemExpr by
        ``parse_subscript``, so the generic ``parse_expression`` path finds it.
        """
        expr = self.parse_expression(node)
        if not isinstance(expr, ir.Expr):
            return None
        if not isinstance(expr.type, ir.TileType) or not expr.type.memref:
            return None
        mem = expr.type.memref.memory_space_
        meta = self.tile_mutex_lock_meta(expr)
        if meta is None:
            return _MutexRef((), (), mem, id(expr))
        buf_id_irs, mutex_ids = meta
        return _MutexRef(buf_id_irs, mutex_ids, mem, id(expr))

    def _emit_auto_mutex(self, op_name: str, call: ast.Call, span: ir.Span):
        """Emit mutex_lock before and mutex_unlock after a block DSL op.

        Scans positional arguments for tile-group tiles, determines the op pipe,
        and emits lock/unlock per unique slot.
        Returns None -the caller still parses the op normally.

        Phase-aware skip on Acc tiles: when matmul/matmul_acc/store carries
        phase="partial"/"final", the cube/fixp handshake on the Acc-memory
        accumulator is taken over by the hardware unit_flag bit
        (AccPhase/STPhase). The software mutex on the Acc buf is redundant
        *and* occupies M-pipe / FIX-pipe instruction slots that otherwise
        let cube/fixp run back-to-back. Skip it.

        Non-Acc tiles (L0A / L0B / L1) MUST keep their mutex: unit_flag does
        NOT cover the MTE1 -> cube path. Removing L0A/L0B mutex causes RAW
        between MTE1 finishing the move and cube starting to read
        (verified: device error 507015 on 2026-05-29).
        """

        from ._op_pipeline import op_accesses_buffer

        # Descriptor-only ops (e.g. set_validshape) rewrite tile metadata but never
        # touch buffer data, so their accesses cannot race -> no buffer mutex needed.
        if not op_accesses_buffer(op_name):
            return

        # 1. Build unique_refs: scan args for slot.tile mutex refs, dedup by slot,
        #    then drop Acc tiles when a phase-aware matmul/store carries the
        #    unit_flag (the hardware handshake replaces the software mutex there).
        if op_name == "simt.launch":
            args_node = next((kw.value for kw in call.keywords if kw.arg == "args"), None)
            if not isinstance(args_node, ast.Tuple):
                return
            scan_args = args_node.elts
        else:
            scan_args = call.args
        tilerefs = [self._try_resolve_tileref(arg) for arg in scan_args]
        unique_refs = []
        seen = set()
        for tref in tilerefs:
            if tref is None or not tref.buf_ids:
                continue
            if tref.slot_id in seen:
                continue
            seen.add(tref.slot_id)
            unique_refs.append(tref)

        if op_name in ("matmul", "matmul_acc", "matmul_mx", "matmul_mx_acc", "store", "store_tile"):
            phase = None
            for kw in call.keywords:
                if kw.arg == "phase":
                    # phase is written as pl.STPhase.X / pl.AccPhase.X, which
                    # resolve_single_kwarg returns as the enum object.
                    phase = self.resolve_single_kwarg("phase", kw.value)
                    break
            # STPhase/AccPhase enum: Partial/Final indicates multi-step accumulation.
            # phase is written as an enum in the DSL (pl.STPhase.X / pl.AccPhase.X),
            # which the parser lowers to the enum object, so match the objects.
            _phase_skip = {
                ir.STPhase.Partial,
                ir.STPhase.Final,
                ir.AccPhase.Partial,
                ir.AccPhase.Final,
            }
            if phase in _phase_skip:
                unique_refs = [r for r in unique_refs if r.memory != ir.MemorySpace.Acc]

        if not unique_refs:
            return

        # 2. Determine pipe
        pipe = self._resolve_auto_mutex_pipe(op_name, tilerefs)
        if pipe is None:
            return

        # 3. Emit lock for each unique _TileRef, with dedup for aliasing tiles.
        # Group once here and reuse the grouping at unlock time.
        groups = self._group_refs_by_mutex_overlap(unique_refs)
        self._emit_mutex_for_groups(groups, pipe, span, is_lock=True)

        # Store grouping for post-op unlock emission (avoids re-grouping)
        self._pending_mutex_unlocks = (groups, pipe, span)

    def _emit_auto_mutex_unlocks(self):
        """Emit mutex_unlock calls queued by _emit_auto_mutex or _parse_func_call."""
        if not hasattr(self, "_pending_mutex_unlocks") or self._pending_mutex_unlocks is None:
            return
        groups, pipe, span = self._pending_mutex_unlocks
        self._emit_mutex_for_groups(groups, pipe, span, is_lock=False)
        self._pending_mutex_unlocks = None

    def _emit_mutex_for_tile(self, tile: ir.Expr, pipe, span: ir.Span, *, is_lock: bool) -> bool:
        """Emit all mutex IDs for one tile in input order and report whether any were emitted."""
        meta = self.tile_mutex_lock_meta(tile)
        if meta is None:
            return False
        buf_ids, mutex_ids = meta
        self._emit_mutex_op(
            buf_ids,
            mutex_ids,
            [0] * len(buf_ids),
            pipe,
            span,
            is_lock=is_lock,
        )
        return True

    def _emit_mutex_op(
        self, buf_ids, mutex_ids, mutex_id_owner_indices, pipe, span: ir.Span, *, is_lock: bool
    ) -> None:
        """Emit static IDs directly or a per-tile-aware dynamic dedup operation."""
        from pypto_pro.ir._utils import _normalize_expr
        from pypto_pro.ir.op.system_ops import _create_mutex_dedup_op, mutex_lock, mutex_unlock

        op_name = "system.mutex_lock" if is_lock else "system.mutex_unlock"
        emit_plain = mutex_lock if is_lock else mutex_unlock
        id_exprs = list(buf_ids)

        if all(isinstance(buf_id, ir.ConstInt) for buf_id in id_exprs):
            unique_ids = list(dict.fromkeys(int(buf_id.value) for buf_id in id_exprs))
            for mutex_id in unique_ids:
                expr = emit_plain(pipe=pipe, mutex_id=mutex_id, span=span)
                self.builder.emit(ir.EvalStmt(expr, span))
            return

        if len(id_exprs) == 1:
            expr = emit_plain(pipe=pipe, mutex_id=id_exprs[0], span=span)
        else:
            expr = _create_mutex_dedup_op(
                op_name,
                pipe=pipe,
                mutex_id_exprs=[_normalize_expr(buf_id, span) for buf_id in id_exprs],
                mutex_id_owner_indices=mutex_id_owner_indices,
                mutex_ids_union=list(mutex_ids or ()),
                auto_mutex=self._auto_mutex,
                span=span,
            )
        self.builder.emit(ir.EvalStmt(expr, span))

    def _emit_mutex_for_groups(self, groups: list, pipe, span: ir.Span, *, is_lock: bool):
        """Emit mutex lock/unlock calls for pre-grouped refs, with dedup if-guards.

        ``groups`` is the output of _group_refs_by_mutex_overlap (computed once at
        lock time and reused at unlock time). Shared by lock (is_lock=True) and unlock
        (is_lock=False). Static IDs are stable-deduplicated and emitted individually;
        dynamic IDs share one mutex_(un)lock_dyn whose CCE codegen guards aliases
        across Tiles while trusting the frontend's per-tile uniqueness validation.
        Lock and unlock visit independent groups and IDs in the same order.
        """
        for group in groups:
            id_exprs = [
                buf_id
                for tref in group
                for buf_id in tref.buf_ids
            ]
            mutex_id_owner_indices = [
                owner_index
                for owner_index, tref in enumerate(group)
                for _ in tref.buf_ids
            ]
            ids_union = list(
                dict.fromkeys(
                    mutex_id
                    for tref in group
                    for mutex_id in (tref.mutex_ids or ())
                )
            )
            self._emit_mutex_op(
                id_exprs,
                ids_union,
                mutex_id_owner_indices,
                pipe,
                span,
                is_lock=is_lock,
            )

    def _emit_vf_func_mutex_lock(
        self,
        param_names: list,
        arg_tilerefs: list,
        used_params: set,
        span: ir.Span,
    ) -> list:
        """Emit mutex_lock(V, buf_id) for tile-valued args whose matching
        parameter is referenced inside a ``@pl.vector_function`` body.

        Called before a VF func.call is emitted (i.e., before the VEC_SCOPE
        is generated). Returns the ref grouping (from _group_refs_by_mutex_overlap)
        so the caller can emit matching unlocks after the call without re-grouping.
        """
        unique_refs = []
        seen = set()
        for param_name, tref in zip(param_names, arg_tilerefs):
            if tref is None or not tref.buf_ids:
                continue
            if param_name not in used_params:
                continue
            if tref.slot_id in seen:
                continue
            seen.add(tref.slot_id)
            unique_refs.append(tref)

        groups = self._group_refs_by_mutex_overlap(unique_refs)
        self._emit_mutex_for_groups(groups, ir.PipeType.V, span, is_lock=True)
        return groups

    def _emit_inline_vf_mutex_unlock(self, groups: list, span: ir.Span) -> None:
        """Emit mutex_unlock(V, buf_id) for each group from _emit_vf_func_mutex_lock."""
        self._emit_mutex_for_groups(groups, ir.PipeType.V, span, is_lock=False)

    # -------------------------------------------------------------------------
    # Block default handler and helpers
    # -------------------------------------------------------------------------

    def _parse_block_default(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Handle block DSL ops not registered in _OP_REGISTRY."""
        span = self.span_tracker.get_span(call)

        args = [self.parse_expression(arg) for arg in call.args]
        kwargs = self.parse_op_kwargs(call)

        # first arg is out; keep out as first arg to match pto-isa convention
        return ir.create_op_call(block_ir_op(op_name), args, kwargs, span)
