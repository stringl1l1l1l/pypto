# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""make_tile_group parsing helpers for ASTParser.

``pl.make_tile_group(type=, addrs=, mutex_ids=, depth=)`` declares a rotating group of
tiles inside a kernel. The parser lowers it to an IR named-tuple handle with
fields ``tiles`` (tuple of tile vars), one or more ``mutex_ids`` column tuples,
and a mutable ``cursor`` struct. The handle's ``next()/current()/previous()`` methods
and its ``group[i]`` subscript return a bare tile (not a slot); the tile <->
mutex_id mapping, when configured, is kept as parser metadata
(``_tile_mutex_meta``) and consumed by auto_mutex, so the runtime never handles
mutex ids or manual lock/unlock.

``next()/current()/previous()`` walk the rotating cursor; ``group[i]`` addresses
a slot directly and leaves the cursor alone, which is what lets a kernel keep
several slots in flight at once (``load(g[k + 1])`` while ``matmul(g[k])``).
Both produce the same metadata, so auto_mutex cannot tell them apart.
"""

from __future__ import annotations

import ast
import warnings
from functools import reduce

from pypto.pypto_impl import ir
from pypto.pypto_impl.ir import DataType
from pypto_pro.ir._utils import _is_int
from pypto_pro.ir.op._op_registry import op_impl
from pypto_pro.ir.op.block_ops import TileType as _TileType
from pypto_pro.ir.op.block_ops import make_tile_expr as _make_tile_expr
from pypto_pro.ir.op.block_ops import tile_slot_size as _tile_slot_size

from .diagnostics import ParserSyntaxError, ParserTypeError


def _is_int_sequence(value) -> bool:
    """Whether *value* is a non-empty tuple of plain integers."""
    return isinstance(value, tuple) and bool(value) and all(_is_int(e) for e in value)


class BufferParserMixin:
    """make_tile_group DSL composite: parse pl.make_tile_group(...) into an IR named tuple."""

    # --- helpers --------------------------------------------------------------

    @staticmethod
    def _normalize_mutex_ids(mutex_ids, span: ir.Span) -> tuple[tuple[int, ...], ...]:
        """Normalize mutex IDs to one fixed-size tuple per tile slot."""
        per_tile_mutex_ids = []
        for mutex_id in mutex_ids:
            if isinstance(mutex_id, (list, tuple)):
                tile_mutex_ids = tuple(mutex_id)
                if not tile_mutex_ids:
                    raise ParserTypeError(
                        "make_tile_group() mutex ID sequence for a tile must not be empty", span=span
                    )
            else:
                tile_mutex_ids = (mutex_id,)

            for value in tile_mutex_ids:
                if not isinstance(value, int) or value < 0 or value > 31:
                    raise ParserTypeError(f"mutex_ids must be ints in [0,31], got {value!r}", span=span)
            if len(set(tile_mutex_ids)) != len(tile_mutex_ids):
                raise ParserTypeError(
                    f"make_tile_group() mutex IDs for one tile must not contain duplicates: {tile_mutex_ids}",
                    span=span,
                )
            per_tile_mutex_ids.append(tile_mutex_ids)

        expected_mutex_id_count = len(per_tile_mutex_ids[0])
        for tile_mutex_ids in per_tile_mutex_ids[1:]:
            if len(tile_mutex_ids) != expected_mutex_id_count:
                raise ParserTypeError(
                    f"all tiles in a tile group must have the same mutex ID count: "
                    f"{expected_mutex_id_count} and {len(tile_mutex_ids)}",
                    span=span,
                )
        return tuple(per_tile_mutex_ids)

    @staticmethod
    def _validate_depth(depth, span: ir.Span) -> None:
        if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
            raise ParserTypeError(
                f"make_tile_group() depth must be a positive compile-time integer, got {depth!r}",
                span=span,
            )

    @staticmethod
    def _tile_type_slot_size(tile_type: _TileType) -> int:
        """Per-tile byte size derived from the TileType static shape and dtype.

        Same helper make_tile() uses for its default ``size``, so a group slot is
        exactly as wide as a standalone tile of the same TileType.
        """
        try:
            return _tile_slot_size(tile_type.shape, tile_type.dtype)
        except ValueError as exc:
            raise ParserTypeError(f"make_tile_group() {exc}") from exc

    def is_tile_group(self, expr) -> bool:
        return "tiles" in self.named_fields(expr)


    # --- factory --------------------------------------------------------------
    @op_impl("make_tile_group")
    def _parse_make_tile_group(self, call: ast.Call) -> ir.Expr:
        span = self.span_tracker.get_span(call)
        if call.args:
            raise ParserSyntaxError(
                "pl.make_tile_group() takes keyword args only "
                "(type=, addrs=, mutex_ids=, depth=)",
                span=span,
            )
        kw = {k.arg: k for k in call.keywords if k.arg is not None}
        for required in ("type", "addrs"):
            if required not in kw:
                raise ParserTypeError(f"pl.make_tile_group() missing required keyword '{required}'", span=span)

        tile_type = self.parse_expression(kw["type"].value)
        if not isinstance(tile_type, _TileType):
            raise ParserTypeError("pl.make_tile_group() 'type' must be a pl.TileType", span=span)

        raw_mutex_ids = self.expr_evaluator.eval_expr(kw["mutex_ids"].value) if "mutex_ids" in kw else None
        if raw_mutex_ids is not None and not isinstance(raw_mutex_ids, (list, tuple)):
            raise ParserTypeError(
                "make_tile_group() mutex_ids must be a list, tuple, or None",
                span=span,
            )
        mutex_ids = self._normalize_mutex_ids(raw_mutex_ids, span) if raw_mutex_ids else None

        depth = self.expr_evaluator.eval_expr(kw["depth"].value) if "depth" in kw else None
        if depth is not None:
            self._validate_depth(depth, span)

        if mutex_ids is None:
            if depth is None:
                raise ParserTypeError(
                    "make_tile_group() depth is required when mutex_ids is None or empty",
                    span=span,
                )
        elif depth is None:
            depth = len(mutex_ids)
        elif len(mutex_ids) != depth:
            raise ParserTypeError(
                f"make_tile_group() mutex_ids length {len(mutex_ids)} must equal depth {depth}",
                span=span,
            )

        if not self._auto_mutex:
            warnings.warn(
                "make_tile_group() mutex_ids specified but auto_mutex is not enabled; "
                "mutex_ids will not be used for automatic synchronization. "
                "Enable auto_mutex via @pl.kernel(auto_mutex=True) to use them.",
                stacklevel=2,
            )

        slot_size = self._tile_type_slot_size(tile_type)
        # addrs: a single base address -> contiguous tiles (base + i*slot_size);
        # a list -> one explicit address per tile (non-contiguous).
        addrs_node = kw["addrs"].value
        addrs = self.require_const_value(
            self.parse_expression(addrs_node),
            addrs_node,
            key="addrs",
            expects="integer, or list of integers",
            check=lambda value: _is_int(value) or _is_int_sequence(value),
            hint="addrs is fixed while parsing; pass a literal, or a variable bound to "
            "literals/constants — not a runtime value such as a tensor shape or loop index",
        )
        if isinstance(addrs, (list, tuple)):
            tile_addrs = list(addrs)
            if len(tile_addrs) != depth:
                raise ParserTypeError(
                    f"make_tile_group() addrs length {len(tile_addrs)} must equal "
                    f"depth/len(mutex_ids) {depth}",
                    span=span,
                )
        else:
            tile_addrs = [addrs + i * slot_size for i in range(depth)]

        return self._build_tile_group_ir(tile_type, tile_addrs, depth, mutex_ids, slot_size, span)

    def _build_tile_group_ir(
        self, tile_type, tile_addrs, depth, mutex_ids, slot_size, span: ir.Span
    ) -> ir.Expr:
        """Build the IR named-tuple handle for a tile group.

        Multi-tile groups carry a rotating ``cursor`` struct ({tiles, mutex_ids,
        cursor}). It starts at the last slot so that next(), which advances before
        selecting, returns slot 0 on its first call. Single-tile groups skip it
        ({tiles, mutex_ids}) since the only tile is statically determined.

        Writes group metadata directly to ``tile_group_meta`` keyed by the
        returned expression; ``_transfer_tile_sync_metadata`` re-keys it to
        the assigned variable after builder.let.
        """
        var_name = self.current_target_name
        per_tile_mutex_ids = mutex_ids or ()
        tile_vars = []
        for i, addr in enumerate(tile_addrs):
            t = _make_tile_expr(
                shape=tile_type.shape,
                dtype=tile_type.dtype,
                target_memory=tile_type.target_memory,
                layout=tile_type.layout,
                fractal=tile_type.fractal,
                pad=tile_type.pad,
                compact=tile_type.compact,
                valid_shape=tile_type.valid_shape,
                addr=addr,
                size=slot_size,
            )
            tile_vars.append(self.builder.let(f"_tg_{var_name}_tiles_{i}", t, span=span))
        tiles_tuple = self.builder.let(f"_tg_{var_name}_tiles", ir.MakeTuple(tile_vars, span), span=span)
        mutex_columns = tuple(zip(*per_tile_mutex_ids)) if per_tile_mutex_ids else ((),)
        mut_tuples = []
        mut_fields = []
        for column, values in enumerate(mutex_columns):
            suffix = "" if column == 0 else f"_{column}"
            mut_tuples.append(
                self.builder.let(
                    f"_tg_{var_name}_mutex_ids{suffix}",
                    ir.MakeTuple([ir.ConstInt(m, DataType.INDEX, span) for m in values], span),
                    span=span,
                )
            )
            mut_fields.append(f"mutex_ids{suffix}")
        if depth == 1:
            result = self.make_named_tuple([tiles_tuple, *mut_tuples], ["tiles", *mut_fields], span)
            self.tile_group_meta[result] = (depth, mutex_ids, tile_type.target_memory)
            return result
        cursor_call = ir.create_op_call(
            "struct.create",
            [ir.ConstInt(depth - 1, DataType.INDEX, span)],
            {"name": "_TileGroupCursor", "fields": ["cursor"]},
            span,
        )
        self.register_tuple_name(cursor_call, "_TileGroupCursor")
        self.register_tuple_fields(cursor_call, ["cursor"])
        cursor = self.builder.let(f"_tg_{var_name}_cursor", cursor_call, span=span)
        result = self.make_named_tuple(
            [tiles_tuple, *mut_tuples, cursor], ["tiles", *mut_fields, "cursor"], span
        )
        self.tile_group_meta[result] = (depth, mutex_ids, tile_type.target_memory)
        return result

    # --- accessors ------------------------------------------------------------

    def _emit_cursor_advance(self, cursor_struct, new_val, span: ir.Span) -> None:
        self.builder.emit(
            ir.EvalStmt(ir.create_op_call("struct.set", [cursor_struct, new_val], {"field": "cursor"}, span), span)
        )

    def _group_meta(self, group_var, span: ir.Span) -> tuple:
        """Return the slot count, per-tile mutex IDs, and candidate IDs."""
        meta = self.tile_group_meta.get(group_var)
        if meta is None:
            raise ParserTypeError(
                "tile group handle has no group metadata; it must come directly from "
                "pl.make_tile_group()",
                span=span,
                hint="Pass the handle itself rather than a copy stored in a struct or container",
            )
        per_tile_mutex_ids = meta[1]
        candidate_ids = tuple(
            dict.fromkeys(value for tile_mutex_ids in (per_tile_mutex_ids or ()) for value in tile_mutex_ids)
        )
        return meta[0], per_tile_mutex_ids, candidate_ids

    def _select_static_slot(
        self, tiles, slot: int, per_tile_mutex_ids, candidate_ids, span: ir.Span
    ) -> ir.Expr:
        """Select tiles[slot] with a compile-time index.

        When mutex metadata exists, attach the slot's static id; it needs no
        ``% n`` and no dynamic lock at lock time.
        """
        tile_ir = ir.GetItemExpr(tiles, ir.ConstInt(slot, DataType.INDEX, span), span)
        if per_tile_mutex_ids is not None:
            mutex_id_irs = tuple(
                ir.ConstInt(value, DataType.INDEX, span) for value in per_tile_mutex_ids[slot]
            )
            self._tile_mutex_meta[tile_ir] = (mutex_id_irs, candidate_ids)
        return tile_ir

    def _select_dynamic_slot(
        self,
        group_var,
        tiles,
        raw_index,
        per_tile_mutex_ids,
        candidate_ids,
        span: ir.Span,
    ):
        """Select a slot with one materialized runtime index.

        The supplied index is used unchanged. When mutex metadata exists, the
        buf_id is ``mutex_ids[idx]``, an Expr, so auto_mutex takes the dynamic
        lock path.
        """
        idx = self.builder.let(
            f"_bufidx_{self._tuple_idx_counter}",
            raw_index,
            span=span,
        )
        self._tuple_idx_counter += 1
        tile_ir = ir.GetItemExpr(tiles, idx, span)
        if per_tile_mutex_ids is not None:
            mutex_id_irs = []
            for column in range(len(per_tile_mutex_ids[0])):
                suffix = "" if column == 0 else f"_{column}"
                mut = self.lower_attr_access(group_var, f"mutex_ids{suffix}", span)
                mutex_id_irs.append(ir.GetItemExpr(mut, idx, span))
            self._tile_mutex_meta[tile_ir] = (tuple(mutex_id_irs), candidate_ids)
        return tile_ir

    def _lower_group_accessor(self, group_var, method_name: str, span: ir.Span):
        """Lower group.next()/current()/previous() to a bare tile.

        next() advances the cursor (+1) then returns the tile at the new index;
        current()/previous() leave the cursor unchanged. When configured, records
        the selected tile's mutex id via _tile_mutex_meta for auto_mutex.
        """
        n_slots, per_tile_mutex_ids, candidate_ids = self._group_meta(group_var, span)
        tiles = self.lower_attr_access(group_var, "tiles", span)
        if n_slots == 1:
            # Single tile: no cursor or modulo; optional mutex metadata stays static.
            return self._select_static_slot(tiles, 0, per_tile_mutex_ids, candidate_ids, span)
        cursor_struct = self.lower_attr_access(group_var, "cursor", span)
        cur_read = self.lower_attr_access(cursor_struct, "cursor", span)
        if method_name == "next":
            new_val = cur_read + ir.ConstInt(1, DataType.INDEX, span)
            self._emit_cursor_advance(cursor_struct, new_val, span)
            raw = self.lower_attr_access(cursor_struct, "cursor", span)
        elif method_name == "current":
            raw = cur_read
        else:  # previous
            raw = cur_read + ir.ConstInt(n_slots - 1, DataType.INDEX, span)
        wrapped = raw % ir.ConstInt(n_slots, DataType.INDEX, span)
        return self._select_dynamic_slot(
            group_var,
            tiles,
            wrapped,
            per_tile_mutex_ids,
            candidate_ids,
            span,
        )

    def lower_group_subscript(self, group_var, slice_node: ast.expr, span: ir.Span) -> ir.Expr:
        """Lower ``group[i]`` to a bare tile — direct slot access, cursor untouched.

        Emits the same optional mutex metadata as next()/current()/previous(). A
        constant index resolves the slot and, when configured, its mutex id at parse
        time; a runtime index is used unchanged and selects a dynamic mutex id only
        when the group has mutex metadata. The caller must keep it in ``[0, n)``.

        The cursor is deliberately neither read nor advanced: subscript is random
        access, so interleaving it with next()/previous() leaves the rotation
        sequence exactly where the last next() put it.
        """
        n_slots, per_tile_mutex_ids, candidate_ids = self._group_meta(group_var, span)
        tiles = self.lower_attr_access(group_var, "tiles", span)
        index_expr = self._reject_non_index_subscript(slice_node, span)

        if isinstance(index_expr, ir.ConstInt):
            slot = int(index_expr.value)
            if not 0 <= slot < n_slots:
                raise ParserTypeError(
                    f"tile group index {index_expr.value} is out of range for a "
                    f"{n_slots}-tile group",
                    span=span,
                    hint=f"Valid indices are in [0, {n_slots})",
                )
            return self._select_static_slot(tiles, slot, per_tile_mutex_ids, candidate_ids, span)

        return self._select_dynamic_slot(
            group_var,
            tiles,
            index_expr,
            per_tile_mutex_ids,
            candidate_ids,
            span,
        )

    def _reject_non_index_subscript(self, slice_node: ast.expr, span: ir.Span) -> ir.Expr:
        """Parse and validate one integer tile-group index."""
        index_expr = self.parse_expression(slice_node)
        index_type = getattr(index_expr, "type", None)
        if not (isinstance(index_type, ir.ScalarType) and index_type.dtype.is_int()):
            actual_type = index_type if index_type is not None else type(index_expr).__name__
            raise ParserTypeError(
                f"tile group index must be an integer scalar, but got {actual_type}",
                span=span,
                hint="Use an integer scalar expression, for example g[0] or g[i]",
            )
        return index_expr
