# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tile groups contain bound tiles and an optional rotating cursor.

Mutex IDs are operands of each internal make_mutex_tile call. Runtime selection selects
the tile descriptor and its IDs together.
"""


from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import IntEnum, auto

from pypto.pypto_impl import ir
from pypto.pypto_impl.ir import DataType
from pypto_pro.ir._utils import (
    _is_none,
    _to_make_tuple,
    require_const_int,
    require_const_int_list,
    require_list,
)
from pypto_pro.ir.op._op_registry import op_impl
from pypto_pro.ir.op.block_ops import TileType as _TileType
from pypto_pro.ir.op.block_ops import _ir_reinterpret
from pypto_pro.ir.op.block_ops import make_mutex_tile_expr as _make_mutex_tile_expr
from pypto_pro.ir.op.block_ops import tile_slot_size as _tile_slot_size
from pypto_pro.ir.op.system_ops import MAX_MUTEX_ID

from ..._errors import (
    InvalidArgument,
    InvalidOperation,
    InvalidShape,
    InvalidType,
    InvalidVal,
    OutOfRange,
    PyptoProError,
    message_of,
)
from ._tuple_type_registry import TupleTypeKind

_TILE_GROUP_TYPE_NAME = "tile_group"


@dataclass(frozen=True)
class TileGroupFacts:
    """One pl.make_tile_group() declaration's resolved facts."""

    memory: ir.MemorySpace
    depth: int  # slot count, inferred from len(mutex_ids) when the source omits depth
    addrs: tuple[int, ...]  # one start address per slot
    slot_size: int
    fwd_ids: tuple[int, ...] | None  # None when the keyword is absent
    bwd_ids: tuple[int, ...] | None
    span: ir.Span
    site: tuple[int, int] | None  # (lineno, col_offset) of the kernel-body statement or call parsing it


class SelectorKind(IntEnum):

    NEXT = auto()
    CURRENT = auto()
    PREVIOUS = auto()
    INDEX = auto()



class BufferParserMixin:
    """make_tile_group DSL composite: parse pl.make_tile_group(...) into an IR named tuple."""

    # --- helpers --------------------------------------------------------------

    @staticmethod
    def _normalize_mutex_ids(mutex_ids) -> tuple[tuple[int, ...], ...]:
        """Check already parsed mutex ID groups for a consistent per-slot layout."""
        for tile_mutex_ids in mutex_ids:
            if not tile_mutex_ids:
                raise InvalidVal(
                    "make_tile_group() mutex ID sequence for a tile must not be empty",
                    parser_retry=True,
                )
            if len(set(tile_mutex_ids)) != len(tile_mutex_ids):
                raise InvalidVal(
                    f"make_tile_group() mutex IDs for one tile must not contain duplicates: {tile_mutex_ids}",
                    parser_retry=True,
                )

        expected_mutex_id_count = len(mutex_ids[0])
        for tile_mutex_ids in mutex_ids[1:]:
            if len(tile_mutex_ids) != expected_mutex_id_count:
                raise InvalidArgument(
                    f"all tiles in a tile group must have the same mutex ID count: "
                    f"{expected_mutex_id_count} and {len(tile_mutex_ids)}",
                    parser_retry=True,
                )
        return mutex_ids


    def _parse_mutex_ids_and_depth(
        self,
        raw_mutex_ids,
        depth: int | None,
        span: ir.Span,
    ) -> tuple[ir.MakeTuple | None, int]:
        """Validate and lower one tile group's mutex IDs, inferring depth when omitted."""

        if not self._auto_mutex and (
            raw_mutex_ids == "auto" or isinstance(raw_mutex_ids, (list, tuple)) and raw_mutex_ids
        ):
            raise InvalidOperation(
                "Configuring mutex_ids requires @pl.jit(auto_mutex=True)",
                parser_retry=True,
            )

        if isinstance(raw_mutex_ids, str):
            if raw_mutex_ids != "auto":
                raise InvalidArgument(
                    'make_tile_group() string mutex_ids only supports "auto", '
                    f"got {raw_mutex_ids!r}",
                    parser_retry=True,
                )
            if depth is None:
                raise InvalidArgument(
                    'make_tile_group() depth is required when mutex_ids is "auto", None, or empty',
                    parser_retry=True,
                )
            auto_mutex_ids = []
            for _ in range(depth):
                placeholder = ir.Var(
                    f"__pypto_auto_mutex_id_{self._auto_mutex_placeholder_counter}",
                    ir.ScalarType(DataType.INT64),
                    span,
                )
                self._auto_mutex_placeholder_counter += 1
                auto_mutex_ids.append(_to_make_tuple([placeholder], span))
            return _to_make_tuple(auto_mutex_ids, span), depth

        if raw_mutex_ids is None or (
            isinstance(raw_mutex_ids, (list, tuple)) and not raw_mutex_ids
        ):
            if depth is None:
                raise InvalidArgument(
                    'make_tile_group() depth is required when mutex_ids is "auto", None, or empty',
                    parser_retry=True,
                )
            return None, depth

        normalized_mutex_ids = self._normalize_mutex_ids(raw_mutex_ids)
        if depth is None:
            depth = len(normalized_mutex_ids)
        elif len(normalized_mutex_ids) != depth:
            raise InvalidShape(
                f"make_tile_group() mutex_ids length {len(normalized_mutex_ids)} must equal depth {depth}",
                parser_retry=True,
            )
        return (
            _to_make_tuple(
                [_to_make_tuple(tile_mutex_ids, span) for tile_mutex_ids in normalized_mutex_ids],
                span,
            ),
            depth,
        )

    @staticmethod
    def _tile_type_slot_size(tile_type: _TileType) -> int:
        """Per-tile byte size derived from the TileType static shape and dtype.

        Same helper make_tile() uses for its byte span, so a group slot is
        exactly as wide as a standalone tile of the same TileType.
        """
        try:
            return _tile_slot_size(tile_type.shape, tile_type.dtype)
        except PyptoProError as exc:
            raise InvalidVal(f"make_tile_group() {message_of(exc)}", parser_retry=True) from exc

    def is_tile_group_type(self, value_type: ir.Type) -> bool:
        if not isinstance(value_type, ir.TupleType):
            return False
        info = self.classify_tuple_type(value_type)
        return info.kind == TupleTypeKind.NAMED_TUPLE and info.name == _TILE_GROUP_TYPE_NAME

    def is_tile_group(self, expr) -> bool:
        return isinstance(expr, ir.Expr) and self.is_tile_group_type(expr.type)

    def _contains_tile_group_type(self, value_type: ir.Type) -> bool:
        if self.is_tile_group_type(value_type):
            return True
        return isinstance(value_type, ir.TupleType) and any(
            self._contains_tile_group_type(element_type) for element_type in value_type.types
        )

    # --- factory --------------------------------------------------------------
    @op_impl("make_tile_group")
    def _build_make_tile_group(
        self,
        *,
        type,
        addrs,
        mutex_ids=None,
        depth=None,
        bwd_ids=None,
        fwd_ids=None,
        span: ir.Span,
    ) -> ir.Expr:
        tile_type = type
        group_target = (
            ir.SectionKind.Vector
            if tile_type.target_memory == ir.MemorySpace.Vec
            else ir.SectionKind.Cube
        )
        if _is_none(mutex_ids):
            mutex_ids = None
        elif not isinstance(mutex_ids, str):
            check_bounds = group_target == self.target and self._auto_mutex
            mutex_id_groups = []
            rows = require_list(mutex_ids, subject="mutex_ids")
            for index, row in enumerate(rows):
                subject = f"mutex_ids[{index}]"
                if isinstance(row.type, ir.TupleType):
                    ids = require_const_int_list(
                        row, subject=subject, lower_bound=0 if check_bounds else None,
                        upper_bound=MAX_MUTEX_ID if check_bounds else None,
                    )
                else:
                    ids = (require_const_int(
                        row, subject=subject, lower_bound=0 if check_bounds else None,
                        upper_bound=MAX_MUTEX_ID if check_bounds else None,
                    ),)
                mutex_id_groups.append(tuple(ids))
            mutex_ids = tuple(mutex_id_groups)
        depth = None if _is_none(depth) else require_const_int(
            depth, subject="depth", lower_bound=1,
        )

        if group_target == self.target:
            mutex_id_irs, depth = self._parse_mutex_ids_and_depth(
                mutex_ids,
                depth,
                span,
            )
        else:
            mutex_id_irs = None
            if depth is not None:
                if isinstance(mutex_ids, (list, tuple)) and mutex_ids and len(mutex_ids) != depth:
                    raise InvalidShape(
                        f"make_tile_group() mutex_ids length {len(mutex_ids)} must equal depth {depth}",
                        parser_retry=True,
                    )
            elif isinstance(mutex_ids, (list, tuple)) and mutex_ids:
                depth = len(mutex_ids)
            else:
                raise InvalidArgument(
                    'make_tile_group() depth is required when mutex_ids is "auto", None, or empty',
                    parser_retry=True,
                )

        slot_size = self._tile_type_slot_size(tile_type)
        # addrs: a single base address -> contiguous tiles (base + i*slot_size);
        # a list -> one explicit address per tile (non-contiguous).
        try:
            if isinstance(addrs.type, ir.TupleType):
                addrs = require_const_int_list(
                    addrs, subject="addrs", lower_bound=0, min_length=1,
                )
            else:
                addrs = require_const_int(
                    addrs, subject="addrs", lower_bound=0,
                )
        except InvalidType as error:
            error.hint = (
                "addrs is fixed while parsing; pass a literal, or a variable bound to "
                "literals/constants — not a runtime value such as a tensor shape or loop index"
            )
            raise
        if isinstance(addrs, list):
            tile_addrs = list(addrs)
            if len(tile_addrs) != depth:
                raise InvalidShape(
                    f"make_tile_group() addrs length {len(tile_addrs)} must equal "
                    f"depth/len(mutex_ids) {depth}",
                    parser_retry=True,
                )
        else:
            tile_addrs = [addrs + i * slot_size for i in range(depth)]

        # fwd_ids/bwd_ids take no part in the IR; they are resolved into the
        # TileGroupFacts below, which the automatic pipeline transform reads.
        decl_ids: dict[str, tuple[int, ...]] = {}
        for ids_key, ids_value in (("fwd_ids", fwd_ids), ("bwd_ids", bwd_ids)):
            if ids_value is None:
                continue
            ids = require_const_int_list(ids_value, subject=ids_key, min_length=1)
            decl_ids[ids_key] = tuple(ids)
        result = self._build_tile_group_ir(
            tile_type,
            tile_addrs,
            depth,
            mutex_id_irs,
            slot_size,
            span,
        )
        self.parse_facts.buffer_decls[result] = TileGroupFacts(
            memory=tile_type.target_memory,
            depth=depth,
            addrs=tuple(tile_addrs),
            slot_size=slot_size,
            fwd_ids=decl_ids.get("fwd_ids"),
            bwd_ids=decl_ids.get("bwd_ids"),
            span=span,
            site=self._access_site,
        )
        return result


    def _build_tile_group_ir(
        self, tile_type, tile_addrs, depth, mutex_id_irs, slot_size, span: ir.Span
    ) -> ir.Expr:
        """Build bound tile elements and preserve the existing rotating cursor."""
        var_name = self.current_target_name
        tile_vars = []
        for i, addr in enumerate(tile_addrs):
            ids = mutex_id_irs.elements[i] if mutex_id_irs is not None else ()
            tile = _make_mutex_tile_expr(
                shape=tile_type.shape, dtype=tile_type.dtype,
                target_memory=tile_type.target_memory, layout=tile_type.layout,
                fractal=tile_type.fractal, pad=tile_type.pad, compact=tile_type.compact,
                valid_shape=tile_type.valid_shape, addr=addr, size=slot_size,
                mutex_ids=ids, span=span,
            )
            self._record_tile_high_water(tile)
            value = self.builder.let(f"_tg_{var_name}_tiles_{i}", tile, span=span)
            tile_vars.append(value)
        tiles = ir.MakeTuple(tile_vars, span)
        tiles_var = self.builder.let(f"_tg_{var_name}_tiles", tiles, span=span)
        elements, fields = [tiles_var], ["tiles"]
        if depth > 1:
            cursor_call = ir.create_op_call(
                "struct.create", [ir.ConstInt(depth - 1, DataType.INT64, span)],
                {"name": "_TileGroupCursor", "fields": ["cursor"]}, span,
            )
            self.register_struct_type(cursor_call, "_TileGroupCursor", ["cursor"])
            elements.append(self.builder.let(f"_tg_{var_name}_cursor", cursor_call, span=span))
            fields.append("cursor")
        result = self.make_named_tuple(elements, fields, span, name=_TILE_GROUP_TYPE_NAME)
        self.tile_group_meta[result] = (depth, mutex_id_irs, tile_type.target_memory)
        return result

    def _build_reinterpreted_group(self, group, kwargs: dict, span: ir.Span) -> ir.Expr:
        """Create independent descriptors while retaining the source cursor."""
        depth, mutex_ids, memory = self._group_meta(group)
        tiles = self.lower_attr_access(group, "tiles", span)
        values = []
        for i in range(depth):
            source = ir.GetItemExpr(tiles, ir.ConstInt(i, DataType.INT64, span), span)
            call = _ir_reinterpret(source, **kwargs, span=span)
            value = self.builder.let(f"_tg_{self.current_target_name}_tiles_{i}", call, span=span)
            values.append(value)
        tiles = ir.MakeTuple(values, span)
        tiles_var = self.builder.let(f"_tg_{self.current_target_name}_tiles", tiles, span=span)
        elements, fields = [tiles_var], ["tiles"]
        if depth > 1:
            elements.append(self.lower_attr_access(group, "cursor", span))
            fields.append("cursor")
        result = self.make_named_tuple(elements, fields, span, name=_TILE_GROUP_TYPE_NAME)
        self.tile_group_meta[result] = (depth, mutex_ids, memory)
        self._bind_buffer_ref(result, group, None)
        return result

    def _emit_cursor_advance(self, cursor_struct, new_val, span: ir.Span) -> None:
        self.builder.emit(
            ir.AssignStmt(
                [], ir.create_op_call("struct.set", [cursor_struct, new_val], {"field": "cursor"}, span), span
            )
        )

    def _group_meta(self, group_var) -> tuple:
        meta = self.tile_group_meta.get(group_var)
        if meta is None:
            raise InvalidType(
                "tile group handle has no group metadata; it must come directly from pl.make_tile_group()",
                parser_retry=True,
            )
        return meta

    def _select_static_slot(self, tiles, slot: int, span: ir.Span) -> ir.Expr:
        return ir.GetItemExpr(tiles, ir.ConstInt(slot, DataType.INT64, span), span)

    def _select_dynamic_slot(self, tiles, raw_index, span: ir.Span) -> ir.Expr:
        """Capture the index before a later cursor update."""
        idx = self.builder.let(f"_bufidx_{self._tuple_idx_counter}", raw_index, span=span)
        self._tuple_idx_counter += 1
        return ir.GetItemExpr(tiles, idx, span)

    def _lower_group_accessor(self, group_var, method_name: str, span: ir.Span):
        """Select a complete bound tile using the rotating cursor."""
        selector = self._new_slot_selector(SelectorKind[method_name.upper()])
        depth, _mutex_ids, _memory = self._group_meta(group_var)
        tiles = self.lower_attr_access(group_var, "tiles", span)
        if depth == 1:
            # Single tile: no cursor or modulo; optional mutex metadata stays static.
            tile = self._select_static_slot(tiles, 0, span)
            self._bind_buffer_ref(tile, group_var, 0, selector)
            return tile
        cursor_struct = self.lower_attr_access(group_var, "cursor", span)
        cur_read = self.lower_attr_access(cursor_struct, "cursor", span)
        if method_name == "next":
            new_val = cur_read + ir.ConstInt(1, DataType.INT64, span)
            self._emit_cursor_advance(cursor_struct, new_val, span)
            raw = self.lower_attr_access(cursor_struct, "cursor", span)
        elif method_name == "current":
            raw = cur_read
        else:  # previous
            raw = cur_read + ir.ConstInt(depth - 1, DataType.INT64, span)
        wrapped = raw % ir.ConstInt(depth, DataType.INT64, span)
        tile = self._select_dynamic_slot(
            tiles,
            wrapped,
            span,
        )
        self._bind_buffer_ref(tile, group_var, None, selector)
        return tile

    def lower_group_subscript(self, group_var, slice_node: ast.expr, span: ir.Span) -> ir.Expr:
        """Select a complete tile without advancing or wrapping the index."""
        depth, _mutex_ids, _memory = self._group_meta(group_var)
        tiles = self.lower_attr_access(group_var, "tiles", span)
        index_expr = self._reject_non_index_subscript(slice_node)

        if isinstance(index_expr, ir.ConstInt):
            slot = int(index_expr.value)
            if not 0 <= slot < depth:
                raise OutOfRange(
                    f"tile group index {index_expr.value} is out of range for a "
                    f"{depth}-tile group",
                    hint=f"Valid indices are in [0, {depth})",
                    parser_retry=True,
                )
            tile = self._select_static_slot(tiles, slot, span)
            selector = self._new_slot_selector(SelectorKind.INDEX, slot=slot, index_node=slice_node)
            self._bind_buffer_ref(tile, group_var, slot, selector)
            return tile

        tile = self._select_dynamic_slot(
            tiles,
            index_expr,
            span,
        )
        selector = self._new_slot_selector(SelectorKind.INDEX, index_node=slice_node)
        self._bind_buffer_ref(tile, group_var, None, selector)
        return tile

    def _reject_non_index_subscript(self, slice_node: ast.expr) -> ir.Expr:
        """Parse and validate one integer tile-group index."""
        index_expr = self.parse_expression(slice_node)
        index_type = getattr(index_expr, "type", None)
        if not (isinstance(index_type, ir.ScalarType) and index_type.dtype.is_int()):
            actual_type = index_type if index_type is not None else type(index_expr).__name__
            raise InvalidType(
                f"tile group index must be an integer scalar, but got {actual_type}",
                hint="Use an integer scalar expression, for example g[0] or g[i]",
                parser_retry=True,
            )
        return index_expr
