# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""What the preload pipeline's auto-sync needs to know about buffers and the accesses to them.

Both come from the probe parse (see _analyzer.probe_kernel_facts), not from the kernel's AST:
which declarations are cross-core (those carrying fwd_ids/bwd_ids), how many slots each
rotates through and over which addresses; and, per op a stage emitted, which declaration every
operand reaches. What is left here is the verdict on each of those — the role (R/W/RW) an
op's arguments carry and the pipe the op runs on, read off the tables below.

The result drives automatic wait/set_cross_core insertion at stage boundaries.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import functools
import inspect
from typing import TYPE_CHECKING, NamedTuple, Union

from pypto.pypto_impl.backend import BackendCCE
from pypto.pypto_impl.ir import MemorySpace, PipeType
from pypto_pro.language import _api as _language_api

from ..._errors import (
    InvalidArgument,
    InvalidOperation,
    InvalidVal,
    NotSupported,
    OutOfRange,
    span_of,
)
from ._astutil import buffer_decl_location, loop_body_sites, stmt_header

if TYPE_CHECKING:
    from pypto_pro.ir.op.ptr_ops import TensorFacts
    from pypto_pro.language.parser._buffer_parser import TileGroupFacts

    # One buffer declaration as the parse resolved it, whichever call declared it.
    BufferFacts = Union[TileGroupFacts, TensorFacts]

# Highest usable cross-core event id: a hardware limit. Shared with _sync_graph, which
# allocates from the same pool for address-reuse edges.
MAX_EVENT_ID = 15

# The memory spaces a cross-core tile group may live in: UB and L1. GM is the third
# cross-core space, declared with make_tensor instead of make_tile_group.
_CROSS_CORE_TILE_MEMORY = frozenset({MemorySpace.Vec, MemorySpace.Mat})


@dataclass
class CrossCoreBuffer:
    """A cross-core NBuffer's configuration."""

    facts: BufferFacts  # the declaration as the parse resolved it
    fwd_ids_node: ast.expr | None  # AST node for cross_core_forward_id value (e.g. a Name)
    bwd_ids_node: ast.expr | None  # AST node for cross_core_backward_id value
    # The ids themselves. Kept rather than only counted: the address-reuse allocator needs
    # to know which ids are already spoken for.
    fwd_ids: tuple[int, ...] = ()
    bwd_ids: tuple[int, ...] = ()

    @property
    def fwd_id_count(self) -> int:
        """How many ids the forward direction declares. NOT the buffer's slot count: a
        buffer may share one id across all its slots (see _validate._check_event_id_counts).
        This is the modulus of the id pick, `ids[task % id_count]`, while the slots turn
        over on `task % slot_count`."""
        return len(self.fwd_ids)

    @property
    def bwd_id_count(self) -> int:
        """How many ids the backward direction declares — see fwd_id_count."""
        return len(self.bwd_ids)


@dataclass
class CrossCoreSyncContext:
    """Buffer declarations and memory layout the sync graph is built from.

    Populated by _analyzer._resolve_kernel_buffers() during analysis.
    """

    # Buffer declarations: name -> CrossCoreBuffer (only cross-core buffers with fwd/bwd ids)
    buffers: dict = field(default_factory=dict)
    # Lifted literal fwd_ids/bwd_ids: [(var_name, ast_literal, buffer_name)], each declared
    # with the pipeline loop that uses that buffer.
    lifted_ids: list = field(default_factory=list)
    # Event-id groups the framework allocated for address reuse: [(var_name, ast_literal)].
    # Kernel-level so the NAMES run continuously across loops; which entries belong to which
    # loop is recorded on that loop's SyncPlan, and the ids inside may repeat between loops.
    reuse_ids: list = field(default_factory=list)
    # Per-buffer slot address ranges: name -> (memory, [(start, end), ...] per slot).
    # Covers ALL buffers (cross-core + local) for address-overlap detection.
    addr_ranges: dict = field(default_factory=dict)
    # buffer name -> static TileGroup depth. AUTO uses this before IDs are planned.
    slot_counts: dict = field(default_factory=dict)
    # buffer name -> the statement declaring it, so an error about the buffer can point
    # at where the user wrote it.
    decl_spans: dict = field(default_factory=dict)
    # Address-overlapping buffer pairs: [(buf_a, buf_b), ...] (same memory, ranges intersect)
    addr_overlaps: list = field(default_factory=list)
    # Address-reuse sync is derived from the graph on demand, not stored here — see
    # _sync_graph.allocate_reuse_ids.


@dataclass(frozen=True)
class AccessTables:
    """The kernel-level tables each recorded op is resolved against.

    Every field answers the same question: *the parse says this operand reaches a declaration
    named X — does the sync graph follow X, and on which pipe is this op touching it?*
    Resolved once for the whole kernel and only read afterwards, so they travel as one value.
    """

    # Cross-core buffer declarations: name -> CrossCoreBuffer. Membership decides whether
    # an access is worth recording at all.
    cross_buffers: dict
    # name -> MemorySpace for ALL buffers, cross-core or local. A move's pipe follows the
    # memory spaces of both its tiles, so the local side has to resolve too.
    all_buffer_memory: dict
    # Every declared tile group's name, so a call-site argument can be recognised as a group.
    group_names: set
    # Buffers whose address range overlaps another buffer's, recorded even when not
    # cross-core themselves: a write over shared memory still has to be ordered. NOT the
    # sync graph's "region", which is every buffer's memory area, reuse or not.
    addr_shared: set
    # The kernel's enclosing scope, for resolving what a callee name stands for.
    closure_vars: dict

    def tracks(self, buf: str | None) -> bool:
        """True if the sync graph follows this buffer: cross-core, or sharing an address
        with one. Everything else is the user's own, and auto_mutex orders it."""
        return buf is not None and (buf in self.cross_buffers or buf in self.addr_shared)


def detect_addr_overlaps(addr_ranges: dict, cross_core_names: set, decl_spans: dict | None = None) -> list:
    """Detect address-overlapping buffer pairs relevant to cross-core sync.

    Two buffers overlap if they share the same MemorySpace and any of their slot
    ranges intersect. Only pairs where **at least one side is a cross-core buffer**
    are reported — overlaps between two local buffers (e.g. multiple views of the
    same UB region like p_f16_db / p_f16_main_db) are the user's / auto_mutex's
    concern, not the pipeline cross-core sync's. Returns (buf_a, buf_b) pairs
    (names sorted).

    Constraint: only pairwise overlap is supported. If any buffer overlaps with
    more than one other buffer (counting only cross-core-relevant overlaps),
    raises ValueError.
    """
    names = sorted(addr_ranges.keys())
    overlaps = []
    # buffer -> set of buffers it overlaps with (for the 3+ check)
    overlap_partners: dict = {}

    def _ranges_intersect(ra, rb) -> bool:
        for sa, ea in ra:
            for sb, eb in rb:
                if sa < eb and sb < ea:  # half-open interval intersection
                    return True
        return False

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            # Skip if neither is a cross-core buffer (local-local overlap is not
            # a pipeline sync concern).
            if a not in cross_core_names and b not in cross_core_names:
                continue
            mem_a, ranges_a = addr_ranges[a]
            mem_b, ranges_b = addr_ranges[b]
            if mem_a != mem_b:
                continue
            if _ranges_intersect(ranges_a, ranges_b):
                # Slot count must be the same for overlapping buffers (precise
                # per-slot overlap tracking is not yet supported).
                if len(ranges_a) != len(ranges_b):
                    raise InvalidArgument(
                        f"pipeline: address-overlapping buffers '{a}' and '{b}' have "
                        f"different slot counts ({len(ranges_a)} vs {len(ranges_b)}). "
                        f"Overlapping buffers must have the same number of slots.",
                        span=(decl_spans or {}).get(a),
                    )
                overlaps.append((a, b))
                overlap_partners.setdefault(a, set()).add(b)
                overlap_partners.setdefault(b, set()).add(a)

    for buf, partners in overlap_partners.items():
        if len(partners) > 1:
            raise NotSupported(
                f"pipeline: buffer '{buf}' has overlapping addresses with multiple "
                f"buffers {sorted(partners)}. Only pairwise address overlap is "
                f"supported (at most 2 buffers may share a region).",
                span=(decl_spans or {}).get(buf),
            )
    return overlaps


def tile_group_names(buffer_facts: dict) -> set[str]:
    """Every tile group's name. A GM buffer, the only other declaration, sits in DDR."""
    return {name for name, decl in buffer_facts.items() if decl.memory != MemorySpace.DDR}


def buffer_memory_spaces(buffer_facts: dict) -> dict:
    """Every declared buffer's memory space: name -> MemorySpace."""
    return {name: decl.memory for name, decl in buffer_facts.items()}


def buffer_slot_counts(buffer_facts: dict) -> dict:
    """Every declared buffer's slot count: name -> depth.

    A declaration whose depth the source leaves out is left out too, so that a reader asking
    whether a name has a slot count gets an answer from its presence alone.
    """
    return {name: decl.depth for name, decl in buffer_facts.items() if decl.depth is not None}


def buffer_addr_ranges(buffer_facts: dict) -> dict:
    """Each buffer's per-slot address ranges: name -> (memory, [(start, end), ...]).

    A declaration with no compile-time base takes no part in overlap detection, so it is
    left out rather than entered with a range nothing can be compared against.
    """
    result = {}
    for name, decl in buffer_facts.items():
        if decl.addrs is None or decl.slot_size is None:
            continue
        result[name] = (decl.memory, [(start, start + decl.slot_size) for start in decl.addrs])
    return result


def _ids_literal(ids: tuple) -> ast.expr:
    """``[0, 1]`` for the ids to be declared ahead of the pipeline loop."""
    return ast.fix_missing_locations(
        ast.List(elts=[ast.Constant(value=value) for value in ids], ctx=ast.Load())
    )


def _validate_cross_core_memory(name: str, decl) -> None:
    """L8: a cross-core tile group has to sit where the other core can reach it."""
    if decl.memory == MemorySpace.DDR or decl.memory in _CROSS_CORE_TILE_MEMORY:
        return
    raise NotSupported(
        f"pipeline: cross-core buffer '{name}' lives in "
        f"pl.MemorySpace.{decl.memory.name}, which cross-core sync does not support. "
        f"A tile group handing data to the other core has to be in pl.MemorySpace.Vec (UB) "
        f"or pl.MemorySpace.Mat (L1); to hand the data over through GM, declare the buffer "
        f"with pl.make_tensor(ws, shape, depth=..., fwd_ids=..., bwd_ids=...) instead.",
        span=decl.span,
    )


def _validate_cross_core_depth(name: str, decl) -> None:
    """A GM buffer rotates through a number of slots that has to be written down.

    A tile group's depth is settled while parsing; a GM view's is whatever its keyword said,
    which the parse only checked is a whole number.
    """
    if decl.memory == MemorySpace.DDR and (decl.depth is None or decl.depth <= 0):
        raise InvalidVal(
            f"pipeline: GM buffer '{name}' needs depth=<positive compile-time int> next to "
            f"its fwd_ids/bwd_ids: it is the number of slots the buffer rotates through.",
            span=decl.span,
        )


def _validate_event_id_range(name: str, label: str, ids: tuple, span) -> None:
    """L11: cross-core event ids are a hardware resource limited to 0..15.

    An out-of-range id only fails once the kernel runs on device, so it is refused where the
    source is still known. The ids themselves are whole integers already — the parse resolved
    them from whatever the declaration wrote.
    """
    bad = [value for value in ids if not 0 <= value <= MAX_EVENT_ID]
    if bad:
        raise OutOfRange(
            f"pipeline: cross-core buffer '{name}' {label} contains out-of-range "
            f"event id(s) {bad}; cross-core event ids must be in 0..{MAX_EVENT_ID}.",
            span=span,
        )


def validate_bound_buffer_decls(func_def: ast.FunctionDef, unbound_decls: list) -> None:
    """Refuse a buffer declaration inside a loop that no kernel-body variable holds.

    The pipeline identifies every buffer by the kernel-body variable bound to its
    declaration, so one declared inside a stage, inside a function a stage calls, or
    handed straight to a call as an argument has nothing to be tracked by. Only a
    declaration whose ``site`` is in a loop is the pipeline's to track; one made outside
    every loop belongs to code the pipeline does not analyse.
    """
    in_loop = loop_body_sites(func_def)
    for decl in sorted(unbound_decls, key=lambda decl: decl.site or (0, 0)):
        if decl.site not in in_loop:
            continue
        # Only a GM view sits in DDR, and only make_tensor declares one.
        written = "pl.make_tensor()" if decl.memory == MemorySpace.DDR else "pl.make_tile_group()"
        span, declared_at = buffer_decl_location(func_def, decl)
        made_by = f" at {declared_at}" if declared_at else ""
        raise InvalidOperation(
            f"pipeline: a buffer declared with `{written}`{made_by} is not bound to a variable in the "
            f"kernel body, so the pipeline cannot track it. A buffer cannot be declared inside a "
            f"stage or a function a stage calls, nor passed to a call directly.\n"
            f"Declare it in the kernel body, or return it from a function called there, and "
            f"bind it to a variable, e.g. `g = make_buffers()`.",
            span=span,
        )


def validate_distinct_buffer_names(func_def: ast.FunctionDef, same_name_decls: list) -> None:
    """Refuse two buffer declarations bound to the same kernel-body variable name.

    The pipeline identifies a buffer by that name across the whole kernel, sections
    included, so a second declaration under it would be taken for the first.
    """
    for name, *pair in same_name_decls:
        first, second = sorted(pair, key=lambda decl: decl.site or (0, 0))
        first_span, first_made_at = buffer_decl_location(func_def, first)
        span, made_at = buffer_decl_location(func_def, second)
        # Both locations name the BINDING, which is the line to change. Where the buffers were
        # declared is added only when that is elsewhere, and once when it is the same function
        # for both — two calls of one factory declare at the same line, so naming it twice
        # would read as two unrelated places.
        if made_at and made_at == first_made_at:
            made_by = f" Both are declared by the same call at {made_at}."
        elif made_at or first_made_at:
            made_by = (
                f" The first is declared at {first_made_at or 'that line'}, "
                f"the second at {made_at or 'this one'}."
            )
        else:
            made_by = ""
        raise InvalidOperation(
            f"pipeline: variable '{name}' is bound to a second buffer here, but "
            f"{first_span.filename}:{first_span.begin_line} already bound it to another one."
            f"{made_by} The pipeline identifies a buffer by its kernel-body variable name "
            f"across the whole kernel, sections included.\n"
            f"Give each buffer a name of its own.",
            span=span,
        )


def cross_core_buffers(buffer_facts: dict) -> tuple[dict, list]:
    """The cross-core buffers, and the variables their event ids are declared in.

    Declaring either direction of ids is what makes a buffer cross-core. Each direction's ids
    are emitted as a variable ahead of the pipeline loop and subscripted at runtime, so every
    direction present gets one.

    Returns:
        (name -> CrossCoreBuffer,
         lifted_ids: (variable name, the ids as a list, buffer name) — the buffer name travels
         with them so the declaration lands with the pipeline that uses that buffer.)
    """
    result: dict[str, CrossCoreBuffer] = {}
    lifted_ids: list[tuple[str, ast.expr, str]] = []

    for name, decl in buffer_facts.items():
        if decl.fwd_ids is None and decl.bwd_ids is None:
            continue
        _validate_cross_core_memory(name, decl)
        _validate_cross_core_depth(name, decl)

        nodes: dict[str, ast.expr] = {}
        for label, ids in (("fwd_ids", decl.fwd_ids), ("bwd_ids", decl.bwd_ids)):
            if ids is None:
                continue
            _validate_event_id_range(name, label, ids, decl.span)
            var_name = f"_pl_{label}_{name}"
            lifted_ids.append((var_name, _ids_literal(ids), name))
            nodes[label] = ast.Name(id=var_name, ctx=ast.Load())

        result[name] = CrossCoreBuffer(
            fwd_ids_node=nodes.get("fwd_ids"),
            bwd_ids_node=nodes.get("bwd_ids"),
            fwd_ids=decl.fwd_ids or (),
            bwd_ids=decl.bwd_ids or (),
            facts=decl,
        )

    return result, lifted_ids


def reject_buffer_access_outside_stages(info, op_accesses: dict, tables: AccessTables) -> None:
    """Refuse a statement in the pipeline loop's own body that touches a tracked buffer.

    Only a stage's accesses are scanned, and the wait/set the transform emits sit around a
    stage call, so such a statement would get no sync at all — and it is not replayed in the
    drain either. Silently wrong, hence refused.

    Tracked means what the sync graph tracks: a cross-core buffer, or one sharing its
    address. Work on a purely local buffer is left alone, so an unrelated store can stay out
    of a stage on purpose.

    ``op_accesses`` is the whole kernel's facts, per target: a statement in a section is
    parsed by one target only, and a bare one by both.
    """
    stage_sites = {
        (stage.call_node.lineno, stage.call_node.col_offset)
        for stage in info.stages
        if stage.call_node is not None
    }
    by_site: dict = {}
    for per_target in op_accesses.values():
        for site, records in per_target.items():
            if site not in stage_sites:
                by_site.setdefault(site, []).extend(records)

    for stmt in info.pipeline_loop.body:
        buffers = set()
        for node in ast.walk(stmt):
            site = (getattr(node, "lineno", None), getattr(node, "col_offset", None))
            for record in by_site.get(site, ()):
                buffers.update(
                    name
                    for operand in record.operands
                    for name in operand.names
                    if tables.tracks(name)
                )
        if not buffers:
            continue
        raise InvalidOperation(
            f"pipeline: the statement `{ast.unparse(stmt).splitlines()[0]}` touches buffer(s) "
            f"{sorted(buffers)} that "
            f"cross-core sync tracks, but it is not inside a stage. Only a stage's accesses "
            f"are scanned, and the wait/set the transform emits sit around the stage call, "
            f"so this access would get no synchronisation at all.\n"
            f"Move it into a stage. Work on a buffer that no cross-core handover involves "
            f"may stay here.",
            span=span_of(stmt_header(stmt)),
        )


# ----------------------------------------------------------------------------
# Per-op tile-argument role table (for cross-core sync direction inference)
# ----------------------------------------------------------------------------
#
# One role per declared parameter, in declaration order:
#   "W"   - written  (producer of cross-core data)
#   "R"   - read     (consumer of cross-core data)
#   "RW"  - read and written (e.g. matmul_acc accumulator)
#   None  - carries no read or write of buffer data
#
# Keyword-only parameters are covered too, so an argument written as a keyword
# takes the role at its own parameter's position. An entry's length equals the
# number of parameters the op declares.
#
# A parameter whose role depends on a keyword takes a pair instead:
#   (role, {keyword: role when the call passes that keyword})
#
# Only ops with their own page in the API docs are listed; a cross-core buffer
# handed to any other op is refused (see _reject_untabled_op).
#
# Note: cross-core sync MUST distinguish read vs write because the sync
# direction depends on it.

# ----------------------------------------------------------------------------
# Per-op tile-argument role table for vector-function (vf.*) ops
# ----------------------------------------------------------------------------
#
# Same roles as _BLOCK_OP_TILE_ROLES below, one per call argument, for the vf.*
# ops that touch a UB tile.  Every other vf op works purely on registers and is
# absent.
#
# A table rather than a rule, because no rule holds: the pointer is not always
# the first argument (the unaligned loads take their alignment-state register
# there), and the name does not say the direction either --- `gather` reads a
# tile, `scatter` and `squeeze_store_unalign` write one, and neither starts
# with "load" or "store".
#
# Adding a vf op that reads or writes a UB tile means adding it here too;
# the pipeline scan reports an op it cannot find when one reaches a buffer
# that cross-core sync tracks.

_VF_OP_TILE_ROLES: dict[str, list] = {
    # ===== Aligned =====
    "load_align": ["R"],  # load_align(src, offset, ...) -> reg
    "store_align": ["W", None],  # store_align(dst, src, ...)
    # ===== Unified (AscendC-style Load/Store) =====
    "load": ["R"],  # load(src_ptr, stride, ...) -> reg
    "store": ["W", None, None],  # store(dst_ptr, src, count)
    # ===== Unaligned: arg 0 is the alignment-state register, not the tile =====
    "load_unalign_pre": [None, "R"],  # load_unalign_pre(ureg, src_ptr)
    "load_unalign": [None, "R", None],  # load_unalign(ureg, src_ptr, stride)
    "store_unalign": ["W", None, None, None],  # (dst_ptr, src, align_reg, stride)
    "store_unalign_post": ["W", None, None],  # (dst_ptr, align_reg, stride)
    "squeeze_store_unalign": ["W", None, None],  # (dst_ptr, src, align_reg)
    "squeeze_store_unalign_post": ["W", None],  # (dst_ptr, align_reg)
    # ===== Indexed =====
    "gather": ["R", None, None],  # gather(src, indices, mask) -- src is a UB tile
    "scatter": ["W", None, None, None],  # scatter(base_ptr, src, index, mask)
}

_BLOCK_OP_TILE_ROLES: dict[str, list] = {
    # ===== Scalar element access (S pipe) =====
    # Reached both as pl.getval(t, i) / pl.setval(t, i, v) and as the subscript spellings
    # t[i] / t[i] = v that the parser lowers to them. A scalar read of a tile another core
    # wrote still has to wait for that write: measured on device, without the handshake the
    # read returns the buffer's previous contents, and wait_cross_core(pipe=PipeType.S)
    # fixes it.
    "getval": ["R", None],  # (container, offset)
    "setval": ["W", None, None],  # (container, offset, value)
    # ===== Data movement =====
    "init_output": ["W", None, None, None],  # (tensor, *, offset, size, value)
    "load": ["W", "R", None, None],  # (dst_tile, src_tensor, offsets, *, order)
    "load_tile": ["W", "R", None, None],  # (dst_tile, src_tensor, tile_offsets, *, order)
    # (dst_tensor, src_tile, offsets, *, relu_pre_mode, scale, order, atomic, phase)
    "store": ["W", "R", None, None, "R", None, None, None],
    "store_tile": ["W", "R", None, None, "R", None, None, None],
    # (dst_tile, src_tile, offset, *, acc_to_vec_mode, relu_pre_mode, scale, phase)
    "move": ["W", "R", None, None, None, "R", None],
    # (dst_tile, src_tile, offset, *, relu_pre_mode, scale, phase)
    "insert": ["W", "R", None, None, "R", None],
    # ===== Matmul =====
    "matmul": ["W", "R", "R", "R", None],  # (dst, lhs, rhs, bias, *, phase)
    "matmul_acc": ["RW", "R", "R", "R", None],  # (dst, acc, lhs, rhs, *, phase); dst is acc
    # (dst, lhs, rhs, scale_a, scale_b, bias, *, phase)
    "matmul_mx": ["W", "R", "R", "R", "R", "R", None],
    "matmul_mx_acc": ["RW", "R", "R", "R", "R", "R", None],
    # pl.gemm()/pl.gemm_mx() are absent: the parse rewrites either into the matmul above plus
    # the moves staging its operands (see _gemm_lowering), and the facts hold what it emitted.
    # ===== Binary element-wise =====
    "add": ["W", "R", "R"],
    "sub": ["W", "R", "R"],
    "mul": ["W", "R", "R"],
    "div": ["W", "R", "R"],
    "and_": ["W", "R", "R"],
    "xor": ["W", "R", "R", None],  # (out, lhs, rhs, tmp)
    # (out, lhs, rhs, *, dim); dim= reduces, which makes rhs a workspace tile
    "maximum": ["W", "R", ("R", {"dim": None}), None],
    # ===== Fused / composite =====
    "add_relu": ["W", "RW", "R"],  # lhs holds the intermediate sum
    "axpy": ["RW", "R", None],  # out = alpha * src + out
    # ===== Unary =====
    "neg": ["W", "R"],
    "abs": ["W", "R"],
    "relu": ["W", "R"],
    # ===== Type cast =====
    "cast": ["W", "R", None],  # (out, src, *, mode)
    # ===== Select / compare =====
    "select": ["W", "R", "R", "R", None],  # (out, mask, lhs, rhs, tmp)
    "eq": ["W", "R", "R"],
    # ===== Quantisation =====
    "quant": ["W", "R", "R", None, "R"],  # (out, src, scale, *, mode, offset)
    "dequant": ["W", "R", "R", "R"],  # (out, src, scale, offset)
    # ===== Reduce =====
    "sum": ["W", "R", None, None],  # (out, src, tmp, *, dim)
    "expands": ["W", None],  # (out, scalar)
    # ===== Layout / element access =====
    "fill_index": ["W", None],  # (out, start)
    "transpose": ["W", "R"],
}


# ----------------------------------------------------------------------------
# Descriptor-only (metadata) ops
# ----------------------------------------------------------------------------
# These block ops never read or write a tile's buffer *data*: they rewrite its
# *descriptor* (e.g. its valid (row, col) window) or hand back another handle on
# the same storage.  Accesses to their tiles therefore cannot race, so they need
# no cross-core / cross-pipe synchronisation.  This mirrors the tile-graph
# sync pass, which likewise excludes the metadata opcodes OP_VIEW /
# OP_VIEW_TYPE / OP_RESHAPE / OP_ASSEMBLE.
_DESCRIPTOR_ONLY_OPS: frozenset[str] = frozenset(
    {
        "make_tile_group",  # Declares storage metadata; does not access tile data.
        "reinterpret",  # Returns another handle on the same storage.
        "set_validshape",
        "subview",
    }
)


def op_accesses_buffer(op_name: str) -> bool:
    """Whether a block op reads or writes its tiles' buffer data.

    False for descriptor-only ops (they only touch tile metadata), which is the
    signal that cross-core sync can skip them.
    """
    return _BLOCK_OP_ALIASES.get(op_name, op_name) not in _DESCRIPTOR_ONLY_OPS


_UNCOVERED = object()


@functools.lru_cache(maxsize=None)
def _declared_signature(op_name: str) -> inspect.Signature | None:
    """The signature the op's API declaration gives, or None when it has no declaration."""
    declared = getattr(_language_api, op_name, None)
    if declared is None:
        return None
    try:
        return inspect.signature(declared)
    except (TypeError, ValueError):
        return None


def _block_op_roles(op_name: str, passed: set) -> list | None:
    """The argument roles for this call, or None when the table does not cover the op.

    A parameter whose role depends on a keyword is tabled as ``(role, {keyword: role})``
    and takes the role of a keyword the call passes. Which keywords those are is what the
    table holds, so a new one needs an entry there and nothing here.
    """
    roles = _BLOCK_OP_TILE_ROLES.get(op_name)
    if roles is None:
        return None
    return [
        next((r for k, r in role[1].items() if k in passed), role[0])
        if isinstance(role, tuple) else role
        for role in roles
    ]


def _reject_untabled_op(shown: str, span, op_name: str, touched: list) -> None:
    """Refuse an op that touches a tracked buffer but has no roles in the table.

    The table covers the ops published in the API docs; reaching one outside it would mean
    recording nothing — a missing wait/set with nothing to announce it. There is no safe
    default: guessing "read" drops the producer's set, guessing "write" drops the consumer's
    wait.
    """
    raise InvalidArgument(
        f"pipeline: `{shown}` touches buffer(s) {sorted(set(touched))} that "
        f"cross-core sync tracks, but '{op_name}' has no entry in _BLOCK_OP_TILE_ROLES, so "
        f"the scan cannot tell which arguments it reads and which it writes — and which way "
        f"a wait/set goes depends on exactly that.\n"
        f"That table covers the ops published in the API docs, so '{op_name}' cannot carry a "
        f"cross-core handover yet. If you added the op yourself, give it one role per "
        f"declared parameter in pypto_pro/runtime/pipeline/_cross_core_scanner.py.",
        span=span,
    )


def _reject_roleless_arg(shown: str, span, op_name: str, param: str, buf: str, role: object, table: str) -> None:
    """Refuse a tracked buffer at an argument the table gives no read/write role.

    ``table`` is the roles table the op was looked up in, so the gap is reported where it
    has to be filled.
    """
    where = f"as '{param}'" if param else "here"
    if role is None:
        raise InvalidArgument(
            f"pipeline: `{shown}` passes buffer '{buf}' {where}, and cross-core "
            f"sync tracks that buffer, but the parameter carries no read or write of the "
            f"buffer's data — it is workspace the op hands to the hardware. No handover can "
            f"be ordered through it, so the buffer would get no wait/set at all.\n"
            f"Pass a buffer local to this core {where}.",
            span=span,
        )
    raise InvalidArgument(
        f"pipeline: `{shown}` passes buffer '{buf}' {where}, and cross-core sync "
        f"tracks that buffer, but {table} gives '{op_name}' no role at that "
        f"parameter, so the scan cannot tell whether it is read or written.\n"
        f"The table holds one role per declared parameter in "
        f"pypto_pro/runtime/pipeline/_cross_core_scanner.py; a parameter it misses is a gap there.",
        span=span,
    )


class StageAccess(NamedTuple):
    """One buffer access from the parse facts.

    ``role``/``pipe`` are set for a buffer the sync graph tracks; ``selector`` is set for a
    declared tile-group slot selection. A local group has role/pipe None; a cross-core
    buffer may have both, but the lifetime check skips it.
    """

    buffer: str
    role: str | None
    pipe: str | None
    selector: object | None = None


def stage_accesses(stage, by_site: dict, tables: AccessTables) -> list:
    """One stage's accesses from the parse facts, in one pass.

    The parse already resolved which declaration every operand reaches, so no name is bound
    here; what is left is the role/pipe view for sync and the slot-selector view for
    lifetime, both emitted as ``StageAccess`` records.

    Args:
        stage: the StageCall whose call site keys the facts.
        by_site: ``{(lineno, col_offset): [StageOpAccess, ...]}`` for this stage's target.
        tables: the kernel-level lookup tables (see AccessTables).
    """
    site = (stage.call_node.lineno, stage.call_node.col_offset) if stage.call_node else None
    records = by_site.get(site)
    if records is None:
        raise InvalidOperation(
            f"pipeline: the parse facts hold no ops for stage '{stage.func_name}'. Every stage "
            f"the pipeline runs is expanded by the probe parse, so one that is missing here "
            f"would be synchronised as though it touched nothing.",
            span=span_of(stage.call_node or stage.source_stmt),
        )
    accesses: list = []
    for access in records:
        accesses.extend(_facts_op(access, tables))
    return accesses


def _facts_op(access, tables: AccessTables) -> list:
    """Record one op from its parse facts.

    A vf instruction differs from a block op in two things only: the pipe, since nothing
    inside a vector function is registered under ``block.`` and the function as a whole runs
    on V, and the table its argument roles come from. Everything else — folding the arguments
    per buffer and emitting one access per buffer — is shared, so ``in_vf`` picks those two
    and nothing more.

    A plain ``pl.*`` op cannot appear inside a vector function (the parser allows only
    ``vf.*``, ``pl.range`` and the scalar ``pl.min``/``pl.max``/``pl.const`` there, none of
    which reaches a buffer), so being inside a vf call and being a vf instruction are the
    same question for every op recorded with operands.
    """
    out: list = []
    # Every slot selection is a lifetime fact, whether or not the op reaches a tracked buffer.
    for operand in access.operands:
        for buffer_name, selector in operand.selectors:
            out.append(StageAccess(buffer_name, None, None, selector))

    # A descriptor-only op touches no buffer data, whatever pipe it is registered on.
    if not op_accesses_buffer(access.op_name):
        return out
    # An op reaching no tracked buffer orders nothing, so its pipe and roles never come up.
    touched = [name for operand in access.operands for name in operand.names if tables.tracks(name)]
    if not touched:
        return out
    in_vf = access.in_vf
    if in_vf:
        op_name = access.op_name.removeprefix("vf.")
        shown = f"vf.{op_name}()"
        table = "_VF_OP_TILE_ROLES"
        pipe = "V"
        roles = _VF_OP_TILE_ROLES.get(op_name)
    else:
        op_name = access.op_name
        shown = f"pl.{op_name}()"
        table = "_BLOCK_OP_TILE_ROLES"
        pipe = _pipe_from_operand_memory(
            op_name, shown, access.span,
            lambda position: _facts_arg_memory(access, position, tables.all_buffer_memory),
        )
        if pipe is None:
            return out
        roles = _block_op_roles(op_name, set(access.keyword_names))
    if roles is None:
        if in_vf:
            _reject_untabled_vf_op(access, op_name, touched[0])
        else:
            _reject_untabled_op(shown, access.span, op_name, touched)
    # One record per buffer, not per argument.
    merged: dict[str, str] = {}
    for param, names, role in _facts_arg_roles(access, roles):
        for name in names:
            if not tables.tracks(name):
                continue
            if role is None or role is _UNCOVERED:
                _reject_roleless_arg(shown, access.span, op_name, param, name, role, table)
            _merge_role(merged, name, role)
    for buf, role in merged.items():
        _record_access(out, buf, role, pipe)
    return out


def _facts_arg_roles(access, roles: list):
    """Yield ``(parameter, buffer names, role)`` for every recorded operand of an op.

    A value written as a keyword gets the role of the parameter it binds to, the same one it
    would get written by position. A vf instruction has no declaration to take parameter names
    from, and the parse records only its positional arguments, so its operands map straight
    to the table's positions.
    """
    sig = _declared_signature(access.op_name)
    params = tuple(sig.parameters) if sig is not None else ()
    for operand in access.operands:
        position, keyword, names = operand.position, operand.keyword, operand.names
        index = (params.index(keyword) if keyword in params else -1) if keyword is not None else position
        role = roles[index] if 0 <= index < len(roles) else _UNCOVERED
        param = keyword if keyword is not None else (params[index] if 0 <= index < len(params) else "")
        yield param, names, role


def _facts_arg_memory(access, position: int, all_buffer_memory: dict) -> MemorySpace | None:
    """The MemorySpace of the operand written at ``position``, or None when it has no buffer."""
    for operand in access.operands:
        if operand.position == position:
            return all_buffer_memory.get(operand.names[0]) if operand.names else None
    return None


def _reject_untabled_vf_op(access, vf_op: str, buf: str) -> None:
    """Refuse a vf op that reaches a tracked buffer but has no roles in the table."""
    raise NotSupported(
        f"pipeline: `vf.{vf_op}()` is handed buffer '{buf}', but '{vf_op}' has no entry in "
        f"_VF_OP_TILE_ROLES, so the scan cannot tell whether it reads or writes the tile — "
        f"and which way a cross-core wait/set goes depends on that.\n"
        f"That table covers the vf ops published in the API docs, so '{vf_op}' cannot carry a "
        f"cross-core handover yet. If you added the op yourself, give it a role per argument "
        f"in pypto_pro/runtime/pipeline/_cross_core_scanner.py.",
        span=access.span,
    )


_BLOCK_OP_ALIASES: dict[str, str] = {
    "and_": "and",
    "eq": "cmp",
    "ne": "cmp",
    "lt": "cmp",
    "le": "cmp",
    "gt": "cmp",
    "ge": "cmp",
    "select": "sel",
    "load_tile": "load",
    "make_tile": "make_mutex_tile",
    "not_": "not",
    "or_": "or",
    "sum": "row_sum",
    "argmax": "row_argmax",
    "argmin": "row_argmin",
    "expand_max": "row_expand_max",
    "expand_min": "row_expand_min",
    "expand_mul": "row_expand_mul",
    "expand_sub": "row_expand_sub",
    "expand_div": "row_expand_div",
    "prod": "row_prod",
    "expand_add": "row_expand_add",
    "expand_expdif": "row_expand_expdif",
}


def get_op_pipe(op_name: str) -> PipeType | None:
    """Return the fixed backend pipe, or None when arguments determine the pipe."""
    backend_op_name = _BLOCK_OP_ALIASES.get(op_name, op_name)
    return BackendCCE.instance().get_op_pipe(f"block.{backend_op_name}")


def get_move_pipe(
    src_memory: MemorySpace | None,
    target_memory: MemorySpace | None,
) -> PipeType:
    """Determine the hardware pipe for a block ``move`` operation."""
    if src_memory == MemorySpace.Acc and target_memory in (MemorySpace.Vec, MemorySpace.Mat):
        return PipeType.FIX
    if src_memory == MemorySpace.Mat:
        if target_memory in (MemorySpace.Left, MemorySpace.Right, MemorySpace.Bias, MemorySpace.ScaleLeft,
                             MemorySpace.ScaleRight):
            return PipeType.MTE1
        if target_memory == MemorySpace.Vec:
            return PipeType.V
        return PipeType.FIX
    if src_memory == MemorySpace.Vec and target_memory == MemorySpace.Mat:
        return PipeType.MTE3
    return PipeType.V


def get_store_pipe(src_memory: MemorySpace | None) -> PipeType:
    """Determine the hardware pipe for a block ``store`` / ``store_tile`` operation."""
    if src_memory == MemorySpace.Acc:
        return PipeType.FIX
    return PipeType.MTE3


def get_expands_pipe(target_memory: MemorySpace) -> PipeType:
    """Scalar broadcast uses MTE2 for Mat tiles and V for Vec tiles."""
    return PipeType.MTE2 if target_memory == MemorySpace.Mat else PipeType.V


def _pipe_from_operand_memory(op_name: str, shown: str, span, memory_at) -> str | None:
    """The pipe a block op's wait/set goes on, or None when sync does not cover the op.

    Sync covers what the backend registers under ``block.``, so the pipe registered there
    answers both questions at once. The ops whose pipe is not one fixed value are branched on
    by name above that, which is also what makes them covered.

    ``memory_at(position)`` gives the MemorySpace of the operand written at that positional
    index, or None when it resolves to no buffer.

    Raises when a covered op's pipe cannot be determined. There is no safe default: the pipe
    decides which queue the wait/set lands on, and a section only has some of the pipes, so a
    guess can even name one that does not exist on that core.
    """
    if op_name in ("move", "insert"):
        # move(dst, src): pipe depends on src/dst memory
        # Tile-to-tile data movement uses a pipe selected by src/dst memory.
        dst_mem = memory_at(0)
        src_mem = memory_at(1)
        if src_mem is None or dst_mem is None:
            unresolved = "destination" if dst_mem is None else "source"
            raise InvalidOperation(
                f"pipeline: cannot determine the pipe of `{shown}`, "
                f"because its {unresolved} tile does not resolve to a declared tile group. "
                f"The move touches a cross-core buffer, so its pipe decides where the sync "
                f"goes. A tile reached through an aggregate (e.g. `tile_groups.x.next()`) is "
                f"not yet traced; pass the tile group to the stage as its own argument.",
                span=span,
            )
        return _pipe_name(get_move_pipe(src_mem, dst_mem))
    if op_name in ("store", "store_tile"):
        src_mem = memory_at(1)
        if src_mem is None:
            raise InvalidOperation(
                f"pipeline: cannot determine the pipe of `{shown}`, "
                f"because its source tile does not resolve to a declared tile group. The store "
                f"touches a cross-core buffer, so its pipe decides where the sync goes. A tile "
                f"reached through an aggregate (e.g. `tile_groups.x.next()`) is not yet traced; "
                f"pass the tile group to the stage as its own argument.",
                span=span,
            )
        return _pipe_name(get_store_pipe(src_mem))
    if op_name in ("expands", "full"):
        dst_mem = memory_at(0)
        if dst_mem is None:
            raise InvalidOperation(
                f"pipeline: cannot determine the pipe of `{shown}`, "
                "because its destination tile does not resolve to a declared tile group.",
                span=span,
            )
        return _pipe_name(get_expands_pipe(dst_mem))
    pipe = get_op_pipe(op_name)
    return _pipe_name(pipe) if pipe is not None else None


def _pipe_name(pipe) -> str:
    """PipeType enum -> short name string used in generated pl.PipeType.<NAME>."""
    # pipe is a PipeType enum; its name attribute gives FIX/V/MTE1/...
    return getattr(pipe, "name", str(pipe).split(".")[-1])


def _record_access(out: list, buf: str, role: str, pipe: str) -> None:
    """Append one op-level access.

    Kept per-op rather than collapsed to (first_pipe, last_pipe): the graph needs each
    access as its own node, and a pair of endpoints cannot express a local access sitting
    BETWEEN two cross-core ones.
    """
    out.append(StageAccess(buf, role, pipe))


def _merge_role(result: dict[str, str], key: str, role: str | None) -> None:
    """Fold one more role for ``key`` into ``result``: differing roles become "RW".

    Differing roles mean the same buffer reached the op at both a read and a write position,
    either written there twice (``pl.muls(x, x, 2.0)``) or reached through operands with more
    than one origin — ONE access, not two. Two accesses would also carry the argument order,
    which is not the order the hardware reads and writes in.
    """
    existing = result.get(key)
    if existing is None:
        result[key] = role
    elif existing != role and role is not None:
        result[key] = "RW"
