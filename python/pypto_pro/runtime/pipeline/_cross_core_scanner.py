# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Cross-core access scanner for preload pipeline auto-sync.

Pure-AST analysis. Scans the kernel body for cross-core tile-group declarations (those
carrying fwd_ids/bwd_ids), then scans each @stage body to determine, per buffer it
touches, the access role (R/W/RW) and the pipe of the op doing the access.

The result drives automatic wait/set_cross_core insertion at stage boundaries.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from pypto.pypto_impl.ir import MemorySpace
from pypto_pro.language import _api as _language_api
from pypto_pro.language._vf_api import Vf
from pypto_pro.language.parser._op_pipeline import (
    _BLOCK_OP_TILE_ROLES,
    _VF_OP_TILE_ROLES,
    get_move_pipe,
    get_op_pipe,
    get_store_pipe,
    op_accesses_buffer,
)

from ..._errors import InvalidArgument, InvalidOperation, InvalidType, InvalidVal, NotSupported, OutOfRange, span_of
from ._astutil import call_name, slot_accessor

# Highest usable cross-core event id: a hardware limit. Shared with _sync_graph, which
# allocates from the same pool for address-reuse edges.
MAX_EVENT_ID = 15

# MemorySpace attribute name (as written in pl.MemorySpace.<X>) -> enum value
_MEMORY_NAMES = {
    "Vec": MemorySpace.Vec,
    "Mat": MemorySpace.Mat,
    "Left": MemorySpace.Left,
    "Right": MemorySpace.Right,
    "Acc": MemorySpace.Acc,
    "ScaleLeft": MemorySpace.ScaleLeft,
    "ScaleRight": MemorySpace.ScaleRight,
}


def _tile_type_memory(type_node: ast.expr | None) -> MemorySpace | None:
    """The MemorySpace a ``pl.TileType(..., target_memory=pl.MemorySpace.X)`` call declares.

    None when the node is not such a call, or its target_memory is not written as a literal
    ``pl.MemorySpace.<X>``.
    """
    if not isinstance(type_node, ast.Call):
        return None
    for kw in type_node.keywords:
        if kw.arg == "target_memory" and isinstance(kw.value, ast.Attribute):
            if kw.value.attr in _MEMORY_NAMES:
                return _MEMORY_NAMES[kw.value.attr]
    return None


def _is_make_tile_group(call: ast.Call) -> bool:
    """True if call is pl.make_tile_group(...) (or bare make_tile_group(...))."""
    return _get_ctor_name(call) == "make_tile_group"


@dataclass
class CrossCoreBuffer:
    """A cross-core NBuffer's configuration (from kernel-body declaration)."""

    fwd_ids_node: ast.expr | None  # AST node for cross_core_forward_id value (e.g. a Name)
    bwd_ids_node: ast.expr | None  # AST node for cross_core_backward_id value
    # The ids themselves, resolved from whichever spelling was used (see _resolve_event_ids).
    # Kept rather than only counted: the address-reuse allocator needs to know which ids are
    # already spoken for, including those given by name.
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

    Populated by _analyzer._scan_kernel_buffers() during analysis.
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
    # buffer name -> tuple of mutex ids. Used to check that co-located buffers hold the
    # same locks (a mutex locks the address, not the variable).
    mutex_ids: dict = field(default_factory=dict)
    # Address-overlapping buffer pairs: [(buf_a, buf_b), ...] (same memory, ranges intersect)
    addr_overlaps: list = field(default_factory=list)
    # Address-reuse sync is derived from the graph on demand, not stored here — see
    # _sync_graph.allocate_reuse_ids.


@dataclass(frozen=True)
class AccessScanTables:
    """The kernel-level tables the per-op ACCESS scan resolves names against.

    Every field answers the same question: *a name appears inside a stage body — which
    declared buffer is it, and on which pipe is this op touching it?* Resolved once for the
    whole kernel and only read afterwards, so they travel as one value.

    The per-stage ``stage_slot_to_buffer`` deliberately lives elsewhere: it is not a table
    the scan consults but the result of consulting these (_build_slot_to_buffer_from_bindings).
    """

    # Cross-core buffer declarations: name -> CrossCoreBuffer. Membership decides whether
    # an access is worth recording at all.
    cross_buffers: dict
    # name -> MemorySpace for ALL buffers, cross-core or local. A move's pipe follows the
    # memory spaces of both its tiles, so the local side has to resolve too.
    all_buffer_memory: dict
    # Every declared tile group's name, so a call-site argument can be recognised as a group.
    group_names: set
    # Slot variables taken in the KERNEL body (`cur_k = k_db.next()`) -> buffer. Read only
    # by build_binding_map, to resolve a call-site actual argument.
    kernel_slot_to_buffer: dict
    # tuple variable -> {field: source variable}, the one hop that rejoins a tile to its
    # group when the kernel bundles groups with pl.make_tuple.
    tuple_fields: dict
    # ``@pl.vector_function`` name -> FunctionDef. One atomic op each: the scan reads their
    # parameter roles instead of walking in (see _scan_vf_roles).
    vf_func_defs: dict
    # Every OTHER plain function a stage may call -> FunctionDef. Walked through as part of
    # the calling stage (see _scan_function).
    helper_func_defs: dict
    # Buffers whose address range overlaps another buffer's, recorded even when not
    # cross-core themselves: a write over shared memory still has to be ordered. NOT the
    # sync graph's "region", which is every buffer's memory area, reuse or not.
    addr_shared: set
    # The kernel's enclosing scope, for resolving what a callee name stands for (see
    # resolve_op_name).
    closure_vars: dict

    def tracks(self, buf: str | None) -> bool:
        """True if the sync graph follows this buffer: cross-core, or sharing an address
        with one. Everything else is the user's own, and auto_mutex orders it."""
        return buf is not None and (buf in self.cross_buffers or buf in self.addr_shared)


@dataclass
class TileGroupDecl:
    """One ``x = pl.make_tile_group(...)`` statement, as written.

    Purely syntactic: the ``_node`` fields are unevaluated AST, and each scan below decides
    for itself what to do when a value will not resolve.

    Two things are resolved here rather than per reader: ``type_node`` is the
    ``pl.TileType(...)`` call itself even when the source wrote a variable bound to it, and
    ``memory`` is ``target_memory`` read off that node by pattern matching.
    """

    names: list[str]  # every variable this statement binds (targets that are plain Names)
    type_node: ast.expr | None  # the pl.TileType(...) call, resolved through a name if needed
    addrs_node: ast.expr | None
    mutex_ids_node: ast.expr | None
    depth_node: ast.expr | None
    fwd_ids_node: ast.expr | None
    bwd_ids_node: ast.expr | None
    memory: MemorySpace | None  # target_memory, or None when it does not resolve


def scan_tile_group_decls(kernel_func_def: ast.FunctionDef) -> list[TileGroupDecl]:
    """Every ``pl.make_tile_group(...)`` declaration in the kernel body, in AST order.

    The one pass over the kernel body that all buffer-declaration scans share.
    """
    tile_types = _scan_tile_type_decls(kernel_func_def)
    decls: list[TileGroupDecl] = []
    for node in ast.walk(kernel_func_def):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            continue
        call = node.value
        if not _is_make_tile_group(call):
            continue
        written = {kw.arg: kw.value for kw in call.keywords if kw.arg}
        # `type=tt` -> the pl.TileType(...) call `tt` was bound to.
        type_node = written.get("type")
        if isinstance(type_node, ast.Name) and type_node.id in tile_types:
            type_node = tile_types[type_node.id]
        decls.append(
            TileGroupDecl(
                names=[t.id for t in node.targets if isinstance(t, ast.Name)],
                type_node=type_node,
                addrs_node=written.get("addrs"),
                mutex_ids_node=written.get("mutex_ids"),
                depth_node=written.get("depth"),
                fwd_ids_node=written.get("fwd_ids"),
                bwd_ids_node=written.get("bwd_ids"),
                # target_memory is a keyword of the TILE TYPE, not of make_tile_group.
                memory=_tile_type_memory(type_node),
            )
        )
    return decls


def _scan_tile_type_decls(kernel_func_def: ast.FunctionDef) -> dict[str, ast.Call]:
    """``{name: the pl.TileType(...) call it was bound to}`` in the kernel body.

    One hop, which is all the spelling needs. An unresolved name simply leaves
    ``kwargs["type"]`` as it was.
    """
    result: dict[str, ast.Call] = {}
    for node in ast.walk(kernel_func_def):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            continue
        if _get_ctor_name(node.value) != "TileType":
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                result[target.id] = node.value
    return result


def scan_all_tile_group_names(decls: list[TileGroupDecl]) -> set[str]:
    """Every variable bound directly to a ``pl.make_tile_group(...)`` result.

    Independent of mutex_ids and cross-core ids: a group whose ids do not resolve statically
    is still a group, and callers that recognise slot selection must not miss it.
    """
    return {name for decl in decls for name in decl.names}


def scan_all_buffer_memory(decls: list[TileGroupDecl]) -> dict[str, MemorySpace]:
    """Return every make_tile_group declaration's memory by buffer variable name."""
    return {name: decl.memory for decl in decls if decl.memory is not None for name in decl.names}


def _eval_const(node: ast.expr, closure_vars: dict):
    """Evaluate a constant expression node using closure_vars as namespace.
    Returns the value, or None if it cannot be evaluated (e.g. references a
    runtime variable)."""
    try:
        code = compile(ast.Expression(body=node), "<addr>", "eval")
        return eval(code, {"__builtins__": {}}, dict(closure_vars))
    except Exception:
        return None


def _tile_type_slot_size(type_call: ast.Call, closure_vars: dict) -> int | None:
    """Compute per-slot byte size from a pl.TileType(shape=..., dtype=...) call.
    Returns None if shape/dtype cannot be resolved.

    slot_size = product(shape dims) * ceil(dtype_bits / 8)  (mirrors _buffer_parser).
    """
    shape_node = dtype_node = None
    for kw in type_call.keywords:
        if kw.arg == "shape":
            shape_node = kw.value
        elif kw.arg == "dtype":
            dtype_node = kw.value
    if shape_node is None or dtype_node is None:
        return None
    shape = _eval_const(shape_node, closure_vars)
    dtype = _eval_const(dtype_node, closure_vars)
    if not isinstance(shape, (list, tuple)) or dtype is None:
        return None
    elems = 1
    for d in shape:
        elems *= int(d)
    try:
        bits = int(dtype.get_bit())
    except Exception:
        return None
    return elems * ((bits + 7) // 8)


def scan_buffer_addr_ranges(decls: list[TileGroupDecl], closure_vars: dict) -> dict:
    """Compute each buffer's per-slot address ranges from its declaration.

    Returns: name -> (memory, [(start, end), ...] per slot).
    Buffers whose addrs/shape/dtype cannot be statically resolved are skipped
    (they simply won't participate in overlap detection).

    addrs handling (mirrors _buffer_parser):
      - single value  -> contiguous slots: base + i*slot_size
      - list/tuple     -> one explicit start address per slot
    slot count comes from depth when present, otherwise len(mutex_ids).
    """
    result: dict = {}
    for decl in decls:
        memory = decl.memory
        if memory is None:
            continue

        type_node = decl.type_node
        addrs_node = decl.addrs_node
        mutex_node = decl.mutex_ids_node
        depth_node = decl.depth_node
        if not (isinstance(type_node, ast.Call) and addrs_node is not None):
            continue

        slot_size = _tile_type_slot_size(type_node, closure_vars)
        mutex_ids = _eval_const(mutex_node, closure_vars) if mutex_node is not None else None
        depth = _eval_const(depth_node, closure_vars) if depth_node is not None else None
        addrs = _eval_const(addrs_node, closure_vars)
        if slot_size is None or addrs is None:
            continue
        if mutex_ids is not None and not isinstance(mutex_ids, (list, tuple)):
            continue
        if depth_node is not None:
            if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
                continue
            num = depth
        elif isinstance(mutex_ids, (list, tuple)) and mutex_ids:
            num = len(mutex_ids)
        else:
            continue
        if isinstance(mutex_ids, (list, tuple)) and mutex_ids and len(mutex_ids) != num:
            continue  # malformed; parser will raise the real error

        if isinstance(addrs, (list, tuple)):
            if len(addrs) != num:
                continue  # malformed; skip (parser will raise the real error)
            starts = list(addrs)
        else:
            starts = [addrs + i * slot_size for i in range(num)]
        ranges = [(int(s), int(s) + slot_size) for s in starts]

        for name in decl.names:
            result[name] = (memory, ranges)
    return result


def scan_buffer_mutex_ids(decls: list[TileGroupDecl], closure_vars: dict) -> dict:
    """Each buffer's mutex_ids. Returns name -> tuple of ids.

    Needed to check that buffers sharing an address also share their locks: a mutex locks
    the address, so different ids over one region provide no mutual exclusion at all.
    Buffers whose ids are not statically resolvable are omitted rather than guessed.
    """
    result: dict = {}
    for decl in decls:
        mutex_node = decl.mutex_ids_node
        if mutex_node is None:
            continue
        mutex_ids = _eval_const(mutex_node, closure_vars)
        if not isinstance(mutex_ids, (list, tuple)):
            continue
        for name in decl.names:
            result[name] = tuple(mutex_ids)
    return result


def detect_addr_overlaps(addr_ranges: dict, cross_core_names: set) -> list:
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
                        f"Overlapping buffers must have the same number of slots."
                    )
                overlaps.append((a, b))
                overlap_partners.setdefault(a, set()).add(b)
                overlap_partners.setdefault(b, set()).add(a)

    for buf, partners in overlap_partners.items():
        if len(partners) > 1:
            raise NotSupported(
                f"pipeline: buffer '{buf}' has overlapping addresses with multiple "
                f"buffers {sorted(partners)}. Only pairwise address overlap is "
                f"supported (at most 2 buffers may share a region)."
            )
    return overlaps


def scan_cross_core_buffers(
    decls: list[TileGroupDecl], closure_vars: dict
) -> tuple[dict[str, CrossCoreBuffer], list]:
    """Pick out the cross-core tile-group declarations (those carrying fwd/bwd ids).

    The only declaration scan that raises: a buffer asking for cross-core sync must have a
    resolvable memory space and valid id tuples, or no sync can be generated for it.

    Returns:
        (name -> CrossCoreBuffer,
         lifted_ids: (var_name, ast_literal, buffer_name) for literal fwd/bwd ids that must
         be declared as variables before the pipeline loop; the buffer name travels with
         them so the declaration lands with the pipeline that uses that buffer.)
    """
    result: dict[str, CrossCoreBuffer] = {}
    lifted_ids: list[tuple[str, ast.expr, str]] = []

    for decl in decls:
        fwd_node = decl.fwd_ids_node
        bwd_node = decl.bwd_ids_node
        if fwd_node is None and bwd_node is None:
            continue

        bufname = decl.names[0] if decl.names else "<tile_group>"

        _validate_buffer_memory(decl, bufname)
        fwd_ids, bwd_ids = _validate_ids(fwd_node, bwd_node, bufname, closure_vars)
        fwd_node, bwd_node = _lift_literal_ids(fwd_node, bwd_node, bufname, lifted_ids)

        for name in decl.names:
            result[name] = CrossCoreBuffer(
                fwd_ids_node=fwd_node,
                bwd_ids_node=bwd_node,
                fwd_ids=fwd_ids,
                bwd_ids=bwd_ids,
            )

    return result, lifted_ids


def _validate_buffer_memory(decl: TileGroupDecl, bufname: str) -> None:
    """L8: Validate that a make_tile_group call declares a resolvable memory space."""
    if decl.memory is None:
        raise InvalidVal(
            f"pipeline: cross-core buffer '{bufname}' has no resolvable memory space. "
            f"Its make_tile_group(type=pl.TileType(..., target_memory=pl.MemorySpace.X)) "
            f"must set target_memory to a literal pl.MemorySpace.<X>."
        )


def _validate_ids(fwd_node, bwd_node, bufname: str, closure_vars: dict) -> tuple[tuple, tuple]:
    """L11: the ids each direction declares, once they are known to be usable.

    Checks the spelling and the values. How MANY there must be depends on the buffer's slot
    count, which is not known here — see _validate._check_event_id_counts.
    """
    resolved = []
    for label, node in (("fwd_ids", fwd_node), ("bwd_ids", bwd_node)):
        if node is None:
            resolved.append(())
            continue
        ids = _resolve_event_ids(node, bufname, label, closure_vars)
        if not ids:
            raise InvalidVal(
                f"pipeline: cross-core buffer '{bufname}' {label} is empty. Drop the "
                f"keyword if this buffer needs no sync in that direction."
            )
        # Cross-core event ids are a hardware resource limited to 0..15; an out-of-range id only
        # fails once the kernel runs on device, so reject it where the source is known.
        bad = [v for v in ids if not 0 <= v <= MAX_EVENT_ID]
        if bad:
            raise OutOfRange(
                f"pipeline: cross-core buffer '{bufname}' {label} contains out-of-range "
                f"event id(s) {bad}; cross-core event ids must be in 0..{MAX_EVENT_ID}."
            )
        resolved.append(tuple(ids))
    return resolved[0], resolved[1]


def _lift_literal_ids(fwd_node, bwd_node, bufname: str, lifted_ids: list) -> tuple[ast.expr, ast.expr]:
    """Lift literal fwd_ids/bwd_ids to variable names for codegen compatibility."""
    if fwd_node is not None and isinstance(fwd_node, (ast.Tuple, ast.List)):
        var_name = f"_pl_fwd_ids_{bufname}"
        lifted_ids.append((var_name, fwd_node, bufname))
        fwd_node = ast.Name(id=var_name, ctx=ast.Load())
    if bwd_node is not None and isinstance(bwd_node, (ast.Tuple, ast.List)):
        var_name = f"_pl_bwd_ids_{bufname}"
        lifted_ids.append((var_name, bwd_node, bufname))
        bwd_node = ast.Name(id=var_name, ctx=ast.Load())
    return fwd_node, bwd_node


def _is_subview(node: ast.expr) -> bool:
    """Whether ``node`` is ``tile[...]`` with a slice among its indices — a sub-view.

    A sub-view is a window onto the same memory as its parent tile, so it resolves to that
    tile's buffer (build_binding_map) and is not itself an access (_handle_tile_subscript).
    Both ask here so the two cannot drift apart.
    """
    if not isinstance(node, ast.Subscript):
        return False
    index = node.slice
    if isinstance(index, ast.Slice):
        return True
    return isinstance(index, ast.Tuple) and any(isinstance(e, ast.Slice) for e in index.elts)


def _slot_accessor_group_name(value: ast.expr) -> str | None:
    """Group name for a slot accessor expression, or None if it is not one."""
    accessor = slot_accessor(value)
    return accessor[0] if accessor else None


def _get_slot_accessor_assignment(node: ast.AST, param_names: set[str]) -> tuple[str, str] | None:
    """Return slot variable and source buffer for ``slot = group.next()`` / ``slot = group[i]``."""
    if not isinstance(node, ast.Assign):
        return None
    if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
        return None

    group_name = _slot_accessor_group_name(node.value)
    if group_name is None or group_name not in param_names:
        return None
    return node.targets[0].id, group_name


def scan_kernel_slot_to_buffer(func_def: ast.FunctionDef, buffer_names: set[str]) -> dict[str, str]:
    """``{slot variable: buffer name}`` for ``slot = buf.next()/.current()/.previous()``
    taken in the KERNEL body and passed on as a stage argument.

    Covers EVERY declared buffer, not just the cross-core ones: an op's pipe is decided by
    the memory spaces of all its tiles, the local side of a ``pl.move`` included.
    """
    result: dict[str, str] = {}
    for node in ast.walk(func_def):
        acc = _get_slot_accessor_assignment(node, buffer_names)
        if acc is not None:
            slot_name, buffer_name = acc
            result[slot_name] = buffer_name
    return result


def build_binding_map(
    func_def: ast.FunctionDef,
    call_args: list | None,
    tables: AccessScanTables,
) -> dict[str, tuple[str, bool]]:
    """Build the name -> (buffer, is_group) binding map for a stage function body.

    Resolves all names that reference a declared tile group, directly or indirectly:
      (a) formal params bound by call-site position;
      (b) ``slot = group.next()`` in the body;
      (c) plain alias assignments ``tmp = known``;
      (d) ``grp = agg.field`` off an aggregate param the call site built with pl.make_tuple.

    (b)-(d) iterate until stable, so alias chains resolve.

    Args:
        func_def: the stage function AST.
        call_args: AST args from the call site. None returns empty — no name-collision
            guessing.
        tables: kernel-level lookup tables; only group_names, kernel_slot_to_buffer and
            tuple_fields are read here.

    Returns: {name: (buffer_declared_name, is_group)}
    """
    param_names = [a.arg for a in func_def.args.args if a.arg != "self"]
    result: dict[str, tuple[str, bool]] = {}
    # Aggregate params: {param name: {field: source variable}}. Kept apart from `result`
    # because an aggregate is not a buffer — it is only a route to one.
    aggregates: dict[str, dict[str, str]] = {}

    # (a) Param bindings from call site (positional). Without call args we cannot
    # resolve bindings — name-collision guessing is intentionally NOT done.
    if call_args is None:
        return result
    for pos, arg in enumerate(call_args):
        if pos >= len(param_names) or not isinstance(arg, ast.Name):
            continue
        actual = arg.id
        # A local group is bound under its real name just like a cross-core one: only cross-core
        # buffers get sync of their own, but an op's pipe follows the memory spaces of ALL its
        # tiles, so the local side of a `pl.move` needs a name that resolves.
        if actual in tables.group_names:
            result[param_names[pos]] = (actual, True)
        elif actual in tables.kernel_slot_to_buffer:
            result[param_names[pos]] = (tables.kernel_slot_to_buffer[actual], False)
        elif actual in tables.tuple_fields:
            aggregates[param_names[pos]] = tables.tuple_fields[actual]

    # (b)(c)(d) Propagate slot accessors (.next() / group[i]), aggregate member reads and
    # alias assignments until stable.
    changed = True
    while changed:
        changed = False
        for node in ast.walk(func_def):
            if not isinstance(node, ast.Assign):
                continue
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            target = node.targets[0].id
            if target in result:
                continue  # already resolved

            # (d) grp = agg.field -> the variable the call site put in that field. Recorded even for a
            # LOCAL group, which pipe resolution needs; not being cross-core keeps it out of the
            # access records.
            if isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name):
                fields = aggregates.get(node.value.value.id)
                if fields is not None:
                    source = fields.get(node.value.attr)
                    if source is not None:
                        result[target] = (source, True)
                        changed = True
                    continue

            if isinstance(node.value, (ast.Call, ast.Subscript)):
                source_name = _slot_accessor_group_name(node.value)
                if source_name is not None and source_name in result:
                    src_buf, src_is_group = result[source_name]
                    # A slot taken from a group, or a sub-view carved out of a tile: either way
                    # the result names memory belonging to src_buf.
                    if src_is_group or _is_subview(node.value):
                        result[target] = (src_buf, False)
                        changed = True
                continue

            if isinstance(node.value, ast.Name) and node.value.id in result:
                result[target] = result[node.value.id]
                changed = True

    return result


def _build_slot_to_buffer_from_bindings(
    func_def: ast.FunctionDef, bindings: dict[str, tuple[str, bool]]
) -> dict[str, str]:
    """This stage's ``name -> buffer`` map, covering ALL params (for pipe resolution).

    Built from build_binding_map's result, which has already re-keyed a kernel-body slot
    from the kernel's variable name to the formal parameter name — the key the stage body's
    ops are written against. On top of it come the slots taken inside this body.
    """
    stage_slot_to_buffer: dict[str, str] = {}
    # All resolved slot names (is_group=False) → buffer
    for name, (buf, is_group) in bindings.items():
        if not is_group:
            stage_slot_to_buffer[name] = buf
    # For pipe resolution: also scan .next() on non-cross-core params (local buffers).
    # Their resolved name is just the param name (standard pipe-table lookup).
    all_param_names = {a.arg for a in func_def.args.args if a.arg != "self"}
    for node in ast.walk(func_def):
        acc = _get_slot_accessor_assignment(node, all_param_names)
        if acc is not None:
            slot_name, param_name = acc
            if slot_name not in stage_slot_to_buffer:
                # Resolve: if param is in bindings use its buffer, else param name itself
                if param_name in bindings:
                    stage_slot_to_buffer[slot_name] = bindings[param_name][0]
                else:
                    stage_slot_to_buffer[slot_name] = param_name
    return stage_slot_to_buffer


def _scan_function(
    func_def: ast.FunctionDef, bindings: dict, tables: AccessScanTables, out: list, seen: set
) -> None:
    """Record every buffer access this function makes, in source order, into ``out``.

    Dispatches on what each call is:

        ``pl.<op>(...)``            a block op            -> _handle_block_op
        a ``@pl.vector_function``   one atomic op         -> _handle_vf_call
        any other plain function    part of THIS function -> recurse into it

    A helper a stage calls is the stage, factored out, so its accesses go into the same
    ``out``. (A nested STAGE would be its own node with its own sync; not supported yet.)

    ``bindings`` is this function's ``name -> (buffer, is_group)`` map, re-derived per
    recursion by argument position; ``seen`` holds the names on the current chain so a cycle
    terminates.

    Control flow is ignored: an op inside a branch is recorded like an unconditional one.
    The graph then syncs every pipe a buffer is touched on — an over-approximation, never a
    missing sync.
    """
    slot_to_buffer = _build_slot_to_buffer_from_bindings(func_def, bindings)
    _reject_tile_returning_function(func_def, bindings, slot_to_buffer)
    _scan_stmts(func_def.body, bindings, slot_to_buffer, tables, out, seen)


def _reject_tile_returning_function(func_def: ast.FunctionDef, bindings: dict, slot_to_buffer: dict) -> None:
    """A function a stage calls must not hand a tile or a tile group back to its caller.

    Buffers are followed by NAME — a parameter by argument position, a slot by its group. A
    return value has neither, so the caller binds it to a name that resolves to nothing and
    every op on it goes unrecorded and unsynchronised.

    Pass the buffer in as an argument instead; returning a scalar read out of one
    (``return t[i, j]``) is untouched by this.
    """
    for node in ast.walk(func_def):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        value = node.value
        held = None
        if isinstance(value, ast.Name) and (value.id in bindings or value.id in slot_to_buffer):
            held = value.id
        else:
            pick = slot_accessor(value)
            if pick is not None:
                group, kind = pick
                entry = bindings.get(group)
                # `.next()` only means one thing; `g[i]` is a slot pick when g is a GROUP and
                # an element read when it is a tile, which is a scalar and perfectly fine.
                if kind != "index" or (entry is not None and entry[1]):
                    held = group
        if held is not None:
            raise NotSupported(
                f"pipeline: '{func_def.name}' returns `{ast.unparse(value)}` (line "
                f"{node.lineno}), a tile or tile group. The scan follows a buffer by the "
                f"name it is bound to — a parameter, or a slot taken from one — and a "
                f"returned value arrives under a name of the caller's own that resolves to "
                f"nothing, so the ops the caller runs on it get no synchronisation.\n"
                f"Pass the buffer in as an argument instead.",
                span=span_of(node),
            )


def _scan_stmts(stmts: list, bindings: dict, slot_to_buffer: dict, tables: AccessScanTables,
                out: list, seen: set) -> None:
    """Record every buffer access these statements make, in source order, into ``out``.

    The dispatch half of _scan_function, split out because it works on a statement list
    while the ``name -> buffer`` map it needs is derived from a whole function. The pipeline
    loop's own body is such a list and must be scanned by exactly this logic.
    """
    for stmt in stmts:
        for node, is_aug_target in _iter_accessors_in_order(stmt):
            if isinstance(node, ast.Subscript):
                _handle_tile_subscript(node, is_aug_target, slot_to_buffer, tables, out)
                continue
            op_name = resolve_op_name(node.func, tables.closure_vars)
            if op_name is not None and _vf_op_of(op_name) is None:
                _handle_block_op(node, op_name.rsplit(".", 1)[-1], slot_to_buffer, tables, out)
                continue
            name = call_name(node)
            if not name:
                continue
            if name in tables.vf_func_defs:
                _handle_vf_call(node, slot_to_buffer, tables, out)
                continue
            callee = tables.helper_func_defs.get(name)
            # Not a known function (a stage, or a name this kernel's scope cannot resolve),
            # or already on this chain.
            if callee is None or name in seen:
                continue
            _reject_tile_group_decl(callee, name)
            _scan_function(
                callee,
                _bind_callee_params(callee, node.args, bindings, slot_to_buffer),
                tables,
                out,
                seen | {name},
            )


def reject_buffer_access_outside_stages(loop_body: list, tables: AccessScanTables, group_names: set) -> None:
    """Refuse a statement in the pipeline loop's own body that touches a tracked buffer.

    Only a stage's accesses are scanned, and the wait/set the transform emits sit around a
    stage call, so such a statement would get no sync at all — and it is not replayed in the
    drain either. Silently wrong, hence refused.

    Tracked means what the sync graph tracks: a cross-core buffer, or one sharing its
    address. Work on a purely local buffer is left alone, so an unrelated store can stay out
    of a stage on purpose.

    Stage calls need no special case: a stage is in neither vf_func_defs nor
    helper_func_defs, so the scan walks straight past one.
    """
    bindings = {name: (name, True) for name in group_names}
    bindings.update({var: (buf, False) for var, buf in tables.kernel_slot_to_buffer.items()})
    for stmt in loop_body:
        touched: list = []
        _scan_stmts([stmt], bindings, dict(tables.kernel_slot_to_buffer), tables, touched, set())
        if not touched:
            continue
        buffers = sorted({access[0] for access in touched})
        raise InvalidOperation(
            f"pipeline: the statement at line {stmt.lineno} touches buffer(s) {buffers} that "
            f"cross-core sync tracks, but it is not inside a stage. Only a stage's accesses "
            f"are scanned, and the wait/set the transform emits sit around the stage call, "
            f"so this access would get no synchronisation at all.\n"
            f"Move it into a stage. Work on a buffer that no cross-core handover involves "
            f"may stay here.",
            span=span_of(stmt),
        )


def _bind_callee_params(callee: ast.FunctionDef, call_args: list, bindings: dict, slot_to_buffer: dict) -> dict:
    """The callee's ``name -> (buffer, is_group)`` map, from the caller's by argument position.

    Same shape as build_binding_map, but resolves a call site inside a function against what
    that function already knows. ``slot_to_buffer`` is consulted too, since it holds the
    names the accessor scan added on top of ``bindings``.
    """
    params = [a.arg for a in callee.args.args if a.arg != "self"]
    result: dict[str, tuple[str, bool]] = {}
    for pos, arg in enumerate(call_args):
        if pos >= len(params) or not isinstance(arg, ast.Name):
            continue
        if arg.id in bindings:
            result[params[pos]] = bindings[arg.id]
        elif arg.id in slot_to_buffer:
            result[params[pos]] = (slot_to_buffer[arg.id], False)
    return result


def _reject_tile_group_decl(func_def: ast.FunctionDef, name: str) -> None:
    """A tile group must be declared in the kernel body, not in a function it calls.

    Every table the transform keys on is indexed by the variable name the declaration binds,
    and scan_tile_group_decls fills them from the kernel body alone — so a group declared in
    a helper is in none of them and its accesses are silently dropped.
    """
    for node in ast.walk(func_def):
        if isinstance(node, ast.Call) and _is_make_tile_group(node):
            raise InvalidOperation(
                f"pipeline: '{name}' declares a tile group with pl.make_tile_group(), but it "
                f"is a function called from a stage. Tile groups must be declared in the "
                f"kernel body — the transform identifies every buffer by the name its "
                f"declaration binds there, so one declared here is invisible to the sync it "
                f"needs.\nMove the declaration into the kernel and pass the group in.",
                span=span_of(func_def),
            )


def _block_op_roles(node: ast.Call, op_name: str) -> list | None:
    """The argument roles for this call, or None when the table does not cover the op.

    An op whose ``mode=`` changes what it reads and writes has one entry per mode, named
    ``<op>_<mode lowercased>`` (e.g. ``fillpad_inplace``, ``fillpad_expand``). A new
    mode-dependent op then needs a table entry and nothing here.
    """
    for kw in node.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Attribute):
            variant = _BLOCK_OP_TILE_ROLES.get(f"{op_name}_{kw.value.attr.lower()}")
            if variant is not None:
                return variant
    return _BLOCK_OP_TILE_ROLES.get(op_name)


def _reject_untabled_op(node: ast.Call, op_name: str, stage_slot_to_buffer: dict, tables) -> None:
    """Refuse an op that touches a tracked buffer but has no roles in the table.

    The hand-written table runs behind the framework's op set, and reaching an uncovered op
    used to mean recording nothing — a missing wait/set with nothing to announce it.

    Refusing only when a resolved argument is a buffer the graph tracks keeps this quiet for
    ops on purely local tiles. There is no safe default: guessing "read" drops the producer's
    set, guessing "write" drops the consumer's wait.
    """
    touched = []
    for arg in node.args:
        buf = _tile_arg_buffer(arg, stage_slot_to_buffer)
        if tables.tracks(buf):
            touched.append(buf)
    if not touched:
        return
    likely = ["W"] + ["R"] * (len(node.args) - 1)
    raise InvalidArgument(
        f"pipeline: `pl.{op_name}` at line {node.lineno} touches buffer(s) "
        f"{sorted(set(touched))} that cross-core sync tracks, but _BLOCK_OP_TILE_ROLES has "
        f"no entry for it, so the scan cannot tell which arguments it reads and which it "
        f"writes — and which way a wait/set goes depends on exactly that.\n"
        f"Add an entry in pypto_pro/language/parser/_op_pipeline.py. With the "
        f"{len(node.args)} positional argument(s) here the roles are probably {likely}, but "
        f"check the two things a signature cannot show: an argument the op accumulates into "
        f"is 'RW', and a position that is not a tile is None.",
        span=span_of(node),
    )


def _handle_block_op(
    node: ast.Call, op_name: str, stage_slot_to_buffer: dict, tables: AccessScanTables, out: list
) -> None:
    """Record a single block-op call. ``op_name`` is resolved by the caller (resolve_op_name)."""
    # Descriptor-only ops (set_validshape and the like) rewrite a tile's metadata and never
    # touch its data, so they can neither race nor need a handover; the parser already keeps
    # that set for auto_mutex. Asking before the role lookup also keeps _reject_untabled_op
    # honest: an op absent from the role table is then genuinely unclassified.
    if not op_accesses_buffer(op_name):
        return
    roles = _block_op_roles(node, op_name)
    if roles is None:
        _reject_untabled_op(node, op_name, stage_slot_to_buffer, tables)
        return
    # One record per buffer, not per argument: an op is a single point in time, so a buffer it
    # both reads and writes is one RW access. The pipe is a property of the op, so merging
    # cannot lose one.
    merged: dict[str, str] = {}
    pipe = None
    for argpos, arg in enumerate(node.args):
        buf = _tile_arg_buffer(arg, stage_slot_to_buffer)
        if buf is None or argpos >= len(roles):
            continue
        role = roles[argpos]
        if role is None:
            continue
        if not tables.tracks(buf):
            continue
        if pipe is None:
            pipe = _block_op_pipe(op_name, node, stage_slot_to_buffer, tables.all_buffer_memory)
        _merge_role(merged, buf, role)
    for buf, role in merged.items():
        _record_access(out, buf, role, pipe)


def _handle_tile_subscript(
    node: ast.Subscript, is_aug_target: bool, stage_slot_to_buffer: dict,
    tables: AccessScanTables, out: list
) -> None:
    """Record ``tile[i, j]`` / ``tile[i, j] = v`` — one scalar element, on the S pipe.

    Non-tuple indices are skipped: the parser requires the index count to equal the
    container's rank, so those shapes do not compile in the first place.

    A sub-view (a slice among the indices) is not an element access — see _is_subview — and
    a group subscript (``group[i]``) selects a slot, which needs no test since only tiles
    reach ``stage_slot_to_buffer``.
    """
    if not isinstance(node.value, ast.Name):
        return
    if _is_subview(node) or not isinstance(node.slice, ast.Tuple):
        return
    buf = stage_slot_to_buffer.get(node.value.id)
    if not tables.tracks(buf):
        return
    if is_aug_target:
        role, pipe_op = "RW", "setval"
    elif isinstance(node.ctx, ast.Store):
        role, pipe_op = "W", "setval"
    else:
        role, pipe_op = "R", "getval"
    _record_access(out, buf, role, _pipe_name(get_op_pipe(pipe_op)))


def _handle_vf_call(node: ast.Call, stage_slot_to_buffer: dict, tables: AccessScanTables, out: list) -> None:
    """Record a single VF helper call, always on pipe V."""
    vf_name = node.func.id
    vf_def = tables.vf_func_defs.get(vf_name)
    if vf_def is None:
        return
    vf_params = [a.arg for a in vf_def.args.args if a.arg != "self"]
    # Which parameters hold a tracked buffer is settled here, where the bindings are, and
    # handed to the scan; it has no way to tell a pointer parameter from a scalar one.
    tile_params: dict[str, str] = {}
    for argpos, arg in enumerate(node.args):
        buf = _tile_arg_buffer(arg, stage_slot_to_buffer)
        if buf is None or argpos >= len(vf_params):
            continue
        if tables.tracks(buf):
            tile_params[vf_params[argpos]] = buf
    if not tile_params:
        return
    vf_roles = _scan_vf_roles(vf_def, tables.vf_func_defs, tile_params, tables.closure_vars)
    # Merged per buffer for the same reason as a block op: one call is one access, even
    # when the same buffer arrives at two parameters with different roles.
    merged: dict[str, str] = {}
    for param, role in vf_roles.items():
        _merge_role(merged, tile_params[param], role)
    for buf, role in merged.items():
        _record_access(out, buf, role, "V")


def _iter_accessors_in_order(node: ast.AST):
    """Yield the ast.Call and ast.Subscript nodes under ``node``, in execution order.

    Subscripts come along because ``t[i]`` and ``t[i] = v`` ARE accesses — the parser lowers
    them to getval/setval.

    An assignment yields its value before its targets, the order the machine runs them in.
    It matters for ``t[i, j] = t[k, l]``: recording the write first would let the read find
    that write as its nearest preceding one.

    Each node is paired with a flag saying whether it is an augmented assignment's target.
    ``t[i, j] += v`` reads AND writes that element, but the tree scanned here still carries
    a lone Store.
    """
    if isinstance(node, ast.Assign):
        children = [(node.value, False), *((t, False) for t in node.targets)]
    elif isinstance(node, ast.AugAssign):
        children = [(node.value, False), (node.target, True)]
    else:
        children = [(child, False) for child in ast.iter_child_nodes(node)]
    for child, is_aug_target in children:
        if isinstance(child, (ast.Call, ast.Subscript)):
            yield child, is_aug_target
        yield from _iter_accessors_in_order(child)


def scan_stage_accesses(
    stage_func_def: ast.FunctionDef,
    call_args: list | None,
    tables: AccessScanTables,
) -> list:
    """One stage's buffer accesses, ``[(buffer, role, pipe), ...]`` one entry per op.

    Control flow is ignored (see _scan_function).

    Args:
        stage_func_def: the @stage function AST.
        call_args: AST args from the call site. None means the bindings cannot be resolved
            and the scan yields nothing — name-collision guessing is not done.
        tables: the kernel-level lookup tables (see AccessScanTables).
    """
    bindings = build_binding_map(stage_func_def, call_args, tables)
    if not bindings:
        return []

    group_param_names = {p for p, (_b, ig) in bindings.items() if ig}
    _validate_slot_accessors(stage_func_def, group_param_names)

    accesses: list = []
    _scan_function(stage_func_def, bindings, tables, accesses, {stage_func_def.name})
    return accesses


def _validate_slot_accessors(stage_func_def: ast.FunctionDef, group_param_names: set[str]) -> None:
    """L10: cross-core buffer slot accessors must be `slot = param.next()` / `slot = param[i]` form.

    group_param_names: formal params that carry a cross-core buffer GROUP (the ones
    the body calls .next()/.current()/.previous() on, or subscripts)."""
    accessor_rhs_ids = set()
    for node in ast.walk(stage_func_def):
        sa = _get_slot_accessor_assignment(node, group_param_names)
        if sa is not None:
            accessor_rhs_ids.add(id(node.value))
    for node in ast.walk(stage_func_def):
        if not isinstance(node, (ast.Call, ast.Subscript)):
            continue
        group_name = _slot_accessor_group_name(node)
        if group_name is None or group_name not in group_param_names:
            continue
        if id(node) in accessor_rhs_ids:
            continue
        accessor = "[...]" if isinstance(node, ast.Subscript) else f".{node.func.attr}()"
        raise NotSupported(
            f"pipeline: cross-core buffer group '{group_name}' slot accessor "
            f"`{accessor}` must be assigned to a simple variable "
            f"(`slot = {group_name}{accessor}`); inline/chained/"
            f"tuple-unpack forms are not supported.",
            span=span_of(stage_func_def),
        )


def scan_tuple_fields(kernel_func_def: ast.FunctionDef) -> dict[str, dict[str, str]]:
    """``{tuple variable: {field name: the variable it was built from}}`` for each
    ``x = pl.make_tuple(field=var, ...)`` in the kernel body.

    Reading a member back as ``agg.field`` breaks the chain from a tile to its declared
    group; this records the one hop that rejoins it — rule (d) in build_binding_map. Only
    keyword members bound to a plain name are traceable, so only those are recorded.
    """
    result: dict[str, dict[str, str]] = {}
    for node in ast.walk(kernel_func_def):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            continue
        if _get_ctor_name(node.value) != "make_tuple":
            continue
        fields = {kw.arg: kw.value.id for kw in node.value.keywords if kw.arg and isinstance(kw.value, ast.Name)}
        if not fields:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                result[target.id] = fields
    return result


def _get_ctor_name(call: ast.Call) -> str | None:
    """Get the constructor name from a call like pl.UBNBuffer(...) -> 'UBNBuffer'."""
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def _const_int(node: ast.expr) -> int | None:
    """The integer a literal element denotes, or None if it is not one.

    A negative literal parses as ``UnaryOp(USub, Constant)``, so matching only Constant
    would read ``-1`` as "not an integer" instead of as the out-of-range id it is.
    """
    sign = 1
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        node, sign = node.operand, -1
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return sign * node.value
    return None


def _resolve_event_ids(node: ast.expr, bufname: str, label: str, closure_vars: dict) -> list[int]:
    """The integer event ids behind an ``fwd_ids=`` / ``bwd_ids=`` node.

    Exactly two spellings are supported::

        fwd_ids=[0, 1]          # a literal list (or tuple)
        IDS = [0, 1]
        fwd_ids=IDS             # a name bound to one outside the kernel

    Every element must be a compile-time integer: the list is emitted VERBATIM ahead of the
    loop as ``_pl_fwd_ids_<buffer> = [...]`` and only then subscripted at runtime, so an
    element this scan cannot evaluate is one whose meaning in the generated kernel is
    anyone's guess. Anything else is refused rather than half-supported.
    """
    hint = (
        f"Write it as a literal list of integers (`{label}=[0, 1]`), or as a name bound to "
        f"one outside the kernel (`IDS = [0, 1]` ... `{label}=IDS`)."
    )
    if isinstance(node, (ast.List, ast.Tuple)):
        ids = []
        for element in node.elts:
            value = _const_int(element)
            if value is None:
                raise InvalidVal(
                    f"pipeline: cross-core buffer '{bufname}' {label} contains "
                    f"`{ast.unparse(element)}`, which is not a compile-time integer. {hint}",
                    span=span_of(element),
                )
            ids.append(value)
        return ids

    if isinstance(node, ast.Name):
        value = closure_vars.get(node.id)
        if not isinstance(value, (list, tuple)):
            raise InvalidOperation(
                f"pipeline: cross-core buffer '{bufname}' {label} is `{node.id}`, which is "
                f"not bound to a list or tuple of integers outside the kernel. {hint}",
                span=span_of(node),
            )
        bad = [v for v in value if not isinstance(v, int) or isinstance(v, bool)]
        if bad:
            raise InvalidType(
                f"pipeline: cross-core buffer '{bufname}' {label} is `{node.id}` = {value!r}, "
                f"which holds non-integer element(s) {bad}. {hint}",
                span=span_of(node),
            )
        return list(value)

    raise NotSupported(
        f"pipeline: cross-core buffer '{bufname}' {label} is `{ast.unparse(node)}`, which is "
        f"not a supported spelling. {hint}",
        span=span_of(node),
    )


def _tile_arg_buffer(arg: ast.expr, stage_slot_to_buffer: dict[str, str]) -> str | None:
    """The buffer behind a bare slot variable argument, or None.

    ``group.next()`` returns a bare tile, so op args are plain Names (``pl.move(left, cur_k)``).
    """
    if isinstance(arg, ast.Name):
        return stage_slot_to_buffer.get(arg.id)
    return None


def _block_op_pipe(
    op_name: str, call: ast.Call, stage_slot_to_buffer: dict[str, str], all_buffer_memory: dict[str, MemorySpace]
) -> str:
    """Determine the pipe name for a block op accessing a cross-core buffer.

    Raises when the pipe cannot be determined. There is no safe default: the pipe decides
    which queue the wait/set lands on, and a section only has some of the pipes, so a guess
    can even name one that does not exist on that core.
    """
    if op_name in ("move", "insert"):
        # move(dst, src): pipe depends on src/dst memory
        # Tile-to-tile data movement uses a pipe selected by src/dst memory.
        dst_mem = _arg_memory(call.args[0] if call.args else None, stage_slot_to_buffer, all_buffer_memory)
        src_mem = _arg_memory(call.args[1] if len(call.args) > 1 else None, stage_slot_to_buffer, all_buffer_memory)
        if src_mem is None or dst_mem is None:
            unresolved = "destination" if dst_mem is None else "source"
            raise InvalidOperation(
                f"pipeline: cannot determine the pipe of `pl.{op_name}` at line {call.lineno}, "
                f"because its {unresolved} tile does not resolve to a declared tile group. "
                f"The move touches a cross-core buffer, so its pipe decides where the sync "
                f"goes. A tile reached through an aggregate (e.g. `tile_groups.x.next()`) is "
                f"not yet traced; pass the tile group to the stage as its own argument.",
                span=span_of(call),
            )
        return _pipe_name(get_move_pipe(src_mem, dst_mem))
    if op_name in ("store", "store_tile"):
        src_mem = _arg_memory(call.args[1] if len(call.args) > 1 else None, stage_slot_to_buffer, all_buffer_memory)
        if src_mem is None:
            raise InvalidOperation(
                f"pipeline: cannot determine the pipe of `pl.{op_name}` at line {call.lineno}, "
                f"because its source tile does not resolve to a declared tile group. The store "
                f"touches a cross-core buffer, so its pipe decides where the sync goes. A tile "
                f"reached through an aggregate (e.g. `tile_groups.x.next()`) is not yet traced; "
                f"pass the tile group to the stage as its own argument.",
                span=span_of(call),
            )
        return _pipe_name(get_store_pipe(src_mem))
    pipe = get_op_pipe(op_name)
    if pipe is None:
        raise InvalidOperation(
            f"pipeline: `pl.{op_name}` at line {call.lineno} touches a cross-core buffer, but "
            f"no pipe is registered for it, so the sync has nowhere to go. Register the op's "
            f"pipe (see get_op_pipe) or keep the cross-core buffer out of this op.",
            span=span_of(call),
        )
    return _pipe_name(pipe)


def _arg_memory(arg, stage_slot_to_buffer, all_buffer_memory) -> MemorySpace | None:
    """Get the MemorySpace of a `slot.tile` arg (any buffer, cross-core or local)."""
    buf = _tile_arg_buffer(arg, stage_slot_to_buffer) if arg is not None else None
    if buf is not None:
        return all_buffer_memory.get(buf)
    return None


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
    out.append((buf, role, pipe))


def _root_name(node):
    """Return the root name of a possibly indexed/attributed expression."""
    cur = node
    while True:
        if isinstance(cur, ast.Name):
            return cur.id
        if isinstance(cur, ast.BinOp):
            cur = cur.left
            continue
        if isinstance(cur, (ast.Attribute, ast.Subscript)):
            cur = cur.value
            continue
        return None


def _merge_role(result: dict[str, str], key: str, role: str | None) -> None:
    """Fold one more role for ``key`` into ``result``: differing roles become "RW".

    Used for a VF's parameters and for one block op's arguments alike. In the latter,
    differing roles mean the same buffer reached the op at both a read and a write position
    (``pl.muls(x, x, 2.0)``) — ONE access, not two.
    """
    existing = result.get(key)
    if existing is None:
        result[key] = role
    elif existing != role and role is not None:
        result[key] = "RW"


def _vf_param_aliases(vf_func_def: ast.FunctionDef, param_names: set[str]) -> dict[str, str]:
    """Local names in a VF body that stand for one of its parameters.

    A VF reaches UB only through its parameters but rarely names them at the point of use.
    Two spellings cover it: ``src = input_tile`` and ``src = input_tile + OFF`` — a rename,
    or a rename at an offset. Chains resolve by iterating to a fixed point.
    """
    aliases: dict[str, str] = {}
    changed = True
    while changed:
        changed = False
        for node in ast.walk(vf_func_def):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            if not isinstance(node.targets[0], ast.Name):
                continue
            target = node.targets[0].id
            if target in param_names or target in aliases:
                continue
            # _root_name walks a BinOp down its left side, so `param + off` and a bare
            # rename land on the same answer.
            if not isinstance(node.value, (ast.Name, ast.BinOp)):
                continue
            source = _root_name(node.value)
            if source in param_names:
                aliases[target] = source
                changed = True
            elif source in aliases:
                aliases[target] = aliases[source]
                changed = True
    return aliases


def _vf_pointer_param(arg: ast.expr, param_names: set[str], aliases: dict[str, str]) -> str | None:
    """The VF parameter this pointer argument refers to, directly or through an alias."""
    root = _root_name(arg)
    if root is None:
        return None
    if root in param_names:
        return root
    return aliases.get(root)


def resolve_op_name(func: ast.expr, closure_vars: dict) -> str | None:
    """The op a callee names, or None if it names no op.

        vf.add / Vf.add / pl.Vf.add / a bare `add` imported from Vf  -> "vf.add"
        pl.move / any other alias for the language module            -> "move"
        pl.system.bar_all                                            -> "system.bar_all"

    Mirrors the parser's own resolution (``_call_parser._extract_op_name``): the leading
    name is the user's import alias and does not identify the op, so a module is recognised
    by IDENTITY against the API objects rather than by the name it was bound to. Keep the
    two in step — a spelling the parser accepts but this does not is an access the scan
    never records, and an access nobody records is a handover nobody synchronises.
    """
    if isinstance(func, ast.Name):
        resolved = closure_vars.get(func.id)
        name = getattr(resolved, "__name__", None)
        if not isinstance(name, str):
            return None
        if getattr(Vf, name, None) is resolved:
            return f"vf.{name}"
        if getattr(_language_api, name, None) is resolved:
            return name
        return None

    attrs: list[str] = []
    node: ast.expr = func
    while isinstance(node, ast.Attribute):
        attrs.insert(0, node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        attrs.insert(0, node.id)
    if len(attrs) < 2:
        return None
    if attrs[0] == "vf" or closure_vars.get(attrs[0]) is Vf:
        return ".".join(["vf", *attrs[1:]])
    if attrs[1] == "Vf":
        return ".".join(["vf", *attrs[2:]])
    return ".".join(attrs[1:])


def _vf_op_of(name: str | None) -> str | None:
    """The vf op ``name`` stands for, or None if it stands for something else."""
    return name[3:] if name is not None and name.startswith("vf.") else None


def _nested_vf_def(call: ast.Call, vf_func_defs: dict) -> ast.FunctionDef | None:
    """The vector function this call invokes, or None if it does not invoke one.

    Membership of ``vf_func_defs`` is the test, and that set is built from the decorator.
    """
    if not isinstance(call.func, ast.Name):
        return None
    return vf_func_defs.get(call.func.id)


def _record_vf_call_role(
    call: ast.Call, op: str, vf_name: str, param_names: set[str], aliases: dict[str, str],
    tile_params: dict[str, str], result: dict[str, str]
) -> None:
    """Record what one ``vf.<op>(...)`` call does to the tiles it was handed.

    Argument roles come from _VF_OP_TILE_ROLES rather than from the op name: `gather` reads
    a tile, `scatter` writes one, and the unaligned loads keep their alignment-state register
    where every other op keeps the pointer.

    Only arguments naming a parameter in ``tile_params`` are considered — a VF reaches UB
    solely through its parameters, and the rest hold scalars at this call site. That is what
    makes refusing an untabled op safe to do on the spot.

    ``call`` must be a vf command; the caller decides that (see resolve_op_name).
    """
    roles = _VF_OP_TILE_ROLES.get(op)
    for argpos, arg in enumerate(call.args):
        param = _vf_pointer_param(arg, param_names, aliases)
        if param is None or param not in tile_params:
            continue
        if roles is None:
            raise NotSupported(
                f"pipeline: `vf.{op}` at line {call.lineno}, in vector function "
                f"'{vf_name}', is handed buffer '{tile_params[param]}' (as '{param}'), but "
                f"_VF_OP_TILE_ROLES has no entry for it, so the scan cannot tell whether it "
                f"reads or writes the tile — and which way a cross-core wait/set goes "
                f"depends on that.\n"
                f"If the op touches a UB tile, add it to _VF_OP_TILE_ROLES in "
                f"pypto_pro/language/parser/_op_pipeline.py, giving each argument position "
                f"'R', 'W', 'RW', or None for the positions that are not tiles.",
                span=span_of(call),
            )
        if argpos < len(roles) and roles[argpos] is not None:
            _merge_role(result, param, roles[argpos])


def _scan_vf_roles(
    vf_func_def: ast.FunctionDef, vf_func_defs: dict, tile_params: dict[str, str],
    closure_vars: dict, path: tuple = ()
) -> dict[str, str]:
    """R/W roles for the parameters of ``vf_func_def`` that hold a tile at this call site.

    ``tile_params`` maps such a parameter to the buffer it was handed. Which parameters are
    tiles is nowhere written in a VF, but the caller resolved exactly that before getting
    here.

    A VF called from another VF is part of it, the way a plain helper is part of its stage:
    its effect folds into the caller's parameters by argument position, which is the whole
    of the folding, and the same mapping carries ``tile_params`` inward. ``path`` holds the
    VFs already on this chain, so A -> B -> A terminates.
    """
    param_names = {a.arg for a in vf_func_def.args.args if a.arg != "self"}
    aliases = _vf_param_aliases(vf_func_def, param_names)
    result: dict[str, str] = {}
    path = path + (vf_func_def.name,)

    # A call in a VF body is one of a closed set:
    #
    #   vf.<op>(...)        one machine op              -> _record_vf_call_role
    #   another VF          part of THIS one            -> recurse
    #   pl.range / pl.min   loop and scalar scaffolding -> nothing to record
    #
    # Nothing else can appear: the parser refuses a vector function that calls an ordinary
    # helper.
    for node in ast.walk(vf_func_def):
        if not isinstance(node, ast.Call):
            continue
        vf_op = _vf_op_of(resolve_op_name(node.func, closure_vars))
        if vf_op is not None:
            _record_vf_call_role(node, vf_op, vf_func_def.name, param_names, aliases, tile_params, result)
            continue
        callee_def = _nested_vf_def(node, vf_func_defs)
        if callee_def is None or callee_def.name in path:
            continue
        callee_params = [a.arg for a in callee_def.args.args if a.arg != "self"]
        # Which of the callee's parameters are tiles follows from which of ours are.
        callee_tiles: dict[str, str] = {}
        outer_of: dict[str, str] = {}
        for argpos, arg in enumerate(node.args):
            if argpos >= len(callee_params):
                break
            param = _vf_pointer_param(arg, param_names, aliases)
            if param is None or param not in tile_params:
                continue
            callee_tiles[callee_params[argpos]] = tile_params[param]
            outer_of[callee_params[argpos]] = param
        if not callee_tiles:
            continue
        callee_roles = _scan_vf_roles(callee_def, vf_func_defs, callee_tiles, closure_vars, path)
        for callee_param, role in callee_roles.items():
            _merge_role(result, outer_of[callee_param], role)
    return result
