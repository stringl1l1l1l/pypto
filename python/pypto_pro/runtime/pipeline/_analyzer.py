# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Analyzer: extract pipeline structure from serial kernel AST."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import NamedTuple

from pypto.pypto_impl.ir import SectionKind

from ..._errors import (
    InvalidArgument,
    InvalidOperation,
    InvalidType,
    KeyNotFound,
    NotSupported,
    PyptoProWarning,
    _warn_located,
    current_span,
    span_of,
)
from ...language.parser._buffer_parser import SelectorKind
from ...language.parser._expr_evaluator import ExprEvaluator
from ._astutil import (
    PL_IS_VALID_FIELD,
    PL_STRUCT_ARG,
    PL_TASK_ID_FIELD,
    call_name,
    get_funcdef,
    is_vf_function,
    slot_accessor,
    slot_index_field,
    spans_in_function,
    stmt_header,
)
from ._cross_core_scanner import (
    AccessTables,
    CrossCoreSyncContext,
    buffer_addr_ranges,
    buffer_memory_spaces,
    buffer_slot_counts,
    cross_core_buffers,
    detect_addr_overlaps,
    reject_buffer_access_outside_stages,
    stage_accesses,
    tile_group_names,
    validate_bound_buffer_decls,
    validate_distinct_buffer_names,
)
from ._stage import is_pipeline_stage
from ._validate import (
    reject_buffer_decls_in_loops,
    validate_kernel_buffers,
    validate_kernel_structure,
    validate_slot_picks,
    validate_structure,
)
from .config import PipelineConfig


@dataclass(frozen=True)
class ArgTypeFacts:
    """The type an argument of a kernel-body call carries, as the probe parse saw it.

    One record per argument rather than a table per side of the type: both sides are keyed by
    the same argument position and filled from the same parsed value, and a reader asking
    about a struct's fields is asking about that argument's type too.
    """

    # The IR type's class name.
    kind: str
    # The scalar dtype, when ``kind`` is "ScalarType".
    dtype: object | None = None
    # {field name: IR type} when the argument is a struct, None when it is not one.
    fields: dict | None = None


@dataclass
class KernelFacts:
    """What one probe parse learns about the kernel (see probe_kernel_facts).

    One object rather than a tuple of its own: a new fact is added here and read where it is
    needed, without threading another argument through transform_pipeline and analyze_pipeline.
    """

    # {(lineno, col_offset) of an `if` test: (is_const, value)}, for dead-branch pruning.
    if_const_map: dict = field(default_factory=dict)
    # {name: (IR type class name, scalar dtype or None)}
    var_types: dict = field(default_factory=dict)
    # {(lineno, col_offset) of an argument a kernel-body call writes: ArgTypeFacts}
    arg_types: dict = field(default_factory=dict)
    # {kernel-body variable holding a buffer: the declared names it reaches}. More than one
    # name when control flow merges two declarations. Only the GM slot selection reads this
    # today; the tile-group slot paths still match on the name as written.
    buffer_vars: dict = field(default_factory=dict)
    # {target: {(lineno, col_offset) of a kernel-body node: [StageOpAccess, ...]}}. Per target
    # because a parse expands only the sections belonging to the one it ran for.
    op_accesses: dict = field(default_factory=dict)
    # {declared name: TileGroupFacts | TensorFacts}, in source order. Merged over the targets:
    # a declaration inside a section is only parsed by the target that section belongs to.
    buffer_facts: dict = field(default_factory=dict)
    # Declarations no kernel-body variable holds, which the pipeline refuses.
    unbound_decls: list = field(default_factory=list)
    # [(name, declaration, another declaration bound to the same name)], which the pipeline
    # refuses: buffer_facts keeps one declaration per name.
    same_name_decls: list = field(default_factory=list)


class OperandFact(NamedTuple):
    """One op argument that refers to a buffer: its binding and slot selectors."""

    position: int | None
    keyword: str | None
    names: tuple  # declared buffer names this argument reaches
    selectors: tuple = ()  # ``(declared name, SlotSelector)`` for slot-selected refs


@dataclass
class StageOpAccess:
    """One op a parse emitted, with every buffer operand resolved to its declared name."""

    op_name: str  # "move", "getval", "vf.load_align", ...
    operands: tuple  # OperandFact per argument that reaches a buffer
    keyword_names: tuple  # every keyword the call wrote
    in_vf: bool  # whether the op is inside a vector function
    span: object


@dataclass
class StageCall:
    """Info about a single stage call in the serial loop body."""

    func_name: str  # e.g. "compute_qk"
    section_kind: str  # "cube" or "vector"
    args: list  # list of ast.expr nodes (call arguments)
    delay: int  # derived from call order: 0, 1, 2, ...
    # The stage's own AST, resolved once in analyze_pipeline (re-reading the source is not
    # cheap, ~0.7 ms). None when the source is unavailable; every reader checks.
    func_def: ast.FunctionDef | None = None
    # The statement this stage was found in — the ``with pl.section_*()`` block, or the bare
    # call when there is no section. The transform looks up WHICH statement holds a stage
    # here rather than deciding again from its shape, which a section holding anything else
    # would throw out of step.
    source_stmt: ast.stmt | None = None
    # The call itself, inside source_stmt. Errors about the stage point here, so that a
    # stage wrapped in a section block is reported at its call rather than at the `with`.
    call_node: ast.Call | None = None
    # Every buffer access this stage makes, one entry per op in source order:
    # [(buffer, role, pipe), ...]. Per-op rather than collapsed to first/last pipe, so the
    # sync graph can see a local access sitting between two cross-core ones, and so every
    # pipe a buffer is touched on gets its own node (and hence its own sync).
    buffer_access: list = field(default_factory=list)


@dataclass
class PipelineInfo:
    """One pipeline: the stages of a single loop, plus the kernel facts they need.

    A kernel may hold several pipeline loops; analyze_pipeline returns one of these per loop.
    Which group a field is in decides whether a second loop gets its own copy:

      per-loop   stages, pipeline_loop*, struct_args, outer_slots, ctx_fields, ctx_sources,
                 scalar_ctx_names, stage_arg_mapping — everything derived FROM the stages.
      kernel     sync, group_names, var_sections, closure_vars, var_types — declarations and
                 scope, SHARED BY REFERENCE with every other loop's info. The event-id
                 bookkeeping in sync has to be one object.

    Sharing them is what keeps the rest of the module unchanged: every pass reads
    ``info.<field>`` and sees one complete pipeline, whether the kernel holds one or five.
    """
    # Which pipeline this is, in source order. Generated names carry it, so two loops'
    # ctx arrays and counters cannot collide (loop 0 keeps the bare names).
    loop_index: int = 0
    stages: list[StageCall] = field(default_factory=list)
    ctx_fields: list[str] = field(default_factory=list)
    # {ctx field: (fill expression, {sections to emit the fill in}, {section kind: the
    # initializer it is declared with there})}. ctx_fields comes from its keys, so a field
    # cannot be declared without a source.
    ctx_sources: dict = field(default_factory=dict)
    # {struct variable: [field names]} per pl.struct passed to a stage. Its fields become ctx
    # fields under their own names, so a stage body reading `ri.ki` needs no rewriting.
    struct_args: dict = field(default_factory=dict)
    # {variable: ctx field name} for scalars. Identical unless the name collides with a
    # struct field, whose name is fixed by the stage body, so the scalar yields.
    scalar_ctx_names: dict = field(default_factory=dict)
    # Maps: stage arg position -> ctx field name | None (the argument passes through).
    stage_arg_mapping: list[list] = field(default_factory=list)
    # The loop the stages are pipelined over, and its bounds.
    pipeline_loop_var: str = ""  # the loop variable (e.g. "ki")
    pipeline_loop_end: ast.expr | None = None  # e.g. ast node for "skv_tiles"
    # None means an implicit 1. The variable is in the user's own unit, so "n iterations
    # later" is `var + n * step`.
    pipeline_loop_step: ast.expr | None = None
    # {variable: (IR type class name, scalar dtype or None)} from the parser probe.
    var_types: dict = field(default_factory=dict)
    # {(lineno, col_offset) of a call argument: ArgTypeFacts}.
    arg_types: dict = field(default_factory=dict)
    # {kernel-body variable holding a buffer: the declared names it reaches}, from the probe.
    buffer_vars: dict = field(default_factory=dict)
    # {struct variable: the pl.struct() call declaring it}, so a field an error rejects can be
    # pointed at.
    struct_defs: dict = field(default_factory=dict)
    # {name: {section kinds it is bound in}}. A name bound in a section is invisible to the
    # other target, so its ctx fill has to sit in that same section — one per section.
    var_sections: dict = field(default_factory=dict)
    # The pipeline loop itself, so later passes can tell "inside the loop" from "outside".
    pipeline_loop: ast.For | None = None
    # The outermost loop enclosing it (itself when top-level). This pipeline's declarations
    # go before it and its drain after it, and two pipelines may not share one.
    outer_loop: ast.For | None = None
    # Every declared tile group's name, cross-core or not: recognising a slot pick needs the
    # full set. Kernel-wide.
    group_names: set = field(default_factory=set)
    # {slot variable: (group, slot count)} for slots chosen OUTSIDE the pipeline loop. The
    # index travels through ctx, so a delayed stage gets the slot of its own iteration.
    outer_slots: dict = field(default_factory=dict)
    # Cross-core sync context (buffers, memory, lifted ids)
    sync: CrossCoreSyncContext = field(default_factory=CrossCoreSyncContext)
    # Reference to closure_vars, so the transform chain need not thread it through.
    closure_vars: dict = field(default_factory=dict)


def analyze_pipeline(
    func_def: ast.FunctionDef,
    closure_vars: dict,
    config: PipelineConfig,
    facts: KernelFacts | None = None,
) -> list:
    """Analyze a serial kernel AST: one PipelineInfo per pipeline loop, in source order.

    Every legality check runs here, config included, so the transform can assume the shape it
    is handed is one it can emit. validate_names is the exception: it checks the framework's
    own generated names rather than the user's code.

    Kernel-level facts — declarations, scope, the sync context — are derived once and shared by
    reference with every loop's info; everything derived from a loop's stages is per loop.

    Structure is validated before the buffers are resolved, so a malformed stage chain is
    reported as such rather than as whatever the resolution trips over.

    Args:
        func_def: The kernel function's AST node
        closure_vars: Closure variables (module-level constants, stage functions)
        config: the pipeline configuration, whose preload must match the loops found here
        var_types: {name: (IR type class name, scalar dtype or None)} from the parser probe
        arg_types: {(lineno, col_offset): ArgTypeFacts} for call arguments, from the probe

    Returns:
        [PipelineInfo, ...], one per pipeline loop, in source order.
    """
    facts = facts or KernelFacts()
    # Find all stage functions.
    stage_func_names = set()
    for name, val in closure_vars.items():
        if is_pipeline_stage(val):
            stage_func_names.add(name)

    pipeline_loops = _find_pipeline_loops(func_def.body, stage_func_names)
    # L3/L5: pipeline enabled but no usable stages found.
    if not pipeline_loops:
        if not stage_func_names:
            raise InvalidOperation(
                "pipeline: no @pl.pipeline.stage functions found, but pipeline=... was "
                "set on the kernel. Decorate your stage functions with @pl.pipeline.stage."
            )
        raise InvalidOperation(
            "pipeline: no pipeline loop found — @pl.pipeline.stage functions exist but "
            "none are called inside a for-loop's `with pl.section_*()` blocks. The "
            "pipeline loop must contain stage calls wrapped in section blocks."
        )

    # Which section each name belongs to, so its ctx fill lands in the same one. A kernel
    # fact: it answers which section a NAME is bound in, which no loop can change.
    var_sections = _collect_var_sections(func_def)

    infos = []
    for loop_index, (loop, stages, outer_loop) in enumerate(pipeline_loops):
        info = PipelineInfo()
        info.loop_index = loop_index
        info.outer_loop = outer_loop
        # Shared by reference — see PipelineInfo. sync is filled in below, after the
        # structural checks, and every info sees the same object filling up.
        info.closure_vars = closure_vars
        info.var_types = facts.var_types
        info.arg_types = facts.arg_types
        info.buffer_vars = facts.buffer_vars
        info.var_sections = var_sections
        info.stages = stages
        _record_pipeline_loop_info(loop, info)
        # The stages' own ASTs, resolved once here and read from StageCall thereafter.
        for stage in stages:
            stage.func_def = get_funcdef(closure_vars.get(stage.func_name))
        # Structs passed to stages: their fields become ctx fields (see _collect_struct_args).
        info.struct_args = _collect_struct_args(func_def, info)
        infos.append(info)

    group_names = tile_group_names(facts.buffer_facts)
    sync = CrossCoreSyncContext()
    for info in infos:
        info.group_names = group_names
        info.sync = sync

    # Two gates for the checks that need only the parsed structure (see _validate): one
    # about the kernel as a whole, one per pipeline.
    validate_kernel_structure(infos, config.preload, func_def)
    for info in infos:
        validate_structure(info, func_def, stage_func_names)

    # Scan the cross-core buffer declarations for auto-sync. Sole writer of info.sync, which
    # the per-loop steps below read: what a group is called and how many slots it rotates
    # through are declaration facts, derived once.
    # Every plain function a stage or a loop-body statement might call, resolved once.
    helper_func_defs = _collect_helper_func_defs(closure_vars)
    tables = _resolve_kernel_buffers(
        sync, func_def, closure_vars, group_names,
        facts.buffer_facts, facts.unbound_decls, facts.same_name_decls,
    )

    for info in infos:
        # Slots chosen outside the pipeline loop: their index travels through ctx. Ahead of the
        # accesses so an unusable pick is reported as a pick, not as whichever op in a stage
        # body first failed to resolve the slot.
        info.outer_slots = _collect_outer_slots(info)
        # What each stage touches, which is the one half that is per pipeline.
        _resolve_loop_accesses(info, tables, facts)
        # And nothing outside a stage may touch what the graph tracks: its accesses would
        # be recorded against no stage and synchronised by nobody, and its cursor would hand
        # stages a slot that is not their task's.
        tracked = set(tables.cross_buffers) | tables.addr_shared
        validate_slot_picks(info, tracked & info.group_names, helper_func_defs)
        reject_buffer_access_outside_stages(info, facts.op_accesses, tables)
    # Every declared cross-core buffer must be used by SOME pipeline. The per-loop check in
    # validate_sync cannot see that: a buffer another loop uses is legitimately absent there.
    validate_kernel_buffers(sync, set().union(*(buffers_touched(i) for i in infos)))

    for info in infos:
        # Derive ctx fields from stage arguments
        _derive_ctx_fields(info)

    # Producer/consumer and address-reuse checks live in validate_sync, which runs once
    # the sync graph exists — see _sync_graph.build_graph.

    return infos


def _resolve_kernel_buffers(
    sync: CrossCoreSyncContext,
    func_def: ast.FunctionDef,
    closure_vars: dict,
    group_names: set,
    buffer_facts: dict,
    unbound_decls: list,
    same_name_decls: list,
) -> AccessTables:
    """Fill ``sync`` from the kernel's buffer declarations; return the lookup tables.

    The kernel-level half: declarations, addresses and the name tables, all of which every
    pipeline loop in the kernel shares. The per-loop half — what each stage actually touches —
    is _resolve_loop_accesses.

    The tables are returned even without a cross-core buffer: the local slot validation reads
    every declared group through them.

    ``buffer_facts`` is what the parse resolved for every declaration.
    The three structural validations run before the cross-core resolution, so a malformed
    declaration is reported as a declaration error rather than as a cross-core one.
    """
    # Filled before the cross-core split: _collect_outer_slots needs a slot count for
    # local groups too.
    sync.slot_counts = buffer_slot_counts(buffer_facts)
    sync.decl_spans = {name: decl.span for name, decl in buffer_facts.items()}

    validate_bound_buffer_decls(func_def, unbound_decls)
    validate_distinct_buffer_names(func_def, same_name_decls)
    reject_buffer_decls_in_loops(func_def, buffer_facts)
    cross_buffers, lifted_ids = cross_core_buffers(buffer_facts)
    sync.buffers = cross_buffers
    sync.lifted_ids = lifted_ids

    all_mem = buffer_memory_spaces(buffer_facts)

    # Detect address overlaps involving cross-core buffers (for auto-sync of
    # address-reused buffers). Local-local overlaps are ignored.
    if cross_buffers:
        sync.addr_ranges = buffer_addr_ranges(buffer_facts)
        sync.addr_overlaps = detect_addr_overlaps(sync.addr_ranges, set(cross_buffers.keys()), sync.decl_spans)

    return AccessTables(
        cross_buffers=cross_buffers,
        all_buffer_memory=all_mem,
        group_names=group_names,
        closure_vars=closure_vars,
        # Buffers sharing a physical region with another buffer, computed once for the
        # whole kernel. Not per stage: a region's members may be touched by different
        # stages, and a per-stage view would miss those.
        addr_shared={name for pair in sync.addr_overlaps for name in pair},
    )

    # Address-reuse sync is not built here: which edges need it and which ids they get both
    # follow from the sync graph (see _sync_graph.allocate_reuse_ids).


def buffers_touched(info: PipelineInfo) -> set:
    """Every buffer the stages of ONE pipeline touch.

    Read off the stages' buffer_access, which is where the per-loop half put it
    (_resolve_loop_accesses). Three readers ask this question — the kernel-level "is any
    buffer unused" check, the event-id allocator ("which ids are taken here"), and the
    declaration builder ("which ids does this loop declare") — and they must agree, so they
    all come here rather than each writing the comprehension out.
    """
    return {access.buffer for stage in info.stages for access in stage.buffer_access}


def _resolve_loop_accesses(info: PipelineInfo, tables: AccessTables, facts: KernelFacts) -> None:
    """Record what every stage of ONE pipeline touches, into each stage's buffer_access.

    The per-loop half (see _resolve_kernel_buffers). Per stage rather than per kernel because
    a stage belongs to exactly one pipeline, and the sync graph is built from one pipeline's
    accesses.
    """
    for stage in info.stages:
        if stage.func_def is None:
            continue
        # A node of the body resolves against the body's own tree; whatever cannot be pinned
        # to a node falls back to the call site.
        fn = info.closure_vars.get(stage.func_name)
        target = SectionKind.Cube if stage.section_kind == "cube" else SectionKind.Vector
        with current_span(span_of(stage.call_node or stage.source_stmt)), spans_in_function(fn):
            stage.buffer_access = stage_accesses(stage, facts.op_accesses.get(target, {}), tables)


def _collect_helper_func_defs(closure_vars: dict) -> dict:
    """``{name: FunctionDef}`` for every plain function a stage might call.

    Selected by the DECORATOR alone. A stage and a ``@pl.vector_function`` are left out: a
    stage is an individual of its own, and a vector function is one atomic op rather than a
    part of the stage that called it, so neither is searched as the caller's own body.

    Nothing needs to be handed in, so this is a kernel-level fact shared by every pipeline.

    They come from ``closure_vars`` — the kernel's scope, captured by @pl.jit — and are turned
    back into ASTs by re-reading their source (see _astutil.get_funcdef), because a helper is
    defined outside the kernel and so is nowhere in the kernel's own tree.
    """
    helper_func_defs: dict[str, ast.FunctionDef] = {}
    for name, val in closure_vars.items():
        if not callable(val) or is_pipeline_stage(val):
            continue
        func_def = get_funcdef(val)
        if func_def is None or is_vf_function(func_def):
            continue
        helper_func_defs[name] = func_def
    return helper_func_defs


def _record_pipeline_loop_info(stmt: ast.For, info: PipelineInfo) -> None:
    """Record loop metadata after the pipeline loop has been found."""
    # L7: loop variable must be a simple Name
    if not isinstance(stmt.target, ast.Name):
        raise InvalidType(
            "pipeline: the pipeline loop variable must be a simple name "
            "(e.g. `for ki in pl.range(...)`); tuple unpacking is not supported.",
            span=span_of(stmt.target),
        )
    info.pipeline_loop = stmt
    info.pipeline_loop_var = stmt.target.id
    # L6: loop must be pl.range(...) with extractable end bound
    info.pipeline_loop_end = None
    info.pipeline_loop_step = None
    if isinstance(stmt.iter, ast.Call):
        args = stmt.iter.args
        if len(args) >= 2:
            info.pipeline_loop_end = args[1]
        elif len(args) == 1:
            info.pipeline_loop_end = args[0]
        if len(args) >= 3:
            info.pipeline_loop_step = args[2]
    if info.pipeline_loop_end is None:
        raise NotSupported(
            f"pipeline: pipeline loop `for {info.pipeline_loop_var} in ...` must iterate "
            f"over pl.range(start, end[, step]) so the end bound can be extracted "
            f"for the is_valid guard; got an unsupported loop iterable.",
            span=span_of(stmt.iter),
        )


def _nested_search_body(stmt: ast.stmt) -> list[ast.stmt] | None:
    """Return the nested body that can contain a pipeline loop."""
    if isinstance(stmt, (ast.For, ast.With)):
        return stmt.body
    return None


def _find_pipeline_loops(stmts: list[ast.stmt], stage_func_names: set, outer: ast.For | None = None) -> list:
    """Every for-loop that drives a pipeline, as ``[(loop, stages, outer), ...]``, SOURCE ORDER.

    A candidate is a loop whose OWN body calls stages, directly or inside a ``pl.section_*()``
    block; an enclosing loop holds the pipelined one rather than stage calls, so nesting is
    unaffected.

    ``outer`` is the first ``for`` on the path down from the kernel body — the loop this
    pipeline's declarations and drain are placed around, with ``with`` blocks transparent to it,
    as they are to the placement search. A top-level pipeline loop is its own outer loop.
    Tracked on the way down, since this walk already knows the path.

    The search neither stops at a match nor skips a matched loop's body: a second stage-bearing
    loop, wherever it sits, is a fact the caller has to see.

    Source order is the order the loops are transformed in, and the order ``preload`` is matched
    against when given one value per loop.
    """
    found = []
    for stmt in stmts:
        if isinstance(stmt, ast.For):
            stages = _extract_stages_from_loop(stmt, stage_func_names)
            if stages:
                found.append((stmt, stages, outer or stmt))
        nested_body = _nested_search_body(stmt)
        if nested_body is not None:
            enclosing = outer or (stmt if isinstance(stmt, ast.For) else None)
            found.extend(_find_pipeline_loops(nested_body, stage_func_names, enclosing))
    return found


def _stage_call_in_section(body: list[ast.stmt], stage_func_names: set):
    """The stage call a section holds, as ``(name, call)``, or ``(None, None)`` for none.

    A section holding a stage may hold nothing else: everything in the pipeline loop between
    the first and the last stage is a stage and only a stage. Statements belong at the top or
    the bottom of the loop body, where they run once per beat, or inside the stage function.
    """
    calls = [
        (call_name(s.value), s.value)
        for s in body
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and call_name(s.value) in stage_func_names
    ]
    if not calls:
        return None, None
    if len(calls) > 1:
        raise InvalidArgument(
            f"pipeline: section block contains multiple stage calls ('{calls[0][0]}' and "
            f"'{calls[1][0]}'). Each `with pl.section_*()` block must contain exactly one "
            f"stage call.",
            span=span_of(calls[1][1]),
        )
    if len(body) > 1:
        extra = next(s for s in body if not (isinstance(s, ast.Expr) and s.value is calls[0][1]))
        raise InvalidOperation(
            f"pipeline: the section holding stage '{calls[0][0]}' also contains the statement "
            f"`{ast.unparse(extra).splitlines()[0]}`. A section that "
            f"holds a stage may hold nothing else.\n"
            f"Move it to the start or the end of the pipeline loop body, where it runs once "
            f"per beat, or into the stage function itself.",
            span=span_of(stmt_header(extra)),
        )
    return calls[0]


def _extract_stages_from_loop(for_stmt: ast.For, stage_func_names: set) -> list:
    """The stage calls in a for-loop body (expecting interleaved sections), in order.

    Returns them rather than appending to a PipelineInfo: the finder probes every loop in
    the kernel, and probing that also accumulates would pour several loops' stages into
    one list. An empty list means this loop drives no pipeline.
    """
    stages: list = []

    for stmt in for_stmt.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            fn = call_name(stmt.value)
            if fn in stage_func_names:
                stages.append(
                    StageCall(
                        func_name=fn,
                        section_kind="",
                        args=list(stmt.value.args),
                        delay=0,
                        source_stmt=stmt,
                        call_node=stmt.value,
                    )
                )
                continue
        if isinstance(stmt, ast.With):
            section_kind = _get_section_kind(stmt)
            if section_kind is None:
                _check_unsupported_section(stmt, stage_func_names)
                continue
            stage_call_info = _extract_stage_from_section(stmt, section_kind, stage_func_names)
            if stage_call_info is not None:
                stage_call_info.source_stmt = stmt
                stages.append(stage_call_info)

    return stages


def _check_unsupported_section(stmt: ast.With, stage_func_names: set):
    """C1: raise if an unsupported section type contains a stage call."""
    for body_stmt in stmt.body:
        if (
            isinstance(body_stmt, ast.Expr)
            and isinstance(body_stmt.value, ast.Call)
            and call_name(body_stmt.value) in stage_func_names
        ):
            raise InvalidOperation(
                f"pipeline: stage call "
                f"'{call_name(body_stmt.value)}' is inside an "
                f"unsupported `with` block. Stage calls must be wrapped in "
                f"`with pl.section_cube()` or `with pl.section_vector()`.",
                span=span_of(stmt_header(stmt)),
            )


def _extract_stage_from_section(stmt: ast.With, section_kind: str, stage_func_names: set) -> StageCall | None:
    """Extract a stage call from a section block, or None if it holds no stage."""
    stage_func_name, stage_call = _stage_call_in_section(stmt.body, stage_func_names)
    if stage_call is None:
        return None
    return StageCall(
        func_name=stage_func_name,
        section_kind=section_kind,
        args=stage_call.args,
        delay=0,
        call_node=stage_call,
    )


def _get_section_kind(with_stmt: ast.With) -> str | None:
    """Extract section kind ('cube' or 'vector') from a with statement."""
    if not with_stmt.items:
        return None
    ctx = with_stmt.items[0].context_expr
    if isinstance(ctx, ast.Call) and isinstance(ctx.func, ast.Attribute):
        if ctx.func.attr == "section_cube":
            return "cube"
        elif ctx.func.attr == "section_vector":
            return "vector"
    return None


# A tile group is a handle too, but shares TupleType with pl.struct, so _name_role
# separates those two by name.
_HANDLE_TYPES = frozenset({"TensorType", "TileType"})


def _name_role(name: str, info: PipelineInfo) -> str:
    """Classify a name for the ctx slot, by its type from the parser probe (info.var_types).

    handle  a tensor, tile or tile group: a reference, fixed for the whole task stream.
    const   no typed binding of its own, hence a compile-time constant (a closure constant,
            an enum, a nested list).
    live    everything else — a scalar, or a struct whose fields the slot carries. Its value
            belongs to one beat.
    """
    recorded = info.var_types.get(name)
    kind = recorded[0] if isinstance(recorded, tuple) else recorded
    if kind is None:
        return "const"
    if kind in _HANDLE_TYPES:
        return "handle"
    if kind == "TupleType" and name not in info.struct_args:
        return "handle"  # a tile group, or any other tuple: a handle rather than a value
    return "live"


def _collect_var_sections(func_def: ast.FunctionDef) -> dict:
    """``{name: {section kinds it is bound in}}`` for names bound in a ``pl.section_*()``.

    A name bound inside a section exists only for that target — the other target's parse skips
    the block — so a ctx field fed from it has to be filled inside the same section.

    A name bound in BOTH sections is two variables, not a conflict: each target compiles to its
    own function with its own ctx array (``..._impl_cube`` / ``..._impl_vector``), so each fills
    its own copy from the name it can see. Hence a SET of sections, and one fill per section in
    _build_ctx_field_fills.

    Names absent from this map come from outside any section and are visible to both.
    """
    sections: dict[str, set] = {}
    for node in ast.walk(func_def):
        if not isinstance(node, ast.With):
            continue
        kind = _get_section_kind(node)
        if kind is None:
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Store):
                sections.setdefault(inner.id, set()).add(kind)
    return sections


def _travels_in_ctx(node: ast.expr, info: PipelineInfo) -> bool:
    """True if this stage argument has to be snapshotted into the ctx slot.

    False for an argument that reads the same on every beat and can stay where it was
    written: one mentioning a handle, or none of whose names is live (``3``, ``BASE * 2``).
    """
    roles = {_name_role(n.id, info) for n in ast.walk(node) if isinstance(n, ast.Name)}
    if "handle" in roles:
        return False
    return "live" in roles


def _collect_outer_slots(info: PipelineInfo) -> dict:
    """{name: (group, kind, slot count)} for slots picked outside the pipeline loop.

    Such a slot advances once per outer iteration while every beat reads it, so a delayed stage
    would see whatever the variable was last rebound to. Carrying the slot index in ctx fixes that:
    each beat snapshots the index it used, and every stage re-selects the slot with its own.

    Renames are followed to a fixed point and every name on the chain keeps its entry, because
    the consumers key on different ones — the rewrite on the assignment target, the ctx field
    and the stage argument on the name the stage was handed.

    Searched inside THIS pipeline's outer loop only. A pick above it runs once for the whole
    task stream, so the variable never moves and the argument passes through; and two
    pipelines' outer loops are disjoint, which keeps their same-named picks apart.

    Only slots a stage actually consumes are kept; pure bookkeeping slots are left alone.
    The rewrite needs the slot count as its modulus, so a group whose depth or mutex_ids do
    not resolve statically is reported as an error rather than silently skipped — skipping
    it would leave the delayed stage reading the wrong slot with no diagnostic.
    """
    groups = info.group_names
    if not groups or info.pipeline_loop is None or info.outer_loop is None:
        return {}
    slot_counts = info.sync.slot_counts

    consumed = {
        arg.id for stage in info.stages for arg in stage.args if isinstance(arg, ast.Name)
    }
    inside = {id(node) for node in ast.walk(info.pipeline_loop)}

    picked: dict = {}
    for node in ast.walk(info.outer_loop):
        if not isinstance(node, ast.Assign) or id(node) in inside:
            continue  # picked per beat: the sync graph already covers it
        pick = slot_accessor(node.value) if len(node.targets) == 1 else None
        if pick is None or pick[0] not in groups:
            _reject_buried_slot_pick(node, groups)
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id in picked:
            raise InvalidOperation(
                f"pipeline: '{target.id}' is assigned a slot more than once outside the "
                f"pipeline loop. Its stages would each have to know "
                f"which of those picks is the current one, and the index ctx carries is one "
                f"per name. Use a separate variable per pick.",
                span=span_of(node),
            )
        group, kind = pick
        picked[target.id] = (group, kind, slot_counts[group])

    # Follow renames to a fixed point, so a chain of them resolves too.
    changed = True
    while changed:
        changed = False
        for node in ast.walk(info.outer_loop):
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name) or target.id in picked:
                continue
            if isinstance(node.value, ast.Name) and node.value.id in picked:
                picked[target.id] = picked[node.value.id]
                changed = True

    # A chain counts if a stage takes ANY name along it; every name on that chain is kept,
    # because the consumers do not all look up the same one.
    by_entry: dict = {}
    for name, entry in picked.items():
        by_entry.setdefault(entry, []).append(name)
    return {
        name: entry
        for entry, names in by_entry.items()
        if any(n in consumed for n in names)
        for name in names
    }


def _reject_buried_slot_pick(node: ast.Assign, groups: set) -> None:
    """Refuse an assignment that takes a slot somewhere inside a larger expression.

    ``slot = g.next()`` is the only spelling the transform can rewrite: it has to replace
    the pick with an explicit index, and an accessor buried in a conditional, a call
    argument or a tuple gives it no single statement to replace. Such an assignment is not
    recognised as a pick at all, so without this it would be skipped in silence and the
    delayed stage would read whatever the variable was last set to.
    """
    for sub_node in ast.walk(node.value):
        buried = slot_accessor(sub_node)
        if buried is not None and buried[0] in groups:
            raise InvalidOperation(
                f"pipeline: a slot is taken from tile group "
                f"'{buried[0]}' inside a larger expression "
                f"(`{ast.unparse(node.value)}`). A slot picked outside the pipeline loop "
                f"must be assigned on its own (`slot = {buried[0]}.next()`), because the "
                f"transform replaces that statement with an explicit index for ctx to carry.",
                span=span_of(node),
            )


def _collect_struct_args(func_def: ast.FunctionDef, info: PipelineInfo) -> dict:
    """{name: [field names]} for every ``pl.struct`` a stage is called with.

    A struct's fields are carried in the ctx slot under their own names, and the slot is
    passed to the stage in place of the struct. Nothing about the stage body changes: the
    slot is itself a struct, so ``ri.ki`` resolves against the delayed slot's ``ki``. This
    is what lets the framework stay ignorant of HOW the user updates the struct — in a
    branch, in a helper, over several statements — since only the field values are read,
    at a point where all of that has already run.

    ``pl.struct_array`` is rejected: an array of contexts is exactly what this transform
    generates, and a user-managed second one cannot be kept in step with it.
    """
    stage_arg_names = {
        arg.id for stage in info.stages for arg in stage.args if isinstance(arg, ast.Name)
    }
    if not stage_arg_names:
        return {}

    # Names bound inside the pipeline loop cannot be a stable struct: they would be
    # rebound every beat, which is the user-managed ring buffer case above.
    loop_bound: set = set()
    for stmt in info.pipeline_loop.body if info.pipeline_loop else []:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                loop_bound.add(node.id)

    result: dict = {}
    for node in ast.walk(func_def):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in stage_arg_names:
            continue
        call = node.value
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)):
            continue
        if call.func.attr == "struct_array":
            raise NotSupported(
                f"pipeline: '{target.id}' comes from pl.struct_array() and is passed to a "
                f"stage. Use pl.struct() instead: the pipeline already gives every stage "
                f"the values from its own iteration, so a hand-rolled array of contexts is "
                f"both unnecessary and impossible to keep in step with it.",
                span=span_of(call),
            )
        if call.func.attr != "struct":
            continue
        if target.id in loop_bound:
            raise InvalidOperation(
                f"pipeline: struct '{target.id}' is bound inside the pipeline loop. Declare "
                f"it outside the loop and update its fields inside — the pipeline snapshots "
                f"the fields each iteration.",
                span=span_of(call),
            )
        result[target.id] = [kw.arg for kw in call.keywords if kw.arg]
        info.struct_defs[target.id] = call
    return result


def _derive_ctx_fields(info: PipelineInfo):
    """Decide how each stage argument reaches its stage, by the argument's TYPE.

    A scalar rides in the ctx slot; a Tile cannot (the slot holds scalars) and travels as the
    index that selects it; a struct travels as its fields; a tensor or tile group is a handle
    and does not travel at all. The type has an exact answer from the parser probe
    (info.var_types), while "does this value change?" would mean reasoning about how it was
    produced, and a wrong answer there silently feeds a delayed stage the current beat's value.

    A field is declared with the type its stage receives, read at the stage call. A call is
    parsed by its own section's target only, so a name bound to different types in the two
    sections declares the field differently in each.

    Sets two fields of ``info``:

      stage_arg_mapping  per argument — the ctx field it reads, or None if it passes through.
      ctx_sources        per ctx field — its fill expression, the sections to emit it in, and
                         its initializer in each section.
    """
    struct_fields = {f for fields in info.struct_args.values() for f in fields}
    # A scalar whose name collides with a struct field is renamed. The struct's field names
    # are dictated by the stage bodies that read them (`ri.ki`), so they cannot move; a
    # scalar's ctx field name is internal to the generated code, so it can.
    info.scalar_ctx_names = {}

    def scalar_field(name: str) -> str:
        if name not in info.scalar_ctx_names:
            info.scalar_ctx_names[name] = f"_pl_{name}" if name in struct_fields else name
        return info.scalar_ctx_names[name]

    info.ctx_sources = {}
    info.stage_arg_mapping = []

    # {variable: [(the argument naming it, the section of the stage it is handed to)]}
    arg_sites: dict = {}
    for stage in info.stages:
        for arg in stage.args:
            if isinstance(arg, ast.Name):
                arg_sites.setdefault(arg.id, []).append((arg, stage.section_kind))

    def register(field_name: str, fill: ast.expr, sections: set, inits: dict) -> None:
        """Record a ctx field: what fills it, where, and the initializer it is declared with
        in each section."""
        info.ctx_sources[field_name] = (fill, sections, inits)

    def both_sections(init: ast.expr) -> dict:
        return {kind: init for kind in _SECTION_KINDS}

    def per_section(name: str, init_of) -> dict:
        """{section kind: initializer} for a field fed from ``name``, from ``init_of(argument)``
        at each stage call handing it over. A section none of whose stages takes ``name``
        declares the field, unread, with the other section's initializer."""
        inits: dict = {}
        for node, kind in arg_sites[name]:
            init = init_of(node)
            if ast.dump(inits.setdefault(kind, init)) != ast.dump(init):
                raise InvalidOperation(
                    f"pipeline: '{name}' is handed to two {kind} stages with different types, "
                    f"but travels between them in one ctx field.\n"
                    f"Pass a value of the same type to both, or a differently named one.",
                    span=span_of(node),
                )
        fallback = next(iter(inits.values()))
        return {kind: inits.get(kind, fallback) for kind in _SECTION_KINDS}

    def call_type(node: ast.expr) -> ArgTypeFacts:
        """The probe's record for the stage argument ``node``, keyed by its position."""
        recorded = info.arg_types.get((node.lineno, node.col_offset))
        if recorded is None:
            raise KeyNotFound(
                f"pipeline (internal): the stage argument `{ast.unparse(node)}` has no type "
                f"recorded by the probe, so its ctx field's type cannot be decided. Please report."
            )
        return recorded

    def fields_of_struct(struct_name: str, node: ast.expr) -> dict:
        """The fields of the struct handed over at ``node``."""
        fields = call_type(node).fields
        if fields is None:
            raise NotSupported(
                f"pipeline: '{struct_name}' is handed to a stage as a struct whose fields travel "
                f"in the pipeline context, but here it holds a value that is not a struct. A name "
                f"a stage reads fields off must be a pl.struct() in every section it is passed in.",
                span=span_of(node),
            )
        return fields

    def register_struct(struct_name: str) -> None:
        """Register a struct's fields under their own names, read as `<struct>.<field>`."""
        sections = info.var_sections.get(struct_name, set())
        struct_def = info.struct_defs.get(struct_name)
        for fname in info.struct_args[struct_name]:
            register(
                fname,
                ast.Attribute(value=ast.Name(id=struct_name, ctx=ast.Load()), attr=fname, ctx=ast.Load()),
                sections,
                per_section(struct_name, lambda node: _struct_field_init(
                    fields_of_struct(struct_name, node).get(fname), struct_def)),
            )

    def register_slot_index(slot_name: str) -> str:
        """Register the index field of an outer slot, read from its like-named counter and
        filled in the section its group is declared in. The counter is an integer."""
        group = info.outer_slots[slot_name][0]
        fname = slot_index_field(group)
        register(
            fname,
            ast.Name(id=fname, ctx=ast.Load()),
            info.var_sections.get(group, set()),
            both_sections(ast.Constant(value=0)),
        )
        return fname

    def register_scalar(var_name: str) -> str:
        """Register a scalar's field, read under the variable's own name."""
        fname = scalar_field(var_name)
        register(
            fname,
            ast.Name(id=var_name, ctx=ast.Load()),
            info.var_sections.get(var_name, set()),
            per_section(var_name, lambda node: _scalar_field_init(call_type(node), node)),
        )
        return fname

    for stage in info.stages:
        arg_map = []
        for argpos, arg in enumerate(stage.args):
            if isinstance(arg, ast.Name):
                if arg.id in info.struct_args:
                    # The whole ctx slot stands in for the struct: same field names, so the
                    # stage body needs no rewriting. Fields are snapshotted individually.
                    register_struct(arg.id)
                    arg_map.append(PL_STRUCT_ARG)
                    continue
                if arg.id in info.outer_slots:
                    # Slot chosen outside the loop: ctx carries the slot index, and the stage
                    # re-selects the slot with its own beat's index (see _build_stage_args).
                    arg_map.append(register_slot_index(arg.id))
                    continue
                if _travels_in_ctx(arg, info):
                    arg_map.append(register_scalar(arg.id))
                else:
                    arg_map.append(None)
                continue
            # An expression: a field of its own, filled from the expression, in the stage's
            # own section — where the field is read and every name in it exists.
            if _travels_in_ctx(arg, info):
                # Unique because a stage function appears once per loop
                # (_validate._check_unique_stage_names).
                field_name = f"_pl_arg_{stage.func_name}_{argpos}"
                init = _scalar_field_init(call_type(arg), arg)
                register(field_name, arg, {stage.section_kind}, both_sections(init))
                arg_map.append(field_name)
            else:
                arg_map.append(None)
        info.stage_arg_mapping.append(arg_map)

    # Framework fields carry the _pl_ prefix so they cannot collide with a user struct's
    # field. The two trailing ones are the transform's own bookkeeping, not stage data, and
    # so the only fields with no ctx_sources entry.
    info.ctx_fields = [PL_IS_VALID_FIELD] + sorted(info.ctx_sources) + [PL_TASK_ID_FIELD]



# The two targets a ctx field is declared for, as section kinds name them.
_SECTION_KINDS = ("cube", "vector")


def _scalar_field_init(recorded: ArgTypeFacts, node=None) -> ast.expr:
    """The initializer for a scalar ctx field, from the argument's ArgTypeFacts.

    ``node`` is the user's argument, named in the error when the value is not a scalar.
    """
    from pypto.pypto_impl import ir

    kind, dtype = recorded.kind, recorded.dtype
    if kind != "ScalarType" or dtype is None:
        raise NotSupported(
            "pipeline: a value carried between stages has no scalar type, so it cannot travel in "
            "the pipeline context. Pass a scalar, or a tensor, tile, tile group, tuple or "
            "pl.make_tuple() result, which are forwarded as they are.",
            span=span_of(node),
        )
    if dtype == ir.DataType.BOOL:
        return ast.Constant(value=False)
    if dtype.is_float():
        return ast.Constant(value=0.0)
    return ast.Constant(value=0)


def _struct_field_init(var_type, node=None) -> ast.expr:
    """The initializer for a ctx field declared from a struct field's type.

    A struct field is a scalar or a fixed-size scalar array; the parser already checks that.
    """
    from pypto.pypto_impl import ir

    if isinstance(var_type, ir.ScalarType):
        return _scalar_field_init(ArgTypeFacts("ScalarType", var_type.dtype), node)
    if isinstance(var_type, ir.TupleType) and var_type.types:
        init = _scalar_field_init(ArgTypeFacts("ScalarType", var_type.types[0].dtype), node)
        return ast.List(elts=[init] * len(var_type.types), ctx=ast.Load())
    raise NotSupported(
        "pipeline: a struct field carried between stages is not a scalar or a fixed-size array of "
        "scalars, so it cannot travel in the pipeline context.",
        span=span_of(node),
    )


def _decl_identity(decl) -> tuple:
    """What tells one declaration from another across the targets' parses.

    The span alone is shared by every declaration one factory function makes; the
    kernel-body site calling it tells them apart.
    """
    span = decl.span
    return (span.filename, span.begin_line, span.begin_column, decl.site)


def _operand_names(operand, decl_names: dict) -> tuple:
    """The declared names of the buffers one parsed operand reaches, in order."""
    found = (decl_names[id(ref.buffer)] for ref in operand.buffers if id(ref.buffer) in decl_names)
    return tuple(dict.fromkeys(found))


def _operand_refs(operand, decl_names: dict) -> tuple:
    """The declared name and slot selector of every selected buffer ref."""
    return tuple(
        (decl_names[id(ref.buffer)], ref.selector)
        for ref in operand.buffers
        if id(ref.buffer) in decl_names and ref.selector is not None
    )


def probe_kernel_facts(kernel_def, bound_signature=None) -> KernelFacts:
    """Probe-parse the kernel once per target, for the facts only the parser can give.

    The parsed IR is discarded; what the parse learns is collected into a KernelFacts.
    """
    from pypto.pypto_impl import ir
    from pypto_pro.language.parser._ast_parser import ASTParser

    facts = KernelFacts()
    if_const = facts.if_const_map
    var_types = facts.var_types
    arg_types = facts.arg_types
    op_accesses = facts.op_accesses
    buffer_facts = facts.buffer_facts
    buffer_vars = facts.buffer_vars
    unbound_decls = facts.unbound_decls
    same_name_decls = facts.same_name_decls
    for target in (ir.SectionKind.Cube, ir.SectionKind.Vector):
        # The probe discards parsed IR, so AUTO placeholders do not need resolution.
        parser = ASTParser(
            kernel_def._source_file,
            kernel_def._source_lines,
            target,
            kernel_def._line_offset,
            kernel_def._col_offset,
            strict_ssa=kernel_def._strict_ssa,
            closure_vars=kernel_def._closure_vars,
            auto_mutex=kernel_def._auto_mutex,
            auto_pipeline=True,
            debug_info=ir.IRDebugInfo(),
            tilingkey_consts=kernel_def._tilingkey_consts,
            datatype_consts=kernel_def._datatype_consts,
            bound_signature=bound_signature,
            void_return_only=True,
            void_return_context="@pl.jit",
            allow_early_return=True,
        )
        # define_var is the single point where every name gets bound, so wrapping it here
        # harvests types without touching the parser itself.
        scope = parser.scope_manager
        original_define = scope.define_var
        # {id(declaration handle IR): the first kernel-body variable bound to that buffer}
        decl_names: dict = {}
        # {kernel-body variable holding a buffer: the declaration handle ids it reaches},
        # named below once decl_names is complete.
        buffer_var_ids: dict = {}

        def _record_type(value):
            var_type = getattr(value, "type", None)
            if var_type is None:
                return None
            dtype = var_type.dtype if isinstance(var_type, ir.ScalarType) else None
            return (type(var_type).__name__, dtype)

        def _define(name, value, allow_redef=False, span=None, _orig=original_define):
            recorded = _record_type(value)
            if recorded is not None:
                var_types.setdefault(name, recorded)
            var_type = getattr(value, "type", None)
            origins = parser.buffers_of(value)
            # Whether the value IS a buffer rather than a slot taken from one.
            holds_buffer = (
                not parser.inline_call_stack
                and bool(origins)
                and (parser.is_tile_group(value) or isinstance(var_type, ir.TensorType))
            )
            # A kernel-body variable names a buffer when it holds exactly that one buffer: a
            # tile group handle, or a GM view. A tile taken from it, or a tuple of several
            # buffers, names none.
            if holds_buffer and len(origins) == 1:
                decl_names.setdefault(id(origins[0]), name)
            # Every variable holding a buffer, the declaration's own name included, so a
            # consumer resolves an alias without having to follow the assignments itself.
            if holds_buffer:
                buffer_var_ids[name] = tuple(id(origin) for origin in origins)
            return _orig(name, value, allow_redef=allow_redef, span=span)

        scope.define_var = _define

        def _struct_fields(expr):
            """{field name: IR type} when ``expr`` is a struct, None when it is not one."""
            var_type = getattr(expr, "type", None)
            if not isinstance(var_type, ir.TupleType):
                return None
            info = parser.debug_info.get_tuple_type_info(var_type)
            if info is None or info.kind != ir.TupleTypeKind.STRUCT:
                return None
            return dict(zip(info.fields, var_type.types))

        # Record the type of each argument a kernel-body call writes, keyed by its position in
        # the kernel's AST. A call inside an inlined function, or a parameter default, sits in
        # another function's AST, whose positions count from its own first line and so can
        # repeat a kernel position.
        original_bind = parser._bind_inline_arguments

        def _bind(func_name, func_def, call, _orig=original_bind):
            bound = _orig(func_name, func_def, call)
            if parser.inline_call_stack:
                return bound
            written = {id(arg) for arg in call.args}
            for _param, (expr, arg_node) in bound.items():
                if id(arg_node) not in written:
                    continue
                recorded = _record_type(expr)
                if recorded is None:
                    continue
                kind, dtype = recorded
                arg_types.setdefault(
                    (arg_node.lineno, arg_node.col_offset),
                    ArgTypeFacts(kind, dtype, _struct_fields(expr)),
                )
            return bound

        parser._bind_inline_arguments = _bind
        parser.parse_function(kernel_def._func_def, func_type=kernel_def._func_type)
        # Each declaration this parse saw, under the name it is known by. The first target to
        # see one wins; the same declaration parsed twice yields equal facts.
        for handle, decl in parser.parse_facts.buffer_decls.items():
            name = decl_names.get(id(handle))
            if name is None:
                unbound_decls.append(decl)
                continue
            known = buffer_facts.setdefault(name, decl)
            if _decl_identity(known) != _decl_identity(decl):
                same_name_decls.append((name, known, decl))
        # The variables, now that every declaration this parse saw has its name. A declaration
        # inside the other target's section is not seen here, so the targets' answers are
        # unioned rather than one overwriting the other.
        for name, handle_ids in buffer_var_ids.items():
            reached = tuple(decl_names[i] for i in handle_ids if i in decl_names)
            if reached:
                buffer_vars[name] = tuple(dict.fromkeys(buffer_vars.get(name, ()) + reached))
        op_accesses[target] = {
            site: [
                StageOpAccess(
                    access.op_name,
                    tuple(
                        OperandFact(
                            operand.position,
                            operand.keyword,
                            _operand_names(operand, decl_names),
                            _operand_refs(operand, decl_names),
                        )
                        for operand in access.operands
                    ),
                    access.keyword_names,
                    access.in_vf,
                    access.span,
                )
                for access in accesses
            ]
            for site, accesses in parser.parse_facts.op_accesses.items()
        }
        # Merge, preferring a CONSTANT verdict: a target that skips a section never sees the
        # condition, so a later (False, None) must not clobber the other target's fold.
        for k, v in parser.parse_facts.if_const_map.items():
            existing = if_const.get(k)
            if existing is None or (not existing[0] and v[0]):
                if_const[k] = v
            elif existing[0] and v[0] and existing[1] != v[1]:
                # Both parses folded it, to different values: not target-independent after all. Fall back
                # to dynamic so a branch wrapping a stage reports CB3 rather than being pruned down a side
                # that is right for one target only.
                if_const[k] = (False, None)
    # The kernel's source order, which the targets' parses do not produce on their own: readers
    # emitting one statement per declaration must do so in a settled order. A declaration is
    # placed by its site, so one a factory function makes sorts where the kernel calls it.
    facts.buffer_facts = dict(sorted(buffer_facts.items(), key=lambda item: item[1].site or (0, 0)))
    return facts


class _UnresolvedSlotIndexError(Exception):
    """A dynamic slot index could not be resolved for one task."""


def analyze_local_slot_events(info, schedule) -> dict:
    """Expand each local group's slot selections into per-slot events.

    Groups whose dynamic index cannot be resolved are reported and left out; the lifetime
    check in _validate only consumes the events this returns.
    """
    events_by_group: dict = {}
    cross = set(info.sync.buffers)
    task_count = _local_task_count(info)
    for group in sorted(info.group_names - cross):
        depth = info.sync.slot_counts.get(group)
        if depth is None:
            continue
        selectors = _local_stage_selectors(info, group)
        if not selectors:
            continue
        kinds = {selector.kind for stage_selectors in selectors.values() for selector in stage_selectors}
        if not ({SelectorKind.CURRENT, SelectorKind.PREVIOUS} & kinds) and SelectorKind.INDEX not in kinds:
            continue
        try:
            events_by_group[group] = _local_slot_events(info, group, depth, selectors, schedule, task_count)
        except _UnresolvedSlotIndexError as exc:
            _warn_located(str(exc), PyptoProWarning, info.sync.decl_spans.get(group))
    return events_by_group


def _local_stage_selectors(info, group: str) -> dict:
    """``{stage index: [SlotSelector, ...]}`` in parser selection order."""
    result: dict = {}
    for index, stage in enumerate(info.stages):
        by_order: dict = {}
        for access in stage.buffer_access:
            if access.buffer != group or access.selector is None:
                continue
            by_order[access.selector.order] = access.selector
        if by_order:
            result[index] = [by_order[order] for order in sorted(by_order)]
    return result


def _local_task_count(info):
    """The pipeline loop's constant trip count, or None when it is not a constant."""
    end = info.pipeline_loop_end
    if end is None:
        return None
    env = dict(info.closure_vars)
    for stage in info.stages:
        function = info.closure_vars.get(stage.func_name)
        env.update(getattr(function, "__globals__", {}))
    try:
        value = ExprEvaluator(env).eval_expr(end)
    except Exception:
        return None
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _local_slot_events(info, group: str, depth: int, selectors: dict, schedule: list, task_count) -> list:
    """Expand one group's selectors into ``(beat, task, stage, kind, order, slot)`` events."""
    max_delay = max(schedule) if schedule else 0
    ntask = task_count if task_count is not None else max_delay + 2 * depth + 4
    max_beat = ntask + max_delay
    cursor = depth - 1
    events: list = []
    for beat in range(max_beat):
        for stage_index in sorted(selectors):
            task = beat - schedule[stage_index]
            if task < 0 or task >= ntask:
                continue
            for order, selector in enumerate(selectors[stage_index]):
                if selector.kind is SelectorKind.NEXT:
                    cursor = (cursor + 1) % depth
                    slot = cursor
                elif selector.kind is SelectorKind.CURRENT:
                    slot = cursor
                elif selector.kind is SelectorKind.PREVIOUS:
                    slot = (cursor + depth - 1) % depth
                else:
                    slot = _local_index_slot(info, group, stage_index, selector, task, depth)
                events.append((beat, task, stage_index, selector.kind, order, slot))
    return events


def _local_index_slot(info, group: str, stage_index: int, selector, task: int, depth: int) -> int:
    """Resolve the slot of an explicit ``group[index]`` selection for one task."""
    if selector.slot is not None:
        return selector.slot
    stage = info.stages[stage_index]
    parameters = [arg.arg for arg in stage.func_def.args.args] if stage.func_def is not None else []
    function = info.closure_vars.get(stage.func_name)
    env = dict(getattr(function, "__globals__", {}))
    env.update(info.closure_vars)
    if parameters:
        env[parameters[0]] = task
        suffix = "_" + parameters[0]
        for node in ast.walk(selector.index_node):
            if isinstance(node, ast.Name) and node.id.startswith("__inline_") and node.id.endswith(suffix):
                env[node.id] = task
    try:
        value = ExprEvaluator(env).eval_expr(selector.index_node)
    except Exception as exc:
        raise _UnresolvedSlotIndexError(
            f"pipeline: skipped local slot lifetime check for buffer '{group}' in stage "
            f"'{stage.func_name}': slot index `{ast.unparse(selector.index_node)}` depends on a "
            f"value not resolvable at compile time (task {task})."
        ) from exc
    if isinstance(value, bool) or not isinstance(value, int):
        raise _UnresolvedSlotIndexError(
            f"pipeline: skipped local slot lifetime check for buffer '{group}' in stage "
            f"'{stage.func_name}': slot index `{ast.unparse(selector.index_node)}` is not an "
            f"integer (task {task})."
        )
    if not 0 <= value < depth:
        raise InvalidArgument(
            f"pipeline: buffer '{group}' selects slot {value}, outside [0, {depth}).",
            span=info.sync.decl_spans.get(group),
        )
    return value
