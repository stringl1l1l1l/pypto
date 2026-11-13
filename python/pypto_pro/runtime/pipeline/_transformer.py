# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Transformer: convert serial kernel AST to preload pipeline AST."""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass

from pypto.pypto_impl.ir import MemorySpace

from ..._errors import CommonInner, KeyNotFound, NotSupported, span_of
from ._analyzer import KernelFacts, PipelineInfo, analyze_local_slot_events, analyze_pipeline, buffers_touched
from ._astutil import (
    PL_IS_VALID_FIELD,
    PL_STRUCT_ARG,
    PL_TASK_ID_FIELD,
    call_name,
    slot_accessor,
    slot_index_field,
    task_id_var,
)
from ._cross_core_scanner import CrossCoreBuffer
from ._stage import is_pipeline_stage
from ._sync_graph import SyncPlan, build_graph, plan_sync, sites_for
from ._validate import validate_local_slot_lifetime, validate_names
from .config import PipelineConfig


def _rewrite_outer_slot_picks(info: PipelineInfo) -> None:
    """Turn each outside-the-loop slot pick into an explicit index, in place.

    ``x = g.next()`` becomes ``_pl_idx_g = (_pl_idx_g + 1) % slots`` followed by
    ``x = g[_pl_idx_g]``; ``x = g[expr]`` becomes ``_pl_idx_g = expr`` plus the same
    subscript. Either way the index survives as a plain scalar, so ctx can carry it and each
    stage can re-select the slot for its own beat.

    ``x`` itself stays: the statements around the pick still mean the CURRENT outer
    iteration's slot. Subscripting leaves the group's cursor untouched.

    Rewrites inside THIS pipeline's outer loop only, the region the picks were collected
    from — the outer loops of several pipelines are disjoint subtrees, so every pick is
    rewritten exactly once, by the pipeline it belongs to.
    """
    if not info.outer_slots or info.outer_loop is None:
        return
    by_group = {var: gs for var, gs in info.outer_slots.items()}
    inside = {id(node) for node in ast.walk(info.pipeline_loop)} if info.pipeline_loop else set()

    class _Rewriter(ast.NodeTransformer):
        def visit_Assign(self, node: ast.Assign):  # noqa: N802 - ast API
            if id(node) in inside or len(node.targets) != 1:
                return node
            target = node.targets[0]
            if not isinstance(target, ast.Name) or target.id not in by_group:
                return node
            if slot_accessor(node.value) is None:
                # A rename of a slot (``alias = cur``) carries the same entry so that the
                # stage's argument resolves, but only the pick itself may be rewritten:
                # rewriting the rename too would advance the group's index a second time.
                return node
            group, kind, slots = by_group[target.id]
            idx_name = slot_index_field(group)
            if kind == "index":
                index_assign = ast.Assign(
                    targets=[ast.Name(id=idx_name, ctx=ast.Store())],
                    value=copy.deepcopy(node.value.slice),
                    lineno=0,
                )
            else:
                # Only next() advances the group's cursor; current()/previous() read it where it
                # stands, so the counter has to move by the same amount the accessor would.
                step = {"next": 1, "current": 0, "previous": -1}[kind]
                value: ast.expr = ast.Name(id=idx_name, ctx=ast.Load())
                if step:
                    value = ast.BinOp(
                        left=ast.BinOp(left=value, op=ast.Add(), right=ast.Constant(value=step)),
                        op=ast.Mod(),
                        right=ast.Constant(value=slots),
                    )
                index_assign = ast.Assign(
                    targets=[ast.Name(id=idx_name, ctx=ast.Store())],
                    value=value,
                    lineno=0,
                )
            pick = ast.Assign(
                targets=[ast.Name(id=target.id, ctx=ast.Store())],
                value=ast.Subscript(
                    value=ast.Name(id=group, ctx=ast.Load()),
                    slice=ast.Name(id=idx_name, ctx=ast.Load()),
                    ctx=ast.Load(),
                ),
                lineno=0,
            )
            return [index_assign, pick]

    _Rewriter().visit(info.outer_loop)


# ---------------------------------------------------------------------------
# Cross-core sync emission
#
# Everything that turns a SyncPlan into statements, in the order the path is read: plan
# it (_build_sync_plan), find one stage's sites (_sites_for_stage), render them
# (_sync_stmts_for), place the two halves outside the loop (_emit_prefire_and_drain).
#
# Nothing here decides WHERE a stage runs — that is arrangement, below. The two share
# only the schedule: _schedule_of reads the delays, and the graph is built from them.


def _build_event_id_index(ids_node: ast.expr, slot_count: int, index_expr: ast.expr) -> ast.expr:
    """Build `<ids_node>[<index_expr> % slot_count]`."""
    idx = ast.BinOp(
        left=copy.deepcopy(index_expr),
        op=ast.Mod(),
        right=ast.Constant(value=slot_count),
    )
    return ast.Subscript(
        value=copy.deepcopy(ids_node),
        slice=idx,
        ctx=ast.Load(),
    )


def _build_system_sync_stmt(fn_name: str, pipe_name: str, event_id: ast.expr) -> ast.stmt:
    """Build `pl.system.<fn_name>(pipe=pl.PipeType.<pipe>, event_id=<event_id>)`."""
    call = ast.Call(
        func=ast.Attribute(
            value=ast.Attribute(value=ast.Name(id="pl", ctx=ast.Load()), attr="system", ctx=ast.Load()),
            attr=fn_name,
            ctx=ast.Load(),
        ),
        args=[],
        keywords=[
            ast.keyword(
                arg="pipe",
                value=ast.Attribute(
                    value=ast.Attribute(value=ast.Name(id="pl", ctx=ast.Load()), attr="PipeType", ctx=ast.Load()),
                    attr=pipe_name,
                    ctx=ast.Load(),
                ),
            ),
            ast.keyword(arg="event_id", value=event_id),
        ],
    )
    return ast.Expr(value=call, lineno=0)


def _cross_loop_barrier(info: PipelineInfo, loop_count: int) -> list[ast.stmt]:
    """The whole-core barrier that closes this pipeline loop off — empty for the last one.

    It goes at the very end of the loop's tail, after the sync drain has collected every
    permit, so the next pipeline starts with both cores aligned on the buffers it reuses.
    """
    if info.loop_index >= loop_count - 1:
        return []
    return [_build_sync_all_stmt()]


def _build_sync_all_stmt() -> ast.stmt:
    """Build `pl.system.sync_all(core_type=pl.SyncCoreType.MIX)`.

    MIX because the statement sits at the kernel's top level, which both core types execute.
    """
    call = ast.Call(
        func=ast.Attribute(
            value=ast.Attribute(value=ast.Name(id="pl", ctx=ast.Load()), attr="system", ctx=ast.Load()),
            attr="sync_all",
            ctx=ast.Load(),
        ),
        args=[],
        keywords=[
            ast.keyword(
                arg="core_type",
                value=ast.Attribute(
                    value=ast.Attribute(value=ast.Name(id="pl", ctx=ast.Load()), attr="SyncCoreType", ctx=ast.Load()),
                    attr="MIX",
                    ctx=ast.Load(),
                ),
            ),
        ],
    )
    return ast.Expr(value=call, lineno=0)


# The serial path's task counter, playing the role the ctx slot's _pl_task_id plays in the
# full pipeline: the running task number that indexes each buffer's event-id tuple. A plain
# variable because that path has no ctx to carry it in.
_PL_SYNC_ID = "_pl_sync_id"


def _emit_sync_site(site, index_expr: ast.expr) -> ast.stmt:
    """Build the sync statement for one SyncSite."""
    event_id = _build_event_id_index(site.ids_node, site.slot_count, index_expr)
    return _build_system_sync_stmt(site.op, site.pipe, event_id)


def _sync_stmts_for(sites: list, index_expr: ast.expr):
    """(pre_stmts, post_stmts) for one stage from its planned SyncSites.

    Emitted unconditionally: a skewed edge's first few waits are covered by pre-fire rather
    than by a guard, so nothing here has to ask whether the partner task exists.
    """
    pre, post = [], []
    for out, planned in ((pre, sites[0]), (post, sites[1])):
        for site in planned:
            out.append(_emit_sync_site(site, index_expr))
    return pre, post


def _build_sync_plan(info: PipelineInfo):
    """Build the sync graph for the current schedule and plan every instruction from it.

    Called once per transform; the result travels down to the emitters as a value. Planning
    twice would allocate a second event-id group for every address-reuse edge.
    """
    schedule = _schedule_of(info)
    graph = build_graph(info, schedule)
    validate_local_slot_lifetime(info, schedule, analyze_local_slot_events(info, schedule))
    return plan_sync(graph, info)


def _schedule_of(info: PipelineInfo) -> list:
    """Which beat each stage runs on — the one input the sync graph needs from arrangement.

    Read off the stages rather than passed around: the delays ARE the schedule. A serial
    pipeline never runs _compute_delays, so its delays are still 0 and this yields an
    all-zero schedule — the degenerate case, not a separate one.
    """
    return [stage.delay for stage in info.stages]


def _sites_for_stage(sync: SyncPlan, stage) -> tuple:
    """(pre_sites, post_sites) for one stage, out of the already-planned in-loop sync."""
    return sites_for(sync.sites, stage.func_name)


def _build_lifted_id_decls(info: PipelineInfo, sync: SyncPlan) -> list[ast.stmt]:
    """Declare this pipeline's lifted variables: the fwd/bwd id lists it uses, then the id
    groups allocated for its address reuse.

    The buffer's ids_node was rewritten to a Name, so sync code uses `var[idx]` instead of
    the unsupported `(1,2)[idx]`. Shared by the full-pipeline path (_build_declarations) and
    the serial one (_build_serial_decls).

    Only the buffers THIS loop touches, so a kernel with several pipelines does not repeat
    every list ahead of each of them.
    """
    touched = buffers_touched(info)
    stmts = []
    entries = [
        (var_name, literal)
        for var_name, literal, buf_name in info.sync.lifted_ids
        if buf_name in touched
    ] + list(sync.reuse_ids)
    for var_name, literal_node in entries:
        stmts.append(
            ast.Assign(
                targets=[ast.Name(id=var_name, ctx=ast.Store())],
                value=copy.deepcopy(literal_node),
                lineno=0,
            )
        )
    return stmts


def _build_serial_decls(info: PipelineInfo, sync: SyncPlan) -> list[ast.stmt]:
    """Sync-only's pre-loop declarations: the lifted event-id variables + `_pl_sync_id = 0`.

    Declarations only; the caller appends the sync pre-fire, as on the full-pipeline path.
    The lifted ids must stay first either way — the pre-fire sets index those variables."""
    decls = list(_build_lifted_id_decls(info, sync))
    decls.append(
        ast.Assign(
            targets=[ast.Name(id=_sync_id_var(info.loop_index), ctx=ast.Store())],
            value=ast.Constant(value=0),
            lineno=0,
        )
    )
    return decls


def _insert_sync_into_loop_body(body: list[ast.stmt], sync: SyncPlan, info: PipelineInfo) -> None:
    """Insert wait(before)/set(after) cross-core sync around each stage call in the loop body,
    indexed by `_pl_sync_id` (current iteration, no delay).

    Which statement holds which stage is looked up by identity, as everywhere else. The
    section it holds contains that one call and nothing else, so the sync goes around
    ``body[0]`` and stays inside the section.
    """
    index_expr = ast.Name(id=_sync_id_var(info.loop_index), ctx=ast.Load())
    stage_of_stmt = {id(s.source_stmt): s for s in info.stages if s.source_stmt is not None}
    for stmt in body:
        stage = stage_of_stmt.get(id(stmt))
        if stage is None or not isinstance(stmt, ast.With):
            continue
        pre, post = _sync_stmts_for(_sites_for_stage(sync, stage), index_expr)
        slots = _select_gm_slots(stmt.body[0].value, index_expr, info)
        stmt.body[0:1] = pre + slots + stmt.body[0:1] + post


def _emit_prefire_and_drain(sync: SyncPlan) -> tuple[list[ast.stmt], list[ast.stmt]]:
    """Emit the plan's two out-of-loop halves: (sync_prefire, sync_drain).

    Pre-fire supplies the permits a skewed edge's first waits have no partner for; the drain
    consumes those permits and the sets the same skew strands at the end, so the counts
    match. Both sit outside the loop, where no task counter exists, so their event ids are
    indexed by a literal.

    The ``sync_`` prefix at the call sites separates them from the drain BEATS
    (_build_drain_beats), which sit after the same loop but let the delayed stages finish
    their tasks.
    """

    def emit(sites) -> list[ast.stmt]:
        by_section: dict[str, list] = {}
        for site in sites:
            by_section.setdefault(site.section, []).append((site.op, site.pipe, site.ids_node, site.id_index))
        out: list[ast.stmt] = []
        for section_kind, entries in by_section.items():
            body = [
                _build_system_sync_stmt(
                    op, pipe, ast.Subscript(value=copy.deepcopy(ids), slice=ast.Constant(value=idx), ctx=ast.Load())
                )
                for op, pipe, ids, idx in entries
            ]
            out.append(_wrap_in_section(section_kind, body))
        return out

    return emit(sync.prefire), emit(sync.drain)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def transform_pipeline(
    func_def: ast.FunctionDef,
    closure_vars: dict,
    config: PipelineConfig,
    facts: KernelFacts | None = None,
    tilingkey_consts: dict | None = None,
    datatype_consts: dict | None = None,
) -> ast.FunctionDef:
    """Transform a serial kernel function into a preload pipeline version.

    Args:
        func_def: The kernel function's AST (will not be mutated; a copy is made)
        closure_vars: Closure variables (contains stage functions, module constants)
        config: Pipeline configuration (depth, dump_generated, etc.)
        facts: what the probe parse learned about this kernel (see KernelFacts): the types
            behind the ctx fill, the constant-folded ``if`` conditions behind the dead-branch
            prune, and each buffer's memory space.
        tilingkey_consts: This launch key's concrete tilingkey field values.
        datatype_consts: This launch's concrete datatype symbol values.

    Returns:
        New ast.FunctionDef with pipeline transformation applied
    """
    facts = facts or KernelFacts()
    # Work on a deep copy to avoid mutating the original
    new_func = copy.deepcopy(func_def)

    # Same namespace the parser evaluates in (see ASTParser.__init__): a tilingkey field or
    # datatype symbol is a compile-time constant for this key but lives outside closure_vars,
    # and without it a `dtype=input_dtype` buffer silently loses its address range.
    closure_vars = {
        **(closure_vars or {}),
        **(tilingkey_consts or {}),
        **(datatype_consts or {}),
    }

    # Prune dead compile-time-constant if/elif/else branches that wrap stage calls,
    # so downstream analysis/transform only ever sees plain unconditional stages.
    _prune_const_branches(new_func, closure_vars, facts.if_const_map)

    # Analyze the serial structure. Everything about whether this kernel is legal is settled
    # in there, so what comes back is a shape that can be emitted: one PipelineInfo per
    # pipeline loop, in source order.
    infos = analyze_pipeline(new_func, closure_vars, config, facts)
    preloads = [config.preload_of(info.loop_index) for info in infos]

    # Arrangement first, for every loop, because the names below depend on the delays it
    # produces: the emitted body declares one ctx variable per distinct delay.
    for info, preload in zip(infos, preloads):
        if preload:
            _compute_delays(info, preload)

    # The names of EVERY pipeline at once: they share one function scope, so a user variable
    # colliding with any of them is a collision. A preload=0 loop introduces only its counter.
    validate_names(
        new_func,
        set().union(*(_pipeline_var_names(i) if p else _serial_var_names(i) for i, p in zip(infos, preloads))),
    )

    # Then emission, one loop at a time in source order. Each transform locates its own loop
    # afresh: the previous one spliced statements in, which moves the indices after it.
    for info, preload in zip(infos, preloads):
        sync = _build_sync_plan(info)
        if preload == 0:
            # Validation mode: keep the serial loop, only auto-insert cross-core sync.
            _transform_serial(new_func.body, info, sync, len(infos))
        else:
            # ctx ring buffer depth = max_delay + 1
            depth = max((s.delay for s in info.stages), default=0) + 1
            _transform_body(new_func.body, info, depth, sync, len(infos))

    ast.fix_missing_locations(new_func)
    return new_func


def _prune_const_branches(func_def: ast.FunctionDef, closure_vars: dict, if_const_map: dict) -> None:
    """Splice out compile-time-constant if/else around stage calls, in place.

    The taken branch's body replaces the whole ``if`` so downstream only sees plain
    unconditional stage calls. ``if_const_map`` maps a condition's source position
    ``(lineno, col_offset)`` -> (is_const, value), from a probe parse that reuses the
    parser's constant folding. Raises on a dynamic condition around a stage call.
    """
    stage_func_names = {name for name, val in closure_vars.items() if is_pipeline_stage(val)}
    func_def.body = _prune_stmts(func_def.body, stage_func_names, if_const_map)


def _prune_stmts(stmts: list[ast.stmt], stage_func_names: set, if_const_map: dict) -> list[ast.stmt]:
    """Return a new statement list with dead constant stage-bearing branches pruned.
    Recurses into For/With/If bodies; drops ``with`` sections that become empty."""
    out: list[ast.stmt] = []
    for stmt in stmts:
        if isinstance(stmt, ast.If):
            out.extend(_prune_if(stmt, stage_func_names, if_const_map))
        elif isinstance(stmt, ast.For):
            stmt.body = _prune_stmts(stmt.body, stage_func_names, if_const_map)
            stmt.orelse = _prune_stmts(stmt.orelse, stage_func_names, if_const_map)
            out.append(stmt)
        elif isinstance(stmt, ast.With):
            stmt.body = _prune_stmts(stmt.body, stage_func_names, if_const_map)
            if stmt.body:  # drop a section block emptied by pruning
                out.append(stmt)
        else:
            out.append(stmt)
    return out


def _prune_if(if_stmt: ast.If, stage_func_names: set, if_const_map: dict) -> list[ast.stmt]:
    """Prune a single ``if`` (recursing into elif chains via orelse). Returns the replacement
    statement list for this ``if``.

    Any compile-time-constant condition is pruned whatever its branches hold, so that the
    downstream scans see one unambiguous AST instead of declarations and stages that never
    execute.

    A runtime condition is left untouched, recursed into only to prune nested constant ones —
    but it must not wrap a stage call (CB3), which cannot be auto-arranged or synced.
    """
    # Keyed by the condition's source position, not its text: two sites can be written
    # the same way and fold to different values (see _record_if_const).
    is_const, value = if_const_map.get(
        (if_stmt.test.lineno, if_stmt.test.col_offset), (False, None)
    )

    if not is_const:
        # Runtime conditional: keep it, but a stage call directly inside a runtime
        # branch is unsupported (dynamic dispatch of stages).
        if _branch_contains_stage(if_stmt, stage_func_names):
            raise NotSupported(
                f"pipeline (CB3): branch condition '{ast.unparse(if_stmt.test)}' "
                f"wraps a stage call but is not a compile-time "
                "constant. Dynamic branch stages are unsupported; use a compile-time "
                "constant condition (e.g. a tiling-key field) or insert sync manually.",
                span=span_of(if_stmt.test),
            )
        if_stmt.body = _prune_stmts(if_stmt.body, stage_func_names, if_const_map)
        if_stmt.orelse = _prune_stmts(if_stmt.orelse, stage_func_names, if_const_map)
        return [if_stmt]

    # Compile-time constant: keep only the taken branch (elif chains are nested If
    # nodes in orelse; a plain else is a stmt list; no else → the if vanishes).
    taken = if_stmt.body if value else if_stmt.orelse
    return _prune_stmts(taken, stage_func_names, if_const_map)


def _branch_contains_stage(if_stmt: ast.If, stage_func_names: set) -> bool:
    """True if any branch of this if (then/elif/else, recursively) calls a stage."""
    for node in ast.walk(if_stmt):
        if isinstance(node, ast.Call) and call_name(node) in stage_func_names:
            return True
    return False


def _build_counter_incr(name: str) -> ast.stmt:
    """Build `<name> = <name> + 1`."""
    return ast.Assign(
        targets=[ast.Name(id=name, ctx=ast.Store())],
        value=ast.BinOp(
            left=ast.Name(id=name, ctx=ast.Load()),
            op=ast.Add(),
            right=ast.Constant(value=1),
        ),
        lineno=0,
    )


def _transform_serial(
    func_body: list[ast.stmt],
    info: PipelineInfo,
    sync: SyncPlan,
    loop_count: int,
) -> None:
    """Serial mode (preload 0): keep the loop structure, only auto-insert cross-core sync.

      1. Stage-to-stage sync goes INSIDE the pipeline loop: wait/set around each stage call,
         indexed by `_pl_sync_id % slot_count`.
      2. decls (lifted ids + `_pl_sync_id = 0`) + pre-fire go BEFORE the OUTERMOST loop and
         the sync drain AFTER it, so they run once and `_pl_sync_id` advances continuously.

    The cross-loop barrier closes the tail off after that drain (_cross_loop_barrier).

    No ctx / guard / delay.
    """
    # Serial: every stage runs in the same beat, so a zero schedule.
    sync_prefire, sync_drain = _emit_prefire_and_drain(sync)

    def transform_loop(site: _LoopSite):
        # Rule 1: stage sync + per-iteration counter, inside the innermost loop.
        # In place (return None); the loop node stays.
        _insert_sync_into_loop_body(site.pipeline_loop.body, sync, info)
        site.pipeline_loop.body.append(_build_counter_incr(_sync_id_var(info.loop_index)))
        return None

    # Rule 2: decls + pre-fire before the outermost loop, the sync drain after it.
    _place_around_pipeline_loop(
        func_body, info,
        transform_loop=transform_loop,
        before_loop=_build_serial_decls(info, sync) + sync_prefire,
        after_loop=sync_drain + _cross_loop_barrier(info, loop_count),
    )


def _place_around_pipeline_loop(
    func_body: list[ast.stmt],
    info: PipelineInfo,
    transform_loop,
    before_loop: list[ast.stmt],
    after_loop: list[ast.stmt],
) -> None:
    """Shared skeleton for both the serial and full-pipeline transforms.

    Locates the pipeline loop and its enclosing outermost loop, then:
      1. calls ``transform_loop(site)`` for the path-specific loop handling. It returns
         either ``None`` (loop mutated in place — serial) or a list of statements to REPLACE
         the pipeline loop node with (full pipeline). It must NOT mutate the tree, so the
         indices stay valid until the splices below.
      2. splices ``before_loop`` BEFORE the outermost loop and ``after_loop`` AFTER it.

    Both lists are named for WHERE they go, not what they hold: the two paths put different
    things there.
    """
    site = _locate_pipeline_loops(func_body, info)
    if site is None:
        # analyze_pipeline found this very node in this very tree, so failing to find it again
        # means the two searches have drifted apart. Refuse rather than emit the serial kernel
        # with neither ctx nor sync.
        raise CommonInner(
            "pipeline (internal): the pipeline loop the analyzer found is no longer "
            "reachable in the kernel body — please report."
        )
    replacement = transform_loop(site)

    top_level = site.pipeline_stmts is site.outer_stmts and site.pipeline_idx == site.outer_idx
    if replacement is not None and top_level:
        # Outermost loop IS the pipeline loop, replaced by N stmts. Splice everything in one go.
        site.outer_stmts[site.outer_idx:site.outer_idx + 1] = before_loop + replacement + after_loop
        return

    if replacement is not None:
        # Nested: replace the inner pipeline loop; the outer loop node is untouched,
        # so its (stmts, idx) stay valid for the splices below.
        site.pipeline_stmts[site.pipeline_idx:site.pipeline_idx + 1] = replacement

    # Splice the after-loop part at outer_idx+1 first (so outer_idx is unaffected).
    site.outer_stmts[site.outer_idx + 1:site.outer_idx + 1] = after_loop
    site.outer_stmts[site.outer_idx:site.outer_idx] = before_loop


@dataclass
class _LoopSite:
    """Where the pipeline loop lives, and its enclosing loop nest.

    - pipeline_stmts/pipeline_idx: the pipeline loop — where stage sync / loop replacement
      happens.
    - outer_stmts/outer_idx: the OUTERMOST enclosing loop — where declarations go before and
      the sync drain after (run-once placement). Coincides with the pipeline loop when that
      one is top-level.
    """
    pipeline_stmts: list  # statement list that directly holds the pipeline loop
    pipeline_idx: int     # index of the pipeline `for <pipeline_loop_var>` in pipeline_stmts
    outer_stmts: list     # statement list that holds the outermost enclosing loop
    outer_idx: int        # index of the outermost loop in outer_stmts

    @property
    def pipeline_loop(self) -> ast.For:
        return self.pipeline_stmts[self.pipeline_idx]

    @property
    def outer_loop(self) -> ast.For:
        return self.outer_stmts[self.outer_idx]


def _locate_pipeline_loops(func_body: list[ast.stmt], info: PipelineInfo) -> "_LoopSite | None":
    """Where this pipeline's two loops currently sit; None if either is unreachable.

    WHICH loop is the outermost one was settled by the analyzer (info.outer_loop); this only
    looks up where the two nodes sit right now, which transforming an earlier pipeline
    changes.

    Matched by node identity, never by loop-variable name: a kernel may hold several loops
    over the same variable, and matching by name would replace whichever came first.
    """
    pipeline_site = _find_loop_site(func_body, info.pipeline_loop)
    outer_site = _find_loop_site(func_body, info.outer_loop)
    if pipeline_site is None or outer_site is None:
        return None
    return _LoopSite(*pipeline_site, *outer_site)


def _find_loop_site(stmts: list[ast.stmt], target: ast.For | None):
    """Return (stmts, index) of ``target`` within ``stmts``, descending through For/With
    bodies, or None. Identity match — see _locate_pipeline_loops.
    """
    for i, stmt in enumerate(stmts):
        if stmt is target:
            return stmts, i
        if isinstance(stmt, (ast.For, ast.With)):
            found = _find_loop_site(stmt.body, target)
            if found is not None:
                return found
    return None


def _compute_delays(info, preload: int):
    """Compute each stage's delay from the preload value.

      - each core keeps its own stage counter, starting at 0;
      - a core's first stage: upstream cross-core stage's delay + 1, or 0 if there is none;
      - a later stage on the same core: previous same-core delay + preload.
    """
    # Track per-core stage counters and last delay
    core_stage_count = {"cube": 0, "vector": 0}
    core_last_delay = {"cube": -1, "vector": -1}

    for stage in info.stages:
        core = stage.section_kind
        stage_count = core_stage_count.get(core)
        last_delay = core_last_delay.get(core)
        if stage_count is None or last_delay is None:
            raise NotSupported(
                f"pipeline (internal): section kind '{core}' is neither cube nor vector, "
                f"which the analyzer should have refused. Please report."
            )

        if stage_count == 0:
            # First stage of this core
            if stage is info.stages[0]:
                # Very first stage overall
                stage.delay = 0
            else:
                # First stage of this core: previous stage (cross-core) delay + 1
                prev_idx = info.stages.index(stage) - 1
                stage.delay = info.stages[prev_idx].delay + 1
        else:
            # Subsequent stage of same core: previous same-core delay + preload
            stage.delay = last_delay + preload

        core_last_delay[core] = stage.delay
        core_stage_count[core] = stage_count + 1


# ---------------------------------------------------------------------------
# Generated names
#
# Every variable and struct the transform introduces is named here, each carrying the
# pipeline's loop index: several pipeline loops share one function scope, so a second loop
# reusing loop 0's ctx array would quietly write over it. Loop 0 keeps the bare names.
#
# Ctx FIELD names are the exception and are not built here — a field already lives in its
# own loop's struct — and so is the group cursor _pl_idx_<group>, shared on purpose
# because it models the group's own cursor and a group is declared once per kernel.


def _loop_suffix(loop_index: int) -> str:
    """`""` for the first pipeline, `"_<i>"` for the rest."""
    return "" if loop_index == 0 else f"_{loop_index}"


def _ctx_arr_name(loop_index: int) -> str:
    """The ctx ring buffer of one pipeline."""
    return f"_pl_ctx_arr{_loop_suffix(loop_index)}"


def _ctx_struct_name(loop_index: int) -> str:
    """The ctx struct type of one pipeline. Per loop because the FIELDS differ: each
    pipeline snapshots its own stages' arguments."""
    return f"PipeCtx{_loop_suffix(loop_index)}"


def _sync_id_var(loop_index: int) -> str:
    """The serial path's task counter (see _PL_SYNC_ID)."""
    return f"{_PL_SYNC_ID}{_loop_suffix(loop_index)}"


def _drain_var_name(loop_index: int) -> str:
    """The drain section's own loop variable.

    Suffixed like the rest because it is a name the transform puts in the user's scope, so
    it belongs in _pipeline_var_names, which must list the name actually emitted.
    """
    return f"_pl_drain{_loop_suffix(loop_index)}"


def _ctx_var_name(delay: int, loop_index: int) -> str:
    """The ctx variable a stage at this delay reads its beat from.

    Delay 0 reads the slot being filled this beat; a delayed stage reads the slot filled
    ``delay`` beats ago. Everything that names one of these goes through here, so the name a
    stage is given and the name its guard reads cannot come apart.
    """
    base = "_pl_ctx_0" if delay == 0 else f"_pl_ctx_neg{delay}"
    return f"{base}{_loop_suffix(loop_index)}"


def _pipeline_var_names(info: PipelineInfo) -> set[str]:
    """The fixed variables the FULL PIPELINE path introduces.

    One ctx variable per distinct delay, plus the ctx array, the task counter and the drain
    loop's variable. The serial path emits none of these — see _serial_var_names.
    """
    names = {
        _ctx_arr_name(info.loop_index),
        task_id_var(info.loop_index),
        _drain_var_name(info.loop_index),
    }
    for stage in info.stages:
        names.add(_ctx_var_name(stage.delay, info.loop_index))
    return names


def _serial_var_names(info: PipelineInfo) -> set[str]:
    """The fixed variables the SERIAL path introduces: just its task counter.

    That path keeps the serial loop and only threads a counter through the event-id indexing.
    """
    return {_sync_id_var(info.loop_index)}


def _transform_body(
    func_body: list[ast.stmt],
    info: PipelineInfo,
    depth: int,
    sync: SyncPlan,
    loop_count: int,
) -> None:
    """Full-pipeline transform, on the shared _place_around_pipeline_loop skeleton:

      1. The pipeline loop is REPLACED by its preload version (_build_pipeline_loop: ctx ring
         buffer + delay + is_valid guards).
      2. Declarations (_pl_ctx_arr, _pl_task_id, lifted ids, pre-fire) go BEFORE the
         OUTERMOST loop; AFTER it come the drain BEATS that let the delayed stages catch up,
         then the sync DRAIN that balances set(bwd)/wait(bwd), then the cross-loop barrier
         (_cross_loop_barrier).
    """
    # Make the index behind every outside-the-loop slot pick explicit first, so the ctx can
    # carry it and each stage re-selects its own beat's slot.
    _rewrite_outer_slot_picks(info)

    def transform_loop(site: _LoopSite):
        # Pure: only build the replacement. Does NOT mutate the tree here, so
        # pipeline_idx stays valid for the skeleton's splice.
        return _build_pipeline_loop(site.pipeline_loop, info, depth, sync)

    sync_prefire, sync_drain = _emit_prefire_and_drain(sync)
    _place_around_pipeline_loop(
        func_body, info,
        transform_loop=transform_loop,
        before_loop=_build_declarations(info, depth, sync) + sync_prefire,
        # Drain beats first, then the sync drain: the beats emit the set(bwd) the last
        # tasks still owe, which the waits in the sync drain then consume.
        after_loop=_build_drain_beats(info, depth, sync) + sync_drain + _cross_loop_barrier(info, loop_count),
    )


def _field_init(info: PipelineInfo, field_name: str, section_kind: str) -> ast.expr:
    """The initializer a ctx field is declared with in ``section_kind``. The two bookkeeping
    fields are integers."""
    if field_name in info.ctx_sources:
        return info.ctx_sources[field_name][2][section_kind]
    return ast.Constant(value=0)


def _ctx_array_assign(info: PipelineInfo, depth: int, section_kind: str) -> ast.Assign:
    """`_pl_ctx_arr = pl.struct_array(depth, 'PipeCtx', <field>=<init>, ...)` as
    ``section_kind`` declares it."""
    keywords = [ast.keyword(arg=f, value=_field_init(info, f, section_kind)) for f in info.ctx_fields]
    ctx_call = ast.Call(
        func=ast.Attribute(value=ast.Name(id="pl", ctx=ast.Load()), attr="struct_array", ctx=ast.Load()),
        args=[ast.Constant(value=depth), ast.Constant(value=_ctx_struct_name(info.loop_index))],
        keywords=keywords,
    )
    return ast.Assign(
        targets=[ast.Name(id=_ctx_arr_name(info.loop_index), ctx=ast.Store())],
        value=ctx_call,
        lineno=0,
    )


def _build_declarations(info: PipelineInfo, depth: int, sync: SyncPlan) -> list[ast.stmt]:
    """The full pipeline's pre-loop declarations: `_pl_ctx_arr`, `_pl_task_id`, the lifted
    event-id variables and the outer-slot rotation counters.

    Declarations only; the sync pre-fire that also sits before the loop is appended by the
    caller.
    """
    stmts = []

    # Each target compiles its own ctx array, so a field whose type differs between the two
    # sections is declared once per section; otherwise one declaration serves both.
    ctx_assigns = {kind: _ctx_array_assign(info, depth, kind) for kind in ("cube", "vector")}
    if ast.dump(ctx_assigns["cube"]) == ast.dump(ctx_assigns["vector"]):
        stmts.append(ctx_assigns["cube"])
    else:
        stmts.extend(_wrap_in_section(kind, [assign]) for kind, assign in ctx_assigns.items())

    _pl_task_id_assign = ast.Assign(
        targets=[ast.Name(id=task_id_var(info.loop_index), ctx=ast.Store())],
        value=ast.Constant(value=0),
        lineno=0,
    )
    stmts.append(_pl_task_id_assign)

    # Lifted ids (shared with the serial path).
    stmts.extend(_build_lifted_id_decls(info, sync))

    # Rotation counters for slots picked outside the pipeline loop. Starting at `slots - 1`
    # mirrors pl.make_tile_group's own cursor, so the first advance hands out slot 0.
    for group, slots in sorted({(g, n) for g, _kind, n in info.outer_slots.values()}):
        stmts.append(
            ast.Assign(
                targets=[ast.Name(id=slot_index_field(group), ctx=ast.Store())],
                value=ast.Constant(value=slots - 1),
                lineno=0,
            )
        )

    return stmts


def _ctx_field_assign(info: PipelineInfo, attr: str, value_node: ast.expr) -> ast.Assign:
    """Build `_pl_ctx_0.<attr> = <value_node>`."""
    return ast.Assign(
        targets=[
            ast.Attribute(
                value=ast.Name(id=_ctx_var_name(0, info.loop_index), ctx=ast.Load()),
                attr=attr,
                ctx=ast.Store(),
            )
        ],
        value=value_node,
        lineno=0,
    )


def _classify_loop_body_stmts(original_for: ast.For, info: PipelineInfo):
    """Walk the pipeline loop body IN ORDER and return an ordered_body list.

    Each entry is either a ("STAGE", stage_idx) marker for the statement holding a stage
    call, or one of the user's own statements, kept verbatim and IN PLACE.

    Which statement holds which stage is looked up from info.stages, not decided again:
    counting every ``ast.With`` as a stage shifts every stage after an unrelated section onto
    the wrong entry.

    Everything that is not a stage call is preserved: written in the loop body it ran once
    per iteration, and kept there it runs once per beat. The transform never models how a
    value was produced, so a branch, a helper call or several statements building one value
    all behave the same.
    """
    stage_of_stmt = {
        id(stage.source_stmt): idx
        for idx, stage in enumerate(info.stages)
        if stage.source_stmt is not None
    }

    ordered_body = []
    for stmt in original_for.body:
        stage_idx = stage_of_stmt.get(id(stmt))
        if stage_idx is not None:
            ordered_body.append(("STAGE", stage_idx))
        else:
            ordered_body.append(copy.deepcopy(stmt))

    return ordered_body


def _build_pipeline_loop(
    original_for: ast.For, info: PipelineInfo, depth: int, sync: SyncPlan
) -> list[ast.stmt]:
    """Build the transformed pipeline for-loop."""

    new_for = copy.deepcopy(original_for)

    # The loop body with each stage replaced by a marker, everything else in place.
    ordered_body = _classify_loop_body_stmts(original_for, info)
    first_stage = next((i for i, item in enumerate(ordered_body) if isinstance(item, tuple)), len(ordered_body))

    # Assemble new loop body
    new_body = []
    # 1. Pick this beat's ctx slot and set is_valid.
    new_body.extend(_build_ctx_slot_pick(info, depth))
    # 2. Whatever the user wrote before the first stage, in place: the snapshot comes after
    #    it, so a value produced here is read where it stands.
    new_body.extend(ordered_body[:first_stage])
    # 3. Snapshot every ctx field, before every stage, so each beat hands all its stages one
    #    consistent set of values.
    new_body.extend(_build_ctx_field_fills(info))
    # 4. The stage chain. Nothing sits between stages, so every statement after the first
    #    stage belongs to the tail (zone 3).
    for item in ordered_body[first_stage:]:
        if isinstance(item, tuple):
            new_body.extend(_build_stage_call(info.stages[item[1]], item[1], info, depth, sync))
    # 5. task_id increment, before the tail. The beat is committed once its stages are
    #    dispatched: the slot is taken and its ctx snapshotted, and a delayed stage consumes
    #    it on a later beat. The counter must advance even when the tail is skipped, or the
    #    next beat re-picks the same slot and overwrites a ctx still waiting to be consumed.
    new_body.append(_build_counter_incr(task_id_var(info.loop_index)))
    # 6. Whatever the user wrote after the last stage, in place. It prepares the next
    #    iteration, so this beat's snapshot above must not see it.
    for item in ordered_body[first_stage:]:
        if not isinstance(item, tuple):
            new_body.append(item)

    new_for.body = new_body
    return [new_for]


def _build_drain_beats(info: PipelineInfo, depth: int, sync: SyncPlan) -> list[ast.stmt]:
    """The beats that run after the outer loop to let the deepest stage catch up.

    A delayed stage is always ``delay`` beats behind, so when the task stream ends its last
    few tasks are still in flight. Draining after the OUTER loop ties this to the end of the
    whole stream: appended to the last outer iteration's inner loop instead, a ``continue``
    above it would skip the drain and strand a set(bwd).

    The beat count is the loop bound (``depth - 1``) while the number of stage blocks is the
    stage list, so a deeper pipeline lengthens the loop rather than the code. Only stages
    with a delay are emitted: a delay-0 stage reads this beat's own ctx slot, which is marked
    invalid here.
    """
    if depth <= 1:
        return []
    body: list[ast.stmt] = [
        _build_ctx_slot_assign(info, depth),
        # No new task this beat. Each stage still checks its OWN (delayed) slot, so it keeps
        # firing while a real task remains behind it.
        _ctx_field_assign(info, PL_IS_VALID_FIELD, ast.Constant(value=0)),
        # Field on the left, this loop's counter variable on the right — see task_id_var.
        _ctx_field_assign(info, PL_TASK_ID_FIELD, ast.Name(id=task_id_var(info.loop_index), ctx=ast.Load())),
    ]
    for stage_idx, stage in enumerate(info.stages):
        if stage.delay:
            body.extend(_build_stage_call(stage, stage_idx, info, depth, sync))
    body.append(_build_counter_incr(task_id_var(info.loop_index)))

    return [
        ast.For(
            target=ast.Name(id=_drain_var_name(info.loop_index), ctx=ast.Store()),
            iter=ast.Call(
                func=ast.Attribute(value=ast.Name(id="pl", ctx=ast.Load()), attr="range", ctx=ast.Load()),
                args=[ast.Constant(value=0), ast.Constant(value=depth - 1)],
                keywords=[],
            ),
            body=body,
            orelse=[],
            lineno=0,
        )
    ]


def _build_ctx_slot_assign(info: PipelineInfo, depth: int) -> ast.Assign:
    """`_pl_ctx_0 = _pl_ctx_arr[_pl_task_id % depth]` — this beat's slot."""
    return ast.Assign(
        targets=[ast.Name(id=_ctx_var_name(0, info.loop_index), ctx=ast.Store())],
        value=ast.Subscript(
            value=ast.Name(id=_ctx_arr_name(info.loop_index), ctx=ast.Load()),
            slice=ast.BinOp(
                left=ast.Name(id=task_id_var(info.loop_index), ctx=ast.Load()),
                op=ast.Mod(),
                right=ast.Constant(value=depth),
            ),
            ctx=ast.Load(),
        ),
        lineno=0,
    )


def _build_ctx_slot_pick(info: PipelineInfo, depth: int) -> list[ast.stmt]:
    """Pick this beat's ctx slot and set is_valid.

    The field values are filled separately (_build_ctx_field_fills), later in the body: they
    snapshot the user's variables and so must run after the statements that set them.
    """
    return [_build_ctx_slot_assign(info, depth), _build_is_valid_guard(info)]


def _build_is_valid_guard(info: PipelineInfo) -> ast.If:
    """Build `if <loop_var> < <range_end>: _pl_ctx_0._pl_is_valid = 1 else: = 0`.

    The end bound is always present: _record_pipeline_loop_info rejects a pipeline loop it
    cannot extract one from (L6).
    """
    return ast.If(
        test=ast.Compare(
            left=ast.Name(id=info.pipeline_loop_var, ctx=ast.Load()),
            ops=[ast.Lt()],
            comparators=[copy.deepcopy(info.pipeline_loop_end)],
        ),
        body=[_ctx_field_assign(info, PL_IS_VALID_FIELD, ast.Constant(value=1))],
        orelse=[_ctx_field_assign(info, PL_IS_VALID_FIELD, ast.Constant(value=0))],
        lineno=0,
    )


def _build_ctx_field_fills(info: PipelineInfo) -> list[ast.stmt]:
    """Build `_pl_ctx_0.<field> = <value>` for EVERY ctx field except the validity flag.

    Each field is filled by READING its source, at a point where the user's statements for
    this beat have already run (see _build_pipeline_loop). Snapshot rather than recompute, so
    a value assigned in a branch, inside a helper, or across several statements all behave
    the same.

    Each field's fill comes from info.ctx_sources, recorded when the field was created. The
    task counter is the exception, being the transform's own (see _astutil.task_id_var).

    A source living inside a ``pl.section_*()`` block gets its fill wrapped in that same
    section, since the other target's parse never binds that name. A source bound in BOTH
    sections gets a fill in each — the two targets compile to separate functions with
    separate ctx arrays.
    """
    plain: list[ast.stmt] = []
    by_section: dict[str, list[ast.stmt]] = {}
    for field_name in info.ctx_fields:
        if field_name == PL_IS_VALID_FIELD:
            continue
        if field_name == PL_TASK_ID_FIELD:
            # Field and variable share a name only while the kernel holds a single pipeline.
            source, sections = ast.Name(id=task_id_var(info.loop_index), ctx=ast.Load()), set()
        elif field_name in info.ctx_sources:
            source, sections, _init = info.ctx_sources[field_name]
        else:
            raise KeyNotFound(
                f"pipeline (internal): ctx field '{field_name}' has no source — fields are "
                f"registered together with their fill (_derive_ctx_fields.register), so the "
                f"two have drifted apart. Please report."
            )
        if not sections:
            plain.append(_ctx_field_assign(info, field_name, copy.deepcopy(source)))
            continue
        for kind in sections:
            by_section.setdefault(kind, []).append(_ctx_field_assign(info, field_name, copy.deepcopy(source)))

    return plain + [_wrap_in_section(kind, body) for kind, body in sorted(by_section.items())]


def _build_stage_ctx_lookup(stage, info: PipelineInfo, depth: int) -> tuple[str, list[ast.stmt]]:
    """Return the ctx variable name and optional delayed ctx lookup statements."""
    delay = stage.delay
    ctx_var = _ctx_var_name(delay, info.loop_index)
    if delay == 0:
        return ctx_var, []

    ctx_assign = ast.Assign(
        targets=[ast.Name(id=ctx_var, ctx=ast.Store())],
        value=ast.Subscript(
            value=ast.Name(id=_ctx_arr_name(info.loop_index), ctx=ast.Load()),
            slice=ast.BinOp(
                left=ast.BinOp(
                    left=ast.Name(id=task_id_var(info.loop_index), ctx=ast.Load()),
                    op=ast.Add(),
                    right=ast.Constant(value=depth - delay),
                ),
                op=ast.Mod(),
                right=ast.Constant(value=depth),
            ),
            ctx=ast.Load(),
        ),
        lineno=0,
    )
    return ctx_var, [ctx_assign]


def _build_stage_args(stage, arg_mapping: list, ctx_var: str, info: PipelineInfo) -> list[ast.expr]:
    """Build stage call args, replacing ctx-backed args with ctx.field references.

    A struct argument is replaced by the ctx slot itself: the slot carries the struct's
    fields under their own names, so a stage body reading `ri.ki` needs no rewriting.
    """
    new_args = []
    for i, orig_arg in enumerate(stage.args):
        if i < len(arg_mapping) and arg_mapping[i] is not None:
            field_name = arg_mapping[i]
            if field_name == PL_STRUCT_ARG:
                new_args.append(ast.Name(id=ctx_var, ctx=ast.Load()))
                continue
            if isinstance(orig_arg, ast.Name) and orig_arg.id in info.outer_slots:
                # Re-select the slot with this stage's own beat index, so a delayed stage
                # gets the slot from the iteration whose data it is handling.
                group = info.outer_slots[orig_arg.id][0]
                new_args.append(
                    ast.Subscript(
                        value=ast.Name(id=group, ctx=ast.Load()),
                        slice=ast.Attribute(
                            value=ast.Name(id=ctx_var, ctx=ast.Load()),
                            attr=field_name,
                            ctx=ast.Load(),
                        ),
                        ctx=ast.Load(),
                    )
                )
                continue
            new_args.append(
                ast.Attribute(
                    value=ast.Name(id=ctx_var, ctx=ast.Load()),
                    attr=field_name,
                    ctx=ast.Load(),
                )
            )
        else:
            new_args.append(copy.deepcopy(orig_arg))
    return new_args


def _gm_buffers(info: PipelineInfo) -> dict:
    """``{name: CrossCoreBuffer}`` for the GM buffers that rotate through more than one slot."""
    return {
        name: buf
        for name, buf in info.sync.buffers.items()
        if buf.facts.memory == MemorySpace.DDR and info.sync.slot_counts.get(name, 1) > 1
    }


def _gm_slot_var(name: str) -> str:
    """The variable holding a GM buffer's slot for one stage call."""
    return f"_pl_gm_{name}"


def _shape_of(name: str) -> ast.Attribute:
    """``<name>.shape``."""
    return ast.Attribute(value=ast.Name(id=name, ctx=ast.Load()), attr="shape", ctx=ast.Load())


def _stride_of(name: str) -> ast.Attribute:
    """``<name>.stride``."""
    return ast.Attribute(value=ast.Name(id=name, ctx=ast.Load()), attr="stride", ctx=ast.Load())


def _axis_of(layout: ast.expr, axis: int) -> ast.Subscript:
    """``<layout>[axis]``."""
    return ast.Subscript(value=copy.deepcopy(layout), slice=ast.Constant(value=axis), ctx=ast.Load())


def _gm_slot_extent(name: str, buf: CrossCoreBuffer) -> ast.expr:
    """How many elements one slot of a GM buffer spans: the number the parse folded, else
    ``max(shape[i] * stride[i])``, or ``prod(shape)`` with no stride."""
    if buf.facts.slot_size is not None:
        return ast.Constant(value=buf.facts.slot_size)
    axes = [_axis_of(_shape_of(name), axis) for axis in range(buf.facts.dims)]
    if buf.facts.has_stride:
        spans = [
            ast.BinOp(left=axis, op=ast.Mult(), right=_axis_of(_stride_of(name), index))
            for index, axis in enumerate(axes)
        ]
        size = spans[0]
        for span in spans[1:]:
            size = ast.Call(func=ast.Name(id="max", ctx=ast.Load()), args=[size, span], keywords=[])
    else:
        size = axes[0]
        for axis in axes[1:]:
            size = ast.BinOp(left=size, op=ast.Mult(), right=axis)
    return ast.fix_missing_locations(size)


def _select_gm_slots(call: ast.Call, index_expr: ast.expr, info: PipelineInfo) -> list[ast.stmt]:
    """Hand a stage call the slot of each GM buffer its task uses, in place.

    Each argument that reaches such a buffer is replaced by ``_pl_gm_<declared name>``; the
    returned statements bind it to slot ``index_expr % depth``. The argument is resolved
    through the probe's buffer_vars, so an argument that only aliases the buffer selects the
    slot the way the declared name would — and under the declared name, so two spellings of
    one buffer share the one assignment.
    """
    gm = _gm_buffers(info)
    stmts: list[ast.stmt] = []
    bound: set[str] = set()
    for pos, arg in enumerate(call.args):
        if not isinstance(arg, ast.Name):
            continue
        reached = [name for name in info.buffer_vars.get(arg.id, (arg.id,)) if name in gm]
        if not reached:
            continue
        if len(reached) > 1:
            raise NotSupported(
                f"pipeline: argument '{arg.id}' of this stage call reaches more than one GM "
                f"buffer ({', '.join(sorted(reached))}), so which buffer's slot the task "
                f"should be handed depends on the path taken. The slot is selected at the "
                f"call site, before the branch is known.\n"
                f"Pass one buffer, choosing between them inside the stage.",
                span=span_of(arg),
            )
        name = reached[0]
        if name not in bound:
            stmts.append(_build_gm_slot_assign(name, gm[name], index_expr, info))
            bound.add(name)
        call.args[pos] = ast.Name(id=_gm_slot_var(name), ctx=ast.Load())
    return stmts


def _build_gm_slot_assign(
    name: str, buf: CrossCoreBuffer, index_expr: ast.expr, info: PipelineInfo
) -> ast.Assign:
    """``_pl_gm_<name> = pl.make_tensor(pl.addptr(pl.make_ptr(<name>), k * size), <name>.shape[, stride])``."""

    def pl_call(fn: str, args: list) -> ast.Call:
        return ast.Call(
            func=ast.Attribute(value=ast.Name(id="pl", ctx=ast.Load()), attr=fn, ctx=ast.Load()),
            args=args,
            keywords=[],
        )

    slot_idx = ast.BinOp(
        left=copy.deepcopy(index_expr), op=ast.Mod(), right=ast.Constant(value=info.sync.slot_counts[name])
    )
    offset = ast.BinOp(left=slot_idx, op=ast.Mult(), right=_gm_slot_extent(name, buf))
    ptr = pl_call("addptr", [pl_call("make_ptr", [ast.Name(id=name, ctx=ast.Load())]), offset])
    layout = [_shape_of(name)]
    if buf.facts.has_stride:
        layout.append(_stride_of(name))
    return ast.Assign(
        targets=[ast.Name(id=_gm_slot_var(name), ctx=ast.Store())],
        value=pl_call("make_tensor", [ptr, *layout]),
        lineno=0,
    )


def _wrap_in_section(section_kind: str, body: list[ast.stmt]) -> ast.With:
    """Wrap statements in `with pl.section_<kind>():`."""
    section_call = ast.Call(
        func=ast.Attribute(value=ast.Name(id="pl", ctx=ast.Load()), attr=f"section_{section_kind}", ctx=ast.Load()),
        args=[],
        keywords=[],
    )
    return ast.With(
        items=[ast.withitem(context_expr=section_call, optional_vars=None)],
        body=body,
        lineno=0,
    )


def _build_guarded_stage_body(
    stage, call_expr: ast.Call, ctx_var: str, sync: SyncPlan, info: PipelineInfo
) -> list[ast.stmt]:
    """Build sync + stage-call statements that run under the ctx validity guard."""
    index_expr = ast.Attribute(
        value=ast.Name(id=ctx_var, ctx=ast.Load()),
        attr=PL_TASK_ID_FIELD,
        ctx=ast.Load(),
    )

    # Every cross-core sync for this stage comes from the graph: RAW and WAR,
    # same-buffer and address-reuse alike (see _sync_graph.plan_sync_sites).
    sites = _sites_for_stage(sync, stage)
    auto_pre, auto_post = _sync_stmts_for(sites, index_expr)
    slots = _select_gm_slots(call_expr, index_expr, info)

    return [*auto_pre, *slots, ast.Expr(value=call_expr, lineno=0), *auto_post]


def _wrap_stage_section(stage, guarded_body: list[ast.stmt], ctx_var: str) -> ast.With:
    """Guard a stage body on the ctx validity flag, and wrap that in its pl.section_*."""
    guarded_call = ast.If(
        test=ast.Attribute(
            value=ast.Name(id=ctx_var, ctx=ast.Load()),
            attr=PL_IS_VALID_FIELD,
            ctx=ast.Load(),
        ),
        body=guarded_body,
        orelse=[],
        lineno=0,
    )
    section_attr = f"section_{stage.section_kind}"
    section_call = ast.Call(
        func=ast.Attribute(value=ast.Name(id="pl", ctx=ast.Load()), attr=section_attr, ctx=ast.Load()),
        args=[],
        keywords=[],
    )
    return ast.With(
        items=[ast.withitem(context_expr=section_call, optional_vars=None)],
        body=[guarded_call],
        lineno=0,
    )


def _build_stage_call(
    stage, stage_idx: int, info: PipelineInfo, depth: int, sync: SyncPlan
) -> list[ast.stmt]:
    """Build a single stage call with delay, ctx lookup, and is_valid guard.

    """
    ctx_var, stmts = _build_stage_ctx_lookup(stage, info, depth)

    # Build the stage function call with args replaced
    arg_mapping = info.stage_arg_mapping[stage_idx]
    call_expr = ast.Call(
        func=ast.Name(id=stage.func_name, ctx=ast.Load()),
        args=_build_stage_args(stage, arg_mapping, ctx_var, info),
        keywords=[],
    )

    # sync + stage call + the user's own surrounding statements, all INSIDE the is_valid
    # guard: a fill or drain beat carries no task, so letting its wait/set run would consume
    # permits belonging to a later real one. The event-id index reads `<ctx_var>._pl_task_id`,
    # the task THIS stage is handling, not the beat number.
    guarded_body = _build_guarded_stage_body(stage, call_expr, ctx_var, sync, info)
    stmts.append(_wrap_stage_section(stage, guarded_body, ctx_var))

    return stmts
