#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Checks on a kernel's pipeline structure, gathered in one place, grouped by what they
need to already know:

``validate_kernel_structure``  every pipeline in the kernel, before any per-loop work
``validate_structure(info)``   after one pipeline's stages, sections, loop info and ctx
                               fields are known, before any sync scanning
``validate_sync(...)``         after the sync graph exists (see _sync_graph)
``validate_names(...)``        once the caller knows which variables it will introduce —
                               both emission paths, each passing its own names

Format checks stay at their point of use: if a buffer's memory space or its id tuple
cannot be resolved, scanning cannot continue, so those raise where they are read.
"""

from __future__ import annotations

import ast

from ..._errors import InvalidArgument, InvalidOperation, InvalidType, NotSupported, span_of
from ._astutil import call_name, get_funcdef, slot_accessor


def buffer_users(graph) -> dict:
    """``{buffer: [(stage_idx, stage, section, roles), ...]}``, ordered by stage.

    One entry per stage that touches the buffer, with ``roles`` the set of op-level roles it
    uses there. Also the basis for producer vs consumer: whichever stage comes FIRST produces
    the data, later ones consume it. Op-level roles cannot decide that — a consumer may write
    too (an in-place ``pl.muls``), so "consumer" means "uses the buffer", not "only reads it".
    """
    by_buf: dict = {}
    for acc in graph.accesses:
        per_stage = by_buf.setdefault(acc.buffer, {})
        entry = per_stage.setdefault(acc.stage_idx, [acc.stage, acc.section, set()])
        entry[2].add(acc.role)
    return {
        buf: [(idx, *per_stage[idx]) for idx in sorted(per_stage)]
        for buf, per_stage in by_buf.items()
    }


def validate_kernel_buffers(sync, used: set) -> None:
    """Every declared cross-core buffer must be used by at least one pipeline.

    ``used`` is every buffer any pipeline touches, handed in rather than gathered here so
    this module keeps importing nothing but its own leaf helper.

    The per-loop half lives in _check_cross_core_users, which cannot say anything here: a
    buffer belonging to another loop is legitimately absent from this one. Unused ANYWHERE
    still means ids were declared for a handover that never happens.
    """
    for buf_name, buf in sync.buffers.items():
        if buf.fwd_ids_node is None and buf.bwd_ids_node is None:
            continue  # not a cross-core buffer
        if buf_name not in used:
            raise InvalidOperation(
                f"pipeline: cross-core buffer '{buf_name}' declares fwd_ids/bwd_ids but no "
                f"stage in any pipeline loop touches it, so there is no cross-core handover "
                f"to synchronise. Either drop those keywords or pass this buffer to the "
                f"stages that share it."
            )


def validate_sync(graph, info, cycles, schedule) -> None:
    """Checks that need the sync graph: regions, op-level accesses, users, edges.

    Called from _sync_graph once the graph is built. ``cycles`` is what ``find_cycles(graph)``
    returned, passed in because walking the graph is the graph's job while deciding what a
    cycle's total distance MEANS is a check.

    ``_check_slot_counts`` runs before the two edge checks on purpose: too few tiles shows up
    there as an unstable distance or a non-positive cycle, and those messages describe the
    symptom rather than the tile count that caused it.
    """
    users = buffer_users(graph)
    _check_cross_core_users(info, users)
    _check_event_id_counts(graph, info)
    _check_reuse_mutex_ids(graph, info)
    _check_slot_counts(graph, info, users, schedule)
    _check_stable_distances(graph)
    _check_no_deadlock_cycle(cycles)


def _check_slot_counts(graph, info, users: dict, schedule: list) -> None:
    """A cross-core buffer needs one tile per beat its two stages sit apart, plus one.

    While the producer works on task t its consumer is still on task ``t - dist``, so tasks
    ``t - dist .. t`` are all live at once: the one being written, the one being read, and
    every one in between that is written but not yet read. One tile short and the producer
    lands on the tile the consumer is reading.

    ``dist`` is read from the schedule rather than from the graph's edges: once the tiles run
    out the edges themselves are distorted (the producer laps the consumer, and the lane scan
    then pairs a read with a write from a later task), so the edge no longer states the
    distance the schedule asked for.
    """
    for buf_name, buf in info.sync.buffers.items():
        if buf.fwd_ids_node is None and buf.bwd_ids_node is None:
            continue  # not a cross-core buffer: a local group's rotation is the user's own
        region = graph.regions.get(buf_name)
        entries = users.get(buf_name, [])
        if region is None or len(entries) != 2:
            continue  # unused here, or already reported by _check_cross_core_users
        producer, consumer = entries[0], entries[1]
        needed = schedule[consumer[0]] - schedule[producer[0]] + 1
        have = graph.slots[region]
        if have >= needed:
            continue
        raise InvalidArgument(
            f"pipeline: cross-core buffer '{buf_name}' has {have} tile(s), but at this "
            f"preload its producer '{producer[1]}' runs {needed - 1} iteration(s) ahead of "
            f"its consumer '{consumer[1]}', so {needed} tiles are live at once. Give it "
            f"{needed} tiles (the length of mutex_ids, and of fwd_ids/bwd_ids unless those "
            f"are a single shared id), or lower `preload`."
        )


def _check_stable_distances(graph) -> None:
    """Every edge must sit at the same distance on every task.

    The whole plan rests on one distance per edge: it decides the skew, and with it how many
    permits pre-fire releases, how many waits drain consumes, and which task each
    ``% slot_count`` event id belongs to. ``build_graph`` expands two full rotations past the
    fill so it has a second sample to compare against; an edge whose distance differs between
    them violates assumption A1, and emitting anyway would pair every wait with a set
    belonging to some other task.
    """
    unstable = [edge for edge in graph.edges if edge.unstable]
    if not unstable:
        return
    detail = "\n".join(
        f"  {edge.kind} {edge.src} -> {edge.dst}   "
        f"(dist={edge.dist:+d}, task_off={edge.task_off:+d}, {edge.slot_count} slots)"
        for edge in unstable
    )
    raise InvalidOperation(
        "pipeline: these cross-core dependencies never reach a steady state — the distance "
        "between their two ends changes from task to task, so no single wait/set placement "
        "covers them:\n"
        f"{detail}\n"
        "This means a stage advances one of these buffers a different number of times per "
        "iteration than the slot timeline (task % slot_count) assumes. Take exactly one "
        "slot per buffer per stage, or give the buffer its own tile_group per handover."
    )


def _check_no_deadlock_cycle(cycles) -> None:
    """A dependency cycle whose distances sum to <= 0 cannot be synchronised at all.

    Every edge means "the destination waits for the source", so a cycle closing on the same
    task or a later one asks each side to wait for the other and hangs on hardware.

    A positive total is healthy — the cycle closes on an EARLIER task, an ordinary
    cross-iteration pipeline. A single negative edge is also fine: an inverse-time edge (wait
    placed before the matching set) works on hardware. Only the total decides.
    """
    deadlocks = [(path, total) for path, total in cycles if total <= 0]
    if not deadlocks:
        return
    routes = "\n".join(f"  {_format_cycle(path, total)}" for path, total in sorted(deadlocks, key=lambda c: c[1]))
    raise InvalidOperation(
        "pipeline: the stage/buffer dependencies form a cycle that no cross-core sync can "
        "satisfy (its distances sum to <= 0, i.e. each side would wait for the other):\n"
        f"{routes}\n"
        "Break the cycle by reordering the stages, giving one of the buffers its own "
        "memory instead of sharing an address, or adding a slot so the dependency reaches "
        "back to an earlier task."
    )


def _format_cycle(path: list, total: int) -> str:
    """Render one cycle from find_cycles() as a readable route."""
    route = " -> ".join(f"{a.stage}[{a.buffer} {a.role} {a.pipe}]" for a in path)
    return f"{route}   (total distance={total})"


def _check_cross_core_users(info, users: dict) -> None:
    """A buffer declared with fwd/bwd ids must hand data from one core to another.

    Counted PER PIPELINE LOOP, so the same buffer may be handed over in each of several
    loops, each handover a complete pair of its own.

      no users      this loop does not use the buffer; another one does. Legal. Unused by
                    EVERY loop is caught once, in validate_kernel_buffers.
      one user      nobody on the other side WITHIN this loop. Its sets pile up unanswered,
                    and the hardware faults once an id is set more than 15 times unmatched.
                    A handover reaching across two loops has this shape and is not supported.
      same core     both users on cube, or both on vector — no core boundary is crossed.
      not adjacent  the two stages have others between them. The handover then has to
                    survive those stages' whole span, which costs one more tile per stage
                    crossed and is not a shape any kernel needs.
      3+ users      one variable standing in for several producer/consumer pairs IN ONE LOOP.
                    Ids are resolved per buffer, so two pairs would share one id group and
                    their differently-skewed edges would interleave. Declare a separate
                    tile_group per pair (the same `addrs` may be reused).
    """
    for buf_name, buf in info.sync.buffers.items():
        if buf.fwd_ids_node is None and buf.bwd_ids_node is None:
            continue  # not a cross-core buffer
        entries = users.get(buf_name, [])
        names = [e[1] for e in entries]
        if not entries:
            continue  # used by another pipeline loop; validate_kernel_buffers catches unused
        if len(entries) == 1:
            raise InvalidOperation(
                f"pipeline: cross-core buffer '{buf_name}' is used by only one stage "
                f"{names} in this pipeline loop, so the handover has no other side here. "
                f"Each pipeline loop must hold a complete producer/consumer pair — its sync "
                f"balances within the loop, so a handover reaching into another loop is not "
                f"supported."
            )
        if len(entries) > 2:
            raise InvalidOperation(
                f"pipeline: cross-core buffer '{buf_name}' is used by {len(entries)} stages "
                f"{names}. One buffer variable carries one producer/consumer pair, since its "
                f"event ids are resolved per buffer. Declare one tile_group per pair — they "
                f"may share the same addrs and mutex_ids."
            )
        sections = {e[2] for e in entries}
        if len(sections) == 1:
            raise InvalidOperation(
                f"pipeline: cross-core buffer '{buf_name}' is used only by "
                f"'{sections.pop()}' stages {names}. Cross-core sync orders work across the "
                f"cube/vector boundary; same-core ordering comes from auto_mutex instead."
            )
        producer, consumer = entries[0], entries[1]
        if consumer[0] != producer[0] + 1:
            between = consumer[0] - producer[0] - 1
            raise InvalidOperation(
                f"pipeline: cross-core buffer '{buf_name}' is handed from stage "
                f"'{producer[1]}' to stage '{consumer[1]}', with {between} stage(s) in "
                f"between. A cross-core buffer must be handed to the stage that follows "
                f"its producer directly. Hand the data over through the stages in between, "
                f"or move the two stages next to each other."
            )


def _check_event_id_counts(graph, info) -> None:
    """A direction's event-id list must hold one id per slot, or exactly one id.

    Two counters run side by side: slots turn over on ``task % slot_count``, while the id for
    a handover is picked with ``ids[task % id_count]``. Only two ratios keep them in step:

      id_count == slot_count   one id per slot; every slot hands over independently.
      id_count == 1            all slots share one id. Consecutive handovers queue up on it,
                               costing parallelism but not correctness — the intended way to
                               cope with a shortage of ids.

    Anything between (three slots on two ids, say) makes the two cycles slide against each
    other, so a wait pairs with the set of a different slot — an intermittent data race
    rather than a failure.

    Buffers no stage touches are skipped — _check_cross_core_users reports those instead.
    """
    for buf_name, buf in info.sync.buffers.items():
        region = graph.regions.get(buf_name)
        if region is None:
            continue
        slots = graph.slots[region]
        for label, id_count in (("fwd_ids", buf.fwd_id_count), ("bwd_ids", buf.bwd_id_count)):
            if id_count and id_count not in (1, slots):
                raise InvalidArgument(
                    f"pipeline: cross-core buffer '{buf_name}' declares {id_count} {label} "
                    f"but rotates through {slots} slots. Give it one id per slot "
                    f"({slots} of them), or a single id shared by all slots — any other "
                    f"count pairs a wait with the set of a different slot."
                )


def _check_reuse_mutex_ids(graph, info) -> None:
    """Buffers sharing a region must share their mutex ids.

    A mutex locks an ADDRESS, so two buffers over the same memory holding different locks do
    not exclude each other at all and the intra-core ordering auto_mutex provides silently
    disappears.
    """
    by_region: dict = {}
    for buf_name, region in graph.regions.items():
        by_region.setdefault(region, []).append(buf_name)

    for region, members in sorted(by_region.items()):
        if len(members) < 2:
            continue
        ids_by_buf = {name: info.sync.mutex_ids.get(name) for name in sorted(members)}
        distinct = {tuple(v) for v in ids_by_buf.values() if v is not None}
        if len(distinct) > 1:
            detail = ", ".join(f"{n}={v}" for n, v in ids_by_buf.items())
            raise InvalidOperation(
                f"pipeline: buffers sharing addresses must share mutex_ids, but region "
                f"{region} has {detail}. A mutex locks the address, so different ids over "
                f"one region give no mutual exclusion."
            )


def validate_names(func_def, framework_names: set) -> None:
    """The variables this transform is about to introduce must not already be in use.

    Called by BOTH emission paths, each passing its own names: what a path may collide over
    is exactly what that path emits.

    Only fixed names are worth listing. Lifted id variables (``_pl_fwd_ids_<buffer>``) end in
    a user-chosen buffer name, and the parser catches a genuine redefinition anyway.
    """
    _check_name_collisions(func_def, framework_names)


def _check_name_collisions(func_def, framework_names: set) -> None:
    """The names the transform introduces must not already be taken by the kernel."""
    user_names = {n.id for n in ast.walk(func_def) if isinstance(n, ast.Name)}
    clash = sorted(framework_names & user_names)
    if clash:
        raise InvalidOperation(
            f"pipeline transform: framework variable name(s) {clash} collide with "
            f"user-defined names in the kernel. The '_pl_' prefix is reserved for "
            f"the pipeline transform — please rename the conflicting user variable(s).",
            span=span_of(func_def),
        )


def validate_kernel_structure(infos: list, preload) -> None:
    """Structural checks about the KERNEL as a whole, not about one pipeline.

    Handed every pipeline the kernel holds, so it can check how they sit together; run from
    inside the per-loop gate it would report the same thing once per loop.

    ``preload`` is the config value, taken as a value rather than as the config object so
    this module stays a leaf that knows about ASTs and nothing else.
    """
    _check_preload_count(preload, len(infos))
    _check_distinct_outer_loops(infos)


def _check_preload_count(preload, loop_count: int) -> None:
    """One preload per pipeline loop, or one for all of them.

    A mismatched list is refused rather than padded or truncated: which loop each value was
    meant for is exactly what a wrong length makes unknowable.
    """
    if not isinstance(preload, tuple) or len(preload) == loop_count:
        return
    raise InvalidArgument(
        f"pipeline: preload has {len(preload)} values {list(preload)} but the kernel holds "
        f"{loop_count} pipeline loop(s). Give one value per loop, in source order, or a "
        f"single value to use for all of them."
    )


def _check_distinct_outer_loops(infos: list) -> None:
    """Two pipelines may not hang off the same enclosing loop.

    A pipeline's declarations and pre-fire go before its outermost enclosing loop and its
    drain after it, so one task stream runs continuously across that loop. Two pipelines
    sharing it would land both sets in the same two places, putting the first one's drain
    after every iteration of the second.

    Nesting a pipeline loop inside another kernel loop is the normal shape and unaffected.
    """
    seen: dict = {}
    for info in infos:
        outer = info.outer_loop
        if outer is None:
            continue
        first = seen.setdefault(id(outer), info)
        if first is not info:
            raise InvalidOperation(
                f"pipeline: the pipeline loops at line {first.pipeline_loop.lineno} and line "
                f"{info.pipeline_loop.lineno} are both inside the loop at line "
                f"{outer.lineno}, but each pipeline places its declarations before that loop "
                f"and its drain after it, so they cannot share one.\n"
                f"Move one of them out, or put the stages of both into a single loop."
            )


def validate_structure(info, func_def=None, stage_func_names: set | None = None) -> None:
    """Checks that need only the parsed structure — no sync or schedule information.

    Called from analyze_pipeline once the stages, sections and loop info exist and before
    anything is derived from them, so a malformed stage chain is reported as such rather than
    as whatever a later pass trips over.

    ``func_def`` and ``stage_func_names`` enable the one check that looks at the whole kernel
    rather than at the stage list.
    """
    _check_stage_returns(info)
    _check_stage_sections(info)
    _check_unique_stage_names(info)
    _check_alternating_sections(info)
    _check_nothing_between_stages(info)
    _check_loop_step(info)
    if func_def is not None and stage_func_names:
        _check_no_nested_stage(func_def, info, stage_func_names)


def _check_no_nested_stage(func_def, info, stage_func_names: set) -> None:
    """A stage must not reach another stage, directly or through the functions it calls.

    Sub-stages need their accesses attributed to the sub-stage and sync placed around the
    inner call; support for that was removed pending a redesign. Without it the inner stage
    reads as an ordinary helper call and its handovers go unsynchronised.

    The search follows plain helper calls too: the access scan walks helpers but skips stage
    names inside them, and the analyzer only lists the stages called in the pipeline loop, so
    a stage buried behind a helper contributes nothing at all — no accesses, no sync, no beat.
    """
    stage_defs = {
        node.name: node
        for node in ast.walk(func_def)
        if isinstance(node, ast.FunctionDef) and node.name in stage_func_names
    }
    # The stages the pipeline loop calls already carry their AST; take those for free.
    for stage in info.stages:
        if stage.func_def is not None:
            stage_defs.setdefault(stage.func_name, stage.func_def)
    # A stage decorated but never called by the loop is not among them, and a stage's body
    # usually lives outside the kernel, so the rest are resolved through closure_vars.
    for name in stage_func_names:
        if name in stage_defs:
            continue
        resolved = _try_get_funcdef_for(info, name)
        if resolved is not None:
            stage_defs[name] = resolved

    def visit(caller_def, path: tuple) -> None:
        for node in ast.walk(caller_def):
            if not isinstance(node, ast.Call):
                continue
            callee = call_name(node)
            if not callee or callee in path:
                continue
            if callee in stage_func_names:
                chain = " -> ".join(path + (callee,))
                where = f" via {' -> '.join(path[1:])}," if len(path) > 1 else ""
                raise InvalidOperation(
                    f"pipeline: stage '{path[0]}' reaches stage '{callee}'{where} at line "
                    f"{node.lineno} ({chain}). A stage may not contain another stage — the "
                    f"inner one's buffer accesses would be attributed to the caller and its "
                    f"handovers left unsynchronised.\n"
                    f"Inline '{callee}' into '{path[0]}', or make it a plain helper function "
                    f"(drop @pl.pipeline.stage) if it needs no sync of its own.",
                    span=span_of(node),
                )
            helper_def = _try_get_funcdef_for(info, callee)
            if helper_def is not None:
                visit(helper_def, path + (callee,))

    for outer_name, outer_def in stage_defs.items():
        visit(outer_def, (outer_name,))


def _try_get_funcdef_for(info, name: str):
    """The AST of a stage defined outside the kernel, via the analyzer's closure_vars."""
    return get_funcdef(info.closure_vars.get(name))


def _check_stage_returns(info) -> None:
    """A stage must not return a value: the transform calls it for its side effects and
    has nowhere to put a result."""
    for stage in info.stages:
        func_def = stage.func_def
        if func_def is None:
            continue
        for node in ast.walk(func_def):
            if isinstance(node, ast.Return) and node.value is not None:
                raise InvalidOperation(
                    f"pipeline: stage '{stage.func_name}' returns a value. Stages are "
                    f"called for their effect on buffers; use an output buffer instead."
                )


def _check_stage_sections(info) -> None:
    """Every stage call must sit inside a section block, since that says which core runs it."""
    for stage in info.stages:
        if stage.section_kind == "":
            raise InvalidOperation(
                f"pipeline: stage call '{stage.func_name}' appears directly in the pipeline "
                f"loop body, not inside a `with pl.section_cube()/section_vector()` block. "
                f"Each stage call must be wrapped in a section block."
            )


def _check_unique_stage_names(info) -> None:
    """A stage function may be called only once per pipeline loop.

    Sync sites are looked up by stage name (_sync_graph records it on every access), so two
    stages sharing a name would each emit the other's waits and sets.
    """
    seen = set()
    for stage in info.stages:
        if stage.func_name in seen:
            raise InvalidOperation(
                f"pipeline: stage '{stage.func_name}' is called more than once in this "
                f"pipeline loop. A stage is identified by its function name — its sync "
                f"belongs to that name — so each stage of the chain needs its own function.\n"
                f"Give the repeated step a second stage function, even if it only calls a "
                f"shared helper."
            )
        seen.add(stage.func_name)


def _check_nothing_between_stages(info) -> None:
    """Between the first and the last stage the loop body holds stages and nothing else.

    A stage runs once per TASK; every other statement in the loop body runs once per BEAT.
    Written between two stages, a statement reads as belonging to one of them and does not,
    so the loop body is split in three: statements before the first stage, the stage chain,
    statements after the last one.

    Runs on the pruned tree (transform_pipeline folds compile-time-constant branches first),
    so an `if` that selects between stages is already gone by the time this looks.
    """
    if info.pipeline_loop is None or not info.stages:
        return
    holders = {id(s.source_stmt) for s in info.stages if s.source_stmt}
    positions = [i for i, stmt in enumerate(info.pipeline_loop.body) if id(stmt) in holders]
    if not positions:
        return
    for stmt in info.pipeline_loop.body[positions[0]:positions[-1]]:
        if id(stmt) in holders:
            continue
        raise InvalidOperation(
            f"pipeline: the statement at line {stmt.lineno} "
            f"(`{ast.unparse(stmt).splitlines()[0]}`) sits between two stage calls. Only "
            f"stage calls belong there — a stage runs once per task, every other statement "
            f"runs once per beat, and between two stages there is no saying which it "
            f"follows.\n"
            f"Move it to the start or the end of the pipeline loop body, or into a stage.",
            span=span_of(stmt),
        )


def _check_alternating_sections(info) -> None:
    """The stage chain has to alternate cube/vector.

    Two consecutive stages on one core would advance that core's delay by the preload
    amount twice in a row, which the delay model has no way to express.
    """
    stages = info.stages
    for i in range(len(stages) - 1):
        cur, nxt = stages[i], stages[i + 1]
        if cur.section_kind == nxt.section_kind:
            raise InvalidOperation(
                f"pipeline: stages '{cur.func_name}' and '{nxt.func_name}' are both "
                f"on the '{cur.section_kind}' core (consecutive same-core stages). The "
                f"delay model requires the stage chain to strictly alternate "
                f"cube/vector (C->V->C->V...)."
            )


def validate_slot_picks(info, tracked: set, helper_defs: dict) -> None:
    """A tracked buffer's slot may only be taken inside a stage, not in the loop body.

    A slot belongs to a TASK, and the sync graph reads a buffer's slot as
    ``task % slot_count``. A stage takes one once per task; a statement in the loop body
    takes one once per ITERATION, and a delayed stage's task is not the current iteration —
    it also takes one during the drain, where the loop body no longer runs. Sharing a
    group's cursor between the two hands the stage a slot that is not its task's, and with
    it the wrong event id.

    ``tracked`` is what the sync graph follows: cross-core buffers and the ones sharing
    their address (AccessScanTables.tracks). A purely local group's rotation is the user's
    own — nothing here indexes it, and auto_mutex orders the accesses whatever the cursor
    does.

    Every spelling counts, wherever it sits: a bare `g.next()`, one buried in a larger
    expression, and one inside a helper the loop body calls. A stage's own body is not
    searched — helper_defs holds no stage.

    Outside the pipeline loop is a different mechanism and unaffected: such a slot's index is
    carried through ctx, so every stage re-selects the slot of the task it is handling.
    """
    if info.pipeline_loop is None or not tracked:
        return
    _reject_slot_picks(info.pipeline_loop, tracked, helper_defs, frozenset(), "")


def _reject_slot_picks(tree, groups: set, helper_defs: dict, seen: frozenset, where: str) -> None:
    """Raise on any slot pick in ``tree``, following calls into the helpers it makes.

    ``groups`` are the names that hold a tracked group in this tree's own scope: the buffer
    names at the top, and inside a helper the parameters one was passed to.
    """
    for node in ast.walk(tree):
        pick = slot_accessor(node)
        if pick is not None and pick[0] in groups:
            group, kind = pick
            accessor = "[...]" if kind == "index" else f".{kind}()"
            raise InvalidOperation(
                f"pipeline: line {node.lineno}{where} takes a slot from tile group '{group}' "
                f"(`{group}{accessor}`) inside the pipeline loop but outside any stage. A slot "
                f"belongs to one task, and in the loop body only a stage runs once per task — "
                f"the rest runs once per beat, including the fill and drain beats where there "
                f"is no task at all.\n"
                f"Take the slot inside the stage function, or outside the pipeline loop, where "
                f"its index is carried through ctx.",
                span=span_of(node),
            )
        if not isinstance(node, ast.Call):
            continue
        callee = helper_defs.get(call_name(node))
        if callee is None or callee.name in seen:
            continue
        params = [a.arg for a in callee.args.args]
        passed = {
            params[pos]
            for pos, arg in enumerate(node.args)
            if pos < len(params) and isinstance(arg, ast.Name) and arg.id in groups
        }
        if passed:
            _reject_slot_picks(
                callee, passed, helper_defs, seen | {callee.name},
                f" of `{callee.name}` (called from line {node.lineno})",
            )


def _check_loop_step(info) -> None:
    """The pipeline loop must count upwards.

    Guards that reason about a task some iterations away compute `var + n * step` and compare
    with `<` against the bound; a negative step would need the opposite comparison throughout.
    """
    step = info.pipeline_loop_step
    if step is None:
        return  # implicit 1
    if isinstance(step, ast.UnaryOp) and isinstance(step.op, ast.USub):
        raise NotSupported(
            f"pipeline: pipeline loop `for {info.pipeline_loop_var} in ...` steps backwards "
            f"({ast.unparse(step)}). Only forward iteration is supported."
        )
    if isinstance(step, ast.Constant) and isinstance(step.value, int) and step.value <= 0:
        raise InvalidType(
            f"pipeline: pipeline loop `for {info.pipeline_loop_var} in ...` has step "
            f"{step.value}. The step must be a positive value."
        )
