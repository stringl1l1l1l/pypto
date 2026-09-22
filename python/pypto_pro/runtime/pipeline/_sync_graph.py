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

"""Cross-core sync dependency graph: the single source for all sync insertion.

One model covers preload=0, preload=N and address reuse.

Model
-----
node    an op-level access ``Access(stage, buffer, role, pipe)``
region  a physical memory area — union-find over ``addr_overlaps``, so co-located buffers
        land in one region and a non-reused buffer is a region of its own
lane    one (region, physical slot): everything competing for that exact memory, in time
        order. Edges only ever form inside a lane, which is why address reuse needs no
        special case.
edge    RAW  a read waits for the nearest preceding write of the same buffer
        WAR  a write waits for the readers of the nearest preceding write per buffer on
             that slot — same buffer gives the ordinary backward edge, a co-located
             neighbour gives the reuse edge

Distances come from expanding the slot timeline, not from arithmetic on slot counts, and
both quantities matter: ``dist`` (beat difference) decides inverse-time and cycles, while
``task_off`` (task difference) is the guard constant and the event-id pairing.

Assumptions
-----------
A1. Each stage advances a cross-core buffer exactly once, so ``slot = task % slot_count``.
    A violation shows up as an edge whose distance differs from task to task, which
    ``Edge.unstable`` records and _validate._check_stable_distances rejects.
A2. A buffer variable carries one producer/consumer pair; a stage may touch it repeatedly.
A3. Sync goes outside the stage call. Op-level detail gets the dependencies right, then
    the result is aggregated back to the stage boundary when emitting.

Control flow inside a stage is ignored on purpose: an access in a branch counts like an
unconditional one, so a buffer touched on several pipes gets a sync on every one of them.
That over-syncs when a path skips some, but the error only ever goes in the safe
direction — narrowing it down needs reasoning about branch conditions, and a wrong answer
there drops a sync instead.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import logging

from ..._errors import InvalidOperation, InvalidVal, OutOfRange
from ._cross_core_scanner import MAX_EVENT_ID
from ._validate import validate_sync

# ---------------------------------------------------------------------------
# Graph data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Access:
    """One op-level access to a buffer (steady-state template, no task attached).

    ``seq`` is the position in the kernel's overall program order. Within a beat, the
    relative order of two accesses decides whether a dependency is same-beat or a whole
    rotation away, so it cannot be left to list insertion order.
    """

    seq: int
    stage_idx: int
    stage: str
    section: str  # "cube" / "vector"
    buffer: str
    role: str  # "W" / "R"
    pipe: str
    region: int = -1

    def __str__(self) -> str:
        return f"{self.stage}({self.buffer} {self.role} {self.pipe})"


@dataclass(frozen=True)
class _Event:
    """A point on the expanded timeline: an Access instantiated for one task."""

    beat: int  # loop iteration = task + schedule[stage_idx]
    order: int  # position within the beat = Access.seq (program order)
    task: int
    slot: int  # physical slot
    acc: Access


@dataclass
class Edge:
    """A sync dependency. ``src`` is the producing side (emits set), ``dst`` the
    consuming side (emits wait)."""

    kind: str  # "RAW" / "WAR"
    src: Access
    dst: Access
    slot_count: int
    dist: int  # consumer beat - producer beat; < 0 means inverse-time
    task_off: int  # consumer task - producer task; the guard constant
    # Distance differed between tasks, i.e. the timeline never settled. Rejected by
    # validate_sync — see _check_stable_distances for why it cannot be emitted.
    unstable: bool = False

    @property
    def inverse_time(self) -> bool:
        return self.dist < 0


@dataclass
class SyncGraph:
    """Build result: the access list, region map, per-region slot count and the edges."""

    accesses: list
    regions: dict  # buffer name -> region id
    slots: dict  # region id -> slot count
    edges: list
    # {id(edge): buffer name} for WAR edges that release ANOTHER buffer's backward ids rather
    # than a group of their own — see _share_intermediate_war_ids.
    war_id_source: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------


def _slot_count(info, buffer: str) -> int:
    """How many slots a buffer rotates through, i.e. the modulus of ``task % slot_count``.

    Read from the buffer's mutex_ids — one lock per slot — rather than from its address
    ranges: both give the same number, but scan_buffer_addr_ranges is allowed to skip a
    buffer it cannot size, while every buffer in the graph needs a slot count.
    """
    ids = info.sync.mutex_ids.get(buffer)
    if ids:
        return len(ids)
    ranges = info.sync.addr_ranges.get(buffer)
    if ranges:
        return len(ranges[1])
    raise InvalidOperation(
        f"pipeline: cannot determine the slot count of buffer '{buffer}'. Its "
        f"make_tile_group needs a statically resolvable mutex_ids list."
    )


def build_regions(addr_overlaps, touched: set) -> dict:
    """Union-find over address-overlapping pairs: co-located buffers share a region.

    Only buffers actually accessed by a stage take part. Unlike
    ``detect_addr_overlaps``'s pairwise restriction this handles N-way sharing, since
    a connected component may hold any number of buffers.
    """
    parent = {buf: buf for buf in touched}

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    for buf_a, buf_b in addr_overlaps:
        if buf_a in parent and buf_b in parent:
            root_a, root_b = find(buf_a), find(buf_b)
            if root_a != root_b:
                parent[root_a] = root_b

    roots = sorted({find(buf) for buf in parent})
    return {buf: roots.index(find(buf)) for buf in parent}


# ---------------------------------------------------------------------------
# Accesses
# ---------------------------------------------------------------------------


def collect_accesses(info) -> list:
    """Flatten every stage's buffer accesses into op-level Access nodes."""
    out = []
    seq = 0
    for idx, stage in enumerate(info.stages):
        # One node per distinct (buffer, role, pipe): one op can be recorded twice (e.g. in
        # two mutually exclusive if-branches), and duplicates would duplicate every edge
        # and set/wait pair through them.
        seen = set()
        for buf_name, role, pipe in stage.buffer_access:
            if (buf_name, role, pipe) in seen:
                continue
            seen.add((buf_name, role, pipe))
            out.append(Access(seq, idx, stage.func_name, stage.section_kind, buf_name, role, pipe))
            seq += 1
    return out


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph(info, schedule: list) -> SyncGraph:
    """Build the sync dependency graph for one schedule.

    ``schedule[stage_idx]`` is that stage's beat offset (its delay):
      - preload=0  : all zeros (serial, every stage in the same beat)
      - preload=N  : ``_compute_delays(info, N)``

    The builder does not care where the schedule came from, which is what makes
    the serial schedule a degenerate case of the same model rather than a second code path.
    """
    accesses = collect_accesses(info)
    touched = {a.buffer for a in accesses}
    regions = build_regions(info.sync.addr_overlaps, touched)
    accesses = [
        Access(a.seq, a.stage_idx, a.stage, a.section, a.buffer, a.role, a.pipe, regions[a.buffer])
        for a in accesses
    ]

    slots = {region: _slot_count(info, buf) for buf, region in regions.items()}

    # Expand enough tasks for a steady middle, and only read edges off that part.
    # Dropping warm/cool admits fill/drain artefacts that do not even look unstable.
    max_delay = max(schedule) if schedule else 0
    max_slots = max(slots.values()) if slots else 1
    # warm: two effects stack at the head — the pipeline needs max_delay beats to fill,
    # and a lane needs a full rotation before a "previous write" exists at all.
    warm = max_delay + max_slots
    # cool: at the tail a write's readers fall outside the range, so its WAR edge would
    # be missed. Those tasks are still expanded as filler, just never read as a consumer
    # — hence cool is added into ntask here and subtracted from the window bound below.
    cool = max_slots
    # One rotation already pins every edge down; the second gives the unstable check a
    # second sample to compare against.
    ntask = warm + 2 * max_slots + cool + 2

    events = []
    for task in range(ntask):
        for acc in accesses:
            events.append(
                _Event(task + schedule[acc.stage_idx], acc.seq, task, task % slots[acc.region], acc)
            )

    # One lane per (region, physical slot): everything competing for that exact memory,
    # ordered by beat then by code order within the beat.
    lanes: dict = {}
    for event in events:
        lanes.setdefault((event.acc.region, event.slot), []).append(event)
    for key in lanes:
        lanes[key].sort(key=lambda e: (e.beat, e.order))

    found: dict = {}  # (kind, src_acc, dst_acc) -> (dist, task_off, unstable)

    def record(kind: str, src: _Event, dst: _Event) -> None:
        if not warm <= dst.task < ntask - cool:
            return  # outside the steady-state window: fill/drain skews the distances
        key = (kind, src.acc, dst.acc)
        value = (dst.beat - src.beat, dst.task - src.task)
        previous = found.get(key)
        if previous is None:
            found[key] = (*value, False)
        elif previous[:2] != value:
            found[key] = (*value, True)  # same edge, different distance => not steady

    for lane in lanes.values():
        for i, event in enumerate(lane):
            # Membership, not equality: an "RW" access both reads and writes, so it takes part on
            # both sides.
            if "R" in event.acc.role:
                # RAW: the nearest preceding write of the SAME buffer produced this data.
                for j in range(i - 1, -1, -1):
                    candidate = lane[j]
                    if "W" in candidate.acc.role and candidate.acc.buffer == event.acc.buffer:
                        record("RAW", candidate, event)
                        break
                continue

            # WAR: this write must wait for every batch still living on the slot. Walk
            # back per buffer, not just to the single nearest write — N co-located buffers
            # mean up to N batches, each with its own readers, hence up to N edges.
            nearest_write_per_buffer = {}
            for j in range(i - 1, -1, -1):
                candidate = lane[j]
                if "W" in candidate.acc.role and candidate.acc.buffer not in nearest_write_per_buffer:
                    nearest_write_per_buffer[candidate.acc.buffer] = candidate
            for previous_write in nearest_write_per_buffer.values():
                readers = [
                    e
                    for e in lane
                    if "R" in e.acc.role
                    and e.acc.buffer == previous_write.acc.buffer
                    and e.task == previous_write.task
                ]
                for src in _gating_readers(readers):
                    record("WAR", src, event)

    edges = [
        Edge(kind, src, dst, slots[src.region], dist, task_off, unstable)
        for (kind, src, dst), (dist, task_off, unstable) in found.items()
    ]
    graph = SyncGraph(accesses, regions, slots, edges, _share_intermediate_war_ids(edges, info))
    # One gate for every check that needs the graph (see _validate), run here so the checks
    # see exactly what the emission will. The cycle walk is graph work and is done here; the
    # verdict on what a cycle's total distance means is a check.
    validate_sync(graph, info, find_cycles(graph), schedule)
    return graph


# ---------------------------------------------------------------------------
# Cycle check
# ---------------------------------------------------------------------------


def find_cycles(graph: SyncGraph) -> list:
    """Find dependency cycles. Returns ``[(access_path, total_distance), ...]``.

    A total of <= 0 is a deadlock: the cycle waits on the present or future. A positive
    total reaches into the past, which is a healthy cross-iteration pipeline; a single
    negative edge is fine, only the total matters.

    Walks ACCESS nodes, not stages, so several edges between one stage pair stay distinct.
    """
    adjacency: dict = {}
    for edge in graph.edges:
        adjacency.setdefault(edge.src, []).append((edge.dst, edge.dist))

    best: dict = {}  # frozenset of accesses -> (path, smallest total)
    limit = len(graph.accesses) + 1

    def walk(start, current, path, total, visited):
        for nxt, dist in adjacency.get(current, []):
            if nxt == start:
                key = frozenset(path)
                found = best.get(key)
                if found is None or total + dist < found[1]:
                    best[key] = (path + [start], total + dist)
            elif nxt not in visited and len(path) < limit:
                walk(start, nxt, path + [nxt], total + dist, visited | {nxt})

    for access in graph.accesses:
        walk(access, access, [access], 0, {access})
    return list(best.values())


def _gating_readers(readers: list) -> list:
    """Which readers of a batch must be waited for before its slot may be overwritten.

    The latest read on every pipe, one edge each; same-pipe reads need only their latest,
    since their queue already runs them in order.

    The single latest read overall would do only if the earlier reads were guaranteed to
    precede it — true within a straight-line stage, but not when the latest read sits in a
    branch that a path skips. Telling the two apart needs condition reasoning, and being
    wrong drops a sync.
    """
    per_pipe: dict = {}
    for reader in readers:
        seen = per_pipe.get(reader.acc.pipe)
        if seen is None or (reader.beat, reader.order) > (seen.beat, seen.order):
            per_pipe[reader.acc.pipe] = reader
    return list(per_pipe.values())


def _share_intermediate_war_ids(edges: list, info) -> dict:
    """Point WAR edges at the backward ids of the buffer whose data they really guard.

    A buffer with no cross-core ids of its own is not a channel — it holds the same batch of
    data in another form (an fp16 copy, say). So within a group of WAR edges sharing a
    destination and task offset, when exactly one source buffer owns ids, every edge in the
    group releases *that* buffer's backward ids instead of a freshly allocated group: N edges
    become N set/wait pairs on the same id and stay balanced. Several owners mean genuinely
    separate channels, left alone.

    Every edge survives — each guards one pipe (see _gating_readers).

    Returns {id(edge): owning buffer name}, for SyncGraph.war_id_source.
    """
    groups: dict = {}
    for edge in edges:
        if edge.kind != "WAR":
            continue
        groups.setdefault((edge.dst, edge.task_off), []).append(edge)

    id_source = {}
    for group in groups.values():
        if len(group) < 2:
            continue
        owners = [e for e in group if _owns_cross_core_ids(e.src.buffer, info)]
        if len(owners) != 1:
            continue
        for edge in group:
            id_source[id(edge)] = owners[0].src.buffer

    return id_source


def _owns_cross_core_ids(buffer_name: str, info) -> bool:
    """True if the buffer declares cross-core ids of its own."""
    buf = info.sync.buffers.get(buffer_name)
    return buf is not None and (buf.fwd_ids_node is not None or buf.bwd_ids_node is not None)


def _is_redundant(edge: Edge) -> bool:
    """True if the stage chain already orders this edge's two ends, making it superfluous.

    Only same-task edges qualify: either both ends on one core (program order plus
    auto_mutex suffice) or the producing stage running first. The mirror case — same task,
    consuming stage first — is a conflict, reported by plan_sync_sites.
    """
    if edge.task_off != 0:
        return False
    if edge.src.section == edge.dst.section:
        return True
    return edge.src.stage_idx < edge.dst.stage_idx


def _needs_allocated_ids(edge: Edge, war_id_source: dict) -> bool:
    """True if this edge needs framework-allocated ids (address reuse, not redundant).

    Edges that borrow another buffer's ids are excluded — that is the whole point of
    borrowing them (see _share_intermediate_war_ids).
    """
    if edge.kind != "WAR" or edge.src.buffer == edge.dst.buffer or _is_redundant(edge):
        return False
    return id(edge) not in war_id_source


def _used_event_ids(graph: SyncGraph, info) -> set:
    """Every event id already spoken for by a buffer THIS pipeline loop touches.

    Scoped to the loop, not the kernel: two pipeline loops run one after another and each
    balances its own sync, so an id another loop declares is free here. With sixteen ids in
    the hardware, a kernel-wide view would run the pool down for no reason.

    Read from the buffer declarations, not from the lifted literals: only the literal
    spelling is lifted, so ``fwd_ids=IDS`` would be invisible to the allocator.
    """
    used = set()
    for buf_name in {access.buffer for access in graph.accesses}:
        decl = info.sync.buffers.get(buf_name)
        if decl is not None:
            used.update(decl.fwd_ids)
            used.update(decl.bwd_ids)
    return used


def _allocate_event_id_groups(edges: list, used: set) -> list:
    """One group of event ids per edge, taken from the ids nothing else claims.

    Each edge wants ``slot_count`` ids so its slots can hand over independently. When the
    pool cannot cover that, edges are degraded to a single shared id — the handover then
    serialises across slots, which costs parallelism but stays correct. Degradation goes in
    ascending slot_count order, so the edges that lose the least go first.

    Returns the groups positionally, one per input edge; nothing is mutated.
    """
    pool_size = MAX_EVENT_ID + 1
    free = [i for i in range(pool_size) if i not in used]

    wants = [edge.slot_count for edge in edges]
    order = sorted(range(len(edges)), key=lambda i: wants[i])
    degraded = 0
    while sum(wants) > len(free) and degraded < len(order):
        idx = order[degraded]
        if wants[idx] > 1:
            logging.warning(
                f"pipeline: not enough event ids for address-reuse sync; degrading "
                f"{edges[idx].src} -> {edges[idx].dst} from {wants[idx]} ids to 1 "
                f"(slots will serialize, correctness preserved)."
            )
            wants[idx] = 1
        degraded += 1

    if sum(wants) > len(free):
        raise OutOfRange(
            f"pipeline: not enough free event ids (0-{MAX_EVENT_ID}) for address-reuse "
            f"sync. Need {sum(wants)}, have {len(free)} free (used: {sorted(used)}). "
            f"Reduce cross-core buffer id usage or overlaps."
        )

    groups = []
    cursor = 0
    for count in wants:
        groups.append(free[cursor:cursor + count])
        cursor += count
    return groups


def allocate_reuse_ids(graph: SyncGraph, info) -> dict:
    """Mint an event-id group per address-reuse edge. Returns {id(edge): ids_node}.

    Address reuse creates a dependency the user never declared ids for, so the framework
    allocates one and declares it as a variable ahead of the loop. Which edges need this is
    derived in the same place the allocation happens, so an edge cannot be emitted without
    being allocated.

    Several edges can share one group — two edges between the same pair of accesses describe
    one handover — hence the keying by access pair.
    """
    needing = [edge for edge in graph.edges if _needs_allocated_ids(edge, graph.war_id_source)]
    if not needing:
        return {}

    keys = [(edge.src.seq, edge.dst.seq) for edge in needing]
    # One group per distinct access pair, in first-seen order.
    first_of_pair: dict = {}
    for key, edge in zip(keys, needing):
        first_of_pair.setdefault(key, edge)

    # Read the used ids BEFORE declaring the new groups below, or each group would count
    # itself as taken.
    id_groups = _allocate_event_id_groups(list(first_of_pair.values()), _used_event_ids(graph, info))

    group_name: dict = {}
    for key, ids in zip(first_of_pair, id_groups):
        # Numbered across the whole kernel, not per call: a per-call counter would give the second
        # loop variable names the first already declared. The ids INSIDE the groups are per loop
        # and may repeat.
        group_name[key] = f"_pl_overlap_ids_{len(info.sync.reuse_ids)}"
        literal = ast.List(elts=[ast.Constant(value=v) for v in ids], ctx=ast.Load())
        info.sync.reuse_ids.append((group_name[key], literal))

    return {id(edge): ast.Name(id=group_name[key], ctx=ast.Load()) for key, edge in zip(keys, needing)}


def resolve_event_ids(graph: SyncGraph, info) -> dict:
    """Decide which event-id group each edge uses, keyed by edge.

    Value is ``(ids_node, slot_count)``, or ``None`` when the edge must not be emitted.

      RAW                        -> the buffer's fwd_ids
      WAR, same buffer           -> the buffer's bwd_ids
      WAR, across buffers        -> a framework-allocated overlap group (address reuse
                                    creates a dependency the user never declared ids for)

    A user who declares fwd_ids but no bwd_ids is stating that this buffer has enough
    slots that overwrite protection is unnecessary; that edge then resolves to None and
    no instruction is emitted. The edge still exists in the graph — cycle and
    inverse-time analysis must keep seeing the real dependency.
    """
    buffers = info.sync.buffers
    reuse_ids = allocate_reuse_ids(graph, info)

    resolved = {}
    for edge in graph.edges:
        if edge.src.section == edge.dst.section:
            # Both ends on one core. A cross-core buffer's two users must sit on different cores
            # (_check_cross_core_users), so an edge inside one section can only join two accesses of
            # the same stage, already ordered by program order and auto_mutex. Emitting for it would
            # put an extra set/wait pair on the buffer's own event id, which can release a wait meant
            # for the other core's set. Resolved here so all three planners agree on what is live.
            resolved[id(edge)] = None
            continue
        if edge.kind == "RAW":
            buf = buffers.get(edge.src.buffer)
            node = buf.fwd_ids_node if buf else None
            count = buf.fwd_id_count if buf else 0
        elif (owner := graph.war_id_source.get(id(edge))) is not None:
            # An edge over reused memory whose source is an intermediate form of another
            # buffer's data (see _share_intermediate_war_ids): it releases that buffer's
            # backward ids — the existing sync, not a newly allocated group.
            buf = buffers.get(owner)
            node = buf.bwd_ids_node if buf else None
            count = buf.bwd_id_count if buf else 0
        elif edge.src.buffer == edge.dst.buffer:
            buf = buffers.get(edge.src.buffer)
            node = buf.bwd_ids_node if buf else None
            count = buf.bwd_id_count if buf else 0
        else:
            # Cross-buffer WAR: no user-declared ids exist for a dependency that only
            # address reuse created, so the framework allocates one (None if the edge
            # turned out redundant — see _is_redundant).
            node = reuse_ids.get(id(edge))
            count = edge.slot_count
        resolved[id(edge)] = (node, count) if node is not None and count > 0 else None
    return resolved


@dataclass
class SyncSite:
    """One sync instruction to emit at a stage boundary.

    ``stage`` says where, ``side`` says which end ("pre" = wait before the
    call, "post" = set after it). A3: this is always outside the stage call, never inside
    its body.
    """

    stage: str
    side: str  # "pre" (wait) / "post" (set)
    op: str  # "wait_cross_core" / "set_cross_core"
    pipe: str
    section: str
    ids_node: object  # AST node for the event-id group
    slot_count: int
    # Drain sites only: the literal event-id index to use. In-loop sites index by
    # task id at runtime instead, so this stays None for them.
    id_index: int = None


def _skew_of(edge: Edge) -> int:
    """Signed skew: an edge pairs ``dst(task t)`` with ``src(task t - off)``.

    0 for every RAW edge — its ends line up task-for-task, so nothing is stranded at either
    end. A WAR edge's skew is the task distance its slots impose.
    """
    return edge.task_off if edge.kind == "WAR" else 0


def plan_sync_sites(graph: SyncGraph, event_ids: dict) -> list:
    """The in-loop sync: a wait before each consuming access, a set after each producing one.

    Every cross-core sync comes from here, with no "is this an overlap?" branch: each edge
    yields a wait before its consuming access and a set after its producing one, each on
    that access's own pipe and section.
    """
    sites = []
    for edge in graph.edges:
        got = event_ids.get(id(edge))
        if got is None:
            continue  # ids not declared -> the user opted out of this edge
        ids_node, slot_count = got
        skew = _skew_of(edge)
        # Same task with the consuming stage first: the stage chain forces the opposite
        # order to the one reuse needs, and no sync satisfies both. (Producer first is
        # merely redundant — see _is_redundant.)
        if (
            edge.kind == "WAR"
            and edge.task_off == 0
            and edge.src.buffer != edge.dst.buffer
            and edge.src.stage_idx > edge.dst.stage_idx
        ):
            raise InvalidOperation(
                f"pipeline: address reuse requires {edge.src} to finish before {edge.dst}, "
                f"but within one task the stage order runs them the other way round. "
                f"Reorder the stages, or stop sharing the address."
            )
        # A negative skew means the consumer waits on a task that has not run yet. Letting those
        # tail waits through needs a guard answering "does the partner task still exist?", which
        # the inner loop cannot: the partner may belong to the NEXT outer iteration. Answering it
        # properly needs this core's total task count, a runtime value — so the shape is rejected
        # rather than synchronised approximately.
        if skew < 0:
            raise InvalidOperation(
                f"pipeline: sync edge {edge.src} -> {edge.dst} runs inverse-time at this "
                f"schedule — the consumer waits on task {-skew} step(s) ahead of it, which "
                f"has not executed yet (dist={edge.dist}, task_off={edge.task_off}).\n"
                f"Releasing such a wait would need a guard on the loop variable, which "
                f"cannot tell whether the partner task exists once the loop nest has more "
                f"than one level.\n"
                f"A lower `preload` often removes the skew (the stages then sit closer "
                f"together in the schedule); otherwise stop sharing the address between "
                f"'{edge.src.buffer}' and '{edge.dst.buffer}', or reorder the stages."
            )
        # Both ends share a lane, keyed by task % slot_count, so the skew must be a whole
        # number of rotations — guards and `% slot_count` id indexing both rely on it.
        if skew % slot_count:
            raise InvalidVal(
                f"pipeline: sync edge {edge.src} -> {edge.dst} has task offset {skew}, "
                f"which is not a multiple of its {slot_count} slots. Guards and event-id "
                f"pairing assume whole-rotation skew."
            )
        sites.append(
            SyncSite(
                stage=edge.dst.stage,
                side="pre",
                op="wait_cross_core",
                pipe=edge.dst.pipe,
                section=edge.dst.section,
                ids_node=ids_node,
                slot_count=slot_count,
            )
        )
        sites.append(
            SyncSite(
                stage=edge.src.stage,
                side="post",
                op="set_cross_core",
                pipe=edge.src.pipe,
                section=edge.src.section,
                ids_node=ids_node,
                slot_count=slot_count,
            )
        )
    return sites


def plan_drain(graph: SyncGraph, event_ids: dict) -> list:
    """Instructions after the outermost loop, absorbing the tail stragglers.

    A skewed edge's first ``skew`` waits have no partner task behind them, so pre-fire hands
    them their permits up front (see plan_prefire). Those permits are sets with no partner
    wait, and at the tail the same skew strands that many sets with no partner wait either.
    Both are absorbed here, since hardware needs the two counts to match exactly.

    Emitted once per kernel, not per outer iteration: ``_pl_task_id`` spans them all.
    """
    drain = []
    for edge in graph.edges:
        got = event_ids.get(id(edge))
        if got is None:
            continue
        ids_node, slot_count = got
        skew = _skew_of(edge)
        if skew <= 0:
            continue  # negative skew is rejected in plan_sync_sites; zero strands nothing
        acc = edge.dst
        op = "wait_cross_core"
        for i in range(skew):
            drain.append(
                SyncSite(
                    stage=acc.stage,
                    side="drain",
                    op=op,
                    pipe=acc.pipe,
                    section=acc.section,
                    ids_node=ids_node,
                    slot_count=slot_count,
                    id_index=i % slot_count,
                )
            )
    return drain


def plan_prefire(graph: SyncGraph, event_ids: dict) -> list:
    """Permits released before the outermost loop, covering a skewed edge's head.

    A WAR edge with skew ``n`` pairs ``dst(task t)`` with ``src(task t - n)``, so its first
    ``n`` waits have no partner behind them. Pre-firing ``n`` permits lets exactly those
    through and leaves the wait itself unconditional; the matching drain consumes them.

    Guarding those waits on a task counter instead was dropped: a guard has to ask whether
    the partner task exists, and the only counters available are per-loop, which cannot
    answer that once the loop nest has more than one level.
    """
    prefire = []
    for edge in graph.edges:
        got = event_ids.get(id(edge))
        if got is None:
            continue
        ids_node, slot_count = got
        skew = _skew_of(edge)
        if skew <= 0:
            continue  # negative skew is rejected in plan_sync_sites; zero needs no head
        acc = edge.src  # the reader side owns the set
        for i in range(skew):
            prefire.append(
                SyncSite(
                    stage=acc.stage,
                    side="prefire",
                    op="set_cross_core",
                    pipe=acc.pipe,
                    section=acc.section,
                    ids_node=ids_node,
                    slot_count=slot_count,
                    id_index=i % slot_count,
                )
            )
    return prefire


@dataclass
class SyncPlan:
    """Every sync instruction the kernel needs, in the three places they go.

    A complete handover is all three together: ``prefire`` releases the permits an unguarded
    head would otherwise wait for, ``sites`` is the steady-state wait/set around each stage,
    and ``drain`` consumes the sets left stranded at the tail. They are planned as one unit
    because they must agree — hardware requires the set and wait counts to match exactly,
    and which edge falls to pre-fire versus a guard decides which of the three covers it.
    """

    sites: list  # in-loop, indexed by the running task id
    prefire: list  # before the outermost loop, literal id indices
    drain: list  # after the outermost loop, literal id indices
    # The event-id groups allocated while planning THIS loop, as [(var_name, ast_literal)].
    # Declared with this loop rather than with the kernel, so another loop's groups are not
    # repeated ahead of it.
    reuse_ids: list = field(default_factory=list)


def plan_sync(graph: SyncGraph, info) -> SyncPlan:
    """Resolve each edge's event ids once, then plan all three parts of the sync from them.

    Resolving is a step of its own, ahead of the three planners, because it ALLOCATES: an
    address-reuse edge has no user-declared ids, so the framework mints a group and appends
    its declaration to ``info.sync.reuse_ids``. Doing it here makes the ids a function of the
    graph, and all three planners read the same answer.
    """
    # Everything appended to the kernel-level list from here on was allocated for THIS
    # plan, which is what the slice below picks out.
    first_new = len(info.sync.reuse_ids)
    event_ids = resolve_event_ids(graph, info)
    return SyncPlan(
        sites=plan_sync_sites(graph, event_ids),
        prefire=plan_prefire(graph, event_ids),
        drain=plan_drain(graph, event_ids),
        reuse_ids=info.sync.reuse_ids[first_new:],
    )


def sites_for(sites: list, stage: str) -> tuple:
    """(pre_sites, post_sites) for one stage."""
    pre, post = [], []
    for site in sites:
        if site.stage != stage:
            continue
        (pre if site.side == "pre" else post).append(site)
    return pre, post
