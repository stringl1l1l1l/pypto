# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""System operations for PyPTO IR.

System operations handle hardware synchronization primitives:
- sync_src / sync_dst: Set/Wait flag-based synchronization between pipes
- bar_*: Barrier synchronization for hardware pipelines
"""

from __future__ import annotations

import ast

from pypto.pypto_impl import ir as _ir_core
from pypto.pypto_impl.ir import (
    CacheLine,
    Call,
    CrossCoreSyncMode,
    DcciDst,
    Expr,
    PipeType,
    Span,
    SyncAllMode,
    SyncCoreType,
)

from .._utils import _get_span_or_capture, _normalize_expr, _to_make_tuple
from ._op_registry import OpSpec, op_impl, register_table

_MAX_EVENT_ID = 16


def _create_sync_op(
    op_name: str,
    *,
    set_pipe: PipeType,
    wait_pipe: PipeType,
    event_id: int | Expr,
    span: Span | None,
) -> Call:
    """Create a flag-based synchronization operation.

    Args:
        op_name: Operation name (e.g., "system.sync_src")
        set_pipe: Pipe that sets the flag
        wait_pipe: Pipe that waits on the flag
        event_id: Event identifier (int for static, Expr for dynamic)
        span: Optional source span for debugging
    """
    actual_span = _get_span_or_capture(span, frame_offset=2)
    if isinstance(event_id, Expr):
        kwargs = {"set_pipe": set_pipe, "wait_pipe": wait_pipe}
        return _ir_core.create_op_call(op_name + "_dyn", [event_id], kwargs, actual_span)
    kwargs = {"set_pipe": set_pipe, "wait_pipe": wait_pipe, "event_id": event_id}
    return _ir_core.create_op_call(op_name, [], kwargs, actual_span)


def _create_barrier_op(op_name: str, *, span: Span | None) -> Call:
    """Create a barrier synchronization operation.

    Args:
        op_name: Operation name (e.g., "system.bar_m")
        span: Optional source span for debugging
    """
    actual_span = _get_span_or_capture(span, frame_offset=2)
    return _ir_core.create_op_call(op_name, [], {}, actual_span)


def sync_src(
    *,
    set_pipe: PipeType,
    wait_pipe: PipeType,
    event_id: int | Expr,
    span: Span | None = None,
) -> Call:
    """Send a synchronization signal (Set Flag).

    Args:
        set_pipe: Pipe that sets the flag
        wait_pipe: Pipe that will wait on the flag
        event_id: Event identifier
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for system.sync_src
    """
    return _create_sync_op("system.sync_src", set_pipe=set_pipe, wait_pipe=wait_pipe, event_id=event_id, span=span)


def sync_dst(
    *,
    set_pipe: PipeType,
    wait_pipe: PipeType,
    event_id: int | Expr,
    span: Span | None = None,
) -> Call:
    """Wait for a synchronization signal (Wait Flag).

    Args:
        set_pipe: Pipe that sets the flag
        wait_pipe: Pipe that waits on the flag
        event_id: Event identifier
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for system.sync_dst
    """
    return _create_sync_op("system.sync_dst", set_pipe=set_pipe, wait_pipe=wait_pipe, event_id=event_id, span=span)


def bar_m(*, span: Span | None = None) -> Call:
    """Matrix unit barrier."""
    return _create_barrier_op("system.bar_m", span=span)


def bar_mte1(*, span: Span | None = None) -> Call:
    """MTE1 pipeline barrier."""
    return _create_barrier_op("system.bar_mte1", span=span)


def bar_mte2(*, span: Span | None = None) -> Call:
    """MTE2 pipeline barrier."""
    return _create_barrier_op("system.bar_mte2", span=span)


def bar_mte3(*, span: Span | None = None) -> Call:
    """MTE3 pipeline barrier."""
    return _create_barrier_op("system.bar_mte3", span=span)


def bar_fix(*, span: Span | None = None) -> Call:
    """FIX pipeline barrier."""
    return _create_barrier_op("system.bar_fix", span=span)


def bar_all(*, span: Span | None = None) -> Call:
    """Global barrier synchronization."""
    return _create_barrier_op("system.bar_all", span=span)


def set_cross_core(
    *,
    pipe: PipeType,
    event_id: int | Expr,
    sync_mode: CrossCoreSyncMode = CrossCoreSyncMode.INTRA_BLOCK,
    span: Span | None = None,
) -> Call:
    """Set for a synchronization signal (Cross core).

    Args:
        pipe: Pipe that sets the flag
        event_id: Event identifier (int for static, Expr for dynamic)
        sync_mode: Cross-core sync mode. ``pl.CrossCoreSyncMode.INTRA_BLOCK`` (default, mode 2)
            for AIC↔AIV both subcores, ``UNICAST_BLOCK`` (mode 3) for AIC↔AIV one subcore,
            ``INTER_BLOCK`` (mode 0) for inter-core, ``INTER_SUBBLOCK`` (mode 1) for AIV-to-AIV.
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for system.set_cross_core
    """
    actual_span = _get_span_or_capture(span)
    if isinstance(event_id, Expr):
        return _ir_core.create_op_call(
            "system.set_cross_core_dyn", [event_id], {"pipe": pipe, "sync_mode": sync_mode}, actual_span
        )
    if not 0 <= event_id < _MAX_EVENT_ID:
        raise ValueError(f"event_id must be in [0, {_MAX_EVENT_ID}), got {event_id}")
    kwargs = {"pipe": pipe, "event_id": event_id, "sync_mode": sync_mode}
    return _ir_core.create_op_call("system.set_cross_core", [], kwargs, actual_span)


def wait_cross_core(
    *,
    pipe: PipeType,
    event_id: int | Expr,
    sync_mode: CrossCoreSyncMode = CrossCoreSyncMode.INTRA_BLOCK,
    span: Span | None = None,
) -> Call:
    """Wait for a synchronization signal (Cross core).

    Args:
        pipe: Pipe that waits the flag
        event_id: Event identifier (int for static, Expr for dynamic)
        sync_mode: Cross-core sync mode. Must match the paired set_cross_core.
            ``pl.CrossCoreSyncMode.INTRA_BLOCK`` (default, mode 2) waits both VEC subcores,
            ``UNICAST_BLOCK`` (mode 3) waits one VEC subcore. ``INTER_BLOCK`` (mode 0)
            for inter-core, ``INTER_SUBBLOCK`` (mode 1) for AIV-to-AIV.
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for system.wait_cross_core
    """
    actual_span = _get_span_or_capture(span)

    if isinstance(event_id, Expr):
        return _ir_core.create_op_call(
            "system.wait_cross_core_dyn",
            [event_id],
            {"pipe": pipe, "sync_mode": sync_mode},
            actual_span,
        )
    if not 0 <= event_id < _MAX_EVENT_ID:
        raise ValueError(f"event_id must be in [0, {_MAX_EVENT_ID}), got {event_id}")
    kwargs = {"pipe": pipe, "event_id": event_id, "sync_mode": sync_mode}
    return _ir_core.create_op_call("system.wait_cross_core", [], kwargs, actual_span)


def sync_all(
    workspaces: list[Expr | int] | None = None,
    *,
    core_type: SyncCoreType = SyncCoreType.MIX,
    mode: SyncAllMode = SyncAllMode.HARD,
    span: Span | None = None,
) -> Call:
    """Global core synchronization (delegates to pto-isa SYNCALL).

    Hard mode (default): no workspace needed, uses FFTS hardware signal.
    Soft mode: requires workspace buffers for GM-polling synchronization.

    Args:
        workspaces: List of workspace parameters for soft mode. Dispatched by type:
              - TensorType → gm_workspace (required for soft mode)
              - Vec TileType → ub_workspace (required for aiv_only/mix)
              - Mat TileType → l1_workspace (required for aic_only/mix)
              - int/Expr scalar → used_cores (optional, defaults to 0 = all cores)
            Defaults to empty list. Not needed for hard mode.
        core_type: ``pl.SyncCoreType.AIV_ONLY``, ``pl.SyncCoreType.AIC_ONLY``, or ``pl.SyncCoreType.MIX`` (default).
            AIV-only syncs vector cores only; Mix syncs both AIC and AIV cores.
        mode: ``pl.SyncAllMode.HARD`` (default) or ``pl.SyncAllMode.SOFT``.
            Hard mode uses FFTS hardware; Soft mode uses GM workspace polling.
        span: Optional source span for debugging (auto-captured if not provided).

    Returns:
        Call expression for global core synchronization.
    """
    if workspaces is None:
        workspaces = []

    kwargs: dict[str, object] = {"mode": mode, "core_type": core_type}

    if mode == SyncAllMode.HARD:
        if workspaces:
            raise ValueError("Hard mode sync_all does not accept workspace arguments")

    elif mode == SyncAllMode.SOFT:
        if not workspaces:
            raise ValueError("Soft mode sync_all requires workspaces list (e.g. [gm, ub])")

    # Build args[0]: always a MakeTuple (empty for hard, populated for soft)
    ws_tuple = _to_make_tuple(workspaces)

    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("system.sync_all", [ws_tuple], kwargs, actual_span)


def dcci(
    target: Expr,
    offset: int | Expr | tuple[int | Expr, ...] | list[int | Expr] | None = None,
    *,
    cache_line: CacheLine = CacheLine.ENTIRE_DATA_CACHE,
    dst: DcciDst = DcciDst.AUTO,
    span: Span | None = None,
) -> Call:
    """Data Cache Clean and Invalid for GM tensor or UB tile.

    Args:
        target: GM tensor or UB tile.
        offset: Tensor target uses per-dimension offsets or a scalar element
            offset. Tile target uses a scalar element offset. If omitted, the
            target base address is used.
        cache_line: ``pl.CacheLine.SINGLE_CACHE_LINE`` or ``pl.CacheLine.ENTIRE_DATA_CACHE``.
        dst: DCCI destination. ``pl.DcciDst.AUTO`` maps tensor to CACHELINE_OUT and tile to
            CACHELINE_UB.
        span: Optional source span for debugging (auto-captured if not provided).

    Returns:
        Call expression for system.dcci.
    """
    actual_span = _get_span_or_capture(span)
    args: list[Expr] = [target]
    if offset is not None:
        if isinstance(offset, (list, tuple)):
            args.append(_to_make_tuple(offset, actual_span))
        elif isinstance(offset, (int, Expr)):
            args.append(_normalize_expr(offset, actual_span))
        else:
            raise TypeError("dcci offset must be an int, Expr, list, tuple, or None")
    kwargs = {"cache_line": cache_line, "dst": dst}
    return _ir_core.create_op_call("system.dcci", args, kwargs, actual_span)


# ============================================================================
# Mutex (Buffer-ID Token) — A5 only
# ----------------------------------------------------------------------------
# Alternative to event-id based sync_src/sync_dst: uses a buffer-id token
# (MutexID, range 0-31) to enforce ordering between pipes. Lowered to
# pto.get_buf / pto.rls_buf.
# ============================================================================


def _create_mutex_op(
    op_name: str,
    *,
    pipe: PipeType,
    mutex_id: int | Expr,
    mode: int,
    actual_span: Span,
) -> Call:
    """Create a manual mutex lock/unlock operation."""
    if isinstance(mutex_id, Expr):
        kwargs: dict = {"pipe": pipe, "mode": mode, "auto_mutex": False}
        return _ir_core.create_op_call(f"{op_name}_dyn", [mutex_id], kwargs, actual_span)
    kwargs = {"pipe": pipe, "mutex_id": mutex_id, "mode": mode, "auto_mutex": False}
    return _ir_core.create_op_call(op_name, [], kwargs, actual_span)


def _create_mutex_dedup_op(
    op_name: str,
    *,
    pipe: PipeType,
    mutex_id_exprs: list[Expr],
    mutex_id_owner_indices: list[int] | None = None,
    mode: int = 0,
    mutex_ids_union: list | None = None,
    auto_mutex: bool = False,
    span: Span | None = None,
) -> Call:
    """Create a dedup mutex lock/unlock for N runtime mutex-id expressions.

    Emits a single ``system.mutex_lock_dyn`` / ``system.mutex_unlock_dyn`` IR Call
    with multiple mutex_id expressions in args. Expressions with the same owner
    index in ``mutex_id_owner_indices`` are known distinct; CCE only generates
    runtime if-guards across different Tiles. Lock and unlock both use
    first-occurrence order.

    Args:
        op_name: Base operation name ("system.mutex_lock" or "system.mutex_unlock").
        pipe: Pipe to lock on.
        mutex_id_exprs: List of N mutex_id IR expressions (already normalized to Expr).
        mutex_id_owner_indices: Owner index for every expression. Expressions with
            the same index come from one Tile and are guaranteed distinct.
        mode: Mutex mode (default 0).
        mutex_ids_union: Union of all candidate mutex_id values (for ShouldSkipVPipeMutex).
        auto_mutex: Whether the parser generated this operation for automatic synchronization.
        span: Source span.
    """
    actual_span = span if span is not None else _get_span_or_capture(span, frame_offset=3)
    kwargs: dict = {
        "pipe": pipe,
        "mode": mode,
        "auto_mutex": auto_mutex,
    }
    if mutex_id_owner_indices is not None:
        if len(mutex_id_owner_indices) != len(mutex_id_exprs):
            raise ValueError("mutex_id_owner_indices length must match mutex_id_exprs length")
        kwargs["mutex_id_owner_indices"] = list(mutex_id_owner_indices)
    if mutex_ids_union is not None:
        kwargs["mutex_ids"] = list(mutex_ids_union)
    return _ir_core.create_op_call(f"{op_name}_dyn", mutex_id_exprs, kwargs, actual_span)


def _mutex_op(
    op_name: str,
    *,
    pipe: PipeType,
    mutex_id: int | Expr,
    mode: int = 0,
    span: Span | None = None,
) -> Call:
    """Shared wrapper for mutex_lock / mutex_unlock (handles span capture)."""
    actual_span = _get_span_or_capture(span, frame_offset=2)
    return _create_mutex_op(
        op_name,
        pipe=pipe,
        mutex_id=mutex_id,
        mode=mode,
        actual_span=actual_span,
    )


def mutex_lock(
    *,
    pipe: PipeType,
    mutex_id: int | Expr,
    mode: int = 0,
    span: Span | None = None,
) -> Call:
    """Acquire a Mutex buffer-id token on ``pipe`` (A5).

    Blocks the ``pipe`` instruction queue until the previous holder of
    ``mutex_id`` releases it via :func:`mutex_unlock`.

    Args:
        pipe: PipeType for which to acquire the lock (e.g. PipeType.MTE2).
        mutex_id: MutexID (0-31, per Ascend C Mutex ISASI spec).
            May be a static int/ConstInt or a dynamic IR Expr; when dynamic, the
            codegen emits an if-chain of static `pto.get_buf` using
            ``mutex_ids`` as the comparison targets.
        mode: Optional mode attribute (default 0).
        span: Optional source span (auto-captured when omitted).

    Returns:
        Call expression for system.mutex_lock / system.mutex_lock_dyn.
    """
    return _mutex_op(
        "system.mutex_lock",
        pipe=pipe,
        mutex_id=mutex_id,
        mode=mode,
        span=span,
    )


def mutex_unlock(
    *,
    pipe: PipeType,
    mutex_id: int | Expr,
    mode: int = 0,
    span: Span | None = None,
) -> Call:
    """Release a previously acquired Mutex buffer-id token on ``pipe`` (A5).

    Must be paired with :func:`mutex_lock` on the same ``pipe`` and
    ``mutex_id``.

    Args:
        pipe: PipeType for which to release the lock.
        mutex_id: MutexID passed to the paired :func:`mutex_lock`.
        mode: Optional mode attribute (default 0).
        span: Optional source span (auto-captured when omitted).

    Returns:
        Call expression for system.mutex_unlock / system.mutex_unlock_dyn.
    """
    return _mutex_op(
        "system.mutex_unlock",
        pipe=pipe,
        mutex_id=mutex_id,
        mode=mode,
        span=span,
    )


def set_mm_layout_transform(*, enabled: bool, span: Span | None = None) -> Call:
    """Set matmul layout transform mode for fixpipe drain direction.

    When enabled, fixpipe drains L0C in N-direction (column-first) instead of
    M-direction (row-first). This allows cube and fixpipe to access L0C along
    orthogonal axes within the same slot, eliminating RAW hazards and enabling
    single-buffer L0C with K-accumulation.

    Args:
        enabled: True to enable N-direction drain, False to restore M-direction.
        span: Optional source span (auto-captured when omitted).

    Returns:
        Call expression for system.set_mm_layout_transform.
    """
    actual_span = _get_span_or_capture(span)
    kwargs = {"enabled": int(enabled)}
    return _ir_core.create_op_call("system.set_mm_layout_transform", [], kwargs, actual_span)


# ---------------------------------------------------------------------------
# Declarative op registration + special handlers
# ---------------------------------------------------------------------------

register_table(
    {
        # kwargs only
        "system.sync_src": OpSpec(builder=sync_src, parse_args=False),
        "system.sync_dst": OpSpec(builder=sync_dst, parse_args=False),
        "system.set_cross_core": OpSpec(builder=set_cross_core, parse_args=False),
        "system.wait_cross_core": OpSpec(builder=wait_cross_core, parse_args=False),
        "system.mutex_lock": OpSpec(builder=mutex_lock, parse_args=False),
        "system.mutex_unlock": OpSpec(builder=mutex_unlock, parse_args=False),
        # no args, no kwargs
        "system.bar_m": OpSpec(builder=bar_m, parse_args=False, parse_kwargs=False),
        "system.bar_mte1": OpSpec(builder=bar_mte1, parse_args=False, parse_kwargs=False),
        "system.bar_mte2": OpSpec(builder=bar_mte2, parse_args=False, parse_kwargs=False),
        "system.bar_mte3": OpSpec(builder=bar_mte3, parse_args=False, parse_kwargs=False),
        "system.bar_fix": OpSpec(builder=bar_fix, parse_args=False, parse_kwargs=False),
        "system.bar_all": OpSpec(builder=bar_all, parse_args=False, parse_kwargs=False),
        "get_block_idx": OpSpec(ir_name="get_block_idx", parse_args=False, parse_kwargs=False),
        "get_subblock_idx": OpSpec(ir_name="get_subblock_idx", parse_args=False, parse_kwargs=False),
        "get_block_num": OpSpec(ir_name="get_block_num", parse_args=False, parse_kwargs=False),
        "get_subblock_num": OpSpec(ir_name="get_subblock_num", parse_args=False, parse_kwargs=False),
        "get_spr": OpSpec(ir_name="get_spr", parse_args=False, parse_kwargs=False),
        # args + kwargs
        "system.dcci": OpSpec(builder=dcci),
    }
)


@op_impl("system.sync_all")
def _parse_system_sync_all(self, call: ast.Call):
    span = self.span_tracker.get_span(call)
    kwargs = self.parse_op_kwargs(call)
    workspaces = self.parse_expression(call.args[0]) if call.args else None
    return sync_all(workspaces, **kwargs, span=span)


@op_impl("system.set_mm_layout_transform")
def _parse_system_set_mm_layout_transform(self, call: ast.Call):
    call_span = self.span_tracker.get_span(call)
    kwargs = self.parse_op_kwargs(call)
    if "enabled" not in kwargs:
        from pypto_pro.language.parser.diagnostics import ParserSyntaxError

        raise ParserSyntaxError(
            "set_mm_layout_transform requires keyword argument 'enabled'",
            span=call_span,
        )
    enabled = bool(kwargs["enabled"])
    return set_mm_layout_transform(enabled=enabled, span=call_span)
