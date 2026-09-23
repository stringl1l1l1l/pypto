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
    SyncCoreType,
)

from ..._errors import InvalidArgument, InvalidOperation, InvalidShape, InvalidType, InvalidVal, NotSupported
from .._utils import _get_span_or_capture, _is_int, _normalize_expr, _to_make_tuple
from ._op_registry import OpSpec, op_impl, register_table

# Hardware resource ranges, inclusive on both ends. These are the single definition of each limit;
# the pipeline scanner and the make_tile_group parser import them rather than restating them.
MAX_EVENT_ID = 15  # cross-core event ids
MAX_FLAG_EVENT_ID = 7  # set_flag / wait_flag event ids
MAX_MUTEX_ID = 31  # tile mutex ids

# A5 intra-core flag paths are execution-side specific. The maps are
# directional: each key is set_pipe and its value contains valid wait_pipe values.
_A5_AIC_SYNC_WAIT_PIPES = {
    PipeType.MTE1: (PipeType.MTE2, PipeType.M, PipeType.MTE3, PipeType.FIX),
    PipeType.MTE2: (
        PipeType.MTE1,
        PipeType.M,
        PipeType.MTE3,
        PipeType.S,
        PipeType.FIX,
    ),
    PipeType.MTE3: (PipeType.MTE1, PipeType.MTE2, PipeType.S),
    PipeType.M: (PipeType.MTE1, PipeType.MTE2, PipeType.FIX, PipeType.S),
    PipeType.S: (PipeType.MTE2, PipeType.MTE3),
    PipeType.FIX: (
        PipeType.M,
        PipeType.MTE2,
        PipeType.S,
        PipeType.MTE3,
        PipeType.MTE1,
    ),
}

_A5_AIV_SYNC_WAIT_PIPES = {
    PipeType.MTE2: (PipeType.V, PipeType.MTE3, PipeType.S),
    PipeType.MTE3: (PipeType.V, PipeType.MTE2, PipeType.S),
    PipeType.V: (PipeType.MTE2, PipeType.MTE3, PipeType.S),
    PipeType.S: (PipeType.V, PipeType.MTE2, PipeType.MTE3),
}


def _is_integer_scalar_expr(value: Expr) -> bool:
    value_type = getattr(value, "type", None)
    return isinstance(value_type, _ir_core.ScalarType) and value_type.dtype.is_int()


def _normalize_integer_id_expr(value: int | Expr, span: Span, *, name: str, max_id: int) -> Expr:
    """Normalize an integer ID to an IR operand and validate statically known values."""
    if not isinstance(value, Expr) and not _is_int(value):
        raise InvalidType(f"{name} must be a Python int or an integer scalar expression", span=span)
    expr = _normalize_expr(value, span)
    if not _is_integer_scalar_expr(expr):
        raise InvalidType(f"{name} must be an integer scalar expression", span=span)
    if isinstance(expr, _ir_core.ConstInt):
        _check_id_range(expr.value, max_id, name, span=span)
    return expr


def _normalize_mutex_ids(mutex_ids: tuple | list | None, span: Span | None = None) -> list[int] | None:
    """Validate candidate IDs before they are converted to vector<int>."""
    if mutex_ids is None:
        return None
    if not isinstance(mutex_ids, (list, tuple)):
        raise InvalidType("mutex_ids must be a list, tuple, or None", span=span)
    normalized_mutex_ids = list(mutex_ids)
    if not normalized_mutex_ids:
        raise InvalidVal("mutex_ids must not be empty", span=span)
    for index, mutex_id in enumerate(normalized_mutex_ids):
        if not _is_int(mutex_id):
            raise InvalidType(f"mutex_ids[{index}] must be a Python int", span=span)
        _check_id_range(mutex_id, MAX_MUTEX_ID, "mutex_ids element", span=span)
    return normalized_mutex_ids


def _validate_concrete_pipe(pipe: PipeType, name: str, span: Span | None = None) -> None:
    if not isinstance(pipe, PipeType):
        raise InvalidType(f"{name} must be a PipeType, got {type(pipe).__name__}", span=span)
    if pipe == PipeType.ALL:
        raise InvalidArgument(f"{name} must identify one concrete pipe, got PipeType.ALL", span=span)


def _check_id_range(value: int, max_id: int, name: str, *, min_id: int = 0, span: Span | None = None) -> None:
    """Reject a hardware resource id outside its inclusive range."""
    from pypto_pro.language.parser.diagnostics import check_in_range

    check_in_range(value, min_id, max_id, subject=name, span=span)


def _validate_sync_pipes(
    set_pipe: PipeType,
    wait_pipe: PipeType,
    target: _ir_core.SectionKind,
    span: Span | None = None,
) -> None:
    _validate_concrete_pipe(set_pipe, "set_pipe", span)
    _validate_concrete_pipe(wait_pipe, "wait_pipe", span)
    if set_pipe == wait_pipe:
        raise InvalidOperation(f"set_pipe and wait_pipe must differ, got {set_pipe}")

    is_cube = target == _ir_core.SectionKind.Cube
    sync_wait_pipes = _A5_AIC_SYNC_WAIT_PIPES if is_cube else _A5_AIV_SYNC_WAIT_PIPES
    supported_wait_pipes = sync_wait_pipes.get(set_pipe, ())
    side = " on AIC" if is_cube else " on AIV"

    if wait_pipe not in supported_wait_pipes:
        supported = ", ".join(str(pipe) for pipe in supported_wait_pipes)
        raise NotSupported(
            f"unsupported 3510 synchronization path{side}: {set_pipe} -> {wait_pipe}; "
            f"supported wait_pipe values for {set_pipe}: {supported}"
        )


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
        event_id: Event identifier
        span: Optional source span for debugging
    """
    actual_span = _get_span_or_capture(span, frame_offset=2)
    kwargs = {"set_pipe": set_pipe, "wait_pipe": wait_pipe}
    event_id_expr = _normalize_integer_id_expr(
        event_id, actual_span, name="event_id", max_id=MAX_FLAG_EVENT_ID
    )
    return _ir_core.create_op_call(op_name + "_dyn", [event_id_expr], kwargs, actual_span)


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
        Call expression for system.sync_src_dyn.
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
        Call expression for system.sync_dst_dyn.
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
    _validate_concrete_pipe(pipe, "pipe")

    if isinstance(event_id, Expr):
        return _ir_core.create_op_call(
            "system.set_cross_core_dyn", [event_id], {"pipe": pipe, "sync_mode": sync_mode}, actual_span
        )
    _check_id_range(event_id, MAX_EVENT_ID, "event_id")
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
    _validate_concrete_pipe(pipe, "pipe")

    if isinstance(event_id, Expr):
        return _ir_core.create_op_call(
            "system.wait_cross_core_dyn",
            [event_id],
            {"pipe": pipe, "sync_mode": sync_mode},
            actual_span,
        )
    _check_id_range(event_id, MAX_EVENT_ID, "event_id")
    kwargs = {"pipe": pipe, "event_id": event_id, "sync_mode": sync_mode}
    return _ir_core.create_op_call("system.wait_cross_core", [], kwargs, actual_span)


def sync_all(
    *,
    core_type: SyncCoreType = SyncCoreType.MIX,
    span: Span | None = None,
) -> Call:
    """Global core synchronization (delegates to pto-isa SYNCALL).

    Uses FFTS hardware signal.

    Args:
        core_type: ``pl.SyncCoreType.AIV_ONLY``, ``pl.SyncCoreType.AIC_ONLY``, or ``pl.SyncCoreType.MIX`` (default).
            AIV-only syncs vector cores only; Mix syncs both AIC and AIV cores.
        span: Optional source span for debugging (auto-captured if not provided).

    Returns:
        Call expression for global core synchronization.
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("system.sync_all", [], {"core_type": core_type}, actual_span)


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
            raise InvalidType("dcci offset must be an int, Expr, list, tuple, or None")
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
    mutex_id: int | Expr,
    *,
    pipe: PipeType,
    actual_span: Span,
) -> Call:
    """Create one user-requested manual mutex operation."""
    _validate_concrete_pipe(pipe, "pipe", actual_span)
    mutex_id_expr = _normalize_integer_id_expr(
        mutex_id, actual_span, name="mutex_id", max_id=MAX_MUTEX_ID
    )

    kwargs: dict = {"pipe": pipe}
    return _ir_core.create_op_call(f"{op_name}_dyn", [mutex_id_expr], kwargs, actual_span)


def _create_mutex_dedup_op(
    op_name: str,
    *,
    pipe: PipeType,
    mutex_id_exprs: list[Expr],
    mutex_id_owner_indices: list[int] | None = None,
    mutex_ids_union: list | None = None,
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
        mutex_ids_union: Union of all TileGroup candidate mutex_id values.
        span: Source span.
    """
    actual_span = span if span is not None else _get_span_or_capture(span, frame_offset=3)
    _validate_concrete_pipe(pipe, "pipe", actual_span)
    if not mutex_id_exprs:
        raise InvalidArgument("mutex_id requires at least one expression")
    normalized_mutex_id_exprs = [
        _normalize_integer_id_expr(mutex_id, actual_span, name="mutex_id", max_id=MAX_MUTEX_ID)
        for mutex_id in mutex_id_exprs
    ]
    kwargs: dict = {"pipe": pipe}
    if mutex_id_owner_indices is not None:
        if len(mutex_id_owner_indices) != len(mutex_id_exprs):
            raise InvalidShape("mutex_id_owner_indices length must match mutex_id_exprs length")
        kwargs["mutex_id_owner_indices"] = list(mutex_id_owner_indices)
    normalized_mutex_ids = _normalize_mutex_ids(mutex_ids_union, actual_span)
    if normalized_mutex_ids is not None:
        kwargs["mutex_ids"] = normalized_mutex_ids
    return _ir_core.create_op_call(f"{op_name}_dyn", normalized_mutex_id_exprs, kwargs, actual_span)


def _mutex_op(
    op_name: str,
    *,
    pipe: PipeType,
    mutex_id: int | Expr,
    span: Span | None = None,
) -> Call:
    """Shared wrapper for mutex_lock / mutex_unlock (handles span capture)."""
    actual_span = _get_span_or_capture(span, frame_offset=2)
    return _create_mutex_op(
        op_name,
        mutex_id,
        pipe=pipe,
        actual_span=actual_span,
    )


def mutex_lock(
    *,
    pipe: PipeType,
    mutex_id: int | Expr,
    span: Span | None = None,
) -> Call:
    """Acquire a Mutex buffer-id token on ``pipe`` (A5).

    Blocks the ``pipe`` instruction queue until the previous holder of
    ``mutex_id`` releases it via :func:`mutex_unlock`.

    Args:
        pipe: PipeType for which to acquire the lock (e.g. PipeType.MTE2).
        mutex_id: MutexID 0-31 as a Python int or integer scalar IR expression.
        span: Optional source span (auto-captured when omitted).

    Returns:
        Call expression for system.mutex_lock_dyn.
    """
    return _mutex_op(
        "system.mutex_lock",
        pipe=pipe,
        mutex_id=mutex_id,
        span=span,
    )


def mutex_unlock(
    *,
    pipe: PipeType,
    mutex_id: int | Expr,
    span: Span | None = None,
) -> Call:
    """Release a previously acquired Mutex buffer-id token on ``pipe`` (A5).

    Must be paired with :func:`mutex_lock` on the same ``pipe`` and
    ``mutex_id``.

    Args:
        pipe: PipeType for which to release the lock.
        mutex_id: MutexID passed to the paired :func:`mutex_lock`.
        span: Optional source span (auto-captured when omitted).

    Returns:
        Call expression for system.mutex_unlock_dyn.
    """
    return _mutex_op(
        "system.mutex_unlock",
        pipe=pipe,
        mutex_id=mutex_id,
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


def _parse_system_sync(self, call: ast.Call, op_name: str):
    span = self.span_tracker.get_span(call)
    kwargs = self.parse_op_kwargs(call)
    _validate_sync_pipes(kwargs["set_pipe"], kwargs["wait_pipe"], self.target, span)
    return _create_sync_op(op_name, **kwargs, span=span)


@op_impl("system.sync_src")
def _parse_system_sync_src(self, call: ast.Call):
    return _parse_system_sync(self, call, "system.sync_src")


@op_impl("system.sync_dst")
def _parse_system_sync_dst(self, call: ast.Call):
    return _parse_system_sync(self, call, "system.sync_dst")


@op_impl("system.sync_all")
def _parse_system_sync_all(self, call: ast.Call):
    span = self.span_tracker.get_span(call)
    if call.args:
        raise InvalidArgument("sync_all does not accept positional arguments", span=span)
    kwargs = self.parse_op_kwargs(call)
    return sync_all(**kwargs, span=span)


@op_impl("system.set_mm_layout_transform")
def _parse_system_set_mm_layout_transform(self, call: ast.Call):
    call_span = self.span_tracker.get_span(call)
    kwargs = self.parse_op_kwargs(call)
    if "enabled" not in kwargs:

        raise InvalidArgument(
            "set_mm_layout_transform requires keyword argument 'enabled'",
            span=call_span,
        )
    enabled = bool(kwargs["enabled"])
    return set_mm_layout_transform(enabled=enabled, span=call_span)
