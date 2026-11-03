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
- dsb: Data Synchronization Barrier
"""

from __future__ import annotations

from pypto.pypto_impl import NpuArch
from pypto.pypto_impl import ir as _ir_core
from pypto.pypto_impl.ir import (
    CacheLine,
    Call,
    CrossCoreSyncMode,
    DcciDst,
    Expr,
    MemDsb,
    PipeType,
    Span,
    SyncCoreType,
)

from ..._errors import InvalidArgument, InvalidOperation, InvalidType, NotSupported, current_span
from .._utils import _get_span_or_capture, require_int
from ._op_registry import op_impl

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


def _validate_integer_id(value: Expr, span: Span, *, name: str, max_id: int) -> int | None:
    """Validate an IR integer ID and its statically known bounds."""
    with current_span(span):
        constant = require_int(value, subject=name)
        if constant is not None:
            _check_id_range(constant, max_id, name, span=span)
    return constant


def _validate_concrete_pipe(pipe: PipeType, name: str, span: Span | None = None) -> None:
    if not isinstance(pipe, PipeType):
        raise InvalidType(f"{name} must be a PipeType, got {type(pipe).__name__}", span=span)
    if pipe == PipeType.ALL:
        raise InvalidArgument(f"{name} must identify one concrete pipe, got PipeType.ALL", span=span)


def _require_bool(value: object, name: str, *, span: Span | None = None) -> bool:
    """Return *value* only when it is already a Python bool (no truthiness coercion)."""
    if not isinstance(value, bool):
        raise InvalidType(f"{name} must be a bool, got {type(value).__name__}", span=span)
    return value


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
            f"unsupported {NpuArch.DAV_3510.name} synchronization path{side}: {set_pipe} -> {wait_pipe}; "
            f"supported wait_pipe values for {set_pipe}: {supported}"
        )


def _create_sync_op(
    op_name: str,
    *,
    set_pipe: PipeType,
    wait_pipe: PipeType,
    event_id: Expr,
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
    _validate_integer_id(
        event_id, actual_span, name="event_id", max_id=MAX_FLAG_EVENT_ID
    )
    return _ir_core.create_op_call(op_name + "_dyn", [event_id], kwargs, actual_span)


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
    event_id: Expr,
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
    event_id: Expr,
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


def wait_pre_task_end(*, force: bool = True, span: Span | None = None) -> Call:
    """Early-start inter-task synchronization (SuperKernel cap bit 1).

    In a fused SuperKernel, ``WaitPreTaskEnd`` lets the instructions *before* the call
    run in parallel with the *previous* sub-kernel (asc-devkit ``WaitPreTaskEnd``). Pair
    with ``pl.jit(sk_cap=pl.SkCap.CAP_1_EARLY_START_WAIT)`` (early start wait flag): the framework fills
    ``sysArgs->skTaskSyncCfg`` and the generated ``__sk__`` entry propagates it into the
    block-local ``g_super_kernel_early_start_config`` (V3 macro behavior). Constraints:
    call it exactly once per core, in every TilingKey branch, and avoid flagId conflicts
    with ``sync_all`` / ``set_cross_core``.

    ``force`` defaults to ``True``: empirically the default ``WaitPreTaskEnd()``
    (FORCE=false) is compiled out in the SK sub-kernel context, while
    ``WaitPreTaskEnd<true>()`` emits the real sync instructions (verified via binary
    diff on Ascend 950 / CANN 9.3.0).

    Args:
        force: Force early-start behavior (``AscendC::WaitPreTaskEnd<true>()``).
        span: Optional source span for debugging (auto-captured if not provided).

    Returns:
        Call expression for system.wait_pre_task_end.
    """
    actual_span = _get_span_or_capture(span)
    kwargs: dict[str, object] = {"force": _require_bool(force, "force", span=actual_span)}
    return _ir_core.create_op_call("system.wait_pre_task_end", [], kwargs, actual_span)


def set_next_task_start(*, force: bool = True, span: Span | None = None) -> Call:
    """Early-start inter-task synchronization (SuperKernel cap bit 2).

    In a fused SuperKernel, ``SetNextTaskStart`` lets the instructions *after* the call
    run in parallel with the *next* sub-kernel (asc-devkit ``SetNextTaskStart``). Pair
    with ``pl.jit(sk_cap=pl.SkCap.CAP_2_EARLY_START_SET)`` (early start set flag): the framework fills
    ``sysArgs->skTaskSyncCfg`` and the generated ``__sk__`` entry propagates it into the
    block-local ``g_super_kernel_early_start_config`` (V3 macro behavior). Constraints:
    call it exactly once per core, after the operator's last data-movement instruction,
    in every TilingKey branch, and avoid flagId conflicts with ``sync_all`` /
    ``set_cross_core``.

    ``force`` defaults to ``True``: empirically the default ``SetNextTaskStart()``
    (FORCE=false) is compiled out in the SK sub-kernel context, so the forced
    ``SetNextTaskStart<..., true>()`` form (emitting the real sync instructions) is used.

    Args:
        force: Force early-start behavior (``AscendC::SetNextTaskStart<..., true>()``).
        span: Optional source span for debugging (auto-captured if not provided).

    Returns:
        Call expression for system.set_next_task_start.
    """
    actual_span = _get_span_or_capture(span)
    kwargs: dict[str, object] = {"force": _require_bool(force, "force", span=actual_span)}
    return _ir_core.create_op_call("system.set_next_task_start", [], kwargs, actual_span)


def set_cross_core(
    *,
    pipe: PipeType,
    event_id: Expr,
    sync_mode: CrossCoreSyncMode = CrossCoreSyncMode.INTRA_BLOCK,
    span: Span | None = None,
) -> Call:
    """Set for a synchronization signal (Cross core).

    Args:
        pipe: Pipe that sets the flag
        event_id: Integer IR constant for static IDs, or an IR expression for dynamic IDs
        sync_mode: Cross-core sync mode. ``pl.CrossCoreSyncMode.INTRA_BLOCK`` (default, mode 2)
            for AIC↔AIV both subcores, ``UNICAST_BLOCK`` (mode 3) for AIC↔AIV one subcore,
            ``INTER_BLOCK`` (mode 0) for inter-core, ``INTER_SUBBLOCK`` (mode 1) for AIV-to-AIV.
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for system.set_cross_core
    """
    actual_span = _get_span_or_capture(span)
    _validate_concrete_pipe(pipe, "pipe")
    if pipe == PipeType.S and sync_mode not in (
        CrossCoreSyncMode.INTRA_BLOCK,
        CrossCoreSyncMode.UNICAST_BLOCK,
    ):
        raise NotSupported(
            "set_cross_core supports PipeType.S only with sync_mode=CrossCoreSyncMode.INTRA_BLOCK "
            "or CrossCoreSyncMode.UNICAST_BLOCK",
            span=actual_span,
        )

    constant = _validate_integer_id(event_id, actual_span, name="event_id", max_id=MAX_EVENT_ID)
    if constant is not None:
        kwargs = {"pipe": pipe, "event_id": constant, "sync_mode": sync_mode}
        return _ir_core.create_op_call("system.set_cross_core", [], kwargs, actual_span)
    return _ir_core.create_op_call(
        "system.set_cross_core_dyn", [event_id], {"pipe": pipe, "sync_mode": sync_mode}, actual_span,
    )


def wait_cross_core(
    *,
    pipe: PipeType,
    event_id: Expr,
    sync_mode: CrossCoreSyncMode = CrossCoreSyncMode.INTRA_BLOCK,
    span: Span | None = None,
) -> Call:
    """Wait for a synchronization signal (Cross core).

    Args:
        pipe: Pipe that waits the flag
        event_id: Integer IR constant for static IDs, or an IR expression for dynamic IDs
        sync_mode: Cross-core sync mode. Prefer the same mode as the paired set_cross_core;
            ensure the participating cores and event IDs match.
            ``pl.CrossCoreSyncMode.INTRA_BLOCK`` (default, mode 2) waits both VEC subcores,
            ``UNICAST_BLOCK`` (mode 3) waits one VEC subcore. ``INTER_BLOCK`` (mode 0)
            for inter-core, ``INTER_SUBBLOCK`` (mode 1) for AIV-to-AIV.
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for system.wait_cross_core
    """
    actual_span = _get_span_or_capture(span)
    _validate_concrete_pipe(pipe, "pipe")

    constant = _validate_integer_id(event_id, actual_span, name="event_id", max_id=MAX_EVENT_ID)
    if constant is not None:
        kwargs = {"pipe": pipe, "event_id": constant, "sync_mode": sync_mode}
        return _ir_core.create_op_call("system.wait_cross_core", [], kwargs, actual_span)
    return _ir_core.create_op_call(
        "system.wait_cross_core_dyn", [event_id], {"pipe": pipe, "sync_mode": sync_mode}, actual_span,
    )


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
    offset: Expr | None = None,
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
    if offset is not None and not isinstance(getattr(offset, "type", None), _ir_core.NoneType):
        args.append(offset)
    kwargs = {"cache_line": cache_line, "dst": dst}
    return _ir_core.create_op_call("system.dcci", args, kwargs, actual_span)


def dsb(*, scope: MemDsb = MemDsb.ALL, span: Span | None = None) -> Call:
    """Data Synchronization Barrier (pto-isa dsb).

    Args:
        scope: ``pl.MemDsb`` value selecting which accesses the barrier orders:
            ``ALL`` (default) all memory accesses, ``DDR`` DDR accesses, ``UB`` UB
            accesses. The scope restricts ordering; the barrier still only orders this core's
            own accesses and never synchronizes other cores.
        span: Optional source span for debugging (auto-captured if not provided).

    Returns:
        Call expression for system.dsb.
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("system.dsb", [], {"scope": scope}, actual_span)


# ============================================================================
# Mutex (Buffer-ID Token) — A5 only
# ----------------------------------------------------------------------------
# Alternative to event-id based sync_src/sync_dst: uses a buffer-id token
# (MutexID, range 0-31) to enforce ordering between pipes. Lowered to
# pto.get_buf / pto.rls_buf.
# ============================================================================


def _create_mutex_op(
    op_name: str,
    mutex_id: Expr,
    *,
    pipe: PipeType,
    actual_span: Span,
) -> Call:
    """Create one user-requested manual mutex operation."""
    _validate_concrete_pipe(pipe, "pipe", actual_span)
    _validate_integer_id(
        mutex_id, actual_span, name="mutex_id", max_id=MAX_MUTEX_ID
    )

    return _ir_core.create_op_call(
        f"{op_name}_dyn",
        [mutex_id],
        {"pipe": pipe},
        actual_span,
    )



def _mutex_op(
    op_name: str,
    *,
    pipe: PipeType,
    mutex_id: Expr,
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
    mutex_id: Expr,
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
    mutex_id: Expr,
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
# Value-facing implementations
# ---------------------------------------------------------------------------

def _build_system_sync(op_name: str, *, context, span: Span, **kwargs):
    _validate_sync_pipes(kwargs["set_pipe"], kwargs["wait_pipe"], context.target, span)
    return _create_sync_op(op_name, **kwargs, span=span)


@op_impl("system.sync_src")
def _build_system_sync_src(self, *, span: Span, **kwargs):
    return _build_system_sync("system.sync_src", context=self, span=span, **kwargs)


@op_impl("system.sync_dst")
def _build_system_sync_dst(self, *, span: Span, **kwargs):
    return _build_system_sync("system.sync_dst", context=self, span=span, **kwargs)


@op_impl("system.wait_pre_task_end")
def _build_system_wait_pre_task_end(self, *, force: bool = True, span: Span):
    force = self.require_const_value(
        force, "force", expects="bool",
        hint="force selects the early-start template argument at compile time; "
        "pass a bool literal, e.g. force=True.",
    )
    return wait_pre_task_end(force=force, span=span)


@op_impl("system.set_next_task_start")
def _build_system_set_next_task_start(self, *, force: bool = True, span: Span):
    force = self.require_const_value(
        force, "force", expects="bool",
        hint="force selects the early-start template argument at compile time; "
        "pass a bool literal, e.g. force=True.",
    )
    return set_next_task_start(force=force, span=span)


@op_impl("system.set_mm_layout_transform")
def _build_system_set_mm_layout_transform(self, *, enabled, span: Span):
    enabled = self.require_const_value(
        enabled, "enabled", expects="bool",
        hint="enabled must be known at compile time.",
    )
    return set_mm_layout_transform(enabled=enabled, span=span)


@op_impl("get_block_idx")
def _build_get_block_idx(self, *, span: Span | None = None):
    return _ir_core.create_op_call("get_block_idx", [], {}, span)


@op_impl("get_subblock_idx")
def _build_get_subblock_idx(self, *, span: Span | None = None):
    return _ir_core.create_op_call("get_subblock_idx", [], {}, span)


@op_impl("get_block_num")
def _build_get_block_num(self, *, span: Span | None = None):
    return _ir_core.create_op_call("get_block_num", [], {}, span)


@op_impl("get_subblock_num")
def _build_get_subblock_num(self, *, span: Span | None = None):
    return _ir_core.create_op_call("get_subblock_num", [], {}, span)


@op_impl("get_spr")
def _build_get_spr(self, *, span: Span | None = None):
    return _ir_core.create_op_call("get_spr", [], {}, span)


@op_impl("get_cycle")
def _build_get_cycle(self, *, span: Span | None = None):
    return _ir_core.create_op_call("get_cycle", [], {}, span)


@op_impl("system.bar_m")
def _build_bar_m(self, *, span: Span):
    return bar_m(span=span)


@op_impl("system.bar_mte1")
def _build_bar_mte1(self, *, span: Span):
    return bar_mte1(span=span)


@op_impl("system.bar_mte2")
def _build_bar_mte2(self, *, span: Span):
    return bar_mte2(span=span)


@op_impl("system.bar_mte3")
def _build_bar_mte3(self, *, span: Span):
    return bar_mte3(span=span)


@op_impl("system.bar_fix")
def _build_bar_fix(self, *, span: Span):
    return bar_fix(span=span)


@op_impl("system.bar_all")
def _build_bar_all(self, *, span: Span):
    return bar_all(span=span)


@op_impl("system.set_cross_core")
def _build_set_cross_core(self, *args, span: Span, **kwargs):
    return set_cross_core(*args, span=span, **kwargs)


@op_impl("system.wait_cross_core")
def _build_wait_cross_core(self, *args, span: Span, **kwargs):
    return wait_cross_core(*args, span=span, **kwargs)


@op_impl("system.sync_all")
def _build_sync_all(self, *args, span: Span, **kwargs):
    return sync_all(*args, span=span, **kwargs)


@op_impl("system.dcci")
def _build_dcci(self, *args, span: Span, **kwargs):
    return dcci(*args, span=span, **kwargs)


@op_impl("system.dsb")
def _build_dsb(self, *args, span: Span, **kwargs):
    return dsb(*args, span=span, **kwargs)


@op_impl("system.mutex_lock")
def _build_mutex_lock(self, *args, span: Span, **kwargs):
    return mutex_lock(*args, span=span, **kwargs)


@op_impl("system.mutex_unlock")
def _build_mutex_unlock(self, *args, span: Span, **kwargs):
    return mutex_unlock(*args, span=span, **kwargs)
