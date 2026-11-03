#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parse-level dispatch contracts of the ``pld.*`` distributed ops (CPU-only).

Every dist op dispatches through the uniform ``@op_impl`` registry except
``pld.channel``, whose let-binding defers to the assignment parser. These
tests pin what that dispatch style must preserve: parse-time handles
(DTensor/Signal) delivered as ordinary operands, offset lists parsed as tuple
operands, enum and literal kwargs, explicit ``None`` defaults, view-dtyped
atomic value literals, and the diagnostics raised for malformed calls.
"""

from pypto_pro import ir
from pypto_pro._errors import InvalidArgument, InvalidOperation, InvalidShape, InvalidType, InvalidVal
import pypto_pro.language as pl
import pypto_pro.language.distribute as pld
from pypto_pro.runtime.platform import NpuArch
import pytest


def _parse(kernel, target=ir.SectionKind.Vector):
    program, _ = kernel.to_kernel_def().parse_target_program(target)
    return program.get_function(kernel.__name__)


def _op_calls(func, op_name):
    """Every IR call to *op_name* in the parsed function body, in program order."""
    calls = []
    for stmt in func.body.stmts:
        value = getattr(stmt, "value", None) if isinstance(stmt, ir.AssignStmt) else getattr(stmt, "expr", None)
        if value is not None and isinstance(value, ir.Call) and value.name == op_name:
            calls.append(value)
    return calls


# =============================================================================
# barrier — the uniform-path reference op
# =============================================================================


def test_barrier_defaults_and_static_attrs():
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            pld.barrier(win)
            pld.barrier(win, sync_local=False, flush_store=False)

    calls = _op_calls(_parse(kernel), "dist.barrier")
    assert len(calls) == 2
    assert (calls[0].kwargs["timeout_cycles"], calls[0].kwargs["sync_local"], calls[0].kwargs["flush_store"]) == (
        0, True, True)
    assert (calls[1].kwargs["sync_local"], calls[1].kwargs["flush_store"]) == (False, False)


def test_barrier_literal_timeout_by_position_and_keyword():
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            pld.barrier(win, 1000)
            pld.barrier(win, timeout=2000, sync_local=False, flush_store=True)

    calls = _op_calls(_parse(kernel), "dist.barrier")
    assert [c.kwargs["timeout_cycles"] for c in calls] == [1000, 2000]
    assert (calls[1].kwargs["sync_local"], calls[1].kwargs["flush_store"]) == (False, True)


def test_barrier_explicit_timeout_none_waits_forever():
    """``timeout=None`` is the documented default and must parse like omission."""

    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            pld.barrier(win, timeout=None)

    call = _op_calls(_parse(kernel), "dist.barrier")[0]
    assert call.kwargs["timeout_cycles"] == 0


def test_barrier_timeout_must_be_compile_time():
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window, i: pl.Tensor[[1], pl.DT_INT64]):
        with pl.section_vector():
            pld.barrier(win, i[0])

    with pytest.raises(InvalidType) as caught:
        _parse(kernel)
    assert "timeout" in str(caught.value)
    assert "compile-time" in str(caught.value)


# =============================================================================
# make_dtensor / signal — parse-time handles from uniform operands
# =============================================================================


def test_make_dtensor_shape_must_be_positive_and_bounded():
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def zero_dim(win: pld.Window):
        with pl.section_vector():
            pld.make_dtensor(win, 0, [0, 8], pl.DT_INT32)

    with pytest.raises(InvalidVal) as caught:
        _parse(zero_dim)
    assert "must have 1-5 positive dims" in str(caught.value)

    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def six_dims(win: pld.Window):
        with pl.section_vector():
            pld.make_dtensor(win, 0, [1, 8, 2, 4, 8, 16], pl.DT_INT32)

    with pytest.raises(InvalidVal) as caught:
        _parse(six_dims)
    assert "must have 1-5 positive dims" in str(caught.value)


def _dtensor_kernel(shape):
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            pld.make_dtensor(win, 0, shape, pl.DT_INT32)

    return kernel


def test_make_dtensor_rejects_coordinate_offset():
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            pld.make_dtensor(win, [0, 0], [1, 8], pl.DT_INT32)

    with pytest.raises(InvalidType) as caught:
        _parse(kernel)
    assert "offset is a scalar byte offset" in str(caught.value)


def test_make_dtensor_negative_offset_is_rejected():
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            pld.make_dtensor(win, -64, [1, 8], pl.DT_INT32)

    with pytest.raises(InvalidVal) as caught:
        _parse(kernel)
    assert "offset must be >= 0 bytes" in str(caught.value)


def test_signal_rejects_negative_literal():
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            pld.signal(win, -1)

    with pytest.raises(InvalidVal) as caught:
        _parse(kernel)
    assert "idx must be >= 0" in str(caught.value)


# =============================================================================
# get / put / get_tile / put_tile — tuple-operand offsets
# =============================================================================


def test_transfer_ops_parse():
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            tt = pl.TileType(shape=[1, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
            t = pl.make_tile(tt, addr=0)
            t2 = pl.make_tile(tt, addr=32)
            dt = pld.make_dtensor(win, 0, [1, 8], pl.DT_INT32)
            pld.put(dt, t, [0, 0], 1)
            pld.get(t2, dt, [0, 0], 1)
            pld.put_tile(dt, t, [0, 0], 1)
            pld.get_tile(t2, dt, [0, 0], 1)
            pld.get(t2, dt, [0, 0], 1, order=[0, 1])

    func = _parse(kernel)
    assert len(_op_calls(func, "dist.put")) == 2
    assert len(_op_calls(func, "dist.get")) == 3


def test_transfer_offsets_may_come_from_a_variable():
    """Parity with pl.load: offsets are operands, not syntax."""

    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            tt = pl.TileType(shape=[1, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
            t = pl.make_tile(tt, addr=0)
            dt = pld.make_dtensor(win, 0, [1, 8], pl.DT_INT32)
            off = [0, 0]
            pld.put(dt, t, off, 1)

    assert _op_calls(_parse(kernel), "dist.put")


def test_put_tile_block_indices_lower_to_scaled_offsets():
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            tt = pl.TileType(shape=[1, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
            t = pl.make_tile(tt, addr=0)
            dt = pld.make_dtensor(win, 0, [1, 8], pl.DT_INT32)
            pld.put_tile(dt, t, [2, 3], 1)

    call = _op_calls(_parse(kernel), "dist.put")[0]
    offsets = call.args[2].elements
    # axis 0 maps onto tile rows (extent 1), axis 1 onto cols (extent 8): each block
    # index lowers to an offset expression scaled by its axis extent.
    assert [type(e).__name__ for e in offsets] == ["Mul", "Mul"]


def test_get_offset_rank_must_match_view():
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            tt = pl.TileType(shape=[1, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
            t = pl.make_tile(tt, addr=0)
            dt = pld.make_dtensor(win, 0, [1, 8], pl.DT_INT32)
            pld.get(t, dt, [0], 1)

    with pytest.raises(InvalidArgument) as caught:
        _parse(kernel)
    assert "has 1 coordinates, but the DTensor view has 2 axes" in str(caught.value)


def test_get_static_offset_bounds_check():
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            tt = pl.TileType(shape=[1, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
            t = pl.make_tile(tt, addr=0)
            dt = pld.make_dtensor(win, 0, [1, 8], pl.DT_INT32)
            pld.get(t, dt, [0, 99], 1)

    with pytest.raises(InvalidShape) as caught:
        _parse(kernel)
    assert "outside the DTensor view shape" in str(caught.value)


def test_get_order_arity_is_checked():
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            tt = pl.TileType(shape=[1, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
            t = pl.make_tile(tt, addr=0)
            dt = pld.make_dtensor(win, 0, [1, 8], pl.DT_INT32)
            pld.get(t, dt, [0, 0], 1, order=[1])

    with pytest.raises(InvalidShape) as caught:
        _parse(kernel)
    assert "one axis per Tile dimension" in str(caught.value)


def test_put_atomic_mode_lowers_to_the_atomic_kwarg():
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            tt = pl.TileType(shape=[1, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
            t = pl.make_tile(tt, addr=0)
            dt = pld.make_dtensor(win, 0, [1, 8], pl.DT_INT32)
            pld.put(dt, t, [0, 0], 1, atomic_mode=pl.AtomicType.AtomicAdd)
            pld.put(dt, t, [0, 0], 1)  # default: overwrite, no atomic attribute

    calls = _op_calls(_parse(kernel), "dist.put")
    assert len(calls) == 2
    assert calls[0].kwargs["atomic"] == int(pl.AtomicType.AtomicAdd)
    assert "atomic" not in calls[1].kwargs


def test_put_tile_accepts_atomic_mode_too():
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            tt = pl.TileType(shape=[1, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
            t = pl.make_tile(tt, addr=0)
            dt = pld.make_dtensor(win, 0, [1, 8], pl.DT_INT32)
            pld.put_tile(dt, t, [0, 0], 1, atomic_mode=pl.AtomicType.AtomicAdd)

    call = _op_calls(_parse(kernel), "dist.put")[0]
    assert call.kwargs["atomic"] == int(pl.AtomicType.AtomicAdd)


def test_put_atomic_mode_must_be_an_atomic_type_member():
    """A raw int (parsed as ConstInt) is rejected; only the enum member lowers."""

    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            tt = pl.TileType(shape=[1, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
            t = pl.make_tile(tt, addr=0)
            dt = pld.make_dtensor(win, 0, [1, 8], pl.DT_INT32)
            pld.put(dt, t, [0, 0], 1, atomic_mode=1)

    with pytest.raises(InvalidType) as caught:
        _parse(kernel)
    assert "atomic_mode must be a pl.AtomicType member" in str(caught.value)


def test_put_atomic_add_rejects_unsupported_dtype():
    """The MTE3 atomic-add engine covers 8/16/32-bit ints and fp16/bf16/fp32, not 64-bit."""

    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            tt = pl.TileType(shape=[1, 8], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec)
            t = pl.make_tile(tt, addr=0)
            dt = pld.make_dtensor(win, 0, [1, 8], pl.DT_INT64)
            pld.put(dt, t, [0, 0], 1, atomic_mode=pl.AtomicType.AtomicAdd)

    with pytest.raises(InvalidType) as caught:
        _parse(kernel)
    assert "atomic_mode=AtomicAdd does not support dtype" in str(caught.value)


def test_get_has_no_atomic_mode():
    """A load cannot accumulate; the kwarg is put-only and the gate rejects it on get."""

    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            tt = pl.TileType(shape=[1, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
            t = pl.make_tile(tt, addr=0)
            dt = pld.make_dtensor(win, 0, [1, 8], pl.DT_INT32)
            pld.get(t, dt, [0, 0], 1, atomic_mode=pl.AtomicType.AtomicAdd)

    with pytest.raises(InvalidArgument) as caught:
        _parse(kernel)
    assert "atomic_mode" in str(caught.value)


# =============================================================================
# notify / wait — enum and timeout kwargs, local notify form
# =============================================================================


def test_notify_remote_local_and_enum_forms():
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            sig = pld.signal(win, 0)
            pld.notify(sig, 1, 7)
            pld.notify(sig, 7)
            pld.notify(sig, 1, 7, op=pld.NotifyOp.ATOMIC_ADD)

    calls = _op_calls(_parse(kernel), "dist.notify")
    assert len(calls) == 3
    # Local form reads this rank from the window descriptor at runtime.
    assert calls[1].args[3].name == "dist.window_get"
    assert calls[2].kwargs["notify_op"] == int(pld.NotifyOp.ATOMIC_ADD)
    assert set(calls[0].kwargs) == {"notify_op"}


def test_notify_explicit_val_none_is_the_local_form():
    """``val=None`` follows the declared binding: the call is the local two-argument form."""

    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            sig = pld.signal(win, 0)
            pld.notify(sig, 1, None)

    call = _op_calls(_parse(kernel), "dist.notify")[0]
    assert call.args[3].name == "dist.window_get"


def test_notify_requires_a_signal_handle():
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            pld.notify(3, 1, 1)

    with pytest.raises(InvalidType) as caught:
        _parse(kernel)
    assert "must be a pld.Signal" in str(caught.value)


def test_notify_all_lowers_with_the_set_default():
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            sig = pld.signal(win, 3)
            pld.notify_all(sig, 7)

    call = _op_calls(_parse(kernel), "dist.notify_all")[0]
    assert isinstance(call.args[1], ir.ConstInt) and call.args[1].value == 3
    assert isinstance(call.args[2], ir.ConstInt) and call.args[2].value == 7
    assert call.kwargs == {"notify_op": int(pld.NotifyOp.SET)}


def test_notify_all_atomic_add_op_mode():
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            sig = pld.signal(win, 3)
            pld.notify_all(sig, 1, op=pld.NotifyOp.ATOMIC_ADD)

    call = _op_calls(_parse(kernel), "dist.notify_all")[0]
    assert call.kwargs == {"notify_op": int(pld.NotifyOp.ATOMIC_ADD)}


def test_notify_all_requires_a_signal_handle():
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            pld.notify_all(win, 1)

    with pytest.raises(InvalidType) as caught:
        _parse(kernel)
    assert "must be a pld.Signal" in str(caught.value)


def test_notify_all_rejects_an_unknown_op_mode():
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            sig = pld.signal(win, 3)
            pld.notify_all(sig, 7, op=3)

    with pytest.raises(InvalidVal) as caught:
        _parse(kernel)
    assert "op must be NotifyOp.SET or NotifyOp.ATOMIC_ADD" in str(caught.value)


def test_wait_explicit_timeout_none_waits_forever():
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            sig = pld.signal(win, 0)
            pld.wait(sig, 1, cmp=pld.Cmp.GE, timeout=None)

    call = _op_calls(_parse(kernel), "dist.wait")[0]
    assert call.kwargs["timeout_cycles"] == 0
    assert call.kwargs["cmp"] == int(pld.Cmp.GE)


def test_wait_cmp_accepts_variable_bound_enum():
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            sig = pld.signal(win, 0)
            mode = pld.Cmp.GE
            pld.wait(sig, 1, cmp=mode)

    call = _op_calls(_parse(kernel), "dist.wait")[0]
    assert call.kwargs["cmp"] == int(pld.Cmp.GE)


# =============================================================================
# atomics — operand forms and value literals
# =============================================================================


def test_atomic_value_operand_keeps_the_parsed_form():
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            dt = pld.make_dtensor(win, 0, [1, 8], pl.DT_INT32)
            pld.atomic_add(dt, 1, [0, 0], 5)

    call = _op_calls(_parse(kernel), "dist.atomic_add")[0]
    value = call.args[5]
    assert isinstance(value, ir.ConstInt)
    assert int(value.value) == 5


def test_atomic_sub_negates_in_the_view_dtype():
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            dt = pld.make_dtensor(win, 0, [1, 8], pl.DT_INT32)
            pld.atomic_sub(dt, 1, [0, 0], 5)

    call = _op_calls(_parse(kernel), "dist.atomic_add")[0]
    value = call.args[5]
    assert isinstance(value, ir.ConstInt)
    assert value.type.dtype == pl.DT_INT32
    assert int(value.value) == -5


def test_atomic_unsigned_sub_wraps_modulo_2n():
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            dt = pld.make_dtensor(win, 0, [1, 8], pl.DT_UINT32)
            pld.atomic_sub(dt, 1, [0, 0], 5)

    call = _op_calls(_parse(kernel), "dist.atomic_add")[0]
    value = call.args[5]
    assert value.type.dtype == pl.DT_UINT32
    assert int(value.value) == (-5) % (1 << 32)


def test_atomic_cas_expect_and_desired_operands():
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            dt = pld.make_dtensor(win, 0, [1, 8], pl.DT_INT32)
            pld.atomic_cas(dt, 1, [0, 0], 3, 7)

    call = _op_calls(_parse(kernel), "dist.atomic_cas")[0]
    expect, desired = call.args[5], call.args[6]
    assert (int(expect.value), int(desired.value)) == (3, 7)


def test_atomic_scalar_linear_offset_form():
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            dt = pld.make_dtensor(win, 0, [1, 8], pl.DT_INT32)
            pld.atomic_add(dt, 1, 3, 5)

    call = _op_calls(_parse(kernel), "dist.atomic_add")[0]
    assert isinstance(call.args[2], ir.ConstInt)
    assert int(call.args[2].value) == 3


def test_atomic_offsets_may_come_from_a_variable():
    """Parity with pl.load: per-axis offsets are operands, not syntax."""

    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            dt = pld.make_dtensor(win, 0, [1, 8], pl.DT_INT32)
            off = [0, 0]
            pld.atomic_add(dt, 1, off, 5)

    call = _op_calls(_parse(kernel), "dist.atomic_add")[0]
    assert isinstance(call.args[2], ir.MakeTuple)
    assert len(call.args[2].elements) == 2


def test_atomic_rejects_float_value_for_int_view():
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            dt = pld.make_dtensor(win, 0, [1, 8], pl.DT_INT32)
            pld.atomic_add(dt, 1, [0, 0], 1.5)

    with pytest.raises(InvalidType) as caught:
        _parse(kernel)
    assert "must be an integer for dtype" in str(caught.value)


def test_star_args_are_rejected_generically():
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            dt = pld.make_dtensor(win, 0, [1, 8], pl.DT_INT32)
            vals = [1, 5]
            pld.atomic_add(dt, 1, [0, 0], *vals)

    with pytest.raises(InvalidArgument) as caught:
        _parse(kernel)
    assert "does not support argument unpacking" in str(caught.value)


# =============================================================================
# channel — the AST-registered exception (lvalue let-binding)
# =============================================================================


def test_channel_lifecycle_parses():
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window, src: pl.Tensor[[1, 8], pl.DT_INT32]):
        with pl.section_vector():
            dt = pld.make_dtensor(win, 0, [1, 8], pl.DT_INT32)
            sig = pld.signal(win, 0)
            ch = pld.channel(win, 64, peer=1)
            ch.write(dt, src, submit=False)
            ch.submit()
            ch.drain()
            pld.notify(sig, 1, 1)

    func = _parse(kernel)
    assert _op_calls(func, "dist.channel")
    assert _op_calls(func, "dist.channel_write")
    assert _op_calls(func, "dist.channel_submit")
    assert _op_calls(func, "dist.channel_drain")


def test_channel_requires_assignment():
    """A channel must be the RHS of a simple assignment — the let-bound variable name is
    part of its lowering. The peer= spelling here keeps the negative example separate from
    the multi-issuer form: ``pld.channel(win, buffer)`` without ``peer`` is itself legal
    (clos meshes), only the missing assignment is rejected."""

    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            pld.channel(win, 64, peer=1)

    with pytest.raises(InvalidOperation) as caught:
        _parse(kernel)
    assert "must be assigned to a variable" in str(caught.value)


def test_channel_multi_issuer_spelling_omits_peer():
    """``pld.channel(win, buffer)`` without ``peer`` is the multi-issuer spelling: the
    channel rides the calling AIV's own jetty and the device constructor receives
    ``peer == -1``."""

    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            ch = pld.channel(win, 64)
            ch.is_exclusive()

    calls = _op_calls(_parse(kernel), "dist.channel")
    assert len(calls) == 1
    assert isinstance(calls[0].args[1], ir.ConstInt) and calls[0].args[1].value == -1
    assert calls[0].kwargs["ub_addr"] == 64


def test_channel_buffer_alignment_is_checked():
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            _ = pld.channel(win, 65)

    with pytest.raises(InvalidVal) as caught:
        _parse(kernel)
    assert "64-byte aligned" in str(caught.value)
