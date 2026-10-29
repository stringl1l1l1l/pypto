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
"""Unit tests for the make_tile_group DSL composite (IR named tuple) and auto_mutex.

``pl.make_tile_group(type=, addrs=, mutex_ids=, depth=)`` is a DSL composite: declared inside a
kernel, lowered by the parser to an IR named tuple ``{tiles, cursor}``.
``group.next()/current()/previous()`` each return a bare tile (not a slot); ``next()``
advances the cursor, ``current()/previous()`` do not. ``group[i]`` addresses a slot
directly and never touches the cursor. Explicit indices must be in ``[0, depth)``;
callers spell any desired wraparound as ``group[index % depth]``. The tile binding is an explicit make_tile operand
consumed by InsertSync, so the frontend never handles mutex ids or manual lock/unlock.
"""



from pypto_pro._errors import InvalidShape, InvalidType, InvalidVal, NameNotFound, OutOfRange, PyptoProError
import pypto_pro.language as pl
from pypto_pro.runtime.platform import NpuArch
import pytest

from pypto.pypto_impl import ir

_GLOBAL_ADDR_BASE = 0x1000
_GLOBAL_ADDR_STRIDE = 0x100
_GLOBAL_ADDR_COUNT = 3


def _ir_to_str(prog: ir.Program) -> str:
    return str(prog)


def _parse_kernel(kernel_def, target=ir.SectionKind.Vector) -> ir.Program:
    """Build IR and run InsertSync for tests that request automatic synchronization."""
    definition = kernel_def.to_kernel_def()
    program = definition.parse_target_program(target)[0]
    return ir.Pass.insert_sync()(program) if definition._auto_mutex else program


def test_make_tile_group_accepts_parsed_pipeline_ids():
    @pl.jit(auto_mutex=False)
    def kernel(entry: pl.DT_INT64):
        tile_type = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        group = pl.make_tile_group(
            type=tile_type, addrs=0, depth=1, fwd_ids=[entry], bwd_ids=(entry,),
        )
        with pl.section_vector():
            pl.expands(group[0], 0)

    assert "block.make_mutex_tile" in _ir_to_str(_parse_kernel(kernel))


@pytest.mark.parametrize(
    "fwd_ids,bwd_ids,invalid_parameter",
    [(1, [0], "fwd_ids"), ([0], 1, "bwd_ids")],
)
def test_make_tile_group_checks_pipeline_ids_types(fwd_ids, bwd_ids, invalid_parameter):
    @pl.jit(auto_mutex=False)
    def kernel(_entry: pl.DT_INT64):
        tile_type = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        group = pl.make_tile_group(
            type=tile_type, addrs=0, depth=1, fwd_ids=fwd_ids, bwd_ids=bwd_ids,
        )
        with pl.section_vector():
            pl.expands(group[0], 0)

    with pytest.raises(InvalidType, match=f"{invalid_parameter} expects"):
        _parse_kernel(kernel)


def _walk_statements(stmt):
    yield stmt
    if isinstance(stmt, ir.SeqStmts):
        for child in stmt.stmts:
            yield from _walk_statements(child)
    elif isinstance(stmt, (ir.ForStmt, ir.WhileStmt, ir.SectionStmt)):
        yield from _walk_statements(stmt.body)
    elif isinstance(stmt, ir.IfStmt):
        yield from _walk_statements(stmt.then_body)
        if stmt.else_body is not None:
            yield from _walk_statements(stmt.else_body)


def _mutex_calls(program: ir.Program, function_name: str):
    function = program.get_function(function_name)
    return [
        stmt.expr
        for stmt in _walk_statements(function.body)
        if isinstance(stmt, ir.EvalStmt)
        and isinstance(stmt.expr, ir.Call)
        and stmt.expr.name in ("system.mutex_lock_dyn", "system.mutex_unlock_dyn")
    ]


def test_constant_mutex_lock_unlock_uses_dynamic_ir():
    @pl.jit
    def k(x: pl.Tensor[[1, 64], pl.DT_FP16]):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        t = pl.make_tile(tt, addr=0)
        pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=0)
        pl.load(t, x, [0, 0])
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=0)

    ir_str = _ir_to_str(_parse_kernel(k))
    assert "system.mutex_lock_dyn" in ir_str
    assert "system.mutex_unlock_dyn" in ir_str


def test_keyword_form():
    @pl.jit
    def k(x: pl.Tensor[[1, 64], pl.DT_FP16]):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        t = pl.make_tile(tt, addr=0)
        pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=1)
        pl.load(t, x, [0, 0])
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=1)

    ir_str = _ir_to_str(_parse_kernel(k))
    assert "system.mutex_lock_dyn" in ir_str
    assert "system.mutex_unlock_dyn" in ir_str


def test_contiguous_addr_auto_offset():
    @pl.jit(auto_mutex=True)
    def k(a: pl.Tensor[[128, 128], pl.DT_FP16]):
        tt = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        db = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
        cur0 = db.next()
        pl.load(cur0, a, [0, 0])
        cur1 = db.next()
        pl.load(cur1, a, [0, 0])

    ir_str = _ir_to_str(_parse_kernel(k))
    # 128*128*2 bytes = 32768 -> second tile at base + 32768.
    assert "memref_addr=" in ir_str
    assert "32768" in ir_str


def test_discrete_addrs_list():
    @pl.jit(auto_mutex=True)
    def k(a: pl.Tensor[[128, 128], pl.DT_FP16]):
        tt = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        db = pl.make_tile_group(type=tt, addrs=[0, 0x10000], mutex_ids=[0, 1])
        cur0 = db.next()
        pl.load(cur0, a, [0, 0])
        cur1 = db.next()
        pl.load(cur1, a, [0, 0])

    ir_str = _ir_to_str(_parse_kernel(k))
    assert "memref_addr=" in ir_str
    assert "65536" in ir_str


def test_discrete_addrs_list_comprehension_from_global_constants():
    @pl.jit(auto_mutex=True)
    def k(a: pl.Tensor[[1, 64], pl.DT_FP16]):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        db = pl.make_tile_group(
            type=tt,
            addrs=[
                _GLOBAL_ADDR_BASE + i * _GLOBAL_ADDR_STRIDE
                for i in range(_GLOBAL_ADDR_COUNT)
            ],
            mutex_ids=[0, 1, 2],
        )
        pl.load(db.next(), a, [0, 0])

    ir_str = _ir_to_str(_parse_kernel(k))
    for i in range(_GLOBAL_ADDR_COUNT):
        assert str(_GLOBAL_ADDR_BASE + i * _GLOBAL_ADDR_STRIDE) in ir_str


def test_addrs_from_kernel_local_constant():
    """addrs=/mutex_ids= accept kernel-local constants, not just literals."""

    @pl.jit(auto_mutex=True)
    def k(a: pl.Tensor[[128, 128], pl.DT_FP16]):
        tt = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        base_addr = 0x10000
        ids = [0, 1]
        db = pl.make_tile_group(type=tt, addrs=base_addr + 0x8000, mutex_ids=ids)
        cur0 = db.next()
        pl.load(cur0, a, [0, 0])
        cur1 = db.next()
        pl.load(cur1, a, [0, 0])

    ir_str = _ir_to_str(_parse_kernel(k))
    assert "98304" in ir_str  # 0x18000
    assert "131072" in ir_str  # 0x18000 + 128*128*2


def test_discrete_addrs_list_from_kernel_local_constant():
    @pl.jit(auto_mutex=True)
    def k(a: pl.Tensor[[128, 128], pl.DT_FP16]):
        tt = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        base_addr = 0x10000
        addrs = [base_addr, base_addr * 2]
        db = pl.make_tile_group(type=tt, addrs=addrs, mutex_ids=[0, 1])
        cur0 = db.next()
        pl.load(cur0, a, [0, 0])
        cur1 = db.next()
        pl.load(cur1, a, [0, 0])

    ir_str = _ir_to_str(_parse_kernel(k))
    assert "65536" in ir_str
    assert "131072" in ir_str


def test_runtime_addr_rejected_with_compile_time_hint():
    @pl.jit(auto_mutex=True)
    def k(a: pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP16]):
        tt = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        addr = a.shape[0] * 2
        db = pl.make_tile_group(type=tt, addrs=addr, mutex_ids=[0, 1])
        cur0 = db.next()
        pl.load(cur0, a, [0, 0])

    with pytest.raises(InvalidType) as excinfo:
        _parse_kernel(k)
    assert "runtime value" in str(excinfo.value)


def test_slot_stride_matches_a_standalone_tile_of_the_same_type():
    """A group slot is exactly as wide as pl.make_tile() of the same TileType."""

    @pl.jit(auto_mutex=True)
    def k(a: pl.Tensor[[64, 96], pl.DT_FP32]):
        tt = pl.TileType(shape=[64, 96], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat)
        standalone = pl.make_tile(tt, addr=0x40000)
        pl.load(standalone, a, [0, 0])
        db = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
        pl.load(db.next(), a, [0, 0])
        pl.load(db.next(), a, [0, 0])

    ir_str = _ir_to_str(_parse_kernel(k))
    slot_size = 64 * 96 * 4
    assert ir_str.count(f"memref_size={slot_size}") == 3, ir_str
    # Second slot starts one slot after the base address.
    assert f"memref_addr={slot_size}" in ir_str, ir_str


def test_slot_stride_packs_sub_byte_elements():
    @pl.jit(auto_mutex=True)
    def k(a: pl.Tensor[[64, 64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_INT4, target_memory=pl.MemorySpace.Mat)
        db = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])  # noqa: F841

    ir_str = _ir_to_str(_parse_kernel(k))
    assert ir_str.count(f"memref_size={64 * 64 // 2}") == 2, ir_str
    assert f"memref_addr={64 * 64 // 2}" in ir_str, ir_str


def test_fp4_right_tile_group_fits_two_slots_in_64k():
    @pl.jit(auto_mutex=True)
    def k(a: pl.Tensor[[1, 1], pl.DT_FP16]):
        tt = pl.TileType(shape=[256, 256], dtype=pl.DT_FP4E2M1, target_memory=pl.MemorySpace.Right)
        db = pl.make_tile_group(type=tt, addrs=[0, 0x8000], mutex_ids=[0, 1])  # noqa: F841

    ir_str = _ir_to_str(_parse_kernel(k))
    assert ir_str.count("memref_size=32768") == 2, ir_str
    assert "memref_addr=32768" in ir_str, ir_str


def test_tile_type_with_a_runtime_shape_rejected():
    """The slot stride is a compile-time byte count, so the shape has to be one too."""

    @pl.jit(auto_mutex=True)
    def k(a: pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP16]):
        # Tuple shapes must be checked for compile-time dimensions just like lists.
        tt = pl.TileType(shape=(a.shape[0], 128), dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        db = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
        pl.load(db.next(), a, [0, 0])

    with pytest.raises(InvalidType, match="must be a compile-time integer"):
        _parse_kernel(k)


def test_auto_mutex_single_tile():
    @pl.jit(auto_mutex=True)
    def k(x: pl.Tensor[[1, 64], pl.DT_FP16]):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[3])
        buf = g.next()
        pl.load(buf, x, [0, 0])
        pl.store(x, buf, [0, 0])

    ir_str = _ir_to_str(_parse_kernel(k))
    assert ir_str.count("mutex_lock") >= 2, ir_str
    assert ir_str.count("mutex_unlock") >= 2, ir_str
    assert "system.mutex_lock_dyn(3" in ir_str, ir_str


def test_auto_mutex_ids_from_closure():
    ids = "auto"

    @pl.jit(auto_mutex=True)
    def k(x: pl.Tensor[[1, 64], pl.DT_FP16]):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        group = pl.make_tile_group(type=tt, addrs=0, mutex_ids=ids, depth=1)
        tile = group.next()
        pl.load(tile, x, [0, 0])
        pl.store(x, tile, [0, 0])

    ir_str = _ir_to_str(_parse_kernel(k))
    assert ir_str.count("system.mutex_lock") >= 2
    assert ir_str.count("system.mutex_unlock") >= 2


def test_auto_mutex_locks_scale_kwarg_tile(monkeypatch):
    monkeypatch.setenv("PYPTOPRO_JIT_ARCH", NpuArch.DAV_3510.name)

    @pl.jit(auto_mutex=True)
    def k(out: pl.Tensor[[64, 64], pl.DT_INT8]):
        acc_tt = pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        )
        fp_tt = pl.TileType(shape=[1, 64], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Scaling)
        acc_g = pl.make_tile_group(type=acc_tt, addrs=0, mutex_ids=[0])
        fp_g = pl.make_tile_group(type=fp_tt, addrs=0, mutex_ids=[1])
        pl.store(out, acc_g[0], [0, 0], scale=fp_g[0])

    ir_str = _ir_to_str(_parse_kernel(k, ir.SectionKind.Cube))
    assert "block.store" in ir_str, ir_str

    assert ir_str.count("system.mutex_lock_dyn") == 2, ir_str
    assert ir_str.count("system.mutex_unlock_dyn") == 2, ir_str
    assert "system.mutex_lock_dyn(0" in ir_str
    assert "system.mutex_lock_dyn(1" in ir_str


def test_auto_mutex_group_loop():
    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[128, 32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        q_l1_db = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
        for i in pl.range(0, 4):
            cur = q_l1_db.next()
            pl.load(cur, gm_q, [i * 32, 0])

    ir_str = _ir_to_str(_parse_kernel(k, ir.SectionKind.Cube))
    assert "mutex_lock" in ir_str, ir_str
    assert "mutex_unlock" in ir_str, ir_str
    assert ir_str.count("block.make_mutex_tile") == 2, ir_str
    assert "block.tile_mutex_id" in ir_str, ir_str


def test_next_advances():
    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[32, 32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
        a = g.next()
        pl.load(a, gm_q, [0, 0])
        b = g.next()
        pl.load(b, gm_q, [0, 0])

    ir_str = _ir_to_str(_parse_kernel(k))
    assert ir_str.count("block.make_mutex_tile") == 2, ir_str
    assert "struct.create" in ir_str
    # Each next() advances the cursor via a struct.set field write.
    assert ir_str.count("struct.set") >= 2


def test_current_does_not_advance():
    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[32, 32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
        a = g.current()
        pl.load(a, gm_q, [0, 0])
        b = g.current()
        pl.load(b, gm_q, [0, 0])

    ir_str = _ir_to_str(_parse_kernel(k))
    assert "struct.create" in ir_str
    # current() never advances -> no cursor writes.
    assert "struct.set" not in ir_str


def test_previous_no_advance():
    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[64, 32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
        a = g.next()
        pl.load(a, gm_q, [0, 0])
        prev = g.previous()
        pl.load(prev, gm_q, [0, 0])

    ir_str = _ir_to_str(_parse_kernel(k))
    assert ir_str.count("block.make_mutex_tile") == 2, ir_str
    # Only the next() advanced the cursor; previous() does not.
    assert ir_str.count("struct.set") == 1


def test_single_tile_group_fast_path():
    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[32, 32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[5])
        cur = g.next()
        pl.load(cur, gm_q, [0, 0])

    ir_str = _ir_to_str(_parse_kernel(k, ir.SectionKind.Cube))
    assert "block.make_mutex_tile" in ir_str
    # Single-tile group: no cursor struct or rotate; mutex metadata is retained.
    assert "struct.create" not in ir_str
    assert "struct.set" not in ir_str
    assert "mutex_lock_dyn" in ir_str
    assert "5" in ir_str


def test_depth_creates_group_without_mutex_ids():
    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[32, 32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=None, depth=2)
        pl.load(g[0], gm_q, [0, 0])

    ir_str = _ir_to_str(_parse_kernel(k))
    assert ir_str.count("block.make_mutex_tile") == 2, ir_str
    assert "mutex_lock" not in ir_str, ir_str


def test_depth_keyword_with_explicit_none_mutex_ids():
    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[32, 32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=None, depth=2)
        pl.load(g[0], gm_q, [0, 0])

    ir_str = _ir_to_str(_parse_kernel(k))
    assert ir_str.count("block.make_mutex_tile") == 2, ir_str
    assert "mutex_lock" not in ir_str, ir_str


def test_depth_must_equal_mutex_ids_length():
    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[32, 32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1], depth=3)
        pl.load(g[0], gm_q, [0, 0])

    with pytest.raises(InvalidShape, match="mutex_ids length 2 must equal depth 3"):
        _parse_kernel(k)


# ---------------------------------------------------------------------------
# group[i] — direct slot access
# ---------------------------------------------------------------------------


def test_subscript_const_index_uses_the_selected_binding():
    """A literal index resolves to the selected tile's static ID."""

    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[32, 32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[6, 7])
        pl.load(g[0], gm_q, [0, 0])
        pl.load(g[1], gm_q, [0, 0])

    program = _parse_kernel(k, ir.SectionKind.Cube)
    ir_str = _ir_to_str(program)
    assert ir_str.count("block.make_mutex_tile") == 2, ir_str
    assert "_bufidx" not in ir_str, ir_str
    assert ir_str.count("system.mutex_lock_dyn") == 2, ir_str

    locks = [call for call in _mutex_calls(program, k.__name__) if call.name == "system.mutex_lock_dyn"]
    assert [call.args[0].value for call in locks] == [6, 7]
    assert all(len(call.args) == 1 for call in locks)


def test_subscript_negative_index_rejected():
    """Negative indices are outside the supported [0, depth) interval."""

    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[32, 32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[6, 7, 8])
        pl.load(g[-1], gm_q, [0, 0])

    with pytest.raises(OutOfRange, match="out of range"):
        _parse_kernel(k)


def test_subscript_does_not_touch_cursor():
    """Subscript is random access: the rotation state stays where next() left it."""

    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[32, 32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
        a = g.next()
        pl.load(a, gm_q, [0, 0])
        b = g[0]
        pl.load(b, gm_q, [0, 0])
        c = g[1]
        pl.load(c, gm_q, [0, 0])

    ir_str = _ir_to_str(_parse_kernel(k))
    # Only the single next() wrote the cursor; neither subscript did.
    assert ir_str.count("struct.set") == 1, ir_str


def test_subscript_dynamic_index_is_not_implicitly_wrapped():
    """A runtime index is used unchanged and locked through the dyn if-chain."""

    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[128, 32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
        for i in pl.range(0, 2):
            pl.load(g[i], gm_q, [i * 32, 0])

    ir_str = _ir_to_str(_parse_kernel(k, ir.SectionKind.Cube))
    assert "mutex_lock_dyn" in ir_str, ir_str
    assert "mutex_unlock_dyn" in ir_str, ir_str
    assert "_bufidx" in ir_str, ir_str
    assert " % " not in ir_str, ir_str
    # The cursor is never consulted by subscript access.
    assert "struct.set" not in ir_str, ir_str


def test_subscript_dynamic_index_expression():
    """The index may be any scalar expression, e.g. a look-ahead slot."""

    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[128, 32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
        for i in pl.range(0, 3):
            pl.load(g[(i + 1) % 2], gm_q, [i * 32, 0])
            pl.load(g[i % 2], gm_q, [i * 32, 0])

    ir_str = _ir_to_str(_parse_kernel(k, ir.SectionKind.Cube))
    assert "mutex_lock_dyn" in ir_str, ir_str
    assert ir_str.count("_bufidx") >= 2, ir_str
    assert " mod " in ir_str, ir_str


def test_subscript_single_tile_group_requires_bounded_index():
    """A dynamic index for a one-slot group must explicitly reduce to zero."""

    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[128, 32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[5])
        for i in pl.range(0, 4):
            pl.load(g[i % 1], gm_q, [i * 32, 0])

    ir_str = _ir_to_str(_parse_kernel(k, ir.SectionKind.Cube))
    assert "mutex_lock_dyn" in ir_str, ir_str
    assert "_bufidx" in ir_str, ir_str
    assert " mod " in ir_str, ir_str


def test_subscript_two_slots_in_one_op_dedup():
    """InsertSync guards the second scalar lock when two dynamic slots may coincide."""

    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[128, 32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[10, 11])
        for i in pl.range(0, 4):
            pl.move(g[i % 2], g[(i + 1) % 2])

    ir_str = _ir_to_str(_parse_kernel(k))
    assert "mutex_lock_dyn" in ir_str, ir_str
    # Both declarations remain available to the pass; generated sync has no candidates.
    assert "10" in ir_str and "11" in ir_str, ir_str


def test_subscript_mixes_with_next():
    """Subscript and cursor access address the same slots of the same group."""

    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[128, 32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
        for i in pl.range(0, 4):
            cur = g.next()
            pl.load(cur, gm_q, [i * 32, 0])
            pl.load(g[i % 2], gm_q, [i * 32, 0])

    ir_str = _ir_to_str(_parse_kernel(k))
    assert ir_str.count("block.make_mutex_tile") == 2, ir_str
    assert ir_str.count("struct.set") == 1, ir_str  # only next() advances


def test_subscript_const_index_out_of_range():
    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[32, 32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
        pl.load(g[2], gm_q, [0, 0])

    with pytest.raises(OutOfRange, match="out of range"):
        _parse_kernel(k)


def test_subscript_slice_rejected():
    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[32, 32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
        pl.load(g[0:2], gm_q, [0, 0])

    with pytest.raises(InvalidType, match="Unsupported expression type: Slice"):
        _parse_kernel(k)


def test_subscript_multi_dim_index_rejected():
    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[32, 32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
        pl.load(g[0, 1], gm_q, [0, 0])

    with pytest.raises(InvalidType, match="integer scalar"):
        _parse_kernel(k)


# ---------------------------------------------------------------------------
# Control-flow isolation of tile variables
# ---------------------------------------------------------------------------


def test_dynamic_slot_assignment_does_not_leak_from_branch():
    """A branch-local tile assignment does not replace the outer tile binding."""

    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[128, 32], pl.DT_FP16], n: pl.DT_INT32):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
        t = g[0]
        if n > 0:
            t = g.current()
            pl.load(t, gm_q, [32, 0])
        pl.load(t, gm_q, [0, 0])

    ir_str = _ir_to_str(_parse_kernel(k, ir.SectionKind.Cube))
    assert "mutex_lock" in ir_str, ir_str
    assert "mutex_lock_dyn" in ir_str, ir_str


def test_dynamic_slot_assignment_does_not_leak_from_loop():
    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[128, 32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
        t = g[0]
        for i in pl.range(0, 4):
            t = g[i % 2]
            pl.load(t, gm_q, [i * 32, 0])
        pl.load(t, gm_q, [0, 0])

    ir_str = _ir_to_str(_parse_kernel(k, ir.SectionKind.Cube))
    assert "mutex_lock" in ir_str, ir_str
    assert "mutex_lock_dyn" in ir_str, ir_str


def test_const_slot_assignment_merges_from_branch():
    """A reassigned tile and its mutex ID are merged across control flow."""

    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[128, 32], pl.DT_FP16], n: pl.DT_INT32):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
        t = g[0]
        if n > 0:
            t = g[1]
            pl.load(t, gm_q, [32, 0])
        pl.load(t, gm_q, [0, 0])

    ir_str = _ir_to_str(_parse_kernel(k, ir.SectionKind.Cube))
    assert ir_str.count("mutex_lock") == 2, ir_str
    assert "mutex_lock_dyn" in ir_str, ir_str


def test_dynamic_slot_used_inside_its_own_block_is_fine():
    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[128, 32], pl.DT_FP16], n: pl.DT_INT32):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
        if n > 0:
            t = g.current()
            pl.load(t, gm_q, [0, 0])

    ir_str = _ir_to_str(_parse_kernel(k, ir.SectionKind.Cube))
    assert "mutex_lock_dyn" in ir_str, ir_str


def test_dynamic_slot_selected_outside_and_used_inside_a_loop():
    """An enclosing block dominates the loop body, so the index is still in scope."""

    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[128, 32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
        t = g.current()
        for i in pl.range(0, 4):
            pl.load(t, gm_q, [i * 32, 0])

    ir_str = _ir_to_str(_parse_kernel(k, ir.SectionKind.Cube))
    assert "mutex_lock_dyn" in ir_str, ir_str


def test_next_inside_loop_still_accepted():
    """A cursor-selected tile remains usable inside the loop scope that defines it."""

    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[128, 32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
        for i in pl.range(0, 4):
            cur = g.next()
            pl.load(cur, gm_q, [i * 32, 0])

    ir_str = _ir_to_str(_parse_kernel(k, ir.SectionKind.Cube))
    assert "mutex_lock_dyn" in ir_str, ir_str


def test_subscript_inside_inline_helper():
    """
    An inline helper emits into the caller's block, so a tile it returns keeps a usable slot
    index.
    """

    def take(group, idx):
        return group[idx]

    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[128, 32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
        for i in pl.range(0, 4):
            t = take(g, i % 2)
            pl.load(t, gm_q, [i * 32, 0])

    ir_str = _ir_to_str(_parse_kernel(k, ir.SectionKind.Cube))
    assert "mutex_lock_dyn" in ir_str, ir_str


def test_set_validshape_covers_every_tile_of_the_group():
    """
    pl.set_validshape(group, ...) is documented to set all tiles, so it must emit one
    set_validshape per slot, not just for slot 0.
    """

    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[128, 32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1, 2, 3])
        pl.set_validshape(g, [16, 32])
        pl.load(g[0], gm_q, [0, 0])

    ir_str = _ir_to_str(_parse_kernel(k))
    assert ir_str.count("set_validshape") == 4, ir_str


# ---------------------------------------------------------------------------
# Fixed-count per-tile mutex IDs
# ---------------------------------------------------------------------------


def test_const_multi_mutex_ids_expand_to_scalar_sync():
    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[32, 32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[[2, 3]])
        pl.load(g[0], gm_q, [0, 0])

    program = _parse_kernel(k, ir.SectionKind.Cube)
    ir_str = _ir_to_str(program)

    assert ir_str.count("system.mutex_lock_dyn") == 2, ir_str
    assert ir_str.count("system.mutex_unlock_dyn") == 2, ir_str
    calls = _mutex_calls(program, k.__name__)
    assert [call.args[0].value for call in calls] == [2, 3, 2, 3]
    assert all(len(call.args) == 1 for call in calls)


def test_dynamic_multi_mutex_ids_use_fixed_slot_getters():
    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[96, 32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[[0, 1], [2, 3], [4, 5]])
        for i in pl.range(0, 3):
            pl.load(g[i], gm_q, [i * 32, 0])

    program = _parse_kernel(k, ir.SectionKind.Cube)
    ir_str = _ir_to_str(program)

    assert ir_str.count("system.mutex_lock_dyn") == 2, ir_str
    assert ir_str.count("system.mutex_unlock_dyn") == 2, ir_str
    assert ir_str.count("block.tile_mutex_id") == 2, ir_str
    assert "_tg_g_mutex_ids" not in ir_str
    assert "__sync_needed" not in ir_str


def test_single_and_multi_id_groups_can_share_one_op():
    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[96, 32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g1 = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1, 2])
        g2 = pl.make_tile_group(type=tt, addrs=0x2000, mutex_ids=[[0, 1], [2, 3], [4, 5]])
        for i in pl.range(0, 3):
            pl.move(g2[i], g1[i])

    program = _parse_kernel(k)
    ir_str = _ir_to_str(program)

    assert ir_str.count("system.mutex_lock_dyn") == 3, ir_str
    assert ir_str.count("system.mutex_unlock_dyn") == 3, ir_str
    assert "__sync_needed" in ir_str
    assert all(len(call.args) == 1 for call in _mutex_calls(program, k.__name__))


def test_mutex_candidate_groups_preserve_each_tile_candidates():
    @pl.jit(auto_mutex=True)
    def k(index: pl.DT_INT32):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        out_group = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[1])
        lhs_group = pl.make_tile_group(type=tt, addrs=128, mutex_ids=[1, 2])
        rhs_group = pl.make_tile_group(type=tt, addrs=384, mutex_ids=[2])
        pl.add(out_group[0], lhs_group[index % 2], rhs_group[0])

    program = _parse_kernel(k)
    ir_str = _ir_to_str(program)

    assert ir_str.count("system.mutex_lock_dyn") == 3, ir_str
    assert ir_str.count("system.mutex_unlock_dyn") == 3, ir_str
    # Dynamic lhs may equal either static ID; both overlaps need a guard.
    guards = [stmt for stmt in _walk_statements(program.get_function(k.__name__).body)
              if isinstance(stmt, ir.AssignStmt) and stmt.var.name.startswith("__sync_needed")]
    assert len(guards) == 2
    assert all(len(call.args) == 1 for call in _mutex_calls(program, k.__name__))


def test_multi_mutex_ids_survive_ternary_merge():
    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[32, 32], pl.DT_FP16], choose: pl.DT_INT32):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[[0, 1], [2, 3]])
        tile = g[0] if choose > 0 else g[1]
        pl.load(tile, gm_q, [0, 0])

    program = _parse_kernel(k, ir.SectionKind.Cube)
    func = program.get_function(k.__name__)
    if_stmt = next(stmt for stmt in func.body.stmts if isinstance(stmt, ir.IfStmt))
    ir_str = _ir_to_str(program)
    assert len(if_stmt.return_vars) == 1
    assert "__mutexid" not in ir_str
    assert "system.mutex_lock_dyn" in ir_str, ir_str


def test_same_name_control_flow_rejects_different_mutex_id_counts():
    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[32, 32], pl.DT_FP16], choose: pl.DT_INT32):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g1 = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        g2 = pl.make_tile_group(type=tt, addrs=0x2000, mutex_ids=[[1, 2]])
        tile = g1[0]
        if choose > 0:
            tile = g2[0]
        pl.load(tile, gm_q, [0, 0])

    with pytest.raises(
        NameNotFound,
        match="potentially undefined variable",
    ):
        _parse_kernel(k, ir.SectionKind.Cube)


def test_same_name_control_flow_merges_equal_mutex_id_counts():
    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[32, 32], pl.DT_FP16], choose: pl.DT_INT32):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g1 = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[[0, 1]])
        g2 = pl.make_tile_group(type=tt, addrs=0x2000, mutex_ids=[[2, 3]])
        tile = g1[0]
        if choose > 0:
            tile = g2[0]
        pl.load(tile, gm_q, [0, 0])

    ir_str = _ir_to_str(_parse_kernel(k, ir.SectionKind.Cube))
    assert "__mutexid" not in ir_str
    assert ir_str.count("block.tile_mutex_id") == 2
    assert "system.mutex_lock_dyn" in ir_str, ir_str


def test_loop_carries_only_original_merge_slots():
    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[32, 32], pl.DT_FP16], choose: pl.DT_INT32):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        group = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[[0, 1], [2, 3]])
        tile = group[0]
        count = 0
        for _ in pl.range(2):
            if choose > 0:
                tile = group[1]
            count = count + 1
        pl.load(tile, gm_q, [0, 0])

    program = _parse_kernel(k, ir.SectionKind.Cube)
    func = program.get_function(k.__name__)
    for_stmt = next(stmt for stmt in func.body.stmts if isinstance(stmt, ir.ForStmt))


    assert len(for_stmt.iter_args) == len(for_stmt.return_vars) == 2
    assert "__mutexid" not in str(program)
    continue_stmt = for_stmt.body.stmts[-1]
    assert isinstance(continue_stmt, ir.ContinueStmt)
    assert len(continue_stmt.value) == 2
    assert str(program).count("block.tile_mutex_id") == 2


def test_while_carries_only_original_values_on_every_jump():
    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[32, 32], pl.DT_FP16], limit: pl.DT_INT32):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        group = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[[0, 1], [2, 3]])
        other_group = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[[4, 5], [6, 7]])
        tile = group[0]
        index = 0
        while index < limit:
            if index == 1:
                tile = other_group[1]
                break
            if index == 2:
                tile = group[0]
                index = index + 1
                continue
            tile = group[1]
            index = index + 1
        pl.load(tile, gm_q, [0, 0])

    program = _parse_kernel(k, ir.SectionKind.Cube)
    func = program.get_function(k.__name__)
    while_stmt = next(stmt for stmt in func.body.stmts if isinstance(stmt, ir.WhileStmt))

    jumps = []

    def collect_jumps(body):
        for stmt in body.stmts:
            if isinstance(stmt, (ir.BreakStmt, ir.ContinueStmt)):
                jumps.append(stmt)
            elif isinstance(stmt, ir.IfStmt):
                collect_jumps(stmt.then_body)
                if stmt.else_body is not None:
                    collect_jumps(stmt.else_body)

    collect_jumps(while_stmt.body)

    assert len(jumps) >= 3
    assert len(while_stmt.iter_args) == len(while_stmt.return_vars) == 2
    assert "__mutexid" not in str(program)
    assert all(len(jump.value) == 2 for jump in jumps)
    assert str(program).count("block.tile_mutex_id") == 2


def test_loop_rejects_different_mutex_id_counts_for_one_slot():
    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[32, 32], pl.DT_FP16], limit: pl.DT_INT32):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        one_id_group = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        two_id_group = pl.make_tile_group(type=tt, addrs=0x2000, mutex_ids=[[1, 2]])
        tile = one_id_group[0]
        index = 0
        while index < limit:
            if index > 0:
                tile = two_id_group[0]
                break
            index = index + 1
        pl.load(tile, gm_q, [0, 0])

    with pytest.raises(
        NameNotFound,
        match="potentially undefined variable",
    ):
        _parse_kernel(k, ir.SectionKind.Cube)


def test_loop_infers_binding_from_reachable_tile_exit():
    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[32, 32], pl.DT_FP16], limit: pl.DT_INT32):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        group = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[[0, 1]])
        tile = None
        while True:
            if limit == 1:
                tile = group[0]
                break
        pl.load(tile, gm_q, [0, 0])


    # None on the backedge is not a tile binding. The only exit carries a bound tile.
    program = _parse_kernel(k, ir.SectionKind.Cube)
    assert str(program).count("system.mutex_lock_dyn") == 2
    assert "__mutexid" not in str(program)


def test_loop_allows_non_mutex_tile_body_with_scalar_result():
    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[32, 32], pl.DT_FP16], limit: pl.DT_INT32):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        value = pl.make_tile(tt, addr=0)
        while True:
            if limit == 1:
                value = limit  # noqa: F841
                break

    program = _parse_kernel(k)
    func = program.get_function(k.__name__)
    while_stmt = next(stmt for stmt in func.body.stmts if isinstance(stmt, ir.WhileStmt))

    assert len(while_stmt.iter_args) == 1
    assert len(while_stmt.return_vars) == 1
    assert isinstance(while_stmt.iter_args[0].iterVar.type, ir.TileType)
    assert isinstance(while_stmt.return_vars[0].type, ir.ScalarType)


def test_undefined_branch_produces_unknown_type_on_use():
    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[32, 32], pl.DT_FP16], choose: pl.DT_INT32):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        group = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[[0, 1]])
        if choose > 0:
            tile = group[0]
        pl.load(tile, gm_q, [0, 0])

    with pytest.raises(NameNotFound, match="Use of potentially undefined variable"):
        _parse_kernel(k)


def test_ternary_rejects_different_mutex_id_counts():
    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[32, 32], pl.DT_FP16], choose: pl.DT_INT32):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g1 = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        g2 = pl.make_tile_group(type=tt, addrs=0x2000, mutex_ids=[[1, 2]])
        tile = g1[0] if choose > 0 else g2[0]
        pl.load(tile, gm_q, [0, 0])

    with pytest.raises(
        NameNotFound,
        match="potentially undefined variable",
    ):
        _parse_kernel(k, ir.SectionKind.Cube)


def test_make_tile_group_rejects_different_mutex_id_counts():
    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[1, 64], pl.DT_FP16]):
        tt = pl.TileType(shape=[1, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, [1, 2]])

    with pytest.raises(
        PyptoProError,
        match="all tiles in a tile group must have the same mutex ID count: 1 and 2",
    ):
        _parse_kernel(k)


def test_make_tile_group_rejects_duplicate_id_for_one_tile():
    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[1, 32], pl.DT_FP16]):
        tt = pl.TileType(shape=[1, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        pl.make_tile_group(type=tt, addrs=0, mutex_ids=[[2, 2]])

    with pytest.raises(InvalidVal, match="must not contain duplicates"):
        _parse_kernel(k)


@pytest.mark.parametrize("mutex_ids", [[True], [[0, False]]])
def test_make_tile_group_rejects_bool_mutex_id(mutex_ids):
    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        pl.make_tile_group(type=tt, addrs=0, mutex_ids=mutex_ids)

    with pytest.raises(InvalidType, match=r"mutex_ids must be ints, got"):
        _parse_kernel(k)


@pytest.mark.parametrize(
    "mutex_ids",
    [[-1], [32], [[0, -1]], [[0, 32]]],
    ids=["single-negative", "single-too-large", "multi-negative", "multi-too-large"],
)
def test_make_tile_group_rejects_out_of_range_mutex_id(mutex_ids):
    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        pl.make_tile_group(type=tt, addrs=0, mutex_ids=mutex_ids)

    with pytest.raises(OutOfRange, match=r"mutex_ids element must be in \[0, 31\]"):
        _parse_kernel(k)


@pytest.mark.parametrize(
    "mutex_ids,message",
    [(1, "pl.make_tile_group: mutex_ids expects"), ({0, 1}, "Unsupported closure variable type: set")],
    ids=["integer", "set"],
)
def test_make_tile_group_rejects_wrong_mutex_ids_container_type(mutex_ids, message):
    @pl.jit(auto_mutex=True)
    def k(gm_q: pl.Tensor[[32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        pl.make_tile_group(type=tt, addrs=0, mutex_ids=mutex_ids)

    with pytest.raises(InvalidType) as caught:
        _parse_kernel(k)
    assert message in caught.value.message
