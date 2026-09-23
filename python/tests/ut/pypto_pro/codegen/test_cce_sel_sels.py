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
"""CCE codegen tests for pl.select (TSEL / TSELS).

pto-isa intrinsics (include/pto/common/pto_instr.hpp):
- TSEL(dst, mask, src0, src1, tmp)        : dst = src0 if mask else src1
- TSELS(dst, mask, src, tmp, scalar)      : dst = src  if mask else scalar

pl.select with a scalar rhs follows the pto-isa TSELS form and is CCE-only (the PTO backend's
pto.tsels is a different, incompatible op, so it raises a clear error).
"""

import logging

from pypto_pro import ir
import pypto_pro.language as pl


def _compile_to_cce(kernel) -> str:
    from pypto_pro.runtime.jit import _assemble_cv_source, _parse_and_codegen_targets

    cube, vector = _parse_and_codegen_targets(kernel.to_kernel_def(), "3510", "")
    return _assemble_cv_source(cube, vector).content


def _parse_select(kernel, op_name):
    kernel_def = kernel.to_kernel_def()
    program, _ = kernel_def.parse_target_program(ir.SectionKind.Vector)
    stmts = program.get_function(kernel_def.func_name).body.stmts
    tiles = [stmt.var for stmt in stmts if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.var.type, ir.TileType)]
    calls = [
        stmt.expr
        for stmt in stmts
        if isinstance(stmt, ir.EvalStmt) and isinstance(stmt.expr, ir.Call) and stmt.expr.name == op_name
    ]
    assert len(calls) == 1
    return tiles, calls[0]


@pl.jit
def _sel_kernel(x: pl.Tensor[[64, 128], pl.DT_FP16]):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    lhs = pl.make_tile(tt, addr=0x0000)
    rhs = pl.make_tile(tt, addr=0x4000)
    tmp = pl.make_tile(tt, addr=0x8000)
    out = pl.make_tile(tt, addr=0xC000)
    mask_t = pl.TileType(shape=[64, 32], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec)
    mask = pl.make_tile(mask_t, addr=0x10000)
    pl.load(lhs, x, [0, 0])
    pl.select(out, mask, lhs, rhs, tmp)


@pl.jit
def _sels_kernel(x: pl.Tensor[[64, 128], pl.DT_FP16]):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    src = pl.make_tile(tt, addr=0x0000)
    tmp = pl.make_tile(tt, addr=0x4000)
    out = pl.make_tile(tt, addr=0x8000)
    mask_t = pl.TileType(shape=[64, 32], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec)
    mask = pl.make_tile(mask_t, addr=0xC000)
    pl.load(src, x, [0, 0])
    pl.select(out, mask, src, 0.0, tmp)


@pl.jit(auto_mutex=True)
def _sel_auto_mutex_kernel(
    x: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    mask_tt = pl.TileType(shape=[64, 32], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec)
    out_group = pl.make_tile_group(type=tt, addrs=0xC000, mutex_ids=[0])
    mask_group = pl.make_tile_group(type=mask_tt, addrs=0x10000, mutex_ids=[1])
    lhs_group = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[2])
    rhs_group = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[3])
    tmp_group = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[4])
    out = out_group.current()
    mask = mask_group.current()
    lhs = lhs_group.current()
    rhs = rhs_group.current()
    tmp = tmp_group.current()
    pl.select(out, mask, lhs, rhs, tmp)
    pl.store(x, out, [0, 0])


def test_cce_sel_emits_tsel():
    (lhs, rhs, tmp, out, mask), call = _parse_select(_sel_kernel, "block.sel")
    # The tile-operand order the TSEL below is generated from.
    assert len(call.args) == 5
    assert all(actual.same_as(expected) for actual, expected in zip(call.args, [out, mask, lhs, rhs, tmp]))

    cpp = _compile_to_cce(_sel_kernel)
    logging.info("\n=== test_cce_sel ===\n%s", cpp)
    assert "TSEL(out_0, mask_0, lhs_0, rhs_0, tmp_0);" in cpp


def test_cce_sels_emits_tsels():
    (src, tmp, out, mask), call = _parse_select(_sels_kernel, "block.sels")
    # The scalar rhs makes this sels rather than sel, and the IR keeps 0.0 a float.
    assert len(call.args) == 5
    assert all(actual.same_as(expected) for actual, expected in zip(call.args[:4], [out, mask, src, tmp]))
    assert isinstance(call.args[4], ir.ConstFloat)
    assert call.args[4].value == 0.0

    cpp = _compile_to_cce(_sels_kernel)
    logging.info("\n=== test_cce_sels ===\n%s", cpp)
    # TSELS(dst, mask, src, tmp, scalar) — pto-isa form.
    assert "TSELS(out_0, mask_0, src_0, tmp_0, " in cpp, "Expected TSELS(dst, mask, src, tmp, scalar)"
    # The scalar operand (0.0) is the last argument. CCE codegen emits float
    # literals with the C++ "f" suffix for correct float type, and keeps the ".0"
    # so an integral value still reads as a float rather than an int.
    tsels_line = next(line for line in cpp.splitlines() if "TSELS(" in line)
    assert tsels_line.rstrip().endswith("0.0f);"), f"scalar should be last arg: {tsels_line}"
