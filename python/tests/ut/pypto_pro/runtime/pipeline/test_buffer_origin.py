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

"""Which declaration each operand of an op reaches, as the probe parse records it.

Every buffer the pipeline synchronises is identified through these names: an operand that
reaches no declaration is an access the sync graph never sees — and is warned about, since
the pipeline will insert no sync for it.
"""

import warnings

from pypto_pro._errors import UntrackedTile
import pypto_pro.language as pl
from pypto_pro.language.parser._ast_parser import ASTParser
from pypto_pro.runtime.pipeline._analyzer import probe_kernel_facts
import pytest

from pypto.pypto_impl import ir

TM, K = 64, 128


def _operands(kernel_def, op_name: str) -> list:
    """The buffer names each operand of ``op_name`` reaches, in argument order."""
    facts = probe_kernel_facts(kernel_def.to_kernel_def())
    return [
        [operand.names for operand in record.operands]
        for by_site in facts.op_accesses.values()
        for records in by_site.values()
        for record in records
        if record.op_name == op_name
    ]


def _group(base, first_id):
    return pl.make_tile_group(
        type=pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=[base, base + 0x4000], mutex_ids=[first_id, first_id + 1])


def _hand_back(group):
    return group.next()


@pl.jit(arch="3510", auto_mutex=True, name="origin_every_spelling")
def every_spelling_of_a_slot(src: pl.Tensor[[TM, K], pl.DT_FP16]):
    with pl.section_vector():
        picked = _group(0x0, 0)
        viewed = _group(0x8000, 2)
        viewed_inline = _group(0x10000, 4)
        returned = _group(0x18000, 6)
        a = picked.next()
        pl.muls(a, a, 2.0)
        b = viewed.next()
        v = b[:, :]
        pl.muls(v, v, 2.0)
        c = viewed_inline.next()
        pl.muls(c[:, :], c[:, :], 2.0)
        d = _hand_back(returned)
        pl.muls(d, d, 2.0)


@pl.jit(arch="3510", auto_mutex=True, name="origin_own_tile")
def a_tile_of_its_own(src: pl.Tensor[[TM, K], pl.DT_FP16]):
    with pl.section_vector():
        t = pl.make_tile(
            pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec), addr=0x8000)
        pl.muls(t, t, 2.0)


def test_every_spelling_of_a_slot_reaches_its_group():
    # A window onto a slot is that slot, and a slot keeps its group across a return value:
    # the name an op is recorded under never depends on how the tile was spelled.
    assert _operands(every_spelling_of_a_slot, "muls") == [
        [("picked",), ("picked",)],
        [("viewed",), ("viewed",)],
        [("viewed_inline",), ("viewed_inline",)],
        [("returned",), ("returned",)],
    ]


def test_an_op_on_a_tile_of_its_own_is_not_recorded():
    # No argument reaches a declaration, so the op is left out rather than recorded with
    # empty names: nothing downstream has a buffer to synchronise.
    assert _operands(a_tile_of_its_own, "muls") == []


@pytest.mark.parametrize("auto_pipeline", [False, True])
def test_ops_are_recorded_only_for_the_automatic_pipeline(auto_pipeline):
    # The gate sits at the top of _record_op_access, so the warning below rides on it too:
    # a parse that records nothing has nothing to warn about.
    kd = every_spelling_of_a_slot.to_kernel_def()
    parser = ASTParser(
        kd._source_file, kd._source_lines, ir.SectionKind.Vector, kd._line_offset, kd._col_offset,
        closure_vars=kd._closure_vars, auto_mutex=True, auto_pipeline=auto_pipeline,
        debug_info=ir.IRDebugInfo(),
    )
    parser.parse_function(kd._func_def, func_type=kd._func_type)
    assert any(parser.parse_facts.op_accesses.values()) == auto_pipeline


# ---------------------------------------------------------------------------
# Warned about: a tile no declaration provides.
# ---------------------------------------------------------------------------

NT = 4
_CONFIG = pl.pipeline.PipelineConfig(preload=1)


def _gm(ws):
    return pl.make_tensor(ws, [TM, K], dtype=pl.DT_FP16, depth=2, fwd_ids=[0, 1], bwd_ids=[2, 3])


def _mat():
    return pl.make_tile_group(
        type=pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=[0x0000, 0x4000], mutex_ids=[2, 3])


@pl.pipeline.stage
def scale_in_own_tile(i, src, gm):
    t = pl.make_tile(pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec), addr=0x0000)
    pl.load(t, src, [i * TM, 0])
    pl.store(gm, t, [0, 0])


@pl.pipeline.stage
def scale_in_group(i, src, gm, ub):
    t = ub.next()
    pl.load(t, src, [i * TM, 0])
    pl.store(gm, t, [0, 0])


@pl.pipeline.stage
def take(i, gm, mat):
    m = mat.next()
    pl.load(m, gm, [0, 0])


@pl.jit(arch="3510", auto_mutex=True, pipeline=_CONFIG, name="untracked_make_tile")
def with_make_tile(src: pl.Tensor[[NT * TM, K], pl.DT_FP16], ws: pl.Ptr[pl.DT_UINT8]):
    gm = _gm(ws)
    mat = _mat()
    for i in pl.range(0, NT):
        with pl.section_vector():
            scale_in_own_tile(i, src, gm)
        with pl.section_cube():
            take(i, gm, mat)


@pl.jit(arch="3510", auto_mutex=True, pipeline=_CONFIG, name="untracked_none")
def with_tile_groups_only(src: pl.Tensor[[NT * TM, K], pl.DT_FP16], ws: pl.Ptr[pl.DT_UINT8]):
    gm = _gm(ws)
    mat = _mat()
    ub = pl.make_tile_group(
        type=pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=[0x0000, 0x4000], mutex_ids=[0, 1])
    for i in pl.range(0, NT):
        with pl.section_vector():
            scale_in_group(i, src, gm, ub)
        with pl.section_cube():
            take(i, gm, mat)


def _untracked_warnings(run) -> list:
    """The UntrackedTile warnings ``run()`` raises, shown once per location as by default."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        run()
    return [w for w in caught if issubclass(w.category, UntrackedTile)]


def test_the_probe_warns_once_per_op_handed_an_untracked_tile():
    found = _untracked_warnings(lambda: probe_kernel_facts(with_make_tile.to_kernel_def()))
    # pl.load and pl.store each take the make_tile tile; both targets parse them, and each
    # location is shown once.
    assert sorted(str(w.message).split("`")[1] for w in found) == ["pl.load()", "pl.store()"]
    assert all("argument #" in str(w.message) for w in found)


def test_tiles_from_tile_groups_are_not_warned_about():
    assert _untracked_warnings(lambda: probe_kernel_facts(with_tile_groups_only.to_kernel_def())) == []
