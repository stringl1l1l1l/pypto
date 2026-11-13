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

"""Local tile-group slot lifetime from parsed selectors."""

import warnings

from pipeline_test_common import transform
from pypto_pro._errors import InvalidArgument, InvalidOperation, PyptoProWarning
import pypto_pro.language as pl
import pytest

T, N, DEPTH = 64, 4, 3


@pl.pipeline.stage
def fill(k, a, db):
    t = db.next()
    pl.load(t, a, [k * T, 0])


@pl.pipeline.stage
def mid(k, x, scratch, ub):
    t = ub.next()
    pl.load(t, x, [k * T, 0])
    pl.store(scratch, t, [k * T, 0])


@pl.pipeline.stage
def use_current(k, db, left):
    t = db.current()
    pl.move(left.next(), t)


@pl.pipeline.stage
def use_previous(k, db, left):
    t = db.previous()
    pl.move(left.next(), t)


@pl.pipeline.stage
def use_index(k, db, left):
    t = db[k % DEPTH]
    pl.move(left.next(), t)


@pl.pipeline.stage
def use_dynamic(k, offset, db, left):
    pl.move(left.next(), db[offset % 3])


@pl.pipeline.stage
def use_index_wrong(k, db, left):
    t = db[(k + 1) % DEPTH]
    pl.move(left.next(), t)


def _kernel(depth, preload, use):
    @pl.jit(arch="3510", auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=[preload]))
    def kernel(a: pl.Tensor[[N * T, T], pl.DT_FP32],
               x: pl.Tensor[[N * T, T], pl.DT_FP32],
               scratch: pl.Tensor[[N * T, T], pl.DT_FP32]):
        db = pl.make_tile_group(
            type=pl.TileType(shape=[T, T], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat),
            addrs=[index * 0x4000 for index in range(depth)], mutex_ids=list(range(depth)))
        ub = pl.make_tile_group(
            type=pl.TileType(shape=[T, T], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0, mutex_ids=[10])
        with pl.section_cube():
            left = pl.make_tile_group(
                type=pl.TileType(shape=[T, T], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left),
                addrs=0, mutex_ids=[11])
        for k in pl.range(0, N):
            with pl.section_cube():
                fill(k, a, db)
            with pl.section_vector():
                mid(k, x, scratch, ub)
            with pl.section_cube():
                use(k, db, left)
    return kernel


def test_single_slot_pipeline_is_rejected():
    with pytest.raises(InvalidArgument, match=r"has 1 slot\(s\).*3 slots are live at once"):
        transform(_kernel(1, 2, use_current))


def test_current_at_delay_reads_wrong_generation():
    with pytest.raises(InvalidOperation, match=r"consumer task 0 reads slot 2, which holds generation 2"):
        transform(_kernel(DEPTH, 2, use_current))


def test_explicit_task_index_is_accepted():
    transform(_kernel(DEPTH, 2, use_index))


def test_explicit_wrong_index_is_rejected():
    with pytest.raises(InvalidOperation, match=r"consumer task .* reads slot"):
        transform(_kernel(DEPTH, 2, use_index_wrong))


def test_serial_current_is_accepted():
    transform(_kernel(1, 0, use_current))


def k_dynamic_index():
    @pl.jit(arch="3510", auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=[2]))
    def kernel(a: pl.Tensor[[N * T, T], pl.DT_FP32],
               x: pl.Tensor[[N * T, T], pl.DT_FP32],
               scratch: pl.Tensor[[N * T, T], pl.DT_FP32]):
        db = pl.make_tile_group(
            type=pl.TileType(shape=[T, T], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat),
            addrs=[index * 0x4000 for index in range(DEPTH)], mutex_ids=list(range(DEPTH)))
        ub = pl.make_tile_group(
            type=pl.TileType(shape=[T, T], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0, mutex_ids=[10])
        with pl.section_cube():
            left = pl.make_tile_group(
                type=pl.TileType(shape=[T, T], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left),
                addrs=0, mutex_ids=[11])
        for k in pl.range(0, N):
            offset = k * 2
            with pl.section_cube():
                fill(k, a, db)
            with pl.section_vector():
                mid(k, x, scratch, ub)
            with pl.section_cube():
                use_dynamic(k, offset, db, left)
    return kernel


def test_unresolved_dynamic_index_warns_and_skips():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        transform(k_dynamic_index())
    messages = [str(item.message) for item in caught if item.category is PyptoProWarning]
    assert messages
    assert "skipped local slot lifetime check" in messages[0]
    assert "slot index" in messages[0]


def test_previous_with_gap_two_reads_wrong_generation():
    with pytest.raises(InvalidOperation, match=r"consumer task .* reads slot .* which holds generation"):
        transform(_kernel(DEPTH, 2, use_previous))
