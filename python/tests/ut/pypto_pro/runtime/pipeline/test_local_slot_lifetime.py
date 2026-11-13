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

"""Local tile-group slot depth from the stage delay gap."""

from types import SimpleNamespace

from pipeline_test_common import transform
from pypto_pro._errors import InvalidArgument
import pypto_pro.language as pl
from pypto_pro.runtime.pipeline._validate import validate_local_slot_lifetime
import pytest

T, N, DEPTH = 64, 4, 3


@pl.pipeline.stage
def fill(k, a, db):
    tile = db.next()
    pl.load(tile, a, [k * T, 0])


@pl.pipeline.stage
def mid(k, x, scratch, ub):
    tile = ub.next()
    pl.load(tile, x, [k * T, 0])
    pl.store(scratch, tile, [k * T, 0])


@pl.pipeline.stage
def use_current(k, db, left):
    tile = db.current()
    pl.move(left.next(), tile)


@pl.pipeline.stage
def use_dynamic(k, offset, db, left):
    pl.move(left.next(), db[offset % 3])


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


def test_enough_slots_is_accepted():
    transform(_kernel(DEPTH, 2, use_current))


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


def test_dynamic_index_is_not_checked():
    transform(k_dynamic_index())


@pl.pipeline.stage
def scan_stage(k, db, src):
    current = db.next()
    previous = db.previous()
    pl.load(current, src, [k * T, 0])
    pl.muls(previous, previous, 1.0)


def _same_stage_ring():
    @pl.jit(arch="3510", auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=[0]))
    def kernel(src: pl.Tensor[[N * T, T], pl.DT_FP32]):
        db = pl.make_tile_group(
            type=pl.TileType(shape=[T, T], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=[index * 0x4000 for index in range(2)], mutex_ids=[0, 1])
        for k in pl.range(0, N):
            with pl.section_vector():
                scan_stage(k, db, src)
    return kernel


def test_same_stage_ring_is_skipped():
    transform(_same_stage_ring())

def _kernel_with_external_access():
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
            with pl.section_cube():
                tile = db.next()
                pl.load(tile, a, [k * T, 0])
            with pl.section_cube():
                fill(k, a, db)
            with pl.section_vector():
                mid(k, x, scratch, ub)
            with pl.section_cube():
                use_current(k, db, left)
    return kernel


def test_stage_external_access_is_ignored():
    transform(_kernel_with_external_access())

def _fake_info(roles, depth):
    stages = [
        SimpleNamespace(
            func_name=f"s{index}",
            section_kind="cube",
            buffer_access=[("db", role, "MTE2")] if role else [],
        )
        for index, role in enumerate(roles)
    ]
    return SimpleNamespace(
        sync=SimpleNamespace(buffers={}, slot_counts={"db": depth}, decl_spans={}),
        group_names={"db"},
        stages=stages,
    )


def test_read_write_stage_closes_and_opens_pairs():
    info = _fake_info(["W", "RW", "R"], depth=3)
    validate_local_slot_lifetime(info, [0, 2, 4])


def test_read_write_stage_uses_worst_pair():
    info = _fake_info(["W", "RW", "R"], depth=2)
    with pytest.raises(InvalidArgument, match=r"has 2 slot\(s\).*3 slots"):
        validate_local_slot_lifetime(info, [0, 2, 4])


def test_multiple_pairs_are_checked():
    info = _fake_info(["W", "R", "W", "R"], depth=2)
    validate_local_slot_lifetime(info, [0, 1, 2, 3])


def test_producer_without_reader_is_skipped():
    info = _fake_info(["W", "W", "R"], depth=1)
    validate_local_slot_lifetime(info, [0, 1, 2])
