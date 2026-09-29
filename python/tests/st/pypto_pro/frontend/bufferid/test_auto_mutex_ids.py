#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software and you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""System tests for whole-kernel ``mutex_ids="auto"`` allocation.

The cases cover balanced reuse once a single group exceeds the 32-ID pool,
AUTO groups coexisting with exhausted manual IDs and a manual candidate chain
in one compilation, per-slot manual-ID inheritance through address aliasing,
three AUTO groups overlapping in every Vec address pattern, non-contiguous
addresses visited out of order, groups created inside nested or branching
Python helpers, and independent Cube/Vector ID pools exercised by a combined
Cube+Vector kernel.
"""

import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


# ===========================================================================
# Pool pressure: more AUTO slots than the 32 available mutex IDs
# ===========================================================================

OVER_32_ROWS = 16
OVER_32_COLS = 32
OVER_32_DEPTH = 33
OVER_32_FULL_ROWS = OVER_32_ROWS * OVER_32_DEPTH


@pl.jit(arch="3510", auto_mutex=True)
def auto_over_32_kernel(
    source: pl.Tensor[[OVER_32_FULL_ROWS, OVER_32_COLS], pl.DT_FP16],
    output: pl.Tensor[[OVER_32_FULL_ROWS, OVER_32_COLS], pl.DT_FP16],
):
    tile_type = pl.TileType(
        shape=[OVER_32_ROWS, OVER_32_COLS],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    group = pl.make_tile_group(
        type=tile_type,
        addrs=0,
        mutex_ids="auto",
        depth=OVER_32_DEPTH,
    )
    with pl.section_vector():
        for index in pl.range(0, OVER_32_DEPTH):
            tile = group[index]
            row = index * OVER_32_ROWS
            pl.load(tile, source, [row, 0])
            pl.add(tile, tile, tile)
            pl.store(output, tile, [row, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_auto_more_than_32_tiles_runs_with_balanced_id_reuse():
    """A single AUTO group with 33 slots forces balanced reuse across the 32-ID pool."""
    _require_a5(ST_DEVICE)
    torch.manual_seed(4)
    source = torch.rand([OVER_32_FULL_ROWS, OVER_32_COLS], device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    output = torch.zeros_like(source)

    auto_over_32_kernel(source, output)
    torch.npu.synchronize()

    torch.testing.assert_close(output.cpu().float(), source.cpu().float() * 2.0, rtol=3e-3, atol=3e-3)


# ===========================================================================
# Manual-ID coexistence and address aliasing
# ===========================================================================

GUARD_ROWS = 16
GUARD_COLS = 32
GUARD_MANUAL_DEPTH = 32
GUARD_MANUAL_IDS = tuple(range(GUARD_MANUAL_DEPTH))
GUARD_CHAIN_DEPTH = 2
GUARD_FULL_ROWS = GUARD_ROWS * (GUARD_MANUAL_DEPTH + GUARD_CHAIN_DEPTH)


@pl.jit(arch="3510", auto_mutex=True)
def auto_and_manual_guard_paths_kernel(
    source: pl.Tensor[[GUARD_FULL_ROWS, GUARD_COLS], pl.DT_FP16],
    output: pl.Tensor[[GUARD_FULL_ROWS, GUARD_COLS], pl.DT_FP16],
):
    tile_type = pl.TileType(
        shape=[GUARD_ROWS, GUARD_COLS],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    all_manual_ids = pl.make_tile_group(
        type=tile_type,
        addrs=0x0000,
        mutex_ids=GUARD_MANUAL_IDS,
    )
    auto_group = pl.make_tile_group(
        type=tile_type,
        addrs=0x10000,
        mutex_ids="auto",
        depth=1,
    )

    # The candidates form a connected chain: out {1} <-> lhs {1, 2}
    # <-> rhs {2}, while out and rhs can never select the same ID.
    chain_out = pl.make_tile_group(type=tile_type, addrs=0x11000, mutex_ids=[1])
    chain_lhs = pl.make_tile_group(type=tile_type, addrs=0x12000, mutex_ids=[1, 2])
    chain_rhs = pl.make_tile_group(type=tile_type, addrs=0x13000, mutex_ids=[2])

    with pl.section_vector():
        for index in pl.range(0, GUARD_MANUAL_DEPTH):
            row = index * GUARD_ROWS
            pl.load(all_manual_ids[index], source, [row, 0])
            pl.move(auto_group[0], all_manual_ids[index])
            pl.store(output, auto_group[0], [row, 0])

        for index in pl.range(0, GUARD_CHAIN_DEPTH):
            row = (GUARD_MANUAL_DEPTH + index) * GUARD_ROWS
            pl.load(chain_lhs[index], source, [row, 0])
            pl.load(chain_rhs[0], source, [row, 0])
            pl.add(chain_out[0], chain_lhs[index], chain_rhs[0])
            pl.store(output, chain_out[0], [row, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_auto_and_manual_runtime_guard_paths():
    """AUTO must land on a manual ID when all 32 are taken and avoid the IDs each mixed op locks."""
    _require_a5(ST_DEVICE)
    torch.manual_seed(24)
    source = torch.rand([GUARD_FULL_ROWS, GUARD_COLS], device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    output = torch.zeros_like(source)

    auto_and_manual_guard_paths_kernel(source, output)
    torch.npu.synchronize()

    golden = source.cpu().float()
    golden[GUARD_MANUAL_DEPTH * GUARD_ROWS:] *= 2.0
    torch.testing.assert_close(output.cpu().float(), golden, rtol=3e-3, atol=3e-3)


IF_MERGE_ROWS = 16
IF_MERGE_COLS = 32
IF_MERGE_DEPTH = 2
IF_MERGE_MANUAL_IDS = (25, 26)
IF_MERGE_FULL_ROWS = IF_MERGE_ROWS * IF_MERGE_DEPTH


@pl.jit(arch="3510", auto_mutex=True)
def auto_if_merge_manual_candidates_kernel(
    source: pl.Tensor[[IF_MERGE_FULL_ROWS, IF_MERGE_COLS], pl.DT_FP16],
    output: pl.Tensor[[IF_MERGE_FULL_ROWS, IF_MERGE_COLS], pl.DT_FP16],
):
    tile_type = pl.TileType(
        shape=[IF_MERGE_ROWS, IF_MERGE_COLS],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    manual_group = pl.make_tile_group(
        type=tile_type,
        addrs=[0x0000, 0x1000],
        mutex_ids=IF_MERGE_MANUAL_IDS,
    )
    branch_auto_group = pl.make_tile_group(
        type=tile_type,
        addrs=0x2000,
        mutex_ids="auto",
        depth=1,
    )
    add_auto_group = pl.make_tile_group(
        type=tile_type,
        addrs=0x3000,
        mutex_ids="auto",
        depth=1,
    )

    with pl.section_vector():
        for index in pl.range(0, IF_MERGE_DEPTH):
            row = index * IF_MERGE_ROWS
            # Dynamic selection gives this Tile one runtime ID with manual candidates {25, 26}.
            selected = manual_group[index]
            pl.load(selected, source, [row, 0])
            if index > 0:
                selected = branch_auto_group[0]
                pl.load(selected, source, [row, 0])

            add_auto = add_auto_group[0]
            pl.load(add_auto, source, [row, 0])
            pl.add(add_auto, selected, add_auto)
            pl.store(output, add_auto, [row, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_auto_if_merge_manual_candidates_with_auto_tile():
    """An if-merged manual/AUTO Tile must retain every possible mutex ID for the following add."""
    _require_a5(ST_DEVICE)
    torch.manual_seed(25)
    source = torch.rand([IF_MERGE_FULL_ROWS, IF_MERGE_COLS], device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    output = torch.zeros_like(source)

    auto_if_merge_manual_candidates_kernel(source, output)
    torch.npu.synchronize()

    torch.testing.assert_close(output.cpu().float(), source.cpu().float() * 2.0, rtol=3e-3, atol=3e-3)


ALIAS_ROWS = 64
ALIAS_HALF_ROWS = ALIAS_ROWS // 2
ALIAS_COLS = 64
ALIAS_DEPTH = 2
ALIAS_TILES = 6
ALIAS_FULL_ROWS = ALIAS_ROWS * ALIAS_TILES


@pl.jit(arch="3510", auto_mutex=True)
def auto_partial_alias_matching_manual_id_kernel(
    source: pl.Tensor[[ALIAS_FULL_ROWS, ALIAS_COLS], pl.DT_FP16],
    output: pl.Tensor[[ALIAS_FULL_ROWS, ALIAS_COLS], pl.DT_FP16],
):
    """Constrain each AUTO half-tile to reuse the overlapping manual ID."""
    full_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[ALIAS_ROWS, ALIAS_COLS],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=[0x0000, 0x4000],
        mutex_ids=[9, 10],
        depth=ALIAS_DEPTH,
    )
    # Per slot, full/top use the same ID. bottom is adjacent to top but overlaps
    # full, so AUTO must inherit that per-slot ID transitively.
    top_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[ALIAS_HALF_ROWS, ALIAS_COLS],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=[0x0000, 0x4000],
        mutex_ids=[9, 10],
        depth=ALIAS_DEPTH,
    )
    bottom_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[ALIAS_HALF_ROWS, ALIAS_COLS],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=[0x1000, 0x5000],
        mutex_ids="auto",
        depth=ALIAS_DEPTH,
    )

    with pl.section_vector():
        for index in pl.range(0, ALIAS_TILES):
            slot = index % ALIAS_DEPTH
            row = index * ALIAS_ROWS
            pl.load(full_group[slot], source, [row, 0])
            pl.add(top_group[slot], top_group[slot], top_group[slot])
            pl.exp(bottom_group[slot], bottom_group[slot])
            pl.store(output, top_group[slot], [row, 0])
            pl.store(output, bottom_group[slot], [row + ALIAS_HALF_ROWS, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_auto_partial_alias_reuses_per_slot_manual_id():
    """AUTO half-tiles inherit the per-slot manual ID of the full tile they overlap."""
    _require_a5(ST_DEVICE)
    torch.manual_seed(12)
    source = torch.rand([ALIAS_FULL_ROWS, ALIAS_COLS], device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    output = torch.zeros_like(source)

    auto_partial_alias_matching_manual_id_kernel(source, output)
    torch.npu.synchronize()

    source_cpu = source.cpu().float().reshape(ALIAS_TILES, ALIAS_ROWS, ALIAS_COLS)
    golden = torch.empty_like(source_cpu)
    golden[:, :ALIAS_HALF_ROWS, :] = source_cpu[:, :ALIAS_HALF_ROWS, :] * 2.0
    golden[:, ALIAS_HALF_ROWS:, :] = torch.exp(source_cpu[:, ALIAS_HALF_ROWS:, :])
    torch.testing.assert_close(
        output.cpu().float(),
        golden.reshape(ALIAS_FULL_ROWS, ALIAS_COLS),
        rtol=3e-3,
        atol=3e-3,
    )


TRIPLE_ROWS = 32
TRIPLE_COLS = 64
TRIPLE_DEPTH = 5
TRIPLE_TILES = 10
TRIPLE_FULL_ROWS = TRIPLE_ROWS * TRIPLE_TILES


@pl.jit(arch="3510", auto_mutex=True)
def auto_three_groups_partial_vec_overlap_kernel(
    source: pl.Tensor[[TRIPLE_FULL_ROWS, TRIPLE_COLS], pl.DT_FP16],
    output: pl.Tensor[[TRIPLE_FULL_ROWS, TRIPLE_COLS], pl.DT_FP16],
):
    tile_type = pl.TileType(
        shape=[TRIPLE_ROWS, TRIPLE_COLS],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    # Each tile occupies 0x1000 bytes. The slots cover all overlap patterns:
    # all three groups, load/compute, load/store, compute/store, and no overlap.
    load_group = pl.make_tile_group(
        type=tile_type,
        addrs=[0x0000, 0x1000, 0x2000, 0x3000, 0x4000],
        mutex_ids="auto",
        depth=TRIPLE_DEPTH,
    )
    compute_group = pl.make_tile_group(
        type=tile_type,
        addrs=[0x0000, 0x1000, 0x5000, 0x6000, 0x7000],
        mutex_ids="auto",
        depth=TRIPLE_DEPTH,
    )
    store_group = pl.make_tile_group(
        type=tile_type,
        addrs=[0x0000, 0x8000, 0x2000, 0x6000, 0x9000],
        mutex_ids="auto",
        depth=TRIPLE_DEPTH,
    )

    with pl.section_vector():
        for index in pl.range(0, TRIPLE_TILES):
            slot = index % TRIPLE_DEPTH
            row = index * TRIPLE_ROWS
            pl.load(load_group[slot], source, [row, 0])
            pl.exp(compute_group[slot], load_group[slot])
            pl.add(store_group[slot], compute_group[slot], compute_group[slot])
            pl.store(output, store_group[slot], [row, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_auto_three_groups_share_ids_for_overlapping_vec_addresses():
    """Three AUTO groups overlapping in every Vec address pattern receive consistent IDs."""
    _require_a5(ST_DEVICE)
    torch.manual_seed(23)
    source = torch.rand([TRIPLE_FULL_ROWS, TRIPLE_COLS], device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    output = torch.zeros_like(source)

    auto_three_groups_partial_vec_overlap_kernel(source, output)
    torch.npu.synchronize()

    golden = torch.exp(source.cpu().float()) * 2.0
    torch.testing.assert_close(output.cpu().float(), golden, rtol=3e-3, atol=3e-3)


NONCONTIG_ROWS = 16
NONCONTIG_COLS = 32
NONCONTIG_DEPTH = 40
NONCONTIG_ADDRS = tuple(index * 0x800 for index in range(NONCONTIG_DEPTH))
NONCONTIG_TILES = NONCONTIG_DEPTH * 2
NONCONTIG_FULL_ROWS = NONCONTIG_ROWS * NONCONTIG_TILES


@pl.jit(arch="3510", auto_mutex=True)
def auto_noncontiguous_over_32_kernel(
    source: pl.Tensor[[NONCONTIG_FULL_ROWS, NONCONTIG_COLS], pl.DT_FP16],
    output: pl.Tensor[[NONCONTIG_FULL_ROWS, NONCONTIG_COLS], pl.DT_FP16],
):
    group = pl.make_tile_group(
        type=pl.TileType(
            shape=[NONCONTIG_ROWS, NONCONTIG_COLS],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=NONCONTIG_ADDRS,
        mutex_ids="auto",
        depth=NONCONTIG_DEPTH,
    )
    with pl.section_vector():
        for index in pl.range(0, NONCONTIG_TILES):
            # Seven is coprime with 40, so every non-contiguous slot is visited
            # once per cycle in an order unrelated to its source address.
            slot = (index * 7) % NONCONTIG_DEPTH
            row = index * NONCONTIG_ROWS
            tile = group[slot]
            pl.load(tile, source, [row, 0])
            pl.exp(tile, tile)
            pl.store(output, tile, [row, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_auto_noncontiguous_more_than_32_slots():
    """Gapped addresses visited in a permuted order still yield correct alias components."""
    _require_a5(ST_DEVICE)
    torch.manual_seed(16)
    source = torch.rand([NONCONTIG_FULL_ROWS, NONCONTIG_COLS], device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    output = torch.zeros_like(source)

    auto_noncontiguous_over_32_kernel(source, output)
    torch.npu.synchronize()

    torch.testing.assert_close(output.cpu().float(), torch.exp(source.cpu().float()), rtol=3e-3, atol=3e-3)


# ===========================================================================
# Groups created inside Python helpers
# ===========================================================================

NESTED_ROWS = 16
NESTED_COLS = 32
NESTED_DEPTH = 3
NESTED_TILES = 9
NESTED_FULL_ROWS = NESTED_ROWS * NESTED_TILES


def _make_nested_auto_group_leaf(tile_type, base_addr):
    return pl.make_tile_group(
        type=tile_type,
        addrs=base_addr,
        mutex_ids="auto",
        depth=NESTED_DEPTH,
    )


def _make_nested_auto_group(tile_type, base_addr):
    return _make_nested_auto_group_leaf(tile_type, base_addr)


@pl.jit(arch="3510", auto_mutex=True)
def auto_nested_helper_permuted_slots_kernel(
    lhs: pl.Tensor[[NESTED_FULL_ROWS, NESTED_COLS], pl.DT_FP16],
    rhs: pl.Tensor[[NESTED_FULL_ROWS, NESTED_COLS], pl.DT_FP16],
    output: pl.Tensor[[NESTED_FULL_ROWS, NESTED_COLS], pl.DT_FP16],
):
    tile_type = pl.TileType(
        shape=[NESTED_ROWS, NESTED_COLS],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    lhs_group = _make_nested_auto_group(tile_type, 0x0000)
    rhs_group = _make_nested_auto_group(tile_type, 0x2000)
    output_group = _make_nested_auto_group(tile_type, 0x4000)

    with pl.section_vector():
        for index in pl.range(0, NESTED_TILES):
            lhs_slot = index % NESTED_DEPTH
            rhs_slot = (index * 2 + 1) % NESTED_DEPTH
            output_slot = (index + 2) % NESTED_DEPTH
            row = index * NESTED_ROWS
            pl.load(lhs_group[lhs_slot], lhs, [row, 0])
            pl.load(rhs_group[rhs_slot], rhs, [row, 0])
            pl.add(output_group[output_slot], lhs_group[lhs_slot], rhs_group[rhs_slot])
            pl.store(output, output_group[output_slot], [row, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_auto_nested_helper_calls_with_permuted_slots():
    """Groups created inside a nested Python helper join one whole-kernel plan with permuted slots."""
    _require_a5(ST_DEVICE)
    torch.manual_seed(14)
    lhs = torch.rand([NESTED_FULL_ROWS, NESTED_COLS], device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    torch.manual_seed(15)
    rhs = torch.rand([NESTED_FULL_ROWS, NESTED_COLS], device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    output = torch.zeros_like(lhs)

    auto_nested_helper_permuted_slots_kernel(lhs, rhs, output)
    torch.npu.synchronize()

    golden = lhs.cpu().float() + rhs.cpu().float()
    torch.testing.assert_close(output.cpu().float(), golden, rtol=3e-3, atol=3e-3)


BRANCH_ROWS = 16
BRANCH_COLS = 32
BRANCH_DEPTH = 2
BRANCH_TILES = 6
BRANCH_FULL_ROWS = BRANCH_ROWS * BRANCH_TILES


def _make_compile_time_branch_group(tile_type, base_addr, use_primary):
    if use_primary:
        group = pl.make_tile_group(
            type=tile_type,
            addrs=base_addr,
            mutex_ids="auto",
            depth=BRANCH_DEPTH,
        )
    else:
        group = pl.make_tile_group(
            type=tile_type,
            addrs=base_addr + 0x2000,
            mutex_ids="auto",
            depth=BRANCH_DEPTH,
        )
    return group


@pl.jit(arch="3510", auto_mutex=True)
def auto_helper_compile_time_branches_kernel(
    source: pl.Tensor[[BRANCH_FULL_ROWS, BRANCH_COLS], pl.DT_FP16],
    output: pl.Tensor[[BRANCH_FULL_ROWS, BRANCH_COLS], pl.DT_FP16],
):
    tile_type = pl.TileType(
        shape=[BRANCH_ROWS, BRANCH_COLS],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    source_group = _make_compile_time_branch_group(tile_type, 0x0000, True)
    output_group = _make_compile_time_branch_group(tile_type, 0x0000, False)
    with pl.section_vector():
        for index in pl.range(0, BRANCH_TILES):
            source_slot = index % BRANCH_DEPTH
            output_slot = (index + 1) % BRANCH_DEPTH
            row = index * BRANCH_ROWS
            pl.load(source_group[source_slot], source, [row, 0])
            pl.add(output_group[output_slot], source_group[source_slot], source_group[source_slot])
            pl.store(output, output_group[output_slot], [row, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_auto_helper_compile_time_branch_calls():
    """A helper's compile-time if/else builds groups at different addresses; both branches plan correctly."""
    _require_a5(ST_DEVICE)
    torch.manual_seed(22)
    source = torch.rand([BRANCH_FULL_ROWS, BRANCH_COLS], device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    output = torch.zeros_like(source)

    auto_helper_compile_time_branches_kernel(source, output)
    torch.npu.synchronize()

    torch.testing.assert_close(output.cpu().float(), source.cpu().float() * 2.0, rtol=3e-3, atol=3e-3)


# ===========================================================================
# Cube/Vector targets: independent 32-ID pools
# ===========================================================================

CV_M = 64
CV_K = 64
CV_N = 64
CV_VEC_ROWS = 32


@pl.jit(arch="3510", auto_mutex=True)
def auto_cube_vector_kernel(
    a: pl.Tensor[[CV_M, CV_N], pl.DT_FP32],
    b: pl.Tensor[[CV_M, CV_K], pl.DT_FP16],
    c: pl.Tensor[[CV_K, CV_N], pl.DT_FP16],
    output: pl.Tensor[[CV_M, CV_N], pl.DT_FP32],
):
    b_mat = pl.make_tile_group(
        type=pl.TileType(shape=[CV_M, CV_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0,
        mutex_ids="auto",
        depth=1,
    )
    c_mat = pl.make_tile_group(
        type=pl.TileType(shape=[CV_K, CV_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x2000,
        mutex_ids="auto",
        depth=1,
    )
    b_left = pl.make_tile_group(
        type=pl.TileType(shape=[CV_M, CV_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0,
        mutex_ids="auto",
        depth=1,
    )
    c_right = pl.make_tile_group(
        type=pl.TileType(shape=[CV_K, CV_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0,
        mutex_ids="auto",
        depth=1,
    )
    accumulator = pl.make_tile_group(
        type=pl.TileType(shape=[CV_M, CV_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0,
        mutex_ids="auto",
        depth=1,
    )
    vec_group = pl.make_tile_group(
        type=pl.TileType(shape=[CV_VEC_ROWS, CV_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0,
        mutex_ids="auto",
        depth=4,
    )

    with pl.section_cube():
        pl.load(b_mat[0], b, [0, 0])
        pl.load(c_mat[0], c, [0, 0])
        pl.move(b_left[0], b_mat[0])
        pl.move(c_right[0], c_mat[0])
        pl.matmul(accumulator[0], b_left[0], c_right[0])
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.move(vec_group[0], accumulator[0], acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    with pl.section_vector():
        sub_index = pl.get_subblock_idx()
        row_offset = sub_index * CV_VEC_ROWS
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=0)
        pl.load(vec_group[1], a, [row_offset, 0])
        pl.add(vec_group[2], vec_group[1], vec_group[0])
        pl.move(vec_group[3], vec_group[2])
        pl.store(output, vec_group[3], [row_offset, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_auto_cube_vector_targets_use_independent_pools():
    """Cube and Vector targets draw from independent 32-ID pools within one kernel."""
    _require_a5(ST_DEVICE)
    torch.manual_seed(5)
    a = (torch.rand([CV_M, CV_N], device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0).to(torch.float32)
    torch.manual_seed(6)
    b = torch.rand([CV_M, CV_K], device=ST_DEVICE, dtype=torch.float16) * 0.1
    torch.manual_seed(7)
    c = torch.rand([CV_K, CV_N], device=ST_DEVICE, dtype=torch.float16) * 0.1
    output = torch.zeros([CV_M, CV_N], device=ST_DEVICE, dtype=torch.float32)

    auto_cube_vector_kernel(a, b, c, output)
    torch.npu.synchronize()

    golden = a.cpu().float() + b.cpu().float() @ c.cpu().float()
    torch.testing.assert_close(output.cpu().float(), golden, rtol=3e-3, atol=3e-3)


if __name__ == "__main__":
    test_auto_more_than_32_tiles_runs_with_balanced_id_reuse()
    test_auto_and_manual_runtime_guard_paths()
    test_auto_if_merge_manual_candidates_with_auto_tile()
    test_auto_partial_alias_reuses_per_slot_manual_id()
    test_auto_three_groups_share_ids_for_overlapping_vec_addresses()
    test_auto_noncontiguous_more_than_32_slots()
    test_auto_nested_helper_calls_with_permuted_slots()
    test_auto_helper_compile_time_branch_calls()
    test_auto_cube_vector_targets_use_independent_pools()
