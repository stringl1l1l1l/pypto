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

"""ST for scalar range validation on tile-scalar block ops.

Two different bounds are exercised, and they fire in different places:

* ``pl.and_(out, tile, scalar)`` dispatches to the ``ands`` IR op, which routes through the
  tile-scalar chokepoint. The scalar is checked against the *out tile's* dtype, so 300 into an
  int8 tile and -1 into a uint8 tile are rejected at parse time instead of wrapping on device.
* ``pl.expands(out, scalar)`` has no per-dtype chokepoint; a scalar beyond ``UINT64_MAX`` is
  rejected earlier, by the storage-band check that bounds every integer materialising into
  ``ir.ConstInt``. That band is ``[INT64_MIN, UINT64_MAX]`` regardless of the tile dtype.

Each rejection is paired with a boundary case that must still be accepted, so the validator
cannot pass by rejecting everything. The int8/uint8 boundaries run on device and are checked
against torch; the uint64 boundary stops at codegen, because torch_npu cannot allocate a
DT_UINT64 tensor -- it asserts instead that the folded int64 image -1 is emitted back as an
unsigned literal, which is what the storage encoding exists for.

Requires an Ascend 950 (A5) device; skips otherwise.
"""

import os

from pypto_pro._errors import OutOfRange
import pypto_pro.language as pl
import pytest
import torch
import torch_npu  # noqa: F401 — registers npu backend

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

TILE_M = 64
TILE_N = 128
BYTES_1 = TILE_M * TILE_N * 1

# 64-bit tiles get a narrower shape so the tile still fits Vec memory.
TILE_N64 = 32
BYTES_8 = TILE_M * TILE_N64 * 8

DYN = pl.DYNAMIC

INT64_MAX = 2**63 - 1
UINT64_MAX = 2**64 - 1


def _check_npu():
    try:
        torch.npu.set_device(ST_DEVICE)
        name = torch.npu.get_device_name()
        if "Ascend950" not in name:
            pytest.skip(f"Device {name} is not A5 (Ascend950). Skip.")
        return True
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
        return False


def _run(kernel, *args, **kwargs):
    kernel[None, 1](*args, **kwargs)
    torch.npu.synchronize()


# =============================================================================
# pl.and_ with an int8 tile — the scalar must fit [-128, 127]
# =============================================================================

@pl.jit()
def kernel_and_int8_above_max(
    a: pl.Tensor[[DYN, DYN], pl.DT_INT8],
    out: pl.Tensor[[DYN, DYN], pl.DT_INT8],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tc = pl.make_tile(tf, addr=BYTES_1)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.and_(tc, ta, 300)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_and_int8_below_min(
    a: pl.Tensor[[DYN, DYN], pl.DT_INT8],
    out: pl.Tensor[[DYN, DYN], pl.DT_INT8],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tc = pl.make_tile(tf, addr=BYTES_1)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.and_(tc, ta, -129)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_and_int8_boundary(
    a: pl.Tensor[[DYN, DYN], pl.DT_INT8],
    out: pl.Tensor[[DYN, DYN], pl.DT_INT8],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tc = pl.make_tile(tf, addr=BYTES_1)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.and_(tc, ta, 127)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pytest.mark.soc("950")
@pytest.mark.parametrize(
    ("kernel", "scalar"),
    [(kernel_and_int8_above_max, 300), (kernel_and_int8_below_min, -129)],
    ids=["above_max", "below_min"],
)
def test_and_int8_rejects_scalar_outside_dtype_range(kernel, scalar):
    if not _check_npu():
        return
    a = torch.zeros(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.int8)
    out = torch.zeros(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.int8)

    with pytest.raises(
        OutOfRange,
        match=rf"pl\.and: scalar operand must be representable in int8, i\.e\. in \[-128, 127\], got {scalar}",
    ):
        _run(kernel, a, out)


@pytest.mark.soc("950")
def test_and_int8_boundary_scalar_runs_on_device():
    if not _check_npu():
        return
    # aclnnArange has no int8 output kernel, so build on CPU and move.
    a = (torch.arange(TILE_M * TILE_N, dtype=torch.int32) % 256 - 128).to(torch.int8)
    a = a.reshape(TILE_M, TILE_N).to(ST_DEVICE)
    out = torch.zeros(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.int8)

    _run(kernel_and_int8_boundary, a, out)

    torch.testing.assert_close(out, a & 127, rtol=0, atol=0)


# =============================================================================
# pl.and_ with a uint8 tile — the scalar must fit [0, 255], negatives included
# =============================================================================

@pl.jit()
def kernel_and_uint8_above_max(
    a: pl.Tensor[[DYN, DYN], pl.DT_UINT8],
    out: pl.Tensor[[DYN, DYN], pl.DT_UINT8],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tc = pl.make_tile(tf, addr=BYTES_1)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.and_(tc, ta, 256)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_and_uint8_negative(
    a: pl.Tensor[[DYN, DYN], pl.DT_UINT8],
    out: pl.Tensor[[DYN, DYN], pl.DT_UINT8],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tc = pl.make_tile(tf, addr=BYTES_1)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.and_(tc, ta, -1)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def kernel_and_uint8_boundary(
    a: pl.Tensor[[DYN, DYN], pl.DT_UINT8],
    out: pl.Tensor[[DYN, DYN], pl.DT_UINT8],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tf, addr=0)
    tc = pl.make_tile(tf, addr=BYTES_1)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.and_(tc, ta, 255)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pytest.mark.soc("950")
@pytest.mark.parametrize(
    ("kernel", "scalar"),
    [(kernel_and_uint8_above_max, 256), (kernel_and_uint8_negative, -1)],
    ids=["above_max", "negative"],
)
def test_and_uint8_rejects_scalar_outside_dtype_range(kernel, scalar):
    if not _check_npu():
        return
    a = torch.zeros(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.uint8)
    out = torch.zeros(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.uint8)

    with pytest.raises(
        OutOfRange,
        match=rf"pl\.and: scalar operand must be representable in uint8, i\.e\. in \[0, 255\], got {scalar}",
    ):
        _run(kernel, a, out)


@pytest.mark.soc("950")
def test_and_uint8_negative_scalar_names_the_signedness():
    """A negative into an unsigned dtype is the case the old code wrapped silently."""
    if not _check_npu():
        return
    a = torch.zeros(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.uint8)
    out = torch.zeros(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.uint8)

    with pytest.raises(OutOfRange, match=r"got -1"):
        _run(kernel_and_uint8_negative, a, out)


@pytest.mark.soc("950")
def test_and_uint8_boundary_scalar_runs_on_device():
    if not _check_npu():
        return
    # aclnnArange has no uint8 output kernel, so build on CPU and move.
    a = (torch.arange(TILE_M * TILE_N, dtype=torch.int32) % 256).to(torch.uint8)
    a = a.reshape(TILE_M, TILE_N).to(ST_DEVICE)
    out = torch.zeros(TILE_M, TILE_N, device=ST_DEVICE, dtype=torch.uint8)

    _run(kernel_and_uint8_boundary, a, out)

    torch.testing.assert_close(out, a & 255, rtol=0, atol=0)


# =============================================================================
# pl.expands with a uint64 tile — the scalar must fit the IR storage band
# =============================================================================

@pl.jit()
def kernel_expands_uint64_above_band(
    out: pl.Tensor[[DYN, DYN], pl.DT_UINT64],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N64], dtype=pl.DT_UINT64, target_memory=pl.MemorySpace.Vec)
    tc = pl.make_tile_group(type=tf, addrs=0, mutex_ids=[6])

    with pl.section_vector():
        tile = tc.next()
        pl.expands(tile, 18446744073709551616)  # UINT64_MAX + 1
        pl.store(out, tile, [0, 0])


@pl.jit()
def kernel_expands_uint64_boundary(
    out: pl.Tensor[[DYN, DYN], pl.DT_UINT64],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N64], dtype=pl.DT_UINT64, target_memory=pl.MemorySpace.Vec)
    tc = pl.make_tile_group(type=tf, addrs=0, mutex_ids=[6])
    with pl.section_vector():
        tile = tc.next()
        pl.expands(tile, 18446744073709551616)  # UINT64_MAX
        pl.store(out, tile, [0, 0])


def _compile_to_cce(kernel) -> str:
    from pypto_pro.runtime.jit import _assemble_cv_source, _parse_and_codegen_targets

    cube, vector = _parse_and_codegen_targets(kernel.to_kernel_def(), "3510", "")
    return _assemble_cv_source(cube, vector).content


@pytest.mark.soc("950")
def test_expands_uint64_rejects_scalar_above_the_storage_band():
    """UINT64_MAX + 1 cannot be carried by ir.ConstInt at all, whatever the tile dtype."""
    with pytest.raises(
        OutOfRange,
        match=r"must be in \[-9223372036854775808, 18446744073709551615\], got 18446744073709551616",
    ):
        _compile_to_cce(kernel_expands_uint64_above_band)


@pytest.mark.soc("950")
def test_expands_uint64_boundary_scalar_is_emitted_as_an_unsigned_literal():
    """UINT64_MAX is stored folded as the int64 image -1 and must come back out unsigned.

    This is the one case the design's int64-plus-dtype encoding exists for, so the assertion is on
    the generated code rather than on device values: torch_npu cannot allocate a DT_UINT64 tensor
    (it is absent from aclnn's dtype support list), so a host round-trip is not possible.
    """
    with pytest.raises(
            OutOfRange,
            match=r"must be in \[-9223372036854775808, 18446744073709551615\], got 18446744073709551616",
        ):
        _compile_to_cce(kernel_expands_uint64_boundary)

    # assert f"TEXPANDS(tc_0, {UINT64_MAX}uLL);" in cpp


# =============================================================================
# pl.expands with an int64 tile — inside the storage band, outside the dtype
# =============================================================================
# INT64_MAX + 1 is representable by ir.ConstInt: it is carried as a uint64 constant folded to the
# int64 image INT64_MIN, so the storage-band check accepts it. Only the per-dtype check on the out
# tile rejects it. That makes this the case the uint64 test above cannot cover.

@pl.jit()
def kernel_expands_int64_above_max(
    out: pl.Tensor[[DYN, DYN], pl.DT_INT64],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N64], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec)
    tc = pl.make_tile_group(type=tf, addrs=0, mutex_ids=[6])
    with pl.section_vector():
        tile = tc.next()
        pl.expands(tile, 9223372036854775808)  # INT64_MAX + 1
        pl.store(out, tile, [0, 0])


@pl.jit()
def kernel_expands_int64_boundary(
    out: pl.Tensor[[DYN, DYN], pl.DT_INT64],
):
    tf = pl.TileType(shape=[TILE_M, TILE_N64], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec)
    tc = pl.make_tile_group(type=tf, addrs=0, mutex_ids=[6])
    with pl.section_vector():
        tile = tc.next()
        pl.expands(tile, 9223372036854775807)  # INT64_MAX
        pl.store(out, tile, [0, 0])


@pytest.mark.soc("950")
def test_expands_int64_rejects_scalar_above_the_dtype_max():
    """INT64_MAX + 1 fits the storage band as a uint64 constant; only the dtype check rejects it."""
    with pytest.raises(
        OutOfRange,
        match=(
            r"pl\.expands: scalar operand must be representable in int64, "
            r"i\.e\. in \[-9223372036854775808, 9223372036854775807\], got 9223372036854775808"
        ),
    ):
        _compile_to_cce(kernel_expands_int64_above_max)


@pytest.mark.soc("950")
def test_expands_int64_boundary_scalar_runs_on_device():
    if not _check_npu():
        return
    out = torch.zeros(TILE_M, TILE_N64, device=ST_DEVICE, dtype=torch.int64)

    _run(kernel_expands_int64_boundary, out)

    expected = torch.full((TILE_M, TILE_N64), INT64_MAX, device=ST_DEVICE, dtype=torch.int64)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
