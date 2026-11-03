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
"""Resource diagnostics preserve locations and stop at the first invalid allocation."""

from types import SimpleNamespace
import warnings

from pypto_pro import ir
from pypto_pro._errors import (
    CapacityUnknown,
    InvalidOperation,
    InvalidTile,
    InvalidType,
    InvalidVal,
    MutexMismatch,
    NotSupported,
    OutOfRange,
    current_span,
)
from pypto_pro.ir.op import block_ops
import pypto_pro.language as pl
from pypto_pro.runtime.platform import NpuArch
import pytest


@pytest.mark.parametrize("space", [
    ir.MemorySpace.Bias, ir.MemorySpace.Scaling, ir.MemorySpace.ScaleLeft, ir.MemorySpace.ScaleRight,
])
def test_parameter_buffer_capacity_is_checked(space):
    capacity = block_ops._get_memory_capacity(space)
    assert capacity > 0
    with pytest.raises(OutOfRange) as caught:
        block_ops._validate_tile_addr_capacity(capacity, 1, space, ir.Span("kernel.py", 7, 2))
    assert caught.value.span["line"] == 7


@pytest.mark.parametrize("space,count", [
    (ir.MemorySpace.Vec, 1), (ir.MemorySpace.ScaleLeft, 1),
    (ir.MemorySpace.ScaleRight, 1), (ir.MemorySpace.DDR, 0),
])
def test_unknown_capacity_warns_only_for_bounded_spaces(monkeypatch, space, count):
    from pypto_pro.runtime import platform
    monkeypatch.setattr(platform, "get_memory_limit", lambda arch, space: 0)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        assert block_ops._get_memory_capacity(space, ir.Span("capacity_kernel.py", 13, 1)) == 0
    assert len(recorded) == count
    if recorded:
        assert recorded[0].category is CapacityUnknown
        assert recorded[0].filename == "capacity_kernel.py" and recorded[0].lineno == 13
        assert space.name in str(recorded[0].message)


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("addr,error", [(1, InvalidTile), (0x40000, OutOfRange)])
def test_make_tile_error_uses_explicit_or_current_span(explicit, addr, error):
    span = ir.Span("explicit_kernel.py", 73, 9) if explicit else None
    with current_span(ir.Span("active_kernel.py", 42, 4)):
        with pytest.raises(error) as caught:
            block_ops.make_mutex_tile_expr([16, 16], pl.DT_FP16, ir.MemorySpace.Vec, addr=addr, span=span)
    location = caught.value.span
    expected = ("explicit_kernel.py", 73, 9) if explicit else ("active_kernel.py", 42, 4)
    assert (location["filename"], location["line"], location["column"]) == expected


@pytest.mark.parametrize("explicit", [False, True])
def test_builder_capacity_warning_uses_explicit_or_current_span(monkeypatch, explicit):
    from pypto_pro.runtime import platform
    monkeypatch.setattr(platform, "get_memory_limit", lambda arch, space: 0)
    span = ir.Span("explicit_kernel.py", 73, 9) if explicit else None
    with current_span(ir.Span("active_kernel.py", 42, 4)):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always", CapacityUnknown)
            block_ops.make_mutex_tile_expr([16, 16], pl.DT_FP16, ir.MemorySpace.Vec, addr=0, span=span)
    assert len(recorded) == 1 and recorded[0].category is CapacityUnknown
    expected = ("explicit_kernel.py", 73) if explicit else ("active_kernel.py", 42)
    assert (recorded[0].filename, recorded[0].lineno) == expected


@pytest.mark.parametrize("acc_addr", [0x40000, -1])
@pytest.mark.parametrize("probe", [False, True])
def test_kernel_stops_at_first_over_capacity_declaration(acc_addr, probe):
    @pl.jit(arch=NpuArch.DAV_3510)
    def oversized(x: pl.Tensor[[1], pl.DT_FP32]):
        vec_type = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        acc_type = pl.TileType(shape=[16, 16], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc)
        _vec = pl.make_tile(vec_type, addr=0x40000)
        _acc = pl.make_tile(acc_type, addr=acc_addr)
    kernel_def = oversized.to_kernel_def()
    with pytest.raises(OutOfRange) as caught:
        if probe:
            from pypto_pro.runtime.pipeline._analyzer import probe_kernel_facts
            probe_kernel_facts(kernel_def)
        else:
            kernel_def.parse_target_program(ir.SectionKind.Cube)
    assert "Vec capacity" in caught.value.message
    assert "Acc capacity" not in caught.value.message
    location = caught.value.span
    assert location["filename"] == __file__
    with open(__file__, encoding="utf-8") as source:
        assert "_vec = pl.make_tile(vec_type, addr=0x40000)" in source.readlines()[location["line"] - 1]


def test_builder_error_inherits_the_active_call_span():
    with current_span(ir.Span("direct_kernel.py", 19, 4)):
        with pytest.raises(InvalidType) as caught:
            block_ops._ir_store(SimpleNamespace(type=None), SimpleNamespace(type=None), [0, 0])
    assert caught.value.span["filename"] == "direct_kernel.py"
    assert caught.value.span["line"] == 19


def test_registered_builder_error_keeps_the_dsl_call_location():
    @pl.jit(arch=NpuArch.DAV_3510)
    def invalid_move(x: pl.Tensor[[1], pl.DT_FP16]):
        tile = pl.make_tile(pl.TileType(shape=[16, 16], dtype=pl.DT_FP16,
                                        target_memory=pl.MemorySpace.Vec), addr=0)
        with pl.section_vector():
            pl.move(tile, tile, phase=pl.STPhase.Final)
    with pytest.raises(NotSupported) as caught:
        invalid_move.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    assert "phase is only supported" in caught.value.message
    location = caught.value.span
    assert location["filename"] == __file__
    with open(__file__, encoding="utf-8") as source:
        assert "pl.move(tile, tile, phase=pl.STPhase.Final)" in source.readlines()[location["line"] - 1]


def test_nested_builder_error_keeps_the_inner_call_location():
    @pl.jit(arch=NpuArch.DAV_3510)
    def invalid_scalar(x: pl.Tensor[[1], pl.DT_FP16]):
        tile = pl.make_tile(pl.TileType(shape=[16, 16], dtype=pl.DT_FP16,
                                        target_memory=pl.MemorySpace.Vec), addr=0)
        packed = pl.make_tile(pl.TileType(shape=[16, 32], dtype=pl.DT_FP8E4M3FN,
                                          target_memory=pl.MemorySpace.Vec), addr=0x1000)
        with pl.section_vector():
            pl.muls(tile, tile,
                    pl.getval(packed, 0))
    with pytest.raises(NotSupported) as caught:
        invalid_scalar.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    assert "getval does not support container dtype" in caught.value.message
    location = caught.value.span
    assert location["filename"] == __file__
    with open(__file__, encoding="utf-8") as source:
        assert "pl.getval(packed, 0)" in source.readlines()[location["line"] - 1]


@pytest.mark.parametrize("shape,error", [
    ([-1, 16], InvalidVal), ([16, -1], InvalidVal),
    ([17, 16], InvalidVal), ([16, 17], InvalidVal),
])
def test_validshape_error_keeps_the_dsl_call_location(shape, error):
    @pl.jit(arch=NpuArch.DAV_3510)
    def invalid_validshape(x: pl.Tensor[[1], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[16, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            pl.set_validshape(tile, shape=shape)
    with pytest.raises(error) as caught:
        invalid_validshape.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    location = caught.value.span
    assert location["filename"] == __file__
    with open(__file__, encoding="utf-8") as source:
        assert "pl.set_validshape(tile, shape=shape)" in source.readlines()[location["line"] - 1]


@pytest.mark.parametrize("cols,error", [(-1, InvalidVal), (17, InvalidVal)])
def test_validshape_checks_static_dimension_alongside_dynamic_dimension(cols, error):
    @pl.jit(arch=NpuArch.DAV_3510)
    def invalid_validshape(x: pl.Tensor[[1], pl.DT_FP16], rows: pl.DT_INT64):
        tile_type = pl.TileType(shape=[16, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            pl.set_validshape(tile, shape=[rows, cols])

    with pytest.raises(error):
        invalid_validshape.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


@pytest.mark.parametrize("shape", [[0, 0], [0, 16], [16, 0], [16, 16]])
def test_validshape_accepts_zero_and_full_tile_dimensions(shape):
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def valid_validshape(x: pl.Tensor[[1], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[16, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            pl.set_validshape(tile, shape=shape)

    program, _ = valid_validshape.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    assert "block.set_validshape" in str(program)


def _manual_alias_kernel(addrs, ids, auto_mutex=True, include_auto=False):
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=auto_mutex)
    def kernel(x: pl.Tensor[[1], pl.DT_FP32]):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        group = pl.make_tile_group(type=tt, addrs=addrs, mutex_ids=ids, depth=2)
        alias = group
        if include_auto:
            _automatic = pl.make_tile_group(type=tt, addrs=0, mutex_ids="auto", depth=1)
        with pl.section_vector():
            pl.adds(alias[0], alias[0], 1.0)
    return kernel


@pytest.mark.parametrize("addrs,ids,auto_mutex,count", [
    ([0, 0], [1, 2], True, 1),
    ([0, 128], [1, 2], True, 1),
    ([0, 0], [1, 1], True, 0),
    ([0, 0], [[1, 2], [2, 3]], True, 0),
    ([0, 256], [1, 1], True, 0),
    ([0, 0], None, True, 0),
    ([0, 0], None, False, 0),
])
def test_manual_mutex_warning_preserves_alias_and_reuse_contracts(addrs, ids, auto_mutex, count):
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", MutexMismatch)
        _manual_alias_kernel(addrs, ids, auto_mutex).to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    assert len(recorded) == count
    if recorded:
        assert recorded[0].category is MutexMismatch
        assert recorded[0].filename == __file__


def test_native_auto_alias_rejection_runs_before_manual_warning():
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", MutexMismatch)
        with pytest.raises(InvalidOperation) as caught:
            _manual_alias_kernel([0, 0], [1, 2], include_auto=True).to_kernel_def().parse_target_program(
                ir.SectionKind.Vector)
    assert "overlapping address ranges must use the same mutex ID" in caught.value.message
    assert not recorded
