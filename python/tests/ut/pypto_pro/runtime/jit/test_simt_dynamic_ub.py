# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Regression tests for parser-collected SIMT dynamic UB inference."""

import importlib
from types import SimpleNamespace

from pypto_pro._errors import OutOfRange
import pypto_pro.language as pl
import pytest

from pypto.pypto_impl import ir

jit = importlib.import_module("pypto_pro.runtime.jit")


@pytest.mark.parametrize(
    ("requires_simt", "tile_high_water", "expected_dynamic_ub"),
    (
        (False, 216 * 1024 + 32, 0),
        (True, 0, 0),
        (True, 8 * 1024, 0),
        (True, 8 * 1024 + 32, 32),
        (True, 24 * 1024, 16 * 1024),
        (True, 216 * 1024, 208 * 1024),
    ),
)
def test_dynamic_ub_inference(requires_simt, tile_high_water, expected_dynamic_ub):
    assert jit._infer_dynamic_ub_size(requires_simt, tile_high_water) == expected_dynamic_ub


def test_dynamic_ub_inference_rejects_tile_beyond_a5_user_ub():
    with pytest.raises(OutOfRange) as exc_info:
        jit._infer_dynamic_ub_size(True, 216 * 1024 + 32)
    message = str(exc_info.value)
    assert "ErrCode: F00008! Enum: ExternalError::OUT_OF_RANGE." in message
    assert "allows at most" in message


def test_parser_collects_vec_high_water_from_tile_group():
    @pl.jit
    def kernel(_src: pl.Tensor[[1, 1], pl.DT_FP32]):
        tile_type = pl.TileType(shape=[1, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        _tiles = pl.make_tile_group(type=tile_type, addrs=[2048, 8192], mutex_ids=[0, 1])

    kernel_def = kernel.to_kernel_def()
    kernel_def.parse_target_program(ir.SectionKind.Vector)

    assert kernel_def.max_vec_tile_end == 8192 + 32
    assert not kernel_def.requires_simt


def test_parser_marks_only_the_kernel_that_launches_simt():
    @pl.vector_function(mode="simt", max_threads=1)
    def simt_copy(src, out):
        out[0, 0] = src[0, 0]

    @pl.jit
    def simt_kernel(
        src: pl.Tensor[[1, 1], pl.DT_FP32],
        out: pl.Tensor[[1, 1], pl.DT_FP32],
    ):
        with pl.section_vector():
            simt_copy[1](src, out)

    @pl.jit
    def simd_kernel(_src: pl.Tensor[[1, 1], pl.DT_FP32]):
        tile_type = pl.TileType(shape=[1, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        _tiles = pl.make_tile_group(type=tile_type, addrs=24 * 1024 - 32, mutex_ids=[0])

    simt_def = simt_kernel.to_kernel_def()
    simt_def.parse_target_program(ir.SectionKind.Vector)
    simd_def = simd_kernel.to_kernel_def()
    simd_def.parse_target_program(ir.SectionKind.Vector)

    assert simt_def.requires_simt
    assert not simd_def.requires_simt
    assert simd_def.max_vec_tile_end == 24 * 1024


def test_cube_and_vector_facts_are_aggregated_for_one_kernel(monkeypatch):
    class FakeKernelDef:
        def __init__(self):
            self.last_param_directions = {}
            self.max_vec_tile_end = 0
            self.requires_simt = False

        def parse_target_program(self, target, bound_signature=None):
            if target == ir.SectionKind.Cube:
                self.max_vec_tile_end = 216 * 1024
                self.requires_simt = False
            else:
                self.max_vec_tile_end = 8 * 1024
                self.requires_simt = True
            return SimpleNamespace(target=target), True

    recorded_dynamic_ub = {}

    def fake_codegen(
        prog,
        arch,
        build_dir,
        target,
        declared_directions=None,
        required_dynamic_ub_size=0,
    ):
        recorded_dynamic_ub[target] = required_dynamic_ub_size
        return SimpleNamespace(sanitizer=False)

    monkeypatch.setattr(jit, "_codegen_target_cce", fake_codegen)

    jit._parse_and_codegen_targets(FakeKernelDef(), "3510", "unused")

    assert recorded_dynamic_ub == {
        ir.SectionKind.Cube: 208 * 1024,
        ir.SectionKind.Vector: 208 * 1024,
    }
