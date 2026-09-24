#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Basic system test for A5 SIMT index-add."""

import math
import os

import pytest
import torch
from vector_testcase.indexadd_simt_onboard_test_case import (
    INDEXADD_SIMT_ONBOARD_TESTS,
    IndexaddSimtOnboardConfig,
)
from vector_testcase.vector_test_case import TORCH_DTYPES

import pypto

pytestmark = pytest.mark.soc("950")


def make_indexadd_simt_inputs(config: IndexaddSimtOnboardConfig):
    target_shape, source_shape, index_shape = config.input_shapes
    target = torch.ones(target_shape, dtype=TORCH_DTYPES[config.input_tensors[0].dtype])
    source = (torch.arange(math.prod(source_shape)) % 3 - 1).reshape(source_shape)
    source = source.to(TORCH_DTYPES[config.input_tensors[1].dtype])
    indices = (torch.arange(index_shape[0], dtype=TORCH_DTYPES[config.input_tensors[2].dtype]) * 3)
    indices = indices % target_shape[config.axis]
    return target, source, indices


@pytest.mark.parametrize(
    "case", INDEXADD_SIMT_ONBOARD_TESTS, ids=[case["case_name"] for case in INDEXADD_SIMT_ONBOARD_TESTS]
)
@pypto.options(pass_options={"enable_slice": True})
def test_indexadd_simt(case: dict):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    config = IndexaddSimtOnboardConfig.from_test_case(case)
    target_cpu, source_cpu, indices_cpu = make_indexadd_simt_inputs(config)
    expected = torch.index_add(target_cpu, config.axis, indices_cpu.long(), source_cpu, alpha=config.alpha)

    @pypto.frontend.jit
    def kernel(target: pypto.Tensor(), source: pypto.Tensor(), indices: pypto.Tensor()):
        pypto.set_vec_tile_shapes(*config.tile_shape)
        pypto.index_add_(target, config.axis, indices, source, alpha=config.alpha)

    target = target_cpu.to(f"npu:{device_id}")
    kernel(target, source_cpu.to(f"npu:{device_id}"), indices_cpu.to(f"npu:{device_id}"))
    torch.testing.assert_close(target.cpu(), expected, rtol=0, atol=0)
