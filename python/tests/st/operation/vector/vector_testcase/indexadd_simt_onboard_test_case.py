#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for A5 SIMT IndexAdd_ vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class IndexaddSimtOnboardConfig(VectorConfig):
    OPERATION = 'IndexAdd_'


INDEXADD_SIMT_ONBOARD_TESTS = [
    {
        'case_index': 0,
        'case_name': 'IndexAddSimt_fp32_axis1',
        'operation': 'IndexAdd_',
        'input_tensors': [
            {'name': 'target', 'shape': (3, 11), 'dtype': 'fp32'},
            {'name': 'source', 'shape': (3, 17), 'dtype': 'fp32'},
            {'name': 'indices', 'shape': (17,), 'dtype': 'int32'},
        ],
        'output_tensors': [
            {'name': 'target', 'shape': (3, 11), 'dtype': 'fp32'},
        ],
        'view_shape': (3, 17),
        'tile_shape': (2, 8),
        'params': {'axis': 1, 'alpha': 1},
    },
    {
        'case_index': 1,
        'case_name': 'IndexAddSimt_fp16_axis0',
        'operation': 'IndexAdd_',
        'input_tensors': [
            {'name': 'target', 'shape': (11, 3), 'dtype': 'fp16'},
            {'name': 'source', 'shape': (17, 3), 'dtype': 'fp16'},
            {'name': 'indices', 'shape': (17,), 'dtype': 'int32'},
        ],
        'output_tensors': [
            {'name': 'target', 'shape': (11, 3), 'dtype': 'fp16'},
        ],
        'view_shape': (17, 3),
        'tile_shape': (8, 2),
        'params': {'axis': 0, 'alpha': 1},
    },
    {
        'case_index': 2,
        'case_name': 'IndexAddSimt_bf16_axis1_alpha',
        'operation': 'IndexAdd_',
        'input_tensors': [
            {'name': 'target', 'shape': (3, 11), 'dtype': 'bf16'},
            {'name': 'source', 'shape': (3, 17), 'dtype': 'bf16'},
            {'name': 'indices', 'shape': (17,), 'dtype': 'int32'},
        ],
        'output_tensors': [
            {'name': 'target', 'shape': (3, 11), 'dtype': 'bf16'},
        ],
        'view_shape': (3, 17),
        'tile_shape': (2, 8),
        'params': {'axis': 1, 'alpha': -2},
    },
]
