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
"""
pypto cast+matmul ST测试用例配置
用于System Test自动化测试框架
测试场景：先将输入进行类型转换（cast），再执行矩阵乘法（matmul）；
另含 Cast+ScaledMM(MX) 的 UB2L1 场景（validK 对齐走直通、未对齐回退 DDR）
"""

from dataclasses import dataclass

import torch

import pypto


@dataclass
class CastMatmulConfig:
    shape: tuple[int, int, int]
    cube_tile_shape: tuple[list, list, list]
    a_vec_tile_shape: list
    b_vec_tile_shape: list
    view_shape: tuple[int, int]
    a_input_dtype: str
    b_input_dtype: str
    matmul_dtype: str
    out_dtype: str
    a_cast: bool
    b_cast: bool
    a_trans: bool = False
    b_trans: bool = False
    batch_shape: tuple[int, ...] = ()
    batch_view_shape: tuple[int, ...] = ()
    enable_ksplit: bool = False

    DTYPE_CONFIG = {
        "DT_FP16": {"pto": pypto.DT_FP16, "torch": torch.float16, "atol": 1e-3, "rtol": 1e-3},
        "DT_FP32": {"pto": pypto.DT_FP32, "torch": torch.float32, "atol": 1e-3, "rtol": 1e-3},
        "DT_BF16": {"pto": pypto.DT_BF16, "torch": torch.bfloat16, "atol": 1e-2, "rtol": 1e-2},
        "DT_INT8": {"pto": pypto.DT_INT8, "torch": torch.int8, "atol": 0, "rtol": 0},
        "DT_INT32": {"pto": pypto.DT_INT32, "torch": torch.int32, "atol": 0, "rtol": 0},
    }

    @property
    def a_input_pto_dtype(self):
        return self.DTYPE_CONFIG[self.a_input_dtype]["pto"]

    @property
    def b_input_pto_dtype(self):
        return self.DTYPE_CONFIG[self.b_input_dtype]["pto"]

    @property
    def matmul_pto_dtype(self):
        return self.DTYPE_CONFIG[self.matmul_dtype]["pto"]

    @property
    def out_pto_dtype(self):
        return self.DTYPE_CONFIG[self.out_dtype]["pto"]

    @classmethod
    def from_test_case(cls, case: dict) -> "CastMatmulConfig":
        return cls(
            shape=(case["m"], case["k"], case["n"]),
            cube_tile_shape=tuple(case["cubetileshape"]),
            a_vec_tile_shape=case["a_vectileshape"],
            b_vec_tile_shape=case["b_vectileshape"],
            view_shape=tuple(case["viewshape"]),
            a_input_dtype=case["a_input_dtype"],
            b_input_dtype=case["b_input_dtype"],
            matmul_dtype=case["matmul_dtype"],
            out_dtype=case["out_dtype"],
            a_cast=case["a_cast"],
            b_cast=case["b_cast"],
            a_trans=case.get("a_trans", False),
            b_trans=case.get("b_trans", False),
            batch_shape=tuple(case.get("batchshape", ())),
            batch_view_shape=tuple(case.get("batchviewshape", ())),
            enable_ksplit=case.get("enableksplit", False),
        )

    @classmethod
    def get_torch_dtype(cls, dtype_str: str) -> torch.dtype:
        return cls.DTYPE_CONFIG[dtype_str]["torch"]

    @classmethod
    def get_tolerance(cls, dtype_str: str) -> tuple[float, float]:
        info = cls.DTYPE_CONFIG[dtype_str]
        return info["atol"], info["rtol"]


CAST_RIGHT_MATMUL_TESTS = [
    {
        "id": "CM01",
        "name": "fp32_to_fp16_matmul_out_fp16",
        "desc": "B矩阵FP32输入Cast为FP16后Matmul,FP16输出",
        "m": 128,
        "k": 512,
        "n": 128,
        "a_input_dtype": "DT_FP16",
        "b_input_dtype": "DT_FP32",
        "matmul_dtype": "DT_FP16",
        "out_dtype": "DT_FP16",
        "a_cast": False,
        "b_cast": True,
        "a_trans": False,
        "b_trans": True,
        "enableksplit": False,
        "viewshape": [64, 320],
        "cubetileshape": [[64, 64], [80, 80], [320, 320]],
        "a_vectileshape": [384, 80],
        "b_vectileshape": [640, 128],
        "extend_params": {},
        "products": ["950"],
    },
]

CAST_LEFT_MATMUL_TESTS = [
    {
        "id": "CM02",
        "name": "fp16_to_int8_matmul_out_int32",
        "desc": "A矩阵FP16输入Cast为INT8后Matmul,INT32输出",
        "m": 16,
        "k": 256,
        "n": 256,
        "a_input_dtype": "DT_FP16",
        "b_input_dtype": "DT_INT8",
        "matmul_dtype": "DT_INT8",
        "out_dtype": "DT_INT32",
        "a_cast": True,
        "b_cast": False,
        "a_trans": False,
        "b_trans": False,
        "enableksplit": False,
        "viewshape": [176, 128],
        "cubetileshape": [[176, 176], [128, 128], [128, 128]],
        "a_vectileshape": [88, 64],
        "b_vectileshape": [128, 128],
        "extend_params": {},
        "products": ["950"],
    },
]

CAST_BOTH_MATMUL_TESTS = [
    {
        "id": "CM03",
        "name": "both_fp16_to_fp32_matmul_out_fp32",
        "desc": "双输入FP16均Cast为FP32后Matmul,FP32输出",
        "m": 128,
        "k": 224,
        "n": 224,
        "a_input_dtype": "DT_FP16",
        "b_input_dtype": "DT_FP16",
        "matmul_dtype": "DT_FP32",
        "out_dtype": "DT_FP32",
        "a_cast": True,
        "b_cast": True,
        "a_trans": False,
        "b_trans": False,
        "enableksplit": False,
        "viewshape": [128, 128],
        "cubetileshape": [[128, 128], [32, 32], [128, 128]],
        "a_vectileshape": [64, 32],
        "b_vectileshape": [32, 64],
        "extend_params": {},
        "products": ["950"],
    },
]



@dataclass
class ScaledMmUb2L1Config:
    m: int
    k: int
    n: int
    view_shape: tuple[int, int]
    cube_tile_shape: tuple[list, list, list]
    b_vec_tile_shape: list
    out_dtype: str

    DTYPE_CONFIG = {
        "DT_FP32": {"pto": pypto.DT_FP32, "torch": torch.float32, "atol": 1e-3, "rtol": 1e-3},
        "DT_FP16": {"pto": pypto.DT_FP16, "torch": torch.float16, "atol": 1e-3, "rtol": 1e-3},
    }

    @property
    def out_pto_dtype(self):
        return self.DTYPE_CONFIG[self.out_dtype]["pto"]

    @classmethod
    def from_test_case(cls, case: dict) -> "ScaledMmUb2L1Config":
        return cls(
            m=case["m"],
            k=case["k"],
            n=case["n"],
            view_shape=tuple(case["viewshape"]),
            cube_tile_shape=tuple(case["cubetileshape"]),
            b_vec_tile_shape=case["b_vectileshape"],
            out_dtype=case["out_dtype"],
        )

    @classmethod
    def get_torch_dtype(cls, dtype_str: str) -> torch.dtype:
        return cls.DTYPE_CONFIG[dtype_str]["torch"]

    @classmethod
    def get_tolerance(cls, dtype_str: str) -> tuple[float, float]:
        info = cls.DTYPE_CONFIG[dtype_str]
        return info["atol"], info["rtol"]


SCALED_MM_UB2L1_TESTS = [
    {
        "id": "SMU01",
        "name": "cast_fp32_to_fp8e4m3_scaledmm_k_aligned_ub2l1",
        "desc": "B矩阵FP32输入Cast为FP8E4M3后ScaledMM,K=128按64对齐,走UB2L1直通",
        "m": 128,
        "k": 128,
        "n": 128,
        "viewshape": [64, 64],
        "cubetileshape": [[64, 128], [64, 128], [64, 64]],
        "b_vectileshape": [16, 128],
        "out_dtype": "DT_FP32",
        "products": ["950"],
    },
    {
        "id": "SMU02",
        "name": "cast_fp32_to_fp8e4m3_scaledmm_k_not_aligned_ddr_fallback",
        "desc": "B矩阵FP32输入Cast为FP8E4M3后ScaledMM,K=96未按64对齐,UB2L1回退DDR搬运",
        "m": 128,
        "k": 96,
        "n": 128,
        "viewshape": [64, 64],
        "cubetileshape": [[64, 128], [64, 128], [64, 64]],
        "b_vectileshape": [16, 96],
        "out_dtype": "DT_FP32",
        "products": ["950"],
    },
]

CAST_3D_MATMUL_TESTS = [
    {
        "id": "CM04",
        "name": "fp16_to_int8_bmm3d_left_cast_out_int32",
        "desc": "3D A矩阵FP16输入Cast为INT8后BatchMatmul,INT32输出",
        "batchshape": [5],
        "m": 674,
        "k": 288,
        "n": 923,
        "a_input_dtype": "DT_FP16",
        "b_input_dtype": "DT_INT8",
        "matmul_dtype": "DT_INT8",
        "out_dtype": "DT_INT32",
        "a_cast": True,
        "b_cast": False,
        "a_trans": False,
        "b_trans": False,
        "enableksplit": False,
        "batchviewshape": [3],
        "viewshape": [128, 128],
        "cubetileshape": [[64, 64], [64, 64], [64, 64]],
        "a_vectileshape": [1, 128, 128],
        "b_vectileshape": [1, 128, 128],
        "extend_params": {},
        "products": ["950"],
    },
]

CAST_4D_MATMUL_TESTS = [
    {
        "id": "CM05",
        "name": "both_fp32_to_bf16_bmm4d_trans_a_out_fp32",
        "desc": "4D双输入FP32均Cast为BF16后BatchMatmul+A转置,FP32输出",
        "batchshape": [3, 4],
        "m": 422,
        "k": 320,
        "n": 451,
        "a_input_dtype": "DT_FP32",
        "b_input_dtype": "DT_FP32",
        "matmul_dtype": "DT_BF16",
        "out_dtype": "DT_FP32",
        "a_cast": True,
        "b_cast": True,
        "a_trans": True,
        "b_trans": False,
        "enableksplit": True,
        "batchviewshape": [2, 3],
        "viewshape": [128, 256],
        "cubetileshape": [[128, 128], [128, 128], [128, 128]],
        "a_vectileshape": [1, 1, 64, 64],
        "b_vectileshape": [1, 1, 64, 64],
        "extend_params": {},
        "products": ["950"],
    },
]
