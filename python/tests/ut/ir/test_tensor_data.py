# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pypto
from pypto.pil.compile_pipeline import compile_new_ir

from .test_common import ssa_verify


def grouped_matmul(
    tensor_x: pypto.Tensor[[pypto.DYNAMIC, ...]],
    tensor_w: pypto.Tensor[[...]],
    output_tensor: pypto.Tensor[[pypto.DYNAMIC, ...]],
):
    m, k = tensor_x.shape
    n = tensor_w.shape[1]
    pypto.set_vec_tile_shapes(32, 32)
    m_tile = 32
    m_loop = (m + m_tile - 1) // m_tile

    assert isinstance(k, int)
    assert isinstance(n, int)

    for _ in pypto.loop(1):
        group_list = pypto.arange(100)

    for m_idx in pypto.loop(m_loop, name="LOOP_GROUP", idx_name="m_loop"):

        def inner():
            actual_seq_len = group_list[m_idx]

            valid_shape1 = [pypto.min(m_tile, m - m_idx * m_tile), actual_seq_len]
            m_view = pypto.view(tensor_x, shape=[m_tile, k], offsets=[m_idx * m_tile, 0], valid_shape=valid_shape1)

            valid_shape_2 = [actual_seq_len, n]
            w_view = pypto.view(tensor_w, shape=[k, n], offsets=[0, 0], valid_shape=valid_shape_2)

            pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
            matmul_out = pypto.matmul(m_view, w_view, out_dtype=pypto.DT_FP32)

            output_tensor[m_idx * m_tile:(m_idx + 1) * m_tile, :] = matmul_out

        inner()


def test_get_tensor_data():

    m = pypto.SymbolicScalar("m")
    n = 256
    k = 128
    x = pypto.Tensor([m, k], pypto.DT_BF16, "x")
    y = pypto.Tensor([k, n], pypto.DT_BF16, "y")
    out = pypto.Tensor([m, n], pypto.DT_FP32, "out")

    func = compile_new_ir(grouped_matmul, x, y, out, create_new_logical_tensor = True)
    ssa_verify(func, "get_tensor_data")
