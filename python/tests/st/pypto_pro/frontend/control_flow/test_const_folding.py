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

"""Element-wise add on Ascend A5 using make_tile_group double buffer.

Shape: [M, N] with M=8192, N=4096, each core processes 128*4096.
Uses make_tile_group with auto_mutex for synchronization (no manual sync_src/sync_dst).
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

_FP32_INT32 = (pl.DT_FP32, pl.DT_INT32)

TILE_M = 128
TILE_N = 128
PAD_MODE = 1
mutex_ids23 = [1, 2]
A = 0x1000


def _addr_b(dtype, switch):
    return 0x10000 if dtype in _FP32_INT32 and switch is True else 0x10000


@pl.jit(auto_mutex=True)
def add_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec,
                            pad=pl.TilePad.zero if PAD_MODE else 0)
    a = 0x0000
    addrs = a
    addrs_real = addrs
    mutex_ids23 = [2, 3]
    switch = True
    dtype_ = pl.DT_FP16
    a_db = pl.make_tile_group(type=tile_type, addrs=addrs_real, mutex_ids=[0, 1])  # 地址手动指定， 使能double buffer
    b_db = pl.make_tile_group(type=tile_type, addrs=_addr_b(dtype_, switch), mutex_ids=mutex_ids23)
    c_db = pl.make_tile_group(type=tile_type, addrs=0x20000, mutex_ids=[30, 31])
    with pl.section_vector():
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx()
        m_tile_num = x.shape[0] // TILE_M
        n_tile_num = x.shape[1] // TILE_N
        for i in pl.range(core_id, m_tile_num, num_cores):
            if i in mutex_ids23:
                pl.printf("i = %d", i)
            for j in pl.range(0, n_tile_num, 1):
                tile_a = a_db.next()  # 自动选择下一块buffer，并插入对应的同步
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])  # 根据坐标自动计算offset
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_add():
    device = ST_DEVICE
    torch.npu.set_device(device)

    m_size = 8192
    n_size = 4096
    # 请求不得超过流预算；按 M tile 数与 vector 核预算取小。
    vector_budget = torch.npu.get_stream_limit(torch.npu.current_stream())["vector_core_num"]
    num_cores = min(m_size // 128, vector_budget)

    torch.manual_seed(0)
    dtype = torch.float16

    x = torch.rand([m_size, n_size], device=device, dtype=dtype)
    y = torch.rand([m_size, n_size], device=device, dtype=dtype)
    z = torch.empty([m_size, n_size], device=device, dtype=dtype)

    add_kernel[None, num_cores](x, y, z)
    torch.npu.synchronize()

    z_ref = x + y
    torch.testing.assert_close(z, z_ref)
    logging.info("test_add [%d, %d] passed!", m_size, n_size)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_add()
    logging.info("\nAll tests passed!")
