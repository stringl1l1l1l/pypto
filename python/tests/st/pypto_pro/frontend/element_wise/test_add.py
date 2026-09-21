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

The second kernel is the same computation with its block ops spelled as bare imported
names instead of ``pl.xxx``, which is the spelling the parser resolves through the API
declarations themselves -- see test_add_bare_name_aliases.
"""

import logging
import os

import pypto_pro.language as pl
from pypto_pro.language import add as add_op
from pypto_pro.language import load_tile as load_tile_op
from pypto_pro.language import store_tile as store_tile_op
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


TILE_M = 128
TILE_N = 128


@pl.jit(auto_mutex=True)
def add_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)

    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])  # 地址手动指定， 使能double buffer
    b_db = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x20000, mutex_ids=[30, 31])

    with pl.section_vector():
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx()
        m_tile_num = x.shape[0] // TILE_M
        n_tile_num = x.shape[1] // TILE_N

        for i in pl.range(core_id, m_tile_num, num_cores):
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
    # 请求不得超过流预算（vector 核 56）；超出会直接报错而非钳制。
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


@pl.jit(auto_mutex=True)
def add_kernel_bare_name_aliases(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    """add_kernel with load_tile/add/store_tile called through bare imported names."""
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)

    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x20000, mutex_ids=[30, 31])

    with pl.section_vector():
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx()
        m_tile_num = x.shape[0] // TILE_M
        n_tile_num = x.shape[1] // TILE_N

        for i in pl.range(core_id, m_tile_num, num_cores):
            for j in pl.range(0, n_tile_num, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                load_tile_op(tile_a, x, [i, j])
                load_tile_op(tile_b, y, [i, j])
                add_op(tile_c, tile_a, tile_b)
                store_tile_op(z, tile_c, [i, j])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_add_bare_name_aliases():
    """A block op imported under a bare name must lower to that op on device.

    The parser resolves a bare name against the API declarations so that an alias such as
    ``m = vf.create_mask`` cannot slip a VF instruction past the execution-domain check.
    Routing the name to parse_op_call is what makes that resolution reachable, and it also
    decides which op the call becomes -- every other kernel in the suite spells ops as
    ``pl.xxx``, so nothing else would notice a bare name resolving to the wrong op. z starts
    filled with a sentinel, so a kernel that parsed but emitted no work fails here too.
    """
    device = ST_DEVICE
    torch.npu.set_device(device)

    m_size = 1024
    n_size = 512
    num_cores = m_size // TILE_M

    torch.manual_seed(0)
    dtype = torch.float16

    x = torch.rand([m_size, n_size], device=device, dtype=dtype)
    y = torch.rand([m_size, n_size], device=device, dtype=dtype)
    z = torch.full([m_size, n_size], -1.0, device=device, dtype=dtype)

    add_kernel_bare_name_aliases[None, num_cores](x, y, z)
    torch.npu.synchronize()

    z_ref = x + y
    torch.testing.assert_close(z, z_ref)
    logging.info("test_add_bare_name_aliases [%d, %d] passed!", m_size, n_size)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_add()
    test_add_bare_name_aliases()
    logging.info("\nAll tests passed!")
