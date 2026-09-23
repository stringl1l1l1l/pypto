# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check scalar values and following pointers/dimensions across the actual ASC launch ABI."""

import os
import struct

import pypto_pro.language as pl
import pytest
import torch
import torch_npu  # noqa: F401 -- registers torch.npu


@pl.jit()
def scalar_launch_abi_kernel(
    flag: pl.DT_BOOL,
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    s8: pl.DT_INT8,
    u8: pl.DT_UINT8,
    s16: pl.DT_INT16,
    u16: pl.DT_UINT16,
    s32: pl.DT_INT32,
    u32: pl.DT_UINT32,
    value: pl.DT_FP32,
    s64: pl.DT_INT64,
    u64: pl.DT_UINT64,
    out: pl.Tensor[[10], pl.DT_INT64],
    real_out: pl.Tensor[[1], pl.DT_FP32],
    unsigned_out: pl.Tensor[[1], pl.DT_UINT64],
):
    with pl.section_vector():
        pl.setval(out, 0, flag)
        pl.setval(out, 1, s8)
        pl.setval(out, 2, u8)
        pl.setval(out, 3, s16)
        pl.setval(out, 4, u16)
        pl.setval(out, 5, s32)
        pl.setval(out, 6, u32)
        pl.setval(out, 7, s64)
        pl.setval(out, 8, x.shape[0])
        pl.setval(out, 9, x.shape[1])
        pl.setval(real_out, 0, value)
        pl.setval(unsigned_out, 0, u64)


@pytest.mark.soc("950")
@pytest.mark.parametrize("capture", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("upper,shape,float_bits", [
    (False, (65, 96), 0x80000000),
    (True, (129, 33), 0x7FC12345),
    (False, (33, 129), 0xBFA00000),
])
def test_scalar_launch_abi(capture, upper, shape, float_bits):
    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", "0")))
    x = torch.zeros(shape, dtype=torch.float32, device="npu")
    out = torch.full((10,), -17, dtype=torch.int64, device="npu")
    real_out = torch.full((1,), -17, dtype=torch.float32, device="npu")
    unsigned_out = torch.tensor([17], dtype=torch.uint64).npu()
    signed = [(1 << (bits - 1)) - 1 if upper else -(1 << (bits - 1)) for bits in (8, 16, 32, 64)]
    unsigned = [(1 << bits) - 1 if upper else 0 for bits in (8, 16, 32, 64)]
    value = struct.unpack("f", struct.pack("I", float_bits))[0]
    args = (upper, x, signed[0], unsigned[0], signed[1], unsigned[1], signed[2], unsigned[2],
            value, signed[3], unsigned[3], out, real_out, unsigned_out)
    expected = [int(upper), signed[0], unsigned[0], signed[1], unsigned[1], signed[2], unsigned[2],
                signed[3], *shape]

    def check():
        torch.npu.synchronize()
        assert out.cpu().tolist() == expected
        assert real_out.cpu().view(torch.uint32).item() == float_bits
        assert unsigned_out.cpu().item() == unsigned[3]

    scalar_launch_abi_kernel[1](*args)
    check()
    if capture:
        stream = torch.npu.Stream()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            scalar_launch_abi_kernel[1](*args)
        for _ in range(2):
            # A warmup or prior replay must not conceal a missing graph write.
            out.fill_(-17)
            real_out.fill_(-17)
            unsigned_out.copy_(torch.tensor([17], dtype=torch.uint64))
            torch.npu.synchronize()
            graph.replay()
            check()
