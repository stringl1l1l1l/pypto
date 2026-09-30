# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 64


@pl.vector_function(mode="simt", max_threads=1)
def bitcast_loaded_gm_value(
    src: pl.Tensor[[1, 1], pl.DT_INT32],
    dst: pl.Tensor[[1, 1], pl.DT_FP32],
):
    dst[0, 0] = pl.simt.bitcast(src[0, 0], pl.DT_FP32)


@pl.jit(auto_mutex=True, arch="3510")
def simt_bitcast_loaded_gm_value(
    src: pl.Tensor[[1, 1], pl.DT_INT32],
    dst: pl.Tensor[[1, 1], pl.DT_FP32],
):
    with pl.section_vector():
        bitcast_loaded_gm_value[1](src, dst)


@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def bitcast_all_scalar_pairs(
    src_int16: pl.Tensor[[1, ELEMENTS], pl.DT_INT16],
    src_uint16: pl.Tensor[[1, ELEMENTS], pl.DT_UINT16],
    src_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    src_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    src_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    src_uint32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    src_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_fp16: pl.Tensor[[2, ELEMENTS], pl.DT_FP16],
    out_bf16: pl.Tensor[[2, ELEMENTS], pl.DT_BF16],
    out_int16: pl.Tensor[[2, ELEMENTS], pl.DT_INT16],
    out_uint16: pl.Tensor[[2, ELEMENTS], pl.DT_UINT16],
    out_fp32: pl.Tensor[[2, ELEMENTS], pl.DT_FP32],
    out_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_uint32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
):
    tid = pl.simt.linear_thread_idx()
    value_int16 = pl.simt.cast(src_int16[0, tid], pl.DT_INT16)
    value_uint16 = pl.simt.cast(src_uint16[0, tid], pl.DT_UINT16)
    value_int32 = pl.simt.cast(src_int32[0, tid], pl.DT_INT32)
    value_uint32 = pl.simt.cast(src_uint32[0, tid], pl.DT_UINT32)
    out_fp16[0, tid] = pl.simt.bitcast(value_int16, pl.DT_FP16)
    out_fp16[1, tid] = pl.simt.bitcast(value_uint16, pl.DT_FP16)
    out_bf16[0, tid] = pl.simt.bitcast(value_int16, pl.DT_BF16)
    out_bf16[1, tid] = pl.simt.bitcast(value_uint16, pl.DT_BF16)
    out_int16[0, tid] = pl.simt.bitcast(src_fp16[0, tid], pl.DT_INT16)
    out_int16[1, tid] = pl.simt.bitcast(src_bf16[0, tid], pl.DT_INT16)
    out_uint16[0, tid] = pl.simt.bitcast(src_fp16[0, tid], pl.DT_UINT16)
    out_uint16[1, tid] = pl.simt.bitcast(src_bf16[0, tid], pl.DT_UINT16)
    out_fp32[0, tid] = pl.simt.bitcast(value_int32, pl.DT_FP32)
    out_fp32[1, tid] = pl.simt.bitcast(value_uint32, pl.DT_FP32)
    out_int32[0, tid] = pl.simt.bitcast(src_fp32[0, tid], pl.DT_INT32)
    out_uint32[0, tid] = pl.simt.bitcast(src_fp32[0, tid], pl.DT_UINT32)


@pl.jit()
def simt_bitcast_all_scalar_pairs(
    src_int16: pl.Tensor[[1, ELEMENTS], pl.DT_INT16],
    src_uint16: pl.Tensor[[1, ELEMENTS], pl.DT_UINT16],
    src_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    src_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    src_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    src_uint32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    src_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_fp16: pl.Tensor[[2, ELEMENTS], pl.DT_FP16],
    out_bf16: pl.Tensor[[2, ELEMENTS], pl.DT_BF16],
    out_int16: pl.Tensor[[2, ELEMENTS], pl.DT_INT16],
    out_uint16: pl.Tensor[[2, ELEMENTS], pl.DT_UINT16],
    out_fp32: pl.Tensor[[2, ELEMENTS], pl.DT_FP32],
    out_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_uint32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
):
    with pl.section_vector():
        bitcast_all_scalar_pairs[ELEMENTS](
            src_int16,
            src_uint16,
            src_fp16,
            src_bf16,
            src_int32,
            src_uint32,
            src_fp32,
            out_fp16,
            out_bf16,
            out_int16,
            out_uint16,
            out_fp32,
            out_int32,
            out_uint32,
        )


def _signed_bits(values, bits, dtype):
    signed = [value if value < (1 << (bits - 1)) else value - (1 << bits) for value in values]
    return torch.tensor(signed, dtype=dtype).repeat(ELEMENTS // len(values)).reshape(1, ELEMENTS)


def _bits16(value):
    return value.contiguous().view(torch.int16)


def _bits32(value):
    return value.contiguous().view(torch.int32)


@pytest.mark.soc("950")
def test_bitcast_accepts_loaded_gm_value(a5_device):
    src = torch.tensor([[0x3F800000]], dtype=torch.int32)
    dst = torch.empty((1, 1), dtype=torch.float32, device=a5_device)

    simt_bitcast_loaded_gm_value[None, 1](src.to(a5_device), dst)
    torch.npu.synchronize()

    assert torch.equal(_bits32(dst.cpu()), src)


@pytest.mark.soc("950")
def test_bitcast_preserves_all_supported_scalar_bit_patterns(a5_device):
    bits16 = _signed_bits(
        [0x0000, 0x3C00, 0x7C00, 0x7E00, 0x8000, 0xBC00, 0xFC00, 0xFFFF],
        16,
        torch.int16,
    )
    bits32 = _signed_bits(
        [
            0x00000000,
            0x3F800000,
            0x7F800000,
            0x7FC00000,
            0x80000000,
            0xBF800000,
            0xFF800000,
            0xFFFFFFFF,
        ],
        32,
        torch.int32,
    )
    sources = (
        bits16,
        bits16.view(torch.uint16),
        bits16.view(torch.float16),
        bits16.view(torch.bfloat16),
        bits32,
        bits32.view(torch.uint32),
        bits32.view(torch.float32),
    )
    outputs = (
        torch.empty((2, ELEMENTS), dtype=torch.float16, device=a5_device),
        torch.empty((2, ELEMENTS), dtype=torch.bfloat16, device=a5_device),
        torch.empty((2, ELEMENTS), dtype=torch.int16, device=a5_device),
        torch.empty((2, ELEMENTS), dtype=torch.uint16, device=a5_device),
        torch.empty((2, ELEMENTS), dtype=torch.float32, device=a5_device),
        torch.empty((1, ELEMENTS), dtype=torch.int32, device=a5_device),
        torch.empty((1, ELEMENTS), dtype=torch.uint32, device=a5_device),
    )

    simt_bitcast_all_scalar_pairs(*(source.to(a5_device) for source in sources), *outputs)
    torch.npu.synchronize()

    fp16, bf16, int16, uint16, fp32, int32, uint32 = (output.cpu() for output in outputs)
    assert torch.equal(_bits16(fp16[0]), _bits16(sources[0][0]))
    assert torch.equal(_bits16(fp16[1]), _bits16(sources[1][0]))
    assert torch.equal(_bits16(bf16[0]), _bits16(sources[0][0]))
    assert torch.equal(_bits16(bf16[1]), _bits16(sources[1][0]))
    assert torch.equal(_bits16(int16[0]), _bits16(sources[2][0]))
    assert torch.equal(_bits16(int16[1]), _bits16(sources[3][0]))
    assert torch.equal(_bits16(uint16[0]), _bits16(sources[2][0]))
    assert torch.equal(_bits16(uint16[1]), _bits16(sources[3][0]))
    assert torch.equal(_bits32(fp32[0]), _bits32(sources[4][0]))
    assert torch.equal(_bits32(fp32[1]), _bits32(sources[5][0]))
    assert torch.equal(_bits32(int32[0]), _bits32(sources[6][0]))
    assert torch.equal(_bits32(uint32[0]), _bits32(sources[6][0]))
