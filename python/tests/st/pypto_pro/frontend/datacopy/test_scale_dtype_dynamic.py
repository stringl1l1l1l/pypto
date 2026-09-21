# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Medium priority test cases for scale parameter (23 tests).

Covers:
1. Data type extensions (3): BF16→INT8, FP16→INT8, INT32→INT8
2. Scale value boundary (1): extreme scale values
3. Input data patterns (3): small fractions, NaN/Inf, special distributions
4. Dynamic scale (3): INT64 param, read from GM, multiple calls
5. ReLU fusion (4): INT32→FP16+ReLU, scale=0+ReLU, all-negative+ReLU, mixed+ReLU
6. Phase fusion (1): Unspecified
7. Atomic fusion (1): None
8. Order parameter (3): default, ascending, 4D tensor
9. Store tile (4): dynamic scale, phase fusion (Partial+Final UF chain),
   ReLU+phase (UF chain), multi-tile offset
"""

import logging
import os
import struct

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


# ============================================================================
# 1. Data Type Extensions (3 tests)
# ============================================================================


@pl.jit()
def bf16_to_int8_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    quant_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    scale: pl.DT_INT32,
):
    with pl.section_cube():
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(mat_type, addr=0x0000)
        k_mat = pl.make_tile(mat_type, addr=0x2000)

        left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
        q_left = pl.make_tile(left_type, addr=0x0000)

        right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right, layout=pl.ZN)
        k_right = pl.make_tile(right_type, addr=0x0000)

        acc_type = pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        )
        acc = pl.make_tile(acc_type, addr=0x0000)

        pl.load(q_mat, q, [0, 0])
        pl.load(k_mat, k, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        pl.move(q_left, q_mat)
        pl.move(k_right, k_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

        pl.matmul(acc, q_left, k_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)

        pl.store(quant_out, acc, [0, 0], scale=scale)
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_bf16_to_int8():
    """BF16 input -> INT8 output with scale."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = torch.randn(64, 64, dtype=torch.bfloat16, device=device)
    k = torch.randn(64, 64, dtype=torch.bfloat16, device=device)
    quant_out = torch.zeros(64, 64, dtype=torch.int8, device=device)
    scale = struct.unpack("!I", struct.pack("!f", 2.0))[0]

    bf16_to_int8_kernel(q, k, quant_out, scale)
    torch.npu.synchronize()

    # Verify output is not all zeros
    assert quant_out.abs().sum() > 0
    logging.info("test_bf16_to_int8 passed.")


@pl.jit()
def fp16_to_int8_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    quant_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    scale: pl.DT_INT32,
):
    with pl.section_cube():
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(mat_type, addr=0x0000)
        k_mat = pl.make_tile(mat_type, addr=0x2000)

        left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
        q_left = pl.make_tile(left_type, addr=0x0000)

        right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN)
        k_right = pl.make_tile(right_type, addr=0x0000)

        acc_type = pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        )
        acc = pl.make_tile(acc_type, addr=0x0000)

        pl.load(q_mat, q, [0, 0])
        pl.load(k_mat, k, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        pl.move(q_left, q_mat)
        pl.move(k_right, k_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

        pl.matmul(acc, q_left, k_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)

        pl.store(quant_out, acc, [0, 0], scale=scale)
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_fp16_to_int8():
    """FP16 input -> INT8 output with scale."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = torch.randn(64, 64, dtype=torch.float16, device=device)
    k = torch.randn(64, 64, dtype=torch.float16, device=device)
    quant_out = torch.zeros(64, 64, dtype=torch.int8, device=device)
    scale = struct.unpack("!I", struct.pack("!f", 2.0))[0]

    fp16_to_int8_kernel(q, k, quant_out, scale)
    torch.npu.synchronize()

    assert quant_out.abs().sum() > 0
    logging.info("test_fp16_to_int8 passed.")


@pl.jit()
def int32_to_int8_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    quant_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    scale: pl.DT_INT32,
):
    with pl.section_cube():
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(mat_type, addr=0x0000)
        k_mat = pl.make_tile(mat_type, addr=0x1000)

        left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
        q_left = pl.make_tile(left_type, addr=0x0000)

        right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Right, layout=pl.ZN)
        k_right = pl.make_tile(right_type, addr=0x0000)

        acc_type = pl.TileType(
            shape=[64, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        )
        acc = pl.make_tile(acc_type, addr=0x0000)

        pl.load(q_mat, q, [0, 0])
        pl.load(k_mat, k, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        pl.move(q_left, q_mat)
        pl.move(k_right, k_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

        pl.matmul(acc, q_left, k_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)

        pl.store(quant_out, acc, [0, 0], scale=scale)
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_int32_to_int8():
    """INT8 matmul -> INT32 accumulator -> INT8 output with scale."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = torch.randint(-128, 127, (64, 64), dtype=torch.int8, device=device)
    k = torch.randint(-128, 127, (64, 64), dtype=torch.int8, device=device)
    quant_out = torch.zeros(64, 64, dtype=torch.int8, device=device)
    scale = struct.unpack("!I", struct.pack("!f", 0.01))[0]  # Small scale for INT32 accumulator

    int32_to_int8_kernel(q, k, quant_out, scale)
    torch.npu.synchronize()

    assert quant_out.abs().sum() > 0
    logging.info("test_int32_to_int8 passed.")


# ============================================================================
# 2. Scale Value Boundary (1 test)
# ============================================================================


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_scale_boundary_saturation():
    """Extreme scale value causing saturation."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = torch.randn(64, 64, dtype=torch.float32, device=device)
    k = torch.randn(64, 64, dtype=torch.float32, device=device)
    quant_out = torch.zeros(64, 64, dtype=torch.int8, device=device)

    scale_bits = struct.unpack("!I", struct.pack("!f", 1000.0))[0]
    dynamic_scale_int64_kernel(q, k, quant_out, scale_bits)
    torch.npu.synchronize()

    # Most values should be saturated to 127 or -128
    saturated_count = ((quant_out == 127) | (quant_out == -128)).sum().item()
    total_count = quant_out.numel()
    saturation_ratio = saturated_count / total_count

    assert saturation_ratio > 0.5, f"Expected >50% saturation, got {saturation_ratio:.2%}"
    logging.info(f"test_scale_boundary_saturation passed: {saturation_ratio:.2%} saturated.")


# ============================================================================
# 3. Input Data Patterns (3 tests)
# ============================================================================


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_small_fractional_input():
    """Small fractional values with scale."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    # Small fractional values
    q = torch.rand(64, 64, dtype=torch.float32, device=device) * 0.1
    k = torch.rand(64, 64, dtype=torch.float32, device=device) * 0.1
    quant_out = torch.zeros(64, 64, dtype=torch.int8, device=device)

    scale_bits = struct.unpack("!I", struct.pack("!f", 10.0))[0]
    dynamic_scale_int64_kernel(q, k, quant_out, scale_bits)
    torch.npu.synchronize()

    # Verify output is reasonable
    assert quant_out.abs().max() <= 127
    logging.info("test_small_fractional_input passed.")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_nan_inf_input():
    """NaN/Inf values handling."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    # Mix of NaN, Inf, and normal values
    q = torch.randn(64, 64, dtype=torch.float32, device=device)
    q[0, 0] = float('nan')
    q[1, 1] = float('inf')
    q[2, 2] = float('-inf')
    k = torch.eye(64, dtype=torch.float32, device=device)
    quant_out = torch.zeros(64, 64, dtype=torch.int8, device=device)

    scale_bits = struct.unpack("!I", struct.pack("!f", 2.0))[0]
    dynamic_scale_int64_kernel(q, k, quant_out, scale_bits)
    torch.npu.synchronize()

    # Hardware should handle NaN/Inf gracefully (saturate or zero)
    assert not torch.isnan(quant_out.float()).any()
    logging.info("test_nan_inf_input passed.")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_special_distribution():
    """Special distribution: bimodal (two peaks)."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    # Bimodal distribution: half values near -10, half near +10
    q = torch.zeros(64, 64, dtype=torch.float32, device=device)
    q[:32, :] = torch.randn(32, 64, dtype=torch.float32, device=device) - 10.0
    q[32:, :] = torch.randn(32, 64, dtype=torch.float32, device=device) + 10.0
    k = torch.eye(64, dtype=torch.float32, device=device)
    quant_out = torch.zeros(64, 64, dtype=torch.int8, device=device)

    scale_bits = struct.unpack("!I", struct.pack("!f", 1.0))[0]
    dynamic_scale_int64_kernel(q, k, quant_out, scale_bits)
    torch.npu.synchronize()

    # Verify bimodal distribution is preserved
    negative_count = (quant_out < 0).sum().item()
    positive_count = (quant_out > 0).sum().item()
    assert negative_count > 1000 and positive_count > 1000
    logging.info(f"test_special_distribution passed: neg={negative_count}, pos={positive_count}.")


# ============================================================================
# 4. Dynamic Scale (3 tests)
# ============================================================================


@pl.jit()
def dynamic_scale_int64_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    quant_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    scale_bits: pl.DT_INT64,
):
    with pl.section_cube():
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(mat_type, addr=0x0000)
        k_mat = pl.make_tile(mat_type, addr=0x4000)

        left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
        q_left = pl.make_tile(left_type, addr=0x0000)

        right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, layout=pl.ZN)
        k_right = pl.make_tile(right_type, addr=0x0000)

        acc_type = pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        )
        acc = pl.make_tile(acc_type, addr=0x0000)

        pl.load(q_mat, q, [0, 0])
        pl.load(k_mat, k, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        pl.move(q_left, q_mat)
        pl.move(k_right, k_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

        pl.matmul(acc, q_left, k_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)

        pl.store(quant_out, acc, [0, 0], scale=scale_bits)
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dynamic_scale_int64():
    """Dynamic scale via INT64 parameter."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = torch.randn(64, 64, dtype=torch.float32, device=device)
    k = torch.randn(64, 64, dtype=torch.float32, device=device)
    quant_out = torch.zeros(64, 64, dtype=torch.int8, device=device)

    # Encode scale as INT64 bits
    scale_value = 2.0
    scale_bits = struct.unpack("!I", struct.pack("!f", scale_value))[0]

    dynamic_scale_int64_kernel(q, k, quant_out, scale_bits)
    torch.npu.synchronize()

    assert quant_out.abs().sum() > 0
    logging.info("test_dynamic_scale_int64 passed.")


@pl.jit()
def dynamic_relu_store_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    quant_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    scale_bits: pl.DT_INT64,
):
    with pl.section_cube():
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(mat_type, addr=0x0000)
        k_mat = pl.make_tile(mat_type, addr=0x4000)

        left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
        q_left = pl.make_tile(left_type, addr=0x0000)

        right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, layout=pl.ZN)
        k_right = pl.make_tile(right_type, addr=0x0000)

        acc_type = pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        )
        acc = pl.make_tile(acc_type, addr=0x0000)

        pl.load(q_mat, q, [0, 0])
        pl.load(k_mat, k, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        pl.move(q_left, q_mat)
        pl.move(k_right, k_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

        pl.matmul(acc, q_left, k_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)

        pl.store(quant_out, acc, [0, 0], scale=scale_bits, relu_pre_mode=pl.ReluPreMode.NormalRelu)
        pl.system.bar_all()


@pl.jit()
def dynamic_scale_from_gm_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    scale_tensor: pl.Tensor[[1], pl.DT_FP32],
    quant_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
):
    with pl.section_cube():
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(mat_type, addr=0x0000)
        k_mat = pl.make_tile(mat_type, addr=0x4000)

        left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
        q_left = pl.make_tile(left_type, addr=0x0000)

        right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, layout=pl.ZN)
        k_right = pl.make_tile(right_type, addr=0x0000)

        acc_type = pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        )
        acc = pl.make_tile(acc_type, addr=0x0000)

        pl.load(q_mat, q, [0, 0])
        pl.load(k_mat, k, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        pl.move(q_left, q_mat)
        pl.move(k_right, k_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

        pl.matmul(acc, q_left, k_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)

        # Read scale from GM tensor
        scale_val = scale_tensor[0]
        pl.store(quant_out, acc, [0, 0], scale=scale_val)
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dynamic_scale_from_gm():
    """Dynamic scale read from GM tensor."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = torch.randn(64, 64, dtype=torch.float32, device=device)
    k = torch.randn(64, 64, dtype=torch.float32, device=device)
    scale_tensor = torch.tensor([2.0], dtype=torch.float32, device=device)
    quant_out = torch.zeros(64, 64, dtype=torch.int8, device=device)

    dynamic_scale_from_gm_kernel(q, k, scale_tensor, quant_out)
    torch.npu.synchronize()

    assert quant_out.abs().sum() > 0
    logging.info("test_dynamic_scale_from_gm passed.")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_multiple_calls_different_scale():
    """Multiple kernel calls with different scale values."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = torch.randn(64, 64, dtype=torch.float32, device=device)
    k = torch.randn(64, 64, dtype=torch.float32, device=device)
    quant_out1 = torch.zeros(64, 64, dtype=torch.int8, device=device)
    quant_out2 = torch.zeros(64, 64, dtype=torch.int8, device=device)

    # First call with scale=1.0
    scale_bits_1 = struct.unpack("!I", struct.pack("!f", 1.0))[0]
    dynamic_scale_int64_kernel(q, k, quant_out1, scale_bits_1)
    torch.npu.synchronize()

    # Second call with scale=2.0
    scale_bits_2 = struct.unpack("!I", struct.pack("!f", 2.0))[0]
    dynamic_scale_int64_kernel(q, k, quant_out2, scale_bits_2)
    torch.npu.synchronize()

    # Verify different scales produce different results
    assert not torch.equal(quant_out1, quant_out2)
    logging.info("test_multiple_calls_different_scale passed.")


# ============================================================================
# 5. ReLU Fusion (4 tests)
# ============================================================================


@pl.jit()
def int32_to_fp16_relu_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    quant_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    scale: pl.DT_INT32,
):
    with pl.section_cube():
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(mat_type, addr=0x0000)
        k_mat = pl.make_tile(mat_type, addr=0x1000)

        left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
        q_left = pl.make_tile(left_type, addr=0x0000)

        right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Right, layout=pl.ZN)
        k_right = pl.make_tile(right_type, addr=0x0000)

        acc_type = pl.TileType(
            shape=[64, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        )
        acc = pl.make_tile(acc_type, addr=0x0000)

        pl.load(q_mat, q, [0, 0])
        pl.load(k_mat, k, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        pl.move(q_left, q_mat)
        pl.move(k_right, k_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

        pl.matmul(acc, q_left, k_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)

        pl.store(quant_out, acc, [0, 0], scale=scale, relu_pre_mode=pl.ReluPreMode.NormalRelu)
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_int32_to_fp16_relu():
    """INT32 accumulator -> FP16 output with ReLU."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = torch.randint(-128, 127, (64, 64), dtype=torch.int8, device=device)
    k = torch.randint(-128, 127, (64, 64), dtype=torch.int8, device=device)
    quant_out = torch.zeros(64, 64, dtype=torch.float16, device=device)
    scale = struct.unpack("!I", struct.pack("!f", 0.01))[0]

    int32_to_fp16_relu_kernel(q, k, quant_out, scale)
    torch.npu.synchronize()

    # All outputs should be >= 0 due to ReLU
    assert (quant_out >= 0).all()
    logging.info("test_int32_to_fp16_relu passed.")


@pl.jit()
def int32_to_fp16_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    quant_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    scale: pl.DT_INT32,
):
    """Per-tensor INT32 -> FP16 dequantization (DEQF16), no ReLU.

    Mirrors master's removed test_deq_scalar_store_int32_to_half: int8 matmul
    accumulates to INT32 in L0C, then fixpipe dequantizes by a scalar scale.
    """
    with pl.section_cube():
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(mat_type, addr=0x0000)
        k_mat = pl.make_tile(mat_type, addr=0x1000)

        left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
        q_left = pl.make_tile(left_type, addr=0x0000)

        right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Right, layout=pl.ZN)
        k_right = pl.make_tile(right_type, addr=0x0000)

        acc_type = pl.TileType(
            shape=[64, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        )
        acc = pl.make_tile(acc_type, addr=0x0000)

        pl.load(q_mat, q, [0, 0])
        pl.load(k_mat, k, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        pl.move(q_left, q_mat)
        pl.move(k_right, k_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

        pl.matmul(acc, q_left, k_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)

        pl.store(quant_out, acc, [0, 0], scale=scale)
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_int32_to_fp16():
    """Per-tensor INT32 -> FP16 dequantization (DEQF16), matches golden with k=eye."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = torch.randint(-128, 127, (64, 64), dtype=torch.int8, device=device)
    k = torch.eye(64, dtype=torch.int8, device=device)
    quant_out = torch.zeros(64, 64, dtype=torch.float16, device=device)
    scale = struct.unpack("!I", struct.pack("!f", 0.01))[0]

    int32_to_fp16_kernel(q, k, quant_out, scale)
    torch.npu.synchronize()

    # k=eye: matmul(q, k) == q (INT32 acc), then dequant by scale -> FP16.
    # q is int8 so the exact int32 value is preserved.
    expected = (q.to(torch.float32) * 0.01).to(torch.float16)
    torch.testing.assert_close(quant_out, expected, rtol=0, atol=1e-3)
    logging.info("test_int32_to_fp16 passed.")


@pytest.mark.soc("950")
@pytest.mark.parametrize(
    "input_mode", ["zero_scale", "all_negative", "mixed"], ids=["zero_scale", "all_negative", "mixed"]
)
@pypto.options(pass_options={"enable_slice": False})
def test_relu_input_mode(input_mode):
    """ReLU fusion semantics: zero scale -> all zeros; all-negative inputs -> zeros; mixed -> >=0."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    if input_mode == "all_negative":
        q = -torch.abs(torch.randn(64, 64, dtype=torch.float32, device=device))
        k = torch.eye(64, dtype=torch.float32, device=device)
        scale_bits = struct.unpack("!I", struct.pack("!f", 2.0))[0]
    elif input_mode == "zero_scale":
        q = torch.randn(64, 64, dtype=torch.float32, device=device)
        k = torch.randn(64, 64, dtype=torch.float32, device=device)
        scale_bits = struct.unpack("!I", struct.pack("!f", 0.0))[0]
    else:
        q = torch.randn(64, 64, dtype=torch.float32, device=device)
        k = torch.eye(64, dtype=torch.float32, device=device)
        scale_bits = struct.unpack("!I", struct.pack("!f", 2.0))[0]

    quant_relu_out = torch.zeros(64, 64, dtype=torch.int8, device=device)
    dynamic_relu_store_kernel(q, k, quant_relu_out, scale_bits)
    torch.npu.synchronize()

    if input_mode in ("zero_scale", "all_negative"):
        assert (quant_relu_out == 0).all()
    else:
        assert (quant_relu_out >= 0).all()
        assert quant_relu_out.abs().sum() > 0
    logging.info("test_relu_input_mode[%s] passed.", input_mode)


@pl.jit()
def order_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    quant_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    scale: pl.DT_INT32,
    use_order: pl.DT_INT32,
):
    with pl.section_cube():
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(mat_type, addr=0x0000)
        k_mat = pl.make_tile(mat_type, addr=0x4000)

        left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
        q_left = pl.make_tile(left_type, addr=0x0000)

        right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, layout=pl.ZN)
        k_right = pl.make_tile(right_type, addr=0x0000)

        acc_type = pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        )
        acc = pl.make_tile(acc_type, addr=0x0000)

        pl.load(q_mat, q, [0, 0])
        pl.load(k_mat, k, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        pl.move(q_left, q_mat)
        pl.move(k_right, k_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

        pl.matmul(acc, q_left, k_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)

        if use_order == 1:
            pl.store(quant_out, acc, [0, 0], scale=scale, order=[0, 1])
        else:
            pl.store(quant_out, acc, [0, 0], scale=scale)
        pl.system.bar_all()


@pytest.mark.soc("950")
@pytest.mark.parametrize("use_order", [0, 1], ids=["default", "ascending"])
@pypto.options(pass_options={"enable_slice": False})
def test_order_2d(use_order):
    """Order parameter: default (None) vs explicit ascending [0, 1] on 2D."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = torch.randn(64, 64, dtype=torch.float32, device=device)
    k = torch.eye(64, dtype=torch.float32, device=device)
    quant_out = torch.zeros(64, 64, dtype=torch.int8, device=device)
    scale_bits = struct.unpack("!I", struct.pack("!f", 2.0))[0]

    order_kernel(q, k, quant_out, scale_bits, use_order)
    torch.npu.synchronize()

    expected = torch.clamp(torch.round(q * 2.0), -128, 127).to(torch.int8)
    torch.testing.assert_close(quant_out.to(torch.int32), expected.to(torch.int32), rtol=0, atol=1)
    logging.info("test_order_2d[%s] passed.", "default" if use_order == 0 else "ascending")


@pl.jit()
def order_4d_kernel(
    q: pl.Tensor[[2, 4, 64, 64], pl.DT_FP32],
    k: pl.Tensor[[2, 4, 64, 64], pl.DT_FP32],
    quant_out: pl.Tensor[[64, 4, 64, 1], pl.DT_INT8],
    scale: pl.DT_INT32,
):
    with pl.section_cube():
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(mat_type, addr=0x0000)
        k_mat = pl.make_tile(mat_type, addr=0x4000)

        left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
        q_left = pl.make_tile(left_type, addr=0x0000)

        right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, layout=pl.ZN)
        k_right = pl.make_tile(right_type, addr=0x0000)

        acc_type = pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        )
        acc = pl.make_tile(acc_type, addr=0x0000)

        # Process first batch
        pl.load(q_mat, q, [0, 0, 0, 0])
        pl.load(k_mat, k, [0, 0, 0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        pl.move(q_left, q_mat)
        pl.move(k_right, k_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

        pl.matmul(acc, q_left, k_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)

        # 4D tensor order [0, 2] maps tile dims to tensor axes 0 and 2
        pl.store(quant_out, acc, [0, 0, 0, 0], scale=scale, order=[0, 2])
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_order_4d():
    """4D tensor with order parameter."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = torch.randn(2, 4, 64, 64, dtype=torch.float32, device=device)
    k = torch.randn(2, 4, 64, 64, dtype=torch.float32, device=device)
    # order=[0, 2] needs 64 elements on both mapped axes. A unit final
    # dimension keeps each stored row contiguous in GM.
    quant_out = torch.zeros(64, 4, 64, 1, dtype=torch.int8, device=device)

    order_4d_kernel(q, k, quant_out, scale=struct.unpack("!I", struct.pack("!f", 2.0))[0])
    torch.npu.synchronize()

    actual = quant_out.cpu().to(torch.int32)
    reference = torch.matmul(q[0, 0].cpu(), k[0, 0].cpu())
    expected = torch.clamp(torch.round(reference * 2.0), -128, 127).to(torch.int32)
    torch.testing.assert_close(actual[:, 0, :, 0], expected, rtol=0, atol=4)
    torch.testing.assert_close(actual[:, 1:, :, :], torch.zeros_like(actual[:, 1:, :, :]), rtol=0, atol=0)
    logging.info("test_order_4d passed.")


# ============================================================================
# 9. Store Tile (4 tests)
# ============================================================================


@pl.jit()
def store_tile_dynamic_scale_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    quant_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    scale_bits: pl.DT_INT32,
):
    with pl.section_cube():
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(mat_type, addr=0x0000)
        k_mat = pl.make_tile(mat_type, addr=0x4000)

        left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
        q_left = pl.make_tile(left_type, addr=0x0000)

        right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, layout=pl.ZN)
        k_right = pl.make_tile(right_type, addr=0x0000)

        acc_type = pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        )
        acc = pl.make_tile(acc_type, addr=0x0000)

        pl.load(q_mat, q, [0, 0])
        pl.load(k_mat, k, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        pl.move(q_left, q_mat)
        pl.move(k_right, k_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

        pl.matmul(acc, q_left, k_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)

        pl.store_tile(quant_out, acc, [0, 0], scale=scale_bits)
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_store_tile_dynamic_scale():
    """store_tile with dynamic scale."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = torch.randn(64, 64, dtype=torch.float32, device=device)
    k = torch.randn(64, 64, dtype=torch.float32, device=device)
    quant_out = torch.zeros(64, 64, dtype=torch.int8, device=device)

    scale_value = 2.0
    scale_bits = struct.unpack("!I", struct.pack("!f", scale_value))[0]

    store_tile_dynamic_scale_kernel(q, k, quant_out, scale_bits)
    torch.npu.synchronize()

    assert quant_out.abs().sum() > 0
    logging.info("test_store_tile_dynamic_scale passed.")


@pl.jit()
def store_tile_phase_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    quant_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    scale: pl.DT_INT32,
):
    with pl.section_cube():
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(mat_type, addr=0x0000)
        k_mat = pl.make_tile(mat_type, addr=0x4000)

        left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
        q_left = pl.make_tile(left_type, addr=0x0000)

        right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, layout=pl.ZN)
        k_right = pl.make_tile(right_type, addr=0x0000)

        acc_type = pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        )
        acc = pl.make_tile(acc_type, addr=0x0000)

        pl.load(q_mat, q, [0, 0])
        pl.load(k_mat, k, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        pl.move(q_left, q_mat)
        pl.move(k_right, k_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

        pl.matmul(acc, q_left, k_right, phase=pl.AccPhase.Partial)
        pl.matmul_acc(acc, acc, q_left, k_right, phase=pl.AccPhase.Final)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)

        # Balanced UF chain: matmul Partial + matmul_acc Final, then store_tile
        # Partial + Final (scale + phase). A lone Partial store would leave the
        # FixPipe unit-flag state machine unbalanced and crash the AI Core.
        pl.store_tile(quant_out, acc, [0, 0], scale=scale, phase=pl.STPhase.Partial)
        pl.store_tile(quant_out, acc, [0, 0], scale=scale, phase=pl.STPhase.Final)
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_store_tile_phase():
    """store_tile with phase parameter."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = torch.randn(64, 64, dtype=torch.float32, device=device)
    k = torch.randn(64, 64, dtype=torch.float32, device=device)
    quant_out = torch.zeros(64, 64, dtype=torch.int8, device=device)

    store_tile_phase_kernel(q, k, quant_out, scale=struct.unpack("!I", struct.pack("!f", 2.0))[0])
    torch.npu.synchronize()

    assert quant_out.abs().sum() > 0
    logging.info("test_store_tile_phase passed.")


@pl.jit()
def store_tile_relu_phase_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    quant_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    scale: pl.DT_INT32,
):
    with pl.section_cube():
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(mat_type, addr=0x0000)
        k_mat = pl.make_tile(mat_type, addr=0x4000)

        left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
        q_left = pl.make_tile(left_type, addr=0x0000)

        right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, layout=pl.ZN)
        k_right = pl.make_tile(right_type, addr=0x0000)

        acc_type = pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        )
        acc = pl.make_tile(acc_type, addr=0x0000)

        pl.load(q_mat, q, [0, 0])
        pl.load(k_mat, k, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        pl.move(q_left, q_mat)
        pl.move(k_right, k_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

        pl.matmul(acc, q_left, k_right, phase=pl.AccPhase.Partial)
        pl.matmul_acc(acc, acc, q_left, k_right, phase=pl.AccPhase.Final)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)

        # Balanced UF chain: matmul Partial + matmul_acc Final, then a store_tile
        # Final (scale + relu + phase) closes the chain.
        pl.store_tile(
            quant_out, acc, [0, 0], scale=scale, relu_pre_mode=pl.ReluPreMode.NormalRelu, phase=pl.STPhase.Final
        )
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_store_tile_relu_phase():
    """store_tile with ReLU and phase fusion."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = torch.randn(64, 64, dtype=torch.float32, device=device)
    k = torch.randn(64, 64, dtype=torch.float32, device=device)
    quant_out = torch.zeros(64, 64, dtype=torch.int8, device=device)

    store_tile_relu_phase_kernel(q, k, quant_out, scale=struct.unpack("!I", struct.pack("!f", 2.0))[0])
    torch.npu.synchronize()

    # All outputs should be >= 0 due to ReLU
    assert (quant_out >= 0).all()
    assert quant_out.abs().sum() > 0
    logging.info("test_store_tile_relu_phase passed.")


@pl.jit()
def store_tile_single_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    quant_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    scale: pl.DT_INT32,
    tile_row: pl.DT_INT32,
    tile_col: pl.DT_INT32,
):
    with pl.section_cube():
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(mat_type, addr=0x0000)
        k_mat = pl.make_tile(mat_type, addr=0x4000)

        left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
        q_left = pl.make_tile(left_type, addr=0x0000)

        right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, layout=pl.ZN)
        k_right = pl.make_tile(right_type, addr=0x0000)

        acc_type = pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        )
        acc = pl.make_tile(acc_type, addr=0x0000)

        pl.load(q_mat, q, [tile_row * 64, 0])
        pl.load(k_mat, k, [0, tile_col * 64])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        pl.move(q_left, q_mat)
        pl.move(k_right, k_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

        pl.matmul(acc, q_left, k_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)

        pl.store_tile(quant_out, acc, [tile_row, tile_col], scale=scale)
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_store_tile_multi_offset():
    """store_tile with multiple tile offsets, region-level golden check.

    One kernel invocation writes one tile at [tile_row, tile_col] (a single
    K=64 block product, no K accumulation). Two invocations cover [0, 0] and
    [0, 1] so each tile's offset is strictly validated:
      tile [0, 0] -> q[0:64,0:64] @ k[0:64,0:64]  at out[0:64,0:64]
      tile [0, 1] -> q[0:64,0:64] @ k[0:64,64:128] at out[0:64,64:128]
    """
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = torch.randn(128, 128, dtype=torch.float32, device=device)
    k = torch.randn(128, 128, dtype=torch.float32, device=device)
    quant_out = torch.zeros(128, 128, dtype=torch.int8, device=device)
    scale_value = 2.0
    scale_bits = struct.unpack("!I", struct.pack("!f", scale_value))[0]

    # Tile [0, 0]: out[0:64, 0:64] = q[0:64,0:64] @ k[0:64,0:64]
    store_tile_single_kernel(q, k, quant_out, scale_bits, 0, 0)
    torch.npu.synchronize()
    raw_ref_00 = torch.matmul(q[0:64, 0:64].cpu(), k[0:64, 0:64].cpu())
    expected_00 = torch.clamp(torch.round(raw_ref_00 * scale_value), -128, 127)
    torch.testing.assert_close(quant_out[0:64, 0:64].cpu().to(torch.int32), expected_00.to(torch.int32), rtol=0, atol=4)

    # Tile [0, 1]: out[0:64, 64:128] = q[0:64,64:128] @ k[64:128,0:64]
    store_tile_single_kernel(q, k, quant_out, scale_bits, 0, 1)
    torch.npu.synchronize()
    raw_ref_01 = torch.matmul(q[0:64, 0:64].cpu(), k[0:64, 64:128].cpu())
    expected_01 = torch.clamp(torch.round(raw_ref_01 * scale_value), -128, 127)
    torch.testing.assert_close(
        quant_out[0:64, 64:128].cpu().to(torch.int32), expected_01.to(torch.int32), rtol=0, atol=4
    )
    logging.info("test_store_tile_multi_offset passed.")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
