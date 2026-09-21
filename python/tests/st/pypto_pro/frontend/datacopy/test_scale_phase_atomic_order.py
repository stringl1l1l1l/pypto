# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""P2: phase/atomic/order fusion tests.

This test file covers:
1. Phase fusion: Partial, Final — the FixPipe unit-flag (UF) chain must be
   balanced (Partial opens, Final closes); the kernels build a valid chain via
   K-split accumulation (matmul Partial + matmul_acc Final) before the store.
2. Atomic fusion: AtomicAdd — scale + atomic on an INT8 GM output.
3. Order parameter: transpose (descending rejected), 4D tensor
"""

import logging
import os
import struct

from pypto_pro._errors import InvalidVal
import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


def _make_q(device: str) -> torch.Tensor:
    row = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], device=device, dtype=torch.float32).repeat(8)
    return row.unsqueeze(0).repeat(64, 1)


def _make_k(device: str) -> torch.Tensor:
    return torch.eye(64, device=device, dtype=torch.float32)


# ============================================================================
# Test 1: Phase fusion tests
# ============================================================================


@pl.jit()
def scale_phase_partial_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    quant_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    scale_value: pl.DT_INT32,
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

        # The FixPipe unit-flag (UF) chain must be balanced: a Partial opens it and a
        # Final closes it. A lone Partial store (or a lone Final store) leaves the UF
        # state machine unbalanced and hangs the AI Core. Use a K-split accumulation
        # (matmul Partial -> matmul_acc Final) to build a valid chain, then exercise
        # the store Partial -> store Final pair (scale + phase).
        pl.matmul(acc, q_left, k_right, phase=pl.AccPhase.Partial)
        pl.matmul_acc(acc, acc, q_left, k_right, phase=pl.AccPhase.Final)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)

        # Store with Partial phase (open) then Final phase (close)
        pl.store(quant_out, acc, [0, 0], scale=scale_value, phase=pl.STPhase.Partial)
        pl.store(quant_out, acc, [0, 0], scale=scale_value, phase=pl.STPhase.Final)
        pl.system.bar_all()


@pl.jit()
def scale_phase_final_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    quant_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    scale_value: pl.DT_INT32,
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

        # Balanced UF chain: matmul Partial + matmul_acc Final, then a store Final
        # (scale + Final phase) closes the chain.
        pl.matmul(acc, q_left, k_right, phase=pl.AccPhase.Partial)
        pl.matmul_acc(acc, acc, q_left, k_right, phase=pl.AccPhase.Final)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)

        # Store with Final phase
        pl.store(quant_out, acc, [0, 0], scale=scale_value, phase=pl.STPhase.Final)
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_phase_partial():
    """Test scale + Partial phase fusion."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device)
    k = _make_k(device)
    quant_out = torch.zeros((64, 64), device=device, dtype=torch.int8)
    scale_value = 2.0
    scale_bits = struct.unpack("!I", struct.pack("!f", scale_value))[0]

    scale_phase_partial_kernel(q, k, quant_out, scale_bits)
    torch.npu.synchronize()

    # The kernel accumulates q@k twice (matmul Partial + matmul_acc Final) to keep
    # the FixPipe UF chain balanced; the store Partial + Final pair quantizes that.
    raw_ref = 2 * torch.matmul(q, k)
    expected = torch.clamp(torch.round(raw_ref * scale_value), -128, 127).to(torch.int8)

    torch.testing.assert_close(quant_out.to(torch.int32), expected.to(torch.int32), rtol=0, atol=1)
    logging.info("test_phase_partial passed.")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_phase_final():
    """Test scale + Final phase fusion."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device)
    k = _make_k(device)
    quant_out = torch.zeros((64, 64), device=device, dtype=torch.int8)
    scale_value = 2.0
    scale_bits = struct.unpack("!I", struct.pack("!f", scale_value))[0]

    scale_phase_final_kernel(q, k, quant_out, scale_bits)
    torch.npu.synchronize()

    # The kernel accumulates q@k twice (matmul Partial + matmul_acc Final) to keep
    # the FixPipe UF chain balanced; the store Final quantizes that.
    raw_ref = 2 * torch.matmul(q, k)
    expected = torch.clamp(torch.round(raw_ref * scale_value), -128, 127).to(torch.int8)

    torch.testing.assert_close(quant_out.to(torch.int32), expected.to(torch.int32), rtol=0, atol=1)
    logging.info("test_phase_final passed.")


# ============================================================================
# Test 2: Atomic fusion tests
# ============================================================================


@pl.jit()
def scale_atomic_add_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    quant_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    scale_value: pl.DT_INT32,
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

        # Store with AtomicAdd
        pl.store(quant_out, acc, [0, 0], scale=scale_value, atomic=pl.AtomicType.AtomicAdd)
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_atomic_add():
    """Test scale + AtomicAdd fusion."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device)
    k = _make_k(device)
    quant_out = torch.zeros((64, 64), device=device, dtype=torch.int8)
    scale_value = 2.0
    scale_bits = struct.unpack("!I", struct.pack("!f", scale_value))[0]

    # First store (initial values)
    scale_atomic_add_kernel[None, 1](q, k, quant_out, scale_bits)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    expected_first = torch.clamp(torch.round(raw_ref * scale_value), -128, 127).to(torch.int8)

    torch.testing.assert_close(quant_out.to(torch.int32), expected_first.to(torch.int32), rtol=0, atol=1)

    # Second store (atomic add)
    scale_atomic_add_kernel[None, 1](q, k, quant_out, scale_bits)
    torch.npu.synchronize()

    # Expected: first + second (with saturation)
    expected_second = torch.clamp(expected_first.to(torch.int32) + expected_first.to(torch.int32), -128, 127).to(
        torch.int8
    )

    torch.testing.assert_close(quant_out.to(torch.int32), expected_second.to(torch.int32), rtol=0, atol=1)
    logging.info("test_atomic_add passed.")


# ============================================================================
# Test 3: Order parameter tests
# ============================================================================


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_order_descending_rejected():
    """store 的 order 仅支持升序；降序 order=[1, 0] 应在解析期被拒绝。

    原 review 版本假设 order=[1,0] 支持转置写出，但前端 _ir_store 明确校验
    ``order must be ascending``，故改为验证该拒绝行为。
    """

    @pl.jit()
    def kernel(
        q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        quant_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
        scale_value: pl.DT_INT32,
    ):
        with pl.section_cube():
            acc_type = pl.TileType(
                shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
            )
            acc = pl.make_tile(acc_type, addr=0x0000)
            pl.store(quant_out, acc, [0, 0], scale=scale_value, order=[1, 0])

    q = _make_q("cpu")
    k = _make_k("cpu")
    quant_out = torch.zeros(64, 64, dtype=torch.int8)
    with pytest.raises(InvalidVal, match="order must be ascending"):
        kernel(q, k, quant_out, 0)
    logging.info("test_order_descending_rejected passed.")


if __name__ == "__main__":
    test_phase_partial()
    test_phase_final()
    test_atomic_add()
    test_order_descending_rejected()
    logging.info("\nP2 phase/atomic/order tests passed.")
