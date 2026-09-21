# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""init_output op: initialize a GM tensor region with a scalar value.

Doc: docs/zh/pypto_pro/api/SIMD-API/计算API/Memory数据搬运/init_output.md

Verifies scalar-fill of GM memory across all 11 doc-supported dtypes
(FP32/FP16/BF16/INT8/INT16/INT32/INT64/UINT8/UINT16/UINT32/UINT64),
pl.const() scalar values, non-16-aligned sizes, non-zero fill values,
offset-based partial init, multi-chunk large size, and dynamic
multi-dimensional multicore scenarios with tail-core data divergence.

Constraint: init_output emits no cross-core barrier; when the initialized
data is consumed by other cores / the cube sub-core, the caller must insert
pl.system.sync_all(core_type=MIX) on both sides (documented in the md).
"""

import ctypes
import logging
import os

import pypto_pro.language as pl
import pytest
import torch
import torch_npu

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


# All independent scalar-fill cases share a single vector kernel. Each output
# retains its original dtype, shape, initial sentinel and expected fill regions.
@pl.jit(auto_mutex=True)
def init_output_scalar_cases(
    fp32_zero: pl.Tensor[[64, 64], pl.DT_FP32],
    fp16_unaligned: pl.Tensor[[100], pl.DT_FP16],
    int32_unaligned_nonzero: pl.Tensor[[50], pl.DT_INT32],
    int16_nonzero: pl.Tensor[[128], pl.DT_INT16],
    bf16_large: pl.Tensor[[4096], pl.DT_BF16],
    uint8_unaligned: pl.Tensor[[100], pl.DT_UINT8],
    int8_negative: pl.Tensor[[128], pl.DT_INT8],
    uint16_unaligned: pl.Tensor[[100], pl.DT_UINT16],
    uint32_unaligned: pl.Tensor[[50], pl.DT_UINT32],
    uint64_large_value: pl.Tensor[[128], pl.DT_UINT64],
    int64_unaligned: pl.Tensor[[96], pl.DT_INT64],
    fp32_const: pl.Tensor[[200], pl.DT_FP32],
    fp32_multiple_calls: pl.Tensor[[256], pl.DT_FP32],
    fp32_offset: pl.Tensor[[128], pl.DT_FP32],
    fp32_size_17: pl.Tensor[[17], pl.DT_FP32],
    fp32_multi_chunk: pl.Tensor[[65536], pl.DT_FP32],
    fp32_float_value: pl.Tensor[[200], pl.DT_FP32],
):
    with pl.section_vector():
        pl.init_output(fp32_zero, offset=0, size=64 * 64, value=0.0)
        pl.init_output(fp16_unaligned, offset=0, size=100, value=0.0)
        pl.init_output(int32_unaligned_nonzero, offset=0, size=50, value=7)
        pl.init_output(int16_nonzero, offset=0, size=128, value=-1)
        pl.init_output(bf16_large, offset=0, size=4096, value=0.0)
        pl.init_output(uint8_unaligned, offset=0, size=100, value=200)
        pl.init_output(int8_negative, offset=0, size=128, value=-3)
        pl.init_output(uint16_unaligned, offset=0, size=100, value=60000)
        pl.init_output(uint32_unaligned, offset=0, size=50, value=4000000000)
        pl.init_output(uint64_large_value, offset=0, size=128, value=2**40)
        pl.init_output(int64_unaligned, offset=0, size=96, value=-1234567890123)
        pl.init_output(fp32_const, offset=0, size=200, value=pl.const(2.5, pl.DT_FP32))
        pl.init_output(fp32_multiple_calls, offset=0, size=128, value=1.5)
        pl.init_output(fp32_multiple_calls, offset=128, size=128, value=-2.0)
        pl.init_output(fp32_multiple_calls, offset=96, size=64, value=7.0)
        pl.init_output(fp32_offset, offset=16, size=96, value=0.0)
        pl.init_output(fp32_size_17, offset=0, size=17, value=0.0)
        pl.init_output(fp32_multi_chunk, offset=0, size=65536, value=0.0)
        pl.init_output(fp32_float_value, offset=0, size=200, value=3.14)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_init_output_scalar_cases():
    _require_a5(ST_DEVICE)
    # name, shape, dtype, initial value, (offset, size, fill value) regions.
    # The order matches the kernel signature. Overlapping regions are applied in order.
    cases = [
        ("fp32_zero", (64, 64), torch.float32, 1, [(0, 4096, 0)]),
        ("fp16_unaligned", (100,), torch.float16, 1, [(0, 100, 0)]),
        ("int32_unaligned_nonzero", (50,), torch.int32, 99, [(0, 50, 7)]),
        ("int16_nonzero", (128,), torch.int16, 99, [(0, 128, -1)]),
        ("bf16_large", (4096,), torch.bfloat16, 1, [(0, 4096, 0)]),
        ("uint8_unaligned", (100,), torch.uint8, 99, [(0, 100, 200)]),
        ("int8_negative", (128,), torch.int8, 99, [(0, 128, -3)]),
        ("uint16_unaligned", (100,), torch.uint16, 99, [(0, 100, 60000)]),
        ("uint32_unaligned", (50,), torch.uint32, 99, [(0, 50, 4000000000)]),
        ("uint64_large_value", (128,), torch.uint64, 99, [(0, 128, 2**40)]),
        ("int64_unaligned", (96,), torch.int64, 99, [(0, 96, -1234567890123)]),
        ("fp32_const", (200,), torch.float32, 99, [(0, 200, 2.5)]),
        ("fp32_multiple_calls", (256,), torch.float32, 99, [(0, 128, 1.5), (128, 128, -2), (96, 64, 7)]),
        ("fp32_offset", (128,), torch.float32, 1, [(16, 96, 0)]),
        ("fp32_size_17", (17,), torch.float32, 1, [(0, 17, 0)]),
        ("fp32_multi_chunk", (65536,), torch.float32, 1, [(0, 65536, 0)]),
        ("fp32_float_value", (200,), torch.float32, 99, [(0, 200, 3.14)]),
    ]
    outputs = [torch.full(shape, initial, dtype=dtype).to(ST_DEVICE) for _, shape, dtype, initial, _ in cases]
    init_output_scalar_cases(*outputs)
    torch.npu.synchronize()
    for output, (name, shape, dtype, initial, regions) in zip(outputs, cases):
        expected = torch.full(shape, initial, dtype=dtype)
        for offset, size, value in regions:
            expected.view(-1)[offset:offset + size] = value
        tolerance = 1e-5 if name == "fp32_float_value" else 0
        torch.testing.assert_close(
            output.cpu(), expected, rtol=tolerance, atol=tolerance, msg=lambda message: f"{name}: {message}"
        )


# ===========================================================================
# Dynamic 3D tensors, with a different value per core. Partitioning follows
# the launched core count; the last core handles the remaining elements.
# ===========================================================================
NUM_CORES = 28


@pl.jit(auto_mutex=True)
def init_output_dynamic_3d_multicore(
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    with pl.section_vector():
        total = out.shape[0] * out.shape[1] * out.shape[2]
        core_id = pl.get_block_idx()
        num_cores = pl.get_block_num()
        per_core = (total + num_cores - 1) // num_cores
        offset = core_id * per_core
        # clamp size for the tail core: min(per_core, total - offset)
        size = pl.min(per_core, total - offset)
        # each core writes a different value: core 0 → 0.0, core 1 → 1.0, ...
        pl.init_output(out, offset=offset, size=size, value=core_id)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_init_output_dynamic_3d_multicore():
    """Dynamic 3D [4, 32, 100] = 12800 elems, per-core value = block_idx."""
    device = ST_DEVICE
    _require_a5(device)
    dims = (4, 32, 100)
    out = torch.full(dims, 99.0, device=device, dtype=torch.float32)
    init_output_dynamic_3d_multicore[None, NUM_CORES](out)
    torch.npu.synchronize()

    total = 1
    for d in dims:
        total *= d
    per_core = (total + NUM_CORES - 1) // NUM_CORES
    expected = torch.zeros(dims, device=device, dtype=torch.float32)
    flat = expected.view(-1)
    for core in range(NUM_CORES):
        offset = core * per_core
        size = min(per_core, total - offset)
        flat[offset:offset + size] = float(core)
    torch.testing.assert_close(out, expected)
    logging.info("init_output dynamic 3D [4,32,100] %d-core per-core-value passed!", NUM_CORES)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_init_output_dynamic_3d_tail_core():
    """Dynamic 3D [4, 33, 100] = 13200 elems, with less work on the last core."""
    device = ST_DEVICE
    _require_a5(device)
    dims = (4, 33, 100)
    out = torch.full(dims, 99.0, device=device, dtype=torch.float32)
    init_output_dynamic_3d_multicore[None, NUM_CORES](out)
    torch.npu.synchronize()

    total = 1
    for d in dims:
        total *= d
    per_core = (total + NUM_CORES - 1) // NUM_CORES
    expected = torch.zeros(dims, device=device, dtype=torch.float32)
    flat = expected.view(-1)
    for core in range(NUM_CORES):
        offset = core * per_core
        size = min(per_core, total - offset)
        flat[offset:offset + size] = float(core)
    torch.testing.assert_close(out, expected)

    # Verify tail core indeed got a different (smaller) size
    tail_offset = (NUM_CORES - 1) * per_core
    tail_size = min(per_core, total - tail_offset)
    assert 0 < tail_size < per_core, f"Tail core size {tail_size} should be between 0 and {per_core}"
    logging.info(
        "init_output dynamic 3D [4,33,100]=13200 %d-core tail-core (per_core=%d, tail_size=%d) passed!",
        NUM_CORES,
        per_core,
        tail_size,
    )


# ===========================================================================
# Dynamic 4D tensor [2, 3, 17, 19] = 1938 elems, FP16 dtype, with a partial last core.
# ===========================================================================
@pl.jit(auto_mutex=True)
def init_output_dynamic_4d_fp16_multicore(
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    with pl.section_vector():
        total = out.shape[0] * out.shape[1] * out.shape[2] * out.shape[3]
        core_id = pl.get_block_idx()
        num_cores = pl.get_block_num()
        per_core = (total + num_cores - 1) // num_cores
        offset = core_id * per_core
        size = pl.min(per_core, total - offset)
        pl.init_output(out, offset=offset, size=size, value=core_id)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_init_output_dynamic_4d_fp16_tail_core():
    """Dynamic 4D [2,3,17,19]=1938 elems FP16, with less work on the last core."""
    device = ST_DEVICE
    _require_a5(device)
    shape = [2, 3, 17, 19]
    total = 1
    for d in shape:
        total *= d
    out = torch.full(shape, 99.0, device=device, dtype=torch.float16)
    init_output_dynamic_4d_fp16_multicore[None, NUM_CORES](out)
    torch.npu.synchronize()

    per_core = (total + NUM_CORES - 1) // NUM_CORES
    expected = torch.zeros(shape, device=device, dtype=torch.float16)
    flat = expected.view(-1)
    for core in range(NUM_CORES):
        offset = core * per_core
        size = min(per_core, total - offset)
        flat[offset:offset + size] = float(core)
    torch.testing.assert_close(out, expected)

    tail_offset = (NUM_CORES - 1) * per_core
    tail_size = min(per_core, total - tail_offset)
    assert 0 < tail_size < per_core, f"Tail core size {tail_size} should be between 0 and {per_core}"
    logging.info(
        "init_output dynamic 4D %s=%d FP16 %d-core tail-core (per_core=%d, tail_size=%d) passed!",
        shape,
        total,
        NUM_CORES,
        per_core,
        tail_size,
    )


# ===========================================================================
# Dynamic values: the fill value may be any runtime scalar expression, not
# just a literal / pl.const(). Covered here:
#   1. pl.range loop induction variable (runtime scalar, INDEX -> FP32 cast)
#   2. dynamic tensor shape dims (out.shape[k] of a DYNAMIC tensor)
#   3. dynamic int value on an INT64 tensor (no float cast path)
# ===========================================================================
@pl.jit(auto_mutex=True)
def init_output_loop_var_value(out: pl.Tensor[[pl.DYNAMIC], pl.DT_FP32]):
    with pl.section_vector():
        n = out.shape[0]
        for i in pl.range(0, n, 128):
            sz = pl.min(128, n - i)
            pl.init_output(out, offset=i, size=sz, value=i)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_init_output_loop_var_value():
    """Fill each 128-elem chunk with its own loop-var base offset (dynamic value)."""
    device = ST_DEVICE
    _require_a5(device)
    n = 500
    out = torch.full([n], -1.0, device=device, dtype=torch.float32)
    init_output_loop_var_value(out)
    torch.npu.synchronize()
    expected = torch.empty(n, device=device, dtype=torch.float32)
    for j in range(n):
        expected[j] = float((j // 128) * 128)
    torch.testing.assert_close(out, expected)
    logging.info("init_output pl.range loop-var value passed!")


@pl.jit(auto_mutex=True)
def init_output_shape_value(out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32]):
    with pl.section_vector():
        m = out.shape[0]
        n = out.shape[1]
        total = m * n
        pl.init_output(out, offset=0, size=total, value=m + n)
        # second call with a different dynamic shape-derived expression
        pl.init_output(out, offset=total // 2, size=total // 2, value=n * 2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_init_output_shape_value():
    """Fill with dynamic shape dims: value = m + n, then n * 2 on the back half."""
    device = ST_DEVICE
    _require_a5(device)
    dims = (3, 100)  # 300 elems; second call covers [150, 300)
    out = torch.full(dims, -1.0, device=device, dtype=torch.float32)
    init_output_shape_value(out)
    torch.npu.synchronize()
    m, n = dims
    expected = torch.full(dims, float(m + n), device=device, dtype=torch.float32)
    expected.view(-1)[m * n // 2:] = float(n * 2)
    torch.testing.assert_close(out, expected)
    logging.info("init_output dynamic-shape value passed!")


@pl.jit(auto_mutex=True)
def init_output_loop_var_int64(out: pl.Tensor[[pl.DYNAMIC], pl.DT_INT64]):
    with pl.section_vector():
        n = out.shape[0]
        for i in pl.range(0, n, 64):
            sz = pl.min(64, n - i)
            pl.init_output(out, offset=i, size=sz, value=i * 1000)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_init_output_loop_var_int64():
    """Dynamic int value on an INT64 tensor (no int->float cast path)."""
    device = ST_DEVICE
    _require_a5(device)
    n = 130
    out = torch.full([n], -1, device=device, dtype=torch.int64)
    init_output_loop_var_int64(out)
    torch.npu.synchronize()
    expected = torch.empty(n, device=device, dtype=torch.int64)
    for j in range(n):
        expected[j] = (j // 64) * 64 * 1000
    torch.testing.assert_close(out, expected)
    logging.info("init_output loop-var INT64 value passed!")


# ===========================================================================
# Integration: init_output + matmul + compute + atomic store (32-core, K-split)
#
# 1. init_output: each core clears its own 16-row slice of workspace to 0
#    (offset = core_id * M_IO * N_IO). workspace is pre-filled with 99.0
#    on the host so a failed init_output would leave 99.0 and be detected.
# 2. sync_all(MIX) on both sides — init_output emits no cross-core barrier,
#    so AIV and AIC must reach a full barrier before the cube section starts.
# 3. Cube: each core loads its K-slice of A and B, matmul -> Acc, store to buf.
# 4. set_cross_core(FIX) -> wait_cross_core(MTE2) — AIC's fixpipe store to buf
#    must complete before AIV's load from buf.
# 5. Vector: each core loads its matmul result from buf, scales by 2 (compute),
#    atomic-adds to workspace[0:M_IO, 0:N_IO] (the shared region cleared by
#    core 0).
#
# Final: workspace[0:M_IO] = 2 * full_matmul(A, B); workspace[M_IO:] = 0
# (cleared by cores 1..31, verifying init_output worked per-core).
# ===========================================================================
M_IO = 448
K_IO = 16
N_IO = 16
TILE_M = 16
NUM_CORES_IO = 28


@pl.jit(auto_mutex=True)
def init_output_matmul_add_atomic(
    a: pl.Tensor[[M_IO, K_IO], pl.DT_FP16],
    b: pl.Tensor[[K_IO, N_IO], pl.DT_FP16],
    workspace: pl.Tensor[[M_IO, N_IO], pl.DT_FP32],
):
    core_id = pl.get_block_idx()
    row0 = core_id * TILE_M

    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M, K_IO], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[K_IO, N_IO], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000,
        mutex_ids=[1],
    )
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M, K_IO], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[K_IO, N_IO], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    tile_acc = pl.make_tile(
        pl.TileType(shape=[TILE_M, N_IO], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addr=0x0000,
    )
    vec_group = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M // 2, N_IO], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x2000,
        mutex_ids=[[6]],
    )

    with pl.section_vector():
        # Each AIV initializes half of the rows produced by its AIC.
        pl.init_output(workspace, offset=core_id * (TILE_M // 2) * N_IO, size=(TILE_M // 2) * N_IO, value=0.0)
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)

    with pl.section_cube():
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        pl.load(a_l1[0], a, [row0, 0])
        pl.load(b_l1[0], b, [0, 0])
        pl.move(a_left[0], a_l1[0])
        pl.move(b_right[0], b_l1[0])
        pl.matmul(tile_acc, a_left[0], b_right[0])
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.move(vec_group[0], tile_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    with pl.section_vector():
        # sub_index = pl.get_subblock_idx()
        row_off = core_id * (TILE_M // 2)
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=0)
        pl.add(vec_group[0], vec_group[0], 0.9)
        pl.store(workspace, vec_group[0], [row_off, 0], atomic=pl.AtomicType.AtomicAdd)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_init_output_matmul_add_atomic():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(42)
    a = torch.randn([M_IO, K_IO], device=device, dtype=torch.float16)
    b = torch.randn([K_IO, N_IO], device=device, dtype=torch.float16)
    workspace = torch.full([M_IO, N_IO], 99.0, device=device, dtype=torch.float32)

    init_output_matmul_add_atomic[None, NUM_CORES_IO](a, b, workspace)
    torch.npu.synchronize()
    expected = torch.matmul(a.float(), b.float()) + 0.9
    torch.testing.assert_close(workspace, expected, rtol=1e-3, atol=1e-3)
    logging.info("init_output + matmul + vec compute (M-split, %d-core) passed!", NUM_CORES_IO)


# ===========================================================================
# Integration: init_output + matmul (workspace as matmul input, 32-core)
#
# 1. init_output fills workspace with 1.0 (each core inits its own M-slice)
# 2. sync_all(MIX) both sides — AIV's MTE3 writes must be visible to the
#    AIC's load before matmul reads it (INTRA_BLOCK cross_core is unreliable
#    at 32 cores; a full barrier between the sections is required)
# 3. Cube: each core loads its workspace slice and shared B, matmul -> Acc,
#    store result to C
#
# With workspace = 1.0, C = ones(M,K) @ B, i.e. every row of C equals
# sum_k B[k, :]. Verifies init_output-written GM data is readable as matmul
# input across the AIV->AIC sub-core boundary.
# ===========================================================================
M_IN = 448
K_IN = 16
N_IN = 16
TILE_M = 16


@pl.jit(auto_mutex=True)
def init_output_matmul_input(
    workspace: pl.Tensor[[M_IN, K_IN], pl.DT_FP16],
    b: pl.Tensor[[K_IN, N_IN], pl.DT_FP16],
    c: pl.Tensor[[M_IN, N_IN], pl.DT_FP32],
):
    core_id = pl.get_block_idx()
    row0 = core_id * TILE_M

    tt_ws = pl.TileType(shape=[TILE_M, K_IN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
    tt_b = pl.TileType(shape=[K_IN, N_IN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
    tt_left = pl.TileType(shape=[TILE_M, K_IN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, compact=1)
    tt_right = pl.TileType(shape=[K_IN, N_IN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, compact=1)
    tt_acc = pl.TileType(shape=[TILE_M, N_IN], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, compact=1)

    ws_l1 = pl.make_tile_group(type=tt_ws, addrs=0x00000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(type=tt_b, addrs=0x10000, mutex_ids=[1])
    ws_l0a = pl.make_tile_group(type=tt_left, addrs=0x0000, mutex_ids=[2])
    b_l0b = pl.make_tile_group(type=tt_right, addrs=0x0000, mutex_ids=[3])
    acc = pl.make_tile_group(type=tt_acc, addrs=0x0000, mutex_ids=[4])

    with pl.section_vector():
        # Each AIV initializes half of the rows consumed by its AIC.
        pl.init_output(workspace, offset=core_id * (TILE_M // 2) * K_IN, size=(TILE_M // 2) * K_IN, value=1.0)
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)

    with pl.section_cube():
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        cur_ws = ws_l1.current()
        cur_b = b_l1.current()
        al = ws_l0a.current()
        br = b_l0b.current()
        ac = acc.current()

        pl.load(cur_ws, workspace, [row0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.move(al, cur_ws)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.store(c, ac, [row0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_init_output_matmul_input():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(7)
    b = torch.randn([K_IN, N_IN], device=device, dtype=torch.float16)
    workspace = torch.full([M_IN, K_IN], 99.0, device=device, dtype=torch.float16)
    c = torch.full([M_IN, N_IN], 99.0, device=device, dtype=torch.float32)
    init_output_matmul_input[None, NUM_CORES](workspace, b, c)
    torch.npu.synchronize()
    expected = torch.ones([M_IN, K_IN], device=device, dtype=torch.float32) @ b.float()
    torch.testing.assert_close(c, expected, rtol=1e-3, atol=1e-3)
    logging.info("init_output + matmul (workspace as input) passed!")


# ===========================================================================
# NZ-layout tensor: init_output addresses GM through a flat [1, numel] ND
# view, so the layout is irrelevant — offset/size count PHYSICAL elements.
#
# The tensor is [128, 128] FP16 NZ (rows and cols both 16-aligned), so its
# physical NZ storage holds exactly 128*128 elements with no padding. The
# raw GM bytes are pre-filled with a sentinel; a partial fill with
# offset/size must land on the flat physical element sequence, exactly as
# it would for an ND tensor.
# ===========================================================================
NZ_M = 128
NZ_N = 128
NZ_NUMEL = NZ_M * NZ_N


@pl.jit(auto_mutex=True)
def init_output_fp16_nz(out: pl.Tensor[[NZ_M, NZ_N], pl.DT_FP16, pl.NZ]):
    with pl.section_vector():
        pl.init_output(out, offset=256, size=512, value=-3.5)


def _nz_raw_copy(dst_ptr: int, src: torch.Tensor, kind: int, nbytes: int) -> None:
    acl = ctypes.CDLL("libascendcl.so")
    acl.aclrtMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    acl.aclrtMemcpy.restype = ctypes.c_int
    assert acl.aclrtMemcpy(dst_ptr, nbytes, src.data_ptr(), nbytes, kind) == 0


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_init_output_fp16_nz():
    device = ST_DEVICE
    _require_a5(device)
    out = torch_npu.empty_with_format([NZ_M, NZ_N], dtype=torch.float16, device=device, acl_format=29)
    assert torch_npu.get_npu_format(out) == 29
    storage = out.untyped_storage()
    numel_phys = storage.nbytes() // 2
    assert numel_phys == NZ_NUMEL, f"aligned NZ storage expected {NZ_NUMEL} elems, got {numel_phys}"

    torch.npu.synchronize()
    sentinel = torch.full([numel_phys], 66.0, dtype=torch.float16)  # H2D: ACL_MEMCPY_HOST_TO_DEVICE
    _nz_raw_copy(storage.data_ptr(), sentinel, 1, storage.nbytes())

    init_output_fp16_nz(out)
    torch.npu.synchronize()

    host = torch.empty(numel_phys, dtype=torch.float16)  # D2H: ACL_MEMCPY_DEVICE_TO_HOST
    _nz_raw_copy(host.data_ptr(), storage, 2, storage.nbytes())

    expected = torch.full([numel_phys], 66.0, dtype=torch.float16)
    expected[256:768] = -3.5
    torch.testing.assert_close(host, expected)
    logging.info("init_output FP16 NZ layout (flat physical offset/size) passed!")
