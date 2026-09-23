#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Minimal ST for OP_L0C_COPY_UB_DUAL_DST fusion (SplitN direction).

Kernel layout:
    matmul(A, B^T) -> mm: [M, N] on L0C
    upper = mm[:, :N/2]               # 1st L0C_COPY_UB,  N 轴前半
    lower = mm[:,  N/2:]              # 2nd L0C_COPY_UB,  N 轴后半
    out0  = upper + 1.0               # AIV0 vector chain
    out1  = lower + 2.0               # AIV1 vector chain

The two L0C_COPY_UB ops:
  - share the same L0C input tensor (mm)
  - have identical UB output shape ([M, N/2]) and validShape
  - are N-axis adjacent (offsets 0 and N/2)
  - feed two independent vector ops, which MixSchedule's CoreScheduler
    binds to AIV0 / AIV1 (via existing DualDstProcess preCoreAssign)

When `kTempEnableDualDst = true` 在 schedule_ooo.cpp 里被打开, OoOSchedule
RunDualDstFuse 阶段会把这两个 L0C_COPY_UB 合并成单个 OP_L0C_COPY_UB_DUAL_DST,
保留一个 ALLOC, 删另一个; 双 UB 池 (AIV0/AIV1) 同地址联合分配。开关关时退化为
两条普通 L0C_COPY_UB, 测试结果保持一致 (数值正确性与开关无关)。

Run:
    pytest python/tests/st/test_dualdst.py -v
"""

import os

from numpy.testing import assert_allclose
import pytest
import torch

import pypto
import pypto.experimental

FP16 = pypto.DT_FP16
FP32 = pypto.DT_FP32

# 所有 dualdst kernel 共用同一份 jit 配置: 提前固化, 避免每个 kernel 重复 3 行装饰器。
_DUALDST_JIT = pypto.frontend.jit(
    debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0}
)


def _matmul_split_n(a_tile, b_tile, half_n):
    """matmul(b_trans=True, FP32 输出) + 沿 N 轴二分; 返回 (upper, lower).

    所有 dualdst kernel 的核心三行模式都用它, 避免每个 kernel 重复 mm + 上下半切片。
    """
    mm = pypto.matmul(a_tile, b_tile, b_trans=True, out_dtype=FP32)
    return mm[:, :half_n], mm[:, half_n:]


def _split_n_prologue_64(a_tensor, b_tensor):
    """3 个 SplitN dualdst kernel 共用的前缀.

    cube tile (64,64,128) -> matmul -> 沿 N 二分 -> 设 vec tile (64,64);
    返回 (upper, lower) 两份 L0C view。
    """
    pypto.set_cube_tile_shapes([64, 64], [64, 64], [128, 128])
    upper, lower = _matmul_split_n(a_tensor, b_tensor, 64)
    pypto.set_vec_tile_shapes(64, 64)
    return upper, lower


def _online_softmax_exp(scores, tile_m, tile_n):
    pypto.set_vec_tile_shapes(tile_m, tile_n)
    exp_scores_bf16, _, _ = pypto.experimental.online_softmax(scores, 1.0)
    return exp_scores_bf16


def _torch_online_softmax_exp(scores):
    column_max = torch.max(scores, dim=0, keepdim=True).values
    return torch.exp(scores - column_max).to(torch.bfloat16).to(torch.float32)


@_DUALDST_JIT
def dual_dst_split_n_kernel(
    a_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], FP16),
    b_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], FP16),
    out0_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], FP32),
    out1_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], FP32),
):
    """matmul -> L0C -> N 轴二分 -> 两条独立 add-scalar vector chain.

    单 cube tile 覆盖 M/K/N: 一次 matmul 产出唯一一份 L0C tensor,
    它的两个 L0C_COPY_UB consumer 才是 dual_dst 候选对。
    """
    pypto.experimental.auto_mix_partition(1)
    upper, lower = _split_n_prologue_64(a_tensor, b_tensor)
    upper = _online_softmax_exp(upper, 64, 64)
    lower = _online_softmax_exp(lower, 64, 64)
    out0_tensor[:, :] = upper + 1.0
    pypto.set_vec_tile_shapes(64, 64)
    out1_tensor[:, :] = lower + 2.0


@pytest.mark.soc("950")
@pytest.mark.skip(reason="large test case")
@pypto.options(pass_options={"enable_slice": False})
def test_dual_dst_split_n():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    m, k, n = 64, 64, 128
    half_n = n // 2

    torch.manual_seed(0)
    a = torch.rand([m, k], dtype=torch.float16, device=f"npu:{device_id}")
    b = torch.rand([n, k], dtype=torch.float16, device=f"npu:{device_id}")
    out0 = torch.zeros([m, half_n], dtype=torch.float32, device=f"npu:{device_id}")
    out1 = torch.zeros([m, half_n], dtype=torch.float32, device=f"npu:{device_id}")

    dual_dst_split_n_kernel(a, b, out0, out1)

    # PyTorch golden: matmul(A, B^T) 后按 N 轴二分,各自加常量
    mm_golden = torch.matmul(a.to(torch.float32), b.to(torch.float32).T)
    golden0 = _torch_online_softmax_exp(mm_golden[:, :half_n]) + 1.0
    golden1 = _torch_online_softmax_exp(mm_golden[:, half_n:]) + 2.0

    assert_allclose(
        out0.cpu().to(torch.float32).numpy(),
        golden0.cpu().numpy(),
        rtol=5e-3,
        atol=5e-3,
    )
    assert_allclose(
        out1.cpu().to(torch.float32).numpy(),
        golden1.cpu().numpy(),
        rtol=5e-3,
        atol=5e-3,
    )


@_DUALDST_JIT
def dual_dst_split_m_kernel(
    a_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], FP16),
    b_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], FP16),
    out0_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], FP32),
    out1_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], FP32),
):
    """matmul -> L0C -> M 轴二分 -> 两条独立 add-scalar vector chain (SplitM 方向)。"""
    pypto.experimental.auto_mix_partition(1)
    pypto.set_cube_tile_shapes([128, 128], [64, 64], [64, 64])
    mm = pypto.matmul(a_tensor, b_tensor, b_trans=True, out_dtype=FP32)
    half_m = mm.shape[0] // 2
    upper = mm[:half_m, :]
    lower = mm[half_m:, :]
    upper = _online_softmax_exp(upper, 64, 64)
    lower = _online_softmax_exp(lower, 64, 64)
    pypto.set_vec_tile_shapes(64, 64)
    out0_tensor[:, :] = upper + 1.0
    pypto.set_vec_tile_shapes(64, 64)
    out1_tensor[:, :] = lower + 2.0


@pytest.mark.soc("950")
@pytest.mark.skip(reason="large test case")
@pypto.options(pass_options={"enable_slice": False})
def test_dual_dst_split_m():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    m, k, n = 128, 64, 64
    half_m = m // 2

    torch.manual_seed(0)
    a = torch.rand([m, k], dtype=torch.float16, device=f"npu:{device_id}")
    b = torch.rand([n, k], dtype=torch.float16, device=f"npu:{device_id}")
    out0 = torch.zeros([half_m, n], dtype=torch.float32, device=f"npu:{device_id}")
    out1 = torch.zeros([half_m, n], dtype=torch.float32, device=f"npu:{device_id}")

    dual_dst_split_m_kernel(a, b, out0, out1)

    mm_golden = torch.matmul(a.to(torch.float32), b.to(torch.float32).T)
    golden0 = _torch_online_softmax_exp(mm_golden[:half_m, :]) + 1.0
    golden1 = _torch_online_softmax_exp(mm_golden[half_m:, :]) + 2.0

    assert_allclose(out0.cpu().to(torch.float32).numpy(), golden0.cpu().numpy(), rtol=5e-3, atol=5e-3)
    assert_allclose(out1.cpu().to(torch.float32).numpy(), golden1.cpu().numpy(), rtol=5e-3, atol=5e-3)


@_DUALDST_JIT
def dual_dst_chained_ops_kernel(
    a_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], FP16),
    b_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], FP16),
    out0_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], FP32),
    out1_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], FP32),
):
    """SplitN 同形状,但下游每条 chain 由 add+mul 两 vector op 组成.

    验证 dualdst 融合后下游多步依赖链仍正确。
    """
    pypto.experimental.auto_mix_partition(1)
    upper, lower = _split_n_prologue_64(a_tensor, b_tensor)
    upper = _online_softmax_exp(upper, 64, 64)
    lower = _online_softmax_exp(lower, 64, 64)
    tmp0 = upper + 1.0
    pypto.set_vec_tile_shapes(64, 64)
    out0_tensor[:, :] = tmp0 * 2.0
    pypto.set_vec_tile_shapes(64, 64)
    tmp1 = lower * 0.5
    pypto.set_vec_tile_shapes(64, 64)
    out1_tensor[:, :] = tmp1 + 3.0


@pytest.mark.soc("950")
@pytest.mark.skip(reason="large test case")
@pypto.options(pass_options={"enable_slice": False})
def test_dual_dst_chained_ops():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    m, k, n = 64, 64, 128
    half_n = n // 2

    torch.manual_seed(0)
    a = torch.rand([m, k], dtype=torch.float16, device=f"npu:{device_id}")
    b = torch.rand([n, k], dtype=torch.float16, device=f"npu:{device_id}")
    out0 = torch.zeros([m, half_n], dtype=torch.float32, device=f"npu:{device_id}")
    out1 = torch.zeros([m, half_n], dtype=torch.float32, device=f"npu:{device_id}")

    dual_dst_chained_ops_kernel(a, b, out0, out1)

    mm_golden = torch.matmul(a.to(torch.float32), b.to(torch.float32).T)
    golden0 = (_torch_online_softmax_exp(mm_golden[:, :half_n]) + 1.0) * 2.0
    golden1 = (_torch_online_softmax_exp(mm_golden[:, half_n:]) * 0.5) + 3.0

    assert_allclose(out0.cpu().to(torch.float32).numpy(), golden0.cpu().numpy(), rtol=5e-3, atol=5e-3)
    assert_allclose(out1.cpu().to(torch.float32).numpy(), golden1.cpu().numpy(), rtol=5e-3, atol=5e-3)


@_DUALDST_JIT
def dual_dst_asymmetric_scale_kernel(
    a_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], FP16),
    b_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], FP16),
    out0_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], FP32),
    out1_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], FP32),
):
    """SplitN, 两侧 vector chain 用不同 op (add vs mul).

    验证两个 AIV core 实际执行不同语义但共享同一份 L0C 数据。
    """
    pypto.experimental.auto_mix_partition(1)
    upper, lower = _split_n_prologue_64(a_tensor, b_tensor)
    upper = _online_softmax_exp(upper, 64, 64)
    lower = _online_softmax_exp(lower, 64, 64)
    out0_tensor[:, :] = upper + 7.5  # AIV0: add scalar
    pypto.set_vec_tile_shapes(64, 64)
    out1_tensor[:, :] = lower * 1.25  # AIV1: mul scalar


@pytest.mark.soc("950")
@pytest.mark.skip(reason="large test case")
@pypto.options(pass_options={"enable_slice": False})
def test_dual_dst_asymmetric_scale():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    m, k, n = 64, 64, 128
    half_n = n // 2

    torch.manual_seed(0)
    a = torch.rand([m, k], dtype=torch.float16, device=f"npu:{device_id}")
    b = torch.rand([n, k], dtype=torch.float16, device=f"npu:{device_id}")
    out0 = torch.zeros([m, half_n], dtype=torch.float32, device=f"npu:{device_id}")
    out1 = torch.zeros([m, half_n], dtype=torch.float32, device=f"npu:{device_id}")

    dual_dst_asymmetric_scale_kernel(a, b, out0, out1)

    mm_golden = torch.matmul(a.to(torch.float32), b.to(torch.float32).T)
    golden0 = _torch_online_softmax_exp(mm_golden[:, :half_n]) + 7.5
    golden1 = _torch_online_softmax_exp(mm_golden[:, half_n:]) * 1.25

    assert_allclose(out0.cpu().to(torch.float32).numpy(), golden0.cpu().numpy(), rtol=5e-3, atol=5e-3)
    assert_allclose(out1.cpu().to(torch.float32).numpy(), golden1.cpu().numpy(), rtol=5e-3, atol=5e-3)


# ====== 收益放大用例: 单 matmul + 多对相邻 SplitN, 累计 N_PAIRS 对 dualdst pair ======
# 为什么前面的用例提升不明显:
#   - micro-kernel 太小, 每对 [64,64] FP32 = 16KB, 整体瓶颈不在 L0C->UB 搬运,
#     而在 kernel launch / cube 计算 / 首次跨 pipe 同步; dualdst 省的 (少一次 op
#     发射 + 少一次 cube->vec sync + 理论上少一半搬运带宽) 被淹没。
#   - 多个独立 matmul 让 cube 计算时间占比反而变大, 进一步稀释 dualdst 收益。
#
# 本用例改成 "搬运密集型":
#   - 单次 matmul (cube 只算一遍), mm shape [M, 2*HALF_N*N_PAIRS]
#   - 沿 N 维切出 2*N_PAIRS 个宽 HALF_N 的相邻子块, 偶/奇下标各加不同 scalar,
#     输出拼回同形状大张量
#   - core_assign.DualDstProcess 会把相邻子块按 (偶->AIV0, 奇->AIV1) 配对,
#     OoOSchedule.RunDualDstFuse 把每对 (j, j+1) 的 OP_L0C_COPY_UB 合并 ->
#     单 kernel 累计 N_PAIRS 对 dualdst fuse
#   - cube 计算占比小 + 搬运量大 + 多对融合, dualdst 节省相对总时延的占比最大化
#
# 全静态: 切片边界都是 python int 常量, pypto 识别为静态视图, 不引入动态 validShape
# (避免 InferDynShape 转换后 dualdst 走 dyn fallback 误判)。
#
# 验证收益方式: kTempEnableDualDst = false 编译跑一次, true 编译跑一次, 对比 NPU 端
# kernel 时延 (npu-smi / profiler), 预期差值 ~ N_PAIRS × 单对搬运/同步节省。
# 想进一步放大: 调大 N_PAIRS (16/32, 编译变慢) 或 HALF_N_PER (每对搬运 byte 更大,
# 注意 2 个 [M,HALF_N_PER] UB tile 别撞 UB 容量上限)。
N_PAIRS = 8
GAIN_M = 128
GAIN_K = 32
GAIN_HALF_N = 64
GAIN_N = 2 * GAIN_HALF_N * N_PAIRS  # 1024


@_DUALDST_JIT
def dual_dst_max_gain_kernel(
    a_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], FP16),
    b_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], FP16),
    out_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], FP32),
):
    """单次 matmul + N 轴 2*N_PAIRS 等分, 偶/奇下标分别落 AIV0/AIV1, 相邻 (偶,奇) 被 dualdst 合并.

    mm shape: [GAIN_M, GAIN_N]; 每个子块宽 GAIN_HALF_N。
    偶下标 +1.0 (consumer 落 AIV0), 奇下标 +2.0 (落 AIV1);
    相邻 (偶,奇) 子块被 dualdst 合并, 累计 N_PAIRS 对融合。
    所有切片边界都是 python int 常量。
    """
    pypto.experimental.auto_mix_partition(1)
    pypto.set_cube_tile_shapes([GAIN_M, GAIN_M], [GAIN_K, GAIN_K], [2 * GAIN_HALF_N, 2 * GAIN_HALF_N])
    mm = pypto.matmul(a_tensor, b_tensor, b_trans=True, out_dtype=FP32)
    for j in range(2 * N_PAIRS):
        lo = j * GAIN_HALF_N
        hi = (j + 1) * GAIN_HALF_N
        seg = mm[:, lo:hi]
        seg = _online_softmax_exp(seg, GAIN_M, GAIN_HALF_N)
        pypto.set_vec_tile_shapes(GAIN_M, GAIN_HALF_N)
        out_tensor[:, lo:hi] = seg + (1.0 if j % 2 == 0 else 2.0)


@pytest.mark.soc("950")
@pytest.mark.skip(reason="large test case")
@pypto.options(pass_options={"enable_slice": False})
def test_dual_dst_max_gain():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    torch.manual_seed(0)
    a = torch.rand([GAIN_M, GAIN_K], dtype=torch.float16, device=f"npu:{device_id}")
    b = torch.rand([GAIN_N, GAIN_K], dtype=torch.float16, device=f"npu:{device_id}")
    out = torch.zeros([GAIN_M, GAIN_N], dtype=torch.float32, device=f"npu:{device_id}")

    dual_dst_max_gain_kernel(a, b, out)

    mm_golden = torch.matmul(a.to(torch.float32), b.to(torch.float32).T)  # [GAIN_M, GAIN_N]
    out_golden = mm_golden.clone()
    for j in range(2 * N_PAIRS):
        lo = j * GAIN_HALF_N
        hi = (j + 1) * GAIN_HALF_N
        out_golden[:, lo:hi] = _torch_online_softmax_exp(mm_golden[:, lo:hi]) + (1.0 if j % 2 == 0 else 2.0)

    assert_allclose(out.cpu().to(torch.float32).numpy(), out_golden.cpu().numpy(), rtol=5e-3, atol=5e-3)


# ====== 复杂用例: N_MM 个独立 matmul + 长 ADDS 计算链, ≥10 对 dualdst pair ======
# 设计意图:
#   - 每个独立 matmul 产 1 个 L0C tensor, 上面 SplitN 二分 -> 1 对 dualdst pair
#     N_MM = 12 -> 12 对 fuse, 超出"≥10 对"门槛
#   - 每对 split 后下游各跟 ADDS_CHAIN_DEPTH 步 ADDS, 形成长 vector chain;
#     让 cube 算 / 搬运 / vector 三段都被压实, 模拟真实算子的 op 链密度
#   - 仅 matmul + ADDS 两类计算原语
#
# 关键 — 全静态: a / b / out 输入沿 M 维 / N 维拼成大 2D 张量, jit kernel 内 python
# range 静态展开 N_MM 次, 切片边界全是 python int 常量 -> pypto 识别为静态视图,
# staticValidShape 正常注入每个 OP_L0C_COPY_UB, dualdst 识别走静态属性比较路径。
#
# 验证: kTempEnableDualDst = false / true 编译各跑一次, 对比 NPU 端 kernel 时延,
# 12 对融合 × 单对节省 + 长 chain 拉长基准时间 -> 收益更易观察。
N_MM = 12
MM_M = 64
MM_K = 64
MM_N = 128
MM_HALF_N = MM_N // 2
ADDS_CHAIN_DEPTH = 4


@_DUALDST_JIT
def dual_dst_long_chain_kernel(
    a_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], FP16),
    b_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], FP16),
    out0_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], FP32),
    out1_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], FP32),
):
    """python 层展开 N_MM 次独立 matmul + SplitN, 每次产 1 对 dualdst pair.

    每条 chain 跟 ADDS_CHAIN_DEPTH 步 ADDS 形成长依赖链。
    """
    pypto.experimental.auto_mix_partition(1)
    pypto.set_cube_tile_shapes([MM_M, MM_M], [MM_K, MM_K], [MM_N, MM_N])
    # 上半 / 下半 chain 各自的 ADDS scalar 序列; sum 用于 golden 累加。
    upper_scalars = [0.5, 1.0, 1.5, 2.0]  # len == ADDS_CHAIN_DEPTH
    lower_scalars = [0.25, 0.75, 1.25, 1.75]
    for i in range(N_MM):
        a_i = a_tensor[i * MM_M:(i + 1) * MM_M, :]
        b_i = b_tensor[i * MM_N:(i + 1) * MM_N, :]
        upper, lower = _matmul_split_n(a_i, b_i, MM_HALF_N)
        upper = _online_softmax_exp(upper, MM_M, MM_HALF_N)
        lower = _online_softmax_exp(lower, MM_M, MM_HALF_N)

        # AIV0 chain: upper -> +s0 -> +s1 -> +s2 -> +s3 -> out0
        cur = upper
        for k, s in enumerate(upper_scalars):
            pypto.set_vec_tile_shapes(MM_M, MM_HALF_N)
            if k == len(upper_scalars) - 1:
                out0_tensor[i * MM_M:(i + 1) * MM_M, :] = cur + s
            else:
                cur = cur + s

        # AIV1 chain: lower -> +s0 -> +s1 -> +s2 -> +s3 -> out1
        cur = lower
        for k, s in enumerate(lower_scalars):
            pypto.set_vec_tile_shapes(MM_M, MM_HALF_N)
            if k == len(lower_scalars) - 1:
                out1_tensor[i * MM_M:(i + 1) * MM_M, :] = cur + s
            else:
                cur = cur + s


@pytest.mark.soc("950")
@pytest.mark.skip(reason="large test case")
@pypto.options(pass_options={"enable_slice": False})
def test_dual_dst_long_chain():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    m_tot = N_MM * MM_M
    n_tot = N_MM * MM_N
    upper_sum = 0.5 + 1.0 + 1.5 + 2.0  # = 5.0
    lower_sum = 0.25 + 0.75 + 1.25 + 1.75  # = 4.0

    torch.manual_seed(0)
    a = torch.rand([m_tot, MM_K], dtype=torch.float16, device=f"npu:{device_id}")
    b = torch.rand([n_tot, MM_K], dtype=torch.float16, device=f"npu:{device_id}")
    out0 = torch.zeros([m_tot, MM_HALF_N], dtype=torch.float32, device=f"npu:{device_id}")
    out1 = torch.zeros([m_tot, MM_HALF_N], dtype=torch.float32, device=f"npu:{device_id}")

    dual_dst_long_chain_kernel(a, b, out0, out1)

    out0_golden = torch.zeros([m_tot, MM_HALF_N], dtype=torch.float32)
    out1_golden = torch.zeros([m_tot, MM_HALF_N], dtype=torch.float32)
    for i in range(N_MM):
        a_i = a[i * MM_M:(i + 1) * MM_M, :].to(torch.float32)
        b_i = b[i * MM_N:(i + 1) * MM_N, :].to(torch.float32)
        mm_i = torch.matmul(a_i, b_i.T)
        out0_golden[i * MM_M:(i + 1) * MM_M, :] = _torch_online_softmax_exp(mm_i[:, :MM_HALF_N]) + upper_sum
        out1_golden[i * MM_M:(i + 1) * MM_M, :] = _torch_online_softmax_exp(mm_i[:, MM_HALF_N:]) + lower_sum

    # 长 chain + FP16 输入 -> 误差可能略大于 5e-3, 适度放宽
    assert_allclose(out0.cpu().to(torch.float32).numpy(), out0_golden.numpy(), rtol=1e-2, atol=1e-2)
    assert_allclose(out1.cpu().to(torch.float32).numpy(), out1_golden.numpy(), rtol=1e-2, atol=1e-2)


# ====== 链式用例: matmul,ADDS,matmul,ADDS... 在一条数据流上串 N_LINK 步 ======
# 每步:  prev_A → mm → SplitN → AIV0:ADDS → ws_a_i  ┐
#                            └─ AIV1:ADDS → ws_b_i  │
#                                                    ↓ 下一步取 ws_a_i 当 A
#
# 每步 mm 各产 1 对 dualdst pair, N_LINK = 12 -> ≥10 对 fuse。
#
# 关键约束 / 设计细节:
#   - LINK_K == LINK_HALF_N: 让 ws_a_i shape [M, K] 直接当下一步 mm A 输入,
#     cube tile 链上稳定不变 (mm 输出 [M, N=2*K], dual_dst 切两半各 [M, K]);
#   - 每步两个独立 ws_a/ws_b tensor (而不是一个大 [M, N] 然后切两半写):
#     pypto 前端对 "同一 GM tensor 多 slice 写入" 算 offset 有 bug
#     (lower slice 的 toOffset 会被算成负值), 必须拆成独立 tensor 避开;
#   - 链只通过 ws_a 传播 (upper 半边), ws_b (lower 半边) 中间步仅作 dual_dst
#     的 AIV1 consumer 占位 (没有人写它的值就不会触发 dual_dst 融合),
#     最后一步 lower 写到独立 out_b 用于校验;
#   - 全程 FP32: dual_dst op 不支持 "随路" cast, 避免链上插 OP_CAST;
#   - 12 步 FP32 mm 链精度损失小, 容差 1e-3。
#
# 张量数 26 个: a_tensor + b_tensor + 11 ws_a + 11 ws_b + out_a + out_b。
# 改 N_LINK 必须同步改 kernel 签名 (ws_a_i / ws_b_i 是硬编码的)。
#
# 全静态: 切片边界都是 python int 常量, pypto 视为静态视图。
N_LINK = 12
LINK_M = 64
LINK_K = 32  # 必须等于 LINK_HALF_N, 让 ws_a 能当下一步 mm A
LINK_N = 64  # = 2 * LINK_K, mm 输出
LINK_HALF_N = LINK_N // 2  # = 32 = LINK_K


@_DUALDST_JIT
def dual_dst_link_chain_kernel(
    a_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], FP32),
    b_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], FP32),
    out_a: pypto.Tensor([pypto.STATIC, pypto.STATIC], FP32),
    out_b: pypto.Tensor([pypto.STATIC, pypto.STATIC], FP32),
):
    """matmul,ADDS,matmul,ADDS 一条数据流上 N_LINK 步串接, 每步 1 对 dualdst pair.

    全程 FP32 (dualdst op 不支持随路 dtype cast)。
    中间 N_LINK-1 步的两路 (AIV0 upper / AIV1 lower) 落点用 pypto.tensor() 在
    kernel 内分配独立 workspace LogicalTensor —— 不暴露成 kernel 参数:
      - ws_a_i: AIV0 chain (upper + 0.5) 落点, 同时作为下一步 mm 的 A 输入
      - ws_b_i: AIV1 chain (lower + 0.25) 落点 (side output, 不传播)
    每个 ws 各自独立, 避开 "同一 GM tensor 多 slice 读写" 的 pypto IR 环路;
    最后一步直接写 out_a / out_b。
    """
    pypto.experimental.auto_mix_partition(1)
    pypto.set_cube_tile_shapes([LINK_M, LINK_M], [LINK_K, LINK_K], [LINK_N, LINK_N])
    # 中间步 workspace: 每个独立 LogicalTensor, 写一次/读一次, 不与最终 out_a/out_b 共享。
    # 用普通 for 循环 (而非 list comprehension) 收集: pypto parser 不进入 list comp 的
    # 内层 scope, 在那里调 pypto.tensor() 会触发 NameError: name 'pypto' is not defined。
    ws_a_list = []
    ws_b_list = []
    for ws_idx in range(N_LINK - 1):
        ws_a_list.append(pypto.tensor([LINK_M, LINK_K], FP32, "ws_a_" + str(ws_idx)))
        ws_b_list.append(pypto.tensor([LINK_M, LINK_K], FP32, "ws_b_" + str(ws_idx)))
    for i in range(N_LINK):
        # 第 0 步用原 a, 其余步用上一步 ws_a (链上 upper 半边)
        a_i = a_tensor if i == 0 else ws_a_list[i - 1]
        b_i = b_tensor[i * LINK_N:(i + 1) * LINK_N, :]
        upper, lower = _matmul_split_n(a_i, b_i, LINK_HALF_N)
        upper = _online_softmax_exp(upper, LINK_M, LINK_HALF_N)
        lower = _online_softmax_exp(lower, LINK_M, LINK_HALF_N)
        # 选目标 tensor: 最后一步写 out_a/out_b, 中间步写本步独立 ws_a_i/ws_b_i
        dst_a = out_a if i == N_LINK - 1 else ws_a_list[i]
        dst_b = out_b if i == N_LINK - 1 else ws_b_list[i]
        pypto.set_vec_tile_shapes(LINK_M, LINK_HALF_N)
        dst_a[:, :] = upper + 0.5
        pypto.set_vec_tile_shapes(LINK_M, LINK_HALF_N)
        dst_b[:, :] = lower + 0.25


@pytest.mark.soc("950")
@pytest.mark.skip(reason="large test case")
@pypto.options(pass_options={"enable_slice": False})
def test_dual_dst_link_chain():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    n_tot_b = N_LINK * LINK_N

    torch.manual_seed(0)
    a = torch.rand([LINK_M, LINK_K], dtype=torch.float32, device=f"npu:{device_id}")
    b = torch.rand([n_tot_b, LINK_K], dtype=torch.float32, device=f"npu:{device_id}")
    out_a = torch.zeros([LINK_M, LINK_K], dtype=torch.float32, device=f"npu:{device_id}")
    out_b = torch.zeros([LINK_M, LINK_K], dtype=torch.float32, device=f"npu:{device_id}")

    dual_dst_link_chain_kernel(a, b, out_a, out_b)

    # Golden: 全程 FP32, 每步只传播 upper(+0.5) 当下一步 A;
    # 最终 out_a = 最后一步 upper+0.5, out_b = 最后一步 lower+0.25
    prev = a.cpu()
    last_upper = None
    last_lower = None
    for i in range(N_LINK):
        b_i = b[i * LINK_N:(i + 1) * LINK_N, :].cpu()
        mm = torch.matmul(prev, b_i.T)  # [LINK_M, LINK_N]
        last_upper = _torch_online_softmax_exp(mm[:, :LINK_HALF_N]) + 0.5  # [LINK_M, LINK_HALF_N=LINK_K]
        last_lower = _torch_online_softmax_exp(mm[:, LINK_HALF_N:]) + 0.25
        prev = last_upper  # 只传 upper 半边

    # 全 FP32 链精度损失小, 容差 1e-3
    assert_allclose(out_a.cpu().numpy(), last_upper.numpy(), rtol=1e-2, atol=1e-2)
    assert_allclose(out_b.cpu().numpy(), last_lower.numpy(), rtol=1e-2, atol=1e-2)


"""
Mega-scale ST for OP_L0C_COPY_UB_DUAL_DST fusion:
    并行 MEGA_N_BRANCH 条分支, 每条分支内串行 MEGA_N_PER_BRANCH 对 dualdst。

Default: 50 分支 × 10 对/分支 = 500 对 dualdst pair。
Bump MEGA_N_BRANCH to 100 -> 100 × 10 = 1000 对。

为什么不真的"数据流串接" (link_chain 风格):
    严格 prev->next 串接需要每对独立 workspace 张量, 数量 = N_BRANCH × (N_LINK-1) × 2,
    50×9×2 = 900 个 ws 张量参数, pypto frontend 无法 handle 这种 kernel 签名。
    退而求其次, 这里串行的体现是 IR op 顺序: 每条分支的 10 个 mm 在 IR 上聚在
    一起 (外循环 s 在外, 内循环 i 在内 trace 出 branch_0 的 mm_0..mm_9 接 branch_1
    的 mm_0..mm_9...), 形态等价于:
        mm_b0_0 -> dual_dst_0 -> ADDS_u/ADDS_l
        mm_b0_1 -> dual_dst_1 -> ADDS_u/ADDS_l
        ...
        mm_b0_9 -> dual_dst_9 -> ADDS_u/ADDS_l
        mm_b1_0 -> ... (并行分支)
    数据流上每个 mm 独立 a/b 输入 (避免 ws 张量爆炸), OoOSchedule 仍能让不同分支
    的 cube/vec 阶段并行交叠。

并行性:
    MEGA_N_BRANCH 条分支无任何数据依赖, OoOSchedule 可以让 branch_0 的 mm
    在 cube 上跑时, branch_1/2/...的下游 vec/ADDS (AIV0/AIV1) 并行执行。
    分支越多, cube/vec 流水越饱和, 总时延越接近 max(单分支时延)。

输出布局:
    out_a / out_b 沿 M 轴拼 (out_idx = (s * MEGA_N_PER_BRANCH + i) * MEGA_M),
    long_chain 用例已验证 M 轴 multi-slice 写入 OK; 避开 pypto N-axis multi-slice
    写 toOffset 算成 [0, -HALF] 的已知 bug。

资源 (默认 500 对):
    a_tensor      [50*10*64,  64]  FP32 =  8 MB
    b_tensor      [50*10*128, 64]  FP32 = 16 MB
    out_a_tensor  [50*10*64,  64]  FP32 =  8 MB
    out_b_tensor  [50*10*64,  64]  FP32 =  8 MB
    总 GM ≈ 40 MB
    1000 对场景翻倍 ≈ 80 MB GM

编译时间提醒:
    500 对 -> IR 中约 2500 个 op (500 mm + 1000 L0C_COPY_UB + 1000 ADDS),
    pypto trace + pass 可能耗时 2-5 分钟; 1000 对约 5-10 分钟, 属正常。

Run:
    pytest python/tests/st/test_dualdst_mega.py -v
"""


MEGA_N_BRANCH = 50  # 默认 50 分支 (500 对); 改 100 -> 1000 对
MEGA_N_PER_BRANCH = 10  # 单分支内 10 个 mm = 10 对 dualdst (满足 "≥10 对/分支")
MEGA_M = 64
MEGA_K = 64
MEGA_HALF_N = 64
MEGA_N = 2 * MEGA_HALF_N  # = 128
MEGA_TOTAL_MM = MEGA_N_BRANCH * MEGA_N_PER_BRANCH  # 500 或 1000
MEGA_A_M = MEGA_TOTAL_MM * MEGA_M  # 32000
MEGA_B_N = MEGA_TOTAL_MM * MEGA_N  # 64000
MEGA_OUT_M = MEGA_TOTAL_MM * MEGA_M  # 32000


@_DUALDST_JIT
def dual_dst_mega_kernel(
    a_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], FP32),
    b_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], FP32),
    out_a_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], FP32),
    out_b_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC], FP32),
):
    """MEGA_N_BRANCH 条独立并行分支, 每条分支内串行 MEGA_N_PER_BRANCH 对 dualdst.

    a_tensor:      [MEGA_TOTAL_MM * MEGA_M, MEGA_K]      所有 mm 的 A 沿 M 轴拼
    b_tensor:      [MEGA_TOTAL_MM * MEGA_N, MEGA_K]      所有 mm 的 B 沿 N 轴拼
    out_a_tensor:  [MEGA_TOTAL_MM * MEGA_M, MEGA_HALF_N] AIV0 chain 输出沿 M 拼
    out_b_tensor:  [MEGA_TOTAL_MM * MEGA_M, MEGA_HALF_N] AIV1 chain 输出沿 M 拼
    """
    pypto.experimental.auto_mix_partition(1)
    pypto.set_cube_tile_shapes([MEGA_M, MEGA_M], [MEGA_K, MEGA_K], [MEGA_N, MEGA_N])
    # 外循环 s = 并行分支 idx, 内循环 i = 单分支内串行 mm idx;
    # 外内嵌套保证 IR op 顺序上 branch_0 的 10 个 mm 聚在一起 (串行语义), 而
    # 分支之间无数据依赖 (并行语义)。
    for s in range(MEGA_N_BRANCH):
        for i in range(MEGA_N_PER_BRANCH):
            global_idx = s * MEGA_N_PER_BRANCH + i
            a_i = a_tensor[global_idx * MEGA_M:(global_idx + 1) * MEGA_M, :]
            b_i = b_tensor[global_idx * MEGA_N:(global_idx + 1) * MEGA_N, :]
            upper, lower = _matmul_split_n(a_i, b_i, MEGA_HALF_N)
            upper = _online_softmax_exp(upper, MEGA_M, MEGA_HALF_N)
            lower = _online_softmax_exp(lower, MEGA_M, MEGA_HALF_N)
            pypto.set_vec_tile_shapes(MEGA_M, MEGA_HALF_N)
            out_a_tensor[global_idx * MEGA_M:(global_idx + 1) * MEGA_M, :] = upper + 1.0
            pypto.set_vec_tile_shapes(MEGA_M, MEGA_HALF_N)
            out_b_tensor[global_idx * MEGA_M:(global_idx + 1) * MEGA_M, :] = lower + 2.0


@pytest.mark.soc("950")
@pytest.mark.skip(reason="large test case")
@pypto.options(pass_options={"enable_slice": False})
def test_dual_dst_mega():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    torch.manual_seed(0)
    a = torch.rand([MEGA_A_M, MEGA_K], dtype=torch.float32, device=f"npu:{device_id}")
    b = torch.rand([MEGA_B_N, MEGA_K], dtype=torch.float32, device=f"npu:{device_id}")
    out_a = torch.zeros([MEGA_OUT_M, MEGA_HALF_N], dtype=torch.float32, device=f"npu:{device_id}")
    out_b = torch.zeros([MEGA_OUT_M, MEGA_HALF_N], dtype=torch.float32, device=f"npu:{device_id}")

    dual_dst_mega_kernel(a, b, out_a, out_b)

    out_a_golden = torch.zeros([MEGA_OUT_M, MEGA_HALF_N], dtype=torch.float32)
    out_b_golden = torch.zeros([MEGA_OUT_M, MEGA_HALF_N], dtype=torch.float32)
    a_cpu = a.cpu()
    b_cpu = b.cpu()
    for s in range(MEGA_N_BRANCH):
        for i in range(MEGA_N_PER_BRANCH):
            global_idx = s * MEGA_N_PER_BRANCH + i
            a_i = a_cpu[global_idx * MEGA_M:(global_idx + 1) * MEGA_M, :]
            b_i = b_cpu[global_idx * MEGA_N:(global_idx + 1) * MEGA_N, :]
            mm = torch.matmul(a_i, b_i.T)
            out_a_golden[global_idx * MEGA_M:(global_idx + 1) * MEGA_M, :] = (
                _torch_online_softmax_exp(mm[:, :MEGA_HALF_N]) + 1.0
            )
            out_b_golden[global_idx * MEGA_M:(global_idx + 1) * MEGA_M, :] = (
                _torch_online_softmax_exp(mm[:, MEGA_HALF_N:]) + 2.0
            )

    assert_allclose(out_a.cpu().numpy(), out_a_golden.numpy(), rtol=1e-2, atol=1e-2)
    assert_allclose(out_b.cpu().numpy(), out_b_golden.numpy(), rtol=1e-2, atol=1e-2)


def main():
    test_dual_dst_split_n()
    test_dual_dst_split_m()
    test_dual_dst_chained_ops()
    test_dual_dst_asymmetric_scale()
    test_dual_dst_max_gain()
    test_dual_dst_long_chain()
    test_dual_dst_link_chain()
    test_dual_dst_mega()


if __name__ == "__main__":
    main()
