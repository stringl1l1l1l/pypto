#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""AICORE Error 触发测试 — 通过 store_tile 越界写入触发硬件异常。

验证 CANN 原生异常 dump 机制在 PyPTO Pro 下的可用性，以及
debug_aicore_error_pro_repro.py 离线复现工具的端到端流程：

1. test_oob_load_aicore_error：独立子进程验证读侧（load_tile 越界）异常经
   torch.npu.synchronize() 上抛给用户代码（多输入多输出、FP16 + FP32
   混合 kernel），随后执行 dump 产物断言与离线复现拦截（复现报错、
   ErrorPC 定位、device_linked.so 落盘），覆盖纯 DYNAMIC tensor kernel
   的 dyn 参数恢复路径。
2. test_fusion_dump_and_repro：子进程隔离触发 cv 融合 kernel（cube matmul
   + vec 越界 store）的异常，带 tiling_data + tiling_key，断言 dump 产物
   （数据文件 + -g 重编译链接的 _call_kernel.so）生成，离线复现工具能从
   dump 恢复 tensor 与 tiling、复现异常，并把 ErrorPC 定位到源码行。
3. test_cube_dump_and_repro：纯 cube kernel（dav-c310-cube）matmul 后
   fixpipe（Acc->GM）越界写出的异常，与前两个用例的 MTE2 读 / MTE3 写
   构成不同管线错误，交叉验证定位正确性。
4. test_ptr_scalar_dump_and_repro：纯 pl.Ptr + scalar 入参的 kernel
   （make_tensor 在函数体内构造形状），覆盖无 shape 签名的 ABI 恢复
   路径——scalar 值与 PTR addr/size 均依赖 launch-args sidecar。

注意：
  - 不要设置 NPU_COLLECT_PATH，否则 dump 模式变为 norm_dump，不生成 dump 文件。
    默认 brief_dump 模式将 dump 文件写入 ASCEND_WORK_PATH/extra-info/data-dump/<device_id>/。
  - 两个用例的触发均放在独立子进程中做：xdist/fork worker 继承的 NPU 上下文
    会让 device error 无法经 synchronize() 上抛（DID NOT RAISE），干净子进程
    与真实用户进程等价；brief_dump 的一次性模式也要求用例间互不干扰。

前置条件：
  export ASCEND_WORK_PATH=./wk
"""

import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

pytestmark = pytest.mark.skip_jit_discovery(
    reason="Fault-trigger and offline-repro subprocesses cannot populate the parent process JIT cache"
)

_REPO_ROOT = Path(__file__).resolve().parents[6]
_REPRO_SCRIPT = _REPO_ROOT / "tools" / "scripts" / "debug_aicore_error_pro_repro.py"


def _check_npu():
    try:
        torch.npu.set_device(ST_DEVICE)
        name = torch.npu.get_device_name()
        if "Ascend950" not in name:
            pytest.skip(f"Device {name} is not A5 (Ascend950). Skip.")
        return True
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
        return False


def _make_work_dir(name: str) -> Path:
    base = os.environ.get("ASCEND_WORK_PATH", "./test_log")
    work_dir = Path(base).resolve() / name
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def _make_env(work_dir: str) -> dict:
    env = os.environ.copy()
    env["ASCEND_WORK_PATH"] = work_dir
    env["TILE_FWK_DEVICE_ID"] = str(ST_DEVICE_ID)
    return env


def _run_subprocess(script_path: Path, work_dir: str, timeout: int = 300) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True, timeout=timeout, env=_make_env(work_dir), cwd=work_dir,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=[sys.executable, str(script_path)], returncode=-1, stdout="", stderr="subprocess timed out"
        )


def _run_repro(work_dir: str, timeout: int = 300) -> subprocess.CompletedProcess:
    dump_dir = os.path.join(work_dir, "extra-info", "data-dump", str(ST_DEVICE_ID))
    try:
        return subprocess.run(
            [sys.executable, str(_REPRO_SCRIPT), "-p", dump_dir],
            capture_output=True, text=True, timeout=timeout, env=_make_env(work_dir), cwd=work_dir,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=[sys.executable, str(_REPRO_SCRIPT)], returncode=-1, stdout="", stderr="repro subprocess timed out"
        )


def _check_dump_files(work_dir: str) -> tuple[bool, list[str]]:
    """dump 目录包含数据文件、-g 的 call_kernel.so 与 *.cpp 源码副本才算通过。

    该 .so 由异常回调用与 JIT 完全一致的参数加 -g 重编译链接而来,device 代码
    与被 launch 的二进制 bit 级一致,同时携带 DWARF 行表,是复现执行与
    ErrorPC 源码行解析共用的唯一产物;kernel/call_kernel 源码与被 launch 的
    原始 JIT .so(call_kernel_<digest>.so)一并复制进来,
    使 dump 目录完全自包含、不依赖 build 目录。
    """
    dump_dir = os.path.join(work_dir, "extra-info", "data-dump", str(ST_DEVICE_ID))
    if not os.path.isdir(dump_dir):
        return False, []
    files = os.listdir(dump_dir)
    has_dump = any(not f.endswith(("_host.o", "_debug.o", ".so", ".cpp")) for f in files)
    has_so_copy = any(f.endswith("_call_kernel.so") for f in files)
    has_sources = any(f.endswith(".cpp") for f in files)
    has_jit_so = any(re.fullmatch(r"call_kernel_[0-9a-f]+\.so", f) for f in files)
    return has_dump and has_so_copy and has_sources and has_jit_so, files


# ---------------------------------------------------------------------------
# 1. 异常上抛（干净子进程）
# ---------------------------------------------------------------------------

_OOB_LOAD_TRIGGER = """\
#!/usr/bin/env python3
from __future__ import annotations
import os, sys, logging
import torch
import pypto_pro.language as pl

logging.basicConfig(level=logging.INFO, format="%(message)s")


@pl.jit(auto_mutex=True)
def oob_load_tile_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out0: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out1: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    \"\"\"load_tile 从 [65535, 0] 越界读取（MTE 读路径），远超 tensor GM 边界。\"\"\"
    tt_fp16 = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tt_fp32 = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt_fp16, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt_fp32, addrs=0x4000, mutex_ids=[1])
    tile_out0 = pl.make_tile_group(type=tt_fp16, addrs=0x0000, mutex_ids=[2])
    tile_out1 = pl.make_tile_group(type=tt_fp32, addrs=0x4000, mutex_ids=[3])

    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out0 = tile_out0.current()
        cur_out1 = tile_out1.current()
        pl.load_tile(cur_a, a, [65535, 0])
        pl.load_tile(cur_b, b, [0, 0])
        pl.store_tile(out0, cur_out0, [0, 0])
        pl.store_tile(out1, cur_out1, [0, 0])


dev = f"npu:{os.environ.get('TILE_FWK_DEVICE_ID', '0')}"
torch.npu.set_device(dev)
a = torch.rand(64, 64, device=dev, dtype=torch.float16)
b = torch.rand(64, 64, device=dev, dtype=torch.float32)
out0 = torch.empty(64, 64, device=dev, dtype=torch.float16)
out1 = torch.empty(64, 64, device=dev, dtype=torch.float32)

oob_load_tile_kernel(a, b, out0, out1)
try:
    torch.npu.synchronize()
    print("ERROR: No AICORE error raised")
    sys.exit(1)
except Exception as e:
    print(f"AICORE error raised: {e}")

print("DONE")
"""


@pytest.mark.soc("950")
def test_oob_load_aicore_error():
    """load_tile 越界读取触发 AICORE 异常：上抛 + dump + 离线复现 + 定位。

    干净子进程验证读侧（MTE load）越界异常经 torch.npu.synchronize()
    上抛（多输入多输出、FP16 + FP32 混合、无 tiling 参数）；随后走完整
    dump 断言与离线复现拦截。ErrorPC 预期落在读侧 intrinsic
    （TLoad.hpp / copy_gm_to_ubuf），与写侧越界（TStore.hpp /
    copy_ubuf_to_gm）区分，交叉验证定位的正确性。
    """
    _check_npu()
    logging.info("------------test_oob_load_aicore_error--------------")

    work_dir = _make_work_dir("oob_load_aicore_error")
    trigger = work_dir / "trigger_oob_load.py"
    trigger.write_text(_OOB_LOAD_TRIGGER)

    result = _run_subprocess(trigger, str(work_dir))
    assert result.returncode == 0, \
        f"AICORE error not raised via synchronize():\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert "AICORE error raised" in result.stdout, \
        f"AICORE error not raised via synchronize():\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert "DONE" in result.stdout, f"trigger script not completed:\n{result.stdout}"

    _assert_dump_and_repro(str(work_dir))


# ---------------------------------------------------------------------------
# 2. dump 产物 + 离线复现（子进程隔离）
# ---------------------------------------------------------------------------

# 同时带 tiling_key（TkMode，编译到 tk_<packed> 子目录）和 tiling_data
# （SimpleTiling dataclass，dump 回调应捕获为 WORKSPACE tensor）。
# cv 融合 kernel：cube 段完整 matmul（GM->L1->L0A/L0B->L0C，Acc 走
# fixpipe 写出），vec 段 load 后越界 store_tile 触发异常——覆盖融合
# 架构（dav-c310 + --cce-fatobj-link）的 -g 重编译与定位链路。
_DUMP_REPRO_TRIGGER = """\
#!/usr/bin/env python3
from __future__ import annotations
import os, sys, logging
from dataclasses import dataclass
import torch
import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField

logging.basicConfig(level=logging.INFO, format="%(message)s")


class TkMode:
    Mode = TilingKeyField(bits=2, values=[0, 1, 2])


@dataclass
class SimpleTiling:
    offset: int
    opkind: int[4]


@pl.jit(auto_mutex=True, tiling_key=TkMode)
def fused_oob_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out2: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    tiling: SimpleTiling,
):
    tt_mat = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
    tt_left = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left)
    tt_right = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right)
    tt_acc = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc)
    tt_vec = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)

    a_l1 = pl.make_tile_group(type=tt_mat, addrs=0x0000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(type=tt_mat, addrs=0x2000, mutex_ids=[1])
    a_l0a = pl.make_tile_group(type=tt_left, addrs=0x0000, mutex_ids=[2])
    b_l0b = pl.make_tile_group(type=tt_right, addrs=0x0000, mutex_ids=[3])
    acc = pl.make_tile_group(type=tt_acc, addrs=0x0000, mutex_ids=[4])
    vec = pl.make_tile_group(type=tt_vec, addrs=0x0000, mutex_ids=[5])

    with pl.section_cube():
        cur_a_l1 = a_l1.current()
        cur_b_l1 = b_l1.current()
        cur_a_l0a = a_l0a.current()
        cur_b_l0b = b_l0b.current()
        cur_acc = acc.current()
        pl.load(cur_a_l1, a, [0, 0])
        pl.load(cur_b_l1, b, [0, 0])
        pl.move(cur_a_l0a, cur_a_l1)
        pl.move(cur_b_l0b, cur_b_l1)
        pl.matmul(cur_acc, cur_a_l0a, cur_b_l0b)
        pl.store(out, cur_acc, [0, 0])
    with pl.section_vector():
        cur_vec = vec.current()
        pl.load_tile(cur_vec, a, [0, 0])
        pl.store_tile(out2, cur_vec, [65535, 0])


dev = f"npu:{os.environ.get('TILE_FWK_DEVICE_ID', '0')}"
torch.npu.set_device(dev)
a = torch.rand(64, 64, device=dev, dtype=torch.float16)
b = torch.rand(64, 64, device=dev, dtype=torch.float16)
out = torch.zeros(64, 64, device=dev, dtype=torch.float32)
out2 = torch.zeros(64, 64, device=dev, dtype=torch.float16)
tiling = SimpleTiling(offset=2, opkind=[1, 0, 0, 0])

# 32 曾被静默钳制到设备预算；显式请求现在必须落在预算内，取当前流的混合块上限。
lim = torch.npu.get_stream_limit(torch.npu.current_stream())
blocks = min(lim["cube_core_num"], lim["vector_core_num"] // 2)

fused_oob_kernel[None, blocks, {"Mode": 2}](a, b, out, out2, tiling)
try:
    torch.npu.synchronize()
    print("ERROR: No AICORE error raised")
    sys.exit(1)
except Exception as e:
    print(f"AICORE error raised: {e}")

print("DONE")
"""


def _assert_dump_and_repro(work_dir: str, *, expect_workspace: bool = False) -> None:
    """Common post-trigger gate: dump artifacts, offline repro, ErrorPC localization."""
    dump_ok, files = _check_dump_files(work_dir)
    assert dump_ok, f"Dump artifacts (data file + _call_kernel.so) not found. Files: {files}"

    repro = _run_repro(work_dir)
    repro_output = repro.stdout + repro.stderr
    assert "AICORE error reproduced" in repro.stdout, \
        f"Repro did not reproduce error:\nstdout: {repro.stdout}\nstderr: {repro.stderr}"
    if expect_workspace:
        assert "workspace" in repro_output.lower(), \
            f"Workspace tensor not found in repro output:\n{repro_output}"
    assert "data-dump" in repro_output, \
        f"Repro should use the .so recompiled into the dump dir:\n{repro_output}"
    assert "ErrorPC" in repro_output, \
        f"ErrorPC not located after reproduction:\n{repro_output}"

    # The PC resolver links the embedded device ELF (relocations applied,
    # .text at 0) and persists it into the dump directory for offline use.
    dump_dir = os.path.join(work_dir, "extra-info", "data-dump", str(ST_DEVICE_ID))
    assert any(f.endswith("_device_linked.so") for f in os.listdir(dump_dir)), \
        f"Linked device ELF not persisted to dump dir:\n{os.listdir(dump_dir)}"


@pytest.mark.soc("950")
def test_fusion_dump_and_repro():
    """cv 融合 kernel 的异常：dump 产物生成且离线复现成功。

    覆盖：融合架构（has_cube + has_vector → dav-c310 + fatobj）下异常
    回调的 -g 重编译链接、tiling 参数被捕获为 WORKSPACE tensor、
    tk_<packed> 子目录 call_kernel.cpp 的重编译，复现工具从 dump 恢复
    全部参数（4 个 DYNAMIC tensor 的 dyn 参数 + tiling）并复现异常，
    以及复现后 ErrorPC -> 源码行的定位。
    """
    _check_npu()
    logging.info("------------test_fusion_dump_and_repro--------------")

    work_dir = _make_work_dir("fusion_dump_and_repro")
    trigger = work_dir / "trigger_fusion_oob.py"
    trigger.write_text(_DUMP_REPRO_TRIGGER)

    result = _run_subprocess(trigger, str(work_dir))
    assert result.returncode == 0, f"Trigger failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert "DONE" in result.stdout, f"trigger script not completed:\n{result.stdout}"

    _assert_dump_and_repro(str(work_dir), expect_workspace=True)


# ---------------------------------------------------------------------------
# 3. 纯 cube kernel 的异常（子进程隔离）
# ---------------------------------------------------------------------------

_CUBE_OOB_TRIGGER = """\
#!/usr/bin/env python3
from __future__ import annotations
import os, sys, logging
import torch
import pypto_pro.language as pl

logging.basicConfig(level=logging.INFO, format="%(message)s")


@pl.jit(auto_mutex=True)
def cube_oob_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    c: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    \"\"\"纯 cube kernel：matmul 后 fixpipe（Acc->GM）越界写出。\"\"\"
    tt_mat = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
    tt_left = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left)
    tt_right = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right)
    tt_acc = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc)
    a_l1 = pl.make_tile_group(type=tt_mat, addrs=0x0000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(type=tt_mat, addrs=0x2000, mutex_ids=[1])
    a_l0a = pl.make_tile_group(type=tt_left, addrs=0x0000, mutex_ids=[2])
    b_l0b = pl.make_tile_group(type=tt_right, addrs=0x0000, mutex_ids=[3])
    acc = pl.make_tile_group(type=tt_acc, addrs=0x0000, mutex_ids=[4])

    with pl.section_cube():
        cur_a_l1 = a_l1.current()
        cur_b_l1 = b_l1.current()
        cur_a_l0a = a_l0a.current()
        cur_b_l0b = b_l0b.current()
        cur_acc = acc.current()
        pl.load(cur_a_l1, a, [0, 0])
        pl.load(cur_b_l1, b, [0, 0])
        pl.move(cur_a_l0a, cur_a_l1)
        pl.move(cur_b_l0b, cur_b_l1)
        pl.matmul(cur_acc, cur_a_l0a, cur_b_l0b)
        pl.store(c, cur_acc, [65535, 0])


dev = f"npu:{os.environ.get('TILE_FWK_DEVICE_ID', '0')}"
torch.npu.set_device(dev)
a = torch.rand(64, 64, device=dev, dtype=torch.float16)
b = torch.rand(64, 64, device=dev, dtype=torch.float16)
c = torch.zeros(64, 64, device=dev, dtype=torch.float32)

cube_oob_kernel(a, b, c)
try:
    torch.npu.synchronize()
    print("ERROR: No AICORE error raised")
    sys.exit(1)
except Exception as e:
    print(f"AICORE error raised: {e}")

print("DONE")
"""


@pytest.mark.soc("950")
def test_cube_dump_and_repro():
    """纯 cube kernel 的 fixpipe 越界异常：上抛 + dump + 离线复现 + 定位。

    覆盖纯 cube 架构（dav-c310-cube）下异常回调的 -g 重编译链接，以及
    fixpipe（Acc->GM）写越界路径——与前两个用例的 MTE2 读（vec load）/
    MTE3 写（vec store）构成不同管线的错误触发，交叉验证定位正确性。
    """
    _check_npu()
    logging.info("------------test_cube_dump_and_repro--------------")

    work_dir = _make_work_dir("cube_dump_and_repro")
    trigger = work_dir / "trigger_cube_oob.py"
    trigger.write_text(_CUBE_OOB_TRIGGER)

    result = _run_subprocess(trigger, str(work_dir))
    assert result.returncode == 0, f"Trigger failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert "DONE" in result.stdout, f"trigger script not completed:\n{result.stdout}"

    _assert_dump_and_repro(str(work_dir))


# ---------------------------------------------------------------------------
# 4. pl.Ptr + scalar 入参的 kernel 异常（子进程隔离）
# ---------------------------------------------------------------------------

_PTR_SCALAR_OOB_TRIGGER = """\
#!/usr/bin/env python3
from __future__ import annotations
import os, sys, logging
import torch
import pypto_pro.language as pl

logging.basicConfig(level=logging.INFO, format="%(message)s")


@pl.jit(auto_mutex=True)
def ptr_oob_store_kernel(
    a_ptr: pl.Ptr[pl.DT_UINT8],
    out_ptr: pl.Ptr[pl.DT_UINT8],
    m_valid: pl.DT_INT64,
    n_valid: pl.DT_INT64,
):
    \"\"\"PTR + scalar 入参：make_tensor 构造后 vec store_tile 越界写。\"\"\"
    io_dtype = pl.DT_FP16
    a = pl.make_tensor(a_ptr, [m_valid, n_valid], [n_valid, 1], dtype=io_dtype)
    out = pl.make_tensor(out_ptr, [m_valid, n_valid], [n_valid, 1], dtype=io_dtype)

    tt_vec = pl.TileType(shape=[64, 64], dtype=io_dtype, target_memory=pl.MemorySpace.Vec)
    vec = pl.make_tile_group(type=tt_vec, addrs=0x0000, mutex_ids=[0])

    with pl.section_vector():
        cur_vec = vec.current()
        pl.load_tile(cur_vec, a, [0, 0])
        pl.store_tile(out, cur_vec, [65535, 0])


dev = f"npu:{os.environ.get('TILE_FWK_DEVICE_ID', '0')}"
torch.npu.set_device(dev)
a = torch.rand(64, 64, device=dev, dtype=torch.float16)
out = torch.zeros(64, 64, device=dev, dtype=torch.float16)

ptr_oob_store_kernel(a, out, 64, 64)
try:
    torch.npu.synchronize()
    print("ERROR: No AICORE error raised")
    sys.exit(1)
except Exception as e:
    print(f"AICORE error raised: {e}")

print("DONE")
"""


@pytest.mark.soc("950")
def test_ptr_scalar_dump_and_repro():
    """pl.Ptr + scalar 入参 kernel 的越界异常：上抛 + dump + 离线复现 + 定位。

    覆盖纯 pl.Ptr（无 shape）与独立 scalar（int64 m/n）入参样式的 ABI
    恢复路径——与 pl.Tensor（shape 随签名走 dyn 参数）不同，这类 kernel
    的形状完全由 scalar 描述，复现工具必须依赖 launch-args sidecar 中的
    scalar 值与 PTR 的 addr/size 才能重建调用，不依赖 dump 文件的 dyn
    推导。触发方式为 vec store_tile 越界写（MTE3 路径）。
    """
    _check_npu()
    logging.info("------------test_ptr_scalar_dump_and_repro--------------")

    work_dir = _make_work_dir("ptr_scalar_dump_and_repro")
    trigger = work_dir / "trigger_ptr_scalar_oob.py"
    trigger.write_text(_PTR_SCALAR_OOB_TRIGGER)

    result = _run_subprocess(trigger, str(work_dir))
    assert result.returncode == 0, f"Trigger failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert "DONE" in result.stdout, f"trigger script not completed:\n{result.stdout}"

    _assert_dump_and_repro(str(work_dir))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pytest.main([__file__, "-v", "-s"])
