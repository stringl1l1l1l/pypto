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
""" """

import csv
import glob
import os
import shutil
import subprocess

import pytest
import torch
import torch_npu

import pypto
from pypto.pypto_impl import NpuArch


def _get_root_dir() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "..", "..", ".."))


def _clean_prof_dirs(prof_base_dir: str) -> None:
    for old_dir in glob.glob(os.path.join(prof_base_dir, "PROF*")):
        shutil.rmtree(old_dir, ignore_errors=True)


def _get_device_id() -> int:
    return int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))


def _run_msprof(
    root_dir: str, script_path: str, env: dict = None, task_time: str = "l3"
) -> subprocess.CompletedProcess:
    cmd = ["msprof"]
    if task_time:
        cmd.append(f"--task-time={task_time}")
    cmd.extend(["python", script_path])
    run_env = os.environ.copy()
    run_env.setdefault("TILE_FWK_DEVICE_ID", str(_get_device_id()))
    if env:
        run_env.update({key: str(value) for key, value in env.items()})
    try:
        result = subprocess.run(
            cmd,
            cwd=root_dir,
            capture_output=True,
            text=True,
            timeout=300,
            env=run_env,
        )
        return result
    except subprocess.TimeoutExpired as exc:
        raise pytest.fail("msprof 命令执行超时") from exc
    except FileNotFoundError as exc:
        raise pytest.fail("msprof 命令未找到，请确保 CANN 环境已正确配置") from exc


def _assert_prof_dirs(root_dir: str, msprof_result: subprocess.CompletedProcess):
    prof_dirs = glob.glob(os.path.join(root_dir, "PROF*"))
    assert len(prof_dirs) > 0, (
        f"未在项目根目录 {root_dir} 下找到 PROF* 文件夹。\n"
        f"msprof returncode={msprof_result.returncode}\n"
        f"stdout:\n{msprof_result.stdout}\n"
        f"stderr:\n{msprof_result.stderr}"
    )
    return prof_dirs


def _get_pmu_event_type() -> int:
    return int(os.environ.get("PYPTO_PROF_PMU_EVENT_TYPE", "2"))


def _get_pmu_arch() -> str:
    return "dav_3510" if pypto.platform.npuarch == NpuArch.DAV_3510.name else "dav_2201"


def _collect_device_data_dirs(prof_dirs):
    data_dirs = []
    for prof_dir in prof_dirs:
        data_dirs.extend(glob.glob(os.path.join(prof_dir, "device_*", "data")))
    return [path for path in data_dirs if os.path.isdir(path)]


def _run_pmu_to_csv(
    root_dir: str,
    data_path: str,
    pmu_event: int,
    arch: str,
    output_dir: str,
) -> subprocess.CompletedProcess:
    script_path = os.path.join(root_dir, "tools", "profiling", "tilefwk_pmu_to_csv.py")
    assert os.path.exists(script_path), f"PMU 解析脚本不存在: {script_path}"
    cmd = [
        "python",
        script_path,
        "-p",
        data_path,
        f"-pe={pmu_event}",
        "--arch",
        arch,
        f"--output={output_dir}",
    ]
    try:
        return subprocess.run(
            cmd,
            cwd=root_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        raise pytest.fail("tilefwk_pmu_to_csv 命令执行超时") from exc


def _row_has_nonzero_pmu(row) -> bool:
    # CSV 列: thread/task/stream/core/seqNo/subtask + total cycle + PMU counters
    # 从 total cycle（下标 6）起校验，不允许全为 0
    pmu_cells = row[6:] if len(row) > 6 else row
    for cell in pmu_cells:
        text = cell.strip()
        if not text:
            continue
        try:
            if float(text) != 0:
                return True
        except ValueError:
            continue
    return False


def _pmu_csv_has_data(csv_path: str) -> bool:
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return False
            return any(
                _row_has_nonzero_pmu(row)
                for row in reader
                if any(cell.strip() for cell in row)
            )
    except Exception:
        return False


def _collect_op_summary_files(prof_dirs):
    op_summary_files_found = []
    for prof_dir in prof_dirs:
        op_summary_pattern = os.path.join(prof_dir, "mindstudio_profiler_output", "op_summary_*.csv")
        op_summary_files_found.extend(glob.glob(op_summary_pattern))
    return op_summary_files_found


def _csv_contains_pypto(csv_file: str) -> bool:
    try:
        with open(csv_file, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return any(
                "PyPTO" in row.get("OP Type", "") for row in reader
            )
    except Exception:
        return False


def _find_pypto_in_csv(op_summary_files):
    return any(_csv_contains_pypto(csv_file) for csv_file in op_summary_files)


def _kernel_details_contains_pypto(csv_file: str) -> bool:
    try:
        with open(csv_file, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return any(
                "PyPTO" in row.get("Type", "") and "PYPTO_add_direct_kernel" in row.get("Name", "") for row in reader
            )
    except Exception:
        return False


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def add_direct_kernel(
    x: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    y: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    z: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    z.move(x + y)


def _build_experimental_config():
    experimental_config_cls = getattr(torch_npu.profiler, "_ExperimentalConfig")
    experimental_config = experimental_config_cls(
        export_type=[torch_npu.profiler.ExportType.Text],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    )
    return experimental_config


def _run_add_direct_profiler(device_id: int, shape: tuple, profiler_output_dir: str) -> None:
    input_data0 = torch.rand(shape, dtype=torch.float, device=f"npu:{device_id}")
    input_data1 = torch.rand(shape, dtype=torch.float, device=f"npu:{device_id}")
    output_data = torch.zeros(shape, dtype=torch.float, device=f"npu:{device_id}")
    experimental_config = _build_experimental_config()

    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        with_stack=False,
        record_shapes=False,
        profile_memory=True,
        experimental_config=experimental_config,
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=5),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiler_output_dir, analyse_flag=True),
    ) as prof:
        for _ in range(10):
            add_direct_kernel(input_data0, input_data1, output_data)
            torch_npu.npu.synchronize()
            prof.step()


def _collect_kernel_detail_files(profiler_output_dir: str):
    return glob.glob(
        os.path.join(profiler_output_dir, "**", "kernel_details.csv"),
        recursive=True,
    )


@pytest.mark.soc("910")
@pypto.options(pass_options={"enable_slice": True})
def test_msprof_profiling_pypto_op_summary():
    """
    看护用例：验证 msprof 性能采集功能
    1. 执行 msprof python examples/03_advanced/advanced_nn/attention/attention.py
    2. 验证 项目根目录/PROF*/mindstudio_profiler_output/op_summary_*.csv 文件生成
    3. 验证 CSV 文件中 OP Type 包含 PyPTO 字样
    """
    root_dir = _get_root_dir()
    _clean_prof_dirs(root_dir)
    add_direct_script = os.path.join(
        root_dir, "examples", "03_advanced", "advanced_nn", "attention", "attention.py"
    )
    assert os.path.exists(add_direct_script), f"脚本不存在: {add_direct_script}"

    msprof_result = _run_msprof(root_dir, add_direct_script, task_time=None)
    prof_dirs = _assert_prof_dirs(root_dir, msprof_result)

    op_summary_files_found = _collect_op_summary_files(prof_dirs)
    pypto_found = _find_pypto_in_csv(op_summary_files_found)
    assert len(op_summary_files_found) > 0, (
        f"未在 PROF* 文件夹中找到 mindstudio_profiler_output/op_summary_*.csv 文件。\n已检查的 PROF 目录: {prof_dirs}"
    )

    assert pypto_found, (
        f"在 op_summary CSV 文件中未找到 OP Type 包含 PyPTO 的记录。\n"
        f"已检查的 CSV 文件: {op_summary_files_found}"
    )

    for prof_dir in prof_dirs:
        shutil.rmtree(prof_dir, ignore_errors=True)


def _parse_and_validate_pmu_data(
    root_dir, data_dirs, pmu_event, arch, pmu_output_dir
):
    pmu_csv_path = os.path.join(pmu_output_dir, "tilefwk_prof_pmu.csv")
    parse_ok = False
    saw_empty_pmu = False
    last_parse_output = ""
    for data_dir in data_dirs:
        parse_result = _run_pmu_to_csv(
            root_dir, data_dir, pmu_event, arch, pmu_output_dir
        )
        last_parse_output = (
            f"stdout:\n{parse_result.stdout}\nstderr:\n{parse_result.stderr}"
        )
        combined_output = f"{parse_result.stdout}\n{parse_result.stderr}"
        if "empty pmu list" in combined_output:
            saw_empty_pmu = True
            continue
        assert parse_result.returncode == 0, (
            f"tilefwk_pmu_to_csv 执行失败, returncode={parse_result.returncode}\n"
            f"data 目录: {data_dir}\n"
            f"{last_parse_output}"
        )
        if os.path.exists(pmu_csv_path) and _pmu_csv_has_data(pmu_csv_path):
            parse_ok = True
            break

    assert parse_ok, (
        "未能采集出正常 PMU 数据：tilefwk_prof_pmu.csv 无有效记录，"
        "或 total cycle/PMU 计数全为 0"
        + ("（解析输出为空 empty pmu list）" if saw_empty_pmu else "")
        + "。\n"
        f"已检查的 data 目录: {data_dirs}\n"
        f"arch={arch}, pe={pmu_event}\n"
        f"期望 CSV: {pmu_csv_path}\n"
        f"最后一次解析输出:\n{last_parse_output}"
    )


@pytest.mark.soc("950")
def test_msprof_pmu_collect_and_parse():
    """
    看护用例：验证 msprof PMU 采集与 tilefwk_pmu_to_csv 解析
    1. 设置 PYPTO_PROF_PMU_EVENT_TYPE 后执行
       msprof --task-time=l3 python examples/03_advanced/advanced_nn/attention/attention.py
       （不指定 --output，PROF* 默认落盘在项目根目录）
    2. 定位 PROF*/device_*/data 产物目录
    3. 执行 tools/profiling/tilefwk_pmu_to_csv.py 解析 PMU 数据
    4. 验证解析结果非 empty pmu list，且 tilefwk_prof_pmu.csv 中
       total cycle / PMU 计数存在非 0 数据
    """
    root_dir = _get_root_dir()
    pmu_event = _get_pmu_event_type()
    arch = _get_pmu_arch()
    pmu_output_dir = os.path.join(root_dir, "pmu_profiling_output")

    _clean_prof_dirs(root_dir)
    shutil.rmtree(pmu_output_dir, ignore_errors=True)
    os.makedirs(pmu_output_dir, exist_ok=True)

    add_direct_script = os.path.join(
        root_dir, "examples", "03_advanced", "advanced_nn", "attention", "attention.py"
    )
    assert os.path.exists(add_direct_script), f"脚本不存在: {add_direct_script}"

    try:
        msprof_result = _run_msprof(
            root_dir,
            add_direct_script,
            env={"PYPTO_PROF_PMU_EVENT_TYPE": pmu_event},
        )
        prof_dirs = _assert_prof_dirs(root_dir, msprof_result)
        data_dirs = _collect_device_data_dirs(prof_dirs)
        assert len(data_dirs) > 0, (
            f"未在 PROF* 下找到 device_*/data 目录。\n"
            f"已检查的 PROF 目录: {prof_dirs}"
        )
        _parse_and_validate_pmu_data(
            root_dir, data_dirs, pmu_event, arch, pmu_output_dir
        )
    finally:
        for prof_dir in glob.glob(os.path.join(root_dir, "PROF*")):
            shutil.rmtree(prof_dir, ignore_errors=True)
        shutil.rmtree(pmu_output_dir, ignore_errors=True)


@pytest.mark.soc("910")
@pypto.options(pass_options={"enable_slice": True})
def test_torch_npu_profiler_collect_pypto_kernel_details():
    """
    看护用例：验证 torch_npu.profiler 能正确采集到 PyPTO 内核信息
    1. 在测试中直接定义并执行 add_direct_kernel
    2. 在 ./add_direct_profiler 下递归查找 kernel_details.csv
    3. 校验 Type 包含 PyPTO 且 Name 包含 PyPYPTO_add_kernel
    4. 清理 add_direct_profiler 文件夹
    """
    root_dir = _get_root_dir()
    profiler_output_dir = os.path.join(root_dir, "add_direct_profiler")
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    shape = (1, 4, 1, 64)

    shutil.rmtree(profiler_output_dir, ignore_errors=True)
    try:
        _run_add_direct_profiler(device_id, shape, profiler_output_dir)
        kernel_detail_files = _collect_kernel_detail_files(profiler_output_dir)
        assert len(kernel_detail_files) > 0, f"未在 {profiler_output_dir} 下递归找到 kernel_details.csv"

        matched = any(_kernel_details_contains_pypto(csv_file) for csv_file in kernel_detail_files)
        assert matched, (
            "在 kernel_details.csv 中未找到 Type 包含 PyPTO 且 "
            "Name 包含 PYPTO_add_direct_kernel 的记录。\n"
            f"已检查文件: {kernel_detail_files}"
        )
    finally:
        shutil.rmtree(profiler_output_dir, ignore_errors=True)
