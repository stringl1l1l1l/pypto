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
r"""
基于 .pyptokb 离线二进制包的单算子复现工具。

自动串联: msaicerr 解析 → info.txt 解析 → bundle 定位 → 生成 test_single_op.py → 执行诊断。

用法示例
--------
python debug_aicore_error.py -p /path/to/report -out /tmp/out

python debug_aicore_error.py -p /path/to/report -out /tmp/out -d 0

python debug_aicore_error.py -p /path/to/report -out /tmp/out -t 1200
"""

import argparse
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

# ===================================================================
# 模块级全局状态（main() 中赋值，函数间共享）
# ===================================================================

_PYTO_ROOT: Optional[str] = None

_DEBUG_LOG_PATH: Optional[str] = None  # _init_debug_log() 后指向 msaicerr_out 下的 debug_info.txt
_BUNDLED_KERNEL_PATH: Optional[str] = None  # _find_and_record_bundled_kernels() 后指向 .pyptokb 路径
_BUNDLED_KERNEL_PATH_UNDEF: Optional[str] = None  # _find_and_record_bundled_kernels() 后指向 *_nosubfunc.pyptokb 路径
_EARLY_LOGS: List[str] = []             # _init_debug_log() 之前暂存日志


# ===================================================================
# 日志：_print_log 是几乎所有函数都依赖的公共出口
# ===================================================================

def _print_log(level: str, msg: str) -> None:
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))
    pid = os.getpid()
    line = current_time + " (" + str(pid) + ") - [" + level + "] " + msg
    print(line)
    sys.stdout.flush()
    if _DEBUG_LOG_PATH is not None:
        try:
            with open(_DEBUG_LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(line + '\n')
        except Exception:
            pass
    else:
        _EARLY_LOGS.append(line)


def _init_debug_log(msaicerr_out: str) -> None:
    """找到 -out 目录下的 debug_info.txt，后续日志直接追加到该文件。"""
    global _DEBUG_LOG_PATH
    target = os.path.join(msaicerr_out, "debug_info.txt")
    _DEBUG_LOG_PATH = os.path.abspath(target)
    # 将 init 前的早期日志刷入
    for line in _EARLY_LOGS:
        try:
            with open(_DEBUG_LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(line + '\n')
        except Exception:
            pass
    _EARLY_LOGS.clear()


# ===================================================================
# 常量
# ===================================================================

# ---- info.txt section header 模式与段标题 ----
_INFO_SECTION_PATTERN = re.compile(
    r'^\*{3,}\d+\.\s+.+?\*{3,}$', re.MULTILINE
)
_TARGET_SECTION_KEYWORD = "6. Execution Result of the Single-Operator Test Case"
_SECTION7_TITLE = "7. Inter-Core Synchronization Diagnosis"
_SECTION8_TITLE = "8. Framework vs Operator (CCE) Root Cause Analysis"

# ---- Bundle SO ----
_BUNDLE_SO_NAMES = ("libtile_fwk_bundle.so",)

# ---- msnpureport 默认路径 ----
_MSNPUREPORT_DEFAULT = "/usr/local/Ascend/driver/tools/msnpureport"

# ---- dtype → numpy / torch dtype 映射 ----
_STR_TO_NP: Dict[str, str] = {
    "float16":  "np.float16",
    "float32":  "np.float32",
    "float64":  "np.float64",
    "int8":     "np.int8",
    "int16":    "np.int16",
    "int32":    "np.int32",
    "int64":    "np.int64",
    "uint8":    "np.uint8",
    "uint16":   "np.uint16",
    "uint32":   "np.uint32",
    "uint64":   "np.uint64",
    "double":   "np.float64",
    "complex64":  "np.complex64",
    "complex128": "np.complex128",
    "bfloat16": "np.int16",   # numpy 不支持 bfloat16，用 int16 先读
    "bool":     "np.bool_",
    # fp8: numpy 不支持，用 uint8 读原始字节后 view 为对应 torch 类型
    "float8_e4m3fn": "np.uint8",
    "fp8e4m3":       "np.uint8",
    "float8_e5m2":   "np.uint8",
    "fp8e5m2":       "np.uint8",
    "fp8":           "np.uint8",
    "float8":        "np.uint8",
    "float8_e8m0":   "np.uint8",  # PyTorch 无原生 float8_e8m0，按 raw bytes 读取
}

_STR_TO_TORCH: Dict[str, str] = {
    "float16":  "torch.float16",
    "float32":  "torch.float32",
    "float64":  "torch.float64",
    "int8":     "torch.int8",
    "int16":    "torch.int16",
    "int32":    "torch.int32",
    "int64":    "torch.int64",
    "uint8":    "torch.uint8",
    "uint16":   "torch.uint16",
    "uint32":   "torch.uint32",
    "uint64":   "torch.uint64",
    "double":   "torch.float64",
    "complex64":  "torch.complex64",
    "complex128": "torch.complex128",
    "bfloat16": "torch.bfloat16",
    "bool":     "torch.bool",
    "float8_e4m3fn": "torch.float8_e4m3fn",
    "fp8e4m3":       "torch.float8_e4m3fn",
    "float8_e5m2":   "torch.float8_e5m2",
    "fp8e5m2":       "torch.float8_e5m2",
    "fp8":           "torch.float8_e4m3fn",
    "float8":        "torch.float8_e4m3fn",
    "float8_e8m0":   "torch.uint8",  # PyTorch 无原生 float8_e8m0，按 raw bytes 持有
}

# ---- DataType enum (tilefwk/data_type.h: DATA_TYPE_ALL 按定义顺序) ----
# DT_INT4=0, INT8=1, INT16=2, INT32=3, INT64=4, FP8=5, FP16=6, FP32=7, BF16=8,
# HF4=9, HF8=10, UINT8=11, UINT16=12, UINT32=13, UINT64=14, BOOL=15, DOUBLE=16,
# FP8E4M3=17, FP8E5M2=18, FP8E8M0=19, FP4_E2M1X2=20, FP4_E1M2X2=21, FP4_E2M1=22, FP4_E1M2=23
# 注：complex64/complex128 在 PyPTO DataType 中无对应枚举，走 float32 fallback
_STR_TO_DT_ENUM: Dict[str, int] = {
    "int4":     0,
    "int8":     1,
    "int16":    2,
    "int32":    3,
    "int64":    4,
    "fp8":      5,
    "float8":   5,
    "float16":  6,
    "float32":  7,
    "bfloat16": 8,
    "hifloat4": 9,
    "hf4":      9,
    "hifloat8": 10,
    "hf8":      10,
    "uint8":    11,
    "uint16":   12,
    "uint32":   13,
    "uint64":   14,
    "bool":     15,
    "double":   16,
    "float64":  16,
    "float8_e4m3fn": 17,
    "fp8e4m3":       17,
    "float8_e5m2":   18,
    "fp8e5m2":       18,
    "float8_e8m0":   19,
    "float4_e2m1":   22,
    "float4_e1m2":   23,
}


# 需要做 view 的 dtype（numpy 无原生支持）：bfloat16 用 int16，fp8 系列用 uint8
_NEEDS_VIEW_DTYPES = frozenset({
    "bfloat16",
    "float8_e4m3fn", "fp8e4m3", "float8_e5m2", "fp8e5m2", "fp8", "float8",
})


# ===================================================================
# 小工具
# ===================================================================

def _is_under(path: str, ancestor: str) -> bool:
    """判断 path 是否等于 ancestor 或位于 ancestor 目录之下（避免 /a 与 /ab 误判）。"""
    return path == ancestor or path.startswith(ancestor + os.sep)


def _is_docker_env():
    """判断当前是否为 docker 环境（通过 /.dockerenv 或 /proc/1/cgroup 检测）。"""
    if os.path.exists("/.dockerenv"):
        return True
    try:
        with open("/proc/1/cgroup", "r") as f:
            content = f.read()
            if "docker" in content or "kubepods" in content:
                return True
    except (IOError, PermissionError):
        pass
    return False


# ===================================================================
# Phase A: 环境检查 / 输入校验 / Python 检测
# ===================================================================

def get_ascend_home() -> str:
    """通过环境变量 ASCEND_HOME_PATH 获取 CANN 包目录。"""
    ascend_home = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_home:
        raise RuntimeError(
            "ASCEND_HOME_PATH env variable not set, please source set_env.sh and retry"
        )
    if not os.path.isdir(ascend_home):
        raise RuntimeError(f"ASCEND_HOME_PATH directory does not exist: {ascend_home}")
    return ascend_home


def _get_ascend_env() -> str:
    """获取并校验 CANN 环境变量：返回 ASCEND_HOME_PATH，并要求 ASCEND_OPP_PATH 已设置。"""
    ascend_home = get_ascend_home()
    _print_log("INFO", f"ASCEND_HOME_PATH = {ascend_home}")
    if not os.environ.get("ASCEND_OPP_PATH"):
        raise RuntimeError("ASCEND_OPP_PATH env variable not set, please source set_env.sh and retry")
    return ascend_home


def _get_app_plog_dir(work_dir: str) -> str:
    """返回工作目录下应用层 PYPTO plog 目录路径。"""
    return os.path.join(work_dir, "log", "debug", "plog")


def _validate_work_dir(work_dir: str) -> None:
    """检查 -p 目录结构，缺失必要子目录时报错中断。"""
    required_dirs = {
        "log/debug/plog": _get_app_plog_dir(work_dir),
        "extra-info/data-dump": os.path.join(work_dir, "extra-info", "data-dump"),
        "pypto": os.path.join(work_dir, "pypto"),
    }
    missing = []
    for label, path in required_dirs.items():
        if not os.path.isdir(path):
            missing.append(f"  {label}: {path}")
    if missing:
        raise RuntimeError(
            "Work directory missing these subdirectories:\n" + "\n".join(missing)
        )


def _validate_paths_not_in_report_dir(report_dir: str, out_dir: str) -> None:
    """新版 msaicerr 前置约束：-out 与 cwd 不得位于 -p 目录之下，且 cwd 可写。"""
    report_abs = os.path.abspath(report_dir)
    out_abs = os.path.abspath(out_dir)
    if _is_under(out_abs, report_abs):
        raise RuntimeError(
            f"-out directory must not be inside -p directory: {out_abs} is under {report_abs}")
    cwd_abs = os.getcwd()
    if _is_under(cwd_abs, report_abs):
        raise RuntimeError(
            f"current working directory must not be inside -p directory: {cwd_abs} is under {report_abs}")
    if not os.access(cwd_abs, os.W_OK):
        raise RuntimeError(f"current working directory is not writable: {cwd_abs}")


def _get_device_id(cli_device_id: Optional[str] = None) -> int:
    """获取 device id。优先级: -d 参数 > TILE_FWK_DEVICE_ID 环境变量。"""
    if cli_device_id is not None:
        try:
            return int(cli_device_id)
        except ValueError:
            raise RuntimeError(f"-d argument value invalid: {cli_device_id}")

    val = os.environ.get("TILE_FWK_DEVICE_ID", "")
    if not val:
        raise RuntimeError("-d argument not specified and TILE_FWK_DEVICE_ID env variable not set, "
                           "please specify -d or set the env variable and retry")
    try:
        return int(val)
    except ValueError:
        raise RuntimeError(f"TILE_FWK_DEVICE_ID value invalid: {val}")


def _check_log_level() -> None:
    """检查 ASCEND_GLOBAL_LOG_LEVEL，非 ERROR 级别时提示可能较慢。"""
    level = os.environ.get("ASCEND_GLOBAL_LOG_LEVEL", "3")
    if level != "3":
        _print_log("WARNING",
            f"ASCEND_GLOBAL_LOG_LEVEL({level}) not set to ERROR(3), single-operator test be slow at current log level, "
            "recommend export ASCEND_GLOBAL_LOG_LEVEL=3")


def _detect_python() -> str:
    """检测可用的 Python 解释器，确保能 import pypto 和 import torch_npu。"""
    candidates = [sys.executable]
    if os.path.basename(sys.executable) != "python3":
        candidates.append("python3")
    candidates.append("python")
    # 去重，避免同一解释器重复探测
    candidates = list(dict.fromkeys([c for c in candidates if c]))

    for exe in candidates:
        if not exe:
            continue
        try:
            result = subprocess.run(
                [exe, "-c", "import pypto; import torch_npu"],
                capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired:
            _print_log("WARNING", f"Phase A: {exe} import check timed out (30s), trying next interpreter")
            continue
        except Exception:
            continue
        if result.returncode == 0:
            _print_log("INFO", f"Using Python: {exe}")
            return exe

    raise RuntimeError("No usable Python interpreter found (must be able to import pypto and torch_npu), "
                       "check the runtime environment")


# ===================================================================
# Phase B: 调用 msaicerr.py
# ===================================================================

def run_msaicerr(report_path: str, output_path: str, device_id: int,
                 ascend_home: str) -> str:
    """
    调用 CANN 包下的 msaicerr.py 解析 AIC Error 报告。

    返回 msaicerr 输出目录（即 info_<timestamp> 目录）。
    """
    msaicerr_script = os.path.join(ascend_home, "tools", "msaicerr", "msaicerr.py")
    if not os.path.isfile(msaicerr_script):
        raise RuntimeError(f"msaicerr.py not found: {msaicerr_script}")

    # 先列出将要输出的目录，用于事后定位 info_<timestamp>
    before_items = set()
    if os.path.isdir(output_path):
        for item in os.listdir(output_path):
            before_items.add(item)

    cmd = [
        sys.executable,
        msaicerr_script,
        "-p", report_path,
        "-out", output_path,
        "-dev", str(device_id),
    ]
    _print_log("INFO", f"Executing: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        raise RuntimeError("Phase B: msaicerr.py parsing timed out (600s), check log size or run manually")

    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)

    if result.returncode != 0:
        _print_log("WARNING", f"msaicerr.py return code: {result.returncode} (continuing to locate info.txt)")

    # 定位 info_<timestamp> 目录
    after_items = set()
    if os.path.isdir(output_path):
        for item in os.listdir(output_path):
            after_items.add(item)

    new_items = after_items - before_items
    for item in new_items:
        full = os.path.join(output_path, item)
        if os.path.isdir(full) and item.startswith("info_"):
            _print_log("INFO", f"msaicerr output directory: {full}")
            return full

    # 备选：在 output_path 下找最新创建的 info_* 目录
    best = None
    best_mtime = 0
    for item in os.listdir(output_path):
        full = os.path.join(output_path, item)
        if os.path.isdir(full) and item.startswith("info_"):
            mtime = os.path.getmtime(full)
            if mtime > best_mtime:
                best_mtime = mtime
                best = full
    if best:
        _print_log("INFO", f"msaicerr output directory (mtime): {best}")
        return best

    raise RuntimeError(
        f"Cannot find info_<timestamp> directory from msaicerr output in {output_path}"
    )


def find_info_txt(msaicerr_out_dir: str) -> str:
    """在 msaicerr 输出目录下查找 info.txt。"""
    for root, dirs, files in os.walk(msaicerr_out_dir):
        for fname in files:
            if fname == "info.txt":
                return os.path.join(root, fname)
    raise RuntimeError(f"Cannot find info.txt in {msaicerr_out_dir}")


def find_debug_info_txt_path(msaicerr_out: str) -> str:
    """返回 msaicerr 输出目录下 debug_info.txt 的路径（不检查是否存在）。"""
    return os.path.join(msaicerr_out, "debug_info.txt")


# ===================================================================
# Phase C-1: 解析 info.txt → (preamble, sections, header_order) + kernel/tensor
# ===================================================================

class TensorInfo:
    """单个 dump tensor 的元信息。"""
    __slots__ = ("path", "shape", "dtype", "io_type", "index")

    def __init__(self, path: str, shape: tuple, dtype: str, io_type: str, index: int):
        self.path = path
        self.shape = shape
        self.dtype = dtype      # 原始 dtype 字符串，如 "float16"
        self.io_type = io_type  # "input" / "output" / "workspace"
        self.index = index


def _parse_info_txt_to_sections(info_txt_path: str):
    """
    将 info.txt 解析为 (preamble, sections_dict, header_order)。

    preamble:    第一个 section header 之前的内容（根因结论 + 空行）
    sections:    {header_line: content}  例如
                 {"********************6. ... ***********************": "执行结果内容"}
    header_order: 保持原始顺序的 header 列表
    """
    with open(info_txt_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    matches = list(_INFO_SECTION_PATTERN.finditer(content))
    if not matches:
        return content, {}, []

    preamble = content[:matches[0].start()]
    sections: Dict[str, str] = {}
    header_order: List[str] = []

    for i, m in enumerate(matches):
        header = m.group().strip()
        header_order.append(header)
        start = m.end() + 1  # +1 跳过 header 后的换行符
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        sections[header] = content[start:end].rstrip()

    return preamble, sections, header_order


def _find_section_key(sections: Dict[str, str], keyword: str) -> Optional[str]:
    """在 sections 中查找匹配 keyword 的 header key。"""
    for header in sections:
        if keyword in header:
            return header
    return None


def _find_section6_key(sections: Dict[str, str]) -> Optional[str]:
    """在 sections 中查找第 6 段 Single-Operator Test Case 的 header key。"""
    return _find_section_key(sections, _TARGET_SECTION_KEYWORD)


def _extract_from_sections(sections: Dict[str, str]) -> Tuple[str, List[TensorInfo]]:
    """从 sections dict 中提取 kernel 名和 tensor 列表。"""
    # --- 提取 kernel name (section 1) ---
    sec1_key = _find_section_key(sections, "1. Basic information")
    if sec1_key is None:
        raise RuntimeError("section 1 (Basic information) not found in info.txt")
    sec1_content = sections[sec1_key]

    kernel_name_match = re.search(r"kernel name\s+:\s*(.+)", sec1_content)
    if not kernel_name_match:
        raise RuntimeError("'kernel name' field not found in section 1")
    kernel_name = kernel_name_match.group(1).strip()

    # 非 PyPTO_ 前缀的 kernel 不在本工具支持范围内，直接退出
    if not kernel_name.startswith("PyPTO_"):
        _print_log("ERROR",
            f"kernel name does not start with 'PyPTO_' prefix: {kernel_name}, "
            "this tool only supports PyPTO kernels, please check the report directory")
        sys.exit(1)

    # 从 "PyPTO_xxx_0_mix_aic" 提取 PyPTO_ 和 _0_mix_aic 之间的部分
    func_match = re.match(r"PyPTO_(.+?)_\d+_mix_aic", kernel_name)
    if not func_match:
        _print_log("ERROR",
            f"kernel name does not match expected pattern 'PyPTO_<func>_<n>_mix_aic': {kernel_name}")
        sys.exit(1)
    kernel_func_name = func_match.group(1)

    _print_log("INFO", f"kernel name = {kernel_name}")
    _print_log("INFO", f"Inferred pypto function name = {kernel_func_name}")

    # --- 提取 section 5: Operator Dump File Parsing ---
    sec5_key = _find_section_key(sections, "5. Operator Dump File Parsing")
    if sec5_key is None:
        raise RuntimeError("section 5 (Operator Dump File Parsing) not found in info.txt")
    sec5_content = sections[sec5_key]

    # dtype 后可能带可选的 " user tag: xxx"（cann-9.2.0 msaicerr），用 [^\n]* 容忍行尾附加字段
    tensor_pattern = re.compile(
        r"shape:\s*\(([^)]*)\)\s+size:\s*\d+\s+dtype:\s*(\S+)[^\n]*\n"
        r"(.+?)\n",
    )
    raw_matches = list(tensor_pattern.finditer(sec5_content))

    tensors: List[TensorInfo] = []
    for i, m in enumerate(raw_matches):
        shape_str = m.group(1)
        dtype_str = m.group(2)
        file_path = m.group(3).strip()

        if shape_str.strip():
            shape = tuple(int(x.strip()) for x in shape_str.split(",") if x.strip())
        else:
            shape = ()

        basename = os.path.basename(file_path)
        parts = basename.split(".")
        io_type = "unknown"
        index = 0
        if len(parts) >= 5:
            io_field = parts[-4]
            if io_field.lower() in ("input", "output", "workspace"):
                io_type = io_field.lower()
            try:
                index = int(parts[-3])
            except ValueError:
                pass

        tensors.append(TensorInfo(
            path=file_path, shape=shape, dtype=dtype_str,
            io_type=io_type, index=index,
        ))
        _print_log("DEBUG", f"  tensor[{i}] {io_type}[{index}] shape={shape} dtype={dtype_str} "
                   f"file={basename}")

    order = {"input": 0, "output": 1, "workspace": 2}
    tensors.sort(key=lambda t: (order.get(t.io_type, 9), t.index))

    _print_log("INFO", f"Parsed {len(tensors)} tensors "
               f"({sum(1 for t in tensors if t.io_type == 'input')} input, "
               f"{sum(1 for t in tensors if t.io_type == 'output')} output, "
               f"{sum(1 for t in tensors if t.io_type == 'workspace')} workspace)")

    return kernel_func_name, tensors


# ===================================================================
# info.txt 回写 & section upsert
# ===================================================================

def _rewrite_info_txt(info_txt_path: str, preamble: str,
                      sections: Dict[str, str], header_order: List[str]):
    """用更新后的 sections 重写 info.txt。"""
    lines = [preamble.rstrip()]
    for header in header_order:
        lines.append("")
        lines.append(header)
        lines.append(sections.get(header, ""))
    lines.append("")  # 文件末尾换行
    with open(info_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        _print_log("INFO", f"Updated {info_txt_path}")


def _upsert_section_into_dict(sections: Dict[str, str], header_order: List[str],
                              section_title: str, content: str):
    """将 content 写入 sections dict 中匹配 section_title 的段；不存在则追加。"""
    for header in list(sections.keys()):
        if section_title in header:
            sections[header] = content
            return
    # 不存在则新建
    new_header = f"********************{section_title}***********************"
    new_key = new_header.strip()
    header_order.append(new_key)
    sections[new_key] = content


def _upsert_section7(sections: Dict[str, str], header_order: List[str], content: str):
    """将内容写入 sections dict 的 section 7 段。"""
    _upsert_section_into_dict(sections, header_order, _SECTION7_TITLE, content)


def _upsert_section8(sections: Dict[str, str], header_order: List[str], content: str):
    """将内容写入 sections dict 的 section 8 段。"""
    _upsert_section_into_dict(sections, header_order, _SECTION8_TITLE, content)


# ===================================================================
# Phase C-2: Bundle (.pyptokb) 定位
# ===================================================================

def find_bundle_so() -> Optional[str]:
    """自动查找 libtile_fwk_bundle.so。"""
    # 1. 环境变量
    env_so = os.environ.get("PYPTO_BUNDLE_SO", "")
    if env_so and os.path.isfile(env_so):
        return env_so

    # 2. 从 _PYTO_ROOT 推断
    if _PYTO_ROOT:
        so_dir = os.path.join(_PYTO_ROOT, "lib")
        for name in _BUNDLE_SO_NAMES:
            cand = os.path.join(so_dir, name)
            if os.path.isfile(cand):
                return cand

    # 3. LD_LIBRARY_PATH 搜索
    for d in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if not d:
            continue
        for name in _BUNDLE_SO_NAMES:
            cand = os.path.join(d, name)
            if os.path.isfile(cand):
                return cand

    return None


def _find_bundled_kernel(report_dir: str) -> Optional[str]:
    """从 -p 目录下 find -name 'PyPTO*0_mix_aic.pyptokb' 获取 .pyptokb 路径（排除 _nosubfunc 后缀）。

    多卡场景下可能命中多个 output_* 目录中的同名产物，统一取时间最早目录中的那份。
    """
    try:
        result = subprocess.run(
            ["find", report_dir, "-name", "PyPTO*0_mix_aic.pyptokb",
             "!", "-name", "*_nosubfunc.pyptokb"],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        _print_log("WARNING", "Phase C: find bundled kernel timed out (30s)")
        return None
    except Exception as e:
        _print_log("WARNING", f"find bundled kernel failed: {e}")
        return None

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        _print_log("WARNING", "PyPTO*0_mix_aic.pyptokb not found under -p directory")
        return None
    if len(lines) > 1:
        path = _select_bundled_kernel_by_earliest_output_dir(lines)
    else:
        path = lines[0]
        _print_log("INFO", f"Found bundled kernel: {path}")
    return path


def _find_undef_bundled_kernel(report_dir: str) -> Optional[str]:
    """从 -p 目录下 find -name 'PyPTO*0_mix_aic_nosubfunc.pyptokb' 获取 *_nosubfunc.pyptokb 路径。

    多卡场景下可能命中多个 output_* 目录中的同名产物，统一取时间最早目录中的那份。
    """
    try:
        result = subprocess.run(
            ["find", report_dir, "-name", "PyPTO*0_mix_aic_nosubfunc.pyptokb"],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        _print_log("WARNING", "Phase C: find undef bundled kernel timed out (30s)")
        return None
    except Exception as e:
        _print_log("WARNING", f"find undef bundled kernel failed: {e}")
        return None

    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    if not lines:
        _print_log("WARNING", "PyPTO*0_mix_aic_nosubfunc.pyptokb not found under -p directory")
        return None
    if len(lines) > 1:
        path = _select_bundled_kernel_by_earliest_output_dir(lines)
    else:
        path = lines[0]
        _print_log("INFO", f"Found undef bundled kernel: {path}")
    return path


def _select_bundled_kernel_by_earliest_output_dir(candidates: List[str]) -> str:
    """
    多卡/多进程场景下，-p 目录可能同时存在多份 pypto/output_<时间戳>_<host>_<pid>/ 编译产物。

    每一份产物位于独立的 output_* 目录下，代表一个子进程（一张卡）的 kernel 编译输出；
    多份产物的同名 .pyptokb 内容一致（同一 kernel 代码），异常复现应选用时间最早的
    output_* 目录中的那份，与 msaicerr 解析的"最早一次报错"保持一致。

    规则：
      - 仅 1 个候选：直接返回；
      - 多个候选分属多个 output_* 目录：取目录名（含定长时间戳 YYYYMMDD_HHMMSS_fff）
        字典序最小者，即时间最早的那份；
      - 同一 output_* 目录下出现多个不同 kernel 名：属"单进程多 kernel 报错"场景，
        无法确定复现目标，报错提示用户介入。
    """
    by_dir: Dict[str, List[str]] = {}
    for p in candidates:
        parent = os.path.dirname(os.path.abspath(p))
        by_dir.setdefault(parent, []).append(p)

    ordered_dirs = sorted(by_dir, key=lambda d: os.path.basename(d))
    if len(ordered_dirs) > 1:
        _print_log("INFO", "Multiple kernel compile output directories found in -p directory: "
                   + ", ".join(os.path.basename(d) for d in ordered_dirs))

    earliest_dir = ordered_dirs[0]
    earliest_group = by_dir[earliest_dir]
    print(earliest_group)
    if len(earliest_group) > 1:
        raise RuntimeError(
            f"Found {len(earliest_group)} kernels in the earliest output directory "
            f"{earliest_dir}, cannot determine the reproduction target:\n"
            + "\n".join(f"  {p}" for p in earliest_group)
        )

    path = earliest_group[0]
    _print_log("INFO", f"{len(candidates)} candidates found, "
               f"use the one in the earliest output directory: {path}")
    return path


def _find_and_record_bundled_kernels(sections: Dict[str, str], work_dir: str = ""):
    """
    从 -p 目录发现 .pyptokb，并把 bundle 路径补充到 info.txt 的 section 1。

    coreType / fixedPC / 符号定位已由 msaicerr (cann-9.2.0) 直接产出在 section 3
    （Corrected Info），本脚本不再从 plog 重新解析。
    """
    global _BUNDLED_KERNEL_PATH, _BUNDLED_KERNEL_PATH_UNDEF
    report_dir = work_dir if work_dir and os.path.isdir(work_dir) else ""
    _BUNDLED_KERNEL_PATH = _find_bundled_kernel(report_dir) if report_dir else None
    _BUNDLED_KERNEL_PATH_UNDEF = _find_undef_bundled_kernel(report_dir) if report_dir else None

    if _BUNDLED_KERNEL_PATH is None and _BUNDLED_KERNEL_PATH_UNDEF is None:
        return


# ===================================================================
# Phase D: Bundle 脚本生成与执行（含 Section 6 单算子测试）
# ===================================================================

def _bundle_tensor_load_code(tensors: List[TensorInfo]) -> Tuple[List[str], List[str]]:
    """
    生成 bundle 模式的 tensor 加载 + PyptoTensorDesc 构造代码。

    与 msaicerr DumpDataParser._build_typed_array 同构：

      1) 一律 np.fromfile(..., dtype=np.int8) 字节读取
         （对应 msaicerr 的 np.frombuffer(raw_data, dtype=np.int8)）
      2) numpy 原生支持的 dtype → np.view(np_dtype) + np.reshape 零拷贝重解释
      3) numpy 不支持的 dtype（bf16 / fp8）→ 保留 int8，由 torch.view + torch.reshape 重解释

    返回 (load_and_desc_lines, tensor_desc_var_names)
    每个 tensor_desc_var 是 PyptoTensorDesc 的变量名。
    """
    lines: List[str] = []
    desc_vars: List[str] = []

    for t in tensors:
        var_name = f"t_{t.io_type}_{t.index}"
        desc_var = f"d_{t.io_type}_{t.index}"
        dt_enum = _STR_TO_DT_ENUM.get(t.dtype, _STR_TO_DT_ENUM["float32"])

        if t.path.endswith(".npy"):
            lines.append(f"{var_name}_np = np.load(r'{t.path}')")
            lines.append(f"{var_name} = torch.tensor({var_name}_np, device=device)")
        elif t.path.endswith(".bin"):
            # 与 msaicerr _build_typed_array 同构：先 int8 字节读取
            shape_repr = repr(t.shape)
            lines.append(f"{var_name}_bytes = np.fromfile(r'{t.path}', dtype=np.int8)")

            if t.dtype in _NEEDS_VIEW_DTYPES:
                # numpy 不原生支持 → 保留 int8，由 torch.view + torch.reshape 重解释
                torch_dtype = _STR_TO_TORCH.get(t.dtype, _STR_TO_TORCH["float16"])
                lines.append(f"# {t.dtype}: numpy 不原生支持，int8 字节流由 torch.view({torch_dtype}) 零拷贝重解释")
                lines.append(
                    f"{var_name} = torch.tensor({var_name}_bytes, device=device)"
                    f".view({torch_dtype}).reshape({shape_repr})"
                )
            else:
                # numpy 原生支持 → np.view + np.reshape 零拷贝重解释
                np_dtype = _STR_TO_NP.get(t.dtype, _STR_TO_NP["float16"])
                lines.append(f"# {t.dtype}: numpy 原生支持，int8 字节流 .view({np_dtype}) 零拷贝重解释")
                lines.append(f"{var_name}_np = {var_name}_bytes.view({np_dtype}).reshape({shape_repr})")
                lines.append(f"{var_name} = torch.tensor({var_name}_np, device=device)")

        # 构造 PyptoTensorDesc
        lines.append(f"{desc_var} = PyptoTensorDesc()")
        lines.append(f"{desc_var}.addr = ctypes.c_void_p({var_name}.data_ptr())")
        lines.append(f"{desc_var}.dataType = {dt_enum}  # {t.dtype}")
        lines.append(f"{desc_var}.rank = {len(t.shape)}")
        for i, s in enumerate(t.shape):
            lines.append(f"{desc_var}.shape[{i}] = {s}")
        lines.append("")

        desc_vars.append(desc_var)

    return lines, desc_vars


def codegen_bundle_script(
    bundle_path: str,
    kernel_func_name: str,
    tensors: List[TensorInfo],
    device_id: int,
    output_path: str,
    bundle_so: str,
) -> None:
    """生成基于 .pyptokb 离线二进制的单算子复现脚本。"""

    load_lines, desc_vars = _bundle_tensor_load_code(tensors)

    lines = [
        "#!/usr/bin/env python3",
        "# coding: utf-8",
        "# Auto-generated by pypto_aicerr_repro.py (bundle mode)",
        f"# Kernel: {kernel_func_name}",
        f"# Bundled kernel:  {bundle_path}",
        f"# Device:  {device_id}",
        "#",
        "",
        "import ctypes",
        "import os",
        "import sys",
        "import numpy as np",
        "import torch",
        "import torch_npu  # noqa: F401",
        "",
        "# ---- PyptoTensorDesc (kernel_bundle_format.h / pypto_bundle_api.h) ----",
        "",
        "class PyptoTensorDesc(ctypes.Structure):",
        '    _fields_ = [',
        '        ("addr",     ctypes.c_void_p),',
        '        ("dataType", ctypes.c_int32),',
        '        ("rank",     ctypes.c_int32),',
        '        ("shape",    ctypes.c_int64 * 8),',
        "    ]",
        "",
        "# ---- Load libtile_fwk_bundle.so ----",
        "",
        f'_BUNDLE_SO = os.environ.get("PYPTO_BUNDLE_SO", r"{bundle_so}")',
        "# Production builds skip RPATH, so dlopen-by-abspath does not search the sibling dir.",
        "# Preload DT_NEEDED of the plain bundle .so first (same order as python/pypto/_loader.py).",
        "_BUNDLE_DIR = os.path.dirname(os.path.abspath(_BUNDLE_SO))",
        "for _dep in (",
        '    "libc_sec.so",',
        '    "libtile_fwk_utils.so",',
        '    "libtile_fwk_adapter.so",',
        '    "libtile_fwk_cann_host_runtime.so",',
        '    "libtile_fwk_platform.so",',
        '    "libtile_fwk_interface.so",',
        '    "libtile_fwk_codegen.so",',
        '    "libtile_fwk_compiler.so",',
        '    "libtile_fwk_runtime.so",',
        "):",
        "    _p = os.path.join(_BUNDLE_DIR, _dep)",
        "    if os.path.isfile(_p):",
        "        ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)",
        "_BUNDLE_LIB = ctypes.CDLL(_BUNDLE_SO, mode=ctypes.RTLD_GLOBAL)",
        "",
        "_desc_p = ctypes.POINTER(PyptoTensorDesc)",
        "_BUNDLE_LIB.PyptoWorkspace.restype = ctypes.c_uint64",
        "_BUNDLE_LIB.PyptoWorkspace.argtypes = [ctypes.c_char_p, _desc_p, ctypes.c_uint32]",
        "_BUNDLE_LIB.PyptoLaunch.restype = ctypes.c_int",
        "_BUNDLE_LIB.PyptoLaunch.argtypes = [",
        "    ctypes.c_char_p, _desc_p, ctypes.c_uint32,",
        "    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,",
        "]",
        "",
        "# ---- Device ----",
        f"device = torch.device(f'npu:{device_id}')",
        "torch.npu.set_device(device)",
        "",
        "# ---- 加载 dump tensor ----",
        "",
    ]
    lines.extend(load_lines)

    lines += [
        "# ---- 构造 PyptoTensorDesc 数组 ----",
        "",
        f"_descs = (PyptoTensorDesc * {len(desc_vars)})({', '.join(desc_vars)})",
        "",
        "# ---- 调用 bundle ----",
        "",
        f'_bundle_path = os.environ.get("PYPTO_BUNDLE_PATH", r"{bundle_path}").encode()',
        "",
        f"print('kernel: {kernel_func_name}  (bundle mode)')",
        "",
        "ws_size = _BUNDLE_LIB.PyptoWorkspace(_bundle_path, _descs, len(_descs))",
        "",
        "_ws_keep = None",
        "_ws_ptr = None",
        "if ws_size > 0:",
        "    _ws_keep = torch.empty(ws_size, dtype=torch.uint8, device=device)",
        "    _ws_ptr = ctypes.c_void_p(_ws_keep.data_ptr())",
        "",
        "rc = _BUNDLE_LIB.PyptoLaunch(_bundle_path, _descs, len(_descs), _ws_ptr, None, 1)",
        "if rc != 0:",
        '    print(f"PyptoLaunch failed, rc={rc}.")',
        "    sys.exit(1)",
        "torch.npu.synchronize()",
        'print("Bundle execution completed.")',
        'print("sync done.")',
        "",
    ]

    content = "\n".join(lines) + "\n"
    with open(output_path, "w") as f:
        f.write(content)

    _print_log("INFO", f"Test script: {output_path}")


def _execute_test_script(
    script_path: str,
    python_exe: str,
    timeout: int = 600,
    bundle_path: str = "",
    section: int = 0,
    out_dir: str = "",
) -> Tuple[bool, str]:
    """
    统一执行测试脚本，返回 (passed: bool, output: str)。
    - python_exe: Python 解释器路径
    - bundle_path: 可选，设置 PYPTO_BUNDLE_PATH 环境变量，让脚本用指定 bundle 执行
    - section: 可选，执行前将 ASCEND_WORK_PATH 修改为 {out_dir}/section{section}
    - out_dir: 可选，debug_info.txt 同级目录（即 msaicerr 输出目录），与 section 拼接作为 ASCEND_WORK_PATH 基准
    """
    env = os.environ.copy()

    if section:
        new_work_path = os.path.join(out_dir, f"tmp/section{section}") if out_dir else f"section{section}"
        env["ASCEND_WORK_PATH"] = new_work_path
        _print_log("INFO", f"Section {section}: ASCEND_WORK_PATH={new_work_path}")

    if bundle_path:
        env["PYPTO_BUNDLE_PATH"] = bundle_path

    # Ensure pypto/lib is on LD_LIBRARY_PATH so libtile_fwk_runtime.so can be found
    if _PYTO_ROOT:
        pypto_lib = os.path.join(_PYTO_ROOT, "lib")
        if os.path.isdir(pypto_lib):
            ld_existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = os.pathsep.join([pypto_lib, ld_existing]) if ld_existing else pypto_lib

    cmd = [python_exe, script_path]
    _print_log("INFO", f"Executing command: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return False, f"[ERROR] {os.path.basename(script_path)} execution timed out ({timeout}s)"
    except Exception as e:
        return False, f"[ERROR] Script execution exception: {e}"

    output = ""
    if result.stdout:
        output += result.stdout
    if result.stderr:
        output += "\n[stderr]\n" + result.stderr
    return (result.returncode == 0), output


def run_test_script_and_update_info(script_path: str, device_id: int,
                                    python_exe: str,
                                    sections: Dict[str, str],
                                    timeout: int = 600,
                                    bundle_path: str = "",
                                    out_dir: str = "") -> bool:
    """执行测试脚本，将 bundle 模式结果覆盖写入 sections dict 中的 section 6。返回 True 表示通过。"""
    output_lines: List[str] = []

    output_lines.append(f"Bundled kernel: {bundle_path}")
    output_lines.append("")

    test_passed, output = _execute_test_script(script_path, python_exe, timeout, bundle_path,
                                               section=6, out_dir=out_dir)

    output_lines.append(f"Re-execute (subfunc):\n{output}")
    output_lines.append("")

    if test_passed:
        conclusion = "Conclusion: PASS in bundle mode → Operator execution is OK"
    else:
        conclusion = "Conclusion: FAIL in bundle mode → Reproduced with bundled kernel"
    output_lines.append(conclusion)
    _print_log("INFO", conclusion)

    # bundle 模式复现结果直接覆盖 section 6 已有内容
    sec6_key = _find_section6_key(sections)
    if sec6_key:
        sections[sec6_key] = "\n".join(output_lines)
    else:
        _print_log("WARNING", "section 6 not found in info.txt, skipping")

    return test_passed


# ===================================================================
# Phase E: Section 7 核内同步诊断（msnpureport）
# ===================================================================

def _find_msnpureport() -> str:
    """定位 msnpureport 工具。

    优先级:
    1. 环境变量 MSNPUREPORT_PATH（显式指定）
    2. ASCEND_HOME_PATH 父目录下 driver/tools/msnpureport（标准昇腾安装布局）
    3. 默认 /usr/local/Ascend/driver/tools/msnpureport（兜底）
    """
    env_path = os.environ.get("MSNPUREPORT_PATH", "")
    if env_path and os.path.isfile(env_path):
        return env_path

    ascend_home = os.environ.get("ASCEND_HOME_PATH", "")
    if ascend_home:
        ascend_root = os.path.dirname(os.path.abspath(ascend_home))
        cand = os.path.join(ascend_root, "driver", "tools", "msnpureport")
        if os.path.isfile(cand):
            return cand

    return _MSNPUREPORT_DEFAULT


def _msnpureport_set_singlecommit(enable: bool, device_id: int, is_docker: bool = False):
    """Enable/disable singlecommit mode. Returns (success, cmd_str, output)."""
    val = "1" if enable else "0"
    cmd = [_find_msnpureport(), "config", "--set", "--singlecommit", val, "-d", str(device_id)]
    if is_docker:
        cmd.append("--docker")
    cmd_str = " ".join(cmd)
    action = "enable" if enable else "restore"
    _print_log("INFO", f"{action} singlecommit: {cmd_str}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout.strip()
        if result.stderr:
            output += "\n" + result.stderr.strip()
        success = (result.returncode == 0)
    except subprocess.TimeoutExpired:
        output = f"[ERROR] msnpureport singlecommit config timed out (30s): {cmd_str}"
        success = False
    except Exception as e:
        output = f"[ERROR] {e}"
        success = False
    return success, cmd_str, output


def run_section7_intercore_sync(script_path: str, device_id: int,
                                 sections: Dict[str, str], header_order: List[str],
                                 python_exe: str,
                                 timeout: int = 600,
                                 bundle_path: str = "",
                                 out_dir: str = ""):
    """
    Phase E: msnpureport singlecommit=1 排除核内同步问题。
    """
    _print_log("INFO", "========== Section 7: Inter-Core Sync Diagnosis ==========")

    output_lines: List[str] = []

    is_docker = _is_docker_env()
    _print_log("INFO", f"Docker environment: {is_docker}")

    success, cmd_str, msn_output = _msnpureport_set_singlecommit(True, device_id, is_docker)
    output_lines.append(f"msnpureport enable single-step: {cmd_str}\n{msn_output}")
    output_lines.append("")
    if not success:
        _print_log("WARNING", f"msnpureport enable singlecommit failed: {msn_output}")

    test_passed, test_output = _execute_test_script(script_path, python_exe, timeout, bundle_path,
                                                    section=7, out_dir=out_dir)
    output_lines.append(f"Bundled kernel: {bundle_path}")
    output_lines.append("")

    output_lines.append(f"Re-execute (singlecommit=1):\n{test_output}")
    output_lines.append("")

    success2, cmd_str2, msn_output2 = _msnpureport_set_singlecommit(False, device_id, is_docker)
    output_lines.append(f"msnpureport restore: {cmd_str2}\n{msn_output2}")
    output_lines.append("")

    if test_passed:
        conclusion = "Conclusion: PASS in single-step mode → Inter-core sync issue"
    else:
        conclusion = "Conclusion: Still FAIL in single-step mode → Not an inter-core sync issue"
    output_lines.append(conclusion)
    _print_log("INFO", conclusion)

    _upsert_section7(sections, header_order, "\n".join(output_lines))
    return test_passed


# ===================================================================
# Phase F: Section 8 框架 vs 算子 CCE 诊断
# ===================================================================

def run_section8_framework_vs_cce(script_path: str, device_id: int,
                                   sections: Dict[str, str], header_order: List[str],
                                   python_exe: str,
                                   timeout: int = 600,
                                   bundle_path_undef: str = "",
                                   out_dir: str = ""):
    """
    Phase F: 用 *_nosubfunc.pyptokb 重新执行，判断 sub-func 是否为根因。
    """
    _print_log("INFO", "========== Section 8: Framework vs Operator CCE Diagnosis ==========")

    output_lines: List[str] = []

    if not bundle_path_undef or not os.path.isfile(bundle_path_undef):
        _print_log("WARNING", "_nosubfunc.pyptokb not found, skipping Section 8")
        output_lines.append("[ERROR] *_nosubfunc.pyptokb not found")
        _upsert_section8(sections, header_order, "\n".join(output_lines))
        return

    output_lines.append(f"Nosubfunc bundled kernel: {bundle_path_undef}")
    output_lines.append("")
    _print_log("INFO", f"Nosubfunc bundled kernel: {bundle_path_undef}")

    test_passed, test_output = _execute_test_script(script_path, python_exe, timeout, bundle_path_undef,
                                                    section=8, out_dir=out_dir)

    output_lines.append(f"Re-execute (nosubfunc):\n{test_output}")
    output_lines.append("")

    if test_passed:
        conclusion = "Conclusion: PASS with nosubfunc bundled kernel → Sub-func (CCE) issue"
    else:
        conclusion = "Conclusion: Still FAIL with nosubfunc bundled kernel → Framework issue (not sub-func)"
    output_lines.append(conclusion)
    _print_log("INFO", conclusion)

    _upsert_section8(sections, header_order, "\n".join(output_lines))


# ===================================================================
# CLI
# ===================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug AICore Error",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python debug_aicore_error.py -p /path/to/report -out /tmp/out\n"
            "  python debug_aicore_error.py -p /path/to/report -out /tmp/out -d 0\n"
            "  python debug_aicore_error.py -p /path/to/report -out /tmp/out -t 1200\n"
        ),
    )
    parser.add_argument(
        "-p", type=str, required=True,
        help="Path to AIC error debug info",
    )
    parser.add_argument(
        "-d", type=str, default=None,
        help="Device ID (optional, defaults to TILE_FWK_DEVICE_ID env variable)",
    )
    parser.add_argument(
        "-out", type=str, required=True,
        help="Output directory for the debug report",
    )
    parser.add_argument(
        "-t", type=int, default=600,
        help="Timeout threshold (seconds) for single-operator reproduction test script execution, default 600",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ============================================================
    # Phase A: 预处理
    # ============================================================

    # 获取 device id（-d 参数 > TILE_FWK_DEVICE_ID 环境变量）
    device_id = _get_device_id(args.d)
    _print_log("INFO", f"device_id = {device_id}")

    # 检查 -p 目录结构
    _validate_work_dir(args.p)

    # 新版 msaicerr 前置约束：-out 与 cwd 不得位于 -p 目录之下，且 cwd 可写
    _validate_paths_not_in_report_dir(args.p, args.out)

    # 获取并校验 CANN 环境变量（ASCEND_HOME_PATH / ASCEND_OPP_PATH）
    ascend_home = _get_ascend_env()

    # 检查日志级别
    _check_log_level()

    # 检测 Python 解释器
    python_exe = _detect_python()

    # 获取 pypto 安装路径
    global _PYTO_ROOT
    try:
        import pypto
        _PYTO_ROOT = os.path.dirname(os.path.abspath(pypto.__file__))
        _print_log("INFO", f"pypto root = {_PYTO_ROOT}")
    except ImportError:
        raise RuntimeError("cannot import pypto, please ensure pypto is installed")

    # 校验 libtile_fwk_bundle.so
    bundle_so = find_bundle_so()
    if not bundle_so:
        _print_log("ERROR", "libtile_fwk_bundle.so not found, please specify via PYPTO_BUNDLE_SO env variable")
        sys.exit(1)

    # ============================================================
    # Phase B: 调用 msaicerr.py 解析 (→ info.txt section 1~5)
    # ============================================================

    msaicerr_out = run_msaicerr(args.p, args.out, device_id, ascend_home)
    _init_debug_log(msaicerr_out)

    # ============================================================
    # Phase C: bundle 定位 (→ info.txt section 1)
    # ============================================================

    info_txt = find_info_txt(msaicerr_out)
    _print_log("INFO", f"info.txt: {info_txt}")
    debug_info_txt = find_debug_info_txt_path(msaicerr_out)

    preamble, sections, header_order = _parse_info_txt_to_sections(info_txt)

    kernel_func_name, tensors = _extract_from_sections(sections)

    if not tensors:
        _print_log("ERROR", "No dump tensors parsed, cannot generate reproduction script")
        sys.exit(1)

    _find_and_record_bundled_kernels(sections, args.p)

    bundle_path = _BUNDLED_KERNEL_PATH
    if not bundle_path:
        _print_log("ERROR", ".pyptokb file not found")
        sys.exit(1)

    # ============================================================
    # Phase D: Single-operator Test (→ info.txt section 6)
    # ============================================================

    _print_log("INFO", "========== Section 6: Single-Operator Test ==========")

    output_script = os.path.join(msaicerr_out, "test_single_op.py")
    codegen_bundle_script(
        bundle_path, kernel_func_name, tensors, device_id,
        output_script, bundle_so,
    )

    section6_passed = run_test_script_and_update_info(
        output_script, device_id, python_exe, sections,
        timeout=args.t, bundle_path=bundle_path, out_dir=msaicerr_out,
    )

    # ============================================================
    # Phase E: Inter-core sync diagnosis (→ info.txt section 7, only if Phase D fails)
    # ============================================================

    if section6_passed:
        _print_log("INFO", "Section 6 passed, skipping Section 7 and Section 8")
    else:
        is_sync_issue = run_section7_intercore_sync(
            output_script, device_id,
            sections, header_order,
            python_exe, timeout=args.t,
            bundle_path=bundle_path, out_dir=msaicerr_out,
        )

        # ============================================================
        # Phase F: Framework vs CCE root cause analysis (→ info.txt section 8, only if Phase E rules out sync)
        # ============================================================

        if not is_sync_issue:
            run_section8_framework_vs_cce(
                output_script, device_id,
                sections, header_order,
                python_exe, timeout=args.t,
                bundle_path_undef=_BUNDLED_KERNEL_PATH_UNDEF,
                out_dir=msaicerr_out,
            )
        else:
            _print_log("INFO", "Inter-core sync issue identified, skipping Section 8")

    # ============================================================
    # Phase G: Finalize
    # ============================================================

    _rewrite_info_txt(info_txt, preamble, sections, header_order)
    _print_log("INFO", f"debug_info.txt: {debug_info_txt}")
    _print_log("INFO", "All phases completed, please check " + info_txt)


if __name__ == "__main__":
    main()
