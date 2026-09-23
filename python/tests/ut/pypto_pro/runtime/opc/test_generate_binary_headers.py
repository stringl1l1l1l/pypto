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
"""Tests for OPC binary-delivery header generation."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
from types import SimpleNamespace

from asc_op_compile_base.asc_op_compiler.ascendc_constants import KernelMetaType
from pypto_pro._errors import InvalidArgument
import pypto_pro.language as pl
from pypto_pro.runtime.opc.pypto_compile import (
    _build_compile_info,
    _build_tiling_info,
    _compile_core_names,
    _compile_target_name,
    _compile_target_options,
    _gen_infer_cpp,
    _kernel_type_from_codegen,
    _kernel_type_from_target,
    _load_kernel,
    generate_binary_headers,
    prepare_binary_headers,
    pypto_compile_op,
)
from pypto_pro.runtime.tilingkey import TilingKeyField
import pytest


@dataclass
class HeaderTiling:
    rows: int
    columns: int
    offsets: int[4]


class HeaderTilingKey:
    Operation = TilingKeyField(bits=1, values=[0, 1])


@pl.jit(auto_mutex=True, tiling_key=HeaderTilingKey, datatype={"x": "input_dtype"})
def header_generation_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    tiling: HeaderTiling,
):
    tile_type = pl.TileType(
        shape=[16, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec
    )
    tile_x = pl.make_tile(tile_type, addr=0x0000)
    tile_y = pl.make_tile(tile_type, addr=0x0200)
    tile_z = pl.make_tile(tile_type, addr=0x0400)

    with pl.section_vector():
        for row in pl.range(0, tiling.rows, 16):
            for column in pl.range(0, tiling.columns, 16):
                pl.load(tile_x, x, [row, column])
                pl.load(tile_y, y, [row, column])
                if Operation == 0:  # noqa: F821
                    pl.add(tile_z, tile_x, tile_y)
                else:
                    pl.sub(tile_z, tile_x, tile_y)
                pl.store(z, tile_z, [row, column])


def test_generate_binary_headers_emits_only_tiling_headers(monkeypatch, tmp_path):
    jit_runtime = importlib.import_module("pypto_pro.runtime.jit")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        jit_runtime,
        "_codegen",
        lambda *args, **kwargs: pytest.fail("generate_binary_headers must not run full codegen"),
    )
    binary_dir = Path(generate_binary_headers(header_generation_kernel, "a5"))
    assert os.environ["PYPTOPRO_JIT_ARCH"] == "a5"
    assert binary_dir.is_dir(), f"binary dir not created: {binary_dir}"

    tiling_header = binary_dir / "HeaderTiling_tiling.h"
    tilingkey_header = binary_dir / "HeaderTilingKey_tilingkey.h"
    assert tiling_header.is_file(), f"missing tiling header: {tiling_header}"
    assert tilingkey_header.is_file(), f"missing tilingkey header: {tilingkey_header}"

    tiling_text = tiling_header.read_text(encoding="utf-8")
    assert "class HeaderTiling" in tiling_text
    assert "int64_t rows;" in tiling_text
    assert "int64_t columns;" in tiling_text
    assert "int64_t offsets[4];" in tiling_text

    tilingkey_text = tilingkey_header.read_text(encoding="utf-8")
    assert "ASCENDC_TPL_ARGS_DECL(header_generation_kernel" in tilingkey_text
    assert "ASCENDC_TPL_SEL(" in tilingkey_text
    assert not list(binary_dir.glob("*_pypto_infer.cpp"))
    assert not list(binary_dir.parent.glob("tk_*"))


def test_prepare_binary_headers_loads_the_kernel_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYPTOPRO_JIT_ARCH", "a3")

    binary_dir = Path(prepare_binary_headers(__file__))

    assert os.environ["PYPTOPRO_JIT_ARCH"] == "a5"
    assert (binary_dir / "HeaderTiling_tiling.h").is_file()
    assert (binary_dir / "HeaderTilingKey_tilingkey.h").is_file()
    assert not list(binary_dir.glob("*_pypto_infer.cpp"))


def test_pypto_compile_op_sets_up_explicit_arch(monkeypatch):
    compile_module = importlib.import_module("pypto_pro.runtime.opc.pypto_compile")
    jit_runtime = importlib.import_module("pypto_pro.runtime.jit")
    received_arch = []

    def stop_after_arch_setup(*args, **kwargs):
        raise RuntimeError("stop after arch setup")

    monkeypatch.setenv("PYPTO_JIT_ARCH", "a2")
    monkeypatch.setattr(jit_runtime, "_setup_arch_env", lambda arch: received_arch.append(arch) or arch)
    monkeypatch.setattr(compile_module, "_setup_options", stop_after_arch_setup)

    with pytest.raises(RuntimeError, match="stop after arch setup"):
        pypto_compile_op("unused.py", "kernel", {"kernel_name": "kernel"}, arch="a5")

    assert received_arch == ["a5"]


@pytest.mark.parametrize(
    "aic_per_block,aiv_per_block,expected",
    [
        (1, 0, KernelMetaType.KERNEL_TYPE_AIC_ONLY),
        (0, 1, KernelMetaType.KERNEL_TYPE_AIV_ONLY),
        (1, 1, KernelMetaType.KERNEL_TYPE_MIX_AIC_1_1),
        (1, 2, KernelMetaType.KERNEL_TYPE_MIX_AIC_1_2),
    ],
)
def test_kernel_type_from_target_uses_configured_core_ratio(aic_per_block, aiv_per_block, expected):
    target = SimpleNamespace(aic_per_block=aic_per_block, aiv_per_block=aiv_per_block)

    assert _kernel_type_from_target(target) is expected


@pytest.mark.parametrize(
    "has_cube,has_vector,expected",
    [
        (True, False, KernelMetaType.KERNEL_TYPE_AIC_ONLY),
        (False, True, KernelMetaType.KERNEL_TYPE_AIV_ONLY),
        (True, True, KernelMetaType.KERNEL_TYPE_MIX_AIC_1_2),
    ],
)
def test_kernel_type_from_codegen_reuses_a5_target_config(has_cube, has_vector, expected):
    codegen_result = SimpleNamespace(has_cube=has_cube, has_vector=has_vector)

    assert _kernel_type_from_codegen(codegen_result, "a5") is expected


@pytest.mark.parametrize(
    "kernel_type,expected",
    [
        (KernelMetaType.KERNEL_TYPE_AIC_ONLY, ("cube",)),
        (KernelMetaType.KERNEL_TYPE_AIV_ONLY, ("vec",)),
        (KernelMetaType.KERNEL_TYPE_MIX_AIC_1_1, ("cube", "vec")),
        (KernelMetaType.KERNEL_TYPE_MIX_AIC_1_2, ("cube", "vec")),
    ],
)
def test_compile_core_names_follow_kernel_type(kernel_type, expected):
    assert _compile_core_names(kernel_type) == expected


def test_compile_target_name_uses_mix_suffix_only_for_mixed_kernel():
    assert _compile_target_name("probe", "3", "cube", KernelMetaType.KERNEL_TYPE_AIC_ONLY) == "probe_3"
    assert (
        _compile_target_name("probe", "3", "cube", KernelMetaType.KERNEL_TYPE_MIX_AIC_1_2)
        == "probe_3_mix_aic"
    )
    assert (
        _compile_target_name("probe", "3", "vec", KernelMetaType.KERNEL_TYPE_MIX_AIC_1_2)
        == "probe_3_mix_aiv"
    )


@pytest.mark.parametrize(
    "kernel_type,expected",
    [
        (KernelMetaType.KERNEL_TYPE_AIC_ONLY, ["-DRAW_AIC_ONLY_DUMP_TENSOR"]),
        (KernelMetaType.KERNEL_TYPE_AIV_ONLY, []),
        (
            KernelMetaType.KERNEL_TYPE_MIX_AIC_1_1,
            ["-D__MIX_CORE_MACRO__=1", "-D__MIX_CORE_AIC_RATION__=1"],
        ),
        (
            KernelMetaType.KERNEL_TYPE_MIX_AIC_1_2,
            ["-D__MIX_CORE_MACRO__=1", "-D__ASCENDC_ENABLE_VEC_TAIL_TILING_COPY__"],
        ),
    ],
)
def test_compile_target_options_match_c310_compile_op(kernel_type, expected):
    assert _compile_target_options(kernel_type, enable_c310_rules=True) == expected


def test_infer_cpp_uses_resolved_kernel_type():
    codegen_result = SimpleNamespace(entry_params=[], kernel_name="probe")

    source = _gen_infer_cpp(
        codegen_result,
        "ProbeKey_tilingkey.h",
        "",
        KernelMetaType.KERNEL_TYPE_AIV_ONLY,
    )

    assert "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);" in source
    assert "KERNEL_TYPE_MIX_AIC_1_2" not in source


@pytest.mark.parametrize(
    "kernel_type,expected_task_ration",
    [
        (KernelMetaType.KERNEL_TYPE_MIX_AIC_1_1, 1),
        (KernelMetaType.KERNEL_TYPE_MIX_AIC_1_2, 2),
        (KernelMetaType.KERNEL_TYPE_AIV_ONLY, 2),
    ],
)
def test_build_tiling_info_uses_inferred_kernel_type(monkeypatch, kernel_type, expected_task_ration):
    compile_module = importlib.import_module("pypto_pro.runtime.opc.pypto_compile")
    tiling_info = SimpleNamespace(task_ration=2)
    monkeypatch.setattr(compile_module, "get_tiling_info_by_tiling", lambda *args: tiling_info)

    result = _build_tiling_info({}, SimpleNamespace(default_kernel_type=kernel_type), None, "probe")

    assert result.task_ration == expected_task_ration


def test_build_compile_info_uses_inferred_kernel_types_and_filtered_keys(monkeypatch, tmp_path):
    compile_module = importlib.import_module("pypto_pro.runtime.opc.pypto_compile")
    inferred_kernel_types = {
        "1": KernelMetaType.KERNEL_TYPE_AIV_ONLY,
        "2": KernelMetaType.KERNEL_TYPE_AIV_ONLY,
    }
    inferred_info = SimpleNamespace(
        code_channel=1,
        tiling_key_list=["1", "2"],
        tiling_key_group_map={},
        hard_sync=False,
        enable_deterministic=False,
        tiling_key_deterministic={},
        tiling_key_kernel_type=inferred_kernel_types,
        no_set_kernel_type=False,
        default_kernel_type=KernelMetaType.KERNEL_TYPE_AIV_ONLY,
        template_tiling_info=None,
        tiling_key_struct_map={},
        register_tiling_struct=set(),
        tpl_tiling_struct=set(),
    )
    monkeypatch.setattr(compile_module.CommonUtility, "get_kernel_meta_dir", lambda: str(tmp_path))
    monkeypatch.setattr(compile_module, "handle_sk_codegen_options", lambda *args: None)

    compile_info = _build_compile_info(
        "probe.cpp",
        "probe",
        "probe_origin",
        {"op_type": "Probe"},
        inferred_info,
        ["2"],
        None,
    )

    assert compile_info.tiling_key_list == ["2"]
    assert compile_info.default_kernel_type == KernelMetaType.KERNEL_TYPE_AIV_ONLY
    assert compile_info.no_set_kernel_type is False
    assert compile_info.tiling_key_kernel_type is inferred_kernel_types
    assert compile_info.raw_tiling_key_kernel_type == inferred_kernel_types
    assert compile_info.raw_tiling_key_kernel_type is not inferred_kernel_types


@pytest.mark.parametrize("kernel_count", [0, 2])
def test_load_kernel_requires_exactly_one_jit_kernel(tmp_path, kernel_count):
    definitions = "\n".join(
        (
            f"@pl.jit\n"
            f"def kernel_{index}(x: pl.Ptr[pl.DT_FP16]):\n"
            "    pass\n"
        )
        for index in range(kernel_count)
    )
    kernel_file = tmp_path / "kernels.py"
    kernel_file.write_text(f"import pypto_pro.language as pl\n{definitions}", encoding="utf-8")

    with pytest.raises(InvalidArgument, match=rf"exactly one @pl.jit kernel, found {kernel_count}"):
        _load_kernel(str(kernel_file))
