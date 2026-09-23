# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Keep compiler target selection, emitted launch geometry and debug compilation consistent."""

from dataclasses import replace
import importlib
from pathlib import Path
from unittest.mock import MagicMock

from pypto_pro._errors import InvalidVal, RuntimeFailure
import pypto_pro.language as pl
from pypto_pro.runtime.compile_config import JitCompileConfig, KernelTarget, get_jit_compile_config
import pytest

jit = importlib.import_module("pypto_pro.runtime.jit")


@pytest.fixture
def future_config():
    """A hypothetical 1:1 architecture must coexist with A5; its compiler name is a test-only placeholder."""
    config = get_jit_compile_config()
    return replace(
        config,
        kernel_targets={
            **config.kernel_targets,
            "future": {"cube_vec": KernelTarget("test-mixed-1-1", 1, 1, True)},
        },
        memory_arch_flags={**config.memory_arch_flags, "future": "-DTEST_MEMORY_ARCH"},
    )


@pytest.mark.parametrize("arch,npu_arch,memory", [
    ("a2", "dav-2201", "-DMEMORY_BASE"),
    ("a3", "dav-2201", "-DMEMORY_BASE"),
    ("3510", "dav-3510", "-DREGISTER_BASE"),
])
@pytest.mark.parametrize("cube,vector,qualifier,cores,kernel_type", [
    (True, False, "__cube__", (1, 0), 0),
    (False, True, "__vector__", (0, 1), 2),
    (True, True, "__mix__(1, 2)", (1, 2), 4),
])
def test_asc_targets_keep_entry_profiling_and_launch_geometry_consistent(
    arch, npu_arch, memory, cube, vector, qualifier, cores, kernel_type,
):
    """ASC needs explicit core qualifiers; inferring from instructions fails for scalar-only probes."""
    config = get_jit_compile_config()
    target = config.resolve_kernel_target(arch, has_cube=cube, has_vector=vector)
    flags = config.build_bisheng_flags(toolkit_home="/toolkit", arch=arch, target=target, enable_print_debug=False)
    entry = jit._make_global_entry("probe", [], target=target)
    caller = jit._generate_caller_cpp(
        [], "kernel.cpp", "probe", target=target, global_entry=entry,
        prof_param_specs=[jit.ParamSpec("out", jit.ParamKind.TENSOR, "int32", [32], pl.Output)],
    )
    assert target.npu_arch == npu_arch
    assert (target.aic_per_block, target.aiv_per_block) == cores
    assert "-xasc" in flags
    assert f"--npu-arch={npu_arch}" in flags
    assert "-xcce" not in flags
    assert "--cce-fatobj-link" not in flags
    assert f"__global__ {qualifier} void probe(" in caller
    assert f"pyptoProfInfo.kernelType = {kernel_type};" in caller
    if cube and vector:
        assert "pyptoProfInfo.blockNums = (blockDim & 0xffffU) | (2U << 16);" in caller
    else:
        assert "pyptoProfInfo.blockNums = blockDim;" in caller
    assert memory in flags
    assert f"ResolveLaunchBlockDim<{cores[0]}, {cores[1]}>" in caller
    if arch in ("a2", "a3"):
        llvm_args = config.build_llvm_args(arch)
        assert llvm_args[llvm_args.index("-include") + 1] == "kernel_operator.h"


def test_simt_launcher_bakes_inferred_dynamic_ub_into_generated_caller():
    target = get_jit_compile_config().resolve_kernel_target("3510", has_cube=False, has_vector=True)
    caller = jit._generate_caller_cpp(
        [], "kernel.cpp", "probe", target=target, required_dynamic_ub_size=16 * 1024,
    )
    flags = jit._build_bisheng_flags(
        "/toolkit", "3510", target, has_cross_sync=False, enable_print_debug=False,
    )
    assert "uint32_t dynamicUbSize" not in caller
    assert 'call_kernel(uint32_t blockDim, void* stream)' in caller
    assert "probe<<<blockDim, 16384, stream>>>" in caller
    assert "get_required_dynamic_ub_size" not in caller
    assert "-cce-dyn-kernel-stack-size=false" not in flags


def test_architecture_specific_mixed_geometry_is_independent(future_config):
    """A future 1:1 target must not overwrite the 1:2 rule of an existing architecture."""
    a5 = future_config.resolve_kernel_target(" 3510 ", has_cube=True, has_vector=True)
    future = future_config.resolve_kernel_target("future", has_cube=True, has_vector=True)
    assert (a5.aic_per_block, a5.aiv_per_block) == (1, 2)
    assert (future.aic_per_block, future.aiv_per_block) == (1, 1)
    assert "__mix__(1, 1)" in jit._make_global_entry("probe", [], target=future)


def test_missing_architecture_is_not_assumed_to_use_a5_geometry():
    """Adding an extensible target table must not advertise unsupported A6 compilation."""
    with pytest.raises(InvalidVal, match="kernel_targets.*a6"):
        get_jit_compile_config().resolve_kernel_target("a6", has_cube=True, has_vector=True)


def test_missing_kernel_mode_is_not_replaced_with_mixed_target(future_config):
    """A missing single-engine compiler target cannot safely use a mixed binary's launch ABI."""
    with pytest.raises(InvalidVal, match=r"kernel_targets.future.cube"):
        future_config.resolve_kernel_target("future", has_cube=True, has_vector=False)


def test_empty_kernel_is_rejected_before_selecting_flags():
    """The old default arch variant could hide a kernel with neither execution engine."""
    with pytest.raises(RuntimeFailure, match="add a target section"):
        get_jit_compile_config().resolve_kernel_target("3510", has_cube=False, has_vector=False)


@pytest.mark.parametrize("cube,vector,missing", [(True, False, "vector"), (False, True, "cube")])
def test_cross_sync_still_requires_both_engines(cube, vector, missing):
    """Switching flag generation to a descriptor must retain the incomplete-sync diagnostic."""
    target = get_jit_compile_config().resolve_kernel_target("3510", has_cube=cube, has_vector=vector)
    with pytest.raises(ValueError, match=f"{missing} code is missing"):
        jit._build_bisheng_flags("/toolkit", "3510", target, has_cross_sync=True, enable_print_debug=False)


@pytest.mark.parametrize("arch,cores", [("3510", (1, 2)), ("future", (1, 1))])
def test_shared_library_and_caller_use_the_same_resolved_target(monkeypatch, tmp_path, future_config, arch, cores):
    """Capture a real build command and caller; independently resolving either half would permit ABI drift."""
    target = future_config.resolve_kernel_target(arch, has_cube=True, has_vector=True)
    monkeypatch.setattr(jit, "get_jit_compile_config", lambda: future_config)
    monkeypatch.setattr(
        JitCompileConfig, "resolve_kernel_target", MagicMock(side_effect=AssertionError("Already resolved")),
    )
    monkeypatch.setenv("ASCEND_TOOLKIT_HOME", "/toolkit")
    monkeypatch.setenv("ASCEND_HOME_PATH", "/toolkit")
    monkeypatch.setattr(jit.shutil, "which", lambda _: "/toolkit/bin/bisheng")
    captured = []

    def compile_stub(_bisheng, flags, build_arch, paths, _links, _timeout, output_path):
        captured.append((flags, build_arch, Path(paths.final_kernel).read_text()))
        Path(output_path).write_bytes(b"compiled test library")
        return True

    monkeypatch.setattr(jit, "_run_bisheng", compile_stub)
    cg = jit.CodegenResult(
        build_dir=str(tmp_path), content="", kernel_name="probe", kernel_params=[],
        caller_cross_core_sync=True, bisheng_cross_sync=True, needs_print_debug=True,
        has_cube=True, has_vector=True, required_dynamic_ub_size=16 * 1024,
    )
    library = jit._build_jit_so(cg, arch, clean_up=False, compile_timeout=30, target=target)
    assert Path(library).read_bytes() == b"compiled test library"
    assert len(captured) == 1
    flags, build_arch, caller = captured[0]
    assert build_arch == arch
    assert f"--npu-arch={target.npu_arch}" in flags
    assert "-xasc" in flags
    assert "--cce-enable-print" not in flags
    assert "/toolkit/asc/include" in flags
    assert "/toolkit/asc" in flags
    assert f"ResolveLaunchBlockDim<{cores[0]}, {cores[1]}>" in caller
    assert "-cce-dyn-kernel-stack-size=false" not in flags
    assert "probe<<<blockDim, 16384, stream>>>" in caller


def test_compilation_preserves_target_for_cached_launch_and_debug(monkeypatch, tmp_path):
    """The compiled object must retain the exact descriptor selected when its binary was built."""
    config = get_jit_compile_config()
    resolve = MagicMock(wraps=config.resolve_kernel_target)
    config_proxy = MagicMock()
    config_proxy.resolve_kernel_target = resolve
    monkeypatch.setattr(jit, "get_jit_compile_config", lambda: config_proxy)
    monkeypatch.setattr(jit, "_setup_arch_env", lambda arch: arch)
    cg = jit.CodegenResult(
        build_dir=str(tmp_path), content="", kernel_name="probe", kernel_params=[],
        caller_cross_core_sync=False, bisheng_cross_sync=False, needs_print_debug=False,
        has_cube=True, has_vector=True,
    )
    monkeypatch.setattr(jit, "_codegen", lambda *_args, **_kwargs: cg)
    build = MagicMock(return_value="/tmp/test-kernel.so")
    monkeypatch.setattr(jit, "_build_jit_so", build)
    kernel = jit._TileJitKernel(lambda: None, arch="3510", compile_timeout=30)
    monkeypatch.setattr(kernel, "to_kernel_def", lambda *_args, **_kwargs: None)
    compiled = kernel._compile_variant(None, (None, None, None, None), None, ())
    resolve.assert_called_once_with("3510", has_cube=True, has_vector=True)
    assert compiled.target is build.call_args.kwargs["target"]
    assert compiled.target is config.kernel_targets["3510"]["cube_vec"]


def test_debug_command_reuses_compiled_target_without_resolving(monkeypatch, tmp_path):
    """Exception-dump setup runs before launches; it must reuse metadata rather than infer core geometry again."""
    dump = importlib.import_module("pypto_pro.runtime.exception_dump")
    (tmp_path / "call_kernel.cpp").write_text("")
    monkeypatch.setenv("ASCEND_WORK_PATH", str(tmp_path))
    monkeypatch.setenv("ASCEND_HOME_PATH", "/toolkit")
    monkeypatch.setenv("ASCEND_TOOLKIT_HOME", "/toolkit")
    monkeypatch.setenv("PYPTOPRO_JIT_ARCH", "3510")
    monkeypatch.setattr(dump.shutil, "which", lambda _: "/toolkit/bin/bisheng")
    monkeypatch.setattr(
        JitCompileConfig, "resolve_kernel_target", MagicMock(side_effect=AssertionError("Already resolved")),
    )
    target = KernelTarget("test-compiled-target", 1, 1, True)
    compiled = jit.CompiledKernel(
        lib_path="/tmp/test-kernel.so", param_specs=[],
        kernel_name="probe", build_dir=str(tmp_path), target=target,
    )
    command = dump._build_debug_compile_cmd(compiled)
    assert "--npu-arch=test-compiled-target" in command
    assert "--npu-arch=dav-3510" not in command
