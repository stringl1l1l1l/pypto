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

"""Generated launch metadata for live reporting and captured graph replay."""

import subprocess

import pypto_pro.language as pl
from pypto_pro.runtime.compile_config import get_jit_compile_config
from pypto_pro.runtime.jit import (
    ParamKind,
    ParamSpec,
    _generate_caller_cpp,
    _generate_prof_range_snippet,
    _parse_and_codegen_targets,
)


def _snippet(kernel_name, specs, dims):
    target = get_jit_compile_config().resolve_kernel_target("3510", has_cube=False, has_vector=True)
    return _generate_prof_range_snippet(
        kernel_name, specs, dims, target=target, launch_stmt="    k<<<blockDim, nullptr, stream>>>(a, out);\n"
    )


def _spec(name, shape, dtype_str, direction=pl.Input):
    return ParamSpec(name, ParamKind.TENSOR, dtype_str, shape, direction)


def test_output_annotation_sets_direction():
    """pl.Output in the annotation flows to ParamSpec.direction; omitted defaults to INPUT."""
    @pl.jit()
    def marked_kernel(
        a: pl.Tensor[[64, 64], pl.DT_FP32],
        out: pl.Tensor[[64, 64], pl.DT_FP32, pl.Output],
    ):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        ta = pl.make_tile(tt, addr=0x0000)
        with pl.section_vector():
            pl.load(ta, a, [0, 0])
            pl.store(out, ta, [0, 0])

    kernel_def = marked_kernel.to_kernel_def()
    cube, vector = _parse_and_codegen_targets(kernel_def, "3510", "")
    cg = cube or vector
    assert kernel_def.last_param_directions == {1: pl.Output}
    assert cg.param_specs[0].direction is pl.Input
    assert cg.param_specs[1].direction is pl.Output


def test_call_style_output_annotation_sets_direction():
    @pl.jit()
    def marked_kernel(
        a: pl.Tensor([64, 64], pl.DT_FP32),
        out: pl.Tensor([64, 64], pl.DT_FP32, pl.Output),
    ):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        ta = pl.make_tile(tt, addr=0x0000)
        with pl.section_vector():
            pl.load(ta, a, [0, 0])
            pl.store(out, ta, [0, 0])

    kernel_def = marked_kernel.to_kernel_def()
    cube, vector = _parse_and_codegen_targets(kernel_def, "3510", "")
    cg = cube or vector
    assert kernel_def.last_param_directions == {1: pl.Output}
    assert cg.param_specs[0].direction is pl.Input
    assert cg.param_specs[1].direction is pl.Output


def test_output_direction_uses_absolute_parameter_index():
    @pl.jit()
    def marked_kernel(
        size: pl.DT_INT32,
        a: pl.Tensor[[64, 64], pl.DT_FP32],
        out: pl.Tensor[[64, 64], pl.DT_FP32, pl.Output],
    ):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        ta = pl.make_tile(tt, addr=0x0000)
        with pl.section_vector():
            pl.load(ta, a, [0, 0])
            pl.store(out, ta, [0, 0])

    kernel_def = marked_kernel.to_kernel_def()
    cube, vector = _parse_and_codegen_targets(kernel_def, "3510", "")
    cg = cube or vector
    assert kernel_def.last_param_directions == {2: pl.Output}
    assert cg.param_specs[0].kind is ParamKind.SCALAR
    assert cg.param_specs[0].direction is None
    assert cg.param_specs[1].direction is pl.Input
    assert cg.param_specs[2].name == "out_0"
    assert cg.param_specs[2].direction is pl.Output


def test_omitted_direction_defaults_to_input():
    """An omitted direction defaults to INPUT even if the kernel stores into the tensor."""

    @pl.jit()
    def unmarked_kernel(
        out: pl.Tensor[[64, 64], pl.DT_FP32],
    ):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        ta = pl.make_tile(tt, addr=0x0000)
        with pl.section_vector():
            pl.store(out, ta, [0, 0])

    cube, vector = _parse_and_codegen_targets(unmarked_kernel.to_kernel_def(), "3510", "")
    cg = cube or vector
    assert cg.param_specs[0].direction is pl.Input


def test_static_shapes_are_literals():
    specs = [
        _spec("a", [64, 64], "fp32"),
        _spec("b", [64, 64], "fp32"),
        _spec("out", [64, 64], "fp32", pl.Output),
    ]
    _inc, push, pop = _snippet("k", specs, set())
    assert '#include "acl/acl_prof.h"' in _inc
    assert "{0, 2, 0, 2, {64, 64, 0, 0, 0, 0, 0, 0}}," in push  # a: input
    assert "{1, 2, 0, 2, {64, 64, 0, 0, 0, 0, 0, 0}}," in push  # out: output
    assert 'BeginCaptureTensorReport(pyptoProfOn)' in push
    assert 'ReportCaptureTensorInfo(pyptoProfInfo, pyptoProfBegin, pyptoCaptureOn)' in pop
    assert 'aclprofRange' not in push + pop
    assert 'aclprofEventAttributes' not in push
    assert "aclprofStr2Id(\"PYPTO_k\")" in push
    assert "pyptoProfInfo.tensorNum = 3;" in push
    assert "pyptoProfInfo.kernelType = 2;" in push


def test_dynamic_dims_reference_launcher_params():
    dims = {"a", "b", "out", "__pypto_dyn_a_0", "__pypto_dyn_a_1"}
    specs = [_spec("a", ["__pypto_dyn_a_0", "__pypto_dyn_a_1"], "fp32")]
    _inc, push, _pop = _snippet("k", specs, dims)
    assert "{static_cast<uint32_t>(__pypto_dyn_a_0), static_cast<uint32_t>(__pypto_dyn_a_1)," in push


def test_unknown_dynamic_degrades_to_zero():
    specs = [_spec("a", [-1, 8], "fp16")]
    _inc, push, _pop = _snippet("k", specs, set())
    assert "{0, 2, 1, 2, {0, 8, 0, 0, 0, 0, 0, 0}}," in push


def test_dtype_mapping():
    specs = [
        _spec("a", [4], "fp16"),
        _spec("b", [4], "bf16"),
        _spec("c", [4], "int32"),
        _spec("d", [4], "nonexistent"),
    ]
    _inc, push, _pop = _snippet("k", specs, set())
    assert "{0, 2, 1, 1," in push  # fp16 -> 1
    assert "{0, 2, 27, 1," in push  # bf16 -> 27
    assert "{0, 2, 3, 1," in push  # int32 -> 3
    assert "{0, 2, 0, 1," in push  # unknown dtype -> 0 fallback


def test_no_tensor_params_disables_snippet():
    assert _snippet("k", [], set()) == ("", "", "")


def test_caller_embeds_metadata_around_launch_static():
    """End-to-end caller generation: pl.Output flows into the snippet."""
    @pl.jit()
    def static_kernel(
        a: pl.Tensor[[64, 64], pl.DT_FP32],
        out: pl.Tensor[[64, 64], pl.DT_FP32, pl.Output],
    ):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        ta = pl.make_tile(tt, addr=0x0000)
        tc = pl.make_tile(tt, addr=0x8000)
        with pl.section_vector():
            pl.load(ta, a, [0, 0])
            pl.add(tc, ta, ta)
            pl.store(out, tc, [0, 0])

    cube, vector = _parse_and_codegen_targets(static_kernel.to_kernel_def(), "3510", "")
    from pypto_pro.runtime.compile_config import get_jit_compile_config

    target = get_jit_compile_config().resolve_kernel_target("3510", has_cube=cube is not None,
                                                            has_vector=vector is not None)
    cg = cube or vector
    content = _generate_caller_cpp(
        kernel_params=cg.kernel_params,
        kernel_cpp_name="kernel.cpp",
        kernel_name=cg.kernel_name,
        target=target,
        prof_param_specs=cg.param_specs,
    )
    assert '#include "acl/acl_prof.h"' in content
    assert 'aclprofStr2Id("PYPTO_static_kernel")' in content
    push_idx = content.index("BeginCaptureTensorReport")
    launch_idx = content.index("<<<blockDim")
    pop_idx = content.index("ReportCaptureTensorInfo")
    assert push_idx < launch_idx < pop_idx
    assert "{0, 2, 0, 2, {64, 64" in content   # a: input
    assert "{1, 2, 0, 2, {64, 64" in content   # out: output


def test_metadata_checks_collection_and_capture_independently():
    """Capture before collection caches tensors; live node reports need active collection."""
    @pl.jit()
    def bare_kernel(
        a: pl.Tensor[[64, 64], pl.DT_FP32],
        out: pl.Tensor[[64, 64], pl.DT_FP32, pl.Output],
    ):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        ta = pl.make_tile(tt, addr=0x0000)
        tc = pl.make_tile(tt, addr=0x8000)
        with pl.section_vector():
            pl.load(ta, a, [0, 0])
            pl.add(tc, ta, ta)
            pl.store(out, tc, [0, 0])

    cube, vector = _parse_and_codegen_targets(bare_kernel.to_kernel_def(), "3510", "")
    from pypto_pro.runtime.compile_config import get_jit_compile_config

    target = get_jit_compile_config().resolve_kernel_target("3510", has_cube=cube is not None,
                                                            has_vector=vector is not None)
    cg = cube or vector
    content = _generate_caller_cpp(
        kernel_params=cg.kernel_params,
        kernel_cpp_name="kernel.cpp",
        kernel_name=cg.kernel_name,
        target=target,
        prof_param_specs=cg.param_specs,
    )
    guard_idx = content.index("if (pyptoProfOn || pyptoCaptureOn) {")
    tensors_idx = content.index("aclprofTensor pyptoProfTensors[]")
    begin_idx = content.index("pypto::BeginCaptureTensorReport(pyptoProfOn)")
    launch_idx = content.index("<<<blockDim")
    report_idx = content.index("pypto::ReportCaptureTensorInfo")
    else_idx = content.index("} else {")
    assert guard_idx < tensors_idx < begin_idx < launch_idx < report_idx < else_idx
    assert "ACL_STREAM_ATTR_CACHE_OP_INFO" in content
    assert "pyptoCaptureAttr.cacheOpInfoSwitch" in content
    assert "const bool pyptoProfOn = true;" not in content
    assert "pypto::GetProfStatus()" in content
    assert "getenv" not in content
    assert content.count("<<<blockDim") == 2
    assert "ReportCaptureTensorInfo" not in content[else_idx:]


def test_native_metadata_gate(tmp_path):
    """Execute the emitted control flow with recording APIs; an ordinary launch must not report."""
    target = get_jit_compile_config().resolve_kernel_target("3510", has_cube=False, has_vector=True)
    _inc, push, pop = _generate_prof_range_snippet(
        "k", [_spec("a", [64], "fp32", pl.Input)], set(), target=target, launch_stmt="    ++launches;\n"
    )
    # Only API types/functions and the device launch are stubs; compile the actual generated guard.
    harness = r"""
#include <cassert>
#include <cstdint>
#include <cstdio>
constexpr int ACL_SUCCESS = 0, ACL_STREAM_ATTR_CACHE_OP_INFO = 7;
struct aclrtStreamAttrValue { bool cacheOpInfoSwitch; };
struct aclprofTensor { uint32_t type, format, dataType, dimNum, shape[8]; };
struct aclprofTensorInfo {
    uint64_t opNameId, opTypeId;
    uint32_t resv, tensorNum, kernelType, blockNums;
    void* stream;
    aclprofTensor* tensors;
};
static int profiling, capture, queryError, queries, launches, reports, caches, strings, activeBegins;
namespace pypto {
static bool GetProfStatus() { return profiling; }
}
static int aclrtGetStreamAttribute(void* stream, int attribute, aclrtStreamAttrValue* value)
{
    assert(stream == reinterpret_cast<void*>(0x1234));
    assert(attribute == ACL_STREAM_ATTR_CACHE_OP_INFO);
    ++queries;
    value->cacheOpInfoSwitch = capture;
    return queryError;
}
static uint64_t aclprofStr2Id(const char*) { return ++strings; }
namespace pypto {
static uint64_t BeginCaptureTensorReport(bool profiling)
{
    activeBegins += profiling;
    return profiling ? 123 : 0;
}
static void ReportCaptureTensorInfo(const aclprofTensorInfo& info, uint64_t begin, bool cache)
{
    assert(launches == 1);
    assert(info.tensors[0].shape[0] == 64);
    reports += begin != 0;
    caches += cache;
}
}
static void launch(uint32_t blockDim, void* stream)
{
""" + push + "    ++launches;\n" + pop + r"""
}
int main()
{
    // Reuse the same stream as collection/capture changes to expose a cached guard.
    // Fields: process-wide collection, capture, query error.
    const int cases[][3] = {{0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 0}, {1, 1, 1}, {0, 1, 1}, {0, 0, 0}};
    for (const auto& test : cases) {
        profiling = test[0]; capture = test[1]; queryError = test[2];
        queries = launches = reports = caches = strings = activeBegins = 0;
        launch(8, reinterpret_cast<void*>(0x1234));
        std::printf("%d %d %d %d %d %d\n", launches, reports, caches, strings, activeBegins, queries);
    }
}
"""
    source = tmp_path / "range_gate.cpp"
    source.write_text(harness)
    binary = tmp_path / "range_gate"
    subprocess.run(["g++", "-std=c++17", "-O2", str(source), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    actual = [tuple(map(int, line.split())) for line in result.stdout.splitlines()]
    # Counts: launches, live reports, caches, registered strings, active begins, capture queries.
    assert actual == [
        (1, 0, 0, 0, 0, 1),  # No collection, no capture.
        (1, 0, 1, 2, 0, 1),  # Capture before collection only caches.
        (1, 1, 0, 2, 1, 1),  # Collection alone reports live metadata.
        (1, 1, 1, 2, 1, 1),  # Capture during collection must also cache for later replay.
        (1, 1, 0, 2, 1, 1),  # Collection still reports despite a failed capture query.
        (1, 0, 0, 0, 0, 1),  # A failed capture query alone does not report.
        (1, 0, 0, 0, 0, 1),  # Returning to ordinary execution must stop all metadata calls.
    ]
