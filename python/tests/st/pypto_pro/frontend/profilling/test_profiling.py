#!/usr/bin/env python3
# coding: utf-8
# ruff: noqa: E501
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Validate core type and tensors for eager and captured PyPTO launches."""

import csv
import ctypes
import glob
import importlib
import json
import logging
import os
import sys
import time

import pypto_pro.language as pl
import pytest
import torch
import torch_npu

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

pytestmark = pytest.mark.skip_jit_discovery(
    reason="Profiler collection and analysis need real launches and run once in the execution phase"
)


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


def _build_experimental_config():
    experimental_config_cls = getattr(torch_npu.profiler, "_ExperimentalConfig")
    return experimental_config_cls(
        export_type=[torch_npu.profiler.ExportType.Text],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        data_simplification=False,
        op_attr=True,
    )


def _kernel_rows(output_dir, kernel_name):
    """Check both the PyTorch export and the underlying CANN op summary."""
    for pattern in ("kernel_details.csv", "op_summary_*.csv"):
        # Some toolkit builds analyze profiler output in a background process that
        # may still be running when the context manager exits; the first profiling
        # test in the suite is the most exposed to this cold-start delay.
        files = glob.glob(os.path.join(output_dir, "**", pattern), recursive=True)
        for _ in range(60):
            if files:
                break
            time.sleep(0.5)
            files = glob.glob(os.path.join(output_dir, "**", pattern), recursive=True)
        assert files, f"No {pattern} under {output_dir}"
        matches = []
        for csv_file in files:
            with open(csv_file, "r", encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    if any(kernel_name in row.get(column, "") for column in ("Type", "Name", "OP Type", "Op Name")):
                        matches.append(row)
        assert matches, f"No {kernel_name} record in {files}"
        logging.info("PyPTO %s: %s", pattern, matches)
        yield matches


# =============================================================================
# 被采集的 kernel — 简单 add 操作
# =============================================================================
@pl.jit()
def prof_add_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32],
    b: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32, pl.Output],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tt, addr=0x0000)
    tb = pl.make_tile(tt, addr=0x4000)
    tc = pl.make_tile(tt, addr=0x8000)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.add(tc, ta, tb)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pl.jit()
def prof_cube_kernel(out: pl.Tensor[[32], pl.DT_INT32, pl.Output]):
    with pl.section_cube():
        out[0] = pl.get_block_num()


@pl.jit()
def prof_mixed_kernel(
    cube: pl.Tensor[[32], pl.DT_INT32, pl.Output],
    vector: pl.Tensor[[64], pl.DT_INT32, pl.Output],
):
    with pl.section_cube():
        cube[0] = pl.get_block_num()
    with pl.section_vector():
        vector[pl.get_block_idx() * 32] = pl.get_subblock_num()


def _case(kind):
    """Reference values also verify which engines and subblocks actually ran."""
    if kind == "vector":
        x = torch.randn(64, 64, device=ST_DEVICE, dtype=torch.float32)
        y = torch.randn_like(x)
        z = torch.zeros_like(x)
        return prof_add_kernel, (x, y, z), (z,), (x + y,), "AI_VECTOR_CORE", "64,64"
    cube = torch.zeros(32, dtype=torch.int32, device=ST_DEVICE)
    expected_cube = torch.zeros_like(cube)
    expected_cube[0] = 1
    if kind == "cube":
        return prof_cube_kernel, (cube,), (cube,), (expected_cube,), "AI_CORE", "32"
    vector = torch.zeros(64, dtype=torch.int32, device=ST_DEVICE)
    expected_vector = torch.zeros_like(vector)
    expected_vector[::32] = 2
    return (
        prof_mixed_kernel, (cube, vector), (cube, vector), (expected_cube, expected_vector), "MIX_AIC", "32;64"
    )


@pytest.mark.soc("950")
@pytest.mark.skip(reason="parallel read/write")
@pytest.mark.parametrize("kind", ["cube", "vector", "mixed"])
@pytest.mark.parametrize("mode", ["graph_before_collection", "eager", "graph"])
@pypto.options(pass_options={"enable_slice": False})
def test_profiler_api_outputs(tmp_path, kind, mode):
    """Check exported tensors and core types, including graphs captured before collection."""
    _check_npu()
    output_dir = str(tmp_path / "profiling_output")
    kernel, args, outputs, expected, core_type, output_shapes = _case(kind)
    kernel[1](*args)
    torch.npu.synchronize()
    skip_first = 2 if mode == "graph_before_collection" else 0
    for actual, reference in zip(outputs, expected):
        torch.testing.assert_close(actual, reference)
        actual.zero_()
    torch.npu.synchronize()

    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        with_stack=False,
        record_shapes=False,
        profile_memory=True,
        experimental_config=_build_experimental_config(),
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=skip_first),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(output_dir, analyse_flag=True),
    ) as prof:
        for step in range(skip_first + 1):
            if mode == "eager":
                kernel[1](*args)
            else:
                if step == 0:
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        kernel[1](*args)
                graph.replay()
            torch.npu.synchronize()
            if step < skip_first:
                for actual in outputs:
                    actual.zero_()
                torch.npu.synchronize()
            prof.step()
    for actual, reference in zip(outputs, expected):
        torch.testing.assert_close(actual, reference)

    output_dtypes = {"vector": "FLOAT", "cube": "INT32", "mixed": "INT32;INT32"}[kind]
    for matches in _kernel_rows(output_dir, kernel.__name__):
        assert len(matches) == 1, matches
        for row in matches:
            assert row.get("Accelerator Core", row.get("Task Type")) == core_type, row
            assert row["Block Num"] == "1", row
            assert row["Mix Block Num"] == ("2" if kind == "mixed" else "0"), row
            assert row["Output Shapes"].strip('"') == output_shapes, row
            assert row["Output Data Types"] == output_dtypes, row
            if kind == "vector":
                assert row["Input Shapes"].strip('"') == "64,64;64,64", row

    trace_files = glob.glob(os.path.join(output_dir, "**", "trace_view.json"), recursive=True)
    if not trace_files:
        trace_files = glob.glob(os.path.join(output_dir, "**", "trace_result.json"), recursive=True)
    for _ in range(20):
        if trace_files:
            break
        time.sleep(0.5)
        trace_files = glob.glob(os.path.join(output_dir, "**", "trace_view.json"), recursive=True)
        if not trace_files:
            trace_files = glob.glob(os.path.join(output_dir, "**", "trace_result.json"), recursive=True)
    assert trace_files, f"No trace_view.json or trace_result.json under {output_dir}"
    for trace_file in trace_files:
        with open(trace_file, "r", encoding="utf-8") as stream:
            trace = json.load(stream)
        events = trace.get("traceEvents", []) if isinstance(trace, dict) else trace
        assert isinstance(events, list) and events, f"No trace events in {trace_file}"
        logging.info("Trace file: %s, events=%d", trace_file, len(events))


@pytest.mark.soc("950")
@pytest.mark.parametrize("warm_kernel", [False, True], ids=["loaded_during_collection", "loaded_before_collection"])
def test_tensor_reports_stop_with_collection(tmp_path, monkeypatch, warm_kernel):
    """Count reporting before/during/after collection, including loading a library while profiling."""
    _check_npu()
    monkeypatch.setenv("ASCEND_WORK_PATH", str(tmp_path))
    jit = importlib.import_module("pypto_pro.runtime.jit")
    generate = jit._generate_caller_cpp

    def instrument(*args, **kwargs):
        source = generate(*args, **kwargs)
        assert "aclprofRange" not in source
        source = source.replace(
            "pypto::ReportCaptureTensorInfo(",
            "pyptoTestReports += pyptoProfBegin != 0;\n"
            "        pyptoTestCaches += pyptoCaptureOn;\n"
            "        pypto::ReportCaptureTensorInfo(",
        )
        return (
            "static unsigned pyptoTestReports = 0, pyptoTestCaches = 0;\n"
            'extern "C" unsigned pypto_test_reports(bool live) { return live ? pyptoTestReports : pyptoTestCaches; }\n'
            + source
        )

    monkeypatch.setattr(jit, "_generate_caller_cpp", instrument)

    @pl.jit()
    def switch_kernel(out: pl.Tensor[[32], pl.DT_INT32, pl.Output]):
        with pl.section_cube():
            out[0] = 3

    out = torch.zeros(32, dtype=torch.int32, device=ST_DEVICE)
    expected = torch.zeros_like(out)
    expected[0] = 3

    def launch():
        out.zero_()
        switch_kernel(out)
        torch.npu.synchronize()
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def counts():
        compiled, = switch_kernel._compiled_by_signature.values()
        counter = compiled.entry.lib.pypto_test_reports
        counter.argtypes = [ctypes.c_bool]
        counter.restype = ctypes.c_uint
        return counter(True), counter(False)

    if warm_kernel:
        launch()
        assert counts() == (0, 0)

    for session in range(2):
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            experimental_config=_build_experimental_config(),
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(tmp_path / f"profile_{session}")),
        ) as prof:
            launch()
            assert counts() == (session + 1, 0)
            prof.step()
        launch()
        assert counts() == (session + 1, 0), "Tensor reports continued after collection stopped"


@pytest.mark.soc("950")
@pytest.mark.skip(reason="parallel read/write")
def test_graph_replay_preserves_each_launch_shape(tmp_path):
    """Two tasks of one dynamic kernel retain distinct shapes across profiled replays."""
    _check_npu()

    @pl.jit()
    def shape_kernel(out: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32, pl.Output]):
        with pl.section_cube():
            out[0] = out.shape[0]

    outputs = [torch.zeros(n, dtype=torch.int32, device=ST_DEVICE) for n in (32, 64)]
    for out in outputs:
        shape_kernel(out)
    torch.npu.synchronize()
    output_dir = str(tmp_path / "profiling_output")
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=False,
        experimental_config=_build_experimental_config(),
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=2, repeat=1, skip_first=2),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(output_dir),
    ) as prof:
        for step in range(4):
            for out in outputs:
                out.zero_()
            torch.npu.synchronize()
            if step == 0:
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    for out in outputs:
                        shape_kernel(out)
            graph.replay()
            torch.npu.synchronize()
            for out in outputs:
                expected = torch.zeros_like(out)
                expected[0] = out.numel()
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
            prof.step()
    for matches in _kernel_rows(output_dir, shape_kernel.__name__):
        assert sorted(row["Output Shapes"].strip('"') for row in matches) == ["32", "32", "64", "64"]
        assert {row["Output Data Types"] for row in matches} == {"INT32"}
        assert {row["Output Formats"] for row in matches} == {"ND"}
        assert len({(row["Stream ID"], row["Task ID"]) for row in matches}) == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
