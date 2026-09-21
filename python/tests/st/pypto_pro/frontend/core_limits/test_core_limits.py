# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED.
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Observe device block counts and complete work coverage under all torch_npu limit scopes.

Every worker writes a separate 128-byte cache line, avoiding scalar-store false sharing.
The work array has more tasks than launched workers, so blindly dropping workers from
fixed host tiling cannot pass these accuracy checks. Count arrays also distinguish a
correct numerical result produced with too many cores from a compliant launch.

An explicit block_dim must fit the stream's budget; a request above it is rejected.
A direct call carries the auto sentinel and resolves to the stream's full budget.
"""

import os
from pathlib import Path
import subprocess
import sys

import pypto_pro.language as pl
import pytest
import torch
import torch_npu  # noqa: F401 - installs torch.npu

SIZE = 4096
TASKS = 101
DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
pytestmark = pytest.mark.soc("950")


@pl.jit()
def cube_probe(
    cube: pl.Tensor[[SIZE], pl.DT_INT32],
    vector: pl.Tensor[[SIZE], pl.DT_INT32],
    work: pl.Tensor[[SIZE], pl.DT_INT32],
):
    with pl.section_cube():
        i = pl.get_block_idx() * 32
        cube[i] = pl.get_block_num()
        cube[i + 1] = pl.get_subblock_num()
        for task in pl.range(pl.get_block_idx(), TASKS, pl.get_block_num()):
            work[task * 32] = task * 3 + 7


@pl.jit()
def vector_probe(
    cube: pl.Tensor[[SIZE], pl.DT_INT32],
    vector: pl.Tensor[[SIZE], pl.DT_INT32],
    work: pl.Tensor[[SIZE], pl.DT_INT32],
):
    with pl.section_vector():
        i = pl.get_block_idx() * 32
        vector[i] = pl.get_block_num()
        vector[i + 1] = pl.get_subblock_num()
        for task in pl.range(pl.get_block_idx(), TASKS, pl.get_block_num()):
            work[task * 32] = task * 3 + 7


@pl.jit()
def mixed_probe(
    cube: pl.Tensor[[SIZE], pl.DT_INT32],
    vector: pl.Tensor[[SIZE], pl.DT_INT32],
    work: pl.Tensor[[SIZE], pl.DT_INT32],
):
    with pl.section_cube():
        i = pl.get_block_idx() * 32
        cube[i] = pl.get_block_num()
        cube[i + 1] = pl.get_subblock_num()
    with pl.section_vector():
        i = pl.get_block_idx() * 32
        vector[i] = pl.get_block_num()
        vector[i + 1] = pl.get_subblock_num()
        workers = pl.get_block_num() * pl.get_subblock_num()
        for task in pl.range(pl.get_block_idx(), TASKS, workers):
            work[task * 32] = task * 3 + 7


KERNELS = (cube_probe, vector_probe, mixed_probe)


@pytest.fixture
def stream():
    torch.npu.set_device(DEVICE_ID)
    s = torch.npu.Stream()
    try:
        yield s
    finally:
        s.synchronize()
        torch.npu.reset_stream_limit(s)


def _outputs():
    return tuple(torch.zeros(SIZE, dtype=torch.int32, device=f"npu:{DEVICE_ID}") for _ in range(3))


def _check(outputs, kind, blocks):
    expected = [torch.zeros(SIZE, dtype=torch.int32).reshape(-1, 32) for _ in range(3)]
    if kind != 1:
        expected[0][:blocks, 0] = blocks
        expected[0][:blocks, 1] = 1
    if kind != 0:
        ratio = 2 if kind == 2 else 1
        expected[1][:blocks * ratio, 0] = blocks
        expected[1][:blocks * ratio, 1] = ratio
    expected[2][:TASKS, 0] = torch.arange(TASKS, dtype=torch.int32) * 3 + 7
    for actual, reference in zip(outputs, expected):
        actual = actual.cpu().reshape(-1, 32)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    print(f"kind={kind} blocks={blocks}: count/work max_abs_error=0, exact_match=100%")

@pytest.mark.parametrize("kind", range(3), ids=["cube", "vector", "mixed"])
@pytest.mark.parametrize("limits", [(4, 10), (8, 5)])
def test_stream_limits(stream, kind, limits):
    """Asymmetric limits catch using cube counts for vectors or ignoring mixed vector demand."""
    outputs = _outputs()
    torch.npu.synchronize()
    torch.npu.set_stream_limit(stream, cube_num=limits[0], vector_num=limits[1])
    blocks = (limits[0], limits[1], min(limits[0], limits[1] // 2))[kind]
    with torch.npu.stream(stream):
        KERNELS[kind](*outputs)
    stream.synchronize()
    _check(outputs, kind, blocks)


@pytest.mark.parametrize("kind", range(3), ids=["cube", "vector", "mixed"])
@pytest.mark.parametrize("use_current", [False, True], ids=["explicit_stream", "current_stream"])
def test_bracket_request_within_limits_launches_exactly(stream, kind, use_current):
    """An explicit request that fits the budget launches exactly that many blocks on the named stream."""
    outputs = _outputs()
    torch.npu.synchronize()
    torch.npu.set_stream_limit(stream, cube_num=4, vector_num=10)
    with torch.npu.stream(stream):
        KERNELS[kind][None if use_current else stream, 4](*outputs)
    stream.synchronize()
    _check(outputs, kind, 4)


@pytest.mark.parametrize("kind", range(3), ids=["cube", "vector", "mixed"])
def test_request_above_stream_limit_is_rejected(stream, kind):
    """A request the stream cannot host fails the launch."""
    outputs = _outputs()
    torch.npu.synchronize()
    torch.npu.set_stream_limit(stream, 4, 6)
    with pytest.raises(RuntimeError, match="Kernel launch failed with error code"):
        KERNELS[kind][stream, 12](*outputs)


@pytest.mark.parametrize("kind", range(3), ids=["cube", "vector", "mixed"])
def test_nested_scope_and_cached_kernel(stream, kind):
    """Direct launches must honor nesting, exception restoration and subsequent resets."""
    torch.npu.set_stream_limit(stream, 8, 10)

    def run(blocks):
        outputs = _outputs()
        torch.npu.synchronize()
        with torch.npu.stream(stream):
            KERNELS[kind](*outputs)
        stream.synchronize()
        _check(outputs, kind, blocks)

    run((8, 10, 5)[kind])
    with torch.npu.npugraph_ex.scope.limit_core_num(4, 6, stream=stream):
        run((4, 6, 3)[kind])
        with pytest.raises(RuntimeError, match="scope exit"):
            with torch.npu.npugraph_ex.scope.limit_core_num(2, 2, stream=stream):
                run((2, 2, 1)[kind])
                raise RuntimeError("scope exit")
        run((4, 6, 3)[kind])
    run((8, 10, 5)[kind])
    torch.npu.reset_stream_limit(stream)
    inherited = torch.npu.get_stream_limit(stream)
    blocks = (inherited["cube_core_num"], inherited["vector_core_num"],
              min(inherited["cube_core_num"], inherited["vector_core_num"] // 2))[kind]
    run(blocks)


def test_explicit_scope_does_not_limit_other_stream(stream):
    """An explicit scope targets its stream even when a different stream is current."""
    other = torch.npu.Stream()
    torch.npu.set_stream_limit(other, 6, 8)
    outputs = _outputs()
    torch.npu.synchronize()
    try:
        with torch.npu.stream(other), torch.npu.npugraph_ex.scope.limit_core_num(2, 2, stream=stream):
            vector_probe(*outputs)
        other.synchronize()
        _check(outputs, 1, 8)
    finally:
        other.synchronize()
        torch.npu.reset_stream_limit(other)


@pytest.mark.parametrize("kind", range(3), ids=["cube", "vector", "mixed"])
def test_capture_scopes_and_replay(stream, kind):
    """Each graph node must retain its own capture-time count after the scopes restore limits."""
    outputs = [_outputs(), _outputs()]
    torch.npu.synchronize()
    # Compile and bind before capture; warmup results must not mask missing graph writes.
    with torch.npu.stream(stream):
        KERNELS[kind](*outputs[0])
    stream.synchronize()
    for group in outputs:
        for tensor in group:
            tensor.zero_()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        with torch.npu.npugraph_ex.scope.limit_core_num(4, 6):
            KERNELS[kind](*outputs[0])
        with torch.npu.npugraph_ex.scope.limit_core_num(2, 2, stream=stream):
            KERNELS[kind](*outputs[1])
    for _ in range(2):
        graph.replay()
        torch.npu.synchronize()
        _check(outputs[0], kind, (4, 6, 3)[kind])
        _check(outputs[1], kind, (2, 2, 1)[kind])
        for group in outputs:
            for tensor in group:
                tensor.zero_()
            torch.npu.synchronize()


@pytest.mark.skip_jit_discovery(reason="Device-limit probe compiles and runs in its own process")
def test_device_limit_in_subprocess():
    """Device setters are once-per-process; isolate them and prove stream override can exceed the device default."""
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "device"],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)


def _device_probe():
    torch.npu.set_device(DEVICE_ID)
    stream = torch.npu.Stream()
    torch.npu.set_device_limit(DEVICE_ID, cube_num=4, vector_num=6)
    for kind in range(3):
        outputs = _outputs()
        torch.npu.synchronize()
        with torch.npu.stream(stream):
            KERNELS[kind](*outputs)
        stream.synchronize()
        _check(outputs, kind, (4, 6, 3)[kind])
    torch.npu.set_stream_limit(stream, cube_num=8, vector_num=10)
    outputs = _outputs()
    torch.npu.synchronize()
    with torch.npu.stream(stream):
        mixed_probe(*outputs)
    stream.synchronize()
    _check(outputs, 2, 5)
    torch.npu.reset_stream_limit(stream)
    outputs = _outputs()
    torch.npu.synchronize()
    with torch.npu.stream(stream):
        mixed_probe(*outputs)
    stream.synchronize()
    _check(outputs, 2, 3)


if __name__ == "__main__":
    _device_probe()
