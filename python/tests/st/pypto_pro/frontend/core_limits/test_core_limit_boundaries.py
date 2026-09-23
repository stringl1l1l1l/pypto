# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED.
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Boundary and state-transition tests for core limits, using device-written worker counts."""

import pytest
from test_core_limits import DEVICE_ID, KERNELS, _check, _outputs
import torch

pytestmark = pytest.mark.soc("950")


@pytest.fixture
def stream():
    torch.npu.set_device(DEVICE_ID)
    s = torch.npu.Stream()
    try:
        yield s
    finally:
        s.synchronize()
        torch.npu.reset_stream_limit(s)


@pytest.mark.parametrize("kind", range(3), ids=["cube", "vector", "mixed"])
@pytest.mark.parametrize("requested", [1, 3])
def test_limit_never_increases_requested_blocks(stream, kind, requested):
    """A request within the budget launches exactly the requested count; available resources are a ceiling."""
    outputs = _outputs()
    torch.npu.synchronize()
    torch.npu.set_stream_limit(stream, 8, 16)
    KERNELS[kind][stream, requested](*outputs)
    stream.synchronize()
    _check(outputs, kind, requested)


@pytest.mark.parametrize("kind", range(3), ids=["cube", "vector", "mixed"])
@pytest.mark.parametrize("requested", [1 << 32, (1 << 32) + 1, 1 << 100])
def test_large_request_is_rejected_not_wrapped(stream, kind, requested):
    """Saturation must keep a huge request positive and rejected, never wrapped onto the auto sentinel."""
    outputs = _outputs()
    torch.npu.synchronize()
    torch.npu.set_stream_limit(stream, 4, 6)
    with pytest.raises(RuntimeError, match="Kernel launch failed with error code"):
        KERNELS[kind][stream, requested](*outputs)


@pytest.mark.parametrize("kind", range(3), ids=["cube", "vector", "mixed"])
def test_direct_call_uses_the_scope_budget(stream, kind):
    """The direct-call path must use the current stream and resolve to its full budget."""
    outputs = _outputs()
    torch.npu.synchronize()
    with torch.npu.stream(stream), torch.npu.npugraph_ex.scope.limit_core_num(4, 8):
        KERNELS[kind](*outputs)
    stream.synchronize()
    _check(outputs, kind, (4, 8, 4)[kind])


@pytest.mark.parametrize("kind", range(3), ids=["cube", "vector", "mixed"])
@pytest.mark.parametrize("update,expected", [({"cube_num": 3}, (3, 16, 3)), ({"vector_num": 5}, (8, 5, 2))])
def test_partial_stream_update(stream, kind, update, expected):
    """Updating only one resource must preserve the other engine's independently configured budget."""
    outputs = _outputs()
    torch.npu.synchronize()
    torch.npu.set_stream_limit(stream, 8, 16)
    torch.npu.set_stream_limit(stream, **update)
    with torch.npu.stream(stream):
        KERNELS[kind](*outputs)
    stream.synchronize()
    _check(outputs, kind, expected[kind])


@pytest.mark.parametrize("kind", range(3), ids=["cube", "vector", "mixed"])
def test_cached_kernel_alternates_streams(stream, kind):
    """One compiled kernel must follow the current stream rather than the stream used on its first call."""
    other = torch.npu.Stream()
    torch.npu.set_stream_limit(stream, 3, 8)
    torch.npu.set_stream_limit(other, 6, 4)
    try:
        for target, blocks in [(stream, (3, 8, 3)), (other, (6, 4, 2)), (stream, (3, 8, 3))]:
            outputs = _outputs()
            torch.npu.synchronize()
            with torch.npu.stream(target):
                KERNELS[kind](*outputs)
            target.synchronize()
            _check(outputs, kind, blocks[kind])
    finally:
        other.synchronize()
        torch.npu.reset_stream_limit(other)


@pytest.mark.parametrize("kind,limits,blocks", [(0, (4, 1), 4), (1, (1, 7), 7)])
def test_unused_engine_does_not_constrain_launch(stream, kind, limits, blocks):
    """A vector budget too small for a mixed pair must still allow a cube-only kernel, and vice versa."""
    outputs = _outputs()
    torch.npu.synchronize()
    torch.npu.set_stream_limit(stream, *limits)
    with torch.npu.stream(stream):
        KERNELS[kind](*outputs)
    stream.synchronize()
    _check(outputs, kind, blocks)


def test_insufficient_mixed_budget_rejects_without_device_writes(stream):
    """Reject before launching any synchronized participants, then allow reuse after the budget is corrected."""
    outputs = _outputs()
    torch.npu.synchronize()
    launcher = KERNELS[2][stream, 2]
    torch.npu.set_stream_limit(stream, 4, 1)
    with pytest.raises(
        RuntimeError,
        match=f"Kernel launch failed with error code {-0x200000001}: "
        "the stream's vector budget of 1 core\\(s\\) cannot host one block of 2 vector core\\(s\\)",
    ):
        launcher(*outputs)
    stream.synchronize()
    assert all(torch.count_nonzero(t.cpu()).item() == 0 for t in outputs)
    torch.npu.set_stream_limit(stream, 4, 4)
    launcher(*outputs)
    stream.synchronize()
    _check(outputs, 2, 2)


@pytest.mark.parametrize("kind", range(3), ids=["cube", "vector", "mixed"])
def test_raw_graph_recapture_observes_new_limits(stream, kind):
    """Raw replay retains the old count, while recapturing the same compiled kernel reads the new stream budget."""
    outputs = _outputs()
    torch.npu.synchronize()
    with torch.npu.stream(stream):
        KERNELS[kind](*outputs)
    stream.synchronize()
    graphs = []
    for limits, expected in [((4, 8), (4, 8, 4)), ((2, 6), (2, 6, 2))]:
        for tensor in outputs:
            tensor.zero_()
        torch.npu.synchronize()
        torch.npu.set_stream_limit(stream, *limits)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            with torch.npu.stream(stream):
                KERNELS[kind](*outputs)
        graphs.append(graph)
        graph.replay()
        torch.npu.synchronize()
        _check(outputs, kind, expected[kind])
    for tensor in outputs:
        tensor.zero_()
    torch.npu.synchronize()
    graphs[0].replay()
    torch.npu.synchronize()
    _check(outputs, kind, (4, 8, 4)[kind])
