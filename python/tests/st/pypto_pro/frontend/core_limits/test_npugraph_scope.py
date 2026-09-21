# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise real npugraph_ex scope lowering around an opaque PyPTO custom operator."""

import pytest
from test_core_limits import DEVICE_ID, KERNELS, SIZE, _check
import torch


@pytest.fixture(autouse=True)
def _isolated_graph_cache():
    """Discovery captures graphs with JIT launches stubbed; phase 3 must capture real kernels.

    Dynamo caches by the model's code, which is reused across the discovery and
    execution phases. Clear graph caches even when discovery's accuracy assertion
    fails, while leaving PyPTO's separately cached compiled kernels available.
    """
    torch.compiler.reset()
    yield
    torch.compiler.reset()


@torch.library.custom_op("pypto_core_limits::probe", mutates_args=())
def _probe(x: torch.Tensor, kind: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = tuple(torch.zeros(SIZE, dtype=torch.int32, device=x.device) for _ in range(3))
    # Direct call: the auto sentinel resolves to the scope's full budget at capture.
    KERNELS[kind](*outputs)
    return outputs


@_probe.register_fake
def _probe_fake(x, kind):
    return tuple(torch.empty(SIZE, dtype=torch.int32, device=x.device) for _ in range(3))


@pytest.mark.soc("950")
@pytest.mark.parametrize("kind", [1, 2], ids=["vector", "mixed"])
@pytest.mark.parametrize("explicit_stream", [False, True], ids=["implicit_scope", "explicit_scope"])
def test_npugraph_ex_scope(kind, explicit_stream):
    """Dynamo markers must set stream resources at actual capture, not only during tracing or warmup."""
    torch.npu.set_device(DEVICE_ID)
    x = torch.zeros(1, device=f"npu:{DEVICE_ID}")
    # Warm the JIT outside capture while preserving the custom-op boundary in the FX graph.
    _probe(x, kind)
    torch.npu.synchronize()

    def model(x):
        scope_stream = torch.npu.current_stream() if explicit_stream else None
        with torch.npu.npugraph_ex.scope.limit_core_num(4, 6, stream=scope_stream):
            first = _probe(x, kind)
        with torch.npu.npugraph_ex.scope.limit_core_num(2, 2, stream=scope_stream):
            second = _probe(x, kind)
        return first, second

    compiled = torch.compile(model, backend="npugraph_ex", dynamic=False, fullgraph=True)
    for _ in range(3):
        first, second = compiled(x)
        torch.npu.synchronize()
        _check(first, kind, 6 if kind == 1 else 3)
        _check(second, kind, 2 if kind == 1 else 1)
