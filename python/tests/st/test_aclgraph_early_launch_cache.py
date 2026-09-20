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
"""aclgraph + early launch + host ctrl-flow cache slot reuse.

Covers the restore race: same PyPTO kernel is launched more times than the
device ring (4) and ping-pong depth (2) inside one captured graph, then the
graph is replayed without host re-launch. A second kernel is interleaved so
per-operator event isolation is also exercised.
"""

import os

import torch
import torch_npu

import pypto

LAYER_COUNT = 6
REPLAY_COUNT = 8
TILING = 32
SHAPE = (64, 64)

@pypto.frontend.jit()
def add_kernel(
    a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    c: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    tiling=TILING,
):
    pypto.set_vec_tile_shapes(tiling, tiling)
    c.move(pypto.add(a, b))


@pypto.frontend.jit()
def mul_kernel(
    a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    c: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    tiling=TILING,
):
    pypto.set_vec_tile_shapes(tiling, tiling)
    c.move(pypto.mul(a, b))


class LayerStack(torch.nn.Module):
    def __init__(self, shape, layers, device):
        super().__init__()
        self.layers = layers
        self.add_bufs = [torch.zeros(shape, dtype=torch.int32, device=device) for _ in range(layers)]
        self.mul_bufs = [torch.zeros(shape, dtype=torch.int32, device=device) for _ in range(layers)]
        self.ones = torch.ones(shape, dtype=torch.int32, device=device)

    def forward(self, data1, data2):
        x = data1
        for i in range(self.layers):
            add_kernel(x, data2, self.add_bufs[i])
            x = self.add_bufs[i]
            mul_kernel(x, self.ones, self.mul_bufs[i])
            x = self.mul_bufs[i]
        return x


@pypto.options(pass_options={"enable_slice": True})
def test_aclgraph_early_launch_host_cache_slot_reuse():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"

    data1 = torch.ones(SHAPE, dtype=torch.int32, device=device)
    data2 = torch.full(SHAPE, 2, dtype=torch.int32, device=device)
    golden = data1 + data2 * LAYER_COUNT

    model = LayerStack(SHAPE, LAYER_COUNT, device)
    # Warmup compiles kernels and builds host ctrl-flow cache before capture.
    _ = model(data1, data2)
    torch_npu.npu.synchronize()

    assert not torch_npu.npu.is_current_stream_capturing()
    s = torch.npu.Stream()
    with torch.npu.stream(s):
        g = torch_npu.npu.NPUGraph()
        torch_npu.npu.empty_cache()
        g.capture_begin()
        npu_out = model(data1, data2)
        assert torch_npu.npu.is_current_stream_capturing()
        g.capture_end()
    torch_npu.npu.current_stream().wait_stream(s)

    for _ in range(REPLAY_COUNT):
        g.replay()
    torch_npu.npu.current_stream().synchronize()
    g.reset()

    assert torch.equal(npu_out.cpu(), golden.cpu())

    # Capture -> eager must restart the ring; wait/record must not mix with the graph.
    eager_out = model(data1, data2)
    torch_npu.npu.synchronize()
    assert torch.equal(eager_out.cpu(), golden.cpu())


if __name__ == "__main__":
    test_aclgraph_early_launch_host_cache_slot_reuse()
    print("=========== pass ==========")
