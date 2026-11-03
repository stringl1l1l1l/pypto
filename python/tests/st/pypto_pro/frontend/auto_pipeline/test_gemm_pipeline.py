# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L1 handover into GEMM: both inputs, preload, slot reuse and drain."""

import ast
import os
import re

import pypto_pro.language as pl
from pypto_pro.runtime.platform import NpuArch
import pytest
import torch

import pypto  # noqa: F401

pytestmark = pytest.mark.soc("950")
DEVICE = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
TM, TN, TK = 64, 64, 64


@pl.pipeline.stage
def prepare_a(i, src, shared, ub, nz):
    tile = ub.next()
    pl.load(tile, src, [i * TM, 0])
    pl.muls(tile, tile, 2.0)
    packed = nz.current()
    pl.move(packed, tile)
    target = shared.next()
    pl.move(target, packed)


@pl.pipeline.stage
def prepare_b(i, src, shared, ub, nz):
    tile = ub.next()
    pl.load(tile, src, [i * TK, 0])
    pl.muls(tile, tile, 4.0)
    packed = nz.current()
    pl.move(packed, tile)
    target = shared.next()
    pl.move(target, packed)


def _pipeline_kernel(preload, *, splitk=0, manual=False):
    stage_k = splitk or TK

    def product(z, x, y, la, lb):
        if manual:
            if splitk:
                for ki in pl.range(0, TK, splitk):
                    left = la.next()
                    right = lb.next()
                    pl.move(left, x, offset=[0, ki])
                    pl.move(right, y, offset=[ki, 0])
                    if ki == 0:
                        pl.matmul(z, left, right)
                    else:
                        pl.matmul_acc(z, z, left, right)
            else:
                left = la.next()
                right = lb.next()
                pl.move(left, x)
                pl.move(right, y)
                pl.matmul(z, left, right)
        elif splitk:
            pl.gemm(z, x, y, l0a=la, l0b=lb, splitk=splitk)
        else:
            pl.gemm(z, x, y, l0a=la, l0b=lb)

    @pl.pipeline.stage
    def multiply_a(i, src, out, shared, local, la, lb, gc):
        x = shared.next()
        y = local.current()
        z = gc.current()
        pl.load(y, src, [i * TK, 0])
        product(z, x, y, la, lb)
        pl.store(out, z, [i * TM, 0])

    @pl.pipeline.stage
    def multiply_b(i, src, out, shared, local, la, lb, gc):
        x = local.current()
        y = shared.next()
        z = gc.current()
        pl.load(x, src, [i * TM, 0])
        product(z, x, y, la, lb)
        pl.store(out, z, [out.shape[0] // 2 + i * TM, 0])

    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=preload),
            name=f"gemm_pipeline_p{preload}_k{splitk}_manual{int(manual)}")
    def kernel(a: pl.Tensor[[pl.DYNAMIC, TK], pl.DT_FP16],
               b: pl.Tensor[[pl.DYNAMIC, TN], pl.DT_FP16],
               out: pl.Tensor[[pl.DYNAMIC, TN], pl.DT_FP32]):
        shared_a = pl.make_tile_group(
            type=pl.TileType(shape=[TM, TK], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=0, mutex_ids="auto", depth=3, fwd_ids=[0, 1, 2], bwd_ids=[3, 4, 5],
        )
        shared_b = pl.make_tile_group(
            type=pl.TileType(shape=[TK, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=0x10000, mutex_ids="auto", depth=3, fwd_ids=[6, 7, 8], bwd_ids=[9, 10, 11],
        )
        with pl.section_cube():
            local_a = pl.make_tile_group(
                type=pl.TileType(shape=[TM, TK], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
                addrs=0x20000, mutex_ids="auto", depth=1,
            )
            local_b = pl.make_tile_group(
                type=pl.TileType(shape=[TK, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
                addrs=0x24000, mutex_ids="auto", depth=1,
            )
            la = pl.make_tile_group(
                type=pl.TileType(shape=[TM, stage_k], dtype=pl.DT_FP16,
                                 target_memory=pl.MemorySpace.Left, compact=1),
                addrs=0, mutex_ids="auto", depth=2,
            )
            lb = pl.make_tile_group(
                type=pl.TileType(shape=[stage_k, TN], dtype=pl.DT_FP16,
                                 target_memory=pl.MemorySpace.Right, compact=1),
                addrs=0, mutex_ids="auto", depth=2,
            )
            gc = pl.make_tile_group(
                type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
                addrs=0, mutex_ids="auto", depth=1,
            )
        with pl.section_vector():
            ub_a = pl.make_tile_group(
                type=pl.TileType(shape=[TM, TK], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
                addrs=0, mutex_ids="auto", depth=1,
            )
            ub_b = pl.make_tile_group(
                type=pl.TileType(shape=[TK, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
                addrs=0x4000, mutex_ids="auto", depth=1,
            )
            ub_nz = pl.make_tile_group(
                type=pl.TileType(shape=[TM, TK], dtype=pl.DT_FP16,
                                 target_memory=pl.MemorySpace.Vec, layout=pl.NZ, compact=1),
                addrs=0x8000, mutex_ids="auto", depth=1,
            )
        for i in pl.range(0, a.shape[0] // TM):
            with pl.section_vector():
                prepare_a(i, a, shared_a, ub_a, ub_nz)
            with pl.section_cube():
                multiply_a(i, b, out, shared_a, local_b, la, lb, gc)
            with pl.section_vector():
                prepare_b(i, b, shared_b, ub_b, ub_nz)
            with pl.section_cube():
                multiply_b(i, a, out, shared_b, local_a, la, lb, gc)
    return kernel


_CASES = [(0, 1, 0), (1, 2, 0), (2, 7, 0), (2, 7, 32)]
_IDS = ["serial", "short_preload", "ring_wrap", "split_k"]


@pytest.mark.parametrize("preload,tasks,splitk", _CASES, ids=_IDS)
def test_gemm_pipeline_matches_reference(preload, tasks, splitk):
    torch.npu.set_device(DEVICE)
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip("GEMM pipeline regressions require Ascend950")
    generator = torch.Generator().manual_seed(71)
    a = torch.randn((tasks, TM, TK), generator=generator, dtype=torch.float16)
    b = torch.randn((tasks, TK, TN), generator=generator, dtype=torch.float16)
    out = torch.full((2 * tasks * TM, TN), 7.0, device=DEVICE)
    _pipeline_kernel(preload, splitk=splitk)(
        a.reshape(-1, TK).to(DEVICE), b.reshape(-1, TN).to(DEVICE), out,
    )
    torch.npu.synchronize()
    golden = a.float() @ b.float()
    expected = torch.cat((golden.reshape(-1, TN) * 2, golden.reshape(-1, TN) * 4))
    torch.testing.assert_close(out.cpu(), expected, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("preload,splitk", [(0, 0), (1, 0), (2, 0), (2, 32)])
def test_gemm_pipeline_sync_matches_native_expansion(preload, splitk, tmp_path):
    from pypto_pro.runtime.jit import _codegen

    def source(manual):
        kernel = _pipeline_kernel(preload, splitk=splitk, manual=manual)
        definition = kernel.to_kernel_def()
        cpp = _codegen(definition, NpuArch.DAV_3510, clean_up=False,
                       out_dir=str(tmp_path / str(manual))).content
        sites = [ast.dump(node) for node in ast.walk(definition._func_def)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and node.func.attr in ("wait_cross_core", "set_cross_core")]
        return cpp, sites

    (sugar, sugar_sites), (native, native_sites) = source(False), source(True)
    assert sugar_sites == native_sites and sugar_sites  # Includes event-ID selectors and drain.
    calls = r"(wait_intra_block|set_intra_block)\((PIPE_\w+),"
    assert re.findall(calls, sugar) == re.findall(calls, native)
    assert {pipe for _, pipe in re.findall(calls, sugar)} == {"PIPE_MTE1", "PIPE_MTE3"}
