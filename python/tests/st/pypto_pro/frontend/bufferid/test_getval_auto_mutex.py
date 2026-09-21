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

"""getval/setval on a buffer-managed (NBuffer) Vec slot under auto_mutex.

Scenario: a Vector op (exp, PIPE_V) writes a managed slot, then getval reads a
scalar from it (PIPE_S) and setval writes it back (PIPE_S). Correctness depends
on a V->S ordering: getval must observe the exp() output, not the stale pre-exp
value. With getval/setval registered on PIPE_S under the "block." namespace, the
auto_mutex buffer token (get_buf/rls_buf on PIPE_S) serialises V<->S on the slot;
ShouldSkipVPipeMutex keeps the producer's V token because the slot is now shared
by V and S.

Run standalone (e.g. under cannsim):  python3 <thisfile> --sim
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


@pl.jit(arch="a5", auto_mutex=True)
def getval_setval_am_kernel(a: pl.Tensor[[64, 128], pl.DT_FP16]):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tt, addrs=[0x00000], mutex_ids=[20])
    with pl.section_vector():
        t = a_db.next()
        pl.load(t, a, [0, 0])
        pl.exp(t, t)  # V writes buf 20
        v00 = t[0, 0]  # S reads t[0,0] == exp(a[0,0])   (needs V->S)
        v63 = t[0, 63]  # S reads t[0,63] == exp(a[0,63])
        t[1, 0] = v00  # S writes t[1,0]  = v00
        t[1, 1] = v63  # S writes t[1,1]  = v63
        pl.store(a, t, [0, 0])  # MTE3 consumer


@pl.jit(arch="a5", auto_mutex=True)
def getval_while_condition_am_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
    out: pl.Tensor[[1], pl.DT_INT64],
):
    """Read a managed tile in the while condition and publish the exit index."""
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tt, addrs=[0x00000], mutex_ids=[20])
    with pl.section_vector():
        t = a_db.next()
        pl.load(t, a, [0, 0])
        i: pl.DT_INT64 = 0
        while t[0, i] > 0:
            i = i + 1
            if i >= 8:
                break
        out[0] = i


def _run_and_measure(a: torch.Tensor):
    """Launch kernel; return (max_abs_diff_vs_golden, details dict)."""
    a_cpu = a.detach().to("cpu", torch.float32)
    getval_setval_am_kernel[None, 1](a)
    torch.npu.synchronize()
    out = a.detach().to("cpu", torch.float32)

    exp_ref = torch.exp(a_cpu)  # golden for the exp() result region
    # golden: whole tile is exp(input); plus t[1,0]=exp(a[0,0]), t[1,1]=exp(a[0,63])
    golden = exp_ref.clone()
    golden[1, 0] = exp_ref[0, 0]
    golden[1, 1] = exp_ref[0, 63]

    diff = (out - golden).abs()
    details = {
        "a00_orig": a_cpu[0, 0].item(),
        "exp_a00": exp_ref[0, 0].item(),
        "got_t10": out[1, 0].item(),  # getval->setval result
        "got_t11": out[1, 1].item(),
        "exp_a63": exp_ref[0, 63].item(),
        "max_diff": diff.max().item(),
    }
    return details["max_diff"], details


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_getval_setval_auto_mutex():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.rand(64, 128, device=device, dtype=torch.float16) * 2.0 - 1.0
    max_diff, d = _run_and_measure(a)
    logging.info("getval/setval auto_mutex: %s", d)
    # getval must observe exp() output, not the stale loaded value
    assert abs(d["got_t10"] - d["exp_a00"]) < 3e-3, f"stale read: t[1,0]={d['got_t10']} expect exp(a00)={d['exp_a00']}"
    assert abs(d["got_t11"] - d["exp_a63"]) < 3e-3, f"stale read: t[1,1]={d['got_t11']} expect exp(a63)={d['exp_a63']}"
    assert max_diff < 3e-3, f"max|diff|={max_diff} too large"


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_getval_while_condition_auto_mutex():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    a = torch.zeros(64, 128, device=device, dtype=torch.float16)
    a[0, :3] = 1.0
    out = torch.zeros(1, device=device, dtype=torch.int64)

    getval_while_condition_am_kernel(a, out)
    torch.npu.synchronize()

    assert out.item() == 3


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.npu.set_device(ST_DEVICE)
    torch.manual_seed(0)
    a = torch.rand(64, 128, device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    max_diff, d = _run_and_measure(a)
    logging.info(
        f"a_orig[0,0]={d['a00_orig']:.5f}  exp(a00)={d['exp_a00']:.5f}  "
        f"t[1,0]={d['got_t10']:.5f}  |  exp(a63)={d['exp_a63']:.5f}  t[1,1]={d['got_t11']:.5f}"
    )
    logging.info("max|diff| vs golden = %.7f", max_diff)
    stale = abs(d["got_t10"] - d["a00_orig"]) < 1e-3 and abs(d["exp_a00"] - d["a00_orig"]) > 1e-3
    ok = abs(d["got_t10"] - d["exp_a00"]) < 3e-3 and abs(d["got_t11"] - d["exp_a63"]) < 3e-3 and max_diff < 3e-3
    logging.info("RESULT: %s   (stale-read detected: %s)", "PASS" if ok else "FAIL", stale)
