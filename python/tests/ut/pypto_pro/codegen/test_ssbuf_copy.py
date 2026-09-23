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
from __future__ import annotations

from dataclasses import dataclass
import logging

import pypto_pro.language as pl


def _codegen_result(kernel):
    from pypto_pro.runtime.jit import _assemble_cv_source, _parse_and_codegen_targets
    from pypto_pro.runtime.kernel import KernelDef

    kernel_def = kernel if isinstance(kernel, KernelDef) else kernel.to_kernel_def()
    cube, vector = _parse_and_codegen_targets(kernel_def, "3510", "")
    return _assemble_cv_source(cube, vector)


def _compile_to_cce(kernel) -> str:
    return _codegen_result(kernel).content


@pl.jit()
def ssbuf_copy_kernel(x: pl.Tensor[[1], pl.DT_INT32]):
    message = pl.struct("Message", batch=0, block=0, offset=0)
    plain_message = pl.struct("PlainMessage", value=0)
    plain_message.value = 1
    with pl.section_vector():
        message.batch = 8
        message.block = 1
        message.offset = 32768
        sub_id = pl.get_subblock_idx()
        if sub_id == 0:
            pl.ssbuf_store(message, 0)
            pl.system.set_cross_core(pipe=pl.PipeType.S, event_id=15)

    with pl.section_cube():
        pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=15, sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
        pl.ssbuf_load(message, 0)
        pl.printf("Get ssbuf mssage: batch=%d, block=%d, offset=%d", message.batch, message.block, message.offset)


@pl.jit()
def ssbuf_single_direction_kernel(x: pl.Tensor[[1], pl.DT_INT32]):
    store_only = pl.struct("StoreOnly", value=0)
    load_only = pl.struct("LoadOnly", value=0)
    with pl.section_vector():
        pl.ssbuf_store(store_only, 0)
        pl.ssbuf_load(load_only, 0)


@dataclass
class SsbufTiling:
    batch: int
    offsets: int[2]


@pl.jit()
def ssbuf_tiling_kernel(
    x: pl.Tensor[[1], pl.DT_INT32],
    tiling: SsbufTiling,
):
    with pl.section_vector():
        pl.ssbuf_store(tiling, 0)
    with pl.section_cube():
        pl.ssbuf_load(tiling, 0)


@pl.jit()
def struct_scalar_initializer_kernel(x: pl.Tensor[[1], pl.DT_INT32]):
    with pl.section_cube():
        for work_id in pl.range(0, 2):
            run_info = pl.struct("RunInfo", workId=work_id, offset=work_id // 2)
            pl.printf("workId=%d, offset=%d", run_info.workId, run_info.offset)


def test_ssbuf_store_and_load_codegen():
    cpp = _compile_to_cce(ssbuf_copy_kernel)
    logging.info("%s", cpp)
    assert "class Message" in cpp
    for field in ("batch", "block", "offset"):
        assert f"volatile int64_t {field};" in cpp
    assert "class PlainMessage {\npublic:\n    int64_t value;" in cpp
    assert "volatile int64_t value;" not in cpp
    assert "reinterpret_cast<__ssbuf__ uint32_t*>((uint64_t)(0))" in cpp
    assert "reinterpret_cast<const uint32_t*>(&message_0)" in cpp
    assert "reinterpret_cast<uint32_t*>(&message_0)" in cpp
    assert "sizeof(message_0) / sizeof(uint32_t)" in cpp
    assert cpp.count("wait_intra_block(PIPE_S, 15);") == 1
    assert "wait_intra_block(PIPE_S, 31);" not in cpp


def test_ssbuf_load_and_store_each_mark_struct_volatile():
    cpp = _compile_to_cce(ssbuf_single_direction_kernel)
    assert "class StoreOnly {\npublic:\n    volatile int64_t value;" in cpp
    assert "class LoadOnly {\npublic:\n    volatile int64_t value;" in cpp


def test_ssbuf_tiling_header_uses_volatile_fields():
    result = _codegen_result(ssbuf_tiling_kernel)
    header = result.tiling_headers["SsbufTiling_tiling.h"]
    assert "volatile int64_t batch;" in header
    assert "volatile int64_t offsets[2];" in header
    assert '#include "SsbufTiling_tiling.h"' in result.content
    assert "class SsbufTiling" not in result.content


def test_struct_scalar_initializers_are_cast_to_member_type():
    cpp = _compile_to_cce(struct_scalar_initializer_kernel)
    assert "for (int64_t work_id__iterator_0 = 0;" in cpp
    assert ".workId=static_cast<int64_t>(work_id__iterator_0)" in cpp
    assert ".offset=static_cast<int64_t>(" in cpp
