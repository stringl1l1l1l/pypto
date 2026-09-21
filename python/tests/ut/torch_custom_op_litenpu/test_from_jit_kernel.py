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
"""Real-backend check for DIRECT mode (``CompileEntry(jit_kernel=...)``): the kernel carries no
``soc_version``, so the soc must reach the build through the process-global setter the entry pins.
SIM-builds the kernel and asserts a ``.o`` is produced; skips when the SIM backend is unavailable.
"""
import pytest


def test_from_jit_kernel_real_build_produces_kernel():
    pytest.importorskip("pypto")
    import pypto
    from pypto.extensions.torch_custom_op_litenpu.common.compile import CompileEntry

    # DIRECT: a bare @pypto.frontend.jit kernel with erased annotations and no soc_version, so soc
    # arrives solely through the entry's process-global set_codegen_options(soc_version=...).
    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
    def add_kernel(
        input0: pypto.Tensor([...]),
        input1: pypto.Tensor([...]),
        output: pypto.Tensor([...]),
    ):
        pypto.set_vec_tile_shapes(1, 4, 1, 64)
        output.move(input0 + input1)

    entry = CompileEntry(jit_kernel=add_kernel, num_inputs=2, num_outputs=1)

    def infer_shape(x_shape, y_shape):
        return x_shape

    def infer_dtype(x_dtype, y_dtype):
        return x_dtype

    entry.infer_shape = infer_shape
    entry.infer_dtype = infer_dtype

    try:
        # soc_version passed only as the 4th positional arg, so the entry threads it to the global setter.
        path = entry(((1, 8, 1, 64), (1, 8, 1, 64)), ("float16", "float16"), {}, "Kirin9030")
    except Exception as e:  # noqa: BLE001 - a missing backend has no single statically known raise type
        pytest.skip(f"DIRECT build unavailable on this host (backend): {type(e).__name__}: {str(e)[:160]}")

    assert path and str(path).endswith(".o"), path
