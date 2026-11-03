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

"""SIMT kernel decorator validation and grid/thread block launch configuration."""

import importlib
from unittest.mock import MagicMock

from pypto_pro._errors import InvalidArgument, InvalidOperation, InvalidShape, InvalidType, NotSupported, OutOfRange
import pypto_pro.language as pl
from pypto_pro.runtime.platform import NpuArch
import pytest

jit_impl = importlib.import_module("pypto_pro.runtime.jit")


@pytest.mark.parametrize(
    "max_threads, error_type, error",
    [
        (0, OutOfRange, r"\[1, 2048\]"),
        (2049, OutOfRange, r"\[1, 2048\]"),
        (True, InvalidType, "integer"),
        (1.5, InvalidType, "integer"),
        ("32", InvalidType, "integer"),
    ],
)
def test_simt_kernel_jit_validates_launch_bound_at_decoration(max_threads, error_type, error):
    with pytest.raises(error_type, match=error):
        pl.jit(mode="simt", max_threads=max_threads)


def test_simt_kernel_jit_requires_and_owns_max_threads():
    with pytest.raises(InvalidArgument, match="required"):
        pl.jit(mode="simt")(lambda: None)
    with pytest.raises(NotSupported, match="only supported"):
        pl.jit(max_threads=32)(lambda: None)


def test_simt_kernel_jit_accepts_3510_arch_and_rejects_2201():
    kernel = pl.jit(mode="simt", max_threads=32, arch=NpuArch.DAV_3510)(lambda: None)
    assert kernel._arch == NpuArch.DAV_3510
    enum_kernel = pl.jit(mode="simt", max_threads=32, arch=NpuArch.DAV_3510)(lambda: None)
    assert enum_kernel._arch == NpuArch.DAV_3510

    with pytest.raises(NotSupported, match="only support arch 'DAV_3510'"):
        pl.jit(mode="simt", max_threads=32, arch=NpuArch.DAV_2201)


def test_jit_rejects_invalid_mode():
    with pytest.raises(InvalidArgument, match="mode must be 'simd' or 'simt'"):
        pl.jit(mode="invalid")


def test_simt_kernel_rejects_direct_host_call():
    @pl.jit(mode="simt", max_threads=32)
    def kernel():
        pass

    with pytest.raises(InvalidOperation, match="requires launch configuration"):
        kernel()


@pytest.mark.parametrize(
    "grid, thread_block, expected_grid, expected_thread_block",
    [
        (4, 256, (4, 1, 1), (256, 1, 1)),
        ((2, 3), (8, 4), (2, 3, 1), (8, 4, 1)),
        ((2, 3, 4), (4, 4, 4), (2, 3, 4), (4, 4, 4)),
    ],
)
def test_simt_kernel_launch_config_normalizes_dim3(grid, thread_block, expected_grid, expected_thread_block):
    actual = jit_impl._validate_simt_launch_config(grid, thread_block, max_threads=256)
    assert actual == (expected_grid, expected_thread_block)


@pytest.mark.parametrize(
    "grid, thread_block, error_type, error",
    [
        (0, 1, InvalidShape, "grid dimensions"),
        ((256, 256), 1, OutOfRange, "grid dimensions product"),
        (1, 257, OutOfRange, "max_threads"),
        (1, (16, 17), OutOfRange, "max_threads"),
    ],
)
def test_simt_kernel_launch_config_rejects_invalid_values(grid, thread_block, error_type, error):
    with pytest.raises(error_type, match=error):
        jit_impl._validate_simt_launch_config(grid, thread_block, max_threads=256)


def test_simt_kernel_launch_key_supports_default_and_explicit_stream():
    parse = jit_impl._parse_simt_launch_key

    assert parse((4, 256)) == (None, 4, 256)
    assert parse(("stream", (2, 3), (8, 4))) == ("stream", (2, 3), (8, 4))
    with pytest.raises(InvalidArgument, match="Expected kernel"):
        parse(("stream", 4, 256, 1024))


def test_simt_kernel_without_parameters_uses_jit_launch(monkeypatch):
    @pl.jit(mode="simt", max_threads=32)
    def kernel():
        pass

    compiled = object()
    launch = MagicMock()
    monkeypatch.setattr(kernel, "_validate_launch_arch", lambda: None)
    monkeypatch.setattr(kernel, "_ensure_compiled", lambda *_args, **_kwargs: compiled)
    monkeypatch.setattr(jit_impl, "_launch", launch)
    kernel[1, 32]()
    launch.assert_called_once_with(compiled, (), ((1, 1, 1), (32, 1, 1)), None)
