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
"""Execute generated host launchers with a fault-injectable ACL stub, without an NPU.

Only the CCE launch expression becomes a recorder. The emitted host control flow,
resource helper, compiler topology and int64_t return ABI are compiled as written,
so these tests detect launches after query failure, an incomplete mixed group,
an over-budget request, and the auto sentinel's full-budget resolution.
"""

import ctypes
from dataclasses import replace
import importlib
from pathlib import Path
import subprocess

import pytest

jit = importlib.import_module("pypto_pro.runtime.jit")


@pytest.fixture(scope="module")
def native_launchers(tmp_path_factory):
    """Link real generated callers against controlled limits and record every launch and query."""
    root = tmp_path_factory.mktemp("native_core_limits")
    (root / "acl").mkdir()
    (root / "runtime").mkdir()
    (root / "acl/acl_rt.h").write_text("""#pragma once
#include <cstdint>
using aclError = int32_t;
using aclrtStream = void*;
constexpr aclError ACL_SUCCESS = 0;
enum aclrtDevResLimitType { ACL_RT_DEV_RES_CUBE_CORE = 0, ACL_RT_DEV_RES_VECTOR_CORE = 1 };
extern "C" aclError aclrtGetStreamResLimit(aclrtStream, aclrtDevResLimitType, uint32_t*);
""")
    (root / "runtime/rt_ffts.h").write_text("")
    (root / "fake_kernel.h").write_text("")
    harness = """#include <acl/acl_rt.h>
static uint32_t budgets[2], errors[2], queries[2];
static void* queryStreams[2];
static uint32_t launches, blocks, syncSetups;
extern "C" void reset(uint32_t cube, uint32_t vector, uint32_t cubeError, uint32_t vectorError)
{
    budgets[0] = cube;
    budgets[1] = vector;
    errors[0] = cubeError;
    errors[1] = vectorError;
    queries[0] = queries[1] = launches = blocks = syncSetups = 0;
    queryStreams[0] = queryStreams[1] = nullptr;
}
extern "C" aclError aclrtGetStreamResLimit(aclrtStream stream, aclrtDevResLimitType type, uint32_t* value)
{
    ++queries[type];
    queryStreams[type] = stream;
    *value = budgets[type];
    return static_cast<aclError>(errors[type]);
}
extern "C" uint32_t launch_count() { return launches; }
extern "C" uint32_t launched_blocks() { return blocks; }
extern "C" uint32_t query_count(uint32_t type) { return queries[type]; }
extern "C" void* query_stream(uint32_t type) { return queryStreams[type]; }
extern "C" uint32_t sync_setup_count() { return syncSetups; }
static void rtGetC2cCtrlAddr(uint64_t* address, uint32_t* length)
{
    ++syncSetups;
    *address = 0;
    *length = 0;
}
static void RecordLaunch(uint32_t dim, void*, int64_t* = nullptr)
{
    ++launches;
    blocks = dim;
}
"""
    config = jit.get_jit_compile_config()
    # A future architecture can use 1:1 without changing A5's 1:2 ABI. This target
    # name is only a test fixture, not a supported Bisheng architecture.
    config = replace(config, kernel_targets={
        **config.kernel_targets,
        "future": {"cube_vec": jit.KernelTarget("test-mixed-1-1", 1, 1, True)},
    })
    modes = {
        "cube": ("3510", True, False, False),
        "vector": ("3510", False, True, False),
        "mixed": ("3510", True, True, False),
        "mixed_1_1": ("future", True, True, False),
        "sync": ("3510", True, True, True),
    }
    for name, (arch, has_cube, has_vec, cross_sync) in modes.items():
        caller = jit._generate_caller_cpp(
            [], "fake_kernel.h", "probe", has_cross_core_sync=cross_sync,
            target=config.resolve_kernel_target(arch, has_cube=has_cube, has_vector=has_vec),
        )
        caller = caller.replace("call_kernel(", f"call_kernel_{name}(")
        # Substitute only the device launch syntax: all surrounding host code is real.
        caller = caller.replace(
            "probe<<<blockDim, nullptr, stream>>>(",
            "RecordLaunch(blockDim, stream" + (", " if cross_sync else ""),
        )
        harness += caller
    source = root / "native_launchers.cpp"
    source.write_text(harness)
    library = root / "native_launchers.so"
    include = Path(jit.__file__).resolve().parents[1] / "include"
    # torch_npu may already have loaded the real ACL globally. Bind the stub inside
    # this test library so an ELF interposition cannot send fake streams to a device.
    subprocess.run(
        ["g++", "-std=c++17", "-O2", "-shared", "-fPIC", "-Wl,-Bsymbolic", f"-I{root}", f"-I{include}",
         str(source), "-o", str(library)], check=True, capture_output=True, text=True,
    )
    lib = ctypes.CDLL(str(library))
    lib.reset.argtypes = [ctypes.c_uint32] * 4
    lib.reset.restype = None
    for name in ("launch_count", "launched_blocks", "sync_setup_count"):
        getattr(lib, name).argtypes = []
        getattr(lib, name).restype = ctypes.c_uint32
    lib.query_count.argtypes = [ctypes.c_uint32]
    lib.query_count.restype = ctypes.c_uint32
    lib.query_stream.argtypes = [ctypes.c_uint32]
    lib.query_stream.restype = ctypes.c_void_p
    for name in modes:
        fn = getattr(lib, f"call_kernel_{name}")
        fn.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
        fn.restype = ctypes.c_int64
    return lib


@pytest.mark.parametrize(
    "mode,cube,vector,requested,expected,queries",
    [
        ("cube", 5, 0, 1, 1, (1, 0)),
        ("cube", 5, 0, 5, 5, (1, 0)),
        ("cube", 5, 0, 0, 5, (1, 0)),
        ("vector", 0, 7, 7, 7, (0, 1)),
        ("vector", 0, 72, 72, 72, (0, 1)),
        ("vector", 0, 7, 0, 7, (0, 1)),
        ("mixed", 20, 40, 3, 3, (1, 1)),
        ("mixed", 8, 8, 4, 4, (1, 1)),
        ("mixed", 4, 20, 0, 4, (1, 1)),
        ("mixed", 20, 9, 0, 4, (1, 1)),
        ("mixed", 20, 2, 0, 1, (1, 1)),
        ("mixed_1_1", 8, 5, 5, 5, (1, 1)),
        ("mixed_1_1", 8, 5, 0, 5, (1, 1)),
        ("sync", 8, 5, 2, 2, (1, 1)),
        ("sync", 8, 5, 0, 2, (1, 1)),
    ],
)
def test_native_launch_counts(native_launchers, mode, cube, vector, requested, expected, queries):
    """Check the recorded launch as well as the return value; an over-budget launch must never pass.

    A requested 0 is the auto sentinel: the launch takes the stream's full budget,
    bounded by the tighter engine for mixed blocks.
    """
    lib = native_launchers
    lib.reset(cube, vector, 0, 0)
    assert getattr(lib, f"call_kernel_{mode}")(requested, 0x1234) == expected
    assert lib.launch_count() == 1
    assert lib.launched_blocks() == expected
    assert tuple(lib.query_count(i) for i in range(2)) == queries
    for i, count in enumerate(queries):
        assert lib.query_stream(i) == (0x1234 if count else None)
    assert lib.sync_setup_count() == (1 if mode == "sync" else 0)


@pytest.mark.parametrize(
    "mode,cube,vector,requested,expected",
    [
        ("cube", 5, 0, 20, -0x500000005),
        ("vector", 0, 7, 20, -0x500000007),
        ("vector", 0, 72, 0xFFFFFFFF, -0x500000048),
        ("mixed", 4, 20, 20, -0x500000004),
        ("mixed", 20, 9, 20, -0x500000004),
        ("mixed", 20, 2, 12, -0x500000001),
        ("mixed_1_1", 8, 5, 12, -0x500000005),
        ("sync", 8, 5, 12, -0x500000002),
    ],
)
def test_native_requests_above_budget_reject_launch(native_launchers, mode, cube, vector, requested, expected):
    """An explicit request the stream cannot host must fail before any device launch or sync setup."""
    lib = native_launchers
    lib.reset(cube, vector, 0, 0)
    assert getattr(lib, f"call_kernel_{mode}")(requested, 0x1234) == expected
    assert lib.launch_count() == 0
    assert lib.sync_setup_count() == 0


@pytest.mark.parametrize(
    "cube,vector,cube_error,vector_error,expected",
    [
        (0, 8, 0, 0, -0x100000000),
        (8, 0, 0, 0, -0x200000000),
        (8, 1, 0, 0, -0x200000001),
        (8, 16, 507000, 0, -(0x300000000 + 507000)),
        (8, 16, 0, 507001, -(0x400000000 + 507001)),
        (8, 16, 0xFFFFFFFF, 0, -0x3FFFFFFFF),
    ],
)
def test_native_errors_prevent_launch_and_sync_setup(
    native_launchers, cube, vector, cube_error, vector_error, expected,
):
    """Fault injection exercises real native-to-Python error bits and rejects before any synchronized participant."""
    lib = native_launchers
    lib.reset(cube, vector, cube_error, vector_error)
    result = lib.call_kernel_sync(12, 0x1234)
    assert result == expected
    assert lib.launch_count() == 0
    assert lib.sync_setup_count() == 0


@pytest.mark.parametrize("mode,cube,vector,cube_error,vector_error", [
    ("cube", 4, 0, 0, 507000), ("vector", 0, 4, 507000, 0),
])
def test_unused_engine_query_errors_do_not_block_launch(
    native_launchers, mode, cube, vector, cube_error, vector_error,
):
    """Querying both engines would incorrectly fail a valid single-engine kernel on the unused engine."""
    lib = native_launchers
    lib.reset(cube, vector, cube_error, vector_error)
    assert getattr(lib, f"call_kernel_{mode}")(4, 0x1234) == 4
    assert lib.launch_count() == 1


def test_native_query_observes_budget_and_stream_changes(native_launchers):
    """A cached native function must requery on every call, even after a previous budget rejected launch."""
    lib = native_launchers
    for vector, stream, expected in [(6, 0x1234, 3), (2, 0x5678, 1), (1, 0x5678, None), (8, 0x1234, 4)]:
        lib.reset(8, vector, 0, 0)
        result = lib.call_kernel_mixed(0, stream)
        if expected is None:
            assert result == -0x200000001
            assert lib.launch_count() == 0
        else:
            assert result == lib.launched_blocks() == expected
            assert lib.launch_count() == 1
        assert lib.query_count(0) == lib.query_count(1) == 1
        assert lib.query_stream(0) == lib.query_stream(1) == stream


def test_native_rejection_does_not_poison_later_launches(native_launchers):
    """A request rejected by one budget must not cache anything that alters the next launch."""
    lib = native_launchers
    lib.reset(8, 6, 0, 0)
    assert lib.call_kernel_mixed(12, 0x1234) == -0x500000003
    assert lib.launch_count() == 0
    lib.reset(20, 40, 0, 0)
    assert lib.call_kernel_mixed(12, 0x1234) == 12
    assert lib.launched_blocks() == 12
