# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exercise the native collection state across independently loaded launchers."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("registration", ["success", "failure", "no_cann"])
def test_native_prof_status(tmp_path, registration):
    root = Path(__file__).resolve().parents[6]
    source = root / "framework/src/interface/pypto_pro/profiler.cpp"
    source_flags = ["-MMD", "-MF", str(tmp_path / "backend.so.d")]
    header_filters = []
    if registration != "no_cann":
        ascend_home = os.environ.get("ASCEND_HOME_PATH")
        if not ascend_home:
            pytest.skip("CANN headers required to compile the native profiling callback")
        header_filters = [str(Path(ascend_home) / "pkg_inc"), str(Path(ascend_home) / "include")]
        source_flags += ["-DBUILD_WITH_CANN", "-I", header_filters[0], "-L", str(tmp_path), "-lprofapi"]
    stub = tmp_path / "profapi.cpp"
    stub.write_text(r"""
#include <cassert>
#include <cstdint>
using Callback = int32_t (*)(uint32_t, void*, uint32_t);
static Callback callback;
extern "C" int32_t MsprofRegisterCallback(uint32_t module, Callback handler)
{
    assert(module == 0x46);
    assert(callback == nullptr);
#ifdef FAIL_REGISTRATION
    return -1;
#else
    callback = handler;
    return 0;
#endif
}
extern "C" void command(uint32_t type, uint64_t value, uint32_t length, bool nullData)
{
    assert(callback != nullptr);
    assert(callback(type, nullData ? nullptr : &value, length) == 0);
}
""")
    launcher = tmp_path / "launcher.cpp"
    launcher.write_text('extern "C" bool GetPyptoProfStatus();\n'
                        'extern "C" bool query() { return GetPyptoProfStatus(); }\n')
    flags = ["-DFAIL_REGISTRATION"] if registration == "failure" else []
    for src, name, extra in (
        (stub, "libprofapi.so", flags),
        (source, "backend.so", source_flags),
        (launcher, "launcher1.so", []),
        (launcher, "launcher2.so", []),
    ):
        subprocess.run(["g++", "-std=c++17", "-shared", "-fPIC", "-O2", str(src),
                        "-o", str(tmp_path / name), *extra], check=True, capture_output=True, text=True)

    # Apply the same transitive header rules as the online build to GCC's real
    # dependency file, for both CANN and CPU-only configurations.
    analysis = subprocess.run([
        sys.executable, str(root / "cmake/scripts/analysis_binary_header_files.py"),
        "-s", str(root), "-b", str(tmp_path), "-t", "libtile_fwk_interface.so",
        "--target_binary_dir", str(tmp_path), "-o", str(tmp_path / "backend.so"),
        "-j", str(root / "cmake/scripts/analysis_binary_header_files.json"), "-f", *header_filters,
    ], capture_output=True, text=True)
    assert analysis.returncode == 0, analysis.stdout + analysis.stderr

    script = r"""
import ctypes
import sys
backend = ctypes.CDLL('./backend.so', mode=ctypes.RTLD_GLOBAL)
backend.GetPyptoProfStatus.restype = ctypes.c_bool
assert not backend.GetPyptoProfStatus()
if sys.argv[1] != 'success':
    sys.exit(0)
prof = ctypes.CDLL('./libprofapi.so')
prof.command.argtypes = [ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_bool]
launchers = []
for session in range(2):
    prof.command(1, 0x4000, 8, False)
    # Each library loads after collection starts, and must see the existing state.
    launcher = ctypes.CDLL(f'./launcher{session + 1}.so')
    launcher.query.restype = ctypes.c_bool
    launchers.append(launcher)
    assert all(lib.query() for lib in launchers)
    # Malformed and unrelated reporter/step notifications must not stop collection.
    for args in [(1, 0, 0, False), (1, 0, 8, True), (2, 0, 8, False), (3, 0, 8, False)]:
        prof.command(*args)
        assert all(lib.query() for lib in launchers)
    prof.command(1, 0, 8, False)
    assert all(not lib.query() for lib in launchers)
"""
    env = dict(os.environ, LD_LIBRARY_PATH=str(tmp_path))
    subprocess.run([sys.executable, "-c", script, registration], cwd=tmp_path, env=env,
                   check=True, capture_output=True, text=True)


@pytest.mark.parametrize("native_provider", [False, True], ids=["standalone", "pypto_backend"])
def test_prof_status_dispatch(tmp_path, native_provider):
    """A dumped launcher loads without PyPTO; an imported backend owns the live flag when present."""
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if not ascend_home:
        pytest.skip("CANN headers required to compile the packaged profiling helper")
    root = Path(__file__).resolve().parents[6]
    source = tmp_path / "status.cpp"
    source.write_text(r"""
#include "pypto_profiler.h"
static bool nativeStatus;
#ifdef NATIVE_PROVIDER
extern "C" bool GetPyptoProfStatus() { return nativeStatus; }
#endif
extern "C" bool query() { return pypto::GetProfStatus(); }
extern "C" void set_status(bool native) { nativeStatus = native; }
""")
    flags = ["-DNATIVE_PROVIDER"] if native_provider else []
    subprocess.run([
        "g++", "-std=c++17", "-shared", "-fPIC", "-O2", str(source), "-o", str(tmp_path / "status.so"),
        "-I", str(root / "python/pypto_pro/include"), "-I", str(Path(ascend_home) / "include"),
        "-I", str(Path(ascend_home) / "pkg_inc"), *flags,
    ], check=True, capture_output=True, text=True)
    # A fresh interpreter must not inherit pytest's RTLD_GLOBAL PyPTO backend.
    script = r"""
import ctypes
import sys
lib = ctypes.CDLL('./status.so')
lib.query.restype = ctypes.c_bool
lib.set_status.argtypes = [ctypes.c_bool]
for native in [False, True, False, True, False]:
    lib.set_status(native)
    expected = native if sys.argv[1] == 'True' else False
    assert lib.query() == expected
"""
    subprocess.run([sys.executable, "-c", script, str(native_provider)], cwd=tmp_path,
                   check=True, capture_output=True, text=True)
