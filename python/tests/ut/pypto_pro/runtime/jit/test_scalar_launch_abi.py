# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Execute both generated ABI adapters and check physical slot sizes without an NPU."""

import ctypes
import importlib
import shutil
import struct
import subprocess

import pytest

jit = importlib.import_module("pypto_pro.runtime.jit")

SCALARS = [
    ("bool", ctypes.c_bool),
    ("int8_t", ctypes.c_int8),
    ("uint8_t", ctypes.c_uint8),
    ("int16_t", ctypes.c_int16),
    ("uint16_t", ctypes.c_uint16),
    ("int32_t", ctypes.c_int32),
    ("uint32_t", ctypes.c_uint32),
    ("float", ctypes.c_float),
    ("int64_t", ctypes.c_int64),
    ("uint64_t", ctypes.c_uint64),
]


@pytest.fixture(scope="module")
def native_scalar_launcher(tmp_path_factory):
    # Use the same compiler as the ASC launcher, including its C++17 bit-cast builtin.
    compiler = shutil.which("bisheng")
    if compiler is None:
        pytest.skip("bisheng is required to compile the scalar launch adapters")
    root = tmp_path_factory.mktemp("scalar_launch_abi")
    (root / "runtime").mkdir()
    (root / "runtime/rt_ffts.h").write_text("#pragma once\n")
    (root / "pypto_launch.h").write_text("""#pragma once
#include <cstdint>
namespace pypto {
template<int, int> int64_t ResolveLaunchBlockDim(uint32_t n, void*) { return n; }
}
""")
    params = [(typ, f"v{i}", False) for i, (typ, _) in enumerate(SCALARS)]
    # Exercise a pointer between narrow scalars, trailing dimensions and the hidden
    # cross-core-sync pointer. None may shift when a preceding scalar is widened.
    params.insert(1, ("uint64_t", "out", True))
    params.extend([("int64_t", "rows", False), ("int64_t", "cols", False)])
    entry_params = params + [("int64_t", "ffts_addr", True)]
    signature = ", ".join(f"{typ}{'*' if ptr else ''} {name}" for typ, name, ptr in entry_params)
    stores = "\n".join(
        f"    std::memcpy(&out[{i}], &v{i}, sizeof(v{i}));" for i in range(len(SCALARS))
    )
    (root / "kernel.cpp").write_text(f"""#include <cstring>
#define __global__
#define __vector__
#define __gm__
static int64_t syncCookie = 0x12345678;
static void rtGetC2cCtrlAddr(uint64_t* addr, uint32_t* size)
{{
    *addr = reinterpret_cast<uint64_t>(&syncCookie);
    *size = sizeof(syncCookie);
}}
static void probe_impl({signature})
{{
{stores}
    out[10] = rows;
    out[11] = cols;
    out[12] = *ffts_addr;
}}
""")
    target = jit.get_jit_compile_config().resolve_kernel_target("a5", has_cube=False, has_vector=True)
    entry = jit._make_global_entry("probe", entry_params, target=target)
    entry += """template<class... Args> void CheckLaunchSlots(Args... args)
{
    static_assert(((sizeof(Args) == 8) && ...), "Every physical launch slot must occupy eight bytes");
    probe(args...);
}
"""
    caller = jit._generate_caller_cpp(
        params, "kernel.cpp", "probe", has_cross_core_sync=True, target=target, global_entry=entry,
    )
    # Replace only the ASC dispatch; compile the real host encoder and device decoder.
    caller = caller.replace("probe<<<blockDim, nullptr, stream>>>(", "CheckLaunchSlots(")
    source = root / "caller.cpp"
    source.write_text(caller)
    library = root / "caller.so"
    result = subprocess.run(
        [compiler, "-x", "c++", "-std=c++17", "-O2", "-shared", "-fPIC", f"-I{root}",
         str(source), "-o", str(library)], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    lib = ctypes.CDLL(str(library))
    scalar_types = [ctype for _, ctype in SCALARS]
    scalar_types.insert(1, ctypes.c_void_p)
    lib.call_kernel.argtypes = [ctypes.c_uint32, ctypes.c_void_p, *scalar_types, ctypes.c_int64, ctypes.c_int64]
    lib.call_kernel.restype = ctypes.c_int64
    return lib


@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("float_bits", [0x00000000, 0x80000000, 0x3FA00000, 0x7F800000, 0xFF800000, 0x7FC12345])
def test_scalar_launch_round_trip(native_scalar_launcher, upper, float_bits):
    values = []
    for typ, ctype in SCALARS:
        bits = ctypes.sizeof(ctype) * 8
        if typ == "float":
            value = struct.unpack("f", struct.pack("I", float_bits))[0]
        elif typ == "bool":
            value = upper
        elif typ.startswith("uint"):
            value = (1 << bits) - 1 if upper else 0
        else:
            value = (1 << (bits - 1)) - 1 if upper else -(1 << (bits - 1))
        values.append(value)
    # Compare bytes, including signed zero/NaN payloads and the full uint64 range.
    expected = [int.from_bytes(bytes(ctype(value)), "little") for (_, ctype), value in zip(SCALARS, values)]
    out = (ctypes.c_uint64 * 13)()
    values.insert(1, ctypes.addressof(out))
    assert native_scalar_launcher.call_kernel(1, None, *values, 65, 96) == 1
    assert list(out) == [*expected, 65, 96, 0x12345678]
