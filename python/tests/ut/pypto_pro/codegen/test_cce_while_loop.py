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
"""CCE codegen smoke tests for frontend while/for break and continue.

The tests verify the generated C++ uses native loop and jump constructs for the
block DSL patterns that codegen owns. They intentionally avoid asserting older
temporary lowering details such as synthetic while guards or SSA slot names.
"""

import logging
import math
import re
import subprocess
from textwrap import dedent

import pypto_pro.language as pl
import pytest


def _compile_to_cce(kernel) -> str:
    from pypto_pro.runtime.jit import _assemble_cv_source, _parse_and_codegen_targets

    cube, vector = _parse_and_codegen_targets(kernel.to_kernel_def(), "3510", "")
    return _assemble_cv_source(cube, vector).content


def _assert_lowered_while_cpp(cpp: str, condition_fragment: str) -> tuple[int, int]:
    """Assert condition evaluation is emitted in a native C++ while loop."""
    loop_index = cpp.index("while (")
    condition_index = cpp.index(condition_fragment, loop_index)
    assert loop_index <= condition_index
    return loop_index, condition_index


def _assert_materialized_jump_guard(cpp: str, condition_fragment: str, jump: str) -> tuple[int, int, int]:
    """Assert a source condition is emitted in an if containing the requested native jump."""
    condition_index = cpp.index(condition_fragment)
    jump_index = cpp.index(jump, condition_index)
    inline_guard = cpp.rfind("if (", 0, condition_index)
    is_inline = inline_guard >= 0 and "\n" not in cpp[inline_guard:condition_index]
    guard_index = inline_guard if is_inline else cpp.index("if (", condition_index)
    if is_inline:
        assert guard_index < condition_index < jump_index
    else:
        assert condition_index < guard_index < jump_index
    return condition_index, guard_index, jump_index


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@pl.jit
def _while_basic_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
    b: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tile_type_a = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile(tile_type_a, addr=0x0000)
    tile_type_b = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_b = pl.make_tile(tile_type_b, addr=0x4000)
    tile_type_c = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_c = pl.make_tile(tile_type_c, addr=0x8000)
    pl.load(tile_a, a, [0, 0])
    pl.load(tile_b, b, [0, 0])
    i = 0
    while i < 5:
        pl.add(tile_c, tile_a, tile_b)
        i = i + 1


@pl.jit
def _while_continue_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
    b: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tile_type_a = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile(tile_type_a, addr=0x0000)
    tile_type_b = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_b = pl.make_tile(tile_type_b, addr=0x4000)
    tile_type_c = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_c = pl.make_tile(tile_type_c, addr=0x8000)
    pl.load(tile_a, a, [0, 0])
    pl.load(tile_b, b, [0, 0])
    i = 0
    while i < 5:
        if i == 2:
            i = i + 1
            continue
        pl.add(tile_c, tile_a, tile_b)
        i = i + 1


@pl.jit
def _while_break_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
    b: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tile_type_a = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile(tile_type_a, addr=0x0000)
    tile_type_b = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_b = pl.make_tile(tile_type_b, addr=0x4000)
    tile_type_c = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_c = pl.make_tile(tile_type_c, addr=0x8000)
    pl.load(tile_a, a, [0, 0])
    pl.load(tile_b, b, [0, 0])
    i = 0
    while i < 5:
        if i == 2:
            break
        pl.add(tile_c, tile_a, tile_b)
        i = i + 1


@pl.jit
def _while_break_carry_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
    b: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tile_type_a = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile(tile_type_a, addr=0x0000)
    tile_type_b = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_b = pl.make_tile(tile_type_b, addr=0x4000)
    pl.load(tile_a, a, [0, 0])
    pl.load(tile_b, b, [0, 0])
    i = 0
    acc = 0
    while i < 5:
        i = i + 1
        if i == 3:
            break
        acc = acc + i


@pl.jit
def _while_accumulate_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
    b: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tile_type_a = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile(tile_type_a, addr=0x0000)
    tile_type_b = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_b = pl.make_tile(tile_type_b, addr=0x4000)
    tile_type_c = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_c = pl.make_tile(tile_type_c, addr=0x8000)
    pl.load(tile_a, a, [0, 0])
    pl.load(tile_b, b, [0, 0])
    i = 0
    acc = 0
    while i < 5:
        pl.add(tile_c, tile_a, tile_b)
        acc = acc + i
        i = i + 1


@pl.jit(auto_mutex=True)
def _while_getval_condition_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_group = pl.make_tile_group(type=tile_type, addrs=[0x0000], mutex_ids=[20])
    with pl.section_vector():
        tile = tile_group.next()
        pl.load(tile, a, [0, 0])
        i = 0
        while tile[0, i] > 0:
            i = i + 1
            if i >= 2:
                break


@pl.jit
def _for_continue_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
    b: pl.Tensor[[64, 128], pl.DT_FP16],
):
    acc = 0
    for i in pl.range(5):
        if i == 2:
            continue
        acc = acc + i
    for _j in pl.range(acc):
        pl.system.bar_all()


@pl.jit
def _for_break_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
    b: pl.Tensor[[64, 128], pl.DT_FP16],
):
    acc = 0
    for i in pl.range(5):
        if i == 3:
            break
        acc = acc + i


@pl.jit
def _loop_tuple_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile(tile_type, addr=0x0000)
    tile_b = pl.make_tile(tile_type, addr=0x4000)
    tile_c = pl.make_tile(tile_type, addr=0x8000)
    bundle = pl.make_tuple(a=tile_a, b=tile_b, c=tile_c)
    for _i in pl.range(1):
        bundle = pl.make_tuple(a=tile_a, b=tile_b, c=tile_c)
    pl.add(bundle.c, bundle.c, bundle.c)


@pl.jit(auto_mutex=False)
def _while_independent_body_result_kernel(flag: pl.DT_BOOL, result_value: pl.DT_INT64):
    value = pl.const(0, pl.DT_INT32)
    while True:
        if flag:
            value = result_value + 0
            break
        value = pl.const(2, pl.DT_INT32)
        continue
    _test_result = value + 1  # noqa: F841


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cce_while_basic():
    cpp = _compile_to_cce(_while_basic_kernel)
    logging.info("\n=== test_cce_while_basic ===\n%s", cpp)

    _assert_lowered_while_cpp(cpp, "< 5")
    assert "_can_continue" not in cpp, "Basic while needs no break flag"
    assert "TADD(" in cpp, "Loop body op should be emitted"
    assert "+ 1" in cpp, "Expected the i = i + 1 update"


def test_cce_while_continue():
    cpp = _compile_to_cce(_while_continue_kernel)
    logging.info("\n=== test_cce_while_continue ===\n%s", cpp)

    _assert_lowered_while_cpp(cpp, "< 5")
    assert "_can_continue" not in cpp, "Native continue needs no break flag"
    assert "continue;" in cpp, "Expected a native C++ continue"
    _, guard_idx, cont_idx = _assert_materialized_jump_guard(cpp, "== 2", "continue;")
    assert guard_idx < cont_idx, "continue must be inside its guard"
    assert re.search(
        r"\bi(?:_iter)?_\d+\s*=\s*i(?:_iter)?_\d+;",
        cpp[guard_idx:cont_idx],
    ), "induction var update must precede continue"
    assert "TADD(" in cpp, "Body op should remain on the fall-through path"


def test_cce_while_break():
    cpp = _compile_to_cce(_while_break_kernel)
    logging.info("\n=== test_cce_while_break ===\n%s", cpp)

    assert "_can_continue" not in cpp, "Native break needs no _can_continue flag"
    _assert_lowered_while_cpp(cpp, "< 5")
    _assert_materialized_jump_guard(cpp, "== 2", "break;")
    assert cpp.count("break;") >= 1, "Expected source C++ break"
    assert "TADD(" in cpp, "Body op present on the no-break path"


def test_cce_while_break_carry():
    cpp = _compile_to_cce(_while_break_carry_kernel)
    logging.info("\n=== test_cce_while_break_carry ===\n%s", cpp)

    assert "break;" in cpp, "Expected a native C++ break"
    assert "_can_continue" not in cpp, "Native break needs no flag"
    _, guard_idx, break_idx = _assert_materialized_jump_guard(cpp, "== 3", "break;")
    assert guard_idx < break_idx, "break must be inside its guard"


def test_cce_while_accumulate():
    cpp = _compile_to_cce(_while_accumulate_kernel)
    logging.info("\n=== test_cce_while_accumulate ===\n%s", cpp)

    assert "while (" in cpp, "Expected a C++ while loop"
    assert "_can_continue" not in cpp, "No break/continue, so no flag"
    assert "TADD(" in cpp, "Body op should be emitted"
    assert "+ 1" in cpp, "i update present"


def test_cce_while_getval_condition():
    cpp = _compile_to_cce(_while_getval_condition_kernel)
    logging.info("\n=== test_cce_while_getval_condition ===\n%s", cpp)

    lock_index = cpp.index("get_buf(PIPE_S")
    getval_index = cpp.index(".GetValue(", lock_index)
    unlock_index = cpp.index("rls_buf(PIPE_S", getval_index)
    assert lock_index < getval_index < unlock_index


def test_cce_for_continue():
    cpp = _compile_to_cce(_for_continue_kernel)
    logging.info("\n=== test_cce_for_continue ===\n%s", cpp)

    assert "for (" in cpp, "Expected a C++ for loop"
    assert "_can_continue" not in cpp, "Native continue needs no flag"
    assert "continue;" in cpp, "Expected a native C++ continue"
    _assert_materialized_jump_guard(cpp, "== 2", "continue;")
    assert re.search(r"\bacc_\d+\b", cpp), "Loop-carried acc threaded through the for loop"


def test_cce_for_break():
    cpp = _compile_to_cce(_for_break_kernel)
    logging.info("\n=== test_cce_for_break ===\n%s", cpp)

    assert "for (" in cpp, "Expected a C++ for loop"
    assert "_can_continue" not in cpp, "Native break needs no flag"
    assert "break;" in cpp, "Expected a native C++ break"
    _assert_materialized_jump_guard(cpp, "== 3", "break;")


def test_cce_loop_tuple_return_propagates_as_one_value():
    cpp = _compile_to_cce(_loop_tuple_kernel)

    assert "TADD(bundle_1__item_2, bundle_1__item_2, bundle_1__item_2);" in cpp
    assert "bundle_1[" not in cpp
    assert "TADD(bundle_0," not in cpp


def test_cce_while_body_and_result_use_independent_slots():
    cpp = _compile_to_cce(_while_independent_body_result_kernel)
    body = cpp.split("#if defined(__DAV_VEC__)")[0]
    expected = dedent(
        """\
        #if defined(__DAV_CUBE__)
        #include <pypto_tprint.h>
        __aicore__ inline void _while_independent_body_result_kernel_impl_cube(bool flag_0, int64_t result_value_0)
        {

            auto value_0 = 0;
            int32_t value_1 = 0;
            int64_t value_4;

            while (true) {
                if (flag_0) {
                    auto value_2 = (result_value_0 + 0);
                    value_4 = value_2;
                    break;
                } else {
                }
                auto value_3 = 2;
                value_1 = 2;
                continue;
            }
            auto _test_result_0 = (value_4 + 1);
            return;
        }
        #endif

        """
    )
    assert body == expected


def _integer_sqrt(x):
    if x < 2:
        return x
    res = x
    nxt = (res + 1) // 2
    while nxt < res:
        res = nxt
        nxt = (res + x // res) // 2
    return res


@pl.jit
def _while_isqrt_carry_kernel(x: pl.DT_INT64, out: pl.Tensor[[1, 3], pl.DT_INT64]):
    with pl.section_vector():
        out[0, 0] = _integer_sqrt(x)


@pl.jit
def _while_swap_carry_kernel(x: pl.DT_INT64, out: pl.Tensor[[1, 3], pl.DT_INT64]):
    with pl.section_vector():
        left = x
        right = x + 1
        steps = 0
        # A bounded guard makes broken parallel copies fail without hanging a test.
        while right != x and steps < 4:
            old_left = left
            left = right
            right = old_left
            steps += 1
            continue
        out[0, 0] = left
        out[0, 1] = right
        out[0, 2] = steps


@pl.jit
def _for_swap_carry_kernel(x: pl.DT_INT64, out: pl.Tensor[[1, 3], pl.DT_INT64]):
    with pl.section_vector():
        left = x
        right = x + 1
        for i in pl.range(3):
            old_left = left
            left = right
            right = old_left
            if i == 2:
                break
            continue
        out[0, 0] = left
        out[0, 1] = right


@pytest.fixture(scope="module")
def scalar_loop_binary(tmp_path_factory):
    """Execute the scalar CCE as C++ to check values independently of an NPU."""
    build = tmp_path_factory.mktemp("scalar_loop_cce")
    (build / "pypto_tprint.h").write_text("")
    kernels = (_while_isqrt_carry_kernel, _while_swap_carry_kernel, _for_swap_carry_kernel)
    source = build / "loops.cpp"
    source.write_text(
        "#include <cstdint>\n#include <cstdio>\n#include <cstdlib>\n"
        "#define __aicore__\n#define __gm__\n#define __DAV_VEC__\n"
        + "\n".join(_compile_to_cce(kernel) for kernel in kernels)
        + dedent(
            """
            int main(int argc, char** argv) {
                if (argc != 3) return 2;
                int64_t x = std::strtoll(argv[2], nullptr, 10);
                int64_t out[3] = {};
                switch (std::atoi(argv[1])) {
                    case 0: _while_isqrt_carry_kernel_impl(x, out); break;
                    case 1: _while_swap_carry_kernel_impl(x, out); break;
                    case 2: _for_swap_carry_kernel_impl(x, out); break;
                    default: return 2;
                }
                for (auto value : out) std::printf("%lld ", static_cast<long long>(value));
                return 0;
            }
            """
        )
    )
    binary = build / "loops"
    subprocess.run(
        ["g++", "-std=c++17", "-O2", f"-I{build}", str(source), "-o", str(binary)],
        check=True, capture_output=True, text=True, timeout=30,
    )
    return binary


@pytest.mark.parametrize("x", [0, 1, 2, 3, 4, 8, 9, 15, 16, 31, 32, 2147395600])
def test_cce_while_isqrt_values(scalar_loop_binary, x):
    result = subprocess.run(
        [str(scalar_loop_binary), "0", str(x)], check=True, capture_output=True, text=True, timeout=5
    )
    assert [int(value) for value in result.stdout.split()] == [math.isqrt(x), 0, 0]


@pytest.mark.parametrize("kind, expected", [(1, [8, 7, 1]), (2, [8, 7, 0])])
def test_cce_loop_swap_values(scalar_loop_binary, kind, expected):
    result = subprocess.run(
        [str(scalar_loop_binary), str(kind), "7"], check=True, capture_output=True, text=True, timeout=5
    )
    assert [int(value) for value in result.stdout.split()] == expected
