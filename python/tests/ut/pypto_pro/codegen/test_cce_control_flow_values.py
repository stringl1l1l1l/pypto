# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Execute scalar CCE to verify control-flow values against Python semantics."""

import subprocess

import pypto_pro.language as pl
import pytest


@pl.jit
def named_for(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[8], pl.DT_INT64]):
    with pl.section_vector():
        state = pl.make_tuple(a=x, b=x + 1)
        for i in pl.range(n):
            state = pl.make_tuple(a=state.b, b=state.a)
        out[0] = state.a
        out[1] = state.b


@pl.jit
def nested_for(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[8], pl.DT_INT64]):
    with pl.section_vector():
        state = pl.make_tuple(a=pl.make_tuple(v=x), b=pl.make_tuple(v=x + 1))
        for i in pl.range(n):
            state = pl.make_tuple(a=state.b, b=state.a)
        out[0] = state.a.v
        out[1] = state.b.v


@pl.jit
def bound_alias(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[8], pl.DT_INT64]):
    with pl.section_vector():
        count = n
        for i in pl.range(1):
            count = count + 1
        total = 0
        for j in pl.range(count):
            count = count - 1
            total = total + 1
        out[0] = total
        out[1] = count


@pl.jit
def induction(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[8], pl.DT_INT64]):
    with pl.section_vector():
        i = x
        for i in pl.range(n):
            if i == 2:
                break
        out[0] = i


def multi(a, b, n):
    for i in pl.range(n):
        if i == 2:
            return b, a
        a, b = b, a
    return a, b


@pl.jit
def multi_return(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[8], pl.DT_INT64]):
    with pl.section_vector():
        a, b = multi(x, x + 1, n)
        out[0] = a
        out[1] = b


@pl.jit
def if_else(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[8], pl.DT_INT64]):
    with pl.section_vector():
        a = x
        b = x + 1
        for i in pl.range(n):
            if i % 2 == 0:
                saved = a
                a = b
                b = saved
            else:
                a, b = b, a
        out[0] = a
        out[1] = b


@pl.jit
def nested_while(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[8], pl.DT_INT64]):
    with pl.section_vector():
        state = pl.make_tuple(a=pl.make_tuple(v=x), b=pl.make_tuple(v=x + 1), count=0)
        while state.count < n:
            state = pl.make_tuple(a=state.b, b=state.a, count=state.count + 1)
            continue
        out[0] = state.a.v
        out[1] = state.b.v
        out[2] = state.count


@pl.jit
def nested_break(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[8], pl.DT_INT64]):
    with pl.section_vector():
        state = pl.make_tuple(a=pl.make_tuple(v=x), b=pl.make_tuple(v=x + 1))
        for i in pl.range(n):
            state = pl.make_tuple(a=state.b, b=state.a)
            break
        out[0] = state.a.v
        out[1] = state.b.v


@pl.jit
def nested_if(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[8], pl.DT_INT64]):
    with pl.section_vector():
        state = pl.make_tuple(a=pl.make_tuple(v=x), b=pl.make_tuple(v=x + 1))
        for i in pl.range(n):
            if i % 2 == 0:
                state = pl.make_tuple(a=state.b, b=state.a)
            else:
                state = pl.make_tuple(a=state.b, b=state.a)
        out[0] = state.a.v
        out[1] = state.b.v


@pl.jit
def iterator_collision(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[8], pl.DT_INT64]):
    with pl.section_vector():
        i__iterator = x
        total = 0
        for i in pl.range(n):
            total = total + i__iterator
        out[0] = total
        out[1] = i__iterator


@pl.jit
def nested_saved(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[8], pl.DT_INT64]):
    with pl.section_vector():
        state = pl.make_tuple(a=pl.make_tuple(v=x), b=pl.make_tuple(v=x + 1))
        for i in pl.range(n):
            old = state.a
            state = pl.make_tuple(a=state.b, b=old)
        out[0] = state.a.v
        out[1] = state.b.v


@pl.jit
def composite_rebind(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[8], pl.DT_INT64]):
    with pl.section_vector():
        state = pl.make_tuple(a=(x, x + 1), b=(x + 2, x + 3))
        for i in pl.range(n):
            state = pl.make_tuple(a=state.b, b=state.a)
        out[0] = state.a[0]
        out[1] = state.a[1]
        out[2] = state.b[0]
        out[3] = state.b[1]


@pl.jit
def nested_induction(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[8], pl.DT_INT64]):
    with pl.section_vector():
        i = x
        for j in pl.range(n):
            for i in pl.range(3):
                pass
        out[0] = i


@pl.jit
def branch_induction(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[8], pl.DT_INT64]):
    with pl.section_vector():
        i = x
        if n > 0:
            for i in pl.range(3):
                pass
        else:
            for i in pl.range(5):
                pass
        out[0] = i


@pl.jit
def while_induction(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[8], pl.DT_INT64]):
    with pl.section_vector():
        i = 0
        steps = 0
        while i < n and steps < 4:
            for i in pl.range(n + 1):
                pass
            steps += 1
        out[0] = i
        out[1] = steps


def hetero_multi_helper(state, n):
    for i in pl.range(n):
        state = pl.make_tuple(a=state.b, b=state.a)
        if i == 1:
            return pl.make_tuple(first=state.b, second=state.a)
    return pl.make_tuple(first=state.a, second=state.b)


@pl.jit
def hetero_multi(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[8], pl.DT_INT64]):
    with pl.section_vector():
        state = pl.make_tuple(a=pl.make_tuple(v=x), b=pl.make_tuple(v=x + 1))
        a, b = hetero_multi_helper(state, n)
        out[0] = a.v
        out[1] = b.v


@pl.jit
def cycle_three(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[8], pl.DT_INT64]):
    with pl.section_vector():
        a = x
        b = x + 1
        c = x + 2
        for i in pl.range(n):
            old = a
            a = b
            b = c
            c = old
            if i % 2 == 0:
                continue
        out[0] = a
        out[1] = b
        out[2] = c


def early_return_helper(a, b, n):
    for i in pl.range(n):
        j = 0
        while j < n:
            if j == 1:
                if i == 0:
                    return b, a, i + j
                else:
                    return a, b, i - j
            a, b = b, a
            j += 1
    return a, b, n


@pl.jit
def nested_returns(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[8], pl.DT_INT64]):
    with pl.section_vector():
        a, b, count = early_return_helper(x, x + 1, n)
        out[0] = a
        out[1] = b
        out[2] = count


@pl.jit
def iterator_body_collision(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[8], pl.DT_INT64]):
    with pl.section_vector():
        total = 0
        for i in pl.range(n):
            i__iterator = x
            for j in pl.range(2):
                total += i__iterator + i
        out[0] = total


def collision_return_val(return_val, other):
    return return_val, other


@pl.jit
def return_name_collision(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[8], pl.DT_INT64]):
    with pl.section_vector():
        a, b = collision_return_val(x, x + 1)
        out[0] = a
        out[1] = b


def collision_returned(a, b, n):
    returned = n
    for i in pl.range(2):
        if i == 1:
            return a, b, returned
        a, b = b, a
    return a, b, returned


@pl.jit
def returned_name_collision(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[8], pl.DT_INT64]):
    with pl.section_vector():
        a, b, c = collision_returned(x, x + 1, n)
        out[0] = a
        out[1] = b
        out[2] = c


@pl.jit
def ternary_collision(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[8], pl.DT_INT64]):
    with pl.section_vector():
        _ifexpr_tmp_0 = x
        selected = 10 if n > 0 else 20
        out[0] = _ifexpr_tmp_0
        out[1] = selected


def side_effect_predicate(out):
    out[0] = out[0] + 1
    return out[0] > 0


@pl.jit
def ternary_condition_twice(x: pl.DT_INT64, n: pl.DT_INT64, out: pl.Tensor[[8], pl.DT_INT64]):
    with pl.section_vector():
        selected = x if side_effect_predicate(out) else n
        out[1] = selected


def _swap_expected(x, n):
    return [x + n % 2, x + 1 - n % 2]


CASES = [
    (named_for, _swap_expected),
    (ternary_condition_twice, lambda x, n:[1, x]),
    (return_name_collision, lambda x, n:[x, x + 1]),
    (returned_name_collision, lambda x, n:[x + 1, x, n]),
    (nested_for, _swap_expected),
    (nested_saved, _swap_expected),
    (nested_if, _swap_expected),
    (if_else, _swap_expected),
    (nested_while, lambda x, n:[*_swap_expected(x, n), n]),
    (nested_break, lambda x, n:_swap_expected(x, min(n, 1))),
    (composite_rebind, lambda x, n:[x + (2 * n + k) % 4 for k in range(4)]),
    (bound_alias, lambda x, n:[n + 1, 0]),
    (induction, lambda x, n:[min(n - 1, 2) if n else x]),
    (nested_induction, lambda x, n:[2 if n else x]),
    (branch_induction, lambda x, n:[2 if n else 4]),
    (while_induction, lambda x, n:[n, int(n > 0)]),
    (iterator_body_collision, lambda x, n:[2 * n * x + n * (n - 1)]),
    (multi_return, lambda x, n:_swap_expected(x, min(n, 3))),
    (hetero_multi, lambda x, n:_swap_expected(x, min(n, 1))),
    (cycle_three, lambda x, n:[x + (n + k) % 3 for k in range(3)]),
    (nested_returns, lambda x, n:[*_swap_expected(x, int(n == 1)), min(n, 1)]),
]


@pytest.fixture(scope="module", params=CASES, ids=lambda case: case[0].__name__)
def control_flow_binary(request, tmp_path_factory):
    from pypto_pro.runtime.jit import _assemble_cv_source, _parse_and_codegen_targets

    kernel, expected = request.param
    build = tmp_path_factory.mktemp(kernel.__name__)
    (build / "pypto_tprint.h").write_text("")
    cube, vector = _parse_and_codegen_targets(kernel.to_kernel_def(), "3510", "")
    source = build / "kernel.cpp"
    source.write_text(
        "#include <cstdint>\n#include <cstdio>\n#include <cstdlib>\n"
        "#define __aicore__\n#define __gm__\n#define __DAV_VEC__\n"
        + _assemble_cv_source(cube, vector).content
        + "\nint main(int argc, char** argv) {\n"
        + "  if (argc != 3) return 2;\n"
        + "  int64_t out[8] = {};\n"
        + f"  {kernel.__name__}_impl(std::strtoll(argv[1], nullptr, 10), "
        + "std::strtoll(argv[2], nullptr, 10), out);\n"
        + '  for (auto v : out) std::printf("%lld ", static_cast<long long>(v));\n}\n'
    )
    binary = build / "kernel"
    subprocess.run(
        ["g++", "-std=c++17", "-O2", f"-I{build}", str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return binary, expected


@pytest.mark.parametrize("x", [-3, 0, 7])
@pytest.mark.parametrize("n", range(5))
def test_control_flow_values(control_flow_binary, x, n):
    binary, expected = control_flow_binary
    result = subprocess.run([str(binary), str(x), str(n)], check=True, capture_output=True, text=True, timeout=5)
    values = expected(x, n)
    assert [int(value) for value in result.stdout.split()] == values + [0] * (8 - len(values))
