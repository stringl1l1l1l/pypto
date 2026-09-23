#!/usr/bin/env python3
# coding: utf-8
# ruff: noqa: E501
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import re
from textwrap import dedent

import pypto_pro.language as pl
import pytest


def _compile_to_cce(kernel) -> str:
    from pypto_pro.runtime.jit import _assemble_cv_source, _parse_and_codegen_targets

    cube, vector = _parse_and_codegen_targets(kernel.to_kernel_def(), "3510", "")
    return _assemble_cv_source(cube, vector).content


def _choose_extent(value):
    if value > 4:
        return value + 1
    return value + 2


def _choose_in_loop(value):
    index = 0
    while True:
        if index >= 2:
            return index
        index = index + 1


def _single_dynamic_pair_in_loop(value):
    while True:
        return value + 0, value + 1


@pl.jit(auto_mutex=False)
def _single_dynamic_pair_in_loop_kernel(value: pl.DT_INT64):
    first, second = _single_dynamic_pair_in_loop(value)
    total = first + second  # noqa: F841


@pl.jit(auto_mutex=False)
def _static_if_break_kernel(n: pl.DT_INT64):
    for i in pl.range(n):
        if True:
            break
        invalid = (1, 2)[3]  # noqa: F841


# Both helpers return from two sites so their result has to travel through a merge variable.
# A helper whose only return is its final top-level statement is expanded straight instead,
# with no merge slot for these tests to inspect.
def _pair(first, second):
    if first > second:
        return second, first
    return first, second


def _bundle(first, second):
    if first > second:
        return (second, first), False
    return (first, second), True


def _consume_pair(bundle, index):
    local_values = bundle[0]
    for _i in pl.range(0, local_values[index], 1):
        pl.system.bar_all()
    return


@pl.jit
def _inline_return_kernel(a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16]):
    extent = _choose_extent(_choose_in_loop(a.shape[0]))
    start, stop = _pair(0, extent)
    bundle = _bundle(start, stop)
    _consume_pair(bundle, 0)
    _consume_pair(bundle, 1)
    for _i in pl.range(start, stop, 1):
        pl.system.bar_all()


def test_cce_inline_multiple_returns_use_typed_uninitialized_slot():
    cpp = _compile_to_cce(_inline_return_kernel)

    declarations = re.findall(r"int64_t (__inline_\d+_return_val_\d+);", cpp)
    assert declarations
    for name in declarations:
        assert f"{name} = 0;" not in cpp
    assert not re.search(r"auto __inline_\d+_return_val_\d+ = __inline_\d+_return_val_\d+;", cpp)
    assert cpp.count("while (true)") >= 3
    assert ">= 2" in cpp
    assert cpp.count("break;") >= 6


def test_cce_inline_array_return_uses_one_backing_array():
    cpp = _compile_to_cce(_inline_return_kernel)

    declarations = set(re.findall(r"int64_t (__inline_\d+_return_val_\d+)\[2\];", cpp))
    assert declarations
    for name in declarations:
        assert re.search(rf"{name}\[0\] = .+\[0\];", cpp)
        assert re.search(rf"{name}\[1\] = .+\[1\];", cpp)
        assert f"int64_t {name}_0;" not in cpp
        assert f"int64_t {name}_1;" not in cpp


def test_cce_single_dynamic_tuple_return_reads_the_loop_result():
    body = _cube_body(_compile_to_cce(_single_dynamic_pair_in_loop_kernel))
    expected = dedent(
        """\
        #if defined(__DAV_CUBE__)
        #include <pypto_tprint.h>
        __aicore__ inline void _single_dynamic_pair_in_loop_kernel_impl_cube(int64_t value_0)
        {

            int64_t __inline_0_return_val_3[2];

            while (true) {
                auto __inline_0_returned_0 = false;
                auto _expr_tmp_0_0 = (value_0 + 0);
                auto _expr_tmp_1_0 = (value_0 + 1);
                int64_t __inline_0_return_val_2[] = {static_cast<int64_t>(_expr_tmp_0_0), static_cast<int64_t>(_expr_tmp_1_0)};
                auto __inline_0_returned_1 = true;
                int64_t __inline_0_return_val_3__next[2];
                __inline_0_return_val_3__next[0] = __inline_0_return_val_2[0];
                __inline_0_return_val_3__next[1] = __inline_0_return_val_2[1];
                __inline_0_return_val_3[0] = __inline_0_return_val_3__next[0];
                __inline_0_return_val_3[1] = __inline_0_return_val_3__next[1];
                break;
            }
            auto first_0 = __inline_0_return_val_3[0];
            auto second_0 = __inline_0_return_val_3[1];
            auto total_0 = (first_0 + second_0);
            return;
        }
        #endif

        """
    )
    assert body == expected


def test_cce_static_if_break_terminates_the_enclosing_loop_block():
    body = _cube_body(_compile_to_cce(_static_if_break_kernel))
    expected = dedent(
        """\
        #if defined(__DAV_CUBE__)
        #include <pypto_tprint.h>
        __aicore__ inline void _static_if_break_kernel_impl_cube(int64_t n_0)
        {

            for (int64_t i__iterator_0 = 0; i__iterator_0 < n_0; i__iterator_0 += 1) {
                break;
            }
            return;
        }
        #endif

        """
    )
    assert body == expected


def test_cce_inline_aggregate_return_flattens_only_aggregate_fields():
    cpp = _compile_to_cce(_inline_return_kernel)

    nested_arrays = set(re.findall(r"int64_t (__inline_\d+_return_val_\d+__item_0)\[2\];", cpp))
    assert nested_arrays
    for name in nested_arrays:
        aggregate = name.removesuffix("__item_0")
        assert f"bool {aggregate}__item_1;" in cpp
        assert f"int64_t {name}_0;" not in cpp
        assert f"int64_t {name}_1;" not in cpp


def test_cce_inline_array_aliases_keep_the_phi_backing_array():
    cpp = _compile_to_cce(_inline_return_kernel)

    nested_arrays = set(re.findall(r"int64_t (__inline_\d+_return_val_\d+__item_0)\[2\];", cpp))
    assert len(nested_arrays) == 1
    [array_name] = nested_arrays
    assert f"{array_name}[0]" in cpp
    assert f"{array_name}[1]" in cpp
    assert not re.search(r"(?:const )?int64_t __inline_\d+_local_values_\d*\[", cpp)


# ---------------------------------------------------------------------------
# Structural invariants every inline-return kernel must satisfy
# ---------------------------------------------------------------------------

_RETURN_SLOT_RE = re.compile(r"\b(__inline_\d+_return_val_\d+(?:__item_\d+)*)\b")
# The `returned` flag is a merge slot too, and shares the declaration block with the value slot.
_ANY_SLOT_RE = re.compile(r"\b__inline_\d+_\w*?_\d+(?:_\d+)*\b")


def _cube_body(cpp: str) -> str:
    """The Cube variant alone: a kernel emits Cube and Vector copies of the same code."""
    return cpp.split("#if defined(__DAV_VEC__)")[0]


def _declares(line: str, pattern: re.Pattern) -> bool:
    """True when the line declares a merge slot rather than reading, writing or using one.

    A declaration names a type before the slot and carries nothing after it but an optional
    array extent and an optional initializer. A write-back (`slot = value;`) leads with the
    slot, a read (`x = slot;`) puts it after the `=`, and a use (`TLOAD(slot, src);`) is
    followed by further arguments.
    """
    match = pattern.search(line)
    if match is None:
        return False
    prefix, suffix = line[:match.start()], line[match.end():]
    return (prefix.strip() != "" and "=" not in prefix
            and re.fullmatch(r"(?:\[\d+\])?(?: = .*)?;", suffix) is not None)


def _assert_merge_slot_declaration_invariants(cpp: str) -> None:
    """Each return slot is declared once, immediately before the structure that merges it.

    Both halves matter. A slot declared inside the body it merges would die at the closing
    brace, and one declared further out than necessary would leak into a scope with no
    business seeing it — so every level splices its own declarations in front of its own
    structure rather than hoisting them to function scope.
    """
    lines = _cube_body(cpp).splitlines()
    declarations = [
        (i, _RETURN_SLOT_RE.search(line).group(1))
        for i, line in enumerate(lines)
        if _declares(line, _RETURN_SLOT_RE) and " = " not in line
    ]
    assert declarations, "expected at least one inline-return merge slot declaration"

    names = [name for _, name in declarations]
    assert len(names) == len(set(names)), f"a merge slot is declared more than once: {names}"

    for index, name in declarations:
        rest = [
            line.strip()
            for line in lines[index + 1:]
            if line.strip() and not _declares(line, _ANY_SLOT_RE)
        ]
        assert rest, f"{name} is declared with nothing after it"
        assert re.match(r"while \(true\)|for \(|if \(", rest[0]), (
            f"{name} is not declared immediately before the structure that merges it; "
            f"the next statement is: {rest[0]}"
        )


def test_cce_merge_var_declarations_precede_their_control_flow():
    _assert_merge_slot_declaration_invariants(_compile_to_cce(_inline_return_kernel))


def test_cce_inner_loop_slots_are_not_hoisted_to_function_scope():
    cpp = _compile_to_cce(_inline_return_kernel)

    # Each level splices its own declarations in front of its own structure, so a slot
    # belonging to a nested loop stays indented inside the enclosing body rather than
    # being lifted all the way out to function scope.
    nested = [
        m
        for m in re.finditer(
            r"^(\s+)(?:int64_t|bool) (__inline_\d+_return_val_\d+)(?: = \2)?;$",
            cpp,
            re.MULTILINE,
        )
        if len(m.group(1)) > 4
    ]
    assert nested, "expected at least one merge slot declared inside an enclosing body"


# ---------------------------------------------------------------------------
# Return-value kinds
#
# Each helper gets a kernel of its own so its expansion is always inline id 0, and so one
# unsupported shape cannot take the other kernels down with it. Every helper returns from
# two sites under a condition derived from a runtime value (`a.shape[0]`), because a
# compile-time condition folds the branch away and no merge variable is ever created.
# ---------------------------------------------------------------------------


def _ret_named_tuple(value):
    if value > 4:
        return pl.make_tuple(lo=value, hi=value + 1)
    return pl.make_tuple(lo=0, hi=value)


def _ret_struct(value):
    if value > 4:
        return pl.struct("RetS", v=value, w=value + 1)
    return pl.struct("RetS", v=0, w=value)


def _ret_struct_in_tuple(value):
    if value > 4:
        return pl.struct("NestedS", v=value, w=value + 1), True
    return pl.struct("NestedS", v=0, w=value), False


def _ret_struct_array(value):
    if value > 4:
        return pl.struct("ArrS", v=value, w=value + 1), pl.struct("ArrS", v=value + 2, w=value + 3)
    return pl.struct("ArrS", v=0, w=value), pl.struct("ArrS", v=1, w=value)


def _ret_tile(value, tile_a, tile_b):
    if value > 4:
        return tile_a
    return tile_b


def _ret_void_twice(value):
    if value > 4:
        pl.system.bar_all()
        return
    pl.system.bar_all()
    return


# Every kernel below consumes the returned value in a statement of its own before using it
# again, so the merge slot has to survive past its first read: a slot that was folded into
# its single use, or clobbered on the way out of the wrapper, would show up here.


@pl.jit
def _named_tuple_return_kernel(a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16]):
    bounds = _ret_named_tuple(a.shape[0])
    total = bounds.lo + bounds.hi
    for _i in pl.range(bounds.lo, total, 1):
        pl.system.bar_all()


@pl.jit
def _struct_return_kernel(a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16]):
    bounds = _ret_struct(a.shape[0])
    total = bounds.v + bounds.w
    for _i in pl.range(bounds.v, total, 1):
        pl.system.bar_all()


@pl.jit
def _struct_in_tuple_return_kernel(a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16]):
    bundle = _ret_struct_in_tuple(a.shape[0])
    bounds = bundle[0]
    total = bounds.v + bounds.w
    for _i in pl.range(bounds.v, total, 1):
        pl.system.bar_all()


@pl.jit
def _struct_array_return_kernel(a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16]):
    first, second = _ret_struct_array(a.shape[0])
    total = first.v + second.w
    for _i in pl.range(first.w, total, 1):
        pl.system.bar_all()


@pl.jit
def _tile_return_kernel(a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16]):
    tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile(tile_type, addr=0x0000)
    tile_b = pl.make_tile(tile_type, addr=0x4000)
    picked = _ret_tile(a.shape[0], tile_a, tile_b)
    pl.load(picked, a, [0, 0])
    pl.store(a, picked, [0, 0])


@pl.jit
def _void_return_kernel(a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16]):
    _ret_void_twice(a.shape[0])
    for _i in pl.range(0, a.shape[0], 1):
        pl.system.bar_all()


def test_cce_named_tuple_return_uses_aggregate_slots():
    body = _cube_body(_compile_to_cce(_named_tuple_return_kernel))

    # Named-tuple fields in IRDebugInfo distinguish this aggregate from a
    # homogeneous positional tuple, so codegen flattens it into named leaf slots.
    assert re.search(r"^\s*int64_t __inline_0_return_val_4__item_0;$", body, re.MULTILINE)
    assert re.search(r"^\s*int64_t __inline_0_return_val_4__item_1;$", body, re.MULTILINE)
    assert not re.search(r"__inline_0_return_val_4\[", body)
    assert not re.search(r"^\s*\S+ __inline_0_return_val_4;$", body, re.MULTILINE)

    # Both branches write the fields, and `.lo` / `.hi` read back through them.
    assert body.count("__inline_0_return_val_4__item_0 = ") == 2
    assert body.count("__inline_0_return_val_4__item_1 = ") == 2
    assert "= __inline_0_return_val_4__item_0;" in body
    assert "= __inline_0_return_val_4__item_1;" in body


def test_cce_struct_return_merges_as_a_single_object():
    body = _cube_body(_compile_to_cce(_struct_return_kernel))

    # A struct has a C++ type name of its own, so the merge slot is one object assigned
    # whole — not an array, and not flattened into per-field leaf slots.
    assert "class RetS {" in body
    assert re.search(r"^\s*RetS __inline_0_return_val_4;$", body, re.MULTILINE)
    assert not re.search(r"__inline_0_return_val_4_\d", body)
    assert not re.search(r"__inline_0_return_val_4\[", body)

    writes = re.findall(r"__inline_0_return_val_4 = (\S+);", body)
    assert len(writes) == 2, f"expected one whole-object write per branch, got {writes}"
    for source in writes:
        assert re.fullmatch(r"__inline_0_return_val_\d+(?:__next)?", source), source
    # Field reads go through the merged object, not through a copy of the branch value.
    assert "__inline_0_return_val_4.v" in body
    assert "__inline_0_return_val_4.w" in body


def test_cce_struct_inside_a_returned_tuple_stays_one_object():
    body = _cube_body(_compile_to_cce(_struct_in_tuple_return_kernel))

    # The tuple is heterogeneous, so it flattens into one leaf slot per element -- but a struct
    # element is a leaf, not something to take apart further: it has a C++ type name of its own.
    assert "class NestedS {" in body
    assert re.search(r"^\s*NestedS __inline_0_return_val_4__item_0;$", body, re.MULTILINE)
    assert re.search(r"^\s*bool __inline_0_return_val_4__item_1;$", body, re.MULTILINE)
    assert not re.search(r"__inline_0_return_val_4__item_0__item_\d", body)
    assert "__inline_0_return_val_4__item_0.v" in body


def test_cce_struct_array_return_uses_one_backing_array():
    body = _cube_body(_compile_to_cce(_struct_array_return_kernel))

    # Both elements are the same struct, so the tuple is homogeneous and merges through one
    # backing array of that struct type -- copied element by element, not flattened into a
    # leaf slot per element the way a heterogeneous tuple is.
    assert "class ArrS {" in body
    assert re.search(r"^\s*ArrS __inline_0_return_val_4\[2\];$", body, re.MULTILINE)
    assert not re.search(r"__inline_0_return_val_4_\d", body)
    assert body.count("__inline_0_return_val_4[0] = ") == 2
    assert body.count("__inline_0_return_val_4[1] = ") == 2
    # Field reads go through the merged array, not through a copy of either branch value.
    assert "__inline_0_return_val_4[0].v" in body
    assert "__inline_0_return_val_4[1].w" in body


def test_cce_tile_return_merges_through_a_typed_slot():
    body = _cube_body(_compile_to_cce(_tile_return_kernel))

    # The declaration needs the tile type and nothing else: no constructor arguments and no
    # TASSIGN, because the slot is immediately overwritten by a whole-object copy of one of
    # the branch tiles, which carries its own valid shape and address.
    declaration = re.search(r"^\s*(Tile<[^;]*>) __inline_0_return_val_4;$", body, re.MULTILINE)
    assert declaration, "expected the tile merge slot to be declared from its type alone"
    assert "TASSIGN(__inline_0_return_val_4" not in body

    writes = re.findall(r"__inline_0_return_val_4 = (\S+);", body)
    assert sorted(writes) == ["tile_a_0", "tile_b_0"]
    # The merged tile, not either branch tile, is what the load targets.
    assert "TLOAD(__inline_0_return_val_4, " in body


def test_cce_void_returns_materialize_no_slot():
    body = _cube_body(_compile_to_cce(_void_return_kernel))

    # Bare returns carry a real NoneType value for return-type merging. NoneType has no C++
    # representation, so only framework-generated mutex companions may remain in the source.
    assert not any(
        "__inline_0_return_val" in line and "__mutexid" not in line
        for line in body.splitlines()
    )
    # The wrapper and its early exits are still emitted.
    assert "while (true)" in body
    assert body.count("break;") >= 2


# ---------------------------------------------------------------------------
# Control-flow shapes around the return
# ---------------------------------------------------------------------------


def _for_if_return(limit):
    for index in pl.range(0, limit, 1):
        if index >= 2:
            return index
    return limit


def _if_for_return(flag, limit):
    if flag > 4:
        for index in pl.range(0, limit, 1):
            if index >= 3:
                return index
        return 0
    return limit


def _then_only_return(value):
    if value > 4:
        return value + 1
    return value


def _else_only_return(value):
    if value > 4:
        value = value + 1
    else:
        return 0
    return value


def _both_branches_return(value):
    # The explicit `else` is what makes this shape differ from _then_only_return: both edges
    # leave the wrapper, so the `if` merges nothing that anyone can reach.
    if value > 4:
        return value + 1
    else:
        return value - 1


def _make_shape_kernel(helper):
    @pl.jit
    def kernel(a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16]):
        limit = helper(a.shape[0])
        doubled = limit + limit
        for _i in pl.range(limit, doubled, 1):
            pl.system.bar_all()

    return kernel


@pl.jit
def _for_if_return_kernel(a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16]):
    limit = _for_if_return(a.shape[0])
    doubled = limit + limit
    for _i in pl.range(limit, doubled, 1):
        pl.system.bar_all()


@pl.jit
def _if_for_return_kernel(a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16]):
    limit = _if_for_return(a.shape[0], a.shape[1])
    doubled = limit + limit
    for _i in pl.range(limit, doubled, 1):
        pl.system.bar_all()


_then_only_return_kernel = _make_shape_kernel(_then_only_return)
_both_branches_return_kernel = _make_shape_kernel(_both_branches_return)
_else_only_return_kernel = _make_shape_kernel(_else_only_return)


def test_cce_for_nested_if_return_ignores_empty_retval_init():
    body = _cube_body(_compile_to_cce(_for_if_return_kernel))

    assert re.search(r"^\s*int64_t __inline_0_return_val_6;$", body, re.MULTILINE)
    assert "Unknown" not in body
    assert "auto doubled_0 = (__inline_0_return_val_6 + __inline_0_return_val_6);" in body


def test_cce_if_nested_for_return_ignores_empty_retval_init():
    body = _cube_body(_compile_to_cce(_if_for_return_kernel))

    assert re.search(r"^\s*int64_t __inline_0_return_val_7;$", body, re.MULTILINE)
    assert "Unknown" not in body
    assert "auto doubled_0 = (__inline_0_return_val_7 + __inline_0_return_val_7);" in body


def test_cce_then_only_return_leaves_the_other_edge_alone():
    body = _cube_body(_compile_to_cce(_then_only_return_kernel))

    # Only the then branch returns; the fall-through path writes the slot after the `if`.
    # Neither edge may emit a self-assignment, which is what a back edge carrying an
    # unmodified value would degenerate into.
    assert not re.search(r"\b(\w+) = \1;", body)
    assert body.count("__inline_0_return_val_4 = ") == 2


def test_cce_both_branches_return_write_one_slot():
    body = _cube_body(_compile_to_cce(_both_branches_return_kernel))

    assert len(re.findall(r"^\s*int64_t __inline_0_return_val_4;$", body, re.MULTILINE)) == 1
    # Both branches terminate, so canonicalization keeps their writes and removes the
    # unreachable merge tail.
    assert body.count("__inline_0_return_val_4 = ") == 2
    assert not re.search(r"\b(\w+) = \1;", body)


def test_cce_else_only_return_resolves_the_slot_type():
    body = _cube_body(_compile_to_cce(_else_only_return_kernel))

    # Only the else branch writes the return slot; the then branch leaves it at the
    # `None` sentinel the lowering seeds it with. BuildIfPhiOutputs must take the
    # phi's type from the edge that carries a value, or the slot has no type to declare.
    assert re.search(r"^\s*int64_t __inline_0_return_val_5;$", body, re.MULTILINE)
    assert not re.search(r"\bUnknown\b", body)
    # Both exits write the slot: the early `return 0` and the fall-through `return value`.
    assert body.count("__inline_0_return_val_5 = ") == 2


@pytest.mark.parametrize(
    "kernel",
    [
        _named_tuple_return_kernel,
        _struct_return_kernel,
        _struct_in_tuple_return_kernel,
        _struct_array_return_kernel,
        _tile_return_kernel,
        _then_only_return_kernel,
        _else_only_return_kernel,
        _both_branches_return_kernel,
    ],
    ids=[
        "named_tuple", "struct", "struct_in_tuple", "struct_array", "tile",
        "then_only", "else_only", "both_branches",
    ],
)
def test_cce_merge_slots_keep_their_declaration_invariants(kernel):
    _assert_merge_slot_declaration_invariants(_compile_to_cce(kernel))
