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

"""How a compile-time error renders, across the shapes the framework raises them in.

Every error carries a first line -- ``[file:line][MODULE]:ErrCode: FXXXXX! Enum: CLASS::NAME. reason``
-- and, when the framework can place it, a location block quoting the user's kernel.
The cases below sample the raising shapes: a builder check with no location, a
parser check pointing at one keyword, one pointing at a positional argument, a
check raised inside a shared helper, a C++ check reaching Python through the
dispatch gate, and an interpreter error wrapped by the kernel entry.
"""

from __future__ import annotations

import pathlib
import re

from pypto_pro._errors import (
    InvalidArgument,
    InvalidOperation,
    InvalidShape,
    InvalidType,
    NameNotFound,
    OutOfRange,
    PyptoProError,
    RuntimeFailure,
)
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.platform import NpuArch
import pytest

from pypto.pypto_impl import ir

# [file:line][MODULE]:ErrCode: FXXXXX! Enum: CLASS::NAME. reason
_FIRST_LINE = re.compile(
    r"^\[(?P<origin>[^\[\]]+:\d+)\]\[(?P<module>PRO_[A-Z]+)\]:"
    r"ErrCode: F(?P<code>[0-9A-F]{5})! Enum: (?P<enum>[A-Za-z_:]+)\. (?P<reason>.+)$"
)
_LOC_LINE = re.compile(r"^  --> (?P<file>.+):(?P<line>\d+):(?P<col>\d+)$")


class Rendered:
    """The parts of a rendered error, so a test can assert on one of them."""

    def __init__(self, error: BaseException):
        self.error = error
        self.text = str(error)
        self.lines = self.text.split("\n")
        self.head = _FIRST_LINE.match(self.lines[0])
        loc = [_LOC_LINE.match(line) for line in self.lines]
        self.loc = next((m for m in loc if m), None)
        self.hint = next((line[8:] for line in self.lines if line.startswith("  hint: ")), None)

    @property
    def preview(self) -> list[str]:
        """The quoted source lines, without the gutter."""
        return [line.split("|", 1)[1] for line in self.lines if re.match(r"^\s*\d+ \|", line)]

    @property
    def caret_line(self) -> tuple[str, str] | None:
        """The caret line and the source line above it."""
        for index, line in enumerate(self.lines):
            if re.match(r"^\s*\|\s*\^+\s*$", line):
                return self.lines[index - 1], line
        return None

    def caret_under(self) -> str:
        """The source text the caret run covers, measured from each line's gutter."""
        pair = self.caret_line
        if pair is None:
            return ""
        quoted, caret = pair
        offset = caret.index("^") - caret.index("|")
        start = quoted.index("|") + offset
        return quoted[start:start + caret.count("^")]


def _render(kernel, section=ir.SectionKind.Vector) -> Rendered:
    """Parse *kernel* and return its error, or fail the test if it parses."""
    with pytest.raises(PyptoProError) as excinfo:
        kernel.to_kernel_def().parse_target_program(section)
    return Rendered(excinfo.value)


# ---------------------------------------------------------------------------
# The first line is mandatory and is rendered exactly once
# ---------------------------------------------------------------------------


def test_first_line_names_the_framework_source_the_module_and_the_code():
    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[-1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        pl.load(pl.make_tile(tile_type, addr=0), x, [0, 0])

    rendered = _render(kernel)
    assert rendered.head, f"first line does not match the spec: {rendered.lines[0]!r}"
    assert rendered.head["origin"].startswith("block_ops.py:")
    assert rendered.head["module"] == "PRO_IR"
    assert rendered.head["code"] == "0000E"
    assert rendered.head["enum"] == "ExternalError::INVALID_ARGUMENT"
    assert isinstance(rendered.error, InvalidArgument)


def test_the_code_in_the_first_line_agrees_with_the_exception_class():
    """The class name is what a traceback prints, so it has to name the same code."""

    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        pl.load(pl.make_tile(tile_type, addr=totally_undefined), x, [0, 0])  # noqa: F821

    rendered = _render(kernel)
    assert isinstance(rendered.error, NameNotFound)
    assert int(rendered.head["code"], 16) == int(NameNotFound.error_code) & 0xFFFFF


def test_only_one_first_line_is_rendered():
    """A wrapping raise quotes the inner reason, never the inner head."""

    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)
        pl.load(tile, x, [0, 0])
        pl.cast(tile, x)

    rendered = _render(kernel)
    assert rendered.text.count("ErrCode:") == 1, rendered.text


# ---------------------------------------------------------------------------
# The location block points into the user's kernel
# ---------------------------------------------------------------------------


def test_a_builder_check_outside_the_dispatch_gate_still_gets_a_location():
    """pl.TileType is a hand-written handler, so the op dispatcher publishes nothing.

    The expression parser publishes the location of every expression it walks, which
    is what covers the builders the dispatcher never reaches.
    """

    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[0, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        pl.load(pl.make_tile(tile_type, addr=0), x, [0, 0])

    rendered = _render(kernel)
    assert rendered.head
    assert rendered.loc, rendered.text
    assert "pl.TileType(" in rendered.caret_under()


def test_a_keyword_check_points_at_the_value_that_keyword_was_given():
    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        group = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=-1,
            mutex_ids=[0, 1],
        )
        pl.load(group.next(), x, [0, 0])

    rendered = _render(kernel, ir.SectionKind.Cube)
    assert rendered.loc, rendered.text
    assert rendered.caret_under() == "-1"
    assert "addrs=-1," in "".join(rendered.preview)


def test_a_positional_check_points_at_that_argument():
    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        _bad = pl.const("not a number", pl.DT_INT32)
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        pl.load(pl.make_tile(tile_type, addr=0), x, [0, 0])

    rendered = _render(kernel)
    assert isinstance(rendered.error, InvalidType)
    assert rendered.caret_under() == '"not a number"'


def test_the_location_line_names_this_test_file_and_the_offending_line():
    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)
        for _ in pl.range(0, 2**70):
            pl.load(tile, x, [0, 0])

    rendered = _render(kernel)
    assert rendered.loc
    assert rendered.loc["file"].endswith("test_error_format.py")
    own_source = pathlib.Path(__file__).read_text().split("\n")
    expected = next(i for i, line in enumerate(own_source, 1) if "2**70" in line and "range" in line)
    assert int(rendered.loc["line"]) == expected


def test_the_preview_quotes_the_offending_line_with_context():
    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)
        pl.load(tile, x, [0, 0])
        _bad = pl.simt.thread_idx()

    rendered = _render(kernel)
    assert isinstance(rendered.error, InvalidOperation)
    quoted = "".join(rendered.preview)
    assert "pl.simt.thread_idx()" in quoted
    assert "pl.load(tile, x, [0, 0])" in quoted, "the line before should be quoted as context"
    assert len(rendered.preview) > 1


def test_a_codegen_check_renders_its_location_block_like_a_parse_one():
    """A C++ check reaching Python from codegen carries the same block a parse check does."""
    from pypto_pro.runtime.jit import _parse_and_codegen_targets

    @pl.vector_function
    def unsqueeze_a_float_mask(in_a, t_u0):
        preg_f32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
        preg_u32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
        reg_a = vf.load_align(in_a, 0)
        cmp_mask = vf.ge(reg_a, 0.0, preg_f32)
        reg_dst = vf.unsqueeze(cmp_mask)
        vf.store_align(t_u0, reg_dst, preg_u32)

    @pl.jit()
    def kernel(a: pl.Tensor[[1, 64], pl.DT_FP32], out: pl.Tensor[[1, 64], pl.DT_UINT32]):
        tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        tu = pl.TileType(shape=[1, 64], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
        in_a = pl.make_tile(tf, addr=0)
        t_u0 = pl.make_tile(tu, addr=0x200)
        with pl.section_vector():
            pl.load(in_a, a, [0, 0])
            unsqueeze_a_float_mask(in_a, t_u0)
            pl.store(out, t_u0, [0, 0])

    with pytest.raises(PyptoProError) as excinfo:
        _parse_and_codegen_targets(kernel.to_kernel_def(), NpuArch.DAV_3510, "")
    rendered = Rendered(excinfo.value)

    # The C++ first line is passed through whole: its own file, module and code.
    assert rendered.head, rendered.lines[0]
    assert rendered.head["module"] == "PRO_CODEGEN"
    assert rendered.head["enum"] == "ExternalError::INVALID_TYPE"
    # The code names the class, here as on the parse path.
    assert isinstance(excinfo.value, InvalidType)
    # The location block is what codegen used to lack: a location, and source under it.
    assert rendered.loc
    assert rendered.loc["file"].endswith("test_error_format.py")
    assert "vf.unsqueeze(cmp_mask)" in "".join(rendered.preview)


# ---------------------------------------------------------------------------
# The shapes a check is raised in
# ---------------------------------------------------------------------------


def test_a_check_raised_inside_a_shared_helper_names_the_helper():
    """The first line names the framework source that raised, as the C++ side does."""

    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)
        for _ in pl.range(0, 2**70):
            pl.load(tile, x, [0, 0])

    rendered = _render(kernel)
    assert isinstance(rendered.error, OutOfRange)
    assert rendered.head["origin"].startswith("_range.py:")
    assert rendered.head["module"] == "PRO_PARSER"


def test_a_cpp_check_keeps_its_own_first_line_code_and_location():
    """A C++ check reaches Python whole: its file:line, its code, its DSL location.

    ``adds`` declares no signature, so the type gate leaves its operands alone and
    the C++ deduction is the one to refuse a Tensor where a Tile belongs.
    """

    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)
        pl.load(tile, x, [0, 0])
        pl.adds(tile, x, 1)

    rendered = _render(kernel)
    assert rendered.head, rendered.lines[0]
    # The origin names the file the check macro expands in: a .cpp call site,
    # or a shared op header such as op_common.h.
    assert rendered.head["origin"].split(":")[0].endswith((".cpp", ".h")), rendered.head["origin"]
    assert rendered.head["module"] == "PRO_IR"
    assert rendered.head["enum"] == "ExternalError::INVALID_TYPE"
    assert isinstance(rendered.error, InvalidType)
    assert rendered.loc, "the dispatch gate should have published the DSL location"


def test_an_interpreter_error_is_wrapped_with_a_code_of_its_own():
    """A TypeError the interpreter raised has no code, so the wrapper supplies one.

    ``row_sum`` declares no signature, so the argument gate lets the call through
    and the builder is the one to refuse it -- with a bare TypeError, which is the
    shape this test is about.
    """

    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)
        out = pl.make_tile(tile_type, addr=0x4000)
        pl.load(tile, x, [0, 0])
        pl.row_sum(out, tile, no_such_keyword=5)

    rendered = _render(kernel)
    assert isinstance(rendered.error, RuntimeFailure)
    assert rendered.head["module"] == "PRO_RUNTIME"
    assert rendered.head["code"] == "00003"
    assert "TypeError" in rendered.text, "the wrapper should name what it caught"
    assert rendered.loc, "the wrapper should still place the call that failed"


# ---------------------------------------------------------------------------
# The optional parts
# ---------------------------------------------------------------------------


def test_a_hint_is_rendered_on_its_own_line_after_the_location():
    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        pl.load(pl.make_tile(tile_type, addr=0, no_such_keyword=1), x, [0, 0])

    rendered = _render(kernel)
    assert isinstance(rendered.error, InvalidArgument)
    assert rendered.hint, rendered.text
    assert rendered.lines.index("  hint: " + rendered.hint) > rendered.lines.index(rendered.loc.string)


def test_an_error_without_a_hint_renders_no_hint_line():
    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        group = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=-1,
            mutex_ids=[0, 1],
        )
        pl.load(group.next(), x, [0, 0])

    rendered = _render(kernel, ir.SectionKind.Cube)
    assert rendered.hint is None, rendered.text


def test_the_caret_lines_up_under_the_text_it_marks():
    """The caret block is read by eye, so its gutter has to match the quoted line's."""

    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        group = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=-1,
            mutex_ids=[0, 1],
        )
        pl.load(group.next(), x, [0, 0])

    rendered = _render(kernel, ir.SectionKind.Cube)
    quoted, caret = rendered.caret_line
    assert quoted.index("|") == caret.index("|"), (
        f"gutters differ:\n{quoted!r}\n{caret!r}"
    )
    assert quoted[caret.index("^"):caret.index("^") + caret.count("^")] == "-1"


# ---------------------------------------------------------------------------
# More of each shape, so one passing case cannot stand in for the rest
# ---------------------------------------------------------------------------


def test_every_check_renders_a_first_line_whatever_the_shape():
    """One assertion over a spread of kernels, so a new shape cannot skip the head."""

    @pl.jit
    def missing_rank(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        pl.load(pl.make_tile(tile_type, addr=0), x, [0, 0])

    @pl.jit
    def bad_dtype_argument(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        _bad = pl.const(1, "not a dtype")
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        pl.load(pl.make_tile(tile_type, addr=0), x, [0, 0])

    @pl.jit
    def wrong_operand(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)
        pl.load(tile, x, [0, 0])
        pl.abs(x, tile)

    for kernel in (missing_rank, bad_dtype_argument, wrong_operand):
        rendered = _render(kernel)
        assert rendered.head, f"{kernel.__name__}: {rendered.lines[0]!r}"
        assert rendered.text.count("ErrCode:") == 1, kernel.__name__
        assert int(rendered.head["code"], 16) == int(rendered.error.error_code) & 0xFFFFF


def test_a_keyword_check_points_at_a_list_value_too():
    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        group = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=[0, 0x4000],
            mutex_ids=[0, 1, 2],
        )
        pl.load(group.next(), x, [0, 0])

    rendered = _render(kernel, ir.SectionKind.Cube)
    assert isinstance(rendered.error, InvalidShape)
    assert rendered.caret_under() == "[0, 0x4000]"


def test_a_keyword_check_points_at_a_wrongly_typed_value():
    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        group = pl.make_tile_group(type=123, addrs=0, mutex_ids=[0, 1])
        pl.load(group.next(), x, [0, 0])

    rendered = _render(kernel, ir.SectionKind.Cube)
    assert isinstance(rendered.error, InvalidType)
    assert rendered.caret_under() == "123"


def test_a_positional_check_points_at_the_second_argument():
    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        _bad = pl.const(1, "not a dtype")
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        pl.load(pl.make_tile(tile_type, addr=0), x, [0, 0])

    rendered = _render(kernel)
    assert isinstance(rendered.error, InvalidType)
    assert rendered.caret_under() == '"not a dtype"'


def test_a_positional_check_points_at_the_first_argument_of_make_tile():
    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile = pl.make_tile("not a tile type", addr=0)
        pl.load(tile, x, [0, 0])

    rendered = _render(kernel)
    assert isinstance(rendered.error, InvalidType)
    assert rendered.caret_under() == '"not a tile type"'


def test_an_unsupported_keyword_is_marked_at_the_keyword_itself():
    """A keyword nobody declared is wrong in its name, so the name is what is marked.

    The value beside it may be perfectly good; marking it would point at the one
    part of the argument that is not at fault.
    """

    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        pl.load(pl.make_tile(tile_type, addr=0, no_such_keyword=1), x, [0, 0])

    rendered = _render(kernel)
    assert isinstance(rendered.error, InvalidArgument)
    assert rendered.caret_under() == "no_such_keyword"
    assert "no_such_keyword" in rendered.head["reason"]


def test_an_unsupported_group_keyword_is_marked_the_same_way():
    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        group = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0,
            mutex_ids=[0, 1],
            no_such_keyword=7,
        )
        pl.load(group.next(), x, [0, 0])

    rendered = _render(kernel, ir.SectionKind.Cube)
    assert isinstance(rendered.error, InvalidArgument)
    assert rendered.caret_under() == "no_such_keyword"


def test_more_cpp_checks_carry_their_own_source_and_the_qualified_enum():
    """Sampled across C++ files, since each macro call site renders its own head."""

    @pl.jit
    def adds_from_a_tensor(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)
        pl.load(tile, x, [0, 0])
        pl.adds(tile, x, 1)

    @pl.jit
    def adds_into_a_tensor(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)
        pl.load(tile, x, [0, 0])
        pl.adds(x, tile, 1)

    @pl.jit
    def rank_one_tile(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        pl.load(pl.make_tile(tile_type, addr=0), x, [0, 0])

    seen = set()
    for kernel in (adds_from_a_tensor, adds_into_a_tensor, rank_one_tile):
        rendered = _render(kernel)
        origin = rendered.head["origin"]
        # A check may expand in a .cpp or in a shared op header (the
        # tile-argument checks live in op_common.h); both carry their own head.
        assert origin.split(":")[0].endswith((".cpp", ".h")), f"{kernel.__name__}: {origin}"
        enum_field = rendered.head["enum"]
        # One level of qualification, on both sides: error.cpp keeps only the
        # member name and puts the class back from the code, so a call site
        # that spells `npu::tile_fwk::InternalError::X` still renders short.
        assert enum_field.startswith(("ExternalError::", "InternalError::")), enum_field
        assert enum_field.count("::") == 1, enum_field
        assert rendered.head["module"].startswith("PRO_")
        assert rendered.loc, f"{kernel.__name__} lost its DSL location"
        seen.add(origin.split(":")[0])
    assert len(seen) > 1, f"all three came from one file: {seen}"


def test_the_caret_lines_up_in_every_shape():
    """The gutter fix has to hold for one-digit and three-digit line numbers alike."""

    @pl.jit
    def keyword_value(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        group = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=-1,
            mutex_ids=[0, 1],
        )
        pl.load(group.next(), x, [0, 0])

    @pl.jit
    def positional(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        _bad = pl.const("not a number", pl.DT_INT32)
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        pl.load(pl.make_tile(tile_type, addr=0), x, [0, 0])

    @pl.jit
    def single_caret(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)
        pl.load(tile, x, [0, 0])
        pl.cast(tile, x)

    for kernel, section in (
        (keyword_value, ir.SectionKind.Cube),
        (positional, ir.SectionKind.Vector),
        (single_caret, ir.SectionKind.Vector),
    ):
        rendered = _render(kernel, section)
        quoted, caret = rendered.caret_line
        assert quoted.index("|") == caret.index("|"), f"{kernel.__name__}:\n{quoted!r}\n{caret!r}"
        marked = quoted[caret.index("^"):caret.index("^") + caret.count("^")]
        assert marked.strip() == marked and marked, f"{kernel.__name__} marks blank: {marked!r}"


def test_a_helper_check_reports_the_helper_for_several_range_bounds():
    @pl.jit
    def stop_too_large(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)
        for _ in pl.range(0, 2**70):
            pl.load(tile, x, [0, 0])

    @pl.jit
    def start_too_small(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)
        for _ in pl.range(-(2**70), 10):
            pl.load(tile, x, [0, 0])

    for kernel in (stop_too_large, start_too_small):
        rendered = _render(kernel)
        assert isinstance(rendered.error, OutOfRange), kernel.__name__
        assert rendered.head["origin"].startswith("_range.py:"), kernel.__name__
        assert rendered.loc, kernel.__name__


def test_the_module_tag_follows_the_file_that_raised():
    """PRO_PARSER for the parser, PRO_IR for builders and IR, PRO_RUNTIME for the entry."""

    @pl.jit
    def from_the_parser(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        group = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=-1,
            mutex_ids=[0, 1],
        )
        pl.load(group.next(), x, [0, 0])

    @pl.jit
    def from_a_builder(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[-1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        pl.load(pl.make_tile(tile_type, addr=0), x, [0, 0])

    @pl.jit
    def from_the_entry(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)
        out = pl.make_tile(tile_type, addr=0x4000)
        pl.load(tile, x, [0, 0])
        pl.row_sum(out, tile, no_such_keyword=5)

    assert _render(from_the_parser, ir.SectionKind.Cube).head["module"] == "PRO_PARSER"
    assert _render(from_a_builder).head["module"] == "PRO_IR"
    assert _render(from_the_entry).head["module"] == "PRO_RUNTIME"


# ---------------------------------------------------------------------------
# The argument gate: names and counts, checked against the declaration
#
# The gate binds a call's shape to the @_api_decl declaration before any argument
# is parsed, so what it reports is always about an argument -- never about the
# value one was given, which is each builder's own business.
# ---------------------------------------------------------------------------


def _tile_type():
    return pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)


def test_the_gate_reports_a_missing_positional_argument():
    """An argument the call never wrote has no place to mark, so the call is marked."""

    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile = pl.make_tile(_tile_type(), addr=0)
        pl.load(tile, x)

    rendered = _render(kernel)
    assert isinstance(rendered.error, InvalidArgument)
    assert rendered.head["module"] == "PRO_PARSER"
    assert rendered.head["code"] == "0000E"
    assert "missing a required argument: 'offsets'" in rendered.head["reason"]
    assert rendered.caret_under() == "pl.load(tile, x)"
    assert rendered.hint == "pl.load(dst_tile, src_tensor, offsets)"


def test_a_missing_keyword_only_argument_is_spelled_as_a_keyword_in_the_hint():
    """``addr`` cannot be supplied positionally, so the hint writes it as ``addr=...``."""

    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile = pl.make_tile(_tile_type())
        pl.load(tile, x, [0, 0])

    rendered = _render(kernel)
    assert "missing a required argument: 'addr'" in rendered.head["reason"]
    assert rendered.hint == "pl.make_tile(tile_type, addr=...)"


def test_an_argument_past_the_last_declared_one_is_the_one_marked():
    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile = pl.make_tile(_tile_type(), addr=0)
        pl.load(tile, x, [0, 0], 99)

    rendered = _render(kernel)
    assert "too many positional arguments" in rendered.head["reason"]
    assert rendered.caret_under() == "99"


def test_a_keyword_only_argument_given_positionally_is_named_in_the_hint():
    """``pl.load(t, x, offsets, order)`` reads as one argument too many; the hint says why."""

    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile = pl.make_tile(_tile_type(), addr=0)
        pl.load(tile, x, [0, 0], [1, 0])

    rendered = _render(kernel)
    assert "too many positional arguments" in rendered.head["reason"]
    assert rendered.caret_under() == "[1, 0]"
    assert rendered.hint.endswith("order must be passed by keyword")


def test_a_near_miss_keyword_is_offered_the_name_it_resembles():
    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile = pl.make_tile(_tile_type(), addr=0)
        pl.load(tile, x, [0, 0], oder=[1, 0])

    rendered = _render(kernel)
    assert "unexpected keyword argument 'oder'" in rendered.head["reason"]
    assert rendered.caret_under() == "oder"
    assert rendered.hint.startswith("did you mean 'order'?")


def test_a_keyword_resembling_nothing_is_only_told_what_the_op_accepts():
    """A suggestion is offered when there is one, never invented."""

    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile = pl.make_tile(_tile_type(), addr=0)
        pl.load(tile, x, [0, 0], zzzz=1)

    rendered = _render(kernel)
    assert rendered.caret_under() == "zzzz"
    assert "did you mean" not in rendered.hint
    # dst_tile and src_tensor are positional-only, so neither can be named
    assert rendered.hint == "pl.load() takes offsets, order"


def test_a_keyword_repeating_a_positional_argument_names_that_position():
    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile = pl.make_tile(_tile_type(), addr=0)
        pl.load(tile, x, [0, 0], offsets=[1, 1])

    rendered = _render(kernel)
    assert "multiple values for argument 'offsets'" in rendered.head["reason"]
    assert rendered.caret_under() == "offsets"
    assert rendered.hint == "'offsets' is already given as positional argument 3"


def test_an_op_taking_no_argument_says_so_rather_than_listing_none():
    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile = pl.make_tile(_tile_type(), addr=0)
        pl.load(tile, x, [0, 0])
        pl.system.bar_all(span=None)

    rendered = _render(kernel)
    assert "unexpected keyword argument 'span'" in rendered.head["reason"]
    assert rendered.hint == "pl.system.bar_all() takes no arguments"


def test_a_call_written_with_the_declared_names_is_accepted():
    """Every name a declaration lets a caller write must work when written."""

    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile = pl.make_tile(_tile_type(), addr=0)
        pl.load(tile, x, offsets=[0, 0])
        pl.store(x, tile, offsets=[0, 0])

    kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_a_positional_only_parameter_written_as_a_keyword_is_named():
    """A parameter declared before ``/`` has a name the caller may not write."""

    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile = pl.make_tile(_tile_type(), addr=0)
        pl.load(dst_tile=tile, src_tensor=x, offsets=[0, 0])

    rendered = _render(kernel)
    assert isinstance(rendered.error, InvalidArgument)
    assert rendered.head["reason"] == (
        "pl.load() 'dst_tile' parameter is positional only, but was passed as a keyword"
    )
    assert rendered.caret_under() == "dst_tile"
    assert rendered.hint == "write it by position: pl.load(dst_tile, src_tensor, offsets)"


def test_each_gated_namespace_is_checked_against_its_own_declarations():
    """Top-level, simt.* and system.* all reach the gate; only the namespace differs."""

    @pl.jit
    def top_level(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile = pl.make_tile(_tile_type(), addr=0)
        pl.load(tile, x, [0, 0], no_such_keyword=1)

    @pl.jit
    def system_namespace(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile = pl.make_tile(_tile_type(), addr=0)
        pl.load(tile, x, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, evnt_id=0)

    @pl.vector_function(mode="simt", max_threads=1)
    def simt_helper(value: pl.DT_FP32):
        _test_result = pl.simt.cast(value)

    @pl.jit(auto_mutex=False)
    def simt_namespace(value: pl.DT_FP32):
        with pl.section_vector():
            simt_helper[1](value)

    assert _render(top_level).head["reason"].startswith("pl.load()")
    assert _render(system_namespace).head["reason"].startswith("pl.system.sync_src()")
    assert _render(simt_namespace).head["reason"].startswith("pl.simt.cast()")


def test_a_vf_assignment_is_checked_although_it_never_reaches_the_dispatcher():
    """``reg = vf.xxx(...)`` is rewritten statement-side, so it carries its own gate."""

    @pl.vector_function
    def helper(in_a, t_f0):
        preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
        reg_a = vf.load_align(in_a, 0)
        reg_d = vf.abs(reg_a, preg, mdoe=1)
        vf.store_align(t_f0, reg_d, preg)

    @pl.jit
    def kernel(a: pl.Tensor[[8, 64], pl.DT_FP32], o: pl.Tensor[[8, 64], pl.DT_FP32]):
        tf = pl.TileType(shape=[8, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        in_a = pl.make_tile(tf, addr=0)
        t_f0 = pl.make_tile(tf, addr=8192)
        with pl.section_vector():
            pl.load(in_a, a, [0, 0])
            helper(in_a, t_f0)
            pl.store(o, t_f0, [0, 0])

    rendered = _render(kernel)
    # the namespace is written as the user imports it, without a pl. prefix
    assert rendered.head["reason"].startswith("vf.abs()")
    assert rendered.caret_under() == "mdoe"
    assert rendered.hint.startswith("did you mean 'mode'?")


def test_an_op_without_a_declaration_is_left_to_its_builder():
    """The gate's reach is the declaration files; an op outside them is not gated."""

    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile = pl.make_tile(_tile_type(), addr=0)
        out = pl.make_tile(_tile_type(), addr=0x4000)
        pl.load(tile, x, [0, 0])
        pl.row_sum(out, tile, no_such_keyword=5)

    rendered = _render(kernel)
    assert isinstance(rendered.error, RuntimeFailure)
    assert rendered.head["code"] == "00003"


# ---------------------------------------------------------------------------
# The type gate: what a declared annotation accepts, checked on the parsed value
#
# The first gate judges the argument, so it marks the argument; this one judges
# the value, so it marks the value. What it does not describe -- an unannotated
# parameter, Any, an op with no declaration -- it leaves to the builder.
# ---------------------------------------------------------------------------


def test_the_type_gate_names_the_role_the_argument_was_declared_in():
    """A Tensor where a Tile belongs: the code, the module and the mark, in one place."""

    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile = pl.make_tile(_tile_type(), addr=0)
        pl.load(tile, x, [0, 0])
        pl.abs(x, tile)

    rendered = _render(kernel)
    assert isinstance(rendered.error, InvalidType)
    assert rendered.head["module"] == "PRO_PARSER"
    assert rendered.head["code"] == "00001"
    assert rendered.head["reason"] == "pl.abs: out expects Tile, got TensorType"
    assert rendered.caret_under() == "x"


def test_a_tile_where_a_tensor_belongs_is_named_the_same_way():
    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile = pl.make_tile(_tile_type(), addr=0)
        pl.load(tile, tile, [0, 0])

    rendered = _render(kernel)
    assert rendered.head["reason"] == "pl.load: src_tensor expects Tensor, got TileType"
    assert rendered.caret_under() == "tile"


def test_a_shape_given_as_one_number_is_refused_before_it_is_unpacked():
    """A shape names every axis, so a bare int is the wrong kind of thing.

    Left to the constructor this used to surface as ``'int' object is not iterable``
    with no code and a mark on whichever argument came last.
    """

    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile_type = pl.TileType(shape=64, dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        pl.load(pl.make_tile(tile_type, addr=0), x, [0, 0])

    rendered = _render(kernel)
    assert rendered.head["reason"] == "pl.TileType: shape expects Shape, got int"
    assert rendered.caret_under() == "64"


def test_an_enum_from_the_wrong_family_is_named_by_the_family_it_came_from():
    """Both are enums, so the message has to say which one arrived."""

    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile = pl.make_tile(_tile_type(), addr=0)
        pl.load(tile, x, [0, 0])
        pl.system.sync_src(set_pipe=pl.SyncCoreType.AIV_ONLY, wait_pipe=pl.PipeType.V, event_id=0)

    rendered = _render(kernel)
    assert rendered.head["reason"] == "pl.system.sync_src: set_pipe expects PipeType, got SyncCoreType"
    assert rendered.caret_under() == "pl.SyncCoreType.AIV_ONLY"


def test_either_branch_of_a_declared_union_is_accepted():
    """``rhs: Union[Tile, Scalar]`` means both spellings of add, not one of them."""

    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        a = pl.make_tile(tile_type, addr=0)
        b = pl.make_tile(tile_type, addr=0x4000)
        c = pl.make_tile(tile_type, addr=0x8000)
        pl.load(a, x, [0, 0])
        pl.add(b, a, a)
        pl.add(c, a, 1)

    kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_an_offset_is_accepted_however_the_caller_spelled_it():
    """A positional argument parses into an Expr; the same keyword stays a Python list.

    The two routes reach the gate as different kinds of value, and the role has to
    cover both or one spelling of a legal call would be refused.
    """

    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile = pl.make_tile(_tile_type(), addr=0)
        pl.load(tile, x, [0, 0])
        pl.load(tile, x, offsets=[0, 0])

    kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_an_int_parameter_accepts_the_scalar_a_loop_index_parses_into():
    """``offset: int`` is what the user writes; a loop index is an int to them too."""

    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile = pl.make_tile(_tile_type(), addr=0)
        pl.load(tile, x, [0, 0])
        for index in pl.range(4):
            _value = pl.getval(tile, index)

    kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_an_op_without_a_declaration_is_not_type_checked_either():
    """The gate's reach is the declaration files, for values as much as for names."""

    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile = pl.make_tile(_tile_type(), addr=0)
        pl.load(tile, x, [0, 0])
        pl.adds(tile, x, 1)

    rendered = _render(kernel)
    # the C++ deduction is the one that refuses it, and says so in its own words
    assert rendered.head["module"] == "PRO_IR"
    assert rendered.head["origin"].split(":")[0].endswith((".cpp", ".h")), rendered.head["origin"]


# ---------------------------------------------------------------------------
# Scalars by the dtype they carry, and the two ops the gate reaches last
#
# Every IR scalar is one class, so a role that stopped there could not tell an
# integer from a float. It reads the dtype instead, which is also what lets the
# message say which scalar arrived rather than that one arrived.
# ---------------------------------------------------------------------------


def test_a_float_scalar_is_not_the_int_a_parameter_declared():
    """``offset: int`` means an integer, and the message says which scalar came."""

    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16], f: pl.DT_FP32):
        tile = pl.make_tile(_tile_type(), addr=0)
        pl.load(tile, x, [0, 0])
        _value = pl.getval(tile, f)

    rendered = _render(kernel)
    assert isinstance(rendered.error, InvalidType)
    assert rendered.head["reason"] == "pl.getval: offset expects int, got a fp32 scalar"
    assert rendered.caret_under() == "f"


def test_a_condition_the_kernel_computes_is_the_bool_the_declaration_means():
    """``condition: bool`` is satisfied by a comparison, which is a BOOL scalar."""

    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16], n: pl.DT_INT32):
        tile = pl.make_tile(_tile_type(), addr=0)
        pl.load(tile, x, [0, 0])
        pl.pto_assert(n < 100, "n=%d out of range", n)

    kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_an_assert_message_must_be_a_string_rather_than_a_number():
    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16], n: pl.DT_INT32):
        tile = pl.make_tile(_tile_type(), addr=0)
        pl.load(tile, x, [0, 0])
        pl.pto_assert(n < 100, 123)

    rendered = _render(kernel)
    assert rendered.head["reason"] == "pl.pto_assert: format_str expects Optional[str], got a int64 scalar"
    assert rendered.caret_under() == "123"


def test_a_dump_flag_must_be_a_string():
    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile = pl.make_tile(_tile_type(), addr=0)
        pl.load(tile, x, [0, 0])
        pl.dump_data(tile, flag=123)

    rendered = _render(kernel)
    assert rendered.head["reason"] == "pl.dump_data: flag expects Optional[str], got int"
    assert rendered.caret_under() == "123"


def test_dump_offsets_name_the_axes_so_a_word_is_refused():
    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile = pl.make_tile(_tile_type(), addr=0)
        pl.load(tile, x, [0, 0])
        pl.dump_data(tile, offsets="row")

    rendered = _render(kernel)
    assert rendered.head["reason"] == "pl.dump_data: offsets expects Optional[List[int]], got str"
    assert rendered.caret_under() == '"row"'


def test_every_shape_a_dump_is_written_in_is_accepted():
    """The forms the system tests use, so the gate cannot narrow them by accident."""

    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile = pl.make_tile(_tile_type(), addr=0)
        pl.load(tile, x, [0, 0])
        pl.dump_data(tile)
        pl.dump_data(tile, offsets=[4, 0], shapes=[8, 8])
        pl.dump_data(tile, loc=True)
        pl.dump_data(tile, flag="checkpoint_A")
        pl.dump_data(x)

    kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
