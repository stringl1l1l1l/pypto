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


def test_a_builder_check_with_no_published_span_renders_only_the_first_line():
    """pl.TileType is evaluated before the dispatch gate publishes a location."""

    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[0, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        pl.load(pl.make_tile(tile_type, addr=0), x, [0, 0])

    rendered = _render(kernel)
    assert rendered.head
    assert rendered.loc is None
    assert rendered.preview == []


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
    """A C++ check reaches Python whole: its file:line, its code, its DSL location."""

    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)
        pl.load(tile, x, [0, 0])
        pl.cast(tile, x)

    rendered = _render(kernel)
    assert rendered.head, rendered.lines[0]
    assert rendered.head["origin"].endswith(".cpp:" + rendered.head["origin"].split(":")[-1])
    assert rendered.head["module"] == "PRO_IR"
    assert rendered.head["enum"] == "ExternalError::INVALID_TYPE"
    assert isinstance(rendered.error, InvalidType)
    assert rendered.loc, "the dispatch gate should have published the DSL location"


def test_an_interpreter_error_is_wrapped_with_a_code_of_its_own():
    """A TypeError the interpreter raised has no code, so the wrapper supplies one."""

    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)
        pl.load(tile, x, [0, 0], no_such_keyword=5)

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


def test_an_unsupported_keyword_is_marked_at_the_value_it_was_given():
    """The mark lands on the value; the keyword the message names is beside it."""

    @pl.jit
    def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        pl.load(pl.make_tile(tile_type, addr=0, no_such_keyword=1), x, [0, 0])

    rendered = _render(kernel)
    assert isinstance(rendered.error, InvalidArgument)
    assert rendered.caret_under() == "1"
    assert "no_such_keyword" in rendered.head["reason"]


def test_an_unsupported_group_keyword_is_marked_at_its_value_too():
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
    assert rendered.caret_under() == "7"


def test_more_cpp_checks_carry_their_own_source_and_the_qualified_enum():
    """Sampled across C++ files, since each macro call site renders its own head."""

    @pl.jit
    def cast_from_a_tensor(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)
        pl.load(tile, x, [0, 0])
        pl.cast(tile, x)

    @pl.jit
    def abs_into_a_tensor(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)
        pl.load(tile, x, [0, 0])
        pl.abs(x, tile)

    @pl.jit
    def rank_one_tile(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        pl.load(pl.make_tile(tile_type, addr=0), x, [0, 0])

    seen = set()
    for kernel in (cast_from_a_tensor, abs_into_a_tensor, rank_one_tile):
        rendered = _render(kernel)
        origin = rendered.head["origin"]
        assert origin.split(":")[0].endswith(".cpp"), f"{kernel.__name__}: {origin}"
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
        pl.load(tile, x, [0, 0], no_such_keyword=5)

    assert _render(from_the_parser, ir.SectionKind.Cube).head["module"] == "PRO_PARSER"
    assert _render(from_a_builder).head["module"] == "PRO_IR"
    assert _render(from_the_entry).head["module"] == "PRO_RUNTIME"
