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

"""PyPTO-Pro exceptions: one class per error code.

The class *is* the error code. Each class also inherits the builtin exception
that matches its semantics, so ``except ValueError`` keeps working for callers
that do not know about this hierarchy.

Raising::

    raise InvalidArgument(f"block.load requires 3 args, got {n}", span=span)

The error code needs no argument (the class carries it), the module tag is
derived from the caller's file, and the DSL location comes from the ambient
``current_span``, so a call site states only what went wrong.
"""

from __future__ import annotations

__all__ = [
    "PyptoProError",
    "argument_span",
    "remembered_span",
    "span_of",
    "span_source",
    "current_span",
    "error_class_of_spec_message",
    "message_of",
    # One class per error code
    "InvalidType",
    "InvalidVal",
    "RuntimeFailure",
    "NameNotFound",
    "NotSupported",
    "KeyNotFound",
    "InvalidOperation",
    "OutOfRange",
    "BadFd",
    "DynamicShapeUnsupported",
    "InvalidShape",
    "InvalidTile",
    "InvalidFormat",
    "InvalidArgument",
    "CommonExternal",
    "CommonInner",
    "PassInner",
    "CodegenInner",
]

import contextlib
import contextvars
import os
import re
import sys

from pypto.error import PyptoError
from pypto.pypto_impl import ir

from ._error_codes import ErrorCode

# ---------------------------------------------------------------------------
# Module tag: the compile stage a message came from (log spec 4.2)
# ---------------------------------------------------------------------------

# Longest path fragment wins, so nested packages override their parent.
_MODULE_BY_PATH = (
    ("pypto_pro/runtime/pipeline", "PRO_PIPELINE"),
    ("pypto_pro/runtime/opc", "PRO_COMPILE"),
    ("pypto_pro/runtime", "PRO_RUNTIME"),
    ("pypto_pro/language", "PRO_PARSER"),
    ("pypto_pro/ir", "PRO_IR"),
)
_DEFAULT_MODULE = "PRO_COMMON"


def _module_of(filename: str) -> str:
    """Map a source path to its ``PRO_*`` module tag."""
    normalized = filename.replace(os.sep, "/")
    for fragment, module in _MODULE_BY_PATH:
        if fragment in normalized:
            return module
    return _DEFAULT_MODULE


def _origin_of_caller(depth: int) -> tuple[str, str]:
    """Module tag and ``basename:lineno`` of the frame ``depth`` levels up.

    Both come from the same frame -- the one that raised -- so the first line can
    name the framework source the C++ side names with ``__FILE__``/``__LINE__``.
    """
    try:
        frame = sys._getframe(depth)
    except ValueError:
        return _DEFAULT_MODULE, ""
    filename = frame.f_code.co_filename
    return _module_of(filename), f"{os.path.basename(filename)}:{frame.f_lineno}"


# ---------------------------------------------------------------------------
# Ambient span: where in the user's DSL we currently are
# ---------------------------------------------------------------------------
#
# Mirrors pypto's ``ir::Span::Current()`` (core.cpp) — the dispatcher that knows
# the location publishes it once, and every error raised below picks it up while
# being constructed. Validation sits several frames below the call the user
# wrote (``TileType.__post_init__`` is two frames under ``pl.make_tile()``), so
# threading a span through every helper is not an option.
#
# A ContextVar rather than a plain global: ``current_span`` restores the previous
# value on exit, so nested op calls do not clobber each other, and separate
# threads compiling separate kernels stay independent.
_CURRENT_SPAN: contextvars.ContextVar = contextvars.ContextVar("pypto_pro_current_span", default=None)

# Where each argument of that call was written: {keyword name or position: Span}.
# An op builder is handed values, not syntax, so this is the only way for it to
# say which argument an error is about.
_CURRENT_ARGUMENTS: contextvars.ContextVar = contextvars.ContextVar("pypto_pro_current_arguments", default=None)


_SPAN_ATTR = "_pypto_pro_span"


@contextlib.contextmanager
def current_span(span: ir.Span | None, arguments: dict | None = None):
    """Publish *span* as the DSL location for errors raised inside this block.

    An error built inside the block picks the span up on construction. One that
    is *not* ours -- a TypeError the interpreter raises because a keyword does
    not fit a builder's signature, say -- is reported by a handler further up,
    by which time this block has already unwound and the span is gone. So the
    span is also recorded on the exception on its way out, where a handler can
    still read it with :func:`remembered_span`. An inner block records first and
    an outer one leaves that alone, so the innermost location wins.
    """
    token = _CURRENT_SPAN.set(span)
    arg_token = _CURRENT_ARGUMENTS.set(arguments)
    try:
        yield
    except BaseException as error:
        if span is not None and getattr(error, _SPAN_ATTR, None) is None:
            try:
                setattr(error, _SPAN_ATTR, span)
            except AttributeError:
                pass  # an exception type that does not take attributes
        raise
    finally:
        _CURRENT_ARGUMENTS.reset(arg_token)
        _CURRENT_SPAN.reset(token)


def argument_span(name: str | int, fallback: ir.Span | None = None) -> ir.Span | None:
    """Where the caller wrote argument *name*, or *fallback* if that is unknown.

    Op builders receive values; the syntax that produced them stays with the
    parser. The dispatcher publishes it here so a builder that knows which
    argument it is rejecting can point at it, instead of at the whole call.
    *name* is the keyword, or the position for an argument passed positionally.
    Returns *fallback* when the builder was called directly rather than parsed.
    """
    arguments = _CURRENT_ARGUMENTS.get()
    if arguments is None:
        return fallback
    written = arguments.get(name)
    return written if written is not None else fallback


# How to turn an AST node into a Span, for code that walks a kernel's syntax
# without the parser's own tracker at hand (the auto-pipeline analysis).
_SPAN_OF_NODE: contextvars.ContextVar = contextvars.ContextVar("pypto_pro_span_of_node", default=None)


@contextlib.contextmanager
def span_source(resolve):
    """Publish *resolve*, an ``ast.AST -> ir.Span`` callable, for this block.

    The pipeline analysis is pure AST work driven from the runtime, where the
    parser's SpanTracker is not in scope; the caller that does hold the kernel's
    source publishes it here so the analysis can report locations without
    threading a tracker through every function it calls.
    """
    token = _SPAN_OF_NODE.set(resolve)
    try:
        yield
    finally:
        _SPAN_OF_NODE.reset(token)


def span_of(node) -> ir.Span | None:
    """Span of the AST *node*, when a source is published; None otherwise.

    None is the right answer outside a published block: the error then falls
    back to the ambient span, or reports no location at all.
    """
    resolve = _SPAN_OF_NODE.get()
    if resolve is None or node is None:
        return None
    try:
        return resolve(node)
    except Exception:
        return None


def remembered_span(error: BaseException) -> ir.Span | None:
    """Where *error* passed through a :func:`current_span` block, if it did."""
    return getattr(error, _SPAN_ATTR, None)


def _normalize_span(span: ir.Span | None) -> dict[str, str | int | None] | None:
    """Convert an IR span to a plain dict, so no C++ object is kept alive."""
    if span is None:
        return None
    if isinstance(span, dict):
        return span

    filename = getattr(span, "filename", None)
    begin_line = getattr(span, "begin_line", 0)
    begin_column = getattr(span, "begin_column", 0)
    end_line = getattr(span, "end_line", 0)
    end_column = getattr(span, "end_column", 0)
    return {
        "filename": filename,
        "line": begin_line,
        "column": begin_column,
        "file": filename,
        "begin_line": begin_line,
        "begin_column": begin_column,
        "end_line": end_line,
        "end_column": end_column,
    }


# A message that already opens with a spec first line must not get a second one.
# The C++ side is the only producer of one: its head carries the framework
# ``file:line`` that ``_first_line`` below deliberately omits, so the shape of the
# leading bracket group tells the two apart. The legacy pattern covers C++ sites
# not yet migrated to PRO_CHECK.
# Both sides render the same first line, so this matches a Python head as well as
# a C++ one; only the legacy CHECK/ASSERT form below is C++-only.
_SPEC_HEAD = re.compile(r"^\[[^\[\]]+:\d+\]\[[A-Za-z_]\w*\]:ErrCode: F[0-9A-F]{5}! Enum: ")
_CPP_LEGACY_HEAD = re.compile(r"^(?:CHECK|ASSERT) FAILED: ErrCode: F[0-9A-F]{5}! Enum: ")
# The C++ side renders a bare "  --> file:line:col"; ours renders the same location
# with a source preview, so its line is dropped rather than shown twice.
_CPP_LOC_LINE = re.compile(r"\n  --> [^\n]*")
_CPP_TRACEBACK = re.compile(r"\n\nC\+\+ Traceback \(most recent call last\):\n.*\Z", re.S)


def carries_spec_head(message: str) -> bool:
    """Whether *message* already opens with the mandatory first line (spec 4.1)."""
    first = message.split("\n", 1)[0]
    return bool(_SPEC_HEAD.match(first) or _CPP_LEGACY_HEAD.match(first))


def _strip_cpp_location(message: str) -> str:
    """Drop the C++ location line and stack trace, which we re-render ourselves."""
    return _CPP_LOC_LINE.sub("", _CPP_TRACEBACK.sub("", message))


# The C++ location line, parsed back into a span so we can re-render it with a
# source preview instead of passing its bare one-liner through.
_CPP_LOC_PARSE = re.compile(r"\n  --> (?P<file>[^\n:]+):(?P<line>\d+):(?P<col>-?\d+)")


def spec_head_of(message: str) -> str | None:
    """The spec first line of *message* plus its reason, if it carries one.

    The migrated form ends its head with ``. `` and the reason follows on the same
    line. The legacy ``CHECK FAILED:`` form puts the reason on the *next* line, so
    that line belongs to the head as far as quoting is concerned -- returning only
    the first line there would drop the reason entirely.
    """
    lines = message.split("\n")
    if _SPEC_HEAD.match(lines[0]):
        return lines[0]
    if _CPP_LEGACY_HEAD.match(lines[0]):
        reason = lines[1].strip() if len(lines) > 1 else ""
        return f"{lines[0]} {reason}".rstrip() if reason else lines[0]
    return None


def span_of_spec_message(message: str) -> dict[str, str | int | None] | None:
    """The location the C++ side reported, as a span dict.

    The C++ side knows exactly which call failed -- it was handed the span at the
    dispatch gate. A Python handler that wraps the failure usually knows less
    (its own ``_current_node`` may sit lines away), so the inner location is the
    one worth keeping.
    """
    match = _CPP_LOC_PARSE.search(message)
    if not match:
        return None
    line = int(match.group("line"))
    column = int(match.group("col"))
    if line <= 0:
        return None
    return {
        "filename": match.group("file"),
        "line": line,
        "column": column if column > 0 else None,
        "file": match.group("file"),
        "begin_line": line,
        "begin_column": column if column > 0 else None,
        "end_line": line,
        "end_column": None,
    }


def _strip_spec_head(message: str) -> str:
    """Drop a leading spec first line, leaving the bare reason.

    Used when quoting a C++ error inside another message: the quoting raise
    renders its own first line, and the quoted one would land mid-sentence.
    """
    first, sep, rest = message.partition("\n")
    for pattern in (_SPEC_HEAD, _CPP_LEGACY_HEAD):
        match = pattern.match(first)
        if match:
            head_tail = first[match.end():]
            # The migrated head ends with ". " before the message; the legacy one
            # puts the message on the next line.
            body = head_tail.split(". ", 1)[-1] if ". " in head_tail else ""
            return (body + sep + rest).lstrip("\n") if body else rest.lstrip("\n")
    return message


def _first_line(code: ErrorCode, module: str, origin: str, message: str) -> str:
    """Render the mandatory first line (log spec 4.1).

    ``origin`` is the framework source that raised, in the same ``file:line``
    position the C++ side fills from ``__FILE__``/``__LINE__``, so both sides
    render one format. It is empty only when the stack is too shallow to read.

    A message that already carries a first line -- an inner error quoted by a
    wrapping raise -- keeps it: that head names the code the inner check
    assigned, which is more specific than the one the wrapping class can offer.
    """
    if carries_spec_head(message):
        return message
    head = f"[{origin}][{module}]" if origin else f"[{module}]"
    masked = int(code) & 0xFFFFF
    # Every InternalError member sits above the low 16 bits (0x1FFFF and up);
    # every ExternalError member stays inside them. The C++ side derives the same
    # prefix from the same bits (error.cpp EnumClassName), so both sides render
    # one format.
    enum_class = "ExternalError" if masked & 0xF0000 == 0 else "InternalError"
    return f"{head}:ErrCode: F{masked:05X}! Enum: {enum_class}::{code.name}. {message}"


# Lines of context shown around a span. pypto's diagnostics show 2 and 4; the
# trailing 4 usually runs past the end of the call the error is about, so the
# call stays visible with 2 on each side.
_PRIOR_CONTEXT_LINES = 2
_FOLLOWING_CONTEXT_LINES = 2


class PyptoProError(PyptoError):
    """Base class for every pypto_pro error.

    Subclasses set ``error_code`` and mix in the builtin exception that matches
    their semantics; they add nothing else.
    """

    error_code = ErrorCode.UNKNOWN

    def __init__(
        self,
        message: str,
        span: ir.Span | None = None,
        hint: str | None = None,
        note: str | None = None,
        source_lines: list[str] | None = None,
        *,
        parser_retry: bool = False,
        previous_span: ir.Span | None = None,
        _depth: int = 2,
    ):
        """Initialize the error.

        Args:
            message: What went wrong, in one sentence.
            span: Location in the user's DSL source.
            hint: Optional suggestion for how to fix it.
            note: Optional extra remark.
            source_lines: Optional source of the file `span` points into,
                used to render the caret preview.
            parser_retry: Mark a rejection the expression parser may retry as a
                plain Python expression. Off by default, so a rejection stands
                unless the site says it is safe to retry. See
                ``ExpressionParserMixin.parse_expression``.
            previous_span: Optional earlier location the message refers to,
                such as a variable's previous definition.
            _depth: Stack depth the module tag and the first line's ``file:line``
                are read from; raise it when the error is constructed by a helper
                rather than at the site.
        """
        self.module, self.origin = _origin_of_caller(_depth)
        inherited_span = None
        if carries_spec_head(message):
            # The inner error already carries the mandatory first line, with the code
            # its own check assigned and the location it was handed. Both are more
            # specific than anything this wrapping raise can offer, so keep them:
            # take the inner location as our span, and let _first_line pass the inner
            # head through untouched rather than stacking a second one on top.
            inherited_span = span_of_spec_message(message)
            message = _strip_cpp_location(message)
        super().__init__(
            int(self.error_code), _first_line(self.error_code, self.module, self.origin, message)
        )
        self.message = message

        if span is None and inherited_span is not None:
            span = inherited_span
        self.span = _normalize_span(span if span is not None else _CURRENT_SPAN.get())
        self.hint = hint
        self.note = note
        self.source_lines = source_lines
        self.parser_retry = parser_retry
        self.previous_span = _normalize_span(previous_span)

    def __str__(self) -> str:
        parts = [super().__str__()]
        self._append_location(parts)
        if self.hint:
            parts.append(f"  hint: {self.hint}")
        if self.note:
            parts.append(f"  note: {self.note}")
        return "\n".join(parts)

    def _append_location(self, parts: list) -> None:
        if not self.span:
            return
        filename = self.span.get("filename") or "<unknown>"
        line = self.span.get("line") or self.span.get("begin_line")
        if not line:
            return
        column = self.span.get("column") or self.span.get("begin_column")
        parts.append(f"  --> {filename}:{line}:{column}")
        self._append_source_preview(parts, line, column)

    def _append_source_preview(self, parts: list, line: int, column: int) -> None:
        """Show the span in its surroundings, with carets under what it covers.

        Same shape as pypto's own diagnostics (frontend/parser/diagnostics.py):
        a few lines of context so the reader sees the call the argument belongs
        to, and a caret run on every line the span reaches -- an argument written
        on its own line is unreadable without the call around it.
        """
        if not column:
            return
        end_line = self.span.get("end_line") or line
        if end_line < line:
            end_line = line
        first = max(1, line - _PRIOR_CONTEXT_LINES)
        last = end_line + _FOLLOWING_CONTEXT_LINES
        width = max(3, len(str(last)))

        # The gutter has to be as wide as the quoted line's, or the caret run --
        # which _caret_for indents from the start of the source text -- lands one
        # column left of what it marks.
        parts.append(" " * width + " |")
        for number in range(first, last + 1):
            text = self._get_source_line(number)
            if text is None:
                continue
            parts.append(f"{number:{width}} | {text}")
            if line <= number <= end_line:
                caret = self._caret_for(text, number, line, end_line, column)
                if caret:
                    parts.append(" " * width + " | " + caret)

    def _caret_for(self, text: str, number: int, line: int, end_line: int, column: int) -> str:
        """The caret run under one line of the span, indented to meet it."""
        start = column - 1 if number == line else len(text) - len(text.lstrip())
        if number == end_line:
            stop = (self.span.get("end_column") or 0) if end_line != line else self.span.get("end_column") or 0
            stop = max(stop, start + 1)
        else:
            stop = len(text.rstrip())
        if stop <= start:
            return ""
        return " " * start + "^" * (stop - start)

    def _get_source_line(self, line: int) -> str | None:
        if not self.source_lines:
            return None
        idx = line - 1
        if 0 <= idx < len(self.source_lines):
            return self.source_lines[idx].rstrip("\n")
        return None


def message_of(error: BaseException) -> str:
    """The text of *error* to quote inside another error's message.

    A handler that wraps a failure into its own error quotes the reason it
    caught. ``str()`` is the wrong thing to quote for a pypto_pro error: it
    renders the first line and the source location too, and the wrapping raise
    is about to render both again. The bare message is what ``str()`` on a
    builtin exception gives, which is what these sites quoted before one class
    per error code made the caught exception one of ours.
    """
    if isinstance(error, PyptoProError):
        return error.message
    # A C++ error renders the same way ours does, so quoting str() drags its first
    # line and location into the middle of the quoting message. Quote the reason only.
    return _strip_spec_head(_strip_cpp_location(str(error)))


def _make(name: str, code: ErrorCode, base: type[Exception], doc: str) -> type[PyptoProError]:
    """Build one error class: the name states the code, the base states the type."""
    return type(name, (PyptoProError, base), {"error_code": code, "__doc__": doc})


# The table below is both the class list and the code -> builtin-base mapping.
InvalidType = _make(
    "InvalidType", ErrorCode.INVALID_TYPE, TypeError, "A value or operand has the wrong type."
)
InvalidArgument = _make(
    "InvalidArgument",
    ErrorCode.INVALID_ARGUMENT,
    ValueError,
    "Argument count, positional/keyword usage, or a value outside the allowed set.",
)
NotSupported = _make(
    "NotSupported",
    ErrorCode.NOT_IMPLEMENTED_ERROR,
    NotImplementedError,
    "A syntax, feature, or platform combination that is not supported.",
)
InvalidOperation = _make(
    "InvalidOperation",
    ErrorCode.INVALID_OPERATION,
    ValueError,
    "The operation is not valid in this context, order, or combination.",
)
InvalidShape = _make(
    "InvalidShape", ErrorCode.INVALID_SHAPE, ValueError, "Shape, rank, or a dimension does not match."
)
InvalidVal = _make("InvalidVal", ErrorCode.INVALID_VAL, ValueError, "A value is otherwise unacceptable.")
OutOfRange = _make(
    # ValueError, not IndexError: this reports a value outside the range its API accepts (a dtype's
    # representable band, a thread count, a mutex id), which is what the callers -- and their
    # ``except ValueError`` -- have always treated it as. IndexError is for a sequence subscript.
    "OutOfRange", ErrorCode.OUT_OF_RANGE, ValueError, "An index, value, or capacity is out of range."
)
InvalidFormat = _make(
    "InvalidFormat", ErrorCode.INVALID_FORMAT, ValueError, "The ND / NZ data format constraint is violated."
)
InvalidTile = _make(
    "InvalidTile", ErrorCode.INVALID_TILE, ValueError, "A tile's own parameters (address, size, alignment) are invalid."
)
DynamicShapeUnsupported = _make(
    "DynamicShapeUnsupported",
    ErrorCode.DYNAMIC_SHAPE_COMPUTE_UNSUPPORTED,
    NotImplementedError,
    "The shape is not available at compile time.",
)
RuntimeFailure = _make(
    "RuntimeFailure", ErrorCode.RUNTIME_ERROR, RuntimeError, "A call into an external dependency failed."
)
KeyNotFound = _make(
    "KeyNotFound", ErrorCode.KEY_ERROR, KeyError, "An attribute, field, or registry entry does not exist."
)
NameNotFound = _make("NameNotFound", ErrorCode.NAME_ERROR, NameError, "A name could not be resolved.")
BadFd = _make("BadFd", ErrorCode.BAD_FD, OSError, "A file or shared library is missing.")
CommonExternal = _make(
    "CommonExternal", ErrorCode.COMMON_EXTERNAL_ERROR, ValueError, "Fallback for a user error with no better code."
)
CommonInner = _make(
    "CommonInner", ErrorCode.COMMON_INNER_ERROR, RuntimeError, "Internal error outside the pass and codegen stages."
)
PassInner = _make(
    "PassInner", ErrorCode.PASS_INNER_ERROR, RuntimeError, "Internal error in an IR pass or transform."
)
CodegenInner = _make(
    "CodegenInner", ErrorCode.CODEGEN_INNER_ERROR, RuntimeError, "Internal error during code generation."
)


# ---------------------------------------------------------------------------
# Code -> class, for wrapping an error that already decided its own code
# ---------------------------------------------------------------------------

# Both sides key off the same enum (tilefwk/error_code.h), so a code the C++ side
# assigned names exactly one of the classes above.
# Keyed on the code's low 20 bits, the same part _first_line renders.
_CLASS_BY_CODE = {
    int(cls.error_code) & 0xFFFFF: cls
    for cls in (
        InvalidType, InvalidArgument, NotSupported, InvalidOperation, InvalidShape, InvalidVal,
        OutOfRange, InvalidFormat, InvalidTile, DynamicShapeUnsupported, RuntimeFailure,
        KeyNotFound, NameNotFound, BadFd, CommonExternal, CommonInner, PassInner, CodegenInner,
    )
}

_SPEC_CODE = re.compile(r"ErrCode: (?P<code>F[0-9A-F]{5})!")


def error_class_of_spec_message(message: str) -> type[PyptoProError] | None:
    """The class matching the code in *message*'s spec first line, if it has one.

    A handler that wraps a failure it did not diagnose should not downgrade the code
    the inner check assigned -- and the class must agree with that code, since the
    class name is what Python prints in the traceback. Returns None when the message
    carries no spec first line (an interpreter-raised error, say), leaving the caller
    to pick its own fallback.
    """
    head = spec_head_of(message)
    if head is None:
        return None
    match = _SPEC_CODE.search(head)
    if match is None:
        return None
    return _CLASS_BY_CODE.get(int(match.group("code")[1:], 16))
