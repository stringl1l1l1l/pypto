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

"""
Internal kernel representation used by the PyPTO Pro JIT.

Public kernels are defined with @pl.jit. KernelDef stores the captured source
and parser state for one specialization after the JIT frontend has resolved its
compile-time configuration.
"""

from __future__ import annotations

__all__ = ["KernelDef"]


import ast
import inspect
import textwrap
from typing import Any, Callable, TypeVar

from pypto.pypto_impl import ir
from pypto_pro.language.parser._ast_parser import ASTParser

from .._errors import (
    NameNotFound,
    PyptoProError,
    RuntimeFailure,
    error_class_of_spec_message,
    message_of,
    remembered_span,
    span_of_spec_message,
    spec_head_of,
)


def _calculate_col_offset(source_lines: list[str]) -> int:
    """Calculate the column offset (indentation) of the first non-empty line.

    This is needed because ast.parse() requires code starting at column 0,
    but we need to report errors at the correct column in the original file.

    Args:
        source_lines: List of source code lines

    Returns:
        Column offset (number of leading spaces/tabs in first non-empty line)
    """
    for line in source_lines:
        if line.strip():  # Skip empty lines
            return len(line) - len(line.lstrip())
    return 0


def _parse_ast_tree(source_code: str, entity_type: str) -> ast.AST:
    """Parse source code into an AST tree with proper error handling.

    Args:
        source_code: Python source code to parse
        entity_type: Type of entity being parsed ("function" or "class") for error messages

    Returns:
        Parsed AST tree

    Raises:
        NotSupported: If the source code cannot be parsed
    """
    try:
        return ast.parse(source_code)
    except SyntaxError as e:
        raise RuntimeFailure(
            f"Failed to parse {entity_type} source: {e.msg}",
            hint=f"Check for Python syntax errors in your {entity_type}",
        ) from e


TypeASTNode = TypeVar("TypeASTNode", ast.FunctionDef, ast.ClassDef)


def _find_ast_node(tree: ast.AST, node_type: type[TypeASTNode], name: str, entity_type: str) -> TypeASTNode:
    """Find a specific AST node by type and name.

    Args:
        tree: AST tree to search
        node_type: Type of AST node to find (ast.FunctionDef or ast.ClassDef)
        name: Name of the node to find
        entity_type: Type of entity for error messages ("function" or "class")

    Returns:
        Found AST node

    Raises:
        NotSupported: If the node cannot be found
    """
    for node in ast.walk(tree):
        if isinstance(node, node_type) and node.name == name:
            return node

    raise NameNotFound(
        f"Could not find {entity_type} definition for {name}",
        hint=f"Ensure the {entity_type} is properly defined",
    )


def _attach_source_lines_to_error(error: PyptoProError, source_file: str, source_lines_raw: list[str]) -> None:
    """Attach source lines to a pypto_pro error if not already present.

    Args:
        error: Error to attach source lines to
        source_file: Path to the source file
        source_lines_raw: Raw source lines as fallback
    """
    if error.source_lines is None:
        # Use the span's filename if it differs (e.g., error in an inline function)
        target_file = source_file
        if error.span and isinstance(error.span, dict):
            span_file = error.span.get("filename")
            if span_file and span_file != source_file:
                target_file = span_file
        try:
            with open(target_file, encoding="utf-8") as f:
                error.source_lines = f.read().split("\n")
        except Exception:
            # Fallback to the raw source lines if we can't read the file
            error.source_lines = source_lines_raw


def extract_func_source_info(f: Callable):
    """Extract source file, lines, offsets, and parsed func_def from a function.

    Returns:
        tuple of (source_file, source_lines, source_lines_raw, line_offset,
                  col_offset, func_def)
    """
    source_file = inspect.getfile(f)
    source_lines_raw, starting_line = inspect.getsourcelines(f)
    source_code = "".join(source_lines_raw)
    col_offset = _calculate_col_offset(source_lines_raw)
    source_code = textwrap.dedent(source_code)
    source_lines = source_code.split("\n")
    line_offset = starting_line - 1

    try:
        tree = _parse_ast_tree(source_code, "function")
        func_def = _find_ast_node(tree, ast.FunctionDef, f.__name__, "function")
    except PyptoProError as e:
        _attach_source_lines_to_error(e, source_file, source_lines_raw)
        raise

    return (source_file, source_lines, source_lines_raw, line_offset, col_offset, func_def)


class KernelDef:
    """Lazy kernel definition — captures source/AST/closure at decoration time,
    defers AST parsing to compile time.

    The JIT codegen path parses the kernel once for each required target.

    Args:
        func: Original Python function.
        source_file: Path to the source file.
        source_lines: Dedented source lines for the parser.
        source_lines_raw: Raw (non-dedented) source lines for error reporting.
        line_offset: Line number offset in the original file.
        col_offset: Column indentation offset.
        func_def: AST FunctionDef node.
        closure_vars: Captured caller scope for name resolution.
        name: Optional program name.
        func_type: IR function type (Opaque, InCore, Helper).
        strict_ssa: Whether to enforce SSA.
        meta_data: Optional metadata.
        auto_mutex: Whether to enable automatic mutex lock/unlock insertion.
    """

    def __init__(
        self,
        func: Callable,
        source_file: str,
        source_lines: list[str],
        source_lines_raw: list[str],
        line_offset: int,
        col_offset: int,
        func_def: ast.FunctionDef,
        closure_vars: dict[str, Any],
        name: str | None,
        func_type: ir.FunctionType,
        strict_ssa: bool,
        meta_data: Any,
        auto_mutex: bool = True,
        pipeline=None,
        tilingkey_consts: dict[str, int] | None = None,
        datatype_consts: dict[str, Any] | None = None,
    ) -> None:
        self._func = func
        self._source_file = source_file
        self._source_lines = source_lines
        self._source_lines_raw = source_lines_raw
        self._line_offset = line_offset
        self._col_offset = col_offset
        self._func_def = func_def
        self._closure_vars = closure_vars
        self._name = name
        self._func_type = func_type
        self._strict_ssa = strict_ssa
        self._auto_mutex = auto_mutex
        self._pipeline = pipeline
        self._meta_data = meta_data
        self._tilingkey_consts = tilingkey_consts
        self._datatype_consts = datatype_consts
        # Populated by parse_target_program: param name -> "in"/"out" from
        # pl.Input/pl.Output annotation markers.
        self._last_param_directions: dict[str, str] = {}
        self._max_vec_tile_end = 0
        self._requires_simt = False

    @property
    def func_def(self) -> ast.FunctionDef:
        return self._func_def

    @property
    def closure_vars(self) -> dict[str, Any]:
        return self._closure_vars

    @property
    def func_name(self) -> str:
        return self._func.__name__

    @property
    def last_param_directions(self) -> dict[str, str]:
        """Direction markers from the most recent parse_target_program call."""
        return dict(self._last_param_directions)

    @property
    def max_vec_tile_end(self) -> int:
        """Vec Tile high-water mark collected by parse_target_program."""
        return self._max_vec_tile_end

    @property
    def requires_simt(self) -> bool:
        """Whether parse_target_program emitted a SIMT launch."""
        return self._requires_simt

    def parse_target_program(
        self,
        target: ir.SectionKind,
        bound_signature=None,
    ) -> tuple[ir.Program, bool]:
        """Parse a fresh target Program and report whether its target section was matched."""
        program_name = self._name if self._name is not None else self._func.__name__

        try:
            # The Program owns one IRDebugInfo; share it with the parser so all
            # semantic tuple metadata lands in the table the Program carries.
            debug_info = ir.IRDebugInfo()
            parser = ASTParser(
                self._source_file,
                self._source_lines,
                target,
                self._line_offset,
                self._col_offset,
                strict_ssa=self._strict_ssa,
                closure_vars=self._closure_vars,
                auto_mutex=self._auto_mutex,
                debug_info=debug_info,
                tilingkey_consts=self._tilingkey_consts,
                datatype_consts=self._datatype_consts,
                bound_signature=bound_signature,
                # Kernels use a void ABI: they may early-return, but cannot return values.
                void_return_only=True,
                void_return_context="@pl.jit",
                allow_early_return=True,
            )

            try:
                ir_func = parser.parse_function(self._func_def, func_type=self._func_type)
            except PyptoProError:
                raise
            except Exception as e:
                # An exception that passed through the op dispatcher carries the
                # span of the call it was parsing -- including one the interpreter
                # raised, such as a keyword the builder's signature does not
                # accept. ``_current_node`` only tracks the last expression
                # parsed, which lags behind inside a larger statement and would
                # point somewhere else entirely.
                # A C++ error states in its first line exactly which call failed --
                # it was handed that span at the dispatch gate. Prefer it: the two
                # fallbacks below only reach statement granularity.
                span = span_of_spec_message(str(e)) or remembered_span(e)
                if span is None:
                    node = getattr(parser, '_current_node', None)
                    if node is not None:
                        span = parser.span_tracker.get_span(node)
                if isinstance(e, (AttributeError, TypeError)):
                    hint = (
                        "an internal type check failed while parsing; an argument may "
                        "have an unsupported type — check that kernel arguments match "
                        "the expected Tile/Tensor/scalar types"
                    )
                else:
                    hint = "Check your function definition for errors"
                # An error that already carries the mandatory first line states the code
                # its own check assigned; quoting only its reason would replace that code
                # with this wrapper's generic one. Keep the inner line whole -- it already
                # ends with the reason -- and append just the context this frame adds.
                context = f"while parsing kernel function '{self._func.__name__}'"
                inner_head = spec_head_of(str(e))
                if inner_head:
                    message = f"{inner_head} ({context})"
                else:
                    message = (f"Failed to parse kernel function '{self._func.__name__}': "
                               f"{type(e).__name__}: {message_of(e)}")
                # The class name is what Python prints in the traceback, so it has to agree
                # with the code in the message. An inner error that carries a spec first
                # line already decided its code -- both sides key off the same enum, so
                # that code names exactly one class. Anything without a first line (an
                # interpreter-raised AttributeError, say) has no code of its own and keeps
                # the RuntimeFailure fallback this handler was written around.
                wrapper = error_class_of_spec_message(str(e)) or RuntimeFailure
                raise wrapper(message, span=span, hint=hint) from e

            external_funcs = list(parser.external_funcs.values())
            starting_line = self._line_offset + 1
            program_span = ir.Span(self._source_file, starting_line, self._col_offset)
            program = ir.Program(
                external_funcs + [ir_func], program_name, program_span, parser.debug_info
            )
            # Direction markers (pl.Input/pl.Output) parsed off the annotations;
            # consumed by the JIT caller for profiling tensor type.
            self._last_param_directions = dict(parser.param_directions)
            self._max_vec_tile_end = parser.max_vec_tile_end
            self._requires_simt = parser.requires_simt
            return program, parser.matched_target

        except PyptoProError as e:
            _attach_source_lines_to_error(e, self._source_file, self._source_lines_raw)
            raise
