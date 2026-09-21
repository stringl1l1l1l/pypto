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
"""Source-text and function-introspection utils for kernel authoring.

Capture and reshape a function's source for the embedded compile snippet (strip decorator lines,
drop a leading ``@<recv>.kernel`` op-marker), plus the canonical-naming, span-editing and symtable
scope-analysis primitives the multi-file kernel tracer uses.
"""
import ast
import re
import symtable


def _unwrap_decorated_func_source(source: str) -> str:
    """Return the def ... body of a function, stripping decorator lines."""
    return source[source.find("def "):]


# ---- identity-based canonical naming ----

_NON_IDENT_RE = re.compile(r"[^A-Za-z0-9_]")


def _sanitize(s: str) -> str:
    """Turn a module/qualname string into a valid Python identifier fragment.

    ``.`` becomes ``__``; every other non-``[A-Za-z0-9_]`` char becomes a single ``_``.
    """
    return _NON_IDENT_RE.sub("_", s.replace(".", "__"))


# ---- byte-to-char span editor ----

def _line_start_offsets(src: str) -> list[int]:
    """Absolute char offset at which each 1-based source line begins.

    Index ``i`` is the start of line ``i+1``, with a trailing entry so a node ending at EOF is
    representable.
    """
    starts = [0]
    offset = 0
    for line in src.splitlines(keepends=True):
        offset += len(line)
        starts.append(offset)
    return starts


def _byte_col_to_char_col(line: str, byte_col: int) -> int:
    """Convert an AST ``col_offset`` (a UTF-8 byte offset into *line*) to a char column.

    ``ast`` reports columns as byte offsets; a non-ASCII identifier (e.g. a Greek letter) occupies
    multiple bytes, so a byte offset must be decoded back to a char index to slice the raw text.
    """
    if byte_col <= 0:
        return 0
    return len(line.encode("utf-8")[:byte_col].decode("utf-8", "surrogatepass"))


def _node_char_span(node: ast.AST, line_starts: list[int], lines: list[str]) -> tuple[int, int]:
    """Absolute ``(start, end)`` char offsets of *node* within the source, or ``(-1, -1)``.

    Converts the node's byte ``col_offset``/``end_col_offset`` to char columns against each line's
    text. *lines* is ``src.splitlines(keepends=True)``.
    """
    if node.lineno is None or getattr(node, "end_lineno", None) is None:
        return (-1, -1)
    start_line = lines[node.lineno - 1] if node.lineno - 1 < len(lines) else ""
    end_line = lines[node.end_lineno - 1] if node.end_lineno - 1 < len(lines) else ""
    start = line_starts[node.lineno - 1] + _byte_col_to_char_col(start_line, node.col_offset)
    end = line_starts[node.end_lineno - 1] + _byte_col_to_char_col(end_line, node.end_col_offset)
    return (start, end)


def _stmt_line_span(node: ast.AST, src: str, line_starts: list[int], lines: list[str]) -> tuple[int, int]:
    """Whole-line char span of a statement node (incl. the trailing newline), for clean deletion.

    Extends the node's span to its first line's start and past its last line's end, so a
    multi-line statement is removed whole without leaving a stray blank line.
    """
    node_start, node_end = _node_char_span(node, line_starts, lines)
    if node_start < 0:
        return (-1, -1)
    start = line_starts[node.lineno - 1]  # beginning of the statement's first physical line
    end_line_idx = getattr(node, "end_lineno", node.lineno) - 1
    end = line_starts[end_line_idx + 1] if end_line_idx + 1 < len(line_starts) else len(src)
    return (start, end)


def _annotation_delete_spans(
    func_def: ast.AST, src: str, line_starts: list[int], lines: list[str]
) -> list[tuple[ast.AST, tuple[int, int]]]:
    """``(annotation node, delete span)`` per annotated param of *func_def*, then its return annotation.

    A param span is ``[name_end .. annotation_end]``, covering the ``: Foo`` of ``def f(x: Foo = 3)``:
    anchored just past the arg's own identifier (its ``lineno``/``col_offset`` plus the byte-to-char
    safe name length), never a backward ``:`` scan, so the ``= 3`` default is preserved and a preceding
    param is never touched. The return span is ``[arrow .. returns_end]``, its ``->`` located by a
    bounded ``rfind`` so the whitespace between arrow and annotation goes too. A span that cannot be
    located is ``(-1, -1)``.
    """
    args = func_def.args
    slots: list[tuple[ast.AST, tuple[int, int]]] = []
    for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
        if arg.annotation is None:
            continue
        line = lines[arg.lineno - 1] if arg.lineno - 1 < len(lines) else ""
        start = line_starts[arg.lineno - 1] + _byte_col_to_char_col(line, arg.col_offset) + len(arg.arg)
        slots.append((arg.annotation, (start, _node_char_span(arg.annotation, line_starts, lines)[1])))
    returns = getattr(func_def, "returns", None)
    if returns is not None:
        ret_start, ret_end = _node_char_span(returns, line_starts, lines)
        # The last ``->`` before the annotation is the return arrow (the arg list is already closed).
        arrow = src.rfind("->", line_starts[func_def.lineno - 1], ret_start) if ret_start >= 0 else -1
        slots.append((returns, (arrow, ret_end) if arrow >= 0 else (-1, -1)))
    return slots


def _apply_span_replacements(src: str, replacements: list[tuple[int, int, str]]) -> str:
    """Apply ``(start, end, text)`` char-offset replacements to *src*, non-overlapping.

    Candidates are taken in (start, longest-first) order and a span overlapping an already-chosen
    one is dropped; edits are applied in descending start order so earlier edits never shift later
    spans. Never re-parses, so the comments and formatting the jit parser re-reads survive.
    """
    chosen: list[tuple[int, int, str]] = []
    occupied: list[tuple[int, int]] = []
    for start, end, text in sorted(replacements, key=lambda r: (r[0], -(r[1]))):
        if start < 0 or end < start:
            continue
        if any(not (end <= os or start >= oe) for os, oe in occupied):
            continue  # overlaps an already-chosen span
        chosen.append((start, end, text))
        occupied.append((start, end))
    out = src
    for start, end, text in sorted(chosen, key=lambda r: r[0], reverse=True):
        out = out[:start] + text + out[end:]
    return out


# ---- symtable-based global-load scope analysis ----

def _scope_classify(src: str) -> tuple[set[str], set[str]]:
    """``(global-load names, ambiguous names)`` for *src* via ``symtable``.

    ``symtable`` is used because it is version-robust: the bytecode ``co_consts``/AST lockstep is
    broken on 3.12+, where PEP 709 inlines comprehensions. Every lexical scope is walked (module +
    nested function/class/lambda/comprehension). A name is a global load iff in some scope its
    symbol ``is_global()`` and is referenced; locals, parameters, comprehension/lambda vars,
    assignment targets and ``global``/``nonlocal`` decls are excluded. A name that is a global load
    in one scope but locally bound in another (shadow-and-global-use) is ambiguous and comes back
    in the second set; the span editor cannot decide those per-occurrence, so callers raise.
    """
    try:
        table = symtable.symtable(src, "<snip>", "exec")
    except (SyntaxError, ValueError):
        return set(), set()
    global_loads: set[str] = set()
    locally_bound: set[str] = set()

    def _visit(scope: symtable.SymbolTable) -> None:
        for sym in scope.get_symbols():
            name = sym.get_name()
            # ``is_global()`` covers implicit module-level frees + explicit ``global`` decls;
            # ``is_referenced()`` excludes a store-only ``global x; x = 5``.
            if sym.is_global() and sym.is_referenced():
                global_loads.add(name)
            # A local binding (parameter / assignment target / for-target / comp var) shadows the
            # module global in this scope.
            if sym.is_local() and not sym.is_global():
                locally_bound.add(name)
        for child in scope.get_children():
            _visit(child)

    _visit(table)
    ambiguous = global_loads & locally_bound
    return global_loads, ambiguous
