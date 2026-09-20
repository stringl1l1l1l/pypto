#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Normalize whitespace around colons and commas in subscript expressions.

Enforces ``a[0:mid + 1]`` instead of ``a[0 : mid + 1]`` and ``a[1:2, :]``
instead of ``a[1:2,:]``.  Complements ruff E203 (which skips balanced
slices) and ruff-format (which preserves spaces in complex slices).
"""

import argparse
import io
import logging
import sys
import tokenize


def _in_subscript(bracket_stack):
    """Return True if the innermost non-paren bracket is ``[``."""
    for b in reversed(bracket_stack):
        if b == "(":
            continue
        return b == "["
    return False


def _update_bracket_stack(bracket_stack, tok_string):
    """Update *bracket_stack* for an opening/closing bracket.

    Return True if *tok_string* is a bracket token.
    """
    if tok_string in ("[", "{", "("):
        bracket_stack.append(tok_string)
        return True
    if tok_string in ("]", "}", ")"):
        if bracket_stack:
            bracket_stack.pop()
        return True
    return False


def _prev_token(tokens, i):
    return tokens[i - 1] if i > 0 else None


def _next_token(tokens, i):
    return tokens[i + 1] if i + 1 < len(tokens) else None


def _colon_space_fixes(tokens, i, tok):
    """Strip spaces around a subscript colon, except after a comma."""
    row, col = tok.start
    end_col = tok.end[1]
    prev_tok = _prev_token(tokens, i)
    next_tok = _next_token(tokens, i)
    fixes = []
    # Remove space before colon, unless the previous token is a comma:
    # a comma in a subscript must keep a following space (e.g. a[1:2, :]),
    # which is handled by the comma rule below.
    if (
        prev_tok
        and prev_tok.string != ","
        and prev_tok.end[0] == row
        and prev_tok.end[1] < col
    ):
        fixes.append((row - 1, prev_tok.end[1], col, ""))
    if next_tok and next_tok.start[0] == row and end_col < next_tok.start[1]:
        fixes.append((row - 1, end_col, next_tok.start[1], ""))
    return fixes


def _comma_space_fixes(tokens, i, tok):
    """Insert a space after a subscript comma when the next token is adjacent."""
    row = tok.start[0]
    end_col = tok.end[1]
    next_tok = _next_token(tokens, i)
    # Ensure a space after the comma when the next token is on the same
    # line and is not a closing bracket or a line break
    # (e.g. a[1:2,:] -> a[1:2, :]).
    if (
        next_tok
        and next_tok.type not in (tokenize.NL, tokenize.NEWLINE)
        and next_tok.start[0] == row
        and next_tok.string not in ("]", "}", ")")
        and end_col == next_tok.start[1]
    ):
        return [(row - 1, end_col, end_col, " ")]
    return []


def _find_fixes(source):
    """Return ``(line_idx, start_col, end_col, replacement)`` edits to apply."""
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    bracket_stack = []
    fixes = []

    for i, tok in enumerate(tokens):
        if tok.type != tokenize.OP:
            continue
        if _update_bracket_stack(bracket_stack, tok.string):
            continue
        if tok.string == ":" and _in_subscript(bracket_stack):
            fixes.extend(_colon_space_fixes(tokens, i, tok))
        elif tok.string == "," and _in_subscript(bracket_stack):
            fixes.extend(_comma_space_fixes(tokens, i, tok))
    return fixes


def _apply(source, fixes):
    if not fixes:
        return source
    lines = source.splitlines(keepends=True)
    # Process right-to-left within each line so earlier edits do not shift
    # the column offsets of later ones. For equal start columns, deletions
    # (end_col > start_col) must precede zero-width insertions.
    fixes.sort(key=lambda f: (f[0], -f[1], -f[2]))
    for line_idx, start_col, end_col, replacement in fixes:
        line = lines[line_idx]
        lines[line_idx] = line[:start_col] + replacement + line[end_col:]
    return "".join(lines)


def fix_file(filepath):
    """Fix *filepath* in place. Return True if it was modified."""
    with open(filepath, "r", encoding="utf-8") as fh:
        source = fh.read()
    try:
        new_source = _apply(source, _find_fixes(source))
    except tokenize.TokenError:
        return False
    if new_source == source:
        return False
    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(new_source)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    modified = False
    for fp in args.files:
        if fix_file(fp):
            logging.info("Fixed slice colon spacing: %s", fp)
            modified = True
    sys.exit(1 if modified else 0)


if __name__ == "__main__":
    main()
