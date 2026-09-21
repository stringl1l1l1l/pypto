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
"""Shared custom-op authoring constants, validation and kernel-form helpers.

Dependency-free pieces the ``ExportedCustomOp`` authoring path needs: the default ONNX
custom-domain coordinates a pypto op exports under, the op_type identifier validation, and the
``kernel=`` form helpers (jit-kernel detection, factory-signature classification, kernel-name
derivation). Pure ``ast``/``inspect``/``textwrap`` logic; this module stays a leaf.
"""
import ast
import inspect
import re
import textwrap

# Default ONNX custom-domain coordinates for a pypto op's node (overridable per op at authoring time).
_DEFAULT_DOMAIN = "pypto"
_DEFAULT_DOMAIN_OPSET_VERSION = 1

_OP_TYPE_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_op_type_identifier(op_type: str) -> None:
    """Ensure *op_type* is a safe program identifier (letters, digits, underscore; not digit-leading)."""
    if not op_type or not _OP_TYPE_ID_RE.match(op_type):
        raise ValueError(
            "op_type must be a non-empty identifier (letters, digits, underscore; "
            f"must not start with a digit), got {op_type!r}"
        )


def _is_jit_kernel(fn) -> bool:
    """True if *fn* is a ``@pypto.frontend.jit`` object (the bare-kernel form passed to ``kernel=``).

    A jit object is a ``JitCallableWrapper`` instance carrying the raw function as
    ``_original_func``; a factory is a plain function. Duck-typed to avoid a frontend import.
    """
    return (not inspect.isfunction(fn)) and hasattr(fn, "_original_func")


def _wrapped_jit_kernel_name(op):
    """The name of the function ``@pypto.frontend.jit`` wraps, the default kernel_name for *op*.

    DIRECT: the jit kernel itself. FACTORY: the inner jit def the factory defines and returns —
    found by scanning the factory's source for the first nested def whose decorators include a
    ``.jit`` attribute-call, alongside whether any ``return`` carries a value (a factory may bind
    then return, branch, or return more than once). The raise below is also the construction-time
    gate rejecting a ``kernel=`` plain function that is not a factory: no inner jit def, no value
    return, or source that does not parse as a plain function.
    """
    if op._bare_kernel_fn is not None:
        return op._bare_kernel_fn._original_func.__name__
    inner_jit_name = None
    has_value_return = False
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(op._create_kernel_fn)))
    except (OSError, TypeError, SyntaxError):
        tree = None
    top = tree.body[0] if tree is not None and tree.body else None
    if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for node in ast.walk(top):
            if node is top:
                continue
            if isinstance(node, ast.Return) and node.value is not None:
                has_value_return = True
            if inner_jit_name is None and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    call = dec.func if isinstance(dec, ast.Call) else dec
                    if isinstance(call, ast.Attribute) and call.attr == "jit":
                        inner_jit_name = node.name
                        break
    if inner_jit_name is None or not has_value_return:
        raise ValueError(
            "kernel=: could not derive kernel_name from the factory; it must define and return a "
            "@pypto.frontend.jit-decorated inner kernel (whose name becomes the op's kernel_name). "
            "For a direct kernel, decorate the kernel itself with @pypto.frontend.jit."
        )
    return inner_jit_name


def _direct_declared_annotations(op):
    """Extract author-pinned literal shapes/dtypes from a DIRECT jit kernel's annotations, or ``None``.

    Reads the cached tensor-defs (``op._bare_kernel_fn._cached_signature[0]``, the inputs-then-output
    param list in the order ``CompileEntry._check_declared_mismatches`` zips against) and returns one
    ``{"shape": [int]|None, "dtype": "DT_*"|None}`` per param: ``shape`` only when every dim is a
    concrete int, ``dtype`` only when one was pinned. ``None`` when no param pins anything.
    """
    kernel = op._bare_kernel_fn
    if kernel is None:
        return None
    defs = kernel._cached_signature[0]
    annotations = []
    any_pinned = False
    for d in defs:
        raw_shape = getattr(d, "shape", None)
        if raw_shape is not None and all(isinstance(dim, int) for dim in raw_shape):
            shape = [int(dim) for dim in raw_shape]
        else:
            shape = None
        explicit_dtype = getattr(d, "explicit_dtype", None)
        dtype = explicit_dtype.name if explicit_dtype is not None else None
        if shape is not None or dtype is not None:
            any_pinned = True
        annotations.append({"shape": shape, "dtype": dtype})
    return annotations if any_pinned else None


def _detect_factory_signature(fn) -> str:
    """Classify a factory as ``"single"`` / ``"lists"`` / ``"full"`` from its parameter names.

    Detection is by name, so renaming a factory parameter changes its classification. The three
    forms are documented in ``CompileEntry``.
    """
    params = list(inspect.signature(fn).parameters)
    if "attrs" in params:
        return "full"
    if params and params[0] == "shapes":
        return "lists"
    if params and params[0] == "shape":
        return "single"
    raise ValueError(
        f"kernel= factory {getattr(fn, '__name__', '?')!r} has an unrecognized signature {params}; "
        "it must start with (shape, dtype, ...) [single], (shapes, dtypes, ...) [lists], or "
        "(shapes, dtypes, attrs, ...) [full]"
    )
