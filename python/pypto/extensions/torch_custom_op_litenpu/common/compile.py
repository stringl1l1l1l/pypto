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

"""Custom-op runtime compile entry.

``CompileEntry`` (constructed with either ``jit_kernel=`` or ``factory=``) satisfies the runtime
compile contract: a callable ``(shapes, dtypes, attrs, soc_version) -> path`` that the host runtime
invokes by module-attribute name.

``torch`` and ``pypto`` are resolved through this module's attributes at call time, so the unit
tests substitute lightweight fakes without an NPU toolchain (see
``python/tests/ut/torch_custom_op_litenpu/test_op_compile_selftest.py``).
"""
import inspect
import json
from typing import Any, Callable, Optional, get_origin
import warnings

import torch

import pypto

__all__ = ["CompileEntry"]


# ``str(torch.dtype)`` basename -> the pypto DataType attribute name; mirrors
# ``JitCallableWrapper._dtype_dict``. Both sides resolve by getattr at call time (the test seam above).
_TORCH_TO_PTO_DTYPE = {
    "float16": "DT_FP16",
    "float32": "DT_FP32",
    "bfloat16": "DT_BF16",
    "int8": "DT_INT8",
    "int16": "DT_INT16",
    "int32": "DT_INT32",
    "int64": "DT_INT64",
    "uint8": "DT_UINT8",
    "bool": "DT_BOOL",
}


def _torch_dtype(dtype_str: str):
    """``"float16"`` -> ``torch.float16``, restricted to the dtypes the pto mapping covers."""
    if dtype_str not in _TORCH_TO_PTO_DTYPE:
        raise ValueError(f"Unsupported dtype string '{dtype_str}'")
    return getattr(torch, dtype_str)


def _pto_dtype(torch_dtype) -> Any:
    """``str(torch.dtype)`` -> ``pypto.DataType``."""
    key = str(torch_dtype)
    name = _TORCH_TO_PTO_DTYPE.get(key.removeprefix("torch."))
    if name is None:
        raise ValueError(f"Unsupported dtype for custom op: {key}")
    return getattr(pypto, name)


def _as_tuple_of(result, n: int) -> tuple:
    """Normalize a hook return (a single shape/dtype, or a sequence of N) to a length-N tuple.

    A *single* shape is a flat sequence of ints, including the 0-d ``()`` case, or a non-sequence
    (e.g. a ``torch.dtype``); a *multi-output* return is a sequence of those.
    """
    is_seq = isinstance(result, (list, tuple))
    looks_single = (not is_seq) or all(isinstance(d, int) for d in result)  # () -> single 0-d shape
    if is_seq and not looks_single:
        out = tuple(result)            # sequence of N shapes/dtypes
    elif n == 1:
        out = (result,)                # the single shape/dtype (incl. flat-int shape and 0-d ())
    else:
        # A single value where n outputs are declared: refusing beats silently splitting a flat
        # shape like (8, 64) into n integer "shapes".
        raise ValueError(f"inference returned a single value but {n} outputs are declared")
    if len(out) != n:
        raise ValueError(f"inference returned {len(out)} values, expected {n}")
    return out


def _convert_shape_attr(raw, annotation, name: str):
    """Convert a stringified attr value to the typed value ``infer_shape`` expects, driven off
    the param annotation, the same conversion the kernel factory does (``list[int]`` ->
    ``json.loads``, ``int`` -> ``int``) so infer_shape and the factory get identical typed
    values. A non-string value (already typed) passes through."""
    if not isinstance(raw, str):
        return raw
    if annotation is list or get_origin(annotation) is list:
        return json.loads(raw)
    if annotation is int:
        return int(raw)
    if annotation is str:
        return raw
    # Unannotated shape-affecting param: accept a JSON-typed value, but never silently hand
    # infer_shape an unconverted string.
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise ValueError(
            f"attr '{name}': cannot convert {raw!r} for infer_shape; annotate the parameter "
            f"(int / str / list[int]) or pass a JSON-parseable value"
        ) from exc


class CompileEntry:
    """Callable runtime kernel-JIT entry: ``op(shapes, dtypes, attrs, soc_version) -> path str``.

    This is the Python callable the host runtime enters at op-compile time (via the embedded
    ``__pypto_compile``): it infers output shapes/dtypes, JIT-builds the pypto kernel for the
    runtime shape, and returns the compiled-kernel path (result cached per key). This is the
    runtime compile, distinct from any build-time kernel preparation.

    Pass exactly one of ``jit_kernel=`` (DIRECT: an already-built ``@pypto.frontend.jit`` kernel, no
    factory and no synthesis) or ``factory=`` (an existing ``create_*_kernel(shape, dtype,
    soc_version) -> jit_obj``), bind the instance to the module attribute name the host runtime
    looks the entry up by, and attach optional hooks via ``infer_shape`` / ``infer_dtype``::

        add_op_compile = CompileEntry(jit_kernel=add_kernel, num_inputs=2, num_outputs=1)
        add_op_compile = CompileEntry(factory=create_add_kernel, num_inputs=2)

    ``num_outputs`` is not derived from the jit wrapper's signature (a ``JitCallableWrapper`` may
    not expose a clean one); pass it explicitly for multi-output kernels. ``factory_signature``
    names which argument shape the factory takes: ``"single"`` = ``create(shape, dtype, soc)``,
    ``"lists"`` = ``create(shapes, dtypes, soc)``, ``"full"`` = ``create(shapes, dtypes, attrs,
    soc)``. For a factory taking none of these forms, pass a small adapter lambda.
    """

    def __init__(
        self,
        *,
        num_inputs: int,
        num_outputs: int = 1,
        factory: Optional[Callable] = None,
        jit_kernel: Optional[Callable] = None,
        factory_signature: str = "single",
        declared_annotations: Optional[list[dict]] = None,
    ):
        if (factory is None) == (jit_kernel is None):
            raise ValueError("Provide exactly one of `factory` or `jit_kernel`")
        if factory_signature not in ("single", "lists", "full"):
            raise ValueError("factory_signature must be 'single', 'lists', or 'full'")
        if num_inputs < 1:
            raise ValueError(f"num_inputs must be >= 1, got {num_inputs}")
        if num_outputs < 1:
            raise ValueError(f"num_outputs must be >= 1, got {num_outputs}")

        self._num_inputs = num_inputs
        self._num_outputs = num_outputs
        self._factory = factory
        self._jit_kernel = jit_kernel
        self._factory_signature = factory_signature
        # Author-declared explicit constants (see authoring._direct_declared_annotations); None = no checks.
        self._declared_annotations = declared_annotations
        self._infer_shape_fn: Optional[Callable] = None
        self._infer_dtype_fn: Optional[Callable] = None
        self._cache: dict[tuple, str] = {}

        # The entry's name in every error message: whichever of the two kernel forms was supplied.
        entry_target = jit_kernel if jit_kernel is not None else factory
        self.__name__ = getattr(entry_target, "__name__", "compile_entry")

    def __call__(self, shapes, dtypes, attrs, soc_version) -> str:
        """Compile for the runtime *shapes*/*dtypes* and return the compiled-kernel path (*soc_version* required)."""
        attrs = {} if attrs is None else dict(attrs)
        if not soc_version:
            raise ValueError(f"{self.__name__}: soc_version is required")
        shapes, dtype_strs = self._prepare(shapes, dtypes)

        # attrs are sorted so the key is order-independent: the same attr dict spelled in any order
        # is one cache entry, not several.
        key = (shapes, dtype_strs, tuple(sorted(attrs.items())), soc_version)
        if key in self._cache:
            return self._cache[key]

        jit_obj, out_shapes, out_torch = self.build_jit(shapes, dtype_strs, soc_version, attrs=attrs)
        # The deploy path always validates the hook's output dtypes: _pto_dtype raises for a dtype
        # the runtime cannot represent, whether or not the kernel declares annotations.
        for dtype in out_torch:
            _pto_dtype(dtype)

        # Dummy values are irrelevant for compilation; torch.empty is dtype-safe across all dtypes.
        in_torch = [_torch_dtype(s) for s in dtype_strs]
        dummies = [torch.empty(s, dtype=d, device="cpu") for s, d in zip(shapes, in_torch)]
        dummies += [torch.empty(s, dtype=d, device="cpu") for s, d in zip(out_shapes, out_torch)]

        path = str(jit_obj.build_kernel(*dummies))
        self._cache[key] = path
        return path

    @property
    def infer_shape(self) -> Optional[Callable]:
        """The output-shape hook, or ``None`` when the op declares none.

        *fn* takes one shape (a ``tuple[int]``) per input, plus one trailing parameter per
        shape-affecting attr (matched to the attr by name, converted per its annotation), and
        returns the output shape, or a sequence of ``num_outputs`` shapes for multi-output ops::

            def add_infer_shape(a_shape, b_shape):
                return a_shape

            add_op_compile.infer_shape = add_infer_shape

        Without a hook every output takes ``shapes[0]``.
        """
        return self._infer_shape_fn

    @infer_shape.setter
    def infer_shape(self, fn: Callable) -> None:
        # A non-callable would otherwise surface far from this assignment (deep inside dispatch),
        # so reject it here; ``None`` stays valid because a hook-less op legitimately has none.
        if fn is not None and not callable(fn):
            raise TypeError(f"infer_shape must be callable or None, got {type(fn).__name__}")
        self._infer_shape_fn = fn

    @property
    def infer_dtype(self) -> Optional[Callable]:
        """The output-dtype hook, or ``None`` when the op declares none.

        *fn* takes one ``torch.dtype`` per input and returns the output dtype, or a sequence of
        ``num_outputs`` dtypes for multi-output ops::

            def norm_infer_dtype(x_dtype, w_dtype):
                return (x_dtype, torch.float32)

            norm_op_compile.infer_dtype = norm_infer_dtype

        Without a hook every output takes the first input's dtype.
        """
        return self._infer_dtype_fn

    @infer_dtype.setter
    def infer_dtype(self, fn: Callable) -> None:
        # As with infer_shape: a non-callable is rejected at the assignment rather than deep in
        # dispatch, while ``None`` keeps meaning "no hook, every output takes the first input's dtype".
        if fn is not None and not callable(fn):
            raise TypeError(f"infer_dtype must be callable or None, got {type(fn).__name__}")
        self._infer_dtype_fn = fn

    def _prepare(self, shapes, dtypes):
        shapes = tuple(tuple(s) for s in shapes)
        dtype_strs = tuple(dtypes)
        if len(shapes) != self._num_inputs or len(dtype_strs) != self._num_inputs:
            raise ValueError(
                f"{self.__name__}: got {len(shapes)} shapes / {len(dtype_strs)} dtypes, "
                f"expected num_inputs={self._num_inputs}"
            )
        return shapes, dtype_strs

    def infer_outputs(self, shapes, dtypes, attrs=None):
        """Return ``(out_shapes, out_torch_dtypes)`` for *shapes*/*dtypes*; inference only, no jit build.

        Used by the run path to allocate the output tensors when the jit object is produced elsewhere
        (the DIRECT-on-NPU re-wrap), so no process-global soc is touched.
        """
        shapes, dtype_strs = self._prepare(shapes, dtypes)
        return self._infer_out_shapes_dtypes(shapes, [_torch_dtype(s) for s in dtype_strs], attrs)

    def build_jit(self, shapes, dtypes, soc_version=None, run_mode=None, attrs=None):
        """Build the pypto jit kernel for *shapes*/*dtypes* and return ``(jit_obj, out_shapes, out_torch_dtypes)``.

        Layers ``infer_outputs`` with the declared-annotation guard and the jit build (``__call__`` in turn
        layers the dummies and ``build_kernel`` on top of this), and returns the live jit object and the
        inferred output shapes/dtypes instead of a compiled ``.o`` path; the run path allocates the outputs
        from these and invokes the jit kernel directly.

        *run_mode* (a ``pypto.RunMode``) is forwarded only for the factory form (its inner jit binds it as a
        closure param); DIRECT kernels pin run_mode at decoration, so the caller re-wraps for NPU separately.

        ``soc_version`` is forwarded unchanged on this run path: a ``None`` soc stays ``None`` (pinning a
        SIM soc onto an NPU factory build would mis-target it). When ``None`` the factory's inner
        ``@jit(codegen_options={"soc_version": None})`` lets ``set_codegen_options`` drop the key so the
        framework auto-detects on-device. (The deploy/compile path is ``__call__``, which requires a soc.)
        """
        attrs = {} if attrs is None else dict(attrs)
        shapes, dtype_strs = self._prepare(shapes, dtypes)
        in_pto = [_pto_dtype(_torch_dtype(s)) for s in dtype_strs]
        out_shapes, out_torch = self.infer_outputs(shapes, dtype_strs, attrs)
        if self._declared_annotations:
            out_pto = [_pto_dtype(d) for d in out_torch]
            self._check_declared_mismatches(shapes, in_pto, out_shapes, out_pto)
        if self._jit_kernel is not None:
            # The author's jit kernel is used as-is (the dummies drive every tensor's shape and dtype);
            # only the process-global soc is pinned. DIRECT mode therefore requires a jit decorator
            # that pins no ``soc_version``: pypto's per-wrapper setter would otherwise clobber the
            # global at build time and silently mis-target the build.
            pypto.set_codegen_options(soc_version=soc_version)
            jit_obj = self._jit_kernel
        else:
            jit_obj = self._build_jit_from_factory(shapes, in_pto, soc_version, attrs, run_mode=run_mode)
        return jit_obj, out_shapes, out_torch

    # ---- internals ----
    def _infer_shape_run_args(self, in_shapes, attrs):
        """Positional args for an ``infer_shape`` call: the tensor shapes, then any shape-affecting
        attr values (the infer_shape params beyond the tensor count ``_num_inputs``), each looked
        up in *attrs* by param name and converted from its stringified form. A shape-invariant op
        has no trailing infer_shape params, so this returns just the shapes (call is
        ``infer_shape(*shapes)``)."""
        args = list(in_shapes)
        if self._infer_shape_fn is None:
            return args
        trailing = list(inspect.signature(self._infer_shape_fn).parameters.values())[self._num_inputs:]
        if not trailing:
            return args
        attrs = attrs or {}
        for p in trailing:
            args.append(_convert_shape_attr(attrs[p.name], p.annotation, p.name))
        return args

    def _infer_out_shapes_dtypes(self, in_shapes, in_torch_dtypes, attrs=None) -> tuple:
        """``(out_shapes, out_torch_dtypes)``; without a hook every output takes the first input's value."""
        if self._infer_shape_fn is not None:  # attrs feed a shape-affecting infer_shape
            out_shapes = _as_tuple_of(
                self._infer_shape_fn(*self._infer_shape_run_args(in_shapes, attrs)), self._num_outputs
            )
        else:
            out_shapes = tuple(in_shapes[0] for _ in range(self._num_outputs))
        if self._infer_dtype_fn is not None:
            out_dtypes = _as_tuple_of(self._infer_dtype_fn(*in_torch_dtypes), self._num_outputs)
        else:
            out_dtypes = tuple(in_torch_dtypes[0] for _ in range(self._num_outputs))
        return out_shapes, out_dtypes

    def _check_declared_mismatches(self, in_shapes, in_pto, out_shapes, out_pto) -> None:
        """Enforce author-declared explicit-constant annotations against the runtime values.

        Only fires for explicit constants (a literal shape / a ``DT_*`` dtype recorded at synthesis);
        placeholder and variable annotations recorded ``None`` and are skipped. Policy:

        * a dtype mismatch raises, since a pinned ``explicit_dtype`` silently overrides the dtype the
          runtime hands the op (a genuine correctness hazard), so it is a hard error (input and output);
        * a shape mismatch warns, since a pinned shape is informational and the dummy drives the real shape.
        """
        # declared_annotations has one entry per kernel positional param (inputs then output slots);
        # zip against (inputs + inferred outputs). Output-slot entries are always erased
        # ({None,None}), so any length divergence only drops non-checkable None entries.
        tensors = list(zip(in_shapes, in_pto)) + list(zip(out_shapes, out_pto))
        for idx, (decl, (shp, pto)) in enumerate(zip(self._declared_annotations, tensors)):
            kind = "input" if idx < self._num_inputs else "output"
            j = idx if kind == "input" else idx - self._num_inputs
            d_shape = decl.get("shape")
            if d_shape is not None and list(d_shape) != list(shp):
                warnings.warn(
                    f"{self.__name__}: declared {kind}[{j}] shape {list(d_shape)} != runtime {list(shp)}",
                    stacklevel=2,
                )
            d_dtype = decl.get("dtype")
            if d_dtype is not None:
                # Resolve the declared "DT_*" token to a pypto.DataType and compare the enums directly
                # (the enum stringifies to its int value, so compare objects, not str forms).
                resolved = getattr(pypto, d_dtype, None)
                if resolved is not None and resolved != pto:
                    runtime_name = getattr(pto, "name", None) or str(pto)
                    raise ValueError(
                        f"{self.__name__}: declared {kind}[{j}] dtype {d_dtype} != runtime {runtime_name}; "
                        f"a pinned dtype silently overrides the runtime dtype, fix the pinned annotation "
                        f"or infer_dtype so they agree"
                    )

    def _build_jit_from_factory(self, shapes, in_pto, soc_version, attrs, run_mode=None):
        # run_mode is forwarded only when explicitly requested (the run path's NPU mode); the factory's
        # own default (SIM) stands otherwise.
        kw = {} if run_mode is None else {"run_mode": run_mode}
        if self._factory_signature == "single":      # create(shape, dtype, soc[, run_mode])
            return self._factory(shapes[0], in_pto[0], soc_version, **kw)
        if self._factory_signature == "full":        # create(shapes, dtypes, attrs, soc[, run_mode]); attrs-aware
            return self._factory(list(shapes), list(in_pto), dict(attrs), soc_version, **kw)
        return self._factory(list(shapes), list(in_pto), soc_version, **kw)  # "lists": create(shapes, dtypes, soc)
