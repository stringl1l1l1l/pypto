#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Python wrapper for the Function class.

This module provides a Pythonic interface to the C++ Function class, which represents
compiled functions in the PTO framework.
"""

from typing import List, Optional

from pypto import pypto_impl
from pypto.error import FeError


class Function:
    """Wrapper for the C++ Function class.

    Represents a compiled function in the PTO framework. This class provides
    access to function metadata, IR representations, and input/output tensors.

    Examples:
        >>> import pypto
        >>>
        >>> # Create a simple function
        >>> @pypto.frontend.jit
        ... def add_func(a: pypto.Tensor((128, 128), pypto.DT_FP32)) -> pypto.Tensor:
        ...     return pypto.add_s(a, pypto.element(pypto.DT_FP32, 1.0))
        >>>
        >>> # Get the last compiled function
        >>> func = pypto.GetLastFunction()
        >>> print(func.raw_name)
        >>> print(func.function_type_str)
        >>> print(func.dump())
    """

    def __init__(self, base: Optional[pypto_impl.Function] = None):
        """Initialize a Function wrapper.

        Args:
            base: The underlying C++ Function object. If None, this creates
                  an empty wrapper (not typically used directly).
        """
        self._base = base

    def __repr__(self) -> str:
        """Get a string representation of the Function.

        Returns:
            A descriptive string.
        """
        if self._base is None:
            return "<Function (uninitialized)>"
        return f"<Function '{self.raw_name}' (magic: {self.func_magic}, type: {self.function_type_str})>"

    def __str__(self) -> str:
        """Get a string representation of the Function.

        Returns:
            The function's raw name.
        """
        if self._base is None:
            return "Function(uninitialized)"
        return f"Function('{self.raw_name}')"

    @property
    def base(self) -> pypto_impl.Function:
        """Get the underlying C++ Function object.

        Returns:
            The C++ Function object.
        """
        if self._base is None:
            raise FeError(RuntimeError("Function base is None"))
        return self._base

    @property
    def magic_name(self) -> str:
        """Get the magic name of the function.

        The magic name is an internal unique identifier for the function.

        Returns:
            The magic name string.
        """
        return self.base.GetMagicName()

    @property
    def raw_name(self) -> str:
        """Get the raw name of the function.

        The raw name is the user-provided or generated function name.

        Returns:
            The raw name string.
        """
        return self.base.GetRawName()

    @property
    def func_magic(self) -> int:
        """Get the function magic number.

        The magic number is a unique integer identifier for the function.

        Returns:
            The magic number.
        """
        return self.base.GetFuncMagic()

    @property
    def function_type_str(self) -> str:
        """Get the function type as a string.

        Returns:
            The function type name (e.g., "STATIC", "DYNAMIC").
        """
        return self.base.GetFunctionTypeStr()

    @property
    def is_eager(self) -> bool:
        """Check if the function is eager.

        Returns:
            True if the function is eager, False otherwise.
        """
        return self.base.IsEager()

    @property
    def is_static(self) -> bool:
        """Check if the function is static.

        Returns:
            True if the function is static, False otherwise.
        """
        return self.base.IsStatic()

    @property
    def is_explicit(self) -> bool:
        """Check if the function is explicit.

        Returns:
            True if the function is explicit, False otherwise.
        """
        return self.base.IsExplicit()

    @property
    def has_parent(self) -> bool:
        """Check if the function has a parent.

        Returns:
            True if the function has a parent, False otherwise.
        """
        return self.base.HasParent()

    @property
    def root_function(self) -> "Function":
        """Get the root function.

        Returns:
            The root Function wrapper.
        """
        return Function.from_base(self.base.GetRootFunction())

    @property
    def incast(self) -> List:
        """Get input cast tensors.

        Returns:
            List of input tensors.
        """
        return self.base.GetIncast()

    @property
    def outcast(self) -> List:
        """Get output cast tensors.

        Returns:
            List of output tensors.
        """
        return self.base.GetOutcast()

    @property
    def origin_incast(self) -> List:
        """Get original input cast tensors.

        Returns:
            List of original input tensors.
        """
        return self.base.GetOriginIncast()

    @property
    def original_body(self):
        """Get the original function body statement.

        Returns:
            The original SeqStmts body of the function.
        """
        return self.base.original_body

    @property
    def origin_outcast(self) -> List:
        """Get original output cast tensors.

        Returns:
            List of original output tensors.
        """
        return self.base.GetOriginOutcast()

    @classmethod
    def from_base(cls, base: pypto_impl.Function) -> "Function":
        """Create a Function wrapper from a C++ Function object.

        Args:
            base: The C++ Function object to wrap.

        Returns:
            A new Function wrapper instance.
        """
        obj = cls.__new__(cls)
        obj._base = base
        return obj

    def dump(self) -> str:
        """Dump the function in brief format.

        Returns:
            A string representation of the function.
        """
        return self.base.Dump()

    def dump_ssa(self) -> str:
        """Dump the function in SSA (Static Single Assignment) format.

        Returns:
            SSA representation of the function.
        """
        return self.base.DumpSSA()

    def dump_file(self, file_path: str) -> None:
        """Dump the function to a file.

        Args:
            file_path: Path to the output file.
        """
        self.base.DumpFile(file_path)

    def dump_json_file(self, file_name: str = "") -> None:
        """Dump the function to a JSON file.

        Args:
            file_name: Path to the JSON file. If empty, a default name is used.
        """
        self.base.DumpJsonFile(file_name)

    def dump_attrs(self) -> dict:
        """Dump all attributes attached to the function.

        Returns:
            A dict mapping attribute names to their dumped values, e.g.
            {"IterNoOverlapRaw": "[3, 5]"}.
            Empty dict if no attribute has been recorded for this function.
        """
        return self.base.DumpAttrs()


def get_last_function() -> Optional[Function]:
    """Get the last compiled function from the Program.

    This is a convenience function that wraps the C++ GetLastFunction.

    Returns:
        The last compiled Function, or None if no function has been compiled yet.

    Examples:
        >>> import pypto
        >>>
        >>> @pypto.frontend.jit
        ... def my_func(a: pypto.Tensor((128, 128), pypto.DT_FP32)) -> pypto.Tensor:
        ...     return a + pypto.element(pypto.DT_FP32, 1.0)
        >>>
        >>> func = pypto.get_last_function()
        >>> print(func.raw_name)
    """
    base = pypto_impl.GetLastFunction()
    if base is None:
        return None
    return Function.from_base(base)


def get_current_function() -> Optional[Function]:
    """Get the current function being built in the Program.

    This is a convenience function that wraps the C++ GetCurrentFunction.

    Returns:
        The current Function being built, or None if no function is currently being built.

    Examples:
        >>> import pypto
        >>>
        >>> a = pypto.Tensor((128, 128), pypto.DT_FP32)
        >>> pypto.BeginFunction("test", pypto.GraphType.TENSOR_GRAPH, pypto.FunctionType.STATIC, a)
        >>> func = pypto.get_current_function()
        >>> print(func.raw_name)
        >>> pypto.EndFunction("test")
    """
    base = pypto_impl.GetCurrentFunction()
    if base is None:
        return None
    return Function.from_base(base)
