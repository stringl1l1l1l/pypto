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
"""Frontend validation tests for pl.struct / pl.struct_array field constraints.

Covers the parser-side rejections added in _struct_parser.py:
  - C++ keyword as struct type name / field name
  - nested named tuple/struct as a field value
  - empty array field
  - mixed-dtype array field
Plus assignment-time rejections in _assignment_parser.py:
  - dtype mismatch on struct.set
  - scalar/list/tensor written to the wrong field kind
Plus positive cases confirming valid declarations are not rejected.
"""

from __future__ import annotations

from pypto_pro import ir
from pypto_pro._errors import InvalidOperation, InvalidType, InvalidVal, NotSupported
import pypto_pro.language as pl
import pytest


def _parse(kernel):
    """Trigger frontend parsing of a @pl.jit kernel."""
    kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


# =============================================================================
# C++ keyword as struct type name / field name
# =============================================================================

def test_err_struct_field_name_is_cpp_keyword():
    """A field named after a C++ keyword ('int') must be rejected."""

    with pytest.raises(InvalidOperation, match="C\\+\\+ keyword"):
        @pl.jit(auto_mutex=False)
        def kernel(_jit_entry: pl.DT_INT64):
            s = pl.struct("S", int=1, true=2, delete=3)
            _test_result = s.int

        _parse(kernel)


def test_err_struct_type_name_is_cpp_keyword():
    """A struct type name that is a C++ keyword ('class') must be rejected."""

    with pytest.raises(InvalidOperation, match="C\\+\\+ keyword"):
        @pl.jit(auto_mutex=False)
        def kernel(_jit_entry: pl.DT_INT64):
            s = pl.struct("class", v=1)
            _test_result = s.v

        _parse(kernel)


def test_err_struct_array_field_name_is_cpp_keyword():
    """struct_array field named after a C++ keyword must be rejected."""

    with pytest.raises(InvalidOperation, match="C\\+\\+ keyword"):
        @pl.jit(auto_mutex=False)
        def kernel(_jit_entry: pl.DT_INT64):
            arr = pl.struct_array(2, "S", new=0)
            _test_result = arr[0].new

        _parse(kernel)


def test_err_struct_array_type_name_is_cpp_keyword():
    """struct_array type name that is a C++ keyword must be rejected."""

    with pytest.raises(InvalidOperation, match="C\\+\\+ keyword"):
        @pl.jit(auto_mutex=False)
        def kernel(_jit_entry: pl.DT_INT64):
            arr = pl.struct_array(2, "delete", v=0)
            _test_result = arr[0].v

        _parse(kernel)


# =============================================================================
# Nested named tuple / struct as a field value
# =============================================================================

def test_err_struct_nested_make_tuple_field():
    """A field whose value is a make_tuple result must be rejected."""

    with pytest.raises(NotSupported, match="nested named tuple/struct"):
        @pl.jit(auto_mutex=False)
        def kernel(_jit_entry: pl.DT_INT64):
            t = pl.make_tuple(x=1)
            s = pl.struct("S", t=t)
            _test_result = s.t

        _parse(kernel)


def test_err_struct_nested_struct_field():
    """A field whose value is another struct result must be rejected."""

    with pytest.raises(NotSupported, match="nested named tuple/struct"):
        @pl.jit(auto_mutex=False)
        def kernel(_jit_entry: pl.DT_INT64):
            inner = pl.struct("Inner", a=0)
            s = pl.struct("Outer", inner=inner)
            _test_result = s.inner

        _parse(kernel)


def test_err_struct_array_nested_make_tuple_field():
    """struct_array field whose value is a make_tuple result must be rejected."""

    with pytest.raises(NotSupported, match="nested named tuple/struct"):
        @pl.jit(auto_mutex=False)
        def kernel(_jit_entry: pl.DT_INT64):
            t = pl.make_tuple(x=1)
            arr = pl.struct_array(2, "S", t=t)
            _test_result = arr[0].t

        _parse(kernel)


# =============================================================================
# Empty array field
# =============================================================================

def test_err_struct_empty_array_field():
    """An empty array field literal must be rejected."""

    with pytest.raises(InvalidVal, match="empty"):
        @pl.jit(auto_mutex=False)
        def kernel(_jit_entry: pl.DT_INT64):
            s = pl.struct("S", arr=[])
            _test_result = s.arr

        _parse(kernel)


def test_err_struct_array_empty_array_field():
    """An empty array field literal in struct_array must be rejected."""

    with pytest.raises(InvalidVal, match="empty"):
        @pl.jit(auto_mutex=False)
        def kernel(_jit_entry: pl.DT_INT64):
            arr = pl.struct_array(2, "S", data=[])
            _test_result = arr[0].data

        _parse(kernel)


# =============================================================================
# Mixed-dtype array field
# =============================================================================

def test_err_struct_mixed_int_float_array_field():
    """A mixed int/float array field ([1, 2.5, 3]) must be rejected."""

    with pytest.raises(InvalidVal, match="mixed element types"):
        @pl.jit(auto_mutex=False)
        def kernel(_jit_entry: pl.DT_INT64):
            s = pl.struct("S", arr=[1, 2.5, 3])
            _test_result = s.arr[0]

        _parse(kernel)


def test_err_struct_mixed_bool_int_array_field():
    """A mixed bool/int array field ([True, 2]) must be rejected."""

    with pytest.raises(InvalidVal, match="mixed element types"):
        @pl.jit(auto_mutex=False)
        def kernel(_jit_entry: pl.DT_INT64):
            s = pl.struct("S", arr=[True, 2])
            _test_result = s.arr[0]

        _parse(kernel)


def test_err_struct_array_mixed_dtype_array_field():
    """A mixed-dtype array field in struct_array must be rejected."""

    with pytest.raises(InvalidVal, match="mixed element types"):
        @pl.jit(auto_mutex=False)
        def kernel(_jit_entry: pl.DT_INT64):
            arr = pl.struct_array(2, "S", data=[1, 2.5])
            _test_result = arr[0].data[0]

        _parse(kernel)


# =============================================================================
# Multi-dimensional / non-scalar array field
# =============================================================================

def test_err_struct_multidim_array_field():
    """A 2D array field ([[1, 2], [3, 4]]) must be rejected."""

    with pytest.raises(InvalidType, match="non-scalar elements"):
        @pl.jit(auto_mutex=False)
        def kernel(_jit_entry: pl.DT_INT64):
            s = pl.struct("S", m=[[1, 2], [3, 4]])
            _test_result = s.m

        _parse(kernel)


def test_err_struct_array_multidim_array_field():
    """A 2D array field in struct_array must be rejected."""

    with pytest.raises(InvalidType, match="non-scalar elements"):
        @pl.jit(auto_mutex=False)
        def kernel(_jit_entry: pl.DT_INT64):
            arr = pl.struct_array(2, "S", m=[[1, 2], [3, 4]])
            _test_result = arr[0].m

        _parse(kernel)


# =============================================================================
# Non-scalar field value (tensor / tile)
# =============================================================================

def test_err_struct_tensor_field():
    """A whole tensor as a field value must be rejected."""

    with pytest.raises(InvalidType, match="must be a scalar or a fixed-size array"):
        @pl.jit(auto_mutex=False)
        def kernel(a_t: pl.Tensor[[16], pl.DT_FP16]):
            s = pl.struct("S", data=a_t)
            _test_result = s.data

        _parse(kernel)


def test_err_struct_array_tensor_field():
    """A whole tensor as a struct_array field value must be rejected."""

    with pytest.raises(InvalidType, match="must be a scalar or a fixed-size array"):
        @pl.jit(auto_mutex=False)
        def kernel(a_t: pl.Tensor[[16], pl.DT_FP16]):
            arr = pl.struct_array(2, "S", data=a_t)
            _test_result = arr[0].data

        _parse(kernel)


# =============================================================================
# Field write dtype mismatch (struct.set)
# =============================================================================

def test_err_struct_field_write_float_to_int():
    """Writing a float to an int-locked field must be rejected."""

    with pytest.raises(InvalidType, match="Struct field 'scale' expects type"):
        @pl.jit(auto_mutex=False)
        def kernel(_jit_entry: pl.DT_INT64):
            s = pl.struct("Cfg", scale=1)
            s.scale = 3.5
            _test_result = s.scale

        _parse(kernel)


def test_err_struct_field_write_int_to_float():
    """Writing an int to a float-locked field must be rejected (exact-match rule)."""

    with pytest.raises(InvalidType, match="Struct field 'scale' expects type"):
        @pl.jit(auto_mutex=False)
        def kernel(_jit_entry: pl.DT_INT64):
            s = pl.struct("Cfg", scale=1.0)
            s.scale = 1
            _test_result = s.scale

        _parse(kernel)


def test_err_struct_array_field_write_dtype_mismatch():
    """Whole-array assignment with mismatched element dtype must be rejected."""

    with pytest.raises(InvalidType, match="Struct field 'a' expects type"):
        @pl.jit(auto_mutex=False)
        def kernel(_jit_entry: pl.DT_INT64):
            s = pl.struct("S", a=[1, 2])
            s.a = [1.0, 2.0]
            _test_result = s.a[0]

        _parse(kernel)


def test_err_struct_array_field_element_write_dtype_mismatch():
    """s.arr[i] = v with a mismatched element dtype must be rejected."""

    with pytest.raises(InvalidType, match="Struct field 'a' expects type"):
        @pl.jit(auto_mutex=False)
        def kernel(_jit_entry: pl.DT_INT64):
            s = pl.struct("S", a=[1, 2])
            s.a[0] = 1.0
            _test_result = s.a[0]

        _parse(kernel)


def test_err_struct_array_field_element_write_non_scalar():
    """s.arr[i] = tensor must be rejected; element writes require a scalar."""

    with pytest.raises(InvalidType, match="Struct field 'a' expects type"):
        @pl.jit(auto_mutex=False)
        def kernel(a_t: pl.Tensor[[16], pl.DT_INT64]):
            s = pl.struct("S", a=[1, 2])
            s.a[0] = a_t
            _test_result = s.a[0]

        _parse(kernel)


def test_ok_struct_field_write_same_dtype():
    """Writing a float to a float-locked field parses without error."""

    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        s = pl.struct("Cfg", scale=1.0)
        s.scale = 3.5
        _test_result = s.scale

    _parse(kernel)


def test_err_struct_array_field_write_scalar():
    """Assigning a scalar to an array field must be rejected."""

    with pytest.raises(InvalidType, match="Struct field 'arr' expects type"):
        @pl.jit(auto_mutex=False)
        def kernel(_jit_entry: pl.DT_INT64):
            s = pl.struct("S", arr=[0, 0, 0])
            s.arr = 1
            _test_result = s.arr[0]

        _parse(kernel)


def test_err_struct_array_field_write_nested_list():
    """Assigning a nested list to a 1-D array field must be rejected."""

    with pytest.raises(InvalidType, match="non-scalar elements"):
        @pl.jit(auto_mutex=False)
        def kernel(_jit_entry: pl.DT_INT64):
            s = pl.struct("S", arr=[0, 0])
            s.arr = [[1, 2], [3, 4]]
            _test_result = s.arr[0]

        _parse(kernel)


def test_err_struct_scalar_field_write_list():
    """Assigning a list to a scalar field must be rejected."""

    with pytest.raises(InvalidType, match="Struct field 'scale' expects type"):
        @pl.jit(auto_mutex=False)
        def kernel(_jit_entry: pl.DT_INT64):
            s = pl.struct("S", scale=0)
            s.scale = [1, 2, 3]
            _test_result = s.scale

        _parse(kernel)


def test_err_struct_scalar_field_write_tensor():
    """Assigning a whole tensor to a scalar field must be rejected."""

    with pytest.raises(InvalidType, match="must be a scalar or a fixed-size array"):
        @pl.jit(auto_mutex=False)
        def kernel(a_t: pl.Tensor[[16], pl.DT_INT64]):
            s = pl.struct("S", scale=0)
            s.scale = a_t
            _test_result = s.scale

        _parse(kernel)


# =============================================================================
# Positive cases — valid declarations must NOT be rejected
# =============================================================================

def test_ok_struct_scalar_and_uniform_array_fields():
    """Scalar fields and a same-dtype array field parse without error."""

    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        s = pl.struct("RunInfo", batch_id=0, offsets=[0, 0, 0, 0])
        _test_result = s.offsets[0] + s.batch_id

    _parse(kernel)


def test_ok_struct_float_array_field():
    """A same-dtype float array field parses without error."""

    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        s = pl.struct("S", arr=[1.0, 2.5, 3.0])
        _test_result = s.arr[0]

    _parse(kernel)


def test_ok_struct_array_uniform_fields():
    """struct_array with scalar and same-dtype array fields parses without error."""

    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        arr = pl.struct_array(2, "S", batch_id=0, data=[0, 0, 0])
        _test_result = arr[0].data[0] + arr[1].batch_id

    _parse(kernel)
