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
"""Integration tests for parser and printer round-trip."""

import pypto_pro
from pypto_pro import ir
import pypto_pro.language as pl


def test_function_printed_with_subscript_types():
    """Test that function parameters use subscript notation."""

    @pl.jit(auto_mutex=False)
    def test_func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
        result: pl.Tensor[[64, 128], pl.DT_FP32] = pl.make_tensor(x, [64, 128], [128, 1], dtype=pl.DT_FP32)
        _test_result = result

    test_func_program, _ = test_func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    test_func = test_func_program.get_function(test_func.__name__)

    # Print the function
    printed = pypto_pro.ir.python_print(test_func)

    # Check subscript notation is used
    assert "ir.Tensor[[64, 128], ir.FP16" in printed
    assert "ir.Tensor[[64, 128], ir.FP32" in printed
    # Check old notation is NOT used
    assert "ir.Tensor((" not in printed


def test_parsed_function_printer_round_trip():
    """Test that parsed functions can be printed correctly."""

    @pl.jit(auto_mutex=False)
    def round_trip(
        x: pl.Tensor[[64], pl.DT_FP32],
        y: pl.Tensor[[64], pl.DT_FP32],
    ):
        sum_val: pl.Tensor[[64], pl.DT_FP32] = pl.make_tensor(x, [64], [1])
        result: pl.Tensor[[64], pl.DT_FP32] = pl.make_tensor(sum_val, [64], [1])
        _test_result = result

    round_trip_program, _ = round_trip.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    round_trip = round_trip_program.get_function(round_trip.__name__)

    # Print and check syntax
    printed = pypto_pro.ir.python_print(round_trip)

    assert "def round_trip" in printed
    assert "ir.Tensor[[64], ir.FP32" in printed
    # Printer renders the op call it parsed
    assert "ptr.make_tensor" in printed


def test_while_loop_natural_syntax():
    """Test that natural while loop can be parsed and printed."""

    @pl.jit(auto_mutex=False)
    def while_natural(n: pl.DT_INT64):
        x: pl.DT_INT64 = 0
        while x < n:
            x = x + 1
        _test_result = x

    while_natural_program, _ = while_natural.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    while_natural = while_natural_program.get_function(while_natural.__name__)

    # Print the function
    printed = pypto_pro.ir.python_print(while_natural)

    # Natural while is represented as an SSA-carrying loop with an in-body exit guard.
    assert "while" in printed
    assert "< n" in printed
    assert "break" in printed

    # Verify structural properties
    assert isinstance(while_natural, ir.Function)
    assert while_natural.name == "while_natural"


def test_while_with_tensor_operations_round_trip():
    """Test while loop with tensor operations."""

    @pl.jit(auto_mutex=False)
    def while_tensors(n: pl.DT_INT64, x: pl.Tensor[[64], pl.DT_FP32]):
        i: pl.DT_INT64 = 0
        tile_type = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        acc = pl.make_tile(tile_type, addr=0)
        while i < n:
            i = i + 1
            pl.add(acc, acc, acc)
        _test_result = acc

    while_tensors_program, _ = while_tensors.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    while_tensors = while_tensors_program.get_function(while_tensors.__name__)

    # Print the function
    printed = pypto_pro.ir.python_print(while_tensors)

    # Should have the while loop and the op in its body
    assert "while" in printed
    assert "block.add" in printed
