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
"""Unit tests for explicit-output block operations."""

import inspect

from pypto_pro import ir
from pypto_pro._errors import (
    InvalidArgument,
    InvalidFormat,
    InvalidShape,
    InvalidTile,
    InvalidType,
    InvalidVal,
    NotSupported,
)
import pypto_pro.language as pl
import pytest

from pypto.pypto_impl import ir as _ir


def _program_ir(func: ir.Function) -> str:
    return str(func)


def _find_call(func: ir.Function, name: str):
    return next(stmt.expr for stmt in func.body.stmts if isinstance(stmt, ir.EvalStmt) and stmt.expr.name == name)


def test_coordinate_apis_use_sequence_parameters():
    assert list(inspect.signature(pl.insert).parameters) == [
        "dst_tile",
        "src_tile",
        "offset",
        "relu_pre_mode",
        "scale",
        "phase",
    ]
    assert list(inspect.signature(pl.set_validshape).parameters) == ["tile", "shape"]


def test_matmul_acc_rejects_mismatched_acc_shape():
    with pytest.raises(InvalidType, match=r"acc_tile type must match dst_tile type"):

        @pl.jit(auto_mutex=False)
        def main(_jit_entry: pl.DT_INT64):
            dst_type = pl.TileType(
                shape=[64, 64],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Acc,
                layout=pl.NZ,
                fractal=1024,
            )
            acc_type = pl.TileType(
                shape=[32, 64],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Acc,
                layout=pl.NZ,
                fractal=1024,
            )
            lhs_type = pl.TileType(
                shape=[64, 64],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Left,
                layout=pl.NZ,
            )
            rhs_type = pl.TileType(
                shape=[64, 64],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Right,
                layout=pl.ZN,
            )
            dst = pl.make_tile(dst_type, addr=0x0000)
            acc = pl.make_tile(acc_type, addr=0x0000)
            lhs = pl.make_tile(lhs_type, addr=0x0000)
            rhs = pl.make_tile(rhs_type, addr=0x0000)
            pl.matmul_acc(dst, acc, lhs, rhs)

        main.to_kernel_def().parse_target_program(ir.SectionKind.Cube)


def test_manual_add():
    @pl.jit(auto_mutex=False)
    def main(
        a: pl.Tensor[[128, 128], pl.DT_FP32],
        b: pl.Tensor[[128, 128], pl.DT_FP32],
        output: pl.Tensor[[128, 128], pl.DT_FP32],
    ):
        tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
        tile_a = pl.make_tile(tile_type, addr=0x0000)
        tile_b = pl.make_tile(tile_type, addr=0x1000)
        tile_out = pl.make_tile(tile_type, addr=0x2000)
        pl.load(tile_a, a, [0, 0])
        pl.load(tile_b, b, [0, 0])
        pl.add(tile_out, tile_a, tile_b)
        _test_result = pl.store(output, tile_out, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    assert "block.add" in _program_ir(main)


def test_manual_mul_scalar():
    @pl.jit(auto_mutex=False)
    def main(
        a: pl.Tensor[[128, 128], pl.DT_FP32],
        output: pl.Tensor[[128, 128], pl.DT_FP32],
    ):
        tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
        tile_a = pl.make_tile(tile_type, addr=0x3000)
        tile_out = pl.make_tile(tile_type, addr=0x4000)
        pl.load(tile_a, a, [0, 0])
        pl.mul(tile_out, tile_a, 2.0)
        _test_result = pl.store(output, tile_out, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    assert "block.muls" in _program_ir(main)


def test_manual_add_scalar():
    @pl.jit(auto_mutex=False)
    def main(
        a: pl.Tensor[[128, 128], pl.DT_FP32],
        output: pl.Tensor[[128, 128], pl.DT_FP32],
    ):
        tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
        tile_a = pl.make_tile(tile_type, addr=0x3000)
        tile_out = pl.make_tile(tile_type, addr=0x4000)
        pl.load(tile_a, a, [0, 0])
        pl.add(tile_out, tile_a, 1.0)
        _test_result = pl.store(output, tile_out, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    assert "block.adds" in _program_ir(main)


def test_manual_maximum_scalar():
    @pl.jit(auto_mutex=False)
    def main(
        a: pl.Tensor[[128, 128], pl.DT_FP32],
        output: pl.Tensor[[128, 128], pl.DT_FP32],
    ):
        tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
        tile_a = pl.make_tile(tile_type, addr=0x3000)
        tile_out = pl.make_tile(tile_type, addr=0x4000)
        pl.load(tile_a, a, [0, 0])
        pl.maximum(tile_out, tile_a, 0.0)
        _test_result = pl.store(output, tile_out, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    assert "block.maxs" in _program_ir(main)


def test_manual_relu():
    @pl.jit(auto_mutex=False)
    def main(
        a: pl.Tensor[[128, 128], pl.DT_FP32],
        output: pl.Tensor[[128, 128], pl.DT_FP32],
    ):
        tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
        tile_a = pl.make_tile(tile_type, addr=0x5000)
        tile_out = pl.make_tile(tile_type, addr=0x6000)
        pl.load(tile_a, a, [0, 0])
        pl.relu(tile_out, tile_a)
        _test_result = pl.store(output, tile_out, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    assert "block.relu" in _program_ir(main)


def test_manual_cmp():
    @pl.jit(auto_mutex=False)
    def main(
        a: pl.Tensor[[128, 128], pl.DT_FP32],
        b: pl.Tensor[[128, 128], pl.DT_FP32],
        output: pl.Tensor[[128, 128], pl.DT_UINT8],
    ):
        tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
        tile_a = pl.make_tile(tile_type, addr=0x7000)
        tile_b = pl.make_tile(tile_type, addr=0x8000)
        mask_type = pl.TileType(shape=[32, 32], dtype=pl.DT_UINT8)
        tile_out = pl.make_tile(mask_type, addr=0x9000)
        pl.load(tile_a, a, [0, 0])
        pl.load(tile_b, b, [0, 0])
        pl.eq(tile_out, tile_a, tile_b)
        _test_result = pl.store(output, tile_out, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    assert "block.cmp" in _program_ir(main)


def test_manual_row_max():
    @pl.jit(auto_mutex=False)
    def main(
        inp: pl.Tensor[[128, 128], pl.DT_FP32],
        output: pl.Tensor[[128, 1], pl.DT_FP32],
    ):
        in_type = pl.TileType(shape=[32, 128], dtype=pl.DT_FP32)
        out_type = pl.TileType(shape=[32, 1], dtype=pl.DT_FP32)
        tile_in = pl.make_tile(in_type, addr=0xA000)
        tmp = pl.make_tile(out_type, addr=0xE000)
        tile_out = pl.make_tile(out_type, addr=0x12000)
        pl.load(tile_in, inp, [0, 0])
        pl.maximum(tile_out, tile_in, tmp, dim=0)
        _test_result = pl.store(output, tile_out, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    assert "block.row_max" in _program_ir(main)


def test_manual_col_expand_mul():
    @pl.jit(auto_mutex=False)
    def main(
        col: pl.Tensor[[128, 128], pl.DT_FP32],
        tile: pl.Tensor[[128, 128], pl.DT_FP32],
        output: pl.Tensor[[128, 128], pl.DT_FP32],
    ):
        tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
        col_type = pl.TileType(shape=[1, 32], dtype=pl.DT_FP32)
        tile_a = pl.make_tile(tile_type, addr=0x13000)
        tile_col = pl.make_tile(col_type, addr=0x14000)
        tile_out = pl.make_tile(tile_type, addr=0x15000)
        pl.load(tile_a, tile, [0, 0])
        pl.load(tile_col, col, [0, 0])
        pl.expand_mul(tile_out, tile_a, tile_col, dim=1)
        _test_result = pl.store(output, tile_out, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    assert "block.col_expand_mul" in _program_ir(main)


def test_manual_col_expand_div():
    @pl.jit(auto_mutex=False)
    def main(
        col: pl.Tensor[[128, 128], pl.DT_FP32],
        tile: pl.Tensor[[128, 128], pl.DT_FP32],
        output: pl.Tensor[[128, 128], pl.DT_FP32],
    ):
        tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
        col_type = pl.TileType(shape=[1, 32], dtype=pl.DT_FP32)
        tile_a = pl.make_tile(tile_type, addr=0x13000)
        tile_col = pl.make_tile(col_type, addr=0x14000)
        tile_out = pl.make_tile(tile_type, addr=0x15000)
        pl.load(tile_a, tile, [0, 0])
        pl.load(tile_col, col, [0, 0])
        pl.expand_div(tile_out, tile_a, tile_col, dim=1)
        _test_result = pl.store(output, tile_out, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    assert "block.col_expand_div" in _program_ir(main)


def test_manual_and_scalar():
    @pl.jit(auto_mutex=False)
    def main(
        a: pl.Tensor[[128, 128], pl.DT_INT32],
        output: pl.Tensor[[128, 128], pl.DT_INT32],
    ):
        tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_INT32)
        tile_a = pl.make_tile(tile_type, addr=0x16000)
        tile_out = pl.make_tile(tile_type, addr=0x17000)
        pl.load(tile_a, a, [0, 0])
        pl.and_(tile_out, tile_a, 7)
        _test_result = pl.store(output, tile_out, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    assert "block.ands" in _program_ir(main)


def test_manual_xor():
    @pl.jit(auto_mutex=False)
    def main(
        a: pl.Tensor[[128, 128], pl.DT_INT32],
        b: pl.Tensor[[128, 128], pl.DT_INT32],
        output: pl.Tensor[[128, 128], pl.DT_INT32],
    ):
        tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_INT32)
        tile_a = pl.make_tile(tile_type, addr=0x18000)
        tile_b = pl.make_tile(tile_type, addr=0x19000)
        tmp = pl.make_tile(tile_type, addr=0x1A000)
        tile_out = pl.make_tile(tile_type, addr=0x1B000)
        pl.load(tile_a, a, [0, 0])
        pl.load(tile_b, b, [0, 0])
        pl.xor(tile_out, tile_a, tile_b, tmp)
        _test_result = pl.store(output, tile_out, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    assert "block.xor" in _program_ir(main)


def test_load_with_dynamic_valid_shape():
    @pl.jit(auto_mutex=False)
    def main(
        a: pl.Tensor[[128, 128], pl.DT_FP32],
        rows: pl.DT_INT64,
        cols: pl.DT_INT64,
        output: pl.Tensor[[128, 128], pl.DT_FP32],
    ):
        tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP32, valid_shape=[-1, -1])
        tile = pl.make_tile(tile_type, addr=0x20000)
        pl.load(tile, a, [0, 0])
        pl.set_validshape(tile, [rows, cols])
        _test_result = pl.store(output, tile, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    ir_str = _program_ir(main)
    assert "block.load" in ir_str
    assert "block.set_validshape" in ir_str


def test_insert_with_offset_sequence():
    @pl.jit(auto_mutex=False)
    def main(
        output: pl.Tensor[[32, 32], pl.DT_FP32],
    ):
        dst_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
        src_type = pl.TileType(shape=[16, 16], dtype=pl.DT_FP32)
        dst = pl.make_tile(dst_type, addr=0x30200)
        src = pl.make_tile(src_type, addr=0x31200)
        pl.insert(dst, src, [8, 16])
        _test_result = pl.store(output, dst, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    ir_str = _program_ir(main)
    assert "block.insert" in ir_str


def test_acc_to_mat_move_with_scalar_offset_preserves_quantization_and_phase():
    @pl.jit(auto_mutex=False)
    def main(_jit_entry: pl.DT_INT64):
        src_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ)
        dst_type = pl.TileType(shape=[32, 32], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        src = pl.make_tile(src_type, addr=0x0000)
        dst = pl.make_tile(dst_type, addr=0x4000)
        pl.move(dst, src, [16, 16], scale=0.5, phase=pl.STPhase.Final)

    program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Cube)
    call = _find_call(program.get_function(main.__name__), "block.move")

    assert len(call.args) == 4
    assert isinstance(call.args[2], ir.MakeTuple)
    assert isinstance(call.args[3], ir.ConstInt)
    assert "phase" in call.kwargs


def test_acc_to_mat_move_with_scaling_tile_offset_builds_move():
    @pl.jit(auto_mutex=False)
    def main(_jit_entry: pl.DT_INT64):
        src_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ)
        dst_type = pl.TileType(shape=[32, 32], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        scale_type = pl.TileType(shape=[1, 32], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Scaling, layout=pl.ND)
        src = pl.make_tile(src_type, addr=0x0000)
        dst = pl.make_tile(dst_type, addr=0x4000)
        scale = pl.make_tile(scale_type, addr=0x0000)
        pl.move(dst, src, [16, 16], scale=scale)

    program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Cube)
    call = _find_call(program.get_function(main.__name__), "block.move")

    assert len(call.args) == 4
    assert isinstance(call.args[2], ir.MakeTuple)
    assert call.args[3].type.memref.memory_space_ == ir.MemorySpace.Scaling


def test_scaling_tile_move_rejects_unsupported_memory_path():
    with pytest.raises(NotSupported, match="Scaling Tile scale only supports Acc-to-Vec or Acc-to-Mat"):

        @pl.jit(auto_mutex=False)
        def main(_jit_entry: pl.DT_INT64):
            src_type = pl.TileType(shape=[16, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
            dst_type = pl.TileType(shape=[16, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left)
            scale_type = pl.TileType(shape=[1, 16], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Scaling)
            src = pl.make_tile(src_type, addr=0x0000)
            dst = pl.make_tile(dst_type, addr=0x0000)
            scale = pl.make_tile(scale_type, addr=0x0000)
            pl.move(dst, src, scale=scale)

        main.to_kernel_def().parse_target_program(ir.SectionKind.Cube)


def test_move_insert_dispatch_validates_fused_parameters_in_insert():
    with pytest.raises(NotSupported, match="only supported for Acc-to-Mat inserts"):

        @pl.jit(auto_mutex=False)
        def main(_jit_entry: pl.DT_INT64):
            src_type = pl.TileType(shape=[16, 16], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
            dst_type = pl.TileType(shape=[32, 32], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Mat)
            src = pl.make_tile(src_type, addr=0x0000)
            dst = pl.make_tile(dst_type, addr=0x0000)
            pl.move(dst, src, [0, 0], scale=0.5)

        main.to_kernel_def().parse_target_program(ir.SectionKind.Cube)


def test_move_rejects_non_2d_offset():
    with pytest.raises(InvalidArgument, match="move: offset must contain exactly 2 elements"):

        @pl.jit(auto_mutex=False)
        def main(_jit_entry: pl.DT_INT64):
            tile_type = pl.TileType(shape=[16, 16], dtype=pl.DT_FP32)
            src = pl.make_tile(tile_type, addr=0x0000)
            dst = pl.make_tile(tile_type, addr=0x1000)
            pl.move(dst, src, [0])

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_acc_to_mat_insert_preserves_scale_relu_and_phase():
    @pl.jit(auto_mutex=False)
    def main(_jit_entry: pl.DT_INT64):
        src_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ)
        dst_type = pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        src = pl.make_tile(src_type, addr=0x0000)
        dst = pl.make_tile(dst_type, addr=0x4000)
        pl.insert(
            dst,
            src,
            [16, 32],
            scale=0.5,
            relu_pre_mode=pl.ReluPreMode.NormalRelu,
            phase=pl.STPhase.Final,
        )

    program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Cube)
    call = _find_call(program.get_function(main.__name__), "block.insert")

    assert len(call.args) == 5
    assert isinstance(call.args[4], ir.ConstInt)
    assert "relu_pre_mode" in call.kwargs
    assert "phase" in call.kwargs


def test_acc_to_mat_move_insert_dispatch_preserves_phase():
    @pl.jit(auto_mutex=False)
    def main(_jit_entry: pl.DT_INT64):
        src_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ)
        dst_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        src = pl.make_tile(src_type, addr=0x0000)
        dst = pl.make_tile(dst_type, addr=0x4000)
        pl.move(dst, src, [16, 16], phase=pl.STPhase.Final)

    program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Cube)
    call = _find_call(program.get_function(main.__name__), "block.insert")

    assert "phase" in call.kwargs


def test_acc_to_mat_whole_move_preserves_phase():
    @pl.jit(auto_mutex=False)
    def main(_jit_entry: pl.DT_INT64):
        src_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ)
        dst_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        src = pl.make_tile(src_type, addr=0x0000)
        dst = pl.make_tile(dst_type, addr=0x4000)
        pl.move(dst, src, phase=pl.STPhase.Final)

    program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Cube)
    call = _find_call(program.get_function(main.__name__), "block.move")

    assert "phase" in call.kwargs


def test_acc_to_mat_insert_with_scaling_tile_preserves_operand():
    @pl.jit(auto_mutex=False)
    def main(_jit_entry: pl.DT_INT64):
        src_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ)
        dst_type = pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        scale_type = pl.TileType(shape=[1, 32], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Scaling, layout=pl.ND)
        src = pl.make_tile(src_type, addr=0x0000)
        dst = pl.make_tile(dst_type, addr=0x4000)
        scale = pl.make_tile(scale_type, addr=0x0000)
        pl.insert(dst, src, [16, 32], scale=scale)

    program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Cube)
    call = _find_call(program.get_function(main.__name__), "block.insert")

    assert len(call.args) == 5
    assert call.args[4].type.memref.memory_space_ == ir.MemorySpace.Scaling


@pytest.mark.parametrize("layout", [pl.ND, pl.DN])
def test_a5_acc_to_mat_move_accepts_supported_destination_layouts(monkeypatch, layout):
    monkeypatch.setenv("PYPTOPRO_JIT_ARCH", "a5")

    @pl.jit(auto_mutex=False)
    def main(_jit_entry: pl.DT_INT64):
        src_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ)
        dst_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=layout)
        src = pl.make_tile(src_type, addr=0x0000)
        dst = pl.make_tile(dst_type, addr=0x4000)
        pl.move(dst, src)

    program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Cube)
    assert "block.move" in _program_ir(program.get_function(main.__name__))


@pytest.mark.parametrize(
    ("src_dtype", "dst_dtype", "with_scale"),
    [
        pytest.param(pl.DT_FP32, pl.DT_FP32, False, id="fp32_to_fp32"),
        pytest.param(pl.DT_FP32, pl.DT_FP16, False, id="fp32_to_fp16"),
        pytest.param(pl.DT_FP32, pl.DT_BF16, False, id="fp32_to_bf16"),
        pytest.param(pl.DT_INT32, pl.DT_INT32, False, id="int32_to_int32"),
        pytest.param(pl.DT_FP32, pl.DT_INT8, True, id="fp32_to_int8_quant"),
        pytest.param(pl.DT_FP32, pl.DT_UINT8, True, id="fp32_to_uint8_quant"),
        pytest.param(pl.DT_FP32, pl.DT_FP16, True, id="fp32_to_fp16_quant"),
        pytest.param(pl.DT_FP32, pl.DT_BF16, True, id="fp32_to_bf16_quant"),
        pytest.param(pl.DT_FP32, pl.DT_HF8, True, id="fp32_to_hf8_quant"),
        pytest.param(pl.DT_FP32, pl.DT_FP8E4M3FN, True, id="fp32_to_fp8e4m3fn_quant"),
        pytest.param(pl.DT_INT32, pl.DT_INT8, True, id="int32_to_int8_quant"),
        pytest.param(pl.DT_INT32, pl.DT_UINT8, True, id="int32_to_uint8_quant"),
        pytest.param(pl.DT_INT32, pl.DT_FP16, True, id="int32_to_fp16_quant"),
        pytest.param(pl.DT_INT32, pl.DT_BF16, True, id="int32_to_bf16_quant"),
    ],
)
def test_acc_to_mat_move_accepts_supported_dtype_matrix(src_dtype, dst_dtype, with_scale):
    @pl.jit(auto_mutex=False)
    def main(_jit_entry: pl.DT_INT64):
        src_type = pl.TileType(shape=[32, 32], dtype=src_dtype, target_memory=pl.MemorySpace.Acc, layout=pl.NZ)
        dst_type = pl.TileType(shape=[32, 32], dtype=dst_dtype, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        src = pl.make_tile(src_type, addr=0x0000)
        dst = pl.make_tile(dst_type, addr=0x4000)
        if with_scale:
            pl.move(dst, src, scale=0.5)
        else:
            pl.move(dst, src)

    program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Cube)
    assert "block.move" in _program_ir(program.get_function(main.__name__))


def test_acc_to_mat_move_rejects_acc_to_vec_mode():
    with pytest.raises(NotSupported, match="only supported for Acc-to-Vec"):

        @pl.jit(auto_mutex=False)
        def main(_jit_entry: pl.DT_INT64):
            src_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ)
            dst_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
            src = pl.make_tile(src_type, addr=0x0000)
            dst = pl.make_tile(dst_type, addr=0x4000)
            pl.move(dst, src, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)

        main.to_kernel_def().parse_target_program(ir.SectionKind.Cube)


def test_vec_to_vec_move_rejects_acc_to_vec_mode():
    with pytest.raises(NotSupported, match="acc_to_vec_mode is only supported for Acc-to-Vec"):

        @pl.jit(auto_mutex=False)
        def main(_jit_entry: pl.DT_INT64):
            tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
            src = pl.make_tile(tile_type, addr=0x0000)
            dst = pl.make_tile(tile_type, addr=0x2000)
            pl.move(dst, src, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_vec_to_vec_move_rejects_relu_pre_mode():
    with pytest.raises(NotSupported, match="relu_pre_mode is only supported for Acc-to-Vec or Acc-to-Mat"):

        @pl.jit(auto_mutex=False)
        def main(_jit_entry: pl.DT_INT64):
            tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
            src = pl.make_tile(tile_type, addr=0x0000)
            dst = pl.make_tile(tile_type, addr=0x2000)
            pl.move(dst, src, relu_pre_mode=pl.ReluPreMode.NormalRelu)

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_acc_to_mat_move_accepts_relu_pre_mode():
    @pl.jit(auto_mutex=False)
    def main(_jit_entry: pl.DT_INT64):
        src_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ)
        dst_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        src = pl.make_tile(src_type, addr=0x0000)
        dst = pl.make_tile(dst_type, addr=0x4000)
        pl.move(dst, src, relu_pre_mode=pl.ReluPreMode.NormalRelu)

    program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Cube)
    call = _find_call(program.get_function(main.__name__), "block.move")

    assert "relu_pre_mode" in call.kwargs


def test_acc_to_vec_move_accepts_acc_to_vec_mode_and_relu_pre_mode():
    @pl.jit(auto_mutex=False)
    def main(_jit_entry: pl.DT_INT64):
        src_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ)
        dst_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        src = pl.make_tile(src_type, addr=0x0000)
        dst = pl.make_tile(dst_type, addr=0x4000)
        pl.move(
            dst,
            src,
            acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0,
            relu_pre_mode=pl.ReluPreMode.NormalRelu,
        )

    program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Cube)
    call = _find_call(program.get_function(main.__name__), "block.move")

    assert "acc_to_vec_mode" in call.kwargs
    assert "relu_pre_mode" in call.kwargs


def test_acc_store_accepts_relu_pre_mode():
    @pl.jit(auto_mutex=False)
    def main(output: pl.Tensor[[128, 128], pl.DT_FP32]):
        tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ)
        tile = pl.make_tile(tile_type, addr=0x0000)
        pl.store(output, tile, [0, 0], relu_pre_mode=pl.ReluPreMode.NormalRelu)

    program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Cube)
    call = _find_call(program.get_function(main.__name__), "block.store")

    assert "relu_pre_mode" in call.kwargs


def test_vec_store_rejects_relu_pre_mode():
    with pytest.raises(NotSupported, match="relu_pre_mode is only supported for Acc"):

        @pl.jit(auto_mutex=False)
        def main(output: pl.Tensor[[128, 128], pl.DT_FP32]):
            tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
            tile = pl.make_tile(tile_type, addr=0x0000)
            pl.store(output, tile, [0, 0], relu_pre_mode=pl.ReluPreMode.NormalRelu)

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_vec_store_tile_rejects_relu_pre_mode():
    with pytest.raises(NotSupported, match="relu_pre_mode is only supported for Acc"):

        @pl.jit(auto_mutex=False)
        def main(output: pl.Tensor[[128, 128], pl.DT_FP32]):
            tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
            tile = pl.make_tile(tile_type, addr=0x0000)
            pl.store_tile(output, tile, [0, 0], relu_pre_mode=pl.ReluPreMode.NormalRelu)

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_acc_to_mat_move_rejects_unsupported_layout():
    with pytest.raises(InvalidType, match="unsupported layout/dtype combination"):

        @pl.jit(auto_mutex=False)
        def main(_jit_entry: pl.DT_INT64):
            src_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ)
            dst_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.ZN)
            src = pl.make_tile(src_type, addr=0x0000)
            dst = pl.make_tile(dst_type, addr=0x4000)
            pl.move(dst, src)

        main.to_kernel_def().parse_target_program(ir.SectionKind.Cube)


def test_acc_to_mat_move_rejects_unsupported_dtype_conversion():
    with pytest.raises(InvalidType, match="unsupported layout/dtype combination"):

        @pl.jit(auto_mutex=False)
        def main(_jit_entry: pl.DT_INT64):
            src_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ)
            dst_type = pl.TileType(shape=[32, 32], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
            src = pl.make_tile(src_type, addr=0x0000)
            dst = pl.make_tile(dst_type, addr=0x4000)
            pl.move(dst, src)

        main.to_kernel_def().parse_target_program(ir.SectionKind.Cube)


@pytest.mark.parametrize("layout", [pl.ND, pl.DN])
def test_acc_to_mat_extract_rejects_non_nz_destination_layout(layout):
    with pytest.raises(InvalidType, match="extract: unsupported layout/dtype combination"):

        @pl.jit(auto_mutex=False)
        def main(_jit_entry: pl.DT_INT64):
            src_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ)
            dst_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=layout)
            src = pl.make_tile(src_type, addr=0x0000)
            dst = pl.make_tile(dst_type, addr=0x4000)
            pl.move(dst, src, [0, 0])

        main.to_kernel_def().parse_target_program(ir.SectionKind.Cube)


def test_acc_to_mat_insert_rejects_unsupported_dtype_conversion():
    with pytest.raises(InvalidType, match="insert: unsupported quantized layout/dtype combination"):

        @pl.jit(auto_mutex=False)
        def main(_jit_entry: pl.DT_INT64):
            src_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ)
            dst_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP8E5M2, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
            src = pl.make_tile(src_type, addr=0x0000)
            dst = pl.make_tile(dst_type, addr=0x4000)
            pl.insert(dst, src, [16, 32], scale=0.5)

        main.to_kernel_def().parse_target_program(ir.SectionKind.Cube)


def test_manual_transpose():
    @pl.jit(auto_mutex=False)
    def main(
        a: pl.Tensor[[128, 128], pl.DT_FP16],
        output: pl.Tensor[[128, 128], pl.DT_FP16],
    ):
        src_type = pl.TileType(shape=[16, 32], dtype=pl.DT_FP16)
        dst_type = pl.TileType(shape=[32, 16], dtype=pl.DT_FP16)
        src = pl.make_tile(src_type, addr=0x30000)
        dst = pl.make_tile(dst_type, addr=0x30100)
        pl.load(src, a, [0, 0])
        pl.transpose(dst, src)
        _test_result = pl.store(output, dst, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    assert "block.transpose" in _program_ir(main)


def test_get_block_idx_without_prefix():
    @pl.jit(auto_mutex=False)
    def main(output: pl.Tensor[[1], pl.DT_INT64]):
        _ = pl.get_block_idx()
        _test_result = output

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    assert "get_block_idx" in _program_ir(main)


def test_set_validshape_tile_group_4buf():
    @pl.jit(auto_mutex=False)
    def main(
        a: pl.Tensor[[128, 128], pl.DT_FP16],
        rows: pl.DT_INT64,
        cols: pl.DT_INT64,
        output: pl.Tensor[[128, 128], pl.DT_FP16],
    ):
        tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, valid_shape=[-1, -1])
        a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1, 2, 3])
        pl.set_validshape(a_db, [rows, cols])
        tile_a = a_db.next()
        pl.load(tile_a, a, [0, 0])
        _test_result = pl.store(output, tile_a, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    ir_str = _program_ir(main)
    assert "block.set_validshape" in ir_str
    vs_pos = ir_str.index("block.set_validshape")
    load_pos = ir_str.index("block.load")
    assert vs_pos < load_pos


def test_set_validshape_tile_group_single_tile():
    @pl.jit(auto_mutex=False)
    def main(
        a: pl.Tensor[[128, 128], pl.DT_FP16],
        cols: pl.DT_INT64,
        output: pl.Tensor[[128, 128], pl.DT_FP16],
    ):
        tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, valid_shape=[-1, -1])
        a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])
        pl.set_validshape(a_db, [128, cols])
        tile_a = a_db.next()
        pl.load(tile_a, a, [0, 0])
        _test_result = pl.store(output, tile_a, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    ir_str = _program_ir(main)
    assert "block.set_validshape" in ir_str
    vs_pos = ir_str.index("block.set_validshape")
    load_pos = ir_str.index("block.load")
    assert vs_pos < load_pos


def test_move_to_insert_subblock():
    @pl.jit(auto_mutex=False)
    def main(
        output: pl.Tensor[[32, 32], pl.DT_FP32],
    ):
        dst_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
        src_type = pl.TileType(shape=[16, 16], dtype=pl.DT_FP32)
        dst = pl.make_tile(dst_type, addr=0x30200)
        src = pl.make_tile(src_type, addr=0x31200)
        pl.move(dst, src, [8, 8])
        _test_result = pl.store(output, dst, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    ir_str = _program_ir(main)
    assert "block.insert" in ir_str
    assert "block.move" not in ir_str


def test_move_equal_shape_stays_move():
    @pl.jit(auto_mutex=False)
    def main(
        output: pl.Tensor[[32, 32], pl.DT_FP32],
    ):
        dst_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
        src_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
        dst = pl.make_tile(dst_type, addr=0x30200)
        src = pl.make_tile(src_type, addr=0x31200)
        pl.move(dst, src, [0, 0])
        _test_result = pl.store(output, dst, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    ir_str = _program_ir(main)
    assert "block.move" in ir_str


def test_move_src_larger_stays_move():
    @pl.jit(auto_mutex=False)
    def main(
        output: pl.Tensor[[32, 32], pl.DT_FP32],
    ):
        dst_type = pl.TileType(shape=[16, 16], dtype=pl.DT_FP32)
        src_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
        dst = pl.make_tile(dst_type, addr=0x30200)
        src = pl.make_tile(src_type, addr=0x31200)
        pl.move(dst, src, [0, 0])
        _test_result = pl.store(output, dst, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    ir_str = _program_ir(main)
    assert "block.move" in ir_str


def test_make_tile_missing_addr_rejected():
    with pytest.raises(InvalidVal, match="missing required keyword 'addr'"):

        @pl.jit(auto_mutex=False)
        def main(
            a: pl.Tensor[[128, 128], pl.DT_FP16],
            output: pl.Tensor[[128, 128], pl.DT_FP16],
        ):
            tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
            tile_a = pl.make_tile(tile_type)
            pl.load(tile_a, a, [0, 0])
            _test_result = pl.store(output, tile_a, [0, 0])

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_make_tile_size_is_derived_from_tile_type():
    @pl.jit(auto_mutex=False)
    def main(
        a: pl.Tensor[[128, 128], pl.DT_FP16],
        output: pl.Tensor[[128, 128], pl.DT_FP16],
    ):
        tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
        tile_a = pl.make_tile(tile_type, addr=0x0000)
        pl.load(tile_a, a, [0, 0])
        _test_result = pl.store(output, tile_a, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    # 128 * 128 elements * 2 bytes (FP16)
    assert 'memref_size=32768' in _program_ir(main)


def test_make_tile_runtime_addr_rejected_with_compile_time_hint():
    with pytest.raises(InvalidType) as excinfo:

        @pl.jit(auto_mutex=False)
        def main(
            a: pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP16],
            output: pl.Tensor[[128, 128], pl.DT_FP16],
        ):
            tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
            tile_a = pl.make_tile(tile_type, addr=a.shape[0] * 32)
            pl.load(tile_a, a, [0, 0])
            _test_result = pl.store(output, tile_a, [0, 0])

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)

    assert "compile-time integer" in str(excinfo.value)
    assert "runtime value" in str(excinfo.value)


def test_make_tile_addr_after_tile_type():
    """The documented prototype: pl.make_tile(tile_type, *, addr)."""

    @pl.jit(auto_mutex=False)
    def main(
        a: pl.Tensor[[128, 128], pl.DT_FP16],
        output: pl.Tensor[[128, 128], pl.DT_FP16],
    ):
        tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
        tile_a = pl.make_tile(tile_type, addr=0x1000)
        pl.load(tile_a, a, [0, 0])
        _test_result = pl.store(output, tile_a, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    ir_str = _program_ir(main)
    assert 'memref_addr=4096' in ir_str
    assert 'memref_size=32768' in ir_str


def test_make_tile_builder_form_rejected():
    """make_tile(shape, dtype, target_memory, ...) is the IR builder, not the DSL op."""

    with pytest.raises(InvalidType, match="takes a pl.TileType as its first argument"):

        @pl.jit(auto_mutex=False)
        def main(
            a: pl.Tensor[[128, 128], pl.DT_FP16],
            output: pl.Tensor[[128, 128], pl.DT_FP16],
        ):
            tile_a = pl.make_tile([128, 128], pl.DT_FP16, pl.MemorySpace.Vec, 0x2000, 32768)
            pl.load(tile_a, a, [0, 0])
            _test_result = pl.store(output, tile_a, [0, 0])

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_make_tile_misaligned_addr_rejected():
    """addr reaches the alignment check as a plain int, whatever expression wrote it."""

    with pytest.raises(InvalidTile, match="32-byte aligned"):

        @pl.jit(auto_mutex=False)
        def main(
            a: pl.Tensor[[128, 128], pl.DT_FP16],
            output: pl.Tensor[[128, 128], pl.DT_FP16],
        ):
            tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
            tile_a = pl.make_tile(tile_type, addr=0x1)
            pl.load(tile_a, a, [0, 0])
            _test_result = pl.store(output, tile_a, [0, 0])

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_make_tile_runtime_addr_rejected_quoting_source():
    with pytest.raises(InvalidType) as excinfo:

        @pl.jit(auto_mutex=False)
        def main(
            a: pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP16],
            output: pl.Tensor[[128, 128], pl.DT_FP16],
        ):
            tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
            tile_a = pl.make_tile(tile_type, addr=a.shape[0] * 32)
            pl.load(tile_a, a, [0, 0])
            _test_result = pl.store(output, tile_a, [0, 0])

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)

    assert "compile-time integer" in str(excinfo.value)
    # the offending expression, not a repr of the parsed IR object
    assert "a.shape[0] * 32" in str(excinfo.value)


def test_make_tile_positional_addr_rejected():
    """The tile type is the only positional argument; addr is keyword-only."""

    with pytest.raises(InvalidArgument, match="takes 1 positional argument .* but 2 were given"):

        @pl.jit(auto_mutex=False)
        def main(
            a: pl.Tensor[[128, 128], pl.DT_FP16],
            output: pl.Tensor[[128, 128], pl.DT_FP16],
        ):
            tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
            tile_a = pl.make_tile(tile_type, 0x1000)
            pl.load(tile_a, a, [0, 0])
            _test_result = pl.store(output, tile_a, [0, 0])

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_make_tile_extra_positional_args_rejected_with_addr_keyword_hint():
    """The rejection carries the fix, so the call site does not have to guess."""

    with pytest.raises(InvalidArgument) as excinfo:

        @pl.jit(auto_mutex=False)
        def main(
            a: pl.Tensor[[128, 128], pl.DT_FP16],
            output: pl.Tensor[[128, 128], pl.DT_FP16],
        ):
            tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
            tile_a = pl.make_tile(tile_type, 0x0000, 32768)
            pl.load(tile_a, a, [0, 0])
            _test_result = pl.store(output, tile_a, [0, 0])

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)

    assert "takes 1 positional argument (the tile type) but 3 were given" in str(excinfo.value)
    # the hint spells out the accepted spelling
    assert "addr=" in str(excinfo.value)


VEC_BASE = 0x1000


def test_make_tile_addr_from_constant_expression():
    """addr goes through the full compile-time path, not just literals."""

    @pl.jit(auto_mutex=False)
    def main(
        a: pl.Tensor[[128, 128], pl.DT_FP16],
        output: pl.Tensor[[128, 128], pl.DT_FP16],
    ):
        tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
        tile_a = pl.make_tile(tile_type, addr=VEC_BASE + 0x20)
        pl.load(tile_a, a, [0, 0])
        _test_result = pl.store(output, tile_a, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    assert 'memref_addr=4128' in _program_ir(main)


@pytest.mark.parametrize("memory", [pl.MemorySpace.Left, pl.MemorySpace.Right, pl.MemorySpace.Acc])
@pytest.mark.parametrize("layout", [pl.ZZ, pl.NN])
def test_a5_rejects_zz_nn_for_cube_buffers(monkeypatch, memory, layout):
    monkeypatch.setenv("PYPTOPRO_JIT_ARCH", "a5")

    with pytest.raises(InvalidVal, match="do not support"):

        @pl.jit(auto_mutex=False)
        def create_tile(_jit_entry: pl.DT_INT64):
            tt = pl.TileType(
                shape=[128, 128],
                dtype=pl.DT_FP16,
                target_memory=memory,
                layout=layout,
            )
            tile = pl.make_tile(tt, addr=0x00000)  # noqa: F841

        create_tile.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_a5_allows_regular_cube_buffer_layouts(monkeypatch):
    monkeypatch.setenv("PYPTOPRO_JIT_ARCH", "a5")

    @pl.jit(auto_mutex=False)
    def create_tile(_jit_entry: pl.DT_INT64):
        tt = pl.TileType(
            shape=[128, 128],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Left,
            layout=pl.NZ,
        )
        tile = pl.make_tile(tt, addr=0x00000)  # noqa: F841

    create_tile_program, _ = create_tile.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    create_tile = create_tile_program.get_function(create_tile.__name__)

    stmt = create_tile.body.stmts[0] if hasattr(create_tile.body, "stmts") else create_tile.body
    assert isinstance(stmt.var.type.hardware_info, _ir.HardwareInfo)


def test_high_dimensional_nz_load_store_use_last_two_axes_by_default():
    @pl.jit(auto_mutex=False)
    def main(
        inp: pl.Tensor[[2, 3, 64, 64], pl.DT_FP16, pl.NZ],
        out: pl.Tensor[[2, 3, 64, 64], pl.DT_FP16, pl.NZ],
    ):
        tile_type = pl.TileType(
            shape=[16, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ
        )
        tile = pl.make_tile(tile_type, addr=0x1000)
        pl.load(tile, inp, [1, 2, 16, 16])
        pl.store(out, tile, [1, 2, 16, 16])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)
    ir_text = _program_ir(main)
    assert "block.load" in ir_text
    assert "block.store" in ir_text


def test_store_and_store_tile_with_scaling_tile_use_store_op(monkeypatch):
    monkeypatch.setenv("PYPTOPRO_JIT_ARCH", "a5")

    @pl.jit(auto_mutex=False)
    def main(out: pl.Tensor[[64, 64], pl.DT_INT8]):
        acc_type = pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
        )
        scale_type = pl.TileType(shape=[1, 64], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Scaling)
        acc = pl.make_tile(acc_type, addr=0x0000)
        scale = pl.make_tile(scale_type, addr=0x0000)
        pl.store(out, acc, [0, 0], scale=scale)
        pl.store_tile(out, acc, [0, 0], scale=scale)

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Cube)
    main = main_program.get_function(main.__name__)
    ir_text = _program_ir(main)
    assert ir_text.count("block.store") == 2


def test_high_dimensional_nz_load_rejects_non_final_transfer_axes():
    with pytest.raises(InvalidFormat, match="NZ transfer only supports the last two tensor axes"):

        @pl.jit(auto_mutex=False)
        def main(
            inp: pl.Tensor[[2, 64, 64, 64], pl.DT_FP16, pl.NZ],
        ):
            tile_type = pl.TileType(
                shape=[16, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ
            )
            tile = pl.make_tile(tile_type, addr=0x1000)
            pl.load(tile, inp, [0, 0, 0, 0], order=[1, 3])

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_high_dimensional_nz_store_rejects_non_final_transfer_axes():
    with pytest.raises(InvalidFormat, match="NZ transfer only supports the last two tensor axes"):

        @pl.jit(auto_mutex=False)
        def main(
            out: pl.Tensor[[2, 64, 64, 64], pl.DT_FP16, pl.NZ],
        ):
            tile_type = pl.TileType(
                shape=[16, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ
            )
            tile = pl.make_tile(tile_type, addr=0x1000)
            pl.store(out, tile, [0, 0, 0, 0], order=[1, 3])

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


# ===========================================================================
# order kwarg length / rank-1 validation (load / store paths)
# ===========================================================================


def test_load_rejects_length1_order():
    with pytest.raises(InvalidShape, match="load: order must be a 2-element list, got 1"):

        @pl.jit(auto_mutex=False)
        def main(a: pl.Tensor[[64, 64], pl.DT_FP32]):
            tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
            tile = pl.make_tile(tile_type, addr=0x0000)
            pl.load(tile, a, [0, 0], order=[0])

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_load_rejects_empty_order():
    with pytest.raises(InvalidShape, match="load: order must be a 2-element list, got 0"):

        @pl.jit(auto_mutex=False)
        def main(a: pl.Tensor[[64, 64], pl.DT_FP32]):
            tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
            tile = pl.make_tile(tile_type, addr=0x0000)
            pl.load(tile, a, [0, 0], order=[])

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_load_rejects_length3_order_on_rank3_tensor():
    """Before the length check this crashed with a bare IndexError instead of a diagnostic."""
    with pytest.raises(InvalidShape, match="load: order must be a 2-element list, got 3"):

        @pl.jit(auto_mutex=False)
        def main(a: pl.Tensor[[4, 64, 64], pl.DT_FP32]):
            tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
            tile = pl.make_tile(tile_type, addr=0x0000)
            pl.load(tile, a, [0, 0, 0], order=[0, 1, 2])

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_store_rejects_length1_order():
    with pytest.raises(InvalidShape, match="store: order must be a 2-element list, got 1"):

        @pl.jit(auto_mutex=False)
        def main(out: pl.Tensor[[64, 64], pl.DT_FP32]):
            tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
            tile = pl.make_tile(tile_type, addr=0x0000)
            pl.store(out, tile, [0, 0], order=[0])

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_load_rejects_order_on_rank1_tensor():
    """Rank-1 tensors express the axis mapping through the Tile shape; order has no meaning."""
    with pytest.raises(InvalidArgument, match="load: order is not supported for rank-1 Tensors"):

        @pl.jit(auto_mutex=False)
        def main(a: pl.Tensor[[1024], pl.DT_FP32]):
            tile_type = pl.TileType(shape=[1, 1024], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
            tile = pl.make_tile(tile_type, addr=0x0000)
            pl.load(tile, a, [0], order=[0])

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_rank1_load_without_order_accepted():
    """The no-order rank-1 path must keep working: the axis mapping comes from the tile shape."""

    @pl.jit(auto_mutex=False)
    def main(a: pl.Tensor[[1024], pl.DT_FP32]):
        tile_type = pl.TileType(shape=[1, 1024], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0x0000)
        pl.load(tile, a, [0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    ir_str = _program_ir(main_program.get_function(main.__name__))
    assert "block.load" in ir_str
    assert "tile_dims" not in ir_str


def test_load_accepts_explicit_2_element_order():
    """A spelled-out legal order is preserved into the IR kwargs verbatim."""

    @pl.jit(auto_mutex=False)
    def main(a: pl.Tensor[[64, 64], pl.DT_FP32]):
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0x0000)
        pl.load(tile, a, [0, 0], order=[0, 1])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    ir_str = _program_ir(main_program.get_function(main.__name__))
    assert "block.load" in ir_str
    assert "tile_dims=[0, 1]" in ir_str


def test_load_order_value_range_still_enforced():
    with pytest.raises(InvalidShape, match="order axis 2 is out of range for Tensor rank 2"):

        @pl.jit(auto_mutex=False)
        def main(a: pl.Tensor[[64, 64], pl.DT_FP32]):
            tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
            tile = pl.make_tile(tile_type, addr=0x0000)
            pl.load(tile, a, [0, 0], order=[0, 2])

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_load_order_uniqueness_still_enforced():
    with pytest.raises(InvalidShape, match="order axes must be unique"):

        @pl.jit(auto_mutex=False)
        def main(a: pl.Tensor[[64, 64], pl.DT_FP32]):
            tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
            tile = pl.make_tile(tile_type, addr=0x0000)
            pl.load(tile, a, [0, 0], order=[0, 0])

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_load_tile_rejects_length1_order():
    with pytest.raises(InvalidShape, match="load_tile: order must be a 2-element list, got 1"):

        @pl.jit(auto_mutex=False)
        def main(a: pl.Tensor[[64, 64], pl.DT_FP32]):
            tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
            tile = pl.make_tile(tile_type, addr=0x0000)
            pl.load_tile(tile, a, [0, 0], order=[0])

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_load_tile_accepts_explicit_2_element_order():
    @pl.jit(auto_mutex=False)
    def main(a: pl.Tensor[[64, 64], pl.DT_FP32]):
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0x0000)
        pl.load_tile(tile, a, [0, 0], order=[0, 1])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    ir_str = _program_ir(main_program.get_function(main.__name__))
    assert "block.load" in ir_str
    assert "tile_dims=[0, 1]" in ir_str


def test_store_accepts_explicit_2_element_order():
    @pl.jit(auto_mutex=False)
    def main(out: pl.Tensor[[64, 64], pl.DT_FP32]):
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0x0000)
        pl.store(out, tile, [0, 0], order=[0, 1])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    ir_str = _program_ir(main_program.get_function(main.__name__))
    assert "block.store" in ir_str


def test_store_tile_rejects_length1_order():
    with pytest.raises(InvalidShape, match="store_tile: order must be a 2-element list, got 1"):

        @pl.jit(auto_mutex=False)
        def main(out: pl.Tensor[[64, 64], pl.DT_FP32]):
            tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
            tile = pl.make_tile(tile_type, addr=0x0000)
            pl.store_tile(out, tile, [0, 0], order=[0])

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_store_tile_accepts_explicit_2_element_order():
    @pl.jit(auto_mutex=False)
    def main(out: pl.Tensor[[64, 64], pl.DT_FP32]):
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0x0000)
        pl.store_tile(out, tile, [0, 0], order=[0, 1])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    ir_str = _program_ir(main_program.get_function(main.__name__))
    assert "block.store" in ir_str
