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
"""Host-side validation tests for MX block matmul construction."""

from itertools import count

from pypto_pro import DataType, ir
from pypto_pro._errors import InvalidType
from pypto_pro.ir.op.block_ops import (
    TileType,
    _ir_load,
    _ir_matmul_mx,
    _ir_matmul_mx_acc,
    _validate_tile_addr_alignment,
)
import pytest

_MEMREF_IDS = count()


_LAYOUT_HARDWARE = {
    ir.TensorLayout.NZ: (ir.TileLayout.col_major, ir.TileLayout.row_major),
    ir.TensorLayout.ZN: (ir.TileLayout.row_major, ir.TileLayout.col_major),
    ir.TensorLayout.NN: (ir.TileLayout.col_major, ir.TileLayout.col_major),
    ir.TensorLayout.ZZ: (ir.TileLayout.row_major, ir.TileLayout.row_major),
}


def _tile(name, shape, dtype, memory_space, addr=0, *, layout=None, fractal=None):
    span = ir.Span.unknown()
    dims = [ir.ConstInt(dim, DataType.INDEX, span) for dim in shape]
    memref = ir.MemRef(
        memory_space,
        ir.ConstInt(addr, DataType.INT64, span),
        65536,
        next(_MEMREF_IDS),
    )
    if layout is None and memory_space == ir.MemorySpace.ScaleLeft:
        layout, fractal = ir.TensorLayout.ZZ, 32
    elif layout is None and memory_space == ir.MemorySpace.ScaleRight:
        layout, fractal = ir.TensorLayout.NN, 32
    hardware_info = None
    if layout is not None:
        blayout, slayout = _LAYOUT_HARDWARE[layout]
        hardware_info = ir.HardwareInfo(blayout, slayout, 512 if fractal is None else fractal)
    return ir.Var(name, ir.TileType(dims, dtype, memref, hardware_info=hardware_info), span)


def _tensor(name, shape, dtype):
    span = ir.Span.unknown()
    dims = [ir.ConstInt(dim, DataType.INDEX, span) for dim in shape]
    return ir.Var(name, ir.TensorType(dims, dtype), span)


def _mx_tiles(*, k=64, lhs_dtype=DataType.FP8E4M3FN,
              rhs_dtype=DataType.FP8E5M2, dst_dtype=DataType.FP32,
              scale_a_dtype=DataType.FP8E8M0,
              scale_a_space=ir.MemorySpace.ScaleLeft):
    return {
        "dst": _tile("dst", [64, 64], dst_dtype, ir.MemorySpace.Acc),
        "acc": _tile("acc", [64, 64], DataType.FP32, ir.MemorySpace.Acc),
        "lhs": _tile("lhs", [64, k], lhs_dtype, ir.MemorySpace.Left),
        "rhs": _tile("rhs", [k, 64], rhs_dtype, ir.MemorySpace.Right),
        "scale_a": _tile("scale_a", [64, k // 32], scale_a_dtype, scale_a_space),
        "scale_b": _tile("scale_b", [k // 32, 64], DataType.FP8E8M0, ir.MemorySpace.ScaleRight),
    }


def test_load_preserves_explicit_default_order_in_ir():
    src = _tensor("scale", [64, 1, 2], DataType.FP8E8M0)
    dst = _tile("scale_l1", [64, 2], DataType.FP8E8M0, ir.MemorySpace.Mat)

    call = _ir_load(dst, src, [0, 0, 0], order=[1, 2])

    assert 'tile_dims=[1, 2]' in str(call)


def test_matmul_mx_builds_registered_call():
    t = _mx_tiles()
    call = _ir_matmul_mx(t["dst"], t["lhs"], t["rhs"], t["scale_a"], t["scale_b"])

    assert call.name == "block.matmul_mx"
    assert call.type.dtype == DataType.FP32
    assert [arg.name for arg in call.args] == ["dst", "lhs", "rhs", "scale_a", "scale_b"]


def test_matmul_mx_acc_builds_registered_call():
    t = _mx_tiles(lhs_dtype=DataType.FP4E2M1, rhs_dtype=DataType.FP4E1M2)
    call = _ir_matmul_mx_acc(
        t["dst"], t["acc"], t["lhs"], t["rhs"], t["scale_a"], t["scale_b"]
    )

    assert call.name == "block.matmul_mx_acc"
    assert call.type.dtype == DataType.FP32
    assert [arg.name for arg in call.args] == ["dst", "acc", "lhs", "rhs", "scale_a", "scale_b"]


def test_matmul_mx_rejects_unaligned_k():
    t = _mx_tiles(k=32)
    with pytest.raises(ValueError, match="K dimension must be a multiple of 64"):
        _ir_matmul_mx(t["dst"], t["lhs"], t["rhs"], t["scale_a"], t["scale_b"])


def test_matmul_mx_checks_dtype_before_shape():
    t = _mx_tiles(k=32, lhs_dtype=DataType.FP16)
    with pytest.raises(InvalidType, match="must be FP8/FP4 combo and dst FP32"):
        _ir_matmul_mx(t["dst"], t["lhs"], t["rhs"], t["scale_a"], t["scale_b"])


def test_matmul_mx_allows_physical_k_selected_by_valid_shape():
    t = _mx_tiles(k=64)
    t["rhs"] = _tile("rhs", [128, 64], DataType.FP8E5M2, ir.MemorySpace.Right)
    t["scale_b"] = _tile("scale_b", [4, 64], DataType.FP8E8M0, ir.MemorySpace.ScaleRight)

    call = _ir_matmul_mx(t["dst"], t["lhs"], t["rhs"], t["scale_a"], t["scale_b"])

    assert call.name == "block.matmul_mx"


def test_matmul_mx_allows_output_shape_selected_by_valid_shape():
    t = _mx_tiles()
    # set_validshape updates descriptor state rather than TileType.shape, so the
    # builder must not infer the logical output window from the input physical shapes.
    t["dst"] = _tile("dst", [32, 16], DataType.FP32, ir.MemorySpace.Acc)

    call = _ir_matmul_mx(t["dst"], t["lhs"], t["rhs"], t["scale_a"], t["scale_b"])

    assert call.name == "block.matmul_mx"


def test_matmul_mx_acc_rejects_mismatched_acc_shape():
    t = _mx_tiles()
    t["acc"] = _tile("acc", [64, 32], DataType.FP32, ir.MemorySpace.Acc)

    with pytest.raises(InvalidType, match=r"acc_tile type must match dst_tile type"):
        _ir_matmul_mx_acc(t["dst"], t["acc"], t["lhs"], t["rhs"], t["scale_a"], t["scale_b"])


def test_matmul_mx_acc_rejects_mismatched_acc_fractal():
    t = _mx_tiles()
    t["dst"] = _tile(
        "dst", [64, 64], DataType.FP32, ir.MemorySpace.Acc,
        layout=ir.TensorLayout.NZ, fractal=1024,
    )
    t["acc"] = _tile(
        "acc", [64, 64], DataType.FP32, ir.MemorySpace.Acc,
        layout=ir.TensorLayout.NZ, fractal=512,
    )

    with pytest.raises(InvalidType, match=r"acc_tile type must match dst_tile type"):
        _ir_matmul_mx_acc(t["dst"], t["acc"], t["lhs"], t["rhs"], t["scale_a"], t["scale_b"])


def test_matmul_mx_acc_rejects_non_fp32_acc():
    t = _mx_tiles()
    t["acc"] = _tile("acc", [64, 64], DataType.FP16, ir.MemorySpace.Acc)

    with pytest.raises(InvalidType, match=r"acc_tile must use FP32 dtype"):
        _ir_matmul_mx_acc(t["dst"], t["acc"], t["lhs"], t["rhs"], t["scale_a"], t["scale_b"])


@pytest.mark.parametrize(
    "lhs_dtype,rhs_dtype,dst_dtype",
    [
        (DataType.FP8E4M3FN, DataType.FP4E2M1, DataType.FP32),
        (DataType.FP8E4M3FN, DataType.FP8E5M2, DataType.FP16),
    ],
)
def test_matmul_mx_rejects_invalid_dtype_combinations(lhs_dtype, rhs_dtype, dst_dtype):
    t = _mx_tiles(lhs_dtype=lhs_dtype, rhs_dtype=rhs_dtype, dst_dtype=dst_dtype)
    with pytest.raises(InvalidType, match="must be FP8/FP4 combo and dst FP32"):
        _ir_matmul_mx(t["dst"], t["lhs"], t["rhs"], t["scale_a"], t["scale_b"])


def test_matmul_mx_rejects_scale_in_wrong_memory_space():
    t = _mx_tiles(scale_a_space=ir.MemorySpace.Mat)
    with pytest.raises(ValueError, match="scale_a must be in L0A.*ScaleLeft"):
        _ir_matmul_mx(t["dst"], t["lhs"], t["rhs"], t["scale_a"], t["scale_b"])


def test_matmul_mx_rejects_scale_with_wrong_dtype():
    t = _mx_tiles(scale_a_dtype=DataType.FP16)
    with pytest.raises(InvalidType, match="scale_a must use FP8E8M0 dtype"):
        _ir_matmul_mx(t["dst"], t["lhs"], t["rhs"], t["scale_a"], t["scale_b"])


@pytest.mark.parametrize(
    "scale_key,shape,error_pattern",
    [
        ("scale_a", [32, 2], r"scale_a shape must match lhs_tile MX groups"),
        ("scale_a", [64, 4], r"scale_a shape must match lhs_tile MX groups"),
        ("scale_b", [4, 64], r"scale_b shape must match rhs_tile MX groups"),
        ("scale_b", [2, 32], r"scale_b shape must match rhs_tile MX groups"),
    ],
)
def test_matmul_mx_rejects_mismatched_scale_shape(scale_key, shape, error_pattern):
    t = _mx_tiles()
    space = ir.MemorySpace.ScaleLeft if scale_key == "scale_a" else ir.MemorySpace.ScaleRight
    t[scale_key] = _tile(scale_key, shape, DataType.FP8E8M0, space)

    with pytest.raises(InvalidType, match=error_pattern):
        _ir_matmul_mx(t["dst"], t["lhs"], t["rhs"], t["scale_a"], t["scale_b"])


def test_matmul_mx_acc_rejects_missing_scale():
    t = _mx_tiles()
    with pytest.raises(ValueError, match="scale_b is required"):
        _ir_matmul_mx_acc(t["dst"], t["acc"], t["lhs"], t["rhs"], t["scale_a"], None)


def test_matmul_mx_accepts_paired_scale_addresses():
    t = _mx_tiles()
    t["lhs"] = _tile("lhs", [64, 64], DataType.FP8E4M3FN, ir.MemorySpace.Left, addr=0x8000)
    t["rhs"] = _tile("rhs", [64, 64], DataType.FP8E5M2, ir.MemorySpace.Right, addr=0x4000)
    t["scale_a"] = _tile(
        "scale_a", [64, 2], DataType.FP8E8M0, ir.MemorySpace.ScaleLeft, addr=0x0800
    )
    t["scale_b"] = _tile(
        "scale_b", [2, 64], DataType.FP8E8M0, ir.MemorySpace.ScaleRight, addr=0x0400
    )

    _ir_matmul_mx(t["dst"], t["lhs"], t["rhs"], t["scale_a"], t["scale_b"])


@pytest.mark.parametrize(
    "data_key,scale_key,data_addr,scale_addr,error_pattern",
    [
        ("lhs", "scale_a", 0x8000, 0x0400, r"scale_a address must equal lhs_tile address >> 4"),
        ("rhs", "scale_b", 0x4000, 0x0200, r"scale_b address must equal rhs_tile address >> 4"),
    ],
)
def test_matmul_mx_rejects_unpaired_scale_address(
    data_key, scale_key, data_addr, scale_addr, error_pattern
):
    t = _mx_tiles()
    if data_key == "lhs":
        t[data_key] = _tile(data_key, [64, 64], DataType.FP8E4M3FN, ir.MemorySpace.Left, addr=data_addr)
        t[scale_key] = _tile(scale_key, [64, 2], DataType.FP8E8M0, ir.MemorySpace.ScaleLeft, addr=scale_addr)
    else:
        t[data_key] = _tile(data_key, [64, 64], DataType.FP8E5M2, ir.MemorySpace.Right, addr=data_addr)
        t[scale_key] = _tile(scale_key, [2, 64], DataType.FP8E8M0, ir.MemorySpace.ScaleRight, addr=scale_addr)

    with pytest.raises(ValueError, match=error_pattern):
        _ir_matmul_mx(t["dst"], t["lhs"], t["rhs"], t["scale_a"], t["scale_b"])


def test_matmul_mx_skips_address_check_without_explicit_memref():
    t = _mx_tiles()
    span = ir.Span.unknown()
    shape = [ir.ConstInt(dim, DataType.INDEX, span) for dim in [64, 64]]
    scale_shape = [ir.ConstInt(dim, DataType.INDEX, span) for dim in [64, 2]]
    t["lhs"] = ir.Var("lhs", ir.TileType(shape, DataType.FP8E4M3FN), span)
    scale_hw = ir.HardwareInfo(ir.TileLayout.row_major, ir.TileLayout.row_major, 32)
    t["scale_a"] = ir.Var(
        "scale_a", ir.TileType(scale_shape, DataType.FP8E8M0, hardware_info=scale_hw), span
    )

    _ir_matmul_mx(t["dst"], t["lhs"], t["rhs"], t["scale_a"], t["scale_b"])


@pytest.mark.parametrize("phase", [ir.STPhase.Final, 7])
def test_matmul_mx_rejects_invalid_phase(phase):
    t = _mx_tiles()
    with pytest.raises(ValueError, match="invalid phase value.*expected AccPhase"):
        _ir_matmul_mx(
            t["dst"], t["lhs"], t["rhs"], t["scale_a"], t["scale_b"], phase=phase
        )


def test_matmul_mx_acc_rejects_invalid_phase():
    t = _mx_tiles()
    with pytest.raises(ValueError, match="invalid phase value.*expected AccPhase"):
        _ir_matmul_mx_acc(
            t["dst"],
            t["acc"],
            t["lhs"],
            t["rhs"],
            t["scale_a"],
            t["scale_b"],
            phase=ir.STPhase.Partial,
        )


@pytest.mark.parametrize(
    "target_memory,layout,shape",
    [
        (ir.MemorySpace.Mat, ir.TensorLayout.ZZ, [64, 2]),
        (ir.MemorySpace.Mat, ir.TensorLayout.NN, [2, 64]),
        (ir.MemorySpace.ScaleLeft, ir.TensorLayout.ZZ, [64, 2]),
        (ir.MemorySpace.ScaleRight, ir.TensorLayout.NN, [2, 64]),
    ],
)
def test_mx_scale_tile_defaults_to_fractal_32(target_memory, layout, shape, monkeypatch):
    if target_memory in (ir.MemorySpace.ScaleLeft, ir.MemorySpace.ScaleRight):
        monkeypatch.setenv("PYPTOPRO_JIT_ARCH", "a5")
    tile_type = TileType(
        shape=shape, dtype=DataType.FP8E8M0, target_memory=target_memory, layout=layout
    )
    assert tile_type.fractal == 32


@pytest.mark.parametrize(
    "target_memory,layout,shape",
    [
        (ir.MemorySpace.Mat, ir.TensorLayout.ZZ, [64, 2]),
        (ir.MemorySpace.Mat, ir.TensorLayout.NN, [2, 64]),
        (ir.MemorySpace.ScaleLeft, ir.TensorLayout.ZZ, [64, 2]),
        (ir.MemorySpace.ScaleRight, ir.TensorLayout.NN, [2, 64]),
    ],
)
def test_mx_scale_tile_rejects_non_32_fractal(target_memory, layout, shape, monkeypatch):
    monkeypatch.setenv("PYPTOPRO_JIT_ARCH", "a5")
    with pytest.raises(ValueError, match="require fractal=32"):
        TileType(
            shape=shape,
            dtype=DataType.FP8E8M0,
            target_memory=target_memory,
            layout=layout,
            fractal=512,
        )


@pytest.mark.parametrize("memory_space", [ir.MemorySpace.ScaleLeft, ir.MemorySpace.ScaleRight])
def test_mx_scale_memory_requires_32_byte_alignment(memory_space):
    _validate_tile_addr_alignment(0x20, memory_space)
    with pytest.raises(ValueError, match="32-byte aligned"):
        _validate_tile_addr_alignment(0x01, memory_space)
