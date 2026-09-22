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
"""Block-facing smoke tests for pypto_pro.ir op registration."""

from pypto_pro import DataType, ir
import pytest


def _tensor_var(name: str, dtype=DataType.FP16):
    span = ir.Span.unknown()
    tensor_type = ir.TensorType([ir.ConstInt(64, DataType.INDEX, span)], dtype)
    return ir.Var(name, tensor_type, span)


@pytest.mark.parametrize(
    "op_name",
    ["block.add", "block.cast", "block.move", "block.quant", "block.load", "block.store"],
)
def test_block_visible_ops_are_registered(op_name):
    assert ir.is_op_registered(op_name)
    assert ir.get_op(op_name).name == op_name


@pytest.mark.parametrize(
    "op_name, attrs",
    [
        ("block.cast", {"target_type", "mode"}),
        ("block.load", {"is_transpose", "tile_dims"}),
        ("block.store", {"atomic", "phase", "relu_pre_mode"}),
        ("block.move", {"acc_to_vec_mode", "phase", "relu_pre_mode"}),
    ],
)
def test_block_op_kwarg_schema_is_exposed(op_name, attrs):
    op = ir.get_op(op_name)

    assert attrs.issubset(set(op.get_attr_keys()))
    assert all(op.has_attr(attr) for attr in attrs)


def test_create_op_call_accepts_registered_kwargs():
    src = _tensor_var("src")
    span = ir.Span.unknown()
    shape = ir.MakeTuple([ir.ConstInt(64, DataType.INDEX, span)], span)
    stride = ir.MakeTuple([ir.ConstInt(1, DataType.INDEX, span)], span)
    call = ir.create_op_call(
        "ptr.make_tensor",
        [src, shape, stride],
        {"dtype": DataType.FP32},
        span,
    )

    assert isinstance(call.type, ir.TensorType)
    assert call.type.dtype == DataType.FP32


@pytest.mark.parametrize(
    "kwargs",
    [
        {"unknown_param": 123},
        {"atomic": "true"},
    ],
)
def test_create_op_call_rejects_invalid_kwargs(kwargs):
    dst = _tensor_var("dst")
    src = _tensor_var("src")
    span = ir.Span.unknown()
    offsets = ir.MakeTuple([ir.ConstInt(0, DataType.INDEX, span)], span)

    with pytest.raises(Exception):
        ir.create_op_call("block.store", [dst, src, offsets], kwargs, span)


# ===================================================================
# VF op registration contract (b64/compare workstream)
# ===================================================================


@pytest.mark.parametrize(
    "op_name",
    [
        "vf.eq",
        "vf.ne",
        "vf.lt",
        "vf.gt",
        "vf.le",
        "vf.ge",
        "vf.addc",
        "vf.subc",
        "vf.select",
        "vf.mull",
        "vf.bit_cast",
    ],
)
def test_vf_ops_are_registered(op_name):
    assert ir.is_op_registered(op_name)
    assert ir.get_op(op_name).name == op_name


def _vf_reg(name: str, dtype=DataType.UINT32):
    return ir.Var(name, ir.ScalarType(dtype), ir.Span.unknown())


@pytest.mark.parametrize(
    "op_name, args",
    [
        ("vf.eq", ["dst", "src0", "src1", "mask"]),
        ("vf.ne", ["dst", "src0", "src1", "mask"]),
        ("vf.lt", ["dst", "src0", "src1", "mask"]),
        ("vf.gt", ["dst", "src0", "src1", "mask"]),
        ("vf.le", ["dst", "src0", "src1", "mask"]),
        ("vf.ge", ["dst", "src0", "src1", "mask"]),
        ("vf.select", ["dst", "src_true", "src_false", "mask"]),
        ("vf.addc", ["carry_out", "dst", "src0", "src1", "carry_in", "mask"]),
        ("vf.subc", ["borrow_out", "dst", "src0", "src1", "borrow_in", "mask"]),
        ("vf.mull", ["dst_lo", "dst_hi", "src0", "src1", "mask"]),
    ],
)
def test_vf_ops_ignore_removed_kwargs(op_name, args):
    # cmp_dtype (compare width override, compare ops) and mode (merge mode,
    # addc/subc/select/mull) were removed: the registrations no longer declare
    # them. vf-category ops do not validate unknown kwargs at create_op_call
    # (unlike tensor.* ops), so the removed kwargs are silently stored in the
    # Call and never read by the emitters — assert that contract here.
    assert not ir.get_op(op_name).has_attr("cmp_dtype")
    assert not ir.get_op(op_name).has_attr("mode")
    call_args = [_vf_reg(n) for n in args]
    call = ir.create_op_call(op_name, call_args, {"cmp_dtype": DataType.UINT8, "mode": 0}, ir.Span.unknown())
    assert call is not None


def test_vf_bit_cast_has_dtype_attr():
    assert ir.get_op("vf.bit_cast").has_attr("dtype")


def test_vf_bit_cast_deduce_keeps_tile_type():
    # The deducer must preserve the register (TileType) nature of the source so a
    # materialized temp is declared as RegTensor<T>, not as a scalar.
    span = ir.Span.unknown()
    src = ir.Var("bf16_reg", ir.TileType([ir.ConstInt(64, DataType.INDEX, span)], DataType.BF16), span)
    call = ir.create_op_call("vf.bit_cast", [src], {"dtype": DataType.UINT16}, span)

    assert isinstance(call.type, ir.TileType)
    assert call.type.dtype == DataType.UINT16
