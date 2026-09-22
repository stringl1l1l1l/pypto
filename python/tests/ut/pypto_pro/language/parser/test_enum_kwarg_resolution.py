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
"""Unit tests for enum-valued kwarg resolution in DSL function bodies.

Covers the "support const enum" / "expects an enum value" changes:

  * An enum kwarg (dtype / target_type / mode / ...) may be written as an enum
    literal (``pl.DT_FP32`` / ``pl.RoundMode.X``), or captured from a closure
    variable — including multi-level closure assignment (``a = enum; b = a``)
    and kernel-factory ``dtype`` parameters used for dtype generalization.
  * The same enum kwarg rejects a plain int, whether written directly
    (``target_type=1``) or captured from an int closure variable — raising
    InvalidArgument ("expects an enum value").
"""
from pypto_pro import ir
from pypto_pro._errors import InvalidOperation, InvalidType, InvalidVal, NotSupported
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
import pytest

# Tile geometry for the VF-op tests below.
_VF_N, _VF_M = 1, 64
_VF_TILE_SIZE = _VF_N * _VF_M * 4  # 32-byte aligned


# ---------------------------------------------------------------------------
# Legal: enum kwarg written as an enum literal
# ---------------------------------------------------------------------------
@pytest.mark.soc("950")
def test_dtype_kwarg_enum_literal():
    """A dtype kwarg written directly as pl.DT_* resolves to a DataType."""

    @pl.jit(auto_mutex=False)
    def func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        result = pl.make_tile(tile_type, addr=0)
        _test_result = result

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)


@pytest.mark.soc("950")
def test_mode_kwarg_enum_literal():
    """A non-dtype enum kwarg (RoundMode) written directly is accepted."""

    @pl.jit(auto_mutex=False)
    def func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        src = pl.make_tile(tile_type, addr=0)
        tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        result = pl.make_tile(tile_type, addr=16384)
        pl.cast(result, src, mode=pl.RoundMode.CAST_ROUND)
        _test_result = result

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)


# ---------------------------------------------------------------------------
# Legal: enum kwarg captured from a closure variable
# ---------------------------------------------------------------------------
@pytest.mark.soc("950")
def test_dtype_kwarg_closure_enum_var():
    """A dtype kwarg captured from a one-level closure enum variable is accepted."""
    dt = pl.DT_FP32

    @pl.jit(auto_mutex=False)
    def func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64, 128], dtype=dt, target_memory=pl.MemorySpace.Vec)
        result = pl.make_tile(tile_type, addr=0)
        _test_result = result

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)


@pytest.mark.soc("950")
def test_mode_kwarg_closure_enum_var():
    """A RoundMode kwarg captured from a closure enum variable is accepted."""
    rounding = pl.RoundMode.CAST_FLOOR

    @pl.jit(auto_mutex=False)
    def func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        src = pl.make_tile(tile_type, addr=0)
        tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        result = pl.make_tile(tile_type, addr=16384)
        pl.cast(result, src, mode=rounding)
        _test_result = result

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)


# ---------------------------------------------------------------------------
# Legal: multi-level closure assignment  a = enum; b = a; kwarg=b
# ---------------------------------------------------------------------------
@pytest.mark.soc("950")
def test_dtype_kwarg_multilevel_closure_assignment():
    """An enum passed through multiple closure assignments still resolves."""
    dt_a = pl.DT_FP32
    dt_b = dt_a

    @pl.jit(auto_mutex=False)
    def func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64, 128], dtype=dt_b, target_memory=pl.MemorySpace.Vec)
        result = pl.make_tile(tile_type, addr=0)
        _test_result = result

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)


# ---------------------------------------------------------------------------
# Legal: kernel-factory dtype generalization (closure dtype param)
# ---------------------------------------------------------------------------
@pytest.mark.soc("950")
def test_kernel_factory_closure_dtype():
    """A dtype closure parameter drives both the annotation and a dtype kwarg."""

    def make_kernel(dtype):
        @pl.jit(auto_mutex=False)
        def func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
            tile_type = pl.TileType(shape=[64, 128], dtype=dtype, target_memory=pl.MemorySpace.Vec)
            result = pl.make_tile(tile_type, addr=0)
            _test_result = result

        func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
        func = func_program.get_function(func.__name__)

        return func

    assert isinstance(make_kernel(pl.DT_FP32), ir.Function)
    assert isinstance(make_kernel(pl.DT_INT32), ir.Function)


# ---------------------------------------------------------------------------
# Illegal: enum kwarg given a plain int (literal)
# ---------------------------------------------------------------------------
@pytest.mark.soc("950")
def test_dtype_kwarg_int_literal_rejected():
    """A dtype kwarg given a raw int literal raises InvalidArgument."""
    with pytest.raises(InvalidVal, match="expects an enum value"):

        @pl.jit(auto_mutex=False)
        def func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
            tile_type = pl.TileType(shape=[64, 128], dtype=1, target_memory=pl.MemorySpace.Vec)
            result = pl.make_tile(tile_type, addr=0)
            _test_result = result

        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


@pytest.mark.soc("950")
def test_mode_kwarg_int_literal_rejected():
    """A RoundMode kwarg given a raw int literal raises InvalidArgument."""
    with pytest.raises(InvalidVal, match="expects an enum value"):

        @pl.jit(auto_mutex=False)
        def func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
            tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
            src = pl.make_tile(tile_type, addr=0)
            tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
            result = pl.make_tile(tile_type, addr=16384)
            pl.cast(result, src, mode=1)
            _test_result = result

        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


# ---------------------------------------------------------------------------
# Illegal: enum kwarg given an int captured from a closure variable
# ---------------------------------------------------------------------------
@pytest.mark.soc("950")
def test_dtype_kwarg_int_closure_var_rejected():
    """A dtype kwarg given an int closure variable raises InvalidArgument.

    ``target_type=iv`` and ``target_type=1`` reach the resolver as the same int,
    so the int closure variable is rejected exactly like the literal.
    """
    iv = 1
    with pytest.raises(InvalidVal, match="expects an enum value"):

        @pl.jit(auto_mutex=False)
        def func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
            tile_type = pl.TileType(shape=[64, 128], dtype=iv, target_memory=pl.MemorySpace.Vec)
            result = pl.make_tile(tile_type, addr=0)
            _test_result = result

        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


# ---------------------------------------------------------------------------
# MemorySpace enum kwarg (target_memory)
# ---------------------------------------------------------------------------
@pytest.mark.soc("950")
def test_target_memory_enum_literal():
    """A MemorySpace kwarg written as an enum literal is accepted."""

    @pl.jit(auto_mutex=False)
    def func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        a = pl.make_tile(tile_type, addr=0)  # noqa: F841
        tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        result = pl.make_tile(tile_type, addr=0)
        _test_result = result

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)


@pytest.mark.soc("950")
def test_target_memory_int_rejected():
    """A MemorySpace kwarg given a raw int raises InvalidArgument."""
    with pytest.raises(InvalidVal, match="expects an enum value"):

        @pl.jit(auto_mutex=False)
        def func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
            tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=1)  # noqa: F841
            tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
            result = pl.make_tile(tile_type, addr=0)
            _test_result = result

        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


# ---------------------------------------------------------------------------
# Non-enum kwargs still accept ints (must NOT be caught by the enum guard)
# ---------------------------------------------------------------------------
@pytest.mark.soc("950")
def test_cmp_mode_is_not_an_enum_kwarg():
    """``cmp_mode`` is a plain-int parameter and must be excluded from the guard.

    Guards against a regression where ``_VF_KWARG_ENUMS`` (which maps cmp_mode to
    CompareMode) is used verbatim as the guard list — that would wrongly reject
    the legitimate ``cmp_mode=<int>`` used by block cmp/gather.
    """
    from pypto_pro.language.parser._call_parser import CallParserMixin

    assert "cmp_mode" not in CallParserMixin._ENUM_KWARGS


@pytest.mark.soc("950")
def test_fractal_int_kwarg_still_allowed():
    """TileType's numeric fractal kwarg accepts a raw int, not an enum value."""

    @pl.jit(auto_mutex=False)
    def func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Acc, fractal=1024)
        a = pl.make_tile(tile_type, addr=0)  # noqa: F841
        tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        result = pl.make_tile(tile_type, addr=0)
        _test_result = result

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)


# ---------------------------------------------------------------------------
# VF-op enum kwargs (parsed when the enclosing jit kernel is parsed)
# ---------------------------------------------------------------------------
def _parse_vf_mask_kernel(pattern):
    """Build a jit kernel whose section_vector calls a VF op with the given
    ``pattern`` kwarg, and force-parse it (VF ops are parsed as part of the
    enclosing kernel's Vector section)."""

    @pl.vector_function
    def vf_body(in_a, t_out):
        preg = vf.create_mask(pattern=pattern, dtype=pl.DT_FP32)
        reg_a = vf.load_align(in_a, 0)
        vf.store_align(t_out, reg_a, preg)

    @pl.jit()
    def kernel(a: pl.Tensor[[_VF_N, _VF_M], pl.DT_FP32], out: pl.Tensor[[_VF_N, _VF_M], pl.DT_FP32]):
        tf = pl.TileType(shape=[_VF_N, _VF_M], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        in_a = pl.make_tile(tf, addr=0)
        t_out = pl.make_tile(tf, addr=_VF_TILE_SIZE)
        with pl.section_vector():
            pl.load(in_a, a, [0, 0])
            vf_body(in_a, t_out)
            pl.store(out, t_out, [0, 0])

    return kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


@pytest.mark.soc("950")
def test_vf_enum_kwarg_literal():
    """A VF-op enum kwarg (create_mask pattern) written as an enum literal parses."""
    _parse_vf_mask_kernel(pl.MaskPattern.ALL)


@pytest.mark.soc("950")
def test_vf_enum_kwarg_closure_var():
    """A VF-op enum kwarg captured from a closure enum variable parses."""
    pat = pl.MaskPattern.ALL
    _parse_vf_mask_kernel(pat)


@pytest.mark.soc("950")
def test_vf_register_writes_are_not_loop_carried_ssa_values():
    @pl.vector_function
    def vf_body(in_a):
        for _ in pl.range(1):
            src0 = vf.load_align(in_a, 0)
            src1 = vf.load_align(in_a, 0)
            dst0, dst1 = vf.de_interleave(src0, src1)

    @pl.jit()
    def kernel(a: pl.Tensor[[_VF_N, _VF_M], pl.DT_FP32]):
        tf = pl.TileType(shape=[_VF_N, _VF_M], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        in_a = pl.make_tile(tf, addr=0)
        with pl.section_vector():
            pl.load(in_a, a, [0, 0])
            vf_body(in_a)

    kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


@pytest.mark.soc("950")
def test_vf_enum_kwarg_int_rejected():
    """A VF-op enum kwarg given a raw int raises InvalidArgument."""
    with pytest.raises(InvalidVal, match="expects an enum value"):
        _parse_vf_mask_kernel(1)


# ---------------------------------------------------------------------------
# pl.const() dtype validation (scalar_ops._parse_typed_constant)
# ---------------------------------------------------------------------------
@pytest.mark.soc("950")
def test_const_valid_dtype_enum_literal():
    """pl.const() with a valid dtype enum literal (pl.DT_INT32) is accepted."""

    @pl.jit(auto_mutex=False)
    def func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
        c = pl.const(42, pl.DT_INT32)  # noqa: F841
        _test_result = x

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)


@pytest.mark.soc("950")
def test_const_valid_dtype_closure_var():
    """pl.const() with a dtype from a closure variable is accepted."""
    dt = pl.DT_FP32

    @pl.jit(auto_mutex=False)
    def func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
        c = pl.const(1.0, dt)  # noqa: F841
        _test_result = x

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)


@pytest.mark.soc("950")
def test_const_dtype_int_literal_rejected():
    """pl.const() with an int instead of a dtype raises InvalidType."""
    with pytest.raises(InvalidType, match="must be a dtype"):

        @pl.jit(auto_mutex=False)
        def func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
            c = pl.const(42, 1)  # noqa: F841
            _test_result = x

        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


@pytest.mark.soc("950")
def test_const_dtype_int_closure_var_rejected():
    """pl.const() with an int closure variable as dtype raises InvalidType."""
    bad_dtype = 1

    with pytest.raises(InvalidType, match="must be a dtype"):

        @pl.jit(auto_mutex=False)
        def func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
            c = pl.const(42, bad_dtype)  # noqa: F841
            _test_result = x

        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


@pytest.mark.soc("950")
def test_const_dtype_string_rejected():
    """pl.const() with a string instead of a dtype raises InvalidType."""
    with pytest.raises(InvalidType, match="must be a dtype"):

        @pl.jit(auto_mutex=False)
        def func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
            c = pl.const(42, "fp32")  # noqa: F841
            _test_result = x

        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


# ---------------------------------------------------------------------------
# Enum comparison: only == and != are supported
# ---------------------------------------------------------------------------
@pytest.mark.soc("950")
def test_enum_compare_eq():
    """Enum == enum folds to a compile-time ConstBool (True case)."""

    @pl.jit(auto_mutex=False)
    def func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
        if pl.DT_FP16 == pl.DT_FP16:
            pass
        _test_result = x

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)


@pytest.mark.soc("950")
def test_enum_compare_eq_false():
    """Enum == enum folds to a compile-time ConstBool (False case)."""

    @pl.jit(auto_mutex=False)
    def func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
        if pl.DT_FP16 == pl.DT_FP32:
            pass
        _test_result = x

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)


@pytest.mark.soc("950")
def test_enum_compare_ne():
    """Enum != enum folds to a compile-time ConstBool."""

    @pl.jit(auto_mutex=False)
    def func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
        if pl.DT_FP16 != pl.DT_FP32:
            pass
        _test_result = x

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)


@pytest.mark.soc("950")
def test_enum_compare_lt_rejected():
    """Enum < enum raises NotSupported (only == and != allowed)."""
    with pytest.raises(NotSupported, match="Only == and != are supported"):

        @pl.jit(auto_mutex=False)
        def func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
            if pl.DT_FP16 < pl.DT_FP32:
                pass
            _test_result = x

        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


@pytest.mark.soc("950")
def test_enum_compare_gt_rejected():
    """Enum > enum raises NotSupported."""
    with pytest.raises(NotSupported, match="Only == and != are supported"):

        @pl.jit(auto_mutex=False)
        def func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
            if pl.DT_FP16 > pl.DT_FP32:
                pass
            _test_result = x

        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


@pytest.mark.soc("950")
def test_enum_compare_le_rejected():
    """Enum <= enum raises NotSupported."""
    with pytest.raises(NotSupported, match="Only == and != are supported"):

        @pl.jit(auto_mutex=False)
        def func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
            if pl.DT_FP16 <= pl.DT_FP32:
                pass
            _test_result = x

        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


@pytest.mark.soc("950")
def test_enum_compare_ge_rejected():
    """Enum >= enum raises NotSupported."""
    with pytest.raises(NotSupported, match="Only == and != are supported"):

        @pl.jit(auto_mutex=False)
        def func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
            if pl.DT_FP16 >= pl.DT_FP32:
                pass
            _test_result = x

        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


@pytest.mark.soc("950")
def test_enum_compare_closure_vars():
    """Enum comparison with closure variables folds correctly."""
    dt_a = pl.DT_FP32
    dt_b = pl.DT_FP32

    @pl.jit(auto_mutex=False)
    def func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
        if dt_a == dt_b:
            pass
        _test_result = x

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)


# ---------------------------------------------------------------------------
# Ternary expressions selecting an enum value (parse_ifexp)
# ---------------------------------------------------------------------------
# An enum is not an ir.Expr -- parse_attribute / parse_name return the Python enum object
# itself, so an enum may only reach a kwarg, never an IR operand. A ternary over a
# *constant* condition is folded at parse time, so it IS the chosen branch: the enum is
# passed through, exactly as if written literally, and the unselected branch is never
# parsed. That is what lets a kernel body pick its own dtype, e.g.
#
#     io_dtype = pl.DT_BF16 if DataType == 0 else pl.DT_FP16
#
# The cases below cover both that the fold accepts enums and that it still selects the
# right branch -- "it parsed" alone would also hold for an implementation that always
# took the then-branch.
#
# A ternary over a runtime condition is a different path: the branches are phi-merged into
# one runtime value, which an enum has no representation for. The rejection tests at the
# end of this section cover that side, and specifically that the diagnostic blames the
# condition -- the branch text is identical to the folded cases above, so the condition is
# the only thing that makes it illegal.

@pytest.mark.soc("950")
def test_dtype_kwarg_enum_ternary_const_condition():
    """A dtype kwarg fed by a ternary over a constant condition is accepted."""
    data_type = 0

    @pl.jit(auto_mutex=False)
    def func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
        target = pl.DT_FP32 if data_type == 0 else pl.DT_INT32
        tile_type = pl.TileType(shape=[64, 128], dtype=target, target_memory=pl.MemorySpace.Vec)
        result = pl.make_tile(tile_type, addr=0)
        _test_result = result

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)


@pytest.mark.soc("950")
def test_dtype_kwarg_enum_ternary_inline():
    """The ternary may sit inline in the kwarg, not only behind a local name."""

    @pl.jit(auto_mutex=False)
    def func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
        tile_type = pl.TileType(
            shape=[64, 128], dtype=pl.DT_FP32 if True else pl.DT_INT32, target_memory=pl.MemorySpace.Vec
        )
        result = pl.make_tile(tile_type, addr=0)
        _test_result = result

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)


@pytest.mark.soc("950")
def test_enum_ternary_condition_is_an_enum_comparison():
    """The condition may itself be an enum comparison (dtype generalization).

    ``in_dtype == pl.DT_FP16`` folds to a ConstBool via parse_compare, so this is the
    kernel-factory idiom: one dtype parameter picking the dtypes of every level below.
    """

    def make(in_dtype):
        @pl.jit(auto_mutex=False)
        def func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
            acc = pl.DT_FP32 if in_dtype == pl.DT_FP16 else pl.DT_INT32
            tile_type = pl.TileType(shape=[64, 128], dtype=acc, target_memory=pl.MemorySpace.Vec)
            result = pl.make_tile(tile_type, addr=0)
            _test_result = result

        func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
        func = func_program.get_function(func.__name__)

        return func

    assert isinstance(make(pl.DT_FP16), ir.Function)
    assert isinstance(make(pl.DT_FP32), ir.Function)


@pytest.mark.soc("950")
def test_non_dtype_enum_ternary():
    """The pass-through is not dtype-specific -- a MemorySpace ternary works too.

    parse_ifexp keys off "is this an enum", not "is this a DataType", and DataType is
    matched by type while other pybind enums are matched by ``__members__``/``.value``.
    Covering a second enum family keeps the predicate from being narrowed to dtypes.
    """
    use_vec = True

    @pl.jit(auto_mutex=False)
    def func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
        space = pl.MemorySpace.Vec if use_vec else pl.MemorySpace.Acc
        tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=space)
        a = pl.make_tile(tile_type, addr=0)  # noqa: F841
        tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        result = pl.make_tile(tile_type, addr=0)
        _test_result = result

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)


@pytest.mark.soc("950")
def test_enum_ternary_unselected_branch_not_parsed():
    """Only the selected branch is parsed; the dead branch is never looked at.

    This is what makes the pass-through safe rather than a loosened check: with a constant
    condition there is no second value to reconcile, so an undefined name in the dead
    branch cannot fail. If the fold ever regressed to parsing both branches, the
    undefined name would raise and this case would catch it.
    """

    @pl.jit(auto_mutex=False)
    def func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
        target = pl.DT_FP32 if True else undefined_name_in_dead_branch  # noqa: F821
        tile_type = pl.TileType(shape=[64, 128], dtype=target, target_memory=pl.MemorySpace.Vec)
        result = pl.make_tile(tile_type, addr=0)
        _test_result = result

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)


@pytest.mark.soc("950")
def test_enum_ternary_runtime_condition_is_rejected():
    """A runtime condition turns the ternary into a select, which an enum cannot feed.

    The branch text is what the folded cases above accept; only the condition differs, so
    the message has to name the condition rather than the branch -- otherwise the rule
    reads as "enums do not work in ternaries", which the tests above disprove.
    """
    with pytest.raises(InvalidOperation) as excinfo:

        @pl.jit(auto_mutex=False)
        def func(x: pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP16]):
            tile_type = pl.TileType(
                shape=[64, 128],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Vec,
                pad=pl.TilePad.zero if x.shape[0] else 0,
            )
            tile = pl.make_tile(tile_type, addr=0)  # noqa: F841

        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)

    message = str(excinfo.value)
    # the value that cannot be selected, quoted from the source
    assert "'pl.TilePad.zero' has no runtime value" in message
    # the condition that made it a runtime select, and the way out
    assert "x.shape[0]" in message
    assert "tiling_key" in message


@pytest.mark.soc("950")
def test_enum_ternary_runtime_condition_rejected_in_the_else_branch():
    """The else branch is held to the same contract, with the same diagnostic.

    Worth its own case: the then and else rejections are separate call sites, and only a
    shared one keeps them from drifting apart again.
    """
    with pytest.raises(InvalidOperation, match="'pl.MemorySpace.Acc' has no runtime value"):

        @pl.jit(auto_mutex=False)
        def func(x: pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP16]):
            n = x.shape[0]
            space = n if n else pl.MemorySpace.Acc  # noqa: F841

        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


@pytest.mark.soc("950")
def test_int_via_ternary_still_rejected_by_enum_kwarg_guard():
    """A ternary is not a way around the enum-kwarg guard.

    The ternary yields an int here, and the kwarg resolver rejects it exactly as it does
    for ``target_type=1`` written directly -- the pass-through widens what a ternary may
    *carry*, not what a kwarg may *accept*.
    """
    with pytest.raises(InvalidVal, match="expects an enum value"):

        @pl.jit(auto_mutex=False)
        def func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
            tile_type = pl.TileType(shape=[64, 128], dtype=1 if True else 2, target_memory=pl.MemorySpace.Vec)
            result = pl.make_tile(tile_type, addr=0)
            _test_result = result

        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


@pytest.mark.soc("950")
def test_mixed_enum_int_branches_follow_the_selected_branch():
    """Branches need not agree in kind, because only one of them is ever parsed.

    Selecting the enum side is fine; selecting the int side hits the kwarg guard. The
    asymmetry is the point: validity is decided by the chosen branch alone.
    """

    @pl.jit(auto_mutex=False)
    def picks_enum(x: pl.Tensor[[64, 128], pl.DT_FP16]):
        target = pl.DT_FP32 if True else 1
        tile_type = pl.TileType(shape=[64, 128], dtype=target, target_memory=pl.MemorySpace.Vec)
        result = pl.make_tile(tile_type, addr=0)
        _test_result = result

    picks_enum_program, _ = picks_enum.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    picks_enum = picks_enum_program.get_function(picks_enum.__name__)

    assert isinstance(picks_enum, ir.Function)

    with pytest.raises(InvalidVal, match="expects an enum value"):

        @pl.jit(auto_mutex=False)
        def picks_int(x: pl.Tensor[[64, 128], pl.DT_FP16]):
            target = pl.DT_FP32 if False else 1
            tile_type = pl.TileType(shape=[64, 128], dtype=target, target_memory=pl.MemorySpace.Vec)
            result = pl.make_tile(tile_type, addr=0)
            _test_result = result

        picks_int.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


# ---------------------------------------------------------------------------
# Legal: an enum reached through an expression with no IR form
# ---------------------------------------------------------------------------
_DTYPE_TABLE = [pl.DT_FP16, pl.DT_FP32]


def _pick_dtype(wide):
    """Plain Python helper selecting a dtype; its ternary has no IR form."""
    return pl.DT_FP32 if wide else pl.DT_FP16


@pytest.mark.soc("950")
def test_dtype_kwarg_from_closure_enum_list():
    """Indexing a closure list of enums yields the enum, not an IR value.

    A list of DataType has no IR form, so the index cannot be lowered; it is
    evaluated at parse time and the enum is used as if written literally.
    """

    @pl.jit(auto_mutex=False)
    def func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64, 128], dtype=_DTYPE_TABLE[1], target_memory=pl.MemorySpace.Vec)
        result = pl.make_tile(tile_type, addr=0)
        _test_result = result

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)
    assert "dtype=float" in str(func)


@pytest.mark.soc("950")
def test_dtype_kwarg_from_helper_returning_enum():
    """A Python helper may return the enum for an enum kwarg."""

    @pl.jit(auto_mutex=False)
    def func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64, 128], dtype=_pick_dtype(True), target_memory=pl.MemorySpace.Vec)
        result = pl.make_tile(tile_type, addr=0)
        _test_result = result

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)
    assert "dtype=float" in str(func)


@pytest.mark.soc("950")
def test_helper_returning_int_still_rejected_for_enum_kwarg():
    """Accepting enums must not let a plain int through the enum guard.

    The kwarg check runs inside the IR path; evaluating the helper in Python
    yields an int, which is not an enum, so the rejection still stands.
    """

    def _pick_int(_wide):
        return 1

    with pytest.raises(InvalidVal, match="expects an enum value"):

        @pl.jit(auto_mutex=False)
        def func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
            tile_type = pl.TileType(shape=[64, 128], dtype=_pick_int(True), target_memory=pl.MemorySpace.Vec)
            result = pl.make_tile(tile_type, addr=0)
            _test_result = result

        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


if __name__ == "__main__":
    _tests = [
        test_dtype_kwarg_enum_literal,
        test_mode_kwarg_enum_literal,
        test_dtype_kwarg_closure_enum_var,
        test_mode_kwarg_closure_enum_var,
        test_dtype_kwarg_multilevel_closure_assignment,
        test_kernel_factory_closure_dtype,
        test_dtype_kwarg_int_literal_rejected,
        test_mode_kwarg_int_literal_rejected,
        test_dtype_kwarg_int_closure_var_rejected,
        test_target_memory_enum_literal,
        test_target_memory_int_rejected,
        test_cmp_mode_is_not_an_enum_kwarg,
        test_fractal_int_kwarg_still_allowed,
        test_vf_enum_kwarg_literal,
        test_vf_enum_kwarg_closure_var,
        test_vf_enum_kwarg_int_rejected,
        test_const_valid_dtype_enum_literal,
        test_const_valid_dtype_closure_var,
        test_const_dtype_int_literal_rejected,
        test_const_dtype_int_closure_var_rejected,
        test_const_dtype_string_rejected,
        test_enum_compare_eq,
        test_enum_compare_eq_false,
        test_enum_compare_ne,
        test_enum_compare_lt_rejected,
        test_enum_compare_gt_rejected,
        test_enum_compare_le_rejected,
        test_enum_compare_ge_rejected,
        test_enum_compare_closure_vars,
        test_dtype_kwarg_enum_ternary_const_condition,
        test_dtype_kwarg_enum_ternary_inline,
        test_enum_ternary_condition_is_an_enum_comparison,
        test_non_dtype_enum_ternary,
        test_enum_ternary_unselected_branch_not_parsed,
        test_enum_ternary_runtime_condition_is_rejected,
        test_enum_ternary_runtime_condition_rejected_in_the_else_branch,
        test_int_via_ternary_still_rejected_by_enum_kwarg_guard,
        test_mixed_enum_int_branches_follow_the_selected_branch,
        test_dtype_kwarg_from_closure_enum_list,
        test_dtype_kwarg_from_helper_returning_enum,
        test_helper_returning_int_still_rejected_for_enum_kwarg,
    ]
    for _t in _tests:
        _t()
        print(f"{_t.__name__} passed!")
    print(f"All {len(_tests)} enum-kwarg resolution tests passed!")
