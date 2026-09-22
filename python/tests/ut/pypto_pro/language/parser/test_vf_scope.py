# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reject operations in the wrong execution domain before building invalid VF IR."""

import ast
import inspect
import textwrap

from pypto_pro import ir
from pypto_pro._errors import InvalidOperation, NotSupported, PyptoProError
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.language.parser._ast_parser import ASTParser
import pytest


def _parse_body(body, *, vector_function=False, target=ir.SectionKind.Vector, closure=None):
    source = "def kernel(n: pl.DT_INT64):\n" + textwrap.indent(body, "    ")
    parser = ASTParser(
        source_file=__file__,
        source_lines=source.splitlines(),
        target=target,
        debug_info=ir.IRDebugInfo(),
        closure_vars={"pl": pl, "vf": vf, **(closure or {})},
    )
    return parser.parse_function(ast.parse(source).body[0], is_vector_function=vector_function)


@pytest.mark.parametrize("namespace", ["vf", "pl.Vf", "vectors", "pl.vf"])
@pytest.mark.parametrize("statement", [
    "mask = {ns}.create_mask()",
    "reg = {ns}.arange(0, dtype=pl.DT_INT32)",
    "low, high = {ns}.de_interleave(missing_a, missing_b)",
    "{ns}.store_align(missing_tile, missing_reg, missing_mask)",
    "value = 1 + {ns}.get_spr(missing_reg)",
])
@pytest.mark.parametrize("section", [None, "section_vector", "section_cube"])
def test_vf_operations_require_vector_function(namespace, statement, section):
    body = statement.format(ns=namespace)
    target = ir.SectionKind.Cube if section == "section_cube" else ir.SectionKind.Vector
    if section:
        body = f"with pl.{section}():\n" + textwrap.indent(body, "    ")
    with pytest.raises(InvalidOperation, match="can only be used inside @pl.vector_function") as error:
        _parse_body(body, target=target, closure={"vectors": vf})
    assert error.value.span["filename"] == __file__
    assert error.value.span["line"] == (3 if section else 2)
    assert "Move VF register operations" in error.value.hint


def test_plain_helper_does_not_create_a_vf_scope():
    def helper():
        mask = vf.create_mask()  # noqa: F841

    @pl.jit(auto_mutex=False)
    def kernel(n: pl.DT_INT64):
        helper()

    with pytest.raises(InvalidOperation, match="vf.create_mask") as error:
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    assert error.value.span["line"] == inspect.getsourcelines(helper)[1] + 1


@pytest.mark.parametrize("statement", [
    "pl.load(missing_tile, missing_tensor, [0, 0])",
    "pl.add(missing_dst, missing_lhs, missing_rhs)",
    "pl.system.bar_all()",
    "pl.mutex.mutex_lock(missing_tile)",
    "tile = pl.make_tile(missing_type)",
    "tile_type = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)",
    "value = pl.get_block_idx()",
    "value: pl.DT_INT64 = pl.get_block_num()",
    "value = vf.arange(pl.get_block_idx(), dtype=pl.DT_INT32)",
    "value = pl.struct(field=1)",
    "for i in pl.range(pl.get_block_num()):\n    value = i",
])
def test_pl_operations_are_rejected_in_vf_scope(statement):
    with pytest.raises(NotSupported, match="is not supported inside @pl.vector_function") as error:
        _parse_body(statement, vector_function=True)
    assert error.value.span["line"] == 2
    assert "Move other pl.* operations outside" in error.value.hint


def test_simt_launch_is_rejected_in_vf_scope():
    @pl.vector_function(mode="simt", max_threads=32)
    def entry():
        return

    with pytest.raises(NotSupported, match="is not supported inside @pl.vector_function"):
        _parse_body("entry[32]()", vector_function=True, closure={"entry": entry})


def test_unknown_indexed_call_does_not_enter_simt_scope_check():
    with pytest.raises(NotSupported, match="Unsupported indexed function call"):
        _parse_body("missing_helper[32]()", vector_function=True)


@pytest.mark.parametrize("section", ["section_vector", "section_cube"])
@pytest.mark.parametrize("target", [ir.SectionKind.Vector, ir.SectionKind.Cube])
def test_sections_cannot_be_nested_in_vector_function(section, target):
    body = f"with pl.{section}():\n    value = 1"
    with pytest.raises(InvalidOperation, match="cannot be nested inside @pl.vector_function") as error:
        _parse_body(body, vector_function=True, target=target)
    assert f"pl.{section}" in error.value.message
    assert error.value.span["line"] == 2
    assert "calling kernel" in error.value.hint


@pytest.mark.parametrize("context", ["pl.section_vector", "scope", "pl.unsupported()"])
@pytest.mark.parametrize("vector_function", [False, True])
def test_unsupported_context_retains_context_diagnostic(context, vector_function):
    with pytest.raises(NotSupported, match="Unsupported context manager") as error:
        _parse_body(f"with {context}:\n    value = 1", vector_function=vector_function)
    assert context in error.value.message


def test_pl_error_reports_the_decorated_helper_source():
    @pl.vector_function
    def inner():
        pl.system.bar_all()

    @pl.vector_function
    def outer():
        inner()

    @pl.jit(auto_mutex=True)
    def kernel(n: pl.DT_INT64):
        outer()

    with pytest.raises(NotSupported, match="pl.system.bar_all") as error:
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    assert error.value.span["line"] == inspect.getsourcelines(inner)[1] + 2


@pytest.mark.parametrize("body, closure, vector_function", [
    ("mask = make_mask()", {"make_mask": vf.create_mask}, False),
    ("reg = sequence(0, dtype=pl.DT_INT32)", {"sequence": vf.arange}, False),
    ("block = get_block()", {"get_block": pl.get_block_idx}, True),
])
def test_operation_aliases_cannot_bypass_scope_checks(body, closure, vector_function):
    with pytest.raises(PyptoProError, match="@pl.vector_function"):
        _parse_body(body, vector_function=vector_function, closure=closure)


def test_vf_operation_aliases_work_inside_vector_function():
    func = _parse_body(
        "mask = make_mask()\nreg = sequence(0, dtype=pl.DT_INT32)",
        vector_function=True,
        closure={"make_mask": vf.create_mask, "sequence": vf.arange},
    )
    assert func.body.stmts[0].section_kind == ir.SectionKind.VF


def test_scalar_operations_remain_valid_in_vector_function():
    func = _parse_body(
        "value = pl.min(n, 64)\nvalue = pl.max(value, 0)\nvalue = value + pl.const(1, pl.DT_INT64)",
        vector_function=True,
    )
    body = func.body.stmts[0].body.stmts
    assert isinstance(body[0].value, ir.Min)
    assert isinstance(body[1].value, ir.Max)
    assert isinstance(body[2].value, ir.Add)


@pytest.mark.parametrize("namespace", ["vf", "pl.Vf", "vectors", "pl.vf"])
def test_vf_range_and_constants_remain_valid(namespace):
    func = _parse_body(
        textwrap.dedent(f"""\
        for i in pl.range(n):
            mask = {namespace}.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
            reg = {namespace}.full(1.0, mask, dtype=pl.DT_FP32)
        """),
        vector_function=True,
        closure={"vectors": vf},
    )
    section = func.body.stmts[0]
    assert isinstance(section, ir.SectionStmt)
    assert section.section_kind == ir.SectionKind.VF
    assert isinstance(section.body.stmts[0], ir.ForStmt)


@pytest.mark.parametrize("after", [False, True])
def test_nested_vector_helpers_restore_caller_scope(after):
    @pl.vector_function
    def inner():
        mask = vf.create_mask()  # noqa: F841

    @pl.vector_function
    def outer():
        inner()
        inner()

    @pl.jit(auto_mutex=False)
    def kernel(n: pl.DT_INT64):
        outer()
        outer()
        if after:
            mask = vf.create_mask()  # noqa: F841
        else:
            block = pl.get_block_idx()  # noqa: F841

    if after:
        with pytest.raises(InvalidOperation, match="vf.create_mask"):
            kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    else:
        program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
        func = program.get_function(kernel.__name__)
        sections = [stmt for stmt in func.body.stmts if isinstance(stmt, ir.SectionStmt)]
        assert len(sections) == 2
        assert all(section.section_kind == ir.SectionKind.VF for section in sections)


# One helper per accessor: the parser reads these from source, so the cursor call has to be
# spelled literally -- a getattr(group, name)() indirection never reaches the tile-group route.
@pl.vector_function
def _advance_next(group):
    tile = group.next()
    reg = vf.load_align(tile, 0)
    vf.store_align(tile, reg, vf.create_mask(), 0)


@pl.vector_function
def _advance_current(group):
    tile = group.current()
    reg = vf.load_align(tile, 0)
    vf.store_align(tile, reg, vf.create_mask(), 0)


@pl.vector_function
def _advance_previous(group):
    tile = group.previous()
    reg = vf.load_align(tile, 0)
    vf.store_align(tile, reg, vf.create_mask(), 0)


def _tile_group_kernel(advance):
    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[1, 128], pl.DT_FP32]):
        tile_type = pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        group = pl.make_tile_group(type=tile_type, addrs=[0, 512], mutex_ids=[0, 1])
        with pl.section_vector():
            advance(group)

    return kernel


@pytest.mark.parametrize(
    "advance, accessor",
    [(_advance_next, "next"), (_advance_current, "current"), (_advance_previous, "previous")],
    ids=["next", "current", "previous"],
)
def test_tile_group_accessor_is_rejected_inside_vector_function(advance, accessor):
    """A group accessor is lowered by _route_ir_node_method, ahead of the op-registry dispatch.

    A domain check placed after that route would let the cursor bump -- a struct.set plus the
    scalar arithmetic that picks the buffer, all plain CCE work -- land inside the VEC_SCOPE
    before the call was rejected, which is exactly what bisheng refuses to expand there. Such a
    kernel parses without a diagnostic, so nothing but this test separates the two orderings.
    """
    kernel = _tile_group_kernel(advance)
    with pytest.raises(InvalidOperation, match=rf"Tile-group accessor '\.{accessor}\(\)'") as error:
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    assert error.value.span["line"] == inspect.getsourcelines(advance)[1] + 2
    assert "pass it to the vector function" in error.value.hint


def test_tile_group_accessor_stays_valid_in_the_calling_kernel():
    """The companion to the rejection above: the accessor is fine one scope out.

    Without this the rejection could be over-broad -- forbidding the tile itself rather than
    where the cursor is advanced -- and every VF kernel reading a rotating group would break.
    """

    @pl.vector_function
    def use(tile):
        reg = vf.load_align(tile, 0)
        vf.store_align(tile, reg, vf.create_mask(), 0)

    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[1, 128], pl.DT_FP32]):
        tile_type = pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        group = pl.make_tile_group(type=tile_type, addrs=[0, 512], mutex_ids=[0, 1])
        with pl.section_vector():
            use(group.next())

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = program.get_function(kernel.__name__)
    sections = [stmt for stmt in func.body.stmts if isinstance(stmt, ir.SectionStmt)]
    assert [section.section_kind for section in sections] == [ir.SectionKind.VF]


def test_lookalike_callable_is_not_mistaken_for_a_declaration():
    """Bare names resolve by identity against the declarations, not by their __module__ string.

    Renaming ``_vf_api``/``_api`` is the case that matters and no test can reach it: a string
    comparison would simply stop matching and reopen the alias bypass with no diagnostic. A
    callable that only *reports* the declaration module stands in for it -- matching on the
    string alone would take this ordinary helper for vf.create_mask and reject the kernel.
    """

    def create_mask():
        return 7

    create_mask.__module__ = "pypto_pro.language._vf_api"

    @pl.jit(auto_mutex=False)
    def kernel(n: pl.DT_INT64):
        value = create_mask()  # noqa: F841

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    assert program.get_function(kernel.__name__) is not None
