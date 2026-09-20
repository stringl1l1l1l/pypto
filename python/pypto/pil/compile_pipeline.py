#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Helpers for lowering PIL-compiled functions through the new IR pipeline."""

import os

from pypto.runtime import setup_verify_data

from .. import ir, pil, pypto_impl


def _dump_program(name, program):
    if not pypto_impl.GetPassDefaultConfig(pypto_impl.KEY_PRINT_GRAPH, False):
        return
    dump_dir = pypto_impl.LogTopFolder() + "/TensorGraph/IR"
    os.makedirs(dump_dir, exist_ok=True)
    safe_name = name.replace(" ", "_")
    with open(os.path.join(dump_dir, f"ir_dump_{safe_name}.txt"), "w") as f:
        f.write(str(program))


def _build_default_pipeline():
    infer_token_pass = ir.Pass.infer_token_pass()
    dce = ir.Pass.aggressive_dce()
    canonicalize = ir.Pass.canonicalize()
    merge_stmts = ir.Pass.merge_stmts_into_if()
    simplify_symbolic_scalar = ir.Pass.simplify_symbolic_scalar()
    remove_redundant_tokens = ir.Pass.remove_redundant_token_pass()
    create_root_functions = ir.Pass.create_root_functions()
    infer_multi_iter_overlap = ir.Pass.infer_multi_iter_overlap()
    finalize = ir.Pass.finalize_dynamic_function()

    return [
        ("infer_token_pass", infer_token_pass),
        ("first_canonicalize_dce", lambda p:dce(canonicalize(p))),
        ("second_canonicalize_dce", lambda p:dce(canonicalize(p))),
        ("canonicalize(merge_stmts)", lambda p:canonicalize(merge_stmts(p))),
        ("simplify_symbolic_scalar", simplify_symbolic_scalar),
        ("remove_redundant_token_pass", remove_redundant_tokens),
        ("create_root_functions", create_root_functions),
        ("infer_multi_iter_overlap", infer_multi_iter_overlap),
        ("finalize", finalize),
    ]


def _run_pipeline(program, pipeline):
    _dump_program("initial", program)
    for name, transform in pipeline:
        program = transform(program)
        _dump_program(f"after {name}", program)
    return program


def compile_new_ir(pyfunc, *args, pipeline=None, **kwargs):
    """Compile a Python function and run the complete new-IR lowering pipeline."""
    create_new_logical_tensor = kwargs.pop("create_new_logical_tensor", False)
    pypto_impl.ir.set_assemble_new_logical_tensor(create_new_logical_tensor)
    try:
        builder = ir.IRBuilder()
        func = pil.compile(
            pyfunc, *args, create_new_logical_tensor=create_new_logical_tensor, **kwargs
        )
        program = builder.create_program([func], "main", ir.Span.unknown())
        setup_verify_data(args)

        if pipeline is None:
            pipeline = _build_default_pipeline()

        program = _run_pipeline(program, pipeline)
        return program.functions[func.name]
    finally:
        pypto_impl.ir.set_assemble_new_logical_tensor(False)
