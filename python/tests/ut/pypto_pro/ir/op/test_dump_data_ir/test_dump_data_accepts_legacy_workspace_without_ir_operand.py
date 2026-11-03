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
"""The sibling ``.pypto`` file holds the complete TextDump output."""

from pathlib import Path
import re

from pypto_pro import ir
import pypto_pro.language as pl
from pypto_pro.runtime.platform import NpuArch


def test_dump_data_accepts_legacy_workspace_without_ir_operand(monkeypatch):
    monkeypatch.setenv("PYPTOPRO_JIT_ARCH", NpuArch.DAV_3510.name)

    @pl.jit(auto_mutex=False)
    def main(workspace: pl.Tensor[[64, 64], pl.DT_FP16]):
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        tile = pl.make_tile(tile_type, addr=0x0000)
        pl.dump_data(tile, workspace=workspace)

    program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Cube)
    func = program.get_function(main.__name__)
    calls = [stmt.expr for stmt in func.body.stmts if isinstance(stmt, ir.EvalStmt)]
    assert len(calls) == 1
    assert calls[0].name == "debug.dump_tile"
    assert len(calls[0].args) == 1

    printed = str(func)
    printed = re.sub(r"memref_id=\d+", "memref_id=<id>", printed)
    golden = Path(__file__).with_suffix(".pypto").read_text(encoding="utf-8")
    assert printed.rstrip("\n") == golden.rstrip("\n")
