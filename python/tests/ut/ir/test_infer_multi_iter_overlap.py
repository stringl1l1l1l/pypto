# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import pypto
from pypto import ir

from .test_common import run_root_function

TILE = 16
HIDDEN = 16

def _run_overlap_pass(kernel, *args):
    with pypto.options(pass_options={"enable_slice": True}):
        prog = run_root_function(kernel, *args, create_new_logical_tensor=True)
    prog =ir.Pass.infer_multi_iter_overlap()(prog)
    attr_dict = {}
    for name, func in sorted(prog.functions.items()):
        if "_hiddenfunc" in name:
            attrs = func.DumpAttrs()
            attr_dict[name] = attrs
    return attr_dict


def _kernel_disjoint(src, out):
    """Assemble offset [i*tile, 0]: windows disjoint across iterations."""
    for i in pypto.loop(2):
        pypto.set_vec_tile_shapes(TILE, TILE)
        v = pypto.view(src, [TILE, HIDDEN], [0, 0])
        pypto.assemble(v, [i * TILE, 0], out)


def test_overlap_disjoint_marked():
    src = pypto.Tensor([TILE, HIDDEN], pypto.DT_FP32)
    out = pypto.Tensor([2 * TILE, HIDDEN], pypto.DT_FP32)
    expected = {
        '_kernel_disjoint_loop_idx_6_Unroll1_PATH0_hiddenfunc_4': {
            'MultiIterNoOverlap': '[5]',
            'RequiresSimt': 'false'
        }
    }
    assert _run_overlap_pass(_kernel_disjoint, src, out) == expected
