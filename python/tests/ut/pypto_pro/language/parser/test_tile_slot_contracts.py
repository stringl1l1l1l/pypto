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
"""Slot provenance protects dynamic aliases and every MX plane address."""

from pypto_pro import ir
from pypto_pro._errors import InvalidVal
import pypto_pro.language as pl
from pypto_pro.runtime.platform import NpuArch
import pytest

from pypto.pypto_impl.codegen import CCECodegen


def _parse(kernel):
    return kernel.to_kernel_def().parse_target_program(ir.SectionKind.Cube)


def _reinterpret_kernel(selection, depth=2):
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(index: pl.DT_INT32):
        tt = pl.TileType(shape=[16, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        group = pl.make_tile_group(type=tt, addrs=0, depth=depth)
        with pl.section_cube():
            if selection == 0:
                tile = group.next()
                alias = tile
                view = pl.reinterpret(alias, shape=[16, 16])
            elif selection == 1:
                view = pl.reinterpret(group[index], shape=[16, 16])
            elif selection == 2:
                view = pl.reinterpret(group[1], shape=[16, 16])
            else:
                view = pl.reinterpret(group, shape=[16, 16])
                tile = view.next()
    return kernel


@pytest.mark.parametrize("selection", [0, 1])
def test_dynamic_slot_reinterpret_uses_runtime_address(selection):
    program, _ = _parse(_reinterpret_kernel(selection))
    code = CCECodegen(ir.SectionKind.Cube).generate_single(program, NpuArch.DAV_3510)
    assert "TASSIGN(view" in code
    assert ".data()" in code


@pytest.mark.parametrize("selection,depth", [(0, 1), (2, 2), (3, 2)])
def test_static_single_slot_and_group_aliases_remain_supported(selection, depth):
    _parse(_reinterpret_kernel(selection, depth))


def _mx_kernel(plane_addrs, mode=0, plane_depth=2, data_index=0, plane_index=0,
               data_addrs=0, accumulate=False):
    @pl.jit(arch=NpuArch.DAV_3510, auto_mutex=False)
    def kernel(x: pl.Tensor[[1], pl.DT_FP32], index: pl.DT_INT32):
        at = pl.TileType(shape=[64, 64], dtype=pl.DT_FP8E4M3FN, target_memory=pl.MemorySpace.Left)
        bt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP8E4M3FN, target_memory=pl.MemorySpace.Right)
        sat = pl.TileType(shape=[64, 2], dtype=pl.DT_FP8E8M0, target_memory=pl.MemorySpace.ScaleLeft)
        sbt = pl.TileType(shape=[2, 64], dtype=pl.DT_FP8E8M0, target_memory=pl.MemorySpace.ScaleRight)
        ct = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc)
        a = pl.make_tile_group(type=at, addrs=data_addrs, depth=2)
        b = pl.make_tile_group(type=bt, addrs=0, depth=2)
        sa = pl.make_tile_group(type=sat, addrs=plane_addrs, depth=plane_depth)
        sb = pl.make_tile_group(type=sbt, addrs=[0, 256], depth=2)
        c = pl.make_tile_group(type=ct, addrs=0, depth=1)
        with pl.section_cube():
            if mode == 1:
                pl.matmul_mx(c.current(), a.current(), b.current(),
                             scale_a=sa.current(), scale_b=sb.current())
            elif mode == 2:
                pl.matmul_mx(c[0], a[data_index], b[data_index], sa[plane_index], sb[data_index])
            elif mode == 3:
                pl.matmul_mx(c[0], a[index], b[index], sa[index], sb[index])
            else:
                av = a.current()
                bv = b.current()
                sav = sa.current()
                sbv = sb.current()
                if accumulate:
                    pl.matmul_mx_acc(c.current(), c.current(), av, bv, sav, sbv)
                else:
                    pl.matmul_mx(c.current(), av, bv, sav, sbv)
    return kernel


@pytest.mark.parametrize("mode", [0, 1, 3])
def test_every_mx_plane_slot_is_checked(mode):
    _parse(_mx_kernel([0, 256], mode))
    with pytest.raises(InvalidVal) as caught:
        _parse(_mx_kernel(0, mode))
    assert "every slot of scale_a" in caught.value.message


def test_mx_plane_requires_one_slot_per_data_slot():
    with pytest.raises(InvalidVal) as caught:
        _parse(_mx_kernel([0], plane_depth=1))
    assert "every slot of scale_a" in caught.value.message


def test_static_mx_selection_does_not_validate_unused_slots():
    _parse(_mx_kernel(0, mode=2))


def test_static_mx_pair_can_use_different_group_depths_and_indices():
    _parse(_mx_kernel([256], mode=2, plane_depth=1, data_index=1))


def test_static_mx_selection_still_checks_its_selected_address():
    with pytest.raises(InvalidVal) as caught:
        _parse(_mx_kernel(0, mode=2, data_index=1, plane_index=1))
    assert "scale_a address must equal lhs_tile address >> 4" in caught.value.message


def test_fixed_mx_plane_can_serve_runtime_slots_aliasing_one_address():
    _parse(_mx_kernel([0], plane_depth=1, data_addrs=[0, 0]))


def test_mx_acc_checks_runtime_plane_layout():
    _parse(_mx_kernel([0, 256], accumulate=True))
    with pytest.raises(InvalidVal) as caught:
        _parse(_mx_kernel(0, accumulate=True))
    assert "every slot of scale_a" in caught.value.message
