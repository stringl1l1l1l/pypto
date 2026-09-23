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

from textwrap import dedent

import pypto_pro.language as pl


def _compile_cube(kernel) -> str:
    from pypto_pro.runtime.jit import _assemble_cv_source, _parse_and_codegen_targets

    cube, vector = _parse_and_codegen_targets(kernel.to_kernel_def(), "3510", "")
    source = _assemble_cv_source(cube, vector).content
    return source.split("#if defined(__DAV_VEC__)")[0]


@pl.jit(auto_mutex=False)
def _sibling_if_kernel(rows: pl.DT_INT64, cols: pl.DT_INT64):
    live = 0
    if rows > 0:
        if cols > 0:
            dead = 1

    if rows > 1:
        dead = 2  # noqa: F841
        live = 1

    for i in pl.range(0, live, 1):
        pl.system.bar_all()


def _then_terminates(flag):
    if flag > 0:
        return 0
    else:
        value = 1
    return value


def _else_terminates(flag):
    if flag > 0:
        value = 1
    else:
        return 0
    return value


@pl.jit(auto_mutex=False)
def _then_terminates_kernel(flag: pl.DT_INT64):
    result = _then_terminates(flag)
    observed = result + 1  # noqa: F841


@pl.jit(auto_mutex=False)
def _else_terminates_kernel(flag: pl.DT_INT64):
    result = _else_terminates(flag)
    observed = result + 1  # noqa: F841


def _nested_return(value, carried):
    carried = carried - 1
    if value > 0:
        if value > 1:
            return


@pl.jit(auto_mutex=False)
def _nested_yield_kernel(value: pl.DT_INT64, carried: pl.DT_INT64):
    _nested_return(value, carried)
    for i in pl.range(0, value, 1):
        pl.system.bar_all()


@pl.jit(auto_mutex=False)
def _loop_induction_shadows_constant_kernel(limit: pl.DT_INT64):
    i = 5
    for i in pl.range(0, limit, 1):
        nested_limit = i + 1
        for j in pl.range(0, nested_limit, 1):
            pl.system.bar_all()


def test_sibling_if_does_not_reference_a_previous_branch_local():
    expected = dedent(
        """\
        #if defined(__DAV_CUBE__)
        #include <pypto_tprint.h>
        __aicore__ inline void _sibling_if_kernel_impl_cube(int64_t rows_0, int64_t cols_0)
        {

            auto live_0 = 0;
            auto _expr_tmp_0_0 = (rows_0 > 0);
            if (_expr_tmp_0_0) {
                auto _expr_tmp_1_0 = (cols_0 > 0);
                if (_expr_tmp_1_0) {
                    auto dead_0 = 1;
                } else {
                }
            } else {
            }
            auto _expr_tmp_2_0 = (rows_0 > 1);
            int64_t live_2;
            if (_expr_tmp_2_0) {
                auto dead_5 = 2;
                auto live_1 = 1;
                live_2 = 1;
            } else {
                live_2 = 0;
            }
            for (int64_t i__iterator_0 = 0; i__iterator_0 < live_2; i__iterator_0 += 1) {
                pipe_barrier(PIPE_ALL);
                continue;
            }
            return;
        }
        #endif

        """
    )
    assert _compile_cube(_sibling_if_kernel) == expected


def test_then_termination_uses_the_only_reachable_continuation_value():
    expected = dedent(
        """\
        #if defined(__DAV_CUBE__)
        #include <pypto_tprint.h>
        __aicore__ inline void _then_terminates_kernel_impl_cube(int64_t flag_0)
        {

            int64_t __inline_0_return_val_4;

            while (true) {
                auto __inline_0_returned_0 = false;
                auto _expr_tmp_0_0 = (flag_0 > 0);
                if (_expr_tmp_0_0) {
                    auto __inline_0_return_val_2 = 0;
                    auto __inline_0_returned_1 = true;
                    __inline_0_return_val_4 = 0;
                    break;
                } else {
                }
                auto __inline_0_value_0 = 1;
                auto __inline_0_return_val_3 = 1;
                auto __inline_0_returned_2 = true;
                __inline_0_return_val_4 = 1;
                break;
            }
            auto observed_0 = (__inline_0_return_val_4 + 1);
            return;
        }
        #endif

        """
    )
    assert _compile_cube(_then_terminates_kernel) == expected


def test_else_termination_uses_the_only_reachable_continuation_value():
    expected = dedent(
        """\
        #if defined(__DAV_CUBE__)
        #include <pypto_tprint.h>
        __aicore__ inline void _else_terminates_kernel_impl_cube(int64_t flag_0)
        {

            int64_t __inline_0_return_val_5;

            while (true) {
                auto __inline_0_returned_0 = false;
                auto _expr_tmp_0_0 = (flag_0 > 0);
                bool __inline_0_returned_2;
                int64_t __inline_0_value_1;
                if (_expr_tmp_0_0) {
                    auto __inline_0_value_0 = 1;
                    __inline_0_returned_2 = false;
                    __inline_0_value_1 = 1;
                } else {
                    auto __inline_0_return_val_2 = 0;
                    auto __inline_0_returned_1 = true;
                    __inline_0_return_val_5 = 0;
                    break;
                }
                auto __inline_0_return_val_4 = 1;
                auto __inline_0_returned_3 = true;
                __inline_0_return_val_5 = 1;
                break;
            }
            auto observed_0 = (__inline_0_return_val_5 + 1);
            return;
        }
        #endif

        """
    )
    assert _compile_cube(_else_terminates_kernel) == expected


def test_nested_if_yield_is_not_consumed_by_the_outer_loop():
    expected = dedent(
        """\
        #if defined(__DAV_CUBE__)
        #include <pypto_tprint.h>
        __aicore__ inline void _nested_yield_kernel_impl_cube(int64_t value_0, int64_t carried_0)
        {

            int64_t __inline_0_carried_1 = carried_0;
            int64_t __inline_0_carried_3;

            while (true) {
                auto __inline_0_returned_0 = false;
                auto __inline_0_carried_2 = (__inline_0_carried_1 - 1);
                auto _expr_tmp_0_0 = (value_0 > 0);
                bool __inline_0_returned_2;
                if (_expr_tmp_0_0) {
                    auto _expr_tmp_1_0 = (value_0 > 1);
                    if (_expr_tmp_1_0) {
                        auto __inline_0_returned_1 = true;
                        int64_t __inline_0_carried_3__next = __inline_0_carried_2;
                        __inline_0_carried_3 = __inline_0_carried_3__next;
                        break;
                    } else {
                    }
                    __inline_0_returned_2 = false;
                } else {
                    __inline_0_returned_2 = false;
                }
                auto __inline_0_returned_3 = true;
                int64_t __inline_0_carried_3__next = __inline_0_carried_2;
                __inline_0_carried_3 = __inline_0_carried_3__next;
                break;
            }
            for (int64_t i__iterator_0 = 0; i__iterator_0 < value_0; i__iterator_0 += 1) {
                pipe_barrier(PIPE_ALL);
                continue;
            }
            return;
        }
        #endif

        """
    )
    assert _compile_cube(_nested_yield_kernel) == expected


def test_loop_induction_variable_does_not_reuse_shadowed_constant():
    expected = dedent(
        """\
        #if defined(__DAV_CUBE__)
        #include <pypto_tprint.h>
        __aicore__ inline void _loop_induction_shadows_constant_kernel_impl_cube(int64_t limit_0)
        {

            auto i_0 = 5;
            int64_t i_1 = 5;

            for (int64_t i__iterator_0 = 0; i__iterator_0 < limit_0; i__iterator_0 += 1) {
                auto nested_limit_0 = (i__iterator_0 + 1);
                for (int64_t j__iterator_0 = 0; j__iterator_0 < nested_limit_0; j__iterator_0 += 1) {
                    pipe_barrier(PIPE_ALL);
                    continue;
                }
                i_1 = i__iterator_0;
                continue;
            }
            return;
        }
        #endif

        """
    )
    assert _compile_cube(_loop_induction_shadows_constant_kernel) == expected
