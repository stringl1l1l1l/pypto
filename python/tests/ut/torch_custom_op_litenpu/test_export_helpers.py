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
"""Unit tests for ``pypto.extensions.torch_custom_op_litenpu.common.source_utils``."""

import textwrap

from pypto.extensions.torch_custom_op_litenpu.common.source_utils import _unwrap_decorated_func_source


def test_unwrap_decorated_func_source_strips_leading_decorator_block():
    raw = textwrap.dedent(
        '''\
        @some_decorator
        @another
        def foo(a, b):
            return a + b
        '''
    )
    out = _unwrap_decorated_func_source(raw)
    assert out.startswith("def foo(")
    assert "@some_decorator" not in out
    assert "return a + b" in out

    plain = textwrap.dedent(
        '''\
        def bar():
            return 42
        '''
    )
    assert _unwrap_decorated_func_source(plain) == plain  # a plain def is returned unchanged
