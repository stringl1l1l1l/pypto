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
"""Pure-Python unit tests for the launch argument ABI.

These drive the real launch closure built by ``_build_launch_entry`` -- the only path a
kernel launch takes -- with the shared library faked out, and assert on the ctypes values
handed to ``call_kernel``. No hardware or external tools are required.
"""

import ctypes
import importlib
import logging
import sys
from unittest.mock import MagicMock

from pypto_pro._errors import InvalidShape
import pytest

# ---------------------------------------------------------------------------
# Mock torch before any pypto_pro imports so jit.py's top-level 'import torch'
# resolves to our MagicMock.  torch.Tensor is replaced with a concrete class
# so that isinstance() checks inside the launch closure work correctly.
# ---------------------------------------------------------------------------
_torch_mock = MagicMock()


class _MockTensor:
    """Minimal stand-in for torch.Tensor used in validation tests."""

    def __init__(self, shape: list[int], dtype):
        self.shape = shape
        self.dtype = dtype

    @staticmethod
    def data_ptr() -> int:
        return 0


# Assign the concrete class so isinstance(x, torch.Tensor) works in jit code.
_torch_mock.Tensor = _MockTensor

DataType = importlib.import_module("pypto_pro").DataType
_jit = importlib.import_module("pypto_pro.runtime.jit")
ParamSpec = _jit.ParamSpec
ParamKind = _jit.ParamKind
CompiledKernel = _jit.CompiledKernel
_build_launch_entry = getattr(_jit, "_build_launch_entry")
_collect_dyn_vars = getattr(_jit, "_collect_dyn_vars")
_pack_factor = getattr(_jit, "_pack_factor")
torch = importlib.import_module("torch")


@pytest.fixture(autouse=True)
def _mock_torch():
    """Inject mock torch into sys.modules for the duration of each test only."""
    _jit_mod = importlib.import_module("pypto_pro.runtime.jit")
    _torch_mock.npu.get_stream_limit.reset_mock(side_effect=True, return_value=True)
    _torch_mock.npu.get_stream_limit.side_effect = AssertionError("The native launcher owns resource queries")

    _saved_torch = sys.modules.get("torch")
    _saved_npu = sys.modules.get("torch.npu")
    _saved_jit_torch = _jit_mod.torch
    sys.modules["torch"] = _torch_mock
    sys.modules["torch.npu"] = MagicMock()
    _jit_mod.torch = _torch_mock
    yield
    if _saved_torch is None:
        sys.modules.pop("torch", None)
    else:
        sys.modules["torch"] = _saved_torch
    if _saved_npu is None:
        sys.modules.pop("torch.npu", None)
    else:
        sys.modules["torch.npu"] = _saved_npu
    _jit_mod.torch = _saved_jit_torch


# ---------------------------------------------------------------------------
# Harness: build the real launch closure against a fake .so
# ---------------------------------------------------------------------------


class _FakeCallKernel:
    """Stands in for the CDLL's call_kernel: records calls, accepts argtypes/restype."""

    def __init__(self):
        self.calls = []
        self.argtypes = None
        self.restype = "unset"
        self.result = None

    def __call__(self, *args):
        self.calls.append(args)
        return args[0] if self.result is None else self.result


class _FakeLib:
    """Stands in for the CDLL handle."""

    def __init__(self):
        self.call_kernel = _FakeCallKernel()


class _CtypesProxy:
    """Real ctypes except that CDLL() hands back the fake library."""

    def __init__(self, lib):
        self._lib = lib

    def CDLL(self, _path):  # noqa: N802 - mirrors the ctypes API name
        return self._lib

    def __getattr__(self, name):
        return getattr(ctypes, name)


_FAKE_STREAM = MagicMock()
_FAKE_STREAM._as_parameter_ = 0
_BLOCK_DIM = 8


@pytest.fixture
def launch(monkeypatch):
    """Return ``launch(specs, args) -> [(declared ctype, value passed), ...]``.

    The launch path hands ctypes plain Python values and lets the declared ``argtypes``
    convert them, so a test has to check both halves: the type call_kernel was declared
    with and the value handed to it. The returned list is the ABI argument vector -- one
    entry per parameter followed by the dynamic-dimension tail -- with ``block_dim`` and
    the stream handle stripped off.
    """

    def _launch(specs, args):
        lib = _FakeLib()
        monkeypatch.setattr(_jit, "ctypes", _CtypesProxy(lib))
        entry = _build_launch_entry(CompiledKernel(lib_path="<fake>", param_specs=list(specs)))
        entry(tuple(args), _BLOCK_DIM, _FAKE_STREAM)
        fn = lib.call_kernel
        block_dim, _stream, *values = fn.calls[-1]
        assert block_dim == _BLOCK_DIM
        assert fn.restype is ctypes.c_int64, "negative native errors need all 64 result bits"
        # blockDim and stream lead the C signature; the rest lines up with `values`.
        assert fn.argtypes[:2] == [ctypes.c_uint32, ctypes.c_void_p]
        declared = fn.argtypes[2:]
        assert len(declared) == len(values), (
            f"argtypes declares {len(declared)} parameters but {len(values)} were passed"
        )
        return list(zip(declared, values))

    return _launch


def _tensor_spec(name, dtype, shape) -> ParamSpec:
    return ParamSpec(name, ParamKind.TENSOR, dtype, shape)


def _ptr_spec(name, dtype) -> ParamSpec:
    return ParamSpec(name, ParamKind.PTR, dtype, None)


def _scalar_spec(name, dtype) -> ParamSpec:
    return ParamSpec(name, ParamKind.SCALAR, dtype, None)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_validate_args_count_mismatch(launch):
    """Wrong number of args raises TypeError."""
    specs = [_tensor_spec("x", DataType.FP32, [64])]
    with pytest.raises(TypeError, match="Expected 1 args, got 2"):
        launch(specs, (_MockTensor([64], torch.float32), _MockTensor([64], torch.float32)))


@pytest.mark.parametrize(
    "specs, args",
    [
        ([_tensor_spec("x", DataType.FP32, [64])], (_MockTensor([64], torch.float32),)),
        ([_tensor_spec("x", DataType.FP32, [-1, 128])], (_MockTensor([999, 128], torch.float32),)),
        ([_ptr_spec("p", DataType.FP32)], (_MockTensor([1], torch.float32),)),
        ([_scalar_spec("n", DataType.INT32)], (42,)),
        ([_scalar_spec("scale", DataType.FP32)], (1.5,)),
        ([_scalar_spec("flag", DataType.INT32)], (True,)),
        ([_scalar_spec("flag", DataType.BOOL)], (False,)),
        ([_tensor_spec("x", DataType.FP32, [-1, 64])], (_MockTensor([999, 64], torch.float32),)),
    ],
)
def test_validate_args_accepts_supported_args(launch, specs, args):
    """Supported tensor, ptr, scalar, bool and dynamic-dim args pass validation."""
    launch(specs, args)


@pytest.mark.parametrize(
    "specs, args, error",
    [
        ([_tensor_spec("x", DataType.FP32, [64])], (3.14,), "expected torch.Tensor"),
        ([_tensor_spec("x", DataType.INT32, [64])], (_MockTensor([64], torch.float32),), "dtype mismatch"),
        ([_tensor_spec("x", DataType.FP32, [64, 128])], (_MockTensor([64], torch.float32),), "rank mismatch"),
        (
            [_tensor_spec("x", DataType.FP32, [64, 128])],
            (_MockTensor([64, 256], torch.float32),),
            "dim\\[1\\] mismatch",
        ),
        ([_scalar_spec("n", DataType.INT32)], (_MockTensor([1], torch.float32),), "expected Python scalar"),
        ([_scalar_spec("n", DataType.INT32)], (1.5,), "float value passed for non-float dtype"),
        ([_scalar_spec("scale", DataType.FP32)], (42,), "int value passed for non-integer dtype"),
        (
            [_scalar_spec("scale", DataType.FP32)],
            (True,),
            "bool value passed for non-boolean/non-integer dtype",
        ),
    ],
)
def test_validate_args_rejects_invalid_args(launch, specs, args, error):
    """Invalid tensor and scalar args fail with the expected validation family."""
    with pytest.raises(TypeError, match=error):
        launch(specs, args)


def test_validate_args_mixed_params(launch):
    """Mixed tensor + scalar params all validated correctly."""
    specs = [
        _tensor_spec("input", DataType.FP32, [64]),
        _tensor_spec("output", DataType.FP32, [64]),
        _scalar_spec("n", DataType.INT32),
    ]
    args = (
        _MockTensor([64], torch.float32),
        _MockTensor([64], torch.float32),
        128,
    )
    launch(specs, args)  # must not raise


def test_unmappable_scalar_dtype_rejected_at_bind_time():
    """A scalar dtype with no ctypes mapping fails when the plan is built, not per launch."""
    with pytest.raises(TypeError, match="No ctypes mapping for scalar dtype"):
        _jit._LaunchPlan([_scalar_spec("h", DataType.FP16)])


@pytest.mark.parametrize(
    "specs, args",
    [
        # Zero in a dynamic dimension.
        ([_tensor_spec("a", DataType.FP32, ["M", 64])], (_MockTensor([0, 64], torch.float32),)),
        # Zero in a dimension the ParamSpec does not constrain at all.
        ([_tensor_spec("a", DataType.FP32, [-1, 64])], (_MockTensor([0, 64], torch.float32),)),
        # Zero contributed by a parameter that does not introduce the dynamic name.
        (
            [
                _tensor_spec("a", DataType.FP32, ["M", 128]),
                _tensor_spec("b", DataType.FP32, ["M", 64]),
            ],
            (_MockTensor([0, 128], torch.float32), _MockTensor([0, 64], torch.float32)),
        ),
    ],
)
def test_non_positive_dimension_rejected(launch, specs, args):
    """A zero-sized dimension is rejected on every launch, not only the one that compiled.

    The shape-policy bind enforces this on the compile path; a cache hit skips the bind, so
    the launch closure has to carry the same rule or the two disagree after the first call.
    """
    with pytest.raises(ValueError, match="runtime dimensions must be positive"):
        launch(specs, args)


def test_dyn_tail_order_must_match_codegen(monkeypatch):
    """_LaunchPlan refuses to bind if its ABI tail order disagrees with the C++ signature.

    Both orders are derived from the same ParamSpecs by separate code, so they can only
    drift through an edit. Simulating that drift is the only way to reach the guard.
    """
    specs = [
        _tensor_spec("a", DataType.FP32, ["M", "N"]),
        _tensor_spec("b", DataType.FP32, ["N", "K"]),
    ]
    _jit._LaunchPlan(specs)  # agrees today

    monkeypatch.setattr(_jit, "_collect_dyn_vars", lambda _specs: ["N", "M", "K"])
    with pytest.raises(RuntimeError, match="disagrees with the generated kernel signature"):
        _jit._LaunchPlan(specs)


# ---------------------------------------------------------------------------
# ABI conversion
# ---------------------------------------------------------------------------


def test_argtypes_match_the_generated_c_signature(launch):
    """What ctypes is told must match what the .so was compiled with.

    ``_entry_params_from_param_specs`` builds the extern "C" parameter list the caller
    wrapper is generated from, so it is the authority on the callee's signature. Passing
    plain Python values only works while the declaration agrees with it -- and the tail is
    declared ``int64_t`` there, matching the launch path and the IR dimension type.
    """
    specs = [
        _tensor_spec("a", DataType.FP32, ["M", 64]),
        _scalar_spec("n", DataType.INT32),
        _ptr_spec("p", DataType.FP32),
        _tensor_spec("b", DataType.FP32, ["M", "N"]),
    ]
    args = (
        _MockTensor([8, 64], torch.float32),
        3,
        _MockTensor([1], torch.float32),
        _MockTensor([8, 16], torch.float32),
    )
    result = launch(specs, args)
    c_params = _jit._entry_params_from_param_specs(specs)

    assert len(result) == len(c_params)
    for (declared, _value), (_c_type, _name, is_ptr) in zip(result, c_params):
        assert (declared is ctypes.c_void_p) == is_ptr
    # The two dynamic names, M and N, close the vector as int64_t.
    assert [c_type for c_type, _name, _is_ptr in c_params[-2:]] == ["int64_t", "int64_t"]
    assert [declared for declared, _value in result[-2:]] == [ctypes.c_int64, ctypes.c_int64]


def test_args_to_ctypes_tensor(launch):
    """torch.Tensor → data_ptr() passed under a declared c_void_p."""
    specs = [_tensor_spec("x", DataType.FP32, [64])]
    tensor = _MockTensor([64], torch.float32)
    result = launch(specs, (tensor,))
    assert result == [(ctypes.c_void_p, tensor.data_ptr())]


@pytest.mark.parametrize(
    "spec, arg, ctype, expected",
    [
        (_ptr_spec("p", DataType.FP32), _MockTensor([1], torch.float32), ctypes.c_void_p, None),
        (_scalar_spec("scale", DataType.FP32), 1.5, ctypes.c_float, 1.5),
        (_scalar_spec("n", DataType.INT32), 42, ctypes.c_int32, 42),
    ],
)
def test_args_to_ctypes_single_arg(launch, spec, arg, ctype, expected):
    """ptr and scalar params are declared with the expected C type."""
    result = launch([spec], (arg,))
    assert len(result) == 1
    declared, value = result[0]
    assert declared is ctype
    if expected is not None:
        assert value == pytest.approx(expected)


def test_args_to_ctypes_mixed(launch):
    """Mixed tensor + scalar → correct ctypes sequence."""
    specs = [
        _tensor_spec("input", DataType.FP32, [64]),
        _scalar_spec("n", DataType.INT32),
    ]
    tensor = _MockTensor([64], torch.float32)
    result = launch(specs, (tensor, 7))
    assert result[0][0] is ctypes.c_void_p
    assert result[1] == (ctypes.c_int32, 7)


def test_none_tensor_arg_is_null_pointer(launch):
    """An omitted optional tensor becomes a null pointer and contributes zeros to the tail.

    Also pins that the positivity rule does not fire here: those zeros are legitimate.
    """
    specs = [_tensor_spec("a", DataType.FP32, ["M", "N"])]
    result = launch(specs, (None,))
    assert len(result) == 3
    assert result[0] == (ctypes.c_void_p, None)
    assert result[1:] == [(ctypes.c_int64, 0), (ctypes.c_int64, 0)]


# ---------------------------------------------------------------------------
# _collect_dyn_vars
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "specs, expected",
    [
        (
            [
                _tensor_spec("a", DataType.FP32, ["M", "N"]),
                _tensor_spec("b", DataType.FP32, ["M", "N"]),
            ],
            ["M", "N"],
        ),
        (
            [
                _tensor_spec("a", DataType.FP32, ["M", "N"]),
                _tensor_spec("b", DataType.FP32, ["N", "K"]),
            ],
            ["M", "N", "K"],
        ),
        (
            [
                _tensor_spec("a", DataType.FP32, [64, 128]),
                _tensor_spec("b", DataType.FP32, [128]),
            ],
            [],
        ),
    ],
)
def test_collect_dyn_vars(specs, expected):
    """Dynamic vars are deduped in appearance order."""
    assert _collect_dyn_vars(specs) == expected


# ---------------------------------------------------------------------------
# Dynamic shape variables
# ---------------------------------------------------------------------------


def test_validate_args_dyn_consistent(launch):
    """M=64 in both tensors — consistent, must not raise."""
    specs = [
        _tensor_spec("a", DataType.FP32, ["M", 128]),
        _tensor_spec("b", DataType.FP32, ["M", 64]),
    ]
    args = (
        _MockTensor([64, 128], torch.float32),
        _MockTensor([64, 64], torch.float32),
    )
    launch(specs, args)  # must not raise


def test_validate_args_dyn_inconsistent(launch):
    """M=64 in tensor a but M=128 in tensor b — must raise TypeError mentioning 'M'."""
    specs = [
        _tensor_spec("a", DataType.FP32, ["M", 128]),
        _tensor_spec("b", DataType.FP32, ["M", 64]),
    ]
    args = (
        _MockTensor([64, 128], torch.float32),
        _MockTensor([128, 64], torch.float32),
    )
    with pytest.raises(InvalidShape, match="M"):
        launch(specs, args)


def test_args_to_ctypes_dyn_appended(launch):
    """Tensor with shape ["M", 64]: result should have c_void_p then c_int64(M_value)."""
    m_var = 32
    specs = [_tensor_spec("a", DataType.FP32, ["M", 64])]
    tensor = _MockTensor([m_var, 64], torch.float32)
    result = launch(specs, (tensor,))
    assert len(result) == 2
    assert result[0][0] is ctypes.c_void_p
    assert result[1] == (ctypes.c_int64, m_var)


def test_args_to_ctypes_dyn_dimension_exceeds_int32(launch):
    """Dynamic dimensions must not narrow when crossing the JIT ABI."""
    dimension = 2**31 + 1
    specs = [_tensor_spec("a", DataType.FP32, ["M"])]
    result = launch(specs, (_MockTensor([dimension], torch.float32),))
    assert result[-1] == (ctypes.c_int64, dimension)


def test_args_to_ctypes_dyn_product_exceeds_int32(launch):
    """Each dimension remains 64-bit when their product exceeds INT32_MAX."""
    dimension = 65536
    specs = [_tensor_spec("a", DataType.FP32, ["M", "N"])]
    result = launch(specs, (_MockTensor([dimension, dimension], torch.float32),))
    assert result[-2:] == [(ctypes.c_int64, dimension), (ctypes.c_int64, dimension)]


def test_args_to_ctypes_dyn_order(launch):
    """Tensor a(M,N) and tensor b(N,K): appended order must be M, N, K."""
    specs = [
        _tensor_spec("a", DataType.FP32, ["M", "N"]),
        _tensor_spec("b", DataType.FP32, ["N", "K"]),
    ]
    args = (
        _MockTensor([10, 20], torch.float32),
        _MockTensor([20, 30], torch.float32),
    )
    result = launch(specs, args)
    assert len(result) == 5
    assert result[2:] == [(ctypes.c_int64, 10), (ctypes.c_int64, 20), (ctypes.c_int64, 30)]  # M, N, K


def test_args_to_ctypes_dyn_dedup(launch):
    """Two tensors sharing M,N: each var appended exactly once."""
    specs = [
        _tensor_spec("a", DataType.FP32, ["M", "N"]),
        _tensor_spec("b", DataType.FP32, ["M", "N"]),
    ]
    args = (
        _MockTensor([8, 16], torch.float32),
        _MockTensor([8, 16], torch.float32),
    )
    result = launch(specs, args)
    assert len(result) == 4
    assert result[2:] == [(ctypes.c_int64, 8), (ctypes.c_int64, 16)]  # M, N


def test_args_to_ctypes_no_dyn(launch):
    """No dynamic dims: result length equals number of args."""
    specs = [
        _tensor_spec("a", DataType.FP32, [64, 128]),
        _tensor_spec("b", DataType.FP32, [128]),
    ]
    args = (
        _MockTensor([64, 128], torch.float32),
        _MockTensor([128], torch.float32),
    )
    result = launch(specs, args)
    assert len(result) == len(args)


# ---------------------------------------------------------------------------
# Sub-byte dtypes (packed storage)
# ---------------------------------------------------------------------------
#
# A sub-byte dtype has no torch dtype of its own, so the caller hands over a uint8
# buffer holding two fp4 per byte and torch reports the innermost axis in those
# storage units. ParamSpec and the kernel ABI count elements, so the launch plan
# resolves a packing factor per parameter and the launch path scales that one axis.


@pytest.mark.parametrize(
    "dtype, expected",
    [
        (DataType.FP4E2M1, 2),
        (DataType.FP4, 2),
        (DataType.UINT8, 1),
        (DataType.HF8, 1),
        # 8 // 32 is 0, not 1: without the clamp a wide dtype would scale its innermost
        # dimension to zero on every launch.
        (DataType.FP32, 1),
        (DataType.INT64, 1),
        (None, 1),
    ],
)
def test_pack_factor(dtype, expected):
    """Only sub-byte dtypes pack; everything else, including a missing dtype, is 1:1."""
    assert _pack_factor(dtype) == expected


def test_packed_const_dim_compared_in_elements(launch):
    """A [64, 64] fp4 param is satisfied by a [64, 32] uint8 buffer."""
    specs = [_tensor_spec("x", DataType.FP4E2M1, [64, 64])]
    launch(specs, (_MockTensor([64, 32], torch.uint8),))  # must not raise


def test_packed_const_dim_mismatch_reports_elements(launch):
    """Passing the logical extent as the storage extent is a mismatch, reported in elements."""
    specs = [_tensor_spec("x", DataType.FP4E2M1, [64, 64])]
    with pytest.raises(TypeError, match=r"dim\[1\] mismatch — expected 64, got 128"):
        launch(specs, (_MockTensor([64, 64], torch.uint8),))


def test_packed_dyn_dim_scaled_in_abi_tail(launch):
    """The dynamic tail carries the element count, not the packed storage count."""
    specs = [_tensor_spec("x", DataType.FP4E2M1, [64, "K"])]
    result = launch(specs, (_MockTensor([64, 32], torch.uint8),))
    assert result[-1] == (ctypes.c_int64, 64)


def test_packed_outer_dims_are_not_scaled(launch):
    """Only the innermost axis is packed; an outer dynamic dim passes through unchanged."""
    specs = [_tensor_spec("x", DataType.FP4E2M1, ["M", 64])]
    result = launch(specs, (_MockTensor([7, 32], torch.uint8),))
    assert result[-1] == (ctypes.c_int64, 7)


def test_packed_dyn_dim_agrees_with_unpacked_param(launch):
    """K is 64 elements in both the fp4 buffer and the fp32 one."""
    specs = [
        _tensor_spec("a", DataType.FP4E2M1, ["M", "K"]),
        _tensor_spec("b", DataType.FP32, ["K", 8]),
    ]
    args = (
        _MockTensor([4, 32], torch.uint8),   # logical [4, 64]
        _MockTensor([64, 8], torch.float32),
    )
    result = launch(specs, args)
    assert result[2:] == [(ctypes.c_int64, 4), (ctypes.c_int64, 64)]  # M, K


def test_packed_dyn_dim_mismatch_with_unpacked_param(launch):
    """The cross-parameter check compares elements: 64 fp4 against 32 fp32 must raise."""
    specs = [
        _tensor_spec("a", DataType.FP4E2M1, ["M", "K"]),
        _tensor_spec("b", DataType.FP32, ["K", 8]),
    ]
    args = (
        _MockTensor([4, 32], torch.uint8),   # logical [4, 64]
        _MockTensor([32, 8], torch.float32),
    )
    with pytest.raises(InvalidShape, match="K"):
        launch(specs, args)


def test_cached_launch_passes_actual_stream_without_python_query(monkeypatch):
    """Binding and cached launches must use the native topology without consulting Python compile configuration."""
    lib = _FakeLib()
    monkeypatch.setattr(_jit, "ctypes", _CtypesProxy(lib))
    config_lookup = MagicMock(side_effect=AssertionError("Core usage belongs only in generated C++"))
    monkeypatch.setattr(_jit, "get_jit_compile_config", config_lookup)
    streams = [MagicMock(), MagicMock(), _FAKE_STREAM]
    _torch_mock.npu.current_stream.side_effect = streams[:2]
    target = _jit.KernelTarget("test-compiled-target", 1, 1, True)
    compiled = CompiledKernel(lib_path="<fake>", param_specs=[], target=target)
    dump = importlib.import_module("pypto_pro.runtime.exception_dump")
    set_dump_info = MagicMock()
    monkeypatch.setattr(dump, "set_dump_info", set_dump_info)
    try:
        _jit._launch(compiled, (), 12, None)
        entry = compiled.entry
        _jit._launch(compiled, (), 12, None)
        _jit._launch(compiled, (), 12, streams[2])
        assert compiled.entry is entry
        assert [call[0] for call in lib.call_kernel.calls] == [12, 12, 12]
        assert [call[1] for call in lib.call_kernel.calls] == [s._as_parameter_ for s in streams]
        _torch_mock.npu.get_stream_limit.assert_not_called()
        config_lookup.assert_not_called()
        assert len(set_dump_info.call_args_list) == 3
        assert all(call.args[0] is compiled for call in set_dump_info.call_args_list)
    finally:
        _torch_mock.npu.current_stream.side_effect = None


@pytest.mark.parametrize("requested", [1 << 32, (1 << 32) + 1, 1 << 100])
def test_large_request_saturates_before_native_abi_conversion(monkeypatch, requested):
    """A valid Python upper bound must not wrap to zero or one before the native resource clamp."""
    lib = _FakeLib()
    monkeypatch.setattr(_jit, "ctypes", _CtypesProxy(lib))
    _jit._launch(CompiledKernel(lib_path="<fake>", param_specs=[]), (), requested, _FAKE_STREAM)
    assert lib.call_kernel.calls[-1][0] == 0xFFFFFFFF


@pytest.mark.parametrize(
    "kind,detail",
    [
        (1, 0),
        (2, 0),
        (2, 1),
        (3, 507000),
        (4, 0xFFFFFFFF),
        (5, 4),
        (9, 1),
    ],
)
def test_launch_propagates_native_errors(monkeypatch, kind, detail):
    """All native errors raise without querying or interpreting the compiler's core topology."""
    lib = _FakeLib()
    lib.call_kernel.result = -((kind << 32) | detail)
    monkeypatch.setattr(_jit, "ctypes", _CtypesProxy(lib))
    config_lookup = MagicMock(side_effect=AssertionError("Reporting a native error must not need core usage"))
    monkeypatch.setattr(_jit, "get_jit_compile_config", config_lookup)
    compiled = CompiledKernel(lib_path="<fake>", param_specs=[], has_cube=True, has_vector=True)
    with pytest.raises(RuntimeError, match=f"Kernel launch failed with error code {lib.call_kernel.result}$"):
        _jit._launch(compiled, (), 8, _FAKE_STREAM)
    assert lib.call_kernel.restype is ctypes.c_int64
    _torch_mock.npu.get_stream_limit.assert_not_called()
    config_lookup.assert_not_called()


def test_direct_call_launches_with_auto_sentinel(monkeypatch):
    """A direct call must hand the native launcher 0, which resolves to the stream's full budget."""
    lib = _FakeLib()
    monkeypatch.setattr(_jit, "ctypes", _CtypesProxy(lib))
    compiled = CompiledKernel(lib_path="<fake>", param_specs=[])
    monkeypatch.setattr(_jit, "_launch", MagicMock())
    kernel = _jit._TileJitKernel(lambda: None, arch="a5")
    monkeypatch.setattr(kernel, "_validate_launch_arch", lambda: None)
    monkeypatch.setattr(kernel, "_ensure_compiled", lambda *_args, **_kwargs: compiled)
    kernel()
    _jit._launch.assert_called_once()
    assert _jit._launch.call_args.args[2] == 0


def test_capture_log_uses_native_block_count(monkeypatch, caplog):
    """Logging the host request would misreport captured nodes after the native launcher clamps them."""
    lib = _FakeLib()
    lib.call_kernel.result = 3
    monkeypatch.setattr(_jit, "ctypes", _CtypesProxy(lib))
    npu_module = MagicMock()
    npu_module.npu.is_current_stream_capturing.return_value = True
    monkeypatch.setitem(sys.modules, "torch_npu", npu_module)
    with caplog.at_level(logging.INFO):
        _jit._launch(CompiledKernel(lib_path="<fake>", param_specs=[]), (), 12, _FAKE_STREAM)
    assert "graph capture mode (block_dim=3)" in caplog.text
    assert "block_dim=12" not in caplog.text


def test_explicit_stream_validated_once_at_launcher_creation(monkeypatch):
    """A foreign stream can expose a pointer too; reject it before ACL dereferences that handle."""
    class NPUStream:
        _as_parameter_ = 0x1234

    class ForeignStream:
        _as_parameter_ = 0xDEAD

    monkeypatch.setattr(_torch_mock.npu, "Stream", NPUStream)
    kernel = _jit._TileJitKernel(lambda: None, arch="a5")
    with pytest.raises(TypeError, match="stream must be torch.npu.Stream or None"):
        kernel[ForeignStream(), 8]

    stream = NPUStream()
    launcher = kernel[stream, 8]
    monkeypatch.setattr(kernel, "_validate_launch_arch", lambda: None)
    monkeypatch.setattr(kernel, "_ensure_compiled", lambda *_args, **_kwargs: None)
    launches = []
    monkeypatch.setattr(_jit, "_launch", lambda _compiled, _args, blocks, target: launches.append((blocks, target)))
    # Invalidating the type only after construction detects an accidental hot-path
    # isinstance check without requiring an NPU to create the test's stream.
    monkeypatch.setattr(_torch_mock.npu, "Stream", None)
    launcher()
    launcher()
    assert launches == [(8, stream), (8, stream)]
