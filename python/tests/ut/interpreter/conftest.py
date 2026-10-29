#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""ut/interpreter fixtures: Host pass_verify only (CPU tensors + compile, no sim / no launch)."""

import pytest
import torch

from pypto.pypto_impl import NpuArch


def pytest_configure(config):
    """Patch NPU APIs before any test module is imported.

    @pypto.frontend.jit decorators call _set_run_mode() at import time,
    which checks torch.npu.is_available(). Without this early patch, modules
    that don't specify run_mode and have ASCEND_HOME_PATH set will raise
    "NPU is not available" during import — before fixtures can intercept.
    """
    if hasattr(torch, "npu"):
        torch.npu.is_available = lambda *args, **kwargs: True
    try:
        import torch_npu

        torch_npu.npu.is_available = lambda *args, **kwargs: True
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _enable_pass_verify():
    """Enable full pass_verify for all ut/interpreter kernels."""
    import pypto

    pypto.set_verify_options(
        enable_pass_verify=True,
        pass_verify_pass_filter=["all"],
    )
    yield


@pytest.fixture(autouse=True)
def _force_950_platform(request):
    """Force Ascend950 / DAV_3510 for soc("950") cases.

    Stop at EXECUTE_GRAPH so pass_verify runs on the 950 pipeline without
    requiring a local Ascend950 CCE toolchain / binary codegen.
    """
    marker = request.node.get_closest_marker("soc")
    force_950 = marker is not None and any(str(a) == "950" for a in marker.args)
    if not force_950:
        yield
        return

    import pypto

    prev_arch = pypto.platform.npuarch
    pypto.platform.npuarch = "DAV_3510"
    # The 950 pipeline is selected by NPU_ARCH (DAV_3510); no soc_version override is needed: platform
    # specs keep the default bin (same behavior as before the platform config normalization), and there
    # is no soc_series config entry - platform inis are keyed by soc_version (chip bin) only.
    pypto.set_host_options(compile_stage=pypto.CompStage.EXECUTE_GRAPH)
    yield
    pypto.platform.npuarch = prev_arch


@pytest.fixture(autouse=True)
def _host_verify_no_device_memcpy(monkeypatch):
    """Wire verify data from CPU tensors; skip CopyToHost (needs NPU context)."""
    import pypto
    import pypto.frontend.parser.entry as entry
    import pypto.pil.compile_pipeline as compile_pipeline
    import pypto.runtime as runtime

    def setup_verify_data_host(pto_tensors):
        if not pypto.get_verify_options().get("enable_pass_verify"):
            return
        host_tensors = list(pto_tensors)
        runtime._pto_verify_datas.set_keepalive_data(host_tensors)
        pypto.pypto_impl.SetVerifyData(
            runtime._pto_to_tensor_data(host_tensors),
            [],
            runtime._pto_verify_datas.get_data(),
        )

    monkeypatch.setattr(runtime, "setup_verify_data", setup_verify_data_host)
    monkeypatch.setattr(entry, "setup_verify_data", setup_verify_data_host)
    monkeypatch.setattr(compile_pipeline, "setup_verify_data", setup_verify_data_host)


@pytest.fixture(autouse=True)
def _compile_only_no_launch(monkeypatch):
    """Always compile + pass_verify; never LaunchKernelTorch / cost-model sim."""
    import pypto
    from pypto import pypto_impl
    from pypto.frontend.parser.entry import JitCallableWrapper

    def _execute_kernel_verify_only(self, torch_tensors, tensor_defs):
        pto_tensors = self._convert_tensors_with_metadata(torch_tensors, tensor_defs)
        with pypto.options("jit_scope"):
            self._set_config_option()
            # Re-apply after jit option merge so the 950 stage sticks.
            if pypto.platform.npuarch == NpuArch.DAV_3510.name:
                pypto.set_host_options(compile_stage=pypto.CompStage.EXECUTE_GRAPH)
            pypto_impl.DeviceInit()
            self.compile(pto_tensors)

    monkeypatch.setattr(JitCallableWrapper, "_execute_kernel", _execute_kernel_verify_only)


@pytest.fixture(autouse=True)
def _cpu_friendly_npu_apis(monkeypatch):
    """Let vendored wrappers that call .npu() / synchronize keep tensors on CPU."""
    import torch

    monkeypatch.setattr(torch.Tensor, "npu", lambda self, *args, **kwargs: self, raising=False)
    if hasattr(torch, "npu"):
        monkeypatch.setattr(torch.npu, "set_device", lambda *args, **kwargs: None, raising=False)
        monkeypatch.setattr(torch.npu, "synchronize", lambda *args, **kwargs: None, raising=False)
        monkeypatch.setattr(torch.npu, "is_available", lambda *args, **kwargs: True, raising=False)
    try:
        import torch_npu

        monkeypatch.setattr(torch_npu.npu, "set_device", lambda *args, **kwargs: None, raising=False)
        monkeypatch.setattr(torch_npu.npu, "synchronize", lambda *args, **kwargs: None, raising=False)
        monkeypatch.setattr(torch_npu.npu, "is_available", lambda *args, **kwargs: True, raising=False)
    except Exception:
        pass


@pytest.fixture
def cpu_device():
    """Canonical device for ut/interpreter tensors (CPU only)."""
    return "cpu"
