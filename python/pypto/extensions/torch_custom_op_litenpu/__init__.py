# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""pypto.extensions.torch_custom_op_litenpu — pack a pypto kernel into a framework custom-op node.

The ``_litenpu`` suffix names the deploy path, not a soc check inside the package: nothing here
inspects ``soc_version`` — it is accepted and forwarded verbatim to ``pypto.set_codegen_options``.
The constraint lives in pypto core, where ``frontend/parser/entry.py`` requires exactly one compiled
kernel object, and LiteNPU is the soc family structurally guaranteed to produce one; a partitioned
(cloud-soc) kernel emits several and cannot deploy through this path.

This package holds ONLY what concerns the pypto kernel SOURCE: authoring constants (common.authoring),
source-manipulation utils (common.source_utils), and the runtime compile contract (common.compile).

Everything downstream of the kernel source — whole-graph orchestration and the build tooling that turns
an exported artifact into a deployable op library — lives with the consuming application, which
reconstructs the kernel from the node's embedded snippet. This package describes only the kernel
source it packs.
"""
# Load order matters: common.compile carries no subpackage dependency, so it loads fully first.
from .common.compile import *  # noqa: F403, I001 - keep the load order described above
