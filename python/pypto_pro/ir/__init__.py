# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
PyPTO IR module with tensor operations.

This module provides:
- Re-exports of all core IR types from pypto_pro.ir
- Organized operation namespaces (e.g., op.block.load)
- IR Builder for incremental IR construction
- Helper utilities
- Enhanced type constructors (e.g., TensorType with integer shape support)
"""

__all__ = [
    "op",
    "IRBuilder",
    "TensorType",
    "TileType",
    "python_print",
    "VerificationMode",
    "VerificationLevel",
    "PassContext",
    "ConversionContext",
    "op_conversion",
    "register_op_conversion",
    "_operators",
]

from pypto.pypto_impl import ir as _core_ir
from pypto.pypto_impl.ir import PassContext, VerificationLevel, VerificationMode

from . import _operators
from . import builder as _builder
from . import op as _op
from . import op_conversion as _conversion
from . import printer as _printer
from . import type as _types

# Re-export the public native IR API.
globals().update({name: value for name, value in vars(_core_ir).items() if not name.startswith("_")})
DataType = _core_ir.DataType

# Overlay the Python extensions after exporting the native IR API.
op = _op
IRBuilder = _builder.IRBuilder
ConversionContext = _conversion.ConversionContext
op_conversion = _conversion.op_conversion
register_op_conversion = _conversion.register_op_conversion
python_print = _printer.python_print
TensorType = _types.TensorType
TileType = _types.TileType

# Export common DataType values for convenience
FP4 = DataType.FP4
FP8E4M3FN = DataType.FP8E4M3FN
FP8E5M2 = DataType.FP8E5M2
FP8E8M0 = DataType.FP8E8M0
FP4E2M1 = DataType.FP4E2M1
FP4E1M2 = DataType.FP4E1M2
FP16 = DataType.FP16
FP32 = DataType.FP32
BF16 = DataType.BF16
HF4 = DataType.HF4
HF8 = DataType.HF8
INT4 = DataType.INT4
INT8 = DataType.INT8
INT16 = DataType.INT16
INT32 = DataType.INT32
INT64 = DataType.INT64
UINT4 = DataType.UINT4
UINT8 = DataType.UINT8
UINT16 = DataType.UINT16
UINT32 = DataType.UINT32
UINT64 = DataType.UINT64
BOOL = DataType.BOOL
