# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PyPTO Language module - Type-safe DSL API for writing IR functions."""

__all__ = [
    "inline",
    "vector_function",
    "parser",
    "Tensor",
    "Ptr",
    "Scalar",
    "DYNAMIC",
    "STATIC",
    "system",
    "simt",
    "mutex",
    "TileType",
    "make_tile",
    "make_tensor",
    "make_ptr",
    "addptr",
    "struct",
    "make_tuple",
    "struct_array",
    "FunctionType",
    "MemRef",
    "MemorySpace",
    "TilePad",
    "PipeType",
    "TensorLayout",
    "ND",
    "DN",
    "NZ",
    "ZN",
    "NN",
    "ZZ",
    "Input",
    "Output",
    "ReluPreMode",
    "AtomicType",
    "STPhase",
    "AccPhase",
    "AccToVecMode",
    "RoundMode",
    "QuantMode",
    "CrossCoreSyncMode",
    "SyncCoreType",
    "CacheLine",
    "DcciDst",
    "DT_FP4",
    "DT_FP8E4M3FN",
    "DT_FP8E5M2",
    "DT_FP8E8M0",
    "DT_FP4E2M1",
    "DT_FP4E1M2",
    "DT_FP16",
    "DT_FP32",
    "DT_BF16",
    "DT_HF4",
    "DT_HF8",
    "DT_INT4",
    "DT_INT8",
    "DT_INT16",
    "DT_INT32",
    "DT_INT64",
    "DT_UINT4",
    "DT_UINT8",
    "DT_UINT16",
    "DT_UINT32",
    "DT_UINT64",
    "DT_BOOL",
    # --- DSL stubs ---
    # Data-movement block ops
    "load",
    "load_tile",
    "store",
    "store_tile",
    "init_output",
    "move",
    "insert",
    "ssbuf_store",
    "ssbuf_load",
    # Binary element-wise (unified: tile-tile + tile-scalar)
    "add",
    "sub",
    "mul",
    "div",
    "maximum",
    "minimum",
    # Bitwise element-wise
    "and_",
    "shl",
    "shr",
    "xor",
    "expands",
    # Unary element-wise
    "neg",
    "abs",
    "exp",
    "log",
    "sqrt",
    "rsqrt",
    "recip",
    "relu",
    "fillpad",
    "FillPadMode",
    # Type conversion
    "cast",
    "add_relu_cast",
    "sub_relu_cast",
    "mul_cast",
    # Compare / select
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "select",
    # Fused ops
    "add_relu",
    "sub_relu",
    "mul_add_dst",
    "fused_mul_add",
    "fused_mul_add_relu",
    "axpy",
    "partadd",
    "partmax",
    "partmin",
    "partmul",
    # Matrix
    "matmul",
    "matmul_acc",
    "matmul_mx",
    "matmul_mx_acc",
    "transpose",
    # Reductions
    "sum",
    "argmax",
    "argmin",
    "expand_max",
    "expand_min",
    "expand_mul",
    "expand_sub",
    "expand_div",
    # Gather / scatter / sort
    "gather",
    "gatherb",
    "gathermask",
    "scatter",
    "mrgsort",
    "mrgsort2",
    "sort32",
    "histogram",
    # Quantization / index / misc
    "quant",
    "dequant",
    "getval",
    "setval",
    "reinterpret",
    "set_validshape",
    "set_mask_count",
    "set_mask_norm",
    "set_vec_mask",
    "set_ctrl_spr",
    "set_saturation_flag",
    "get_ctrl_spr",
    "get_spr",
    "get_saturation_flag",
    "reset_ctrl_spr",
    "reset_mask",
    "fill_index",
    # Namespaces
    "Vf",
    # Debug
    "pto_assert",
    "printf",
    "dump_data",
    "trap",
    # Scalar
    "min",
    "max",
    "const",
    "astype",
    # Control flow
    "range",
    "section_vector",
    "section_cube",
    # Utility
    "get_block_idx",
    "get_subblock_idx",
    "get_block_num",
    "get_subblock_num",
    "make_tile_group",
    # Frontend (runtime) aliases
    "jit",
    "pipeline",
    # VF enums
    "MaskPattern",
    "MergeMode",
    "ReduceMode",
    "CompareMode",
    "DuplicatePos",
    "CastLayout",
    "VFRoundMode",
    "SaturateMode",
    "SaturationFlagMode",
    "BinType",
    "HistType",
    "SqueezeMode",
    "PackPart",
    "MaskWidth",
    "LoadDist",
    "StoreDist",
    "DataCopyMode",
    "MemBarMode",
    "IndexOrder",
]

from types import SimpleNamespace
from typing import Any

from pypto.ir import (
    AccPhase,
    AccToVecMode,
    AtomicType,
    BinType,
    CacheLine,
    CastLayout,
    CompareMode,
    CrossCoreSyncMode,
    DataCopyMode,
    DcciDst,
    DuplicatePos,
    HistType,
    IndexOrder,
    LoadDist,
    MaskPattern,
    MaskWidth,
    MemBarMode,
    MergeMode,
    PackPart,
    QuantMode,
    ReduceMode,
    ReluPreMode,
    RoundMode,
    SaturateMode,
    SaturationFlagMode,
    SqueezeMode,
    StoreDist,
    STPhase,
    SyncCoreType,
    VFRoundMode,
)
from pypto.pypto_impl.ir import (
    DataType,
    FunctionType,
    MemorySpace,
    MemRef,
    PipeType,
    TensorLayout,
    TilePad,
)
from pypto_pro.ir.op import system_ops as system
from pypto_pro.ir.op.block_ops import FillPadMode, TileType
from pypto_pro.ir.op.ptr_ops import addptr, make_ptr, make_tensor
from pypto_pro.ir.op.system_ops import mutex_lock as _mutex_lock
from pypto_pro.ir.op.system_ops import mutex_unlock as _mutex_unlock
from pypto_pro.runtime import jit, pipeline

from . import parser
from ._api import (
    abs,
    add,
    add_relu,
    add_relu_cast,
    and_,
    argmax,
    argmin,
    astype,
    axpy,
    cast,
    const,
    dequant,
    div,
    dump_data,
    eq,
    exp,
    expand_div,
    expand_max,
    expand_min,
    expand_mul,
    expand_sub,
    expands,
    fill_index,
    fillpad,
    fused_mul_add,
    fused_mul_add_relu,
    gather,
    gatherb,
    gathermask,
    ge,
    get_block_idx,
    get_block_num,
    get_ctrl_spr,
    get_saturation_flag,
    get_spr,
    get_subblock_idx,
    get_subblock_num,
    getval,
    gt,
    histogram,
    init_output,
    insert,
    le,
    load,
    load_tile,
    log,
    lt,
    make_tile,
    make_tile_group,
    matmul,
    matmul_acc,
    matmul_mx,
    matmul_mx_acc,
    max,
    maximum,
    min,
    minimum,
    move,
    mrgsort,
    mrgsort2,
    mul,
    mul_add_dst,
    mul_cast,
    ne,
    neg,
    partadd,
    partmax,
    partmin,
    partmul,
    printf,
    pto_assert,
    quant,
    range,
    recip,
    reinterpret,
    relu,
    reset_ctrl_spr,
    reset_mask,
    rsqrt,
    scatter,
    section_cube,
    section_vector,
    select,
    set_ctrl_spr,
    set_mask_count,
    set_mask_norm,
    set_saturation_flag,
    set_validshape,
    set_vec_mask,
    setval,
    shl,
    shr,
    sort32,
    sqrt,
    ssbuf_load,
    ssbuf_store,
    store,
    store_tile,
    sub,
    sub_relu,
    sub_relu_cast,
    sum,
    transpose,
    trap,
    xor,
)
from ._simt_api import Simt as simt  # noqa: N813
from ._vf_api import Vf
from .parser.decorator import inline, vector_function
from .typing import DYNAMIC, STATIC, Input, Output, Ptr, Scalar, Tensor

mutex = SimpleNamespace(mutex_lock=_mutex_lock, mutex_unlock=_mutex_unlock)

# Public frontend dtype constants follow the python/pypto DT_* spelling.
DT_FP4 = DataType.FP4
DT_FP8E4M3FN = DataType.FP8E4M3FN
DT_FP8E5M2 = DataType.FP8E5M2
DT_FP8E8M0 = DataType.FP8E8M0
DT_FP4E2M1 = DataType.FP4E2M1
DT_FP4E1M2 = DataType.FP4E1M2
DT_FP16 = DataType.FP16
DT_FP32 = DataType.FP32
DT_BF16 = DataType.BF16
DT_HF4 = DataType.HF4
DT_HF8 = DataType.HF8
DT_INT4 = DataType.INT4
DT_INT8 = DataType.INT8
DT_INT16 = DataType.INT16
DT_INT32 = DataType.INT32
DT_INT64 = DataType.INT64
DT_UINT4 = DataType.UINT4
DT_UINT8 = DataType.UINT8
DT_UINT16 = DataType.UINT16
DT_UINT32 = DataType.UINT32
DT_UINT64 = DataType.UINT64
DT_BOOL = DataType.BOOL


def struct(*args: Any, **kwargs: Any) -> Any:
    """Create a compile-time struct grouping IR expressions for attribute access.

    Two forms are accepted:
      - Legacy: ``pl.struct(field=val, ...)`` — anonymous struct, fields-only.
      - With name: ``pl.struct("StructName", field=val, ...)`` — explicit C++ struct
        type name (preferred). The first positional argument must be a string literal
        when used.
    The runtime returns a ``SimpleNamespace``; the parser detects the call form
    and lowers it through the new ``struct.create`` Expression op when invoked
    inside a kernel body.
    """
    if args:
        # Discard the leading name positionally at runtime; the parser uses the AST
        # to extract it during IR generation.
        _name = args[0]  # kept for clarity; not used at runtime
        return SimpleNamespace(**kwargs)
    return SimpleNamespace(**kwargs)


def struct_array(size: int, *args: Any, **kwargs: Any) -> Any:
    """Create a struct array.

    Two forms are accepted:
      - Legacy: ``pl.struct_array(N, field=val, ...)``
      - With name: ``pl.struct_array(N, "StructName", field=val, ...)``
    """
    return [SimpleNamespace(**kwargs) for _ in range(size)]


def make_tuple(**kwargs: Any) -> Any:
    """Create a named-tuple aggregate of IR expressions for attribute access.

    Unlike ``pl.struct``, ``pl.make_tuple`` aggregates IR variables under field names
    without forcing a C++ struct to be generated. The parser lowers it to a
    ``MakeTuple(elements)`` and records its field names in the parser's tuple type
    registry; subsequent ``t.field`` accesses become ``GetItem(t, index)`` and
    parser-side constant propagation resolves constant indices to the original
    elements (no struct definition emitted).
    """
    return SimpleNamespace(**kwargs)


ND = TensorLayout.ND
DN = TensorLayout.DN
NZ = TensorLayout.NZ
ZN = TensorLayout.ZN
NN = TensorLayout.NN
ZZ = TensorLayout.ZZ
