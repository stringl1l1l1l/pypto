#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
__all__ = [
    "IRBuilder",
    "IRNode",
    "InsertPoint",
    "Span",
    "Type",
    "Expr",
    "Stmt",
    "Var",
    "MemRef",
    "Function",
    "Program",
    "LogicalTensor",
    "Pass",
    "IRVerifier",
    "UnknownType",
    "ScalarType",
    "TensorType",
    "TupleType",
    "PtrType",
    "TokenKind",
    "TokenType",
    "NoneType",
    "LogicalTensorType",
    "DataType",
    "FunctionType",
    "TensorLayout",
    "MemorySpace",
    "PipeType",
    "CoreType",
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
    "MaskPattern",
    "MergeMode",
    "ReduceMode",
    "CompareMode",
    "CmpMode",
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
    "IndexOrder",
    "MemBarMode",
    "ConstInt",
    "ConstFloat",
    "ConstBool",
    "Call",
    "MakeTuple",
    "GetItemExpr",
    "TupleGetItem",
    # BinaryExpr
    "Add",
    "Sub",
    "Mul",
    "FloorDiv",
    "FloorMod",
    "FloatDiv",
    "Min",
    "Max",
    "Pow",
    "Eq",
    "Ne",
    "Lt",
    "Le",
    "Gt",
    "Ge",
    "And",
    "Or",
    "Xor",
    "BitAnd",
    "BitOr",
    "BitXor",
    "BitShiftLeft",
    "BitShiftRight",
    # UnaryExpr
    "Abs",
    "Neg",
    "Not",
    "BitNot",
    "Cast",
    # stmt
    "IterArg",
    "AssignStmt",
    "IfStmt",
    "YieldStmt",
    "ReturnStmt",
    "ForStmt",
    "WhileStmt",
    "SeqStmts",
    "EvalStmt",
    "BreakStmt",
    "ContinueStmt",
    "TensorOpStmt",
    "type_equal",
]

# --- Type classes ---
from .pypto_impl import LogicalTensor

# --- Data types ---
# --- Enums ---
# --- Expression base & leaf classes ---
# --- Binary expression ops ---
# --- Unary expression ops ---
# --- Statement classes ---
# --- Function / Program ---
from .pypto_impl.ir import (
    Abs,
    AccPhase,
    AccToVecMode,
    Add,
    And,
    AssignStmt,
    AtomicType,
    BinType,
    BitAnd,
    BitNot,
    BitOr,
    BitShiftLeft,
    BitShiftRight,
    BitXor,
    BreakStmt,
    CacheLine,
    Call,
    Cast,
    CastLayout,
    CmpMode,
    CompareMode,
    ConstBool,
    ConstFloat,
    ConstInt,
    ContinueStmt,
    CoreType,
    CrossCoreSyncMode,
    DataCopyMode,
    DataType,
    DcciDst,
    DuplicatePos,
    Eq,
    EvalStmt,
    Expr,
    FloatDiv,
    FloorDiv,
    FloorMod,
    ForStmt,
    Function,
    FunctionType,
    Ge,
    GetItemExpr,
    Gt,
    HistType,
    IfStmt,
    IndexOrder,
    InsertPoint,
    IRBuilder,
    IRNode,
    IRVerifier,
    IterArg,
    Le,
    LoadDist,
    LogicalTensorType,
    Lt,
    MakeTuple,
    MaskPattern,
    MaskWidth,
    Max,
    MemBarMode,
    MemorySpace,
    MemRef,
    MergeMode,
    Min,
    Mul,
    Ne,
    Neg,
    NoneType,
    Not,
    Or,
    PackPart,
    Pass,
    PipeType,
    Pow,
    Program,
    PtrType,
    QuantMode,
    ReduceMode,
    ReluPreMode,
    ReturnStmt,
    RoundMode,
    SaturateMode,
    SaturationFlagMode,
    ScalarType,
    SeqStmts,
    Span,
    SqueezeMode,
    Stmt,
    StoreDist,
    STPhase,
    Sub,
    SyncCoreType,
    TensorLayout,
    TensorOpStmt,
    TensorType,
    TokenKind,
    TokenType,
    TupleType,
    Type,
    UnknownType,
    Var,
    VFRoundMode,
    WhileStmt,
    Xor,
    YieldStmt,
    type_equal,
)


def tuple_get_item(value, index, span):
    if isinstance(index, int):
        index = ConstInt(index, INDEX, span)
    return GetItemExpr(value, index, span)


TupleGetItem = tuple_get_item

# --- DataType static instances ---
BOOL = DataType.BOOL
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
FP4 = DataType.FP4
FP8E4M3FN = DataType.FP8E4M3FN
FP8E5M2 = DataType.FP8E5M2
FP8E8M0 = DataType.FP8E8M0
FP4E2M1 = DataType.FP4E2M1
FP4E1M2 = DataType.FP4E1M2
FP16 = DataType.FP16
FP32 = DataType.FP32
FP64 = DataType.FP64
BF16 = DataType.BF16
FP64 = DataType.FP64
HF4 = DataType.HF4
HF8 = DataType.HF8
INDEX = DataType.INDEX
