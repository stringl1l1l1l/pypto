/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PYPTO_IR_OP_ATTR_TYPES_H_
#define PYPTO_IR_OP_ATTR_TYPES_H_

#include <cstring>
#include <stdexcept>
#include <string>

/**
 * \brief Variadic FOR_EACH preprocessor utility
 *
 * PYPTO_FOR_EACH(ACTION, ctx, v1, v2, ...) -> ACTION(ctx, v1) ACTION(ctx, v2) ...
 *
 * Supports up to 24 variadic arguments.
 * The first parameter (ctx) is passed to ACTION but can be ignored if not needed.
 */

#define PYPTO_ENUM_EXPAND(x) x

#define PYPTO_ENUM_FE_1(ACTION, ctx, x) ACTION(ctx, x)
#define PYPTO_ENUM_FE_2(ACTION, ctx, x, ...) ACTION(ctx, x) PYPTO_ENUM_EXPAND(PYPTO_ENUM_FE_1(ACTION, ctx, __VA_ARGS__))
#define PYPTO_ENUM_FE_3(ACTION, ctx, x, ...) ACTION(ctx, x) PYPTO_ENUM_EXPAND(PYPTO_ENUM_FE_2(ACTION, ctx, __VA_ARGS__))
#define PYPTO_ENUM_FE_4(ACTION, ctx, x, ...) ACTION(ctx, x) PYPTO_ENUM_EXPAND(PYPTO_ENUM_FE_3(ACTION, ctx, __VA_ARGS__))
#define PYPTO_ENUM_FE_5(ACTION, ctx, x, ...) ACTION(ctx, x) PYPTO_ENUM_EXPAND(PYPTO_ENUM_FE_4(ACTION, ctx, __VA_ARGS__))
#define PYPTO_ENUM_FE_6(ACTION, ctx, x, ...) ACTION(ctx, x) PYPTO_ENUM_EXPAND(PYPTO_ENUM_FE_5(ACTION, ctx, __VA_ARGS__))
#define PYPTO_ENUM_FE_7(ACTION, ctx, x, ...) ACTION(ctx, x) PYPTO_ENUM_EXPAND(PYPTO_ENUM_FE_6(ACTION, ctx, __VA_ARGS__))
#define PYPTO_ENUM_FE_8(ACTION, ctx, x, ...) ACTION(ctx, x) PYPTO_ENUM_EXPAND(PYPTO_ENUM_FE_7(ACTION, ctx, __VA_ARGS__))
#define PYPTO_ENUM_FE_9(ACTION, ctx, x, ...) ACTION(ctx, x) PYPTO_ENUM_EXPAND(PYPTO_ENUM_FE_8(ACTION, ctx, __VA_ARGS__))
#define PYPTO_ENUM_FE_10(ACTION, ctx, x, ...) \
    ACTION(ctx, x) PYPTO_ENUM_EXPAND(PYPTO_ENUM_FE_9(ACTION, ctx, __VA_ARGS__))
#define PYPTO_ENUM_FE_11(ACTION, ctx, x, ...) \
    ACTION(ctx, x) PYPTO_ENUM_EXPAND(PYPTO_ENUM_FE_10(ACTION, ctx, __VA_ARGS__))
#define PYPTO_ENUM_FE_12(ACTION, ctx, x, ...) \
    ACTION(ctx, x) PYPTO_ENUM_EXPAND(PYPTO_ENUM_FE_11(ACTION, ctx, __VA_ARGS__))
#define PYPTO_ENUM_FE_13(ACTION, ctx, x, ...) \
    ACTION(ctx, x) PYPTO_ENUM_EXPAND(PYPTO_ENUM_FE_12(ACTION, ctx, __VA_ARGS__))
#define PYPTO_ENUM_FE_14(ACTION, ctx, x, ...) \
    ACTION(ctx, x) PYPTO_ENUM_EXPAND(PYPTO_ENUM_FE_13(ACTION, ctx, __VA_ARGS__))
#define PYPTO_ENUM_FE_15(ACTION, ctx, x, ...) \
    ACTION(ctx, x) PYPTO_ENUM_EXPAND(PYPTO_ENUM_FE_14(ACTION, ctx, __VA_ARGS__))
#define PYPTO_ENUM_FE_16(ACTION, ctx, x, ...) \
    ACTION(ctx, x) PYPTO_ENUM_EXPAND(PYPTO_ENUM_FE_15(ACTION, ctx, __VA_ARGS__))
#define PYPTO_ENUM_FE_17(ACTION, ctx, x, ...) \
    ACTION(ctx, x) PYPTO_ENUM_EXPAND(PYPTO_ENUM_FE_16(ACTION, ctx, __VA_ARGS__))
#define PYPTO_ENUM_FE_18(ACTION, ctx, x, ...) \
    ACTION(ctx, x) PYPTO_ENUM_EXPAND(PYPTO_ENUM_FE_17(ACTION, ctx, __VA_ARGS__))
#define PYPTO_ENUM_FE_19(ACTION, ctx, x, ...) \
    ACTION(ctx, x) PYPTO_ENUM_EXPAND(PYPTO_ENUM_FE_18(ACTION, ctx, __VA_ARGS__))
#define PYPTO_ENUM_FE_20(ACTION, ctx, x, ...) \
    ACTION(ctx, x) PYPTO_ENUM_EXPAND(PYPTO_ENUM_FE_19(ACTION, ctx, __VA_ARGS__))
#define PYPTO_ENUM_FE_21(ACTION, ctx, x, ...) \
    ACTION(ctx, x) PYPTO_ENUM_EXPAND(PYPTO_ENUM_FE_20(ACTION, ctx, __VA_ARGS__))
#define PYPTO_ENUM_FE_22(ACTION, ctx, x, ...) \
    ACTION(ctx, x) PYPTO_ENUM_EXPAND(PYPTO_ENUM_FE_21(ACTION, ctx, __VA_ARGS__))
#define PYPTO_ENUM_FE_23(ACTION, ctx, x, ...) \
    ACTION(ctx, x) PYPTO_ENUM_EXPAND(PYPTO_ENUM_FE_22(ACTION, ctx, __VA_ARGS__))
#define PYPTO_ENUM_FE_24(ACTION, ctx, x, ...) \
    ACTION(ctx, x) PYPTO_ENUM_EXPAND(PYPTO_ENUM_FE_23(ACTION, ctx, __VA_ARGS__))

#define PYPTO_ENUM_GET_MACRO(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, \
                             _20, _21, _22, _23, _24, NAME, ...)                                                       \
    NAME

#define PYPTO_FOR_EACH(action, ctx, ...)                                                                            \
    PYPTO_ENUM_EXPAND(PYPTO_ENUM_GET_MACRO(                                                                         \
        _0, __VA_ARGS__, PYPTO_ENUM_FE_24, PYPTO_ENUM_FE_23, PYPTO_ENUM_FE_22, PYPTO_ENUM_FE_21, PYPTO_ENUM_FE_20,  \
        PYPTO_ENUM_FE_19, PYPTO_ENUM_FE_18, PYPTO_ENUM_FE_17, PYPTO_ENUM_FE_16, PYPTO_ENUM_FE_15, PYPTO_ENUM_FE_14, \
        PYPTO_ENUM_FE_13, PYPTO_ENUM_FE_12, PYPTO_ENUM_FE_11, PYPTO_ENUM_FE_10, PYPTO_ENUM_FE_9, PYPTO_ENUM_FE_8,   \
        PYPTO_ENUM_FE_7, PYPTO_ENUM_FE_6, PYPTO_ENUM_FE_5, PYPTO_ENUM_FE_4, PYPTO_ENUM_FE_3, PYPTO_ENUM_FE_2,       \
        PYPTO_ENUM_FE_1)(action, ctx, __VA_ARGS__))

/**
 * \brief PYPTO_DECLARE_ENUM — one-line enum class with auto EnumToString
 *
 * Usage (must be inside the target namespace):
 * \code
 *   namespace pypto { namespace ir {
 *     PYPTO_DECLARE_ENUM(MyEnum, VALUE_A, VALUE_B, VALUE_C)
 *   }}
 * \endcode
 *
 * Generates:
 *   - enum class MyEnum { VALUE_A, VALUE_B, VALUE_C };
 *   - inline const char* EnumToString(MyEnum v);
 */

#define PYPTO_ENUM_DECL_VALUE(_, v) v,

#define PYPTO_ENUM_DECL_CASE(EnumName, v) \
    case EnumName::v:                     \
        return #EnumName "::" #v;

#define PYPTO_DECLARE_ENUM(EnumName, ...)                                          \
    enum class EnumName { PYPTO_FOR_EACH(PYPTO_ENUM_DECL_VALUE, _, __VA_ARGS__) }; \
                                                                                   \
    inline const char* EnumToString(EnumName v)                                    \
    {                                                                              \
        switch (v) {                                                               \
            PYPTO_FOR_EACH(PYPTO_ENUM_DECL_CASE, EnumName, __VA_ARGS__)            \
            default:                                                               \
                return "UNKNOWN";                                                  \
        }                                                                          \
    }

namespace pypto {
namespace ir {

/**
 * \brief Block-level DSL API parameter enumerations
 *
 * Each enum is declared via PYPTO_DECLARE_ENUM, which auto-generates:
 *   - EnumToString(EnumName) -> const char*  (returns "EnumName::VALUE")
 */

// TensorLayout: see documentation in ir/type.h
PYPTO_DECLARE_ENUM(TensorLayout, ND, DN, NZ, ZN, NN, ZZ)

PYPTO_DECLARE_ENUM(ReluPreMode, NormalRelu)

PYPTO_DECLARE_ENUM(AtomicType, AtomicNone, AtomicAdd)

PYPTO_DECLARE_ENUM(STPhase, Unspecified, Partial, Final)

PYPTO_DECLARE_ENUM(AccPhase, Unspecified, Partial, Final)

PYPTO_DECLARE_ENUM(AccToVecMode, SingleModeVec0, SingleModeVec1, DualModeSplitM, DualModeSplitN)

PYPTO_DECLARE_ENUM(RoundMode, CAST_NONE, CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC, CAST_ODD)

PYPTO_DECLARE_ENUM(QuantMode, SYM, ASYM)

PYPTO_DECLARE_ENUM(CrossCoreSyncMode, INTER_BLOCK, INTER_SUBBLOCK, INTRA_BLOCK, UNICAST_BLOCK)

PYPTO_DECLARE_ENUM(SyncCoreType, AIV_ONLY, AIC_ONLY, MIX)

PYPTO_DECLARE_ENUM(CacheLine, SINGLE_CACHE_LINE, ENTIRE_DATA_CACHE)

PYPTO_DECLARE_ENUM(DcciDst, AUTO, CACHELINE_OUT, CACHELINE_UB, CACHELINE_ALL, CACHELINE_ATOMIC)

/**
 * \brief VF (Vector Function) API parameter enumerations
 */

PYPTO_DECLARE_ENUM(MaskPattern, ALL, ALLF, VL1, VL2, VL3, VL4, VL8, VL16, VL32, VL64, VL128, M3, M4, H, Q)

PYPTO_DECLARE_ENUM(MergeMode, ZEROING, MERGING)

PYPTO_DECLARE_ENUM(ReduceMode, SUM, MAX, MIN)

PYPTO_DECLARE_ENUM(CompareMode, EQ, NE, LT, LE, GT, GE)

PYPTO_DECLARE_ENUM(CmpMode, EQ, NE, LT, LE, GT, GE)

PYPTO_DECLARE_ENUM(DuplicatePos, LOWEST, HIGHEST)

PYPTO_DECLARE_ENUM(CastLayout, ZERO, ONE, TWO, THREE)

PYPTO_DECLARE_ENUM(VFRoundMode, CAST_ROUND, CAST_RINT, CAST_FLOOR, CAST_CEIL, CAST_TRUNC, CAST_ODD, CAST_HYBRID)

PYPTO_DECLARE_ENUM(SaturateMode, OFF, ON)

PYPTO_DECLARE_ENUM(SaturationFlagMode, FLOAT, FLOAT8, INT, CAST)

PYPTO_DECLARE_ENUM(BinType, BIN0, BIN1)

PYPTO_DECLARE_ENUM(HistType, ACCUMULATE, FREQUENCY)

PYPTO_DECLARE_ENUM(SqueezeMode, STORE_REG, NO_STORE_REG)

PYPTO_DECLARE_ENUM(PackPart, LOWER, UPPER)

PYPTO_DECLARE_ENUM(MaskWidth, B32, B16)

PYPTO_DECLARE_ENUM(LoadDist, NORM, BRC, BRC_B8, BRC_B16, BRC_B32, US, US_B8, US_B16, DS, DS_B8, DS_B16, UNPK, UNPK_B8,
                   UNPK_B16, UNPK_B32, UNPK4, BLK, E2B, E2B_B16, E2B_B32, DINTLV_B8, DINTLV_B16, DINTLV_B32)

// StoreDist: coarse names (NORM/FIRST_ELEMENT/PACK/PACK4/INTLV) auto-expand to the
// element-width variant based on the source dtype; the width-qualified names
// (mirroring AscendC Reg::StoreDist) select the granularity explicitly.
// Append-only: existing numeric values are serialized in IR payloads.
PYPTO_DECLARE_ENUM(StoreDist, NORM, NORM_B16, FIRST_ELEMENT, PACK, PACK4, INTLV, INTLV_B32, NORM_B8, NORM_B32,
                   FIRST_ELEMENT_B8, FIRST_ELEMENT_B16, FIRST_ELEMENT_B32, PACK_B16, PACK_B32, PACK_B64, PACK4_B32,
                   INTLV_B8, INTLV_B16)

PYPTO_DECLARE_ENUM(DataCopyMode, NORM, DATA_BLOCK_LOAD, DATA_BLOCK_COPY)

PYPTO_DECLARE_ENUM(IndexOrder, INCREASE_ORDER, DECREASE_ORDER)

PYPTO_DECLARE_ENUM(MemBarMode, VST_VLD, VLD_VST, VST_VST, VST_LD, VST_ST, VLD_ST, ST_VLD, ST_VST, LD_VST, VV_ALL,
                   VS_ALL, SV_ALL)

} // namespace ir
} // namespace pypto

#endif // PYPTO_IR_OP_ATTR_TYPES_H_
