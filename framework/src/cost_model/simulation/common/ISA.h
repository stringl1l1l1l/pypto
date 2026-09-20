/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ISA.h
 * \brief
 */

#pragma once

#include <sstream>
#include <memory>
#include <vector>
#include <string>
#include <set>
#include <map>
#include "cost_model/simulation/common/CommonType.h"
#include "cost_model/simulation/common/CommonTools.h"
#include "tilefwk/data_type.h"
#include "tilefwk/element.h"
#include "interface/operation/operation.h"

namespace CostModel {

constexpr uint64_t RAW_MAGIC_MAX_SIZE = 1024 * 1024;

const std::map<std::string, CorePipeType> SCHED_CORE_PIPE_TYPE{// Unary Vector
                                                               {"EXP", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"NEG", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"RSQRT", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"SQRT", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"RECIPROCAL", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"CAST", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"EXPAND", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"ONEHOT", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"COMPACT", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"LOGICALNOT", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"ROWMAX", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"ROWSUM", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"ROWEXPMAX", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"ROWEXPSUM", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"ROWSUMLINE", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"ROWMAXLINE", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"ROWMINLINE", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"ADDS", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"SUBS", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"MULS", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"DIVS", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"MAXS", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"MINS", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"POW", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"POWS", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"LReLU", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"S_ADDS", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"S_SUBS", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"S_MULS", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"S_DIVS", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"S_MAXS", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"S_MINS", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"TRANSPOSE_MOVEIN", CorePipeType::PIPE_MTE_IN},
                                                               {"TRANSPOSE_MOVEOUT", CorePipeType::PIPE_MTE_OUT},
                                                               {"L1_RESHAPE_COPY_IN", CorePipeType::PIPE_MTE_IN},
                                                               {"L0C_RESHAPE_COPY_OUT", CorePipeType::PIPE_MTE_OUT},
                                                               {"TRANSPOSE_VNCHWCONV", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"ABS", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"PERMUTE", CorePipeType::PIPE_S},
                                                               {"PERMUTE_ELEMENT", CorePipeType::PIPE_S},
                                                               {"LN", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"ISFINITE", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"ISNAN", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"HUB", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"BRCB", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"FLOOR", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"CEIL", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"ROUND", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"TRUNC", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"SIGN", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"SIGNBIT", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"BITWISENOT", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"FILLPAD", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"RELU", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"EXPM1", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"EXP2", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"SIN", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"COS", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"SINH", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"COSH", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"TAN", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"TANH", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"ATAN", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"ASIN", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"ACOS", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"ERF", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"ERFC", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"Log1p", CorePipeType::PIPE_VECTOR_ALU},
                                                               // Binary Vector
                                                               {"ADD", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"SUB", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"MUL", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"DIV", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"ADD_BRC", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"SUB_BRC", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"MUL_BRC", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"DIV_BRC", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"MAX_BRC", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"MIN_BRC", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"S_ADD", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"S_SUB", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"S_MUL", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"S_DIV", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"S_MAX", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"S_MIN", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"MAXIMUM", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"MINIMUM", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"GATHER_FROM_UB", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"GATHER", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"GATHER_ELEMENT", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"SCATTER_ELEMENT", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"SCATTER", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"INDEX_PUT", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"INDEX_ADD", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"CONCAT", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"CUM_SUM", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"SCATTER_UPDATE", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"SCATTER_SCALAR", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"PAIRMAX", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"PAIRMIN", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"PAIRSUM", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"PAIRPROD", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"ROWMAX_SINGLE", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"ROWMIN_SINGLE", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"ROWSUM_SINGLE", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"ROWARGMAX_SINGLE", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"ROWARGMIN_SINGLE", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"ROWARGMAXLINE", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"ROWARGMINLINE", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"ROWPROD_SINGLE", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"ROWPRODLINE", CorePipeType::PIPE_VECTOR_ALU},

                                                               {"CMP", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"CMPS", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"WHERE_TT", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"WHERE_TS", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"WHERE_ST", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"WHERE_SS", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"LOGICALAND", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"BITWISEAND", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"BITWISEOR", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"BITWISEXOR", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"BITWISEANDS", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"BITWISEORS", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"BITWISEXORS", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"BITWISERIGHTSHIFT", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"BITWISELEFTSHIFT", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"BITWISERIGHTSHIFTS", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"BITWISELEFTSHIFTS", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"SBITWISERIGHTSHIFT", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"SBITWISELEFTSHIFT", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"COPYSIGN", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"FLOORDIV", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"FLOORDIVS", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"EXPANDEXPDIF", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"GATHER_MASK", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"GATHER_MASK_BUILDIN", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"GCD", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"GCDS", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"GCD_BRC", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"HYPOT", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"PRELU", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"AXPY", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"ATAN2", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"REM", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"REMS", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"REMRS", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"MOD", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"MODS", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"CUM_PROD", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"TRIUL", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"INDEX_ADD_UB", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"QUANTIZE_SYM", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"QUANTIZE_ASYM", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"DEQUANTIZE", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"QUANT_MX", CorePipeType::PIPE_VECTOR_ALU},
                                                               // Cube
                                                               {"A_MUL_B", CorePipeType::PIPE_CUBE},
                                                               {"A_MULACC_B", CorePipeType::PIPE_CUBE},
                                                               {"A_MUL_Bt", CorePipeType::PIPE_CUBE},
                                                               {"A_MULACC_Bt", CorePipeType::PIPE_CUBE},
                                                               {"At_MUL_B", CorePipeType::PIPE_CUBE},
                                                               {"At_MUL_Bt", CorePipeType::PIPE_CUBE},
                                                               {"CONV", CorePipeType::PIPE_CUBE},
                                                               {"CONV_ADD", CorePipeType::PIPE_CUBE},
                                                               {"CUBE_CONV2D", CorePipeType::PIPE_CUBE},
                                                               {"CUBE_CONV3D", CorePipeType::PIPE_CUBE},
                                                               {"CUBE_CONV_BP_INPUT_2D", CorePipeType::PIPE_CUBE},
                                                               {"CUBE_CONV_BP_INPUT_3D", CorePipeType::PIPE_CUBE},
                                                               {"FAKE_TRANS", CorePipeType::PIPE_MTE_OUT},
                                                               {"CONV_D2S", CorePipeType::PIPE_FIX},
                                                               {"CONCAT_C", CorePipeType::PIPE_CUBE},
                                                               {"L1_TO_L0A", CorePipeType::PIPE_MTE1},
                                                               {"L1_TO_L0B", CorePipeType::PIPE_MTE1},
                                                               {"L1_TO_BT", CorePipeType::PIPE_MTE1},
                                                               {"L1_TO_L0A_SCALE", CorePipeType::PIPE_MTE1},
                                                               {"L1_TO_L0B_SCALE", CorePipeType::PIPE_MTE1},
                                                               {"LOAD2D", CorePipeType::PIPE_MTE1},
                                                               {"LOAD2DDX", CorePipeType::PIPE_MTE1},
                                                               {"IMG2COL", CorePipeType::PIPE_MTE1},
                                                               // ANY
                                                               {"DUPLICATE", CorePipeType::PIPE_VECTOR_ALU},
                                                               // View
                                                               {"RESHAPE", CorePipeType::PIPE_S},
                                                               {"RESHAPE_COPY_IN", CorePipeType::PIPE_MTE_IN},
                                                               {"RESHAPE_COPY_OUT", CorePipeType::PIPE_MTE_OUT},
                                                               {"ASSEMBLE", CorePipeType::PIPE_S},
                                                               {"ASSEMBLE_SSA", CorePipeType::PIPE_S},
                                                               {"VIEW", CorePipeType::PIPE_S},
                                                               {"VIEW_TYPE", CorePipeType::PIPE_S},

                                                               // Move
                                                               {"INDEX_OUTCAST", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"REGISTER_COPY", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"ONLINE_SOFTMAX", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"ONLINE_SOFTMAX_UPDATE", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"CONVERT", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"COPY_IN", CorePipeType::PIPE_MTE_IN},
                                                               {"COPY_OUT", CorePipeType::PIPE_MTE_OUT},
                                                               {"GATHER_IN_L1", CorePipeType::PIPE_MTE_IN},
                                                               {"GATHER_IN_UB", CorePipeType::PIPE_MTE_IN},
                                                               {"LOAD", CorePipeType::PIPE_S},
                                                               // Special
                                                               {"CALL", CorePipeType::PIPE_CALL},
                                                               {"BLOCK_CALL", CorePipeType::PIPE_CALL},
                                                               {"CALL_NOT_EXPAND", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"OP_DUMP", CorePipeType::PIPE_S},
                                                               {"NOP", CorePipeType::TOTAL_CORE_PIPE_TYPE},

                                                               {"UB_ALLOC", CorePipeType::PIPE_VECTOR_BMU},
                                                               {"UB_COPY_IN", CorePipeType::PIPE_MTE_IN},
                                                               {"UB_COPY_OUT", CorePipeType::PIPE_MTE_OUT},
                                                               {"VEC_DUP", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"REG_ALLOC", CorePipeType::PIPE_S},
                                                               {"VLD", CorePipeType::PIPE_S},
                                                               {"VST", CorePipeType::PIPE_S},

                                                               // Cube
                                                               {"L1_ALLOC", CorePipeType::PIPE_CUBE_BMU_L1},
                                                               {"L0A_ALLOC", CorePipeType::PIPE_CUBE_BMU_L0A},
                                                               {"L0AMX_ALLOC", CorePipeType::PIPE_S},
                                                               {"L0B_ALLOC", CorePipeType::PIPE_CUBE_BMU_L0B},
                                                               {"L0BMX_ALLOC", CorePipeType::PIPE_S},
                                                               {"L0C_ALLOC", CorePipeType::PIPE_CUBE_BMU_L0C},
                                                               {"FIX_ALLOC", CorePipeType::PIPE_CUBE_BMU_L0C},
                                                               {"BT_ALLOC", CorePipeType::PIPE_CUBE_BMU_L0B},

                                                               // MTE
                                                               {"L1_COPY_IN", CorePipeType::PIPE_MTE_IN},
                                                               {"L1_COPY_IN_CONV", CorePipeType::PIPE_MTE_IN},
                                                               {"L1_COPY_IN_CONV_BP", CorePipeType::PIPE_MTE_IN},
                                                               {"L1_COPY_IN_CONV_BP_DX_DY", CorePipeType::PIPE_MTE_IN},
                                                               {"L1_COPY_IN_A_SCALE", CorePipeType::PIPE_MTE_IN},
                                                               {"L1_COPY_IN_B_SCALE", CorePipeType::PIPE_MTE_IN},
                                                               {"L1_COPY_IN_FractalZ", CorePipeType::PIPE_MTE_IN},
                                                               {"L1_COPY_OUT", CorePipeType::PIPE_MTE_OUT},
                                                               {"L1_LOOP_ENHANCE", CorePipeType::PIPE_MTE_IN},

                                                               {"L1_COPY_IN_DMA", CorePipeType::PIPE_MTE_IN},
                                                               {"L1_COPY_OUT_DMA", CorePipeType::PIPE_MTE_OUT},
                                                               {"NCHW2NC1HWC0", CorePipeType::PIPE_MTE_OUT},
                                                               {"NCHW2Fractal_Z", CorePipeType::PIPE_MTE_OUT},
                                                               {"NC1HWC02NCHW", CorePipeType::PIPE_MTE_OUT},
                                                               {"NCDHW2NDC1HWC0", CorePipeType::PIPE_MTE_OUT},
                                                               {"NCDHW2FRACTAL_Z_3D", CorePipeType::PIPE_MTE_OUT},
                                                               {"NDC1HWC02NCDHW", CorePipeType::PIPE_MTE_OUT},

                                                               {"L0C_COPY_OUT", CorePipeType::PIPE_MTE_OUT},
                                                               {"L0C_COPY_OUT_CONV", CorePipeType::PIPE_FIX},
                                                               {"L0C_COPY_UB_DUAL_DST", CorePipeType::PIPE_FIX},
                                                               {"L0C_TO_L1", CorePipeType::PIPE_FIX},
                                                               {"CUBE_CONCAT_C", CorePipeType::PIPE_MTE1},
                                                               {"L1_TO_L0At", CorePipeType::PIPE_MTE1},
                                                               {"L1_TO_L0Bt", CorePipeType::PIPE_MTE1},
                                                               {"FIX_COPY_IN", CorePipeType::PIPE_MTE1},
                                                               {"FIX_COPY_IN_QUANT_PRE", CorePipeType::PIPE_MTE1},
                                                               {"FIX_COPY_IN_RELU_PRE", CorePipeType::PIPE_MTE1},
                                                               {"FIX_COPY_IN_RELU_POST", CorePipeType::PIPE_MTE1},
                                                               {"FIX_COPY_IN_QUANT_POST", CorePipeType::PIPE_MTE1},
                                                               {"FIX_COPY_IN_ELT_ANTIQ", CorePipeType::PIPE_MTE1},
                                                               {"FIX_COPY_IN_MTE2_ANTIQ", CorePipeType::PIPE_MTE1},
                                                               {"L1_COPY_UB", CorePipeType::PIPE_MTE1},
                                                               {"L0C_COPY_UB", CorePipeType::PIPE_MTE1},
                                                               {"UB_COPY_L1", CorePipeType::PIPE_MTE1},
                                                               {"UB_COPY_L1_ND", CorePipeType::PIPE_MTE1},
                                                               {"L1_TO_L1", CorePipeType::PIPE_MTE1},
                                                               {"UB_TO_UB", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"L0C_COPY_L1", CorePipeType::PIPE_MTE1},
                                                               {"UB_COPY_ND2NZ", CorePipeType::PIPE_VECTOR_ALU},

                                                               // Scala
                                                               {"SYNC_SRC", CorePipeType::PIPE_S},
                                                               {"SYNC_DST", CorePipeType::PIPE_S},
                                                               {"CV_SYNC_SRC", CorePipeType::PIPE_S},
                                                               {"CV_SYNC_DST", CorePipeType::PIPE_S},
                                                               {"PHASE1", CorePipeType::PIPE_S},
                                                               {"PHASE2", CorePipeType::PIPE_S},
                                                               {"BAR.V", CorePipeType::PIPE_S},
                                                               {"BAR.M", CorePipeType::PIPE_S},
                                                               {"BAR.ALL", CorePipeType::PIPE_S},
                                                               {"BAR_V", CorePipeType::PIPE_S},
                                                               {"BAR_M", CorePipeType::PIPE_S},
                                                               {"BAR_ALL", CorePipeType::PIPE_S},
                                                               {"PAD", CorePipeType::PIPE_S},

                                                               // Distributed
                                                               {"REMOTE_GATHER", CorePipeType::PIPE_MTE_OUT},
                                                               {"LOCAL_COPY_OUT", CorePipeType::PIPE_MTE_OUT},
                                                               {"WRITE_REMOTE", CorePipeType::PIPE_MTE_OUT},
                                                               {"REMOTE_REDUCE", CorePipeType::PIPE_MTE_OUT},
                                                               {"COMM_WAIT_FLAG", CorePipeType::PIPE_S},
                                                               {"DEPEND_ON", CorePipeType::PIPE_S},
                                                               {"SHMEM_SET", CorePipeType::PIPE_S},
                                                               {"SHMEM_PUT", CorePipeType::PIPE_MTE_OUT},
                                                               {"SHMEM_PUT_UB2GM", CorePipeType::PIPE_MTE_OUT},
                                                               {"SHMEM_SIGNAL", CorePipeType::PIPE_S},
                                                               {"SHMEM_WAIT_UNTIL", CorePipeType::PIPE_S},
                                                               {"SHMEM_GET", CorePipeType::PIPE_MTE_IN},
                                                               {"SHMEM_GET_GM2UB", CorePipeType::PIPE_MTE_IN},
                                                               {"BIND_TENSOR", CorePipeType::PIPE_S},
                                                               // TOPK and ArgSort
                                                               {"TOPK", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"TILEDMRGSORT", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"BITSORT", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"MRGSORT", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"ARGSORT", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"EXTRACT", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"EXTRACTSINGLE", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"TWOTILEMRGSORT", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"SORTUB", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"RADIX_SELECT", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"FUSED_OP", CorePipeType::PIPE_VECTOR_ALU},
                                                               // topk for DS3.2-Day0
                                                               {"TOPK_SORT", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"TOPK_MERGE", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"TOPK_EXTRACT", CorePipeType::PIPE_VECTOR_ALU},
                                                               // Reduce Atomic
                                                               {"REDUCE_ACC", CorePipeType::PIPE_VECTOR_ALU},
                                                               // aicpu-aicore communication
                                                               {"AICPU_CALL_AIC", CorePipeType::PIPE_S},
                                                               {"AICPU_CALL_AIV", CorePipeType::PIPE_S},

                                                               {"MAX_POOL", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"RANGE", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"UNIFORM", CorePipeType::PIPE_VECTOR_ALU},
                                                               // parallel sort
                                                               {"SORT", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"COMP_SWAP", CorePipeType::PIPE_VECTOR_ALU},
                                                               {"MERGE", CorePipeType::PIPE_VECTOR_ALU},

                                                               {"UNKNOWN", CorePipeType::PIPE_UNKNOW}};

struct ExecuteInfo {
    int exePipeId = -1;
    uint64_t latency = 1;
    bool isIncast = false;
    bool isOutcast = false;
    // For Scheduler Sort TileOp and Tile.
    int copyOutIdx = -1;
    int domCount = 1;
    int sequenceToIssue = 0;

    // For get sequence for schedule_ooo pass
    bool visited = false;

    bool issued = false;
    bool retired = false;

    // For Example Function: tile[BUF_UNKNOWN] -> copy_out -> tile[BUF_UNKNOWN]
    bool noSrcWakeup = false;
    bool noDstWakeup = false;

    // Tile Execute Info
    bool isAllocated = false;
    bool isWritten = false;
    uint64_t writeReference = 0;
    uint64_t readReference = 0;

    CycleInfo cycleInfo;
    void Reset();
};
class TileOp;
using TileOpPtr = std::shared_ptr<TileOp>;
class Function;
using FunctionPtr = std::shared_ptr<CostModel::Function>;

class Tile {
public:
    int magic = -1;
    int subgraphId = -1;

    std::string symbol = "";
    std::string dataTypeStr = "";
    npu::tile_fwk::DataType dataType = npu::tile_fwk::DataType::DT_FP16;
    NodeType nodeType = NodeType::LOCAL;
    std::string bufferType = "";
    OperandType bufType = BUF_UNKNOWN;
    CorePipeType pipeType = CorePipeType::PIPE_UNKNOW;

    std::vector<int> offset;
    std::vector<int> shape;

    int rawMagic = -1;
    std::vector<int> rawShape;

    TileOpPtr producer = nullptr;
    std::vector<TileOpPtr> producers;
    std::vector<TileOpPtr> consumers;

    bool isSubGraphBoundary = false;

    FunctionPtr funcPtr = nullptr;

    ExecuteInfo exeInfo;
    std::string semanticLabels;

    // For Calculate
    bool isScale = false;
    bool isAddr = false;
    int scaleData = 0;
    uint64_t addrReg = 0;
    std::vector<std::vector<uint64_t>> tileData;

    explicit Tile() {}
    explicit Tile(const std::string& str);
    void GetPipeType();
    std::string Dump();
    int SizeinBytes();
};

using TilePtr = std::shared_ptr<Tile>;

class TileOp {
public:
    int magic = -1;
    int subgraphId = -1;
    uint64_t taskId = 0;
    bool specialOp = false; // For empty tileOp, (reshape)

    std::string opcode = "";
    OperandType bufType = BUF_UNKNOWN;
    CorePipeType pipeType = CorePipeType::PIPE_UNKNOW;
    ExecuteInfo exeInfo;
    uint64_t calleeHash{};

    std::string semanticLabel;

    bool scalarVld = false;
    Element scalarVal;
    std::vector<TilePtr> iOperand;
    std::vector<TilePtr> oOperand;

    npu::tile_fwk::Operation* operation{};
    FunctionPtr funcPtr = nullptr;

    explicit TileOp() = default;

    void GetPipeType();
    // For MTE_IN/MTE_OUT access cache.
    uint64_t GetAddress();
    uint64_t GetSize();
    bool IsCall();
    bool IsSpecial();
    std::string Dump(bool outDetail = false);
};

class Subgraph {};

class FunctionInvokeInfo {
public:
    std::unordered_set<int> args;
    std::unordered_map<int, TilePtr> binds;

    TilePtr Bind(int magic)
    {
        if (binds.count(magic)) {
            return binds[magic];
        }
        return nullptr;
    }
};

class Function {
public:
    int magic = -1;
    int pSgId = -1; // SubgraphId
    CostModel::MachineType machineType;
    npu::tile_fwk::Function* parentFunction;
    uint64_t functionHash = -1;
    std::string funcName = "";
    std::vector<TileOpPtr> tileOps;
    std::vector<TilePtr> tiles;
    std::unordered_map<int, TilePtr> tileMap;
    std::unordered_map<int, TileOpPtr> tileOpMap;

    std::vector<int> incastMagic;
    std::vector<int> outcastMagic;
    std::vector<int> tileMagic;
    std::vector<int> opMagic;
    int esgId;
    std::unordered_map<int, FunctionInvokeInfo> invoke;

    // For root function
    bool topoFromRootFunc = false;
    std::vector<TopoInfoEntry> inputTopo;

    // Schedule info.
    bool hasSchedule = false;
    std::vector<std::vector<int>> tileAllocSequence;

    // TILEOP sequence from ooo pass. <magic, seq>
    std::unordered_map<int, uint64_t> opSequenceAfterOOO_;
    std::vector<int> opMagicSequence;
    void GetOpSequeceAfterOOO(int opmagic, uint64_t& index);

    // semanticLabels
    std::string semanticLabels;

    bool hasRecordInfo = false;
    uint64_t startCycles = 0;
    uint64_t totalCycles = 0;
    std::unordered_map<CostModel::CorePipeType, uint64_t> pipeLastEndCycle;
    std::unordered_map<CostModel::CorePipeType, uint64_t> pipeExecuteTime;
    void InitPipeExecTime();
    Json DumpExecuteInfo();
    uint64_t GetOpRelativeReadyCycle(TileOpPtr tileOp, uint64_t newBaseCycle);
    void CalculateRelativeCycle(uint64_t newBaseCycle, double proportion);
};
} // namespace CostModel
