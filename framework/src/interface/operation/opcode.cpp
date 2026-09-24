/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file opcode.cpp
 * \brief
 */

#include "opcode.h"
#include <string>
#include <array>
#include <sstream>
#include <unordered_set>
#include "interface/operation/operation.h"
#include "interface/utils/common.h"
#include "tilefwk/data_type.h"
#include "tilefwk/error.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "tile_shape_verifier.h"

namespace npu::tile_fwk {

void OpcodeManager::RegisterInfo(
    Opcode opcode, OpCoreType coreType, std::string str, std::vector<MemoryType> inputsMemType,
    std::vector<MemoryType> outputsMemType, const TileOpCfg tileOpCfg, OpCalcType calcType,
    const std::vector<std::string>& attrs, VerifyOperationEntry verifyOperationEntry,
    std::unordered_map<std::string, std::vector<std::vector<TileOpFormat>>> supportOpFormatList)
{
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, opcode < Opcode::OP_UNKNOWN)
        << "opcode: " << static_cast<int64_t>(opcode) << " is unknown";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, strToEnum_.count(str) == 0) << str << " doesn't exist.";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, registered_.count(opcode) == 0)
        << "opcode: " << static_cast<int64_t>(opcode) << " not registered.";
    registered_.emplace(opcode);
    strToEnum_.emplace(str, opcode);
    opcodeInfos_[static_cast<int>(opcode)] = OpcodeInfo{
        opcode,    coreType, std::move(str), std::move(inputsMemType), std::move(outputsMemType),
        tileOpCfg, calcType, attrs,          verifyOperationEntry,     std::move(supportOpFormatList)};
}

void OpcodeManager::RegisterVectorBinary()
{
    RegisterInfo(Opcode::OP_ADD_BRC, OpCoreType::AIV, "ADD_BRC", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Taddbrc", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::BROADCAST, {OpAttributeKey::inputCombineAxis});
    RegisterInfo(Opcode::OP_SUB_BRC, OpCoreType::AIV, "SUB_BRC", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Tsubbrc", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::BROADCAST, {OpAttributeKey::inputCombineAxis});
    RegisterInfo(Opcode::OP_MUL_BRC, OpCoreType::AIV, "MUL_BRC", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Tmulbrc", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::BROADCAST, {OpAttributeKey::inputCombineAxis});
    RegisterInfo(Opcode::OP_DIV_BRC, OpCoreType::AIV, "DIV_BRC", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Tdivbrc", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::BROADCAST, {OpAttributeKey::inputCombineAxis});
    RegisterInfo(Opcode::OP_MAX_BRC, OpCoreType::AIV, "MAX_BRC", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Tmaxbrc", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::BROADCAST, {OpAttributeKey::inputCombineAxis});
    RegisterInfo(Opcode::OP_MIN_BRC, OpCoreType::AIV, "MIN_BRC", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Tminbrc", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::BROADCAST, {OpAttributeKey::inputCombineAxis});
    RegisterInfo(Opcode::OP_GCD_BRC, OpCoreType::AIV, "GCD_BRC", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::TGcdbrc", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::BROADCAST, {OpAttributeKey::inputCombineAxis});
    RegisterInfo(Opcode::OP_ADD, OpCoreType::AIV, "ADD", {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Tadd", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::BROADCAST,
                 {OpAttributeKey::inputCombineAxis}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_SUB, OpCoreType::AIV, "SUB", {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Tsub", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::BROADCAST,
                 {OpAttributeKey::inputCombineAxis}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_MUL, OpCoreType::AIV, "MUL", {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Tmul", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::BROADCAST,
                 {OpAttributeKey::inputCombineAxis}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_DIV, OpCoreType::AIV, "DIV", {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Tdiv", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::BROADCAST,
                 {OpAttributeKey::inputCombineAxis}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_REM, OpCoreType::AIV, "REM", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::TRemainder", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::BROADCAST, {OpAttributeKey::inputCombineAxis, OpAttributeKey::excludeBufferReuse},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_MAXIMUM, OpCoreType::AIV, "MAXIMUM", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB}, {"TileOp::Tmax", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::BROADCAST,
                 {OpAttributeKey::inputCombineAxis}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_MINIMUM, OpCoreType::AIV, "MINIMUM", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB}, {"TileOp::Tmin", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::BROADCAST,
                 {OpAttributeKey::inputCombineAxis}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_BITWISERIGHTSHIFT, OpCoreType::AIV, "BITWISERIGHTSHIFT",
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {"TileOp::Tbitwiserightshift", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::BROADCAST,
                 {OpAttributeKey::inputCombineAxis, OpAttributeKey::excludeBufferReuse}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_BITWISELEFTSHIFT, OpCoreType::AIV, "BITWISELEFTSHIFT",
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {"TileOp::Tbitwiseleftshift", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::BROADCAST,
                 {OpAttributeKey::inputCombineAxis, OpAttributeKey::excludeBufferReuse}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_BITWISEAND, OpCoreType::AIV, "BITWISEAND", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB}, {"TileOp::TbitwiseAnd", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::BROADCAST,
                 {OpAttributeKey::inputCombineAxis}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_BITWISEOR, OpCoreType::AIV, "BITWISEOR", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB}, {"TileOp::TbitwiseOr", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::BROADCAST,
                 {OpAttributeKey::inputCombineAxis}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_BITWISEXOR, OpCoreType::AIV, "BITWISEXOR", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::TbitwiseXor", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::BROADCAST, {OpAttributeKey::inputCombineAxis}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_COPYSIGN, OpCoreType::AIV, "COPYSIGN", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Tcopysign", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::BROADCAST, {OpAttributeKey::inputCombineAxis}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_ATAN2, OpCoreType::AIV, "ATAN2", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::TAtan2", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::BROADCAST, {OpAttributeKey::inputCombineAxis, OpAttributeKey::excludeBufferReuse},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_GCD, OpCoreType::AIV, "GCD", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::TGcd", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::BROADCAST, {OpAttributeKey::inputCombineAxis}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_S_ADD, OpCoreType::AIV, "S_ADD", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB}, {"TileOp::TSadd", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::BROADCAST,
                 {OpAttributeKey::inputCombineAxis});
    RegisterInfo(Opcode::OP_S_SUB, OpCoreType::AIV, "S_SUB", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB}, {"TileOp::TSsub", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::BROADCAST,
                 {OpAttributeKey::inputCombineAxis});
    RegisterInfo(Opcode::OP_S_MUL, OpCoreType::AIV, "S_MUL", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB}, {"TileOp::TSmul", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::BROADCAST,
                 {OpAttributeKey::inputCombineAxis});
    RegisterInfo(Opcode::OP_S_DIV, OpCoreType::AIV, "S_DIV", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB}, {"TileOp::TSdiv", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::BROADCAST,
                 {OpAttributeKey::inputCombineAxis});
    RegisterInfo(Opcode::OP_S_MAX, OpCoreType::AIV, "S_MAX", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB}, {"TileOp::TSmax", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::BROADCAST,
                 {OpAttributeKey::inputCombineAxis});
    RegisterInfo(Opcode::OP_S_MIN, OpCoreType::AIV, "S_MIN", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB}, {"TileOp::TSmin", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::BROADCAST,
                 {OpAttributeKey::inputCombineAxis});
    RegisterInfo(Opcode::OP_ADDS, OpCoreType::AIV, "ADDS", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Tadds", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
                 {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OP_ATTR_PREFIX + "reverseOperand",
                  OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_SUBS, OpCoreType::AIV, "SUBS", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Tsubs", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
                 {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OP_ATTR_PREFIX + "reverseOperand",
                  OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_MULS, OpCoreType::AIV, "MULS", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Tmuls", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
                 {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OP_ATTR_PREFIX + "reverseOperand",
                  OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis},
                 TileShapeVerifier::Verify);
    RegisterInfo(
        Opcode::OP_DIVS, OpCoreType::AIV, "DIVS", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
        {"TileOp::Tdivs", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
        {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OP_ATTR_PREFIX + "reverseOperand",
         OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis, OpAttributeKey::excludeBufferReuse},
        TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_MAXS, OpCoreType::AIV, "MAXS", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Tmaxs", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
                 {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OP_ATTR_PREFIX + "reverseOperand",
                  OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_MINS, OpCoreType::AIV, "MINS", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Tmins", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
                 {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OP_ATTR_PREFIX + "reverseOperand",
                  OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_LRELU, OpCoreType::AIV, "LReLU", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::TLReLU", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE, {OpAttributeKey::scalar},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_BITWISERIGHTSHIFTS, OpCoreType::AIV, "BITWISERIGHTSHIFTS", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB}, {"TileOp::Tbitwiserightshifts", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::ELMWISE,
                 {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OP_ATTR_PREFIX + "reverseOperand",
                  OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_BITWISELEFTSHIFTS, OpCoreType::AIV, "BITWISELEFTSHIFTS", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB}, {"TileOp::Tbitwiseleftshifts", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::ELMWISE,
                 {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OP_ATTR_PREFIX + "reverseOperand",
                  OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis},
                 TileShapeVerifier::Verify);
    RegisterInfo(
        Opcode::OP_SBITWISERIGHTSHIFT, OpCoreType::AIV, "SBITWISERIGHTSHIFT", {MemoryType::MEM_UB},
        {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::TSbitwiserightshift", PIPE_V, PIPE_V, CoreType::AIV},
        OpCalcType::ELMWISE,
        {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OP_ATTR_PREFIX + "reverseOperand",
         OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis, OpAttributeKey::excludeBufferReuse});
    RegisterInfo(
        Opcode::OP_SBITWISELEFTSHIFT, OpCoreType::AIV, "SBITWISELEFTSHIFT", {MemoryType::MEM_UB},
        {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::TSbitwiseleftshift", PIPE_V, PIPE_V, CoreType::AIV},
        OpCalcType::ELMWISE,
        {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OP_ATTR_PREFIX + "reverseOperand",
         OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis, OpAttributeKey::excludeBufferReuse});
    RegisterInfo(
        Opcode::OP_BITWISEANDS, OpCoreType::AIV, "BITWISEANDS", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
        {"TileOp::Tbitwiseands", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
        {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OP_ATTR_PREFIX + "reverseOperand",
         OpAttributeKey::excludeBufferReuse, OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis},
        TileShapeVerifier::Verify);
    RegisterInfo(
        Opcode::OP_BITWISEORS, OpCoreType::AIV, "BITWISEORS", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
        {"TileOp::Tbitwiseors", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
        {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OP_ATTR_PREFIX + "reverseOperand",
         OpAttributeKey::excludeBufferReuse, OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis},
        TileShapeVerifier::Verify);
    RegisterInfo(
        Opcode::OP_BITWISEXORS, OpCoreType::AIV, "BITWISEXORS", {MemoryType::MEM_UB},
        {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Tbitwisexors", PIPE_V, PIPE_V, CoreType::AIV},
        OpCalcType::ELMWISE,
        {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OpAttributeKey::excludeBufferReuse,
         OP_ATTR_PREFIX + "reverseOperand", OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis},
        TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_REMS, OpCoreType::AIV, "REMS", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::TRemainderS", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::ELMWISE,
                 {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OpAttributeKey::inputCombineAxis,
                  OpAttributeKey::outputCombineAxis, OpAttributeKey::excludeBufferReuse},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_REMRS, OpCoreType::AIV, "REMRS", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::TRemainderS", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::ELMWISE,
                 {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OpAttributeKey::inputCombineAxis,
                  OpAttributeKey::outputCombineAxis, OpAttributeKey::excludeBufferReuse},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_GCDS, OpCoreType::AIV, "GCDS", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::TGcds", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::ELMWISE,
                 {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OP_ATTR_PREFIX + "reverseOperand",
                  OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_S_ADDS, OpCoreType::AIV, "S_ADDS", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::TSadds", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::ELMWISE,
                 {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OpAttributeKey::inputCombineAxis,
                  OpAttributeKey::outputCombineAxis});
    RegisterInfo(Opcode::OP_S_SUBS, OpCoreType::AIV, "S_SUBS", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::TSsubs", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::ELMWISE,
                 {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OpAttributeKey::inputCombineAxis,
                  OpAttributeKey::outputCombineAxis});
    RegisterInfo(Opcode::OP_S_MULS, OpCoreType::AIV, "S_MULS", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::TSmuls", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::ELMWISE,
                 {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OpAttributeKey::inputCombineAxis,
                  OpAttributeKey::outputCombineAxis});
    RegisterInfo(Opcode::OP_S_DIVS, OpCoreType::AIV, "S_DIVS", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::TSdivs", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::ELMWISE,
                 {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OpAttributeKey::inputCombineAxis,
                  OpAttributeKey::outputCombineAxis});
    RegisterInfo(Opcode::OP_S_MAXS, OpCoreType::AIV, "S_MAXS", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::TSmaxs", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::ELMWISE,
                 {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OpAttributeKey::inputCombineAxis,
                  OpAttributeKey::outputCombineAxis});
    RegisterInfo(Opcode::OP_S_MINS, OpCoreType::AIV, "S_MINS", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::TSmins", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::ELMWISE,
                 {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OpAttributeKey::inputCombineAxis,
                  OpAttributeKey::outputCombineAxis});
    RegisterInfo(Opcode::OP_MOD, OpCoreType::AIV, "MOD", {MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Tmod", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::BROADCAST,
                 {OpAttributeKey::inputCombineAxis, OpAttributeKey::excludeBufferReuse}, TileShapeVerifier::Verify);
    RegisterInfo(
        Opcode::OP_MODS, OpCoreType::AIV, "MODS", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
        {"TileOp::Tmods", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
        {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OP_ATTR_PREFIX + "reverseOperand",
         OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis, OpAttributeKey::excludeBufferReuse},
        TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_EXPANDEXPDIF, OpCoreType::AIV, "EXPANDEXPDIF", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB}, {"TileOp::TExpandExpDif", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
                 {OpAttributeKey::inputCombineAxis}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_FLOORDIV, OpCoreType::AIV, "FLOORDIV", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::TFloorDiv", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::BROADCAST, {OpAttributeKey::inputCombineAxis, OpAttributeKey::excludeBufferReuse},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_FLOORDIVS, OpCoreType::AIV, "FLOORDIVS", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::TFloorDivS", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::BROADCAST, {OpAttributeKey::inputCombineAxis, OpAttributeKey::excludeBufferReuse},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_AXPY, OpCoreType::AIV, "AXPY", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB}, {"TileOp::TAXPY", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::BROADCAST,
                 {OpAttributeKey::scalar, OpAttributeKey::inplaceInfo, OpAttributeKey::inputCombineAxis},
                 TileShapeVerifier::Verify);
    RegisterInfo(
        Opcode::OP_INTERLEAVE, OpCoreType::AIV, "INTERLEAVE", {MemoryType::MEM_UB, MemoryType::MEM_UB},
        {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::TInterleave", PIPE_V, PIPE_V, CoreType::AIV},
        OpCalcType::ELMWISE,
        {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis, OpAttributeKey::excludeBufferReuse},
        TileShapeVerifier::Verify);
    RegisterInfo(
        Opcode::OP_DEINTERLEAVE, OpCoreType::AIV, "DEINTERLEAVE", {MemoryType::MEM_UB, MemoryType::MEM_UB},
        {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::TDeInterleave", PIPE_V, PIPE_V, CoreType::AIV},
        OpCalcType::ELMWISE,
        {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis, OpAttributeKey::excludeBufferReuse},
        TileShapeVerifier::Verify);
    RegisterInfo(
        Opcode::OP_DEINTERLEAVE_SINGLE, OpCoreType::AIV, "DEINTERLEAVE_SINGLE", {MemoryType::MEM_UB},
        {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::TDeInterleave", PIPE_V, PIPE_V, CoreType::AIV},
        OpCalcType::ELMWISE,
        {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis, OpAttributeKey::excludeBufferReuse},
        TileShapeVerifier::Verify);
}

void OpcodeManager::RegisterVectorUnary()
{
    RegisterInfo(Opcode::OP_EXP, OpCoreType::AIV, "EXP", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Texp", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
                 {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_EXPM1, OpCoreType::AIV, "EXPM1", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Texpm1", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::ELMWISE, {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_NEG, OpCoreType::AIV, "NEG", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Tneg", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
                 {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis}, TileShapeVerifier::Verify);
    RegisterInfo(
        Opcode::OP_RECIPROCAL, OpCoreType::AIV, "RECIPROCAL", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
        {"TileOp::Trec", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
        {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis, OpAttributeKey::excludeBufferReuse});
    RegisterInfo(Opcode::OP_RSQRT, OpCoreType::AIV, "RSQRT", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Trsqrt", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
                 {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_RELU, OpCoreType::AIV, "RELU", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Trelu", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
                 {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_PAD, OpCoreType::AIV, "PAD", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Tpad", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
                 {OpAttributeKey::scalar, OP_ATTR_PREFIX + "pad_right", OP_ATTR_PREFIX + "pad_bottom",
                  OpAttributeKey::excludeBufferReuse},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_FILLPAD, OpCoreType::AIV, "FILLPAD", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::TFillPad", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
                 {OpAttributeKey::scalar, OpAttributeKey::excludeBufferReuse}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_SQRT, OpCoreType::AIV, "SQRT", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Tsqrt", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
                 {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_CEIL, OpCoreType::AIV, "CEIL", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Tceil", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
                 {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_FLOOR, OpCoreType::AIV, "FLOOR", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Tfloor", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
                 {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_TRUNC, OpCoreType::AIV, "TRUNC", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Ttrunc", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
                 {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_ROUND, OpCoreType::AIV, "ROUND", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Tround", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::ELMWISE, {OP_ATTR_PREFIX + "powDecimals", OpAttributeKey::excludeBufferReuse},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_BITWISENOT, OpCoreType::AIV, "BITWISENOT", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Tbitwisenot", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
                 {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_ABS, OpCoreType::AIV, "ABS", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Tabs", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
                 {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_LN, OpCoreType::AIV, "LN", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Tln", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
                 {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_ISFINITE, OpCoreType::AIV, "ISFINITE", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Tisfinite", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::ELMWISE, {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_ISNAN, OpCoreType::AIV, "ISNAN", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::TisNan", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::ELMWISE,
                 {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis, OpAttributeKey::rowPad},
                 TileShapeVerifier::Verify);
    RegisterInfo(
        Opcode::OP_ATAN, OpCoreType::AIV, "ATAN", {MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB},
        {"TileOp::Tatan", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
        {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis, OpAttributeKey::excludeBufferReuse},
        TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_SIGN, OpCoreType::AIV, "SIGN", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Tsign", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::ELMWISE, {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_SIGNBIT, OpCoreType::AIV, "SIGNBIT", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Tsignbit", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::ELMWISE, {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_TAN, OpCoreType::AIV, "TAN", {MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {"TileOp::Ttan", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
                 {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis}, TileShapeVerifier::Verify);
    RegisterInfo(
        Opcode::OP_ERF, OpCoreType::AIV, "ERF", {MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB},
        {"TileOp::TErf", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
        {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis, OpAttributeKey::excludeBufferReuse},
        TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_PACK, OpCoreType::AIV, "PACK", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::TPack", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
                 {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis}, TileShapeVerifier::Verify);
    RegisterInfo(
        Opcode::OP_UNPACK, OpCoreType::AIV, "UNPACK", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
        {"TileOp::TUnPack", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
        {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis, OpAttributeKey::excludeBufferReuse},
        TileShapeVerifier::Verify);
    RegisterInfo(
        Opcode::OP_SIN, OpCoreType::AIV, "SIN", {MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB},
        {"TileOp::TSin", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
        {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis, OpAttributeKey::excludeBufferReuse},
        TileShapeVerifier::Verify);
    RegisterInfo(
        Opcode::OP_COS, OpCoreType::AIV, "COS", {MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB},
        {"TileOp::TCos", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
        {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis, OpAttributeKey::excludeBufferReuse},
        TileShapeVerifier::Verify);
    RegisterInfo(
        Opcode::OP_SINH, OpCoreType::AIV, "SINH", {MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB},
        {"TileOp::TSinh", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
        {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis, OpAttributeKey::excludeBufferReuse},
        TileShapeVerifier::Verify);
    RegisterInfo(
        Opcode::OP_COSH, OpCoreType::AIV, "COSH", {MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB},
        {"TileOp::TCosh", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
        {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis, OpAttributeKey::excludeBufferReuse},
        TileShapeVerifier::Verify);
    RegisterInfo(
        Opcode::OP_ASINH, OpCoreType::AIV, "ASINH", {MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB},
        {"TileOp::TASinh", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
        {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis, OpAttributeKey::excludeBufferReuse},
        TileShapeVerifier::Verify);
    RegisterInfo(
        Opcode::OP_ACOSH, OpCoreType::AIV, "ACOSH", {MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB},
        {"TileOp::TACosh", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
        {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis, OpAttributeKey::excludeBufferReuse},
        TileShapeVerifier::Verify);
    RegisterInfo(
        Opcode::OP_ERFC, OpCoreType::AIV, "ERFC", {MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB},
        {"TileOp::TErfc", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
        {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis, OpAttributeKey::excludeBufferReuse},
        TileShapeVerifier::Verify);
    RegisterInfo(
        Opcode::OP_TANH, OpCoreType::AIV, "TANH", {MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB},
        {"TileOp::Ttanh", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
        {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis, OpAttributeKey::excludeBufferReuse},
        TileShapeVerifier::Verify);
    RegisterInfo(
        Opcode::OP_ASIN, OpCoreType::AIV, "ASIN", {MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB},
        {"TileOp::TAsin", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
        {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis, OpAttributeKey::excludeBufferReuse},
        TileShapeVerifier::Verify);
    RegisterInfo(
        Opcode::OP_ACOS, OpCoreType::AIV, "ACOS", {MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB},
        {"TileOp::TAcos", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
        {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis, OpAttributeKey::excludeBufferReuse},
        TileShapeVerifier::Verify);
    RegisterInfo(
        Opcode::OP_ATANH, OpCoreType::AIV, "ATANH", {MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB},
        {"TileOp::TAtanh", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
        {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis, OpAttributeKey::excludeBufferReuse},
        TileShapeVerifier::Verify);
}

void OpcodeManager::RegisterVectorSort()
{
    RegisterInfo(
        Opcode::OP_TOPK, OpCoreType::ANY, "TOPK", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
        {"TileOp::MrgSort", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER,
        {OP_ATTR_PREFIX + "axis", OP_ATTR_PREFIX + "order", OP_ATTR_PREFIX + "kvalue", OpAttributeKey::topkAlgo},
        TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_BITSORT, OpCoreType::ANY, "BITSORT", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::BitSort", PIPE_S, PIPE_V, CoreType::AIV},
                 OpCalcType::OTHER, {OP_ATTR_PREFIX + "axis", OP_ATTR_PREFIX + "order", OP_ATTR_PREFIX + "offset"},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_MRGSORT, OpCoreType::ANY, "MRGSORT", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::MrgSort", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::OTHER, {OP_ATTR_PREFIX + "axis", OP_ATTR_PREFIX + "mergeSize", OP_ATTR_PREFIX + "kvalue"},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_ARGSORT, OpCoreType::ANY, "ARGSORT", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::ArgSort", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER,
                 {OP_ATTR_PREFIX + "axis", OP_ATTR_PREFIX + "order", OP_ATTR_PREFIX + "kvalue"});
    RegisterInfo(Opcode::OP_EXTRACT, OpCoreType::ANY, "EXTRACT", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Extract", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER,
                 {OP_ATTR_PREFIX + "kvalue", OP_ATTR_PREFIX + "order", OP_ATTR_PREFIX + "makeMode"},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_TILEDMRGSORT, OpCoreType::ANY, "TILEDMRGSORT",
                 {MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::TiledMrgSort", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::OTHER, {OP_ATTR_PREFIX + "validBit", OP_ATTR_PREFIX + "kvalue"});
    RegisterInfo(Opcode::OP_TWOTILEMRGSORT, OpCoreType::ANY, "TWOTILEMRGSORT", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB}, {"TileOp::TwoTileMrgSort", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER,
                 {OpAttributeKey::excludeBufferReuse, OP_ATTR_PREFIX + "firstShape"});
    RegisterInfo(Opcode::OP_EXTRACT_SINGLE, OpCoreType::ANY, "EXTRACTSINGLE", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB}, {"TileOp::ExtractSingle", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER,
                 {OP_ATTR_PREFIX + "order", OP_ATTR_PREFIX + "makeMode"});
    RegisterInfo(Opcode::OP_SORT_UB, OpCoreType::ANY, "SORTUB", {MemoryType::MEM_DEVICE_DDR},
                 {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR},
                 {"TileOp::SortUB", PIPE_S, PIPE_V, CoreType::AIV}, OpCalcType::OTHER,
                 {OP_ATTR_PREFIX + "axis", OP_ATTR_PREFIX + "order"});
    RegisterInfo(Opcode::OP_RADIX_SELECT, OpCoreType::ANY, "RADIX_SELECT", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {"TileOp::RadixSelect", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER,
                 {OP_ATTR_PREFIX + "order", OP_ATTR_PREFIX + "kvalue", OpAttributeKey::excludeBufferReuse});

    // parallel sort
    RegisterInfo(Opcode::OP_SORT, OpCoreType::AIV, "SORT", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {"TileOp::Sort", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER,
                 {OpAttributeKey::inplaceInfo, OP_ATTR_PREFIX + "start_index", OP_ATTR_PREFIX + "order"});
    RegisterInfo(Opcode::OP_COMPARE_SWAP, OpCoreType::AIV, "COMP_SWAP",
                 {MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {"TileOp::CompareAndSwap", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER,
                 {OpAttributeKey::inplaceInfo, OP_ATTR_PREFIX + "order"});
    RegisterInfo(Opcode::OP_EXP2, OpCoreType::AIV, "EXP2", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {"TileOp::Texp2", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
                 {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_MERGE, OpCoreType::AIV, "MERGE", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {"TileOp::Merge", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER,
                 {OpAttributeKey::inplaceInfo, OP_ATTR_PREFIX + "order", OP_ATTR_PREFIX + "full_sort"});
    // topk for DS3.2-Day0
    RegisterInfo(Opcode::OP_TOPK_SORT, OpCoreType::AIV, "TOPK_SORT", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::TopKSort", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::OTHER, {OP_ATTR_PREFIX + "start_index"});
    RegisterInfo(Opcode::OP_TOPK_MERGE, OpCoreType::AIV, "TOPK_MERGE", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::TopKMerge", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER,
                 {OpAttributeKey::excludeBufferReuse, OP_ATTR_PREFIX + "merge_size"});
    RegisterInfo(Opcode::OP_TOPK_EXTRACT, OpCoreType::AIV, "TOPK_EXTRACT", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::TopKExtract", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER,
                 {OP_ATTR_PREFIX + "is_index", OP_ATTR_PREFIX + "k"});
    RegisterInfo(Opcode::OP_FAKE_TRANS, OpCoreType::ANY, "FAKE_TRANS", {}, {},
                 {"FAKE_TRANS", PIPE_MTE3, PIPE_MTE3, CoreType::AIV}, OpCalcType::MOVE_OUT, {});
}

void OpcodeManager::RegisterVectorReduction()
{
    RegisterInfo(Opcode::OP_PAIRMAX, OpCoreType::AIV, "PAIRMAX", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB}, {"TileOp::Tmaxpair", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::BROADCAST,
                 {OpAttributeKey::inputCombineAxis, OpAttributeKey::excludeBufferReuse});
    RegisterInfo(Opcode::OP_PAIRMIN, OpCoreType::AIV, "PAIRMIN", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB}, {"TileOp::Tminpair", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::BROADCAST,
                 {OpAttributeKey::inputCombineAxis, OpAttributeKey::excludeBufferReuse});
    RegisterInfo(Opcode::OP_PAIRSUM, OpCoreType::AIV, "PAIRSUM", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB}, {"TileOp::Taddpair", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::BROADCAST,
                 {OpAttributeKey::inputCombineAxis, OpAttributeKey::excludeBufferReuse});
    RegisterInfo(Opcode::OP_PAIRARGMAX, OpCoreType::AIV, "PAIRARGMAX",
                 {MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Tpairargmax", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::BROADCAST, {OpAttributeKey::inputCombineAxis, OpAttributeKey::excludeBufferReuse});
    RegisterInfo(Opcode::OP_PAIRARGMIN, OpCoreType::AIV, "PAIRARGMIN",
                 {MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Tpairargmin", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::BROADCAST, {OpAttributeKey::inputCombineAxis, OpAttributeKey::excludeBufferReuse});
    RegisterInfo(Opcode::OP_PAIRPROD, OpCoreType::AIV, "PAIRPROD", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB}, {"TileOp::Tmulpair", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::BROADCAST,
                 {OpAttributeKey::inputCombineAxis, OpAttributeKey::excludeBufferReuse});
    RegisterInfo(Opcode::OP_ROWMAX_SINGLE, OpCoreType::AIV, "ROWMAX_SINGLE", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Trowmaxsingle", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::REDUCE, {OP_ATTR_PREFIX + "AXIS", OpAttributeKey::excludeBufferReuse},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_ROWMIN_SINGLE, OpCoreType::AIV, "ROWMIN_SINGLE", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Trowminsingle", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::REDUCE, {OP_ATTR_PREFIX + "AXIS", OpAttributeKey::excludeBufferReuse},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_ROWSUM_SINGLE, OpCoreType::AIV, "ROWSUM_SINGLE", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Trowsumsingle", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::REDUCE, {OP_ATTR_PREFIX + "AXIS", OpAttributeKey::excludeBufferReuse},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_ROWPROD_SINGLE, OpCoreType::AIV, "ROWPROD_SINGLE", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Trowprodsingle", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::REDUCE, {OP_ATTR_PREFIX + "AXIS", OpAttributeKey::excludeBufferReuse},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_ROWARGMAX_SINGLE, OpCoreType::AIV, "ROWARGMAX_SINGLE", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Trowargmaxsingle", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::REDUCE, {OP_ATTR_PREFIX + "AXIS", OpAttributeKey::excludeBufferReuse},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_ROWARGMIN_SINGLE, OpCoreType::AIV, "ROWARGMIN_SINGLE", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Trowargminsingle", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::REDUCE, {OP_ATTR_PREFIX + "AXIS", OpAttributeKey::excludeBufferReuse},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_ROWARGMAXWITHVALUE_SINGLE, OpCoreType::AIV, "ROWARGMAXWITHVALUE_SINGLE",
                 {MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {"TileOp::Trowargmaxwithvaluesingle", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::REDUCE,
                 {OP_ATTR_PREFIX + "AXIS", OpAttributeKey::excludeBufferReuse}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_ROWARGMINWITHVALUE_SINGLE, OpCoreType::AIV, "ROWARGMINWITHVALUE_SINGLE",
                 {MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {"TileOp::Trowargminwithvaluesingle", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::REDUCE,
                 {OP_ATTR_PREFIX + "AXIS", OpAttributeKey::excludeBufferReuse}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_ROWARGMAXWITHVALUE_LINE, OpCoreType::AIV, "ROWARGMAXWITHVALUE_LINE", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {"TileOp::Trowargmaxwithvalueline", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::REDUCE,
                 {OP_ATTR_PREFIX + "AXIS", OpAttributeKey::excludeBufferReuse}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_ROWARGMINWITHVALUE_LINE, OpCoreType::AIV, "ROWARGMINWITHVALUE_LINE", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {"TileOp::Trowargminwithvalueline", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::REDUCE,
                 {OP_ATTR_PREFIX + "AXIS", OpAttributeKey::excludeBufferReuse}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_ONLINE_SOFTMAX, OpCoreType::AIV, "ONLINE_SOFTMAX", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {"TileOp::TOnlineSoftmax", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::BROADCAST,
                 {OpAttributeKey::scalar, OpAttributeKey::excludeBufferReuse}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_ONLINE_SOFTMAX_UPDATE, OpCoreType::AIV, "ONLINE_SOFTMAX_UPDATE",
                 {MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB,
                  MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {"TileOp::TOnlineSoftmaxUpdate", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::BROADCAST,
                 {OpAttributeKey::excludeBufferReuse}, TileShapeVerifier::Verify);

    RegisterInfo(Opcode::OP_ROWSUMLINE, OpCoreType::AIV, "ROWSUMLINE", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Trowsumline", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::REDUCE, {}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_ROWMAXLINE, OpCoreType::AIV, "ROWMAXLINE", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB}, {"TileOp::Trowmaxline", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::REDUCE, {},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_ROWMINLINE, OpCoreType::AIV, "ROWMINLINE", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB}, {"TileOp::Trowminline", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::REDUCE, {},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_ROWPRODLINE, OpCoreType::AIV, "ROWPRODLINE", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Trowprodline", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::REDUCE, {}, TileShapeVerifier::Verify);

    RegisterInfo(Opcode::OP_ROWMAX, OpCoreType::AIV, "ROWMAX", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Trowmaxexpand", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::REDUCE,
                 {OpAttributeKey::excludeBufferReuse});
    RegisterInfo(Opcode::OP_ROWSUM, OpCoreType::AIV, "ROWSUM", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Treducesum", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::REDUCE);
    RegisterInfo(Opcode::OP_ROWEXPMAX, OpCoreType::AIV, "ROWEXPMAX", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Trowmaxexpand", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::REDUCE,
                 {OpAttributeKey::excludeBufferReuse});
    RegisterInfo(Opcode::OP_ROWEXPSUM, OpCoreType::AIV, "ROWEXPSUM", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Trowsumexpand", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::REDUCE,
                 {OpAttributeKey::excludeBufferReuse});
}

void OpcodeManager::RegisterVectorQuant()
{
    RegisterInfo(Opcode::OP_QUANTIZE_SYM, OpCoreType::AIV, "QUANTIZE_SYM", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::TQuant", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::OTHER,
                 {OpAttributeKey::excludeBufferReuse, OP_ATTR_PREFIX + "axis", OP_ATTR_PREFIX + "dtype"},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_QUANTIZE_ASYM, OpCoreType::AIV, "QUANTIZE_ASYM",
                 {MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {"TileOp::TQuant", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER,
                 {OpAttributeKey::excludeBufferReuse, OP_ATTR_PREFIX + "axis", OP_ATTR_PREFIX + "dtype"},
                 TileShapeVerifier::Verify);
    // Dequantization: INT8/INT16 -> FP32
    // Always 4 params: dst, src, scale, offset (symmetric: offset=0)
    RegisterInfo(Opcode::OP_DEQUANTIZE, OpCoreType::AIV, "DEQUANTIZE",
                 {MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::TDequant", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER,
                 {OpAttributeKey::excludeBufferReuse, OP_ATTR_PREFIX + "axis"}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_QUANT_MX, OpCoreType::AIV, "QUANT_MX", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {"TileOp::TQuantMX", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER,
                 {OpAttributeKey::excludeBufferReuse, OpAttributeKey::mxQuantMode, OpAttributeKey::mxQuantAxis,
                  OpAttributeKey::mxQuantPerformanceMode});
}

void OpcodeManager::RegisterVector()
{
    RegisterVectorBinary();
    RegisterVectorUnary();
    RegisterVectorSort();
    RegisterVectorReduction();
    RegisterVectorQuant();
    RegisterInfo(Opcode::OP_CAST, OpCoreType::AIV, "CAST", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Tcast", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::CAST, {OP_ATTR_PREFIX + "mode", OP_ATTR_PREFIX + "satmode"}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_LOGICALNOT, OpCoreType::AIV, "LOGICALNOT", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::TlogicalNot", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::ELMWISE, {}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_BRCB, OpCoreType::AIV, "BRCB", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Tbrcb", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
                 {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_TRANSPOSE_MOVEIN, OpCoreType::ANY, "TRANSPOSE_MOVEIN", {MemoryType::MEM_DEVICE_DDR},
                 {MemoryType::MEM_UB}, {"TileOp::TtransposeMoveIn", PIPE_MTE2, PIPE_MTE2, CoreType::AIV},
                 OpCalcType::MOVE_IN, {OP_ATTR_PREFIX + "shape"}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_TRANSPOSE_MOVEOUT, OpCoreType::ANY, "TRANSPOSE_MOVEOUT", {MemoryType::MEM_UB},
                 {MemoryType::MEM_DEVICE_DDR}, {"TileOp::TtransposeMoveOut", PIPE_MTE3, PIPE_MTE3, CoreType::AIV},
                 OpCalcType::MOVE_OUT, {OP_ATTR_PREFIX + "shape"}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_TRANSPOSE_VNCHWCONV, OpCoreType::ANY, "TRANSPOSE_VNCHWCONV", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {"TileOp::Ttranspose_vnchwconv", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::MOVE_LOCAL,
                 {OP_ATTR_PREFIX + "shape", OpAttributeKey::excludeBufferReuse}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_NCHW2NC1HWC0, OpCoreType::AIV, "NCHW2NC1HWC0", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {"TileOp::TtransData_nvhw2nc1hwc0", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER,
                 {OpAttributeKey::transDataOffset, OP_ATTR_PREFIX + "group", OpAttributeKey::excludeBufferReuse});
    RegisterInfo(Opcode::OP_NCHW2Fractal_Z, OpCoreType::AIV, "NCHW2Fractal_Z", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {"TileOp::TtransData_nchw2fractal_z", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER,
                 {OpAttributeKey::transDataOffset, OP_ATTR_PREFIX + "group", OpAttributeKey::excludeBufferReuse});
    RegisterInfo(Opcode::OP_NC1HWC02NCHW, OpCoreType::AIV, "NC1HWC02NCHW", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {"TileOp::TtransData_nc1hwc02nchw", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER,
                 {OpAttributeKey::transDataOffset, OP_ATTR_PREFIX + "group", OpAttributeKey::excludeBufferReuse});
    RegisterInfo(Opcode::OP_NCDHW2NDC1HWC0, OpCoreType::AIV, "NCDHW2NDC1HWC0", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {"TileOp::TtransData_ncdhw2ndc1hwc0", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER,
                 {OpAttributeKey::transDataOffset, OP_ATTR_PREFIX + "group", OpAttributeKey::excludeBufferReuse});
    RegisterInfo(Opcode::OP_NCDHW2FRACTAL_Z_3D, OpCoreType::AIV, "NCDHW2FRACTAL_Z_3D", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {"TileOp::TtransData_ncdhw2fractal_z_3d", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER,
                 {OpAttributeKey::transDataOffset, OP_ATTR_PREFIX + "group", OpAttributeKey::excludeBufferReuse});
    RegisterInfo(Opcode::OP_NDC1HWC02NCDHW, OpCoreType::AIV, "NDC1HWC02NCDHW", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {"TileOp::TtransData_ndc1hwc02ncdhw", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER,
                 {OpAttributeKey::transDataOffset, OP_ATTR_PREFIX + "group", OpAttributeKey::excludeBufferReuse});
    RegisterInfo(Opcode::OP_FractalZ2NCHW, OpCoreType::AIV, "FractalZ2NCHW", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {"TileOp::TtransData_fractalz2nchw", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::OTHER,
                 {OpAttributeKey::transDataOffset, OP_ATTR_PREFIX + "group", OpAttributeKey::excludeBufferReuse});
    RegisterInfo(Opcode::OP_FractalZ3D2NCDHW, OpCoreType::AIV, "FractalZ3D2NCDHW", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {"TileOp::TtransData_fractalz3d2ncdhw", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::OTHER,
                 {OpAttributeKey::transDataOffset, OP_ATTR_PREFIX + "group", OpAttributeKey::excludeBufferReuse});
    RegisterInfo(Opcode::OP_PERMUTE, OpCoreType::AIV, "PERMUTE", {MemoryType::MEM_DEVICE_DDR}, {MemoryType::MEM_UB},
                 {"TileOp::TPermute", PIPE_S, PIPE_MTE2, CoreType::AIV}, OpCalcType::OTHER, {OpAttributeKey::perm},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_PERMUTE_ELEMENT, OpCoreType::AIV, "PERMUTE_ELEMENT", {MemoryType::MEM_DEVICE_DDR},
                 {MemoryType::MEM_UB}, {"TileOp::TPermuteElewise", PIPE_S, PIPE_MTE2, CoreType::AIV}, OpCalcType::OTHER,
                 {OpAttributeKey::perm}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_EXPAND, OpCoreType::AIV, "EXPAND", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Texpand", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE, {OpAttributeKey::expandDims},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_ONEHOT, OpCoreType::AIV, "ONEHOT", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Tonehot", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
                 {OP_ATTR_PREFIX + "numClasses", OpAttributeKey::excludeBufferReuse}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_CONCAT, OpCoreType::AIV, "CONCAT", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Tconcat", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
                 {"concat", OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis});
    RegisterInfo(Opcode::OP_COMPACT, OpCoreType::AIV, "COMPACT", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Tcompact", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER);
    RegisterInfo(
        Opcode::OP_POW, OpCoreType::AIV, "POW", {MemoryType::MEM_UB, MemoryType::MEM_UB},
        {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Tpow", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
        {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OpAttributeKey::inputCombineAxis,
         OpAttributeKey::outputCombineAxis, OpAttributeKey::excludeBufferReuse, OpAttributeKey::precisionType});
    RegisterInfo(
        Opcode::OP_POWS, OpCoreType::AIV, "POWS", {MemoryType::MEM_UB, MemoryType::MEM_UB},
        {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Tpows", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
        {OpAttributeKey::scalar, OP_ATTR_PREFIX + "reverseOperand", OpAttributeKey::inputCombineAxis,
         OpAttributeKey::outputCombineAxis, OpAttributeKey::excludeBufferReuse, OpAttributeKey::precisionType});
    RegisterInfo(Opcode::OP_GATHER_FROM_UB, OpCoreType::AIV, "GATHER_FROM_UB", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB}, {"TileOp::TgatherFromUB", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER,
                 {OP_ATTR_PREFIX + "axis", OpAttributeKey::excludeBufferReuse}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_GATHER, OpCoreType::ANY, "GATHER", {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR},
                 {MemoryType::MEM_UB}, {"TileOp::Tgather", PIPE_S, PIPE_MTE2, CoreType::AIV}, OpCalcType::OTHER,
                 {OP_ATTR_PREFIX + "axis"}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_GATHER_ELEMENT, OpCoreType::AIV, "GATHER_ELEMENT",
                 {MemoryType::MEM_UNKNOWN, MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {"TileOp::TgatherElement", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER,
                 {OP_ATTR_PREFIX + "axis", OpAttributeKey::excludeBufferReuse, OP_ATTR_PREFIX + "requires_simt"},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_GATHER_MASK, OpCoreType::AIV, "GATHER_MASK", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::TgatherMask", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER,
                 {OP_ATTR_PREFIX + "patternMode", OpAttributeKey::excludeBufferReuse}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_GATHER_MASK_BUILDIN, OpCoreType::AIV, "GATHER_MASK_BUILDIN", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB}, {"TileOp::TgatherMaskBuildIn", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER,
                 {OP_ATTR_PREFIX + "patternMode", OpAttributeKey::excludeBufferReuse}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_CMP, OpCoreType::AIV, "CMP", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Compare", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::BROADCAST,
                 {OP_ATTR_PREFIX + "cmp_operation", OpAttributeKey::rowPad, OP_ATTR_PREFIX + "cmp_mode"},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_CMPS, OpCoreType::AIV, "CMPS", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Cmps", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::BROADCAST,
                 {OP_ATTR_PREFIX + "cmp_operation", OpAttributeKey::rowPad, OP_ATTR_PREFIX + "cmp_mode",
                  OpAttributeKey::scalar});
    RegisterInfo(Opcode::OP_HYPOT, OpCoreType::AIV, "HYPOT", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Hypot", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::BROADCAST);
    RegisterInfo(
        Opcode::OP_LOG1P, OpCoreType::AIV, "Log1p", {MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB},
        {"TileOp::Log1p", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE,
        {OpAttributeKey::inputCombineAxis, OpAttributeKey::outputCombineAxis, OpAttributeKey::excludeBufferReuse},
        TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_PRELU, OpCoreType::AIV, "PRELU", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::PReLU", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::OTHER,
                 {OP_ATTR_PREFIX + "axis", OpAttributeKey::rowPad, OpAttributeKey::excludeBufferReuse},
                 TileShapeVerifier::Verify);
    RegisterInfo(
        Opcode::OP_SCATTER_ELEMENT, OpCoreType::AIV, "SCATTER_ELEMENT", {MemoryType::MEM_UB, MemoryType::MEM_UB},
        {MemoryType::MEM_UB}, {"TileOp::TscatterElementS", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::OTHER,
        {OP_ATTR_PREFIX + "axis", OpAttributeKey::scalar, OP_ATTR_PREFIX + "scatter_mode"}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_SCATTER, OpCoreType::AIV, "SCATTER",
                 {MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {"TileOp::Tscatter", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::OTHER,
                 {OP_ATTR_PREFIX + "axis", OP_ATTR_PREFIX + "scatter_mode"}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_INDEX_PUT, OpCoreType::ANY, "INDEX_PUT",
                 {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB,
                  MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB},
                 {"TileOp::TIndexPut", PIPE_MTE3, PIPE_MTE3, CoreType::AIV}, OpCalcType::MOVE_OUT,
                 {OpAttributeKey::accumulate, OpAttributeKey::indicesSize, OP_ATTR_PREFIX + "requires_simt"});
    RegisterInfo(Opcode::OP_SCATTER_UPDATE, OpCoreType::ANY, "SCATTER_UPDATE", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB}, {}, OpCalcType::OTHER);
    RegisterInfo(Opcode::OP_SCATTER_SCALAR, OpCoreType::ANY, "SCATTER_SCALAR", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB}, {}, OpCalcType::OTHER);
    RegisterInfo(Opcode::OP_CUM_SUM, OpCoreType::AIV, "CUM_SUM", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::TcumSum", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER,
                 {OP_ATTR_PREFIX + "axis", OP_ATTR_PREFIX + "flag", OpAttributeKey::excludeBufferReuse});
    RegisterInfo(Opcode::OP_CUM_PROD, OpCoreType::AIV, "CUM_PROD", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::TcumProd", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER,
                 {OP_ATTR_PREFIX + "axis", OP_ATTR_PREFIX + "flag", OpAttributeKey::excludeBufferReuse});
    RegisterInfo(Opcode::OP_TRIUL, OpCoreType::AIV, "TRIUL", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::TTriUL", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER,
                 {OpAttributeKey::dynScalar, OpAttributeKey::isUpper, OpAttributeKey::excludeBufferReuse},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_LOGICALAND, OpCoreType::AIV, "LOGICALAND", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::TlogicalAnd", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::ELMWISE, {}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_RANGE, OpCoreType::AIV, "RANGE", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Range", PIPE_S, PIPE_V, CoreType::AIV}, OpCalcType::OTHER,
                 {OP_ATTR_PREFIX + "START", OP_ATTR_PREFIX + "STEP", OpAttributeKey::dynScalar},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_UNIFORM, OpCoreType::AIV, "UNIFORM", {}, {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {"TileOp::TUniform", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER,
                 {OpAttributeKey::vectorScalar, OpAttributeKey::dynScalar, OP_ATTR_PREFIX + "SHAPE"},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_VEC_DUP, OpCoreType::AIV, "VEC_DUP", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Tduplicate", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER,
                 {OpAttributeKey::scalar, OpAttributeKey::dynScalar, OP_ATTR_PREFIX + "shape"},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_UB_COPY_IN, OpCoreType::AIV, "UB_COPY_IN", {MemoryType::MEM_DEVICE_DDR},
                 {MemoryType::MEM_UB}, {"TileOp::UBCopyIn", PIPE_MTE2, PIPE_MTE2, CoreType::AIV}, OpCalcType::MOVE_IN,
                 {OpAttributeKey::excludeBufferReuse});
    RegisterInfo(Opcode::OP_UB_COPY_OUT, OpCoreType::AIV, "UB_COPY_OUT", {MemoryType::MEM_UB},
                 {MemoryType::MEM_DEVICE_DDR}, {"TileOp::UBCopyOut", PIPE_MTE3, PIPE_MTE3, CoreType::AIV},
                 OpCalcType::MOVE_OUT, {OpAttributeKey::excludeBufferReuse});
    RegisterInfo(Opcode::OP_INDEX_ADD_UB, OpCoreType::AIV, "INDEX_ADD_UB",
                 {MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {"TileOp::TIndexAddUB", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER,
                 {OP_ATTR_PREFIX + "axis", OpAttributeKey::scalar}, TileShapeVerifier::Verify);
    RegisterInfo(
        Opcode::OP_INDEX_ADD, OpCoreType::ANY, "INDEX_ADD",
        {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB},
        {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB}, {"TileOp::TIndexAdd", PIPE_MTE3, PIPE_MTE3, CoreType::AIV},
        OpCalcType::MOVE_OUT,
        {OP_ATTR_PREFIX + "axis", OpAttributeKey::scalar, OpAttributeKey::inplaceIdx, OP_ATTR_PREFIX + "requires_simt"},
        TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_WHERE_TT, OpCoreType::AIV, "WHERE_TT",
                 {MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB}, {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {"TileOp::Where_TT", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::ELMWISE, {},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_WHERE_TS, OpCoreType::AIV, "WHERE_TS", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Where_TS", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::ELMWISE, {OpAttributeKey::scalar}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_WHERE_ST, OpCoreType::AIV, "WHERE_ST", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Where_ST", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::ELMWISE, {OpAttributeKey::scalar}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_WHERE_SS, OpCoreType::AIV, "WHERE_SS", {MemoryType::MEM_UB},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::Where_SS", PIPE_V, PIPE_V, CoreType::AIV},
                 OpCalcType::ELMWISE, {OpAttributeKey::vectorScalar}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_INDEX_OUTCAST, OpCoreType::ANY, "INDEX_OUTCAST",
                 {MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR}, {MemoryType::MEM_DEVICE_DDR},
                 {"TileOp::TIndexoutcast", PIPE_MTE3, PIPE_MTE3, CoreType::AIV}, OpCalcType::MOVE_OUT,
                 {"axis", OpAttributeKey::inputCombineAxis, OpAttributeKey::cacheMode, OpAttributeKey::panzBlockSize},
                 TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_GATHER_IN_UB, OpCoreType::ANY, "GATHER_IN_UB",
                 {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR},
                 {MemoryType::MEM_UB}, {"TileOp::GatherInUB", PIPE_MTE2, PIPE_MTE2, CoreType::AIV}, OpCalcType::OTHER);
    RegisterInfo(Opcode::OP_MAX_POOL, OpCoreType::AIV, "MAX_POOL", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Tmaxpool", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER,
                 {OpAttributeKey::excludeBufferReuse});
    RegisterInfo(Opcode::OP_REDUCE_ACC, OpCoreType::GMATOMIC, "REDUCE_ACC", {MEM_DEVICE_DDR, MEM_DEVICE_DDR},
                 {MEM_DEVICE_DDR}, {}, OpCalcType::OTHER, {OP_ATTR_PREFIX + "atomic_add"});
}

void OpcodeManager::RegisterCube()
{
    std::vector<std::string> convAttrStrList{
        ConvOpAttributeKey::cin,
        ConvOpAttributeKey::cout,
        ConvOpAttributeKey::paddingLeft,
        ConvOpAttributeKey::paddingTop,
        ConvOpAttributeKey::paddingRight,
        ConvOpAttributeKey::paddingBottom,
        ConvOpAttributeKey::strideh,
        ConvOpAttributeKey::stridew,
        ConvOpAttributeKey::hposX,
        ConvOpAttributeKey::hsteP,
        ConvOpAttributeKey::wposX,
        ConvOpAttributeKey::wstep,
        ConvOpAttributeKey::hoffsetY,
        ConvOpAttributeKey::woffsetY,
        ConvOpAttributeKey::reluType,
        ConvOpAttributeKey::reluAlpha,
        ConvOpAttributeKey::clearFlag,
        ConvOpAttributeKey::hasAccFlag,
        ConvOpAttributeKey::hasEltFlag,
        ConvOpAttributeKey::hasBiasFlag,
        ConvOpAttributeKey::eltBrcbFlag,
        ConvOpAttributeKey::eltMode,
        ConvOpAttributeKey::fmapSrcNum,
        FixpOpAttributeKey::hasQuantPreVector,
        FixpOpAttributeKey::hasQuantPostVector,
        FixpOpAttributeKey::hasAntiqVector,
        FixpOpAttributeKey::quantPreScalar,
        FixpOpAttributeKey::quantPostScalar,
        FixpOpAttributeKey::antiqScalar,
    };

    RegisterInfo(Opcode::OP_A_MUL_B, OpCoreType::AIC, "A_MUL_B",
                 {MemoryType::MEM_L0A, MemoryType::MEM_L0B, MemoryType::MEM_L0AMX, MemoryType::MEM_L0BMX,
                  MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_BT, MemoryType::MEM_FIX_QUANT_PRE},
                 {MemoryType::MEM_L0C}, {"TileOp::Tmad", PIPE_M, PIPE_M, CoreType::AIC}, OpCalcType::MATMUL);
    RegisterInfo(Opcode::OP_A_MULACC_B, OpCoreType::AIC, "A_MULACC_B",
                 {MemoryType::MEM_L0A, MemoryType::MEM_L0B, MemoryType::MEM_L0C, MemoryType::MEM_L0AMX,
                  MemoryType::MEM_L0BMX, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_BT, MemoryType::MEM_FIX_QUANT_PRE},
                 {MemoryType::MEM_L0C}, {"TileOp::Tmad", PIPE_M, PIPE_M, CoreType::AIC}, OpCalcType::MATMUL);
    RegisterInfo(Opcode::OP_A_MUL_BT, OpCoreType::AIC, "A_MUL_Bt",
                 {MemoryType::MEM_L0A, MemoryType::MEM_L0B, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_BT,
                  MemoryType::MEM_FIX_QUANT_PRE},
                 {MemoryType::MEM_L0C}, {"TileOp::Tmad", PIPE_M, PIPE_M, CoreType::AIC}, OpCalcType::MATMUL);
    RegisterInfo(Opcode::OP_A_MULACC_BT, OpCoreType::AIC, "A_MULACC_Bt",
                 {MemoryType::MEM_L0A, MemoryType::MEM_L0B, MemoryType::MEM_L0C, MemoryType::MEM_DEVICE_DDR,
                  MemoryType::MEM_BT, MemoryType::MEM_FIX_QUANT_PRE},
                 {MemoryType::MEM_L0C}, {"TileOp::Tmad", PIPE_M, PIPE_M, CoreType::AIC}, OpCalcType::MATMUL);
    RegisterInfo(Opcode::OP_AT_MUL_B, OpCoreType::AIC, "At_MUL_B",
                 {MemoryType::MEM_L0A, MemoryType::MEM_L0B, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_BT,
                  MemoryType::MEM_FIX_QUANT_PRE},
                 {MemoryType::MEM_L0C}, {"TileOp::Tmad", PIPE_M, PIPE_M, CoreType::AIC}, OpCalcType::MATMUL);
    RegisterInfo(Opcode::OP_AT_MUL_BT, OpCoreType::AIC, "At_MUL_Bt",
                 {MemoryType::MEM_L0A, MemoryType::MEM_L0B, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_BT,
                  MemoryType::MEM_FIX_QUANT_PRE},
                 {MemoryType::MEM_L0C}, {"TileOp::Tmad", PIPE_M, PIPE_M, CoreType::AIC}, OpCalcType::MATMUL);

    RegisterInfo(Opcode::OP_CONV2D, OpCoreType::AIC, "CONV2D", {}, {}, {"CUBE_CONV2D", PIPE_M, PIPE_M, CoreType::AIC},
                 OpCalcType::CONV, convAttrStrList, nullptr,
                 {{"DAV_2201",
                   {{TileOpFormat::TILEOP_NC1HWC0, TileOpFormat::TILEOP_FRACTAL_Z, TileOpFormat::TILEOP_ND},
                    {TileOpFormat::TILEOP_NC1HWC0}}}});
    RegisterInfo(Opcode::OP_CONV3D, OpCoreType::AIC, "CONV3D", {}, {}, {"CUBE_CONV3D", PIPE_M, PIPE_M, CoreType::AIC},
                 OpCalcType::CONV, convAttrStrList, nullptr,
                 {{"DAV_2201",
                   {{TileOpFormat::TILEOP_NDC1HWC0, TileOpFormat::TILEOP_FRACTAL_Z_3D, TileOpFormat::TILEOP_ND},
                    {TileOpFormat::TILEOP_NDC1HWC0}}}});
    RegisterInfo(Opcode::OP_CONV_BP_INPUT_2D, OpCoreType::AIC, "CONV_BP_INPUT_2D", {}, {},
                 {"CUBE_CONV_BP_INPUT_2D", PIPE_M, PIPE_M, CoreType::AIC}, OpCalcType::CONV, convAttrStrList, nullptr,
                 {{"DAV_2201",
                   {{TileOpFormat::TILEOP_NC1HWC0, TileOpFormat::TILEOP_FRACTAL_Z, TileOpFormat::TILEOP_ND},
                    {TileOpFormat::TILEOP_NC1HWC0}}}});
    RegisterInfo(Opcode::OP_CONV_BP_INPUT_3D, OpCoreType::AIC, "CONV_BP_INPUT_3D", {}, {},
                 {"CUBE_CONV_BP_INPUT_3D", PIPE_M, PIPE_M, CoreType::AIC}, OpCalcType::CONV, convAttrStrList, nullptr,
                 {{"DAV_2201",
                   {{TileOpFormat::TILEOP_NDC1HWC0, TileOpFormat::TILEOP_FRACTAL_Z_3D, TileOpFormat::TILEOP_ND},
                    {TileOpFormat::TILEOP_NDC1HWC0}}}});
    RegisterInfo(Opcode::OP_CONV_ADD, OpCoreType::AIC, "CONV_ADD", {}, {},
                 {"CUBE_CONV_ADD", PIPE_M, PIPE_M, CoreType::AIC}, OpCalcType::CONV, convAttrStrList);
    RegisterInfo(Opcode::OP_CUBE_CONV_D2S, OpCoreType::AIC, "CONV_D2S", {}, {},
                 {"CUBE_CONV_D2S", PIPE_FIX, PIPE_FIX, CoreType::AIC}, OpCalcType::CONV);
    RegisterInfo(Opcode::OP_CUBE_CONCAT_C, OpCoreType::AIC, "CONCAT_C", {}, {},
                 {"CUBE_CONCAT_C", PIPE_MTE1, PIPE_FIX, CoreType::AIC}, OpCalcType::CONV);
    RegisterInfo(Opcode::OP_L1_COPY_IN_CONV, OpCoreType::AIC, "L1_COPY_IN_CONV", {MemoryType::MEM_DEVICE_DDR},
                 {MemoryType::MEM_L1}, {"TileOp::L1CopyInConv", PIPE_MTE2, PIPE_MTE2, CoreType::AIC},
                 OpCalcType::MOVE_IN, {OpAttributeKey::srcGmConvValidShape});
    RegisterInfo(Opcode::OP_L1_COPY_IN_CONV_BP_DX_DY, OpCoreType::AIC, "L1_COPY_IN_CONV_BP_DX_DY",
                 {MemoryType::MEM_DEVICE_DDR}, {MemoryType::MEM_L1},
                 {"TileOp::TLoadConvBpDxDedy", PIPE_MTE2, PIPE_MTE2, CoreType::AIC}, OpCalcType::MOVE_IN,
                 {OpAttributeKey::srcGmConvValidShape, OpAttributeKey::strideH, OpAttributeKey::strideW,
                  OpAttributeKey::skipH, OpAttributeKey::skipW});
    RegisterInfo(Opcode::OP_L1_COPY_IN_CONV_BP, OpCoreType::AIC, "L1_COPY_IN_CONV_BP", {MemoryType::MEM_DEVICE_DDR},
                 {MemoryType::MEM_L1}, {"TileOp::L1CopyInConvBpNZ", PIPE_MTE2, PIPE_MTE2, CoreType::AIC},
                 OpCalcType::MOVE_IN, {OpAttributeKey::srcGmConvValidShape});
    RegisterInfo(Opcode::OP_LOAD3D_CONV, OpCoreType::AIC, "IMG2COL", {MemoryType::MEM_L1}, {MemoryType::MEM_L0A},
                 {"TileOp::TLoad3D", PIPE_MTE1, PIPE_MTE1, CoreType::AIC}, OpCalcType::MOVE_LOCAL,
                 {OpAttributeKey::postK, OpAttributeKey::postM, OpAttributeKey::filterH, OpAttributeKey::filterW,
                  OpAttributeKey::strideH, OpAttributeKey::strideW, OpAttributeKey::dilationH,
                  OpAttributeKey::dilationW, OpAttributeKey::paddingLeft, OpAttributeKey::paddingRight,
                  OpAttributeKey::paddingTop, OpAttributeKey::paddingBottom, OpAttributeKey::padValue,
                  OpAttributeKey::repeatStride, OpAttributeKey::repeatTime, OpAttributeKey::wStride});
    RegisterInfo(Opcode::OP_LOAD2D_CONV, OpCoreType::AIC, "LOAD2D", {MemoryType::MEM_L1}, {MemoryType::MEM_L0B},
                 {"TileOp::TLoad2D", PIPE_MTE1, PIPE_MTE1, CoreType::AIC}, OpCalcType::MOVE_LOCAL,
                 {OpAttributeKey::postK, OpAttributeKey::postN});
    RegisterInfo(Opcode::OP_LOAD2DDX_CONV, OpCoreType::AIC, "LOAD2DDX", {MemoryType::MEM_L1}, {MemoryType::MEM_L0B},
                 {"TileOp::TLoad2DDX", PIPE_MTE1, PIPE_MTE1, CoreType::AIC}, OpCalcType::MOVE_LOCAL,
                 {OpAttributeKey::postK, OpAttributeKey::postN, OpAttributeKey::hwk, OpAttributeKey::kL0Size,
                  OpAttributeKey::nL0Size, OpAttributeKey::k0Idx, OpAttributeKey::n0Idx});
    RegisterInfo(Opcode::OP_L0C_COPY_OUT_CONV, OpCoreType::AIC, "L0C_COPY_OUT_CONV", {MemoryType::MEM_L0C},
                 {MemoryType::MEM_DEVICE_DDR}, {"TileOp::L0CCopyOutConv", PIPE_FIX, PIPE_FIX, CoreType::AIC},
                 OpCalcType::MOVE_OUT, {OpAttributeKey::excludeBufferReuse, OpAttributeKey::l0cValidMN});
    RegisterInfo(Opcode::OP_TRANS_FORMAT_L0C, OpCoreType::AIC, "TRANS_FORMAT_L0C", {MemoryType::MEM_L0C},
                 {MemoryType::MEM_L0C}, {"TileOp::TransFormatL0C", PIPE_FIX, PIPE_FIX, CoreType::AIC},
                 OpCalcType::CAST);
    RegisterInfo(Opcode::OP_TRANS_FORMAT_L1, OpCoreType::AIC, "TRANS_FORMAT_L1", {MemoryType::MEM_L1},
                 {MemoryType::MEM_L1}, {"TileOp::TransFormatL1", PIPE_MTE1, PIPE_MTE1, CoreType::AIC},
                 OpCalcType::CAST);

    RegisterInfo(Opcode::OP_L1_ALLOC, OpCoreType::AIC, "L1_ALLOC", {}, {MemoryType::MEM_L1},
                 {"L1_ALLOC", PIPE_S, PIPE_S, CoreType::AIC}, OpCalcType::SYS);
    RegisterInfo(Opcode::OP_L0A_ALLOC, OpCoreType::AIC, "L0A_ALLOC", {}, {MemoryType::MEM_L0A},
                 {"L0A_ALLOC", PIPE_S, PIPE_S, CoreType::AIC}, OpCalcType::SYS);
    RegisterInfo(Opcode::OP_L0AMX_ALLOC, OpCoreType::AIC, "L0AMX_ALLOC", {}, {MemoryType::MEM_L0AMX},
                 {"L0AMX_ALLOC", PIPE_S, PIPE_S, CoreType::AIC}, OpCalcType::SYS);
    RegisterInfo(Opcode::OP_L0B_ALLOC, OpCoreType::AIC, "L0B_ALLOC", {}, {MemoryType::MEM_L0B},
                 {"L0B_ALLOC", PIPE_S, PIPE_S, CoreType::AIC}, OpCalcType::SYS);
    RegisterInfo(Opcode::OP_L0BMX_ALLOC, OpCoreType::AIC, "L0BMX_ALLOC", {}, {MemoryType::MEM_L0BMX},
                 {"L0BMX_ALLOC", PIPE_S, PIPE_S, CoreType::AIC}, OpCalcType::SYS);
    RegisterInfo(Opcode::OP_L0C_ALLOC, OpCoreType::AIC, "L0C_ALLOC", {}, {MemoryType::MEM_L0C},
                 {"L0C_ALLOC", PIPE_S, PIPE_S, CoreType::AIC}, OpCalcType::SYS);
    RegisterInfo(Opcode::OP_FIX_ALLOC, OpCoreType::AIC, "FIX_ALLOC", {}, {},
                 {"FIX_ALLOC", PIPE_S, PIPE_S, CoreType::AIC}, OpCalcType::SYS);
    RegisterInfo(Opcode::OP_BT_ALLOC, OpCoreType::AIC, "BT_ALLOC", {}, {}, {"BT_ALLOC", PIPE_S, PIPE_S, CoreType::AIC},
                 OpCalcType::SYS);

    RegisterInfo(Opcode::OP_L1_COPY_IN, OpCoreType::AIC, "L1_COPY_IN", {MemoryType::MEM_DEVICE_DDR},
                 {MemoryType::MEM_L1}, {"TileOp::L1CopyIn", PIPE_MTE2, PIPE_MTE2, CoreType::AIC}, OpCalcType::MOVE_IN);
    RegisterInfo(Opcode::OP_L1_COPY_IN_FRACTAL_Z, OpCoreType::AIC, "L1_COPY_IN_FractalZ", {MemoryType::MEM_DEVICE_DDR},
                 {MemoryType::MEM_L1}, {"TileOp::L1CopyInFractalZ", PIPE_MTE2, PIPE_MTE2, CoreType::AIC},
                 OpCalcType::MOVE_IN, {ConvOpAttributeKey::fmapC0});
    RegisterInfo(Opcode::OP_L1_COPY_OUT, OpCoreType::AIC, "L1_COPY_OUT", {MemoryType::MEM_L1},
                 {MemoryType::MEM_DEVICE_DDR}, {"TileOp::L1CopyOut", PIPE_FIX, PIPE_FIX, CoreType::AIC},
                 OpCalcType::MOVE_OUT);
    RegisterInfo(Opcode::OP_L1_LOOP_ENHANCE, OpCoreType::AIC, "L1_LOOP_ENHANCE", {MemoryType::MEM_L1},
                 {MemoryType::MEM_L1}, {"TileOp::Depth2Space", PIPE_FIX, PIPE_FIX, CoreType::AIC},
                 OpCalcType::MOVE_LOCAL);
    RegisterInfo(Opcode::OP_L1_COPY_IN_DMA, OpCoreType::AIC, "L1_COPY_IN_DMA", {MemoryType::MEM_DEVICE_DDR},
                 {MemoryType::MEM_L1}, {"TileOp::L1CopyInDMA", PIPE_MTE2, PIPE_MTE2, CoreType::AIC},
                 OpCalcType::MOVE_IN);
    RegisterInfo(Opcode::OP_L1_COPY_OUT_DMA, OpCoreType::AIC, "L1_COPY_OUT_DMA", {MemoryType::MEM_L1},
                 {MemoryType::MEM_DEVICE_DDR}, {"TileOp::L1CopyOutDMA", PIPE_MTE3, PIPE_MTE3, CoreType::AIC},
                 OpCalcType::MOVE_OUT);
    RegisterInfo(Opcode::OP_L1_TO_L0A, OpCoreType::AIC, "L1_TO_L0A", {MemoryType::MEM_L1}, {MemoryType::MEM_L0A},
                 {"TileOp::L1ToL0A", PIPE_MTE1, PIPE_MTE1, CoreType::AIC}, OpCalcType::MOVE_LOCAL);
    RegisterInfo(Opcode::OP_L1_TO_L0B, OpCoreType::AIC, "L1_TO_L0B", {MemoryType::MEM_L1}, {MemoryType::MEM_L0B},
                 {"TileOp::L1ToL0B", PIPE_MTE1, PIPE_MTE1, CoreType::AIC}, OpCalcType::MOVE_LOCAL);
    RegisterInfo(Opcode::OP_L1_TO_L0_AT, OpCoreType::AIC, "L1_TO_L0At", {MemoryType::MEM_L1}, {MemoryType::MEM_L0A},
                 {"TileOp::L1ToL0At", PIPE_MTE1, PIPE_MTE1, CoreType::AIC}, OpCalcType::MOVE_LOCAL);
    RegisterInfo(Opcode::OP_L1_TO_L0_BT, OpCoreType::AIC, "L1_TO_L0Bt", {MemoryType::MEM_L1}, {MemoryType::MEM_L0B},
                 {"TileOp::L1ToL0Bt", PIPE_MTE1, PIPE_MTE1, CoreType::AIC}, OpCalcType::MOVE_LOCAL);
    RegisterInfo(Opcode::OP_L0C_COPY_OUT, OpCoreType::AIC, "L0C_COPY_OUT", {MemoryType::MEM_L0C},
                 {MemoryType::MEM_DEVICE_DDR}, {"TileOp::L0CCopyOut", PIPE_FIX, PIPE_FIX, CoreType::AIC},
                 OpCalcType::MOVE_OUT, {OpAttributeKey::excludeBufferReuse});
    RegisterInfo(Opcode::OP_L1_TO_FIX, OpCoreType::AIC, "FIX_COPY_IN", {MemoryType::MEM_L1}, {MemoryType::MEM_FIX},
                 {"TileOp::L1CopyFix", PIPE_MTE1, PIPE_MTE1, CoreType::AIC}, OpCalcType::MOVE_LOCAL,
                 {FixpOpAttributeKey::fbAddrSpace});
    RegisterInfo(Opcode::OP_L1_TO_FIX_QUANT_PRE, OpCoreType::AIC, "FIX_COPY_IN_QUANT_PRE", {MemoryType::MEM_L1},
                 {MemoryType::MEM_FIX_QUANT_PRE}, {"TileOp::L1ToFB", PIPE_FIX, PIPE_FIX, CoreType::AIC},
                 OpCalcType::MOVE_LOCAL, {FixpOpAttributeKey::fbAddrSpace});
    RegisterInfo(Opcode::OP_L1_TO_FIX_RELU_PRE, OpCoreType::AIC, "FIX_COPY_IN_RELU_PRE", {MemoryType::MEM_L1},
                 {MemoryType::MEM_FIX_RELU_PRE}, {"TileOp::L1CopyFix", PIPE_MTE1, PIPE_MTE1, CoreType::AIC},
                 OpCalcType::MOVE_LOCAL, {FixpOpAttributeKey::fbAddrSpace});
    RegisterInfo(Opcode::OP_L1_TO_FIX_RELU_POST, OpCoreType::AIC, "FIX_COPY_IN_RELU_POST", {MemoryType::MEM_L1},
                 {MemoryType::MEM_FIX_RELU_POST}, {"TileOp::L1CopyFix", PIPE_MTE1, PIPE_MTE1, CoreType::AIC},
                 OpCalcType::MOVE_LOCAL, {FixpOpAttributeKey::fbAddrSpace});
    RegisterInfo(Opcode::OP_L1_TO_FIX_QUANT_POST, OpCoreType::AIC, "FIX_COPY_IN_QUANT_POST", {MemoryType::MEM_L1},
                 {MemoryType::MEM_FIX_QUANT_POST}, {"TileOp::L1CopyFix", PIPE_MTE1, PIPE_MTE1, CoreType::AIC},
                 OpCalcType::MOVE_LOCAL, {FixpOpAttributeKey::fbAddrSpace});
    RegisterInfo(Opcode::OP_L1_TO_FIX_ELT_ANTIQ, OpCoreType::AIC, "FIX_COPY_IN_ELT_ANTIQ", {MemoryType::MEM_L1},
                 {MemoryType::MEM_FIX_ELT_ANTIQ}, {"TileOp::L1CopyFix", PIPE_MTE1, PIPE_MTE1, CoreType::AIC},
                 OpCalcType::MOVE_LOCAL, {FixpOpAttributeKey::fbAddrSpace});
    RegisterInfo(Opcode::OP_L1_TO_FIX_MTE2_ANTIQ, OpCoreType::AIC, "FIX_COPY_IN_MTE2_ANTIQ", {MemoryType::MEM_L1},
                 {MemoryType::MEM_FIX_MTE2_ANTIQ}, {"TileOp::L1CopyFix", PIPE_MTE1, PIPE_MTE1, CoreType::AIC},
                 OpCalcType::MOVE_LOCAL, {FixpOpAttributeKey::fbAddrSpace});

    RegisterInfo(Opcode::OP_L1_COPY_UB, OpCoreType::AIC, "L1_COPY_UB", {MemoryType::MEM_L1}, {MemoryType::MEM_UB},
                 {"TileOp::L1CopyUB", PIPE_FIX, PIPE_FIX, CoreType::AIC}, OpCalcType::MOVE_OUT);
    RegisterInfo(Opcode::OP_L0C_COPY_UB, OpCoreType::AIC, "L0C_COPY_UB", {MemoryType::MEM_L0C}, {MemoryType::MEM_UB},
                 {"TileOp::L0CCopyUB", PIPE_FIX, PIPE_FIX, CoreType::AIC}, OpCalcType::MOVE_OUT);
    RegisterInfo(Opcode::OP_L0C_COPY_UB_DUAL_DST, OpCoreType::AIC, "L0C_COPY_UB_DUAL_DST", {MemoryType::MEM_L0C},
                 {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"TileOp::TCopyL0C2UB", PIPE_FIX, PIPE_FIX, CoreType::AIC},
                 OpCalcType::MOVE_OUT);
    RegisterInfo(Opcode::OP_UB_COPY_L1, OpCoreType::AIV, "UB_COPY_L1", {MemoryType::MEM_UB}, {MemoryType::MEM_L1},
                 {"TileOp::UBCopyL1", PIPE_MTE3, PIPE_MTE3, CoreType::AIV}, OpCalcType::MOVE_IN);
    RegisterInfo(Opcode::OP_UB_COPY_L1_ND, OpCoreType::AIV, "UB_COPY_L1_ND", {MemoryType::MEM_UB}, {MemoryType::MEM_L1},
                 {"TileOp::UBCopyL1ND", PIPE_MTE3, PIPE_MTE3, CoreType::AIV}, OpCalcType::MOVE_IN);
    RegisterInfo(Opcode::OP_L1_TO_L1, OpCoreType::AIC, "L1_TO_L1", {MemoryType::MEM_L1}, {MemoryType::MEM_L1},
                 {"TileOp::L1CopyL1", PIPE_MTE1, PIPE_MTE1, CoreType::AIC}, OpCalcType::MOVE_LOCAL);
    RegisterInfo(Opcode::OP_COPY_UB_TO_UB, OpCoreType::AIV, "UB_TO_UB", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::Tvcopy", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::MOVE_LOCAL,
                 {OpAttributeKey::excludeBufferReuse});
    RegisterInfo(Opcode::OP_L0C_TO_L1, OpCoreType::AIC, "L0C_COPY_L1", {MemoryType::MEM_L0C}, {MemoryType::MEM_L1},
                 {"TileOp::L0CToL1", PIPE_FIX, PIPE_FIX, CoreType::AIC}, OpCalcType::MOVE_OUT);
    RegisterInfo(Opcode::OP_L1_TO_BT, OpCoreType::AIC, "L1_TO_BT", {MemoryType::MEM_L1}, {MemoryType::MEM_BT},
                 {"TileOp::L1ToBT", PIPE_MTE1, PIPE_MTE1, CoreType::AIC}, OpCalcType::MOVE_LOCAL);
    RegisterInfo(Opcode::OP_UB_COPY_ND2NZ, OpCoreType::AIV, "UB_COPY_ND2NZ", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"TileOp::UBCopyUB", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::MOVE_LOCAL);
    RegisterInfo(Opcode::OP_GATHER_IN_L1, OpCoreType::AIC, "GATHER_IN_L1",
                 {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR},
                 {MemoryType::MEM_L1}, {"TileOp::GatherInL1", PIPE_MTE2, PIPE_MTE2, CoreType::AIC}, OpCalcType::OTHER,
                 {OpAttributeKey::startOffset});
    RegisterInfo(Opcode::OP_L1_COPY_IN_A_SCALE, OpCoreType::AIC, "L1_COPY_IN_A_SCALE", {MemoryType::MEM_DEVICE_DDR},
                 {MemoryType::MEM_L1}, {"TileOp::TLoadAMX", PIPE_MTE2, PIPE_MTE2, CoreType::AIC}, OpCalcType::MOVE_IN);
    RegisterInfo(Opcode::OP_L1_COPY_IN_B_SCALE, OpCoreType::AIC, "L1_COPY_IN_B_SCALE", {MemoryType::MEM_DEVICE_DDR},
                 {MemoryType::MEM_L1}, {"TileOp::TLoadAMX", PIPE_MTE2, PIPE_MTE2, CoreType::AIC}, OpCalcType::MOVE_IN);
    RegisterInfo(Opcode::OP_L1_TO_L0A_SCALE, OpCoreType::AIC, "L1_TO_L0A_SCALE", {MemoryType::MEM_L1},
                 {MemoryType::MEM_L0AMX}, {"TileOp::TEXtractMX", PIPE_MTE1, PIPE_MTE1, CoreType::AIC},
                 OpCalcType::MOVE_LOCAL);
    RegisterInfo(Opcode::OP_L1_TO_L0B_SCALE, OpCoreType::AIC, "L1_TO_L0B_SCALE", {MemoryType::MEM_L1},
                 {MemoryType::MEM_L0BMX}, {"TileOp::TEXtractMX", PIPE_MTE1, PIPE_MTE1, CoreType::AIC},
                 OpCalcType::MOVE_LOCAL);
    RegisterInfo(Opcode::OP_L1_RESHAPE_COPY_IN, OpCoreType::AIC, "L1_RESHAPE_COPY_IN", {MemoryType::MEM_DEVICE_DDR},
                 {MemoryType::MEM_L1}, {"TReshapeLoad", PIPE_MTE2, PIPE_MTE2, CoreType::AIC}, OpCalcType::MOVE_IN);
    RegisterInfo(Opcode::OP_L0C_RESHAPE_COPY_OUT, OpCoreType::AIC, "L0C_RESHAPE_COPY_OUT", {MemoryType::MEM_L0C},
                 {MemoryType::MEM_DEVICE_DDR}, {"TReshapeStore", PIPE_FIX, PIPE_FIX, CoreType::AIC},
                 OpCalcType::MOVE_OUT, {OpAttributeKey::excludeBufferReuse});
}

void OpcodeManager::RegisterDistribute()
{
    RegisterInfo(Opcode::OP_SHMEM_SET, OpCoreType::AIV, "SHMEM_SET",
                 {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR},
                 {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB},
                 {"TileOp::Distributed::ShmemSet", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::DISTRIBUTED,
                 {OpAttributeKey::requiresBoundaryCopy, OpAttributeKey::distOpAttr, OpAttributeKey::ownerRank});
    RegisterInfo(Opcode::OP_SHMEM_PUT, OpCoreType::AIV, "SHMEM_PUT",
                 {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR},
                 {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB},
                 {"TileOp::Distributed::ShmemPut", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::DISTRIBUTED,
                 {OpAttributeKey::requiresBoundaryCopy, OpAttributeKey::distOpAttr, OpAttributeKey::ownerRank});
    RegisterInfo(Opcode::OP_SHMEM_STORE, OpCoreType::AIV, "SHMEM_STORE",
                 {MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR},
                 {MemoryType::MEM_DEVICE_DDR}, {"TileOp::Distributed::ShmemStore", PIPE_S, PIPE_S, CoreType::AIV},
                 OpCalcType::DISTRIBUTED,
                 {OpAttributeKey::requiresBoundaryCopy, OpAttributeKey::distOpAttr, OpAttributeKey::ownerRank});
    RegisterInfo(Opcode::OP_SHMEM_SIGNAL, OpCoreType::AIV, "SHMEM_SIGNAL",
                 {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR},
                 {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB},
                 {"TileOp::Distributed::ShmemSignal", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::DISTRIBUTED,
                 {OpAttributeKey::requiresBoundaryCopy, OpAttributeKey::distOpAttr, OpAttributeKey::ownerRank});
    RegisterInfo(Opcode::OP_SHMEM_WAIT_UNTIL, OpCoreType::AICPU, "SHMEM_WAIT_UNTIL",
                 {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR}, {MemoryType::MEM_DEVICE_DDR}, TileOpCfg(),
                 OpCalcType::DISTRIBUTED,
                 {OpAttributeKey::requiresBoundaryCopy, OpAttributeKey::distOpAttr, OpAttributeKey::ownerRank});
    RegisterInfo(Opcode::OP_SHMEM_GET, OpCoreType::AIV, "SHMEM_GET",
                 {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR},
                 {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB},
                 {"TileOp::Distributed::ShmemGet", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::DISTRIBUTED,
                 {OpAttributeKey::requiresBoundaryCopy, OpAttributeKey::distOpAttr, OpAttributeKey::ownerRank});
    RegisterInfo(Opcode::OP_SHMEM_LOAD, OpCoreType::AIV, "SHMEM_LOAD",
                 {MemoryType::MEM_DEVICE_DDR /* dummy */, MemoryType::MEM_DEVICE_DDR /* shmemData */},
                 {MemoryType::MEM_UB /* UBData */}, {"TileOp::Distributed::ShmemLoad", PIPE_S, PIPE_S, CoreType::AIV},
                 OpCalcType::DISTRIBUTED,
                 {OpAttributeKey::requiresBoundaryCopy, OpAttributeKey::distOpAttr, OpAttributeKey::ownerRank});
    RegisterInfo(Opcode::OP_BIND_TENSOR, OpCoreType::ANY, "BIND_TENSOR", {}, {MemoryType::MEM_DEVICE_DDR},
                 {"TileOp::Distributed::ShmemGet", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::DISTRIBUTED,
                 {OP_ATTR_PREFIX + "BindTensor"});
}

void OpcodeManager::RegisterCommon()
{
    RegisterInfo(Opcode::OP_HUB, OpCoreType::HUB, "HUB", {MemoryType::MEM_DEVICE_DDR}, {MemoryType::MEM_DEVICE_DDR},
                 {"TileOp::Thub", PIPE_V, PIPE_V, CoreType::HUB}, OpCalcType::ELMWISE, {OpAttributeKey::inplaceInfo});
    RegisterInfo(Opcode::OP_REGISTER_COPY, OpCoreType::AIV, "REGISTER_COPY", {MemoryType::MEM_UB}, {MemoryType::MEM_UB},
                 {"REGISTER_COPY", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::SYS, {}, TileShapeVerifier::Verify);
    RegisterInfo(Opcode::OP_UB_ALLOC, OpCoreType::AIV, "UB_ALLOC", {}, {MemoryType::MEM_UB},
                 {"UB_ALLOC", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::SYS, {OpAttributeKey::excludeBufferReuse});
    RegisterInfo(Opcode::OP_REG_ALLOC, OpCoreType::AIV, "REG_ALLOC", {}, {MemoryType::MEM_VECTOR_REG},
                 {"REG_ALLOC", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::SYS);
    RegisterInfo(Opcode::OP_DUPLICATE, OpCoreType::ANY, "DUPLICATE", {}, {MemoryType::MEM_DEVICE_DDR}, {},
                 OpCalcType::MOVE_LOCAL);
    RegisterInfo(Opcode::OP_RESHAPE, OpCoreType::ANY, "RESHAPE", {}, {},
                 {"TileOp::Treshape", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::OTHER, {});
    RegisterInfo(Opcode::OP_RESHAPE_COPY_IN, OpCoreType::ANY, "RESHAPE_COPY_IN", {MemoryType::MEM_DEVICE_DDR},
                 {MemoryType::MEM_UB}, {"TileOp::ReshapeCopyIn", PIPE_MTE2, PIPE_MTE2, CoreType::AIV},
                 OpCalcType::OTHER, {});
    RegisterInfo(Opcode::OP_RESHAPE_COPY_OUT, OpCoreType::ANY, "RESHAPE_COPY_OUT", {MemoryType::MEM_UB},
                 {MemoryType::MEM_DEVICE_DDR}, {"TileOp::ReshapeCopyOut", PIPE_MTE3, PIPE_MTE3, CoreType::AIV},
                 OpCalcType::OTHER, {});
    RegisterInfo(Opcode::OP_ASSEMBLE, OpCoreType::ANY, "ASSEMBLE", {}, {}, {"ASSEMBLE", PIPE_S, PIPE_S, CoreType::AIV},
                 OpCalcType::MOVE_LOCAL);
    RegisterInfo(Opcode::OP_CONTRACT, OpCoreType::ANY, "CONTRACT", {}, {MemoryType::MEM_DEVICE_DDR},
                 {"CONTRACT", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::MOVE_LOCAL);
    RegisterInfo(Opcode::OP_ASSEMBLE_SSA, OpCoreType::ANY, "ASSEMBLE_SSA",
                 {MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR}, {MemoryType::MEM_DEVICE_DDR},
                 {"ASSEMBLE_SSA", PIPE_MTE3, PIPE_MTE3, CoreType::AIV},
                 OpCalcType::MOVE_OUT); // 输出输出支持其他类型
    RegisterInfo(Opcode::OP_ATOMIC_RMW, OpCoreType::ANY, "ATOMIC_RMW", {MemoryType::MEM_DEVICE_DDR},
                 {MemoryType::MEM_DEVICE_DDR}, {"AtomicRMW", PIPE_MTE3, PIPE_MTE3, CoreType::AIV}, OpCalcType::MOVE_OUT,
                 {OpAttributeKey::rmwMode});
    RegisterInfo(Opcode::OP_VIEW, OpCoreType::ANY, "VIEW", {}, {}, {"VIEW", PIPE_S, PIPE_S, CoreType::AIV},
                 OpCalcType::MOVE_LOCAL);
    RegisterInfo(Opcode::OP_SLICE, OpCoreType::ANY, "SLICE", {}, {}, {"SLICE", PIPE_S, PIPE_S, CoreType::AIV},
                 OpCalcType::MOVE_LOCAL);
    RegisterInfo(Opcode::OP_VIEW_TYPE, OpCoreType::ANY, "VIEW_TYPE", {}, {},
                 {"VIEW_TYPE", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::MOVE_LOCAL);
    RegisterInfo(Opcode::OP_CONVERT, OpCoreType::ANY, "CONVERT", {}, {}, {}, OpCalcType::SYS);
    RegisterInfo(Opcode::OP_COPY_IN, OpCoreType::ANY, "COPY_IN", {}, {}, {}, OpCalcType::MOVE_IN,
                 {OpAttributeKey::outputCombineAxis, OpAttributeKey::excludeBufferReuse});
    RegisterInfo(Opcode::OP_COPY_OUT, OpCoreType::ANY, "COPY_OUT", {}, {}, {}, OpCalcType::MOVE_OUT,
                 {OP_ATTR_PREFIX + "atomic_add", OpAttributeKey::inputCombineAxis, OpAttributeKey::excludeBufferReuse});
    RegisterInfo(Opcode::OP_CALL, OpCoreType::ANY, "CALL", {}, {}, {}, OpCalcType::SYS);
    RegisterInfo(Opcode::OP_PRINT, OpCoreType::ANY, "OP_DUMP", {}, {}, {}, OpCalcType::SYS, {});
    RegisterInfo(Opcode::OP_BLOCK_CALL, OpCoreType::ANY, "BLOCK_CALL", {}, {}, {}, OpCalcType::SYS);
    RegisterInfo(Opcode::OP_FUSED_OP, OpCoreType::AIV, "FUSED_OP", {MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {MemoryType::MEM_UB}, {"TileOp::fusedOP", PIPE_V, PIPE_V, CoreType::AIV}, OpCalcType::BROADCAST);
    RegisterInfo(Opcode::OP_VLD, OpCoreType::ANY, "VLD", {}, {}, {}, OpCalcType::MOVE_IN);
    RegisterInfo(Opcode::OP_VST, OpCoreType::ANY, "VST", {}, {}, {}, OpCalcType::MOVE_IN);

    RegisterInfo(Opcode::OP_SYNC_SRC, OpCoreType::ANY, "SYNC_SRC", {}, {}, {"SYNC_SRC", PIPE_S, PIPE_S, CoreType::AIC},
                 OpCalcType::SYNC);
    RegisterInfo(Opcode::OP_SYNC_DST, OpCoreType::ANY, "SYNC_DST", {}, {}, {"SYNC_DST", PIPE_S, PIPE_S, CoreType::AIC},
                 OpCalcType::SYNC);
    RegisterInfo(Opcode::OP_CV_SYNC_SRC, OpCoreType::ANY, "CV_SYNC_SRC", {}, {},
                 {"CV_SYNC_SRC", PIPE_S, PIPE_S, CoreType::AIC}, OpCalcType::SYNC);
    RegisterInfo(Opcode::OP_CV_SYNC_DST, OpCoreType::ANY, "CV_SYNC_DST", {}, {},
                 {"CV_SYNC_DST", PIPE_S, PIPE_S, CoreType::AIC}, OpCalcType::SYNC);
    RegisterInfo(Opcode::OP_FFTS_CROSS_CORE_SYNC, OpCoreType::ANY, "FFTS_CROSS_CORE_SYNC", {}, {},
                 {"FFTS_CROSS_CORE_SYNC", PIPE_S, PIPE_S, CoreType::AIC}, OpCalcType::SYNC);
    RegisterInfo(Opcode::OP_WAIT_FLAG_DEV, OpCoreType::ANY, "WAIT_FLAG_DEV", {}, {},
                 {"WAIT_FLAG_DEV", PIPE_S, PIPE_S, CoreType::AIC}, OpCalcType::SYNC);
    RegisterInfo(Opcode::OP_PHASE1, OpCoreType::ANY, "PHASE1", {}, {}, {"PHASE1", PIPE_S, PIPE_S, CoreType::AIC},
                 OpCalcType::SYNC);
    RegisterInfo(Opcode::OP_PHASE2, OpCoreType::ANY, "PHASE2", {}, {}, {"PHASE2", PIPE_S, PIPE_S, CoreType::AIC},
                 OpCalcType::SYNC);
    RegisterInfo(Opcode::OP_BAR_V, OpCoreType::ANY, "BAR.V", {}, {}, {"BAR.V", PIPE_S, PIPE_S, CoreType::AIC},
                 OpCalcType::SYNC);
    RegisterInfo(Opcode::OP_BAR_M, OpCoreType::ANY, "BAR.M", {}, {}, {"BAR.M", PIPE_S, PIPE_S, CoreType::AIC},
                 OpCalcType::SYNC);
    RegisterInfo(Opcode::OP_BAR_ALL, OpCoreType::ANY, "BAR.ALL", {}, {}, {"BAR.ALL", PIPE_S, PIPE_S, CoreType::AIC},
                 OpCalcType::SYNC);
    RegisterInfo(Opcode::OP_AICPU_CALL_AIC, OpCoreType::ANY, "AICPU_CALL_AIC", {}, {},
                 {"TileOp::AicpuCall", PIPE_S, PIPE_S, CoreType::AIC}, OpCalcType::SYS);
    RegisterInfo(Opcode::OP_AICPU_CALL_AIV, OpCoreType::ANY, "AICPU_CALL_AIV", {}, {},
                 {"TileOp::AicpuCall", PIPE_S, PIPE_S, CoreType::AIV}, OpCalcType::SYS);
    RegisterInfo(Opcode::OP_NOP, OpCoreType::ANY, "NOP", {}, {}, {}, OpCalcType::OTHER);
}

OpcodeManager::OpcodeManager()
{
    RegisterVector();
    RegisterCube();
    RegisterDistribute();
    RegisterCommon();
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, strToEnum_.size() == static_cast<size_t>(Opcode::OP_UNKNOWN))
        << "strToEnum_.size() should equal to Opcode::OP_UNKNOWN";
}

// NEXTNEXT: delete after tile op register has supported tile tensor
std::unordered_map<Opcode, std::string> SUPPORT_TILETENSOR_OPS{
    {Opcode::OP_UB_COPY_IN, "TLoad"},
    {Opcode::OP_UB_COPY_OUT, "TStoreVec"},
    {Opcode::OP_RESHAPE_COPY_IN, "TReshapeLoad"},
    {Opcode::OP_RESHAPE_COPY_OUT, "TReshapeStore"},
    {Opcode::OP_TRANSPOSE_VNCHWCONV, "TTrans"},
    {Opcode::OP_TRANSPOSE_MOVEIN, "TTransMoveIn"},
    {Opcode::OP_TRANSPOSE_MOVEOUT, "TTransMoveOut"},
    {Opcode::OP_NCHW2NC1HWC0, "TTransDataNCHW2NC1HWC0"},
    {Opcode::OP_NCHW2Fractal_Z, "TTransDataNCHW2Fractal_Z"},
    {Opcode::OP_NC1HWC02NCHW, "TTransDataNC1HWC02NCHW"},
    {Opcode::OP_NCDHW2NDC1HWC0, "TTransDataNCDHW2NDC1HWC0"},
    {Opcode::OP_NCDHW2FRACTAL_Z_3D, "TTransDataNCDHW2FRACTAL_Z_3D"},
    {Opcode::OP_NDC1HWC02NCDHW, "TTransDataNDC1HWC02NCDHW"},
    {Opcode::OP_FractalZ2NCHW, "TTransDataFractalZ2NCHW"},
    {Opcode::OP_FractalZ3D2NCDHW, "TTransDataFractalZ3D2NCDHW"},
    {Opcode::OP_INDEX_PUT, "TIndexPut"},
    {Opcode::OP_GCD, "TGcd"},
    {Opcode::OP_ERF, "TErf"},
    {Opcode::OP_SIN, "TSin"},
    {Opcode::OP_COS, "TCos"},
    {Opcode::OP_ERFC, "TErfc"},
    {Opcode::OP_PACK, "TPack"},
    {Opcode::OP_UNPACK, "TUnPack"},
    {Opcode::OP_ADD, "TAdd"},
    {Opcode::OP_AXPY, "TAxpy"},
    {Opcode::OP_CUM_SUM, "TCumOperation"},
    {Opcode::OP_CUM_PROD, "TCumOperation"},
    {Opcode::OP_ATAN, "TAtan"},
    {Opcode::OP_ATAN2, "TAtan2"},
    {Opcode::OP_SUB, "TSub"},
    {Opcode::OP_TRIUL, "TTriUL"},
    {Opcode::OP_DIV, "TDiv"},
    {Opcode::OP_MOD, "TMod"},
    {Opcode::OP_POW, "TPow"},
    {Opcode::OP_POWS, "TPowS"},
    {Opcode::OP_MUL, "TMul"},
    {Opcode::OP_INTERLEAVE, "TInterleave"},
    {Opcode::OP_DEINTERLEAVE, "TDeInterleave"},
    {Opcode::OP_DEINTERLEAVE_SINGLE, "TDeInterleave"},
    {Opcode::OP_REM, "TRemainder"},
    {Opcode::OP_REMS, "TRemainderS"},
    {Opcode::OP_REMRS, "TRemainderRS"},
    {Opcode::OP_INDEX_ADD_UB, "TIndexAddUB"},
    {Opcode::OP_INDEX_ADD, "TIndexAdd"},
    {Opcode::OP_GATHER_ELEMENT, "TgatherElement"},
    {Opcode::OP_GATHER, "Tgather"},
    {Opcode::OP_GATHER_IN_UB, "TgatherInUB"},
    {Opcode::OP_SCATTER, "Tscatter"},
    {Opcode::OP_SCATTER_ELEMENT, "TscatterElementS"},
    {Opcode::OP_EXPAND, "TExpand"},
    {Opcode::OP_BITSORT, "TBitSort"},
    {Opcode::OP_MRGSORT, "TMrgSort"},
    {Opcode::OP_TILEDMRGSORT, "TTiledMrgSort"},
    {Opcode::OP_EXTRACT, "TExtract"},
    {Opcode::OP_TWOTILEMRGSORT, "TTwoTileMrgSort"},
    {Opcode::OP_EXTRACT_SINGLE, "TExtractSingle"},
    {Opcode::OP_GATHER_MASK, "TGatherMask"},
    {Opcode::OP_QUANT_MX, "TQuantMX"},
    {Opcode::OP_CAST, "TCast"},
    {Opcode::OP_ROWSUM_SINGLE, "TRowSumSingle"},
    {Opcode::OP_ROWARGMAXWITHVALUE_SINGLE, "TRowArgMaxWithValueSingle"},
    {Opcode::OP_ROWARGMINWITHVALUE_SINGLE, "TRowArgMinWithValueSingle"},
    {Opcode::OP_ROWARGMAXWITHVALUE_LINE, "TRowArgMaxWithValueLine"},
    {Opcode::OP_ROWARGMINWITHVALUE_LINE, "TRowArgMinWithValueLine"},
    {Opcode::OP_ROWMAX_SINGLE, "TRowMaxSingle"},
    {Opcode::OP_ROWMIN_SINGLE, "TRowMinSingle"},
    {Opcode::OP_ROWPROD_SINGLE, "TRowProdSingle"},
    {Opcode::OP_ONLINE_SOFTMAX, "TOnlineSoftmax"},
    {Opcode::OP_ONLINE_SOFTMAX_UPDATE, "TOnlineSoftmaxUpdate"},
    {Opcode::OP_ROWSUMLINE, "TRowSumLine"},
    {Opcode::OP_ROWMAXLINE, "TRowMaxLine"},
    {Opcode::OP_ROWMINLINE, "TRowMinLine"},
    {Opcode::OP_ROWPRODLINE, "TRowProdLine"},
    {Opcode::OP_LOGICALAND, "TLogicalAnd"},
    {Opcode::OP_WHERE_TT, "TWhereTT"},
    {Opcode::OP_WHERE_TS, "TWhereTS"},
    {Opcode::OP_WHERE_ST, "TWhereST"},
    {Opcode::OP_WHERE_SS, "TWhereSS"},
    {Opcode::OP_CMP, "TCompare"},
    {Opcode::OP_CMPS, "TCompareScalar"},
    {Opcode::OP_HYPOT, "THypot"},
    {Opcode::OP_PRELU, "TPRelu"},
    {Opcode::OP_ADDS, "TAddS"},
    {Opcode::OP_MODS, "TModS"},
    {Opcode::OP_SUBS, "TSubS"},
    {Opcode::OP_MAXS, "TMaxS"},
    {Opcode::OP_MINS, "TMinS"},
    {Opcode::OP_MULS, "TMulS"},
    {Opcode::OP_LRELU, "TLReLU"},
    {Opcode::OP_DIVS, "TDivS"},
    {Opcode::OP_GCDS, "TGcdS"},
    {Opcode::OP_RSQRT, "TRsqrt"},
    {Opcode::OP_RELU, "TRelu"},
    {Opcode::OP_LOG1P, "TLog1p"},
    {Opcode::OP_RADIX_SELECT, "TRadixSelect"},
    {Opcode::OP_PAD, "TPad"},
    {Opcode::OP_FILLPAD, "TFillPad"},
    {Opcode::OP_SQRT, "TSqrt"},
    {Opcode::OP_SIGN, "TSign"},
    {Opcode::OP_SIGNBIT, "TSignbit"},
    {Opcode::OP_TANH, "TTanh"},
    {Opcode::OP_TAN, "Ttan"},
    {Opcode::OP_CEIL, "TCeil"},
    {Opcode::OP_FLOOR, "TFloor"},
    {Opcode::OP_TRUNC, "TTrunc"},
    {Opcode::OP_ROUND, "TRound"},
    {Opcode::OP_RECIPROCAL, "TReciprocal"},
    {Opcode::OP_EXP, "TExp"},
    {Opcode::OP_EXP2, "TExp2"},
    {Opcode::OP_EXPM1, "TExpm1"},
    {Opcode::OP_ABS, "TAbs"},
    {Opcode::OP_LOGICALNOT, "TLogicalNot"},
    {Opcode::OP_MAXIMUM, "TMax"},
    {Opcode::OP_MINIMUM, "TMin"},
    {Opcode::OP_PAIRSUM, "TPairSum"},
    {Opcode::OP_PAIRARGMAX, "TPairArgMax"},
    {Opcode::OP_PAIRARGMIN, "TPairArgMin"},
    {Opcode::OP_PAIRMAX, "TPairMax"},
    {Opcode::OP_PAIRMIN, "TPairMin"},
    {Opcode::OP_PAIRPROD, "TPairProd"},
    {Opcode::OP_PERMUTE, "TPermute"},
    {Opcode::OP_PERMUTE_ELEMENT, "TPermuteElewise"},
    {Opcode::OP_ONEHOT, "TOneHot"},
    {Opcode::OP_VEC_DUP, "TVecDup"},
    {Opcode::OP_RANGE, "TRange"},
    {Opcode::OP_UNIFORM, "TUniform"},
    {Opcode::OP_BRCB, "Tbrcb"},
    {Opcode::OP_LN, "TLog"},
    {Opcode::OP_INDEX_OUTCAST, "TIndexOutcast"},
    {Opcode::OP_BITWISERIGHTSHIFT, "TBitrshift"},
    {Opcode::OP_BITWISELEFTSHIFT, "TBitlshift"},
    {Opcode::OP_BITWISERIGHTSHIFTS, "TBitrshiftS"},
    {Opcode::OP_BITWISELEFTSHIFTS, "TBitlshiftS"},
    {Opcode::OP_SBITWISERIGHTSHIFT, "TSBitrshift"},
    {Opcode::OP_SBITWISELEFTSHIFT, "TSBitlshift"},
    {Opcode::OP_BITWISEAND, "TBitwiseAnd"},
    {Opcode::OP_BITWISEOR, "TBitwiseOr"},
    {Opcode::OP_BITWISEXOR, "TBitwiseXor"},
    {Opcode::OP_BITWISEANDS, "TBitwiseAndS"},
    {Opcode::OP_BITWISEORS, "TBitwiseOrS"},
    {Opcode::OP_BITWISEXORS, "TBitwiseXorS"},
    {Opcode::OP_BITWISENOT, "TBitwiseNot"},
    {Opcode::OP_EXPANDEXPDIF, "TExpandExpDif"},
    {Opcode::OP_COPYSIGN, "TCopysign"},
    {Opcode::OP_FLOORDIV, "TFloorDiv"},
    {Opcode::OP_FLOORDIVS, "TFloorDivS"},
    {Opcode::OP_QUANTIZE_SYM, "TQuant"},
    {Opcode::OP_QUANTIZE_ASYM, "TQuant"},
    {Opcode::OP_DEQUANTIZE, "TDequant"},
    {Opcode::OP_SINH, "TSinh"},
    {Opcode::OP_COSH, "TCosh"},
    {Opcode::OP_ASIN, "TAsin"},
    {Opcode::OP_ACOS, "TAcos"},
    {Opcode::OP_ASINH, "TASinh"},
    {Opcode::OP_ACOSH, "TACosh"},
    {Opcode::OP_ATANH, "TAtanh"},
    {Opcode::OP_L1_TO_L0A, "TExtractL1ToL0"},
    {Opcode::OP_L1_TO_L0B, "TExtractL1ToL0"},
    {Opcode::OP_L1_TO_L0_AT, "TExtractL1ToL0"},
    {Opcode::OP_L1_TO_L0_BT, "TExtractL1ToL0"},
    {Opcode::OP_A_MUL_B, "TMatmul"},
    {Opcode::OP_A_MULACC_B, "TMatmul"},
    {Opcode::OP_L1_TO_FIX_QUANT_PRE, "TExtractL1ToFB"},
    {Opcode::OP_L1_TO_BT, "TExtractL1ToBT"},
    {Opcode::OP_UB_COPY_L1, "TCopyUB2L1"},
    {Opcode::OP_L0C_COPY_UB, "TCopyL0C2UB"},
    {Opcode::OP_L0C_COPY_UB_DUAL_DST, "TCopyL0C2UB"},
    {Opcode::OP_L0C_TO_L1, "TExtract"},
    {Opcode::OP_UB_COPY_ND2NZ, "TMoveND2NZ"},
    {Opcode::OP_L1_COPY_IN, "TLoad"},
    {Opcode::OP_L0C_COPY_OUT, "TStore"},
    {Opcode::OP_L1_COPY_OUT, "TStore"},
    {Opcode::OP_GATHER_IN_L1, "TGatherInL1"},
    {Opcode::OP_ISFINITE, "TIsFinite"},
    {Opcode::OP_ISNAN, "TIsNan"},
    {Opcode::OP_L1_COPY_IN_A_SCALE, "TLoadAMX"},
    {Opcode::OP_L1_COPY_IN_B_SCALE, "TLoadBMX"},
    {Opcode::OP_L1_TO_L0A_SCALE, "TExtractMX"},
    {Opcode::OP_L1_TO_L0B_SCALE, "TExtractMX"},
    {Opcode::OP_L1_COPY_IN_CONV, "TLoadConv"},
    {Opcode::OP_LOAD3D_CONV, "TLoad3D"},
    {Opcode::OP_LOAD2D_CONV, "TLoad2D"},
    {Opcode::OP_L1_COPY_IN_CONV_BP_DX_DY, "TLoadConvBpDxDedy"},
    {Opcode::OP_L1_COPY_IN_CONV_BP, "TLoadConvBPNZ"},
    {Opcode::OP_LOAD2DDX_CONV, "TLoad2DDX"},
    {Opcode::OP_L0C_COPY_OUT_CONV, "TStoreConv"},
    {Opcode::OP_SHMEM_PUT, "TileOp::Distributed::ShmemPut"},
    {Opcode::OP_SHMEM_GET, "TileOp::Distributed::ShmemGet"},
    {Opcode::OP_SHMEM_SIGNAL, "TileOp::Distributed::ShmemSignal"},
    {Opcode::OP_SHMEM_STORE, "TileOp::Distributed::ShmemStore"},
    {Opcode::OP_SHMEM_LOAD, "TileOp::Distributed::ShmemLoad"},
    {Opcode::OP_SHMEM_SET, "TileOp::Distributed::ShmemSet"},
    {Opcode::OP_L1_RESHAPE_COPY_IN, "TReshapeLoad"},
    {Opcode::OP_L0C_RESHAPE_COPY_OUT, "TReshapeStore"},
    {Opcode::OP_LOG1P, "TLog1p"},
};

std::unordered_set<Opcode> SUPPORT_VF_FUSE_OPS{
    Opcode::OP_ADD,
    Opcode::OP_SUB,
    Opcode::OP_DIV,
    Opcode::OP_MUL,
    Opcode::OP_ADDS,
    Opcode::OP_MULS,
    Opcode::OP_SUBS,
    Opcode::OP_DIVS,
    Opcode::OP_RSQRT,
    Opcode::OP_SQRT,
    Opcode::OP_EXP,
    Opcode::OP_MAXIMUM,
    Opcode::OP_MINIMUM,
    Opcode::OP_ROWSUM_SINGLE,
    Opcode::OP_ROWMAX_SINGLE,
    Opcode::OP_ONLINE_SOFTMAX,
    Opcode::OP_ONLINE_SOFTMAX_UPDATE,
    Opcode::OP_ROWMIN_SINGLE,
    Opcode::OP_ROWPROD_SINGLE,
    Opcode::OP_CAST,
    Opcode::OP_EXPAND,
    Opcode::OP_GCD,
    Opcode::OP_GCDS,
    Opcode::OP_INTERLEAVE,
    Opcode::OP_DEINTERLEAVE,
    Opcode::OP_DEINTERLEAVE_SINGLE,
};

std::unordered_set<Opcode> SKIP_OPCODE_FOR_CODEGEN = {
    Opcode::OP_VIEW,
    Opcode::OP_SLICE,
    Opcode::OP_ASSEMBLE,
    Opcode::OP_CONTRACT,
    Opcode::OP_RESHAPE,
    Opcode::OP_UB_ALLOC,
    Opcode::OP_L1_ALLOC,
    Opcode::OP_L0A_ALLOC,
    Opcode::OP_L0B_ALLOC,
    Opcode::OP_L0C_ALLOC,
    Opcode::OP_FIX_ALLOC,
    Opcode::OP_BT_ALLOC,
    Opcode::OP_BIND_TENSOR,
    Opcode::OP_TRANS_FORMAT_L0C,
    Opcode::OP_NOP,
    Opcode::OP_HUB,
    Opcode::OP_TRANS_FORMAT_L1,
};

std::unordered_set<Opcode> SUPPORT_BRC_INLINE{
    Opcode::OP_ADD,     Opcode::OP_SUB,     Opcode::OP_DIV,          Opcode::OP_MUL,
    Opcode::OP_MAXIMUM, Opcode::OP_MINIMUM, Opcode::OP_EXPANDEXPDIF,
};
} // namespace npu::tile_fwk
