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
 * \file ISA.cpp
 * \brief
 */

#include "cost_model/simulation/common/ISA.h"
#include "nlohmann/json.hpp"
#include "interface/operation/opcode.h"
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_code.h"

namespace CostModel {

using Json = nlohmann::json;

Tile::Tile(const std::string& str)
{
    Json j = Json::parse(str);
    magic = j.at("magic");
    const auto& shapeJson = j.at("shape");
    for (const auto& value : shapeJson) {
        shape.emplace_back(value.get<int>());
    }
    const auto& offsetJson = j.at("offset");
    for (const auto& value : offsetJson) {
        offset.emplace_back(value.get<int>());
    }
    bufferType = j.at("memorytype").at("tobe");
    bufType = BufferNameToType(bufferType);

    GetPipeType();

    symbol = j.at("rawtensor").at("symbol");
    dataTypeStr = j.at("rawtensor").at("datatype");
    dataType = CostModel::ToDataType(dataTypeStr);
    std::string type = j.at("nodetype");
    nodeType = CostModel::ToNodeType(type);
    rawMagic = j.at("rawtensor").at("rawmagic");
    const auto& rawShapeJson = j.at("rawtensor").at("rawshape");
    for (const auto& value : rawShapeJson) {
        rawShape.emplace_back(value.get<int>());
    }
}

void Tile::GetPipeType()
{
    switch (bufType) {
        case BUF_UB:
            pipeType = CorePipeType::PIPE_VECTOR_BMU;
            break;
        case BUF_L1:
            pipeType = CorePipeType::PIPE_CUBE_BMU_L1;
            break;
        case BUF_L0A:
            pipeType = CorePipeType::PIPE_CUBE_BMU_L0A;
            break;
        case BUF_L0B:
            pipeType = CorePipeType::PIPE_CUBE_BMU_L0B;
            break;
        case BUF_L0C:
            pipeType = CorePipeType::PIPE_CUBE_BMU_L0C;
            break;
        default:
            pipeType = CorePipeType::PIPE_TILE_ALLOC;
    }
}

std::string Tile::Dump()
{
    std::stringstream oss;
    oss << magic << " ";
    oss << OperandTypeToStr(bufType);
    oss << "_[";
    for (size_t i = 0; i < offset.size(); ++i) {
        oss << offset[i];
        if (i != offset.size() - 1) {
            oss << ",";
        }
    }
    oss << "]";
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        oss << shape[i];
        if (i != shape.size() - 1) {
            oss << ",";
        }
    }
    oss << "]-" << magic << "-" << rawMagic;
    oss << " " << SizeinBytes() << "B";
    return oss.str();
}

int Tile::SizeinBytes()
{
    if (shape.empty()) {
        return 0;
    }
    int result = BytesOf(dataType);
    for (auto it : shape) {
        result *= it;
    }
    return result;
}

void TileOp::GetPipeType()
{
    auto coreTypeQuery = SCHED_CORE_PIPE_TYPE.find(opcode);
    if (IsCall()) {
        pipeType = CorePipeType::PIPE_CALL;
    } else if (coreTypeQuery != SCHED_CORE_PIPE_TYPE.end()) {
        pipeType = coreTypeQuery->second;
    } else {
        SIMULATION_LOGW("[SIMULATION]: No pipe type corresponding to opcode is found, "
                        "fallback to PIPE_VECTOR_ALU. opcode=%s",
                        opcode.c_str());
        pipeType = CorePipeType::PIPE_VECTOR_ALU;
    }
}

uint64_t TileOp::GetAddress()
{
    uint64_t addr = 0;
    TilePtr tile = nullptr;
    if (IsReadCache(pipeType)) {
        tile = iOperand[0];
    } else if (IsWriteCache(pipeType)) {
        tile = oOperand[0];
    } else {
        ASSERT(static_cast<unsigned>(CostModel::ForwardSimErrorScene::INVALID_PIPE_TYPE), false)
            << ",[SIMULATION]: "
            << "PipeType Unrecognized." << Dump() << CorePipeName(pipeType);
    }
    addr = tile->rawMagic * RAW_MAGIC_MAX_SIZE;
    // calculate addr based on rawShape and offset
    for (size_t i = 0; i < tile->offset.size(); i++) {
        addr += tile->offset[i] * tile->rawShape[i];
    }
    return addr;
}

uint64_t TileOp::GetSize()
{
    uint64_t size = 0;
    TilePtr tile = nullptr;
    if (IsReadCache(pipeType)) {
        tile = iOperand[0];
    } else if (IsWriteCache(pipeType)) {
        tile = oOperand[0];
    } else {
        ASSERT(static_cast<unsigned>(CostModel::ForwardSimErrorScene::INVALID_PIPE_TYPE), false)
            << ",[SIMULATION]: "
            << "PipeType Unrecognized." << Dump() << CorePipeName(pipeType);
    }
    uint64_t shapeSize = 1;
    for (auto& s : tile->shape) {
        shapeSize *= s;
    }
    size = shapeSize * BytesOf(tile->dataType);
    return size;
}

bool TileOp::IsCall() { return opcode.find("CALL") != std::string::npos; }

bool TileOp::IsSpecial()
{
    if (opcode == "RESHAPE" || opcode == "VIEW" || opcode == "ASSEMBLE") {
        specialOp = true;
        return true;
    }
    return false;
}

std::string TileOp::Dump(bool outDetail)
{
    std::stringstream oss;
    int formatOffset = 3;
    oss << magic << " ";
    oss << opcode << " ";

    for (size_t i = 0; i < iOperand.size(); ++i) {
        if (!outDetail) {
            oss << std::setw(formatOffset) << std::setfill(' ') << iOperand[i]->magic;
        } else {
            oss << iOperand[i]->Dump();
        }
        if (i != iOperand.size() - 1) {
            oss << ",";
        }
    }
    oss << " TO ";
    for (size_t i = 0; i < oOperand.size(); ++i) {
        if (!outDetail) {
            oss << std::setw(formatOffset) << std::setfill(' ') << oOperand[i]->magic;
        } else {
            oss << oOperand[i]->Dump();
        }
        if (i != iOperand.size() - 1) {
            oss << ",";
        }
    }
    return oss.str();
}

void CycleInfo::Reset()
{
    fetchCycle = 0;
    decodeCycle = 0;
    renameCycle = 0;
    dispatchCycle = 0;
    insertIqCycle = 0;
    readyCycle = 0;
    pickedCycle = 0;
    issueCycle = 0;
    completedCycle = 0;
    retireCycle = 0;
    allocCycle = 0;
    writeCycle = 0;
    freeCycle = 0;
}

void ExecuteInfo::Reset()
{
    exePipeId = -1;
    isIncast = false;
    isOutcast = false;
    copyOutIdx = -1;
    domCount = 1;
    sequenceToIssue = 0;
    issued = false;
    retired = false;
    isAllocated = false;
    isWritten = false;
    writeReference = 0;
    readReference = 0;

    cycleInfo.Reset();
}

void Function::GetOpSequeceAfterOOO(int opmagic, uint64_t& index)
{
    if (opSequenceAfterOOO_.find(opmagic) != opSequenceAfterOOO_.end()) {
        index = opSequenceAfterOOO_[opmagic];
    }
}

void Function::InitPipeExecTime()
{
    pipeExecuteTime[CorePipeType::PIPE_VECTOR_BMU] = 0;
    pipeExecuteTime[CorePipeType::PIPE_CUBE_BMU_L1] = 0;
    pipeExecuteTime[CorePipeType::PIPE_CUBE_BMU_L0A] = 0;
    pipeExecuteTime[CorePipeType::PIPE_CUBE_BMU_L0B] = 0;
    pipeExecuteTime[CorePipeType::PIPE_CUBE_BMU_L0C] = 0;
    pipeExecuteTime[CorePipeType::PIPE_MTE_IN] = 0;
    pipeExecuteTime[CorePipeType::PIPE_MTE1] = 0;
    pipeExecuteTime[CorePipeType::PIPE_VECTOR_ALU] = 0;
    pipeExecuteTime[CorePipeType::PIPE_CUBE] = 0;
    pipeExecuteTime[CorePipeType::PIPE_MTE_OUT] = 0;
}

Json Function::DumpExecuteInfo()
{
    Json res;
    res["FuncName"] = funcName;
    res["FuncHash"] = functionHash;
    res["TotalCycles"] = totalCycles;
    Json pipe;
    for (auto& entry : pipeExecuteTime) {
        pipe[CorePipeName(entry.first)] = entry.second;
    }
    res["pipes"] = pipe;
    return res;
}

uint64_t Function::GetOpRelativeReadyCycle(TileOpPtr tileOp, uint64_t newBaseCycle)
{
    uint64_t relativeStartCycle = tileOp->exeInfo.cycleInfo.executeStartCycle - startCycles;
    uint64_t pipeFreeCycle = pipeLastEndCycle[tileOp->pipeType];
    uint64_t res = newBaseCycle + relativeStartCycle; // base start cycle;
    res = std::max(res, pipeFreeCycle);
    for (auto& srcTile : tileOp->iOperand) {
        for (auto& producer : srcTile->producers) {
            res = std::max(res, producer->exeInfo.cycleInfo.relativeEndCycle);
        }
    }
    return res;
}

void Function::CalculateRelativeCycle(uint64_t newBaseCycle, double proportion)
{
    pipeLastEndCycle.clear();
    for (const auto& m : opMagicSequence) {
        auto tileOp = tileOpMap[m];
        uint64_t simCycle = tileOp->exeInfo.cycleInfo.executeEndCycle - tileOp->exeInfo.cycleInfo.executeStartCycle;
        uint64_t realCycle = simCycle;
        if (IsMTEPipe(tileOp->pipeType)) {
            // scale mte tile op cycles
            realCycle = uint64_t(double(simCycle) * proportion);
        }
        uint64_t readyCycle = GetOpRelativeReadyCycle(tileOp, newBaseCycle);
        tileOp->exeInfo.cycleInfo.relativeStartCycle = readyCycle;
        tileOp->exeInfo.cycleInfo.relativeEndCycle = readyCycle + realCycle;
        pipeLastEndCycle[tileOp->pipeType] = tileOp->exeInfo.cycleInfo.relativeEndCycle;
    }
}
} // namespace CostModel
