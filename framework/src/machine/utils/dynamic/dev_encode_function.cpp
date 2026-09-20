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
 * \file dev_encode_function.cpp
 * \brief
 */

#include "machine/utils/dynamic/dev_encode_function.h"
#include "machine/utils/dynamic/dev_cell_match_mem_layout.h"

namespace npu::tile_fwk::dynamic {
namespace {
std::string DumpSymInt(const SymInt& s, const uint64_t* runtimeExpressionList)
{
    std::ostringstream oss;
    if (s.IsExpression()) {
        if (runtimeExpressionList == nullptr) {
            oss << "?" << s.Value();
        } else {
            oss << runtimeExpressionList[s.Value()];
        }
    } else {
        oss << s.Value();
    }
    return oss.str();
}

std::string DumpSymIntList(const SymInt* s, int count, uint64_t* runtimeExpressionList)
{
    std::ostringstream oss;
    oss << "<";
    for (int i = 0; i < count; i++) {
        oss << Delim(i != 0, ",") << DumpSymInt(s[i], runtimeExpressionList);
    }
    oss << ">";
    return oss.str();
}
} // namespace

std::string DevAscendFunction::DumpTensor(int tensorIndex) const
{
    std::ostringstream oss;
    oss << "%" << tensorIndex << "@" << GetTensor(tensorIndex)->rawIndex;
    return oss.str();
}

std::string DevAscendFunction::DumpOperationAttr(int operationIndex, uint64_t* runtimeExpressionList,
                                                 bool dumpIndex) const
{
    std::ostringstream oss;
    oss << SchemaGetCoa(operationIndex, runtimeExpressionList, dumpIndex).Dump();
    oss << " " << schema::pred(GetOperationDepGraphPredCount(operationIndex)).Dump();
    const DevLocalVector<uint32_t>& succList = GetOperationDepGraphSuccList(operationIndex);
    std::vector<schema::operation> succDataList;
    for (size_t j = 0; j < succList.size(); j++) {
        succDataList.push_back(At(succList, j));
    }
    oss << " " << schema::succ(succDataList).Dump();

    const DevLocalVector<int>& copyOutResolveCounterIndexList = GetOperationDepGraphCopyOutResolveSuccIndexList(
        operationIndex);
    std::vector<int64_t> outSuccIndexDataList;
    for (size_t j = 0; j < copyOutResolveCounterIndexList.size(); j++) {
        outSuccIndexDataList.push_back(At(copyOutResolveCounterIndexList, j));
    }
    oss << " " << schema::outSuccIndex(outSuccIndexDataList).Dump();
    oss << " "
        << "#stitchIndex{" << GetOperationStitchIndex(operationIndex) << "}";
    return oss.str();
}

std::string DevAscendFunction::DumpOperation(int operationIndex, int& totalAttrStartIdx,
                                             const std::vector<uintdevptr_t>& ooperandAddrList,
                                             const std::vector<uintdevptr_t>& ioperandAddrList,
                                             uint64_t* runtimeExpressionList) const
{
    std::ostringstream oss;
    for (size_t j = 0; j < GetOperationOOperandSize(operationIndex); j++) {
        oss << Delim(j != 0, ",") << DumpTensor(GetOperationOOperandInfo(operationIndex, j).tensorIndex);
        if (j < ooperandAddrList.size()) {
            oss << AddressDescriptor::DumpAddress(ooperandAddrList[j]);
        }
    }
    oss << " = "
        << "!" << operationIndex << " ";
    for (size_t j = 0; j < GetOperationIOperandSize(operationIndex); j++) {
        oss << Delim(j != 0, ",") << DumpTensor(GetOperationIOperandInfo(operationIndex, j).tensorIndex);
        if (j < ioperandAddrList.size()) {
            oss << AddressDescriptor::DumpAddress(ioperandAddrList[j]);
        }
    }

    oss << " " << DumpOperationAttr(operationIndex, runtimeExpressionList, true);
    totalAttrStartIdx += static_cast<int>(GetOperationAttrSize(operationIndex));
    return oss.str();
}

std::string DevAscendFunction::DumpRawTensor(int rawIndex, uintdevptr_t addr) const
{
    std::ostringstream oss;
    auto rawTensor = GetRawTensor(rawIndex);
    auto rawTensorDesc = GetRawTensorDesc(rawIndex);
    oss << rawTensor->DumpType() << " @" << rawIndex << "&" << rawTensor->linkedIncastId << " = ";
    oss << rawTensor->DumpAttr() << " ";
    oss << DevAscendRawTensor::DumpAttrDesc(rawTensorDesc);
    if (addr != 0) {
        oss << AddressDescriptor::DumpAddress(addr);
    }
    return oss.str();
}

std::string DevAscendFunction::DumpIncast(int incastIndex, const std::string& indent, uint64_t* runtimeExpressionList,
                                          const std::vector<uintdevptr_t>& slotAddrList) const
{
    std::ostringstream oss;
    const DevAscendFunctionIncast& incast = GetIncast(incastIndex);
    oss << "#incast:" << incastIndex << " = " << DumpTensor(incast.tensorIndex);
    for (size_t j = 0; j < incast.fromSlotList.size(); j++) {
        int slot = At(incast.fromSlotList, j);
        oss << " <- #slot:" << slot;
        if (slot < static_cast<int>(slotAddrList.size())) {
            oss << AddressDescriptor::DumpAddress(slotAddrList[slot]);
        }
    }
    oss << "\n";
    oss << indent;
    oss << " | #cellMatchTableDesc:" << DumpCellMatchTableDesc(incast.cellMatchTableDesc);
    oss << "\n";

    oss << indent << " | #stitchPolicyFullCoverConsumerAllOpIdxList:[";
    for (size_t j = 0; j < incast.stitchPolicyFullCoverConsumerAllOpIdxList.size(); j++) {
        oss << Delim(j != 0, ",") << At(incast.stitchPolicyFullCoverConsumerAllOpIdxList, j);
    }
    oss << "]\n";

    auto dumpConsumer = [this, &oss, &indent, &incast, runtimeExpressionList](
                            const DevAscendFunctionCallOperandUse& consumer, const char* tag) {
        int consumerIdx = consumer.operationIdx;
        int offsetAttrIdx = consumer.offsetAttrIdx;
        int shapeAttrIdx = offsetAttrIdx + incast.dim;
        oss << indent;
        oss << " | #" << tag << ":!" << consumerIdx;
        oss << " | #offsetAttrIdx:" << offsetAttrIdx;
        oss << " | #shapeAttrIdx:" << shapeAttrIdx;
        oss << " | #offsetAttr:" << DumpSymIntList(GetSymoffset(offsetAttrIdx), incast.dim, runtimeExpressionList);
        oss << " | #shapeAttr:" << DumpSymIntList(GetSymoffset(shapeAttrIdx), incast.dim, runtimeExpressionList);
        oss << "\n";
    };
    for (size_t j = 0; j < incast.consumerList.size(); j++) {
        dumpConsumer(At(incast.consumerList, j), "partialConsumerIdx");
    }
    for (size_t j = 0; j < incast.stitchPolicyFullCoverConsumerList.size(); j++) {
        dumpConsumer(At(incast.stitchPolicyFullCoverConsumerList, j), "fullCoverConsumerIdx");
    }
    return oss.str();
}

std::string DevAscendFunction::DumpOutcast(int outcastIndex, const std::string& indent, uint64_t* runtimeExpressionList,
                                           const std::vector<uintdevptr_t>& slotAddrList) const
{
    std::ostringstream oss;
    const DevAscendFunctionOutcast& outcast = GetOutcast(outcastIndex);
    auto dumpProducer = [this, &oss, &indent, &outcast, &runtimeExpressionList](
                            const DevLocalVector<DevAscendFunctionCallOperandUse>& producerList) -> void {
        for (size_t j = 0; j < producerList.size(); j++) {
            auto& producer = At(producerList, j);
            int producerIdx = producer.operationIdx;
            int offsetAttrIdx = producer.offsetAttrIdx;
            int shapeAttrIdx = offsetAttrIdx + outcast.dim;
            oss << indent;
            oss << " | #opIdx:!" << producerIdx;
            oss << " | #opType:" << (producer.opType == CellMatchOpType::READ ? "consumer" : "producer");
            oss << " | #offsetAttrIdx:" << offsetAttrIdx;
            oss << " | #shapeAttrIdx:" << shapeAttrIdx;
            oss << " | #offsetAttr:" << DumpSymIntList(GetSymoffset(offsetAttrIdx), outcast.dim, runtimeExpressionList);
            oss << " | #shapeAttr:" << DumpSymIntList(GetSymoffset(shapeAttrIdx), outcast.dim, runtimeExpressionList);
            oss << "\n";
        }
    };

    oss << "#outcast:" << outcastIndex << " = " << DumpTensor(outcast.tensorIndex);
    for (size_t j = 0; j < outcast.toSlotList.size(); j++) {
        int slot = At(outcast.toSlotList, j);
        oss << " -> #slot:" << slot;
        if (slot < static_cast<int>(slotAddrList.size())) {
            oss << AddressDescriptor::DumpAddress(slotAddrList[slot]);
        }
    }
    oss << "\n" << indent;
    oss << " | #cellMatchTableDesc:" << DumpCellMatchTableDesc(outcast.cellMatchTableDesc);
    oss << " | #cellMatchFullUpdateTable:" << outcast.cellMatchRuntimeFullUpdateTable.size();
    oss << "\n";

    oss << indent << " | #stitchPolicyFullCoverProducerList:[";
    for (size_t j = 0; j < outcast.stitchPolicyFullCoverProducerList.size(); j++) {
        oss << Delim(j != 0, ",") << At(outcast.stitchPolicyFullCoverProducerList, j).operationIdx;
    }
    oss << "]\n";
    dumpProducer(outcast.stitchPolicyFullCoverProducerList);

    oss << indent << " | #stitchPolicyFullCoverProducerHubOpIdx:" << outcast.stitchPolicyFullCoverProducerHubOpIdx
        << "\n";
    oss << indent << " | #stitchPolicyFullCoverAllOpIdxList:[";
    for (size_t j = 0; j < outcast.stitchPolicyFullCoverAllOpIdxList.size(); j++) {
        oss << Delim(j != 0, ",") << At(outcast.stitchPolicyFullCoverAllOpIdxList, j);
    }
    oss << "]\n";
    dumpProducer(outcast.producerConsumerList);

    return oss.str();
}

std::string DevAscendFunction::Dump(int indent) const
{
    std::string INDENT(indent, ' ');
    std::string INDENTINNER(indent + IDENT_SIZE, ' ');
    std::ostringstream oss;

    oss << INDENT << "DevFunction " << funcKey;
    oss << " " << schema::name(GetRawName()).Dump();
    oss << " " << schema::mem(rootInnerTensorWsMemoryRequirement).Dump();
    oss << " " << schema::memOut(exclusiveOutcastWsMemoryRequirement).Dump();
    oss << " {\n";
    for (size_t i = 0; i < GetRawTensorSize(); i++) {
        oss << INDENTINNER << DumpRawTensor(i) << "\n";
    }
    for (size_t i = 0; i < GetIncastSize(); i++) {
        oss << INDENTINNER << DumpIncast(i, INDENTINNER) << "\n";
    }
    for (size_t i = 0; i < GetOutcastSize(); i++) {
        oss << INDENTINNER << DumpOutcast(i, INDENTINNER) << "\n";
    }

    {
        oss << INDENTINNER << "#assembleSlotSize{" << GetRedaccAssembleSlotListSize() << "}\n";
        for (size_t j = 0; j < GetRedaccAssembleSlotListSize(); j++) {
            oss << INDENTINNER << "#assembleSlot_" << j << "{" << GetRedaccAssembleSlotList(j) << "}\n";
        }
    }
    {
        oss << INDENTINNER << "#updateOutCastSlotList{" << updateOutCastSlotList.size() << "}\n";
        for (size_t j = 0; j < updateOutCastSlotList.size(); j++) {
            auto& e = At(updateOutCastSlotList, j);
            oss << INDENTINNER << "#updateOutCast_" << j << "{slot=" << e.slotIdx << ",outcast=" << e.outcastIdx
                << "}\n";
        }
        oss << INDENTINNER << "#stitchIncastSlotList{" << stitchIncastSlotList.size() << "}\n";
        for (size_t j = 0; j < stitchIncastSlotList.size(); j++) {
            auto& e = At(stitchIncastSlotList, j);
            oss << INDENTINNER << "#stitchIncast_" << j << "{slot=" << e.slotIdx << ",incast=" << e.incastIdx << "}\n";
        }
        oss << INDENTINNER << "#updateIncastSlotList{" << updateIncastSlotList.size() << "}\n";
        for (size_t j = 0; j < updateIncastSlotList.size(); j++) {
            auto& e = At(updateIncastSlotList, j);
            oss << INDENTINNER << "#updateIncast_" << j << "{slot=" << e.slotIdx << ",incast=" << e.incastIdx << "}\n";
        }
        oss << INDENTINNER << "#stitchOutCastSlotList{" << stitchOutCastSlotList.size() << "}\n";
        for (size_t j = 0; j < stitchOutCastSlotList.size(); j++) {
            auto& e = At(stitchOutCastSlotList, j);
            oss << INDENTINNER << "#stitchOutCast_" << j << "{slot=" << e.slotIdx << ",outcast=" << e.outcastIdx
                << "}\n";
        }
    }
    oss << INDENTINNER << "#hubOpCount:" << hubOpCount_ << "\n";
    oss << INDENTINNER << "#zeropred:" << predInfo_.totalZeroPred << "\n";
    oss << INDENTINNER << "#zeropred-aiv:" << predInfo_.totalZeroPredAIV << "\n";
    oss << INDENTINNER << "#zeropred-aic:" << predInfo_.totalZeroPredAIC << "\n";
    oss << INDENTINNER << "#zeropred-aicpu:" << predInfo_.totalZeroPredAicpu << "\n";
    int totalAttrStartIdx = 0;
    for (size_t i = 0; i < GetOperationSize(); i++) {
        oss << INDENTINNER << DumpOperation(i, totalAttrStartIdx) << "\n";
    }
    oss << INDENT << "}";
    return oss.str();
}

int DevAscendFunction::LookupIncastBySlotIndex(int slotIndex) const
{
    for (size_t incastIndex = 0; incastIndex < GetIncastSize(); incastIndex++) {
        const DevAscendFunctionIncast& incast = GetIncast(incastIndex);
        for (size_t fromIndex = 0; fromIndex < incast.fromSlotList.size(); fromIndex++) {
            int slot = At(incast.fromSlotList, fromIndex);
            if (slot == slotIndex) {
                return static_cast<int>(incastIndex);
            }
        }
    }
    return INVALID_INDEX;
}

std::vector<int> DevAscendFunction::LookupIncastBySlotIndexList(const std::vector<int>& slotIndexList) const
{
    std::vector<int> resultList(slotIndexList.size());
    for (size_t i = 0; i < slotIndexList.size(); i++) {
        resultList[i] = LookupIncastBySlotIndex(slotIndexList[i]);
    }
    return resultList;
}

int DevAscendFunction::LookupOutcastBySlotIndex(int slotIndex) const
{
    for (size_t outcastIndex = 0; outcastIndex < GetOutcastSize(); outcastIndex++) {
        const DevAscendFunctionOutcast& outcast = GetOutcast(outcastIndex);
        for (size_t toIndex = 0; toIndex < outcast.toSlotList.size(); toIndex++) {
            int slot = At(outcast.toSlotList, toIndex);
            if (slot == slotIndex) {
                return static_cast<int>(outcastIndex);
            }
        }
    }
    return INVALID_INDEX;
}

std::vector<int> DevAscendFunction::LookupOutcastBySlotIndexList(const std::vector<int>& slotIndexList) const
{
    std::vector<int> resultList(slotIndexList.size());
    for (size_t i = 0; i < slotIndexList.size(); i++) {
        resultList[i] = LookupOutcastBySlotIndex(slotIndexList[i]);
    }
    return resultList;
}

void DevAscendFunction::AppendOutcastConnections(std::vector<std::tuple<int, int, int>>& connectionList, int fromSlot,
                                                 int incastIndex, const DevAscendFunction* func) const
{
    for (size_t outcastIndex = 0; outcastIndex < func->GetOutcastSize(); outcastIndex++) {
        const DevAscendFunctionOutcast& outcast = func->GetOutcast(outcastIndex);
        for (size_t toIndex = 0; toIndex < outcast.toSlotList.size(); toIndex++) {
            int toSlot = func->At(outcast.toSlotList, toIndex);
            if (fromSlot == toSlot) {
                connectionList.push_back(std::tuple(outcastIndex, incastIndex, fromSlot));
            }
        }
    }
}

std::vector<std::tuple<int, int, int>> DevAscendFunction::LookupConnectionSlotIndexFrom(
    const DevAscendFunction* func) const
{
    std::vector<std::tuple<int, int, int>> connectionList;
    for (size_t incastIndex = 0; incastIndex < GetIncastSize(); incastIndex++) {
        const DevAscendFunctionIncast& incast = GetIncast(incastIndex);
        for (size_t fromIndex = 0; fromIndex < incast.fromSlotList.size(); fromIndex++) {
            int fromSlot = At(incast.fromSlotList, fromIndex);
            AppendOutcastConnections(connectionList, fromSlot, incastIndex, func);
        }
    }
    return connectionList;
}
} // namespace npu::tile_fwk::dynamic
