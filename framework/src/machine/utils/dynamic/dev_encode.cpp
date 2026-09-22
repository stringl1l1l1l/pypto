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
 * \file dev_encode.cpp
 * \brief
 */
#include "tilefwk/platform.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "machine/utils/dynamic/dev_encode_workspace.h"
#include "machine/utils/dynamic/workspace_budget_calculator.h"
#include "machine/utils/dynamic/dev_encode_function_stitch.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "machine/utils/dynamic/dev_cell_match_mem_layout.h"
#include "machine/host/main_block.h"
#include "machine/host/dump_host_topo.h"

#include "interface/operation/attribute.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/tensor_slot.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "interface/function/rebuildable_attribute.h"
#include "machine/utils/dynamic/rebuildable_workspace_desc.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "interface/configs/config_manager_ng.h"
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_code.h"

#include <stdexcept>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <queue>
#include <sstream>
using namespace npu::tile_fwk;
namespace npu::tile_fwk {
namespace dynamic {
#define ONFILLCONTENT if (fillContent)
#ifndef PAGE_SIZE
#define PAGE_SIZE 4096
#endif

constexpr int32_t CALLOP_ARG_ATTR_BASE_INDEX = 1;
constexpr int32_t MINI_TILE_LIST_SIZE_THRESHOLD = 16;
constexpr int64_t MAX_SHAPE_WARN_THRESHOLE = 512 * 512;
constexpr int64_t DEFAULT_CACHE_DEVICE_TASK_NUM = 10000;
constexpr uint32_t FRIENDLY_CACHE_ALIGN_U64_SIZE = 2; // 友好的cache对齐是2个u64
constexpr int INVALID_WRAPID = -1;

static bool IsComputeDeterminismEnabled()
{
    return ConfigManagerNg::GetGlobalConfig<int64_t>("compute_determinism_level") >= 1;
}

void DevAscendFunction::InitIncastOutcastAttr(uintdevptr_t& initOffset,
                                              const std::vector<std::shared_ptr<LogicalTensor>>& iList,
                                              const std::vector<std::shared_ptr<LogicalTensor>>& oList,
                                              bool /* fillContent */)
{
    incastAddressList.HostInitDataSizeOffset(initOffset, iList.size());
    outcastAddressList.HostInitDataSizeOffset(initOffset, oList.size());
}

static void FillDuppedDataFields(DevAscendFunctionDuppedData* dupData, uint64_t operationSize, uint64_t incastSize,
                                 uint64_t outcastSize, uint64_t expressionSize, uint32_t stitchCount,
                                 uint64_t predCountListDataSize, uint64_t incastDataSize, uint64_t outcastDataSize,
                                 uint64_t expressionDataSize, uint64_t stitchDataSize, uint64_t totalDataSize,
                                 uint64_t duppedDataAllocSize)
{
    dupData->operationList_.size = operationSize;

    uint64_t offset = 0;
    dupData->operationList_.predCountBase = offset;
    offset += predCountListDataSize;

    dupData->incastList_.size = incastSize;
    dupData->incastList_.base = offset;
    offset += incastDataSize;

    dupData->outcastList_.size = outcastSize;
    dupData->outcastList_.base = offset;
    offset += outcastDataSize;

    dupData->expressionList_.size = expressionSize;
    dupData->expressionList_.base = offset;
    offset += expressionDataSize;

    dupData->operationList_.stitchBase = offset;
    dupData->operationList_.stitchCount = stitchCount;
    offset += stitchDataSize;
    ASSERT(DevCommonErr::PARAM_CHECK_FAILED, offset == totalDataSize)
        << "Offset mismatch: offset " << offset << " != totalDataSize " << totalDataSize;

    memset_s(dupData->data_, totalDataSize, 0, totalDataSize);

    uint8_t* dataEnd = &dupData->data_[totalDataSize];
    uint8_t* dataEndAlloc = &dupData->data_[duppedDataAllocSize - sizeof(DevAscendFunctionDuppedData)];
    ASSERT(DevCommonErr::PARAM_CHECK_FAILED, dataEnd == dataEndAlloc)
        << "Pointer mismatch: dataEnd " << dataEnd << " != dataEndAlloc " << dataEndAlloc;
}

static void VerifyDuppedDataRanges(DevAscendFunctionDuppedData* dupData, uint64_t operationSize, uint64_t incastSize,
                                   uint64_t outcastSize, uint64_t expressionSize)
{
    uint8_t* dataBegin = &dupData->data_[0];
    uint8_t* incastBegin = &dupData->data_[dupData->incastList_.base];
    uint8_t* outcastBegin = &dupData->data_[dupData->outcastList_.base];
    uint8_t* expressionBegin = &dupData->data_[dupData->expressionList_.base];
    uint8_t* stitchBegin = &dupData->data_[dupData->operationList_.stitchBase];

    for (uint64_t i = 0; i < operationSize; i++) {
        uint8_t* ptr = reinterpret_cast<uint8_t*>(&dupData->GetOperationCurrPredCount(i));
        ASSERT(ProgEncodeErr::RANGE_VERIFY_FAILED, dataBegin <= ptr && ptr < incastBegin)
            << "OperationCurrPredCount out of range:  ptr " << ptr << " not in [" << dataBegin << ", " << incastBegin
            << ")";
    }
    for (uint64_t i = 0; i < incastSize; i++) {
        uint8_t* ptr = reinterpret_cast<uint8_t*>(&dupData->GetIncastAddress(i));
        ASSERT(ProgEncodeErr::RANGE_VERIFY_FAILED, incastBegin <= ptr && ptr < outcastBegin)
            << "Incast address out of range:  ptr " << ptr << " not in [" << incastBegin << ", " << outcastBegin << ")";
    }
    for (uint64_t i = 0; i < outcastSize; i++) {
        uint8_t* ptr = reinterpret_cast<uint8_t*>(&dupData->GetOutcastAddress(i));
        ASSERT(ProgEncodeErr::RANGE_VERIFY_FAILED, outcastBegin <= ptr && ptr < expressionBegin)
            << "Outcast address out of range:  ptr " << ptr << " not in [" << outcastBegin << ", " << expressionBegin
            << ")";
    }
    for (uint64_t i = 0; i < expressionSize; i++) {
        uint8_t* ptr = reinterpret_cast<uint8_t*>(&dupData->GetExpression(i));
        ASSERT(ProgEncodeErr::RANGE_VERIFY_FAILED, expressionBegin <= ptr && ptr < stitchBegin)
            << "Expression address out of range:  ptr " << ptr << " not in [" << expressionBegin << ", " << stitchBegin
            << ")";
    }
}

void DevAscendFunction::InitOperationDynamicField(
    uintdevptr_t& initOffset, DevAscendFunctionPredInfo predInfo, uint32_t stitchCount,
    [[maybe_unused]] const std::unordered_map<uint64_t, int>& calleeHashIndexDict,
    const SymbolicExpressionTable* expressionTable, const OrderedSet<Operation*>& callList,
    const std::vector<std::shared_ptr<LogicalTensor>>& incastTensorList,
    const std::vector<std::shared_ptr<LogicalTensor>>& outcastTensorList,
    const std::unordered_map<Operation*, OrderedSet<Operation*>>& /* callOpSuccDict */, bool fillContent)
{
    expressionList.HostInitDataSizeOffset(initOffset, expressionTable->GetPrimaryExpressionSize());

    uint64_t operationSize = callList.size();
    uint64_t incastSize = incastTensorList.size();
    uint64_t outcastSize = outcastTensorList.size();
    uint64_t expressionSize = expressionTable->GetPrimaryExpressionSize();

    uint64_t predCountListDataSize = PRED_COUNT_PINGPONG_NUM *
                                     AlignUp(operationSize * sizeof(predcount_t), sizeof(uint64_t));
    uint64_t incastDataSize = AlignUp(incastSize * sizeof(void*), sizeof(uint64_t));
    uint64_t outcastDataSize = AlignUp(outcastSize * sizeof(void*), sizeof(uint64_t));
    uint64_t expressionDataSize = AlignUp(expressionSize * sizeof(uint64_t), sizeof(uint64_t));
    uint64_t stitchDataSize = AlignUp(stitchCount * sizeof(DevAscendFunctionDuppedStitchList), sizeof(uint64_t));
    uint64_t totalDataSize = predCountListDataSize + incastDataSize + outcastDataSize + expressionDataSize +
                             stitchDataSize;
    duppedDataAllocSize_ = sizeof(DevAscendFunctionDuppedData) + totalDataSize;
    duppedDataCopySize_ = sizeof(DevAscendFunctionDuppedData) + predCountListDataSize;
    duppedData_.HostInitDataSizeOffset(initOffset, duppedDataAllocSize_);
    predInfo_ = predInfo;
    MACHINE_LOGI(
        "Pred: zero= %lu aiv= %lu aic= %lu hub= %lu aicpu=%lu", static_cast<unsigned long>(predInfo.totalZeroPred),
        static_cast<unsigned long>(predInfo.totalZeroPredAIV), static_cast<unsigned long>(predInfo.totalZeroPredAIC),
        static_cast<unsigned long>(predInfo.totalZeroPredHub), static_cast<unsigned long>(predInfo.totalZeroPredAicpu));

    ONFILLCONTENT
    {
        DevAscendFunctionDuppedData* dupData = reinterpret_cast<DevAscendFunctionDuppedData*>(&At(duppedData_, 0));
        FillDuppedDataFields(dupData, operationSize, incastSize, outcastSize, expressionSize, stitchCount,
                             predCountListDataSize, incastDataSize, outcastDataSize, expressionDataSize, stitchDataSize,
                             totalDataSize, duppedDataAllocSize_);
        VerifyDuppedDataRanges(dupData, operationSize, incastSize, outcastSize, expressionSize);
    }
}

void HandleActualRaw(const OrderedSet<std::shared_ptr<RawTensor>>& incastRawList,
                     const OrderedSet<std::shared_ptr<RawTensor>>& outcastRawList,
                     const std::unordered_map<int, std::shared_ptr<RawTensor>>& rawMagicToRawTensor,
                     const std::unordered_map<const RawTensor*, int>& rawToOutcastIdx,
                     const std::shared_ptr<RawTensor>& rawTensor, DevAscendRawTensor& encoded)
{
    auto iter = rawMagicToRawTensor.find(rawTensor->actualRawmagic);
    if (iter != rawMagicToRawTensor.end()) {
        if (iter->second->addrOffset == UINT64_MAX) {
            MACHINE_LOGE(ProgEncodeErr::ADDR_OFFSET_RAW_MAGIC_MISMATCH,
                         "addrOffset is invalid actual raw magic %d, original raw magic %d", rawTensor->actualRawmagic,
                         rawTensor->rawmagic);
            encoded.addrOffset = 0;
        } else {
            encoded.addrOffset = iter->second->addrOffset;
            if (outcastRawList.count(iter->second)) {
                auto ownerIter = rawToOutcastIdx.find(iter->second.get());
                encoded.ioIndex = ownerIter != rawToOutcastIdx.end() ?
                                      ownerIter->second :
                                      static_cast<int>(outcastRawList.GetIndex(iter->second));
                encoded.ioProperty = DevIOProperty::ROOT_OUTCAST;
            } else if (incastRawList.count(iter->second)) {
                encoded.ioIndex = incastRawList.GetIndex(iter->second);
                encoded.ioProperty = DevIOProperty::ROOT_INCAST;
            } else {
                encoded.ioIndex = -1;
                encoded.ioProperty = DevIOProperty::NONE;
            }
            MACHINE_LOGD("Tensor %d use tensor %d's addr io index %d", rawTensor->rawmagic, rawTensor->actualRawmagic,
                         encoded.ioIndex);
        }
    }
}

void DevAscendFunction::UpdateRawTensorDesc(const std::shared_ptr<RawTensor>& rawTensor, size_t i,
                                            size_t incastRawListSize, DevAscendRawTensor& encoded)
{
    if (rawTensor->actualRawmagic != -1) {
        MACHINE_LOGD("[%3zu] raw %d, actualRaw %d, IOType <%s>, addrOffset 0x%lx, ioIndex %d.", i, rawTensor->rawmagic,
                     rawTensor->actualRawmagic, DevIOProperty2String(encoded.ioProperty).c_str(), encoded.addrOffset,
                     encoded.ioIndex);
    }
    uint32_t location = 0;
    uint32_t offsetOrIndex = 0;
    switch (encoded.ioProperty) {
        case DevIOProperty::ROOT_INCAST:
            location = RAW_TENSOR_LOCATION_INCAST;
            offsetOrIndex = encoded.ioIndex;
            break;
        case DevIOProperty::ROOT_OUTCAST:
            location = RAW_TENSOR_LOCATION_OUTCAST;
            offsetOrIndex = encoded.ioIndex + incastRawListSize;
            break;
        case DevIOProperty::NONE:
            location = RAW_TENSOR_LOCATION_LOCAL;
            offsetOrIndex = encoded.addrOffset;
            break;
        default:
            ASSERT(DevCommonErr::PARAM_INVALID, false)
                << "Unexpected ioProperty value: " << static_cast<int>(encoded.ioProperty);
            break;
    }
    DevRawTensorDesc* desc = GetRawTensorDesc(i);
    desc->location = location;
    desc->offsetOrIndex = offsetOrIndex;
}

static int64_t GetShapeSizeSafe(const std::vector<int64_t>& shape)
{
    int64_t nelm = shape.empty() ? 0 : 1;
    for (auto x : shape) {
        if (x == -1) {
            return 0;
        }
        nelm *= x;
    }
    return nelm;
}

static std::string FormatShape(const std::vector<int64_t>& shape)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); i++) {
        if (i > 0) {
            oss << ", ";
        }
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

static void EncodeRawShape(const SymbolicExpressionTable* expressionTable, DevAscendRawTensor* encoded,
                           std::shared_ptr<RawTensor> rawTensor, bool needIndependentlyAlloc,
                           const std::string rootName = "")
{
    std::vector<SymInt> shape;
    bool isDyn = false;
    for (auto x : rawTensor->GetDynRawShape()) {
        if (x.IsImmediate()) {
            shape.emplace_back(x.Concrete());
        } else {
            shape.emplace_back(true, expressionTable->LookupPrimaryExpressionIndex(x));
            isDyn = true;
        }
    }
    encoded->shape.SetShape(shape);
    encoded->dataType = rawTensor->GetDataType();
    encoded->memoryRequirement = isDyn ? 0 : AlignUp(rawTensor->GetRawDataSize(), TENSOR_ADDR_ALIGNMENT);

    if (!needIndependentlyAlloc) {
        encoded->maxStaticMemReq = 0;
        return;
    }

    int64_t nelm = GetShapeSizeSafe(rawTensor->rawshape);
    encoded->maxStaticMemReq = AlignUp(DataSizeOf(nelm, rawTensor->GetDataType()), TENSOR_ADDR_ALIGNMENT);
    if (nelm > MAX_SHAPE_WARN_THRESHOLE) {
        MACHINE_LOGW(
            "[workspaceSize] Root=[%s], symbol=[%s],rawmagic=[%d]: staticMemReq=[%lu] bytes is too large, which might "
            "indicate an error",
            rootName.c_str(), rawTensor->symbol.c_str(), rawTensor->GetRawMagic(), encoded->maxStaticMemReq);
    }
}

static bool ShouldDropBudget(const OrderedSet<std::shared_ptr<RawTensor>>& outcastRawList,
                             const IncastOutcastSlot* slotInfo,
                             const std::vector<npu::tile_fwk::RuntimeSlotKindSet>& runtimeSlotKindSetList,
                             const std::shared_ptr<RawTensor> rawTensor, const DevAscendRawTensor& encoded)
{
    if (!outcastRawList.count(rawTensor)) {
        return false;
    }

    for (int slotIdx : slotInfo->outcastSlot[encoded.ioIndex]) {
        if (runtimeSlotKindSetList[slotIdx].Contains(RuntimeSlotKind::ADDRESS_EXPRESSION)) {
            return true;
        }
    }
    return false;
}

static bool HasInputOutputOrAssembleDst(const std::vector<int>& outcastSlots,
                                        const std::vector<npu::tile_fwk::RuntimeSlotKindSet>& runtimeSlotKindSetList)
{
    for (int slotIdx : outcastSlots) {
        if (runtimeSlotKindSetList[slotIdx].Contains(RuntimeSlotKind::INPUT) ||
            runtimeSlotKindSetList[slotIdx].Contains(RuntimeSlotKind::OUTPUT) ||
            runtimeSlotKindSetList[slotIdx].Contains(RuntimeSlotKind::ASSEMBLE_OUTCAST)) {
            return true;
        }
    }
    return false;
}

void DevAscendFunction::InitRawTensorAndMemoryRequirement(
    uintdevptr_t& initOffset, const OrderedSet<std::shared_ptr<RawTensor>>& incastRawList,
    const OrderedSet<std::shared_ptr<RawTensor>>& outcastRawList, const OrderedSet<std::shared_ptr<RawTensor>>& rawList,
    const std::unordered_map<int, std::shared_ptr<RawTensor>>& rawMagicToRawTensor,
    const std::vector<EncodeRawTensorAttr>& rawAttrs, const EncodeDevAscendFunctionParam& param,
    const SymbolicExpressionTable* expressionTable, bool fillContent)
{
    auto inoutLink = param.inoutLink;
    auto slot = param.slot;
    rawTensorList_.HostInitDataSizeOffset(initOffset, rawList.size());
    rawTensorDescList_.HostInitDataSizeOffset(initOffset, rawList.size());

    /* Assemble versions of one slot share a RawTensor, so outcastRawList is deduped while the
     * runtime outcast table keeps one entry per version. ioIndex must reference the owning
     * outcast entry (first occurrence in devRoot's outcast list), otherwise GetOutcastAddress
     * and COA location encode resolve the wrong outcast (pr !6322 precision regression). */
    std::unordered_map<const RawTensor*, int> rawToOutcastIdx;
    {
        const auto& rootOutcasts = param.devRoot->GetOutcast();
        for (size_t oi = 0; oi < rootOutcasts.size(); oi++) {
            rawToOutcastIdx.emplace(rootOutcasts[oi]->GetRawTensor().get(), static_cast<int>(oi));
        }
    }

    ONFILLCONTENT
    {
        const std::vector<RuntimeSlotKindSet>& runtimeSlotKindSetList = inoutLink->runtimeSlotKindSetList;
        MACHINE_LOGD("incast raw size %zu, outcast raw size %zu, rawlist size %zu", incastRawList.size(),
                     outcastRawList.size(), rawList.size());
        rootInnerTensorWsMemoryRequirement = 0;
        exclusiveOutcastWsMemoryRequirement = 0;
        for (size_t idx = 0; idx < rawList.size(); idx++) {
            const auto& rawTensor = rawList[idx];
            auto& encoded = *GetRawTensor(idx);
            // shmem need to drop budget
            bool dropBudget = ShouldDropBudget(outcastRawList, slot, runtimeSlotKindSetList, rawTensor, encoded);
            // inplace (basically reshape) need to drop budget
            bool isInplace = param.devRoot->outIncastLinkMap.count(rawTensor);
            isInplace |= rawTensor->actualRawmagic != -1 && rawTensor->actualRawmagic != rawTensor->rawmagic;
            EncodeRawShape(expressionTable, &encoded, rawTensor, !dropBudget && !isInplace,
                           param.devRoot->GetRawName());
        }
        for (size_t idx = 0; idx < rawList.size(); idx++) {
            const auto& rawTensor = rawList[idx];
            auto& encoded = *GetRawTensor(idx);
            encoded.rawMagic = rawTensor->GetRawMagic();
            if (rawTensor->actualRawmagic != -1 && rawTensor->actualRawmagic != rawTensor->rawmagic) {
                continue;
            }
            if (incastRawList.count(rawTensor)) {
                // No need to allocate memory for root incasts
                encoded.ioProperty = DevIOProperty::ROOT_INCAST;
                encoded.ioIndex = incastRawList.GetIndex(rawTensor);
                rawTensor->addrOffset = 0;
            } else if (outcastRawList.count(rawTensor)) {
                encoded.ioProperty = DevIOProperty::ROOT_OUTCAST;
                auto ownerIter = rawToOutcastIdx.find(rawTensor.get());
                encoded.ioIndex = ownerIter != rawToOutcastIdx.end() ?
                                      ownerIter->second :
                                      static_cast<int>(outcastRawList.GetIndex(rawTensor));
                rawTensor->addrOffset = 0;
                if (!HasInputOutputOrAssembleDst(slot->outcastSlot[encoded.ioIndex], runtimeSlotKindSetList)) {
                    encoded.addrOffset = exclusiveOutcastWsMemoryRequirement;
                    rawTensor->addrOffset = exclusiveOutcastWsMemoryRequirement;
                    exclusiveOutcastWsMemoryRequirement += encoded.maxStaticMemReq;
                }
            } else {
                // For workspace tensors, the memoryRequirement property is deprecated, please don't use its value
                encoded.ioProperty = DevIOProperty::NONE;
                encoded.ioIndex = -1;
#if DEBUG_INFINITE_LIFETIME
                UNUSED(rawAttrs);
                encoded.addrOffset = rootInnerTensorWsMemoryRequirement;
                rawTensor->addrOffset = encoded.addrOffset;
                rootInnerTensorWsMemoryRequirement += encoded.maxStaticMemReq;
#else
                ASSERT(DevCommonErr::NULLPTR, rawAttrs[idx].storage.get() != nullptr)
                    << "rawTensor(rawmagic:" << rawTensor->GetRawMagic()
                    << " )'s storage is null, should be incast or outcast tensor";
                encoded.addrOffset = rawAttrs[idx].storage->start_ + rawAttrs[idx].storageOffset;
                rawTensor->addrOffset = encoded.addrOffset;
                rootInnerTensorWsMemoryRequirement = std::max(
                    rootInnerTensorWsMemoryRequirement, rawAttrs[idx].storage->start_ + rawAttrs[idx].storage->length_);
#endif
            }
            /* Outcast offsetOrIndex indexes the combined per-entry address table
             * [incast entries..., outcast entries...]; the base is the incast ENTRY count
             * (devRoot->GetIncast().size()), not the deduped raw count. */
            UpdateRawTensorDesc(rawTensor, idx, param.devRoot->GetIncast().size(), encoded);
        }

        for (size_t i = 0; i < rawList.size(); i++) {
            const auto& rawTensor = rawList[i];
            if (rawTensor->actualRawmagic == -1 || rawTensor->actualRawmagic == rawTensor->rawmagic) {
                continue;
            }
            auto& encoded = *GetRawTensor(i);
            HandleActualRaw(incastRawList, outcastRawList, rawMagicToRawTensor, rawToOutcastIdx, rawTensor, encoded);
            UpdateRawTensorDesc(rawTensor, i, param.devRoot->GetIncast().size(), encoded);
        }

        for (size_t i = 0; i < rawList.size(); i++) {
            const auto& rawTensor = rawList[i];
            std::string rawShape = FormatShape(rawTensor->GetRawShape()).c_str();
            if (rawTensor->actualRawmagic != -1 && rawTensor->actualRawmagic != rawTensor->rawmagic) {
                auto it = rawMagicToRawTensor.find(rawTensor->actualRawmagic);
                ASSERT(ProgEncodeErr::RANGE_VERIFY_FAILED, it != rawMagicToRawTensor.end())
                    << "rawMagic is not found in rawMagicToRawTensor: " << rawTensor->actualRawmagic;
                auto& actualRaw = it->second;
                std::string actualrawShape = FormatShape(actualRaw->GetRawShape()).c_str();
                const auto& rawTensorRawShape = rawTensor->GetRawShape();
                bool isDynamicShape = std::find_if(rawTensorRawShape.begin(), rawTensorRawShape.end(),
                                                   [](int64_t dimShape) { return dimShape < 0; }) !=
                                      rawTensorRawShape.end();
                if (isDynamicShape)
                    continue;

                auto fromType = rawTensor->datatype;
                auto toType = actualRaw->datatype;
                if (fromType != toType) {
                    int inSize = BytesOf(fromType);
                    int outSize = BytesOf(toType);
                    ASSERT(DevCommonErr::PARAM_INVALID, inSize != 0 && outSize != 0)
                        << "Detected zero byte size data type, fromType: " << static_cast<int>(fromType)
                        << ", toType: " << static_cast<int>(toType);
                    if (inSize > outSize) {
                        ASSERT(DevCommonErr::PARAM_CHECK_FAILED,
                               (rawTensor->GetRawShapeSize() * (inSize / outSize)) == actualRaw->GetRawShapeSize())
                            << "Shape size mismatch: expected " << rawTensor->GetRawShapeSize() * (inSize / outSize)
                            << ", got: " << actualRaw->GetRawShapeSize();
                    } else {
                        ASSERT(DevCommonErr::PARAM_CHECK_FAILED,
                               rawTensor->GetRawShapeSize() == (actualRaw->GetRawShapeSize() * (outSize / inSize)))
                            << "Shape size mismatch: expected " << actualRaw->GetRawShapeSize() * (outSize / inSize)
                            << ", got " << rawTensor->GetRawShapeSize();
                    }
                    ASSERT(DevCommonErr::PARAM_CHECK_FAILED, rawTensor->GetRawDataSize() == actualRaw->GetRawDataSize())
                        << "Data size mismatch:" << rawTensor->GetRawDataSize() << "!=" << actualRaw->GetRawDataSize();
                    continue;
                }
                ASSERT(DevCommonErr::PARAM_CHECK_FAILED, rawTensor->GetRawShapeSize() == actualRaw->GetRawShapeSize())
                    << "Shape size mismatch:" << rawTensor->GetRawShapeSize() << "!=" << actualRaw->GetRawShapeSize()
                    << ", rootMagic=" << param.devRoot->GetMagicName()
                    << ", rootHash=" << param.devRoot->GetFunctionHash().GetHash() << ", rawShape=" << rawShape
                    << ", actualrawShape=" << actualrawShape << ", rawTensor->rawMagic=" << rawTensor->GetRawMagic()
                    << ", rawTensor->actualRawmagic=" << rawTensor->actualRawmagic
                    << ", actualRaw->rawMagic=" << actualRaw->rawmagic;
                ASSERT(DevCommonErr::PARAM_CHECK_FAILED, rawTensor->GetRawDataSize() == actualRaw->GetRawDataSize())
                    << "Data size mismatch:" << rawTensor->GetRawDataSize() << "!=" << actualRaw->GetRawDataSize()
                    << ", rootMagic=" << param.devRoot->GetMagicName()
                    << ", rootHash=" << param.devRoot->GetFunctionHash().GetHash() << ", rawShape=" << rawShape
                    << ", actualrawShape=" << actualrawShape << ", rawTensor->rawMagic=" << rawTensor->GetRawMagic()
                    << ", rawTensor->actualRawmagic=" << rawTensor->actualRawmagic
                    << ", actualRaw->rawMagic=" << actualRaw->rawmagic;
            }
        }

        // file linkedIncastId
        auto outIncastLinkMap = param.devRoot->outIncastLinkMap;
        MACHINE_LOGD("rootName is %s", param.devRoot->GetRawName().c_str());
        for (size_t i = 0; i < rawList.size(); i++) {
            auto& encoded = *GetRawTensor(i);
            if (outIncastLinkMap.find(rawList[i]) != outIncastLinkMap.end()) {
                ASSERT(DevCommonErr::PARAM_CHECK_FAILED,
                       outIncastLinkMap[rawList[i]]->actualRawmagic != rawList[i]->rawmagic)
                    << "Unexpected rawmagic match: actualRawmagic " << outIncastLinkMap[rawList[i]]->actualRawmagic
                    << " == rawmagic " << rawList[i]->rawmagic;
                auto replacedIncast = outIncastLinkMap[rawList[i]];
                if (std::find(rawList.begin(), rawList.end(), replacedIncast) != rawList.end()) {
                    encoded.linkedIncastId = incastRawList.GetIndex(replacedIncast); // 换成incast的下标 ioidx
                    MACHINE_LOGD("linkedIncastId is %d", encoded.linkedIncastId);
                } else {
                    encoded.linkedIncastId = -1;
                }
            } else {
                encoded.linkedIncastId = -1;
                MACHINE_LOGD("raw tensor linkedIncastId is %d", encoded.linkedIncastId);
            }
        }
    }; // ONFILLCONTENT
}

void DevAscendFunction::InitTensor(uintdevptr_t& initOffset, const OrderedSet<std::shared_ptr<LogicalTensor>>& tlist,
                                   const OrderedSet<std::shared_ptr<RawTensor>>& rawList, bool fillContent)
{
    tensorList_.HostInitDataSizeOffset(initOffset, tlist.size());
    for (size_t i = 0; i < tlist.size(); i++) {
        ONFILLCONTENT { GetTensor(i)->rawIndex = rawList.find(tlist[i]->tensor)->second; };
    }
}

static int GetCceIndex(const std::unordered_map<uint64_t, int>& calleeHashIndexDict,
                       const std::shared_ptr<CallOpAttribute>& callop)
{
    int cceIndex = calleeHashIndexDict.at(callop->GetCalleeHash().GetHash());
    bool enableVF = Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510;
    enableVF = enableVF && config::GetPassGlobalConfig(KEY_ENABLE_VF, false);
    if (config::GetRuntimeOption<int64_t>(CFG_VALID_SHAPE_OPTIMIZE) == 1 || enableVF) {
        cceIndex = std::max(0, cceIndex * MAIN_BLOCK_SIZE - 1);
    }
    return cceIndex;
}

void DevAscendFunction::InitOperation(
    uintdevptr_t& initOffset, const SymbolicExpressionTable* expressionTable, const OrderedSet<Operation*>& callList,
    const OrderedSet<std::shared_ptr<LogicalTensor>>& tlist, const OrderedSet<std::shared_ptr<RawTensor>>& rawList,
    const std::unordered_map<Operation*, uint64_t>& callOpPredDict,
    const std::unordered_map<Operation*, OrderedSet<Operation*>>& callOpSuccDict,
    const std::unordered_map<uint64_t, int>& calleeHashIndexDict, const std::vector<int32_t>& stitchIndexList,
    const std::vector<int>& noPredOpList, const std::vector<int>& noSuccOpList,
    const std::unordered_map<Operation*, std::vector<int>>& copyOutResolveSuccIndexListDict,
    const std::vector<CceCodeInfo>& cceCodeInfoList, bool fillContent)
{
    InitOperationNoPredNoSuccIndices(initOffset, callList, callOpPredDict, callOpSuccDict, noPredOpList, noSuccOpList,
                                     fillContent);
    InitOperationBufferLayouts(initOffset, callList, callOpSuccDict, copyOutResolveSuccIndexListDict);
    FillOperationEncodedContent(expressionTable, callList, tlist, rawList, callOpPredDict, callOpSuccDict,
                                calleeHashIndexDict, stitchIndexList, copyOutResolveSuccIndexListDict, cceCodeInfoList,
                                fillContent);
}

void DevAscendFunction::InitOperationNoPredNoSuccIndices(
    uintdevptr_t& initOffset, const OrderedSet<Operation*>& callList,
    const std::unordered_map<Operation*, uint64_t>& callOpPredDict,
    const std::unordered_map<Operation*, OrderedSet<Operation*>>& callOpSuccDict, const std::vector<int>& noPredOpList,
    const std::vector<int>& noSuccOpList, bool fillContent)
{
    noPredOpList_.HostInitDataSizeOffset(initOffset, noPredOpList.size());
    noSuccOpList_.HostInitDataSizeOffset(initOffset, noSuccOpList.size());

    ONFILLCONTENT
    {
        DevMemcpyS(&At(noPredOpList_, 0), noPredOpList_.ByteSize(), noPredOpList.data(),
                   noPredOpList.size() * sizeof(int));
        DevMemcpyS(&At(noSuccOpList_, 0), noSuccOpList_.ByteSize(), noSuccOpList.data(),
                   noSuccOpList.size() * sizeof(int));

        ASSERT(DevCommonErr::PARAM_CHECK_FAILED, noPredOpList_.size() == noPredOpList.size())
            << "Size mismatch: noPredOpList size: " << noPredOpList_.size()
            << " != noPredOpList size: " << noPredOpList.size();
        for (size_t i = 0; i < noPredOpList.size(); i++) {
            int opIdx = At(noPredOpList_, i);
            auto* op = callList[opIdx];
            ASSERT(DevCommonErr::PARAM_CHECK_FAILED, !callOpPredDict.count(op) || callOpPredDict.at(op) == 0)
                << "callOpPredDict for op: " << op << " is not zero: " << callOpPredDict.at(op);
        }
        ASSERT(DevCommonErr::PARAM_CHECK_FAILED, noSuccOpList_.size() == noSuccOpList.size())
            << "Size mismatch: noSuccOpList size: " << noSuccOpList_.size()
            << " != noSuccOpList size: " << noSuccOpList.size();
        for (size_t i = 0; i < noSuccOpList.size(); i++) {
            int opIdx = At(noSuccOpList_, i);
            auto* op = callList[opIdx];
            ASSERT(DevCommonErr::PARAM_CHECK_FAILED, !callOpSuccDict.count(op) || callOpSuccDict.at(op).empty())
                << "callOpSuccDict for op: " << op << " is not empty";
        }
    }
}

void DevAscendFunction::InitOperationBufferLayouts(
    uintdevptr_t& initOffset, const OrderedSet<Operation*>& callList,
    const std::unordered_map<Operation*, OrderedSet<Operation*>>& callOpSuccDict,
    const std::unordered_map<Operation*, std::vector<int>>& copyOutResolveSuccIndexListDict)
{
    operationList_.HostInitDataSizeOffset(initOffset, callList.size());

    int operanSize = 0;
    int staticAttributeSize = 0;
    int sucSize = 0;
    int copyOutResolveSuccIdxSize = 0;
    for (size_t i = 0; i < callList.size(); i++) {
        Operation* op = callList[i];
        auto callop = std::static_pointer_cast<CallOpAttribute>(callList[i]->GetOpAttribute());

        operanSize += op->GetIOperands().size() + op->GetOOperands().size();
        staticAttributeSize += callop->GetLinearArgList().size();
        sucSize += callOpSuccDict.find(op)->second.size();
        copyOutResolveSuccIdxSize += copyOutResolveSuccIndexListDict.find(op)->second.size();
    }
    operationOperandInfoList_.HostInitDataSizeOffset(initOffset, operanSize);
    operationAttrList_.HostInitDataSizeOffset(initOffset, staticAttributeSize);
    opAttrOffsetList_.HostInitDataSizeOffset(initOffset, callList.size());
    opCalleeList_.HostInitDataSizeOffset(initOffset, callList.size());
    operationSuccList_.HostInitDataSizeOffset(initOffset, sucSize);
    drcoEncodedSuccList_.HostInitDataSizeOffset(initOffset, sucSize);
    operationSuccInfoList_.HostInitDataSizeOffset(initOffset, callList.size());
    operationCopyOutResolveSuccIndexList_.HostInitDataSizeOffset(initOffset, copyOutResolveSuccIdxSize);
}

void DevAscendFunction::FillOperationEncodedContent(
    const SymbolicExpressionTable* expressionTable, const OrderedSet<Operation*>& callList,
    const OrderedSet<std::shared_ptr<LogicalTensor>>& tlist, const OrderedSet<std::shared_ptr<RawTensor>>& rawList,
    const std::unordered_map<Operation*, uint64_t>& callOpPredDict,
    const std::unordered_map<Operation*, OrderedSet<Operation*>>& callOpSuccDict,
    const std::unordered_map<uint64_t, int>& calleeHashIndexDict, const std::vector<int32_t>& stitchIndexList,
    const std::unordered_map<Operation*, std::vector<int>>& copyOutResolveSuccIndexListDict,
    const std::vector<CceCodeInfo>& cceCodeInfoList, bool fillContent)
{
    ONFILLCONTENT
    {
        auto* dupData = reinterpret_cast<DevAscendFunctionDuppedData*>(&At(duppedData_, 0));
        PopulateOperationEncodedContent(expressionTable, callList, tlist, rawList, callOpSuccDict, calleeHashIndexDict,
                                        stitchIndexList, copyOutResolveSuccIndexListDict, cceCodeInfoList, dupData);
        VerifyOperationEncodedContent(callList, callOpPredDict, dupData);
        dupData->GetSource() = this;
        for (size_t index = 0; index < callList.size(); index++) {
            uint8_t* ptr = reinterpret_cast<uint8_t*>(&dupData->GetOperationStitch(index));
            uint8_t* stitchBegin = &dupData->data_[dupData->operationList_.stitchBase];
            uint8_t* dataEndAlloc = &dupData->data_[duppedDataAllocSize_ - sizeof(DevAscendFunctionDuppedData)];
            ASSERT(ProgEncodeErr::RANGE_VERIFY_FAILED, stitchBegin <= ptr && ptr < dataEndAlloc)
                << "Address out of range: ptr " << ptr << " not in [" << stitchBegin << ", " << dataEndAlloc << ")";
        }
        dupData->GetSource() = nullptr;
    }
}

void DevAscendFunction::PopulateOperationEncodedContent(
    const SymbolicExpressionTable* expressionTable, const OrderedSet<Operation*>& callList,
    const OrderedSet<std::shared_ptr<LogicalTensor>>& tlist, const OrderedSet<std::shared_ptr<RawTensor>>& rawList,
    const std::unordered_map<Operation*, OrderedSet<Operation*>>& callOpSuccDict,
    const std::unordered_map<uint64_t, int>& calleeHashIndexDict, const std::vector<int32_t>& stitchIndexList,
    const std::unordered_map<Operation*, std::vector<int>>& copyOutResolveSuccIndexListDict,
    const std::vector<CceCodeInfo>& cceCodeInfoList, DevAscendFunctionDuppedData* dupData)
{
    int operanSize = 0;
    int staticAttributeSize = 0;
    int sucSize = 0;
    int copyOutResolveSuccIdxSize = 0;
    std::vector<uint32_t> drcoSuccScratch;
    uint64_t drcoPairCnt = 0;
    uint64_t drcoSuccCnt = 0;
    for (size_t index = 0; index < callList.size(); index++) {
        PopulateOneEncodedOpOperandsAndAttrs(index, operanSize, staticAttributeSize, expressionTable, callList, tlist,
                                             rawList, calleeHashIndexDict, stitchIndexList);
        PopulateOneEncodedOpGraphEdges(index, sucSize, copyOutResolveSuccIdxSize, callList, callOpSuccDict,
                                       copyOutResolveSuccIndexListDict, stitchIndexList, calleeHashIndexDict,
                                       cceCodeInfoList, dupData, drcoSuccScratch, drcoPairCnt, drcoSuccCnt);
    }
    if (drcoSuccCnt > 0) {
        MACHINE_LOGI("DrcoSuccPair: staticEdges=%llu, pairedEdges=%llu, atomicSaved=%llu (%.1f%%)",
                     static_cast<unsigned long long>(drcoSuccCnt), static_cast<unsigned long long>(drcoPairCnt * 2),
                     static_cast<unsigned long long>(drcoPairCnt), drcoPairCnt * 200.0 / drcoSuccCnt);
    }
}

static int64_t MaybeRawTensorIndex(int64_t val, const OrderedSet<std::shared_ptr<RawTensor>>& rawList)
{
    if ((val == -1) || !(val & RAW_TENSOR_INDEX_BIT_MASK)) {
        // concrete may -1, ignore this bad case
        return val;
    }

    auto magic = val & (~RAW_TENSOR_INDEX_BIT_MASK);
    for (auto rawTensor : rawList) {
        if (rawTensor->GetRawMagic() == magic) {
            return rawList.GetIndex(rawTensor);
        }
    }

    return val;
}

void DevAscendFunction::PopulateOneEncodedOpOperandsAndAttrs(
    size_t index, int& operanSize, int& staticAttributeSize, const SymbolicExpressionTable* expressionTable,
    const OrderedSet<Operation*>& callList, const OrderedSet<std::shared_ptr<LogicalTensor>>& tlist,
    const OrderedSet<std::shared_ptr<RawTensor>>& rawList, const std::unordered_map<uint64_t, int>& calleeHashIndexDict,
    const std::vector<int32_t>& stitchIndexList)
{
    Operation* op = callList[index];
    auto callop = std::static_pointer_cast<CallOpAttribute>(callList[index]->GetOpAttribute());
    DevAscendOperation& staticField = At(operationList_, index);

    staticField.stitchIndex = stitchIndexList[index];
    staticField.debugOpmagic = op->GetOpMagic();
    staticField.ioperandList.AssignRangeOffsetSize(operationOperandInfoList_, operanSize, op->GetIOperands().size());
    std::map<int, int> rawTensorIndex;
    for (size_t k = 0; k < op->GetIOperands().size(); k++) {
        auto coaIndex = op->GetIOpAttrOffset(k);
        const std::shared_ptr<LogicalTensor>& tensor = op->GetIOperands()[k];
        rawTensorIndex[coaIndex] = rawList.find(tensor->tensor)->second;
        At(staticField.ioperandList, k) = DevAscendOperationOperandInfo(
            tlist.GetIndex(tensor), coaIndex + COA_INDEX_DIM_BASE, tensor->GetShape().size());
    }
    operanSize += op->GetIOperands().size();

    staticField.ooperandList.AssignRangeOffsetSize(operationOperandInfoList_, operanSize, op->GetOOperands().size());
    for (size_t k = 0; k < op->GetOOperands().size(); k++) {
        auto coaIndex = op->GetOOpAttrOffset(k);
        const std::shared_ptr<LogicalTensor>& tensor = op->GetOOperands()[k];
        rawTensorIndex[coaIndex] = rawList.find(tensor->tensor)->second;
        At(staticField.ooperandList, k) = DevAscendOperationOperandInfo(
            tlist.GetIndex(tensor), coaIndex + COA_INDEX_DIM_BASE, tensor->GetShape().size());
    }
    operanSize += op->GetOOperands().size();
    MACHINE_LOGD("Producer %zu oOperand list size is %zu.", index, op->GetOOperands().size());

    auto callArgs = callop->GetLinearArgList();
    int opStaticAttrSize = callArgs.size();
    staticField.attrList.AssignRangeOffsetSize(operationAttrList_, staticAttributeSize, opStaticAttrSize);
    At(staticField.attrList, 0) = GetCceIndex(calleeHashIndexDict, callop);
    for (size_t k = CALLOP_ARG_ATTR_BASE_INDEX; k < (size_t)opStaticAttrSize; k++) {
        int fillValue = 0;
        if (callArgs[k].IsImmediate()) {
            fillValue = MaybeRawTensorIndex(callArgs[k].Concrete(), rawList);
        } else {
            fillValue = expressionTable->LookupPrimaryExpressionIndex(callArgs[k]);
        }
        At(staticField.attrList, k) = SymInt(!callArgs[k].IsImmediate(), fillValue);
    }

    At(opAttrOffsetList_, index) = staticAttributeSize;
    At(opCalleeList_, index) = calleeHashIndexDict.at(callop->GetCalleeHash().GetHash());
    staticAttributeSize += opStaticAttrSize;
}

void DevAscendFunction::PopulateOneEncodedOpGraphEdges(
    size_t index, int& sucSize, int& copyOutResolveSuccIdxSize, const OrderedSet<Operation*>& callList,
    const std::unordered_map<Operation*, OrderedSet<Operation*>>& callOpSuccDict,
    const std::unordered_map<Operation*, std::vector<int>>& copyOutResolveSuccIndexListDict,
    const std::vector<int32_t>& stitchIndexList, const std::unordered_map<uint64_t, int>& calleeHashIndexDict,
    const std::vector<CceCodeInfo>& cceCodeInfoList, DevAscendFunctionDuppedData* dupData,
    std::vector<uint32_t>& drcoSuccScratch, uint64_t& drcoPairCnt, uint64_t& drcoSuccCnt)
{
    Operation* op = callList[index];
    DevAscendOperation& staticField = At(operationList_, index);

    int opSuccSize = callOpSuccDict.find(op)->second.size();

    npu::tile_fwk::DevAscendFunctionOperationSuccInfo& succInfo = At(operationSuccInfoList_, index);
    succInfo.staticIndex = static_cast<uint16_t>(sucSize);
    succInfo.staticSize = static_cast<uint16_t>(opSuccSize);
    succInfo.stitchIndex = stitchIndexList[index];

    staticField.depGraphSuccList.AssignRangeOffsetSize(operationSuccList_, sucSize, opSuccSize);
    // The DRCO succ segment is sorted by succ opIdx (depGraphSuccList keeps dict order: it feeds the
    // AICPU/DRCU resolve and the dyn_topo dump). Sorting groups (2k, 2k+1) successors adjacently so
    // the device resolve can decrement both packed predCount halves with one u64 atomicAdd; the
    // first entry of each pair carries DRCO_SUCC_PAIR_BIT (aikernel_data.h: entry bits 16-28 are free).
    drcoSuccScratch.clear();
    drcoSuccScratch.reserve(static_cast<size_t>(opSuccSize));
    for (int k = 0; k < opSuccSize; k++) {
        uint32_t succ = callList.GetIndex(callOpSuccDict.find(op)->second[k]);
        At(staticField.depGraphSuccList, k) = succ;
        // Parallel DRCO-only entry: succOpIdx | (succ CoreType << 29).
        auto succCallop = std::static_pointer_cast<CallOpAttribute>(callList[succ]->GetOpAttribute());
        uint32_t succCoreType = cceCodeInfoList[calleeHashIndexDict.at(succCallop->GetCalleeHash().GetHash())].coreType;
        // Fail fast when the encoded table is actually consumed (same gate as the stitch-side
        // check): a successor not consumable by the DRCO resolve would OOB-index batchTaskIds
        // on device (AICPU-dispatch graphs may legally carry such successors, so no assert there).
        static const bool drcoConsume = IsAicoreResolveEnabled() &&
                                        config::GetRuntimeOption<int64_t>(CFG_RUN_MODE) != CFG_RUN_MODE_SIM;
        if (drcoConsume) {
            ASSERT(DevCommonErr::PARAM_INVALID, npu::tile_fwk::IsValidDrcoCoreType(succCoreType))
                << "DRCO successor coreType " << succCoreType << " (succ op " << succ << ") is not consumable";
        }
        drcoSuccScratch.push_back(npu::tile_fwk::EncodeDrcoCoreType(succ, succCoreType));
        At(operationList_, succ).depGraphPredCount++;
        dupData->GetOperationCurrPredCount(succ)++;
    }
    // Sort by opIdx only: coreType lives in the high bits and would break opIdx adjacency.
    std::sort(drcoSuccScratch.begin(), drcoSuccScratch.end(),
              [](uint32_t lhs, uint32_t rhs) { return (lhs & TASKID_TASK_MASK) < (rhs & TASKID_TASK_MASK); });
    for (size_t k = 0; k < drcoSuccScratch.size();) {
        uint32_t curOpIdx = drcoSuccScratch[k] & TASKID_TASK_MASK;
        if (k + 1 < drcoSuccScratch.size() && (curOpIdx & 1u) == 0u &&
            (drcoSuccScratch[k + 1] & TASKID_TASK_MASK) == curOpIdx + 1u) {
            At(drcoEncodedSuccList_, sucSize + static_cast<int>(k)) = static_cast<int32_t>(
                drcoSuccScratch[k] | npu::tile_fwk::DRCO_SUCC_PAIR_BIT);
            At(drcoEncodedSuccList_, sucSize + static_cast<int>(k) + 1) = static_cast<int32_t>(drcoSuccScratch[k + 1]);
            drcoPairCnt++;
            k += 2;
        } else {
            At(drcoEncodedSuccList_, sucSize + static_cast<int>(k)) = static_cast<int32_t>(drcoSuccScratch[k]);
            k += 1;
        }
    }
    drcoSuccCnt += static_cast<uint64_t>(opSuccSize);
    sucSize += opSuccSize;

    const std::vector<int>& copyOutResolveSuccIndexList = copyOutResolveSuccIndexListDict.find(op)->second;
    int opCopyOutResolveSuccIndexSize = copyOutResolveSuccIndexList.size();
    staticField.depGraphCopyOutResolveSuccIndexList.AssignRangeOffsetSize(
        operationCopyOutResolveSuccIndexList_, copyOutResolveSuccIdxSize, opCopyOutResolveSuccIndexSize);
    for (int k = 0; k < opCopyOutResolveSuccIndexSize; k++) {
        At(staticField.depGraphCopyOutResolveSuccIndexList, k) = copyOutResolveSuccIndexList[k];
    }
    copyOutResolveSuccIdxSize += copyOutResolveSuccIndexList.size();
}

void DevAscendFunction::VerifyOperationEncodedContent(const OrderedSet<Operation*>& callList,
                                                      const std::unordered_map<Operation*, uint64_t>& callOpPredDict,
                                                      DevAscendFunctionDuppedData* dupData)
{
    for (size_t idx = 0; idx < callList.size(); idx++) {
        Operation* op = callList[idx];
        ASSERT(DevCommonErr::PARAM_CHECK_FAILED, callOpPredDict.count(op))
            << "callOpPredDict does not contain op " << op;
        ASSERT(DevCommonErr::PARAM_CHECK_FAILED,
               At(operationList_, idx).depGraphPredCount == callOpPredDict.find(op)->second)
            << "depGraphPredCount mismatch: expected " << callOpPredDict.find(op)->second << ", got "
            << At(operationList_, idx).depGraphPredCount;
        if (static_cast<uint32_t>(dupData->GetOperationCurrPredCount(idx)) != callOpPredDict.find(op)->second) {
            MACHINE_LOGE(ProgEncodeErr::CALL_OP_COUNT_EXCEEDS_UINT16_MAX,
                         "OperationCurrPredCount: %d Callopsize is %u exceeds the maximum allowed value of 65535.",
                         dupData->GetOperationCurrPredCount(idx), dupData->GetOperationSize());
        }
        ASSERT(DevCommonErr::PARAM_CHECK_FAILED,
               static_cast<uint32_t>(dupData->GetOperationCurrPredCount(idx)) == callOpPredDict.find(op)->second)
            << "GetOperationCurrPredCount mismatch: expected " << dupData->GetOperationCurrPredCount(idx) << ", got "
            << callOpPredDict.find(op)->second << ", Callopsize is " << dupData->GetOperationSize()
            << " exceeds the maximum allowed value of 65535.";
    }
}

static void ValidateWrapGroupConsistency(int32_t wrapId, const std::vector<Operation*>& ops,
                                         const std::unordered_map<uint64_t, int>& calleeHashIndexDict,
                                         const std::vector<CceCodeInfo>& cceCodeInfoList)
{
    auto firstCallop = std::static_pointer_cast<CallOpAttribute>(ops[0]->GetOpAttribute());
    int cceIndex = calleeHashIndexDict.at(firstCallop->GetCalleeHash().GetHash());
    uint32_t mixResourceType = cceCodeInfoList[cceIndex].mixResourceType;
    for (size_t j = 1; j < ops.size(); j++) {
        auto callopJ = std::static_pointer_cast<CallOpAttribute>(ops[j]->GetOpAttribute());
        int leafIndexJ = calleeHashIndexDict.at(callopJ->GetCalleeHash().GetHash());
        ASSERT(DevCommonErr::PARAM_CHECK_FAILED, cceCodeInfoList[leafIndexJ].mixResourceType == mixResourceType)
            << "wrapId " << wrapId << ": callops have different mixResourceType";
    }

    int aicCount = 0;
    int aivCount = 0;
    for (auto* op : ops) {
        auto callop = std::static_pointer_cast<CallOpAttribute>(op->GetOpAttribute());
        int leafIndex = calleeHashIndexDict.at(callop->GetCalleeHash().GetHash());
        uint32_t coreType = cceCodeInfoList[leafIndex].coreType;
        if (coreType == static_cast<uint32_t>(CoreType::AIC)) {
            aicCount++;
        } else if (coreType == static_cast<uint32_t>(CoreType::AIV)) {
            aivCount++;
        }
    }

    if (mixResourceType == static_cast<uint32_t>(MixResourceType::ONE_CUBE_ONE_VECTOR)) {
        ASSERT(DevCommonErr::PARAM_CHECK_FAILED, aicCount == 1 && aivCount == 1)
            << "wrapId " << wrapId << ": MIX_1C1V requires 1 CUBE and 1 VECTOR, got " << aicCount << " and "
            << aivCount;
    } else if (mixResourceType == static_cast<uint32_t>(MixResourceType::ONE_CUBE_TWO_VECTOR)) {
        ASSERT(DevCommonErr::PARAM_CHECK_FAILED, aicCount == 1 && aivCount == 2)
            << "wrapId " << wrapId << ": MIX_1C2V requires 1 CUBE and 2 VECTOR, got " << aicCount << " and "
            << aivCount;
    }
}

static void ValidateWrapInfo(const std::unordered_map<int32_t, std::vector<Operation*>>& wrapIdToCallops,
                             const std::unordered_map<uint64_t, int>& calleeHashIndexDict,
                             const std::vector<CceCodeInfo>& cceCodeInfoList)
{
    for (const auto& [wrapId, ops] : wrapIdToCallops) {
        for (auto* op : ops) {
            auto callop = std::static_pointer_cast<CallOpAttribute>(op->GetOpAttribute());
            int leafIndex = calleeHashIndexDict.at(callop->GetCalleeHash().GetHash());
            uint32_t mixResourceType = cceCodeInfoList[leafIndex].mixResourceType;
            int wrapVecId = cceCodeInfoList[leafIndex].wrapVecId;
            MACHINE_LOGI("MixCallopInfo: opmagic=%d, wrapid=%d, coreType=%u, mixResourceType=%u, wrapVecId=%d",
                         op->opmagic, wrapId, cceCodeInfoList[leafIndex].coreType, mixResourceType, wrapVecId);
            ASSERT(DevCommonErr::PARAM_CHECK_FAILED, mixResourceType != 1 || wrapVecId != 1)
                << "Invalid mixResourceType and wrapVecId combination: opmagic = " << op->opmagic
                << ", wrapid = " << wrapId << ", mixResourceType = " << mixResourceType
                << ", wrapVecId = " << wrapVecId;
        }
        ValidateWrapGroupConsistency(wrapId, ops, calleeHashIndexDict, cceCodeInfoList);
    }
}

void DevAscendFunction::InitWrapInfo(uintdevptr_t& initOffset, const OrderedSet<Operation*>& callList, bool fillContent,
                                     const std::unordered_map<uint64_t, int>& calleeHashIndexDict,
                                     const std::vector<CceCodeInfo>& cceCodeInfoList)
{
    if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510) {
        return;
    }
    opWrapList_.HostInitDataSizeOffset(initOffset, callList.size());

    ONFILLCONTENT
    {
        std::unordered_set<uint32_t> wrapTaskNumSet;
        std::unordered_map<int32_t, std::vector<Operation*>> wrapIdToCallops;
        for (size_t i = 0; i < callList.size(); i++) {
            auto callop = std::static_pointer_cast<CallOpAttribute>(callList[i]->GetOpAttribute());
            At(opWrapList_, i) = callop->wrapId;
            if (callop->wrapId != INVALID_WRAPID) {
                wrapTaskNumSet.insert(callop->wrapId);
                wrapIdToCallops[callop->wrapId].push_back(callList[i]);
            }
        }
        wrapIdNum_ = wrapTaskNumSet.size();
        ValidateWrapInfo(wrapIdToCallops, calleeHashIndexDict, cceCodeInfoList);
    }
}

void DevAscendFunction::FillIncastUseList(
    DevLocalVector<DevAscendFunctionCallOperandUse>& fillUseList, uint64_t& useSize,
    const std::vector<std::shared_ptr<LogicalTensor>>& tensorList,
    const std::unordered_map<std::shared_ptr<LogicalTensor>, InoutOperationAttr>& attrDict, bool fillContent,
    const std::unordered_map<int, int>& opIdxToHubOpIdx)
{
    for (size_t index = 0; index < tensorList.size(); index++) {
        auto& attr = attrDict.at(tensorList[index]);
        auto& incast = At(incastList, index);
        ONFILLCONTENT
        {
            incast.consumerList.AssignRangeOffsetSize(fillUseList, useSize, attr.useList.size());
            for (size_t k = 0; k < attr.useList.size(); k++) {
                At(incast.consumerList, k) = attr.useList[k];
                auto& use = At(incast.consumerList, k);
                int opAttrBase = At(opAttrOffsetList_, use.operationIdx);
                use.offsetAttrIdx += opAttrBase;
                auto it = opIdxToHubOpIdx.find(attr.useList[k].operationIdx);
                use.stitchOpIdx = (it != opIdxToHubOpIdx.end()) ? it->second : use.operationIdx;
            }
        }
        useSize += attr.useList.size();
        ONFILLCONTENT
        {
            incast.stitchPolicyFullCoverConsumerList.AssignRangeOffsetSize(
                fillUseList, useSize, attr.stitchPolicyFullCoverConsumerList.size());
            for (size_t k = 0; k < attr.stitchPolicyFullCoverConsumerList.size(); k++) {
                At(incast.stitchPolicyFullCoverConsumerList, k) = attr.stitchPolicyFullCoverConsumerList[k];
                auto& use = At(incast.stitchPolicyFullCoverConsumerList, k);
                int opAttrBase = At(opAttrOffsetList_, use.operationIdx);
                use.offsetAttrIdx += opAttrBase;
                auto it = opIdxToHubOpIdx.find(attr.stitchPolicyFullCoverConsumerList[k].operationIdx);
                use.stitchOpIdx = (it != opIdxToHubOpIdx.end()) ? it->second : use.operationIdx;
            }
        }
        useSize += attr.stitchPolicyFullCoverConsumerList.size();
    }
}

void DevAscendFunction::FillOutcastUseList(
    DevLocalVector<DevAscendFunctionCallOperandUse>& fillUseList, uint64_t& useSize,
    const std::vector<std::shared_ptr<LogicalTensor>>& tensorList,
    const std::unordered_map<std::shared_ptr<LogicalTensor>, InoutOperationAttr>& attrDict, bool fillContent,
    const std::unordered_map<int, int>& opIdxToHubOpIdx)
{
    for (size_t index = 0; index < tensorList.size(); index++) {
        auto& attr = attrDict.at(tensorList[index]);
        auto& outcast = At(outcastList, index);
        ONFILLCONTENT
        {
            outcast.producerConsumerList.AssignRangeOffsetSize(fillUseList, useSize, attr.useList.size());
            for (size_t k = 0; k < attr.useList.size(); k++) {
                At(outcast.producerConsumerList, k) = attr.useList[k];
                auto& use = At(outcast.producerConsumerList, k);
                int opAttrBase = At(opAttrOffsetList_, use.operationIdx);
                use.offsetAttrIdx += opAttrBase;
                auto it = opIdxToHubOpIdx.find(attr.useList[k].operationIdx);
                use.stitchOpIdx = (it != opIdxToHubOpIdx.end()) ? it->second : use.operationIdx;
            }
        }
        useSize += attr.useList.size();
    }
}

void DevAscendFunction::InitStitchUpdateSlotPrefilterLists(
    uintdevptr_t& initOffset, const IncastOutcastSlot* slot, int totalSlot,
    const std::vector<std::shared_ptr<LogicalTensor>>& incastTensorList,
    const std::vector<std::shared_ptr<LogicalTensor>>& outcastTensorList, bool fillContent)
{
    uint64_t updateOutCnt = 0, stitchInCnt = 0, updateInCnt = 0, stitchOutCnt = 0;
    for (size_t index = 0; index < outcastTensorList.size(); index++) {
        updateOutCnt += slot->outcastSlot[index].size();
        stitchOutCnt += slot->outcastSlot[index].size();
    }
    for (size_t index = 0; index < incastTensorList.size(); index++) {
        stitchInCnt += slot->incastSlot[index].size();
        updateInCnt += slot->incastSlot[index].size();
    }

    updateOutCastSlotList.HostInitDataSizeOffset(initOffset, updateOutCnt);
    updateIncastSlotList.HostInitDataSizeOffset(initOffset, updateInCnt);
    stitchIncastSlotList.HostInitDataSizeOffset(initOffset, stitchInCnt);
    stitchOutCastSlotList.HostInitDataSizeOffset(initOffset, stitchOutCnt);

    ONFILLCONTENT
    {
        uint64_t uo = 0, ui = 0, si = 0, so = 0;
        for (size_t index = 0; index < outcastTensorList.size(); index++) {
            for (size_t j = 0; j < slot->outcastSlot[index].size(); j++) {
                int slotIdx = slot->outcastSlot[index][j];
                ASSERT(DevCommonErr::PARAM_CHECK_FAILED, slotIdx >= 0 && slotIdx < totalSlot)
                    << "invalid slotIdx " << slotIdx << " in outcastSlot, totalSlot=" << totalSlot;
                At(updateOutCastSlotList, uo++) = {slotIdx, static_cast<int>(index)};
                At(stitchOutCastSlotList, so++) = {slotIdx, static_cast<int>(index)};
            }
        }
        for (size_t index = 0; index < incastTensorList.size(); index++) {
            for (size_t j = 0; j < slot->incastSlot[index].size(); j++) {
                int slotIdx = slot->incastSlot[index][j];
                ASSERT(DevCommonErr::PARAM_CHECK_FAILED, slotIdx >= 0 && slotIdx < totalSlot)
                    << "invalid slotIdx " << slotIdx << " in incastSlot, totalSlot=" << totalSlot;
                At(updateIncastSlotList, ui++) = {slotIdx, static_cast<int>(index)};
                At(stitchIncastSlotList, si++) = {slotIdx, static_cast<int>(index)};
            }
        }
    }
}

void DevAscendFunction::InitIncastOutcast(
    uintdevptr_t& initOffset, const std::vector<std::shared_ptr<LogicalTensor>>& incastTensorList,
    const std::vector<std::shared_ptr<LogicalTensor>>& outcastTensorList,
    const OrderedSet<std::shared_ptr<LogicalTensor>>& tlist,
    const std::unordered_map<std::shared_ptr<LogicalTensor>, InoutOperationAttr>& incastOpAttrDict,
    const std::unordered_map<std::shared_ptr<LogicalTensor>, InoutOperationAttr>& outcastOpAttrDict,
    const EncodeDevAscendFunctionParam& param, const std::string& initRawName, bool fillContent,
    const std::unordered_map<int, int>& opIdxToHubOpIdx)
{
    {
        // Fill metadata
        incastList.HostInitDataSizeOffset(initOffset, incastTensorList.size());
        outcastList.HostInitDataSizeOffset(initOffset, outcastTensorList.size());
        for (size_t index = 0; index < incastTensorList.size(); index++) {
            auto& inAttr = incastOpAttrDict.at(incastTensorList[index]);
            auto& incast = At(incastList, index);
            ONFILLCONTENT
            {
                incast.tensorIndex = tlist.GetIndex(incastTensorList[index]);
                incast.dim = inAttr.dim;
                incast.cellMatchTableDesc = inAttr.cellMatchTableDesc;
            }
        }
        for (size_t index = 0; index < outcastTensorList.size(); index++) {
            auto& outAttr = outcastOpAttrDict.at(outcastTensorList[index]);
            auto& outcast = At(outcastList, index);
            ONFILLCONTENT
            {
                outcast.tensorIndex = tlist.GetIndex(outcastTensorList[index]);
                outcast.dim = outAttr.dim;
                outcast.cellMatchTableDesc = outAttr.cellMatchTableDesc;
                outcast.exprListIndex = outAttr.bindTensorExprIndex;
                outcast.desc = param.outcastDescList[index];
            }
        }
    }
    {
        const IncastOutcastSlot* slot = param.slot;
        [[maybe_unused]] const IncastOutcastLink* inoutLink = param.inoutLink;
        // Fill slot list
        slotList.HostInitDataSizeOffset(initOffset, 0);
        uint64_t slotSize = 0;
        for (size_t index = 0; index < incastTensorList.size(); index++) {
            auto& incast = At(incastList, index);
            ONFILLCONTENT
            {
                incast.fromSlotList.AssignRangeOffsetSize(slotList, slotSize, slot->incastSlot[index].size());
                for (size_t j = 0; j < slot->incastSlot[index].size(); j++) {
                    At(incast.fromSlotList, j) = slot->incastSlot[index][j];
                }
            }
            slotSize += slot->incastSlot[index].size();
        }
        for (size_t index = 0; index < outcastTensorList.size(); index++) {
            auto& outcast = At(outcastList, index);
            ONFILLCONTENT
            {
                outcast.toSlotList.AssignRangeOffsetSize(slotList, slotSize, slot->outcastSlot[index].size());
                for (size_t j = 0; j < slot->outcastSlot[index].size(); j++) {
                    At(outcast.toSlotList, j) = slot->outcastSlot[index][j];
                }
            }
            slotSize += slot->outcastSlot[index].size();
        }
        slotList.HostInitDataSizeOffset(initOffset, slotSize);

        redaccAssembleSlotList_.HostInitDataSizeOffset(initOffset, param.assembleSlotList.size());
        ONFILLCONTENT
        {
            for (size_t k = 0; k < param.assembleSlotList.size(); k++) {
                At(redaccAssembleSlotList_, k) = param.assembleSlotList[k];
            }
        }
    }
    {
        // Fill use list
        useList.HostInitDataSizeOffset(initOffset, 0);
        uint64_t useSize = 0;
        FillIncastUseList(useList, useSize, incastTensorList, incastOpAttrDict, fillContent, opIdxToHubOpIdx);
        FillOutcastUseList(useList, useSize, outcastTensorList, outcastOpAttrDict, fillContent, opIdxToHubOpIdx);
        useList.HostInitDataSizeOffset(initOffset, useSize);
    }
    {
        // Fill runtime full update table
        uint64_t cellMatchSizeOutcastTotal = 0;
        cellMatchRuntimeFullUpdateTableList.HostInitDataSizeOffset(initOffset, 0);
        for (size_t index = 0; index < outcastTensorList.size(); index++) {
            auto& outAttr = outcastOpAttrDict.at(outcastTensorList[index]);
            auto& outcast = At(outcastList, index);
            ONFILLCONTENT
            {
                outcast.cellMatchRuntimeFullUpdateTable.AssignRangeOffsetSize(
                    cellMatchRuntimeFullUpdateTableList, cellMatchSizeOutcastTotal, outAttr.cellMatchSize);
                for (int k = 0; k < outAttr.cellMatchSize; k++) {
                    At(outcast.cellMatchRuntimeFullUpdateTable, k) = (uint32_t)-1;
                };
            }
            cellMatchSizeOutcastTotal += outAttr.cellMatchSize;
        }
        cellMatchRuntimeFullUpdateTableList.HostInitDataSizeOffset(initOffset, cellMatchSizeOutcastTotal);
    }
    {
        // Fill stitchPolicyFullCoverProducerList
        uint64_t fullCoverTotal = 0;
        stitchPolicyFullCoverProducerList_.HostInitDataSizeOffset(initOffset, 0);
        for (size_t index = 0; index < outcastTensorList.size(); index++) {
            auto& outAttr = outcastOpAttrDict.at(outcastTensorList[index]);
            auto& outcast = At(outcastList, index);
            ONFILLCONTENT
            {
                outcast.stitchPolicyFullCoverProducerHubOpIdx = outAttr.stitchPolicyFullCoverProducerHubOpIdx;
                outcast.stitchPolicyFullCoverProducerList.AssignRangeOffsetSize(
                    stitchPolicyFullCoverProducerList_, fullCoverTotal,
                    outAttr.stitchPolicyFullCoverProducerList.size());
                for (size_t k = 0; k < outAttr.stitchPolicyFullCoverProducerList.size(); k++) {
                    At(outcast.stitchPolicyFullCoverProducerList, k) = outAttr.stitchPolicyFullCoverProducerList[k];
                    auto& use = At(outcast.stitchPolicyFullCoverProducerList, k);
                    int opAttrBase = At(opAttrOffsetList_, use.operationIdx);
                    use.offsetAttrIdx += opAttrBase;
                    auto it = opIdxToHubOpIdx.find(outAttr.stitchPolicyFullCoverProducerList[k].operationIdx);
                    use.stitchOpIdx = (it != opIdxToHubOpIdx.end()) ? it->second : use.operationIdx;
                };
            }
            fullCoverTotal += outAttr.stitchPolicyFullCoverProducerList.size();
        }
        stitchPolicyFullCoverProducerList_.HostInitDataSizeOffset(initOffset, fullCoverTotal);
    }
    {
        // Fill stitchPolicyFullCoverOpList_
        uint32_t fullCoverOpTotal = 0;
        stitchPolicyFullCoverOpList_.HostInitDataSizeOffset(initOffset, 0);
        for (size_t index = 0; index < incastTensorList.size(); index++) {
            auto& inAttr = incastOpAttrDict.at(incastTensorList[index]);
            auto& incast = At(incastList, index);
            std::vector<uint32_t> opIdxList;
            for (size_t k = 0; k < inAttr.useOpList.size(); k++) {
                uint32_t opIdx = inAttr.useOpList[k];
                auto it = opIdxToHubOpIdx.find(static_cast<int>(opIdx));
                if (it != opIdxToHubOpIdx.end()) {
                    opIdx = static_cast<uint32_t>(it->second);
                }
                if (opIdxList.empty() || std::find(opIdxList.begin(), opIdxList.end(), opIdx) == opIdxList.end()) {
                    opIdxList.push_back(opIdx);
                }
            }
            ONFILLCONTENT
            {
                incast.stitchPolicyFullCoverConsumerAllOpIdxList.AssignRangeOffsetSize(
                    stitchPolicyFullCoverOpList_, fullCoverOpTotal, opIdxList.size());
                for (size_t k = 0; k < opIdxList.size(); k++) {
                    At(incast.stitchPolicyFullCoverConsumerAllOpIdxList, k) = opIdxList[k];
                }
            }
            fullCoverOpTotal += opIdxList.size();
        }
        for (size_t index = 0; index < outcastTensorList.size(); index++) {
            auto& outAttr = outcastOpAttrDict.at(outcastTensorList[index]);
            auto& outcast = At(outcastList, index);
            ONFILLCONTENT
            {
                outcast.stitchPolicyFullCoverAllOpIdxList.AssignRangeOffsetSize(
                    stitchPolicyFullCoverOpList_, fullCoverOpTotal, outAttr.useOpList.size());
                for (size_t k = 0; k < outAttr.useOpList.size(); k++) {
                    At(outcast.stitchPolicyFullCoverAllOpIdxList, k) = outAttr.useOpList[k];
                }
            }
            fullCoverOpTotal += outAttr.useOpList.size();
        }
        stitchPolicyFullCoverOpList_.HostInitDataSizeOffset(initOffset, fullCoverOpTotal);
    }

    InitStitchUpdateSlotPrefilterLists(initOffset, param.slot, param.inoutLink->totalSlot, incastTensorList,
                                       outcastTensorList, fillContent);

    rawName_.HostInitDataSizeOffset(initOffset, (initRawName.size() / 8 + 1) * 8); // 8 byte align
    ONFILLCONTENT
    {
        DevMemcpyS(&At(rawName_, 0), rawName_.size(), initRawName.c_str(), initRawName.size());
        memset_s(&At(rawName_, initRawName.size()), rawName_.size() - initRawName.size(), 0,
                 rawName_.size() - initRawName.size());
    };
}

void AddTokenDependEdgesToColorGraph(std::vector<Operation*>& callopList,
                                     std::unordered_map<Operation*, int>& callopIndexDict,
                                     std::unordered_map<int, std::vector<int>>& colorOutGraph)
{
    for (auto* op : callopList) {
        if (op->GetOpcode() != Opcode::OP_CALL) {
            continue;
        }
        int producerIdx = callopIndexDict[op];
        for (auto* consumer : op->ConsumerOpsByToken()) {
            auto it = callopIndexDict.find(consumer);
            if (it == callopIndexDict.end() || consumer == op) {
                continue;
            }
            colorOutGraph[producerIdx].push_back(it->second);
        }
    }
}

struct EncodeDevAscendFunctionInfo {
    Function* devRoot{nullptr};
    Function* devTile{nullptr};

    const std::unordered_map<uint64_t, int>& calleeHashIndexDict;
    const std::vector<CceCodeInfo>& cceCodeInfoList;
    const SymbolicExpressionTable* expressionTable{nullptr};

    std::string rawName;

    OrderedSet<Operation*> callList;

    uint64_t totalZeroPred{0};
    uint64_t totalZeroPredAIV{0};
    uint64_t totalZeroPredAIC{0};
    uint64_t totalZeroPredHub{0};
    uint64_t totalZeroPredAicpu{0};
    uint32_t hubOpCount{0};

    std::unordered_map<Operation*, uint64_t> callOpPredDict;
    std::unordered_map<Operation*, OrderedSet<Operation*>> callOpSuccDict;
    std::unordered_map<int, std::vector<int>> colorOutGraph;
    std::vector<std::shared_ptr<Operation>> dummyOpList;

    std::vector<int> noSuccOpList;
    std::vector<int> noPredOpList;

    uint32_t stitchCount{0};
    std::vector<int32_t> stitchIndexList;

    OrderedSet<std::shared_ptr<RawTensor>> incastRawTensorList;
    OrderedSet<std::shared_ptr<RawTensor>> outcastRawTensorList;
    OrderedSet<std::shared_ptr<RawTensor>> rawTensorList;
    std::vector<EncodeRawTensorAttr> rawAttrs;

    std::unordered_map<int, std::shared_ptr<RawTensor>> rawMagicToRawTensor;
    OrderedSet<std::shared_ptr<LogicalTensor>> tensorList;

    std::vector<std::shared_ptr<LogicalTensor>> incastList;
    std::vector<std::shared_ptr<LogicalTensor>> outcastList;

    std::unordered_set<std::shared_ptr<LogicalTensor>> incastSet;
    std::unordered_set<std::shared_ptr<LogicalTensor>> outcastSet;

    std::unordered_map<std::shared_ptr<LogicalTensor>, InoutOperationAttr> incastOpAttrDict;
    std::unordered_map<std::shared_ptr<LogicalTensor>, InoutOperationAttr> outcastOpAttrDict;

    std::unordered_map<Operation*, std::vector<int>> copyOutResolveSuccIndexListDict;

    DyndevFunctionAttribute::ValueDependDesc valueDependDesc;

    std::unordered_map<int, int> opIdxToHubOpIdx_;

    static DevShape InitShape(const std::vector<int64_t>& shape)
    {
        DevShape initShape;
        initShape.dimSize = shape.size();
        for (size_t index = 0; index < DEV_SHAPE_DIM_MAX; index++) {
            if (index < shape.size()) {
                initShape.dim[index] = shape[index];
            } else {
                initShape.dim[index] = 0;
            }
        }
        return initShape;
    }
    static DevAscendStride InitStride(const std::vector<int64_t>& stride)
    {
        DevAscendStride initStride;
        initStride.dimSize = stride.size();
        for (size_t index = 0; index < DEV_SHAPE_DIM_MAX; index++) {
            if (index < stride.size()) {
                initStride.dimStride[index] = stride[index];
            } else {
                initStride.dimStride[index] = 0;
            }
        }
        return initStride;
    }
    static DevCellMatchTableDesc InitCellMatchTableDesc(const std::vector<int64_t>& shape,
                                                        const std::vector<int64_t>& stride)
    {
        DevCellMatchTableDesc desc = {
            InitShape(shape),
            InitStride(stride),
        };
        return desc;
    }

    std::vector<int> ShapeToVector(const DevShape& shape)
    {
        std::vector<int> data(&shape.dim[0], &shape.dim[shape.dimSize]);
        return data;
    }
    std::vector<int> StrideToVector(const DevAscendStride& stride)
    {
        std::vector<int> data(&stride.dimStride[0], &stride.dimStride[stride.dimSize]);
        return data;
    }

    void UpdateCellMatchShape(DevCellMatchTableDesc& cellMatchTableDesc, const std::vector<int64_t>& shape)
    {
        auto& cellMatchShape = cellMatchTableDesc.cellShape;
        for (size_t index = 0; index < shape.size(); ++index) {
            auto dimValue = shape[index];
            // Dynamic axis (-1) should be materialized to real consumed shape first.
            if (cellMatchShape.dim[index] == -1 || cellMatchShape.dim[index] > dimValue) {
                cellMatchShape.dim[index] = dimValue;
                if (cellMatchShape.dim[index] == 0) {
                    MACHINE_LOGE(ProgEncodeErr::CELL_MATCH_DIM_ZERO, "cellMatchShape.dim[%zu] is zero after assignment",
                                 index);
                }
                DEV_ASSERT(ProgEncodeErr::CELL_MATCH_DIM_ZERO, cellMatchShape.dim[index]);
            }
        }
    }

    void UpdateCellMatchStrideAndSize(int& cellMatchSize, DevCellMatchTableDesc& cellMatchTableDesc,
                                      const std::shared_ptr<LogicalTensor>& tensor, int dim)
    {
        auto& cellMatchShape = cellMatchTableDesc.cellShape;
        auto& cellMatchStride = cellMatchTableDesc.stride;

        cellMatchSize = 1;
        cellMatchStride.dimSize = dim;
        bool hasDynamicDim = false;
        for (int r = (dim - 1); r >= 0; --r) {
            int tile = 0;
            if (tensor->shape[r] < 0 || cellMatchShape.dim[r] <= 0) {
                hasDynamicDim = true;
            } else {
                tile = tensor->shape[r] / cellMatchShape.dim[r];
                if (tensor->shape[r] % cellMatchShape.dim[r] != 0) {
                    // should not happen
                    tile += 1;
                }
            }
            cellMatchSize *= tile;
            cellMatchStride[r] = cellMatchSize;
        }
        MACHINE_LOGD("Outcast %d rawtensor magic %d shape %s | cellMatchSize %d cellMatchShape %s cellMatchStride %s\n",
                     tensor->magic, tensor->GetRawMagic(), IntVecToStr(tensor->shape).c_str(), cellMatchSize,
                     IntVecToStr(ShapeToVector(cellMatchShape)).c_str(),
                     IntVecToStr(StrideToVector(cellMatchStride)).c_str());
        if (hasDynamicDim) {
            return;
        }
        if (cellMatchStride[0] > MAX_CELLMATCHSSTRIDE) {
            MACHINE_LOGE(
                ProgEncodeErr::ASSEMBLE_STITCH_MEMORY_EXCESS,
                "Assemble out-cast %d raw %d stitch results in excessive memory consumption "
                "Please appropriately configure the view shape and tile shape, and ensure aligned with the input "
                "shape, cellMatchStride[0]=%lu, MAX_CELLMATCHSSTRIDE=%lu",
                tensor->magic, tensor->GetRawMagic(), static_cast<unsigned long>(cellMatchStride[0]),
                static_cast<unsigned long>(MAX_CELLMATCHSSTRIDE));
        }
        ASSERT(DevCommonErr::PARAM_CHECK_FAILED, cellMatchStride[0] < MAX_CELLMATCHSSTRIDE)
            << " Assemble outcast " << tensor->magic << " raw " << tensor->GetRawMagic()
            << " stitch results in excessive memory consumption,"
            << " Please appropriately configure the view shape and tile shape, and ensure aligned with the input "
               "shape.";
    }

    void RecordRawTensor(const std::shared_ptr<LogicalTensor>& tensor)
    {
        if (rawTensorList.Insert(tensor->GetRawTensor())) {
            EncodeRawTensorAttr& attr = rawAttrs.emplace_back();
            attr.storage = tensor->storage_;
            attr.storageOffset = tensor->storageOffset_;
            rawMagicToRawTensor[tensor->GetRawTensor()->rawmagic] = tensor->GetRawTensor();
        }
    }

    Opcode GetOutcastProducerOpcode(const Operation& op, size_t oOperandIdx) const
    {
        if (op.GetOpcode() != Opcode::OP_CALL) {
            return Opcode::OP_UNKNOWN;
        }
        FunctionCache& cache = Program::GetInstance().GetFunctionCache();
        Function* devLeafFunc = cache.GetCacheFunction(op.GetCalleeHash());
        if (devLeafFunc == nullptr) {
            return Opcode::OP_UNKNOWN;
        }
        const auto& leafOutcastList = devLeafFunc->GetOutcast();
        if (oOperandIdx >= leafOutcastList.size() || leafOutcastList[oOperandIdx] == nullptr) {
            return Opcode::OP_UNKNOWN;
        }
        const auto& leafOutcast = leafOutcastList[oOperandIdx];
        const auto& producers = leafOutcast->GetProducers();
        if (producers.size() != 1) {
            for (auto* producer : producers) {
                if (producer != nullptr && producer->GetOpcode() == Opcode::OP_RESHAPE) {
                    return Opcode::OP_RESHAPE;
                }
            }
            return Opcode::OP_UNKNOWN;
        }
        auto* producerOp = *producers.begin();
        if (producerOp == nullptr) {
            return Opcode::OP_UNKNOWN;
        }
        return producerOp != nullptr ? producerOp->GetOpcode() : Opcode::OP_UNKNOWN;
    }

    void RebuildOutcastUseListByLastReadPos(std::vector<DevAscendFunctionCallOperandUse>& useList,
                                            const std::unordered_map<int, size_t>& lastReadPos)
    {
        std::vector<DevAscendFunctionCallOperandUse> next;
        next.reserve(useList.size());
        for (size_t i = 0; i < useList.size(); ++i) {
            if (useList[i].opType == CellMatchOpType::READ) {
                auto it = lastReadPos.find(useList[i].operationIdx);
                if (it == lastReadPos.end() || it->second != i) {
                    continue;
                }
            }
            next.push_back(useList[i]);
        }
        useList.swap(next);
    }

    void EncodeAnalysisOutCastConsumerByProducer(const std::shared_ptr<LogicalTensor>& o,
                                                 std::vector<DevAscendFunctionCallOperandUse>& useList,
                                                 InoutOperationAttr& outcastOpAttr, std::set<uint32_t>& outcastUseOpSet,
                                                 int producerIdx, std::unordered_map<int, size_t>& lastReadPos,
                                                 bool& hasDupRead)
    {
        MACHINE_LOGD("Enter EncodeAnalysisOutCastConsumerByProducer: outcast magic=%d rawmagic=%d producerIdx=%d.",
                     o->magic, o->GetRawMagic(), producerIdx);
        std::vector<int> hubWorklist = {producerIdx};
        for (size_t wi = 0; wi < hubWorklist.size(); ++wi) {
            auto* nodeOp = callList[static_cast<size_t>(hubWorklist[wi])];
            if (!callOpSuccDict.count(nodeOp)) {
                MACHINE_LOGD("worklist[%zu]=%d has no succ ops, skip.", wi, hubWorklist[wi]);
                continue;
            }
            auto& succOps = callOpSuccDict.at(nodeOp);
            MACHINE_LOGD("worklist[%zu]=%d has %zu successor ops.", wi, hubWorklist[wi], succOps.size());
            for (auto* succOp : succOps) {
                if (!callList.HasData(succOp)) {
                    MACHINE_LOGD("succOp not in callList, skip.");
                    continue;
                }
                int succOpIdx = callList.GetIndex(succOp);
                if (IsHubType(GetCoreType(succOp))) {
                    hubWorklist.push_back(succOpIdx);
                    continue;
                }
                outcastUseOpSet.insert(succOpIdx);
                auto callAttrSucc = dynamic_cast<CallOpAttribute*>(succOp->GetOpAttribute().get());
                for (size_t k = 0; k < succOp->GetIOperands().size(); ++k) {
                    auto& iOperand = succOp->GetIOperands()[k];
                    if (o->tensor->rawmagic != iOperand->tensor->rawmagic) {
                        continue;
                    }
                    auto coaIndex = succOp->GetIOpAttrOffset(k) + COA_INDEX_DIM_BASE;
                    if (lastReadPos.count(succOpIdx) > 0) {
                        hasDupRead = true;
                    }
                    useList.emplace_back(succOpIdx, coaIndex, CellMatchOpType::READ);
                    lastReadPos[succOpIdx] = useList.size() - 1;
                    MACHINE_LOGD("MATCH! consumerList.emplace_back succOpIdx=%d coaIndex=%d dim=%d iOperandIdx=%zu.",
                                 succOpIdx, coaIndex, outcastOpAttr.dim, k);

                    auto shape = callAttrSucc->GetLinearImmediateArgList(coaIndex + outcastOpAttr.dim,
                                                                         coaIndex + outcastOpAttr.dim * 0x2, false);
                    UpdateCellMatchShape(outcastOpAttr.cellMatchTableDesc, shape);
                    MACHINE_LOGD("Minimal shape for outcast %d rawtensor magic %d consumer op %d %d is %s.\n", o->magic,
                                 o->GetRawMagic(), succOpIdx, succOp->GetOpMagic(),
                                 IntVecToStr(ShapeToVector(outcastOpAttr.cellMatchTableDesc.cellShape)).c_str());
                }
            }
        }
    }

    void EncodeAnalysisOutCastProducerUse(const std::shared_ptr<LogicalTensor>& o, Operation& op,
                                          CallOpAttribute* callAttr, size_t oOperandIdx, size_t opIdx, int dimSize,
                                          InoutOperationAttr& outcastOpAttr,
                                          DevAscendFunctionCallOperandUse& stitchPolicyFullCoverProducer,
                                          std::vector<DevAscendFunctionCallOperandUse>& useList)
    {
        auto& oOperand = op.GetOOperands()[oOperandIdx];
        auto coaIndex = op.GetOOpAttrOffset(oOperandIdx) + COA_INDEX_DIM_BASE;
        CellMatchOpType producerOpType = CellMatchOpType::NORMAL_WRITE;
        if (op.GetOOpAttr(oOperandIdx).isAtomic()) {
            if (IsComputeDeterminismEnabled()) {
                o->SetAttr("NORMAL", true);
            } else {
                producerOpType = CellMatchOpType::ATOMIC_WRITE;
            }
        }
        Opcode opCode = GetOutcastProducerOpcode(op, oOperandIdx);
        std::vector<int64_t> offset = callAttr->GetLinearImmediateArgList(coaIndex, coaIndex + dimSize, true);
        std::vector<int64_t> shape = callAttr->GetLinearImmediateArgList(coaIndex + dimSize, coaIndex + dimSize * 0x2,
                                                                         false);
        if (offset == std::vector<int64_t>(dimSize, 0) && shape == oOperand->GetShape()) {
            stitchPolicyFullCoverProducer = DevAscendFunctionCallOperandUse(opIdx, coaIndex, producerOpType, opCode);
        } else {
            useList.emplace_back(opIdx, coaIndex, producerOpType, opCode);
        }
        MACHINE_LOGD("Outcast oOperandIdx for outcast %d rawtensor magic %d is %zu.", o->magic, o->GetRawMagic(),
                     oOperandIdx);
        auto expr = callAttr->GetOutcastSymbolicExpr(oOperandIdx);
        outcastOpAttr.bindTensorExprIndex = -1;
        if ((expr.has_value()) && (expressionTable != nullptr)) {
            outcastOpAttr.bindTensorExprIndex = expressionTable->LookupPrimaryExpressionIndex(expr.value());
        }
    }

    void EncodeAnalysisOpUseOutCasts(const std::shared_ptr<LogicalTensor>& o, std::set<uint32_t>& allOutcastUseOpSet,
                                     InoutOperationAttr& outcastOpAttr)
    {
        std::set<uint32_t> outcastUseOpSet;
        std::unordered_map<int, size_t> useListLastReadPos;
        std::unordered_map<int, size_t> fullCoverLastReadPos;
        bool useListHasDupRead = false;
        bool fullCoverHasDupRead = false;
        int dimSize = outcastOpAttr.dim;
        for (size_t i = 0; i < callList.size(); i++) {
            auto& op = *callList[i];
            auto callAttr = dynamic_cast<CallOpAttribute*>(op.GetOpAttribute().get());
            std::vector<DevAscendFunctionCallOperandUse> useList;
            DevAscendFunctionCallOperandUse stitchPolicyFullCoverProducer;
            for (size_t j = 0; j < op.GetOOperands().size(); ++j) {
                auto& oOperand = op.GetOOperands()[j];
                if (o->tensor->rawmagic != oOperand->tensor->rawmagic) {
                    continue;
                }
                EncodeAnalysisOutCastProducerUse(o, op, callAttr, j, i, dimSize, outcastOpAttr,
                                                 stitchPolicyFullCoverProducer, useList);
                outcastUseOpSet.insert(i);
            }

            if (stitchPolicyFullCoverProducer.operationIdx != -1) {
                outcastOpAttr.stitchPolicyFullCoverProducerList.push_back(stitchPolicyFullCoverProducer);
                EncodeAnalysisOutCastConsumerByProducer(o, outcastOpAttr.stitchPolicyFullCoverProducerList,
                                                        outcastOpAttr, outcastUseOpSet, i, fullCoverLastReadPos,
                                                        fullCoverHasDupRead);
            } else if (useList.size() > 0) {
                outcastOpAttr.useList.insert(outcastOpAttr.useList.end(), useList.begin(), useList.end());
                for (auto& use : useList) {
                    UNUSED(use.operationIdx);
                    UNUSED(use.opType);
                    auto shape = callAttr->GetLinearImmediateArgList(use.offsetAttrIdx + dimSize,
                                                                     use.offsetAttrIdx + dimSize * 2, false);
                    UpdateCellMatchShape(outcastOpAttr.cellMatchTableDesc, shape);
                    MACHINE_LOGD("Minimal shape for outcast %d rawtensor magic %d op %zu %d is %s.\n", o->magic,
                                 o->GetRawMagic(), i, op.GetOpMagic(),
                                 IntVecToStr(ShapeToVector(outcastOpAttr.cellMatchTableDesc.cellShape)).c_str());
                }
                EncodeAnalysisOutCastConsumerByProducer(o, outcastOpAttr.useList, outcastOpAttr, outcastUseOpSet, i,
                                                        useListLastReadPos, useListHasDupRead);
            }
        }
        if (useListHasDupRead) {
            RebuildOutcastUseListByLastReadPos(outcastOpAttr.useList, useListLastReadPos);
        }
        if (fullCoverHasDupRead) {
            RebuildOutcastUseListByLastReadPos(outcastOpAttr.stitchPolicyFullCoverProducerList, fullCoverLastReadPos);
        }
        outcastOpAttr.useOpList.insert(outcastOpAttr.useOpList.end(), outcastUseOpSet.begin(), outcastUseOpSet.end());
        allOutcastUseOpSet.insert(outcastUseOpSet.begin(), outcastUseOpSet.end());
        UpdateCellMatchStrideAndSize(outcastOpAttr.cellMatchSize, outcastOpAttr.cellMatchTableDesc, o, dimSize);
        outcastOpAttrDict.insert({o, outcastOpAttr});
    }

    void EncodeOutCasts(std::set<uint32_t>& allInOutcastUseOpSet, std::set<int>& noSuccOpSet)
    {
        for (auto& i : outcastList) {
            tensorList.Insert(i);
            outcastRawTensorList.Insert(i->GetRawTensor());
            RecordRawTensor(i);
            InoutOperationAttr outcastOpAttr;
            auto dimSize = i->shape.size();
            outcastOpAttr.dim = dimSize;
            outcastOpAttr.cellMatchTableDesc = InitCellMatchTableDesc(i->GetShape(), std::vector<int64_t>(dimSize, 1));
            EncodeAnalysisOpUseOutCasts(i, allInOutcastUseOpSet, outcastOpAttr);
        }

        // Add edge from all stitchPolicyFullCoverProducerList's node to the single node
        size_t hubEntryLeast = 2;

        // Create 普通 HUB for stitchPolicyFullCover producers
        for (auto& i : outcastList) {
            auto& outcastOpAttr = outcastOpAttrDict[i];
            std::vector<DevAscendFunctionCallOperandUse> producers;
            for (auto& entry : outcastOpAttr.stitchPolicyFullCoverProducerList) {
                if (entry.opType == CellMatchOpType::READ) {
                    continue;
                }
                producers.push_back(entry);
            }
            if (producers.size() > hubEntryLeast) {
                outcastOpAttr.stitchPolicyFullCoverProducerHubOpIdx = callList.size();
                auto dummyOp = MakeDummyCall();
                callList.Insert(dummyOp);
                callOpPredDict[dummyOp] = producers.size();
                for (auto& producer : producers) {
                    auto callOp = callList[producer.operationIdx];
                    callOpSuccDict[callOp].Insert(dummyOp);
                }
                callOpSuccDict[dummyOp].Clear();
            } else {
                outcastOpAttr.stitchPolicyFullCoverProducerHubOpIdx = -1;
            }
        }

        for (size_t index = 0; index < callList.size(); index++) {
            Operation* op = callList[index];
            if (!callOpSuccDict.count(op) || callOpSuccDict.at(op).empty()) {
                noSuccOpList.push_back(index);
                noSuccOpSet.insert(index);
            }
        }
    }

    void EncodeAnalysisInCastConsumerUse(const std::shared_ptr<LogicalTensor>& index, Operation& op,
                                         CallOpAttribute* callAttr, size_t iOperandIdx, size_t opIdx, size_t dimSize,
                                         InoutOperationAttr& incastOpAttr, std::set<uint32_t>& incastUseOpSet)
    {
        auto& iOperand = op.GetIOperands()[iOperandIdx];
        auto coaIndex = op.GetIOpAttrOffset(iOperandIdx) + COA_INDEX_DIM_BASE;
        if (index->tensor->rawmagic != iOperand->tensor->rawmagic) {
            return;
        }
        ASSERT(DevCommonErr::PARAM_CHECK_FAILED, iOperand->GetShape().size() == dimSize)
            << "Shape size mismatch: expected: " << dimSize << ", got " << iOperand->GetShape().size()
            << " for operand: " << iOperandIdx;
        std::vector<int64_t> offset = callAttr->GetLinearImmediateArgList(coaIndex, coaIndex + dimSize, true);
        std::vector<int64_t> shape = callAttr->GetLinearImmediateArgList(coaIndex + dimSize, coaIndex + dimSize * 0x2,
                                                                         false);
        if (shape == Shape(shape.size())) { // 跳过全0
            return;
        }
        // fullCover → POLICY list only (skip DEFAULT cell match); partial → DEFAULT useList
        if (offset == std::vector<int64_t>(dimSize, 0) && shape == index->GetShape()) {
            incastOpAttr.stitchPolicyFullCoverConsumerList.emplace_back(opIdx, coaIndex, CellMatchOpType::READ);
            MACHINE_LOGD("FullCover incast consumer: incast %d rawtensor magic %d coaIndex %d op %zu %d shape %s.\n",
                         index->magic, index->GetRawMagic(), coaIndex, opIdx, op.GetOpMagic(),
                         IntVecToStr(shape).c_str());
        } else {
            incastOpAttr.useList.emplace_back(opIdx, coaIndex, CellMatchOpType::READ);
            UpdateCellMatchShape(incastOpAttr.cellMatchTableDesc, shape);
            MACHINE_LOGD("Partial incast consumer: incast %d rawtensor magic %d coaIndex %d op %zu %d cellShape %s.\n",
                         index->magic, index->GetRawMagic(), coaIndex, opIdx, op.GetOpMagic(),
                         IntVecToStr(ShapeToVector(incastOpAttr.cellMatchTableDesc.cellShape)).c_str());
        }
        incastUseOpSet.insert(static_cast<uint32_t>(opIdx));
    }

    void EncodeIncasts(std::set<uint32_t>& allInOutcastUseOpSet)
    {
        for (auto& index : incastList) {
            tensorList.Insert(index);
            incastRawTensorList.Insert(index->GetRawTensor());
            RecordRawTensor(index);
            InoutOperationAttr incastOpAttr;
            auto dimSize = index->shape.size();
            incastOpAttr.dim = dimSize;
            incastOpAttr.cellMatchTableDesc = InitCellMatchTableDesc(index->GetShape(),
                                                                     std::vector<int64_t>(dimSize, 1));
            std::set<uint32_t> incastUseOpSet;
            for (size_t j = 0; j < callList.size(); j++) {
                auto& op = *callList[j];
                auto callAttr = dynamic_cast<CallOpAttribute*>(op.GetOpAttribute().get());
                for (size_t k = 0; k < op.GetIOperands().size(); ++k) {
                    EncodeAnalysisInCastConsumerUse(index, op, callAttr, k, j, dimSize, incastOpAttr, incastUseOpSet);
                }
            }
            UpdateCellMatchStrideAndSize(incastOpAttr.cellMatchSize, incastOpAttr.cellMatchTableDesc, index, dimSize);
            incastOpAttr.useOpList.insert(incastOpAttr.useOpList.end(), incastUseOpSet.begin(), incastUseOpSet.end());
            incastOpAttrDict.insert({index, incastOpAttr});
            allInOutcastUseOpSet.insert(incastUseOpSet.begin(), incastUseOpSet.end());
        }

        for (size_t index = 0; index < callList.size(); index++) {
            Operation* op = callList[index];
            if (!callOpPredDict.count(op) || callOpPredDict.at(op) == 0) {
                noPredOpList.push_back(index);
            }
        }
    }

    struct Hasher {
        template <typename T>
        std::size_t operator()(const OrderedSet<T>& operationSet) const
        {
            size_t res = 0;
            for (auto op : operationSet) {
                res ^= std::hash<Operation*>{}(op);
            }
            return res;
        }
    };

    Operation* MakeDummyCall()
    {
        LogicalTensors inputs, outputs;
        auto opAttr = std::make_shared<CallOpAttribute>();
        auto dummyOp = std::make_shared<Operation>(*devRoot, Opcode::OP_CALL);
        dummyOp->SetOpAttribute(opAttr);
        dummyOpList.push_back(dummyOp);
        ASSERT(DevCommonErr::PARAM_INVALID, GetCoreType(dummyOp.get()) == static_cast<int>(CoreType::HUB))
            << "GetCoreType return unexpected value: " << GetCoreType(dummyOp.get())
            << ", expected:  " << static_cast<int>(CoreType::HUB);
        return dummyOp.get();
    }

    Operation* MakeDummyCallHubMix()
    {
        auto opAttr = std::make_shared<CallOpAttribute>();
        opAttr->SetCalleeHash(FunctionHash(HUB_MIX_DUMMY_HASH));
        auto dummyOp = std::make_shared<Operation>(*devRoot, Opcode::OP_CALL);
        dummyOp->SetOpAttribute(opAttr);
        dummyOpList.push_back(dummyOp);
        ASSERT(DevCommonErr::PARAM_INVALID, GetCoreType(dummyOp.get()) == static_cast<int>(CoreType::HUB_MIX))
            << "GetCoreType return unexpected value: " << GetCoreType(dummyOp.get())
            << ", expected:  " << static_cast<int>(CoreType::HUB_MIX);
        return dummyOp.get();
    }

    int GetCoreType(Operation* callop)
    {
        int leafIndex = calleeHashIndexDict.at(callop->GetCalleeHash().GetHash());
        return cceCodeInfoList[leafIndex].coreType;
    }

    void RemoveDeadHubCall(Function* tdevRoot, std::vector<Operation*>& /* callOpList */)
    {
        std::vector<Operation*> deadCallOps;
        for (auto& [callOp, succOps] : callOpSuccDict) {
            if (!IsHubType(GetCoreType(callOp)) || (succOps.size() != 0)) {
                continue;
            }
            /* When HUB's oOperands have rootFunc outcast, do not remove it. (eg. Reshape as rootFunc output) */
            bool needSave = false;
            for (const auto& out : callOp->oOperand) {
                if (tdevRoot->IsFromOutCast(out)) {
                    needSave = true;
                    break;
                }
            }
            if (needSave) {
                continue;
            }
            /*  Find all hub callop that has no successors, mark it is no need to schedule:
             *  1. mark pred to be zero
             *  2. remove it from the successor of all callop
             */
            callOpPredDict[callOp] = 0;
            deadCallOps.push_back(callOp);
        }

        for (auto& [callOp, succOps] : callOpSuccDict) {
            UNUSED(callOp);
            succOps.Remove(deadCallOps);
        }
    }

    void ReplaceSuccessorWithHub(std::vector<Operation*>& callOpList, int optimizeLimit)
    {
        // dict from successor set to callop set that has the successor.
        OrderedMap<OrderedSet<Operation*>, OrderedSet<Operation*>, Hasher> predDict;
        for (auto& callOp : callOpList) {
            if (callOpSuccDict.count(callOp)) {
                auto succOps = callOpSuccDict[callOp];
                predDict[succOps].Insert(callOp);
            }
        }

        for (auto& [succSet, predSet] : predDict) {
            int optimizeCnt = succSet.size() * predSet.size() - succSet.size() - predSet.size();
            if (optimizeCnt > optimizeLimit) {
                auto dummyOp = MakeDummyCall();
                callOpList.push_back(dummyOp);

                callOpSuccDict[dummyOp] = succSet;
                for (auto& pred : predSet) {
                    callOpSuccDict[pred].Clear();
                    callOpSuccDict[pred].Insert(dummyOp);
                }

                callOpPredDict[dummyOp] = predSet.size();
                for (auto& succ : succSet) {
                    callOpPredDict[succ] -= predSet.size() - 1;
                }
            }
        }
    }

    void PrintColorGraph(int colorNum)
    {
        MACHINE_LOGI("*********** Call OP Graph ***********\n");
        for (int index = 0; index < colorNum; index++) {
            MACHINE_LOGI("%d: %zu", index, colorOutGraph[index].size());
            MACHINE_LOGI("%s", IntVecToStr(colorOutGraph[index]).c_str());
        }
        int outCount = 0;
        for (int index = 0; index < colorNum; index++) {
            outCount += colorOutGraph[index].size();
        }
        MACHINE_LOGI("Total out: %d\n", outCount);
    }

    inline void FindAllReachableNodes(int start_node, std::unordered_map<int, std::vector<int>>& outGraph,
                                      std::vector<std::unordered_set<int>>& reachable, std::vector<int>& visited)
    {
        visited[start_node] = 1;
        reachable[start_node].insert(start_node);
        for (int v : outGraph[start_node]) {
            if (visited[v] == 0) {
                FindAllReachableNodes(v, outGraph, reachable, visited);
            }
            reachable[start_node].insert(reachable[v].begin(), reachable[v].end());
        }
    }

    void FindRedundantEdges(int colorNum, std::vector<std::vector<int>>& redundantColorOutGraph)
    {
        std::vector<std::unordered_set<int>> reachable(colorNum);
        std::vector<int> visited(colorNum, 0);
        for (int index = 0; index < colorNum; ++index) {
            if (visited[index] == 0) {
                FindAllReachableNodes(index, colorOutGraph, reachable, visited); // DFS记忆化计算
            }
        }
        for (int j = 0; j < colorNum; ++j) {
            for (int k : colorOutGraph[j]) {
                bool is_redundant = false;
                for (int w : colorOutGraph[j]) {
                    if (w == k) {
                        continue;
                    }
                    if (reachable[w].count(k)) {
                        is_redundant = true;
                        break;
                    }
                }
                if (is_redundant) {
                    redundantColorOutGraph[j].push_back(k);
                }
            }
        }
    }

    void EraseRedundantColorEdges(std::vector<Operation*>& callopList)
    {
        int colorNum = callopList.size();
        std::vector<std::vector<int>> redundantColorOutGraph(colorNum);
        // Find redundant edges
        FindRedundantEdges(colorNum, redundantColorOutGraph);
        // Erase redundant edges
        for (int index = 0; index < colorNum; index++) {
            // make redundantColorOutGraph[index]'s order grow
            std::sort(redundantColorOutGraph[index].begin(), redundantColorOutGraph[index].end());
            MACHINE_LOGI("Redundant outgraph of %d is %s.", index, IntVecToStr(redundantColorOutGraph[index]).c_str());
            // update color_out_graph
            std::vector<int> newGraph;
            size_t n = 0U;
            // for each index -> * -> k && index -> k
            for (int k : redundantColorOutGraph[index]) {
                // for each index -> x before x is k (so that x != k), add index -> x
                while (colorOutGraph[index][n] != k) {
                    newGraph.push_back(colorOutGraph[index][n]);
                    callOpSuccDict[callopList[index]].Insert(callopList[colorOutGraph[index][n]]);
                    callOpPredDict[callopList[colorOutGraph[index][n]]]++;
                    n++;
                }
                // until x == k, skip x
                n++;
            }
            // add index -> x, where x is the rest (larger than the largest k)
            while (n < colorOutGraph[index].size()) {
                newGraph.push_back(colorOutGraph[index][n]);
                callOpSuccDict[callopList[index]].Insert(callopList[colorOutGraph[index][n]]);
                callOpPredDict[callopList[colorOutGraph[index][n]]]++;
                n++;
            }
            colorOutGraph[index] = newGraph;
        }
    }

    // 为单个 wrapId 分组插入 HUB_MIX，重接依赖图：
    //   pred -> [callop...]   变为   pred -> hubMix -> [callop...]
    // - 汇聚分组内所有 op 的公共前驱 predSet；
    // - 前驱的后继由原 callop 改指 HUB_MIX；HUB_MIX 后继指向全部 callop，callop 前驱数置 1；
    // - 记录 callop -> hubMix 到 opToHubOp。
    void InsertHubMixForWrapGroup(std::vector<Operation*>& callopList, const std::vector<Operation*>& callops,
                                  const std::unordered_map<Operation*, std::vector<Operation*>>& predListDict,
                                  std::unordered_map<Operation*, Operation*>& opToHubOp)
    {
        std::unordered_set<Operation*> predSet;
        for (auto* op : callops) {
            auto it = predListDict.find(op);
            if (it != predListDict.end()) {
                for (auto* pred : it->second) {
                    predSet.insert(pred);
                }
            }
        }

        auto* hubMixOp = MakeDummyCallHubMix();
        callopList.push_back(hubMixOp);

        for (auto* pred : predSet) {
            callOpSuccDict[pred].Remove(callops);
            callOpSuccDict[pred].Insert(hubMixOp);
        }
        callOpPredDict[hubMixOp] = static_cast<uint64_t>(predSet.size());

        for (auto* callOp : callops) {
            callOpSuccDict[hubMixOp].Insert(callOp);
            callOpPredDict[callOp] = 1;
            opToHubOp[callOp] = hubMixOp;
        }
    }

    // 为所有 mix 任务按 wrapId 统一插入 HUB_MIX 同步节点（含无前驱的，作为零前驱同步屏障）；
    // opToHubOp（→opIdxToHubOpIdx_）收集所有 wrap 的映射，供 incast/outcast 消费者查表替换为 HUB_MIX。
    std::unordered_map<Operation*, Operation*> AddHubMixForWrapTask(std::vector<Operation*>& callopList)
    {
        std::unordered_map<Operation*, std::vector<Operation*>> predListDict;
        for (auto& [producer, succSet] : callOpSuccDict) {
            for (auto& succ : succSet) {
                predListDict[succ].push_back(producer);
            }
        }

        std::map<int, std::vector<Operation*>> wrapIdToCallops;
        for (auto* op : callopList) {
            auto callAttr = dynamic_cast<CallOpAttribute*>(op->GetOpAttribute().get());
            if (callAttr == nullptr || callAttr->wrapId == INVALID_WRAPID) {
                continue;
            }
            wrapIdToCallops[callAttr->wrapId].push_back(op);
        }

        std::unordered_map<Operation*, Operation*> opToHubOp;
        for (auto& [wrapId, callops] : wrapIdToCallops) {
            (void)wrapId;
            InsertHubMixForWrapGroup(callopList, callops, predListDict, opToHubOp);
        }
        return opToHubOp;
    }

    void AddDummyCallsAtBeginningAndEnding(std::vector<Operation*>& callopList)
    {
        static constexpr size_t OPTIMIZATION_THRESHOLD = 3;

        std::vector<Operation*> zeroPreds;
        std::vector<Operation*> zeroSuccs;
        for (auto* op : callopList) {
            if (callOpPredDict[op] == 0) {
                zeroPreds.push_back(op);
            }
            if (callOpSuccDict[op].empty()) {
                zeroSuccs.push_back(op);
            }
        }

        // Zero predecessors
        if (zeroPreds.size() >= OPTIMIZATION_THRESHOLD) {
            auto* dummyOp = MakeDummyCall();
            callopList.push_back(dummyOp);
            callOpPredDict[dummyOp] = 0;
            for (auto* op : zeroPreds) {
                ASSERT(DevCommonErr::PARAM_CHECK_FAILED, callOpPredDict[op] == 0)
                    << "callOpPredDict[op] is not zero:" << callOpPredDict[op] << ", expected 0";
                callOpSuccDict[dummyOp].Insert(op);
                callOpPredDict[op] = 1;
            }
            ASSERT(DevCommonErr::PARAM_CHECK_FAILED, callOpSuccDict[dummyOp].size() == zeroPreds.size())
                << "callOpSuccDict[dummyOp] size mismatch: expected: " << zeroPreds.size()
                << ", got: " << callOpSuccDict[dummyOp].size();
        }

        // Zero successors
        if (zeroSuccs.size() >= OPTIMIZATION_THRESHOLD) {
            auto* dummyOp = MakeDummyCall();
            callopList.push_back(dummyOp);
            callOpSuccDict[dummyOp] = {};
            callOpPredDict[dummyOp] = zeroSuccs.size();
            for (auto* op : zeroSuccs) {
                ASSERT(DevCommonErr::PARAM_CHECK_FAILED, callOpSuccDict[op].empty())
                    << "callOpSuccDict[op] is not empty, expect empty";
                callOpSuccDict[op].Insert(dummyOp);
            }
        }
    }

    void AddDependOperandsToColorGraphForMix(std::vector<Operation*>& callopListVec,
                                             std::unordered_map<Operation*, int>& callopIndexDict)
    {
        for (auto& op : callopListVec) {
            for (auto& depend : op->GetDependOperands()) {
                for (auto& producer : depend->GetProducers()) {
                    colorOutGraph[callopIndexDict[producer]].push_back(callopIndexDict[op]);
                }
            }
        }
    }

    void SortCallOpList(std::vector<Operation*>& callopList,
                        const std::unordered_map<Operation*, int>& callopCoreTypeDict)
    {
        std::sort(callopList.begin(), callopList.end(), [&](Operation* lhs, Operation* rhs) {
            if (callOpPredDict[lhs] != callOpPredDict[rhs]) {
                return callOpPredDict[lhs] < callOpPredDict[rhs];
            }
            ASSERT(DevCommonErr::PARAM_CHECK_FAILED, callopCoreTypeDict.count(lhs))
                << "lhs operation " << lhs << " is not found in callopCoreTypeDict";
            ASSERT(DevCommonErr::PARAM_CHECK_FAILED, callopCoreTypeDict.count(rhs))
                << "rhs operation " << rhs << " is not found in callopCoreTypeDict";
            return callopCoreTypeDict.at(lhs) < callopCoreTypeDict.at(rhs);
        });
    }

    void EncodeZeroPredCount(std::vector<Operation*>& callopList)
    {
        std::unordered_map<Operation*, int> callopCoreTypeDict;
        for (auto& op : callopList) {
            auto callOpAttr = std::static_pointer_cast<CallOpAttribute>(op->GetOpAttribute());
            auto calleeHash = callOpAttr->GetCalleeHash().GetHash();
            ASSERT(DevCommonErr::PARAM_CHECK_FAILED, calleeHashIndexDict.count(calleeHash))
                << "calleeHash 0x" << std::hex << calleeHash << " is not found in calleeHashIndexDict";
            int cceIndex = calleeHashIndexDict.find(calleeHash)->second;
            ASSERT(DevCommonErr::PARAM_CHECK_FAILED, cceIndex < static_cast<int>(cceCodeInfoList.size()))
                << "cceIndex " << cceIndex << " exceeds cceCodeInfoList size: " << cceCodeInfoList.size();

            uint32_t coreType = cceCodeInfoList[cceIndex].coreType;
            ASSERT(DevCommonErr::PARAM_INVALID, coreType == static_cast<uint32_t>(CoreType::AIV) ||
                                                    coreType == static_cast<uint32_t>(CoreType::AIC) ||
                                                    coreType == static_cast<uint32_t>(CoreType::HUB) ||
                                                    coreType == static_cast<uint32_t>(CoreType::HUB_MIX) ||
                                                    coreType == static_cast<uint32_t>(CoreType::AICPU))
                << "invalid coreType " << coreType << " for op " << op;
            callopCoreTypeDict[op] = coreType;
        }

        SortCallOpList(callopList, callopCoreTypeDict);

        totalZeroPred = callopList.size();
        for (size_t index = 0; index < callopList.size(); index++) {
            if (callOpPredDict[callopList[index]] != 0) {
                totalZeroPred = index;
                break;
            }
        }
        for (size_t index = totalZeroPred; index < callopList.size(); index++) {
            ASSERT(DevCommonErr::PARAM_CHECK_FAILED, callOpPredDict[callopList[index]] != 0)
                << "callOpPredDict[callopList[" << index << "]] is zero, callopList[" << index
                << "] = " << callopList[index];
        }

        for (uint32_t index = 0; index < totalZeroPred; index++) {
            if (callopCoreTypeDict[callopList[index]] == static_cast<uint32_t>(CoreType::AIV)) {
                totalZeroPredAIV++;
            } else if (callopCoreTypeDict[callopList[index]] == static_cast<uint32_t>(CoreType::AIC)) {
                totalZeroPredAIC++;
            } else if (callopCoreTypeDict[callopList[index]] == static_cast<uint32_t>(CoreType::HUB) ||
                       callopCoreTypeDict[callopList[index]] == static_cast<uint32_t>(CoreType::HUB_MIX)) {
                totalZeroPredHub++;
            } else if (callopCoreTypeDict[callopList[index]] == static_cast<uint32_t>(CoreType::AICPU)) {
                totalZeroPredAicpu++;
            } else {
                ASSERT(DevCommonErr::PARAM_INVALID, false)
                    << "Invalid coreType for callopList[" << index << "], op : " << callopList[index];
            }
        }
    }

    void EncodeCopyOutReslove(
        std::unordered_map<Operation*, std::unordered_map<Operation*, int>>& producerConsumerOOperandIndexDict)
    {
        FunctionCache& cache = Program::GetInstance().GetFunctionCache();
        for (auto& [callop, succSet] : callOpSuccDict) {
            Function* devLeafFunc = cache.GetCacheFunction(callop->GetCalleeHash());
            if (devLeafFunc == nullptr) {
                ASSERT(DevCommonErr::PARAM_CHECK_FAILED, IsHubType(GetCoreType(callop)))
                    << "GetCoreType return unexpected value: " << GetCoreType(callop)
                    << ", expectedBlockFunction: " << static_cast<int>(CoreType::HUB) << " for callop: " << callop;
                copyOutResolveSuccIndexListDict[callop] = std::vector<int>({0});
                continue;
            }
            std::shared_ptr<LeafFuncAttribute> leafAttr = devLeafFunc->GetLeafFuncAttribute();

            if (leafAttr == nullptr) {
                MACHINE_LOGE(ProgEncodeErr::LEAF_CALLEE_ATTR_NULL, "Leaf Attr of leaf function %s is nullptr.",
                             callop->GetCalleeMagicName().c_str());
                continue;
            }
            if (leafAttr->outcastCopyOutResolveCounterList.size() == 0) {
                copyOutResolveSuccIndexListDict[callop] = std::vector<int>({0});
                continue;
            }

            std::vector<OrderedSet<Operation*>> copyOutResolveSetList;
            copyOutResolveSetList.resize(leafAttr->copyOutResolveSize);

            OrderedSet<Operation*> nonCopyOutResolveSuccSet;
            for (auto& succ : succSet) {
                if (producerConsumerOOperandIndexDict.count(callop) &&
                    producerConsumerOOperandIndexDict[callop].count(succ)) {
                    auto ooperandIndex = producerConsumerOOperandIndexDict[callop][succ];
                    int copyOutResolveCounter = leafAttr->outcastCopyOutResolveCounterList[ooperandIndex];
                    copyOutResolveSetList[copyOutResolveCounter].Insert(succ);
                } else {
                    nonCopyOutResolveSuccSet.Insert(succ);
                }
            }

            std::vector<int> copyOutResolveSuccIndexList;
            std::vector<Operation*> copyOutResolveSuccList;
            for (int k = 0; k < leafAttr->copyOutResolveSize; k++) {
                OrderedSet<Operation*>& succ = copyOutResolveSetList[k];
                copyOutResolveSuccIndexList.push_back(copyOutResolveSuccList.size());
                copyOutResolveSuccList.insert(copyOutResolveSuccList.end(), succ.begin(), succ.end());
            }
            copyOutResolveSuccIndexList.push_back(copyOutResolveSuccList.size());
            ASSERT(DevCommonErr::PARAM_CHECK_FAILED, copyOutResolveSuccIndexList[0] == 0)
                << "copyOutResolveSuccIndexList[0] is " << copyOutResolveSuccIndexList[0] << ", expected 0";
            copyOutResolveSuccList.insert(copyOutResolveSuccList.end(), nonCopyOutResolveSuccSet.begin(),
                                          nonCopyOutResolveSuccSet.end());

            // Assert: succ set are the same
            ASSERT(DevCommonErr::PARAM_CHECK_FAILED,
                   std::set<Operation*>(succSet.begin(), succSet.end()) ==
                       std::set<Operation*>(copyOutResolveSuccList.begin(), copyOutResolveSuccList.end()))
                << "succSet and copyOutResolveSuccList content mismatch";

            succSet.Clear();
            for (Operation* copyOutResolveSucc : copyOutResolveSuccList) {
                // Assert: no duplicated item in copyOutResolveSuccList
                ASSERT(DevCommonErr::PARAM_CHECK_FAILED, succSet.Insert(copyOutResolveSucc))
                    << "Duplicate item " << copyOutResolveSucc << " found in copyOutResolveSuccList";
            }

            copyOutResolveSuccIndexListDict[callop] = copyOutResolveSuccIndexList;
        }
    }

    void InsertProducerConsmerOOperandIndexDict(
        std::shared_ptr<LeafFuncAttribute> leafAttr, Operation* op, Operation* consumer,
        std::shared_ptr<LogicalTensor> o,
        std::unordered_map<Operation*, std::unordered_map<Operation*, int>>& producerConsumerOOperandIndexDict)
    {
        if (producerConsumerOOperandIndexDict.count(op) && producerConsumerOOperandIndexDict[op].count(consumer)) {
            // There might be multiple ooperand of op that is consumed by the same consumer. So when
            // it happens, we need to select the ooperand with the biggest counter.
            int currIndex = producerConsumerOOperandIndexDict[op][consumer];
            int oIndex = op->GetOOperandIndex(o);
            if (leafAttr != nullptr && leafAttr->outcastCopyOutResolveCounterList.size() != 0) {
                // When there is leaf, and the root is marked as resolve, the leafAttr records the biggest counter.
                int currCounter = leafAttr->outcastCopyOutResolveCounterList[currIndex];
                int oCounter = leafAttr->outcastCopyOutResolveCounterList[oIndex];
                if (oCounter > currCounter) {
                    producerConsumerOOperandIndexDict[op][consumer] = oIndex;
                }
            } else {
                // Otherwise, we use any, which is the first
            }
        } else {
            producerConsumerOOperandIndexDict[op][consumer] = op->GetOOperandIndex(o);
        }
    }

    void BuildColorOutGraphAndProducerConsumerOOperandDict(
        std::vector<Operation*>& callopList,
        std::unordered_map<std::shared_ptr<LogicalTensor>, OrderedSet<Operation*>>& consumerDict,
        std::unordered_map<Operation*, int>& callopIndexDict,
        std::unordered_map<Operation*, std::unordered_map<Operation*, int>>& producerConsumerOOperandIndexDict)
    {
        FunctionCache& cache = Program::GetInstance().GetFunctionCache();
        for (auto& op : callopList) {
            Function* devLeafFunc = cache.GetCacheFunction(op->GetCalleeHash());
            std::shared_ptr<LeafFuncAttribute> leafAttr = devLeafFunc != nullptr ? devLeafFunc->GetLeafFuncAttribute() :
                                                                                   nullptr;

            for (auto& o : op->GetOOperands()) {
                for (auto& consumer : consumerDict[o]) {
                    if (consumer->GetOpcode() != Opcode::OP_CALL) {
                        // This should be prevented from the above: only call op is considered as consumer
                        continue;
                    }
                    if (op == consumer) {
                        // Consumer and producer can not be the same.
                        continue;
                    }
                    // Index for callop to its ooperand's consumer callop index list
                    colorOutGraph[callopIndexDict[op]].push_back(callopIndexDict[consumer]);
                    InsertProducerConsmerOOperandIndexDict(leafAttr, op, consumer, o,
                                                           producerConsumerOOperandIndexDict);
                }
            }
        }
        AddDependOperandsToColorGraphForMix(callopList, callopIndexDict);
        AddTokenDependEdgesToColorGraph(callopList, callopIndexDict, colorOutGraph);
        for (size_t idx = 0; idx < callopList.size(); idx++) {
            std::sort(colorOutGraph[idx].begin(), colorOutGraph[idx].end());
            // remove repeated idx in ooperand's consumer callop idx list
            colorOutGraph[idx].resize(std::unique(colorOutGraph[idx].begin(), colorOutGraph[idx].end()) -
                                      colorOutGraph[idx].begin());
        }
    }

    void BuildCallopList(std::vector<Operation*>& callopList, std::unordered_map<Operation*, int>& callopIndexDict,
                         std::unordered_map<std::shared_ptr<LogicalTensor>, OrderedSet<Operation*>>& consumerDict)
    {
        for (auto& op : devRoot->Operations()) {
            if (op.GetOpcode() == Opcode::OP_CALL) {
                callopIndexDict[&op] = callopList.size();
                callopList.push_back(&op);

                for (auto& i : op.GetIOperands()) {
                    tensorList.Insert(i);
                    RecordRawTensor(i);
                    consumerDict[i].Insert(&op);
                }
                for (auto& o : op.GetOOperands()) {
                    tensorList.Insert(o);
                    RecordRawTensor(o);
                }
            }
        }
        for (auto& op : callopList) {
            callOpPredDict[op] = 0;
            callOpSuccDict[op].clear();
        }
    }

    void EncodeInOutCast()
    {
        std::set<uint32_t> allInOutcastUseOpSet;
        std::set<int> noSuccOpSet;
        EncodeIncasts(allInOutcastUseOpSet);
        EncodeOutCasts(allInOutcastUseOpSet, noSuccOpSet);

        /*
         * As we need a reference of null, we use the 0-th element for the reference of null for
         * DevAscendFunctionDuppedData So stitchCount starts from 1, as the 0-th is for the reference of null.
         */
        stitchCount = 1;
        for (size_t index = 0; index < callList.size(); index++) {
            if (allInOutcastUseOpSet.count(index) || noSuccOpSet.count(index)) {
                stitchIndexList.push_back(stitchCount);
                stitchCount++;
            } else {
                stitchIndexList.push_back(0);
            }
        }
    }

    EncodeDevAscendFunctionInfo(Function* dyndev, const std::unordered_map<uint64_t, int>& tHashIndexDict,
                                const std::vector<CceCodeInfo>& tCceCodeInfoList,
                                const SymbolicExpressionTable* tExpressionTable, Function* tdevRoot)
        : devRoot(tdevRoot),
          calleeHashIndexDict(tHashIndexDict),
          cceCodeInfoList(tCceCodeInfoList),
          expressionTable(tExpressionTable)
    {
        ASSERT(DevCommonErr::PARAM_CHECK_FAILED, dyndev->GetDyndevAttribute()->rootTileDict.count(devRoot))
            << "devRoot: " << devRoot << " not found in rootTileDict of dyndev";
        devTile = dyndev->GetDyndevAttribute()->rootTileDict[devRoot];
        if (dyndev->GetDyndevAttribute()->valueDependDescDict.count(devTile)) {
            valueDependDesc = dyndev->GetDyndevAttribute()->valueDependDescDict[devTile];
        }

        std::unordered_map<std::shared_ptr<LogicalTensor>, OrderedSet<Operation*>> consumerDict;
        std::unordered_map<Operation*, int> callopIndexDict;
        std::vector<Operation*> callopList;
        std::unordered_map<Operation*, std::unordered_map<Operation*, int>> producerConsumerOOperandIndexDict;

        rawName = devRoot->GetRawName();
        incastList = devRoot->GetIncast();
        outcastList = devRoot->GetOutcast();
        incastSet.insert(incastList.begin(), incastList.end());
        outcastSet.insert(outcastList.begin(), outcastList.end());

        BuildCallopList(callopList, callopIndexDict, consumerDict);
        BuildColorOutGraphAndProducerConsumerOOperandDict(callopList, consumerDict, callopIndexDict,
                                                          producerConsumerOOperandIndexDict);

        PrintColorGraph(callopList.size());
        EraseRedundantColorEdges(callopList);
        PrintColorGraph(callopList.size());

        RemoveDeadHubCall(tdevRoot, callopList);
        ReplaceSuccessorWithHub(callopList, 10); // add dummp op at least 10 depends can be reduced

        AddDummyCallsAtBeginningAndEnding(callopList);
        auto opToHubOp = AddHubMixForWrapTask(callopList);

        EncodeCopyOutReslove(producerConsumerOOperandIndexDict);
        EncodeZeroPredCount(callopList);

        for (auto& op : callopList) {
            callList.Insert(op);
        }
        for (auto& [consumerOp, hubMixOp] : opToHubOp) {
            opIdxToHubOpIdx_[static_cast<int>(callList.GetIndex(consumerOp))] = static_cast<int>(
                callList.GetIndex(hubMixOp));
        }

        EncodeInOutCast();

        // dummy op might be inserted in EncodeOutcast, add dummy copy out resolve.
        for (auto& op : callList) {
            if (!copyOutResolveSuccIndexListDict.count(op)) {
                copyOutResolveSuccIndexListDict[op] = std::vector<int>({0});
            }

            if (IsHubType(GetCoreType(op))) {
                hubOpCount++;
            }
        }
    }

    // bitmap 位操作：idx 为 op 索引，bitmap 以 uint64 为单位
    static void SetBitmapBit(std::vector<uint64_t>& bits, int idx) { bits[idx / 64] |= (1ULL << (idx % 64)); }
    static bool IsBitmapBitSet(const std::vector<uint64_t>& bits, int idx)
    {
        return (bits[idx / 64] & (1ULL << (idx % 64))) != 0;
    }

    // 标记尾Hub，Hub节点后为空，或者多级Hub串联，最后一个Hub为空的场景
    // 场景：HubA; HubA -> HubB -> HubC
    void MarkDeadEndHubs(std::vector<uint64_t>& deadEndBits)
    {
        std::vector<Operation*> hubNodes;
        std::unordered_map<Operation*, std::vector<Operation*>> hubPreds;
        // 1. 标记 succ 为空的叶子 HUB，同时构建 HUB->HUB 前驱映射
        for (auto& [op, succs] : callOpSuccDict) {
            if (!IsHubType(GetCoreType(op)))
                continue;
            hubNodes.push_back(op);
            if (succs.empty()) {
                SetBitmapBit(deadEndBits, callList.GetIndex(op));
            }
            for (auto* succ : succs) {
                if (IsHubType(GetCoreType(succ))) {
                    hubPreds[succ].push_back(op);
                }
            }
        }

        // 2. 叶子 HUB 的 HUB 前驱入队
        std::queue<Operation*> worklist;
        for (auto* op : hubNodes) {
            if (IsBitmapBitSet(deadEndBits, callList.GetIndex(op))) {
                for (auto* pred : hubPreds[op])
                    worklist.push(pred);
            }
        }

        // 3. 反向传播：后继全 dead 则标记自身，继续通知前驱
        while (!worklist.empty()) {
            auto* op = worklist.front();
            worklist.pop();
            int idx = callList.GetIndex(op);
            if (IsBitmapBitSet(deadEndBits, idx)) {
                continue;
            }
            bool allDead = true;
            for (auto* succ : callOpSuccDict[op]) {
                if (!IsBitmapBitSet(deadEndBits, callList.GetIndex(succ))) {
                    allDead = false;
                    break;
                }
            }
            if (allDead && !callOpSuccDict[op].empty()) {
                SetBitmapBit(deadEndBits, idx);
                for (auto* pred : hubPreds[op])
                    worklist.push(pred);
            }
        }
    }

    // 标记尾task（后继为空或者后继均为 deadEnd HUB）。
    // 场景： taskA; taskA -> HubA
    void MarkTailTasks(const std::vector<uint64_t>& deadEndBits, std::vector<uint64_t>& tailBits)
    {
        for (auto& [op, succs] : callOpSuccDict) {
            if (IsHubType(GetCoreType(op)))
                continue;
            bool allDeadEnd = true;
            for (auto* succ : succs) {
                if (!IsBitmapBitSet(deadEndBits, callList.GetIndex(succ))) {
                    allDeadEnd = false;
                    break;
                }
            }
            if (allDeadEnd) {
                SetBitmapBit(tailBits, callList.GetIndex(op));
            }
        }
    }

    void MarkResolveBitmaps(DevAscendFunction* devFunc, uintdevptr_t& initOffset, bool fillContent)
    {
        int opCount = static_cast<int>(callList.size());
        int bitmapWords = (opCount + 63) / 64;
        devFunc->deadEndHubBitmap_.HostInitDataSizeOffset(initOffset, bitmapWords);
        devFunc->tailTaskBitmap_.HostInitDataSizeOffset(initOffset, bitmapWords);
        if (!fillContent)
            return;

        std::vector<uint64_t> deadEndBits(bitmapWords, 0);
        std::vector<uint64_t> tailBits(bitmapWords, 0);

        MarkDeadEndHubs(deadEndBits);
        MarkTailTasks(deadEndBits, tailBits);

        for (int i = 0; i < bitmapWords; i++) {
            devFunc->At(devFunc->deadEndHubBitmap_, i) = deadEndBits[i];
            devFunc->At(devFunc->tailTaskBitmap_, i) = tailBits[i];
        }
    }

    void Init(DevAscendFunction* devFunc, const EncodeDevAscendFunctionParam& param, bool fillContent)
    {
        uintdevptr_t initOffset = reinterpret_cast<uintdevptr_t>(&devFunc->data) -
                                  reinterpret_cast<uintdevptr_t>(devFunc);
        DevAscendFunctionPredInfo predInfo = {totalZeroPred, totalZeroPredAIV, totalZeroPredAIC, totalZeroPredHub,
                                              totalZeroPredAicpu};
        devFunc->sourceFunc = nullptr;
        devFunc->getInputDataCount = valueDependDesc.getInputDataCount;
        devFunc->getTensorDataCount = valueDependDesc.getTensorDataCount;
        devFunc->hubOpCount_ = hubOpCount;
        devFunc->unrollTimes = ParseUnrollTimesFromName(rawName);
        devFunc->InitIncastOutcastAttr(initOffset, incastList, outcastList, fillContent);
        devFunc->InitOperationDynamicField(initOffset, predInfo, stitchCount, calleeHashIndexDict, expressionTable,
                                           callList, incastList, outcastList, callOpSuccDict, fillContent);
        devFunc->InitRawTensorAndMemoryRequirement(initOffset, incastRawTensorList, outcastRawTensorList, rawTensorList,
                                                   rawMagicToRawTensor, rawAttrs, param, expressionTable, fillContent);
        devFunc->InitTensor(initOffset, tensorList, rawTensorList, fillContent);
        devFunc->InitOperation(initOffset, expressionTable, callList, tensorList, rawTensorList, callOpPredDict,
                               callOpSuccDict, calleeHashIndexDict, stitchIndexList, noPredOpList, noSuccOpList,
                               copyOutResolveSuccIndexListDict, cceCodeInfoList, fillContent);
        devFunc->InitWrapInfo(initOffset, callList, fillContent, calleeHashIndexDict, cceCodeInfoList);

        MarkResolveBitmaps(devFunc, initOffset, fillContent);

        devFunc->InitIncastOutcast(initOffset, incastList, outcastList, tensorList, incastOpAttrDict, outcastOpAttrDict,
                                   param, rawName, fillContent, opIdxToHubOpIdx_);
    }
};

void EncodeDevAscendFunction(Function* dyndev, const EncodeDevAscendFunctionParam& param, uint64_t& offset,
                             DevAscendFunction* base)
{
    EncodeDevAscendFunctionInfo encodeInfo(dyndev, param.calleeHashIndexDict, param.cceCodeInfoList,
                                           param.expressionTable, param.devRoot);

    if (base == nullptr) {
        DevAscendFunction devfunc;
        encodeInfo.Init(&devfunc, param, false);
        offset = devfunc.GetSize();
    } else {
        encodeInfo.Init(base, param, true);
        offset = base->GetSize();
    }
}

void DevAscendProgram::InitSymbolTable(uintdevptr_t& initOffset, SymbolicSymbolTable* symbolTableInput,
                                       bool fillContent)
{
    symbolTable.HostInitDataSizeOffset(initOffset, symbolTableInput->GetSymbolTable().size());

    symbolTableNameList.HostInitDataSizeOffset(initOffset, 0);
    uint64_t offset = 0;
    for (size_t index = 0; index < symbolTableInput->GetSymbolTable().size(); index++) {
        std::string name = symbolTableInput->GetSymbolTable()[index];
        ONFILLCONTENT { symbolTable[index].index = index; };
        ONFILLCONTENT
        {
            symbolTable[index].name.HostAssignRangeOffsetSize(symbolTableNameList, offset, name.size());
            DevMemcpyS(symbolTable[index].name.Data(), symbolTable[index].name.size(), name.c_str(), name.size());
        }
        offset += AlignUp(name.size(), sizeof(uint64_t));
    }
    symbolTableNameList.HostInitDataSizeOffset(initOffset, offset);
}
void DevAscendProgram::InitExpressionTableBinary(
    uintdevptr_t& initOffset, const std::vector<std::vector<uint8_t>>& expressionTableBinaryListInput, bool fillContent)
{
    expressionTableOffsetList.HostInitDataSizeOffset(initOffset, expressionTableBinaryListInput.size());
    preGuardPage.HostInitDataSizeOffset(initOffset, PAGE_SIZE);
    ONFILLCONTENT { memset_s(preGuardPage.Data(), PAGE_SIZE, 0, PAGE_SIZE); }

    expressionTableBinary.HostInitDataSizeOffset(initOffset, 0);
    uint64_t offset = 0;
    for (size_t i = 0; i < expressionTableBinaryListInput.size(); i++) {
        ONFILLCONTENT { expressionTableOffsetList[i] = offset; }
        ONFILLCONTENT
        {
            DevMemcpyS(expressionTableBinary.Data() + offset, expressionTableBinaryListInput[i].size(),
                       expressionTableBinaryListInput[i].data(), expressionTableBinaryListInput[i].size());
        }
        offset += expressionTableBinaryListInput[i].size();
    }
    expressionTableBinary.HostInitDataSizeOffset(initOffset, offset);
}
void DevAscendProgram::InitControlFlowBinary(uintdevptr_t& initOffset,
                                             const std::vector<uint8_t>& hostControlFlowBinaryInput,
                                             const std::vector<uint8_t>& devControlFlowBinaryInput, bool fillContent)
{
    uint64_t alignedHostControlFlowBinaryInputSize = AlignUp(hostControlFlowBinaryInput.size(), sizeof(uint64_t));
    hostControlFlowBinary.HostInitDataSizeOffset(initOffset, alignedHostControlFlowBinaryInputSize);
    ONFILLCONTENT
    {
        DevMemcpyS(hostControlFlowBinary.Data(), hostControlFlowBinaryInput.size(), hostControlFlowBinaryInput.data(),
                   hostControlFlowBinaryInput.size());
    }

    uint64_t alignedDevControlFlowBinaryInputSize = AlignUp(devControlFlowBinaryInput.size(), sizeof(uint64_t));
    devControlFlowBinary.HostInitDataSizeOffset(initOffset, alignedDevControlFlowBinaryInputSize);
    ONFILLCONTENT
    {
        DevMemcpyS(devControlFlowBinary.Data(), devControlFlowBinaryInput.size(), devControlFlowBinaryInput.data(),
                   devControlFlowBinaryInput.size());
    }
}
void DevAscendProgram::InitDevEncodeList(uintdevptr_t& initOffset,
                                         const std::vector<std::vector<uint8_t>>& devEncodeListInput, bool fillContent)
{
    devEncodeList.HostInitDataSizeOffset(initOffset, devEncodeListInput.size());
    devEncodeDataList.HostInitDataSizeOffset(initOffset, 0);
    uint64_t offset = 0;
    for (size_t index = 0; index < devEncodeListInput.size(); index++) {
        uint64_t alignedDevEncodeListInputSize = AlignUp(devEncodeListInput[index].size(), sizeof(uint64_t));
        ONFILLCONTENT
        {
            devEncodeList[index].HostAssignRangeOffsetSize(devEncodeDataList, offset, alignedDevEncodeListInputSize);
        };
        ONFILLCONTENT
        {
            DevMemcpyS(devEncodeList[index].Data(), devEncodeList[index].size(), devEncodeListInput[index].data(),
                       devEncodeListInput[index].size());
        };
        offset += alignedDevEncodeListInputSize;
    }
    devEncodeDataList.HostInitDataSizeOffset(initOffset, offset);
}
void DevAscendProgram::InitCceCodeList(uintdevptr_t& initOffset, const std::vector<CceCodeInfo>& cceInfo,
                                       bool fillContent)
{
    cceCodeList.HostInitDataSizeOffset(initOffset, cceInfo.size());
    aicpuLeafCodeList.HostInitDataSizeOffset(initOffset, cceInfo.size());
    aicpuLeafCodeDataList.HostInitDataSizeOffset(initOffset, 0);
    size_t offset = 0;
    for (size_t index = 0; index < cceInfo.size(); index++) {
        ONFILLCONTENT
        {
            cceCodeList[index].coreType = cceInfo[index].coreType;
            cceCodeList[index].psgId = cceInfo[index].psgId;
            cceCodeList[index].funcHash = cceInfo[index].funcHash;
            cceCodeList[index].wrapVecId = cceInfo[index].wrapVecId;
            cceCodeList[index].mixResourceType = static_cast<uint8_t>(cceInfo[index].mixResourceType);
            auto dataLen = cceInfo[index].aicpuLeafCode.size();
            aicpuLeafCodeList[index].aicpuLeafCode.HostAssignRangeOffsetSize(aicpuLeafCodeDataList, offset, dataLen);
            (void)memcpy_s(aicpuLeafCodeList[index].aicpuLeafCode.Data(), sizeof(int32_t) * dataLen,
                           cceInfo[index].aicpuLeafCode.data(), sizeof(int32_t) * dataLen);
        };
        offset += cceInfo[index].aicpuLeafCode.size();
    }
    aicpuLeafCodeDataList.HostInitDataSizeOffset(initOffset, offset);
}

void DevAscendProgram::InitPrefetchInfoList(uintdevptr_t& initOffset, const std::vector<L2Info>& l2InfoList,
                                            bool fillContent)
{
    prefetchInfoList.HostInitDataSizeOffset(initOffset, l2InfoList.size());
    for (size_t index = 0; index < l2InfoList.size(); index++) {
        ONFILLCONTENT
        {
            DevMemcpyS(&prefetchInfoList[index], sizeof(PrefetchInfo), &l2InfoList[index], sizeof(L2Info));
        };
    }
    return;
}

void DevAscendProgram::InitDisableL2List(uintdevptr_t& initOffset, const std::vector<uint8_t>& disableL2,
                                         bool fillContent)
{
    disableL2List.HostInitDataSizeOffset(initOffset, disableL2.size());
    ONFILLCONTENT { (void)memcpy_s(disableL2List.Data(), disableL2.size(), disableL2.data(), disableL2.size()); };
    return;
}

void DevAscendProgram::InitStartArgsABIParamList(uintdevptr_t& initOffset,
                                                 const std::vector<int>& tStartArgsInputTensorSlotIndexList,
                                                 const std::vector<int>& tStartArgsOutputTensorSlotIndexList,
                                                 const std::vector<int>& tStartArgsInputSymbolIndexList,
                                                 const std::vector<int>& tAsembleSlotIndexList,
                                                 const std::vector<int>& tInplaceSlotIndexList, bool fillContent)
{
    this->startArgsInputTensorSlotIndexList.HostInitDataSizeOffset(initOffset,
                                                                   tStartArgsInputTensorSlotIndexList.size());
    this->startArgsOutputTensorSlotIndexList.HostInitDataSizeOffset(initOffset,
                                                                    tStartArgsOutputTensorSlotIndexList.size());
    this->startArgsInputSymbolIndexList.HostInitDataSizeOffset(initOffset, tStartArgsInputSymbolIndexList.size());
    this->assembleSlotIndexList.HostInitDataSizeOffset(initOffset, tAsembleSlotIndexList.size());
    this->outputInplaceSlotList.HostInitDataSizeOffset(initOffset, tInplaceSlotIndexList.size());

    ONFILLCONTENT
    {
        for (size_t index = 0; index < tStartArgsInputTensorSlotIndexList.size(); index++) {
            this->startArgsInputTensorSlotIndexList[index] = tStartArgsInputTensorSlotIndexList[index];
        }
        for (size_t index = 0; index < tStartArgsOutputTensorSlotIndexList.size(); index++) {
            this->startArgsOutputTensorSlotIndexList[index] = tStartArgsOutputTensorSlotIndexList[index];
        }
        for (size_t index = 0; index < tStartArgsInputSymbolIndexList.size(); index++) {
            this->startArgsInputSymbolIndexList[index] = tStartArgsInputSymbolIndexList[index];
        }
        for (size_t index = 0; index < tAsembleSlotIndexList.size(); index++) {
            this->assembleSlotIndexList[index] = tAsembleSlotIndexList[index];
        }
        for (size_t index = 0; index < tInplaceSlotIndexList.size(); index++) {
            this->outputInplaceSlotList[index] = tInplaceSlotIndexList[index];
        }
    }
}

static Opcode GetFullCoverProducerOpcodeForOutcast(const DevAscendFunction& devFunc,
                                                   const DevAscendFunctionOutcast& outcast)
{
    if (outcast.stitchPolicyFullCoverProducerList.size() == 0) {
        return Opcode::OP_UNKNOWN;
    }
    auto* fullCoverProducerList = &devFunc.At(outcast.stitchPolicyFullCoverProducerList, 0);
    for (size_t i = 0; i < outcast.stitchPolicyFullCoverProducerList.size(); i++) {
        if (fullCoverProducerList[i].opType != CellMatchOpType::READ) {
            return fullCoverProducerList[i].opCode;
        }
    }
    return Opcode::OP_UNKNOWN;
}

static void InitPartialUpdateCellMatch(const std::vector<const DevAscendFunctionOutcast*>& outcastList,
                                       const std::vector<Opcode>& outcastOpcodeList,
                                       DevCellMatchTableDesc* partialUpdateCellMatchTableDesc)
{
    std::vector<int> tensorShape;
    for (size_t index = 0; index < outcastList.size(); index++) {
        std::vector<int> outcastShape;
        for (int d = 0; d < outcastList[index]->dim; d++) {
            outcastShape.push_back(outcastList[index]->cellMatchTableDesc.GetCellShape(d) *
                                   outcastList[index]->cellMatchTableDesc.GetStrideShape(d));
            if (index > 0) {
                tensorShape[d] = std::max(tensorShape[d], outcastShape[d]);
            }
        }
        if (index == 0) {
            tensorShape = outcastShape;
        }
    }

    std::vector<int> cellShape;
    for (size_t d = 0; d < tensorShape.size(); d++) {
        int mergeDim = -1;
        // Primary merge: non-reshape outcasts only; reshape full-cover cellShape is tensor-level initial value.
        for (size_t index = 0; index < outcastList.size(); index++) {
            if (outcastOpcodeList[index] == Opcode::OP_RESHAPE) {
                continue;
            }
            int outcastDim = outcastList[index]->cellMatchTableDesc.GetCellShape(d);
            ASSERT(DevCommonErr::PARAM_CHECK_FAILED, outcastDim != -1)
                << "Invalid cell shape for outcastList[" << index << "], dimension: " << d
                << ", expected concrete value, got -1";
            mergeDim = mergeDim == -1 ? outcastDim : std::gcd(mergeDim, outcastDim);
        }
        // Fallback when all outcasts are reshape: gcd concrete dims from original cellShape, skip -1.
        if (mergeDim == -1) {
            for (size_t index = 0; index < outcastList.size(); index++) {
                int outcastDim = outcastList[index]->cellMatchTableDesc.GetCellShape(d);
                if (outcastDim == -1) {
                    continue;
                }
                mergeDim = mergeDim == -1 ? outcastDim : std::gcd(mergeDim, outcastDim);
            }
        }
        cellShape.push_back(mergeDim);
    }
    partialUpdateCellMatchTableDesc->SetCellShape(cellShape);

    std::vector<int> strideShape;
    for (size_t i = 0; i < tensorShape.size(); i++) {
        strideShape.push_back(cellShape[i] != 0 ? tensorShape[i] / cellShape[i] : 0);
    }
    partialUpdateCellMatchTableDesc->SetStrideShape(strideShape);
}
static void DumpFullCoverCellTableForNonPartialSlots(
    const std::vector<std::vector<uint8_t>>& devEncodeListInput,
    const std::unordered_map<Function*, int>& rootFuncKeyDict,
    const std::unordered_map<int, std::unordered_map<Function*, int>>& slotRootOutcastDict,
    const std::unordered_set<int>& partialSlotSet, ::npu::tile_fwk::topo_dump::SlotCellTableCsvWriter& slotCellTable)
{
    if (!slotCellTable.Enabled()) {
        return;
    }
    for (const auto& [slotIndex, rootOutcastMap] : slotRootOutcastDict) {
        if (partialSlotSet.count(slotIndex)) {
            continue;
        }
        for (const auto& [root, outcastIndex] : rootOutcastMap) {
            ASSERT(rootFuncKeyDict.count(root)) << "root: " << root << " not found in rootFuncKeyDict";
            int funcKey = rootFuncKeyDict.find(root)->second;
            DevAscendFunction* devFunc = reinterpret_cast<DevAscendFunction*>(
                const_cast<uint8_t*>(devEncodeListInput[funcKey].data()));
            const DevAscendFunctionOutcast& outcast = devFunc->GetOutcast(outcastIndex);
            slotCellTable.WriteFullCover(slotIndex, devFunc->rootHash, funcKey, outcast.cellMatchTableDesc);
        }
    }
}

static bool HasDynamicShape(const DevCellMatchTableDesc& desc)
{
    for (int d = 0; d < desc.GetDimensionSize(); d++) {
        if (desc.GetStrideShape(d) <= 0) {
            return true;
        }
    }
    return false;
}

static void FillPartialUpdateSlotContent(DevAscendProgram* prog, int slotIndex,
                                         const DevCellMatchTableDesc& partialUpdateCellMatchTableDesc,
                                         int totalCellMatchSize, size_t tableSize, bool hasDynamicShape,
                                         StitchCtrlBitMask stitchCtrlBitMask)
{
    auto& partialUpdate = prog->At(prog->updateList, slotIndex);
    partialUpdate.slotIndex = slotIndex;
    partialUpdate.stitchCtrlBitMask = stitchCtrlBitMask;
    partialUpdate.isPartial = true;
    partialUpdate.cellMatchTableDesc = partialUpdateCellMatchTableDesc;

    if (hasDynamicShape) {
        partialUpdate.cellMatchRuntimePartialUpdateTable.HostAssignDataSize(0, 0);
        return;
    }

    partialUpdate.cellMatchRuntimePartialUpdateTable.HostAssignRangeOffsetSize(
        prog->cellMatchRuntimePartialUpdateTableList, totalCellMatchSize, tableSize);
    auto tableData = partialUpdate.cellMatchRuntimePartialUpdateTable.Data();

    for (size_t j = 0; j < tableSize; j++) {
        tableData[j] = AICORE_TASK_INIT;
    }
}

static bool HasReadConsumerInOutcast(DevAscendFunction* devFunc, const DevAscendFunctionOutcast& outcast)
{
    auto* producerConsumerList = &devFunc->At(outcast.producerConsumerList, 0);
    for (size_t i = 0; i < outcast.producerConsumerList.size(); i++) {
        if (producerConsumerList[i].opType == CellMatchOpType::READ)
            return true;
    }
    auto* fullCoverProducerList = &devFunc->At(outcast.stitchPolicyFullCoverProducerList, 0);
    for (size_t i = 0; i < outcast.stitchPolicyFullCoverProducerList.size(); i++) {
        if (fullCoverProducerList[i].opType == CellMatchOpType::READ)
            return true;
    }
    return false;
}

static uint32_t CalculateSlotMaxReadCount(
    int slotIndex, const std::unordered_map<int, std::unordered_map<Function*, int>>& slotRootIncastDict,
    const std::unordered_map<int, std::unordered_map<Function*, int>>& slotRootOutcastDict,
    const std::unordered_map<Function*, int>& rootFuncKeyDict,
    const std::vector<std::vector<uint8_t>>& devEncodeListInput)
{
    uint32_t incastCount = 0;
    if (slotRootIncastDict.count(slotIndex) != 0) {
        incastCount = static_cast<uint32_t>(slotRootIncastDict.at(slotIndex).size());
    }

    uint32_t outcastConsumerCount = 0;
    for (auto& [root, outcastIndex] : slotRootOutcastDict.find(slotIndex)->second) {
        int funcKey = rootFuncKeyDict.find(root)->second;
        DevAscendFunction* devFunc = reinterpret_cast<DevAscendFunction*>(
            const_cast<uint8_t*>(devEncodeListInput[funcKey].data()));
        auto& outcast = devFunc->GetOutcast(outcastIndex);
        if (HasReadConsumerInOutcast(devFunc, outcast)) {
            MACHINE_LOGD("slot %d outcast %d have read consumer op.", slotIndex, outcastIndex);
            outcastConsumerCount++;
        }
    }

    return std::min(incastCount + outcastConsumerCount, static_cast<uint32_t>(CELL_MATCH_MAX_READ_COUNT));
}

static bool HasAtomicWriteInOutcast(DevAscendFunction* devFunc, const DevAscendFunctionOutcast& outcast)
{
    auto* producerConsumerList = &devFunc->At(outcast.producerConsumerList, 0);
    for (size_t i = 0; i < outcast.producerConsumerList.size(); i++) {
        if (producerConsumerList[i].opType == CellMatchOpType::ATOMIC_WRITE)
            return true;
    }
    auto* fullCoverProducerList = &devFunc->At(outcast.stitchPolicyFullCoverProducerList, 0);
    for (size_t i = 0; i < outcast.stitchPolicyFullCoverProducerList.size(); i++) {
        if (fullCoverProducerList[i].opType == CellMatchOpType::ATOMIC_WRITE)
            return true;
    }
    return false;
}

static void FillEmptyPartialUpdateStub(DevAscendProgram* prog, int slotIndex)
{
    auto& partialUpdate = prog->At(prog->updateList, slotIndex);
    partialUpdate.slotIndex = slotIndex;
    partialUpdate.stitchCtrlBitMask = STITCH_CTRL_NONE;
    partialUpdate.isPartial = true;
    partialUpdate.cellMatchTableDesc.SetCacheOpMaxCount({CELL_MATCH_MAX_NORMAL_WRITE_COUNT, 0, 0});
    partialUpdate.cellMatchRuntimePartialUpdateTable.HostAssignDataSize(0, 0);
}

// Same counting as old MarkOutputTensorStitch: unique rootFuncKey over incast ∪ outcast.
static size_t CountSlotRootFuncNum(
    int slotIndex, const std::unordered_map<Function*, int>& rootFuncKeyDict,
    const std::unordered_map<int, std::unordered_map<Function*, int>>& slotRootIncastDict,
    const std::unordered_map<int, std::unordered_map<Function*, int>>& slotRootOutcastDict)
{
    std::unordered_set<int> funcKeys;
    auto insertKeys = [&](const std::unordered_map<int, std::unordered_map<Function*, int>>& slotRootDict) {
        auto slotIt = slotRootDict.find(slotIndex);
        if (slotIt == slotRootDict.end()) {
            return;
        }
        for (const auto& rootEntry : slotIt->second) {
            auto rootIt = rootFuncKeyDict.find(rootEntry.first);
            if (rootIt != rootFuncKeyDict.end()) {
                funcKeys.insert(rootIt->second);
            }
        }
    };
    insertKeys(slotRootIncastDict);
    insertKeys(slotRootOutcastDict);
    return funcKeys.size();
}

static bool HasMultiIterNoOverlap(Function* root, const LogicalTensorPtr& outTensor)
{
    if (root == nullptr || outTensor == nullptr) {
        return false;
    }
    auto* attr = RebuildableAttributeManager::GetInstance().GetAttr<RebuildableMultiIterNoOverlap>(root);
    return attr != nullptr && attr->Has(outTensor->GetRawMagic());
}

static uint8_t CalculateStitchCtrlBitMaskForSlot(
    int slotIndex, const std::vector<std::vector<uint8_t>>& devEncodeListInput,
    const std::unordered_map<Function*, int>& rootFuncKeyDict,
    const std::unordered_map<int, std::unordered_map<Function*, int>>& slotRootIncastDict,
    const std::unordered_map<int, std::unordered_map<Function*, int>>& slotRootOutcastDict,
    uint32_t* outMaxReadCount = nullptr, bool* outHasAtomicWrite = nullptr)
{
    bool hasWaw = false;
    bool hasAtomicWrite = false;

    auto outcastIt = slotRootOutcastDict.find(slotIndex);
    if (outcastIt == slotRootOutcastDict.end()) {
        return STITCH_CTRL_NONE;
    }

    const size_t rootFuncNum = CountSlotRootFuncNum(slotIndex, rootFuncKeyDict, slotRootIncastDict,
                                                    slotRootOutcastDict);

    for (auto& [root, outcastIndex] : outcastIt->second) {
        auto rootIt = rootFuncKeyDict.find(root);
        if (rootIt == rootFuncKeyDict.end()) {
            continue;
        }
        const auto& outTensor = root->GetOutcast()[outcastIndex];

        if (outTensor != nullptr && outTensor->HasAttr("NORMAL") &&
            !(rootFuncNum < 2 && HasMultiIterNoOverlap(root, outTensor))) {
            hasWaw = true;
        }
        if (!hasAtomicWrite) {
            int funcKey = rootIt->second;
            DevAscendFunction* devFunc = reinterpret_cast<DevAscendFunction*>(
                const_cast<uint8_t*>(devEncodeListInput[funcKey].data()));
            auto& outcast = devFunc->GetOutcast(outcastIndex);
            hasAtomicWrite = HasAtomicWriteInOutcast(devFunc, outcast);
        }
    }
    if (outHasAtomicWrite != nullptr) {
        *outHasAtomicWrite = hasAtomicWrite;
    }

    const uint32_t maxReadCount = CalculateSlotMaxReadCount(slotIndex, slotRootIncastDict, slotRootOutcastDict,
                                                            rootFuncKeyDict, devEncodeListInput);
    if (outMaxReadCount != nullptr) {
        *outMaxReadCount = maxReadCount;
    }

    uint8_t stitchCtrlBitMask = STITCH_CTRL_NONE;
    if (maxReadCount > 0) {
        stitchCtrlBitMask |= static_cast<StitchCtrlBitMask>(STITCH_CTRL_WAR | STITCH_CTRL_RAW);
    }
    if (hasWaw) {
        stitchCtrlBitMask |= STITCH_CTRL_WAW;
    }
    return stitchCtrlBitMask;
}

std::unordered_map<int, SlotMaskEntry> BuildStitchUpdateSlotMaskMap(
    const std::shared_ptr<DyndevFunctionAttribute>& dyndevAttr)
{
    std::unordered_map<int, SlotMaskEntry> stitchUpdateSlotMaskMap;
    const auto& inoutLink = dyndevAttr->inoutLink;
    std::unordered_set<int> partialSlotSet(inoutLink.partialUpdateSlotIdexList.begin(),
                                           inoutLink.partialUpdateSlotIdexList.end());
    for (int slotIndex = 0; slotIndex < inoutLink.totalSlot; slotIndex++) {
        uint32_t maxReadCount = 0;
        bool hasAtomicWrite = false;
        uint8_t mask = CalculateStitchCtrlBitMaskForSlot(
            slotIndex, dyndevAttr->devEncodeList, dyndevAttr->rootFuncKeyDict, dyndevAttr->slotRootIncastDict,
            dyndevAttr->slotRootOutcastDict, &maxReadCount, &hasAtomicWrite);
        if (mask == STITCH_CTRL_NONE) {
            continue;
        }
        stitchUpdateSlotMaskMap[slotIndex] = {mask, partialSlotSet.count(slotIndex) != 0, maxReadCount, hasAtomicWrite};
    }
    return stitchUpdateSlotMaskMap;
}

void DevAscendFunction::RefilterStitchUpdateSlotLists(
    const std::unordered_map<int, SlotMaskEntry>& stitchUpdateSlotMaskMap)
{
    auto filterWar = [&](int slotIdx) -> bool {
        auto it = stitchUpdateSlotMaskMap.find(slotIdx);
        return it != stitchUpdateSlotMaskMap.end() && (it->second.mask & STITCH_CTRL_WAR);
    };
    auto filterRawPartial = [&](int slotIdx) -> bool {
        auto it = stitchUpdateSlotMaskMap.find(slotIdx);
        return it != stitchUpdateSlotMaskMap.end() && it->second.isPartial && (it->second.mask & STITCH_CTRL_RAW);
    };
    auto filterRawWawPartial = [&](int slotIdx) -> bool {
        auto it = stitchUpdateSlotMaskMap.find(slotIdx);
        return it != stitchUpdateSlotMaskMap.end() && it->second.isPartial &&
               (it->second.mask & (STITCH_CTRL_RAW | STITCH_CTRL_WAW));
    };

    auto filterFillContent = [&](auto& list, auto&& filter) {
        size_t writeIdx = 0;
        for (size_t readIdx = 0; readIdx < list.size(); readIdx++) {
            auto& entry = At(list, static_cast<int>(readIdx));
            if (filter(entry.slotIdx)) {
                At(list, static_cast<int>(writeIdx)) = entry;
                writeIdx++;
            }
        }
        list.AssignOffsetSize(list.Offset(0), writeIdx);
    };

    filterFillContent(stitchIncastSlotList, filterWar);
    filterFillContent(updateIncastSlotList, filterRawPartial);
    filterFillContent(stitchOutCastSlotList, filterRawWawPartial);
}

static void InitSingleFullCoverSlot(DevAscendProgram* prog, int slotIndex,
                                    const std::unordered_map<int, SlotMaskEntry>* stitchUpdateSlotMaskMap,
                                    bool fillContent)
{
    auto it = stitchUpdateSlotMaskMap->find(slotIndex);
    uint8_t stitchCtrlBitMask = (it != stitchUpdateSlotMaskMap->end()) ? it->second.mask : STITCH_CTRL_NONE;
    if (stitchCtrlBitMask == STITCH_CTRL_NONE) {
        return;
    }

    if (fillContent) {
        auto& update = prog->At(prog->updateList, slotIndex);
        update.slotIndex = slotIndex;
        update.stitchCtrlBitMask = stitchCtrlBitMask;
        update.cellMatchRuntimePartialUpdateTable.HostAssignDataSize(0, 0);
    }
}

static int InitSinglePartialUpdateSlot(
    DevAscendProgram* prog, int slotIndex, const std::vector<std::vector<uint8_t>>& devEncodeListInput,
    const std::unordered_map<Function*, int>& rootFuncKeyDict,
    const std::unordered_map<int, std::unordered_map<Function*, int>>& slotRootOutcastDict,
    const std::unordered_map<int, SlotMaskEntry>* stitchUpdateSlotMaskMap, int totalCellMatchSize, bool fillContent,
    topo_dump::SlotCellTableCsvWriter* slotCellTable)
{
    std::vector<const DevAscendFunctionOutcast*> outcastList;
    std::vector<Opcode> outcastOpcodeList;

    ASSERT(DevCommonErr::PARAM_CHECK_FAILED, slotRootOutcastDict.count(slotIndex))
        << "slotIndex: " << slotIndex << " not found in slotRootOutcastDict";
    const auto& rootOutcastMap = slotRootOutcastDict.find(slotIndex)->second;
    for (auto& [root, outcastIndex] : rootOutcastMap) {
        ASSERT(DevCommonErr::PARAM_CHECK_FAILED, rootFuncKeyDict.count(root))
            << "root: " << root << " not found in rootFuncKeyDict";
        int funcKey = rootFuncKeyDict.find(root)->second;
        DevAscendFunction* devFunc = reinterpret_cast<DevAscendFunction*>(
            const_cast<uint8_t*>(devEncodeListInput[funcKey].data()));
        auto& outcast = devFunc->GetOutcast(outcastIndex);
        outcastList.push_back(&outcast);
        outcastOpcodeList.push_back(GetFullCoverProducerOpcodeForOutcast(*devFunc, outcast));
    }

    auto maskIt = stitchUpdateSlotMaskMap->find(slotIndex);
    uint8_t stitchCtrlBitMask = (maskIt != stitchUpdateSlotMaskMap->end()) ? maskIt->second.mask : STITCH_CTRL_NONE;
    if (stitchCtrlBitMask == STITCH_CTRL_NONE) {
        if (fillContent) {
            FillEmptyPartialUpdateStub(prog, slotIndex);
        }
        return 0;
    }
    uint32_t maxReadCount = maskIt->second.maxReadCount;
    bool hasAtomicWrite = maskIt->second.hasAtomicWrite;

    DevCellMatchTableDesc partialUpdateCellMatchTableDesc;
    InitPartialUpdateCellMatch(outcastList, outcastOpcodeList, &partialUpdateCellMatchTableDesc);

    partialUpdateCellMatchTableDesc.SetCacheOpMaxCount({CELL_MATCH_MAX_NORMAL_WRITE_COUNT,
                                                        hasAtomicWrite ? CELL_MATCH_MAX_ATOMIC_WRITE_COUNT : 0U,
                                                        outcastList.size() > 1 ? maxReadCount : 0u});

    size_t cellNum = partialUpdateCellMatchTableDesc.GetStride(0);
    bool hasDynamicShape = HasDynamicShape(partialUpdateCellMatchTableDesc);
    size_t actualTableSize = hasDynamicShape ? 0 : cellNum * partialUpdateCellMatchTableDesc.cellUint64Size;

    if (fillContent) {
        FillPartialUpdateSlotContent(prog, slotIndex, partialUpdateCellMatchTableDesc, totalCellMatchSize,
                                     actualTableSize, hasDynamicShape, stitchCtrlBitMask);
        if (slotCellTable != nullptr && !hasDynamicShape) {
            slotCellTable->WritePartial(slotIndex, partialUpdateCellMatchTableDesc, outcastList.size());
        }
    }
    return static_cast<int>(actualTableSize);
}

void DevAscendProgram::InitUpdateSlot(
    uintdevptr_t& initOffset, const std::vector<std::vector<uint8_t>>& devEncodeListInput,
    const std::unordered_map<Function*, int>& rootFuncKeyDict,
    const std::unordered_map<int, std::unordered_map<Function*, int>>& slotRootOutcastDict,
    const std::vector<int>& tPartialUpdateSlotIndexList, bool fillContent,
    const std::unordered_map<int, SlotMaskEntry>* stitchUpdateSlotMaskMap)
{
    this->updateList.HostInitDataSizeOffset(initOffset, slotSize);
    if (fillContent) {
        for (size_t i = 0; i < static_cast<size_t>(slotSize); ++i) {
            this->At(this->updateList, i).slotIndex = -1;
        }
    }

    this->cellMatchRuntimePartialUpdateTableList.HostInitDataSizeOffset(initOffset, 0);
    int totalCellMatchSize = 0;
    topo_dump::SlotCellTableCsvWriter slotCellTable(fillContent);

    std::unordered_set<int> partialSlotSet(tPartialUpdateSlotIndexList.begin(), tPartialUpdateSlotIndexList.end());

    if (stitchUpdateSlotMaskMap == nullptr) {
        return;
    }

    for (uint32_t slotIndex = 0; slotIndex < slotSize; slotIndex++) {
        if (partialSlotSet.count(static_cast<int>(slotIndex))) {
            totalCellMatchSize += InitSinglePartialUpdateSlot(
                this, static_cast<int>(slotIndex), devEncodeListInput, rootFuncKeyDict, slotRootOutcastDict,
                stitchUpdateSlotMaskMap, totalCellMatchSize, fillContent, &slotCellTable);
        } else {
            InitSingleFullCoverSlot(this, static_cast<int>(slotIndex), stitchUpdateSlotMaskMap, fillContent);
        }
    }

    DumpFullCoverCellTableForNonPartialSlots(devEncodeListInput, rootFuncKeyDict, slotRootOutcastDict, partialSlotSet,
                                             slotCellTable);
    totalCellMatchSize = AlignUp(totalCellMatchSize,
                                 sizeof(uint64_t) * FRIENDLY_CACHE_ALIGN_U64_SIZE / sizeof(uint64_t));
    this->cellMatchRuntimePartialUpdateTableList.HostInitDataSizeOffset(initOffset, totalCellMatchSize);
}

void DevAscendProgram::InitControlFlowCache(uintdevptr_t& initOffset,
                                            const std::shared_ptr<DyndevFunctionAttribute>& dyndevAttr,
                                            bool fillContent, uint32_t outcastCacheDepthFallback)
{
    (void)fillContent;
    ctrlFlowCacheSize = DEFAULT_STITCH_CFGCACHE_SIZE;
    const uint64_t slottedOutcastBlockCount = GetCtrlFlowCacheSlottedOutcastBlockCount(dyndevAttr->inoutLink.totalSlot,
                                                                                       outcastCacheDepthFallback);
    controlFlowCache.Init(dyndevAttr.get(), ctrlFlowCacheSize, memBudget.tensor.runtimeOutcastPoolSize, initOffset,
                          slottedOutcastBlockCount);
}
struct EncodeDevAscendProgramInfo {
    Function* func;
    std::shared_ptr<DyndevFunctionAttribute> dyndevAttr;
    uint64_t getTensorDataCount = 0;
    uint64_t getInputDataCount = 0;

    explicit EncodeDevAscendProgramInfo(Function* tfunc) : func(tfunc)
    {
        ASSERT(DevCommonErr::PARAM_CHECK_FAILED, func->GetDyndevAttribute() != nullptr)
            << "DyndevAttribute is null for function: " << func;
        dyndevAttr = func->GetDyndevAttribute();
    }

    bool GetEnableVFFusion()
    {
        bool enableVFFusion = Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510;
        enableVFFusion = enableVFFusion && config::GetPassGlobalConfig(KEY_ENABLE_VF, false);
        return (config::GetRuntimeOption<int64_t>(CFG_VALID_SHAPE_OPTIMIZE) == 1 || enableVFFusion);
    };

    bool inline HasAicpuTask()
    {
        for (size_t i = 0; i < dyndevAttr->cceCodeInfo.size(); i++) {
            if (dyndevAttr->cceCodeInfo[i].coreType == static_cast<uint32_t>(CoreType::AICPU)) {
                return true;
            }
        }
        return false;
    }

    void Init(DevAscendProgram* devProg, bool fillContent,
              const std::unordered_map<int, SlotMaskEntry>* stitchUpdateSlotMaskMap)
    {
        uintdevptr_t initOffset = reinterpret_cast<uintdevptr_t>(devProg->data);
        devProg->devArgs.archInfo = static_cast<ArchInfo>(Platform::Instance().GetSoc().GetNPUArch());
        devProg->devArgs.enableVFFusion = GetEnableVFFusion();
        devProg->devArgs.hasAicpuTask = HasAicpuTask();
        auto* requiresSimtAttr = RebuildableAttributeManager::GetInstance().GetAttr<RebuildableRequiresSimt>(func);
        devProg->requiresSimt = requiresSimtAttr->Get();
        devProg->hashKey = func->ComputeHash().GetHash();
        ASSERT(DevCommonErr::PARAM_CHECK_FAILED, devProg->hashKey != 0)
            << "Function hash must be non-zero: AOT code pool uses 0 as empty-entry sentinel";
        devProg->slotSize = dyndevAttr->inoutLink.totalSlot;
        devProg->assembleSlotSize = dyndevAttr->inoutLink.assembleSlotIndexList.size();
        devProg->InitSymbolTable(initOffset, &dyndevAttr->symbolTable, fillContent);
        devProg->InitExpressionTableBinary(initOffset, dyndevAttr->expressionTableBinaryList, fillContent);
        uint64_t expressionTableSize = 0;
        for (auto& [root, exprTable] : dyndevAttr->exprTableDictGroup.devRootCoaDict) {
            (void)root;
            expressionTableSize = std::max(expressionTableSize, (uint64_t)exprTable.GetPrimaryExpressionSize());
        }
        devProg->expressionTableSize = expressionTableSize;
        devProg->InitControlFlowBinary(initOffset, dyndevAttr->hostControlFlowBinary, dyndevAttr->devControlFlowBinary,
                                       fillContent);
        devProg->InitDevEncodeList(initOffset, dyndevAttr->devEncodeList, fillContent);
        devProg->InitCceCodeList(initOffset, dyndevAttr->cceCodeInfo, fillContent);
        devProg->InitStartArgsABIParamList(
            initOffset, dyndevAttr->inoutLink.inputSlotIndexList, dyndevAttr->inoutLink.outputSlotIndexList,
            dyndevAttr->startArgsInputSymbolIndexList, dyndevAttr->inoutLink.assembleSlotIndexList,
            dyndevAttr->inoutLink.inplaceSlotIndexList, fillContent);
        devProg->InitUpdateSlot(initOffset, dyndevAttr->devEncodeList, dyndevAttr->rootFuncKeyDict,
                                dyndevAttr->slotRootOutcastDict, dyndevAttr->inoutLink.partialUpdateSlotIdexList,
                                fillContent, stitchUpdateSlotMaskMap);
        devProg->InitPrefetchInfoList(initOffset, dyndevAttr->l2InfoList, fillContent);
        devProg->InitDisableL2List(initOffset, dyndevAttr->disableL2List, fillContent);
        devProg->valueDependInputIndices = dyndevAttr->valueDependInputIndices;
        devProg->dataSize = initOffset - reinterpret_cast<uintdevptr_t>(devProg->data);
    }
};

static WorkspaceDesc CollectWorkspaceDescForSizeOnlyEncode(Function* func, EncodeDevAscendProgramInfo& encodeInfo)
{
    return CollectWorkspaceDescFromHostEncodeList(func, *encodeInfo.dyndevAttr,
                                                  encodeInfo.dyndevAttr->constructAssembleNeedAllocRuntimeSlots);
}

void EncodeDevAscendProgramSizeOnly(uint64_t& offset, EncodeDevAscendProgramInfo& encodeInfo,
                                    const std::unordered_map<int, SlotMaskEntry>* stitchUpdateSlotMaskMap)
{
    DevAscendProgram devfunc;
    devfunc.SetParallelism(config::GetDeviceSchedParallelism());
    encodeInfo.Init(&devfunc, false, stitchUpdateSlotMaskMap);

    WorkspaceDesc wsDesc = CollectWorkspaceDescForSizeOnlyEncode(encodeInfo.func, encodeInfo);
    RuntimeWorkspaceConfig runtimeCfg = LoadRuntimeWorkspaceConfig(wsDesc.maxUnrollTimes);
    runtimeCfg.aicoreSpilled = wsDesc.maxLeafPerCoreSpilledMem * wsDesc.platform.aicoreCount;
    runtimeCfg.debugTotal = DumpTensorWorkspace() + LeafDumpWorkspace();
    runtimeCfg.parallelism = devfunc.GetParallelism();
    runtimeCfg.workspaceStitchMin = TensorWorkspaceBytesAtMinimumStitchDepth(
        wsDesc, runtimeCfg.parallelism, runtimeCfg.aicoreSpilled, runtimeCfg.debugTotal);

    StitchDepthConfig depthConfig = ResolveStitchDepthConfig(wsDesc, runtimeCfg);
    ApplyStitchDepthConfig(&devfunc, wsDesc, depthConfig, encodeInfo.dyndevAttr->inoutLink.totalSlot);

    uintdevptr_t cacheInitOffset = reinterpret_cast<uintdevptr_t>(devfunc.data) + devfunc.dataSize;
    devfunc.InitControlFlowCache(cacheInitOffset, encodeInfo.dyndevAttr, false, depthConfig.outcastCacheDepth);
    devfunc.dataSize = cacheInitOffset - reinterpret_cast<uintdevptr_t>(devfunc.data);
    ASSERT(DevCommonErr::PARAM_CHECK_FAILED, devfunc.GetSize() == sizeof(devfunc) + devfunc.dataSize)
        << "devProg->GetSize() does not match expected size, expected: " << sizeof(devfunc) + devfunc.dataSize
        << ", got: " << devfunc.GetSize();
    offset = devfunc.GetSize();
}

static WorkspaceDesc CollectWorkspaceDescForEncode(Function* func, DevAscendProgram* base)
{
    return CollectWorkspaceDesc(func, *base, func->GetDyndevAttribute()->constructAssembleNeedAllocRuntimeSlots);
}

static RuntimeWorkspaceConfig BuildFullEncodeRuntimeWorkspaceConfig(DevAscendProgram* base, const WorkspaceDesc& wsDesc)
{
    RuntimeWorkspaceConfig runtimeCfg = LoadRuntimeWorkspaceConfig(wsDesc.maxUnrollTimes);
    runtimeCfg.aicoreSpilled = wsDesc.maxLeafPerCoreSpilledMem * wsDesc.platform.aicoreCount;
    runtimeCfg.debugTotal = base->memBudget.debug.dumpTensor + base->memBudget.debug.leafDump;
    runtimeCfg.parallelism = base->GetParallelism();
    runtimeCfg.workspaceStitchMin = TensorWorkspaceBytesAtMinimumStitchDepth(
        wsDesc, runtimeCfg.parallelism, runtimeCfg.aicoreSpilled, runtimeCfg.debugTotal);
    return runtimeCfg;
}

static void ValidateCtrlFlowCacheBackupCapacity(const EncodeDevAscendProgramInfo& encodeInfo,
                                                const WorkspaceDesc& wsDesc, const StitchDepthConfig& depthConfig)
{
    const uint64_t requiredSlotBlocks = wsDesc.devTaskBoundaryOutcastNum + wsDesc.devTaskInnerTemporalOutcastNum;
    const uint64_t ctrlFlowSlotBackupCount = requiredSlotBlocks != 0 ?
                                                 requiredSlotBlocks :
                                                 EstimateCtrlFlowCacheSlottedBlockCount(
                                                     encodeInfo.dyndevAttr->inoutLink.totalSlot,
                                                     depthConfig.outcastCacheDepth > 0 ?
                                                         depthConfig.outcastCacheDepth :
                                                         (depthConfig.stitchMaxFunctionNum > 0 ?
                                                              depthConfig.stitchMaxFunctionNum :
                                                              static_cast<uint32_t>(MAX_STITCH_FUNC_NUM)));
    ASSERT(DevCommonErr::PARAM_CHECK_FAILED, ctrlFlowSlotBackupCount >= requiredSlotBlocks)
        << "Control flow cache slot backup capacity is smaller than boundary outcast slot budget, backup="
        << ctrlFlowSlotBackupCount << ", boundary=" << wsDesc.devTaskBoundaryOutcastNum
        << ", assembleOutcast=" << wsDesc.devTaskInnerTemporalOutcastNum;
}

static void EncodeProgramMetadataWorkspace(DevAscendProgram* base)
{
    base->stitchFunctionsize = MAX_STITCH_LEAFFUNC_NUM;
    base->memBudget.metadata.dynamicCellMatch = 0;
    base->memBudget.metadata.general = CalcGeneralMetadataSlotWorkspace(base);
    base->memBudget.metadata.general += CalcGeneralMetadataSlabWorkspace(base);
    base->memBudget.metadata.stitchCacheSize = CalcStitchCacheSize(base);
    base->memBudget.metadata.general += base->memBudget.metadata.stitchCacheSize;
    const bool aicoreResolve = IsAicoreResolveEnabled() &&
                               config::GetRuntimeOption<int64_t>(CFG_RUN_MODE) != CFG_RUN_MODE_SIM;
    base->memBudget.metadata.stitchPool = CalcStitchWorkspace(*base, aicoreResolve);
}

static void FinalizeEncodedDevAscendProgram(Function* func, DevAscendProgram* base, uint64_t& offset,
                                            const WorkspaceDesc& wsDesc, const RuntimeWorkspaceConfig& runtimeCfg,
                                            const StitchDepthConfig& depthConfig)
{
    base->workspaceSize = base->memBudget.Total();
    offset = base->GetSize();
    LogWorkspaceEncodeSummary(1, *base, depthConfig, runtimeCfg, wsDesc);
    MACHINE_LOGD("StitchPool:%lu bytes, aicoreSpilled:%lu bytes.", base->memBudget.metadata.stitchPool,
                 base->memBudget.aicoreSpilled.Total());
    func->GetDyndevAttribute()->maxDynamicAssembleOutcastMem = wsDesc.maxDynamicAssembleOutcastMem;
    func->GetDyndevAttribute()->maxDynamicCellMatchTableMem = wsDesc.cellMatch.maxDynamicCellMatchTableMem;
    BuildDynamicCellMatchLaunchMeta(func, *base);
}

void EncodeDevAscendProgramFull(Function* func, DevAscendProgram* base, uint64_t& offset,
                                EncodeDevAscendProgramInfo& encodeInfo,
                                const std::unordered_map<int, SlotMaskEntry>* stitchUpdateSlotMaskMap)
{
    base->SetParallelism(config::GetDeviceSchedParallelism());
    MACHINE_LOGD("device sched parallelism is %u.", base->GetParallelism());
    encodeInfo.Init(base, true, stitchUpdateSlotMaskMap);

    base->memBudget.debug.dumpTensor = DumpTensorWorkspace();
    base->memBudget.debug.leafDump = LeafDumpWorkspace();

    WorkspaceDesc wsDesc = CollectWorkspaceDescForEncode(func, base);
    RuntimeWorkspaceConfig runtimeCfg = BuildFullEncodeRuntimeWorkspaceConfig(base, wsDesc);
    if (HasPreciseStitchFunctionNumPerPool(runtimeCfg.stitchFunctionNumPerPool)) {
        MACHINE_LOGW("[workspaceSize] Precise workspace mode is enabled via stitch_function_num_per_pool; "
                     "stitch_function_max_num is forced to %u, and stitch_function_max_num / max_workspace_kb "
                     "are ignored.",
                     static_cast<uint32_t>(MAX_STITCH_FUNC_NUM));
    } else {
        ValidateMaxWorkspaceOrThrow(runtimeCfg.maxWorkspaceBytes, runtimeCfg.workspaceStitchMin);
    }

    StitchDepthConfig depthConfig = ResolveStitchDepthConfig(wsDesc, runtimeCfg);
    ApplyStitchDepthConfig(base, wsDesc, depthConfig, encodeInfo.dyndevAttr->inoutLink.totalSlot);
    ValidateCtrlFlowCacheBackupCapacity(encodeInfo, wsDesc, depthConfig);

    RebuildableAttributeManager::GetInstance().ResetAttr<RebuildableWorkspaceDesc>(func, &wsDesc);
    base->devArgs.machineConfig = func->paramConfigs_.machineConfig_;

    EncodeProgramMetadataWorkspace(base);

    uintdevptr_t cacheInitOffset = reinterpret_cast<uintdevptr_t>(base->data) + base->dataSize;
    base->InitControlFlowCache(cacheInitOffset, encodeInfo.dyndevAttr, true, depthConfig.outcastCacheDepth);
    base->dataSize = cacheInitOffset - reinterpret_cast<uintdevptr_t>(base->data);
    ASSERT(DevCommonErr::PARAM_CHECK_FAILED, reinterpret_cast<uint8_t*>(base->controlFlowCache.cacheData.end()) ==
                                                 reinterpret_cast<uint8_t*>(cacheInitOffset))
        << "controlFlowCache.cacheData.end() does not match cacheInitOffset";
    ASSERT(DevCommonErr::PARAM_CHECK_FAILED, base->GetSize() == sizeof(*base) + base->dataSize)
        << "devProg->GetSize() does not match expected size, expected: " << sizeof(*base) + base->dataSize
        << ", got: " << base->GetSize();

    FinalizeEncodedDevAscendProgram(func, base, offset, wsDesc, runtimeCfg, depthConfig);
}

void EncodeDevAscendProgram(Function* func, uint64_t& offset, DevAscendProgram* base,
                            const std::unordered_map<int, SlotMaskEntry>* stitchUpdateSlotMaskMap)
{
    EncodeDevAscendProgramInfo encodeInfo(func);
    if (base == nullptr) {
        EncodeDevAscendProgramSizeOnly(offset, encodeInfo, stitchUpdateSlotMaskMap);
        return;
    }
    EncodeDevAscendProgramFull(func, base, offset, encodeInfo, stitchUpdateSlotMaskMap);
}

void DevControlFlowCache::Init(void* dyndevAttrPtr, uint64_t cacheSize, uint64_t runtimeOutcastPoolSize,
                               uint64_t& initOffset, uint64_t slottedOutcastBlockCount)
{
    DyndevFunctionAttribute* dyndevAttr = reinterpret_cast<DyndevFunctionAttribute*>(dyndevAttrPtr);
    initOffset = AlignUp(initOffset, alignof(DevTensorData));
    inputTensorDataList.HostInitDataSizeOffset(initOffset, dyndevAttr->startArgsInputTensorList.size());
    outputTensorDataList.HostInitDataSizeOffset(initOffset, dyndevAttr->startArgsOutputTensorList.size());
    for (uint32_t i = 0; i < SCH_DEVTASK_MAX_PARALLELISM; i++) {
        runtimeBackup.workspace.tensorAllocators[i].slottedOutcastsBlockList.HostInitDataSizeOffset(
            initOffset, slottedOutcastBlockCount);
    }

    runtimeBackup.slotContext.slotList.HostInitDataSizeOffset(initOffset, dyndevAttr->inoutLink.totalSlot);
    runtimeBackup.workspace.runtimeOutcastTensorPool.HostInitDataSizeOffset(initOffset, runtimeOutcastPoolSize);

    initOffset = AlignUp(initOffset, alignof(DynFuncHeader*));
    deviceTaskCacheList.HostInitDataSizeOffset(initOffset, DEFAULT_CACHE_DEVICE_TASK_NUM);
    cacheData.HostInitDataSizeOffset(initOffset, cacheSize);
    isRecording = false;
    isRecordingStopped = false;
    isActivated = false;
    deviceTaskCount = 0;
    deviceTaskSkippedCount = 0;
    cacheDataOffset = 0;
    workspaceAddr = 0;
    taskRelocTableOffset = 0;
    taskRelocTableCount = 0;
    dataSize = initOffset - reinterpret_cast<uintdevptr_t>(data);
}

} // namespace dynamic
} // namespace npu::tile_fwk
