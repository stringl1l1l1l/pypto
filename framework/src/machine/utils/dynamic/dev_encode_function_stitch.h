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
 * \file dev_encode_function_stitch.h
 * \brief
 */

#pragma once

#include "machine/utils/dynamic/dev_encode_types.h"
#include "machine/utils/dynamic/dev_encode_function.h"
#include "machine/utils/dynamic/dev_cell_match_mem_layout.h"
#include "tilefwk/aicpu_common.h"
#include "machine/utils/dynamic/dev_callop_attribute.h"
namespace npu::tile_fwk::dynamic {
using npu::tile_fwk::DevAscendFunctionDuppedStitchNode;
using npu::tile_fwk::DUPPED_STITCH_SIZE;
using DevAscendFunctionDuppedStitch = npu::tile_fwk::DevAscendFunctionDuppedStitchNode;
static_assert(sizeof(DevAscendFunctionDuppedStitch) == npu::tile_fwk::DUPPED_STITCH_NODE_U32_SIZE * sizeof(uint32_t),
              "Invalid size");
// Max cell-match table entry count (desc stride[0]); checked at encode and in host GetWorkspaceSize.
constexpr int64_t MAX_CELLMATCHSSTRIDE = 20000000;

struct DevAscendFunctionDuppedStitchList {
    DevAscendFunctionDuppedStitchList() = default;

    bool IsNull() const { return head_ == nullptr; }

    DevAscendFunctionDuppedStitch* const& Head() const { return head_; }
    DevAscendFunctionDuppedStitch*& Head() { return head_; }

    // Low performance, only used in debug
    void ForEach(const std::function<void(uint32_t id)>& callback) const
    {
        for (auto* p = head_; p != nullptr; p = p->Next()) {
            p->ForEach(callback);
        }
    }

    template <typename AllocFn>
    void PushBack(uint32_t taskId, AllocFn&& allocate)
    {
        if (head_ == nullptr || head_->Size() == DUPPED_STITCH_SIZE) {
            auto* newNode = allocate();
            DEV_VERBOSE_DEBUG("New node %p", newNode);
            newNode->InitWithNext(head_);
            head_ = newNode;
        }
        head_->SafePushBack(taskId);
    }

    template <typename T = uint32_t>
    static std::string DumpTask(T id)
    {
        std::ostringstream oss;
        if constexpr (std::is_same<T, uint64_t>::value) {
            oss << (id >> CELL_MATCH_META_TAGID_SHIFT32) << "!"; // devicetaskid
        }
        oss << FuncID(static_cast<uint32_t>(id)) << "!" << TaskID(static_cast<uint32_t>(id));
        return oss.str();
    }

    // 子链长度编码（节点地址 64 字节对齐 → 低 6 位恒 0，可复用）：
    // 低 6 位写"本节点之后的节点数 R"（子链大小 S = R + 1，R ≤ 63）。
    // 链头写 min(N-1,63)；其余节点 idx（从 1 起）写 min(lowbit(idx)-1, 63, N-1-idx)，
    // 即把链按 2 的幂（1,2,4,8,16,32 / 每 64 一轮）切成子链，末段截断到链尾。
    // 两遍线性遍历（第一遍统计 N，第二遍写 R），无递归、无重复 walk；
    // 叶子/链尾节点低 6 位天然为 0，不编码；任务数始终读 nodeSize，不参与编码。
    void EncodeStitchNodes()
    {
        uint32_t chainNodeCount = 0;
        for (auto* n = head_; n != nullptr; n = n->Next()) {
            chainNodeCount++;
        }
        if (chainNodeCount <= 1) {
            return;
        }
        uint32_t headSuccNodeCount = chainNodeCount - 1;
        if (headSuccNodeCount > npu::tile_fwk::DUPPED_STITCH_NODE_REMAIN_COUNT_MASK) {
            headSuccNodeCount = npu::tile_fwk::DUPPED_STITCH_NODE_REMAIN_COUNT_MASK;
        }
        head_->NextRaw() = reinterpret_cast<DevAscendFunctionDuppedStitch*>(
            reinterpret_cast<uint64_t>(head_->Next()) |
            (static_cast<uint64_t>(headSuccNodeCount) & npu::tile_fwk::DUPPED_STITCH_NODE_REMAIN_COUNT_MASK));
        uint32_t nodeIndex = 1;
        for (auto* n = head_->Next(); n != nullptr; n = n->Next(), nodeIndex++) {
            uint32_t subChainFullSize = nodeIndex & (~nodeIndex + 1u);
            uint32_t succNodeCount = subChainFullSize - 1;
            if (succNodeCount > npu::tile_fwk::DUPPED_STITCH_NODE_REMAIN_COUNT_MASK) {
                succNodeCount = npu::tile_fwk::DUPPED_STITCH_NODE_REMAIN_COUNT_MASK;
            }
            uint32_t nodesToChainEnd = chainNodeCount - 1 - nodeIndex;
            if (nodesToChainEnd < succNodeCount) {
                succNodeCount = nodesToChainEnd;
            }
            n->NextRaw() = reinterpret_cast<DevAscendFunctionDuppedStitch*>(
                reinterpret_cast<uint64_t>(n->Next()) |
                (static_cast<uint64_t>(succNodeCount) & npu::tile_fwk::DUPPED_STITCH_NODE_REMAIN_COUNT_MASK));
        }
    }

public:
    template <typename T = uint32_t>
    static std::string DumpTask(T* idx, int size)
    {
        std::ostringstream oss;
        oss << "{";
        oss << "size = " << size << " -> ";
        for (int i = 0; i < size; i++) {
            if (idx[i] != AICORE_TASK_INIT) {
                oss << Delim(i != 0, ",");
                oss << "[" << std::dec << i << "]=" << DumpTask<T>(idx[i]);
            }
        }
        oss << "}";
        return oss.str();
    }

    std::string Dump() const
    {
        std::ostringstream oss;

        uint32_t index = 0;
        oss << "[";
        for (auto p = head_; p != nullptr; p = p->Next()) {
            oss << Delim(p != head_, ";");
            for (uint32_t i = 0; i < p->Size(); i++) {
                oss << Delim(i != 0, ",");
                oss << "[" << index++ << "]=" << DumpTask(p->At(i));
            }
        }
        oss << "]";
        return oss.str();
    }

private:
    DevAscendFunctionDuppedStitch* head_{nullptr};
};
static_assert(sizeof(DevAscendFunctionDuppedStitchList) == sizeof(void*));

// Stitch / cell-match control bit mask (0 = no stitch).
// NORMAL→WAW; maxReadCount>0→WAR|RAW.
using StitchCtrlBitMask = uint8_t;
constexpr StitchCtrlBitMask STITCH_CTRL_NONE = 0;
constexpr StitchCtrlBitMask STITCH_CTRL_WAW = 1u << 0; // 写后写
constexpr StitchCtrlBitMask STITCH_CTRL_WAR = 1u << 1; // 写后读
constexpr StitchCtrlBitMask STITCH_CTRL_RAW = 1u << 2; // 读后写

struct DevAscendProgramUpdate {
    int slotIndex{-1};
    StitchCtrlBitMask stitchCtrlBitMask{STITCH_CTRL_NONE};
    bool isPartial{false};

    DevCellMatchTableDesc cellMatchTableDesc;
    DevRelocVector<uint64_t> cellMatchRuntimePartialUpdateTable;

    bool Empty() const { return cellMatchRuntimePartialUpdateTable.size() == 0; }
};

template <typename HandleType, typename... TyArgs>
static uint32_t CellMatch5Dimension(const DevCellMatchTableDesc& cellMatchTableDesc, uint64_t* rangeBegin,
                                    uint64_t* rangeEnd, TyArgs... args)
{
    uint32_t errCode = 0;
    int s0 = cellMatchTableDesc.GetStride(1), s1 = cellMatchTableDesc.GetStride(2);
    int s2 = cellMatchTableDesc.GetStride(3), s3 = cellMatchTableDesc.GetStride(4), s4 = 1;
    for (int d0 = 0 + rangeBegin[0] * s0, e0 = 0 + rangeEnd[0] * s0; d0 <= e0; d0 += s0) {
        for (int d1 = d0 + rangeBegin[1] * s1, e1 = d0 + rangeEnd[1] * s1; d1 <= e1; d1 += s1) {
            for (int d2 = d1 + rangeBegin[2] * s2, e2 = d1 + rangeEnd[2] * s2; d2 <= e2; d2 += s2) {
                for (int d3 = d2 + rangeBegin[3] * s3, e3 = d2 + rangeEnd[3] * s3; d3 <= e3; d3 += s3) {
                    for (int d4 = d3 + rangeBegin[4] * s4, e4 = d3 + rangeEnd[4] * s4; d4 <= e4; d4 += s4) {
                        errCode = HandleType::Process(d4, cellMatchTableDesc, args...);
                        if (errCode != 0) {
                            return errCode;
                        }
                    }
                }
            }
        }
    }
    return errCode;
}

template <typename HandleType, typename... TyArgs>
static uint32_t CellMatch4Dimension(const DevCellMatchTableDesc& cellMatchTableDesc, uint64_t* rangeBegin,
                                    uint64_t* rangeEnd, TyArgs... args)
{
    uint32_t errCode = 0;
    int s0 = cellMatchTableDesc.GetStride(1), s1 = cellMatchTableDesc.GetStride(2);
    int s2 = cellMatchTableDesc.GetStride(3), s3 = 1;
    for (int d0 = 0 + rangeBegin[0] * s0, e0 = 0 + rangeEnd[0] * s0; d0 <= e0; d0 += s0) {
        for (int d1 = d0 + rangeBegin[1] * s1, e1 = d0 + rangeEnd[1] * s1; d1 <= e1; d1 += s1) {
            for (int d2 = d1 + rangeBegin[2] * s2, e2 = d1 + rangeEnd[2] * s2; d2 <= e2; d2 += s2) {
                for (int d3 = d2 + rangeBegin[3] * s3, e3 = d2 + rangeEnd[3] * s3; d3 <= e3; d3 += s3) {
                    errCode = HandleType::Process(d3, cellMatchTableDesc, args...);
                    if (errCode != 0) {
                        return errCode;
                    }
                }
            }
        }
    }
    return errCode;
}

template <typename HandleType, typename... TyArgs>
static uint32_t CellMatchProcessByDim(int dims, const DevCellMatchTableDesc& cellMatchTableDesc, uint64_t* rangeBegin,
                                      uint64_t* rangeEnd, TyArgs... args)
{
    uint32_t errCode = 0;
    switch (dims) {
        case 1: {
            int s0 = 1;
            for (int d0 = 0 + rangeBegin[0] * s0, e0 = 0 + rangeEnd[0] * s0; d0 <= e0; d0 += s0) {
                errCode = HandleType::Process(d0, cellMatchTableDesc, args...);
                if (errCode != 0) {
                    return errCode;
                }
            }
        } break;
        case DEV_SHAPE_DIM_NUM_2: {
            int s0 = cellMatchTableDesc.GetStride(1), s1 = 1;
            for (int d0 = 0 + rangeBegin[0] * s0, e0 = 0 + rangeEnd[0] * s0; d0 <= e0; d0 += s0)
                for (int d1 = d0 + rangeBegin[1] * s1, e1 = d0 + rangeEnd[1] * s1; d1 <= e1; d1 += s1) {
                    errCode = HandleType::Process(d1, cellMatchTableDesc, args...);
                    if (errCode != 0) {
                        return errCode;
                    }
                }
        } break;
        case DEV_SHAPE_DIM_NUM_3: {
            int s0 = cellMatchTableDesc.GetStride(1), s1 = cellMatchTableDesc.GetStride(2), s2 = 1;
            for (int d0 = 0 + rangeBegin[0] * s0, e0 = 0 + rangeEnd[0] * s0; d0 <= e0; d0 += s0)
                for (int d1 = d0 + rangeBegin[1] * s1, e1 = d0 + rangeEnd[1] * s1; d1 <= e1; d1 += s1)
                    for (int d2 = d1 + rangeBegin[2] * s2, e2 = d1 + rangeEnd[2] * s2; d2 <= e2; d2 += s2) {
                        errCode = HandleType::Process(d2, cellMatchTableDesc, args...);
                        if (errCode != 0) {
                            return errCode;
                        }
                    }
        } break;
        case DEV_SHAPE_DIM_NUM_4: {
            errCode = CellMatch4Dimension<HandleType>(cellMatchTableDesc, rangeBegin, rangeEnd, args...);
            if (errCode != 0) {
                return errCode;
            }
        } break;
        case DEV_SHAPE_DIM_NUM_5: {
            errCode = CellMatch5Dimension<HandleType>(cellMatchTableDesc, rangeBegin, rangeEnd, args...);
            if (errCode != 0) {
                return errCode;
            }
        } break;
        default:
            DEV_ERROR(ProgEncodeErr::CELL_MATCH_PARAM_INVALID,
                      "#ctrl.encode.stitch.dim: [Stitch] Too many dimensions: dimSize=%d, max=%d\n", dims,
                      DEV_SHAPE_DIM_NUM_5);
            break;
    }
    return errCode;
}

template <typename HandleType, typename... TyArgs>
static uint32_t CellMatchHandle(const uint64_t offset[DEV_SHAPE_DIM_MAX], const uint64_t shape[DEV_SHAPE_DIM_MAX],
                                const DevCellMatchTableDesc& cellMatchTableDesc, TyArgs... args)
{
    const int dims = cellMatchTableDesc.GetDimensionSize();
    uint64_t rangeBegin[DEV_SHAPE_DIM_MAX];
    uint64_t rangeEnd[DEV_SHAPE_DIM_MAX];
    for (int i = 0; i < dims; ++i) {
        auto cellMatchShapeDim = cellMatchTableDesc.GetCellShape(i);
        DEV_IF_NONDEVICE
        {
            if (cellMatchShapeDim == 0) {
                DEV_ERROR(ProgEncodeErr::CELL_MATCH_DIM_ZERO,
                          "#ctrl.encode.cell_match: cellMatchShapeDim is zero for dimension=%d", i);
                DEV_ASSERT(ProgEncodeErr::CELL_MATCH_DIM_ZERO, 0);
            }
        }
        rangeBegin[i] = offset[i] / cellMatchShapeDim;
        if (shape[i] == 0) {
            return 0;
        }
        rangeEnd[i] = (offset[i] + shape[i] - 1) / cellMatchShapeDim;
    }
    return CellMatchProcessByDim<HandleType>(dims, cellMatchTableDesc, rangeBegin, rangeEnd, args...);
}

template <typename... TyArgs>
static uint32_t CellMatchFill(const uint64_t offset[DEV_SHAPE_DIM_MAX], const uint64_t shape[DEV_SHAPE_DIM_MAX],
                              uint32_t operationIdx, const DevCellMatchTableDesc& cellMatchTableDesc, TyArgs... args)
{
    if constexpr (sizeof...(args) == 1) {
        auto argsTuple = std::make_tuple(args...);
        uint32_t* cellMatchTableData = std::get<0>(argsTuple);
        struct HandleFill {
            static inline uint32_t Process(int index, const DevCellMatchTableDesc& desc, uint32_t* cellMatchTableData,
                                           uint32_t operationIdx)
            {
                UNUSED(desc);
                cellMatchTableData[index] = operationIdx;
                DEV_VERBOSE_DEBUG("cell match fill, operation %u , cellindex[%d] = operationindex(%u)", operationIdx,
                                  index, operationIdx);
                return 0;
            }
        };
        return CellMatchHandle<HandleFill>(offset, shape, cellMatchTableDesc, cellMatchTableData, operationIdx);
    }
    if constexpr (sizeof...(args) == 3) {
        auto argsTuple = std::make_tuple(args...);
        uint64_t* cellMatchTableData = std::get<0>(argsTuple);
        uint32_t tagId = std::get<1>(argsTuple);
        uint32_t funcIdx = std::get<2>(argsTuple);
        struct HandleFill {
            static inline uint32_t Process(int index, const DevCellMatchTableDesc& desc, uint64_t* cellMatchTableData,
                                           uint32_t tagId, uint32_t funcIdx, uint32_t operationIdx)
            {
                UNUSED(desc);
                cellMatchTableData[index] = (static_cast<uint64_t>(tagId) << CELL_MATCH_META_TAGID_SHIFT32) |
                                            MakeTaskID(funcIdx, operationIdx);
                DEV_VERBOSE_DEBUG("cell match fill, tagid:%u funcIdx %u operation %u , cellindex[%d] = taskid(%lx)",
                                  tagId, funcIdx, operationIdx, index, cellMatchTableData[index]);
                return 0;
            }
        };
        return CellMatchHandle<HandleFill>(offset, shape, cellMatchTableDesc, cellMatchTableData, tagId, funcIdx,
                                           operationIdx);
    }
    return 0;
}

inline uint32_t CellMatchHandleFillEnhanceExec(int cellIndex, uint64_t* cellMatchTableData, uint32_t myOpType,
                                               uint64_t updateTagId, uint32_t taskId, uint32_t maxCount,
                                               const DevCellMatchTableDesc& desc)
{
    DEV_VERBOSE_DEBUG("CellMatchHandleFillEnhanceExec: cell[%d], cellMatchTableData=%p", cellIndex, cellMatchTableData);
    uint64_t cellMemBase = CellMatchCellIndexToMemBase(static_cast<uint64_t>(cellIndex), desc);
    uint64_t meta = cellMatchTableData[cellMemBase];
    uint32_t curActiveOpType = CellMatchGetCurrentOpType(meta);
    uint32_t curActiveOpCount = CellMatchGetCurrentOpCount(meta);
    uint64_t curTagId = CellMatchGetTagId(meta);
    uint32_t targetCount = 0, targetIndex = 0;
    if (maxCount == 0) {
        DEV_VERBOSE_DEBUG("Op type %u not supported in cell[%d], maxCount=0", myOpType, cellIndex);
        return static_cast<uint32_t>(CtrlErr::CELL_MATCH_OP_TYPE_NOT_SUPPORTED);
    }

    if (CellMatchIsMutexOp(myOpType, curActiveOpType)) {
        CellMatchSetCurrentOpType(meta, myOpType);
        CellMatchSetPrevMutexOpType(meta, curActiveOpType);
        CellMatchSetPrevMutexOpCount(meta, curActiveOpCount);
        targetCount = 1;
        targetIndex = 0;
        DEV_VERBOSE_DEBUG("Update mutex op: cell[%d], prev mutex type=%u (count=%u), active=%u (count=1)", cellIndex,
                          curActiveOpType, curActiveOpCount, myOpType);
    } else {
        targetCount = (curTagId != updateTagId) ? 1 : curActiveOpCount + 1;
        if (targetCount <= maxCount) {
            targetIndex = targetCount - 1;
            DEV_VERBOSE_DEBUG("Update multi-concurrent op : cell[%d], active=%u, count=%u -> %u", cellIndex, myOpType,
                              curActiveOpCount, targetCount);
        } else {
            DEV_VERBOSE_DEBUG("Op count not enough for cell[%d], opType=%u, newCount=%u, maxCount=%u", cellIndex,
                              myOpType, targetCount, maxCount);
            return static_cast<uint32_t>(CtrlErr::CELL_MATCH_FILL_OP_NOT_ENOUGH);
        }
    }

    if (curTagId != updateTagId) {
        // set pre mutex op as dirtry data, it's invalid
        CellMatchSetPrevMutexOpType(meta, CELL_MATCH_OP_TYPE_NONE);
        CellMatchSetPrevMutexOpCount(meta, CELL_MATCH_INVALID_OP_COUNT);
    }

    CellMatchSetCurrentOpCount(meta, targetCount);
    CellMatchSetTagId(meta, updateTagId);

    uint64_t fullTaskId = (static_cast<uint64_t>(updateTagId) << CELL_MATCH_META_TAGID_SHIFT32) | taskId;
    CellMatchAddOpId(cellMatchTableData, cellMemBase, fullTaskId, targetIndex, myOpType, desc);
    DEV_VERBOSE_DEBUG("Added opId to cell[%d]: taskId=0x%lx (Tagid=%lx, funcIdx=%u, opIdx=%u), index=%u, opType=%u",
                      cellIndex, fullTaskId, updateTagId, FuncID(taskId), taskId & TASKID_TASK_MASK, targetIndex,
                      myOpType);
    cellMatchTableData[cellMemBase] = meta;
    return 0;
}

template <typename... TyArgs>
static uint32_t CellMatchFillEnhance(const uint64_t offset[DEV_SHAPE_DIM_MAX], const uint64_t shape[DEV_SHAPE_DIM_MAX],
                                     uint32_t operationIdx, const DevCellMatchTableDesc& cellMatchTableDesc,
                                     uint32_t opType, TyArgs... args)
{
    if constexpr (sizeof...(args) == 1) {
        UNUSED(opType);
        auto argsTuple = std::make_tuple(args...);
        uint32_t* cellMatchTableData = std::get<0>(argsTuple);
        struct HandleFillFull {
            static inline uint32_t Process(int index, const DevCellMatchTableDesc& desc, uint32_t* cellMatchTableData,
                                           uint32_t operationIdx)
            {
                UNUSED(desc);
                cellMatchTableData[index] = operationIdx;
                DEV_VERBOSE_DEBUG("cell match fill full, operation %u , cellindex[%d] = operationindex(%u)",
                                  operationIdx, index, operationIdx);
                return 0;
            }
        };
        return CellMatchHandle<HandleFillFull>(offset, shape, cellMatchTableDesc, cellMatchTableData, operationIdx);
    }

    if constexpr (sizeof...(args) == 3) {
        auto argsTuple = std::make_tuple(args...);
        uint64_t* cellMatchTableData = std::get<0>(argsTuple);
        uint64_t tagId = std::get<1>(argsTuple);
        uint32_t maxCount = cellMatchTableDesc.GetCacheOpMaxCount(opType);
        uint32_t taskId = MakeTaskID(std::get<2>(argsTuple), operationIdx);

        struct HandleFillEnhance {
            static inline uint32_t Process(int cellIndex, const DevCellMatchTableDesc& desc, uint64_t* data,
                                           uint64_t tagId, uint32_t taskId, uint32_t type, uint32_t maxCount)
            {
                return CellMatchHandleFillEnhanceExec(cellIndex, data, type, tagId, taskId, maxCount, desc);
            }
        };

        return CellMatchHandle<HandleFillEnhance>(offset, shape, cellMatchTableDesc, cellMatchTableData, tagId, taskId,
                                                  opType, maxCount);
    }
    return 0;
}

template <typename... TyArgs>
static uint32_t CellMatchFillIncastOutcast(DevAscendFunction* devFunc, DevAscendFunctionCallOperandUse* operandUseList,
                                           size_t useSize, const uint64_t* runtimeExpressionList,
                                           const DevCellMatchTableDesc& cellMatchTableDesc, TyArgs... args)
{
    for (size_t i = 0; i < useSize; i++) {
        auto& use = operandUseList[i];
        uint64_t offset[DEV_SHAPE_DIM_MAX];
        uint64_t validShape[DEV_SHAPE_DIM_MAX];

        GetTensorOffsetAndValidShape(devFunc, offset, validShape, runtimeExpressionList, cellMatchTableDesc,
                                     cellMatchTableDesc.GetDimensionSize(), use.offsetAttrIdx, use.cachedAttrBase,
                                     use.operationIdx);

        DEV_IF_VERBOSE_DEBUG
        {
            for (int j = 0; j < cellMatchTableDesc.GetDimensionSize(); j++) {
                DEV_VERBOSE_DEBUG("CellMatchFillIncastOutcast, op[%d] -> opType:%u -> dimension[%d] = (offset:%lu "
                                  ", validShape:%lu, cellshape:%d)",
                                  use.operationIdx, static_cast<uint32_t>(use.opType), j, offset[j], validShape[j],
                                  cellMatchTableDesc.cellShape.dim[j]);
            }
        }

        uint32_t errCode = CellMatchFillEnhance(offset, validShape, use.operationIdx, cellMatchTableDesc,
                                                static_cast<uint32_t>(use.opType), args...);
        if (errCode != 0) {
            return errCode;
        }
    }
    return 0;
}
} // namespace npu::tile_fwk::dynamic
