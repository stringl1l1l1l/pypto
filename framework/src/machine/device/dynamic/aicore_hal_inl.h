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
 * \file aicore_hal_inl.h
 * \brief Out-of-class inline definitions for AicoreHAL (include only after AicoreHAL is defined).
 */

#pragma once

namespace npu::tile_fwk::dynamic {

inline void AicoreHAL::WriteReg32(int coreStart, int coreEnd, int offset, uint32_t val)
{
    if constexpr (IsDeviceMode()) {
        int i, idx = coreStart;
        int n = coreEnd - coreStart;
        for (i = 0; i < (n & (~CORE_QUEUE_MODE_NUM_7)); i += CORE_QUEUE_MODE_NUM_8) {
            *(reinterpret_cast<volatile uint32_t*>(regAddrs_[GetPhyIdByBlockId(idx++)] + offset)) = val;
            *(reinterpret_cast<volatile uint32_t*>(regAddrs_[GetPhyIdByBlockId(idx++)] + offset)) = val;
            *(reinterpret_cast<volatile uint32_t*>(regAddrs_[GetPhyIdByBlockId(idx++)] + offset)) = val;
            *(reinterpret_cast<volatile uint32_t*>(regAddrs_[GetPhyIdByBlockId(idx++)] + offset)) = val;
            *(reinterpret_cast<volatile uint32_t*>(regAddrs_[GetPhyIdByBlockId(idx++)] + offset)) = val;
            *(reinterpret_cast<volatile uint32_t*>(regAddrs_[GetPhyIdByBlockId(idx++)] + offset)) = val;
            *(reinterpret_cast<volatile uint32_t*>(regAddrs_[GetPhyIdByBlockId(idx++)] + offset)) = val;
            *(reinterpret_cast<volatile uint32_t*>(regAddrs_[GetPhyIdByBlockId(idx++)] + offset)) = val;
        }
        switch (n & CORE_QUEUE_MODE_NUM_7) {
            case CORE_QUEUE_MODE_NUM_7:
                *(reinterpret_cast<volatile uint32_t*>(regAddrs_[GetPhyIdByBlockId(idx++)] + offset)) = val;
                [[fallthrough]];
            case CORE_QUEUE_MODE_NUM_6:
                *(reinterpret_cast<volatile uint32_t*>(regAddrs_[GetPhyIdByBlockId(idx++)] + offset)) = val;
                [[fallthrough]];
            case CORE_QUEUE_MODE_NUM_5:
                *(reinterpret_cast<volatile uint32_t*>(regAddrs_[GetPhyIdByBlockId(idx++)] + offset)) = val;
                [[fallthrough]];
            case CORE_QUEUE_MODE_NUM_4:
                *(reinterpret_cast<volatile uint32_t*>(regAddrs_[GetPhyIdByBlockId(idx++)] + offset)) = val;
                [[fallthrough]];
            case CORE_QUEUE_MODE_NUM_3:
                *(reinterpret_cast<volatile uint32_t*>(regAddrs_[GetPhyIdByBlockId(idx++)] + offset)) = val;
                [[fallthrough]];
            case CORE_QUEUE_MODE_NUM_2:
                *(reinterpret_cast<volatile uint32_t*>(regAddrs_[GetPhyIdByBlockId(idx++)] + offset)) = val;
                [[fallthrough]];
            case CORE_QUEUE_MODE_NUM_1:
                *(reinterpret_cast<volatile uint32_t*>(regAddrs_[GetPhyIdByBlockId(idx++)] + offset)) = val;
                [[fallthrough]];
            default:
                break;
        }
    } else {
        for (int i = coreStart; i < coreEnd; ++i) {
            WriteReg32(i, offset, val);
        }
    }
}

inline void AicoreHAL::SetReadyQueue(int coreStart, int coreEnd, uint32_t val)
{
    if constexpr (IsDeviceMode()) {
        int i, idx = coreStart;
        int n = coreEnd - coreStart;
        for (i = 0; i < (n & (~CORE_QUEUE_MODE_NUM_7)); i += CORE_QUEUE_MODE_NUM_8) {
            *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
            *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
            *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
            *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
            *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
            *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
            *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
            *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
        }
        switch (n & CORE_QUEUE_MODE_NUM_7) {
            case CORE_QUEUE_MODE_NUM_7:
                *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                [[fallthrough]];
            case CORE_QUEUE_MODE_NUM_6:
                *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                [[fallthrough]];
            case CORE_QUEUE_MODE_NUM_5:
                *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                [[fallthrough]];
            case CORE_QUEUE_MODE_NUM_4:
                *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                [[fallthrough]];
            case CORE_QUEUE_MODE_NUM_3:
                *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                [[fallthrough]];
            case CORE_QUEUE_MODE_NUM_2:
                *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                [[fallthrough]];
            case CORE_QUEUE_MODE_NUM_1:
                *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                [[fallthrough]];
            default:
                break;
        }
    } else {
        for (int i = coreStart; i < coreEnd; ++i) {
            SetReadyQueue(i, static_cast<uint64_t>(val));
        }
    }
}

inline void AicoreHAL::GetFinishQueue(const uint32_t* coreIdx, uint32_t* vals, int n)
{
    if constexpr (IsDeviceMode()) {
        for (int i = 0; i < (n & (~CORE_QUEUE_MODE_NUM_7)); i += CORE_QUEUE_MODE_NUM_8) {
            *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
            *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
            *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
            *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
            *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
            *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
            *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
            *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
        }
        switch (n & CORE_QUEUE_MODE_NUM_7) {
            case CORE_QUEUE_MODE_NUM_7:
                *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                [[fallthrough]];
            case CORE_QUEUE_MODE_NUM_6:
                *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                [[fallthrough]];
            case CORE_QUEUE_MODE_NUM_5:
                *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                [[fallthrough]];
            case CORE_QUEUE_MODE_NUM_4:
                *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                [[fallthrough]];
            case CORE_QUEUE_MODE_NUM_3:
                *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                [[fallthrough]];
            case CORE_QUEUE_MODE_NUM_2:
                *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                [[fallthrough]];
            case CORE_QUEUE_MODE_NUM_1:
                *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                [[fallthrough]];
            default:
                break;
        }
    } else {
        for (int i = 0; i < n; i++) {
            vals[i] = static_cast<uint32_t>(GetFinishedTask(coreIdx[i]));
        }
    }
}

inline void AicoreHAL::SendWaveGoodbye(int coreStart, int coreEnd)
{
    if constexpr (IsDeviceMode()) {
        int i, idx = coreStart;
        int n = coreEnd - coreStart;
        for (i = 0; i < (n & (~CORE_QUEUE_MODE_NUM_7)); i += CORE_QUEUE_MODE_NUM_8) {
            SendWaveGoodbyeDeviceOne(idx++);
            SendWaveGoodbyeDeviceOne(idx++);
            SendWaveGoodbyeDeviceOne(idx++);
            SendWaveGoodbyeDeviceOne(idx++);
            SendWaveGoodbyeDeviceOne(idx++);
            SendWaveGoodbyeDeviceOne(idx++);
            SendWaveGoodbyeDeviceOne(idx++);
            SendWaveGoodbyeDeviceOne(idx++);
        }
        switch (n & CORE_QUEUE_MODE_NUM_7) {
            case CORE_QUEUE_MODE_NUM_7:
                SendWaveGoodbyeDeviceOne(idx++);
                [[fallthrough]];
            case CORE_QUEUE_MODE_NUM_6:
                SendWaveGoodbyeDeviceOne(idx++);
                [[fallthrough]];
            case CORE_QUEUE_MODE_NUM_5:
                SendWaveGoodbyeDeviceOne(idx++);
                [[fallthrough]];
            case CORE_QUEUE_MODE_NUM_4:
                SendWaveGoodbyeDeviceOne(idx++);
                [[fallthrough]];
            case CORE_QUEUE_MODE_NUM_3:
                SendWaveGoodbyeDeviceOne(idx++);
                [[fallthrough]];
            case CORE_QUEUE_MODE_NUM_2:
                SendWaveGoodbyeDeviceOne(idx++);
                [[fallthrough]];
            case CORE_QUEUE_MODE_NUM_1:
                SendWaveGoodbyeDeviceOne(idx++);
                [[fallthrough]];
            default:
                break;
        }
    } else {
        for (int i = coreStart; i < coreEnd; ++i) {
            SendWaveGoodbye(i);
        }
    }
}

inline void AicoreHAL::ResetCoreStopSlotDeviceOne(int coreIdx)
{
    volatile ParallelDevTask* task = &args_[coreIdx]->parallelDevTask;
    task->version = 0;
    task->front = 0;
    task->rear = 0;
    task->reserver = 0;
    for (uint32_t i = 0; i < npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM; i++) {
        task->ptrElements[i] = 0;
        task->idElements[i] = 0;
    }
    args_[coreIdx]->shakeBuffer[0] = 0;
    args_[coreIdx]->shakeBufferCpuToCore[CPU_TO_CORE_SHAK_BUF_COREFUNC_DATA_INDEX] = 0;
}

inline void AicoreHAL::ResetCoreStopSlot(int coreStart, int coreEnd)
{
    if constexpr (IsDeviceMode()) {
        int i, idx = coreStart;
        int n = coreEnd - coreStart;
        for (i = 0; i < (n & (~CORE_QUEUE_MODE_NUM_7)); i += CORE_QUEUE_MODE_NUM_8) {
            ResetCoreStopSlotDeviceOne(idx++);
            ResetCoreStopSlotDeviceOne(idx++);
            ResetCoreStopSlotDeviceOne(idx++);
            ResetCoreStopSlotDeviceOne(idx++);
            ResetCoreStopSlotDeviceOne(idx++);
            ResetCoreStopSlotDeviceOne(idx++);
            ResetCoreStopSlotDeviceOne(idx++);
            ResetCoreStopSlotDeviceOne(idx++);
        }
        switch (n & CORE_QUEUE_MODE_NUM_7) {
            case CORE_QUEUE_MODE_NUM_7:
                ResetCoreStopSlotDeviceOne(idx++);
                [[fallthrough]];
            case CORE_QUEUE_MODE_NUM_6:
                ResetCoreStopSlotDeviceOne(idx++);
                [[fallthrough]];
            case CORE_QUEUE_MODE_NUM_5:
                ResetCoreStopSlotDeviceOne(idx++);
                [[fallthrough]];
            case CORE_QUEUE_MODE_NUM_4:
                ResetCoreStopSlotDeviceOne(idx++);
                [[fallthrough]];
            case CORE_QUEUE_MODE_NUM_3:
                ResetCoreStopSlotDeviceOne(idx++);
                [[fallthrough]];
            case CORE_QUEUE_MODE_NUM_2:
                ResetCoreStopSlotDeviceOne(idx++);
                [[fallthrough]];
            case CORE_QUEUE_MODE_NUM_1:
                ResetCoreStopSlotDeviceOne(idx++);
                [[fallthrough]];
            default:
                break;
        }
    } else {
        for (int i = coreStart; i < coreEnd; ++i) {
            ResetCoreStopSlot(i);
        }
    }
}

} // namespace npu::tile_fwk::dynamic
