# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Validate the packaged helper's CANN wire records independently of the exporter."""

import os
from pathlib import Path
import subprocess

import pytest


def test_native_prof_tensor_info(tmp_path):
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if not ascend_home:
        pytest.skip("CANN headers required to compile the packaged profiling helper")
    root = Path(__file__).resolve().parents[6]
    source = tmp_path / "tensor_info.cpp"
    source.write_text(r"""
#include "pypto_profiler.h"
#include <cassert>
#include <vector>

static std::vector<unsigned char> cached;
static std::vector<MsprofAdditionalInfo> tensors;
static MsprofCompactInfo basic;
static MsprofApi api;
static unsigned cacheCalls, basicCalls, apiCalls, cycles;
static int reportError;

extern "C" aclError aclrtCacheLastTaskOpInfo(const void* const data, const size_t size)
{
    ++cacheCalls;
    const auto* bytes = static_cast<const unsigned char*>(data);
    cached.assign(bytes, bytes + size);
    return reportError;
}
extern "C" uint64_t MsprofSysCycleTime() { return 100 + ++cycles; }
extern "C" int32_t MsprofReportCompactInfo(uint32_t aging, void* data, uint32_t size)
{
    assert(aging && size == sizeof(basic));
    ++basicCalls;
    std::memcpy(&basic, data, size);
    return reportError;
}
extern "C" int32_t MsprofReportApi(uint32_t aging, const MsprofApi* data)
{
    assert(aging);
    ++apiCalls;
    api = *data;
    return reportError;
}
extern "C" int32_t MsprofReportAdditionalInfo(uint32_t aging, void* data, uint32_t size)
{
    assert(aging && size == sizeof(MsprofAdditionalInfo));
    MsprofAdditionalInfo record;
    std::memcpy(&record, data, size);
    tensors.push_back(record);
    return reportError;
}
template <typename T> T read(size_t offset)
{
    T value;
    assert(offset + sizeof(value) <= cached.size());
    std::memcpy(&value, cached.data() + offset, sizeof(value));
    return value;
}
static void checkTensor(const MsrofTensorData& tensor, unsigned index)
{
    assert(tensor.tensorType == index % 2);
    assert(tensor.format == 2 && tensor.dataType == index);
    for (unsigned dim = 0; dim < 8; ++dim) {
        assert(tensor.shape[dim] == (dim < index % 9 ? 32 + index + dim : 0));
    }
}
static void run(bool live, bool capture, unsigned count, int error = 0)
{
    cached.clear(); tensors.clear();
    cacheCalls = basicCalls = apiCalls = cycles = 0;
    reportError = error;
    std::vector<aclprofTensor> inputs(count);
    for (unsigned i = 0; i < count; ++i) {
        inputs[i].type = i % 2;
        inputs[i].format = 2;
        inputs[i].dataType = i;
        inputs[i].shapeDim = i % 9;
        // Nonzero padding must not leak into either wire format.
        for (unsigned dim = 0; dim < 8; ++dim) {
            inputs[i].shape[dim] = 32 + i + dim;
        }
    }
    aclprofTensorInfo info{};
    info.opNameId = 0x1122334455667788;
    info.opTypeId = 0x8877665544332211;
    info.kernelType = 3;
    info.blockNums = (2U << 16) | 7;
    info.tensorNum = count;
    info.tensors = inputs.data();
    const uint64_t begin = pypto::BeginCaptureTensorReport(live);
    assert(begin == (live ? 101 : 0));
    pypto::ReportCaptureTensorInfo(info, begin, capture);
    if (capture && count <= 1488) {
        assert(cacheCalls == 1 && cached.size() == 48 + 44 * count);
        // CANN Runtime's CacheTaskInfo ABI: 48-byte header followed by packed
        // 44-byte tensors. These offsets also match machine's host_prof.h.
        assert(read<uint32_t>(0) == 3 && read<uint32_t>(4) == ((2U << 16) | 7));
        assert(read<uint64_t>(8) == info.opNameId && read<uint64_t>(16) == info.opTypeId);
        assert(read<uint64_t>(24) == 0 && read<uint64_t>(32) == 0);
        assert(read<uint32_t>(40) == 0 && read<uint32_t>(44) == count);
        for (unsigned i = 0; i < count; ++i) {
            checkTensor(read<MsrofTensorData>(48 + 44 * i), i);
        }
    } else {
        assert(cacheCalls == 0 && cached.empty());
    }
    assert(basicCalls == live && apiCalls == live);
    if (!live) {
        assert(cycles == 0 && tensors.empty());
        return;
    }
    assert(basic.level == MSPROF_REPORT_NODE_LEVEL && basic.type == MSPROF_REPORT_NODE_BASIC_INFO_TYPE);
    assert(basic.dataLen == sizeof(MsprofNodeBasicInfo));
    assert(basic.data.nodeBasicInfo.opName == info.opNameId && basic.data.nodeBasicInfo.opType == info.opTypeId);
    assert(basic.data.nodeBasicInfo.taskType == info.kernelType && basic.data.nodeBasicInfo.blockDim == 7);
    assert(api.level == MSPROF_REPORT_NODE_LEVEL && api.type == MSPROF_REPORT_NODE_LAUNCH_TYPE);
    assert(api.itemId == info.opNameId && api.beginTime == begin && api.endTime > begin);
    assert(api.threadId == basic.threadId && basic.timeStamp == api.endTime);
    assert(tensors.size() == (count + 4) / 5);
    unsigned index = 0;
    for (const auto& record : tensors) {
        assert(record.level == MSPROF_REPORT_NODE_LEVEL && record.type == MSPROF_REPORT_NODE_TENSOR_INFO_TYPE);
        assert(record.threadId == api.threadId && record.timeStamp == basic.timeStamp);
        assert(record.dataLen == sizeof(MsprofTensorInfo));
        MsprofTensorInfo tensorInfo;
        std::memcpy(&tensorInfo, record.data, sizeof(tensorInfo));
        assert(tensorInfo.opName == info.opNameId);
        assert(tensorInfo.tensorNum == (count - index < 5 ? count - index : 5));
        for (unsigned i = 0; i < tensorInfo.tensorNum; ++i) {
            checkTensor(tensorInfo.tensorData[i], index++);
        }
    }
    assert(index == count);
}
int main()
{
    run(false, false, 9);
    run(false, true, 9);
    run(true, false, 9);
    run(true, true, 9);
    run(true, true, 9, 1);  // API failures never suppress the remaining reports or throw.
    run(false, true, 1488); // Largest payload below the 64 KiB API limit.
    run(false, true, 1489); // Oversized payload is not submitted.
}
""")
    binary = tmp_path / "tensor_info"
    subprocess.run([
        "g++", "-std=c++17", "-O2", str(source), "-o", str(binary),
        "-I", str(root / "python/pypto_pro/include"), "-I", str(Path(ascend_home) / "include"),
        "-I", str(Path(ascend_home) / "pkg_inc"),
    ], check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
