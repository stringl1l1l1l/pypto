/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file torch_tensor_converter.cpp
 * \brief Implementation of PyTorch tensor to DeviceTensorData conversion.
 */

#include "bindings/torch_tensor_converter.h"
#include "tilefwk/error.h"
#include "interface/utils/error.h"

#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

using namespace npu::tile_fwk;
namespace {

constexpr int kNpuFormatNz = 29; // NZ in CANN format is 29

struct TensorDeviceInfo {
    std::string type;
    int index{0};
};

uintptr_t ReadTensorDataPtr(const py::object& torchTensor)
{
    return static_cast<uintptr_t>(py::cast<int64_t>(torchTensor.attr("data_ptr")()));
}

void SetNZTensorShapeAlign(std::vector<int64_t>& shape, DataType dtype)
{
    if (shape.size() <= 1)
        return;
    int64_t dtypeBytes = static_cast<int64_t>(BytesOf(dtype));
    // Ensure dtypeBytes is positive to avoid division by zero
    if (dtypeBytes <= 0)
        return;
    // Calculate inner axis alignment size: 32 bytes / data type byte size
    int64_t blockAlignBytes = 32;
    int64_t c0Size = blockAlignBytes / dtypeBytes;
    // Additional safety check to prevent division by zero in later calculations
    if (c0Size <= 0)
        return;
    int64_t blockAlignSize = 16;
    int64_t inner_index = shape.size() - 1;
    int64_t outer_index = shape.size() - 2;
    int64_t inner_shape = shape[inner_index];
    int64_t outer_shape = shape[outer_index];
    // Align inner axis to 32-byte boundary using ceiling division
    shape[inner_index] = (inner_shape + c0Size - 1) / c0Size * c0Size;
    // Align outer axis to 16 elements using ceiling division
    shape[outer_index] = (outer_shape + blockAlignSize - 1) / blockAlignSize * blockAlignSize;
}

std::vector<int64_t> ReadTensorShape(const py::object& torchTensor, DataType dtype, TileOpFormat format)
{
    py::object tensorShape = torchTensor.attr("shape");
    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(py::len(tensorShape)));
    for (auto dim : tensorShape) {
        shape.push_back(py::cast<int64_t>(dim));
    }
    if (format == TileOpFormat::TILEOP_NZ) {
        SetNZTensorShapeAlign(shape, dtype);
    }
    if (dtype == DataType::DT_FP4_E1M2 || dtype == DataType::DT_FP4_E2M1) {
        shape.back() *= 0x2;
    }
    return shape;
}

TensorDeviceInfo ReadDeviceInfo(const py::object& torchTensor)
{
    py::object device = torchTensor.attr("device");
    std::string type = py::getattr(device, "type").cast<std::string>();
    int index = type == "cpu" ? 0 : py::getattr(device, "index").cast<int>();
    return TensorDeviceInfo{std::move(type), index};
}

DataType TorchDtypeToDataType(const py::object& torchDtype)
{
    static const std::unordered_map<std::string, DataType> dtypeMap = {
        {"torch.float16", DataType::DT_FP16},
        {"torch.bfloat16", DataType::DT_BF16},
        {"torch.float32", DataType::DT_FP32},
        {"torch.float64", DataType::DT_DOUBLE},
        {"torch.int8", DataType::DT_INT8},
        {"torch.uint8", DataType::DT_UINT8},
        {"torch.int16", DataType::DT_INT16},
        {"torch.uint16", DataType::DT_UINT16},
        {"torch.int32", DataType::DT_INT32},
        {"torch.uint32", DataType::DT_UINT32},
        {"torch.int64", DataType::DT_INT64},
        {"torch.uint64", DataType::DT_UINT64},
        {"torch.bool", DataType::DT_BOOL},
        {"torch.float8_e4m3fn", DataType::DT_FP8E4M3},
        {"torch.float8_e5m2", DataType::DT_FP8E5M2},
        {"torch.float8_e8m0fnu", DataType::DT_FP8E8M0},
        {"torch.float4_e2m1fn_x2", DataType::DT_FP4_E2M1X2},
    };

    const std::string dtype = py::str(torchDtype).cast<std::string>();
    auto iter = dtypeMap.find(dtype);
    if (iter != dtypeMap.end()) {
        return iter->second;
    }
    throw std::runtime_error("Input torch.dtype is not supported. Got " + dtype);
}

DataType ReadTensorDataType(const py::object& tensorDef, const py::object& torchTensor, Tensor& baseTensor)
{
    if (!tensorDef.attr("explicit_dtype").is_none()) {
        return baseTensor.GetDataType();
    }
    return TorchDtypeToDataType(torchTensor.attr("dtype"));
}

TileOpFormat ReadTensorFormat(const py::object& tensorDef, const py::object& torchTensor, Tensor& baseTensor,
                              const TensorDeviceInfo& deviceInfo, py::module_& torch_npu)
{
    if (!tensorDef.attr("explicit_format").is_none()) {
        return baseTensor.Format();
    }
    if (deviceInfo.type != "npu") {
        return TileOpFormat::TILEOP_ND;
    }
    if (torch_npu.ptr() == nullptr) {
        torch_npu = py::module::import("torch_npu");
    }
    int npuFormat = py::cast<int>(torch_npu.attr("get_npu_format")(torchTensor));
    return npuFormat == kNpuFormatNz ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
}

Tensor& ReadTensorDefBase(const py::object& tensorDef)
{
    auto baseObj = py::getattr(tensorDef, "_base", py::none());
    FE_ASSERT(FeError::INVALID_TYPE, py::isinstance<Tensor>(baseObj)) << "the '_base' attribute must be a Tensor type";
    return baseObj.cast<Tensor&>();
}

bool IsSameDevice(const TensorDeviceInfo& lhs, const TensorDeviceInfo& rhs)
{
    return lhs.type == rhs.type && lhs.index == rhs.index;
}

TensorDeviceInfo ConvertSingleTensor(const py::object& torchTensor, const py::object& tensorDef, py::module_& torch_npu,
                                     npu::tile_fwk::dynamic::DeviceTensorData& out)
{
    TensorDeviceInfo deviceInfo = ReadDeviceInfo(torchTensor);
    Tensor& baseTensor = ReadTensorDefBase(tensorDef);
    const DataType dtype = ReadTensorDataType(tensorDef, torchTensor, baseTensor);

    const uintptr_t dataPtr = ReadTensorDataPtr(torchTensor);
    const TileOpFormat format = ReadTensorFormat(tensorDef, torchTensor, baseTensor, deviceInfo, torch_npu);
    std::vector<int64_t> shape = ReadTensorShape(torchTensor, dtype, format);
    out = npu::tile_fwk::dynamic::DeviceTensorData(dtype, dataPtr, shape, format);
    return deviceInfo;
}

int ValidateDeviceAndReturnIndex(const std::optional<TensorDeviceInfo>& deviceInfo)
{
    if (config::GetRuntimeOption<int64_t>(CFG_RUN_MODE) == CFG_RUN_MODE_SIM) {
        return 0;
    }
    if (!deviceInfo.has_value() || deviceInfo->type != "npu") {
        throw std::runtime_error("Not npu device");
    }
    return deviceInfo->index;
}

} // namespace

namespace pypto {

int TorchTensorConverter::Convert(py::sequence& tensors, py::sequence& tensor_defs,
                                  std::vector<npu::tile_fwk::dynamic::DeviceTensorData>& tensors_data)
{
    const size_t n = static_cast<size_t>(py::len(tensors));
    CHECK(FeError::INVALID_VAL, n != 0) << "Empty tensor list";

    tensors_data.reserve(n);

    py::module torch_npu;
    std::optional<TensorDeviceInfo> commonDeviceInfo;

    for (size_t i = 0; i < n; i++) {
        py::int_ index(i);
        py::object torchTensor = tensors[index];
        py::object tensorDef = tensor_defs[index];

        tensors_data.emplace_back();

        TensorDeviceInfo tensorDeviceInfo = ConvertSingleTensor(torchTensor, tensorDef, torch_npu, tensors_data.back());
        // Skip device consistency check for cpu tensors (used as host-side data for value-depend).
        if (tensorDeviceInfo.type == "cpu") {
            continue;
        }

        if (!commonDeviceInfo.has_value()) {
            commonDeviceInfo.emplace(std::move(tensorDeviceInfo));
        } else if (!IsSameDevice(*commonDeviceInfo, tensorDeviceInfo)) {
            throw std::runtime_error("All input tensors must be on the same device");
        }
    }
    return ValidateDeviceAndReturnIndex(commonDeviceInfo);
}

size_t ValidateInputs(py::sequence& tensors, py::sequence& tensorDefs)
{
    size_t n = static_cast<size_t>(py::len(tensors));
    CHECK(FeError::INVALID_VAL, n == static_cast<size_t>(py::len(tensorDefs)))
        << "Input length mismatch: tensors(" << n << ") vs tensor_defs(" << py::len(tensorDefs) << ")";
    CHECK(FeError::INVALID_VAL, n != 0) << "Empty tensor list";
    return n;
}

} // namespace pypto
