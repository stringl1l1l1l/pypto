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
 * \file raw_tensor_data.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <fstream>

#include "interface/interpreter/interpreter_log.h"
#include "tilefwk/data_type.h"
#include "tilefwk/tensor.h"
#include "interface/inner/element.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/tensor_offset.h"
#include "interface/interpreter/calculator/dtype_utils.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {

template <typename T, std::size_t Align>
class AlignedAllocator {
public:
    using value_type = T;

    template <class U>
    struct rebind {
        using other = AlignedAllocator<U, Align>;
    };

    AlignedAllocator() = default;
    template <class U>
    AlignedAllocator(const AlignedAllocator<U, Align>&) noexcept
    {}

    T* allocate(std::size_t n)
    {
        if (n > std::size_t(-1) / sizeof(T))
            throw std::bad_alloc();
        void* p = nullptr;
        if (::posix_memalign(&p, Align, n * sizeof(T)) != 0)
            throw std::bad_alloc();
        return static_cast<T*>(p);
    }
    void deallocate(T* p, std::size_t) noexcept { std::free(p); }

    // 无状态 allocator：任意实例等价。std::vector 的拷贝赋值 / swap 需要比较 allocator，
    // C++17 下 == 与 != 都必须显式提供（C++20 才可由 == 合成 !=）。
    template <class U>
    bool operator==(const AlignedAllocator<U, Align>&) const noexcept
    {
        return true;
    }
    template <class U>
    bool operator!=(const AlignedAllocator<U, Align>&) const noexcept
    {
        return false;
    }
};

using StorageData = std::vector<uint8_t, AlignedAllocator<uint8_t, 0x40>>;

struct RawTensorData {
    static int64_t Numel(const std::vector<int64_t>& shape)
    {
        if (shape.empty()) {
            return 0;
        }
        int64_t n = 1;
        for (auto d : shape) {
            n *= d;
        }
        return n;
    }

    static std::vector<int64_t> PackedShapeFromLogical(const std::vector<int64_t>& logicalShape, DataType dtype)
    {
        if (!IsFp4PackedDtype(dtype) || logicalShape.empty()) {
            return logicalShape;
        }
        auto packed = logicalShape;
        // One uint8 stores two FP4 elements.
        packed.back() = (packed.back() + 1) / 0x2;
        return packed;
    }

    static int GetDataSize(DataType dataType)
    {
        constexpr int DATA_SIZE_SUB_BYTE = -1;
        int64_t bits = 0;
        try {
            bits = BitsOf(dataType);
        } catch (...) {
            return 0;
        }
        if (bits <= 0) {
            return 0;
        }
        if (bits < 0x8) {
            return DATA_SIZE_SUB_BYTE;
        }
        return static_cast<int>(bits / 0x8);
    }

    static size_t CalcRequiredSize(DataType dataType, const std::vector<int64_t>& shape)
    {
        int64_t n = Numel(shape);
        int elemSize = GetDataSize(dataType);
        if (elemSize > 0) {
            return static_cast<size_t>(n) * static_cast<size_t>(elemSize);
        }
        if (IsFp4PackedDtype(dataType)) {
            auto packedShape = PackedShapeFromLogical(shape, dataType);
            return static_cast<size_t>(Numel(packedShape));
        }
        return static_cast<size_t>((n + 1) / 0x2);
    }

    static std::vector<int64_t> ShapeToStride(const std::vector<int64_t>& shape)
    {
        ASSERT(ControlFlowScene::INVALID_TENSOR_SHAPE, !shape.empty())
            << "ShapeToStride: empty shape is not allowed" << shape;
        std::vector<int64_t> stride;
        stride.resize(shape.size());
        stride[shape.size() - 1] = 1;
        for (int k = static_cast<int>(shape.size()) - 2; k >= 0; k--) {
            stride[k] = stride[k + 1] * shape[k + 1];
        }
        return stride;
    }

    RawTensorData() : RawTensorData(DT_UINT8, {}) {}
    RawTensorData(DataType dataType, const std::vector<int64_t>& shape)
        : dataType_(dataType),
          shape_(shape),
          stride_(ShapeToStride(shape)),
          nelem(Numel(shape)),
          elemSize_(GetDataSize(dataType))
    {
        size_t bytes = 0;
        if (elemSize_ > 0) {
            bytes = static_cast<size_t>(nelem) * static_cast<size_t>(elemSize_);
        } else if (IsFp4PackedDtype(dataType_)) {
            auto packedShape = PackedShapeFromLogical(shape_, dataType_);
            bytes = static_cast<size_t>(Numel(packedShape));
        } else {
            bytes = static_cast<size_t>((nelem + 1) / 0x2);
        }
        data_ = std::make_shared<StorageData>();
        data_->resize(bytes);
    }

    RawTensorData(std::shared_ptr<StorageData> data, DataType dataType, const std::vector<int64_t>& shape)
        : data_(data),
          dataType_(dataType),
          shape_(shape),
          stride_(ShapeToStride(shape)),
          nelem(Numel(shape)),
          elemSize_(GetDataSize(dataType))
    {}

    RawTensorData(const RawTensorData& other)
        : devPtr_(nullptr),
          dataType_(other.dataType_),
          shape_(other.shape_),
          stride_(other.stride_),
          nelem(other.nelem),
          elemSize_(other.elemSize_),
          isShmTensor_(other.isShmTensor_),
          shmOffset_(other.shmOffset_)
    {
        if (other.data_) {
            data_ = std::make_shared<StorageData>(*other.data_);
        }
    }

    const Shape& GetShape() const { return shape_; }
    const Stride& GetStride() const { return stride_; }
    DataType GetDataType() const { return dataType_; }
    int64_t GetSize() const { return nelem; }
    int64_t GetElementSize() const { return elemSize_; }
    std::shared_ptr<StorageData> GetRawData() const { return data_; }
    uint8_t* data() { return data_->data(); }
    const uint8_t* data() const { return data_->data(); }
    size_t size() const { return data_->size(); }
    void resize(size_t n) { data_->resize(n); }
    uint8_t& at(size_t pos) { return data_->at(pos); }
    const uint8_t& at(size_t pos) const { return data_->at(pos); }
    void reserve(size_t n) { data_->reserve(n); }
    size_t capacity() const { return data_->capacity(); }
    bool empty() const { return data_->empty(); }

    template <typename T>
    const T& Get(int index) const
    {
        ASSERT(ExecuteOperationScene::INVALID_TENSOR_DTYPE, elemSize_ > 0)
            << "Get() is not supported for packed sub-byte dtypes (use raw bytes).";
        const void* addr = &data_->data()[static_cast<size_t>(index) * static_cast<size_t>(elemSize_)];
        return *static_cast<const T*>(addr);
    }

    template <typename T>
    T& Get(int index)
    {
        ASSERT(ExecuteOperationScene::INVALID_TENSOR_DTYPE, elemSize_ > 0)
            << "Get() is not supported for packed sub-byte dtypes (use raw bytes).";
        void* addr = &data_->data()[static_cast<size_t>(index) * static_cast<size_t>(elemSize_)];
        return *static_cast<T*>(addr);
    }

    Element GetElement(int index) const
    {
        switch (GetDataType()) {
#define CASE_DATA_TYPE_DIS(ast2Type, dataType, calcType, indexArg) \
    case ast2Type:                                                 \
        return Element(ast2Type, static_cast<calcType>(Get<dataType>(indexArg)))
            DISPATCH_DATA_TYPE(CASE_DATA_TYPE_DIS, index);
#undef CASE_DATA_TYPE_DIS
            case DT_BOOL:
                return Element(DT_BOOL, Get<bool>(index));
            default:
                ASSERT(ExecuteOperationScene::INVALID_TENSOR_DTYPE, false);
                return Element();
        }
    }

    Element GetElement(int64_t* coords, size_t n) const
    {
        int64_t index = 0;
        ASSERT(ExecuteOperationScene::INVALID_TENSOR_SHAPE, n == shape_.size());
        index = std::inner_product(coords, coords + n, stride_.begin(), 0);
        return GetElement(index);
    }

    std::string DumpElement(int index) const
    {
        switch (GetDataType()) {
            case DT_INT8:
                return std::to_string(Get<int8_t>(index));
            case DT_BOOL:
                return std::to_string(Get<bool>(index));
            case DT_INT16:
                return std::to_string(Get<int16_t>(index));
            case DT_INT32:
                return std::to_string(Get<int32_t>(index));
            case DT_INT64:
                return std::to_string(Get<int64_t>(index));
            case DT_FP16:
                return std::to_string(Get<npu::tile_fwk::float16>(index));
            case DT_FP32:
                return std::to_string(Get<float>(index));
            case DT_BF16:
                return std::to_string(Get<npu::tile_fwk::bfloat16>(index));
            case DT_UINT8:
                return std::to_string(Get<uint8_t>(index));
            case DT_UINT16:
                return std::to_string(Get<uint16_t>(index));
            case DT_UINT32:
                return std::to_string(Get<uint32_t>(index));
            case DT_UINT64:
                return std::to_string(Get<uint64_t>(index));
            case DT_DOUBLE:
                return std::to_string(Get<double>(index));
            default:
                ASSERT(ExecuteOperationScene::INVALID_TENSOR_DTYPE, false);
                return "";
        }
    }

    void DumpElement(int index, ElementDump* dump) const
    {
        switch (GetDataType()) {
            case DT_INT8:
                dump->DumpElement(static_cast<int64_t>(Get<int8_t>(index)));
                break;
            case DT_BOOL:
                dump->DumpElement(static_cast<int64_t>(Get<int8_t>(index)));
                break;
            case DT_INT16:
                dump->DumpElement(static_cast<int64_t>(Get<int16_t>(index)));
                break;
            case DT_INT32:
                dump->DumpElement(static_cast<int64_t>(Get<int32_t>(index)));
                break;
            case DT_INT64:
                dump->DumpElement(static_cast<int64_t>(Get<int64_t>(index)));
                break;
            case DT_FP16:
                dump->DumpElement(static_cast<double>(Get<npu::tile_fwk::float16>(index)));
                break;
            case DT_FP32:
                dump->DumpElement(static_cast<double>(Get<float>(index)));
                break;
            case DT_BF16:
                dump->DumpElement(static_cast<double>(Get<npu::tile_fwk::bfloat16>(index)));
                break;
            case DT_UINT8:
                dump->DumpElement(static_cast<uint64_t>(Get<uint8_t>(index)));
                break;
            case DT_UINT16:
                dump->DumpElement(static_cast<uint64_t>(Get<uint16_t>(index)));
                break;
            case DT_UINT32:
                dump->DumpElement(static_cast<uint64_t>(Get<uint32_t>(index)));
                break;
            case DT_UINT64:
                dump->DumpElement(static_cast<uint64_t>(Get<uint64_t>(index)));
                break;
            case DT_DOUBLE:
                dump->DumpElement(static_cast<double>(Get<double>(index)));
                break;
            default:
                ASSERT(ExecuteOperationScene::INVALID_TENSOR_DTYPE, false);
        }
    }

    template <typename T>
    static std::shared_ptr<RawTensorData> CreateConstantTensor(const Tensor& t, T value)
    {
        auto tensorData = std::make_shared<RawTensorData>(t.GetDataType(), t.GetShape());

        T* data = reinterpret_cast<T*>(tensorData->data());
        ASSERT(ExecuteOperationScene::INVALID_TENSOR_DTYPE, sizeof(T) == tensorData->GetElementSize())
            << "ConstantTensor's dtype and value's type don't match!";
        for (size_t i = 0; i < tensorData->nelem; i++) {
            data[i] = value;
        }
        return tensorData;
    }

    template <typename T>
    static std::shared_ptr<RawTensorData> CreateTensor(const Tensor& t, const std::vector<T>& values)
    {
        auto tensorData = std::make_shared<RawTensorData>(t.GetDataType(), t.GetShape());
        T* data = reinterpret_cast<T*>(tensorData->data());
        ASSERT(ExecuteOperationScene::INVALID_TENSOR_DTYPE, sizeof(T) == tensorData->GetElementSize())
            << "CreateTensor's dtype and value's type don't match!";
        StringUtils::DataCopy(data, tensorData->GetDataSize(), values.data(), values.size() * sizeof(T));
        return tensorData;
    }

    template <typename T>
    static std::shared_ptr<RawTensorData> CreateConstantTensorData(const Shape& shape, DataType dType, T value)
    {
        auto tensorData = std::make_shared<RawTensorData>(dType, shape);

        T* data = reinterpret_cast<T*>(tensorData->data());
        ASSERT(ExecuteOperationScene::INVALID_TENSOR_DTYPE, sizeof(T) == tensorData->GetElementSize())
            << "ConstantTensor's dtype and value's type don't match!";
        for (size_t i = 0; i < tensorData->nelem; i++) {
            data[i] = value;
        }
        return tensorData;
    }

    template <typename T>
    static std::shared_ptr<RawTensorData> CreateTensorData(const Shape& shape, DataType dType,
                                                           const std::vector<T>& values)
    {
        auto tensorData = std::make_shared<RawTensorData>(dType, shape);
        T* data = reinterpret_cast<T*>(tensorData->data());
        ASSERT(ExecuteOperationScene::INVALID_TENSOR_DTYPE, sizeof(T) == tensorData->GetElementSize())
            << "CreateTensor's dtype and value's type don't match!";
        StringUtils::DataCopy(data, tensorData->GetDataSize(), values.data(), values.size() * sizeof(T));
        return tensorData;
    }

    static std::shared_ptr<RawTensorData> CreateTensor(DataType dtype, const std::vector<int64_t>& shape, uint8_t* data)
    {
        auto tensorData = std::make_shared<RawTensorData>(dtype, shape);
        StringUtils::DataCopy(tensorData->data(), tensorData->GetDataSize(), data, tensorData->GetDataSize());
        return tensorData;
    }

    static std::shared_ptr<RawTensorData> CreateTensorZero(const Tensor& t)
    {
        auto tensorData = std::make_shared<RawTensorData>(t.GetDataType(), t.GetShape());

        uint8_t* data = reinterpret_cast<uint8_t*>(tensorData->data());
        StringUtils::DataSet(data, tensorData->GetDataSize(), 0, tensorData->GetDataSize());
        return tensorData;
    }

    void SetDevPtr(uint8_t* ptr) { devPtr_ = ptr; }
    uint8_t* GetDevPtr() { return devPtr_; }

    void ToFile(const std::string& path) const
    {
        std::ofstream ofile(path, std::ios::out | std::ios::binary);
        if (!ofile) {
            INTERPRETER_LOGE_FULL(OpDumpScene::DUMP_OPEN_FILE_FAILED, "open file %s failed!!!!", path.c_str());
        }
        ofile.write(reinterpret_cast<const char*>(data_->data()), data_->size());
        ofile.close();
    }

    size_t GetDataSize() const
    {
        if (elemSize_ > 0) {
            return static_cast<size_t>(nelem) * static_cast<size_t>(elemSize_);
        }
        if (IsFp4PackedDtype(GetDataType())) {
            auto packedShape = PackedShapeFromLogical(shape_, GetDataType());
            return static_cast<size_t>(Numel(packedShape));
        }
        return static_cast<size_t>((nelem + 1) / 0x2);
    }

    size_t GetShmOffset() const { return shmOffset_; }

    void SetShmOffset(size_t offset) { shmOffset_ = offset; }

    bool IsShmTensor() { return isShmTensor_; }

    void SetAsShmTensor() { isShmTensor_ = true; }

    void UpdateShapeAndDtype(DataType dataType, const std::vector<int64_t>& shape)
    {
        dataType_ = dataType;
        shape_ = shape;
        stride_ = ShapeToStride(shape);
        nelem = Numel(shape);
        elemSize_ = GetDataSize(dataType);
    }

private:
    uint8_t* devPtr_{nullptr};
    std::shared_ptr<StorageData> data_;
    DataType dataType_;
    Shape shape_;
    Stride stride_;
    size_t nelem;
    // Signed: GetDataSize(DataType) uses -1 for sub-byte dtypes; storing as size_t wrapped to huge
    // and broke vector allocation in SetVerifyData / RawTensorData::CreateTensor.
    int elemSize_;
    bool isShmTensor_ = false;
    size_t shmOffset_ = 0;
};

using RawTensorDataPtr = std::shared_ptr<RawTensorData>;

struct LogicalTensorData {
    LogicalTensorData() = default;

    LogicalTensorData(RawTensorDataPtr data)
        : LogicalTensorData(data, data->GetShape(), data->GetShape(), std::vector<int64_t>(data->GetShape().size(), 0))
    {}

    LogicalTensorData(RawTensorDataPtr data, const std::vector<int64_t>& shape, const std::vector<int64_t>& validShape,
                      const std::vector<int64_t>& offset)
        : data_(data),
          shape_(shape),
          validShape_(validShape),
          offset_(offset),
          stride_(RawTensorData::ShapeToStride(shape)),
          size_(shape_[0] * stride_[0]),
          isSpilled_(false)
    {
        if (validShape.empty()) {
            validShape_ = shape;
        }
        for (size_t i = 0; i < shape.size(); i++) {
            if (shape[i] >= 0) {
                validShape_[i] = std::min(shape[i], validShape_[i]);
            }
        }
    }

    LogicalTensorData(RawTensorDataPtr data, const std::vector<int64_t>& shape, const std::vector<int64_t>& offset)
        : LogicalTensorData(data, shape, shape, offset)
    {}

    const RawTensorDataPtr& GetData() const { return data_; }
    RawTensorDataPtr GetData() { return data_; }

    const Shape& GetShape() const { return shape_; }
    int64_t GetShape(int axis) const
    {
        if (axis < 0)
            axis += shape_.size();
        return shape_[axis];
    }
    const Shape& GetValidShape() const { return validShape_; }
    const Stride& GetStride() const { return stride_; }
    int64_t GetStride(int axis) const { return stride_[axis]; }
    const Offset& GetOffset() const { return offset_; }
    bool GetIsSpilled() const { return isSpilled_; }
    void SetIsSpilled(bool isSpilled) { isSpilled_ = isSpilled; }
    int GetSize() const { return size_; }
    DataType GetDataType() const { return GetData()->GetDataType(); }

    void UpdateValidShape(std::vector<int64_t> shape, const std::vector<int64_t>& offset)
    {
        validShape_ = shape;
        offset_ = offset;
    }
    int64_t GetStorageOffset() const
    {
        auto& strides = data_->GetStride();
        int64_t offset = 0;
        for (size_t i = 0; i < strides.size(); i++) {
            offset += strides[i] * offset_[i];
        }
        return offset;
    }

    size_t GetShmStorageOffset() { return data_->GetShmOffset(); }

    bool IsShmTensor() { return data_->IsShmTensor(); }

    int ViewIndexToDataIndex(int viewIndex) const
    {
        int offset[0x8];
        for (size_t i = 0; i < GetShape().size(); i++) {
            offset[i] = viewIndex / stride_[i];
            viewIndex %= stride_[i];
        }
        for (size_t i = 0; i < GetShape().size(); i++) {
            offset[i] += offset_[i];
        }
        int dataIndex = 0;
        for (size_t i = 0; i < GetShape().size(); i++) {
            dataIndex += offset[i] * GetData()->GetStride()[i];
        }
        return dataIndex;
    }

    template <typename T>
    const T& Get(int index) const
    {
        return GetData()->Get<T>(index);
    }
    template <typename T>
    T& Get(int index)
    {
        return GetData()->Get<T>(index);
    }

    Element GetElement(int index) const { return GetData()->GetElement(ViewIndexToDataIndex(index)); }
    std::string DumpElement(int index) const { return GetData()->DumpElement(ViewIndexToDataIndex(index)); }
    void DumpElement(int index, ElementDump* dump) const
    {
        return GetData()->DumpElement(ViewIndexToDataIndex(index), dump);
    }
    std::string DumpType() const
    {
        std::ostringstream oss;
        oss << "<";
        for (size_t k = 0; k < GetShape().size(); k++) {
            oss << GetShape()[k] << "x";
        }
        oss << DataType2String(GetDataType(), true) << "/";
        for (size_t k = 0; k < validShape_.size(); k++) {
            oss << validShape_[k] << "x";
        }
        oss << DataType2String(GetDataType(), true) << ">";
        return oss.str();
    }

    static std::shared_ptr<LogicalTensorData> CreateMove(RawTensorData&& data)
    {
        auto tensorData = std::make_shared<RawTensorData>(std::move(data));
        return std::make_shared<LogicalTensorData>(tensorData);
    }

    static std::shared_ptr<LogicalTensorData> Create(const RawTensorData& data)
    {
        auto tensorData = std::make_shared<RawTensorData>(data);
        return std::make_shared<LogicalTensorData>(tensorData);
    }

    static std::shared_ptr<LogicalTensorData> CreateEmpty(DataType dataType, const std::vector<int64_t>& shape,
                                                          const std::vector<int64_t>& validShape,
                                                          const std::vector<int64_t>& rawShape)
    {
        auto tensorData = std::make_shared<RawTensorData>(dataType, rawShape);
        return std::make_shared<LogicalTensorData>(tensorData, shape, validShape,
                                                   std::vector<int64_t>(shape.size(), 0));
    }

    std::shared_ptr<LogicalTensorData> View(const std::vector<int64_t>& viewShape,
                                            const std::vector<int64_t>& viewOffset)
    {
        return std::make_shared<LogicalTensorData>(GetData(), viewShape, viewShape, viewOffset);
    }

    std::shared_ptr<LogicalTensorData> DeepCopy() const
    {
        auto tensorData = std::make_shared<RawTensorData>(*data_);
        return std::make_shared<LogicalTensorData>(tensorData, shape_, validShape_, offset_);
    }

    std::string Dump(const std::vector<ElementDump>* elementDumpList) const
    {
        constexpr int INDENT_TWO = 2;
        return DumpData(INDENT_TWO, elementDumpList);
    }

    std::string ToString(const PrintOptions* options = nullptr) const;

    void Save(const std::string& filepath) const;
    void SaveFile(const char* filepath) const;
    static std::shared_ptr<LogicalTensorData> Load(const std::string& filepath);

    void SetAxisCombine(bool val) { axisCombine = val; }
    bool IsAxisCombine() const { return axisCombine; }

private:
    template <typename T>
    void HandleSave(FILE* fdata, int totalSize, int rowSize) const
    {
        if (fdata == nullptr) {
            ASSERT(OpDumpScene::DUMP_OPEN_FILE_FAILED, false);
        }
        for (int k = 0; k < totalSize / rowSize; k++) {
            size_t result = fwrite(&Get<T>(k), sizeof(T), rowSize, fdata);
            if (result != static_cast<size_t>(rowSize)) {
                ASSERT(OpDumpScene::DUMP_WRITE_FILE_FAILED, false);
            }
        }
    }

    std::string DumpRange(int idxBegin, int idxEnd, const std::vector<ElementDump>* elementDumpList) const;
    std::string DumpCoord(int row) const;
    std::string DumpData(int indent, const std::vector<ElementDump>* elementDumpList) const;

private:
    std::shared_ptr<RawTensorData> data_;
    Shape shape_;
    Shape validShape_;
    Offset offset_;
    Stride stride_;
    int64_t size_;
    bool isSpilled_;
    bool axisCombine{false};
};

using LogicalTensorDataPtr = std::shared_ptr<LogicalTensorData>;

template <>
inline std::shared_ptr<RawTensorData> RawTensorData::CreateTensor<uint8_t>(const Tensor& t,
                                                                           const std::vector<uint8_t>& values)
{
    auto tensorData = std::make_shared<RawTensorData>(t.GetDataType(), t.GetShape());
    uint8_t* data = reinterpret_cast<uint8_t*>(tensorData->data());
    StringUtils::DataCopy(data, tensorData->GetDataSize(), values.data(), values.size());
    return tensorData;
}

struct ProgramData {
    std::vector<RawTensorDataPtr> inputDataList_;
    std::vector<RawTensorDataPtr> outputDataList_;
    std::vector<RawTensorDataPtr> goldenDataList_;

    const std::vector<RawTensorDataPtr>& GetInputDataList() const { return inputDataList_; }
    std::vector<RawTensorDataPtr>& GetInputDataList() { return inputDataList_; }
    RawTensorDataPtr GetInputData(int idx) { return inputDataList_[idx]; }

    const std::vector<RawTensorDataPtr>& GetOutputDataList() const { return outputDataList_; }
    std::vector<RawTensorDataPtr>& GetOutputDataList() { return outputDataList_; }
    RawTensorDataPtr GetOutputData(int idx) { return outputDataList_[idx]; }

    const std::vector<RawTensorDataPtr> GetGoldenDataList() const { return goldenDataList_; }
    std::vector<RawTensorDataPtr> GetGoldenDataList() { return goldenDataList_; }
    RawTensorDataPtr GetGoldenData(int idx) { return goldenDataList_[idx]; }

    void AppendInput(RawTensorDataPtr data) { inputDataList_.push_back(data); }
    void AppendInputs(const std::vector<RawTensorDataPtr>& dataList)
    {
        for (const auto& data : dataList) {
            AppendInput(data);
        }
    }

    void AppendOutput(RawTensorDataPtr data) { outputDataList_.push_back(data); }
    void AppendOutputs(const std::vector<RawTensorDataPtr>& dataList)
    {
        for (const auto& data : dataList) {
            AppendOutput(data);
        }
    }

    void AppendGolden(RawTensorDataPtr data) { goldenDataList_.push_back(data); }
    void AppendGoldens(const std::vector<RawTensorDataPtr>& dataList)
    {
        for (const auto& data : dataList) {
            AppendGolden(data);
        }
    }

    void PrepareData(const std::vector<RawTensorDataPtr> inputDataList,
                     const std::vector<RawTensorDataPtr> outputDataList,
                     const std::vector<RawTensorDataPtr> goldenDataList)
    {
        AppendInputs(inputDataList);
        AppendOutputs(outputDataList);
        AppendGoldens(goldenDataList);
    }

    void CopyTo(std::vector<std::shared_ptr<LogicalTensorData>>& dataViewList,
                const std::vector<RawTensorDataPtr>& dataList)
    {
        for (auto data : dataList) {
            if (data) {
                auto shape = data->GetShape();
                dataViewList.push_back(
                    std::make_shared<LogicalTensorData>(data, shape, shape, std::vector<int64_t>(shape.size(), 0)));
            } else {
                dataViewList.push_back(nullptr);
            }
        }
    }

    void CopyToInputDataViewList(std::vector<std::shared_ptr<LogicalTensorData>>& inputDataViewList)
    {
        CopyTo(inputDataViewList, inputDataList_);
    }

    void CopyToOutputDataViewList(std::vector<std::shared_ptr<LogicalTensorData>>& outputDataViewList)
    {
        CopyTo(outputDataViewList, outputDataList_);
    }

    void CopyToGoldenDataViewList(std::vector<std::shared_ptr<LogicalTensorData>>& goldenDataViewList)
    {
        CopyTo(goldenDataViewList, goldenDataList_);
    }

    void Reset()
    {
        inputDataList_.clear();
        outputDataList_.clear();
        goldenDataList_.clear();
    }

    static ProgramData& GetInstance();
};
} // namespace npu::tile_fwk
