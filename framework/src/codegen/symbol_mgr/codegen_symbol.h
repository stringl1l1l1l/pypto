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
 * \file codegen_symbol.h
 * \brief
 */

#pragma once

#include <deque>
#include <functional>
#include <map>
#include <sstream>
#include <tuple>
#include <cstdint>
#include <string>
#include <unordered_map>

#include "tilefwk/error.h"
#include "interface/utils/common.h"
#include "interface/tensor/logical_tensor.h"
#include "codegen/utils/codegen_utils.h"
#include "tilefwk/error_code.h"
#include "tilefwk/platform.h"
#include "symbol_id_gen.h"

namespace npu::tile_fwk {
const std::string TILE_TENSOR = "TileTensor";
const std::string LAYOUT = "Layout";
const std::string SCOPE_NAMESPACE = "Hardware";
const std::string DIM = "Dim";
const std::string COORD = "Coord";
using BufferType = enum OperandType;

// A2 (DAV_2201) 编译器在 __CCE_AICORE__ 下不定义 fp8 原生类型（hifloat8_t/float8_e4m3_t 等仅存在于
// V310+ 的 typedef 或 proxy 结构）。fp8 属 1 字节类型，Transpose 等纯数据搬移场景（MOVEOUT 按字节 DMA、
// TTRANS 走 b8 路径）按 uint8_t 生成内核缓冲类型即可，字节精度无损。
inline const char* CodeGenDataTypeStr(DataType dtype)
{
    if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_2201) {
        switch (dtype) {
            case DT_FP8:
            case DT_HF8:
            case DT_FP8E4M3:
            case DT_FP8E5M2:
            case DT_FP8E8M0:
                return "uint8_t";
            default:
                break;
        }
    }
    return DataType2CCEStr(dtype);
}
using AllocKey = std::tuple<BufferType, int64_t /*RangeStart*/, int64_t /*RangeEnd*/>;

struct AddrAlloc {
    AllocKey allocKey;
    std::string allocNormal;
    std::string allocForTileTensor;
};

struct TileTensorShape {
    bool isInLoop{false};
    std::vector<int64_t> shape;
    std::vector<int64_t> rawShape;
    std::vector<SymbolicScalar> dynamicValidShape;
};

inline std::string GetLayoutType(BufferType bufType, int dim, bool isConst = false)
{
    std::string prefix = bufType == BUF_DDR ? "Dyn" : isConst ? "Static" : "Local";
    std::ostringstream ss;
    ss << prefix << LAYOUT << dim << DIM;
    return ss.str();
}

// e.g.
// UBTileTensorFP32Dim2 ubTile_0((__ubuf__ float*)UB_S0_E16384, DimLayout2(Shape<int, int>(sym_18_dim_0, sym_18_dim_1),
// Stride<int, int>(64, 1)));
struct TileTensor {
    bool isConstant;
    bool isInLoop{false};
    int magic; // tensor magic numbuer
    int dim;
    DataType dtype;
    BufferType bufType;
    std::string bufVar;
    std::string usingType;
    std::string tensorName;         // e.g. "ubTile_0"
    std::vector<std::string> shape; // valid shape
    std::vector<std::string> stride;
    std::vector<int64_t> rawShape;
    std::vector<int64_t> localBufOffset;

    /*  e.g.
        ((__ubuf__ float*)UB_S0_E16384,
        Layout2Dim(Shape2Dim<int, int>(sym_18_dim_0, sym_18_dim_1), Stride2Dim<int, int>(64, 1)));
    */
    std::string GenInitParam() const
    {
        std::ostringstream oss;
        std::vector<std::string> params;
        // ddr: e.g. (__gm__ float*)GET_PARAM_ADDR(...)
        // local: e.g. (uint64_t)UB_S0_E16384
        oss << "(";
        if (bufType == BUF_DDR) {
            oss << OPERAND_TYPE_TO_ADDR_TYPE.at(bufType) << " " << CodeGenDataTypeStr(dtype) << "*)" << bufVar;
        } else {
            // cast local buffer pointer to uint64_t to adapt TileTensor mode
            oss << "uint64_t)";
            int64_t linearOffset{0};
            if (!localBufOffset.empty() && !isInLoop) {
                // only calc linear offset in the outermost loop, tensor in loop use base addr from tensor out of loop
                linearOffset = CalcLinearOffset(rawShape, localBufOffset);
            }
            if (linearOffset != 0 && bufType != BUF_L1) {
                // append linear offset, e.g. UBTileTensorFP32Dim2_1 ubTensor_1((uint64_t)((float *)UB_S0_E4096 + 32))
                oss << "((" << CodeGenDataTypeStr(dtype) << " *)" << bufVar << " + " << linearOffset << ")";
            } else {
                oss << bufVar;
            }
        }

        if ((isConstant) && bufType != BUF_DDR) {
            return "(" + oss.str() + ")";
        }
        params.emplace_back(oss.str());
        oss.str("");
        // ddr: e.g. DynLayout2Dim(Shape2Dim<int, int>(sym_18_dim_0, sym_18_dim_1), Stride2Dim<int, int>(64, 1)));
        // local: e.g. Shape2Dim(sym_18_dim_0, sym_18_dim_1));
        if (bufType == BUF_DDR) {
            oss << GetLayoutType(bufType, dim);
        }
        oss << "(" << GenShapeParam();
        if (bufType == BUF_DDR) {
            oss << ", " << GenStrideParam();
        }
        oss << ")";
        params.emplace_back(oss.str());
        return WrapParamByParentheses(params);
    }

    std::string ToString() const
    {
        std::ostringstream oss;
        oss << usingType << " " << tensorName << GenInitParam() << STMT_END;
        return oss.str();
    }

private:
    std::string GenLayoutParam(const std::string& paramName, const std::vector<std::string>& paramValue) const
    {
        std::ostringstream oss;
        oss << paramName << dim << DIM;
        oss << WrapParamByParentheses(paramValue);
        return oss.str();
    }
    std::string GenShapeParam() const { return GenLayoutParam("Shape", shape); }
    std::string GenStrideParam() const { return GenLayoutParam("Stride", stride); }
};

struct TileTensorKey {
    int dim;
    DataType dtype;
    std::string bufVar;
    std::vector<std::string> shape;
    std::vector<int64_t> rawShape;
    std::vector<int64_t> localBufOffset;

    bool operator==(const TileTensorKey& other) const
    {
        return dim == other.dim && bufVar == other.bufVar && shape == other.shape && dtype == other.dtype &&
               localBufOffset == other.localBufOffset && rawShape == other.rawShape;
    }

    std::string ToString() const
    {
        std::ostringstream oss;
        oss << "dim=" << dim << ", dtype=" << ToUnderlying(dtype) << ", bufVar=" << bufVar
            << ", shape=" << IntVecToStr(shape) << ", rawShape=" << IntVecToStr(rawShape)
            << ", localBufOffset=" << IntVecToStr(localBufOffset);
        return oss.str();
    }
};

struct TileTensorKeyHash {
    std::size_t operator()(const TileTensorKey& key) const noexcept
    {
        std::size_t seed = 0;
        HashCombine(seed, key.dim);
        HashCombine(seed, key.bufVar);
        HashCombine(seed, ToUnderlying(key.dtype));
        for (const auto& s : key.shape) {
            HashCombine(seed, s);
        }
        for (const auto& s : key.rawShape) {
            HashCombine(seed, s);
        }
        for (const auto& s : key.localBufOffset) {
            HashCombine(seed, s);
        }
        return seed;
    }
};

using TileTensorMagicKey = std::pair<int, int>; // <tensor magic, op magic>

struct TileTensorMagicKeyHash {
    std::size_t operator()(const TileTensorMagicKey& key) const noexcept
    {
        std::size_t seed = 0;
        HashCombine(seed, key.first);
        HashCombine(seed, key.second);
        return seed;
    }
};

struct TileTensorUsing {
    bool isConstant;
    DataType dtype;
    BufferType bufType;
    int dim;
    std::vector<int64_t> shape; // only used for static shape
    std::vector<int64_t> rawShape;

    bool operator==(const TileTensorUsing& other) const
    {
        if (dtype != other.dtype || bufType != other.bufType || dim != other.dim) {
            return false;
        }
        if (bufType == BUF_DDR) {
            return true;
        }
        if (isConstant != other.isConstant) {
            return false;
        }
        bool baseCompare = rawShape == other.rawShape;
        return isConstant ? baseCompare && shape == other.shape : baseCompare;
    }

    std::string GenName() const
    {
        std::ostringstream oss;
        oss << BUFFER_TYPE_TO_PREFIX.at(bufType) << TILE_TENSOR << DataType2String(dtype, true) << DIM << dim << "_";
        return oss.str();
    }

    // dynamic shape: e.g. "TileTensor<__gm__ float, DynLayout4Dim, Hardware::GM>"
    // static shape: e.g. "TileTensor<float, LocalLayout4Dim<16, 16>, Hardware::UB>"
    std::string ToString() const
    {
        std::ostringstream ss;
        ss << TILE_TENSOR << "<";
        if (bufType == BUF_DDR) {
            ss << GetAddrTypeByOperandType(bufType) << " ";
        }
        ss << CodeGenDataTypeStr(dtype) << ", ";
        ss << GetLayoutType(bufType, dim, isConstant);
        if (bufType != BUF_DDR) {
            ss << GetLayoutParams();
        }
        ss << ", " << SCOPE_NAMESPACE << "::" << BUFFER_TYPE_TO_PREFIX.at(bufType) << ">;\n";
        return ss.str();
    }

private:
    constexpr static int SHAPE_KIND = 2; // origin shape; raw shape
    std::string GetLayoutParams() const
    {
        std::vector<int64_t> params;
        params.reserve(dim * SHAPE_KIND);
        if (isConstant) {
            params.insert(params.end(), shape.begin(), shape.end());
        }
        params.insert(params.end(), rawShape.begin(), rawShape.end());
        return WrapParamByAngleBrackets(params);
    }
};

class SymbolManager {
public:
    SymbolManager() = default;
    virtual ~SymbolManager() = default;

    using AllocRecord = std::pair<uint64_t /*AllocaAddr*/, unsigned /*AllocaSize*/>;

    virtual std::string QueryVariableName(const AllocKey& key);
    std::string QueryVariableNameTileTensor(const AllocKey& key);
    std::string QueryVarNameByTensorMagic(int magic, bool isTileTensor = false);
    SymbolManager(SymbolManager& other) = delete;

    void operator=(const SymbolManager& other) = delete;

    bool BindAddrWithVariableName(const AllocKey& key, const std::string& varName,
                                  const std::string& varNameTileTensor);

    void AddToTensorMap(int magicNum, const std::shared_ptr<LogicalTensor>& tensor)
    {
        auto res = tensorMap_.insert({magicNum, tensor});
        if (!res.second) {
            ASSERT(GenCodeErr::TENSOR_MAGIC_CONFLICT, tensor == tensorMap_[magicNum])
                << "tensor magic : " << magicNum
                << " is conflicted!!!\ninsert tensor key: " << FormatAllocKey(CreateAllocKey(tensor))
                << "\ntensor dump info -- " << tensor->Dump()
                << "\nexisted tensor key: " << FormatAllocKey(CreateAllocKey(tensorMap_[magicNum]))
                << "\ntensor dump info -- " << tensorMap_[magicNum]->Dump();
        }
    }

    static std::string FormatAllocKey(const AllocKey& key);

    std::string AddTileTensorUsing(const TileTensorUsing& tileTensorUsing);
    std::string AddTileTensor(int opMagic, const TileTensor& tileTensor);
    const TileTensor* QueryTileTensorByMagic(int tensorMagic, int opMagic) const;
    const TileTensor* QueryTileTensorInLoopByMagic(int tensorMagic, int opMagic) const;
    void InsertTensorNameInLoopToFullDim(const std::string& tensorName, const std::string& fullDimTensorName);
    std::string QueryTileTensorFullDimByTensorInLoop(const std::string& tensorName);

    void GenUsingList(std::ostringstream& oss);
    std::string GenTileTensorDefList();
    std::string GenNewTileTensorDefs();

    std::string GenTensorName(BufferType bufType)
    {
        return BUFFER_TYPE_TO_PREFIX_LC.at(bufType) + "Tensor_" +
               std::to_string(idGenMgr_.NewId<SymbolIdType::CG_VAR_NAME>());
    }

    void OutForLoop()
    {
        tileTensorByMagicInLoop_.clear();
        tensorNameInLoopToFullDim_.clear();
    }

    void AddTensorUseNormalAlloc(const std::shared_ptr<LogicalTensor>& tensor)
    {
        tensorUseNormalAlloc_.insert(CreateAllocKey(tensor));
    }
    void RecordTensorAddrAlloc(const AddrAlloc& addrAlloc) { addrAllocs_.emplace_back(addrAlloc); }
    void PrintAddrAllocs(std::ostringstream& oss) const;

private:
    std::string GenTensorUsingName(const TileTensorUsing& tileTensorUsing)
    {
        return tileTensorUsing.GenName() + std::to_string(idGenMgr_.NewId<SymbolIdType::CG_USING_NAME>());
    }

    TileTensorKey BuildTileTensorKey(const TileTensor& tileTensor) const;
    std::shared_ptr<LogicalTensor> GetTensorByMagic(int magicNum) const;
    AllocKey CreateAllocKey(const std::shared_ptr<LogicalTensor>& tensor) const;
    AllocKey CreateAllocKey(int tensorMagicNum) const;
    std::string FindUsingName(const TileTensorUsing& tileTensorUsing) const;

    // <AllocKey, buffer variable name>
    std::map<AllocKey, std::string> key2VariableName_;
    // <AllocKey, buffer variable name of TileTensor mode>
    std::map<AllocKey, std::string> key2VariableNameTileTensor_;
    // <tensor magic, LogicalTensor>
    std::unordered_map<int, std::shared_ptr<LogicalTensor>> tensorMap_;
    // Own TileTensor objects and keep their addresses stable for secondary indexes.
    std::deque<TileTensor> tileTensorStorage_;
    // Use explicit semantic key for dedup instead of treating hash as identity.
    std::unordered_map<TileTensorKey, std::reference_wrapper<const TileTensor>, TileTensorKeyHash> tileTensorByKey_;
    // <tensor magic, op magic> -> TileTensor
    std::unordered_map<TileTensorMagicKey, std::reference_wrapper<const TileTensor>, TileTensorMagicKeyHash>
        tileTensorByMagic_;
    std::unordered_map<TileTensorMagicKey, std::reference_wrapper<const TileTensor>, TileTensorMagicKeyHash>
        tileTensorByMagicInLoop_;
    // <tensorName in for loop, tensorName with full dim out of loop>
    // both key and value are from same tile operation
    std::unordered_map<std::string, std::string> tensorNameInLoopToFullDim_;
    // <using type, TileTensorUsing>
    std::unordered_map<std::string, TileTensorUsing> tileTensorUsing_;
    SymbolIdGenMgr idGenMgr_;
    size_t tileTensorOutputIdx_{0};
    std::vector<AddrAlloc> addrAllocs_;
    std::set<AllocKey> tensorUseNormalAlloc_;
};
} // namespace npu::tile_fwk
