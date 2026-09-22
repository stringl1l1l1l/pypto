/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conv_utils.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/utils/common.h"
#include "tilefwk/platform.h"

namespace npu {
namespace tile_fwk {
namespace Conv {

int64_t ConvComputeHo(const Tensor& inputTensor, const Tensor& weightTensor, const ConvAttrParam& attrParam)
{
    if (attrParam.isConv1D) {
        return 1;
    }
    uint32_t indexH = attrParam.isConv3D ? NCDHW_H_IDX : NCHW_H_IDX;
    std::vector<int64_t> strides = attrParam.strides;
    int64_t strideH = strides[PAD_STRIDE_H];
    if (strideH == 0) {
        return 1;
    }
    std::vector<int64_t> paddings = attrParam.paddings;
    std::vector<int64_t> dilations = attrParam.dilations;
    int64_t padTop = paddings[PAD_TOP_INDEX];
    int64_t padBottom = paddings[PAD_BOTTOM_INDEX];
    int64_t dilationH = dilations[PAD_STRIDE_H];
    int64_t hin = inputTensor.GetShape()[indexH];
    int64_t kh = weightTensor.GetShape()[indexH];
    int64_t cmpHo = (hin + padTop + padBottom - dilationH * (kh - 1) - 1) / strideH + 1;
    return cmpHo;
}

int64_t ConvComputeWo(const Tensor& inputTensor, const Tensor& weightTensor, const ConvAttrParam& attrParam)
{
    uint32_t indexW = attrParam.isConv3D ? NCDHW_W_IDX : (attrParam.isConv1D ? NCHW_H_IDX : NCHW_W_IDX);
    uint32_t indexAttr = attrParam.isConv1D ? PAD_STRIDE_H : PAD_STRIDE_W;

    std::vector<int64_t> strides = attrParam.strides;
    int64_t strideW = strides[indexAttr];
    if (strideW == 0) {
        return 1;
    }
    std::vector<int64_t> paddings = attrParam.paddings;
    std::vector<int64_t> dilations = attrParam.dilations;
    int64_t dilationW = dilations[indexAttr];
    int64_t padLeft = paddings[2 * indexAttr];
    int64_t padRight = paddings[2 * indexAttr + 1];
    int64_t win = inputTensor.GetShape()[indexW];
    int64_t kw = weightTensor.GetShape()[indexW];
    int64_t cmpWo = (win + padLeft + padRight - dilationW * (kw - 1) - 1) / strideW + 1;
    return cmpWo;
}

int64_t ConvComputeDo(const Tensor& inputTensor, const Tensor& weightTensor, const ConvAttrParam& attrParam)
{
    std::vector<int64_t> strides = attrParam.strides;
    int64_t strideD = strides[PAD_STRIDE_D];
    if (strideD == 0) {
        return 1;
    }
    std::vector<int64_t> paddings = attrParam.paddings;
    std::vector<int64_t> dilations = attrParam.dilations;
    int64_t padHead = paddings[PAD_HEAD_INDEX];
    int64_t padTail = paddings[PAD_TAIL_INDEX];
    int64_t dilationD = dilations[PAD_STRIDE_D];
    int64_t din = inputTensor.GetShape()[NCDHW_D_IDX];
    int64_t kd = weightTensor.GetShape()[NCDHW_D_IDX];
    int64_t cmpDo = (din + padHead + padTail - dilationD * (kd - 1) - 1) / strideD + 1;
    return cmpDo;
}

SymbolicScalar ConvComputeValidHo(const Tensor& inputTensor, const Tensor& weightTensor, const ConvAttrParam& attrParam)
{
    if (attrParam.isConv1D) {
        return SymbolicScalar(1);
    }
    uint32_t indexH = attrParam.isConv3D ? NCDHW_H_IDX : NCHW_H_IDX;
    std::vector<int64_t> strides = attrParam.strides;
    int64_t strideH = strides[PAD_STRIDE_H];
    if (strideH == 0) {
        return SymbolicScalar(1);
    }
    std::vector<int64_t> paddings = attrParam.paddings;
    std::vector<int64_t> dilations = attrParam.dilations;
    int64_t padTop = paddings[PAD_TOP_INDEX];
    int64_t padBottom = paddings[PAD_BOTTOM_INDEX];
    int64_t dilationH = dilations[PAD_STRIDE_H];
    SymbolicScalar hin = inputTensor.GetValidShape()[indexH];
    SymbolicScalar kh = weightTensor.GetValidShape()[indexH];
    SymbolicScalar cmpHo = (hin + padTop + padBottom - dilationH * (kh - 1) - 1) / strideH + 1;
    return cmpHo;
}

SymbolicScalar ConvComputeValidWo(const Tensor& inputTensor, const Tensor& weightTensor, const ConvAttrParam& attrParam)
{
    uint32_t indexW = attrParam.isConv3D ? NCDHW_W_IDX : (attrParam.isConv1D ? NCHW_H_IDX : NCHW_W_IDX);
    uint32_t indexAttr = attrParam.isConv1D ? PAD_STRIDE_H : PAD_STRIDE_W;

    std::vector<int64_t> strides = attrParam.strides;
    int64_t strideW = strides[indexAttr];
    if (strideW == 0) {
        return SymbolicScalar(1);
    }
    std::vector<int64_t> paddings = attrParam.paddings;
    std::vector<int64_t> dilations = attrParam.dilations;
    int64_t dilationW = dilations[indexAttr];
    int64_t padLeft = paddings[2 * indexAttr];
    int64_t padRight = paddings[2 * indexAttr + 1];
    SymbolicScalar win = inputTensor.GetValidShape()[indexW];
    SymbolicScalar kw = weightTensor.GetValidShape()[indexW];
    SymbolicScalar cmpWo = (win + padLeft + padRight - dilationW * (kw - 1) - 1) / strideW + 1;
    return cmpWo;
}

SymbolicScalar ConvComputeValidDo(const Tensor& inputTensor, const Tensor& weightTensor, const ConvAttrParam& attrParam)
{
    std::vector<int64_t> strides = attrParam.strides;
    int64_t strideD = strides[PAD_STRIDE_D];
    if (strideD == 0) {
        return SymbolicScalar(1);
    }
    std::vector<int64_t> paddings = attrParam.paddings;
    std::vector<int64_t> dilations = attrParam.dilations;
    int64_t padHead = paddings[PAD_HEAD_INDEX];
    int64_t padTail = paddings[PAD_TAIL_INDEX];
    int64_t dilationD = dilations[PAD_STRIDE_D];
    SymbolicScalar din = inputTensor.GetValidShape()[NCDHW_D_IDX];
    SymbolicScalar kd = weightTensor.GetValidShape()[NCDHW_D_IDX];
    SymbolicScalar cmpDo = (din + padHead + padTail - dilationD * (kd - 1) - 1) / strideD + 1;
    return cmpDo;
}

namespace {

void CheckValueRange(int64_t value, const std::string& name, int64_t min, int64_t max, const std::string& formula = "")
{
    std::ostringstream oss;
    oss << "Invalid " << name << ":" << value << ", expected range [" << min << "," << max << "].";
    if (!formula.empty()) {
        oss << "Formula: " << formula;
    }
    CHECK(ExternalError::OUT_OF_RANGE, value >= min && value <= max) << oss.str();
}

void CheckAlignment(int64_t value, int64_t alignment, const std::string& valueName)
{
    CHECK(ExternalError::INVALID_VAL, alignment != 0) << "Error in alignment check for " << valueName << ".";
    CHECK(ExternalError::INVALID_VAL, value % alignment == 0)
        << "Invalid " << valueName << ":" << value << ", requires " << alignment << "-element alignment.";
}

int64_t ConvAlignB(int64_t a, int64_t b)
{
    if (b == 0) {
        return 0;
    }
    return ((a + b - 1) / b) * b;
}

std::vector<int64_t> rotateVector(const std::vector<int64_t>& input, size_t shift)
{
    std::vector<int64_t> result = input;
    std::rotate(result.begin(), result.begin() + shift, result.end());
    return result;
}

void CheckOutputShape(const Tensor& inputTensor, const Tensor& weightTensor, const ConvAttrParam& attrParam)
{
    int64_t hOut = ConvComputeHo(inputTensor, weightTensor, attrParam);
    std::string hOutFormula = "hOut = (hin + 2 * pad_h - (kh - 1) * dilation_h - 1) / stride_h + 1";
    CheckValueRange(hOut, "hOut", NUM1, MAX_SIZE, hOutFormula);
    int64_t wOut = ConvComputeWo(inputTensor, weightTensor, attrParam);
    std::string wOutFormula = "wOut = (win + 2 * pad_w - (kw - 1) * dilation_w - 1) / stride_w + 1";
    CheckValueRange(wOut, "wOut", NUM1, MAX_SIZE, wOutFormula);
    if (attrParam.isConv3D) {
        int64_t dOut = ConvComputeDo(inputTensor, weightTensor, attrParam);
        std::string dOutFormula = "dOut = (din + 2 * pad_d - (kd - 1) * dilation_d - 1) / stride_d + 1";
        CheckValueRange(dOut, "dOut", NUM1, MAX_SIZE, dOutFormula);
    }
}

void CheckHowoTile(const Tensor& inputTensor, const Tensor& weightTensor, const ConvAttrParam& attrParam)
{
    auto& convTile = TileShape::Current().GetConvTile();
    int64_t tileHout = convTile.tileL1Info.tileHout;
    int64_t tileWout = convTile.tileL1Info.tileWout;
    int64_t hOut = ConvComputeHo(inputTensor, weightTensor, attrParam);
    int64_t wOut = ConvComputeWo(inputTensor, weightTensor, attrParam);
    if (wOut % NUM16 != 0) {
        CHECK(ExternalError::INVALID_VAL, tileHout == 1) << "When wOut is not a multiple of 16, tileHout should be 1.";
    }
    CheckValueRange(tileHout, "tileHout", NUM1, hOut);
    CheckValueRange(tileWout, "tileWout", NUM1, ConvAlignB(wOut, NUM16));
    CheckAlignment(tileWout, NUM16, "tileWout");
}

void ValidateL0Constraint(int64_t tile1, int64_t tile2, int64_t tile3, size_t dtypeSize, size_t cacheSize,
                          const std::string& cacheName, const std::string& dim1Name, const std::string& dim2Name,
                          const std::string& dim3Name)
{
    ASSERT(ConvOperationError::OVER_BUFFER_LIMIT, tile1 * tile2 * tile3 * dtypeSize <= cacheSize)
        << "Shape does not satisfy " << cacheName << " load constraints, " << dim1Name << ":" << tile1 << ", "
        << dim2Name << ":" << tile2 << ", " << dim3Name << ":" << tile3 << ", which must satisfy " << dim1Name << " × "
        << dim2Name << " × " << dim3Name << " × dtypesize ≤ " << cacheName << "Size(" << cacheSize << ").";
}

void CheckL0TileTiling(DataType outType, const ConvAttrParam& attrParam, const Tensor& weightTensor,
                       const Tensor& inputTensor)
{
    auto& convTile = TileShape::Current().GetConvTile();
    int64_t tileH = convTile.tileL0Info.tileH, tileW = convTile.tileL0Info.tileW;
    int64_t tileN = convTile.tileL0Info.tileN, tileK = convTile.tileL0Info.tileK;
    int64_t tileHout = convTile.tileL1Info.tileHout, tileWout = convTile.tileL1Info.tileWout;
    int64_t tileCout = convTile.tileL1Info.tileN, k0 = ALIGN_SIZE_32 / BytesOf(outType);
    int64_t tileCinFmap = convTile.tileL1Info.tileCinFmap;
    int64_t tileCinWeight = convTile.tileL1Info.tileCinWeight;
    uint32_t indexH = attrParam.isConv3D ? NCDHW_H_IDX : NCHW_H_IDX;
    uint32_t indexW = attrParam.isConv3D ? NCDHW_W_IDX : (attrParam.isConv1D ? NCHW_H_IDX : NCHW_W_IDX);
    int64_t kh = attrParam.isConv1D ? 1 : weightTensor.GetShape()[indexH];
    int64_t kw = weightTensor.GetShape()[indexW];
    int64_t cin = inputTensor.GetShape()[NCHW_C_IDX];
    int64_t cout = weightTensor.GetShape()[NCHW_N_IDX];
    int64_t kAL1 = ConvAlignB(tileCinFmap, k0) * kh * kw, oriK = ConvAlignB(cin, k0) * kh * kw;
    int64_t kBL1 = ConvAlignB(tileCinWeight, k0) * kh * kw;
    int64_t batch = inputTensor.GetShape()[NCHW_N_IDX], groups = attrParam.groups;
    int64_t hOut = ConvComputeHo(inputTensor, weightTensor, attrParam);
    int64_t wOut = ConvComputeWo(inputTensor, weightTensor, attrParam);
    int64_t numTileL0 = batch * groups * CeilDiv(cout / groups, tileN) * CeilDiv(hOut, tileH) * CeilDiv(wOut, tileW);
    if (attrParam.isConv3D) {
        int64_t kd = weightTensor.GetShape()[NCDHW_D_IDX];
        int64_t dout = ConvComputeDo(inputTensor, weightTensor, attrParam);
        numTileL0 *= dout;
        oriK *= kd;
    }
    if (numTileL0 * CeilDiv(oriK, tileK) > MAX_LOOP) {
        CONV_LOGW("Suggestion: Consider increasing tile size to reduce compilation time.");
    }
    int64_t minKL1 = std::min(kAL1, kBL1);
    CheckAlignment(tileK, k0, "tileK");
    CheckValueRange(tileH, "tileH", NUM1, tileHout);
    CheckValueRange(tileW, "tileW", NUM1, tileWout);
    CheckValueRange(tileK, "tileK", NUM1, minKL1);
    CheckAlignment(tileN, MKN_N_VALUE, "tileL0Info.tileN");
    CheckAlignment(tileW, MKN_N_VALUE, "tileW");
    CheckValueRange(tileN, "tileL0Info.tileN", NUM1, ConvAlignB(tileCout, MKN_N_VALUE));
    CHECK(ExternalError::INVALID_VAL, kAL1 % tileK == 0 && kBL1 % tileK == 0)
        << "Invalid tileK: " << tileK << ", must be a factor of both kAL1:" << kAL1 << " and kBL1:" << kBL1;
    Platform& platform = Platform::Instance();
    size_t l0aSize = platform.GetAICCore().GetMemorySize(MemoryType::MEM_L0A);
    size_t l0bSize = platform.GetAICCore().GetMemorySize(MemoryType::MEM_L0B);
    size_t l0cSize = platform.GetAICCore().GetMemorySize(MemoryType::MEM_L0C);
    ValidateL0Constraint(tileH, tileW, tileK, BytesOf(outType), l0aSize, "L0A", "tileH", "tileW", "tileK");
    ValidateL0Constraint(tileK, tileN, 1, BytesOf(outType), l0bSize, "L0B", "tileK", "tileN", "");
    ValidateL0Constraint(tileH, tileW, tileN, BytesOf(DataType::DT_FP32), l0cSize, "L0C", "tileH", "tileW", "tileN");
}

void CheckDivisible(int64_t value, int64_t divisor, const std::string& valueName, const std::string& divisorName)
{
    CHECK(ExternalError::INVALID_VAL, divisor != 0) << divisorName << " cannot be zero.";
    CHECK(ExternalError::INVALID_VAL, value % divisor == 0)
        << "The value of " << divisorName << " (" << divisor << ") does not divide " << valueName << "(" << value
        << "). Adjusting " << divisorName << " to the nearest value such that " << valueName << " % " << divisorName
        << " == 0.";
}

void CheckGroupsShape(const int64_t cinFmap, const int64_t cinWeight, const int64_t cOut, const int64_t groups)
{
    CheckValueRange(groups, "groups", NUM1, SHAPE_INNER_AXIS_MAX_SIZE);
    CheckDivisible(cinFmap, groups, "Cin", "groups");
    CheckDivisible(cOut, groups, "Cout", "groups");
    CHECK(ExternalError::INVALID_VAL, cinFmap == cinWeight * groups)
        << "Fmap Cin (" << cinFmap << ") != weight Cin (" << cinWeight << ") * groups (" << groups << ").";
}

void CheckDimParam(const std::vector<int64_t>& vec, const std::string& name, int expectedDim)
{
    CHECK(ExternalError::INVALID_VAL, vec.size() == static_cast<size_t>(expectedDim))
        << "Input attr " << name << " dim: " << vec.size() << " != " << expectedDim << ".";
}

void CheckDimensionRange(const std::vector<int64_t>& vec, const std::string& name, int minVal, int maxVal)
{
    for (size_t i = 0; i < vec.size(); ++i) {
        CHECK(ExternalError::OUT_OF_RANGE, vec[i] >= minVal && vec[i] <= maxVal)
            << "The value of the " << i << "-th dimension of " << name << " must be in the range [" << minVal << ","
            << maxVal << "]. Current value:" << vec[i] << ".";
    }
}

void CheckLoad3dShape(DataType outType, const Tensor& weightTensor, const ConvAttrParam& attrParam)
{
    std::vector<int64_t> paddings = attrParam.paddings;
    std::vector<int64_t> dilations = attrParam.dilations;
    std::vector<int64_t> strides = attrParam.strides;
    if (attrParam.isConv3D) {
        paddings = rotateVector(paddings, NUM4);
        dilations = rotateVector(dilations, NUM2);
        strides = rotateVector(strides, NUM2);
    }
    CheckDimensionRange(paddings, "paddings", 0, MAX_PAD_KERNEL);
    CheckDimensionRange(dilations, "dilations", NUM1, MAX_DILATION_STRIDE);
    CheckDimensionRange(strides, "strides", NUM1, MAX_DILATION_STRIDE);

    uint32_t indexH = attrParam.isConv3D ? NCDHW_H_IDX : NCHW_H_IDX;
    uint32_t indexW = attrParam.isConv3D ? NCDHW_W_IDX : (attrParam.isConv1D ? NCHW_H_IDX : NCHW_W_IDX);
    int64_t kw = weightTensor.GetShape()[indexW];
    int64_t kh = attrParam.isConv1D ? 1 : weightTensor.GetShape()[indexH];
    CHECK(ExternalError::OUT_OF_RANGE, kh <= MAX_PAD_KERNEL && kw <= MAX_PAD_KERNEL)
        << "Weight shapes do not satisfy Load3D's" << (attrParam.isConv1D ? " limit: kw=" : " limits: kh=")
        << (attrParam.isConv1D ? kw : kh) << (attrParam.isConv1D ? "" : ", kw=" + std::to_string(kw))
        << ", which must <= " << MAX_PAD_KERNEL << ".";

    int64_t k0 = ALIGN_SIZE_32 / BytesOf(outType);
    CHECK(ExternalError::OUT_OF_RANGE, kh * kw * k0 <= SHAPE_INNER_AXIS_MAX_SIZE)
        << "Weight shapes do not satisfy Load3D's limits: kh*kw*k0=" << kh * kw * k0
        << "(k0 = 32 bytes / dtypesize), which must <=" << SHAPE_INNER_AXIS_MAX_SIZE << ".";
}

void CheckAttrShape(DataType outType, const Tensor& inputTensor, const Tensor& weightTensor,
                    const ConvAttrParam& attrParam)
{
    std::vector<int64_t> paddings = attrParam.paddings;
    uint32_t index = attrParam.isConv3D ? SHAPE_DIM3 : (attrParam.isConv1D ? SHAPE_DIM1 : SHAPE_DIM2);
    CheckDimParam(attrParam.paddings, "paddings", index * NUM2);
    CheckDimParam(attrParam.dilations, "dilations", index);
    CheckDimParam(attrParam.strides, "strides", index);
    int64_t groups = attrParam.groups;
    int64_t cinFmap = inputTensor.GetShape()[NCHW_C_IDX];
    int64_t cinWeight = weightTensor.GetShape()[NCHW_C_IDX];
    int64_t cOut = weightTensor.GetShape()[NCHW_N_IDX];

    if (attrParam.isConv3D) {
        paddings = rotateVector(paddings, NUM4);
    }
    const std::vector<std::string> dimNames = attrParam.isConv1D ? std::vector<std::string>{"L"} :
                                              attrParam.isConv3D ? std::vector<std::string>{"D", "H", "W"} :
                                                                   std::vector<std::string>{"H", "W"};
    for (size_t i = 0; i < paddings.size() / NUM2; ++i) {
        int weightVal = weightTensor.GetShape()[i + NUM2];
        int paddingLeft = paddings[i * NUM2];
        int paddingRight = paddings[i * NUM2 + 1];
        CHECK(ExternalError::INVALID_VAL, paddingLeft < weightVal && paddingRight < weightVal)
            << "The value of the " << dimNames[i]
            << " dimension of weight must be > padding. Current weight value:" << weightVal
            << ", padding value:" << paddingLeft << " and " << paddingRight << ".";
    }
    CheckGroupsShape(cinFmap, cinWeight, cOut, groups);
    CheckLoad3dShape(outType, weightTensor, attrParam);
}

void CheckOriginShape(const Tensor& inputTensor, const Tensor& weightTensor, const Tensor& biasTensor)
{
    CheckDimensionRange(inputTensor.GetShape(), "fmap", NUM1, MAX_SIZE);
    CheckDimensionRange(weightTensor.GetShape(), "weight", NUM1, MAX_SIZE);

    if (biasTensor.IsEmpty()) {
        return;
    }
    int64_t cOut = weightTensor.GetShape()[NCHW_N_IDX];
    CHECK(ExternalError::INVALID_VAL, biasTensor.GetShape()[0] == cOut)
        << "Input illegal bias shape:" << biasTensor.GetShape()[0] << ", which must equal to Cout:" << cOut << ".";
}

} // namespace

void CheckConvOperands(DataType outType, const Tensor& inputTensor, const Tensor& weightTensor,
                       const Tensor& biasTensor, ConvAttrParam& attrParam)
{
    CHECK(ExternalError::INVALID_TYPE,
          outType == DataType::DT_FP32 || outType == DataType::DT_FP16 || outType == DataType::DT_BF16)
        << "Unsupported output data type. Only DT_FP32, DT_FP16, DT_BF16 are supported.";
    if (inputTensor.Dim() == CONV1D_INPUT_DIM && weightTensor.Dim() == CONV1D_INPUT_DIM) {
        attrParam.isConv1D = true;
    } else if (inputTensor.Dim() == CONV3D_INPUT_DIM && weightTensor.Dim() == CONV3D_INPUT_DIM) {
        attrParam.isConv3D = true;
    }
    CheckOriginShape(inputTensor, weightTensor, biasTensor);
    CheckOutputShape(inputTensor, weightTensor, attrParam);
    CheckAttrShape(outType, inputTensor, weightTensor, attrParam);
    CheckTileTiling(outType, inputTensor, weightTensor, attrParam);
    CheckL1SizeTiling(outType, inputTensor, weightTensor, biasTensor, attrParam);
}

void CheckTileTiling(DataType outType, const Tensor& inputTensor, const Tensor& weightTensor,
                     const ConvAttrParam& attrParam)
{
    auto convTile = TileShape::Current().GetConvTile();
    int64_t tileHin = convTile.tileL1Info.tileHin;
    int64_t tileWin = convTile.tileL1Info.tileWin;
    int64_t tileCinFmap = convTile.tileL1Info.tileCinFmap;
    int64_t tileCinWeight = convTile.tileL1Info.tileCinWeight;
    int64_t tileN = convTile.tileL1Info.tileN;
    int64_t tileBatch = convTile.tileL1Info.tileBatch;
    int64_t groups = attrParam.groups;

    uint32_t indexH = attrParam.isConv3D ? NCDHW_H_IDX : NCHW_H_IDX;
    uint32_t indexW = attrParam.isConv3D ? NCDHW_W_IDX : (attrParam.isConv1D ? NCHW_H_IDX : NCHW_W_IDX);
    int64_t cOut = weightTensor.GetShape()[NCHW_N_IDX];
    int64_t hin = attrParam.isConv1D ? 1 : inputTensor.GetShape()[indexH];
    int64_t win = inputTensor.GetShape()[indexW];
    int64_t k0 = ALIGN_SIZE_32 / BytesOf(outType);
    int64_t cinWeight = weightTensor.GetShape()[NCHW_C_IDX];

    CheckValueRange(tileHin, "tileHin", NUM1, hin);
    CheckValueRange(tileBatch, "tileBatch", NUM1, NUM1);
    CheckValueRange(tileWin, "tileWin", NUM1, win);
    CheckValueRange(tileCinFmap, "tileCinFmap", k0, ConvAlignB(cinWeight, k0));
    CheckValueRange(tileCinWeight, "tileCinWeight", k0, ConvAlignB(cinWeight, k0));
    CheckValueRange(tileN, "tileL1Info.tileN", NUM1, ConvAlignB(cOut / groups, MKN_N_VALUE));
    CheckAlignment(tileN, MKN_N_VALUE, "tileL1Info.tileN");

    CheckHowoTile(inputTensor, weightTensor, attrParam);
    CheckAlignment(tileCinFmap, k0, "tileCinFmap");
    CheckAlignment(tileCinWeight, k0, "tileCinWeight");
    if (attrParam.isConv3D) {
        CheckDivisible(ConvAlignB(cinWeight, k0), tileCinFmap, "cin", "tileCinFmap");
        CheckDivisible(ConvAlignB(cinWeight, k0), tileCinWeight, "cin", "tileCinWeight");
    }
    if (convTile.setL0Tile) {
        CheckL0TileTiling(outType, attrParam, weightTensor, inputTensor);
    }
}

void CheckL1SizeTiling(DataType outType, const Tensor& inputTensor, const Tensor& weightTensor,
                       const Tensor& biasTensor, const ConvAttrParam& attrParam)
{
    auto convTile = TileShape::Current().GetConvTile();
    Platform& platform = Platform::Instance();
    uint64_t l1Size = platform.GetAIVCore().GetMemorySize(MemoryType::MEM_L1);
    uint32_t indexH = attrParam.isConv3D ? NCDHW_H_IDX : NCHW_H_IDX;
    uint32_t indexW = attrParam.isConv3D ? NCDHW_W_IDX : (attrParam.isConv1D ? NCHW_H_IDX : NCHW_W_IDX);

    uint64_t kh = attrParam.isConv1D ? 1 : weightTensor.GetShape()[indexH];
    uint64_t hin = attrParam.isConv1D ? 1 : inputTensor.GetShape()[indexH];
    uint64_t kw = weightTensor.GetShape()[indexW];
    uint64_t win = inputTensor.GetShape()[indexW];
    uint64_t k0 = ALIGN_SIZE_32 / BytesOf(outType);

    std::vector<int64_t> strides = attrParam.strides;
    std::vector<int64_t> dilations = attrParam.dilations;
    uint32_t indexAttrW = attrParam.isConv1D ? PAD_STRIDE_H : PAD_STRIDE_W;
    uint64_t strideH = attrParam.isConv1D ? 1 : strides[PAD_STRIDE_H];
    uint64_t strideW = strides[indexAttrW];
    uint64_t dilationH = attrParam.isConv1D ? 1 : dilations[PAD_STRIDE_H];
    uint64_t dilationW = dilations[indexAttrW];

    uint64_t biasL1Size = 0;
    uint64_t tileN = convTile.tileL1Info.tileN;
    if (!biasTensor.IsEmpty()) {
        biasL1Size = ConvAlignB(tileN * BytesOf(outType), ALIGN_SIZE_32);
    }
    uint64_t tileCinFmap = convTile.tileL1Info.tileCinFmap;
    uint64_t tileCinWeight = convTile.tileL1Info.tileCinWeight;
    uint64_t kBL1 = ConvAlignB(tileCinWeight * kh * kw, k0);
    uint64_t weightL1Size = ConvAlignB(kBL1 * tileN * BytesOf(outType), ALIGN_SIZE_32);

    uint64_t inputL1Size = 0;
    uint64_t tileWout = convTile.tileL1Info.tileWout;
    uint64_t tileHout = convTile.tileL1Info.tileHout;
    uint64_t khDilated = (kh - 1) * dilationH + 1;
    uint64_t hiAL1 = std::min((tileHout - 1) * strideH + khDilated, hin);
    uint64_t kwDilated = (kw - 1) * dilationW + 1;
    uint64_t wiAL1 = std::min((tileWout - 1) * strideW + kwDilated, win);

    inputL1Size = ConvAlignB(hiAL1 * wiAL1 * tileCinFmap * BytesOf(outType), ALIGN_SIZE_32);
    uint64_t minL1LoadSize = biasL1Size + inputL1Size + weightL1Size;
    ASSERT(ConvOperationError::OVER_BUFFER_LIMIT, minL1LoadSize <= l1Size)
        << "MinL1LoadSize > L1size, current MinL1LoadSize: " << minL1LoadSize << ", L1size: " << l1Size << ".";
}

} // namespace Conv
} // namespace tile_fwk
} // namespace npu
