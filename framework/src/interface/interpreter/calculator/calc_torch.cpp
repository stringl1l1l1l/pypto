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
 * \file calc_torch.cpp
 * \brief
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <torch/torch.h>
#include "calc_api.h"
#include "fp_convert.h"
#include "tilefwk/error.h"
#include "securec.h"
#include "calc_error.h"

namespace npu::tile_fwk {

// Logical shape is exposed to calculator. Convert back to packed view shape when touching raw storage.
static std::vector<int64_t> ShapePackedView(const std::vector<int64_t>& logicalShape, DataType dtype)
{
    if (logicalShape.empty() || !IsFp4PackedDtype(dtype)) {
        return logicalShape;
    }
    std::vector<int64_t> s = logicalShape;
    if (s.back() >= 0) {
        s.back() = (s.back() + 1) / 0x2;
    }
    return s;
}

static int64_t LastDimPackedCount(int64_t logicalLast, DataType dtype)
{
    if (logicalLast < 0) {
        return logicalLast;
    }
    return IsFp4PackedDtype(dtype) ? ((logicalLast + 1) / 0x2) : logicalLast;
}

static int64_t StorageOffsetFloatToPacked(int64_t logicalOffset, DataType dtype)
{
    if (!IsFp4PackedDtype(dtype)) {
        return logicalOffset;
    }
    return logicalOffset / 0x2;
}

static std::vector<int64_t> StrideFloatToPacked(const std::vector<int64_t>& logicalStride, DataType dtype)
{
    if (!IsFp4PackedDtype(dtype)) {
        return logicalStride;
    }
    if (logicalStride.empty()) {
        return logicalStride;
    }
    std::vector<int64_t> packedStride = logicalStride;
    const size_t last = packedStride.size() - 1U;
    // RawStride + ExpandLastDimForFp4 on last only (e.g. [32, 2]): only halve the last dim.
    // Logical contiguous stride (e.g. [64, 1]): halve outer dims and last dim for packed uint8 view.
    if (packedStride[last] > 1) {
        packedStride[last] = std::max<int64_t>(1LL, packedStride[last] / 2LL);
        return packedStride;
    }
    for (size_t i = 0; i + 1 < packedStride.size(); ++i) {
        if (packedStride[i] > 0) {
            packedStride[i] /= 2LL;
        }
    }
    if (packedStride[last] >= 0) {
        packedStride[last] = std::max<int64_t>(1LL, packedStride[last] / 2LL);
    }
    return packedStride;
}

static int64_t LastDimFloatCount(int64_t packedLast, DataType dtype)
{
    if (packedLast < 0) {
        return packedLast;
    }
    return IsFp4PackedDtype(dtype) ? (packedLast * 0x2) : packedLast;
}

#define AXIS_TO_LAST -2
#define NUM_VALUE_8 8
#define BLOCK_SIZE 32
#define MX_QUANT_TILE_BLOCK 32

static torch::ScalarType FromDataType(DataType t)
{
    switch (t) {
        case DT_INT8:
            return torch::kInt8;
        case DT_INT16:
            return torch::kInt16;
        case DT_INT32:
            return torch::kInt32;
        case DT_INT64:
            return torch::kInt64;
        case DT_FP16:
            return torch::kFloat16;
        case DT_FP32:
            return torch::kFloat32;
        case DT_BF16:
            return torch::kBFloat16;
        case DT_UINT8:
            return torch::kInt8;
        case DT_UINT16:
            return torch::kInt16;
        case DT_UINT32:
            return torch::kInt32;
        case DT_UINT64:
            return torch::kInt64;
        case DT_BOOL:
            return torch::kBool;
        case DT_DOUBLE:
            return torch::kDouble;
        case DT_INT4:
        case DT_FP8:
        case DT_HF8:
        case DT_FP8E5M2:
            return torch::kUInt8;
        case DT_FP8E4M3:
            return torch::kUInt8;
        case DT_FP8E8M0:
            return torch::kUInt8;
        case DT_FP4_E2M1:
        case DT_FP4_E1M2:
            return torch::kUInt8;
        case DT_HF4:
        default:
            assert(0);
    }
    return torch::ScalarType::Undefined;
}

static at::Scalar From(const Element& elem)
{
    switch (elem.GetDataType()) {
        case DT_BOOL:
        case DT_INT4:
        case DT_INT8:
        case DT_INT16:
        case DT_INT32:
            return at::Scalar(elem.GetSignedData());
        case DT_INT64:
            return at::Scalar(elem.GetSignedData());
        case DT_FP16:
        case DT_BF16:
        case DT_DOUBLE:
            return at::Scalar(elem.GetFloatData());
        case DT_FP32: {
            // Clamp FP32 scalar into finite FP32 range to avoid INF
            double data = elem.GetFloatData();
            constexpr double kMaxF32 = static_cast<double>(std::numeric_limits<float>::max());
            if (data > kMaxF32) {
                data = kMaxF32;
            } else if (data < -kMaxF32) {
                data = -kMaxF32;
            }
            return at::Scalar(static_cast<float>(data));
        }
        case DT_UINT8:
        case DT_UINT16:
        case DT_UINT32:
        case DT_UINT64:
            // lower version of pytorch not support uint64 type, use int64 for temp
            return at::Scalar(static_cast<int64_t>(elem.GetUnsignedData()));
        case DT_FP8:
        case DT_FP8E5M2:
        case DT_FP8E4M3:
        case DT_FP8E8M0:
        case DT_HF4:
        case DT_HF8:
        default:
            assert(0);
    }
    return at::Scalar();
}

static void ToOperand(const torch::Tensor& src, const torch::Tensor& dst, DataType actualType)
{
    if (IsFp8Dtype(actualType)) {
        dst.copy_(Float32ToFp8(src, actualType));
    } else if (IsFp4PackedDtype(actualType)) {
        dst.copy_(Float32ToFp4Packed(src, actualType));
    } else {
        dst.copy_(src);
    }
}

static std::pair<torch::Tensor, torch::Tensor> From(const TensorData& data)
{
    auto ScalarDataType = FromDataType(data.dtype);
    const bool isFp4Packed = IsFp4PackedDtype(data.dtype);
    torch::Tensor view;
    torch::Tensor actualView;
    if (isFp4Packed) {
        auto packedRawShape = ShapePackedView(data.rawShape, data.dtype);
        auto tensor = torch::from_blob(data.dataPtr, packedRawShape, ScalarDataType);
        auto packedShape = ShapePackedView(data.shape, data.dtype);
        auto packedStride = StrideFloatToPacked(data.stride, data.dtype);
        auto packedOffset = StorageOffsetFloatToPacked(data.storageOffset, data.dtype);
        view = tensor.as_strided(packedShape, packedStride, packedOffset);
        if (data.isAxisCombine) {
            view = view.transpose_(-1, AXIS_TO_LAST);
        }
        actualView = Fp4PackedToFloat32(view, data.dtype);
    } else {
        auto tensor = torch::from_blob(data.dataPtr, data.rawShape, ScalarDataType);
        view = tensor.as_strided(data.shape, data.stride, data.storageOffset);
        if (data.isAxisCombine) {
            view = view.transpose_(-1, AXIS_TO_LAST);
        }
        actualView = view;
    }
    if (IsFp8Dtype(data.dtype)) {
        actualView = Fp8ToFloat32(view, data.dtype);
    } else if (ScalarDataType == torch::kUInt8 && !isFp4Packed) {
        actualView = view.to(torch::kFloat32);
    }
    // view == actualView if ScalarDataType != torch::kUInt8
    return {view, actualView};
}

static uint32_t FloatToBits(float value)
{
    union {
        float floatValue;
        uint32_t bits;
    } converter = {value};
    return converter.bits;
}

static float BitsToFloat(uint32_t bits)
{
    union {
        uint32_t bits;
        float floatValue;
    } converter = {bits};
    return converter.floatValue;
}

static float RoundToBf16(float value) { return static_cast<float>(c10::BFloat16(value)); }

static float RoundToFp16(float value)
{
    const uint16_t rounded = c10::detail::fp16_ieee_from_fp32_value(value);
    return c10::detail::fp16_ieee_to_fp32_value(rounded);
}

static float TruncateToBf16(float value) { return BitsToFloat(FloatToBits(value) & 0xFFFF0000u); }

// MX quantization constants (OCP Microscaling Formats MX v1.0)
// These are parameterized per target dtype for future fp4 extensibility.
struct MXQuantDtypeParams {
    int targetMaxPow2; // max representable power-of-2 exponent in the target format
    float maxPos;      // max representable positive value
    float minNormal;   // smallest normal value
    int expBias;       // exponent bias of the target format
    int mbits;         // number of mantissa bits
};

static constexpr MXQuantDtypeParams kFP8E4M3Params = {
    .targetMaxPow2 = 8,
    .maxPos = 448.0f,
    .minNormal = 0.015625f, // 2^(1-7) = 2^-6
    .expBias = 7,
    .mbits = 3,
};

static constexpr MXQuantDtypeParams kFP4E2M1Params = {
    .targetMaxPow2 = 2,
    .maxPos = 6.0f,
    .minNormal = 1.0f,
    .expBias = 1,
    .mbits = 1,
};

static constexpr int kE8M0ExponentBias = 127;
static constexpr int kF32ExpBias = 127;
static constexpr int kF32Mbits = 23;
static constexpr int64_t kMxQuantModeRoundUp = 0;

// Compute OCP FLOOR-mode shared exponent (E8M0 biased byte) for a group.
// Reference: OCP MX Spec 1.0 — scale = 2^floor(log2(max_abs)) / 2^target_max_pow2
static uint8_t ComputeSharedExponent(float maxAbsValue, int targetMaxPow2)
{
    if (std::isnan(maxAbsValue)) {
        return 0xFFu; // NaN -> E8M0 NaN. Inf follows the FLOOR exponent path.
    }
    const uint32_t bits = FloatToBits(maxAbsValue);
    const uint32_t fpExponent = (bits & 0x7F800000u) >> kF32Mbits;
    // scale_unbiased = (fpExponent - F32_EXP_BIAS) - targetMaxPow2
    // scale_biased   = scale_unbiased + E8M0_EXPONENT_BIAS = fpExponent - targetMaxPow2
    // Clamp to valid E8M0 range [0, 254] (255 reserved for NaN)
    if (fpExponent <= static_cast<uint32_t>(targetMaxPow2)) {
        return 0u;
    }
    const uint32_t biased = fpExponent - static_cast<uint32_t>(targetMaxPow2);
    return static_cast<uint8_t>(std::min(biased, 254u));
}

static uint8_t ComputeSharedExponentNV(float maxAbsValue, float maxPos)
{
    if (std::isnan(maxAbsValue)) {
        return 0xFFu;
    }
    const long double descale = static_cast<long double>(maxAbsValue) / static_cast<long double>(maxPos);
    if (descale <= 0.0L) {
        return 0u;
    }
    if (std::isinf(static_cast<double>(descale))) {
        return 0xFEu;
    }
    if (descale <= std::ldexp(1.0L, -kE8M0ExponentBias)) {
        return 0u;
    }
    const int e8m0 = static_cast<int>(std::ceil(std::log2(descale))) + kE8M0ExponentBias;
    return static_cast<uint8_t>(std::clamp(e8m0, 0, 0xfe));
}

// Compute the reciprocal scaling factor from an E8M0 biased exponent.
// reciprocal_scale = 2^(E8M0_BIAS - e8m0) so that data * reciprocal_scale = data / scale.
static float ComputeScalingFromExponent(uint8_t e8m0)
{
    if (e8m0 == 0xFFu) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const uint32_t scaleExp = 254u - static_cast<uint32_t>(e8m0);
    if (scaleExp == 0u) {
        return std::ldexp(1.0f, -kE8M0ExponentBias);
    }
    return BitsToFloat(scaleExp << kF32Mbits);
}

static float ComputeScalingFromExponentMath(uint8_t e8m0)
{
    if (e8m0 == 0xFFu) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return std::ldexp(1.0f, kE8M0ExponentBias - static_cast<int>(e8m0));
}

static std::pair<uint8_t, float> ComputeB16OcpExponentAndScaling(float maxAbsValue, bool isFp4E2M1)
{
    if (std::isnan(maxAbsValue)) {
        return {0xFFu, std::numeric_limits<float>::quiet_NaN()};
    }
    const int targetMaxPow2 = isFp4E2M1 ? kFP4E2M1Params.targetMaxPow2 : kFP8E4M3Params.targetMaxPow2;
    if (std::isinf(maxAbsValue)) {
        const auto e8m0 = static_cast<uint8_t>(std::clamp(0xFF - targetMaxPow2, 0, 0xFE));
        return {e8m0, std::ldexp(1.0f, kE8M0ExponentBias - static_cast<int>(e8m0))};
    }
    if (maxAbsValue <= std::ldexp(1.0f, targetMaxPow2 - kE8M0ExponentBias)) {
        return {0u, std::ldexp(1.0f, kE8M0ExponentBias)};
    }
    const int e8m0 = static_cast<int>(std::floor(std::log2(static_cast<long double>(maxAbsValue)))) - targetMaxPow2 +
                     kE8M0ExponentBias;
    const auto clamped = static_cast<uint8_t>(std::clamp(e8m0, 0, 0xFE));
    return {clamped, std::ldexp(1.0f, kE8M0ExponentBias - static_cast<int>(clamped))};
}

// Encode a float32 value to FP8 E4M3 (round-to-nearest-even) via bit manipulation.
// Reference: torchao _f32_to_floatx_unpacked (OCP MX Formats)
static uint8_t EncodeE4M3Fn(float value)
{
    if (std::isnan(value)) {
        return 0x7Fu;
    }

    constexpr auto& p = kFP8E4M3Params;
    constexpr uint8_t kMaxCode = 0x7Eu; // max magnitude (not NaN)
    constexpr uint8_t kSignMask = 0x80u;
    constexpr int kShift = kF32Mbits - p.mbits; // 23 - 3 = 20
    // magic_adder for RNE: (1 << (shift - 1)) - 1
    constexpr uint32_t kMagicAdder = (1u << (kShift - 1)) - 1u;
    // denorm_exp = (F32_EXP_BIAS - fp8_exp_bias) + (F32_MBITS - fp8_mbits) + 1
    constexpr uint32_t kDenormExp = (kF32ExpBias - p.expBias) + kShift + 1u;
    constexpr uint32_t kDenormMaskInt = kDenormExp << kF32Mbits;
    static const float kDenormMaskFloat = BitsToFloat(kDenormMaskInt);

    const uint32_t bits = FloatToBits(value);
    const uint8_t sign = static_cast<uint8_t>((bits >> 24) & kSignMask);
    const uint32_t absBits = bits & 0x7FFFFFFFu;
    const float absVal = BitsToFloat(absBits);
    // Branch 1: saturation
    if (absVal >= p.maxPos) {
        return sign | kMaxCode;
    }
    // Branch 2: denormal in fp8 (abs < min_normal)
    if (absVal < p.minNormal) {
        // Denormal trick: add a magic float then subtract the integer representation
        const float temp = absVal + kDenormMaskFloat;
        const uint32_t tempBits = FloatToBits(temp) - kDenormMaskInt;
        return sign | static_cast<uint8_t>(tempBits);
    }
    // Branch 3: normal — adjust exponent and round-to-nearest-even
    const uint32_t mantOdd = (absBits >> kShift) & 1u;
    // Reinterpret as int32 for the exponent/rounding adjustment
    const int32_t expBiasDelta = static_cast<int32_t>(p.expBias - kF32ExpBias);
    const int32_t valToAdd = expBiasDelta * static_cast<int32_t>(uint32_t{1} << kF32Mbits) +
                             static_cast<int32_t>(kMagicAdder);
    const uint32_t adjusted = absBits + static_cast<uint32_t>(valToAdd) + mantOdd;
    return sign | static_cast<uint8_t>((adjusted >> kShift) & 0x7Fu);
}

static uint8_t EncodeE2M1Magic(float value)
{
    if (std::isnan(value)) {
        return 0x7u;
    }
    const uint32_t valueBits = FloatToBits(value);
    const uint8_t sign = static_cast<uint8_t>((valueBits >> 28) & 0x8u);
    const float absValue = std::fabs(value);
    if (std::isinf(absValue)) {
        return static_cast<uint8_t>(sign | 0x7u);
    }

    const uint32_t absBits = FloatToBits(absValue);
    uint32_t biasedExp = (absBits & 0x7F800000u) >> kF32Mbits;
    biasedExp = std::clamp<uint32_t>(biasedExp, 127u, 129u);

    const uint32_t magicBits = (biasedExp + 22u) << kF32Mbits;
    const uint32_t q = FloatToBits(absValue + BitsToFloat(magicBits)) - magicBits;
    const uint32_t baseCode = (biasedExp - 127u) << 1u;
    const uint32_t magCode = std::min<uint32_t>(q + baseCode, 7u);
    return static_cast<uint8_t>(sign | magCode);
}

static torch::Tensor View(const torch::Tensor& self, const std::vector<int64_t>& shape,
                          const std::vector<int64_t>& offset)
{
    int64_t storageOffset = self.storage_offset();
    for (size_t dim = 0; dim < offset.size(); dim++) {
        storageOffset += self.stride(dim) * offset[dim];
    }
    return self.as_strided(shape, self.strides(), storageOffset);
}

static bool AllClose(const TensorData& self, const TensorData& other, double atol, double rtol)
{
    return From(self).second.allclose(From(other).second, atol, rtol);
}

static void Random(const TensorData& out)
{
    auto tout = From(out);
    torch::rand_out(tout.second, tout.second.sizes());
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Exp(const TensorData& out, const TensorData& self)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::exp_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Exp2(const TensorData& out, const TensorData& self)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::exp2_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Expm1(const TensorData& out, const TensorData& self)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::expm1_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Sin(const TensorData& out, const TensorData& self)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::sin_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Cos(const TensorData& out, const TensorData& self)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::cos_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Erf(const TensorData& out, const TensorData& self)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::erf_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Sinh(const TensorData& out, const TensorData& self)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::sinh_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Cosh(const TensorData& out, const TensorData& self)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::cosh_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Erfc(const TensorData& out, const TensorData& self)
{
    auto tout = From(out);
    auto tself = From(self);
    tout.second = torch::erfc(tself.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Asin(const TensorData& out, const TensorData& self)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::asin_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Acos(const TensorData& out, const TensorData& self)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::acos_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void ASinh(const TensorData& out, const TensorData& self)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::asinh_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void ACosh(const TensorData& out, const TensorData& self)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::acosh_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Atanh(const TensorData& out, const TensorData& self)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::atanh_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Neg(const TensorData& out, const TensorData& self)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::neg_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void PackOrUnPack(const TensorData& out, const TensorData& self)
{
    auto tself = From(self);
    auto totalBytes = tself.first.nbytes();
    auto srcPtr = self.dataPtr;
    auto dstPtr = out.dataPtr;
    errno_t ret = memcpy_s(dstPtr, totalBytes, srcPtr, totalBytes);
    ASSERT(CalculatorErrorScene::PACKORUNPACK_MEMCPY_FAILED, ret == 0) << "memcpy_s failed, ret=" << ret;
    auto tout = From(out);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Pack(const TensorData& out, const TensorData& self) { PackOrUnPack(out, self); }

static void UnPack(const TensorData& out, const TensorData& self) { PackOrUnPack(out, self); }

static void Sign(const TensorData& out, const TensorData& self)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::sign_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Signbit(const TensorData& out, const TensorData& self)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::signbit_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Tanh(const TensorData& out, const TensorData& self)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::tanh_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Tan(const TensorData& out, const TensorData& self)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::tan_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Ceil(const TensorData& out, const TensorData& self)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::ceil_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Log1p(const TensorData& out, const TensorData& self)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::log1p_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Floor(const TensorData& out, const TensorData& self)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::floor_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Trunc(const TensorData& out, const TensorData& self)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::trunc_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Round(const TensorData& out, const TensorData& self, int decimals)
{
    auto tout = From(out);
    auto tself = From(self);
    tout.second.copy_(torch::round(tself.second, decimals));
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Rsqrt(const TensorData& out, const TensorData& self)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::rsqrt_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Sqrt(const TensorData& out, const TensorData& self)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::sqrt_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Reciprocal(const TensorData& out, const TensorData& self)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::reciprocal_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Relu(const TensorData& out, const TensorData& self)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::relu_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Pad(const TensorData& out, const TensorData& input, const Element& padValue)
{
    auto tinput = From(input);
    auto tout = From(out);

    std::vector<int64_t> in_shape = tinput.second.sizes().vec();
    std::vector<int64_t> out_shape = tout.second.sizes().vec();
    size_t ndim = out_shape.size();
    int64_t pad_right = 0;
    int64_t pad_bottom = 0;

    if (ndim >= 0x2) {
        pad_right = std::max(static_cast<int64_t>(0), out_shape[ndim - 1] - in_shape[ndim - 1]);
        pad_bottom = std::max(static_cast<int64_t>(0), out_shape[ndim - 2] - in_shape[ndim - 2]);
    } else if (ndim == 1) {
        pad_right = std::max(static_cast<int64_t>(0), out_shape[0] - in_shape[0]);
    }
    std::vector<int64_t> pad_vec = {0, pad_right, 0, pad_bottom};
    double pad_val_double = padValue.Cast<double>();
    namespace F = torch::nn::functional;
    F::PadFuncOptions options(pad_vec);
    options.mode(torch::kConstant).value(pad_val_double);
    auto result = F::pad(tinput.second, options);
    tout.second.copy_(result);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void FillPad(const TensorData& out, const TensorData& input, const Element& padValue)
{
    auto tinput = From(input);
    auto tout = From(out);

    std::vector<int64_t> in_shape = tinput.second.sizes().vec();
    std::vector<int64_t> out_shape = tout.second.sizes().vec();
    size_t ndim = out_shape.size();

    std::vector<int64_t> rawFloatShape = input.rawShape;
    std::vector<int64_t> valid_shape = in_shape;
    if (ndim >= 0x2) {
        valid_shape[ndim - 1] = std::min(in_shape[ndim - 1], rawFloatShape[ndim - 1]);
        valid_shape[ndim - 2] = std::min(in_shape[ndim - 2], rawFloatShape[ndim - 2]);
    } else if (ndim == 1) {
        valid_shape[0] = std::min(in_shape[0], rawFloatShape[0]);
    }

    double pad_val_double = padValue.Cast<double>();
    tout.second.fill_(pad_val_double);

    if (ndim == 1) {
        int64_t valid_w = valid_shape[0];
        if (valid_w > 0) {
            auto in_view = tinput.second.slice(0, 0, valid_w);
            auto out_view = tout.second.slice(0, 0, valid_w);
            out_view.copy_(in_view);
        }
    } else if (ndim == 0x2) {
        int64_t valid_h = valid_shape[0];
        int64_t valid_w = valid_shape[1];
        if (valid_h > 0 && valid_w > 0) {
            auto in_view = tinput.second.slice(1, 0, valid_w).slice(0, 0, valid_h);
            auto out_view = tout.second.slice(1, 0, valid_w).slice(0, 0, valid_h);
            out_view.copy_(in_view);
        }
    }

    ToOperand(tout.second, tout.first, out.dtype);
}

static void BitwiseNot(const TensorData& out, const TensorData& self)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::bitwise_not_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void LogicalNot(const TensorData& out, const TensorData& self)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::logical_not_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void LogicalAnd(const TensorData& out, const TensorData& self, const TensorData& other)
{
    auto tout = From(out);
    auto tself = From(self);
    auto tother = From(other);
    torch::logical_and_out(tout.second, tself.second, tother.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Abs(const TensorData& out, const TensorData& self)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::abs_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void WhereTT(const TensorData& out, const TensorData& condition, const TensorData& input,
                    const TensorData& other)
{
    auto tout = From(out);
    auto tcondition = From(condition);
    auto tinput = From(input);
    auto tother = From(other);
    torch::where_out(tout.second, tcondition.second, tinput.second, tother.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void WhereTS(const TensorData& out, const TensorData& condition, const TensorData& input, const Element& other)
{
    auto tout = From(out);
    auto tcondition = From(condition);
    auto tinput = From(input);
    torch::Tensor tother = torch::tensor(static_cast<float>(other.GetFloatData()), torch::kFloat32);
    torch::where_out(tout.second, tcondition.second, tinput.second, tother);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void WhereST(const TensorData& out, const TensorData& condition, const Element& input, const TensorData& other)
{
    auto tout = From(out);
    auto tcondition = From(condition);
    torch::Tensor tinput = torch::tensor(static_cast<float>(input.GetFloatData()), torch::kFloat32);
    auto tother = From(other);
    torch::where_out(tout.second, tcondition.second, tinput, tother.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void WhereSS(const TensorData& out, const TensorData& condition, const Element& input, const Element& other)
{
    auto tout = From(out);
    auto tcondition = From(condition);
    torch::Tensor tinput = torch::tensor(static_cast<float>(input.GetFloatData()), torch::kFloat32);
    torch::Tensor tother = torch::tensor(static_cast<float>(other.GetFloatData()), torch::kFloat32);
    torch::where_out(tout.second, tcondition.second, tinput, tother);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Ln(const TensorData& out, const TensorData& self)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::log_out(tout.second, tself.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void IsFinite(const TensorData& out, const TensorData& self)
{
    auto tout = From(out);
    auto tself = From(self);
    tout.second = torch::isfinite(tself.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void IsNan(const TensorData& out, const TensorData& self)
{
    auto tout = From(out);
    auto tself = From(self);
    tout.second = torch::isnan(tself.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

#define DEFINE_BINARY_S_OPS(Name, op_out)                                                                \
    static void Name(const TensorData& out, const TensorData& self, const Element& scalar, bool reverse) \
    {                                                                                                    \
        auto tout = From(out);                                                                           \
        auto tself = From(self);                                                                         \
        if (reverse) {                                                                                   \
            torch::full_out(tout.second, out.shape, From(scalar));                                       \
            torch::op_out(tout.second, tout.second, tself.second);                                       \
        } else {                                                                                         \
            torch::op_out(tout.second, tself.second, From(scalar));                                      \
        }                                                                                                \
        ToOperand(tout.second, tout.first, out.dtype);                                                   \
    }

DEFINE_BINARY_S_OPS(AddS, add_out)
DEFINE_BINARY_S_OPS(SubS, sub_out)
DEFINE_BINARY_S_OPS(MulS, mul_out)
DEFINE_BINARY_S_OPS(DivS, div_out)
DEFINE_BINARY_S_OPS(FmodS, fmod_out)
DEFINE_BINARY_S_OPS(RemainderS, remainder_out)
DEFINE_BINARY_S_OPS(RemainderRS, remainder_out)
DEFINE_BINARY_S_OPS(PowS, pow_out)
DEFINE_BINARY_S_OPS(BitwiseAndS, bitwise_and_out)
DEFINE_BINARY_S_OPS(BitwiseOrS, bitwise_or_out)
DEFINE_BINARY_S_OPS(BitwiseXorS, bitwise_xor_out)

static void FloorDivS(const TensorData& out, const TensorData& self, const Element& scalar, bool reverse)
{
    auto tout = From(out);
    auto tself = From(self);
    auto tscalar = torch::full({}, From(scalar), tself.second.options());
    if (reverse) {
        torch::full_out(tout.second, out.shape, From(scalar));
        torch::floor_divide_out(tout.second, tout.second, tself.second);
    } else {
        torch::floor_divide_out(tout.second, tself.second, tscalar);
    }
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Add(const TensorData& out, const TensorData& self, const TensorData& other)
{
    auto tself = From(self);
    auto tother = From(other);
    auto tout = From(out);

    std::vector<int64_t> shape_self = tself.second.sizes().vec();
    std::vector<int64_t> shape_other = tother.second.sizes().vec();

    if (shape_self.size() == 0x2 && shape_other.size() == 0x2 && shape_self[0] == shape_other[0] &&
        shape_self[1] != shape_other[1]) {
        if (shape_other[1] == 0x8) {
            int64_t cols_self = shape_self[1];
            int64_t cols_other = shape_other[1];
            int64_t repeat_times = (cols_self + cols_other - 1) / cols_other;
            auto tother_expanded = tother.second.repeat({1, repeat_times});
            auto tother_final = tother_expanded.index({torch::indexing::Slice(), torch::indexing::Slice(0, cols_self)});
            torch::add_out(tout.second, tself.second, tother_final);
        } else {
            torch::add_out(tout.second, tself.second, tother.second);
        }
    } else {
        torch::add_out(tout.second, tself.second, tother.second);
    }
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Sub(const TensorData& out, const TensorData& self, const TensorData& other)
{
    auto tself = From(self);
    auto tother = From(other);
    auto tout = From(out);

    std::vector<int64_t> shape_self = tself.second.sizes().vec();
    std::vector<int64_t> shape_other = tother.second.sizes().vec();

    if (shape_self.size() == 0x2 && shape_other.size() == 0x2 && shape_self[0] == shape_other[0] &&
        shape_self[1] != shape_other[1]) {
        if (shape_other[1] == 0x8) {
            int64_t cols_self = shape_self[1];
            int64_t cols_other = shape_other[1];
            int64_t repeat_times = (cols_self + cols_other - 1) / cols_other;
            auto tother_expanded = tother.second.repeat({1, repeat_times});
            auto tother_final = tother_expanded.index({torch::indexing::Slice(), torch::indexing::Slice(0, cols_self)});
            torch::sub_out(tout.second, tself.second, tother_final);
        } else {
            torch::sub_out(tout.second, tself.second, tother.second);
        }
    } else {
        torch::sub_out(tout.second, tself.second, tother.second);
    }
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Mul(const TensorData& out, const TensorData& self, const TensorData& other)
{
    auto tself = From(self);
    auto tother = From(other);
    auto tout = From(out);

    std::vector<int64_t> shape_self = tself.second.sizes().vec();
    std::vector<int64_t> shape_other = tother.second.sizes().vec();

    if (shape_self.size() == 0x2 && shape_other.size() == 0x2 && shape_self[0] == shape_other[0] &&
        shape_self[1] != shape_other[1]) {
        if (shape_other[1] == 0x8) {
            int64_t cols_self = shape_self[1];
            int64_t cols_other = shape_other[1];
            int64_t repeat_times = (cols_self + cols_other - 1) / cols_other;
            auto tother_expanded = tother.second.repeat({1, repeat_times});
            auto tother_final = tother_expanded.index({torch::indexing::Slice(), torch::indexing::Slice(0, cols_self)});
            torch::mul_out(tout.second, tself.second, tother_final);
        } else {
            torch::mul_out(tout.second, tself.second, tother.second);
        }
    } else {
        torch::mul_out(tout.second, tself.second, tother.second);
    }
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Div(const TensorData& out, const TensorData& self, const TensorData& other)
{
    auto tself = From(self);
    auto tother = From(other);
    auto tout = From(out);

    std::vector<int64_t> shape_self = tself.second.sizes().vec();
    std::vector<int64_t> shape_other = tother.second.sizes().vec();

    if (shape_self.size() == 0x2 && shape_other.size() == 0x2 && shape_self[0] == shape_other[0] &&
        shape_self[1] != shape_other[1]) {
        if (shape_other[1] == 0x8) {
            int64_t cols_self = shape_self[1];
            int64_t cols_other = shape_other[1];
            int64_t repeat_times = (cols_self + cols_other - 1) / cols_other;
            auto tother_expanded = tother.second.repeat({1, repeat_times});
            auto tother_final = tother_expanded.index({torch::indexing::Slice(), torch::indexing::Slice(0, cols_self)});
            torch::div_out(tout.second, tself.second, tother_final);
        } else {
            torch::div_out(tout.second, tself.second, tother.second);
        }
    } else {
        torch::div_out(tout.second, tself.second, tother.second);
    }
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Hypot(const TensorData& out, const TensorData& self, const TensorData& other)
{
    auto tself = From(self);
    auto tother = From(other);
    auto tout = From(out);
    std::vector<int64_t> shape_self = tself.second.sizes().vec();
    std::vector<int64_t> shape_other = tother.second.sizes().vec();

    if (shape_self.size() == 0x2 && shape_other.size() == 0x2 && shape_self[0] == shape_other[0] &&
        shape_self[1] != shape_other[1]) {
        if (shape_other[1] == 0x8) {
            int64_t cols_self = shape_self[1];
            int64_t cols_other = shape_other[1];
            int64_t repeat_times = (cols_self + cols_other - 1) / cols_other;
            auto tother_expanded = tother.second.repeat({1, repeat_times});
            auto tother_final = tother_expanded.index({torch::indexing::Slice(), torch::indexing::Slice(0, cols_self)});
            torch::hypot_out(tout.second, tself.second, tother_final);
        } else {
            torch::hypot_out(tout.second, tself.second, tother.second);
        }
    } else {
        torch::hypot_out(tout.second, tself.second, tother.second);
    }
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Fmod(const TensorData& out, const TensorData& self, const TensorData& other)
{
    auto tself = From(self);
    auto tother = From(other);
    auto tout = From(out);

    std::vector<int64_t> shape_self = tself.second.sizes().vec();
    std::vector<int64_t> shape_other = tother.second.sizes().vec();

    if (shape_self.size() == 0x2 && shape_other.size() == 0x2 && shape_self[0] == shape_other[0] &&
        shape_self[1] != shape_other[1]) {
        if (shape_other[1] == 0x8) {
            int64_t cols_self = shape_self[1];
            int64_t cols_other = shape_other[1];
            int64_t repeat_times = (cols_self + cols_other - 1) / cols_other;
            auto tother_expanded = tother.second.repeat({1, repeat_times});
            auto tother_final = tother_expanded.index({torch::indexing::Slice(), torch::indexing::Slice(0, cols_self)});
            torch::fmod_out(tout.second, tself.second, tother_final);
        } else {
            torch::fmod_out(tout.second, tself.second, tother.second);
        }
    } else {
        torch::fmod_out(tout.second, tself.second, tother.second);
    }
    ToOperand(tout.second, tout.first, out.dtype);
}

static void PReLU(const TensorData& out, const TensorData& self, const TensorData& weight)
{
    auto tout = From(out);
    auto tself = From(self);
    auto tweight = From(weight);

    auto result = torch::where(tself.second >= 0, tself.second, tweight.second * tself.second);
    tout.second.copy_(result);
    ToOperand(tout.second, tout.first, out.dtype);
}

#define DEFINE_BINARY_OPS(Name, op_out)                                                      \
    static void Name(const TensorData& out, const TensorData& self, const TensorData& other) \
    {                                                                                        \
        auto tout = From(out);                                                               \
        auto tself = From(self);                                                             \
        auto tother = From(other);                                                           \
        torch::op_out(tout.second, tself.second, tother.second);                             \
        ToOperand(tout.second, tout.first, out.dtype);                                       \
    }

DEFINE_BINARY_OPS(Remainder, remainder_out)
DEFINE_BINARY_OPS(Gcd, gcd_out)
DEFINE_BINARY_OPS(Pow, pow_out)
DEFINE_BINARY_OPS(FloorDiv, floor_divide_out)

static void BitwiseAnd(const TensorData& out, const TensorData& self, const TensorData& other)
{
    auto tout = From(out);
    auto tself = From(self);
    auto tother = From(other);
    torch::bitwise_and_out(tout.second, tself.second, tother.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void BitwiseOr(const TensorData& out, const TensorData& self, const TensorData& other)
{
    auto tout = From(out);
    auto tself = From(self);
    auto tother = From(other);
    torch::bitwise_or_out(tout.second, tself.second, tother.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void BitwiseXor(const TensorData& out, const TensorData& self, const TensorData& other)
{
    auto tout = From(out);
    auto tself = From(self);
    auto tother = From(other);
    torch::bitwise_xor_out(tout.second, tself.second, tother.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void ExpandExpDif(const TensorData& out, const TensorData& self, const TensorData& other)
{
    auto tself = From(self);
    auto tother = From(other);
    auto tout = From(out);

    auto shape = tself.second.sizes().vec();
    auto expand = tother.second.expand(torch::IntArrayRef(shape));

    torch::sub_out(tout.second, tself.second, expand);
    torch::exp_out(tout.second, tout.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void CopySign(const TensorData& out, const TensorData& self, const TensorData& other)
{
    auto tout = From(out);
    auto tself = From(self);
    auto tother = From(other);
    torch::copysign_out(tout.second, tself.second, tother.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void BitwiseRightShift(const TensorData& out, const TensorData& self, const TensorData& other)
{
    auto tout = From(out);
    auto tself = From(self);
    auto tother = From(other);
    torch::bitwise_right_shift_out(tout.second, tself.second, tother.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void BitwiseLeftShift(const TensorData& out, const TensorData& self, const TensorData& other)
{
    auto tout = From(out);
    auto tself = From(self);
    auto tother = From(other);
    torch::bitwise_left_shift_out(tout.second, tself.second, tother.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void BitwiseRightShiftS(const TensorData& out, const TensorData& self, const Element& scalar)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::bitwise_right_shift_out(tout.second, tself.second, From(scalar));
    ToOperand(tout.second, tout.first, out.dtype);
}

static void BitwiseLeftShiftS(const TensorData& out, const TensorData& self, const Element& scalar)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::bitwise_left_shift_out(tout.second, tself.second, From(scalar));
    ToOperand(tout.second, tout.first, out.dtype);
}

static void SBitwiseRightShift(const TensorData& out, const Element& scalar, const TensorData& other)
{
    auto tout = From(out);
    auto tother = From(other);
    torch::bitwise_right_shift_out(tout.second, From(scalar), tother.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void SBitwiseLeftShift(const TensorData& out, const Element& scalar, const TensorData& other)
{
    auto tout = From(out);
    auto tother = From(other);
    torch::bitwise_left_shift_out(tout.second, From(scalar), tother.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void GcdS(const TensorData& out, const TensorData& self, const Element& scalar)
{
    auto tout = From(out);
    auto tself = From(self);
    auto tdata = torch::tensor(0);
    if (out.dtype == DataType::DT_UINT8) {
        tdata = torch::tensor(static_cast<uint8_t>(scalar.GetUnsignedData()), torch::dtype(torch::kUInt8));
    } else {
        tdata = torch::tensor(scalar.GetSignedData());
    }
    torch::gcd_out(tout.second, tself.second, tdata);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Cast(const TensorData& out, const TensorData& self, CastMode mode)
{
    auto tout = From(out);
    auto tself = From(self);
    if (mode == CastMode::CAST_ROUND) {
        ToOperand(tself.second.round(), tout.first, out.dtype);
    } else if (mode == CastMode::CAST_FLOOR) {
        ToOperand(tself.second.floor(), tout.first, out.dtype);
    } else if (mode == CastMode::CAST_CEIL) {
        ToOperand(tself.second.ceil(), tout.first, out.dtype);
    } else if (mode == CastMode::CAST_TRUNC) {
        ToOperand(tself.second.trunc(), tout.first, out.dtype);
    } else {
        if (IsFloat(out.dtype)) {
            ToOperand(tself.second, tout.first, out.dtype);
        } else {
            ToOperand(tself.second.round(), tout.first, out.dtype);
        }
    }
}

static void Min(const TensorData& out, const TensorData& self, const TensorData& other)
{
    auto tself = From(self);
    auto tother = From(other);
    auto tout = From(out);

    std::vector<int64_t> shape_self = tself.second.sizes().vec();
    std::vector<int64_t> shape_other = tother.second.sizes().vec();

    if (shape_self.size() == 0x2 && shape_other.size() == 0x2 && shape_self[0] == shape_other[0] &&
        shape_self[1] != shape_other[1]) {
        if (shape_other[1] == 0x8) {
            int64_t cols_self = shape_self[1];
            int64_t cols_other = shape_other[1];
            int64_t repeat_times = (cols_self + cols_other - 1) / cols_other;
            auto tother_expanded = tother.second.repeat({1, repeat_times});
            auto tother_final = tother_expanded.index({torch::indexing::Slice(), torch::indexing::Slice(0, cols_self)});
            torch::min_out(tout.second, tself.second, tother_final);
        } else {
            torch::min_out(tout.second, tself.second, tother.second);
        }
    } else {
        torch::min_out(tout.second, tself.second, tother.second);
    }
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Max(const TensorData& out, const TensorData& self, const TensorData& other)
{
    auto tself = From(self);
    auto tother = From(other);
    auto tout = From(out);

    std::vector<int64_t> shape_self = tself.second.sizes().vec();
    std::vector<int64_t> shape_other = tother.second.sizes().vec();

    if (shape_self.size() == 0x2 && shape_other.size() == 0x2 && shape_self[0] == shape_other[0] &&
        shape_self[1] != shape_other[1]) {
        if (shape_other[1] == 0x8) {
            int64_t cols_self = shape_self[1];
            int64_t cols_other = shape_other[1];
            int64_t repeat_times = (cols_self + cols_other - 1) / cols_other;
            auto tother_expanded = tother.second.repeat({1, repeat_times});
            auto tother_final = tother_expanded.index({torch::indexing::Slice(), torch::indexing::Slice(0, cols_self)});
            torch::max_out(tout.second, tself.second, tother_final);
        } else {
            torch::max_out(tout.second, tself.second, tother.second);
        }
    } else {
        torch::max_out(tout.second, tself.second, tother.second);
    }
    ToOperand(tout.second, tout.first, out.dtype);
}

static void MinS(const TensorData& out, const TensorData& self, const Element& elem)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::clamp_max_out(tout.second, tself.second, From(elem));
    ToOperand(tout.second, tout.first, out.dtype);
}

static void MaxS(const TensorData& out, const TensorData& self, const Element& elem)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::clamp_min_out(tout.second, tself.second, From(elem));
    ToOperand(tout.second, tout.first, out.dtype);
}

static void LReLU(const TensorData& out, const TensorData& self,
                  const Element& negative_slope = Element(DataType::DT_FP32, 0.01))
{
    auto tout = From(out);
    auto tself = From(self);
    torch::leaky_relu_out(tout.second, tself.second, From(negative_slope));
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Range(const TensorData& out, const Element& start, const Element& end, const Element& step)
{
    auto tmp = torch::arange(From(start), From(end), From(step));
    int64_t expected_numel = 1;
    for (int64_t dim : out.shape) {
        expected_numel *= dim;
    }
    ASSERT(CalculatorErrorScene::RANGE_NUMEL_MISMATCH, tmp.numel() == expected_numel)
        << "Range numel mismatch: generated " << tmp.numel() << ", expected " << expected_numel;
    auto tout = From(out);
    tout.second.copy_(tmp);
    ToOperand(tout.second, tout.first, out.dtype);
}

static uint32_t MultiplyHighLow(uint32_t a, uint32_t b, uint32_t& hi)
{
    uint64_t product = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
    hi = static_cast<uint32_t>(product >> 0x20);
    return static_cast<uint32_t>(product & 0xFFFFFFFF);
}

static void PhiloxRandomGolden(std::vector<uint32_t>& counter, std::vector<uint32_t>& key, int rounds)
{
    for (int i = 0; i < rounds; ++i) {
        uint32_t hi0, hi1;
        uint32_t lo0 = MultiplyHighLow(0xD2511F53, counter[0], hi0);
        uint32_t lo1 = MultiplyHighLow(0xCD9E8D57, counter[2], hi1);

        counter = {hi1 ^ counter[1] ^ key[0], lo1, hi0 ^ counter[3] ^ key[1], lo0};

        key[0] += 0x9E3779B9;
        key[1] += 0xBB67AE85;
    }
}

static void Uniform(const TensorData& out, const Element& key, const Element& counter0, const Element& counter1,
                    const Element& rounds, DataType dtype)
{
    std::vector<uint32_t> keyVec(0x2);
    keyVec[0] = static_cast<uint32_t>(key.Cast<uint64_t>() & 0xFFFFFFFF);
    keyVec[1] = static_cast<uint32_t>(key.Cast<uint64_t>() >> 0x20);

    std::vector<uint32_t> counterVec(0x4);
    counterVec[0] = static_cast<uint32_t>(counter0.Cast<uint64_t>() & 0xFFFFFFFF);
    counterVec[1] = static_cast<uint32_t>(counter0.Cast<uint64_t>() >> 0x20);
    counterVec[2] = static_cast<uint32_t>(counter1.Cast<uint64_t>() & 0xFFFFFFFF);
    counterVec[3] = static_cast<uint32_t>(counter1.Cast<uint64_t>() >> 0x20);

    int64_t totalElements = 1;
    for (int64_t dim : out.shape) {
        totalElements *= dim;
    }

    std::vector<uint32_t> result(totalElements);
    std::vector<uint32_t> currentKey = keyVec;
    std::vector<uint32_t> currentCounter = counterVec;

    uint16_t roundsVal = rounds.Cast<uint16_t>();

    for (int64_t i = 0; i < totalElements; i += 0x4) {
        PhiloxRandomGolden(currentCounter, currentKey, roundsVal);

        for (int j = 0; j < 0x4 && (i + j) < totalElements; ++j) {
            result[i + j] = currentCounter[j];
        }

        currentCounter[0]++;
        if (currentCounter[0] == 0) {
            currentCounter[1]++;
            if (currentCounter[1] == 0) {
                currentCounter[2]++;
                if (currentCounter[2] == 0) {
                    currentCounter[3]++;
                }
            }
        }
    }

    auto tout = From(out);

    if (dtype == DT_FP32) {
        std::vector<float> resultFloat(totalElements);
        for (int64_t i = 0; i < totalElements; ++i) {
            uint32_t x = result[i];
            uint32_t man = x & 0x7fffff;
            uint32_t exp = 127;
            uint32_t val = (exp << 23) | man;
            float f;
            memcpy_s(&f, sizeof(f), &val, sizeof(val));
            resultFloat[i] = f - 1.0f;
        }
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        auto tmp = torch::from_blob(resultFloat.data(), {totalElements}, options).clone();
        tout.second.copy_(tmp.reshape(tout.second.sizes()));
    } else if (dtype == DT_FP16) {
        std::vector<uint16_t> resultHalf(totalElements);
        for (int64_t i = 0; i < totalElements; ++i) {
            uint32_t x = result[i];
            uint16_t x16 = static_cast<uint16_t>(x & 0xFFFF);
            uint16_t man = x16 & 0x3ff;
            uint16_t exp = 15;
            uint16_t val = (exp << 10) | man;
            resultHalf[i] = val;
        }
        auto options = torch::TensorOptions().dtype(torch::kInt16);
        auto tmp = torch::from_blob(resultHalf.data(), {totalElements}, options).clone();
        auto tmpHalf = tmp.to(torch::kFloat16);
        auto tmpFloat = (tmpHalf - torch::scalar_tensor(1.0, torch::kFloat16)).to(torch::kFloat32);
        tout.second.copy_(tmpFloat.reshape(tout.second.sizes()).to(torch::kFloat16));
    } else if (dtype == DT_BF16) {
        std::vector<uint16_t> resultBfloat16(totalElements);
        for (int64_t i = 0; i < totalElements; ++i) {
            uint32_t x = result[i];
            uint16_t x16 = static_cast<uint16_t>(x & 0xFFFF);
            uint16_t man = x16 & 0x7f;
            uint16_t exp = 127;
            uint16_t val = (exp << 7) | man;
            resultBfloat16[i] = val;
        }
        auto options = torch::TensorOptions().dtype(torch::kInt16);
        auto tmp = torch::from_blob(resultBfloat16.data(), {totalElements}, options).clone();
        auto tmpBfloat16 = tmp.to(torch::kBFloat16);
        auto tmpFloat = (tmpBfloat16 - torch::scalar_tensor(1.0, torch::kBFloat16)).to(torch::kFloat32);
        tout.second.copy_(tmpFloat.reshape(tout.second.sizes()).to(torch::kBFloat16));
    } else {
        std::vector<float> resultFloat(totalElements);
        for (int64_t i = 0; i < totalElements; ++i) {
            uint32_t x = result[i];
            uint32_t man = x & 0x7fffff;
            uint32_t exp = 127;
            uint32_t val = (exp << 23) | man;
            float f;
            memcpy_s(&f, sizeof(f), &val, sizeof(val));
            resultFloat[i] = f - 1.0f;
        }
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        auto tmp = torch::from_blob(resultFloat.data(), {totalElements}, options).clone();
        tout.second.copy_(tmp.reshape(tout.second.sizes()));
    }
    ToOperand(tout.second, tout.first, out.dtype);
}

template <typename T>
static void CompareImpl(const TensorData& out, const torch::Tensor& tself, const T& other_op,
                        CmpOperationType operation, CmpModeType mode)
{
    auto tout = From(out);
    torch::Tensor tmp_result;
    switch (operation) {
        case CmpOperationType::EQ:
            tmp_result = torch::eq(tself, other_op);
            break;
        case CmpOperationType::NE:
            tmp_result = torch::ne(tself, other_op);
            break;
        case CmpOperationType::LT:
            tmp_result = torch::lt(tself, other_op);
            break;
        case CmpOperationType::LE:
            tmp_result = torch::le(tself, other_op);
            break;
        case CmpOperationType::GT:
            tmp_result = torch::gt(tself, other_op);
            break;
        case CmpOperationType::GE:
            tmp_result = torch::ge(tself, other_op);
            break;
        default:
            ASSERT(CalculatorErrorScene::COMPARE_UNSUPPORTED_TYPE, false) << "Unsupported compare type";
            break;
    }

    if (mode == CmpModeType::BIT) {
        if (tmp_result.dim() > 0) {
            int64_t last_dim = tmp_result.size(-1);
            ASSERT(CalculatorErrorScene::BITMODE_LAST_DIM_INVALID, last_dim % NUM_VALUE_8 == 0)
                << "Last dimension must be divisible by 8 in BIT mode";

            auto shape = tmp_result.sizes().vec();
            shape.back() = last_dim / NUM_VALUE_8;

            torch::Tensor packed = torch::empty(shape, torch::kUInt8);
            auto tmp_result_contig = tmp_result.contiguous();
            auto tmp_data = tmp_result_contig.data_ptr<bool>();
            auto packed_data = packed.data_ptr<uint8_t>();

            const int64_t num_elements = tmp_result.numel();
            for (int64_t i = 0; i < num_elements / NUM_VALUE_8; ++i) {
                uint8_t byte = 0;
                for (int j = 0; j < NUM_VALUE_8; ++j) {
                    if (tmp_data[i * NUM_VALUE_8 + j]) {
                        byte |= (1 << j);
                    }
                }
                packed_data[i] = byte;
            }
            tout.second.copy_(packed);
            ToOperand(tout.second, tout.first, out.dtype);
        }
    } else {
        tout.second.copy_(tmp_result);
        ToOperand(tout.second, tout.first, out.dtype);
    }
}

static void Compare(const TensorData& out, const TensorData& self, const TensorData& other, CmpOperationType operation,
                    CmpModeType mode)
{
    CompareImpl(out, From(self).second, From(other).second, operation, mode);
}

static void Cmps(const TensorData& out, const TensorData& self, const Element& elem, CmpOperationType operation,
                 CmpModeType mode)
{
    CompareImpl(out, From(self).second, From(elem), operation, mode);
}

#define DEFINE_BINARY_PAIR_OPS(Name, bop)                                                          \
    static void Pair##Name(const TensorData& out, const TensorData& self, const TensorData& other) \
    {                                                                                              \
        auto big = self, small = other;                                                            \
        if (self.shape < other.shape) {                                                            \
            big = other, small = self;                                                             \
        }                                                                                          \
        auto tout = From(out);                                                                     \
        std::vector<int64_t> offset(self.shape.size(), 0);                                         \
        auto tbig = View(tout.second, big.shape, offset);                                          \
        tbig.copy_(From(big).second);                                                              \
        auto tsmall = View(tout.second, small.shape, offset);                                      \
        torch::bop(tsmall, tsmall, From(small).second);                                            \
        ToOperand(tout.second, tout.first, out.dtype);                                             \
    }

DEFINE_BINARY_PAIR_OPS(Sum, add_out)
DEFINE_BINARY_PAIR_OPS(Max, max_out)
DEFINE_BINARY_PAIR_OPS(Min, min_out)
DEFINE_BINARY_PAIR_OPS(Prod, mul_out)

std::vector<int64_t> GenAxesForTranspose(const int64_t offset, const std::vector<int64_t>& base)
{
    std::vector<int64_t> axes;
    for (int64_t i = 0; i < offset; i++) {
        axes.push_back(i);
    }
    for (auto x : base) {
        axes.push_back(x + offset);
    }
    return axes;
}

static inline int64_t alignup(int64_t x, int64_t align) { return (x + (align - 1)) & ~(align - 1); }

static void FormatND2NZ(const TensorData& out, const TensorData& self)
{
    auto& shape = self.shape;
    ASSERT(CalculatorErrorScene::FORMAT_ND2NZ_RANK_LT_2, shape.size() >= 0x2)
        << "Input tensor must have at least 2 dimensions";

    int64_t ndim = shape.size();
    int64_t m = shape[ndim - 0x2];
    int64_t m0 = 16; // m0 16
    int64_t padm = alignup(m, m0);
    auto tself_pair = From(self);
    // Under mixed call paths, FP4 shape metadata may still be packed in some places.
    // Use actual float-view tensor width as source of truth for ND<->NZ transform.
    int64_t nFloat = tself_pair.second.size(tself_pair.second.dim() - 1);
    int64_t nPacked = LastDimPackedCount(nFloat, self.dtype);
    int64_t n0Packed = BLOCK_SIZE / BytesOf(self.dtype);
    int64_t padnPacked = alignup(nPacked, n0Packed);
    int64_t padnFloat = LastDimFloatCount(padnPacked, self.dtype);
    int64_t n1 = padnPacked / n0Packed;
    int64_t n0Float = LastDimFloatCount(n0Packed, self.dtype);

    auto tself = tself_pair.second.reshape({-1, m, nFloat}); // [b, m1*m0, n1*n0] in float elems
    if (padm != m || padnPacked != nPacked) {
        tself = torch::constant_pad_nd(tself, {0, padnFloat - nFloat, 0, padm - m}, 0); // [b, padm, padn]
    }

    tself = tself.reshape({-1, padm, n1, n0Float}); // [b, padm, n1, n0]
    tself = tself.permute({0, 0x2, 1, 0x3});        // [b, n1, padm, n0]

    std::vector<int64_t> nzShape(shape.begin(), shape.end() - 2); // remove last 2 dim, keep only batch dims
    nzShape.push_back(padm);
    nzShape.push_back(IsFp4PackedDtype(self.dtype) ? padnFloat : padnPacked);
    tself = tself.reshape(nzShape); // [b, padm, padn]
    auto tout = From(out);
    ToOperand(tself, tout.first, out.dtype);
}

static void FormatNZ2ND(const TensorData& out, const TensorData& self)
{
    auto& shape = self.shape;
    ASSERT(CalculatorErrorScene::FORMAT_NZ2ND_RANK_LT_2, shape.size() >= 0x2)
        << "Input tensor must have at least 2 dimensions";

    auto tself_pair = From(self);
    auto tself = tself_pair.second; // [b, m1*m0, n1*n0]
    int64_t ndim = shape.size();
    int64_t m = shape[ndim - 0x2];
    int64_t n0Packed = BLOCK_SIZE / BytesOf(self.dtype);
    int64_t n0Float = LastDimFloatCount(n0Packed, self.dtype);
    // Trans() may expand FP4 last dim to logical width while NZ storage last dim is
    // alignup(packed, n0Packed) (see FormatND2NZ). Using unpacked nPacked alone yields n1==0.
    int64_t selfLastFloat = tself.size(tself.dim() - 1);
    int64_t nPackedUnc = LastDimPackedCount(selfLastFloat, self.dtype);
    int64_t nPacked = IsFp4PackedDtype(self.dtype) ? alignup(nPackedUnc, n0Packed) : nPackedUnc;
    int64_t n1 = nPacked / n0Packed;

    tself = tself.reshape({-1, n1, m, n0Float}); // [b, n1, m1*m0, n0]
    tself = tself.permute({0, 0x2, 1, 0x3});     // [b, m1*m0, n1, n0]
    tself = tself.reshape(shape);                // [b, m1*m0, n1*n0] float elems

    std::vector<int64_t> offset(ndim, 0);
    auto tout = From(out);
    auto view = View(tself, out.shape, offset);
    tout.second.copy_(view);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void MatmulMultiDataLoad(torch::Tensor& out, const torch::Tensor& lhs, const torch::Tensor& rhs,
                                const torch::Tensor& bias, int64_t kstep)
{
    auto shapeL = lhs.sizes().vec();
    auto shapeR = rhs.sizes().vec();
    auto offsetL = std::vector<int64_t>(shapeL.size(), 0);
    auto offsetR = std::vector<int64_t>(shapeR.size(), 0);
    int64_t kdimL = shapeL.size() - 1;
    int64_t kdimR = shapeR.size() - 0x2;
    int64_t k = shapeL[kdimL];
    auto biasShape = bias.sizes().vec();
    for (int64_t offset = 0; offset < k; offset += kstep) {
        shapeL[kdimL] = std::min(kstep, k - offset);
        shapeR[kdimR] = std::min(kstep, k - offset);
        offsetL[kdimL] = offset;
        offsetR[kdimR] = offset;
        auto viewL = View(lhs, shapeL, offsetL);
        auto viewR = View(rhs, shapeR, offsetR);
        out.add_(torch::matmul(viewL, viewR));
    }
    if (biasShape.size() == 0x2) {
        out.add_(bias);
    }
}

static void QuantExecute(torch::Tensor& tout, const TensorData* scalePtr, uint64_t scale, int relu)
{
    if (relu == 1) {
        tout.relu_();
    }
    if (scale != 0) {
        uint32_t low32 = static_cast<uint32_t>(scale & 0xFFFFE000);
        float scaleValue = 0.0;
        memcpy_s(&scaleValue, sizeof(float), &low32, sizeof(float));
        tout.mul_(scaleValue);
    } else if (scalePtr != nullptr) {
        auto scaleTensor = From(*scalePtr);
        auto scaleU32 = scaleTensor.second.to(torch::kInt32);
        auto* u32Data = scaleU32.data_ptr<int32_t>();
        auto scaleF32 = torch::from_blob(reinterpret_cast<float*>(u32Data), scaleU32.sizes(),
                                         torch::TensorOptions().dtype(torch::kFloat32))
                            .clone();
        tout.mul_(scaleF32);
    }
}

static void QuantPreCompute(const TensorData& out, const TensorData& self, const TensorData* scalePtr, uint64_t scale,
                            int relu)
{
    ASSERT(CalculatorErrorScene::QUANTPRECOMPUTE_NULL_DATAPTR, out.dataPtr != nullptr && self.dataPtr != nullptr);
    ASSERT(CalculatorErrorScene::QUANTPRECOMPUTE_DTYPE_MISMATCH,
           out.dtype == DataType::DT_FP16 && self.dtype == DataType::DT_INT32);
    auto tself = From(self);
    auto tout = From(out);
    auto dtype = tout.second.scalar_type();
    auto calcType = dtype;
    if (dtype == torch::kFloat16 || dtype == torch::kBFloat16) {
        calcType = torch::kFloat;
        tout.second = tout.second.to(calcType);
    }
    tout.second.copy_(tself.second);
    QuantExecute(tout.second, scalePtr, scale, relu);
    if (calcType != dtype) {
        tout.second = tout.second.to(dtype);
    }
    ToOperand(tout.second, tout.first, out.dtype);
}

static void CheckMXScaleBatchDims(const torch::Tensor& matrix, const torch::Tensor& scale, const std::string& scaleName)
{
    ASSERT(CalculatorErrorScene::MATMUL_INPUT_SHAPE_MISMATCH, scale.dim() == matrix.dim() + 1)
        << "MX " << scaleName << " dim must equal its paired matrix dim plus 1, matrix dim: " << matrix.dim()
        << ", scale dim: " << scale.dim();
    const int64_t batchDim = matrix.dim() - 0x2;
    for (int64_t dim = 0; dim < batchDim; ++dim) {
        ASSERT(CalculatorErrorScene::MATMUL_INPUT_SHAPE_MISMATCH, scale.size(dim) == matrix.size(dim))
            << "MX " << scaleName << " batch dim " << dim
            << " must equal its paired matrix batch dim, matrix: " << matrix.size(dim)
            << ", scale: " << scale.size(dim);
    }
}

static torch::Tensor BuildMXScaleForA(const torch::Tensor& scaleA, const torch::Tensor& matrixA, bool scaleATrans)
{
    CheckMXScaleBatchDims(matrixA, scaleA, "scale_a");
    const int64_t mSize = matrixA.size(-0x2);
    const int64_t kSize = matrixA.size(-1);
    torch::Tensor merged;
    if (!scaleATrans) {
        // test reference: scale_a.flatten(-2, -1)
        ASSERT(CalculatorErrorScene::MATMUL_INPUT_SHAPE_MISMATCH, scaleA.size(-0x3) == mSize && scaleA.size(-1) == 0x2)
            << "MX scale_a shape mismatch, expected [..., M, K/64, 2], got M: " << scaleA.size(-0x3)
            << ", inner dim: " << scaleA.size(-1);
        merged = scaleA.flatten(-0x2, -1);
    } else {
        // test reference: scale_a.transpose(-3, -2).flatten(-2, -1)
        ASSERT(CalculatorErrorScene::MATMUL_INPUT_SHAPE_MISMATCH, scaleA.size(-0x2) == mSize && scaleA.size(-1) == 0x2)
            << "MX trans scale_a shape mismatch, expected [..., K/64, M, 2], got M: " << scaleA.size(-0x2)
            << ", inner dim: " << scaleA.size(-1);
        merged = scaleA.transpose(-0x3, -0x2).flatten(-0x2, -1);
    }
    auto expanded = merged.repeat_interleave(0x20, -1);
    ASSERT(CalculatorErrorScene::MATMUL_INPUT_SHAPE_MISMATCH, expanded.size(-1) >= kSize)
        << "MX scale_a expanded K is smaller than matrix K, got " << expanded.size(-1) << ", expect at least " << kSize;
    return expanded.narrow(-1, 0, kSize);
}

static torch::Tensor BuildMXScaleForB(const torch::Tensor& scaleB, const torch::Tensor& matrixB, bool scaleBTrans)
{
    CheckMXScaleBatchDims(matrixB, scaleB, "scale_b");
    const int64_t kSize = matrixB.size(-0x2);
    const int64_t nSize = matrixB.size(-1);
    torch::Tensor merged;
    if (!scaleBTrans) {
        // test reference: scale_b.transpose(-3, -2).flatten(-2, -1).transpose(-2, -1)
        ASSERT(CalculatorErrorScene::MATMUL_INPUT_SHAPE_MISMATCH, scaleB.size(-0x2) == nSize && scaleB.size(-1) == 0x2)
            << "MX scale_b shape mismatch, expected [..., K/64, N, 2], got N: " << scaleB.size(-0x2)
            << ", inner dim: " << scaleB.size(-1);
        merged = scaleB.transpose(-0x3, -0x2).flatten(-0x2, -1).transpose(-0x2, -1);
    } else {
        // test reference: scale_b.flatten(-2, -1).transpose(-2, -1)
        ASSERT(CalculatorErrorScene::MATMUL_INPUT_SHAPE_MISMATCH, scaleB.size(-0x3) == nSize && scaleB.size(-1) == 0x2)
            << "MX trans scale_b shape mismatch, expected [..., N, K/64, 2], got N: " << scaleB.size(-0x3)
            << ", inner dim: " << scaleB.size(-1);
        merged = scaleB.flatten(-0x2, -1).transpose(-0x2, -1);
    }
    auto expanded = merged.repeat_interleave(0x20, -0x2);
    ASSERT(CalculatorErrorScene::MATMUL_INPUT_SHAPE_MISMATCH, expanded.size(-0x2) >= kSize)
        << "MX scale_b expanded K is smaller than matrix K, got " << expanded.size(-0x2) << ", expect at least "
        << kSize;
    return expanded.narrow(-0x2, 0, kSize);
}

static void MatMul(const TensorData& out, const TensorData& self, const TensorData& other, const TensorData* acc,
                   MatMulParam& param)
{
    auto tout = From(out);
    auto dtype = tout.second.scalar_type();
    auto calcType = dtype;
    if (dtype == torch::kFloat16 || dtype == torch::kBFloat16) {
        calcType = torch::kFloat;
        tout.second = tout.second.to(calcType);
    }

    auto tself = From(self);
    auto tother = From(other);
    std::pair<torch::Tensor, torch::Tensor> bias_tensor;
    if (param.biasPtr != nullptr) {
        bias_tensor = From(*param.biasPtr);
    }
    if (acc) {
        tout.second.copy_(From(*acc).second);
    } else {
        tout.second.zero_();
    }
    if (param.aTrans) {
        tself.second.transpose_(-1, AXIS_TO_LAST);
    }
    if (param.bTrans) {
        tother.second.transpose_(-1, AXIS_TO_LAST);
    }
    if (tself.second.scalar_type() != calcType) {
        tself.second = tself.second.to(calcType);
    }
    if (tother.second.scalar_type() != calcType) {
        tother.second = tother.second.to(calcType);
    }
    if (param.aScalePtr != nullptr && param.bScalePtr != nullptr) {
        auto taScale = From(*param.aScalePtr).second.to(calcType);
        auto tbScale = From(*param.bScalePtr).second.to(calcType);
        const int64_t aDim = tself.second.dim();
        const int64_t bDim = tother.second.dim();
        ASSERT(CalculatorErrorScene::MATMUL_INPUT_SHAPE_MISMATCH, aDim >= 0x2 && aDim <= 0x4 && bDim == aDim)
            << "MX MatMul matrix dim must be 2, 3 or 4, and A/B dim must be equal. A dim: " << aDim
            << ", B dim: " << bDim;
        auto aScaleExpanded = BuildMXScaleForA(taScale, tself.second, param.aScaleTrans);
        auto bScaleExpanded = BuildMXScaleForB(tbScale, tother.second, param.bScaleTrans);
        tself.second = tself.second.mul(aScaleExpanded);
        tother.second = tother.second.mul(bScaleExpanded);
    }
    if (!param.kStep || param.kStep == tself.second.size(-1)) {
        if (param.biasPtr != nullptr) {
            tout.second.add_(torch::matmul(tself.second, tother.second) + bias_tensor.second);
        } else {
            tout.second.add_(torch::matmul(tself.second, tother.second));
        }
    } else {
        MatmulMultiDataLoad(tout.second, tself.second, tother.second, bias_tensor.second, param.kStep);
    }
    if (self.dtype == DataType::DT_INT8 && out.dtype == DataType::DT_FP16 &&
        (param.scale != 0 || param.scalePtr != nullptr)) {
        QuantExecute(tout.second, param.scalePtr, param.scale, param.relu);
    }
    if (calcType != dtype) {
        tout.second = tout.second.to(dtype);
    }
    ToOperand(tout.second, tout.first, out.dtype);
}

void OneHot(const TensorData& out, const TensorData& self, int numClasses)
{
    auto ret = From(out);
    auto src = From(self);
    ret.second.copy_(torch::nn::functional::one_hot(src.second.to(torch::kInt64), numClasses));
    ToOperand(ret.second, ret.first, out.dtype);
}

static void ExpandS(const TensorData& out, const Element& elem)
{
    auto tout = From(out);
    torch::full_out(tout.second, out.shape, From(elem));
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Expand(const TensorData& out, const TensorData& self)
{
    auto tself = From(self);
    if (self.shape[self.shape.size() - 1] != out.shape[out.shape.size() - 1] &&
        self.shape[self.shape.size() - 1] != 1) {
        // possible block align
        tself.second = tself.second.slice(tself.second.dim() - 1, 0, 1);
    }
    auto tout = From(out);
    ToOperand(tself.second, tout.first, out.dtype);
}
void Gather(const TensorData& out, const TensorData& params, const TensorData& indices, int64_t axis)
{
    auto tout = From(out);
    auto tparams = From(params);
    auto tindices = From(indices);
    auto paramsRank = params.shape.size();
    if (axis < 0) {
        axis += paramsRank;
    }
    ASSERT(CalculatorErrorScene::GATHER_AXIS_OUT_OF_RANGE, axis >= 0 && axis < static_cast<int64_t>(paramsRank))
        << "axis " << axis << " out of range [0, " << paramsRank << ")";
    auto idxFlat = tindices.second.to(torch::kLong).reshape({-1});
    auto gathered = tparams.second.index_select(/*dim=*/axis, /*index=*/idxFlat);
    std::vector<int64_t> outSize{};
    outSize.insert(outSize.end(), tparams.second.sizes().begin(), tparams.second.sizes().begin() + axis);
    outSize.insert(outSize.end(), tindices.second.sizes().begin(), tindices.second.sizes().end());
    outSize.insert(outSize.end(), tparams.second.sizes().begin() + axis + 1, tparams.second.sizes().end());
    tout.second = tout.second.view(outSize);
    tout.second.copy_(gathered.reshape(outSize));
    ToOperand(tout.second, tout.first, out.dtype);
}
void GatherINUBGolden(torch::Tensor& out, const torch::Tensor& params, const torch::Tensor& indices,
                      const torch::Tensor& pageTable, int64_t blockSize, int64_t axis)
{
    // ---- 基本约束：只做 CPU，不考虑 CUDA ----
    ASSERT(CalculatorErrorScene::GATHER_INUB_DEVICE_INVALID,
           params.is_cpu() && indices.is_cpu() && pageTable.is_cpu() && out.is_cpu())
        << "CPU-only: params/indices/pageTable/out must all be on CPU.";

    // ---- axis：严格等价你 golden（token 维），只允许 axis==0 ----
    if (axis < 0)
        axis += params.dim();
    ASSERT(CalculatorErrorScene::GATHER_INUB_AXIS_INVALID, axis == 0)
        << "Only axis==0 is supported to match the original golden logic.";
    ASSERT(CalculatorErrorScene::GATHER_INUB_BLOCKSIZE_INVALID, blockSize > 0) << "blockSize must be > 0.";

    // ---- 形状严格限制：indices/pageTable 只能是 [1, a] ----
    ASSERT(CalculatorErrorScene::GATHER_INUB_SHAPE_INVALID, params.dim() == 0x2)
        << "params must be [num_buffer_tokens, hidden_dim]";
    ASSERT(CalculatorErrorScene::GATHER_INUB_SHAPE_INVALID, indices.dim() == 0x2 && indices.size(0) == 1)
        << "indices must be [1, topk_count]";
    ASSERT(CalculatorErrorScene::GATHER_INUB_SHAPE_INVALID, pageTable.dim() == 0x2 && pageTable.size(0) == 1)
        << "pageTable must be [1, num_logical_blocks]";
    ASSERT(CalculatorErrorScene::GATHER_INUB_SHAPE_INVALID, out.dim() == 0x2) << "out must be [topk_count, hidden_dim]";

    const int64_t hidden_dim = params.size(1);
    const int64_t topk_count = indices.size(1);
    const int64_t num_logical_blocks = pageTable.size(1);

    ASSERT(CalculatorErrorScene::GATHER_INUB_SHAPE_INVALID, out.size(0) == topk_count && out.size(1) == hidden_dim)
        << "out must have shape [topk_count, hidden_dim]";

    // ---- dtype：indices/pageTable 必须是整数；统一转 int64（不转 params）----
    ASSERT(CalculatorErrorScene::GATHER_INUB_DTYPE_INVALID,
           indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong)
        << "indices must be int32 or int64";
    ASSERT(CalculatorErrorScene::GATHER_INUB_DTYPE_INVALID,
           pageTable.scalar_type() == at::kInt || pageTable.scalar_type() == at::kLong)
        << "pageTable must be int32 or int64";

    // out/params dtype 必须一致（index_select 不会帮你做 dtype cast）
    ASSERT(CalculatorErrorScene::GATHER_INUB_DTYPE_INVALID, out.scalar_type() == params.scalar_type())
        << "out and params must have the same dtype";

    // ---- 1) logical indices: [topk] int64 ----
    at::Tensor logical = indices.reshape({-1}).to(at::kLong);

    // ---- logical 越界检查： [0, num_logical_blocks * blockSize) ----
    const int64_t total_logical_tokens = num_logical_blocks * blockSize;
    ASSERT(CalculatorErrorScene::GATHER_INUB_LOGICAL_INDEX_INVALID, total_logical_tokens >= 0)
        << "total_logical_tokens overflow?";
    ASSERT(CalculatorErrorScene::GATHER_INUB_LOGICAL_INDEX_INVALID, logical.ge(0).all().item<bool>())
        << "logical_index < 0 exists in indices";
    ASSERT(CalculatorErrorScene::GATHER_INUB_LOGICAL_INDEX_INVALID, logical.lt(total_logical_tokens).all().item<bool>())
        << "logical_index out of range: must be < num_logical_blocks * blockSize";

    // ---- 2) pageTable: [num_logical_blocks] int64 ----
    at::Tensor pt = pageTable.reshape({-1}).to(at::kLong);
    ASSERT(CalculatorErrorScene::GATHER_INUB_PAGETABLE_NUMEL_MISMATCH, pt.numel() == num_logical_blocks)
        << "pageTable numel mismatch";

    // ---- 3) compute physical indices (完全等价 golden) ----
    // logical_block = logical / blockSize
    // offset        = logical % blockSize
    // physical_blk  = pt[logical_block]
    // physical      = physical_blk * blockSize + offset
    at::Tensor logical_block = logical.floor_divide(blockSize); // trunc div for int64
    at::Tensor offset = logical.remainder(blockSize);           // same as % for non-negative

    // 逻辑块 id 范围检查（其实 logical 已经检查过，这里更保险）
    ASSERT(CalculatorErrorScene::GATHER_INUB_LOGICAL_BLOCK_INVALID, logical_block.ge(0).all().item<bool>())
        << "logical_block_id < 0 exists";
    ASSERT(CalculatorErrorScene::GATHER_INUB_LOGICAL_BLOCK_INVALID,
           logical_block.lt(num_logical_blocks).all().item<bool>())
        << "logical_block_id out of range for pageTable";

    at::Tensor physical_block = pt.index_select(0, logical_block);
    at::Tensor physical = physical_block.mul(blockSize).add(offset); // int64

    // ---- physical 越界检查：[0, num_buffer_tokens) ----
    ASSERT(CalculatorErrorScene::GATHER_INUB_PHYSICAL_INDEX_INVALID, physical.ge(0).all().item<bool>())
        << "physical_index < 0 exists";

    // ---- 4) index_select gather: params[physical, :] -> [topk, hidden_dim] ----
    at::Tensor selected = params.index_select(0, physical); // dtype 跟 params 一样

    // 写到 out（不要求 out contiguous；copy_ 会处理）
    out.copy_(selected);
}

void GatherINUB(const TensorData& out, const TensorData& params, const TensorData& indices, const TensorData& pageTable,
                int64_t blockSize, int64_t axis)
{
    auto tout = From(out);
    auto tparams = From(params);
    auto tindices = From(indices);
    auto tpageTable = From(pageTable);
    GatherINUBGolden(tout.second, tparams.second, tindices.second, tpageTable.second, blockSize, axis);
    ToOperand(tout.second, tout.first, out.dtype);
}

void GatherInL1Golden(torch::Tensor& out, const torch::Tensor& params, const torch::Tensor& indices,
                      const torch::Tensor& pageTable, int64_t blockSize)
{
    torch::Tensor logical = indices.reshape({-1}).to(torch::kLong);
    torch::Tensor pt = pageTable.reshape({-1}).to(torch::kLong);
    torch::Tensor logical_block = logical.floor_divide(blockSize);
    torch::Tensor offset = logical.remainder(blockSize);
    torch::Tensor physical_block = torch::index_select(pt, 0, logical_block);
    torch::Tensor physical = physical_block.mul(blockSize).add(offset);
    torch::Tensor selected = torch::index_select(params, 0, physical);
    out.copy_(selected);
}

static torch::Tensor FromGatherInL1(const TensorData& data)
{
    auto tensor = torch::from_blob(data.dataPtr, data.rawShape, FromDataType(data.dtype));
    auto view = tensor.as_strided({data.shape[0], data.shape[1]}, data.stride, data.storageOffset);
    if (data.isAxisCombine) {
        view = view.transpose_(-1, AXIS_TO_LAST);
    }
    return view;
}

void GatherInL1(const TensorData& out, const TensorData& params, const TensorData& indices, const TensorData& pageTable,
                int64_t blockSize)
{
    auto tout = From(out);
    auto tparams = FromGatherInL1(params);
    auto tindices = From(indices);
    auto tpageTable = From(pageTable);
    GatherInL1Golden(tout.second, tparams, tindices.second, tpageTable.second, blockSize);
    ToOperand(tout.second, tout.first, out.dtype);
}

void GatherElements(const TensorData& out, const TensorData& params, const TensorData& indices, int axis)
{
    auto ret = From(out);
    auto src = From(params);
    auto index = From(indices).second.to(torch::kInt64);
    torch::gather_out(ret.second, src.second, axis, index);
    ToOperand(ret.second, ret.first, out.dtype);
}

void GatherMask(const TensorData& out, const TensorData& self, int patternMode)
{
    auto ret = From(out);
    auto src = From(self);
    auto last_dim = src.second.size(src.second.dim() - 1);
    if (patternMode == 0x7) {
        ret.second = src.second;
    } else {
        torch::Tensor selected_indices;
        switch (patternMode) {
            case 0x1:
                selected_indices = torch::arange(0, last_dim, 0x2);
                break;
            case 0x2:
                selected_indices = torch::arange(0x1, last_dim, 0x2);
                break;
            case 0x3:
                selected_indices = torch::arange(0, last_dim, 0x4);
                break;
            case 0x4:
                selected_indices = torch::arange(0x1, last_dim, 0x4);
                break;
            case 0x5:
                selected_indices = torch::arange(0x2, last_dim, 0x4);
                break;
            case 0x6:
                selected_indices = torch::arange(0x3, last_dim, 0x4);
                break;
            default:
                ASSERT(CalculatorErrorScene::GATHERMASK_PATTERNMODE_INVALID, patternMode >= 0x1 && patternMode <= 0x7)
                    << "Invalid patternMode";
        }
        ret.second = src.second.index_select(-1, selected_indices);
    }
    ToOperand(ret.second, ret.first, out.dtype);
}

void IndexAdd(const TensorData& out, const TensorData& self, const TensorData& src, const TensorData& indices, int axis,
              const Element& alpha)
{
    auto tout = From(out);
    auto inputSelf = From(self);
    auto inputSrc = From(src);
    auto inputIndices = From(indices);
    torch::index_add_out(tout.second, inputSelf.second, axis, inputIndices.second, inputSrc.second, From(alpha));
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Quantize(const TensorData& out, const TensorData& input, const TensorData& scale,
                     const TensorData& zeroPoints)
{
    auto tout = From(out);
    auto tinput = From(input);
    auto tscale = From(scale);

    int inputRank = tinput.second.dim();
    int normalizedAxis = inputRank - 1;

    // Broadcast scale to match input shape based on axis
    auto scaleTensor = tscale.second;
    while (scaleTensor.dim() < inputRank) {
        scaleTensor = scaleTensor.unsqueeze(normalizedAxis);
    }
    scaleTensor = scaleTensor.expand_as(tinput.second);

    auto scaled = tinput.second * scaleTensor;

    // Apply zero_points for asymmetric quantization
    if (zeroPoints.dataPtr != nullptr) {
        auto tzeroPoints = From(zeroPoints);
        auto zeroPointsTensor = tzeroPoints.second;

        // Broadcast zero_points based on axis (same as scale)
        while (zeroPointsTensor.dim() < inputRank) {
            zeroPointsTensor = zeroPointsTensor.unsqueeze(normalizedAxis);
        }
        zeroPointsTensor = zeroPointsTensor.expand_as(tinput.second);

        scaled = scaled + zeroPointsTensor;
    }

    auto rounded = torch::round(scaled);

    if (out.dtype == DT_UINT8) {
        rounded = torch::clamp(rounded, 0, 0xff);
    } else if (out.dtype == DT_INT8) {
        rounded = torch::clamp(rounded, -0x80, 0x7f);
    }

    ToOperand(rounded, tout.first, out.dtype);
}

static void Dequantize(const TensorData& out, const TensorData& input, const TensorData& scale,
                       const TensorData& zeroPoints)
{
    auto tout = From(out);
    auto tinput = From(input);
    auto tscale = From(scale);

    int inputRank = tinput.second.dim();
    int normalizedAxis = inputRank - 1;

    // Broadcast scale to match input shape based on axis
    auto scaleTensor = tscale.second;
    while (scaleTensor.dim() < inputRank) {
        scaleTensor = scaleTensor.unsqueeze(normalizedAxis);
    }
    scaleTensor = scaleTensor.expand_as(tinput.second);

    auto result = tinput.second * scaleTensor;

    // Apply zero_points for asymmetric dequantization
    if (zeroPoints.dataPtr != nullptr) {
        auto tzeroPoints = From(zeroPoints);
        auto zeroPointsTensor = tzeroPoints.second;

        // Broadcast zero_points based on axis (same as scale)
        while (zeroPointsTensor.dim() < inputRank) {
            zeroPointsTensor = zeroPointsTensor.unsqueeze(normalizedAxis);
        }
        zeroPointsTensor = zeroPointsTensor.expand_as(tinput.second);

        result = result - zeroPointsTensor;
    }

    ToOperand(result, tout.first, out.dtype);
}

void TriU(const TensorData& out, const TensorData& in, int diagonal)
{
    auto output = From(out);
    auto input = From(in);

    torch::triu_out(output.second, input.second, diagonal);
    ToOperand(output.second, output.first, out.dtype);
}

void TriL(const TensorData& out, const TensorData& in, int diagonal)
{
    auto output = From(out);
    auto input = From(in);

    torch::tril_out(output.second, input.second, diagonal);
    ToOperand(output.second, output.first, out.dtype);
}

void CumSum(const TensorData& out, const TensorData& in, int axis)
{
    auto tout = From(out);
    auto input = From(in);
    torch::cumsum_out(tout.second, input.second, axis);
    ToOperand(tout.second, tout.first, out.dtype);
}

void CumProd(const TensorData& out, const TensorData& in, int axis)
{
    auto tout = From(out);
    auto input = From(in);
    torch::cumprod_out(tout.second, input.second, axis);
    ToOperand(tout.second, tout.first, out.dtype);
}

void IndexPut(const TensorData& out, const TensorData& self, const std::vector<TensorData>& indices,
              const TensorData& values, bool accumulate)
{
    c10::List<c10::optional<at::Tensor>> indicesList;
    for (auto idx : indices) {
        indicesList.push_back(From(idx).second);
    }
    auto tout = From(out);
    auto result = torch::index_put(From(self).second, indicesList, From(values).second, accumulate);
    ToOperand(result, tout.first, out.dtype);
}

static void Atan(const TensorData& out, const TensorData& in)
{
    auto tout = From(out);
    auto input = From(in);
    torch::atan_out(tout.second, input.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Atan2(const TensorData& out, const TensorData& y, const TensorData& x)
{
    auto tout = From(out);
    auto input0 = From(y);
    auto input1 = From(x);
    torch::atan2_out(tout.second, input0.second, input1.second);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Copy(const TensorData& out, const TensorData& self, bool trans, bool isMx)
{
    auto tout = From(out);
    auto tself = From(self);
    if (trans) {
        auto res = tself.second.transpose(-1, AXIS_TO_LAST);
        if (isMx) {
            // scaleMM输入量化参数为三维(m0,k0,2),转置需要对高两维进行转置操作
            res = tself.second.transpose(0, 1);
        }
        ToOperand(res, tout.first, out.dtype);
    } else {
        ToOperand(tself.second, tout.first, out.dtype);
    }
}

static torch::Tensor InterleaveTensor(const torch::Tensor& src0, const torch::Tensor& src1)
{
    std::vector<torch::Tensor> inputs = {src0, src1};
    auto interleavedShape = src0.sizes().vec();
    interleavedShape.back() *= 0x2;
    return torch::stack(inputs, -1).reshape(interleavedShape);
}

static void Interleave(const TensorData& out0, const TensorData& out1, const TensorData& self, const TensorData& other)
{
    auto tout0 = From(out0);
    auto tout1 = From(out1);
    auto tself = From(self);
    auto tother = From(other);

    auto interleaved = InterleaveTensor(tself.second, tother.second);
    auto lastDim = tself.second.size(-1);
    ToOperand(interleaved.slice(-1, 0, lastDim), tout0.first, out0.dtype);
    ToOperand(interleaved.slice(-1, lastDim, interleaved.size(-1)), tout1.first, out1.dtype);
}

static void DeInterleaveTensor(const torch::Tensor& interleaved, const std::pair<torch::Tensor, torch::Tensor>& out0,
                               const std::pair<torch::Tensor, torch::Tensor>& out1, DataType out0Dtype,
                               DataType out1Dtype)
{
    auto lastDim = interleaved.size(-1);
    ToOperand(interleaved.slice(-1, 0, lastDim, 0x2), out0.first, out0Dtype);
    ToOperand(interleaved.slice(-1, 1, lastDim, 0x2), out1.first, out1Dtype);
}

static void DeInterleave(const TensorData& out0, const TensorData& out1, const TensorData& self,
                         const TensorData& other)
{
    auto tout0 = From(out0);
    auto tout1 = From(out1);
    auto tself = From(self);
    auto tother = From(other);

    std::vector<torch::Tensor> inputs = {tself.second, tother.second};
    DeInterleaveTensor(torch::cat(inputs, -1), tout0, tout1, out0.dtype, out1.dtype);
}

static void DeInterleaveSingle(const TensorData& out0, const TensorData& out1, const TensorData& self)
{
    auto tout0 = From(out0);
    auto tout1 = From(out1);
    auto tself = From(self);

    DeInterleaveTensor(tself.second, tout0, tout1, out0.dtype, out1.dtype);
}

static void RowSumExpand(const TensorData& out, const TensorData& self, int dim)
{
    auto tout = From(out);
    auto tself = From(self);
    auto res = torch::sum(tself.second, {dim}, true);
    ToOperand(res, tout.first, out.dtype);
}

static void RowSumSingle(const TensorData& out, const TensorData& self, int dim)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::sum_out(tout.second, tself.second, {dim}, true);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void RowMinExpand(const TensorData& out, const TensorData& self, int dim)
{
    auto tself = From(self);
    auto ret = torch::min(tself.second, dim, true);
    auto tout = From(out);
    ToOperand(std::get<0>(ret), tout.first, out.dtype);
}

static void RowMaxExpand(const TensorData& out, const TensorData& self, int dim)
{
    auto tself = From(self);
    auto ret = torch::max(tself.second, dim, true);
    auto tout = From(out);
    ToOperand(std::get<0>(ret), tout.first, out.dtype);
}

static void RowMinSingle(const TensorData& out, const TensorData& self, int dim)
{
    auto tself = From(self);
    auto ret = torch::min(tself.second, dim, true);
    auto tout = From(out);
    ToOperand(std::get<0>(ret), tout.first, out.dtype);
}

static void RowMinLine(const TensorData& out, const TensorData& self, int dim)
{
    auto tself = From(self);
    auto tout = From(out);
    auto ret = torch::min(tself.second, dim, true);
    ToOperand(std::get<0>(ret), tout.first, out.dtype);
}

static void RowMaxSingle(const TensorData& out, const TensorData& self, int dim)
{
    auto tself = From(self);
    auto ret = torch::max(tself.second, dim, true);
    auto tout = From(out);
    ToOperand(std::get<0>(ret), tout.first, out.dtype);
}

static void RowMaxLine(const TensorData& out, const TensorData& self, int dim)
{
    auto tself = From(self);
    auto tout = From(out);
    auto ret = torch::max(tself.second, dim, true);
    ToOperand(std::get<0>(ret), tout.first, out.dtype);
}

static void RowProdSingle(const TensorData& out, const TensorData& self, int dim)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::prod_out(tout.second, tself.second, {dim}, true);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void RowProdLine(const TensorData& out, const TensorData& self, int dim)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::prod_out(tout.second, tself.second, {dim}, true);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void RowArgMaxSingle(const TensorData& out, const TensorData& self, int dim)
{
    auto tout = From(out);
    auto tself = From(self);
    auto ret = torch::max(tself.second, dim, true);
    auto idxResult = std::get<1>(ret).to(torch::kInt32);
    ToOperand(idxResult, tout.first, out.dtype);
}

static void RowArgMinSingle(const TensorData& out, const TensorData& self, int dim)
{
    auto tout = From(out);
    auto tself = From(self);
    auto ret = torch::min(tself.second, dim, true);
    auto idxResult = std::get<1>(ret).to(torch::kInt32);
    ToOperand(idxResult, tout.first, out.dtype);
}

static void RowArgMaxWithValueSingle(const TensorData& outValue, const TensorData& outIndex, const TensorData& outTemp,
                                     const TensorData& self, int dim)
{
    auto toutValue = From(outValue);
    auto toutIndex = From(outIndex);
    auto tself = From(self);
    (void)outTemp;
    auto ret = torch::max(tself.second, dim, true);
    ToOperand(std::get<0>(ret), toutValue.first, outValue.dtype);
    auto idxResult = std::get<1>(ret).to(torch::kInt32);
    ToOperand(idxResult, toutIndex.first, outIndex.dtype);
}

static void RowArgMinWithValueSingle(const TensorData& outValue, const TensorData& outIndex, const TensorData& outTemp,
                                     const TensorData& self, int dim)
{
    auto toutValue = From(outValue);
    auto toutIndex = From(outIndex);
    auto tself = From(self);
    (void)outTemp;
    auto ret = torch::min(tself.second, dim, true);
    ToOperand(std::get<0>(ret), toutValue.first, outValue.dtype);
    auto idxResult = std::get<1>(ret).to(torch::kInt32);
    ToOperand(idxResult, toutIndex.first, outIndex.dtype);
}

static void RowArgMaxWithValueLine(const TensorData& outValue, const TensorData& outIndex, const TensorData& outTemp,
                                   const TensorData& self, int dim)
{
    RowArgMaxWithValueSingle(outValue, outIndex, outTemp, self, dim);
}

static void RowArgMinWithValueLine(const TensorData& outValue, const TensorData& outIndex, const TensorData& outTemp,
                                   const TensorData& self, int dim)
{
    RowArgMinWithValueSingle(outValue, outIndex, outTemp, self, dim);
}

static void PairArgMax(const TensorData& outValue, const TensorData& outIndex, const TensorData& value1,
                       const TensorData& index1, const TensorData& value2, const TensorData& index2)
{
    auto toutValue = From(outValue);
    auto toutIndex = From(outIndex);
    auto tvalue1 = From(value1);
    auto tindex1 = From(index1);
    auto tvalue2 = From(value2);
    auto tindex2 = From(index2);
    auto cmpResult = tvalue1.second >= tvalue2.second;
    auto selectdValue = torch::where(cmpResult, tvalue1.second, tvalue2.second);
    auto selectdIndex = torch::where(cmpResult, tindex1.second.to(torch::kInt32), tindex2.second.to(torch::kInt32));
    ToOperand(selectdValue, toutValue.first, outValue.dtype);
    ToOperand(selectdIndex, toutIndex.first, outIndex.dtype);
}

static void PairArgMin(const TensorData& outValue, const TensorData& outIndex, const TensorData& value1,
                       const TensorData& index1, const TensorData& value2, const TensorData& index2)
{
    auto toutValue = From(outValue);
    auto toutIndex = From(outIndex);
    auto tvalue1 = From(value1);
    auto tindex1 = From(index1);
    auto tvalue2 = From(value2);
    auto tindex2 = From(index2);
    auto cmpResult = tvalue1.second <= tvalue2.second;
    auto selectdValue = torch::where(cmpResult, tvalue1.second, tvalue2.second);
    auto selectdIndex = torch::where(cmpResult, tindex1.second.to(torch::kInt32), tindex2.second.to(torch::kInt32));
    ToOperand(selectdValue, toutValue.first, outValue.dtype);
    ToOperand(selectdIndex, toutIndex.first, outIndex.dtype);
}

static void Reshape(const TensorData& out, const TensorData& self)
{
    auto tout = From(out);
    auto tself = From(self);

    torch::Tensor res;
    if (out.dataPtr == self.dataPtr) {
        res = torch::reshape(tself.second.clone(), out.shape);
    } else {
        res = torch::reshape(tself.second, out.shape);
    }

    ToOperand(res, tout.first, out.dtype);
}

static void Transpose(const TensorData& out, const TensorData& self, int64_t dim0, int64_t dim1)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::transpose_copy_out(tout.second, tself.second, dim0, dim1);
    ToOperand(tout.second, tout.first, out.dtype);
}

void Permute(const TensorData& out, const TensorData& self, const std::vector<int64_t>& dim)
{
    auto tout = From(out);
    auto tself = From(self);
    torch::permute_copy_out(tout.second, tself.second, dim);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void ReduceAcc(const TensorData& out, const std::vector<TensorData>& tdatas)
{
    auto tout = From(out);
    std::vector<torch::Tensor> tensors;
    for (auto& tdata : tdatas) {
        tensors.push_back(From(tdata).second);
    }
    torch::sum_out(tout.second, torch::stack(tensors, 0), 0);
    ToOperand(tout.second, tout.first, out.dtype);
}

// Reorder topk outputs so that equal values keep ascending index order
// (smaller index first). Uses secondary-then-primary stable multi-key sort.
static void StabilizeTopKTieIndices(torch::Tensor& values, torch::Tensor& indices, int64_t axis, bool descending)
{
    torch::Tensor orderByIdx;
    std::tie(std::ignore, orderByIdx) = indices.sort(/*stable=*/true, axis, /*descending=*/false);
    values = values.gather(axis, orderByIdx);
    indices = indices.gather(axis, orderByIdx);

    torch::Tensor orderByVal;
    std::tie(std::ignore, orderByVal) = values.sort(/*stable=*/true, axis, descending);
    values = values.gather(axis, orderByVal);
    indices = indices.gather(axis, orderByVal);
}

// grouped last dim is [value, index]; sort by value with ascending-index tie-break
static torch::Tensor StableSortValueIndexPairs(torch::Tensor grouped, int64_t sortAxis, bool valueDescending)
{
    constexpr int kPairSize = 2;
    torch::Tensor orderByIdx;
    std::tie(std::ignore, orderByIdx) = grouped.select(-1, 1).sort(/*stable=*/true, sortAxis, /*descending=*/false);
    auto indexShape = orderByIdx.sizes().vec();
    indexShape.push_back(kPairSize);
    auto expandIdx = orderByIdx.unsqueeze(-1).expand(torch::IntArrayRef(indexShape));
    grouped = grouped.gather(sortAxis, expandIdx);

    torch::Tensor orderByVal;
    std::tie(std::ignore, orderByVal) = grouped.select(-1, 0).sort(/*stable=*/true, sortAxis, valueDescending);
    expandIdx = orderByVal.unsqueeze(-1).expand(torch::IntArrayRef(indexShape));
    return grouped.gather(sortAxis, expandIdx);
}

/**
 * @brief Perform a bitwise sort of 32 elements on the input tensor according to the specified dimension
 *        and return the output tensor
 *        e.g.,1.If the shape of the input tensor is {2,33}, a temporary tensor will be created based on the
 *             input pad to {2,64}, and an index tensor with a shape of {2,64} and values from 0 to 63 will
 *             be created along the sorting axis;
 *             2. Then stack the temporary tensor and index tensor into a new tensor with a shape of {2,64,2}
 *             3. Then, along the original sorting axis, the new tensor is transformed into a temporary tensor
 *             with 32 elements per group to {2, 32, 2, 2}. Next, the temporary tensor is sorted within the group
 *             along the axis with a size of 32 to make it ordered within the group. Finally, the sorted tensor
 *             is expanded into a tensor with a size of {2128} using reshape. Finally, the data distribution on
 *             the one-dimensional sorting axis is arranged alternately in order of value index
 *
 * @param out output tensor
 * @param self input tensor
 * @param axis Indicate on which axis of self for grouping and sorting
 * @param descending Indicate whether the sorting direction is ascending or descending
 */
static void BitSort(const TensorData& out, const TensorData& self, int64_t axis, bool descending, int64_t offset)
{
    auto tself = From(self);
    auto tout = From(out);
    constexpr int DIM_SIZE_TWO = 2;
    axis = axis < 0 ? (axis + tself.second.dim()) : axis;

    std::vector<int64_t> viewOffset(tself.second.dim(), 0);
    const int64_t groupSize = 32;
    auto tselfAlignShape = tself.second.sizes().vec();
    tselfAlignShape[axis] = (tself.second.size(axis) + groupSize - 1) / groupSize * groupSize;
    float padValue = descending ? (-1.0f / 0.0f) : (1.0f / 0.0f);
    auto tselfAlign = torch::full(tselfAlignShape, padValue);
    torch::Tensor tselfAlignSubview = View(tselfAlign, tself.second.sizes().vec(), viewOffset);
    tselfAlignSubview.copy_(tself.second);
    if (!descending) {
        tselfAlign.neg_();
    }

    auto indices = torch::arange(0, tselfAlign.size(axis), 1, torch::dtype(torch::kLong)) + offset;
    std::vector<int64_t> indexShape(tselfAlign.dim(), 1);
    indexShape[axis] = tselfAlign.size(axis);
    indices = indices.reshape(indexShape).broadcast_to(tselfAlign.sizes());

    auto combined = torch::stack({tselfAlign, indices.to(tselfAlign.dtype())}, tselfAlign.dim());
    std::vector<int64_t> groupedShape;
    for (int64_t i = 0; i < tselfAlign.dim(); ++i) {
        if (i == axis) {
            groupedShape.push_back(tselfAlign.size(axis) / groupSize);
            groupedShape.push_back(groupSize);
        } else {
            groupedShape.push_back(tselfAlign.size(i));
        }
    }
    groupedShape.push_back(DIM_SIZE_TWO);
    auto grouped = combined.reshape(torch::IntArrayRef(groupedShape));
    auto sortedGroups = StableSortValueIndexPairs(grouped, axis + 1, /*valueDescending=*/true);

    std::vector<int64_t> dstShape;
    for (int64_t i = 0; i < sortedGroups.dim(); ++i) {
        if (i == axis) {
            dstShape.push_back(DIM_SIZE_TWO * tselfAlign.size(axis));
        } else if (i != axis + 1 && i != sortedGroups.dim() - 1) {
            dstShape.push_back(sortedGroups.size(i));
        }
    }

    auto tres = sortedGroups.reshape(torch::IntArrayRef(dstShape));
    torch::Tensor dstSubview = View(tout.second, tres.sizes().vec(), viewOffset);
    dstSubview.copy_(tres);
    ToOperand(tout.second, tout.first, out.dtype);
}

/**
 * @brief extract elements from the target dimension of tensors and ajust the output according to the param
 *        require the data distribution if the input tensor sorting axis to be value indexed alternately
 *        arranged in order
 *
 * @param out output tensor
 * @param self input tensor
 * @param mod used to extract elements from the target dimension of tensors, mod=0 means to obtain
 *            elements with even indices, and mod=1 means to obtain elements with odd indices
 * @param descending Indicate whether the obtained k values are the maximum or minimum k values,
 *                   and true returns the maximum k values
 */
static void Extract(const TensorData& out, const TensorData& self, int mod, bool descending)
{
    auto tself = From(self);
    auto tout = From(out);
    constexpr int INDICE_STEP = 2;

    std::vector<int64_t> viewOffset(tself.second.dim(), 0);
    int dim = tself.second.dim() - 1;
    if (tself.second.size(dim) == 0) {
        return;
    }
    auto indices = torch::arange((mod == 1 ? 1 : 0), tself.second.size(dim), INDICE_STEP, torch::dtype(torch::kLong));
    torch::Tensor selfSubview = View(tself.second.index_select(dim, indices), tout.second.sizes().vec(), viewOffset);
    tout.second.copy_(selfSubview);

    if (!descending && mod == 0) {
        tout.second.neg_();
    }
    ToOperand(tout.second, tout.first, out.dtype);
}

/**
 * @brief Sort the input tensor according to the specified dimension and return the output tensor, requiring the
 *        data distribution of the sorting axis of the input tensor to be alternately arranged by value index
 *        e.g.,1.If the shape of the input tensor is {2,256}, then half of the sorting axis in the input tensor
 *             will be truncated as a valid tensor with a shape of {2,128}
 *             2. Then group the values and indexes along the sorting axis, dividing them into 64 value index
 *             pairs, and reshape the effective tensor to a new tensor with a shape of {2,64,2}
 *             3. Sort the new tensor along the original sorting axis in the numerical dimension, and finally
 *             use reshape expansion to sort the new tensor into output tensors of shape and size {2,128}.
 *             Finally, the data distribution of the entire tensor on the one-dimensional sorting axis is still
 *             sorted alternately by value index, and the values are ordered
 *
 * @param out output tensor
 * @param self input tensor
 * @param axis Indicate on which axis of self to obtain topk
 * @param k  Indicate the  maximum or minimum k values are obtained
 * @param descending Indicate whether the obtained k values are the maximum or minimum k values,
 *                   and true returns the maximum k values
 */
static torch::Tensor SortGroupedValueIndexPairs(const torch::Tensor& packed, int64_t axis)
{
    constexpr int kPairSize = 2;
    std::vector<int64_t> newShape;
    newShape.reserve(packed.dim() + 1);
    for (int64_t i = 0; i < packed.dim(); ++i) {
        if (i == axis) {
            newShape.push_back(packed.size(axis) / kPairSize);
            newShape.push_back(kPairSize);
        } else {
            newShape.push_back(packed.size(i));
        }
    }
    auto grouped = packed.reshape(torch::IntArrayRef(newShape));
    return StableSortValueIndexPairs(grouped, axis, /*valueDescending=*/true);
}

static torch::Tensor PadSelectTopKGroups(torch::Tensor sortedGroups, int64_t axis, int64_t k)
{
    int64_t sortedSize = sortedGroups.size(axis);
    if (sortedSize < k) {
        auto padShape = sortedGroups.sizes().vec();
        padShape[axis] = k - sortedSize;
        auto padTensor = torch::full(padShape, -std::numeric_limits<float>::infinity(), sortedGroups.dtype());
        sortedGroups = torch::cat({sortedGroups, padTensor}, axis);
    }
    return sortedGroups.index_select(axis, torch::arange(k, torch::dtype(torch::kLong)));
}

static std::vector<int64_t> FlattenPairAxisShape(const torch::Tensor& topkGroups, int64_t axis, int64_t k)
{
    constexpr int kPairSize = 2;
    std::vector<int64_t> dstShape;
    dstShape.reserve(topkGroups.dim() - 1);
    for (int64_t i = 0; i < topkGroups.dim(); ++i) {
        if (i == axis) {
            dstShape.push_back(kPairSize * k);
        } else if (i != axis + 1) {
            dstShape.push_back(topkGroups.size(i));
        }
    }
    return dstShape;
}

static void MrgSort(const TensorData& out, const TensorData& self, int64_t axis, int64_t k)
{
    auto tself = From(self);
    auto tout = From(out);
    axis = axis < 0 ? (axis + tself.second.dim()) : axis;
    int actShape = tself.second.size(axis);
    if (actShape == 0) {
        tout.second.fill_(-std::numeric_limits<float>::infinity());
        ToOperand(tout.second, tout.first, out.dtype);
        return;
    }
    auto tselfHalf = tself.second.index_select(axis, torch::arange(actShape, torch::dtype(torch::kLong)));
    ASSERT(CalculatorErrorScene::MRGSORT_AXIS_OUT_OF_RANGE, axis >= 0 && axis < tselfHalf.dim())
        << "axis" << axis << " is out of bounds for tensor of dimension " << tselfHalf.dim();

    auto topkGroups = PadSelectTopKGroups(SortGroupedValueIndexPairs(tselfHalf, axis), axis, k);
    auto dstShape = FlattenPairAxisShape(topkGroups, axis, k);
    View(tout.second, dstShape, std::vector<int64_t>(tself.second.dim(), 0))
        .copy_(topkGroups.reshape(torch::IntArrayRef(dstShape)));
    ToOperand(tout.second, tout.first, out.dtype);
}

static void TiledMrgSort(const TensorData& out, const TensorData& src1, const TensorData& src2, const TensorData& src3,
                         const TensorData& src4, int validBit, int kvalue)
{
    auto self1 = From(src1);
    auto self2 = From(src2);
    auto self3 = From(src3);
    auto self4 = From(src4);
    auto tout = From(out);
    constexpr int SORT_NUM_TWO = 2;
    constexpr int SORT_NUM_THREE = 3;
    constexpr int SORT_NUM_FOUR = 4;
    torch::Tensor tself;
    if (validBit == SORT_NUM_TWO) {
        tself = torch::cat({self1.second, self2.second}, -1);
    } else if (validBit == SORT_NUM_THREE) {
        tself = torch::cat({self1.second, self2.second, self3.second}, -1);
    } else if (validBit == SORT_NUM_FOUR) {
        tself = torch::cat({self1.second, self2.second, self3.second, self4.second}, -1);
    }
    auto axis = tself.dim() - 1;
    if (tself.size(axis) == 0) {
        tout.second.fill_(-std::numeric_limits<float>::infinity());
        ToOperand(tout.second, tout.first, out.dtype);
        return;
    }

    auto topkGroups = PadSelectTopKGroups(SortGroupedValueIndexPairs(tself, axis), axis, kvalue);
    auto dstShape = FlattenPairAxisShape(topkGroups, axis, kvalue);
    View(tout.second, dstShape, {0, 0}).copy_(topkGroups.reshape(torch::IntArrayRef(dstShape)));
    ToOperand(tout.second, tout.first, out.dtype);
}

static void TopK(const TensorData& outValue, const TensorData& outIndex, const TensorData& self, int k, int axis,
                 bool descending)
{
    auto tself = From(self);
    auto toutValue = From(outValue);
    auto toutIndex = From(outIndex);
    axis = axis < 0 ? (axis + tself.second.dim()) : axis;
    torch::Tensor tempIdxInt64 = torch::zeros(toutValue.second.sizes().vec(), torch::kInt64);
    torch::topk_out(toutValue.second, tempIdxInt64, tself.second, k, axis, descending);
    torch::Tensor values = toutValue.second;
    StabilizeTopKTieIndices(values, tempIdxInt64, axis, descending);
    toutValue.second.copy_(values);
    toutIndex.second.copy_(tempIdxInt64.to(torch::kInt32));
    ToOperand(toutValue.second, toutValue.first, outValue.dtype);
    ToOperand(toutIndex.second, toutIndex.first, outIndex.dtype);
}

static void TopkSort(const TensorData& outValue, const TensorData& outTemp, const TensorData& self, int startIndex)
{
    auto tself = From(self);
    auto toutValue = From(outValue);
    auto toutTemp = From(outTemp);

    constexpr int GROUP_SIZE = 32;
    int axis = tself.second.dim() - 1;

    // 1. Generate indices starting from startIndex*len
    int64_t len = tself.second.size(axis);
    int64_t baseIdx = startIndex * len;
    auto indices = torch::arange(baseIdx, baseIdx + len, 1, torch::dtype(torch::kFloat));
    std::vector<int64_t> indexShape(tself.second.dim(), 1);
    indexShape[axis] = len;
    indices = indices.reshape(indexShape).broadcast_to(tself.second.sizes());

    // 2. Align to GROUP_SIZE (32)
    auto tselfAlignShape = tself.second.sizes().vec();
    int64_t alignedLen = (len + GROUP_SIZE - 1) / GROUP_SIZE * GROUP_SIZE;
    tselfAlignShape[axis] = alignedLen;

    float padValue = -1.0f / 0.0f; // Negative infinity for descending sort
    auto valuesAlign = torch::full(tselfAlignShape, padValue, tself.second.dtype());
    torch::Tensor valueView = View(valuesAlign, tself.second.sizes().vec(), {0, 0});
    valueView.copy_(tself.second);

    auto indicesAlign = torch::full(tselfAlignShape, padValue, torch::kFloat);
    torch::Tensor indexView = View(indicesAlign, indices.sizes().vec(), {0, 0});
    indexView.copy_(indices);

    // 3. Group and sort (every 32 elements)
    std::vector<int64_t> groupShape;
    for (int64_t i = 0; i < valuesAlign.dim(); ++i) {
        if (i == axis) {
            groupShape.push_back(alignedLen / GROUP_SIZE);
            groupShape.push_back(GROUP_SIZE);
        } else {
            groupShape.push_back(valuesAlign.size(i));
        }
    }

    auto valsGrouped = valuesAlign.reshape(torch::IntArrayRef(groupShape));
    auto idxsGrouped = indicesAlign.reshape(torch::IntArrayRef(groupShape));

    StabilizeTopKTieIndices(valsGrouped, idxsGrouped, axis + 1, /*descending=*/true);

    // 4. Flatten
    valsGrouped = valsGrouped.flatten(axis, axis + 1);
    idxsGrouped = idxsGrouped.flatten(axis, axis + 1);

    // 5. Create pack: [v0, i0, v1, i1, ...]
    auto stacked = torch::stack({valsGrouped, idxsGrouped}, -1); // [..., len, 2]
    auto packed = stacked.flatten(axis, -1);                     // [..., len*2]

    // 6. Output
    torch::Tensor tempView = View(toutTemp.second, packed.sizes().vec(), {0, 0});
    tempView.copy_(packed);
    torch::Tensor valView = View(toutValue.second, packed.sizes().vec(), {0, 0});
    valView.copy_(packed);
    ToOperand(toutTemp.second, toutTemp.first, outTemp.dtype);
    ToOperand(toutValue.second, toutValue.first, outValue.dtype);
}

static void TopkMerge(const TensorData& out, const TensorData& self, int mergeSize)
{
    (void)mergeSize;
    auto tself = From(self);
    auto tout = From(out);

    int axis = tself.second.dim() - 1;

    // Input is pack format: [v0, i0, v1, i1, ...]
    // mergeSize: number of already-sorted packs
    // Note: Current implementation uses global sort for simplicity (sufficient for precision verification)
    (void)mergeSize; // Suppress unused parameter warning

    // Extract all values (even positions) and indices (odd positions)
    auto evenIndices = torch::arange(0, tself.second.size(axis), 2, torch::dtype(torch::kLong));
    auto oddIndices = torch::arange(1, tself.second.size(axis), 2, torch::dtype(torch::kLong));
    auto values = tself.second.index_select(axis, evenIndices);
    auto indices = tself.second.index_select(axis, oddIndices);

    // Equal values: smaller index first, then value descending
    StabilizeTopKTieIndices(values, indices, axis, /*descending=*/true);

    // Rebuild pack: [v0, i0, v1, i1, ...]
    auto stacked = torch::stack({values, indices}, -1);
    auto sorted = stacked.flatten(axis, -1);

    torch::Tensor outView = View(tout.second, sorted.sizes().vec(), {0, 0});
    outView.copy_(sorted);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void TopkExtract(const TensorData& out, const TensorData& self, int k, bool isIndex)
{
    auto tself = From(self);
    auto tout = From(out);

    int axis = tself.second.dim() - 1;

    // Input is pack format: [v0, i0, v1, i1, ...]
    // isIndex=false: extract first k values (even positions: 0, 2, 4, ...)
    // isIndex=true:  extract first k indices (odd positions: 1, 3, 5, ...)

    int startOffset = isIndex ? 1 : 0; // index starts from 1, value from 0
    int stride = 2;                    // Values and indices are interleaved in pack

    // Generate extraction indices: startOffset, startOffset+2, startOffset+4, ..., startOffset+2*(k-1)
    auto indices = torch::arange(startOffset, startOffset + k * stride, stride, torch::dtype(torch::kLong));

    // Extract
    auto extracted = tself.second.index_select(axis, indices);

    // If extracting indices, convert to INT32
    if (isIndex) {
        extracted = extracted.to(torch::kInt);
    }

    // Reshape to [1, k] (according to output shape in operation_impl.cpp)
    extracted = extracted.reshape({1, k});

    torch::Tensor outView = View(tout.second, extracted.sizes().vec(), {0, 0});
    outView.copy_(extracted);
    ToOperand(tout.second, tout.first, out.dtype);
}

void TwoTileMrgSort(const TensorData& out, const TensorData& self)
{
    auto tself = From(self);
    auto tout = From(out);
    constexpr int SIZE_TWO = 2;
    int axis = tself.second.dim() - 1;

    std::vector<int64_t> viewOffset(tself.second.dim(), 0);
    std::vector<int64_t> newShape;
    newShape.reserve(tself.second.dim() + 1);
    for (int64_t i = 0; i < tself.second.dim(); i++) {
        if (i == axis) {
            newShape.push_back(tself.second.size(axis) / SIZE_TWO);
            newShape.push_back(SIZE_TWO);
        } else {
            newShape.push_back(tself.second.size(i));
        }
    }

    auto tselfGrouped = tself.second.reshape(torch::IntArrayRef(newShape));
    torch::Tensor sortedIndices;
    std::tie(std::ignore, sortedIndices) = tselfGrouped.select(-1, 0).sort(axis, true);

    std::vector<int64_t> indexShape;
    for (int64_t i = 0; i < sortedIndices.dim(); i++) {
        indexShape.push_back(sortedIndices.size(i));
    }
    indexShape.push_back(SIZE_TWO);

    auto expanded_indices = sortedIndices.unsqueeze(-1).expand(torch::IntArrayRef(indexShape));
    auto sortedGroups = tselfGrouped.gather(axis, expanded_indices);

    std::vector<int64_t> dstShape;
    for (int64_t i = 0; i < tself.second.dim(); i++) {
        dstShape.push_back(tself.second.size(i));
    }
    torch::Tensor dstSubview = View(tout.second, dstShape, viewOffset);
    dstSubview.copy_(sortedGroups.reshape(torch::IntArrayRef(dstShape)));
    ToOperand(tout.second, tout.first, out.dtype);
}

void Sort(const TensorData& value, const TensorData& index, const TensorData& self, int64_t axis, bool descending)
{
    auto tself = From(self);
    auto tvalue = From(value);
    auto tindex = From(index);
    auto [sortValue, sortIndex] = tself.second.sort(axis, descending);
    std::vector<int64_t> viewOffset(tself.second.dim(), 0);
    std::vector<int64_t> dstShape;
    for (int64_t i = 0; i < tvalue.second.dim(); i++) {
        dstShape.push_back(tvalue.second.size(i));
    }
    torch::Tensor outValue = View(tvalue.second, dstShape, viewOffset);
    torch::Tensor outIndex = View(tindex.second, dstShape, viewOffset);
    outValue.copy_(sortValue);
    outIndex.copy_(sortIndex);
    ToOperand(tvalue.second, tvalue.first, value.dtype);
    ToOperand(tindex.second, tindex.first, index.dtype);
}

bool ScatterDateCopy(const std::vector<int64_t>& loopIdx, torch::Tensor& src, torch::Tensor& indices,
                     torch::Tensor& ret, int blockSize)
{
    bool flag = false;
    int64_t s = indices.size(1);
    int64_t i = loopIdx[0];
    int64_t j = loopIdx[1];
    int64_t dataIdx = indices.index({i, j}).item<int64_t>();

    ASSERT(CalculatorErrorScene::SCATTER_BLOCKSIZE_ZERO, blockSize != 0);
    if (ret.dim() == 2) { // 2 dim
        int64_t srcIdx = i * s + j;
        if ((dataIdx < 0 || dataIdx >= ret.size(0)) || (srcIdx < 0 || srcIdx >= src.size(0))) {
            return flag;
        }
        ret[dataIdx] = src[srcIdx];
        flag = true;
    } else if (ret.dim() == 4) { // 4 dim
        int64_t bIdx = dataIdx / blockSize;
        int64_t sIdx = dataIdx % blockSize;
        if ((bIdx < 0 || bIdx >= ret.size(0)) || (sIdx < 0 || sIdx >= ret.size(1))) {
            return flag;
        }
        ret[bIdx][sIdx] = src[i][j];
        flag = true;
    }

    return flag;
}

static void ScatterUpdate(const TensorData& out, const TensorData& self, const TensorData& index, const TensorData& dst,
                          int axis, std::string cacheMode, int blockSize)
{
    (void)axis;
    (void)cacheMode;

    auto inplace = From(dst);
    auto ret = From(out);
    ret.second.copy_(inplace.second);
    auto src = From(self);
    auto indices = From(index);

    ASSERT(CalculatorErrorScene::SCATTER_INDICES_DIM_INVALID,
           indices.second.dim() == 2); // indices should be 2 dim
    ASSERT(CalculatorErrorScene::SCATTER_SRC_RET_DIM_UNSUPPORTED,
           (src.second.dim() == 2) || (src.second.dim() == 4)); // only 2, 4 dim support
    ASSERT(CalculatorErrorScene::SCATTER_SRC_RET_DIM_UNSUPPORTED,
           (ret.second.dim() == 2) || (ret.second.dim() == 4)); // only 2, 4 dim support
    ASSERT(CalculatorErrorScene::SCATTER_SRC_RET_DIM_MISMATCH, src.second.dim() == ret.second.dim());

    int64_t b = indices.second.size(0);
    int64_t s = indices.second.size(1);
    for (int64_t i = 0; i < b; i++) {
        for (int64_t j = 0; j < s; j++) {
            if (ScatterDateCopy({i, j}, src.second, indices.second, ret.second, blockSize) == false) {
                return;
            }
        }
    }
    ToOperand(ret.second, ret.first, out.dtype);
}

static const std::vector<std::string> scatterModeString = {"add", "multiply"};

static void ScatterElement(const TensorData& out, const TensorData& self, const TensorData& index, const Element& src,
                           int axis, int reduce)
{
    auto output = From(out);
    auto inputSelf = From(self);
    auto inputIndices = From(index);

    if (index.dtype == DT_INT32) {
        inputIndices.second = inputIndices.second.to(torch::kInt64);
    }
    if (reduce == 0) {
        auto res = torch::scatter(inputSelf.second, axis, inputIndices.second, From(src));
        ToOperand(res, output.first, out.dtype);
    } else {
        auto res = torch::scatter(inputSelf.second, axis, inputIndices.second, From(src),
                                  scatterModeString.at(reduce - 1));
        ToOperand(res, output.first, out.dtype);
    }
}

static void Brcb(const TensorData& out, const TensorData& self)
{
    auto tself = From(self);
    auto tout = From(out);

    // BRCB broadcasts the first element of the last dim across the output last dim.
    // Leading dims are preserved, so 2D ([M,1]->[M,N]) and 3D ([B,M,1]->[B,M,N]) share the same path.
    auto firstAlongLast = tself.second.select(/*dim=*/-1, /*index=*/0);
    auto expanded = firstAlongLast.unsqueeze(-1).expand(tout.second.sizes());
    tout.second.copy_(expanded);
    ToOperand(tout.second, tout.first, out.dtype);
}

static void Scatter(const TensorData& out, const TensorData& self, const TensorData& index, const TensorData& src,
                    int axis, int reduce)
{
    auto output = From(out);
    auto inputSelf = From(self);
    auto inputIndices = From(index);
    auto inputSrc = From(src);

    if (index.dtype == DT_INT32) {
        inputIndices.second = inputIndices.second.to(torch::kInt64);
    }
    if (reduce == 0) {
        output.second = torch::scatter(inputSelf.second, axis, inputIndices.second, inputSrc.second);
        ToOperand(output.second, output.first, out.dtype);
    } else {
        output.second = torch::scatter(inputSelf.second, axis, inputIndices.second, inputSrc.second,
                                       scatterModeString.at(reduce - 1));
        ToOperand(output.second, output.first, out.dtype);
    }
}

struct QuantMXContext {
    DataType srcDtype;
    DataType quantDtype;
    bool performanceMode;
    bool isFp4E2M1;
    bool isNv;
    bool usePlainFp8MaxAbs;
    int64_t scalingFactor;
    const MXQuantDtypeParams* dtypeParams;
};

static QuantMXContext MakeQuantMXContext(DataType srcDtype, DataType quantDtype, bool performanceMode, int64_t mode)
{
    const bool isFp4E2M1 = quantDtype == DT_FP4_E2M1X2;
    const bool isNv = mode == kMxQuantModeRoundUp;
    return {
        .srcDtype = srcDtype,
        .quantDtype = quantDtype,
        .performanceMode = performanceMode,
        .isFp4E2M1 = isFp4E2M1,
        .isNv = isNv,
        .usePlainFp8MaxAbs = !isFp4E2M1 && !isNv,
        .scalingFactor = srcDtype == DT_FP32 ? 2 : 1,
        .dtypeParams = isFp4E2M1 ? &kFP4E2M1Params : &kFP8E4M3Params,
    };
}

static float NormalizeQuantMXMaxValue(float srcValue, const QuantMXContext& ctx)
{
    const float absValue = std::fabs(srcValue);
    if (ctx.usePlainFp8MaxAbs) {
        return absValue;
    }
    if (ctx.isNv && ctx.srcDtype == DT_FP16) {
        return RoundToFp16(absValue);
    }
    if (ctx.isNv && ctx.srcDtype == DT_BF16) {
        return RoundToBf16(absValue);
    }
    if (ctx.isFp4E2M1 && ctx.srcDtype == DT_FP16) {
        return std::fabs(TruncateToBf16(srcValue));
    }
    return ctx.isFp4E2M1 ? RoundToBf16(absValue) : absValue;
}

static float ComputeQuantMXGroupMax(const float* inputPtr, int64_t row, int64_t cols, int64_t group,
                                    const QuantMXContext& ctx)
{
    float maxAbsValue = 0.0f;
    bool hasNaN = false;
    for (int64_t inner = 0; inner < MX_QUANT_TILE_BLOCK; ++inner) {
        const int64_t col = group * MX_QUANT_TILE_BLOCK + inner;
        if (col >= cols) {
            continue;
        }
        const float val = NormalizeQuantMXMaxValue(inputPtr[row * cols + col], ctx);
        hasNaN = hasNaN || std::isnan(val);
        maxAbsValue = std::isnan(val) ? maxAbsValue : std::max(maxAbsValue, val);
    }
    return hasNaN ? std::numeric_limits<float>::quiet_NaN() : maxAbsValue;
}

static std::pair<uint8_t, float> ComputeQuantMXExponentAndScaling(float maxAbsValue, const QuantMXContext& ctx)
{
    if (ctx.isNv) {
        const uint8_t e8m0 = ComputeSharedExponentNV(maxAbsValue, ctx.dtypeParams->maxPos);
        return {e8m0, ComputeScalingFromExponentMath(e8m0)};
    }
    if (ctx.isFp4E2M1) {
        return ComputeB16OcpExponentAndScaling(maxAbsValue, true);
    }
    const uint8_t e8m0 = ComputeSharedExponent(maxAbsValue, ctx.dtypeParams->targetMaxPow2);
    return {e8m0, ComputeScalingFromExponent(e8m0)};
}

static void StoreQuantMXScaling(float* scalingPtr, int64_t row, int64_t groupCols, int64_t group, float scaling,
                                const QuantMXContext& ctx)
{
    const int64_t offset = row * groupCols * ctx.scalingFactor + group * ctx.scalingFactor;
    scalingPtr[offset] = scaling;
    if (ctx.scalingFactor == 0x2) {
        scalingPtr[offset + 1] = scaling;
    }
}

static float ScaleQuantMXValue(float srcValue, float groupScaling, const QuantMXContext& ctx)
{
    if (ctx.isFp4E2M1 && ctx.srcDtype == DT_BF16) {
        return RoundToBf16(srcValue * groupScaling);
    }
    if (ctx.isNv && ctx.srcDtype == DT_FP16) {
        return srcValue * RoundToBf16(groupScaling);
    }
    if (ctx.isNv && ctx.srcDtype == DT_BF16) {
        return RoundToBf16(srcValue * RoundToBf16(groupScaling));
    }
    return srcValue * groupScaling;
}

static void StoreQuantMXValue(uint8_t* quantPtr, int64_t row, int64_t cols, int64_t quantCols, int64_t col,
                              float srcValue, float groupScaling, const QuantMXContext& ctx)
{
    const float scaled = ScaleQuantMXValue(srcValue, groupScaling, ctx);
    if (ctx.isFp4E2M1) {
        const uint8_t encoded = EncodeE2M1Magic(scaled);
        uint8_t& packed = quantPtr[row * quantCols + col / 2];
        packed = (col & 1) == 0 ? static_cast<uint8_t>((packed & 0xF0u) | encoded) :
                                  static_cast<uint8_t>((packed & 0x0Fu) | (encoded << 0x4));
        return;
    }
    quantPtr[row * cols + col] = EncodeE4M3Fn(scaled);
}

static void FillQuantMXGroup(const float* inputPtr, uint8_t* quantPtr, uint8_t* expPtr, float* scalingPtr,
                             float* maxPtr, int64_t row, int64_t group, int64_t cols, int64_t expCols,
                             int64_t groupCols, int64_t quantCols, const QuantMXContext& ctx)
{
    const float maxAbsValue = ComputeQuantMXGroupMax(inputPtr, row, cols, group, ctx);
    const auto exponentAndScaling = ComputeQuantMXExponentAndScaling(maxAbsValue, ctx);
    expPtr[row * expCols + group] = exponentAndScaling.first;
    maxPtr[row * groupCols + group] = maxAbsValue;
    StoreQuantMXScaling(scalingPtr, row, groupCols, group, exponentAndScaling.second, ctx);
    for (int64_t inner = 0; inner < MX_QUANT_TILE_BLOCK; ++inner) {
        const int64_t col = group * MX_QUANT_TILE_BLOCK + inner;
        if (col < cols) {
            StoreQuantMXValue(quantPtr, row, cols, quantCols, col, inputPtr[row * cols + col],
                              exponentAndScaling.second, ctx);
        }
    }
}

static void FillQuantMXRows(const float* inputPtr, uint8_t* quantPtr, uint8_t* expPtr, float* scalingPtr, float* maxPtr,
                            int64_t rows, int64_t cols, int64_t expCols, int64_t groupCols, int64_t quantCols,
                            DataType srcDtype, DataType quantDtype, bool performanceMode, int64_t mode)
{
    const QuantMXContext ctx = MakeQuantMXContext(srcDtype, quantDtype, performanceMode, mode);
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t group = 0; group < groupCols; ++group) {
            FillQuantMXGroup(inputPtr, quantPtr, expPtr, scalingPtr, maxPtr, row, group, cols, expCols, groupCols,
                             quantCols, ctx);
        }
    }
}

struct QuantMXShapes {
    std::vector<int64_t> quantShape;
    std::vector<int64_t> groupedShape;
    std::vector<int64_t> expShape;
    std::vector<int64_t> performanceGroupedShape;
    std::vector<int64_t> performanceScalingShape;
    std::vector<int64_t> scalingShape;
    bool isDnAxis;
    int64_t rows;
    int64_t cols;
    int64_t groupCols;
    int64_t expCols;
    int64_t quantCols;
    int64_t prefixRows;
    int64_t mDim;
    int64_t nDim;
};

static void StoreQuantMXDnValue(uint8_t* quantPtr, int64_t row, int64_t nDim, int64_t quantCols, int64_t col,
                                float srcValue, float groupScaling, const QuantMXContext& ctx)
{
    const float scaled = ScaleQuantMXValue(srcValue, groupScaling, ctx);
    if (ctx.isFp4E2M1) {
        const uint8_t encoded = EncodeE2M1Magic(scaled);
        uint8_t& packed = quantPtr[row * quantCols + col / 2];
        packed = (col & 1) == 0 ? static_cast<uint8_t>((packed & 0xF0u) | encoded) :
                                  static_cast<uint8_t>((packed & 0x0Fu) | (encoded << 0x4));
        return;
    }
    quantPtr[row * nDim + col] = EncodeE4M3Fn(scaled);
}

static void FillQuantMXDn(const float* inputPtr, uint8_t* quantPtr, uint8_t* expPtr, float* scalingPtr, float* maxPtr,
                          const QuantMXShapes& shapes, DataType srcDtype, DataType quantDtype, int64_t mode)
{
    const QuantMXContext ctx = MakeQuantMXContext(srcDtype, quantDtype, false, mode);
    for (int64_t prefix = 0; prefix < shapes.prefixRows; ++prefix) {
        for (int64_t group = 0; group < shapes.groupCols; ++group) {
            for (int64_t col = 0; col < shapes.nDim; ++col) {
                float maxAbsValue = 0.0f;
                bool hasNaN = false;
                for (int64_t inner = 0; inner < MX_QUANT_TILE_BLOCK; ++inner) {
                    const int64_t row = group * MX_QUANT_TILE_BLOCK + inner;
                    const int64_t inputOffset = (prefix * shapes.mDim + row) * shapes.nDim + col;
                    const float val = NormalizeQuantMXMaxValue(inputPtr[inputOffset], ctx);
                    hasNaN = hasNaN || std::isnan(val);
                    maxAbsValue = std::isnan(val) ? maxAbsValue : std::max(maxAbsValue, val);
                }
                maxAbsValue = hasNaN ? std::numeric_limits<float>::quiet_NaN() : maxAbsValue;
                const auto exponentAndScaling = ComputeQuantMXExponentAndScaling(maxAbsValue, ctx);
                const int64_t scratchOffset = (prefix * shapes.groupCols + group) * shapes.nDim + col;
                const int64_t expOffset = (prefix * (shapes.groupCols / 2) + group / 2) * (shapes.nDim * 2) + col * 2 +
                                          group % 2;
                expPtr[expOffset] = exponentAndScaling.first;
                maxPtr[scratchOffset] = maxAbsValue;
                scalingPtr[scratchOffset] = exponentAndScaling.second;
                for (int64_t inner = 0; inner < MX_QUANT_TILE_BLOCK; ++inner) {
                    const int64_t row = group * MX_QUANT_TILE_BLOCK + inner;
                    const int64_t flatRow = prefix * shapes.mDim + row;
                    StoreQuantMXDnValue(quantPtr, flatRow, shapes.nDim, shapes.quantCols, col,
                                        inputPtr[flatRow * shapes.nDim + col], exponentAndScaling.second, ctx);
                }
            }
        }
    }
}

static std::vector<int64_t> BuildQuantMXPerformanceGroupedShape(const std::vector<int64_t>& inputShape,
                                                                int64_t groupCols)
{
    auto shape = inputShape;
    if (shape.size() == 1) {
        shape.back() = groupCols;
    } else {
        shape.pop_back();
        shape.back() *= groupCols;
    }
    return shape;
}

static QuantMXShapes BuildQuantMXShapes(const torch::Tensor& input, DataType srcDtype, DataType quantDtype,
                                        bool performanceMode, int64_t axis)
{
    QuantMXShapes shapes;
    shapes.quantShape = input.sizes().vec();
    const int64_t rank = static_cast<int64_t>(shapes.quantShape.size());
    const int64_t normalizedAxis = axis < 0 ? axis + rank : axis;
    shapes.isDnAxis = rank >= 0x2 && normalizedAxis == rank - 0x2;
    if (shapes.isDnAxis) {
        shapes.mDim = shapes.quantShape[rank - 2];
        shapes.nDim = shapes.quantShape.back();
        ASSERT(CalculatorErrorScene::QUANTMX_RANK_INVALID, shapes.mDim % 0x40 == 0)
            << "QuantMX axis=-2 interpreter requires the second-last dimension to be 64-aligned.";
        ASSERT(CalculatorErrorScene::QUANTMX_RANK_INVALID, shapes.nDim != 0)
            << "QuantMX input last dimension must not be zero.";
        if (quantDtype == DT_FP4_E2M1X2) {
            shapes.quantShape.back() = LastDimPackedCount(shapes.nDim, quantDtype);
        }
        shapes.groupedShape = input.sizes().vec();
        shapes.groupedShape[rank - 2] = shapes.mDim / MX_QUANT_TILE_BLOCK;
        shapes.expShape = input.sizes().vec();
        shapes.expShape[rank - 2] = shapes.mDim / 0x40;
        shapes.expShape.back() = shapes.nDim * 0x2;
        shapes.performanceGroupedShape = shapes.expShape;
        shapes.performanceScalingShape = shapes.groupedShape;
        shapes.scalingShape = shapes.groupedShape;
        shapes.rows = input.numel() / shapes.nDim;
        shapes.cols = shapes.nDim;
        shapes.groupCols = shapes.mDim / MX_QUANT_TILE_BLOCK;
        shapes.expCols = shapes.nDim * 0x2;
        shapes.quantCols = shapes.quantShape.back();
        shapes.prefixRows = input.numel() / (shapes.mDim * shapes.nDim);
        return shapes;
    }
    shapes.groupedShape = shapes.quantShape;
    shapes.cols = shapes.groupedShape.back();
    ASSERT(CalculatorErrorScene::QUANTMX_RANK_INVALID, shapes.cols != 0)
        << "QuantMX input last dimension must not be zero.";
    if (!performanceMode) {
        ASSERT(CalculatorErrorScene::QUANTMX_RANK_INVALID, shapes.cols % 0x40 == 0)
            << "QuantMX non-performance mode requires input last dimension to be a multiple of 64. Current last dim: "
            << shapes.cols;
    }
    if (quantDtype == DT_FP4_E2M1X2) {
        shapes.quantShape.back() = LastDimPackedCount(shapes.cols, quantDtype);
    }
    shapes.groupedShape.back() = (shapes.cols + MX_QUANT_TILE_BLOCK - 1) / MX_QUANT_TILE_BLOCK;
    shapes.performanceGroupedShape = BuildQuantMXPerformanceGroupedShape(shapes.quantShape, shapes.groupedShape.back());
    shapes.performanceScalingShape = shapes.performanceGroupedShape;
    if (srcDtype == DT_FP32) {
        shapes.performanceScalingShape.back() *= 0x2;
    }
    shapes.expShape = performanceMode ? shapes.performanceGroupedShape : input.sizes().vec();
    if (!performanceMode) {
        shapes.expShape.back() = shapes.groupedShape.back();
    }
    shapes.scalingShape = shapes.performanceScalingShape;
    shapes.rows = input.numel() / shapes.cols;
    shapes.groupCols = shapes.groupedShape.back();
    shapes.expCols = shapes.expShape.back();
    shapes.quantCols = shapes.quantShape.back();
    shapes.prefixRows = 0;
    shapes.mDim = 0;
    shapes.nDim = 0;
    return shapes;
}

static torch::Tensor CreateQuantMXQuantRaw(const std::vector<int64_t>& quantShape, DataType quantDtype)
{
    auto options = torch::TensorOptions().dtype(torch::kUInt8);
    return quantDtype == DT_FP4_E2M1X2 ? torch::zeros(quantShape, options) : torch::empty(quantShape, options);
}

static torch::Tensor CreateQuantMXScalingTemp(const std::vector<int64_t>& scalingShape, DataType quantDtype)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    return quantDtype == DT_FP4_E2M1X2 ? torch::zeros(scalingShape, options) : torch::empty(scalingShape, options);
}

static void CopyQuantMXOutputs(const std::pair<torch::Tensor, torch::Tensor>& tout,
                               const std::pair<torch::Tensor, torch::Tensor>& texp,
                               const std::pair<torch::Tensor, torch::Tensor>& tmax,
                               const std::pair<torch::Tensor, torch::Tensor>& tscaling, const torch::Tensor& quantRaw,
                               const torch::Tensor& expRaw, const torch::Tensor& maxTemp,
                               const torch::Tensor& scalingTemp, const QuantMXShapes& shapes, bool performanceMode)
{
    tout.first.copy_(quantRaw);
    const bool tailPerformanceMode = performanceMode && !shapes.isDnAxis;
    texp.first.copy_(tailPerformanceMode ? expRaw.reshape(shapes.performanceGroupedShape) : expRaw);
    tmax.second.copy_(tailPerformanceMode ? maxTemp.reshape(shapes.performanceGroupedShape) : maxTemp);
    tscaling.second.copy_(scalingTemp);
}

static void QuantMX(const TensorData& out, const TensorData& exp, const TensorData& max, const TensorData& scaling,
                    const TensorData& self, bool performanceMode, int64_t mode, int64_t axis)
{
    auto tout = From(out);
    auto texp = From(exp);
    auto tmax = From(max);
    auto tscaling = From(scaling);
    auto tself = From(self);

    auto input = tself.second.to(torch::kFloat32).contiguous();
    ASSERT(CalculatorErrorScene::QUANTMX_RANK_INVALID, input.dim() >= 1 && input.dim() <= 0x4)
        << "QuantMX interpreter only supports 1D to 4D input.";

    const QuantMXShapes shapes = BuildQuantMXShapes(input, self.dtype, out.dtype, performanceMode, axis);
    auto quantRaw = CreateQuantMXQuantRaw(shapes.quantShape, out.dtype);
    auto expRaw = torch::zeros(shapes.expShape, torch::TensorOptions().dtype(torch::kUInt8));
    auto scalingTemp = CreateQuantMXScalingTemp(shapes.scalingShape, out.dtype);
    auto maxTemp = torch::zeros(shapes.groupedShape, torch::TensorOptions().dtype(torch::kFloat32));
    if (shapes.isDnAxis) {
        auto inputFlat = input.view({shapes.prefixRows, shapes.mDim, shapes.nDim});
        auto inputContiguous = inputFlat.contiguous();
        FillQuantMXDn(inputContiguous.data_ptr<float>(), quantRaw.data_ptr<uint8_t>(), expRaw.data_ptr<uint8_t>(),
                      scalingTemp.data_ptr<float>(), maxTemp.data_ptr<float>(), shapes, self.dtype, out.dtype, mode);
        CopyQuantMXOutputs(tout, texp, tmax, tscaling, quantRaw, expRaw, maxTemp, scalingTemp, shapes, performanceMode);
        return;
    }
    auto inputFlat = input.view({shapes.rows, shapes.cols});
    auto quantFlat = quantRaw.view({shapes.rows, shapes.quantCols});
    auto expFlat = expRaw.view({shapes.rows, shapes.expCols});
    auto scalingFlat = scalingTemp.view({shapes.rows, shapes.groupCols * (self.dtype == DT_FP32 ? 2 : 1)});
    auto maxFlat = maxTemp.view({shapes.rows, shapes.groupCols});

    const auto* inputPtr = inputFlat.data_ptr<float>();
    auto* quantPtr = quantFlat.data_ptr<uint8_t>();
    auto* expPtr = expFlat.data_ptr<uint8_t>();
    auto* scalingPtr = scalingFlat.data_ptr<float>();
    auto* maxPtr = maxFlat.data_ptr<float>();

    FillQuantMXRows(inputPtr, quantPtr, expPtr, scalingPtr, maxPtr, shapes.rows, shapes.cols, shapes.expCols,
                    shapes.groupCols, shapes.quantCols, self.dtype, out.dtype, performanceMode, mode);
    CopyQuantMXOutputs(tout, texp, tmax, tscaling, quantRaw, expRaw, maxTemp, scalingTemp, shapes, performanceMode);
}

static struct CalcOps calcOps = {
    .Random = Random,
    .AllClose = AllClose,
    .Cast = Cast,
    .Exp = Exp,
    .Exp2 = Exp2,
    .Expm1 = Expm1,
    .Sin = Sin,
    .Cos = Cos,
    .Erf = Erf,
    .Sinh = Sinh,
    .Cosh = Cosh,
    .Erfc = Erfc,
    .Asin = Asin,
    .Acos = Acos,
    .ASinh = ASinh,
    .ACosh = ACosh,
    .Atanh = Atanh,
    .Neg = Neg,
    .Pack = Pack,
    .UnPack = UnPack,
    .Rsqrt = Rsqrt,
    .Sign = Sign,
    .Signbit = Signbit,
    .Tanh = Tanh,
    .Tan = Tan,
    .Sqrt = Sqrt,
    .Ceil = Ceil,
    .Floor = Floor,
    .Trunc = Trunc,
    .Round = Round,
    .Reciprocal = Reciprocal,
    .Relu = Relu,
    .Log1p = Log1p,
    .Pad = Pad,
    .FillPad = FillPad,
    .BitwiseNot = BitwiseNot,
    .Abs = Abs,
    .Brcb = Brcb,
    .WhereTT = WhereTT,
    .WhereTS = WhereTS,
    .WhereST = WhereST,
    .WhereSS = WhereSS,
    .LReLU = LReLU,
    .Ln = Ln,
    .IsFinite = IsFinite,
    .IsNan = IsNan,
    .LogicalNot = LogicalNot,
    .Range = Range,
    .Compare = Compare,
    .Cmps = Cmps,
    .Hypot = Hypot,
    .PReLU = PReLU,
    .LogicalAnd = LogicalAnd,
    .Uniform = Uniform,
    .AddS = AddS,
    .SubS = SubS,
    .MulS = MulS,
    .DivS = DivS,
    .FloorDivS = FloorDivS,
    .FmodS = FmodS,
    .RemainderS = RemainderS,
    .RemainderRS = RemainderRS,
    .PowS = PowS,
    .BitwiseAndS = BitwiseAndS,
    .BitwiseOrS = BitwiseOrS,
    .BitwiseXorS = BitwiseXorS,
    .GcdS = GcdS,
    .Add = Add,
    .Sub = Sub,
    .Mul = Mul,
    .Div = Div,
    .FloorDiv = FloorDiv,
    .Fmod = Fmod,
    .Remainder = Remainder,
    .Pow = Pow,
    .BitwiseAnd = BitwiseAnd,
    .BitwiseOr = BitwiseOr,
    .BitwiseXor = BitwiseXor,
    .ExpandExpDif = ExpandExpDif,
    .CopySign = CopySign,
    .Gcd = Gcd,
    .PairSum = PairSum,
    .PairMax = PairMax,
    .PairMin = PairMin,
    .PairProd = PairProd,
    .Min = Min,
    .Max = Max,
    .MinS = MinS,
    .MaxS = MaxS,
    .RowSumExpand = RowSumExpand,
    .RowMinExpand = RowMinExpand,
    .RowMaxExpand = RowMaxExpand,
    .RowSumSingle = RowSumSingle,
    .RowMinSingle = RowMinSingle,
    .RowMaxSingle = RowMaxSingle,
    .RowProdSingle = RowProdSingle,
    .RowMinLine = RowMinLine,
    .RowMaxLine = RowMaxLine,
    .RowProdLine = RowProdLine,
    .RowArgMaxSingle = RowArgMaxSingle,
    .RowArgMinSingle = RowArgMinSingle,
    .RowArgMaxWithValueSingle = RowArgMaxWithValueSingle,
    .RowArgMinWithValueSingle = RowArgMinWithValueSingle,
    .RowArgMaxWithValueLine = RowArgMaxWithValueLine,
    .RowArgMinWithValueLine = RowArgMinWithValueLine,
    .PairArgMax = PairArgMax,
    .PairArgMin = PairArgMin,
    .OneHot = OneHot,
    .ExpandS = ExpandS,
    .Expand = Expand,
    .GatherElements = GatherElements,
    .GatherMask = GatherMask,
    .IndexAdd = IndexAdd,
    .TriU = TriU,
    .TriL = TriL,
    .CumSum = CumSum,
    .CumProd = CumProd,
    .IndexPut = IndexPut,
    .Atan = Atan,
    .Atan2 = Atan2,
    .Reshape = Reshape,
    .Permute = Permute,
    .Transpose = Transpose,
    .ReduceAcc = ReduceAcc,
    .Copy = Copy,
    .ScatterUpdate = ScatterUpdate,
    .ScatterElement = ScatterElement,
    .Scatter = Scatter,
    .FormatND2NZ = FormatND2NZ,
    .FormatNZ2ND = FormatNZ2ND,
    .QuantPreCompute = QuantPreCompute,
    .MatMul = MatMul,
    .Quantize = Quantize,
    .Dequantize = Dequantize,
    .BitSort = BitSort,
    .TiledMrgSort = TiledMrgSort,
    .Extract = Extract,
    .MrgSort = MrgSort,
    .TopK = TopK,
    .QuantMX = QuantMX,
    .Interleave = Interleave,
    .DeInterleave = DeInterleave,
    .DeInterleaveSingle = DeInterleaveSingle,
    .TopkSort = TopkSort,
    .TopkMerge = TopkMerge,
    .TopkExtract = TopkExtract,
    .TwoTileMrgSort = TwoTileMrgSort,
    .Sort = Sort,
    .Gather = Gather,
    .GatherINUB = GatherINUB,
    .GatherInL1 = GatherInL1,
    .BitwiseRightShift = BitwiseRightShift,
    .BitwiseLeftShift = BitwiseLeftShift,
    .BitwiseRightShiftS = BitwiseRightShiftS,
    .BitwiseLeftShiftS = BitwiseLeftShiftS,
    .SBitwiseRightShift = SBitwiseRightShift,
    .SBitwiseLeftShift = SBitwiseLeftShift,
};

extern "C" struct CalcOps* GetCalcOps() { return &calcOps; }
} // namespace npu::tile_fwk
