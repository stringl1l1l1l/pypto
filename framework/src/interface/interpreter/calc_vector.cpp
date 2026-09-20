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
 * \file calc_vector.cpp
 * \brief
 */

#include "interface/interpreter/function.h"
#include "interface/interpreter/operation.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {

constexpr int GATHER_IGNORE_AXIS = -2;

template <Opcode opcode>
void ExecuteOpBinary(ExecuteOperationContext* ctx)
{
    if (opcode == Opcode::OP_ADD_BRC || opcode == Opcode::OP_SUB_BRC || opcode == Opcode::OP_MUL_BRC ||
        opcode == Opcode::OP_DIV_BRC) {
        ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == SIZE_TWO);
    } else if (opcode == Opcode::OP_BITWISEXOR || opcode == Opcode::OP_COPYSIGN || opcode == Opcode::OP_POW ||
               opcode == Opcode::OP_FLOORDIV || opcode == Opcode::OP_REM || opcode == Opcode::OP_GCD ||
               opcode == Opcode::OP_GCD_BRC) {
        ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() <= SIZE_TWO);
    } else {
        ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    }
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == SIZE_TWO);
    auto ret = ctx->ooperandInplaceDataViewList->at(0);
    auto tlhs = ctx->ioperandDataViewList->at(0);
    auto lhs = tlhs;
    auto trhs = ctx->ioperandDataViewList->at(1);
    auto rhs = trhs;
    auto lhsTensor = ctx->op->GetIOperands()[0];
    auto rhsTensor = ctx->op->GetIOperands()[1];
    bool lhsFromBrcb = !lhsTensor->GetProducers().empty() &&
                       (*lhsTensor->GetProducers().begin())->GetOpcode() == Opcode::OP_BRCB;
    bool rhsFromBrcb = !rhsTensor->GetProducers().empty() &&
                       (*rhsTensor->GetProducers().begin())->GetOpcode() == Opcode::OP_BRCB;

    // BRCB pads the last dim to block size; interpreter must view it as last-dim=1 for broadcast.
    // Preserve leading dims so 3D (e.g. [1,4,8] -> [1,4,1]) works the same as 2D ([4,8] -> [4,1]).
    auto MakeBrcbViewShape = [](const std::vector<int64_t>& shape) {
        auto viewShape = shape;
        if (!viewShape.empty()) {
            viewShape.back() = 1;
        }
        return viewShape;
    };
    if (lhsFromBrcb) {
        lhs = tlhs->View(MakeBrcbViewShape(tlhs->GetShape()), tlhs->GetOffset());
    } else if (rhsFromBrcb) {
        rhs = trhs->View(MakeBrcbViewShape(trhs->GetShape()), trhs->GetOffset());
    }

    if (lhsFromBrcb || rhsFromBrcb) {
        INTERPRETER_LOGI("AxisCombine: detected by BRCB, opcode=%s lhsFromBrcb=%d rhsFromBrcb=%d",
                         ctx->op->GetOpcodeStr().c_str(), static_cast<int>(lhsFromBrcb), static_cast<int>(rhsFromBrcb));
        INTERPRETER_LOGI("AxisCombine: lhs(shape=%s validShape=%s offset=%s) rhs(shape=%s validShape=%s offset=%s)",
                         IntVecToStr(lhs->GetShape()).c_str(), IntVecToStr(lhs->GetValidShape()).c_str(),
                         IntVecToStr(lhs->GetOffset()).c_str(), IntVecToStr(rhs->GetShape()).c_str(),
                         IntVecToStr(rhs->GetValidShape()).c_str(), IntVecToStr(rhs->GetOffset()).c_str());
    }

    if (opcode == Opcode::OP_ADD_BRC || opcode == Opcode::OP_SUB_BRC || opcode == Opcode::OP_MUL_BRC ||
        opcode == Opcode::OP_DIV_BRC) {
        bool axisCombine = ctx->op->GetBoolAttribute("input_combine_axis_done");
        lhs->SetAxisCombine(axisCombine);
        rhs->SetAxisCombine(axisCombine);
    }

    switch (opcode) {
        case Opcode::OP_ADD:
            calc::Add(ret, lhs, rhs);
            break;
        case Opcode::OP_ADD_BRC:
            calc::Add(ret, lhs, rhs);
            break;
        case Opcode::OP_PAIRSUM:
            calc::PairSum(ret, lhs, rhs);
            break;
        case Opcode::OP_PAIRPROD:
            calc::PairProd(ret, lhs, rhs);
            break;
        case Opcode::OP_SUB:
            calc::Sub(ret, lhs, rhs);
            break;
        case Opcode::OP_SUB_BRC:
            calc::Sub(ret, lhs, rhs);
            break;
        case Opcode::OP_MUL:
            calc::Mul(ret, lhs, rhs);
            break;
        case Opcode::OP_MUL_BRC:
            calc::Mul(ret, lhs, rhs);
            break;
        case Opcode::OP_DIV:
            calc::Div(ret, lhs, rhs);
            break;
        case Opcode::OP_DIV_BRC:
            calc::Div(ret, lhs, rhs);
            break;
        case Opcode::OP_FLOORDIV:
            calc::FloorDiv(ret, lhs, rhs);
            break;
        case Opcode::OP_ATAN2:
            calc::Atan2(ret, lhs, rhs);
            break;
        case Opcode::OP_POW:
            calc::Pow(ret, lhs, rhs);
            break;
        case Opcode::OP_REM:
            calc::Remainder(ret, lhs, rhs);
            break;
        case Opcode::OP_S_MAX:
            calc::Max(ret, lhs, rhs);
            break;
        case Opcode::OP_PAIRMAX:
            calc::PairMax(ret, lhs, rhs);
            break;
        case Opcode::OP_PAIRMIN:
            calc::PairMin(ret, lhs, rhs);
            break;
        case Opcode::OP_S_MIN:
            calc::Min(ret, lhs, rhs);
            break;
        case Opcode::OP_BITWISEAND:
            calc::BitwiseAnd(ret, lhs, rhs);
            break;
        case Opcode::OP_BITWISEOR:
            calc::BitwiseOr(ret, lhs, rhs);
            break;
        case Opcode::OP_BITWISEXOR:
            calc::BitwiseXor(ret, lhs, rhs);
            break;
        case Opcode::OP_EXPANDEXPDIF:
            calc::ExpandExpDif(ret, lhs, rhs);
            break;
        case Opcode::OP_COPYSIGN:
            calc::CopySign(ret, lhs, rhs);
            break;
        case Opcode::OP_GCD:
            calc::Gcd(ret, lhs, rhs);
            break;
        case Opcode::OP_GCD_BRC:
            calc::Gcd(ret, lhs, rhs);
            break;
        default:
            ASSERT(ExecuteOperationScene::UNSUPPORTED_OPCODE, false);
    }
}
REGISTER_CALC_OP(OP_ADD, Opcode::OP_ADD, ExecuteOpBinary<Opcode::OP_ADD>);
REGISTER_CALC_OP(OP_ADD_BRC, Opcode::OP_ADD_BRC, ExecuteOpBinary<Opcode::OP_ADD_BRC>);
REGISTER_CALC_OP(OP_SUB, Opcode::OP_SUB, ExecuteOpBinary<Opcode::OP_SUB>);
REGISTER_CALC_OP(OP_SUB_BRC, Opcode::OP_SUB_BRC, ExecuteOpBinary<Opcode::OP_SUB_BRC>);
REGISTER_CALC_OP(OP_MUL, Opcode::OP_MUL, ExecuteOpBinary<Opcode::OP_MUL>);
REGISTER_CALC_OP(OP_MUL_BRC, Opcode::OP_MUL_BRC, ExecuteOpBinary<Opcode::OP_MUL_BRC>);
REGISTER_CALC_OP(OP_DIV, Opcode::OP_DIV, ExecuteOpBinary<Opcode::OP_DIV>);
REGISTER_CALC_OP(OP_DIV_BRC, Opcode::OP_DIV_BRC, ExecuteOpBinary<Opcode::OP_DIV_BRC>);
REGISTER_CALC_OP(OP_FLOORDIV, Opcode::OP_FLOORDIV, ExecuteOpBinary<Opcode::OP_FLOORDIV>);
REGISTER_CALC_OP(OP_ATAN2, Opcode::OP_ATAN2, ExecuteOpBinary<Opcode::OP_ATAN2>);
REGISTER_CALC_OP(OP_POW, Opcode::OP_POW, ExecuteOpBinary<Opcode::OP_POW>);
REGISTER_CALC_OP(OP_REM, Opcode::OP_REM, ExecuteOpBinary<Opcode::OP_REM>);
REGISTER_CALC_OP(OP_S_ADD, Opcode::OP_S_ADD, ExecuteOpBinary<Opcode::OP_ADD>);
REGISTER_CALC_OP(OP_S_SUB, Opcode::OP_S_SUB, ExecuteOpBinary<Opcode::OP_SUB>);
REGISTER_CALC_OP(OP_S_MUL, Opcode::OP_S_MUL, ExecuteOpBinary<Opcode::OP_MUL>);
REGISTER_CALC_OP(OP_S_DIV, Opcode::OP_S_DIV, ExecuteOpBinary<Opcode::OP_DIV>);
REGISTER_CALC_OP(OP_PAIRMAX, Opcode::OP_PAIRMAX, ExecuteOpBinary<Opcode::OP_PAIRMAX>);
REGISTER_CALC_OP(OP_PAIRMIN, Opcode::OP_PAIRMIN, ExecuteOpBinary<Opcode::OP_PAIRMIN>);
REGISTER_CALC_OP(OP_PAIRSUM, Opcode::OP_PAIRSUM, ExecuteOpBinary<Opcode::OP_PAIRSUM>);
REGISTER_CALC_OP(OP_PAIRPROD, Opcode::OP_PAIRPROD, ExecuteOpBinary<Opcode::OP_PAIRPROD>);
REGISTER_CALC_OP(OP_S_MAX, Opcode::OP_S_MAX, ExecuteOpBinary<Opcode::OP_S_MAX>);
REGISTER_CALC_OP(OP_S_MIN, Opcode::OP_S_MIN, ExecuteOpBinary<Opcode::OP_S_MIN>);
REGISTER_CALC_OP(OP_MAXIMUM, Opcode::OP_MAXIMUM, ExecuteOpBinary<Opcode::OP_S_MAX>);
REGISTER_CALC_OP(OP_MINIMUM, Opcode::OP_MINIMUM, ExecuteOpBinary<Opcode::OP_S_MIN>);
REGISTER_CALC_OP(OP_BITWISEAND, Opcode::OP_BITWISEAND, ExecuteOpBinary<Opcode::OP_BITWISEAND>);
REGISTER_CALC_OP(OP_BITWISEOR, Opcode::OP_BITWISEOR, ExecuteOpBinary<Opcode::OP_BITWISEOR>);
REGISTER_CALC_OP(OP_BITWISEXOR, Opcode::OP_BITWISEXOR, ExecuteOpBinary<Opcode::OP_BITWISEXOR>);
REGISTER_CALC_OP(OP_EXPANDEXPDIF, Opcode::OP_EXPANDEXPDIF, ExecuteOpBinary<Opcode::OP_EXPANDEXPDIF>);
REGISTER_CALC_OP(OP_COPYSIGN, Opcode::OP_COPYSIGN, ExecuteOpBinary<Opcode::OP_COPYSIGN>);
REGISTER_CALC_OP(OP_GCD, Opcode::OP_GCD, ExecuteOpBinary<Opcode::OP_GCD>);
REGISTER_CALC_OP(OP_GCD_BRC, Opcode::OP_GCD_BRC, ExecuteOpBinary<Opcode::OP_GCD_BRC>);

void ExecuteOpFmod(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() <= SIZE_TWO);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == SIZE_TWO);
    auto ret = ctx->ooperandInplaceDataViewList->at(0);
    auto lhs = ctx->ioperandDataViewList->at(0);
    auto rhs = ctx->ioperandDataViewList->at(1);
    calc::Fmod(ret, lhs, rhs);
}
REGISTER_CALC_OP(OP_MOD, Opcode::OP_MOD, ExecuteOpFmod);

void ExecuteOpFmods(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() <= SIZE_TWO);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto& ret = ctx->ooperandInplaceDataViewList->at(0);
    auto& lhs = ctx->ioperandDataViewList->at(0);
    auto element = Element(DT_FP32, 0.0f);
    ctx->op->GetAttr(OpAttributeKey::scalar, element);
    calc::FmodS(ret, lhs, element);
}
REGISTER_CALC_OP(OP_MODS, Opcode::OP_MODS, ExecuteOpFmods);

void ExecuteOpVecDup(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 0);
    auto& ret = ctx->ooperandInplaceDataViewList->at(0);
    auto element = Element(DT_FP32, 0.0f);
    ctx->op->GetAttr(OpAttributeKey::scalar, element);
    calc::ExpandS(ret, element);
}
REGISTER_CALC_OP(OP_VEC_DUP, Opcode::OP_VEC_DUP, ExecuteOpVecDup);

void ExecuteOpWhereTT(ExecuteOperationContext* ctx)
{
    auto result = ctx->ooperandInplaceDataViewList->at(0);
    auto condition = ctx->ioperandDataViewList->at(0);
    auto input = ctx->ioperandDataViewList->at(1);
    auto other = ctx->ioperandDataViewList->at(2);
    calc::WhereTT(result, condition, input, other);
}
REGISTER_CALC_OP(OP_WHERE_TT, Opcode::OP_WHERE_TT, ExecuteOpWhereTT);

void ExecuteOpWhereTS(ExecuteOperationContext* ctx)
{
    auto result = ctx->ooperandInplaceDataViewList->at(0);
    auto condition = ctx->ioperandDataViewList->at(0);
    auto input = ctx->ioperandDataViewList->at(1);
    auto other = ctx->op->GetElementAttribute(OpAttributeKey::scalar);
    // OpAttributeKey::dynScalar
    calc::WhereTS(result, condition, input, other);
}
REGISTER_CALC_OP(OP_WHERE_TS, Opcode::OP_WHERE_TS, ExecuteOpWhereTS);

void ExecuteOpWhereST(ExecuteOperationContext* ctx)
{
    auto result = ctx->ooperandInplaceDataViewList->at(0);
    auto condition = ctx->ioperandDataViewList->at(0);
    auto other = ctx->ioperandDataViewList->at(1);
    auto input = ctx->op->GetElementAttribute(OpAttributeKey::scalar);
    // OpAttributeKey::dynScalar
    calc::WhereST(result, condition, input, other);
}
REGISTER_CALC_OP(OP_WHERE_ST, Opcode::OP_WHERE_ST, ExecuteOpWhereST);

void ExecuteOpWhereSS(ExecuteOperationContext* ctx)
{
    auto result = ctx->ooperandInplaceDataViewList->at(0);
    auto condition = ctx->ioperandDataViewList->at(0);
    auto scalars = ctx->op->GetVectorElementAttribute(OpAttributeKey::vectorScalar);
    auto input = scalars[0];
    auto other = scalars[1];
    calc::WhereSS(result, condition, input, other);
}
REGISTER_CALC_OP(OP_WHERE_SS, Opcode::OP_WHERE_SS, ExecuteOpWhereSS);

template <Opcode opcode>
void ExecuteOpReduce(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() <= SIZE_TWO);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto iop = ctx->ioperandDataViewList->at(0);
    int axis = ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "AXIS");
    if (oop->GetShape()[axis] != 1) {
        std::vector<int64_t> oopShape = oop->GetShape();
        oopShape[axis] = 1;
        oop = oop->View(oopShape, std::vector<int64_t>(oopShape.size(), 0));
    }

    switch (opcode) {
        case Opcode::OP_ROWSUM_SINGLE:
            calc::RowSumSingle(oop, iop, axis);
            break;
        case Opcode::OP_ROWMAX_SINGLE:
            calc::RowMaxSingle(oop, iop, axis);
            break;
        case Opcode::OP_ROWMIN_SINGLE:
            calc::RowMinSingle(oop, iop, axis);
            break;
        case Opcode::OP_ROWPROD_SINGLE:
            calc::RowProdSingle(oop, iop, axis);
            break;
        case Opcode::OP_ROWSUMLINE:
            calc::RowSumExpand(oop, iop, axis);
            break;
        case Opcode::OP_ROWMAXLINE:
            calc::RowMaxLine(oop, iop, axis);
            break;
        case Opcode::OP_ROWMINLINE:
            calc::RowMinLine(oop, iop, axis);
            break;
        case Opcode::OP_ROWPRODLINE:
            calc::RowProdLine(oop, iop, axis);
            break;
        default:
            ASSERT(ExecuteOperationScene::UNSUPPORTED_OPCODE, false) << "opcode not support" << ctx->op->GetOpcodeStr();
    }
}
REGISTER_CALC_OP(OP_ROWSUM_SINGLE, Opcode::OP_ROWSUM_SINGLE, ExecuteOpReduce<Opcode::OP_ROWSUM_SINGLE>);
REGISTER_CALC_OP(OP_ROWSUMLINE, Opcode::OP_ROWSUMLINE, ExecuteOpReduce<Opcode::OP_ROWSUMLINE>);
REGISTER_CALC_OP(OP_ROWMAX_SINGLE, Opcode::OP_ROWMAX_SINGLE, ExecuteOpReduce<Opcode::OP_ROWMAX_SINGLE>);
REGISTER_CALC_OP(OP_ROWMAXLINE, Opcode::OP_ROWMAXLINE, ExecuteOpReduce<Opcode::OP_ROWMAXLINE>);
REGISTER_CALC_OP(OP_ROWMIN_SINGLE, Opcode::OP_ROWMIN_SINGLE, ExecuteOpReduce<Opcode::OP_ROWMIN_SINGLE>);
REGISTER_CALC_OP(OP_ROWMINLINE, Opcode::OP_ROWMINLINE, ExecuteOpReduce<Opcode::OP_ROWMINLINE>);
REGISTER_CALC_OP(OP_ROWPROD_SINGLE, Opcode::OP_ROWPROD_SINGLE, ExecuteOpReduce<Opcode::OP_ROWPROD_SINGLE>);
REGISTER_CALC_OP(OP_ROWPRODLINE, Opcode::OP_ROWPRODLINE, ExecuteOpReduce<Opcode::OP_ROWPRODLINE>);

template <Opcode opcode>
void ExecuteOpArgReduceSingle(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto iop = ctx->ioperandDataViewList->at(0);
    int axis = ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "AXIS");
    if (oop->GetShape()[axis] != 1) {
        std::vector<int64_t> oopShape = oop->GetShape();
        oopShape[axis] = 1;
        oop = oop->View(oopShape, std::vector<int64_t>(oopShape.size(), 0));
    }

    switch (opcode) {
        case Opcode::OP_ROWARGMAX_SINGLE:
            calc::RowArgMaxSingle(oop, iop, axis);
            break;
        case Opcode::OP_ROWARGMIN_SINGLE:
            calc::RowArgMinSingle(oop, iop, axis);
            break;
        default:
            ASSERT(ExecuteOperationScene::UNSUPPORTED_OPCODE, false) << "opcode not support" << ctx->op->GetOpcodeStr();
    }
}
REGISTER_CALC_OP(OP_ROWARGMAX_SINGLE, Opcode::OP_ROWARGMAX_SINGLE, ExecuteOpReduce<Opcode::OP_ROWARGMAX_SINGLE>);
REGISTER_CALC_OP(OP_ROWARGMIN_SINGLE, Opcode::OP_ROWARGMIN_SINGLE, ExecuteOpReduce<Opcode::OP_ROWARGMIN_SINGLE>);

template <Opcode opcode>
void ExecuteOpArgReduceWithValue(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto outValue = ctx->ooperandInplaceDataViewList->at(0);
    auto outIndex = ctx->ooperandInplaceDataViewList->at(1);
    auto outTemp = ctx->ooperandInplaceDataViewList->at(2);
    auto iop = ctx->ioperandDataViewList->at(0);
    int axis = ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "AXIS");
    switch (opcode) {
        case Opcode::OP_ROWARGMAXWITHVALUE_SINGLE:
            calc::RowArgMaxWithValueSingle(outValue, outIndex, outTemp, iop, axis);
            break;
        case Opcode::OP_ROWARGMINWITHVALUE_SINGLE:
            calc::RowArgMinWithValueSingle(outValue, outIndex, outTemp, iop, axis);
            break;
        case Opcode::OP_ROWARGMAXWITHVALUE_LINE:
            calc::RowArgMaxWithValueLine(outValue, outIndex, outTemp, iop, axis);
            break;
        case Opcode::OP_ROWARGMINWITHVALUE_LINE:
            calc::RowArgMinWithValueLine(outValue, outIndex, outTemp, iop, axis);
            break;
        default:
            ASSERT(ExecuteOperationScene::UNSUPPORTED_OPCODE, false) << "opcode not support" << ctx->op->GetOpcodeStr();
    }
}
REGISTER_CALC_OP(OP_ROWARGMAXWITHVALUE_SINGLE, Opcode::OP_ROWARGMAXWITHVALUE_SINGLE,
                 ExecuteOpReduce<Opcode::OP_ROWARGMAXWITHVALUE_SINGLE>);
REGISTER_CALC_OP(OP_ROWARGMAXWITHVALUE_LINE, Opcode::OP_ROWARGMAXWITHVALUE_LINE,
                 ExecuteOpReduce<Opcode::OP_ROWARGMAXWITHVALUE_LINE>);
REGISTER_CALC_OP(OP_ROWARGMINWITHVALUE_SINGLE, Opcode::OP_ROWARGMINWITHVALUE_SINGLE,
                 ExecuteOpReduce<Opcode::OP_ROWARGMINWITHVALUE_SINGLE>);
REGISTER_CALC_OP(OP_ROWARGMINWITHVALUE_LINE, Opcode::OP_ROWARGMINWITHVALUE_LINE,
                 ExecuteOpReduce<Opcode::OP_ROWARGMINWITHVALUE_LINE>);

template <Opcode opcode>
void ExecuteOpPairArgRedyce(ExecuteOperationContext* ctx)
{
    auto outValue = ctx->ooperandInplaceDataViewList->at(0);
    auto outIndex = ctx->ooperandInplaceDataViewList->at(1);
    auto value1 = ctx->ioperandDataViewList->at(0);
    auto index1 = ctx->ioperandDataViewList->at(1);
    auto value2 = ctx->ioperandDataViewList->at(2);
    auto index2 = ctx->ioperandDataViewList->at(3);
    int axis = ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "AXIS");
    switch (opcode) {
        case Opcode::OP_PAIRARGMAX:
            calc::PairArgMax(outValue, outIndex, value1, index1, value2, index2);
            break;
        case Opcode::OP_PAIRARGMIN:
            calc::PairArgMin(outValue, outIndex, value1, index1, value2, index2);
            break;
        default:
            ASSERT(ExecuteOperationScene::UNSUPPORTED_OPCODE, false) << "opcode not support" << ctx->op->GetOpcodeStr();
    }
}
REGISTER_CALC_OP(OP_PAIRARGMAX, Opcode::OP_PAIRARGMAX, ExecuteOpReduce<Opcode::OP_PAIRARGMAX>);
REGISTER_CALC_OP(OP_PAIRARGMIN, Opcode::OP_PAIRARGMIN, ExecuteOpReduce<Opcode::OP_PAIRARGMIN>);

void ExecuteOpCast(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() <= 0x2);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto& ret = ctx->ooperandInplaceDataViewList->at(0);
    auto& iop = ctx->ioperandDataViewList->at(0);
    CastMode mode = static_cast<CastMode>(ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "mode"));
    calc::Cast(ret, iop, mode);
}
REGISTER_CALC_OP(OP_CAST, Opcode::OP_CAST, ExecuteOpCast);

template <Opcode opcode>
void ExecuteOpUnary(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto& ret = ctx->ooperandInplaceDataViewList->at(0);
    auto& iop = ctx->ioperandDataViewList->at(0);
    switch (opcode) {
        case Opcode::OP_EXP:
            calc::Exp(ret, iop);
            break;
        case Opcode::OP_SINH:
            calc::Sinh(ret, iop);
            break;
        case Opcode::OP_COSH:
            calc::Cosh(ret, iop);
            break;
        case Opcode::OP_ASIN:
            calc::Asin(ret, iop);
            break;
        case Opcode::OP_ACOS:
            calc::Acos(ret, iop);
            break;
        case Opcode::OP_ASINH:
            calc::ASinh(ret, iop);
            break;
        case Opcode::OP_ACOSH:
            calc::ACosh(ret, iop);
            break;
        case Opcode::OP_ATANH:
            calc::Atanh(ret, iop);
            break;
        case Opcode::OP_NEG:
            calc::Neg(ret, iop);
            break;
        case Opcode::OP_SIGN:
            calc::Sign(ret, iop);
            break;
        case Opcode::OP_SIGNBIT:
            calc::Signbit(ret, iop);
            break;
        case Opcode::OP_TANH:
            calc::Tanh(ret, iop);
            break;
        case Opcode::OP_TAN:
            calc::Tan(ret, iop);
            break;
        case Opcode::OP_RSQRT:
            calc::Rsqrt(ret, iop);
            break;
        case Opcode::OP_SQRT:
            calc::Sqrt(ret, iop);
            break;
        case Opcode::OP_RECIPROCAL:
            calc::Reciprocal(ret, iop);
            break;
        case Opcode::OP_RELU:
            calc::Relu(ret, iop);
            break;
        case Opcode::OP_ATAN:
            calc::Atan(ret, iop);
            break;
        case Opcode::OP_BITWISENOT:
            calc::BitwiseNot(ret, iop);
            break;
        case Opcode::OP_ABS:
            calc::Abs(ret, iop);
            break;
        case Opcode::OP_BRCB:
            calc::Brcb(ret, iop);
            break;
        case Opcode::OP_LN:
            calc::Ln(ret, iop);
            break;
        case Opcode::OP_ISFINITE:
            calc::IsFinite(ret, iop);
            break;
        case Opcode::OP_ISNAN:
            calc::IsNan(ret, iop);
            break;
        case Opcode::OP_ERF:
            calc::Erf(ret, iop);
            break;
        case Opcode::OP_PACK:
            calc::Pack(ret, iop);
            break;
        case Opcode::OP_UNPACK:
            calc::UnPack(ret, iop);
            break;
        case Opcode::OP_SIN:
            calc::Sin(ret, iop);
            break;
        case Opcode::OP_COS:
            calc::Cos(ret, iop);
            break;
        default:
            ASSERT(ExecuteOperationScene::UNSUPPORTED_OPCODE, false);
    }
}
REGISTER_CALC_OP(OP_EXP, Opcode::OP_EXP, ExecuteOpUnary<Opcode::OP_EXP>);
REGISTER_CALC_OP(OP_SINH, Opcode::OP_SINH, ExecuteOpUnary<Opcode::OP_SINH>);
REGISTER_CALC_OP(OP_COSH, Opcode::OP_COSH, ExecuteOpUnary<Opcode::OP_COSH>);
REGISTER_CALC_OP(OP_ASIN, Opcode::OP_ASIN, ExecuteOpUnary<Opcode::OP_ASIN>);
REGISTER_CALC_OP(OP_ACOS, Opcode::OP_ACOS, ExecuteOpUnary<Opcode::OP_ACOS>);
REGISTER_CALC_OP(OP_ASINH, Opcode::OP_ASINH, ExecuteOpUnary<Opcode::OP_ASINH>);
REGISTER_CALC_OP(OP_ACOSH, Opcode::OP_ACOSH, ExecuteOpUnary<Opcode::OP_ACOSH>);
REGISTER_CALC_OP(OP_ATANH, Opcode::OP_ATANH, ExecuteOpUnary<Opcode::OP_ATANH>);
REGISTER_CALC_OP(OP_NEG, Opcode::OP_NEG, ExecuteOpUnary<Opcode::OP_NEG>);
REGISTER_CALC_OP(OP_SIGN, Opcode::OP_SIGN, ExecuteOpUnary<Opcode::OP_SIGN>);
REGISTER_CALC_OP(OP_SIGNBIT, Opcode::OP_SIGNBIT, ExecuteOpUnary<Opcode::OP_SIGNBIT>);
REGISTER_CALC_OP(OP_TANH, Opcode::OP_TANH, ExecuteOpUnary<Opcode::OP_TANH>);
REGISTER_CALC_OP(OP_TAN, Opcode::OP_TAN, ExecuteOpUnary<Opcode::OP_TAN>);
REGISTER_CALC_OP(OP_RSQRT, Opcode::OP_RSQRT, ExecuteOpUnary<Opcode::OP_RSQRT>);
REGISTER_CALC_OP(OP_SQRT, Opcode::OP_SQRT, ExecuteOpUnary<Opcode::OP_SQRT>);
REGISTER_CALC_OP(OP_RECIPROCAL, Opcode::OP_RECIPROCAL, ExecuteOpUnary<Opcode::OP_RECIPROCAL>);
REGISTER_CALC_OP(OP_RELU, Opcode::OP_RELU, ExecuteOpUnary<Opcode::OP_RELU>);
REGISTER_CALC_OP(OP_ATAN, Opcode::OP_ATAN, ExecuteOpUnary<Opcode::OP_ATAN>);
REGISTER_CALC_OP(OP_BITWISENOT, Opcode::OP_BITWISENOT, ExecuteOpUnary<Opcode::OP_BITWISENOT>);
REGISTER_CALC_OP(OP_ABS, Opcode::OP_ABS, ExecuteOpUnary<Opcode::OP_ABS>);
REGISTER_CALC_OP(OP_BRCB, Opcode::OP_BRCB, ExecuteOpUnary<Opcode::OP_BRCB>);
REGISTER_CALC_OP(OP_LN, Opcode::OP_LN, ExecuteOpUnary<Opcode::OP_LN>);
REGISTER_CALC_OP(OP_ISFINITE, Opcode::OP_ISFINITE, ExecuteOpUnary<Opcode::OP_ISFINITE>);
REGISTER_CALC_OP(OP_ISNAN, Opcode::OP_ISNAN, ExecuteOpUnary<Opcode::OP_ISNAN>);
REGISTER_CALC_OP(OP_ERF, Opcode::OP_ERF, ExecuteOpUnary<Opcode::OP_ERF>);
REGISTER_CALC_OP(OP_PACK, Opcode::OP_PACK, ExecuteOpUnary<Opcode::OP_PACK>);
REGISTER_CALC_OP(OP_UNPACK, Opcode::OP_UNPACK, ExecuteOpUnary<Opcode::OP_UNPACK>);
REGISTER_CALC_OP(OP_SIN, Opcode::OP_SIN, ExecuteOpUnary<Opcode::OP_SIN>);
REGISTER_CALC_OP(OP_COS, Opcode::OP_COS, ExecuteOpUnary<Opcode::OP_COS>);

void ExecuteOpCeil(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto& ret = ctx->ooperandInplaceDataViewList->at(0);
    auto& iop = ctx->ioperandDataViewList->at(0);
    calc::Ceil(ret, iop);
}
REGISTER_CALC_OP(OP_CEIL, Opcode::OP_CEIL, ExecuteOpCeil);

void ExecuteOpFloor(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto& ret = ctx->ooperandInplaceDataViewList->at(0);
    auto& iop = ctx->ioperandDataViewList->at(0);
    calc::Floor(ret, iop);
}
REGISTER_CALC_OP(OP_FLOOR, Opcode::OP_FLOOR, ExecuteOpFloor);

void ExecuteOpTrunc(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto& ret = ctx->ooperandInplaceDataViewList->at(0);
    auto& iop = ctx->ioperandDataViewList->at(0);
    calc::Trunc(ret, iop);
}
REGISTER_CALC_OP(OP_TRUNC, Opcode::OP_TRUNC, ExecuteOpTrunc);

void ExecuteOpExp2(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() <= 0x3);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto& ret = ctx->ooperandInplaceDataViewList->at(0);
    auto& iop = ctx->ioperandDataViewList->at(0);
    calc::Exp2(ret, iop);
}
REGISTER_CALC_OP(OP_EXP2, Opcode::OP_EXP2, ExecuteOpExp2);

void ExecuteOpPad(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);

    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto iop_input = ctx->ioperandDataViewList->at(0);
    auto element = Element(DT_FP32, 0.0f);
    ctx->op->GetAttr(OpAttributeKey::scalar, element);
    calc::Pad(oop, iop_input, element);
}
REGISTER_CALC_OP(OP_PAD, Opcode::OP_PAD, ExecuteOpPad);

void ExecuteOpFillPad(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);

    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto iop_input = ctx->ioperandDataViewList->at(0);
    auto element = Element(DT_FP32, 0.0f);
    ctx->op->GetAttr(OpAttributeKey::scalar, element);
    calc::FillPad(oop, iop_input, element);
}
REGISTER_CALC_OP(OP_FILLPAD, Opcode::OP_FILLPAD, ExecuteOpFillPad);

void ExecuteOpRound(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() <= SIZE_TWO);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto& output = ctx->ooperandInplaceDataViewList->at(0);
    auto& input = ctx->ioperandDataViewList->at(0);

    int decimals = ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "decimals");
    calc::Round(output, input, decimals);
}
REGISTER_CALC_OP(OP_ROUND, Opcode::OP_ROUND, ExecuteOpRound);

void ExecuteOpExpm1(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() <= SIZE_TWO);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto& output = ctx->ooperandInplaceDataViewList->at(0);
    auto& input = ctx->ioperandDataViewList->at(0);

    calc::Expm1(output, input);
}
REGISTER_CALC_OP(OP_EXPM1, Opcode::OP_EXPM1, ExecuteOpExpm1);

void ExecuteOpErfc(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() <= SIZE_TWO);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto& output = ctx->ooperandInplaceDataViewList->at(0);
    auto& input = ctx->ioperandDataViewList->at(0);

    calc::Erfc(output, input);
}
REGISTER_CALC_OP(OP_ERFC, Opcode::OP_ERFC, ExecuteOpErfc);

void ExecuteOpOneHot(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto& ret = ctx->ooperandInplaceDataViewList->at(0);
    auto& iop = ctx->ioperandDataViewList->at(0);
    int numClasses = ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "numClasses");
    calc::OneHot(ret, iop, numClasses);
}
REGISTER_CALC_OP(OP_ONEHOT, Opcode::OP_ONEHOT, ExecuteOpOneHot);

void ExecuteOpExpand(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto iop = ctx->ioperandDataViewList->at(0);
    calc::Expand(oop, iop);
}
REGISTER_CALC_OP(OP_EXPAND, Opcode::OP_EXPAND, ExecuteOpExpand);

void ExecuteOpTransposeMoveOut(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() <= SIZE_TWO);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto iop = ctx->ioperandDataViewList->at(0);

    std::vector<int64_t> axises = ctx->op->GetVectorIntAttribute(OP_ATTR_PREFIX + "shape");
    if (std::dynamic_pointer_cast<CopyOpAttribute>(ctx->op->GetOpAttribute())) {
        auto copyoutAttr = std::dynamic_pointer_cast<CopyOpAttribute>(ctx->op->GetOpAttribute());
        std::vector<int64_t> shape = ctx->opInter->EvaluateOpImmediate(ctx->frame, copyoutAttr->GetShape());
        if (ctx->op->GetOpcode() == Opcode::OP_TRANSPOSE_MOVEOUT) {
            std::vector<int64_t> toOffset = ctx->opInter->EvaluateOpImmediate(ctx->frame, copyoutAttr->GetToOffset());
            std::vector<int64_t> iopShape = iop->GetShape();
            std::swap(iopShape[axises[0]], iopShape[axises[1]]);
            auto oopCopy = std::make_shared<LogicalTensorData>(oop->GetData(), iopShape, toOffset);
            return calc::Transpose(oopCopy, iop, axises[0], axises[1]);
        } else {
            std::vector<int64_t> fromOffset = ctx->opInter->EvaluateOpImmediate(ctx->frame,
                                                                                copyoutAttr->GetFromOffset());
            std::vector<int64_t> oopShape = oop->GetShape();
            std::swap(oopShape[axises[0]], oopShape[axises[1]]);
            auto iopCopy = std::make_shared<LogicalTensorData>(iop->GetData(), oopShape, fromOffset);
            return calc::Transpose(oop, iopCopy, axises[0], axises[1]);
        }
    }
    calc::Transpose(oop, iop, axises[0], axises[1]);
}
REGISTER_CALC_OP(OP_TRANSPOSE_MOVEOUT, Opcode::OP_TRANSPOSE_MOVEOUT, ExecuteOpTransposeMoveOut);
REGISTER_CALC_OP(OP_TRANSPOSE_MOVEIN, Opcode::OP_TRANSPOSE_MOVEIN, ExecuteOpTransposeMoveOut);

void ExecuteOpTranspose(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() <= SIZE_TWO);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto iop = ctx->ioperandDataViewList->at(0);
    auto axises = ctx->op->GetVectorIntAttribute(OP_ATTR_PREFIX + "shape");
    calc::Transpose(oop, iop, axises[0], axises[1]);
}
REGISTER_CALC_OP(OP_TRANSPOSE_VNCHWCONV, Opcode::OP_TRANSPOSE_VNCHWCONV, ExecuteOpTranspose);

void ExecuteOpPermute(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() <= SIZE_TWO);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto& oop = ctx->ooperandInplaceDataViewList->at(0);
    auto& iop = ctx->ioperandDataViewList->at(0);

    std::vector<int64_t> perm = ctx->op->GetVectorIntAttribute(OpAttributeKey::perm);

    auto iopDataView = iop->View(iop->GetValidShape(), iop->GetOffset());
    auto oopDataView = oop->View(oop->GetValidShape(), oop->GetOffset());
    calc::Permute(oopDataView, iopDataView, perm);
}
REGISTER_CALC_OP(OP_PERMUTE, Opcode::OP_PERMUTE, ExecuteOpPermute);
REGISTER_CALC_OP(OP_PERMUTE_ELEMENT, Opcode::OP_PERMUTE_ELEMENT, ExecuteOpPermute);

void ExecuteOpLogicalNot(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto iop = ctx->ioperandDataViewList->at(0);
    calc::LogicalNot(oop, iop);
}
REGISTER_CALC_OP(OP_LOGICALNOT, Opcode::OP_LOGICALNOT, ExecuteOpLogicalNot);

void ExecuteOpLogicalAnd(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == SIZE_TWO);
    auto ret = ctx->ooperandInplaceDataViewList->at(0);
    auto lhs = ctx->ioperandDataViewList->at(0);
    auto rhs = ctx->ioperandDataViewList->at(1);
    calc::LogicalAnd(ret, lhs, rhs);
}
REGISTER_CALC_OP(OP_LOGICALAND, Opcode::OP_LOGICALAND, ExecuteOpLogicalAnd);

void ExecuteOpIndexOutcast(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == SIZE_THREE);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto src = ctx->ioperandDataViewList->at(0);
    auto index = ctx->ioperandDataViewList->at(1);
    auto dst = ctx->ioperandDataViewList->at(2);
    int axis = ctx->op->GetIntAttribute("axis");
    int blockSize = ctx->op->GetIntAttribute(OpAttributeKey::panzBlockSize);
    std::string cacheMode = ctx->op->GetStringAttribute(OpAttributeKey::cacheMode);
    auto actualOop = std::make_shared<LogicalTensorData>(dst->GetData());
    if (dst->GetSize() != oop->GetSize()) {
        INTERPRETER_EVENT("%s", ctx->op->Dump().c_str());
        INTERPRETER_EVENT("dst validShape: %s ---> oop validShape: %s", IntVecToStr(dst->GetShape()).c_str(),
                          IntVecToStr(oop->GetShape()).c_str());
        INTERPRETER_EVENT("IndexOutcast: oop validShape is not equal to dst validShape");
        calc::ScatterUpdate(actualOop, src, index, dst, axis, cacheMode, blockSize);
    } else {
        calc::ScatterUpdate(oop, src, index, dst, axis, cacheMode, blockSize);
    }
}
REGISTER_CALC_OP(OP_INDEX_OUTCAST, Opcode::OP_INDEX_OUTCAST, ExecuteOpIndexOutcast);

void ExecuteOpScatterElement(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == SIZE_TWO);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto self = ctx->ioperandDataViewList->at(0);
    auto indices = ctx->ioperandDataViewList->at(1);
    int axis = ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "axis");
    auto src = Element(DT_FP32, 0.0f);
    ctx->op->GetAttr(OpAttributeKey::scalar, src);
    int reduce = ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "scatter_mode");

    calc::ScatterElement(oop, self, indices, src, axis, reduce);
}
REGISTER_CALC_OP(OP_SCATTER_ELEMENT, Opcode::OP_SCATTER_ELEMENT, ExecuteOpScatterElement);

void ExecuteOpScatter(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == SIZE_THREE);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto self = ctx->ioperandDataViewList->at(0);
    auto indices = ctx->ioperandDataViewList->at(1);
    auto src = ctx->ioperandDataViewList->at(2);
    int axis = ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "axis");
    int reduce = ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "scatter_mode");

    calc::Scatter(oop, self, indices, src, axis, reduce);
}
REGISTER_CALC_OP(OP_SCATTER, Opcode::OP_SCATTER, ExecuteOpScatter);

template <typename T, DataType dataType>
Element GetEndBySize(Element start, Element size, Element step)
{
    T startValue;
    T stepValue;
    if (dataType == DT_INT32 || dataType == DT_INT64) {
        startValue = start.GetSignedData();
        stepValue = step.GetSignedData();
    } else if (dataType == DT_FP32) {
        startValue = (float)start.GetFloatData();
        stepValue = (float)step.GetFloatData();
    }
    T endValue = startValue + size.GetSignedData() * stepValue - stepValue / 2;
    Element end(dataType, endValue);
    return end;
}

void ExecuteOpRange(ExecuteOperationContext* ctx)
{
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto start = ctx->op->GetElementAttribute(OP_ATTR_PREFIX + "START");
    auto size = ctx->op->GetElementAttribute(OP_ATTR_PREFIX + "SIZE");
    auto step = ctx->op->GetElementAttribute(OP_ATTR_PREFIX + "STEP");
    Element curStart = start;
    if (ctx->op->HasAttr(OpAttributeKey::dynScalar)) {
        SymbolicScalar tileIdx = ctx->op->GetSymbolicScalarAttribute(OpAttributeKey::dynScalar);
        if (tileIdx.ConcreteValid()) {
            int64_t tileIdxVal = tileIdx.Concrete();
            Element tileIdxElem = Element(start.GetDataType(), tileIdxVal);
            curStart = start + step * tileIdxElem;
        } else {
            std::vector<OpImmediate> dynScalarImmList = {OpImmediate::Specified(tileIdx)};
            auto resolved = ctx->opInter->EvaluateOpImmediate(ctx->frame, dynScalarImmList);
            if (!resolved.empty()) {
                int64_t tileIdxVal = resolved[0];
                Element tileIdxElem = Element(start.GetDataType(), tileIdxVal);
                curStart = start + step * tileIdxElem;
            }
        }
    }
    Element end;
    if (start.GetDataType() == DT_INT32) {
        end = GetEndBySize<int32_t, DT_INT32>(curStart, size, step);
    } else if (start.GetDataType() == DT_INT64) {
        end = GetEndBySize<int64_t, DT_INT64>(curStart, size, step);
    } else if (start.GetDataType() == DT_FP32) {
        end = GetEndBySize<float, DT_FP32>(curStart, size, step);
    } else {
        ASSERT(ExecuteOperationScene::INVALID_TENSOR_DTYPE, false)
            << "Unsupported DataType " << DataType2String(start.GetDataType());
    }
    calc::Range(oop, curStart, end, step);
}
REGISTER_CALC_OP(OP_RANGE, Opcode::OP_RANGE, ExecuteOpRange);

void ExecuteOpUniform(ExecuteOperationContext* ctx)
{
    auto oop = ctx->ooperandInplaceDataViewList->at(0);

    auto scalars = ctx->op->GetVectorElementAttribute(OpAttributeKey::vectorScalar);
    Element key = scalars[0];
    Element counter1 = scalars[1];
    Element rounds = scalars[2];
    DataType dtype = static_cast<DataType>(scalars[3].Cast<int32_t>());

    Element counter0(DT_UINT64, static_cast<uint64_t>(0));
    if (ctx->op->HasAttr(OpAttributeKey::dynScalar)) {
        SymbolicScalar dynScalar = ctx->op->GetSymbolicScalarAttribute(OpAttributeKey::dynScalar);
        if (dynScalar.ConcreteValid()) {
            counter0 = Element(DT_UINT64, static_cast<uint64_t>(dynScalar.Concrete()));
        }
    }

    calc::Uniform(oop, key, counter0, counter1, rounds, dtype);
}
REGISTER_CALC_OP(OP_UNIFORM, Opcode::OP_UNIFORM, ExecuteOpUniform);
void ExecuteOpQuantMX(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 0x4);
    int64_t mode = 0;
    ASSERT(ExecuteOperationScene::RUNTIME_EXCEPTION, ctx->op->GetAttr(OpAttributeKey::mxQuantMode, mode))
        << "QuantMX missing required attribute: " << OpAttributeKey::mxQuantMode;
    constexpr int64_t kMxQuantModeRoundUp = 0;
    constexpr int64_t kMxQuantModeRoundDown = 1;
    ASSERT(ExecuteOperationScene::RUNTIME_EXCEPTION, mode == kMxQuantModeRoundDown || mode == kMxQuantModeRoundUp)
        << "QuantMX interpreter currently only supports ROUND_DOWN (OCP) and ROUND_UP (NV) modes.";
    int64_t axis = 0;
    ASSERT(ExecuteOperationScene::RUNTIME_EXCEPTION, ctx->op->GetAttr(OpAttributeKey::mxQuantAxis, axis))
        << "QuantMX missing required attribute: " << OpAttributeKey::mxQuantAxis;
    int64_t performanceMode = 0;
    ASSERT(ExecuteOperationScene::RUNTIME_EXCEPTION,
           ctx->op->GetAttr(OpAttributeKey::mxQuantPerformanceMode, performanceMode))
        << "QuantMX missing required attribute: " << OpAttributeKey::mxQuantPerformanceMode;
    auto out = ctx->ooperandInplaceDataViewList->at(0);
    auto exp = ctx->ooperandInplaceDataViewList->at(1);
    auto max = ctx->ooperandInplaceDataViewList->at(2);
    auto scaling = ctx->ooperandInplaceDataViewList->at(3);
    auto src = ctx->ioperandDataViewList->at(0);
    const auto srcRank = static_cast<int64_t>(src->GetShape().size());
    const auto normalizedAxis = axis < 0 ? axis + srcRank : axis;
    ASSERT(ExecuteOperationScene::RUNTIME_EXCEPTION, normalizedAxis >= 0 && normalizedAxis < srcRank)
        << "QuantMX axis is out of range. Current axis: " << axis << ", input rank: " << srcRank;
    ASSERT(ExecuteOperationScene::RUNTIME_EXCEPTION,
           normalizedAxis == srcRank - 1 || (srcRank >= 0x2 && normalizedAxis == srcRank - 0x2))
        << "QuantMX interpreter currently only supports the last axis and second-last axis. Current axis: " << axis
        << ", input rank: " << srcRank;
    calc::QuantMX(out, exp, max, scaling, src, performanceMode != 0, mode, normalizedAxis);
}
REGISTER_CALC_OP(OP_QUANT_MX, Opcode::OP_QUANT_MX, ExecuteOpQuantMX);

void ExecuteOpInterleave(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == SIZE_TWO);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == SIZE_TWO);
    auto out0 = ctx->ooperandInplaceDataViewList->at(0);
    auto out1 = ctx->ooperandInplaceDataViewList->at(1);
    auto src0 = ctx->ioperandDataViewList->at(0);
    auto src1 = ctx->ioperandDataViewList->at(1);
    calc::Interleave(out0, out1, src0, src1);
}
REGISTER_CALC_OP(OP_INTERLEAVE, Opcode::OP_INTERLEAVE, ExecuteOpInterleave);

void ExecuteOpDeInterleave(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == SIZE_TWO);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == SIZE_TWO);
    auto out0 = ctx->ooperandInplaceDataViewList->at(0);
    auto out1 = ctx->ooperandInplaceDataViewList->at(1);
    auto src0 = ctx->ioperandDataViewList->at(0);
    auto src1 = ctx->ioperandDataViewList->at(1);
    calc::DeInterleave(out0, out1, src0, src1);
}
REGISTER_CALC_OP(OP_DEINTERLEAVE, Opcode::OP_DEINTERLEAVE, ExecuteOpDeInterleave);

void ExecuteOpDeInterleaveSingle(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == SIZE_TWO);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto out0 = ctx->ooperandInplaceDataViewList->at(0);
    auto out1 = ctx->ooperandInplaceDataViewList->at(1);
    auto src = ctx->ioperandDataViewList->at(0);
    calc::DeInterleaveSingle(out0, out1, src);
}
REGISTER_CALC_OP(OP_DEINTERLEAVE_SINGLE, Opcode::OP_DEINTERLEAVE_SINGLE, ExecuteOpDeInterleaveSingle);

void ExecuteOpLog1p(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() <= SIZE_TWO);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto& ret = ctx->ooperandInplaceDataViewList->at(0);
    auto& iop = ctx->ioperandDataViewList->at(0);
    calc::Log1p(ret, iop);
}
REGISTER_CALC_OP(OP_LOG1P, Opcode::OP_LOG1P, ExecuteOpLog1p);

void ExecuteOpCompare(ExecuteOperationContext* ctx)
{
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto iop_self = ctx->ioperandDataViewList->at(0);
    auto iop_other = ctx->ioperandDataViewList->at(1);
    auto operation = static_cast<CmpOperationType>(ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "cmp_operation"));
    auto mode = static_cast<CmpModeType>(ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "cmp_mode"));
    calc::Compare(oop, iop_self, iop_other, operation, mode);
}
REGISTER_CALC_OP(OP_CMP, Opcode::OP_CMP, ExecuteOpCompare);

void ExecuteOpCmps(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto iop_self = ctx->ioperandDataViewList->at(0);
    auto element = Element(DT_FP32, 0.0f);
    ctx->op->GetAttr(OpAttributeKey::scalar, element);

    auto operation = static_cast<CmpOperationType>(ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "cmp_operation"));
    auto mode = static_cast<CmpModeType>(ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "cmp_mode"));
    calc::Cmps(oop, iop_self, element, operation, mode);
}
REGISTER_CALC_OP(OP_CMPS, Opcode::OP_CMPS, ExecuteOpCmps);

void ExecuteOpHypot(ExecuteOperationContext* ctx)
{
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto iop_self = ctx->ioperandDataViewList->at(0);
    auto iop_other = ctx->ioperandDataViewList->at(1);
    calc::Hypot(oop, iop_self, iop_other);
}
REGISTER_CALC_OP(OP_HYPOT, Opcode::OP_HYPOT, ExecuteOpHypot);

void ExecuteOpPReLU(ExecuteOperationContext* ctx)
{
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto iop_self = ctx->ioperandDataViewList->at(0);
    auto iop_weight = ctx->ioperandDataViewList->at(1);
    calc::PReLU(oop, iop_self, iop_weight);
}
REGISTER_CALC_OP(OP_PRELU, Opcode::OP_PRELU, ExecuteOpPReLU);

void ExecuteOpExtract(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto src = ctx->ioperandDataViewList->at(0);
    auto maskMode = ctx->op->GetIntAttribute("op_attr_makeMode");
    int descending = ctx->op->GetIntAttribute("op_attr_order");
    calc::Extract(oop, src, maskMode, descending);
}
REGISTER_CALC_OP(OP_EXTRACT, Opcode::OP_EXTRACT, ExecuteOpExtract);
REGISTER_CALC_OP(OP_EXTRACT_SINGLE, Opcode::OP_EXTRACT_SINGLE, ExecuteOpExtract);

void ExecuteOpGather(ExecuteOperationContext* ctx)
{
    auto output = ctx->ooperandInplaceDataViewList->at(0);
    auto parmas = ctx->ioperandDataViewList->at(0);
    auto indices = ctx->ioperandDataViewList->at(1);
    int axis = ctx->op->GetIntAttribute("op_attr_axis");
    calc::Gather(output, parmas, indices, axis);
}
REGISTER_CALC_OP(OP_GATHER, Opcode::OP_GATHER, ExecuteOpGather);
void ExecuteOpGatherINUB(ExecuteOperationContext* ctx)
{
    auto output = ctx->ooperandInplaceDataViewList->at(0);
    auto parmas = ctx->ioperandDataViewList->at(0);
    auto indices = ctx->ioperandDataViewList->at(1);
    auto pageTable = ctx->ioperandDataViewList->at(2);
    int blocksize = ctx->op->GetIntAttribute(OpAttributeKey::blockSize);
    calc::GatherINUB(output, parmas, indices, pageTable, blocksize, GATHER_IGNORE_AXIS);
}
REGISTER_CALC_OP(OP_GATHER_IN_UB, Opcode::OP_GATHER_IN_UB, ExecuteOpGatherINUB);

void ExecuteOpIndexAdd(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() <= SIZE_TWO);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == SIZE_THREE);
    auto& ret = ctx->ooperandInplaceDataViewList->at(0);
    auto& self = ctx->ioperandDataViewList->at(0);
    auto& src = ctx->ioperandDataViewList->at(1);
    auto& indices = ctx->ioperandDataViewList->at(2);
    auto alpha = Element(DT_FP32, 1.0);
    if (ctx->op->HasAttribute(OpAttributeKey::scalar)) {
        alpha = ctx->op->GetElementAttribute(OpAttributeKey::scalar);
    }
    int axis = ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "axis");
    if (ctx->op->GetOpcode() == Opcode::OP_INDEX_ADD) {
        calc::IndexAdd(self, self, src, indices, axis, alpha);
    } else {
        calc::IndexAdd(ret, self, src, indices, axis, alpha);
    }
}
REGISTER_CALC_OP(OP_INDEX_ADD_UB, Opcode::OP_INDEX_ADD_UB, ExecuteOpIndexAdd);
REGISTER_CALC_OP(OP_INDEX_ADD, Opcode::OP_INDEX_ADD, ExecuteOpIndexAdd);

void ExecuteOpTri(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto& output = ctx->ooperandInplaceDataViewList->at(0);
    auto& input = ctx->ioperandDataViewList->at(0);

    // dynScalar 可能为 RUNTIME_COA_GET_PARAM(idx)，需经 callop linearArgList 解析
    SymbolicScalar diaSym = ctx->op->GetSymbolicScalarAttribute(OpAttributeKey::dynScalar);
    std::vector<OpImmediate> diagonalImmList = {OpImmediate::Specified(diaSym)};
    int diagonal = static_cast<int>(ctx->opInter->EvaluateOpImmediate(ctx->frame, diagonalImmList)[0]);
    INTERPRETER_LOGI("diagonal: %d", diagonal);
    bool isUpper = ctx->op->GetBoolAttribute(OpAttributeKey::isUpper);
    isUpper ? calc::TriU(output, input, diagonal) : calc::TriL(output, input, diagonal);
}
REGISTER_CALC_OP(OP_TRIUL, Opcode::OP_TRIUL, ExecuteOpTri);

void ExecuteOpCumSum(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto& output = ctx->ooperandInplaceDataViewList->at(0);
    auto& input = ctx->ioperandDataViewList->at(0);

    int axis = ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "axis");
    calc::CumSum(output, input, axis);
}
REGISTER_CALC_OP(OP_CUM_SUM, Opcode::OP_CUM_SUM, ExecuteOpCumSum);

void ExecuteOpCumProd(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto& output = ctx->ooperandInplaceDataViewList->at(0);
    auto& input = ctx->ioperandDataViewList->at(0);

    int axis = ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "axis");
    calc::CumProd(output, input, axis);
}
REGISTER_CALC_OP(OP_CUM_PROD, Opcode::OP_CUM_PROD, ExecuteOpCumProd);

void ExecuteOpIndexPut(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() <= SIZE_SIX);
    auto out = ctx->ooperandInplaceDataViewList->at(0);
    auto self = ctx->ioperandDataViewList->at(0);
    auto values = ctx->ioperandDataViewList->at(1);
    std::vector<LogicalTensorDataPtr> indices;
    for (int i = SIZE_TWO; i < static_cast<int>(ctx->ioperandDataViewList->size()); i++) {
        auto indicesTemp = ctx->ioperandDataViewList->at(i);
        indices.push_back(indicesTemp);
    }
    bool accumulate = ctx->op->GetBoolAttribute(OpAttributeKey::accumulate);
    calc::IndexPut(out, self, indices, values, accumulate);
}
REGISTER_CALC_OP(OP_INDEX_PUT, Opcode::OP_INDEX_PUT, ExecuteOpIndexPut);

void ExecuteOpMrgSort(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto src = ctx->ioperandDataViewList->at(0);
    auto topk_axis = ctx->op->GetIntAttribute("op_attr_axis");
    auto kValue = ctx->op->GetIntAttribute("op_attr_kvalue");
    calc::MrgSort(oop, src, topk_axis, kValue);
}
REGISTER_CALC_OP(OP_MRGSORT, Opcode::OP_MRGSORT, ExecuteOpMrgSort);

void ExecuteOpTwoTileMrgSort(ExecuteOperationContext* ctx)
{
    auto src = ctx->ioperandDataViewList->at(0);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    calc::TwoTileMrgSort(oop, src);
}
REGISTER_CALC_OP(OP_TWOTILEMRGSORT, Opcode::OP_TWOTILEMRGSORT, ExecuteOpTwoTileMrgSort);

void ExecuteOpSort(ExecuteOperationContext* ctx)
{
    auto src = ctx->ioperandDataViewList->at(0);
    auto value = ctx->ooperandInplaceDataViewList->at(0);
    auto index = ctx->ooperandInplaceDataViewList->at(1);
    auto axis = ctx->op->GetIntAttribute("op_attr_axis");
    int descending = ctx->op->GetIntAttribute("op_attr_order");
    calc::Sort(value, index, src, axis, descending);
}
REGISTER_CALC_OP(OP_SORT_UB, Opcode::OP_SORT_UB, ExecuteOpSort);

void ExecuteOpTopK(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto outValue = ctx->ooperandInplaceDataViewList->at(0);
    auto outIndex = ctx->ooperandInplaceDataViewList->at(1);
    auto src = ctx->ioperandDataViewList->at(0);
    auto topk_axis = ctx->op->GetIntAttribute("op_attr_axis");
    auto kValue = ctx->op->GetIntAttribute("op_attr_kvalue");
    int descending = ctx->op->GetIntAttribute("op_attr_order");
    calc::TopK(outValue, outIndex, src, kValue, topk_axis, descending);
}
REGISTER_CALC_OP(OP_TOPK, Opcode::OP_TOPK, ExecuteOpTopK);
REGISTER_CALC_OP(OP_RADIX_SELECT, Opcode::OP_RADIX_SELECT, ExecuteOpTopK);

void ExecuteOpQuantizeSym(ExecuteOperationContext* ctx)
{
    auto& ret = ctx->ooperandInplaceDataViewList->at(0);
    auto& input = ctx->ioperandDataViewList->at(0);
    auto& scale = ctx->ioperandDataViewList->at(1);
    calc::Quantize(ret, input, scale, nullptr);
}
REGISTER_CALC_OP(OP_QUANTIZE_SYM, Opcode::OP_QUANTIZE_SYM, ExecuteOpQuantizeSym);

void ExecuteOpQuantizeAsym(ExecuteOperationContext* ctx)
{
    auto& ret = ctx->ooperandInplaceDataViewList->at(0);
    auto& input = ctx->ioperandDataViewList->at(0);
    auto& scale = ctx->ioperandDataViewList->at(1);
    auto& zeropoints = ctx->ioperandDataViewList->at(2);
    calc::Quantize(ret, input, scale, zeropoints);
}
REGISTER_CALC_OP(OP_QUANTIZE_ASYM, Opcode::OP_QUANTIZE_ASYM, ExecuteOpQuantizeAsym);

void ExecuteOpDequantize(ExecuteOperationContext* ctx)
{
    auto& ret = ctx->ooperandInplaceDataViewList->at(0);
    auto& input = ctx->ioperandDataViewList->at(0);
    auto& scale = ctx->ioperandDataViewList->at(1);
    auto& zeropoints = ctx->ioperandDataViewList->at(2);
    calc::Dequantize(ret, input, scale, zeropoints);
}
REGISTER_CALC_OP(OP_DEQUANTIZE, Opcode::OP_DEQUANTIZE, ExecuteOpDequantize);

void ExecuteOpBitSort(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto src = ctx->ioperandDataViewList->at(0);
    auto topk_axis = ctx->op->GetIntAttribute("op_attr_axis");
    int descending = ctx->op->GetIntAttribute("op_attr_order");
    int offset = ctx->op->GetIntAttribute("op_attr_offset");
    calc::BitSort(oop, src, topk_axis, descending, offset);
}
REGISTER_CALC_OP(OP_BITSORT, Opcode::OP_BITSORT, ExecuteOpBitSort);

void ExecuteOpTiledMrgSort(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == SIZE_FOUR);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto src1 = ctx->ioperandDataViewList->at(0);
    auto src2 = ctx->ioperandDataViewList->at(1);
    auto src3 = ctx->ioperandDataViewList->at(2);
    auto src4 = ctx->ioperandDataViewList->at(3);
    auto validBit = ctx->op->GetIntAttribute("op_attr_validBit");
    auto kvalue = ctx->op->GetIntAttribute("op_attr_kvalue");
    calc::TiledMrgSort(oop, src1, src2, src3, src4, validBit, kvalue);
}
REGISTER_CALC_OP(OP_TILEDMRGSORT, Opcode::OP_TILEDMRGSORT, ExecuteOpTiledMrgSort);

void ExecuteOpTopkSort(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH,
           ctx->ooperandInplaceDataViewList->size() == 0x2); // value + temp

    auto iop = ctx->ioperandDataViewList->at(0);
    auto oop_value = ctx->ooperandInplaceDataViewList->at(0);
    auto oop_temp = ctx->ooperandInplaceDataViewList->at(1);

    int startIndex = ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "start_index");

    calc::TopkSort(oop_value, oop_temp, iop, startIndex);
}
REGISTER_CALC_OP(OP_TOPK_SORT, Opcode::OP_TOPK_SORT, ExecuteOpTopkSort);

void ExecuteOpTopkMerge(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);

    auto iop = ctx->ioperandDataViewList->at(0);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);

    int mergeSize = ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "merge_size");

    calc::TopkMerge(oop, iop, mergeSize);
}
REGISTER_CALC_OP(OP_TOPK_MERGE, Opcode::OP_TOPK_MERGE, ExecuteOpTopkMerge);

void ExecuteOpTopkExtract(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);

    auto iop = ctx->ioperandDataViewList->at(0);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);

    int k = ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "k");
    bool isIndex = static_cast<bool>(ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "is_index"));

    calc::TopkExtract(oop, iop, k, isIndex);
}
REGISTER_CALC_OP(OP_TOPK_EXTRACT, Opcode::OP_TOPK_EXTRACT, ExecuteOpTopkExtract);

void ExecuteOpReduceAcc(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    auto& ret = ctx->ooperandInplaceDataViewList->at(0);
    calc::ReduceAcc(ret, *ctx->ioperandDataViewList);
}
REGISTER_CALC_OP(OP_REDUCE_ACC, Opcode::OP_REDUCE_ACC, ExecuteOpReduceAcc);

template <Opcode opcode>
void ExecuteOpBinaryScalar(ExecuteOperationContext* ctx)
{
    if (opcode == Opcode::OP_BITWISEXOR || opcode == Opcode::OP_REMRS || opcode == Opcode::OP_FLOORDIVS ||
        opcode == Opcode::OP_REMS || opcode == Opcode::OP_POWS || opcode == Opcode::OP_GCDS) {
        ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() <= SIZE_TWO);
    } else {
        ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    }
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto& ret = ctx->ooperandInplaceDataViewList->at(0);
    auto& lhs = ctx->ioperandDataViewList->at(0);
    auto element = Element(DT_FP32, 0.0f);
    ctx->op->GetAttr(OpAttributeKey::scalar, element);
    bool reverse = ctx->op->GetBoolAttribute(OP_ATTR_PREFIX + "reverseOperand");

    switch (opcode) {
        case Opcode::OP_ADDS:
            calc::AddS(ret, lhs, element);
            break;
        case Opcode::OP_SUBS:
            calc::SubS(ret, lhs, element, reverse);
            break;
        case Opcode::OP_MULS:
            calc::MulS(ret, lhs, element);
            break;
        case Opcode::OP_MAXS:
            calc::MaxS(ret, lhs, element);
            break;
        case Opcode::OP_MINS:
            calc::MinS(ret, lhs, element);
            break;
        case Opcode::OP_DIVS:
            calc::DivS(ret, lhs, element, reverse);
            break;
        case Opcode::OP_FLOORDIVS:
            calc::FloorDivS(ret, lhs, element, reverse);
            break;
        case Opcode::OP_REMS:
            calc::RemainderS(ret, lhs, element, reverse);
            break;
        case Opcode::OP_POWS:
            calc::PowS(ret, lhs, element);
            break;
        case Opcode::OP_REMRS:
            calc::RemainderRS(ret, lhs, element, reverse);
            break;
        case Opcode::OP_S_MAXS:
            calc::MaxS(ret, lhs, element);
            break;
        case Opcode::OP_S_MINS:
            calc::MinS(ret, lhs, element);
            break;
        case Opcode::OP_LRELU:
            calc::LReLU(ret, lhs, element);
            break;
        case Opcode::OP_BITWISEANDS:
            calc::BitwiseAndS(ret, lhs, element);
            break;
        case Opcode::OP_BITWISEORS:
            calc::BitwiseOrS(ret, lhs, element);
            break;
        case Opcode::OP_BITWISEXORS:
            calc::BitwiseXorS(ret, lhs, element);
            break;
        case Opcode::OP_GCDS:
            calc::GcdS(ret, lhs, element);
            break;
        default:
            ASSERT(ExecuteOperationScene::UNSUPPORTED_OPCODE, false);
    }
}
REGISTER_CALC_OP(OP_ADDS, Opcode::OP_ADDS, ExecuteOpBinaryScalar<Opcode::OP_ADDS>);
REGISTER_CALC_OP(OP_SUBS, Opcode::OP_SUBS, ExecuteOpBinaryScalar<Opcode::OP_SUBS>);
REGISTER_CALC_OP(OP_MULS, Opcode::OP_MULS, ExecuteOpBinaryScalar<Opcode::OP_MULS>);
REGISTER_CALC_OP(OP_DIVS, Opcode::OP_DIVS, ExecuteOpBinaryScalar<Opcode::OP_DIVS>);
REGISTER_CALC_OP(OP_FLOORDIVS, Opcode::OP_FLOORDIVS, ExecuteOpBinaryScalar<Opcode::OP_FLOORDIVS>);
REGISTER_CALC_OP(OP_MAXS, Opcode::OP_MAXS, ExecuteOpBinaryScalar<Opcode::OP_MAXS>);
REGISTER_CALC_OP(OP_MINS, Opcode::OP_MINS, ExecuteOpBinaryScalar<Opcode::OP_MINS>);
REGISTER_CALC_OP(OP_LRELU, Opcode::OP_LRELU, ExecuteOpBinaryScalar<Opcode::OP_LRELU>);
REGISTER_CALC_OP(OP_BITWISEANDS, Opcode::OP_BITWISEANDS, ExecuteOpBinaryScalar<Opcode::OP_BITWISEANDS>);
REGISTER_CALC_OP(OP_BITWISEORS, Opcode::OP_BITWISEORS, ExecuteOpBinaryScalar<Opcode::OP_BITWISEORS>);
REGISTER_CALC_OP(OP_BITWISEXORS, Opcode::OP_BITWISEXORS, ExecuteOpBinaryScalar<Opcode::OP_BITWISEXORS>);
REGISTER_CALC_OP(OP_GCDS, Opcode::OP_GCDS, ExecuteOpBinaryScalar<Opcode::OP_GCDS>);
REGISTER_CALC_OP(OP_REMS, Opcode::OP_REMS, ExecuteOpBinaryScalar<Opcode::OP_REMS>);
REGISTER_CALC_OP(OP_REMRS, Opcode::OP_REMRS, ExecuteOpBinaryScalar<Opcode::OP_REMRS>);
REGISTER_CALC_OP(OP_POWS, Opcode::OP_POWS, ExecuteOpBinaryScalar<Opcode::OP_POWS>);
REGISTER_CALC_OP(OP_S_ADDS, Opcode::OP_S_ADDS, ExecuteOpBinaryScalar<Opcode::OP_ADDS>);
REGISTER_CALC_OP(OP_S_SUBS, Opcode::OP_S_SUBS, ExecuteOpBinaryScalar<Opcode::OP_SUBS>);
REGISTER_CALC_OP(OP_S_MULS, Opcode::OP_S_MULS, ExecuteOpBinaryScalar<Opcode::OP_MULS>);
REGISTER_CALC_OP(OP_S_DIVS, Opcode::OP_S_DIVS, ExecuteOpBinaryScalar<Opcode::OP_DIVS>);
REGISTER_CALC_OP(OP_S_MAXS, Opcode::OP_S_MAXS, ExecuteOpBinaryScalar<Opcode::OP_S_MAXS>);
REGISTER_CALC_OP(OP_S_MINS, Opcode::OP_S_MINS, ExecuteOpBinaryScalar<Opcode::OP_S_MINS>);

void ExecuteOpGatherElement(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() <= SIZE_TWO);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == SIZE_TWO);
    auto& ret = ctx->ooperandInplaceDataViewList->at(0);
    auto& params = ctx->ioperandDataViewList->at(0);
    auto& indices = ctx->ioperandDataViewList->at(1);
    int axis = ctx->op->GetIntAttribute("op_attr_axis");
    calc::GatherElements(ret, params, indices, axis);
}
REGISTER_CALC_OP(OP_GATHER_ELEMENT, Opcode::OP_GATHER_ELEMENT, ExecuteOpGatherElement);

void ExecuteOpGatherMask(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto& ret = ctx->ooperandInplaceDataViewList->at(0);
    auto& self = ctx->ioperandDataViewList->at(0);
    int patternMode = ctx->op->GetIntAttribute("op_attr_patternMode");
    calc::GatherMask(ret, self, patternMode);
}
REGISTER_CALC_OP(OP_GATHER_MASK, Opcode::OP_GATHER_MASK, ExecuteOpGatherMask);
REGISTER_CALC_OP(OP_GATHER_MASK_BUILDIN, Opcode::OP_GATHER_MASK_BUILDIN, ExecuteOpGatherMask);

template <Opcode opcode>
void ExecuteOpBitwiseShift(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() <= SIZE_TWO);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == SIZE_TWO);
    auto ret = ctx->ooperandInplaceDataViewList->at(0);
    auto lhs = ctx->ioperandDataViewList->at(0);
    auto rhs = ctx->ioperandDataViewList->at(1);

    switch (opcode) {
        case Opcode::OP_BITWISERIGHTSHIFT:
            calc::BitwiseRightShift(ret, lhs, rhs);
            break;
        case Opcode::OP_BITWISELEFTSHIFT:
            calc::BitwiseLeftShift(ret, lhs, rhs);
            break;
        default:
            ASSERT(ExecuteOperationScene::UNSUPPORTED_OPCODE, false);
    }
}

template <Opcode opcode>
void ExecuteOpBitwiseShiftScalar(ExecuteOperationContext* ctx)
{
    if (opcode == Opcode::OP_SBITWISERIGHTSHIFT || opcode == Opcode::OP_SBITWISELEFTSHIFT) {
        ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() <= SIZE_TWO);
    } else {
        ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    }
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto ret = ctx->ooperandInplaceDataViewList->at(0);
    auto lhs = ctx->ioperandDataViewList->at(0);
    auto element = Element(DT_INT32, 0);
    ctx->op->GetAttr(OpAttributeKey::scalar, element);

    switch (opcode) {
        case Opcode::OP_BITWISERIGHTSHIFTS:
            calc::BitwiseRightShiftS(ret, lhs, element);
            break;
        case Opcode::OP_BITWISELEFTSHIFTS:
            calc::BitwiseLeftShiftS(ret, lhs, element);
            break;
        case Opcode::OP_SBITWISERIGHTSHIFT:
            calc::SBitwiseRightShift(ret, element, lhs);
            break;
        case Opcode::OP_SBITWISELEFTSHIFT:
            calc::SBitwiseLeftShift(ret, element, lhs);
            break;
        default:
            ASSERT(ExecuteOperationScene::UNSUPPORTED_OPCODE, false);
    }
}
REGISTER_CALC_OP(OP_BITWISERIGHTSHIFT, Opcode::OP_BITWISERIGHTSHIFT,
                 ExecuteOpBitwiseShift<Opcode::OP_BITWISERIGHTSHIFT>);
REGISTER_CALC_OP(OP_BITWISELEFTSHIFT, Opcode::OP_BITWISELEFTSHIFT, ExecuteOpBitwiseShift<Opcode::OP_BITWISELEFTSHIFT>);
REGISTER_CALC_OP(OP_BITWISERIGHTSHIFTS, Opcode::OP_BITWISERIGHTSHIFTS,
                 ExecuteOpBitwiseShiftScalar<Opcode::OP_BITWISERIGHTSHIFTS>);
REGISTER_CALC_OP(OP_BITWISELEFTSHIFTS, Opcode::OP_BITWISELEFTSHIFTS,
                 ExecuteOpBitwiseShiftScalar<Opcode::OP_BITWISELEFTSHIFTS>);
REGISTER_CALC_OP(OP_SBITWISERIGHTSHIFT, Opcode::OP_SBITWISERIGHTSHIFT,
                 ExecuteOpBitwiseShiftScalar<Opcode::OP_SBITWISERIGHTSHIFT>);
REGISTER_CALC_OP(OP_SBITWISELEFTSHIFT, Opcode::OP_SBITWISELEFTSHIFT,
                 ExecuteOpBitwiseShiftScalar<Opcode::OP_SBITWISELEFTSHIFT>);
} // namespace npu::tile_fwk
