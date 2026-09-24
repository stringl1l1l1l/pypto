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
 * \file codegen_vector.cpp
 * \brief
 */

#include "interface/tensor/logical_tensor.h"
#include "codegen_op_npu.h"
#include "securec.h"
#include "codegen/utils/codegen_utils.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include <string>

namespace npu::tile_fwk {
std::string CodeGenOpNPU::GenCastOp() const
{
    if (isSupportTileTensor) {
        return PrintCastTileTensor();
    }

    bool hasTmpBuffer = (operandCnt == NUM3);
    int srcIdx = hasTmpBuffer ? ID2 : ID1;

    std::string s0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[srcIdx]);
    std::string dVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);

    std::vector srcShape = rawShape[srcIdx];
    CODEGEN_LOGI("genCastOp %s, srcShape is %s", tileOpName.c_str(), IntVecToStr(srcShape).c_str());

    std::vector dstShape = rawShape[ID0];
    CODEGEN_LOGI("genCastOp %s, dstShape is %s", tileOpName.c_str(), IntVecToStr(dstShape).c_str());

    std::string srcDtypeStr = DataType2CCEStr(operandDtype[srcIdx]);
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);

    AppendLocalBufVarOffsetInOrder(dVar, s0Var);
    std::vector<int64_t> os = NormalizeShape(shape[0], SHAPE_DIM4);
    std::vector<int64_t> ss = NormalizeShape(rawShape[srcIdx], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);

    char buffer[BUFFER_SIZE_1024] = "CG_ERROR";
    int ret = 0;
    if (isDynamicFunction) {
        return PrintCastDynamicUnaligned({s0Var, dVar, srcDtypeStr, dstDtypeStr});
    }
    int64_t modeEnum = 0;
    GetOpAttr(OP_ATTR_PREFIX + "mode", modeEnum);
    ret = sprintf_s(
        buffer, sizeof(buffer),
        "%s_<%s, %s, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %lld>((__ubuf__ %s *)%s,  (__ubuf__ %s *)%s);\n",
        tileOpName.c_str(), dstDtypeStr.c_str(), srcDtypeStr.c_str(), os[0], os[1], os[2], os[3], ds[1], ds[2], ds[3],
        ss[1], ss[2], ss[3], modeEnum, dstDtypeStr.c_str(), dVar.c_str(), srcDtypeStr.c_str(), s0Var.c_str());
    ASSERT(GenCodeErr::PRINT_FAILED, ret >= 0) << "GenCastOp sprintf_s failed " << ret;
    std::string ostring(buffer);
    return ostring;
}

std::string CodeGenOpNPU::PrintDupOpDynUnaligned(const PrintDupOpParam& param) const
{
    const std::string& dstDtypeStr = param.dstDtypeStr;
    const std::string& dVar = param.dVar;
    const std::string& dupV = param.dupV;
    // dst origin shape
    std::vector<int64_t> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    for (int i = 1; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    paramList.clear();

    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    paramList.insert(paramList.end(), {dst, dupV});
    auto dynDstShape = dynamicValidShape[0];
    FillVecWithDummyInHead<SymbolicScalar>(dynDstShape, SHAPE_DIM4 - dynDstShape.size(), 1);
    FillParamWithFullInput(paramList, dynDstShape);

    auto startOffset = GetOperandStartOffset(0);
    if (!startOffset.ConcreteValid() || startOffset.Concrete() != 0) {
        paramList.emplace_back(startOffset.Dump());
    }

    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName.c_str() << "_<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpNPU::PrintDupOpStatic(const PrintDupOpParam& param) const
{
    const std::string& dstDtypeStr = param.dstDtypeStr;
    std::string dVar = param.dVar;
    const std::string& dupV = param.dupV;
    AppendLocalBufVarOffsetInOrder(dVar);
    // dst origin shape
    std::vector<int64_t> dos = NormalizeShape(shape[0], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[0], SHAPE_DIM4);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    for (auto oriShape : dos) {
        paramList.emplace_back(std::to_string(oriShape));
    }
    for (int i = 1; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    paramList.insert(paramList.end(), {dst, dupV});
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName.c_str() << "_<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpNPU::PrintDupTileTensor(const PrintDupOpParam& param) const
{
    const std::string& dupV = param.dupV;
    const std::string& dstDtypeStr = param.dstDtypeStr;
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));

    std::ostringstream oss;
    oss << tileOpName;
    oss << WrapParamByAngleBrackets({dstDtypeStr});
    oss << WrapParamByParentheses({dstTensor, dupV});
    oss << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::PrintDupOp(const PrintDupOpParam& param) const
{
    if (isSupportTileTensor) {
        return PrintDupTileTensor(param);
    }

    if (isDynamicFunction) {
        return PrintDupOpDynUnaligned(param);
    }
    return PrintDupOpStatic(param);
}

std::string CodeGenOpNPU::GenDupOp() const
{
    std::string dVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);

    std::string dupV;
    if (extSymbolicScalar.IsValid()) {
        dupV = SymbolicExpressionTable::BuildExpression(extSymbolicScalar);
        return PrintDupOp({dVar, dstDtypeStr, dupV});
    }

    ASSERT(OperErr::ATTRIBUTE_INVALID, opAttrs.count(OpAttributeKey::scalar))
        << "OpAttributeKey::scalar is invalid, dstDtypeStr: " << dstDtypeStr;

    if (dstDtypeStr == "float" || dstDtypeStr == "half" || dstDtypeStr == "bfloat16_t") {
        ASSERT(OperErr::ATTRIBUTE_INVALID, extOperandVal.IsFloat())
            << "SCALAR attribute must be float value, data type: " << DataType2String(extOperandVal.GetDataType());
        dupV = FormatFloat(extOperandVal.Cast<float>(), operandDtype[ToUnderlying(MISOIdx::DST_IDX)]);
    } else if (dstDtypeStr == "bool" || dstDtypeStr == "int8_t" || dstDtypeStr == "int16_t" ||
               dstDtypeStr == "int32_t" || dstDtypeStr == "int64_t") {
        ASSERT(OperErr::ATTRIBUTE_INVALID, extOperandVal.IsSigned())
            << "SCALAR attribute has to be int value, data type: " << DataType2String(extOperandVal.GetDataType());
        dupV = std::to_string(extOperandVal.Cast<int64_t>());
    } else if (dstDtypeStr == "uint8_t" || dstDtypeStr == "uint16_t" || dstDtypeStr == "uint32_t") {
        ASSERT(OperErr::ATTRIBUTE_INVALID, extOperandVal.IsUnsigned())
            << "SCALAR attribute has to be uint value, data type: " << DataType2String(extOperandVal.GetDataType());
        dupV = std::to_string(extOperandVal.Cast<uint64_t>());
    } else {
        ASSERT(OperErr::ATTRIBUTE_INVALID, false) << "unsupported type, dstDtypeStr: " << dstDtypeStr;
    }

    return PrintDupOp({dVar, dstDtypeStr, dupV});
}

std::string CodeGenOpNPU::PrintPermuteLayout() const
{
    size_t srcDim = rawShape[ToUnderlying(MISOIdx::SRC0_IDX)].size();
    auto srcOffsetSymbol = GenGetParamMacroPacked(ToUnderlying(MISOIdx::SRC0_IDX), srcDim, PREFIX_STR_OFFSET);
    std::string coordCpSrc = WrapParamByParentheses(srcOffsetSymbol);
    std::string coord4Src = PrintCoord(srcDim, coordCpSrc);

    auto permAttr = opAttrs.at(OpAttributeKey::perm);
    const auto& permVec = AnyCast<std::vector<int64_t>>(permAttr);
    std::vector<int> axes(MAX_DIM + 1, -1);
    for (size_t i = 0; i < permVec.size() && i < MAX_DIM; ++i) {
        axes[i] = static_cast<int>(permVec[i]);
    }
    axes[MAX_DIM] = permVec.size();
    std::vector<std::string> tileOpParamList = GetTileOpParamsByOrder();
    tileOpParamList.emplace_back(coord4Src);
    std::ostringstream oss;
    oss << tileOpName << WrapParamByAngleBrackets(axes) << WrapParamByParentheses(tileOpParamList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenPermuteOp() const { return PrintPermuteLayout(); }

std::string CodeGenOpNPU::GenTransData() const
{
    std::vector<SymbolicScalar> transDataAttr;
    GetOpAttr(OpAttributeKey::transDataOffset, transDataAttr);
    std::vector<std::string> tileParamsStr = {};
    for (auto tileParam : transDataAttr) {
        tileParamsStr.emplace_back("(int)(" + SymbolicExpressionTable::BuildExpression(tileParam) + ")");
    }

    if (opCode == Opcode::OP_NCHW2NC1HWC0 || opCode == Opcode::OP_NC1HWC02NCHW || opCode == Opcode::OP_NCHW2Fractal_Z ||
        opCode == Opcode::OP_NCDHW2NDC1HWC0 || opCode == Opcode::OP_NCDHW2FRACTAL_Z_3D ||
        opCode == Opcode::OP_NDC1HWC02NCDHW || opCode == Opcode::OP_FractalZ2NCHW ||
        opCode == Opcode::OP_FractalZ3D2NCDHW) {
        return PrintTransDataLayoutTmp();
    }

    return PrintTransDataLayout(tileParamsStr);
}

std::string CodeGenOpNPU::PrintTransDataLayoutTmp() const
{
    std::string dstTensor = QueryTileTensorNameByIdx(ID0);
    std::string tmpTensor = QueryTileTensorNameByIdx(ID1);
    std::string inputTensor = QueryTileTensorNameByIdx(ID2);
    std::vector<std::string> paramList = {dstTensor, tmpTensor, inputTensor};
    std::ostringstream oss;
    oss << tileOpName << WrapParamByParentheses(paramList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::PrintTransDataLayout(const std::vector<std::string>& param) const
{
    std::string dstTensor = QueryTileTensorNameByIdx(ID0);
    std::vector<std::string> gmOffsetExpr = GetGmOffsetForTileTensor(ID0);
    std::string coordCp = WrapParamByParentheses(gmOffsetExpr);
    std::string coord = PrintCoord(rawShape[ID0].size(), coordCp);
    std::string tmpTensor = QueryTileTensorNameByIdx(ID1);
    std::string inputTensor = QueryTileTensorNameByIdx(ID2);
    std::vector<std::string> paramList = {dstTensor, coord, tmpTensor, inputTensor};
    std::vector<std::string> templateParam = {};
    static const std::unordered_map<Opcode, unsigned> opParamPos{
        {Opcode::OP_NCHW2NC1HWC0, 4}, {Opcode::OP_NCHW2Fractal_Z, 6}, {Opcode::OP_NCDHW2NDC1HWC0, 5},
        {Opcode::OP_NC1HWC02NCHW, 8}, {Opcode::OP_NDC1HWC02NCDHW, 9}, {Opcode::OP_NCDHW2FRACTAL_Z_3D, 7}};
    auto iter = opParamPos.find(opCode);
    ASSERT(OperErr::ATTRIBUTE_INVALID, iter != opParamPos.end()) << "This transData conversion is not supported.";
    unsigned pos = iter->second;
    paramList.insert(paramList.end(), param.begin(), param.begin() + pos);
    templateParam = std::vector<std::string>(param.begin() + pos, param.end());
    std::ostringstream oss;
    oss << tileOpName << WrapParamByAngleBrackets(templateParam) << WrapParamByParentheses(paramList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenTransposeDataMove() const
{
    bool isCopyLocalToGM = opCode == Opcode::OP_TRANSPOSE_MOVEOUT;
    unsigned gmIdx = isCopyLocalToGM ? 0 : 1;
    unsigned localIdx = isCopyLocalToGM ? 1 : 0;

    std::string localVar = sm->QueryVarNameByTensorMagic(operandWithMagic[localIdx]);
    std::string gmVar = GenGmParamVar(gmIdx);

    std::vector<int64_t> srcShape = rawShape[localIdx];
    CODEGEN_LOGI("GenTransposeDataMove: srcShape is %s", IntVecToStr(srcShape).c_str());
    std::vector<int64_t> gmShape = rawShape[gmIdx];
    CODEGEN_LOGI("GenTransposeDataMove: gmShape is %s", IntVecToStr(gmShape).c_str());

    AppendLocalBufferVarOffset({{gmIdx, std::ref(gmVar)}, {localIdx, std::ref(localVar)}});

    std::string localDtypeStr = CodeGenDataTypeStr(operandDtype[localIdx]);
    std::string gmDtypeStr = CodeGenDataTypeStr(operandDtype[gmIdx]);
    return PrintTransposeDataMove({gmIdx, localIdx, localVar, gmShape, localDtypeStr, gmDtypeStr});
}

std::string CodeGenOpNPU::PrintTransposeDataMove(const PrintTransposeDataMoveParam& param) const
{
    if (isSupportTileTensor) {
        return PrintTransposeDataMoveLayout(param);
    }
    if (isSupportDynamicAligned) {
        return PrintTransposeDataMoveDynamic(param);
    } else if (isDynamicFunction) {
        return PrintTransposeDataMoveDynamicUnaligned(param);
    }
    return PrintTransposeDataMoveStatic(param);
}

std::string CodeGenOpNPU::PrintTransposeDataMoveLayout(const PrintTransposeDataMoveParam& param) const
{
    std::string gmVarName = GenGmParamVar(param.gmIdx);
    std::vector<int64_t> transposeAxis = AnyCast<std::vector<int64_t>>(opAttrs.at(OP_ATTR_PREFIX + "shape"));
    int correctionAxis = SHAPE_DIM5 - shape[param.localIdx].size();
    std::vector<std::string> uselessVector0;
    std::vector<std::string> uselessVector1;
    std::vector<std::string> uselessVector2;
    std::vector<std::string> gmOffsetExpr = GetGmOffsetForTileTensor(param.gmIdx);
    std::string coordCp = WrapParamByParentheses(gmOffsetExpr);
    int dim = static_cast<int>(rawShape[param.gmIdx].size());
    std::string coord = "Coord" + std::to_string(dim) + DIM + coordCp;

    std::vector<std::string> tileOpParamList = GetTileOpParamsByOrder();
    tileOpParamList.emplace_back(coord);
    std::ostringstream oss;
    oss << tileOpName << "<" << (transposeAxis[0] + correctionAxis) << ", " << (transposeAxis[1] + correctionAxis)
        << ">" << WrapParamByParentheses(tileOpParamList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::PrintTransposeDataMoveStatic(const PrintTransposeDataMoveParam& param) const
{
    const std::string& localVar = param.localVar;
    const std::string& localDtypeStr = param.localDtypeStr;
    const std::string& gmDtypeStr = param.gmDtypeStr;
    std::string dstVar = GenGmParamVar(ID0);
    std::vector<int64_t> os = NormalizeShape(shape[1], SHAPE_DIM4);
    std::vector<int64_t> gmShape = NormalizeShape(param.gmShape, SHAPE_DIM4);
    std::vector<int64_t> srcShape = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::ostringstream oss;
    std::vector<std::string> paramList;

    paramList.emplace_back(gmDtypeStr);
    for (auto oriShape : os) {
        paramList.emplace_back(std::to_string(oriShape));
    }
    for (int i = 1; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(std::to_string(gmShape[i]));
    }
    for (int i = 1; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(std::to_string(srcShape[i]));
    }
    std::vector<int64_t> transposeAxis = AnyCast<std::vector<int64_t>>(opAttrs.at(OP_ATTR_PREFIX + "shape"));
    int correctionAxis = SHAPE_DIM4 - shape[0].size();
    for (auto& axis : transposeAxis) {
        axis += correctionAxis;
        paramList.emplace_back(std::to_string(axis));
    }
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    paramList.clear();

    std::string dst = "(__gm__ " + gmDtypeStr + "*)" + dstVar;
    std::string src = "(__ubuf__ " + localDtypeStr + "*)" + localVar;
    paramList.insert(paramList.end(), {dst, src});
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);

    oss << tileOpName.c_str() << "_<" << templateParam << ">"
        << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpNPU::PrintTransposeDataMoveDynamic(const PrintTransposeDataMoveParam& param) const
{
    const std::string& localVar = param.localVar;
    const std::string& localDtypeStr = param.localDtypeStr;
    const std::string& gmDtypeStr = param.gmDtypeStr;
    std::string dstVar = GenGmParamVar(ID0);

    int dim = static_cast<int>(rawShape[ID0].size());
    std::vector<std::string> gmShapeExpr = GenGetParamMacroPacked(ID0, dim, PREFIX_STR_RAW_SHAPE);
    FillVecWithDummyInHead<std::string>(gmShapeExpr, SHAPE_DIM4 - dim, "1");
    CODEGEN_LOGI("dynamic gmShape param: %s", IntVecToStr(gmShapeExpr).c_str());

    std::vector<std::string> gmOffsetExpr = GenGetParamMacroPacked(ID0, dim, PREFIX_STR_OFFSET);
    FillVecWithDummyInHead<std::string>(gmOffsetExpr, SHAPE_DIM4 - dim, "0");
    CODEGEN_LOGI("dynamic gmOffset param: %s", IntVecToStr(gmOffsetExpr).c_str());

    std::vector<int64_t> os = NormalizeShape(shape[1], SHAPE_DIM4);
    std::vector<int64_t> srcShape = NormalizeShape(rawShape[1], SHAPE_DIM4);
    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.emplace_back(gmDtypeStr);
    for (auto oriShape : os) {
        paramList.emplace_back(std::to_string(oriShape));
    }
    for (int i = 1; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(std::to_string(srcShape[i]));
    }
    std::vector<int64_t> transposeAxis = AnyCast<std::vector<int64_t>>(opAttrs.at(OP_ATTR_PREFIX + "shape"));
    int correctionAxis = SHAPE_DIM4 - shape[1].size();
    for (auto& axis : transposeAxis) {
        axis += correctionAxis;
        paramList.emplace_back(std::to_string(axis));
    }
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    paramList.clear();

    std::string dst = "(__gm__ " + gmDtypeStr + "*)" + dstVar;
    std::string src = "(__ubuf__ " + localDtypeStr + "*)" + localVar;
    paramList.insert(paramList.end(), {dst, src});
    for (auto gs : gmShapeExpr) {
        paramList.emplace_back(gs);
    }
    for (auto go : gmOffsetExpr) {
        paramList.emplace_back(go);
    }
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    oss << tileOpName << "<" << templateParam << ">"
        << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpNPU::PrintTransposeDataMoveDynamicUnaligned(const PrintTransposeDataMoveParam& param) const
{
    const int gmIdx = param.gmIdx;
    const int localIdx = param.localIdx;
    std::string gmVar = GenGmParamVar(gmIdx);

    int dim = static_cast<int>(rawShape[gmIdx].size());
    std::vector<std::string> gmShapeExpr = GenGetParamMacroPacked(gmIdx, dim, PREFIX_STR_RAW_SHAPE);
    FillVecWithDummyInHead<std::string>(gmShapeExpr, SHAPE_DIM5 - dim, "1");
    CODEGEN_LOGI("dynamic gmShape param: %s", IntVecToStr(gmShapeExpr).c_str());

    std::vector<std::string> gmOffsetExpr = GenGetParamMacroPacked(gmIdx, dim, PREFIX_STR_OFFSET);
    FillVecWithDummyInHead<std::string>(gmOffsetExpr, SHAPE_DIM5 - dim, "0");
    CODEGEN_LOGI("dynamic gmOffset param: %s", IntVecToStr(gmOffsetExpr).c_str());
    auto newDynLocalValidShape = dynamicValidShape[localIdx];
    FillVecWithDummyInHead<SymbolicScalar>(newDynLocalValidShape, SHAPE_DIM5 - dim, 1);

    std::vector<int64_t> localShape = NormalizeShape(rawShape[localIdx], SHAPE_DIM5);
    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.emplace_back(param.gmDtypeStr);
    for (int i = 1; i < SHAPE_DIM5; i++) {
        paramList.emplace_back(std::to_string(localShape[i]));
    }
    std::vector<int64_t> transposeAxis = AnyCast<std::vector<int64_t>>(opAttrs.at(OP_ATTR_PREFIX + "shape"));
    int correctionAxis = SHAPE_DIM5 - shape[localIdx].size();
    for (auto& axis : transposeAxis) {
        axis += correctionAxis;
        paramList.emplace_back(std::to_string(axis));
    }
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    paramList.clear();

    std::string gm = "(__gm__ " + param.gmDtypeStr + "*)" + gmVar;
    std::string ub = "(__ubuf__ " + param.localDtypeStr + "*)" + param.localVar;

    if (gmIdx == 0) {
        paramList.insert(paramList.end(), {gm, ub});
    } else {
        paramList.insert(paramList.end(), {ub, gm});
    }

    FillParamWithFullInput(paramList, newDynLocalValidShape);
    for (auto gs : gmShapeExpr) {
        paramList.emplace_back(gs);
    }
    for (auto go : gmOffsetExpr) {
        paramList.emplace_back(go);
    }
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    oss << tileOpName << "_<" << templateParam << ">"
        << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpNPU::PrintIndexPutDynamicUnaligned(const PrintIndexPutParam& param) const
{
    const std::string& dstVar = param.dVar;
    const std::string& src1Var = param.s1Var;
    std::vector<std::string> src2Var = param.s2Var;
    const std::vector<std::string>& dataTypeExpr = param.dataTypeExpr;
    size_t dstRank = param.gmShape.size();
    std::vector<int64_t> s1rs = NormalizeShape(param.src1RawShape, SHAPE_DIM4);
    int dim = static_cast<int>(rawShape[ID0].size());
    auto paramPack = GenParamIdxExprByIndex(ID0, dim, PREFIX_STR_RAW_SHAPE);
    FillVecWithDummyInHead<std::string>(paramPack, ID4 - dim, "1");
    bool accumulate = param.accumulate;

    // template param
    std::vector<std::string> paramList;
    paramList.insert(paramList.end(), {dataTypeExpr[ID1], dataTypeExpr[ID3]});
    paramList.emplace_back(std::to_string(dstRank));
    for (int i = 1; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(std::to_string(s1rs[i]));
    }
    paramList.emplace_back(std::to_string(accumulate));
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    // function actual params
    paramList.clear();
    std::string dst = "(__gm__ " + dataTypeExpr[ID0] + "*)" + dstVar;
    std::string src1 = "(__ubuf__ " + dataTypeExpr[ID2] + "*)" + src1Var;
    paramList.insert(paramList.end(), {dst, src1});
    for (size_t i = 0; i < src2Var.size(); i++) {
        std::string src2Temp = "(__ubuf__ " + dataTypeExpr[ID3] + "*)" + src2Var[i];
        paramList.emplace_back(src2Temp);
    }
    auto validShape = dynamicValidShape[ID2]; // src1
    paramList.emplace_back(SymbolicExpressionTable::BuildExpression(validShape[0]));
    paramList.insert(paramList.end(), paramPack.begin(), paramPack.end());

    std::string tileOpCallParam = JoinString(paramList, CONN_COMMA);
    std::ostringstream os;
    os << tileOpName.c_str() << "<" << templateParam << ">"
       << "(" << tileOpCallParam << ");\n";
    return os.str();
}

std::string CodeGenOpNPU::PrintIndexPut(const PrintIndexPutParam& param) const
{
    ASSERT(GenCodeErr::PRINT_MODE_ERROR, isDynamicFunction) << "Only Support the DynamicUnaligned tileOp";
    return PrintIndexPutDynamicUnaligned(param);
}

std::string CodeGenOpNPU::PrintIndexPutLayout(size_t indicesSize, bool accumulate, bool requiresSimt) const
{
    std::string dstTensor = QueryTileTensorNameByIdx(ID0);
    std::vector<std::string> gmOffsetExpr = GetGmOffsetForTileTensor(ID0);
    std::string coordCp = WrapParamByParentheses(gmOffsetExpr);
    std::string coord = PrintCoord(rawShape[ID0].size(), coordCp);
    size_t valuesIndex = requiresSimt ? ID3 : ID2;
    std::string valuesTensor = QueryTileTensorNameByIdx(valuesIndex);
    std::vector<std::string> paramList = {dstTensor, coord, valuesTensor};
    if (requiresSimt) {
        paramList.push_back(QueryTileTensorNameByIdx(ID1));
    }
    for (size_t i = 0; i < SHAPE_DIM4; ++i) {
        if (i < indicesSize) {
            std::string indices = QueryTileTensorNameByIdx(valuesIndex + ID1 + i);
            paramList.push_back(indices);
        } else {
            paramList.push_back(paramList.back());
        }
    }
    std::ostringstream oss;
    oss << tileOpName << "<" << accumulate << ", " << indicesSize << ">" << WrapParamByParentheses(paramList)
        << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenIndexPutOp() const
{
    ASSERT(OperErr::ATTRIBUTE_INVALID, opAttrs.count(OpAttributeKey::accumulate)) << "cannot get accumulate attr";
    ASSERT(OperErr::ATTRIBUTE_INVALID, opAttrs.count(OpAttributeKey::indicesSize)) << "cannot get indicesSize attr";
    bool accumulate = AnyCast<bool>(opAttrs.at(OpAttributeKey::accumulate));
    int64_t indicesSize = AnyCast<int64_t>(opAttrs.at(OpAttributeKey::indicesSize));
    if (isSupportTileTensor) {
        bool requiresSimt = false;
        GetOpAttr(OP_ATTR_PREFIX + "requires_simt", requiresSimt);
        return PrintIndexPutLayout(indicesSize, accumulate, requiresSimt);
    }
    // dst:gm, s0/self:gm, s1/values:ub, s2/indices:ub
    std::string dstVar = GenGmParamVar(ID0);
    std::string s1Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID2]);
    std::vector<std::string> s2Var;
    for (int i = 0; i < indicesSize; i++) {
        std::string s2VarTemp = sm->QueryVarNameByTensorMagic(operandWithMagic[ID3 + i]);
        s2Var.emplace_back(s2VarTemp);
    }
    std::vector gmShape = rawShape[ID0];
    std::vector src1RawShape = rawShape[ID2];

    std::vector<std::string> dataTypeExpr;
    for (int i = 0; i < NUM4; i++) {
        dataTypeExpr.emplace_back(DataType2CCEStr(operandDtype[i]));
    }

    std::map<unsigned, std::reference_wrapper<std::string>> vars;
    vars.insert({ID1, s1Var});
    for (int i = 0; i < indicesSize; i++) {
        vars.insert({i + ID2, s2Var[i]});
    }
    AppendLocalBufferVarOffset(vars);

    return PrintIndexPut({dstVar, s1Var, s2Var, gmShape, src1RawShape, dataTypeExpr, accumulate});
}

std::string CodeGenOpNPU::PrintRangeTileTensor(const std::string& startVal, const std::string& stepVal,
                                               const std::string& tileIdxExpr) const
{
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    auto dstValidShape = dynamicValidShape[ToUnderlying(MISOIdx::DST_IDX)];
    std::vector<std::string> paramList = {dstTensor, SymbolicExpressionTable::BuildExpression(dstValidShape[ID0]),
                                          startVal, stepVal, tileIdxExpr};
    std::ostringstream oss;
    oss << tileOpName;
    oss << PrintParams({"(", ")"}, paramList, ", ");
    oss << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenUniformOp() const
{
    auto scalarsAttr = opAttrs.at(OpAttributeKey::vectorScalar);
    auto counter0Attr = opAttrs.at(OpAttributeKey::dynScalar);
    auto shapeAttr = opAttrs.at(OP_ATTR_PREFIX + "SHAPE");

    auto scalars = AnyCast<std::vector<Element>>(scalarsAttr);
    uint64_t key = scalars[0].Cast<uint64_t>();
    uint64_t counter1 = scalars[1].Cast<uint64_t>();
    uint16_t rounds = scalars[2].Cast<uint16_t>();

    SymbolicScalar counter0 = AnyCast<SymbolicScalar>(counter0Attr);

    std::vector<int64_t> randomShape;
    if (shapeAttr.has_value()) {
        randomShape = AnyCast<std::vector<int64_t>>(shapeAttr);
    }

    std::string keyStr = std::to_string(key) + "ULL";
    std::string counter0Str = "(uint64_t)(" + SymbolicExpressionTable::BuildExpression(counter0) + ")";
    std::string counter1Str = std::to_string(counter1) + "ULL";
    std::string roundsStr = std::to_string(rounds);

    std::vector<std::string> paramList = GetTileOpParamsByOrder();
    paramList.insert(paramList.end(), {keyStr, counter0Str, counter1Str, roundsStr});
    std::ostringstream oss;
    oss << tileOpName << WrapParamByParentheses(paramList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenRangeOp() const
{
    auto start = opAttrs.at(OP_ATTR_PREFIX + "START");
    auto step = opAttrs.at(OP_ATTR_PREFIX + "STEP");
    std::string startVal, stepVal, tileIdxExpr;
    ASSERT(OperErr::ATTRIBUTE_INVALID, start.has_value() && step.has_value()) << "GenRangeOp failed ";

    switch (operandDtype[ID0]) {
        case DataType::DT_FP32:
            startVal = FormatFloat(AnyCast<Element>(start).Cast<float>());
            stepVal = FormatFloat(AnyCast<Element>(step).Cast<float>());
            break;
        case DataType::DT_INT32:
            startVal = std::to_string(AnyCast<Element>(start).Cast<int>());
            stepVal = std::to_string(AnyCast<Element>(step).Cast<int>());
            break;
        case DataType::DT_INT64:
            startVal = std::to_string(AnyCast<Element>(start).Cast<int64_t>());
            stepVal = std::to_string(AnyCast<Element>(step).Cast<int64_t>());
            break;
        default:
            CODEGEN_LOGE(GenCodeErr::DATA_TYPE_UNSUPPORTED, "RangeOp from PASS encountered an unsupported DataType: %d",
                         operandDtype[ID0]);
            return CG_ERROR;
    }
    if (opAttrs.count(OpAttributeKey::dynScalar)) {
        auto scalarAny = opAttrs.at(OpAttributeKey::dynScalar);
        ASSERT(OperErr::ATTRIBUTE_INVALID, (scalarAny.has_value()) && (scalarAny.type() == typeid(SymbolicScalar)))
            << AnyCast<SymbolicScalar>(scalarAny).IsValid() << "SCALAR attribute has to have symbolic value.";
        auto scalarExpr = AnyCast<SymbolicScalar>(scalarAny);
        tileIdxExpr = "((int64_t)(" + SymbolicExpressionTable::BuildExpression(scalarExpr) + "))";
    }
    if (isSupportTileTensor) {
        return PrintRangeTileTensor(startVal, stepVal, tileIdxExpr);
    }
    // only support 1 dim
    std::string dVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);

    AppendLocalBufVarOffsetInOrder(dVar);
    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.emplace_back(dstDtypeStr);
    paramList.emplace_back(std::to_string(rawShape[0][0]));
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    paramList.clear();
    std::string dst = "(" + GetAddrTypeByOperandType(BUF_UB) + " " + dstDtypeStr + "*)" + dVar;
    paramList.emplace_back(dst);
    paramList.emplace_back(dynamicValidShape[ID0][ID0].Dump());
    paramList.insert(paramList.end(), {startVal, stepVal});
    paramList.emplace_back(tileIdxExpr);
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    oss << tileOpName << "<" << templateParam << ">"
        << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpNPU::PrintIndexAddUBDynamicUnaligned(const PrintIndexAddParam& param) const
{
    // support 2-4 dims
    const std::string& dstVar = param.dstVar;
    const std::string& srcVar = param.srcVar;
    const std::string& indicesVar = param.indicesVar;
    std::vector<int64_t> dstRawShape = NormalizeShape(param.dstRawShape, SHAPE_DIM4);
    std::vector<int64_t> srcRawShape = NormalizeShape(param.srcRawShape, SHAPE_DIM4);
    const std::vector<std::string>& dataTypeExpr = param.dataTypeExpr;

    const Element& alpha = extOperandVal;

    // template params
    std::vector<std::string> paramList;
    paramList.insert(paramList.end(), {dataTypeExpr[ID0], dataTypeExpr[ID2], DataType2CCEStr(alpha.GetDataType())});
    for (size_t i = 1; i < srcRawShape.size(); ++i) {
        paramList.emplace_back(std::to_string(srcRawShape[i]));
    }
    for (size_t i = 1; i < dstRawShape.size(); ++i) {
        paramList.emplace_back(std::to_string(dstRawShape[i]));
    }
    int axis = param.axis + SHAPE_DIM4 - param.srcRawShape.size(); // 调用4维tileop需要切换axis
    paramList.emplace_back(std::to_string(axis));
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    // function actual params
    paramList.clear();
    std::string addrType = GetAddrTypeByOperandType(BUF_UB);
    std::string dst = "(" + addrType + " " + dataTypeExpr[ID0] + "*)" + dstVar;
    std::string src = "(" + addrType + " " + dataTypeExpr[ID1] + "*)" + srcVar;
    std::string indices = "(" + addrType + " " + dataTypeExpr[ID2] + "*)" + indicesVar;
    paramList.insert(paramList.end(), {dst, src, indices});
    std::string scalarTmpBuffer = FormatScalarLiteral(alpha);
    paramList.emplace_back("(" + std::string(DataType2CCEStr(alpha.GetDataType())) + ")" + scalarTmpBuffer);
    auto validShape = dynamicValidShape[ID3]; // srcvalidshape
    FillVecWithDummyInHead<SymbolicScalar>(validShape, SHAPE_DIM4 - validShape.size(), 1);
    FillParamWithFullInput(paramList, validShape);
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);

    std::ostringstream oss;
    oss << tileOpName << "<" << templateParam << ">"
        << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpNPU::PrintIndexAddUBTileTensor(const PrintIndexAddParam& param) const
{
    std::vector<std::string> paramList;
    int axis = param.axis + SHAPE_DIM5 - param.srcRawShape.size();
    paramList.emplace_back(std::to_string(axis));
    std::string templateParam = JoinString(paramList, CONN_COMMA);

    paramList.clear();

    paramList = GetTileOpParamsWithTmpBuf({ToUnderlying(MIMOIdx::TMP_IDX)});
    const Element& alpha = extOperandVal;
    std::string scalarTmpBuffer = FormatScalarLiteral(alpha);
    paramList.emplace_back("(" + std::string(DataType2CCEStr(alpha.GetDataType())) + ")" + scalarTmpBuffer);
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    std::ostringstream oss;
    oss << tileOpName << "<" << templateParam << ">"
        << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpNPU::GenIndexAddUBOp() const
{
    std::string dstVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string selfVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID2]);
    std::string srcVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID3]);
    std::string indicesVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID4]);

    std::vector dstRawShape = rawShape[ID0];
    std::vector srcRawShape = rawShape[ID3];

    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string srcDtypeStr = DataType2CCEStr(operandDtype[ID3]);
    std::string indicesDtypeStr = DataType2CCEStr(operandDtype[ID4]);
    const std::vector<std::string> dataTypeExpr = {dstDtypeStr, srcDtypeStr, indicesDtypeStr};

    AppendLocalBufVarOffsetInOrder(dstVar, selfVar, srcVar, indicesVar);

    ASSERT(OperErr::ATTRIBUTE_INVALID, opAttrs.count(OP_ATTR_PREFIX + "axis")) << "cannot get axis attr";
    int axis = AnyCast<int64_t>(opAttrs.at(OP_ATTR_PREFIX + "axis"));
    if (isSupportTileTensor) {
        return PrintIndexAddUBTileTensor({axis, dstVar, srcVar, indicesVar, dstRawShape, srcRawShape, dataTypeExpr});
    }
    return PrintIndexAddUBDynamicUnaligned({axis, dstVar, srcVar, indicesVar, dstRawShape, srcRawShape, dataTypeExpr});
}

std::string CodeGenOpNPU::GenIndexAddOp() const
{
    ASSERT(OperErr::ATTRIBUTE_INVALID, opAttrs.count(OP_ATTR_PREFIX + "axis")) << "cannot get axis attr";
    ASSERT(GenCodeErr::PRINT_MODE_ERROR, isSupportTileTensor) << "IndexAdd operation only support TileTensor mode";
    int axis = AnyCast<int64_t>(opAttrs.at(OP_ATTR_PREFIX + "axis"));
    axis += SHAPE_DIM5 - rawShape[ID0].size();
    std::vector<std::string> gmOffsetExpr = GetGmOffsetForTileTensor(ID0);
    std::string coord = PrintCoord(rawShape[ID0].size(), WrapParamByParentheses(gmOffsetExpr));

    std::vector<std::string> templateParamList{std::to_string(axis)};
    std::vector<std::string> tileOpParamList = GetTileOpParamsWithTmpBuf({ToUnderlying(MIMOIdx::TMP_IDX)});
    tileOpParamList.emplace_back(coord);
    const Element& alpha = extOperandVal;
    std::string scalarTmpBuffer = FormatScalarLiteral(alpha);
    tileOpParamList.emplace_back("(" + std::string(DataType2CCEStr(alpha.GetDataType())) + ")" + scalarTmpBuffer);
    std::ostringstream oss;
    bool requiresSimt = false;
    GetOpAttr(OP_ATTR_PREFIX + "requires_simt", requiresSimt);
    oss << (requiresSimt ? "TIndexAddSimt" : tileOpName) << WrapParamByAngleBrackets(templateParamList)
        << WrapParamByParentheses(tileOpParamList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::PrintCumSumDynamicUnaligned(const PrintCumSumParam& param) const
{
    // support 2-4 dims
    const std::string& dstVar = param.dVar;
    const std::string& inputVar = param.inputVar;

    std::vector<int64_t> inputRawShape = NormalizeShape(param.inputRawShape, SHAPE_DIM4);
    const std::string* dataTypeExpr = param.dataTypeExpr;

    // template params
    std::vector<std::string> paramList;
    paramList.insert(paramList.end(), {dataTypeExpr[ID0]});
    for (size_t i = 0; i < inputRawShape.size(); ++i) {
        paramList.emplace_back(std::to_string(inputRawShape[i]));
    }

    bool flag = param.flag;
    paramList.emplace_back(std::to_string(param.axis));
    paramList.emplace_back(std::to_string(flag));
    std::string templateParam = JoinString(paramList, ", ");

    // function actual params
    paramList.clear();
    std::string addrType = GetAddrTypeByOperandType(BUF_UB);
    std::string dst = "(" + addrType + " " + dataTypeExpr[ID0] + "*)" + dstVar;
    std::string input = "(" + addrType + " " + dataTypeExpr[ID1] + "*)" + inputVar;

    paramList.insert(paramList.end(), {dst, input});

    auto validShape = dynamicValidShape[ID1];
    FillVecWithDummyInHead<SymbolicScalar>(validShape, SHAPE_DIM4 - validShape.size(), 1);
    FillParamWithFullInput(paramList, validShape);
    std::string tiloOpCallParam = JoinString(paramList, ", ");
    std::ostringstream oss;
    oss << tileOpName << "<" << templateParam << ">"
        << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpNPU::PrintCumOperationTileTensor(int axis, bool is_sum) const
{
    axis = axis + 1;
    std::vector<std::string> paramList = GetTileOpParamsByOrder();
    std::vector<std::string> templateParam = {std::to_string(axis), std::to_string(is_sum)};
    std::ostringstream oss;
    oss << tileOpName << WrapParamByAngleBrackets(templateParam) << WrapParamByParentheses(paramList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenCumOperationOp() const
{
    std::string dstVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string inputVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);

    std::vector inputRawShape = rawShape[ID1];

    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string inputDtypeStr = DataType2CCEStr(operandDtype[ID1]);

    constexpr int NumOperands = 2;
    std::string dataTypeExpr[NumOperands] = {dstDtypeStr, inputDtypeStr};
    AppendLocalBufVarOffsetInOrder(dstVar, inputVar);

    ASSERT(OperErr::ATTRIBUTE_INVALID, opAttrs.count(OP_ATTR_PREFIX + "axis")) << "cannot get axis attr";
    int axis = AnyCast<int64_t>(opAttrs.at(OP_ATTR_PREFIX + "axis"));
    axis = axis + SHAPE_DIM4 - inputRawShape.size(); // 调用4维tileop需要切换axis

    ASSERT(OperErr::ATTRIBUTE_INVALID, opAttrs.count(OP_ATTR_PREFIX + "flag")) << "cannot get flag attr";
    bool is_sum = AnyCast<bool>(opAttrs.at(OP_ATTR_PREFIX + "flag"));

    if (isSupportTileTensor) {
        return PrintCumOperationTileTensor(axis, is_sum);
    } else {
        return PrintCumSumDynamicUnaligned({axis, is_sum, dstVar, inputVar, inputRawShape, dataTypeExpr});
    }
}

std::string CodeGenOpNPU::PrintTriULTileTensor(const std::string& diagonal, bool isUpper) const
{
    std::vector<std::string> paramList = GetTileOpParamsByOrder();
    paramList.emplace_back(diagonal);

    std::ostringstream oss;
    oss << tileOpName << "<" << isUpper << ">" << WrapParamByParentheses(paramList) << ";\n";
    return oss.str();
}

std::string CodeGenOpNPU::GenTriULOp() const
{
    ASSERT(OperErr::ATTRIBUTE_INVALID, opAttrs.count(OpAttributeKey::dynScalar)) << "cannot get diagonal attr";
    ASSERT(OperErr::ATTRIBUTE_INVALID, opAttrs.count(OpAttributeKey::isUpper)) << "cannot get isUpper attr";
    auto scalarAny = opAttrs.at(OpAttributeKey::dynScalar);
    ASSERT(OperErr::ATTRIBUTE_INVALID, (scalarAny.has_value()) && (scalarAny.type() == typeid(SymbolicScalar)))
        << AnyCast<SymbolicScalar>(scalarAny).IsValid() << "diagonal must have symbolic value.";
    auto scalarExpr = AnyCast<SymbolicScalar>(scalarAny);

    std::string diagonal = "(int)(" + SymbolicExpressionTable::BuildExpression(scalarExpr) + ")";
    bool isUpper = AnyCast<bool>(opAttrs.at(OpAttributeKey::isUpper));

    ASSERT(GenCodeErr::PRINT_MODE_ERROR, isSupportTileTensor) << "TriU or TriL only support TileTensor mode";
    return PrintTriULTileTensor(diagonal, isUpper);
}

void CodeGenOpNPU::GetWhereVarAndType(std::vector<std::string>& varExpr, std::vector<std::string>& dataTypeExpr) const
{
    varExpr.clear();
    dataTypeExpr.clear();

    const int paramCnt = 5;
    varExpr.reserve(paramCnt);

    varExpr.emplace_back(
        sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(WhereOpIdx::resIdx)])); // 0: dstVar
    varExpr.emplace_back(
        sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(WhereOpIdx::tempIdx)])); // 1: tempVar
    varExpr.emplace_back(
        sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(WhereOpIdx::condIdx)])); // 2: condVar

    const int inValidIdx = -1;
    int src0Idx = inValidIdx, src1Idx = inValidIdx;
    if (opCode == Opcode::OP_WHERE_ST || opCode == Opcode::OP_WHERE_TS || opCode == Opcode::OP_WHERE_TT) {
        // 3: src0Var
        varExpr.emplace_back(sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(WhereOpIdx::src0Idx)]));
        src0Idx = varExpr.size() - 1;
    }
    if (opCode == Opcode::OP_WHERE_TT) {
        // 4: src1Var
        varExpr.emplace_back(sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(WhereOpIdx::src1Idx)]));
        src1Idx = varExpr.size() - 1;
    }

    std::map<unsigned, std::reference_wrapper<std::string>> varMap;
    std::vector<unsigned> idxs = {ToUnderlying(WhereOpIdx::resIdx), ToUnderlying(WhereOpIdx::tempIdx),
                                  ToUnderlying(WhereOpIdx::condIdx)};
    for (unsigned i = 0; i < idxs.size(); ++i) {
        varMap.emplace(idxs[i], std::ref(varExpr[i]));
    }
    if (src0Idx != inValidIdx) {
        varMap.emplace(ToUnderlying(WhereOpIdx::src0Idx), std::ref(varExpr[src0Idx]));
    }
    if (src1Idx != inValidIdx) {
        varMap.emplace(ToUnderlying(WhereOpIdx::src1Idx), std::ref(varExpr[src1Idx]));
    }

    AppendLocalBufferVarOffset(varMap);

    dataTypeExpr = {DataType2CCEStr(operandDtype[ToUnderlying(WhereOpIdx::resIdx)]),
                    DataType2CCEStr(operandDtype[ToUnderlying(WhereOpIdx::tempIdx)]),
                    DataType2CCEStr(operandDtype[ToUnderlying(WhereOpIdx::condIdx)])};
}

WhereParam CodeGenOpNPU::PrepareWhereParam() const
{
    std::vector<std::string> varExpr;
    std::vector<std::string> dataTypeExpr;
    GetWhereVarAndType(varExpr, dataTypeExpr);
    std::vector<int64_t> ds = NormalizeShape(rawShape[ToUnderlying(WhereOpIdx::resIdx)], SHAPE_DIM4);
    std::vector<int64_t> c0s = NormalizeShape(rawShape[ToUnderlying(WhereOpIdx::condIdx)], SHAPE_DIM4);
    std::vector<int64_t> s0s = NormalizeShape(rawShape[ToUnderlying(WhereOpIdx::src0Idx)], SHAPE_DIM4);
    std::vector<std::string> templateList;
    templateList.emplace_back(dataTypeExpr[ToUnderlying(WhereOpIdx::resIdx)]);
    templateList.emplace_back(dataTypeExpr[ToUnderlying(WhereOpIdx::condIdx)]);
    templateList.emplace_back("/*DstRawShape*/");
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        templateList.emplace_back(std::to_string(ds[i]));
    }
    templateList.emplace_back("/*ConditionRawShape*/");
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        templateList.emplace_back(std::to_string(c0s[i]));
    }
    templateList.emplace_back("/*Src0RawShape*/");
    for (int i = 1; i < SHAPE_DIM4; ++i) {
        templateList.emplace_back(std::to_string(s0s[i]));
    }

    std::vector<std::string> paramList;
    paramList.emplace_back("(__ubuf__ " + dataTypeExpr[ToUnderlying(WhereOpIdx::resIdx)] + "*)" +
                           varExpr[ToUnderlying(WhereOpIdx::resIdx)]);
    paramList.emplace_back("(__ubuf__ " + dataTypeExpr[ToUnderlying(WhereOpIdx::tempIdx)] + "*)" +
                           varExpr[ToUnderlying(WhereOpIdx::tempIdx)]);
    paramList.emplace_back("(__ubuf__ " + dataTypeExpr[ToUnderlying(WhereOpIdx::condIdx)] + "*)" +
                           varExpr[ToUnderlying(WhereOpIdx::condIdx)]);
    std::vector<std::string> dynParamList;
    auto dynSrcShape = dynamicValidShape[ToUnderlying(WhereOpIdx::resIdx)];
    FillVecWithDummyInHead<SymbolicScalar>(dynSrcShape,
                                           SHAPE_DIM4 - dynamicValidShape[ToUnderlying(WhereOpIdx::resIdx)].size(), 1);
    for (int i = 0; i < SHAPE_DIM4; i++) {
        dynParamList.emplace_back(dynSrcShape[i].Dump());
    }
    WhereParam param{templateList, paramList, dynParamList, varExpr, dataTypeExpr};
    return param;
}

std::string CodeGenOpNPU::PrintWhereOp(const WhereParam& param) const
{
    std::vector<std::string> templateList = param.templateList;
    std::vector<std::string> paramList = param.paramList;
    std::vector<std::string> dynParamList = param.dynParamList;
    std::vector<std::string> varExpr = param.varExpr;
    std::vector<std::string> dataTypeExpr = param.dataTypeExpr;
    std::string templateParam = JoinString(templateList, CONN_COMMA);
    std::string funcParam = JoinString(paramList, CONN_COMMA);
    std::string dynFuncParam = JoinString(dynParamList, CONN_COMMA);
    std::vector<std::string> extList;

    std::ostringstream os;
    if (opCode == Opcode::OP_WHERE_SS) {
        std::string src0Var = FormatScalarLiteral(extScalarVec[0]);
        std::string src1Var = FormatScalarLiteral(extScalarVec[1]);
        extList.emplace_back(dataTypeExpr[0] + "(" + src0Var + ")");
        extList.emplace_back(dataTypeExpr[0] + "(" + src1Var + ")");
        auto extParam = JoinString(extList, ", ");
        os << tileOpName.c_str() << "<" << templateParam << ">"
           << "(" << funcParam << ", " << extParam << ", " << dynFuncParam << ");\n";
        return os.str();
    } else if (opCode == Opcode::OP_WHERE_ST) {
        std::string scalarVar = FormatScalarLiteral(extOperandVal);
        std::string src0Var = varExpr[ToUnderlying(WhereOpIdx::src0Idx)];
        std::string src1DtypeStr = DataType2CCEStr(operandDtype[ToUnderlying(WhereOpIdx::src0Idx)]);
        extList.emplace_back(dataTypeExpr[0] + "(" + scalarVar + ")");
        extList.emplace_back("(__ubuf__ " + src1DtypeStr + "*)" + src0Var);
        auto extParam = JoinString(extList, ", ");
        os << tileOpName.c_str() << "<" << templateParam << ">"
           << "(" << funcParam << ", " << extParam << ", " << dynFuncParam << ");\n";
        return os.str();
    } else if (opCode == Opcode::OP_WHERE_TS) {
        std::string scalarVar = FormatScalarLiteral(extOperandVal);
        std::string src0Var = varExpr[ToUnderlying(WhereOpIdx::src0Idx)];
        std::string src0DtypeStr = DataType2CCEStr(operandDtype[ToUnderlying(WhereOpIdx::src0Idx)]);
        extList.emplace_back("(__ubuf__ " + src0DtypeStr + "*)" + src0Var);
        extList.emplace_back(dataTypeExpr[0] + "(" + scalarVar + ")");
        auto extParam = JoinString(extList, ", ");
        os << tileOpName.c_str() << "<" << templateParam << ">"
           << "(" << funcParam << ", " << extParam << ", " << dynFuncParam << ");\n";
        return os.str();
    } else { // opCode == Opcode::OP_WHERE_TT
        std::string src0Var = varExpr[ToUnderlying(WhereOpIdx::src0Idx)];
        std::string src0DtypeStr = DataType2CCEStr(operandDtype[ToUnderlying(WhereOpIdx::src0Idx)]);
        std::string src1Var = varExpr[ToUnderlying(WhereOpIdx::src1Idx)];
        std::string src1DtypeStr = DataType2CCEStr(operandDtype[ToUnderlying(WhereOpIdx::src1Idx)]);
        extList.emplace_back("(__ubuf__ " + src0DtypeStr + "*)" + src0Var);
        extList.emplace_back("(__ubuf__ " + src1DtypeStr + "*)" + src1Var);
        auto extParam = JoinString(extList, ", ");
        os << tileOpName.c_str() << "<" << templateParam << ">"
           << "(" << funcParam << ", " << extParam << ", " << dynFuncParam << ");\n";
        return os.str();
    }
}

std::string CodeGenOpNPU::PrintWhereOpTileTensor(const WhereParam& param) const
{
    std::vector<std::string> dataTypeExpr = param.dataTypeExpr;

    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(WhereOpIdx::resIdx));
    std::string tempTensor = QueryTileTensorNameByIdx(ToUnderlying(WhereOpIdx::tempIdx));
    std::string condTensor = QueryTileTensorNameByIdx(ToUnderlying(WhereOpIdx::condIdx));
    std::ostringstream oss;
    oss << tileOpName << "(" << dstTensor << ", " << tempTensor << ", " << condTensor << ", ";
    if (opCode == Opcode::OP_WHERE_TT) {
        std::string src0Tensor = QueryTileTensorNameByIdx(ToUnderlying(WhereOpIdx::src0Idx));
        std::string src1Tensor = QueryTileTensorNameByIdx(ToUnderlying(WhereOpIdx::src1Idx));
        oss << src0Tensor << ", " << src1Tensor << ");\n";
    }
    if (opCode == Opcode::OP_WHERE_TS) {
        std::string src0Tensor = QueryTileTensorNameByIdx(ToUnderlying(WhereOpIdx::src0Idx));
        std::string scalarVar = FormatScalarLiteral(extOperandVal);
        oss << src0Tensor << ", " << dataTypeExpr[0] + "(" + scalarVar + ")" << ");\n";
    }
    if (opCode == Opcode::OP_WHERE_ST) {
        std::string src0Tensor = QueryTileTensorNameByIdx(ToUnderlying(WhereOpIdx::src0Idx));
        std::string scalarVar = FormatScalarLiteral(extOperandVal);
        oss << dataTypeExpr[0] + "(" + scalarVar + ")" << ", " << src0Tensor << ");\n";
    }
    if (opCode == Opcode::OP_WHERE_SS) {
        std::string src0Var = FormatScalarLiteral(extScalarVec[0]);
        std::string src1Var = FormatScalarLiteral(extScalarVec[1]);
        std::vector<std::string> extList;
        extList.emplace_back(dataTypeExpr[0] + "(" + src0Var + ")");
        extList.emplace_back(dataTypeExpr[0] + "(" + src1Var + ")");
        auto extParam = JoinString(extList, ", ");
        oss << extParam << ");\n";
    }
    return oss.str();
}

std::string CodeGenOpNPU::GenWhereOp() const
{
    WhereParam param = PrepareWhereParam();
    if (isSupportTileTensor) {
        return PrintWhereOpTileTensor(param);
    }
    return PrintWhereOp(param);
}

std::string CodeGenOpNPU::GenLogicalNotOp() const
{
    if (isSupportTileTensor) {
        return PrintLogicalNotTileTensor();
    }
    // Support 2 dim
    enum class OpIdx : int { resIdx = 0, tmpIdx, srcIdx };

    std::string dstVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(OpIdx::resIdx)]);
    std::string tmpVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(OpIdx::tmpIdx)]);
    std::string srcVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(OpIdx::srcIdx)]);

    std::vector dstShape = rawShape[ToUnderlying(OpIdx::resIdx)];
    std::vector srcShape = rawShape[ToUnderlying(OpIdx::srcIdx)];

    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ToUnderlying(OpIdx::resIdx)]);
    std::string tmpDtypeStr = DataType2CCEStr(operandDtype[ToUnderlying(OpIdx::tmpIdx)]);
    std::string srcDtypeStr = DataType2CCEStr(operandDtype[ToUnderlying(OpIdx::srcIdx)]);

    AppendLocalBufVarOffsetInOrder(dstVar, tmpVar, srcVar);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(srcDtypeStr);
    int dim = dstShape.size();
    for (auto i = 1; i < dim; i++) {
        paramList.emplace_back(std::to_string(dstShape[i]));
    }
    for (auto i = 1; i < dim; i++) {
        paramList.emplace_back(std::to_string(srcShape[i]));
    }

    std::string templateParam = JoinString(paramList, CONN_COMMA);

    paramList.clear();

    paramList.emplace_back("(__ubuf__ " + dstDtypeStr + "*)" + dstVar);
    paramList.emplace_back("(__ubuf__ " + srcDtypeStr + "*)" + srcVar);
    paramList.emplace_back("(__ubuf__ " + tmpDtypeStr + "*)" + tmpVar);

    auto dynSrcShape = dynamicValidShape[ToUnderlying(OpIdx::srcIdx)];
    for (auto dyn : dynSrcShape) {
        paramList.emplace_back(dyn.Dump());
    }
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);

    os << tileOpName.c_str() << "<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpNPU::PrintCmpTileTensor() const
{
    auto cmpOp = opAttrs.at(OP_ATTR_PREFIX + "cmp_operation");
    auto mode = opAttrs.at(OP_ATTR_PREFIX + "cmp_mode");
    std::string cmpOpVal = std::to_string(AnyCast<int64_t>(cmpOp));
    std::string modeVal = std::to_string(AnyCast<int64_t>(mode));

    std::vector<std::string> tileOpParamList = GetTileOpParamsWithTmpBuf({ToUnderlying(MIMOIdx::TMP_IDX)});
    std::vector<std::string> templateParamList = {cmpOpVal, modeVal};
    if (opCode == Opcode::OP_CMPS) {
        auto scalarAttr = opAttrs.at(OpAttributeKey::scalar);
        auto scalarElement = AnyCast<Element>(scalarAttr);
        auto scalarType = scalarElement.GetDataType();
        templateParamList.emplace_back(DataType2CCEStr(scalarType));
        tileOpParamList.emplace_back(FormatScalarLiteral(scalarElement));
    }
    std::ostringstream oss;
    oss << tileOpName;
    oss << WrapParamByAngleBrackets(templateParamList);
    oss << WrapParamByParentheses(tileOpParamList);
    oss << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenCmpOp() const
{
    if (isSupportTileTensor) {
        return PrintCmpTileTensor();
    }
    enum class TensorIdx : int { dstIdx = 0, tmpIdx, src0Idx, src1Idx };

    std::string dVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(TensorIdx::dstIdx)]);
    std::string tVar1 = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(TensorIdx::tmpIdx)]);

    bool isScalarMode = (opCode == Opcode::OP_CMPS);
    std::string s0Var, s1Var;
    std::vector<int64_t> src0RawShape, src1RawShape;
    auto newDynSrcValidShape = dynamicValidShape[ToUnderlying(TensorIdx::src0Idx)];

    if (isScalarMode) {
        s0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(TensorIdx::src0Idx)]);
        src0RawShape = NormalizeShape(rawShape[ToUnderlying(TensorIdx::src0Idx)], SHAPE_DIM4);
        FillVecWithDummyInHead<SymbolicScalar>(
            newDynSrcValidShape, SHAPE_DIM4 - dynamicValidShape[ToUnderlying(TensorIdx::src0Idx)].size(), 1);
    } else {
        s0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(TensorIdx::src0Idx)]);
        s1Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(TensorIdx::src1Idx)]);
        src0RawShape = NormalizeShape(rawShape[ToUnderlying(TensorIdx::src0Idx)], SHAPE_DIM4);
        src1RawShape = NormalizeShape(rawShape[ToUnderlying(TensorIdx::src1Idx)], SHAPE_DIM4);
        FillVecWithDummyInHead<SymbolicScalar>(
            newDynSrcValidShape, SHAPE_DIM4 - dynamicValidShape[ToUnderlying(TensorIdx::src0Idx)].size(), 1);
    }

    std::vector<int64_t> dstRawShape = NormalizeShape(rawShape[ToUnderlying(TensorIdx::dstIdx)], SHAPE_DIM4);
    std::string srcDtypeStr = DataType2CCEStr(operandDtype[ToUnderlying(TensorIdx::src0Idx)]);

    if (isScalarMode) {
        AppendLocalBufVarOffsetInOrder(dVar, tVar1, s0Var);
    } else {
        AppendLocalBufVarOffsetInOrder(dVar, tVar1, s0Var, s1Var);
    }

    auto cmpOp = opAttrs.at(OP_ATTR_PREFIX + "cmp_operation");
    auto mode = opAttrs.at(OP_ATTR_PREFIX + "cmp_mode");
    std::string cmpOpVal = std::to_string(AnyCast<int64_t>(cmpOp));
    std::string modeVal = std::to_string(AnyCast<int64_t>(mode));

    std::ostringstream oss;
    std::vector<std::string> paramList;
    paramList.emplace_back(srcDtypeStr);
    for (int i = ID1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(dstRawShape[i]));
    }
    for (int i = ID1; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(src0RawShape[i]));
    }
    if (!isScalarMode) {
        for (int i = ID1; i < SHAPE_DIM4; ++i) {
            paramList.emplace_back(std::to_string(src1RawShape[i]));
        }
    }
    paramList.emplace_back(cmpOpVal);
    paramList.emplace_back(modeVal);
    std::string templateParam = JoinString(paramList, ", ");

    paramList.clear();
    std::string dst = "(" + GetAddrTypeByOperandType(BUF_UB) + " uint8_t*)" + dVar;
    std::string src0 = "(" + GetAddrTypeByOperandType(BUF_UB) + " " + srcDtypeStr + "*)" + s0Var;

    std::string tmp1 = "(" + GetAddrTypeByOperandType(BUF_UB) + " uint8_t*)" + tVar1;

    paramList.insert(paramList.end(), {dst, src0});
    if (!isScalarMode) {
        std::string src1 = "(" + GetAddrTypeByOperandType(BUF_UB) + " " + srcDtypeStr + "*)" + s1Var;
        paramList.emplace_back(src1);
    }

    for (auto dynShape : newDynSrcValidShape) {
        paramList.emplace_back(dynShape.Dump());
    }

    paramList.emplace_back(tmp1);

    if (isScalarMode) {
        auto scalarAttr = opAttrs.at(OpAttributeKey::scalar);
        auto scalarElement = AnyCast<Element>(scalarAttr);
        paramList.emplace_back(FormatScalarLiteral(scalarElement));
    }

    std::string tiloOpCallParam = JoinString(paramList, ", ");
    oss << tileOpName << "<" << templateParam << ">"
        << "(" << tiloOpCallParam << ");\n";
    return oss.str();
}

std::string CodeGenOpNPU::PrintHypotTileTensor() const
{
    return PrintTileOpWithFullParamsTmpBuf({ToUnderlying(MIMOIdx::TMP_IDX)});
}

std::string CodeGenOpNPU::GenHypotOp() const
{
    ASSERT(GenCodeErr::PRINT_MODE_ERROR, isSupportTileTensor) << "Hypot only support tile tensor";
    return PrintHypotTileTensor();
}

std::string CodeGenOpNPU::PrintPreluTileTensor() const
{
    int64_t axis = 1;
    GetOpAttr(OP_ATTR_PREFIX + "axis", axis);

    std::vector<std::string> tileOpParamList = GetTileOpParamsWithTmpBuf({ToUnderlying(MIMOIdx::TMP_IDX)});

    std::ostringstream oss;
    oss << tileOpName << "<" << axis << ">" << WrapParamByParentheses(tileOpParamList) << STMT_END;

    return oss.str();
}

std::string CodeGenOpNPU::PrintPadTileTensor() const
{
    DataType dstDtype = operandDtype[ToUnderlying(MISOIdx::DST_IDX)];
    std::vector<std::string> tileOpParamList = GetTileOpParamsByOrder();
    std::ostringstream oss;

    const std::string fillPadCols = originalOp.GetOpcode() == Opcode::OP_FILLPAD ?
                                        ", " + std::to_string(shape[0].back()) :
                                        "";
    bool isFloatType = (dstDtype == DT_FP32 || dstDtype == DT_FP16 || dstDtype == DT_BF16);

    if (isFloatType) {
        auto c = extOperandVal.Cast<float>();
        if (std::isinf(c)) {
            oss << tileOpName << "<" << (c < 0 ? "pto::PadValue::Min" : "pto::PadValue::Max") << fillPadCols << ">"
                << WrapParamByParentheses(tileOpParamList) << STMT_END;
            return oss.str();
        }
    }

    std::string padValueStr;
    if (isFloatType) {
        padValueStr = FormatFloat(extOperandVal.Cast<float>());
    } else if (dstDtype == DT_INT16 || dstDtype == DT_UINT16) {
        padValueStr = std::to_string(extOperandVal.Cast<int16_t>());
    } else if (dstDtype == DT_INT8 || dstDtype == DT_UINT8) {
        padValueStr = std::to_string(extOperandVal.Cast<int8_t>());
    } else {
        padValueStr = std::to_string(extOperandVal.Cast<int32_t>());
    }

    std::string padValueArg = "(" + std::string(DataType2CCEStr(dstDtype)) + ")" + padValueStr;
    oss << tileOpName << "<pto::PadValueCustom(" << padValueArg << ")" << fillPadCols << ">"
        << WrapParamByParentheses(tileOpParamList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenPadOp() const { return PrintPadTileTensor(); }

std::string CodeGenOpNPU::GenPreluOp() const
{
    ASSERT(GenCodeErr::PRINT_MODE_ERROR, isSupportTileTensor) << "PReLU only support tile tensor";
    return PrintPreluTileTensor();
}

std::string CodeGenOpNPU::PrintLogicalAndTileTensor() const
{
    return PrintTileOpWithFullParamsTmpBuf({ToUnderlying(MIMOIdx::TMP_IDX)});
}

std::string CodeGenOpNPU::PrintLogicalNotTileTensor() const
{
    return PrintTileOpWithFullParamsTmpBuf({ToUnderlying(MIMOIdx::TMP_IDX)});
}

std::string CodeGenOpNPU::GenLogicalAndOp() const
{
    if (isSupportTileTensor) {
        return PrintLogicalAndTileTensor();
    }
    // Support 2 dim
    enum class OpIdx : int { resIdx = 0, tmpIdx, srcIdx0, srcIdx1 };

    std::string dstVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(OpIdx::resIdx)]);
    std::string tmpVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(OpIdx::tmpIdx)]);
    std::string srcVar0 = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(OpIdx::srcIdx0)]);
    std::string srcVar1 = sm->QueryVarNameByTensorMagic(operandWithMagic[ToUnderlying(OpIdx::srcIdx1)]);

    std::vector dstShape = rawShape[ToUnderlying(OpIdx::resIdx)];
    std::vector srcShape0 = rawShape[ToUnderlying(OpIdx::srcIdx0)];
    std::vector srcShape1 = rawShape[ToUnderlying(OpIdx::srcIdx1)];

    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ToUnderlying(OpIdx::resIdx)]);
    std::string tmpDtypeStr = DataType2CCEStr(operandDtype[ToUnderlying(OpIdx::tmpIdx)]);
    std::string srcDtypeStr0 = DataType2CCEStr(operandDtype[ToUnderlying(OpIdx::srcIdx0)]);
    std::string srcDtypeStr1 = DataType2CCEStr(operandDtype[ToUnderlying(OpIdx::srcIdx1)]);

    AppendLocalBufVarOffsetInOrder(dstVar, tmpVar, srcVar0, srcVar1);

    std::ostringstream os;
    std::vector<std::string> paramList;
    paramList.emplace_back(srcDtypeStr0);
    paramList.emplace_back(srcDtypeStr1);

    int dim = dstShape.size(); // 输入输出Tensor维度相同
    for (auto i = 1; i < dim; i++) {
        paramList.emplace_back(std::to_string(dstShape[i]));
    }
    for (auto i = 1; i < dim; i++) {
        paramList.emplace_back(std::to_string(srcShape0[i]));
    }
    for (auto i = 1; i < dim; i++) {
        paramList.emplace_back(std::to_string(srcShape1[i]));
    }

    std::string templateParam = JoinString(paramList, CONN_COMMA);

    paramList.clear();

    std::string addrType = GetAddrTypeByOperandType(BUF_UB);
    paramList.emplace_back("(" + addrType + " " + dstDtypeStr + "*)" + dstVar);
    paramList.emplace_back("(" + addrType + " " + srcDtypeStr0 + "*)" + srcVar0);
    paramList.emplace_back("(" + addrType + " " + srcDtypeStr1 + "*)" + srcVar1);
    paramList.emplace_back("(" + addrType + " " + tmpDtypeStr + "*)" + tmpVar);

    auto dynSrcShape = dynamicValidShape[ToUnderlying(OpIdx::srcIdx0)];
    FillParamWithFullInput(paramList, dynSrcShape);
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);

    os << tileOpName.c_str() << "<" << templateParam << ">"
       << "(" << tiloOpCallParam << ");\n";

    return os.str();
}

std::string CodeGenOpNPU::GenQuantizeOp() const
{
    ASSERT(GenCodeErr::PRINT_MODE_ERROR, isSupportTileTensor) << "Quantize only support TileTensor mode";
    return PrintQuantizeTileTensor();
}

std::string CodeGenOpNPU::PrintQuantizeTileTensor() const
{
    // Note: axis parameter is handled at Operation layer via Transpose
    // TileOp layer only supports axis=-1 (per-row quantization)

    // Determine quantization type based on opcode
    std::string quantType;
    if (opCode == Opcode::OP_QUANTIZE_SYM) {
        quantType = "pto::QuantType::INT8_SYM";
    } else {
        quantType = "pto::QuantType::INT8_ASYM";
    }

    std::ostringstream oss;
    std::vector<std::string> templateParamList;
    templateParamList.emplace_back(quantType);

    // Use GetTileOpParamsWithTmpBuf to put tmpbuf at the end
    std::vector<std::string> paramList = GetTileOpParamsWithTmpBuf({ToUnderlying(MIMOIdx::TMP_IDX)});

    oss << tileOpName;
    oss << WrapParamByAngleBrackets(templateParamList);
    oss << WrapParamByParentheses(paramList);
    oss << STMT_END;

    return oss.str();
}

std::string CodeGenOpNPU::GenDequantizeOp() const
{
    ASSERT(GenCodeErr::PRINT_MODE_ERROR, isSupportTileTensor) << "Dequantize only support TileTensor mode";
    return PrintDequantizeTileTensor();
}

std::string CodeGenOpNPU::PrintDequantizeTileTensor() const
{
    // TDequant always has 4 params: dst, src, scale, offset
    // - INT8 -> FP32: uses TDequant<INT8>
    // - INT16 -> FP32: uses TDequant<INT16>
    // symmetric: offset is all zeros
    // asymmetric: offset is actual zero_points

    // Determine dequant type based on input tensor dtype
    std::string dequantType;
    auto srcDtype = operandDtype[ID1];
    if (srcDtype == DataType::DT_INT8) {
        dequantType = "pto::DequantType::INT8";
    } else if (srcDtype == DataType::DT_INT16) {
        dequantType = "pto::DequantType::INT16";
    } else {
        ASSERT(GenCodeErr::DATA_TYPE_UNSUPPORTED, false) << "PrintDequantizeTileTensor: unsupported input dtype "
                                                         << static_cast<int>(srcDtype) << ", expected INT8 or INT16";
    }

    std::ostringstream oss;
    std::vector<std::string> templateParamList;
    templateParamList.emplace_back(dequantType);

    std::vector<std::string> paramList = GetTileOpParamsByOrder();

    oss << tileOpName;
    oss << WrapParamByAngleBrackets(templateParamList);
    oss << WrapParamByParentheses(paramList);
    oss << STMT_END;

    return oss.str();
}

std::string CodeGenOpNPU::GenQuantMXOp() const
{
    ASSERT(GenCodeErr::PRINT_MODE_ERROR, isSupportTileTensor) << "QuantMX only supports tile tensor codegen.";

    int64_t mode = 0;
    ASSERT(OperErr::ATTRIBUTE_INVALID, GetOpAttr(OpAttributeKey::mxQuantMode, mode))
        << "QuantMX missing required attribute: " << OpAttributeKey::mxQuantMode;
    int64_t axis = 0;
    ASSERT(OperErr::ATTRIBUTE_INVALID, GetOpAttr(OpAttributeKey::mxQuantAxis, axis))
        << "QuantMX missing required attribute: " << OpAttributeKey::mxQuantAxis;
    int64_t performanceMode = 0;
    ASSERT(OperErr::ATTRIBUTE_INVALID, GetOpAttr(OpAttributeKey::mxQuantPerformanceMode, performanceMode))
        << "QuantMX missing required attribute: " << OpAttributeKey::mxQuantPerformanceMode;

    std::vector<int64_t> templateParams = {mode, axis, performanceMode};

    std::ostringstream oss;
    oss << tileOpName;
    oss << WrapParamByAngleBrackets(templateParams);
    oss << WrapParamByParentheses(GetTileOpParamsByOrder());
    oss << STMT_END;
    return oss.str();
}
} // namespace npu::tile_fwk
