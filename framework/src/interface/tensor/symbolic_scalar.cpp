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
 * \file symbolic_scalar.cpp
 * \brief
 */

#include "interface/tensor/symbolic_scalar.h"
#include <sys/mman.h>
#include <thread>
#include <sstream>
#include <numeric>
#include <functional>
#include "utils/file_utils.h"
#include "interface/utils/common.h"
#include "tilefwk/pypto_fwk_log.h"
#include "symbolic_scalar_simplify.h"
#include "symbolic_scalar_solver.h"
#include "ir/kind_traits.h"
#include "interface/tensor/irbuilder.h"

constexpr uint64_t IMMEDIATE = 0;
constexpr uint64_t SYMBOL = 1;
constexpr uint64_t EXPRESSION = 2;
constexpr int OPERAND_NUM = 2;
constexpr size_t MIN_EXTREMA_OPERANDS = 2;
namespace npu::tile_fwk {

static std::vector<std::string> SplitExtraCflags(const std::string& extraCflag)
{
    std::vector<std::string> result;
    std::string token;
    char quoteChar = '\0';

    for (size_t i = 0; i < extraCflag.size(); ++i) {
        char c = extraCflag[i];

        if (c == '\\' && i + 1 < extraCflag.size()) {
            token += extraCflag[++i];
            continue;
        }

        if (c == '"' || c == '\'') {
            if (quoteChar == '\0') {
                quoteChar = c;
            } else if (quoteChar == c) {
                quoteChar = '\0';
            } else {
                token += c;
            }
            continue;
        }

        if ((c == ' ' || c == '\t') && quoteChar == '\0') {
            if (!token.empty()) {
                result.push_back(token);
                token.clear();
            }
            continue;
        }

        token += c;
    }

    if (!token.empty()) {
        result.push_back(token);
    }
    return result;
}

static bool IsArmCrossCompiler(const std::string& compiler) { return compiler.find("aarch64") != std::string::npos; }

static void AppendArmMarchFlag(std::vector<std::string>& args, const std::string& compiler)
{
    if (IsArmCrossCompiler(compiler)) {
        args.push_back("-march=armv8-a");
    }
}

std::string CompileSourceCode(const std::string& sourceFilePath, const std::string& gcc, const std::string& extraCflag)
{
    std::string assembleFilePath = sourceFilePath + ".s";
    std::string objectFilePath = sourceFilePath + "_t.o";
    std::string includePath = GetPyptoLibPath() + "/../include/tile_fwk";
    std::string macro = extraCflag.empty() ? "-D__DEVICE__" : "";

    std::vector<std::string> argsGcc = {gcc, "-fPIC", "-fno-stack-protector", "-O2"};
    AppendArmMarchFlag(argsGcc, gcc);
    auto extraFlags = SplitExtraCflags(extraCflag);
    argsGcc.insert(argsGcc.end(), extraFlags.begin(), extraFlags.end());
    if (!macro.empty()) {
        argsGcc.push_back(macro);
    }
    argsGcc.insert(argsGcc.end(), {"-I" + includePath, "-I" + GetPyptoLibPath() + "/include/",
                                   "-I" + includePath + "/tilefwk", "-S", sourceFilePath, "-o", assembleFilePath});

    FE_LOGI(
        "[RunCmd] %s.",
        std::accumulate(argsGcc.begin(), argsGcc.end(), std::string(), [](const std::string& a, const std::string& b) {
            return a.empty() ? b : a + " " + b;
        }).c_str());
    FE_ASSERT(SafeExecCommand(argsGcc) == 0);

    std::vector<std::string> argsAs = {gcc, "-fno-stack-protector", "-O2"};
    AppendArmMarchFlag(argsAs, gcc);
    argsAs.insert(argsAs.end(), {"-c", assembleFilePath, "-o", objectFilePath});
    FE_LOGI(
        "[RunCmd] %s.",
        std::accumulate(argsAs.begin(), argsAs.end(), std::string(), [](const std::string& a, const std::string& b) {
            return a.empty() ? b : a + " " + b;
        }).c_str());
    FE_ASSERT(SafeExecCommand(argsAs) == 0);
    return objectFilePath;
}

std::vector<std::string> ParallelCompile(const std::vector<std::string>& sourceFiles, const std::string& gcc,
                                         const std::string& extraCflag)
{
    std::vector<std::string> objs(sourceFiles.size());
    std::vector<std::thread> threads;
    const size_t maxThreads = 8;
    size_t numThreads = std::min(maxThreads, sourceFiles.size());
    FE_ASSERT(numThreads > 0);
    auto worker = [&sourceFiles, &objs, &gcc, &extraCflag](size_t startIdx, size_t endIdx) {
        for (size_t i = startIdx; i < endIdx; ++i) {
            objs[i] = CompileSourceCode(sourceFiles[i], gcc, extraCflag);
        }
    };

    size_t filesPerThread = sourceFiles.size() / numThreads;
    size_t remainingFiles = sourceFiles.size() % numThreads;
    size_t currentIdx = 0;
    for (size_t i = 0; i < numThreads; ++i) {
        size_t threadFiles = filesPerThread + (i < remainingFiles ? 1 : 0);
        size_t endIdx = currentIdx + threadFiles;
        threads.emplace_back(worker, currentIdx, endIdx);
        currentIdx = endIdx;
    }
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    return objs;
}

std::vector<uint8_t> CompileAndLoadSection(const std::string& code, const std::string& sourceFilePath,
                                           const std::string& aicpuPath, std::vector<std::string>& exprSrcFiles,
                                           const std::string& gcc, const std::string& ld, const std::string& objcopy,
                                           const std::string& sectionName, bool needDump, const std::string& extraCflag)
{
    if (needDump) {
        FILE* fsrc = fopen(sourceFilePath.c_str(), "w");
        if (fsrc == nullptr) {
            FE_LOGE(FeError::BAD_FD, "Fail to open source file %s", sourceFilePath.c_str());
            return {};
        }
        fprintf(fsrc, "%s", code.c_str());
        fclose(fsrc);
    }
    std::string objectFilePath = sourceFilePath + ".o";
    std::vector<std::string> allSourceFiles = {sourceFilePath};
    allSourceFiles.insert(allSourceFiles.end(), exprSrcFiles.begin(), exprSrcFiles.end());
    std::vector<std::string> objs = ParallelCompile(allSourceFiles, gcc, extraCflag);

    std::vector<std::string> argsLd = {ld};
    for (const auto& obj : objs) {
        argsLd.push_back(obj);
    }
    argsLd.insert(argsLd.end(), {"-o", objectFilePath, "-O2", "-T", aicpuPath + "/merge.link"});
    FE_LOGI(
        "[RunCmd] %s",
        std::accumulate(argsLd.begin(), argsLd.end(), std::string(), [](const std::string& a, const std::string& b) {
            return a.empty() ? b : a + " " + b;
        }).c_str());
    FE_ASSERT(SafeExecCommand(argsLd) == 0);

    std::string binaryFilePath = sourceFilePath + ".bin";
    std::vector<std::string> argsObjcopy = {objcopy, "--dump-section", sectionName + "=" + binaryFilePath,
                                            objectFilePath};
    FE_LOGI("[RunCmd] %s",
            std::accumulate(argsObjcopy.begin(), argsObjcopy.end(), std::string(),
                            [](const std::string& a, const std::string& b) { return a.empty() ? b : a + " " + b; })
                .c_str());
    FE_ASSERT(SafeExecCommand(argsObjcopy) == 0);

    FILE* fbin = fopen(binaryFilePath.c_str(), "rb");
    if (fbin == nullptr) {
        FE_LOGE(FeError::BAD_FD, "open binary file name failed");
        return {};
    }
    fseek(fbin, 0, SEEK_END);
    int size = static_cast<int>(ftell(fbin));
    fseek(fbin, 0, SEEK_SET);
    std::vector<uint8_t> binary(size);
    size_t readSize = fread(binary.data(), 1, size, fbin);
    fclose(fbin);
    return (readSize == static_cast<size_t>(size)) ? binary : std::vector<uint8_t>{};
}

void SymbolicExpressionTable::SetElementKeyOnce(const std::string& key)
{
    if (elementKey_.size() == 0) {
        elementKey_ = key;
    } else {
        FE_ASSERT(FeError::INVALID_VAL, elementKey_ == key) << "elementKey_: " << elementKey_ << ", key: " << key;
    }
}

void SymbolicExpressionTable::SetTitleOnce(const std::string& title)
{
    if (title_.size() == 0) {
        title_ = title;
    } else {
        FE_ASSERT(FeError::INVALID_VAL, title_ == title) << "title_: " << title_ << ", title: " << title;
    }
}

std::string SymbolicExpressionTable::BuildExpression(const SymbolicScalar& ss) { return BuildExpression(ss.Raw()); }

std::string SymbolicExpressionTable::BuildExpression(const RawSymbolicScalarPtr& ss)
{
    return BuildExpressionByRaw(ss, {});
}

std::string SymbolicExpressionTable::BuildExpression(const RawSymbolicScalarPtr& ss,
                                                     const std::unordered_map<std::string, std::string>* structuralCse)
{
    return BuildExpressionByRaw(ss, {}, structuralCse);
}

int SymbolicExpressionTable::CompareRaw(const RawSymbolicScalarPtr& lhs, const RawSymbolicScalarPtr& rhs)
{
    if (lhs.get() == rhs.get()) {
        return 0;
    }
    auto kindLhs = static_cast<int>(lhs->Kind());
    auto kindRhs = static_cast<int>(rhs->Kind());
    if (kindLhs != kindRhs) {
        return kindLhs - kindRhs;
    }
    switch (lhs->Kind()) {
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE: {
            auto immLhs = std::static_pointer_cast<RawSymbolicImmediate>(lhs)->Immediate();
            auto immRhs = std::static_pointer_cast<RawSymbolicImmediate>(rhs)->Immediate();
            if (immLhs < immRhs) {
                return -1;
            }
            if (immLhs > immRhs) {
                return 1;
            }
            return 0;
        }
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL: {
            const auto& nameLhs = std::static_pointer_cast<RawSymbolicSymbol>(lhs)->Name();
            const auto& nameRhs = std::static_pointer_cast<RawSymbolicSymbol>(rhs)->Name();
            return nameLhs.compare(nameRhs);
        }
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION: {
            auto exprLhs = std::static_pointer_cast<RawSymbolicExpression>(lhs);
            auto exprRhs = std::static_pointer_cast<RawSymbolicExpression>(rhs);
            auto opLhs = static_cast<int>(exprLhs->Opcode());
            auto opRhs = static_cast<int>(exprRhs->Opcode());
            if (opLhs != opRhs) {
                return opLhs - opRhs;
            }
            const auto& operandsLhs = exprLhs->OperandList();
            const auto& operandsRhs = exprRhs->OperandList();
            if (operandsLhs.size() != operandsRhs.size()) {
                return static_cast<int>(operandsLhs.size()) - static_cast<int>(operandsRhs.size());
            }
            for (size_t operandIdx = 0; operandIdx < operandsLhs.size(); operandIdx++) {
                int sub = CompareRaw(operandsLhs[operandIdx], operandsRhs[operandIdx]);
                if (sub != 0) {
                    return sub;
                }
            }
            return 0;
        }
        default:
            FE_ASSERT(false) << SymbolicScalarKind2Name(lhs->Kind()) << " undefined behavior";
            return 0;
    }
}

namespace {
// 沿一组 path 同时克隆原表达式：路径经过的节点新建副本，旁路子树共享 shared_ptr。
RawSymbolicScalarPtr CloneAlongPathsWithReplacements(
    const RawSymbolicScalarPtr& raw, size_t depth,
    const std::vector<std::pair<std::vector<int>, RawSymbolicScalarPtr>>& replacements)
{
    for (const auto& r : replacements) {
        if (r.first.size() == depth) {
            FE_ASSERT(replacements.size() == 1) << "path collision: multiple replacements target the same leaf";
            FE_ASSERT(raw->Kind() == SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE)
                << "placeholder path must land on an immediate";
            return r.second;
        }
    }
    FE_ASSERT(raw->Kind() == SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION);
    auto expr = std::static_pointer_cast<RawSymbolicExpression>(raw);
    const auto& originalOperands = expr->OperandList();

    std::vector<RawSymbolicScalarPtr> patchedOperands(originalOperands);
    for (size_t opIdx = 0; opIdx < originalOperands.size(); opIdx++) {
        std::vector<std::pair<std::vector<int>, RawSymbolicScalarPtr>> sub;
        for (const auto& r : replacements) {
            FE_ASSERT(depth < r.first.size());
            if (r.first[depth] == static_cast<int>(opIdx)) {
                sub.push_back(r);
            }
        }
        if (!sub.empty()) {
            patchedOperands[opIdx] = CloneAlongPathsWithReplacements(originalOperands[opIdx], depth + 1, sub);
        }
    }
    return std::make_shared<RawSymbolicExpression>(expr->Opcode(), patchedOperands);
}

// CSE map 只收录 GetInputShapeDim / GetInputData 的展开文本。仅这些 CALL 才整节点展开当 key；
// And/Min 等深节点若也走全量 key，会把左链 stringify 做成近似 O(n^3)。
bool IsStructuralCseTargetCall(const RawSymbolicScalarPtr& raw)
{
    if (raw == nullptr || !raw->IsExpression() || raw->GetExpressionOpcode() != SymbolicOpcode::T_MOP_CALL) {
        return false;
    }
    const auto& ops = raw->GetExpressionOperandList();
    if (ops.empty() || !ops[0]->IsSymbol()) {
        return false;
    }
    const auto& callee = ops[0]->GetSymbolName();
    return CallIsGetInputShapeDim(callee) || CallIsGetInputData(callee);
}

} // namespace

std::string SymbolicExpressionTable::BuildExpressionWithPlaceholders(
    const RawSymbolicScalarPtr& raw, const std::vector<std::pair<std::vector<int>, RawSymbolicScalarPtr>>& replacements,
    const std::unordered_map<std::string, std::string>* structuralCse)
{
    FE_ASSERT(!replacements.empty()) << "BuildExpressionWithPlaceholders requires at least one replacement";
    auto patched = CloneAlongPathsWithReplacements(raw, 0, replacements);
    return BuildExpressionByRaw(patched, {}, structuralCse);
}

bool SymbolicExpressionTable::FindAllImmediateDifferences(const RawSymbolicScalarPtr& lhs,
                                                          const RawSymbolicScalarPtr& rhs,
                                                          std::vector<SymbolicExpressionTable::ImmediateDiff>& diffs)
{
    diffs.clear();
    std::vector<int> currentPath;
    return CollectImmediateDifferences(lhs, rhs, currentPath, diffs);
}

bool SymbolicExpressionTable::CollectImmediateDifferences(const RawSymbolicScalarPtr& lhs,
                                                          const RawSymbolicScalarPtr& rhs,
                                                          std::vector<int>& currentPath,
                                                          std::vector<SymbolicExpressionTable::ImmediateDiff>& diffs)
{
    if (lhs.get() == rhs.get()) {
        return true;
    }
    if (lhs->Kind() != rhs->Kind()) {
        return false;
    }
    switch (lhs->Kind()) {
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE: {
            auto immLhs = std::static_pointer_cast<RawSymbolicImmediate>(lhs)->Immediate();
            auto immRhs = std::static_pointer_cast<RawSymbolicImmediate>(rhs)->Immediate();
            if (immLhs != immRhs) {
                diffs.push_back({currentPath, immLhs, immRhs});
            }
            return true;
        }
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL: {
            // 符号名不同则无法折叠（语义会变）。
            const auto& nameLhs = std::static_pointer_cast<RawSymbolicSymbol>(lhs)->Name();
            const auto& nameRhs = std::static_pointer_cast<RawSymbolicSymbol>(rhs)->Name();
            return nameLhs == nameRhs;
        }
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION: {
            auto exprLhs = std::static_pointer_cast<RawSymbolicExpression>(lhs);
            auto exprRhs = std::static_pointer_cast<RawSymbolicExpression>(rhs);
            if (exprLhs->Opcode() != exprRhs->Opcode()) {
                return false;
            }
            const auto& operandsLhs = exprLhs->OperandList();
            const auto& operandsRhs = exprRhs->OperandList();
            if (operandsLhs.size() != operandsRhs.size()) {
                return false;
            }
            for (size_t operandIdx = 0; operandIdx < operandsLhs.size(); operandIdx++) {
                currentPath.push_back(static_cast<int>(operandIdx));
                bool ok = CollectImmediateDifferences(operandsLhs[operandIdx], operandsRhs[operandIdx], currentPath,
                                                      diffs);
                currentPath.pop_back();
                if (!ok) {
                    return false;
                }
            }
            return true;
        }
        default:
            return false;
    }
}

std::string SymbolicExpressionTable::BuildSymbolName(const std::string& name)
{
    if (CheckRuntimePrefix(name) || CheckArgPrefix(name) || name.rfind("sym_", 0) == 0) {
        return name;
    }
    return "VALUE_" + name;
}

std::string SymbolicExpressionTable::BuildExpressionByRaw(
    const RawSymbolicScalarPtr& raw, const std::unordered_map<RawSymbolicScalarPtr, std::string>& exprDict,
    const std::unordered_map<std::string, std::string>* structuralCse)
{
    auto it = exprDict.find(raw);
    if (it != exprDict.end()) {
        return it->second;
    }

    if (structuralCse != nullptr && IsStructuralCseTargetCall(raw)) {
        // Key is the fully expanded form without structural CSE.
        const std::string key = BuildExpressionByRaw(raw, exprDict, nullptr);
        auto sit = structuralCse->find(key);
        if (sit != structuralCse->end()) {
            return sit->second;
        }
    }

    switch (raw->Kind()) {
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE: {
            auto immediate = std::dynamic_pointer_cast<RawSymbolicImmediate>(raw);
            return std::to_string(immediate->Immediate());
        }
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL: {
            auto symbol = std::dynamic_pointer_cast<RawSymbolicSymbol>(raw);
            return BuildSymbolName(symbol->Name());
        }
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION: {
            auto expr = std::dynamic_pointer_cast<RawSymbolicExpression>(raw);
            return BuildExpressionCode(expr, exprDict, structuralCse);
        }
        default:
            FE_ASSERT(false) << SymbolicScalarKind2Name(raw->Kind()) << " undefined behavior";
            return "";
    }
}

void SymbolicExpressionTable::BuildExtremaExpressionCode(
    const RawSymbolicExpPtr& expr, const std::unordered_map<RawSymbolicScalarPtr, std::string>& exprDict,
    std::ostringstream& oss, const std::unordered_map<std::string, std::string>* structuralCse)
{
    const auto& operands = expr->OperandList();
    FE_ASSERT(FeError::INVALID_VAL, operands.size() >= MIN_EXTREMA_OPERANDS)
        << "Extrema expression must have at least 2 operands";
    std::string funcName = (expr->Opcode() == SymbolicOpcode::T_MOP_MAX) ? "RUNTIME_Max" : "RUNTIME_Min";
    const size_t operandSize = operands.size();

    // 写前operandSize-2层: fn(op_i,
    for (size_t i = 0; i < operandSize - 2; ++i) {
        oss << funcName << "(" << BuildExpressionByRaw(operands[i], exprDict, structuralCse) << ", ";
    }

    // 最内层: fn(op_{operandSize-2}, op_{operandSize-1})
    oss << funcName << "(" << BuildExpressionByRaw(operands[operandSize - 2], exprDict, structuralCse) << ", "
        << BuildExpressionByRaw(operands[operandSize - 1], exprDict, structuralCse) << ")";

    // 补齐右括号
    for (size_t i = 0; i < operandSize - 0x2; ++i) {
        oss << ")";
    }
}

std::string SymbolicExpressionTable::BuildExpressionCode(
    const RawSymbolicExpPtr& expr, const std::unordered_map<RawSymbolicScalarPtr, std::string>& exprDict,
    const std::unordered_map<std::string, std::string>* structuralCse)
{
    std::ostringstream oss;
    oss << "(";
    if (SymbolicOpcode::T_UOP_BEGIN <= expr->Opcode() && expr->Opcode() < SymbolicOpcode::T_UOP_END) {
        oss << RawSymbolicExpression::GetSymbolicCalcOpcode(expr->Opcode());
        oss << BuildExpressionByRaw(expr->OperandList()[0], exprDict, structuralCse);
    } else if (RawSymbolicExpression::IsBinaryCalcOpcode(expr->Opcode())) {
        for (size_t idx = 0; idx < expr->OperandList().size(); idx++) {
            if (idx != 0) {
                oss << " " + RawSymbolicExpression::GetSymbolicCalcOpcode(expr->Opcode()) + " ";
            }
            oss << BuildExpressionByRaw(expr->OperandList()[idx], exprDict, structuralCse);
        }
    } else if (expr->Opcode() == SymbolicOpcode::T_MOP_MAX || expr->Opcode() == SymbolicOpcode::T_MOP_MIN) {
        BuildExtremaExpressionCode(expr, exprDict, oss, structuralCse);
    } else if (expr->Opcode() == SymbolicOpcode::T_MOP_CALL) {
        std::string callee = BuildExpressionByRaw(expr->OperandList()[0], exprDict, structuralCse);
        if (CheckRuntimePrefix(callee)) {
            oss << callee;
        } else {
            oss << "((Call" << expr->OperandList().size() << "EntryType)" << callee << ")";
        }
        oss << "(";
        for (size_t idx = 1; idx < expr->OperandList().size(); idx++) {
            oss << (idx == 1 ? "" : ", ");
            oss << BuildExpressionByRaw(expr->OperandList()[idx], exprDict, structuralCse);
        }
        oss << ")";
    }
    oss << ")";
    return oss.str();
}

std::string SymbolicExpressionTable::BuildExpressionList() const
{
    constexpr int INDENT = 0x20;
    std::ostringstream oss;
    std::unordered_map<RawSymbolicScalarPtr, std::string> exprDict;

    oss << "\n";
    oss << "/* Function info " << elementKey_ << ": " << title_ << " */\n";
    for (auto& expr : expressionSet) {
        int index = expressionSet.GetIndex(expr);
        std::string exprNameTempVarInit = GetExprNameTempVarInit(elementKey_, index);
        std::string exprNameCalc = GetExprNameCalc(elementKey_, index);
        std::string exprNameGet = GetExprNameUse(elementKey_, index);
        std::string calc = BuildExpressionByRaw(expr, exprDict);

        if (primaryExpressionSet.count(expr)) {
            oss << "\n";
            oss << "/* Full Expression: " << BuildExpressionByRaw(expr, {}) << " */"
                << "\n";
        }

        oss << "#define " << std::left << std::setw(INDENT) << (exprNameCalc + " ") << calc << "\n";
        oss << "#define " << std::left << std::setw(INDENT) << (exprNameTempVarInit + " ") << "\n";
        oss << "#define " << std::left << std::setw(INDENT) << (exprNameGet + " ") << exprNameCalc << "\n";
        exprDict[expr] = exprNameGet;
    }
    return oss.str();
}

std::string SymbolicExpressionTable::BuildExpressionTempVarInit(int indent)
{
    std::ostringstream oss;
    for (auto& expr : expressionSet) {
        int index = expressionSet.GetIndex(expr);
        std::string exprNameTempVarInit = GetExprNameTempVarInit(elementKey_, index);
        oss << std::setw(indent) << " " << exprNameTempVarInit << ";";
    }
    return oss.str();
}

bool SymbolicExpressionTable::CheckExprDependCore(const RawSymbolicScalarPtr& raw,
                                                  const std::unordered_map<std::string, bool>& tensorNameToDependCore,
                                                  std::unordered_map<RawSymbolicScalarPtr, bool>& valDependMap,
                                                  std::unordered_set<std::string>& valueDependTensorNames)
{
    switch (raw->Kind()) {
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE:
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL:
            return false;
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION: {
            auto expr = std::dynamic_pointer_cast<RawSymbolicExpression>(raw);
            if (expr->Opcode() == SymbolicOpcode::T_MOP_CALL) {
                auto operandList = expr->OperandList();
                if (operandList.size() < 0x2) {
                    return false;
                }
                const auto& calleeExpr = operandList[0];
                if (calleeExpr->Kind() != SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL) {
                    return false;
                }
                const auto iter = valDependMap.find(calleeExpr);
                if (iter != valDependMap.end()) {
                    return iter->second;
                }
                const auto& callee = std::dynamic_pointer_cast<RawSymbolicSymbol>(calleeExpr)->Name();
                if (CallIsGetInputData(callee)) {
                    auto argExpr = operandList[1];
                    const std::string& argName = std::dynamic_pointer_cast<RawSymbolicSymbol>(argExpr)->Name();
                    FE_LOGI("[RunCmd] Value depend tensor name:%s", argName.c_str());
                    valueDependTensorNames.insert(argName);
                    auto it = tensorNameToDependCore.find(argName);
                    FE_ASSERT(FeError::NOT_EXIST, it != tensorNameToDependCore.end())
                        << "Tensor " << argName << " not found in tensorNameToDependCore";
                    valDependMap[calleeExpr] = it->second;
                    return it->second;
                }
            }
            // Recursively check all operands
            for (const auto& operand : expr->OperandList()) {
                if (CheckExprDependCore(operand, tensorNameToDependCore, valDependMap, valueDependTensorNames)) {
                    return true;
                }
            }
            return false;
        }
        default:
            return false;
    }
}

static RawSymbolicScalarPtr AsRawSymbolicScalar(const pypto::ir::ExprPtr& expr)
{
    if (auto val = ir::AsMut<ir::ConstInt>(expr))
        return std::static_pointer_cast<RawSymbolicImmediate>(val);
    if (auto var = ir::AsMut<ir::Var>(expr))
        return std::static_pointer_cast<RawSymbolicSymbol>(var);
    if (auto sexpr = ir::AsMut<ir::ScalarExpr>(expr)) {
        return std::static_pointer_cast<RawSymbolicExpression>(sexpr);
    }
    return nullptr;
}

void RawSymbolicScalar::FlattenOperands(const std::vector<RawSymbolicScalarPtr>& inOperandList,
                                        SymbolicOpcode objOpcode, std::vector<RawSymbolicScalarPtr>& outOperandList)
{
    for (auto& operand : inOperandList) {
        if (!operand) {
            continue;
        }

        if (operand->Kind() == SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION) {
            auto expr = std::static_pointer_cast<RawSymbolicExpression>(operand);
            if (expr->Opcode() == objOpcode) {
                const auto& sub = expr->OperandList();
                outOperandList.insert(outOperandList.end(), sub.begin(), sub.end());
                continue;
            }
        }
        outOperandList.push_back(operand);
    }
}

ScalarImmediateType RawSymbolicScalar::GetImmediateValue() const
{
    FE_ASSERT(FeError::INVALID_TYPE, IsImmediate()) << "Mismatch immediate type: " << SymbolicScalarKind2Name(Kind());
    auto immediate = static_cast<const RawSymbolicImmediate*>(this);
    return immediate->Immediate();
}
const std::string& RawSymbolicScalar::GetSymbolName() const
{
    FE_ASSERT(FeError::INVALID_TYPE, IsSymbol()) << "Mismatch symbol type: " << SymbolicScalarKind2Name(Kind());
    auto symbol = static_cast<const RawSymbolicSymbol*>(this);
    return symbol->Name();
}
SymbolicOpcode RawSymbolicScalar::GetExpressionOpcode() const
{
    FE_ASSERT(FeError::INVALID_TYPE, IsExpression()) << "Mismatch expression type: " << SymbolicScalarKind2Name(Kind());
    auto expression = static_cast<const RawSymbolicExpression*>(this);
    return expression->Opcode();
}
const std::vector<RawSymbolicScalarPtr>& RawSymbolicScalar::GetExpressionOperandList() const
{
    FE_ASSERT(FeError::INVALID_TYPE, IsExpression()) << "Mismatch expression type: " << SymbolicScalarKind2Name(Kind());
    auto expression = static_cast<const RawSymbolicExpression*>(this);
    return expression->OperandList();
}

bool RawSymbolicScalar::IsExpressionCall(const std::string& calleeName) const
{
    if (!IsExpression()) {
        return false;
    }
    if (GetExpressionOpcode() != SymbolicOpcode::T_MOP_CALL) {
        return false;
    }
    auto caller = GetExpressionOperandList()[0];
    if (!caller->IsSymbol()) {
        return false;
    }
    if (caller->GetSymbolName() != calleeName) {
        return false;
    }
    return true;
}

std::string RawSymbolicScalar::Dump() const
{
    std::stringstream buf;
    DumpBuffer(buf);
    return buf.str();
}

static void DumpSymbolicScalar(const RawSymbolicScalarPtr& raw, Json& jarray)
{
    switch (raw->Kind()) {
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE: {
            jarray.emplace_back(IMMEDIATE);
            auto immediate = std::dynamic_pointer_cast<RawSymbolicImmediate>(raw);
            jarray.emplace_back(static_cast<uint64_t>(immediate->Immediate()));
        } break;
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL: {
            jarray.emplace_back(SYMBOL);
            auto symbol = std::dynamic_pointer_cast<RawSymbolicSymbol>(raw);
            jarray.emplace_back(symbol->Name());
        } break;
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION: {
            jarray.emplace_back(EXPRESSION);
            RawSymbolicExpPtr expr = std::dynamic_pointer_cast<RawSymbolicExpression>(raw);
            jarray.emplace_back(static_cast<int32_t>(expr->Opcode()));
            if (expr->Opcode() == SymbolicOpcode::T_MOP_CALL || expr->Opcode() == SymbolicOpcode::T_MOP_MAX ||
                expr->Opcode() == SymbolicOpcode::T_MOP_MIN) {
                jarray.emplace_back(static_cast<int32_t>(expr->OperandList().size()));
            }
            for (auto& op : expr->OperandList()) {
                DumpSymbolicScalar(op, jarray);
            }
        } break;
        default:
            FE_ASSERT(false) << SymbolicScalarKind2Name(raw->Kind()) << " undefined behavior";
            break;
    }
}

Json ToJson(const SymbolicScalar& sval)
{
    Json jdata;
    DumpSymbolicScalar(sval.Raw(), jdata);
    return jdata;
}

static RawSymbolicScalarPtr LoadRawSymbolicScalar(const Json& symbolicJson, int& despos)
{
    RawSymbolicScalarPtr raw;
    SymbolicScalarKind kind = static_cast<SymbolicScalarKind>(symbolicJson[despos++]);
    switch (kind) {
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE: {
            uint64_t immediateData = static_cast<uint64_t>(symbolicJson[despos++]);
            raw = std::static_pointer_cast<RawSymbolicScalar>(std::make_shared<RawSymbolicImmediate>(immediateData));
        } break;
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL: {
            std::string nameData = static_cast<std::string>(symbolicJson[despos++]);
            raw = std::static_pointer_cast<RawSymbolicScalar>(std::make_shared<RawSymbolicSymbol>(nameData));
        } break;
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION: {
            SymbolicOpcode opcode = static_cast<SymbolicOpcode>(symbolicJson[despos++]);
            std::vector<RawSymbolicScalarPtr> operandList;
            if (opcode == SymbolicOpcode::T_MOP_CALL || opcode == SymbolicOpcode::T_MOP_MAX ||
                opcode == SymbolicOpcode::T_MOP_MIN) {
                int size = symbolicJson[despos++];
                for (int i = 0; i < size; i++) {
                    operandList.push_back(LoadRawSymbolicScalar(symbolicJson, despos));
                }
            } else {
                for (int i = 0; i < OPERAND_NUM; i++) {
                    operandList.push_back(LoadRawSymbolicScalar(symbolicJson, despos));
                }
            }
            raw = std::static_pointer_cast<RawSymbolicScalar>(
                std::make_shared<RawSymbolicExpression>(opcode, operandList));
        } break;
        default:
            break;
    }
    return raw;
}

SymbolicScalar LoadSymbolicScalar(const Json& jval)
{
    int pos = 0;
    return SymbolicScalar(LoadRawSymbolicScalar(jval, pos));
}

void SymbolicScalar::AsIntermediateVariable() { raw_->AsIntermediateVariable(); }

bool SymbolicScalar::IsIntermediateVariable() const { return raw_->IsIntermediateVariable(); }

#define SYMBOLIC_SCALAR_DEFINE_UOP(name, uop, rawname) \
    SymbolicScalar SymbolicScalar::name() const        \
    {                                                  \
        auto raw = rawname(raw_);                      \
        if (ConcreteValid()) {                         \
            return SymbolicScalar(uop Concrete());     \
        } else {                                       \
            return SymbolicScalar(raw);                \
        }                                              \
    }
SYMBOLIC_SCALAR_DEFINE_UOP(Pos, +, RawSymbolicExpression::CreateUopPos)
SYMBOLIC_SCALAR_DEFINE_UOP(Neg, -, RawSymbolicExpression::CreateUopNeg)
SYMBOLIC_SCALAR_DEFINE_UOP(Not, !, RawSymbolicExpression::CreateUopNot)
#undef SYMBOLIC_SCALAR_DEFINE_UOP

#define SYMBOLIC_SCALAR_DEFINE_BOP(name, bop, rawname)                    \
    SymbolicScalar SymbolicScalar::name(const SymbolicScalar& sval) const \
    {                                                                     \
        auto raw = rawname(raw_, sval.raw_);                              \
        if (ConcreteValid() && sval.ConcreteValid()) {                    \
            return SymbolicScalar(Concrete() bop sval.Concrete());        \
        } else {                                                          \
            return SymbolicScalar(raw);                                   \
        }                                                                 \
    }

SYMBOLIC_SCALAR_DEFINE_BOP(Add, +, RawSymbolicExpression::CreateBopAdd)
SYMBOLIC_SCALAR_DEFINE_BOP(Sub, -, RawSymbolicExpression::CreateBopSub)
SYMBOLIC_SCALAR_DEFINE_BOP(Mul, *, RawSymbolicExpression::CreateBopMul)
SYMBOLIC_SCALAR_DEFINE_BOP(Div, /, RawSymbolicExpression::CreateBopDiv)
SYMBOLIC_SCALAR_DEFINE_BOP(Mod, %, RawSymbolicExpression::CreateBopMod)
SYMBOLIC_SCALAR_DEFINE_BOP(Eq, ==, RawSymbolicExpression::CreateBopEq)
SYMBOLIC_SCALAR_DEFINE_BOP(Ne, !=, RawSymbolicExpression::CreateBopNe)
SYMBOLIC_SCALAR_DEFINE_BOP(Lt, <, RawSymbolicExpression::CreateBopLt)
SYMBOLIC_SCALAR_DEFINE_BOP(Le, <=, RawSymbolicExpression::CreateBopLe)
SYMBOLIC_SCALAR_DEFINE_BOP(Gt, >, RawSymbolicExpression::CreateBopGt)
SYMBOLIC_SCALAR_DEFINE_BOP(Ge, >=, RawSymbolicExpression::CreateBopGe)
SYMBOLIC_SCALAR_DEFINE_BOP(And, &&, RawSymbolicExpression::CreateBopAnd)
SYMBOLIC_SCALAR_DEFINE_BOP(Or, ||, RawSymbolicExpression::CreateBopOr)
#undef SYMBOLIC_SCALAR_DEFINE_BOP

SymbolicScalar SymbolicScalar::operator()() const
{
    auto raw = RawSymbolicExpression::CreateMopCall(raw_);
    return SymbolicScalar(raw);
}
SymbolicScalar SymbolicScalar::operator()(const SymbolicScalar& arg0) const
{
    std::vector<RawSymbolicScalarPtr> args = {raw_, arg0.raw_};
    auto raw = RawSymbolicExpression::CreateMopCall(args);
    return SymbolicScalar(raw);
}
SymbolicScalar SymbolicScalar::operator()(const SymbolicScalar& arg0, const SymbolicScalar& arg1) const
{
    std::vector<RawSymbolicScalarPtr> args = {raw_, arg0.raw_, arg1.raw_};
    auto raw = RawSymbolicExpression::CreateMopCall(args);
    return SymbolicScalar(raw);
}
SymbolicScalar SymbolicScalar::operator()(const SymbolicScalar& arg0, const SymbolicScalar& arg1,
                                          const SymbolicScalar& arg2) const
{
    std::vector<RawSymbolicScalarPtr> args = {raw_, arg0.raw_, arg1.raw_, arg2.raw_};
    auto raw = RawSymbolicExpression::CreateMopCall(args);
    return SymbolicScalar(raw);
}
SymbolicScalar SymbolicScalar::operator()(const SymbolicScalar& arg0, const SymbolicScalar& arg1,
                                          const SymbolicScalar& arg2, const SymbolicScalar& arg3) const
{
    std::vector<RawSymbolicScalarPtr> args = {raw_, arg0.raw_, arg1.raw_, arg2.raw_, arg3.raw_};
    auto raw = RawSymbolicExpression::CreateMopCall(args);
    return SymbolicScalar(raw);
}
SymbolicScalar SymbolicScalar::operator()(const SymbolicScalar& arg0, const SymbolicScalar& arg1,
                                          const SymbolicScalar& arg2, const SymbolicScalar& arg3,
                                          const SymbolicScalar& arg4) const
{
    std::vector<RawSymbolicScalarPtr> args = {raw_, arg0.raw_, arg1.raw_, arg2.raw_, arg3.raw_, arg4.raw_};
    auto raw = RawSymbolicExpression::CreateMopCall(args);
    return SymbolicScalar(raw);
}

SymbolicScalar SymbolicScalar::operator()(const std::vector<SymbolicScalar>& argList) const
{
    std::vector<RawSymbolicScalarPtr> args = {raw_};
    for (auto& a : argList) {
        args.push_back(a.raw_);
    }
    auto raw = RawSymbolicExpression::CreateMopCall(args);
    return SymbolicScalar(raw);
}

std::string SymbolicScalar::Dump() const
{
    std::stringstream buf;
    if (raw_) {
        raw_->DumpBuffer(buf);
    }
    return buf.str();
}

SymbolicScalar SymbolicScalar::Simplify() const
{
    if (!raw_ || concreteValid_) {
        return *this;
    }
    SymbolicScalarSimplify simplifier;
    auto simplified = simplifier.Simplify(raw_);
    return SymbolicScalar(simplified);
}

bool SymbolicScalar::IsImmediate() const { return raw_ && raw_->IsImmediate(); }
bool SymbolicScalar::IsSymbol() const { return raw_ && raw_->IsSymbol(); }
bool SymbolicScalar::IsExpression() const { return raw_ && raw_->IsExpression(); }

pypto::ir::VarPtr SymbolicScalar::AsVar() const
{
    ASSERT(IsSymbol());
    return std::dynamic_pointer_cast<RawSymbolicSymbol>(raw_);
}

std::shared_ptr<const ir::Var> RawSymbolicSymbol::Clone() const
{
    IRBuilder builder;
    return builder.CreateScalarVar(name_).AsVar();
}

pypto::ir::ExprPtr SymbolicScalar::AsExpr() const
{
    if (IsSymbol()) {
        return std::dynamic_pointer_cast<RawSymbolicSymbol>(raw_);
    } else if (IsImmediate()) {
        return std::dynamic_pointer_cast<RawSymbolicImmediate>(raw_);
    } else {
        return std::dynamic_pointer_cast<RawSymbolicExpression>(raw_);
    }
}

void SymbolicScalar::GetVarRefs(std::unordered_set<const ir::Var*>& out) const
{
    std::function<void(RawSymbolicScalarPtr)> collect;

    collect = [&](const auto& raw) {
        if (!raw || raw->IsImmediate()) {
            return;
        }
        if (raw->IsSymbol()) {
            out.insert(std::dynamic_pointer_cast<RawSymbolicSymbol>(raw).get());
        } else if (raw->IsExpression()) {
            for (const auto& operand : raw->GetExpressionOperandList()) {
                collect(operand);
            }
        }
    };

    collect(raw_);
}

SymbolicScalar SymbolicScalar::SubstituteVars(const std::unordered_map<ir::VarPtr, ir::ExprPtr>& varMap)
{
    std::function<RawSymbolicScalarPtr(const RawSymbolicScalarPtr&)> substitute;
    std::unordered_map<RawSymbolicScalar*, RawSymbolicScalarPtr> exprMap;

    substitute = [&](const RawSymbolicScalarPtr& raw) -> RawSymbolicScalarPtr {
        if (!raw || raw->IsImmediate()) {
            return raw;
        }
        if (raw->IsSymbol()) {
            auto v = std::dynamic_pointer_cast<RawSymbolicSymbol>(raw);
            auto it = varMap.find(v);
            if (it != varMap.end()) {
                return AsRawSymbolicScalar(it->second);
            }
            return raw;
        }

        auto it = exprMap.find(raw.get());
        if (it != exprMap.end()) {
            return it->second;
        }
        auto expr = std::dynamic_pointer_cast<RawSymbolicExpression>(raw);
        bool changed = false;
        std::vector<RawSymbolicScalarPtr> nops;
        for (auto& operand : expr->OperandList()) {
            auto nop = substitute(operand);
            nops.push_back(nop);
            changed = changed || (nop.get() != operand.get());
        }
        if (!changed) {
            return raw;
        }
        expr = std::make_shared<RawSymbolicExpression>(expr->Opcode(), nops);
        exprMap[raw.get()] = expr;
        return expr;
    };

    return SymbolicScalar(substitute(raw_));
}

SymbolicScalar SymbolicScalar::Substitute(
    const std::vector<std::pair<RawSymbolicScalarPtr, RawSymbolicScalarPtr>>& vals)
{
    std::function<RawSymbolicScalarPtr(const RawSymbolicScalarPtr&)> substitute =
        [&](const RawSymbolicScalarPtr& raw) -> RawSymbolicScalarPtr {
        if (!raw) {
            return raw;
        }
        for (auto& [key, val] : vals) {
            if (key == raw) {
                return val;
            }
        }
        if (!raw->IsExpression()) {
            return raw;
        }
        auto expr = std::static_pointer_cast<RawSymbolicExpression>(raw);
        std::vector<RawSymbolicScalarPtr> operands;
        operands.reserve(expr->OperandList().size());
        bool changed = false;
        for (auto& operand : expr->OperandList()) {
            auto substituted = substitute(operand);
            operands.push_back(substituted);
            changed = changed || substituted != operand;
        }
        if (!changed) {
            return raw;
        }
        return std::make_shared<RawSymbolicExpression>(expr->Opcode(), operands);
    };
    return SymbolicScalar(substitute(raw_));
}

SymbolicScalar SymbolicScalar::Min(const SymbolicScalar& sval) const
{
    if (ConcreteValid() && sval.ConcreteValid()) {
        return SymbolicScalar(std::min(Concrete(), sval.Concrete()));
    }
    auto raw = RawSymbolicExpression::CreateMopMin({raw_, sval.raw_});
    return SymbolicScalar(raw);
}

SymbolicScalar SymbolicScalar::Max(const SymbolicScalar& sval) const
{
    if (ConcreteValid() && sval.ConcreteValid()) {
        return SymbolicScalar(std::max(Concrete(), sval.Concrete()));
    }
    auto raw = RawSymbolicExpression::CreateMopMax({raw_, sval.raw_});
    return SymbolicScalar(raw);
}

SymbolicScalar SymbolicScalar::Ternary(const SymbolicScalar& sval1, const SymbolicScalar& sval2) const
{
    std::string ternaryOpName = SymbolHandler::GetNameByHandlerId(SymbolHandlerId::TernaryOP);
    ternaryOpName = AddRuntimePrefix(ternaryOpName);
    SymbolicScalar ternaryOp(ternaryOpName);
    auto result = ternaryOp(raw_, sval1, sval2);
    return result;
}

SymbolicScalar::SymbolicScalar(int64_t value)
    : raw_(RawSymbolicImmediate::Create(value)), concreteValid_(true), concrete_(value)
{}
SymbolicScalar::SymbolicScalar(const std::string& name) : raw_(RawSymbolicSymbol::Create(name)) {}
SymbolicScalar::SymbolicScalar(const std::string& name, int64_t value)
    : raw_(RawSymbolicSymbol::Create(name)), concreteValid_(true), concrete_(value)
{}
SymbolicScalar::SymbolicScalar(RawSymbolicScalarPtr raw) : raw_(raw)
{
    if (raw_->IsImmediate()) {
        concreteValid_ = true;
        concrete_ = std::dynamic_pointer_cast<RawSymbolicImmediate>(raw)->Immediate();
    }
}

SymbolicScalar SymbolicScalar::FromExpr(const pypto::ir::ExprPtr& expr)
{
    return SymbolicScalar(AsRawSymbolicScalar(expr));
}

std::vector<int64_t> SymbolicScalar::Concrete(const std::vector<SymbolicScalar>& scalarList, int64_t defValue)
{
    std::vector<int64_t> concreteList;
    for (auto& s : scalarList) {
        if (s.ConcreteValid()) {
            concreteList.push_back(s.Concrete());
        } else {
            concreteList.push_back(defValue);
        }
    }
    return concreteList;
}

std::vector<SymbolicScalar> SymbolicScalar::FromConcrete(const std::vector<int64_t>& values)
{
    std::vector<SymbolicScalar> result;
    for (auto x : values) {
        result.push_back(SymbolicScalar(x));
    }
    return result;
}

SatStatus SymbolicScalar::Check(const std::vector<SymbolicScalar>& conds)
{
    std::vector<RawSymbolicScalarPtr> raws;
    raws.reserve(conds.size());
    for (const auto& c : conds) {
        raws.push_back(c.Simplify().Raw()); // const-fold first: a & ~a -> 0, etc.
    }
    return SatChecker(raws).Check();
}

std::unordered_map<std::string, ScalarImmediateType>& SymbolicScalarTracker::GetSymbolDict()
{
    static std::unordered_map<std::string, ScalarImmediateType> symbolDict;
    return symbolDict;
}

static void LookupExpressionByOpcode(std::vector<RawSymbolicScalarPtr>& exprList, SymbolicOpcode opcode,
                                     const RawSymbolicScalarPtr& raw)
{
    switch (raw->Kind()) {
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE:
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL:
            break;
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION: {
            if (raw->GetExpressionOpcode() == opcode) {
                exprList.emplace_back(raw);
            }
            for (auto& op : raw->GetExpressionOperandList()) {
                LookupExpressionByOpcode(exprList, opcode, op);
            }
        } break;
        default:
            FE_ASSERT(false) << SymbolicScalarKind2Name(raw->Kind()) << " undefined behavior";
            break;
    }
}

std::vector<RawSymbolicScalarPtr> LookupExpressionByOpcode(const RawSymbolicScalarPtr& value, SymbolicOpcode opcode)
{
    std::vector<RawSymbolicScalarPtr> exprList;
    LookupExpressionByOpcode(exprList, opcode, value);
    return exprList;
}

void RawSymbolicExpression::DumpRuntimeExtrema(std::ostream& out) const
{
    FE_ASSERT(FeError::INVALID_VAL, operandList_.size() >= MIN_EXTREMA_OPERANDS)
        << "DumpRuntimeExtrema expects at least 2 operands, but got " << operandList_.size();
    const char* funcName = (opcode_ == SymbolicOpcode::T_MOP_MAX) ? "RUNTIME_Max" : "RUNTIME_Min";

    const size_t n = operandList_.size();
    for (size_t i = 0; i < n - 0x2; ++i) {
        out << funcName << "(";
        operandList_[i]->DumpBuffer(out);
        out << ", ";
    }

    out << funcName << "(";
    operandList_[n - 0x2]->DumpBuffer(out);
    out << ", ";
    operandList_[n - 1]->DumpBuffer(out);
    out << ")";

    for (size_t i = 0; i < n - 0x2; ++i) {
        out << ")";
    }
}

void RawSymbolicExpression::DumpBuffer(std::ostream& buffer) const
{
    if (SymbolicOpcode::T_UOP_BEGIN <= opcode_ && opcode_ < SymbolicOpcode::T_UOP_END) {
        buffer << "(" << GetSymbolicCalcOpcode(opcode_);
        operandList_[0]->DumpBuffer(buffer);
        buffer << ")";
    } else if (RawSymbolicExpression::IsBinaryCalcOpcode(opcode_)) {
        buffer << "(";
        for (size_t i = 0; i < operandList_.size(); i++) {
            if (i != 0) {
                buffer << GetSymbolicCalcOpcode(opcode_);
            }
            operandList_[i]->DumpBuffer(buffer);
        }
        buffer << ")";
    } else if (opcode_ == SymbolicOpcode::T_MOP_MAX || opcode_ == SymbolicOpcode::T_MOP_MIN) {
        DumpRuntimeExtrema(buffer);
    } else if (opcode_ == SymbolicOpcode::T_MOP_CALL) {
        operandList_[0]->DumpBuffer(buffer);
        buffer << "(";
        for (size_t i = 1; i < operandList_.size(); i++) {
            if (i != 1) {
                buffer << ",";
            }
            operandList_[i]->DumpBuffer(buffer);
        }
        buffer << ")";
    }
}
} // namespace npu::tile_fwk
