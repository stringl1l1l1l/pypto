/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "machine/host/expr_generator.h"

#include <algorithm>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>

#include "tilefwk/error.h"
#include "tilefwk/error_code.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk {

ExprBatchGenerator::ExprBatchGenerator(const std::string& outputDir, int devRootKey, size_t totalExprs)
    : outputDir_(outputDir), devRootKey_(devRootKey), totalExprs_(totalExprs)
{
    CalculateBatches();
}

void ExprBatchGenerator::HeaderFileBegin(std::ostringstream& out) const
{
    out << "#pragma once\n"
        << "#include <cstdint>\n\n"
        << "namespace npu::tile_fwk {\n\n";
    GenerateLinkScript();
}

void ExprBatchGenerator::HeaderFileEnd(std::ostringstream& out) const
{
    std::string headerPath = outputDir_ + "/control_flow_expr_table.h";
    std::ofstream header(headerPath);
    if (!header.is_open()) {
        ASSERT(DevCommonErr::FILE_ERROR, false) << "File batch_expr.h open failed!";
        return;
    }
    out << "\n} // namespace npu::tile_fwk\n";
    header << out.str();
    header.close();
}

// Collect launch-hoistable fixed GetInputShapeDim across all expression tables.
GetInputCse ExprBatchGenerator::CollectGetInputCse(const std::vector<SymbolicExpressionTable*>& tables)
{
    GetInputCse getInputCse;
    std::map<std::string, int> seen; // stable order by key
    for (auto* table : tables) {
        if (table == nullptr) {
            continue;
        }
        for (const auto& expr : table->GetPrimaryExpressionSet()) {
            CollectShapeDimCalls(expr, [&](const RawSymbolicScalarPtr& call) {
                if (!CallOperandsAreFixed(call)) {
                    return;
                }
                const std::string key = SymbolicExpressionTable::BuildExpression(call);
                seen.emplace(key, 0);
            });
        }
    }
    int idx = 0;
    for (const auto& kv : seen) {
        const std::string name = "CSE_sd[" + std::to_string(idx++) + "]";
        getInputCse.keyToName.emplace(kv.first, name);
        getInputCse.ordered.emplace_back(name, kv.first);
    }
    return getInputCse;
}

// Emit stack array + inits inside ControlFlowEntry. Returns true if array was emitted.
bool ExprBatchGenerator::EmitGetInputCseStackInits(std::ostringstream& controlFlowOss, const GetInputCse& getInputCse,
                                                   int indent)
{
    if (getInputCse.ordered.empty()) {
        return false;
    }
    controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "int64_t CSE_sd[" << getInputCse.ordered.size() << "];\n";
    for (size_t i = 0; i < getInputCse.ordered.size(); ++i) {
        controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "CSE_sd[" << i
                       << "] = " << getInputCse.ordered[i].second << ";\n";
    }
    return true;
}

void ExprBatchGenerator::GenerateBatchFile(SymbolicExpressionTable* exprTable, std::ostringstream& controlFlowOss,
                                           std::ostringstream& exprHeaderOss, const std::string& expName,
                                           const OrderedSet<RawSymbolicScalarPtr>& expressions,
                                           std::vector<std::string>& exprSrcFiles, int indent, int devRootKey,
                                           const GetInputCse* getInputCse)
{
    for (auto& batch : batches_) {
        std::string filePath = outputDir_ + "/" + batch.fileName;
        std::ofstream out(filePath);
        if (!out.is_open()) {
            ASSERT(DevCommonErr::FILE_ERROR, false) << "File set_expr open failed!";
            return;
        }
        // Write file header
        out << "#define __TILE_FWK_AICPU__ 1\n"
            << "#include <stdint.h>\n\n"
            << "#include \"" << expName << "\"\n"
            << "#include \"tilefwk/aikernel_data.h\"\n"
            << "#include \"tilefwk/aicpu_runtime.h\"\n"
            << "#include \"tilefwk/aicpu_distributed.h\"\n"
            << "#include \"control_flow_expr_table.h\"\n"
            << "namespace npu::tile_fwk {\n\n"
            << "__attribute__((section(\".pypto.func\")))\n"
            << "void " << batch.functionName
            << "(void *ctx, int64_t *symbolTable, RuntimeCallEntryType runtimeCallList[], DevStartArgsBase "
               "*startArgs, uint64_t *exprList, int64_t *CSE_sd) {\n"
            << "    (void)CSE_sd;\n";
        EmitBatchBody(out, exprTable, expressions, batch.startExprIndex, batch.endExprIndex, getInputCse);
        out << "}\n\n"
            << "} // namespace npu::tile_fwk\n";
        out.close();
        const char* cseArg = (getInputCse != nullptr && !getInputCse->ordered.empty()) ? "CSE_sd" : "nullptr";
        controlFlowOss << std::setw(indent * TABSIZE) << ' ' << batch.functionName
                       << "(ctx, symbolTable, runtimeCallList, startArgs, exprList" << devRootKey << ", " << cseArg
                       << ");\n";
        exprSrcFiles.emplace_back(filePath);
        exprHeaderOss << "void " << batch.functionName
                      << "(void *ctx, int64_t *symbolTable, RuntimeCallEntryType runtimeCallList[], "
                         "DevStartArgsBase *startArgs, uint64_t *exprList, int64_t *CSE_sd);\n";
    }
    return;
}

// Immediate / ARG_* / RUNTIME_* only — no VALUE_*, sym_*, or nested expressions.
bool ExprBatchGenerator::IsFixedGetInputCseOperandLeaf(const RawSymbolicScalarPtr& raw)
{
    if (raw == nullptr) {
        return false;
    }
    if (raw->IsImmediate()) {
        return true;
    }
    if (raw->IsSymbol()) {
        const auto& name = raw->GetSymbolName();
        return CheckArgPrefix(name) || CheckRuntimePrefix(name);
    }
    return false;
}

// All operands after the callee symbol are fixed leaves (see GetInputCse comment).
bool ExprBatchGenerator::CallOperandsAreFixed(const RawSymbolicScalarPtr& call)
{
    if (call == nullptr || !call->IsExpression()) {
        return false;
    }
    const auto& ops = call->GetExpressionOperandList();
    if (ops.size() < 2) {
        return false;
    }
    for (size_t i = 1; i < ops.size(); ++i) {
        if (!IsFixedGetInputCseOperandLeaf(ops[i])) {
            return false;
        }
    }
    return true;
}

bool ExprBatchGenerator::IsGetInputCseTargetCall(const RawSymbolicScalarPtr& raw)
{
    if (raw == nullptr || !raw->IsExpression() || raw->GetExpressionOpcode() != SymbolicOpcode::T_MOP_CALL) {
        return false;
    }
    const auto& ops = raw->GetExpressionOperandList();
    if (ops.empty() || !ops[0]->IsSymbol()) {
        return false;
    }
    const auto& callee = ops[0]->GetSymbolName();
    if (!CallIsGetInputShapeDim(callee) && !CallIsGetInputData(callee)) {
        return false;
    }
    return CallOperandsAreFixed(raw);
}

void ExprBatchGenerator::CollectShapeDimCalls(const RawSymbolicScalarPtr& raw,
                                              const std::function<void(const RawSymbolicScalarPtr&)>& fn)
{
    if (raw == nullptr) {
        return;
    }
    if (raw->IsExpression()) {
        if (raw->GetExpressionOpcode() == SymbolicOpcode::T_MOP_CALL) {
            const auto& ops = raw->GetExpressionOperandList();
            if (!ops.empty() && ops[0]->IsSymbol() && CallIsGetInputShapeDim(ops[0]->GetSymbolName())) {
                fn(raw);
            }
        }
        for (const auto& op : raw->GetExpressionOperandList()) {
            CollectShapeDimCalls(op, fn);
        }
    }
}

void ExprBatchGenerator::WalkGetInputCseTargetCalls(const RawSymbolicScalarPtr& raw,
                                                    const std::function<void(const RawSymbolicScalarPtr&)>& fn)
{
    if (raw == nullptr) {
        return;
    }
    if (raw->IsExpression()) {
        if (IsGetInputCseTargetCall(raw)) {
            fn(raw);
        }
        for (const auto& op : raw->GetExpressionOperandList()) {
            WalkGetInputCseTargetCalls(op, fn);
        }
    }
}

// Batch-local CSE for fixed-arg target calls appearing >= 2 times; merge launch shape dims.
std::unordered_map<std::string, std::string> ExprBatchGenerator::BuildBatchStructuralGetInputCse(
    const OrderedSet<RawSymbolicScalarPtr>& expressions, size_t batchStart, size_t batchEnd,
    const GetInputCse* getInputCse, std::vector<std::pair<std::string, std::string>>& batchLocalOrdered)
{
    std::unordered_map<std::string, int> counts;
    for (size_t i = batchStart; i < batchEnd; ++i) {
        WalkGetInputCseTargetCalls(expressions[static_cast<int>(i)], [&](const RawSymbolicScalarPtr& call) {
            // IsGetInputCseTargetCall already requires fixed operands.
            const std::string key = SymbolicExpressionTable::BuildExpression(call);
            counts[key]++;
        });
    }

    std::unordered_map<std::string, std::string> structural;
    if (getInputCse != nullptr) {
        structural = getInputCse->keyToName;
    }

    int localIdx = 0;
    // Stable order by key string
    std::map<std::string, int> sortedCounts(counts.begin(), counts.end());
    for (const auto& kv : sortedCounts) {
        if (kv.second < 2) {
            continue;
        }
        if (structural.count(kv.first) != 0) {
            continue; // already covered by launch CSE
        }
        const std::string name = "__get_input_cse_" + std::to_string(localIdx++);
        structural.emplace(kv.first, name);
        batchLocalOrdered.emplace_back(name, kv.first);
    }
    return structural;
}

// 用相邻两表达式的立即数差异初始化 tracked；失败返回 false（段长只能为 1）。
bool ExprBatchGenerator::SeedArithmeticTracked(const RawSymbolicScalarPtr& lhs, const RawSymbolicScalarPtr& rhs,
                                               std::vector<TrackedDiff>& tracked)
{
    tracked.clear();
    std::vector<SymbolicExpressionTable::ImmediateDiff> headDiffs;
    if (!SymbolicExpressionTable::FindAllImmediateDifferences(lhs, rhs, headDiffs) || headDiffs.empty()) {
        return false;
    }
    tracked.reserve(headDiffs.size());
    for (const auto& d : headDiffs) {
        tracked.push_back({d.path, d.immLhs, d.immRhs - d.immLhs, d.immRhs});
    }
    std::sort(tracked.begin(), tracked.end(),
              [](const TrackedDiff& a, const TrackedDiff& b) { return a.path < b.path; });
    return true;
}

// 校验 lhs→rhs 是否沿 tracked 的等差路径再前进一步，成功则刷新 lastImm。
bool ExprBatchGenerator::AdvanceArithmeticTracked(const RawSymbolicScalarPtr& lhs, const RawSymbolicScalarPtr& rhs,
                                                  std::vector<TrackedDiff>& tracked)
{
    std::vector<SymbolicExpressionTable::ImmediateDiff> pairDiffs;
    if (!SymbolicExpressionTable::FindAllImmediateDifferences(lhs, rhs, pairDiffs) ||
        pairDiffs.size() != tracked.size()) {
        return false;
    }
    std::sort(pairDiffs.begin(), pairDiffs.end(),
              [](const SymbolicExpressionTable::ImmediateDiff& a, const SymbolicExpressionTable::ImmediateDiff& b) {
                  return a.path < b.path;
              });
    for (size_t i = 0; i < pairDiffs.size(); i++) {
        if (pairDiffs[i].path != tracked[i].path || pairDiffs[i].immLhs != tracked[i].lastImm ||
            pairDiffs[i].immRhs - pairDiffs[i].immLhs != tracked[i].step) {
            return false;
        }
    }
    for (size_t i = 0; i < pairDiffs.size(); i++) {
        tracked[i].lastImm = pairDiffs[i].immRhs;
    }
    return true;
}

// 寻找模板相同、若干立即数同步等差的最长连续段，返回段长（≥1）。
size_t ExprBatchGenerator::DetectArithmeticRun(const OrderedSet<RawSymbolicScalarPtr>& expressions, size_t runStart,
                                               size_t batchEnd, std::vector<TrackedDiff>& tracked) const
{
    tracked.clear();
    if (runStart + 1 >= batchEnd) {
        return 1;
    }
    if (!SeedArithmeticTracked(expressions[runStart], expressions[runStart + 1], tracked)) {
        return 1;
    }

    size_t runLength = 2;
    for (size_t cursor = runStart + 2; cursor < batchEnd; cursor++) {
        if (!AdvanceArithmeticTracked(expressions[cursor - 1], expressions[cursor], tracked)) {
            break;
        }
        runLength++;
    }
    return runLength;
}

// 构造 (firstImm + step * loopVar) 的占位 AST，step==1 / firstImm==0 会化简。
RawSymbolicScalarPtr ExprBatchGenerator::BuildLoopAffinePlaceholder(int64_t firstImm, int64_t step,
                                                                    const RawSymbolicScalarPtr& loopVarNode)
{
    RawSymbolicScalarPtr kPart = loopVarNode;
    if (step != 1) {
        auto stepImm = RawSymbolicImmediate::Create(step);
        kPart = std::make_shared<RawSymbolicExpression>(SymbolicOpcode::T_BOP_MUL,
                                                        std::vector<RawSymbolicScalarPtr>{stepImm, loopVarNode});
    }
    if (firstImm == 0) {
        return kPart;
    }
    auto firstImmNode = RawSymbolicImmediate::Create(firstImm);
    return std::make_shared<RawSymbolicExpression>(SymbolicOpcode::T_BOP_ADD,
                                                   std::vector<RawSymbolicScalarPtr>{firstImmNode, kPart});
}

// 输出折叠后的 for 循环：单差异沿用紧凑头，多差异切到计数头并展开每处 affine 子树。
void ExprBatchGenerator::EmitFoldedLoop(std::ostream& out, const RawSymbolicScalarPtr& firstExpr,
                                        const std::vector<TrackedDiff>& tracked, size_t startExprIdx, size_t runLength,
                                        size_t loopId,
                                        const std::unordered_map<std::string, std::string>* structuralGetInputCse) const
{
    FE_ASSERT(!tracked.empty());
    // sym_ 前缀让 BuildSymbolName 不再附加 VALUE_。
    std::string loopVarName = "sym_expr_loop_k_" + std::to_string(loopId);
    auto loopVarNode = RawSymbolicSymbol::Create(loopVarName);

    std::vector<std::pair<std::vector<int>, RawSymbolicScalarPtr>> replacements;
    replacements.reserve(tracked.size());
    const bool singleDiff = tracked.size() == 1;
    for (const auto& d : tracked) {
        if (singleDiff) {
            replacements.emplace_back(d.path, loopVarNode);
        } else {
            replacements.emplace_back(d.path, BuildLoopAffinePlaceholder(d.firstImm, d.step, loopVarNode));
        }
    }
    std::string loopBodyTemplate = SymbolicExpressionTable::BuildExpressionWithPlaceholders(firstExpr, replacements,
                                                                                            structuralGetInputCse);

    out << "    {\n"
        << "        int64_t __expr_loop_idx = " << static_cast<long long>(startExprIdx) << ";\n";
    if (singleDiff) {
        int64_t firstImm = tracked.front().firstImm;
        int64_t step = tracked.front().step;
        int64_t lastImmValue = firstImm + step * static_cast<int64_t>(runLength - 1);
        out << "        for (int64_t " << loopVarName << " = " << static_cast<long long>(firstImm) << "; "
            << loopVarName << " <= " << static_cast<long long>(lastImmValue) << "; " << loopVarName
            << " += " << static_cast<long long>(step) << ") {\n";
    } else {
        out << "        for (int64_t " << loopVarName << " = 0; " << loopVarName << " < "
            << static_cast<long long>(runLength) << "; " << loopVarName << "++) {\n";
    }
    out << "            RUNTIME_SetExpr(exprList, __expr_loop_idx++, " << loopBodyTemplate << ");\n"
        << "        }\n"
        << "    }\n";
}

// 主驱动：先 CSE 局部，再扫等差段折叠 / 单行展开。
void ExprBatchGenerator::EmitBatchBody(std::ostream& out, SymbolicExpressionTable* exprTable,
                                       const OrderedSet<RawSymbolicScalarPtr>& expressions, size_t batchStart,
                                       size_t batchEnd, const GetInputCse* getInputCse) const
{
    (void)exprTable;
    std::vector<std::pair<std::string, std::string>> batchLocalOrdered;
    auto structural = BuildBatchStructuralGetInputCse(expressions, batchStart, batchEnd, getInputCse,
                                                      batchLocalOrdered);
    for (const auto& item : batchLocalOrdered) {
        out << "    int64_t " << item.first << " = " << item.second << ";\n";
    }
    const std::unordered_map<std::string, std::string>* csePtr = structural.empty() ? nullptr : &structural;

    constexpr size_t MIN_LOOP_LEN = 3;
    size_t exprIdx = batchStart;
    size_t loopId = 0;
    while (exprIdx < batchEnd) {
        std::vector<TrackedDiff> tracked;
        size_t runLength = DetectArithmeticRun(expressions, exprIdx, batchEnd, tracked);
        if (runLength >= MIN_LOOP_LEN) {
            EmitFoldedLoop(out, expressions[exprIdx], tracked, exprIdx, runLength, loopId++, csePtr);
            exprIdx += runLength;
        } else {
            auto exprStr = SymbolicExpressionTable::BuildExpression(expressions[exprIdx], csePtr);
            out << "    RUNTIME_SetExpr(exprList, " << exprIdx << ", " << exprStr << ");\n";
            exprIdx++;
        }
    }
}

void ExprBatchGenerator::CalculateBatches()
{
    size_t numBatches = (totalExprs_ + EXPRS_PER_BATCH - 1) / EXPRS_PER_BATCH;

    for (size_t i = 0; i < numBatches; ++i) {
        ExprBatchInfo batch;
        batch.devRootKey = devRootKey_;
        batch.batchIndex = i;
        batch.startExprIndex = i * EXPRS_PER_BATCH;
        batch.endExprIndex = std::min(batch.startExprIndex + EXPRS_PER_BATCH, totalExprs_);
        batch.totalExprs = totalExprs_;
        batch.fileName = "control_flow_expr_table_" + std::to_string(devRootKey_) + "_" + std::to_string(i) + ".cpp";
        batch.functionName = "SetExprBatch_" + std::to_string(devRootKey_) + "_" + std::to_string(i);
        batches_.emplace_back(batch);
    }
}
void ExprBatchGenerator::GenerateLinkScript() const
{
    std::string scriptFile = outputDir_ + "/merge.link";
    std::ofstream file(scriptFile);
    if (!file.is_open()) {
        ASSERT(DevCommonErr::FILE_ERROR, false) << "File merge.link open failed!";
        return;
    }
    // Orphan .eh_frame is not dumped into the AOT bin, but ld places it at 0x10000
    // and shifts .pypto off the page. The pool base is 4K-aligned, so adrp/lo12
    // then miss .rodata.cst16. ALIGN keeps .pypto on a page boundary.
    file << "SECTIONS\n{\n"
         << "    . = 0x10000;\n"
         << "    .pypto : ALIGN(4096)\n"
         << "    {\n"
         << "        _start = .;\n"
         << "        *(.pypto.entry)\n"
         << "        *(.pypto.func)\n"
         << "        *(.rodata.*)\n"
         << "    }\n"
         << "}\n";
    file.close();
}
size_t totalExprs_;

} // namespace npu::tile_fwk
