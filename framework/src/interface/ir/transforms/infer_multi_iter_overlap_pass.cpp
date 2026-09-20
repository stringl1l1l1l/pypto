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
 * \file infer_multi_iter_overlap_pass.cpp
 * \brief After create_root: if the sole hidden ASSEMBLE under a For is disjoint across
 *        iterations (i→i+step), Mark RebuildableMultiIterNoOverlap (writer + rootFunc_).
 *
 * Loop var/step: entry For. Assemble: IsHiddenFunction() only (path glue ignored).
 * Writers keyed by outcast rawMagic across entry.
 */

#include "ir/transforms/passes.h"

#include "interface/configs/config_manager_ng.h"
#include "interface/function/function.h"
#include "interface/function/rebuildable_attribute.h"
#include "interface/operation/attribute.h"
#include "interface/operation/operation.h"
#include "interface/program/program.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/symbolic_scalar.h"
#include "ir/kind_traits.h"
#include "ir/program.h"
#include "ir/stmt.h"
#include "tilefwk/symbolic_scalar.h"

#include <algorithm>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace pypto::ir {
namespace {

using npu::tile_fwk::AssembleOpAttribute;
using npu::tile_fwk::CallOpAttribute;
using npu::tile_fwk::LogicalTensorPtr;
using npu::tile_fwk::Opcode;
using npu::tile_fwk::Operation;
using npu::tile_fwk::Program;
using npu::tile_fwk::RebuildableAttributeManager;
using npu::tile_fwk::RebuildableMultiIterNoOverlap;
using npu::tile_fwk::SymbolicScalar;
using FwkFunction = npu::tile_fwk::Function;

struct LoopContext {
    std::unordered_map<const Var*, SymbolicScalar> stepByVar;
    std::unordered_map<int, const Var*> innermostByRawMagic;
    std::unordered_map<int, std::vector<Operation*>> loopWritersByRaw;
};

FwkFunction& AsFrameworkFunction(const FunctionPtr& func)
{
    return const_cast<FwkFunction&>(*std::static_pointer_cast<const FwkFunction>(func));
}

bool IsAssembleLike(const Operation& op)
{
    return op.GetOpcode() == Opcode::OP_ASSEMBLE || op.GetOpcode() == Opcode::OP_ASSEMBLE_SSA;
}

bool ProvablyTrue(const SymbolicScalar& cond)
{
    SymbolicScalar simplified = cond.Simplify();
    return simplified.ConcreteValid() && simplified.Concrete() != 0;
}

void WalkOffsetSymbols(const std::vector<SymbolicScalar>& offset,
                       const std::function<void(const SymbolicScalar&)>& onSymbol)
{
    std::function<void(const SymbolicScalar&)> walk = [&](const SymbolicScalar& scalar) {
        if (scalar.IsSymbol()) {
            onSymbol(scalar);
            return;
        }
        if (!scalar.IsExpression()) {
            return;
        }
        for (const auto& operand : scalar.Raw()->GetExpressionOperandList()) {
            walk(SymbolicScalar(operand));
        }
    };
    for (const auto& scalar : offset) {
        walk(scalar);
    }
}

bool FindInductionInOffset(const std::vector<SymbolicScalar>& offset, const Var* target, SymbolicScalar& found)
{
    bool ok = false;
    WalkOffsetSymbols(offset, [&](const SymbolicScalar& scalar) {
        if (!ok && scalar.AsVar().get() == target) {
            found = scalar;
            ok = true;
        }
    });
    return ok;
}

bool SeparableUnderInduction(const SymbolicScalar& sym, const SymbolicScalar& step,
                             const std::vector<SymbolicScalar>& offset, const std::vector<SymbolicScalar>& shape)
{
    const VarPtr var = sym.AsVar();
    const std::unordered_map<VarPtr, ExprPtr> bumpMap{{var, (sym + step).AsExpr()}};
    for (size_t dim = 0; dim < offset.size(); ++dim) {
        SymbolicScalar bumped = offset[dim];
        if (ProvablyTrue(bumped.SubstituteVars(bumpMap) - offset[dim] >= shape[dim])) {
            return true;
        }
    }
    return false;
}

bool GetAssembleOffsetShape(const Operation& op, std::vector<SymbolicScalar>& offset,
                            std::vector<SymbolicScalar>& shape)
{
    auto attr = std::static_pointer_cast<AssembleOpAttribute>(op.GetOpAttribute());
    offset = attr->GetToDynOffset();
    if (offset.empty()) {
        return false;
    }
    shape = SymbolicScalar::FromConcrete(op.GetIOperands()[0]->GetShape());
    return offset.size() == shape.size();
}

FwkFunction* ResolveCalleeFunction(const Operation& callOp)
{
    auto attr = std::static_pointer_cast<CallOpAttribute>(callOp.GetOpAttribute());
    if (attr == nullptr || attr->GetCalleeMagicName().empty()) {
        return nullptr;
    }
    return Program::GetInstance().GetFunctionByMagicName(attr->GetCalleeMagicName());
}

void RegisterLoopAssemble(Operation& op, const Var* loopVar, LoopContext& loops)
{
    for (const auto& o : op.GetOOperands()) {
        const int raw = o->GetRawMagic();
        loops.innermostByRawMagic[raw] = loopVar;
        auto& writers = loops.loopWritersByRaw[raw];
        if (std::find(writers.begin(), writers.end(), &op) == writers.end()) {
            writers.push_back(&op);
        }
    }
}

void RegisterFromCallee(FwkFunction& func, const Var* loopVar, LoopContext& loops,
                        std::unordered_set<FwkFunction*>& visited)
{
    if (!visited.insert(&func).second) {
        return;
    }
    if (func.IsHiddenFunction()) {
        for (auto& op : func.Operations(false)) {
            if (IsAssembleLike(op)) {
                RegisterLoopAssemble(op, loopVar, loops);
            }
        }
        return;
    }
    for (auto& op : func.Operations(false)) {
        if (op.GetOpcode() != Opcode::OP_CALL) {
            continue;
        }
        if (auto* callee = ResolveCalleeFunction(op)) {
            RegisterFromCallee(*callee, loopVar, loops, visited);
        }
    }
}

void CollectLoopsAndWriters(const StmtPtr& stmt, const Var* loopVar, LoopContext& loops)
{
    if (stmt == nullptr) {
        return;
    }
    switch (stmt->GetKind()) {
        case ObjectKind::ForStmt: {
            auto forStmt = As<ForStmt>(stmt);
            const Var* forVar = forStmt->loopVar_.get();
            loops.stepByVar[forVar] = SymbolicScalar::FromExpr(forStmt->step_);
            CollectLoopsAndWriters(forStmt->body_, forVar, loops);
            break;
        }
        case ObjectKind::SeqStmts: {
            for (const auto& s : As<SeqStmts>(stmt)->stmts_) {
                CollectLoopsAndWriters(s, loopVar, loops);
            }
            break;
        }
        case ObjectKind::IfStmt: {
            auto ifStmt = As<IfStmt>(stmt);
            CollectLoopsAndWriters(ifStmt->thenBody_, loopVar, loops);
            if (ifStmt->elseBody_.has_value()) {
                CollectLoopsAndWriters(ifStmt->elseBody_.value(), loopVar, loops);
            }
            break;
        }
        case ObjectKind::TensorOpStmt: {
            if (loopVar == nullptr) {
                break;
            }
            auto* op = static_cast<Operation*>(const_cast<Stmt*>(stmt.get()));
            if (op->GetOpcode() != Opcode::OP_CALL) {
                break;
            }
            if (auto* callee = ResolveCalleeFunction(*op)) {
                std::unordered_set<FwkFunction*> visited;
                RegisterFromCallee(*callee, loopVar, loops, visited);
            }
            break;
        }
        default:
            break;
    }
}

bool TensorHasAtomicAdd(const LogicalTensorPtr& tensor)
{
    for (const auto& prod : tensor->GetProducers()) {
        if (prod->GetOpcode() == Opcode::OP_ATOMIC_RMW) {
            return true;
        }
    }
    return false;
}

bool IsComputeDeterminismEnabled()
{
    using npu::tile_fwk::ConfigManagerNg;
    return ConfigManagerNg::GetGlobalConfig<int64_t>("compute_determinism_level") >= 1;
}

void MarkNoOverlap(FwkFunction& writer, int rawMagic)
{
    auto& mgr = RebuildableAttributeManager::GetInstance();
    mgr.GetAttr<RebuildableMultiIterNoOverlap>(&writer)->Mark(rawMagic);
    if (auto* root = writer.GetRootFunction()) {
        mgr.GetAttr<RebuildableMultiIterNoOverlap>(root)->Mark(rawMagic);
    }
}

void InferRawMultiIterOverlap(int rawMagic, const std::vector<Operation*>& writers, const LoopContext& loops)
{
    if (writers.size() != 1) {
        return;
    }
    Operation* op = writers[0];
    if (IsComputeDeterminismEnabled() && TensorHasAtomicAdd(op->GetOOperands()[0])) {
        return;
    }
    const Var* loopVar = loops.innermostByRawMagic.at(rawMagic);
    const SymbolicScalar& step = loops.stepByVar.at(loopVar);

    std::vector<SymbolicScalar> offset;
    std::vector<SymbolicScalar> shape;
    if (!GetAssembleOffsetShape(*op, offset, shape)) {
        return;
    }
    SymbolicScalar inductionSym;
    if (!FindInductionInOffset(offset, loopVar, inductionSym)) {
        return;
    }
    if (!SeparableUnderInduction(inductionSym, step, offset, shape)) {
        return;
    }
    MarkNoOverlap(*op->BelongTo(), rawMagic);
}

} // namespace

Pass pass::InferMultiIterOverlap()
{
    return pass::CreateProgramPass(
        [](const ProgramPtr& irProgram) -> ProgramPtr {
            LoopContext loops;
            for (const auto& [name, funcPtr] : irProgram->functions_) {
                (void)name;
                FwkFunction& f = AsFrameworkFunction(funcPtr);
                if (f.entry_ && funcPtr->body_ != nullptr) {
                    CollectLoopsAndWriters(funcPtr->body_, nullptr, loops);
                }
            }
            for (const auto& [rawMagic, writers] : loops.loopWritersByRaw) {
                InferRawMultiIterOverlap(rawMagic, writers, loops);
            }
            return irProgram;
        },
        "InferMultiIterOverlap");
}

} // namespace pypto::ir
