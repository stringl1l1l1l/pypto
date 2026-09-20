/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_infer_multi_iter_overlap_pass.cpp
 * \brief Core UT for InferMultiIterOverlap Mark contract (create_root CALL shape only).
 *
 *   1) For→CALL→path→CALL→hidden, disjoint window: Mark writer + rootFunc_
 *   2) overlapping write window: no Mark
 *   3) two loop-enclosed writers on same raw: no Mark
 */

#include "gtest/gtest.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "interface/function/function.h"
#include "interface/function/rebuildable_attribute.h"
#include "interface/operation/attribute.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation.h"
#include "interface/program/program.h"
#include "interface/tensor/irbuilder.h"
#include "interface/tensor/logical_tensor.h"
#include "ir/program.h"
#include "ir/stmt.h"
#include "ir/transforms/passes.h"
#include "tilefwk/symbolic_scalar.h"
#include "tilefwk/tilefwk.h"

using namespace npu::tile_fwk;

namespace {

constexpr int64_t BS = 32;
constexpr int64_t HIDDEN = 64;
constexpr int64_t TILE = 8;

ir::Span Sp() { return ir::Span("test_infer_multi_iter_overlap_pass", 1, 1); }

bool HasNoOverlap(Function* f, int rawMagic)
{
    if (f == nullptr) {
        return false;
    }
    auto* attr = RebuildableAttributeManager::GetInstance().GetAttr<RebuildableMultiIterNoOverlap>(f);
    return attr != nullptr && attr->Has(rawMagic);
}

std::shared_ptr<Function> MakeFunc(const std::string& name, bool entry)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), name + "_magic", name, nullptr);
    func->SetFunctionType(FunctionType::DYNAMIC);
    func->SetGraphType(GraphType::TENSOR_GRAPH);
    func->entry_ = entry;
    func->name_ = name;
    Program::GetInstance().InsertFuncToFunctionMap(func->GetMagicName(), func);
    return func;
}

std::shared_ptr<ir::SeqStmts> MakeForAround(const ir::StmtPtr& bodyStmt, const SymbolicScalar& induction)
{
    auto body = std::make_shared<ir::SeqStmts>(std::vector<ir::StmtPtr>{bodyStmt}, Sp());
    auto zero = SymbolicScalar(0).AsExpr();
    auto stop = SymbolicScalar((BS + TILE - 1) / TILE).AsExpr();
    auto step = SymbolicScalar(1).AsExpr();
    auto forStmt = std::make_shared<ir::ForStmt>(induction.AsVar(), zero, stop, step, std::vector<ir::IterArgPtr>{},
                                                 body, std::vector<ir::VarPtr>{}, Sp());
    return std::make_shared<ir::SeqStmts>(std::vector<ir::StmtPtr>{forStmt}, Sp());
}

// entry For → CALL path → CALL hidden; assembles live in hidden.
struct CallPathCase {
    IRBuilder builder;
    SymbolicScalar i;
    std::shared_ptr<Function> entry;
    std::shared_ptr<Function> root;
    std::shared_ptr<Function> path;
    std::shared_ptr<Function> hidden;
    LogicalTensorPtr dst;
    int raw{0};

    using OffsetFactory = std::function<std::vector<std::vector<SymbolicScalar>>(const SymbolicScalar&)>;

    CallPathCase(const std::string& prefix, const OffsetFactory& makeOffsets) : i(SymbolicScalar("i"))
    {
        Program::GetInstance().Reset();
        RebuildableAttributeManager::GetInstance().Clear();

        entry = MakeFunc(prefix + "_entry", true);
        root = MakeFunc(prefix + "_root", false);
        path = MakeFunc(prefix + "_path", false);
        hidden = MakeFunc(prefix + "_hidden", false);
        hidden->SetHiddenFunction(true);
        path->rootFunc_ = root.get();
        hidden->rootFunc_ = root.get();
        Program::GetInstance().SetCurrentFunction(hidden.get());

        auto src = builder.CreateTensorVar(*hidden, DT_FP32, {TILE, HIDDEN}, TileOpFormat::TILEOP_ND, "src");
        dst = builder.CreateTensorVar(*hidden, DT_FP32, {BS, HIDDEN}, TileOpFormat::TILEOP_ND, "dst");
        hidden->AppendOutcast(dst, 0, 0);
        root->AppendOutcast(dst, 0, 0);
        raw = dst->GetRawMagic();

        std::vector<ir::StmtPtr> hiddenStmts;
        for (const auto& dynOffset : makeOffsets(i)) {
            auto& assemble = hidden->AddRawOperation(Opcode::OP_ASSEMBLE, {src}, {dst}, Sp());
            assemble.SetOpAttribute(std::make_shared<AssembleOpAttribute>(Offset{0, 0}, dynOffset));
            hiddenStmts.push_back(std::static_pointer_cast<const ir::Stmt>(assemble.shared_from_this()));
        }
        hidden->body_ = std::make_shared<ir::SeqStmts>(hiddenStmts, Sp());

        auto& callHidden = path->AddRawOperation(Opcode::OP_CALL, {}, {}, Sp());
        callHidden.SetOpAttribute(hidden->CreateCallOpAttribute({}, {}));
        path->body_ = std::make_shared<ir::SeqStmts>(
            std::vector<ir::StmtPtr>{std::static_pointer_cast<const ir::Stmt>(callHidden.shared_from_this())}, Sp());

        auto& callPath = entry->AddRawOperation(Opcode::OP_CALL, {}, {}, Sp());
        callPath.SetOpAttribute(path->CreateCallOpAttribute({}, {}));
        entry->body_ = MakeForAround(std::static_pointer_cast<const ir::Stmt>(callPath.shared_from_this()), i);
    }

    ir::ProgramPtr Program() const
    {
        return std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{entry, path, hidden, root}, "entry", Sp());
    }
};

} // namespace

class TestInferMultiIterOverlapPass : public testing::Test {
protected:
    void TearDown() override
    {
        RebuildableAttributeManager::GetInstance().Clear();
        Program::GetInstance().Reset();
    }
};

TEST_F(TestInferMultiIterOverlapPass, CallPathMarksHiddenAndRoot)
{
    CallPathCase c("Disjoint", [](const SymbolicScalar& i) {
        return std::vector<std::vector<SymbolicScalar>>{{i * SymbolicScalar(TILE), SymbolicScalar(0)}};
    });
    auto result = pypto::ir::pass::InferMultiIterOverlap()(c.Program());
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(HasNoOverlap(c.hidden.get(), c.raw));
    EXPECT_TRUE(HasNoOverlap(c.root.get(), c.raw)) << "rootFunc_ must carry Mark for encode";
}

TEST_F(TestInferMultiIterOverlapPass, AssembleOverlapLeavesUnmarked)
{
    CallPathCase c("Overlap", [](const SymbolicScalar& i) {
        return std::vector<std::vector<SymbolicScalar>>{{i * SymbolicScalar(TILE / 2), SymbolicScalar(0)}};
    });
    auto result = pypto::ir::pass::InferMultiIterOverlap()(c.Program());
    ASSERT_NE(result, nullptr);
    EXPECT_FALSE(HasNoOverlap(c.hidden.get(), c.raw));
    EXPECT_FALSE(HasNoOverlap(c.root.get(), c.raw));
}

// Both under the same For (via CALL→hidden): immediate [0,0] + induced → two writers → no Mark.
TEST_F(TestInferMultiIterOverlapPass, ImmediateAndInducedUnderSameLoopLeavesUnmarked)
{
    CallPathCase c("ImmAndInduced", [](const SymbolicScalar& i) {
        return std::vector<std::vector<SymbolicScalar>>{
            {SymbolicScalar(0), SymbolicScalar(0)},
            {i * SymbolicScalar(TILE), SymbolicScalar(0)},
        };
    });
    auto result = pypto::ir::pass::InferMultiIterOverlap()(c.Program());
    ASSERT_NE(result, nullptr);
    EXPECT_FALSE(HasNoOverlap(c.hidden.get(), c.raw));
    EXPECT_FALSE(HasNoOverlap(c.root.get(), c.raw));
}
