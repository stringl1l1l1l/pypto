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
 * \file symbolic_scalar_simplify.cpp
 * \brief Algebraic simplification for SymbolicScalar expressions.
 *
 * Uses PYPTO_TRY_REWRITE pattern matching adapted for RawSymbolicScalar.
 * Visit dispatch follows SymbolicScalar definition (SymbolicScalarKind + SymbolicOpcode).
 */
#pragma once

#include "symbolic_scalar.h"

#include <algorithm>
#include <optional>
#include <utility>

namespace npu::tile_fwk {

// ============================================================================
// Pattern matching infrastructure for RawSymbolicScalar
// ============================================================================

using RawPtr = RawSymbolicScalarPtr;
using Imm = RawSymbolicImmediate;
using Expr = RawSymbolicExpression;

static inline RawPtr MakeConst(ScalarImmediateType v) { return std::make_shared<Imm>(v); }

static inline std::optional<ScalarImmediateType> GetConstVal(const RawPtr& n)
{
    if (n && n->IsImmediate()) {
        return std::static_pointer_cast<Imm>(n)->Immediate();
    }
    return std::nullopt;
}

// Fully-qualified runtime call name of the ternary op, matching SymbolicScalar::Ternary.
inline const std::string& TernaryOpCallName()
{
    static const std::string name = AddRuntimePrefix(SymbolHandler::GetNameByHandlerId(SymbolHandlerId::TernaryOP));
    return name;
}

// ============================================================================
// CRTP Pattern base
// ============================================================================

template <typename Derived>
class SPattern {
    friend Derived;
    SPattern() = default;

public:
    using Nested = Derived;

    template <typename NodeType>
    [[nodiscard]] bool Match(const NodeType& value) const
    {
        derived().InitMatch_();
        return derived().Match_(value);
    }

    template <typename NodeType, typename Condition>
    [[nodiscard]] bool Match(const NodeType& value, Condition cond) const
    {
        derived().InitMatch_();
        return derived().Match_(value) && cond();
    }

    [[nodiscard]] const Derived& derived() const { return *static_cast<const Derived*>(this); }
};

// ============================================================================
// SEqualChecker — equality for pattern dedup
// ============================================================================

class SEqualChecker {
public:
    static bool Equal(const RawPtr& lhs, const RawPtr& rhs)
    {
        if (lhs.get() == rhs.get()) {
            return true;
        }
        if (!lhs || !rhs || lhs->Kind() != rhs->Kind()) {
            return false;
        }
        // Value-based for immediates
        if (lhs->IsImmediate()) {
            return std::static_pointer_cast<Imm>(lhs)->Immediate() == std::static_pointer_cast<Imm>(rhs)->Immediate();
        }
        return lhs->Dump() == rhs->Dump();
    }
};

// ============================================================================
// PVar<RawPtr> — pattern variable matching any RawSymbolicScalar
// ============================================================================

class PVarRaw : public SPattern<PVarRaw> {
public:
    using Nested = const PVarRaw&;

    void InitMatch_() const { filled_ = false; }

    [[nodiscard]] bool Match_(const RawPtr& value) const
    {
        if (!filled_) {
            value_ = value;
            filled_ = true;
            return true;
        }
        return SEqualChecker::Equal(value_, value);
    }

    [[nodiscard]] RawPtr Eval() const { return value_; }

protected:
    mutable RawPtr value_;
    mutable bool filled_{false};
};

// ============================================================================
// PVarImm — pattern variable matching only RawSymbolicImmediate (constants)
// ============================================================================

class PVarImm : public SPattern<PVarImm> {
public:
    using Nested = const PVarImm&;

    void InitMatch_() const { filled_ = false; }

    [[nodiscard]] bool Match_(const RawPtr& value) const
    {
        if (!value || !value->IsImmediate()) {
            return false;
        }
        if (!filled_) {
            value_ = value;
            filled_ = true;
            return true;
        }
        return SEqualChecker::Equal(value_, value);
    }

    [[nodiscard]] RawPtr Eval() const { return value_; }

    [[nodiscard]] ScalarImmediateType Val() const { return std::static_pointer_cast<Imm>(value_)->Immediate(); }

protected:
    mutable RawPtr value_;
    mutable bool filled_{false};
};

// ============================================================================
// SLiteral — wraps a pre-built RawPtr as a pattern literal
// ============================================================================

class SLiteral : public SPattern<SLiteral> {
public:
    explicit SLiteral(RawPtr value) : value_(std::move(value)) {}

    void InitMatch_() const {}
    [[nodiscard]] bool Match_(const RawPtr& value) const { return SEqualChecker::Equal(value_, value); }
    [[nodiscard]] RawPtr Eval() const { return value_; }

private:
    RawPtr value_;
};

// ============================================================================
// SBinaryPattern — matches a binary expression (T_BOP_*) with two sub-patterns
// ============================================================================

template <SymbolicOpcode Opcode, typename TA, typename TB>
class SBinaryPattern : public SPattern<SBinaryPattern<Opcode, TA, TB>> {
public:
    SBinaryPattern(const TA& a, const TB& b) : a_(a), b_(b) {}

    void InitMatch_() const
    {
        a_.InitMatch_();
        b_.InitMatch_();
    }

    [[nodiscard]] bool Match_(const RawPtr& value) const
    {
        if (!value || !value->IsExpression()) {
            return false;
        }
        auto e = std::static_pointer_cast<Expr>(value);
        auto op = e->Opcode();
        // For min/max: match both T_BOP and T_MOP variants
        bool opcodeMatch = (op == Opcode) || (Opcode == SymbolicOpcode::T_MOP_MIN && op == SymbolicOpcode::T_MOP_MIN) ||
                           (Opcode == SymbolicOpcode::T_MOP_MAX && op == SymbolicOpcode::T_MOP_MAX);
        if (!opcodeMatch || e->OperandList().size() != 0x2) {
            return false;
        }
        if (!a_.Match_(e->OperandList()[0])) {
            return false;
        }
        if (!b_.Match_(e->OperandList()[1])) {
            return false;
        }
        return true;
    }

    [[nodiscard]] RawPtr Eval() const { return Expr::Create(Opcode, {a_.Eval(), b_.Eval()}); }

private:
    typename TA::Nested a_;
    typename TB::Nested b_;
};

// ============================================================================
// SUnaryPattern — matches a unary expression (T_UOP_*) with one sub-pattern
// ============================================================================

template <SymbolicOpcode Opcode, typename TA>
class SUnaryPattern : public SPattern<SUnaryPattern<Opcode, TA>> {
public:
    explicit SUnaryPattern(const TA& a) : a_(a) {}

    void InitMatch_() const { a_.InitMatch_(); }

    [[nodiscard]] bool Match_(const RawPtr& value) const
    {
        if (!value || !value->IsExpression()) {
            return false;
        }
        auto e = std::static_pointer_cast<Expr>(value);
        if (e->Opcode() != Opcode || e->OperandList().size() != 1) {
            return false;
        }
        return a_.Match_(e->OperandList()[0]);
    }

    [[nodiscard]] RawPtr Eval() const { return Expr::Create(Opcode, {a_.Eval()}); }

private:
    typename TA::Nested a_;
};

// ============================================================================
// Binary pattern operator overloads
// ============================================================================

template <typename TA, typename TB>
inline SBinaryPattern<SymbolicOpcode::T_BOP_ADD, TA, TB> operator+(const SPattern<TA>& a, const SPattern<TB>& b)
{
    return SBinaryPattern<SymbolicOpcode::T_BOP_ADD, TA, TB>(a.derived(), b.derived());
}

template <typename TA>
inline SBinaryPattern<SymbolicOpcode::T_BOP_ADD, TA, SLiteral> operator+(const SPattern<TA>& a, ScalarImmediateType b)
{
    return SBinaryPattern<SymbolicOpcode::T_BOP_ADD, TA, SLiteral>(a.derived(), SLiteral(MakeConst(b)));
}

template <typename TA>
inline SBinaryPattern<SymbolicOpcode::T_BOP_ADD, SLiteral, TA> operator+(ScalarImmediateType b, const SPattern<TA>& a)
{
    return SBinaryPattern<SymbolicOpcode::T_BOP_ADD, SLiteral, TA>(SLiteral(MakeConst(b)), a.derived());
}

template <typename TA, typename TB>
inline SBinaryPattern<SymbolicOpcode::T_BOP_SUB, TA, TB> operator-(const SPattern<TA>& a, const SPattern<TB>& b)
{
    return SBinaryPattern<SymbolicOpcode::T_BOP_SUB, TA, TB>(a.derived(), b.derived());
}

template <typename TA>
inline SBinaryPattern<SymbolicOpcode::T_BOP_SUB, TA, SLiteral> operator-(const SPattern<TA>& a, ScalarImmediateType b)
{
    return SBinaryPattern<SymbolicOpcode::T_BOP_SUB, TA, SLiteral>(a.derived(), SLiteral(MakeConst(b)));
}

template <typename TA>
inline SBinaryPattern<SymbolicOpcode::T_BOP_SUB, SLiteral, TA> operator-(ScalarImmediateType b, const SPattern<TA>& a)
{
    return SBinaryPattern<SymbolicOpcode::T_BOP_SUB, SLiteral, TA>(SLiteral(MakeConst(b)), a.derived());
}

template <typename TA, typename TB>
inline SBinaryPattern<SymbolicOpcode::T_BOP_MUL, TA, TB> operator*(const SPattern<TA>& a, const SPattern<TB>& b)
{
    return SBinaryPattern<SymbolicOpcode::T_BOP_MUL, TA, TB>(a.derived(), b.derived());
}

template <typename TA>
inline SBinaryPattern<SymbolicOpcode::T_BOP_MUL, TA, SLiteral> operator*(const SPattern<TA>& a, ScalarImmediateType b)
{
    return SBinaryPattern<SymbolicOpcode::T_BOP_MUL, TA, SLiteral>(a.derived(), SLiteral(MakeConst(b)));
}

template <typename TA>
inline SBinaryPattern<SymbolicOpcode::T_BOP_MUL, SLiteral, TA> operator*(ScalarImmediateType b, const SPattern<TA>& a)
{
    return SBinaryPattern<SymbolicOpcode::T_BOP_MUL, SLiteral, TA>(SLiteral(MakeConst(b)), a.derived());
}

// Named binary pattern constructors

// Macro to generate named binary pattern constructors with RawPtr overloads
#define SYM_PATTERN_BINARY_NAMED(FuncName, Opcode)                                                     \
    template <typename TA, typename TB>                                                                \
    inline SBinaryPattern<Opcode, TA, TB> FuncName(const SPattern<TA>& a, const SPattern<TB>& b)       \
    {                                                                                                  \
        return SBinaryPattern<Opcode, TA, TB>(a.derived(), b.derived());                               \
    }                                                                                                  \
    template <typename TA>                                                                             \
    inline SBinaryPattern<Opcode, TA, SLiteral> FuncName(const SPattern<TA>& a, ScalarImmediateType b) \
    {                                                                                                  \
        return SBinaryPattern<Opcode, TA, SLiteral>(a.derived(), SLiteral(MakeConst(b)));              \
    }                                                                                                  \
    template <typename TA>                                                                             \
    inline SBinaryPattern<Opcode, SLiteral, TA> FuncName(ScalarImmediateType b, const SPattern<TA>& a) \
    {                                                                                                  \
        return SBinaryPattern<Opcode, SLiteral, TA>(SLiteral(MakeConst(b)), a.derived());              \
    }                                                                                                  \
    template <typename TA>                                                                             \
    inline SBinaryPattern<Opcode, TA, SLiteral> FuncName(const SPattern<TA>& a, const RawPtr& b)       \
    {                                                                                                  \
        return SBinaryPattern<Opcode, TA, SLiteral>(a.derived(), SLiteral(b));                         \
    }                                                                                                  \
    template <typename TA>                                                                             \
    inline SBinaryPattern<Opcode, SLiteral, TA> FuncName(const RawPtr& b, const SPattern<TA>& a)       \
    {                                                                                                  \
        return SBinaryPattern<Opcode, SLiteral, TA>(SLiteral(b), a.derived());                         \
    }

SYM_PATTERN_BINARY_NAMED(sym_div, SymbolicOpcode::T_BOP_DIV)
SYM_PATTERN_BINARY_NAMED(sym_mod, SymbolicOpcode::T_BOP_MOD)
SYM_PATTERN_BINARY_NAMED(sym_min, SymbolicOpcode::T_MOP_MIN)
SYM_PATTERN_BINARY_NAMED(sym_max, SymbolicOpcode::T_MOP_MAX)

// Comparison pattern constructors
SYM_PATTERN_BINARY_NAMED(sym_eq, SymbolicOpcode::T_BOP_EQ)
SYM_PATTERN_BINARY_NAMED(sym_ne, SymbolicOpcode::T_BOP_NE)
SYM_PATTERN_BINARY_NAMED(sym_lt, SymbolicOpcode::T_BOP_LT)
SYM_PATTERN_BINARY_NAMED(sym_le, SymbolicOpcode::T_BOP_LE)
SYM_PATTERN_BINARY_NAMED(sym_gt, SymbolicOpcode::T_BOP_GT)
SYM_PATTERN_BINARY_NAMED(sym_ge, SymbolicOpcode::T_BOP_GE)

// Logical and/or pattern constructors
SYM_PATTERN_BINARY_NAMED(sym_and, SymbolicOpcode::T_MOP_AND)
SYM_PATTERN_BINARY_NAMED(sym_or, SymbolicOpcode::T_MOP_OR)

// Unary pattern: neg()
template <typename TA>
inline SUnaryPattern<SymbolicOpcode::T_UOP_NEG, TA> sym_neg(const SPattern<TA>& a)
{
    return SUnaryPattern<SymbolicOpcode::T_UOP_NEG, TA>(a.derived());
}

// Unary pattern: not() via operator!
template <typename TA>
inline SUnaryPattern<SymbolicOpcode::T_UOP_NOT, TA> operator!(const SPattern<TA>& a)
{
    return SUnaryPattern<SymbolicOpcode::T_UOP_NOT, TA>(a.derived());
}

// Pattern eval helper
template <typename T>
inline auto SPatternEval(T&& val) -> decltype(val.Eval())
{
    return val.Eval();
}

inline RawPtr SPatternEval(const RawPtr& val) { return val; }

// ============================================================================
// TRY_REWRITE macros
// ============================================================================

#define SYM_TRY_REWRITE(SrcExpr, ResExpr) \
    if ((SrcExpr).Match(ret)) {           \
        auto r = SPatternEval(ResExpr);   \
        return RecursiveRewrite(r);       \
    }

#define SYM_TRY_REWRITE_IF(SrcExpr, ResExpr, Cond) \
    if ((SrcExpr).Match(ret)) {                    \
        if (Cond) {                                \
            auto r = SPatternEval(ResExpr);        \
            return RecursiveRewrite(r);            \
        }                                          \
    }

// ============================================================================
// SymbolicScalarSimplify
// ============================================================================

class SymbolicScalarSimplify {
public:
    SymbolicScalarSimplify() = default;

    RawPtr Simplify(const RawPtr& node) { return Visit(node); }

private:
    static constexpr int kMaxRecursiveDepth = 5;
    int recursive_depth_ = 0;

    RawPtr RecursiveRewrite(const RawPtr& node)
    {
        if (recursive_depth_ >= kMaxRecursiveDepth || !node) {
            return node;
        }
        ++recursive_depth_;
        RawPtr result = Visit(node);
        --recursive_depth_;
        return result;
    }

    RawPtr Visit(const RawPtr& node)
    {
        if (!node) {
            return node;
        }
        switch (node->Kind()) {
            case SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE:
                return node;
            case SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL:
                return node;
            case SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION:
                return VisitExpression(node);
            default:
                return node;
        }
    }

    RawPtr VisitExpression(const RawPtr& node)
    {
        auto e = std::static_pointer_cast<Expr>(node);
        auto& ops = e->OperandList();

        // Recursively simplify all operands
        std::vector<RawPtr> newOps;
        newOps.reserve(ops.size());
        for (auto& op : ops) {
            newOps.push_back(Visit(op));
        }

        // Constant folding
        if (Expr::AllImmediate(newOps)) {
            auto immList = Expr::ToImmediateList(newOps);
            return MakeConst(Expr::FoldAllImmediate(e->Opcode(), immList));
        }

        // RUNTIME_TernaryOP(c, lhs, rhs) with a decided condition folds to the selected operand
        if (newOps.size() == 0x4 && e->IsExpressionCall(TernaryOpCallName())) {
            if (auto condVal = GetConstVal(newOps[0x1])) {
                return condVal.value() != 0 ? newOps[0x2] : newOps[0x3];
            }
        }

        SymbolicOpcode opcode = e->Opcode();
        // Unary ops
        if (opcode == SymbolicOpcode::T_UOP_NEG) {
            return VisitNeg(newOps[0]);
        }
        if (opcode == SymbolicOpcode::T_UOP_POS) {
            return newOps[0];
        }
        if (opcode == SymbolicOpcode::T_UOP_NOT) {
            return VisitNot(newOps[0]);
        }

        // Binary ops
        if (opcode == SymbolicOpcode::T_BOP_ADD) {
            return VisitAdd(newOps[0], newOps[1]);
        }
        if (opcode == SymbolicOpcode::T_BOP_SUB) {
            return VisitSub(newOps[0], newOps[1]);
        }
        if (opcode == SymbolicOpcode::T_BOP_MUL) {
            return VisitMul(newOps[0], newOps[1]);
        }
        if (opcode == SymbolicOpcode::T_BOP_DIV) {
            return VisitDiv(newOps[0], newOps[1]);
        }
        if (opcode == SymbolicOpcode::T_BOP_MOD) {
            return VisitMod(newOps[0], newOps[1]);
        }
        if (opcode == SymbolicOpcode::T_MOP_MIN) {
            return VisitMin(newOps[0], newOps[1]);
        }
        if (opcode == SymbolicOpcode::T_MOP_MAX) {
            return VisitMax(newOps[0], newOps[1]);
        }

        // Comparisons
        if (opcode == SymbolicOpcode::T_BOP_EQ) {
            return VisitEq(newOps[0], newOps[1]);
        }
        if (opcode == SymbolicOpcode::T_BOP_NE) {
            return VisitNe(newOps[0], newOps[1]);
        }
        if (opcode == SymbolicOpcode::T_BOP_LT) {
            return VisitLt(newOps[0], newOps[1]);
        }
        if (opcode == SymbolicOpcode::T_BOP_LE) {
            return VisitLe(newOps[0], newOps[1]);
        }
        // Gt/Ge delegate
        if (opcode == SymbolicOpcode::T_BOP_GT) {
            return VisitLt(newOps[1], newOps[0]);
        }
        if (opcode == SymbolicOpcode::T_BOP_GE) {
            return VisitLe(newOps[1], newOps[0]);
        }

        // Logical and/or
        if (opcode == SymbolicOpcode::T_MOP_AND) {
            return VisitAnd(newOps[0], newOps[1]);
        }
        if (opcode == SymbolicOpcode::T_MOP_OR) {
            return VisitOr(newOps[0], newOps[1]);
        }

        // MOP min/max with 2 operands — delegate to binary min/max visitors
        if (opcode == SymbolicOpcode::T_MOP_MIN && newOps.size() == 0x2) {
            return VisitMin(newOps[0], newOps[1]);
        }
        if (opcode == SymbolicOpcode::T_MOP_MAX && newOps.size() == 0x2) {
            return VisitMax(newOps[0], newOps[1]);
        }

        return Expr::Create(opcode, newOps);
    }

    // ========================================================================
    // Neg
    // ========================================================================

    RawPtr VisitNeg(const RawPtr& a)
    {
        RawPtr ret = Expr::CreateUopNeg(a);
        PVarRaw x, y;

        // neg(neg(x)) => x
        SYM_TRY_REWRITE(sym_neg(sym_neg(x)), x);
        // neg(x - y) => y - x
        SYM_TRY_REWRITE(sym_neg(x - y), y - x);

        return ret;
    }

    // ========================================================================
    // Not
    // ========================================================================

    RawPtr VisitNot(const RawPtr& a)
    {
        RawPtr ret = Expr::CreateUopNot(a);
        PVarRaw x, y;

        // not(not(x)) => x
        SYM_TRY_REWRITE(!(!x), x);
        // Comparison negation
        SYM_TRY_REWRITE(!(sym_lt(x, y)), sym_le(y, x));
        SYM_TRY_REWRITE(!(sym_le(x, y)), sym_lt(y, x));
        SYM_TRY_REWRITE(!(sym_eq(x, y)), sym_ne(x, y));
        SYM_TRY_REWRITE(!(sym_ne(x, y)), sym_eq(x, y));
        // De Morgan's laws
        SYM_TRY_REWRITE(!(sym_and(x, y)), sym_or(!x, !y));
        SYM_TRY_REWRITE(!(sym_or(x, y)), sym_and(!x, !y));

        return ret;
    }

    // ========================================================================
    // Add
    // ========================================================================

    RawPtr VisitAdd(const RawPtr& a, const RawPtr& b)
    {
        RawPtr ret = Expr::CreateBopAdd(a, b);
        PVarRaw x, y, z;
        PVarImm c1, c2;

        // Constant reassociation
        SYM_TRY_REWRITE((x + c1) + c2, x + (c1.Val() + c2.Val()));
        SYM_TRY_REWRITE((c1 + x) + c2, x + (c1.Val() + c2.Val()));

        // Cancellation
        SYM_TRY_REWRITE((x - y) + y, x);
        SYM_TRY_REWRITE(x + (y - x), y);
        SYM_TRY_REWRITE((x - y) + (y - z), x - z);
        SYM_TRY_REWRITE((x - y) + (z - x), z - y);

        // Coefficient folding
        SYM_TRY_REWRITE(x + x, x * 2);
        SYM_TRY_REWRITE_IF(x * y + x, (y + 1) * x, !x.Eval()->IsImmediate());
        SYM_TRY_REWRITE_IF(x + x * y, (y + 1) * x, !x.Eval()->IsImmediate());
        SYM_TRY_REWRITE_IF(y * x + x, (y + 1) * x, !x.Eval()->IsImmediate());
        SYM_TRY_REWRITE_IF(x + y * x, (y + 1) * x, !x.Eval()->IsImmediate());
        SYM_TRY_REWRITE_IF(x * y + x * z, (y + z) * x, !x.Eval()->IsImmediate());
        SYM_TRY_REWRITE_IF(y * x + x * z, (y + z) * x, !x.Eval()->IsImmediate());
        SYM_TRY_REWRITE_IF(x * y + z * x, (y + z) * x, !x.Eval()->IsImmediate());
        SYM_TRY_REWRITE_IF(y * x + z * x, (y + z) * x, !x.Eval()->IsImmediate());

        // Min/Max interactions
        SYM_TRY_REWRITE(sym_min(x, y - z) + z, sym_min(x + z, y));
        SYM_TRY_REWRITE(sym_min(x - z, y) + z, sym_min(x, y + z));
        SYM_TRY_REWRITE(z + sym_min(x, y - z), sym_min(x + z, y));
        SYM_TRY_REWRITE(sym_max(x, y - z) + z, sym_max(x + z, y));
        SYM_TRY_REWRITE(sym_max(x - z, y) + z, sym_max(x, y + z));
        SYM_TRY_REWRITE(z + sym_max(x, y - z), sym_max(x + z, y));

        // max(x,y) + min(x,y) => x + y
        SYM_TRY_REWRITE(sym_max(x, y) + sym_min(x, y), x + y);
        SYM_TRY_REWRITE(sym_min(x, y) + sym_max(x, y), x + y);
        SYM_TRY_REWRITE(sym_max(x, y) + sym_min(y, x), x + y);
        SYM_TRY_REWRITE(sym_min(x, y) + sym_max(y, x), x + y);

        // Canonicalization: constants to right
        SYM_TRY_REWRITE(c1 + x, x + c1);
        SYM_TRY_REWRITE(x + (c1 - y), (x - y) + c1);
        SYM_TRY_REWRITE((c1 - y) + x, (x - y) + c1);
        SYM_TRY_REWRITE((x + c1) + y, (x + y) + c1);

        return ret;
    }

    // ========================================================================
    // Sub
    // ========================================================================

    RawPtr VisitSub(const RawPtr& a, const RawPtr& b)
    {
        RawPtr ret = Expr::CreateBopSub(a, b);
        PVarRaw x, y, z;
        PVarImm c1, c2;

        // Self-subtraction
        SYM_TRY_REWRITE(x - x, MakeConst(0));
        SYM_TRY_REWRITE(x - 0, x);
        SYM_TRY_REWRITE(0 - x, sym_neg(x));

        // Cancellation
        SYM_TRY_REWRITE((x + y) - y, x);
        SYM_TRY_REWRITE((y + x) - x, y);
        SYM_TRY_REWRITE((x + y) - x, y);
        SYM_TRY_REWRITE((y + x) - y, x);
        SYM_TRY_REWRITE(x - (x + y), sym_neg(y));
        SYM_TRY_REWRITE(x - (y + x), sym_neg(y));

        // Constant reassociation
        SYM_TRY_REWRITE((x + c1) - c2, x + (c1.Val() - c2.Val()));
        SYM_TRY_REWRITE((c1 + x) - c2, x + (c1.Val() - c2.Val()));
        SYM_TRY_REWRITE((c1 - x) - c2, (c1.Val() - c2.Val()) - x);
        SYM_TRY_REWRITE((x - c1) - c2, x - (c1.Val() + c2.Val()));
        SYM_TRY_REWRITE(c1 - (c2 - x), x + (c1.Val() - c2.Val()));
        SYM_TRY_REWRITE(c1 - (x + c2), (c1.Val() - c2.Val()) - x);
        SYM_TRY_REWRITE(c1 - (c2 + x), (c1.Val() - c2.Val()) - x);
        SYM_TRY_REWRITE((c1 - x) - (c2 - y), (y - x) + (c1.Val() - c2.Val()));

        // Cross-subtraction cancellation
        SYM_TRY_REWRITE((x - y) - (x - z), z - y);
        SYM_TRY_REWRITE((x + y) - (x + z), y - z);
        SYM_TRY_REWRITE((y + x) - (x + z), y - z);
        SYM_TRY_REWRITE((x + y) - (z + x), y - z);
        SYM_TRY_REWRITE((y + x) - (z + x), y - z);

        // Coefficient folding
        SYM_TRY_REWRITE(x * y - x, (y - 1) * x);
        SYM_TRY_REWRITE(y * x - x, (y - 1) * x);
        SYM_TRY_REWRITE(x - x * y, (1 - y) * x);
        SYM_TRY_REWRITE(x - y * x, (1 - y) * x);
        SYM_TRY_REWRITE(x * y - x * z, (y - z) * x);
        SYM_TRY_REWRITE(y * x - x * z, (y - z) * x);
        SYM_TRY_REWRITE(x * y - z * x, (y - z) * x);
        SYM_TRY_REWRITE(y * x - z * x, (y - z) * x);

        // Min/Max interactions
        SYM_TRY_REWRITE(sym_min(x, y) - x, sym_min(MakeConst(0), y - x));
        SYM_TRY_REWRITE(sym_min(x, y) - y, sym_min(x - y, MakeConst(0)));
        SYM_TRY_REWRITE(sym_max(x, y) - x, sym_max(MakeConst(0), y - x));
        SYM_TRY_REWRITE(sym_max(x, y) - y, sym_max(x - y, MakeConst(0)));
        SYM_TRY_REWRITE(x - sym_min(x, y), sym_max(MakeConst(0), x - y));
        SYM_TRY_REWRITE(x - sym_max(x, y), sym_min(MakeConst(0), x - y));

        // min/max - factoring out x
        SYM_TRY_REWRITE(sym_min(x + y, z) - x, sym_min(y, z - x));
        SYM_TRY_REWRITE(sym_min(y + x, z) - x, sym_min(y, z - x));
        SYM_TRY_REWRITE(sym_min(z, x + y) - x, sym_min(z - x, y));
        SYM_TRY_REWRITE(sym_min(z, y + x) - x, sym_min(z - x, y));
        SYM_TRY_REWRITE(sym_max(x + y, z) - x, sym_max(y, z - x));
        SYM_TRY_REWRITE(sym_max(y + x, z) - x, sym_max(y, z - x));
        SYM_TRY_REWRITE(sym_max(z, x + y) - x, sym_max(z - x, y));
        SYM_TRY_REWRITE(sym_max(z, y + x) - x, sym_max(z - x, y));

        // min(x,y) - min(y,x) => 0
        SYM_TRY_REWRITE(sym_min(x, y) - sym_min(y, x), MakeConst(0));
        SYM_TRY_REWRITE(sym_max(x, y) - sym_max(y, x), MakeConst(0));

        // Canonicalization
        SYM_TRY_REWRITE(x - (y + c1), (x - y) + (0 - c1.Val()));
        SYM_TRY_REWRITE((x + c1) - y, (x - y) + c1);
        SYM_TRY_REWRITE(x - (y - z), (x + z) - y);

        return ret;
    }

    // ========================================================================
    // Mul
    // ========================================================================

    RawPtr VisitMul(const RawPtr& a, const RawPtr& b)
    {
        RawPtr ret = Expr::CreateBopMul(a, b);
        PVarRaw x, y;
        PVarImm c1, c2;

        // Associativity with constants
        SYM_TRY_REWRITE((x * c1) * c2, x * (c1.Val() * c2.Val()));
        SYM_TRY_REWRITE((c1 * x) * c2, x * (c1.Val() * c2.Val()));

        // min(x,y) * max(x,y) => x * y
        SYM_TRY_REWRITE(sym_min(x, y) * sym_max(x, y), x * y);
        SYM_TRY_REWRITE(sym_max(x, y) * sym_min(x, y), x * y);

        // Canonicalize: const to right
        SYM_TRY_REWRITE(c1 * x, x * c1);
        SYM_TRY_REWRITE(x * (c1 * y), (x * y) * c1);

        // Distributive law
        SYM_TRY_REWRITE((x + c1) * c2, x * c2 + c1.Val() * c2.Val());

        // Flip negative constant
        SYM_TRY_REWRITE_IF((x - y) * c1, (y - x) * (0 - c1.Val()), c1.Val() < 0);

        return ret;
    }

    // ========================================================================
    // Div
    // ========================================================================

    RawPtr VisitDiv(const RawPtr& a, const RawPtr& b)
    {
        RawPtr ret = Expr::CreateBopDiv(a, b);
        PVarRaw x, y, z;
        PVarImm c1, c2;

        // x * c1 / c2 => x * (c1/c2) when c1 % c2 == 0 and c2 > 0
        SYM_TRY_REWRITE_IF(sym_div(x * c1, c2), x * (c1.Val() / c2.Val()), c2.Val() > 0 && c1.Val() % c2.Val() == 0);
        SYM_TRY_REWRITE_IF(sym_div(c1 * x, c2), x * (c1.Val() / c2.Val()), c2.Val() > 0 && c1.Val() % c2.Val() == 0);

        // floordiv(floordiv(x, c1), c2) => floordiv(x, c1*c2) when c1 > 0 and c2 > 0
        SYM_TRY_REWRITE_IF(sym_div(sym_div(x, c1), c2), sym_div(x, c1.Val() * c2.Val()), c1.Val() > 0 && c2.Val() > 0);

        // floordiv(x + c1, c2) => floordiv(x, c2) + c1/c2 when c2>0 and c1%c2==0
        SYM_TRY_REWRITE_IF(sym_div(x + c1, c2), sym_div(x, c2) + (c1.Val() / c2.Val()),
                           c2.Val() > 0 && c1.Val() % c2.Val() == 0);
        SYM_TRY_REWRITE_IF(sym_div(c1 + x, c2), sym_div(x, c2) + (c1.Val() / c2.Val()),
                           c2.Val() > 0 && c1.Val() % c2.Val() == 0);

        // floordiv(x * c1 + y, c2) => x * (c1/c2) + floordiv(y, c2)
        SYM_TRY_REWRITE_IF(sym_div(x * c1 + y, c2), x * (c1.Val() / c2.Val()) + sym_div(y, c2),
                           c2.Val() > 0 && c1.Val() % c2.Val() == 0);
        SYM_TRY_REWRITE_IF(sym_div(c1 * x + y, c2), x * (c1.Val() / c2.Val()) + sym_div(y, c2),
                           c2.Val() > 0 && c1.Val() % c2.Val() == 0);
        SYM_TRY_REWRITE_IF(sym_div(y + x * c1, c2), sym_div(y, c2) + x * (c1.Val() / c2.Val()),
                           c2.Val() > 0 && c1.Val() % c2.Val() == 0);
        SYM_TRY_REWRITE_IF(sym_div(y + c1 * x, c2), sym_div(y, c2) + x * (c1.Val() / c2.Val()),
                           c2.Val() > 0 && c1.Val() % c2.Val() == 0);

        return ret;
    }

    // ========================================================================
    // Mod
    // ========================================================================

    RawPtr VisitMod(const RawPtr& a, const RawPtr& b)
    {
        RawPtr ret = Expr::CreateBopMod(a, b);
        PVarRaw x, y;
        PVarImm c1, c2;

        // mod(x * c1, c2) => 0 when c2 > 0 and c1 % c2 == 0
        SYM_TRY_REWRITE_IF(sym_mod(x * c1, c2), MakeConst(0), c2.Val() > 0 && c1.Val() % c2.Val() == 0);
        SYM_TRY_REWRITE_IF(sym_mod(c1 * x, c2), MakeConst(0), c2.Val() > 0 && c1.Val() % c2.Val() == 0);

        // mod(x + c1, c2) => mod(x, c2) when c2 > 0 and c1 % c2 == 0
        SYM_TRY_REWRITE_IF(sym_mod(x + c1, c2), sym_mod(x, c2), c2.Val() > 0 && c1.Val() % c2.Val() == 0);
        SYM_TRY_REWRITE_IF(sym_mod(c1 + x, c2), sym_mod(x, c2), c2.Val() > 0 && c1.Val() % c2.Val() == 0);

        // mod(x * c1 + y, c2) => mod(y, c2) when c2>0 and c1%c2==0
        SYM_TRY_REWRITE_IF(sym_mod(x * c1 + y, c2), sym_mod(y, c2), c2.Val() > 0 && c1.Val() % c2.Val() == 0);
        SYM_TRY_REWRITE_IF(sym_mod(c1 * x + y, c2), sym_mod(y, c2), c2.Val() > 0 && c1.Val() % c2.Val() == 0);
        SYM_TRY_REWRITE_IF(sym_mod(y + x * c1, c2), sym_mod(y, c2), c2.Val() > 0 && c1.Val() % c2.Val() == 0);
        SYM_TRY_REWRITE_IF(sym_mod(y + c1 * x, c2), sym_mod(y, c2), c2.Val() > 0 && c1.Val() % c2.Val() == 0);

        return ret;
    }

    // ========================================================================
    // Min
    // ========================================================================

    RawPtr VisitMin(const RawPtr& a, const RawPtr& b)
    {
        RawPtr ret = Expr::Create(SymbolicOpcode::T_MOP_MIN, {a, b});
        PVarRaw x, y, z;
        PVarImm c1, c2, c3, c4, c5;

        // min(x, x) => x
        SYM_TRY_REWRITE(sym_min(x, x), x);

        // Constant reassociation
        SYM_TRY_REWRITE(sym_min(sym_min(x, c1), c2), sym_min(x, std::min(c1.Val(), c2.Val())));
        SYM_TRY_REWRITE(sym_min(c1, sym_min(x, c2)), sym_min(x, std::min(c1.Val(), c2.Val())));

        // Factor out common subexpressions
        SYM_TRY_REWRITE(sym_min(y - x, z - x), sym_min(y, z) - x);
        SYM_TRY_REWRITE(sym_min(x - y, x - z), x - sym_max(y, z));
        SYM_TRY_REWRITE(sym_min(x + y, x + z), x + sym_min(y, z));
        SYM_TRY_REWRITE(sym_min(y + x, x + z), x + sym_min(y, z));
        SYM_TRY_REWRITE(sym_min(x + y, z + x), x + sym_min(y, z));
        SYM_TRY_REWRITE(sym_min(y + x, z + x), x + sym_min(y, z));

        // Constant comparison
        SYM_TRY_REWRITE(sym_min(x + c1, x + c2), x + std::min(c1.Val(), c2.Val()));

        // Nested collapse
        SYM_TRY_REWRITE(sym_min(sym_min(x, y), x), sym_min(x, y));
        SYM_TRY_REWRITE(sym_min(sym_min(x, y), y), sym_min(x, y));
        SYM_TRY_REWRITE(sym_min(x, sym_min(x, y)), sym_min(x, y));
        SYM_TRY_REWRITE(sym_min(y, sym_min(x, y)), sym_min(x, y));

        // Absorption
        SYM_TRY_REWRITE(sym_min(sym_max(x, y), y), y);
        SYM_TRY_REWRITE(sym_min(sym_max(y, x), x), x);
        SYM_TRY_REWRITE(sym_min(y, sym_max(x, y)), y);
        SYM_TRY_REWRITE(sym_min(sym_max(x, y), x), x);
        SYM_TRY_REWRITE(sym_min(x, sym_max(x, y)), x);
        SYM_TRY_REWRITE(sym_min(x, sym_max(y, x)), x);

        // Cross min/max distribution
        SYM_TRY_REWRITE(sym_min(sym_max(x, y), sym_max(x, z)), sym_max(sym_min(y, z), x));
        SYM_TRY_REWRITE(sym_min(sym_max(x, y), sym_max(z, x)), sym_max(sym_min(y, z), x));
        SYM_TRY_REWRITE(sym_min(sym_max(y, x), sym_max(x, z)), sym_max(sym_min(y, z), x));
        SYM_TRY_REWRITE(sym_min(sym_max(y, x), sym_max(z, x)), sym_max(sym_min(y, z), x));

        // Clamp composition: clamp(clamp(x, L1, H1) - c, L2, H2) => clamp(x - c, L2, H2)
        // i.e. min(max(min(max(x, c1), c2) - c3, c4), c5) => min(max(x - c3, c4), c5).
        // The inner clamp bounds become redundant when its achievable range [L1, H1] shifted
        // by -c still covers the outer range [L2, H2]: L1 <= L2 + c and H1 >= H2 + c.
        SYM_TRY_REWRITE_IF(sym_min(sym_max(sym_min(sym_max(x, c1), c2) - c3, c4), c5), sym_min(sym_max(x - c3, c4), c5),
                           c1.Val() <= c4.Val() + c3.Val() && c2.Val() >= c5.Val() + c3.Val());

        SYM_TRY_REWRITE_IF(sym_min(sym_max(sym_min(sym_max(x, c1), c2), c4), c5), sym_min(sym_max(x, c4), c5),
                           c1.Val() <= c4.Val() && c2.Val() >= c5.Val());

        SYM_TRY_REWRITE(sym_min(sym_min(x, y), sym_min(x, z)), sym_min(sym_min(y, z), x));
        SYM_TRY_REWRITE(sym_min(sym_min(x, y), sym_min(z, x)), sym_min(sym_min(y, z), x));
        SYM_TRY_REWRITE(sym_min(sym_min(y, x), sym_min(x, z)), sym_min(sym_min(y, z), x));
        SYM_TRY_REWRITE(sym_min(sym_min(y, x), sym_min(z, x)), sym_min(sym_min(y, z), x));

        // Scaling: min(x * c1, y * c1) => min(x,y) * c1 when c1 > 0
        SYM_TRY_REWRITE_IF(sym_min(x * c1, y * c1), sym_min(x, y) * c1, c1.Val() > 0);
        SYM_TRY_REWRITE_IF(sym_min(x * c1, y * c1), sym_max(x, y) * c1, c1.Val() < 0);

        // Canonicalization
        SYM_TRY_REWRITE(sym_min(c1, x), sym_min(x, c1));
        SYM_TRY_REWRITE(sym_min(sym_min(x, c1), y), sym_min(sym_min(x, y), c1));

        return ret;
    }

    // ========================================================================
    // Max
    // ========================================================================

    RawPtr VisitMax(const RawPtr& a, const RawPtr& b)
    {
        RawPtr ret = Expr::Create(SymbolicOpcode::T_MOP_MAX, {a, b});
        PVarRaw x, y, z;
        PVarImm c1, c2;

        // max(x, x) => x
        SYM_TRY_REWRITE(sym_max(x, x), x);

        // Constant reassociation
        SYM_TRY_REWRITE(sym_max(sym_max(x, c1), c2), sym_max(x, std::max(c1.Val(), c2.Val())));
        SYM_TRY_REWRITE(sym_max(c1, sym_max(x, c2)), sym_max(x, std::max(c1.Val(), c2.Val())));

        // Factor out common subexpressions
        SYM_TRY_REWRITE(sym_max(y - x, z - x), sym_max(y, z) - x);
        SYM_TRY_REWRITE(sym_max(x - y, x - z), x - sym_min(y, z));
        SYM_TRY_REWRITE(sym_max(x + y, x + z), x + sym_max(y, z));
        SYM_TRY_REWRITE(sym_max(y + x, x + z), x + sym_max(y, z));
        SYM_TRY_REWRITE(sym_max(x + y, z + x), x + sym_max(y, z));
        SYM_TRY_REWRITE(sym_max(y + x, z + x), x + sym_max(y, z));

        // Constant comparison
        SYM_TRY_REWRITE(sym_max(x + c1, x + c2), x + std::max(c1.Val(), c2.Val()));

        // Nested collapse
        SYM_TRY_REWRITE(sym_max(sym_max(x, y), x), sym_max(x, y));
        SYM_TRY_REWRITE(sym_max(sym_max(x, y), y), sym_max(x, y));
        SYM_TRY_REWRITE(sym_max(x, sym_max(x, y)), sym_max(x, y));
        SYM_TRY_REWRITE(sym_max(y, sym_max(x, y)), sym_max(x, y));

        // Absorption
        SYM_TRY_REWRITE(sym_max(sym_min(x, y), y), y);
        SYM_TRY_REWRITE(sym_max(sym_min(y, x), x), x);
        SYM_TRY_REWRITE(sym_max(y, sym_min(x, y)), y);
        SYM_TRY_REWRITE(sym_max(sym_min(x, y), x), x);
        SYM_TRY_REWRITE(sym_max(x, sym_min(x, y)), x);
        SYM_TRY_REWRITE(sym_max(x, sym_min(y, x)), x);

        // Cross max/min distribution
        SYM_TRY_REWRITE(sym_max(sym_min(x, y), sym_min(x, z)), sym_min(sym_max(y, z), x));
        SYM_TRY_REWRITE(sym_max(sym_min(x, y), sym_min(z, x)), sym_min(sym_max(y, z), x));
        SYM_TRY_REWRITE(sym_max(sym_min(y, x), sym_min(x, z)), sym_min(sym_max(y, z), x));
        SYM_TRY_REWRITE(sym_max(sym_min(y, x), sym_min(z, x)), sym_min(sym_max(y, z), x));

        SYM_TRY_REWRITE(sym_max(sym_max(x, y), sym_max(x, z)), sym_max(sym_max(y, z), x));
        SYM_TRY_REWRITE(sym_max(sym_max(x, y), sym_max(z, x)), sym_max(sym_max(y, z), x));
        SYM_TRY_REWRITE(sym_max(sym_max(y, x), sym_max(x, z)), sym_max(sym_max(y, z), x));
        SYM_TRY_REWRITE(sym_max(sym_max(y, x), sym_max(z, x)), sym_max(sym_max(y, z), x));

        // Scaling
        SYM_TRY_REWRITE_IF(sym_max(x * c1, y * c1), sym_max(x, y) * c1, c1.Val() > 0);
        SYM_TRY_REWRITE_IF(sym_max(x * c1, y * c1), sym_min(x, y) * c1, c1.Val() < 0);

        // Canonicalization
        SYM_TRY_REWRITE(sym_max(c1, x), sym_max(x, c1));
        SYM_TRY_REWRITE(sym_max(sym_max(x, c1), y), sym_max(sym_max(x, y), c1));

        return ret;
    }

    // ========================================================================
    // Eq
    // ========================================================================

    RawPtr VisitEq(const RawPtr& a, const RawPtr& b)
    {
        RawPtr ret = Expr::CreateBopEq(a, b);
        PVarRaw x, y, z;
        PVarImm c1, c2;

        SYM_TRY_REWRITE(sym_eq(x, x), MakeConst(1));

        SYM_TRY_REWRITE(sym_eq(x + y, x + z), sym_eq(y, z));
        SYM_TRY_REWRITE(sym_eq(y + x, x + z), sym_eq(y, z));
        SYM_TRY_REWRITE(sym_eq(x + y, z + x), sym_eq(y, z));
        SYM_TRY_REWRITE(sym_eq(y + x, z + x), sym_eq(y, z));

        SYM_TRY_REWRITE(sym_eq(x - c1, c2), sym_eq(x, c1.Val() + c2.Val()));
        SYM_TRY_REWRITE(sym_eq(x + c1, c2), sym_eq(x, c2.Val() - c1.Val()));
        SYM_TRY_REWRITE(sym_eq(c1, x), sym_eq(x, c1));

        return ret;
    }

    // ========================================================================
    // Ne
    // ========================================================================

    RawPtr VisitNe(const RawPtr& a, const RawPtr& b)
    {
        RawPtr ret = Expr::CreateBopNe(a, b);
        PVarRaw x, y, z;
        PVarImm c1;

        SYM_TRY_REWRITE(sym_ne(x, x), MakeConst(0));

        SYM_TRY_REWRITE(sym_ne(x + y, x + z), sym_ne(y, z));
        SYM_TRY_REWRITE(sym_ne(y + x, x + z), sym_ne(y, z));
        SYM_TRY_REWRITE(sym_ne(x + y, z + x), sym_ne(y, z));
        SYM_TRY_REWRITE(sym_ne(y + x, z + x), sym_ne(y, z));

        SYM_TRY_REWRITE(sym_ne(c1, x), sym_ne(x, c1));

        return ret;
    }

    // ========================================================================
    // Lt
    // ========================================================================

    RawPtr VisitLt(const RawPtr& a, const RawPtr& b)
    {
        RawPtr ret = Expr::CreateBopLt(a, b);
        PVarRaw x, y, z;
        PVarImm c1, c2;

        SYM_TRY_REWRITE(sym_lt(x, x), MakeConst(0));

        SYM_TRY_REWRITE(sym_lt(x + y, x + z), sym_lt(y, z));
        SYM_TRY_REWRITE(sym_lt(y + x, x + z), sym_lt(y, z));
        SYM_TRY_REWRITE(sym_lt(x + y, z + x), sym_lt(y, z));
        SYM_TRY_REWRITE(sym_lt(y + x, z + x), sym_lt(y, z));
        SYM_TRY_REWRITE(sym_lt(y - x, z - x), sym_lt(y, z));
        SYM_TRY_REWRITE(sym_lt(x - y, x - z), sym_lt(z, y));

        SYM_TRY_REWRITE(sym_lt(x, x + z), sym_lt(MakeConst(0), z));
        SYM_TRY_REWRITE(sym_lt(x, z + x), sym_lt(MakeConst(0), z));
        SYM_TRY_REWRITE(sym_lt(x, x - z), sym_lt(z, MakeConst(0)));

        SYM_TRY_REWRITE(sym_lt(x + c1, c2), sym_lt(x, c2.Val() - c1.Val()));
        SYM_TRY_REWRITE(sym_lt(x - c1, c2), sym_lt(x, c2.Val() + c1.Val()));

        // Multiply by positive/negative constant
        SYM_TRY_REWRITE_IF(sym_lt(x * c1, y * c1), sym_lt(x, y), c1.Val() > 0);
        SYM_TRY_REWRITE_IF(sym_lt(x * c1, y * c1), sym_lt(y, x), c1.Val() < 0);

        // floordiv(x, c1) < c2 => x < c1*c2 when c1 > 0
        SYM_TRY_REWRITE_IF(sym_lt(sym_div(x, c1), c2), sym_lt(x, c1.Val() * c2.Val()), c1.Val() > 0);

        return ret;
    }

    // ========================================================================
    // Le
    // ========================================================================

    RawPtr VisitLe(const RawPtr& a, const RawPtr& b)
    {
        RawPtr ret = Expr::CreateBopLe(a, b);
        PVarRaw x, y, z;
        PVarImm c1, c2;

        SYM_TRY_REWRITE(sym_le(x, x), MakeConst(1));

        SYM_TRY_REWRITE(sym_le(x + y, x + z), sym_le(y, z));
        SYM_TRY_REWRITE(sym_le(y + x, x + z), sym_le(y, z));
        SYM_TRY_REWRITE(sym_le(x + y, z + x), sym_le(y, z));
        SYM_TRY_REWRITE(sym_le(y + x, z + x), sym_le(y, z));
        SYM_TRY_REWRITE(sym_le(y - x, z - x), sym_le(y, z));
        SYM_TRY_REWRITE(sym_le(x - y, x - z), sym_le(z, y));

        SYM_TRY_REWRITE(sym_le(x + c1, c2), sym_le(x, c2.Val() - c1.Val()));
        SYM_TRY_REWRITE(sym_le(x - c1, c2), sym_le(x, c2.Val() + c1.Val()));

        SYM_TRY_REWRITE_IF(sym_le(x * c1, y * c1), sym_le(x, y), c1.Val() > 0);
        SYM_TRY_REWRITE_IF(sym_le(x * c1, y * c1), sym_le(y, x), c1.Val() < 0);

        return ret;
    }

    // ========================================================================
    // And (logical)
    // ========================================================================

    RawPtr VisitAnd(const RawPtr& a, const RawPtr& b)
    {
        RawPtr ret = Expr::CreateBopAnd(a, b);
        PVarRaw x, y;

        // Idempotence
        SYM_TRY_REWRITE(sym_and(x, x), x);
        // Annihilator: x && 0 => 0
        SYM_TRY_REWRITE(sym_and(x, 0), MakeConst(0));
        SYM_TRY_REWRITE(sym_and(0, x), MakeConst(0));
        // Identity: x && 1 => x
        SYM_TRY_REWRITE(sym_and(x, 1), x);
        SYM_TRY_REWRITE(sym_and(1, x), x);
        // Complement: x && !x => 0
        SYM_TRY_REWRITE(sym_and(x, !x), MakeConst(0));
        SYM_TRY_REWRITE(sym_and(!x, x), MakeConst(0));
        // Absorption: x && (x || y) => x
        SYM_TRY_REWRITE(sym_and(x, sym_or(x, y)), x);
        SYM_TRY_REWRITE(sym_and(x, sym_or(y, x)), x);

        return ret;
    }

    // ========================================================================
    // Or (logical)
    // ========================================================================

    RawPtr VisitOr(const RawPtr& a, const RawPtr& b)
    {
        RawPtr ret = Expr::CreateBopOr(a, b);
        PVarRaw x, y;

        // Idempotence
        SYM_TRY_REWRITE(sym_or(x, x), x);
        // Annihilator: x || 1 => 1
        SYM_TRY_REWRITE(sym_or(x, 1), MakeConst(1));
        SYM_TRY_REWRITE(sym_or(1, x), MakeConst(1));
        // Identity: x || 0 => x
        SYM_TRY_REWRITE(sym_or(x, 0), x);
        SYM_TRY_REWRITE(sym_or(0, x), x);
        // Complement: x || !x => 1
        SYM_TRY_REWRITE(sym_or(x, !x), MakeConst(1));
        SYM_TRY_REWRITE(sym_or(!x, x), MakeConst(1));
        // Absorption: x || (x && y) => x
        SYM_TRY_REWRITE(sym_or(x, sym_and(x, y)), x);
        SYM_TRY_REWRITE(sym_or(x, sym_and(y, x)), x);

        return ret;
    }
};
} // namespace npu::tile_fwk
