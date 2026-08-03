/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen/cce/cce_codegen.h"

#include <cctype>
#include <cstddef>
#include <functional>
#include <ios>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "backend/common/backend.h"
#include "backend/common/backend_utils.h"
#include "core/error.h"
#include "core/logging.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/pipe.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/scalar_expr_ops.h"
#include "ir/stmt.h"
#include "ir/kind_traits.h"
#include "ir/transforms/base/mutator.h"
#include "ir/transforms/passes.h"
#include "ir/transforms/structural_comparison.h"
#include "ir/type.h"
#include "tilefwk/error.h"

namespace pypto {
namespace codegen {
using ir::DataType;

namespace {

bool IsNZTensorType(const ir::TensorTypePtr& tensor_type)
{
    return tensor_type && tensor_type->tensor_view_.has_value() &&
           tensor_type->tensor_view_->layout == ir::TensorLayout::NZ;
}

void ValidateStaticNZTensorShape(const ir::TensorTypePtr& tensor_type, const std::vector<int64_t>& logical_dims)
{
    CHECK(tensor_type != nullptr) << "CCE NZ tensor lowering requires a valid TensorType";
    if (logical_dims.size() != 2) {
        throw pypto::ir::ValueError("CCE NZ tensor lowering currently requires a 2D tensor");
    }

    const int64_t rows = logical_dims[0];
    const int64_t cols = logical_dims[1];
    const int64_t c0 = backend::cce::GetNZInnerCols(tensor_type->dtype_);
    CHECK(c0 > 0) << "NZ C0 size must be positive";
    if (rows % 16 != 0 || cols % c0 != 0) {
        throw pypto::ir::ValueError(
            "CCE NZ tensor lowering requires rows divisible by 16 and cols divisible by the destination C0 size");
    }
}

std::vector<int64_t> BuildNZPhysicalShapeDims(const ir::TensorTypePtr& tensor_type,
                                              const std::vector<int64_t>& logical_dims)
{
    ValidateStaticNZTensorShape(tensor_type, logical_dims);
    const int64_t c0 = backend::cce::GetNZInnerCols(tensor_type->dtype_);
    CHECK(c0 > 0) << "NZ C0 size must be positive";
    return {1, logical_dims[1] / c0, logical_dims[0] / 16, 16, c0};
}

} // namespace

CCECodegen::CCECodegen(ir::SectionKind target) : backend_(backend::GetBackend()), target_(target)
{
    CHECK(target_ == ir::SectionKind::Cube || target_ == ir::SectionKind::Vector)
        << "CCE target must be Cube or Vector";
}

// ============================================================================
// Helper function inlining
// ============================================================================

namespace {

/// Renames all variable definitions in a function body by adding a prefix,
/// and substitutes parameter references with call-site argument expressions.
class BodyRenamer : public ir::IRMutator {
public:
    using ir::IRMutator::VisitExpr_;
    using ir::IRMutator::VisitStmt_;
    BodyRenamer(const std::string& prefix, const std::vector<ir::VarPtr>& params, const std::vector<ir::ExprPtr>& args,
                const std::unordered_map<const ir::Expr*, ir::ExprPtr>& parent_remap = {})
        : prefix_(prefix)
    {
        // Seed with the enclosing inline's remap so free-variable references to an outer
        // helper's vars (e.g. an auto_mutex buf_id captured from the caller and carried into a
        // nested helper) are rewritten consistently. Params override any inherited entry.
        var_remap_ = parent_remap;
        // The Python->IR conversion gives each function its own Var objects, so a nested helper's
        // free reference to an outer helper's var is a *distinct* pointer with the same name. Build
        // a name-keyed fallback from the enclosing remap to bridge those (see VisitExpr_(Var)).
        for (const auto& [key, val] : parent_remap) {
            if (key != nullptr && key->GetKind() == ir::ObjectKind::Var) {
                name_remap_[static_cast<const ir::Var*>(key)->name_] = val;
            }
        }
        for (size_t i = 0; i < params.size() && i < args.size(); ++i) {
            var_remap_[params[i].get()] = args[i];
        }
    }

    /// Final var mapping after visiting; used to seed nested helper inlines.
    const std::unordered_map<const ir::Expr*, ir::ExprPtr>& GetRemap() const { return var_remap_; }

private:
    std::string prefix_;
    // Name-keyed fallback for free variables inherited from an enclosing inline (see ctor).
    std::unordered_map<std::string, ir::ExprPtr> name_remap_;

    ir::VarPtr RenameVar(const ir::VarPtr& var)
    {
        auto it = var_remap_.find(var.get());
        if (it != var_remap_.end()) {
            if (auto v = ir::As<ir::Var>(it->second))
                return v;
            return var;
        }
        auto renamed = std::make_shared<const ir::Var>(prefix_ + var->name_, var->GetType(), var->span_);
        var_remap_[var.get()] = renamed;
        return renamed;
    }

    ir::ExprPtr VisitExpr_(const ir::VarPtr& op) override
    {
        auto it = var_remap_.find(op.get());
        if (it != var_remap_.end())
            return it->second;
        // Pointer miss on a var whose definition is not in this body means it is a free variable
        // inherited from an enclosing inline; bridge it by name (its defining object lives in the
        // enclosing helper and was already renamed there). Helper-local uses always hit above,
        // since their defining statement is visited (and pointer-mapped) before any use.
        auto name_it = name_remap_.find(op->name_);
        if (name_it != name_remap_.end())
            return name_it->second;
        return op;
    }

    ir::StmtPtr VisitStmt_(const ir::AssignStmtPtr& op) override
    {
        auto new_value = VisitExpr(op->value_);
        auto target_var = op->var_;
        auto new_var = RenameVar(target_var);
        if (new_var.get() != target_var.get() || new_value.get() != op->value_.get()) {
            return std::make_shared<const ir::AssignStmt>(std::move(new_var), std::move(new_value), op->span_);
        }
        return op;
    }

    std::vector<ir::IterArgPtr> RenameIterArgs(const std::vector<ir::IterArgPtr>& iter_args)
    {
        std::vector<ir::IterArgPtr> new_iter_args;
        new_iter_args.reserve(iter_args.size());
        for (const auto& ia : iter_args) {
            auto new_init = VisitExpr(ia->initValue_);
            // Iter args may shadow helper params, so keep them isolated from call-site remaps.
            auto renamed_var = std::make_shared<const ir::Var>(prefix_ + ia->iterVar_->name_, ia->iterVar_->GetType(),
                                                               ia->iterVar_->span_);
            auto renamed = std::make_shared<const ir::IterArg>(renamed_var, new_init);
            var_remap_[ia->iterVar_.get()] = renamed_var;
            new_iter_args.push_back(renamed);
        }
        return new_iter_args;
    }

    ir::StmtPtr VisitStmt_(const ir::ForStmtPtr& op) override
    {
        auto new_loop_var = RenameVar(op->loopVar_);
        var_remap_[op->loopVar_.get()] = new_loop_var;
        auto new_start = VisitExpr(op->start_);
        auto new_stop = VisitExpr(op->stop_);
        auto new_step = VisitExpr(op->step_);

        auto new_iter_args = RenameIterArgs(op->iterArgs_);
        auto new_body = VisitStmt(op->body_);

        std::vector<ir::VarPtr> new_return_vars;
        new_return_vars.reserve(op->returnVars_.size());
        for (const auto& rv : op->returnVars_) {
            new_return_vars.push_back(RenameVar(rv));
        }

        return std::make_shared<const ir::ForStmt>(new_loop_var, new_start, new_stop, new_step,
                                                   std::move(new_iter_args), new_body, std::move(new_return_vars),
                                                   op->span_);
    }

    ir::StmtPtr VisitStmt_(const ir::WhileStmtPtr& op) override
    {
        auto new_iter_args = RenameIterArgs(op->iterArgs_);
        auto new_cond = VisitExpr(op->condition_);
        auto new_body = VisitStmt(op->body_);

        std::vector<ir::VarPtr> new_return_vars;
        new_return_vars.reserve(op->returnVars_.size());
        for (const auto& rv : op->returnVars_) {
            new_return_vars.push_back(RenameVar(rv));
        }

        return std::make_shared<const ir::WhileStmt>(new_cond, std::move(new_iter_args), new_body,
                                                     std::move(new_return_vars), op->span_);
    }

    ir::StmtPtr VisitStmt_(const ir::IfStmtPtr& op) override
    {
        auto new_cond = VisitExpr(op->condition_);
        auto new_then = VisitStmt(op->thenBody_);
        std::optional<ir::StmtPtr> new_else;
        if (op->elseBody_.has_value()) {
            new_else = VisitStmt(*op->elseBody_);
        }
        std::vector<ir::VarPtr> new_rvs;
        new_rvs.reserve(op->returnVars_.size());
        for (const auto& rv : op->returnVars_) {
            new_rvs.push_back(RenameVar(rv));
        }
        return std::make_shared<const ir::IfStmt>(new_cond, new_then, new_else, std::move(new_rvs), op->span_);
    }

    bool RenameJumpValues(const std::vector<ir::ExprPtr>& values, std::vector<ir::ExprPtr>& new_values)
    {
        new_values.reserve(values.size());
        bool changed = false;
        for (const auto& value : values) {
            auto new_value = VisitExpr(value);
            changed = changed || (new_value.get() != value.get());
            new_values.push_back(std::move(new_value));
        }
        return changed;
    }

    // ConvertToSSA (which runs before inlining) snapshots each loop-carried
    // value's live SSA version at a break/continue and stores it in value_, so
    // the backend can write those values back before the jump. The base mutator
    // returns Break/Continue unchanged, so without these overrides the snapshot
    // references keep their pre-inline names and become undefined (or collide
    // with outer-scope vars) once the rest of the body is prefix-renamed.
    ir::StmtPtr VisitStmt_(const ir::BreakStmtPtr& op) override
    {
        std::vector<ir::ExprPtr> new_values;
        if (!RenameJumpValues(op->value_, new_values))
            return op;
        return std::make_shared<const ir::BreakStmt>(std::move(new_values), op->span_);
    }

    ir::StmtPtr VisitStmt_(const ir::ContinueStmtPtr& op) override
    {
        std::vector<ir::ExprPtr> new_values;
        if (!RenameJumpValues(op->value_, new_values))
            return op;
        return std::make_shared<const ir::ContinueStmt>(std::move(new_values), op->span_);
    }
};

/// Replaces all ReturnStmt nodes in a body with an assignment to `target`.
///
/// The call site is always a single var (tuple unpacking is lowered to
/// `_tuple_tmp = helper(); a = _tuple_tmp[0]` by the parser), and a tuple
/// return is a single MakeTuple expression. So a return lowers to exactly
/// `target = value`; tuple resolution is left to backend codegen.
class ReturnReplacer : public ir::IRMutator {
public:
    using ir::IRMutator::VisitStmt_;
    explicit ReturnReplacer(ir::VarPtr target) : target_(std::move(target)) {}

    ir::StmtPtr VisitStmt_(const ir::ReturnStmtPtr& op) override
    {
        if (target_ && op->value_.empty()) {
            // The call result is used as a value, but this path is a void `return` (a
            // `return None` in the helper). Assign an empty-tuple sentinel: if this path is
            // dead it is dropped by const-folding; if it is the taken path the sentinel
            // survives and codegen reports a clear error at the use site (see
            // VisitExpr_(GetItemExprPtr)) instead of emitting an undeclared identifier that
            // only fails later inside the CCE compiler.
            return std::make_shared<const ir::AssignStmt>(
                target_, std::make_shared<const ir::MakeTuple>(std::vector<ir::ExprPtr>{}, op->span_), op->span_);
        }
        if (!target_ || op->value_.empty()) {
            // Void call — replace return with empty SeqStmts
            return std::make_shared<const ir::SeqStmts>(std::vector<ir::StmtPtr>{}, op->span_);
        }
        CHECK(op->value_.size() == 1)
            << "helper return must carry a single value (tuple returns are a single MakeTuple)";
        return std::make_shared<const ir::AssignStmt>(target_, op->value_[0], op->span_);
    }

private:
    ir::VarPtr target_;
};

/// Build the inlined body for a helper call. Replaces ReturnStmt with an
/// assignment to `target` (null for void/EvalStmt calls).
ir::StmtPtr BuildInlinedBody(const ir::FunctionPtr& helper, const std::vector<ir::ExprPtr>& call_args,
                             const std::string& prefix, const ir::VarPtr& target,
                             const std::unordered_map<const ir::Expr*, ir::ExprPtr>& parent_remap = {},
                             std::unordered_map<const ir::Expr*, ir::ExprPtr>* out_remap = nullptr)
{
    BodyRenamer renamer(prefix, helper->params_, call_args, parent_remap);
    auto renamed_body = renamer.VisitStmt(helper->body_);
    if (out_remap != nullptr) {
        *out_remap = renamer.GetRemap();
    }

    ReturnReplacer replacer(target);
    return replacer.VisitStmt(renamed_body);
}

/// Check if a Call targets a helper function by name.
bool IsHelperCall(const ir::CallPtr& call, const std::map<std::string, ir::FunctionPtr>& helpers)
{
    if (!call || call->name_.empty())
        return false;
    return helpers.count(call->name_) > 0;
}

/// Walks the kernel body, replacing helper calls with inlined bodies.
///
/// Tuple handling is intentionally left to backend codegen: a tuple-returning
/// helper inlines to `target = MakeTuple(...)` (via ReturnReplacer), and a
/// later `target[i]` is resolved by CCE/PTO codegen's tuple_var_to_make_tuple_.
class CallInliner : public ir::IRMutator {
public:
    using ir::IRMutator::VisitStmt_;
    explicit CallInliner(const std::map<std::string, ir::FunctionPtr>& helpers) : helpers_(helpers) {}

    ir::StmtPtr VisitStmt_(const ir::AssignStmtPtr& op) override
    {
        auto call = ir::As<ir::Call>(op->value_);
        if (!IsHelperCall(call, helpers_)) {
            return IRMutator::VisitStmt_(op);
        }
        auto target_var = op->var_;
        std::string prefix = "_inl" + std::to_string(counter_++) + "_";
        const auto& helper = helpers_.at(call->name_);

        std::vector<ir::ExprPtr> args;
        args.reserve(call->args_.size());
        for (const auto& a : call->args_) {
            args.push_back(VisitExpr(a));
        }

        // Single target: tuple returns lower to `target = MakeTuple(...)`.
        std::unordered_map<const ir::Expr*, ir::ExprPtr> child_remap;
        auto inlined = BuildInlinedBody(helper, args, prefix, target_var, ambient_remap_, &child_remap);
        auto result = VisitInlinedWithAmbient(inlined, std::move(child_remap)); // recurse for nested helper calls
        return UnwrapIfNestedVF(result);
    }

    ir::StmtPtr VisitStmt_(const ir::EvalStmtPtr& op) override
    {
        auto call = ir::As<ir::Call>(op->expr_);
        if (!IsHelperCall(call, helpers_)) {
            return IRMutator::VisitStmt_(op);
        }
        std::string prefix = "_inl" + std::to_string(counter_++) + "_";
        const auto& helper = helpers_.at(call->name_);

        std::vector<ir::ExprPtr> args;
        args.reserve(call->args_.size());
        for (const auto& a : call->args_) {
            args.push_back(VisitExpr(a));
        }

        std::unordered_map<const ir::Expr*, ir::ExprPtr> child_remap;
        auto inlined = BuildInlinedBody(helper, args, prefix, /*target=*/nullptr, ambient_remap_, &child_remap);
        auto result = VisitInlinedWithAmbient(inlined, std::move(child_remap));
        return UnwrapIfNestedVF(result);
    }

    /// Track whether we are inlining inside an enclosing VF section. A VF helper
    /// (@pl.vector_function) is compiled with its body wrapped in a SectionStmt(VF);
    /// when such a helper is called from another VF helper, naive inlining would
    /// nest two VF sections, which the codegen VF-section invariant forbids
    /// (see EmitSection: "VF section must be nested inside a Vector section").
    /// Flatten by tracking the current section kind here and unwrapping a nested
    /// VF section at the inline site (see UnwrapIfNestedVF).
    ir::StmtPtr VisitStmt_(const ir::SectionStmtPtr& op) override
    {
        bool prev_in_vf = in_vf_section_;
        if (op->sectionKind_ == ir::SectionKind::VF) {
            in_vf_section_ = true;
        }
        auto result = IRMutator::VisitStmt_(op);
        in_vf_section_ = prev_in_vf;
        return result;
    }

private:
    /// If we are already inside a VF section and the freshly-inlined statement is
    /// itself a SectionStmt(VF), splice in its body instead of the nested section,
    /// collapsing multiple VF-helper calls into a single VF section.
    ir::StmtPtr UnwrapIfNestedVF(const ir::StmtPtr& inlined)
    {
        if (!in_vf_section_) {
            return inlined;
        }
        auto section = ir::As<ir::SectionStmt>(inlined);
        if (section && section->sectionKind_ == ir::SectionKind::VF) {
            return section->body_;
        }
        return inlined;
    }

    /// Recurse into a freshly-inlined body with `child_remap` (this inline's var mapping) as the
    /// ambient remap, so a nested helper's free references to this helper's vars are rewritten
    /// consistently. Restores the previous ambient afterward.
    ir::StmtPtr VisitInlinedWithAmbient(const ir::StmtPtr& inlined,
                                        std::unordered_map<const ir::Expr*, ir::ExprPtr>&& child_remap)
    {
        auto saved = std::move(ambient_remap_);
        ambient_remap_ = std::move(child_remap);
        auto result = VisitStmt(inlined);
        ambient_remap_ = std::move(saved);
        return result;
    }

    const std::map<std::string, ir::FunctionPtr>& helpers_;
    int counter_ = 0;
    // Var mapping from the enclosing inline(s); seeds nested helper inlines so their free-variable
    // references (e.g. auto_mutex buf_ids captured from an outer scope) remap correctly.
    std::unordered_map<const ir::Expr*, ir::ExprPtr> ambient_remap_;
    // True while visiting inside a VF section, so a VF helper inlined into another
    // VF helper has its redundant nested VF section unwrapped (see UnwrapIfNestedVF).
    bool in_vf_section_ = false;
};

} // namespace

// ============================================================================
// Single-program CCE generation (skip ptoas)
// ============================================================================

void CCECodegen::PreScanKernel(const ir::FunctionPtr& kernel_func)
{
    var_read_names_.clear();
    CollectVarReadNames(kernel_func->body_, var_read_names_);

    mutex_pipes_.clear();
    CollectMutexPipeInfo(kernel_func->body_);
}

void CCECodegen::PrepareBodyGeneration()
{
    tuple_var_to_make_tuple_.clear();
    tuple_backing_arr_.clear();
    loop_target_stack_.clear();
    yield_buffer_.clear();
}

std::string CCECodegen::GenerateSingle(const ir::ProgramPtr& program, const std::string& arch)
{
    CHECK(program != nullptr) << "Cannot generate code for null program";

    arch_ = arch;
    in_vf_section_ = false;

    // Capture the tuple/struct field-name side table from the entry program (may be null).
    // Passes below rebuild the Program (dropping this annotation), but the TupleType pointers
    // they key on stay valid, so codegen reads names through this captured table.
    debug_info_ = program->GetDebugInfo();

    // break/continue are handled natively in VisitStmt_(Break/Continue) below.
    // ConvertToSSA still models loop-carried scalars as iter_args +
    // a trailing yield, so the native-jump visitors write those values back before jumping.
    ir::ProgramPtr ssa_program = ir::pass::ConvertToSSA()(program);

    ir::FunctionPtr kernel_func;
    for (const auto& func_entry : ssa_program->functions_) {
        const auto& func = func_entry.second;
        if (func->funcType_ == ir::FunctionType::ORCHESTRATION) {
            continue;
        }
        if (!kernel_func) {
            kernel_func = func;
        }
    }
    CHECK(kernel_func != nullptr) << "No kernel function found in program";

    emitter_.Clear();
    context_.Clear();
    tensor_to_pointer_.clear();
    tiling_headers_.clear();
    struct_definitions_.clear();

    PreScanKernel(kernel_func);

    // Detect cross-core sync (a5 uses hardware sync, not ffts)
    bool has_cross_sync = DetectCrossCoreSyncOps(kernel_func->body_);
    bool needs_ffts = has_cross_sync && (arch_ != "a5");

    RegisterTilingStructTypes(kernel_func);
    GenerateSinglePrologue(kernel_func, needs_ffts);
    PrepareBodyGeneration();
    try {
        GenerateBody(kernel_func);
    } catch (const npu::tile_fwk::Error& e) {
        const ir::Span& best_span = current_expr_span_.IsUnknown() ? current_stmt_span_ : current_expr_span_;
        if (!best_span.IsUnknown()) {
            std::string enriched = std::string(e.what()) + "\n  --> " + best_span.ToString();
            throw npu::tile_fwk::Error("GenerateBody", __FILE__, __LINE__, enriched, nullptr);
        }
        throw;
    }

    std::string function_code = emitter_.GetCode();
    emitter_.Clear();
    if (target_ == ir::SectionKind::Cube) {
        emitter_.EmitLine("#if defined(__DAV_CUBE__)");
    } else {
        emitter_.EmitLine("#if defined(__DAV_VEC__)");
    }
    EmitStructTypes();
    emitter_.EmitRaw(function_code);
    emitter_.EmitLine("#endif");

    return emitter_.GetCode();
}

namespace {
/**
 * \brief Collect tile definitions sourced from block.make_tile calls.
 *
 * Scans AssignStmt nodes and records each block.make_tile allocation with:
 *   - The var that holds the result (or a synthetic var_N for tuple elements)
 *   - The TileType of the allocation
 *   - The SectionKind in which the make_tile appears (nullopt = shared/outside section)
 *
 * Inline tuple elements are named with a flat depth-first index (var_0, var_1, ...).
 */
class MakeTileDefCollector : public ir::IRVisitor {
    using ir::IRVisitor::VisitExpr_;
    using ir::IRVisitor::VisitStmt_;

public:
    std::vector<TileDef> tile_defs_;

    void VisitStmt_(const ir::AssignStmtPtr& op) override
    {
        auto target_var = op->var_;
        if (!op->value_)
            return;
        if (auto call = ir::As<ir::Call>(op->value_)) {
            // Direct block.make_tile assignment ->primary collection path.
            if (call->name_ == "block.make_tile") {
                if (auto tile_type = ir::As<ir::TileType>(call->GetType())) {
                    tile_defs_.push_back({target_var, tile_type});
                }
            }
            return;
        }
    }
};

// Extract valid_shape constructor arguments from a TileType for CCE code generation.
//
// needs_ctor == true - template has -1 params (dynamic), Tile constructor must receive runtime values.
// needs_ctor == false - template has explicit static params, no constructor args needed.
//
// Mapping for each valid_shape element:
//   absent / ConstInt(-1) - template param = -1, ctor_arg = rows/cols  (needs_ctor = true)
//   ConstInt(N > 0)       - template param = N,  ctor_arg unused        (needs_ctor = false)
//   Var(name)             - template param = -1 (skipped by ConvertTileType), ctor_arg = var_name
struct ValidShapeInfo {
    std::string row_ctor_arg;
    std::string col_ctor_arg;
    bool needs_ctor = false; // true when template uses -1 and constructor args are required
};

inline ValidShapeInfo ExtractValidShapeInfo(const ir::TileTypePtr& tile_type, int64_t rows, int64_t cols,
                                            std::function<std::string(const ir::VarPtr&)> get_var_name)
{
    ValidShapeInfo info;
    if (!tile_type->tileView_.has_value()) {
        // No tile_view: absent valid_shape - template uses -1, ctor uses full shape.
        info.row_ctor_arg = std::to_string(rows);
        info.col_ctor_arg = std::to_string(cols);
        info.needs_ctor = true;
        return info;
    }
    const auto& tv = tile_type->tileView_.value();
    if (tv.validShape.empty()) {
        // valid_shape absent: template uses -1, ctor uses full shape.
        info.row_ctor_arg = std::to_string(rows);
        info.col_ctor_arg = std::to_string(cols);
        info.needs_ctor = true;
        return info;
    }
    // valid_shape provided: check whether any element is -1 or a runtime Var.
    auto resolve_dim = [&](const ir::ExprPtr& expr, int64_t fallback, std::string& out_arg) -> bool {
        if (auto var = ir::As<ir::Var>(expr)) {
            out_arg = get_var_name(var);
            return true; // runtime var - needs ctor
        }
        if (auto c = ir::As<ir::ConstInt>(expr)) {
            if (c->value_ == -1) {
                out_arg = std::to_string(fallback);
                return true; // -1 sentinel - template gets -1, ctor gets actual dim
            }
            out_arg = std::to_string(c->value_);
            return false; // explicit static value - no ctor needed
        }
        return false;
    };
    bool row_dynamic = false;
    bool col_dynamic = false;
    if (tv.validShape.size() >= 1)
        row_dynamic = resolve_dim(tv.validShape[0], rows, info.row_ctor_arg);
    if (tv.validShape.size() >= 2)
        col_dynamic = resolve_dim(tv.validShape[1], cols, info.col_ctor_arg);
    info.needs_ctor = row_dynamic || col_dynamic;
    return info;
}

// Build Tile constructor argument string.
// Only call when ValidShapeInfo::needs_ctor is true.
inline std::string BuildTileCtorArgs(const ValidShapeInfo& vs, int64_t rows, int64_t cols)
{
    std::string r = vs.row_ctor_arg.empty() ? std::to_string(rows) : vs.row_ctor_arg;
    std::string c = vs.col_ctor_arg.empty() ? std::to_string(cols) : vs.col_ctor_arg;
    return r + ", " + c;
}

std::vector<ir::VarPtr> CollectDynamicDimVars(const ir::FunctionPtr& func)
{
    std::vector<ir::VarPtr> dyn_dim_vars;
    std::set<std::string> seen_dyn_names;
    for (const auto& param : func->params_) {
        auto tensor_type = std::dynamic_pointer_cast<const ir::TensorType>(param->GetType());
        if (!tensor_type) {
            continue;
        }
        for (const auto& dim_expr : tensor_type->shape_) {
            auto dim_var = std::dynamic_pointer_cast<const ir::Var>(dim_expr);
            if (dim_var && seen_dyn_names.insert(dim_var->name_).second) {
                dyn_dim_vars.push_back(dim_var);
            }
        }
    }
    return dyn_dim_vars;
}

bool IsOnlyYieldStmts(const ir::StmtPtr& stmt)
{
    if (!stmt) {
        return false;
    }
    if (ir::As<ir::YieldStmt>(stmt)) {
        return true;
    }

    auto seq = ir::As<ir::SeqStmts>(stmt);
    if (!seq || seq->stmts_.empty()) {
        return false;
    }
    for (const auto& st : seq->stmts_) {
        if (!ir::As<ir::YieldStmt>(st)) {
            return false;
        }
    }
    return true;
}

} // namespace

// ========================================================================
// Phase 6 helpers: GenerateSinglePrologue sub-functions
// ========================================================================

void CCECodegen::EmitSingleFunctionSignature(const ir::FunctionPtr& func, bool has_cross_sync)
{
    // Collect dynamic dim variables from tensor shapes (first-occurrence order)
    std::vector<ir::VarPtr> dyn_dim_vars = CollectDynamicDimVars(func);

    // Emit the kernel body as a device function `<name>_impl`; the `__global__` entry that
    // forwards to it is generated by the launcher (JIT call_kernel.cpp) or the binary-delivery
    // op cpp. Keeps the body identical while letting the entry layer add adaptation logic later.
    std::ostringstream sig;
    sig << "__aicore__ inline void " << func->name_ << "_impl";
    if (target_ == ir::SectionKind::Cube) {
        sig << "_cube";
    } else {
        sig << "_vector";
    }
    sig << "(";
    bool first = true;
    for (const auto& param : func->params_) {
        if (!first)
            sig << ", ";
        first = false;
        // Each parameter is registered here while building the signature (the single place
        // that walks all params): the C++ var name, plus the raw pointer for tensor params.
        if (auto tensor_type = std::dynamic_pointer_cast<const ir::TensorType>(param->GetType())) {
            std::string param_name = context_.SanitizeName(param);
            auto ptr_var = std::const_pointer_cast<ir::Var>(ir::As<ir::Var>(*tensor_type->tensor_view_->ptr));
            std::string ptr_name = param_name + "_ptr";
            context_.RegisterVar(ptr_var, ptr_name);
            sig << GetGeneratedType(tensor_type) << " " << ptr_name;
            context_.RegisterVar(param, param_name);
            RegisterPointer(param_name, ptr_name);
        } else if (auto scalar_type = std::dynamic_pointer_cast<const ir::ScalarType>(param->GetType())) {
            std::string param_name = context_.SanitizeName(param);
            sig << GetGeneratedType(scalar_type) << " " << param_name;
            context_.RegisterVar(param, param_name);
        } else if (auto ptr_type = std::dynamic_pointer_cast<const ir::PtrType>(param->GetType())) {
            std::string param_name = context_.SanitizeName(param);
            sig << GetGeneratedType(ptr_type) << " " << param_name;
            context_.RegisterVar(param, param_name);
        } else if (std::dynamic_pointer_cast<const ir::TupleType>(param->GetType())) {
            // Tiling struct param: pass a raw GM byte pointer `<name>_ptr`; the local
            // struct `<name>` (filled by EmitTilingStructCopy) takes the bare name,
            // so `GetItemExpr` reads emit `<name>.field`.
            std::string param_name = context_.SanitizeName(param);
            sig << "__gm__ uint8_t* " << param_name << "_ptr";
            context_.RegisterVar(param, param_name);
        }
    }
    // Append dynamic dim variables as int64_t parameters
    for (const auto& dyn_var : dyn_dim_vars) {
        if (!first)
            sig << ", ";
        first = false;
        sig << "int32_t " << context_.SanitizeName(dyn_var);
    }
    if (has_cross_sync) {
        if (!first)
            sig << ", ";
        sig << "__gm__ int64_t* ffts_addr";
    }
    sig << ")";

    emitter_.EmitLine(sig.str());
    emitter_.EmitLine("{");
    emitter_.IncreaseIndent();

    // Register dynamic dim parameters directly (no _local_ copy needed)
    for (const auto& dyn_var : dyn_dim_vars) {
        std::string param_name = context_.SanitizeName(dyn_var);
        context_.RegisterVar(dyn_var, param_name);
    }

    // Emit set_ffts_base_addr if cross-core sync
    if (has_cross_sync) {
        emitter_.EmitLine("set_ffts_base_addr((unsigned long)ffts_addr);");
    }
    if (target_ == ir::SectionKind::Vector && arch_ == "a3") {
        emitter_.EmitLine("set_mask_norm();");
        emitter_.EmitLine("set_vector_mask(-1, -1);");
    }
    emitter_.EmitLine("");
}

void CCECodegen::EmitSingleTensorDeclarations(const ir::FunctionPtr& func)
{
    // Prescan all accessed tensor vars into tensor_defs_ (params + ptr.make_tensor views).
    // Parameter pointers are already registered while building the signature (see
    // EmitSingleFunctionSignature). Here we emit GlobalTensor declarations for the tensor
    // parameters only; ptr.make_tensor views are emitted in place at their op (see
    // MakeBlockMakeTensorCodegenCCE), where the source pointer's C++ code is already in scope.
    CollectTensorDefs(func);

    std::vector<TensorDef> param_defs;
    for (const auto& param : func->params_) {
        auto it = tensor_defs_.find(context_.SanitizeName(param));
        if (it != tensor_defs_.end()) {
            param_defs.push_back(it->second);
        }
    }

    for (const auto& def : param_defs) {
        GenerateGlobalTensorTypeDeclaration(def);
        emitter_.EmitLine("");
    }
}

void CCECodegen::EmitSingleTileDeclarations(const ir::FunctionPtr& func)
{
    if (!func->body_)
        return;

    // Collect tile definitions from block.make_tile sources only.
    MakeTileDefCollector def_collector;
    def_collector.VisitStmt(func->body_);

    auto has_independent_runtime_metadata = [](const ir::TileTypePtr& tile_type) {
        return tile_type != nullptr && tile_type->tileView_.has_value() && !tile_type->tileView_->validShape.empty();
    };

    std::vector<TileDef> kept_defs;
    std::vector<std::pair<ir::VarPtr, ir::VarPtr>> deduped_aliases;
    std::map<std::string, ir::VarPtr> kept_tile_addr_vars;

    for (const auto& td : def_collector.tile_defs_) {
        const auto& tile_type = td.tile_type;
        if (tile_type->memref_.has_value()) {
            // Same-address tiles with explicit valid_shape metadata must stay as
            // distinct C++ objects; a later SetValidShape() on one tile would otherwise
            // narrow every deduped alias sharing the same backing storage.
            if (has_independent_runtime_metadata(tile_type)) {
                kept_defs.push_back(td);
                continue;
            }
            int64_t addr = ExtractConstInt((*tile_type->memref_)->addr_);
            auto space = (*tile_type->memref_)->memorySpace_;
            std::string type_key = GetGeneratedType(tile_type);
            std::string dedup_key = std::to_string(static_cast<int>(space)) + ":" + std::to_string(addr) + ":" +
                                    type_key;
            auto it = kept_tile_addr_vars.find(dedup_key);
            if (it != kept_tile_addr_vars.end()) {
                deduped_aliases.emplace_back(td.var, it->second);
                continue;
            }
            kept_tile_addr_vars[dedup_key] = td.var;
        }
        kept_defs.push_back(td);
    }

    if (!kept_defs.empty() || !deduped_aliases.empty()) {
        EmitTileDeclarations(kept_defs, deduped_aliases);
    }
}

void CCECodegen::EmitDedupedTileAliases(const std::vector<std::pair<ir::VarPtr, ir::VarPtr>>& deduped_aliases)
{
    for (const auto& [dup_var, kept_var] : deduped_aliases) {
        std::string san_dup = context_.SanitizeName(dup_var);
        std::string san_kept = context_.SanitizeName(kept_var);
        emitter_.EmitLine("auto& " + san_dup + " = " + san_kept + ";");
        emitted_tile_aliases_.insert(san_dup);
    }
}

void CCECodegen::EmitTileDeclarations(const std::vector<TileDef>& tile_defs,
                                      const std::vector<std::pair<ir::VarPtr, ir::VarPtr>>& deduped_aliases)
{
    for (const auto& td : tile_defs) {
        GenerateTileTypeDeclaration(context_.SanitizeName(td.var), td.tile_type);
    }
    EmitDedupedTileAliases(deduped_aliases);
    emitter_.EmitLine("");
}

void CCECodegen::GenerateSinglePrologue(const ir::FunctionPtr& func, bool has_cross_sync)
{
    EmitSingleFunctionSignature(func, has_cross_sync);
    // Tiling struct variable definition + GM copy must come first: the very next decls
    // (tensors/tiles with dynamic shapes) and the body read `tiling.field`.
    EmitTilingStructCopy(func);
    EmitSingleTensorDeclarations(func);
    EmitSingleTileDeclarations(func);
}

void CCECodegen::VisitStmt_(const ir::SectionStmtPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null SectionStmt";

    // VF section: emit a __VEC_SCOPE__ { ... } block with all tile->ptr base declarations
    // hoisted to just before the scope (section_hoist), so they never sit after a mem_bar.
    if (op->sectionKind_ == ir::SectionKind::VF) {
        INTERNAL_CHECK(target_ == ir::SectionKind::Vector) << "VF section is only valid in a Vector target Program";
        // VF sections do not nest, so the hoist buffer must be empty on entry.
        INTERNAL_CHECK(section_hoisted_decls_.empty());

        int saved_indent = emitter_.GetIndentLevel();
        std::string pre_scope = emitter_.GetCode();
        emitter_.Clear();
        emitter_.SetIndentLevel(saved_indent);

        INTERNAL_CHECK(!in_vf_section_) << "Nested VF sections are not supported";
        in_vf_section_ = true;
        vf_reg_hoisted_decls_.clear();

        emitter_.EmitLine("__VEC_SCOPE__ {");
        emitter_.IncreaseIndent();

        // Capture header position so RegTensor declarations can be hoisted to the
        // top of __VEC_SCOPE__ body (must be inside the scope but before any loops).
        std::string scope_header = emitter_.GetCode();
        int body_indent = emitter_.GetIndentLevel();
        emitter_.Clear();
        emitter_.SetIndentLevel(body_indent);

        if (op->body_) {
            VisitStmt(op->body_); // auto-declare pushes RegTensor decls to vf_reg_hoisted_decls_
        }

        std::string scope_body = emitter_.GetCode();
        emitter_.Clear();
        emitter_.SetIndentLevel(body_indent);

        // Reassemble: header + hoisted RegTensor decls + body
        emitter_.EmitRaw(scope_header);
        for (const auto& decl : vf_reg_hoisted_decls_) {
            emitter_.EmitLine(decl);
        }
        vf_reg_hoisted_decls_.clear();
        emitter_.EmitRaw(scope_body);

        emitter_.DecreaseIndent();
        emitter_.EmitLine("}");
        in_vf_section_ = false;

        std::string scope_code = emitter_.GetCode();
        emitter_.Clear();
        emitter_.SetIndentLevel(saved_indent);
        emitter_.EmitRaw(pre_scope);
        // vf_tile_ptr_N names are globally unique, so the hoisted decls can sit before __VEC_SCOPE__
        // at this scope level without colliding across sections.
        for (const auto& decl : section_hoisted_decls_) {
            emitter_.EmitLine(decl); // flush before __VEC_SCOPE__ (ahead of every mem_bar inside)
        }
        section_hoisted_decls_.clear();
        emitter_.EmitRaw(scope_code);

        vf_tile_ptrs_.clear();
        return;
    }

    INTERNAL_CHECK(false) << "Cube/Vector SectionStmt must be projected before CCE CodeGen";
}

bool CCECodegen::DetectCrossCoreSyncOps(const ir::StmtPtr& stmt)
{
    if (!stmt)
        return false;

    class CrossCoreSyncDetector : public ir::IRVisitor {
        using ir::IRVisitor::VisitExpr_;
        using ir::IRVisitor::VisitStmt_;

    public:
        bool found = false;
        void VisitExpr_(const ir::CallPtr& op) override
        {
            const std::string& name = op->name_;
            if (name == "system.set_cross_core" || name == "system.wait_cross_core" ||
                name == "system.set_cross_core_dyn" || name == "system.wait_cross_core_dyn") {
                found = true;
            }
            ir::IRVisitor::VisitExpr_(op);
        }
    };

    CrossCoreSyncDetector detector;
    detector.VisitStmt(stmt);
    return detector.found;
}

void CCECodegen::VisitStmt(const ir::StmtPtr& stmt)
{
    if (stmt) {
        current_stmt_span_ = stmt->span_;
    }
    ir::IRVisitor::VisitStmt(stmt);
}

void CCECodegen::VisitExpr(const ir::ExprPtr& expr)
{
    if (expr) {
        current_expr_span_ = expr->span_;
    }
    ir::IRVisitor::VisitExpr(expr);
}

void CCECodegen::GenerateBody(const ir::FunctionPtr& func)
{
    if (func->body_) {
        VisitStmt(func->body_);
    }

    emitter_.DecreaseIndent();
    emitter_.EmitLine("}");
}

void CCECodegen::VisitStmt_(const ir::AssignStmtPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null AssignStmt";
    INTERNAL_CHECK(op->var_ != nullptr) << "Internal error: AssignStmt has null var";
    INTERNAL_CHECK(op->value_ != nullptr) << "Internal error: AssignStmt has null value";

    auto target_var = op->var_;
    std::string var_name = context_.SanitizeName(target_var);
    context_.RegisterVar(target_var, var_name);

    // Both outputs of the same visit are needed here -- the code to bind the name to, and the
    // underlying MakeTuple to register -- so this walks the expression directly instead of
    // going through GetExprAsCode/GetExprAsMakeTuple, which each surface only one of the two.
    current_target_var_ = var_name;
    current_expr_value_ = "";
    current_tuple_ = nullptr;
    VisitExpr(op->value_);

    if (!current_expr_value_.empty()) {
        auto var_type = target_var->GetType();
        bool is_tile = ir::As<ir::TileType>(var_type) != nullptr;
        bool is_tuple = ir::As<ir::TupleType>(var_type) != nullptr;
        bool rhs_is_var = ir::As<ir::Var>(op->value_) != nullptr;
        // Tile/tuple aliases and simple var-to-var copies: register as alias (no code emitted).
        // Scalar complex expressions: emit "auto x = expr;" so codegen hoists the expression
        // into a C++ local variable rather than inlining it at every use site.
        if (is_tile || is_tuple || rhs_is_var || tile_addresses_.count(current_expr_value_)) {
            context_.RegisterVar(target_var, current_expr_value_);
        } else {
            // Scalar non-trivial expression: emit a local variable declaration so the
            // expression (e.g. min(), %, *) is computed once and not re-expanded at
            // every use site (avoids VFLoop Cond containing function calls).
            emitter_.EmitLine("auto " + var_name + " = " + current_expr_value_ + ";");
            context_.RegisterVar(target_var, var_name);
        }
    }
    if (current_tuple_) {
        tuple_var_to_make_tuple_.emplace(var_name, current_tuple_); // first-write wins

        // Materialize an array tuple only once. Constant propagation may expose
        // this same MakeTuple through multiple tuple Vars, but all aliases must
        // resolve through the tuple's unique backing array.
        auto make_tuple = current_tuple_;
        auto tt = ir::As<ir::TupleType>(target_var->GetType());
        if (IsArrayTuple(tt) && tuple_backing_arr_.count(make_tuple.get()) == 0) {
            auto elem_names = CollectTupleElemNames(op->var_);
            CHECK(!elem_names.empty()) << "Array tuple assignment to '" << var_name
                                       << "' has elements without a C++ name at " << op->span_.ToString();
            emitter_.EmitLine(BuildDynamicTupleArrayDecl(tt->types_[0], elem_names, var_name));
            tuple_backing_arr_.emplace(make_tuple.get(), var_name);
        }
    }
    if (tile_addresses_.count(current_expr_value_)) {
        tile_addresses_[var_name] = tile_addresses_[current_expr_value_];
    }

    current_tuple_ = nullptr;
    current_expr_value_ = "";
    current_target_var_ = "";
}

void CCECodegen::VisitStmt_(const ir::EvalStmtPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null EvalStmt";
    INTERNAL_CHECK(op->expr_ != nullptr) << "Internal error: EvalStmt has null expression";

    // EvalStmt: evaluate expression for side effects (e.g., sync operations)
    // Sync ops (set_flag, wait_flag, pipe_barrier) are registered with f_codegen_cce
    // and will be invoked via VisitExpr_(Call). Cross-core sync redundancy is
    // checked inside the per-op codegen handler (Make{Set,Wait}CrossCoreCodegenCCE).
    VisitExpr(op->expr_);
}

void CCECodegen::VisitStmt_(const ir::ReturnStmtPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null ReturnStmt";
    // A ReturnStmt reaching codegen is a kernel-level return. Emit a native C++ return so the
    // function exits immediately, matching Python's return semantics.
    // Kernel functions are void, so return values are discarded.
    emitter_.EmitLine("return;");
    current_expr_value_ = "";
}

void CCECodegen::VisitStmt_(const ir::YieldStmtPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null YieldStmt";

    if (op->value_.empty()) {
        yield_buffer_.clear();
        return; // No values to yield
    }

    yield_buffer_ = op->value_;
    current_expr_value_ = "";
}

void CCECodegen::EmitYieldAssignments(const std::vector<ir::VarPtr>& return_vars)
{
    if (return_vars.empty() || yield_buffer_.empty())
        return;
    CHECK(return_vars.size() == yield_buffer_.size()) << "IfStmt yield values must match its return variables";
    for (size_t i = 0; i < return_vars.size(); ++i) {
        EmitVariable(return_vars[i], yield_buffer_[i], false);
    }
    yield_buffer_.clear();
}

// ========================================================================
// Dynamic tuple array declaration: visitExpr-driven resolution
// ========================================================================

std::vector<std::string> CCECodegen::CollectTupleElemNames(const ir::ExprPtr& tuple_value)
{
    // Step 1: resolve the tuple expression to its underlying MakeTuple.
    auto mt = GetExprAsMakeTuple(tuple_value);
    if (!mt)
        return {};

    // Step 2: name each element. For dynamic-GetItem the elements are tile/scalar Var or const
    // literals, so each one has a C++ name of its own; anything else cannot back an array.
    std::vector<std::string> names;
    names.reserve(mt->elements_.size());
    for (const auto& elem : mt->elements_) {
        std::string name = GetExprAsCode(elem);
        if (name.empty())
            return {};
        names.push_back(std::move(name));
    }
    return names;
}

std::string CCECodegen::GetGeneratedType(const ir::TypePtr& type) const
{
    if (auto scalar_type = ir::As<ir::ScalarType>(type)) {
        return scalar_type->dtype_.ToCTypeString();
    }
    if (auto ptr_type = ir::As<ir::PtrType>(type)) {
        return "__gm__ " + ptr_type->dtype_.ToCTypeString() + "*";
    }
    if (auto tensor_type = ir::As<ir::TensorType>(type)) {
        return "__gm__ " + tensor_type->dtype_.ToCTypeString() + "*";
    }
    if (auto tile_type = ir::As<ir::TileType>(type)) {
        auto shape_dims = ExtractShapeDimensions(tile_type->shape_);
        int64_t rows = shape_dims.size() >= 1 ? shape_dims[0] : 1;
        int64_t cols = shape_dims.size() >= 2 ? shape_dims[1] : 1;
        return type_converter_.ConvertTileType(tile_type, rows, cols);
    }
    if (auto tuple_type = ir::As<ir::TupleType>(type)) {
        if (const std::string* struct_name = GetStructName(tuple_type)) {
            return *struct_name;
        }
    }
    // Single-element basic types only. Aggregate tuples and arrays are flattened to their
    // elements before reaching here, so an unregistered composite type is a codegen bug.
    CHECK(false) << "Unsupported type for CCE codegen: " << type->TypeName();
    return "";
}

std::string CCECodegen::BuildDynamicTupleArrayDecl(const ir::TypePtr& elem_type,
                                                   const std::vector<std::string>& elem_names,
                                                   const std::string& arr_name) const
{
    CHECK(!elem_names.empty()) << "Array tuple '" << arr_name << "' has no elements";

    std::string elem_cpp_type = GetGeneratedType(elem_type);

    std::ostringstream init;
    for (size_t i = 0; i < elem_names.size(); ++i) {
        if (i > 0)
            init << ", ";
        if (ir::As<ir::ScalarType>(elem_type)) {
            init << "static_cast<" << elem_cpp_type << ">(" << elem_names[i] << ")";
        } else {
            init << elem_names[i];
        }
    }

    return elem_cpp_type + " " + arr_name + "[] = {" + init.str() + "};";
}

bool CCECodegen::IsHomogeneousTuple(const ir::TupleTypePtr& tt) const
{
    if (!tt || tt->types_.empty())
        return false;
    for (size_t i = 1; i < tt->types_.size(); ++i) {
        if (!ir::structural_equal(tt->types_[i], tt->types_[0]))
            return false;
    }
    return true;
}

bool CCECodegen::IsArrayTuple(const ir::TupleTypePtr& tt) const
{
    if (!tt || tt->types_.empty())
        return false;
    // Named tuples / structs render as `base.field`; see VisitExpr_(GetItemExprPtr) case 1.
    if (debug_info_ != nullptr && debug_info_->GetTupleFields(tt.get()) != nullptr)
        return false;
    return IsHomogeneousTuple(tt);
}

void CCECodegen::EmitFullPhiIf(const ir::IfStmtPtr& op)
{
    // Declare the phi slots before the `if`: a phi has no initial value, but its C++ shape
    // follows from its type alone, including a struct name, which codegen seeds from IRDebugInfo
    // at entry rather than discovering as it walks the branches.
    for (const auto& return_var : op->returnVars_) {
        context_.RegisterVar(return_var, context_.SanitizeName(return_var));
        EmitVariable(return_var, nullptr, true);
    }

    VisitExpr(op->condition_);
    std::string condition = current_expr_value_;
    current_expr_value_ = "";

    emitter_.EmitLine("if (" + condition + ") {");
    emitter_.IncreaseIndent();
    VisitStmt(op->thenBody_);
    EmitYieldAssignments(op->returnVars_);
    emitter_.DecreaseIndent();

    if (op->elseBody_.has_value()) {
        emitter_.EmitLine("} else {");
        emitter_.IncreaseIndent();
        VisitStmt(*op->elseBody_);
        EmitYieldAssignments(op->returnVars_);
        emitter_.DecreaseIndent();
    }
    emitter_.EmitLine("}");
}

void CCECodegen::VisitStmt_(const ir::IfStmtPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null IfStmt";
    INTERNAL_CHECK(op->condition_ != nullptr) << "Internal error: IfStmt has null condition";
    INTERNAL_CHECK(op->thenBody_ != nullptr) << "Internal error: IfStmt has null then_body";

    // Drop phi return_vars with no downstream consumer. The SSA pass
    // conservatively inserts phi nodes whenever a variable is re-assigned
    // across control flow, but if no one reads the phi output, emitting a
    // declaration + per-branch yield-assignment is pure dead code.
    ir::IfStmtPtr effective_op = op;
    if (!op->returnVars_.empty()) {
        bool any_used = false;
        for (const auto& rv : op->returnVars_) {
            if (rv && var_read_names_.count(rv->name_)) {
                any_used = true;
                break;
            }
        }
        if (!any_used) {
            // An else-body consisting only of YieldStmts also becomes dead once
            // return_vars are dropped ->the yields have no consumer.
            std::optional<ir::StmtPtr> new_else = op->elseBody_;
            if (new_else.has_value() && IsOnlyYieldStmts(*new_else)) {
                new_else = std::nullopt;
            }
            effective_op = std::make_shared<ir::IfStmt>(op->condition_, op->thenBody_, new_else,
                                                        std::vector<ir::VarPtr>{}, op->span_);
        }
    }

    EmitFullPhiIf(effective_op);
}

// ========================================================================
// Phase 5 helpers: ForStmt iter-arg registration and yield assignments
// ========================================================================

bool CCECodegen::CheckEmitVariable(const ir::VarPtr& target, ir::ExprPtr& value, bool initialize)
{
    // The target is a declared slot, so its type must always be resolved.
    INTERNAL_CHECK(!ir::As<ir::NoneType>(target->GetType()))
        << "Variable '" << target->name_ << "' has an unresolved type";

    // A NoneType source carries no value: it is the `None` sentinel on a slot with no initial
    // value, or the placeholder a yield/break/continue emits for a merge variable that a branch
    // never wrote. Either way there is nothing to assign.
    if (value && ir::As<ir::NoneType>(value->GetType())) {
        value = nullptr;
    }
    // A control-flow back edge carries a value the body never modified, so the yielded
    // expression *is* the slot's own variable. Writing it back would emit `x = x`.
    if (value == target) {
        value = nullptr;
    }
    // With no value left, only a pending declaration is still worth emitting.
    return initialize || value != nullptr;
}

void CCECodegen::EmitVariable(const ir::VarPtr& target, ir::ExprPtr value, bool initialize)
{
    if (!CheckEmitVariable(target, value, initialize)) {
        return;
    }

    const auto& type = target->GetType();
    std::string name = context_.SanitizeName(target);
    if (ir::As<ir::TensorType>(type)) {
        throw ir::RuntimeError("Tensor variable code generation is not supported");
    }
    // A struct is a single C++ object with a type name of its own, so it is emitted like any
    // other scalar. Only a tuple without one needs splitting into an array or leaf slots.
    if (auto tuple_type = ir::As<ir::TupleType>(type); tuple_type && GetStructName(tuple_type) == nullptr) {
        EmitTupleVariable(name, tuple_type, value, initialize);
        return;
    }

    std::string source = value ? GetExprAsCode(value) : "";
    if (ir::As<ir::TileType>(type) && value) {
        auto address = tile_addresses_.find(source);
        if (address != tile_addresses_.end()) {
            tile_addresses_[name] = address->second;
        }
    }

    if (initialize) {
        emitter_.EmitLine(GetGeneratedType(type) + " " + name + (value ? " = " + source : "") + ";");
    } else {
        emitter_.EmitLine(name + " = " + source + ";");
    }
}

void CCECodegen::EmitTupleVariable(const std::string& name, const ir::TupleTypePtr& type, const ir::ExprPtr& value,
                                   bool initialize)
{
    ir::MakeTuplePtr source;
    ir::Span span = value ? value->span_ : ir::Span::Unknown();
    if (value) {
        source = GetExprAsMakeTuple(value);
        INTERNAL_CHECK_SPAN(source, span) << "Tuple assignment source does not resolve to MakeTuple";
    }

    // The representation follows the type, not the source value, so a merge variable can be
    // declared from its type alone once its control-flow structure has been traversed.
    // `initialize` means this is a fresh slot: define it and take ownership of its storage.
    if (IsArrayTuple(type)) {
        const auto& elem_type = type->types_[0];
        // A C++ array needs a single element spelling. Elements are basic types or registered
        // structs; a plain nested tuple would need a second dimension, which is not supported.
        auto elem_tuple = ir::As<ir::TupleType>(elem_type);
        CHECK(!elem_tuple || GetStructName(elem_tuple) != nullptr)
            << "Only one-dimensional array tuples are supported, but '" << name << "' has tuple elements at "
            << span.ToString();

        if (initialize) {
            // The declaration depends only on the type; only the initial value comes from source.
            auto target_tuple = std::make_shared<ir::MakeTuple>(source ? source->elements_ : std::vector<ir::ExprPtr>{},
                                                                span);
            tuple_var_to_make_tuple_[name] = target_tuple;
            tuple_backing_arr_[target_tuple.get()] = name;
            emitter_.EmitLine(GetGeneratedType(elem_type) + " " + name + "[" + std::to_string(type->types_.size()) +
                              "];");
        }
        if (source) {
            auto source_array = tuple_backing_arr_.find(source.get());
            INTERNAL_CHECK_SPAN(source_array != tuple_backing_arr_.end(), span)
                << "Array tuple assignment to '" << name << "' requires an array source";
            if (name != source_array->second) {
                for (size_t i = 0; i < type->types_.size(); ++i) {
                    std::string index = "[" + std::to_string(i) + "]";
                    emitter_.EmitLine(name + index + " = " + source_array->second + index + ";");
                }
            }
        }
        return;
    }

    // Aggregate: flatten into one leaf slot per element, recursing into nested tuples.
    if (initialize) {
        std::vector<ir::ExprPtr> elements;
        elements.reserve(type->types_.size());
        for (size_t i = 0; i < type->types_.size(); ++i) {
            std::string element_name = name + "_" + std::to_string(i);
            ir::ExprPtr element_value = source ? source->elements_[i] : nullptr;
            // A struct is a leaf: it has a C++ type name of its own, so it is declared and
            // assigned whole. Only a tuple without one is taken apart further.
            if (auto element_type = ir::As<ir::TupleType>(type->types_[i]);
                element_type && GetStructName(element_type) == nullptr) {
                EmitTupleVariable(element_name, element_type, element_value, true);
                elements.push_back(tuple_var_to_make_tuple_.at(element_name));
            } else {
                auto element = std::make_shared<ir::Var>(element_name, type->types_[i], span);
                context_.RegisterVar(element, element_name);
                elements.push_back(element);
                EmitVariable(element, element_value, true);
            }
        }
        tuple_var_to_make_tuple_[name] = std::make_shared<ir::MakeTuple>(std::move(elements), span);
        return;
    }

    // Reuse the leaf slots recorded when this aggregate was defined.
    auto target = tuple_var_to_make_tuple_.find(name);
    INTERNAL_CHECK_SPAN(target != tuple_var_to_make_tuple_.end(), span)
        << "Assignment to undefined aggregate tuple '" << name << "'";
    for (size_t i = 0; i < type->types_.size(); ++i) {
        ir::ExprPtr element_value = source ? source->elements_[i] : nullptr;
        if (auto element_type = ir::As<ir::TupleType>(type->types_[i]);
            element_type && GetStructName(element_type) == nullptr) {
            EmitTupleVariable(name + "_" + std::to_string(i), element_type, element_value, false);
            continue;
        }
        auto element = ir::As<ir::Var>(target->second->elements_[i]);
        INTERNAL_CHECK_SPAN(element, span) << "Aggregate tuple '" << name << "' has no leaf slot at " << i;
        EmitVariable(element, element_value, false);
    }
}

std::vector<std::string> CCECodegen::RegisterLoopIterArgs(const std::vector<ir::IterArgPtr>& iterArgs)
{
    // Registration only: the declarations are emitted by EmitLoop after the body has been
    // traversed, because a slot whose init is UnknownType is not fully typed until then.
    std::vector<std::string> iterArgNames;
    iterArgNames.reserve(iterArgs.size());

    for (auto& iterArg : iterArgs) {
        const auto& var = iterArg->iterVar_;
        std::string iterArgName = context_.SanitizeName(var);

        context_.RegisterVar(var, iterArgName);
        iterArgNames.push_back(iterArgName);
    }
    return iterArgNames;
}

void CCECodegen::EmitCarriedAssignments(const std::vector<ir::IterArgPtr>& targets,
                                        const std::vector<ir::ExprPtr>& sources)
{
    CHECK(targets.size() == sources.size())
        << "Loop-carried write-back expects " << targets.size() << " values but got " << sources.size();

    for (size_t i = 0; i < targets.size(); ++i) {
        const auto& var = targets[i]->iterVar_;
        EmitVariable(var, sources[i], false);
    }
}

void CCECodegen::EmitForYieldAssignments(const std::vector<ir::IterArgPtr>& iterArgs)
{
    if (yield_buffer_.empty()) {
        return;
    }
    EmitCarriedAssignments(iterArgs, yield_buffer_);
    yield_buffer_.clear();
}

void CCECodegen::RegisterLoopReturnVars(const std::vector<ir::VarPtr>& returnVars,
                                        const std::vector<std::string>& iterArgNames)
{
    if (returnVars.empty()) {
        return;
    }
    for (size_t i = 0; i < returnVars.size(); ++i) {
        const auto& returnVar = returnVars[i];
        if (i < iterArgNames.size()) {
            context_.RegisterVar(returnVar, iterArgNames[i]);
            if (ir::As<ir::TensorType>(returnVar->GetType())) {
                RegisterPointer(context_.SanitizeName(returnVar), GetPointer(iterArgNames[i]));
            }
            if (ir::As<ir::TupleType>(returnVar->GetType())) {
                auto tuple_it = tuple_var_to_make_tuple_.find(iterArgNames[i]);
                if (tuple_it != tuple_var_to_make_tuple_.end()) {
                    tuple_var_to_make_tuple_[context_.SanitizeName(returnVar)] = tuple_it->second;
                }
            }
        } else {
            throw ir::RuntimeError("Loop return_var has no corresponding iter_arg");
        }
    }
}

void CCECodegen::VisitStmt_(const ir::ForStmtPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null ForStmt";
    INTERNAL_CHECK(op->loopVar_ != nullptr) << "Internal error: ForStmt has null loop_var";
    INTERNAL_CHECK(op->start_ != nullptr) << "Internal error: ForStmt has null start";
    INTERNAL_CHECK(op->stop_ != nullptr) << "Internal error: ForStmt has null stop";
    INTERNAL_CHECK(op->step_ != nullptr) << "Internal error: ForStmt has null step";
    INTERNAL_CHECK(op->body_ != nullptr) << "Internal error: ForStmt has null body";

    // Check consistency: iter_args and return_vars must have same size
    CHECK(op->iterArgs_.size() == op->returnVars_.size())
        << "ForStmt iter_args size (" << op->iterArgs_.size() << ") must equal return_vars size ("
        << op->returnVars_.size() << ")";

    auto stop_ci = ir::As<ir::ConstInt>(op->stop_);

    // Register loop variable
    std::string loop_var_name = context_.SanitizeName(op->loopVar_);
    context_.RegisterVar(op->loopVar_, loop_var_name);

    std::vector<std::string> iter_arg_names = RegisterLoopIterArgs(op->iterArgs_);

    // Evaluate loop range
    VisitExpr(op->start_);
    std::string start = current_expr_value_;
    current_expr_value_ = "";

    VisitExpr(op->stop_);
    std::string stop = current_expr_value_;
    current_expr_value_ = "";

    VisitExpr(op->step_);
    std::string step = current_expr_value_;
    current_expr_value_ = "";

    // --- Emit for-loop with hoisting ---
    // In __VEC_SCOPE__, bisheng requires uint16_t loop variables AND the bound expression to also
    // resolve to uint16_t (otherwise -Wcce-compat -Werror fails the build), so cast inside a VF section.
    std::string loop_type = IsInVFSection() ? "uint16_t" : "uint64_t";
    std::string stop_expr = IsInVFSection() ? ("(uint16_t)(" + stop + ")") : stop;
    // Perf: in a VF section, hoist a NON-constant loop bound into a `const` local
    // and use that as the trip count.  A literal-constant bound is already ideal,
    // so it is left inline.  Binding a variable bound to a const evaluates it once
    // and lets the bisheng VF compiler treat the trip count as loop-invariant
    // (enabling unrolling / vectorization it would otherwise skip).
    if (IsInVFSection() && !stop_ci) {
        std::string bound_name = loop_var_name + "_ub";
        emitter_.EmitLine("const uint16_t " + bound_name + " = " + stop_expr + ";");
        stop_expr = bound_name;
    }
    std::string header = "for (" + loop_type + " " + loop_var_name + " = " + start + "; " + loop_var_name + " < " +
                         stop_expr + "; " + loop_var_name + " += " + step + ") {";
    EmitLoop(header, op->body_, op->iterArgs_, op->returnVars_, iter_arg_names);
}

void CCECodegen::EmitLoop(const std::string& header, const ir::StmtPtr& body,
                          const std::vector<ir::IterArgPtr>& iterArgs, const std::vector<ir::VarPtr>& returnVars,
                          const std::vector<std::string>& iterArgNames)
{
    // The iter-arg slots are declared before the loop: every type they can take is known up
    // front, including a struct name, which codegen seeds from IRDebugInfo at entry rather than
    // discovering as it walks the body. An UnknownType init (the `None` sentinel for a slot with
    // no initial value) becomes a plain declaration inside EmitVariable.
    for (const auto& iterArg : iterArgs) {
        EmitVariable(iterArg->iterVar_, iterArg->initValue_, true);
    }
    if (!iterArgs.empty()) {
        emitter_.EmitLine("");
    }

    emitter_.EmitLine(header);
    emitter_.IncreaseIndent();

    // Expose this loop's write-back targets so a native break/continue in the body can assign its
    // self-described carried values (BreakStmt::value_) to the iter_args before jumping.
    loop_target_stack_.push_back(iterArgs);
    yield_buffer_.clear();
    VisitStmt(body);
    loop_target_stack_.pop_back();

    if (!iterArgs.empty()) {
        EmitForYieldAssignments(iterArgs);
    }

    emitter_.DecreaseIndent();
    emitter_.EmitLine("}");

    // Return variables capture the final iteration values; alias them to iter_args.
    RegisterLoopReturnVars(returnVars, iterArgNames);
}

void CCECodegen::VisitStmt_(const ir::WhileStmtPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null WhileStmt";
    INTERNAL_CHECK(op->condition_ != nullptr) << "Internal error: WhileStmt has null condition";
    INTERNAL_CHECK(op->body_ != nullptr) << "Internal error: WhileStmt has null body";

    // iter_args and return_vars must agree in size (each loop-carried value has a phi).
    CHECK(op->iterArgs_.size() == op->returnVars_.size())
        << "WhileStmt iter_args size (" << op->iterArgs_.size() << ") must equal return_vars size ("
        << op->returnVars_.size() << ")";

    // Native break/continue inside the body are emitted directly; each
    // jump carries its own loop-carried values (BreakStmt::value_, populated by ConvertToSSA) and
    // writes them to the iter_args before jumping. The body otherwise ends in a YieldStmt that feeds
    // the loop-carried iter_args on the normal loop-back path.
    std::vector<std::string> iter_arg_names = RegisterLoopIterArgs(op->iterArgs_);

    // The condition is re-evaluated each iteration, so emit it inline in the while header.
    // Comparison/logical exprs lower to pure C++ expression strings (see IMPLEMENT_BINARY_OP),
    // so they are safe to place directly in `while (...)`.
    VisitExpr(op->condition_);
    std::string condition = current_expr_value_;
    current_expr_value_ = "";

    EmitLoop("while (" + condition + ") {", op->body_, op->iterArgs_, op->returnVars_, iter_arg_names);
}

// ========================================================================
// Native break/continue support
// ========================================================================

// Write a native jump's self-described carried values (BreakStmt/ContinueStmt::value_, populated by
// ConvertToSSA) back to the innermost loop's iter_args before the jump skips the trailing yield.
void CCECodegen::EmitJumpCarriedWriteback(const std::vector<ir::ExprPtr>& values)
{
    if (values.empty() || loop_target_stack_.empty()) {
        return; // bare jump: loop carries nothing, or jump is outside a tracked loop
    }
    EmitCarriedAssignments(loop_target_stack_.back(), values);
}

void CCECodegen::VisitStmt_(const ir::BreakStmtPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null BreakStmt";
    EmitJumpCarriedWriteback(op->value_);
    emitter_.EmitLine("break;");
}

void CCECodegen::VisitStmt_(const ir::ContinueStmtPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null ContinueStmt";
    EmitJumpCarriedWriteback(op->value_);
    emitter_.EmitLine("continue;");
}

void CCECodegen::VisitStmt_(const ir::SeqStmtsPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null SeqStmts";
    for (const auto& stmt : op->stmts_) {
        VisitStmt(stmt);
        // A native break/continue/return ends this straight-line sequence; statements after it are
        // unreachable (e.g. the dead trailing yield ConvertToSSA leaves after a body-tail jump).
        if (ir::As<ir::BreakStmt>(stmt) || ir::As<ir::ContinueStmt>(stmt) || ir::As<ir::ReturnStmt>(stmt)) {
            break;
        }
    }
}

// ========================================================================
// Expression Visitor Methods - Dual-Mode Pattern
// ========================================================================
// - Statement-Emitting Mode (Call): Uses current_target_var_, emits instructions
// - Value-Returning Mode (others): Sets current_expr_value_ with inline C++ code

// ---- Leaf Nodes ----
void CCECodegen::VisitExpr_(const ir::VarPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null Var";
    std::string name = context_.SanitizeName(op);
    auto it = tuple_var_to_make_tuple_.find(name);
    if (it == tuple_var_to_make_tuple_.end())
        it = tuple_var_to_make_tuple_.find(name + "_0");
    // Positional tuple Var: produce the underlying MakeTuple in current_tuple_
    // (with _0 SSA fallback). Empty current_expr_value_ signals "no direct identifier".
    current_tuple_ = (it != tuple_var_to_make_tuple_.end()) ? it->second : nullptr;
    current_expr_value_ = context_.GetVarName(op);
}

void CCECodegen::VisitExpr_(const ir::MakeTuplePtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null MakeTuple";
    current_tuple_ = op;
    current_expr_value_ = "";
}

void CCECodegen::VisitExpr_(const ir::ConstIntPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null ConstInt";
    current_expr_value_ = std::to_string(op->value_);
}

void CCECodegen::VisitExpr_(const ir::ConstFloatPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null ConstFloat";
    std::string literal = std::to_string(op->value_);
    literal += "f";
    current_expr_value_ = literal;
}

void CCECodegen::VisitExpr_(const ir::ConstBoolPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null ConstBool";
    current_expr_value_ = op->value_ ? "true" : "false";
}

std::string CCECodegen::GetOrCreateVFTilePtr(const ir::ExprPtr& expr, bool is_post_update)
{
    std::string code = GetExprAsCode(expr);
    // Key on is_post_update so a dedicated cursor (e.g. a POST_UPDATE store target) and a plain
    // base pointer to the same tile are kept as independent hoisted vars — a cursor that is
    // mutated in place must not be shared with a base-pointer load/store of the same tile.
    auto cache_key = std::make_pair(is_post_update, code);
    auto it = vf_tile_ptrs_.find(cache_key);
    if (it != vf_tile_ptrs_.end())
        return it->second;

    auto tile_type = ir::As<ir::TileType>(expr->GetType());
    INTERNAL_CHECK(tile_type != nullptr) << "GetOrCreateVFTilePtr expects a tile expr, got " << expr->TypeName();
    std::string elem_ctype = tile_type->dtype_.ToCTypeString();

    // tile[offset] is the rvalue `(vf_tile_ptr_N + off)`; a load uses it as-is, no new variable.
    bool is_offset = code.find("vf_tile_ptr_") != std::string::npos;
    if (is_offset && !is_post_update)
        return code;

    // Declare a vf_tile_ptr_N once, cached and hoisted before __VEC_SCOPE__: plain tile -> base from
    // .data(); post_update -> a cursor the intrinsic advances in place.
    std::string var = "vf_tile_ptr_" + std::to_string(vf_tile_ptr_counter_++);
    std::string init = is_offset ? code : ("(__ubuf__ " + elem_ctype + " *)" + code + ".data()");
    section_hoisted_decls_.push_back("__ubuf__ " + elem_ctype + " *" + var + " = " + init + ";");
    vf_tile_ptrs_[cache_key] = var;
    return var;
}

void CCECodegen::VisitExpr_(const ir::GetItemExprPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null GetItemExpr";

    auto value_type = op->value_->GetType();
    CHECK(ir::As<ir::TupleType>(value_type))
        << "GetItemExpr requires value to have TupleType, got " << value_type->TypeName();

    std::string index_code = GetExprAsCode(op->slice_);
    // Which of the three cases below applies depends on both outputs of one visit of the base,
    // so this walks it directly rather than through the single-output accessors.
    current_tuple_ = nullptr;
    current_expr_value_ = "";
    VisitExpr(op->value_);

    // Case 1: the value is already a materialized expression. Named TupleTypes
    // represent structs and require a constant field ordinal. Other TupleTypes
    // represent indexable expressions such as an array-valued struct field.
    if (current_tuple_ == nullptr) {
        std::string base_code = current_expr_value_;
        auto tuple_type = ir::As<ir::TupleType>(value_type);
        const std::vector<std::string>* fields = debug_info_ != nullptr ?
                                                     debug_info_->GetTupleFields(tuple_type.get()) :
                                                     nullptr;
        if (fields != nullptr) {
            auto const_idx = ir::As<ir::ConstInt>(op->slice_);
            CHECK(const_idx != nullptr) << "GetItemExpr struct field requires a constant index at "
                                        << op->span_.ToString();
            int idx = static_cast<int>(const_idx->value_);
            CHECK(idx >= 0 && idx < static_cast<int>(fields->size()))
                << "GetItemExpr struct field index " << idx << " out of bounds";
            current_expr_value_ = base_code + "." + (*fields)[idx];
        } else {
            current_expr_value_ = base_code + "[" + index_code + "]";
        }
        current_tuple_ = nullptr;
        return;
    }

    // Case 2: a homogeneous MakeTuple has a backing C++ array, so both constant
    // and dynamic indices must be emitted as array accesses.
    auto arr_it = tuple_backing_arr_.find(current_tuple_.get());
    if (arr_it != tuple_backing_arr_.end()) {
        current_tuple_ = nullptr;
        current_expr_value_ = arr_it->second + "[" + index_code + "]";
        return;
    }

    // Case 3: a MakeTuple without a backing array can only be evaluated
    // statically. Dynamic indexing is valid only for homogeneous tuples, which
    // must have been materialized as a backing array above.
    auto const_idx = ir::As<ir::ConstInt>(op->slice_);
    CHECK(const_idx != nullptr) << "Dynamic GetItemExpr requires a backing array at " << op->span_.ToString();
    int idx = static_cast<int>(const_idx->value_);
    CHECK(idx >= 0 && idx < static_cast<int>(current_tuple_->elements_.size()))
        << "GetItemExpr index " << idx << " out of bounds";
    ir::ExprPtr elem = current_tuple_->elements_[idx];
    current_tuple_ = nullptr;
    current_expr_value_ = "";
    VisitExpr(elem);
    return;
}

// ========================================================================
// CodegenBase interface and CCE-specific helper methods
// ========================================================================

// Both accessors below clear current_expr_value_ and current_tuple_ on the way in and on the
// way out. Clearing on the way in is what makes the result describe this expression alone:
// only Var and MakeTuple write current_tuple_ on every visit, so a scalar expression would
// otherwise inherit whichever tuple the previous visit left behind. Clearing on the way out
// keeps that same leftover from reaching whatever is parsed next.

std::string CCECodegen::GetExprAsCode(const ir::ExprPtr& expr)
{
    auto saved_span = current_expr_span_;
    current_expr_value_ = "";
    current_tuple_ = nullptr;
    VisitExpr(expr);
    auto result = current_expr_value_;
    current_expr_value_ = "";
    current_tuple_ = nullptr;
    current_expr_span_ = saved_span;
    return result;
}

ir::MakeTuplePtr CCECodegen::GetExprAsMakeTuple(const ir::ExprPtr& expr)
{
    auto saved_span = current_expr_span_;
    current_expr_value_ = "";
    current_tuple_ = nullptr;
    VisitExpr(expr);
    auto result = current_tuple_;
    current_expr_value_ = "";
    current_tuple_ = nullptr;
    current_expr_span_ = saved_span;
    return result;
}

void CCECodegen::Emit(const std::string& line) { emitter_.EmitLine(line); }

std::string CCECodegen::GetTypeString(const ir::DataType& dtype) const { return dtype.ToCTypeString(); }

int64_t CCECodegen::GetConstIntValue(const ir::ExprPtr& expr) { return ExtractConstInt(expr); }

std::string CCECodegen::GetVarName(const ir::VarPtr& var) { return context_.GetVarName(var); }

void CCECodegen::RegisterPointer(const std::string& tensor_var_name, const std::string& ptr_name)
{
    CHECK(!tensor_var_name.empty()) << "Cannot register pointer with empty tensor var name";
    CHECK(!ptr_name.empty()) << "Cannot register pointer with empty pointer name";

    auto it = tensor_to_pointer_.find(tensor_var_name);
    if (it != tensor_to_pointer_.end() && it->second != ptr_name) {
        IR_LOGW() << "Pointer for tensor " << tensor_var_name << " re-registered with: " << ptr_name << " vs "
                  << it->second;
    }
    tensor_to_pointer_[tensor_var_name] = ptr_name;
}

std::string CCECodegen::GetPointer(const std::string& var_name)
{
    auto it = tensor_to_pointer_.find(var_name);
    CHECK(it != tensor_to_pointer_.end()) << "Pointer for tensor " << var_name << " not found";
    return it->second;
}

bool CCECodegen::HasPointer(const std::string& var_name) const
{
    return tensor_to_pointer_.find(var_name) != tensor_to_pointer_.end();
}

bool CCECodegen::HasTileAddress(const std::string& tile_name) const
{
    return tile_addresses_.find(tile_name) != tile_addresses_.end();
}

const TensorDef* CCECodegen::GetTensorDef(const std::string& name) const
{
    auto it = tensor_defs_.find(name);
    return it != tensor_defs_.end() ? &it->second : nullptr;
}

std::string CCECodegen::ComputeIRBasedOffset(const ir::TensorTypePtr& tensor_type, const ir::MakeTuplePtr& offsets)
{
    size_t ndim = tensor_type->shape_.size();
    CHECK(offsets->elements_.size() == ndim)
        << "Offset dimensions (" << offsets->elements_.size() << ") != tensor dimensions (" << ndim << ")";
    const auto stride_exprs = BuildTensorStrideExpressions(tensor_type);

    std::ostringstream result;
    result << "(";
    bool first = true;
    for (size_t i = 0; i < ndim; ++i) {
        std::string off_expr = GetExprAsCode(offsets->elements_[i]);

        // TensorView stride is used when present; otherwise a contiguous stride is derived from the shape.
        // For a unit stride, add the offset directly.
        if (!first)
            result << " + ";
        first = false;
        result << off_expr;
        if (stride_exprs[i] != "1") {
            result << " * (" << stride_exprs[i] << ")";
        }
    }
    result << ")";
    return result.str();
}

// ========================================================================
// Call Expression Visitor (uses operator registry codegen functions)
// ========================================================================

void CCECodegen::VisitExpr_(const ir::CallPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null Call";

    CHECK(backend_ != nullptr) << "CCE backend must not be null";
    const auto* op_info = backend_->GetOpInfo(op->name_);
    CHECK(op_info != nullptr) << "Unknown call '" << op->name_ << "' reached CCE codegen; helper calls must be inlined";

    // Auto-declare RegTensor for VF compute ops whose dst variable hasn't been
    // declared yet. This handles the VF assignment form (dst = vf.xxx(...))
    // where the RegTensor let-binding may end up inside a loop body.
    // Only applies to compute ops where args[0] is a register (not a tile/store dst).
    if (op->name_.rfind("vf.", 0) == 0 && !op->args_.empty()) {
        // Skip store/side-effect ops — args[0] is a tile pointer, not a register
        static const std::set<std::string> skip_ops = {
            "vf.store_align",
            "vf.store_unalign",
            "vf.store_unalign_post",
            "vf.store",
            "vf.scatter",
            "vf.store_align_pack",
            "vf.store_align_intlv",
            "vf.store_align_pack_postupdate",
            "vf.mem_bar",
            "vf.clear_spr",
            "vf.create_mask",
            "vf.update_mask",
            "vf.reg_tensor",
            "vf.mask_reg",
            "vf.create_addr_reg",
            "vf.move",
            "vf.unalign_reg_for_store",
            "vf.load_unalign_init",
            "vf.load_unalign_pre",
            "vf.get_mask_spr",
            "vf.mask_gen_with_reg_tensor",
            "vf.compare",
            "vf.compares",
            "vf.bit_cast",
        };
        if (skip_ops.count(op->name_) == 0) {
            static const std::set<std::string> dual_dst_ops = {
                "vf.interleave",
                "vf.de_interleave",
                "vf.mull",
            };
            bool is_dual_dst = dual_dst_ops.count(op->name_) > 0;

            for (size_t dst_idx = 0; dst_idx < (is_dual_dst ? 2 : 1); ++dst_idx) {
                auto dst_var = ir::As<ir::Var>(op->args_[dst_idx]);
                if (!dst_var)
                    continue;
                std::string dst_name = GetVarName(dst_var);
                if (dst_name.empty() || IsRegTensorVar(dst_name) || IsMaskRegVar(dst_name))
                    continue;
                // Skip if the variable name contains '[' or '(' — it's a tile
                // group index expression, not a simple register variable.
                if (dst_name.find('[') != std::string::npos || dst_name.find('(') != std::string::npos)
                    continue;

                // Determine dtype: prefer the dst var's own type, then source args.
                DataType dt = DataType::FP32;
                auto dst_type = op->args_[dst_idx]->GetType();
                if (auto dst_st = ir::As<ir::ScalarType>(dst_type)) {
                    dt = dst_st->dtype_;
                } else if (auto dst_sh = ir::As<ir::ShapedType>(dst_type)) {
                    dt = dst_sh->dtype_;
                } else {
                    size_t first_src_idx = is_dual_dst ? 2 : 1;
                    if (op->args_.size() > first_src_idx) {
                        auto src_type = op->args_[first_src_idx]->GetType();
                        if (auto src_st = ir::As<ir::ScalarType>(src_type))
                            dt = src_st->dtype_;
                        else if (auto src_sh = ir::As<ir::ShapedType>(src_type))
                            dt = src_sh->dtype_;
                    }
                }
                vf_reg_hoisted_decls_.push_back("RegTensor<" + dt.ToCTypeString() + "> " + dst_name + ";");
                RegisterRegTensorVar(dst_name);
            }
        }
    }

    std::string result = op_info->codegen_func(op, *this);
    current_expr_value_ = result;
}

// ---- Binary Operators ----

#define IMPLEMENT_BINARY_OP(OpType, OpName, CppOp)                            \
    void CCECodegen::VisitExpr_(const ir::OpType##Ptr& op)                    \
    {                                                                         \
        INTERNAL_CHECK(op != nullptr) << "Internal error: null " << (OpName); \
        VisitExpr(op->left_);                                                 \
        std::string left = current_expr_value_;                               \
        VisitExpr(op->right_);                                                \
        std::string right = current_expr_value_;                              \
        current_expr_value_ = "(" + left + " " + (CppOp) + " " + right + ")"; \
    }

// Arithmetic operators
IMPLEMENT_BINARY_OP(Add, "Add", "+")
IMPLEMENT_BINARY_OP(Sub, "Sub", "-")
IMPLEMENT_BINARY_OP(Mul, "Mul", "*")
IMPLEMENT_BINARY_OP(FloorDiv, "FloorDiv", "/")
IMPLEMENT_BINARY_OP(FloorMod, "FloorMod", "%")
IMPLEMENT_BINARY_OP(FloatDiv, "FloatDiv", "/")

// Comparison operators
IMPLEMENT_BINARY_OP(Eq, "Eq", "==")
IMPLEMENT_BINARY_OP(Ne, "Ne", "!=")
IMPLEMENT_BINARY_OP(Lt, "Lt", "<")
IMPLEMENT_BINARY_OP(Le, "Le", "<=")
IMPLEMENT_BINARY_OP(Gt, "Gt", ">")
IMPLEMENT_BINARY_OP(Ge, "Ge", ">=")

// Logical operators
IMPLEMENT_BINARY_OP(And, "And", "&&")
IMPLEMENT_BINARY_OP(Or, "Or", "||")
IMPLEMENT_BINARY_OP(Xor, "Xor", "^")

// Bitwise operators
IMPLEMENT_BINARY_OP(BitAnd, "BitAnd", "&")
IMPLEMENT_BINARY_OP(BitOr, "BitOr", "|")
IMPLEMENT_BINARY_OP(BitXor, "BitXor", "^")
IMPLEMENT_BINARY_OP(BitShiftLeft, "BitShiftLeft", "<<")
IMPLEMENT_BINARY_OP(BitShiftRight, "BitShiftRight", ">>")

#undef IMPLEMENT_BINARY_OP

// Special binary operators (function calls)
void CCECodegen::VisitExpr_(const ir::MinPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null Min";
    auto scalar_type = std::dynamic_pointer_cast<const ir::ScalarType>(op->GetType());
    std::string cpp_type = scalar_type ? scalar_type->dtype_.ToCTypeString() : "int64_t";
    VisitExpr(op->left_);
    std::string left = current_expr_value_;
    VisitExpr(op->right_);
    std::string right = current_expr_value_;
    current_expr_value_ = "min((" + cpp_type + ")(" + left + "), (" + cpp_type + ")(" + right + "))";
}

void CCECodegen::VisitExpr_(const ir::MaxPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null Max";
    auto scalar_type = std::dynamic_pointer_cast<const ir::ScalarType>(op->GetType());
    std::string cpp_type = scalar_type ? scalar_type->dtype_.ToCTypeString() : "int64_t";
    VisitExpr(op->left_);
    std::string left = current_expr_value_;
    VisitExpr(op->right_);
    std::string right = current_expr_value_;
    current_expr_value_ = "max((" + cpp_type + ")(" + left + "), (" + cpp_type + ")(" + right + "))";
}

void CCECodegen::VisitExpr_(const ir::PowPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null Pow";
    VisitExpr(op->left_);
    std::string left = current_expr_value_;
    VisitExpr(op->right_);
    std::string right = current_expr_value_;
    current_expr_value_ = "pow(" + left + ", " + right + ")";
}

// ---- Unary Operators ----

#define IMPLEMENT_UNARY_OP(OpType, OpName, CppOp)                                     \
    void CCECodegen::VisitExpr_(const ir::OpType##Ptr& op)                            \
    {                                                                                 \
        INTERNAL_CHECK(op != nullptr) << "Internal error: null " << (OpName);         \
        VisitExpr(op->operand_);                                                      \
        current_expr_value_ = std::string("(") + (CppOp) + current_expr_value_ + ")"; \
    }

IMPLEMENT_UNARY_OP(Neg, "Neg", "-")
IMPLEMENT_UNARY_OP(Not, "Not", "!")
IMPLEMENT_UNARY_OP(BitNot, "BitNot", "~")

#undef IMPLEMENT_UNARY_OP

// Special unary operators
void CCECodegen::VisitExpr_(const ir::AbsPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null Abs";
    VisitExpr(op->operand_);
    std::string operand = current_expr_value_;
    current_expr_value_ = "abs(" + operand + ")";
}

void CCECodegen::VisitExpr_(const ir::CastPtr& op)
{
    INTERNAL_CHECK(op != nullptr) << "Internal error: null Cast";
    VisitExpr(op->operand_);
    std::string operand = current_expr_value_;

    auto scalar_type = std::dynamic_pointer_cast<const ir::ScalarType>(op->GetType());
    CHECK(scalar_type != nullptr) << "Cast target must be ScalarType";

    std::string cpp_type = scalar_type->dtype_.ToCTypeString();
    current_expr_value_ = "((" + cpp_type + ")" + operand + ")";
}

// ========================================================================
// End of Expression Visitor Methods
// ========================================================================

int64_t CCECodegen::ExtractConstInt(const ir::ExprPtr& expr) const
{
    auto const_int = std::dynamic_pointer_cast<const ir::ConstInt>(expr);
    CHECK(const_int != nullptr) << "Expected constant integer expression";
    return const_int->value_;
}

namespace {

/**
 * \brief Helper visitor for collecting TileType variables from IR
 *
 * Traverses the IR tree and collects all variables with TileType.
 * Uses the visitor pattern for clean, extensible traversal.
 */
class TileCollector : public ir::IRVisitor {
    using ir::IRVisitor::VisitExpr_;
    using ir::IRVisitor::VisitStmt_;

public:
    std::vector<std::pair<ir::VarPtr, ir::TileTypePtr>> tile_vars_;

    // Extract the first TileType from a possibly nested TupleType
    static ir::TileTypePtr ExtractLeafTileType(const ir::TypePtr& type)
    {
        if (auto tile = std::dynamic_pointer_cast<const ir::TileType>(type)) {
            return tile;
        }
        if (auto tuple = std::dynamic_pointer_cast<const ir::TupleType>(type)) {
            for (const auto& elem : tuple->types_) {
                auto result = ExtractLeafTileType(elem);
                if (result)
                    return result;
            }
        }
        return nullptr;
    }

    // Recursively collect all TileTypes from a type (handles TupleType nesting)
    static void CollectTileTypesFromType(const ir::TypePtr& type, std::vector<ir::TileTypePtr>& result)
    {
        if (auto tile = std::dynamic_pointer_cast<const ir::TileType>(type)) {
            result.push_back(tile);
        } else if (auto tuple = std::dynamic_pointer_cast<const ir::TupleType>(type)) {
            for (const auto& elem : tuple->types_) {
                CollectTileTypesFromType(elem, result);
            }
        }
    }

    void VisitStmt_(const ir::AssignStmtPtr& op) override
    {
        auto target_var = op->var_;
        auto var_type = target_var->GetType();
        // Direct TileType variable
        if (auto tile_type = std::dynamic_pointer_cast<const ir::TileType>(var_type)) {
            tile_vars_.emplace_back(target_var, tile_type);
            return;
        }
        // TupleType variable (from make_tile or double-buffer patterns):
        // Expand each TileType element as a separate tile with indexed name.
        if (std::dynamic_pointer_cast<const ir::TupleType>(var_type)) {
            std::vector<ir::TileTypePtr> tile_types;
            CollectTileTypesFromType(var_type, tile_types);
            for (size_t i = 0; i < tile_types.size(); ++i) {
                // Create a synthetic Var for each tuple element: varname_0, varname_1, ...
                auto elem_var = std::make_shared<ir::Var>(target_var->name_ + "_" + std::to_string(i), tile_types[i],
                                                          target_var->span_);
                tile_vars_.emplace_back(elem_var, tile_types[i]);
            }
        }
    }
};

/**
 * \brief Helper visitor for collecting tensor definitions from block.load/store
 *
 * Traverses the IR tree to find block.load/block.store calls and aggregates, for
 * each tensor variable (keyed by its cce variable name), the access window shape,
 * tile_dims and DN flag into a single TensorDef.
 */
class TensorDefCollector : public ir::IRVisitor {
    using ir::IRVisitor::VisitExpr_;
    using ir::IRVisitor::VisitStmt_;

public:
    explicit TensorDefCollector(const CodeContext& ctx) : ctx_(ctx) {}

    std::map<std::string, TensorDef> defs_; ///< cce var name -> aggregated TensorDef

    void VisitStmt_(const ir::AssignStmtPtr& op) override
    {
        // Keep aliases distinct for SSA/body codegen, but attribute their accesses to the
        // underlying parameter or view whose GlobalTensor declaration owns the base pointer.
        auto source = ir::As<ir::Var>(op->value_);
        if (ir::As<ir::TensorType>(op->var_->GetType()) && source && ir::As<ir::TensorType>(source->GetType())) {
            tensor_aliases_[op->var_.get()] = ResolveTensorAlias(source);
        }
        ir::IRVisitor::VisitStmt_(op);
    }

    void VisitExpr_(const ir::CallPtr& op) override
    {
        AccessArgIndices indices = ResolveAccessArgIndices(op->name_);
        if (indices.tensor_arg_idx >= 0 && static_cast<int>(op->args_.size()) > indices.tensor_arg_idx) {
            auto tensor_var = std::dynamic_pointer_cast<const ir::Var>(op->args_[indices.tensor_arg_idx]);
            RecordTensorDef(op, ResolveTensorAlias(tensor_var), indices);
        }

        ir::IRVisitor::VisitExpr_(op);
    }

private:
    std::shared_ptr<const ir::Var> ResolveTensorAlias(std::shared_ptr<const ir::Var> var) const
    {
        while (var) {
            auto it = tensor_aliases_.find(var.get());
            if (it == tensor_aliases_.end()) {
                break;
            }
            var = it->second;
        }
        return var;
    }

    struct AccessArgIndices {
        int tensor_arg_idx = -1;
        int tile_arg_idx = -1;
    };

    AccessArgIndices ResolveAccessArgIndices(const std::string& op_name) const
    {
        AccessArgIndices indices;
        if (op_name == "block.load") {
            indices.tensor_arg_idx = 1;
            indices.tile_arg_idx = 0;
        } else if (op_name == "block.store") {
            indices.tensor_arg_idx = 0;
            indices.tile_arg_idx = 1;
        } else if (op_name == "block.store_fp") {
            indices.tensor_arg_idx = 0;
            indices.tile_arg_idx = 1;
        } else if (op_name == "debug.dump_tile") {
            indices.tensor_arg_idx = 3;
            indices.tile_arg_idx = 0;
        }
        return indices;
    }

    // Access window shape: explicit shapes tuple if present, otherwise the tile's shape.
    std::optional<std::vector<ir::ExprPtr>> ResolveAccessShape(const ir::CallPtr& op,
                                                               const AccessArgIndices& indices) const
    {
        if (indices.tile_arg_idx >= 0 && indices.tile_arg_idx < static_cast<int>(op->args_.size())) {
            auto tile_type = std::dynamic_pointer_cast<const ir::TileType>(op->args_[indices.tile_arg_idx]->GetType());
            if (tile_type) {
                return tile_type->shape_;
            }
        }
        return std::nullopt;
    }

    void RecordTensorDef(const ir::CallPtr& op, const std::shared_ptr<const ir::Var>& tensor_var,
                         const AccessArgIndices& indices)
    {
        if (!tensor_var) {
            return;
        }
        auto var = std::const_pointer_cast<ir::Var>(tensor_var);
        TensorDef& def = defs_[ctx_.SanitizeName(var)];
        if (def.var == nullptr) {
            def.var = var;
        }
        if (def.access_shape.empty()) {
            if (auto shape = ResolveAccessShape(op, indices)) {
                def.access_shape = *shape;
            }
        }
        if (!def.tile_dims.has_value() && op->HasKwarg("tile_dims")) {
            def.tile_dims = op->GetKwarg<std::vector<int>>("tile_dims");
        }
        if (op->name_ == "block.load" && op->HasKwarg("is_transpose") && op->GetKwarg<bool>("is_transpose")) {
            def.is_transpose = true;
        }
    }

    const CodeContext& ctx_; ///< for SanitizeName (pure: derives the cce var-name key)
    std::map<const ir::Var*, std::shared_ptr<const ir::Var>> tensor_aliases_;
};

} // namespace

std::vector<std::pair<ir::VarPtr, ir::TileTypePtr>> CCECodegen::CollectTileVariables(const ir::StmtPtr& stmt)
{
    if (!stmt) {
        return {};
    }

    TileCollector collector;
    collector.VisitStmt(stmt);
    return collector.tile_vars_;
}

namespace {
// Collect names of Var nodes that appear in any read position ->Call args,
// Yield values, AssignStmt RHS, return values, expressions inside subscripts,
// for-loop ranges, etc. A phi return_var whose name is not in this set has
// no consumer and can be safely dropped from IfStmt codegen.
class VarReadCollector : public ir::IRVisitor {
public:
    using ir::IRVisitor::VisitExpr_;
    using ir::IRVisitor::VisitStmt_;

    std::set<std::string> var_read_names_;
    std::unordered_map<std::string, int> var_read_counts_;

    // Any time we visit a Var as a sub-expression (via the default Expr walk),
    // it is in a read position ->the write site is an AssignStmt.var_ which
    // the default Stmt walker does not route through VisitExpr.
    void VisitExpr_(const ir::VarPtr& op) override
    {
        if (op) {
            var_read_names_.insert(op->name_);
            var_read_counts_[op->name_]++;
        }
    }

    void VisitStmt_(const ir::AssignStmtPtr& op) override
    {
        // Skip op->var_ (LHS is a definition, not a use). Walk the RHS.
        if (op && op->value_)
            VisitExpr(op->value_);
    }

    void VisitStmt_(const ir::IfStmtPtr& op) override
    {
        // Skip op->return_vars_ ->they are phi outputs (definitions), not reads.
        // Walk condition + branches explicitly.
        if (!op)
            return;
        if (op->condition_)
            VisitExpr(op->condition_);
        if (op->thenBody_)
            VisitStmt(op->thenBody_);
        if (op->elseBody_.has_value() && *op->elseBody_)
            VisitStmt(*op->elseBody_);
    }

    void VisitStmt_(const ir::ForStmtPtr& op) override
    {
        // Similar concern for For loops: return_vars_ (output iter args) are
        // written, not read, on each iteration's phi merge.
        if (!op)
            return;
        if (op->start_)
            VisitExpr(op->start_);
        if (op->stop_)
            VisitExpr(op->stop_);
        if (op->step_)
            VisitExpr(op->step_);
        for (const auto& ia : op->iterArgs_) {
            if (ia && ia->initValue_)
                VisitExpr(ia->initValue_);
        }
        if (op->body_)
            VisitStmt(op->body_);
    }
};

} // namespace

void CCECodegen::CollectVarReadNames(const ir::StmtPtr& stmt, std::set<std::string>& out) const
{
    if (!stmt)
        return;
    VarReadCollector collector;
    collector.VisitStmt(stmt);
    out = std::move(collector.var_read_names_);
    // Also populate read counts on the mutable codegen instance
    const_cast<CCECodegen*>(this)->var_read_counts_ = std::move(collector.var_read_counts_);
}

namespace {

class MutexPipeCollector : public ir::IRVisitor {
public:
    using ir::IRVisitor::VisitExpr_;
    using ir::IRVisitor::VisitStmt_;

    std::map<int, std::set<ir::PipeType>> mutex_pipes;

    void VisitExpr_(const ir::CallPtr& op) override
    {
        if (op) {
            CollectMutexInfo(op);
        }
        ir::IRVisitor::VisitExpr_(op);
    }

    void CollectMutexInfo(const ir::CallPtr& op)
    {
        const std::string& name = op->name_;
        bool is_mutex = (name == "system.mutex_lock" || name == "system.mutex_unlock" ||
                         name == "system.mutex_lock_dyn" || name == "system.mutex_unlock_dyn");
        if (!is_mutex) {
            return;
        }
        ir::PipeType pipe = ir::PipeType::S;
        int static_bid = -1;
        std::vector<int> dyn_bids;
        for (const auto& [key, value] : op->kwargs_) {
            if (key == "pipe")
                pipe = static_cast<ir::PipeType>(std::any_cast<int>(value));
            if (key == "mutex_id")
                static_bid = std::any_cast<int>(value);
            if (key == "mutex_ids")
                dyn_bids = std::any_cast<std::vector<int>>(value);
        }
        auto record = [&](int bid) { mutex_pipes[bid].insert(pipe); };
        if (static_bid >= 0)
            record(static_bid);
        for (int bid : dyn_bids)
            record(bid);
    }
};

} // namespace

void CCECodegen::CollectMutexPipeInfo(const ir::StmtPtr& stmt)
{
    if (!stmt)
        return;
    MutexPipeCollector collector;
    collector.VisitStmt(stmt);
    mutex_pipes_ = std::move(collector.mutex_pipes);
}

bool CCECodegen::ShouldSkipVPipeMutex(ir::PipeType pipe, const std::vector<int>& buf_ids) const
{
    if (pipe != ir::PipeType::V || arch_ != "a5")
        return false;
    for (int bid : buf_ids) {
        auto it = mutex_pipes_.find(bid);
        if (it == mutex_pipes_.end())
            continue;
        for (auto p : it->second) {
            if (p != ir::PipeType::V)
                return false;
        }
    }
    return true;
}

void CCECodegen::CollectTensorDefs(const ir::FunctionPtr& func)
{
    if (!func || !func->body_)
        return;

    TensorDefCollector collector(context_);
    collector.VisitStmt(func->body_);
    tensor_defs_ = std::move(collector.defs_);
}

std::vector<int64_t> CCECodegen::ExtractShapeDimensions(const std::vector<ir::ExprPtr>& shape_exprs) const
{
    std::vector<int64_t> dims;
    dims.reserve(shape_exprs.size());
    for (const auto& expr : shape_exprs) {
        dims.push_back(ExtractConstInt(expr));
    }
    return dims;
}

std::string CCECodegen::FormatAddressHex(int64_t addr)
{
    std::ostringstream oss;
    oss << "0x" << std::hex << addr;
    return oss.str();
}

namespace {

// Map an IR field type to a best-effort C++ type string for struct member declaration.
// Scalar types resolve through DataType::ToCTypeString; an uncovered code is an upstream
// validation gap and fails loudly instead of emitting a mismatched struct definition.
std::string CppTypeForField(const ir::TypePtr& t)
{
    if (auto scalar = ir::As<ir::ScalarType>(t)) {
        std::string ctype = scalar->dtype_.ToCTypeString();
        CHECK(ctype != "unknown") << "Struct field uses unsupported scalar dtype: " << scalar->dtype_.ToString();
        return ctype;
    }
    if (ir::As<ir::TileType>(t)) {
        // Tile fields are stored as opaque tile references; v1 emits as auto-deduced.
        return "auto";
    }
    // Nested named tuple ->the field stores another C++ struct; we have to look up its
    // type name. For v1 we fall back to int64_t and rely on CCE flat expansion if the
    // nested tuple type isn't already materialized; full nested support is a follow-up.
    return "int64_t";
}

// Build a C++ `struct Name { <type> <field>; ... };` definition from a struct type name,
// its field names, and per-field IR types. A TupleType field (Array[T, N]) becomes an
// array member `<T> name[N];`.
std::string BuildStructTypeDef(const std::string& name, const std::vector<std::string>& fields,
                               const std::vector<ir::TypePtr>& types, bool requires_volatile)
{
    std::string def = "class " + name + " {\npublic:\n    ";
    const std::string qualifier = requires_volatile ? "volatile " : "";
    for (size_t i = 0; i < fields.size(); ++i) {
        const auto& ft = types[i];
        if (auto arr = ir::As<ir::TupleType>(ft)) {
            def += qualifier + CppTypeForField(arr->types_[0]) + " " + fields[i] + "[" +
                   std::to_string(arr->types_.size()) + "]; ";
        } else {
            def += qualifier + CppTypeForField(ft) + " " + fields[i] + "; ";
        }
    }
    def += "\n};";
    return def;
}

std::string StructFieldTypeSignature(const ir::TypePtr& type)
{
    if (auto array_type = ir::As<ir::TupleType>(type)) {
        CHECK(!array_type->types_.empty()) << "Struct array field type must not be empty";
        return CppTypeForField(array_type->types_[0]) + "[" + std::to_string(array_type->types_.size()) + "]";
    }
    return CppTypeForField(type);
}

} // namespace

std::string CCECodegen::GetStructFieldTypeString(const ir::TypePtr& type) const { return CppTypeForField(type); }

void CCECodegen::RegisterStructDefinition(const ir::TupleTypePtr& tuple_type, const std::string& type_name,
                                          const std::vector<std::string>& fields, bool is_tiling)
{
    CHECK(tuple_type != nullptr) << "Cannot register a null TupleType for struct '" << type_name << "'";
    CHECK(fields.size() == tuple_type->types_.size()) << "Struct field count mismatch for type '" << type_name << "'";

    auto [it, inserted] = struct_definitions_.try_emplace(
        type_name, StructDefinition{type_name, fields, tuple_type->types_, is_tiling, false});
    if (!inserted) {
        const auto& existing = it->second;
        CHECK(existing.is_tiling == is_tiling)
            << "Struct type '" << type_name << "' cannot be both tiling and non-tiling";
        CHECK(existing.fields == fields && existing.types.size() == tuple_type->types_.size())
            << "Conflicting definitions for struct type '" << type_name << "'";
        for (size_t i = 0; i < existing.types.size(); ++i) {
            CHECK(StructFieldTypeSignature(existing.types[i]) == StructFieldTypeSignature(tuple_type->types_[i]))
                << "Conflicting type for field '" << fields[i] << "' in struct '" << type_name << "'";
        }
    }
}

const std::string* CCECodegen::GetStructName(const ir::TupleTypePtr& tuple_type) const
{
    return debug_info_ != nullptr ? debug_info_->GetTupleName(tuple_type.get()) : nullptr;
}

void CCECodegen::MarkStructVolatile(const ir::TypePtr& type)
{
    auto tuple_type = ir::As<ir::TupleType>(type);
    CHECK(tuple_type != nullptr) << "ssbuf_load/store requires a struct TupleType argument";
    const std::string* struct_name = GetStructName(tuple_type);
    CHECK(struct_name != nullptr) << "ssbuf_load/store struct type is not registered for CCE codegen";
    struct_definitions_.at(*struct_name).requires_volatile = true;
}

void CCECodegen::RegisterTilingStructTypes(const ir::FunctionPtr& func)
{
    for (const auto& param : func->params_) {
        auto tuple_type = ir::As<ir::TupleType>(param->GetType());
        if (tuple_type == nullptr)
            continue;
        const std::vector<std::string>* fields = debug_info_ != nullptr ?
                                                     debug_info_->GetTupleFields(tuple_type.get()) :
                                                     nullptr;
        CHECK(fields != nullptr) << "Tiling struct param '" << param->name_
                                 << "' has no field names registered in IRDebugInfo";
        CHECK(fields->size() == tuple_type->types_.size())
            << "Tiling struct field count mismatch for param '" << param->name_ << "'";
        const std::string* registered_name = debug_info_->GetTupleName(tuple_type.get());
        const std::string struct_name = registered_name != nullptr ? *registered_name : "Tiling";
        RegisterStructDefinition(tuple_type, struct_name, *fields, true);
    }
}

void CCECodegen::EmitStructTypes()
{
    for (const auto& [type_name, info] : struct_definitions_) {
        std::string def = BuildStructTypeDef(type_name, info.fields, info.types, info.requires_volatile);
        if (!info.is_tiling) {
            emitter_.EmitLine(def);
            continue;
        }
        const std::string header_name = type_name + "_tiling.h";
        tiling_headers_[header_name] = "#pragma once\n" + def + "\n";
        emitter_.EmitLine("#include \"" + header_name + "\"");
    }
}

void CCECodegen::EmitTilingStructCopy(const ir::FunctionPtr& func)
{
    for (const auto& param : func->params_) {
        auto tuple_type = ir::As<ir::TupleType>(param->GetType());
        if (tuple_type == nullptr)
            continue;
        std::string name = context_.GetVarName(param);
        const std::string* registered_name = debug_info_->GetTupleName(tuple_type.get());
        const std::string struct_name = registered_name != nullptr ? *registered_name : "Tiling";
        emitter_.EmitLine("constexpr uint32_t " + name + "_all_bytes = sizeof(" + struct_name + ");");
        emitter_.EmitLine(struct_name + " " + name + ";");
        auto emit_cube_copy = [&]() {
            emitter_.EmitLine("copy_data_align64((uint8_t*)&" + name + ", (__gm__ uint8_t *)" + name + "_ptr, " + name +
                              "_all_bytes);");
        };
        auto emit_vector_copy = [&]() {
            emitter_.EmitLine("__ubuf__ uint8_t *" + name + "_in_ub = (__ubuf__ uint8_t *)get_imm(0);");
            emitter_.EmitLine("constexpr uint32_t " + name + "_len_burst = (" + name + "_all_bytes + 31) / 32;");
            emitter_.EmitLine("copy_gm_to_ubuf_align_v2((__ubuf__ uint8_t *)" + name + "_in_ub, (__gm__ uint8_t *)" +
                              name + "_ptr, 0, 1, " + name + "_len_burst * 32, 0, 0, false, 0, 0, 0);");
            emitter_.EmitLine("set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);");
            emitter_.EmitLine("wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);");
            emitter_.EmitLine("copy_data_align64((uint8_t*)&" + name + ", (__ubuf__ uint8_t *)" + name + "_in_ub, " +
                              name + "_all_bytes);");
        };
        if (target_ == ir::SectionKind::Cube) {
            emit_cube_copy();
        } else {
            emit_vector_copy();
        }
        emitter_.EmitLine("");
    }
}

void CCECodegen::GenerateTileTypeDeclaration(const std::string& var_name, const ir::TileTypePtr& tile_type)
{
    INTERNAL_CHECK(!var_name.empty()) << "Internal error: var_name cannot be empty";
    INTERNAL_CHECK(tile_type != nullptr) << "Internal error: tile_type is null";

    // Extract tile shape dimensions
    std::vector<int64_t> shape_dims = ExtractShapeDimensions(tile_type->shape_);

    // CCE codegen only supports 1D and 2D tiles
    CHECK(shape_dims.size() <= 2) << "CCE codegen only supports 1D and 2D TileType, but got " << shape_dims.size()
                                  << " dimensions. Multi-dimensional tiles (>2D) are supported at IR level "
                                  << "but not yet in code generation.";

    // Determine tile dimensions (default to 1 if not specified)
    int64_t rows = shape_dims.size() >= 1 ? shape_dims[0] : 1;
    int64_t cols = shape_dims.size() >= 2 ? shape_dims[1] : 1;

    // Extract valid_shape: compute runtime ctor args.
    auto vs = ExtractValidShapeInfo(tile_type, rows, cols, [this](const ir::VarPtr& v) { return GetVarName(v); });
    std::string ctor_args = BuildTileCtorArgs(vs, rows, cols);

    // Generate Tile type alias (with dedup: reuse alias if same type string already emitted)
    std::string tile_type_str = GetGeneratedType(tile_type);
    std::string type_alias_name;
    auto dedup_it = emitted_tile_types_.find(tile_type_str);
    if (dedup_it != emitted_tile_types_.end()) {
        type_alias_name = dedup_it->second;
    } else {
        // Derive alias name from var_name base (strip trailing _N suffix for dedup readability)
        auto last_underscore = var_name.rfind('_');
        bool has_index_suffix = (last_underscore != std::string::npos && last_underscore + 1 < var_name.size() &&
                                 std::isdigit(var_name[last_underscore + 1]));
        std::string base_name = has_index_suffix ? var_name.substr(0, last_underscore) : var_name;
        type_alias_name = base_name + "_Type";
        // Ensure uniqueness against existing aliases
        if (emitted_tile_types_.count(type_alias_name + "_used") > 0 &&
            emitted_tile_types_[type_alias_name + "_used"] != tile_type_str) {
            type_alias_name = var_name + "Type";
        }
        emitted_tile_types_[tile_type_str] = type_alias_name;
        emitted_tile_types_[type_alias_name + "_used"] = tile_type_str;
        emitter_.EmitLine("using " + type_alias_name + " = " + tile_type_str + ";");
    }

    // Generate Tile instance + TASSIGN on one line (compact)
    // Only pass ctor args when template has -1 params (dynamic valid_shape).
    std::string ctor_suffix = vs.needs_ctor ? ("(" + ctor_args + ")") : "";
    if (tile_type->memref_.has_value()) {
        int64_t addr = ExtractConstInt((*tile_type->memref_)->addr_); // NOLINT(bugprone-unchecked-optional-access)
        std::string addr_str = FormatAddressHex(addr);
        emitter_.EmitLine(type_alias_name + " " + var_name + ctor_suffix + "; TASSIGN(" + var_name + ", " + addr_str +
                          ");");
        tile_addresses_[var_name] = addr_str;
    } else {
        emitter_.EmitLine(type_alias_name + " " + var_name + ctor_suffix + ";");
    }
}

// ========================================================================
// Phase 7 helpers: Stride type generation and GlobalTensor instance emission
// ========================================================================

std::string CCECodegen::GenerateSingleFileStrideType(const std::vector<int64_t>& dims) const
{
    // ND (row-major) static strides: stride[i] = product(dims[i+1..n-1]), padded with
    // leading 1s to five slots. Used for the NZ physical fractal shape (5D).
    std::ostringstream oss;
    oss << "pto::Stride<";
    const size_t target_dims = 5;
    size_t n = dims.size();
    for (size_t i = 0; i < target_dims - n; ++i) {
        oss << "1, ";
    }
    for (size_t i = 0; i < n; ++i) {
        int64_t stride = 1;
        for (size_t j = i + 1; j < n; ++j) {
            stride *= dims[j];
        }
        oss << stride;
        if (i < n - 1)
            oss << ", ";
    }
    oss << ">";
    return oss.str();
}

std::vector<std::string> CCECodegen::BuildTensorStrideExpressions(const ir::TensorTypePtr& tensor_type)
{
    CHECK(tensor_type != nullptr) << "Cannot build stride expressions for a null tensor type";
    const size_t ndim = tensor_type->shape_.size();

    if (tensor_type->tensor_view_.has_value() && !tensor_type->tensor_view_->stride.empty()) {
        const auto& view_stride = tensor_type->tensor_view_->stride;
        CHECK(view_stride.size() == ndim)
            << "TensorView stride rank (" << view_stride.size() << ") != tensor rank (" << ndim << ")";

        std::vector<std::string> stride_exprs;
        stride_exprs.reserve(ndim);
        for (const auto& stride : view_stride) {
            stride_exprs.push_back(GetExprAsCode(stride));
        }
        return stride_exprs;
    }

    std::vector<std::string> stride_exprs(ndim, "1");
    for (size_t i = 0; i < ndim; ++i) {
        std::ostringstream stride;
        for (size_t j = i + 1; j < ndim; ++j) {
            if (j > i + 1) {
                stride << " * ";
            }
            stride << GetExprAsCode(tensor_type->shape_[j]);
        }
        if (i + 1 < ndim) {
            stride_exprs[i] = stride.str();
        }
    }
    return stride_exprs;
}

void CCECodegen::AppendDynamicStrideGlobalTensorArgs(
    std::ostringstream& global_instance, const std::string& shape_type_name, const std::string& stride_type_name,
    const ir::TensorTypePtr& tensor_type, const std::optional<std::vector<int>>& tile_dims, bool is_transpose)
{
    const auto stride_exprs = BuildTensorStrideExpressions(tensor_type);
    const size_t ndim = tensor_type->shape_.size();
    std::string row_stride_expr;
    std::string col_stride_expr;
    if (tile_dims.has_value() && tile_dims->size() == 2) {
        row_stride_expr = stride_exprs[static_cast<size_t>((*tile_dims)[0])];
        col_stride_expr = stride_exprs[static_cast<size_t>((*tile_dims)[1])];
    } else if (ndim >= 2) {
        row_stride_expr = stride_exprs[ndim - 2];
        col_stride_expr = stride_exprs[ndim - 1];
    } else {
        row_stride_expr = stride_exprs[0];
        col_stride_expr = "1";
    }
    global_instance << ", " << shape_type_name << "(), " << stride_type_name << "(1, 1, 1, ";
    if (is_transpose) {
        global_instance << col_stride_expr << ", " << row_stride_expr;
    } else {
        global_instance << row_stride_expr << ", " << col_stride_expr;
    }
    global_instance << ")";
}

void CCECodegen::EmitGlobalTensorInstance(const std::string& var_name, const std::string& global_type_name,
                                          const std::string& shape_type_name, const std::string& stride_type_name,
                                          const ir::TensorTypePtr& tensor_type,
                                          const std::optional<std::vector<int>>& tile_dims, bool is_transpose,
                                          const std::string& base_pointer)
{
    std::ostringstream global_instance;
    global_instance << global_type_name << " " << var_name << "(" << base_pointer;
    AppendDynamicStrideGlobalTensorArgs(global_instance, shape_type_name, stride_type_name, tensor_type, tile_dims,
                                        is_transpose);
    global_instance << ");";
    emitter_.EmitLine(global_instance.str());
}

std::string CCECodegen::BuildDynamicNZTensorDimArg(const ir::TensorTypePtr& tensor_type, size_t axis)
{
    if (auto const_dim = ir::As<ir::ConstInt>(tensor_type->shape_[axis])) {
        return std::to_string(const_dim->value_);
    }
    return GetExprAsCode(tensor_type->shape_[axis]);
}

std::string CCECodegen::BuildDynamicNZShapeArg(const ir::TensorTypePtr& tensor_type, size_t axis,
                                               const std::optional<std::vector<ir::ExprPtr>>& access_shape)
{
    if (!access_shape.has_value() || axis >= access_shape->size()) {
        return BuildDynamicNZTensorDimArg(tensor_type, axis);
    }

    const auto& expr = access_shape->at(axis);
    if (auto const_dim = ir::As<ir::ConstInt>(expr)) {
        return std::to_string(const_dim->value_);
    }

    auto var_dim = ir::As<ir::Var>(expr);
    auto tensor_dim_var = ir::As<ir::Var>(tensor_type->shape_[axis]);
    if (var_dim && tensor_dim_var && var_dim->name_ == tensor_dim_var->name_) {
        return BuildDynamicNZTensorDimArg(tensor_type, axis);
    }
    return GetExprAsCode(expr);
}

void CCECodegen::EmitDynamicNZGlobalTensorDeclaration(
    const std::string& var_name, const ir::TensorTypePtr& tensor_type, const std::optional<std::string>& base_pointer,
    const std::optional<std::vector<ir::ExprPtr>>& access_shape, const std::string& element_type,
    const std::string& shape_type_name, const std::string& stride_type_name, const std::string& global_type_name)
{
    if (tensor_type->shape_.size() != 2) {
        throw pypto::ir::ValueError("CCE NZ tensor lowering currently requires a 2D tensor");
    }
    (void)backend::cce::GetNZInnerCols(tensor_type->dtype_);

    const std::string row_arg = BuildDynamicNZShapeArg(tensor_type, 0, access_shape);
    const std::string col_arg = BuildDynamicNZShapeArg(tensor_type, 1, access_shape);
    const std::string full_row_arg = BuildDynamicNZTensorDimArg(tensor_type, 0);
    const std::string full_col_arg = BuildDynamicNZTensorDimArg(tensor_type, 1);

    emitter_.EmitLine("using " + shape_type_name + " = pto::TileShape2D<" + element_type +
                      ", pto::DYNAMIC, pto::DYNAMIC, Layout::NZ>;");
    emitter_.EmitLine("using " + stride_type_name + " = pto::BaseShape2D<" + element_type +
                      ", pto::DYNAMIC, pto::DYNAMIC, Layout::NZ>;");
    emitter_.EmitLine("using " + global_type_name + " = GlobalTensor<" + element_type + ", " + shape_type_name + ", " +
                      stride_type_name + ", Layout::NZ>;");

    std::ostringstream nz_instance;
    nz_instance << global_type_name << " " << var_name << "(";
    if (base_pointer.has_value()) {
        nz_instance << base_pointer.value() << ", " << shape_type_name << "(" << row_arg << ", " << col_arg << "), "
                    << stride_type_name << "(" << full_row_arg << ", " << full_col_arg << ")";
    }
    nz_instance << ");";
    emitter_.EmitLine(nz_instance.str());
}

std::string CCECodegen::BuildGlobalTensorLayoutArg(const std::string& stride_type_name,
                                                   const std::vector<int64_t>& shape_dims, bool is_transpose) const
{
    if (*shape_dims.rbegin() == 1 || is_transpose) {
        return stride_type_name + ", Layout::DN";
    }
    return stride_type_name;
}

void CCECodegen::EmitNZGlobalTensorDeclaration(const TensorDef& def, const std::string& var_name,
                                               const ir::TensorTypePtr& tensor_type)
{
    std::optional<std::string> base_pointer;
    if (HasPointer(var_name)) {
        base_pointer = GetPointer(var_name);
    }
    std::optional<std::vector<ir::ExprPtr>> access_shape = def.access_shape.empty() ?
                                                               std::optional<std::vector<ir::ExprPtr>>() :
                                                               std::optional<std::vector<ir::ExprPtr>>(
                                                                   def.access_shape);

    std::string element_type = tensor_type->dtype_.ToCTypeString();
    std::string shape_type_name = var_name + "ShapeDim5";
    std::string stride_type_name = var_name + "StrideDim5";
    std::string global_type_name = var_name + "Type";

    bool all_static = true;
    for (const auto& dim : tensor_type->shape_) {
        if (!std::dynamic_pointer_cast<const ir::ConstInt>(dim)) {
            all_static = false;
            break;
        }
    }

    // Dynamic NZ tensors use special TileShape2D/BaseShape2D types and a runtime ctor.
    if (!all_static) {
        EmitDynamicNZGlobalTensorDeclaration(var_name, tensor_type, base_pointer, access_shape, element_type,
                                             shape_type_name, stride_type_name, global_type_name);
        return;
    }

    // Static NZ: physical fractal shape {1, cols/c0, rows/16, 16, c0} with fully static strides.
    std::vector<int64_t> tensor_dims = ExtractShapeDimensions(tensor_type->shape_);
    std::vector<int64_t> emitted_shape_dims = BuildNZPhysicalShapeDims(tensor_type, tensor_dims);

    emitter_.EmitLine("using " + shape_type_name + " = " + type_converter_.GenerateShapeType(emitted_shape_dims) + ";");
    emitter_.EmitLine("using " + stride_type_name + " = " + GenerateSingleFileStrideType(emitted_shape_dims) + ";");
    emitter_.EmitLine("using " + global_type_name + " = GlobalTensor<" + element_type + ", " + shape_type_name + ", " +
                      stride_type_name + ", Layout::NZ>;");

    std::ostringstream nz_instance;
    nz_instance << global_type_name << " " << var_name << "(";
    if (base_pointer.has_value()) {
        nz_instance << base_pointer.value();
    }
    nz_instance << ");";
    emitter_.EmitLine(nz_instance.str());
}

void CCECodegen::GenerateGlobalTensorTypeDeclaration(const TensorDef& def)
{
    INTERNAL_CHECK(def.var != nullptr) << "Internal error: TensorDef.var is null";
    std::string var_name = context_.SanitizeName(def.var);
    auto tensor_type = ir::As<ir::TensorType>(def.var->GetType());
    INTERNAL_CHECK(tensor_type != nullptr) << "Internal error: TensorDef.var is not a tensor";

    // NZ tensors follow a distinct fractal layout; handle them entirely in one place.
    if (IsNZTensorType(tensor_type)) {
        EmitNZGlobalTensorDeclaration(def, var_name, tensor_type);
        return;
    }

    // Non-NZ invariants: every accessed tensor has a tile-derived access_shape and a base
    // pointer registered before this point (params at signature build, ptr.make_tensor
    // views at their op).
    INTERNAL_CHECK(HasPointer(var_name)) << "Internal error: tensor '" << var_name << "' has no base pointer";
    INTERNAL_CHECK(!def.access_shape.empty()) << "Internal error: tensor '" << var_name << "' has no access_shape";

    const bool is_transpose = def.is_transpose;
    const std::optional<std::vector<int>>& tile_dims = def.tile_dims;
    const std::string base_pointer = GetPointer(var_name);

    // shape_dims: tile/access shape, used for the last-dim==1 DN detection and as the
    // row-stride fallback. Must contain integer values (no -1).
    std::vector<int64_t> shape_dims = ExtractShapeDimensions(def.access_shape);
    std::string element_type = tensor_type->dtype_.ToCTypeString();

    std::string shape_type_name = var_name + "ShapeDim5";
    std::string stride_type_name = var_name + "StrideDim5";
    std::string global_type_name = var_name + "Type";

    // Shape and stride are always dynamic: Shape<1,1,1,-1,-1> + Stride<-1,-1,-1,-1,-1>.
    // DIM_3/DIM_4 are configured per access via SetShape; the stride values are supplied
    // through the instance ctor below.
    emitter_.EmitLine("using " + shape_type_name + " = " + type_converter_.GenerateShapeType({-1, -1}) + ";");
    emitter_.EmitLine("using " + stride_type_name + " = pto::Stride<-1, -1, -1, -1, -1>;");

    // Layout: DN if last dim is 1 or if the def is flagged is_transpose.
    std::string global_layout_arg = BuildGlobalTensorLayoutArg(stride_type_name, shape_dims, is_transpose);
    emitter_.EmitLine("using " + global_type_name + " = GlobalTensor<" + element_type + ", " + shape_type_name + ", " +
                      global_layout_arg + ">;");

    EmitGlobalTensorInstance(var_name, global_type_name, shape_type_name, stride_type_name, tensor_type, tile_dims,
                             is_transpose, base_pointer);
}

} // namespace codegen

} // namespace pypto
