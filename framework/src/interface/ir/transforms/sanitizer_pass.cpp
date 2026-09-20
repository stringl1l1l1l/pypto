/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <utility>

#include "backend/common/backend_utils.h"
#include "ir/kind_traits.h"
#include "ir/scalar_expr.h"
#include "ir/scalar_expr_ops.h"
#include "ir/type.h"
#include "ir/transforms/base/mutator.h"
#include "ir/transforms/base/visitor.h"
#include "ir/transforms/passes.h"
#include "pypto_pro/error.h"

namespace pypto {
namespace ir {

namespace {

// det_id values on the wire; each owns a distinct record format. Must stay
// in sync with the Python DET_* constants in sanitizer_replay.py.
enum class SanitizerDetection : uint32_t {
    GmAccess = 1,    // GM access: tensor_id, ndim, off[5], acc_row, acc_col
    TileAccess = 2,  // tile access: tile_id, dim0, dim1, off_row, off_col, acc_row, acc_col
    MutexAccess = 3, // mutex lock/unlock: mutex_id, pipe, is_lock, span_id
    ViewShape = 4,   // runtime view shape: tensor_id, dim[5] (exprs), footprint (expr), span
    TileScalar = 5,  // tile scalar access: dims, linear offset, span
    TileDecl = 6,    // make_tile declaration: addr, size, space_id, span (overlap scan)
};

/// Max logical tensor rank supported by the GM bounds check.
constexpr uint32_t kSanitizerMaxTensorDims = 5;

// Runtime-evaluated GetValidRow()/GetValidCol() of the tile.
ExprPtr TileDim(const ExprPtr& tile, int axis)
{
    return std::make_shared<Call>("block.tile_valid_shape", std::vector<ExprPtr>{tile},
                                  std::vector<std::pair<std::string, std::any>>{{"axis", axis}},
                                  std::make_shared<ScalarType>(DataType(DataType::UINT32)), Span::Unknown());
}

ExprPtr Int64Const(int64_t v) { return std::make_shared<ConstInt>(v, DataType::INT64, Span::Unknown()); }

// ---------------------------------------------------------------------------
// SanitizerInstrumenter injects the log records (an IRMutator).
//
// Two record handlers: RecordBounds (every memory-access op -- GM
// loads/stores, tile transfers at an offset, scalar access, set_validshape
// windows) and RecordMutex (lock/unlock pairing); the op name picks the
// operand layout inside the handler. Records carry values only -- the pass
// never decides, the replay computes the detections. Declarations (make_tile,
// ptr.make_tensor) get their own declaration records at the AssignStmt hook.
// ---------------------------------------------------------------------------

class SanitizerInstrumenter : public IRMutator {
public:
    using IRMutator::VisitExpr_;
    using IRMutator::VisitStmt_;

    SanitizerInstrumenter()
    {
        // Table entries name the check the record serves, not the op spell.
        // Same-entry ops differ only in how they collect the operands (that
        // part lives inside each builder).
        auto tensor_bounds = [this](const CallPtr& c) { return RecordGmAccess(c); };
        auto tile_transfer_bounds = [this](const CallPtr& c) {
            // move/move_fp (TEXTRACT): the offset is the SOURCE extraction
            // start, the window the DESTINATION's valid shape.
            ExprPtr src_off_row = Int64Const(0);
            ExprPtr src_off_col = Int64Const(0);
            if (auto off_tuple = As<MakeTuple>(c->args_.back())) {
                if (!off_tuple->elements_.empty())
                    src_off_row = off_tuple->elements_[0];
                if (off_tuple->elements_.size() > 1)
                    src_off_col = off_tuple->elements_[1];
            }
            ExprPtr win_row = TileDim(c->args_[0], 0);
            ExprPtr win_col = TileDim(c->args_[0], 1);
            auto src = RecordTileAccess(c, 1, src_off_row, src_off_col, win_row, win_col);
            auto dst = RecordTileAccess(c, 0, Int64Const(0), Int64Const(0), win_row, win_col);
            return SeqStmts::Flatten({src, dst}, c->span_);
        };
        auto tile_insert_bounds = [this](const CallPtr& c) {
            // insert (TINSERT): the offset is the DESTINATION write start,
            // the window the SOURCE's valid shape.
            ExprPtr win_row = TileDim(c->args_[1], 0);
            ExprPtr win_col = TileDim(c->args_[1], 1);
            auto src = RecordTileAccess(c, 1, Int64Const(0), Int64Const(0), win_row, win_col);
            auto dst = RecordTileAccess(c, 0, c->args_[2], c->args_[3], win_row, win_col);
            return SeqStmts::Flatten({src, dst}, c->span_);
        };
        auto scalar_bounds = [this](const CallPtr& c) { return RecordScalarAccess(c); };
        auto validshape_bounds = [this](const CallPtr& c) { return RecordValidShape(c); };
        auto mutex_pairing = [this](const CallPtr& c) { return RecordMutex(c); };
        auto tile_overlap_decl = [this](const CallPtr& c) { return CollectMakeTile(c); };
        auto view_source_decl = [this](const CallPtr& c) { return CollectMakeTensor(c); };

        op_records_ = {
            {"block.make_tile", tile_overlap_decl},
            {"ptr.make_tensor", view_source_decl},
            {"block.load", tensor_bounds},
            {"block.store", tensor_bounds},
            {"block.move", tile_transfer_bounds},
            {"block.move_fp", tile_transfer_bounds},
            {"block.insert", tile_insert_bounds},
            {"block.getval", scalar_bounds},
            {"block.setval", scalar_bounds},
            {"block.set_validshape", validshape_bounds},
            {"system.mutex_lock_dyn", mutex_pairing},
            {"system.mutex_unlock_dyn", mutex_pairing},
        };
    }

    ProgramPtr Instrument(ProgramPtr program);

private:
    // -- Layer 1: traversal hooks -------------------------------------------
    StmtPtr VisitStmt_(const EvalStmtPtr& op) override;
    StmtPtr VisitStmt_(const AssignStmtPtr& op) override;
    // ------------------------------------------------------------------------

    // Record builders shared by the table entries: every entry produces its
    // records through these, varying only how it collects the operands.
    // RecordGmAccess: GM loads/stores (per-dim record). RecordTileAccess:
    // one tile record (dims from the access-site type, caller supplies the
    // offset and window expressions). RecordMutex: one record per mutex id.
    StmtPtr RecordGmAccess(const CallPtr& call);
    StmtPtr RecordTileAccess(const CallPtr& call, uint32_t tile_arg_idx, const ExprPtr& off_row, const ExprPtr& off_col,
                             const ExprPtr& win_row, const ExprPtr& win_col);
    StmtPtr RecordMutex(const CallPtr& call);
    StmtPtr RecordScalarAccess(const CallPtr& call);
    StmtPtr RecordValidShape(const CallPtr& call);
    // Op name -> record functions. Same-handler ops (load/store) share one
    // entry; per-op differences live in the collected operands only.
    using RecordFunc = std::function<StmtPtr(const CallPtr&)>;
    std::unordered_map<std::string, RecordFunc> op_records_;
    // Declaration records (table entries like every other op). The
    // make_tile builder reads the assigned variable's type from
    // target_type_ (set by the AssignStmt hook: the call's own args[0]
    // types as the TileType *constructor* expression, not a Tile).
    StmtPtr CollectMakeTile(const CallPtr& call);
    StmtPtr CollectMakeTensor(const CallPtr& call);
    // Table lookup for the op call hooks.
    StmtPtr RecordForCall(const CallPtr& call);
    // Collection helpers shared by the record builders.
    // Declared tile dims (static by frontend contract); nullptr when the
    // expression is not a Tile.
    std::pair<int64_t, int64_t> TileDims(const ExprPtr& tile);
    // Whole-shape element count as an expression product (static and
    // dynamic dims alike travel in the record).
    ExprPtr ShapeProduct(const ShapedTypePtr& type, const Span& span);

    // Unified exit of every record function: assembles the block.sanitizer_log
    // call (det_id + fields + hidden log params + source line).
    StmtPtr MakeLogStmt(uint32_t det_id, std::vector<ExprPtr> fields, const Span& span);

    // Hidden log-buffer parameters, appended to the entry signature; passed
    // as Var references so codegen resolves their final (SSA-suffixed) names.
    ExprPtr log_param_;
    ExprPtr capacity_param_;
    // Type of the current AssignStmt target (set by the hook, read by the
    // make_tile builder; empty while visiting EvalStmts).
    TypePtr target_type_;
};

StmtPtr SanitizerInstrumenter::RecordForCall(const CallPtr& call)
{
    auto it = op_records_.find(call->name_);
    return (it == op_records_.end()) ? nullptr : it->second(call);
}

StmtPtr SanitizerInstrumenter::VisitStmt_(const EvalStmtPtr& op)
{
    auto new_stmt = IRMutator::VisitStmt_(op);
    auto call = As<Call>(op->expr_);
    if (call == nullptr)
        return new_stmt;
    auto record = RecordForCall(call);
    if (record == nullptr)
        return new_stmt;
    return SeqStmts::Flatten({record, new_stmt}, op->span_);
}

// AssignStmt runs the same op dispatch as EvalStmt: value-producing ops
// (getval...) and the declaration builders (make_tile / make_tensor) are
// all table entries; the hook knows nothing about names.
StmtPtr SanitizerInstrumenter::VisitStmt_(const AssignStmtPtr& op)
{
    auto new_stmt = IRMutator::VisitStmt_(op);
    target_type_ = op->var_->GetType();
    auto call = As<Call>(op->value_);
    if (call != nullptr) {
        auto record = RecordForCall(call);
        if (record != nullptr)
            new_stmt = SeqStmts::Flatten({record, new_stmt}, op->span_);
    }
    target_type_ = nullptr;
    return new_stmt;
}

// ---------------------------------------------------------------------------
// Record functions: collect the operands, hand fields to MakeLogStmt
// ---------------------------------------------------------------------------

// Collection helpers shared by the record builders.

std::pair<int64_t, int64_t> SanitizerInstrumenter::TileDims(const ExprPtr& tile)
{
    auto tile_type = As<TileType>(tile->GetType());
    if (tile_type == nullptr)
        return {-1, -1};
    auto dim0 = As<ConstInt>(tile_type->shape_[0]);
    auto dim1 = As<ConstInt>(tile_type->shape_[1]);
    PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, dim0 != nullptr && dim1 != nullptr)
        << "sanitizer: tile shape dims must be static";
    return {dim0->value_, dim1->value_};
}

ExprPtr SanitizerInstrumenter::ShapeProduct(const ShapedTypePtr& type, const Span& span)
{
    ExprPtr numel = Int64Const(1);
    for (const auto& d : type->shape_)
        numel = MakeMul(numel, d, span);
    return numel;
}

// Unified exit of every record function: assembles the block.sanitizer_log
// call (det_id + fields + hidden log params + source line).
StmtPtr SanitizerInstrumenter::MakeLogStmt(uint32_t det_id, std::vector<ExprPtr> fields, const Span& span)
{
    std::vector<ExprPtr> args;
    args.push_back(Int64Const(det_id)); // codegen / decoder dispatch on it
    args.insert(args.end(), std::make_move_iterator(fields.begin()), std::make_move_iterator(fields.end()));
    args.push_back(log_param_);
    args.push_back(capacity_param_);
    // Source line for the replay report; the file comes from the Python
    // side, which knows the kernel's source path.
    args.push_back(Int64Const(static_cast<int64_t>(span.BeginLine())));
    auto call = std::make_shared<Call>("block.sanitizer_log", std::move(args), Span::Unknown());
    return std::make_shared<EvalStmt>(call, Span::Unknown());
}

// getval/setval: args = [container, index]. A Tensor container becomes a
// linear GM record (ndim = 0, acc_row = element count); a Tile container a
// tile-count record.
StmtPtr SanitizerInstrumenter::RecordScalarAccess(const CallPtr& call)
{
    PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, call->args_.size() >= 0x2)
        << "sanitizer: " << call->name_ << " requires (container, offset)";
    const auto& container = call->args_[0];
    auto [tile_dim0, tile_dim1] = TileDims(container);
    if (tile_dim0 >= 0) {
        std::vector<ExprPtr> fields;
        fields.push_back(Int64Const(tile_dim0));
        fields.push_back(Int64Const(tile_dim1));
        fields.push_back(call->args_[1]);
        return MakeLogStmt(static_cast<uint32_t>(SanitizerDetection::TileScalar), std::move(fields), call->span_);
    }
    auto tensor_type = As<TensorType>(container->GetType());
    if (tensor_type == nullptr)
        return nullptr;
    // Linear bound: the whole-tensor element count as an expression product
    // (static and dynamic dims alike travel in the record). Same length as
    // the per-dim GM record (one det owns one format): shape slots zero,
    // acc_row carries the count.
    ExprPtr numel = ShapeProduct(tensor_type, call->span_);
    std::vector<ExprPtr> fields;
    fields.push_back(Int64Const(0)); // ndim = 0: linear-offset access
    fields.push_back(call->args_[1]);
    for (uint32_t i = 0; i < kSanitizerMaxTensorDims - 1; ++i)
        fields.push_back(Int64Const(0));
    for (uint32_t i = 0; i < kSanitizerMaxTensorDims; ++i)
        fields.push_back(Int64Const(0)); // shape slots (length only)
    fields.push_back(numel);
    fields.push_back(Int64Const(1));
    return MakeLogStmt(static_cast<uint32_t>(SanitizerDetection::GmAccess), std::move(fields), call->span_);
}

// set_validshape: window being set vs the declared dims (constant windows
// included -- everything is judged by the replay).
StmtPtr SanitizerInstrumenter::RecordValidShape(const CallPtr& call)
{
    if (call->args_.size() != 0x3)
        return nullptr;
    auto [dim0, dim1] = TileDims(call->args_[0]);
    PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, dim0 >= 0)
        << "sanitizer: set_validshape target must be a Tile";
    std::vector<ExprPtr> fields;
    fields.push_back(Int64Const(dim0));
    fields.push_back(Int64Const(dim1));
    fields.push_back(Int64Const(0));
    fields.push_back(Int64Const(0));
    fields.push_back(call->args_[1]);
    fields.push_back(call->args_[2]);
    return MakeLogStmt(static_cast<uint32_t>(SanitizerDetection::TileAccess), std::move(fields), call->span_);
}

StmtPtr SanitizerInstrumenter::RecordGmAccess(const CallPtr& call)
{
    // Operand positions follow the shared access-op contract; the offsets are
    // always a MakeTuple (frontend normalizes, CCE codegen CHECKs the same).
    auto indices = backend::cce::ResolveAccessArgIndices(call->name_);
    PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR,
                            indices.tensor_arg_idx >= 0 && indices.tile_arg_idx >= 0 && indices.offsets_arg_idx >= 0)
        << "sanitizer: no access-arg contract for op " << call->name_;
    ExprPtr tensor = call->args_[indices.tensor_arg_idx];
    ExprPtr tile = call->args_[indices.tile_arg_idx];
    auto off_tuple = As<MakeTuple>(call->args_[indices.offsets_arg_idx]);
    PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, off_tuple != nullptr)
        << "sanitizer: " << call->name_ << " offsets must be a MakeTuple";

    // Fields: ndim, off[5] (padded), shape[5] (padded), acc_row/acc_col.
    // The tensor shape comes from the access-site TensorType -- static dims
    // are ConstInt, dynamic dims are Var expressions the codegen resolves to
    // the ABI scalars, so the record carries the runtime truth with no host
    // id<->shape mapping (aliases and dynamic dims included).
    auto tensor_type = As<TensorType>(tensor->GetType());
    PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, tensor_type != nullptr)
        << "sanitizer: " << call->name_ << " tensor operand must be a Tensor";
    uint32_t ndim = std::min<uint32_t>(static_cast<uint32_t>(off_tuple->elements_.size()), kSanitizerMaxTensorDims);
    std::vector<ExprPtr> fields;
    fields.push_back(Int64Const(ndim));
    // Offsets (up to 5, pad with 0).
    for (uint32_t i = 0; i < kSanitizerMaxTensorDims; ++i) {
        if (i < off_tuple->elements_.size()) {
            fields.push_back(off_tuple->elements_[i]);
        } else {
            fields.push_back(Int64Const(0));
        }
    }
    // Shape dims aligned with the offsets (same alignment as the replay's
    // trailing-dim windowing; dynamic dims travel as expressions).
    uint32_t rank = std::min<uint32_t>(static_cast<uint32_t>(tensor_type->shape_.size()), kSanitizerMaxTensorDims);
    for (uint32_t i = 0; i < kSanitizerMaxTensorDims; ++i) {
        uint32_t dim = (rank <= ndim) ? i : (ndim - rank) + i; // leading dims dropped
        fields.push_back(dim < rank ? tensor_type->shape_[dim] : Int64Const(0));
    }
    // The order kwarg decomposes into ascending tile_dims + is_transpose;
    // a transposed trailing-axes load swaps the recorded windows. Orders
    // selecting other axes are inexpressible in the record: emit nothing
    // rather than misjudge (known limitation).
    ExprPtr acc_row = TileDim(tile, 0);
    ExprPtr acc_col = TileDim(tile, 1);
    if (call->HasKwarg("tile_dims")) {
        auto tile_dims = call->GetKwarg<std::vector<int>>("tile_dims");
        if (tile_dims.size() == 0x2) {
            auto off_rank = static_cast<int>(off_tuple->elements_.size());
            if (tile_dims[0] == off_rank - 0x2 && tile_dims[1] == off_rank - 1) {
                if (call->GetKwarg<bool>("is_transpose", false)) {
                    std::swap(acc_row, acc_col);
                }
            } else {
                return nullptr;
            }
        }
    }
    fields.push_back(std::move(acc_row));
    fields.push_back(std::move(acc_col));
    return MakeLogStmt(static_cast<uint32_t>(SanitizerDetection::GmAccess), std::move(fields), call->span_);
}

StmtPtr SanitizerInstrumenter::RecordTileAccess(const CallPtr& call, uint32_t tile_arg_idx, const ExprPtr& off_row,
                                                const ExprPtr& off_col, const ExprPtr& win_row, const ExprPtr& win_col)
{
    PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, tile_arg_idx < call->args_.size())
        << "sanitizer: missing tile operand at " << tile_arg_idx;
    // Dims come from the access-site tile type -- what the codegen
    // instantiates the transfer with, i.e. the runtime truth of the access.
    // This keeps the bound check independent of the tile table (overlap-scan
    // only) and lets group cursors record: unresolvable name, known type.
    auto [dim0, dim1] = TileDims(call->args_[tile_arg_idx]);
    PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, dim0 >= 0)
        << "sanitizer: tile operand must be a Tile";
    std::vector<ExprPtr> fields;
    fields.push_back(Int64Const(dim0));
    fields.push_back(Int64Const(dim1));
    fields.push_back(off_row);
    fields.push_back(off_col);
    fields.push_back(win_row);
    fields.push_back(win_col);
    return MakeLogStmt(static_cast<uint32_t>(SanitizerDetection::TileAccess), std::move(fields), call->span_);
}

StmtPtr SanitizerInstrumenter::RecordMutex(const CallPtr& call)
{
    bool is_lock = (call->name_ == "system.mutex_lock_dyn");
    int pipe = call->GetKwarg<int>("pipe", 0);
    // One record per mutex_id argument (a deduped auto_mutex call carries
    // one id per tile group); single-id calls keep the one-statement shape.
    std::vector<StmtPtr> logs;
    for (const auto& mutex_id : call->args_) {
        std::vector<ExprPtr> fields;
        fields.push_back(mutex_id);
        fields.push_back(Int64Const(pipe));
        fields.push_back(Int64Const(is_lock ? 1 : 0));
        logs.push_back(
            MakeLogStmt(static_cast<uint32_t>(SanitizerDetection::MutexAccess), std::move(fields), call->span_));
    }
    if (logs.empty())
        return nullptr;
    return SeqStmts::Flatten(std::move(logs), call->span_);
}

StmtPtr SanitizerInstrumenter::CollectMakeTile(const CallPtr& call)
{
    // make_tile: the assigned variable's type (target_type_, set by the
    // AssignStmt hook) carries the TileType; addr
    // comes from the memref_addr kwarg, size =
    // shape product x dtype bytes from the TileType (the frontend enforces
    // compile-time-constant shapes and a resolved MemRef). Emits a
    // declaration record for the overlap scan; the replay dedups identical
    // re-registrations (the top-level statements are copied into both
    // programs of one kernel) by (addr, size, space, span) -- same-statement
    // copies share the span, distinct same-range tiles keep both records and
    // are exactly what the scan must report.
    auto tile_type = As<TileType>(target_type_);
    PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR,
                            tile_type != nullptr && tile_type->shape_.size() >= 0x2)
        << "sanitizer: make_tile value must be a 2-D TileType";
    PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR,
                            tile_type->memref_.has_value() && *tile_type->memref_ != nullptr)
        << "sanitizer: make_tile requires a resolved MemRef (frontend enforces addr)";
    int64_t dtype_bytes = std::max<int64_t>(1, tile_type->dtype_.GetBit() / 8);
    int64_t sz = 1;
    for (const auto& d : tile_type->shape_) {
        auto c = As<ConstInt>(d);
        PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, c != nullptr)
            << "sanitizer: tile shape dims must be static";
        sz *= c->value_;
    }
    int64_t addr = -1;
    for (const auto& [k, v] : call->kwargs_) {
        if (k == "memref_addr") {
            try {
                addr = std::any_cast<int64_t>(v);
            } catch (...) {
            }
            try {
                addr = static_cast<int64_t>(std::any_cast<int32_t>(v));
            } catch (...) {
            }
        }
    }
    std::vector<ExprPtr> fields;
    fields.push_back(Int64Const(addr));
    fields.push_back(Int64Const(sz * dtype_bytes));
    fields.push_back(Int64Const(static_cast<int64_t>((*tile_type->memref_)->memorySpace_)));
    fields.push_back(Int64Const(sz)); // element count (report display)
    return MakeLogStmt(static_cast<uint32_t>(SanitizerDetection::TileDecl), std::move(fields), call->span_);
}

StmtPtr SanitizerInstrumenter::CollectMakeTensor(const CallPtr& call)
{
    // ptr.make_tensor: args = [ptr_or_tensor, shape_tuple, stride_tuple].
    // The view shares its source's storage, so the op contract ("all accesses
    // must stay inside the source") is checked at declaration time with one
    // ViewShape record carrying expressions only (design principle: every
    // check value travels in the record): the declared dims, the byte
    // footprint sum((dim-1)*stride) + 1 scaled by the view dtype width, and
    // the source's byte capacity (a raw-pointer source records -1: no known
    // bound). Nothing is registered -- the replay needs no host-side view
    // table.
    PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, call->args_.size() >= 0x2)
        << "sanitizer: ptr.make_tensor requires (ptr, shape)";
    auto shape_tuple = As<MakeTuple>(call->args_[1]);
    PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, shape_tuple != nullptr)
        << "sanitizer: ptr.make_tensor shape must be a MakeTuple";
    auto stride_tuple = As<MakeTuple>(call->args_[2]);
    PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, stride_tuple != nullptr)
        << "sanitizer: ptr.make_tensor stride must be a MakeTuple";
    PRO_PASS_INTERNAL_CHECK(
        npu::tile_fwk::InternalError::PASS_INNER_ERROR,
        stride_tuple->elements_.empty() || stride_tuple->elements_.size() == shape_tuple->elements_.size())
        << "sanitizer: ptr.make_tensor stride rank must match the shape";

    // Byte footprint expression: (sum((dim-1)*stride) + 1) * elem_bytes, with
    // an omitted stride read as the compact row-major product of inner dims.
    DataType view_dtype;
    if (call->HasKwarg("dtype")) {
        view_dtype = call->GetKwarg<DataType>("dtype");
    } else if (auto src_shaped = As<ShapedType>(call->args_[0]->GetType())) {
        view_dtype = src_shaped->dtype_;
    } else {
        auto src_ptr = As<PtrType>(call->args_[0]->GetType());
        PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, src_ptr != nullptr)
            << "sanitizer: ptr.make_tensor source must be a Ptr or Tensor";
        view_dtype = src_ptr->dtype_;
    }
    int64_t elem_bytes = std::max<int64_t>(1, static_cast<int64_t>(view_dtype.GetBit()) / 8);
    auto stride_of = [&](size_t i) -> ExprPtr {
        if (i < stride_tuple->elements_.size())
            return stride_tuple->elements_[i];
        ExprPtr product = Int64Const(1);
        for (size_t j = i + 1; j < shape_tuple->elements_.size(); ++j)
            product = MakeMul(product, shape_tuple->elements_[j], call->span_);
        return product;
    };
    ExprPtr fp_elems = Int64Const(1);
    for (size_t i = 0; i < shape_tuple->elements_.size(); ++i) {
        ExprPtr term = MakeMul(MakeSub(shape_tuple->elements_[i], Int64Const(1), call->span_), stride_of(i),
                               call->span_);
        fp_elems = MakeAdd(fp_elems, term, call->span_);
    }
    ExprPtr fp_bytes = MakeMul(fp_elems, Int64Const(elem_bytes), call->span_);

    // Source capacity in bytes: the element-count product of the source
    // tensor's shape scaled by its element width. A raw pointer has no
    // known bound (-1); the record skips the check then.
    ExprPtr src_bytes = Int64Const(-1);
    auto src_tensor = As<TensorType>(call->args_[0]->GetType());
    if (src_tensor != nullptr) {
        int64_t src_elem = std::max<int64_t>(1, static_cast<int64_t>(src_tensor->dtype_.GetBit()) / 8);
        src_bytes = MakeMul(ShapeProduct(src_tensor, call->span_), Int64Const(src_elem), call->span_);
    }

    std::vector<ExprPtr> fields;
    for (uint32_t i = 0; i < kSanitizerMaxTensorDims; ++i) {
        fields.push_back(i < shape_tuple->elements_.size() ? shape_tuple->elements_[i] : Int64Const(0));
    }
    // Source dims for the report display; a raw-pointer source pads zeros
    // (its records skip the check via src_bytes = -1 anyway).
    for (uint32_t i = 0; i < kSanitizerMaxTensorDims; ++i) {
        fields.push_back((src_tensor != nullptr && i < src_tensor->shape_.size()) ? src_tensor->shape_[i] :
                                                                                    Int64Const(0));
    }
    fields.push_back(std::move(fp_bytes));
    fields.push_back(std::move(src_bytes));
    return MakeLogStmt(static_cast<uint32_t>(SanitizerDetection::ViewShape), std::move(fields), call->span_);
}

// ---------------------------------------------------------------------------
// Instrument
// ---------------------------------------------------------------------------

ProgramPtr SanitizerInstrumenter::Instrument(ProgramPtr program)
{
    // Instrument the entry function only, appending the hidden
    // sanitizer_log (ptr) + sanitizer_log_capacity (scalar) parameters.
    //    SIMT functions (SIMT_VF / SIMT_CALLEE) are outside the detection
    //    scope: excluded by function type (the codegen's own criterion), not
    //    only by name -- a same-named SIMT function can never be mistaken
    //    for the entry.
    std::map<std::string, FunctionPtr> new_functions;
    for (const auto& [name, func] : program->functions_) {
        bool is_simt = func != nullptr &&
                       (func->funcType_ == FunctionType::SIMT_VF || func->funcType_ == FunctionType::SIMT_CALLEE);
        if (func == nullptr || is_simt || func->name_ != program->name_) {
            new_functions[name] = func;
            continue;
        }
        auto log_var = std::make_shared<Var>("sanitizer_log", std::make_shared<PtrType>(DataType(DataType::INT8)),
                                             Span::Unknown());
        auto capacity_var = std::make_shared<Var>("sanitizer_log_capacity",
                                                  std::make_shared<ScalarType>(DataType::INT64), Span::Unknown());
        log_param_ = log_var;
        capacity_param_ = capacity_var;

        auto new_body = VisitStmt(func->body_);
        StmtPtr body = (new_body != func->body_) ? new_body : func->body_;

        auto params = func->params_;
        params.push_back(log_var);
        params.push_back(capacity_var);
        new_functions[name] = std::make_shared<Function>(func->name_, std::move(params), func->returnTypes_, body,
                                                         func->span_, func->funcType_, func->entry_);
    }
    return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_, program->debugInfo_);
}

} // namespace

Pass pass::Sanitizer()
{
    return CreateProgramPass(
        [](const ProgramPtr& program) -> ProgramPtr {
            PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, program)
                << "Sanitizer pass cannot run on a null program";
            SanitizerInstrumenter instrumenter;
            return instrumenter.Instrument(program);
        },
        "Sanitizer");
}

} // namespace ir
} // namespace pypto
