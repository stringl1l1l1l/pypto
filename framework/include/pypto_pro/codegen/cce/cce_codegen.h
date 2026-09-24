/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PYPTO_CODEGEN_CCE_CCE_CODEGEN_H_
#define PYPTO_CODEGEN_CCE_CCE_CODEGEN_H_

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "backend/common/backend.h"
#include "codegen/cce/code_context.h"
#include "codegen/cce/code_emitter.h"
#include "codegen/cce/type_converter.h"
#include "codegen/codegen_base.h"
#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/pipe.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/scalar_expr_ops.h"
#include "ir/stmt.h"
#include "ir/type.h"
#include "tilefwk/platform.h"

namespace pypto {

namespace codegen {

/// Tile definition collected from a block.make_tile source.
struct TileDef {
    ir::VarPtr var;
    ir::TileTypePtr tile_type;
};

/// Tensor definition collected from block.load/store access windows.
/// tensor_type and the base pointer are derived from `var` at emit time (not stored):
/// tensor_type = ir::As<ir::TensorType>(var->GetType()); the base pointer is the view's
/// source pointer (ptr.make_tensor) or "<name>_ptr" for a plain tensor parameter.
struct TensorDef {
    ir::VarPtr var;                            ///< tensor variable (parameter or ptr.make_tensor view)
    std::string cce_name;                      ///< emitted declaration name: the tensor's plain cce name for the
                                               ///< first layout it is accessed with, "<name>__v<n>" for each later one
    std::vector<ir::ExprPtr> access_shape;     ///< access window shape (tile-derived) from load/store
    std::optional<std::vector<int>> tile_dims; ///< tile_dims kwarg for strided views (if any)
    bool is_transpose = false;                 ///< loaded with is_transpose=true (needs Layout::DN)
    std::string layout;                        ///< explicit Layout enum (MX loads); else derived from the above
};

/**
 * \brief Key identifying the layout variant one load/store access needs.
 *
 * Layout::DN and the row/col stride order are baked into the GlobalTensor *type*, so accesses
 * that walk one tensor differently cannot share a declaration. The key is a pure function of
 * the access (its is_transpose / tile_dims kwargs and its access shape), so the prescan and
 * the load/store codegen agree on which declaration an access belongs to. Accesses that share
 * a key share a declaration.
 */
[[nodiscard]] std::string TensorLayoutVariantKey(const ir::CallPtr& op);

/// Build a side-effect-free FP32 fmod expression for CCE device code.
[[nodiscard]] std::string BuildFP32FmodExpression(const std::string& lhs, const std::string& rhs);

/// Definition of one user-visible C++ class materialized from a TupleType.
struct StructDefinition {
    std::string name;
    std::vector<std::string> fields;
    std::vector<ir::TypePtr> types;
    bool is_tiling = false;
    bool requires_volatile = false;
};

/**
 * \brief CCE code generator for converting PyPTO IR to pto-isa C++ code
 *
 * CCECodegen traverses the IR using the visitor pattern and generates
 * compilable C++ code using pto-isa instructions. It handles:
 * - Function prologue (signature, argument unpacking, type definitions)
 * - Function body (block operations, sync operations, control flow)
 * - Type conversions and memory management
 */
class CCECodegen : public CodegenBase {
protected:
    using CodegenBase::VisitExpr_;
    using CodegenBase::VisitStmt_;

public:
    /** \brief Construct a CCE code generator for one fixed Cube/Vector target. */
    explicit CCECodegen(ir::SectionKind target);

    /**
     * \brief Generate a standalone TilingData header from an ordered field definition.
     *
     * This is the same header format emitted during full kernel codegen, but does not
     * require a Program or run any IR passes.
     */
    [[nodiscard]] static std::string GenerateTilingHeader(const std::string& type_name,
                                                          const std::vector<std::string>& fields,
                                                          const std::vector<ir::TypePtr>& types,
                                                          bool requires_volatile = false);

    /**
     * \brief Generate a single C++ file from a PyPTO IR Program
     *
     * Generates a single __global__ AICORE kernel from SSA IR with:
     * - PTO-style function signature
     * - Target-specific function naming and DAV macro guards
     * - constexpr for compile-time constants
     * - FFTS support for cross-core sync
     *
     * \param program The IR Program to generate code for
     * \return Generated C++ code as a single string
     */
    [[nodiscard]] std::string GenerateSingle(const ir::ProgramPtr& program, const std::string& arch);

    /**
     * \brief Tiling struct headers generated by the last GenerateSingle call.
     *
     * Maps header filename (e.g. "OpTiling_tiling.h") to its content (a `#pragma once`
     * guarded `struct <ClassName> { ... };`). The kernel.cpp `#include`s these by name;
     * the caller (jit.py) writes them next to kernel.cpp in the build dir.
     */
    [[nodiscard]] const std::map<std::string, std::string>& GetTilingHeaders() const { return tiling_headers_; }

    // CodegenBase interface (unified API for operator codegen callbacks)
    [[nodiscard]] std::string GetCurrentResultTarget() const override { return current_target_var_; }
    void Emit(const std::string& line) override;
    std::string GetExprAsCode(const ir::ExprPtr& expr) override;
    /// Resolve a tuple-typed expression to the MakeTuple backing it (the tuple counterpart of
    /// GetExprAsCode, which yields a C++ string). Returns null when the expression is not a tuple.
    ir::MakeTuplePtr GetExprAsMakeTuple(const ir::ExprPtr& expr);
    [[nodiscard]] std::string GetTypeString(const ir::DataType& dtype) const override;
    /// True when currently generating code inside a VF section (CCE __VEC_SCOPE__).
    [[nodiscard]] bool IsInVFSection() const { return in_vf_section_; }
    /// True when emitting an A5 `__simt_vf__` or `__simt_callee__` body.
    [[nodiscard]] bool IsInSimtContext() const
    {
        return current_function_type_ == ir::FunctionType::SIMT_VF ||
               current_function_type_ == ir::FunctionType::SIMT_CALLEE;
    }
    int GetTileOffsetCounter() { return tile_offset_counter_++; }
    /// Resolve a VF tile expr (plain tile or `tile[offset]`) to its `__ubuf__` pointer, typed as the
    /// tile element and hoisted before `__VEC_SCOPE__`. A `tile[offset]` load reuses the rvalue
    /// `(vf_tile_ptr_N + off)`; otherwise a `vf_tile_ptr_N` is declared once and cached. With
    /// `is_post_update`, the access gets its own hoisted variable, kept separate from a plain base
    /// pointer to the same tile (e.g. a POST_UPDATE store cursor that must not alias a base load),
    /// and even a `tile[offset]` materialises a real lvalue var rather than the shared rvalue.
    std::string GetOrCreateVFTilePtr(const ir::ExprPtr& expr, bool is_post_update = false);
    int64_t GetConstIntValue(const ir::ExprPtr& expr) override;
    std::string GetVarName(const ir::VarPtr& var) override;

    const TypeConverter& GetTypeConverter() const { return type_converter_; }

    /** \brief Get the target architecture */
    npu::tile_fwk::NPUArch GetArch() const { return arch_; }

    /** \brief Get the fixed Cube/Vector target selected for this generation. */
    ir::SectionKind GetTarget() const { return target_; }

    /** \brief Get the base address of a tile variable (from TASSIGN in prologue) */
    std::string GetTileAddress(const std::string& tile_name) const
    {
        auto it = tile_addresses_.find(tile_name);
        if (it != tile_addresses_.end())
            return it->second;
        return "0x0";
    }

    /** \brief Set the base address of a tile variable (for subview offset registration). */
    void SetTileAddress(const std::string& tile_name, const std::string& addr) { tile_addresses_[tile_name] = addr; }

    /** \brief Register a VF RegTensor variable (called from EmitVFRegTensor). */
    void RegisterRegTensorVar(const std::string& cpp_name) { reg_tensor_vars_.insert(cpp_name); }

    /** \brief Push a RegTensor declaration to the hoisting buffer (emitted at __VEC_SCOPE__ top). */
    void HoistRegTensorDecl(const std::string& decl) { vf_reg_hoisted_decls_.push_back(decl); }

    /** \brief Check if a C++ variable name was declared as a VF RegTensor. */
    bool IsRegTensorVar(const std::string& cpp_name) const { return reg_tensor_vars_.count(cpp_name) > 0; }

    /** \brief Register a VF MaskReg variable (called from EmitVFCreateMask/Compare/etc). */
    void RegisterMaskRegVar(const std::string& cpp_name) { mask_reg_vars_.insert(cpp_name); }

    /** \brief Check if a C++ variable name was declared as a VF MaskReg. */
    bool IsMaskRegVar(const std::string& cpp_name) const { return mask_reg_vars_.count(cpp_name) > 0; }

    /** \brief Register a VF AddrReg variable (called from EmitVFCreateAddrReg). */
    void RegisterAddrRegVar(const std::string& cpp_name) { addr_reg_vars_.insert(cpp_name); }

    /** \brief Check if a C++ variable name was declared as a VF AddrReg. */
    bool IsAddrRegVar(const std::string& cpp_name) const { return addr_reg_vars_.count(cpp_name) > 0; }

    /** \brief Register a VF UnalignReg variable (called from EmitVFUnalignRegForStore/Load). */
    void RegisterUnalignRegVar(const std::string& cpp_name) { unalign_reg_vars_.insert(cpp_name); }

    /** \brief Check if a C++ variable name was declared as a VF UnalignReg. */
    bool IsUnalignRegVar(const std::string& cpp_name) const { return unalign_reg_vars_.count(cpp_name) > 0; }

    /** \brief Map logical tensor coordinates to a layout-aware physical element offset. */
    std::string ComputeTensorOffset(const ir::TensorTypePtr& tensor_type, const ir::MakeTuplePtr& offsets);

    /** \brief Render a shape dimension rounded up to an alignment, folding constants when possible. */
    std::string ComputeAlignedShapeDimension(const ir::ExprPtr& dimension, int64_t alignment);

    const ir::Span& GetCurrentStmtSpan() const { return current_stmt_span_; }
    const ir::Span& GetCurrentExprSpan() const { return current_expr_span_; }

    /**
     * \brief Register a tensor variable's underlying raw pointer (CCE-specific).
     *
     * Associates a tensor variable (parameter, ptr.make_tensor view, or iter_arg) with its
     * raw pointer expression. Used for GlobalTensor construction and TASSIGN address
     * computation in block.load/store. Same-key re-registration with a different value warns.
     */
    void RegisterPointer(const std::string& tensor_var_name, const std::string& ptr_name);

    /**
     * \brief Get pointer name for a variable (CCE-specific)
     */
    std::string GetPointer(const std::string& var_name);

    /**
     * \brief Check whether a raw pointer mapping has been registered (non-throwing).
     *
     * Distinguishes a tensor parameter / view (pointer already registered) from an
     * IfStmt phi return var (pointer not yet known).
     */
    [[nodiscard]] bool HasPointer(const std::string& var_name) const;

    /**
     * \brief Check whether a tile address (TASSIGN) has been registered.
     *
     * Distinguishes a tile with a known base address from one that needs to be
     * resolved via .data() or memref.
     */
    [[nodiscard]] bool HasTileAddress(const std::string& tile_name) const;

    /**
     * \brief Register a struct TupleType definition for deferred class emission.
     */
    void RegisterStructDefinition(const ir::TupleTypePtr& tuple_type, const std::string& type_name,
                                  const std::vector<std::string>& fields, bool is_tiling);

    /** \brief Get the emitted C++ member type for a struct field. */
    [[nodiscard]] std::string GetStructFieldTypeString(const ir::TypePtr& type) const;

    /**
     * \brief Mark a registered TupleType as requiring volatile member declarations.
     */
    void MarkStructVolatile(const ir::TypePtr& type);

    /**
     * \brief Check whether a PIPE_V mutex lock/unlock should be skipped.
     *
     * Returns true (skip) when arch is A5, pipe is PIPE_V, and all buf_ids
     * in the vector are only used by PIPE_V (no cross-pipe synchronization needed).
     */
    bool ShouldSkipVPipeMutex(ir::PipeType pipe, const std::vector<int>& buf_ids) const;

    /**
     * \brief Generate GlobalTensor type declaration and instance for a TensorDef.
     *
     * Emits shape type alias, stride type alias, GlobalTensor type alias, and instance
     * declaration, all named after def.cce_name -- this layout variant's name, as recorded by
     * CollectTensorDefs. The tensor type and the base pointer still come from def.var, which
     * every variant of one tensor shares; dn / tile_dims come from def. Public so ptr.make_tensor
     * codegen can emit a view in place.
     */
    void GenerateGlobalTensorTypeDeclaration(const TensorDef& def);

    /**
     * \brief Emit a fresh [1, numel] ND GlobalTensor declaration over *tensor_var*, in place.
     *
     * For GM-writing ops whose declaration has no consumer other than the op itself (e.g.
     * block.init_output): the declaration is a linearized [1, numel] view -- *numel_expr*
     * is the C++ element-count product of the tensor's shape dims (static dims fold to
     * literals, dynamic dims reference runtime scalars). Returns a unique instance name
     * ("<base>__io<n>"), so repeated calls each get their own declaration with no dedup.
     */
    std::string DeclareFlatGlobalTensor(const ir::VarPtr& tensor_var, const std::string& numel_expr);
    /**
     * \brief Look up every prescanned layout variant of a tensor by its cce variable name.
     *
     * Empty when the name was never accessed by a block.load/store (no declaration needed);
     * one def per layout the accesses need. Populated by CollectTensorDefs during the prologue.
     */
    [[nodiscard]] std::vector<const TensorDef*> GetTensorDefs(const std::string& name) const;

    /**
     * \brief Bind one access to its declaration and return the instance to read or write through.
     *
     * Selects the declaration matching this access's layout -- a tensor also read with a different
     * order has one per layout, and only the matching one walks it correctly -- then points its
     * hoisted instance at the access and resizes it in place. valid_rows/valid_cols are the logical
     * transfer shape; layout-specific physical dimensions are derived while binding.
     */
    [[nodiscard]] std::string BindGlobalTensor(const ir::VarPtr& tensor_var, const ir::CallPtr& op,
                                               const std::string& pointer_expr, const std::string& valid_rows,
                                               const std::string& valid_cols);

protected:
    // Override visitor methods for code generation - Statements
    void VisitStmt_(const ir::AssignStmtPtr& op) override;
    void VisitStmt_(const ir::EvalStmtPtr& op) override;
    void VisitStmt_(const ir::ReturnStmtPtr& op) override;
    void VisitStmt_(const ir::ForStmtPtr& op) override;
    void VisitStmt_(const ir::WhileStmtPtr& op) override;
    void VisitStmt_(const ir::IfStmtPtr& op) override;
    void VisitStmt_(const ir::YieldStmtPtr& op) override;
    void VisitStmt_(const ir::SectionStmtPtr& op) override;
    void VisitStmt_(const ir::SeqStmtsPtr& op) override;
    void VisitStmt_(const ir::BreakStmtPtr& op) override;
    void VisitStmt_(const ir::ContinueStmtPtr& op) override;

    void VisitStmt(const ir::StmtPtr& stmt) override;
    void VisitExpr(const ir::ExprPtr& expr) override;

    // Override visitor methods for code generation - Expressions
    // Leaf nodes
    void VisitExpr_(const ir::VarPtr& op) override;

    void VisitExpr_(const ir::ConstIntPtr& op) override;
    void VisitExpr_(const ir::ConstFloatPtr& op) override;
    void VisitExpr_(const ir::ConstBoolPtr& op) override;
    void VisitExpr_(const ir::CallPtr& op) override;
    void VisitExpr_(const ir::GetItemExprPtr& op) override;
    void VisitExpr_(const ir::MakeTuplePtr& op) override;

    // Binary operations
    void VisitExpr_(const ir::AddPtr& op) override;
    void VisitExpr_(const ir::SubPtr& op) override;
    void VisitExpr_(const ir::MulPtr& op) override;
    void VisitExpr_(const ir::FloorDivPtr& op) override;
    void VisitExpr_(const ir::FloorModPtr& op) override;
    void VisitExpr_(const ir::FloatDivPtr& op) override;
    void VisitExpr_(const ir::MinPtr& op) override;
    void VisitExpr_(const ir::MaxPtr& op) override;
    void VisitExpr_(const ir::PowPtr& op) override;
    void VisitExpr_(const ir::EqPtr& op) override;
    void VisitExpr_(const ir::NePtr& op) override;
    void VisitExpr_(const ir::LtPtr& op) override;
    void VisitExpr_(const ir::LePtr& op) override;
    void VisitExpr_(const ir::GtPtr& op) override;
    void VisitExpr_(const ir::GePtr& op) override;
    void VisitExpr_(const ir::AndPtr& op) override;
    void VisitExpr_(const ir::OrPtr& op) override;
    void VisitExpr_(const ir::XorPtr& op) override;
    void VisitExpr_(const ir::BitAndPtr& op) override;
    void VisitExpr_(const ir::BitOrPtr& op) override;
    void VisitExpr_(const ir::BitXorPtr& op) override;
    void VisitExpr_(const ir::BitShiftLeftPtr& op) override;
    void VisitExpr_(const ir::BitShiftRightPtr& op) override;

    // Unary operations
    void VisitExpr_(const ir::AbsPtr& op) override;
    void VisitExpr_(const ir::NegPtr& op) override;
    void VisitExpr_(const ir::NotPtr& op) override;
    void VisitExpr_(const ir::BitNotPtr& op) override;
    void VisitExpr_(const ir::CastPtr& op) override;

private:
    /**
     * \brief Emit yield-assignment code for if-stmt return variables.
     *
     * For each return variable, writes the buffered YieldStmt expression to the
     * corresponding return variable and clears yield_buffer_ when done.
     */
    void EmitYieldAssignments(const std::vector<ir::VarPtr>& return_vars);

    /**
     * \brief Emit full phi-style IfStmt codegen.
     *
     * Declares return variables before the if, emits then and else bodies, and
     * writes yield-assignment code in each branch.
     */
    void EmitFullPhiIf(const ir::IfStmtPtr& op);

    /**
     * \brief Generate function body
     *
     * Visits the function body statement to generate the main code.
     *
     * \param func The function to generate body for
     */
    void GenerateBody(const ir::FunctionPtr& func);

    /**
     * \brief Extract constant integer value from expression
     *
     * \param expr The expression (must be ConstInt)
     * \return The integer value
     */
    int64_t ExtractConstInt(const ir::ExprPtr& expr) const;

    /**
     * \brief Collect all TileType variables from function body
     *
     * Recursively traverses the statement tree to find all variables
     * with TileType that need Tile declarations in the prologue.
     *
     * \param stmt The statement to scan (typically func->body_)
     * \return Vector of (Var, TileType) pairs
     */
    std::vector<std::pair<ir::VarPtr, ir::TileTypePtr>> CollectTileVariables(const ir::StmtPtr& stmt);

    /// Collect one TensorDef per accessed tensor var *and layout variant* from block.load/store
    /// access windows into tensor_defs_, keyed by (tensor cce name, layout key) (parameters and
    /// ptr.make_tensor views both collected; used for prologue param-tensor emission and
    /// in-place view emission at the make_tensor op).
    void CollectTensorDefs(const ir::FunctionPtr& func);

    /**
     * \brief Extract shape dimensions from shape expressions
     *
     * Converts a vector of shape expressions (assumed to be ConstInt)
     * into a vector of integer dimensions.
     *
     * \param shape_exprs Vector of shape expressions (ConstInt)
     * \return Vector of integer dimensions
     */
    std::vector<int64_t> ExtractShapeDimensions(const std::vector<ir::ExprPtr>& shape_exprs) const;

    /**
     * \brief Format address as hexadecimal string
     *
     * Converts an integer address to hex format for TASSIGN instructions.
     *
     * \param addr Address value
     * \return Hex string (e.g., "0x0", "0x10000")
     */
    std::string FormatAddressHex(int64_t addr);

    /**
     * \brief Register each tiling TupleType parameter for deferred header emission.
     */
    void RegisterTilingStructTypes(const ir::FunctionPtr& func);

    /**
     * \brief Emit all registered class definitions and materialize tiling headers.
     */
    void EmitStructTypes();

    /**
     * \brief Emit the kernel-entry copy from the `<name>_ptr` GM pointer into the local
     *        struct `<name>` (cube: copy_data_align64 GM→struct; vector: copy GM→UB scratch,
     *        then copy_data_align64 UB→struct).
     */
    void EmitTilingStructCopy(const ir::FunctionPtr& func);

    /**
     * \brief Generate Tile type declaration and instance
     *
     * Emits type alias and instance declaration for a Tile variable.
     * Automatically extracts memref address from tile_type if present and emits TASSIGN.
     *
     * \param var_name Variable name for the tile
     * \param tile_type The TileType to generate declaration for (memref extracted automatically)
     */
    void GenerateTileTypeDeclaration(const std::string& var_name, const ir::TileTypePtr& tile_type);

    /**
     * \brief Generate PTO-style function signature and prologue
     *
     * Emits __global__ AICORE void func_name(__gm__ type* p, ...) with constexpr
     * scalars and target-specific declarations.
     */
    void GenerateSinglePrologue(const ir::FunctionPtr& func, bool has_cross_sync);
    std::string BuildSimtFunctionSignature(const ir::FunctionPtr& func);
    void GenerateSimtFunction(const ir::FunctionPtr& func);
    std::string GenerateSimtCalleeCall(const ir::CallPtr& op, const ir::FunctionPtr& callee);
    void PreScanKernel(const ir::FunctionPtr& kernel_func);
    void ResetFunctionGenerationState();

    bool DetectCrossCoreSyncOps(const ir::StmtPtr& stmt);

    // --- Phase 5: ForStmt helpers ---

    /**
     * \brief Register loop iteration arguments and emit their initialization.
     *
     * Unknown init values use the uninitialized declaration path; all other init
     * values use the initialized declaration path. Returns the sanitized names
     * of each iter-arg for later yield assignment. Shared by ForStmt and WhileStmt.
     */
    std::vector<std::string> RegisterLoopIterArgs(const std::vector<ir::IterArgPtr>& iterArgs);

    /// Write source expressions to typed loop-carried slots, skipping self-assignments.
    /// TupleType is dispatched here by its CCE representation: tuples with a
    /// backing array are copied element-wise, while aggregates recurse into leaf slots.
    /// Shared by If/loop yields and native jumps.
    void EmitCarriedAssignments(const std::vector<ir::VarPtr>& targets, const std::vector<ir::ExprPtr>& sources);

    /// Generate `type name`, `type name = value`, or `name = value`. A null value
    /// omits assignment; initialize controls whether the generated statement includes the type.
    /// Shared entry validation for EmitVariable: skips UnknownType/NoneType targets, treats
    /// UnknownType/NoneType sources as valueless, and drops self-assignment back edges.
    bool CheckEmitVariable(const ir::VarPtr& target, ir::ExprPtr& value, bool initialize);
    void EmitVariable(const ir::VarPtr& target, ir::ExprPtr value, bool initialize);
    void EmitTupleVariable(const std::string& name, const ir::TupleTypePtr& type, const ir::ExprPtr& value,
                           bool initialize);
    void RegisterLoopReturnVars(const std::vector<ir::VarPtr>& returnVars,
                                const std::vector<std::string>& iterArgNames);

    /**
     * \brief Emit a C++ loop shared by both loop kinds.
     *
     * The caller supplies the fully-formed loop header line ("for (...) {" or "while (...) {");
     * For loops write both break and continue values to the iter-arg slots. While loops keep
     * continue values in iter-arg slots and break values in independent return-var slots.
     */
    void EmitLoop(const std::string& header, const ir::StmtPtr& body, const std::vector<ir::IterArgPtr>& iterArgs,
                  const std::vector<ir::VarPtr>& returnVars, const std::vector<std::string>& iterArgNames,
                  bool separateWhileResults);

    // --- Native break/continue support ---

    /// Write a native jump's self-described values to the innermost loop's matching slots.
    void EmitJumpCarriedWriteback(const std::vector<ir::ExprPtr>& values, bool isBreak);

    // --- Phase 6: GenerateSinglePrologue helpers ---

    /**
     * \brief Emit the __global__ AICORE function signature and opening boilerplate.
     *
     * Collects dynamic dim variables, builds the parameter list, emits the opening
     * brace, registers dynamic dims, and emits FFTS setup if needed.
     */
    void EmitSingleFunctionSignature(const ir::FunctionPtr& func, bool has_cross_sync);

    /**
     * \brief Emit GlobalTensor declarations for one target Program.
     *
     * Registers the C++ name (= tensor var name) and pointer ("<name>_ptr") for every
     * tensor parameter and collects TensorDefs.
     */
    void EmitSingleTensorDeclarations(const ir::FunctionPtr& func);

    /** \brief Emit Tile declarations for one target Program. */
    void EmitSingleTileDeclarations(const ir::FunctionPtr& func);

    void EmitTileDeclarations(const std::vector<TileDef>& tile_defs,
                              const std::vector<std::pair<ir::VarPtr, ir::VarPtr>>& deduped_aliases);
    void EmitDedupedTileAliases(const std::vector<std::pair<ir::VarPtr, ir::VarPtr>>& deduped_aliases);

    // --- Offset computation helpers ---
    std::string ComputeStrideOffset(const ir::TensorTypePtr& tensor_type, const ir::MakeTuplePtr& offsets);
    std::string ComputeNZStrideOffset(const ir::TensorTypePtr& tensor_type, const ir::MakeTuplePtr& offsets);

    // --- Phase 7: GenerateGlobalTensorTypeDeclaration helpers ---

    std::vector<std::string> BuildTensorStrideExpressions(const ir::TensorTypePtr& tensor_type);

    /// Whether an access walks the tensor down columns: transposed, or a single-column window.
    static bool IsDNAccessLayout(const std::vector<int64_t>& shape_dims, bool is_transpose);

    /// The Stride ctor arguments an access walks the tensor with, row/col exchanged when transposed.
    std::string BuildAccessStrideArgs(const ir::TensorTypePtr& tensor_type,
                                      const std::optional<std::vector<int>>& tile_dims, bool is_transpose, bool is_mx);

    /**
     * \brief Emit one GlobalTensor instance over a declaration made by GenerateGlobalTensorTypeDeclaration.
     *
     * Renders `<decl>Type <instance>(<pointer>, <decl>ShapeDim5(<shape_args>), <decl>StrideDim5(<stride_args>));`.
     * When both metadata argument strings are empty, only the pointer is emitted.
     */
    void EmitGlobalTensorInstance(const std::string& instance_name, const std::string& decl_name,
                                  const std::string& pointer_expr, const std::string& shape_args,
                                  const std::string& stride_args);

    // Dual-mode context for expression visitor pattern
    std::string current_target_var_;              ///< INPUT: Assignment target variable name (for Call expressions)
    std::string current_expr_value_;              ///< OUTPUT: Inline C++ value for scalar / tile expressions
    ir::MakeTuplePtr current_tuple_;              ///< OUTPUT: underlying MakeTuple for tuple-typed expressions
    std::vector<ir::ExprPtr> yield_buffer_;       ///< If-branch Yield expressions awaiting phi assignment
    const ir::IRDebugInfo* debug_info_ = nullptr; ///< Program tuple metadata, required at codegen entry
    std::map<std::string, std::string> tiling_headers_;          ///< Tiling struct headers (filename -> content)
    std::map<std::string, StructDefinition> struct_definitions_; ///< Struct type name ->definition

    CodeEmitter emitter_;                                            ///< Code emitter for structured output
    CodeContext context_;                                            ///< Context for variable tracking
    TypeConverter type_converter_;                                   ///< Type converter
    const backend::Backend* backend_;                                ///< CCE backend instance (for op info, core type)
    npu::tile_fwk::NPUArch arch_ = npu::tile_fwk::NPUArch::DAV_2201; ///< Target architecture
    const ir::SectionKind target_;                                   ///< Fixed Cube/Vector target for this generator
    bool in_vf_section_ = false;                                     ///< True only while emitting a nested VF section
    ir::FunctionType current_function_type_ = ir::FunctionType::OPAQUE;
    std::map<std::string, ir::FunctionPtr> simt_callees_;
    std::unordered_map<std::string, std::string> tensor_to_pointer_; ///< Tensor var name ->raw pointer expression
    /// Prescan: (tensor cce name, layout key) ->TensorDef. One entry per layout an access needs,
    /// so every variant of one tensor is a contiguous range (see GetTensorDefs).
    std::map<std::pair<std::string, std::string>, TensorDef> tensor_defs_;
    int flat_tensor_decl_count_ = 0; ///< Monotonic suffix for DeclareFlatGlobalTensor names, reset per function
    std::map<std::string, std::string> tile_addresses_; ///< tile_name ->TASSIGN address expression

    std::map<std::pair<bool, std::string>, std::string>
        vf_tile_ptrs_; ///< (is_post_update, VF tile expr code) -> hoisted vf_tile_ptr_N var. A dedicated var (e.g. a
                       ///< POST_UPDATE store cursor) and a plain base pointer to the same tile are kept separate, so a
                       ///< cursor store does not corrupt a base load of the same tile.
    std::vector<std::string> section_hoisted_decls_;    ///< VF section decls hoisted before __VEC_SCOPE__ (pre mem_bar)
    std::map<int, std::set<ir::PipeType>> mutex_pipes_; ///< buf_id ->pipes in this target Program

    struct LoopWritebackTargets {
        std::vector<ir::VarPtr> continueVars;
        std::vector<ir::VarPtr> breakVars;
    };

    /// Per active loop, keep separate continue and break write-back slots.
    std::vector<LoopWritebackTargets> loop_target_stack_;

    /**
     * \brief Pre-scan the IR for mutex_id ->pipe mappings.
     *
     * Used on A5 to eliminate redundant PIPE_V get_buf/rls_buf: if a buf_id is
     * only used by PIPE_V (never by MTE2/MTE3/M/etc.), the V-side mutex
     * synchronization is unnecessary (V→V ops execute in order within the pipe).
     */
    void CollectMutexPipeInfo(const ir::StmtPtr& stmt);

    int tile_offset_counter_ = 0; ///< Counter for unique tile-offset GetItemExpr temp names
    int vf_tile_ptr_counter_ = 0; ///< Counter for unique VF base-ptr var names (vf_tile_ptr_N)

    // Tile type dedup: maps tile_type_str ->alias name already emitted
    std::map<std::string, std::string> emitted_tile_types_;

    /// Track emitted tile reference aliases to avoid C++ redefinition errors
    std::set<std::string> emitted_tile_aliases_;

    /// Track VF RegTensor variable names (C++ names set by EmitVFRegTensor)
    std::set<std::string> reg_tensor_vars_;

    /// RegTensor declarations to hoist to the top of __VEC_SCOPE__ (avoid loop-interior placement)
    std::vector<std::string> vf_reg_hoisted_decls_;

    /// Track VF MaskReg variable names (C++ names set by EmitVFCreateMask/Compare/etc)
    std::set<std::string> mask_reg_vars_;

    /// Track VF AddrReg variable names (C++ names set by EmitVFCreateAddrReg)
    std::set<std::string> addr_reg_vars_;

    /// Track VF UnalignReg variable names (C++ names set by EmitVFUnalignRegForStore/Load)
    std::set<std::string> unalign_reg_vars_;

    // Tuple Var C++ name -> the MakeTuple that was assigned to it.
    // Populated during body codegen by VisitStmt_(AssignStmtPtr) on tuple-typed lhs:
    // VisitExpr(rhs) drives current_tuple_ through MakeTuple literals, tuple-var aliases,
    // and chained static GetItem, all via the visitExpr chain. ForStmt iterArg/returnVar
    // are propagated in VisitStmt_(ForStmtPtr).
    std::map<std::string, ir::MakeTuplePtr> tuple_var_to_make_tuple_;

    // Array-valued MakeTuple object -> its unique C++ backing array. Parser constant
    // propagation can replace tuple Vars with their MakeTuple, so array identity
    // follows the IR object rather than a particular Var name.
    std::map<const ir::MakeTuple*, std::string> tuple_backing_arr_;

    /**
     * \brief Whether all element types of a tuple are structurally identical.
     *
     * Gate for emitting the dynamic-index array at the MakeTuple assignment: an array
     * is only valid when every element maps to the same C++ type. Empty tuple -> false.
     */
    bool IsHomogeneousTuple(const ir::TupleTypePtr& tt) const;

    /// Semantic metadata registered for one exact tuple type, or null for a plain tuple.
    const ir::TupleTypeInfo* GetTupleTypeInfo(const ir::TupleTypePtr& tuple_type) const;

    /// The C++ struct type name registered for a tuple type, or null if it is not a struct.
    /// Reads the parser's side table, so the answer does not depend on how far codegen has
    /// walked the body.
    const std::string* GetStructName(const ir::TupleTypePtr& tuple_type) const;

    /**
     * \brief Whether a tuple is represented as a C++ array (vs. flattened leaf slots).
     *
     * Pure function of the type: it reads only the TupleType and debug_info_, which is
     * captured once at GenerateSingle entry, so it does not depend on how far codegen has
     * walked the body.
     *
     * A tuple gets an array only when it is homogeneous and has no TupleTypeInfo
     * registration. Named tuples and structs are emitted as aggregate leaf slots.
     */
    bool IsArrayTuple(const ir::TupleTypePtr& tt) const;

    /**
     * \brief Collect element C++ names from a tuple value expression.
     *
     * Body-codegen use. visitExpr-driven: VisitExpr(tuple_value) resolves the
     * underlying MakeTuple into current_tuple_, then each element is VisitExpr'd
     * to produce its C++ name in current_expr_value_. Dynamic-GetItem callers
     * guarantee all elements are tile/scalar Var or const literals, so each
     * element visit yields a non-empty current_expr_value_.
     * Returns empty vector on failure.
     */
    std::vector<std::string> CollectTupleElemNames(const ir::ExprPtr& tuple_value);
    std::string GetGeneratedType(const ir::TypePtr& type) const;
    std::string BuildDynamicTupleArrayDecl(const ir::TypePtr& elem_type, const std::vector<std::string>& elem_names,
                                           const std::string& arr_name) const;

    ir::Span current_stmt_span_;
    ir::Span current_expr_span_;
};

} // namespace codegen
} // namespace pypto

#endif // PYPTO_CODEGEN_CCE_CCE_CODEGEN_H_
