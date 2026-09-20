/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bindings/ir/bindings.h"
#include "core/error.h"
#include "ir/transforms/ir_property.h"
#include "ir/transforms/pass_context.h"
#include "ir/transforms/passes.h"
#include "ir/verifier/verification_error.h"
#include "ir/verifier/verifier.h"

namespace pypto {
namespace ir {

void BindPasses(py::module_& m)
{
    py::enum_<IRProperty>(m, "IRProperty", "Verifiable IR properties")
        .value("SSAForm", IRProperty::SSAForm, "IR is in SSA form")
        .value("TypeChecked", IRProperty::TypeChecked, "IR has passed type checking")
        .value("NoNestedCalls", IRProperty::NoNestedCalls, "No nested call expressions")
        .value("NormalizedStmtStructure", IRProperty::NormalizedStmtStructure, "Statement structure normalized")
        .value("FlattenedSingleStmt", IRProperty::FlattenedSingleStmt, "Single-statement blocks flattened")
        .value("SplitIncoreOrch", IRProperty::SplitIncoreOrch, "InCore scopes outlined into separate functions")
        .value("HasMemRefs", IRProperty::HasMemRefs, "MemRef objects initialized on variables")
        .value("IncoreBlockOps", IRProperty::IncoreBlockOps, "InCore functions use block ops (tile types, load/store)")
        .value("AllocatedMemoryAddr", IRProperty::AllocatedMemoryAddr,
               "All MemRefs have valid addresses within buffer limits");

    py::class_<IRPropertySet>(m, "IRPropertySet", "A set of IR properties")
        .def(py::init<>(), "Create an empty property set")
        .def("insert", &IRPropertySet::Insert, py::arg("prop"), "Insert a property")
        .def("remove", &IRPropertySet::Remove, py::arg("prop"), "Remove a property")
        .def("contains", &IRPropertySet::Contains, py::arg("prop"), "Check if property is in set")
        .def("contains_all", &IRPropertySet::ContainsAll, py::arg("other"), "Check if set contains all of other")
        .def("union_with", &IRPropertySet::Union, py::arg("other"), "Return union of this and other")
        .def("intersection", &IRPropertySet::Intersection, py::arg("other"), "Return intersection")
        .def("difference", &IRPropertySet::Difference, py::arg("other"), "Return this minus other")
        .def("empty", &IRPropertySet::Empty, "Check if empty")
        .def("to_list", &IRPropertySet::ToVector, "Convert to list of properties")
        .def("__str__", &IRPropertySet::ToString)
        .def("__repr__", &IRPropertySet::ToString)
        .def("__eq__", &IRPropertySet::operator==)
        .def("__ne__", &IRPropertySet::operator!=);

    py::enum_<VerificationMode>(m, "VerificationMode", "Controls when property verification runs")
        .value("NONE", VerificationMode::None, "No automatic verification")
        .value("BEFORE", VerificationMode::Before, "Verify required properties before each pass")
        .value("AFTER", VerificationMode::After, "Verify produced properties after each pass")
        .value("BEFORE_AND_AFTER", VerificationMode::BeforeAndAfter, "Verify both before and after each pass");

    py::enum_<VerificationLevel>(m, "VerificationLevel", "Controls automatic verification in PassPipeline")
        .value("NONE", VerificationLevel::None, "No automatic verification (fastest)")
        .value("BASIC", VerificationLevel::Basic, "Verify lightweight properties once per pipeline (default)");

    m.def(
        "get_verified_properties", []() { return GetVerifiedProperties(); },
        "Get the set of properties automatically verified during compilation");
    m.def("get_default_verification_level", &GetDefaultVerificationLevel,
          "Get the default verification level (from PYPTO_VERIFY_LEVEL env var, default: Basic)");

    py::class_<Pass>(m, "Pass", "Opaque pass object. Do not instantiate directly - use factory functions.")
        .def("__call__", &Pass::operator(), py::arg("program"), "Execute pass on program")
        .def("get_name", &Pass::GetName, "Get the name of the pass")
        .def("get_required_properties", &Pass::GetRequiredProperties, "Get required properties")
        .def("get_produced_properties", &Pass::GetProducedProperties, "Get produced properties")
        .def("get_invalidated_properties", &Pass::GetInvalidatedProperties, "Get invalidated properties")
        .def_static("init_mem_ref", &pass::InitMemRef, "Create a memory reuse pass")
        .def_static("basic_memory_reuse", &pass::BasicMemoryReuse, "Create a basic memory reuse pass")
        .def_static("allocate_memory_addr", &pass::AllocateMemoryAddr, "Create an allocate memory address pass")
        .def_static("outline_incore_scopes", &pass::OutlineIncoreScopes,
                    "Create a pass that outlines InCore scopes into separate functions")
        .def_static("convert_tensor_to_block_ops", &pass::ConvertTensorToBlockOps,
                    "Create a pass that converts tensor ops to block ops in InCore functions")
        .def_static("flatten_call_expr", &pass::FlattenCallExpr, "Create a pass that flattens nested call expressions")
        .def_static("normalize_stmt_structure", &pass::NormalizeStmtStructure,
                    "Create a pass that normalizes statement structure")
        .def_static("flatten_single_stmt", &pass::FlattenSingleStmt,
                    "Create a pass that recursively flattens single-statement blocks")
        .def_static("run_verifier", &pass::RunVerifier, py::arg("disabled_rules") = std::vector<std::string>{},
                    "Create a verifier pass with configurable rules")
        .def_static("aggressive_dce", &pass::AggressiveDCE, "Eliminate dead code")
        .def_static("canonicalize", &pass::Canonicalize, "Canonicalize IR")
        .def_static("infer_token_pass", &pass::InferTokenPass, "Infer token dependencies")
        .def_static("remove_redundant_token_pass", &pass::RemoveRedundantTokenPass,
                    "Remove redundant token dependencies")
        .def_static("merge_stmts_into_if", &pass::MergeStmtsIntoIf, "Merge stmts into if branches")
        .def_static("simplify_symbolic_scalar", &pass::SimplifySymbolicScalar,
                    "Simplify symbolic scalars under their enclosing branch condition")
        .def_static("create_root_functions", &pass::CreateRootFunctions, "Create root functions from IR")
        .def_static("infer_multi_iter_overlap", &pass::InferMultiIterOverlap,
                    "Mark Function RebuildableMultiIterNoOverlap for assemble-level loop-carried non-overlap")
        .def_static("finalize_dynamic_function", &pass::FinalizeDynamicFunction,
                    "Finalize dynamic functions built from new IR")
        .def_static("sanitizer", &pass::Sanitizer,
                    "Create a sanitizer instrumentation pass (stateless; every check value travels in the records)");

    py::class_<PassInstrument, std::shared_ptr<PassInstrument>>(m, "PassInstrument",
                                                                "Abstract base class for pass instrumentation")
        .def("get_name", &PassInstrument::GetName, "Get the name of this instrument");

    py::class_<CallbackInstrument, PassInstrument, std::shared_ptr<CallbackInstrument>>(
        m, "CallbackInstrument", "Instrument that invokes callbacks before/after each pass")
        .def(py::init<CallbackInstrument::Callback, CallbackInstrument::Callback, std::string>(),
             py::arg("before_pass") = nullptr, py::arg("after_pass") = nullptr, py::arg("name") = "CallbackInstrument",
             "Create a callback instrument with optional before/after callbacks");

    py::class_<PassContext>(m, "PassContext",
                            "Context that holds instruments and pass configuration.\n\n"
                            "When active, Pass.__call__ will run the context's instruments\n"
                            "before/after each pass execution. Also controls automatic\n"
                            "verification level for PassPipeline.")
        .def(py::init<std::vector<PassInstrumentPtr>, VerificationLevel>(), py::arg("instruments"),
             py::arg("verification_level") = VerificationLevel::Basic,
             "Create a PassContext with instruments and optional verification level")
        .def("__enter__",
             [](PassContext& self) -> PassContext& {
                 self.EnterContext();
                 return self;
             })
        .def("__exit__", [](PassContext& self, const py::args&) { self.ExitContext(); })
        .def("get_verification_level", &PassContext::GetVerificationLevel,
             "Get the verification level for this context")
        .def("get_instruments", &PassContext::GetInstruments, "Get the instruments registered on this context")
        .def_static("current", &PassContext::Current, py::return_value_policy::reference,
                    "Get the currently active context, or None if no context is active");

    py::class_<PassPipeline>(m, "PassPipeline", "A pipeline of passes executed in sequence")
        .def(py::init<>(), "Create an empty pipeline")
        .def("add_pass", &PassPipeline::AddPass, py::arg("pass_obj"), "Add a pass to the pipeline")
        .def("run", &PassPipeline::Run, py::arg("program"), "Execute all passes in sequence")
        .def("get_pass_names", &PassPipeline::GetPassNames, "Get names of all passes");

    py::enum_<ssa::ErrorType>(m, "SSAErrorType", "SSA verification error types")
        .value("MULTIPLE_ASSIGNMENT", ssa::ErrorType::MULTIPLE_ASSIGNMENT, "Variable assigned more than once")
        .value("NAME_SHADOWING", ssa::ErrorType::NAME_SHADOWING, "Variable name shadows outer scope variable")
        .value("MISSING_YIELD", ssa::ErrorType::MISSING_YIELD, "ForStmt or IfStmt missing required YieldStmt")
        .value("USE_BEFORE_DEF", ssa::ErrorType::USE_BEFORE_DEF,
               "LogicalTensor used before its dominating definition (out of scope)")
        .value("ARITY_MISMATCH", ssa::ErrorType::ARITY_MISMATCH,
               "iter_args/return_vars/yield-or-continue value counts disagree");

    py::enum_<typecheck::ErrorType>(m, "TypeCheckErrorType", "Type checking error types")
        .value("TYPE_KIND_MISMATCH", typecheck::ErrorType::TYPE_KIND_MISMATCH, "Type kind mismatch")
        .value("DTYPE_MISMATCH", typecheck::ErrorType::DTYPE_MISMATCH, "Data type mismatch")
        .value("SHAPE_DIMENSION_MISMATCH", typecheck::ErrorType::SHAPE_DIMENSION_MISMATCH,
               "Shape dimension count mismatch")
        .value("SHAPE_VALUE_MISMATCH", typecheck::ErrorType::SHAPE_VALUE_MISMATCH, "Shape dimension value mismatch")
        .value("SIZE_MISMATCH", typecheck::ErrorType::SIZE_MISMATCH, "Vector size mismatch in control flow")
        .value("IF_CONDITION_MUST_BE_SCALAR", typecheck::ErrorType::IF_CONDITION_MUST_BE_SCALAR,
               "IfStmt condition must be ScalarType")
        .value("FOR_RANGE_MUST_BE_SCALAR", typecheck::ErrorType::FOR_RANGE_MUST_BE_SCALAR,
               "ForStmt range must be ScalarType");

    py::enum_<nested_call::ErrorType>(m, "NestedCallErrorType", "Nested call verification error types")
        .value("CALL_IN_CALL_ARGS", nested_call::ErrorType::CALL_IN_CALL_ARGS,
               "Call expression appears in call arguments")
        .value("CALL_IN_IF_CONDITION", nested_call::ErrorType::CALL_IN_IF_CONDITION,
               "Call expression appears in if condition")
        .value("CALL_IN_FOR_RANGE", nested_call::ErrorType::CALL_IN_FOR_RANGE,
               "Call expression appears in for range (start/stop/step)")
        .value("CALL_IN_BINARY_EXPR", nested_call::ErrorType::CALL_IN_BINARY_EXPR,
               "Call expression appears in binary expression operands")
        .value("CALL_IN_UNARY_EXPR", nested_call::ErrorType::CALL_IN_UNARY_EXPR,
               "Call expression appears in unary expression operand")
        .value("CALL_IN_WHILE_CONDITION", nested_call::ErrorType::CALL_IN_WHILE_CONDITION,
               "Call expression appears in while condition");

    py::enum_<DiagnosticSeverity>(m, "DiagnosticSeverity", "Severity level for diagnostics")
        .value("Error", DiagnosticSeverity::ERROR, "Error that must be fixed")
        .value("Warning", DiagnosticSeverity::WARNING, "Warning that should be reviewed");

    py::class_<Diagnostic>(m, "Diagnostic", "Single diagnostic message from verification")
        .def_readonly("severity", &Diagnostic::severity, "Severity level (Error or Warning)")
        .def_readonly("rule_name", &Diagnostic::ruleName, "Name of the verification rule")
        .def_readonly("error_code", &Diagnostic::errorCode, "Specific error code")
        .def_readonly("message", &Diagnostic::message, "Human-readable error message")
        .def_readonly("span", &Diagnostic::span, "Source location of the issue");

    py::class_<IRVerifier>(m, "IRVerifier",
                           "IR verification system that manages verification rules\n\n"
                           "IRVerifier collects verification rules and applies them to programs.\n"
                           "Rules can be enabled/disabled individually.")
        .def(py::init<>(), "Create an empty verifier with no rules")
        .def_static("create_default", &IRVerifier::CreateDefault,
                    "Create a verifier with default built-in rules (SSAVerify, TypeCheck)")
        .def("enable_rule", &IRVerifier::EnableRule, py::arg("name"), "Enable a previously disabled rule")
        .def("disable_rule", &IRVerifier::DisableRule, py::arg("name"), "Disable a rule")
        .def("is_rule_enabled", &IRVerifier::IsRuleEnabled, py::arg("name"), "Check if a rule is enabled")
        .def("verify", &IRVerifier::Verify, py::arg("program"),
             "Verify a program and collect diagnostics (does not throw)")
        .def("verify_or_throw", &IRVerifier::VerifyOrThrow, py::arg("program"),
             "Verify a program and throw VerificationError if errors are found")
        .def_static("generate_report", &IRVerifier::GenerateReport, py::arg("diagnostics"),
                    "Generate a formatted report from diagnostics");
}
} // namespace ir
} // namespace pypto
