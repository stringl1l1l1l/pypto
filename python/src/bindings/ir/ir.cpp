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

#include <any>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "bindings/ir/bindings.h"
#include "core/any_cast.h"
#include "core/common.h"
#include "core/dtype.h"
#include "core/error.h"
#include "ir/core.h"
#include "ir/expr.h"
#include "ir/op_attr_types.h"
#include "ir/function.h"
#include "ir/memref.h"
#include "ir/op_registry.h"
#include "ir/pipe.h"
#include "ir/program.h"
#include "ir/reflection/field_visitor.h"
#include "ir/scalar_expr.h"
#include "ir/scalar_expr_ops.h"
#include "ir/stmt.h"
#include "ir/transforms/io_text.h"
#include "ir/transforms/op_conversion_registry.h"
#include "ir/transforms/printer.h"
#include "ir/transforms/structural_comparison.h"
#include "ir/type.h"
#include "tilefwk/symbolic_scalar.h"
#include "interface/configs//config_manager_ng.h"

namespace py = pybind11;

using npu::tile_fwk::ConfigScope;
using npu::tile_fwk::ConfigScopePtr;
using npu::tile_fwk::SymbolicScalar;

namespace pypto {
namespace ir {
using pypto::ir::DataType;

// Helper from pypto_impl: get outer-IR compatibility pointer type (INT8-based)
std::shared_ptr<PtrType> GetOuterCompatPtrType()
{
    static auto ptr_type = std::make_shared<PtrType>(DataType::INT8);
    return ptr_type;
}

// Helper from pypto_impl: get program functions as a list (sorted by name)
std::vector<FunctionPtr> GetOuterProgramFunctions(const ProgramPtr& program)
{
    std::vector<FunctionPtr> functions;
    functions.reserve(program->functions_.size());
    for (const auto& entry : program->functions_) {
        functions.push_back(entry.second);
    }
    return functions;
}

template <typename T>
bool TryConvertAnyToPy(const std::any& value, py::object& out)
{
    if (value.type() != typeid(T)) {
        return false;
    }
    out = py::cast(AnyCastRef<T>(value, "converting to Python"));
    return true;
}

template <typename... Ts>
py::object AnyToPyObject(const std::any& value, const std::string& key)
{
    py::object out;
    if ((TryConvertAnyToPy<Ts>(value, out) || ...)) {
        return out;
    }
    throw pypto::ir::TypeError("Attribute '" + key + "' has unsupported type");
}

template <typename T>
std::vector<T> ConvertSequenceToVector(py::handle obj)
{
    auto seq = py::reinterpret_borrow<py::sequence>(obj);
    std::vector<T> ret;
    ret.reserve(py::len(seq));
    for (auto item : seq) {
        ret.push_back(py::cast<T>(item));
    }
    return ret;
}

// Helper to bind a single field using reflection
template <typename ClassType, typename PyClassType, typename FieldDesc>
void BindField(PyClassType& py_class, const FieldDesc& desc)
{
    py_class.def_readonly(desc.name, desc.fieldPtr);
}

// Helper to bind all fields from a tuple of field descriptors
template <typename ClassType, typename PyClassType, typename DescTuple, std::size_t... Is>
void BindFieldsImpl(PyClassType& py_class, const DescTuple& descriptors, std::index_sequence<Is...>)
{
    (BindField<ClassType>(py_class, std::get<Is>(descriptors)), ...);
}

// Main function to bind all fields using reflection
template <typename ClassType, typename PyClassType>
void BindFields(PyClassType& py_class)
{
    constexpr auto descriptors = ClassType::GetFieldDescriptors();
    constexpr auto num_fields = std::tuple_size_v<decltype(descriptors)>;
    BindFieldsImpl<ClassType>(py_class, descriptors, std::make_index_sequence<num_fields>{});
}

// Helper function to convert py::dict to vector<pair<string, any>>
std::vector<std::pair<std::string, std::any>> ConvertKwargsDict(const py::dict& kwargs_dict)
{
    std::vector<std::pair<std::string, std::any>> kwargs;
    for (auto item : kwargs_dict) {
        std::string key = py::cast<std::string>(item.first);

        // Try to cast to common types.
        // NOTE: Check enum-like values BEFORE plain int, and bool BEFORE int.
        if (py::isinstance<DataType>(item.second)) {
            kwargs.emplace_back(key, py::cast<DataType>(item.second));
        } else if (py::isinstance<MemorySpace>(item.second)) {
            kwargs.emplace_back(key, py::cast<MemorySpace>(item.second));
        } else if (py::hasattr(item.second, "value")) {
            // Generic pybind11 enum: extract underlying int value
            kwargs.emplace_back(key, py::cast<int>(item.second.attr("value")));
        } else if (py::isinstance<py::bool_>(item.second)) {
            kwargs.emplace_back(key, py::cast<bool>(item.second));
        } else if (py::isinstance<py::int_>(item.second)) {
            kwargs.emplace_back(key, py::cast<int>(item.second));
        } else if (py::isinstance<py::tuple>(item.second) || py::isinstance<py::list>(item.second)) {
            // Dispatch on element type: list[str] -> vector<string>, list[int] -> vector<int>.
            auto seq = py::cast<py::sequence>(item.second);
            if (py::len(seq) > 0 && py::isinstance<py::str>(seq[0])) {
                kwargs.emplace_back(key, ConvertSequenceToVector<std::string>(seq));
            } else {
                kwargs.emplace_back(key, ConvertSequenceToVector<int>(seq));
            }
        } else if (py::isinstance<py::str>(item.second)) {
            kwargs.emplace_back(key, py::cast<std::string>(item.second));
        } else if (py::isinstance<py::float_>(item.second)) {
            kwargs.emplace_back(key, py::cast<double>(item.second));
        } else {
            throw pypto::ir::TypeError("Unsupported kwarg type for key: " + key);
        }
    }
    return kwargs;
}

// Helper function to convert a single Python attribute to std::any (for TensorOpStmt)
std::any ConvertAttr(const py::object& attr)
{
    if (py::isinstance<DataType>(attr)) {
        return py::cast<DataType>(attr);
    } else if (py::isinstance<MemorySpace>(attr)) {
        return py::cast<MemorySpace>(attr);
    } else if (py::isinstance<TensorLayout>(attr)) {
        return py::cast<TensorLayout>(attr);
    } else if (py::isinstance<py::bool_>(attr)) {
        return py::cast<bool>(attr);
    } else if (py::isinstance<py::int_>(attr)) {
        return py::cast<int>(attr);
    } else if (py::isinstance<py::str>(attr)) {
        return py::cast<std::string>(attr);
    } else if (py::isinstance<py::float_>(attr)) {
        return py::cast<double>(attr);
    } else if (py::isinstance<SymbolicScalar>(attr)) {
        return py::cast<SymbolicScalar>(attr);
    } else if (py::isinstance<ExprPtr>(attr)) {
        return py::cast<ExprPtr>(attr);
    } else if (py::isinstance<ConfigScope>(attr)) {
        return py::cast<ConfigScopePtr>(attr);
    } else {
        throw TypeError("Unsupported attr type");
    }
}

std::any ConvertListAttr(const py::list& list)
{
    if (list.empty()) {
        return std::any();
    }
    auto item0 = list[0];
    if (py::isinstance<SymbolicScalar>(item0)) {
        std::vector<SymbolicScalar> ret;
        for (auto item : list) {
            ret.push_back(py::cast<SymbolicScalar>(item));
        }
        return ret;
    } else if (py::isinstance<ExprPtr>(item0)) {
        return ConvertSequenceToVector<ExprPtr>(list);
    } else if (py::isinstance<std::string>(item0)) {
        return ConvertSequenceToVector<std::string>(list);
    } else if (py::isinstance<py::int_>(item0)) {
        return ConvertSequenceToVector<int>(list);
    } else {
        throw TypeError("Unsupported list attr");
    }
}

std::vector<std::pair<std::string, std::any>> ConvertAttrDict(const py::dict& attrs)
{
    std::vector<std::pair<std::string, std::any>> kwargs;
    for (auto item : attrs) {
        std::string key = py::cast<std::string>(item.first);
        std::any value;
        if (py::isinstance<py::list>(item.second)) {
            value = ConvertListAttr(py::cast<py::list>(item.second));
        } else {
            value = ConvertAttr(py::cast<py::object>(item.second));
        }
        if (value.has_value()) {
            kwargs.emplace_back(key, value);
        }
    }
    return kwargs;
}

void BindDType(py::module_& ir)
{
    // DataType - registered in ir submodule (tile_fwk::DataType lives at root via BindEnum)
    py::class_<DataType>(ir, "DataType", "Data type representation for IR tensors and operations")
        .def_readonly_static("BOOL", &DataType::BOOL, "Boolean (true/false)")
        .def_readonly_static("INT4", &DataType::INT4, "4-bit signed integer")
        .def_readonly_static("INT8", &DataType::INT8, "8-bit signed integer")
        .def_readonly_static("INT16", &DataType::INT16, "16-bit signed integer")
        .def_readonly_static("INT32", &DataType::INT32, "32-bit signed integer")
        .def_readonly_static("INT64", &DataType::INT64, "64-bit signed integer")
        .def_readonly_static("UINT4", &DataType::UINT4, "4-bit unsigned integer")
        .def_readonly_static("UINT8", &DataType::UINT8, "8-bit unsigned integer")
        .def_readonly_static("UINT16", &DataType::UINT16, "16-bit unsigned integer")
        .def_readonly_static("UINT32", &DataType::UINT32, "32-bit unsigned integer")
        .def_readonly_static("UINT64", &DataType::UINT64, "64-bit unsigned integer")
        .def_readonly_static("FP4", &DataType::FP4, "4-bit floating point")
        .def_readonly_static("FP8E4M3FN", &DataType::FP8E4M3FN, "8-bit floating point (E4M3FN format)")
        .def_readonly_static("FP8E5M2", &DataType::FP8E5M2, "8-bit floating point (E5M2 format)")
        .def_readonly_static("FP8E8M0", &DataType::FP8E8M0, "8-bit floating point (E8M0 format)")
        .def_readonly_static("FP4E2M1", &DataType::FP4E2M1, "4-bit floating point (E2M1 format)")
        .def_readonly_static("FP4E1M2", &DataType::FP4E1M2, "4-bit floating point (E1M2 format)")
        .def_readonly_static("FP16", &DataType::FP16, "16-bit floating point (IEEE 754 half precision)")
        .def_readonly_static("FP32", &DataType::FP32, "32-bit floating point (IEEE 754 single precision)")
        .def_readonly_static("BF16", &DataType::BF16, "16-bit brain floating point")
        .def_readonly_static("FP64", &DataType::FP64, "64-bit floating point (IEEE 754 double precision)")
        .def_readonly_static("HF4", &DataType::HF4, "4-bit Hisilicon float")
        .def_readonly_static("HF8", &DataType::HF8, "8-bit Hisilicon float")
        .def_readonly_static("INDEX", &DataType::INDEX, "Machine-word sized integer type for index computations")
        .def_readonly_static("DEFAULT_CONST_FLOAT", &DataType::FP32,
                             "Default dtype for bare float constant literals (= FP32)")
        .def("get_bit", &DataType::GetBit, "Get the size in bits of this data type.")
        .def("to_string", &DataType::ToString, "Get a human-readable string name.")
        .def("to_c_type_string", &DataType::ToCTypeString, "Get C style type string for code generation.")
        .def("is_float", &DataType::IsFloat, "Check if floating point type.")
        .def("is_signed_int", &DataType::IsSignedInt, "Check if signed integer type.")
        .def("is_unsigned_int", &DataType::IsUnsignedInt, "Check if unsigned integer type.")
        .def("is_int", &DataType::IsInt, "Check if any integer type.")
        .def("code", &DataType::Code, "Get the underlying type code.")
        // Aliases from pypto_impl for backward compatibility
        .def("bits", &DataType::GetBit, "Get the size in bits (alias for get_bit).")
        .def("is_signed", &DataType::IsSignedInt, "Check if signed integer (alias for is_signed_int).")
        .def("is_unsigned", &DataType::IsUnsignedInt, "Check if unsigned integer (alias for is_unsigned_int).")
        .def("__int__", &DataType::Code, "Get the underlying type code as int.")
        .def("__eq__", &DataType::operator==, py::arg("other"))
        .def("__ne__", &DataType::operator!=, py::arg("other"))
        .def("__repr__", &DataType::ToString)
        .def("__str__", &DataType::ToString);
}

void BindSpan(py::module_& ir)
{
    // Span - value type, copy semantics
    py::class_<Span>(ir, "Span", "Source location information tracking file, line, and column positions")
        .def(py::init<std::string, int, int, int, int>(), py::arg("filename"), py::arg("begin_line"),
             py::arg("begin_column"), py::arg("end_line") = -1, py::arg("end_column") = -1, "Create a source span")
        .def("to_string", &Span::ToString, "Convert span to string representation")
        .def("is_valid", &Span::IsValid, "Check if the span has valid coordinates")
        .def_static("unknown", &Span::Unknown,
                    "Create an unknown/invalid span for cases where source location is unavailable",
                    py::return_value_policy::reference)
        .def(
            "is_unknown", [](const Span& self) { return self.IsUnknown(); }, "Check if the span is unknown")
        .def("__repr__", &Span::ToString)
        .def("__str__", &Span::ToString)
        .def_property_readonly("filename", &Span::Filename, "Source filename")
        .def_property_readonly("begin_line", &Span::BeginLine, "Beginning line (1-indexed)")
        .def_property_readonly("begin_column", &Span::BeginColumn, "Beginning column (1-indexed)")
        .def_property_readonly("end_line", &Span::EndLine, "Ending line (1-indexed)")
        .def_property_readonly("end_column", &Span::EndColumn, "Ending column (1-indexed)");
}

void BindOp(py::module_& ir)
{
    // Op - operation/function
    py::class_<Op, std::shared_ptr<Op>>(
        ir, "Op",
        "Represents callable operations in the IR. Stores the schema of allowed kwargs (key -> type "
        "mapping). Actual kwarg values are stored per-Call instance in Call.kwargs")
        .def(py::init<std::string>(), py::arg("name"), "Create an operation with the given name")
        .def_readonly("name", &Op::name_, "Operation name")
        .def("has_attr", &Op::HasAttr, py::arg("key"), "Check if a kwarg is registered in the schema")
        .def("get_attr_keys", &Op::GetAttrKeys, "Get all registered kwarg keys from the schema")
        .def_property_readonly(
            "pipe", [](const Op& self) -> std::optional<PipeType> { return self.GetPipe(); },
            "Pipeline type (optional)");
}

void BindTypeClass(py::module_& ir)
{
    // Type - abstract base, const shared_ptr
    auto type_class = py::class_<Type, std::shared_ptr<Type>>(ir, "Type", "Base class for type representations");
    BindFields<Type>(type_class);
    type_class.def("__str__", [](const TypePtr& self) { return TextDumpType(self); }, "IR text representation");
    type_class.def(
        "__eq__", [](const TypePtr& self, const TypePtr& other) { return structural_equal(self, other); },
        "Equality comparison");

    // UnknownType - const shared_ptr
    auto unknown_type_class = py::class_<UnknownType, Type, std::shared_ptr<UnknownType>>(
        ir, "UnknownType", "Unknown or unspecified type representation");
    unknown_type_class.def(py::init<>(), "Create an unknown type");
    unknown_type_class.def_static("get", []() { return GetUnknownType(); }, "Get the singleton UnknownType instance");
    BindFields<UnknownType>(unknown_type_class);

    // ScalarType - const shared_ptr
    auto scalar_type_class = py::class_<ScalarType, Type, std::shared_ptr<ScalarType>>(ir, "ScalarType",
                                                                                       "Scalar type representation");
    scalar_type_class.def(py::init<DataType>(), py::arg("dtype"), "Create a scalar type");
    BindFields<ScalarType>(scalar_type_class);

    // PtrType - const shared_ptr
    auto ptr_type_class = py::class_<PtrType, Type, std::shared_ptr<PtrType>>(
        ir, "PtrType", "Pointer type representation (!pto.ptr<dtype>)");
    ptr_type_class.def(py::init<DataType>(), py::arg("dtype"), "Create a pointer type");
    ptr_type_class.def(py::init([]() { return GetOuterCompatPtrType(); }),
                       "Create the outer-IR compatibility pointer type (INT8-based)");
    ptr_type_class.def_static("get", &GetOuterCompatPtrType, "Get the outer-IR compatibility pointer type singleton");
    BindFields<PtrType>(ptr_type_class);

    // IRNode - abstract base, const shared_ptr
    auto irnode_class = py::class_<IRNode, std::shared_ptr<IRNode>>(ir, "IRNode", "Base class for all IR nodes");
    BindFields<IRNode>(irnode_class);
    irnode_class
        .def(
            "same_as", [](const IRNodePtr& self, const IRNodePtr& other) { return self == other; }, py::arg("other"),
            "Check if this IR node is the same as another IR node.")
        .def(
            "__str__", [](const IRNodePtr& self) { return TextDump(self); }, "IR text representation")
        .def(
            "as_python", [](const IRNodePtr& self, const std::string& prefix) { return PythonPrint(self, prefix); },
            py::arg("prefix") = "ir",
            "Convert to Python-style string representation.\n\n"
            "Args:\n"
            "    prefix: Module prefix (default 'ir' for 'import pypto_pro.ir as ir')");

    // Expr - abstract base, const shared_ptr
    auto expr_class = py::class_<Expr, IRNode, std::shared_ptr<Expr>>(ir, "Expr", "Base class for all expressions");
    BindFields<Expr>(expr_class);

    // ShapedType - abstract base for types with shape and optional memref
    auto shaped_type_class = py::class_<ShapedType, Type, std::shared_ptr<ShapedType>>(
        ir, "ShapedType", "Base class for shaped types (tensors and tiles)");
    BindFields<ShapedType>(shaped_type_class);
    shaped_type_class.def(
        "shares_memref_with",
        [](const ShapedTypePtr& self, const ShapedTypePtr& other) {
            if (!self->memref_.has_value() || !other->memref_.has_value()) {
                return false;
            }
            return self->memref_.value().get() == other->memref_.value().get();
        },
        py::arg("other"), "Check if this ShapedType shares the same MemRef object with another ShapedType");

    // TensorLayout enum - must be before TensorView and TensorType
    py::enum_<TensorLayout>(ir, "TensorLayout", "Tensor layout enumeration")
        .value("ND", TensorLayout::ND, "ND layout")
        .value("DN", TensorLayout::DN, "DN layout")
        .value("NZ", TensorLayout::NZ, "NZ layout")
        .value("ZN", TensorLayout::ZN, "ZN layout (Tile only)")
        .value("NN", TensorLayout::NN, "NN layout (Tile only)")
        .value("ZZ", TensorLayout::ZZ, "ZZ layout (Tile only)")
        .export_values();

    // TensorView - struct for tensor view information - must be before TensorType
    py::class_<TensorView>(ir, "TensorView", "Tensor view representation with stride and layout")
        .def(py::init<>(), "Create an empty tensor view")
        .def(py::init<const std::vector<ExprPtr>&, TensorLayout>(), py::arg("stride"), py::arg("layout"),
             "Create a tensor view with stride and layout")
        .def(py::init<const std::vector<ExprPtr>&, const std::vector<ExprPtr>&, TensorLayout>(), py::arg("valid_shape"),
             py::arg("stride"), py::arg("layout"), "Create a tensor view with valid_shape, stride and layout")
        .def_readwrite("valid_shape", &TensorView::validShape, "Valid shape dimensions")
        .def_readwrite("stride", &TensorView::stride, "Stride for each dimension")
        .def_readwrite("layout", &TensorView::layout, "Tensor layout type")
        .def_readwrite("ptr", &TensorView::ptr,
                       "Source pointer ExprPtr (set for ptr.make_tensor-created views; None otherwise).");

    // TensorType - const shared_ptr
    auto tensor_type_class = py::class_<TensorType, ShapedType, std::shared_ptr<TensorType>>(
        ir, "TensorType", "Tensor type representation");
    tensor_type_class.def(py::init<const std::vector<ExprPtr>&, DataType, std::optional<MemRefPtr>>(), py::arg("shape"),
                          py::arg("dtype"), py::arg("memref") = py::none(), "Create a tensor type");
    tensor_type_class.def(py::init<const std::vector<int64_t>&, DataType, std::optional<MemRefPtr>>(), py::arg("shape"),
                          py::arg("dtype"), py::arg("memref") = py::none(), "Create a tensor type");
    tensor_type_class.def(
        py::init<const std::vector<ExprPtr>&, DataType, std::optional<MemRefPtr>, std::optional<TensorView>>(),
        py::arg("shape"), py::arg("dtype"), py::arg("memref") = py::none(), py::arg("tensor_view") = py::none(),
        "Create a tensor type with optional memory reference and tensor view");
    tensor_type_class.def(
        py::init<const std::vector<int64_t>&, DataType, std::optional<MemRefPtr>, std::optional<TensorView>>(),
        py::arg("shape"), py::arg("dtype"), py::arg("memref") = py::none(), py::arg("tensor_view") = py::none(),
        "Create a tensor type with constant shape, optional memory reference and tensor view");
    BindFields<TensorType>(tensor_type_class);

    // TileType - const shared_ptr
    auto tile_type_class = py::class_<TileType, ShapedType, std::shared_ptr<TileType>>(
        ir, "TileType", "Tile type representation (multi-dimensional tensor)");
    tile_type_class.def(py::init<const std::vector<ExprPtr>&, DataType, std::optional<MemRefPtr>,
                                 std::optional<TileView>, std::optional<HardwareInfo>>(),
                        py::arg("shape"), py::arg("dtype"), py::arg("memref") = py::none(),
                        py::arg("tile_view") = py::none(), py::arg("hardware_info") = py::none(),
                        "Create a tile type (supports multi-dimensional tensors; code generation has constraints)");
    tile_type_class.def(py::init<const std::vector<int64_t>&, DataType, std::optional<MemRefPtr>,
                                 std::optional<TileView>, std::optional<HardwareInfo>>(),
                        py::arg("shape"), py::arg("dtype"), py::arg("memref") = py::none(),
                        py::arg("tile_view") = py::none(), py::arg("hardware_info") = py::none(),
                        "Create a tile type (supports multi-dimensional tensors; code generation has constraints)");
    BindFields<TileType>(tile_type_class);

    // TupleType - const shared_ptr
    auto tuple_type_class = py::class_<TupleType, Type, std::shared_ptr<TupleType>>(
        ir, "TupleType", "Tuple type representation (contains multiple types)");
    tuple_type_class.def(py::init<const std::vector<TypePtr>&>(), py::arg("types"),
                         "Create a tuple type from a list of types");
    BindFields<TupleType>(tuple_type_class);

    py::enum_<TokenKind>(ir, "TokenKind", "Token semantic kind")
        .value("NORMAL", TokenKind::NORMAL, "Ordinary dependency token")
        .value("READ", TokenKind::READ, "Tensor graph read token")
        .value("WRITE", TokenKind::WRITE, "Tensor graph write token");

    // TokenType - const shared_ptr
    auto token_type_class = py::class_<TokenType, Type, std::shared_ptr<TokenType>>(ir, "TokenType",
                                                                                    "Opaque token type");
    token_type_class.def(py::init<TokenKind>(), py::arg("kind") = TokenKind::NORMAL, "Create a token type")
        .def_static("get", &GetTokenType, py::arg("kind") = TokenKind::NORMAL, "Get the token type");
    BindFields<TokenType>(token_type_class);

    // NoneType - const shared_ptr
    py::class_<NoneType, Type, std::shared_ptr<NoneType>>(ir, "NoneType", "None (void-like) type")
        .def(py::init<>(), "Create a none type")
        .def_static("get", GetNoneType, "Get the none type");

    // LogicalTensorType - const shared_ptr
    py::class_<LogicalTensorType, Type, std::shared_ptr<LogicalTensorType>>(ir, "LogicalTensorType",
                                                                            "Logical tensor type")
        .def(py::init<>(), "Create a logical tensor type");

    // MemorySpace enum
    py::enum_<MemorySpace>(ir, "MemorySpace", "Memory space enumeration")
        .value("DDR", MemorySpace::DDR, "DDR memory (off-chip)")
        .value("Vec", MemorySpace::Vec, "Vector/unified buffer (on-chip)")
        .value("Mat", MemorySpace::Mat, "Matrix/L1 buffer")
        .value("Left", MemorySpace::Left, "Left matrix operand buffer")
        .value("Right", MemorySpace::Right, "Right matrix operand buffer")
        .value("Scaling", MemorySpace::Scaling, "Scaling/FBuffer buffer")
        .value("Acc", MemorySpace::Acc, "Accumulator buffer")
        .value("Bias", MemorySpace::Bias, "Bias buffer")
        .value("ScaleLeft", MemorySpace::ScaleLeft, "MX scale buffer in L0A")
        .value("ScaleRight", MemorySpace::ScaleRight, "MX scale buffer in L0B")
        .export_values();

    ir.attr("Mem") = ir.attr("MemorySpace");

    // PipeType enum
    py::enum_<PipeType>(ir, "PipeType", py::arithmetic(), "Pipeline type enumeration")
        .value("MTE1", PipeType::MTE1, "Memory Transfer Engine 1")
        .value("MTE2", PipeType::MTE2, "Memory Transfer Engine 2")
        .value("MTE3", PipeType::MTE3, "Memory Transfer Engine 3")
        .value("M", PipeType::M, "Matrix Unit")
        .value("V", PipeType::V, "Vector Unit")
        .value("S", PipeType::S, "Scalar Unit")
        .value("FIX", PipeType::FIX, "Fix Pipe")
        .value("ALL", PipeType::ALL, "All Pipes")
        .export_values();

    // CoreType enum
    py::enum_<CoreType>(ir, "CoreType", py::arithmetic(), "Core type enumeration")
        .value("VECTOR", CoreType::VECTOR, "Vector Core")
        .value("CUBE", CoreType::CUBE, "Cube Core")
        .export_values();

    // TileLayout enum - must be before TileView
    py::enum_<TileLayout>(ir, "TileLayout", "Tile layout enumeration")
        .value("none_box", TileLayout::none_box, "No layout constraint")
        .value("row_major", TileLayout::row_major, "Row-major layout")
        .value("col_major", TileLayout::col_major, "Column-major layout")
        .export_values();

    // TilePad enum - must be before TileView
    py::enum_<TilePad>(ir, "TilePad", "Tile pad mode enumeration")
        .value("null", TilePad::null, "No padding")
        .value("zero", TilePad::zero, "Zero padding")
        .value("max", TilePad::max, "Max value padding")
        .value("min", TilePad::min, "Min value padding")
        .export_values();

    // CompactMode enum - must be before TileView
    py::enum_<CompactMode>(ir, "CompactMode", "Compact mode for tile buffer")
        .value("null", CompactMode::null, "No compact mode")
        .value("normal", CompactMode::normal, "Normal compact mode")
        .value("row_plus_one", CompactMode::row_plus_one, "Row plus one compact mode")
        .export_values();

    // -----------------------------------------------------------------------
    // Block-level API parameter enums
    // -----------------------------------------------------------------------

    py::enum_<ir::ReluPreMode>(ir, "ReluPreMode", "ReLU pre-processing mode for store/move")
        .value("NormalRelu", ir::ReluPreMode::NormalRelu, "Normal ReLU fusion");

    py::enum_<ir::AtomicType>(ir, "AtomicType", "Atomic write mode for store")
        .value("AtomicNone", ir::AtomicType::AtomicNone, "Overwrite (default)")
        .value("AtomicAdd", ir::AtomicType::AtomicAdd, "Atomic accumulate");

    py::enum_<ir::STPhase>(ir, "STPhase", "Fixpipe drain phase for store")
        .value("Unspecified", ir::STPhase::Unspecified, "Unspecified phase")
        .value("Partial", ir::STPhase::Partial, "Partial drain")
        .value("Final", ir::STPhase::Final, "Final drain");

    py::enum_<ir::AccPhase>(ir, "AccPhase", "Fixpipe drain phase for matmul/matmul_acc")
        .value("Unspecified", ir::AccPhase::Unspecified, "Unspecified phase")
        .value("Partial", ir::AccPhase::Partial, "Partial drain")
        .value("Final", ir::AccPhase::Final, "Final drain");

    py::enum_<ir::AccToVecMode>(ir, "AccToVecMode", "Accumulator to vector transfer mode")
        .value("SingleModeVec0", ir::AccToVecMode::SingleModeVec0, "Single vector sub-core 0")
        .value("SingleModeVec1", ir::AccToVecMode::SingleModeVec1, "Single vector sub-core 1")
        .value("DualModeSplitM", ir::AccToVecMode::DualModeSplitM, "Dual split along M dimension")
        .value("DualModeSplitN", ir::AccToVecMode::DualModeSplitN, "Dual split along N dimension");

    py::enum_<ir::RoundMode>(ir, "RoundMode", "Rounding mode for cast operations")
        .value("CAST_NONE", ir::RoundMode::CAST_NONE, "No rounding (0)")
        .value("CAST_RINT", ir::RoundMode::CAST_RINT, "Round to nearest integer (1)")
        .value("CAST_ROUND", ir::RoundMode::CAST_ROUND, "Round half away from zero (2)")
        .value("CAST_FLOOR", ir::RoundMode::CAST_FLOOR, "Round toward negative infinity (3)")
        .value("CAST_CEIL", ir::RoundMode::CAST_CEIL, "Round toward positive infinity (4)")
        .value("CAST_TRUNC", ir::RoundMode::CAST_TRUNC, "Truncate toward zero (5)")
        .value("CAST_ODD", ir::RoundMode::CAST_ODD, "Round to odd (6)");

    py::enum_<ir::QuantMode>(ir, "QuantMode", "Quantization mode")
        .value("SYM", ir::QuantMode::SYM, "Symmetric quantization")
        .value("ASYM", ir::QuantMode::ASYM, "Asymmetric quantization");

    py::enum_<ir::CrossCoreSyncMode>(ir, "CrossCoreSyncMode", "Cross-core synchronization mode")
        .value("INTER_BLOCK", ir::CrossCoreSyncMode::INTER_BLOCK, "Inter-core sync (mode 0)")
        .value("INTER_SUBBLOCK", ir::CrossCoreSyncMode::INTER_SUBBLOCK, "AIV-to-AIV sync (mode 1)")
        .value("INTRA_BLOCK", ir::CrossCoreSyncMode::INTRA_BLOCK, "AIC<->AIV sync, both subcores (mode 2)")
        .value("UNICAST_BLOCK", ir::CrossCoreSyncMode::UNICAST_BLOCK, "AIC<->AIV sync, one subcore (mode 3)");

    py::enum_<ir::SyncCoreType>(ir, "SyncCoreType", "Core type for sync_all")
        .value("AIV_ONLY", ir::SyncCoreType::AIV_ONLY, "Sync vector cores only")
        .value("AIC_ONLY", ir::SyncCoreType::AIC_ONLY, "Sync cube cores only")
        .value("MIX", ir::SyncCoreType::MIX, "Sync both AIC and AIV cores");

    py::enum_<ir::CacheLine>(ir, "CacheLine", "Cache line scope for DCCI")
        .value("SINGLE_CACHE_LINE", ir::CacheLine::SINGLE_CACHE_LINE, "Single cache line")
        .value("ENTIRE_DATA_CACHE", ir::CacheLine::ENTIRE_DATA_CACHE, "Entire data cache");

    py::enum_<ir::DcciDst>(ir, "DcciDst", "DCCI destination")
        .value("AUTO", ir::DcciDst::AUTO, "Auto: tensor->CACHELINE_OUT, tile->CACHELINE_UB")
        .value("CACHELINE_OUT", ir::DcciDst::CACHELINE_OUT, "Cache line out")
        .value("CACHELINE_UB", ir::DcciDst::CACHELINE_UB, "Cache line UB")
        .value("CACHELINE_ALL", ir::DcciDst::CACHELINE_ALL, "Cache line all")
        .value("CACHELINE_ATOMIC", ir::DcciDst::CACHELINE_ATOMIC, "Cache line atomic");

    // --- VF (Vector Function) API enumerations ---

    py::enum_<ir::MaskPattern>(ir, "MaskPattern", R"pbdoc(
Mask pattern for vf.create_mask / mask_reg.

Members:
    ALL           all elements valid
    ALLF          all elements invalid
    VL1           lowest 1 element valid
    VL2           lowest 2 elements valid
    VL3           lowest 3 elements valid
    VL4           lowest 4 elements valid
    VL8           lowest 8 elements valid
    VL16          lowest 16 elements valid
    VL32          lowest 32 elements valid
    VL64          lowest 64 elements valid
    VL128         lowest 128 elements valid
    M3            every 3rd element valid
    M4            every 4th element valid
    H             lower half valid
    Q             lower quarter valid)pbdoc")
        .value("ALL", ir::MaskPattern::ALL)
        .value("ALLF", ir::MaskPattern::ALLF)
        .value("VL1", ir::MaskPattern::VL1)
        .value("VL2", ir::MaskPattern::VL2)
        .value("VL3", ir::MaskPattern::VL3)
        .value("VL4", ir::MaskPattern::VL4)
        .value("VL8", ir::MaskPattern::VL8)
        .value("VL16", ir::MaskPattern::VL16)
        .value("VL32", ir::MaskPattern::VL32)
        .value("VL64", ir::MaskPattern::VL64)
        .value("VL128", ir::MaskPattern::VL128)
        .value("M3", ir::MaskPattern::M3)
        .value("M4", ir::MaskPattern::M4)
        .value("H", ir::MaskPattern::H)
        .value("Q", ir::MaskPattern::Q);

    py::enum_<ir::MergeMode>(ir, "MergeMode", R"pbdoc(
Mask merge mode for VF ops (handling of mask-inactive elements in dst).

Members:
    ZEROING       zero masked-out positions
    MERGING       retain original dst value at masked-out positions)pbdoc")
        .value("ZEROING", ir::MergeMode::ZEROING)
        .value("MERGING", ir::MergeMode::MERGING);

    py::enum_<ir::ReduceMode>(ir, "ReduceMode", "Reduction mode for vf.reduce")
        .value("SUM", ir::ReduceMode::SUM)
        .value("MAX", ir::ReduceMode::MAX)
        .value("MIN", ir::ReduceMode::MIN);

    py::enum_<ir::CompareMode>(ir, "CompareMode", "Comparison mode for vf.compare")
        .value("EQ", ir::CompareMode::EQ)
        .value("NE", ir::CompareMode::NE)
        .value("LT", ir::CompareMode::LT)
        .value("LE", ir::CompareMode::LE)
        .value("GT", ir::CompareMode::GT)
        .value("GE", ir::CompareMode::GE);

    py::enum_<ir::CmpMode>(ir, "CmpMode", "Comparison mode for block.cmp/cmps (isa TCMP/TCMPS)")
        .value("EQ", ir::CmpMode::EQ)
        .value("NE", ir::CmpMode::NE)
        .value("LT", ir::CmpMode::LT)
        .value("LE", ir::CmpMode::LE)
        .value("GT", ir::CmpMode::GT)
        .value("GE", ir::CmpMode::GE);

    py::enum_<ir::DuplicatePos>(ir, "DuplicatePos", R"pbdoc(
Position selector for vf.full (Tensor/broadcast mode).

Members:
    LOWEST        broadcast the lowest-indexed element (default)
    HIGHEST       broadcast the highest-indexed element)pbdoc")
        .value("LOWEST", ir::DuplicatePos::LOWEST)
        .value("HIGHEST", ir::DuplicatePos::HIGHEST);

    py::enum_<ir::CastLayout>(ir, "CastLayout", R"pbdoc(
Destination half-select for vf.astype, exp_sub, muls_cast.

Members:
    ZERO          write result to even half (PART_EVEN)
    ONE           write result to odd half (PART_ODD)
    TWO           third half
    THREE         fourth half)pbdoc")
        .value("ZERO", ir::CastLayout::ZERO)
        .value("ONE", ir::CastLayout::ONE)
        .value("TWO", ir::CastLayout::TWO)
        .value("THREE", ir::CastLayout::THREE);

    py::enum_<ir::VFRoundMode>(ir, "VFRoundMode", R"pbdoc(
Rounding mode for VF type conversion (vcvt).

Members:
    CAST_ROUND    round to nearest, ties away from zero
    CAST_RINT     default rounding (round to nearest even)
    CAST_FLOOR    round down (floor)
    CAST_CEIL     round up (ceil)
    CAST_TRUNC    round toward zero (truncate)
    CAST_ODD      von Neumann rounding (nearest odd)
    CAST_HYBRID   hybrid rounding (Ascend 950PR/DT only))pbdoc")
        .value("CAST_ROUND", ir::VFRoundMode::CAST_ROUND)
        .value("CAST_RINT", ir::VFRoundMode::CAST_RINT)
        .value("CAST_FLOOR", ir::VFRoundMode::CAST_FLOOR)
        .value("CAST_CEIL", ir::VFRoundMode::CAST_CEIL)
        .value("CAST_TRUNC", ir::VFRoundMode::CAST_TRUNC)
        .value("CAST_ODD", ir::VFRoundMode::CAST_ODD)
        .value("CAST_HYBRID", ir::VFRoundMode::CAST_HYBRID);

    py::enum_<ir::SaturateMode>(ir, "SaturateMode", R"pbdoc(
Saturation mode for narrowing type conversions.

Members:
    OFF           no saturation; overflow wraps/truncates (default)
    ON            saturate: clamp to target type min/max)pbdoc")
        .value("OFF", ir::SaturateMode::OFF)
        .value("ON", ir::SaturateMode::ON);

    py::enum_<ir::SaturationFlagMode>(ir, "SaturationFlagMode", R"pbdoc(
Saturation flag mode for CTRL register bit control.

Members:
    FLOAT         float compute/convert saturation (CTRL bit 48)
    FLOAT8        float8 compute saturation (CTRL bit 50)
    INT           int compute saturation (CTRL bit 53)
    CAST          float->int / int->int convert saturation (CTRL bit 59))pbdoc")
        .value("FLOAT", ir::SaturationFlagMode::FLOAT)
        .value("FLOAT8", ir::SaturationFlagMode::FLOAT8)
        .value("INT", ir::SaturationFlagMode::INT)
        .value("CAST", ir::SaturationFlagMode::CAST);

    py::enum_<ir::BinType>(ir, "BinType", R"pbdoc(
Histogram bin range for vf.histograms.

Members:
    BIN0          lower half bin range (indices [0-127], default)
    BIN1          upper half bin range (indices [128-255]))pbdoc")
        .value("BIN0", ir::BinType::BIN0)
        .value("BIN1", ir::BinType::BIN1);

    py::enum_<ir::HistType>(ir, "HistType", R"pbdoc(
Histogram statistical mode for vf.histograms.

Members:
    ACCUMULATE    accumulate counts onto existing dst values (cumulative, default)
    FREQUENCY     frequency count (per-bin occurrence count of each exact value))pbdoc")
        .value("ACCUMULATE", ir::HistType::ACCUMULATE)
        .value("FREQUENCY", ir::HistType::FREQUENCY);

    py::enum_<ir::SqueezeMode>(ir, "SqueezeMode", R"pbdoc(
Gather mode for vf.squeeze.

Members:
    NO_STORE_REG  do not store effective element byte count into AR SPR (default)
    STORE_REG     store total byte count of valid elements into AR SPR)pbdoc")
        .value("STORE_REG", ir::SqueezeMode::STORE_REG)
        .value("NO_STORE_REG", ir::SqueezeMode::NO_STORE_REG);

    py::enum_<ir::PackPart>(ir, "PackPart", R"pbdoc(
Half selector for vf.pack / vf.unpack.

Members:
    LOWER         lower half (default)
    UPPER         upper half (RegTraitNumTwo supports LOWER only))pbdoc")
        .value("LOWER", ir::PackPart::LOWER)
        .value("UPPER", ir::PackPart::UPPER);

    py::enum_<ir::MaskWidth>(ir, "MaskWidth", R"pbdoc(
SPR mask bit-width for vf.get_mask_spr.

Members:
    B32           read 64-bit MASK0, expand each bit to 4 bits (movp_b32, default)
    B16           read 128-bit {MASK1,MASK0}, expand each bit to 2 bits (movp_b16))pbdoc")
        .value("B32", ir::MaskWidth::B32)
        .value("B16", ir::MaskWidth::B16);

    py::enum_<ir::LoadDist>(ir, "LoadDist", R"pbdoc(
Data distribution pattern for vf.load_align.

RegTensor dst:
    NORM          normal element load (default)
    BRC           broadcast single element to entire register
    BRC_B8        broadcast by B8 granularity
    BRC_B16       broadcast by B16 granularity
    BRC_B32       broadcast by B32 granularity
    US            upsample (each bit repeated twice)
    US_B8         upsample by B8 granularity
    US_B16        upsample by B16 granularity
    DS            downsample (every other bit discarded)
    DS_B8         downsample by B8 granularity
    DS_B16        downsample by B16 granularity
    UNPK          unpack
    UNPK_B8       unpack by B8 granularity
    UNPK_B16      unpack by B16 granularity
    UNPK_B32      unpack by B32 granularity
    UNPK4         4-element unpack
    BLK           block copy
    E2B           B16->B32 expand
    E2B_B16       expand by B16 granularity
    E2B_B32       expand by B32 granularity
    DINTLV_B8     de-interleave by B8 (split even/odd regs)
    DINTLV_B16    de-interleave by B16
    DINTLV_B32    de-interleave by B32
MaskReg dst:
    NORM          normal mode, moves VL/8
    US            upsample
    DS            downsample)pbdoc")
        .value("NORM", ir::LoadDist::NORM)
        .value("BRC", ir::LoadDist::BRC)
        .value("BRC_B8", ir::LoadDist::BRC_B8)
        .value("BRC_B16", ir::LoadDist::BRC_B16)
        .value("BRC_B32", ir::LoadDist::BRC_B32)
        .value("US", ir::LoadDist::US)
        .value("US_B8", ir::LoadDist::US_B8)
        .value("US_B16", ir::LoadDist::US_B16)
        .value("DS", ir::LoadDist::DS)
        .value("DS_B8", ir::LoadDist::DS_B8)
        .value("DS_B16", ir::LoadDist::DS_B16)
        .value("UNPK", ir::LoadDist::UNPK)
        .value("UNPK_B8", ir::LoadDist::UNPK_B8)
        .value("UNPK_B16", ir::LoadDist::UNPK_B16)
        .value("UNPK_B32", ir::LoadDist::UNPK_B32)
        .value("UNPK4", ir::LoadDist::UNPK4)
        .value("BLK", ir::LoadDist::BLK)
        .value("E2B", ir::LoadDist::E2B)
        .value("E2B_B16", ir::LoadDist::E2B_B16)
        .value("E2B_B32", ir::LoadDist::E2B_B32)
        .value("DINTLV_B8", ir::LoadDist::DINTLV_B8)
        .value("DINTLV_B16", ir::LoadDist::DINTLV_B16)
        .value("DINTLV_B32", ir::LoadDist::DINTLV_B32);

    py::enum_<ir::StoreDist>(ir, "StoreDist", R"pbdoc(
Data distribution pattern for vf.store_align.

Coarse names (NORM/FIRST_ELEMENT/PACK/PACK4/INTLV) auto-select the
granularity variant from the source dtype. Width-qualified names select
the granularity explicitly (mirrors AscendC Reg::StoreDist).

Members:
    NORM               normal aligned store (default); auto B8/B16/B32 by dtype
    NORM_B8            normal store with B8 granularity
    NORM_B16           normal store with B16 granularity
    NORM_B32           normal store with B32 granularity
    FIRST_ELEMENT      store lane 0 only (first element); auto B8/B16/B32 by dtype
    FIRST_ELEMENT_B8   store lane 0 only, B8 granularity
    FIRST_ELEMENT_B16  store lane 0 only, B16 granularity
    FIRST_ELEMENT_B32  store lane 0 only, B32 granularity
    PACK               compressed store (pack lower bits); auto B16/B32/B64 by dtype
    PACK_B16           compressed store, B16 granularity
    PACK_B32           compressed store, B32 granularity
    PACK_B64           compressed store, B64 granularity
    PACK4              4-element compressed store (B32 granularity)
    PACK4_B32          4-element compressed store, B32 granularity
    INTLV              interleaved store; auto B8/B16/B32 by dtype (two source regs)
    INTLV_B8           interleaved store at B8 granularity (two source regs)
    INTLV_B16          interleaved store at B16 granularity (two source regs)
    INTLV_B32          interleaved store at B32 granularity (two source regs))pbdoc")
        .value("NORM", ir::StoreDist::NORM)
        .value("NORM_B16", ir::StoreDist::NORM_B16)
        .value("FIRST_ELEMENT", ir::StoreDist::FIRST_ELEMENT)
        .value("PACK", ir::StoreDist::PACK)
        .value("PACK4", ir::StoreDist::PACK4)
        .value("INTLV", ir::StoreDist::INTLV)
        .value("INTLV_B32", ir::StoreDist::INTLV_B32)
        .value("NORM_B8", ir::StoreDist::NORM_B8)
        .value("NORM_B32", ir::StoreDist::NORM_B32)
        .value("FIRST_ELEMENT_B8", ir::StoreDist::FIRST_ELEMENT_B8)
        .value("FIRST_ELEMENT_B16", ir::StoreDist::FIRST_ELEMENT_B16)
        .value("FIRST_ELEMENT_B32", ir::StoreDist::FIRST_ELEMENT_B32)
        .value("PACK_B16", ir::StoreDist::PACK_B16)
        .value("PACK_B32", ir::StoreDist::PACK_B32)
        .value("PACK_B64", ir::StoreDist::PACK_B64)
        .value("PACK4_B32", ir::StoreDist::PACK4_B32)
        .value("INTLV_B8", ir::StoreDist::INTLV_B8)
        .value("INTLV_B16", ir::StoreDist::INTLV_B16);

    py::enum_<ir::DataCopyMode>(ir, "DataCopyMode", R"pbdoc(
Data copy granularity for vf.load_align / store_align / gather.

Members:
    NORM              normal element-by-element copy (default)
    DATA_BLOCK_LOAD   for vf.gather: gather by 32B DataBlock
    DATA_BLOCK_COPY   non-contiguous DataBlock copy (block_stride-based))pbdoc")
        .value("NORM", ir::DataCopyMode::NORM)
        .value("DATA_BLOCK_LOAD", ir::DataCopyMode::DATA_BLOCK_LOAD)
        .value("DATA_BLOCK_COPY", ir::DataCopyMode::DATA_BLOCK_COPY);

    py::enum_<ir::IndexOrder>(ir, "IndexOrder", R"pbdoc(
Index sequence direction for vf.arange.

Members:
    INCREASE_ORDER  increasing sequence: dst[i] = start + i (default)
    DECREASE_ORDER  decreasing sequence: dst[i] = start - i)pbdoc")
        .value("INCREASE_ORDER", ir::IndexOrder::INCREASE_ORDER)
        .value("DECREASE_ORDER", ir::IndexOrder::DECREASE_ORDER);

    py::enum_<ir::MemBarMode>(ir, "MemBarMode", R"pbdoc(
Memory barrier src->dst ordering constraint for vf.mem_bar.

Members:
    VST_VLD       Vector Store -> Vector Load (RAW, default)
    VLD_VST       Vector Load -> Vector Store (WAR)
    VST_VST       Vector Store -> Vector Store (WAW)
    VST_LD        Vector Store -> Scalar Load
    VST_ST        Vector Store -> Scalar Store
    VLD_ST        Vector Load -> Scalar Store
    ST_VLD        Scalar Store -> Vector Load
    ST_VST        Scalar Store -> Vector Store
    LD_VST        Scalar Load -> Vector Store
    VV_ALL        all Vector <-> all Vector
    VS_ALL        all Vector <-> all Scalar
    SV_ALL        all Scalar <-> all Vector)pbdoc")
        .value("VST_VLD", ir::MemBarMode::VST_VLD)
        .value("VLD_VST", ir::MemBarMode::VLD_VST)
        .value("VST_VST", ir::MemBarMode::VST_VST)
        .value("VST_LD", ir::MemBarMode::VST_LD)
        .value("VST_ST", ir::MemBarMode::VST_ST)
        .value("VLD_ST", ir::MemBarMode::VLD_ST)
        .value("ST_VLD", ir::MemBarMode::ST_VLD)
        .value("ST_VST", ir::MemBarMode::ST_VST)
        .value("LD_VST", ir::MemBarMode::LD_VST)
        .value("VV_ALL", ir::MemBarMode::VV_ALL)
        .value("VS_ALL", ir::MemBarMode::VS_ALL)
        .value("SV_ALL", ir::MemBarMode::SV_ALL);

    // HardwareInfo - struct for hardware-specific tile information
    py::class_<HardwareInfo>(ir, "HardwareInfo", "Hardware-specific tile information (layout, fractal, pad, compact)")
        .def(py::init<>(), "Create a default hardware info")
        .def(py::init<TileLayout, TileLayout, uint64_t, TilePad, CompactMode>(),
             py::arg("blayout") = TileLayout::row_major, py::arg("slayout") = TileLayout::none_box,
             py::arg("fractal") = HardwareInfo::kDefaultFractal, py::arg("pad") = TilePad::null,
             py::arg("compact") = CompactMode::null,
             "Create hardware info with blayout, slayout, fractal, pad, and compact")
        .def_readwrite("blayout", &HardwareInfo::blayout, "Block layout")
        .def_readwrite("slayout", &HardwareInfo::slayout, "Scatter layout")
        .def_readwrite("fractal", &HardwareInfo::fractal, "Fractal size")
        .def_readwrite("pad", &HardwareInfo::pad, "Pad mode")
        .def_readwrite("compact", &HardwareInfo::compact, "Compact mode");

    // TileView - struct for tile view information
    py::class_<TileView>(ir, "TileView", "Tile view representation with valid shape, stride, and start offset")
        .def(py::init<>(), "Create an empty tile view")
        .def(py::init<const std::vector<ExprPtr>&, const std::vector<ExprPtr>&, ExprPtr>(), py::arg("valid_shape"),
             py::arg("stride"), py::arg("start_offset"),
             "Create a tile view with valid_shape, stride, and start_offset")
        .def_readwrite("valid_shape", &TileView::validShape, "Valid shape dimensions")
        .def_readwrite("stride", &TileView::stride, "Stride for each dimension")
        .def_readwrite("start_offset", &TileView::startOffset, "Starting offset");

    // Dynamic dimension constant
    ir.attr("DYNAMIC_DIM") = kDynamicDim;

    // OpRegistry
    ir.def(
        "create_op_call",
        [](const std::string& op_name, const std::vector<ExprPtr>& args, const Span& span) {
            return OpRegistry::GetInstance().Create(op_name, args, span);
        },
        py::arg("op_name"), py::arg("args"), py::arg("span"), "Create a Call expression (backward compatibility)");

    ir.def(
        "create_op_call",
        [](const std::string& op_name, const std::vector<ExprPtr>& args, const py::dict& kwargs_dict,
           const Span& span) {
            // Convert Python dict to C++ vector<pair<string, any>> to preserve order
            auto kwargs = ConvertKwargsDict(kwargs_dict);
            return OpRegistry::GetInstance().Create(op_name, args, kwargs, span);
        },
        py::arg("op_name"), py::arg("args"), py::arg("kwargs"), py::arg("span"),
        "Create a Call expression with args and kwargs");

    ir.def(
        "is_op_registered", [](const std::string& op_name) { return OpRegistry::GetInstance().IsRegistered(op_name); },
        py::arg("op_name"), "Check if an operator is registered");

    ir.def(
        "get_op", [](const std::string& op_name) { return OpRegistry::GetInstance().GetOp(op_name); },
        py::arg("op_name"), "Get an operator instance by name");
}

void BindExpr(py::module_& ir)
{
    // Var - const shared_ptr
    auto var_class = py::class_<Var, Expr, std::shared_ptr<Var>>(ir, "Var", "Variable reference expression");

    var_class.def(py::init<const std::string&, const TypePtr&, const Span&>(), py::arg("name"), py::arg("type"),
                  py::arg("span"),
                  "Create a variable reference (memory reference is stored in ShapedType for Tensor/Tile types)");
    BindFields<Var>(var_class);

    // IterArg - const shared_ptr
    auto iterarg_class = py::class_<IterArg, std::shared_ptr<IterArg>>(ir, "IterArg", "Iteration argument variable");
    iterarg_class
        .def(py::init<const std::string&, const TypePtr&, const ExprPtr&, const Span&>(), py::arg("name"),
             py::arg("type"), py::arg("initValue"), py::arg("span"), "Create an iteration argument with initial value")
        .def(py::init<VarPtr, const ExprPtr&>(), py::arg("iterVar"), py::arg("initValue"),
             "Create an iteration argument with initial value")
        .def("__str__", [](const std::shared_ptr<const IterArg>& self) -> std::string { return self->iterVar_->name_; })
        .def_property_readonly(
            "name",
            [](const std::shared_ptr<const IterArg>& self) -> const std::string& { return self->iterVar_->name_; },
            "Variable name");
    BindFields<IterArg>(iterarg_class);

    // MemRef - now inherits from Var (first-class expression)
    auto memref_class = py::class_<MemRef, Var, std::shared_ptr<MemRef>>(
        ir, "MemRef", "Memory reference variable for shaped types (inherits from Var)");
    memref_class
        .def(py::init<MemorySpace, ExprPtr, uint64_t, Span>(), py::arg("memory_space"), py::arg("addr"),
             py::arg("size"), py::arg("span") = Span::Unknown(),
             "Create a memory reference with memory_space, addr, size, and span")
        .def(py::init<MemorySpace, ExprPtr, uint64_t, uint64_t, Span>(), py::arg("memory_space"), py::arg("addr"),
             py::arg("size"), py::arg("id"), py::arg("span") = Span::Unknown(),
             "Create a memory reference with memory_space, addr, size, id, and span")
        .def_readonly("memory_space", &MemRef::memorySpace_, "Memory space (DDR, Vec, Mat, Left, Right, Scaling, Acc)")
        .def_readonly("addr", &MemRef::addr_, "Starting address expression")
        .def_readonly("size", &MemRef::size_, "Size in bytes (64-bit unsigned)")
        .def_readwrite("memory_space_", &MemRef::memorySpace_,
                       "Memory space (DDR, Vec, Mat, Left, Right, Scaling, Acc)")
        .def_readwrite("addr_", &MemRef::addr_, "Starting address expression")
        .def_readwrite("size_", &MemRef::size_, "Size in bytes (64-bit unsigned)")
        .def_static("same_allocation", &MemRef::SameAllocation, py::arg("a"), py::arg("b"),
                    "Check if two MemRefs share the same allocation (same base pointer)")
        .def_static("may_alias", &MemRef::MayAlias, py::arg("a"), py::arg("b"),
                    "Check if two MemRefs may alias (same base + overlapping byte ranges)");

    // ConstInt - const shared_ptr
    auto constint_class = py::class_<ConstInt, Expr, std::shared_ptr<ConstInt>>(ir, "ConstInt",
                                                                                "Constant integer expression");
    constint_class.def(py::init<int64_t, DataType, const Span&>(), py::arg("value"), py::arg("dtype"), py::arg("span"),
                       "Create a constant integer expression");
    BindFields<ConstInt>(constint_class);
    constint_class.def_property_readonly("dtype", &ConstInt::dtype, "Data type of the expression");

    // ConstFloat - const shared_ptr
    auto constfloat_class = py::class_<ConstFloat, Expr, std::shared_ptr<ConstFloat>>(ir, "ConstFloat",
                                                                                      "Constant float expression");
    constfloat_class.def(py::init<double, DataType, const Span&>(), py::arg("value"), py::arg("dtype"), py::arg("span"),
                         "Create a constant float expression");
    BindFields<ConstFloat>(constfloat_class);
    constfloat_class.def_property_readonly("dtype", &ConstFloat::dtype, "Data type of the expression");

    // ConstBool - const shared_ptr
    auto constbool_class = py::class_<ConstBool, Expr, std::shared_ptr<ConstBool>>(ir, "ConstBool",
                                                                                   "Constant boolean expression");
    constbool_class.def(py::init<bool, const Span&>(), py::arg("value"), py::arg("span"),
                        "Create a constant boolean expression");
    BindFields<ConstBool>(constbool_class);
    constbool_class.def_property_readonly("dtype", &ConstBool::dtype, "Data type of the expression (always BOOL)");

    // Call - const shared_ptr
    auto call_class = py::class_<Call, Expr, std::shared_ptr<Call>>(ir, "Call", "Function call expression");

    // Constructors taking op name string
    call_class.def(py::init<std::string, const std::vector<ExprPtr>&, const Span&>(), py::arg("op"), py::arg("args"),
                   py::arg("span"), "Create a function call expression from op name string");
    call_class.def(py::init<std::string, const std::vector<ExprPtr>&, const TypePtr&, const Span&>(), py::arg("op"),
                   py::arg("args"), py::arg("type"), py::arg("span"),
                   "Create a function call expression with explicit type");

    // Constructors taking Op object (extract name for backward compat)
    call_class.def(
        py::init([](const OpPtr& op, const std::vector<ExprPtr>& args, const Span& span) -> std::shared_ptr<Call> {
            return std::make_shared<Call>(op->name_, args, span);
        }),
        py::arg("op"), py::arg("args"), py::arg("span"), "Create a function call expression");
    call_class.def(py::init([](const OpPtr& op, const std::vector<ExprPtr>& args, const TypePtr& type,
                               const Span& span) -> std::shared_ptr<Call> {
                       return std::make_shared<Call>(op->name_, args, type, span);
                   }),
                   py::arg("op"), py::arg("args"), py::arg("type"), py::arg("span"),
                   "Create a function call expression with explicit type");

    // Constructors with kwargs (using py::dict)
    call_class.def(py::init([](const std::string& name, const std::vector<ExprPtr>& args, const py::dict& kwargs_dict,
                               const Span& span) -> std::shared_ptr<Call> {
                       auto kwargs = ConvertKwargsDict(kwargs_dict);
                       return std::make_shared<Call>(name, args, kwargs, span);
                   }),
                   py::arg("op"), py::arg("args"), py::arg("kwargs"), py::arg("span"),
                   "Create a function call expression with kwargs");
    call_class.def(py::init([](const std::string& name, const std::vector<ExprPtr>& args, const py::dict& kwargs_dict,
                               const TypePtr& type, const Span& span) -> std::shared_ptr<Call> {
                       auto kwargs = ConvertKwargsDict(kwargs_dict);
                       return std::make_shared<Call>(name, args, kwargs, type, span);
                   }),
                   py::arg("op"), py::arg("args"), py::arg("kwargs"), py::arg("type"), py::arg("span"),
                   "Create a function call expression with kwargs and explicit type");

    // Kwargs with Op object (backward compat)
    call_class.def(py::init([](const OpPtr& op, const std::vector<ExprPtr>& args, const py::dict& kwargs_dict,
                               const Span& span) -> std::shared_ptr<Call> {
                       auto kwargs = ConvertKwargsDict(kwargs_dict);
                       return std::make_shared<Call>(op->name_, args, kwargs, span);
                   }),
                   py::arg("op"), py::arg("args"), py::arg("kwargs"), py::arg("span"),
                   "Create a function call expression with kwargs");
    call_class.def(py::init([](const OpPtr& op, const std::vector<ExprPtr>& args, const py::dict& kwargs_dict,
                               const TypePtr& type, const Span& span) -> std::shared_ptr<Call> {
                       auto kwargs = ConvertKwargsDict(kwargs_dict);
                       return std::make_shared<Call>(op->name_, args, kwargs, type, span);
                   }),
                   py::arg("op"), py::arg("args"), py::arg("kwargs"), py::arg("type"), py::arg("span"),
                   "Create a function call expression with kwargs and explicit type");

    BindFields<Call>(call_class);

    // Expose kwargs as a read-only property
    call_class.def_property_readonly(
        "kwargs",
        [](const CallPtr& self) {
            py::dict result;
            for (const auto& [key, value] : self->kwargs_) {
                if (value.type() == typeid(int)) {
                    result[key.c_str()] = AnyCast<int>(value, "converting to Python: " + key);
                } else if (value.type() == typeid(int64_t)) {
                    result[key.c_str()] = AnyCast<int64_t>(value, "converting to Python: " + key);
                } else if (value.type() == typeid(bool)) {
                    result[key.c_str()] = AnyCast<bool>(value, "converting to Python: " + key);
                } else if (value.type() == typeid(std::string)) {
                    result[key.c_str()] = AnyCast<std::string>(value, "converting to Python: " + key);
                } else if (value.type() == typeid(double)) {
                    result[key.c_str()] = AnyCast<double>(value, "converting to Python: " + key);
                } else if (value.type() == typeid(float)) {
                    result[key.c_str()] = AnyCast<float>(value, "converting to Python: " + key);
                } else if (value.type() == typeid(DataType)) {
                    result[key.c_str()] = AnyCast<DataType>(value, "converting to Python: " + key);
                } else if (value.type() == typeid(TilePad)) {
                    result[key.c_str()] = AnyCast<TilePad>(value, "converting to Python: " + key);
                }
            }
            return result;
        },
        "Keyword arguments (metadata) for this call");

    // MakeTuple - const shared_ptr
    auto make_tuple_class = py::class_<MakeTuple, Expr, std::shared_ptr<MakeTuple>>(ir, "MakeTuple",
                                                                                    "Tuple construction expression");
    make_tuple_class.def(py::init<const std::vector<ExprPtr>&, const Span&>(), py::arg("elements"), py::arg("span"),
                         "Create a tuple construction expression");
    BindFields<MakeTuple>(make_tuple_class);

    // GetItemExpr - unified subscript expression (tuple element access or tile offset)
    auto get_item_class = py::class_<GetItemExpr, Expr, std::shared_ptr<GetItemExpr>>(
        ir, "GetItemExpr",
        "Unified subscript expression: value[slice]. Dispatch on value's static type "
        "(TupleType => tuple element access with ConstInt slice; TileType => tile element offset).");
    get_item_class.def(py::init<const ExprPtr&, const ExprPtr&, const Span&>(), py::arg("value"), py::arg("slice"),
                       py::arg("span"), "Create a subscript expression (value[slice])");
    BindFields<GetItemExpr>(get_item_class);

    // BinaryExpr - abstract, const shared_ptr
    auto binaryexpr_class = py::class_<BinaryExpr, Expr, std::shared_ptr<BinaryExpr>>(
        ir, "BinaryExpr", "Base class for binary operations");
    BindFields<BinaryExpr>(binaryexpr_class);

    // UnaryExpr - abstract, const shared_ptr
    auto unaryexpr_class = py::class_<UnaryExpr, Expr, std::shared_ptr<UnaryExpr>>(ir, "UnaryExpr",
                                                                                   "Base class for unary operations");
    BindFields<UnaryExpr>(unaryexpr_class);

// Macro to bind binary expression nodes
#define BIND_BINARY_EXPR(OpName, Description)                                                                      \
    py::class_<OpName, BinaryExpr, std::shared_ptr<OpName>>(ir, #OpName, Description)                              \
        .def(py::init<const ExprPtr&, const ExprPtr&, DataType, const Span&>(), py::arg("left"), py::arg("right"), \
             py::arg("dtype"), py::arg("span"), "Create " Description)

    // Bind all binary expression nodes
    BIND_BINARY_EXPR(Add, "Addition expression (left + right)");
    BIND_BINARY_EXPR(Sub, "Subtraction expression (left - right)");
    BIND_BINARY_EXPR(Mul, "Multiplication expression (left * right)");
    BIND_BINARY_EXPR(FloorDiv, "Floor division expression (left // right)");
    BIND_BINARY_EXPR(FloorMod, "Floor modulo expression (left % right)");
    BIND_BINARY_EXPR(FloatDiv, "Float division expression (left / right)");
    BIND_BINARY_EXPR(Min, "Minimum expression (min(left, right))");
    BIND_BINARY_EXPR(Max, "Maximum expression (max(left, right))");
    BIND_BINARY_EXPR(Pow, "Power expression (left ** right)");
    BIND_BINARY_EXPR(Eq, "Equality expression (left == right)");
    BIND_BINARY_EXPR(Ne, "Inequality expression (left != right)");
    BIND_BINARY_EXPR(Lt, "Less than expression (left < right)");
    BIND_BINARY_EXPR(Le, "Less than or equal to expression (left <= right)");
    BIND_BINARY_EXPR(Gt, "Greater than expression (left > right)");
    BIND_BINARY_EXPR(Ge, "Greater than or equal to expression (left >= right)");
    BIND_BINARY_EXPR(And, "Logical and expression (left and right)");
    BIND_BINARY_EXPR(Or, "Logical or expression (left or right)");
    BIND_BINARY_EXPR(Xor, "Logical xor expression (left xor right)");
    BIND_BINARY_EXPR(BitAnd, "Bitwise and expression (left & right)");
    BIND_BINARY_EXPR(BitOr, "Bitwise or expression (left | right)");
    BIND_BINARY_EXPR(BitXor, "Bitwise xor expression (left ^ right)");
    BIND_BINARY_EXPR(BitShiftLeft, "Bitwise left shift expression (left << right)");
    BIND_BINARY_EXPR(BitShiftRight, "Bitwise right shift expression (left >> right)");

#undef BIND_BINARY_EXPR

// Macro to bind unary expression nodes
#define BIND_UNARY_EXPR(OpName, Description)                                                                           \
    py::class_<OpName, UnaryExpr, std::shared_ptr<OpName>>(ir, #OpName, Description)                                   \
        .def(py::init<const ExprPtr&, DataType, const Span&>(), py::arg("operand"), py::arg("dtype"), py::arg("span"), \
             "Create " Description)

    // Bind all unary expression nodes
    BIND_UNARY_EXPR(Abs, "Absolute value expression (abs(operand))");
    BIND_UNARY_EXPR(Neg, "Negation expression (-operand)");
    BIND_UNARY_EXPR(Not, "Logical not expression (not operand)");
    BIND_UNARY_EXPR(BitNot, "Bitwise not expression (~operand)");
    BIND_UNARY_EXPR(Cast, "Cast expression (cast operand to dtype)");

#undef BIND_UNARY_EXPR

    // Bind structural hash and equality functions
    ir.def("structural_hash", static_cast<uint64_t (*)(const IRNodePtr&, bool)>(&structural_hash_with_var_identity),
           py::arg("node"), py::arg("enable_auto_mapping") = false,
           "Compute deterministic structural hash of an IR node (ignores Span). "
           "If enable_auto_mapping=True, variable names are ignored (e.g., x+1 and y+1 hash the same). "
           "If enable_auto_mapping=False (default), different variable objects produce different hashes.");
    ir.def("structural_hash", static_cast<uint64_t (*)(const TypePtr&, bool)>(&structural_hash_with_var_identity),
           py::arg("type"), py::arg("enable_auto_mapping") = false,
           "Compute deterministic structural hash of a type. "
           "enable_auto_mapping only affects variables embedded in the type (e.g., shape expressions).");

    ir.def("structural_equal", static_cast<bool (*)(const IRNodePtr&, const IRNodePtr&, bool)>(&structural_equal),
           py::arg("lhs"), py::arg("rhs"), py::arg("enable_auto_mapping") = false,
           "Check if two IR nodes are structurally equal. "
           "Ignores source location (Span). Returns True if IR nodes have identical structure. "
           "If enable_auto_mapping=True, automatically map variables (e.g., x+1 equals y+1). "
           "If enable_auto_mapping=False (default), variable objects must be exactly the same (not just same "
           "name).");
    ir.def("structural_equal", static_cast<bool (*)(const TypePtr&, const TypePtr&, bool)>(&structural_equal),
           py::arg("lhs"), py::arg("rhs"), py::arg("enable_auto_mapping") = false,
           "Check if two types are structurally equal. "
           "Ignores source location (Span). Returns True if types have identical structure. "
           "If enable_auto_mapping=True, automatically map variables (e.g., x+1 equals y+1). "
           "If enable_auto_mapping=False (default), variable objects must be exactly the same (not just same "
           "name).");

    ir.def("assert_structural_equal",
           static_cast<void (*)(const IRNodePtr&, const IRNodePtr&, bool)>(&assert_structural_equal), py::arg("lhs"),
           py::arg("rhs"), py::arg("enable_auto_mapping") = false,
           "Assert two IR nodes are structurally equal. "
           "Raises RuntimeError with FE error code and detailed error message showing the first mismatch location "
           "if they differ. "
           "Ignores source location (Span). "
           "If enable_auto_mapping=True, automatically map variables (e.g., x+1 equals y+1). "
           "If enable_auto_mapping=False (default), variable objects must be exactly the same (not just same "
           "name).");
    ir.def("assert_structural_equal",
           static_cast<void (*)(const TypePtr&, const TypePtr&, bool)>(&assert_structural_equal), py::arg("lhs"),
           py::arg("rhs"), py::arg("enable_auto_mapping") = false,
           "Assert two types are structurally equal. "
           "Raises RuntimeError with FE error code and detailed error message showing the first mismatch location "
           "if they differ. "
           "Ignores source location (Span). "
           "If enable_auto_mapping=True, automatically map variables (e.g., x+1 equals y+1). "
           "If enable_auto_mapping=False (default), variable objects must be exactly the same (not just same "
           "name).");
}

void BindStmt(py::module_& ir)
{
    // ========== Statements ==========

    // Stmt - abstract base, const shared_ptr
    auto stmt_class = py::class_<Stmt, IRNode, std::shared_ptr<Stmt>>(ir, "Stmt", "Base class for all statements");
    BindFields<Stmt>(stmt_class);

    // AssignStmt - const shared_ptr
    auto assign_stmt_class = py::class_<AssignStmt, Stmt, std::shared_ptr<AssignStmt>>(
        ir, "AssignStmt", "Assignment statement: var = value");
    assign_stmt_class.def(py::init<const VarPtr&, const ExprPtr&, const Span&>(), py::arg("var"), py::arg("value"),
                          py::arg("span"), "Create an assignment statement");
    BindFields<AssignStmt>(assign_stmt_class);

    // IfStmt - const shared_ptr
    auto if_stmt_class = py::class_<IfStmt, Stmt, std::shared_ptr<IfStmt>>(
        ir, "IfStmt", "Conditional statement: if condition then then_body else else_body");
    if_stmt_class.def(py::init<const ExprPtr&, const StmtPtr&, const std::optional<StmtPtr>&,
                               const std::vector<VarPtr>&, const Span&>(),
                      py::arg("condition"), py::arg("then_body"), py::arg("else_body") = py::none(),
                      py::arg("return_vars"), py::arg("span"),
                      "Create a conditional statement with then and else branches (else_body can be None)");
    BindFields<IfStmt>(if_stmt_class);

    // YieldStmt - const shared_ptr
    auto yield_stmt_class = py::class_<YieldStmt, Stmt, std::shared_ptr<YieldStmt>>(ir, "YieldStmt",
                                                                                    "Yield statement: yield value");
    yield_stmt_class.def(py::init<const std::vector<ExprPtr>&, const Span&>(), py::arg("value"), py::arg("span"),
                         "Create a yield statement with a list of expressions");
    yield_stmt_class.def(py::init<const Span&>(), py::arg("span"), "Create a yield statement without values");
    BindFields<YieldStmt>(yield_stmt_class);

    // ReturnStmt - const shared_ptr
    auto return_stmt_class = py::class_<ReturnStmt, Stmt, std::shared_ptr<ReturnStmt>>(
        ir, "ReturnStmt", "Return statement: return value");
    return_stmt_class.def(py::init<const std::vector<ExprPtr>&, const Span&>(), py::arg("value"), py::arg("span"),
                          "Create a return statement with a list of expressions");
    return_stmt_class.def(py::init<const Span&>(), py::arg("span"), "Create a return statement without values");
    BindFields<ReturnStmt>(return_stmt_class);

    // ForStmt - const shared_ptr
    auto for_stmt_class = py::class_<ForStmt, Stmt, std::shared_ptr<ForStmt>>(
        ir, "ForStmt", "For loop statement: for loop_var in range(start, stop, step): body");
    for_stmt_class.def(py::init([](const VarPtr& loop_var, const ExprPtr& start, const ExprPtr& stop,
                                   const ExprPtr& step, const std::vector<IterArgPtr>& iter_args, const StmtPtr& body,
                                   const std::vector<VarPtr>& return_vars, const Span& span, const py::object& attrs) {
                           py::dict attr_dict;
                           if (!attrs.is_none()) {
                               attr_dict = attrs.cast<py::dict>();
                           }
                           auto attr_list = ConvertAttrDict(attr_dict);
                           return std::make_shared<ForStmt>(loop_var, start, stop, step, iter_args, body, return_vars,
                                                            span, attr_list);
                       }),
                       py::arg("loop_var"), py::arg("start"), py::arg("stop"), py::arg("step"), py::arg("iter_args"),
                       py::arg("body"), py::arg("return_vars"), py::arg("span"), py::arg("attrs") = py::none(),
                       "Create a for loop statement with attributes");
    BindFields<ForStmt>(for_stmt_class);

    // WhileStmt - const shared_ptr
    auto while_stmt_class = py::class_<WhileStmt, Stmt, std::shared_ptr<WhileStmt>>(
        ir, "WhileStmt", "While loop statement: while condition: body");
    while_stmt_class.def(py::init<const ExprPtr&, const std::vector<IterArgPtr>&, const StmtPtr&,
                                  const std::vector<VarPtr>&, const Span&>(),
                         py::arg("condition"), py::arg("iter_args"), py::arg("body"), py::arg("return_vars"),
                         py::arg("span"), "Create a while loop statement");
    BindFields<WhileStmt>(while_stmt_class);

    // SectionKind enum
    py::enum_<SectionKind>(ir, "SectionKind", "Section kind classification")
        .value("Vector", SectionKind::Vector, "Vector section for vector operations")
        .value("Cube", SectionKind::Cube, "Cube section for cube operations")
        .value("VF", SectionKind::VF, "VF section for A5 VF API code (__VEC_SCOPE__)")
        .export_values();

    // SectionStmt - const shared_ptr
    auto section_stmt_class = py::class_<SectionStmt, Stmt, std::shared_ptr<SectionStmt>>(
        ir, "SectionStmt", "Section statement: marks a region with specific section context (Vector or Cube)");
    section_stmt_class.def(py::init<SectionKind, const StmtPtr&, const Span&>(), py::arg("section_kind"),
                           py::arg("body"), py::arg("span"), "Create a section statement");
    BindFields<SectionStmt>(section_stmt_class);

    // SeqStmts - const shared_ptr
    auto seq_stmts_class = py::class_<SeqStmts, Stmt, std::shared_ptr<SeqStmts>>(
        ir, "SeqStmts", "Sequence of statements: a sequence of statements");
    seq_stmts_class.def(py::init<const std::vector<StmtPtr>&, const Span&>(), py::arg("stmts"), py::arg("span"),
                        "Create a sequence of statements");
    seq_stmts_class.def(py::init<const Span&>(), py::arg("span"), "Create a sequence of statements");
    seq_stmts_class.def(
        "__getitem__",
        [](SeqStmtsPtr& self, int index) {
            int size = static_cast<int>(self->stmts_.size());
            if (index < -size || index >= size) {
                throw IndexError("SeqStmts index " + std::to_string(index) + " out of range [" + std::to_string(-size) +
                                 ", " + std::to_string(size - 1) + "]");
            }
            if (index < 0)
                index += size;
            return self->stmts_[index];
        },
        py::arg("index"), "Get statement by index, supports negative indexing");
    // Expose the sequence protocol directly. Without __iter__ Python falls back to
    // calling __getitem__ with increasing indices and stops on IndexError; the
    // out-of-range throw here is a backtrace-capturing pypto::ir::IndexError, which
    // forces a full CaptureStackTrace() (one addr2line per frame) on every loop end.
    seq_stmts_class.def(
        "__iter__",
        [](SeqStmtsPtr& self) {
            auto& stmts = self->stmts_;
            return py::make_iterator(stmts.begin(), stmts.end());
        },
        py::keep_alive<0, 1>());
    seq_stmts_class.def("__len__", [](SeqStmtsPtr& self) { return self->stmts_.size(); }, "Number of statements");
    BindFields<SeqStmts>(seq_stmts_class);

    // EvalStmt - const shared_ptr
    auto eval_stmt_class = py::class_<EvalStmt, Stmt, std::shared_ptr<EvalStmt>>(ir, "EvalStmt",
                                                                                 "Evaluation statement: expr");
    eval_stmt_class.def(py::init<const ExprPtr&, const Span&>(), py::arg("expr"), py::arg("span"),
                        "Create an evaluation statement");
    BindFields<EvalStmt>(eval_stmt_class);

    // BreakStmt - const shared_ptr
    auto break_stmt_class = py::class_<BreakStmt, Stmt, std::shared_ptr<BreakStmt>>(ir, "BreakStmt",
                                                                                    "Break statement: break");
    break_stmt_class.def(py::init<const Span&>(), py::arg("span"), "Create a break statement");
    break_stmt_class.def(py::init<const std::vector<ExprPtr>&, const Span&>(), py::arg("operands"), py::arg("span"),
                         "Create a break statement with operands");
    BindFields<BreakStmt>(break_stmt_class);

    // ContinueStmt - const shared_ptr
    auto continue_stmt_class = py::class_<ContinueStmt, Stmt, std::shared_ptr<ContinueStmt>>(
        ir, "ContinueStmt", "Continue statement: continue");
    continue_stmt_class.def(py::init<const Span&>(), py::arg("span"), "Create a continue statement");
    continue_stmt_class.def(py::init<const std::vector<ExprPtr>&, const Span&>(), py::arg("operands"), py::arg("span"),
                            "Create a continue statement with operands");
    BindFields<ContinueStmt>(continue_stmt_class);

    // ScalarOpStmt - const shared_ptr
    auto scalarop_stmt_class = py::class_<ScalarOpStmt, Stmt, std::shared_ptr<ScalarOpStmt>>(
        ir, "ScalarOpStmt", "Scalar operation statement: result, result_token = opcode(args, tokens)");
    scalarop_stmt_class.def(py::init<VarPtr, VarPtr, std::string, const std::vector<ExprPtr>&, const Span&>(),
                            py::arg("result"), py::arg("result_token"), py::arg("opcode"), py::arg("args"),
                            py::arg("span"), "Create a scalar operation statement");
    BindFields<ScalarOpStmt>(scalarop_stmt_class);

    // TensorOpStmt - const shared_ptr
    auto tensorop_stmt_class = py::class_<TensorOpStmt, Stmt, std::shared_ptr<TensorOpStmt>>(
        ir, "TensorOpStmt", "Tensor operation statement: results, result_token = opcode(args, attrs, tokens)");
    tensorop_stmt_class.def(
        py::init([](std::vector<VarPtr> results, std::vector<VarPtr> result_tokens, std::string opcode,
                    std::vector<ExprPtr> args, const std::vector<VarPtr>& tokens, py::dict attrs, Span span) {
            auto attr_list = ConvertAttrDict(attrs);
            return std::make_shared<TensorOpStmt>(results, result_tokens, opcode, args, tokens, attr_list, span);
        }),
        py::arg("results"), py::arg("result_tokens"), py::arg("opcode"), py::arg("args"), py::arg("tokens"),
        py::arg("attrs"), py::arg("span"), "Create a tensor operation statement");
    BindFields<TensorOpStmt>(tensorop_stmt_class);
}

void BindProgram(py::module_& ir)
{
    // FunctionType enum
    py::enum_<FunctionType>(ir, "FunctionType", "Function type classification")
        .value("Opaque", FunctionType::OPAQUE, "Unspecified function type (default)")
        .value("Orchestration", FunctionType::ORCHESTRATION, "Host/AICPU control and coordination")
        .value("InCore", FunctionType::IN_CORE, "AICore sub-graph execution")
        .value("Helper", FunctionType::HELPER, "Scalar helper callable from kernels (generates func.call)")
        .value("SimtVF", FunctionType::SIMT_VF, "A5 SIMT vector function launched from an AIV kernel")
        .value("SimtCallee", FunctionType::SIMT_CALLEE, "A5 helper callable from SIMT functions")
        .export_values();

    // Function - const shared_ptr
    auto function_class = py::class_<Function, IRNode, std::shared_ptr<Function>>(
        ir, "Function", py::dynamic_attr(), "Function definition with name, parameters, return types, and body");
    function_class.def(
        py::init([](const std::string& name, const py::list& params, const std::vector<TypePtr>& return_types,
                    const StmtPtr& body, const Span& span, FunctionType type, bool entry,
                    const py::object& attrs) -> std::shared_ptr<Function> {
            std::vector<VarPtr> param_vars;
            param_vars.reserve(py::len(params));
            for (auto item : params) {
                param_vars.push_back(py::cast<VarPtr>(item));
            }
            py::dict attr_dict;
            if (!attrs.is_none()) {
                attr_dict = attrs.cast<py::dict>();
            }
            auto attr_list = ConvertAttrDict(attr_dict);
            return std::make_shared<Function>(name, std::move(param_vars), return_types, body, span, type, entry,
                                              std::move(attr_list));
        }),
        py::arg("name"), py::arg("params"), py::arg("return_types"), py::arg("body"), py::arg("span"),
        py::arg("type") = FunctionType::OPAQUE, py::arg("entry") = false, py::arg("attrs") = py::none(),
        "Create a function definition");
    BindFields<Function>(function_class);
    function_class.def(
        "__str__", [](const std::shared_ptr<Function>& self) { return TextDump(self); }, "IR text representation");
    function_class.def("has_attr", &Function::HasAttr, py::arg("key"), "Check whether a function attribute exists");
    function_class.def("get_attr", &Function::GetAttr<int>, py::arg("key"), py::arg("default_value") = 0,
                       "Get an integer function attribute");

    py::enum_<TupleTypeKind>(ir, "TupleTypeKind", "Semantic tuple classification")
        .value("TUPLE", TupleTypeKind::TUPLE, "Plain positional tuple")
        .value("NAMED_TUPLE", TupleTypeKind::NAMED_TUPLE, "Named immutable tuple")
        .value("STRUCT", TupleTypeKind::STRUCT, "Mutable C++ struct");

    py::class_<TupleTypeInfo>(ir, "TupleTypeInfo", "Semantic metadata associated with one TupleType")
        .def(py::init<TupleTypeKind, std::optional<std::string>, std::vector<std::string>>(), py::arg("kind"),
             py::arg("name") = py::none(), py::arg("fields") = std::vector<std::string>{})
        .def_readonly("kind", &TupleTypeInfo::kind)
        .def_readonly("name", &TupleTypeInfo::name)
        .def_readonly("fields", &TupleTypeInfo::fields)
        .def("__eq__", &TupleTypeInfo::operator==, py::arg("other"))
        .def("__ne__", &TupleTypeInfo::operator!=, py::arg("other"));

    // IRDebugInfo - compilation-session side table for semantic tuple metadata.
    auto debug_info_class = py::class_<IRDebugInfo, std::shared_ptr<IRDebugInfo>>(
        ir, "IRDebugInfo",
        "Side table mapping a tuple type to its semantic TupleTypeInfo. "
        "Populated by the parser, carried on Program, read by codegen.");
    debug_info_class.def(py::init<>());
    debug_info_class.def("register_tuple_type_info", &IRDebugInfo::RegisterTupleTypeInfo, py::arg("type"),
                         py::arg("info"), "Record semantic metadata for a named tuple or struct type.");
    debug_info_class.def(
        "get_tuple_type_info",
        [](const IRDebugInfo& self, const TupleTypePtr& type) -> py::object {
            const auto* info = self.GetTupleTypeInfo(type.get());
            if (info == nullptr) {
                return py::none();
            }
            return py::cast(*info);
        },
        py::arg("type"), "Look up tuple metadata by type; returns None for a plain tuple.");

    // Program - const shared_ptr
    auto program_class = py::class_<Program, IRNode, std::shared_ptr<Program>>(
        ir, "Program",
        "Program definition with functions mapped by name. "
        "Functions are automatically sorted by name for deterministic ordering.");
    program_class.def(py::init([](const std::vector<FunctionPtr>& functions, const std::string& name, const Span& span,
                                  IRDebugInfoPtr debug_info) {
                          return std::make_shared<Program>(functions, name, span, std::move(debug_info));
                      }),
                      py::arg("functions"), py::arg("name"), py::arg("span"), py::arg("debug_info") = nullptr,
                      "Create a program from a list of functions. "
                      "Functions are keyed by their names automatically.");
    program_class.def_property_readonly(
        "debug_info", [](const ProgramPtr& self) { return self->debugInfo_; },
        "Semantic tuple metadata side table; None if the Program was built without one.");
    program_class.def("get_function", &Program::GetFunction, py::arg("name"),
                      "Get a function by name, returns None if not found");
    program_class.def(
        "__getitem__", [](const ProgramPtr& self, const std::string& name) { return self->GetFunction(name); },
        py::arg("name"), "Get function by name (dict-like access), returns None if not found");
    // Custom property for functions_ map that converts to Python dict
    program_class.def_property_readonly(
        "functions",
        [](const std::shared_ptr<const Program>& self) {
            py::dict result;
            for (const auto& [name, func] : self->functions_) {
                result[py::cast(name)] = py::cast(func);
            }
            return result;
        },
        "Map of function names to their corresponding functions, sorted by name");
    program_class.def_readonly("name", &Program::name_, "Program name");
    program_class.def_readonly("span", &Program::span_, "Source location");
    program_class.def(
        "__str__", [](const std::shared_ptr<const Program>& self) { return TextDump(self); }, "IR text representation");

    // Python-style printer function - unified API for IRNode
    ir.def(
        "python_print", [](const IRNodePtr& node, const std::string& prefix) { return PythonPrint(node, prefix); },
        py::arg("node"), py::arg("prefix") = "ir",
        "Print IR node (Expr, Stmt, Function, or Program) in Python IR syntax.\n\n"
        "Args:\n"
        "    node: IR node to print\n"
        "    prefix: Module prefix (default 'ir' for 'import pypto_pro.ir as ir')");

    // Python-style printer function for Type objects
    ir.def(
        "python_print_type", [](const TypePtr& type, const std::string& prefix) { return PythonPrint(type, prefix); },
        py::arg("type"), py::arg("prefix") = "ir",
        "Print Type object in Python IR syntax.\n\n"
        "Args:\n"
        "    type: Type to print\n"
        "    prefix: Module prefix (default 'ir' for 'import pypto_pro.ir as ir')");
}

void BindHelpers(py::module_& ir)
{
    // operator functions for Var (wrapped in Python for span capture and normalization)
    ir.def("add", &MakeAdd, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::Unknown(), "Addition operator");
    ir.def("sub", &MakeSub, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::Unknown(), "Subtraction operator");
    ir.def("mul", &MakeMul, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::Unknown(),
           "Multiplication operator");
    ir.def("truediv", &MakeFloatDiv, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::Unknown(),
           "True division operator");
    ir.def("floordiv", &MakeFloorDiv, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::Unknown(),
           "Floor division operator");
    ir.def("mod", &MakeFloorMod, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::Unknown(), "Modulo operator");
    ir.def("pow", &MakePow, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::Unknown(), "Power operator");
    ir.def("eq", &MakeEq, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::Unknown(), "Equality operator");
    ir.def("ne", &MakeNe, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::Unknown(), "Inequality operator");
    ir.def("lt", &MakeLt, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::Unknown(), "Less than operator");
    ir.def("le", &MakeLe, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::Unknown(),
           "Less than or equal operator");
    ir.def("gt", &MakeGt, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::Unknown(), "Greater than operator");
    ir.def("ge", &MakeGe, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::Unknown(),
           "Greater than or equal operator");
    ir.def("neg", &MakeNeg, py::arg("operand"), py::arg("span") = Span::Unknown(), "Negation operator");
    ir.def("cast", &MakeCast, py::arg("operand"), py::arg("dtype"), py::arg("span") = Span::Unknown(), "Cast operator");
    ir.def("bit_and", &MakeBitAnd, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::Unknown(),
           "Bitwise and operator");
    ir.def("bit_or", &MakeBitOr, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::Unknown(),
           "Bitwise or operator");
    ir.def("bit_xor", &MakeBitXor, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::Unknown(),
           "Bitwise xor operator");
    ir.def("bit_shift_left", &MakeBitShiftLeft, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::Unknown(),
           "Bitwise left shift operator");
    ir.def("bit_shift_right", &MakeBitShiftRight, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::Unknown(),
           "Bitwise right shift operator");
    ir.def("bit_not", &MakeBitNot, py::arg("operand"), py::arg("span") = Span::Unknown(), "Bitwise not operator");
    ir.def("not_", &MakeNot, py::arg("operand"), py::arg("span") = Span::Unknown(), "Logical not operator");
    ir.def("min_", &MakeMin, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::Unknown(), "Minimum operator");
    ir.def("max_", &MakeMax, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::Unknown(), "Maximum operator");

    // Op conversion registry bindings
    ir.def(
        "register_op_conversion",
        [](const std::string& from_op, const std::string& to_op) {
            OpConversionRegistry::GetInstance().RegisterSimple(from_op, to_op);
        },
        py::arg("from_op"), py::arg("to_op"),
        "Register a simple tensor-to-block op name mapping.\n\n"
        "Args:\n"
        "    from_op: Source op name (e.g., 'tensor.add')\n"
        "    to_op: Target op name (e.g., 'block.add')");

    ir.def(
        "register_op_conversion_custom",
        [](const std::string& from_op, py::object func) {
            // Capture Python callable in a C++ ConversionFunc
            OpConversionRegistry::GetInstance().RegisterCustom(
                from_op,
                [func](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
                       const Span& span) -> ConversionResult {
                    py::gil_scoped_acquire guard;
                    // Convert kwargs to Python list of (key, value) tuples
                    py::list py_kwargs_list;
                    for (const auto& [key, val] : kwargs) {
                        py::object py_val = AnyToPyObject<DataType, MemorySpace, bool, int, std::string, double>(val,
                                                                                                                 key);
                        py::tuple pair = py::make_tuple(py::cast(key), py_val);
                        py_kwargs_list.append(pair);
                    }
                    py::object result = func(py::cast(args), py_kwargs_list, py::cast(span));
                    // Result can be:
                    // 1. An ExprPtr (simple conversion)
                    // 2. A tuple of (list[StmtPtr], ExprPtr) (complex conversion)
                    if (py::isinstance<py::tuple>(result)) {
                        py::tuple result_tuple = py::cast<py::tuple>(result);
                        auto prologue = py::cast<std::vector<StmtPtr>>(result_tuple[0]);
                        auto expr = py::cast<ExprPtr>(result_tuple[1]);
                        return ConversionResult{std::move(prologue), std::move(expr)};
                    }
                    return ConversionResult{py::cast<ExprPtr>(result)};
                });
        },
        py::arg("from_op"), py::arg("func"),
        "Register a custom conversion function for a tensor op.\n\n"
        "The function receives (args, kwargs, span) and should return either:\n"
        "- An Expr (simple conversion)\n"
        "- A tuple (list[Stmt], Expr) for complex conversions with prologue statements");
}

} // namespace ir

void BindIR(py::module_& m)
{
    py::module_ ir_module = m.def_submodule("ir", "PyPTO IR (Intermediate Representation) module");

    ir::BindDType(ir_module);
    ir::BindSpan(ir_module);
    ir::BindOp(ir_module);
    ir::BindTypeClass(ir_module);
    ir::BindExpr(ir_module);
    ir::BindStmt(ir_module);
    ir::BindProgram(ir_module);
    ir::BindHelpers(ir_module);
    ir::BindIRBuilder(m);
    ir::BindPasses(ir_module);
}

} // namespace pypto
