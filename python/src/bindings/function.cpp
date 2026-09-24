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
 * \file function.cpp
 * \brief Python bindings for Function class
 */

#include "pybind_common.h"
#include "interface/function/function.h"
#include "interface/function/rebuildable_attribute.h"
#include "interface/program/program.h"

using namespace npu::tile_fwk;

namespace pypto {

void BindFunction(py::module_& m)
{
    // Bind the Function class
    py::class_<Function, ir::Function, std::shared_ptr<Function>>(m, "Function")
        .def("GetMagicName", &Function::GetMagicName, "Get the magic name of the function")
        .def("GetRawName", &Function::GetRawName, "Get the raw name of the function")
        .def("GetFuncMagic", &Function::GetFuncMagic, "Get the function magic number")
        .def("Dump", &Function::Dump, "Dump the function in brief format")
        .def("DumpSSA", &Function::DumpSSA, "Dump the function in SSA format")
        .def("GetFunctionType", &Function::GetFunctionType, "Get the function type")
        .def("GetFunctionTypeStr", &Function::GetFunctionTypeStr, "Get the function type as string")
        .def("GetGraphType", &Function::GetGraphType, "Get the graph type")
        .def("IsEager", &Function::IsEager, "Check if function is eager")
        .def("IsStatic", &Function::IsStatic, "Check if function is static")
        .def("IsExplicit", &Function::IsExplicit, "Check if function is explicit")
        .def("IsFunctionType", py::overload_cast<FunctionType>(&Function::IsFunctionType, py::const_), py::arg("type"),
             "Check if function is of given type")
        .def("IsGraphType", py::overload_cast<GraphType>(&Function::IsGraphType, py::const_), py::arg("type"),
             "Check if function has given graph type")
        .def("IsFunctionTypeAndGraphType",
             py::overload_cast<FunctionType, GraphType>(&Function::IsFunctionTypeAndGraphType, py::const_),
             py::arg("func_type"), py::arg("graph_type"), "Check if function has given function type and graph type")
        .def("HasParent", &Function::HasParent, "Check if function has a parent")
        .def("GetRootFunction", &Function::GetRootFunction, py::return_value_policy::reference, "Get the root function")
        .def(
            "GetIncast",
            [](const Function& self) -> std::vector<Tensor> {
                return std::vector<Tensor>(self.GetIncast().begin(), self.GetIncast().end());
            },
            py::return_value_policy::reference_internal, "Get input casts")
        .def(
            "GetOutcast",
            [](const Function& self) -> std::vector<Tensor> {
                return std::vector<Tensor>(self.GetOutcast().begin(), self.GetOutcast().end());
            },
            py::return_value_policy::reference_internal, "Get output casts")
        .def(
            "GetOriginIncast",
            [](const Function& self) -> std::vector<Tensor> {
                return std::vector<Tensor>(self.GetOriginIncast().begin(), self.GetOriginIncast().end());
            },
            py::return_value_policy::reference_internal, "Get original input casts")
        .def(
            "GetOriginOutcast",
            [](const Function& self) -> std::vector<Tensor> {
                return std::vector<Tensor>(self.GetOriginOutcast().begin(), self.GetOriginOutcast().end());
            },
            py::return_value_policy::reference_internal, "Get original output casts")
        .def("DumpFile", &Function::DumpFile, py::arg("file_path"), "Dump the function to a file")
        .def(
            "DumpJsonFile", [](Function& self, const std::string& fileName) { self.DumpJsonFile(fileName); },
            py::arg("file_name") = "", "Dump the function to a JSON file")
        .def("__repr__",
             [](const Function& self) {
                 return "<Function '" + self.GetRawName() + "' (magic: " + std::to_string(self.GetFuncMagic()) + ")>";
             })
        .def_property_readonly(
            "original_body", [](const Function& self) -> ir::SeqStmtsPtr { return self.originalBody_; },
            "Original function body statement")
        .def(
            "RegisterDivisibleAssumption",
            [](Function&, const SymbolicScalar& expr, int64_t divisor) {
                Program::GetInstance().RegisterDivisibleAssumption(expr, divisor);
            },
            py::arg("expr"), py::arg("divisor"), "Register an assumption that expr is divisible by divisor")
        .def(
            "IsKnownDivisible",
            [](const Function&, const SymbolicScalar& expr, int64_t divisor) {
                return Program::GetInstance().IsKnownDivisible(expr, divisor);
            },
            py::arg("expr"), py::arg("divisor"), "Check if expr is known to be divisible by divisor")
        .def("DumpAttrs", [](Function& self) { return RebuildableAttributeManager::GetInstance().DumpAttrs(&self); });

    // Add a function to get the last function from the Program
    m.def(
        "GetLastFunction", []() -> Function* { return Program::GetInstance().GetLastFunction(); },
        py::return_value_policy::reference, "Get the last compiled function from the Program");

    // Also add GetCurrentFunction for completeness
    m.def(
        "GetCurrentFunction", []() -> Function* { return Program::GetInstance().GetCurrentFunction(); },
        py::return_value_policy::reference, "Get the current function being built in the Program");

    // All slot info (logical tensor -> slot id)
    m.def(
        "GetSlotInfo",
        []() -> py::dict {
            py::dict slots;
            auto slotManager = Program::GetInstance().GetTensorSlotManager();
            for (auto& [lt, tensor] : slotManager->slotTensorDict) {
                slots[lt->name_.c_str()] = tensor->Id();
            }
            return slots;
        },
        "Every slot entry as {tensor name: slot id}; tensors sharing a slot id share one runtime buffer");
}

} // namespace pypto
