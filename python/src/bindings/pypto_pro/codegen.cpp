/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "backend/common/backend.h"
#include "codegen/cce/cce_codegen.h"

namespace py = pybind11;

namespace pypto {
namespace ir {

using namespace pypto::backend; // NOLINT(build/namespaces)
using namespace pypto::codegen; // NOLINT(build/namespaces)
using namespace pypto::ir;      // NOLINT(build/namespaces)

void BindCodegen(py::module_& m)
{
    // Create a new 'codegen' submodule
    py::module_ codegen_module = m.def_submodule("codegen", "Code generation module for converting IR to pto-isa C++");

    // TypeConverter class for type conversions
    py::class_<TypeConverter>(codegen_module, "TypeConverter", "Utility for converting IR types to pto-isa C++ types")
        .def(py::init<>(), "Create a type converter")
        .def("ConvertPipeType", &TypeConverter::ConvertPipeType, py::arg("pipe"),
             "Convert PipeType to pto-isa pipe type string\n\n"
             "Args:\n"
             "    pipe: Pipeline type\n\n"
             "Returns:\n"
             "    C++ pipe type string with 'PIPE_' prefix (e.g., 'PIPE_MTE1', 'PIPE_V')")
        .def("ConvertEventId", &TypeConverter::ConvertEventId, py::arg("event_id"),
             "Convert event ID to pto-isa event ID string\n\n"
             "Args:\n"
             "    event_id: Event ID (must be in range [0, 7])\n\n"
             "Returns:\n"
             "    C++ event ID string with 'EVENT_ID' prefix (e.g., 'EVENT_ID0')")
        .def("GenerateShapeType", &TypeConverter::GenerateShapeType, py::arg("dims"),
             "Generate Shape type instantiation\n\n"
             "Args:\n"
             "    dims: Shape dimensions\n\n"
             "Returns:\n"
             "    Shape type string with 5D padding (e.g., 'Shape<1, 1, 1, 128, 64>')")
        .def("GenerateStrideType", &TypeConverter::GenerateStrideType, py::arg("shape"),
             "Generate Stride type instantiation for row-major layout\n\n"
             "Args:\n"
             "    shape: Shape dimensions\n\n"
             "Returns:\n"
             "    Stride type string with 5D padding");

    // CCECodegen - CCE/pto-isa C++ code generator (unified in codegen module)
    py::class_<CCECodegen>(codegen_module, "CCECodegen",
                           "CCE code generator for converting PyPTO IR to pto-isa C++ code")
        .def(py::init<SectionKind>(), py::arg("target"),
             "Create a CCE code generator for one fixed Cube or Vector target")
        .def(
            "generate_single",
            [](CCECodegen& self, const ProgramPtr& program, npu::tile_fwk::NPUArch arch) {
                return self.GenerateSingle(program, arch);
            },
            py::arg("program"), py::arg("arch"),
            "Generate a single C++ file from a target-specific PyPTO IR Program. "
            "Runs IR passes and generates an AICORE function with its target guard, "
            "constexpr scalars, and FFTS support. Returns C++ code as a single string.")
        .def(
            "get_tiling_headers", [](const CCECodegen& self) { return self.GetTilingHeaders(); },
            "Tiling struct headers from the last generate_single call, as a dict mapping "
            "header filename (e.g. 'OpTiling_tiling.h') to its content. The kernel.cpp "
            "includes these by name; write them next to kernel.cpp in the build dir.")
        .def_static("generate_tiling_header", &CCECodegen::GenerateTilingHeader, py::arg("type_name"),
                    py::arg("fields"), py::arg("types"), py::arg("requires_volatile") = false,
                    "Generate a standalone TilingData header without running full kernel codegen.");
}

} // namespace ir
} // namespace pypto
