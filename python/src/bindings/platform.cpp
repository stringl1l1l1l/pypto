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
 * \file platform.cpp
 * \brief
 */

#include "pybind_common.h"

#include "tilefwk/platform.h"

using namespace npu::tile_fwk;

namespace pypto {
void BindPlatform(py::module_& m)
{
    auto arch_enum = py::enum_<NPUArch>(m, "NpuArch", py::arithmetic());
    arch_enum.value("DAV_1001", NPUArch::DAV_1001);
    arch_enum.value("DAV_2201", NPUArch::DAV_2201);
    arch_enum.value("DAV_3510", NPUArch::DAV_3510);
    arch_enum.value("DAV_3003", NPUArch::DAV_3003);
    arch_enum.value("DAV_3113", NPUArch::DAV_3113);
    arch_enum.value("DAV_UNKNOWN", NPUArch::DAV_UNKNOWN);
    // str() must yield the canonical __NPU_ARCH__ number ("3510"): the string form crosses
    // the std::string pybind boundaries (env var, bisheng flags, generated paths).
    arch_enum.attr("__str__") = py::cpp_function([](NPUArch self) { return std::to_string(static_cast<int>(self)); },
                                                 py::is_method(arch_enum), py::name("__str__"));

    m.def("GetNPUArch", []() -> std::string {
         auto npuArch = Platform::Instance().GetSoc().GetNPUArch();
         return NPUArchToString(npuArch);
     }).def("SetNPUArch", [](const std::string& value) { Platform::Instance().GetSoc().SetNPUArch(value); });

    m.def("GetAICoreNum", []() -> size_t { return Platform::Instance().GetSoc().GetAICoreNum(); });

    m.def("GetAICCoreNum", []() -> size_t { return Platform::Instance().GetSoc().GetAICCoreNum(); });

    m.def("GetAIVCoreNum", []() -> size_t { return Platform::Instance().GetSoc().GetAIVCoreNum(); });

    m.def("GetMemoryLimitForArch", [](const std::string& arch, const std::string& space) -> size_t {
        return GetMemoryLimitForArch(arch, space);
    });
}
} // namespace pypto
