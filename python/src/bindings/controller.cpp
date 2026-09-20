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
 * \file controller.cpp
 * \brief
 */

#include "pybind_common.h"
#include "tilefwk/pypto_fwk_log.h"

#include <utility>
#include <vector>
#include <string>
#include <cctype>

using namespace npu::tile_fwk;
using ref_tensors = std::vector<std::reference_wrapper<const Tensor>>;

namespace pypto {
void BindControllerConfig(py::module_& m)
{
    m.def("SetBuildStatic", [](const bool& value) { config::SetBuildStatic(value); }, py::arg("value"));

    m.def("ResetOptions", []() { config::Reset(); });

    m.def(
        "SetPrintOptions",
        [](int edgeItems, int precision, int threshold, int linewidth) {
            config::SetPrintOptions(edgeItems, precision, threshold, linewidth);
        },
        py::arg("edgeItems"), py::arg("precision"), py::arg("threshold"), py::arg("linewidth"));

    m.def(
        "SetSemanticLabel",
        [](const std::string& label, const std::string& filename, int lineno) {
            config::SetSemanticLabel(label, filename.c_str(), lineno);
        },
        py::arg("label"), py::arg("filename"), py::arg("lineno"));

    m.def("IsVerifyEnabled", &calc::IsVerifyEnabled);
    m.def("LogTopFolder", []() { return py::cast(ConfigManager::Instance().LogTopFolder()); });
    m.def("ResetLog", [](const std::string& path) { ConfigManager::Instance().ResetLog(path); });
    m.def("SetLogRotationEnabled", [](bool enabled) { ConfigManager::Instance().SetLogRotationEnabled(enabled); });
}

void BindControllerSetTile(py::module_& m)
{
    m.def("SetVecTile", [](py::args args) {
        std::vector<int64_t> v;
        v.reserve(args.size());
        for (auto& a : args) {
            v.push_back(a.cast<int64_t>());
        }
        TileShape::Current().SetVecTile(v);
    });
    m.def("GetVecTile", []() { return TileShape::Current().GetVecTile().tile; });
    m.def(
        "SetMatrixSize", [](const std::vector<int64_t>& size) { TileShape::Current().SetMatrixSize(size); },
        py::arg("size"));
    m.def(
        "SetCubeTile",
        [](const std::vector<int64_t>& mvec, const std::vector<int64_t>& kvec, const std::vector<int64_t>& nvec,
           bool enableSplitK) {
            if (mvec.size() > MAX_M_DIM_SIZE) {
                throw py::value_error("Parameter 'm' must have exactly " + std::to_string(MAX_M_DIM_SIZE) +
                                      " elements");
            }
            if (kvec.size() > MAX_K_DIM_SIZE) {
                throw py::value_error("Parameter 'k' must have exactly " + std::to_string(MAX_K_DIM_SIZE) +
                                      " elements");
            }
            if (nvec.size() > MAX_N_DIM_SIZE) {
                throw py::value_error("Parameter 'n' must have exactly " + std::to_string(MAX_N_DIM_SIZE) +
                                      " elements");
            }

            std::array<int64_t, MAX_M_DIM_SIZE> marr = {0};
            std::array<int64_t, MAX_K_DIM_SIZE> karr = {0};
            std::array<int64_t, MAX_N_DIM_SIZE> narr = {0};

            std::copy(mvec.begin(), mvec.end(), marr.begin());
            std::copy(kvec.begin(), kvec.end(), karr.begin());
            std::copy(nvec.begin(), nvec.end(), narr.begin());
            TileShape::Current().SetCubeTile(marr, karr, narr, enableSplitK);
        },
        py::arg("m"), py::arg("k"), py::arg("n"), py::arg("enable_split_k"),
        "Set cube tile shapes with specified dimensions");
    m.def("GetCubeTile", []() {
        auto cubeTile = TileShape::Current().GetCubeTile();
        return std::tuple(cubeTile.m, cubeTile.k, cubeTile.n, cubeTile.enableSplitK);
    });
    py::class_<Conv::TileL1Info>(m, "TileL1Info")
        .def(py::init<>())
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>(), py::arg("tileHin"),
             py::arg("tileHout"), py::arg("tileWin"), py::arg("tileWout"), py::arg("tileCinFmap"),
             py::arg("tileCinWeight"), py::arg("tileN"), py::arg("tileBatch"))
        .def_readwrite("tileHin", &Conv::TileL1Info::tileHin)
        .def_readwrite("tileHout", &Conv::TileL1Info::tileHout)
        .def_readwrite("tileWin", &Conv::TileL1Info::tileWin)
        .def_readwrite("tileWout", &Conv::TileL1Info::tileWout)
        .def_readwrite("tileCinFmap", &Conv::TileL1Info::tileCinFmap)
        .def_readwrite("tileCinWeight", &Conv::TileL1Info::tileCinWeight)
        .def_readwrite("tileN", &Conv::TileL1Info::tileN)
        .def_readwrite("tileBatch", &Conv::TileL1Info::tileBatch);
    py::class_<Conv::TileL0Info>(m, "TileL0Info")
        .def(py::init<>())
        .def(py::init<int64_t, int64_t, int64_t, int64_t>(), py::arg("tileH"), py::arg("tileW"), py::arg("tileK"),
             py::arg("tileN"))
        .def_readwrite("tileH", &Conv::TileL0Info::tileH)
        .def_readwrite("tileW", &Conv::TileL0Info::tileW)
        .def_readwrite("tileK", &Conv::TileL0Info::tileK)
        .def_readwrite("tileN", &Conv::TileL0Info::tileN);
    m.def(
        "SetConvTile",
        [](const Conv::TileL1Info& tileL1Info, const Conv::TileL0Info& tileL0Info, bool setL0Tile) {
            TileShape::Current().SetConvTile(tileL1Info, tileL0Info, setL0Tile);
        },
        py::arg("tileL1Info"), py::arg("tileL0Info"), py::arg("setL0Tile"), "Set conv tile shapes");
    m.def("GetConvTile", []() {
        auto convTile = TileShape::Current().GetConvTile();
        return std::tuple(convTile.tileL1Info, convTile.tileL0Info, convTile.setL0Tile);
    });
    py::class_<ConvBp::ConvBpTileL1Info>(m, "ConvBpTileL1Info")
        .def(py::init<>())
        .def(py::init<int64_t, int64_t, int64_t>(), py::arg("tileML1"), py::arg("tileKL1"), py::arg("tileNL1"))
        .def_readwrite("tileML1", &ConvBp::ConvBpTileL1Info::tileML1)
        .def_readwrite("tileKL1", &ConvBp::ConvBpTileL1Info::tileKL1)
        .def_readwrite("tileNL1", &ConvBp::ConvBpTileL1Info::tileNL1);
    py::class_<ConvBp::ConvBpTileL0Info>(m, "ConvBpTileL0Info")
        .def(py::init<>())
        .def(py::init<int64_t, int64_t, int64_t>(), py::arg("tileML0"), py::arg("tileKL0"), py::arg("tileNL0"))
        .def_readwrite("tileML0", &ConvBp::ConvBpTileL0Info::tileML0)
        .def_readwrite("tileKL0", &ConvBp::ConvBpTileL0Info::tileKL0)
        .def_readwrite("tileNL0", &ConvBp::ConvBpTileL0Info::tileNL0);
    m.def(
        "SetConvBpTile",
        [](const ConvBp::ConvBpTileL1Info& tileL1Info, const ConvBp::ConvBpTileL0Info& tileL0Info) {
            TileShape::Current().SetConvBpTile(tileL1Info, tileL0Info);
        },
        py::arg("tileL1Info"), py::arg("tileL0Info"), "Set convBp tile shapes");
    m.def("GetConvBpTile", []() {
        auto convBpTile = TileShape::Current().GetConvBpTile();
        return std::tuple(convBpTile.tileL1Info, convBpTile.tileL0Info);
    });
}

void BindControllerFunction(py::module_& m)
{
    m.def("BeginFunction", [](const std::string& funcName, GraphType graphType, FunctionType funcType, py::args args) {
        std::vector<std::reference_wrapper<const Tensor>> tensors;
        tensors.reserve(args.size());
        for (auto& a : args) {
            tensors.push_back(a.cast<Tensor&>());
        }
        Program::GetInstance().Reset();
        Program::GetInstance().BeginFunction(FUNCTION_PREFIX + funcName, funcType, graphType, tensors);
    });
    m.def("EndFunction", [](const std::string& funcName, bool generateCall) {
        Program::GetInstance().EndFunction(FUNCTION_PREFIX + funcName, generateCall);
    });
    py::class_<RecordFunc>(m, "RecordFunc")
        .def(py::init<const std::string&>(), py::arg("name"))
        .def(py::init<const std::string&, const std::vector<std::reference_wrapper<const Tensor>>&>(), py::arg("name"),
             py::arg("explicit_op_args"))
        .def(py::init<const std::string&, const ref_tensors&, const ref_tensors&,
                      const std::vector<
                          std::pair<std::reference_wrapper<const Tensor>, std::reference_wrapper<const Tensor>>>&>(),
             py::arg("name"), py::arg("inputs"), py::arg("outputs"), py::arg("in_place_args"))
        .def("EndFunction", &RecordFunc::EndFunction)
        .def("AbortRecording", &RecordFunc::AbortRecording)
        .def("__iter__", [](RecordFunc& c) {
            // Return Python iterator from C++ begin/end
            return py::make_iterator(c.begin(), c.end());
        });
    py::class_<RecordLoopFunc>(m, "RecordLoopFunc")
        .def(py::init<const std::string&, FunctionType, const std::string&, const LoopRange&, const std::set<int>&,
                      bool, bool>(),
             py::arg("name"), py::arg("func_type"), py::arg("iter_name"), py::arg("loop_range"), py::arg("unroll_List"),
             py::arg("submit_before_loop"), py::arg("parallel"))
        .def("__iter__", [](RecordLoopFunc& c) {
            // Return Python iterator from C++ begin/end
            return py::make_iterator(c.begin(), c.end());
        });
}

void BindControllerLoop(py::module_& m)
{
    py::class_<RecordIfBranch>(m, "RecordIfBranch")
        .def(py::init<SymbolicScalar, const std::string&, int>(), py::arg("cond"), py::arg("file") = "",
             py::arg("line") = 0)
        .def("__bool__", py::overload_cast<>(&RecordIfBranch::operator bool, py::const_));
    py::class_<LoopRange>(m, "LoopRange")
        .def(py::init<const SymbolicScalar& /* rangeBegin */, const SymbolicScalar& /* rangeEnd */,
                      const SymbolicScalar& /* rangeStep */>())
        .def(py::init<const SymbolicScalar& /* rangeBegin */, const SymbolicScalar& /* rangeEnd */>())
        .def(py::init<const SymbolicScalar& /* rangeEnd */>())
        .def(py::init<std::int64_t>()) // C++ Implicit conversion int64_t -> SymbolicScalar
        .def("Dump", (std::string(LoopRange::*)()) & LoopRange::Dump)
        .def("Begin", (SymbolicScalar & (LoopRange::*)()) & LoopRange::Begin,
             py::return_value_policy::reference_internal)
        .def("End", (SymbolicScalar & (LoopRange::*)()) & LoopRange::End, py::return_value_policy::reference_internal)
        .def("Step", (SymbolicScalar & (LoopRange::*)()) & LoopRange::Step,
             py::return_value_policy::reference_internal);

    m.def("IsLoopBegin", &IsLoopBegin, py::arg("symbol"), py::arg("begin"));
    m.def("IsLoopEnd", &IsLoopEnd, py::arg("symbol"), py::arg("end"));
}

void BindControllerUtils(py::module_& m)
{
    m.def("Dump", []() { return Program::GetInstance().Dump(); });
    m.def("BytesOf", [](DataType t) { return BytesOf(t); });
    m.def("Reset", []() { Program::GetInstance().Reset(); });
    m.def(
        "SetSpan", [](const std::string& fname, int lineno) { ir::Span::SetCurrent(ir::Span(fname, lineno, 0)); },
        py::arg("fname"), py::arg("lineno"));
    m.def("ClearSpan", &ir::Span::ClearCurrent);
}

static std::string GetElemTypeName(const py::handle& elem)
{
    return py::str(py::type::handle_of(elem).attr("__name__")).cast<std::string>();
}

static std::any ConvertListList(const std::string& key, const py::list& lst)
{
    std::vector<std::pair<std::string, std::string>> pairs;
    for (size_t i = 0; i < lst.size(); ++i) {
        if (!py::isinstance<py::list>(lst[i])) {
            throw py::type_error("Option '" + key + "' has invalid list element type at index " + std::to_string(i) +
                                 ". Expected list, but got " + GetElemTypeName(lst[i]));
        }
        py::list innerList = py::cast<py::list>(lst[i]);
        if (innerList.size() != 0x2) {
            throw py::type_error("Option '" + key + "' has invalid inner list length at index " + std::to_string(i) +
                                 ". Expected 2 elements [device_name, cpu_name], but got " +
                                 std::to_string(innerList.size()));
        }
        if (!py::isinstance<py::str>(innerList[0]) || !py::isinstance<py::str>(innerList[1])) {
            throw py::type_error("Option '" + key + "' has invalid inner list element type at index " +
                                 std::to_string(i) + ". Expected [str, str]");
        }
        pairs.push_back({py::cast<std::string>(innerList[0]), py::cast<std::string>(innerList[1])});
    }
    return pairs;
}

std::any ConvertPyList(const std::string& key, const py::list& lst)
{
    if (lst.size() == 0) {
        const auto& expectedType = ConfigManagerNg::GetInstance().Type(key);
        if (expectedType == typeid(std::vector<std::string>)) {
            return std::vector<std::string>();
        }
        if (expectedType == typeid(std::vector<double>)) {
            return std::vector<double>();
        }
        return std::vector<int64_t>();
    }
    auto getElemTypeName = [](const py::handle& elem) {
        return py::str(py::type::handle_of(elem).attr("__name__")).cast<std::string>();
    };
    if (py::isinstance<py::int_>(lst[0])) {
        std::vector<int64_t> intVec;
        for (size_t i = 0; i < lst.size(); ++i) {
            if (!py::isinstance<py::int_>(lst[i])) {
                throw py::type_error("Option '" + key + "' has invalid list element type at index " +
                                     std::to_string(i) + ". Expected int, but got " + getElemTypeName(lst[i]));
            }
            intVec.push_back(py::cast<int64_t>(lst[i]));
        }
        return intVec;
    } else if (py::isinstance<py::str>(lst[0])) {
        std::vector<std::string> strVec;
        for (size_t i = 0; i < lst.size(); ++i) {
            if (!py::isinstance<py::str>(lst[i])) {
                throw py::type_error("Option '" + key + "' has invalid list element type at index " +
                                     std::to_string(i) + ". Expected str, but got " + getElemTypeName(lst[i]));
            }
            strVec.push_back(py::cast<std::string>(lst[i]));
        }
        return strVec;
    } else if (py::isinstance<py::float_>(lst[0])) {
        std::vector<double> doubleVec;
        for (size_t i = 0; i < lst.size(); ++i) {
            if (!py::isinstance<py::float_>(lst[i])) {
                throw py::type_error("Option '" + key + "' has invalid list element type at index " +
                                     std::to_string(i) + ". Expected float, but got " + getElemTypeName(lst[i]));
            }
            doubleVec.push_back(py::cast<double>(lst[i]));
        }
        return doubleVec;
    } else if (py::isinstance<py::list>(lst[0])) {
        return ConvertListList(key, lst);
    }
    throw py::type_error("Unsupported list element type for key: " + key);
}

namespace {
constexpr const char* FUNC_HASH_ORDER_DEFAULT_KEY = "DEFAULT";
} // namespace

// Check if key matches "func{magic}_{order}" pattern (e.g., "func123_0", "func456_1")
static bool IsFuncHashOrderFormat(const std::string& key)
{
    // Pattern: func{magic}_{order} where magic is digits and order is digits
    // Regex equivalent: ^func\d+_\d+$
    if (key.size() < 0x6 || key.substr(0, 0x4) != "func") {
        return false;
    }
    size_t underscore_pos = key.find('_');
    if (underscore_pos == std::string::npos || underscore_pos == 0x4) {
        return false; // No underscore or underscore right after "func"
    }
    // Check that magic part (between "func" and "_") is all digits
    for (size_t i = 0x4; i < underscore_pos; ++i) {
        if (!std::isdigit(key[i])) {
            return false;
        }
    }
    // Check that order part (after "_") is all digits
    for (size_t i = underscore_pos + 1; i < key.size(); ++i) {
        if (!std::isdigit(key[i])) {
            return false;
        }
    }
    return true;
}

static void SplitByFuncAndLabel(const py::dict& dict_value, std::map<std::string, int64_t>& func_map,
                                std::map<std::string, int64_t>& label_map)
{
    for (auto dict_item : dict_value) {
        if (py::isinstance<py::str>(dict_item.first)) {
            int64_t val = py::cast<int64_t>(dict_item.second);
            std::string str_key = py::cast<std::string>(dict_item.first);
            if (IsFuncHashOrderFormat(str_key) || str_key == FUNC_HASH_ORDER_DEFAULT_KEY) {
                func_map[str_key] = val;
            } else {
                label_map[str_key] = val;
            }
        }
    }
}

static void ClassifyKeys(const py::dict& dict_value, bool& has_int, bool& has_str, bool& has_func, bool& has_label)
{
    for (auto dict_item : dict_value) {
        if (py::isinstance<py::int_>(dict_item.first)) {
            has_int = true;
        } else if (py::isinstance<py::str>(dict_item.first)) {
            has_str = true;
            std::string str_key = py::cast<std::string>(dict_item.first);
            if (IsFuncHashOrderFormat(str_key) || str_key == FUNC_HASH_ORDER_DEFAULT_KEY) {
                has_func = true;
            } else {
                has_label = true;
            }
        }
    }
}

static void HandleStrOnlyKeys(const std::string& key, const py::dict& dict_value, bool has_func, bool has_label,
                              std::map<std::string, std::any>& cpp_values)
{
    cpp_values[key] = std::map<int64_t, int64_t>();
    if (has_func && has_label) {
        std::map<std::string, int64_t> func_map, label_map;
        SplitByFuncAndLabel(dict_value, func_map, label_map);
        cpp_values[key + "_by_func"] = func_map;
        cpp_values[key + "_by_label"] = label_map;
    } else if (has_func) {
        std::map<std::string, int64_t> func_map;
        for (auto item : dict_value) {
            func_map[py::cast<std::string>(item.first)] = py::cast<int64_t>(item.second);
        }
        cpp_values[key + "_by_func"] = func_map;
    } else {
        std::map<std::string, int64_t> label_map;
        for (auto item : dict_value) {
            label_map[py::cast<std::string>(item.first)] = py::cast<int64_t>(item.second);
        }
        cpp_values[key + "_by_label"] = label_map;
    }
}

static void HandleMixedKeys(const std::string& key, const py::dict& dict_value,
                            std::map<std::string, std::any>& cpp_values)
{
    std::map<int64_t, int64_t> int_map;
    std::map<std::string, int64_t> func_map, label_map;
    for (auto item : dict_value) {
        int64_t val = py::cast<int64_t>(item.second);
        if (py::isinstance<py::int_>(item.first)) {
            int_map[py::cast<int64_t>(item.first)] = val;
        } else if (py::isinstance<py::str>(item.first)) {
            std::string str_key = py::cast<std::string>(item.first);
            if (IsFuncHashOrderFormat(str_key) || str_key == FUNC_HASH_ORDER_DEFAULT_KEY) {
                func_map[str_key] = val;
            } else {
                label_map[str_key] = val;
            }
        }
    }
    cpp_values[key] = int_map;
    if (!func_map.empty()) {
        cpp_values[key + "_by_func"] = func_map;
    }
    if (!label_map.empty()) {
        cpp_values[key + "_by_label"] = label_map;
    }
}

void ConvertPyDict(const std::string& key, const py::object& value, std::map<std::string, std::any>& cpp_values)
{
    py::dict dict_value = py::cast<py::dict>(value);
    bool has_int = false, has_str = false, has_func = false, has_label = false;
    ClassifyKeys(dict_value, has_int, has_str, has_func, has_label);

    if (!has_str) {
        cpp_values[key] = has_int ? value.cast<std::map<int64_t, int64_t>>() : std::map<int64_t, int64_t>();
    } else if (!has_int) {
        HandleStrOnlyKeys(key, dict_value, has_func, has_label, cpp_values);
    } else {
        HandleMixedKeys(key, dict_value, cpp_values);
    }
}

std::any ConvertPyValue(const std::string& key, const py::object& value)
{
    if (py::isinstance<py::bool_>(value)) {
        return value.cast<bool>();
    } else if (py::isinstance<py::int_>(value)) {
        return value.cast<int64_t>();
    } else if (py::isinstance<py::float_>(value)) {
        return value.cast<double>();
    } else if (py::isinstance<py::str>(value)) {
        return value.cast<std::string>();
    } else if (py::isinstance<CubeTile>(value)) {
        return value.cast<CubeTile>();
    } else if (py::isinstance<ConvTile>(value)) {
        return value.cast<ConvTile>();
    } else if (py::isinstance<ConvBpTile>(value)) {
        return value.cast<ConvBpTile>();
    } else if (py::isinstance<py::list>(value) || py::isinstance<py::tuple>(value)) {
        return ConvertPyList(key, py::cast<py::list>(value));
    } else if (py::isinstance<py::dict>(value)) {
        return value.cast<std::map<int64_t, int64_t>>();
    }
    throw py::type_error("Unsupported value type for key: " + key);
}

std::map<std::string, std::any> ConvertPyDictToCppMap(const py::dict& values)
{
    std::map<std::string, std::any> cpp_values;
    for (auto item : values) {
        std::string key = py::str(item.first);
        py::object value = py::reinterpret_borrow<py::object>(item.second);
        if (py::isinstance<py::dict>(value)) {
            // Check if key already has _by_func or _by_label suffix
            // If so, directly cast to the appropriate type without splitting
            bool has_by_func_suffix = (key.size() > 9 && key.substr(key.size() - 9) == "_by_func");
            bool has_by_label_suffix = (key.size() > 10 && key.substr(key.size() - 10) == "_by_label");
            if (has_by_func_suffix || has_by_label_suffix) {
                // Key already has the suffix, manually convert to string map
                std::map<std::string, int64_t> str_map;
                py::dict dict_val = py::cast<py::dict>(value);
                for (auto dict_item : dict_val) {
                    str_map[py::cast<std::string>(dict_item.first)] = py::cast<int64_t>(dict_item.second);
                }
                cpp_values[key] = str_map;
            } else {
                ConvertPyDict(key, value, cpp_values);
            }
        } else {
            cpp_values[key] = ConvertPyValue(key, value);
        }
    }

    return cpp_values;
}

void BindControllerScope(py::module_& m)
{
    m.def(
        "BeginScope",
        [](const std::string& name, const py::dict& values, const std::string& filename, int lineno) {
            auto cpp_values = ConvertPyDictToCppMap(values);
            ConfigManagerNg::GetInstance().BeginScope(name, std::move(cpp_values), filename.c_str(), lineno);
        },
        py::arg("name"), py::arg("values"), py::arg("filename"), py::arg("lineno"));

    m.def(
        "EndScope",
        [](const std::string& filename, int lineno) {
            ConfigManagerNg::GetInstance().EndScope(filename.c_str(), lineno);
        },
        py::arg("filename") = "default", py::arg("lineno") = -1);

    m.def(
        "SetScope",
        [](const py::dict& values, const std::string& filename, int lineno) {
            auto cpp_values = ConvertPyDictToCppMap(values);
            ConfigManagerNg::GetInstance().SetScope(std::move(cpp_values), filename.c_str(), lineno);
        },
        py::arg("values"), py::arg("filename") = "default", py::arg("lineno") = -1);

    m.def(
        "SetGlobalConfig",
        [](const py::dict& values, const std::string& filename, int lineno) {
            auto cpp_values = ConvertPyDictToCppMap(values);
            ConfigManagerNg::GetInstance().SetGlobalConfig(std::move(cpp_values), filename.c_str(), lineno);
        },
        py::arg("values"), py::arg("filename") = "default", py::arg("lineno") = -1);

    m.def("CurrentScope", []() { return ConfigManagerNg::GetInstance().CurrentScope(); });

    m.def("GlobalScope", []() { return ConfigManagerNg::GetInstance().GlobalScope(); });

    m.def("GetOptionsTree", []() { return ConfigManagerNg::GetInstance().GetOptionsTree(); });
}

py::object AnyToPyObject(const std::any& val)
{
    using Fn = std::function<py::object(const std::any&)>;
    static const std::unordered_map<std::type_index, Fn> table = {
        {typeid(bool), [](const std::any& a) { return py::cast(AnyCast<bool>(a)); }},
        {typeid(int64_t), [](const std::any& a) { return py::cast(AnyCast<int64_t>(a)); }},
        {typeid(double), [](const std::any& a) { return py::cast(AnyCast<double>(a)); }},
        {typeid(std::string), [](const std::any& a) { return py::cast(AnyCast<std::string>(a)); }},
        {typeid(std::vector<int64_t>), [](const std::any& a) { return py::cast(AnyCast<std::vector<int64_t>>(a)); }},
        {typeid(std::vector<std::string>),
         [](const std::any& a) { return py::cast(AnyCast<std::vector<std::string>>(a)); }},
        {typeid(std::vector<std::pair<std::string, std::string>>),
         [](const std::any& a) {
             py::list outer;
             for (const auto& p : AnyCast<std::vector<std::pair<std::string, std::string>>>(a)) {
                 py::list inner;
                 inner.append(p.first);
                 inner.append(p.second);
                 outer.append(inner);
             }
             return outer;
         }},
        {typeid(std::vector<double>), [](const std::any& a) { return py::cast(AnyCast<std::vector<double>>(a)); }},
        {typeid(std::map<int64_t, int64_t>),
         [](const std::any& a) { return py::cast(AnyCast<std::map<int64_t, int64_t>>(a)); }},
        {typeid(std::map<std::string, int64_t>),
         [](const std::any& a) { return py::cast(AnyCast<std::map<std::string, int64_t>>(a)); }},
        {typeid(CubeTile), [](const std::any& a) { return py::cast(AnyCast<CubeTile>(a)); }},
        {typeid(ConvTile), [](const std::any& a) { return py::cast(AnyCast<ConvTile>(a)); }},
        {typeid(ConvBpTile), [](const std::any& a) { return py::cast(AnyCast<ConvBpTile>(a)); }},
        {typeid(DistTile), [](const std::any& a) { return py::str(AnyCast<DistTile>(a).ToString()); }},
    };

    auto it = table.find(std::type_index(val.type()));
    if (it != table.end())
        return it->second(val);

    throw py::type_error("Unsupported config value type: " + std::string(val.type().name()));
}

void BindControllerScopeClasses(py::module_& m)
{
    py::class_<ConfigScope, std::shared_ptr<ConfigScope>>(m, "ConfigScope")
        .def(
            "GetAnyConfig",
            [](const ConfigScope& scope, const std::string& key) -> py::object {
                return AnyToPyObject(scope.GetAnyConfig(key));
            },
            py::arg("key"))
        .def("GetAllConfig",
             [](const ConfigScope& scope) -> py::dict {
                 py::dict result;
                 auto config_map = scope.GetAllConfig();

                 for (const auto& [key, val] : config_map) {
                     try {
                         result[py::str(key)] = AnyToPyObject(val);
                     } catch (const py::type_error& e) {
                         PYPTO_LOGW("Warning: Skipping key '%s' - %s", key.c_str(), e.what());
                     }
                 }
                 return result;
             })
        .def("HasConfig", &ConfigScope::HasConfig, py::arg("key"))
        .def(
            "Type",
            [](const ConfigScope& scope, const std::string& key) -> std::string { return scope.Type(key).name(); },
            py::arg("key"))
        .def("ToString", &ConfigScope::ToString);

    py::class_<CubeTile>(m, "CubeTile")
        .def(py::init<>())
        .def(py::init<const std::array<int64_t, MAX_M_DIM_SIZE>&, const std::array<int64_t, MAX_K_DIM_SIZE>&,
                      const std::array<int64_t, MAX_N_DIM_SIZE>&, bool>(),
             py::arg("m"), py::arg("k"), py::arg("n"), py::arg("enableSplitK") = false)
        .def_readwrite("m", &CubeTile::m)
        .def_readwrite("k", &CubeTile::k)
        .def_readwrite("n", &CubeTile::n)
        .def_readwrite("enableSplitK", &CubeTile::enableSplitK)
        .def("valid", &CubeTile::valid)
        .def("ToString", &CubeTile::ToString)
        .def("__repr__", [](const CubeTile& t) { return t.ToString(); })
        .def("__str__", [](const CubeTile& t) { return t.ToString(); });

    py::class_<ConvTile>(m, "ConvTile")
        .def(py::init<>())
        .def(py::init<const Conv::TileL1Info&, const Conv::TileL0Info&, bool>(), py::arg("tileL1Info"),
             py::arg("tileL0Info") = Conv::TileL0Info(), py::arg("setL0Tile") = false)
        .def_readwrite("tileL1Info", &ConvTile::tileL1Info)
        .def_readwrite("tileL0Info", &ConvTile::tileL0Info)
        .def_readwrite("setL0Tile", &ConvTile::setL0Tile)
        .def("valid", &ConvTile::valid)
        .def("ToString", &ConvTile::ToString)
        .def("__repr__", [](const ConvTile& t) { return t.ToString(); })
        .def("__str__", [](const ConvTile& t) { return t.ToString(); });

    py::class_<ConvBpTile>(m, "ConvBpTile")
        .def(py::init<>())
        .def(py::init<const ConvBp::ConvBpTileL1Info&, const ConvBp::ConvBpTileL0Info&>(), py::arg("tileL1Info"),
             py::arg("tileL0Info"))
        .def_readwrite("tileL1Info", &ConvBpTile::tileL1Info)
        .def_readwrite("tileL0Info", &ConvBpTile::tileL0Info)
        .def("valid", &ConvBpTile::valid)
        .def("ToString", &ConvBpTile::ToString)
        .def("__repr__", [](const ConvBpTile& t) { return t.ToString(); })
        .def("__str__", [](const ConvBpTile& t) { return t.ToString(); });
}

void BindController(py::module_& m)
{
    BindControllerConfig(m);
    BindControllerSetTile(m);
    BindControllerFunction(m);
    BindControllerLoop(m);
    BindControllerUtils(m);
    BindControllerScope(m);
    BindControllerScopeClasses(m);
}
} // namespace pypto
