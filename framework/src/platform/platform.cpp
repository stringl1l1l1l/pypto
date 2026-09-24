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
 * \file platform.cpp
 * \brief
 */

#include <fstream>
#include "tilefwk/platform.h"
#include "parser/platform_parser.h"
#include "parser/internal_parser.h"
#include "interface/utils/string_utils.h"

namespace npu::tile_fwk {
const std::string version = "version";
const std::string socVersionInfo = "SoC_version";
const std::string npuArchInfo = "NpuArch";
const std::string socSeriesInfo = "Short_SoC_version";
const std::string socInfo = "SoCInfo";
const std::string cubeVectorMix = "cube_vector_mix";
const std::string aiCoreCnt = "ai_core_cnt";
const std::string cubeCoreCnt = "cube_core_cnt";
const std::string vectorCoreCnt = "vector_core_cnt";
const std::string aiCpuCnt = "ai_cpu_cnt";
const std::string aiCoreSpec = "AICoreSpec";
const std::string l0aSize = "l0_a_size";
const std::string l0bSize = "l0_b_size";
const std::string l0cSize = "l0_c_size";
const std::string l1Size = "l1_size";
const std::string ubSize = "ub_size";
const std::string btSize = "bt_size";
const std::string fixQuantPreSize = "fb0_size";

constexpr size_t kDefaultFixSize = 1 * 1024;
constexpr size_t kDefaultL0mxSize = 2 * 1024;

const std::unordered_map<std::string, NPUArch> npuArchMap = {
    {"1001", NPUArch::DAV_1001}, {"2201", NPUArch::DAV_2201}, {"3003", NPUArch::DAV_3003},
    {"3113", NPUArch::DAV_3113}, {"3510", NPUArch::DAV_3510},
};

NPUArch StringToNPUArch(const std::string& npuArch)
{
    auto it = npuArchMap.find(npuArch);
    if (it != npuArchMap.end()) {
        PLATFORM_LOGD("Set NpuArch as %s.", npuArch.c_str());
        return it->second;
    }
    return NPUArch::DAV_2201;
}

size_t Core::GetMemorySize(MemoryType type) const
{
    auto it = memories_.find(type);
    if (it != memories_.end()) {
        return it->second.size;
    }
    return 0;
}

size_t Die::GetMemoryLimit(MemoryType type) const
{
    size_t aic_limit = core_wrap_.GetAICMemorySize(type);
    size_t aiv_limit = core_wrap_.GetAIVMemorySize(type);
    if (aic_limit == 0 && aiv_limit == 0) {
        // ERROR Note
        return 0;
    }
    return aic_limit == 0 ? aiv_limit : aic_limit;
}

bool Die::SetMemoryPath(const std::vector<std::pair<MemoryType, MemoryType>>& dataPaths)
{
    for (const auto& pathDesc : dataPaths) {
        if (pathDesc.first != MemoryType::MEM_UNKNOWN && pathDesc.second != MemoryType::MEM_UNKNOWN) {
            memoryGraph_.AddPath(pathDesc.first, pathDesc.second);
        }
    }
    return true;
}

bool Die::FindNearestPath(MemoryType from, MemoryType to, std::vector<MemoryType>& paths) const
{
    auto res = memoryGraph_.FindNearestPath(from, to, paths);
    if (res == true) {
        return true;
    }
    paths.clear();
    return false;
}

bool Die::HasDirectPath(MemoryType from, MemoryType to) const { return memoryGraph_.HasDirectPath(from, to); }

void SoC::SetNPUArch(const std::string& versionStr)
{
    if (StringUtils::StartsWith(versionStr, "DAV_")) {
        version_ = StringToNPUArch(versionStr.substr(0x4));
    } else {
        version_ = StringToNPUArch(versionStr);
    }
}

void SoC::SetCCECVersion(const std::unordered_map<std::string, std::string>& ver)
{
    for (const auto& pair : ver) {
        if (pair.first == "AIC") {
            GetAICCore().SetCCECVersion(pair.second);
        } else if (pair.first == "AIV") {
            GetAIVCore().SetCCECVersion(pair.second);
        }
    }
}

std::string SoC::GetCCECVersion(std::string CoreType)
{
    if (CoreType == "AIC") {
        return GetAICCore().GetCCECVersion();
    } else if (CoreType == "AIV") {
        return GetAIVCore().GetCCECVersion();
    } else {
        return "UNKNOWN_CORE";
    }
}

size_t SoC::GetAICPUNum()
{
    if (aicpu_cache_ != static_cast<size_t>(-1)) {
        return aicpu_cache_;
    }
    size_t aiCpuNum = 0;
    PLATFORM_LOGD("Try to obtain real time aicpu count through aclrtGetDeviceInfo.");
    if (CannHostRuntime::Instance().GetAICPUCnt(aiCpuNum)) {
        aicpu_cache_ = aiCpuNum;
        return aiCpuNum;
    }
    PLATFORM_LOGD("Cannot obtain real time aicpu count,using the initialized value.");
    return ai_cpu_cnt_;
}

void MemoryGraph::AddPath(MemoryType from, MemoryType to)
{
    if (from == to) {
        return;
    }
    std::shared_ptr<MemoryNode> fromNode = GetNode(from);
    std::shared_ptr<MemoryNode> toNode = GetNode(to);
    if ((fromNode == nullptr) || (toNode == nullptr)) {
        return;
    }
    fromNode->AddDest(toNode);
}

std::shared_ptr<MemoryNode> MemoryGraph::GetNode(MemoryType type)
{
    std::shared_ptr<MemoryNode> node;
    if (nodes.count(type) != 0) {
        node = nodes[type];
        return node;
    }
    node = std::make_shared<MemoryNode>();
    if (node == nullptr) {
        return nullptr;
    }
    node->type = type;
    nodes.insert({type, node});
    return node;
}

void MemoryGraph::DFS(MemoryType target, const std::shared_ptr<MemoryNode>& node, std::vector<MemoryType>& candidate,
                      std::vector<MemoryType>& paths) const
{
    for (auto& dest : node->dests) {
        if (std::find(candidate.begin(), candidate.end(), dest) != candidate.end()) {
            continue;
        }
        candidate.push_back(dest);
        if (dest != target) {
            DFS(target, nodes.at(dest), candidate, paths);
            candidate.pop_back();
            continue;
        }
        if ((!paths.empty()) && (paths.size() <= candidate.size())) {
            candidate.pop_back();
            continue;
        }
        paths.clear();
        for (auto& t : candidate) {
            paths.push_back(t);
        }
        candidate.pop_back();
    }
}

bool MemoryGraph::FindNearestPath(MemoryType from, MemoryType to, std::vector<MemoryType>& paths) const
{
    if (nodes.count(from) == 0) {
        return false;
    }
    if (nodes.count(to) == 0) {
        return false;
    }
    std::vector<MemoryType> candidate = {from};
    paths.clear();
    const auto it = nodes.find(from);
    DFS(to, it->second, candidate, paths);
    return true;
}

bool MemoryGraph::HasDirectPath(MemoryType from, MemoryType to) const
{
    auto it = nodes.find(from);
    if (it == nodes.end()) {
        return false;
    }
    return it->second->dests.count(to) > 0;
}

Platform& Platform::Instance()
{
    static Platform instance;
    return instance;
}

void Platform::SetMemoryLimit(const PlatformParser& parser)
{
    size_t memoryLimit;
    PLATFORM_LOGD("Start set memory limit.");
    if (parser.GetSizeVal(aiCoreSpec, l0aSize, memoryLimit)) {
        GetAICCore().AddMemory(MemoryInfo(MemoryType::MEM_L0A, memoryLimit));
    }
    if (parser.GetSizeVal(aiCoreSpec, l0bSize, memoryLimit)) {
        GetAICCore().AddMemory(MemoryInfo(MemoryType::MEM_L0B, memoryLimit));
    }
    if (parser.GetSizeVal(aiCoreSpec, l0cSize, memoryLimit)) {
        GetAICCore().AddMemory(MemoryInfo(MemoryType::MEM_L0C, memoryLimit));
    }
    if (parser.GetSizeVal(aiCoreSpec, l1Size, memoryLimit)) {
        GetAIVCore().AddMemory(MemoryInfo(MemoryType::MEM_L1, memoryLimit));
    }
    if (parser.GetSizeVal(aiCoreSpec, ubSize, memoryLimit)) {
        GetAIVCore().AddMemory(MemoryInfo(MemoryType::MEM_UB, memoryLimit));
    }
    if (parser.GetSizeVal(aiCoreSpec, btSize, memoryLimit)) {
        GetAIVCore().AddMemory(MemoryInfo(MemoryType::MEM_BT, memoryLimit));
    }
    if (parser.GetSizeVal(aiCoreSpec, fixQuantPreSize, memoryLimit)) {
        GetAIVCore().AddMemory(MemoryInfo(MemoryType::MEM_FIX_QUANT_PRE, memoryLimit));
    }
    // 插桩
    GetAIVCore().AddMemory(MemoryInfo(MemoryType::MEM_FIX, kDefaultFixSize));
    GetAICCore().AddMemory(MemoryInfo(MemoryType::MEM_L0AMX, kDefaultL0mxSize));
    GetAICCore().AddMemory(MemoryInfo(MemoryType::MEM_L0BMX, kDefaultL0mxSize));
}

void Platform::LoadPlatformInfo(const PlatformParser& parser)
{
    std::string archType;
    std::string socVersion;
    std::string socSeries;
    std::string cvMix;
    std::unordered_map<std::string, std::string> versionInfo;
    PLATFORM_LOGD("Start load platform info.");
    if (parser.GetStringVal(version, npuArchInfo, archType)) {
        GetSoc().SetNPUArch(archType);
    }
    if (parser.GetStringVal(version, socVersionInfo, socVersion)) {
        GetSoc().SetSocVersion(socVersion);
    }
    if (parser.GetStringVal(version, socSeriesInfo, socSeries)) {
        GetSoc().SetSocSeries(socSeries);
    }
    if (IsLiteNPU(GetSoc().GetNPUArch()) &&
        parser.GetStringVal(socInfo, cubeVectorMix, cvMix)) { // for litenpu ini CV mix, cloudnpu ini uses different key
        GetSoc().GetCoreWrap().SetIsCVMix(cvMix == cubeVectorMix);
    }
    if (parser.GetCCECVersion(versionInfo)) {
        GetSoc().SetCCECVersion(versionInfo);
    }
    size_t coreNum;
    if (parser.GetSizeVal(socInfo, aiCoreCnt, coreNum)) {
        GetSoc().SetAICoreNum(coreNum);
    }
    if (parser.GetSizeVal(socInfo, cubeCoreCnt, coreNum)) {
        GetSoc().SetAICCoreNum(coreNum);
    }
    if (parser.GetSizeVal(socInfo, vectorCoreCnt, coreNum)) {
        GetSoc().SetAIVCoreNum(coreNum);
    }
    if (parser.GetSizeVal(socInfo, aiCpuCnt, coreNum)) {
        GetSoc().SetAICPUNum(coreNum);
    }
    SetMemoryLimit(parser);
    std::vector<std::pair<MemoryType, MemoryType>> dataPath;
    InternalParser internalParser = InternalParser(archType);
    PLATFORM_LOGD("Start obtaining data path.");
    if (internalParser.LoadInternalInfo()) {
        if (internalParser.GetDataPath(dataPath)) {
            GetDie().SetMemoryPath(dataPath);
        }
    }
}

Platform::Platform()
{
    PLATFORM_LOGD("Start initializing platform.");
    ObtainPlatformInfo();
    PLATFORM_LOGD("Initialized platform.");
}

void Platform::ObtainPlatformInfo()
{
    static bool initialized = false;
    if (initialized) {
        return;
    }
    std::string socVersion;
    std::string platformSocVersion;
    std::unique_ptr<PlatformParser> parser;
    PLATFORM_LOGD("Start obtaining platform info.");
    platformSocVersion = PlatformSocVersionManager::Instance().GetPlatformSocVersion();
    if (!platformSocVersion.empty()) {
        PLATFORM_LOGD("User specified socVersion:%s, use specified socVersion.", platformSocVersion.c_str());
        parser = std::make_unique<INIParser>(platformSocVersion);
    } else if (CannHostRuntime::Instance().GetSocVersion(socVersion)) {
        PLATFORM_LOGD("Obtain platform through cann package(socVersion:%s), use runtime function.", socVersion.c_str());
        parser = std::make_unique<CmdParser>();
    } else {
        PLATFORM_LOGD("Cannot obtain platform through cann package, use simulation info.");
        parser = std::make_unique<INIParser>();
    }
    PLATFORM_LOGD("Try to load platform info.");
    LoadPlatformInfo(*parser);
    PLATFORM_LOGD("Loaded platform info.");
    initialized = true;
}

void Platform::ReloadMemoryPaths(const std::string& archType)
{
    PLATFORM_LOGD("Reload memory paths for arch: %s.", archType.c_str());
    GetDie().ResetMemoryPath();
    if (archType.empty()) {
        PLATFORM_LOGD("Empty archType, memory graph cleared.");
        return;
    }
    InternalParser internalParser(archType);
    std::vector<std::pair<MemoryType, MemoryType>> dataPath;
    if (internalParser.LoadInternalInfo() && internalParser.GetDataPath(dataPath)) {
        GetDie().SetMemoryPath(dataPath);
        PLATFORM_LOGD("Reloaded memory paths for arch: %s.", archType.c_str());
    } else {
        PLATFORM_LOGW("Failed to reload memory paths for arch: %s.", archType.c_str());
    }
}

size_t GetMemoryLimitForArch(const std::string& arch, const std::string& space)
{
    // Arch (machine-level legacy name kept as pypto_pro API) -> standard soc_version ini.
    // PyPTO Pro only supports A5 (NPU_ARCH 3510, Ascend950 series).
    static const std::unordered_map<NPUArch, std::string> archToIni = {
        {NPUArch::DAV_3510, "Ascend950PR_9579"},
    };
    const auto iniIt = archToIni.find(ParseNPUArch(arch));
    if (iniIt == archToIni.end()) {
        return 0;
    }
    // Space key (kMemorySpace*) -> ini key.
    static const std::unordered_map<std::string, std::string> spaceKeys = {
        {kMemorySpaceVec, ubSize},    {kMemorySpaceMat, l1Size},  {kMemorySpaceLeft, l0aSize},
        {kMemorySpaceRight, l0bSize}, {kMemorySpaceAcc, l0cSize},
    };
    const auto keyIt = spaceKeys.find(space);
    if (keyIt == spaceKeys.end()) {
        return 0;
    }
    size_t memoryLimit = 0;
    try {
        INIParser parser(iniIt->second);
        if (!parser.GetSizeVal(aiCoreSpec, keyIt->second, memoryLimit)) {
            return 0;
        }
    } catch (const std::exception& e) {
        PLATFORM_LOGW("GetMemoryLimitForArch(%s) failed: %s.", arch.c_str(), e.what());
        return 0;
    }
    return memoryLimit;
}
} // namespace npu::tile_fwk
