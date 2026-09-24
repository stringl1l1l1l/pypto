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
 * \file platform.h
 * \brief
 */

#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <map>
#include <set>
#include <memory>
#include <unordered_map>
#include "data_type.h"
#include "pypto_fwk_log.h"
#include "cann_host_runtime.h"

namespace npu::tile_fwk {
// Standard soc_version (chip bin within a soc_series), consistent with CANN platform_config ini naming,
// e.g. Ascend950PR_9589 / Ascend910B2 / Kirin9030. One ini file per soc_version.
enum class DPlatform {
    ASCEND_910B1,
    ASCEND_910B2,
    ASCEND_910B3,
    ASCEND_910B4,
    ASCEND_910_9363,
    ASCEND_950DT_9572,
    ASCEND_950PR_9579,
    ASCEND_950DT_9581,
    ASCEND_950DT_9581X,
    ASCEND_950DT_9582,
    ASCEND_950PR_9589,
    KIRIN_9030,
    KIRIN_X90,

    UNKNOWN_DEVICE,
};

inline DPlatform StringToDPlatform(std::string platform)
{
    const std::unordered_map<std::string, DPlatform> mappings = {
        {"Ascend910B1", DPlatform::ASCEND_910B1},
        {"Ascend910B2", DPlatform::ASCEND_910B2},
        {"Ascend910B3", DPlatform::ASCEND_910B3},
        {"Ascend910B4", DPlatform::ASCEND_910B4},
        {"Ascend910_9363", DPlatform::ASCEND_910_9363},
        {"Ascend950DT_9572", DPlatform::ASCEND_950DT_9572},
        {"Ascend950PR_9579", DPlatform::ASCEND_950PR_9579},
        {"Ascend950DT_9581", DPlatform::ASCEND_950DT_9581},
        {"Ascend950DT_9581X", DPlatform::ASCEND_950DT_9581X},
        {"Ascend950DT_9582", DPlatform::ASCEND_950DT_9582},
        {"Ascend950PR_9589", DPlatform::ASCEND_950PR_9589},
        {"Kirin9030", DPlatform::KIRIN_9030},
        {"KirinX90", DPlatform::KIRIN_X90},
        // Deprecated device_platform formats, kept for compatibility. Legacy values whose names did not
        // match the actual chip bin are mapped to the real soc_version of the ini they used to load:
        // ASCEND_950DT_9579 -> Ascend950DT_9572, ASCEND_950PR_9582 -> Ascend950PR_9589.
        {"ASCEND_910B1", DPlatform::ASCEND_910B1},
        {"ASCEND_910B2", DPlatform::ASCEND_910B2},
        {"ASCEND_910B3", DPlatform::ASCEND_910B3},
        {"ASCEND_910B4", DPlatform::ASCEND_910B4},
        {"ASCEND_910_9363", DPlatform::ASCEND_910_9363},
        {"ASCEND_950DT_9572", DPlatform::ASCEND_950DT_9572},
        {"ASCEND_950DT_9579", DPlatform::ASCEND_950DT_9572},
        {"ASCEND_950PR_9579", DPlatform::ASCEND_950PR_9579},
        {"ASCEND_950DT_9581", DPlatform::ASCEND_950DT_9581},
        {"ASCEND_950DT_9581X", DPlatform::ASCEND_950DT_9581X},
        {"ASCEND_950DT_9582", DPlatform::ASCEND_950DT_9582},
        {"ASCEND_950PR_9582", DPlatform::ASCEND_950PR_9589},
        {"ASCEND_950PR_9589", DPlatform::ASCEND_950PR_9589},
        {"KIRIN_9030", DPlatform::KIRIN_9030},
        {"KIRIN_X90", DPlatform::KIRIN_X90},
    };

    if (mappings.count(platform) != 0) {
        return mappings.at(platform);
    }

    return DPlatform::UNKNOWN_DEVICE;
}

struct CacheInfo {
    size_t l2Size;
    size_t l2LineSize;
    size_t l2HitLatency;
    size_t l2MissExtraLatency;
};

struct MemoryInfo {
    MemoryType type;
    size_t size;
    size_t validSize; // 实际可用size
    CacheInfo cache;
    bool operator==(const MemoryType& s) const { return type == s; }
    MemoryInfo() { type = MemoryType::MEM_UNKNOWN; }
    MemoryInfo(MemoryType memtype, size_t sz) : type(memtype), size(sz) {}
};

struct DataPath {
    MemoryInfo source;
    MemoryInfo destination;
    size_t bindWidth;
};

class Pipe {
public:
private:
};

struct InstVariant {
    std::string raw;
    std::vector<std::string> tokens;
};

enum class InstCategory {
    Unknown,
    VectorAlu,
    CubeMatmul,
    DataMove,
    FixPipe,
    Reduction,
    GatherScatter,
    Compare,
    LayoutTransform,
};

struct MemoryNode {
    MemoryType type;
    std::set<MemoryType> dests;
    void AddDest(const std::shared_ptr<MemoryNode>& to) { dests.insert({to->type}); }
};

struct MemoryGraph {
    std::map<MemoryType, std::shared_ptr<MemoryNode>> nodes;
    void AddPath(MemoryType from, MemoryType to);
    std::shared_ptr<MemoryNode> GetNode(MemoryType type);
    void DFS(MemoryType target, const std::shared_ptr<MemoryNode>& node, std::vector<MemoryType>& candidate,
             std::vector<MemoryType>& paths) const;
    bool FindNearestPath(MemoryType from, MemoryType to, std::vector<MemoryType>& paths) const;
    bool HasDirectPath(MemoryType from, MemoryType to) const;
    void Reset() { nodes.clear(); }
};

class PlatformParser {
public:
    PlatformParser() = default;
    virtual ~PlatformParser() = default;

    virtual bool GetStringVal(const std::string& column, const std::string& key, std::string& val) const = 0;

    bool GetSizeVal(const std::string& column, const std::string& key, size_t& val) const;
    bool GetCCECVersion(std::unordered_map<std::string, std::string>& ccecVersion) const;
    bool FilterCCECVersion(const std::string& key, std::string& coreType) const;
};

class Inst {
public:
    int id = -1;

    size_t cycle_cnt; // costmodle/ooo
    size_t event_cnt; // codegen/ooo
    size_t time_;

    std::string name;                  // Intrinsic_vadd
    std::vector<InstVariant> variants; // "|"后面的datatype
    InstCategory category = InstCategory::Unknown;
};

class Core {
protected:
    std::map<MemoryType, MemoryInfo> memories_;
    std::vector<Inst> instructions_;
    std::string version;
    std::string ccec_version;

private:
    size_t num_ = 0;
    std::vector<Pipe> pipes_;

public:
    virtual ~Core() = default;
    void AddMemory(const MemoryInfo& mem_info) { memories_[mem_info.type] = mem_info; }
    void SetNum(size_t num) { num_ = num; }
    void SetVersion(const std::string& ver) { version = ver; }
    void SetCCECVersion(const std::string& ver) { ccec_version = ver; }

    std::string GetVersion() const { return version; }
    std::string GetCCECVersion() const { return ccec_version; }
    size_t GetNum() const { return num_; }
    size_t GetMemorySize(MemoryType type) const;
};

class AivCore : public Core {
public:
    AivCore() {}

    // [VectorCoreSpec]
    void SetVecFreq(int freq) { vec_freq_ = freq; }
    void SetVectorRegWidth(int width) { vector_reg_width_ = width; }
    void SetPredicateRegWidth(int width) { predicate_reg_width_ = width; }
    void SetWideRegWidth(int width) { wide_reg_width_ = width; }

    std::string Dump() const
    {
        std::stringstream ss;
        ss << "{\n";
        ss << "AIV_CORE_INFO : {\n";
        ss << "    \"VERSION\" : \"" << version << "\",\n";
        ss << "    \"VEC_FREQ\" : " << vec_freq_ << ",\n";
        ss << "    \"VECTOR_REG_WIDTH\" : " << vector_reg_width_ << ",\n";
        ss << "    \"PREDICATE_REG_WIDTH\" : " << predicate_reg_width_ << ",\n";
        ss << "    \"WIDE_REG_WIDTH\" : " << wide_reg_width_ << "\n";
        ss << "},\n";
        ss << "}\n";
        return ss.str();
    };

private:
    int vec_freq_ = 0;
    int vector_reg_width_ = 0;
    int predicate_reg_width_ = 0;
    int wide_reg_width_ = 0;
};

class AicCore : public Core {
public:
    AicCore() {}
    void SetCubeFreq(int freq) { cube_freq_ = freq; }
    void SetFixPipeSupport(bool support) { support_fixpipe_ = support; }
    std::string Dump() const
    {
        std::stringstream ss;
        ss << "{\n";
        ss << "AIC_CORE_INFO : {\n";
        ss << "    \"VERSION\" : \"" << version << "\",\n";
        ss << "    \"CUBE_FREQ\" : " << cube_freq_ << ",\n";
        ss << "    \"SUPPORT_FIXPIPE\" : " << (support_fixpipe_ ? "true" : "false") << "\n";
        ss << "},\n";
        ss << "}\n";
        return ss.str();
    };

private:
    int cube_freq_ = 0;
    bool support_fixpipe_ = false;
};

class AICPU {
public:
    void SetSyncBySW(bool sync) { aicpu_sync_by_sw_ = sync; }
    void SetTSCPUSyncBySW(bool sync) { tscpu_sync_by_sw_ = sync; }

    std::string Dump() const
    {
        std::stringstream ss;
        ss << "{\n";
        ss << "AICPU_INFO : {\n";
        ss << "    \"AICPU_SYNC_BY_SW\" : " << (aicpu_sync_by_sw_ ? "true" : "false") << ",\n";
        ss << "    \"TSCPU_SYNC_BY_SW\" : " << (tscpu_sync_by_sw_ ? "true" : "false") << ",\n";
        ss << "},\n";
        ss << "}\n";
        return ss.str();
    };

private:
    bool aicpu_sync_by_sw_ = false;
    bool tscpu_sync_by_sw_ = false;
    size_t completionCycles_ = 0;
    size_t schedulerCycles_ = 0;
    size_t resolveCycles_ = 0;
    size_t threadsNum_ = 0;
};

// 950 aic_cnt_ = 1, aiv_cnt_ = 2
// 其它芯片corewrap无意义 aic_cnt_ = 1, aiv_cnt_ = 1
class CoreWrap {
private:
    AicCore aic_core_;
    AivCore aiv_core_;
    size_t aic_cnt_;
    size_t aiv_cnt_;
    bool is_cv_mix_ = false;

public:
    size_t GetAICNum() const { return aic_cnt_; }
    size_t GetAIVNum() const { return aiv_cnt_; }
    size_t GetAICMemorySize(MemoryType type) const { return aic_core_.GetMemorySize(type); }
    size_t GetAIVMemorySize(MemoryType type) const { return aiv_core_.GetMemorySize(type); }
    AicCore& GetAICCore() { return aic_core_; };
    AivCore& GetAIVCore() { return aiv_core_; };
    bool IsCVMix() const { return is_cv_mix_; }

    void SetAICCore(const AicCore& core) { aic_core_ = core; }
    void SetAIVCore(const AivCore& core) { aiv_core_ = core; }
    void SetAICNum(size_t num) { aic_cnt_ = num; }
    void SetAIVNum(size_t num) { aiv_cnt_ = num; }
    void SetIsCVMix(const bool& isCVMix) { is_cv_mix_ = isCVMix; }

    std::string Dump() const
    {
        std::stringstream ss;
        ss << "{\n";
        ss << "CORE_WRAP_INFO : {\n";
        ss << "    AIC_CORE : {\n";
        ss << "        \"VERSION\" : \"" << aic_core_.GetVersion() << "\",\n";
        ss << "        \"NUM\" : " << aic_cnt_ << "\n";
        ss << "    },\n";
        ss << "    AIV_CORE : {\n";
        ss << "        \"VERSION\" : \"" << aiv_core_.GetVersion() << "\",\n";
        ss << "        \"NUM\" : " << aiv_cnt_ << "\n";
        ss << "    },\n";
        ss << "},\n";
        ss << "}\n";
        return ss.str();
    }
};

class Die {
private:
    size_t mem_device_ddr_size_ = 0;
    size_t mem_host1_size_ = 0;

    CoreWrap core_wrap_;
    AICPU aicpu_;
    std::vector<DataPath> data_paths_;

    size_t core_wrap_cnt_;
    size_t aicpu_cnt_;
    MemoryGraph memoryGraph_;

public:
    size_t GetMemDeviceDDRSize() const { return mem_device_ddr_size_; }
    size_t GetMemHost1Size() const { return mem_host1_size_; }
    size_t GetCoreWrapNum() const { return core_wrap_cnt_; }

    size_t GetMemoryLimit(MemoryType type) const;

    void SetMemDeviceDDRSize(size_t size) { mem_device_ddr_size_ = size; }
    void SetMemHost1Size(size_t size) { mem_host1_size_ = size; }

    bool SetMemoryPath(const std::vector<std::pair<MemoryType, MemoryType>>& dataPaths);
    void ResetMemoryPath() { memoryGraph_.Reset(); }
    bool FindNearestPath(MemoryType from, MemoryType to, std::vector<MemoryType>& paths) const;
    bool HasDirectPath(MemoryType from, MemoryType to) const;
    const MemoryGraph& GetMemoryGraph() const { return memoryGraph_; }

    std::string Dump() const
    {
        std::stringstream ss;
        ss << "{\n";
        ss << "DIE_INFO : {\n";
        // MEMORY_LIMITS
        ss << "        \"MEMORY_LIMITS\" : {\n";
        ss << "          \"MEM_UB\" : " << GetMemoryLimit(MemoryType::MEM_UB) << ",\n";
        ss << "          \"MEM_L1\" : " << GetMemoryLimit(MemoryType::MEM_L1) << ",\n";
        ss << "          \"MEM_L0A\" : " << GetMemoryLimit(MemoryType::MEM_L0A) << ",\n";
        ss << "          \"MEM_L0B\" : " << GetMemoryLimit(MemoryType::MEM_L0B) << ",\n";
        ss << "          \"MEM_L0C\" : " << GetMemoryLimit(MemoryType::MEM_L0C) << ",\n";
        ss << "          \"MEM_DEVICE_DDR\" : " << mem_device_ddr_size_ << ",\n";
        ss << "          \"MEM_HOST1\" : " << mem_host1_size_ << "\n";
        ss << "        },\n";
        ss << "    },\n";
        ss << "}\n";
        return ss.str();
    }

    // Get下层参数
    CoreWrap& GetCoreWrap() { return core_wrap_; }
    AICPU& GetAICPU() { return aicpu_; }
    AicCore& GetAICCore() { return core_wrap_.GetAICCore(); }
    AivCore& GetAIVCore() { return core_wrap_.GetAIVCore(); }
};

enum class NPUArch { DAV_1001 = 1001, DAV_2201 = 2201, DAV_3510 = 3510, DAV_3003 = 3003, DAV_3113 = 3113, DAV_UNKNOWN };

inline std::string NPUArchToString(NPUArch npu_arch)
{
    switch (npu_arch) {
        case NPUArch::DAV_1001:
            return "DAV_1001";
        case NPUArch::DAV_2201:
            return "DAV_2201";
        case NPUArch::DAV_3510:
            return "DAV_3510";
        case NPUArch::DAV_3003:
            return "DAV_3003";
        case NPUArch::DAV_3113:
            return "DAV_3113";
        default:
            return "UNKNOWN_NPU_ARCH";
    }
}

// Parse an arch value: the __NPU_ARCH__ version number ("3510", "2201", ...) or a legacy
// internal name ("a5" -> 3510, "a2"/"a3" -> 2201). Returns DAV_UNKNOWN when nothing matches.
inline NPUArch ParseNPUArch(const std::string& arch)
{
    if (arch == "3510" || arch == "a5") {
        return NPUArch::DAV_3510;
    }
    if (arch == "2201" || arch == "a2" || arch == "a3") {
        return NPUArch::DAV_2201;
    }
    if (arch == "1001") {
        return NPUArch::DAV_1001;
    }
    if (arch == "3003") {
        return NPUArch::DAV_3003;
    }
    if (arch == "3113") {
        return NPUArch::DAV_3113;
    }
    return NPUArch::DAV_UNKNOWN;
}

inline bool IsLiteNPU(NPUArch arch) { return arch == NPUArch::DAV_3113 || arch == NPUArch::DAV_3003; }

// NPU_ARCH (core architecture) of each standard soc_version, consistent with the NpuArch key
// in CANN platform_config ini, e.g. Ascend950PR_9589 -> 3510, Ascend910B2 -> 2201.
inline NPUArch DPlatformToNPUArch(DPlatform platform)
{
    switch (platform) {
        case DPlatform::ASCEND_910B1:
        case DPlatform::ASCEND_910B2:
        case DPlatform::ASCEND_910B3:
        case DPlatform::ASCEND_910B4:
        case DPlatform::ASCEND_910_9363:
            return NPUArch::DAV_2201;
        case DPlatform::ASCEND_950DT_9572:
        case DPlatform::ASCEND_950PR_9579:
        case DPlatform::ASCEND_950DT_9581:
        case DPlatform::ASCEND_950DT_9581X:
        case DPlatform::ASCEND_950DT_9582:
        case DPlatform::ASCEND_950PR_9589:
            return NPUArch::DAV_3510;
        case DPlatform::KIRIN_9030:
            return NPUArch::DAV_3113;
        case DPlatform::KIRIN_X90:
            return NPUArch::DAV_3003;
        default:
            return NPUArch::DAV_UNKNOWN;
    }
}

enum class SoCAICToAIVCoreRatio { ONE_AIC_TO_ONE_AIV_CORE, ONE_AIC_TO_TWO_AIV_CORE, UNSUPPORTED_AIC_TO_AIV_CORE_RATIO };

inline constexpr size_t ONE_AIC_TO_TWO_AIV_CORE_NUM = 2;

class SoC {
private:
    Die die_;
    NPUArch version_;
    std::string soc_version_;
    std::string soc_series_;
    size_t dies_cnt_;
    size_t ai_core_cnt_;
    size_t cube_core_cnt_;
    size_t vector_core_cnt_;
    size_t ai_cpu_cnt_;
    size_t aicpu_cache_ = static_cast<size_t>(-1);

public:
    void SetDie(const Die& die) { die_ = die; }
    void SetNPUArch(NPUArch version) { version_ = version; }
    void SetNPUArch(const std::string& version);
    void SetSocVersion(const std::string& version) { soc_version_ = version; }
    void SetSocSeries(const std::string& series) { soc_series_ = series; }
    void SetDiesNum(size_t cnt) { dies_cnt_ = cnt; }
    void SetCCECVersion(const std::unordered_map<std::string, std::string>& ver);

    Die& GetDies() { return die_; }
    NPUArch GetNPUArch() const { return version_; }
    size_t GetDiesNum() const { return dies_cnt_; }
    std::string GetSocVersion() const { return soc_version_; }
    std::string GetSocSeries() const { return soc_series_; }
    std::string GetCCECVersion(std::string CoreType);

    SoCAICToAIVCoreRatio GetAICToAIVCoreRatio()
    {
        if (cube_core_cnt_ == vector_core_cnt_) {
            return SoCAICToAIVCoreRatio::ONE_AIC_TO_ONE_AIV_CORE;
        } else if (ONE_AIC_TO_TWO_AIV_CORE_NUM * cube_core_cnt_ == vector_core_cnt_) {
            return SoCAICToAIVCoreRatio::ONE_AIC_TO_TWO_AIV_CORE;
        } else {
            return SoCAICToAIVCoreRatio::UNSUPPORTED_AIC_TO_AIV_CORE_RATIO;
        }
    }

    // SOCINFO
    size_t GetAICPUNum();
    size_t GetAICoreNum() const { return ai_core_cnt_; }
    size_t GetAICCoreNum() const { return cube_core_cnt_; }
    size_t GetAIVCoreNum() const { return vector_core_cnt_; }

    void SetAICPUNum(size_t num) { ai_cpu_cnt_ = num; }
    void SetAICoreNum(size_t num) { ai_core_cnt_ = num; }
    void SetAICCoreNum(size_t num) { cube_core_cnt_ = num; }
    void SetAIVCoreNum(size_t num) { vector_core_cnt_ = num; }

    // Get下层参数
    Die& GetDie() { return die_; }
    CoreWrap& GetCoreWrap() { return die_.GetCoreWrap(); }
    AicCore& GetAICCore() { return GetCoreWrap().GetAICCore(); }
    AivCore& GetAIVCore() { return GetCoreWrap().GetAIVCore(); }

    std::string Dump() const
    {
        std::stringstream ss;
        ss << "{\n";
        ss << "SOC_INFO : {\n";
        ss << "    \"SOC_VERSION\" : \"" << soc_version_ << "\",\n";
        ss << "    \"SOC_SERIES\" : \"" << soc_series_ << "\",\n";
        ss << "    \"NPU_ARCH\" : " << static_cast<int>(version_) << ",\n";
        ss << "    \"DIES_NUM\" : " << dies_cnt_ << ",\n";
        ss << "    \"AI_CPU_NUM\" : " << ai_cpu_cnt_ << ",\n";
        ss << "    \"AI_CORE_NUM\" : " << ai_core_cnt_ << ",\n";
        ss << "    \"CUBE_CORE_NUM\" : " << cube_core_cnt_ << ",\n";
        ss << "    \"VECTOR_CORE_NUM\" : " << vector_core_cnt_ << "\n";
        ss << "},\n";
        ss << "}\n";
        return ss.str();
    }
};

class Cluster {
private:
    SoC SoC_;
    size_t soc_cnt_;

public:
    void SetSoC(const SoC& SoC) { SoC_ = SoC; }
    void SetSoCNum(size_t cnt) { soc_cnt_ = cnt; }

    size_t GetSoCNum() const { return soc_cnt_; }

    // Get下层参数
    SoC& GetSoC() { return SoC_; }

    std::string Dump() const
    {
        std::stringstream ss;
        ss << "{\n";
        ss << "CLUSTER_INFO : {\n";
        ss << "    \"SOC_NUM\" : " << soc_cnt_ << "\n";
        ss << "},\n";
        ss << "}\n";
        return ss.str();
    }
};

class Host {};

class PlatformSocVersionManager {
private:
    PlatformSocVersionManager() = default;
    ~PlatformSocVersionManager() = default;

    std::string platformSocVersion_;

public:
    static PlatformSocVersionManager& Instance()
    {
        static PlatformSocVersionManager instance;
        return instance;
    }

    PlatformSocVersionManager(const PlatformSocVersionManager&) = delete;
    PlatformSocVersionManager& operator=(const PlatformSocVersionManager&) = delete;

    std::string GetPlatformSocVersion() const { return platformSocVersion_; }
    void SetPlatformSocVersion(const std::string& version) { platformSocVersion_ = version; }
};

class Platform {
private:
    Platform();
    ~Platform() = default;

    Cluster cluster_;
    Host host_;
    size_t cluster_cnt_;
    size_t host_cnt_;

public:
    static Platform& Instance();

    Platform(const Platform&) = delete;
    Platform& operator=(const Platform&) = delete;

    void SetCluster(const Cluster& cluster) { cluster_ = cluster; }
    void SetHost(const Host& host) { host_ = host; }
    void SetClusterNum(size_t cnt) { cluster_cnt_ = cnt; }
    void SetHostNum(size_t cnt) { host_cnt_ = cnt; }

    size_t GetClusterNum() const { return cluster_cnt_; }
    size_t GetHostNum() const { return host_cnt_; }

    // Get下层参数
    Cluster& GetCluster() { return cluster_; }
    Host& GetHost() { return host_; }
    SoC& GetSoc() { return cluster_.GetSoC(); }
    Die& GetDie() { return GetSoc().GetDie(); }
    CoreWrap& GetCoreWrap() { return GetDie().GetCoreWrap(); }
    AicCore& GetAICCore() { return GetCoreWrap().GetAICCore(); }
    AivCore& GetAIVCore() { return GetCoreWrap().GetAIVCore(); }

    void SetMemoryLimit(const PlatformParser& parser);
    void LoadPlatformInfo(const PlatformParser& parser);
    void ObtainPlatformInfo();
    void ReloadMemoryPaths(const std::string& archType);

    std::string Dump()
    {
        std::ostringstream ss;
        ss << "{\n";

        // 1. Platform
        ss << "  \"PLATFORM_INFO\" : {\n";
        ss << "    \"CLUSTER_NUM\" : " << cluster_cnt_ << ",\n";
        ss << "    \"HOST_NUM\" : " << host_cnt_ << "\n";
        ss << "  },\n";

        auto appendInlineObject = [&](const std::string& child_dump, bool with_trailing_comma) {
            constexpr size_t kLeftWrapLen = std::char_traits<char>::length("{");
            constexpr size_t kRightWrapLen = std::char_traits<char>::length("}");
            if (child_dump.size() <= kLeftWrapLen + kRightWrapLen) {
                return;
            }
            const size_t inner_len = child_dump.size() - kLeftWrapLen - kRightWrapLen;
            ss.write(child_dump.data() + kLeftWrapLen, static_cast<std::streamsize>(inner_len));
            ss << (with_trailing_comma ? ",\n" : "\n");
        };

        // 2. Cluster
        appendInlineObject(cluster_.Dump(), true);

        // 3. SoC
        appendInlineObject(GetSoc().Dump(), true);

        // 4. Die
        appendInlineObject(GetDie().Dump(), true);

        // 5. CoreWrap
        appendInlineObject(GetCoreWrap().Dump(), true);

        // 6. AIVCore
        appendInlineObject(GetAIVCore().Dump(), true);

        // 7. AICCore
        appendInlineObject(GetAICCore().Dump(), false);

        ss << "}\n";
        return ss.str();
    }
};

// Memory-space keys accepted by GetMemoryLimitForArch().
inline constexpr char kMemorySpaceVec[] = "Vec";
inline constexpr char kMemorySpaceMat[] = "Mat";
inline constexpr char kMemorySpaceLeft[] = "Left";
inline constexpr char kMemorySpaceRight[] = "Right";
inline constexpr char kMemorySpaceAcc[] = "Acc";

// Buffer capacity (bytes) of ``space`` for ``arch``, read from the platform
// ini shipped with the package. Returns 0 when the arch or space key is not
// recognized / the ini cannot be found or parsed.
size_t GetMemoryLimitForArch(const std::string& arch, const std::string& space);
} // namespace npu::tile_fwk
