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
 * \file codegen_litenpu.cpp
 * \brief
 */

#include "codegen_litenpu.h"

#include <fstream>

#include "codegen_op_litenpu.h"
#include "codegen/utils/codegen_utils.h"
#include "codegen/utils/parallel_execute.h"
#include "interface/function/function.h"

namespace npu::tile_fwk {

static std::string GetDtype(DataType dtype)
{
    switch (dtype) {
        case DataType::DT_UINT8:
            return "uint8_t";
        case DataType::DT_UINT16:
            return "uint16_t";
        case DataType::DT_UINT32:
            return "uint32_t";
        case DataType::DT_UINT64:
            return "uint64_t";
        case DataType::DT_INT8:
            return "int8_t";
        case DataType::DT_INT16:
            return "int16_t";
        case DataType::DT_INT32:
            return "int32_t";
        case DataType::DT_INT64:
            return "int64_t";
        case DataType::DT_FP16:
            return "half";
        case DataType::DT_FP32:
            return "float";
        case DataType::DT_BOOL:
            return "bool";
        default:
            return "unknown";
    }
}

std::unordered_map<int, std::string> CodeGenLiteNPU::GenParamsSymbolMap(const SubfuncParam& subFuncParam,
                                                                        std::vector<std::string>& params,
                                                                        std::map<std::string, std::string>& dTypeMap)
{
    auto& tensorInvokeArgs = subFuncParam.tensorsArgs_;

    std::unordered_map<int, std::string> symbolMap;
    std::vector<std::string> paramsInOrder;
    std::unordered_set<std::string> seen;
    auto f = [&paramsInOrder, &seen, &dTypeMap, &symbolMap](size_t offset, auto& invokeArgs) {
        CODEGEN_LOGI("start offset is %zu, arg size is %zu", offset, invokeArgs.size());
        for (size_t i = 0; i < invokeArgs.size(); i++) {
            size_t paramOff = (offset + i);
            uint32_t paramLoc = invokeArgs[i].paramLoc;
            CODEGEN_LOGD("paramLoc %u --> offset %zu", paramLoc, paramOff);
            CODEGEN_LOGI(
                " paramLoc is %u, paramOff is %zu, SymDDRId is %d, SymName is %s, Symbol is %s, dataType is %zu",
                paramLoc, paramOff, invokeArgs[i].symDDRId, invokeArgs[i].symName.c_str(), invokeArgs[i].symbol.c_str(),
                static_cast<size_t>(invokeArgs[i].dataType));
            symbolMap.insert({paramLoc, invokeArgs[i].symbol});
            if (seen.find(invokeArgs[i].symbol) == seen.end()) {
                paramsInOrder.push_back(invokeArgs[i].symbol);
                seen.insert(invokeArgs[i].symbol);
            }
            dTypeMap[invokeArgs[i].symbol] = GetDtype(invokeArgs[i].dataType);
        }
    };

    CODEGEN_LOGI("---  start tensorInvokeArgs paramLoc map ---- ");
    f(0, tensorInvokeArgs);

    if ((Program::GetInstance().GetLastFunction() != nullptr) &&
        (Program::GetInstance().GetLastFunction()->GetDyndevAttribute() != nullptr)) {
        std::map<std::string, int32_t> startArgsOrder;
        int32_t order = 0;
        for (auto t : Program::GetInstance().GetLastFunction()->GetDyndevAttribute()->startArgsInputTensorList) {
            CODEGEN_LOGI("Tensor[%s] already exists in inputTensorList", t.get().GetName().c_str());
            startArgsOrder[t.get().GetName()] = order++;
        }
        startArgsOrder.insert(std::make_pair("workspace", startArgsOrder.size()));
        std::sort(paramsInOrder.begin(), paramsInOrder.end(), [&startArgsOrder](std::string& a, std::string& b) {
            return startArgsOrder.at(a) < startArgsOrder.at(b);
        });
    }

    params = paramsInOrder;
    return symbolMap;
}

void CodeGenLiteNPU::GenCode(Function& topFunc)
{
    COMPILER_LOGI("Start Generate AI_CORE code for topFunc: %s, hash: %s", topFunc.GetMagicName().c_str(),
                  topFunc.GetFunctionHash().c_str());

    compileTasks_.clear();

    std::deque<std::function<void(void)>> tasks;
    for (auto& subFuncPair : topFunc.rootFunc_->programs_) {
        std::function task = [this, subFuncPair, &topFunc]() {
            CODEGEN_LOGI(" ----- subprogram id [%lu] -----", subFuncPair.first);
            auto subFunc = subFuncPair.second;
            if (HandleForAICpuSubFunc(*subFunc)) {
                return;
            }
            bool isCube = subFunc->IsCube();
            CompileInfoLiteNPU compileInfo(topFunc, ctx, subFuncPair, isCube);
            std::ostringstream leafKernelFunc;
            GenFuncBody(*subFunc, topFunc, leafKernelFunc);

            // kirin gm addr replace
            std::string funcStr = GenFuncGlobalCodeAfterReplace(topFunc, subFuncPair, leafKernelFunc.str());
            leafKernelFunc.str("");
            leafKernelFunc << funcStr;
#ifdef BUILD_WITH_CANN
            if (std::getenv(ENV_ASCEND_HOME_PATH.c_str()) != nullptr) {
                GenCodeToBinaryTask(leafKernelFunc, compileInfo, "");

                // kirin json codegen
                std::vector<std::string> inOutParams = GetInOutParams(subFuncPair);
                int blockDim = 1; // NEXTNEXT: currently only support one block dim
                int jsonWorkspaceSize = subFunc->GetStackWorkespaceSize();
                GenConfigJson(compileInfo.GetJsonAbsPath(), compileInfo.GetCCEAbsPath(), compileInfo.GetBinAbsPath(),
                              topFunc.GetMagicNameWithHash(), jsonWorkspaceSize, inOutParams, blockDim);
            }
#endif
            UpdateSubFunc(subFuncPair, compileInfo);
        };
        tasks.push_back(task);
    }
    unsigned threadNum = GetCGThreadNum();
    ParallelExecuteAndWait(threadNum, tasks);

#ifdef BUILD_WITH_CANN
    if (std::getenv(ENV_ASCEND_HOME_PATH.c_str()) != nullptr) {
        ExecuteParallelCompile(topFunc);
    }
#endif
}

void CodeGenLiteNPU::GenFuncBody(Function& subFunc, Function& topFunc, std::ostringstream& oss)
{
    OperationsViewer operationList = subFunc.Operations(false);
    if (operationList.IsEmpty()) {
        CODEGEN_LOGW("operationList from PASS is empty, func magic name: %s, func hash: %s",
                     subFunc.GetMagicName().c_str(), subFunc.GetFunctionHash().c_str());
    }

    CODEGEN_LOGI("TopFunc Type is %s\nFunction to codegen:\n %s\n", topFunc.GetFunctionTypeStr().c_str(),
                 topFunc.Dump().c_str());

    std::shared_ptr<SymbolManager> symbolMgr = std::make_shared<SymbolManager>();
    std::shared_ptr<ForBlockManager> forBlkMgr = std::make_shared<ForBlockManager>(symbolMgr);
    FloatSpecValMgr floatSpecValMgr;
    std::string tileOpSourceRegion;
    tileOpSourceRegion.reserve(CODE_RESERVED_SIZE);
    for (const auto& op : operationList) {
        CODEGEN_LOGI("======================== Op CodeGenNPU Start ========================\nGen OP IS: %s",
                     op.Dump().c_str());
        // Skip ops (VIEW/RESHAPE/ALLOC/...) emit no TileOp, but may carry needAlloc on
        // in/out tensors; still register symbols before ignoring instruction codegen.
        GenAllocForLocalBuffer(op, symbolMgr);
        if (SKIP_OPCODE_FOR_CODEGEN.find(op.GetOpcode()) != SKIP_OPCODE_FOR_CODEGEN.end()) {
            CODEGEN_LOGI("ignore this op\n------------------------ Op CodeGenNPU Finish -----------------------");
            continue;
        }

        floatSpecValMgr.UpdateByOp(op);

        // kirin only supports static function
        topFunc.SetFunctionType(FunctionType::STATIC);
        topFunc.SetUnderDynamicFunction(false);
        CodeGenOpLiteNPU cop({symbolMgr, topFunc, subFunc, op, ctx.isMainBlock, false, forBlkMgr});
        std::string tileOpSourceCode = cop.GenOpCode();
        ASSERT(GenCodeErr::GEN_OP_CODE_FAILED, tileOpSourceCode.find(CG_ERROR) == tileOpSourceCode.npos)
            << "Generate code of op failed, op is " << op.Dump();

        tileOpSourceRegion.append(tileOpSourceCode);

        CODEGEN_LOGI("------------------------ Op CodeGenNPU Finish -----------------------");
    }

    floatSpecValMgr.PrintFloatSpecVal(oss);
    symbolMgr->PrintAddrAllocs(oss);
    symbolMgr->GenUsingList(oss);
    oss << symbolMgr->GenTileTensorDefList() << tileOpSourceRegion;
}

void CodeGenLiteNPU::BuildBaseOptions(std::ostringstream& oss, [[maybe_unused]] const CompileInfo& compileInfo) const
{
    oss << "bisheng -c -O2 -x cce -std=c++17 ";
}

void CodeGenLiteNPU::BuildArchOptions(std::ostringstream& oss, const CompileInfo& compileInfo) const
{
    std::vector<std::string> compileOpts;
    if (ConfigManager::Instance().GetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false)) {
        compileOpts.emplace_back("-DSUPPORT_TILE_TENSOR");
    }

    compileOpts.emplace_back("-D__LITE_NPU"); // kirin macro for tileop
    compileOpts.emplace_back("--cce-aicore-only");
    // standard-version alignment options (position-matched: after
    // --cce-aicore-only, before --cce-aicore-arch). --cce-linker-relax
    // excluded: relaxed binary fails on dav-l300 (sim + real device).
    compileOpts.emplace_back("--cce-long-call=true");
    compileOpts.emplace_back("-fno-jump-tables");
    std::string coreArch = GetCoreArch(compileInfo);
    compileOpts.emplace_back("--cce-aicore-arch=" + coreArch);

    std::string allCompileOpts = JoinString(compileOpts, " ");
    oss << allCompileOpts << " ";
}

std::string CodeGenLiteNPU::GetCoreArch([[maybe_unused]] const CompileInfo& compileInfo) const
{
    if (platform_ == NPUArch::DAV_3113) {
        return "dav-l311";
    } else if (platform_ == NPUArch::DAV_3003) {
        return "dav-l300";
    } else {
        return "dav-l311";
    }
}

void CodeGenLiteNPU::AppendLiteNPUVFOptions(std::ostringstream& oss) const
{
    if (!config::GetPassGlobalConfig(KEY_ENABLE_VF, true)) {
        oss << "--cce-simd-vf-fusion=false ";
        return;
    }

    oss << "--enable-pto-tile-fusion "
        << "-mllvm --tile-fusion-skip-shape-inference=true "
        << "-mllvm --tile-fusion-skip-reduceop-fusion=false "
        << "-mllvm --tile-fusion-skip-legality-check=false "
        << "-Rpass=tile-fusion "
        << "-Rpass-missed=tile-fusion "
        << "-mllvm -cce-vf-fusion-max-candidate-set-threshold=10 "
        << "-mllvm -enable-vexpdif-fusion=false "
        << "--cce-simd-vf-fusion=true";
}

void CodeGenLiteNPU::BuildExtraOptions(std::ostringstream& oss, [[maybe_unused]] const CompileInfo& compileInfo,
                                       const std::string& compileOptions) const
{
    oss << "-mllvm -cce-aicore-jump-expand=true "
        << "-mllvm -cce-aicore-function-stack-size=16384 "
        << "-mllvm -cce-aicore-record-overflow=false "
        << "-mllvm -cce-aicore-addr-transform "
        << "-mllvm -cce-aicore-dcci-insert-for-scalar=false "
        << "--cce-aicore-input-parameter-size=4096 ";
    AppendLiteNPUVFOptions(oss);
    oss << compileOptions << " ";
}

void CodeGenLiteNPU::BuildIncludes(std::ostringstream& oss) const
{
    // used for compiling cce
    std::string ptoTileLibPath = GetPtoTileLibPathByEnv();
    if (!ptoTileLibPath.empty()) {
        oss << "-I" << ptoTileLibPath << " ";
    }

    std::string includePath = GetIncludePathForCompileCCE();
    oss << "-I" << includePath << "/tilefwk "
        << "-I" << includePath << "/tileop "
        << "-I" << includePath << " ";
}

std::vector<std::string> CodeGenLiteNPU::GetInOutParams(std::pair<uint64_t, Function*> subFuncPair)
{
    std::vector<std::string> inOutParams;
    std::map<std::string, std::string> dTypeMap;
    auto symbolMap = GenParamsSymbolMap(subFuncPair.second->GetParameter(), inOutParams, dTypeMap);

    return inOutParams;
}

void CodeGenLiteNPU::GenConfigJson(const std::string& jsonName, const std::string& cppName, const std::string& binName,
                                   const std::string& kernelName, const int& workspaceSize,
                                   const std::vector<std::string>& argNames, const int& blockDim) const
{
    std::ofstream file;
    file.open(jsonName);

    file << "{\n"
         << "   \"kernelFile\": \"" << cppName << "\",\n"
         << "   \"kernelBin\": \"" << binName << "\",\n"
         << "   \"kernelName\": \"" << kernelName + "_main" << "\",\n"
         << "   \"workspaceSize\": " << workspaceSize << ",\n"
         << "   \"blockDim\": " << blockDim << ",\n"
         << "   \"argNames\": [";

    for (size_t i = 0; i < argNames.size(); i++) {
        file << "\"" << argNames[i] << "\"";

        if (i < argNames.size() - 1) {
            file << ", ";
        }
    }

    file << "]\n}";
}

std::string CodeGenLiteNPU::GenFuncGlobalCodeAfterReplace(const Function& func,
                                                          std::pair<uint64_t, Function*> subFuncPair,
                                                          const std::string& subProgramCode)
{
    std::string tpl = R"!!!(
#include "TileOpImpl.h"

extern "C" __global__ [aicore] void ${FunctionName}$_main(${GlobalParams}$) {
    ${SubProgCode}$
}
)!!!";
    std::vector<std::string> inOutParams;
    std::map<std::string, std::string> dTypeMap;
    auto symbolMap = GenParamsSymbolMap(subFuncPair.second->GetParameter(), inOutParams, dTypeMap);

    std::string globalParams = "";
    std::string subParams = "";
    for (auto& p : inOutParams) {
        globalParams += "__gm__ " + dTypeMap[p] + "* " + "__restrict__ " + p + ", ";
        subParams += p + ", ";
    }
    if (subFuncPair.second->GetStackWorkespaceSize() > 0) {
        globalParams += "__gm__ int8_t* __restrict__ " + CODEGEN_LITENPU_WORKSPACE + ", ";
        subParams += CODEGEN_LITENPU_WORKSPACE + ", ";
    }

    SubstMap substMap = {
        {"FunctionName", func.GetMagicNameWithHash()},
        {"ProgramId", std::to_string(subFuncPair.first)},
        {"SubProgCode", subProgramCode},
        {"GlobalParams", globalParams.substr(0, globalParams.length() - 2)},
        {"SubParams", subParams.substr(0, subParams.length() - 2)},
    };

    std::string funCode = StringSubstitute(tpl, substMap);

    // GM replace
    for (auto& ele : symbolMap) {
        std::string oldStr = "RealizedGM" + std::to_string(ele.first) + ".Addr";
        std::string newStr = ele.second;
        size_t pos = 0;
        while ((pos = funCode.find(oldStr, pos)) != std::string::npos) {
            funCode.replace(pos, oldStr.length(), newStr);
            pos += newStr.length();
        }
    }
    return funCode;
}

} // namespace npu::tile_fwk
