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
 * \file memory_path_utils.h
 * \brief
 */

#ifndef TILE_FWK_MEMORY_PATH_UTILS_H
#define TILE_FWK_MEMORY_PATH_UTILS_H

#include <functional>
#include <map>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "interface/function/function.h"
#include "interface/operation/attribute.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/checker_utils.h"
#include "tilefwk/data_type.h"
#include "tilefwk/tilefwk.h"

// 本头文件内的公共函数从 AssignMemoryType 与 legacy::AssignMemoryType 两版本中完全一致的实现提取而来。
// APASS_LOG_* 宏展开依赖 MODULE_NAME：当前包含本头文件的 4 个 TU（assign_memory_type*.cpp、
// convert_op_inserter*.cpp）均在包含本头文件之后定义了相同的 MODULE_NAME "AssignMemoryType"，
// 此处兜底定义保证宏展开结果与拆分前完全一致，日志输出不变。
#ifndef MODULE_NAME
#define MODULE_NAME "AssignMemoryType"
#endif
#ifndef RETURN_IF_NOT_SUCCESS
#define RETURN_IF_NOT_SUCCESS(expr)                \
    do {                                           \
        Status assignMemoryReturnStatus = (expr);  \
        if (assignMemoryReturnStatus != SUCCESS) { \
            return assignMemoryReturnStatus;       \
        }                                          \
    } while (0)
#endif

namespace npu::tile_fwk {

class MemoryPathUtils {
public:
    using OutputRequirementResolver = std::function<MemoryType(const LogicalTensorPtr&)>;

    static bool IsSpecialDirectMemoryPath(MemoryType from, MemoryType to);

    static bool IsDifferentKnownRequirement(MemoryType requirement, MemoryType targetType);

    static bool ShouldUseDdrForSpecialPath(bool hasParallelDifferentRequirement, MemoryType from, MemoryType to);

    // UB2L1约束：当前op的copy_in_l1_padding_mode为MX_PADDING_MODE时，取拷入源tensor（view输入）
    // 的dynValidShape（validK，K维由copy_in_l1_k_index指示位于第0/1维）：validK非concrete直接
    // 回退经DDR搬运（MX补齐语义仅由DDR路径TLoad支持）；validK为concrete且按64（MX scale块
    // 大小）对齐时无需K向补齐，允许UB->L1直连，否则同样回退DDR。
    static bool IsMxPaddingMode(const Operation& operation);

    // 判断tensor的消费者中是否存在需回退DDR的MX_PADDING_MODE拷入op（validK非concrete或
    // 未按64对齐），用于UB2L1路径回退DDR。
    static bool HasMxPaddingModeConsumer(const LogicalTensorPtr& tensor);

    // 在判断直接搬运路径冲突前，解析 view/assemble 语义 op 背后的有效消费者需求。
    // OP_VIEW/OP_SLICE: 通过 ViewOpAttribute.GetTo() 获取目的地类型。
    // OP_ASSEMBLE/OP_CONTRACT: 若处于特殊直连路径且 output 有下游消费者，推断 targetType；
    //                          无下游消费者（如 outcast）则保留 output MemoryTypeOriginal。
    // resolver 的语义与阶段相关：AssignMemoryType 阶段可递归穿透 view 消费者，
    // ConvertInserter 阶段应基于 tensorTobeMap 中已经规划好的直接需求判断。
    static MemoryType ResolveEffectiveConsumerRequirement(Operation* consumerOp, MemoryType directRequirement,
                                                          MemoryType targetType,
                                                          const OutputRequirementResolver& resolveOutputRequirement);

    // ==================== 以下为 AssignMemoryType（含 legacy）两版本完全一致的公共实现 ====================
    // 非模板函数定义在 memory_path_utils.cpp；依赖 inserter 的函数以模板形式内联定义，
    // InserterT 分别为 ConvertInserter / legacy::ConvertInserter（二者被依赖方法实现完全一致）。

    static Status GetFirstInputOutputIfOpcode(Operation& operation, Opcode expectedOpcode, const std::string& action,
                                              LogicalTensorPtr& input, LogicalTensorPtr& output, bool& shouldHandle);

    // 特殊进阶数据通路，不满足特定条件时回退到通过DDR搬运：L0C2L1, L0C2UB, UB2L1
    static bool IsAdvancedMemoryPath(MemoryType from, MemoryType to);

    static bool IsAssembleProducer(Operation* operation);

    static void CollectProducerAIVFlags(Operation* op, std::vector<bool>& isProducerVector);

    static void CollectConsumerAICFlags(Operation* op, std::vector<bool>& isConsumerCube);

    static bool FitsAssembleOutputMemoryLimit(const LogicalTensorPtr& output, MemoryType memoryType);

    static bool FitsTensorInUb(const LogicalTensorPtr& tensor);

    static bool ExceedsMemoryLimit(const LogicalTensorPtr& tensor, size_t threshold);

    static bool IsOversizedLocalBuffer(const LogicalTensorPtr& tensor, MemoryType memoryType, bool useStrictUbLimit,
                                       bool allowL1Fallback);

    static bool IsDynamicReshape(Operation& operation, const LogicalTensorPtr& output);

    static bool CheckUBTileShape(const LogicalTensorPtr& moveTensor);

    static bool IsDimMultiple(const Shape& shape1, const Shape& shape2);

    static bool CheckInnerAxisC0Size(const LogicalTensorPtr& input, const LogicalTensorPtr& output);

    static Status SyncTensorToBe(Function& function);

    static Status SetOriginalChecked(const LogicalTensorPtr& tensor, MemoryType memoryType,
                                     const std::string& reason = "unknown", bool allowOverride = false);

    static void ForceSetOriginal(const LogicalTensorPtr& tensor, MemoryType memoryType,
                                 const std::string& reason = "unknown");

    static bool CanUseUbForReshape(const LogicalTensorPtr& input, const LogicalTensorPtr& output,
                                   MemoryType inputRequirement, MemoryType outputOriginal);

    static Status AssignInOutCastMemoryTypes(Function& function);

    template <typename InserterT>
    static Status SetRequirementChecked(InserterT& inserter, const LogicalTensorPtr& tensor, Operation& operation,
                                        MemoryType memoryType, const std::string& reason, bool allowOverride = false)
    {
        std::string context = reason.empty() ? "unknown" : reason;
        if (tensor == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor,
                              "SetRequirementChecked failed because tensor is null for operation %s[%d], reason: %s.",
                              operation.GetOpcodeStr().c_str(), operation.GetOpMagic(), context.c_str());
            return FAILED;
        }
        if (!tensor->HasConsumer(operation)) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Operation %s[%d] is not a consumer of tensor %d, reason: %s.",
                              operation.GetOpcodeStr().c_str(), operation.GetOpMagic(), tensor->GetMagic(),
                              context.c_str());
            return FAILED;
        }
        bool hasRequirement = inserter.HasRequirement(tensor, operation);
        MemoryType currentType = inserter.GetRequirementOrUnknown(tensor, operation);
        if (hasRequirement && currentType != MemoryType::MEM_UNKNOWN && memoryType == MemoryType::MEM_UNKNOWN) {
            return SUCCESS;
        }
        if (currentType != MemoryType::MEM_UNKNOWN && memoryType != MemoryType::MEM_UNKNOWN &&
            currentType != memoryType && !allowOverride) {
            APASS_LOG_WARN_F(
                Elements::Tensor,
                "Skip tensor %d requirement update for operation %s[%d] because current %s conflicts with new %s, "
                "reason: %s.",
                tensor->GetMagic(), operation.GetOpcodeStr().c_str(), operation.GetOpMagic(),
                BriefMemoryTypeToString(currentType).c_str(), BriefMemoryTypeToString(memoryType).c_str(),
                context.c_str());
            return SUCCESS;
        }
        inserter.UpdateTensorTobeMap(tensor, operation, memoryType, context.c_str());
        return SUCCESS;
    }

    template <typename InserterT>
    static void ForceSetRequirement(InserterT& inserter, const LogicalTensorPtr& tensor, Operation& operation,
                                    MemoryType memoryType, const std::string& reason)
    {
        if (tensor != nullptr && memoryType != MemoryType::MEM_UNKNOWN) {
            APASS_LOG_DEBUG_F(Elements::Tensor, "Force tensor[%d] requirement for %s[%d] as %s, reason %s.",
                              tensor->GetMagic(), operation.GetOpcodeStr().c_str(), operation.GetOpMagic(),
                              BriefMemoryTypeToString(memoryType).c_str(), reason.c_str());
        }
        SetRequirementChecked(inserter, tensor, operation, memoryType, reason, true);
    }

    template <typename InserterT>
    static void FillUnknownRequirementsWith(InserterT& inserter, const LogicalTensorPtr& tensor, MemoryType memoryType,
                                            const char* reason)
    {
        if (tensor == nullptr || memoryType == MemoryType::MEM_UNKNOWN) {
            return;
        }
        auto requirements = inserter.GetConsumerRequirements(tensor);
        for (const auto& item : requirements) {
            if (item.second == MemoryType::MEM_UNKNOWN) {
                inserter.UpdateTensorTobeMap(tensor, *item.first, memoryType, reason);
            }
        }
    }

    template <typename InserterT>
    static Status EnsureAllConsumerRequirementsExist(InserterT& inserter, Function& function)
    {
        std::unordered_set<LogicalTensorPtr> visited;
        auto ensureTensor = [&inserter, &visited](const LogicalTensorPtr& tensor) -> Status {
            if (tensor == nullptr || !visited.insert(tensor).second)
                return SUCCESS;
            for (const auto& consumerOp : tensor->GetConsumers()) {
                if (inserter.HasRequirement(tensor, *consumerOp))
                    continue;
                RETURN_IF_NOT_SUCCESS(SetRequirementChecked(inserter, tensor, *consumerOp, MemoryType::MEM_UNKNOWN,
                                                            "EnsureAllConsumerRequirementsExist"));
            }
            return SUCCESS;
        };
        for (auto& op : function.Operations()) {
            for (auto& input : op.iOperand) {
                RETURN_IF_NOT_SUCCESS(ensureTensor(input));
            }
            for (auto& output : op.oOperand) {
                RETURN_IF_NOT_SUCCESS(ensureTensor(output));
            }
        }
        return SUCCESS;
    }

    template <typename InserterT>
    static Status HandleNopMemoryType(InserterT& inserter, Operation& operation)
    {
        LogicalTensorPtr input;
        LogicalTensorPtr output;
        bool shouldHandle = false;
        RETURN_IF_NOT_SUCCESS(GetFirstInputOutputIfOpcode(operation, Opcode::OP_NOP, "Handle OP_NOP memory type", input,
                                                          output, shouldHandle));
        if (!shouldHandle) {
            return SUCCESS;
        }
        MemoryType inputRequirement = inserter.GetRequirementOrUnknown(input, operation);
        MemoryType outputOriginal = output->GetMemoryTypeOriginal();
        if (inputRequirement == MemoryType::MEM_UNKNOWN || outputOriginal == MemoryType::MEM_UNKNOWN) {
            return SUCCESS;
        }
        if (inputRequirement != outputOriginal) {
            ForceSetRequirement(inserter, input, operation, MemoryType::MEM_DEVICE_DDR, "HandleNopMismatchFallbackDdr");
            ForceSetOriginal(output, MemoryType::MEM_DEVICE_DDR, "HandleNopMismatchFallbackDdr");
            return SUCCESS;
        }
        return SUCCESS;
    }

    template <typename InserterT>
    static Status ApplyOtherSpecialOpcodeRules(InserterT& inserter, Function& function)
    {
        for (auto& op : function.Operations()) {
            RETURN_IF_NOT_SUCCESS(HandleNopMemoryType(inserter, op));
        }
        return SUCCESS;
    }

    template <typename InserterT>
    static Status AssignReduceAccInputRequirements(InserterT& inserter, Operation& operation)
    {
        for (auto& tensor : operation.iOperand) {
            RETURN_IF_NOT_SUCCESS(SetRequirementChecked(inserter, tensor, operation, MemoryType::MEM_DEVICE_DDR,
                                                        "AssignReduceAccInputRequirements"));
        }
        return SUCCESS;
    }

    template <typename InserterT>
    static Status AssignOpcodeDefinedMemoryTypes(InserterT& inserter, Operation& operation)
    {
        auto opcode = operation.GetOpcode();
        bool hasSpecialInputRule = opcode == Opcode::OP_REDUCE_ACC ||
                                   OpChecker::check(operation, OpChecker::CalcTypeChecker(OpCalcType::MATMUL));
        const auto& inputsMemType = OpcodeManager::Inst().GetInputsMemType(opcode);
        if (!hasSpecialInputRule) {
            for (size_t i = 0; i < operation.iOperand.size(); ++i) {
                MemoryType inputMemType = (i < inputsMemType.size()) ? inputsMemType[i] : MemoryType::MEM_UNKNOWN;
                RETURN_IF_NOT_SUCCESS(SetRequirementChecked(inserter, operation.iOperand[i], operation, inputMemType,
                                                            "AssignOpcodeDefinedInput"));
            }
        }
        const auto& outputsMemType = OpcodeManager::Inst().GetOutputsMemType(opcode);
        for (size_t i = 0; i < operation.oOperand.size(); ++i) {
            if (i >= outputsMemType.size())
                continue;
            RETURN_IF_NOT_SUCCESS(
                SetOriginalChecked(operation.oOperand[i], outputsMemType[i], "AssignOpcodeDefinedOutput"));
        }
        return SUCCESS;
    }

    template <typename InserterT>
    static MemoryType GetAssembleInputType(InserterT& inserter, Operation& operation)
    {
        if (operation.iOperand.empty() || operation.iOperand.front() == nullptr)
            return MemoryType::MEM_UNKNOWN;
        auto input = operation.iOperand.front();
        MemoryType fromType = inserter.GetRequirementOrUnknown(input, operation);
        return fromType != MemoryType::MEM_UNKNOWN ? fromType : input->GetMemoryTypeOriginal();
    }

    template <typename InserterT>
    static MemoryType GetReshapeInputRequirement(InserterT& inserter, Operation& operation,
                                                 const LogicalTensorPtr& input, MemoryType inputOriginal)
    {
        MemoryType inputRequirement = inserter.GetRequirementOrUnknown(input, operation);
        if (inputRequirement != MemoryType::MEM_UNKNOWN || inputOriginal == MemoryType::MEM_UNKNOWN) {
            return inputRequirement;
        }
        ForceSetRequirement(inserter, input, operation, inputOriginal, "InferReshapeInputOriginal");
        return inputOriginal;
    }

    template <typename InserterT>
    static Status InferReshapeOutputFromRequirement(InserterT& inserter, const LogicalTensorPtr& output,
                                                    MemoryType& outputOriginal)
    {
        if (outputOriginal != MemoryType::MEM_UNKNOWN) {
            return SUCCESS;
        }
        MemoryType outputRequirement = InferUniqueRequirementThroughViewConsumers(inserter, output);
        if (outputRequirement == MemoryType::MEM_UNKNOWN) {
            std::unordered_set<const LogicalTensor*> visitedTensors;
            if (HasRequirementThroughViewConsumers(inserter, output, MemoryType::MEM_UB, visitedTensors)) {
                outputRequirement = MemoryType::MEM_UB;
            }
        }
        if (outputRequirement == MemoryType::MEM_UNKNOWN) {
            return SUCCESS;
        }
        RETURN_IF_NOT_SUCCESS(SetOriginalChecked(output, outputRequirement, "InferReshapeOutputRequirement"));
        outputOriginal = output->GetMemoryTypeOriginal();
        return SUCCESS;
    }

    template <typename InserterT>
    static MemoryType InferUniqueRequirementThroughViewConsumers(InserterT& inserter, const LogicalTensorPtr& tensor)
    {
        std::unordered_set<const LogicalTensor*> visitedTensors;
        return InferUniqueRequirementThroughViewConsumers(inserter, tensor, visitedTensors);
    }

    template <typename InserterT>
    static MemoryType InferUniqueRequirementThroughViewConsumers(
        InserterT& inserter, const LogicalTensorPtr& tensor, std::unordered_set<const LogicalTensor*>& visitedTensors)
    {
        if (tensor == nullptr || !visitedTensors.insert(tensor.get()).second) {
            return MemoryType::MEM_UNKNOWN;
        }
        std::set<MemoryType> candidates;
        auto addCandidate = [&candidates](MemoryType candidate) {
            if (candidate != MemoryType::MEM_UNKNOWN) {
                candidates.insert(candidate);
            }
        };
        auto consumerRequirements = inserter.GetConsumerRequirements(tensor);
        for (const auto& item : consumerRequirements) {
            Operation* consumerOp = item.first;
            addCandidate(item.second);
            if (consumerOp == nullptr || consumerOp->GetOpcode() != Opcode::OP_VIEW) {
                continue;
            }
            auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(consumerOp->GetOpAttribute());
            if (viewOpAttribute != nullptr) {
                addCandidate(viewOpAttribute->GetTo());
            }
            if (consumerOp->oOperand.empty() || consumerOp->oOperand.front() == nullptr) {
                continue;
            }
            auto viewOutput = consumerOp->oOperand.front();
            addCandidate(viewOutput->GetMemoryTypeOriginal());
            addCandidate(InferUniqueRequirementThroughViewConsumers(inserter, viewOutput, visitedTensors));
        }
        if (candidates.size() == 1) {
            return *candidates.begin();
        }
        return MemoryType::MEM_UNKNOWN;
    }

    template <typename InserterT>
    static bool HasRequirementThroughViewConsumers(InserterT& inserter, const LogicalTensorPtr& tensor,
                                                   MemoryType targetRequirement,
                                                   std::unordered_set<const LogicalTensor*>& visitedTensors)
    {
        if (tensor == nullptr || targetRequirement == MemoryType::MEM_UNKNOWN ||
            !visitedTensors.insert(tensor.get()).second) {
            return false;
        }
        auto consumerRequirements = inserter.GetConsumerRequirements(tensor);
        for (const auto& item : consumerRequirements) {
            Operation* consumerOp = item.first;
            if (item.second == targetRequirement) {
                return true;
            }
            if (consumerOp == nullptr || consumerOp->GetOpcode() != Opcode::OP_VIEW) {
                continue;
            }
            auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(consumerOp->GetOpAttribute());
            if (viewOpAttribute != nullptr && viewOpAttribute->GetTo() == targetRequirement) {
                return true;
            }
            if (consumerOp->oOperand.empty() || consumerOp->oOperand.front() == nullptr) {
                continue;
            }
            auto viewOutput = consumerOp->oOperand.front();
            auto viewOutputOriginal = viewOutput->GetMemoryTypeOriginal();
            bool canUseAdvancedPath = IsAdvancedMemoryPath(targetRequirement, viewOutputOriginal) &&
                                      IsDimMultiple(tensor->GetShape(), viewOutput->GetShape());
            if (viewOutput->GetMemoryTypeOriginal() == targetRequirement || canUseAdvancedPath ||
                HasRequirementThroughViewConsumers(inserter, viewOutput, targetRequirement, visitedTensors)) {
                return true;
            }
        }
        return false;
    }

    template <typename InserterT>
    static Status ApplyReshapeMemoryType(InserterT& inserter, Operation& operation, const LogicalTensorPtr& input,
                                         const LogicalTensorPtr& output, bool isDynamic, bool canUseUb)
    {
        if (canUseUb) {
            const char* reason = isDynamic ? "InferDynamicReshapeUb" : "InferStaticReshapeUb";
            ForceSetRequirement(inserter, input, operation, MemoryType::MEM_UB, reason);
            ForceSetOriginal(output, MemoryType::MEM_UB, reason);
            return SUCCESS;
        }
        const char* reason = isDynamic ? "InferDynamicReshapeFallbackDdr" : "InferStaticReshapeFallbackDdr";
        ForceSetRequirement(inserter, input, operation, MemoryType::MEM_DEVICE_DDR, reason);
        ForceSetOriginal(output, MemoryType::MEM_DEVICE_DDR, reason);
        return SUCCESS;
    }

    template <typename InserterT>
    static Status InferViewTypeInput(InserterT& inserter, Operation& operation, const LogicalTensorPtr& input,
                                     const LogicalTensorPtr& output, MemoryType targetType)
    {
        MemoryType inputOriginal = input->GetMemoryTypeOriginal();
        if (targetType != MemoryType::MEM_UNKNOWN && inputOriginal == targetType) {
            ForceSetRequirement(inserter, input, operation, targetType, "InferViewTypeSameMemory");
            ForceSetOriginal(output, targetType, "InferViewTypeSameMemory");
            return SUCCESS;
        }
        ForceSetRequirement(inserter, input, operation, MemoryType::MEM_DEVICE_DDR, "InferViewTypeFallbackDdr");
        ForceSetOriginal(output, MemoryType::MEM_DEVICE_DDR, "InferViewTypeFallbackDdr");
        return SUCCESS;
    }

    template <typename InserterT>
    static MemoryType InferTargetTypeThroughForwardViews(InserterT& inserter, const LogicalTensorPtr& tensor)
    {
        std::unordered_set<LogicalTensorPtr> visitedTensors;
        return InferTargetTypeThroughForwardViews(inserter, tensor, visitedTensors);
    }

    template <typename InserterT>
    static MemoryType InferTargetTypeThroughForwardViews(InserterT& inserter, const LogicalTensorPtr& tensor,
                                                         std::unordered_set<LogicalTensorPtr>& visitedTensors)
    {
        if (tensor == nullptr || !visitedTensors.insert(tensor).second) {
            return MemoryType::MEM_UNKNOWN;
        }
        // 仅当唯一 consumer 为 OP_VIEW 时沿视图链前向推导，规避多分支分歧
        const auto& consumers = tensor->GetConsumers();
        if (consumers.size() != 1) {
            return MemoryType::MEM_UNKNOWN;
        }
        auto consumerOp = *consumers.begin();
        if (consumerOp == nullptr || consumerOp->GetOpcode() != Opcode::OP_VIEW) {
            return MemoryType::MEM_UNKNOWN;
        }
        if (consumerOp->oOperand.empty() || consumerOp->oOperand.front() == nullptr) {
            return MemoryType::MEM_UNKNOWN;
        }
        auto viewOutput = consumerOp->oOperand.front();
        if (viewOutput->GetMemoryTypeOriginal() != MemoryType::MEM_UNKNOWN) {
            return viewOutput->GetMemoryTypeOriginal();
        }
        MemoryType viewOutputRequirement = inserter.TryGetUniqueKnownRequiredType(viewOutput);
        if (viewOutputRequirement != MemoryType::MEM_UNKNOWN) {
            return viewOutputRequirement;
        }
        return InferTargetTypeThroughForwardViews(inserter, viewOutput, visitedTensors);
    }

    template <typename InserterT>
    static Status ResolveTensorMemoryUnknowns(InserterT& inserter, const LogicalTensorPtr& tensor)
    {
        if (tensor == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Resolve tensor memory unknown failed because tensor is null.");
            return FAILED;
        }
        MemoryType original = tensor->GetMemoryTypeOriginal();
        if (original == MemoryType::MEM_UNKNOWN) {
            MemoryType inferredOriginal = InferOriginalFromRequirements(inserter, tensor);
            RETURN_IF_NOT_SUCCESS(SetOriginalChecked(tensor, inferredOriginal, "ResolveMemoryUnknowns"));
            original = tensor->GetMemoryTypeOriginal();
        }
        FillUnknownRequirementsWith(inserter, tensor, original, "ResolveMemoryUnknowns");
        return SUCCESS;
    }

    template <typename InserterT>
    static MemoryType InferOriginalFromRequirements(InserterT& inserter, const LogicalTensorPtr& tensor)
    {
        if (tensor == nullptr) {
            return MemoryType::MEM_DEVICE_DDR;
        }
        auto knownRequirements = inserter.GetKnownRequiredTypes(tensor);
        if (knownRequirements.size() == 1) {
            return *knownRequirements.begin();
        }
        return MemoryType::MEM_DEVICE_DDR;
    }

private:
    // 与 assign_memory_type.h / assign_memory_type_legacy.h 中的同名常量保持一致（拆分副本）
    static constexpr double UB_THRESHOLD_ASSEMBLE = 0.35;
    static constexpr double UB_THRESHOLD_NORMAL = 1.0;
    static constexpr double L1_THRESHOLD = 0.5;
    static constexpr uint16_t L0C_TILE_SIZE = 16;
    static constexpr uint16_t INT8_ALIGN_SIZE = 32;
};

} // namespace npu::tile_fwk

#endif // TILE_FWK_MEMORY_PATH_UTILS_H
