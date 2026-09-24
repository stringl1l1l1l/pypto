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
 * \file rebuildable_attribute.cpp
 * \brief
 */

#include "rebuildable_attribute.h"

#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk {

void RebuildableAttributeBase::Rebuild(Function* func)
{
    (void)func;
    FE_LOGE(InternalError::FE_INNER_ERROR, "Rebuild is not implemented!");
}

void RebuildableAttributeBase::Reset(void* data)
{
    (void)data;
    FE_LOGE(InternalError::FE_INNER_ERROR, "Reset is not implemented!");
}

EntryRegistrarGroup& RebuildableAttributeManager::GetRegistrarGroup()
{
    static EntryRegistrarGroup group;
    return group;
}

RebuildableAttributeManager& RebuildableAttributeManager::GetInstance()
{
    static RebuildableAttributeManager instance;
    return instance;
}

void RebuildableAttributeManager::InitAttrsForFunc(Function* func)
{
    RebuildableAttrInitContext ctx{this, func};
    GetRegistrarGroup().Init(&ctx);
}

std::map<std::string, std::string> RebuildableAttributeManager::DumpAttrs(Function* func) const
{
    std::map<std::string, std::string> result;
    auto funcIt = attrDict_.find(func);
    if (funcIt == attrDict_.end()) {
        return result;
    }
    for (const auto& entry : funcIt->second) {
        auto str = entry.second->DumpValue();
        if (!str.empty()) {
            result[entry.second->Name()] = str;
        }
    }
    return result;
}

RBUILDABLE_ATTRIBUTE_REGISTER(RebuildableRequiresSimt);
RBUILDABLE_ATTRIBUTE_REGISTER(RebuildableMultiIterNoOverlap);

} // namespace npu::tile_fwk
