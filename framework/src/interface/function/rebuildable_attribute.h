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
 * \file rebuildable_attribute.h
 * \brief
 */

#pragma once

#include <type_traits>
#include <algorithm>
#include <cstddef>
#include <memory>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "tilefwk/symbolic_scalar.h"
#include "interface/utils/entry_registrar.h"

namespace npu::tile_fwk {
namespace dynamic {
class DevAscendFunction;
}

class Function;

struct RebuildableAttributeBase {
    virtual void Rebuild(Function* func);
    virtual void Reset(void* data);
    virtual ~RebuildableAttributeBase() = default;
    virtual bool AllowRead() const { return true; }
    virtual bool AllowWrite() const { return true; }
};

template <typename T>
struct RebuildableAttribute : RebuildableAttributeBase {
    virtual void Reset(void* dataPtr) override
    {
        FE_ASSERT(this->AllowWrite());
        data = *(T*)dataPtr;
    }
    const T& Get() const
    {
        FE_ASSERT(this->AllowRead());
        return data;
    }

protected:
    T data;
};

class RebuildableAttributeManager {
public:
    static EntryRegistrarGroup& GetRegistrarGroup();

    static RebuildableAttributeManager& GetInstance();

    template <typename T>
    T* GetAttr(Function* func)
    {
        static_assert(std::is_base_of_v<RebuildableAttributeBase, T>, "T must inherit from RebuildableAttributeBase");
        if (attrDict_.find(func) == attrDict_.end()) {
            InitAttrsForFunc(func);
        }
        std::shared_ptr<RebuildableAttributeBase> attrBase = attrDict_[func][typeid(T).name()];
        std::shared_ptr<T> attr = std::static_pointer_cast<T>(attrBase);
        return attr.get();
    }

    template <typename T>
    void ResetAttr(Function* func, void* data)
    {
        GetAttr<T>(func)->Reset(data);
    }

    template <typename T>
    void BuildAttr(Function* func)
    {
        GetAttr<T>(func)->Build(func);
    }

    void InitAttr(Function* func, const std::string& name, std::shared_ptr<RebuildableAttributeBase> base)
    {
        attrDict_[func][name] = base;
    }

    void Clear() { attrDict_.clear(); }

    RebuildableAttributeManager() = default;

private:
    void InitAttrsForFunc(Function* func);

    std::unordered_map<Function*, std::unordered_map<std::string, std::shared_ptr<RebuildableAttributeBase>>> attrDict_;
};

struct RebuildableAttrInitContext {
    RebuildableAttributeManager* manager;
    Function* func;
};

#define RBUILDABLE_ATTRIBUTE_REGISTER(TyAttr)                                \
    static void Entry##TyAttr(void* data)                                    \
    {                                                                        \
        auto ctx = reinterpret_cast<RebuildableAttrInitContext*>(data);      \
        auto ptr = std::make_shared<TyAttr>();                               \
        auto base = std::static_pointer_cast<RebuildableAttributeBase>(ptr); \
        const std::string name = typeid(TyAttr).name();                      \
        ctx->manager->InitAttr(ctx->func, name, base);                       \
    }                                                                        \
    static EntryRegistrarNode node##TyAttr(RebuildableAttributeManager::GetRegistrarGroup(), Entry##TyAttr, #TyAttr)

struct RebuildableRequiresSimt : RebuildableAttribute<bool> {
    RebuildableRequiresSimt() { data = false; }
};

struct RebuildableMultiIterNoOverlap : RebuildableAttribute<std::unordered_set<int>> {
    void Mark(int rawMagic) { data.insert(rawMagic); }
    bool Has(int rawMagic) const { return data.count(rawMagic) != 0; }
};

} // namespace npu::tile_fwk
