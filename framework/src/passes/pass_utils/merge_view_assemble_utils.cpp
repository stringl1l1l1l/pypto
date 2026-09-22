/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file merge_view_assemble_utils.cpp
 * \brief dispatcher of view and assemble operation merging
 */

#include "merge_view_assemble_utils.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_utils/merge_view_assemble_legal.h"
#include "passes/pass_utils/merge_view_assemble_token.h"

namespace npu::tile_fwk {
Status MergeViewAssembleUtils::MergeViewAssemble(Function& function)
{
    if (IRContext::Get().AssembleNewLogicalTensor()) {
        return TokenMergeAssembleUtils::MergeViewAssemble(function);
    }
    return LegalMergeAssembleUtils::MergeViewAssemble(function);
}
} // namespace npu::tile_fwk
