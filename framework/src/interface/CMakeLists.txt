# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB _Interface_Srcs
        # Compiler monitor
        compiler_monitor/*.cpp
        # Core infrastructure
        core/*.cpp
        ir/*.cpp
        ir/op/*.cpp
        ir/op/*/*.cpp
        ir/reporter/*.cpp
        ir/transforms/*.cpp
        ir/transforms/utils/*.cpp
        ir/verifier/*.cpp
        # 前端框架
        schema/*.cpp
        configs/*.cpp
        tensor/*.cpp
        program/*.cpp
        function/*.cpp
        interpreter/*.cpp
        utils/*.cpp
        plugin/*.cpp

        # Operation
        operation/*.cpp
        operation/conv/*.cpp
        operation/vector/*.cpp
        operation/distributed/*.cpp

        # Machine
        cache/*.cpp
        machine/*.cpp
        machine/host/*.cpp
        registry/*.cpp
)
# pypto_pro 含嵌套子目录, 需 GLOB_RECURSE 递归收集
file(GLOB_RECURSE _PyptoPro_Srcs
        pypto_pro/*.cpp
)
list(APPEND _Interface_Srcs ${_PyptoPro_Srcs})
#PTO_Fwk_Debug_Source_List(NAME "_Interface_Srcs" LIST ${_Interface_Srcs} DETAIL OFF)

add_library(tile_fwk_interface SHARED)
target_sources(tile_fwk_interface PRIVATE ${_Interface_Srcs})
target_compile_definitions(tile_fwk_interface
        PRIVATE
            $<$<BOOL:${ENABLE_TESTS}>:ENABLE_TESTS>
)
target_include_directories(tile_fwk_interface
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        PRIVATE
            $<BUILD_INTERFACE:${PTO_FWK_SRC_ROOT}/framework/include/pypto_pro>
)
target_link_libraries(tile_fwk_interface
        PUBLIC
            $<BUILD_INTERFACE:json>
            $<BUILD_INTERFACE:c_sec>
            $<BUILD_INTERFACE:$<$<BOOL:${ENABLE_COMPILE_DEPENDENCY_CHECK}>:tile_fwk_utils>>
        PRIVATE
            tile_fwk_intf_pub
            dl rt
            Threads::Threads
            tile_fwk_platform
)

# Profiling uses the public CANN API, without a dependency on adapter-private
# headers. CPU-only and C++ unit-test builds keep the status provider inactive.
if (BUILD_WITH_CANN AND NOT ENABLE_UTEST)
    find_cann_package(msprof REQUIRED)
    target_compile_definitions(tile_fwk_interface PRIVATE BUILD_WITH_CANN)
    target_link_libraries(tile_fwk_interface PRIVATE profapi)
endif ()
PTO_Fwk_AnalysisTargetHeaderFiles(TARGET tile_fwk_interface)
PTO_Fwk_AnalysisTargetSymbols(TARGET tile_fwk_interface)

add_subdirectory(interpreter)
