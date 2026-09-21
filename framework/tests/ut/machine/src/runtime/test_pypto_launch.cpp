/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <acl/acl_rt.h>
#include <cstdint>
#include <limits>
#include <vector>
#include <gtest/gtest.h>

namespace {

struct QueryResult {
    aclrtStream stream;
    aclrtDevResLimitType resource;
    uint32_t limit;
    aclError status;
};

std::vector<QueryResult> gQueries;
size_t gQueryIndex = 0;

aclError GetStreamResLimitStub(aclrtStream stream, aclrtDevResLimitType resource, uint32_t* limit)
{
    if (gQueryIndex >= gQueries.size()) {
        ADD_FAILURE() << "Unexpected resource query: " << resource;
        return ACL_ERROR_INVALID_PARAM;
    }
    const auto& query = gQueries[gQueryIndex++];
    EXPECT_EQ(stream, query.stream);
    EXPECT_EQ(resource, query.resource);
    EXPECT_NE(limit, nullptr);
    // A failed ACL call need not initialize its output. The caller must report
    // the query error before interpreting the untouched value as a zero budget.
    if (query.status == ACL_SUCCESS && limit != nullptr) {
        *limit = query.limit;
    }
    return query.status;
}

// Include ACL and standard headers above before isolating the real helper here.
// Redirect only this translation unit's query, so neither its inline templates
// nor the stub can interpose on production ACL calls in the combined UT binary.
#define aclrtGetStreamResLimit GetStreamResLimitStub
#include "pypto_launch.h"
#undef aclrtGetStreamResLimit

inline int64_t RequestError(uint32_t budget) { return pypto::EncodeLaunchError(pypto::LaunchError::Request, budget); }

class PyptoLaunchTest : public testing::Test {
protected:
    void SetUp() override
    {
        gQueries.clear();
        gQueryIndex = 0;
    }

    void TearDown() override
    {
        EXPECT_EQ(gQueryIndex, gQueries.size()) << "A required engine was not queried";
        gQueries.clear();
    }

    void ExpectQuery(aclrtDevResLimitType resource, uint32_t limit, aclError status = ACL_SUCCESS)
    {
        gQueries.push_back({stream_, resource, limit, status});
    }

    void ExpectMixedLimits(uint32_t cube, uint32_t vector)
    {
        ExpectQuery(ACL_RT_DEV_RES_CUBE_CORE, cube);
        ExpectQuery(ACL_RT_DEV_RES_VECTOR_CORE, vector);
    }

    // Opaque addresses are never dereferenced; the stub verifies exact forwarding.
    uint8_t streamToken_ = 0;
    aclrtStream stream_ = &streamToken_;
};

TEST_F(PyptoLaunchTest, ErrorEncodingPreservesPythonAbiTagsAndAllDetailBits)
{
    // Pin the wire values independently of the encoder: changing enum values
    // must not silently change the error code reported across the Python boundary.
    EXPECT_EQ(pypto::EncodeLaunchError(pypto::LaunchError::CubeLimit, 0), -0x100000000LL);
    EXPECT_EQ(pypto::EncodeLaunchError(pypto::LaunchError::VectorLimit, 0), -0x200000000LL);
    EXPECT_EQ(pypto::EncodeLaunchError(pypto::LaunchError::CubeQuery, 0), -0x300000000LL);
    EXPECT_EQ(pypto::EncodeLaunchError(pypto::LaunchError::VectorQuery, 0), -0x400000000LL);
    EXPECT_EQ(pypto::EncodeLaunchError(pypto::LaunchError::Request, 0), -0x500000000LL);
    constexpr uint32_t maxDetail = std::numeric_limits<uint32_t>::max();
    EXPECT_EQ(pypto::EncodeLaunchError(pypto::LaunchError::CubeLimit, maxDetail), -0x1FFFFFFFFLL);
    EXPECT_EQ(pypto::EncodeLaunchError(pypto::LaunchError::VectorLimit, maxDetail), -0x2FFFFFFFFLL);
    EXPECT_EQ(pypto::EncodeLaunchError(pypto::LaunchError::CubeQuery, maxDetail), -0x3FFFFFFFFLL);
    EXPECT_EQ(pypto::EncodeLaunchError(pypto::LaunchError::VectorQuery, maxDetail), -0x4FFFFFFFFLL);
    EXPECT_EQ(pypto::EncodeLaunchError(pypto::LaunchError::Request, maxDetail), -0x5FFFFFFFFLL);
}

TEST_F(PyptoLaunchTest, CubeOnlyAcceptsUpToBudgetWithoutQueryingVector)
{
    // No vector response is supplied: even a successful extra query is a failure.
    for (uint32_t request : {1U, 5U}) {
        SCOPED_TRACE(request);
        ExpectQuery(ACL_RT_DEV_RES_CUBE_CORE, 5);
        EXPECT_EQ((pypto::ResolveLaunchBlockDim<1, 0>(request, stream_)), request);
    }
}

TEST_F(PyptoLaunchTest, CubeOnlyRejectsRequestsAboveBudget)
{
    ExpectQuery(ACL_RT_DEV_RES_CUBE_CORE, 5);
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<1, 0>(20, stream_)), RequestError(5));
}

TEST_F(PyptoLaunchTest, VectorOnlyAcceptsUpToBudgetWithoutQueryingCube)
{
    for (uint32_t request : {1U, 7U}) {
        SCOPED_TRACE(request);
        ExpectQuery(ACL_RT_DEV_RES_VECTOR_CORE, 7);
        EXPECT_EQ((pypto::ResolveLaunchBlockDim<0, 1>(request, stream_)), request);
    }
}

TEST_F(PyptoLaunchTest, VectorOnlyRejectsRequestsAboveBudget)
{
    ExpectQuery(ACL_RT_DEV_RES_VECTOR_CORE, 7);
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<0, 1>(20, stream_)), RequestError(7));
}

TEST_F(PyptoLaunchTest, MixedBlocksRejectCubeBottleneckWithBudgetDetail)
{
    ExpectMixedLimits(4, 20);
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<1, 2>(20, stream_)), RequestError(4));
}

TEST_F(PyptoLaunchTest, MixedBlocksRejectOddVectorBudgetWithoutRoundingUp)
{
    // Nine vectors can complete only four pairs; a fifth cube would deadlock.
    ExpectMixedLimits(20, 9);
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<1, 2>(20, stream_)), RequestError(4));
}

TEST_F(PyptoLaunchTest, MixedBlocksRejectWhenExactlyOneCompletePairFits)
{
    ExpectMixedLimits(1, 2);
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<1, 2>(20, stream_)), RequestError(1));
}

TEST_F(PyptoLaunchTest, MixedBlocksPreserveSmallerRequest)
{
    ExpectMixedLimits(20, 40);
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<1, 2>(3, stream_)), 3);
}

TEST_F(PyptoLaunchTest, TopologyUsesConfiguredCoreCounts)
{
    // A hardcoded 1:2 ratio would fail valid compiler-provided topologies.
    ExpectMixedLimits(8, 5);
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<1, 1>(12, stream_)), RequestError(5));
    ExpectMixedLimits(5, 20);
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<2, 3>(12, stream_)), RequestError(2));
    ExpectMixedLimits(20, 5);
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<2, 3>(12, stream_)), RequestError(1));
}

TEST_F(PyptoLaunchTest, RequestEqualToBudgetIsAccepted)
{
    // The budget itself must launch; only strictly larger requests are rejected.
    ExpectQuery(ACL_RT_DEV_RES_CUBE_CORE, 8);
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<1, 0>(8, stream_)), 8);
    ExpectMixedLimits(8, 8);
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<1, 2>(4, stream_)), 4);
}

TEST_F(PyptoLaunchTest, Uint32MaximumRemainsAPositiveBlockCount)
{
    // The return type must distinguish the whole uint32_t request range from errors.
    constexpr uint32_t maxBlocks = std::numeric_limits<uint32_t>::max();
    ExpectQuery(ACL_RT_DEV_RES_CUBE_CORE, maxBlocks);
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<1, 0>(maxBlocks, stream_)), 4294967295LL);
}

TEST_F(PyptoLaunchTest, AutoSentinelUsesTheFullStreamBudget)
{
    // The host's auto sentinel takes the whole budget of every engine the
    // kernel uses, mixed blocks still bound by the tighter engine.
    ExpectMixedLimits(4, 20);
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<1, 2>(pypto::kAutoBlockDim, stream_)), 4);
    ExpectMixedLimits(20, 9);
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<1, 2>(pypto::kAutoBlockDim, stream_)), 4);
    ExpectMixedLimits(1, 2);
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<1, 2>(pypto::kAutoBlockDim, stream_)), 1);
    ExpectQuery(ACL_RT_DEV_RES_CUBE_CORE, 5);
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<1, 0>(pypto::kAutoBlockDim, stream_)), 5);
    ExpectQuery(ACL_RT_DEV_RES_VECTOR_CORE, 7);
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<0, 1>(pypto::kAutoBlockDim, stream_)), 7);
    ExpectMixedLimits(8, 5);
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<1, 1>(pypto::kAutoBlockDim, stream_)), 5);
    ExpectMixedLimits(5, 20);
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<2, 3>(pypto::kAutoBlockDim, stream_)), 2);
}

TEST_F(PyptoLaunchTest, AutoSentinelWithMaxBudgetsStaysPositive)
{
    constexpr uint32_t maxBlocks = std::numeric_limits<uint32_t>::max();
    ExpectMixedLimits(maxBlocks, maxBlocks);
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<1, 2>(pypto::kAutoBlockDim, stream_)), 2147483647LL);
}

TEST_F(PyptoLaunchTest, ZeroCubeBudgetStopsBeforeVectorQuery)
{
    ExpectQuery(ACL_RT_DEV_RES_CUBE_CORE, 0);
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<1, 2>(12, stream_)), -0x100000000LL);
}

TEST_F(PyptoLaunchTest, ZeroVectorBudgetRejectsVectorOnlyAndMixedBlocks)
{
    ExpectQuery(ACL_RT_DEV_RES_VECTOR_CORE, 0);
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<0, 1>(12, stream_)), -0x200000000LL);
    ExpectMixedLimits(8, 0);
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<1, 2>(12, stream_)), -0x200000000LL);
}

TEST_F(PyptoLaunchTest, IncompleteCoreGroupIsNeverRoundedUp)
{
    ExpectMixedLimits(8, 1);
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<1, 2>(12, stream_)), -0x200000001LL);
    ExpectQuery(ACL_RT_DEV_RES_CUBE_CORE, 1);
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<2, 3>(12, stream_)), -0x100000001LL);
}

TEST_F(PyptoLaunchTest, CubeQueryFailureTakesPriorityOverUntouchedLimit)
{
    ExpectQuery(ACL_RT_DEV_RES_CUBE_CORE, 0, 507000);
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<1, 2>(12, stream_)), -(0x300000000LL + 507000));
}

TEST_F(PyptoLaunchTest, VectorQueryFailureTakesPriorityOverUntouchedLimit)
{
    ExpectQuery(ACL_RT_DEV_RES_CUBE_CORE, 8);
    ExpectQuery(ACL_RT_DEV_RES_VECTOR_CORE, 0, 507001);
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<1, 2>(12, stream_)), -(0x400000000LL + 507001));
}

TEST_F(PyptoLaunchTest, SignedAclErrorsPreserveAll32Bits)
{
    ExpectQuery(ACL_RT_DEV_RES_CUBE_CORE, 0, static_cast<aclError>(-1));
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<1, 2>(12, stream_)), -0x3FFFFFFFFLL);
    ExpectQuery(ACL_RT_DEV_RES_VECTOR_CORE, 0, static_cast<aclError>(-1));
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<0, 1>(12, stream_)), -0x4FFFFFFFFLL);
}

TEST_F(PyptoLaunchTest, EveryCallObservesCurrentStreamAndLimitsAfterFailure)
{
    // Reuse the same template instantiation while scopes and streams change;
    // caching either a successful budget or an error would keep stale limits.
    ExpectMixedLimits(8, 6);
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<1, 2>(pypto::kAutoBlockDim, stream_)), 3);
    ExpectMixedLimits(8, 2);
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<1, 2>(pypto::kAutoBlockDim, stream_)), 1);
    uint8_t otherStreamToken = 0;
    stream_ = &otherStreamToken;
    ExpectMixedLimits(8, 1);
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<1, 2>(pypto::kAutoBlockDim, stream_)), -0x200000001LL);
    ExpectMixedLimits(8, 8);
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<1, 2>(pypto::kAutoBlockDim, stream_)), 4);
    ExpectQuery(ACL_RT_DEV_RES_CUBE_CORE, 0, 507000);
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<1, 2>(pypto::kAutoBlockDim, stream_)), -(0x300000000LL + 507000));
    // ACL also accepts the default stream; it must be forwarded without substitution.
    stream_ = nullptr;
    ExpectMixedLimits(2, 8);
    EXPECT_EQ((pypto::ResolveLaunchBlockDim<1, 2>(pypto::kAutoBlockDim, stream_)), 2);
}

} // namespace
