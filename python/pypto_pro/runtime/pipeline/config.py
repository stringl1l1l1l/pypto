#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Pipeline configuration."""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for preload pipeline transformation.

    Args:
        preload: How many iterations ahead an upstream stage runs — one value, or one per
                 pipeline loop when the kernel holds several (in source order); a single value
                 applies to all of them. A larger value hides more of the transfer/compute
                 latency, at the cost of a longer fill and drain — tune it per kernel.
                 Concretely it is the delay step between two consecutive stages on the SAME
                 core (see _compute_delays); the ctx ring-buffer depth follows from the
                 resulting delays (max_delay + 1), not from preload directly.

                 Zero pulls no stage ahead of any other, leaving the serial loop with
                 cross-core sync inserted around each stage. Start there to confirm the serial
                 kernel is correct, then raise it.

                 Must be an int for every value (floats and bools are refused) and must
                 not be negative.
    """

    preload: int | tuple[int, ...] = 2

    def __post_init__(self) -> None:
        if isinstance(self.preload, Sequence):
            # Frozen dataclasses derive __hash__ from their fields, and a list is not
            # hashable, so a sequence is stored as a tuple whatever the user wrote.
            object.__setattr__(self, "preload", tuple(self.preload))
            if not self.preload:
                raise ValueError(
                    "pipeline: preload is an empty sequence. Give one value per pipeline "
                    "loop, or a single value to use for all of them."
                )
        for value in self.preload if isinstance(self.preload, tuple) else (self.preload,):
            # bool is a subclass of int, so the isinstance check alone would let it through.
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    f"pipeline: preload must be an int, got {type(value).__name__}: {value!r}. "
                    f"It is how many iterations ahead a stage runs, so a float or a bool "
                    f"has no meaning here; write it as a plain int."
                )
            if value < 0:
                raise ValueError(
                    f"pipeline: preload must be >= 0, got {value}. It is how many "
                    f"iterations ahead a stage runs, so a negative value has no meaning; use "
                    f"preload=0 to keep the serial loop and only insert cross-core sync."
                )

    def preload_of(self, loop_index: int) -> int:
        """This pipeline loop's preload.

        A single value is shared by every loop rather than meaning "the first one only": one
        number reads as a property of the kernel.
        """
        return self.preload[loop_index] if isinstance(self.preload, tuple) else self.preload
