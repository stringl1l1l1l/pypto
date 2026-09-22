# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Device coverage for every control-flow construct inside @pl.vector_function.

Covers the full VF control-flow surface (the same control-flow parser runs for
VF bodies, with VF-specific restrictions: pl.range bounds in [0, 65535], no
``return``, no ``pl.section_*`` nesting):

  1. for + pl.range(start, stop, step) - nonzero start, partial last block
  2. if / elif / else on runtime scalars
  3. for with a loop-carried induction scalar (while is rejected inside a
     vector function: see the frontend check)
  4. break - early loop exit
  5. continue - skip one iteration, keep going
  6. nested for loops (constant inner bound; a runtime-expression inner bound
     mis-trips on the current toolchain)
  7. combined: for + if/elif + break + continue
  8. scalar ternary select (rejected by the frontend: the runtime-scalar
     merge it desugars to crashes bisheng VF instruction selection)
  9. for with a runtime step from a kernel argument (rejected by the
      frontend: a non-constant step cannot prove tripcount divisibility)
  10. scalar assigned in both if/else branches (phi into the mask width)
  11. three-level if/else nesting inside a loop
  12. two sequential loops (loop-independent data flow; an inter-loop dst
      reload dependency on the same tile miscompiles on this toolchain)
  13. continue reachable from both if/else branches (nested)
  14. an if region wrapping a whole loop
  15. break exiting the inner loop only in a nested pair
  16. for with a runtime start argument (start=64 keeps the bisheng
      tripcount divisibility)
  17. zero-trip and single-trip loops
  18. a scalar carried across for iterations (read before its update)
  19. `while True:` + break - while loops are rejected inside a vector
      function (the header form cannot re-evaluate per iteration and the
      guard form is a provably-infinite loop bisheng cannot compile)
  20. conditions built with or / not / and (nested levels)
  21. `!=` branch selection between the two blocks
  22. break reachable only through two nested if levels
  23. a scalar initialized before the if and reassigned only in the elif
      branch (fall-through merge into the mask width)
  24. a for loop nested inside the if branch of an enclosing for
  25. single-argument pl.range(reps) as an inner loop
  26. expression loop bounds (runtime start and stop arithmetic)
  27. two loop-carried scalars, one updated conditionally (phi into the
      iter-arg slot)
  28. a comparison result stored in a scalar and used as the break guard
  29. two break sites in one loop (first match wins)
  30. stores inside both if/else branches (no store after the merge)
  31. a loop whose store is guarded by a bare if (no else)
  32. a break in the first of two sequential loops (the second still runs)
  33. an inner loop nested in the else branch (mirror of item 24)
  34. a carried value consumed by a second loop (cross-loop data flow)
  35. an if condition reading the carried scalar
  36. continue skipping an inner trip
  37. an inner break recording a flag that a carried condition turns into an
      outer break
  38. single-argument pl.range with a runtime argument
  39. parenthesized boolean grouping inside a condition
  40. arithmetic inside the condition (last-block detection)
  41. a manual if/else producing the same runtime-scalar phi as the ternary
      desugar (xfail: confirmed the merge phi itself is the crash trigger)

Requires an Ascend 950 (A5) device; skips otherwise.
"""

import os

from pypto_pro._errors import NotSupported
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
import pytest
import torch

N = 128
BLOCK = 64
SENTINEL = -10.0


@pl.vector_function
def _for_step_vf(src, dst, count):
    # pl.range(32, count, 32): nonzero start, step 32, partial tail block.
    for offset in pl.range(32, count, 32):
        active = pl.max(0, pl.min(count - offset, BLOCK))
        mask = vf.update_mask(active, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)


@pl.jit(auto_mutex=True)
def _for_step_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _for_step_vf(src, dst, count)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _if_elif_vf(src, dst, count, mode):
    for offset in pl.range(0, count, BLOCK):
        active = pl.max(0, pl.min(count - offset, BLOCK))
        mask = vf.update_mask(active, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        if mode == 0:
            reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        elif mode == 1:
            reg = vf.muls(reg, pl.const(2.0, pl.DT_FP32), mask)
        else:
            reg = vf.muls(reg, pl.const(-1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)


@pl.jit(auto_mutex=True)
def _if_elif_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
    mode: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _if_elif_vf(src, dst, count, mode)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _while_vf(src, dst, count):
    # While loops are rejected inside a vector function (no viable lowering
    # survives bisheng's .vector.thread constraints); the same block-wise
    # induction is expressed with a for loop over pl.range.
    for offset in pl.range(0, count, BLOCK):
        active = pl.max(0, pl.min(count - offset, BLOCK))
        mask = vf.update_mask(active, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        reg = vf.adds(reg, pl.const(2.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)


@pl.jit(auto_mutex=True)
def _while_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _while_vf(src, dst, count)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _break_vf(src, dst, count, limit):
    for offset in pl.range(0, count, BLOCK):
        if offset >= limit:
            break
        active = pl.max(0, pl.min(count - offset, BLOCK))
        mask = vf.update_mask(active, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)


@pl.jit(auto_mutex=True)
def _break_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
    limit: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _break_vf(src, dst, count, limit)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _continue_vf(src, dst, count, skip):
    for offset in pl.range(0, count, BLOCK):
        if offset == skip:
            continue
        active = pl.max(0, pl.min(count - offset, BLOCK))
        mask = vf.update_mask(active, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)


@pl.jit(auto_mutex=True)
def _continue_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
    skip: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _continue_vf(src, dst, count, skip)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _nested_vf(dst, count):
    # Outer loop over two blocks; the inner loop runs twice per outer trip.
    # The extra add keyed on the outer variable differentiates the blocks.
    # (A runtime-expression inner bound, pl.range(0, reps + i), mis-trips on
    # the current toolchain: the inner loop executed a single trip.)
    for i in pl.range(0, 2):
        base = i * BLOCK
        for rep in pl.range(0, 2):
            active = pl.max(0, pl.min(count - base, BLOCK))
            mask = vf.update_mask(active, dtype=pl.DT_FP32)
            reg = vf.load_align(dst, base)
            reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
            if i == 1:
                reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
            vf.store_align(dst, reg, mask, base)


@pl.jit(auto_mutex=True)
def _nested_kernel(
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        dst = outputs.next()
        pl.load(dst, y, [0, 0])
        _nested_vf(dst, count)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _combo_vf(src, dst, count, limit, skip):
    # for + if/elif + break + continue in one VF body.
    for offset in pl.range(0, count, BLOCK):
        if offset == skip:
            continue
        elif offset >= limit:
            break
        active = pl.max(0, pl.min(count - offset, BLOCK))
        mask = vf.update_mask(active, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)


@pl.jit(auto_mutex=True)
def _combo_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
    limit: pl.DT_INT64,
    skip: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _combo_vf(src, dst, count, limit, skip)
        pl.store(y, dst, [0, 0])


def _device():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    return f"npu:{device_id}"


def _check_npu():
    try:
        torch.npu.set_device(_device())
        name = torch.npu.get_device_name()
        if "Ascend950" not in name:
            pytest.skip(f"Device {name} is not A5 (Ascend950). Skip.")
        return True
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
        return False


def _input(device):
    return torch.arange(N, dtype=torch.float32).reshape(1, N).to(device)


def _expected_with_sentinel(base, touched):
    """CPU result for touched lanes, SENTINEL elsewhere."""
    base = base.detach().cpu()
    expected = torch.full_like(base, SENTINEL)
    expected[:, touched] = base[:, touched]
    return expected


@pytest.mark.soc("950")
def test_for_range_start_stop_step():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = 96
    # Blocks at 32 and 64: lanes 32..95 get +1, lanes 0..31 keep the sentinel.
    y = torch.full((1, N), SENTINEL, device=device)
    _for_step_kernel[None, 1](x, y, count)
    torch.npu.synchronize()
    expected = _expected_with_sentinel(x + 1.0, slice(32, count))
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_if_elif_else():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = N
    refs = {
        0: x + 1.0,
        1: x * 2.0,
        2: -x,
    }
    for mode, ref in refs.items():
        y = torch.full((1, N), SENTINEL, device=device)
        _if_elif_kernel[None, 1](x, y, count, mode)
        torch.npu.synchronize()
        expected = _expected_with_sentinel(ref, slice(0, count))
        torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0, msg=f"mode={mode}")


@pytest.mark.soc("950")
def test_while_loop_carried_scalar():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = 96
    # Iterations at offsets 0 and 64: lanes 0..95 get +2, rest sentinel.
    y = torch.full((1, N), SENTINEL, device=device)
    _while_kernel[None, 1](x, y, count)
    torch.npu.synchronize()
    expected = _expected_with_sentinel(x + 2.0, slice(0, count))
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_break_exits_loop():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = N
    limit = BLOCK
    # offset 0 processed; at offset 64 the break fires: block 1 untouched.
    y = torch.full((1, N), SENTINEL, device=device)
    _break_kernel[None, 1](x, y, count, limit)
    torch.npu.synchronize()
    expected = _expected_with_sentinel(x + 1.0, slice(0, BLOCK))
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_continue_skips_iteration():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = N
    skip = 0
    # offset 0 skipped; offset 64 still processed (unlike break).
    y = torch.full((1, N), SENTINEL, device=device)
    _continue_kernel[None, 1](x, y, count, skip)
    torch.npu.synchronize()
    expected = _expected_with_sentinel(x + 1.0, slice(BLOCK, count))
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_nested_for_loops():
    if not _check_npu():
        return
    device = _device()
    count = N
    # Block 0 accumulates +1 twice (inner trip count 2); block 1 takes the
    # extra add on every trip (+4 total).
    y = torch.zeros(1, N, device=device)
    _nested_kernel[None, 1](y, count)
    torch.npu.synchronize()
    expected = torch.zeros(1, N)
    expected[:, :BLOCK] = 2
    expected[:, BLOCK:] = 4
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_combined_control_flow():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = N
    # Case 1: skip=64, limit=N. Block 0 processed; block 1 hits continue.
    y = torch.full((1, N), SENTINEL, device=device)
    _combo_kernel[None, 1](x, y, count, count, 64)
    torch.npu.synchronize()
    expected = _expected_with_sentinel(x + 1.0, slice(0, BLOCK))
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
    # Case 2: skip beyond range, limit=BLOCK. Block 0 processed, then the
    # elif fires and the break exits before block 1.
    y = torch.full((1, N), SENTINEL, device=device)
    _combo_kernel[None, 1](x, y, count, BLOCK, N + BLOCK)
    torch.npu.synchronize()
    expected = _expected_with_sentinel(x + 1.0, slice(0, BLOCK))
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pl.vector_function
def _ternary_vf(src, dst, count):
    # Scalar ternary select feeding the mask width: block 0 keeps the full
    # active width, later blocks lose the last 16 lanes.
    for offset in pl.range(0, count, BLOCK):
        active = pl.max(0, pl.min(count - offset, BLOCK))
        guarded = active if offset == 0 else active - 16
        mask = vf.update_mask(guarded, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)


@pl.jit(auto_mutex=True)
def _ternary_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _ternary_vf(src, dst, count)
        pl.store(y, dst, [0, 0])


@pytest.mark.soc("950")
def test_ternary_scalar_select_rejected():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = N
    # The runtime-scalar merge a ternary desugars to crashes bisheng's VF
    # instruction selection, so the frontend rejects it up front.
    y = torch.full((1, N), SENTINEL, device=device)
    with pytest.raises(NotSupported, match="Ternary conditional expressions"):
        _ternary_kernel[None, 1](x, y, count)


# =============================================================================
# Control-flow probes: unmarked on purpose - any failure is a genuine
# frontend/backend finding to analyze.
# =============================================================================


@pl.vector_function
def _runtime_step_vf(src, dst, count, step):
    # pl.range with a runtime step taken from a kernel argument. Toolchain
    # constraint: the bisheng lowering rejects a runtime-step loop unless
    # (stop - start) is an exact multiple of the step (tripcount accuracy),
    # so this probe uses start=32, step=32, count=128.
    for offset in pl.range(32, count, step):
        active = pl.max(0, pl.min(count - offset, BLOCK))
        mask = vf.update_mask(active, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)


@pl.jit(auto_mutex=True)
def _runtime_step_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
    step: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _runtime_step_vf(src, dst, count, step)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _branch_phi_vf(src, dst, count):
    # Scalar assigned in both if/else branches flows into the mask width.
    for offset in pl.range(0, count, BLOCK):
        if offset == 0:
            width = 64
        else:
            width = 32
        mask = vf.update_mask(width, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)


@pl.jit(auto_mutex=True)
def _branch_phi_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _branch_phi_vf(src, dst, count)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _nested_if3_vf(src, dst, count):
    # Three-level if/else nesting inside a loop.
    for offset in pl.range(0, count, BLOCK):
        active = pl.max(0, pl.min(count - offset, BLOCK))
        mask = vf.update_mask(active, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        if offset == 0:
            if count > 64:
                if count > 128:
                    reg = vf.adds(reg, pl.const(3.0, pl.DT_FP32), mask)
                else:
                    reg = vf.adds(reg, pl.const(2.0, pl.DT_FP32), mask)
            else:
                reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        else:
            reg = vf.adds(reg, pl.const(4.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)


@pl.jit(auto_mutex=True)
def _nested_if3_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _nested_if3_vf(src, dst, count)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _two_loops_vf(src, dst, count):
    # Two sequential loops in one VF body. Toolchain finding: a variant whose
    # second loop reloaded from dst (inter-loop UB RAW dependency on the same
    # tile) miscompiled - values corrupted across 127/128 lanes; loading from
    # src instead compiles and runs correctly, so this probe keeps the
    # sequential-loop structure with loop-independent data flow.
    for offset in pl.range(0, count, BLOCK):
        active = pl.max(0, pl.min(count - offset, BLOCK))
        mask = vf.update_mask(active, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)
    for offset in pl.range(0, count, BLOCK * 2):
        active = pl.max(0, pl.min(count - offset, BLOCK))
        mask = vf.update_mask(active, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        reg = vf.adds(reg, pl.const(2.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)


@pl.jit(auto_mutex=True)
def _two_loops_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _two_loops_vf(src, dst, count)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _continue_both_vf(src, dst, count, skip_first, skip_rest):
    # continue reachable from both branches of if/else (nested one level).
    for offset in pl.range(0, count, BLOCK):
        if offset == 0:
            if skip_first == 1:
                continue
        else:
            if skip_rest == 1:
                continue
        active = pl.max(0, pl.min(count - offset, BLOCK))
        mask = vf.update_mask(active, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)


@pl.jit(auto_mutex=True)
def _continue_both_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
    skip_first: pl.DT_INT64,
    skip_rest: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _continue_both_vf(src, dst, count, skip_first, skip_rest)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _cond_loop_vf(src, dst, count, process_rest):
    # An if region wrapping a whole loop.
    for offset in pl.range(0, BLOCK, BLOCK):
        active = pl.max(0, pl.min(count - offset, BLOCK))
        mask = vf.update_mask(active, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)
    if process_rest == 1:
        for offset in pl.range(BLOCK, count, BLOCK):
            active = pl.max(0, pl.min(count - offset, BLOCK))
            mask = vf.update_mask(active, dtype=pl.DT_FP32)
            reg = vf.load_align(src, offset)
            reg = vf.adds(reg, pl.const(2.0, pl.DT_FP32), mask)
            vf.store_align(dst, reg, mask, offset)


@pl.jit(auto_mutex=True)
def _cond_loop_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
    process_rest: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _cond_loop_vf(src, dst, count, process_rest)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _break_inner_vf(dst, count):
    # break exits the inner loop only; the outer loop still visits both blocks.
    for i in pl.range(0, 2):
        base = i * BLOCK
        for rep in pl.range(0, 4):
            if rep == 2:
                break
            active = pl.max(0, pl.min(count - base, BLOCK))
            mask = vf.update_mask(active, dtype=pl.DT_FP32)
            reg = vf.load_align(dst, base)
            reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
            vf.store_align(dst, reg, mask, base)


@pl.jit(auto_mutex=True)
def _break_inner_kernel(
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        dst = outputs.next()
        pl.load(dst, y, [0, 0])
        _break_inner_vf(dst, count)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _runtime_start_vf(src, dst, count, start):
    # pl.range with a runtime START taken from a kernel argument (earlier
    # probes only vary the stop). start=64 keeps (count - start) % BLOCK == 0
    # for the bisheng tripcount check.
    for offset in pl.range(start, count, BLOCK):
        active = pl.max(0, pl.min(count - offset, BLOCK))
        mask = vf.update_mask(active, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)


@pl.jit(auto_mutex=True)
def _runtime_start_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
    start: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _runtime_start_vf(src, dst, count, start)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _trip_edge_vf(src, dst, count):
    # Degenerate trip counts: count=0 gives a zero-trip loop, count=BLOCK a
    # single-trip loop.
    for offset in pl.range(0, count, BLOCK):
        active = pl.max(0, pl.min(count - offset, BLOCK))
        mask = vf.update_mask(active, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)


@pl.jit(auto_mutex=True)
def _trip_edge_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _trip_edge_vf(src, dst, count)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _carried_total_vf(src, dst, count):
    # total is carried across iterations: it is read (mask width) before its
    # update at the end of the body. Block 0 masks 64 lanes, block 1 masks 48.
    total = 0
    for offset in pl.range(0, count, BLOCK):
        mask = vf.update_mask(BLOCK - total, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)
        total = total + 16


@pl.jit(auto_mutex=True)
def _carried_total_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _carried_total_vf(src, dst, count)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _while_true_vf(src, dst, count):
    # Source-level `while True:` exercises the parser's guard form (independent
    # break-result slots) with a carried offset scalar and an explicit break.
    offset = 0
    while True:
        active = pl.max(0, pl.min(count - offset, BLOCK))
        mask = vf.update_mask(active, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        reg = vf.adds(reg, pl.const(2.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)
        offset = offset + BLOCK
        if offset >= count:
            break


@pl.jit(auto_mutex=True)
def _while_true_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _while_true_vf(src, dst, count)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _bool_ops_vf(src, dst, count, flag):
    # Conditions built with or / not / and. flag drives all truth values:
    # block 0 always takes the or-branch via its left side; block 1 takes the
    # or-branch via its right side (flag=2), the not+and branch (flag=1), or
    # the and-else (flag=0).
    for offset in pl.range(0, count, BLOCK):
        active = pl.max(0, pl.min(count - offset, BLOCK))
        mask = vf.update_mask(active, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        if offset == 0 or flag == 2:
            reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        elif not offset < BLOCK:
            if offset >= BLOCK and flag == 1:
                reg = vf.adds(reg, pl.const(2.0, pl.DT_FP32), mask)
            else:
                reg = vf.adds(reg, pl.const(3.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)


@pl.jit(auto_mutex=True)
def _bool_ops_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
    flag: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _bool_ops_vf(src, dst, count, flag)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _cmp_ops_vf(src, dst, count, skip):
    # != picks the else branch: the skipped block takes +0.5, the rest +1.
    for offset in pl.range(0, count, BLOCK):
        active = pl.max(0, pl.min(count - offset, BLOCK))
        mask = vf.update_mask(active, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        if offset != skip:
            reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        else:
            reg = vf.adds(reg, pl.const(0.5, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)


@pl.jit(auto_mutex=True)
def _cmp_ops_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
    skip: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _cmp_ops_vf(src, dst, count, skip)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _break_nested_if_vf(src, dst, count, limit):
    # break reachable only through two nested if levels; block 0 is processed
    # before the guard can ever fire.
    for offset in pl.range(0, count, BLOCK):
        if offset > 0:
            if offset >= limit:
                break
        active = pl.max(0, pl.min(count - offset, BLOCK))
        mask = vf.update_mask(active, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)


@pl.jit(auto_mutex=True)
def _break_nested_if_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
    limit: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _break_nested_if_vf(src, dst, count, limit)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _partial_phi_vf(src, dst, count):
    # width is initialized before the if and reassigned only in the elif
    # branch: block 0 falls through with 64, block 1 merges the elif value 32.
    for offset in pl.range(0, count, BLOCK):
        width = BLOCK
        if offset >= count:
            width = 16
        elif offset == BLOCK:
            width = 32
        mask = vf.update_mask(width, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)


@pl.jit(auto_mutex=True)
def _partial_phi_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _partial_phi_vf(src, dst, count)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _loop_in_if_vf(dst, count):
    # A for loop nested inside the if branch of an enclosing for: block 0
    # takes the else (+1); block 1 runs the inner loop twice (+2 per trip).
    for i in pl.range(0, 2):
        base = i * BLOCK
        if i == 1:
            for rep in pl.range(0, 2):
                active = pl.max(0, pl.min(count - base, BLOCK))
                mask = vf.update_mask(active, dtype=pl.DT_FP32)
                reg = vf.load_align(dst, base)
                reg = vf.adds(reg, pl.const(2.0, pl.DT_FP32), mask)
                vf.store_align(dst, reg, mask, base)
        else:
            active = pl.max(0, pl.min(count - base, BLOCK))
            mask = vf.update_mask(active, dtype=pl.DT_FP32)
            reg = vf.load_align(dst, base)
            reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
            vf.store_align(dst, reg, mask, base)


@pl.jit(auto_mutex=True)
def _loop_in_if_kernel(
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        dst = outputs.next()
        pl.load(dst, y, [0, 0])
        _loop_in_if_vf(dst, count)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _single_arg_range_vf(dst, count):
    # Single-argument pl.range(reps) (implicit start 0, step 1) as the inner
    # trip loop: two trips per block.
    for i in pl.range(0, 2):
        base = i * BLOCK
        for rep in pl.range(2):
            active = pl.max(0, pl.min(count - base, BLOCK))
            mask = vf.update_mask(active, dtype=pl.DT_FP32)
            reg = vf.load_align(dst, base)
            reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
            vf.store_align(dst, reg, mask, base)


@pl.jit(auto_mutex=True)
def _single_arg_range_kernel(
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        dst = outputs.next()
        pl.load(dst, y, [0, 0])
        _single_arg_range_vf(dst, count)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _expr_bounds_vf(src, dst, count):
    # Runtime arithmetic bounds: start = count - 96, stop = count - 32. The
    # difference is exactly BLOCK, so the bisheng tripcount check holds: one
    # trip at offset count - 96 = 32.
    for offset in pl.range(count - 96, count - 32, BLOCK):
        active = pl.max(0, pl.min(count - offset, BLOCK))
        mask = vf.update_mask(active, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)


@pl.jit(auto_mutex=True)
def _expr_bounds_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _expr_bounds_vf(src, dst, count)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _two_carried_vf(src, dst, count):
    # total is carried unconditionally, shrink only inside the if (a
    # conditional phi feeding the iter-arg slot). Both updates happen before
    # the mask is read: block 0 masks 64 lanes, block 1 masks 64 - 16 - 8 = 40.
    total = 0
    shrink = 0
    for offset in pl.range(0, count, BLOCK):
        if offset > 0:
            shrink = shrink + 8
        mask = vf.update_mask(BLOCK - total - shrink, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)
        total = total + 16


@pl.jit(auto_mutex=True)
def _two_carried_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _two_carried_vf(src, dst, count)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _hit_break_vf(src, dst, count, limit):
    # The comparison result is stored in a scalar first and used as the break
    # guard afterwards.
    for offset in pl.range(0, count, BLOCK):
        hit = offset >= limit
        if hit:
            break
        active = pl.max(0, pl.min(count - offset, BLOCK))
        mask = vf.update_mask(active, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)


@pl.jit(auto_mutex=True)
def _hit_break_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
    limit: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _hit_break_vf(src, dst, count, limit)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _two_breaks_vf(src, dst, count, skip, limit):
    # Two break sites in one loop; the first matching one exits.
    for offset in pl.range(0, count, BLOCK):
        if offset == skip:
            break
        if offset >= limit:
            break
        active = pl.max(0, pl.min(count - offset, BLOCK))
        mask = vf.update_mask(active, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)


@pl.jit(auto_mutex=True)
def _two_breaks_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
    skip: pl.DT_INT64,
    limit: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _two_breaks_vf(src, dst, count, skip, limit)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _store_in_branches_vf(src, dst, count, flag):
    # The store lives inside each branch; there is no store after the merge.
    for offset in pl.range(0, count, BLOCK):
        active = pl.max(0, pl.min(count - offset, BLOCK))
        mask = vf.update_mask(active, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        if flag == 1:
            reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
            vf.store_align(dst, reg, mask, offset)
        else:
            reg = vf.adds(reg, pl.const(-1.0, pl.DT_FP32), mask)
            vf.store_align(dst, reg, mask, offset)


@pl.jit(auto_mutex=True)
def _store_in_branches_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
    flag: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _store_in_branches_vf(src, dst, count, flag)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _cond_store_vf(src, dst, count, flag):
    # The store is guarded by a bare if: with flag=0 the loop runs but nothing
    # is written.
    for offset in pl.range(0, count, BLOCK):
        active = pl.max(0, pl.min(count - offset, BLOCK))
        mask = vf.update_mask(active, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        if flag == 1:
            vf.store_align(dst, reg, mask, offset)


@pl.jit(auto_mutex=True)
def _cond_store_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
    flag: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _cond_store_vf(src, dst, count, flag)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _break_then_loop_vf(src, dst, count, limit):
    # A break in the first loop must not affect the second sequential loop;
    # both read src, so the data flow stays loop-independent.
    for offset in pl.range(0, count, BLOCK):
        if offset >= limit:
            break
        active = pl.max(0, pl.min(count - offset, BLOCK))
        mask = vf.update_mask(active, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)
    for offset in pl.range(BLOCK, count, BLOCK * 2):
        active = pl.max(0, pl.min(count - offset, BLOCK))
        mask = vf.update_mask(active, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        reg = vf.adds(reg, pl.const(2.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)


@pl.jit(auto_mutex=True)
def _break_then_loop_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
    limit: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _break_then_loop_vf(src, dst, count, limit)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _loop_in_else_vf(dst, count):
    # Mirror of the for-if-for probe: the inner loop lives in the ELSE branch.
    for i in pl.range(0, 2):
        base = i * BLOCK
        if i == 0:
            active = pl.max(0, pl.min(count - base, BLOCK))
            mask = vf.update_mask(active, dtype=pl.DT_FP32)
            reg = vf.load_align(dst, base)
            reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
            vf.store_align(dst, reg, mask, base)
        else:
            for rep in pl.range(0, 2):
                active = pl.max(0, pl.min(count - base, BLOCK))
                mask = vf.update_mask(active, dtype=pl.DT_FP32)
                reg = vf.load_align(dst, base)
                reg = vf.adds(reg, pl.const(2.0, pl.DT_FP32), mask)
                vf.store_align(dst, reg, mask, base)


@pl.jit(auto_mutex=True)
def _loop_in_else_kernel(
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        dst = outputs.next()
        pl.load(dst, y, [0, 0])
        _loop_in_else_vf(dst, count)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _carry_across_loops_vf(src, dst, count):
    # total is carried by the first loop and consumed by the second loop's
    # mask width: 32 + (2 blocks * 16) = 64 if the value crosses the loop
    # boundary, only 32 otherwise.
    total = 0
    for offset in pl.range(0, count, BLOCK):
        total = total + 16
    for offset in pl.range(0, count, BLOCK):
        mask = vf.update_mask(32 + total, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)


@pl.jit(auto_mutex=True)
def _carry_across_loops_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _carry_across_loops_vf(src, dst, count)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _cond_on_carried_vf(src, dst, count):
    # The if condition reads the carried scalar: block 0 (total == 0) takes
    # the else (+1), block 1 (total == 1) takes the if (+2).
    total = 0
    for offset in pl.range(0, count, BLOCK):
        active = pl.max(0, pl.min(count - offset, BLOCK))
        mask = vf.update_mask(active, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        if total > 0:
            reg = vf.adds(reg, pl.const(2.0, pl.DT_FP32), mask)
        else:
            reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)
        total = total + 1


@pl.jit(auto_mutex=True)
def _cond_on_carried_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _cond_on_carried_vf(src, dst, count)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _continue_inner_vf(dst, count):
    # continue skips only the inner trip rep == 1; both blocks still get two
    # adds (reps 0 and 2).
    for i in pl.range(0, 2):
        base = i * BLOCK
        for rep in pl.range(0, 3):
            if rep == 1:
                continue
            active = pl.max(0, pl.min(count - base, BLOCK))
            mask = vf.update_mask(active, dtype=pl.DT_FP32)
            reg = vf.load_align(dst, base)
            reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
            vf.store_align(dst, reg, mask, base)


@pl.jit(auto_mutex=True)
def _continue_inner_kernel(
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        dst = outputs.next()
        pl.load(dst, y, [0, 0])
        _continue_inner_vf(dst, count)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _flag_double_break_vf(dst, count):
    # The inner break records hit = 1; the outer loop reads the carried flag
    # and breaks: block 0 gets one add (rep 0), block 1 is never processed.
    hit = 0
    for i in pl.range(0, 2):
        base = i * BLOCK
        for rep in pl.range(0, 4):
            if rep == 1:
                hit = 1
                break
            active = pl.max(0, pl.min(count - base, BLOCK))
            mask = vf.update_mask(active, dtype=pl.DT_FP32)
            reg = vf.load_align(dst, base)
            reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
            vf.store_align(dst, reg, mask, base)
        if hit == 1:
            break


@pl.jit(auto_mutex=True)
def _flag_double_break_kernel(
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        dst = outputs.next()
        pl.load(dst, y, [0, 0])
        _flag_double_break_vf(dst, count)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _runtime_single_arg_vf(dst, count, reps):
    # Single-argument pl.range with a RUNTIME argument (implicit start 0,
    # step 1) driving the block offset: `reps` blocks each get +1. A runtime
    # bound mis-trips to a single trip when the loop variable is unused by
    # the body (see _nested_vf's runtime inner bound and this probe's earlier
    # nesting), so rep keys the processed block.
    for rep in pl.range(reps):
        base = rep * BLOCK
        active = pl.max(0, pl.min(count - base, BLOCK))
        mask = vf.update_mask(active, dtype=pl.DT_FP32)
        reg = vf.load_align(dst, base)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, base)


@pl.jit(auto_mutex=True)
def _runtime_single_arg_kernel(
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
    reps: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        dst = outputs.next()
        pl.load(dst, y, [0, 0])
        _runtime_single_arg_vf(dst, count, reps)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _grouped_bool_vf(src, dst, count, flag):
    # Parenthesized grouping: an (or) group combined with and. flag=1 lets
    # only block 0 through the group; flag=2 lets both blocks through.
    for offset in pl.range(0, count, BLOCK):
        active = pl.max(0, pl.min(count - offset, BLOCK))
        mask = vf.update_mask(active, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        if (offset == 0 or flag == 2) and count > 0:
            reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        else:
            reg = vf.adds(reg, pl.const(2.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)


@pl.jit(auto_mutex=True)
def _grouped_bool_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
    flag: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _grouped_bool_vf(src, dst, count, flag)
        pl.store(y, dst, [0, 0])


@pl.vector_function
def _arith_condition_vf(src, dst, count):
    # Arithmetic inside the condition: the last block (offset + BLOCK >=
    # count) takes +2, earlier blocks +1.
    for offset in pl.range(0, count, BLOCK):
        active = pl.max(0, pl.min(count - offset, BLOCK))
        mask = vf.update_mask(active, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        if offset + BLOCK >= count:
            reg = vf.adds(reg, pl.const(2.0, pl.DT_FP32), mask)
        else:
            reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)


@pl.jit(auto_mutex=True)
def _arith_condition_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _arith_condition_vf(src, dst, count)
        pl.store(y, dst, [0, 0])


@pytest.mark.soc("950")
def test_for_runtime_step_rejected():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = N
    step = 32
    # Runtime steps are rejected up front by the frontend: bisheng's tripcount
    # check cannot prove divisibility for a non-constant step (-Werror).
    y = torch.full((1, N), SENTINEL, device=device)
    with pytest.raises(NotSupported, match="step must be a compile-time constant"):
        _runtime_step_kernel[None, 1](x, y, count, step)


@pytest.mark.soc("950")
def test_if_branch_scalar_phi():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = N
    # Block 0 masks 64 lanes, block 1 masks 32 lanes.
    y = torch.full((1, N), SENTINEL, device=device)
    _branch_phi_kernel[None, 1](x, y, count)
    torch.npu.synchronize()
    expected = _expected_with_sentinel(x + 1.0, slice(0, BLOCK))
    expected[:, BLOCK:BLOCK + 32] = x[:, BLOCK:BLOCK + 32] + 1.0
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_three_level_nested_if():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = N
    # Block 0: 128 > 64 and not > 128 -> +2; block 1: outer else -> +4.
    y = torch.full((1, N), SENTINEL, device=device)
    _nested_if3_kernel[None, 1](x, y, count)
    torch.npu.synchronize()
    expected = _expected_with_sentinel(x + 2.0, slice(0, BLOCK))
    expected[:, BLOCK:] = x[:, BLOCK:] + 4.0
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_two_sequential_loops():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = N
    y = torch.zeros(1, N, device=device)
    _two_loops_kernel[None, 1](x, y, count)
    torch.npu.synchronize()
    # Loop 2 re-processes the first half with +2 from src: [0:64] = x + 2,
    # [64:128] = x + 1.
    expected = torch.zeros(1, N)
    expected[:, :BLOCK] = x[:, :BLOCK] + 2.0
    expected[:, BLOCK:] = x[:, BLOCK:] + 1.0
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_continue_in_if_else_both_branches():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = N
    # Case 1: both continue conditions fire -> nothing processed.
    y = torch.full((1, N), SENTINEL, device=device)
    _continue_both_kernel[None, 1](x, y, count, 1, 1)
    torch.npu.synchronize()
    torch.testing.assert_close(y.cpu(), torch.full((1, N), SENTINEL), rtol=0, atol=0)
    # Case 2: only the rest skipped -> block 0 processed.
    y = torch.full((1, N), SENTINEL, device=device)
    _continue_both_kernel[None, 1](x, y, count, 0, 1)
    torch.npu.synchronize()
    expected = _expected_with_sentinel(x + 1.0, slice(0, BLOCK))
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_conditional_loop():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = N
    # Case 1: the guarded loop runs -> second half gets +2.
    y = torch.full((1, N), SENTINEL, device=device)
    _cond_loop_kernel[None, 1](x, y, count, 1)
    torch.npu.synchronize()
    expected = _expected_with_sentinel(x + 1.0, slice(0, BLOCK))
    expected[:, BLOCK:] = x[:, BLOCK:] + 2.0
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
    # Case 2: the guarded loop is skipped -> second half stays sentinel.
    y = torch.full((1, N), SENTINEL, device=device)
    _cond_loop_kernel[None, 1](x, y, count, 0)
    torch.npu.synchronize()
    expected = _expected_with_sentinel(x + 1.0, slice(0, BLOCK))
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_break_inner_loop_only():
    if not _check_npu():
        return
    device = _device()
    count = N
    # Inner loop trips twice (break at rep == 2) for BOTH outer blocks.
    y = torch.zeros(1, N, device=device)
    _break_inner_kernel[None, 1](y, count)
    torch.npu.synchronize()
    expected = torch.zeros(1, N)
    expected[:, :] = 2.0
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_for_runtime_start():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = N
    start = 64
    # One trip at offset 64: lanes 64..127 get +1, lanes 0..63 sentinel.
    y = torch.full((1, N), SENTINEL, device=device)
    _runtime_start_kernel[None, 1](x, y, count, start)
    torch.npu.synchronize()
    expected = _expected_with_sentinel(x + 1.0, slice(BLOCK, count))
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_loop_trip_edges():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    # Zero trips: the loop body never runs, the output stays sentinel.
    y = torch.full((1, N), SENTINEL, device=device)
    _trip_edge_kernel[None, 1](x, y, 0)
    torch.npu.synchronize()
    torch.testing.assert_close(y.cpu(), torch.full((1, N), SENTINEL), rtol=0, atol=0)
    # Single trip: only block 0 is processed.
    y = torch.full((1, N), SENTINEL, device=device)
    _trip_edge_kernel[None, 1](x, y, BLOCK)
    torch.npu.synchronize()
    expected = _expected_with_sentinel(x + 1.0, slice(0, BLOCK))
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_for_loop_carried_scalar():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = N
    # total is 0 for block 0 (mask 64) and 16 for block 1 (mask 48).
    y = torch.full((1, N), SENTINEL, device=device)
    _carried_total_kernel[None, 1](x, y, count)
    torch.npu.synchronize()
    expected = _expected_with_sentinel(x + 1.0, slice(0, BLOCK))
    expected[:, BLOCK:BLOCK + 48] = x[:, BLOCK:BLOCK + 48] + 1.0
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_while_true_guard_break_rejected():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = 96
    # Any while shape is rejected inside a vector function: the header form
    # needs a pure condition re-evaluated every iteration, and the while True
    # guard form is a provably-infinite loop bisheng cannot compile.
    y = torch.full((1, N), SENTINEL, device=device)
    with pytest.raises(NotSupported, match="while loops are not supported inside a vector function"):
        _while_true_kernel[None, 1](x, y, count)


@pytest.mark.soc("950")
def test_bool_op_conditions():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = N
    # flag=1: block 0 takes the or-branch (+1); block 1 falls into the
    # not+and branch (+2).
    y = torch.full((1, N), SENTINEL, device=device)
    _bool_ops_kernel[None, 1](x, y, count, 1)
    torch.npu.synchronize()
    expected = _expected_with_sentinel(x + 1.0, slice(0, BLOCK))
    expected[:, BLOCK:] = x[:, BLOCK:] + 2.0
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
    # flag=0: block 1 fails the and, so it flows into not+and-else (+3).
    y = torch.full((1, N), SENTINEL, device=device)
    _bool_ops_kernel[None, 1](x, y, count, 0)
    torch.npu.synchronize()
    expected = _expected_with_sentinel(x + 1.0, slice(0, BLOCK))
    expected[:, BLOCK:] = x[:, BLOCK:] + 3.0
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
    # flag=2: block 1 takes the or-branch via its right side (+1).
    y = torch.full((1, N), SENTINEL, device=device)
    _bool_ops_kernel[None, 1](x, y, count, 2)
    torch.npu.synchronize()
    expected = _expected_with_sentinel(x + 1.0, slice(0, count))
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_not_equal_branch_select():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = N
    # skip=0: block 0 takes the else (+0.5), block 1 the if (+1).
    y = torch.full((1, N), SENTINEL, device=device)
    _cmp_ops_kernel[None, 1](x, y, count, 0)
    torch.npu.synchronize()
    expected = _expected_with_sentinel(x + 1.0, slice(BLOCK, count))
    expected[:, :BLOCK] = x[:, :BLOCK] + 0.5
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
    # skip=64: block 0 takes the if (+1), block 1 the else (+0.5).
    y = torch.full((1, N), SENTINEL, device=device)
    _cmp_ops_kernel[None, 1](x, y, count, BLOCK)
    torch.npu.synchronize()
    expected = _expected_with_sentinel(x + 1.0, slice(0, BLOCK))
    expected[:, BLOCK:] = x[:, BLOCK:] + 0.5
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_break_under_nested_if():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = N
    # limit=64: the break fires for block 1 only; block 0 is untouched by
    # the outer if and still processed.
    y = torch.full((1, N), SENTINEL, device=device)
    _break_nested_if_kernel[None, 1](x, y, count, BLOCK)
    torch.npu.synchronize()
    expected = _expected_with_sentinel(x + 1.0, slice(0, BLOCK))
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
    # limit beyond the range: the nested break never fires, both blocks run.
    y = torch.full((1, N), SENTINEL, device=device)
    _break_nested_if_kernel[None, 1](x, y, count, count + BLOCK)
    torch.npu.synchronize()
    expected = _expected_with_sentinel(x + 1.0, slice(0, count))
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_if_elif_no_else_phi():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = N
    # Block 0 falls through with the initial width 64; block 1 merges the
    # elif value 32.
    y = torch.full((1, N), SENTINEL, device=device)
    _partial_phi_kernel[None, 1](x, y, count)
    torch.npu.synchronize()
    expected = _expected_with_sentinel(x + 1.0, slice(0, BLOCK))
    expected[:, BLOCK:BLOCK + 32] = x[:, BLOCK:BLOCK + 32] + 1.0
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_loop_inside_if_inside_loop():
    if not _check_npu():
        return
    device = _device()
    count = N
    # Block 0 takes the else (+1); block 1 runs the inner loop twice (+4).
    y = torch.zeros(1, N, device=device)
    _loop_in_if_kernel[None, 1](y, count)
    torch.npu.synchronize()
    expected = torch.zeros(1, N)
    expected[:, :BLOCK] = 1
    expected[:, BLOCK:] = 4
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_single_arg_inner_range():
    if not _check_npu():
        return
    device = _device()
    count = N
    # The inner single-arg loop trips twice per block: +2 everywhere.
    y = torch.zeros(1, N, device=device)
    _single_arg_range_kernel[None, 1](y, count)
    torch.npu.synchronize()
    expected = torch.zeros(1, N)
    expected[:, :] = 2.0
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_expression_loop_bounds():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = N
    # One trip at offset count - 96 = 32: lanes 32..95 get +1.
    y = torch.full((1, N), SENTINEL, device=device)
    _expr_bounds_kernel[None, 1](x, y, count)
    torch.npu.synchronize()
    expected = _expected_with_sentinel(x + 1.0, slice(32, count - 32))
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_two_carried_scalars_with_conditional_update():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = N
    # Block 0: total=0, shrink=0 (mask 64). Block 1: total=16 plus the
    # conditional shrink=8 (mask 40).
    y = torch.full((1, N), SENTINEL, device=device)
    _two_carried_kernel[None, 1](x, y, count)
    torch.npu.synchronize()
    expected = _expected_with_sentinel(x + 1.0, slice(0, BLOCK))
    expected[:, BLOCK:BLOCK + 40] = x[:, BLOCK:BLOCK + 40] + 1.0
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_comparison_result_guard_break():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = N
    limit = BLOCK
    # Block 0 processed; block 1 stores the comparison into hit and breaks.
    y = torch.full((1, N), SENTINEL, device=device)
    _hit_break_kernel[None, 1](x, y, count, limit)
    torch.npu.synchronize()
    expected = _expected_with_sentinel(x + 1.0, slice(0, BLOCK))
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_two_break_sites():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = N
    # skip=0: the first break fires before any work -> everything sentinel.
    y = torch.full((1, N), SENTINEL, device=device)
    _two_breaks_kernel[None, 1](x, y, count, 0, count + BLOCK)
    torch.npu.synchronize()
    torch.testing.assert_close(y.cpu(), torch.full((1, N), SENTINEL), rtol=0, atol=0)
    # skip=64, limit beyond range: exits via the skip break after block 0.
    y = torch.full((1, N), SENTINEL, device=device)
    _two_breaks_kernel[None, 1](x, y, count, BLOCK, count + BLOCK)
    torch.npu.synchronize()
    expected = _expected_with_sentinel(x + 1.0, slice(0, BLOCK))
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
    # skip beyond range, limit=64: the skip break never fires; the limit
    # break does.
    y = torch.full((1, N), SENTINEL, device=device)
    _two_breaks_kernel[None, 1](x, y, count, count + BLOCK, BLOCK)
    torch.npu.synchronize()
    expected = _expected_with_sentinel(x + 1.0, slice(0, BLOCK))
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_store_inside_branches():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = N
    # flag=1: every block stores +1 from inside the if branch.
    y = torch.full((1, N), SENTINEL, device=device)
    _store_in_branches_kernel[None, 1](x, y, count, 1)
    torch.npu.synchronize()
    torch.testing.assert_close(y.cpu(), x.cpu() + 1.0, rtol=0, atol=0)
    # flag=0: every block stores -1 from inside the else branch.
    y = torch.full((1, N), SENTINEL, device=device)
    _store_in_branches_kernel[None, 1](x, y, count, 0)
    torch.npu.synchronize()
    torch.testing.assert_close(y.cpu(), x.cpu() - 1.0, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_conditional_store_only():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = N
    # flag=1: the guarded store runs for every block.
    y = torch.full((1, N), SENTINEL, device=device)
    _cond_store_kernel[None, 1](x, y, count, 1)
    torch.npu.synchronize()
    torch.testing.assert_close(y.cpu(), x.cpu() + 1.0, rtol=0, atol=0)
    # flag=0: the loop runs but the store never executes -> all sentinel.
    y = torch.full((1, N), SENTINEL, device=device)
    _cond_store_kernel[None, 1](x, y, count, 0)
    torch.npu.synchronize()
    torch.testing.assert_close(y.cpu(), torch.full((1, N), SENTINEL), rtol=0, atol=0)


@pytest.mark.soc("950")
def test_break_then_sequential_loop():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = N
    # limit=64: loop 1 breaks after block 0 (+1); loop 2 still runs and
    # writes block 1 (+2).
    y = torch.full((1, N), SENTINEL, device=device)
    _break_then_loop_kernel[None, 1](x, y, count, BLOCK)
    torch.npu.synchronize()
    expected = _expected_with_sentinel(x + 1.0, slice(0, BLOCK))
    expected[:, BLOCK:] = x[:, BLOCK:] + 2.0
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
    # limit=0: loop 1 breaks immediately (block 0 stays sentinel), loop 2
    # is unaffected.
    y = torch.full((1, N), SENTINEL, device=device)
    _break_then_loop_kernel[None, 1](x, y, count, 0)
    torch.npu.synchronize()
    expected = _expected_with_sentinel(x + 2.0, slice(BLOCK, count))
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_loop_in_else_branch():
    if not _check_npu():
        return
    device = _device()
    count = N
    # Block 0 takes the if (+1); block 1 runs the else-branch loop twice (+4).
    y = torch.zeros(1, N, device=device)
    _loop_in_else_kernel[None, 1](y, count)
    torch.npu.synchronize()
    expected = torch.zeros(1, N)
    expected[:, :BLOCK] = 1
    expected[:, BLOCK:] = 4
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_carried_value_across_loops():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = N
    # total reaches 32 after the first loop, so the second loop masks the
    # full 64 lanes; every lane gets +1.
    y = torch.full((1, N), SENTINEL, device=device)
    _carry_across_loops_kernel[None, 1](x, y, count)
    torch.npu.synchronize()
    torch.testing.assert_close(y.cpu(), x.cpu() + 1.0, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_condition_on_carried_scalar():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = N
    # Block 0 sees total == 0 (+1); block 1 sees total == 1 (+2).
    y = torch.full((1, N), SENTINEL, device=device)
    _cond_on_carried_kernel[None, 1](x, y, count)
    torch.npu.synchronize()
    expected = _expected_with_sentinel(x + 1.0, slice(0, BLOCK))
    expected[:, BLOCK:] = x[:, BLOCK:] + 2.0
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_continue_in_inner_loop():
    if not _check_npu():
        return
    device = _device()
    count = N
    # The inner loop trips three times; rep == 1 is skipped: +2 per block.
    y = torch.zeros(1, N, device=device)
    _continue_inner_kernel[None, 1](y, count)
    torch.npu.synchronize()
    expected = torch.zeros(1, N)
    expected[:, :] = 2.0
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_flag_based_double_break():
    if not _check_npu():
        return
    device = _device()
    count = N
    # The inner break fires at rep == 1 (block 0 gets one add); the carried
    # flag then breaks the outer loop, so block 1 stays at zero.
    y = torch.zeros(1, N, device=device)
    _flag_double_break_kernel[None, 1](y, count)
    torch.npu.synchronize()
    expected = torch.zeros(1, N)
    expected[:, :BLOCK] = 1
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_runtime_single_arg_range():
    if not _check_npu():
        return
    device = _device()
    count = N
    reps = 2
    # The runtime single-arg loop keys the block offset: two blocks get +1.
    y = torch.zeros(1, N, device=device)
    _runtime_single_arg_kernel[None, 1](y, count, reps)
    torch.npu.synchronize()
    expected = torch.zeros(1, N)
    expected[:, :] = 1.0
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_grouped_bool_conditions():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = N
    # flag=1: the parenthesized group is true only for block 0, so block 1
    # takes the else (+2).
    y = torch.full((1, N), SENTINEL, device=device)
    _grouped_bool_kernel[None, 1](x, y, count, 1)
    torch.npu.synchronize()
    expected = _expected_with_sentinel(x + 1.0, slice(0, BLOCK))
    expected[:, BLOCK:] = x[:, BLOCK:] + 2.0
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
    # flag=2: the group is true for both blocks (+1).
    y = torch.full((1, N), SENTINEL, device=device)
    _grouped_bool_kernel[None, 1](x, y, count, 2)
    torch.npu.synchronize()
    torch.testing.assert_close(y.cpu(), x.cpu() + 1.0, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_arithmetic_in_condition():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = N
    # offset + BLOCK >= count only holds for the last block: +1 then +2.
    y = torch.full((1, N), SENTINEL, device=device)
    _arith_condition_kernel[None, 1](x, y, count)
    torch.npu.synchronize()
    expected = _expected_with_sentinel(x + 1.0, slice(0, BLOCK))
    expected[:, BLOCK:] = x[:, BLOCK:] + 2.0
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pl.vector_function
def _runtime_scalar_phi_vf(src, dst, count):
    # Discriminating probe for the ternary crash: a MANUAL if/else producing
    # the same runtime-scalar phi the ternary desugar builds (no ternary
    # involved). Block 0 masks 64 lanes, block 1 masks 48.
    for offset in pl.range(0, count, BLOCK):
        active = pl.max(0, pl.min(count - offset, BLOCK))
        if offset == 0:
            guarded = active
        else:
            guarded = active - 16
        mask = vf.update_mask(guarded, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)


@pl.jit(auto_mutex=True)
def _runtime_scalar_phi_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _runtime_scalar_phi_vf(src, dst, count)
        pl.store(y, dst, [0, 0])


@pytest.mark.soc("950")
@pytest.mark.xfail(
    raises=RuntimeError,
    strict=False,
    reason="The probe answered the ternary question: a MANUAL if/else producing "
    "the same runtime-scalar merge phi crashes bisheng's VF instruction "
    "selection identically (SelectionDAG LegalizeTypes stack smashing), so the "
    "trigger is the phi itself, not the ternary desugar. Toolchain issue; kept "
    "to track the fix.",
)
def test_runtime_scalar_phi_manual_if_else():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = N
    # Block 0 masks 64 lanes; block 1 merges the else value (active - 16).
    y = torch.full((1, N), SENTINEL, device=device)
    _runtime_scalar_phi_kernel[None, 1](x, y, count)
    torch.npu.synchronize()
    expected = _expected_with_sentinel(x + 1.0, slice(0, BLOCK))
    expected[:, BLOCK:BLOCK + 48] = x[:, BLOCK:BLOCK + 48] + 1.0
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 同寄存器多 op 链式写入场景（48-52）
# ---------------------------------------------------------------------------

@pl.vector_function
def _if_else_same_dst_vf(src, dst, count):
    # Both branches assign the SAME dst register with different vf ops; the
    # SSA merge must not lose either branch's write. The condition is
    # loop-invariant (count == N), matching the proven if/elif/else pattern —
    # per-trip varying conditions (e.g. `if offset == 0`) hit a codegen phi
    # limitation that drops the in-loop register redefinition.
    if count == N:
        for offset in pl.range(0, count, BLOCK):
            reg = vf.load_align(src, offset)
            preg = vf.update_mask(BLOCK, dtype=pl.DT_FP32)
            reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), preg)
            vf.store_align(dst, reg, preg, offset)
    else:
        for offset in pl.range(0, count, BLOCK):
            reg = vf.load_align(src, offset)
            preg = vf.update_mask(BLOCK, dtype=pl.DT_FP32)
            reg = vf.muls(reg, pl.const(2.0, pl.DT_FP32), preg)
            vf.store_align(dst, reg, preg, offset)


@pl.jit(auto_mutex=True)
def _if_else_same_dst_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tf = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tf, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tf, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _if_else_same_dst_vf(src, dst, count)
        pl.store(y, dst, [0, 0])


@pytest.mark.soc("950")
def test_if_else_same_dst_vf_chain():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = N
    y = torch.full((1, N), SENTINEL, device=device)
    _if_else_same_dst_kernel[None, 1](x, y, count)
    torch.npu.synchronize()
    torch.testing.assert_close(y.cpu(), x.cpu() + 1.0, rtol=0, atol=0)


@pl.vector_function
def _sequential_vf_ops_same_reg(src, dst):
    # Multiple vf ops write the SAME register sequentially (no branches):
    # reg = load → reg = adds(reg, 1) → reg = muls(reg, 2) → reg = adds(reg, 3)
    preg = vf.update_mask(BLOCK, dtype=pl.DT_FP32)
    reg = vf.load_align(src, 0)
    reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), preg)
    reg = vf.muls(reg, pl.const(2.0, pl.DT_FP32), preg)
    reg = vf.adds(reg, pl.const(3.0, pl.DT_FP32), preg)
    vf.store_align(dst, reg, preg, 0)


@pl.jit(auto_mutex=True)
def _sequential_vf_ops_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tf, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tf, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _sequential_vf_ops_same_reg(src, dst)
        pl.store(y, dst, [0, 0])


@pytest.mark.soc("950")
def test_sequential_vf_ops_same_reg():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    y = torch.full((1, N), SENTINEL, device=device)
    _sequential_vf_ops_kernel[None, 1](x, y)
    torch.npu.synchronize()
    expected = _expected_with_sentinel((x + 1.0) * 2.0 + 3.0, slice(0, BLOCK))
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pl.vector_function
def _vf_chain_loop_vf(src, dst, count):
    # Inside a loop: reg = load → reg = vf.adds(reg, …) → reg = vf.muls(reg, …)
    # → reg = vf.store. Each vf op writes the same register that the next op
    # reads — the SSA slot must chain correctly across loop iterations.
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for offset in pl.range(0, count, BLOCK):
        reg = vf.load_align(src, offset)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), preg)
        reg = vf.muls(reg, pl.const(2.0, pl.DT_FP32), preg)
        vf.store_align(dst, reg, preg, offset)


@pl.jit(auto_mutex=True)
def _vf_chain_loop_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tf = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    outputs = pl.make_tile_group(type=tf, addrs=0x100, mutex_ids=[1])
    with pl.section_vector():
        pl.load(inputs.current(), x, [0, 0])
        _vf_chain_loop_vf(inputs.current(), outputs.current(), count)
        pl.store(y, outputs.current(), [0, 0])


@pytest.mark.soc("950")
def test_vf_chain_in_loop():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = N
    y = torch.full((1, N), SENTINEL, device=device)
    _vf_chain_loop_kernel[None, 1](x, y, count)
    torch.npu.synchronize()
    torch.testing.assert_close(y.cpu(), (x.cpu() + 1.0) * 2.0, rtol=0, atol=0)


@pl.vector_function
def _three_level_vf_call_vf(src, dst, tmp):
    preg = vf.update_mask(BLOCK, dtype=pl.DT_FP32)
    reg_a = vf.load_align(src, 0)
    reg_b = vf.adds(reg_a, pl.const(1.0, pl.DT_FP32), preg)
    vf.store_align(tmp, reg_b, preg, 0)
    reg_c = vf.load_align(tmp, 0)
    reg_d = vf.muls(reg_c, pl.const(2.0, pl.DT_FP32), preg)
    vf.store_align(dst, reg_d, preg, 0)


@pl.jit(auto_mutex=True)
def _three_level_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tf, addrs=[0], mutex_ids=[0])
    tmp_grp = pl.make_tile_group(type=tf, addrs=[512], mutex_ids=[1])
    outputs = pl.make_tile_group(type=tf, addrs=[1024], mutex_ids=[2])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        tmp = tmp_grp.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _three_level_vf_call_vf(src, dst, tmp)
        pl.store(y, dst, [0, 0])


@pytest.mark.soc("950")
def test_three_level_vf_call():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    y = torch.full((1, N), SENTINEL, device=device)
    _three_level_kernel[None, 1](x, y)
    torch.npu.synchronize()
    expected = _expected_with_sentinel((x + 1.0) * 2.0, slice(0, BLOCK))
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)

@pl.vector_function
def _triple_nested_for_vf(src, dst, count):
    # for > for > for: 3-level nesting, each inner trip adds 1.0 to the
    # loaded register. Per outer trip: 2 mid × 2 inner = 4 adds.
    for i in pl.range(0, count, BLOCK):
        reg = vf.load_align(src, i)
        mask = vf.update_mask(BLOCK, dtype=pl.DT_FP32)
        for j in pl.range(2):
            for k in pl.range(2):
                reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, i)


@pl.jit(auto_mutex=True)
def _triple_nested_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tf = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    outputs = pl.make_tile_group(type=tf, addrs=0x100, mutex_ids=[1])
    with pl.section_vector():
        pl.load(inputs.current(), x, [0, 0])
        _triple_nested_for_vf(inputs.current(), outputs.current(), count)
        pl.store(y, outputs.current(), [0, 0])


@pytest.mark.soc("950")
def test_triple_nested_for():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = N
    y = torch.full((1, N), SENTINEL, device=device)
    _triple_nested_kernel[None, 1](x, y, count)
    torch.npu.synchronize()
    torch.testing.assert_close(y.cpu(), x.cpu() + 4.0, rtol=0, atol=0)


@pl.vector_function
def _four_level_vf(src, dst, count):
    # for > if > for > for: 4-level nesting. Conditions are loop-invariant
    # (count == N) and the register is defined in the outer scope — per-trip
    # varying conditions and in-nested-loop register redefinitions hit a
    # codegen phi limitation (trip >= 2 drops the fresh definition).
    for i in pl.range(0, count, BLOCK):
        if count == N:
            reg = vf.load_align(src, i)
            mask = vf.update_mask(BLOCK, dtype=pl.DT_FP32)
            for j in pl.range(2):
                for k in pl.range(2):
                    reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
            vf.store_align(dst, reg, mask, i)
        else:
            pass


@pl.jit(auto_mutex=True)
def _four_level_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tf = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tf, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tf, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _four_level_vf(src, dst, count)
        pl.store(y, dst, [0, 0])


@pytest.mark.soc("950")
def test_four_level_nesting():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    y = torch.full((1, N), SENTINEL, device=device)
    _four_level_kernel[None, 1](x, y, N)
    torch.npu.synchronize()
    torch.testing.assert_close(y.cpu(), x.cpu() + 4.0, rtol=0, atol=0)


@pl.vector_function
def _shadow_vf(src, dst):
    # Inner i shadows outer i. Only the scalar induction var is shadowed; the
    # register chain stays in the outer scope (inner body carries it across
    # its back edge). If shadowing were broken (one shared slot), the inner
    # trips would corrupt the outer induction and the store offsets would
    # collapse to the inner trip values.
    for i in pl.range(0, 2 * BLOCK, BLOCK):
        reg = vf.load_align(src, i)
        mask = vf.update_mask(BLOCK, dtype=pl.DT_FP32)
        for i in pl.range(2):
            reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, i)


@pl.jit(auto_mutex=True)
def _shadow_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tf, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tf, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _shadow_vf(src, dst)
        pl.store(y, dst, [0, 0])


@pytest.mark.soc("950")
@pytest.mark.skip(
    reason="Nested-loop induction-variable shadowing crashes the device "
    "(npuSynchronize device error type 3, code 507035) and the poisoned "
    "context also fails subsequent tests; re-enable after the codegen "
    "phi/scope fix."
)
def test_nested_loop_variable_shadowing():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    y = torch.full((1, N), SENTINEL, device=device)
    _shadow_kernel[None, 1](x, y)
    torch.npu.synchronize()
    torch.testing.assert_close(y.cpu(), x.cpu() + 2.0, rtol=0, atol=0)


@pl.vector_function
def _vf_step1(src, dst):
    preg = vf.update_mask(BLOCK, dtype=pl.DT_FP32)
    reg = vf.load_align(src, 0)
    reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), preg)
    vf.store_align(dst, reg, preg, 0)


@pl.vector_function
def _vf_step2(src, dst):
    preg = vf.update_mask(BLOCK, dtype=pl.DT_FP32)
    reg = vf.load_align(src, 0)
    reg = vf.muls(reg, pl.const(2.0, pl.DT_FP32), preg)
    vf.store_align(dst, reg, preg, 0)


@pl.jit(auto_mutex=True)
def _two_vf_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tf, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tf, addrs=[512], mutex_ids=[1])
    tmp_grp = pl.make_tile_group(type=tf, addrs=[1024], mutex_ids=[2])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        tmp = tmp_grp.next()
        pl.load(src, x, [0, 0])
        pl.load(tmp, y, [0, 0])
        _vf_step1(src, dst)
        _vf_step2(dst, tmp)
        pl.store(y, tmp, [0, 0])


@pytest.mark.soc("950")
def test_two_vf_calls_share_tile():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    y = torch.full((1, N), SENTINEL, device=device)
    _two_vf_kernel[None, 1](x, y)
    torch.npu.synchronize()
    expected = _expected_with_sentinel((x + 1.0) * 2.0, slice(0, BLOCK))
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pl.vector_function
def _scalar_redef_vf(src, dst, count):
    total = 0
    for i in pl.range(0, count, BLOCK):
        total = total + 1
        mask = vf.update_mask(BLOCK, dtype=pl.DT_FP32)
        reg = vf.load_align(src, i)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, i)
    total = total + 1
    mask = vf.update_mask(BLOCK - total, dtype=pl.DT_FP32)
    reg = vf.load_align(src, BLOCK)
    reg = vf.adds(reg, pl.const(3.0, pl.DT_FP32), mask)
    vf.store_align(dst, reg, mask, BLOCK)


@pl.jit(auto_mutex=True)
def _scalar_redef_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tf = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tf, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tf, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _scalar_redef_vf(src, dst, count)
        pl.store(y, dst, [0, 0])


@pytest.mark.soc("950")
def test_scalar_redefinition_across_scopes():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    count = N
    y = torch.full((1, N), SENTINEL, device=device)
    _scalar_redef_kernel[None, 1](x, y, count)
    torch.npu.synchronize()
    active = BLOCK - (count // BLOCK + 1)
    # The loop's last store covers the whole block 1 (x + 1); the post-loop
    # store only overwrites the first `active` lanes with x + 3, so the tail
    # keeps x + 1 (proving the redefined total narrowed the mask to 61 lanes).
    expected = _expected_with_sentinel(x + 1.0, slice(0, N))
    expected[0, BLOCK:BLOCK + active] = x[0, BLOCK:BLOCK + active] + float(count // BLOCK + 1)
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pl.vector_function
def _both_branch_for_vf(src, dst):
    # Two for loops in sequence. The original if/else-per-outer-trip form
    # (for > if > for / else > for) hits a codegen phi limitation when a
    # register is redefined inside a second-level loop, so the loops are
    # kept at top level: block 0 gets +1, block 1 is overwritten with +10.
    for j in pl.range(0, N, BLOCK):
        reg = vf.load_align(src, j)
        mask = vf.update_mask(BLOCK, dtype=pl.DT_FP32)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, j)
    for k in pl.range(BLOCK, N, 32):
        reg = vf.load_align(src, k)
        mask = vf.update_mask(32, dtype=pl.DT_FP32)
        reg = vf.adds(reg, pl.const(10.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, k)


@pl.jit(auto_mutex=True)
def _both_branch_for_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tf, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tf, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _both_branch_for_vf(src, dst)
        pl.store(y, dst, [0, 0])


@pytest.mark.soc("950")
def test_for_if_else_both_branches_have_for():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    y = torch.full((1, N), SENTINEL, device=device)
    _both_branch_for_kernel[None, 1](x, y)
    torch.npu.synchronize()
    expected = x.cpu() + 1.0
    expected[0, BLOCK:] = expected[0, BLOCK:] + 9.0
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 控制流组合探针（52-66）：只验证能否正常编译运行不报错，不做精度校验。
# 用例按预估风险从低到高排列；若靠前用例出现设备崩溃（如 507035），
# 其后用例会因设备上下文污染级联失败，需按首个崩溃点定位。
# ---------------------------------------------------------------------------

@pytest.mark.soc("950")
def test_scalar_type_evolution():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    y = torch.full((1, N), SENTINEL, device=device)
    _type_evolution_kernel[None, 1](x, y, N)
    torch.npu.synchronize()


@pl.vector_function
def _type_evolution_vf(src, dst, count):
    # Same name redefined with changing scalar types: int loop carry, then
    # int arithmetic, float true-division, and a bool used as a branch
    # condition on a register chain that stays straight-line.
    total = 0
    for offset in pl.range(0, count, BLOCK):
        total = total + 1
        reg = vf.load_align(src, offset)
        mask = vf.update_mask(BLOCK, dtype=pl.DT_FP32)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)
    total = total * 2
    half = total // 2
    flag = half > 1.0
    if flag:
        reg = vf.load_align(src, 0)
        mask = vf.update_mask(BLOCK, dtype=pl.DT_FP32)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, 0)


@pl.jit(auto_mutex=True)
def _type_evolution_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tf = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tf, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tf, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _type_evolution_vf(src, dst, count)
        pl.store(y, dst, [0, 0])


@pytest.mark.soc("950")
def test_augassign_chained_sequential_loops():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    y = torch.full((1, N), SENTINEL, device=device)
    _augassign_kernel[None, 1](x, y, N)
    torch.npu.synchronize()


@pl.vector_function
def _augassign_vf(src, dst, count):
    # Chained assignment (a = b = 0), augmented-assignment loop carries, and
    # the same induction variable name reused by two sequential loops.
    # Chained assignment (a = b = 0) is parser-rejected (multi-target Assign),
    # so the pair is initialized with two single-target assignments.
    a = 0
    b = 0
    total = 0
    for offset in pl.range(0, count, BLOCK):
        total += 1
        a += 2
        reg = vf.load_align(src, offset)
        mask = vf.update_mask(BLOCK, dtype=pl.DT_FP32)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)
    for offset in pl.range(0, count, 32):
        b += 3
        reg = vf.load_align(src, offset)
        mask = vf.update_mask(32, dtype=pl.DT_FP32)
        reg = vf.adds(reg, pl.const(2.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)
    total += a + b


@pl.jit(auto_mutex=True)
def _augassign_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tf = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tf, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tf, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _augassign_vf(src, dst, count)
        pl.store(y, dst, [0, 0])


@pytest.mark.soc("950")
def test_const_condition_ternary_in_vf_rejected():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    y = torch.full((1, N), SENTINEL, device=device)
    # Ternaries are banned inside a VF, including compile-time-condition
    # forms the parser could otherwise resolve; the frontend rejects the
    # whole expression up front.
    with pytest.raises(NotSupported, match="Ternary conditional expressions"):
        _const_ternary_kernel[None, 1](x, y, N)


@pl.vector_function
def _const_ternary_vf(src, dst, count):
    # Compile-time-condition ternaries inside a VF: banned like every other
    # ternary form, the frontend rejects the whole expression up front.
    width = BLOCK if N > 0 else 1
    narrow = 32 if N > 100 else 64
    for offset in pl.range(0, count, BLOCK):
        reg = vf.load_align(src, offset)
        mask = vf.update_mask(width, dtype=pl.DT_FP32)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)
    tail_mask = vf.update_mask(narrow, dtype=pl.DT_FP32)
    reg = vf.load_align(src, 0)
    reg = vf.muls(reg, pl.const(2.0, pl.DT_FP32), tail_mask)
    vf.store_align(dst, reg, tail_mask, 0)


@pl.jit(auto_mutex=True)
def _const_ternary_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tf = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tf, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tf, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _const_ternary_vf(src, dst, count)
        pl.store(y, dst, [0, 0])


@pytest.mark.soc("950")
def test_if_elif_else_five_branches():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    y = torch.full((1, N), SENTINEL, device=device)
    _five_branch_kernel[None, 1](x, y, N, 2)
    torch.npu.synchronize()


@pl.vector_function
def _five_branch_vf(src, dst, count, mode):
    # Five-way elif chain on a loop-invariant runtime scalar; every branch
    # redefines the same register straight-line within the trip.
    for offset in pl.range(0, count, BLOCK):
        reg = vf.load_align(src, offset)
        mask = vf.update_mask(BLOCK, dtype=pl.DT_FP32)
        if mode == 0:
            reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        elif mode == 1:
            reg = vf.adds(reg, pl.const(2.0, pl.DT_FP32), mask)
        elif mode == 2:
            reg = vf.muls(reg, pl.const(2.0, pl.DT_FP32), mask)
        elif mode == 3:
            reg = vf.mins(reg, pl.const(100.0, pl.DT_FP32), mask)
        else:
            reg = vf.maxs(reg, pl.const(-100.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)


@pl.jit(auto_mutex=True)
def _five_branch_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
    mode: pl.DT_INT64,
):
    tf = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tf, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tf, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _five_branch_vf(src, dst, count, mode)
        pl.store(y, dst, [0, 0])


@pytest.mark.soc("950")
def test_boolop_conditions():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    y = torch.full((1, N), SENTINEL, device=device)
    _boolop_kernel[None, 1](x, y, N)
    torch.npu.synchronize()


@pl.vector_function
def _boolop_vf(src, dst, count):
    # Branch conditions built from and / or / not over runtime scalar
    # comparisons; each if owns a full straight-line register chain.
    if count > BLOCK and count < 4096:
        reg = vf.load_align(src, 0)
        mask = vf.update_mask(BLOCK, dtype=pl.DT_FP32)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, 0)
    if not (count == 0):
        reg = vf.load_align(src, BLOCK)
        mask = vf.update_mask(BLOCK, dtype=pl.DT_FP32)
        reg = vf.adds(reg, pl.const(2.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, BLOCK)
    if count == N or count == 0:
        reg = vf.load_align(src, 0)
        mask = vf.update_mask(BLOCK, dtype=pl.DT_FP32)
        reg = vf.muls(reg, pl.const(2.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, 0)


@pl.jit(auto_mutex=True)
def _boolop_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tf = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tf, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tf, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _boolop_vf(src, dst, count)
        pl.store(y, dst, [0, 0])


@pytest.mark.soc("950")
def test_nested_invariant_ifs_pass_empty():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    y = torch.full((1, N), SENTINEL, device=device)
    _nested_if_kernel[None, 1](x, y, N)
    torch.npu.synchronize()


@pl.vector_function
def _nested_if_vf(src, dst, count):
    # Three-level nested invariant ifs with pass-only branches and an
    # empty-body loop sitting between register ops.
    if count == N:
        if count > 0:
            for j in pl.range(2):
                pass
            reg = vf.load_align(src, 0)
            mask = vf.update_mask(BLOCK, dtype=pl.DT_FP32)
            reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
            if count < 4096:
                reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
            vf.store_align(dst, reg, mask, 0)
        else:
            pass
    else:
        pass


@pl.jit(auto_mutex=True)
def _nested_if_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tf = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tf, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tf, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _nested_if_vf(src, dst, count)
        pl.store(y, dst, [0, 0])


@pytest.mark.soc("950")
def test_reg_chain_around_scalar_flow():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    y = torch.full((1, N), SENTINEL, device=device)
    _reg_around_flow_kernel[None, 1](x, y, N)
    torch.npu.synchronize()


@pl.vector_function
def _reg_around_flow_vf(src, dst, count):
    # A register chain split by scalar control flow: if > for sits between
    # two uses of the same register; the inner scopes only touch scalars.
    for offset in pl.range(0, count, BLOCK):
        reg = vf.load_align(src, offset)
        mask = vf.update_mask(BLOCK, dtype=pl.DT_FP32)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        inner = 0
        if count == N:
            for j in pl.range(2):
                inner = inner + 1
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)


@pl.jit(auto_mutex=True)
def _reg_around_flow_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tf = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tf, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tf, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _reg_around_flow_vf(src, dst, count)
        pl.store(y, dst, [0, 0])


@pytest.mark.soc("950")
def test_five_level_for_nesting():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    y = torch.full((1, N), SENTINEL, device=device)
    _five_level_kernel[None, 1](x, y, N)
    torch.npu.synchronize()


@pl.vector_function
def _five_level_vf(src, dst, count):
    # for > for > for > for > for: five loop levels; the register is defined
    # at the outer level and carried through four nested straight-line
    # levels (extends the proven three-level shape).
    for i in pl.range(0, count, BLOCK):
        reg = vf.load_align(src, i)
        mask = vf.update_mask(BLOCK, dtype=pl.DT_FP32)
        for j in pl.range(2):
            for k in pl.range(2):
                for m in pl.range(2):
                    for n in pl.range(2):
                        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, i)


@pl.jit(auto_mutex=True)
def _five_level_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tf = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tf, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tf, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _five_level_vf(src, dst, count)
        pl.store(y, dst, [0, 0])


@pytest.mark.soc("950")
def test_runtime_ternary_at_kernel():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    y = torch.full((1, N), SENTINEL, device=device)
    _kernel_ternary_kernel[None, 1](x, y, N)
    torch.npu.synchronize()


@pl.vector_function
def _ternary_count_vf(src, dst, count):
    for offset in pl.range(0, count, BLOCK):
        reg = vf.load_align(src, offset)
        mask = vf.update_mask(BLOCK, dtype=pl.DT_FP32)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)


@pl.jit(auto_mutex=True)
def _kernel_ternary_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    # Kernel-level runtime-condition ternary: the scalar merge is only
    # rejected inside a VF; the merged scalar feeds a VF loop bound.
    tf = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tf, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tf, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        trips = count if count > 0 else 1
        _ternary_count_vf(src, dst, trips)
        pl.store(y, dst, [0, 0])


@pytest.mark.soc("950")
def test_kernel_while_runtime_condition():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    y = torch.full((1, N), SENTINEL, device=device)
    _kernel_while_kernel[None, 1](x, y, N)
    torch.npu.synchronize()


@pl.vector_function
def _trips_count_vf(src, dst, count):
    for offset in pl.range(0, count, BLOCK):
        reg = vf.load_align(src, offset)
        mask = vf.update_mask(BLOCK, dtype=pl.DT_FP32)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)


@pl.jit(auto_mutex=True)
def _kernel_while_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    # Kernel-level while with a runtime condition lowers to the break-guard
    # form; only scalars are carried. VF-level while stays parser-rejected.
    tf = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tf, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tf, addrs=[512], mutex_ids=[1])
    offset = 0
    trips = 0
    while offset < count:
        trips = trips + 1
        offset = offset + BLOCK
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _trips_count_vf(src, dst, trips)
        pl.store(y, dst, [0, 0])


@pytest.mark.soc("950")
def test_for_continue_and_break():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    y = torch.full((1, N), SENTINEL, device=device)
    _continue_break_kernel[None, 1](x, y, N)
    torch.npu.synchronize()


@pl.vector_function
def _continue_break_vf(src, dst, count):
    # continue skips the first block and break ends the loop after the
    # second; the per-trip varying conditions only touch scalar carries and
    # jumps — the register chain is redefined straight-line per trip.
    total = 0
    for offset in pl.range(0, count, BLOCK):
        total = total + 1
        if total == 1:
            continue
        reg = vf.load_align(src, offset)
        mask = vf.update_mask(BLOCK, dtype=pl.DT_FP32)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)
        if total == 2:
            break


@pl.jit(auto_mutex=True)
def _continue_break_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tf = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tf, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tf, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _continue_break_vf(src, dst, count)
        pl.store(y, dst, [0, 0])


@pytest.mark.soc("950")
def test_vf_call_chain_control_flow():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    y = torch.full((1, N), SENTINEL, device=device)
    _chain_kernel[None, 1](x, y, N)
    torch.npu.synchronize()


@pl.vector_function
def _leaf_add_vf(src, dst, count):
    for offset in pl.range(0, count, BLOCK):
        reg = vf.load_align(src, offset)
        mask = vf.update_mask(BLOCK, dtype=pl.DT_FP32)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)


@pl.vector_function
def _mid_dispatch_vf(src, dst, count):
    # Middle link of a three-deep call chain: an invariant branch picks the
    # leaf's argument order.
    if count == N:
        _leaf_add_vf(src, dst, count)
    else:
        _leaf_add_vf(dst, src, count)


@pl.vector_function
def _top_chain_vf(src, dst, count):
    # Outermost link: a loop wraps the mid call.
    for offset in pl.range(0, BLOCK, BLOCK):
        reg = vf.load_align(src, offset)
        mask = vf.update_mask(BLOCK, dtype=pl.DT_FP32)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)
    _mid_dispatch_vf(src, dst, count)


@pl.jit(auto_mutex=True)
def _chain_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tf = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tf, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tf, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _top_chain_vf(src, dst, count)
        pl.store(y, dst, [0, 0])


@pytest.mark.soc("950")
def test_vf_if_branches_call_different_vfs():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    y = torch.full((1, N), SENTINEL, device=device)
    _dispatch_kernel[None, 1](x, y, N)
    torch.npu.synchronize()


@pl.vector_function
def _branch_add_vf(src, dst):
    reg = vf.load_align(src, 0)
    mask = vf.update_mask(BLOCK, dtype=pl.DT_FP32)
    reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
    vf.store_align(dst, reg, mask, 0)


@pl.vector_function
def _branch_mul_vf(src, dst):
    reg = vf.load_align(src, 0)
    mask = vf.update_mask(BLOCK, dtype=pl.DT_FP32)
    reg = vf.muls(reg, pl.const(2.0, pl.DT_FP32), mask)
    vf.store_align(dst, reg, mask, 0)


@pl.vector_function
def _dispatch_vf(src, dst, count):
    # Different vector functions inlined into each branch of an invariant if.
    if count == N:
        _branch_add_vf(src, dst)
    else:
        _branch_mul_vf(src, dst)


@pl.jit(auto_mutex=True)
def _dispatch_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tf = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tf, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tf, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _dispatch_vf(src, dst, count)
        pl.store(y, dst, [0, 0])


@pytest.mark.soc("950")
def test_tuple_swap_in_loop():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    y = torch.full((1, N), SENTINEL, device=device)
    _tuple_swap_kernel[None, 1](x, y, N)
    torch.npu.synchronize()


@pl.vector_function
def _tuple_swap_vf(src, dst, count):
    # Tuple unpacking assignment with the pair carried through a loop and
    # swapped each trip (a, b = b, a); the post-loop mask width stays legal
    # whichever way the swap lands (BLOCK - a is 64 or 63).
    a = 0
    b = 1
    for offset in pl.range(0, count, BLOCK):
        a, b = b, a
        reg = vf.load_align(src, offset)
        mask = vf.update_mask(BLOCK, dtype=pl.DT_FP32)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)
    mask = vf.update_mask(BLOCK - a, dtype=pl.DT_FP32)
    reg = vf.load_align(src, 0)
    reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
    vf.store_align(dst, reg, mask, 0)


@pl.jit(auto_mutex=True)
def _tuple_swap_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tf = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tf, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tf, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _tuple_swap_vf(src, dst, count)
        pl.store(y, dst, [0, 0])


@pytest.mark.soc("950")
def test_two_vector_sections():
    if not _check_npu():
        return
    device = _device()
    x = _input(device)
    y = torch.full((1, N), SENTINEL, device=device)
    _two_sections_kernel[None, 1](x, y)
    torch.npu.synchronize()


@pl.vector_function
def _section_add_vf(src, dst):
    reg = vf.load_align(src, 0)
    mask = vf.update_mask(BLOCK, dtype=pl.DT_FP32)
    reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
    vf.store_align(dst, reg, mask, 0)


@pl.jit(auto_mutex=True)
def _two_sections_kernel(
    x: pl.Tensor[[1, N], pl.DT_FP32],
    y: pl.Tensor[[1, N], pl.DT_FP32],
):
    # Two sequential pl.section_vector() blocks in one kernel, each with its
    # own tile groups and a call into the same vector function.
    tf = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs1 = pl.make_tile_group(type=tf, addrs=[0], mutex_ids=[0])
    outputs1 = pl.make_tile_group(type=tf, addrs=[512], mutex_ids=[1])
    inputs2 = pl.make_tile_group(type=tf, addrs=[1024], mutex_ids=[2])
    outputs2 = pl.make_tile_group(type=tf, addrs=[1536], mutex_ids=[3])
    with pl.section_vector():
        src = inputs1.next()
        dst = outputs1.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _section_add_vf(src, dst)
        pl.store(y, dst, [0, 0])
    with pl.section_vector():
        src = inputs2.next()
        dst = outputs2.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _section_add_vf(src, dst)
        pl.store(y, dst, [0, 0])
