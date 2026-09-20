# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Python API declarations for the PyPTO VF namespace (``pl.vf.xxx``).

These declarations exist so that:
- IDE "Go to Definition" works for every ``pl.vf.xxx`` call
- Python catches typos at import time
- Type checkers can validate argument types
- Docstrings document the user-facing calling convention

None of these functions are meant to be called at runtime.  Inside a PyPTO
kernel the AST parser intercepts every ``pl.vf.xxx`` call before Python executes
it.  Outside a kernel, calling a declaration raises ``RuntimeError``.
"""

from __future__ import annotations

from typing import Optional

from pypto.ir import (
    BinType,
    CastLayout,
    DataCopyMode,
    DuplicatePos,
    HistType,
    IndexOrder,
    LoadDist,
    MaskPattern,
    MaskWidth,
    MemBarMode,
    MergeMode,
    PackPart,
    SaturateMode,
    SqueezeMode,
    StoreDist,
    VFRoundMode,
)

from ._api import DType, _api_decl

# ===================================================================
# VF namespace (``pl.vf.*``)
# ===================================================================


class Vf:
    """Vector Function unit operations (A5 architecture).

    Used inside ``@pl.vector_function`` decorated functions.  Compute ops
    produce a result and must use the assignment form ``dst = vf.xxx(...)``;
    the destination register is declared implicitly.  Only store/side-effect
    ops are called as bare statements.
    """

    @staticmethod
    @_api_decl
    def create_mask(pattern: Optional[MaskPattern] = None,
                    dtype: Optional[DType] = None):
        """Create and initialize a VF predicate mask register.

        Both kwargs are optional and may be given independently:
            preg = vf.create_mask(dtype=pl.DT_FP16)          # pattern defaults to ALL
            preg = vf.create_mask(pattern=pl.MaskPattern.VL8) # dtype defaults to FP32
            preg = vf.create_mask()                           # both defaults

        Args:
            pattern: Mask pattern, ``pl.MaskPattern.ALL`` (default) or other
                ``MaskPattern`` enum value (VL1..VL128, M3, M4, H, Q, ALLF)
            dtype: Data type that determines mask granularity (default FP32;
                FP16/BF16/UINT8/INT8/FP8E4M3FN/FP8E5M2/FP8E8M0/HF8/FP4E2M1/FP4E1M2 etc.
                — all b8/b4 types are treated as b8 mask width;
                INT64/UINT64 are treated as b64 mask width, internally using
                pset_b32 + punpack to match 2-bit-per-element granularity)

        Returns:
            mask_reg: Initialized mask register
        """

    @staticmethod
    @_api_decl
    def update_mask(scalar, dtype: Optional[DType] = None):
        """Update a VF mask register from a scalar value.

        Sets the mask register bits according to the scalar value.  The scalar
        value's bits define the new mask pattern.

        Args:
            scalar: Scalar value whose bits define the new mask pattern

        Kwargs:
            dtype: Data type for mask width selection (default FP32 -> b32)

        Returns:
            Updated mask register (``MaskReg``).
        """

    @staticmethod
    @_api_decl
    def full(src, preg, dtype: Optional[DType] = None,
             mode: Optional[MergeMode] = None,
             pos: Optional[DuplicatePos] = None):
        """Broadcast a scalar or vector-source element into all lanes of a VF register.

        Two modes:

        - **Scalar mode**: ``vf.full(2.5, preg, dtype=pl.DT_FP32)`` --- broadcasts a scalar
          value to all register lanes (vbr/vdup instruction).
        - **Tensor mode**: ``vf.full(src_reg, preg)`` --- broadcasts the lowest or highest
          element of a source register to all lanes of the destination register
          (vdup instruction). Mask is required in Tensor mode.

        Args:
            scalar_or_src: Scalar value (Scalar mode) or source register (Tensor mode)
            preg: Predicate mask register. Required for Tensor mode; optional for Scalar mode

        Kwargs:
            dtype: Required for Scalar mode (cannot infer from scalar). Auto-inferred
                for Tensor mode from the source register.
            mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.
            pos: ``pl.DuplicatePos.LOWEST`` (default) or ``pl.DuplicatePos.HIGHEST``
                selects which element to broadcast in Tensor mode

        Returns:
            Destination register (``RegTensor``) with all lanes set to the
            broadcast value.
        """

    @staticmethod
    @_api_decl
    def load_align(tile, offset=None, dtype: Optional[DType] = None,
                   dist: Optional[LoadDist] = None,
                   data_copy_mode: Optional[DataCopyMode] = None,
                   block_stride=None, post_update: bool = False):
        """Load aligned data from a UB Tile into a VF register (vlds instruction).

        Loads contiguous data from the source UB Tile at the given element
        offset into the destination register.  Assignment form (the
        destination register is declared implicitly)::

            dst = vf.load_align(src, offset)

        The ``offset`` may be an integer, an AddrReg, or a 2-element list
        ``[row, col]`` where ``row`` is the row offset (unit = number of
        columns, i.e. ``shape[1]``) and ``col`` is the element offset within
        the row. Both ``row`` and ``col`` may be expressions::

            dst = vf.load_align(src, [i, 0])          # row i, col 0
            dst = vf.load_align(src, [i, j + k])      # expression offsets

        Args:
            src: Source UB Tile pointer
            offset: Element offset into the tile, an AddrReg, or ``[row, col]``
                list/tuple (linear offset = ``row * shape[1] + col``)

        Kwargs:
            dtype: Data type for type-specific variants (e.g. ``pl.DT_UINT32``).
                Supports all b8 types including FP8 (``DT_FP8E4M3FN``/``DT_FP8E5M2``/
                ``DT_FP8E8M0``/``DT_HF8``) and b4-packed FP4 (``DT_FP4E2M1``/
                ``DT_FP4E1M2``).
            dist: ``pl.LoadDist`` value selecting load distribution pattern
                (e.g. ``pl.LoadDist.BRC``, ``pl.LoadDist.US``, ``pl.LoadDist.BRC_B32``)
            data_copy_mode: ``pl.DataCopyMode.DATA_BLOCK_COPY`` (AscendC's name for
                the non-contiguous datablock load) selects the vsldb instruction.
                ``DATA_BLOCK_LOAD`` is accepted as an equivalent legacy alias.
            block_stride: Datablock stride in bytes for the datablock-copy mode
            post_update: ``True`` for post-increment addressing

        Returns:
            Destination register (``RegTensor``) with loaded data.
        """

    @staticmethod
    @_api_decl
    def store_align(tile, src, preg, offset, dist: Optional[StoreDist] = None,
                    data_copy_mode: Optional[DataCopyMode] = None,
                    block_stride=None, repeat_stride=None,
                    post_update: bool = False):
        """Store aligned data from a VF register to a UB Tile (vsts instruction).

        Statement form only (no assignment form --- dst is a UB tile, not a register):

            vf.store_align(dst_tile, src, preg)

        An optional offset may be appended as the last positional argument.
        It may be an integer, an AddrReg, or a 2-element list ``[row, col]``
        where ``row`` is the row offset (unit = number of columns, i.e.
        ``shape[1]``) and ``col`` is the element offset within the row::

            vf.store_align(dst_tile, src, preg, [i, j])   # list offset
            vf.store_align(dst_tile, src, preg, addr_reg)  # AddrReg offset

        When ``src`` is a MaskReg, ``preg`` is omitted (the mask_reg IS the
        source). For interleaved dual-register store (``dist=StoreDist.INTLV*``),
        use ``store_align(tile, src_even, src_odd, preg, dist=...)``.

        Args:
            tile: Destination UB Tile pointer
            src: Source register (RegTensor or MaskReg)
            preg: Predicate mask register (None when src is a MaskReg)
            offset: Optional trailing positional arg — integer, AddrReg, or
                ``[row, col]`` list/tuple (linear offset = ``row * shape[1] + col``)

        Kwargs:
            dist: ``pl.StoreDist`` value selecting the store distribution pattern.
                Coarse names (``NORM`` (default), ``FIRST_ELEMENT``, ``PACK``,
                ``PACK4``, ``INTLV``) auto-select the granularity variant from
                the source dtype; width-qualified names select it explicitly
                (mirrors AscendC Reg::StoreDist): ``NORM_B8``/``NORM_B16``/
                ``NORM_B32``, ``FIRST_ELEMENT_B8``/``_B16``/``_B32``,
                ``PACK_B16``/``_B32``/``_B64``, ``PACK4_B32``,
                ``INTLV_B8``/``_B16``/``_B32``. ``INTLV``/``INTLV_Bx``
                (interleaved) requires two src registers.
                ``pl.StoreDist.PACK`` for MaskReg src (psts PK mode)
            post_update: True to auto-advance destination address after store
            data_copy_mode: ``pl.DataCopyMode.DATA_BLOCK_COPY`` for vsstb instruction
            block_stride: DataBlock copy block stride
            repeat_stride: DataBlock copy repeat stride
        """

    @staticmethod
    @_api_decl
    def store_unalign(tile, src, align_reg, stride=None,
                      post_update: bool = False):
        """Store unaligned data from a VF register to UB (vstus/vstu instruction).

        When ``stride`` is an integer scalar, emits ``vstus`` (strided mode).
        When ``stride`` is an ``AddrReg``, emits ``vstu`` (AddrReg mode) —
        the AddrReg provides a vector of element offsets for scatter-pattern
        unaligned stores. ``vstu`` always uses POST_UPDATE.

        Args:
            dst_ptr: Destination UB pointer
            src: Source register or mask_reg
            align_reg: Alignment register (from unalign_reg_for_store)
            stride: Optional stride (int scalar → vstus) or AddrReg (→ vstu).
                Required for reg_tensor src; not used for mask_reg src.

        Kwargs:
            post_update: ``True`` to auto-advance destination address after store
        """

    @staticmethod
    @_api_decl
    def store_unalign_post(tile, align_reg, stride,
                           post_update: bool = False):
        """Complete an unaligned store sequence (vstas/vsta instruction).

        When ``stride`` is an integer scalar, emits ``vstas`` (strided mode).
        When ``stride`` is an ``AddrReg``, emits ``vsta`` (AddrReg mode) —
        must be paired with ``vstu`` from ``store_unalign`` with AddrReg.
        ``vsta`` always uses POST_UPDATE.

        Args:
            dst_ptr: Destination UB pointer
            align_reg: Alignment register (from unalign_reg_for_store)
            stride: Optional stride (int scalar → vstas) or AddrReg (→ vsta).
                Not used for mask_reg store post-processing.

        Kwargs:
            post_update: ``True`` to auto-advance destination address after store
        """

    @staticmethod
    @_api_decl
    def squeeze_store_unalign(tile, src, align_reg):
        """Squeeze-based unaligned store (vstur instruction).

        Reads the valid byte count from the AR register (written by
        ``vf.squeeze`` with ``gather_mode=STORE_REG``) as the implicit
        stride. Must be called after ``vf.squeeze(STORE_REG)`` and
        paired with ``vf.squeeze_store_unalign_post``.

        Args:
            dst_ptr: Destination UB pointer
            src: Source register
            align_reg: Alignment register (from unalign_reg_for_store)
        """

    @staticmethod
    @_api_decl
    def squeeze_store_unalign_post(tile, align_reg):
        """Complete a squeeze-based unaligned store sequence (vstar instruction).

        Reads the remaining byte count from the AR register. Must be
        called after ``vf.squeeze_store_unalign``.

        Args:
            dst_ptr: Destination UB pointer
            align_reg: Alignment register (from unalign_reg_for_store)
        """

    @staticmethod
    @_api_decl
    def unalign_reg_for_store():
        """Declare an unaligned register for store operations.

        Must be called before store_unalign/store_unalign_post to allocate the
        alignment state register.

        Returns:
            Unaligned register handle
        """

    @staticmethod
    @_api_decl
    def mem_bar(mode: Optional[MemBarMode] = None):
        """Insert a VF memory barrier (maps to AscendC ``LocalMemBar<src,dst>``).

        Orders memory ops of the ``src`` kind before those of the ``dst`` kind.
        Statement form only --- no return value.

        Select the pair via the ``mode`` kwarg (default ``VST_VLD``)::

            vf.mem_bar(mode=pl.MemBarMode.VST_VLD)   # vector store -> vector load
            vf.mem_bar(mode=pl.MemBarMode.VST_VST)   # vector store -> vector store (WAW)
            vf.mem_bar(mode=pl.MemBarMode.VV_ALL)    # all vector -> all vector

        Supported modes (12, matching AscendC's legal MemType combinations):
            VST_VLD, VLD_VST, VST_VST, VST_LD, VST_ST, VLD_ST,
            ST_VLD, ST_VST, LD_VST, VV_ALL, VS_ALL, SV_ALL
        where V*=vector, S*/*_LD/*_ST=scalar, *_ALL=full barrier of that unit.

        Kwargs:
            mode: ``pl.MemBarMode`` value selecting the src->dst ordering
        """

    @staticmethod
    @_api_decl
    def max(src0, src1, preg, mode: Optional[MergeMode] = None):
        r"""Element-wise maximum of two source registers.

        For each lane ``i`` where ``mask[i]`` is active, compares the
        corresponding elements of ``src0`` and ``src1`` and writes the larger
        value to ``dst[i]``.

        .. math:: dstReg_i = \max(srcReg0_i,\; srcReg1_i)

        Args:
            src0: First source register
            src1: Second source register
            preg: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.

        Returns:
            Destination register (``RegTensor``) holding the element-wise
            maximum of ``src0`` and ``src1`` for each active lane.
        """

    @staticmethod
    @_api_decl
    def add(src0, src1, preg, mode: Optional[MergeMode] = None):
        r"""Element-wise addition of two source registers.

        For each lane ``i`` where ``mask[i]`` is active, computes the sum of
        the corresponding elements in ``src0`` and ``src1`` and writes the
        result to ``dst[i]``.

        .. math:: dstReg_i = srcReg0_i + srcReg1_i

        Assignment form (the destination register is declared implicitly)::

            dst = vf.add(src0, src1, pred)

        Args:
            src0: First source register
            src1: Second source register
            preg: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.

        Returns:
            Destination register (``RegTensor``) holding the element-wise
            sum ``src0 + src1`` for each active lane.
        """

    @staticmethod
    @_api_decl
    def sub(src0, src1, preg, mode: Optional[MergeMode] = None):
        r"""Element-wise subtraction of two source registers.

        For each lane ``i`` where ``mask[i]`` is active, subtracts ``src1[i]``
        from ``src0[i]`` and writes the result to ``dst[i]``.

        .. math:: dstReg_i = srcReg0_i - srcReg1_i

        Args:
            src0: First source register (minuend)
            src1: Second source register (subtrahend)
            preg: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.

        Returns:
            Destination register (``RegTensor``) holding the element-wise
            difference ``src0 - src1`` for each active lane.
        """

    @staticmethod
    @_api_decl
    def mul(src0, src1, preg, mode: Optional[MergeMode] = None):
        r"""Element-wise multiplication of two source registers.

        For each lane ``i`` where ``mask[i]`` is active, computes the product
        of the corresponding elements in ``src0`` and ``src1`` and writes the
        result to ``dst[i]``.

        .. math:: dstReg_i = srcReg0_i \times srcReg1_i

        Args:
            src0: First source register
            src1: Second source register
            preg: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.

        Returns:
            Destination register (``RegTensor``) holding the element-wise
            product ``src0 * src1`` for each active lane.
        """

    @staticmethod
    @_api_decl
    def div(src0, src1, preg, mode: Optional[MergeMode] = None,
            precision: Optional[bool] = None):
        r"""Element-wise division of two source registers.

        For each lane ``i`` where ``mask[i]`` is active, divides ``src0[i]``
        by ``src1[i]`` and writes the result to ``dst[i]``.

        .. math:: dstReg_i = srcReg0_i \div srcReg1_i

        Args:
            src0: Numerator register
            src1: Denominator register
            preg: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.
            precision: When ``True``, enables high-precision mode using the
                error-compensation algorithm (0-ulp precision error). Only
                effective for ``DT_FP32`` source type. Default ``False``
                (standard mode).

        Returns:
            Destination register (``RegTensor``) holding the element-wise
            quotient ``src0 / src1`` for each active lane.
        """

    @staticmethod
    @_api_decl
    def muls(src, scalar, preg, mode: Optional[MergeMode] = None):
        r"""Multiply all elements by a scalar.

        For each lane ``i`` where ``mask[i]`` is active, multiplies ``src[i]``
        by the scalar value and writes the result to ``dst[i]``.

        .. math:: dstReg_i = srcReg_i \times scalar

        Args:
            src: Source register
            scalar: Scalar multiplier value
            preg: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.

        Returns:
            Destination register (``RegTensor``) holding ``src * scalar``
            for each active lane.

        """

    @staticmethod
    @_api_decl
    def mul_add_dst(src0, src1, preg, mode: Optional[MergeMode] = None):
        r"""Fused multiply-add into destination.

        For each lane ``i`` where ``mask[i]`` is active, multiplies ``src0[i]``
        by ``src1[i]``, adds the product to the current value of ``dst[i]``,
        and writes the sum back to ``dst[i]``.  The destination register is
        both read (as addend) and written.  Maps to hardware ``vmula``
        instruction.

        .. math:: dstReg_i = srcReg0_i \times srcReg1_i + dstReg_i

        Args:
            src0: First multiplicand register
            src1: Second multiplicand register
            preg: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.

        Returns:
            Destination register (``RegTensor``), updated in-place.
        """

    @staticmethod
    @_api_decl
    def and_(src0, src1, preg, mode: Optional[MergeMode] = None):
        r"""Element-wise bitwise AND of two source registers.

        For each lane ``i`` where ``mask[i]`` is active, computes the bitwise
        AND of ``src0[i]`` and ``src1[i]`` and writes the result to
        ``dst[i]``.

        .. math:: dstReg_i = srcReg0_i \;\&\; srcReg1_i

        Args:
            src0: First source register
            src1: Second source register
            preg: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.

        Returns:
            Destination register (``RegTensor``) holding the element-wise
            bitwise AND of ``src0`` and ``src1``.
        """

    @staticmethod
    @_api_decl
    def or_(src0, src1, preg, mode: Optional[MergeMode] = None):
        r"""Element-wise bitwise OR of two source registers.

        For each lane ``i`` where ``mask[i]`` is active, computes the bitwise
        OR of ``src0[i]`` and ``src1[i]`` and writes the result to
        ``dst[i]``.

        .. math:: dstReg_i = srcReg0_i \;|\; srcReg1_i

        Args:
            src0: First source register
            src1: Second source register
            preg: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.

        Returns:
            Destination register (``RegTensor``) holding the element-wise
            bitwise OR of ``src0`` and ``src1``.
        """

    @staticmethod
    @_api_decl
    def xor(src0, src1, preg, mode: Optional[MergeMode] = None,
            dtype: Optional[DType] = None):
        r"""Element-wise bitwise XOR of two source registers.

        For each lane ``i`` where ``mask[i]`` is active, computes the bitwise
        XOR of ``src0[i]`` and ``src1[i]`` and writes the result to
        ``dst[i]``.

        .. math:: dstReg_i = srcReg0_i \;\oplus\; srcReg1_i

        Args:
            src0: First source register
            src1: Second source register
            preg: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.
            dtype: Data type for type-specific variants (e.g. ``pl.DT_UINT16``)

        Returns:
            Destination register (``RegTensor``) holding the element-wise
            bitwise XOR of ``src0`` and ``src1``.
        """

    @staticmethod
    @_api_decl
    def reduce_sum(src, preg, datablock: bool = False,
                   merge_mode: Optional[MergeMode] = None):
        r"""In-register sum reduction across all lanes (vcadd / vcgadd).

        Reduces all active lanes of the source register into the first element
        of the destination register.  The remaining lanes are zeroed.

        .. math:: dstReg_0 = \sum_{i \in \text{active}} srcReg_i

        Args:
            src: Source register
            preg: Predicate mask register

        Kwargs:
            datablock: ``True`` to use datablock-granularity reduction (vcgadd)
            merge_mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.

        Returns:
            Destination register (``RegTensor``) with the reduction result in
            lane 0.
        """

    @staticmethod
    @_api_decl
    def reduce_max(src, preg, datablock: bool = False,
                   merge_mode: Optional[MergeMode] = None):
        r"""In-register max reduction across all lanes (vcmax / vcgmax).

        Reduces all active lanes of the source register into the first element
        of the destination register.  The remaining lanes are zeroed.

        .. math:: dstReg_0 = \max_{i \in \text{active}} srcReg_i

        Args:
            src: Source register
            preg: Predicate mask register

        Kwargs:
            datablock: ``True`` to use datablock-granularity reduction (vcgmax)
            merge_mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.

        Returns:
            Destination register (``RegTensor``) with the reduction result in
            lane 0.
        """

    @staticmethod
    @_api_decl
    def reduce_min(src, preg, datablock: bool = False,
                   merge_mode: Optional[MergeMode] = None):
        r"""In-register min reduction across all lanes (vcmin / vcgmin).

        Reduces all active lanes of the source register into the first element
        of the destination register.  The remaining lanes are zeroed.

        .. math:: dstReg_0 = \min_{i \in \text{active}} srcReg_i

        Args:
            src: Source register
            preg: Predicate mask register

        Kwargs:
            datablock: ``True`` to use datablock-granularity reduction (vcgmin)
            merge_mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.

        Returns:
            Destination register (``RegTensor``) with the reduction result in
            lane 0.
        """

    @staticmethod
    @_api_decl
    def ln(src, preg, mode: Optional[MergeMode] = None,
           precision: Optional[bool] = None):
        r"""Natural logarithm of each element.

        For each lane ``i`` where ``mask[i]`` is active, computes the natural
        logarithm of ``src[i]`` and writes the result to ``dst[i]``.  The
        source must be positive; non-positive values produce undefined
        results.  Maps to hardware ``vln`` instruction.

        .. math:: dstReg_i = \ln(srcReg_i)

        Args:
            src: Source register (must be positive)
            preg: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.
            precision: When ``True``, enables high-precision mode that
                preserves subnormal output results (1-ulp precision error).
                Only effective for ``DT_FP32`` source type. Default ``False``
                (standard mode, subnormal outputs are flush-to-zero).

        Returns:
            Destination register (``RegTensor``) holding the natural
            logarithm ``ln(src)`` for each active lane.
        """

    @staticmethod
    @_api_decl
    def exp_sub(src0, src1, preg, layout: Optional[CastLayout] = None):
        r"""Fused exp-subtract for numerical stability.

        For each lane ``i`` where ``mask[i]`` is active, subtracts ``max_val[i]``
        from ``src[i]`` for numerical stability, then computes the exponential
        of the difference and writes the result to ``dst[i]``.  Commonly used
        in softmax computation.

        .. math:: dstReg_i = e^{\,srcReg_i - maxVal_i}

        Args:
            src: Source register
            max_val: Max register (subtracted before exp)
            preg: Predicate mask register

        Kwargs:
            layout: ``pl.CastLayout.ZERO`` (even half, default) or
                ``pl.CastLayout.ONE`` (odd half) --- for half-width results

        Returns:
            Destination register (``RegTensor``) holding ``e^(src - max_val)``
            for each active lane.
        """

    @staticmethod
    @_api_decl
    def astype(src, preg, dtype: DType,
               layout: Optional[CastLayout] = None,
               round_mode: Optional[VFRoundMode] = None,
               saturate: Optional[SaturateMode] = None,
               mode: Optional[MergeMode] = None):
        r"""Type conversion between register types (vcvt instruction).

        Converts each element of ``src`` to the destination data type.  For
        each lane ``i`` where ``mask[i]`` is active, the converted value is
        written to ``dst[i]``.  Supports same-width and cross-width
        conversions: float->int, int->int narrowing/widening, float precision
        changes, and low-precision FP8/FP4 conversions.

        **FP8/FP4 conversion support** (Ascend 950PR/950DT only):

        ==================================  ==================================
        Source type                         Target type
        ==================================  ==================================
        ``DT_HF8``                          ``DT_FP16``, ``DT_FP32``
        ``DT_FP8E4M3FN``                    ``DT_FP32``
        ``DT_FP8E5M2``                      ``DT_FP32``
        ``DT_FP4E2M1``                      ``DT_BF16``
        ``DT_FP4E1M2``                      ``DT_BF16``
        ``DT_FP16``                         ``DT_HF8``
        ``DT_FP32``                         ``DT_HF8``, ``DT_FP8E4M3FN``, ``DT_FP8E5M2``
        ``DT_BF16``                         ``DT_FP4E2M1``, ``DT_FP4E1M2``
        ==================================  ==================================

        Args:
            src: Source register (source type)
            preg: Predicate mask register

        Kwargs:
            layout: ``pl.CastLayout.ZERO`` (default) / ``ONE`` / ``TWO`` / ``THREE``
            round_mode: ``pl.VFRoundMode.CAST_RINT`` (default) / ``CAST_ROUND`` /
                ``CAST_FLOOR`` / ``CAST_CEIL`` / ``CAST_TRUNC`` /
                ``CAST_ODD`` / ``CAST_HYBRID``
            saturate: ``pl.SaturateMode.OFF`` (default) or ``pl.SaturateMode.ON``
            mode: ``pl.MergeMode.ZEROING`` (default). Current device only supports
                ZEROING mode. Non-active lanes are zeroed.

        Returns:
            Destination register (``RegTensor``) with the converted type.
        """

    @staticmethod
    @_api_decl
    def de_interleave(src0, src1, dtype: Optional[DType] = None):
        r"""De-interleave: split even/odd elements into two registers.

        Given two interleaved source registers, extracts the even-indexed
        elements into ``dst0`` and the odd-indexed elements into ``dst1``.

        .. math::
            dstReg0_i = srcReg_{2i}     \quad \text{(even elements)}
            dstReg1_i = srcReg_{2i+1}   \quad \text{(odd elements)}

        Tuple assignment form (the destination registers are declared
        implicitly)::

            dst0, dst1 = vf.de_interleave(src0, src1)

        Args:
            src0: First source register
            src1: Second source register

        Kwargs:
            dtype: When src operands are MaskReg, specifies the interleave bit-width
                (selects ``pdintlv_b8``/``b16``/``b32``). Inferred from src0 if omitted.

        Returns:
            Tuple of ``(dst0, dst1)``: two ``RegTensor`` registers containing
            even and odd elements respectively.
        """

    @staticmethod
    @_api_decl
    def select(src0, src1, preg):
        r"""Conditional select between two source registers.

        For each lane ``i``, selects ``src_true[i]`` when ``mask[i]`` is active
        and ``src_false[i]`` when ``mask[i]`` is inactive, writing the selected
        value to ``dst[i]``.

        .. math::
            dstReg_i = \begin{cases} srcTrueReg_i & \text{if } mask_i = 1 \\
            srcFalseReg_i & \text{if } mask_i = 0 \end{cases}

        Args:
            src_true: Register selected when mask bit is 1
            src_false: Register selected when mask bit is 0
            preg: Predicate mask register

        Returns:
            Destination register (``RegTensor``) holding the selected
            elements based on mask polarity.
        """

    @staticmethod
    @_api_decl
    def shift_left(src, shift, preg, mode: Optional[MergeMode] = None):
        """Left shift: ``dst[i] = src[i] << shift``

        The shift amount may be a scalar (all lanes shifted by the same amount,
        emits ``vshls``) or a vector register (per-lane shift amount, emits
        ``vshl``). The form is selected automatically from the argument type::

            dst = vf.shift_left(src, 2, preg)          # scalar: uniform shift
            dst = vf.shift_left(src, shift_reg, preg)   # register: per-lane shift

        This unified op replaces the former separate ``vf.shift_lefts`` (scalar)
        and ``vf.shift_left`` (vector) entry points.

        Args:
            src: Source register
            shift: Shift amount --- scalar integer (uniform) or a vector register
                (per-lane). Negative values are undefined.
            preg: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.

        Returns:
            Destination register (``RegTensor``) holding ``src[i] << shift[i]``
            for each active lane.
        """

    @staticmethod
    @_api_decl
    def shift_right(src, shift, preg, mode: Optional[MergeMode] = None):
        """Right shift: ``dst[i] = src[i] >> shift``

        The shift amount may be a scalar (all lanes shifted by the same amount,
        emits ``vshrs``) or a vector register (per-lane shift amount, emits
        ``vshr``). The form is selected automatically from the argument type::

            dst = vf.shift_right(src, 24, preg)         # scalar: uniform shift
            dst = vf.shift_right(src, shift_reg, preg)   # register: per-lane shift

        This unified op replaces the former separate ``vf.shift_rights`` (scalar)
        and ``vf.shift_right`` (vector) entry points. Unsigned src does a logical
        shift; signed src does an arithmetic shift.

        Args:
            src: Source register
            shift: Shift amount --- scalar integer (uniform) or a vector register
                (per-lane). Negative values are undefined.
            preg: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.

        Returns:
            Destination register (``RegTensor``) holding ``src[i] >> shift[i]``
            for each active lane (logical shift for unsigned src, arithmetic
            shift for signed src).
        """

    @staticmethod
    @_api_decl
    def histograms(src, preg, bin_type: Optional[BinType] = None,
                   hist_type: Optional[HistType] = None):
        """Histogram accumulation (chistv2/dhistv2 instruction).

        Computes histogram on UINT8 data. Supports both cumulative (chistv2)
        and frequency (dhistv2) modes.  The destination register is both read
        and written --- ``chistv2`` accumulates into the existing value, so the
        dst must be pre-initialized (e.g. via ``vf.full(0, ...)``) before the
        first call.  Subsequent ``dst = vf.histograms(...)`` calls reuse the
        same register and continue accumulating.

        Args:
            src: Source register (data to bin)
            preg: Predicate mask register

        Kwargs:
            bin_type: ``pl.BinType.BIN0`` (default) or ``pl.BinType.BIN1`` --- selects bin mapping
            hist_type: ``pl.HistType.ACCUMULATE`` (default, chistv2) or
                ``pl.HistType.FREQUENCY`` (dhistv2)

        Returns:
            Destination histogram register (same register passed as dst,
            with accumulated histogram values)
        """

    @staticmethod
    @_api_decl
    def eq(src0, src1, preg):
        """Element-wise equality comparison.

        Compares two source elements and writes the result to the
        destination mask register.  If the second argument is a scalar
        literal the vector-scalar compare path is used; otherwise the
        vector-vector compare path is used.

        Args:
            src0: First source register
            src1: Second source register or scalar value
            preg: Source predicate mask

        Returns:
            MaskReg with comparison result (True where src0_i == src1_i)
        """

    @staticmethod
    @_api_decl
    def ne(src0, src1, preg):
        """Element-wise not-equal comparison.

        Args:
            src0: First source register
            src1: Second source register or scalar value
            preg: Source predicate mask

        Returns:
            MaskReg with comparison result (True where src0_i != src1_i)
        """

    @staticmethod
    @_api_decl
    def lt(src0, src1, preg):
        """Element-wise less-than comparison.

        Args:
            src0: First source register
            src1: Second source register or scalar value
            preg: Source predicate mask

        Returns:
            MaskReg with comparison result (True where src0_i < src1_i)
        """

    @staticmethod
    @_api_decl
    def gt(src0, src1, preg):
        """Element-wise greater-than comparison.

        Args:
            src0: First source register
            src1: Second source register or scalar value
            preg: Source predicate mask

        Returns:
            MaskReg with comparison result (True where src0_i > src1_i)
        """

    @staticmethod
    @_api_decl
    def le(src0, src1, preg):
        """Element-wise less-or-equal comparison.

        Args:
            src0: First source register
            src1: Second source register or scalar value
            preg: Source predicate mask

        Returns:
            MaskReg with comparison result (True where src0_i <= src1_i)
        """

    @staticmethod
    @_api_decl
    def ge(src0, src1, preg):
        """Element-wise greater-or-equal comparison.

        Args:
            src0: First source register
            src1: Second source register or scalar value
            preg: Source predicate mask

        Returns:
            MaskReg with comparison result (True where src0_i >= src1_i)
        """

    @staticmethod
    @_api_decl
    def squeeze(src, preg, gather_mode: Optional[SqueezeMode] = None):
        """Squeeze mask to index register (vsqz instruction).

        Converts active mask bits into a packed index sequence in the
        destination register.  For each active lane in ``mask``, the
        corresponding element of ``src`` is compressed (squeezed) into a
        contiguous region at the start of ``dst``.

        Args:
            src: Source register
            preg: Predicate mask register

        Kwargs:
            gather_mode: ``pl.SqueezeMode.STORE_REG`` or ``pl.SqueezeMode.NO_STORE_REG``

        Returns:
            Destination register (``RegTensor``) with packed/squeezed elements.
        """

    @staticmethod
    @_api_decl
    def arange(start, dtype: DType,
               index_order: Optional[IndexOrder] = None):
        r"""Generate an index sequence starting from ``start`` (vci instruction).

        Fills each lane with a sequential value.  The per-lane step is fixed
        at +/-1 (a hardware property of vci); use a following ``vf.muls`` to
        scale the step if needed.  ``index_order`` selects the direction::

            dst = vf.arange(start)                                          # dst[i] = start + i (INC)
            dst = vf.arange(start, index_order=pl.IndexOrder.INCREASE_ORDER)  # same as default
            dst = vf.arange(start, index_order=pl.IndexOrder.DECREASE_ORDER)  # dst[i] = start - i (DEC)

        .. math::
            dstReg_i = start + i \quad (\text{INC, default})
            dstReg_i = start - i \quad (\text{DEC})

        Args:
            start: Starting value (scalar)

        Kwargs:
            index_order: ``pl.IndexOrder.INCREASE_ORDER`` (default, dst[i]=start+i)
                or ``pl.IndexOrder.DECREASE_ORDER`` (dst[i]=start-i)

        Returns:
            Destination register (``RegTensor``) with sequential values.
        """

    @staticmethod
    @_api_decl
    def gather(src, index, preg,
               data_copy_mode: Optional[DataCopyMode] = None):
        r"""Gather elements by index.

        Two forms, dispatched by src argument type:

        - **Tile→Reg**: ``src`` is a UB tile, ``preg`` required.
          For each active lane ``i``, loads ``src[index[i]]`` into ``dst[i]``.
          NORM mode: per-element gather. DATA_BLOCK_LOAD mode: per-32B-DataBlock gather.
        - **Reg→Reg**: ``src`` is a register, ``preg`` omitted.
          For each lane ``i``, copies ``src[index[i]]`` into ``dst[i]``.
          Source and destination must have the same data type.

        .. math:: dstReg_i = src[\text{index}_i]

        Args:
            src: Source UB tile or register
            index: Index register
            preg: Predicate mask register (required for Tile→Reg, omitted for Reg→Reg)

        Kwargs:
            data_copy_mode: ``pl.DataCopyMode.NORM`` (default, per-element)
                or ``pl.DataCopyMode.DATA_BLOCK_LOAD`` (per 32B datablock)

        Returns:
            Destination register (``RegTensor``) with gathered elements.
        """

    @staticmethod
    @_api_decl
    def clear_spr():
        """Clear special purpose register (AR register).

        Resets the accumulator register used by certain VF instructions.
        Statement form only --- no return value.
        """

    @staticmethod
    @_api_decl
    def log(src, preg, mode: Optional[MergeMode] = None,
            precision: Optional[bool] = None):
        r"""Natural logarithm (alias for :func:`vf.ln`).

        Convenience wrapper that maps to the same ``vln`` hardware instruction
        as :func:`vf.ln`.  Accepts identical arguments; see :func:`vf.ln` for
        the full parameter and type details.

        .. math:: dstReg_i = \log(srcReg_i)

        Args:
            src: Source register (must be positive)
            preg: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.
            precision: When ``True``, enables high-precision mode that
                preserves subnormal output results (1-ulp precision error).
                Only effective for ``DT_FP32`` source type. Default ``False``
                (standard mode, subnormal outputs are flush-to-zero).

        Returns:
            Destination register (``RegTensor``) with ``ln(src)`` per lane.
        """

    @staticmethod
    @_api_decl
    def min(src0, src1, preg, mode: Optional[MergeMode] = None):
        r"""Element-wise minimum of two source registers.

        For each lane ``i`` where ``mask[i]`` is active, compares the
        corresponding elements of ``src0`` and ``src1`` and writes the smaller
        value to ``dst[i]``.

        .. math:: dstReg_i = \min(srcReg0_i,\; srcReg1_i)

        Args:
            src0: First source register
            src1: Second source register
            preg: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.

        Returns:
            Destination register (``RegTensor``) holding the element-wise
            minimum of ``src0`` and ``src1`` for each active lane.
        """

    @staticmethod
    @_api_decl
    def exp(src, preg, mode: Optional[MergeMode] = None,
            precision: Optional[bool] = None):
        r"""Exponential function of each element.

        For each lane ``i`` where ``mask[i]`` is active, computes ``e``
        raised to the power of ``src[i]`` and writes the result to
        ``dst[i]``.  Maps to hardware ``vexp`` instruction.

        .. math:: dstReg_i = e^{srcReg_i}

        Args:
            src: Source register
            preg: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.
            precision: When ``True``, enables high-precision mode that
                preserves subnormal output results (1-ulp precision error).
                Only effective for ``DT_FP32`` source type. Default ``False``
                (standard mode, subnormal outputs are flush-to-zero).

        Returns:
            Destination register (``RegTensor``) holding ``e^src``
            for each active lane.
        """

    @staticmethod
    @_api_decl
    def abs(src, preg, mode: Optional[MergeMode] = None):
        r"""Element-wise absolute value.

        For each lane ``i`` where ``mask[i]`` is active, computes the absolute
        value of ``src[i]`` and writes the result to ``dst[i]``.

        .. math:: dstReg_i = |srcReg_i|

        Args:
            src: Source register
            preg: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.

        Returns:
            Destination register (``RegTensor``) holding the absolute
            value ``|src|`` for each active lane.
        """

    @staticmethod
    @_api_decl
    def not_(src, preg, mode: Optional[MergeMode] = None):
        r"""Element-wise bitwise NOT.

        For each lane ``i`` where ``mask[i]`` is active, computes the bitwise
        NOT (one's complement) of ``src[i]`` and writes the result to
        ``dst[i]``.

        .. math:: dstReg_i = \sim srcReg_i

        Args:
            src: Source register
            preg: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.

        Returns:
            Destination register (``RegTensor``) holding the bitwise
            NOT ``~src`` for each active lane.
        """

    @staticmethod
    @_api_decl
    def sqrt(src, preg, mode: Optional[MergeMode] = None,
             precision: Optional[bool] = None):
        r"""Square root of each element.

        For each lane ``i`` where ``mask[i]`` is active, computes the square
        root of ``src[i]`` and writes the result to ``dst[i]``.  The source
        must be non-negative; negative values produce undefined results.
        Maps to hardware ``vsqrt`` instruction.

        .. math:: dstReg_i = \sqrt{srcReg_i}

        Args:
            src: Source register (must be non-negative)
            preg: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.
            precision: When ``True``, enables high-precision mode that
                preserves subnormal output results (1-ulp precision error).
                Only effective for ``DT_FP32`` source type. Default ``False``
                (standard mode, subnormal outputs are flush-to-zero).

        Returns:
            Destination register (``RegTensor``) holding the square
            root ``sqrt(src)`` for each active lane.
        """

    @staticmethod
    @_api_decl
    def relu(src, preg, mode: Optional[MergeMode] = None):
        r"""ReLU activation.

        For each lane ``i`` where ``mask[i]`` is active, writes ``src[i]`` to
        ``dst[i]`` if ``src[i] >= 0``, otherwise writes 0.

        .. math:: dstReg_i = \max(0,\; srcReg_i)

        Args:
            src: Source register
            preg: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.

        Returns:
            Destination register (``RegTensor``) holding ``max(0, src)``
            for each active lane.
        """

    @staticmethod
    @_api_decl
    def neg(src, preg, mode: Optional[MergeMode] = None):
        r"""Element-wise negation.

        For each lane ``i`` where ``mask[i]`` is active, computes the
        arithmetic negation of ``src[i]`` and writes the result to ``dst[i]``.

        .. math:: dstReg_i = -srcReg_i

        Args:
            src: Source register
            preg: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device..

        Returns:
            Destination register (``RegTensor``) holding the negated
            value ``-src`` for each active lane.
        """

    @staticmethod
    @_api_decl
    def adds(src, scalar, preg, mode: Optional[MergeMode] = None):
        r"""Add scalar to each element.

        For each lane ``i`` where ``mask[i]`` is active, adds the scalar value
        to ``src[i]`` and writes the result to ``dst[i]``.

        .. math:: dstReg_i = srcReg_i + scalar

        Args:
            src: Source register
            scalar: Scalar addend value
            preg: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.

        Returns:
            Destination register (``RegTensor``) holding ``src + scalar``
            for each active lane.
        """

    @staticmethod
    @_api_decl
    def mins(src, scalar, preg, mode: Optional[MergeMode] = None):
        r"""Element-wise minimum with scalar.

        For each lane ``i`` where ``mask[i]`` is active, compares ``src[i]``
        with the scalar value and writes the smaller one to ``dst[i]``.

        .. math:: dstReg_i = \min(srcReg_i,\; scalar)

        Args:
            src: Source register
            scalar: Scalar value to compare against
            preg: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.

        Returns:
            Destination register (``RegTensor``) holding the element-wise
            minimum of ``src`` and ``scalar`` for each active lane.
        """

    @staticmethod
    @_api_decl
    def maxs(src, scalar, preg, mode: Optional[MergeMode] = None):
        r"""Element-wise maximum with scalar.

        For each lane ``i`` where ``mask[i]`` is active, compares ``src[i]``
        with the scalar value and writes the larger one to ``dst[i]``.

        .. math:: dstReg_i = \max(srcReg_i,\; scalar)

        Args:
            src: Source register
            scalar: Scalar value to compare against
            preg: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.

        Returns:
            Destination register (``RegTensor``) holding the element-wise
            maximum of ``src`` and ``scalar`` for each active lane.
        """

    @staticmethod
    @_api_decl
    def leaky_relu(src, scalar, preg, mode: Optional[MergeMode] = None):
        r"""Leaky ReLU activation.

        For each lane ``i`` where ``mask[i]`` is active, writes ``src[i]`` to
        ``dst[i]`` if ``src[i] >= 0``, otherwise writes ``alpha * src[i]``.

        .. math::
            dstReg_i = \begin{cases} srcReg_i & \text{if } srcReg_i \geq 0 \\
            \alpha \times srcReg_i & \text{if } srcReg_i < 0 \end{cases}

        Args:
            src: Source register
            alpha: Negative slope scalar (e.g. 0.1)
            preg: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.

        Returns:
            Destination register (``RegTensor``) holding the Leaky
            ReLU result for each active lane.
        """

    @staticmethod
    @_api_decl
    def interleave(src0, src1, dtype: Optional[DType] = None):
        r"""Interleave two registers: combine even/odd elements.

        Interleaves elements from ``src0`` and ``src1`` so that ``src0``
        elements occupy even positions and ``src1`` elements occupy odd
        positions in the output.

        .. math::
            dstReg0_{2i}   = srcReg0_i
            dstReg0_{2i+1} = srcReg1_i

        Tuple assignment form (the destination registers are declared
        implicitly)::

            dst0, dst1 = vf.interleave(src0, src1)

        Args:
            src0: First source register
            src1: Second source register

        Kwargs:
            dtype: When src operands are MaskReg, specifies the interleave bit-width
                (selects ``pintlv_b8``/``b16``/``b32``). Inferred from src0 if omitted.

        Returns:
            Tuple of ``(dst0, dst1)``: two ``RegTensor`` registers containing
            interleaved elements.
        """

    @staticmethod
    @_api_decl
    def pair_reduce_sum(src, preg, mode: Optional[MergeMode] = None):
        r"""Pairwise reduction sum.

        For each pair of adjacent elements, adds them together and writes the
        result to ``dst[i]``.  Maps to hardware ``vcpadd`` instruction.

        .. math:: dstReg_i = srcReg_{2i} + srcReg_{2i+1}

        Args:
            src: Source register
            preg: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.

        Returns:
            Destination register (``RegTensor``) with pairwise sums.
        """

    @staticmethod
    @_api_decl
    def abs_sub(src0, src1, preg, mode: Optional[MergeMode] = None):
        r"""Absolute difference of two source registers.

        For each lane ``i`` where ``mask[i]`` is active, computes the absolute
        value of the difference between ``src0[i]`` and ``src1[i]`` and writes
        the result to ``dst[i]``.  Maps to hardware ``vabsdif`` instruction.

        .. math:: dstReg_i = |srcReg0_i - srcReg1_i|

        Args:
            src0: First source register
            src1: Second source register
            preg: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.

        Returns:
            Destination register (``RegTensor``) holding the absolute
            difference ``|src0 - src1|`` for each active lane.
        """

    @staticmethod
    @_api_decl
    def axpy(src, scalar, preg, mode: Optional[MergeMode] = None):
        r"""Fused AXPY: multiply src by scalar and add to dst.

        For each lane ``i`` where ``mask[i]`` is active, multiplies ``src[i]``
        by the scalar, adds the product to the current value of ``dst[i]``,
        and writes the sum back to ``dst[i]``.  The destination register is
        both read (as addend) and written.  Maps to hardware ``vaxpy``
        instruction.

        .. math:: dstReg_i = srcReg_i \times scalar + dstReg_i

        Args:
            src: Source register
            scalar: Scalar multiplier
            preg: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.

        Returns:
            Destination register (``RegTensor``), updated in-place.
        """

    @staticmethod
    @_api_decl
    def bit_cast(src, *, dtype: DType):
        r"""Bitwise type reinterpretation of a register.

        Returns a view of ``src`` reinterpreted as the target ``dtype``,
        without converting any element values. The bit pattern is preserved
        exactly; only the C++ type used for code generation changes.

        This is primarily used as an argument inside other ``vf.xxx`` calls
        to satisfy AscendC instruction type requirements. For example, to
        perform ``vf.xor`` on FP8E4M3 registers:

        .. code-block:: python

            reg_c = vf.xor(vf.bit_cast(reg_a, dtype=pl.DT_FP32),
                           vf.bit_cast(reg_b, dtype=pl.DT_FP32), preg)

        The generated C++ code applies ``RegTensor<T>&`` reference casts:

        .. code-block:: cpp

            RegTensor<float> reg_c;
            vxor(reg_c, (RegTensor<float>&)reg_a,
                 (RegTensor<float>&)reg_b, preg, MODE_ZEROING);

        Args:
            src: Source register (``RegTensor``)

        Kwargs:
            dtype: Target data type (e.g. ``pl.DT_FP32``, ``pl.DT_UINT16``)

        Returns:
            A type-annotated reference to ``src``, for use as an argument
            in other ``vf.xxx`` calls.
        """

    @staticmethod
    @_api_decl
    def mul_dst_add(src0, src1, preg, mode: Optional[MergeMode] = None):
        r"""Multiply-dst-add: multiply dst by src0, then add src1.

        For each lane ``i`` where ``mask[i]`` is active, multiplies ``dst[i]``
        by ``src0[i]``, adds ``src1[i]`` to the product, and writes the sum
        back to ``dst[i]``.  The destination register is both read (as
        multiplicand) and written.  Maps to hardware ``vmadd`` instruction
        (AscendC MulDstAdd).

        .. math:: dstReg_i = dstReg_i \times srcReg0_i + srcReg1_i

        Args:
            src0: First multiplicand register
            src1: Addend register
            preg: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.

        Returns:
            Destination register (``RegTensor``), updated in-place.
        """

    @staticmethod
    @_api_decl
    def pack(src, part: Optional[PackPart] = None,
             dtype: Optional[DType] = None):
        r"""Pack/narrow data type (e.g. u32->u16, u16->u8).

        For each element of ``src``, extracts the low bits of the element
        (low 8 bits for b16 sources, low 16 bits for b32 sources, low 32
        bits for b64 sources) and writes them into the low half or the high
        half of ``dst`` as selected by ``part``; the other half of ``dst``
        is zero. ``part`` selects the destination half — it does NOT choose
        between the upper and lower bits of the source element.

        Args:
            src: Source register (wider type)

        Kwargs:
            part: ``pl.PackPart.LOWER`` (default, writes into dst's low half)
                or ``pl.PackPart.UPPER`` (writes into dst's high half)
            dtype: Destination data type (e.g. ``pl.DT_UINT16``)

        Returns:
            Destination register (``RegTensor``) with the narrowed type.
        """

    @staticmethod
    @_api_decl
    def unpack(src, dtype: Optional[DType] = None,
               part: Optional[PackPart] = None):
        """Unpack/widen data type (e.g. u8->u16, u16->u32).

        Zero-extends or sign-extends narrower elements into wider type.

        Args:
            src: Source register (narrower type)

        Kwargs:
            dtype: Destination data type (e.g. ``pl.DT_UINT32``)
            part: ``pl.PackPart.LOWER`` (default) or ``pl.PackPart.UPPER`` --- which half of src to unpack

        Returns:
            Destination register (``RegTensor``) holding the widened elements
            of ``src``.
        """

    @staticmethod
    @_api_decl
    def prelu(src, slope, preg, mode: Optional[MergeMode] = None):
        """Parametric ReLU with per-element slope register.

        ``dst[i] = src[i] if src[i] >= 0 else src[i] * slope[i]``

        Unlike leaky_relu which uses a scalar alpha, prelu uses a per-element
        slope vector.

        Args:
            src: Source register
            slope: Slope register (per-element negative slope values)
            preg: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.

        Returns:
            Destination register (``RegTensor``) holding ``src[i]`` where
            ``src[i] >= 0`` and ``src[i] * slope[i]`` otherwise, for each
            active lane.
        """

    @staticmethod
    @_api_decl
    def mull(src0, src1, preg):
        """Long multiply: 32x32->64, output split into lo/hi register pair.

        Multiplies two 32-bit registers and produces 64-bit result split
        across two 32-bit destination registers.

        Tuple assignment form (the destination registers are declared implicitly):

            dst_lo, dst_hi = vf.mull(src0, src1, pred)

        Args:
            src0: First source register (32-bit)
            src1: Second source register (32-bit)
            preg: Predicate mask register

        Returns:
            Tuple of destination registers ``(dst_lo, dst_hi)`` holding the low
            and high 32 bits of the 64-bit product ``src0 * src1``.
        """

    @staticmethod
    @_api_decl
    def addc(src0, src1, carry_src, preg):
        """Add with carry (vaddcs): ``carry_out, dst = src0 + src1 + carry_in``

        Used for multi-word (e.g. 64-bit) arithmetic on 32-bit registers.
        Produces two outputs via tuple unpacking --- the carry-out flag register
        (declared as a MaskReg) and the sum register (RegTensor)::

            carry_out, dst = vf.addc(src0, src1, carry_src, preg)

        Args:
            src0: First source register
            src1: Second source register
            carry_in: Input carry flag register (MaskReg)
            preg: Predicate mask register

        Returns:
            (carry_out, dst): carry-out flag register and the sum register
        """

    @staticmethod
    @_api_decl
    def subc(src0, src1, borrow_src, preg):
        """Subtract with borrow (vsubcs): ``borrow_out, dst = src0 - src1 - borrow_in``

        Used for multi-word (e.g. 64-bit) arithmetic on 32-bit registers.
        Produces two outputs via tuple unpacking --- the borrow-out flag register
        (declared as a MaskReg) and the difference register (RegTensor)::

            borrow_out, dst = vf.subc(src0, src1, borrow_src, preg)

        Args:
            src0: First source register
            src1: Second source register
            borrow_in: Input borrow flag register (MaskReg)
            preg: Predicate mask register

        Returns:
            (borrow_out, dst): borrow-out flag register and the difference register
        """

    @staticmethod
    @_api_decl
    def load_unalign_init():
        """Declare an unaligned register for load operations.

        Must be called before load_unalign_pre/load_unalign to allocate
        the alignment state register.

        Returns:
            Unaligned register handle
        """

    @staticmethod
    @_api_decl
    def load_unalign_pre(align_reg, tile):
        """Setup unaligned load (vldas instruction).

        Initializes the alignment state for subsequent unaligned loads.

        Args:
            ureg: UnalignRegForLoad register (from load_unalign_init)
            src_ptr: Source UB pointer
        """

    @staticmethod
    @_api_decl
    def load_unalign(tile, align_reg, stride=None, post_update: bool = False):
        """Unaligned load from UB to register (vldus instruction).

        Loads data from an unaligned UB address. Supports optional stride
        for POST_UPDATE mode.

        Args:
            ureg: UnalignRegForLoad register
            src_ptr: Source UB pointer
            stride: Optional post-update stride in bytes

        Returns:
            Destination register (``RegTensor``) holding the data loaded from
            the unaligned UB address.
        """

    @staticmethod
    @_api_decl
    def scatter(tile, src, index, preg):
        """Scatter store by index (vscatter instruction).

        Writes register elements to non-contiguous UB locations specified
        by an index register.

        Args:
            base_ptr: Base UB pointer
            src: Source register (data to scatter)
            index: Index register (destination offsets)
            preg: Predicate mask register
        """

    @staticmethod
    @_api_decl
    def unsqueeze(preg):
        """Unsqueeze mask bits into a register (vusqz instruction).

        Expands each mask bit into the corresponding register lane
        (1 for active, 0 for inactive).

        Args:
            preg: Mask register to unsqueeze

        Returns:
            Destination register (``RegTensor``) holding one lane per mask bit:
            1 for active bits, 0 for inactive bits.
        """

    @staticmethod
    @_api_decl
    def get_mask_spr(width: MaskWidth = MaskWidth.B32):
        """Get mask from special purpose register (movp_b32/movp_b16).

        Reads the {MASK1, MASK0} SPR set by SetVectorMask and converts it
        to a MaskReg. This is the pypto equivalent of AscendC ``MoveMask<T>``.

        - ``width=pl.MaskWidth.B32``: reads 64-bit MASK0, expands each bit to 4 bits (movp_b32)
        - ``width=pl.MaskWidth.B16``: reads full 128-bit {MASK1, MASK0}, expands each bit to 2 bits (movp_b16)

        Kwargs:
            width: ``pl.MaskWidth.B32`` (default) or ``pl.MaskWidth.B16`` --- selects mask width

        Returns:
            mask_reg with current SPR value
        """

    @staticmethod
    @_api_decl
    def log2(src, preg, mode: Optional[MergeMode] = None,
             precision: Optional[bool] = None):
        r"""Base-2 logarithm of each element.

        For each lane ``i`` where ``mask[i]`` is active, computes the base-2
        logarithm of ``src[i]`` and writes the result to ``dst[i]``.
        Synthesized as ``vln(src) * (1/ln(2))``.

        .. math:: dstReg_i = \log_2(srcReg_i) = \frac{\ln(srcReg_i)}{\ln 2}

        Args:
            src: Source register (must be positive)
            preg: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.
            precision: When ``True``, enables high-precision mode that
                preserves subnormal output results (1-ulp precision error).
                Only effective for ``DT_FP32`` source type. Default ``False``
                (standard mode, subnormal outputs are flush-to-zero).

        Returns:
            Destination register (``RegTensor``) holding ``log2(src)``
            for each active lane.
        """

    @staticmethod
    @_api_decl
    def log10(src, preg, mode: Optional[MergeMode] = None,
              precision: Optional[bool] = None):
        r"""Base-10 logarithm of each element.

        For each lane ``i`` where ``mask[i]`` is active, computes the base-10
        logarithm of ``src[i]`` and writes the result to ``dst[i]``.
        Synthesized as ``vln(src) * (1/ln(10))``.

        .. math:: dstReg_i = \log_{10}(srcReg_i) = \frac{\ln(srcReg_i)}{\ln 10}

        Args:
            src: Source register (must be positive)
            preg: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.
            precision: When ``True``, enables high-precision mode that
                preserves subnormal output results (1-ulp precision error).
                Only effective for ``DT_FP32`` source type. Default ``False``
                (standard mode, subnormal outputs are flush-to-zero).

        Returns:
            Destination register (``RegTensor``) holding ``log10(src)``
            for each active lane.
        """

    @staticmethod
    @_api_decl
    def muls_cast(src, scalar, preg, dtype: DType,
                  layout: Optional[CastLayout] = None):
        r"""Multiply by scalar then cast.

        For each lane ``i`` where ``mask[i]`` is active, multiplies ``src[i]``
        by the scalar and casts the result to the destination data type.  Fused
        operation combining vmuls and vcvt.

        .. math:: dstReg_i = \text{cast}_{dtype}(srcReg_i \times scalar)

        Args:
            src: Source register (fp32)
            scalar: Scalar multiplier
            preg: Predicate mask register

        Kwargs:
            dtype: Destination data type (e.g. ``pl.DT_FP16``)
            layout: ``pl.CastLayout.ZERO`` (even half, default) or
                ``pl.CastLayout.ONE`` (odd half) for the half-width result

        Returns:
            Destination register (``RegTensor``) with the cast type.
        """

    @staticmethod
    @_api_decl
    def load(tile, stride=None):
        """Unified load (vldas+vldus, matches AscendC Load interface).

        Simple load from UB to register. Passing stride as 3rd positional
        arg enables post-update mode.

        Args:
            tile: Source UB tile
            stride: Post-update stride (optional positional arg)

        Returns:
            Destination register (``RegTensor``) holding the data loaded from
            the source UB tile.
        """

    @staticmethod
    @_api_decl
    def store(tile, src, count):
        """Unified store (vstus+vstas, matches AscendC Store interface).

        Simple store from register to UB.

        Args:
            dst_ptr: Destination UB pointer
            src: Source register
            count: Element count (optional, defaults to 256/elem_bytes)
        """

    @staticmethod
    @_api_decl
    def truncate(src, preg, mode: Optional[MergeMode] = None):
        """Truncate to integer (round toward zero): ``dst[i] = trunc(src[i])``

        Maps to hardware ``vtrc`` with ROUND_Z mode.

        Args:
            src: Source register
            preg: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default). MERGING mode is not supported on current device.

        Returns:
            Destination register (``RegTensor``) holding ``trunc(src[i])`` for
            each active lane.
        """

    @staticmethod
    @_api_decl
    def mask_gen_with_reg_tensor(src, offset: Optional[int] = None):
        """Generate mask from a register tensor bit at a given offset (movvp instruction).

        Converts a bit in a register element into a mask predicate.

        Args:
            src: Source register (uint16 or uint32)

        Kwargs:
            offset: Bit offset within the element

        Returns:
            mask_reg with generated mask
        """

    @staticmethod
    @_api_decl
    def create_addr_reg(stride0, stride1=None, stride2=None, stride3=None,
                        dtype: Optional[DType] = None):
        """Create an address offset register for aligned load/store (CreateAddrReg).

        Returns an ``AddrReg`` that can be passed to ``vf.load_align`` /
        ``vf.store_align`` as the offset parameter. Supports 1-4 loop axes
        (one stride per axis); the offset accumulates by each stride as the
        bound loop iterates.

        Usage::

            aReg = vf.create_addr_reg(64, dtype=pl.DT_FP32)
            reg = vf.load_align(src_tile, aReg)
            vf.store_align(dst_tile, reg, preg, aReg)

        Args:
            stride0: Loop axis 0 stride in elements
            stride1: Optional loop axis 1 stride in elements
            stride2: Optional loop axis 2 stride in elements
            stride3: Optional loop axis 3 stride in elements

        Kwargs:
            dtype: Data type for template instantiation (default ``pl.DT_FP32``).
                Determines the element width: b8/b16/b32/b64.

        Returns:
            AddrReg handle for use as offset in load_align/store_align
        """

    @staticmethod
    @_api_decl
    def move(src, preg, mode: Optional[MergeMode] = None):
        r"""Move/copy register elements (vmov for RegTensor, pmov for MaskReg).

        For RegTensor: copies valid elements from src to dst; masked-out
        positions retain dst's original value (MODE_MERGING).

        For MaskReg: copies src bits to dst; with mask, only masked bits
        are copied.

        .. math:: dstReg_i = srcReg_i

        Usage::

            # RegTensor with mask
            dst = vf.move(src_reg, preg)
            # RegTensor without mask
            dst = vf.move(src_reg)
            # MaskReg with mask
            dst = vf.move(src_mask, preg)
            # MaskReg without mask
            dst = vf.move(src_mask)

        Args:
            src: Source register (RegTensor or MaskReg)
            preg: Optional mask register

        Kwargs:
            mode: ``pl.MergeMode.MERGING`` (default, only supported mode).
                Supports b8/b16/b32/b64 element widths.

        Returns:
            Destination register (``RegTensor`` or ``MaskReg``, matching the
            source type).
        """
