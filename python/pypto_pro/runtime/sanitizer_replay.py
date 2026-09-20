#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host replay for the PyPTO Pro sanitizer.

Scans the single ``sanitizer_report.bin`` file (metadata header + device-side
log records) produced by a kernel launch and computes the detections:

    1. GM_OUT_OF_BOUNDS   - GM access window must fit each tensor dimension
                            (linear scalar access included).
    2. TILE_OUT_OF_BOUNDS - tile access window must fit the declared tile
                            dims (transfers, set_validshape, scalar access).
    3. TILE_OVERLAP       - no two tiles may overlap in address space.
    4. MUTEX_*            - mutex lock/unlock pairing per (region, id, pipe).
    5. GM (declaration)   - a ptr.make_tensor view must fit its source.

Record layouts are declared in ``_RECORD_FORMATS`` (the single source both
the decoder and the size/capacity math derive from); checks are registered
in ``_DETECTIONS``. Every check value travels in the records -- the pass
never decides, this file does.
"""

from __future__ import annotations

import dataclasses
import struct
from typing import Callable

# Record detection ids (must match C++ SanitizerDetection enum).
DET_GM_ACCESS = 1
DET_TILE_ACCESS = 2
DET_MUTEX_ACCESS = 3
DET_VIEW_SHAPE = 4
DET_TILE_SCALAR = 5
DET_TILE_DECL = 6

# Max logical dims per GM record (matches kSanitizerMaxTensorDims in the
# C++ pass).
_MAX_RECORD_DIMS = 5

# Pipe names, mirroring ir::PipeType (MTE1=0 ... ALL=7), for readable reports.
_PIPE_NAMES = ["MTE1", "MTE2", "MTE3", "M", "V", "S", "FIX", "ALL"]


def _pipe_name(pipe: int) -> str:
    if 0 <= pipe < len(_PIPE_NAMES):
        return _PIPE_NAMES[pipe]
    return f"PIPE({pipe})"

# Per-detection record formats. Each record is a sequence of named u32
# slots; "u" is an unsigned u32, "s" a signed one (offsets and runtime
# expressions can be negative only on a bug, but the sign bit must survive
# the round trip). One det owns one flat format -- the C++ side writes one
# u32 per field in this exact order (the sizes must match field-for-field).
_RECORD_FORMATS: dict[int, list[tuple[str, str]]] = {
    # det 1: a GM access. For ndim = 0 (linear access) shape[] is zero and
    # acc_row carries the whole-tensor element count.
    DET_GM_ACCESS: [
        ("det_id", "u"), ("ndim", "u"),
        ("off0", "s"), ("off1", "s"), ("off2", "s"), ("off3", "s"), ("off4", "s"),
        ("dim0", "s"), ("dim1", "s"), ("dim2", "s"), ("dim3", "s"), ("dim4", "s"),
        ("acc_row", "u"), ("acc_col", "u"), ("line", "u"),
    ],
    # det 2: one tile side of a transfer -- dims from the access-site type.
    DET_TILE_ACCESS: [
        ("det_id", "u"), ("dim_row", "u"), ("dim_col", "u"),
        ("off_row", "s"), ("off_col", "s"),
        ("acc_row", "u"), ("acc_col", "u"), ("line", "u"),
    ],
    # det 3: one mutex lock/unlock (per mutex id of a deduped call).
    DET_MUTEX_ACCESS: [
        ("det_id", "u"), ("mutex_id", "u"), ("pipe", "u"), ("is_lock", "u"), ("line", "u"),
    ],
    # det 4: view-over-source declaration check, all runtime values.
    DET_VIEW_SHAPE: [
        ("det_id", "u"),
        ("dim0", "s"), ("dim1", "s"), ("dim2", "s"), ("dim3", "s"), ("dim4", "s"),
        ("src_dim0", "s"), ("src_dim1", "s"), ("src_dim2", "s"), ("src_dim3", "s"), ("src_dim4", "s"),
        ("fp_bytes", "s"), ("src_bytes", "s"), ("line", "u"),
    ],
    # det 5: getval/setval on a tile -- linear index vs dim0 * dim1.
    DET_TILE_SCALAR: [
        ("det_id", "u"), ("dim_row", "u"), ("dim_col", "u"), ("offset", "s"), ("line", "u"),
    ],
    # det 6: a make_tile declaration for the overlap scan.
    DET_TILE_DECL: [
        ("det_id", "u"), ("addr", "s"), ("size", "s"), ("space", "u"), ("numel", "u"), ("line", "u"),
    ],
}

# Compiled struct per det ("I" for unsigned, "i" for signed).
_RECORD_STRUCTS: dict[int, struct.Struct] = {
    det: struct.Struct("<" + "".join("I" if kind == "u" else "i" for _, kind in fields))
    for det, fields in _RECORD_FORMATS.items()
}

# Maximum record size in bytes and u32 words; per-region capacity is a
# multiple of the latter. The GM record (15 u32) is always the largest.
_MAX_RECORD_SIZE = max(st.size for st in _RECORD_STRUCTS.values())
MAX_RECORD_U32 = _MAX_RECORD_SIZE // 4


@dataclasses.dataclass
class DetectionSpec:
    det_id: int | None          # None for detections without runtime records
    name: str                   # finding kind
    level: str                  # "ERROR" / "WARNING"
    check: Callable             # (records, ctx) -> [SanitizerFinding]


class SanitizerFinding:
    def __init__(self, kind: str, message: str, location: str = "<unknown>",
                 hint: str = "", level: str = "ERROR"):
        self.kind = kind
        self.message = message
        self.location = location
        self.hint = hint
        self.level = level
        self.hits = 1  # how many records raised this (kind, location)

    def __str__(self) -> str:
        loc = f" at {self.location}" if self.location else ""
        msg = f"{self.kind}: {self.message}{loc}"
        if self.hint:
            msg += f"\nHint: {self.hint}"
        return msg


class SanitizerReplayError(RuntimeError):
    """Aggregated sanitizer findings from one replay pass."""

    def __init__(self, findings: list[SanitizerFinding]):
        self.findings = findings
        lines = [f"\nPyPTO Sanitizer Replay: {len(findings)} issue(s) detected\n"]
        for i, f in enumerate(findings, 1):
            lines.append(f"  [{i}/{len(findings)}] {f}")
            lines.append("")
        self.detail = "\n".join(lines)
        super().__init__(f"PyPTO Sanitizer Replay: {len(findings)} issue(s) detected")


class ReplayContext:
    """State the checks need: the kernel's source file and the tensor names
    (report display). Every check value -- offsets, windows, shapes, tile
    declaration ranges, source lines -- travels in the records themselves."""

    def __init__(self, source_file: str, tensor_names: list | None = None):
        self.tensor_names = tensor_names or []
        self.source_file = source_file

    def location(self, line: int) -> str:
        return f"{self.source_file}:{line}" if line > 0 else "<unknown>"

    def tensor_name(self, idx: int) -> str:
        if 0 <= idx < len(self.tensor_names):
            return self.tensor_names[idx]
        return f"tensor#{idx}"


# ---------------------------------------------------------------------------
# Record decode
# ---------------------------------------------------------------------------

def _decode_record(raw: bytes, offset: int) -> dict | None:
    """Decode one record at *offset* (bytes) by its leading det_id.

    The field names come straight from _RECORD_FORMATS, so the dict keys
    always match the declared layout (no hand-copied unpack lists).
    """
    if offset + 4 > len(raw):
        return None
    det_id = struct.unpack_from("<I", raw, offset)[0]
    struct_ = _RECORD_STRUCTS.get(det_id)
    if struct_ is None or offset + struct_.size > len(raw):
        return None
    names = [name for name, _ in _RECORD_FORMATS[det_id]]
    return dict(zip(names, struct_.unpack_from(raw, offset)))


def decode_regions(region_chunks: list[bytes]) -> list[dict]:
    """Decode per-region byte chunks into a flat list of records.

    Each chunk is one sub-block region as laid out on the device:
    [ctr u32, records...]; ctr = number of u32 words written after the ctr
    word. Chunks are trimmed to the written part by the caller (lazy
    copy-back), so decoding never scans unwritten space. Every record is
    tagged with its region index (the executing sub-block), which is the
    per-block key dimension for pairing detections (e.g. mutex: different
    sub-blocks own independent mutex hardware slots).
    """
    records: list[dict] = []
    for region_idx, chunk in enumerate(region_chunks):
        if len(chunk) < 4:
            continue
        (ctr,) = struct.unpack_from("<I", chunk, 0)
        pos = 4
        end = 4 + ctr * 4
        while pos + 4 <= end:
            rec = _decode_record(chunk, pos)
            if rec is None:
                pos += _MAX_RECORD_SIZE
                continue
            rec["region"] = region_idx
            records.append(rec)
            pos += _RECORD_STRUCTS[rec["det_id"]].size
    return records


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _check_gm_bounds(records: list[dict], ctx: ReplayContext) -> list[SanitizerFinding]:
    """GM_OUT_OF_BOUNDS: per-dim offset + window must fit the tensor shape.

    Both the offsets and the shape travel in the record (runtime values: the
    shape fields are the access-site TensorType dims -- constants, or the
    dynamic-dim ABI scalars resolved at execution). The tile's 2-D valid
    window applies to the innermost two dimensions; the leading ndim-2
    dimensions use a window of 1 (offset only). ndim = 0 marks a linear
    access (getval/setval): off0 is an element index, acc_row the
    whole-tensor element count."""
    findings: list[SanitizerFinding] = []
    for rec in records:  # same-det records only (pre-grouped by the caller)
        ndim = rec["ndim"]
        off = [rec["off0"], rec["off1"], rec["off2"], rec["off3"], rec["off4"]]
        shape = [rec["dim0"], rec["dim1"], rec["dim2"], rec["dim3"], rec["dim4"]]
        acc_row, acc_col = rec["acc_row"], rec["acc_col"]
        if ndim == 0:
            # Linear scalar access: acc_row carries the element count.
            if acc_row <= 0:
                continue  # unknown (dynamic) bound
            if off[0] < 0 or off[0] + 1 > acc_row:
                findings.append(
                    SanitizerFinding(
                        "GM_OUT_OF_BOUNDS",
                        f"scalar access at linear offset {off[0]} exceeds "
                        f"{acc_row} elements",
                        ctx.location(rec["line"]),
                        hint="getval/setval offsets must stay inside the tensor element count",
                    )
                )
            continue
        # The offsets and shape are aligned (same trailing-dim layout). The
        # tile's 2-D window applies to the innermost two dimensions only;
        # leading dimensions are offset-only (window of 1).
        details = []
        for i in range(ndim):
            dim_size = shape[i]
            if dim_size <= 0:
                continue  # unknown (dynamic) dim
            dim_off = off[i]
            if i == ndim - 2:
                dim_win = acc_row
            elif i == ndim - 1:
                dim_win = acc_col
            else:
                dim_win = 1
            if dim_off < 0:
                details.append(f"dim{i}: negative offset {dim_off}")
            elif dim_off + dim_win > dim_size:
                over = dim_off + dim_win - dim_size
                if dim_win == 1:
                    # Leading dimension: offset-only access (no tile window).
                    details.append(f"dim{i}: offset {dim_off} ≥ {dim_size} (over by {over})")
                else:
                    details.append(
                        f"dim{i}: offset {dim_off} + valid_shape {dim_win} = {dim_off + dim_win} "
                        f"> {dim_size} (over by {over})"
                    )
        if not details:
            continue
        access = f"offsets {off[:ndim]} + valid_shape [{acc_row}, {acc_col}]"
        msg = (f"{access} exceed tensor shape "
               f"{tuple(shape[:ndim])} — {'; '.join(details)}")
        findings.append(
            SanitizerFinding(
                "GM_OUT_OF_BOUNDS",
                msg,
                ctx.location(rec["line"]),
                hint=(
                    "The tile valid_shape plus the loop offsets must stay inside "
                    "the tensor logical shape on every dimension"
                ),
            )
        )
    return findings


def _check_tile_bounds(records: list[dict], ctx: ReplayContext) -> list[SanitizerFinding]:
    """TILE_OUT_OF_BOUNDS: tile accesses at a runtime offset (block.move /
    block.insert operands, dynamic set_validshape windows) must keep
    offset + window within the declared dims.  The dims come from the record
    itself (the access-site tile type -- what the codegen instantiates the
    transfer with, i.e. the runtime truth of the access), not from the tile
    table; the table only names the tile in the report.  Whole-tile accesses
    (load/store/compute) with constant windows are bounded by the compile-time
    set_validshape check."""
    findings: list[SanitizerFinding] = []
    for rec in records:  # same-det records only (pre-grouped by the caller)
        dim0, dim1 = rec["dim_row"], rec["dim_col"]
        off_row, off_col = rec["off_row"], rec["off_col"]
        acc_row, acc_col = rec["acc_row"], rec["acc_col"]
        violations = []
        # Per-dimension logical bounds (0 = unknown, skipped).
        if dim0 > 0 and off_row + acc_row > dim0:
            over = off_row + acc_row - dim0
            violations.append(f"dim0: offset {off_row} + valid_shape {acc_row} = "
                              f"{off_row + acc_row} > {dim0} (over by {over})")
        if dim1 > 0 and off_col + acc_col > dim1:
            over = off_col + acc_col - dim1
            violations.append(f"dim1: offset {off_col} + valid_shape {acc_col} = "
                              f"{off_col + acc_col} > {dim1} (over by {over})")
        if not violations:
            continue
        findings.append(
            SanitizerFinding(
                "TILE_OUT_OF_BOUNDS",
                f"offsets [{off_row}, {off_col}] + valid_shape [{acc_row}, {acc_col}] "
                f"exceed tile shape [{dim0}, {dim1}] — " + "; ".join(violations),
                ctx.location(rec["line"]),
                hint="Keep the tile's valid shape within the declared TileType dims",
            )
        )
    return findings


def _check_mutex_pairing(records: list[dict], ctx: ReplayContext) -> list[SanitizerFinding]:
    """MUTEX pairing: every lock must be matched by an unlock on the same
    (region, mutex_id, pipe) key, in record order. Region distinguishes
    independent sub-block mutex hardware slots (prevents cross-block
    false matches). Pending is a list to support multiple outstanding
    locks on the same key (prevents overwrite-based false negatives)."""
    findings: list[SanitizerFinding] = []
    key_span_ids: dict[tuple, list[int]] = {}
    for rec in records:  # same-det records only (pre-grouped by the caller)
        key = (rec.get("region", 0), rec["mutex_id"], rec["pipe"])
        if rec["is_lock"]:
            key_span_ids.setdefault(key, []).append(rec["line"])
        else:
            spans = key_span_ids.get(key)
            if spans:
                spans.pop()
                if not spans:
                    del key_span_ids[key]
            else:
                findings.append(
                    SanitizerFinding(
                        "MUTEX_UNLOCK_BEFORE_LOCK",
                        f"mutex_unlock(mutex_id={rec['mutex_id']}, pipe={_pipe_name(rec['pipe'])}, "
                        f"region={rec.get('region', 0)}) without a preceding mutex_lock",
                        ctx.location(rec["line"]),
                        hint=(
                            "Every mutex_unlock must be paired with a mutex_lock on the "
                            "same (region, mutex_id, pipe)"
                        ),
                    )
                )
    for key, spans in key_span_ids.items():
        region, mutex_id, pipe = key
        count = len(spans)
        msg = f"mutex_lock(mutex_id={mutex_id}, pipe={_pipe_name(pipe)}, region={region})"
        if count > 1:
            msg += f" never unlocked ({count} outstanding locks)"
        else:
            msg += " never unlocked"
        findings.append(
            SanitizerFinding(
                "UNPAIRED_MUTEX_LOCK",
                msg,
                ctx.location(spans[-1]),
                hint=(
                    "Every mutex_lock must be followed by a mutex_unlock on the same "
                    "(region, mutex_id, pipe)"
                ),
            )
        )
    return findings


_MEMORY_SPACES = {0: "DDR", 1: "Vec", 2: "Mat", 3: "Left", 4: "Right", 5: "Scaling",
                  6: "Acc", 7: "Bias", 8: "ScaleLeft", 9: "ScaleRight"}


def _check_tile_overlap(records: list[dict], ctx: ReplayContext) -> list[SanitizerFinding]:
    """TILE_OVERLAP (declaration records): two make_tile declarations with
    intersecting address ranges in the same memory space. The declarations
    travel in the TileDecl records; identical re-registrations (the same
    statement copied into both programs of one kernel) share (addr, size,
    space, span) and collapse, while distinct statements with the same
    range keep both records -- exactly what the scan must report."""
    findings: list[SanitizerFinding] = []
    decls = {}
    for rec in records:  # same-det records only (pre-grouped by the caller)
        if rec["addr"] < 0 or rec["size"] <= 0:
            continue
        key = (rec["addr"], rec["size"], rec["space"], rec["line"])
        decls.setdefault(key, rec)
    items = list(decls.values())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            ta, tb = items[i], items[j]
            if ta["space"] != tb["space"]:
                continue
            a0, sz_a = ta["addr"], ta["size"]
            b0, sz_b = tb["addr"], tb["size"]
            if not (a0 + sz_a <= b0 or b0 + sz_b <= a0):  # ranges intersect
                sp = _MEMORY_SPACES.get(ta["space"], f"space{ta['space']}")
                findings.append(
                    SanitizerFinding(
                        "TILE_OVERLAP",
                        f"tile range [{sp} 0x{a0:x}, 0x{a0 + sz_a:x}) overlaps "
                        f"another tile [{sp} 0x{b0:x}, 0x{b0 + sz_b:x})",
                        ctx.location(ta["line"]),
                        hint="Two tiles must not share on-chip address space",
                    )
                )
    return findings


def _check_tile_scalar_bounds(records: list[dict], ctx: ReplayContext) -> list[SanitizerFinding]:
    """TILE_OUT_OF_BOUNDS (scalar): getval/setval on a Tile -- the linear
    element index must stay inside dim0 * dim1."""
    findings: list[SanitizerFinding] = []
    for rec in records:  # same-det records only (pre-grouped by the caller)
        dim_row, dim_col, offset = rec["dim_row"], rec["dim_col"], rec["offset"]
        numel = dim_row * dim_col
        if offset < 0 or offset + 1 > numel:
            findings.append(
                SanitizerFinding(
                    "TILE_OUT_OF_BOUNDS",
                    f"scalar tile access at linear offset {offset} "
                    f"exceeds {numel} elements (dims [{dim_row},{dim_col}])",
                    ctx.location(rec["line"]),
                    hint="getval/setval offsets must stay inside the tile element count",
                )
            )
    return findings


def _check_view_over_source(records: list[dict], ctx: ReplayContext) -> list[SanitizerFinding]:
    """GM_OUT_OF_BOUNDS (declaration-time): a ptr.make_tensor view whose byte
    footprint exceeds its source's byte capacity. All values travel in the
    ViewShape record (runtime expressions resolved at execution): the
    declared shape, the footprint sum((dim-1)*stride)+1 scaled by the view
    dtype width, and the source capacity (-1 = raw pointer: no known bound,
    skip). The view shares the source's storage, so an oversized declaration
    reads/writes past the source even for in-bounds accesses."""
    findings: list[SanitizerFinding] = []
    for rec in records:  # same-det records only (pre-grouped by the caller)
        fp_bytes, src_bytes = rec["fp_bytes"], rec["src_bytes"]
        if src_bytes < 0 or fp_bytes < 0:
            continue  # pointer source or unknown bound
        if fp_bytes > src_bytes:
            dims = [rec[f"dim{i}"] for i in range(_MAX_RECORD_DIMS)]
            while dims and dims[-1] == 0:  # trailing pad zeros
                dims.pop()
            src_dims = [rec[f"src_dim{i}"] for i in range(_MAX_RECORD_DIMS)]
            while src_dims and src_dims[-1] == 0:
                src_dims.pop()
            view_desc = f"make_tensor view {dims}" if dims else "make_tensor view"
            src_desc = f"source tensor {src_dims}" if src_dims else "source tensor"
            # The covered bytes ride along: under wide strides the shape
            # stays innocent while the coverage explains the overflow.
            msg = f"{view_desc} covers {fp_bytes} bytes, {src_desc} holds {src_bytes} bytes"
            findings.append(
                SanitizerFinding(
                    "GM_OUT_OF_BOUNDS",
                    msg,
                    ctx.location(rec["line"]),
                    hint="A view shares its source's storage; keep its shape/stride size within the source",
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Detection registry: the unified registration point. Adding a detection =
# appending one DetectionSpec here (plus the C++ instrumentation).
# ---------------------------------------------------------------------------

_DETECTIONS: list[DetectionSpec] = [
    DetectionSpec(DET_GM_ACCESS, "GM_OUT_OF_BOUNDS", "ERROR", _check_gm_bounds),
    DetectionSpec(DET_TILE_ACCESS, "TILE_OUT_OF_BOUNDS", "ERROR", _check_tile_bounds),
    DetectionSpec(DET_TILE_SCALAR, "TILE_OUT_OF_BOUNDS", "ERROR", _check_tile_scalar_bounds),
    DetectionSpec(DET_VIEW_SHAPE, "GM_OUT_OF_BOUNDS", "ERROR", _check_view_over_source),
    DetectionSpec(DET_MUTEX_ACCESS, "MUTEX", "ERROR", _check_mutex_pairing),
    DetectionSpec(DET_TILE_DECL, "TILE_OVERLAP", "WARNING", _check_tile_overlap),
]


def run_replay(records: list[dict], ctx: ReplayContext) -> list[SanitizerFinding]:
    """Run every registered check over the decoded records.

    The records are grouped by det_id once, so each check receives only its
    own records (no per-check filtering) and the pass over the record list
    is linear. The finding level comes from the registry -- the single
    source -- and repeats of the same defect (kind, location) collapse with
    a count, since every loop iteration touching the same statement raises
    the same finding.
    """
    buckets: dict[int, list[dict]] = {}
    for rec in records:
        buckets.setdefault(rec["det_id"], []).append(rec)
    by_key: dict[tuple, SanitizerFinding] = {}
    for spec in _DETECTIONS:
        for f in spec.check(buckets.get(spec.det_id, []), ctx):
            f.level = spec.level
            key = (f.kind, f.location)
            if key in by_key:
                by_key[key].hits += 1
            else:
                f.hits = 1
                by_key[key] = f
    return [
        f if f.hits == 1
        else SanitizerFinding(f.kind, f"{f.message}  (same defect hit {f.hits} times "
                                       f"at this location)", f.location, f.hint, f.level)
        for f in by_key.values()
    ]


# ---------------------------------------------------------------------------
# Report file (sanitizer_report.bin)
#
#   u32 region_count | per region: u32 chunk_bytes + chunk
#
# Each chunk is one device region trimmed to its written part
# ([ctr u32, records...]). An immediate write-then-scan round trip inside
# one launch call (built and parsed by the adjacent functions), not a
# persistence contract -- no magic or version guard. Every check value
# travels in the records.
# ---------------------------------------------------------------------------


def build_report_file(region_chunks: list[bytes], path: str) -> None:
    """Write the report file: per-region written chunks."""
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(region_chunks)))
        for chunk in region_chunks:
            f.write(struct.pack("<I", len(chunk)))
            f.write(chunk)


def scan_report_file(path: str, source_file: str) -> list[SanitizerFinding]:
    """Scan one report file and return the findings. The records carry source
    lines; *source_file* renders the report locations."""
    with open(path, "rb") as f:
        data = f.read()
    (region_count,) = struct.unpack_from("<I", data, 0)
    pos = 4
    region_chunks = []
    for _ in range(region_count):
        (chunk_len,) = struct.unpack_from("<I", data, pos)
        pos += 4
        region_chunks.append(data[pos:pos + chunk_len])
        pos += chunk_len
    records = decode_regions(region_chunks)
    return run_replay(records, ReplayContext(source_file))
