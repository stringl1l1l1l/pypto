#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
""" """

import enum
from functools import wraps
import logging
import re
import sys
from typing import Dict, List, Optional, Tuple, Union

from . import pypto_impl
from .error import FeError


class CompStage(enum.Enum):
    ALL_COMPLETE = 0
    TENSOR_GRAPH = 1
    TILE_GRAPH = 2
    EXECUTE_GRAPH = 3
    CODEGEN_INSTRUCTION = 4
    CODEGEN_BINARY = 5


_FUNC_HASH_ORDER_PATTERN = re.compile(r'^func\d+_\d+$')
_DEFAULT_KEY = 'DEFAULT'

# Maximum allowed scope_id (ooo_scope or atomic_scope). Combined with _ATOMIC_SCOPE_ENCODE_BASE below,
# max user encoding = 10000 * 10000 + 9999 = 100,009,999. C++ side VF_CLUSTER_ID_START
# = 200,000,000, ensuring no overlap between user and VF cluster encoding spaces.
_ATOMIC_SCOPE_MAX = 10000

# Encoding base for packing (scopeId, iter) into a single int stored in atomicScopeId_.
# C++ side VF_CLUSTER_ID_START = 200,000,000 starts well above max user encoding.
_ATOMIC_SCOPE_ENCODE_BASE = 10000


def _validate_and_encode_scope_id(scope_id: int, param_name: str) -> List[int]:
    """Validate and encode a scope ID for sg_set_ooo_scope / sg_set_atomic_scope.

    Returns the encoded value list to be stored in pass_options.
    Shared by top-level set_pass_options for both sg_set_ooo_scope and experimental['sg_set_atomic_scope'].
    """
    if scope_id == -1:
        return [-1]
    elif 1 <= scope_id <= _ATOMIC_SCOPE_MAX:
        from pypto._compile_state import CompileState

        encoded = scope_id * _ATOMIC_SCOPE_ENCODE_BASE + CompileState.get_atomic_scope_iter()
        return [encoded]
    else:
        raise ValueError(
            f"Invalid {param_name}: '{scope_id}'. Expected -1 or an integer in [1, {_ATOMIC_SCOPE_MAX}]."
        )


def _decode_scope_id(config_val: Optional[List[int]]) -> int:
    """Decode an encoded scope ID from config value back to user-facing scope_id."""
    if config_val and config_val[0] > 0:
        return config_val[0] // _ATOMIC_SCOPE_ENCODE_BASE
    elif config_val:
        return config_val[0]
    else:
        return 0

# cube_l1_reuse_setting matrix-side encoding. The (count, side) value is packed into the
# existing cube_l1_reuse_setting int value as side * _L1_REUSE_SIDE_BASE + count, so no extra
# config keys are needed; L1CopyInReuseMerge decodes it. "auto" packs to just count (fully
# backward compatible). _L1_REUSE_SIDE_BASE MUST match L1_REUSE_SIDE_BASE in function.h.
_MATRIX_SIDE_TO_CODE = {'auto': 0, 'left': 1, 'right': 2}
_MATRIX_SIDE_FROM_CODE = {1: 'left', 2: 'right'}
_L1_REUSE_SIDE_BASE = 1000000


def _encode_cube_l1_reuse_side(setting: dict) -> dict:
    """Encode a cube_l1_reuse_setting dict whose value may be an int merge count or a
    (count, side) tuple into a plain {key: int} dict, packing side into the value as
    side * _L1_REUSE_SIDE_BASE + count. Plain int values pass through unchanged.
    """
    encoded = {}
    for key, val in setting.items():
        if isinstance(val, (tuple, list)):
            if len(val) != 2:
                raise FeError(
                    ValueError(
                        f"cube_l1_reuse_setting[{key!r}] tuple must be (count, side), but got length {len(val)}."
                    )
                )
            count, side = val
            # count is the merge granularity (>=1; 1 means no merge). Reject <1 here so the
            # user gets a clear message instead of C++'s "Invalid merge count" after decoding.
            if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count < _L1_REUSE_SIDE_BASE:
                raise FeError(
                    ValueError(
                        f"cube_l1_reuse_setting[{key!r}] merge count must be an int in [1, {_L1_REUSE_SIDE_BASE}), "
                        f"but got {count!r}."
                    )
                )
            if not isinstance(side, str) or side.lower() not in _MATRIX_SIDE_TO_CODE:
                raise FeError(
                    ValueError(
                        f"cube_l1_reuse_setting[{key!r}] side must be one of "
                        f"{sorted(_MATRIX_SIDE_TO_CODE)}, but got {side!r}."
                    )
                )
            encoded[key] = _MATRIX_SIDE_TO_CODE[side.lower()] * _L1_REUSE_SIDE_BASE + count
        elif isinstance(val, int) and not isinstance(val, bool):
            encoded[key] = val
        else:
            raise FeError(
                TypeError(
                    f"cube_l1_reuse_setting[{key!r}] value must be int or (count, side) tuple, "
                    f"but got {type(val).__name__}."
                )
            )
    return encoded


def _decode_cube_l1_reuse_side(value: int):
    """Inverse of the packing: int -> int (count only) or (count, side) tuple."""
    if not isinstance(value, int) or value < _L1_REUSE_SIDE_BASE:
        return value
    side_code = value // _L1_REUSE_SIDE_BASE
    count = value % _L1_REUSE_SIDE_BASE
    side = _MATRIX_SIDE_FROM_CODE.get(side_code)
    if side is None:
        # Should not happen on the normal path (encode validates side). A stray code here
        # usually means _L1_REUSE_SIDE_BASE is out of sync with C++ L1_REUSE_SIDE_BASE.
        logging.warning(
            "cube_l1_reuse_setting: cannot decode matrix side from packed value %d "
            "(side code %d not in %s); returning merge count %d only.",
            value,
            side_code,
            sorted(_MATRIX_SIDE_FROM_CODE),
            count,
        )
        return count
    return (count, side)


def _validate_hash_order_setting(setting_dict: Optional[Dict[Union[int, str], int]], param_name: str):
    if setting_dict is None:
        return
    if not isinstance(setting_dict, dict):
        return

    has_int_keys = False
    has_func_keys = False

    for key in setting_dict.keys():
        if isinstance(key, int):
            has_int_keys = True
        elif isinstance(key, str):
            if _FUNC_HASH_ORDER_PATTERN.match(key) or key == _DEFAULT_KEY:
                has_func_keys = True

    if has_int_keys and has_func_keys:
        raise FeError(
            ValueError(
                f"{param_name} cannot mix integer keys with func/DEFAULT keys. "
                f"Please use either all integer keys or all func/DEFAULT keys."
            )
        )


def _validate_scope_id(scope_id: int):
    if scope_id < -1 or scope_id > 2147483647:
        raise FeError(
            ValueError(f"Option 'pass.sg_set_scope' scope_id {scope_id} is out of range. Expected -1~2147483647.")
        )


def _process_sg_set_scope(sg_set_scope: Union[int, tuple, list]) -> list:
    if isinstance(sg_set_scope, int):
        _validate_scope_id(sg_set_scope)
        return [sg_set_scope, False, False]
    if isinstance(sg_set_scope, (tuple, list)):
        if len(sg_set_scope) != 3:
            raise FeError(
                ValueError(f"Option 'pass.sg_set_scope' has invalid length. Expected 3, but got {len(sg_set_scope)}.")
            )
        if not isinstance(sg_set_scope[0], int):
            raise FeError(
                ValueError(
                    f"Option 'pass.sg_set_scope[0]' has invalid type. "
                    f"Expected int, but got {type(sg_set_scope[0]).__name__}."
                )
            )
        _validate_scope_id(sg_set_scope[0])
        if not isinstance(sg_set_scope[1], bool):
            raise FeError(
                ValueError(
                    f"Option 'pass.sg_set_scope[1]' has invalid type. "
                    f"Expected bool, but got {type(sg_set_scope[1]).__name__}."
                )
            )
        if not isinstance(sg_set_scope[2], bool):
            raise FeError(
                ValueError(
                    f"Option 'pass.sg_set_scope[2]' has invalid type. "
                    f"Expected bool, but got {type(sg_set_scope[2]).__name__}."
                )
            )
        return list(sg_set_scope)
    raise FeError(
        TypeError(
            f"Option 'pass.sg_set_scope' has invalid type. "
            f"Expected int64 or tuple, but got {type(sg_set_scope).__name__}."
        )
    )


def set_print_options(
    *,
    edgeitems: Optional[int] = 3,
    precision: Optional[int] = 4,
    threshold: Optional[int] = 10,
    linewidth: Optional[int] = 10,
) -> None:
    """
    Set tensor print options.

    Parameters
    ----------
    edge_items : int
        Print max items in tensor head and tail.

    precision : int
        Print precision.

    threshold : int
        Threshold to use.

    linewidth : int
        Max line width.
    """
    pypto_impl.SetPrintOptions(edgeitems, precision, threshold, linewidth)


def set_pass_options(
    *,
    vec_nbuffer_setting: Optional[Dict[Union[int, str], int]] = None,
    cube_l1_reuse_setting: Optional[Dict[Union[int, str], Union[int, Tuple[int, str]]]] = None,
    cube_nbuffer_setting: Optional[Dict[Union[int, str], int]] = None,
    sg_set_scope: Optional[Union[int, Tuple[int, bool, bool]]] = None,
    sg_set_ooo_scope: Optional[int] = None,
    experimental: Optional[Dict[str, int]] = None,
    ooo_sched_mode: Optional[str] = None,
    sg_set_tunevf_mode: Optional[int] = None,
    enable_slice: Optional[bool] = None,
) -> None:
    """
    Set pass options.

    Parameters
    ---------
    vec_nbuffer_setting : Dict[Union[int, str], int]
        Merged graph parameter, used to configure
        the merging quantity of AIV subgraphs with the same structure.

        Key format:
        - Integer key (e.g., 0, 1, -1): Global setting, applies to all functions
        - String key in format "func{magic}_{order}" (e.g., "func123_0"):
          Function-granularity setting, only applies to the specified function
          with functionMagic=magic and local hashOrder=order

    cube_l1_reuse_setting : Dict[Union[int, str], Union[int, Tuple[int, str]]]
        Merged graph parameter, used to configure
        the merging quantity of subgraphs with the same structure
        and repeated transfer of the same GM data.
        Supports same key formats as vec_nbuffer_setting.

        The value may be either an int merge count (current behavior) or a
        (count, side) tuple, where side is "left" / "right" / "auto":
        - "left"  bias the L1 reuse merge towards the left  matrix (consumer -> L0A)
        - "right" bias the L1 reuse merge towards the right matrix (consumer -> L0B)
        - "auto"  (default) keep the existing merge-axis selection
        The side bias is applied per matched subgraph; if no candidate exists on
        the chosen side, it falls back to the default selection. Example:
        cube_l1_reuse_setting={"DEFAULT": 4, "func8_0": (1, "left"), "MM1": (2, "right")}

    cube_nbuffer_setting : Dict[Union[int, str], int]
        Merged graph parameter, used to configure
        the merging quantity of AIC subgraphs with the same structure.
        Supports same key formats as vec_nbuffer_setting.

    sg_set_scope : Union[int, tuple]
        Merged graph parameter, used to manually control graph merging.
        - If int: only set scopeid (backward compatible)
        - If tuple: (scopeid, allow_parallel_merge, allow_cross_scope_merge)
          * scopeid: int, scope ID
          * allow_parallel_merge: bool, enable parallel branch merging
          * allow_cross_scope_merge: bool, allow supernode with scope to merge with others

    sg_set_tunevf_mode : int
        Control the VF (Vector Fusion) tuning pass behavior.
        - 0: default behavior (no change)
        - 1: bypass TuneTileOpSeqForVF and TuneSyncForVF passes
        - 2: ignore performance modeling and maximize fusion in TuneSyncForVF

    enable_slice : bool
        Whether to enable slice-related processing. Defaults to False.

    sg_set_ooo_scope : int
        Control OoO scheduling manual task grouping. Connected ops with the
        same ooo_scope will be merged into one task. Will be deprecated in a
        future release.

    experimental : Dict[str, int]
        Experimental pass options. Currently supports 'sg_set_atomic_scope':
        set indivisible atomic scope for subsequent operations, affecting VF
        fusion, OoO scheduling, on-chip buffer reuse, sync insertion, mix
        subgraph split, and loop grouping. This is an experimental feature
        that may change or be removed in future releases.

    Raises
    ------
    ValueError
        If mixing function-granularity keys ("func{magic}_{order}") with integer keys
        in the same setting parameter.
    """
    _validate_hash_order_setting(vec_nbuffer_setting, 'vec_nbuffer_setting')
    _validate_hash_order_setting(cube_l1_reuse_setting, 'cube_l1_reuse_setting')
    _validate_hash_order_setting(cube_nbuffer_setting, 'cube_nbuffer_setting')

    pass_options = {}
    if sg_set_scope is not None:
        pass_options['sg_set_scope'] = _process_sg_set_scope(sg_set_scope)
    if vec_nbuffer_setting is not None:
        pass_options['vec_nbuffer_setting'] = vec_nbuffer_setting
    if cube_l1_reuse_setting is not None:
        pass_options['cube_l1_reuse_setting'] = cube_l1_reuse_setting
    if cube_nbuffer_setting is not None:
        pass_options['cube_nbuffer_setting'] = cube_nbuffer_setting
    if sg_set_tunevf_mode is not None:
        if sg_set_tunevf_mode not in (0, 1, 2):
            raise ValueError(f"Invalid sg_set_tunevf_mode: '{sg_set_tunevf_mode}'. Expected 0, 1 or 2.")
        pass_options['sg_set_tunevf_mode'] = sg_set_tunevf_mode
    # Legacy sg_set_ooo_scope is converted to the sg_set_atomic_scope config key here,
    # so the framework only consumes sg_set_atomic_scope.
    if sg_set_ooo_scope is not None:
        if experimental is not None and 'sg_set_atomic_scope' in experimental:
            raise ValueError(
                "Cannot specify both 'sg_set_ooo_scope' and experimental['sg_set_atomic_scope']; "
                "set only one of them."
            )
        pass_options['sg_set_atomic_scope'] = _validate_and_encode_scope_id(sg_set_ooo_scope, 'sg_set_ooo_scope')
    if experimental is not None:
        if not isinstance(experimental, dict):
            raise ValueError(f"Invalid experimental: '{experimental}'. Expected dict.")
        for key, val in experimental.items():
            if key == 'sg_set_atomic_scope':
                pass_options['sg_set_atomic_scope'] = _validate_and_encode_scope_id(val, 'sg_set_atomic_scope')
            else:
                raise ValueError(
                    f"Invalid experimental key: '{key}'. Supported keys: 'sg_set_atomic_scope'."
                )
    if ooo_sched_mode is not None:
        if ooo_sched_mode not in ("", "GAPMIN", "HLF"):
            raise ValueError(f"Invalid ooo_sched_mode: '{ooo_sched_mode}'. Expected '', 'GAPMIN' or 'HLF'.")
        pass_options['ooo_sched_mode'] = ooo_sched_mode
    if enable_slice is not None:
        if not isinstance(enable_slice, bool):
            raise ValueError(f"Invalid enable_slice: '{enable_slice}'. Expected bool.")
        pass_options['enable_slice'] = enable_slice

    if pass_options:
        set_options(pass_options=pass_options)


def _merge_split_settings(rst: dict, base_key: str) -> dict:
    merged = {}
    if base_key in rst:
        merged.update(rst[base_key])
    by_func_key = base_key + '_by_func'
    if by_func_key in rst:
        merged.update(rst[by_func_key])
    by_label_key = base_key + '_by_label'
    if by_label_key in rst:
        merged.update(rst[by_label_key])
    return merged


def get_pass_options() -> Dict[str, Union[str, int, bool, List[int], Dict[int, int], Dict[str, int]]]:
    """
    Get pass options.

    Returns
    -------
    Dict[str, Union[str, int, List[int], Dict[int, int], Dict[str, int]]]
        All pass options from C++ scope, with _by_func/_by_label merged back
        into their parent setting keys (vec_nbuffer_setting, etc.).
    """
    scope = get_current_scope()
    rst = scope.get_pass_options()
    result = {}
    for base_key in ('vec_nbuffer_setting', 'cube_l1_reuse_setting', 'cube_nbuffer_setting'):
        result[base_key] = _merge_split_settings(rst, base_key)
    # Decode the packed (count, side) values of cube_l1_reuse_setting back to tuples so the
    # round-trip mirrors what the user passed in.
    result['cube_l1_reuse_setting'] = {
        key: _decode_cube_l1_reuse_side(val) for key, val in result['cube_l1_reuse_setting'].items()
    }
    val = rst.get("sg_set_scope", (-1, False, False))
    result['sg_set_scope'] = (int(val[0]), bool(val[1]), bool(val[2]))
    # sg_set_ooo_scope is stored as the sg_set_atomic_scope config key (converted at set time).
    scope_val = _decode_scope_id(rst.get('sg_set_atomic_scope', [0]))
    result['sg_set_ooo_scope'] = scope_val
    result['experimental'] = {'sg_set_atomic_scope': scope_val}
    result['ooo_sched_mode'] = rst.get('ooo_sched_mode', '')
    result['sg_set_tunevf_mode'] = rst.get('sg_set_tunevf_mode', 0)
    result['enable_slice'] = rst.get('enable_slice', False)
    return result


def set_host_options(
    *,
    compile_stage: Optional[CompStage] = None,
    compile_monitor_enable: Optional[int] = None,
    compile_timeout: Optional[int] = None,
    compile_timeout_stage: Optional[int] = None,
    compile_monitor_print_interval: Optional[int] = None,
) -> None:
    """
    Set host options.

    Parameters
    ---------
    compile_stage : CompStage
        Control the compilation phase.

    compile_monitor_enable : int
        Control compiler monitor mode (Watchdog). 0 disables monitor, 1 enables monitor without pass details,
        2 enables monitor with pass details. Default is 1 (enabled).

    compile_timeout : int
        Control the timeout duration for the entire compilation process.

    compile_timeout_stage : int
        Control the timeout duration of a certain stage of the compilation process.

    compile_monitor_print_interval : int
        Control the frequency of printing the compilation progress for a certain stage.
    """
    options_dict = {k: v.value if isinstance(v, CompStage) else v for k, v in locals().items() if v is not None}
    set_options(host_options=options_dict)


def get_host_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]:
    """
    Get host options.

    Returns
    -------
    Dict[str, Union[str, int, List[int], Dict[int, int]]]
        All host options
    """
    scope = get_current_scope()
    return scope.get_host_options()


def set_codegen_options(
    *,
    support_dynamic_aligned: Optional[bool] = None,
    soc_version: Optional[str] = None,
    enable_pmu_trace: Optional[bool] = None,
    vf_options: Optional[str] = None,
) -> None:
    """
    Set codegen options.

    Parameters
    ---------
    support_dynamic_aligned : bool
        Whether to support dynamic shape which is aligned.

    soc_version : str
        User specified soc_version for compile, codegen and runtime.

    enable_pmu_trace : bool
        Whether to enable PMU trace data collection.

    vf_options : str
        User specified vf_options for compile.
    """
    options_dict = {k: v for k, v in locals().items() if v is not None}
    set_options(codegen_options=options_dict)


def get_codegen_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]:
    """
    Get codegen options.

    Returns
    -------
    Dict[str, Union[str, int, List[int], Dict[int, int]]]
        All codegen options
    """
    scope = get_current_scope()
    return scope.get_codegen_options()


def set_verify_options(
    *,
    enable_pass_verify: Optional[bool] = None,
    pass_verify_save_tensor: Optional[bool] = None,
    pass_verify_save_tensor_dir: Optional[str] = None,
    pass_verify_pass_filter: Optional[List[str]] = None,
    pass_verify_error_tol: Optional[List[float]] = None,
) -> None:
    """
    Set verify options.

    Parameters
    ---------
    enable_pass_verify : bool
        Whether to verify pass.

    pass_verify_save_tensor : bool
        Whether to dump the tensor.

    pass_verify_save_tensor_dir : str
        Pass verify tensor save path.

    pass_verify_pass_filter : List
        Filting pass to verify.

    pass_verify_error_tol : List
        Customize atol and rtol.
    """
    if pass_verify_pass_filter == []:
        pass_verify_pass_filter = ["no_verify"]
    if pass_verify_error_tol is None or len(pass_verify_error_tol) != 2:
        pass_verify_error_tol = [1e-3, 1e-3]
    pass_verify_error_tol = [float(x) for x in pass_verify_error_tol]
    options_dict = {k: v for k, v in locals().items() if v is not None}
    set_options(verify_options=options_dict)


def get_verify_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]:
    """
    Get verify options.

    Returns
    -------
    Dict[str, Union[str, int, List[int], Dict[int, int]]]
        All verify options
    """
    scope = get_current_scope()
    return scope.get_verify_options()


def set_debug_options(
    *,
    compile_debug_mode: Optional[int] = None,
    runtime_debug_mode: Optional[int] = None,
    dump_pass_graph: Optional[List[str]] = None,
) -> None:
    """
    Set debug options.

    Parameters
    ---------
    compile_debug_mode : int
        Whether to enable debug mode during compilation stage.
        0: disabled;
        1: enabled, one-click to enable graph compilation related configs (e.g. compute graph);
        2: enabled, one-click to enable fixed CCE (fix device-side code output path,
           do not overwrite existing CCE files). Parallel compile threads are still
           controlled by codegen.parallel_compile; kernel names are deterministically
           assigned so multi-thread codegen outputs remain reproducible.

    runtime_debug_mode : int
        Whether to enable debug mode during execution stage.
        0: disabled;
        1: enabled, one-click to enable execution-related configs (e.g. swimlane graph);
        2: enable AICORE_MODEL simulation;
        3: enable runtime dependency-verification data dump;
        4: enable runtime GM memory out-of-bounds checks and emit modulo assertions
           at the control-flow entry for every explicitly registered, normalized
           `assume_divisible` expression; compiler-derived expressions that were
           not explicitly registered are not checked.

    dump_pass_graph : List[str]
        Pass identifiers to dump computation graph and print IR. Empty list disables it.
        Unmatched identifiers are ignored with a warning. If compile_debug_mode is 1,
        all passes dump graph and print IR regardless of dump_pass_graph.
    """
    options_dict = {k: v for k, v in locals().items() if v is not None}
    set_options(debug_options=options_dict)


def get_debug_options() -> Dict[str, Union[str, int, List[str], List[int], Dict[int, int]]]:
    """
    Get debug options.

    Returns
    -------
    Dict[str, Union[str, int, List[int], Dict[int, int]]]
        All verify options
    """
    scope = get_current_scope()
    return scope.get_debug_options()


def set_semantic_label(label: str) -> None:
    """
    Set the semantic label object.

    Parameters
    ---------
    label: str
        Semantic label.
        Note: label will be attached to subsequent operations

    Raises
    ------
    ValueError
        If label conflicts with function-granularity hashOrder format ('func{magic}_{order}').

    """
    # Function-granularity hashOrder pattern: func{magic}_{order}
    _func_hash_order_pattern = re.compile(r'^func\d+_\d+$')
    if _func_hash_order_pattern.match(label):
        raise FeError(
            ValueError(
                f"Semantic label '{label}' conflicts with function-granularity hashOrder format. "
                f"Please use a different label pattern."
            )
        )

    frame = sys._getframe(1)
    pypto_impl.SetSemanticLabel(label, frame.f_code.co_filename, frame.f_lineno)


def reset_options() -> None:
    """
    Reset all configuration items to their default values.
    """
    pypto_impl.ResetOptions()


class _Options:
    """Configuration options class, supports context manager and decorator modes"""

    INIT_FIELDS = [
        "name",
        "codegen_options",
        "host_options",
        "pass_options",
        "runtime_options",
        "verify_options",
        "debug_options",
        "vec_tile_shapes",
        "cube_tile_shapes",
        "conv_tile_shapes",
        "convbp_tile_shapes",
        "matrix_size",
        "operation_options",
    ]

    PREFIX_MAP = {
        "codegen_options": "codegen.",
        "host_options": "host.",
        "pass_options": "pass.",
        "runtime_options": "runtime.",
        "verify_options": "verify.",
        "debug_options": "debug.",
        "operation_options": "operation.",
    }

    def __init__(self, **kwargs):
        for field in self.INIT_FIELDS:
            setattr(self, field, kwargs.get(field, None))

    def prepare_options(self):
        """Convert configuration to target format"""
        opts = {}

        for attr, prefix in self.PREFIX_MAP.items():
            value = getattr(self, attr)
            if isinstance(value, dict):
                opts.update({f"{prefix}{k}": v.value if isinstance(v, enum.Enum) else v for k, v in value.items()})

        if self.vec_tile_shapes is not None:
            opts["vec_tile_shapes"] = self.vec_tile_shapes

        if self.cube_tile_shapes is not None:
            if isinstance(self.cube_tile_shapes, CubeTile):
                opts["cube_tile_shapes"] = self.cube_tile_shapes._impl
            else:
                opts["cube_tile_shapes"] = CubeTile(*self.cube_tile_shapes)._impl

        if self.conv_tile_shapes is not None:
            if isinstance(self.conv_tile_shapes, ConvTile):
                opts["conv_tile_shapes"] = self.conv_tile_shapes._impl
            else:
                opts["conv_tile_shapes"] = ConvTile(*self.conv_tile_shapes)._impl

        if self.convbp_tile_shapes is not None:
            if isinstance(self.convbp_tile_shapes, ConvBpTile):
                opts["convbp_tile_shapes"] = self.convbp_tile_shapes._impl
            else:
                opts["convbp_tile_shapes"] = ConvBpTile(*self.convbp_tile_shapes)._impl

        if self.matrix_size is not None:
            opts["matrix_size"] = self.matrix_size

        # Encode cube_l1_reuse_setting (count | (count, side)) into the existing key by
        # packing side into the int value. Done here so every entry path (jit / options
        # decorator / context manager / set_options / set_pass_options) is covered by the
        # single chokepoint. No extra config key is introduced; the pass decodes it.
        l1_key = "pass.cube_l1_reuse_setting"
        if l1_key in opts and isinstance(opts[l1_key], dict):
            opts[l1_key] = _encode_cube_l1_reuse_side(opts[l1_key])

        return opts

    def __enter__(self):
        """Context manager enter logic"""
        opts = self.prepare_options()
        frame = sys._getframe(1)
        # Use decorator position if available, otherwise use caller position
        filename = getattr(self, 'decorator_filename', frame.f_code.co_filename) or '<unknown>'
        lineno = getattr(self, 'decorator_lineno', frame.f_lineno) or 0

        pypto_impl.BeginScope(self.name, opts, filename, lineno)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit logic"""
        frame = sys._getframe(1)
        pypto_impl.EndScope(frame.f_code.co_filename, frame.f_lineno)

    def __call__(self, func):
        """Decorator mode logic: capture function definition location and wrap"""
        self.decorator_filename = func.__code__.co_filename
        self.decorator_lineno = func.__code__.co_firstlineno

        if not self.name:
            self.name = func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper


def options(
    name="",
    codegen_options=None,
    host_options=None,
    pass_options=None,
    runtime_options=None,
    verify_options=None,
    operation_options=None,
    debug_options=None,
    vec_tile_shapes=None,
    cube_tile_shapes=None,
    conv_tile_shapes=None,
    convbp_tile_shapes=None,
    matrix_size=None,
):
    """
    Create an Options instance. Can be used as decorator or context manager.

    Parameters
    ---------
    name: Scope name
    codegen_options: Code generation options (dict)
    host_options: Host options (dict)
    pass_options: Pass options (dict)
    runtime_options: Runtime options (dict)
    verify_options: Verify options (dict)
    debug_options: Debug options (dict)
    operation_options: Operation options (dict)
    vec_tile_shapes: Vector tile shapes (list)
    cube_tile_shapes: Cube tile shapes (CubeTile instance or list)
    matrix_size: Matrix size (list)

    Returns:
    -------
    Options instance

    Examples:
    -------
    # As decorator
    @pypto.options(pass_options={"cube_l1_reuse_setting": {-1: 4}})
    def func():
        pass

    # As context manager
    with pypto.options(name="test", cube_tile_shapes=[[16, 16], [256, 512, 128], [128, 128], True]):
        pass
    """
    # Automatically collect parameters and pass them with unpacking (eliminate duplicate parameter writing)
    return _Options(**locals())


def get_current_scope():
    """Get current config scope."""
    cpp_scope = pypto_impl.CurrentScope()
    return ConfigScope(cpp_scope)


def get_global_config(key: str):
    """Get global config config."""
    cpp_scope = pypto_impl.GlobalScope()
    py_scope = ConfigScope(cpp_scope)
    return py_scope.get_options_prefix("global." + key)


def set_global_config(key, value):
    """Set global config config."""
    pypto_impl.SetGlobalConfig({"global." + key: value})


def set_options(
    codegen_options=None,
    host_options=None,
    pass_options=None,
    runtime_options=None,
    verify_options=None,
    debug_options=None,
    operation_options=None,
    vec_tile_shapes=None,
    cube_tile_shapes=None,
    conv_tile_shapes=None,
    convbp_tile_shapes=None,
    matrix_size=None,
):
    """
    Finish the old scope and start a new scope.

    Parameters
    ---------
    codegen_options: Code generation options (dict)
    host_options: Host options (dict)
    pass_options: Pass options (dict)
    runtime_options: Runtime options (dict)
    verify_options: Verify options (dict)
    debug_options: Debug options (dict)
    operation_options: Operation options (dict)
    vec_tile_shapes: Vector tile shapes (list)
    cube_tile_shapes: Cube tile shapes (CubeTile instance or list)
    matrix_size: Matrix size (list)

    Examples:
    ---------
    set_options(pass_options={"cube_l1_reuse_setting": {-1: 4}})
    set_options(cube_tile_shapes=[[16, 16], [256, 512, 128], [128, 128], True])
    """
    temp_opts = options(**locals())
    opts = temp_opts.prepare_options()
    frame = sys._getframe(1)
    pypto_impl.SetScope(opts, frame.f_code.co_filename, frame.f_lineno)


def get_options_tree():
    """Get the tree structure string of configuration options"""
    return pypto_impl.GetOptionsTree()


class CubeTile:
    """CubeTile"""

    def __init__(self, m: List[int], k: List[int], n: List[int], enable_split_k: bool = False):
        """
        CubeTile tile for matmul operation, m[0], k[0], n[0] for L0 Cache, m[1], k[1], n[1] for L1 Cache

        Parameters
        ---------
        m: List[int]
        the value of the tile shape in m dimension.
        The length of the list must be 2.

        k: List[int]
            the value of the tile shape in k dimension
            The length of the list must be 2.

        n: List[int]
            the value of the tile shape in n dimension
            The length of the list must be 2.

        enable_split_k: bool
            whether the matmul result accumulated in the GM.
            default is false (i.e. not GM ACC)
        """

        if len(m) != 2:
            raise FeError(ValueError(f"m must have exactly 2 elements, got {len(m)}"))
        if len(n) != 2:
            raise FeError(ValueError(f"n must have exactly 2 elements, got {len(n)}"))
        if len(k) not in [2, 3]:
            raise FeError(ValueError(f"k must have 2 or 3 elements, got {len(k)}"))

        k_padded = list(k)
        if len(k_padded) == 2:
            k_padded.append(k_padded[1])  # k[2] = k[1]

        self._impl = pypto_impl.CubeTile(list(m), k_padded, list(n), enable_split_k)

    def __getattr__(self, name):
        return getattr(self._impl, name)

    def __repr__(self):
        return repr(self._impl)

    def __str__(self):
        return str(self._impl)

    def impl(self) -> pypto_impl.CubeTile:
        return self._impl


class ConvTile:
    """ConvTile"""

    def __init__(
        self, tile_l1_info: pypto_impl.TileL1Info, tile_l0_info: pypto_impl.TileL0Info, set_l0_tile: bool = False
    ):
        """
        ConvTile tile for convolution operation, tile_l1_info for L1 Cache, tile_l0_info for L0 Cache

        Parameters
        ---------
        tile_l1_info: pypto_impl.TileL1Info
            Tile configuration for L1 Cache (convolution dimensions):
            - tileHin: Input height tile size
            - tileHout: Output height tile size
            - tileWin: Input weight tile size
            - tileWout: Output weight tile size
            - tileCinFmap: Input channel tile size for feature map
            - tileCinWeight: Input channel tile size for weight
            - tileN: Output channel tile size
            - tileBatch: Batch dimension tile size
        tile_l0_info: pypto_impl.TileL0Info, optional
            Tile configuration for L0 Cache (H/W/K/N dimensions):
            - tileH: H dimension tile size
            - tileW: W dimension tile size
            - tileK: K dimension tile size
            - tileN: N dimension tile size
        set_l0_tile: bool, optional
            Flag to enable L0 Tile configuration, default False.
        """

        self._impl = pypto_impl.ConvTile(tile_l1_info, tile_l0_info, set_l0_tile)

    def __getattr__(self, name):
        attr_map = {
            'tile_l1_info': 'tileL1Info',
            'tile_l0_info': 'tileL0Info',
            'set_l0_tile': 'setL0Tile',
        }
        impl_name = attr_map.get(name, name)
        return getattr(self._impl, impl_name)

    def __repr__(self):
        return repr(self._impl)

    def __str__(self):
        return str(self._impl)

    def impl(self) -> pypto_impl.ConvTile:
        return self._impl


class ConvBpTile:
    """ConvBpTile"""
    def __init__(self, tile_l1_info: pypto_impl.ConvBpTileL1Info, tile_l0_info: pypto_impl.ConvBpTileL0Info):
        """
        ConvBpTile tile for convolution backward input operation, tile_l1_info for L1 Cache, tile_l0_info for L0 Cache

        Parameters
        ---------
        tile_l1_info: pypto_impl.ConvBpTileL1Info
            Tile configuration for L1 Cache (conv backward dimensions):
            - tileML1: M dimension tile size for L1 Cache
            - tileKL1: K dimension tile size for L1 Cache
            - tileNL1: N dimension tile size for L1 Cache
        tile_l0_info: pypto_impl.ConvBpTileL0Info
            Tile configuration for L0 Cache (M/K/N dimensions):
            - tileML0: M dimension tile size for L0 Cache
            - tileKL0: K dimension tile size for L0 Cache
            - tileNL0: N dimension tile size for L0 Cache
        """

        self._impl = pypto_impl.ConvBpTile(tile_l1_info, tile_l0_info)

    def __getattr__(self, name):
        attr_map = {
            'tile_l1_info': 'tileL1Info',
            'tile_l0_info': 'tileL0Info',
        }
        impl_name = attr_map.get(name, name)
        return getattr(self._impl, impl_name)

    def __repr__(self):
        return repr(self._impl)

    def __str__(self):
        return str(self._impl)

    def impl(self) -> pypto_impl.ConvBpTile:
        return self._impl


class ConfigScope:

    def __init__(self, cpp_config_scope=None):
        self._options = {}

        if cpp_config_scope is not None:
            self._options = cpp_config_scope.GetAllConfig()

    def __repr__(self):
        lines = []
        for key, value in sorted(self._options.items()):
            lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def get_options_prefix(self, key):
        if key not in self._options:
            raise FeError(KeyError(f"Option not found: {key}"))
        return self._options[key]

    def get_options(self, prefix):
        prefix = f"{prefix}."
        return {k[len(prefix):]: v for k, v in self._options.items() if k.startswith(prefix)}

    def get_pass_options(self):
        return self.get_options("pass")

    def get_codegen_options(self):
        return self.get_options("codegen")

    def get_host_options(self):
        return self.get_options("host")

    def get_debug_options(self):
        return self.get_options("debug")

    def get_verify_options(self):
        return self.get_options("verify")

    def get_operation_options(self):
        return self.get_options("operation")

    def get_runtime_options(self):
        return self.get_options("runtime")

    def get_vec_tile_shapes(self):
        return self._options.get("vec_tile_shapes")

    def get_cube_tile_shapes(self):
        return self._options.get("cube_tile_shapes")

    def get_conv_tile_shapes(self):
        return self._options.get("conv_tile_shapes")

    def get_convbp_input_tile_shapes(self):
        return self._options.get("convbp_tile_shapes")

    def get_matrix_size(self):
        return self._options.get("matrix_size")

    def get_all(self):
        return self._options.copy()

    def has(self, key):
        return key in self._options
