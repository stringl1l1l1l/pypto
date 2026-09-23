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

"""``pypto_compile_op`` — a drop-in replacement for the real ``asc_op_compiler.compile_op`` *leaf*.

The whole surrounding asc_opc flow is reused verbatim (op-store lookup, ``SingleOpCompile`` with its
``build_config`` + ``op_context``, ``SingleOpPostCompile`` writing ``supportInfo`` / json post-process).
The generated per-op wrapper (``ascendc_impl_build.py``) builds the exact same ``options`` / ``OpInfo`` as
the template wrapper and, instead of ``compile_op(src_cpp, ...)``, calls **this** with the PyPTO DSL ``.py``
as ``cce_file``.

The *only* PyPTO-specific difference is the kernel source: the template kernel is one
``template<TEMPLATE_PARAMS> .cpp`` compiled N times; PyPTO folds each tilingkey to a constant and codegens
a distinct concrete ``kernel.cpp`` per key. So this function does per-tilingkey codegen and compiles each
concrete key, while **reusing the real ``asc_op_compile_base`` backend leaves** for everything else:

- ``compile_pre_process`` / ``_add_op_compile_options_by_customized_json`` / ``_update_compile_option``
  (+ ``CompileOptionTuple``) — identical option preprocessing + ``global_var_storage`` / bisheng-path bootstrap;
- ``gen_compile_cmd_v220`` — the bisheng compile command (replaces the hand-ported flag list);
- ``get_ktype_section_variable`` — the ``FunLevelMixCoreType`` meta section;
- ``fatbin_objs`` — the fat ``.o`` link;
- ``CommonUtility.get_kernel_meta_dir`` / ``get_distinct_filename_tag`` — the real flat ``kernel_meta`` layout.

Artifacts land flat in ``kernel_meta`` exactly like the real compiler: one per-key object for every
physical core required by the inferred kernel type, one fat ``<kernel>.o`` and ``<kernel>.json``. Per-key
codegen output (``kernel.cpp`` + the per-key wrapper ``.cpp``) is kept on disk for debugging (not cleaned).
The json path is recorded via ``op_context.add_build_res("json_file_path", ...)`` so the reused
``SingleOpCompile`` picks it up and ``SingleOpPostCompile`` appends ``supportInfo``.
"""

from __future__ import annotations

import copy
import inspect
import os
from pathlib import Path
import shutil
import time

from asc_op_compile_base.asc_op_compiler.ascendc_common_utility import CommonUtility, CompileInfo
from asc_op_compile_base.asc_op_compiler.ascendc_compile_base import (
    compile_multi_tilingkey,
    compile_pre_process,
    fatbin_objs,
    link_relocatable,
)
from asc_op_compile_base.asc_op_compiler.ascendc_compile_dfx import (
    DFXArgInfo,
    DFXParamType,
    DFXPointType,
    DFXSectionGenerator,
)
from asc_op_compile_base.asc_op_compiler.ascendc_compile_gen_json import (
    _generate_final_json,
)
from asc_op_compile_base.asc_op_compiler.ascendc_compile_utils import check_if_gen_placehoder
from asc_op_compile_base.asc_op_compiler.ascendc_compile_v220 import (
    gen_compile_cmd_v220,
    get_ktype_section_variable,
)
from asc_op_compile_base.asc_op_compiler.ascendc_constants import (
    CORE_TYPE_CUBE,
    CORE_TYPE_MIX,
    CORE_TYPE_VEC,
    MIX_CORE_MACRO,
    TILING_KEY_MACRO,
    CompileOptionTuple,
    KernelMetaType,
)
from asc_op_compile_base.asc_op_compiler.compile_op import (
    _add_op_compile_options_by_customized_json,
    _json_post_process,
    _update_compile_option,
    handle_compile_options,
    handle_sk_codegen_options,
)
from asc_op_compile_base.asc_op_compiler.get_op_tiling import TilingInfo, get_tiling_info_by_tiling
from asc_op_compile_base.asc_op_compiler.global_storage import global_var_storage
from asc_op_compile_base.asc_op_compiler.kernel_info_infer import KernelInfoInfer
from asc_op_compile_base.common.context import op_context

# --- reused asc_op_compile_base backend leaves -------------------------------------------------------
from asc_op_compile_base.common.utils import log as logger

from pypto_pro import DataType
from pypto_pro.runtime.compile_config import get_jit_compile_config

from ..._errors import (
    InvalidArgument,
    InvalidOperation,
    InvalidType,
    InvalidVal,
    OutOfRange,
    RuntimeFailure,
    message_of,
)

# (AscendC core channel, object suffix, compile-make suffix)
_CORE_COMPILE_TARGETS = {
    "cube": (CORE_TYPE_CUBE, "mix_aic", "aic"),
    "vec": (CORE_TYPE_VEC, "mix_aiv", "aiv"),
}
_KERNEL_TYPE_BY_CORE_RATIO = {
    (1, 0): KernelMetaType.KERNEL_TYPE_AIC_ONLY,
    (0, 1): KernelMetaType.KERNEL_TYPE_AIV_ONLY,
    (1, 1): KernelMetaType.KERNEL_TYPE_MIX_AIC_1_1,
    (1, 2): KernelMetaType.KERNEL_TYPE_MIX_AIC_1_2,
}
_ORIG_DTYPE_TO_PYPTO = {
    "dt_bool": DataType.BOOL,
    "dt_int8": DataType.INT8,
    "dt_int16": DataType.INT16,
    "dt_int32": DataType.INT32,
    "dt_int64": DataType.INT64,
    "dt_uint8": DataType.UINT8,
    "dt_uint16": DataType.UINT16,
    "dt_uint32": DataType.UINT32,
    "dt_uint64": DataType.UINT64,
    "dt_float16": DataType.FP16,
    "dt_float": DataType.FP32,
    "dt_bf16": DataType.BF16,
    "dt_float8_e4m3fn": DataType.FP8E4M3FN,
    "dt_float8_e5m2": DataType.FP8E5M2,
    "dt_float8_e8m0": DataType.FP8E8M0,
    "dt_fp4_e2m1": DataType.FP4E2M1,
    "dt_fp4_e1m2": DataType.FP4E1M2,
    "dt_hifloat8": DataType.HF8,
}

_ORIG_DTYPE_MACRO_PREFIX = "-DORIG_DTYPE_"


def _load_kernel(op_path: str):
    """Import a PyPTO kernel module that defines exactly one ``_TileJitKernel``."""
    import importlib.util
    import sys

    from pypto_pro.runtime.jit import _TileJitKernel

    spec = importlib.util.spec_from_file_location("_pypto_opc_kernel_mod", op_path)
    if spec is None or spec.loader is None:
        raise RuntimeFailure(f"cannot load kernel module from {op_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    kernels = [v for v in vars(mod).values() if isinstance(v, _TileJitKernel)]
    if len(kernels) != 1:
        raise InvalidArgument(f"kernel module '{op_path}' must define exactly one @pl.jit kernel, found {len(kernels)}")
    return kernels[0]


def _write_tilingkey_header(schema, kernel_name: str, output_dir: str) -> None:
    """Generate ``<TilingKey>_tilingkey.h`` for binary delivery."""
    t_start = time.perf_counter()
    fields = schema._fields
    valid_combos = schema.enumerate_valid()

    header = '#include "ascendc/host_api/tiling/template_argument.h"\n\n'

    for bw in sorted({f.bits for f in fields}):
        header += f"#define ASCENDC_TPL_{bw}_BW {bw}\n"
    header += "\n"

    header += f"ASCENDC_TPL_ARGS_DECL({kernel_name},\n"
    for field in fields:
        values_str = ", ".join(str(v) for v in field.values)
        comment = f"// bit:{field.offset + field.bits - 1}-{field.offset}"
        header += f"    {comment}\n"
        header += (
            f"    ASCENDC_TPL_UINT_DECL({field.name}, ASCENDC_TPL_{field.bits}_BW, "
            f"ASCENDC_TPL_UI_LIST, {values_str}),\n"
        )
    header += ");\n\n"

    header += "ASCENDC_TPL_SEL(\n"
    for i, combo in enumerate(valid_combos):
        comma = "," if i < len(valid_combos) - 1 else ""
        header += "    ASCENDC_TPL_ARGS_SEL(\n"
        for j, field in enumerate(fields):
            jcomma = "," if j < len(fields) - 1 else ""
            header += f"        ASCENDC_TPL_UINT_SEL({field.name}, ASCENDC_TPL_UI_LIST, {combo[j]}){jcomma}\n"
        header += f"    ){comma}\n"
    header += ");\n"

    tilingkey_path = Path(output_dir) / f"{schema.cls_name}_tilingkey.h"
    tilingkey_path.write_text(header, encoding="utf-8")

    total_combos = 1
    for field in fields:
        total_combos *= len(field.values)
    logger.info(
        "tilingkey header '%s': %d valid / %d total combos | %.3fs",
        tilingkey_path.name,
        len(valid_combos),
        total_combos,
        time.perf_counter() - t_start,
    )


def _write_tilingdata_header(kernel, output_dir: str) -> None:
    """Generate the TilingData header without running kernel codegen."""
    from pypto.pypto_impl.codegen import CCECodegen
    from pypto_pro.language.typing._tiling import get_tiling_fields, get_tiling_tuple_type, is_tiling_class
    from pypto_pro.runtime.shape_policy import _get_annotations

    func = kernel._func
    namespace = dict(getattr(func, "__globals__", {}))
    namespace.update(kernel._closure_vars or {})
    try:
        annotations = _get_annotations(func, namespace)
    except (NameError, TypeError, ValueError) as exc:
        raise RuntimeFailure(
            f"Failed to evaluate annotations for kernel '{kernel.__name__}': {message_of(exc)}"
        ) from exc
    parameter_names = list(inspect.signature(func).parameters)
    tiling_params = [
        (name, annotations[name]) for name in parameter_names if is_tiling_class(annotations.get(name))
    ]
    if len(tiling_params) != 1:
        raise InvalidArgument(
            f"kernel '{kernel.__name__}' must have exactly one tiling-class parameter, found {len(tiling_params)}"
        )
    tiling_param_name, tiling_cls = tiling_params[0]
    if not parameter_names or parameter_names[-1] != tiling_param_name:
        raise InvalidOperation(f"tiling parameter '{tiling_param_name}' must be the last kernel parameter")

    fields = get_tiling_fields(tiling_cls)
    tiling_type = get_tiling_tuple_type(tiling_cls)
    header = CCECodegen.generate_tiling_header(tiling_cls.__name__, list(fields), list(tiling_type.types))
    Path(output_dir, f"{tiling_cls.__name__}_tiling.h").write_text(header, encoding="utf-8")


def _signature_parts(entry_params):
    decls, names = [], []
    for typ, name, is_ptr in entry_params:
        decls.append(f"__gm__ {typ}* {name}" if is_ptr else f"{typ} {name}")
        names.append(name)
    return ", ".join(decls), names


def _kernel_type_from_target(target) -> KernelMetaType:
    kernel_type = _KERNEL_TYPE_BY_CORE_RATIO.get((target.aic_per_block, target.aiv_per_block))
    if kernel_type is None:
        raise InvalidVal(
            "PyPTO OPC does not support the resolved core ratio "
            f"{target.aic_per_block}:{target.aiv_per_block}"
        )
    return kernel_type


def _kernel_type_from_codegen(cg, arch: str) -> KernelMetaType:
    target = get_jit_compile_config().resolve_kernel_target(
        arch,
        has_cube=cg.has_cube,
        has_vector=cg.has_vector,
    )
    return _kernel_type_from_target(target)


def _compile_core_names(kernel_type: KernelMetaType) -> tuple[str, ...]:
    if kernel_type == KernelMetaType.KERNEL_TYPE_AIC_ONLY:
        return ("cube",)
    if kernel_type == KernelMetaType.KERNEL_TYPE_AIV_ONLY:
        return ("vec",)
    if kernel_type in {
        KernelMetaType.KERNEL_TYPE_MIX_AIC_1_1,
        KernelMetaType.KERNEL_TYPE_MIX_AIC_1_2,
    }:
        return ("cube", "vec")
    raise InvalidVal(f"PyPTO OPC does not support kernel type {kernel_type.name}")


def _compile_target_name(kernel_name: str, tiling_key: str, core_name: str, kernel_type: KernelMetaType) -> str:
    target_name = f"{kernel_name}_{tiling_key}"
    if kernel_type in {
        KernelMetaType.KERNEL_TYPE_MIX_AIC_1_1,
        KernelMetaType.KERNEL_TYPE_MIX_AIC_1_2,
    }:
        target_name += "_mix_aic" if core_name == "cube" else "_mix_aiv"
    return target_name


def _compile_target_options(kernel_type: KernelMetaType, enable_c310_rules: bool) -> list[str]:
    options = []
    if kernel_type in {
        KernelMetaType.KERNEL_TYPE_MIX_AIC_1_1,
        KernelMetaType.KERNEL_TYPE_MIX_AIC_1_2,
    }:
        options.append(f"-D{MIX_CORE_MACRO}=1")
    if kernel_type == KernelMetaType.KERNEL_TYPE_MIX_AIC_1_1:
        options.append("-D__MIX_CORE_AIC_RATION__=1")
    if enable_c310_rules and kernel_type == KernelMetaType.KERNEL_TYPE_MIX_AIC_1_2:
        options.append("-D__ASCENDC_ENABLE_VEC_TAIL_TILING_COPY__")
    if enable_c310_rules and kernel_type == KernelMetaType.KERNEL_TYPE_AIC_ONLY:
        options.append("-DRAW_AIC_ONLY_DUMP_TENSOR")
    return options


def _gen_infer_cpp(cg, tilingkey_header: str, kernel_cpp: str, kernel_type: KernelMetaType) -> str:
    sig, names = _signature_parts(cg.entry_params)
    ws_idx = len(names) - 2 if len(names) >= 2 else None
    inner = list(names)
    ws_lines = ""
    if ws_idx is not None:
        ws = names[ws_idx]
        ws_lines = (
            f"    AscendC::SetSysWorkspaceForce({ws});\n    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace({ws});\n"
        )
        inner[ws_idx] = "usrWorkspace"
    return (
        '#include "kernel_operator.h"\n'
        f'#include "{tilingkey_header}"\n'
        f"{kernel_cpp}\n"
        f'extern "C" __global__ AICORE void {cg.kernel_name}({sig})\n'
        "{\n"
        f"    KERNEL_TASK_TYPE_DEFAULT({kernel_type.name});\n"
        f"{ws_lines}"
        f"    {cg.kernel_name}_impl({', '.join(inner)});\n"
        "}\n"
    )


def _prepare_infer_cpp(
    kernel,
    schema,
    dtype_consts,
    arch,
    cce_file: str,
    kernel_name: str,
    kernel_meta_dir: str,
) -> tuple[str, KernelMetaType]:
    """Generate the infer source from one default legal TilingKey and prepare its headers."""
    from pypto_pro.runtime.jit import _codegen

    valid_combos = schema.enumerate_valid()
    if not valid_combos:
        raise InvalidVal(f"tiling_key schema '{schema.cls_name}' has no valid tilingkey combination")
    concrete_key = dict(zip(schema.field_names(), valid_combos[0]))
    default_tiling_key = schema.pack(concrete_key)
    infer_cg = _codegen(
        kernel.to_kernel_def(concrete_key, dtype_consts),
        arch,
        clean_up=False,
        tilingkey_packed=default_tiling_key,
        out_dir=os.path.join(kernel_meta_dir, "_cg_infer"),
    )
    if infer_cg is None:
        raise RuntimeFailure(f"pypto codegen failed for default tilingkey {default_tiling_key}")
    kernel_type = _kernel_type_from_codegen(infer_cg, arch)
    for header in (name for name in os.listdir(infer_cg.build_dir) if name.endswith(".h")):
        shutil.copyfile(os.path.join(infer_cg.build_dir, header), os.path.join(kernel_meta_dir, header))
    tilingkey_header = f"{schema.cls_name}_tilingkey.h"
    shutil.copyfile(
        os.path.join(os.path.dirname(cce_file), tilingkey_header),
        os.path.join(kernel_meta_dir, tilingkey_header),
    )
    kernel_cpp = Path(infer_cg.build_dir, "kernel.cpp").read_text(encoding="utf-8")
    infer_cpp_path = os.path.join(kernel_meta_dir, f"{kernel_name}_pypto_infer.cpp")
    _write(infer_cpp_path, _gen_infer_cpp(infer_cg, tilingkey_header, kernel_cpp, kernel_type))
    return infer_cpp_path, kernel_type


def generate_binary_headers(kernel, arch="a5") -> str:
    """Generate the TilingData and TilingKey headers required by binary delivery."""
    from pypto_pro.runtime.jit import (
        _artifact_prefix_from_filename,
        _make_artifact_build_dir,
        _setup_arch_env,
        _TileJitKernel,
    )

    if not isinstance(kernel, _TileJitKernel):
        raise InvalidType("generate_binary_headers() expects a @pl.jit kernel")
    if kernel.tilingkey_schema is None:
        raise InvalidVal(
            f"generate_binary_headers() requires a tiling_key schema, but kernel "
            f"'{kernel.__name__}' has none; binary delivery needs a tilingkey header"
        )

    arch = _setup_arch_env(arch)
    schema = kernel.tilingkey_schema
    valid_combos = schema.enumerate_valid()
    if not valid_combos:
        raise InvalidVal(f"tiling_key schema '{schema.cls_name}' has no valid tilingkey combination")

    kernel_def = kernel.to_kernel_def()
    build_dir = _make_artifact_build_dir(
        kernel_def,
        arch,
        test_prefix=_artifact_prefix_from_filename(kernel_def._source_file),
    )
    binary_dir = os.path.join(build_dir, "binary")
    Path(binary_dir).mkdir(parents=True, exist_ok=True)
    _write_tilingdata_header(kernel, binary_dir)
    _write_tilingkey_header(schema, kernel.__name__, output_dir=binary_dir)
    return binary_dir


def prepare_binary_headers(op_path: str, arch="a5") -> str:
    """Load the sole ``@pl.jit`` kernel in ``op_path`` and prepare its binary-delivery headers."""
    return generate_binary_headers(_load_kernel(op_path), arch)


def _op_info_get(op_info, name, default=None):
    """OpInfo is an object (attrs) in the real flow; tolerate a dict too."""
    if isinstance(op_info, dict):
        return op_info.get(name, default)
    return getattr(op_info, name, default)


def _orig_dtype_to_pypto(dtype_name: str, param_name: str) -> DataType:
    if not isinstance(dtype_name, str):
        raise InvalidType(f"param '{param_name}' dtype must be a string, got {type(dtype_name).__name__}")
    key = dtype_name.lower()
    dtype = _ORIG_DTYPE_TO_PYPTO.get(key)
    if dtype is None:
        raise InvalidType(f"param '{param_name}' dtype '{dtype_name}' is not supported by PyPTO datatype")
    return dtype


def _extract_datatype_key_from_compile_options(compile_options, datatype_schema) -> dict | None:
    """Build the launch dtype dict from AscendC wrapper's -DORIG_DTYPE_<PARAM>=<dtype> options."""
    if datatype_schema is None:
        return None
    wanted = set(datatype_schema)
    by_name = {}
    for option in compile_options or []:
        if not isinstance(option, str) or not option.startswith(_ORIG_DTYPE_MACRO_PREFIX):
            continue
        macro = option[len(_ORIG_DTYPE_MACRO_PREFIX):]
        if "=" not in macro:
            continue
        param_macro, dtype_name = macro.split("=", 1)
        param_name = param_macro.lower()
        if param_name in wanted and param_name not in by_name:
            by_name[param_name] = _orig_dtype_to_pypto(dtype_name, param_name)
    missing = sorted(wanted - set(by_name))
    if missing:
        raise InvalidVal(f"compile_options is missing ORIG_DTYPE macros for datatype params: {missing}")
    return by_name


def _setup_options(op_info, compile_options, op_compile_option, extend_options):
    """Mirror ``compile_op`` opening (compile_op.py:1378-1397): build the CompileOptionTuple and run the
    real option preprocessing. ``compile_pre_process`` also bootstraps the bisheng path + ``global_var_storage``
    (it internally calls ``CommonUtility.get_ascendc_compiler_path``). Runs inside the reused ``build_config``."""
    global_var_storage.global_storage_reset()
    opt = CompileOptionTuple([] if compile_options is None else list(compile_options), [])
    impl_mode = _op_info_get(op_info, "impl_mode")
    if CommonUtility.is_c310() and isinstance(impl_mode, str) and impl_mode != "":
        impl_mode_def = f"-D{impl_mode.upper()}_"
        if impl_mode_def not in opt.compile_options:
            opt.compile_options.append(impl_mode_def)
    _add_op_compile_options_by_customized_json(op_compile_option, opt)
    opt.compile_options = compile_pre_process(op_info, opt.compile_options)
    _update_compile_option(_op_info_get(op_info, "kernel_name"), opt.compile_options, extend_options)
    opt.compile_options.append("-DASCENDC_TPL_KERNEL")
    pypto_include = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "include")
    opt.compile_options.append(f"-I{pypto_include}")
    return opt


def _core_arch(core_type: int) -> str:
    chip_version = CommonUtility.get_chip_version()
    suffix = "cube" if core_type == CORE_TYPE_CUBE else "vec"
    return f"dav-{chip_version}-{suffix}"


def _normalize_dfx_op_info(op_info):
    if _op_info_get(op_info, "mc2_ctx") is not None:
        return op_info
    if hasattr(op_info, "_replace"):
        return op_info._replace(mc2_ctx=[])
    if isinstance(op_info, dict):
        copied = dict(op_info)
        copied["mc2_ctx"] = []
        return copied
    return op_info


def _prepare_dfx(op_info, tiling_info: TilingInfo, compile_info: CompileInfo) -> object:
    dfx_op_info = _normalize_dfx_op_info(op_info)
    dfx_generator = DFXSectionGenerator()
    dfx_generator.dfx_info_reset(dfx_op_info)
    if not dfx_generator.is_support:
        return dfx_op_info

    inputs = _op_info_get(dfx_op_info, "inputs") or []
    outputs = _op_info_get(dfx_op_info, "outputs") or []
    mc2_ctx = _op_info_get(dfx_op_info, "mc2_ctx") or []
    needs_ffts = compile_info.code_channel == CORE_TYPE_MIX and not CommonUtility.is_c310()
    needs_ffts = needs_ffts and not CommonUtility.is_m510()
    if needs_ffts:
        dfx_generator.insert_param(DFXArgInfo("ffts", DFXParamType.FFTS))
    for ctx_name in mc2_ctx:
        dfx_generator.insert_param(DFXArgInfo(ctx_name, DFXParamType.MC2CTX))
    for input_info in inputs:
        if input_info is not None:
            dfx_generator.insert_param(DFXArgInfo(input_info["param_name"], DFXParamType.INPUT))
    for output_info in outputs:
        if output_info is not None:
            dfx_generator.insert_param(DFXArgInfo(output_info["param_name"], DFXParamType.OUTPUT))
    output_shape_depend = _op_info_get(dfx_op_info, "output_shape_depend_on_compute") or []
    if len(output_shape_depend) > 0:
        dfx_generator.insert_param(DFXArgInfo("shape_tensor", DFXParamType.SHAPE_TENSOR))
        for index in output_shape_depend:
            parameter = dfx_generator.get_param(outputs[index]["param_name"])
            parameter.point_type = DFXPointType.LEVEL_1_FOR_SHAPE_TENSOR
        if tiling_info.static_shape_flag:
            dfx_generator.set_size_of_dfx_info("shape_tensor", len(output_shape_depend) * 8 * 8)
    if not tiling_info.static_shape_flag or tiling_info.static_workspace_size >= 0:
        dfx_generator.insert_param(DFXArgInfo("workspace", DFXParamType.WORKSPACE))
    if not tiling_info.static_shape_flag:
        dfx_generator.insert_param(DFXArgInfo("tiling", DFXParamType.TILING))
    dfx_generator.generate_dfx_binary(compile_info, dfx_op_info, tiling_info)
    return dfx_op_info


def _build_tiling_info(op_info, infered_info_from_ifile, value_depend, origin_func_name):
    tiling_info = get_tiling_info_by_tiling(
        op_info,
        infered_info_from_ifile,
        value_depend,
        origin_func_name,
    )
    if infered_info_from_ifile.default_kernel_type == KernelMetaType.KERNEL_TYPE_MIX_AIC_1_1:
        tiling_info.task_ration = 1
    return tiling_info


def _build_compile_info(
    cce_file,
    kernel_name,
    origin_func_name,
    op_info,
    infered_info_from_ifile,
    tiling_key_list,
    compile_log_path,
):
    compile_info = CompileInfo()
    compile_info.src_file = cce_file
    compile_info.dst_file = os.path.join(CommonUtility.get_kernel_meta_dir(), f"{kernel_name}.o")
    compile_info.kernel_name = kernel_name
    compile_info.origin_func_name = origin_func_name
    compile_info.op_type = _op_info_get(op_info, "op_type")
    compile_info.code_channel = infered_info_from_ifile.code_channel
    compile_info.tiling_key_list = tiling_key_list
    compile_info.tiling_key_group_map = infered_info_from_ifile.tiling_key_group_map
    compile_info.compile_log_path = compile_log_path
    compile_info.hard_sync = infered_info_from_ifile.hard_sync
    compile_info.enable_deterministic = infered_info_from_ifile.enable_deterministic
    compile_info.tiling_key_deterministic = infered_info_from_ifile.tiling_key_deterministic
    compile_info.tiling_key_kernel_type = infered_info_from_ifile.tiling_key_kernel_type
    compile_info.raw_tiling_key_kernel_type = copy.deepcopy(compile_info.tiling_key_kernel_type)
    compile_info.no_set_kernel_type = infered_info_from_ifile.no_set_kernel_type
    compile_info.default_kernel_type = infered_info_from_ifile.default_kernel_type
    compile_info.dump_info = {"dump_type": "", "dump_size": 1024}
    compile_info.template_tiling_info = infered_info_from_ifile.template_tiling_info
    compile_info.tiling_key_struct_map = infered_info_from_ifile.tiling_key_struct_map
    compile_info.register_tiling_struct = infered_info_from_ifile.register_tiling_struct
    compile_info.tpl_tiling_struct = infered_info_from_ifile.tpl_tiling_struct
    handle_sk_codegen_options(compile_info, infered_info_from_ifile)
    return compile_info


def _gen_binary_meta_sections(kernel_name):
    binsec = '".ascend.meta"'
    return (
        f"static const struct BinaryMetaVersion {kernel_name}_kernel_metainfo_version_section "
        f"__attribute__ ((used, section ({binsec}))) = {{{{B_TYPE_BIN_VERSION_INFO, sizeof(unsigned int)}}, 0x01}};\n"
        f"static const struct BinaryMetaDebug {kernel_name}_kernel_metainfo_debug_section "
        f"__attribute__ ((used, section ({binsec}))) = {{{{B_TYPE_DEBUG_INFO, 8}}, 0, 0}};\n"
        f"static const struct BinaryMetaDynamicParam {kernel_name}_kernel_metainfo_dynamicparam_section "
        f"__attribute__ ((used, section ({binsec}))) = {{{{B_TYPE_DYNAMIC_PARAM, 4}}, 0, 0}};\n"
        f"static const struct BinaryMetaOptionalParam {kernel_name}_kernel_metainfo_optionalparam_section "
        f"__attribute__ ((used, section ({binsec}))) = {{{{B_TYPE_OPTIONAL_PARAM, 4}}, 1, 1}};\n"
    )


def _gen_meta_sections(kernel_name, packed, tiling_info: TilingInfo, compile_info: CompileInfo):
    """Per-key ``.ascend.meta`` sections, using AscendC's ktype and DFX generators."""
    out = []
    dfx_generator = DFXSectionGenerator()
    dfx_generator.gen_dfx_struct_flag = False
    old_sub_core_type = compile_info.sub_core_type
    tiling_key = str(packed)
    kernel_type = compile_info.tiling_key_kernel_type[tiling_key]
    for core_name in _compile_core_names(kernel_type):
        core_type, _channel, _short = _CORE_COMPILE_TARGETS[core_name]
        section_kernel = _compile_target_name(kernel_name, tiling_key, core_name, kernel_type)
        out.append(
            get_ktype_section_variable(
                f"{section_kernel}_section",
                section_kernel,
                kernel_type,
            )
        )
        compile_info.sub_core_type = core_type
        out.append(dfx_generator.generate_dfx_section(str(packed), tiling_info, section_kernel, compile_info))
    compile_info.sub_core_type = old_sub_core_type
    out.append(_gen_binary_meta_sections(kernel_name))
    return "".join(out)


def _gen_key_src(kernel_cpp_path, origin_func, impl_name, entry_params, kernel_name, packed, tiling_info, compile_info):
    """The per-key src ``.cpp``: ``kernel_operator.h`` + the codegen'd ``kernel.cpp`` + a concrete
    ``__global__`` entry forwarding to ``<impl>`` (workspace offset like AscendC's gen_kernel_fun) + the
    per-key meta sections. Compiled once with ``-DTILING_KEY_VAR=<packed>``."""
    sig, names = _signature_parts(entry_params)
    _ws_idx = len(names) - 2 if len(names) >= 2 else None
    inner = list(names)
    kernel_type = compile_info.tiling_key_kernel_type[str(packed)]
    return (
        '#include "kernel_operator.h"\n'
        f'#include "{kernel_cpp_path}"\n'
        f"__aicore__ inline __attribute__((always_inline)) void ascendc_auto_gen_{origin_func}_kernel({sig})\n"
        "{\n"
        f"    {impl_name}({', '.join(inner)});\n"
        "}\n"
        f'extern "C" __global__ AICORE void auto_gen_{origin_func}_kernel({sig})\n'
        "{\n"
        f"    KERNEL_TASK_TYPE_DEFAULT({kernel_type.name});\n"
        f"    ascendc_auto_gen_{origin_func}_kernel({', '.join(names)});\n"
        "}\n" + _gen_meta_sections(kernel_name, packed, tiling_info, compile_info)
    )


def _decode_tiling_key(schema, packed: int):
    if schema is None:
        return None
    concrete = {}
    for field in schema._fields:
        value_index = (packed >> field.offset) & ((1 << field.bits) - 1)
        if value_index >= len(field.values):
            raise OutOfRange(
                f"tilingkey field '{field.name}' index {value_index} is out of range for values {list(field.values)}"
            )
        concrete[field.name] = field.values[value_index]
    return concrete


def _filter_tiling_keys(tiling_key_list, extend_options, ctx, kernel_name):
    if "customized_tiling_key_list" in extend_options:
        context_tiling_key = extend_options.get("customized_tiling_key_list")
    else:
        context_tiling_key = ctx.get_addition("tiling_key") if ctx is not None else None
    if not context_tiling_key:
        return list(tiling_key_list)
    wanted = {str(int(k)) for k in context_tiling_key}
    filtered = [tiling_key for tiling_key in tiling_key_list if str(int(tiling_key)) in wanted]
    if not filtered:
        raise InvalidArgument(f"None of the given tiling keys are supported by {kernel_name}: {context_tiling_key}")
    return filtered


def pypto_compile_op(
    cce_file,
    origin_func_name,
    op_info,
    compile_options=None,
    code_channel=-1,
    op_compile_option="{}",
    extend_options=None,
    arch="a5",
):
    """PyPTO leaf replacing ``asc_op_compiler.compile_op``. Signature-compatible; ``cce_file`` is the PyPTO
    DSL ``.py``. Writes the flat ``kernel_meta`` artifacts + ``<kernel>.o``/``.json`` and records the json
    path into the op_context for the reused post-compile step. Called inside the reused ``build_config`` +
    ``op_context`` of ``SingleOpCompile``.
    """
    from pypto_pro.runtime.jit import (
        _codegen,
        _setup_arch_env,
        _validate_datatype_key,
    )

    extend_options = extend_options or {}
    kernel_name = _op_info_get(op_info, "kernel_name")
    arch = _setup_arch_env(arch)
    opt = _setup_options(op_info, compile_options, op_compile_option, extend_options)

    kernel = _load_kernel(cce_file)
    if origin_func_name != kernel.__name__:
        raise InvalidVal(
            f"origin_func_name '{origin_func_name}' does not match the sole @pl.jit kernel "
            f"'{kernel.__name__}' in {cce_file}"
        )
    schema = getattr(kernel, "_tilingkey_schema", None)
    if schema is None:
        raise InvalidVal(f"binary delivery requires a tiling_key schema on kernel '{kernel.__name__}'")
    datatype_schema = getattr(kernel, "_datatype_schema", None)
    dtype_key = _extract_datatype_key_from_compile_options(compile_options, datatype_schema)
    dtype_consts = _validate_datatype_key(datatype_schema, dtype_key)

    ctx = op_context.get_context()

    kernel_meta_dir = CommonUtility.get_kernel_meta_dir()
    distinct_tag = CommonUtility.get_distinct_filename_tag()
    compile_log_path = None
    if global_var_storage.get_variable("ascendc_compile_debug_config"):
        compile_log_path = os.path.join(kernel_meta_dir, kernel_name + distinct_tag + ".log")

    infer_cpp_path, kernel_type = _prepare_infer_cpp(
        kernel,
        schema,
        dtype_consts,
        arch,
        cce_file,
        kernel_name,
        kernel_meta_dir,
    )

    infer_i = os.path.join(kernel_meta_dir, kernel_name + ".i")
    logger.info("pypto_compile_op: infer cpp=%s -> %s", infer_cpp_path, infer_i)
    infered_info = KernelInfoInfer.get_tiling_key_list_and_simple_infer_code_channel(
        op_info,
        infer_cpp_path,
        infer_i,
        opt,
        compile_log_path,
        origin_func_name,
    )
    is_const_propagation = "-DFORCE_TILING_CONST_PROPAGATION" in opt.compile_options
    global_var_storage.set_variable("ascendc_tiling_const_propagation", is_const_propagation)
    tiling_info = _build_tiling_info(
        op_info,
        infered_info,
        extend_options.get("valueDepend"),
        origin_func_name,
    )
    tiling_data_file_path = os.path.join(kernel_meta_dir, kernel_name + distinct_tag + "_tiling_data.h")
    tiling_info.save_file(tiling_data_file_path)
    global_var_storage.set_variable("ascendc_is_static_op", tiling_info.static_shape_flag)
    tiling_keys = _filter_tiling_keys(infered_info.tiling_key_list, extend_options, ctx, kernel_name)
    compile_info = _build_compile_info(
        cce_file,
        kernel_name,
        origin_func_name,
        op_info,
        infered_info,
        tiling_keys,
        compile_log_path,
    )
    logger.info(
        "pypto_compile_op: op=%s kernel_name=%s arch=%s keys=%d dtype=%s -> %s",
        _op_info_get(op_info, "op_type"),
        kernel_name,
        arch,
        len(tiling_keys),
        dtype_key or "none",
        kernel_meta_dir,
    )

    op_info = _prepare_dfx(op_info, tiling_info, compile_info)
    obj_files: list[str] = []
    cmds_by_core: dict[str, list] = {core: [] for core in _CORE_COMPILE_TARGETS}
    tiling_keys_by_core: dict[str, list] = {core: [] for core in _CORE_COMPILE_TARGETS}
    compile_options_ready = False
    for tiling_key in tiling_keys:
        packed = int(tiling_key)
        concrete = _decode_tiling_key(schema, packed)

        # Codegen into a per-key subdir under this kernel's (unique) kernel_meta dir. asc_opc forks one
        # process per dtype-combo, all sharing cwd — the default ./build/<prog_name>__<arch>/tk_<packed>
        # location is keyed only on prog.name, so concurrent processes would clobber each other's
        # kernel.cpp (yielding empty/half-written copies). kernel_meta_dir is unique per kernel_name.
        cg_dir = os.path.join(kernel_meta_dir, f"_cg_tk_{packed}")
        cg = _codegen(
            kernel.to_kernel_def(concrete, dtype_consts),
            arch,
            clean_up=False,
            tilingkey_packed=packed,
            out_dir=cg_dir,
        )
        if cg is None:
            raise RuntimeFailure(f"pypto codegen failed for tilingkey {packed}")
        if _kernel_type_from_codegen(cg, arch) != kernel_type:
            raise InvalidOperation(
                "PyPTO OPC requires one kernel type for all tiling keys, but codegen "
                f"resolved a different type for tiling key {packed}"
            )
        origin_func = cg.kernel_name
        impl_name = f"{origin_func}_impl"
        if not compile_options_ready:
            workspace_idx = len(cg.entry_params) - 2 if len(cg.entry_params) >= 2 else 0
            handle_compile_options(compile_info, opt, tiling_info, workspace_idx)
            compile_options_ready = True

        # Co-locate the codegen'd kernel.cpp (+ its tiling headers) into kernel_meta under a per-key name,
        # so the per-key src #includes it by a bare relative name (no absolute path; bisheng resolves a
        # quoted include against the including file's own dir). Tiling headers are dtype/key-agnostic
        # (same content across keys) — copying per key is idempotent.
        impl_cpp_name = f"{kernel_name}_{packed}_impl.cpp"
        impl_src = os.path.join(cg.build_dir, "kernel.cpp")
        impl_dst = os.path.join(kernel_meta_dir, impl_cpp_name)
        shutil.copyfile(impl_src, impl_dst)
        for hdr in (f for f in os.listdir(cg.build_dir) if f.endswith(".h")):
            shutil.copyfile(os.path.join(cg.build_dir, hdr), os.path.join(kernel_meta_dir, hdr))

        src = _gen_key_src(
            impl_cpp_name, origin_func, impl_name, cg.entry_params, kernel_name, packed, tiling_info, compile_info
        )
        src_path = os.path.join(kernel_meta_dir, f"{kernel_name}_{packed}_kernel.cpp")
        _write(src_path, src)

        for core_name in _compile_core_names(kernel_type):
            core_type, channel, _short = _CORE_COMPILE_TARGETS[core_name]
            dst_o = os.path.join(kernel_meta_dir, f"{kernel_name}_{channel}_{packed}.o")
            sub_arch = _core_arch(core_type)
            cmd = gen_compile_cmd_v220(src_path, dst_o, opt, sub_arch, "")  # kernel.cpp self-includes its tiling.h
            cmd += [f"-D{TILING_KEY_MACRO}={packed}UL"]
            cmd += _compile_target_options(kernel_type, CommonUtility.is_c310())
            target_name = _compile_target_name(kernel_name, str(packed), core_name, kernel_type)
            cmd += [f"-Dauto_gen_{origin_func}_kernel={target_name}"]
            cmd += [f"-D{impl_name}={impl_name}_{packed}"]
            cmds_by_core[core_name].append(cmd)
            tiling_keys_by_core[core_name].append(packed)
            obj_files.append(dst_o)

    # Per-key compile via the real backend: compile_multi_tilingkey writes <op>_tmp_<core>_<pid>.mk (one
    # target per tilingkey, +a -E precompile line under dump_cce) and runs `make -j`, exactly as
    # compile_op's compile_kernel_and_meta does. One .mk per core (aic/aiv), mirroring the golden layout.
    for core_name, (_core_type, _channel, short) in _CORE_COMPILE_TARGETS.items():
        if cmds_by_core[core_name]:
            compile_multi_tilingkey(
                tiling_keys_by_core[core_name],
                cmds_by_core[core_name],
                f"{kernel_name}_tmp_{short}",
                compile_log_path,
            )
    # compile_multi_tilingkey swallows make failures unless build-log is enabled; verify every obj landed.
    missing = [o for o in obj_files if not os.path.exists(o)]
    if missing:
        raise RuntimeFailure(f"pypto compile failed (make); missing {len(missing)} objs e.g. {missing[:3]}")

    dst_o = os.path.join(kernel_meta_dir, f"{kernel_name}.o")
    fatbin_objs(obj_files, dst_o, compile_info.is_debug, compile_log_path)
    link_relocatable(dst_o)

    json_path = os.path.join(kernel_meta_dir, f"{kernel_name}.json")
    _generate_final_json(compile_info, tiling_info)
    _json_post_process(
        compile_info,
        op_info,
        tiling_info,
        check_if_gen_placehoder(op_info, True),
        check_if_gen_placehoder(op_info, False),
        compile_log_path,
    )
    ctx.add_build_res("json_file_path", os.path.abspath(json_path))
    logger.info("pypto_compile_op done: %s (%d tilingkeys)", json_path, len(tiling_keys))


def _write(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
